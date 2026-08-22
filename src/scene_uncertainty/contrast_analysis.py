"""Source scores turned into contrast rows, and those rows turned into candidate summaries.

A row is one image, severity, arm, summary and method; a candidate is a row family with the
scene coordinates taken out, carrying one locked orientation and five per-severity AUROCs.

This is where the experiment's shape lives: which arms exist, which signals each produces,
which fold's line each residual is allowed to see, and what makes two rows the same candidate.
Everything numeric it does is a call into `contrast_scores`, `contrast_diagnostics` or
`corruption_metrics`; everything structural it does is here and nowhere else, so a change to
the arm table changes one file.

It stops short of deciding anything. There is no `deployable` flag and no ranking here,
because the spec's eligibility list includes "a computable confidence-only twin" -- a fact
about a *pair* of candidates that nothing in one candidate's own rows can see. This module
publishes what each candidate measured; the module that gates and orders them reads these
dictionaries and the `DEPLOYABLE_SIGNAL` constant below.
"""
from __future__ import annotations

import numpy as np

from .contrast_diagnostics import between_image_spread, clean_relationship, within_image_drift
from .contrast_inputs import (
    AGGREGATIONS,
    ARMS,
    COMBINED_SCOPE,
    CONFIDENCE_SCOPE,
    CONFIDENCE_SIGNAL,
    EXPECTED_SEVERITIES,
    PERSISTENCE_SIGNAL,
    Arm,
    ContrastInputs,
)
from .contrast_scores import FOLD_COUNT, SCORE_METHODS, assign_folds, contrast_score, robust_line
from .corruption_metrics import (
    choose_orientation,
    complete_trend_metrics,
    oriented_curve_metrics,
    severity_aurocs,
)

RESIDUAL_METHOD = "clean_residual"
RELATIVE_GAP_METHOD = "relative_gap"

RELATIVE_GAP_UNAVAILABLE = (
    "the combined scope is a signed z-score, and the symmetric relative gap's scale invariance "
    "and +/-2 bounds both require non-negative inputs"
)
"""Why arm 4 declares three methods where the other three arms declare four.

Spelled once, here, because it is written into every `combined` fit and a reader comparing two
arms' method lists needs the two strings to be the same string. `contrast_scores.relative_gap`
*refuses* a negative input rather than accommodating it, so this is not a defensive skip around
a value that would come out odd -- it is the arm table agreeing with the maths in advance.
"""


class ContrastAnalysisError(ValueError):
    """A source that loaded cleanly but cannot be turned into the declared candidates.

    A `ValueError` so `pipeline.command_analyze_within_image_contrast` turns it into one line on
    stderr through the same `except ValueError` its siblings use, matching `ContrastInputError`.
    """


CONTRAST_ROW_KEY = ("image_id", "severity", "arm", "signal", "aggregation", "method")
"""What makes two contrast rows measurements of different things.

`arm` carries an arm name on a persistence row and a *pair* name on a confidence row, and that
asymmetry is deliberate rather than sloppy: confidence has no decoder layer, so the `layer_2`
and `combined` differential arms have one twin between them. Keying the twin by arm would
produce two identical copies whose separately-locked orientations could disagree by nothing but
which copy a grouping saw first.

`score_scope` is absent because `arm` already determines it -- the two differential arms differ
in name precisely so that they can differ in scope -- and a redundant key field is a second
place for two rows to be told apart, which is one more than there should be.
"""

CONTRAST_ROW_FIELDS = (
    *CONTRAST_ROW_KEY,
    "arm_family", "declared_before_data", "score_scope",
    "reference_bin", "responsive_bin", "reference", "responsive", "score",
    "fold", "fit_slope", "fit_offset",
)
"""Every column a contrast row carries, key first, in the order a CSV should publish them.

The three provenance fields sit immediately after the key rather than at the end, because
`arm_family` and `declared_before_data` are what stop a tuning macro AUROC from a differential
arm being read as performance, and a reader who has to scroll twelve columns to find out which
kind of arm a number came from will not.

`reference` and `responsive` are published beside `score` even though `score` is derived from
them. A residual is `responsive - (offset + slope * reference)`, and without both inputs on the
row nobody reading the file can recompute it or see which half moved.
"""


def _signal_plan(arm: Arm) -> tuple[tuple[str, str, str], ...]:
    """The `(row arm label, signal, source scope)` entries one arm contributes.

    Two for most arms: its persistence rows under its own name and scope, and its confidence
    twin under the pair name. The `combined` differential arm contributes only the persistence
    entry, because its twin is already produced by the `layer_2` arm that shares its pair.

    The twin is labelled `arm.pair_name` and not `arm.name`, and the distinction is invisible in
    the current table -- every arm that is first with its pair happens to have `name ==
    pair_name`. It stops being invisible the moment the arm table is reordered, which is why
    `test_the_confidence_twin_is_labelled_by_pair_and_not_by_whichever_arm_came_first` reorders
    it. Labelling by `arm.name` would make the twin's identity a function of the tuple order in
    `ARMS`, so moving the `combined` arm above the `layer_2` one would silently rename a series
    nine tasks read.
    """
    entries = [(arm.name, PERSISTENCE_SIGNAL, arm.score_scope)]
    first_with_pair = next(other for other in ARMS if other.pair_name == arm.pair_name)
    if first_with_pair.name == arm.name:
        entries.append((arm.pair_name, CONFIDENCE_SIGNAL, CONFIDENCE_SCOPE))
    return tuple(entries)


def _unique(
    seen: set[tuple[str, str, str]], key: tuple[str, str, str]
) -> tuple[str, str, str]:
    """Record one `(arm-or-pair, signal, aggregation)` series, refusing a second of the same.

    Called by both public functions rather than only by the one that keys a dictionary. A
    repeated key in `build_contrast_rows` overwrites a fit and doubles its rows; in
    `build_anchor_diagnostics` it appends a duplicate to a list, which is quieter still --
    nothing there is keyed, so a Task 7 or 8 caller that reads the diagnostics on their own
    would get 24 entries where the 21 are documented and no signal that two of them describe
    experiments filed under one heading.

    Refused rather than deduplicated: two arms with one name are two different hypotheses, and
    silently keeping either one is choosing which gets reported.
    """
    if key in seen:
        raise ContrastAnalysisError(
            f"two arms produce the same contrast series {key}; the arm table must "
            "give every arm its own name and every bucket pair one confidence twin"
        )
    seen.add(key)
    return key


def _curve(
    inputs: ContrastInputs, confidence_bin: str, aggregation: str, scope: str, signal: str
) -> dict[int, dict[int, float]]:
    """`{image_id: {severity: score}}` for one source series.

    Indexes `inputs.scores` directly and lets a missing key raise `KeyError`. The loader
    refuses any bundle that does not cover every required series at every image and severity, so
    a miss here is a broken arm table or a broken loader and not a bundle the operator can fix --
    and wrapping it in a `ContrastAnalysisError` would send them to inspect their source.
    """
    return {
        image_id: {
            severity: inputs.scores[
                (image_id, severity, signal, confidence_bin, aggregation, scope)
            ]
            for severity in EXPECTED_SEVERITIES
        }
        for image_id in inputs.image_ids
    }


def _clean(curves: dict[int, dict[int, float]]) -> dict[int, float]:
    """Severity zero of every image's curve, which is the whole of what a fit may see.

    Severity zero only, because the line encodes the *normal clean relationship* between the two
    ranges. A line fitted on corrupted rows would already contain the effect the residual is
    trying to measure, and every residual would then be a deviation from corruption-adjusted
    normal -- which is zero on average by construction.
    """
    return {image_id: curve[0] for image_id, curve in curves.items()}


def _fit(
    references: dict[int, dict[int, float]],
    responsives: dict[int, dict[int, float]],
    folds: dict[int, int],
    scope: str,
) -> dict:
    """The final all-clean line, the five fold lines, and whether the residual is usable.

    The final line is what a deployment would store and is never used to score a tuning image;
    the fold lines are what every reported residual is built from. Both are kept because the
    figures show the first and the numbers come from the second, and a bundle that published
    only one of them would leave a reader unable to tell which the picture was drawn from.

    A fold's line is fitted on the images *not* in that fold. Fitting on all of them is the
    mistake this whole structure exists to prevent: an image would then help construct the line
    that predicts it, and its residual would be shrunk by exactly the amount of its own
    influence -- smallest for the images most unlike the rest, which are the ones a residual is
    supposed to flag.

    The residual is unavailable if *any* fold's line is missing, not only if the final one is.
    A reference that is constant within one fold's training set but varies overall gives a
    perfectly fittable final line and one unfittable fold, and publishing the four folds that
    worked would quietly drop a fifth of the images from a candidate whose row count nothing
    else re-checks.
    """
    clean_reference = _clean(references)
    clean_responsive = _clean(responsives)
    image_ids = sorted(clean_reference)
    final = robust_line(
        [clean_reference[image_id] for image_id in image_ids],
        [clean_responsive[image_id] for image_id in image_ids],
    )
    fold_lines: dict[int, tuple[float, float] | None] = {}
    for fold in range(FOLD_COUNT):
        held = [image_id for image_id in image_ids if folds[image_id] == fold]
        trained = [image_id for image_id in image_ids if folds[image_id] != fold]
        if not held:
            continue
        fold_lines[fold] = robust_line(
            [clean_reference[image_id] for image_id in trained],
            [clean_responsive[image_id] for image_id in trained],
        )
    unavailable = sorted(fold for fold, line in fold_lines.items() if line is None)
    if final is None:
        reason = "the clean reference is constant across images, so no line can be fitted"
    elif unavailable:
        reason = f"folds {unavailable} have a constant clean reference in their training images"
    else:
        reason = None
    relative_gap_available = scope != COMBINED_SCOPE
    return {
        "final_line": list(final) if final is not None else None,
        "fold_lines": {
            fold: list(line) if line is not None else None
            for fold, line in fold_lines.items()
        },
        "residual_available": final is not None and not unavailable,
        "unavailable_reason": reason,
        "relative_gap_available": relative_gap_available,
        "relative_gap_unavailable_reason": (
            None if relative_gap_available else RELATIVE_GAP_UNAVAILABLE
        ),
    }


def _methods(fit: dict) -> tuple[str, ...]:
    """The methods this fit can actually produce, in `SCORE_METHODS` order.

    Both exclusions are recorded on the fit rather than inferred by a reader from a shorter
    method list. A candidate missing its residual is not a candidate with a gap in it, and the
    `unavailable_reason` beside it is what a reader gets instead of a column of blanks.
    """
    return tuple(
        method for method in SCORE_METHODS
        if not (method == RESIDUAL_METHOD and not fit["residual_available"])
        and not (method == RELATIVE_GAP_METHOD and not fit["relative_gap_available"])
    )


def build_contrast_rows(inputs: ContrastInputs) -> tuple[list[dict], dict]:
    """Every contrast row the four arms declare, and the 21 fits behind their residuals.

    Returns `(rows, fits)` where `fits` is keyed `(arm-or-pair, signal, aggregation)`. Residual
    rows are omitted entirely when the fit is unavailable rather than written with a `None`
    score: a `None` in a score column reads as a measurement that came out empty, and downstream
    every ranking, orientation and AUROC would have to learn to skip it. The excluded candidate
    and its recorded reason is the honest shape.

    One `assign_folds` call for the whole run, over `inputs.image_ids`, rather than one per arm.
    The fold assignment is a property of the image roster, so an arm-local call would be five
    chances to hand one arm a different partition than its neighbour -- and `clean_relationship`
    validates the folds it is given against the roster it is scoring, which only means anything
    if the two came from the same place.
    """
    folds = assign_folds(inputs.image_ids)
    seen: set[tuple[str, str, str]] = set()
    rows: list[dict] = []
    fits: dict[tuple[str, str, str], dict] = {}
    for arm in ARMS:
        for label, signal, scope in _signal_plan(arm):
            for aggregation in AGGREGATIONS:
                key = _unique(seen, (label, signal, aggregation))
                references = _curve(inputs, arm.reference_bin, aggregation, scope, signal)
                responsives = _curve(inputs, arm.responsive_bin, aggregation, scope, signal)
                fit = _fit(references, responsives, folds, scope)
                fits[key] = fit
                for method in _methods(fit):
                    residual = method == RESIDUAL_METHOD
                    for image_id in inputs.image_ids:
                        fold = folds[image_id]
                        line = fit["fold_lines"].get(fold)
                        for severity in EXPECTED_SEVERITIES:
                            reference = references[image_id][severity]
                            responsive = responsives[image_id][severity]
                            rows.append({
                                "image_id": image_id,
                                "severity": severity,
                                "arm": label,
                                "signal": signal,
                                "aggregation": aggregation,
                                "method": method,
                                "arm_family": arm.family,
                                "declared_before_data": arm.declared_before_data,
                                "score_scope": scope,
                                "reference_bin": arm.reference_bin,
                                "responsive_bin": arm.responsive_bin,
                                "reference": reference,
                                "responsive": responsive,
                                "score": contrast_score(
                                    method, reference, responsive,
                                    line=tuple(line) if line is not None else None,
                                ),
                                "fold": fold,
                                "fit_slope": line[0] if residual else None,
                                "fit_offset": line[1] if residual else None,
                            })
    return rows, fits


def build_anchor_diagnostics(inputs: ContrastInputs) -> list[dict]:
    """Drift, spread and clean relationship for all 21 arm-summary-signal configurations.

    All four arms are measured, including the two whose reference is not an anchor. Reporting
    only the arms expected to pass would leave a reader unable to see how far a differential
    reference moves, which is the number that says the two families are different experiments
    rather than one experiment with a weak member. This module computes and labels; the module
    that decides carries the `arm_family` field these rows hand it.

    Split from `build_contrast_rows` rather than folded into it because the two answer different
    questions and Task 7 needs one without the other: the rows are the candidate scores, and
    these are the evidence about whether the reference each was built on deserved to be
    subtracted. Returning them together would make a caller that wants only the diagnostics
    build 60,000 rows to get 21 dictionaries.
    """
    folds = assign_folds(inputs.image_ids)
    seen: set[tuple[str, str, str]] = set()
    diagnostics: list[dict] = []
    for arm in ARMS:
        for label, signal, scope in _signal_plan(arm):
            for aggregation in AGGREGATIONS:
                _unique(seen, (label, signal, aggregation))
                references = _curve(inputs, arm.reference_bin, aggregation, scope, signal)
                responsives = _curve(inputs, arm.responsive_bin, aggregation, scope, signal)
                drift = within_image_drift(references)
                diagnostics.append({
                    "arm": label,
                    "arm_family": arm.family,
                    "declared_before_data": arm.declared_before_data,
                    "signal": signal,
                    "aggregation": aggregation,
                    "score_scope": scope,
                    "reference_bin": arm.reference_bin,
                    "responsive_bin": arm.responsive_bin,
                    "drift": drift,
                    "spread": between_image_spread(references, drift),
                    "relationship": clean_relationship(
                        _clean(references), _clean(responsives), folds=folds
                    ),
                })
    return diagnostics


CONTRAST_CANDIDATE_KEY = ("arm", "signal", "aggregation", "method")
"""A row key with the scene coordinates taken out.

`CONTRAST_ROW_KEY` minus `image_id` and `severity`: what is left is exactly one curve family,
and grouping by it is what turns per-scene rows into the thing a deployment would ship.

Note what stays separate: a persistence candidate and its confidence twin differ in `signal`,
so they are two candidates with two independently chosen orientations. That independence is
the whole point of the twin -- a twin forced to share its candidate's orientation would be
measuring how well the candidate's direction happens to suit confidence, not how well
confidence does on its own terms. On the shared fixture the two do disagree, which is why
`test_a_twin_locks_its_own_direction_and_is_not_handed_the_persistence_one` exists.

`score_scope` is not in the key for the same reason it is not in `CONTRAST_ROW_KEY`: `arm`
already determines it. It travels with the candidate as provenance instead.
"""

DEPLOYABLE_SIGNAL = PERSISTENCE_SIGNAL
"""Only persistence candidates are ranked. Confidence twins are summarised in full and
published in full; it is the ranking they stay out of, because a control that can win the
comparison it exists to lose is not a control.

Named here and consumed by Task 6 rather than spelled as the literal `"persistence"` at the
gate, so that the ranked signal and the signal the rows carry cannot drift apart into two
strings that happen to match today.
"""

_CANDIDATE_PROVENANCE = (
    "arm_family", "declared_before_data", "score_scope", "reference_bin", "responsive_bin",
)
"""Row columns that describe the candidate rather than the scene, carried through unchanged.

Every one of these is constant across a candidate's rows -- they are functions of the arm, and
the arm is part of the candidate key -- so the first row's copy is the candidate's copy. That
constancy is checked on every row rather than assumed, because it is the one thing about
provenance that copying the first row silently relies on. The confidence twin is the reason
it is not obvious: it is labelled by bucket *pair*, so the two differential arms that share a
pair contribute to one candidate, and an arm table that ever gave them different provenance
would publish whichever of the two `ARMS` happened to list first.

`reference_bin` and `responsive_bin` are an *ordered* pair and are copied as two named fields
rather than as a set or a joined string. `contrast_scores.raw_gap(reference, responsive)`
turns a swap into a sign flip on every contrast in the experiment, and a candidate summary that
recorded only which two bins were involved would leave a reader unable to see which way round
the surviving score was taken.
"""


def _severity_statistics(values: np.ndarray) -> dict:
    """Count, mean, variance, median and quartiles of one severity's scores across images.

    Six keys, always all six, because Task 7 draws a box per severity and a plotting caller
    that has to test for the existence of `q25` before drawing it will eventually draw a box
    without whiskers instead of a gap. A severity that no image scored returns `count=0` and
    five `None`s rather than being left out of the dictionary: "no image reached this severity"
    is a fact about the run, and an absent key is indistinguishable from a caller that forgot
    to ask.

    Both a mean and a median are published because they answer different questions about the
    same column and diverge exactly where it matters. The mean is what a cross-scene AUROC is
    sensitive to -- it moves with the outlying scenes -- while the median is what the per-image
    trend behaves like. A candidate whose per-image median falls while its across-image mean
    rises is not a contradiction to resolve, it is the shape that makes a locked orientation
    score below 0.5, and collapsing the two into one number would hide it.

    `np.var` is population variance (`ddof=0`), matching `np.mean` over the same finite set:
    these are descriptions of the images that were scored, not estimates of a wider population
    the run is a sample of. The quartiles are `np.percentile`'s default linear interpolation,
    so on four sorted values `[1, 2, 3, 4]` `q25` is 1.75 and not the lower order statistic.
    """
    if values.size == 0:
        return {"count": 0, "mean": None, "variance": None,
                "median": None, "q25": None, "q75": None}
    q25, median, q75 = (float(value) for value in np.percentile(values, [25, 50, 75]))
    return {
        "count": int(values.size), "mean": float(np.mean(values)),
        "variance": float(np.var(values)), "median": median, "q25": q25, "q75": q75,
    }


def summarize_contrast_candidates(
    rows: list[dict], *, expected_image_count: int
) -> list[dict]:
    """One dictionary per candidate: its trend, its one locked orientation, and its five AUROCs.

    Orientation is chosen from the median signed Spearman of this candidate's own tuning
    images, once, before any AUROC is computed -- so no direction can be picked because it
    scored better. An unorientable candidate gets `orientation=None` and `macro_auroc=None`
    rather than a defaulted `+1`: "these images do not agree which way this moves" is a
    finding, and a defaulted direction would publish it as a measurement of roughly 0.5.

    The median rather than the mean, and not only because `choose_orientation` says so. Two
    images at `+0.94` against three at `-0.2` average to `+0.26` and median to `-0.2`, which is
    a candidate that falls on most of its images being read as one that rises;
    `test_the_locked_direction_follows_the_median_and_not_the_mean` is that fixture. The same
    reasoning is why `median_signed_spearman` and `median_absolute_spearman` are two separate
    medians and not one median and its absolute value -- a candidate that rises on half its
    images and falls on the other half has a signed median of `0.0` and an absolute median of
    `1.0`, and those two numbers say "no agreed direction" and "moves hard on every image",
    which is a very different report from "moves not at all".

    `expected_image_count` is an argument rather than a count of the rows, for the reason the
    corruption-sensitivity command gives: the rows cannot tell a run of 240 images apart from a
    run of 250 that lost ten before scoring, and only the first of those is a smaller run.

    The two per-image curve checks are averaged over whichever images were fully measured, but
    the AUROCs wait for `complete`. The difference is that a curve check is a statement about
    one image and survives its neighbour going missing, while an AUROC ranks scores *across*
    images: computing it on a short roster would compare a clean group of one size against
    corrupted groups of another, and the macro average would then weight the five severities by
    how many images happened to survive at each -- which is precisely the size-weighting
    `severity_aurocs` refuses in its own averaging.

    `missing_count` counts images that produced rows but not a full six-point curve. An image
    that produced no rows at all cannot appear in it -- there is nothing here to count -- and
    shows up instead as `image_count` falling short of `expected_image_count`, which is also
    what withholds `complete`. The direction and measurement fractions divide by `image_count`
    for the matching reason: they describe the images this candidate actually scored, and
    dividing by `measured_count` would let a candidate that failed on half its images report
    the surviving half's agreement as unanimity.
    """
    index: dict[tuple, dict] = {}
    provenance: dict[tuple, dict] = {}
    seen: set[tuple] = set()
    for position, row in enumerate(rows):
        missing = [field for field in CONTRAST_ROW_KEY if field not in row]
        if missing:
            raise ContrastAnalysisError(f"contrast row {position} is missing {missing}")
        key = tuple(row[field] for field in CONTRAST_ROW_KEY)
        if key in seen:
            raise ContrastAnalysisError(f"duplicate contrast row key: {key}")
        seen.add(key)
        candidate_key = tuple(row[field] for field in CONTRAST_CANDIDATE_KEY)
        index.setdefault(candidate_key, {}).setdefault(row["image_id"], {})[
            row["severity"]
        ] = row["score"]
        carried = {field: row.get(field) for field in _CANDIDATE_PROVENANCE}
        established = provenance.setdefault(candidate_key, carried)
        if established != carried:
            raise ContrastAnalysisError(
                f"contrast rows disagree about the provenance of candidate {candidate_key}: "
                f"{established} then {carried}"
            )

    candidates: list[dict] = []
    for candidate_key in sorted(index, key=lambda key: tuple(str(part) for part in key)):
        by_image = index[candidate_key]
        trends = {}
        for image_id, curve in by_image.items():
            severities = sorted(curve)
            trends[image_id] = complete_trend_metrics(
                severities, [curve[severity] for severity in severities]
            )
        signed = [
            trend["signed_spearman"] for trend in trends.values()
            if trend["signed_spearman"] is not None
        ]
        absolute = [
            trend["absolute_spearman"] for trend in trends.values()
            if trend["absolute_spearman"] is not None
        ]
        directions = [trend["direction"] for trend in trends.values()]
        orientation = choose_orientation(signed)
        measured = sum(1 for trend in trends.values() if trend["fully_measured"])
        complete = measured == len(by_image) == expected_image_count

        oriented = [
            oriented_curve_metrics(
                [by_image[image_id][severity] for severity in sorted(by_image[image_id])],
                orientation,
            )
            for image_id in by_image
            if orientation in (-1, 1) and trends[image_id]["fully_measured"]
        ]

        auroc_by_severity = macro = None
        if orientation in (-1, 1) and complete:
            scores_by_severity = {
                severity: [by_image[image_id][severity] for image_id in sorted(by_image)]
                for severity in EXPECTED_SEVERITIES
            }
            auroc_by_severity, macro = severity_aurocs(scores_by_severity, orientation)

        candidate = dict(zip(CONTRAST_CANDIDATE_KEY, candidate_key))
        candidate.update(provenance[candidate_key])
        candidate.update({
            "expected_image_count": expected_image_count,
            "image_count": len(by_image),
            "measured_count": measured,
            "missing_count": len(by_image) - measured,
            "positive_count": directions.count("increasing"),
            "negative_count": directions.count("decreasing"),
            "flat_count": directions.count("flat"),
            "median_signed_spearman": float(np.median(signed)) if signed else None,
            "median_absolute_spearman": float(np.median(absolute)) if absolute else None,
            "orientation": orientation,
            "orientable": orientation in (-1, 1),
            "complete": complete,
            "oriented_adjacent_consistency": (
                float(np.mean([item["adjacent_consistency"] for item in oriented]))
                if oriented else None
            ),
            "max_blur_above_clean_rate": (
                float(np.mean([item["max_blur_above_clean"] for item in oriented]))
                if oriented else None
            ),
            "auroc_by_severity": auroc_by_severity,
            "macro_auroc": macro,
            "severity_statistics": {
                severity: _severity_statistics(
                    np.array(
                        [
                            by_image[image_id][severity]
                            for image_id in by_image
                            if severity in by_image[image_id]
                        ],
                        dtype=float,
                    )
                )
                for severity in EXPECTED_SEVERITIES
            },
        })
        total = len(by_image) or 1
        candidate["measured_fraction"] = measured / total
        candidate["missing_fraction"] = candidate["missing_count"] / total
        candidate["positive_fraction"] = candidate["positive_count"] / total
        candidate["negative_fraction"] = candidate["negative_count"] / total
        candidate["flat_fraction"] = candidate["flat_count"] / total
        candidate["dominant_direction_fraction"] = max(
            candidate["positive_fraction"],
            candidate["negative_fraction"],
            candidate["flat_fraction"],
        )
        candidates.append(candidate)
    return candidates
