"""Source scores turned into contrast rows, one per image, severity, arm, summary and method.

This is where the experiment's shape lives: which arms exist, which signals each produces, and
which fold's line each residual is allowed to see. Everything numeric it does is a call into
`contrast_scores` or `contrast_diagnostics`; everything structural it does is here and nowhere
else, so a change to the arm table changes one file.
"""
from __future__ import annotations

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
    rows: list[dict] = []
    fits: dict[tuple[str, str, str], dict] = {}
    for arm in ARMS:
        for label, signal, scope in _signal_plan(arm):
            for aggregation in AGGREGATIONS:
                key = (label, signal, aggregation)
                if key in fits:
                    raise ContrastAnalysisError(
                        f"two arms produce the same contrast series {key}; the arm table must "
                        "give every arm its own name and every bucket pair one confidence twin"
                    )
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
    diagnostics: list[dict] = []
    for arm in ARMS:
        for label, signal, scope in _signal_plan(arm):
            for aggregation in AGGREGATIONS:
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
