"""The three things a contrast has to beat, and the order the survivors are read in.

A contrast is only interesting relative to something. Three somethings, and each rules out a
different way of being fooled:

* the **raw responsive** range alone -- does combining two ranges beat using one?
* the **raw reference** range alone -- a differential arm subtracts two responsive ranges, so a
  gap that beats the 50--60 percent range while losing to the 90--100 percent range has found
  nothing except which of its two inputs was stronger;
* the **confidence-only twin** -- the same contrast built from detector confidence, which is
  free at deployment where persistence is not.

The completed deployment analysis is why the third exists: at the 90--100 percent decile,
persistence and confidence showed equal-magnitude, opposite-sign trends, which is the shape a
restatement of confidence takes.

Nothing here re-derives a score, an orientation or an AUROC. Every number this module publishes
is either a difference between two `summarize_contrast_candidates` outputs, a resampling of the
rows those outputs were built from, or a sort. That is deliberate: orientation is chosen in
exactly one place, and a second implementation of it here would be a second place for it to be
chosen and therefore a place for the two to drift.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from .contrast_analysis import CONTRAST_CANDIDATE_KEY, DEPLOYABLE_SIGNAL
from .contrast_inputs import (
    AGGREGATIONS,
    ARMS,
    CONFIDENCE_SIGNAL,
    FULL_TUNING_IMAGE_COUNT,
)
from .contrast_scores import SCORE_METHODS
from .corruption_metrics import EXPECTED_SEVERITIES

BOOTSTRAP_SEED = 20260821
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_PERCENTILES = (2.5, 97.5)
"""The spec's three bootstrap constants, named so the report and the run cannot disagree.

The seed is fixed rather than configurable because a deployment constant that moves between two
runs of the same command is not a measurement of anything. `BOOTSTRAP_SAMPLES` is read at *call*
time rather than baked into a default argument, so an integration test can patch this module
constant down and have every bootstrap in the run honour it; see `paired_macro_bootstrap`.
"""

SUPPORTED_VERDICT = "supported on tuning"
INCONCLUSIVE_VERDICT = "inconclusive on tuning"
UNSUPPORTED_VERDICT = "not supported on tuning"
"""The spec's three labels, verbatim, and every one of them says "on tuning".

Named here so the easy report quotes the same strings the bootstrap wrote rather than its own
copies. The wording is load-bearing: the arms, the summaries and the methods were all selected
using this same tuning programme, so neither `supported` nor `inconclusive` is a held-out claim
and a label that dropped the qualifier would read as one.
"""

REFERENCE_CONTROL_METHOD = "raw_reference"
"""The method label the reference control's rows carry.

Deliberately *not* one of `contrast_scores.SCORE_METHODS`. The spec calls the reference range "a
reported control, not a ranked candidate: it adds no rows to the candidate list and does not
change the count of four declared score methods", and a label collision with a real method is
exactly how it would acquire both.

Being outside that set is also the *whole* of how `rank_contrast_candidates` excludes it. There
is no second clause naming this constant at the gate, because the spec's eligibility list already
says "one of the four declared score methods" and a control that is not one of them is refused by
that line. Two clauses saying the same thing would leave a reader unsure which was load-bearing.
"""

RESPONSIVE_CONTROL_METHOD = "raw_responsive"
"""The declared score method that doubles as the matched responsive control.

It is a real candidate -- it is ranked, plotted and published like the other three -- and it is
also what every contrast on the same arm and aggregation is measured against. So a
`raw_responsive` candidate is its own responsive control, its macro difference is exactly zero
and its bootstrap interval is exactly zero wide. That is the correct answer to "does using one
range beat using one range?" rather than a degenerate case to special-case away.
"""

CORRUPTED_SEVERITIES = EXPECTED_SEVERITIES[1:]
"""Severities 1 through 5. Severity 0 is the clean group of all five comparisons, never a
comparison of its own, and slicing the declared ladder is what keeps that true if the ladder
ever changes length.
"""

ARM_NAMES = frozenset(arm.name for arm in ARMS)
"""The four declared arm names, for the eligibility gate's closed-set check.

Derived from the arm table rather than listed, so adding a fifth arm cannot leave the gate
refusing it.
"""

_PAIR_BY_ARM = {arm.name: arm.pair_name for arm in ARMS}
"""Which confidence twin each arm resolves to.

Built from `Arm.pair_name` rather than by stripping a suffix off `Arm.name`. Confidence has no
decoder layer, so the `layer_2` and `combined` differential arms have one twin between them, and
that identity is something the arm table states rather than something a naming convention
implies.
"""


def reference_control_rows(rows: list[dict]) -> list[dict]:
    """The reference range as if it were a candidate, so it gets its own orientation and AUROCs.

    Built only from the `raw_responsive` rows, because all four methods of one
    `(arm, signal, aggregation)` carry the same `reference` value and taking them all would
    produce four identical copies whose row keys collide -- which
    `summarize_contrast_candidates` refuses outright, so the mistake is loud rather than a
    quadrupled control. One method's rows are a complete cover of the 21 series precisely
    because `raw_responsive` is the one method no fit can fail to produce: the residual needs a
    fittable clean line and the symmetric relative gap needs a non-negative scope, and either
    exclusion would drop an arm's control while leaving its candidates in place.

    New dictionaries rather than mutated ones. The caller still holds `rows` and goes on to
    build the real candidates out of them, so rewriting `method` and `score` in place would
    turn every `raw_responsive` candidate in the run into a second copy of its own control.

    The `reference` and `responsive` columns are carried through untouched, so a control row
    still says which two ranges it came from and a reader can see that its score is the first of
    them. `fit_slope` and `fit_offset` come along as the `None`s the `raw_responsive` rows carry;
    no line was fitted for this control and none is claimed.
    """
    return [
        {**row, "method": REFERENCE_CONTROL_METHOD, "score": row["reference"]}
        for row in rows
        if row["method"] == RESPONSIVE_CONTROL_METHOD
    ]


def index_curves(rows: list[dict]) -> dict[tuple, dict[int, dict[int, float]]]:
    """`{candidate key: {image_id: {severity: score}}}`, built in one pass over the rows.

    One pass, not one pass per candidate. The real table is 126,000 rows and every candidate
    needs two lookups for its bootstraps, so a per-candidate scan is twelve million row
    comparisons for a dictionary that could have been built once.

    Keyed by `CONTRAST_CANDIDATE_KEY` rather than by a tuple spelled out here, so the curves and
    the candidate summaries they are looked up beside cannot come to disagree about what makes
    two rows the same candidate.
    """
    curves: dict[tuple, dict[int, dict[int, float]]] = {}
    for row in rows:
        key = tuple(row[field] for field in CONTRAST_CANDIDATE_KEY)
        curves.setdefault(key, {}).setdefault(row["image_id"], {})[
            row["severity"]
        ] = row["score"]
    return curves


def _macro_from_draws(
    clean: np.ndarray, by_severity: dict[int, np.ndarray], draws: np.ndarray, orientation: int
) -> np.ndarray:
    """Macro AUROC for every bootstrap draw at once.

    Ranked in batch with `rankdata(..., axis=1)` rather than one draw at a time. The paired
    bootstrap needs 2,000 draws for each of 45 candidates against each of three counterparts, and
    a per-draw loop turns a minute-long report into a quarter-hour one. `method="average"` is the
    same tie handling `corruption_metrics.binary_auroc` uses, so a bootstrap median and the
    point estimate cannot disagree about what a tie is worth -- and a resampled image list draws
    duplicates by construction, so ties are the common case here rather than the exotic one.

    The closed form is `binary_auroc`'s: the Mann-Whitney U statistic over the clean and
    corrupted groups, normalised by the product of their sizes. A bootstrap draw takes the same
    image list for both groups, so both sizes are the roster size and the denominator is its
    square.

    `draws` is an index matrix rather than a list of image IDs, so one identity row recovers the
    unresampled point estimate through this same code -- which is what stops the point estimate
    and the interval around it from being computed two different ways.
    """
    count = clean.size
    sign = float(orientation)
    clean_draws = sign * clean[draws]
    totals = np.zeros(draws.shape[0], dtype=float)
    for severity in CORRUPTED_SEVERITIES:
        corrupted_draws = sign * by_severity[severity][draws]
        joint = np.concatenate([clean_draws, corrupted_draws], axis=1)
        ranks = rankdata(joint, method="average", axis=1)
        corrupted_rank_sum = ranks[:, count:].sum(axis=1)
        totals += (corrupted_rank_sum - count * (count + 1) / 2) / (count * count)
    return totals / len(CORRUPTED_SEVERITIES)


def _severity_arrays(
    scores: dict[int, dict[int, float]], ordered: list[int]
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """One curve dictionary as `(clean column, {severity: corrupted column})`, image-aligned.

    Both halves are indexed by position in `ordered`, which is the same list for the candidate
    and for the control -- that alignment is the whole of what makes the bootstrap *paired*, and
    reading each method's own sorted keys instead would silently pair image 7 of one with image
    9 of the other on any roster where the two differ.
    """
    clean = np.array([scores[image_id][0] for image_id in ordered], dtype=float)
    corrupted = {
        severity: np.array([scores[image_id][severity] for image_id in ordered], dtype=float)
        for severity in CORRUPTED_SEVERITIES
    }
    return clean, corrupted


def paired_macro_bootstrap(
    image_ids,
    candidate_scores: dict[int, dict[int, float]],
    control_scores: dict[int, dict[int, float]],
    *,
    candidate_orientation: int,
    control_orientation: int,
    seed: int = BOOTSTRAP_SEED,
    samples: int | None = None,
) -> dict:
    """How far apart two methods' macro AUROCs are, and how much of that survives resampling.

    One sampled list of image IDs is applied to *both* methods and all six severities. Drawing
    separately would compare one method on one set of scenes against the other on a different
    set, and most of the resulting spread would be scene sampling rather than the difference
    between the methods. A method compared with itself therefore returns an interval of exactly
    zero width, which is the property that says the pairing is real.

    Each method is read in its *own* locked orientation. The control's direction was chosen from
    the control's own images and has no reason to match the candidate's -- a persistence gap that
    rises with blur is regularly twinned with a confidence gap that falls -- and handing the
    candidate's orientation to both would score the control on how well the candidate's direction
    happens to suit it. Applying an orientation is an exact mirror, so getting this wrong does
    not add noise: it reports `1 - auroc` for the control and shifts every difference by a fixed
    amount.

    `samples` resolves against the module constant at call time rather than in the signature's
    default, so an integration test that patches `BOOTSTRAP_SAMPLES` down changes every bootstrap
    in the run rather than none of them. `seed` is a plain default because nothing patches it:
    it is the spec's fixed number, and a run whose seed moved would not be reproducible whatever
    it was patched to.

    The verdict wording is the spec's. Neither label is a held-out claim, because the arms and
    methods were selected using this same tuning programme.
    """
    for label, orientation in (
        ("candidate", candidate_orientation), ("control", control_orientation)
    ):
        if orientation not in (-1, 1):
            raise ValueError(
                f"the {label} orientation must be -1 or 1, not {orientation!r}; an unorientable "
                "candidate has no direction to bootstrap and must be excluded instead"
            )
    samples = BOOTSTRAP_SAMPLES if samples is None else samples
    ordered = sorted(image_ids)
    count = len(ordered)
    candidate_clean, candidate_by_severity = _severity_arrays(candidate_scores, ordered)
    control_clean, control_by_severity = _severity_arrays(control_scores, ordered)

    generator = np.random.default_rng(seed)
    draws = generator.integers(0, count, size=(samples, count))
    differences = _macro_from_draws(
        candidate_clean, candidate_by_severity, draws, candidate_orientation
    ) - _macro_from_draws(control_clean, control_by_severity, draws, control_orientation)

    identity = np.arange(count).reshape(1, count)
    point = float(
        _macro_from_draws(
            candidate_clean, candidate_by_severity, identity, candidate_orientation
        )[0]
        - _macro_from_draws(
            control_clean, control_by_severity, identity, control_orientation
        )[0]
    )
    low, high = (
        float(value) for value in np.percentile(differences, list(BOOTSTRAP_PERCENTILES))
    )
    if point > 0 and low > 0:
        verdict = SUPPORTED_VERDICT
    elif point > 0:
        verdict = INCONCLUSIVE_VERDICT
    else:
        verdict = UNSUPPORTED_VERDICT
    return {
        "macro_difference": point, "low": low, "high": high,
        "verdict": verdict, "seed": seed, "samples": samples,
    }


def beats_both_inputs(candidate: dict) -> bool:
    """True only when the contrast is ahead of the responsive *and* the reference range.

    Both strictly. A contrast level with one of its inputs has not improved on it, and a
    differential arm level with its stronger input has only rediscovered which input that was.
    An absent difference is not a pass either: a comparison that could not be computed has not
    been won.
    """
    responsive = candidate.get("responsive_control_macro_difference")
    reference = candidate.get("reference_control_macro_difference")
    return bool(
        responsive is not None and reference is not None
        and responsive > 0 and reference > 0
    )


def is_confidence_redundant(candidate: dict) -> bool:
    """True unless the candidate's macro AUROC is strictly ahead of its twin's.

    Strictly, and default-true. A candidate level with its confidence twin has not been shown
    to add anything, and a candidate or twin that produced no AUROC at all has been shown even
    less -- so an absent measurement reads as redundant rather than as a win it never earned.
    A `>=` here would let an exact tie be published as beating confidence.

    The comparison is between the two raw macro AUROCs and nothing else, so a candidate whose
    twin ranks *below* chance is not redundant even when the candidate is barely above it. That
    is the spec's rule read literally, and it is the right one: "worth more than the detector's
    own confidence" is a comparison between two signals, not a claim that either is any good.
    Whether the candidate is worth deploying at all is what the macro AUROC beside this flag is
    for.
    """
    own = candidate.get("macro_auroc")
    twin = candidate.get("twin_macro_auroc")
    return not (own is not None and twin is not None and own > twin)


def _by_key(items: list[dict]) -> dict[tuple, dict]:
    return {
        tuple(item[field] for field in CONTRAST_CANDIDATE_KEY): item for item in items
    }


def attach_controls(
    candidates: list[dict],
    controls: list[dict],
    rows: list[dict],
    *,
    samples: int | None = None,
) -> None:
    """Write every control comparison onto the candidate dictionaries, in place.

    In place rather than returning new dictionaries, so a reader that has a candidate has all
    of it: the ranking, the plots, the CSV and the report each hold the same object, and there
    is no second copy that could quote a different macro AUROC for one candidate.

    Only persistence candidates are given controls. A confidence twin is what the comparison is
    *against*; giving it a twin of its own would be comparing confidence with itself, and giving
    it a `confidence_redundant` flag would invite a reader to ask whether the control passed the
    test it exists to be the baseline of.

    Confidence twins are looked up by *pair*, so the `layer_2` and `combined` differential arms
    share one. Their persistence numbers differ; their twin does not, and that is correct --
    there is only one confidence measurement of that bucket pair to be redundant with.

    Three bootstraps per candidate: the responsive control, the reference control and the twin.
    The spec's anchored success criterion asks for a contrast that beats *both* of its inputs and
    whose macro improvement survives resampling, and with an interval on only one of the two that
    criterion is not checkable as written. It is also the reference comparison that most needs
    one, because on a differential arm the reference is a second responsive range rather than an
    anchor -- which is exactly where a bare point estimate is least trustworthy.

    The reference range's curves are not in `rows`, so they are re-derived here from
    `reference_control_rows`. Re-derived rather than taken as a fourth argument, and the claim
    that buys is narrow: the control *curves* and the candidate *rows* are then two views of one
    list, and a caller cannot hand over curves describing rows it did not also pass. It says
    nothing about `controls`, which is still the caller's own summary list and could in principle
    have been built from something else -- that is the caller's to get right, the same way
    `candidates` is. The cost is 14 ms of re-indexing against a 50-second call.
    """
    indexed = _by_key(candidates)
    curves = index_curves(rows)
    reference_curves = index_curves(reference_control_rows(rows))
    control_index = {
        (item["arm"], item["signal"], item["aggregation"]): item for item in controls
    }
    for candidate in candidates:
        if candidate["signal"] != DEPLOYABLE_SIGNAL:
            continue
        arm, aggregation, method = (
            candidate["arm"], candidate["aggregation"], candidate["method"]
        )
        responsive = indexed.get(
            (arm, DEPLOYABLE_SIGNAL, aggregation, RESPONSIVE_CONTROL_METHOD)
        )
        reference = control_index.get((arm, DEPLOYABLE_SIGNAL, aggregation))
        twin_arm = _PAIR_BY_ARM[arm]
        twin = indexed.get((twin_arm, CONFIDENCE_SIGNAL, aggregation, method))

        candidate["twin_arm"] = twin_arm
        candidate["twin_macro_auroc"] = twin["macro_auroc"] if twin else None
        twin_severities = twin["auroc_by_severity"] if twin else None
        candidate["twin_auroc_by_severity"] = (
            dict(twin_severities) if twin_severities is not None else None
        )
        candidate["twin_orientation"] = twin["orientation"] if twin else None
        candidate["responsive_control_macro_auroc"] = (
            responsive["macro_auroc"] if responsive else None
        )
        candidate["reference_control_macro_auroc"] = (
            reference["macro_auroc"] if reference else None
        )

        for label, other in (
            ("responsive_control", responsive),
            ("reference_control", reference),
            ("twin", twin),
        ):
            if candidate["macro_auroc"] is None or other is None or other["macro_auroc"] is None:
                candidate[f"{label}_macro_difference"] = None
                candidate[f"{label}_auroc_difference_by_severity"] = None
                continue
            candidate[f"{label}_macro_difference"] = (
                candidate["macro_auroc"] - other["macro_auroc"]
            )
            candidate[f"{label}_auroc_difference_by_severity"] = {
                severity: candidate["auroc_by_severity"][severity]
                - other["auroc_by_severity"][severity]
                for severity in CORRUPTED_SEVERITIES
            }

        candidate["beats_both_inputs"] = beats_both_inputs(candidate)
        candidate["confidence_redundant"] = is_confidence_redundant(candidate)

        candidate["responsive_control_bootstrap"] = None
        candidate["reference_control_bootstrap"] = None
        candidate["twin_bootstrap"] = None
        if candidate["macro_auroc"] is None:
            continue
        own = curves[(arm, DEPLOYABLE_SIGNAL, aggregation, method)]
        image_ids = sorted(own)
        if responsive is not None and responsive["macro_auroc"] is not None:
            candidate["responsive_control_bootstrap"] = paired_macro_bootstrap(
                image_ids, own,
                curves[(arm, DEPLOYABLE_SIGNAL, aggregation, RESPONSIVE_CONTROL_METHOD)],
                candidate_orientation=candidate["orientation"],
                control_orientation=responsive["orientation"],
                samples=samples,
            )
        if reference is not None and reference["macro_auroc"] is not None:
            candidate["reference_control_bootstrap"] = paired_macro_bootstrap(
                image_ids, own,
                reference_curves[
                    (arm, DEPLOYABLE_SIGNAL, aggregation, REFERENCE_CONTROL_METHOD)
                ],
                candidate_orientation=candidate["orientation"],
                control_orientation=reference["orientation"],
                samples=samples,
            )
        if twin is not None and twin["macro_auroc"] is not None:
            candidate["twin_bootstrap"] = paired_macro_bootstrap(
                image_ids, own,
                curves[(twin_arm, CONFIDENCE_SIGNAL, aggregation, method)],
                candidate_orientation=candidate["orientation"],
                control_orientation=twin["orientation"],
                samples=samples,
            )


def _sort_key(candidate: dict) -> tuple:
    """The spec's seven ranking criteria as one tuple, negated where higher is better.

    Every field read here is guaranteed to be a number by the gate in
    `rank_contrast_candidates` that decides which candidates reach this function: `complete`
    and `orientable` together are exactly what `summarize_contrast_candidates` requires before
    it publishes an AUROC, a median absolute Spearman or an oriented consistency, so none of the
    six can be `None` for a deployable candidate. Defaulting them to zero would rank a candidate
    that somehow arrived without them below every real one instead of saying so.
    """
    auroc = candidate["auroc_by_severity"]
    return (
        -candidate["macro_auroc"],
        -auroc[1],
        -auroc[2],
        -candidate["median_absolute_spearman"],
        -candidate["dominant_direction_fraction"],
        -candidate["oriented_adjacent_consistency"],
        candidate["arm"], candidate["aggregation"], candidate["method"],
    )


def rank_contrast_candidates(candidates: list[dict]) -> list[dict]:
    """Mark every candidate `deployable` or not, and return the deployable ones in order.

    Severities 1 and 2 sit second and third on purpose. The completed deployment analysis left
    every candidate it produced near chance at mild blur, so a macro gain driven entirely by
    severities 4 and 5 repeats a result already in hand and is not what this experiment is for.

    Every candidate is flagged, not only the survivors, so the CSV and the report can say of any
    row whether it was ranked. The two exclusions that are not failures are flagged the same way:
    a confidence twin and a raw reference control are both complete, orientable, fully measured
    controls, and they are excluded because a control that can win the comparison it exists to
    lose is not a control.

    The gate below is the spec's seven-item eligibility list, one clause per item and in its
    order, and it is written out even where a clause cannot fire on a candidate that came from
    `summarize_contrast_candidates`. Complete coverage is one such: withholding completeness
    withholds the AUROC, so on pipeline output the `complete` clause never fires alone. The three
    closed-set clauses -- one of the four declared arms, one of the three matched summaries, one
    of the four declared score methods -- are three more.

    They are written out because this function takes a list of dictionaries, not a pipeline
    stage's output, and reaches it from tests, from a re-read CSV and from whatever Task 9
    assembles. The alternative -- keeping the clauses that happen to fire and dropping the rest --
    would leave a reader comparing the spec's list with this code unable to tell an item that was
    considered and shown redundant from one that was forgotten.

    The closed method set is also the *only* thing that keeps the raw reference control out of the
    ranking. `REFERENCE_CONTROL_METHOD` is deliberately not in `SCORE_METHODS`, so a control is
    refused by the same line that refuses a typo, and there is no second clause naming it.

    The returned list holds the same dictionaries, not copies -- a reader that edits one sees
    the other, which is what keeps a report from quoting two different values for one candidate.
    """
    for candidate in candidates:
        candidate["deployable"] = bool(
            candidate["signal"] == DEPLOYABLE_SIGNAL
            and candidate.get("complete")
            and candidate.get("expected_image_count") == FULL_TUNING_IMAGE_COUNT
            and candidate.get("orientable")
            and candidate.get("macro_auroc") is not None
            and candidate["arm"] in ARM_NAMES
            and candidate["aggregation"] in AGGREGATIONS
            and candidate["method"] in SCORE_METHODS
            and candidate.get("twin_macro_auroc") is not None
        )
    return sorted(
        [candidate for candidate in candidates if candidate["deployable"]], key=_sort_key
    )
