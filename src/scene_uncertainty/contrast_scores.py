"""The four contrast methods, the fold rule, and the robust clean line.

Everything here takes numbers and returns numbers. Nothing here knows what an arm is, which
bucket a score came from, or that a filesystem exists -- which is what makes the whole module
testable by hand arithmetic, and what stops a change of experiment design from needing a change
of maths.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

SCORE_METHODS = ("raw_responsive", "raw_gap", "relative_gap", "clean_residual")
FOLD_COUNT = 5


def _validated(
    reference: float, responsive: float, *, non_negative: bool = False
) -> tuple[float, float]:
    """Both inputs as finite floats, and non-negative too when the caller needs that.

    Non-finite is always refused, for the reason `complete_trend_metrics` refuses it: a `nan`
    propagates through subtraction into a score that is neither a measurement nor an absence,
    and lands in a CSV cell that reads as neither.

    Non-negativity is *not* always refused, and that is the correction the completed run
    forced. A `layer_N` score is a raw mean-kNN distance and cannot be negative. A `combined`
    score is a robust z-score against the clean median and goes negative for any selection
    below it -- 548 of the completed run's published `combined` statistics are, against none at
    `layer_2`. So the check belongs to the one method that mathematically requires it rather
    than to every method: `relative_gap`'s scale invariance and its bounds both collapse on a
    signed input, while a raw gap and a residual are perfectly well defined on one.
    """
    reference = float(reference)
    responsive = float(responsive)
    if not (math.isfinite(reference) and math.isfinite(responsive)):
        raise ValueError(
            f"contrast inputs must be finite: reference={reference}, responsive={responsive}"
        )
    if non_negative and (reference < 0.0 or responsive < 0.0):
        raise ValueError(
            f"the symmetric relative gap needs non-negative inputs: reference={reference}, "
            f"responsive={responsive}; a signed scope must not declare this method"
        )
    return reference, responsive


def raw_responsive(reference: float, responsive: float) -> float:
    """The responsive range alone: the control every contrast has to beat.

    Takes `reference` it does not use, so that the four methods share one signature and the
    caller's dispatch table cannot pair a method with the wrong arguments. Validation still
    runs on both, because a control computed from a row whose reference is corrupt is a control
    over a different population than the contrast it is being compared with.
    """
    _, responsive = _validated(reference, responsive)
    return responsive


def raw_gap(reference: float, responsive: float) -> float:
    """`responsive - reference`, signed.

    Signed rather than absolute. An absolute value would map "the responsive range moved
    unusually far above its baseline" and "the responsive range collapsed below it" onto the
    same number, and those are opposite events. The candidate's locked orientation is what
    turns a consistently negative score into a usable one, and it can only do that if the sign
    survives to it.
    """
    reference, responsive = _validated(reference, responsive)
    return responsive - reference


def relative_gap(reference: float, responsive: float) -> float:
    """`2 * (responsive - reference) / (responsive + reference)`, the scale-free contrast.

    Two scenes whose raw distances differ tenfold get the same value when the
    responsive-to-reference relationship is proportional, which is the multiplicative
    counterpart to what `raw_gap` removes additively.

    This method alone requires non-negative inputs, and refuses a negative one rather than
    accommodating it: an arm whose scope can go negative has no business declaring this method,
    and Task 4 excludes it there. Given non-negative inputs the denominator cannot be negative
    and the only degenerate case is both being exactly zero -- defined as zero, because
    "neither range moved at all" is an absence of contrast and not an undefined one. No epsilon
    is added. An epsilon would be a tunable constant sitting inside a score that is otherwise
    entirely determined by the data, and it would make the near-zero region's values a function
    of a number nobody chose on evidence.
    """
    reference, responsive = _validated(reference, responsive, non_negative=True)
    total = responsive + reference
    if total == 0.0:
        return 0.0
    return 2.0 * (responsive - reference) / total


def clean_residual(
    reference: float, responsive: float, *, slope: float, offset: float
) -> float:
    """How far the responsive range sits above what a clean image with this baseline would show.

    `slope` and `offset` come from clean severity-zero rows only, so the line encodes the normal
    clean relationship and nothing about corruption. A positive residual therefore means "higher
    than clean-normal for this scene", which is the claim the score is making.
    """
    reference, responsive = _validated(reference, responsive)
    return responsive - (offset + slope * reference)


def contrast_score(
    method: str,
    reference: float,
    responsive: float,
    *,
    line: tuple[float, float] | None = None,
) -> float:
    """One score by name, with the residual's line supplied rather than looked up.

    `line` is an argument instead of module state because the residual is cross-fitted: the
    same `(arm, summary, signal)` uses five different lines across the five folds, and a
    module-level cache would hand one image the line its own fold fitted. Passing it in makes
    the fold the caller's responsibility, which is where the fold assignment already lives.

    A missing line for `clean_residual` raises rather than returning `None`. The caller knows
    before it starts whether `robust_line` returned a line, and a residual candidate whose line
    is unavailable is excluded with a recorded reason -- not silently filled with a value that
    would read as a measurement.
    """
    if method == "raw_responsive":
        return raw_responsive(reference, responsive)
    if method == "raw_gap":
        return raw_gap(reference, responsive)
    if method == "relative_gap":
        return relative_gap(reference, responsive)
    if method == "clean_residual":
        if line is None:
            raise ValueError(
                "clean_residual needs a fitted line; a candidate whose clean relationship is "
                "unavailable must be excluded, not scored"
            )
        slope, offset = line
        return clean_residual(reference, responsive, slope=slope, offset=offset)
    raise ValueError(f"unknown contrast score method: {method!r}")


def assign_folds(image_ids: Iterable[int]) -> dict[int, int]:
    """Sorted image position modulo five, as `{image_id: fold}`.

    Sorted rather than input-ordered, so the assignment is a property of the image roster and
    not of whatever order the rows happened to arrive in. Two runs over the same 250 images
    produce the same five folds whether the CSV was written by severity or by candidate, which
    is what makes a stored fold assignment worth comparing across runs at all.

    All six severities of an image share its fold -- enforced by keying on `image_id` alone,
    which is the only place that rule can be enforced once. Assigning per row would let an
    image's severity-3 measurement help fit the line that predicts its severity-0 value.
    """
    ordered = sorted({int(image_id) for image_id in image_ids})
    return {image_id: index % FOLD_COUNT for index, image_id in enumerate(ordered)}


def robust_line(
    references: Sequence[float], responsives: Sequence[float]
) -> tuple[float, float] | None:
    """Median pairwise slope, then median intercept. `None` when the reference is constant.

    A Theil-Sen line rather than least squares, and deterministic rather than sampled. Least
    squares would let one clean scene with an extreme baseline set the slope that every other
    scene's residual is measured against; the median of the pairwise slopes needs more than
    half the pairs to move before it does. With 250 clean images the 31,125 pairs are cheap
    enough to enumerate outright, so there is no sampling step and therefore no seed and no
    run-to-run variation in a number that becomes a deployment constant.

    Pairs with equal reference values are dropped rather than counted: their slope is a
    division by zero, and treating it as infinite or as zero would both be inventing a
    measurement from two points that say nothing about slope. Fewer than two distinct finite
    reference values leaves nothing to fit, and that returns `None` -- never a zero slope,
    which would look like "the ranges are unrelated" rather than "this could not be measured".
    """
    reference = np.asarray(references, dtype=float)
    responsive = np.asarray(responsives, dtype=float)
    if reference.shape != responsive.shape:
        raise ValueError(
            "robust line needs one responsive value per reference value: "
            f"{reference.shape} against {responsive.shape}"
        )
    finite = np.isfinite(reference) & np.isfinite(responsive)
    reference = reference[finite]
    responsive = responsive[finite]
    if np.unique(reference).size < 2:
        return None
    left, right = np.triu_indices(reference.size, k=1)
    run = reference[right] - reference[left]
    usable = run != 0.0
    slopes = (responsive[right][usable] - responsive[left][usable]) / run[usable]
    slope = float(np.median(slopes))
    offset = float(np.median(responsive - slope * reference))
    return slope, offset
