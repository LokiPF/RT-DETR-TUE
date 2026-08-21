"""Four separate questions about one candidate's six-severity score curve.

A "candidate" here is one row family from the corruption analysis -- one signal, bucket
scheme, confidence bin, membership mode, padding mode, aggregation and score scope -- observed
at severities 0 through 5 on each image. This module answers, in order:

1. **does the signal move with blur inside one image?** `complete_trend_metrics` reports a
   *signed* Spearman and its absolute value as two separate numbers;
2. **which single direction should this candidate be read in?** `choose_orientation` picks
   `+1` or `-1` once, from the tuning group's signed trends;
3. **does the oriented curve behave?** `oriented_curve_metrics` reports the fraction of
   adjacent severity steps that move the chosen way, and whether severity 5 clears severity 0;
4. **does the oriented raw score separate corrupted images from clean ones across scenes?**
   `binary_auroc` and `severity_aurocs`.

Four decisions in here look arbitrary out of context and are not.

**Absolute Spearman does not make AUROC redundant, and neither replaces the other.** Absolute
Spearman is a *within-image* statistic: it ranks one scene's six blurred copies against each
other, so the scene's own baseline distance cancels out of it entirely. AUROC is a
*across-scene* statistic on the raw score: it asks whether a corrupted image's score outranks
a *different, unrelated* clean image's score. A candidate can score a near-perfect absolute
Spearman on every image while its clean and corrupted score distributions overlap almost
completely in the pooled comparison, because different clean scenes sit at very different
baseline distances and blur moves each one by less than the spread between them. Both numbers
are reported, and a reader who collapses them into one has thrown away the distinction between
"blur moves this feature" and "this feature identifies blurred images".

**A flat curve and a missing curve are different facts.** Six finite identical scores are a
completed measurement that found no trend; they are recorded as `direction="flat"` with
`signed_spearman` and `absolute_spearman` set to a documented `0.0`, and they *count in the
denominator* of any later "fraction of images with a trend". A curve that is short, out of
order or carries a non-finite score is `direction="unmeasured"` with both correlations `None`,
and must never be folded into the zeros: doing so would let a candidate that failed to produce
scores masquerade as one that produced scores showing nothing. This is a declared convention
of the metric, not silent missing-data imputation -- which is why `spearmanr`'s own `nan` for
a constant input is refused rather than converted (see `complete_trend_metrics`), and why
`scene_uncertainty.metrics.monotonicity_metrics`, which does drop missing severities and does
map a non-finite correlation to `0.0`, is a different function for a different question rather
than something to reuse here.

**Orientation is chosen once per candidate and then locked.** Never per severity, per image or
per metric. The reason is that "which direction does this feature move under blur?" is a claim
about the feature, and re-deciding it downstream turns any noise into apparent signal: pick the
better of the two directions at each severity and a pure-noise candidate reports five AUROCs
all at or above 0.5. So `choose_orientation` consumes a *group* of signed trends and returns
one integer, `binary_auroc` and `oriented_curve_metrics` take that integer as an argument, and
nothing in this module ever substitutes a value's direction-reversed twin for it. A candidate
whose median signed trend is exactly zero has no direction to lock; `choose_orientation`
returns `None` for it, and the caller is expected to exclude it from deployment ranking rather
than break the tie.

**AUROC here is a ranking statistic, never a probability of corruption.** `0.5` is chance-level
ranking and `1.0` is perfect ranking; a value *below* `0.5` after the locked orientation is
applied means that severity ranks the other way -- the candidate behaves against its own
selected direction there, which is a finding worth keeping and not a defect to flip away. It is
not a calibrated probability that an image is corrupted and must never be read, named or
reported as one.

The module is deliberately pure: NumPy and SciPy over sequences the caller already holds, no
file access, no torch, and no import from another `scene_uncertainty` module. Every number it
returns is a function of its arguments alone.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr


EXPECTED_SEVERITIES = tuple(range(6))
"""The blur ladder every curve must present, in order: clean plus five increasing severities.

Written as an exact ordered tuple rather than a count because severity is an *identity* on
this axis, not a position. A curve handed over as `[5, 4, 3, 2, 1, 0]` has six points and
correlates `+1` against its own axis, so a length check alone would report a strongly
increasing trend for a decreasing one.
"""


def complete_trend_metrics(severities: Iterable[int], scores: Iterable[float]) -> dict:
    """How one image's score moves across the full blur ladder, signed and unsigned.

    Returns four keys. `signed_spearman` keeps the direction, because the direction is what
    `choose_orientation` later aggregates into a locked orientation; `absolute_spearman` drops
    it, because "blur moves this feature" is true whichever way it moves and a candidate that
    is strongly negative on every image is as usable as one that is strongly positive. They are
    two numbers rather than one so that a downstream ranking cannot silently read strength as
    direction, and both are always present so no caller has to derive one from the other.

    Incompleteness is refused rather than worked around. Any of: fewer or more than six points,
    a severity axis that is not exactly `EXPECTED_SEVERITIES` in order, or a non-finite score,
    returns `fully_measured=False`, both correlations `None`, and `direction="unmeasured"`.
    Interpolating, dropping the bad points and correlating the rest, or substituting `0.0`
    would each turn "this candidate did not produce a curve here" into a number that reads like
    a measurement. `scene_uncertainty.metrics.monotonicity_metrics` deliberately makes the
    other choice -- it drops non-finite severities and grades what survives -- and reports
    `finite_count` so the reader can see it happened; this function has a different job and
    reports no partial curve at all.

    A constant curve is short-circuited before `spearmanr` is called. Six identical finite
    scores really do contain no trend, and `0.0` with `direction="flat"` is this module's
    declared value for that -- distinct from `unmeasured`, and included in the denominator of
    any later "how many images showed a trend". The short circuit is not merely cosmetic:
    `spearmanr` on a constant input returns `nan` and raises `ConstantInputWarning`, so the
    alternative is either a non-finite correlation flowing into the sign comparison below, or a
    `nan`-to-zero conversion indistinguishable from the missing-data imputation the paragraph
    above refuses. Deciding it here keeps the two cases separable and keeps the warning out of
    the run.

    `direction` is derived from the sign of `signed_spearman` alone, so it never disagrees with
    it: positive is `increasing`, negative `decreasing`, exactly zero `flat`.
    """
    severity = np.asarray(list(severities), dtype=int)
    values = np.asarray(list(scores), dtype=float)
    complete = (
        severity.shape == (6,)
        and values.shape == (6,)
        and tuple(severity.tolist()) == EXPECTED_SEVERITIES
        and bool(np.isfinite(values).all())
    )
    if not complete:
        return {
            "fully_measured": False,
            "signed_spearman": None,
            "absolute_spearman": None,
            "direction": "unmeasured",
        }
    if bool(np.all(values == values[0])):
        signed = 0.0
    else:
        signed = float(spearmanr(severity, values).statistic)
    direction = "increasing" if signed > 0 else "decreasing" if signed < 0 else "flat"
    return {
        "fully_measured": True,
        "signed_spearman": signed,
        "absolute_spearman": abs(signed),
        "direction": direction,
    }


def choose_orientation(signed_spearman: Iterable[float]) -> int | None:
    """The one direction a candidate is read in, decided once from a group of signed trends.

    Takes every image's `signed_spearman` for a single candidate -- the tuning partition's, so
    the direction is not chosen on the data it is later scored against -- and returns `+1` if
    the candidate rises with blur, `-1` if it falls, or `None` if the group does not say.

    The median rather than the mean, because the input is a group of correlations in which a
    few images can be extreme and most are not: two images at `+0.9` outvote three spread over
    `-0.3` to `-0.1` on a mean, at `+0.24`, and lose on a median, at `-0.1`. The question being
    asked is "which way do these images mostly move?", not "what is the average movement?", and
    a candidate that drifts down on most images is a candidate to read downwards even when a
    couple of scenes swing hard the other way. Non-finite entries are dropped, not
    read as zero. They are the `unmeasured` images from `complete_trend_metrics`, and counting
    them as no-trend would pull the median towards zero in proportion to how often the
    candidate failed to produce a curve -- turning missing measurements into evidence of
    ambiguity.

    Returns `None` in the two cases where no direction is supported: nothing finite survived
    the filter, and a median of exactly zero. `None` is not a third direction; it marks a
    candidate that cannot be oriented, and the caller is expected to exclude it from
    deployment ranking. Breaking the tie towards `+1` would give every unorientable candidate
    a direction and, with it, a full set of curve checks and AUROCs that look exactly like an
    oriented candidate's.

    Whatever this returns is the *only* direction the candidate is ever read in. It is not
    re-derived per severity, per image or per metric, and no AUROC is replaced by its
    direction-reversed value afterwards.
    """
    values = np.asarray(list(signed_spearman), dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    median = float(np.median(values))
    return 1 if median > 0 else -1 if median < 0 else None


def oriented_curve_metrics(scores: Sequence[float], orientation: int) -> dict:
    """Two deployment checks on one image's curve, read in the candidate's locked direction.

    Multiplying by `orientation` turns "does this fall with blur?" into "does this rise with
    blur?" without changing any gap between severities, so a single pair of comparisons covers
    both kinds of candidate. `orientation` must be `-1` or `1`; anything else -- including `0`,
    a float, or `None` from an unorientable candidate -- is a caller error rather than a
    direction, and is refused instead of being coerced.

    `adjacent_consistency` is the fraction of the five severity steps that do not move against
    the locked direction, so it is one of `0.0, 0.2, 0.4, 0.6, 0.8, 1.0` and distinguishes a
    curve with one local reversal from one that is backwards throughout. A step of exactly zero
    counts as consistent: it does not contradict the direction, and treating flat stretches as
    violations would penalise a saturating feature for the severities after it saturates.
    Complementing this, `max_blur_above_clean` is strict -- severity 5 must be *above* severity
    0, not merely level with it -- because a curve that ends where it started has not separated
    the strongest blur from clean no matter how orderly its steps were. A perfectly flat curve
    therefore scores `1.0` and `False`, and the pair is what says so; either number alone would
    misdescribe it.

    A curve that is not six finite points returns both keys as `None`, matching
    `complete_trend_metrics`' refusal to convert a missing curve into a measured one. Both keys
    are always present, so a caller can read them without first checking for their existence.
    """
    if orientation not in (-1, 1):
        raise ValueError("orientation must be -1 or 1")
    values = np.asarray(scores, dtype=float)
    if values.shape != (6,) or not bool(np.isfinite(values).all()):
        return {"adjacent_consistency": None, "max_blur_above_clean": None}
    oriented = values * orientation
    return {
        "adjacent_consistency": float(np.mean(np.diff(oriented) >= 0)),
        "max_blur_above_clean": bool(oriented[5] > oriented[0]),
    }


def binary_auroc(
    clean_scores: Sequence[float],
    corrupted_scores: Sequence[float],
    *,
    orientation: int,
) -> float:
    """How well the oriented raw score ranks corrupted images above clean ones, across scenes.

    This is the cross-scene counterpart to `complete_trend_metrics`. Each score comes from a
    different image, so nothing here is normalised by a scene's own clean baseline: a candidate
    passes only if the absolute score of a blurred scene tends to outrank the absolute score of
    an unrelated clean scene. That is a strictly harder question than the per-image trend, and
    a candidate can pass one and fail the other.

    Computed as the Mann-Whitney U statistic normalised by `len(clean) * len(corrupted)`, which
    equals the area under the ROC curve exactly. `rankdata(..., method="average")` is what makes
    it tie-aware: a clean and a corrupted score that are exactly equal share a rank and
    contribute half a point, the same weight ROC integration gives them. `method="min"`,
    `"max"` or `"ordinal"` would each resolve those ties in one group's favour and shift the
    result -- on two clean scores `[0, 1]` against two corrupted `[1, 2]`, averaging gives
    0.875 where `min` gives 0.75 and `max` and `ordinal` give 1.0. SciPy's `rankdata` is used
    rather than a scikit-learn ROC helper because this project does not depend on
    scikit-learn and the closed form above needs nothing else.

    Read the result as a ranking quality, never as a probability that an image is corrupted:
    `0.5` is chance-level ranking, `1.0` perfect, and a value below `0.5` means this comparison
    ranks the *other* way under the candidate's locked orientation. That last case is a real
    result about the candidate and is reported as-is; reversing the orientation to bring it
    above `0.5` would be choosing a direction from the scored data, which is exactly what
    locking the orientation in `choose_orientation` exists to prevent. Applying `orientation`
    is an exact mirror -- flipping it returns `1 - auroc` for any input, ties included -- so
    nothing is lost by reporting the low value.

    Refusals rather than sentinel values, because each of these makes the ratio meaningless
    rather than merely unusual: an orientation that is not `-1` or `1`, an empty clean or
    corrupted group (the denominator would be zero, and "no clean images" is not a score of
    0.5), and any non-finite score (`nan` sorts unpredictably and would silently take a rank).
    """
    if orientation not in (-1, 1):
        raise ValueError("orientation must be -1 or 1")
    clean = np.asarray(clean_scores, dtype=float) * orientation
    corrupted = np.asarray(corrupted_scores, dtype=float) * orientation
    if not clean.size or not corrupted.size:
        raise ValueError("AUROC needs non-empty clean and corrupted groups")
    if not bool(np.isfinite(clean).all() and np.isfinite(corrupted).all()):
        raise ValueError("AUROC scores must be finite")
    ranks = rankdata(np.concatenate([clean, corrupted]), method="average")
    positive_count = corrupted.size
    positive_rank_sum = float(ranks[clean.size:].sum())
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2
    ) / (clean.size * positive_count)


def severity_aurocs(
    scores_by_severity: dict[int, Sequence[float]], orientation: int
) -> tuple[dict[int, float], float]:
    """Severity 0 against each of severities 1 to 5, plus the macro average of the five.

    Returns the five per-severity AUROCs *and* their mean, never the mean alone. Blur strength
    is the axis along which a corruption detector is most likely to be uneven -- separating
    severity 5 from clean is easy and separating severity 1 from clean is the part that decides
    whether a deployment check fires before the damage is done -- so a single number would let
    strong performance at severe blur cover for chance-level performance at mild blur. Keeping
    all five available means a reader can see that shape; the macro average is offered as a
    convenience for ordering candidates, not as a summary that replaces its parts.

    The macro value is the ordinary arithmetic mean of the five, each severity weighted equally
    regardless of how many images it contributed. Weighting by group size would let whichever
    severity happened to retain the most scored images dominate the ranking.

    Every severity in `EXPECTED_SEVERITIES` must be present and no others: severity 0 supplies
    the clean group for all five comparisons, so a missing severity would silently shrink the
    macro average's denominator, and an unexpected key would mean the caller and this function
    disagree about what the blur ladder is. `orientation` is passed through to `binary_auroc`
    unchanged and unexamined -- it is the candidate's single locked direction, not something
    re-decided per severity -- so every value returned here is read the same way, including any
    that land below 0.5.
    """
    if set(scores_by_severity) != set(EXPECTED_SEVERITIES):
        raise ValueError("AUROC needs clean severity 0 and corrupted severities 1 through 5")
    clean = scores_by_severity[0]
    values = {
        severity: binary_auroc(clean, scores_by_severity[severity], orientation=orientation)
        for severity in EXPECTED_SEVERITIES[1:]
    }
    return values, float(np.mean(list(values.values())))
