"""Whether the reference range behaves like an anchor, measured three ways.

Three questions, deliberately not collapsed into one score. Does the reference stay still
within an image as blur increases (`within_image_drift`)? Does it differ enough between scenes
to be worth subtracting (`between_image_spread`)? And does it actually predict the responsive
range's clean level, better than a constant would (`clean_relationship`)? A reference can pass
any one of these and be useless: a perfectly stable reference that is identical for every
scene explains no scene-specific baseline at all, and a reference with enormous between-scene
spread that wanders under blur injects that wander into every contrast built on it.

Nothing here decides anything. The differential arms are expected to fail the drift and
stability checks by construction, and the module that reads these numbers is the one that
knows which family an arm belongs to.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .contrast_scores import FOLD_COUNT, robust_line
from .corruption_metrics import EXPECTED_SEVERITIES, complete_trend_metrics


def _statistics(values: np.ndarray) -> dict:
    """Ten numbers describing one severity's spread of reference scores.

    Population variance (`ddof=0`), not sample variance. These 250 images are the tuning
    partition entire, not a sample drawn from it, and the question being asked is how widely
    this reference actually spread over the images it was measured on.

    The absolute deviation is taken about the median, not the mean, so that `mad` and the
    quartiles beside it describe the same centre. A mean-centred deviation on a spread with a
    few extreme scenes reports the extremes twice -- once by pulling the centre towards them
    and again by measuring every other scene's distance from that displaced centre.
    """
    q25, median, q75 = (float(value) for value in np.percentile(values, [25, 50, 75]))
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "variance": float(np.var(values)),
        "median": median,
        "q25": q25,
        "q75": q75,
        "iqr": q75 - q25,
        "mad": float(np.median(np.abs(values - median))),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def within_image_drift(curves: dict[int, dict[int, float]]) -> dict:
    """How far each image's reference moves from its own severity-zero value.

    Measured against the image's own clean reference, never against a group mean. The whole
    premise of an internal anchor is that its level is scene-specific, so a drift measured
    against anything but that scene's own starting point would mostly be measuring
    between-scene spread instead.

    `zero_drift_fraction` counts exact zeros rather than near-zeros. A tolerance would be a
    threshold nobody chose on evidence, and the number it is there to expose -- a reference
    that is bit-identical across severities because the dynamic percentile happened to select
    the same population -- is exactly zero when it happens.

    Per-image Spearman comes from `complete_trend_metrics`, so a flat complete curve is flat
    with both correlations zero and an incomplete one is `unmeasured` with both `None`. That
    distinction is the difference between "this anchor holds still", which is the desired
    result, and "this anchor produced no curve", which is not a result at all.

    Every curve must carry all of `EXPECTED_SEVERITIES` with finite values. That is a
    precondition, not something checked here: `contrast_inputs` already refuses a non-finite
    score at load, so the pipeline cannot present one. It matters because only half of this
    function is defended against a breach -- `by_image` routes an incomplete curve to
    `unmeasured`, but `by_severity` does not filter, and one `nan` would turn that severity's
    two medians into `nan` while counting as a non-zero drift in `zero_drift_fraction`. A
    caller that stops loading through `contrast_inputs` inherits that.
    """
    by_severity: dict[int, dict] = {}
    for severity in EXPECTED_SEVERITIES:
        signed = np.array(
            [curve[severity] - curve[0] for curve in curves.values()], dtype=float
        )
        absolute = np.abs(signed)
        signed_q25, signed_q75 = (float(value) for value in np.percentile(signed, [25, 75]))
        absolute_q25, absolute_q75 = (
            float(value) for value in np.percentile(absolute, [25, 75])
        )
        by_severity[severity] = {
            "median_signed_drift": float(np.median(signed)),
            "median_absolute_drift": float(np.median(absolute)),
            "signed_q25": signed_q25,
            "signed_q75": signed_q75,
            "absolute_q25": absolute_q25,
            "absolute_q75": absolute_q75,
            "zero_drift_fraction": float(np.mean(absolute == 0.0)),
        }

    by_image: dict[int, dict] = {}
    for image_id, curve in curves.items():
        severities = sorted(curve)
        scores = [curve[severity] for severity in severities]
        trend = complete_trend_metrics(severities, scores)
        finite = [value for value in scores if math.isfinite(value)]
        by_image[image_id] = {
            "reference_range": (max(finite) - min(finite)) if finite else None,
            "signed_spearman": trend["signed_spearman"],
            "absolute_spearman": trend["absolute_spearman"],
        }
    return {"by_severity": by_severity, "by_image": by_image}


def between_image_spread(curves: dict[int, dict[int, float]], drift: dict) -> dict:
    """The reference's spread across images at each severity, and drift relative to it.

    `stability_to_spread` is the median absolute within-image drift at this severity divided
    by the *clean* interquartile range. Lower is better: it asks whether the anchor moves less
    under corruption than it differs between scenes, which is the only sense in which
    subtracting it removes more signal than it adds noise. The denominator is severity zero's
    spread rather than this severity's, so the yardstick is fixed and five severities' ratios
    can be compared with each other.

    Severity zero has no ratio. Its drift is identically zero by definition, so a ratio there
    would be a guaranteed 0.0 that reads like a passing score.

    A zero clean interquartile range makes the ratio `None`, never a substituted constant. A
    reference identical across every scene has no scene-specific baseline to explain, and
    dividing by an epsilon would turn that finding into a very large number that looks like a
    measurement of instability instead.
    """
    result: dict[int, dict] = {}
    clean_iqr = None
    for severity in EXPECTED_SEVERITIES:
        values = np.array([curve[severity] for curve in curves.values()], dtype=float)
        statistics = _statistics(values)
        if severity == 0:
            clean_iqr = statistics["iqr"]
            statistics["stability_to_spread"] = None
        elif clean_iqr == 0.0:
            statistics["stability_to_spread"] = None
        else:
            statistics["stability_to_spread"] = (
                drift["by_severity"][severity]["median_absolute_drift"] / clean_iqr
            )
        result[severity] = statistics
    return result


def _correlation(reference: np.ndarray, responsive: np.ndarray) -> tuple:
    """Pearson and Spearman, or `None` when either side is constant.

    SciPy returns `nan` with a `ConstantInputWarning` for a constant input, and a `nan` in
    `summary.json` is not writable JSON. `None` says "undefined here" in a way a reader and a
    serialiser both understand.
    """
    if np.unique(reference).size < 2 or np.unique(responsive).size < 2:
        return None, None
    return float(pearsonr(reference, responsive).statistic), float(
        spearmanr(reference, responsive).statistic
    )


def clean_relationship(
    references: dict[int, float], responsives: dict[int, float], *, folds: dict[int, int]
) -> dict:
    """The normal clean relationship between the two ranges, cross-fitted and challenged.

    Two errors are reported and they are the point of the function. The first is what the
    robust line achieves on images it never saw. The second is what a fold-specific constant
    achieves -- always predicting the other folds' median clean responsive score, ignoring the
    reference entirely. The line earns the word "predictive" only by beating that constant,
    because a line that cannot is not explaining scene-specific baseline; it is reproducing the
    group average with extra steps, and an in-sample correlation of 0.9 can sit on top of
    exactly that.

    The comparison is strict. A line that ties with the constant has demonstrated nothing --
    the tie is what happens when the fitted slope is zero and the line *is* the constant with
    a fitted intercept -- and calling that predictive would let every arm whose reference
    carries no scene information collect the label for free.

    Both errors are medians of absolute errors rather than means. One clean scene with an
    extreme baseline would otherwise decide which of the two predictors wins.

    `final_slope` and `final_offset` are fitted on all clean images and are what a deployment
    would store, but they never touch the two error numbers above -- an image must not help
    construct the line that predicts it.

    `folds` must give every image in `references` a fold in `range(FOLD_COUNT)`, and one
    outside that range is refused rather than skipped. It is the only way this function can
    return a wrong number that looks right: the cross-fitting loop visits `range(FOLD_COUNT)`,
    so an image assigned past the end is never held out, stays in every training set, and
    disappears from both error medians -- which then move without anything reading as missing.
    `assign_folds` cannot produce one, so the guard exists for the caller that stops using it.
    """
    image_ids = sorted(references)
    stray = sorted({folds[image_id] for image_id in image_ids} - set(range(FOLD_COUNT)))
    if stray:
        raise ValueError(
            f"every fold must lie in range({FOLD_COUNT}); got {stray}. An image assigned "
            "outside that range is never held out, never predicted, and silently absent from "
            "both error medians"
        )
    reference = np.array([references[image_id] for image_id in image_ids], dtype=float)
    responsive = np.array([responsives[image_id] for image_id in image_ids], dtype=float)
    pearson, spearman = _correlation(reference, responsive)
    final = robust_line(reference, responsive)

    fold_lines: dict[int, tuple[float, float] | None] = {}
    line_errors: list[float] = []
    constant_errors: list[float] = []
    residuals: list[float] = []
    for fold in range(FOLD_COUNT):
        held = [image_id for image_id in image_ids if folds[image_id] == fold]
        trained = [image_id for image_id in image_ids if folds[image_id] != fold]
        if not held:
            continue
        train_reference = np.array([references[i] for i in trained], dtype=float)
        train_responsive = np.array([responsives[i] for i in trained], dtype=float)
        line = robust_line(train_reference, train_responsive) if trained else None
        fold_lines[fold] = line
        constant = float(np.median(train_responsive)) if trained else None
        for image_id in held:
            if constant is not None:
                constant_errors.append(abs(responsives[image_id] - constant))
            if line is not None:
                slope, offset = line
                residual = responsives[image_id] - (offset + slope * references[image_id])
                residuals.append(residual)
                line_errors.append(abs(residual))

    crossfit_error = float(np.median(line_errors)) if line_errors else None
    constant_error = float(np.median(constant_errors)) if constant_errors else None
    residual_array = np.array(residuals, dtype=float)
    residual_median = float(np.median(residual_array)) if residuals else None
    if residuals:
        q25, q75 = (float(value) for value in np.percentile(residual_array, [25, 75]))
        residual_iqr = q75 - q25
        residual_mad = float(np.median(np.abs(residual_array - residual_median)))
    else:
        residual_iqr = residual_mad = None

    return {
        "pearson": pearson,
        "spearman": spearman,
        "final_slope": final[0] if final else None,
        "final_offset": final[1] if final else None,
        "fold_lines": {
            fold: list(line) if line else None for fold, line in fold_lines.items()
        },
        "crossfit_median_absolute_error": crossfit_error,
        "constant_median_absolute_error": constant_error,
        "residual_median": residual_median,
        "residual_iqr": residual_iqr,
        "residual_mad": residual_mad,
        "predictive": bool(
            crossfit_error is not None
            and constant_error is not None
            and crossfit_error < constant_error
        ),
    }
