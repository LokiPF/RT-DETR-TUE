"""Separation, harm and inference helpers; every score is oriented so higher means more degraded."""
from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.stats import ConstantInputWarning, pearsonr, rankdata, spearmanr
from sklearn.metrics import average_precision_score

from ..evaluation import binary_auroc

COVERAGES = tuple(np.round(np.linspace(1.0, 0.05, 20), 4))
UQ_DETR_LAMBDA_GRID = (0, 0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20)  # uq_detr.fit_lambda default


def auroc(clean, degraded) -> float:
    return binary_auroc(clean, degraded)


def aupr(clean, degraded) -> float:
    """Average precision with degraded images as the positive class."""
    clean, degraded = np.asarray(clean, float), np.asarray(degraded, float)
    labels = np.concatenate([np.zeros(clean.size), np.ones(degraded.size)])
    return float(average_precision_score(labels, np.concatenate([clean, degraded])))


def fpr_at_95_tpr(clean, degraded) -> float:
    degraded = np.sort(np.asarray(degraded, float))
    threshold = degraded[int(math.floor(0.05 * degraded.size))]
    return float(np.mean(np.asarray(clean, float) >= threshold))


def condition_aurocs(clean, degraded) -> np.ndarray:
    """AUROC of clean (n,) against each row of degraded (c, n), vectorised with average ranks."""
    clean, degraded = np.asarray(clean, float), np.asarray(degraded, float)
    n, m = clean.size, degraded.shape[1]
    joined = np.concatenate([np.broadcast_to(clean, (degraded.shape[0], n)), degraded], axis=1)
    ranks = rankdata(joined, method="average", axis=1)
    return (ranks[:, n:].sum(axis=1) - m * (m + 1) / 2) / (n * m)


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        return float(spearmanr(x[keep], y[keep]).statistic)


def mean_within_condition_spearman(delta_scores, delta_risks) -> float:
    """Mean over conditions (columns) of the per-image Spearman correlation."""
    values = [spearman(delta_scores[:, c], delta_risks[:, c]) for c in range(delta_scores.shape[1])]
    values = [v for v in values if not math.isnan(v)]
    return float(np.mean(values)) if values else math.nan


def risk_coverage(scores, risks, coverages=COVERAGES):
    """Mean risk of the images kept when the highest-scoring ones are rejected first."""
    scores, risks = np.asarray(scores, float), np.asarray(risks, float)
    keep = np.isfinite(risks)
    ranked = risks[keep][np.argsort(scores[keep], kind="stable")]
    kept = [ranked[: max(1, int(math.ceil(c * ranked.size)))].mean() for c in coverages]
    return np.asarray(coverages, float), np.asarray(kept, float)


def aurc(scores, risks, coverages=COVERAGES) -> float:
    return float(risk_coverage(scores, risks, coverages)[1].mean())


def fit_lambda_from_parts(conf_pos, conf_neg, reliability, grid=UQ_DETR_LAMBDA_GRID):
    """uq_detr.fit_lambda on precomputed Conf+/Conf-: first lambda with the highest Pearson r."""
    conf_pos, conf_neg, reliability = (np.asarray(v, float) for v in (conf_pos, conf_neg, reliability))
    valid = ~np.isnan(reliability)
    if valid.sum() < 3:
        raise ValueError("need at least 3 images with a defined reliability value")
    best_lambda, best_pcc = 0.0, -np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        for lam in grid:
            pcc = pearsonr(conf_pos[valid] - lam * conf_neg[valid], reliability[valid])[0]
            if pcc > best_pcc:
                best_lambda, best_pcc = float(lam), float(pcc)
    return best_lambda, best_pcc


def cross_fit_lambda(conf_pos, conf_neg, reliability, folds):
    """Fit lambda for each fold on the other folds only."""
    conf_pos, conf_neg, reliability, folds = (np.asarray(v) for v in (conf_pos, conf_neg, reliability, folds))
    per_fold = {}
    for fold in np.unique(folds):
        others = folds != fold
        per_fold[int(fold)] = fit_lambda_from_parts(conf_pos[others], conf_neg[others], reliability[others])[0]
    return np.array([per_fold[int(f)] for f in folds]), per_fold


def bootstrap(statistic, n_images: int, samples: int = 1000, seed: int = 44) -> dict:
    """Paired whole-image bootstrap: the same resampled images feed every quantity in `statistic`."""
    generator = np.random.default_rng(seed)
    draws = [statistic(generator.integers(0, n_images, n_images)) for _ in range(samples)]
    return {key: tuple(float(v) for v in np.nanpercentile([d[key] for d in draws], (2.5, 97.5)))
            for key in draws[0]}
