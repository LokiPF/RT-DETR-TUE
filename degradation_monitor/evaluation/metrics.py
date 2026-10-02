"""Separation metrics, per-stage z-scoring and the paired bootstrap; every score is oriented so higher means more degraded."""
from __future__ import annotations

import math
import multiprocessing

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score


def _score_vector(values, *, name: str) -> np.ndarray:
    try:
        source = values if isinstance(values, np.ndarray) else list(values)
        array = np.asarray(source, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a non-empty finite one-dimensional input") from error
    if array.ndim != 1 or not array.size or not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} must be a non-empty finite one-dimensional input")
    return array


def binary_auroc(clean, corrupted) -> float:
    """Return tie-correct AUROC, with larger values indicating corruption."""
    clean_values = _score_vector(clean, name="clean scores")
    corrupted_values = _score_vector(corrupted, name="corrupted scores")
    ranks = rankdata(np.concatenate((clean_values, corrupted_values)), method="average")
    positives = corrupted_values.size
    rank_sum = float(ranks[clean_values.size :].sum())
    return float(
        (rank_sum - positives * (positives + 1) / 2)
        / (clean_values.size * positives)
    )


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


def group_separation(clean, degraded) -> tuple[float, float, float]:
    """Mean AUROC, AUPR and FPR95 over the conditions (rows) of `degraded`."""
    return (float(condition_aurocs(clean, degraded).mean()),
            float(np.mean([aupr(clean, row) for row in degraded])),
            float(np.mean([fpr_at_95_tpr(clean, row) for row in degraded])))


def stage_zstats(stage_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and population standard deviation of each stage's score over clean images."""
    stage_scores = np.asarray(stage_scores, dtype=np.float64)
    mean, std = stage_scores.mean(axis=0), stage_scores.std(axis=0)
    if not np.all(std > 0):
        raise ValueError("every stage needs a positive spread over the clean images")
    return mean, std


def zscored_sum(stage_scores: np.ndarray, mean, std) -> np.ndarray:
    """Sum over stages of each stage's score standardised with clean-image statistics."""
    return ((np.asarray(stage_scores, dtype=np.float64) - np.asarray(mean)) / np.asarray(std)).sum(axis=1)


_STATISTIC = None  # the statistic forked bootstrap workers evaluate


def _evaluate(draw):
    return _STATISTIC(draw)


def bootstrap(statistic, n_images: int, samples: int = 1000, seed: int = 44, workers: int = 1) -> dict:
    """Paired whole-image bootstrap: the same resampled images feed every quantity in `statistic`.

    The draws come from one seeded generator in a fixed order, so `workers` (forked processes) changes only the speed.
    """
    global _STATISTIC
    generator = np.random.default_rng(seed)
    draws = [generator.integers(0, n_images, n_images) for _ in range(samples)]
    if workers > 1:
        _STATISTIC = statistic
        try:
            with multiprocessing.get_context("fork").Pool(workers) as pool:
                values = pool.map(_evaluate, draws, chunksize=max(1, samples // (4 * workers)))
        finally:
            _STATISTIC = None
    else:
        values = [statistic(draw) for draw in draws]
    return {key: tuple(float(v) for v in np.nanpercentile([d[key] for d in values], (2.5, 97.5)))
            for key in values[0]}
