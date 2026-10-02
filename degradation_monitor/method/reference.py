"""The clean reference: per-dimension spreads, and the stage-4 content key's search for the most similar clean scenes.

The fixed choices below were pre-registered before the 5,000-image confirmation; ablations pass other values explicitly.
"""
from __future__ import annotations

import numpy as np

STD_FLOOR = 0.01  # every dimension's spread is at least 1% of the median positive spread of its layer
KEY_LAYER = "s4"
SCORED_LAYERS = ("s1", "s2", "s3")
NEIGHBOURS = 50
BANK_IMAGES = 2000  # clean train images in the reference bank
ZSTAT_IMAGES = 500  # further clean train images that set each stage's z-statistics
CHUNK = 2048


def fit_own_average(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension clean mean and population spread; the spread is floored so dead dimensions stay finite."""
    reference = np.asarray(reference, dtype=np.float64)
    if reference.ndim != 2 or reference.shape[0] < 2:
        raise ValueError("need at least two clean rows of shape (rows, dim)")
    mean, std = reference.mean(axis=0), reference.std(axis=0)
    positive = std[std > 0]
    if positive.size == 0:
        raise ValueError("the clean rows have no spread")
    return mean, np.maximum(std, STD_FLOOR * float(np.median(positive)))


def nearest_rows(queries: np.ndarray, reference: np.ndarray, k: int, chunk: int = CHUNK) -> np.ndarray:
    """Indices of each query's k nearest reference rows (Euclidean), in no particular order."""
    queries, reference = np.asarray(queries, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if not 1 <= k <= len(reference):
        raise ValueError("k must be between 1 and the number of reference rows")
    reference_sq = (reference ** 2).sum(axis=1)
    out = np.empty((len(queries), k), dtype=np.int64)
    for start in range(0, len(queries), chunk):
        rows = queries[start:start + chunk]
        distances = (rows ** 2).sum(axis=1)[:, None] + reference_sq[None] - 2 * rows @ reference.T
        out[start:start + chunk] = np.argpartition(distances, k - 1, axis=1)[:, :k]
    return out
