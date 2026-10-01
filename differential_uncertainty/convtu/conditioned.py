"""Content-conditioned reference: judge the early stages' channel means against clean images of similar scenes.

The back of the detector (stage 4) barely reacts to corruption but still describes the scene, so its channel means
pick the k most similar clean bank images. Each early stage is then scored by its standardised deviation from those
neighbours' channel means instead of from the global clean mean. See docs/dev-log.md (2026-10-01, evening).
"""
from __future__ import annotations

import numpy as np

from ..baselines.activation_cdf import stage_zstats, zscored_sum
from .channels import fit_own_average

KEY_LAYER = "s4"
SCORED_LAYERS = ("s1", "s2", "s3")
NEIGHBOURS = 50
CHUNK = 2048


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


def _flat(values) -> tuple[np.ndarray, tuple]:
    values = np.asarray(values, dtype=np.float64)
    return values.reshape(-1, values.shape[-1]), values.shape[:-1]


def _deviation(values: np.ndarray, reference_mean: np.ndarray, spread: np.ndarray) -> np.ndarray:
    return (np.abs(values - reference_mean) / spread).mean(axis=1)


def _conditioned_columns(values: dict, bank: dict, key: str, scored, k: int) -> np.ndarray:
    centre, spread = fit_own_average(bank[f"means_{key}"])
    bank_keys = (np.asarray(bank[f"means_{key}"], dtype=np.float64) - centre) / spread
    queries, _ = _flat(values[f"means_{key}"])
    references = {layer: np.asarray(bank[f"means_{layer}"], dtype=np.float64) for layer in scored}
    spreads = {layer: fit_own_average(references[layer])[1] for layer in scored}
    flat = {layer: _flat(values[f"means_{layer}"])[0] for layer in scored}
    out = np.empty((len(queries), len(scored)))
    for start in range(0, len(queries), CHUNK):
        neighbours = nearest_rows((queries[start:start + CHUNK] - centre) / spread, bank_keys, k)
        for column, layer in enumerate(scored):
            local_mean = references[layer][neighbours].mean(axis=1)
            out[start:start + CHUNK, column] = _deviation(flat[layer][start:start + CHUNK], local_mean, spreads[layer])
    return out


def _global_columns(values: dict, bank: dict, scored) -> np.ndarray:
    columns = []
    for layer in scored:
        mean, spread = fit_own_average(bank[f"means_{layer}"])
        columns.append(_deviation(_flat(values[f"means_{layer}"])[0], mean, spread))
    return np.stack(columns, axis=1)


def _combined(columns_of, test: dict, zstats: dict, scored) -> tuple[np.ndarray, np.ndarray]:
    leading = _flat(test[f"means_{scored[0]}"])[1]
    mean, std = stage_zstats(columns_of(zstats))
    values = columns_of(test)
    return zscored_sum(values, mean, std).reshape(leading), values.reshape(*leading, len(scored))


def conditioned_scores(test: dict, bank: dict, zstats: dict, key: str = KEY_LAYER, scored=SCORED_LAYERS,
                       k: int = NEIGHBOURS) -> tuple[np.ndarray, np.ndarray]:
    """The z-scored sum over the scored stages and the per-stage scores, keeping test's leading shape.

    test, bank and zstats map f"means_{layer}" to channel means: test (..., C), the clean bank and the clean
    z-statistics images (rows, C). The z-statistics images are scored the same way, with neighbours from the bank.
    """
    return _combined(lambda values: _conditioned_columns(values, bank, key, scored, k), test, zstats, scored)


def global_scores(test: dict, bank: dict, zstats: dict, scored=SCORED_LAYERS) -> tuple[np.ndarray, np.ndarray]:
    """The same score against the global clean mean: what the conditioning is compared with."""
    return _combined(lambda values: _global_columns(values, bank, scored), test, zstats, scored)
