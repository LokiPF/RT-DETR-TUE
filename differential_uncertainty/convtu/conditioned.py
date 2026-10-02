"""Content-conditioned reference: judge the early stages' channels against clean images of similar scenes.

The back of the detector (stage 4) barely reacts to corruption but still describes the scene, so its channel means
pick the k most similar clean bank images. Each early stage is then scored against those neighbours, in two ways:
- level: the channel means (docs/dev-log.md, 2026-10-01 evening);
- peak share: log(mean of the strongest 1% of positions) - log(mean), which falls when fog, contrast or blur flatten
  a channel (docs/dev-log.md, 2026-10-01 night).
The two-axis score takes the larger of the flattening and the level score, so an image is flagged if its early
channels are either unusually flat or unusually shifted for a scene like it.
"""
from __future__ import annotations

import numpy as np

from degradation_monitor.evaluation.metrics import stage_zstats, zscored_sum
from .channels import fit_own_average

KEY_LAYER = "s4"
SCORED_LAYERS = ("s1", "s2", "s3")
NEIGHBOURS = 50
CHUNK = 2048
EPS = 1e-6  # keeps the logarithm finite for a channel that is silent on an image


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


def peak_share(values: dict, layer: str) -> np.ndarray:
    """Per channel, log(mean of its strongest 1% of positions) - log(its mean): how concentrated the response is."""
    top, means = _flat(values[f"top_{layer}"])[0], _flat(values[f"means_{layer}"])[0]
    return np.log(top + EPS) - np.log(means + EPS)


def _columns(values: dict, bank: dict, key: str, scored, k: int, kinds) -> dict:
    """Per-stage deviations from the k nearest clean scenes, one neighbour search for every kind requested.

    level: mean over channels of |m - mean_N m| / sd. shape: the same on the peak share. flatter: the signed
    -(share - mean_N share) / sd averaged over channels, positive when the channels are flatter than the neighbours'.
    """
    centre, spread = fit_own_average(bank[f"means_{key}"])
    bank_keys = (np.asarray(bank[f"means_{key}"], dtype=np.float64) - centre) / spread
    queries, _ = _flat(values[f"means_{key}"])
    level_reference = {layer: np.asarray(bank[f"means_{layer}"], dtype=np.float64) for layer in scored}
    level_spread = {layer: fit_own_average(level_reference[layer])[1] for layer in scored}
    level_values = {layer: _flat(values[f"means_{layer}"])[0] for layer in scored}
    shaped = any(kind != "level" for kind in kinds)
    if shaped:
        share_reference = {layer: peak_share(bank, layer) for layer in scored}
        share_spread = {layer: fit_own_average(share_reference[layer])[1] for layer in scored}
        share_values = {layer: peak_share(values, layer) for layer in scored}
    out = {kind: np.empty((len(queries), len(scored))) for kind in kinds}
    for start in range(0, len(queries), CHUNK):
        rows = slice(start, start + CHUNK)
        neighbours = nearest_rows((queries[rows] - centre) / spread, bank_keys, k)
        for column, layer in enumerate(scored):
            if "level" in kinds:
                out["level"][rows, column] = _deviation(level_values[layer][rows],
                                                        level_reference[layer][neighbours].mean(axis=1),
                                                        level_spread[layer])
            if shaped:
                deviation = ((share_values[layer][rows] - share_reference[layer][neighbours].mean(axis=1))
                             / share_spread[layer])
                if "shape" in kinds:
                    out["shape"][rows, column] = np.abs(deviation).mean(axis=1)
                if "flatter" in kinds:
                    out["flatter"][rows, column] = (-deviation).mean(axis=1)
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
    """The level score: z-scored sum over the scored stages and the per-stage scores, keeping test's leading shape.

    test, bank and zstats map f"means_{layer}" to channel means: test (..., C), the clean bank and the clean
    z-statistics images (rows, C). The z-statistics images are scored the same way, with neighbours from the bank.
    """
    return _combined(lambda values: _columns(values, bank, key, scored, k, ("level",))["level"], test, zstats, scored)


def peak_share_scores(test: dict, bank: dict, zstats: dict, key: str = KEY_LAYER, scored=SCORED_LAYERS,
                      k: int = NEIGHBOURS) -> tuple[np.ndarray, np.ndarray]:
    """The same on the peak share, which also needs f"top_{layer}" (the mean of each channel's strongest 1%)."""
    return _combined(lambda values: _columns(values, bank, key, scored, k, ("shape",))["shape"], test, zstats, scored)


def two_axis_scores(test: dict, bank: dict, zstats: dict, key: str = KEY_LAYER, scored=SCORED_LAYERS,
                    k: int = NEIGHBOURS) -> tuple[np.ndarray, dict]:
    """The larger of the flattening and the level score, each standardised on the z-statistics images.

    Returns the score and both arms, keeping test's leading shape.
    """
    leading = _flat(test[f"means_{scored[0]}"])[1]
    test_columns = _columns(test, bank, key, scored, k, ("level", "flatter"))
    clean_columns = _columns(zstats, bank, key, scored, k, ("level", "flatter"))
    arms = {}
    for kind in ("flatter", "level"):
        mean, std = stage_zstats(clean_columns[kind])
        clean = zscored_sum(clean_columns[kind], mean, std)
        if not clean.std() > 0:
            raise ValueError(f"the {kind} arm has no spread over the z-statistics images")
        arms[kind] = ((zscored_sum(test_columns[kind], mean, std) - clean.mean()) / clean.std()).reshape(leading)
    return np.maximum(arms["flatter"], arms["level"]), arms


def global_scores(test: dict, bank: dict, zstats: dict, scored=SCORED_LAYERS) -> tuple[np.ndarray, np.ndarray]:
    """The level score against the global clean mean: what the conditioning is compared with."""
    return _combined(lambda values: _global_columns(values, bank, scored), test, zstats, scored)
