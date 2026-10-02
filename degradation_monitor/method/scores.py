"""Our scores: level, peak share and the two-axis score, all judged against similar clean scenes, plus three ablation rows.

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
import torch

from ..evaluation.metrics import stage_zstats, zscored_sum
from .reference import CHUNK, KEY_LAYER, NEIGHBOURS, SCORED_LAYERS, fit_own_average, nearest_rows

EPS = 1e-6  # keeps the logarithm finite for a channel that is silent on an image
KNN_NEIGHBOURS = 5


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


def level_scores(test: dict, bank: dict, zstats: dict, key: str = KEY_LAYER, scored=SCORED_LAYERS,
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


def global_level_scores(test: dict, bank: dict, zstats: dict, scored=SCORED_LAYERS) -> tuple[np.ndarray, np.ndarray]:
    """The level score against the global clean mean: what the conditioning is compared with."""
    return _combined(lambda values: _global_columns(values, bank, scored), test, zstats, scored)


def knn_scores(queries: torch.Tensor, bank: torch.Tensor, neighbours: int = KNN_NEIGHBOURS,
               chunk: int = 64) -> np.ndarray:
    """Mean Euclidean distance from each query row to its nearest `neighbours` bank rows (no normalization)."""
    if queries.ndim != 2 or bank.ndim != 2 or queries.shape[1] != bank.shape[1]:
        raise ValueError("queries and bank must be (rows, dim) with the same dim")
    if not 1 <= neighbours <= bank.shape[0]:
        raise ValueError("neighbours must be between 1 and the bank size")
    bank = bank.float()
    out = []
    for start in range(0, queries.shape[0], chunk):
        rows = queries[start:start + chunk].to(bank.device, torch.float32)
        distances = torch.cdist(rows, bank, compute_mode="donot_use_mm_for_euclid_dist")
        out.append(distances.topk(neighbours, dim=1, largest=False).values.mean(dim=1))
    return torch.cat(out).cpu().numpy().astype(np.float64)


def own_average_scores(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Mean over dimensions of |value - clean mean| / clean spread: one score per row."""
    return (np.abs(np.asarray(values, dtype=np.float64) - mean) / std).mean(axis=1)


FOUR_LAYERS = ("s1", "s2", "s3", "s4")


def _float32(array) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


def _four_stage_rows(column_of, test: dict, zstats: dict) -> tuple[np.ndarray, np.ndarray]:
    leading = _flat(test["means_s1"])[1]
    clean = np.stack([column_of(layer, zstats[f"means_{layer}"]) for layer in FOUR_LAYERS], axis=1)
    mean, std = stage_zstats(clean)
    values = np.stack([column_of(layer, test[f"means_{layer}"].reshape(-1, test[f"means_{layer}"].shape[-1]))
                       for layer in FOUR_LAYERS], axis=1)
    return zscored_sum(values, mean, std).reshape(leading), values.reshape(*leading, len(FOUR_LAYERS))


def means_knn_scores(test: dict, bank: dict, zstats: dict) -> tuple[np.ndarray, np.ndarray]:
    """Ablation: kNN (5 nearest, unnormalised) on the channel means of all four stages; the pilot's control row."""
    def column(layer, values):
        return knn_scores(_float32(values), _float32(bank[f"means_{layer}"]))
    return _four_stage_rows(column, test, zstats)


def means_own_scores(test: dict, bank: dict, zstats: dict) -> tuple[np.ndarray, np.ndarray]:
    """Ablation: each channel mean against its own clean average, all four stages (Neural Mean Discrepancy-style)."""
    def column(layer, values):
        mean, std = fit_own_average(bank[f"means_{layer}"])
        return own_average_scores(values, mean, std)
    return _four_stage_rows(column, test, zstats)
