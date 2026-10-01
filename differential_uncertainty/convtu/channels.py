"""Per-channel statistics of the four pilot conv inputs, and the two ways of comparing them with clean images.

means: each channel's mean |x|. top: the mean of each channel's largest 1% of |x|. p99: the smallest of those,
the nearest-rank 99th percentile. grid: each channel's mean |x| on a 4 x 4 grid of the map. kNN compares an
image's vector with its nearest clean images; "own" compares every dimension with its own clean average, in
units of its clean spread, the way the activation-distribution monitor compares every channel with itself.
"""
from __future__ import annotations

import numpy as np
import torch

from ..baselines.activation_cdf import stage_zstats, zscored_sum
from .features import knn_scores

STATISTICS = ("means", "top", "p99", "grid")
COMPARISONS = ("knn", "own")
TOP_FRACTION = 0.01
GRID = 4
STD_FLOOR = 0.01  # every dimension's spread is at least 1% of the median positive spread of its layer


@torch.inference_mode()
def channel_statistics(x: torch.Tensor) -> dict:
    """For a batch of conv inputs (N, C, H, W): every statistic as a float32 (N, dim) array."""
    if x.ndim != 4:
        raise ValueError("expected a batch of shape (N, C, H, W)")
    x_abs = x.abs().float()
    n, channel_count, height, width = x_abs.shape
    k = max(1, round(TOP_FRACTION * height * width))
    top = torch.topk(x_abs.reshape(n, channel_count, -1), k, dim=2).values
    grid = torch.nn.functional.adaptive_avg_pool2d(x_abs, GRID).reshape(n, channel_count * GRID * GRID)
    out = {"means": x_abs.mean(dim=(2, 3)), "top": top.mean(dim=2), "p99": top[:, :, -1], "grid": grid}
    return {key: value.cpu().numpy().astype(np.float32) for key, value in out.items()}


@torch.inference_mode()
def channel_means(x: torch.Tensor) -> dict:
    """Only the means statistic of channel_statistics, computed the same way; small enough for 5,000 images."""
    if x.ndim != 4:
        raise ValueError("expected a batch of shape (N, C, H, W)")
    return {"means": x.abs().float().mean(dim=(2, 3)).cpu().numpy().astype(np.float32)}


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


def floored_mask(reference: np.ndarray) -> np.ndarray:
    """True where a dimension's clean spread is below the floor, so fit_own_average floors it."""
    std = np.asarray(reference, dtype=np.float64).std(axis=0)
    positive = std[std > 0]
    if positive.size == 0:
        raise ValueError("the clean rows have no spread")
    return std < STD_FLOOR * float(np.median(positive))


def floored_counts(bank: dict, layers) -> dict:
    """How many dimensions of each statistic and layer the floor touches."""
    return {f"{statistic}_{layer}": int(floored_mask(bank[f"{statistic}_{layer}"]).sum())
            for statistic in STATISTICS for layer in layers}


def own_average_scores(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Mean over dimensions of |value - clean mean| / clean spread: one score per row."""
    return (np.abs(np.asarray(values, dtype=np.float64) - mean) / std).mean(axis=1)


LABELS = {
    "ch_means_knn": "Channel means, kNN (check: the pilot's control)",
    "ch_means_own": "Channel means vs own training average",
    "ch_top_knn": "Per-channel top-1% means, kNN",
    "ch_top_own": "Per-channel top-1% means vs own training average",
    "ch_p99_knn": "Per-channel 99th percentiles, kNN",
    "ch_p99_own": "Per-channel 99th percentiles vs own training average",
    "ch_grid_knn": "4 × 4-grid channel means, kNN",
    "ch_grid_own": "4 × 4-grid channel means vs own training average",
}


def method_name(statistic: str, comparison: str) -> str:
    return f"ch_{statistic}_{comparison}"


def _float32(array) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


def channel_method_scores(bank: dict, zstats: dict, test: dict, layers, drop_floored: bool = False) -> tuple[dict, dict]:
    """Every statistic compared both ways: z-summed (images, conditions) scores and the per-layer ones.

    bank and zstats map f"{statistic}_{layer}" to clean (rows, dim) arrays; test maps it to
    (images, conditions, dim). Per-layer scores are z-scored with the z-statistics images and summed over
    the layers, as in the pilot. With drop_floored, the own-average comparison leaves out the dimensions
    whose clean spread the floor touches, so they cannot dominate it.
    """
    summed, per_layer = {}, {}
    for statistic in STATISTICS:
        for comparison in COMPARISONS:
            clean_columns, test_columns = [], []
            for layer in layers:
                key = f"{statistic}_{layer}"
                images, conditions, dim = test[key].shape
                flat = test[key].reshape(images * conditions, dim)
                if comparison == "knn":
                    reference = _float32(bank[key])
                    clean_columns.append(knn_scores(_float32(zstats[key]), reference))
                    test_columns.append(knn_scores(_float32(flat), reference))
                else:
                    mean, std = fit_own_average(bank[key])
                    keep = ~floored_mask(bank[key]) if drop_floored else np.ones(dim, dtype=bool)
                    clean_columns.append(own_average_scores(zstats[key][:, keep], mean[keep], std[keep]))
                    test_columns.append(own_average_scores(flat[:, keep], mean[keep], std[keep]))
            mean, std = stage_zstats(np.stack(clean_columns, axis=1))
            values = np.stack(test_columns, axis=1)
            name = method_name(statistic, comparison)
            per_layer[name] = values.reshape(images, conditions, len(layers))
            summed[name] = zscored_sum(values, mean, std).reshape(images, conditions)
    return summed, per_layer
