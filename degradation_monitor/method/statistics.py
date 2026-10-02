"""Per-channel statistics of the four early backbone maps: the level (mean |x|) and the mean of the top 1% of |x|."""
from __future__ import annotations

import numpy as np
import torch

TOP_FRACTION = 0.01
STATISTICS = ("means", "top")
KEYS = tuple(f"{statistic}_s{stage}" for statistic in STATISTICS for stage in range(1, 5))  # one per stored array


@torch.inference_mode()
def channel_statistics(x: torch.Tensor) -> dict:
    """For a batch of maps (N, C, H, W): each channel's mean |x| and the mean of its largest 1% of |x|, float32 (N, C)."""
    if x.ndim != 4:
        raise ValueError("expected a batch of shape (N, C, H, W)")
    x_abs = x.abs().float()
    n, channel_count, height, width = x_abs.shape
    k = max(1, round(TOP_FRACTION * height * width))
    top = torch.topk(x_abs.reshape(n, channel_count, -1), k, dim=2).values
    out = {"means": x_abs.mean(dim=(2, 3)), "top": top.mean(dim=2)}
    return {key: value.cpu().numpy().astype(np.float32) for key, value in out.items()}
