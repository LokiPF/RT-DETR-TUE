"""Activation-distribution monitor, reproduced from Becker, Bayer, Hübner & Arens (ICPR 2026).

Each channel of each monitored layer is summarised, per image, by a histogram of its activations over all
spatial positions. The histogram has 1,000 bins on a fixed per-channel range: the training minimum and
maximum, widened by 20% of their span. It becomes a CDF and is compared with the same channel's CDF pooled
over all training images, using the Earth Mover's distance in units of the channel's range. Two image scores
are kept: the plain sum over all channels and layers, and the sum over layers of each layer's channel sum
z-scored with clean-image statistics (their "z-score normalization"). Without the z-scoring, the 512
channels of C5 dominate the plain sum.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

BINS = 1000
MARGIN = 0.2
STAGES = ("C1", "C2", "C3", "C4", "C5")


class ChannelRanges:
    """Streaming per-channel minimum and maximum over images and positions."""

    def __init__(self):
        self.low = None
        self.high = None

    def update(self, values: torch.Tensor) -> None:
        low = values.detach().amin(dim=(0, 2, 3)).double()
        high = values.detach().amax(dim=(0, 2, 3)).double()
        if self.low is None:
            self.low, self.high = low, high
            return
        if low.shape != self.low.shape:
            raise ValueError(f"channel count changed from {self.low.numel()} to {low.numel()}")
        self.low, self.high = torch.minimum(self.low, low), torch.maximum(self.high, high)

    def result(self, margin: float = MARGIN) -> tuple[np.ndarray, np.ndarray]:
        if self.low is None:
            raise ValueError("no activations were seen")
        span = self.high - self.low
        span = torch.where(span > 0, span, torch.ones_like(span))  # a constant channel gets span 1
        return (self.low - margin * span).cpu().numpy(), (self.high + margin * span).cpu().numpy()


def channel_histograms(values: torch.Tensor, low: torch.Tensor, high: torch.Tensor, bins: int = BINS) -> torch.Tensor:
    """Per image and channel, counts in `bins` equal bins on [low, high]; values outside go to the edge bins."""
    n, channels = values.shape[:2]
    if low.numel() != channels:
        raise ValueError(f"activations have {channels} channels, the fit has {low.numel()}")
    flat = values.detach().float().flatten(2)
    low = low.float().view(1, channels, 1)
    width = high.float().view(1, channels, 1) - low
    index = ((flat - low) / width * bins).floor_().clamp_(0, bins - 1).long()
    counts = torch.zeros(n, channels, bins, device=values.device)
    counts.scatter_add_(2, index, torch.ones_like(flat))
    return counts


def emd_to_reference(counts: torch.Tensor, reference_cdf: torch.Tensor) -> torch.Tensor:
    """Per image, the sum over channels of the EMD between its CDF and the reference CDF, in range units."""
    cdf = counts.cumsum(dim=2) / counts.sum(dim=2, keepdim=True)
    return (cdf - reference_cdf).abs().mean(dim=2).sum(dim=1)


class ReferenceHistograms:
    """Training-set histograms per stage and channel, on fixed ranges."""

    def __init__(self, bounds: dict, device, bins: int = BINS):
        self.bins = bins
        self.low = {s: torch.as_tensor(bounds[s][0], device=device) for s in STAGES}
        self.high = {s: torch.as_tensor(bounds[s][1], device=device) for s in STAGES}
        self.counts = {s: torch.zeros(self.low[s].numel(), bins, dtype=torch.float64, device=device) for s in STAGES}

    def update(self, stages) -> None:
        if len(stages) != len(STAGES):
            raise ValueError(f"expected the five backbone stages, got {len(stages)}")
        for stage, values in zip(STAGES, stages):
            self.counts[stage] += channel_histograms(values, self.low[stage], self.high[stage], self.bins).sum(dim=0).double()

    def save(self, path, images: int) -> None:
        arrays = {}
        for stage in STAGES:
            counts = self.counts[stage]
            if bool((counts.sum(dim=1) == 0).any()):
                raise ValueError(f"no training activations were counted for {stage}")
            arrays[f"{stage}_low"] = self.low[stage].double().cpu().numpy()
            arrays[f"{stage}_high"] = self.high[stage].double().cpu().numpy()
            arrays[f"{stage}_cdf"] = (counts.cumsum(dim=1) / counts.sum(dim=1, keepdim=True)).cpu().numpy()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, images=np.array(images), bins=np.array(self.bins), margin=np.array(MARGIN), **arrays)
        temporary.replace(path)


class CdfMonitor:
    def __init__(self, path, device):
        with np.load(path, allow_pickle=False) as data:
            self.bins = int(data["bins"])
            self.low = {s: torch.from_numpy(data[f"{s}_low"]).float().to(device) for s in STAGES}
            self.high = {s: torch.from_numpy(data[f"{s}_high"]).float().to(device) for s in STAGES}
            self.cdf = {s: torch.from_numpy(data[f"{s}_cdf"]).float().to(device) for s in STAGES}

    def stage_scores(self, stages) -> np.ndarray:
        """Per image and stage, the sum over the stage's channels of the EMD to the reference: (n, 5)."""
        if len(stages) != len(STAGES):
            raise ValueError(f"expected the five backbone stages, got {len(stages)}")
        parts = []
        for stage, values in zip(STAGES, stages):
            counts = channel_histograms(values, self.low[stage], self.high[stage], self.bins)
            parts.append(emd_to_reference(counts, self.cdf[stage]).double())
        return torch.stack(parts, dim=1).cpu().numpy()

    def scores(self, stages) -> np.ndarray:
        """The plain sum over all channels of all stages."""
        return self.stage_scores(stages).sum(axis=1)
