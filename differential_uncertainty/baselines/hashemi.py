"""Gaussian neuron-interval runtime monitor (Hashemi, Křetínský, Rieder & Schmidt, FM 2023).

Every monitored neuron gets the interval mu +- k*sigma from clean training images, with class
information discarded (their Sec. 3.1). An image's score is the share of monitored neurons outside
their interval (their Eq. 6). Their conformal p-value is a decreasing step function of this score,
so threshold-free metrics use the score directly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

K = 2.0  # "k is a value close to 2" (their Sec. 2.2)
LAYERS = ("decoder", "encoder_s8", "encoder_s16", "encoder_s32")


class NeuronStats:
    """Exact streaming mean and population standard deviation per neuron (Chan et al.), in float64."""

    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None

    def update(self, batch: torch.Tensor) -> None:
        batch = batch.detach().to(torch.float64)
        n = batch.shape[0]
        if n == 0:
            return
        if self.mean is not None and tuple(batch.shape[1:]) != tuple(self.mean.shape):
            raise ValueError(f"neuron shape changed from {tuple(self.mean.shape)} to {tuple(batch.shape[1:])}")
        batch_mean = batch.mean(dim=0)
        batch_m2 = ((batch - batch_mean) ** 2).sum(dim=0)
        if self.mean is None:
            self.count, self.mean, self.m2 = n, batch_mean, batch_m2
            return
        total = self.count + n
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (n / total)
        self.m2 = self.m2 + batch_m2 + delta ** 2 * (self.count * n / total)
        self.count = total

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count < 2:
            raise ValueError("need at least two images to estimate a standard deviation")
        return self.mean.float().cpu().numpy(), (self.m2 / self.count).sqrt().float().cpu().numpy()


def outside_counts(values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, k: float = K) -> torch.Tensor:
    """Neurons per image with |h - mu| > k*sigma; strict, so a constant neuron at its mean stays inside."""
    if tuple(values.shape[1:]) != tuple(mean.shape):
        raise ValueError(f"activations {tuple(values.shape[1:])} do not match the fitted neurons {tuple(mean.shape)}")
    outside = (values.float() - mean).abs() > k * std
    return outside.flatten(1).sum(dim=1)


def save_intervals(path, stats: dict, images: int) -> None:
    arrays = {}
    for name in LAYERS:
        arrays[f"{name}_mean"], arrays[f"{name}_std"] = stats[name].result()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, images=np.array(images), **arrays)
    temporary.replace(path)


class HashemiMonitor:
    def __init__(self, path, device, k: float = K):
        with np.load(path, allow_pickle=False) as data:
            self.stats = {name: (torch.from_numpy(data[f"{name}_mean"]).to(device),
                                 torch.from_numpy(data[f"{name}_std"]).to(device)) for name in LAYERS}
        self.k = k

    def decoder_share(self, decoder: torch.Tensor) -> np.ndarray:
        mean, std = self.stats["decoder"]
        return (outside_counts(decoder, mean, std, self.k).double() / mean.numel()).cpu().numpy()

    def scores(self, decoder: torch.Tensor, encoder) -> tuple[np.ndarray, np.ndarray]:
        """(decoder share, encoder share) per image; the encoder share pools all three maps' neurons."""
        if len(encoder) != len(LAYERS) - 1:
            raise ValueError(f"expected the encoder's three output maps, got {len(encoder)}")
        counts, total = None, 0
        for name, values in zip(LAYERS[1:], encoder):
            mean, std = self.stats[name]
            part = outside_counts(values, mean, std, self.k).double()
            counts = part if counts is None else counts + part
            total += mean.numel()
        return self.decoder_share(decoder), (counts / total).cpu().numpy()
