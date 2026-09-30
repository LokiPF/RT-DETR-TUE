"""Per-layer fingerprint of the conv graph, three simpler summaries of the same layer, and their kNN scores.

mst: the K largest diagram values (the method). edges: the K heaviest edge weights, the same numbers without
the cycle rule. acts: the K largest activations, no kernel and no graph (zero-padded when K exceeds the input
cells). means: the mean absolute activation of each input channel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .graph import EDGE_CAP, conv_top_merges, fingerprint_length

REPRESENTATIONS = ("mst", "edges", "acts", "means")
KNN_NEIGHBOURS = 5


@dataclass(frozen=True)
class LayerSpec:
    stage: int
    c_in: int
    c_out: int
    height: int
    width: int

    @property
    def nodes(self) -> int:
        return (self.c_in + self.c_out) * self.height * self.width

    @property
    def k(self) -> int:
        return fingerprint_length(self.nodes)

    @property
    def name(self) -> str:
        return f"s{self.stage}"


def layer_specs(inputs: list[torch.Tensor], kernels: list[torch.Tensor]) -> list[LayerSpec]:
    """One spec per captured conv input of shape (N, C, H, W) and its (C_out, C, 3, 3) kernel."""
    return [LayerSpec(stage + 1, x.shape[1], kernel.shape[0], x.shape[2], x.shape[3])
            for stage, (x, kernel) in enumerate(zip(inputs, kernels))]


def layer_features(x: torch.Tensor, kernel_abs: torch.Tensor, spec: LayerSpec, cut: float,
                   cap: int = EDGE_CAP) -> dict:
    """The fingerprint and the three controls of one image's conv input x, shape (C, H, W)."""
    x_abs = x.abs()
    top = conv_top_merges(x_abs, kernel_abs, spec.k, cut, cap)
    count = min(spec.k, x_abs.numel())
    acts = np.zeros(spec.k, dtype=np.float32)
    acts[:count] = torch.topk(x_abs.flatten(), count).values.cpu().numpy()
    return {"mst": top.values, "edges": top.heaviest.astype(np.float32), "acts": acts,
            "means": x_abs.mean(dim=(1, 2)).cpu().numpy().astype(np.float32),
            "tau_k": float(top.values[-1]), "read": int(top.read), "gathered": int(top.gathered),
            "rounds": int(top.rounds)}


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
