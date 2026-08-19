from __future__ import annotations

import torch
from torch import Tensor


def _robust_state(values: Tensor) -> tuple[Tensor, Tensor]:
    median = values.median(dim=0).values
    q25 = torch.quantile(values, 0.25, dim=0)
    q75 = torch.quantile(values, 0.75, dim=0)
    scale = (q75 - q25).clamp_min(1e-6)
    return median, scale


def fit_normalizer(bank: Tensor, mode: str) -> dict:
    bank = bank.float()
    if mode == "raw":
        return {"mode": mode}
    if mode == "robust_z":
        center, scale = _robust_state(bank)
        return {"mode": mode, "center": center, "scale": scale}
    if mode == "unit":
        return {"mode": mode}
    if mode == "shape_scale":
        magnitudes = bank.norm(dim=1, keepdim=True).log1p()
        center, scale = _robust_state(magnitudes)
        return {"mode": mode, "magnitude_center": center, "magnitude_scale": scale}
    raise ValueError(f"Unknown normalization mode: {mode}")


def transform_vectors(vectors: Tensor, state: dict) -> Tensor:
    vectors = vectors.float()
    mode = state["mode"]
    if mode == "raw":
        return vectors
    if mode == "robust_z":
        return (vectors - state["center"]) / state["scale"]
    norms = vectors.norm(dim=1, keepdim=True).clamp_min(1e-12)
    shape = vectors / norms
    if mode == "unit":
        return shape
    if mode == "shape_scale":
        magnitude = (norms.log1p() - state["magnitude_center"]) / state["magnitude_scale"]
        return torch.cat((shape, magnitude), dim=1)
    raise ValueError(f"Unknown normalization mode: {mode}")
