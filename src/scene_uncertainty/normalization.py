from __future__ import annotations

import torch
from torch import Tensor


_ABSOLUTE_FLOOR = 1e-6
_RELATIVE_FLOOR = 1e-3


def _spread_floor(spread: Tensor, center: Tensor) -> float:
    """The smallest spread a feature may claim, relative to how much its neighbours move.

    A feature whose bank inter-quartile range is zero has no measured scale of its own,
    and `q75 == q25` needs only half the bank at one value, not all of it. Dividing such
    a feature by a bare `1e-6` would score any evaluation-time movement in it a million
    times higher than the same movement in a well-spread feature, so it would be the only
    feature a distance ever sees. Falling back to a fraction of the typical spread keeps
    that feature sensitive -- it may well be where corruption shows up first -- without
    letting it drown out the rest.
    """
    positive = spread[spread > 0]
    reference = positive.median() if positive.numel() else center.abs().median()
    return max(_ABSOLUTE_FLOOR, _RELATIVE_FLOOR * float(reference))


def _robust_state(values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return the per-feature median, the floored spread, and which features were floored."""
    median = values.median(dim=0).values
    q25 = torch.quantile(values, 0.25, dim=0)
    q75 = torch.quantile(values, 0.75, dim=0)
    spread = q75 - q25
    floor = _spread_floor(spread, median)
    return median, spread.clamp_min(floor), spread < floor


def fit_normalizer(bank: Tensor, mode: str) -> dict:
    bank = bank.float()
    if mode == "raw":
        return {"mode": mode}
    if mode == "robust_z":
        center, scale, zero_spread = _robust_state(bank)
        return {
            "mode": mode,
            "center": center,
            "scale": scale,
            "zero_spread": zero_spread,
            "zero_spread_count": int(zero_spread.sum()),
        }
    if mode == "unit":
        return {"mode": mode}
    if mode == "shape_scale":
        magnitudes = bank.norm(dim=1, keepdim=True).log1p()
        center, scale, zero_spread = _robust_state(magnitudes)
        return {
            "mode": mode,
            "magnitude_center": center,
            "magnitude_scale": scale,
            "magnitude_zero_spread": zero_spread,
            "magnitude_zero_spread_count": int(zero_spread.sum()),
        }
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
