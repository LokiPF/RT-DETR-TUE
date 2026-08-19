from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from src.core import YAMLConfig


def checkpoint_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_state(checkpoint: dict) -> dict[str, torch.Tensor]:
    if "ema" in checkpoint:
        return checkpoint["ema"]["module"]
    if "model" in checkpoint:
        return checkpoint["model"]
    if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint
    raise KeyError("Checkpoint has neither ema.module nor model state")


def load_frozen_detector(
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: torch.device,
) -> nn.Module:
    cfg = YAMLConfig(str(config_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = _checkpoint_state(checkpoint)
    incompatible = cfg.model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model = cfg.model.to(device).eval()
    model.requires_grad_(False)
    return model
