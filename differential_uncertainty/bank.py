from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral

import numpy as np
import torch
from torch import Tensor

from .artifacts import (
    _open_regular_file,
    _validate_safe_artifact_value,
    atomic_torch,
)
from .config import ExperimentConfig, FIXED_CONFIG
from .scoring import detect_padded_tail


def _validate_record(record, config: ExperimentConfig) -> None:
    severity = record.get("severity", 0)
    if (
        isinstance(severity, bool)
        or not isinstance(severity, Integral)
        or int(severity) != 0
    ):
        raise ValueError("reference bank records must be clean severity 0")
    expected_shapes = {
        "boxes": (config.query_count, 4),
        "logits": (config.query_count, config.class_count),
        "persistence": (config.query_count, config.persistence_dim),
    }
    for name, expected in expected_shapes.items():
        value = record[name]
        if not isinstance(value, Tensor) or tuple(value.shape) != expected:
            raise ValueError(f"unexpected {name} shape; expected {expected}")
        if not value.is_floating_point():
            raise ValueError(f"{name} must have a floating-point dtype")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")


def _valid_vectors(records, config: ExperimentConfig) -> Iterable[Tensor]:
    for record in sorted(records, key=lambda item: str(item["image_id"])):
        _validate_record(record, config)
        padded = set(detect_padded_tail(record).tolist())
        for query_id, vector in enumerate(record["persistence"]):
            if query_id not in padded:
                yield vector.detach().cpu().float().clone()


def build_reference_bank(
    records, config: ExperimentConfig = FIXED_CONFIG
) -> Tensor:
    rng = np.random.default_rng(config.bank_seed)
    reservoir: list[Tensor] = []
    seen = 0
    for seen, vector in enumerate(_valid_vectors(records, config), start=1):
        if len(reservoir) < config.bank_capacity:
            reservoir.append(vector)
        else:
            replacement = int(rng.integers(0, seen))
            if replacement < config.bank_capacity:
                reservoir[replacement] = vector
    if seen < config.bank_capacity:
        raise ValueError(
            f"reference bank needs exactly {config.bank_capacity} valid vectors; got {seen}"
        )
    return torch.stack(reservoir)


def _validate_bank_vectors(bank: Tensor) -> None:
    if not isinstance(bank, Tensor) or bank.ndim != 2:
        raise ValueError("reference bank vectors must be a two-dimensional tensor")
    if bank.dtype != torch.float32:
        raise ValueError("reference bank vectors must have dtype float32")
    if not bool(torch.isfinite(bank).all()):
        raise ValueError("reference bank vectors must be finite")


def _validate_metadata(metadata) -> None:
    if type(metadata) is not dict:
        raise TypeError("reference bank metadata must be a plain dictionary")
    _validate_safe_artifact_value(
        metadata,
        path="reference bank metadata",
    )


def save_reference_bank(bank: Tensor, path, metadata: dict) -> None:
    _validate_bank_vectors(bank)
    _validate_metadata(metadata)
    atomic_torch(
        {
            "vectors": bank.detach().cpu().clone(),
            "metadata": dict(metadata),
        },
        path,
    )


def load_reference_bank(path) -> tuple[Tensor, dict]:
    with _open_regular_file(
        path,
        error_message="reference bank must be a regular file",
    ) as handle:
        artifact = torch.load(handle, map_location="cpu", weights_only=True)
    if type(artifact) is not dict or set(artifact) != {"vectors", "metadata"}:
        raise ValueError(
            "reference bank artifact must contain exactly vectors and metadata"
        )
    vectors = artifact["vectors"]
    metadata = artifact["metadata"]
    _validate_bank_vectors(vectors)
    _validate_metadata(metadata)
    return vectors.clone(), dict(metadata)
