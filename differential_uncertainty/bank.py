from __future__ import annotations

import tempfile
from collections.abc import Iterator
from numbers import Integral

import numpy as np
import torch
from torch import Tensor

from .artifacts import (
    _open_regular_file,
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
        if value.layout != torch.strided:
            raise ValueError(f"{name} must have a strided layout")
        if not value.is_floating_point():
            raise ValueError(f"{name} must have a floating-point dtype")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")


def _stage_clean_vectors(
    records,
    config: ExperimentConfig,
    spool,
) -> list[tuple[str, int, int]]:
    index: list[tuple[str, int, int]] = []
    image_ids: set[str] = set()
    query_ids = torch.arange(config.query_count)
    for record in records:
        _validate_record(record, config)
        image_id = str(record["image_id"])
        if image_id in image_ids:
            raise ValueError(
                "reference bank has a duplicate or string-colliding image ID"
            )
        image_ids.add(image_id)
        padded = detect_padded_tail(record)
        keep = torch.ones(config.query_count, dtype=torch.bool)
        keep[padded] = False
        persistence = record["persistence"]
        valid_ids = query_ids[keep].to(persistence.device)
        cleaned = (
            persistence.index_select(0, valid_ids)
            .detach()
            .cpu()
            .float()
            .contiguous()
        )
        if not bool(torch.isfinite(cleaned).all()):
            raise ValueError(
                "persistence must remain finite after float32 conversion"
            )
        payload = cleaned.numpy().tobytes(order="C")
        offset = spool.tell()
        if spool.write(payload) != len(payload):
            raise OSError("could not write the complete reference-bank spool record")
        index.append((image_id, offset, int(cleaned.shape[0])))
    return index


def _iter_staged_vectors(
    spool,
    index: list[tuple[str, int, int]],
    persistence_dim: int,
) -> Iterator[Tensor]:
    item_size = np.dtype(np.float32).itemsize
    for _image_id, offset, valid_count in sorted(
        index, key=lambda item: item[0]
    ):
        byte_count = valid_count * persistence_dim * item_size
        spool.seek(offset)
        payload = spool.read(byte_count)
        if len(payload) != byte_count:
            raise OSError("could not read the complete reference-bank spool record")
        values = np.frombuffer(payload, dtype=np.float32).copy()
        vectors = torch.from_numpy(
            values.reshape(valid_count, persistence_dim)
        )
        for vector in vectors:
            yield vector.clone()


def build_reference_bank(
    records, config: ExperimentConfig = FIXED_CONFIG
) -> Tensor:
    rng = np.random.default_rng(config.bank_seed)
    reservoir: list[Tensor] = []
    seen = 0
    with tempfile.TemporaryFile(mode="w+b") as spool:
        index = _stage_clean_vectors(records, config, spool)
        spool.flush()
        for seen, vector in enumerate(
            _iter_staged_vectors(spool, index, config.persistence_dim),
            start=1,
        ):
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
    if bank.layout != torch.strided:
        raise ValueError("reference bank vectors must have a strided layout")
    if bank.dtype != torch.float32:
        raise ValueError("reference bank vectors must have dtype float32")
    if bank.shape[0] == 0:
        raise ValueError("reference bank vectors need at least one row")
    if bank.shape[1] == 0:
        raise ValueError("reference bank vectors need at least one feature")
    if not bool(torch.isfinite(bank).all()):
        raise ValueError("reference bank vectors must be finite")


def _validated_metadata(metadata, bank: Tensor) -> dict:
    if type(metadata) is not dict:
        raise TypeError("reference bank metadata must be a plain dictionary")
    expected_keys = {
        "reference_manifest_sha256",
        "capacity",
        "seed",
        "padding_removed",
    }
    if set(metadata) != expected_keys:
        raise ValueError("reference bank metadata must have the exact metadata keys")
    digest = metadata["reference_manifest_sha256"]
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(
            "reference_manifest_sha256 must be lowercase 64-character hexadecimal"
        )
    capacity = metadata["capacity"]
    if (
        isinstance(capacity, bool)
        or not isinstance(capacity, Integral)
        or capacity <= 0
    ):
        raise ValueError("reference bank capacity must be a positive integer")
    if int(capacity) != bank.shape[0]:
        raise ValueError("reference bank capacity must equal the number of bank rows")
    seed = metadata["seed"]
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("reference bank seed must be an integer")
    if metadata["padding_removed"] is not True:
        raise ValueError("reference bank padding_removed must be true")
    return {
        "reference_manifest_sha256": digest,
        "capacity": int(capacity),
        "seed": int(seed),
        "padding_removed": True,
    }


def save_reference_bank(bank: Tensor, path, metadata: dict) -> None:
    _validate_bank_vectors(bank)
    validated_metadata = _validated_metadata(metadata, bank)
    atomic_torch(
        {
            "vectors": bank.detach().cpu().clone(),
            "metadata": validated_metadata,
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
    validated_metadata = _validated_metadata(metadata, vectors)
    return vectors.clone(), validated_metadata
