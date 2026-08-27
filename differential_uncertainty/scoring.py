from __future__ import annotations

import math
from numbers import Integral

import torch
from torch import Tensor

from .config import ExperimentConfig, FIXED_CONFIG


def _require_floating_finite(values: Tensor, name: str) -> None:
    if not isinstance(values, Tensor):
        raise ValueError(f"{name} must be a tensor")
    if not values.is_floating_point():
        raise ValueError(f"{name} must have a floating-point dtype")
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must be finite")


def _equal_to_last(values: Tensor) -> Tensor:
    if values.ndim < 2:
        raise ValueError("query fields need query and feature dimensions")
    return values.eq(values[-1]).reshape(values.shape[0], -1).all(dim=1)


def detect_padded_tail(record: dict) -> Tensor:
    fields = [record["boxes"], record["logits"], record["persistence"]]
    for field in fields:
        _require_floating_finite(field, "query fields")
        if field.ndim < 2:
            raise ValueError("query fields need query and feature dimensions")
    counts = {int(field.shape[0]) for field in fields}
    if len(counts) != 1:
        raise ValueError("query-count mismatch inside extraction record")
    count = counts.pop()
    if count == 0:
        return torch.empty(0, dtype=torch.long)
    repeated = torch.stack([_equal_to_last(field.cpu()) for field in fields]).all(dim=0)
    start = count - 1
    while start > 0 and bool(repeated[start - 1]):
        start -= 1
    if count - start < 2:
        return torch.empty(0, dtype=torch.long)
    return torch.arange(start, count, dtype=torch.long)


def union_padded_query_ids(records) -> Tensor:
    records = list(records)
    if not records:
        raise ValueError("cannot build a padding union from zero records")
    expected = {str(record["image_id"]) for record in records}
    if len(expected) != 1:
        raise ValueError("padding union needs records from exactly one image")
    counts = {int(record["logits"].shape[0]) for record in records}
    if len(counts) != 1:
        raise ValueError("query-count mismatch across severities")
    masks = [detect_padded_tail(record) for record in records]
    return torch.cat(masks).unique() if masks else torch.empty(0, dtype=torch.long)


def confidence_from_logits(logits: Tensor) -> Tensor:
    if logits.ndim != 2:
        raise ValueError("logits must have shape query,class")
    if logits.shape[1] == 0:
        raise ValueError("logits must contain at least one class")
    _require_floating_finite(logits, "logits")
    return logits.float().sigmoid().amax(dim=-1).cpu()


def confidence_deciles(confidence: Tensor, valid_ids: Tensor) -> tuple[Tensor, ...]:
    _require_floating_finite(confidence, "confidence")
    if confidence.ndim != 1:
        raise ValueError("confidence must have one dimension")
    if not isinstance(valid_ids, Tensor):
        raise ValueError("valid query IDs must be a tensor")
    if valid_ids.ndim != 1:
        raise ValueError("valid query IDs must have one dimension")
    if valid_ids.dtype in (torch.bool, torch.uint8) or valid_ids.is_floating_point() or valid_ids.is_complex():
        raise ValueError("valid query IDs must have an integer query IDs dtype")
    scores = confidence.float().cpu()
    valid = torch.sort(valid_ids.long().cpu()).values
    if valid.numel() < 10:
        raise ValueError(f"ten confidence deciles need at least ten valid queries, got {valid.numel()}")
    if valid.unique().numel() != valid.numel():
        raise ValueError("valid query IDs must be unique")
    if int(valid.min()) < 0 or int(valid.max()) >= scores.numel():
        raise ValueError("valid query ID lies outside the confidence vector")
    selected = scores.index_select(0, valid)
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("confidence must be finite")
    order = torch.argsort(selected, stable=True)
    return tuple(torch.tensor_split(valid.index_select(0, order), 10))


def _euclidean_distances(queries: Tensor, bank: Tensor) -> Tensor:
    return torch.cdist(
        queries,
        bank,
        p=2.0,
        compute_mode="donot_use_mm_for_euclid_dist",
    )


def mean_knn_distance(
    queries: Tensor, bank: Tensor, k: int, bank_chunk_size: int = 8_192
) -> Tensor:
    if not isinstance(queries, Tensor) or not isinstance(bank, Tensor):
        raise ValueError("queries and bank must be tensors")
    if queries.ndim != 2 or bank.ndim != 2:
        raise ValueError("queries and bank must be two-dimensional")
    _require_floating_finite(queries, "queries")
    _require_floating_finite(bank, "bank")
    if queries.shape[1] != bank.shape[1]:
        raise ValueError("queries and bank must have matching feature dimensions")
    if isinstance(k, bool) or not isinstance(k, Integral):
        raise ValueError("k must be an integer")
    if isinstance(bank_chunk_size, bool) or not isinstance(bank_chunk_size, Integral):
        raise ValueError("bank chunk size must be an integer")
    k = int(k)
    bank_chunk_size = int(bank_chunk_size)
    if bank_chunk_size <= 0:
        raise ValueError("bank chunk size must be positive")
    queries = queries.float()
    bank = bank.float().to(queries.device)
    if k <= 0 or k > bank.shape[0]:
        raise ValueError(f"k must lie in [1, {bank.shape[0]}]")
    best = torch.full((queries.shape[0], k), float("inf"), device=queries.device)
    for chunk in bank.split(bank_chunk_size):
        local = _euclidean_distances(queries, chunk)
        local = local.topk(min(k, chunk.shape[0]), dim=1, largest=False).values
        best = torch.cat((best, local), dim=1).topk(k, dim=1, largest=False).values
    return best.mean(dim=1)


def relative_gap(reference: float, responsive: float) -> float:
    reference, responsive = float(reference), float(responsive)
    if not math.isfinite(reference) or not math.isfinite(responsive):
        raise ValueError("relative-gap inputs must be finite")
    if reference < 0 or responsive < 0:
        raise ValueError("relative-gap inputs must be non-negative")
    total = reference + responsive
    return 0.0 if total == 0.0 else 2.0 * (responsive - reference) / total


def score_image_records(
    records, bank: Tensor, config: ExperimentConfig = FIXED_CONFIG
) -> list[dict]:
    expected_bank_shape = (config.bank_capacity, config.persistence_dim)
    if not isinstance(bank, Tensor) or tuple(bank.shape) != expected_bank_shape:
        raise ValueError(f"bank must have shape {expected_bank_shape}")
    records = list(records)
    severities = [item["severity"] for item in records]
    if any(
        isinstance(severity, bool) or not isinstance(severity, Integral)
        for severity in severities
    ):
        raise ValueError("severity values must be integers and not booleans")
    records = sorted(records, key=lambda item: item["severity"])
    if [int(item["severity"]) for item in records] != list(range(6)):
        raise ValueError("each image must have exactly severities 0 through 5")
    image_ids = {str(item["image_id"]) for item in records}
    if len(image_ids) != 1:
        raise ValueError("score_image_records needs exactly one image")
    expected_shapes = {
        "boxes": (config.query_count, 4),
        "logits": (config.query_count, config.class_count),
        "persistence": (config.query_count, config.persistence_dim),
    }
    for record in records:
        for name, expected in expected_shapes.items():
            value = record[name]
            if not isinstance(value, Tensor) or tuple(value.shape) != expected:
                raise ValueError(f"unexpected {name} shape; expected {expected}")
    padded = union_padded_query_ids(records)
    all_ids = torch.arange(config.query_count)
    keep = torch.ones(config.query_count, dtype=torch.bool)
    keep[padded] = False
    valid = all_ids[keep]
    if valid.numel() < 10:
        raise ValueError("padding leaves too few valid queries to form ten deciles")
    rows: list[dict] = []
    for record in records:
        confidence = confidence_from_logits(record["logits"])
        direct_confidence = confidence.index_select(0, valid)
        direct_confidence_mean = float(direct_confidence.mean())
        direct_confidence_max = float(direct_confidence.max())
        bins = confidence_deciles(confidence, valid)
        reference_ids = bins[config.reference_decile]
        responsive_ids = bins[config.responsive_decile]
        persistence = record["persistence"].float()
        reference_distance = float(mean_knn_distance(
            persistence.index_select(0, reference_ids), bank, config.k, config.bank_chunk_size
        ).mean())
        responsive_distance = float(mean_knn_distance(
            persistence.index_select(0, responsive_ids), bank, config.k, config.bank_chunk_size
        ).mean())
        uncertainty = 1.0 - confidence
        reference_uncertainty = float(uncertainty.index_select(0, reference_ids).mean())
        responsive_uncertainty = float(uncertainty.index_select(0, responsive_ids).mean())
        rows.append({
            "image_id": next(iter(image_ids)),
            "severity": int(record["severity"]),
            "padded_count": int(padded.numel()),
            "valid_count": int(valid.numel()),
            "reference_count": int(reference_ids.numel()),
            "responsive_count": int(responsive_ids.numel()),
            "persistence_reference": reference_distance,
            "persistence_responsive": responsive_distance,
            "persistence_relative_gap": relative_gap(reference_distance, responsive_distance),
            "confidence_reference": reference_uncertainty,
            "confidence_responsive": responsive_uncertainty,
            "confidence_relative_gap": relative_gap(reference_uncertainty, responsive_uncertainty),
            "direct_confidence_mean": direct_confidence_mean,
            "direct_confidence_max": direct_confidence_max,
        })
    return rows
