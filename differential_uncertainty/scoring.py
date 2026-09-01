from __future__ import annotations

import math

import torch
from torch import Tensor

from .config import FIXED_CONFIG, ExperimentConfig


def _finite_matrix(values: Tensor, name: str) -> None:
    if not isinstance(values, Tensor) or values.ndim != 2 or not values.is_floating_point():
        raise ValueError(f"{name} must be a rank-two floating tensor")
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must be finite")


def _float32_matrix(values: Tensor, name: str) -> Tensor:
    _finite_matrix(values, name)
    converted = values.detach().to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(converted).all()):
        raise ValueError(f"{name} must remain finite after float32 conversion")
    return converted


def normalize_bank(bank: Tensor) -> Tensor:
    values = _float32_matrix(bank, "bank")
    if values.shape[0] < 5:
        raise ValueError("bank needs at least five rows")
    norms = values.norm(dim=1)
    if not bool(torch.isfinite(norms).all()):
        raise ValueError("bank norms must be finite")
    if bool((norms == 0).any()):
        raise ValueError("bank contains a zero norm row")
    normalized = values / norms[:, None]
    if not bool(torch.isfinite(normalized).all()):
        raise ValueError("normalized bank must be finite")
    return normalized


def mean_five_cosine(queries: Tensor, normalized_bank: Tensor, chunk_size: int = 512) -> Tensor:
    queries = _float32_matrix(queries, "queries")
    normalized_bank = _float32_matrix(normalized_bank, "normalized_bank")
    if queries.shape[0] == 0:
        raise ValueError("queries must be nonempty")
    if queries.shape[1] != normalized_bank.shape[1]:
        raise ValueError("queries and bank feature dimensions must match")
    if normalized_bank.shape[0] < 5:
        raise ValueError("bank needs at least five rows")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    query_norms = queries.norm(dim=1)
    if not bool(torch.isfinite(query_norms).all()):
        raise ValueError("query norms must be finite")
    if bool((query_norms == 0).any()):
        raise ValueError("queries contain a zero norm row")
    bank_norms = normalized_bank.norm(dim=1)
    if not bool(torch.isfinite(bank_norms).all()):
        raise ValueError("normalized bank norms must be finite")
    if bool((bank_norms == 0).any()):
        raise ValueError("normalized_bank contains a zero norm row")
    normalized_queries = queries / query_norms[:, None]
    if not bool(torch.isfinite(normalized_queries).all()):
        raise ValueError("normalized queries must be finite")
    best = torch.full((queries.shape[0], 5), float("inf"))
    for chunk in normalized_bank.split(chunk_size):
        distances = 1.0 - normalized_queries @ chunk.T
        if not bool(torch.isfinite(distances).all()):
            raise ValueError("cosine distances must be finite")
        local = distances.topk(min(5, chunk.shape[0]), largest=False, dim=1).values
        best = torch.cat((best, local), dim=1).topk(5, largest=False, dim=1).values
    result = best.mean(dim=1)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("mean cosine distances must be finite")
    return result


def top_query_entropy(logits: Tensor) -> float:
    logits = _float32_matrix(logits, "logits")
    if logits.shape[0] == 0 or logits.shape[1] < 2:
        raise ValueError("logits need a nonempty query axis and at least two classes")
    confidence = logits.sigmoid().amax(dim=1)
    selected = logits[int(confidence.argmax())]
    probabilities = selected.softmax(dim=0)
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("softmax probabilities must be finite")
    entropy = -torch.xlogy(probabilities, probabilities).sum() / math.log(logits.shape[1])
    if not bool(torch.isfinite(entropy)):
        raise ValueError("entropy must be finite")
    return float(entropy)


def _union_padded_ids(records: list[dict], query_count: int) -> Tensor:
    result: set[int] = set()
    for record in records:
        padded = record.get("padded_ids")
        if not isinstance(padded, Tensor) or padded.ndim != 1 or padded.is_floating_point() or padded.dtype == torch.bool:
            raise ValueError("padded_ids must be a one-dimensional integer tensor")
        ids = padded.detach().to(device="cpu", dtype=torch.long)
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= query_count):
            raise ValueError("padded_ids lie outside query range")
        result.update(ids.tolist())
    return torch.tensor(sorted(result), dtype=torch.long)


def score_triplet(records, normalized_bank: Tensor, config: ExperimentConfig = FIXED_CONFIG) -> list[dict]:
    records = list(records)
    severities = [record.get("severity") for record in records]
    if len(records) != 3 or any(type(severity) is not int for severity in severities) or severities != [0, 4, 5]:
        raise ValueError("records must contain exact severities 0, 4, 5")
    image_ids = {record.get("image_id") for record in records}
    corruptions = {record.get("corruption") for record in records}
    if len(image_ids) != 1 or None in image_ids or len(corruptions) != 1 or None in corruptions:
        raise ValueError("records must describe one image and one corruption")
    _finite_matrix(normalized_bank, "normalized_bank")
    if normalized_bank.shape[0] < 5 or normalized_bank.shape[1] != config.persistence_dim:
        raise ValueError("normalized bank must have configured width and at least five rows")
    for record in records:
        logits, persistence = record.get("logits"), record.get("persistence")
        _finite_matrix(logits, "logits")
        _finite_matrix(persistence, "persistence")
        if tuple(logits.shape) != (config.query_count, config.class_count):
            raise ValueError("logits shape does not match configuration")
        if tuple(persistence.shape) != (config.query_count, config.persistence_dim):
            raise ValueError("persistence shape does not match configuration")
    padded = _union_padded_ids(records, config.query_count)
    keep = torch.ones(config.query_count, dtype=torch.bool)
    keep[padded] = False
    if not bool(keep.any()):
        raise ValueError("padding leaves no valid queries")
    rows = []
    for record in records:
        logits = record["logits"].detach().to(device="cpu", dtype=torch.float32)[keep]
        persistence = record["persistence"].detach().to(device="cpu", dtype=torch.float32)[keep]
        confidence = logits.sigmoid().amax(dim=1)
        distances = mean_five_cosine(persistence, normalized_bank, config.bank_chunk_size)
        fingerprint = float((distances * confidence).sum() / confidence.sum())
        row = {
            "image_id": next(iter(image_ids)),
            "corruption": next(iter(corruptions)),
            "severity": record["severity"],
            "fingerprint": fingerprint,
            "confidence": float(1.0 - confidence.max()),
            "entropy": top_query_entropy(logits),
        }
        if not all(math.isfinite(value) for value in row.values() if isinstance(value, float)):
            raise ValueError("scores must be finite")
        rows.append(row)
    return rows
