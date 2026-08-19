from __future__ import annotations

from collections.abc import Iterable

import math

import numpy as np
import torch
from torch import Tensor


def _stored_vector(vector: Tensor) -> Tensor:
    """Copy one vector into memory it owns.

    Callers stream row views of whole `(query, feature)` record tensors, and a view
    keeps its entire record alive; without the copy a capped reservoir still pins the
    whole cache.
    """
    return vector.detach().cpu().float().clone()


def deterministic_reservoir(vectors: Iterable[Tensor], capacity: int, seed: int) -> Tensor:
    rng = np.random.default_rng(seed)
    reservoir: list[Tensor] = []
    for seen, vector in enumerate(vectors, start=1):
        value = _stored_vector(vector)
        if len(reservoir) < capacity:
            reservoir.append(value)
            continue
        replacement = int(rng.integers(0, seen))
        if replacement < capacity:
            reservoir[replacement] = value
    if not reservoir:
        raise ValueError("Cannot build a bank from zero vectors")
    return torch.stack(reservoir)


def _sample_indices(indices: Tensor, count: int, rng: np.random.Generator) -> list[int]:
    values = indices.cpu().numpy()
    chosen = rng.choice(values, size=min(count, len(values)), replace=False)
    return [int(value) for value in chosen]


def make_coverage_bank(
    vectors: Tensor,
    matched_gt_class: Tensor,
    capacity: int,
    object_fraction: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    if not 0.0 < object_fraction < 1.0:
        raise ValueError("object_fraction must lie between zero and one")
    rng = np.random.default_rng(seed)
    object_budget = round(capacity * object_fraction)
    background_budget = capacity - object_budget
    object_indices = torch.where(matched_gt_class >= 0)[0]
    background_indices = torch.where(matched_gt_class < 0)[0]
    selected: list[int] = []
    classes = sorted(int(value) for value in matched_gt_class[object_indices].unique().tolist())
    per_class = max(1, object_budget // max(1, len(classes)))
    for class_id in classes:
        class_indices = torch.where(matched_gt_class == class_id)[0]
        selected.extend(_sample_indices(class_indices, per_class, rng))
    selected = selected[:object_budget]
    selected.extend(_sample_indices(background_indices, background_budget, rng))
    if len(selected) < capacity:
        selected_set = set(selected)
        remaining = torch.tensor(
            [index for index in range(vectors.shape[0]) if index not in selected_set],
            dtype=torch.long,
        )
        selected.extend(_sample_indices(remaining, capacity - len(selected), rng))
    selected_tensor = torch.tensor(selected, dtype=torch.long)
    return vectors.index_select(0, selected_tensor).float(), selected_tensor


def streaming_coverage_bank(
    records: Iterable[dict],
    layer_id: int,
    capacity: int,
    object_fraction: float,
    class_count: int,
    seed: int,
) -> tuple[Tensor, dict]:
    if capacity < 2 or not 0.0 < object_fraction < 1.0:
        raise ValueError("Coverage bank needs capacity >= 2 and object_fraction between zero and one")
    object_budget = round(capacity * object_fraction)
    per_class_capacity = max(1, math.ceil(object_budget / class_count))
    class_vectors: dict[int, list[Tensor]] = {class_id: [] for class_id in range(class_count)}
    class_seen = {class_id: 0 for class_id in range(class_count)}
    background: list[Tensor] = []
    background_seen = 0
    generators = {class_id: np.random.default_rng(seed + class_id) for class_id in range(class_count)}
    background_rng = np.random.default_rng(seed + class_count + 1)

    def update(reservoir, seen, value, limit, rng):
        seen += 1
        if len(reservoir) < limit:
            reservoir.append(_stored_vector(value))
        else:
            replacement = int(rng.integers(0, seen))
            if replacement < limit:
                reservoir[replacement] = _stored_vector(value)
        return seen

    for record in records:
        for vector, class_id in zip(record["layers"][layer_id], record["matched_gt_class"]):
            class_value = int(class_id)
            if class_value < 0:
                background_seen = update(background, background_seen, vector, capacity, background_rng)
            else:
                class_seen[class_value] = update(
                    class_vectors[class_value], class_seen[class_value], vector,
                    per_class_capacity, generators[class_value],
                )
    objects = [vector for class_id in range(class_count) for vector in class_vectors[class_id]]
    if not objects or not background:
        raise ValueError("Coverage bank requires both matched-object and background vectors")
    object_bank = deterministic_reservoir(objects, min(object_budget, len(objects)), seed + 10_000)
    background_needed = min(capacity - len(object_bank), len(background))
    background_bank = deterministic_reservoir(background, background_needed, seed + 20_000)
    bank = torch.cat((object_bank, background_bank))
    return bank, {
        "object_vectors": int(len(object_bank)),
        "background_vectors": int(len(background_bank)),
        "per_class_available": class_seen,
    }
