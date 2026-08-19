from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


def _image_categories(coco) -> dict[int, frozenset[int]]:
    return {
        int(image_id): frozenset(
            int(annotation["category_id"])
            for annotation in coco.imgToAnns.get(image_id, [])
            if int(annotation.get("iscrowd", 0)) == 0
        )
        for image_id in coco.imgs
    }


def _counts(ids: Iterable[int], categories: dict[int, frozenset[int]]) -> Counter:
    result = Counter()
    for image_id in ids:
        result.update(categories[int(image_id)])
    return result


def select_reference_ids(
    coco,
    natural_count: int = 4000,
    augmentation_budget: int = 1000,
    quota: int = 50,
    seed: int = 42,
) -> dict:
    image_ids = np.asarray(sorted(int(image_id) for image_id in coco.imgs), dtype=np.int64)
    if natural_count + augmentation_budget > image_ids.size:
        raise ValueError("Reference request exceeds available COCO images")
    rng = np.random.default_rng(seed)
    natural = sorted(int(value) for value in rng.choice(image_ids, natural_count, replace=False))
    selected = set(natural)
    categories = _image_categories(coco)
    category_ids = sorted(int(category_id) for category_id in coco.cats)
    counts = _counts(natural, categories)
    augmentation: list[int] = []

    while len(augmentation) < augmentation_budget:
        under = {category_id for category_id in category_ids if counts[category_id] < quota}
        if not under:
            break
        candidates = [int(image_id) for image_id in image_ids if int(image_id) not in selected]
        best_id = min(
            candidates,
            key=lambda image_id: (
                -len(categories[image_id] & under),
                image_id,
            ),
        )
        if not (categories[best_id] & under):
            break
        augmentation.append(best_id)
        selected.add(best_id)
        counts.update(categories[best_id])

    remaining_budget = augmentation_budget - len(augmentation)
    if remaining_budget:
        remaining = np.asarray(
            [int(image_id) for image_id in image_ids if int(image_id) not in selected],
            dtype=np.int64,
        )
        fill = sorted(int(value) for value in rng.choice(remaining, remaining_budget, replace=False))
        augmentation.extend(fill)
        selected.update(fill)
        counts.update(_counts(fill, categories))

    return {
        "seed": seed,
        "natural_ids": natural,
        "augmentation_ids": sorted(augmentation),
        "category_image_counts": {category_id: counts[category_id] for category_id in category_ids},
        "unmet_category_quotas": {
            category_id: quota - counts[category_id]
            for category_id in category_ids
            if counts[category_id] < quota
        },
    }


def select_evaluation_ids(
    image_ids: Iterable[int],
    seed: int = 42,
    tuning_fraction: float = 0.5,
    pilot_per_partition: int = 250,
) -> dict:
    ordered = np.asarray(sorted(int(image_id) for image_id in image_ids), dtype=np.int64)
    shuffled = np.random.default_rng(seed).permutation(ordered)
    split = round(len(shuffled) * tuning_fraction)
    tuning_sequence = [int(value) for value in shuffled[:split]]
    test_sequence = [int(value) for value in shuffled[split:]]
    tuning = sorted(tuning_sequence)
    test = sorted(test_sequence)
    return {
        "seed": seed,
        "tuning_ids": tuning,
        "test_ids": test,
        "tuning_pilot_ids": sorted(tuning_sequence[: min(pilot_per_partition, len(tuning_sequence))]),
        "test_pilot_ids": sorted(test_sequence[: min(pilot_per_partition, len(test_sequence))]),
    }
