from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset


def deterministic_split(
    num_items: int,
    calibration_fraction: float,
    seed: int,
    max_items: int | None = None,
) -> tuple[list[int], list[int]]:
    if num_items < 2:
        raise ValueError("Need at least two items to split.")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in the open interval (0, 1)")

    generator = np.random.default_rng(seed)
    permutation = generator.permutation(num_items)

    if max_items is not None:
        if max_items < 2:
            raise ValueError("max_items must be at least 2.")
        pool = permutation[: min(max_items, num_items)]
    else:
        pool = permutation

    pool_size = len(pool)
    num_calibration = max(1, round(calibration_fraction * pool_size))
    num_calibration = min(num_calibration, pool_size - 1)  # keep >=1 for frechet

    calibration = sorted(int(i) for i in pool[:num_calibration])
    frechet = sorted(int(i) for i in pool[num_calibration:])

    # Disjointness is guaranteed by construction; assert to be safe.
    assert not (set(frechet) & set(calibration)), "split partitions overlap"
    return frechet, calibration


def _image_ids(dataset, indices: list[int]) -> list[int] | None:
    ids = getattr(dataset, "ids", None)
    if ids is None:
        return None
    return [int(ids[i]) for i in indices]


def class_histogram(dataset, indices: list[int]) -> dict[int, int] | None:
    coco = getattr(dataset, "coco", None)
    ids = getattr(dataset, "ids", None)
    if coco is None or ids is None:
        return None

    histogram: dict[int, int] = {}
    for index in indices:
        image_id = ids[index]
        for annotation in coco.loadAnns(coco.getAnnIds(imgIds=image_id)):
            category = int(annotation["category_id"])
            histogram[category] = histogram.get(category, 0) + 1
    return histogram


def save_split(
    dataset,
    output_path,
    calibration_fraction: float,
    seed: int,
    max_items: int | None = None,
    metadata: dict | None = None,
) -> dict:
    """Compute and write the split JSON; return the payload."""
    total = len(dataset)
    frechet, calibration = deterministic_split(
        total, calibration_fraction, seed, max_items=max_items
    )

    payload = {
        "total": total,
        "max_items": max_items,
        "pool_size": len(frechet) + len(calibration),
        "seed": seed,
        "calibration_fraction": calibration_fraction,
        "num_frechet": len(frechet),
        "num_calibration": len(calibration),
        "frechet_indices": frechet,
        "calibration_indices": calibration,
        "frechet_image_ids": _image_ids(dataset, frechet),
        "calibration_image_ids": _image_ids(dataset, calibration),
        "metadata": metadata or {},
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return payload


def load_split(path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def subset_dataloader(
    base_loader,
    indices,
    shuffle: bool = False,
    num_workers: int | None = None,
) -> DataLoader:
    subset = Subset(base_loader.dataset, list(indices))
    batch_size = getattr(base_loader, "batch_size", None) or 1
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=(
            getattr(base_loader, "num_workers", 0)
            if num_workers is None
            else num_workers
        ),
        collate_fn=getattr(base_loader, "collate_fn", None),
        pin_memory=getattr(base_loader, "pin_memory", False),
        drop_last=False,
    )
