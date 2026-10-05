"""COCO 2017: clean train images, the seed-44 val order and its folds, and the ground truth the baselines need."""
from __future__ import annotations

import contextlib
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ..method.reference import BANK_IMAGES, ZSTAT_IMAGES

FOLDS = 5
SEED = 44
# Disjoint seeded draws of train images. "reserved" was the archived conv-TU pilot's calibration set; the slot is kept
# so that the bank and z-statistics images stay exactly those the stored references were computed from.
SPLIT_IMAGES = (("reserved", 200), ("bank", BANK_IMAGES), ("zstats", ZSTAT_IMAGES))
_SUFFIXES = {".jpg", ".jpeg", ".png"}


def list_images(root, recursive: bool = False) -> list[Path]:
    """The images in root (and its subfolders when recursive), sorted; refused when two share a name, because result
    files and the ground truth are keyed by name."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"image directory does not exist: {root}")
    candidates = root.rglob("*") if recursive else root.iterdir()
    images = sorted(p for p in candidates if p.is_file() and p.suffix.lower() in _SUFFIXES)
    if len({p.stem for p in images}) != len(images):
        raise ValueError(f"image names repeat under {root}; result files are keyed by name")
    return images


def evaluation_images(val_root, *, seed: int, recursive: bool = False) -> list[Path]:
    """All val images in the old benchmark's seeded shuffle order; the order defines the folds."""
    images = list_images(val_root, recursive)
    if not images:
        raise ValueError(f"no images found in {val_root}")
    np.random.default_rng(seed).shuffle(images)
    return images


def assign_folds(count: int, folds: int = FOLDS) -> np.ndarray:
    """Fold of each image from its shuffled position: 0, 1, ..., folds - 1, 0, 1, ..."""
    if count <= 0 or folds < 2:
        raise ValueError("count must be positive and folds at least 2")
    return np.arange(count) % folds


def train_splits(count: int, seed: int = SEED) -> dict:
    """Sorted, disjoint, seeded draws of train-image indices: reserved, bank and z-statistics."""
    needed = sum(size for _, size in SPLIT_IMAGES)
    if count < needed:
        raise ValueError(f"the reference needs {needed} train images, found {count}")
    order = np.random.default_rng(seed).permutation(count)
    out, start = {}, 0
    for name, size in SPLIT_IMAGES:
        out[name] = np.sort(order[start:start + size])
        start += size
    return out


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


class CocoGroundTruth:
    def __init__(self, annotation_file, expected_categories=80):
        with _quiet():
            self.coco = COCO(str(annotation_file))
        self.category_ids = tuple(sorted(self.coco.getCatIds()))
        if expected_categories is not None and len(self.category_ids) != expected_categories:
            raise ValueError(f"expected {expected_categories} categories, found {len(self.category_ids)}")
        self.label_of = {c: i for i, c in enumerate(self.category_ids)}
        self._ids = {}
        for image_id, info in self.coco.imgs.items():
            name = Path(info["file_name"]).name  # Cityscapes stores split/city/name; images are looked up by name
            if name in self._ids:
                raise ValueError(f"{name} appears twice in {annotation_file}")
            self._ids[name] = image_id

    def image_id(self, file_name: str) -> int:
        try:
            return self._ids[file_name]
        except KeyError as error:
            raise KeyError(f"{file_name} is not in the annotation file") from error

    def boxes(self, image_id: int):
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image_id]))
        def xyxy(items):
            return np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2],
                              a["bbox"][1] + a["bbox"][3]] for a in items], dtype=np.float64).reshape(-1, 4)
        plain = [a for a in anns if not a.get("iscrowd", 0)]
        crowd = [a for a in anns if a.get("iscrowd", 0)]
        labels = np.array([self.label_of[a["category_id"]] for a in plain], dtype=np.int64)
        return xyxy(plain), labels, xyxy(crowd)


def coco_results(image_id, scores, labels, boxes, category_ids) -> list[dict]:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return [{"image_id": int(image_id), "category_id": int(category_ids[int(l)]),
             "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
             "score": float(s)} for s, l, b in zip(scores, labels, boxes)]


def _evaluate(gt, detections, image_ids):
    evaluator = COCOeval(gt.coco, detections, "bbox")
    evaluator.params.imgIds = list(image_ids)
    with _quiet():
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return float(evaluator.stats[0])


def coco_map(gt, results, image_ids) -> float:
    if not results:
        return 0.0
    with _quiet():
        detections = gt.coco.loadRes(list(results))
    return max(_evaluate(gt, detections, image_ids), 0.0)


def per_image_ap(gt, results_by_image, image_ids) -> np.ndarray:
    """COCO AP@[.5:.95] for each image separately; NaN for images without objects."""
    all_results = [r for image_id in image_ids for r in results_by_image.get(image_id, [])]
    has_objects = [bool(gt.boxes(i)[1].size) for i in image_ids]
    if not all_results:
        return np.array([0.0 if objects else math.nan for objects in has_objects])
    with _quiet():
        detections = gt.coco.loadRes(all_results)
    values = []
    for image_id, objects in zip(image_ids, has_objects):
        ap = _evaluate(gt, detections, [image_id]) if objects else -1.0
        values.append(ap if ap >= 0 else math.nan)
    return np.array(values)


@dataclass(frozen=True)
class Coco:
    """What a stage needs from the dataset: train images, reference splits, the evaluation order, folds, ground truth.

    The class attributes are what differs between benchmarks: the name, the ground truth's class count, the floor for
    the detector's clean val AP, whether the images sit in subfolders, and whether the method was screened on these
    images (then the report adds the screen, held-out and untouched sets and their pre-registered decisions).
    """
    train_root: Path
    val_root: Path
    annotations: Path
    seed: int = SEED
    limit: Optional[int] = None
    name: ClassVar[str] = "coco"
    classes: ClassVar[int] = 80
    min_clean_ap: ClassVar[float] = 0.45  # the COCO checkpoint's clean val AP is about 0.48
    recursive: ClassVar[bool] = False
    screened: ClassVar[bool] = True

    def train_images(self) -> list[Path]:
        return list_images(self.train_root, self.recursive)

    def reference_split(self, name: str) -> list[Path]:
        paths = self.train_images()
        return [paths[i] for i in train_splits(len(paths), self.seed)[name]]

    def evaluation_images(self) -> list[Path]:
        images = evaluation_images(self.val_root, seed=self.seed, recursive=self.recursive)
        return images[: self.limit] if self.limit else images

    def folds(self) -> np.ndarray:
        return assign_folds(len(self.evaluation_images()))

    def ground_truth(self) -> CocoGroundTruth:
        return CocoGroundTruth(self.annotations, expected_categories=self.classes)
