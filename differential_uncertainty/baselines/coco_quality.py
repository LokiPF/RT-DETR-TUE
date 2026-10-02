"""Detection quality on COCO: per-image AP and mAP (pycocotools)."""
from __future__ import annotations

import contextlib
import io
import math

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


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
        self._ids = {info["file_name"]: image_id for image_id, info in self.coco.imgs.items()}

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
