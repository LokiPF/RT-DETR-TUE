import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest

from differential_uncertainty.baselines import coco_quality as cq


def _gt(tmp_path):
    data = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100},
                   {"id": 2, "file_name": "b.jpg", "width": 100, "height": 100}],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 3, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
            {"id": 11, "image_id": 1, "category_id": 7, "bbox": [60, 60, 30, 30], "area": 900, "iscrowd": 1},
        ],
        "categories": [{"id": 3, "name": "car"}, {"id": 7, "name": "train"}],
    }
    path = tmp_path / "ann.json"
    path.write_text(json.dumps(data))
    return cq.CocoGroundTruth(path, expected_categories=None)


def test_ground_truth_splits_crowd_regions_and_maps_labels(tmp_path):
    gt = _gt(tmp_path)
    boxes, labels, crowd = gt.boxes(gt.image_id("a.jpg"))
    assert gt.category_ids == (3, 7)
    assert boxes.tolist() == [[10, 10, 30, 30]] and labels.tolist() == [0]
    assert crowd.tolist() == [[60, 60, 90, 90]]


def test_coco_map_and_per_image_ap(tmp_path):
    gt = _gt(tmp_path)
    results = cq.coco_results(1, np.array([0.9]), np.array([0]), np.array([[10, 10, 30, 30.0]]), gt.category_ids)
    assert results[0]["bbox"] == pytest.approx([10, 10, 20, 20]) and results[0]["category_id"] == 3
    assert cq.coco_map(gt, results, [1, 2]) == pytest.approx(1.0)
    ap = cq.per_image_ap(gt, {1: results, 2: []}, [1, 2])
    assert ap[0] == pytest.approx(1.0) and math.isnan(ap[1])
    assert cq.coco_map(gt, [], [1, 2]) == 0.0


def test_label_order_matches_rtdetr_mscoco_mapping():
    source = Path("/home/yuchen/YuchenZ/RT-DETR/rtdetrv2_pytorch/src/data/dataset/coco_dataset.py")
    annotations = Path("/home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json")
    if not (source.exists() and annotations.exists()):
        pytest.skip("RT-DETR repository or COCO annotations not available")
    tree = ast.parse(source.read_text())
    mapping = next(
        ast.literal_eval(node.value) for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "mscoco_category2name"
    )
    assert tuple(mapping) == cq.CocoGroundTruth(annotations).category_ids
