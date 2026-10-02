import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest

from degradation_monitor.datasets import coco
from degradation_monitor.datasets import coco as cq


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


def test_evaluation_images_use_the_old_benchmark_shuffle(tmp_path):
    for index in range(20):
        (tmp_path / f"{index:03d}.jpg").write_bytes(b"x")
    expected = sorted(tmp_path.iterdir())
    np.random.default_rng(44).shuffle(expected)
    images = coco.evaluation_images(tmp_path, seed=44)
    assert images == expected  # the archived benchmark selected its images with this same shuffle


def test_evaluation_images_reject_an_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="no images"):
        coco.evaluation_images(tmp_path, seed=44)


def test_folds_are_balanced_and_follow_shuffled_position():
    assert np.bincount(coco.assign_folds(5000)).tolist() == [1000] * 5
    assert coco.assign_folds(7).tolist() == [0, 1, 2, 3, 4, 0, 1]


def test_train_splits_are_disjoint_seeded_and_sized(monkeypatch):
    monkeypatch.setattr(coco, "SPLIT_IMAGES", (("reserved", 2), ("bank", 3), ("zstats", 2)))
    splits = coco.train_splits(10, seed=44)
    assert [len(splits[k]) for k in ("reserved", "bank", "zstats")] == [2, 3, 2]
    assert len(set(np.concatenate(list(splits.values())).tolist())) == 7
    again = coco.train_splits(10, seed=44)
    assert all(np.array_equal(splits[k], again[k]) for k in splits)
    with pytest.raises(ValueError, match="needs 7 train images"):
        coco.train_splits(6, seed=44)


def test_train_splits_reproduce_the_stored_bank_draw():
    # the pilot drew calibration, bank and z-statistics images as order[:200], order[200:2200], order[2200:2700]
    order = np.random.default_rng(44).permutation(5000)
    splits = coco.train_splits(5000, seed=44)
    assert np.array_equal(splits["bank"], np.sort(order[200:2200]))
    assert np.array_equal(splits["zstats"], np.sort(order[2200:2700]))


def test_the_coco_dataset_gives_the_limited_seeded_order_its_folds_and_the_reference_splits(tmp_path, monkeypatch):
    for folder, count in (("train", 9), ("val", 6)):
        (tmp_path / folder).mkdir()
        for index in range(count):
            (tmp_path / folder / f"{index:03d}.jpg").write_bytes(b"x")
    monkeypatch.setattr(coco, "SPLIT_IMAGES", (("reserved", 2), ("bank", 4), ("zstats", 3)))
    dataset = coco.Coco(tmp_path / "train", tmp_path / "val", tmp_path / "ann.json", limit=4)
    assert dataset.evaluation_images() == coco.evaluation_images(tmp_path / "val", seed=44)[:4]
    assert dataset.folds().tolist() == [0, 1, 2, 3]
    bank = dataset.reference_split("bank")
    assert len(bank) == 4 and set(bank) <= set(dataset.train_images())
    assert not set(bank) & set(dataset.reference_split("zstats"))
