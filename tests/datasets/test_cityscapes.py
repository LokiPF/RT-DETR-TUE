import json
from pathlib import Path

import pytest
from PIL import Image

from degradation_monitor.datasets import coco
from degradation_monitor.datasets.cityscapes import CLASSES, Cityscapes
from degradation_monitor.datasets.coco import Coco, CocoGroundTruth, list_images

ROOT = Path("/home/yuchen/YuchenZ/Datasets/cityscape")


def _city_images(root, split, cities, per_city):
    for city in cities:
        folder = root / split / city
        folder.mkdir(parents=True)
        for index in range(per_city):
            Image.new("RGB", (8, 4)).save(folder / f"{city}_{index:06d}_000019_leftImg8bit.png")
    return root / split


def _annotations(path, file_names):
    path.write_text(json.dumps({
        "images": [{"id": i, "file_name": n, "width": 2048, "height": 1024} for i, n in enumerate(file_names)],
        "annotations": [{"id": 0, "image_id": 0, "category_id": 2, "bbox": [10, 20, 30, 40], "area": 1200,
                         "iscrowd": 0}],
        "categories": [{"id": i, "name": n} for i, n in enumerate(CLASSES)]}))
    return path


def test_cityscapes_lists_images_in_city_folders_in_a_seeded_order(tmp_path):
    val = _city_images(tmp_path, "val", ("frankfurt", "lindau"), 3)
    data = Cityscapes(tmp_path / "train", val, tmp_path / "ann.json")
    images = data.evaluation_images()
    assert sorted(p.name for p in images) == sorted(p.name for p in val.rglob("*.png"))
    assert images == Cityscapes(tmp_path / "train", val, tmp_path / "ann.json").evaluation_images()
    limited = Cityscapes(tmp_path / "train", val, tmp_path / "ann.json", limit=2).evaluation_images()
    assert limited == images[:2]
    assert data.folds().tolist() == [0, 1, 2, 3, 4, 0]


def test_cityscapes_splits_train_images_into_disjoint_reference_sets(tmp_path, monkeypatch):
    monkeypatch.setattr(coco, "SPLIT_IMAGES", (("reserved", 1), ("bank", 4), ("zstats", 2)))
    train = _city_images(tmp_path, "train", ("aachen", "bochum"), 4)
    data = Cityscapes(train, tmp_path / "val", tmp_path / "ann.json")
    splits = {name: data.reference_split(name) for name in ("reserved", "bank", "zstats")}
    assert [len(v) for v in splits.values()] == [1, 4, 2]
    assert len({p for v in splits.values() for p in v}) == 7


def test_cityscapes_ground_truth_finds_images_by_name_with_8_classes(tmp_path):
    path = _annotations(tmp_path / "ann.json", ["val/lindau/lindau_000000_000019_leftImg8bit.png"])
    gt = Cityscapes(tmp_path, tmp_path, path).ground_truth()
    boxes, labels, crowd = gt.boxes(gt.image_id("lindau_000000_000019_leftImg8bit.png"))
    assert boxes.tolist() == [[10, 20, 40, 60]] and labels.tolist() == [2] and crowd.size == 0
    assert gt.category_ids == tuple(range(8))


def test_an_annotation_file_naming_one_image_twice_is_refused(tmp_path):
    path = _annotations(tmp_path / "ann.json", ["val/a/x_leftImg8bit.png", "val/b/x_leftImg8bit.png"])
    with pytest.raises(ValueError, match="x_leftImg8bit.png appears twice"):
        CocoGroundTruth(path, expected_categories=8)


def test_repeated_image_names_are_refused(tmp_path):
    for city in ("a", "b"):
        (tmp_path / city).mkdir()
        Image.new("RGB", (4, 4)).save(tmp_path / city / "same.png")
    with pytest.raises(ValueError, match="image names repeat"):
        list_images(tmp_path, recursive=True)


def test_the_benchmarks_differ_in_layout_classes_and_history():
    assert (Coco.name, Coco.classes, Coco.recursive, Coco.screened) == ("coco", 80, False, True)
    assert (Cityscapes.name, Cityscapes.classes, Cityscapes.recursive, Cityscapes.screened) == (
        "cityscapes", 8, True, False)
    assert (Coco.min_clean_ap, Cityscapes.min_clean_ap) == (0.45, 0.35)


def test_the_real_cityscapes_folders_give_the_expected_images():
    annotations = ROOT / "annotations_coco" / "cityscapes_val_8cls.json"
    if not annotations.exists():
        pytest.skip("Cityscapes is not available")
    data = Cityscapes(ROOT / "leftImg8bit" / "train", ROOT / "leftImg8bit" / "val", annotations)
    assert len(data.train_images()) == 2975 and len(data.evaluation_images()) == 500
    assert len(data.reference_split("bank")) == 2000 and len(data.reference_split("zstats")) == 500
    gt = data.ground_truth()
    assert all(gt.image_id(p.name) is not None for p in data.evaluation_images())
