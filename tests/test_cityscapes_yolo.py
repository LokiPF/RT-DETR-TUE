import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image
from ultralytics.data.utils import img2label_paths, verify_image_label

from cityscapes.models import yolo11m

CITIES = ("aachen", "bochum")


def _fake_annotations(root: Path) -> Path:
    """Two 20 x 10 val images in city folders: aachen's holds a person and a car, bochum's no box."""
    for city in CITIES:
        folder = root / "leftImg8bit" / "val" / city
        folder.mkdir(parents=True)
        Image.new("RGB", (20, 10)).save(folder / f"{city}_000000_000019_leftImg8bit.png")
    path = root / "val.json"
    path.write_text(json.dumps({
        "images": [{"id": i, "file_name": f"val/{city}/{city}_000000_000019_leftImg8bit.png", "width": 20,
                    "height": 10} for i, city in enumerate(CITIES)],
        "annotations": [{"id": 0, "image_id": 0, "category_id": 0, "bbox": [2, 1, 4, 6], "area": 24, "iscrowd": 0},
                        {"id": 1, "image_id": 0, "category_id": 2, "bbox": [10, 5, 10, 5], "area": 50, "iscrowd": 0}],
        "categories": [{"id": i, "name": name} for i, name in enumerate(yolo11m.CLASSES)]}))
    return path


def _prepared(root: Path) -> Path:
    yolo11m.prepare_split(_fake_annotations(root), root / "leftImg8bit", root / "out", "val")
    return root / "out"


def _ultralytics_reads(out: Path, city: str):
    """What Ultralytics' dataset check makes of one image and its label file: (labels, missing, found, empty,
    corrupt, message)."""
    name = f"{city}_000000_000019_leftImg8bit"
    image, label = out / "images" / "val" / f"{name}.png", out / "labels" / "val" / f"{name}.txt"
    _, labels, _, _, _, missing, found, empty, corrupt, message = verify_image_label(
        (str(image), str(label), "", False, len(yolo11m.CLASSES), 0, 0, False))
    return labels, missing, found, empty, corrupt, message


def test_a_box_becomes_its_normalised_line_and_person_stays_class_0(tmp_path):
    counts = yolo11m.prepare_split(_fake_annotations(tmp_path), tmp_path / "leftImg8bit", tmp_path / "out", "val")
    assert counts == (2, 2)
    label = tmp_path / "out" / "labels" / "val" / "aachen_000000_000019_leftImg8bit.txt"
    assert label.read_text().splitlines() == ["0 0.200000 0.400000 0.200000 0.600000",
                                              "2 0.750000 0.750000 0.500000 0.500000"]


def test_ultralytics_reads_the_labels_with_person_as_class_0(tmp_path):
    labels, missing, found, empty, corrupt, message = _ultralytics_reads(_prepared(tmp_path), "aachen")
    assert (missing, found, empty, corrupt) == (0, 1, 0, 0), message
    assert labels[:, 0].tolist() == [0.0, 2.0]


def test_an_image_without_a_box_gets_an_empty_label_that_ultralytics_reads_as_background(tmp_path):
    out = _prepared(tmp_path)
    assert (out / "labels" / "val" / "bochum_000000_000019_leftImg8bit.txt").read_text() == ""
    _, missing, found, empty, corrupt, message = _ultralytics_reads(out, "bochum")
    assert (missing, found, empty, corrupt) == (0, 1, 1, 0), message


def test_each_link_points_at_its_image_and_ultralytics_finds_its_label(tmp_path):
    links = sorted((_prepared(tmp_path) / "images" / "val").iterdir())
    assert all(link.is_symlink() for link in links)
    assert [link.resolve() for link in links] == [
        (tmp_path / "leftImg8bit" / "val" / city / f"{city}_000000_000019_leftImg8bit.png").resolve()
        for city in CITIES]
    assert img2label_paths([str(link) for link in links]) == [
        str(tmp_path / "out" / "labels" / "val" / f"{link.stem}.txt") for link in links]


def test_a_second_run_replaces_the_links_and_drops_ultralytics_label_cache(tmp_path):
    out = _prepared(tmp_path)
    cache = out / "labels" / "val.cache"  # Ultralytics keys it on file sizes and paths only
    cache.write_text("stale")
    annotations = tmp_path / "val.json"
    assert yolo11m.prepare_split(annotations, tmp_path / "leftImg8bit", out, "val") == (2, 2)
    assert not cache.exists()
    assert len(list((out / "images" / "val").iterdir())) == 2


def test_a_missing_image_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="no image at"):
        yolo11m.prepare_split(_fake_annotations(tmp_path), tmp_path / "elsewhere", tmp_path / "out", "val")


def test_the_data_file_names_the_8_classes_in_order_and_points_at_the_dataset():
    data = yaml.safe_load(yolo11m.DATA_YAML.read_text())
    assert data["names"] == dict(enumerate(yolo11m.CLASSES))
    assert Path(data["path"]) == yolo11m.DATASET
    assert (data["train"], data["val"]) == ("images/train", "images/val")


class FakeYolo:
    """Records what the script asks of Ultralytics' YOLO, without loading a model."""
    calls = []

    def __init__(self, weights):
        FakeYolo.calls.append(("load", weights))

    def train(self, **arguments):
        FakeYolo.calls.append(("train", arguments))

    def val(self, **arguments):
        FakeYolo.calls.append(("val", arguments))
        return SimpleNamespace(box=SimpleNamespace(map=0.3, map50=0.5))


@pytest.fixture
def fake_yolo(tmp_path, monkeypatch):
    FakeYolo.calls = []
    monkeypatch.setattr("ultralytics.YOLO", FakeYolo)
    monkeypatch.setattr(yolo11m, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(yolo11m, "cap_gpu_memory", lambda gib: FakeYolo.calls.append(("cap", gib)))
    monkeypatch.chdir(tmp_path)  # train() moves into RUNS; this puts the working folder back afterwards
    return FakeYolo.calls


def test_train_passes_the_standard_recipe_and_nothing_else(fake_yolo, tmp_path):
    yolo11m.train()
    yolo11m.train(smoke=True)
    where = {"data": str(yolo11m.DATA_YAML), "imgsz": 640, "batch": 16, "device": 0, "project": str(tmp_path / "runs")}
    load = ("load", str(yolo11m.COCO_WEIGHTS))
    assert fake_yolo == [("cap", 12.0), load, ("train", {**where, "epochs": 100, "name": "train"}),
                         ("cap", 12.0), load, ("train", {**where, "epochs": 1, "name": "smoke"})]
    assert Path.cwd() == (tmp_path / "runs").resolve()  # the AMP check downloads yolo11n.pt here, not into the repo


def test_val_measures_the_frozen_weights_at_640(fake_yolo, tmp_path):
    assert yolo11m.val() == (0.3, 0.5)
    assert fake_yolo == [("cap", 12.0), ("load", str(yolo11m.FROZEN_WEIGHTS)),
                         ("val", {"data": str(yolo11m.DATA_YAML), "imgsz": 640, "batch": 16, "device": 0,
                                  "project": str(tmp_path / "runs"), "name": "val"})]
