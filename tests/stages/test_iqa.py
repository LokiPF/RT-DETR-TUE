import json
import os
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image

from degradation_monitor import corruptions
from degradation_monitor.detectors import COCO_CATEGORY_IDS
from degradation_monitor.runs import Manifest, RunLayout, atomic_npz
from degradation_monitor.stages import iqa as stage

ROWS = ("niqe", "niqe_default", "arniqa", "arniqa_proto", "clipiqa")
BASE = """
run = "{root}/runs/coco"
checkpoint = "{root}/rtdetr.pth"
train_images = "{images}/train"
val_images = "{images}/val"
annotations = "{images}/ann.json"
discopatch_root = "{root}"
device = "cpu"
batch_size = 16
workers = 0
"""
DETECTORS = """
base = "base.toml"
run = "{root}/runs/coco-detectors"
reference_run = "{root}/runs/coco"
gpu_memory_gib = 8.0

[weights]
yolo11m = "{root}/yolo.pt"

[clean_ap_floor]
yolo11m = 0.0
"""
IQA = """
base = "base.toml"
run = "{root}/runs/coco-iqa"
reference_run = "{root}/runs/coco"
detectors_config = "detectors.toml"
gpu_memory_gib = 8.0
"""


class FakeIqa:
    """An image-quality stand-in: rows that brighten with the image, and features that make a valid NIQE fit."""
    batch_size, fit_batch_size = 16, 4
    protocol = {"fake": True}
    calls = 0

    def __init__(self, device, niqe_refit=None, prototype=None):
        self.niqe_refit, self.prototype = niqe_refit, prototype

    def fit_features(self, arrays):
        FakeIqa.calls += 1
        n = len(arrays)
        generator = torch.Generator().manual_seed(n)
        features = torch.rand(n, 4, 36, dtype=torch.float64, generator=generator)
        sharpness = torch.tensor([[1.0, 1.0, 1.0, 0.1]] * n)  # three sharp blocks per image
        embedding = torch.tensor([[a.mean() / 255.0] * 8 for a in arrays], dtype=torch.float64)
        if arrays[0].shape[0] < 40:  # the fixture's small images stand in for images without a whole NIQE block
            features = sharpness = None
        return features, sharpness, embedding

    def _rows(self, arrays):
        level = torch.tensor([a.mean() / 255.0 for a in arrays], dtype=torch.float64)
        return {row: level + k for k, row in enumerate(ROWS)}

    def niqe_rows(self, arrays):
        return {**{k: v for k, v in self._rows(arrays).items() if k.startswith("niqe")},
                "niqe_blocks": torch.full((len(arrays),), 4)}

    def arniqa_rows(self, arrays):
        return {k: v for k, v in self._rows(arrays).items() if k.startswith("arniqa")}

    def clipiqa_rows(self, arrays):
        return {"clipiqa": self._rows(arrays)["clipiqa"]}

    def scores(self, arrays):
        FakeIqa.calls += 1
        return {**{k: v.numpy() for k, v in self._rows(arrays).items()}, "niqe_blocks": np.full(len(arrays), 4)}


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    """24 clean train images (four of them too small for a NIQE block), 3 val images and their 96 digests."""
    root = tmp_path_factory.mktemp("images")
    rng = np.random.default_rng(0)
    for folder, count in (("train", 24), ("val", 3)):
        (root / folder).mkdir()
        for index in range(count):
            shape = (32, 64, 3) if folder == "train" and index < 4 else (48, 64, 3)
            Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8)).save(root / folder / f"{index:012d}.jpg")
    (root / "ann.json").write_text(json.dumps({
        "images": [{"id": i + 1, "file_name": f"{i:012d}.jpg", "width": 64, "height": 48} for i in range(3)],
        "annotations": [], "categories": [{"id": c, "name": str(c)} for c in COCO_CATEGORY_IDS]}))
    digests = {}
    for path in sorted((root / "val").iterdir()):
        name, arrays = corruptions.load_variants(path)
        digests[name] = np.array([corruptions.digest(a) for a in arrays])
    return root, digests


@pytest.fixture
def config(tmp_path, images, monkeypatch):
    root, digests = images
    for name in ("rtdetr.pth", "yolo.pt"):
        (tmp_path / name).write_bytes(name.encode())
    weights = {}
    for name in ("arniqa_encoder", "arniqa_regressor", "niqe_published", "clip_rn50"):
        weights[name] = tmp_path / f"{name}.bin"
        weights[name].write_bytes(name.encode())
    (tmp_path / "base.toml").write_text(BASE.format(root=tmp_path, images=root))
    (tmp_path / "detectors.toml").write_text(DETECTORS.format(root=tmp_path))
    (tmp_path / "iqa.toml").write_text(IQA.format(root=tmp_path))
    monkeypatch.setattr(stage, "load_iqa_models", lambda device, niqe_refit=None, prototype=None:
                        FakeIqa(device, niqe_refit, prototype))
    monkeypatch.setattr(stage, "weight_files", lambda: weights)
    FakeIqa.calls = 0
    reference = RunLayout(tmp_path / "runs" / "coco")  # RT-DETR's run: its digests, read only
    for name, values in digests.items():
        atomic_npz(reference.score_file("detector", name), digests=values)
    return stage.load_config(tmp_path / "iqa.toml")


def _listing(folder):
    return sorted((str(p.relative_to(folder)), p.stat().st_mtime_ns) for p in folder.rglob("*") if p.is_file())


def test_the_iqa_config_lists_every_detectors_run(config):
    assert list(config.detectors) == ["rtdetrv2_r18", "yolo11m"]
    assert config.detectors["rtdetrv2_r18"].run == config.reference_run
    assert config.detectors["yolo11m"].run.name == "yolo11m" and config.gpu_memory_gib == 8.0


def test_the_run_root_keeps_clear_of_every_detectors_run(config, tmp_path):
    yolo = config.detectors["yolo11m"].run
    for root in (config.reference_run, config.reference_run / "inner", yolo, yolo.parent, tmp_path / "runs"):
        with pytest.raises(ValueError, match="must lie outside the detectors' run folders"):
            stage.load_config(tmp_path / "iqa.toml", run=root)
    assert stage.main(["report", "--config", str(tmp_path / "iqa.toml"), "--run", str(yolo.parent)]) == 2


def test_the_fit_writes_the_refit_and_the_prototype_once(config):
    stage.fit(config)
    refit = config.layout.reference("iqa", "niqe_refit.npz")
    with np.load(refit) as data:
        assert data["mu"].shape == (36,) and data["cov"].shape == (36, 36) and int(data["blocks"]) == 20 * 3
    assert np.load(config.layout.reference("iqa", "arniqa_prototype.npy")).shape == (8,)
    record = json.loads(config.layout.reference("iqa", "fit.json").read_text())
    assert record["images"] == 24 and record["niqe_images"] == 20 and record["niqe_skipped"] == 4
    assert record["niqe_dropped_blocks"] == 0
    manifest = json.loads(config.layout.manifest.read_text())
    assert manifest["protocol"]["models"] == {"fake": True} and set(manifest["protocol"]["weights"]) == {
        "arniqa_encoder", "arniqa_regressor", "niqe_published", "clip_rn50"}
    calls = FakeIqa.calls
    stage.fit(config)
    assert FakeIqa.calls == calls


def test_the_pass_writes_the_five_rows_and_resumes(config):
    stage.fit(config)
    stage.iqa_pass(config, first=1)
    assert len(list(config.layout.scores("iqa").glob("*.npz"))) == 1
    stage.iqa_pass(config)
    calls = FakeIqa.calls
    files = sorted(config.layout.scores("iqa").glob("*.npz"))
    assert len(files) == 3
    with np.load(files[0]) as data:
        assert set(data.files) == {*ROWS, "niqe_blocks", "digests"} and all(data[row].shape == (96,) for row in ROWS)
    assert set(json.loads(config.layout.manifest.read_text())["inputs"]) == {"iqa"}
    stage.iqa_pass(config)
    assert FakeIqa.calls == calls


def test_the_pass_refuses_corruptions_that_differ_from_the_reference_run(config):
    stage.fit(config)
    first = config.base.dataset.evaluation_images()[0].name
    atomic_npz(RunLayout(config.reference_run).score_file("detector", first), digests=np.array(["0" * 16] * 96))
    with pytest.raises(ValueError, match="corruptions differ from the reference run"):
        stage.iqa_pass(config)


def test_the_pass_refuses_scores_from_a_changed_fit(config):
    stage.fit(config)
    stage.iqa_pass(config, first=1)
    np.save(config.layout.reference("iqa", "arniqa_prototype.npy"), np.ones(8))
    with pytest.raises(ValueError, match="inputs of scores/iqa changed"):
        stage.iqa_pass(config)


def test_the_report_writes_every_detectors_report_under_the_iqa_run_only(config, monkeypatch):
    stage.fit(config)
    stage.iqa_pass(config)
    reported = []

    def write_report(settings, manifest, out=None, extra_rows=None, extra_inputs=None):
        reported.append((settings.run, out, sorted(extra_rows), extra_rows["niqe"].shape, settings.limit))
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps({"headline": {}, "intervals": {}}))

    monkeypatch.setattr(stage, "write_report", write_report)
    before = _listing(config.reference_run)
    stage.report(config, first=2)
    assert [r[0] for r in reported] == [config.reference_run, config.detectors["yolo11m"].run]
    assert [r[1] for r in reported] == [config.layout.report("rtdetrv2_r18"), config.layout.report("yolo11m")]
    assert all(r[2] == sorted(ROWS) and r[3] == (2, 96) and r[4] == 2 for r in reported)
    assert _listing(config.reference_run) == before  # the reference run is only read
    assert (config.run / "summary.md").exists() and (config.run / "summary.csv").exists()


def test_the_report_refuses_a_detector_scored_on_other_images(config):
    stage.fit(config)
    stage.iqa_pass(config)
    yolo = replace(config.detectors["yolo11m"], limit=2)
    with pytest.raises(ValueError, match="evaluation images of yolo11m differ"):
        stage.report(replace(config, detectors={**config.detectors, "yolo11m": yolo}))
    assert not config.layout.report("rtdetrv2_r18").exists()  # checked before any report is written


def test_the_iqa_table_gives_the_two_axis_score_minus_each_row(config):
    head = {row: {"auroc_common": 0.6, "auroc_extra": 0.55} for row in ROWS}
    head["two_axis"] = {"auroc_common": 0.9, "auroc_extra": 0.85}
    cell = {"point": 0.3, "low": 0.28, "high": 0.32}
    intervals = {s: {f"two_axis - {row}:auroc_{g}": cell for row in ROWS for g in ("common", "extra")}
                 for s in ("all", "untouched")}
    for name in config.detectors:
        folder = config.layout.report(name)
        folder.mkdir(parents=True)
        (folder / "summary.json").write_text(json.dumps({"headline": {"all": head, "untouched": head},
                                                         "intervals": intervals}))
    rows = stage.iqa_table(config)
    assert len(rows) == 2 * 5 and rows[0]["detector"] == "rtdetrv2_r18" and rows[0]["row"] == "niqe"
    assert rows[0]["all_auroc_common"] == 0.6 and rows[0]["untouched_auroc_extra"] == 0.55
    assert rows[0]["all_two_axis_minus_common"] == 0.3 and rows[0]["untouched_two_axis_minus_extra_low"] == 0.28
    lines = (config.run / "summary.md").read_text().splitlines()
    labels = [line.split(" | ")[1] for line in lines if line.startswith("| rtdetrv2_r18 |")]
    assert len(labels) == len(set(labels)) == 5  # the two NIQE rows told apart


def test_timing_records_each_model_and_the_detector(config):
    stage.fit(config)
    (config.reference_run / "timing.json").write_text(json.dumps({"detector_ms": 5.9}))
    stage.timing(config)
    result = json.loads(config.layout.timing.read_text())
    assert {"niqe_ms", "arniqa_ms", "clipiqa_ms"} <= set(result) and result["detector_ms"] == 5.9
    assert result["batch_size"] == 1 and result["images"] == 3


def test_first_applies_to_the_pass_and_report_only(config, tmp_path):
    with pytest.raises(SystemExit):
        stage.main(["fit", "--config", str(tmp_path / "iqa.toml"), "--first", "2"])
