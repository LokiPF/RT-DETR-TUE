import json
import os
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image

from degradation_monitor import corruptions
from degradation_monitor.datasets import coco
from degradation_monitor.detectors import COCO_CATEGORY_IDS, Outputs
from degradation_monitor.method.statistics import KEYS
from degradation_monitor.runs import Manifest, RunLayout, atomic_npz
from degradation_monitor.stages import detectors as stage

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
rfdetr_m = "{root}/rfdetr.pth"

[clean_ap_floor]
yolo11m = 0.0
rfdetr_m = 0.0
"""


class FakeAdapter:
    """A detector stand-in whose maps brighten with the image; the DETR parts when it plays RF-DETR."""
    calls = 0

    def __init__(self, name):
        self.name, self.detr = name, name == "rfdetr_m"
        self.batch_size, self.fit_batch_size = 16, 4

    def __call__(self, arrays, heads=True):
        FakeAdapter.calls += 1
        n = len(arrays)
        level = torch.tensor([a.mean() / 255.0 for a in arrays], dtype=torch.float32)
        generator = torch.Generator().manual_seed(n)

        def maps(channels, side):
            return level.view(n, 1, 1, 1) + torch.rand(n, channels, side, side, generator=generator)

        out = Outputs(levels={"s1": maps(4, 8), "s2": maps(5, 4), "s3": maps(6, 2), "s4": maps(7, 2)},
                      cdf=[maps(3, 8), maps(3, 8), maps(3, 4), maps(3, 2), maps(3, 2)],
                      pooled=maps(16, 1).flatten(1) + 1.0)
        if heads:
            out.scores = np.tile(np.linspace(0.9, 0.0, 100, dtype=np.float32), (n, 1))
            out.labels = np.zeros((n, 100), np.int64)
            out.boxes = np.tile(np.float32([0, 0, 8, 8]), (n, 100, 1))
        if heads and self.detr:
            out.query_logits = np.full((n, 300, 80), -6.0)
            out.query_logits[:, :3, 0] = 2.0
            out.query_boxes = np.full((n, 300, 4), 0.25)
            out.decoder = maps(8, 1).flatten(1)[:, None, :].expand(n, 300, 8).contiguous()
        return out

    def close(self):
        pass


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    """24 clean train images, 3 val images without objects, and the 96 digests of each val image."""
    root = tmp_path_factory.mktemp("images")
    rng = np.random.default_rng(0)
    for folder, count in (("train", 24), ("val", 3)):
        (root / folder).mkdir()
        for index in range(count):
            Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)).save(
                root / folder / f"{index:012d}.jpg")
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
    for name in ("rtdetr.pth", "yolo.pt", "rfdetr.pth"):
        (tmp_path / name).write_bytes(name.encode())
    (tmp_path / "base.toml").write_text(BASE.format(root=tmp_path, images=root))
    (tmp_path / "detectors.toml").write_text(DETECTORS.format(root=tmp_path))
    monkeypatch.setattr(coco, "SPLIT_IMAGES", (("reserved", 1), ("bank", 8), ("zstats", 4)))
    monkeypatch.setattr(stage, "KNN_K_MAX", 5)
    monkeypatch.setattr(stage, "CDF_ZSTAT_IMAGES", 6)
    monkeypatch.setattr(stage, "load_adapter", lambda name, weights, device: FakeAdapter(name))
    FakeAdapter.calls = 0
    reference = RunLayout(tmp_path / "runs" / "coco")  # RT-DETR's run: its digests and its DisCoPatch scores
    rng = np.random.default_rng(1)
    for name, values in digests.items():
        atomic_npz(reference.score_file("detector", name), digests=values)
        atomic_npz(reference.score_file("discopatch", name), dcp=rng.uniform(0, 1, 96))
    Manifest(reference).update(inputs={"discopatch": {"discriminator": "abc"}})
    return stage.load_config(tmp_path / "detectors.toml")


def _write_summary(folder, two_axis, cdf, subsets=("all", "untouched")):
    head = {"two_axis": {"auroc_common": two_axis, "auroc_extra": two_axis - 0.05},
            "cdf": {"auroc_common": cdf, "auroc_extra": cdf - 0.02},
            "saod_top3": {"auroc_common": 0.6, "auroc_extra": 0.6}}
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.json").write_text(json.dumps({
        "images": 5000, "clean_map": 0.45, "headline": {s: head for s in subsets}, "headline_decision": "confirmed",
        "by_severity": {"all": {"two_axis": {"common": [two_axis - 0.1] * 5}}}}))


def test_the_detectors_config_gives_one_settings_per_detector(config):
    assert config.detectors == ("yolo11m", "rfdetr_m")
    settings = config.settings("rfdetr_m")
    assert settings.run == config.run / "rfdetr_m" and settings.checkpoint.name == "rfdetr.pth"
    assert settings.gpu_memory_gib == 8.0 and settings.workers == 0


def test_the_run_root_must_lie_outside_the_reference_run(config, tmp_path):
    """The reference run is only read: a mistyped --run must not put detector folders or the table into it."""
    path, reference = tmp_path / "detectors.toml", config.reference_run
    (tmp_path / "link").symlink_to(reference, target_is_directory=True)
    for run in (reference, reference / "detectors", reference / ".." / "coco", tmp_path / "link" / "detectors"):
        with pytest.raises(ValueError, match="reference run"):
            stage.load_config(path, run=run)
    assert stage.main(["check", "--config", str(path), "--run", str(reference)]) == 2
    assert not (reference / "yolo11m").exists()
    assert stage.load_config(path).run == tmp_path / "runs" / "coco-detectors"  # a sibling named coco-... is fine


def test_the_check_records_each_clean_ap_and_refuses_a_detector_below_its_floor(config):
    stage.check(config)
    manifest = json.loads(config.settings("yolo11m").layout.manifest.read_text())
    assert manifest["check"]["images"] == 3 and manifest["check"]["coco_val_ap"] == 0.0
    with pytest.raises(RuntimeError, match=r"yolo11m 0\.000 \(floor 0\.5\)"):
        stage.check(replace(config, floors={"yolo11m": 0.5, "rfdetr_m": 0.0}))


def test_the_check_records_the_choices_each_folders_results_depend_on(config):
    """The check writes each folder's protocol first, and a protocol that changes later locks the folder out."""
    stage.check(config)
    protocols = {name: json.loads(config.settings(name).layout.manifest.read_text())["protocol"]
                 for name in config.detectors}
    for protocol in protocols.values():
        assert protocol["float32_matmul_precision"] == "highest"
        assert protocol["cudnn_allow_tf32"] == torch.backends.cudnn.allow_tf32  # cuDNN's convolutions, TF32 or not
    rfdetr, yolo = protocols["rfdetr_m"]["adapter"], protocols["yolo11m"]["adapter"]
    assert rfdetr["decoder"] == "transformer.decoder.layers[-1] output, before the decoder's final LayerNorm"
    assert rfdetr["maps"] == "levels and cdf: raw block outputs (0: the embeddings), before the backbone's LayerNorm"
    assert rfdetr["resize"] == "bilinear, no antialiasing"
    assert yolo["letterbox"] == "Ultralytics LetterBox, auto=True, stride 32, centred, grey 114"
    assert yolo["nms"] == "one label per box, as predict does, no time limit"


SWITCHES = {"float32_matmul_precision": lambda: torch.set_float32_matmul_precision("high"),  # as importing rfdetr does
            "cudnn_allow_tf32": lambda: setattr(torch.backends.cudnn, "allow_tf32", False)}


@pytest.fixture
def restored_precision():
    """Every float32 setting a switch changes, put back exactly, so that nothing leaks into later tests."""
    backends = torch.backends
    saved = (torch.get_float32_matmul_precision(), backends.cudnn.allow_tf32, backends.cuda.matmul.fp32_precision,
             backends.mkldnn.matmul.fp32_precision)
    yield
    torch.set_float32_matmul_precision(saved[0])
    backends.cudnn.allow_tf32 = saved[1]
    backends.cuda.matmul.fp32_precision, backends.mkldnn.matmul.fp32_precision = saved[2:]


@pytest.mark.parametrize("setting", list(SWITCHES))
@pytest.mark.parametrize("which", ["check", "fit", "pass"])
def test_a_stage_refuses_to_compute_when_loading_an_adapter_changed_the_precision(config, monkeypatch, which, setting,
                                                                                restored_precision):
    assert (torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32) == ("highest", True)
    if which == "pass":
        stage.fit(config)  # at the precision the protocols record

    def switching(name, weights, device):
        SWITCHES[setting]()
        return FakeAdapter(name)

    monkeypatch.setattr(stage, "load_adapter", switching)
    calls = FakeAdapter.calls
    with pytest.raises(RuntimeError, match=setting):
        {"check": stage.check, "fit": stage.fit, "pass": stage.shared_pass}[which](config)
    assert FakeAdapter.calls == calls  # refused before any forward pass


def test_the_fit_writes_every_reference_and_records_the_adapters_protocol(config):
    stage.fit(config)
    calls = FakeAdapter.calls
    for name in config.detectors:
        layout = config.settings(name).layout
        assert np.load(layout.knn_bank).shape == (24, 16)
        assert layout.cdf_reference.exists() and layout.cdf_zstats.exists()
        assert layout.hashemi_intervals.exists() == (name == "rfdetr_m")
        with np.load(layout.method_bank) as bank, np.load(layout.method_zstats) as zstats:
            assert set(bank.files) == set(KEYS) and bank["means_s1"].shape == (8, 4)
            assert zstats["top_s4"].shape == (4, 7)
        manifest = json.loads(layout.manifest.read_text())
        assert manifest["protocol"]["detector"] == name and "levels" in manifest["protocol"]["adapter"]
        assert manifest["protocol"]["float32_matmul_precision"] == "highest"
        assert {"ultralytics", "rfdetr"} <= set(manifest["environment"]["packages"])
    stage.fit(config)
    assert FakeAdapter.calls == calls


def test_an_interrupted_fit_redoes_only_the_missing_passes(config):
    stage.fit(config)
    calls = FakeAdapter.calls
    layout = config.settings("yolo11m").layout
    layout.cdf_zstats.unlink()
    stage.fit(config)
    assert FakeAdapter.calls == calls + 2 and layout.cdf_zstats.exists()  # 6 sampled images, 4 per batch


def test_the_shared_pass_writes_three_files_per_detector_and_resumes(config):
    stage.fit(config)
    layouts = {name: config.settings(name).layout for name in config.detectors}
    stage.shared_pass(config, first=1)
    assert [len(list(l.scores("method").glob("*.npz"))) for l in layouts.values()] == [1, 1]
    stage.shared_pass(config)
    calls = FakeAdapter.calls
    for name, layout in layouts.items():
        files = sorted(p.name for p in layout.scores("method").glob("*.npz"))
        assert len(files) == 3
        with np.load(layout.score_file("detector", files[0])) as detector:
            assert ("conf_pos" in detector.files) == (name == "rfdetr_m") and detector["knn"].shape == (96, 5)
        with np.load(layout.score_file("activations", files[0])) as activations:
            assert ("hashemi_decoder" in activations.files) == (name == "rfdetr_m")
            assert activations["cdf_stages"].shape == (96, 5)
        with np.load(layout.score_file("method", files[0])) as method:
            assert set(method.files) == set(KEYS) and method["means_s4"].shape == (96, 7)
        assert set(json.loads(layout.manifest.read_text())["inputs"]) == {"detector", "activations", "method"}
    stage.shared_pass(config)
    assert FakeAdapter.calls == calls


def test_the_pass_refuses_corruptions_that_differ_from_the_reference_run(config):
    stage.fit(config)
    first = config.base.dataset.evaluation_images()[0].name
    atomic_npz(RunLayout(config.reference_run).score_file("detector", first), digests=np.array(["0" * 16] * 96))
    with pytest.raises(ValueError, match="corruptions differ from the reference run"):
        stage.shared_pass(config)


def test_the_pass_refuses_scores_from_a_changed_fit(config):
    stage.fit(config)
    stage.shared_pass(config, first=1)
    layout = config.settings("yolo11m").layout
    zstats = json.loads(layout.cdf_zstats.read_text())
    layout.cdf_zstats.write_text(json.dumps({**zstats, "mean": [m + 1.0 for m in zstats["mean"]]}))
    with pytest.raises(ValueError, match="inputs of scores/activations changed"):
        stage.shared_pass(config)


def test_the_report_stage_links_discopatch_and_writes_the_table(config, tmp_path, monkeypatch):
    reported = []

    def write_report(settings, manifest):
        reported.append((settings.run.name, settings.limit))
        _write_summary(settings.layout.report(), 0.9, 0.8, subsets=("all",))

    monkeypatch.setattr(stage, "write_report", write_report)
    _write_summary(RunLayout(config.reference_run).report(), 0.917, 0.821)
    assert stage.main(["report", "--config", str(tmp_path / "detectors.toml"), "--first", "2"]) == 0
    assert reported == [("yolo11m", 2), ("rfdetr_m", 2)]
    linked = sorted(config.settings("yolo11m").layout.scores("discopatch").glob("*.npz"))
    assert len(linked) == 3 and all(os.stat(p).st_nlink >= 2 for p in linked)
    assert "| yolo11m |" in (config.run / "summary.md").read_text()


def test_first_applies_to_the_pass_and_report_only(config, tmp_path):
    with pytest.raises(SystemExit):
        stage.main(["fit", "--config", str(tmp_path / "detectors.toml"), "--first", "2"])


def test_the_cross_detector_table_lists_every_detector(config):
    _write_summary(RunLayout(config.reference_run).report(), 0.917, 0.821)
    _write_summary(config.settings("yolo11m").layout.report(), 0.9, 0.8)
    _write_summary(config.settings("rfdetr_m").layout.report(), 0.86, 0.79, subsets=("all",))  # a smoke report
    rows = stage.cross_table(config)
    assert [r["detector"] for r in rows] == ["rtdetrv2_r18", "yolo11m", "rfdetr_m"]
    assert rows[1]["all_two_axis_auroc_common"] == 0.9 and rows[1]["best_baseline"] == "cdf"
    assert rows[2]["untouched_two_axis_auroc_common"] is None
    assert "| rfdetr_m |" in (config.run / "summary.md").read_text() and (config.run / "summary.csv").exists()
