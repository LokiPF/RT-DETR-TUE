import json

import numpy as np
import pytest
from PIL import Image

from degradation_monitor.evaluation import report as tables
from degradation_monitor.method import reference as method_reference
from degradation_monitor.runs import atomic_npz
from degradation_monitor.settings import Settings
from degradation_monitor.stages import baselines as baseline_stages
from degradation_monitor.stages import run_stage

IMAGES = 20
WIDTHS = (2, 3, 4, 5)


class QualityTap:
    """Detector stand-in: one box whose confidence falls as the image gets noisier."""

    def __init__(self, *_a, **_k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def run(self, arrays, batch_size=32):
        n = len(arrays)
        quality = np.array([1.0 - a.std() / 128.0 for a in arrays])
        logits = np.full((n, 300, 80), -6.0)
        logits[:, 0, 0] = 6.0 * quality - 2.0
        boxes = np.tile([0.3125, 0.3125, 0.375, 0.375], (n, 300, 1))
        pooled = np.random.default_rng(0).normal(size=(n, 512)) + quality[:, None]
        return logits, boxes, pooled


@pytest.fixture
def run(tmp_path, monkeypatch):
    """IMAGES fixture images with one annotated object each, after the detector pass."""
    val = tmp_path / "val"
    val.mkdir()
    rng = np.random.default_rng(7)
    images, annotations = [], []
    for index in range(IMAGES):
        name = f"{index:012d}.jpg"
        Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)).save(val / name)
        images.append({"id": index + 1, "file_name": name, "width": 64, "height": 48})
        annotations.append({"id": 100 + index, "image_id": index + 1, "category_id": 1,
                            "bbox": [8, 6, 24, 18], "area": 432, "iscrowd": 0})
    (tmp_path / "ann.json").write_text(json.dumps({
        "images": images, "annotations": annotations,
        "categories": [{"id": c, "name": f"c{c}"} for c in range(1, 81)]}))
    (tmp_path / "ckpt.pth").write_bytes(b"weights")
    monkeypatch.setattr(baseline_stages, "DetectorTap", QualityTap)
    monkeypatch.setattr(tables, "BOOTSTRAP_SAMPLES", 5)
    monkeypatch.setattr(tables, "SCREEN_IMAGES", 5)
    monkeypatch.setattr(tables, "UNTOUCHED_START", 8)
    settings = Settings(run=tmp_path / "run", checkpoint=tmp_path / "ckpt.pth", train_images=tmp_path,
                        val_images=val, annotations=tmp_path / "ann.json", discopatch_root=tmp_path,
                        limit=IMAGES, workers=0, device="cpu")
    settings.layout.knn_bank.parent.mkdir(parents=True)
    np.save(settings.layout.knn_bank, rng.normal(size=(256, 512)).astype(np.float16))
    run_stage("detector-pass", settings)
    return settings


def _statistics(rng, *shape):
    return {f"{s}_s{l}": rng.uniform(0.1, 1.0, (*shape, c)).astype(np.float32)
            for s in ("means", "top") for l, c in zip(range(1, 5), WIDTHS)}


def test_the_report_stage_writes_every_table_for_a_run_with_only_the_detector_pass(run):
    run_stage("report", run)
    folder = run.layout.report()
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["images"] == IMAGES
    assert summary["image_sets"] == {"all": 20, "untouched": 12, "held_out": 15, "screen": 5}
    assert len(summary["lambda_per_fold"]["all"]) == 5
    assert summary["rows"] == ["saod_top3", "saod_min", "contrastive", "knn"]
    assert summary["headline_decision"] == "unavailable: the two-axis or the activation-CDF scores are missing"
    assert summary["level_decision"] == "unavailable: our method's scores are missing"
    for name in ("separation", "aggregates", "intervals", "conditions", "knn_k"):
        assert (folder / f"{name}.csv").exists(), name
    header = (folder / "conditions.csv").read_text().splitlines()[0].split(",")
    assert "map" in header and "mean_contrastive" in header
    assert "ContrastiveConf" in (folder / "report.md").read_text()


def test_the_report_stage_adds_the_activation_monitors_and_our_method_when_they_were_run(run, monkeypatch):
    monkeypatch.setattr(method_reference, "NEIGHBOURS", 3)
    rng = np.random.default_rng(8)
    for path in sorted(run.layout.scores("detector").glob("*.npz")):
        atomic_npz(run.layout.score_file("activations", path.name), hashemi_decoder=rng.uniform(0, 1, 96),
                   hashemi_encoder=rng.uniform(0, 1, 96), hashemi_encoder_maps=rng.uniform(0, 1, (96, 3)),
                   cdf_backbone=rng.uniform(0, 50, 96), cdf_backbone_z=rng.normal(0, 3, 96),
                   cdf_stages=rng.uniform(0, 10, (96, 5)))
        atomic_npz(run.layout.score_file("method", path.name), **_statistics(rng, 96))
    atomic_npz(run.layout.method_bank, **_statistics(rng, 6))
    atomic_npz(run.layout.method_zstats, **_statistics(rng, 4))
    run_stage("report", run)
    summary = json.loads((run.layout.report() / "summary.json").read_text())
    assert {"two_axis", "level", "means_own", "hashemi", "hashemi_enc", "cdf", "cdf_sum"} <= set(summary["rows"])
    assert set(summary["inputs"]["method"]) == {"method_bank", "method_zstats"}
    assert summary["headline_decision"] == tables.headline_decision(summary["intervals"])
    assert summary["level_decision"] == tables.level_decision(summary["intervals"]["held_out"])
    text = (run.layout.report() / "report.md").read_text()
    assert tables.LABELS["hashemi"] in text and tables.LABELS["two_axis"] in text


def test_the_report_stage_needs_the_method_reference_when_the_method_pass_ran(run):
    rng = np.random.default_rng(8)
    for path in sorted(run.layout.scores("detector").glob("*.npz")):
        atomic_npz(run.layout.score_file("method", path.name), **_statistics(rng, 96))
    with pytest.raises(ValueError, match="run the method-reference stage first"):
        run_stage("report", run)


def test_the_report_stage_leaves_out_the_rows_a_detector_does_not_have(run):
    """A CNN detector: no ContrastiveConf and no Hashemi; the activation CDFs alone among the activation monitors."""
    rng = np.random.default_rng(11)
    for path in sorted(run.layout.scores("detector").glob("*.npz")):
        with np.load(path) as data:
            kept = {k: data[k] for k in data.files if k not in ("conf_pos", "conf_neg")}
        atomic_npz(path, **kept)
        stages = rng.uniform(0, 1, (96, 5))
        atomic_npz(run.layout.score_file("activations", path.name), cdf_backbone=stages.sum(1),
                   cdf_backbone_z=stages.sum(1), cdf_stages=stages)
    run_stage("report", run)
    summary = json.loads((run.layout.report() / "summary.json").read_text())
    assert {"saod_top3", "saod_min", "knn", "cdf", "cdf_sum"} <= set(summary["rows"])
    assert not {"contrastive", "hashemi", "hashemi_enc"} & set(summary["rows"])
    assert summary["lambda_folds_agree"] == {}


def test_the_report_adds_extra_rows_and_writes_where_it_is_told(run, tmp_path, monkeypatch):
    from degradation_monitor.runs import Manifest
    from degradation_monitor.stages.report import write_report

    monkeypatch.setattr(method_reference, "NEIGHBOURS", 3)
    rng = np.random.default_rng(9)
    for path in sorted(run.layout.scores("detector").glob("*.npz")):
        atomic_npz(run.layout.score_file("method", path.name), **_statistics(rng, 96))
    atomic_npz(run.layout.method_bank, **_statistics(rng, 6))
    atomic_npz(run.layout.method_zstats, **_statistics(rng, 4))
    extra = {"niqe": rng.uniform(0, 1, (IMAGES, 96)), "clipiqa": rng.uniform(0, 1, (IMAGES, 96))}
    out = tmp_path / "iqa-report"
    write_report(run, Manifest(run.layout), out=out, extra_rows=extra, extra_inputs={"iqa": {"scores": "here"}})
    summary = json.loads((out / "summary.json").read_text())
    assert {"two_axis", "niqe", "clipiqa"} <= set(summary["rows"])
    assert "two_axis - niqe:auroc_common" in summary["intervals"]["all"]
    assert summary["inputs"]["iqa"] == {"scores": "here"}
    assert not run.layout.report().exists()  # the run folder itself gets no report


def test_the_report_refuses_extra_rows_it_cannot_take(run, tmp_path):
    from degradation_monitor.runs import Manifest
    from degradation_monitor.stages.report import write_report

    for rows, message in (({"psnr": np.zeros((IMAGES, 96))}, "cannot take from elsewhere: psnr"),
                          ({"knn": np.zeros((IMAGES, 96))}, "cannot take from elsewhere: knn"),
                          ({"niqe": np.zeros((IMAGES - 1, 96))}, "one per image and condition: niqe")):
        with pytest.raises(ValueError, match=message):
            write_report(run, Manifest(run.layout), out=tmp_path, extra_rows=rows)
