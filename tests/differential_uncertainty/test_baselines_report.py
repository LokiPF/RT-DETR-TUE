import csv
import json

import numpy as np
import pytest

from degradation_monitor import corruptions
from degradation_monitor.datasets import coco
from differential_uncertainty.baselines import report

SEVERITY = np.array([s for _, s in corruptions.CONDITIONS], float)


def _scores(n=30):
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, (n, 1))
    return {m: base + SEVERITY[None, :] * 2.0 + rng.normal(0, 0.1, (n, 96)) for m in report.METHODS}


def test_separation_rows_cover_every_method_and_condition_with_the_right_pooling():
    folds = coco.assign_folds(30)
    rows = report.separation_rows(_scores(), folds, per_fold_methods=("contrastive",))
    assert len(rows) == len(report.METHODS) * 95
    assert {r["pooling"] for r in rows if r["method"] == "contrastive"} == {"fold-averaged"}
    assert {r["pooling"] for r in rows if r["method"] == "knn"} == {"pooled"}
    assert all(r["auroc"] > 0.9 and r["aupr"] > 0.9 for r in rows if r["severity"] >= 3)


def test_aggregate_rows_average_common_and_extra_families_separately():
    rows = report.separation_rows(_scores(), coco.assign_folds(30))
    keys = {(r["method"], r["group"], r["severity"]) for r in report.aggregate_rows(rows)}
    assert {("knn", "common", "all"), ("knn", "extra", 5), ("knn", "all", "all")} <= keys


def test_headline_numbers_include_pairwise_differences_and_no_harm():
    numbers = report.headline_numbers(_scores())
    difference = numbers["saod_top3 - knn:auroc_common"]
    assert difference == pytest.approx(numbers["saod_top3:auroc_common"] - numbers["knn:auroc_common"])
    assert not any(key.endswith((":rho_within", ":rho_condition_lrp", ":aurc_all")) for key in numbers)


def test_headline_numbers_fold_average_the_methods_asked_for():
    scores, folds = _scores(), coco.assign_folds(30)
    numbers = report.headline_numbers(scores, folds, per_fold_methods=("contrastive",))
    values = scores["contrastive"]
    expected = np.mean([report.metrics.condition_aurocs(values[folds == f, 0], values[folds == f][:, report.COMMON].T).mean()
                        for f in range(5)])
    assert numbers["contrastive:auroc_common"] == pytest.approx(expected)


def test_write_outputs_creates_csv_json_and_markdown(tmp_path):
    rows = report.separation_rows(_scores(), coco.assign_folds(30))
    report.write_outputs(tmp_path, {"separation": rows, "aggregates": report.aggregate_rows(rows)},
                         {"lambda_per_fold": {"0": 5.0}, "clean_map": 0.48})
    assert (tmp_path / "separation.csv").exists()
    assert json.loads((tmp_path / "summary.json").read_text())["clean_map"] == 0.48
    text = (tmp_path / "report.md").read_text()
    assert "ContrastiveConf" in text and "DisCoPatch" in text and "AUPR" in text


def _tiny_run(tmp_path, monkeypatch):
    """15 fixture images through the test phase with a fake detector; returns the settings."""
    import torch
    from PIL import Image
    from differential_uncertainty.baselines import pipeline

    val = tmp_path / "val"
    val.mkdir()
    rng = np.random.default_rng(7)
    images, annotations = [], []
    for index in range(15):  # enough that every bootstrap fold complement keeps >= 3 images
        name = f"{index:012d}.jpg"
        Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)).save(val / name)
        images.append({"id": index + 1, "file_name": name, "width": 64, "height": 48})
        annotations.append({"id": 100 + index, "image_id": index + 1, "category_id": 1,
                            "bbox": [8, 6, 24, 18], "area": 432, "iscrowd": 0})
    (tmp_path / "ann.json").write_text(json.dumps({
        "images": images, "annotations": annotations,
        "categories": [{"id": c, "name": f"c{c}"} for c in range(1, 81)]}))

    class FakeTap:
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

    monkeypatch.setattr(pipeline, "DetectorTap", FakeTap)
    monkeypatch.setattr(pipeline, "_load_bank", lambda _s, _d: torch.nn.functional.normalize(torch.randn(256, 512), dim=1))
    monkeypatch.setattr(report, "BOOTSTRAP_SAMPLES", 5)
    settings = pipeline.Settings(output=tmp_path / "out", checkpoint=tmp_path / "c.pth", train_images=tmp_path,
                                 val_images=val, annotations=tmp_path / "ann.json", discopatch_root=tmp_path,
                                 limit=15, workers=0, device="cpu")
    pipeline.run_phase("test", settings)
    return settings


def test_build_report_end_to_end_on_a_tiny_fixture(tmp_path, monkeypatch):
    from differential_uncertainty.baselines import pipeline
    settings = _tiny_run(tmp_path, monkeypatch)
    pipeline.run_phase("report", settings)

    results = settings.output / "results"
    summary = json.loads((results / "summary.json").read_text())
    assert summary["images"] == 15 and len(summary["lambda_per_fold"]) == 5
    assert summary["discopatch_included"] is False and summary["activation_monitors_included"] is False
    for name in ("separation", "aggregates", "conditions", "intervals", "differences", "knn_k"):
        assert (results / f"{name}.csv").exists(), name
    assert "ContrastiveConf" in (results / "report.md").read_text()
    header = (results / "conditions.csv").read_text().splitlines()[0].split(",")
    assert "map" in header and "images_undefined_lrp" not in header


def test_build_report_includes_the_activation_monitors_when_scored(tmp_path, monkeypatch):
    from differential_uncertainty.baselines import pipeline
    settings = _tiny_run(tmp_path, monkeypatch)
    folder = settings.output / "test_activation"
    folder.mkdir()
    rng = np.random.default_rng(8)
    for path in sorted((settings.output / "test").glob("*.npz")):
        np.savez(folder / path.name, hashemi_decoder=rng.uniform(0, 1, 96), hashemi_encoder=rng.uniform(0, 1, 96),
                 hashemi_encoder_maps=rng.uniform(0, 1, (96, 3)), cdf_backbone=rng.uniform(0, 50, 96),
                 cdf_backbone_z=rng.normal(0, 3, 96), cdf_stages=rng.uniform(0, 10, (96, 5)))
    pipeline.run_phase("report", settings)

    results = settings.output / "results"
    summary = json.loads((results / "summary.json").read_text())
    assert summary["activation_monitors_included"] is True
    assert summary["hashemi_k"] == 2.0 and summary["cdf_bins"] == 1000
    with (results / "separation.csv").open() as handle:
        methods = {row["method"] for row in csv.DictReader(handle)}
    assert {"hashemi", "hashemi_enc", "cdf", "cdf_sum"} <= methods
    text = (results / "report.md").read_text()
    assert "Hashemi et al., decoder queries" in text and "Activation CDFs (Becker et al., ICPR 2026)" in text


def test_method_scores_add_the_activation_monitors_only_when_given():
    test = {"saod_top3": np.zeros((2, 96)), "saod_min": np.zeros((2, 96)), "conf_pos": np.ones((2, 96)),
            "conf_neg": np.zeros((2, 96)), "knn": np.zeros((2, 96, 200))}
    activation = {"hashemi_decoder": np.full((2, 96), 0.1), "hashemi_encoder": np.full((2, 96), 0.2),
                  "cdf_backbone": np.full((2, 96), 3.0), "cdf_backbone_z": np.full((2, 96), -1.5)}
    scores = report.method_scores(test, None, np.ones(2), activation=activation)
    assert (scores["hashemi"][0, 0], scores["hashemi_enc"][0, 0]) == (0.1, 0.2)
    assert (scores["cdf"][0, 0], scores["cdf_sum"][0, 0]) == (-1.5, 3.0)
    assert "hashemi" not in report.method_scores(test, None, np.ones(2))


def test_differences_cover_every_separation_metric_for_both_family_groups():
    numbers = report.headline_numbers(_scores())
    for metric in report.SEPARATION:
        for group in ("common", "extra"):
            key = f"saod_top3 - knn:{metric}_{group}"
            assert numbers[key] == pytest.approx(numbers[f"saod_top3:{metric}_{group}"] - numbers[f"knn:{metric}_{group}"])


def test_write_outputs_takes_other_methods_labels_and_title(tmp_path):
    scores = {"mine": np.random.default_rng(0).normal(size=(6, len(corruptions.CONDITIONS)))}
    rows = report.separation_rows(scores, coco.assign_folds(6))
    report.write_outputs(tmp_path, {"separation": rows, "aggregates": report.aggregate_rows(rows)}, {"x": 1},
                         methods=("mine",), labels={"mine": "My method"}, title="# Pilot")
    text = (tmp_path / "report.md").read_text()
    assert text.startswith("# Pilot") and "My method" in text
