import json

import numpy as np
import pytest

from differential_uncertainty.baselines import protocol, report

SEVERITY = np.array([s for _, s in protocol.CONDITIONS], float)


def _scores(n=30):
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, (n, 1))
    return {m: base + SEVERITY[None, :] * 2.0 + rng.normal(0, 0.1, (n, 96)) for m in report.METHODS}


def test_separation_rows_cover_every_method_and_condition_with_the_right_pooling():
    folds = protocol.assign_folds(30)
    rows = report.separation_rows(_scores(), folds, per_fold_methods=("contrastive",))
    assert len(rows) == len(report.METHODS) * 95
    assert {r["pooling"] for r in rows if r["method"] == "contrastive"} == {"fold-averaged"}
    assert {r["pooling"] for r in rows if r["method"] == "knn"} == {"pooled"}
    assert all(r["auroc"] > 0.9 and r["aupr"] > 0.9 for r in rows if r["severity"] >= 3)


def test_aggregate_rows_average_common_and_extra_families_separately():
    rows = report.separation_rows(_scores(), protocol.assign_folds(30))
    keys = {(r["method"], r["group"], r["severity"]) for r in report.aggregate_rows(rows)}
    assert {("knn", "common", "all"), ("knn", "extra", 5), ("knn", "all", "all")} <= keys


def test_harm_rows_correlations_and_aurc_pools():
    scores = _scores()
    condition_map = 0.5 - 0.05 * SEVERITY
    lrp = np.tile(1.0 - condition_map, (30, 1)) + np.random.default_rng(2).normal(0, 0.01, (30, 96))
    harm, pools = report.harm_rows(scores, lrp, condition_map)
    knn = next(r for r in harm if r["method"] == "knn")
    assert knn["rho_condition_map"] < -0.9 and knn["rho_condition_lrp"] > 0.9
    assert len(pools) == len(report.METHODS) * (1 + 5 + 19)
    assert all(p["aurc"] >= p["aurc_oracle"] - 1e-12 for p in pools)


def test_headline_numbers_include_pairwise_differences():
    rng = np.random.default_rng(4)
    numbers = report.headline_numbers(_scores(), rng.uniform(0, 1, (30, 96)))
    difference = numbers["saod_top3 - knn:auroc_common"]
    assert difference == pytest.approx(numbers["saod_top3:auroc_common"] - numbers["knn:auroc_common"])


def test_headline_numbers_fold_average_the_methods_asked_for():
    scores, folds = _scores(), protocol.assign_folds(30)
    lrp = np.random.default_rng(5).uniform(0, 1, (30, 96))
    numbers = report.headline_numbers(scores, lrp, folds, per_fold_methods=("contrastive",))
    values = scores["contrastive"]
    expected = np.mean([report.metrics.condition_aurocs(values[folds == f, 0], values[folds == f][:, report.COMMON].T).mean()
                        for f in range(5)])
    assert numbers["contrastive:auroc_common"] == pytest.approx(expected)


def test_write_outputs_creates_csv_json_and_markdown(tmp_path):
    rows = report.separation_rows(_scores(), protocol.assign_folds(30))
    report.write_outputs(tmp_path, {"separation": rows, "aggregates": report.aggregate_rows(rows)},
                         {"lambda_per_fold": {"0": 5.0}, "lrp_threshold": 0.3})
    assert (tmp_path / "separation.csv").exists()
    assert json.loads((tmp_path / "summary.json").read_text())["lrp_threshold"] == 0.3
    text = (tmp_path / "report.md").read_text()
    assert "ContrastiveConf" in text and "DisCoPatch" in text and "AUPR" in text


def test_build_report_end_to_end_on_a_tiny_fixture(tmp_path, monkeypatch):
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
    pipeline.run_phase("report", settings)

    results = settings.output / "results"
    summary = json.loads((results / "summary.json").read_text())
    assert summary["images"] == 15 and len(summary["lambda_per_fold"]) == 5
    assert summary["discopatch_included"] is False
    for name in ("separation", "aggregates", "harm", "aurc_pools", "conditions", "intervals", "differences", "knn_k"):
        assert (results / f"{name}.csv").exists(), name
    assert "ContrastiveConf" in (results / "report.md").read_text()
    header = (results / "conditions.csv").read_text().splitlines()[0].split(",")
    assert "images_undefined_lrp" in header


def test_differences_cover_every_separation_metric_for_both_family_groups():
    numbers = report.headline_numbers(_scores(), np.random.default_rng(6).uniform(0, 1, (30, 96)))
    for metric in report.SEPARATION:
        for group in ("common", "extra"):
            key = f"saod_top3 - knn:{metric}_{group}"
            assert numbers[key] == pytest.approx(numbers[f"saod_top3:{metric}_{group}"] - numbers[f"knn:{metric}_{group}"])
