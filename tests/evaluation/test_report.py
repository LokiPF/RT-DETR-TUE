import json

import numpy as np
import pytest

from degradation_monitor import corruptions
from degradation_monitor.baselines.contrastive_conf import contrastive_degradation, cross_fit_lambda
from degradation_monitor.datasets.coco import assign_folds
from degradation_monitor.evaluation import metrics, report
from degradation_monitor.method import reference as method_reference
from degradation_monitor.method import scores as method_scores

SEVERITY = np.array([s for _, s in corruptions.CONDITIONS], float)


def _scores(n=30, rows=report.ROWS):
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, (n, 1))
    return {m: base + SEVERITY[None, :] * 2.0 + rng.normal(0, 0.1, (n, 96)) for m in rows}


def _detector(n, seed=1):
    rng = np.random.default_rng(seed)
    knn = np.sort(rng.uniform(0.5, 1.5, (n, 96, 200)), axis=2) + 0.01 * SEVERITY[None, :, None]
    return {"conf_pos": rng.uniform(0.2, 0.9, (n, 96)) - 0.004 * SEVERITY[None],
            "conf_neg": rng.uniform(0.0, 0.1, (n, 96)) + 0.002 * SEVERITY[None],
            "knn": knn.astype(np.float32), "saod_top3": rng.uniform(0, 1, (n, 96)), "saod_min": rng.uniform(0, 1, (n, 96))}


def _tables(scores, n, workers=1, samples=20):
    ap = np.random.default_rng(3).uniform(0, 1, n)
    return report.build_tables(scores, _detector(n), ap, assign_folds(n), 0.5 - 0.05 * SEVERITY, seed=44,
                               samples=samples, workers=workers)


def _interval(low, high):
    return {"low": low, "high": high}


@pytest.fixture
def small_sets(monkeypatch):
    monkeypatch.setattr(report, "SCREEN_IMAGES", 10)
    monkeypatch.setattr(report, "UNTOUCHED_START", 20)


def test_separation_rows_cover_every_row_and_condition_with_the_right_pooling():
    rows = report.separation_rows(_scores(), assign_folds(30), per_fold=("contrastive",))
    assert len(rows) == len(report.ROWS) * 95
    assert {r["pooling"] for r in rows if r["method"] == "contrastive"} == {"fold-averaged"}
    assert {r["pooling"] for r in rows if r["method"] == "knn"} == {"pooled"}
    assert all(r["auroc"] > 0.9 and r["aupr"] > 0.9 for r in rows if r["severity"] >= 3)


def test_aggregate_rows_average_common_and_extra_families_separately():
    rows = report.separation_rows(_scores(), assign_folds(30))
    keys = {(r["method"], r["group"], r["severity"]) for r in report.aggregate_rows(rows)}
    assert {("knn", "common", "all"), ("knn", "extra", 5), ("knn", "all", "all")} <= keys


def test_headline_numbers_give_six_quantities_per_row_and_each_reference_minus_the_others():
    scores = _scores(rows=("two_axis", "level", "cdf"))
    numbers = report.headline_numbers(scores, assign_folds(30))
    for quantity in report.QUANTITIES:
        expected = numbers[f"two_axis:{quantity}"] - numbers[f"cdf:{quantity}"]
        assert numbers[f"two_axis - cdf:{quantity}"] == pytest.approx(expected)
    assert "level - cdf:auroc_common" in numbers and "two_axis - level:auroc_common" in numbers
    assert "level - two_axis:auroc_common" not in numbers and "cdf - level:auroc_common" not in numbers
    clean, degraded = scores["cdf"][:, 0], scores["cdf"][:, corruptions.COMMON_CONDITIONS].T
    assert numbers["cdf:fpr95_common"] == pytest.approx(np.mean([metrics.fpr_at_95_tpr(clean, r) for r in degraded]))


def test_headline_numbers_fold_average_the_rows_asked_for():
    scores, folds = _scores(), assign_folds(30)
    numbers = report.headline_numbers(scores, folds, per_fold=("contrastive",))
    values = scores["contrastive"]
    expected = np.mean([metrics.condition_aurocs(values[folds == f, 0],
                                                 values[folds == f][:, corruptions.COMMON_CONDITIONS].T).mean()
                        for f in range(5)])
    assert numbers["contrastive:auroc_common"] == pytest.approx(expected)


def test_the_image_sets_follow_the_evaluation_order():
    sets = report.image_sets(5000)
    assert {name: len(rows) for name, rows in sets.items()} == {"all": 5000, "untouched": 3030, "held_out": 4800,
                                                                 "screen": 200}
    assert sets["untouched"][0] == 1970 and sets["held_out"][0] == 200 and sets["screen"][-1] == 199
    assert set(report.image_sets(150)) == {"all", "screen"}  # a limited run has no images after the screen


def test_contrastive_scores_fit_lambda_on_the_given_images_only():
    detector, folds = _detector(30), assign_folds(30)
    ap = np.random.default_rng(3).uniform(0, 1, 30)
    scores, per_fold = report.contrastive_scores(detector["conf_pos"], detector["conf_neg"], ap, folds)
    lam, expected = cross_fit_lambda(detector["conf_pos"][:, 0], detector["conf_neg"][:, 0], ap, folds)
    assert per_fold == expected
    np.testing.assert_array_equal(scores, contrastive_degradation(detector["conf_pos"], detector["conf_neg"],
                                                                  lam[:, None]))


def test_baseline_rows_add_discopatch_and_the_activation_monitors_only_when_given():
    detector = _detector(2)
    activations = {"hashemi_decoder": np.full((2, 96), 0.1), "hashemi_encoder": np.full((2, 96), 0.2),
                   "cdf_backbone": np.full((2, 96), 3.0), "cdf_backbone_z": np.full((2, 96), -1.5)}
    rows = report.baseline_rows(detector, discopatch=np.ones((2, 96)), activations=activations)
    assert (rows["hashemi"][0, 0], rows["hashemi_enc"][0, 0], rows["cdf"][0, 0], rows["cdf_sum"][0, 0]) == \
        (0.1, 0.2, -1.5, 3.0)
    assert np.array_equal(rows["knn"], detector["knn"][:, :, 99]) and rows["discopatch"][0, 0] == 1.0
    assert set(report.baseline_rows(detector)) == {"saod_top3", "saod_min", "knn"}


def test_method_rows_score_our_six_rows_from_the_statistics(monkeypatch):
    monkeypatch.setattr(method_reference, "NEIGHBOURS", 3)
    rng = np.random.default_rng(5)
    widths = {1: 2, 2: 3, 3: 4, 4: 5}

    def statistics(*shape):
        return {f"{s}_s{l}": rng.uniform(0.1, 1.0, (*shape, c)) for s in ("means", "top") for l, c in widths.items()}

    test, bank, zstats = statistics(4, 96), statistics(8), statistics(5)
    rows = report.method_rows(test, bank, zstats)
    assert list(rows) == list(report.OURS) and all(v.shape == (4, 96) for v in rows.values())
    np.testing.assert_array_equal(rows["two_axis"], method_scores.two_axis_scores(test, bank, zstats, k=3)[0])


def test_the_level_decision_follows_the_preregistered_rule():
    ahead = {f"level - {other}:auroc_{group}": _interval(0.01, 0.03)
             for other in ("cdf", "global_level") for group in report.GROUPS}
    assert report.level_decision(ahead) == "confirmed"
    behind_cdf = dict(ahead, **{"level - cdf:auroc_common": _interval(-0.004, 0.02)})
    assert report.level_decision(behind_cdf) == "conditioning confirmed, not ahead of the activation CDFs"
    behind = dict(ahead, **{"level - global_level:auroc_extra": _interval(-0.01, 0.002)})
    assert report.level_decision(behind) == "not confirmed"
    assert report.level_decision({k: v for k, v in ahead.items() if "cdf" not in k}) == \
        "unavailable: the activation-CDF scores are missing"
    assert report.level_decision({}) == "unavailable: our method's scores are missing"


def _headline(low_cdf_common=0.05, low_cdf_extra=0.02, low_level=0.01):
    return {"two_axis - cdf:auroc_common": _interval(low_cdf_common, 0.1),
            "two_axis - cdf:auroc_extra": _interval(low_cdf_extra, 0.1),
            "two_axis - level:auroc_common": _interval(low_level, 0.1)}


def test_the_headline_decision_needs_both_image_sets():
    good = _headline()
    assert report.headline_decision({"all": good, "untouched": good}) == "confirmed"
    assert report.headline_decision({"all": good, "untouched": _headline(low_cdf_extra=-0.01)}) == \
        "ahead of the activation CDFs on all images, but not on the untouched images"
    level = _headline(low_level=-0.002)
    assert report.headline_decision({"all": level, "untouched": level}) == \
        "ahead of the activation CDFs, but the flattening adds nothing over the level score on the common families"
    behind = _headline(low_cdf_common=-0.01)
    assert report.headline_decision({"all": behind, "untouched": good}) == "not ahead of the activation CDFs"
    assert report.headline_decision({"all": {}, "untouched": {}}) == \
        "unavailable: the two-axis or the activation-CDF scores are missing"


def test_the_tables_give_every_present_row_on_every_image_set_and_intervals_after_the_screen(small_sets):
    rows = ("two_axis", "level", "global_level", "saod_top3", "knn", "cdf")
    tables, summary = _tables(_scores(40, rows), 40)
    assert summary["image_sets"] == {"all": 40, "untouched": 20, "held_out": 30, "screen": 10}
    assert set(summary["intervals"]) == {"all", "untouched", "held_out"}
    assert set(summary["headline"]["screen"]) == {*rows, "contrastive"}
    assert {"two_axis - cdf:fpr95_extra", "level - global_level:auroc_common",
            "contrastive:aupr_common"} <= set(summary["intervals"]["all"])
    assert summary["headline_decision"] == report.headline_decision(summary["intervals"])
    assert summary["level_decision"] == report.level_decision(summary["intervals"]["held_out"])
    assert set(summary["lambda_per_fold"]) == {"all", "untouched", "held_out", "screen"}
    assert {r["subset"] for r in tables["separation"]} == {"all", "untouched", "held_out", "screen"}
    assert len(tables["conditions"]) == 96 and tables["conditions"][0]["map"] == pytest.approx(0.5)
    assert [r["k"] for r in tables["knn_k"]] == list(report.KNN_KS)


def test_worker_processes_give_the_same_report(small_sets):
    scores = _scores(40, ("two_axis", "level", "cdf"))
    assert _tables(scores, 40, workers=2)[1] == _tables(scores, 40, workers=1)[1]


def test_report_without_the_activation_monitors_says_the_headline_is_unavailable(small_sets, tmp_path):
    rows = ("two_axis", "peak_share", "level", "global_level", "saod_top3", "saod_min", "knn")
    scores = _scores(40, rows)
    scores["global_level"] = np.random.default_rng(9).normal(size=(40, 96))  # no signal, so level is clearly ahead
    tables, summary = _tables(scores, 40)
    assert summary["headline_decision"] == "unavailable: the two-axis or the activation-CDF scores are missing"
    assert summary["level_decision"] == "unavailable: the activation-CDF scores are missing"
    assert {r["method"] for r in tables["separation"]} == {*rows, "contrastive"}
    report.write_outputs(tmp_path, tables, summary)
    text = (tmp_path / "report.md").read_text()
    assert "unavailable" in text and report.LABELS["two_axis"] in text and report.LABELS["cdf"] not in text


def test_the_report_title_counts_the_baseline_families_present(small_sets):
    """A CNN detector has neither ContrastiveConf nor Hashemi et al.; RT-DETR has all six families."""
    title = "# Corruption detection on COCO: our method and {} baselines\n"
    rows = ("two_axis", "level", "global_level", "saod_top3", "saod_min", "knn", "discopatch", "cdf", "cdf_sum")
    cnn = {key: value for key, value in _detector(40).items() if key not in ("conf_pos", "conf_neg")}
    ap = np.random.default_rng(3).uniform(0, 1, 40)
    tables, summary = report.build_tables(_scores(40, rows), cnn, ap, assign_folds(40), 0.5 - 0.05 * SEVERITY,
                                          seed=44, samples=2)
    assert report.markdown(summary, tables).startswith(title.format("four"))
    tables, summary = _tables(_scores(40, rows + ("hashemi", "hashemi_enc")), 40, samples=2)  # and ContrastiveConf
    assert report.markdown(summary, tables).startswith(title.format("six"))


def test_write_outputs_creates_csv_json_and_markdown(small_sets, tmp_path):
    tables, summary = _tables(_scores(40, ("two_axis", "level", "global_level", "cdf", "discopatch")), 40)
    report.write_outputs(tmp_path, tables, summary)
    for name in ("separation", "aggregates", "intervals", "conditions", "knn_k"):
        assert (tmp_path / f"{name}.csv").exists(), name
    assert not (tmp_path / "timing.csv").exists()  # no timing was measured
    assert json.loads((tmp_path / "summary.json").read_text())["images"] == 40
    text = (tmp_path / "report.md").read_text()
    assert text.startswith("# Corruption detection on COCO") and "DisCoPatch" in text and "AUPR common" in text
