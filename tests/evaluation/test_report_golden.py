"""The report's numbers on a fixed synthetic input, recomputed in a few seconds.

tests/test_equivalence.py compares a stored report with the old code's, so it stays green after a later code change.
These tests recompute instead. Their numbers were captured at commit b47587c, whose report reproduced the stored
results bit for bit, so a change to any of these numbers is a change to the method's or the baselines' arithmetic.
"""
import numpy as np
import pytest

from degradation_monitor import corruptions
from degradation_monitor.datasets.coco import assign_folds
from degradation_monitor.evaluation import report
from degradation_monitor.method import reference as method_reference

SEVERITY = np.array([s for _, s in corruptions.CONDITIONS], float)
IMAGES = 40
# (image, condition) of the pinned values of our rows; the two-axis score takes its level arm at the first two and its
# flatter arm at the last two
POSITIONS = ((0, 0), (1, 40), (2, 0), (3, 17))


@pytest.fixture
def small_sets(monkeypatch):
    """40 images then give every image set: screen 0-9, held-out 10-39, untouched 20-39."""
    monkeypatch.setattr(report, "SCREEN_IMAGES", 10)
    monkeypatch.setattr(report, "UNTOUCHED_START", 20)


def _inputs():
    """Three rows with a severity signal, the detector's Conf+, Conf- and kNN distances, and the clean images' AP."""
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, (IMAGES, 1))  # every row sees the same images
    scores = {row: base + strength * SEVERITY[None] + rng.normal(0, 1, (IMAGES, 96))
              for row, strength in (("two_axis", 0.4), ("level", 0.3), ("cdf", 0.2))}
    conf_pos = rng.uniform(0.2, 0.9, (IMAGES, 96)) - 0.004 * SEVERITY[None]
    conf_neg = rng.uniform(0.0, 0.1, (IMAGES, 96)) + 0.002 * SEVERITY[None]
    knn = np.sort(rng.uniform(0.5, 1.5, (IMAGES, 96, 200)), axis=2) + 0.01 * SEVERITY[None, :, None]
    ap = conf_pos[:, 0] - 3.0 * conf_neg[:, 0] + rng.normal(0, 0.05, IMAGES)
    ap[[4, 27]] = np.nan  # images without objects have no AP
    return scores, {"conf_pos": conf_pos, "conf_neg": conf_neg, "knn": knn.astype(np.float32)}, ap


def _build():
    scores, detector, ap = _inputs()
    return report.build_tables(scores, detector, ap, assign_folds(IMAGES), 0.5 - 0.05 * SEVERITY, seed=44, samples=10,
                               workers=1)


def _statistics():
    """Channel statistics of 4 test images under 96 conditions, a bank of 8 and 5 z-statistics images."""
    rng = np.random.default_rng(5)
    widths = {1: 2, 2: 3, 3: 4, 4: 5}

    def statistics(*shape):
        return {f"{s}_s{l}": rng.uniform(0.1, 1.0, (*shape, c)) for s in ("means", "top") for l, c in widths.items()}

    return statistics(4, 96), statistics(8), statistics(5)


def test_the_tables_keep_their_numbers_on_a_fixed_input(small_sets):
    tables, summary = _build()
    # ContrastiveConf takes both branches: its lambdas agree over the folds of the held-out images only.
    assert summary["lambda_folds_agree"] == {"all": False, "untouched": False, "held_out": True, "screen": False}
    assert summary["lambda_per_fold"]["all"] == pytest.approx({0: 3.0, 1: 4.0, 2: 3.0, 3: 3.0, 4: 3.0}, abs=1e-12)
    assert summary["headline"]["all"]["level"] == pytest.approx(
        {"auroc_common": 0.7356666666666667, "aupr_common": 0.7737466932022063, "fpr95_common": 0.8103333333333333,
         "auroc_extra": 0.7278749999999999, "aupr_extra": 0.7733470735782199, "fpr95_extra": 0.8625}, abs=1e-12)
    intervals = summary["intervals"]
    assert intervals["all"]["contrastive:auroc_common"] == pytest.approx(
        {"point": 0.4207916666666667, "low": 0.3578867315130377, "high": 0.4720190776212888}, abs=1e-12)
    assert intervals["untouched"]["two_axis - cdf:auroc_common"] == pytest.approx(
        {"point": 0.15189999999999992, "low": 0.09594083333333323, "high": 0.1750283333333333}, abs=1e-12)
    rows = {(r["subset"], r["method"], r["family"], r["severity"]): r for r in tables["separation"]}
    averaged, pooled = rows["all", "contrastive", "fog", 3], rows["held_out", "contrastive", "fog", 3]
    assert averaged["pooling"] == "fold-averaged" and averaged["auroc"] == pytest.approx(0.490625, abs=1e-12)
    assert pooled["pooling"] == "pooled" and pooled["auroc"] == pytest.approx(0.4122222222222222, abs=1e-12)


def test_our_rows_keep_their_numbers_on_fixed_statistics(monkeypatch):
    monkeypatch.setattr(method_reference, "NEIGHBOURS", 3)
    rows = report.method_rows(*_statistics())
    assert [rows["two_axis"][p] for p in POSITIONS] == pytest.approx(
        [3.156216051715737, -1.1925544702157014, 2.0594309127824437, -0.4328757830450244], abs=1e-12)
    assert [rows["means_own"][p] for p in POSITIONS] == pytest.approx(
        [1.6772955174621145, 1.7233257693822255, 0.08829165086861845, -6.185606162501102], abs=1e-12)
