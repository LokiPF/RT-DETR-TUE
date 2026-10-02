import json

import numpy as np
import pytest

from degradation_monitor.evaluation import metrics
from differential_uncertainty.baselines import report
from differential_uncertainty.convtu import confirmation


def test_group_aurocs_match_the_per_condition_metric():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(30, 96)) + np.linspace(0, 1, 96)[None]
    common, extra = confirmation.group_aurocs(scores, np.arange(30))
    assert common == pytest.approx(np.mean([metrics.auroc(scores[:, 0], scores[:, c]) for c in report.COMMON]))
    assert extra == pytest.approx(np.mean([metrics.auroc(scores[:, 0], scores[:, c]) for c in report.EXTRA]))
    rows = np.array([0, 0, 3, 7])
    assert confirmation.group_aurocs(scores, rows)[0] == pytest.approx(
        np.mean([metrics.auroc(scores[rows, 0], scores[rows, c]) for c in report.COMMON]))


def _interval(low, high):
    return {"low": low, "high": high}


def test_decision_follows_the_preregistered_rule():
    ahead = {f"conditioned - {other}:{group}": _interval(0.01, 0.03)
             for other in ("cdf", "global_s123") for group in ("auroc_common", "auroc_extra")}
    assert confirmation.decision(ahead) == "confirmed"
    level = dict(ahead, **{"conditioned - cdf:auroc_common": _interval(-0.004, 0.02)})
    assert confirmation.decision(level) == "conditioning confirmed, not ahead of the activation CDFs"
    behind = dict(ahead, **{"conditioned - global_s123:auroc_extra": _interval(-0.01, 0.002)})
    assert confirmation.decision(behind) == "not confirmed"
    assert confirmation.decision({k: v for k, v in ahead.items() if "cdf" not in k}) == \
        "unavailable: the activation-CDF scores are missing"


def _headline(low_cdf_common=0.05, low_cdf_extra=0.02, low_level=0.01):
    return {"two_axis - cdf:auroc_common": _interval(low_cdf_common, 0.1),
            "two_axis - cdf:auroc_extra": _interval(low_cdf_extra, 0.1),
            "two_axis - conditioned:auroc_common": _interval(low_level, 0.1)}


def test_headline_decision_needs_both_image_sets():
    good = _headline()
    assert confirmation.headline_decision({"all": good, "untouched": good}) == "confirmed"
    assert confirmation.headline_decision({"all": good, "untouched": _headline(low_cdf_extra=-0.01)}) == \
        "ahead of the activation CDFs on all images, but not on the untouched images"
    level = _headline(low_level=-0.002)
    assert confirmation.headline_decision({"all": level, "untouched": level}) == \
        "ahead of the activation CDFs, but the flattening adds nothing over the level score on the common families"
    behind = _headline(low_cdf_common=-0.01)
    assert confirmation.headline_decision({"all": behind, "untouched": good}) == "not ahead of the activation CDFs"
    assert confirmation.headline_decision({"all": {}, "untouched": {}}) == \
        "unavailable: the two-axis or the activation-CDF scores are missing"


def test_bootstrap_compares_every_named_reference_row():
    rng = np.random.default_rng(2)
    base = rng.normal(size=(20, 96))
    scores = {"two_axis": base + 0.4 * (np.arange(96) > 0), "conditioned": base + 0.2 * (np.arange(96) > 0), "cdf": base}
    out = confirmation.bootstrap_intervals(scores, np.arange(20), samples=30, seed=44,
                                           references=("two_axis", "conditioned"))
    assert {"two_axis - cdf:auroc_common", "two_axis - conditioned:auroc_extra",
            "conditioned - cdf:auroc_common"} <= set(out)
    assert "conditioned - two_axis:auroc_common" not in out  # each pair once, from the earlier reference


def test_bootstrap_reports_each_row_and_each_difference():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(25, 96))
    scores = {"conditioned": base + 0.5 * (np.arange(96) > 0), "cdf": base, "global_s123": base * 0.9}
    out = confirmation.bootstrap_intervals(scores, np.arange(25), samples=50, seed=44)
    assert set(out) >= {"conditioned:auroc_common", "cdf:auroc_extra", "conditioned - cdf:auroc_common",
                        "conditioned - global_s123:auroc_extra"}
    assert out["conditioned - cdf:auroc_common"]["low"] > 0
    again = confirmation.bootstrap_intervals(scores, np.arange(25), samples=50, seed=44)
    assert again == out
