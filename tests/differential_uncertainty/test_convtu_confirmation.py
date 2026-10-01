import json

import numpy as np
import pytest

from differential_uncertainty.baselines import metrics, report
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
