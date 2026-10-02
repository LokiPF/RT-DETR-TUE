"""The clean report on the converted run folder reproduces the old code's numbers (spec, success criterion 3)."""
import csv
import json
from pathlib import Path

import pytest

from degradation_monitor.evaluation.report import QUANTITIES
from degradation_monitor.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]
REPORT = load_settings(ROOT / "configs" / "coco.toml").layout.report()
OLD_BASELINES = ROOT / "docs" / "results" / "coco-baselines"
OLD_CONFIRMATION = ROOT / "docs" / "results" / "conv-tu-conditioned" / "summary.json"
RENAMED = {"conditioned": "level", "global_s123": "global_level", "ch_means_knn": "means_knn",
           "ch_means_own": "means_own"}
pytestmark = pytest.mark.skipif(not (REPORT / "summary.json").exists(),
                                reason="needs the report of the converted run folder (Task 15)")


def _rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _renamed(quantity: str) -> str:
    """An old quantity name with the renamed rows, e.g. 'two_axis - conditioned:auroc_common'."""
    rows, _, metric = quantity.partition(":")
    return " - ".join(RENAMED.get(row, row) for row in rows.split(" - ")) + ":" + metric


def _summary() -> dict:
    return json.loads((REPORT / "summary.json").read_text())


def _flat(value, path=()) -> dict:
    """{path: number} of a nested dict or list; pytest.approx compares only flat mappings."""
    if isinstance(value, dict):
        return {k: v for key, item in value.items() for k, v in _flat(item, (*path, key)).items()}
    if isinstance(value, list):
        return {k: v for index, item in enumerate(value) for k, v in _flat(item, (*path, index)).items()}
    return {path: value}


def test_the_baselines_per_condition_separation_is_reproduced():
    old = {(r["method"], r["family"], r["severity"]): r for r in _rows(OLD_BASELINES / "separation.csv")}
    new = {(r["method"], r["family"], r["severity"]): r for r in _rows(REPORT / "separation.csv")
           if r["subset"] == "all"}
    assert len(old) == 9 * 95
    for key, row in old.items():
        assert new[key]["pooling"] == row["pooling"], key
        for metric in ("auroc", "aupr", "fpr95"):
            assert float(new[key][metric]) == pytest.approx(float(row[metric]), abs=1e-12), (key, metric)


def test_the_baselines_separation_intervals_are_reproduced():
    new = _summary()["intervals"]["all"]
    old = [r for r in _rows(OLD_BASELINES / "intervals.csv") if r["quantity"].split(":")[1] in QUANTITIES]
    assert len(old) == 9 * len(QUANTITIES)
    for row in old:
        for field in ("point", "low", "high"):
            assert new[row["quantity"]][field] == pytest.approx(float(row[field]), abs=1e-12), (row["quantity"], field)


def test_the_confirmation_is_reproduced():
    old, new = json.loads(OLD_CONFIRMATION.read_text()), _summary()
    assert (new["headline_decision"], new["level_decision"]) == (old["headline_decision"], old["decision"])
    assert new["image_sets"] == {"all": old["images"], "untouched": old["untouched_images"],
                                 "held_out": old["held_out_images"], "screen": old["screen_images"]}
    for part in ("headline", "by_severity", "by_family"):
        for subset, rows in old[part].items():
            for row, values in rows.items():
                got = _flat(new[part][subset][RENAMED.get(row, row)])
                assert got == pytest.approx(_flat(values), abs=1e-12), (part, subset, row)
    for subset, quantities in old["intervals"].items():
        for quantity, bounds in quantities.items():
            got = new["intervals"][subset][_renamed(quantity)]
            assert (got["low"], got["high"]) == pytest.approx((bounds["low"], bounds["high"]), abs=1e-12), \
                (subset, quantity)
