"""Shared loading for the paper's evidence scripts: the stored statistics, the clean reference and the AUROC helpers.

The scripts read runs/coco/ and never write to it. The first call stacks the 5,000 per-image files of each score
folder into one array per key under PAPER_CACHE (default runs/paper-cache/, git-ignored), so later scripts load in
seconds. Results go to docs/results/paper-evidence/.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from degradation_monitor.corruptions import COMMON_FAMILIES, CONDITIONS, EXTRA_FAMILIES, FAMILIES  # noqa: E402
from degradation_monitor.evaluation.metrics import condition_aurocs  # noqa: E402
from degradation_monitor.evaluation.report import image_sets  # noqa: E402
from degradation_monitor.method.statistics import KEYS  # noqa: E402
from degradation_monitor.runs import SCORE_KEYS, stack  # noqa: E402
from degradation_monitor.settings import load_settings  # noqa: E402

OUT = ROOT / "docs" / "results" / "paper-evidence"
CACHE = Path(os.environ.get("PAPER_CACHE", ROOT / "runs" / "paper-cache"))
SCORED = ("s1", "s2", "s3")
STAGES = ("s1", "s2", "s3", "s4")
FAMILY = np.array([family for family, _ in CONDITIONS])
SEVERITY = np.array([severity for _, severity in CONDITIONS])
COMMON = np.array([f in COMMON_FAMILIES for f in FAMILY])
EXTRA = np.array([f in EXTRA_FAMILIES for f in FAMILY])


def settings():
    return load_settings(ROOT / "configs" / "coco.toml")


def evaluation_names() -> list[str]:
    return [p.name for p in settings().dataset.evaluation_images()]


def cached(folder: str, keys=None) -> dict:
    """KEYS -> (5000, 96, ...) arrays of one score folder, in the evaluation order, memory-mapped from the cache."""
    keys = tuple(keys or (KEYS if folder == "method" else SCORE_KEYS[folder]))
    target = CACHE / folder
    missing = [k for k in keys if not (target / f"{k}.npy").exists()]
    if missing:
        target.mkdir(parents=True, exist_ok=True)
        started = time.time()
        layout = settings().layout
        arrays = stack(layout.scores(folder), evaluation_names(), missing)
        for key, value in arrays.items():
            np.save(target / f"{key}.tmp.npy", value)
            os.replace(target / f"{key}.tmp.npy", target / f"{key}.npy")
        print(f"[cache] stacked {folder} {missing} in {time.time() - started:.0f} s", flush=True)
    return {k: np.load(target / f"{k}.npy", mmap_mode="r") for k in keys}


def statistics() -> dict:
    """The method's stored channel statistics, loaded into memory: KEYS -> float32 (5000, 96, C)."""
    return {k: np.asarray(v) for k, v in cached("method").items()}


def reference() -> tuple[dict, dict]:
    layout = settings().layout
    with np.load(layout.method_bank) as bank, np.load(layout.method_zstats) as zstats:
        return {k: bank[k] for k in KEYS}, {k: zstats[k] for k in KEYS}


def per_condition(scores, rows=None) -> np.ndarray:
    """AUROC of each condition against clean, for scores (images, 96); entry 0 (clean) is nan."""
    scores = np.asarray(scores, dtype=np.float64)
    if rows is not None:
        scores = scores[rows]
    out = np.full(len(CONDITIONS), np.nan)
    out[1:] = condition_aurocs(scores[:, 0], scores[:, 1:].T)
    return out


def summarise(aurocs: np.ndarray) -> dict:
    """Group means of a per-condition AUROC array: common, extra, and each at severity 1."""
    return {"common": float(np.nanmean(aurocs[COMMON])), "extra": float(np.nanmean(aurocs[EXTRA])),
            "common_s1": float(np.nanmean(aurocs[COMMON & (SEVERITY == 1)])),
            "extra_s1": float(np.nanmean(aurocs[EXTRA & (SEVERITY == 1)]))}


def family_severity(aurocs: np.ndarray, family: str, severity: int) -> float:
    return float(aurocs[(FAMILY == family) & (SEVERITY == severity)][0])


def write_csv(name: str, rows: list[dict]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})
    return path


def write_json(name: str, value) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


__all__ = ["CACHE", "COMMON", "COMMON_FAMILIES", "CONDITIONS", "EXTRA", "EXTRA_FAMILIES", "FAMILIES", "FAMILY", "KEYS",
           "OUT", "ROOT", "SCORED", "SEVERITY", "STAGES", "cached", "evaluation_names", "family_severity",
           "image_sets", "per_condition", "reference", "settings", "statistics", "summarise", "write_csv",
           "write_json"]
