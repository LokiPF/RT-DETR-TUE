"""Each run's two-axis score and its two arms per corruption condition (AUROC): which arm catches which corruption.

    python scripts/paper/detector_arms.py RUN [RUN ...] --out docs/results/coco-detectors/arms.csv

Each RUN is a run folder laid out like runs/coco/: scores/method/ and reference/method/. The evaluation images are
configs/coco.toml's. The arms come from the package's own two_axis_scores, so they are exactly the two halves of the
reported two-axis score. Runs are read one after another; Faster R-CNN's needs up to about 50 GiB of RAM.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from degradation_monitor.corruptions import CONDITIONS  # noqa: E402
from degradation_monitor.evaluation.metrics import condition_aurocs  # noqa: E402
from degradation_monitor.method.scores import two_axis_scores  # noqa: E402
from degradation_monitor.method.statistics import KEYS  # noqa: E402
from degradation_monitor.runs import RunLayout, stack  # noqa: E402
from degradation_monitor.settings import load_settings  # noqa: E402


def arm_rows(run: Path, names: list) -> list:
    """Per corrupted condition, the AUROC of the two-axis score and of its flattening and level arms."""
    layout = RunLayout(run)
    statistics = stack(layout.scores("method"), names, KEYS)
    with np.load(layout.method_bank) as bank, np.load(layout.method_zstats) as zstats:
        bank, zstats = {k: bank[k] for k in KEYS}, {k: zstats[k] for k in KEYS}
    two_axis, arms = two_axis_scores(statistics, bank, zstats)
    rows = []
    for arm, values in (("two_axis", two_axis), ("flatter", arms["flatter"]), ("level", arms["level"])):
        aurocs = condition_aurocs(values[:, 0], values[:, 1:].T)
        rows += [{"run": run.name, "arm": arm, "family": family, "severity": int(severity), "auroc": float(value)}
                 for (family, severity), value in zip(CONDITIONS[1:], aurocs)]
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Each run's two-axis score and its two arms per condition.")
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    names = [p.name for p in load_settings(ROOT / "configs" / "coco.toml").dataset.evaluation_images()]
    rows = [row for run in args.runs for row in arm_rows(run, names)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
