"""Plot the corruption benchmark: corruption x severity heatmaps.
Usage: python plot_corruptions.py [benchmark_corruptions.csv]
One figure per ID dataset, with a heatmap per metric (rows=corruption sorted by
mean value, cols=severity 1-5), so you can read the severity trend per corruption
and compare uncertainty vs 1-confidence AUROC and mAP side by side."""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

csv_path = Path(
    sys.argv[1] if len(sys.argv) > 1 else "output/auroc/exp9/benchmark_corruptions.csv"
)

# (column, title, cmap, vmin, vmax)
METRICS = [
    ("AP", r"mAP$_{[.5:.95]}$", "viridis", None, None),
    ("auroc_det_unc", "AUROC uncertainty (det)", "cividis", 0.4, 1.0),
    ("auroc_det_1conf", "AUROC 1-confidence (det)", "cividis", 0.4, 1.0),
]


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


# rows: dataset -> corruption -> severity -> {col: value}
data = defaultdict(lambda: defaultdict(dict))
severities = set()
with open(csv_path) as f:
    for r in csv.DictReader(f):
        if r["corruption"] == "clean":
            continue  # severity 0 baseline has no AUROC; skip from the grid
        sev = int(r["severity"])
        severities.add(sev)
        data[r["dataset"]][r["corruption"]][sev] = r

severities = sorted(severities)
plt.rcParams.update({"font.family": "serif", "font.size": 9})

for dataset, by_corr in data.items():
    corruptions = sorted(by_corr)
    # order rows by mean AP (hardest corruption at the bottom)
    corruptions.sort(
        key=lambda c: np.nanmean(
            [fnum(by_corr[c].get(s, {}).get("AP", "nan")) for s in severities]
        )
    )

    fig, axes = plt.subplots(
        1,
        len(METRICS),
        figsize=(3.2 * len(METRICS), 0.42 * len(corruptions) + 1.5),
        squeeze=False,
    )
    for ax, (col, title, cmap, vmin, vmax) in zip(axes[0], METRICS):
        M = np.array(
            [
                [fnum(by_corr[c].get(s, {}).get(col, "nan")) for s in severities]
                for c in corruptions
            ]
        )
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(len(severities)))
        ax.set_xticklabels(severities)
        ax.set_xlabel("severity")
        ax.set_yticks(range(len(corruptions)))
        ax.set_yticklabels(corruptions)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if not np.isnan(v):
                    ax.text(
                        j,
                        i,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="w",
                    )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # only leftmost keeps y tick labels
    for ax in axes[0][1:]:
        ax.set_yticklabels([])
    fig.suptitle(dataset, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    safe = "".join(ch if ch.isalnum() else "_" for ch in dataset)
    out = csv_path.with_name(f"corruptions_{safe}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out)
