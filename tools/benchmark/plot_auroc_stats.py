"""Plot mAP, AUROC (all vs correct-only) and detection counts per dataset.
Usage: python plot_benchmark.py [benchmark.csv]
Three stacked panels sharing the dataset axis (best mAP first):
  (1) mAP@[.5:.95] bars,
  (2) AUROC: 4 metrics (color), solid=all detections, dashed=correct-only,
  (3) #detections (conf>=0.5): total (light) with correct (dark) overlaid.
AUROC points appear only where defined (OOD datasets)."""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

csv_path = sys.argv[1] if len(sys.argv) > 1 else "output/auroc/exp4/benchmark.csv"
out_png = Path(csv_path).with_name("benchmark_plot.png")

# metric label -> (all col, correct col, color, marker)
METRICS = [
    ("detection / uncertainty",  "auroc_det_unc",   "auroc_detC_unc",   "C0", "o"),
    ("detection / 1-confidence", "auroc_det_1conf", "auroc_detC_1conf", "C1", "s"),
    ("image / uncertainty",      "auroc_img_unc",   "auroc_imgC_unc",   "C2", "^"),
    ("image / 1-confidence",     "auroc_img_1conf", "auroc_imgC_1conf", "C3", "D"),
]

mAP, ndet, ndetc = {}, {}, {}
vals = {}   # col -> {dataset: value}
with open(csv_path) as f:
    for r in csv.DictReader(f):
        d = r["dataset"]
        mAP[d] = float(r["AP"]); ndet[d] = int(r["n_det"])
        ndetc[d] = int(r.get("n_det_correct", 0) or 0)
        for _, ac, cc, *_ in METRICS:
            for col in (ac, cc):
                if r.get(col, "") != "":
                    vals.setdefault(col, {})[d] = float(r[col])

order = sorted(mAP, key=mAP.get, reverse=True)
x = list(range(len(order)))

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.grid": True,
    "grid.alpha": 0.3, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
})
fig, (a1, a2, a3) = plt.subplots(
    3, 1, sharex=True, figsize=(max(6, 1.6 * len(order)), 9),
    gridspec_kw={"height_ratios": [2, 3.4, 1.5], "hspace": 0.12})

a1.bar(x, [mAP[d] for d in order], color="#4C72B0", width=0.6)
a1.set_ylabel(r"mAP$_{[.5:.95]}$"); a1.set_ylim(0, max(mAP.values()) * 1.25)
a1.bar_label(a1.containers[0], fmt="%.3f", padding=2, fontsize=8)

def draw(col, color, marker, ls):
    dv = vals.get(col, {})
    xs = [i for i, d in enumerate(order) if d in dv]
    if xs:
        a2.plot(xs, [dv[order[i]] for i in xs], color=color, marker=marker,
                ms=6, lw=1.5, ls=ls, zorder=3)

for _, ac, cc, color, marker in METRICS:
    draw(ac, color, marker, "-")     # all detections
    draw(cc, color, marker, "--")    # correct-only
a2.axhline(0.5, color="0.5", lw=0.8, ls=":", zorder=0)
a2.set_ylabel("AUROC"); a2.set_ylim(0, 1)

metric_leg = [Line2D([], [], color=c, marker=m, ls="-", label=n)
              for n, _, _, c, m in METRICS]
style_leg = [Line2D([], [], color="0.3", ls="-", label="all detections"),
             Line2D([], [], color="0.3", ls="--", label="correct only")]
a2.add_artist(a2.legend(handles=metric_leg, fontsize=8, ncol=2, loc="upper left"))
a2.legend(handles=style_leg, fontsize=8, loc="upper right")

a3.bar(x, [ndet[d] for d in order], color="#55A868", width=0.6, label="all")
a3.bar(x, [ndetc[d] for d in order], color="#2E6B45", width=0.6, label="correct")
a3.set_ylabel("# det.\n(conf$\\geq$0.5)"); a3.set_yscale("log")
a3.legend(fontsize=8, loc="upper right")
a3.set_xticks(x); a3.set_xticklabels(order, rotation=30, ha="right")

fig.align_ylabels([a1, a2, a3])
fig.savefig(out_png, dpi=200, bbox_inches="tight")
print("saved:", out_png)