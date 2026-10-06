"""The figure of Sec. III-B: how far corruptions move RT-DETRv2-R18's channels at each stage, in clean SDs.

    python scripts/paper/stage_shift_figure.py

Reads docs/results/paper-evidence/stage_shift.csv (from stage_shift.py, all 5,000 val images) and plots the shift of
the mean response and of the peak-to-mean ratio by stage, averaged over the 19 corruption types at five strengths.
Writes IV_2027_Yuchen/Figures/stage_shift.pdf.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "docs" / "results" / "paper-evidence" / "stage_shift.csv"
OUT = ROOT / "IV_2027_Yuchen" / "Figures" / "stage_shift.pdf"
STAGES = ("s1", "s2", "s3", "s4")
STYLES = {  # two print-style looks: thin lines, distinct markers, full frame
    "mono": {"mu": {"color": "black", "marker": "o"},
             "rho": {"color": "#6b6b6b", "marker": "s", "markerfacecolor": "white"}, "grid": False},
    "muted": {"mu": {"color": "#1f3b63", "marker": "o"},
              "rho": {"color": "#9c3a2e", "marker": "s"}, "grid": True},
}
WIDTH = 3.5  # inches, one IEEE column


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", choices=sorted(STYLES), default="muted")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    values = defaultdict(list)  # (family or "all", number, stage) -> shifts over conditions
    with TABLE.open() as handle:
        for row in csv.DictReader(handle):
            for number in ("shift_mean_response", "shift_peak_to_mean_ratio"):
                values[("all", number, row["stage"])].append(float(row[number]))
                values[(row["family"], number, row["stage"])].append(float(row[number]))
    mean = lambda key: sum(values[key]) / len(values[key])  # noqa: E731
    x = range(1, len(STAGES) + 1)

    style = STYLES[args.style]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 7, "axes.linewidth": 0.6,
                         "xtick.direction": "in", "ytick.direction": "in", "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
    mu = [mean(("all", "shift_mean_response", s)) for s in STAGES]
    rho = [mean(("all", "shift_peak_to_mean_ratio", s)) for s in STAGES]
    fig, ax = plt.subplots(figsize=(WIDTH, 1.3))
    for values, (label, line) in zip((mu, rho), ((r"Mean response $\mu$", style["mu"]),
                                                 (r"Peak-to-mean ratio $\rho$", style["rho"]))):
        ax.plot(x, values, label=label, lw=1.0, ms=4, markeredgewidth=0.8, zorder=3, **line)
    ax.set_xticks(list(x), [f"Stage {i}" for i in x])
    ax.set_xlim(0.7, len(STAGES) + 0.3)
    ax.set_ylim(0, 1.4)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("Shift (clean SD)")
    ax.tick_params(top=True, right=True)
    if style["grid"]:
        ax.grid(axis="y", color="#bbbbbb", lw=0.4, ls=":")
        ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False, fontsize=6.5, handlelength=2.0)
    fig.tight_layout(pad=0.2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print("mean response", [round(v, 2) for v in mu], "| peak-to-mean ratio", [round(v, 2) for v in rho])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
