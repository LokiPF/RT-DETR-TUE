"""Draft figures for the paper from the evidence tables in docs/results/paper-evidence/ (PDF and PNG in figures/).

    python scripts/paper/figures.py

- flatten_or_shift: per family at severity 1, how far stage 1's channel levels move against how much flatter the
  channels get, coloured by the arm that is larger on most images (claim C5, candidate Figure 1);
- depth_curve: AUROC of each depth's own score, ours and the baselines' (C1, candidate Figure 3);
- dose_response: AUROC of the two-axis score under pure operations at graded strengths, with the training's jitter
  range shaded (C5, C8);
- ablation: k and bank size (C4).
Each figure is drawn only when its tables exist.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "docs" / "results" / "paper-evidence"
FIGURES = TABLES / "figures"
WIDTH = 3.5  # inches, one IEEE column
COLOURS = {"flatten": "#1f77b4", "shift": "#d62728", "neither": "#7f7f7f"}
SHORT = {"gaussian_noise": "Gauss. noise", "shot_noise": "shot", "impulse_noise": "impulse", "defocus_blur": "defocus",
         "glass_blur": "glass", "motion_blur": "motion", "zoom_blur": "zoom", "elastic_transform": "elastic",
         "jpeg_compression": "JPEG", "speckle_noise": "speckle", "gaussian_blur": "Gauss. blur"}


OFFSETS = {"gaussian_noise": (-14, 6), "shot_noise": (3, -8), "impulse_noise": (4, 1), "frost": (3, -8),
           "saturate": (3, 3), "elastic_transform": (-12, 5), "defocus_blur": (3, -8), "zoom_blur": (-18, -9), "fog": (3, -7),
           "contrast": (-34, 2)}


def read(name: str) -> list[dict]:
    path = TABLES / f"{name}.csv"
    return list(csv.DictReader(path.open())) if path.exists() else []


def save(figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(FIGURES / f"{name}.{suffix}", bbox_inches="tight", dpi=200)
    plt.close(figure)
    print(f"wrote {FIGURES / name}.pdf")


def flatten_or_shift() -> None:
    effects, flatten, arms = read("effect_sizes"), read("flatten_or_shift"), read("arms_by_family")
    if not (effects and flatten and arms):
        return
    figure, axis = plt.subplots(figsize=(WIDTH, 2.8))
    for row in arms:
        if row["severity"] != "1":
            continue
        family = row["family"]
        x = next(float(r["abs_in_bank_sd"]) for r in effects if r["family"] == family and r["severity"] == "1"
                 and r["stage"] == "s1" and r["statistic"] == "level")
        y = next(float(r["d_peak_share_in_bank_sd"]) for r in flatten if r["family"] == family
                 and r["severity"] == "1" and r["stage"] == "s1")
        flatter_larger = float(row["flatter_arm_larger"])
        best = max(float(row["two_axis"]), 0.5)
        kind = "neither" if best < 0.7 else ("flatten" if flatter_larger > 0.5 else "shift")
        axis.scatter(x, y, s=18, color=COLOURS[kind], zorder=3)
        axis.annotate(SHORT.get(family, family), (x, y), fontsize=6, xytext=OFFSETS.get(family, (3, 2)),
                      textcoords="offset points")
    axis.axhline(0, color="black", linewidth=0.5)
    axis.set_xlabel("stage-1 level change, mean |Δ| per channel (clean SD)", fontsize=7)
    axis.set_ylabel("stage-1 peak-share change (clean SD)\n← flatter        peakier →", fontsize=7)
    axis.tick_params(labelsize=6)
    for kind, label in (("flatten", "flattening arm larger"), ("shift", "level arm larger"),
                        ("neither", "two-axis AUROC < 0.7")):
        axis.scatter([], [], color=COLOURS[kind], s=18, label=label)
    axis.legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.28, 1.0), frameon=False)
    axis.set_title("Severity 1, all 5,000 images", fontsize=7)
    save(figure, "flatten_or_shift")


def depth_curve() -> None:
    rows = read("depth_curve")
    if not rows:
        return
    position = {"C1 stem": 0.0, "s1": 0.5, "C2 res1": 1.0, "s2": 1.5, "C3 res2": 2.0, "s3": 2.5, "C4 res3": 3.0,
                "s4": 3.5, "C5 res4": 4.0, "encoder map 1 (stride 8)": 4.9, "encoder map 2 (stride 16)": 5.0,
                "encoder map 3 (stride 32)": 5.1, "decoder queries": 6.0}
    series = {"level vs bank mean": ("level (ours, global)", "o-", "#d62728"),
              "peak share vs bank mean": ("peak share (ours, global)", "s-", "#1f77b4"),
              "activation CDF": ("activation CDFs [Becker]", "^--", "#2ca02c"),
              "Hashemi": ("Hashemi et al.", "v:", "#9467bd")}
    figure, axis = plt.subplots(figsize=(WIDTH, 2.4))
    for score, (label, style, colour) in series.items():
        points = sorted((position[r["stage"]], float(r["common"])) for r in rows if r["score"] == score)
        axis.plot([p[0] for p in points], [p[1] for p in points], style, color=colour, label=label, markersize=3,
                  linewidth=1)
    untrained = read("untrained_backbone")
    if untrained:
        points = [(position[f"s{k}"], float(r["common"])) for k in range(1, 5) for r in untrained
                  if r["score"] == f"global_level_s{k}" and r["backbone"] == "untrained"]
        axis.plot([p[0] for p in points], [p[1] for p in points], "o-", color="#ff9896", markersize=3, linewidth=1,
                  label="level, untrained backbone (200 images)")
    axis.axhline(0.5, color="black", linewidth=0.5, linestyle="--")
    axis.set_xticks([0, 1, 2, 3, 4, 5, 6], ["stem", "stage 1", "stage 2", "stage 3", "stage 4", "encoder", "decoder"],
                    fontsize=6)
    axis.set_ylabel("AUROC, common families", fontsize=7)
    axis.tick_params(labelsize=6)
    axis.legend(fontsize=5.5, frameon=False, loc="lower left")
    save(figure, "depth_curve")


def dose_response() -> None:
    rows = read("dose_response")
    if not rows:
        return
    ranges = {"gain": (0.875, 1.125), "contrast": (0.5, 1.5), "saturation": (0.5, 1.5), "hue": (0.0, 0.05)}
    panels = (("gain", "gain g (x·g)"), ("contrast", "contrast factor"), ("saturation", "saturation factor"),
              ("hue", "hue shift"), ("veil", "white veil t"), ("offset", "RGB offset b"),
              ("hsv_value", "HSV value offset"), ("blur", "Gaussian blur σ (px)"), ("noise", "Gaussian noise σ"))
    figure, axes = plt.subplots(3, 3, figsize=(2 * WIDTH, 5.0), sharey=True)
    for axis, (operation, label) in zip(axes.flat, panels):
        selected = sorted((float(r["strength"]), r) for r in rows if r["operation"] == operation)
        x = [s for s, _ in selected]
        for name, colour, style in (("two_axis", "black", "o-"), ("flatter_arm", "#1f77b4", "s--"),
                                    ("level_arm", "#d62728", "^--")):
            axis.plot(x, [float(r[f"auroc_{name}"]) for _, r in selected], style, color=colour, markersize=3,
                      linewidth=1, label=name.replace("_", " "))
        if operation in ranges:
            axis.axvspan(*ranges[operation], color="#cccccc", alpha=0.5, linewidth=0)
        if operation in ("gain", "contrast", "saturation"):
            axis.axvline(1.0, color="black", linewidth=0.5)
        axis.axhline(0.5, color="black", linewidth=0.5, linestyle=":")
        axis.set_xlabel(label, fontsize=7)
        axis.tick_params(labelsize=6)
    axes[0, 0].set_ylabel("AUROC vs clean", fontsize=7)
    axes[1, 0].set_ylabel("AUROC vs clean", fontsize=7)
    axes[2, 0].set_ylabel("AUROC vs clean", fontsize=7)
    axes[0, 0].legend(fontsize=6, frameon=False)
    figure.suptitle("Pure operations on 200 COCO val images; grey: the training's jitter range", fontsize=7)
    figure.tight_layout()
    save(figure, "dose_response")


def ablation() -> None:
    k_rows, bank_rows = read("ablation_k"), read("ablation_bank")
    if not (k_rows and bank_rows):
        return
    figure, (left, right) = plt.subplots(1, 2, figsize=(WIDTH, 1.8), sharey=True)
    for group, style in (("common", "o-"), ("extra", "s--")):
        left.plot([int(r["k"]) for r in k_rows], [float(r[f"two_axis_{group}"]) for r in k_rows], style,
                  markersize=3, linewidth=1, label=group, color="black")
        sizes = sorted({int(r["bank"]) for r in bank_rows})
        means = [sum(float(r[f"two_axis_{group}"]) for r in bank_rows if int(r["bank"]) == size)
                 / sum(1 for r in bank_rows if int(r["bank"]) == size) for size in sizes]
        right.plot(sizes, means, style, markersize=3, linewidth=1, color="black")
    left.set_xscale("log")
    right.set_xscale("log")
    left.axvline(50, color="#999999", linewidth=0.5)
    right.axvline(2000, color="#999999", linewidth=0.5)
    left.set_xlabel("neighbours k", fontsize=7)
    right.set_xlabel("bank images (k = 50)", fontsize=7)
    left.set_ylabel("two-axis AUROC", fontsize=7)
    for axis in (left, right):
        axis.tick_params(labelsize=6)
    left.legend(fontsize=6, frameon=False)
    save(figure, "ablation")


if __name__ == "__main__":
    flatten_or_shift()
    depth_curve()
    dose_response()
    ablation()
