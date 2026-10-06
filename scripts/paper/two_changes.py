"""The figure of Sec. III-A: how fog and noise change one channel's map of RT-DETRv2-R18, on one development image.

    CUDA_VISIBLE_DEVICES= python scripts/paper/two_changes.py [--channel C] [--preview C ...]

Runs the frozen detector on the CPU on COCO val image 000000026204 (position 309 of the seed-44 order, a development
image), clean and at severity 5 of fog and of Gaussian noise, and reads its stage-1 maps with the method's own taps.
The top row shows the image clean, fogged and noisy; the bottom row shows one channel's map for each, on the colour
scale of its clean map, with the channel's mean response mu and peak response p (the mean of its strongest 1% of positions) as a
ratio to the clean map's.
Writes IV_2027_Yuchen/Figures/two_changes.pdf; --preview writes a PNG of candidate channels to the current folder.

Why channel 51 (6 October): of the 64 stage-1 channels, it is the one whose clean map follows the image most closely
(correlation 0.77 with its brightness), among those that show both changes on this image and also on average over the
1,970 development images (fog: mu x1.00, p x0.68; noise: mu x1.72, p x1.12) and over all 5,000 val images (fog: mu
x1.00, p x0.68; noise: mu x1.71, p x1.12), from runs/coco's stored statistics.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from degradation_monitor.corruptions import corrupt  # noqa: E402
from degradation_monitor.detector.model import load_frozen_detector, prepare_image  # noqa: E402
from degradation_monitor.detector.taps import EarlyChannelTaps  # noqa: E402
from degradation_monitor.method.statistics import channel_statistics  # noqa: E402

CHECKPOINT = "/home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
IMAGE = Path("/home/yuchen/YuchenZ/Datasets/coco/val2017/000000026204.jpg")
OUT = ROOT / "IV_2027_Yuchen" / "Figures" / "two_changes.pdf"
SEVERITY = 5
ROWS = (("fog", "Fog", "fog"), ("gaussian_noise", "Gaussian noise", "noise"))
WIDTH = 3.5  # inches, one IEEE column


def stage1(images):
    model = load_frozen_detector(CHECKPOINT, torch.device("cpu"))
    with EarlyChannelTaps(model.backbone) as taps:
        maps = taps(torch.stack([prepare_image(im, (640, 640)) for im in images]))[0]
    stats = channel_statistics(maps)
    return maps.abs().numpy(), stats["means"], stats["top"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", type=int, default=51)
    parser.add_argument("--preview", type=int, nargs="*", help="noise-row candidate channels to preview")
    args = parser.parse_args()
    torch.set_num_threads(6)
    clean = Image.open(IMAGE).convert("RGB")
    images = [clean] + [corrupt(clean, IMAGE.name, family, SEVERITY) for family, _, _ in ROWS]
    maps, level, top = stage1(images)
    for row, (family, _, _) in enumerate(ROWS, start=1):
        for c in [args.channel] + list(args.preview or []):
            print(f"{family} sev {SEVERITY} ch {c}: level x{level[row, c] / level[0, c]:.2f}, "
                  f"top x{top[row, c] / top[0, c]:.2f}")
    if args.preview:
        cols = len(args.preview)
        fig, axes = plt.subplots(2, cols, figsize=(3 * cols, 4))
        for j, c in enumerate(args.preview):
            vmax = np.percentile(maps[0, c], 99.7)
            for i, v in enumerate((0, 2)):
                axes[i, j].imshow(maps[v, c], cmap="magma", vmin=0, vmax=vmax, aspect=clean.height / clean.width * 0 + 0.667)
                axes[i, j].set_title(f"ch {c} {'clean' if v == 0 else 'noise'}", fontsize=8)
                axes[i, j].axis("off")
        fig.savefig("two_changes_preview.png", dpi=110, bbox_inches="tight")
        return

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 7})
    c = args.channel
    aspect = clean.height / clean.width
    titles = ("Clean",) + tuple(name for _, name, _ in ROWS)  # the severity is not introduced before Sec. IV
    margins = {"left": 0.06, "right": 0.995, "top": 0.92, "bottom": 0.01, "wspace": 0.04, "hspace": 0.06}
    # the panels' height in inches, so that the figure is as tall as its two rows and leaves no empty band
    panel_height = WIDTH * (margins["right"] - margins["left"]) / (3 + 2 * margins["wspace"]) * aspect
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, panel_height * (2 + margins["hspace"])
                                            / (margins["top"] - margins["bottom"])), gridspec_kw=margins)
    vmax = np.percentile(maps[0, c], 99.7)
    for col in range(3):
        axes[0, col].imshow(np.asarray(images[col]))
        axes[1, col].imshow(maps[col, c], cmap="magma", vmin=0, vmax=vmax, interpolation="bilinear",
                            extent=(0, 1, 0, aspect))
        axes[0, col].set_title(titles[col], fontsize=7, pad=2)
        for ax in axes[:, col]:
            ax.set_xticks([]), ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        if col > 0:
            axes[1, col].text(0.98, 0.04, f"$\\mu$ ×{level[col, c] / level[0, c]:.2f}   $p$ ×{top[col, c] / top[0, c]:.2f}",
                              transform=axes[1, col].transAxes, ha="right", va="bottom", color="white", fontsize=6.5,
                              bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.2, "linewidth": 0})
    axes[0, 0].set_ylabel("Image", fontsize=7, labelpad=2)
    axes[1, 0].set_ylabel("Channel", fontsize=7, labelpad=2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=600)  # the photos and maps are embedded as rasters; 600 dpi keeps them sharp in print
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
