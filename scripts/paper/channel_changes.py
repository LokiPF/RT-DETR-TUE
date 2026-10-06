"""How a corruption changes each channel's mean response and peak response, per image, on RT-DETRv2-R18 (COCO val).

    python scripts/paper/channel_changes.py [--development]

Reads the stored channel statistics in runs/coco/scores/method/ (read-only) for all 5,000 COCO val images, or with
--development only for positions 0-1969 of the seed-44 order, the images read while the method was designed (written
to channel_changes_development.csv). For every corrupted version of an image,
each stage's channels are compared with the clean version's:
- flatter: the share of channels whose peak-to-mean ratio (log peak response - log mean response) fell;
- level_up, level_down: the shares of channels whose mean response rose, or fell, by more than 10%;
- summed_level, summed_top: the stage's summed mean response and summed peak response (the mean of the strongest 1%
  of positions), as a ratio to the clean version's.
Each is averaged over the images. Writes docs/results/paper-evidence/channel_changes.csv.
"""
from __future__ import annotations

import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "runs" / "coco" / "scores" / "method"
VAL = Path("/home/yuchen/YuchenZ/Datasets/coco/val2017")
OUT = ROOT / "docs" / "results" / "paper-evidence" / "channel_changes.csv"
DEVELOPMENT = 1970
STAGES = ("s1", "s2", "s3", "s4")
COMMON = ("gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
          "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression")
EXTRA = ("speckle_noise", "gaussian_blur", "spatter", "saturate")
CONDITIONS = [("clean", 0)] + [(f, s) for f in COMMON + EXTRA for s in range(1, 6)]
EPS = 1e-6  # as in degradation_monitor/method/scores.py
FIELDS = ("flatter", "level_up", "level_down", "summed_level", "summed_top")


def evaluation_images() -> list[Path]:
    """The evaluation order of degradation_monitor/datasets/coco.py: sorted val images, shuffled with seed 44."""
    images = sorted(p for p in VAL.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    np.random.default_rng(44).shuffle(images)
    return images


def one(path: Path) -> np.ndarray:
    z = np.load(path)
    out = np.empty((len(STAGES), len(CONDITIONS) - 1, len(FIELDS)))
    for s, stage in enumerate(STAGES):
        m, t = z[f"means_{stage}"].astype(np.float64), z[f"top_{stage}"].astype(np.float64)  # (96, C)
        share = np.log(t + EPS) - np.log(m + EPS)
        ratio = (m[1:] + EPS) / (m[0] + EPS)
        out[s, :, 0] = (share[1:] < share[0]).mean(axis=1)
        out[s, :, 1] = (ratio > 1.1).mean(axis=1)
        out[s, :, 2] = (ratio < 1 / 1.1).mean(axis=1)
        out[s, :, 3] = m[1:].sum(axis=1) / m[0].sum()
        out[s, :, 4] = t[1:].sum(axis=1) / t[0].sum()
    return out


def main() -> None:
    development = "--development" in sys.argv[1:]
    images = evaluation_images()[:DEVELOPMENT] if development else evaluation_images()
    files = [SCORES / (p.stem + ".npz") for p in images]
    out = OUT.with_name("channel_changes_development.csv") if development else OUT
    with Pool(6) as pool:
        mean = np.mean(pool.map(one, files, chunksize=40), axis=0)
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("stage", "family", "severity", "images", *FIELDS))
        for s, stage in enumerate(STAGES):
            for c, (family, severity) in enumerate(CONDITIONS[1:]):
                writer.writerow((stage, family, severity, len(files), *(f"{v:.4f}" for v in mean[s, c])))
    print(f"wrote {out} from {len(files)} images")


if __name__ == "__main__":
    main()
