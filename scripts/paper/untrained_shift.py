"""Is the weaker reaction of the deep stage learned? The shift in clean SDs by stage, trained against random weights.

    PAPER_CACHE=<dir> python scripts/paper/untrained_shift.py

Reads what `forward.py untrained` cached under PAPER_CACHE/forward/untrained/ (the random-weight backbone's statistics
for its 2,000-image bank and for the 200 evaluation images, clean and under the 19 families at severities 1, 3 and 5)
and the trained RT-DETRv2-R18's stored statistics for the same 200 images (runs/coco). For each backbone, stage and
severity: the mean over the 15 common families, the images and the channels of |change against the clean image|, in
units of the channel's spread over that backbone's own clean bank (floored as in the method), for the mean response and
for the peak-to-mean ratio. Writes docs/results/paper-evidence/untrained_shift.csv.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("PAPER_CACHE", ROOT / "runs" / "paper-cache"))
UNTRAINED = CACHE / "forward" / "untrained"
STORED = ROOT / "runs" / "coco" / "scores" / "method"
TRAINED_BANK = ROOT / "runs" / "coco" / "reference" / "method" / "bank.npz"
VAL = Path("/home/yuchen/YuchenZ/Datasets/coco/val2017")
OUT = ROOT / "docs" / "results" / "paper-evidence" / "untrained_shift.csv"
SEED, IMAGES = 44, 200  # forward.py's evaluation_positions
STAGES = ("s1", "s2", "s3", "s4")
COMMON = ("gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
          "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression")
EXTRA = ("speckle_noise", "gaussian_blur", "spatter", "saturate")
STORED_CONDITIONS = [("clean", 0)] + [(f, s) for f in COMMON + EXTRA for s in range(1, 6)]
UNTRAINED_CONDITIONS = [("clean", 0)] + [(f, s) for f in COMMON + EXTRA for s in (1, 3, 5)]
EPS = 1e-6  # as in degradation_monitor/method/scores.py


def floored_sd(rows: np.ndarray) -> np.ndarray:
    sd = rows.std(axis=0)
    return np.maximum(sd, 0.01 * np.median(sd[sd > 0]))


def numbers(z, stage: str) -> tuple[np.ndarray, np.ndarray]:
    mean, top = z[f"means_{stage}"].astype(np.float64), z[f"top_{stage}"].astype(np.float64)
    return mean, np.log(top + EPS) - np.log(mean + EPS)


def main() -> None:
    images = sorted(p for p in VAL.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    np.random.default_rng(44).shuffle(images)  # the evaluation order
    stems = [images[i].stem for i in np.sort(np.random.default_rng(SEED).choice(5000, IMAGES, replace=False))]
    sources = {"trained": (TRAINED_BANK, STORED, STORED_CONDITIONS),
               "untrained": (UNTRAINED / "bank.npz", UNTRAINED / "images", UNTRAINED_CONDITIONS)}
    rows = []
    for label, (bank_path, folder, conditions) in sources.items():
        with np.load(bank_path) as bank:
            spread = {stage: tuple(floored_sd(a) for a in numbers(bank, stage)) for stage in STAGES}
        for severity in (1, 3, 5):
            picked = [conditions.index((family, severity)) for family in COMMON]
            total = np.zeros((2, len(STAGES)))
            for stem in stems:
                with np.load(folder / f"{stem}.npz") as z:
                    for s, stage in enumerate(STAGES):
                        mean, ratio = numbers(z, stage)
                        for n, values in enumerate((mean, ratio)):
                            total[n, s] += np.mean([(np.abs(values[i] - values[0]) / spread[stage][n]).mean()
                                                    for i in picked])
            for n, number in enumerate(("mean_response", "peak_to_mean_ratio")):
                rows.append({"backbone": label, "number": number, "severity": severity, "images": len(stems),
                             **{stage: f"{total[n, s] / len(stems):.3f}" for s, stage in enumerate(STAGES)}})
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
