"""Table I of the paper: how far a corruption moves RT-DETRv2-R18's channels at each stage, in clean SDs.

    python scripts/paper/stage_shift.py

Reads the stored channel statistics in runs/coco/scores/method/ (all 5,000 val images) and the clean bank in
runs/coco/reference/method/bank.npz, read-only. For each channel, the shift of its mean response (and likewise of its
peak-to-mean ratio) is |value on the corrupted image - value on its clean version| divided by the channel's standard
deviation over the bank, floored as in the method. Averaged over the channels of a stage and over the images, for each
condition. Writes docs/results/paper-evidence/stage_shift.csv, one row per condition and stage, and prints the averages
over all 19 corruption types at five strengths and over the 15 common types.
"""
from __future__ import annotations

import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "runs" / "coco" / "scores" / "method"
BANK = ROOT / "runs" / "coco" / "reference" / "method" / "bank.npz"
OUT = ROOT / "docs" / "results" / "paper-evidence" / "stage_shift.csv"
STAGES = ("s1", "s2", "s3", "s4")
COMMON = ("gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
          "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression")
EXTRA = ("speckle_noise", "gaussian_blur", "spatter", "saturate")
CONDITIONS = [("clean", 0)] + [(f, s) for f in COMMON + EXTRA for s in range(1, 6)]
EPS = 1e-6  # as in degradation_monitor/method/scores.py


def numbers(z, stage: str) -> tuple[np.ndarray, np.ndarray]:
    mean, top = z[f"means_{stage}"].astype(np.float64), z[f"top_{stage}"].astype(np.float64)
    return mean, np.log(top + EPS) - np.log(mean + EPS)


def floored_sd(rows: np.ndarray) -> np.ndarray:
    sd = rows.std(axis=0)
    return np.maximum(sd, 0.01 * np.median(sd[sd > 0]))


with np.load(BANK) as _bank:
    SPREAD = {stage: tuple(floored_sd(a) for a in numbers(_bank, stage)) for stage in STAGES}


def one(path: Path) -> np.ndarray:
    """(stages, 2 numbers, 95 conditions): the shift averaged over the stage's channels."""
    with np.load(path) as z:
        out = np.empty((len(STAGES), 2, len(CONDITIONS) - 1))
        for s, stage in enumerate(STAGES):
            for n, values in enumerate(numbers(z, stage)):
                out[s, n] = (np.abs(values[1:] - values[0]) / SPREAD[stage][n]).mean(axis=1)
    return out


def main() -> None:
    files = sorted(SCORES.glob("*.npz"))
    with Pool(6) as pool:
        mean = np.mean(pool.map(one, files, chunksize=50), axis=0)
    with OUT.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("family", "severity", "stage", "images", "shift_mean_response", "shift_peak_to_mean_ratio"))
        for c, (family, severity) in enumerate(CONDITIONS[1:]):
            for s, stage in enumerate(STAGES):
                writer.writerow((family, severity, stage, len(files), f"{mean[s, 0, c]:.4f}", f"{mean[s, 1, c]:.4f}"))
    common = [c for c, (family, _) in enumerate(CONDITIONS[1:]) if family in COMMON]
    for name, columns in (("all 19 types at 5 strengths", slice(None)), ("15 common types at 5 strengths", common)):
        for n, number in enumerate(("mean response", "peak-to-mean ratio")):
            print(f"{name:32s} {number:20s}", " ".join(f"{mean[s, n, columns].mean():.2f}" for s in range(len(STAGES))))
    print(f"wrote {OUT} from {len(files)} images")


if __name__ == "__main__":
    main()
