"""Why noise moves the channels' mean responses in both directions: RT-DETRv2-R18 under Gaussian noise, layer by layer.

    CUDA_VISIBLE_DEVICES= python scripts/paper/noise_layers.py [--images N]

Runs the frozen backbone on the CPU on the first N development images (positions 0 to N-1 of the seed-44 order; 40 by
default), clean and under Gaussian noise at severities 1, 3 and 5. For each layer from the stem's first conv-BN-ReLU to
the stage-1 map the method reads, each channel's mean |activation| is compared with the clean image's, as the mean
over images of the log ratio. Writes docs/results/paper-evidence/noise_layers.csv: per layer and severity, the shares
of channels whose mean response rises at all, rises by more than 10%, and falls by more than 10%.
"""
from __future__ import annotations

import argparse
import csv
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from degradation_monitor.corruptions import corrupt  # noqa: E402
from degradation_monitor.detector.model import load_frozen_detector, prepare_image  # noqa: E402

CHECKPOINT = "/home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
VAL = Path("/home/yuchen/YuchenZ/Datasets/coco/val2017")
OUT = ROOT / "docs" / "results" / "paper-evidence" / "noise_layers.csv"
SEVERITIES = (1, 3, 5)


def development_images(count: int) -> list[Path]:
    images = sorted(p for p in VAL.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    np.random.default_rng(44).shuffle(images)
    return images[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=40)
    args = parser.parse_args()
    torch.set_num_threads(8)
    backbone = load_frozen_detector(CHECKPOINT, torch.device("cpu")).backbone
    block = backbone.res_layers[0].blocks[0]
    layers = {"stem conv 1 (after ReLU)": backbone.conv1.conv1_1, "stem conv 2 (after ReLU)": backbone.conv1.conv1_2,
              "stem conv 3 (after ReLU)": backbone.conv1.conv1_3, "stage-1 block, first conv (after ReLU)": block.branch2a,
              "stage-1 map (the block's output, after ReLU)": block}
    current = {}

    def keep(name, _module, _inputs, output):
        current[name] = output.detach().abs().mean(dim=(2, 3)).numpy()

    handles = [module.register_forward_hook(partial(keep, name)) for name, module in layers.items()]
    means = {name: [] for name in layers}  # per image: (1 + severities, channels)
    files = development_images(args.images)
    for path in files:
        clean = Image.open(path).convert("RGB")
        versions = [clean] + [corrupt(clean, path.name, "gaussian_noise", s) for s in SEVERITIES]
        with torch.inference_mode():
            backbone(torch.stack([prepare_image(v, (640, 640)) for v in versions]))
        for name in layers:
            means[name].append(current[name])
    for handle in handles:
        handle.remove()
    with OUT.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("layer", "channels", "severity", "images", "rise", "rise_over_10pct", "fall_over_10pct"))
        for name in layers:
            values = np.array(means[name])  # (images, 1 + severities, channels)
            for i, severity in enumerate(SEVERITIES, start=1):
                log_ratio = np.log((values[:, i] + 1e-12) / (values[:, 0] + 1e-12)).mean(axis=0)
                row = (np.mean(log_ratio > 0), np.mean(log_ratio > np.log(1.1)), np.mean(log_ratio < -np.log(1.1)))
                writer.writerow((name, log_ratio.size, severity, len(files), *(f"{v:.3f}" for v in row)))
                print(f"{name:46s} sev {severity}: rise {row[0]:4.0%}, up >10% {row[1]:4.0%}, down >10% {row[2]:4.0%}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
