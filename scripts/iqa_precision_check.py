"""The image-quality baselines' GPU precision against fp32 on the CPU: the plan's Task 6 check, with the tie counts.

    CUDA_VISIBLE_DEVICES= python scripts/iqa_precision_check.py    (from the repository root)

On the GPU, ARNIQA's encoder runs under autocast and CLIP-IQA with fp16 weights, as their official code runs them. The
check rescores all 96 versions of the first three evaluation images on the CPU in fp32 and compares them, row by row,
with the GPU scores the pass stored in configs/coco-iqa.toml's run folder: the largest absolute difference, Kendall's
tau, and the tied values on each device. It only reads the run folder. Its output for the COCO-C run is
docs/results/coco-iqa/precision-check.txt.

variant_stream builds the corrupted versions in spawned worker processes, and a spawned process re-imports the main
script. Hence the __main__ guard, so that a re-imported copy does not start the check again, and workers=0, which
builds the versions in this process, so that no worker is spawned at all.
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from degradation_monitor.baselines.iqa import ROWS  # noqa: E402
from degradation_monitor.runs import load_npz  # noqa: E402
from degradation_monitor.stages.common import variant_stream  # noqa: E402
from degradation_monitor.stages.iqa import _loaded_models, load_config  # noqa: E402


def main():
    config = load_config("configs/coco-iqa.toml")
    # the CPU in fp32, and no corruption workers: spawned workers re-import this script
    config = replace(config, base=replace(config.base, device="cpu", workers=0))
    models, _, _ = _loaded_models(config)
    gpu, cpu = {row: [] for row in ROWS}, {row: [] for row in ROWS}
    for image, arrays in variant_stream(config.base, config.base.dataset.evaluation_images()[:3]):
        stored = load_npz(config.layout.score_file("iqa", image), ROWS)
        parts = [models.scores(arrays[start:start + 16]) for start in range(0, len(arrays), 16)]
        for row in ROWS:
            gpu[row].append(stored[row])
            cpu[row].append(np.concatenate([part[row] for part in parts]))
    for row in ROWS:
        a, b = np.concatenate(gpu[row]), np.concatenate(cpu[row])
        ties = len(a) - len(np.unique(a)), len(b) - len(np.unique(b))
        print(f"{row:13s} max |GPU - CPU| {np.abs(a - b).max():.1e}, Kendall's tau {kendalltau(a, b).statistic:.4f}, "
              f"tied values GPU {ties[0]} / CPU {ties[1]} of {len(a)}")


if __name__ == "__main__":
    main()
