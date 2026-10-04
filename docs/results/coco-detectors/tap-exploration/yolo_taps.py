"""Which YOLO11m layers should the method read? YOLO11m on 300 development images.

Records each backbone layer's per-channel level (mean |x|) and top-1% mean over the cells that lie wholly on the image
(the letterbox's grey rows cropped away), for layers 0-10, then scores the two-axis method with these taps:
  A: layers 2, 4, 6 + key 8   (stages 1-3, key stage 4: the plan's choice)
  B: layers 2, 4, 6 + key 10  (the same stages, key the end of the backbone, which the head reads)
  C: layers 1, 3, 5 + key 7   (each stage's first module: its stride-2 downsampling conv)
  D: layers 1, 2, 3 + key 10  (the three earliest layers after the stem, key the end of the backbone)
  E: layers 0, 1, 2 + key 10  (the three earliest layers, key the end of the backbone)
YOLO11m's layers: 0 Conv s2, 1 Conv s4, 2 C3k2 s4, 3 Conv s8, 4 C3k2 s8, 5 Conv s16, 6 C3k2 s16, 7 Conv s32, 8 C3k2 s32,
9 SPPF, 10 C2PSA; the head reads 4, 6 and 10. Everything else is the method as published, through the package's own
two_axis_scores: 50 neighbours, bank 2,000 and z-statistics 500 clean COCO train images, the top 1%, the larger of the
flattening and the level arm. The images are 300 of positions 0-1969 of the evaluation order (seed 7, as the ViT
check), so the untouched images stay unread. The input is letterboxed as the planned adapter does (long side 640,
padded to a multiple of 32, centred); clean train images go one at a time, each image's 96 versions in batches of 16.

    python yolo_taps.py extract [--device cuda:0]
    python yolo_taps.py score
"""
from __future__ import annotations

import argparse
import math
import multiprocessing
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home/yuchen/YuchenZ/UE/philip_sa/scripts/paper")
from common import COMMON, EXTRA, FAMILY, SEVERITY, settings, summarise  # noqa: E402
from degradation_monitor import corruptions  # noqa: E402
from degradation_monitor.evaluation.metrics import bootstrap, condition_aurocs  # noqa: E402
from degradation_monitor.method.reference import fit_own_average  # noqa: E402
from degradation_monitor.method.scores import peak_share, two_axis_scores  # noqa: E402
from degradation_monitor.method.statistics import channel_statistics  # noqa: E402
from degradation_monitor.stages.common import bounded  # noqa: E402

OUT = Path(__file__).resolve().parent / "cache"
WEIGHTS = Path("/home/yuchen/YuchenZ/lab/Detector_test/yolo11m.pt")
LAYERS = 11  # the backbone: layers 0-10 run in sequence
STRIDES = (2, 4, 4, 8, 8, 16, 16, 32, 32, 32, 32)
LEVELS = [f"l{i}" for i in range(LAYERS)]
DEV, COUNT, SEED, BATCH, SIZE = 1970, 300, 7, 16, 640
GPU_GIB = 6.0
TAPS = {"A: 2, 4, 6 + key 8 (plan)": (("l2", "l4", "l6"), "l8"),
        "B: 2, 4, 6 + key 10": (("l2", "l4", "l6"), "l10"),
        "C: 1, 3, 5 + key 7": (("l1", "l3", "l5"), "l7"),
        "D: 1, 2, 3 + key 10": (("l1", "l2", "l3"), "l10"),
        "E: 0, 1, 2 + key 10": (("l0", "l1", "l2"), "l10")}
NOTED = ("fog", "contrast", "zoom_blur", "gaussian_noise", "brightness", "saturate", "spatter", "frost",
         "elastic_transform", "motion_blur")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def letterboxed(arrays):
    """Ultralytics' letterbox for same-size images, and where the image lies in it (top, left, height, width)."""
    from ultralytics.data.augment import LetterBox

    height, width = arrays[0].shape[:2]
    box = LetterBox(new_shape=(SIZE, SIZE), auto=True, stride=32)
    boxed = np.stack([box(image=np.ascontiguousarray(a)) for a in arrays])
    ratio = min(SIZE / height, SIZE / width)
    new_height, new_width = round(height * ratio), round(width * ratio)
    region = ((boxed.shape[1] - new_height) / 2, (boxed.shape[2] - new_width) / 2, new_height, new_width)
    return torch.from_numpy(boxed).permute(0, 3, 1, 2).float().div_(255.0), region


def crop(maps, region, stride):
    """The cells of a stride-`stride` map that lie wholly on the image."""
    top, left, height, width = region
    first_row, first_column = math.ceil(top / stride), math.ceil(left / stride)
    last_row = max(first_row + 1, math.floor((top + height) / stride))
    last_column = max(first_column + 1, math.floor((left + width) / stride))
    return maps[..., first_row:last_row, first_column:last_column]


class YoloTaps:
    """Per-channel level and top-1% mean of every backbone layer, computed inside the hooks."""

    def __init__(self, device):
        from ultralytics import YOLO

        self.device = torch.device(device)
        self.model = YOLO(str(WEIGHTS)).model.to(self.device).eval().requires_grad_(False)
        self.region, self.out = None, {}
        for index in range(LAYERS):
            self.model.model[index].register_forward_hook(partial(self._record, index))

    def _record(self, index, _module, _inputs, output):
        maps = crop(output, self.region, STRIDES[index]).float()
        for statistic, values in channel_statistics(maps).items():
            self.out.setdefault(f"{statistic}_l{index}", []).append(values)

    @torch.inference_mode()
    def __call__(self, arrays):
        batch, self.region = letterboxed(arrays)
        x = batch.to(self.device)
        for layer in self.model.model[:LAYERS]:
            x = layer(x)

    def take(self) -> dict:
        out, self.out = {k: np.concatenate(v) for k, v in self.out.items()}, {}
        return out


def extract(device):
    OUT.mkdir(parents=True, exist_ok=True)
    if torch.device(device).type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, GPU_GIB * 2**30 / total), device)
    taps = YoloTaps(device)
    dataset = settings().dataset
    for split in ("bank", "zstats"):
        path = OUT / f"{split}.npz"
        if path.exists():
            continue
        started = time.time()
        paths = dataset.reference_split(split)
        for p in paths:  # one at a time: the letterbox follows each image's size
            with Image.open(p) as source:
                taps([np.asarray(source.convert("RGB"), dtype=np.uint8)])
        np.savez(path, **taps.take())
        log(f"{split}: {len(paths)} images in {time.time() - started:.0f} s")
    positions = np.sort(np.random.default_rng(SEED).choice(DEV, COUNT, replace=False))
    paths = [dataset.evaluation_images()[i] for i in positions]
    pending = [p for p in paths if not (OUT / f"{p.stem}.npz").exists()]
    started = time.time()
    with multiprocessing.get_context("spawn").Pool(10) as pool:
        for done, (name, arrays) in enumerate(bounded(pool, corruptions.load_variants, pending, 20), start=1):
            for start in range(0, len(arrays), BATCH):
                taps(arrays[start:start + BATCH])
            np.savez(OUT / f"{Path(name).stem}.npz", **taps.take())
            if done % 25 == 0:
                peak = torch.cuda.max_memory_allocated(device) / 2**30 if torch.device(device).type == "cuda" else 0
                log(f"{done}/{len(pending)} images, {(time.time() - started) / done:.1f} s per image, "
                    f"peak {peak:.2f} GiB")
    np.save(OUT / "positions.npy", positions)


def load():
    bank = dict(np.load(OUT / "bank.npz"))
    zstats = dict(np.load(OUT / "zstats.npz"))
    positions = np.load(OUT / "positions.npy")
    evaluation = settings().dataset.evaluation_images()
    test = {}
    for i in positions:
        with np.load(OUT / f"{evaluation[i].stem}.npz") as item:
            for k in item.files:
                test.setdefault(k, []).append(item[k])
    return {k: np.stack(v) for k, v in test.items()}, bank, zstats


def per_condition(scores):
    out = np.full(scores.shape[1], np.nan)
    out[1:] = condition_aurocs(scores[:, 0], scores[:, 1:].T)
    return out


def flat(a):
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(-1, a.shape[-1])


def score():
    test, bank, zstats = load()
    images, conditions = test["means_l0"].shape[:2]
    log(f"{images} development images x {conditions} conditions")
    rows = {}
    for name, (scored, key) in TAPS.items():
        two_axis, arms = two_axis_scores(test, bank, zstats, key=key, scored=scored)
        rows[name] = {"two-axis": two_axis, "flatter arm": arms["flatter"], "level arm": arms["level"]}
    for name, parts in rows.items():
        for part, values in parts.items():
            aurocs = per_condition(values)
            s = summarise(aurocs)
            families = " ".join(f"{f[:6]} {aurocs[(FAMILY == f) & (SEVERITY == 1)][0]:.3f}" for f in NOTED)
            log(f"{name:<26} {part:<12} {s['common']:.4f} / {s['extra']:.4f}  sev1 {s['common_s1']:.4f} / "
                f"{s['extra_s1']:.4f} | sev1: {families}")
    groups = {"common": np.flatnonzero(COMMON), "extra": np.flatnonzero(EXTRA),
              "common_s1": np.flatnonzero(COMMON & (SEVERITY == 1)), "extra_s1": np.flatnonzero(EXTRA & (SEVERITY == 1))}
    reference_name = list(TAPS)[0]
    reference = rows[reference_name]["two-axis"]
    for name in list(TAPS)[1:]:
        other = rows[name]["two-axis"]

        def statistic(draw, other=other):
            return {g: float(condition_aurocs(other[draw, 0], other[draw][:, c].T).mean()
                             - condition_aurocs(reference[draw, 0], reference[draw][:, c].T).mean())
                    for g, c in groups.items()}
        point = statistic(np.arange(images))
        interval = bootstrap(statistic, images, samples=1000, seed=44, workers=8)
        log(f"{name[:1]} - A: " + "; ".join(f"{g} {point[g]:+.4f} [{interval[g][0]:+.4f}, {interval[g][1]:+.4f}]"
                                           for g in groups))
    # every layer's own score against the bank mean: YOLO11m's depth curve
    for level in LEVELS:
        mean, spread = fit_own_average(bank[f"means_{level}"])
        level_dev = (np.abs(flat(test[f"means_{level}"]) - mean) / spread).mean(1).reshape(images, conditions)
        share_mean, share_spread = fit_own_average(peak_share(bank, level))
        deviation = (peak_share(test, level) - share_mean) / share_spread
        share_dev = np.abs(deviation).mean(1).reshape(images, conditions)
        flatter = (-deviation).mean(1).reshape(images, conditions)
        cells = []
        for label, values in (("level", level_dev), ("share", share_dev), ("flatter", flatter)):
            s = summarise(per_condition(values))
            cells.append(f"{label} {s['common']:.3f}/{s['extra']:.3f}")
        log(f"depth {level:<4} " + "  ".join(cells))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("extract", "score"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    extract(args.device) if args.command == "extract" else score()
