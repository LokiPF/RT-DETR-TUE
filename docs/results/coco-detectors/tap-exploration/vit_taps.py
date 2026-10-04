"""Which ViT blocks should the method read? RF-DETR-M (DINOv2 ViT-S, 12 blocks) on 300 development images.

Records each block's per-channel level (mean |x|) and top-1% mean over the image's patch tokens, for the embeddings
(b0) and blocks 1-12, then scores the two-axis method with
  A: blocks 3, 6, 9 scored, block 12 the key (four depth quarters);
  B: blocks 1, 2, 3 scored, block 12 the key.
Everything else is the method as published: 50 neighbours, bank 2,000 and z-statistics 500 clean COCO train images,
the top 1%, the larger of the flattening and the level arm. The images are positions 0-1969 of the evaluation order
(seed 7, as the earlier screens), so the untouched images stay unread.

    python vit_taps.py extract [--device cuda:0]
    python vit_taps.py score
"""
from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2.functional as F
from PIL import Image

sys.path.insert(0, "/home/yuchen/YuchenZ/UE/philip_sa/scripts/paper")
from common import COMMON, EXTRA, FAMILY, SEVERITY, settings, summarise  # noqa: E402
from degradation_monitor import corruptions  # noqa: E402
from degradation_monitor.evaluation.metrics import bootstrap, condition_aurocs, stage_zstats, zscored_sum  # noqa: E402
from degradation_monitor.method.reference import CHUNK, fit_own_average, nearest_rows  # noqa: E402
from degradation_monitor.method.statistics import channel_statistics  # noqa: E402
from degradation_monitor.method.scores import EPS, peak_share  # noqa: E402
from degradation_monitor.stages.common import bounded  # noqa: E402

OUT = Path(__file__).resolve().parent / "cache"
LEVELS = [f"b{i}" for i in range(13)]  # b0 = embeddings, b1-b12 = block outputs
DEV, COUNT, SEED, BATCH = 1970, 300, 7, 32
RESOLUTION, WINDOWS = 576, 2
MEANS, STDS = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TAPS = {"A: blocks 3, 6, 9 + key 12": ("b3", "b6", "b9"), "B: blocks 1, 2, 3 + key 12": ("b1", "b2", "b3")}
KEY = "b12"
REPRESENTATIONS = {"raw block outputs": "", "layer-normed (the detector's features)": "ln"}
NOTED = ("fog", "contrast", "zoom_blur", "gaussian_noise", "brightness", "saturate", "spatter", "frost",
         "elastic_transform", "motion_blur")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class ViTTaps:
    """Per-channel level and top-1% mean of every block's patch tokens, regathered from RF-DETR's windows."""

    def __init__(self, device):
        from rfdetr import RFDETRMedium

        self.device = torch.device(device)
        core = RFDETRMedium().model.model
        self.dino = core.backbone[0].encoder.to(self.device).eval()
        hf = self.dino.encoder
        self.grid = RESOLUTION // hf.config.patch_size
        self.skip = 1 + hf.config.num_register_tokens  # the class token and any register tokens of each window
        self.outputs = {}
        hf.embeddings.register_forward_hook(lambda m, i, o: self.outputs.__setitem__("b0", o))
        for index, layer in enumerate(hf.encoder.layer, start=1):
            layer.register_forward_hook(lambda m, i, o, name=f"b{index}": self.outputs.__setitem__(name, o[0]))

    def maps(self, tokens):
        """(B * windows^2, skip + T, C) -> (B, C, grid, grid), undoing the window layout of the embeddings."""
        tokens = tokens[:, self.skip:]
        per_window = self.grid // WINDOWS
        batch = tokens.shape[0] // WINDOWS ** 2
        x = tokens.reshape(batch, WINDOWS, WINDOWS, per_window, per_window, -1)  # (B, wh, ww, h, w, C)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(batch, -1, self.grid, self.grid)
        return x

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor) -> dict:
        batch = F.normalize(F.resize(images.to(self.device), [RESOLUTION, RESOLUTION], antialias=False), MEANS, STDS)
        self.outputs.clear()
        self.dino(batch)
        out = {}
        layernorm = self.dino.encoder.layernorm  # what the backbone applies to the levels the detector reads
        for level in LEVELS:
            raw = self.outputs[level]
            for prefix, tokens in (("", raw), ("ln", layernorm(raw))):
                for statistic, values in channel_statistics(self.maps(tokens).float()).items():
                    out[f"{statistic}_{prefix}{level}"] = values
        return out


def to_tensor(array):
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float() / 255.0


def statistics_of(taps, arrays) -> dict:
    parts = []
    for start in range(0, len(arrays), BATCH):
        chunk = arrays[start:start + BATCH]
        # images differ in size: resize each to the model's square input first, as RF-DETR's predict does
        batch = torch.stack([F.resize(to_tensor(a), [RESOLUTION, RESOLUTION], antialias=False) for a in chunk])
        parts.append(taps(batch))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def extract(device):
    OUT.mkdir(parents=True, exist_ok=True)
    taps = ViTTaps(device)
    dataset = settings().dataset
    for split in ("bank", "zstats"):
        path = OUT / f"{split}.npz"
        if path.exists():
            continue
        started = time.time()
        arrays = []
        for p in dataset.reference_split(split):
            with Image.open(p) as source:
                arrays.append(np.asarray(source.convert("RGB"), dtype=np.uint8))
        np.savez(path, **statistics_of(taps, arrays))
        log(f"{split}: {len(arrays)} images in {time.time() - started:.0f} s")
    positions = np.sort(np.random.default_rng(SEED).choice(DEV, COUNT, replace=False))
    paths = [dataset.evaluation_images()[i] for i in positions]
    pending = [p for p in paths if not (OUT / f"{p.stem}.npz").exists()]
    started = time.time()
    with multiprocessing.get_context("spawn").Pool(10) as pool:
        for done, (name, arrays) in enumerate(bounded(pool, corruptions.load_variants, pending, 20), start=1):
            np.savez(OUT / f"{Path(name).stem}.npz", **statistics_of(taps, arrays))
            if done % 25 == 0:
                log(f"{done}/{len(pending)} images, {(time.time() - started) / done:.1f} s per image")
    np.save(OUT / "positions.npy", positions)


def load():
    bank = dict(np.load(OUT / "bank.npz"))
    zstats = dict(np.load(OUT / "zstats.npz"))
    positions = np.load(OUT / "positions.npy")
    paths = [settings().dataset.evaluation_images()[i] for i in positions]
    test = {}
    for p in paths:
        with np.load(OUT / f"{p.stem}.npz") as item:
            for k in item.files:
                test.setdefault(k, []).append(item[k])
    return {k: np.stack(v) for k, v in test.items()}, bank, zstats, positions


def flat(a):
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(-1, a.shape[-1])


def neighbours_of(values, bank, key, k=50):
    centre, spread = fit_own_average(bank[f"means_{key}"])
    bank_keys = (np.asarray(bank[f"means_{key}"], dtype=np.float64) - centre) / spread
    queries = flat(values[f"means_{key}"])
    out = np.empty((len(queries), k), dtype=np.int64)
    for start in range(0, len(queries), CHUNK):
        out[start:start + CHUNK] = nearest_rows((queries[start:start + CHUNK] - centre) / spread, bank_keys, k)
    return out


def columns(values, bank, neighbours, scored):
    """The method's level and flattening columns per scored level (as degradation_monitor.method.scores._columns)."""
    out = {"level": [], "flatter": []}
    for layer in scored:
        level_reference = np.asarray(bank[f"means_{layer}"], dtype=np.float64)
        level_spread = fit_own_average(level_reference)[1]
        share_reference = peak_share(bank, layer)
        share_spread = fit_own_average(share_reference)[1]
        level_values, share_values = flat(values[f"means_{layer}"]), peak_share(values, layer)
        level_column, flatter_column = np.empty(len(neighbours)), np.empty(len(neighbours))
        for start in range(0, len(neighbours), CHUNK):
            rows = slice(start, start + CHUNK)
            nb = neighbours[rows]
            level_column[rows] = (np.abs(level_values[rows] - level_reference[nb].mean(1)) / level_spread).mean(1)
            flatter_column[rows] = (-(share_values[rows] - share_reference[nb].mean(1)) / share_spread).mean(1)
        out["level"].append(level_column)
        out["flatter"].append(flatter_column)
    return {k: np.stack(v, 1) for k, v in out.items()}


def arm(test_cols, clean_cols):
    mean, std = stage_zstats(clean_cols)
    clean = zscored_sum(clean_cols, mean, std)
    return (zscored_sum(test_cols, mean, std) - clean.mean()) / clean.std()


def per_condition(scores):
    out = np.full(scores.shape[1], np.nan)
    out[1:] = condition_aurocs(scores[:, 0], scores[:, 1:].T)
    return out


def score():
    test, bank, zstats, positions = load()
    for representation, prefix in REPRESENTATIONS.items():
        log(f"=== {representation}")
        score_one(test, bank, zstats, prefix)


def score_one(test, bank, zstats, prefix):
    images, conditions = test["means_b1"].shape[:2]
    key = prefix + KEY
    nb_test, nb_clean = neighbours_of(test, bank, key), neighbours_of(zstats, bank, key)
    rows = {}
    for name, scored in TAPS.items():
        scored = tuple(prefix + level for level in scored)
        t, c = columns(test, bank, nb_test, scored), columns(zstats, bank, nb_clean, scored)
        flatter = arm(t["flatter"], c["flatter"]).reshape(images, conditions)
        level = arm(t["level"], c["level"]).reshape(images, conditions)
        rows[name] = {"two-axis": np.maximum(flatter, level), "flatter arm": flatter, "level arm": level}
    for name, parts in rows.items():
        for part, values in parts.items():
            aurocs = per_condition(values)
            s = summarise(aurocs)
            fams = " ".join(f"{f[:6]} {aurocs[(FAMILY == f) & (SEVERITY == 1)][0]:.3f}" for f in NOTED)
            log(f"{name:<28} {part:<12} {s['common']:.4f} / {s['extra']:.4f}  sev1 {s['common_s1']:.4f} / "
                f"{s['extra_s1']:.4f} | sev1: {fams}")
    a, b = rows[list(TAPS)[0]]["two-axis"], rows[list(TAPS)[1]]["two-axis"]
    groups = {"common": np.flatnonzero(COMMON), "extra": np.flatnonzero(EXTRA),
              "common_s1": np.flatnonzero(COMMON & (SEVERITY == 1)), "extra_s1": np.flatnonzero(EXTRA & (SEVERITY == 1))}

    def statistic(draw):
        return {g: float(condition_aurocs(a[draw, 0], a[draw][:, c].T).mean()
                         - condition_aurocs(b[draw, 0], b[draw][:, c].T).mean()) for g, c in groups.items()}
    point = statistic(np.arange(images))
    interval = bootstrap(statistic, images, samples=1000, seed=44, workers=8)
    log("A - B: " + "; ".join(f"{g} {point[g]:+.4f} [{interval[g][0]:+.4f}, {interval[g][1]:+.4f}]" for g in groups))
    # every level's own score against the bank mean: the ViT's depth curve
    for level in (prefix + level for level in LEVELS):
        mean, spread = fit_own_average(bank[f"means_{level}"])
        level_dev = (np.abs(flat(test[f"means_{level}"]) - mean) / spread).mean(1).reshape(images, conditions)
        share_mean, share_spread = fit_own_average(peak_share(bank, level))
        deviation = ((peak_share(test, level) - share_mean) / share_spread)
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
