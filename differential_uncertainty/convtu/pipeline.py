"""Resumable phases of the conv-TU pilot on the fixed COCO protocol; they run through `baselines-coco --phase`.

Clean COCO train images give each layer's cut (convtu-calibrate), the kNN bank (convtu-bank) and the
z-statistics (convtu-zstats), from three disjoint seeded draws.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..baselines import protocol
from ..baselines.activation_cdf import stage_zstats, zscored_sum
from ..baselines.pipeline import (TEST_KEYS, Settings, _atomic_json, _atomic_npz, _load_npz, _progress,
                                  _train_loader, _valid_existing, _variant_stream, evaluation)
from ..extraction import load_frozen_detector, prepare_image
from .features import KNN_NEIGHBOURS, REPRESENTATIONS, knn_scores, layer_features, layer_specs
from .graph import EDGE_CAP, FRACTION, conv_top_merges, heaviest_weight
from .tap import LAYER_NAMES, ConvInputs

CALIBRATION_IMAGES = 200
BANK_IMAGES = 2000
ZSTAT_IMAGES = 500
CUT_MARGIN = 0.5    # each layer's cut: half the smallest K-th value over the calibration images
START_MULTIPLE = 8  # calibration starts each image at its (8 K)-th heaviest edge
PILOT_IMAGES = 200  # the first images of the seed-44 evaluation order, all 96 conditions
IMAGE_SIZE = (640, 640)
SCORE_KEYS = (*REPRESENTATIONS, *(f"{rep}_layers" for rep in REPRESENTATIONS), "tau_k", "read", "rounds")


def _folder(settings: Settings):
    return settings.output / "convtu"


def calibration_path(settings: Settings):
    return _folder(settings) / "calibration.json"


def bank_path(settings: Settings):
    return _folder(settings) / "bank.npz"


def zstats_path(settings: Settings):
    return _folder(settings) / "zstats.json"


def _sha1(path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def train_splits(count: int, seed: int) -> dict:
    """Disjoint seeded draws of train-image indices for calibration, the bank and the z-statistics."""
    needed = CALIBRATION_IMAGES + BANK_IMAGES + ZSTAT_IMAGES
    if count < needed:
        raise ValueError(f"the pilot needs {needed} train images, found {count}")
    order = np.random.default_rng(seed).permutation(count)
    first, second = CALIBRATION_IMAGES, CALIBRATION_IMAGES + BANK_IMAGES
    return {"calibration": np.sort(order[:first]), "bank": np.sort(order[first:second]),
            "zstats": np.sort(order[second:needed])}


def _conv_inputs(settings: Settings) -> ConvInputs:
    return ConvInputs(load_frozen_detector(settings.checkpoint, torch.device(settings.device)).backbone)


def _clean_batches(settings: Settings, split: str):
    paths = protocol.list_images(settings.train_images)
    return _train_loader(settings, [paths[i] for i in train_splits(len(paths), settings.seed)[split]])


def _sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def _quantiles(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {"min": float(values.min()), "p50": float(np.median(values)), "p90": float(np.quantile(values, 0.9)),
            "p99": float(np.quantile(values, 0.99)), "max": float(values.max())}


def _last_positive(values: np.ndarray) -> float:
    """The K-th value, or the last positive one when a small graph's diagram ends in zeros."""
    positive = values[values > 0]
    if positive.size == 0:
        raise ValueError("a calibration image has no positive diagram value")
    return float(positive[-1])


def phase_calibrate(settings: Settings) -> None:
    """Where the K-th merge falls on clean train images, how many edges it takes, and each layer's cut."""
    path = calibration_path(settings)
    if path.exists():
        return
    on_gpu = torch.device(settings.device).type == "cuda"
    tau, read, specs = [], [], None
    with _conv_inputs(settings) as taps:
        for batch in _clean_batches(settings, "calibration"):
            inputs = taps(batch.to(settings.device))
            specs = specs or layer_specs(inputs, taps.kernels)
            for image in range(batch.shape[0]):
                row_tau, row_read = [], []
                for layer, spec in enumerate(specs):
                    x_abs = inputs[layer][image].abs()
                    start = heaviest_weight(x_abs, taps.kernels[layer], START_MULTIPLE * spec.k)
                    top = conv_top_merges(x_abs, taps.kernels[layer], spec.k, start)
                    row_tau.append(_last_positive(top.values))
                    row_read.append(top.read)
                tau.append(row_tau)
                read.append(row_read)
        tau, read = np.array(tau), np.array(read)
        cuts = CUT_MARGIN * tau.min(axis=0)
        if on_gpu:
            torch.cuda.reset_peak_memory_stats()
        gathered, ms, rounds = [], [], []
        for batch in _clean_batches(settings, "calibration"):
            inputs = taps(batch.to(settings.device))
            for image in range(batch.shape[0]):
                row_gathered, row_ms, row_rounds = [], [], []
                for layer, spec in enumerate(specs):
                    _sync(settings.device)
                    started = time.perf_counter()
                    features = layer_features(inputs[layer][image], taps.kernels[layer], spec, float(cuts[layer]))
                    _sync(settings.device)
                    row_ms.append(1000.0 * (time.perf_counter() - started))
                    row_gathered.append(features["gathered"])
                    row_rounds.append(features["rounds"])
                gathered.append(row_gathered)
                ms.append(row_ms)
                rounds.append(row_rounds)
    gathered, ms, rounds = np.array(gathered), np.array(ms), np.array(rounds)
    layers = {}
    for layer, spec in enumerate(specs):
        layers[spec.name] = {
            "conv": LAYER_NAMES[layer], "input_shape": [spec.c_in, spec.height, spec.width],
            "nodes": spec.nodes, "k": spec.k, "cut": float(cuts[layer]),
            "tau_k": _quantiles(tau[:, layer]), "tau_k_cv": float(tau[:, layer].std() / tau[:, layer].mean()),
            "edges_read": _quantiles(read[:, layer]), "edges_gathered_at_cut": _quantiles(gathered[:, layer]),
            "rounds_at_cut_max": int(rounds[:, layer].max()), "ms_per_image": _quantiles(ms[:, layer]),
        }
    _atomic_json(path, {"images": int(tau.shape[0]), "seed": settings.seed, "fraction": FRACTION,
                        "cut_margin": CUT_MARGIN, "edge_cap": EDGE_CAP,
                        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if on_gpu else None,
                        "layers": layers})


def _cuts(settings: Settings) -> dict:
    path = calibration_path(settings)
    if not path.exists():
        raise ValueError("run the convtu-calibrate phase first")
    return {name: layer["cut"] for name, layer in json.loads(path.read_text())["layers"].items()}


def features_of(inputs, kernels, specs, cuts) -> list[dict]:
    """Per image of the batch: every representation of every layer, plus the per-layer diagnostics."""
    rows = []
    for image in range(inputs[0].shape[0]):
        row = {}
        for layer, spec in enumerate(specs):
            features = layer_features(inputs[layer][image], kernels[layer], spec, cuts[spec.name])
            for key in (*REPRESENTATIONS, "tau_k", "read", "rounds"):
                row[f"{key}_{spec.name}"] = features[key]
        rows.append(row)
    return rows


def phase_bank(settings: Settings) -> None:
    """The fingerprint and the three controls of 2,000 clean train images: the kNN bank."""
    path = bank_path(settings)
    if path.exists():
        return
    cuts = _cuts(settings)
    rows, specs, started = [], None, time.time()
    with _conv_inputs(settings) as taps:
        for batch in _clean_batches(settings, "bank"):
            inputs = taps(batch.to(settings.device))
            specs = specs or layer_specs(inputs, taps.kernels)
            rows += features_of(inputs, taps.kernels, specs, cuts)
            _progress("convtu-bank", len(rows), BANK_IMAGES, started)
    arrays = {f"{rep}_{spec.name}": np.stack([row[f"{rep}_{spec.name}"] for row in rows])
              for rep in REPRESENTATIONS for spec in specs}
    _atomic_npz(path, calibration_sha1=np.array(_sha1(calibration_path(settings))), **arrays)


def load_bank(settings: Settings, device) -> dict:
    path = bank_path(settings)
    if not path.exists():
        raise ValueError("run the convtu-bank phase first")
    with np.load(path, allow_pickle=False) as data:
        if str(data["calibration_sha1"]) != _sha1(calibration_path(settings)):
            raise ValueError("the calibration changed since the bank was built")
        return {key: torch.from_numpy(data[key]).to(device) for key in data.files if key != "calibration_sha1"}


def layer_scores(rows, bank, specs) -> dict:
    """Per representation, (images, layers) mean distances to the nearest bank rows."""
    out = {}
    for rep in REPRESENTATIONS:
        columns = [knn_scores(torch.from_numpy(np.stack([row[f"{rep}_{spec.name}"] for row in rows])),
                              bank[f"{rep}_{spec.name}"]) for spec in specs]
        out[rep] = np.stack(columns, axis=1)
    return out


def phase_zstats(settings: Settings) -> None:
    """Mean and spread of each layer's kNN score over 500 clean train images outside the bank."""
    path = zstats_path(settings)
    if path.exists():
        return
    cuts = _cuts(settings)
    bank = load_bank(settings, settings.device)
    rows, specs = [], None
    with _conv_inputs(settings) as taps:
        for batch in _clean_batches(settings, "zstats"):
            inputs = taps(batch.to(settings.device))
            specs = specs or layer_specs(inputs, taps.kernels)
            rows += features_of(inputs, taps.kernels, specs, cuts)
    stats = {}
    for rep, values in layer_scores(rows, bank, specs).items():
        mean, std = stage_zstats(values)
        stats[rep] = {"mean": mean.tolist(), "std": std.tolist()}
    _atomic_json(path, {"images": len(rows), "layers": [spec.name for spec in specs],
                        "neighbours": KNN_NEIGHBOURS, "stats": stats, "bank_sha1": _sha1(bank_path(settings))})


def pilot_images(settings: Settings) -> list:
    return evaluation(settings)[:PILOT_IMAGES]


def image_scores(taps, arrays, bank, zstats, cuts, batch_size) -> dict:
    """Every condition of one image: per-layer kNN distances, their z-scored sums and the diagnostics."""
    rows, specs = [], None
    for start in range(0, len(arrays), batch_size):
        batch = torch.stack([prepare_image(Image.fromarray(a), IMAGE_SIZE) for a in arrays[start:start + batch_size]])
        inputs = taps(batch.to(taps.kernels[0].device))
        specs = specs or layer_specs(inputs, taps.kernels)
        rows += features_of(inputs, taps.kernels, specs, cuts)
    out = {}
    for rep, values in layer_scores(rows, bank, specs).items():
        stats = zstats["stats"][rep]
        out[f"{rep}_layers"] = values
        out[rep] = zscored_sum(values, stats["mean"], stats["std"])
    for key in ("tau_k", "read", "rounds"):
        out[key] = np.array([[row[f"{key}_{spec.name}"] for spec in specs] for row in rows])
    if not all(np.isfinite(value).all() for value in out.values()):
        raise ValueError("conv-TU produced non-finite scores")
    return out


def phase_scores(settings: Settings) -> None:
    """The fingerprint and its three controls for every condition of the pilot images."""
    fits = {"convtu-calibrate": calibration_path(settings), "convtu-bank": bank_path(settings),
            "convtu-zstats": zstats_path(settings)}
    missing = [phase for phase, path in fits.items() if not path.exists()]
    if missing:
        raise ValueError("run the " + " and ".join(missing) + " phase first")
    folder = settings.output / "test_convtu"
    record = {phase: {"path": str(path), "sha1": _sha1(path)} for phase, path in fits.items()}
    marker = folder / "fits.json"
    if marker.exists():
        if json.loads(marker.read_text()) != record:
            raise ValueError(f"the calibration, bank or z-statistics changed since {marker} was written")
    else:
        _atomic_json(marker, record)
    pending = [p for p in pilot_images(settings) if not _valid_existing(folder / f"{p.stem}.npz", SCORE_KEYS)]
    absent = [p.name for p in pending if not (settings.output / "test" / f"{p.stem}.npz").exists()]
    if absent:
        raise ValueError(f"run the test phase first: {len(absent)} detector results are missing, e.g. {absent[0]}")
    if not pending:
        return
    cuts = _cuts(settings)
    bank = load_bank(settings, settings.device)
    zstats = json.loads(fits["convtu-zstats"].read_text())
    if zstats["bank_sha1"] != _sha1(bank_path(settings)):
        raise ValueError("the bank changed since the z-statistics were computed")
    started = time.time()
    with _conv_inputs(settings) as taps:
        for done, (name, arrays) in enumerate(_variant_stream(settings, pending), start=1):
            stem = Path(name).stem
            stored = _load_npz(settings.output / "test" / f"{stem}.npz", TEST_KEYS)["digests"]
            if list(stored) != [protocol.digest(a) for a in arrays]:
                raise ValueError(f"corruptions differ from the detector pass for {name}")
            _atomic_npz(folder / f"{stem}.npz", **image_scores(taps, arrays, bank, zstats, cuts, settings.batch_size))
            if done % 5 == 0:
                _progress("convtu-scores", done, len(pending), started)


PHASES = {"convtu-calibrate": phase_calibrate, "convtu-bank": phase_bank, "convtu-zstats": phase_zstats,
          "convtu-scores": phase_scores}
