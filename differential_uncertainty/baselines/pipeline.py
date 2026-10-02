"""Resumable phases that compute the baseline scores on the fixed COCO protocol."""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from degradation_monitor.detector.model import prepare_image
from . import discopatch, protocol
from .activation_cdf import BINS as CDF_BINS
from .activation_cdf import MARGIN as CDF_MARGIN
from .activation_cdf import STAGES as CDF_STAGES
from .activation_cdf import CdfMonitor, ChannelRanges, ReferenceHistograms, stage_zstats, zscored_sum
from .coco_quality import CocoGroundTruth, coco_map, coco_results
from degradation_monitor.detector.taps import DetectorTap
from .discopatch import DisCoPatchScorer, train_discopatch
from .hashemi import K as HASHEMI_K
from .hashemi import LAYERS as HASHEMI_LAYERS
from .hashemi import HashemiMonitor, NeuronStats, save_intervals
from .scores import (contrastive_parts, knn_distances, normalize_rows, query_detections,
                     saod_uncertainty, top_detections)

TOP_K = 100
KNN_K = 100
KNN_K_MAX = 200
THETA = 0.3
TIMING_IMAGES = 100
TIMING_WARMUP = 10
TEST_KEYS = ("saod_min", "saod_top3", "conf_pos", "conf_neg", "knn",
             "det_scores", "det_labels", "det_boxes", "digests", "size")
ACTIVATION_KEYS = ("hashemi_decoder", "hashemi_encoder", "hashemi_encoder_maps",
                   "cdf_backbone", "cdf_backbone_z", "cdf_stages")
CDF_ZSTAT_IMAGES = 5000  # seeded sample of clean train images for the per-stage z-statistics
_PACKAGES = ("torch", "torchvision", "numpy", "scipy", "scikit-image", "scikit-learn",
             "imagecorruptions", "uq-detr", "pycocotools", "Pillow")


@dataclass(frozen=True)
class Settings:
    output: Path
    checkpoint: Path
    train_images: Path
    val_images: Path
    annotations: Path
    discopatch_root: Path
    limit: Optional[int] = None
    seed: int = 44
    epochs: int = 65
    device: str = "cuda:0"
    batch_size: int = 32
    workers: int = 9

    def experiment(self) -> dict:
        return {
            "checkpoint": str(self.checkpoint), "train_images": str(self.train_images),
            "val_images": str(self.val_images), "annotations": str(self.annotations),
            "discopatch_root": str(self.discopatch_root), "limit": self.limit, "seed": self.seed,
            "folds": protocol.FOLDS, "top_k": TOP_K, "knn_k": KNN_K,
            "knn_k_max": KNN_K_MAX, "theta": THETA,
            "conditions": [list(c) for c in protocol.CONDITIONS],
        }


def evaluation(settings: Settings) -> list[Path]:
    images = protocol.evaluation_images(settings.val_images, seed=settings.seed)
    return images[: settings.limit] if settings.limit else images


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def _load_npz(path: Path, keys) -> dict:
    try:
        with np.load(path, allow_pickle=False) as data:
            loaded = {key: data[key] for key in data.files}
    except Exception as error:
        raise ValueError(f"malformed result file {path.name}: {path}") from error
    missing = set(keys) - set(loaded)
    if missing:
        raise ValueError(f"result file {path.name} lacks {sorted(missing)}: {path}")
    return loaded


def ensure_run_config(settings: Settings) -> None:
    path = settings.output / "run_config.json"
    expected = settings.experiment()
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError(f"existing run_config.json does not match this run: {path}")
    else:
        _atomic_json(path, expected)


def _record_environment(settings: Settings) -> None:
    path = settings.output / "environment.json"
    if path.exists():
        return
    versions = {}
    for package in _PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    commit = subprocess.run(["git", "-C", str(settings.discopatch_root), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    _atomic_json(path, {"packages": versions, "discopatch_commit": commit, "gpu": gpu})


def _open(path) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(source.convert("RGB"), dtype=np.uint8).copy()


def _size(array) -> tuple[int, int]:
    return array.shape[1], array.shape[0]


def _load_bank(settings: Settings, device) -> torch.Tensor:
    path = settings.output / "bank" / "knn_bank.npy"
    if not path.exists():
        raise ValueError(f"run the bank phase first: {path} is missing")
    return normalize_rows(torch.from_numpy(np.load(path)).float()).to(device).half()


def _progress(label, done, total, started):
    rate = done / max(time.time() - started, 1e-9)
    remaining = (total - done) / max(rate, 1e-9)
    print(f"[{label}] {done}/{total} images, {rate:.2f}/s, about {remaining / 60:.0f} min left", flush=True)


def _bounded(pool, function, items, in_flight):
    """Ordered results with at most `in_flight` tasks queued, so memory stays bounded."""
    iterator = iter(items)
    # range first: zip stops on range without pulling (and losing) an extra item
    queue = deque(pool.apply_async(function, (item,)) for _, item in zip(range(in_flight), iterator))
    while queue:
        result = queue.popleft().get()
        following = next(iterator, None)
        if following is not None:
            queue.append(pool.apply_async(function, (following,)))
        yield result


def _variant_stream(settings: Settings, paths):
    if settings.workers == 0:
        for path in paths:
            yield protocol.load_variants(path)
        return
    context = multiprocessing.get_context("spawn")
    with context.Pool(settings.workers) as pool:
        yield from _bounded(pool, protocol.load_variants, paths, 2 * settings.workers)


def _detector_scores(tap, bank, arrays, batch_size, device) -> dict:
    logits, boxes, pooled = tap.run(arrays, batch_size)
    size = _size(arrays[0])
    detections = [top_detections(l, b, size, TOP_K) for l, b in zip(logits, boxes)]
    conf_pos, conf_neg = contrastive_parts([query_detections(l, b, size) for l, b in zip(logits, boxes)], THETA)
    knn = knn_distances(torch.from_numpy(pooled).to(device), bank, KNN_K_MAX).cpu().numpy()
    return {
        "saod_min": np.array([saod_uncertainty(d[0], 1) for d in detections]),
        "saod_top3": np.array([saod_uncertainty(d[0], 3) for d in detections]),
        "conf_pos": conf_pos, "conf_neg": conf_neg, "knn": knn.astype(np.float32),
        "det_scores": np.stack([d[0] for d in detections]),
        "det_labels": np.stack([d[1] for d in detections]).astype(np.int16),
        "det_boxes": np.stack([d[2] for d in detections]),
        "size": np.array(size),
    }


def phase_sanity(settings: Settings) -> float:
    gt = CocoGroundTruth(settings.annotations)
    images = protocol.list_images(settings.val_images)
    results = []
    with DetectorTap(settings.checkpoint, settings.device) as tap:
        for start in range(0, len(images), settings.batch_size):
            chunk = images[start:start + settings.batch_size]
            arrays = [_open(p) for p in chunk]
            logits, boxes, _ = tap.run(arrays, settings.batch_size)
            for path, array, l, b in zip(chunk, arrays, logits, boxes):
                s, labels, xyxy = top_detections(l, b, _size(array), TOP_K)
                results += coco_results(gt.image_id(path.name), s, labels, xyxy, gt.category_ids)
    ap = coco_map(gt, results, [gt.image_id(p.name) for p in images])
    _atomic_json(settings.output / "sanity.json", {"coco_val_ap": ap, "images": len(images)})
    print(f"[sanity] clean COCO val AP = {ap:.4f} on {len(images)} images", flush=True)
    if ap < 0.45:
        raise RuntimeError(f"clean COCO val AP is {ap:.3f}; expected about 0.48 for this checkpoint")
    return ap


class _Prepared(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as source:
            return prepare_image(source.convert("RGB"), (640, 640))


def phase_bank(settings: Settings) -> None:
    path = settings.output / "bank" / "knn_bank.npy"
    if path.exists():
        return
    paths = protocol.list_images(settings.train_images)
    loader = DataLoader(_Prepared(paths), batch_size=settings.batch_size,
                        num_workers=settings.workers, pin_memory=True)
    features, started = [], time.time()
    with DetectorTap(settings.checkpoint, settings.device) as tap:
        for index, batch in enumerate(loader):
            _, _, pooled = tap.forward(batch)
            features.append(normalize_rows(torch.from_numpy(pooled)).numpy().astype(np.float16))
            if index % 200 == 0:
                _progress("bank", min((index + 1) * settings.batch_size, len(paths)), len(paths), started)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".knn_bank.tmp.npy")
    np.save(temporary, np.concatenate(features))
    temporary.replace(path)
    _atomic_json(path.with_name("bank_names.json"), [p.name for p in paths])


def _valid_existing(path: Path, keys) -> bool:
    if not path.exists():
        return False
    _load_npz(path, keys)
    return True


def phase_test(settings: Settings) -> None:
    folder = settings.output / "test"
    pending = [p for p in evaluation(settings) if not _valid_existing(folder / f"{p.stem}.npz", TEST_KEYS)]
    if not pending:
        return
    bank = _load_bank(settings, settings.device)
    started = time.time()
    with DetectorTap(settings.checkpoint, settings.device) as tap:
        for done, (name, arrays) in enumerate(_variant_stream(settings, pending), start=1):
            values = _detector_scores(tap, bank, arrays, settings.batch_size, settings.device)
            values["digests"] = np.array([protocol.digest(a) for a in arrays])
            _atomic_npz(folder / f"{Path(name).stem}.npz", **values)
            if done % 25 == 0:
                _progress("test", done, len(pending), started)


def phase_train_discopatch(settings: Settings) -> None:
    # The official loop saves no optimiser state, so a finished discriminator is never retrained over.
    if _discopatch_checkpoint(settings).exists():
        raise ValueError(f"{_discopatch_checkpoint(settings)} already exists; remove it to train again")
    _atomic_json(settings.output / "discopatch" / "training.json",
                 {"epochs": settings.epochs, "seed": settings.seed, "numerics": discopatch.TRAINING_NUMERICS})
    train_discopatch(protocol.list_images(settings.train_images), settings.output / "discopatch",
                     epochs=settings.epochs, num_workers=settings.workers, seed=settings.seed,
                     root=settings.discopatch_root)


def _discopatch_checkpoint(settings: Settings) -> Path:
    return settings.output / "discopatch" / "DisCoPatch" / "Discriminator_coco.pt"


def phase_discopatch_scores(settings: Settings) -> None:
    checkpoint = _discopatch_checkpoint(settings)
    if not checkpoint.exists():
        raise ValueError(f"train DisCoPatch first: {checkpoint} is missing")
    folder = settings.output / "test_dcp"
    record = {"checkpoint": str(checkpoint), "sha1": hashlib.sha1(checkpoint.read_bytes()).hexdigest()}
    if (folder / "checkpoint.json").exists():
        if json.loads((folder / "checkpoint.json").read_text()) != record:
            raise ValueError(f"the discriminator checkpoint changed since {folder / 'checkpoint.json'} was written")
    else:
        _atomic_json(folder / "checkpoint.json", record)
    scorer = DisCoPatchScorer(checkpoint, settings.output / "discopatch", settings.device,
                              root=settings.discopatch_root)
    pending = [p for p in evaluation(settings) if not _valid_existing(folder / f"{p.stem}.npz", ("dcp",))]
    missing = [p.name for p in pending if not (settings.output / "test" / f"{p.stem}.npz").exists()]
    if missing:
        raise ValueError(f"run the test phase first: {len(missing)} detector results are missing, e.g. {missing[0]}")
    started = time.time()
    for done, (name, arrays) in enumerate(_variant_stream(settings, pending), start=1):
        stem = Path(name).stem
        stored = _load_npz(settings.output / "test" / f"{stem}.npz", TEST_KEYS)["digests"]
        if list(stored) != [protocol.digest(a) for a in arrays]:
            raise ValueError(f"corruptions differ from the detector pass for {name}")
        dcp = scorer.score(arrays, name)
        if not np.isfinite(dcp).all():
            raise ValueError(f"DisCoPatch returned non-finite scores for {name}")
        _atomic_npz(folder / f"{stem}.npz", dcp=dcp)
        if done % 25 == 0:
            _progress("discopatch", done, len(pending), started)


def _hashemi_intervals(settings: Settings) -> Path:
    return settings.output / "hashemi" / "intervals.npz"


def _cdf_reference(settings: Settings) -> Path:
    return settings.output / "cdf" / "reference.npz"


def _cdf_zstats(settings: Settings) -> Path:
    return settings.output / "cdf" / "zstats.json"


def _train_loader(settings: Settings, paths) -> DataLoader:
    return DataLoader(_Prepared(paths), batch_size=settings.batch_size,
                      num_workers=settings.workers, pin_memory=True)


def phase_hashemi_fit(settings: Settings) -> None:
    """Per-neuron mean and standard deviation over every clean COCO train image (Hashemi et al., Sec. 3.1)."""
    path = _hashemi_intervals(settings)
    if path.exists():
        return
    paths = protocol.list_images(settings.train_images)
    stats = {name: NeuronStats() for name in HASHEMI_LAYERS}
    started = time.time()
    with DetectorTap(settings.checkpoint, settings.device, hidden=True) as tap:
        for index, batch in enumerate(_train_loader(settings, paths)):
            hidden = tap.forward_hidden(batch)
            if len(hidden["encoder"]) != len(HASHEMI_LAYERS) - 1:
                raise ValueError(f"expected the encoder's three output maps, got {len(hidden['encoder'])}")
            stats["decoder"].update(hidden["decoder"])
            for name, values in zip(HASHEMI_LAYERS[1:], hidden["encoder"]):
                stats[name].update(values)
            if index % 200 == 0:
                _progress("hashemi-fit", min((index + 1) * settings.batch_size, len(paths)), len(paths), started)
    save_intervals(path, stats, images=len(paths))
    _atomic_json(path.with_name("fit.json"), {"images": len(paths), "k": HASHEMI_K,
                                              "std": "population (ddof=0)", "layers": list(HASHEMI_LAYERS)})


def phase_cdf_fit(settings: Settings) -> None:
    """Per-channel ranges, then training histograms of backbone stages C1-C5 (Becker et al., ICPR 2026)."""
    path = _cdf_reference(settings)
    if path.exists():
        return
    paths = protocol.list_images(settings.train_images)
    ranges = {stage: ChannelRanges() for stage in CDF_STAGES}
    with DetectorTap(settings.checkpoint, settings.device, hidden=True) as tap:
        started = time.time()
        for index, batch in enumerate(_train_loader(settings, paths)):
            stages = tap.forward_hidden(batch)["backbone"]
            if len(stages) != len(CDF_STAGES):
                raise ValueError(f"expected the five backbone stages, got {len(stages)}")
            for stage, values in zip(CDF_STAGES, stages):
                ranges[stage].update(values)
            if index % 200 == 0:
                _progress("cdf-fit ranges", min((index + 1) * settings.batch_size, len(paths)), len(paths), started)
        reference = ReferenceHistograms({s: r.result() for s, r in ranges.items()}, settings.device)
        started = time.time()
        for index, batch in enumerate(_train_loader(settings, paths)):
            reference.update(tap.forward_hidden(batch)["backbone"])
            if index % 200 == 0:
                _progress("cdf-fit histograms", min((index + 1) * settings.batch_size, len(paths)), len(paths), started)
    reference.save(path, images=len(paths))
    _atomic_json(path.with_name("fit.json"), {"images": len(paths), "bins": CDF_BINS, "margin": CDF_MARGIN,
                                              "stages": list(CDF_STAGES), "passes": 2})


def phase_cdf_zstats(settings: Settings) -> None:
    """Mean and spread of each stage's EMD over a seeded sample of clean train images (for the z-scored sum)."""
    path = _cdf_zstats(settings)
    if path.exists():
        return
    reference = _cdf_reference(settings)
    if not reference.exists():
        raise ValueError(f"run the cdf-fit phase first: {reference} is missing")
    paths = protocol.list_images(settings.train_images)
    chosen = np.sort(np.random.default_rng(settings.seed).choice(len(paths), size=min(CDF_ZSTAT_IMAGES, len(paths)),
                                                                replace=False))
    monitor = CdfMonitor(reference, settings.device)
    values = []
    with DetectorTap(settings.checkpoint, settings.device, hidden=True) as tap:
        for batch in _train_loader(settings, [paths[i] for i in chosen]):
            values.append(monitor.stage_scores(tap.forward_hidden(batch)["backbone"]))
    mean, std = stage_zstats(np.concatenate(values))
    _atomic_json(path, {"images": len(chosen), "seed": settings.seed, "stages": list(CDF_STAGES),
                        "mean": mean.tolist(), "std": std.tolist(),
                        "reference_sha1": hashlib.sha1(reference.read_bytes()).hexdigest()})


def _activation_scores(tap, hashemi_monitor, cdf_monitor, zstats, arrays, batch_size) -> dict:
    parts = {key: [] for key in ACTIVATION_KEYS}
    for start in range(0, len(arrays), batch_size):
        hidden = tap.forward_hidden(tap.prepare(arrays[start:start + batch_size]))
        decoder_share, encoder_share = hashemi_monitor.scores(hidden["decoder"], hidden["encoder"])
        stage_scores = cdf_monitor.stage_scores(hidden["backbone"])
        parts["hashemi_decoder"].append(decoder_share)
        parts["hashemi_encoder"].append(encoder_share)
        parts["hashemi_encoder_maps"].append(hashemi_monitor.encoder_shares(hidden["encoder"]))
        parts["cdf_backbone"].append(stage_scores.sum(axis=1))
        parts["cdf_backbone_z"].append(zscored_sum(stage_scores, zstats["mean"], zstats["std"]))
        parts["cdf_stages"].append(stage_scores)
    return {key: np.concatenate(values) for key, values in parts.items()}


def phase_activation_scores(settings: Settings) -> None:
    """Both activation monitors in one pass over every evaluation image and condition."""
    fits = {"hashemi-fit": _hashemi_intervals(settings), "cdf-fit": _cdf_reference(settings),
            "cdf-zstats": _cdf_zstats(settings)}
    missing = [phase for phase, path in fits.items() if not path.exists()]
    if missing:
        raise ValueError("run the " + " and ".join(missing) + " phase first")
    folder = settings.output / "test_activation"
    record = {name: {"path": str(path), "sha1": hashlib.sha1(path.read_bytes()).hexdigest()}
              for name, path in fits.items()}
    record.update(hashemi_k=HASHEMI_K, cdf_bins=CDF_BINS)
    marker = folder / "fits.json"
    if marker.exists():
        if json.loads(marker.read_text()) != record:
            raise ValueError(f"the fitted intervals or reference changed since {marker} was written")
    else:
        _atomic_json(marker, record)
    pending = [p for p in evaluation(settings) if not _valid_existing(folder / f"{p.stem}.npz", ACTIVATION_KEYS)]
    absent = [p.name for p in pending if not (settings.output / "test" / f"{p.stem}.npz").exists()]
    if absent:
        raise ValueError(f"run the test phase first: {len(absent)} detector results are missing, e.g. {absent[0]}")
    if not pending:
        return
    hashemi_monitor = HashemiMonitor(fits["hashemi-fit"], settings.device)
    cdf_monitor = CdfMonitor(fits["cdf-fit"], settings.device)
    zstats = json.loads(fits["cdf-zstats"].read_text())
    started = time.time()
    with DetectorTap(settings.checkpoint, settings.device, hidden=True) as tap:
        for done, (name, arrays) in enumerate(_variant_stream(settings, pending), start=1):
            stem = Path(name).stem
            stored = _load_npz(settings.output / "test" / f"{stem}.npz", TEST_KEYS)["digests"]
            if list(stored) != [protocol.digest(a) for a in arrays]:
                raise ValueError(f"corruptions differ from the detector pass for {name}")
            values = _activation_scores(tap, hashemi_monitor, cdf_monitor, zstats, arrays, settings.batch_size)
            if not all(np.isfinite(v).all() for v in values.values()):
                raise ValueError(f"an activation monitor returned non-finite scores for {name}")
            _atomic_npz(folder / f"{stem}.npz", **values)
            if done % 25 == 0:
                _progress("activation", done, len(pending), started)


def phase_timing(settings: Settings) -> None:
    """Median ms per image at batch 1 on a warm GPU, preprocessing included."""
    images = [_open(p) for p in evaluation(settings)[:TIMING_IMAGES]]
    bank = _load_bank(settings, settings.device)
    on_gpu = torch.cuda.is_available() and settings.device.startswith("cuda")

    def timed(function):
        for array in images[:TIMING_WARMUP]:
            function(array)
        values = []
        for array in images:
            if on_gpu:
                torch.cuda.synchronize()
            start = time.perf_counter()
            function(array)
            if on_gpu:
                torch.cuda.synchronize()
            values.append(1000.0 * (time.perf_counter() - start))
        return float(np.median(values))

    result = {"images": len(images), "batch_size": 1, "device": settings.device}
    with DetectorTap(settings.checkpoint, settings.device) as tap:
        def detector(array):
            return tap.forward(tap.prepare([array]))

        def saod(array):
            logits, boxes, _ = detector(array)
            saod_uncertainty(top_detections(logits[0], boxes[0], _size(array), TOP_K)[0], 3)

        def contrastive(array):
            logits, boxes, _ = detector(array)
            contrastive_parts([query_detections(logits[0], boxes[0], _size(array))], THETA)

        def knn(array):
            _, _, pooled = detector(array)
            knn_distances(torch.from_numpy(pooled).to(settings.device), bank, KNN_K)

        # every *_plus_* figure includes the detector forward pass it depends on
        result.update(detector_ms=timed(detector), detector_plus_saod_ms=timed(saod),
                      detector_plus_contrastive_ms=timed(contrastive), detector_plus_knn_ms=timed(knn))
    checkpoint = _discopatch_checkpoint(settings)
    if checkpoint.exists():
        scorer = DisCoPatchScorer(checkpoint, settings.output / "discopatch", settings.device,
                                  root=settings.discopatch_root)
        result["discopatch_ms"] = timed(lambda array: scorer.score([array], "timing.jpg"))
    intervals, reference = _hashemi_intervals(settings), _cdf_reference(settings)
    if intervals.exists() or reference.exists():
        # a separate tap, so the hidden-layer hooks never touch the other timings
        with DetectorTap(settings.checkpoint, settings.device, hidden=True) as hidden_tap:
            if intervals.exists():
                hashemi_monitor = HashemiMonitor(intervals, settings.device)

                def hashemi_score(array):
                    hashemi_monitor.decoder_share(hidden_tap.forward_hidden(hidden_tap.prepare([array]))["decoder"])

                result["detector_plus_hashemi_ms"] = timed(hashemi_score)
            if reference.exists():
                cdf_monitor = CdfMonitor(reference, settings.device)

                def cdf_score(array):
                    cdf_monitor.scores(hidden_tap.forward_hidden(hidden_tap.prepare([array]))["backbone"])

                result["detector_plus_cdf_ms"] = timed(cdf_score)
    _atomic_json(settings.output / "timing.json", result)
    print(f"[timing] {result}", flush=True)


def phase_report(settings: Settings) -> None:
    from .report import build_report
    build_report(settings)


PHASES = {
    "sanity": phase_sanity, "bank": phase_bank, "test": phase_test,
    "train-discopatch": phase_train_discopatch, "discopatch-scores": phase_discopatch_scores,
    "hashemi-fit": phase_hashemi_fit, "cdf-fit": phase_cdf_fit, "cdf-zstats": phase_cdf_zstats,
    "activation-scores": phase_activation_scores,
    "timing": phase_timing, "report": phase_report,
}


CONVTU_PHASES = ("convtu-calibrate", "convtu-bank", "convtu-zstats", "convtu-scores",
                 "convtu-channels", "convtu-means", "convtu-conditioned-report")


def _convtu_phase(name: str):
    def run(settings: Settings) -> None:
        from ..convtu import pipeline as convtu  # imported late: convtu.pipeline imports this module
        convtu.PHASES[name](settings)
    return run


PHASES.update({name: _convtu_phase(name) for name in CONVTU_PHASES})


def run_phase(phase: str, settings: Settings) -> None:
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}; choose from {sorted(PHASES)}")
    settings.output.mkdir(parents=True, exist_ok=True)
    ensure_run_config(settings)
    _record_environment(settings)
    PHASES[phase](settings)
