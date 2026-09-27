"""Resumable phases that compute the four baseline scores on the fixed COCO protocol."""
from __future__ import annotations

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

from ..extraction import prepare_image
from . import protocol
from .coco_quality import CocoGroundTruth, coco_map, coco_results
from .detector import DetectorTap
from .discopatch import DisCoPatchScorer, train_discopatch
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
            "epochs": self.epochs, "folds": protocol.FOLDS, "top_k": TOP_K, "knn_k": KNN_K,
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
    train_discopatch(protocol.list_images(settings.train_images), settings.output / "discopatch",
                     epochs=settings.epochs, num_workers=settings.workers, seed=settings.seed,
                     root=settings.discopatch_root)


def _discopatch_checkpoint(settings: Settings) -> Path:
    return settings.output / "discopatch" / "DisCoPatch" / "Discriminator_coco.pt"


def phase_discopatch_scores(settings: Settings) -> None:
    checkpoint = _discopatch_checkpoint(settings)
    if not checkpoint.exists():
        raise ValueError(f"train DisCoPatch first: {checkpoint} is missing")
    scorer = DisCoPatchScorer(checkpoint, settings.output / "discopatch", settings.device,
                              root=settings.discopatch_root)
    folder = settings.output / "test_dcp"
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
        _atomic_npz(folder / f"{stem}.npz", dcp=scorer.score(arrays, name))
        if done % 25 == 0:
            _progress("discopatch", done, len(pending), started)


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
    _atomic_json(settings.output / "timing.json", result)
    print(f"[timing] {result}", flush=True)


def phase_report(settings: Settings) -> None:
    from .report import build_report
    build_report(settings)


PHASES = {
    "sanity": phase_sanity, "bank": phase_bank, "test": phase_test,
    "train-discopatch": phase_train_discopatch, "discopatch-scores": phase_discopatch_scores,
    "timing": phase_timing, "report": phase_report,
}


def run_phase(phase: str, settings: Settings) -> None:
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}; choose from {sorted(PHASES)}")
    settings.output.mkdir(parents=True, exist_ok=True)
    ensure_run_config(settings)
    _record_environment(settings)
    PHASES[phase](settings)
