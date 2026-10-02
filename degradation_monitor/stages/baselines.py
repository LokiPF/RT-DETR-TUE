"""The baselines' stages: the clean references they need, and their per-image scores on every condition."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from .. import corruptions
from ..baselines import discopatch
from ..baselines.activation_cdf import BINS as CDF_BINS
from ..baselines.activation_cdf import MARGIN as CDF_MARGIN
from ..baselines.activation_cdf import STAGES as CDF_STAGES
from ..baselines.activation_cdf import ZSTAT_IMAGES as CDF_ZSTAT_IMAGES
from ..baselines.activation_cdf import CdfMonitor, ChannelRanges, ReferenceHistograms
from ..baselines.contrastive_conf import THETA, contrastive_parts, query_detections
from ..baselines.discopatch import DisCoPatchScorer, train_discopatch
from ..baselines.hashemi import K as HASHEMI_K
from ..baselines.hashemi import LAYERS as HASHEMI_LAYERS
from ..baselines.hashemi import HashemiMonitor, NeuronStats, save_intervals
from ..baselines.knn import KNN_K, KNN_K_MAX, knn_distances, normalize_rows
from ..baselines.saod import saod_uncertainty
from ..datasets.coco import coco_map, coco_results, list_images
from ..detector.postprocess import TOP_K, top_detections
from ..detector.taps import DetectorTap
from ..evaluation.metrics import stage_zstats, zscored_sum
from ..runs import SCORE_KEYS, atomic_json, atomic_npz, progress, sha1
from ..settings import PATH_FIELDS
from .common import (cap_gpu_memory, check_digests, clean_loader, image_size, open_rgb, pending_images,
                     variant_stream)

TIMING_IMAGES = 100
TIMING_WARMUP = 10
MIN_CLEAN_AP = 0.45  # the checkpoint's clean COCO val AP is about 0.48


def _load_bank(settings, device) -> torch.Tensor:
    path = settings.layout.knn_bank
    if not path.exists():
        raise ValueError(f"run the knn-bank stage first: {path} is missing")
    return normalize_rows(torch.from_numpy(np.load(path)).float()).to(device).half()


def detector_scores(tap, bank, arrays, batch_size, device) -> dict:
    """SAOD's confidences, ContrastiveConf's parts, kNN distances and the top detections of every array."""
    logits, boxes, pooled = tap.run(arrays, batch_size)
    size = image_size(arrays[0])
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


def check(settings, manifest) -> float:
    """Every input exists, then the detector's clean AP on every COCO val image: about 0.48 for this checkpoint.

    run_stage has already recorded the checkpoint's sha256 in the protocol.
    """
    for name in PATH_FIELDS:
        if name != "run" and not getattr(settings, name).exists():
            raise ValueError(f"{name} does not exist: {getattr(settings, name)}")
    gt = settings.dataset.ground_truth()
    images = list_images(settings.val_images)
    results = []
    cap_gpu_memory(settings.device, settings.gpu_memory_gib)
    with DetectorTap(settings.checkpoint, settings.device) as tap:
        for start in range(0, len(images), settings.batch_size):
            chunk = images[start:start + settings.batch_size]
            arrays = [open_rgb(p) for p in chunk]
            logits, boxes, _ = tap.run(arrays, settings.batch_size)
            for path, array, l, b in zip(chunk, arrays, logits, boxes):
                s, labels, xyxy = top_detections(l, b, image_size(array), TOP_K)
                results += coco_results(gt.image_id(path.name), s, labels, xyxy, gt.category_ids)
    ap = coco_map(gt, results, [gt.image_id(p.name) for p in images])
    manifest.update(check={"coco_val_ap": ap, "images": len(images)})
    print(f"[check] clean COCO val AP = {ap:.4f} on {len(images)} images", flush=True)
    if ap < MIN_CLEAN_AP:
        raise RuntimeError(f"clean COCO val AP is {ap:.3f}; expected about 0.48 for this checkpoint")
    return ap


def knn_bank(settings, manifest) -> None:
    """The kNN baseline's bank: L2-normalised pooled last-stage features of every clean train image, float16."""
    layout = settings.layout
    if layout.knn_bank.exists():
        return
    paths = settings.dataset.train_images()
    features, started = [], time.time()
    cap_gpu_memory(settings.device, settings.gpu_memory_gib)
    with DetectorTap(settings.checkpoint, settings.device) as tap:
        for index, batch in enumerate(clean_loader(settings, paths)):
            _, _, pooled = tap.forward(batch)
            features.append(normalize_rows(torch.from_numpy(pooled)).numpy().astype(np.float16))
            if index % 200 == 0:
                progress("knn-bank", min((index + 1) * settings.batch_size, len(paths)), len(paths), started)
    layout.knn_bank.parent.mkdir(parents=True, exist_ok=True)
    temporary = layout.knn_bank.with_name(".bank.tmp.npy")
    np.save(temporary, np.concatenate(features))
    temporary.replace(layout.knn_bank)
    atomic_json(layout.knn_names, [p.name for p in paths])


def detector_pass(settings, manifest) -> None:
    """Every evaluation image under all 96 conditions through the detector: SAOD, ContrastiveConf, kNN, detections."""
    layout = settings.layout
    pending = pending_images(settings, "detector")
    if not pending:
        return
    bank = _load_bank(settings, settings.device)
    manifest.check_inputs("detector", {"knn_bank": sha1(layout.knn_bank)})
    cap_gpu_memory(settings.device, settings.gpu_memory_gib)
    started = time.time()
    with DetectorTap(settings.checkpoint, settings.device) as tap:
        for done, (name, arrays) in enumerate(variant_stream(settings, pending), start=1):
            values = detector_scores(tap, bank, arrays, settings.batch_size, settings.device)
            values["digests"] = np.array([corruptions.digest(a) for a in arrays])
            atomic_npz(layout.score_file("detector", name), **values)
            if done % 25 == 0:
                progress("detector-pass", done, len(pending), started)


def discopatch_train(settings, manifest) -> None:
    """Train DisCoPatch's discriminator on clean train patches with the official code and README settings."""
    layout = settings.layout
    # The official loop saves no optimiser state, so a finished discriminator is never trained over.
    if layout.discopatch_checkpoint.exists():
        raise ValueError(f"{layout.discopatch_checkpoint} already exists; remove it to train again")
    atomic_json(layout.discopatch_training,
                {"epochs": settings.epochs, "seed": settings.seed, "numerics": discopatch.TRAINING_NUMERICS})
    trained = train_discopatch(settings.dataset.train_images(), layout.discopatch_dir / "training",
                               epochs=settings.epochs, num_workers=settings.workers, seed=settings.seed,
                               root=settings.discopatch_root)
    os.link(trained, layout.discopatch_checkpoint)


def discopatch_pass(settings, manifest) -> None:
    layout = settings.layout
    checkpoint = layout.discopatch_checkpoint
    if not checkpoint.exists():
        raise ValueError(f"run the discopatch-train stage first: {checkpoint} is missing")
    manifest.check_inputs("discopatch", {"discriminator": sha1(checkpoint)})
    pending = pending_images(settings, "discopatch")
    if not pending:
        return
    cap_gpu_memory(settings.device, settings.gpu_memory_gib)
    scorer = DisCoPatchScorer(checkpoint, layout.discopatch_dir / "training", settings.device,
                              root=settings.discopatch_root)
    started = time.time()
    for done, (name, arrays) in enumerate(variant_stream(settings, pending), start=1):
        check_digests(layout, name, arrays)
        dcp = scorer.score(arrays, name)
        if not np.isfinite(dcp).all():
            raise ValueError(f"DisCoPatch returned non-finite scores for {name}")
        atomic_npz(layout.score_file("discopatch", name), dcp=dcp)
        if done % 25 == 0:
            progress("discopatch-pass", done, len(pending), started)


def hashemi_fit(settings, manifest) -> None:
    """Per-neuron mean and standard deviation over every clean COCO train image (Hashemi et al., Sec. 3.1)."""
    layout = settings.layout
    if layout.hashemi_intervals.exists():
        return
    paths = settings.dataset.train_images()
    stats = {name: NeuronStats() for name in HASHEMI_LAYERS}
    cap_gpu_memory(settings.device, settings.gpu_memory_gib)
    started = time.time()
    with DetectorTap(settings.checkpoint, settings.device, hidden=True) as tap:
        for index, batch in enumerate(clean_loader(settings, paths)):
            hidden = tap.forward_hidden(batch)
            if len(hidden["encoder"]) != len(HASHEMI_LAYERS) - 1:
                raise ValueError(f"expected the encoder's three output maps, got {len(hidden['encoder'])}")
            stats["decoder"].update(hidden["decoder"])
            for name, values in zip(HASHEMI_LAYERS[1:], hidden["encoder"]):
                stats[name].update(values)
            if index % 200 == 0:
                progress("hashemi-fit", min((index + 1) * settings.batch_size, len(paths)), len(paths), started)
    save_intervals(layout.hashemi_intervals, stats, images=len(paths))
    atomic_json(layout.hashemi_fit, {"images": len(paths), "k": HASHEMI_K, "std": "population (ddof=0)",
                                     "layers": list(HASHEMI_LAYERS)})


def cdf_fit(settings, manifest) -> None:
    """Per-channel ranges, then training histograms of backbone stages C1-C5 (Becker et al., ICPR 2026)."""
    layout = settings.layout
    if layout.cdf_reference.exists():
        return
    paths = settings.dataset.train_images()
    ranges = {stage: ChannelRanges() for stage in CDF_STAGES}
    cap_gpu_memory(settings.device, settings.gpu_memory_gib)
    with DetectorTap(settings.checkpoint, settings.device, hidden=True) as tap:
        started = time.time()
        for index, batch in enumerate(clean_loader(settings, paths)):
            stages = tap.forward_hidden(batch)["backbone"]
            if len(stages) != len(CDF_STAGES):
                raise ValueError(f"expected the five backbone stages, got {len(stages)}")
            for stage, values in zip(CDF_STAGES, stages):
                ranges[stage].update(values)
            if index % 200 == 0:
                progress("cdf-fit ranges", min((index + 1) * settings.batch_size, len(paths)), len(paths), started)
        reference = ReferenceHistograms({s: r.result() for s, r in ranges.items()}, settings.device)
        started = time.time()
        for index, batch in enumerate(clean_loader(settings, paths)):
            reference.update(tap.forward_hidden(batch)["backbone"])
            if index % 200 == 0:
                progress("cdf-fit histograms", min((index + 1) * settings.batch_size, len(paths)), len(paths), started)
    reference.save(layout.cdf_reference, images=len(paths))
    atomic_json(layout.cdf_fit, {"images": len(paths), "bins": CDF_BINS, "margin": CDF_MARGIN,
                                 "stages": list(CDF_STAGES), "passes": 2})


def cdf_zstats(settings, manifest) -> None:
    """Mean and spread of each stage's EMD over a seeded sample of clean train images, for the z-scored sum."""
    layout = settings.layout
    if layout.cdf_zstats.exists():
        return
    if not layout.cdf_reference.exists():
        raise ValueError(f"run the cdf-fit stage first: {layout.cdf_reference} is missing")
    paths = settings.dataset.train_images()
    chosen = np.sort(np.random.default_rng(settings.seed).choice(len(paths), size=min(CDF_ZSTAT_IMAGES, len(paths)),
                                                                replace=False))
    monitor = CdfMonitor(layout.cdf_reference, settings.device)
    values = []
    cap_gpu_memory(settings.device, settings.gpu_memory_gib)
    with DetectorTap(settings.checkpoint, settings.device, hidden=True) as tap:
        for batch in clean_loader(settings, [paths[i] for i in chosen]):
            values.append(monitor.stage_scores(tap.forward_hidden(batch)["backbone"]))
    mean, std = stage_zstats(np.concatenate(values))
    atomic_json(layout.cdf_zstats, {"images": len(chosen), "seed": settings.seed, "stages": list(CDF_STAGES),
                                    "mean": mean.tolist(), "std": std.tolist(),
                                    "reference_sha1": sha1(layout.cdf_reference)})


def activation_scores(tap, hashemi_monitor, cdf_monitor, zstats, arrays, batch_size) -> dict:
    """Both activation monitors' scores for every array, from one hidden-layer pass."""
    parts = {key: [] for key in SCORE_KEYS["activations"]}
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


def activation_pass(settings, manifest) -> None:
    """Both activation monitors in one pass over every evaluation image and condition."""
    layout = settings.layout
    fits = {"hashemi-fit": layout.hashemi_intervals, "cdf-fit": layout.cdf_reference, "cdf-zstats": layout.cdf_zstats}
    missing = [stage for stage, path in fits.items() if not path.exists()]
    if missing:
        raise ValueError("run the " + " and ".join(missing) + " stage first")
    manifest.check_inputs("activations", {"hashemi_intervals": sha1(layout.hashemi_intervals),
                                          "cdf_reference": sha1(layout.cdf_reference),
                                          "cdf_zstats": sha1(layout.cdf_zstats),
                                          "hashemi_k": HASHEMI_K, "cdf_bins": CDF_BINS})
    pending = pending_images(settings, "activations")
    if not pending:
        return
    hashemi_monitor = HashemiMonitor(layout.hashemi_intervals, settings.device)
    cdf_monitor = CdfMonitor(layout.cdf_reference, settings.device)
    zstats = json.loads(layout.cdf_zstats.read_text())
    cap_gpu_memory(settings.device, settings.gpu_memory_gib)
    started = time.time()
    with DetectorTap(settings.checkpoint, settings.device, hidden=True) as tap:
        for done, (name, arrays) in enumerate(variant_stream(settings, pending), start=1):
            check_digests(layout, name, arrays)
            values = activation_scores(tap, hashemi_monitor, cdf_monitor, zstats, arrays, settings.batch_size)
            if not all(np.isfinite(v).all() for v in values.values()):
                raise ValueError(f"an activation monitor returned non-finite scores for {name}")
            atomic_npz(layout.score_file("activations", name), **values)
            if done % 25 == 0:
                progress("activation-pass", done, len(pending), started)


def timing(settings, manifest) -> None:
    """Median ms per image at batch 1 on a warm GPU, preprocessing included."""
    layout = settings.layout
    images = [open_rgb(p) for p in settings.dataset.evaluation_images()[:TIMING_IMAGES]]
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
            saod_uncertainty(top_detections(logits[0], boxes[0], image_size(array), TOP_K)[0], 3)

        def contrastive(array):
            logits, boxes, _ = detector(array)
            contrastive_parts([query_detections(logits[0], boxes[0], image_size(array))], THETA)

        def knn(array):
            _, _, pooled = detector(array)
            knn_distances(torch.from_numpy(pooled).to(settings.device), bank, KNN_K)

        # every *_plus_* figure includes the detector forward pass it depends on
        result.update(detector_ms=timed(detector), detector_plus_saod_ms=timed(saod),
                      detector_plus_contrastive_ms=timed(contrastive), detector_plus_knn_ms=timed(knn))
    if layout.discopatch_checkpoint.exists():
        scorer = DisCoPatchScorer(layout.discopatch_checkpoint, layout.discopatch_dir / "training", settings.device,
                                  root=settings.discopatch_root)
        result["discopatch_ms"] = timed(lambda array: scorer.score([array], "timing.jpg"))
    if layout.hashemi_intervals.exists() or layout.cdf_reference.exists():
        # a separate tap, so the hidden-layer hooks never touch the other timings
        with DetectorTap(settings.checkpoint, settings.device, hidden=True) as hidden_tap:
            if layout.hashemi_intervals.exists():
                hashemi_monitor = HashemiMonitor(layout.hashemi_intervals, settings.device)

                def hashemi_score(array):
                    hashemi_monitor.decoder_share(hidden_tap.forward_hidden(hidden_tap.prepare([array]))["decoder"])

                result["detector_plus_hashemi_ms"] = timed(hashemi_score)
            if layout.cdf_reference.exists():
                cdf_monitor = CdfMonitor(layout.cdf_reference, settings.device)

                def cdf_score(array):
                    cdf_monitor.scores(hidden_tap.forward_hidden(hidden_tap.prepare([array]))["backbone"])

                result["detector_plus_cdf_ms"] = timed(cdf_score)
    atomic_json(layout.timing, result)
    print(f"[timing] {result}", flush=True)
