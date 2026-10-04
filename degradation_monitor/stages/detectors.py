"""Three more frozen COCO detectors on COCO-C: their clean fits, one shared pass, their reports and a table.

    python -m degradation_monitor.stages.detectors <stage> [--config configs/coco-detectors.toml] [--first N]

Stages, each over every detector of the config:
- check: the detector's clean AP on every val image (COCO's AP@[.5:.95]) must reach its floor;
- fit: the clean references, in three passes over COCO train images, each skipped when its files exist: (1) the kNN
  bank, the CDF ranges, the method's bank and z-statistics and, for DETR-type detectors, Hashemi's decoder intervals;
  (2) the CDF histograms; (3) the CDFs' z-statistics;
- pass: every val image's 96 versions, generated once, through every detector: a detector, an activations and a
  method file per image and detector; --first N stops after N images, to time the pass;
- report: each detector's report (reports/coco/ in its folder), then the table of all four detectors in summary.md;
  --first N reports on the first N evaluation images only, a smoke test that the full report overwrites.

Each detector's folder (<run>/<detector>/) is laid out like runs/coco/. Its manifest's protocol adds the detector's
name and its adapter's protocol (taps, input size, thresholds) to RT-DETR's, with the detector's weights. DisCoPatch
reads the image, not the detector, so its scores are hard links to the reference run's. Every corrupted image must
match the reference run's digest.
"""
from __future__ import annotations

import argparse
import csv
import errno
import json
import os
import shutil
import sys
import time
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .. import corruptions
from ..baselines.activation_cdf import BINS as CDF_BINS
from ..baselines.activation_cdf import MARGIN as CDF_MARGIN
from ..baselines.activation_cdf import STAGES as CDF_STAGES
from ..baselines.activation_cdf import ZSTAT_IMAGES as CDF_ZSTAT_IMAGES
from ..baselines.activation_cdf import CdfMonitor, ChannelRanges, ReferenceHistograms
from ..baselines.contrastive_conf import THETA, contrastive_parts, query_detections
from ..baselines.hashemi import K as HASHEMI_K
from ..baselines.hashemi import HashemiMonitor, NeuronStats, save_intervals
from ..baselines.knn import KNN_K_MAX, knn_distances, normalize_rows
from ..baselines.saod import saod_uncertainty
from ..cli import positive_int
from ..datasets.coco import coco_map, coco_results, list_images
from ..detectors import DETECTORS, LEVELS, adapter_class, load_adapter
from ..evaluation.metrics import stage_zstats, zscored_sum
from ..evaluation.report import OURS
from ..method.statistics import KEYS, channel_statistics
from ..runs import Manifest, RunLayout, atomic_json, atomic_npz, load_npz, progress, sha1
from ..settings import load_settings
from .common import cap_gpu_memory, image_size, variant_stream
from .report import write_report

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "coco-detectors.toml"
CONFIG_KEYS = {"base", "run", "reference_run", "gpu_memory_gib", "weights", "clean_ap_floor"}
FOLDERS = ("detector", "activations", "method")
PROGRESS_IMAGES = 2000  # the fit reports its progress about every this many images


@dataclass(frozen=True)
class DetectorsConfig:
    base: object  # the base config's Settings: the dataset, the seed, the device and the workers
    run: Path
    reference_run: Path  # RT-DETR's run: the corruption digests and DisCoPatch's scores
    weights: dict
    floors: dict
    gpu_memory_gib: float

    @property
    def detectors(self) -> tuple:
        return tuple(self.weights)

    def settings(self, name: str):
        """One detector's Settings: its run folder, and its weights as the protocol's checkpoint."""
        return replace(self.base, run=self.run / name, checkpoint=Path(self.weights[name]),
                       gpu_memory_gib=self.gpu_memory_gib)


def load_config(path, run=None) -> DetectorsConfig:
    path = Path(path)
    values = tomllib.loads(path.read_text())
    unknown = sorted(set(values) - CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unknown settings in {path}: {', '.join(unknown)}")
    weights = {name: Path(value) for name, value in values["weights"].items()}
    strange = sorted(set(weights) - set(DETECTORS))
    if strange:
        raise ValueError(f"unknown detectors in {path}: {', '.join(strange)}; choose from {', '.join(DETECTORS)}")
    if set(values["clean_ap_floor"]) != set(weights):
        raise ValueError(f"{path}: every detector needs a clean AP floor, and only those")
    return DetectorsConfig(base=load_settings(path.parent / values["base"]), run=Path(run or values["run"]),
                           reference_run=Path(values["reference_run"]), weights=weights,
                           floors=dict(values["clean_ap_floor"]), gpu_memory_gib=float(values["gpu_memory_gib"]))


def _manifest(config, name) -> Manifest:
    """The detector folder's manifest, once its protocol is checked: the weights, the evaluation and the adapter's."""
    settings = config.settings(name)
    settings.layout.root.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(settings.layout)
    manifest.check_protocol({**settings.protocol(), "detector": name, "adapter": adapter_class(name).protocol,
                             "float32_matmul_precision": torch.get_float32_matmul_precision()})
    manifest.record_environment(settings.discopatch_root)
    return manifest


class _RgbImages(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as source:
            return np.asarray(source.convert("RGB"), dtype=np.uint8).copy()


def _batches(paths, size, workers):
    """Lists of `size` RGB arrays, in the order of `paths`."""
    return DataLoader(_RgbImages(paths), batch_size=size, num_workers=workers, collate_fn=list)


def _method_statistics(out) -> dict:
    """Each channel's level and top-1% mean in the four levels: KEYS -> float32 (N, C)."""
    return {f"{statistic}_{level}": values for level in LEVELS
            for statistic, values in channel_statistics(out.levels[level]).items()}


def _report_peak_memory(label, device) -> None:
    """The process's peak GPU memory so far, for the sessions that share the card."""
    if torch.device(device).type == "cuda":
        print(f"[{label}] peak GPU memory {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB", flush=True)


def check(config) -> None:
    """Every detector's clean AP on every COCO val image; refuses a detector below its floor."""
    failed = []
    for name in config.detectors:
        settings, manifest = config.settings(name), _manifest(config, name)
        gt = settings.dataset.ground_truth()
        images = list_images(settings.val_images)
        cap_gpu_memory(settings.device, settings.gpu_memory_gib)
        adapter = load_adapter(name, config.weights[name], settings.device)
        results = []
        for path, arrays in zip(images, _batches(images, 1, settings.workers)):  # val images differ in size
            out = adapter(arrays)
            results += coco_results(gt.image_id(path.name), out.scores[0], out.labels[0], out.boxes[0],
                                    gt.category_ids)  # an empty slot scores 0 with an empty box: it never matches
        adapter.close()
        ap = coco_map(gt, results, [gt.image_id(p.name) for p in images])
        manifest.update(check={"coco_val_ap": ap, "images": len(images), "floor": config.floors[name]})
        print(f"[check] {name}: clean COCO val AP = {ap:.4f} on {len(images)} images", flush=True)
        if ap < config.floors[name]:
            failed.append(f"{name} {ap:.3f} (floor {config.floors[name]})")
    if failed:
        raise RuntimeError("clean COCO val AP below the floor: " + "; ".join(failed))


def _ranges_path(layout) -> Path:
    return layout.reference("activation_cdf", "ranges.npz")


def _clean_pass_done(layout, detr: bool) -> bool:
    paths = [layout.knn_bank, layout.method_bank, layout.method_zstats, _ranges_path(layout)]
    return all(path.exists() for path in paths + ([layout.hashemi_intervals] if detr else []))


def _fitted(layout, detr: bool) -> bool:
    return _clean_pass_done(layout, detr) and layout.cdf_reference.exists() and layout.cdf_zstats.exists()


def _clean_pass(adapter, settings) -> None:
    """Pass 1 over every clean train image: the kNN bank, the CDF ranges, the method's bank and z-statistics, and for
    DETR-type detectors Hashemi's decoder intervals. The ranges are written last: they mark the pass as done."""
    layout, dataset = settings.layout, settings.dataset
    paths = dataset.train_images()
    splits = {split: [str(p) for p in dataset.reference_split(split)] for split in ("bank", "zstats")}
    wanted = {p for members in splits.values() for p in members}
    pooled, ranges, method = [], [ChannelRanges() for _ in CDF_STAGES], {}
    decoder = NeuronStats() if adapter.detr else None
    size, started = adapter.fit_batch_size, time.time()
    for index, arrays in enumerate(_batches(paths, size, settings.workers)):
        names = [str(p) for p in paths[index * size:(index + 1) * size]]
        out = adapter(arrays, heads=adapter.detr)  # only Hashemi reads past the backbone
        pooled.append(normalize_rows(out.pooled).cpu().numpy().astype(np.float16))
        for kept, maps in zip(ranges, out.cdf):
            kept.update(maps)
        if decoder is not None:
            decoder.update(out.decoder)
        if wanted.intersection(names):
            statistics = _method_statistics(out)
            method.update({p: {k: v[row] for k, v in statistics.items()} for row, p in enumerate(names) if p in wanted})
        if index % max(1, PROGRESS_IMAGES // size) == 0:
            progress(f"fit {adapter.name} 1/3", min((index + 1) * size, len(paths)), len(paths), started)
    layout.knn_bank.parent.mkdir(parents=True, exist_ok=True)
    temporary = layout.knn_bank.with_name(".bank.tmp.npy")
    np.save(temporary, np.concatenate(pooled))
    temporary.replace(layout.knn_bank)
    atomic_json(layout.knn_names, [p.name for p in paths])
    for split, path in (("bank", layout.method_bank), ("zstats", layout.method_zstats)):
        atomic_npz(path, **{key: np.stack([method[p][key] for p in splits[split]]) for key in KEYS})
    if decoder is not None:
        save_intervals(layout.hashemi_intervals, {"decoder": decoder}, images=len(paths))
        atomic_json(layout.hashemi_fit, {"images": len(paths), "k": HASHEMI_K, "std": "population (ddof=0)",
                                         "layers": ["decoder"]})
    bounds = {stage: kept.result() for stage, kept in zip(CDF_STAGES, ranges)}
    atomic_npz(_ranges_path(layout), **{f"{stage}_{bound}": values for stage, pair in bounds.items()
                                        for bound, values in zip(("low", "high"), pair)})


def _cdf_histograms(adapter, settings) -> None:
    """Pass 2: every clean train image's CDF maps, counted into histograms on the ranges of pass 1."""
    layout = settings.layout
    paths = settings.dataset.train_images()
    with np.load(_ranges_path(layout)) as data:
        bounds = {stage: (data[f"{stage}_low"], data[f"{stage}_high"]) for stage in CDF_STAGES}
    reference = ReferenceHistograms(bounds, settings.device)
    size, started = adapter.fit_batch_size, time.time()
    for index, arrays in enumerate(_batches(paths, size, settings.workers)):
        reference.update(adapter(arrays, heads=False).cdf)
        if index % max(1, PROGRESS_IMAGES // size) == 0:
            progress(f"fit {adapter.name} 2/3", min((index + 1) * size, len(paths)), len(paths), started)
    reference.save(layout.cdf_reference, images=len(paths))
    atomic_json(layout.cdf_fit, {"images": len(paths), "bins": CDF_BINS, "margin": CDF_MARGIN,
                                 "stages": list(CDF_STAGES), "passes": 2})


def _cdf_zstats(adapter, settings) -> None:
    """Pass 3: each CDF stage's mean and spread over a seeded sample of clean train images."""
    layout = settings.layout
    paths = settings.dataset.train_images()
    chosen = np.sort(np.random.default_rng(settings.seed).choice(len(paths), size=min(CDF_ZSTAT_IMAGES, len(paths)),
                                                                replace=False))
    monitor, values = CdfMonitor(layout.cdf_reference, settings.device), []
    for arrays in _batches([paths[i] for i in chosen], adapter.fit_batch_size, settings.workers):
        values.append(monitor.stage_scores(adapter(arrays, heads=False).cdf))
    mean, std = stage_zstats(np.concatenate(values))
    atomic_json(layout.cdf_zstats, {"images": len(chosen), "seed": settings.seed, "stages": list(CDF_STAGES),
                                    "mean": mean.tolist(), "std": std.tolist(),
                                    "reference_sha1": sha1(layout.cdf_reference)})


def fit(config) -> None:
    """Every detector's clean references, from the first pass whose files are missing onward."""
    for name in config.detectors:
        settings = config.settings(name)
        _manifest(config, name)
        layout, detr = settings.layout, adapter_class(name).detr
        passes = ((_clean_pass, _clean_pass_done(layout, detr)), (_cdf_histograms, layout.cdf_reference.exists()),
                  (_cdf_zstats, layout.cdf_zstats.exists()))
        start = next((i for i, (_, done) in enumerate(passes) if not done), None)
        if start is None:
            continue
        cap_gpu_memory(settings.device, settings.gpu_memory_gib)
        adapter = load_adapter(name, config.weights[name], settings.device)
        for step, _ in passes[start:]:  # a redone pass makes every later pass stale
            step(adapter, settings)
        adapter.close()
        _report_peak_memory(f"fit {name}", settings.device)


def _references(config, name) -> dict:
    settings = config.settings(name)
    layout, device = settings.layout, settings.device
    bank = normalize_rows(torch.from_numpy(np.load(layout.knn_bank)).float()).to(device).half()
    return {"knn_bank": bank, "cdf": CdfMonitor(layout.cdf_reference, device),
            "cdf_zstats": json.loads(layout.cdf_zstats.read_text()),
            "hashemi": HashemiMonitor(layout.hashemi_intervals, device) if adapter_class(name).detr else None}


def _inputs(layout, detr: bool) -> dict:
    """What each score folder is computed from: the manifest refuses a pass after a fit changed."""
    activations = {"cdf_reference": sha1(layout.cdf_reference), "cdf_zstats": sha1(layout.cdf_zstats),
                   "cdf_bins": CDF_BINS}
    if detr:
        activations.update(hashemi_intervals=sha1(layout.hashemi_intervals), hashemi_k=HASHEMI_K)
    return {"detector": {"knn_bank": sha1(layout.knn_bank)}, "activations": activations,
            "method": {"method_bank": sha1(layout.method_bank), "method_zstats": sha1(layout.method_zstats)}}


def image_files(adapter, references, arrays) -> dict:
    """The detector, activations and method arrays of one image's versions, from one forward pass per batch."""
    size = image_size(arrays[0])
    parts = {folder: [] for folder in FOLDERS}
    for start in range(0, len(arrays), adapter.batch_size):
        out = adapter(arrays[start:start + adapter.batch_size])
        detector = {"saod_min": np.array([saod_uncertainty(s, 1) for s in out.scores]),
                    "saod_top3": np.array([saod_uncertainty(s, 3) for s in out.scores]),
                    "knn": knn_distances(out.pooled.to(references["knn_bank"].device), references["knn_bank"],
                                         KNN_K_MAX).cpu().numpy().astype(np.float32),
                    "det_scores": out.scores, "det_labels": out.labels.astype(np.int16), "det_boxes": out.boxes}
        if out.query_logits is not None:
            detector["conf_pos"], detector["conf_neg"] = contrastive_parts(
                [query_detections(l, b, size) for l, b in zip(out.query_logits, out.query_boxes)], THETA)
        stage_scores, zstats = references["cdf"].stage_scores(out.cdf), references["cdf_zstats"]
        activations = {"cdf_backbone": stage_scores.sum(axis=1),
                       "cdf_backbone_z": zscored_sum(stage_scores, zstats["mean"], zstats["std"]),
                       "cdf_stages": stage_scores}
        if references["hashemi"] is not None:
            activations["hashemi_decoder"] = references["hashemi"].decoder_share(out.decoder)
        for folder, values in zip(FOLDERS, (detector, activations, _method_statistics(out))):
            parts[folder].append(values)
        del out  # its maps are views into the full feature maps: free them before the next forward pass
    files = {folder: {key: np.concatenate([p[key] for p in pieces]) for key in pieces[0]}
             for folder, pieces in parts.items()}
    for folder, values in files.items():
        if not all(np.isfinite(v).all() for v in values.values() if v.dtype.kind == "f"):
            raise ValueError(f"{adapter.name} gave non-finite {folder} values")
    files["detector"]["size"] = np.array(size)
    return files


def _complete(layout, image) -> bool:
    return all(layout.score_file(folder, image).exists() for folder in FOLDERS)


def shared_pass(config, first=None) -> None:
    """Every evaluation image's 96 versions, generated once, through every detector; resumable image by image."""
    names = config.detectors
    layouts = {name: config.settings(name).layout for name in names}
    for name in names:
        manifest, detr = _manifest(config, name), adapter_class(name).detr
        if not _fitted(layouts[name], detr):
            raise ValueError(f"run the fit stage first: {name}'s references are missing")
        for folder, inputs in _inputs(layouts[name], detr).items():
            manifest.check_inputs(folder, inputs)
    base = config.base
    pending = [p for p in base.dataset.evaluation_images() if not all(_complete(layouts[n], p) for n in names)]
    pending = pending[:first] if first else pending
    if not pending:
        return
    cap_gpu_memory(base.device, config.gpu_memory_gib)
    adapters = {name: load_adapter(name, config.weights[name], base.device) for name in names}
    references = {name: _references(config, name) for name in names}
    reference_run, started = RunLayout(config.reference_run), time.time()
    for done, (image, arrays) in enumerate(variant_stream(base, pending), start=1):
        digests = np.array([corruptions.digest(a) for a in arrays])
        if list(load_npz(reference_run.score_file("detector", image), ("digests",))["digests"]) != list(digests):
            raise ValueError(f"corruptions differ from the reference run for {image}")
        for name, adapter in adapters.items():
            if not _complete(layouts[name], image):
                files = image_files(adapter, references[name], arrays)
                files["detector"]["digests"] = digests
                for folder in FOLDERS:
                    atomic_npz(layouts[name].score_file(folder, image), **files[folder])
        if done % 25 == 0:
            progress("detectors pass", done, len(pending), started)
    for adapter in adapters.values():
        adapter.close()
    _report_peak_memory("pass", base.device)


def _link_discopatch(config, layout, manifest) -> None:
    """DisCoPatch's scores do not depend on the detector: hard links to the reference run's files."""
    reference = RunLayout(config.reference_run)
    inputs = Manifest(reference).read().get("inputs", {}).get("discopatch")
    if inputs is None:
        raise ValueError(f"the reference run records no DisCoPatch inputs: {reference.manifest}")
    manifest.check_inputs("discopatch", {**inputs, "reference_run": str(config.reference_run)})
    target = layout.scores("discopatch")
    target.mkdir(parents=True, exist_ok=True)
    for path in reference.scores("discopatch").glob("*.npz"):
        if not (target / path.name).exists():
            try:
                os.link(path, target / path.name)
            except OSError as error:  # a run root on another filesystem: copy instead
                if error.errno != errno.EXDEV:
                    raise
                shutil.copy2(path, target / path.name)


def _auroc(summary, subset, row, group):
    """A row's AUROC on one image set; None when a smoke report has no such set."""
    return summary["headline"].get(subset, {}).get(row, {}).get(f"auroc_{group}")


def _cell(value) -> str:
    return "–" if value is None else f"{value:.3f}"


def cross_table(config) -> list:
    """RT-DETR's and every detector's headline: the two-axis score, the CDFs, the best baseline and the clean mAP."""
    sources = {"rtdetrv2_r18": RunLayout(config.reference_run).report() / "summary.json"}
    sources.update({name: config.settings(name).layout.report() / "summary.json" for name in config.detectors})
    rows = []
    for name, path in sources.items():
        summary = json.loads(path.read_text())
        row = {"detector": name, "images": summary["images"], "clean_map": summary["clean_map"],
               "headline_decision": summary["headline_decision"]}
        for subset in ("all", "untouched"):
            for method in ("two_axis", "cdf"):
                for group in ("common", "extra"):
                    row[f"{subset}_{method}_auroc_{group}"] = _auroc(summary, subset, method, group)
        row["two_axis_severity1_common"] = summary["by_severity"]["all"]["two_axis"]["common"][0]
        head = summary["headline"]["all"]
        best = max((m for m in head if m not in OURS), key=lambda m: head[m]["auroc_common"])
        row["best_baseline"], row["best_baseline_auroc_common"] = best, head[best]["auroc_common"]
        rows.append(row)
    config.run.mkdir(parents=True, exist_ok=True)
    with (config.run / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# The two-axis score on four COCO detectors", "",
             "AUROC, common / extra families. Untouched: positions 1970 and later.", "",
             "| Detector | Images | Clean mAP | Two-axis, all | Two-axis, untouched | Two-axis, severity 1 common "
             "| CDFs, all | Best baseline, common | Headline |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['detector']} | {r['images']} | {_cell(r['clean_map'])} "
            f"| {_cell(r['all_two_axis_auroc_common'])} / {_cell(r['all_two_axis_auroc_extra'])} "
            f"| {_cell(r['untouched_two_axis_auroc_common'])} / {_cell(r['untouched_two_axis_auroc_extra'])} "
            f"| {_cell(r['two_axis_severity1_common'])} "
            f"| {_cell(r['all_cdf_auroc_common'])} / {_cell(r['all_cdf_auroc_extra'])} "
            f"| {r['best_baseline']} {_cell(r['best_baseline_auroc_common'])} | {r['headline_decision']} |")
    (config.run / "summary.md").write_text("\n".join(lines) + "\n")
    return rows


def report(config, first=None) -> None:
    """Each detector's report, then the table of all four; with `first`, a smoke report on the first images."""
    for name in config.detectors:
        settings, manifest = config.settings(name), _manifest(config, name)
        _link_discopatch(config, settings.layout, manifest)
        write_report(replace(settings, limit=first) if first else settings, manifest)
    cross_table(config)


STAGES = {"check": check, "fit": fit, "pass": shared_pass, "report": report}
WITH_FIRST = ("pass", "report")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m degradation_monitor.stages.detectors",
                                     description="Three more frozen COCO detectors on COCO-C: run one stage.")
    parser.add_argument("stage", choices=list(STAGES))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run", type=Path, help="the run root, instead of the config's")
    parser.add_argument("--first", type=positive_int,
                        help="pass: stop after N images; report: a smoke report on the first N evaluation images")
    args = parser.parse_args(argv)
    if args.first and args.stage not in WITH_FIRST:
        parser.error("--first applies to the pass and report stages only")
    try:
        config = load_config(args.config, run=args.run)
        STAGES[args.stage](config, **({"first": args.first} if args.stage in WITH_FIRST else {}))
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
