"""Four detector-free image-quality baselines on COCO-C: their clean references, one pass over the corrupted val
images, their timing, and every detector's report with their five rows.

    python -m degradation_monitor.stages.iqa <stage> [--config configs/coco-iqa.toml] [--first N]

Stages:
- fit: from every clean train image, NIQE's pristine model, refitted (estimatemodelparam.m), and ARNIQA's clean
  prototype, the mean embedding;
- pass: every val image's 96 versions, checked against the reference run's digests, through the four models: one
  file per image with the five rows and NIQE's block counts; --first N stops after N images;
- timing: ms per image of each model at batch 1, on the first evaluation images, as the timing stage measures the
  detector;
- report: every detector's report with the five rows added, written under <run>/reports/<detector>/ (the detectors'
  run folders, the read-only reference run included, are only read), then the table of the two-axis score against
  each row in summary.md; --first N reports on the first N evaluation images only.

The models read the image, not the detector, so one set of scores serves every detector. The dataset and its train
and evaluation images come from the base config, so a Cityscapes config can reuse every stage.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from .. import corruptions
from ..baselines.iqa import NIQE_BLOCK, NIQE_SHARPNESS, ROWS, NiqeFit, load_iqa_models, weight_files
from ..cli import positive_int
from ..corruptions import CONDITIONS
from ..datasets.coco import FOLDS
from ..evaluation.report import LABELS
from ..runs import Manifest, RunLayout, atomic_json, atomic_npz, load_npz, progress, sha1, sha256, stack
from ..settings import load_settings
from .baselines import TIMING_IMAGES, TIMING_WARMUP
from .common import cap_gpu_memory, median_ms, open_rgb, report_peak_memory, rgb_batches, variant_stream
from .detectors import load_config as load_detectors_config
from .report import write_report

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "coco-iqa.toml"
CONFIG_KEYS = {"base", "run", "reference_run", "detectors_config", "gpu_memory_gib"}
REFERENCE_NAME = "rtdetrv2_r18"  # the reference run is RT-DETR's
PROGRESS_IMAGES = 2000
SHORT_LABELS = {"niqe": "NIQE (refit)", "niqe_default": "NIQE (published, sensitivity)", "arniqa": "ARNIQA quality",
                "arniqa_proto": "ARNIQA prototype", "clipiqa": "CLIP-IQA"}  # summary.md's row names


@dataclass(frozen=True)
class IqaConfig:
    base: object  # the base config's Settings: the dataset, the seed, the device and the workers
    run: Path
    reference_run: Path  # RT-DETR's run: the corruption digests, and one of the detectors
    detectors: dict  # detector name -> its Settings, whose run folder holds its scores
    gpu_memory_gib: float

    @property
    def layout(self) -> RunLayout:
        return RunLayout(self.run)


def load_config(path, run=None) -> IqaConfig:
    path = Path(path)
    values = tomllib.loads(path.read_text())
    unknown = sorted(set(values) - CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unknown settings in {path}: {', '.join(unknown)}")
    base = load_settings(path.parent / values["base"])
    reference_run = Path(values["reference_run"])
    if base.run.resolve() != reference_run.resolve():
        raise ValueError(f"{path}: the base config's run must be the reference run")
    others = load_detectors_config(path.parent / values["detectors_config"])
    detectors = {REFERENCE_NAME: base, **{name: others.settings(name) for name in others.detectors}}
    root = Path(run or values["run"])
    for folder in (others.run, *(settings.run for settings in detectors.values())):
        if root.resolve().is_relative_to(folder.resolve()) or folder.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"the run root {root} must lie outside the detectors' run folders and hold none of them, "
                             f"but {folder} is one: choose another run root (--run DIR)")
    return IqaConfig(base=base, run=root, reference_run=reference_run, detectors=detectors,
                     gpu_memory_gib=float(values["gpu_memory_gib"]))


def _manifest(config, models) -> Manifest:
    """The run's manifest, once its protocol is checked: the evaluation, the models' choices and their weights."""
    missing = [name for name, path in weight_files().items() if not Path(path).exists()]
    if missing:
        raise ValueError(f"the model files are missing ({', '.join(missing)}): load the models once (the fit stage)")
    base = config.base
    config.layout.root.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(config.layout)
    manifest.check_protocol({
        "dataset": base.benchmark, "seed": base.seed, "limit": base.limit, "folds": FOLDS,
        "conditions": [list(c) for c in CONDITIONS], "models": models.protocol,
        "weights": {name: sha256(path) for name, path in weight_files().items()},
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32})
    manifest.record_environment(base.discopatch_root)
    return manifest


def _paths_of_references(layout) -> tuple:
    return layout.reference("iqa", "niqe_refit.npz"), layout.reference("iqa", "arniqa_prototype.npy")


def _references(layout) -> tuple:
    refit, prototype = _paths_of_references(layout)
    if not (refit.exists() and prototype.exists()):
        raise ValueError("run the fit stage first: the NIQE refit or the ARNIQA prototype is missing")
    return refit, prototype


def fit(config) -> None:
    """NIQE's refitted pristine model and ARNIQA's clean prototype, from every clean train image."""
    layout = config.layout
    if all(path.exists() for path in (*_paths_of_references(layout), layout.reference("iqa", "fit.json"))):
        return
    base = config.base
    cap_gpu_memory(base.device, config.gpu_memory_gib)
    models = load_iqa_models(base.device)
    _manifest(config, models)
    paths = base.dataset.train_images()
    niqe, total, count, skipped = NiqeFit(), None, 0, 0
    size, started = models.fit_batch_size, time.time()
    for index, arrays in enumerate(rgb_batches(paths, size, base.workers)):
        features, sharpness, embedding = models.fit_features(arrays)
        if features is None:
            skipped += len(arrays)  # no whole NIQE block: left out of the NIQE refit only
        else:
            niqe.update(features, sharpness)
        summed = embedding.double().sum(dim=0)
        total = summed if total is None else total + summed
        count += len(arrays)
        if index % max(1, PROGRESS_IMAGES // size) == 0:
            progress("iqa fit", min((index + 1) * size, len(paths)), len(paths), started)
    mu, cov, blocks = niqe.result()
    refit, prototype = _paths_of_references(layout)
    atomic_npz(refit, mu=mu, cov=cov, images=np.array(niqe.images), blocks=np.array(blocks))
    temporary = prototype.with_name(".arniqa_prototype.tmp.npy")
    np.save(temporary, (total / count).cpu().numpy())
    temporary.replace(prototype)
    atomic_json(layout.reference("iqa", "fit.json"), {
        "images": count, "niqe_images": niqe.images, "niqe_skipped": skipped, "niqe_blocks": blocks,
        "niqe_dropped_blocks": niqe.dropped, "niqe_block": NIQE_BLOCK, "niqe_sharpness": NIQE_SHARPNESS,
        "prototype": "mean of the clean embeddings"})
    report_peak_memory("iqa fit", base.device)


def _loaded_models(config):
    refit, prototype = _references(config.layout)
    with np.load(refit) as data:
        niqe_refit = (data["mu"], data["cov"])
    return load_iqa_models(config.base.device, niqe_refit=niqe_refit, prototype=np.load(prototype)), refit, prototype


def iqa_pass(config, first=None) -> None:
    """Every evaluation image's 96 versions through the four models; resumable image by image."""
    layout, base = config.layout, config.base
    refit, prototype = _references(layout)
    pending = [p for p in base.dataset.evaluation_images() if not layout.score_file("iqa", p).exists()]
    pending = pending[:first] if first else pending
    if not pending:
        return  # before the models load: the card is shared
    cap_gpu_memory(base.device, config.gpu_memory_gib)
    models, _, _ = _loaded_models(config)
    manifest = _manifest(config, models)
    manifest.check_inputs("iqa", {"niqe_refit": sha1(refit), "arniqa_prototype": sha1(prototype)})
    reference, started = RunLayout(config.reference_run), time.time()
    for done, (image, arrays) in enumerate(variant_stream(base, pending), start=1):
        digests = np.array([corruptions.digest(a) for a in arrays])
        if list(load_npz(reference.score_file("detector", image), ("digests",))["digests"]) != list(digests):
            raise ValueError(f"corruptions differ from the reference run for {image}")
        parts = [models.scores(arrays[start:start + models.batch_size])
                 for start in range(0, len(arrays), models.batch_size)]
        rows = {key: np.concatenate([part[key] for part in parts]) for key in (*ROWS, "niqe_blocks")}
        if not all(np.isfinite(rows[row]).all() for row in ROWS):
            raise ValueError(f"an image-quality score is not finite for {image}")
        atomic_npz(layout.score_file("iqa", image), digests=digests, **rows)
        if done % 25 == 0:
            progress("iqa pass", done, len(pending), started)
    report_peak_memory("iqa pass", base.device)


def timing(config) -> None:
    """Median ms per image of each model at batch 1, preprocessing included, as the timing stage measures."""
    base = config.base
    cap_gpu_memory(base.device, config.gpu_memory_gib)
    models, _, _ = _loaded_models(config)
    _manifest(config, models)
    images = [open_rgb(p) for p in base.dataset.evaluation_images()[:TIMING_IMAGES]]

    def timed(part):
        return median_ms(lambda array: part([array]), images, TIMING_WARMUP, base.device)

    detector = RunLayout(config.reference_run).timing
    result = {"images": len(images), "batch_size": 1, "device": str(base.device),
              "niqe_ms": timed(models.niqe_rows), "arniqa_ms": timed(models.arniqa_rows),
              "clipiqa_ms": timed(models.clipiqa_rows),
              "detector_ms": json.loads(detector.read_text()).get("detector_ms") if detector.exists() else None}
    atomic_json(config.layout.timing, result)
    print(f"[iqa timing] {result}", flush=True)
    report_peak_memory("iqa timing", base.device)


def _cell(value) -> str:
    return "–" if value is None else f"{value:+.3f}"


def iqa_table(config) -> list:
    """For every detector and row: the row's AUROC and the two-axis score minus it, on all images and, for a benchmark
    with a screening history, on the untouched ones."""
    subsets = ("all", "untouched") if config.base.dataset.screened else ("all",)
    rows = []
    for name in config.detectors:
        summary = json.loads((config.layout.report(name) / "summary.json").read_text())
        headline, intervals = summary["headline"], summary["intervals"]
        for row in ROWS:
            entry = {"detector": name, "row": row, "label": LABELS[row]}
            for subset in subsets:
                for group in ("common", "extra"):
                    entry[f"{subset}_auroc_{group}"] = headline.get(subset, {}).get(row, {}).get(f"auroc_{group}")
            for subset in subsets:
                for group in ("common", "extra"):
                    cell = intervals.get(subset, {}).get(f"two_axis - {row}:auroc_{group}")
                    key = f"{subset}_two_axis_minus_{group}"
                    entry[key] = None if cell is None else cell["point"]
                    entry[f"{key}_low"] = None if cell is None else cell["low"]
                    entry[f"{key}_high"] = None if cell is None else cell["high"]
            rows.append(entry)
    config.run.mkdir(parents=True, exist_ok=True)
    temporary = config.run / ".summary.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(config.run / "summary.csv")
    where = "on all images and on the untouched ones" if len(subsets) == 2 else "on all images"
    if len(subsets) == 2:
        header = ("| Detector | Row | AUROC all: common / extra | untouched: common / extra | Two-axis − row, all: "
                  "common | extra | untouched: common | extra |")
    else:
        header = "| Detector | Row | AUROC all: common / extra | Two-axis − row, all: common | extra |"
    lines = ["# The two-axis score against four image-quality baselines", "",
             f"Each row's AUROC, and the two-axis score minus it with 95% paired bootstrap intervals, {where}.", "",
             header, "|" + "---|" * (2 + 3 * len(subsets))]
    for r in rows:
        aurocs = [" / ".join("–" if r[f"{s}_auroc_{g}"] is None else f"{r[f'{s}_auroc_{g}']:.3f}"
                             for g in ("common", "extra")) for s in subsets]
        cells = [f"{_cell(r[k])} [{_cell(r[k + '_low'])}, {_cell(r[k + '_high'])}]"
                 for k in (f"{s}_two_axis_minus_{g}" for s in subsets for g in ("common", "extra"))]
        lines.append(f"| {r['detector']} | {SHORT_LABELS[r['row']]} | " + " | ".join(aurocs + cells) + " |")
    temporary = config.run / ".summary.md.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(config.run / "summary.md")
    return rows


def report(config, first=None) -> None:
    """Every detector's report with the five rows, under <run>/reports/<detector>/, then the table."""
    layout = config.layout
    recorded = Manifest(layout).read()
    if "protocol" not in recorded:
        raise ValueError("run the pass stage first: this run folder records no protocol")
    settings_first = replace(config.base, limit=first) if first else config.base
    names = [p.name for p in settings_first.dataset.evaluation_images()]
    rows = stack(layout.scores("iqa"), names, ROWS)
    inputs = {"iqa": {**recorded.get("inputs", {}).get("iqa", {}), "models": recorded["protocol"]["models"],
                      "weights": recorded["protocol"]["weights"], "run": str(config.run)}}
    chosen = {name: replace(settings, limit=first) if first else settings
              for name, settings in config.detectors.items()}
    for name, settings in chosen.items():  # all checked before any report is written
        if [p.name for p in settings.dataset.evaluation_images()] != names:
            raise ValueError(f"the evaluation images of {name} differ from the image-quality pass's")
    for name, settings in chosen.items():
        write_report(settings, Manifest(settings.layout), out=layout.report(name), extra_rows=rows,
                     extra_inputs=inputs)
    iqa_table(config)


STAGES = {"fit": fit, "pass": iqa_pass, "timing": timing, "report": report}
WITH_FIRST = ("pass", "report")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m degradation_monitor.stages.iqa",
                                     description="Four detector-free image-quality baselines on COCO-C: run one stage.")
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
