"""The report stage: read every stored score and write the report to reports/<benchmark>/, or to another folder (out)."""
from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import numpy as np

from .. import corruptions
from ..datasets.coco import coco_map, coco_results, per_image_ap
from ..evaluation import report as tables
from ..method.statistics import KEYS
from ..runs import sha1, stack

DETECTOR_ARRAYS = ("saod_min", "saod_top3", "knn", "det_scores", "det_labels", "det_boxes")
CONTRASTIVE_ARRAYS = ("conf_pos", "conf_neg")  # DETR-type detectors only
ACTIVATION_ARRAYS = ("cdf_backbone_z", "cdf_backbone")
HASHEMI_ARRAYS = ("hashemi_decoder", "hashemi_encoder")  # RT-DETR both, RF-DETR the decoder, CNN detectors neither
_MAP_INPUTS = None  # what the forked mAP workers read


def _results(gt, ids, detector, condition) -> list:
    """COCO result records of every image under one condition."""
    return [r for k, i in enumerate(ids)
            for r in coco_results(i, detector["det_scores"][k, condition], detector["det_labels"][k, condition],
                                  detector["det_boxes"][k, condition], gt.category_ids)]


def _condition_map(condition: int) -> float:
    gt, ids, detector = _MAP_INPUTS
    return coco_map(gt, _results(gt, ids, detector, condition), ids)


def condition_maps(gt, ids, detector, workers: int) -> np.ndarray:
    """Each condition's mAP over the evaluation images, in forked worker processes when workers > 1."""
    global _MAP_INPUTS
    _MAP_INPUTS = (gt, ids, detector)
    try:
        conditions = range(len(corruptions.CONDITIONS))
        if workers > 1:
            with multiprocessing.get_context("fork").Pool(workers) as pool:
                return np.array(pool.map(_condition_map, conditions))
        return np.array([_condition_map(c) for c in conditions])
    finally:
        _MAP_INPUTS = None


def _method_reference(layout):
    """The sha1 of the method's reference, or None when the method pass never ran; refuses a pass without it."""
    if not layout.scores("method").exists():
        return None
    missing = [path.name for path in (layout.method_bank, layout.method_zstats) if not path.exists()]
    if missing:
        raise ValueError(f"run the method-reference stage first: {', '.join(missing)} missing")
    return {"method_bank": sha1(layout.method_bank), "method_zstats": sha1(layout.method_zstats)}


def _method_rows(layout, names) -> dict:
    """Our six rows, from the stored statistics and the reference's."""
    statistics = stack(layout.scores("method"), names, KEYS)
    with np.load(layout.method_bank) as bank, np.load(layout.method_zstats) as zstats:
        bank, zstats = {k: bank[k] for k in KEYS}, {k: zstats[k] for k in KEYS}
    return tables.method_rows(statistics, bank, zstats)


def _stack_present(folder, names, required, optional) -> dict:
    """The required arrays, and those optional ones the score files hold: one pass writes every file alike."""
    first, present = folder / f"{Path(names[0]).stem}.npz", ()
    if first.exists():
        with np.load(first) as data:
            present = tuple(key for key in optional if key in data.files)
    return stack(folder, names, required + present)


def write_report(settings, manifest, out=None, extra_rows=None, extra_inputs=None) -> None:
    """Every table of the report, from the stored scores of every pass that ran.

    extra_rows adds rows computed elsewhere, the image-quality baselines (EXTRA_ROWS): (images, 96) arrays in
    evaluation order. out writes the report to another folder, so that a read-only run can get one. extra_inputs is
    merged into the summary's inputs.
    """
    layout = settings.layout
    reference = _method_reference(layout)  # checked first: the mAP below takes minutes on all 5,000 images
    dataset = settings.dataset
    names = [p.name for p in dataset.evaluation_images()]
    detector = _stack_present(layout.scores("detector"), names, DETECTOR_ARRAYS, CONTRASTIVE_ARRAYS)
    discopatch = stack(layout.scores("discopatch"), names, ("dcp",))["dcp"] if layout.scores("discopatch").exists() \
        else None
    activations = _stack_present(layout.scores("activations"), names, ACTIVATION_ARRAYS, HASHEMI_ARRAYS) \
        if layout.scores("activations").exists() else None
    scores = tables.baseline_rows(detector, discopatch=discopatch, activations=activations)
    if extra_rows:
        refused = sorted(set(extra_rows) - set(tables.EXTRA_ROWS))
        if refused:
            raise ValueError(f"rows a report cannot take from elsewhere: {', '.join(refused)}")
        shape = (len(names), len(corruptions.CONDITIONS))
        wrong = sorted(k for k, v in extra_rows.items() if np.shape(v) != shape)
        if wrong:
            raise ValueError(f"extra rows need {shape[0]} x {shape[1]} values, one per image and condition: "
                             f"{', '.join(wrong)}")
        scores.update(extra_rows)
    gt = dataset.ground_truth()
    ids = [gt.image_id(n) for n in names]
    condition_map = condition_maps(gt, ids, detector, settings.workers)
    clean = {i: coco_results(i, detector["det_scores"][k, 0], detector["det_labels"][k, 0], detector["det_boxes"][k, 0],
                             gt.category_ids) for k, i in enumerate(ids)}
    ap = per_image_ap(gt, clean, ids)
    if reference:
        scores.update(_method_rows(layout, names))
    timing = json.loads(layout.timing.read_text()) if layout.timing.exists() else {}
    sets = None if dataset.screened else {"all": np.arange(len(names))}
    report_tables, summary = tables.build_tables(scores, detector, ap, dataset.folds(), condition_map,
                                                 seed=settings.seed, samples=tables.BOOTSTRAP_SAMPLES,
                                                 workers=settings.workers, timing=timing, sets=sets,
                                                 benchmark=dataset.name)
    summary["inputs"] = {**manifest.read().get("inputs", {}), **({"method": reference} if reference else {}),
                         **(extra_inputs or {})}
    tables.write_outputs(out or layout.report(dataset.name), report_tables, summary)
    if "headline_decision" in summary:
        print(f"[report] headline: {summary['headline_decision']}; level score: {summary['level_decision']}",
              flush=True)
    else:
        print(f"[report] {dataset.name}: {summary['images']} images, one image set, no decision rule", flush=True)
