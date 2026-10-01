"""Pilot tables: the conv-TU fingerprint and its three controls next to the nine baselines, on the pilot images.

The baselines are restricted to the same images and use the full run's stored LRP threshold and per-fold
ContrastiveConf lambda, so their rows differ from docs/coco-baseline-numbers.md only through the image subset.
"""
from __future__ import annotations

import json

import numpy as np

from ..baselines import metrics, protocol, report
from ..baselines.coco_quality import CocoGroundTruth, coco_map, coco_results, image_lrp
from .features import REPRESENTATIONS

METHODS = report.METHODS + tuple(f"convtu_{rep}" for rep in REPRESENTATIONS)
LABELS = {**report.LABELS,
          "convtu_mst": "Conv TU: top 1% of the diagram",
          "convtu_edges": "Control: top 1% heaviest edges",
          "convtu_acts": "Control: largest activations",
          "convtu_means": "Control: channel means"}
BOOTSTRAP_SAMPLES = 1000


def depth_rows(layer_scores: dict, lrp: np.ndarray) -> list[dict]:
    """Per representation and stage: separation and harm tracking of that one layer's score."""
    delta_risk = lrp[:, report.CORRUPTED] - lrp[:, [0]]
    rows = []
    for name, values in layer_scores.items():
        for layer in range(values.shape[2]):
            v = values[:, :, layer]
            rows.append({"representation": name, "stage": layer + 1,
                         "auroc_common": float(metrics.condition_aurocs(v[:, 0], v[:, report.COMMON].T).mean()),
                         "auroc_extra": float(metrics.condition_aurocs(v[:, 0], v[:, report.EXTRA].T).mean()),
                         "rho_within": metrics.mean_within_condition_spearman(
                             v[:, report.CORRUPTED] - v[:, [0]], delta_risk)})
    return rows


def _depth_markdown(rows: list[dict]) -> str:
    lines = ["## Depth: one layer at a time", "",
             "| Representation | Stage | AUROC common | AUROC extra | ρ(Δscore, ΔLRP) within |",
             "| --- | ---: | ---: | ---: | ---: |"]
    lines += [f"| {r['representation']} | {r['stage']} | {report._fmt(r['auroc_common'])} | "
              f"{report._fmt(r['auroc_extra'])} | {report._fmt(r['rho_within'])} |" for r in rows]
    return "\n".join(lines) + "\n"


def build_pilot_report(settings, names, extra=None, extra_layers=None, extra_labels=None,
                       folder_name="results_convtu", title="# Conv TU pilot") -> None:
    names = list(names)
    folds = protocol.assign_folds(len(names))
    baseline = json.loads((settings.output / "results" / "summary.json").read_text())
    test = report._stack(settings.output / "test", names, report.TEST_ARRAYS)
    dcp_folder = settings.output / "test_dcp"
    dcp = report._stack(dcp_folder, names, ("dcp",))["dcp"] if dcp_folder.exists() else None
    activation_folder = settings.output / "test_activation"
    activation = (report._stack(activation_folder, names, report.ACTIVATION_ARRAYS)
                  if activation_folder.exists() else None)
    conv = report._stack(settings.output / "test_convtu", names,
                         (*REPRESENTATIONS, *(f"{rep}_layers" for rep in REPRESENTATIONS)))
    per_image_lambda = np.array([baseline["lambda_per_fold"][str(f)] for f in folds])
    per_fold_methods = () if baseline["lambda_folds_agree"] else ("contrastive",)
    ordered = report.method_scores(test, dcp, per_image_lambda, activation=activation)
    ordered.update({f"convtu_{rep}": conv[rep] for rep in REPRESENTATIONS})
    ordered.update(extra or {})
    methods = METHODS + tuple(extra or {})
    labels = {**LABELS, **(extra_labels or {})}
    scores = {m: ordered[m] for m in methods if m in ordered}

    gt = CocoGroundTruth(settings.annotations)
    ids = [gt.image_id(n) for n in names]
    threshold = baseline["lrp_threshold"]
    lrp = np.full((len(names), len(protocol.CONDITIONS)), np.nan)
    for k, image_id in enumerate(ids):
        gt_boxes, gt_labels, crowd = gt.boxes(image_id)
        for c in range(len(protocol.CONDITIONS)):
            lrp[k, c] = image_lrp(test["det_scores"][k, c], test["det_labels"][k, c], test["det_boxes"][k, c],
                                  gt_boxes, gt_labels, crowd, threshold)
    condition_map = np.array([
        coco_map(gt, [r for k, i in enumerate(ids) for r in coco_results(
            i, test["det_scores"][k, c], test["det_labels"][k, c], test["det_boxes"][k, c], gt.category_ids)], ids)
        for c in range(len(protocol.CONDITIONS))])

    separation = report.separation_rows(scores, folds, per_fold_methods=per_fold_methods)
    harm, pools = report.harm_rows(scores, lrp, condition_map)
    point = report.headline_numbers(scores, lrp, folds, per_fold_methods)

    def statistic(draw):
        return report.headline_numbers({m: v[draw] for m, v in scores.items()}, lrp[draw], folds[draw],
                                       per_fold_methods)

    ranges = metrics.bootstrap(statistic, len(names), samples=BOOTSTRAP_SAMPLES, seed=settings.seed)
    interval_rows = [{"quantity": key, "point": point[key], "low": ranges[key][0], "high": ranges[key][1]}
                     for key in point]
    layer_scores = {rep: conv[f"{rep}_layers"] for rep in REPRESENTATIONS}
    layer_scores.update(extra_layers or {})
    depth = depth_rows(layer_scores, lrp)
    calibration = json.loads((settings.output / "convtu" / "calibration.json").read_text())
    summary = {"images": len(names), "chosen_as": "first images of the seed-44 evaluation order",
               "lrp_threshold": threshold, "lambda_per_fold": baseline["lambda_per_fold"],
               "contrastive_lambda": "stored per-fold values of the full run, not refitted per bootstrap draw",
               "bootstrap_samples": BOOTSTRAP_SAMPLES, "clean_map_of_these_images": float(condition_map[0]),
               "fraction": calibration["fraction"], "cut_margin": calibration["cut_margin"],
               "calibration": calibration["layers"]}
    folder = settings.output / folder_name
    report.write_outputs(folder, {
        "separation": separation, "aggregates": report.aggregate_rows(separation), "harm": harm,
        "aurc_pools": pools, "intervals": [r for r in interval_rows if " - " not in r["quantity"]],
        "differences": [r for r in interval_rows if " - " in r["quantity"]], "depth": depth,
    }, summary, methods=methods, labels=labels, title=title)
    with (folder / "report.md").open("a") as handle:
        handle.write("\n" + _depth_markdown(depth))
