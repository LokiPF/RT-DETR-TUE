"""Turn stored baseline scores into separation, harm, interval and runtime tables."""
from __future__ import annotations

import csv
import json
import math
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np

from .activation_cdf import BINS as CDF_BINS
from .hashemi import K as HASHEMI_K
from . import metrics, protocol
from .coco_quality import (CocoGroundTruth, coco_map, coco_results, image_lrp, per_image_ap,
                           select_lrp_threshold)

METHODS = ("saod_top3", "saod_min", "contrastive", "knn", "discopatch", "hashemi", "hashemi_enc", "cdf", "cdf_sum")
LABELS = {"saod_top3": "SAOD, mean of top 3", "saod_min": "SAOD, min (1 − max confidence)",
          "contrastive": "ContrastiveConf", "knn": "kNN (k = 100)", "discopatch": "DisCoPatch",
          "hashemi": "Hashemi et al., decoder queries",
          "hashemi_enc": "Hashemi et al., encoder maps (sensitivity)",
          "cdf": "Activation CDFs (Becker et al., ICPR 2026)",
          "cdf_sum": "Activation CDFs, plain channel sum (sensitivity)"}
ACTIVATION_ARRAYS = ("hashemi_decoder", "hashemi_encoder", "cdf_backbone_z", "cdf_backbone")
KNN_K = 100
KNN_KS = (1, 10, 50, 100, 200)
SEPARATION = ("auroc", "aupr", "fpr95")
BOOTSTRAP_SAMPLES = 1000
DIFFERENCE_METRICS = ("auroc_common", "auroc_extra", "aupr_common", "aupr_extra", "fpr95_common",
                      "fpr95_extra", "rho_within", "rho_condition_lrp", "aurc_all")
SEVERITY = np.array([s for _, s in protocol.CONDITIONS])
CORRUPTED = np.arange(1, len(protocol.CONDITIONS))
COMMON = np.array([c for c, (f, _) in enumerate(protocol.CONDITIONS) if f in protocol.COMMON_FAMILIES])
EXTRA = np.array([c for c, (f, _) in enumerate(protocol.CONDITIONS) if f in protocol.EXTRA_FAMILIES])
TEST_ARRAYS = ("saod_min", "saod_top3", "conf_pos", "conf_neg", "knn", "det_scores", "det_labels", "det_boxes")


def method_scores(test: dict, dcp, per_image_lambda, k: int = KNN_K, activation=None) -> dict:
    scores = {"saod_top3": test["saod_top3"], "saod_min": test["saod_min"],
              "contrastive": -(test["conf_pos"] - np.asarray(per_image_lambda)[:, None] * test["conf_neg"]),
              "knn": test["knn"][:, :, k - 1]}
    if dcp is not None:
        scores["discopatch"] = dcp
    if activation is not None:
        scores["hashemi"] = activation["hashemi_decoder"]
        scores["hashemi_enc"] = activation["hashemi_encoder"]
        scores["cdf"] = activation["cdf_backbone_z"]
        scores["cdf_sum"] = activation["cdf_backbone"]
    return scores


def _separation(clean, degraded) -> dict:
    return {"auroc": metrics.auroc(clean, degraded), "aupr": metrics.aupr(clean, degraded),
            "fpr95": metrics.fpr_at_95_tpr(clean, degraded)}


def separation_rows(scores: dict, folds, per_fold_methods=()) -> list[dict]:
    folds = np.asarray(folds)
    rows = []
    for method, values in scores.items():
        by_fold = method in per_fold_methods
        for c in CORRUPTED:
            family, severity = protocol.CONDITIONS[c]
            if by_fold:
                parts = [_separation(values[folds == f, 0], values[folds == f, c]) for f in np.unique(folds)]
                result = {key: float(np.mean([p[key] for p in parts])) for key in SEPARATION}
            else:
                result = _separation(values[:, 0], values[:, c])
            rows.append({"method": method, "family": family, "severity": int(severity),
                         "group": "common" if family in protocol.COMMON_FAMILIES else "extra",
                         "pooling": "fold-averaged" if by_fold else "pooled", **result})
    return rows


def aggregate_rows(rows: list[dict]) -> list[dict]:
    out = []
    for method in dict.fromkeys(r["method"] for r in rows):
        for group in ("common", "extra", "all"):
            for severity in (*protocol.SEVERITIES, "all"):
                chosen = [r for r in rows if r["method"] == method
                          and (group == "all" or r["group"] == group)
                          and (severity == "all" or r["severity"] == severity)]
                out.append({"method": method, "group": group, "severity": severity,
                            **{key: float(np.mean([r[key] for r in chosen])) for key in SEPARATION}})
    return out


def _pools():
    yield "all", np.arange(len(protocol.CONDITIONS))
    for s in protocol.SEVERITIES:
        yield f"severity {s}", np.r_[0, CORRUPTED[SEVERITY[CORRUPTED] == s]]
    for family in protocol.FAMILIES:
        yield family, np.r_[0, [c for c in CORRUPTED if protocol.CONDITIONS[c][0] == family]]


def _nanmean_columns(values):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=0)


def harm_rows(scores: dict, lrp: np.ndarray, condition_map: np.ndarray):
    mean_lrp = _nanmean_columns(lrp)
    delta_risk = lrp[:, CORRUPTED] - lrp[:, [0]]
    harm, pools = [], []
    for method, values in scores.items():
        means = values.mean(axis=0)
        delta_score = values[:, CORRUPTED] - values[:, [0]]
        row = {"method": method,
               "rho_condition_map": metrics.spearman(means[CORRUPTED], condition_map[CORRUPTED]),
               "rho_condition_lrp": metrics.spearman(means[CORRUPTED], mean_lrp[CORRUPTED]),
               "rho_within": metrics.mean_within_condition_spearman(delta_score, delta_risk)}
        for s in protocol.SEVERITIES:
            columns = SEVERITY[CORRUPTED] == s
            row[f"rho_within_sev{s}"] = metrics.mean_within_condition_spearman(
                delta_score[:, columns], delta_risk[:, columns])
        harm.append(row)
        for pool, columns in _pools():
            pooled_scores, pooled_risk = values[:, columns].ravel(), lrp[:, columns].ravel()
            pools.append({"method": method, "pool": pool, "aurc": metrics.aurc(pooled_scores, pooled_risk),
                          "aurc_oracle": metrics.aurc(pooled_risk, pooled_risk)})
    return harm, pools


def _group_separation(clean, degraded):
    """Mean AUROC, AUPR and FPR95 over the conditions (rows) of `degraded`."""
    return (float(metrics.condition_aurocs(clean, degraded).mean()),
            float(np.mean([metrics.aupr(clean, row) for row in degraded])),
            float(np.mean([metrics.fpr_at_95_tpr(clean, row) for row in degraded])))


def headline_numbers(scores: dict, lrp: np.ndarray, folds=None, per_fold_methods=()) -> dict:
    """Aggregates used for intervals, computed identically on the full set and on each draw.

    Methods in `per_fold_methods` get fold-averaged separation numbers, matching separation_rows.
    """
    out = {}
    delta_risk = lrp[:, CORRUPTED] - lrp[:, [0]]
    mean_lrp = _nanmean_columns(lrp)
    for method, values in scores.items():
        for group, columns in (("common", COMMON), ("extra", EXTRA)):
            if method in per_fold_methods:
                parts = [_group_separation(values[folds == f, 0], values[folds == f][:, columns].T)
                         for f in np.unique(folds)]
                auroc_value, aupr_value, fpr_value = np.mean(parts, axis=0)
            else:
                auroc_value, aupr_value, fpr_value = _group_separation(values[:, 0], values[:, columns].T)
            out[f"{method}:auroc_{group}"] = float(auroc_value)
            out[f"{method}:aupr_{group}"] = float(aupr_value)
            out[f"{method}:fpr95_{group}"] = float(fpr_value)
        out[f"{method}:rho_within"] = metrics.mean_within_condition_spearman(
            values[:, CORRUPTED] - values[:, [0]], delta_risk)
        out[f"{method}:rho_condition_lrp"] = metrics.spearman(values.mean(axis=0)[CORRUPTED], mean_lrp[CORRUPTED])
        out[f"{method}:aurc_all"] = metrics.aurc(values.ravel(), lrp.ravel())
    for a, b in combinations(scores, 2):
        for metric in DIFFERENCE_METRICS:
            out[f"{a} - {b}:{metric}"] = out[f"{a}:{metric}"] - out[f"{b}:{metric}"]
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value) -> str:
    return "–" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.3f}"


def _with_ci(intervals: dict, key: str) -> str:
    row = intervals.get(key)
    return "–" if row is None else f"{_fmt(row['point'])} [{_fmt(row['low'])}, {_fmt(row['high'])}]"


def _markdown(tables: dict, summary: dict, methods=METHODS, labels=LABELS, title="# COCO baseline numbers") -> str:
    intervals = {r["quantity"]: r for r in tables.get("intervals", []) + tables.get("differences", [])}
    aggregates = tables.get("aggregates", [])
    present = [m for m in methods if any(r["method"] == m for r in aggregates)]
    lines = [title, "",
             "All scores are oriented so that higher means more likely degraded. AUROC and AUPR: higher is "
             "better (chance 0.5). FPR95: lower is better. Brackets are 95% paired bootstrap intervals over "
             "images.", ""]
    for group, title in (("common", "15 common families"), ("extra", "4 extra families")):
        lines += [f"## Separation, {title}", "",
                  "| Method | AUROC sev 1 | sev 2 | sev 3 | sev 4 | sev 5 | AUROC all | AUPR all | FPR95 all |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for method in present:
            cells = {r["severity"]: r for r in aggregates if r["method"] == method and r["group"] == group}
            by_severity = " | ".join(_fmt(cells[s]["auroc"]) if s in cells else "–" for s in protocol.SEVERITIES)
            lines.append(f"| {labels[method]} | {by_severity} | "
                         f"{_with_ci(intervals, f'{method}:auroc_{group}') if intervals else _fmt(cells['all']['auroc'])} | "
                         f"{_with_ci(intervals, f'{method}:aupr_{group}') if intervals else _fmt(cells['all']['aupr'])} | "
                         f"{_with_ci(intervals, f'{method}:fpr95_{group}') if intervals else _fmt(cells['all']['fpr95'])} |")
        lines.append("")
    harm = tables.get("harm", [])
    pools = {(r["method"], r["pool"]): r for r in tables.get("aurc_pools", [])}
    if harm:
        lines += ["## Harm alignment", "",
                  "| Method | ρ(score, mAP), conditions | ρ(score, LRP), conditions | ρ(Δscore, ΔLRP), within condition | AURC all (oracle) |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for row in harm:
            m = row["method"]
            pool = pools.get((m, "all"), {})
            lines.append(f"| {labels[m]} | {_fmt(row['rho_condition_map'])} | "
                         f"{_with_ci(intervals, f'{m}:rho_condition_lrp')} | {_with_ci(intervals, f'{m}:rho_within')} | "
                         f"{_with_ci(intervals, f'{m}:aurc_all')} ({_fmt(pool.get('aurc_oracle'))}) |")
        lines.append("")
    differences = tables.get("differences", [])
    if differences:
        lines += ["## Differences between methods", "",
                  "| Pair | Δ AUROC common | Δ ρ within condition | Δ AURC all |", "| --- | ---: | ---: | ---: |"]
        pairs = dict.fromkeys(r["quantity"].split(":")[0] for r in differences)
        for pair in pairs:
            lines.append(f"| {pair} | {_with_ci(intervals, f'{pair}:auroc_common')} | "
                         f"{_with_ci(intervals, f'{pair}:rho_within')} | {_with_ci(intervals, f'{pair}:aurc_all')} |")
        lines.append("")
    timing = tables.get("timing", [])
    if timing:
        lines += ["## Runtime (median ms per image, batch 1)", "", "| Part | ms |", "| --- | ---: |"]
        lines += [f"| {r['part']} | {_fmt(r['ms'])} |" for r in timing] + [""]
    lines += ["## Fixed choices", ""] + [f"- {key}: {value}" for key, value in sorted(summary.items())]
    return "\n".join(lines) + "\n"


def write_outputs(folder, tables: dict, summary: dict, methods=METHODS, labels=LABELS,
                  title="# COCO baseline numbers") -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        if rows:
            _write_csv(folder / f"{name}.csv", rows)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    (folder / "report.md").write_text(_markdown(tables, summary, methods, labels, title))


def _stack(folder: Path, names, keys) -> dict:
    missing = [n for n in names if not (folder / f"{Path(n).stem}.npz").exists()]
    if missing:
        raise ValueError(f"{folder.name} is incomplete: {len(missing)} of {len(names)} missing, e.g. {missing[0]}")
    columns = {key: [] for key in keys}
    for name in names:
        with np.load(folder / f"{Path(name).stem}.npz") as item:
            for key in keys:
                columns[key].append(item[key])
    return {key: np.stack(values) for key, values in columns.items()}


def build_report(settings) -> None:
    from .pipeline import evaluation
    names = [p.name for p in evaluation(settings)]
    folds = protocol.assign_folds(len(names))
    test = _stack(settings.output / "test", names, TEST_ARRAYS)
    dcp_folder = settings.output / "test_dcp"
    dcp = _stack(dcp_folder, names, ("dcp",))["dcp"] if dcp_folder.exists() else None
    activation_folder = settings.output / "test_activation"
    activation = _stack(activation_folder, names, ACTIVATION_ARRAYS) if activation_folder.exists() else None

    gt = CocoGroundTruth(settings.annotations)
    ids = [gt.image_id(n) for n in names]
    clean_records = [(i, test["det_scores"][k, 0], test["det_labels"][k, 0], test["det_boxes"][k, 0])
                     for k, i in enumerate(ids)]
    ap = per_image_ap(gt, {i: coco_results(i, s, l, b, gt.category_ids) for i, s, l, b in clean_records}, ids)
    clean_pos, clean_neg = test["conf_pos"][:, 0], test["conf_neg"][:, 0]
    per_image_lambda, per_fold = metrics.cross_fit_lambda(clean_pos, clean_neg, ap, folds)
    consistent = len(set(per_fold.values())) == 1
    ordered = {m: v for m, v in method_scores(test, dcp, per_image_lambda, activation=activation).items()}
    scores = {m: ordered[m] for m in METHODS if m in ordered}

    threshold = select_lrp_threshold(clean_records, gt)
    lrp = np.full((len(names), len(protocol.CONDITIONS)), np.nan)
    for k, image_id in enumerate(ids):
        gt_boxes, gt_labels, crowd = gt.boxes(image_id)
        for c in range(len(protocol.CONDITIONS)):
            lrp[k, c] = image_lrp(test["det_scores"][k, c], test["det_labels"][k, c], test["det_boxes"][k, c],
                                  gt_boxes, gt_labels, crowd, threshold)
    condition_map = np.array([
        coco_map(gt, [r for k, i in enumerate(ids) for r in coco_results(
            i, test["det_scores"][k, c], test["det_labels"][k, c], test["det_boxes"][k, c], gt.category_ids)], ids)
        for c in range(len(protocol.CONDITIONS))
    ])

    per_fold_methods = () if consistent else ("contrastive",)
    separation = separation_rows(scores, folds, per_fold_methods=per_fold_methods)
    harm, pools = harm_rows(scores, lrp, condition_map)
    point = headline_numbers(scores, lrp, folds, per_fold_methods)

    def statistic(draw):
        # Duplicated images stay in their own fold, so a fold's lambda never sees its own images.
        lam, _ = metrics.cross_fit_lambda(clean_pos[draw], clean_neg[draw], ap[draw], folds[draw])
        drawn = {m: v[draw] for m, v in scores.items()}
        drawn["contrastive"] = -(test["conf_pos"][draw] - lam[:, None] * test["conf_neg"][draw])
        return headline_numbers({m: drawn[m] for m in scores}, lrp[draw], folds[draw], per_fold_methods)

    ranges = metrics.bootstrap(statistic, len(names), samples=BOOTSTRAP_SAMPLES, seed=settings.seed)
    interval_rows = [{"quantity": key, "point": point[key], "low": ranges[key][0], "high": ranges[key][1]}
                     for key in point]
    conditions = [{"family": f, "severity": s, "map": float(condition_map[c]),
                   "mean_lrp": float(_nanmean_columns(lrp)[c]), "images_undefined_lrp": int(np.isnan(lrp[:, c]).sum()),
                   **{f"mean_{m}": float(v[:, c].mean()) for m, v in scores.items()}}
                  for c, (f, s) in enumerate(protocol.CONDITIONS)]
    knn_k = [{"k": k, "mean_auroc_common": float(metrics.condition_aurocs(
        test["knn"][:, 0, k - 1], test["knn"][:, COMMON, k - 1].T).mean())} for k in KNN_KS]
    timing_path = settings.output / "timing.json"
    timing = []
    if timing_path.exists():
        timing = [{"part": key, "ms": value} for key, value in json.loads(timing_path.read_text()).items()
                  if key.endswith("_ms")]
    summary = {
        "images": len(names), "folds": protocol.FOLDS, "lambda_per_fold": per_fold,
        "lambda_folds_agree": consistent, "images_with_ap": int(np.isfinite(ap).sum()),
        "lrp_threshold": threshold, "images_with_undefined_clean_lrp": int(np.isnan(lrp[:, 0]).sum()),
        "clean_map": float(condition_map[0]), "knn_k": KNN_K, "theta": 0.3,
        "bootstrap_samples": BOOTSTRAP_SAMPLES, "discopatch_included": dcp is not None,
        "activation_monitors_included": activation is not None, "hashemi_k": HASHEMI_K, "cdf_bins": CDF_BINS,
    }
    write_outputs(settings.output / "results", {
        "separation": separation, "aggregates": aggregate_rows(separation), "harm": harm,
        "aurc_pools": pools, "conditions": conditions,
        "intervals": [r for r in interval_rows if " - " not in r["quantity"]],
        "differences": [r for r in interval_rows if " - " in r["quantity"]],
        "knn_k": knn_k, "timing": timing,
    }, summary)
