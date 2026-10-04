"""The report: separation of our method and the six baselines on four image sets.

Every score is oriented so that higher means more likely corrupted. For each image set the report gives:
- every row's AUROC, AUPR and FPR95 per condition;
- their means over the 15 common and the 4 extra families;
- 95% paired bootstrap intervals over images, ContrastiveConf's lambda being cross-fitted again in every draw.

It also gives the two pre-registered decisions, each condition's mAP and the kNN baseline's k.

The image sets, by position in the seed-44 evaluation order:
- all: every evaluation image (5,000), the headline, on the same images as every baseline;
- untouched: positions UNTOUCHED_START and later (3,030), which nobody read while the method was designed;
- held_out: positions SCREEN_IMAGES and later (4,800), the pre-registered check of the level score;
- screen: the first SCREEN_IMAGES (200), on which the method was screened. It gets no intervals.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from .. import corruptions
from ..baselines.activation_cdf import BINS as CDF_BINS
from ..baselines.contrastive_conf import THETA, contrastive_degradation, cross_fit_lambda
from ..baselines.hashemi import K as HASHEMI_K
from ..baselines.knn import KNN_K
from ..datasets.coco import FOLDS
from ..method import reference as method_reference
from ..method import scores as method_scores
from ..runs import atomic_json
from .metrics import aupr, auroc, bootstrap, condition_aurocs, fpr_at_95_tpr, group_separation

OURS = ("two_axis", "peak_share", "level", "global_level", "means_knn", "means_own")
BASELINES = ("saod_top3", "saod_min", "contrastive", "knn", "discopatch", "hashemi", "hashemi_enc", "cdf", "cdf_sum")
ROWS = OURS + BASELINES
LABELS = {
    "two_axis": "Two-axis: flatter or shifted vs the 50 most similar clean scenes (headline)",
    "peak_share": "Peak share vs the 50 most similar clean scenes",
    "level": "Level vs the 50 most similar clean scenes (stage-4 key)",
    "global_level": "Level vs the average of all clean images (stages 1–3)",
    "means_knn": "Channel means, kNN, 4 stages",
    "means_own": "Channel means vs own average, 4 stages",
    "saod_top3": "SAOD, mean of top 3",
    "saod_min": "SAOD, min (1 − max confidence)",
    "contrastive": "ContrastiveConf",
    "knn": "kNN (k = 100)",
    "discopatch": "DisCoPatch",
    "hashemi": "Hashemi et al., decoder queries",
    "hashemi_enc": "Hashemi et al., encoder maps (sensitivity)",
    "cdf": "Activation CDFs (Becker et al., ICPR 2026)",
    "cdf_sum": "Activation CDFs, plain channel sum (sensitivity)",
}
REFERENCES = ("two_axis", "level")  # rows whose differences with every other row get intervals
METRICS = ("auroc", "aupr", "fpr95")
GROUPS = ("common", "extra")
QUANTITIES = tuple(f"{metric}_{group}" for metric in METRICS for group in GROUPS)
BOOTSTRAP_SAMPLES = 1000
SCREEN_IMAGES = 200      # the first images of the evaluation order; the method was screened on them
UNTOUCHED_START = 1970   # the roundtable read positions 200-1969 of the running pass; nobody read these
KNN_KS = (1, 10, 50, 100, 200)
SECTION_REFERENCE = {"all": "two_axis", "untouched": "two_axis", "held_out": "level"}  # differences shown per set
FAMILY_ROWS = {"two_axis": "Two-axis", "peak_share": "Peak share", "level": "Level (similar scenes)",
               "cdf": "Activation CDFs", "discopatch": "DisCoPatch"}  # short column names for the family table


def image_sets(count: int) -> dict:
    """Positions of each image set in the evaluation order; a set without images (a limited run) is left out."""
    screen = min(SCREEN_IMAGES, count)
    sets = {"all": np.arange(count), "untouched": np.arange(min(UNTOUCHED_START, count), count),
            "held_out": np.arange(screen, count), "screen": np.arange(screen)}
    return {name: rows for name, rows in sets.items() if len(rows)}


def method_rows(statistics: dict, bank: dict, zstats: dict) -> dict:
    """Our six rows (images, 96) from the stored channel statistics and the clean reference's."""
    k = method_reference.NEIGHBOURS
    return {"two_axis": method_scores.two_axis_scores(statistics, bank, zstats, k=k)[0],
            "peak_share": method_scores.peak_share_scores(statistics, bank, zstats, k=k)[0],
            "level": method_scores.level_scores(statistics, bank, zstats, k=k)[0],
            "global_level": method_scores.global_level_scores(statistics, bank, zstats)[0],
            "means_knn": method_scores.means_knn_scores(statistics, bank, zstats)[0],
            "means_own": method_scores.means_own_scores(statistics, bank, zstats)[0]}


def baseline_rows(detector: dict, discopatch=None, activations=None) -> dict:
    """The baselines' rows (images, 96), except ContrastiveConf, whose lambda is fitted on each image set."""
    rows = {"saod_top3": detector["saod_top3"], "saod_min": detector["saod_min"],
            "knn": detector["knn"][:, :, KNN_K - 1]}
    if discopatch is not None:
        rows["discopatch"] = discopatch
    if activations is not None:
        rows.update(cdf=activations["cdf_backbone_z"], cdf_sum=activations["cdf_backbone"])
        if "hashemi_decoder" in activations:
            rows["hashemi"] = activations["hashemi_decoder"]
        if "hashemi_encoder" in activations:
            rows["hashemi_enc"] = activations["hashemi_encoder"]
    return rows


def contrastive_scores(conf_pos, conf_neg, ap, folds) -> tuple[np.ndarray, dict]:
    """ContrastiveConf on the given images; each fold's lambda is fitted on the other folds' clean images (UQ-DETR)."""
    lam, per_fold = cross_fit_lambda(conf_pos[:, 0], conf_neg[:, 0], ap, folds)
    return contrastive_degradation(conf_pos, conf_neg, lam[:, None]), per_fold


def _separation(clean, degraded) -> dict:
    return {"auroc": auroc(clean, degraded), "aupr": aupr(clean, degraded), "fpr95": fpr_at_95_tpr(clean, degraded)}


def separation_rows(scores: dict, folds, per_fold=()) -> list[dict]:
    """Each row's separation of every corrupted condition from the clean images; rows in `per_fold` fold-averaged."""
    folds = np.asarray(folds)
    out = []
    for method, values in scores.items():
        by_fold = method in per_fold
        for c in corruptions.CORRUPTED_CONDITIONS:
            family, severity = corruptions.CONDITIONS[c]
            if by_fold:
                parts = [_separation(values[folds == f, 0], values[folds == f, c]) for f in np.unique(folds)]
                result = {key: float(np.mean([p[key] for p in parts])) for key in METRICS}
            else:
                result = _separation(values[:, 0], values[:, c])
            out.append({"method": method, "family": family, "severity": int(severity),
                        "group": "common" if family in corruptions.COMMON_FAMILIES else "extra",
                        "pooling": "fold-averaged" if by_fold else "pooled", **result})
    return out


def aggregate_rows(rows: list[dict]) -> list[dict]:
    out = []
    for method in dict.fromkeys(r["method"] for r in rows):
        for group in ("common", "extra", "all"):
            for severity in (*corruptions.SEVERITIES, "all"):
                chosen = [r for r in rows if r["method"] == method
                          and (group == "all" or r["group"] == group)
                          and (severity == "all" or r["severity"] == severity)]
                out.append({"method": method, "group": group, "severity": severity,
                            **{key: float(np.mean([r[key] for r in chosen])) for key in METRICS}})
    return out


def headline_numbers(scores: dict, folds, per_fold=(), references=REFERENCES) -> dict:
    """Each row's six separation quantities, and each reference row minus every other row.

    Rows in `per_fold` are averaged over folds, as in separation_rows. A pair of two references appears once, as the
    earlier reference minus the later one.
    """
    folds = np.asarray(folds)
    out = {}
    for method, values in scores.items():
        for group, columns in (("common", corruptions.COMMON_CONDITIONS), ("extra", corruptions.EXTRA_CONDITIONS)):
            if method in per_fold:
                numbers = np.mean([group_separation(values[folds == f, 0], values[folds == f][:, columns].T)
                                   for f in np.unique(folds)], axis=0)
            else:
                numbers = group_separation(values[:, 0], values[:, columns].T)
            for metric, value in zip(METRICS, numbers):
                out[f"{method}:{metric}_{group}"] = float(value)
    present = [reference for reference in references if reference in scores]
    for index, reference in enumerate(present):
        for other in scores:
            if other != reference and other not in present[:index]:
                for quantity in QUANTITIES:
                    out[f"{reference} - {other}:{quantity}"] = out[f"{reference}:{quantity}"] - out[f"{other}:{quantity}"]
    return out


def level_decision(intervals: dict) -> str:
    """The rule pre-registered for the level score, applied to the held-out images' intervals."""
    needed = {other: [f"level - {other}:auroc_{group}" for group in GROUPS] for other in ("global_level", "cdf")}
    if not all(key in intervals for key in needed["global_level"]):
        return "unavailable: our method's scores are missing"
    if not all(intervals[key]["low"] > 0 for key in needed["global_level"]):
        return "not confirmed"
    if not all(key in intervals for key in needed["cdf"]):
        return "unavailable: the activation-CDF scores are missing"
    if all(intervals[key]["low"] > 0 for key in needed["cdf"]):
        return "confirmed"
    return "conditioning confirmed, not ahead of the activation CDFs"


def headline_decision(intervals: dict) -> str:
    """The headline rule, fixed before the untouched images were read.

    Confirmed when, on all images and on the untouched ones, the two-axis score beats the activation CDFs on the
    common and the extra families, and beats the level score on the common families, every interval excluding 0.
    """
    sets = ("all", "untouched")
    needed = [f"two_axis - cdf:auroc_{group}" for group in GROUPS] + ["two_axis - level:auroc_common"]
    if not all(key in intervals.get(name, {}) for name in sets for key in needed):
        return "unavailable: the two-axis or the activation-CDF scores are missing"
    ahead = {name: all(intervals[name][f"two_axis - cdf:auroc_{group}"]["low"] > 0 for group in GROUPS)
             for name in sets}
    if not ahead["all"]:
        return "not ahead of the activation CDFs"
    if not ahead["untouched"]:
        return "ahead of the activation CDFs on all images, but not on the untouched images"
    if not all(intervals[name]["two_axis - level:auroc_common"]["low"] > 0 for name in sets):
        return "ahead of the activation CDFs, but the flattening adds nothing over the level score on the common families"
    return "confirmed"


def _headline(aggregates: list) -> dict:
    out = {}
    for row in aggregates:
        if row["severity"] == "all" and row["group"] in GROUPS:
            for metric in METRICS:
                out.setdefault(row["method"], {})[f"{metric}_{row['group']}"] = row[metric]
    return out


def _by_severity(aggregates: list) -> dict:
    out = {}
    for row in aggregates:
        if row["severity"] != "all" and row["group"] in GROUPS:
            out.setdefault(row["method"], {}).setdefault(row["group"], []).append(row["auroc"])
    return out


def _by_family(separation: list) -> dict:
    out = {}
    for row in separation:
        out.setdefault(row["method"], {}).setdefault(row["family"], {})[str(row["severity"])] = row["auroc"]
    return out


def build_tables(scores: dict, detector: dict, ap, folds, condition_map, *, seed: int,
                 samples: int = BOOTSTRAP_SAMPLES, workers: int = 1, timing=None) -> tuple[dict, dict]:
    """Every table of the report, and its summary.

    `scores` holds every present row except ContrastiveConf as (images, 96) arrays in evaluation order.
    ContrastiveConf is built for each image set, and in every draw, from the detector's Conf+ and Conf- and the
    clean images' AP.
    """
    folds, ap = np.asarray(folds), np.asarray(ap)

    contrastive_ready = "conf_pos" in detector and "conf_neg" in detector

    def rows_of(chosen):
        """Every present row on the chosen images (repeats allowed), and ContrastiveConf's lambda per fold."""
        drawn, lam = {m: v[chosen] for m, v in scores.items()}, {}
        if contrastive_ready:
            drawn["contrastive"], lam = contrastive_scores(detector["conf_pos"][chosen], detector["conf_neg"][chosen],
                                                           ap[chosen], folds[chosen])
        return {m: drawn[m] for m in ROWS if m in drawn}, lam

    sets = image_sets(len(folds))
    tables = {"separation": [], "aggregates": [], "intervals": []}
    headline, by_severity, by_family, intervals, lambdas = {}, {}, {}, {}, {}
    for name, rows in sets.items():
        point_scores, lambdas[name] = rows_of(rows)
        if name == "all":
            all_scores = point_scores
        per_fold = () if len(set(lambdas[name].values())) <= 1 else ("contrastive",)
        separation = separation_rows(point_scores, folds[rows], per_fold)
        aggregates = aggregate_rows(separation)
        tables["separation"] += [{"subset": name, **row} for row in separation]
        tables["aggregates"] += [{"subset": name, **row} for row in aggregates]
        headline[name], by_severity[name] = _headline(aggregates), _by_severity(aggregates)
        by_family[name] = _by_family(separation)
        if name == "screen":
            continue

        def statistic(draw):
            drawn, _ = rows_of(rows[draw])
            return headline_numbers(drawn, folds[rows[draw]], per_fold)

        point = headline_numbers(point_scores, folds[rows], per_fold)
        ranges = bootstrap(statistic, len(rows), samples=samples, seed=seed, workers=workers)
        intervals[name] = {key: {"point": point[key], "low": low, "high": high} for key, (low, high) in ranges.items()}
        tables["intervals"] += [{"subset": name, "quantity": key, **value} for key, value in intervals[name].items()]
    knn = detector["knn"]
    tables["conditions"] = [{"family": f, "severity": s, "map": float(condition_map[c]),
                             **{f"mean_{m}": float(v[:, c].mean()) for m, v in all_scores.items()}}
                            for c, (f, s) in enumerate(corruptions.CONDITIONS)]
    tables["knn_k"] = [{"k": k, "mean_auroc_common": float(condition_aurocs(
        knn[:, 0, k - 1], knn[:, corruptions.COMMON_CONDITIONS, k - 1].T).mean())} for k in KNN_KS]
    tables["timing"] = [{"part": key, "ms": value} for key, value in (timing or {}).items() if key.endswith("_ms")]
    summary = {
        "images": len(folds), "image_sets": {name: len(rows) for name, rows in sets.items()},
        "screen_images": SCREEN_IMAGES, "untouched_start": UNTOUCHED_START, "folds": FOLDS,
        "lambda_per_fold": lambdas, "lambda_folds_agree": {n: len(set(v.values())) == 1 for n, v in lambdas.items() if v},
        "images_with_ap": int(np.isfinite(ap).sum()), "clean_map": float(condition_map[0]),
        "rows": list(all_scores), "knn_k": KNN_K, "theta": THETA, "hashemi_k": HASHEMI_K, "cdf_bins": CDF_BINS,
        "neighbours": method_reference.NEIGHBOURS, "key": method_reference.KEY_LAYER,
        "scored": list(method_reference.SCORED_LAYERS), "bootstrap_samples": samples, "seed": seed,
        "headline_row": "two_axis", "headline_decision": headline_decision(intervals),
        "level_decision": level_decision(intervals.get("held_out", {})),
        "headline": headline, "by_severity": by_severity, "by_family": by_family, "intervals": intervals,
        "labels": {m: LABELS[m] for m in all_scores},
    }
    return tables, summary


def _number(value) -> str:
    return "–" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.3f}"


def _signed(value) -> str:
    return f"{value:+.3f}".replace("-", "−")


def _cell(point, interval=None, show=_number) -> str:
    return show(point) + (f" [{show(interval['low'])}, {show(interval['high'])}]" if interval else "")


def _set_title(summary: dict, name: str) -> str:
    n = summary["image_sets"][name]
    return {"all": f"All {n} images (the headline, on the same images as every baseline)",
            "untouched": f"Untouched images ({n}, positions {summary['untouched_start']} and later, read by nobody "
                         "while the method was designed)",
            "held_out": f"Held-out images ({n}, positions {summary['screen_images']} and later: the pre-registered "
                        "check of the level score)",
            "screen": f"The {n} screening images"}[name]


def _set_section(summary: dict, name: str) -> list:
    head, intervals = summary["headline"][name], summary["intervals"].get(name, {})
    lines = [f"## {_set_title(summary, name)}", "",
             "| Row | AUROC common ↑ | AUROC extra ↑ | AUPR common ↑ | AUPR extra ↑ | FPR95 common ↓ | FPR95 extra ↓ |",
             "|---|---|---|---|---|---|---|"]
    for row in (r for r in ROWS if r in head):
        cells = [_cell(head[row][q], intervals.get(f"{row}:{q}")) for q in QUANTITIES]
        lines.append(f"| {LABELS[row]} | " + " | ".join(cells) + " |")
    reference = SECTION_REFERENCE.get(name)
    others = [r for r in ROWS if f"{reference} - {r}:auroc_common" in intervals]
    if others:
        lines += ["", f"{LABELS[reference]} minus each other row (Δ AUROC > 0 or Δ FPR95 < 0: it is better):", "",
                  "| Other row | Δ AUROC common | Δ AUROC extra | Δ FPR95 common | Δ FPR95 extra |",
                  "|---|---|---|---|---|"]
        for other in others:
            cells = [_cell(intervals[f"{reference} - {other}:{q}"]["point"], intervals[f"{reference} - {other}:{q}"],
                           _signed) for q in ("auroc_common", "auroc_extra", "fpr95_common", "fpr95_extra")]
            lines.append(f"| {LABELS[other]} | " + " | ".join(cells) + " |")
    return lines + [""]


def markdown(summary: dict, tables: dict) -> str:
    lines = ["# Corruption detection on COCO: our method and six baselines", "",
             f"**Headline (two-axis score, all images and the untouched ones):** {summary['headline_decision']}.", "",
             f"**Pre-registered level score (held-out images):** {summary['level_decision']}.", "",
             "Every score is oriented so that higher means more likely corrupted. ↑ higher is better, ↓ lower is "
             "better; an AUROC of 0.5 is chance. Brackets are 95% paired bootstrap intervals over images "
             f"({summary['bootstrap_samples']} draws, seed {summary['seed']}).", ""]
    for name in summary["headline"]:
        lines += _set_section(summary, name)
    severity = summary["by_severity"]["all"]
    lines += ["## AUROC by severity (all images)", "",
              "| Row | Common, severities 1–5 ↑ | Extra, severities 1–5 ↑ |", "|---|---|---|"]
    for row in (r for r in ROWS if r in severity):
        lines.append(f"| {LABELS[row]} | " + " / ".join(f"{v:.3f}" for v in severity[row]["common"]) + " | "
                     + " / ".join(f"{v:.3f}" for v in severity[row]["extra"]) + " |")
    family = summary["by_family"]["all"]
    shown = [r for r in FAMILY_ROWS if r in family]
    if shown:
        lines += ["", "## AUROC by family at severities 1 / 3 / 5 (all images)", "",
                  "| Family | " + " | ".join(FAMILY_ROWS[r] for r in shown) + " |", "|---|" + "---|" * len(shown)]
        for name in corruptions.FAMILIES:
            marker = " *" if name in corruptions.EXTRA_FAMILIES else ""
            cells = [" / ".join(f"{family[r][name][s]:.2f}" for s in ("1", "3", "5")) for r in shown]
            lines.append(f"| {name.replace('_', ' ')}{marker} | " + " | ".join(cells) + " |")
        lines += ["", "`*` marks the extra families."]
    conditions = tables.get("conditions", [])
    if conditions:
        lines += ["", "## Detector mAP (all images)", "", f"Clean: {conditions[0]['map']:.3f}.", "",
                  "| Severity | Common families | Extra families |", "|---|---|---|"]
        for level in corruptions.SEVERITIES:
            means = [np.mean([r["map"] for r in conditions if r["severity"] == level
                              and (r["family"] in corruptions.COMMON_FAMILIES) == (group == "common")])
                     for group in GROUPS]
            lines.append(f"| {level} | {means[0]:.3f} | {means[1]:.3f} |")
    if tables.get("knn_k"):
        lines += ["", "## kNN baseline by k (all images)", "", "| k | AUROC common |", "|---|---|"]
        lines += [f"| {r['k']} | {r['mean_auroc_common']:.3f} |" for r in tables["knn_k"]]
    if tables.get("timing"):
        lines += ["", "## Runtime (median ms per image, batch 1)", "", "| Part | ms |", "|---|---|"]
        lines += [f"| {r['part']} | {_number(r['ms'])} |" for r in tables["timing"]]
    scalars = {k: v for k, v in summary.items()
               if not isinstance(v, dict) and k not in ("headline_decision", "level_decision")}
    lines += ["", "## Fixed choices", ""] + [f"- {key}: {value}" for key, value in sorted(scalars.items())]
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(folder, tables: dict, summary: dict) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        if rows:
            _write_csv(folder / f"{name}.csv", rows)
    atomic_json(folder / "summary.json", summary)
    (folder / "report.md").write_text(markdown(summary, tables))
