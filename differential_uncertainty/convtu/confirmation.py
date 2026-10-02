"""The 5,000-image confirmation: separation only, with the pre-registered rules.

Plan: docs/superpowers/plans/2026-10-01-content-conditioned-confirmation.md.
- Headline (user, 1 October): the two-axis score on all 5,000 images, the same images as the baselines, checked on the
  images nobody looked at while the method was designed (positions UNTOUCHED_START and later, "untouched").
- Pre-registered before the run: the level score ("conditioned") on the 4,800 images after the screen ("held_out").
"""
from __future__ import annotations

import numpy as np

from degradation_monitor import corruptions
from degradation_monitor.datasets import coco
from degradation_monitor.evaluation import metrics
from degradation_monitor.method import reference as method_reference
from degradation_monitor.method import scores as method_scores
from ..baselines import report as baseline_report

FOLDER = "results_convtu_conditioned"
ROWS = ("two_axis", "peak_share", "conditioned", "global_s123", "ch_means_knn", "ch_means_own", "cdf", "discopatch",
        "saod_min", "knn", "hashemi")
LABELS = {
    "two_axis": "Two-axis: flatter or shifted vs the 50 most similar clean scenes (headline)",
    "peak_share": "Peak share vs the 50 most similar clean scenes",
    "conditioned": "Level vs the 50 most similar clean scenes (stage-4 key)",
    "global_s123": "Level vs the average of all clean images (stages 1–3)",
    "ch_means_knn": "Channel means, kNN, 4 stages (the pilot's control)",
    "ch_means_own": "Channel means vs own average, 4 stages",
    "cdf": "Activation CDFs (Becker et al., ICPR 2026)",
    "discopatch": "DisCoPatch",
    "saod_min": "SAOD, min (1 − max confidence)",
    "knn": "kNN (k = 100)",
    "hashemi": "Hashemi et al., decoder queries",
}
BOOTSTRAP_ROWS = ("two_axis", "peak_share", "conditioned", "global_s123", "ch_means_knn", "cdf", "discopatch")
REFERENCES = ("two_axis", "conditioned")  # rows whose differences with every other row get intervals
BOOTSTRAP_SAMPLES = 1000
PRIMARY = ("global_s123", "cdf")  # the pre-registered rule compares the level score with these two
GROUPS = ("auroc_common", "auroc_extra")
SCREEN_AUROC = {"auroc_common": 0.841, "auroc_extra": 0.870}  # the screen, docs/dev-log.md
SCREEN_TOLERANCE = 0.003
UNTOUCHED_START = 1970  # the roundtable read positions 200-1969 of the running pass; nobody read these
BASELINE_KEYS = ("cdf", "discopatch", "saod_min", "knn", "hashemi")
FAMILY_ROWS = {"two_axis": "Two-axis", "peak_share": "Peak share", "conditioned": "Level (similar scenes)",
               "cdf": "Activation CDFs", "discopatch": "DisCoPatch"}  # short column names for the family table


def group_aurocs(scores: np.ndarray, rows) -> tuple[float, float]:
    """Mean AUROC over the common and over the extra conditions, on the given images."""
    return metrics.group_aurocs(scores, rows)


def bootstrap_intervals(scores: dict, rows, samples: int = BOOTSTRAP_SAMPLES, seed: int = 44,
                        references=("conditioned",)) -> dict:
    """Paired image bootstrap of each row's mean AUROCs and of each reference row minus every other row.

    A pair of two reference rows appears once, as the earlier reference minus the later one.
    """
    rows = np.asarray(rows)
    present = [reference for reference in references if reference in scores]

    def statistic(draw):
        point = {}
        for method, values in scores.items():
            common, extra = group_aurocs(values, rows[draw])
            point[f"{method}:auroc_common"], point[f"{method}:auroc_extra"] = common, extra
        for index, reference in enumerate(present):
            for other in scores:
                if other == reference or other in present[:index]:
                    continue
                for group in GROUPS:
                    point[f"{reference} - {other}:{group}"] = point[f"{reference}:{group}"] - point[f"{other}:{group}"]
        return point

    ranges = metrics.bootstrap(statistic, len(rows), samples=samples, seed=seed)
    return {key: {"low": low, "high": high} for key, (low, high) in ranges.items()}


def decision(intervals: dict) -> str:
    """The plan's pre-registered rule for the level score, applied to the held-out intervals."""
    keys = {other: [f"conditioned - {other}:{group}" for group in GROUPS] for other in PRIMARY}
    if not all(key in intervals for key in keys["global_s123"]):
        raise ValueError("the global-average row is missing from the intervals")
    if not all(intervals[key]["low"] > 0 for key in keys["global_s123"]):
        return "not confirmed"
    if not all(key in intervals for key in keys["cdf"]):
        return "unavailable: the activation-CDF scores are missing"
    if all(intervals[key]["low"] > 0 for key in keys["cdf"]):
        return "confirmed"
    return "conditioning confirmed, not ahead of the activation CDFs"


def headline_decision(intervals: dict) -> str:
    """The headline rule, fixed before the untouched images were read.

    Confirmed when, on all images and on the untouched ones, the two-axis score beats the activation CDFs on the
    common and the extra families, and beats the level score on the common families, every interval excluding 0.
    """
    subsets = ("all", "untouched")
    needed = [f"two_axis - cdf:{group}" for group in GROUPS] + ["two_axis - conditioned:auroc_common"]
    if not all(key in intervals.get(subset, {}) for subset in subsets for key in needed):
        return "unavailable: the two-axis or the activation-CDF scores are missing"
    ahead = {subset: all(intervals[subset][f"two_axis - cdf:{group}"]["low"] > 0 for group in GROUPS)
             for subset in subsets}
    if not ahead["all"]:
        return "not ahead of the activation CDFs"
    if not ahead["untouched"]:
        return "ahead of the activation CDFs on all images, but not on the untouched images"
    if not all(intervals[subset]["two_axis - conditioned:auroc_common"]["low"] > 0 for subset in subsets):
        return "ahead of the activation CDFs, but the flattening adds nothing over the level score on the common families"
    return "confirmed"


def _reproduction(settings, screen_names, means) -> dict:
    """The screen images' channel statistics must equal the pilot's stored ones (same GPU, same code path)."""
    from .pipeline import CHANNELS_FOLDER, MEANS_KEYS, REPRODUCTION_RTOL
    pilot = baseline_report._stack(settings.output / CHANNELS_FOLDER, screen_names, MEANS_KEYS)
    out = {}
    for key in MEANS_KEYS:
        got, expected = means[key][:len(screen_names)], pilot[key]
        out[key] = float(np.abs(got - expected).max() / np.abs(expected).max())
        if out[key] > REPRODUCTION_RTOL:
            raise ValueError(f"the channel means of the screen images differ from the pilot's ({key}: {out[key]:.1e})")
    return out


def _baselines(settings, names) -> dict:
    test = baseline_report._stack(settings.output / "test", names, ("saod_min", "saod_top3", "conf_pos", "conf_neg", "knn"))
    folders = {"dcp": settings.output / "test_dcp", "activation": settings.output / "test_activation"}
    dcp = baseline_report._stack(folders["dcp"], names, ("dcp",))["dcp"] if folders["dcp"].exists() else None
    activation = (baseline_report._stack(folders["activation"], names, baseline_report.ACTIVATION_ARRAYS)
                  if folders["activation"].exists() else None)
    scores = baseline_report.method_scores(test, dcp, np.zeros(len(names)), activation=activation)
    return {key: scores[key] for key in BASELINE_KEYS if key in scores}


def _headline(aggregates: list) -> dict:
    out = {}
    for row in aggregates:
        if row["severity"] == "all" and row["group"] in ("common", "extra"):
            for metric in baseline_report.SEPARATION:
                out.setdefault(row["method"], {})[f"{metric}_{row['group']}"] = row[metric]
    return out


def _by_severity(aggregates: list) -> dict:
    out = {}
    for row in aggregates:
        if row["severity"] != "all" and row["group"] in ("common", "extra"):
            out.setdefault(row["method"], {}).setdefault(row["group"], []).append(row["auroc"])
    return out


def _by_family(separation: list) -> dict:
    out = {}
    for row in separation:
        out.setdefault(row["method"], {}).setdefault(row["family"], {})[str(row["severity"])] = row["auroc"]
    return out


def _cell(point, interval=None) -> str:
    return f"{point:.3f}" + (f" [{interval['low']:.3f}, {interval['high']:.3f}]" if interval else "")


def _signed(value: float) -> str:
    return f"{value:+.3f}".replace("-", "−")


def _difference_cell(point, interval) -> str:
    return f"{_signed(point)} [{_signed(interval['low'])}, {_signed(interval['high'])}]"


def _subset_section(summary: dict, subset: str, title: str, reference) -> list:
    head, intervals = summary["headline"][subset], summary["intervals"].get(subset, {})
    lines = [f"## {title}", "", "| Row | AUROC common ↑ | AUROC extra ↑ | FPR95 common ↓ | FPR95 extra ↓ |",
             "|---|---|---|---|---|"]
    for method in (m for m in ROWS if m in head):
        h = head[method]
        lines.append(f"| {LABELS[method]} | {_cell(h['auroc_common'], intervals.get(f'{method}:auroc_common'))} | "
                     f"{_cell(h['auroc_extra'], intervals.get(f'{method}:auroc_extra'))} | "
                     f"{h['fpr95_common']:.3f} | {h['fpr95_extra']:.3f} |")
    others = [m for m in ROWS if reference and f"{reference} - {m}:auroc_common" in intervals]
    if others:
        lines += ["", f"Differences, {LABELS[reference].split(' (')[0].lower()} minus each other row "
                      "(positive Δ: it is better):", "", "| Other row | Δ AUROC common ↑ | Δ AUROC extra ↑ |",
                  "|---|---|---|"]
        for method in others:
            cells = [_difference_cell(head[reference][g] - head[method][g], intervals[f"{reference} - {method}:{g}"])
                     for g in GROUPS]
            lines.append(f"| {LABELS[method]} | {cells[0]} | {cells[1]} |")
    return lines + [""]


def _markdown(summary: dict) -> str:
    lines = ["# Corruption detection from a frozen detector's early channels: 5,000-image confirmation", "",
             f"**Headline (two-axis score; all images and the untouched ones):** {summary['headline_decision']}.", "",
             f"**Pre-registered level score (held-out images):** {summary['decision']}.", "",
             "↑ higher is better, ↓ lower is better. Brackets are 95% paired bootstrap intervals over images "
             f"({summary['bootstrap_samples']} draws, seed {summary['seed']}).", ""]
    sections = (("all", f"All {summary['images']} images (the headline; the same images as the baselines)", "two_axis"),
                ("untouched", f"Untouched images ({summary['untouched_images']}, positions {summary['untouched_start']} "
                              "and later, read by nobody while the method was designed)", "two_axis"),
                ("held_out", f"Held-out images ({summary['held_out_images']}, positions {summary['screen_images']} and "
                             "later: the pre-registered check of the level score)", "conditioned"),
                ("screen", f"The {summary['screen_images']} screening images", None))
    for subset, title, reference in sections:
        if subset in summary["headline"]:
            lines += _subset_section(summary, subset, title, reference if reference in summary["headline"][subset] else None)
    lines += ["## AUROC by severity (all images)", "", "| Row | Common, severities 1–5 ↑ | Extra, severities 1–5 ↑ |",
              "|---|---|---|"]
    severity = summary["by_severity"]["all"]
    for method in (m for m in ROWS if m in severity):
        lines.append(f"| {LABELS[method]} | " + " / ".join(f"{v:.3f}" for v in severity[method]["common"]) + " | "
                     + " / ".join(f"{v:.3f}" for v in severity[method]["extra"]) + " |")
    family = summary["by_family"]["all"]
    shown = [m for m in FAMILY_ROWS if m in family]
    if shown:
        lines += ["", "## AUROC by family at severities 1 / 3 / 5 (all images)", "",
                  "| Family | " + " | ".join(FAMILY_ROWS[m] for m in shown) + " |",
                  "|---|" + "---|" * len(shown)]
        for name in corruptions.FAMILIES:
            marker = " *" if name in corruptions.EXTRA_FAMILIES else ""
            cells = [" / ".join(f"{family[m][name][s]:.2f}" for s in ("1", "3", "5")) for m in shown]
            lines.append(f"| {name.replace('_', ' ')}{marker} | " + " | ".join(cells) + " |")
        lines += ["", "`*` marks the extra families."]
    lines += ["", "## Checks", "",
              "- Screen images' channel statistics vs the pilot's stored ones, largest relative difference: "
              + ", ".join(f"{k} {v:.1e}" for k, v in summary["reproduction_max_relative"].items()) + ".",
              f"- Screen AUROC of the level row: {summary['headline']['screen']['conditioned']['auroc_common']:.3f} / "
              f"{summary['headline']['screen']['conditioned']['auroc_extra']:.3f} (the screen: "
              + (f"{SCREEN_AUROC['auroc_common']:.3f} / {SCREEN_AUROC['auroc_extra']:.3f}" if SCREEN_AUROC else "not checked")
              + ").",
              f"- Neighbours k = {summary['neighbours']}, key {summary['key']}, scored stages {', '.join(summary['scored'])}.", ""]
    return "\n".join(lines)


def build_confirmation_report(settings) -> None:
    from ..baselines.pipeline import _atomic_json, evaluation
    from .pipeline import MEANS_FOLDER, MEANS_KEYS, channels_bank_path, channels_zstats_path, pilot_images
    names = [p.name for p in evaluation(settings)]
    screen_count = len(pilot_images(settings))
    means = baseline_report._stack(settings.output / MEANS_FOLDER, names, MEANS_KEYS)
    with np.load(channels_bank_path(settings)) as bank, np.load(channels_zstats_path(settings)) as zstats:
        bank, zstats = {k: bank[k] for k in MEANS_KEYS}, {k: zstats[k] for k in MEANS_KEYS}
    reproduction = _reproduction(settings, names[:screen_count], means)
    k = method_reference.NEIGHBOURS
    scores = {"two_axis": method_scores.two_axis_scores(means, bank, zstats, k=k)[0],
              "peak_share": method_scores.peak_share_scores(means, bank, zstats, k=k)[0],
              "conditioned": method_scores.level_scores(means, bank, zstats, k=k)[0],
              "global_s123": method_scores.global_level_scores(means, bank, zstats)[0]}
    scores.update({"ch_means_knn": method_scores.means_knn_scores(means, bank, zstats)[0],
                   "ch_means_own": method_scores.means_own_scores(means, bank, zstats)[0]})
    scores.update(_baselines(settings, names))
    folds = coco.assign_folds(len(names))
    subsets = {"all": np.arange(len(names)), "untouched": np.arange(min(UNTOUCHED_START, len(names)), len(names)),
               "held_out": np.arange(screen_count, len(names)), "screen": np.arange(screen_count)}
    subsets = {name: rows for name, rows in subsets.items() if len(rows)}
    separation, aggregates, headline, by_severity, by_family, intervals = [], [], {}, {}, {}, {}
    for subset, rows in subsets.items():
        part = baseline_report.separation_rows({m: v[rows] for m, v in scores.items()}, folds[rows])
        summary_rows = baseline_report.aggregate_rows(part)
        separation += [{"subset": subset, **row} for row in part]
        aggregates += [{"subset": subset, **row} for row in summary_rows]
        headline[subset], by_severity[subset] = _headline(summary_rows), _by_severity(summary_rows)
        by_family[subset] = _by_family(part)
        if subset != "screen":
            intervals[subset] = bootstrap_intervals({m: scores[m] for m in BOOTSTRAP_ROWS if m in scores}, rows,
                                                    samples=BOOTSTRAP_SAMPLES, seed=settings.seed, references=REFERENCES)
    if SCREEN_AUROC:
        for group, expected in SCREEN_AUROC.items():
            if abs(headline["screen"]["conditioned"][group] - expected) > SCREEN_TOLERANCE:
                raise ValueError(f"the screen images do not reproduce the screen's {group} ({expected})")
    summary = {"images": len(names), "screen_images": screen_count, "held_out_images": len(names) - screen_count,
               "untouched_start": UNTOUCHED_START, "untouched_images": len(subsets.get("untouched", [])),
               "neighbours": k, "key": method_reference.KEY_LAYER, "scored": list(method_reference.SCORED_LAYERS),
               "bootstrap_samples": BOOTSTRAP_SAMPLES, "seed": settings.seed, "reproduction_max_relative": reproduction,
               "headline_row": "two_axis", "headline_decision": headline_decision(intervals),
               "decision": decision(intervals["held_out"]), "headline": headline, "by_severity": by_severity,
               "by_family": by_family, "intervals": intervals, "labels": {m: LABELS[m] for m in scores}}
    folder = settings.output / FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    baseline_report._write_csv(folder / "separation.csv", separation)
    baseline_report._write_csv(folder / "aggregates.csv", aggregates)
    baseline_report._write_csv(folder / "intervals.csv", [{"subset": s, "quantity": q, **v}
                                                         for s, part in intervals.items() for q, v in part.items()])
    (folder / "report.md").write_text(_markdown(summary))
    _atomic_json(folder / "summary.json", summary)
