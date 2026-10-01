"""The 5,000-image confirmation of the content-conditioned reference: separation only, with the pre-registered rule.

Plan: docs/superpowers/plans/2026-10-01-content-conditioned-confirmation.md. The screen's 200 images are the first
images of the seed-44 order, so the primary numbers come from the other 4,800 ("held_out").
"""
from __future__ import annotations

import json

import numpy as np

from ..baselines import metrics, protocol
from ..baselines import report as baseline_report
from . import conditioned
from .channels import channel_method_scores

FOLDER = "results_convtu_conditioned"
ROWS = ("conditioned", "global_s123", "ch_means_knn", "ch_means_own", "cdf", "discopatch", "saod_min", "knn",
        "hashemi")
LABELS = {
    "conditioned": "Stages 1–3 vs the 50 most similar clean scenes (stage-4 key)",
    "global_s123": "Stages 1–3 vs the average of all clean images",
    "ch_means_knn": "Channel means, kNN, 4 stages (the pilot's control)",
    "ch_means_own": "Channel means vs own average, 4 stages",
    "cdf": "Activation CDFs (Becker et al., ICPR 2026)",
    "discopatch": "DisCoPatch",
    "saod_min": "SAOD, min (1 − max confidence)",
    "knn": "kNN (k = 100)",
    "hashemi": "Hashemi et al., decoder queries",
}
BOOTSTRAP_ROWS = ("conditioned", "global_s123", "ch_means_knn", "cdf", "discopatch")
BOOTSTRAP_SAMPLES = 1000
PRIMARY = ("global_s123", "cdf")  # the decision rule compares the conditioned score with these two
GROUPS = ("auroc_common", "auroc_extra")
SCREEN_AUROC = {"auroc_common": 0.841, "auroc_extra": 0.870}  # the screen, docs/dev-log.md
SCREEN_TOLERANCE = 0.003
BASELINE_KEYS = ("cdf", "discopatch", "saod_min", "knn", "hashemi")


def group_aurocs(scores: np.ndarray, rows) -> tuple[float, float]:
    """Mean AUROC over the common and over the extra conditions, on the given images."""
    values = np.asarray(scores, dtype=np.float64)[np.asarray(rows)]
    clean = values[:, 0]
    return (float(metrics.condition_aurocs(clean, values[:, baseline_report.COMMON].T).mean()),
            float(metrics.condition_aurocs(clean, values[:, baseline_report.EXTRA].T).mean()))


def bootstrap_intervals(scores: dict, rows, samples: int = BOOTSTRAP_SAMPLES, seed: int = 44) -> dict:
    """Paired image bootstrap of each row's mean AUROCs and of the conditioned row minus each other row."""
    rows = np.asarray(rows)

    def statistic(draw):
        point = {}
        for method, values in scores.items():
            common, extra = group_aurocs(values, rows[draw])
            point[f"{method}:auroc_common"], point[f"{method}:auroc_extra"] = common, extra
        if "conditioned" in scores:
            for other in scores:
                if other != "conditioned":
                    for group in GROUPS:
                        point[f"conditioned - {other}:{group}"] = point[f"conditioned:{group}"] - point[f"{other}:{group}"]
        return point

    ranges = metrics.bootstrap(statistic, len(rows), samples=samples, seed=seed)
    return {key: {"low": low, "high": high} for key, (low, high) in ranges.items()}


def decision(intervals: dict) -> str:
    """The plan's rule, applied to the held-out intervals."""
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


def _reproduction(settings, screen_names, means) -> dict:
    """The screen images' channel means must equal the pilot's stored means (same GPU, same code path)."""
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


def _cell(point, interval=None) -> str:
    return f"{point:.3f}" + (f" [{interval['low']:.3f}, {interval['high']:.3f}]" if interval else "")


def _signed(value: float) -> str:
    return f"{value:+.3f}".replace("-", "−")


def _difference_cell(point, interval) -> str:
    return f"{_signed(point)} [{_signed(interval['low'])}, {_signed(interval['high'])}]"


def _markdown(summary: dict) -> str:
    lines = ["# Content-conditioned reference: 5,000-image confirmation", "",
             f"**Decision (pre-registered rule, held-out images):** {summary['decision']}.", "",
             "↑ higher is better, ↓ lower is better. Brackets are 95% paired bootstrap intervals over images "
             f"({summary['bootstrap_samples']} draws, seed {summary['seed']}).", ""]
    for subset, title in (("held_out", f"Held-out images ({summary['held_out_images']}, the primary result)"),
                          ("all", f"All {summary['images']} images"),
                          ("screen", f"The {summary['screen_images']} screening images")):
        head, intervals = summary["headline"][subset], summary["intervals"].get(subset, {})
        lines += [f"## {title}", "", "| Row | AUROC common ↑ | AUROC extra ↑ | FPR95 common ↓ | FPR95 extra ↓ |",
                  "|---|---|---|---|---|"]
        for method in (m for m in ROWS if m in head):
            h = head[method]
            lines.append(f"| {LABELS[method]} | {_cell(h['auroc_common'], intervals.get(f'{method}:auroc_common'))} | "
                         f"{_cell(h['auroc_extra'], intervals.get(f'{method}:auroc_extra'))} | "
                         f"{h['fpr95_common']:.3f} | {h['fpr95_extra']:.3f} |")
        differences = [m for m in ROWS if f"conditioned - {m}:auroc_common" in intervals]
        if differences:
            lines += ["", "Differences, the conditioned row minus each other row (positive Δ: the conditioned row is better):",
                      "", "| Other row | Δ AUROC common ↑ | Δ AUROC extra ↑ |", "|---|---|---|"]
            for method in differences:
                cells = [_difference_cell(head["conditioned"][g] - head[method][g],
                                          intervals[f"conditioned - {method}:{g}"]) for g in GROUPS]
                lines.append(f"| {LABELS[method]} | {cells[0]} | {cells[1]} |")
        lines.append("")
    lines += ["## AUROC by severity (held-out images)", "", "| Row | Common, severities 1–5 ↑ | Extra, severities 1–5 ↑ |",
              "|---|---|---|"]
    severity = summary["by_severity"]["held_out"]
    for method in (m for m in ROWS if m in severity):
        lines.append(f"| {LABELS[method]} | " + " / ".join(f"{v:.3f}" for v in severity[method]["common"]) + " | "
                     + " / ".join(f"{v:.3f}" for v in severity[method]["extra"]) + " |")
    lines += ["", "## Checks", "",
              f"- Screen images' channel means vs the pilot's stored means, largest relative difference: "
              + ", ".join(f"{k} {v:.1e}" for k, v in summary["reproduction_max_relative"].items()) + ".",
              f"- Screen AUROC of the conditioned row: {summary['headline']['screen']['conditioned']['auroc_common']:.3f} / "
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
    scores = {"conditioned": conditioned.conditioned_scores(means, bank, zstats, k=conditioned.NEIGHBOURS)[0],
              "global_s123": conditioned.global_scores(means, bank, zstats)[0]}
    summed, _ = channel_method_scores(bank, zstats, means, [f"s{stage}" for stage in range(1, 5)], statistics=("means",))
    scores.update({"ch_means_knn": summed["ch_means_knn"], "ch_means_own": summed["ch_means_own"]})
    scores.update(_baselines(settings, names))
    folds = protocol.assign_folds(len(names))
    subsets = {"held_out": np.arange(screen_count, len(names)), "all": np.arange(len(names)),
               "screen": np.arange(screen_count)}
    separation, aggregates, headline, by_severity, intervals = [], [], {}, {}, {}
    for subset, rows in subsets.items():
        part = baseline_report.separation_rows({m: v[rows] for m, v in scores.items()}, folds[rows])
        summary_rows = baseline_report.aggregate_rows(part)
        separation += [{"subset": subset, **row} for row in part]
        aggregates += [{"subset": subset, **row} for row in summary_rows]
        headline[subset], by_severity[subset] = _headline(summary_rows), _by_severity(summary_rows)
        if subset != "screen":
            intervals[subset] = bootstrap_intervals({m: scores[m] for m in BOOTSTRAP_ROWS if m in scores}, rows,
                                                    samples=BOOTSTRAP_SAMPLES, seed=settings.seed)
    if SCREEN_AUROC:
        for group, expected in SCREEN_AUROC.items():
            if abs(headline["screen"]["conditioned"][group] - expected) > SCREEN_TOLERANCE:
                raise ValueError(f"the screen images do not reproduce the screen's {group} ({expected})")
    summary = {"images": len(names), "screen_images": screen_count, "held_out_images": len(names) - screen_count,
               "neighbours": conditioned.NEIGHBOURS, "key": conditioned.KEY_LAYER, "scored": list(conditioned.SCORED_LAYERS),
               "bootstrap_samples": BOOTSTRAP_SAMPLES, "seed": settings.seed, "reproduction_max_relative": reproduction,
               "decision": decision(intervals["held_out"]), "headline": headline, "by_severity": by_severity,
               "intervals": intervals, "labels": {m: LABELS[m] for m in scores}}
    folder = settings.output / FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    baseline_report._write_csv(folder / "separation.csv", separation)
    baseline_report._write_csv(folder / "aggregates.csv", aggregates)
    baseline_report._write_csv(folder / "intervals.csv", [{"subset": s, "quantity": q, **v}
                                                         for s, part in intervals.items() for q, v in part.items()])
    (folder / "report.md").write_text(_markdown(summary))
    _atomic_json(folder / "summary.json", summary)
