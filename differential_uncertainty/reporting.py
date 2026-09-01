from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .corruptions import CORRUPTION_NAMES


_METHODS = ("fingerprint", "confidence", "entropy")
_SCORE_COLUMNS = ("image_id", "corruption", "severity", *_METHODS)
_TASK_COLUMNS = ("corruption", "severity", *(f"{method}_auroc" for method in _METHODS))


def _ordered_families(families) -> tuple[str, ...]:
    try:
        ordered = tuple(families)
    except TypeError as error:
        raise ValueError("families must be the fixed corruption roster in approved order") from error
    if ordered != CORRUPTION_NAMES:
        raise ValueError("families must be the fixed corruption roster in approved order")
    return CORRUPTION_NAMES


def _finite_number(value, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _ordered_scores(score_rows, families: tuple[str, ...]) -> list[dict]:
    family_order = {family: index for index, family in enumerate(families)}
    rows = []
    for index, row in enumerate(score_rows):
        if not isinstance(row, Mapping) or set(row) != set(_SCORE_COLUMNS):
            raise ValueError(f"score row {index} must have exactly the required keys")
        if not isinstance(row["image_id"], str) or not row["image_id"].strip():
            raise ValueError("score row image_id must be a non-empty string")
        if row["corruption"] not in family_order:
            raise ValueError("score row corruption must be requested")
        if isinstance(row["severity"], bool) or not isinstance(row["severity"], Integral):
            raise ValueError("score row severity must be an integer")
        normalized = dict(row)
        for method in _METHODS:
            normalized[method] = _finite_number(row[method], name=f"score row {method}")
        rows.append(normalized)
    return sorted(rows, key=lambda row: (
        family_order[row["corruption"]], row["image_id"], int(row["severity"])
    ))


def _validated_evaluation(evaluation, families: tuple[str, ...]) -> dict:
    if not isinstance(evaluation, Mapping) or set(evaluation) != {"tasks", "aggregate", "comparisons"}:
        raise ValueError("evaluation must contain exactly tasks, aggregate, and comparisons")
    tasks = evaluation["tasks"]
    expected = [(family, severity) for family in families for severity in (4, 5)]
    if not isinstance(tasks, list) or len(tasks) != len(expected):
        raise ValueError("evaluation tasks must cover every requested family at levels 4 and 5")
    normalized_tasks = []
    for task, identity in zip(tasks, expected):
        if not isinstance(task, Mapping) or set(task) != set(_TASK_COLUMNS):
            raise ValueError("evaluation task has an invalid schema")
        if (task["corruption"], task["severity"]) != identity:
            raise ValueError("evaluation tasks are not in requested family/severity order")
        normalized = dict(task)
        for method in _METHODS:
            normalized[f"{method}_auroc"] = _finite_number(
                task[f"{method}_auroc"], name=f"{method} AUROC"
            )
        normalized_tasks.append(normalized)
    aggregate = evaluation["aggregate"]
    if not isinstance(aggregate, Mapping) or set(aggregate) != set(_METHODS):
        raise ValueError("evaluation aggregate has an invalid schema")
    normalized_aggregate = {
        method: _finite_number(aggregate[method], name=f"aggregate {method}")
        for method in _METHODS
    }
    comparison_names = ("fingerprint_minus_confidence", "fingerprint_minus_entropy")
    comparisons = evaluation["comparisons"]
    if not isinstance(comparisons, Mapping) or set(comparisons) != set(comparison_names):
        raise ValueError("evaluation comparisons have an invalid schema")
    normalized_comparisons = {}
    for name in comparison_names:
        comparison = comparisons[name]
        if not isinstance(comparison, Mapping) or set(comparison) != {"point", "low", "high"}:
            raise ValueError("evaluation comparison has an invalid schema")
        normalized_comparisons[name] = {
            key: _finite_number(comparison[key], name=f"comparison {key}")
            for key in ("point", "low", "high")
        }
    return {
        "tasks": normalized_tasks,
        "aggregate": normalized_aggregate,
        "comparisons": normalized_comparisons,
    }


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _comparison_sentence(control: str, comparison: dict) -> str:
    point, low, high = (comparison[key] for key in ("point", "low", "high"))
    interval = f"point {point:.3f}, paired 95% interval [{low:.3f}, {high:.3f}]"
    if low > 0:
        return f"Fingerprint advantage over {control}: {interval}."
    if high < 0:
        return f"{control.capitalize()} advantage over fingerprint: {interval}."
    return f"Fingerprint versus {control}: inconclusive ({interval})."


def _render_report(evaluation: dict, families: tuple[str, ...]) -> str:
    tasks = {(task["corruption"], task["severity"]): task for task in evaluation["tasks"]}
    lines = [
        "# Fixed Corruption Evidence",
        "",
        "Fingerprint is, for each retained query, the mean cosine distance to the five nearest bank vectors, then the confidence-weighted mean across the retained scene queries.",
        "Confidence is one minus the maximum sigmoid confidence across retained queries.",
        "Entropy is normalized Shannon entropy of the highest-confidence retained query.",
        "Paired comparisons report the equal-weight macro-AUROC difference as the point estimate. The 2.5th and 97.5th percentiles from paired whole-image bootstrap resampling of the supplied evaluation set are descriptive stability intervals, not population or analytic confidence intervals.",
        "",
        "## Aggregate AUROC",
        "",
        "| Method | Equal-weight macro AUROC |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {method.capitalize()} | {evaluation['aggregate'][method]:.3f} |"
        for method in _METHODS
    )
    lines.extend([
        "",
        "## Paired comparisons",
        "",
        _comparison_sentence("confidence", evaluation["comparisons"]["fingerprint_minus_confidence"]),
        "",
        _comparison_sentence("entropy", evaluation["comparisons"]["fingerprint_minus_entropy"]),
        "",
        "## Per-corruption AUROC",
        "",
        "| Corruption | Fingerprint L4 | Confidence L4 | Entropy L4 | Fingerprint L5 | Confidence L5 | Entropy L5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for family in families:
        level4 = tasks[(family, 4)]
        level5 = tasks[(family, 5)]
        values = [
            level4["fingerprint_auroc"], level4["confidence_auroc"], level4["entropy_auroc"],
            level5["fingerprint_auroc"], level5["confidence_auroc"], level5["entropy_auroc"],
        ]
        lines.append(f"| {family} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
    lines.extend([
        "",
        "Levels 1 through 3 were not evaluated.",
        "",
        "![Per-corruption AUROC](corruption_auroc_bars.png)",
    ])
    return "\n".join(lines) + "\n"


def _write_chart(path: Path, tasks: list[dict], families: tuple[str, ...]) -> None:
    task_index = {(task["corruption"], task["severity"]): task for task in tasks}
    figure, axes = plt.subplots(1, 2, figsize=(13, 9), sharey=True, constrained_layout=True)
    try:
        positions = np.arange(len(families))
        offsets = (-0.24, 0.0, 0.24)
        colors = ("#3b6fb6", "#d48536", "#4f9d69")
        for axis, severity in zip(axes, (4, 5)):
            for method, offset, color in zip(_METHODS, offsets, colors):
                values = [task_index[(family, severity)][f"{method}_auroc"] for family in families]
                axis.barh(positions + offset, values, height=0.22, label=method.capitalize(), color=color)
            axis.set_title(f"Severity {severity}")
            axis.set_xlim(0, 1)
            axis.axvline(0.5, color="black", linewidth=0.9, linestyle="--")
            axis.set_xlabel("AUROC")
            axis.set_yticks(positions, families, fontsize=8)
            axis.invert_yaxis()
        axes[1].legend(loc="lower right")
        figure.savefig(path, dpi=160)
    finally:
        plt.close(figure)


def write_results(output: Path, score_rows: list[dict], evaluation: dict, families) -> None:
    """Atomically publish the five fixed corruption evidence files."""
    output = Path(output)
    ordered_families = _ordered_families(families)
    scores = _ordered_scores(score_rows, ordered_families)
    checked_evaluation = _validated_evaluation(evaluation, ordered_families)
    output.mkdir(parents=True, exist_ok=True)

    score_temporary = output / ".per_image_scores.tmp.csv"
    _write_csv(score_temporary, _SCORE_COLUMNS, scores)
    score_temporary.replace(output / "per_image_scores.csv")

    results_temporary = output / ".results.tmp.csv"
    _write_csv(results_temporary, _TASK_COLUMNS, checked_evaluation["tasks"])
    results_temporary.replace(output / "results.csv")

    summary_temporary = output / ".summary.tmp.json"
    with summary_temporary.open("w", encoding="utf-8") as handle:
        json.dump(checked_evaluation, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    summary_temporary.replace(output / "summary.json")

    chart_temporary = output / ".corruption_auroc_bars.tmp.png"
    _write_chart(chart_temporary, checked_evaluation["tasks"], ordered_families)
    chart_temporary.replace(output / "corruption_auroc_bars.png")

    report_temporary = output / ".report.tmp.md"
    report_temporary.write_text(
        _render_report(checked_evaluation, ordered_families), encoding="utf-8"
    )
    report_temporary.replace(output / "report.md")
