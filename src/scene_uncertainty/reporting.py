"""Turn scored scene rows into the study's CSV, summary, and trend plots.

Three things in here are easy to get subtly wrong, so they are stated once, up front:

* **The per-image trend statistics are computed over the unfiltered rows.** A severity
  where the query policy selected nothing has no score, and `monotonicity_metrics` drops
  those points itself and reports how many survived. Handing it a frame with the unscored
  severities already removed would make `finite_count == total_count` on every row and hide
  exactly the collapse the counts exist to expose.
* **The adjacent-step diagnostics are computed over the scored rows only.** Comparing a
  scored severity against an unscored one is meaningless for both of them:
  `jaccard_overlap` scores two empty selections 1.0, which would read as "selection was
  perfectly stable" when nothing was selected at either end, and a difference against a
  `nan` score is not a downward step. `empty_selection_frequency` and the severity counts
  are what report the missing severities, not the step statistics.
* **The class-switch split is blind to detection loss.** `has_class_switch` compares only
  annotations matched at *both* severities, so `no_switch_step_count` means "no annotation
  that was still detected changed its predicted class" and never "the detector was
  unaffected". Blur mostly makes detections disappear, and that does not show up here.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib

# The report is written on a headless box, so the backend is pinned before pyplot is
# imported rather than left to whatever matplotlib would autodetect.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import has_class_switch, jaccard_overlap, monotonicity_metrics


STRUCTURED_COLUMNS = ("layer_scores", "clean_scaled_layer_scores", "selected_query_ids", "matched_predictions")

# One scored row per key. `evaluate-knn` produces exactly one, but `report --results`
# accepts any CSV, and concatenating two result files is an obvious operator move.
RESULT_KEY_COLUMNS = ("image_id", "severity", "policy", "aggregation", "source_partition")


def find_duplicate_result_key(rows: list[dict]) -> tuple | None:
    """The first `RESULT_KEY_COLUMNS` tuple that appears twice, or None.

    A duplicated key is not a cosmetic problem. `write_report` joins every row against
    its own severity-0 row, so a key that appears twice multiplies through that join --
    two copies of one image's six rows produce four severity-0 pairings, and the adjacent
    step counts inflate quadratically rather than doubling. Measured on the six-severity
    fixture in `tests/scene_uncertainty/test_reporting.py`: duplicating the rows takes
    `no_switch_step_count` from 4 to 22 (not 8) and `scored_severity_count` from 6 to 12.
    None of the trend statistics notice: `median_spearman` still reads a confident 1.0,
    because every duplicated pair is perfectly consistent with itself.
    """
    seen = set()
    for row in rows:
        key = tuple(row[column] for column in RESULT_KEY_COLUMNS)
        if key in seen:
            return key
        seen.add(key)
    return None


def describe_result_key(key) -> str:
    return ", ".join(f"{column}={value}" for column, value in zip(RESULT_KEY_COLUMNS, key))


def write_result_csv(rows: list[dict], path: str | Path) -> None:
    """Write scored rows to CSV, JSON-encoding the nested columns.

    Row order is preserved and the header is the sorted union of every row's keys. The
    round trip through `read_result_csv` is value-preserving except for one thing: the
    nested mappings come back with **string** keys, because that is what JSON object keys
    are. `layer_scores` keyed by layer id and `matched_predictions` keyed by annotation id
    both go out as ints and come back as strings.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            for column in STRUCTURED_COLUMNS:
                if column in encoded:
                    encoded[column] = json.dumps(encoded[column], sort_keys=True)
            writer.writerow(encoded)
    os.replace(temporary, destination)


def read_result_csv(path: str | Path) -> list[dict]:
    """Read back what `write_result_csv` wrote, decoding the nested columns.

    `float_precision="round_trip"` is not optional. Pandas' default float converter is
    fast rather than exact, so a score written as `0.30000000000000004` comes back as
    `0.3`; it is intermittent, and most values survive it, which is exactly what makes it
    dangerous. The whole summary is computed from what this function returns, and
    `adjacent_monotonicity` and `endpoint_increase` are strict comparisons that a
    one-ULP shift can flip on a near-tie.
    """
    try:
        frame = pd.read_csv(path, float_precision="round_trip")
    except pd.errors.EmptyDataError:
        # `write_result_csv([])` has no columns to declare, so the file holds a bare
        # header line. Reading it back as no rows keeps the round trip total.
        return []
    for column in STRUCTURED_COLUMNS:
        if column in frame:
            frame[column] = frame[column].map(json.loads)
    if frame["valid"].dtype == object:
        frame["valid"] = frame["valid"].map({"True": True, "False": False})
    return frame.to_dict(orient="records")


def _expanded_frame(rows: list[dict]) -> pd.DataFrame:
    layer_ids = sorted({str(layer_id) for row in rows for layer_id in row["layer_scores"]})
    expanded = []
    for row in rows:
        expanded.append({**row, "score_scope": "combined", "score": row["raw_score"]})
        scores = {str(layer_id): score for layer_id, score in row["layer_scores"].items()}
        for layer_id in layer_ids:
            expanded.append({
                **row,
                "score_scope": f"layer_{layer_id}",
                "score": scores.get(layer_id, float("nan")),
            })
    return pd.DataFrame(expanded)


def _adjacent_diagnostics(image_group: pd.DataFrame) -> list[dict]:
    ordered = image_group.sort_values("severity", kind="stable")
    score_range = max(float(ordered["score"].max() - ordered["score"].min()), 1e-12)
    result = []
    records = ordered.to_dict(orient="records")
    for previous, current in zip(records, records[1:]):
        change = float(current["score"] - previous["score"])
        result.append({
            "query_overlap": jaccard_overlap(previous["selected_query_ids"], current["selected_query_ids"]),
            "class_switch": has_class_switch(previous["matched_predictions"], current["matched_predictions"]),
            "nondecreasing": change >= 0,
            "normalized_downward_change": max(-change, 0.0) / score_range,
        })
    return result


def _safe_mean(values) -> float | None:
    finite = np.asarray(list(values), dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else None


def _safe_median(values) -> float | None:
    finite = np.asarray(list(values), dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else None


def _plot_trend(valid: pd.DataFrame, column: str, ylabel: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for keys, group in valid.groupby(["source_partition", "policy", "aggregation", "score_scope"], sort=True):
        severity = group.groupby("severity")[column]
        median, q25, q75 = severity.median(), severity.quantile(0.25), severity.quantile(0.75)
        axis.plot(median.index, median.values, marker="o", label=" / ".join(keys))
        axis.fill_between(median.index, q25.values, q75.values, alpha=0.12)
    axis.set_xlabel("Gaussian blur severity")
    axis.set_ylabel(ylabel)
    if axis.get_legend_handles_labels()[0]:
        axis.legend(fontsize=6, ncol=2)
    figure.tight_layout()
    temporary = path.with_suffix(path.suffix + ".tmp")
    figure.savefig(temporary, format="png", dpi=160)
    plt.close(figure)
    os.replace(temporary, path)


def write_report(rows: list[dict], output_dir: str | Path, run_metadata: dict | None = None) -> None:
    """Write `per_scene.csv`, `summary.json`, `run_metadata.json`, and the three trend plots.

    Every group in `summary.json` carries `scored_severity_count`,
    `total_severity_count`, and `images_with_unscored_severities` beside its trend
    statistics, and `summary["diagnostics"]` rolls the same counts up over the whole run.
    They are what separate a config whose score rose across the entire blur sweep from one
    that rose only over the severities it survived -- those two publish identical
    `median_spearman` and `endpoint_increase_rate`, so without the counts the second one
    can outrank the first.

    `image_count` is every image the config swept; `scored_image_count` is the subset with
    two or more surviving severities, which is the denominator all four trend statistics
    are actually computed over. Both are published because they diverge exactly when the
    trend statistics get thin: a median over one surviving image out of four reads as a
    four-image finding without the second number next to it.
    """
    if not rows:
        raise ValueError("write_report needs at least one scored row")
    duplicate = find_duplicate_result_key(rows)
    if duplicate is not None:
        raise ValueError(
            "write_report needs one row per (image_id, severity, policy, aggregation, "
            f"source_partition); {describe_result_key(duplicate)} appears more than once. "
            "Concatenated result CSVs inflate the severity-0 join and the adjacent-step "
            "counts with no error, so this is refused rather than summarized."
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_result_csv(rows, output / "per_scene.csv")
    frame = _expanded_frame(rows)
    valid = frame[frame["valid"] & frame["score"].notna()].copy()
    clean = valid[valid["severity"] == 0][
        ["image_id", "source_partition", "policy", "aggregation", "score_scope", "score"]
    ].rename(columns={"score": "clean_score"})
    valid = valid.merge(clean, on=["image_id", "source_partition", "policy", "aggregation", "score_scope"], how="left")
    valid["clean_relative_score"] = valid["score"] - valid["clean_score"]

    groups = []
    key_columns = ["source_partition", "policy", "aggregation", "score_scope"]
    for keys, all_group in frame.groupby(key_columns, sort=True):
        group = valid[
            (valid["source_partition"] == keys[0])
            & (valid["policy"] == keys[1])
            & (valid["aggregation"] == keys[2])
            & (valid["score_scope"] == keys[3])
        ]
        # Trend statistics come from the unfiltered rows so that the severities with no
        # score are counted rather than silently absent; the adjacent-step diagnostics
        # come from the scored rows, where a comparison between two severities means
        # something. See the module docstring.
        image_metrics = [
            monotonicity_metrics(image_group["severity"], image_group["score"])
            for _, image_group in all_group.groupby("image_id", sort=True)
        ]
        # An image with fewer than two surviving severities has no trend to describe.
        # `monotonicity_metrics` says so with `nan` for three of the four statistics, but
        # `endpoint_increase` comes back a definite `False` there, which a plain mean
        # would happily average in and publish as "this image did not rise". Selecting
        # the measured images once gives all four statistics the same denominator, so
        # they go null together and `scored_image_count` reports what that denominator is.
        scored_metrics = [value for value in image_metrics if value["finite_count"] >= 2]
        adjacent = []
        for _, image_group in group.groupby("image_id", sort=True):
            adjacent.extend(_adjacent_diagnostics(image_group))
        switched = [step for step in adjacent if step["class_switch"]]
        stable = [step for step in adjacent if not step["class_switch"]]
        groups.append({
            "source_partition": keys[0],
            "policy": keys[1],
            "aggregation": keys[2],
            "score_scope": keys[3],
            "image_count": len(image_metrics),
            "scored_image_count": len(scored_metrics),
            "scored_severity_count": sum(value["finite_count"] for value in image_metrics),
            "total_severity_count": sum(value["total_count"] for value in image_metrics),
            "images_with_unscored_severities": sum(
                value["finite_count"] < value["total_count"] for value in image_metrics
            ),
            "empty_selection_frequency": float((~all_group["valid"]).mean()),
            "mean_adjacent_query_overlap": _safe_mean(step["query_overlap"] for step in adjacent),
            "median_spearman": _safe_median(value["spearman"] for value in scored_metrics),
            "mean_adjacent_monotonicity": _safe_mean(value["adjacent_monotonicity"] for value in scored_metrics),
            "mean_violation_magnitude": _safe_mean(value["violation_magnitude"] for value in scored_metrics),
            "endpoint_increase_rate": _safe_mean(value["endpoint_increase"] for value in scored_metrics),
            "class_switch_step_count": len(switched),
            "class_switch_monotonicity": _safe_mean(step["nondecreasing"] for step in switched),
            "class_switch_violation_magnitude": _safe_mean(step["normalized_downward_change"] for step in switched),
            "no_switch_step_count": len(stable),
            "no_switch_monotonicity": _safe_mean(step["nondecreasing"] for step in stable),
            "no_switch_violation_magnitude": _safe_mean(step["normalized_downward_change"] for step in stable),
            "median_selected_count_by_severity": {
                str(int(severity)): float(values.median())
                for severity, values in all_group.groupby("severity")["selected_count"]
            },
        })

    # A run-level flag, so a reader who never opens the per-group counts still sees that
    # some config lost severities. Only the count of affected groups is rolled up: the
    # severity counts themselves are per scope, and summing them across the combined and
    # per-layer scopes would count every scored image-severity several times over.
    diagnostics = {
        "group_count": len(groups),
        "groups_with_unscored_severities": sum(
            group["scored_severity_count"] < group["total_severity_count"] for group in groups
        ),
    }

    summary_path = output / "summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {"run_metadata": run_metadata or {}, "diagnostics": diagnostics, "groups": groups},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, summary_path)
    metadata_path = output / "run_metadata.json"
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(run_metadata or {}, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, metadata_path)
    _plot_trend(valid, "score", "Raw scene uncertainty", output / "raw_trend.png")
    _plot_trend(valid, "clean_relative_score", "Clean-relative scene uncertainty", output / "relative_trend.png")

    combined = pd.DataFrame([group for group in groups if group["score_scope"] == "combined"])
    figure, axis = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(combined))
    switch_values = [np.nan if value is None else value for value in combined["class_switch_monotonicity"]]
    stable_values = [np.nan if value is None else value for value in combined["no_switch_monotonicity"]]
    axis.bar(positions - 0.2, switch_values, width=0.4, label="class switch")
    axis.bar(positions + 0.2, stable_values, width=0.4, label="no switch")
    axis.set_xticks(positions, [
        f"{partition}/{policy}/{aggregation}"
        for partition, policy, aggregation in zip(
            combined["source_partition"], combined["policy"], combined["aggregation"]
        )
    ], rotation=90)
    axis.set_ylabel("Adjacent non-decrease rate")
    axis.legend()
    figure.tight_layout()
    temporary = output / "class_switch.png.tmp"
    figure.savefig(temporary, format="png", dpi=160)
    plt.close(figure)
    os.replace(temporary, output / "class_switch.png")
