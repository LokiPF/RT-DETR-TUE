from __future__ import annotations

import ctypes
import errno
import html
import json
import math
import os
import stat
import tempfile
import threading
import unicodedata
from collections.abc import Mapping
from contextlib import nullcontext
from numbers import Integral, Real
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import (
    SEVERITIES,
    paired_macro_bootstrap,
    summarize_series,
)
from .artifacts import _open_regular_file


REPORT_FILES = (
    "per-image-scores.csv",
    "metrics.csv",
    "bootstrap-comparisons.csv",
    "summary.json",
    "report.md",
    "figures/reference-and-responsive-distance.png",
    "figures/clean-reference-relationship.png",
    "figures/relative-gap-by-severity.png",
    "figures/auroc-by-corruption-severity.png",
)

_SCORE_COLUMNS = (
    "image_id",
    "severity",
    "padded_count",
    "valid_count",
    "reference_count",
    "responsive_count",
    "persistence_reference",
    "persistence_responsive",
    "persistence_relative_gap",
    "confidence_reference",
    "confidence_responsive",
    "confidence_relative_gap",
)
_COUNT_COLUMNS = (
    "padded_count",
    "valid_count",
    "reference_count",
    "responsive_count",
)
_SCORE_FIELDS = (
    "persistence_reference",
    "persistence_responsive",
    "persistence_relative_gap",
    "confidence_reference",
    "confidence_responsive",
    "confidence_relative_gap",
)
_SERIES = (
    "persistence_relative_gap",
    "confidence_relative_gap",
    "persistence_responsive",
    "persistence_reference",
)
_CONTROLS = (
    "confidence_relative_gap",
    "persistence_responsive",
    "persistence_reference",
)
_FIXED_ORIENTATIONS = {
    "persistence_relative_gap": 1,
    "confidence_relative_gap": -1,
    "raw_responsive": 1,
    "raw_reference": -1,
}
_SERIES_ORIENTATION_KEYS = {
    "persistence_relative_gap": "persistence_relative_gap",
    "confidence_relative_gap": "confidence_relative_gap",
    "persistence_responsive": "raw_responsive",
    "persistence_reference": "raw_reference",
}
_PYPLOT_LOCK = threading.RLock()


def _validated_text(value, *, name: str, maximum_length: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} text must be a nonempty string")
    if len(value) > maximum_length:
        raise ValueError(
            f"{name} text must be at most {maximum_length} characters"
        )
    if any(
        unicodedata.category(character).startswith("C")
        for character in value
    ):
        raise ValueError(f"{name} text must not contain control characters")
    return value


def _markdown_text(value: str) -> str:
    escaped = html.escape(value, quote=True)
    for character in "\\`*_[]|":
        escaped = escaped.replace(character, f"\\{character}")
    return escaped


def _markdown_code(value: str) -> str:
    escaped = html.escape(value, quote=True).replace("`", "&#96;")
    return f"`{escaped}`"


def _json_copy(value, *, name: str):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON values") from error


def _require_finite_tree(value, *, name: str) -> None:
    if isinstance(value, Mapping):
        for child in value.values():
            _require_finite_tree(child, name=name)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _require_finite_tree(child, name=name)
    elif isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must contain only finite numbers")


def _validate_provenance(provenance) -> dict:
    if not isinstance(provenance, Mapping):
        raise ValueError("provenance must be a mapping")
    normalized = _json_copy(provenance, name="provenance")
    _validated_text(
        normalized.get("checkpoint_sha256"),
        name="provenance checkpoint_sha256",
        maximum_length=256,
    )

    config = normalized.get("config")
    required_config = {
        "image_size",
        "class_count",
        "query_count",
        "persistence_layer",
        "persistence_dim",
        "bank_capacity",
        "bank_seed",
        "k",
        "bank_chunk_size",
        "reference_decile",
        "responsive_decile",
        "orientations",
        "bootstrap_samples",
        "bootstrap_seed",
        "default_blur_radii",
        "feature_normalization",
    }
    if not isinstance(config, dict) or set(config) != required_config:
        raise ValueError(
            "provenance scientific config must have the exact fixed fields"
        )

    def require_integer(key, *, minimum):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(
                f"provenance scientific config {key} is invalid"
            )
        return value

    image_size = config["image_size"]
    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in image_size
        )
    ):
        raise ValueError(
            "provenance scientific config image_size is invalid"
        )

    require_integer("class_count", minimum=1)
    require_integer("query_count", minimum=10)
    require_integer("persistence_layer", minimum=0)
    require_integer("persistence_dim", minimum=1)
    bank_capacity = require_integer("bank_capacity", minimum=1)
    require_integer("bank_seed", minimum=0)
    k = require_integer("k", minimum=1)
    require_integer("bank_chunk_size", minimum=1)
    reference_decile = require_integer("reference_decile", minimum=0)
    responsive_decile = require_integer("responsive_decile", minimum=0)
    require_integer("bootstrap_samples", minimum=1)
    require_integer("bootstrap_seed", minimum=0)
    if bank_capacity < k:
        raise ValueError(
            "provenance scientific config bank_capacity must cover k"
        )
    if (
        reference_decile >= 10
        or responsive_decile >= 10
        or reference_decile == responsive_decile
    ):
        raise ValueError(
            "provenance scientific config decile indices are invalid"
        )
    if config["orientations"] != _FIXED_ORIENTATIONS:
        raise ValueError(
            "provenance scientific config must use the fixed orientation mapping"
        )
    radii = config["default_blur_radii"]
    if (
        not isinstance(radii, list)
        or len(radii) != len(SEVERITIES)
        or any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in radii
        )
        or float(radii[0]) != 0.0
    ):
        raise ValueError(
            "provenance scientific config default_blur_radii is invalid"
        )
    if config["feature_normalization"] != "raw":
        raise ValueError(
            "provenance scientific config feature_normalization must be raw"
        )

    corruption = normalized.get("corruption")
    if not isinstance(corruption, dict):
        raise ValueError("provenance needs a corruption mapping")
    _validated_text(
        corruption.get("name"),
        name="provenance corruption name",
        maximum_length=128,
    )
    severities = corruption.get("severities")
    if not isinstance(severities, list) or len(severities) != len(SEVERITIES):
        raise ValueError("provenance corruption needs levels 0 through 5")
    levels = []
    for item in severities:
        if not isinstance(item, dict):
            raise ValueError("provenance corruption severities must be mappings")
        level = item.get("level")
        parameter = item.get("parameter")
        if isinstance(level, bool) or not isinstance(level, int):
            raise ValueError("provenance corruption levels must be integers")
        if (
            isinstance(parameter, bool)
            or not isinstance(parameter, Real)
            or not math.isfinite(float(parameter))
        ):
            raise ValueError("provenance corruption parameters must be finite numbers")
        levels.append(level)
    if levels != list(SEVERITIES):
        raise ValueError("provenance corruption needs levels 0 through 5")
    if (
        normalized["corruption"]["name"] == "gaussian_blur"
        and [float(item["parameter"]) for item in severities]
        != [float(radius) for radius in config["default_blur_radii"]]
    ):
        raise ValueError(
            "Gaussian corruption parameters must match config "
            "default_blur_radii"
        )
    return normalized


def _expected_gap(reference: float, responsive: float) -> float:
    total = reference + responsive
    return 0.0 if total == 0.0 else 2.0 * (responsive - reference) / total


def _validated_frame(rows, *, config: Mapping) -> pd.DataFrame:
    rows = list(rows)
    if not rows:
        raise ValueError("report rows need at least one complete image")
    checked = []
    seen: set[tuple[str, int]] = set()
    by_image: dict[str, set[int]] = {}
    count_signatures: dict[str, set[tuple[int, int, int, int]]] = {}
    query_count = config["query_count"]
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"score row {index} must be a mapping")
        if set(raw) != set(_SCORE_COLUMNS):
            missing = sorted(set(_SCORE_COLUMNS) - set(raw))
            extra = sorted(set(raw) - set(_SCORE_COLUMNS))
            raise ValueError(
                f"score row {index} does not match Task 8 columns; "
                f"missing={missing}, extra={extra}"
            )
        row = dict(raw)
        image_id = _validated_text(
            row["image_id"],
            name="score row image_id",
            maximum_length=256,
        )
        if image_id.lstrip().startswith(("=", "+", "-", "@")):
            raise ValueError(
                "score row image_id violates CSV safety: leading formula "
                "characters are not allowed"
            )
        severity = row["severity"]
        if isinstance(severity, bool) or not isinstance(severity, Integral):
            raise ValueError("score row severity must be an integer")
        severity = int(severity)
        if severity not in SEVERITIES:
            raise ValueError("score row severity must lie between 0 and 5")
        key = (image_id, severity)
        if key in seen:
            raise ValueError(
                f"duplicate score row for image {image_id!r}, severity {severity}"
            )
        seen.add(key)
        by_image.setdefault(image_id, set()).add(severity)
        row["severity"] = severity

        for field in _COUNT_COLUMNS:
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(
                    f"score row {field} must be a non-negative integer"
                )
            value = int(value)
            if value < 0:
                raise ValueError(
                    f"score row {field} must be a non-negative integer"
                )
            row[field] = value
        if row["padded_count"] + row["valid_count"] != query_count:
            raise ValueError(
                "padded_count plus valid_count must equal query_count"
            )
        if row["valid_count"] < 10:
            raise ValueError(
                "score rows need at least ten valid queries for deciles"
            )
        quotient, remainder = divmod(row["valid_count"], 10)

        def decile_size(decile):
            return quotient + int(decile < remainder)

        expected_reference = decile_size(config["reference_decile"])
        expected_responsive = decile_size(config["responsive_decile"])
        if (
            row["reference_count"] != expected_reference
            or row["responsive_count"] != expected_responsive
        ):
            raise ValueError(
                "reference/responsive counts do not match the configured decile"
            )
        signature = tuple(row[field] for field in _COUNT_COLUMNS)
        count_signatures.setdefault(image_id, set()).add(signature)

        for field in _SCORE_FIELDS:
            value = row[field]
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"score row {field} must be a finite real number"
                )
            row[field] = float(value)
        if (
            row["persistence_reference"] < 0.0
            or row["persistence_responsive"] < 0.0
        ):
            raise ValueError("persistence means must be non-negative")
        if not (
            0.0 <= row["confidence_reference"] <= 1.0
            and 0.0 <= row["confidence_responsive"] <= 1.0
        ):
            raise ValueError("confidence means must lie in [0, 1]")
        for prefix in ("persistence", "confidence"):
            expected = _expected_gap(
                row[f"{prefix}_reference"], row[f"{prefix}_responsive"]
            )
            if not math.isclose(
                row[f"{prefix}_relative_gap"],
                expected,
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{prefix}_relative_gap does not match its group means"
                )
        checked.append(row)

    for image_id, levels in by_image.items():
        if levels != set(SEVERITIES):
            raise ValueError(
                f"image {image_id!r} does not have severities 0 through 5"
            )
        if len(count_signatures[image_id]) != 1:
            raise ValueError(
                f"image {image_id!r} must have the same counts across all six "
                "severities"
            )

    return (
        pd.DataFrame(checked, columns=_SCORE_COLUMNS)
        .sort_values(["image_id", "severity"], kind="stable")
        .reset_index(drop=True)
    )


def _validate_evaluation(evaluation, frame: pd.DataFrame, provenance: dict) -> dict:
    if not isinstance(evaluation, Mapping):
        raise ValueError("evaluation must be a mapping")
    _require_finite_tree(evaluation, name="evaluation")
    _json_copy(evaluation, name="evaluation")
    if set(evaluation) != {"series", "bootstrap_comparisons"}:
        raise ValueError("evaluation must keep the exact Task 8 structure")
    series = evaluation["series"]
    if not isinstance(series, Mapping) or set(series) != set(_SERIES):
        raise ValueError("evaluation series must contain the four fixed scores")

    score_rows = frame.to_dict(orient="records")
    canonical_series = {}
    for name in _SERIES:
        supplied = series[name]
        if not isinstance(supplied, Mapping):
            raise ValueError(f"evaluation series {name} must be a mapping")
        if supplied.get("field") != name:
            raise ValueError(f"evaluation series {name} has the wrong field")
        orientation = supplied.get("orientation")
        if (
            isinstance(orientation, bool)
            or not isinstance(orientation, Integral)
            or int(orientation) not in (-1, 1)
        ):
            raise ValueError(
                f"evaluation series {name} has an invalid orientation"
            )
        expected_orientation = provenance["config"]["orientations"][
            _SERIES_ORIENTATION_KEYS[name]
        ]
        if int(orientation) != expected_orientation:
            raise ValueError(
                f"evaluation series {name} orientation differs from provenance"
            )
        expected = summarize_series(score_rows, name, expected_orientation)
        if _json_copy(supplied, name="evaluation") != _json_copy(
            expected, name="expected evaluation"
        ):
            raise ValueError(
                f"evaluation series {name} does not match supplied rows"
            )
        canonical_series[name] = expected

    comparisons = evaluation["bootstrap_comparisons"]
    if not isinstance(comparisons, list) or len(comparisons) != len(_CONTROLS):
        raise ValueError("evaluation needs the three fixed bootstrap comparisons")
    config = provenance["config"]
    canonical_comparisons = []
    for index, (item, control) in enumerate(zip(comparisons, _CONTROLS)):
        if not isinstance(item, Mapping):
            raise ValueError(f"bootstrap comparison {index} must be a mapping")
        if item.get("candidate") != "persistence_relative_gap":
            raise ValueError(
                "bootstrap candidate must be persistence_relative_gap"
            )
        if item.get("control") != control:
            raise ValueError("bootstrap controls must keep their fixed order")
        expected_comparison = paired_macro_bootstrap(
            score_rows,
            "persistence_relative_gap",
            control,
            candidate_orientation=series[
                "persistence_relative_gap"
            ]["orientation"],
            control_orientation=series[control]["orientation"],
            samples=config["bootstrap_samples"],
            seed=config["bootstrap_seed"],
        )
        if _json_copy(item, name="bootstrap comparison") != _json_copy(
            expected_comparison,
            name="expected bootstrap comparison",
        ):
            raise ValueError(
                f"bootstrap comparison {index} does not match supplied "
                "rows and provenance"
            )
        canonical_comparisons.append(expected_comparison)
    return {
        "series": canonical_series,
        "bootstrap_comparisons": canonical_comparisons,
    }


def _metric_rows(evaluation: Mapping) -> list[dict]:
    rows = []
    for name in sorted(evaluation["series"]):
        values = evaluation["series"][name]
        row = {
            key: value
            for key, value in values.items()
            if key not in {"auroc_by_severity", "severity_statistics"}
        }
        row["series"] = name
        row.update({
            f"auroc_severity_{level}": values["auroc_by_severity"][level]
            for level in SEVERITIES[1:]
        })
        rows.append(row)
    return rows


def _median_curve(frame: pd.DataFrame, field: str) -> np.ndarray:
    return (
        frame.groupby("severity", sort=True)[field]
        .median()
        .reindex(SEVERITIES)
        .to_numpy()
    )


def _save_close(fig, path: Path) -> None:
    try:
        fig.savefig(
            path,
            dpi=180,
            facecolor="white",
            metadata={"Software": "differential_uncertainty"},
        )
    finally:
        plt.close(fig)


def _write_figures_impl(
    frame: pd.DataFrame,
    evaluation: Mapping,
    directory: Path,
    config: Mapping,
) -> None:
    severity = np.asarray(SEVERITIES)
    reference_label = (
        f"{10 * config['reference_decile']}-"
        f"{10 * (config['reference_decile'] + 1)}% reference"
    )
    responsive_label = (
        f"{10 * config['responsive_decile']}-"
        f"{10 * (config['responsive_decile'] + 1)}% responsive"
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.5), constrained_layout=True
    )
    for axis, prefix, title in (
        (
            axes[0],
            "persistence",
            f"Layer-{config['persistence_layer']} distance",
        ),
        (axes[1], "confidence", "1 - maximum sigmoid confidence"),
    ):
        axis.plot(
            severity,
            _median_curve(frame, f"{prefix}_reference"),
            marker="o",
            label=reference_label,
        )
        axis.plot(
            severity,
            _median_curve(frame, f"{prefix}_responsive"),
            marker="o",
            label=responsive_label,
        )
        axis.set(
            title=title,
            xlabel="Corruption level",
            ylabel="Median group mean",
            xticks=severity,
        )
        axis.grid(alpha=0.2)
        axis.legend()
    _save_close(fig, directory / "reference-and-responsive-distance.png")

    clean = frame[frame["severity"] == 0]
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 4.5), constrained_layout=True
    )
    axes[0].scatter(
        clean["persistence_reference"],
        clean["persistence_responsive"],
        s=24,
    )
    axes[1].scatter(
        clean["confidence_reference"],
        clean["confidence_responsive"],
        s=24,
    )
    for axis, title in zip(
        axes,
        (
            "Persistence on clean images",
            "Confidence uncertainty on clean images",
        ),
    ):
        axis.set(
            title=title,
            xlabel="Reference mean",
            ylabel="Responsive mean",
        )
        axis.locator_params(axis="x", nbins=5)
        axis.grid(alpha=0.2)
    _save_close(fig, directory / "clean-reference-relationship.png")

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.5), constrained_layout=True
    )
    for axis, field, title in (
        (
            axes[0],
            "persistence_relative_gap",
            "Persistence relative gap",
        ),
        (
            axes[1],
            "confidence_relative_gap",
            "Matched confidence relative gap",
        ),
    ):
        groups = [
            frame.loc[frame["severity"] == level, field].to_numpy()
            for level in SEVERITIES
        ]
        axis.boxplot(groups, positions=severity, showfliers=False)
        axis.axhline(0.0, color="black", linestyle=":", linewidth=1)
        axis.set(
            title=title,
            xlabel="Corruption level",
            ylabel="Relative gap",
            xticks=severity,
        )
        axis.title.set_fontsize(15)
        axis.grid(axis="y", alpha=0.2)
    _save_close(fig, directory / "relative-gap-by-severity.png")

    fig, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    labels = {
        "persistence_relative_gap": "Persistence relative gap",
        "confidence_relative_gap": "Matched confidence",
        "persistence_responsive": "Raw responsive control",
        "persistence_reference": "Raw reference control",
    }
    for name, label in labels.items():
        values = evaluation["series"][name]["auroc_by_severity"]
        axis.plot(
            list(SEVERITIES[1:]),
            [values[level] for level in SEVERITIES[1:]],
            marker="o",
            label=label,
        )
    axis.axhline(
        0.5,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Chance (0.5)",
    )
    axis.set(
        xlabel="Corruption level",
        ylabel="AUROC",
        ylim=(0, 1.03),
        xticks=SEVERITIES[1:],
        yticks=np.linspace(0, 1, 6),
        title="Clean versus corrupted ranking",
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, ncols=2)
    _save_close(fig, directory / "auroc-by-corruption-severity.png")


def _write_figures(
    frame: pd.DataFrame,
    evaluation: Mapping,
    directory: Path,
    config: Mapping,
) -> None:
    with _PYPLOT_LOCK:
        caller_figures = set(plt.get_fignums())
        try:
            _write_figures_impl(frame, evaluation, directory, config)
        finally:
            for number in set(plt.get_fignums()) - caller_figures:
                plt.close(number)


def render_report(
    frame: pd.DataFrame, evaluation: Mapping, provenance: Mapping
) -> str:
    primary = evaluation["series"]["persistence_relative_gap"]
    confidence = evaluation["series"]["confidence_relative_gap"]
    example = frame.sort_values(
        ["image_id", "severity"], kind="stable"
    ).iloc[0]
    reference = example["persistence_reference"]
    responsive = example["persistence_responsive"]
    gap = example["persistence_relative_gap"]
    confidence_reference = example["confidence_reference"]
    confidence_responsive = example["confidence_responsive"]
    confidence_gap = example["confidence_relative_gap"]
    config = provenance["config"]
    corruption_name = _markdown_text(
        provenance["corruption"]["name"].replace("_", " ")
    )
    example_id = _markdown_code(str(example["image_id"]))
    checkpoint = _markdown_code(provenance["checkpoint_sha256"])
    reference_range = (
        f"{10 * config['reference_decile']}-"
        f"{10 * (config['reference_decile'] + 1)}%"
    )
    responsive_range = (
        f"{10 * config['responsive_decile']}-"
        f"{10 * (config['responsive_decile'] + 1)}%"
    )
    severity_parameters = ", ".join(
        f"level {item['level']} = {item['parameter']:g}"
        for item in provenance["corruption"]["severities"]
    )
    auroc_lines = "\n".join(
        f"| {level} | {primary['auroc_by_severity'][level]:.3f} | "
        f"{confidence['auroc_by_severity'][level]:.3f} |"
        for level in SEVERITIES[1:]
    )
    bootstrap_lines = "\n".join(
        f"| {item['control'].replace('_', ' ')} | "
        f"{item['point_difference']:+.3f} | "
        f"[{item['ci_low']:+.3f}, {item['ci_high']:+.3f}] |"
        for item in evaluation["bootstrap_comparisons"]
    )

    return f"""# Differential corruption uncertainty report

## What was tested

We tested **{corruption_name}** at six ordered levels. Level 0 is the clean image,
and levels 1 through 5 are the configured corrupted versions. The detector made
{config['query_count']} query guesses for every version of each image. Sometimes
a detector fills unused spaces by repeating its last guess. We removed those
exact repeated, padded guesses before doing any calculation.

The configured corruption parameters were: {severity_parameters}.

For each image, we ranked the remaining guesses by their maximum **sigmoid**
class confidence. We compared two fixed groups: the {reference_range} group,
called the reference group, and the {responsive_range} group, called the
responsive group. Both groups use the detector fingerprint from decoder layer
{config['persistence_layer']}.

For every selected query, we found its **{config['k']} nearest clean** bank
fingerprints and averaged their distances. Then we averaged those query
distances inside each group. Here is a toy five-neighbor example, separate from
this run: its actual k is {config['k']}. If a query's five nearest distances are
0.10, 0.14, 0.17, 0.21, and 0.28, its toy score is:

    (0.10 + 0.14 + 0.17 + 0.21 + 0.28) / 5 = 0.18

A larger distance means that the fingerprint looks less like the saved clean
fingerprints.

## How the persistence relative gap was calculated

For image {example_id} at corruption level
{int(example['severity'])}, the reference mean was {reference:.6f} and the
responsive mean was {responsive:.6f}. We used:

    2 x (responsive - reference) / (responsive + reference)

Putting this run's real numbers into the formula gives:

    2 x ({responsive:.6f} - {reference:.6f})
        / ({responsive:.6f} + {reference:.6f}) = {gap:.6f}

This **relative gap** is scale independent. For example, 2 and 6 give the same
gap as 20 and 60. Multiplying both inputs by the same positive number does not
change the answer. We chose it because that meaning is easy to compare across images.
The earlier work did not prove that it was better than a raw subtraction.

## How the matched-confidence score was calculated

The matched-confidence baseline uses the **same query IDs** as the persistence
test, so the two methods judge exactly the same detector guesses. First, a
query's uncertainty is one minus its maximum sigmoid confidence. A highest
confidence of 0.90 becomes:

    1 - 0.90 = 0.10

For a small example, confidences 0.90 and 0.80 have uncertainties 0.10 and 0.20,
whose mean is 0.15. Confidences 0.60 and 0.50 have uncertainties 0.40 and 0.50,
whose mean is 0.45. The same relative-gap formula gives:

    2 x (0.45 - 0.15) / (0.45 + 0.15) = 1.00

In this run's concrete row, the reference and responsive uncertainty means
were {confidence_reference:.6f} and {confidence_responsive:.6f}:

    2 x ({confidence_responsive:.6f} - {confidence_reference:.6f})
        / ({confidence_responsive:.6f} + {confidence_reference:.6f})
        = {confidence_gap:.6f}

People sometimes call this “plain softmax” in conversation, but it is
**not softmax**. The detector has independent classes, so the implementation uses the
maximum sigmoid class confidence. The archived experiment fixed this score's
direction at -1, which means a smaller raw confidence gap ranks as more
corrupted.

## Results

| Corruption level | Persistence AUROC | Matched confidence AUROC |
|---|---:|---:|
{auroc_lines}
| Average | {primary['macro_auroc']:.3f} | {confidence['macro_auroc']:.3f} |

### How AUROC was calculated for each severity

Each severity gets its own AUROC. For level 1, for example, the clean group
contains the level-0 clean scores from all images, and the corrupted group
contains the level-1 corrupted scores from all images. We do the same separate
comparison for each severity from 1 through 5. We compare every corrupted score
with every clean score after applying the score's frozen direction.

If the clean scores are `[0, 1]` and the corrupted scores are `[1, 2]`,
three pairs are wins and one pair is a tie worth half a win. Therefore AUROC is
`3.5 / 4 = 0.875`. AUROC is **not a probability** that one particular image
is corrupted. A value of 0.5 is chance ranking, while 1.0 is perfect ranking.
The average in the table is the macro-AUROC: the simple average of the five
severity AUROCs.

### Other checks, not only AUROC

We also calculated **Spearman** rank correlation. It asks whether each image's
six scores move in order as corruption grows. Levels
`[0, 1, 2, 3, 4, 5]` and scores `[10, 20, 30, 40, 50, 60]` have the same
ranks, so Spearman is +1. Reversing the score order gives -1. A completely flat
curve is recorded as 0. Persistence's median signed Spearman in this run was
{primary['median_signed_spearman']:+.3f}; its median absolute Spearman was
{primary['median_absolute_spearman']:.3f}.

We also checked **adjacent consistency**. One image has five adjacent steps:
0→1, 1→2, 2→3, 3→4, and 4→5. If four steps move in the expected direction,
its adjacent consistency is 4 / 5 = 0.80. Across this run, the oriented
persistence score moved the expected way on
{primary['oriented_adjacent_consistency']:.1%} of adjacent steps.

Finally, the **strongest corruption** check asks whether level 5 ranks above
level 0 for each image. That happened for
{primary['strongest_blur_above_clean_rate']:.1%} of images. These checks tell
us about smooth change within images, while AUROC tells us about ranking across
the whole group.

## Paired bootstrap comparisons

| Control | Persistence macro-AUROC minus control | 95% interval |
|---|---:|---:|
{bootstrap_lines}

The paired bootstrap redraws whole image IDs
{config['bootstrap_samples']:,} times. Each redraw keeps all six levels from an
image together, so the clean and corrupted versions stay paired. The interval
shows how much the result changes when this supplied image set is resampled.
It does not turn AUROC into a probability and does not prove a universal effect.

## What this does not prove

This workflow has no object labels, so it **does not measure mAP**, detection accuracy,
or calibration. It cannot tell us whether the detector found the
right objects. The bins, layer, relative-gap formula, and score directions came
from earlier tuning. Fresh numbers may differ from the historical run
because this clean workflow also removes padded queries from the reference
bank.

The complete per-image rows, metric tables, bootstrap comparisons, figures, and
reproducibility details are stored beside this report. Checkpoint SHA-256:
{checkpoint}.
"""


_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | os.O_DIRECTORY
    | os.O_NOFOLLOW
    | getattr(os, "O_CLOEXEC", 0)
)
_AT_FDCWD = -100


class _OwnedDirectoryDescriptor(ctypes.c_int):
    """Own a libc-opened directory fd from the C return boundary."""

    @property
    def fd(self) -> int:
        self._reconcile_pending()
        descriptor = int(self.value)
        if descriptor < 0:
            raise RuntimeError("directory descriptor is closed")
        return descriptor

    @property
    def closed(self) -> bool:
        self._reconcile_pending()
        return int(self.value) < 0

    def remember_identity(self, state) -> None:
        position = id(self)
        os.lseek(int(self.value), position, os.SEEK_SET)
        self.identity = (
            state.st_dev,
            state.st_ino,
            stat.S_IFMT(state.st_mode),
            position,
        )
        self._pending_descriptor = -1

    def _still_owns(self, descriptor: int) -> bool:
        try:
            state = os.fstat(descriptor)
            position = os.lseek(descriptor, 0, os.SEEK_CUR)
        except OSError:
            return False
        identity = getattr(self, "identity", None)
        if identity is None:
            return True
        return (
            state.st_dev,
            state.st_ino,
            stat.S_IFMT(state.st_mode),
            position,
        ) == identity

    def _reconcile_pending(self) -> None:
        pending = int(getattr(self, "_pending_descriptor", -1))
        if pending < 0:
            return
        if self._still_owns(pending):
            self.value = pending
            self._pending_descriptor = -1
        else:
            self._pending_descriptor = -1
            self.value = -1

    def close(self) -> None:
        self._reconcile_pending()
        descriptor = int(self.value)
        if descriptor < 0:
            return
        self._pending_descriptor = descriptor
        self.value = -1
        os.close(descriptor)
        self._pending_descriptor = -1

    def __del__(self) -> None:
        for _attempt in range(2):
            try:
                self.close()
            except BaseException:
                continue
            if self.closed:
                return


_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC_OPENAT = _LIBC.openat
_LIBC_OPENAT.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int)
_LIBC_OPENAT.restype = _OwnedDirectoryDescriptor


def _open_owned_directory(path, *, directory_fd=None):
    base = _AT_FDCWD if directory_fd is None else int(directory_fd)
    owner = _LIBC_OPENAT(
        base, os.fsencode(os.fspath(path)), _DIRECTORY_OPEN_FLAGS
    )
    if owner.value < 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number, os.strerror(error_number), os.fspath(path)
        )
    try:
        owner.remember_identity(os.fstat(owner.fd))
        return owner
    except BaseException:
        owner.close()
        raise


class _DirectoryLease:
    """Own and verify one directly opened, no-follow directory fd."""

    def __init__(
        self, path: Path, *, message: str, directory_fd: int | None = None
    ) -> None:
        self.path = Path(path)
        self.message = message
        self.directory_fd = directory_fd
        self.owner = None
        self.fd = None
        self.identity = None

    def _entry_state(self):
        if self.directory_fd is None:
            return os.stat(self.path, follow_symlinks=False)
        name = os.fspath(self.path)
        if not name or Path(name).name != name:
            raise ValueError(self.message)
        return os.stat(
            name, dir_fd=self.directory_fd, follow_symlinks=False
        )

    def __enter__(self):
        owner = None
        try:
            before = self._entry_state()
            if not stat.S_ISDIR(before.st_mode):
                raise ValueError(self.message)
            identity = (before.st_dev, before.st_ino)
            owner = _open_owned_directory(
                self.path, directory_fd=self.directory_fd
            )
            self.owner = owner
            descriptor = owner.fd
            opened = os.fstat(descriptor)
            current = self._entry_state()
            if (
                not stat.S_ISDIR(opened.st_mode)
                or not stat.S_ISDIR(current.st_mode)
                or (opened.st_dev, opened.st_ino) != identity
                or (current.st_dev, current.st_ino) != identity
            ):
                raise ValueError(self.message)
            self.fd = descriptor
            self.identity = identity
            return self
        except OSError as error:
            if owner is not None:
                owner.close()
            self.owner = None
            self.fd = None
            raise ValueError(self.message) from error
        except BaseException:
            if owner is not None:
                owner.close()
            self.owner = None
            self.fd = None
            raise

    def __exit__(self, _type, _value, _traceback):
        owner = self.owner
        if owner is None:
            return
        try:
            owner.close()
        finally:
            if owner.closed:
                self.owner = None
                self.fd = None

    def verify_path(self) -> None:
        try:
            state = self._entry_state()
        except OSError as error:
            raise ValueError(self.message) from error
        if (
            not stat.S_ISDIR(state.st_mode)
            or (state.st_dev, state.st_ino) != self.identity
        ):
            raise ValueError(self.message)

    def verify_entry(self, parent_fd: int, name: str) -> None:
        try:
            state = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as error:
            raise ValueError(self.message) from error
        if (
            not stat.S_ISDIR(state.st_mode)
            or (state.st_dev, state.st_ino) != self.identity
        ):
            raise ValueError(self.message)


def _directory_entries(lease: _DirectoryLease) -> tuple[set[str], set[str]]:
    files = set()
    directories = set()
    with os.scandir(f"/proc/self/fd/{lease.fd}") as entries:
        for entry in entries:
            state = entry.stat(follow_symlinks=False)
            if stat.S_ISREG(state.st_mode):
                files.add(entry.name)
            elif stat.S_ISDIR(state.st_mode):
                directories.add(entry.name)
            else:
                raise ValueError("report bundle contains an unsafe entry")
    return files, directories


def _open_bundle(root: Path, *, message: str):
    return _DirectoryLease(root, message=message)


def _bundle_bytes(root: Path, *, message: str) -> dict[str, bytes]:
    with _open_bundle(root, message=message) as root_lease:
        root_files = {
            relative for relative in REPORT_FILES if "/" not in relative
        }
        if _directory_entries(root_lease) != (root_files, {"figures"}):
            raise ValueError(message)
        with _DirectoryLease(
            "figures", message=message, directory_fd=root_lease.fd
        ) as figures_lease:
            figure_files = {
                relative.split("/", 1)[1]
                for relative in REPORT_FILES
                if relative.startswith("figures/")
            }
            if _directory_entries(figures_lease) != (figure_files, set()):
                raise ValueError(message)
            content = {}
            for relative in REPORT_FILES:
                if relative.startswith("figures/"):
                    directory_fd = figures_lease.fd
                    name = relative.split("/", 1)[1]
                else:
                    directory_fd = root_lease.fd
                    name = relative
                with _open_regular_file(
                    name,
                    directory_fd=directory_fd,
                    error_message=message,
                ) as handle:
                    content[relative] = handle.read()
                    opened = os.fstat(handle.fileno())
                    visible = os.stat(
                        name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                    if (
                        not stat.S_ISREG(opened.st_mode)
                        or not stat.S_ISREG(visible.st_mode)
                        or (opened.st_dev, opened.st_ino)
                        != (visible.st_dev, visible.st_ino)
                    ):
                        raise ValueError(message)
            figures_lease.verify_entry(root_lease.fd, "figures")
        root_lease.verify_path()
        return content


def _bundle_is_exact(root: Path) -> bool:
    try:
        message = "report bundle is incomplete"
        with _open_bundle(root, message=message) as root_lease:
            root_files = {
                relative for relative in REPORT_FILES if "/" not in relative
            }
            if _directory_entries(root_lease) != (root_files, {"figures"}):
                return False
            with _DirectoryLease(
                "figures", message=message, directory_fd=root_lease.fd
            ) as figures_lease:
                figure_files = {
                    relative.split("/", 1)[1]
                    for relative in REPORT_FILES
                    if relative.startswith("figures/")
                }
                if _directory_entries(figures_lease) != (figure_files, set()):
                    return False
                figures_lease.verify_entry(root_lease.fd, "figures")
            root_lease.verify_path()
    except (OSError, ValueError):
        return False
    return True


def _same_bundle(existing: Path, intended: Path) -> bool:
    message = "existing report bundle differs"
    try:
        existing_content = _bundle_bytes(existing, message=message)
        intended_content = _bundle_bytes(intended, message=message)
    except (OSError, ValueError):
        return False
    return existing_content == intended_content


def _remove_directory_contents(directory_fd: int) -> None:
    visible = Path(f"/proc/self/fd/{directory_fd}")
    with os.scandir(visible) as entries:
        names = [entry.name for entry in entries]
    for name in names:
        try:
            state = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if stat.S_ISDIR(state.st_mode):
            with _DirectoryLease(
                name,
                message="report staging directory changed",
                directory_fd=directory_fd,
            ) as child:
                _remove_directory_contents(child.fd)
                child.verify_entry(directory_fd, name)
            os.rmdir(name, dir_fd=directory_fd)
        else:
            os.unlink(name, dir_fd=directory_fd)


def _find_owned_directory_name(
    parent_fd: int, identity: tuple[int, int]
) -> str | None:
    with os.scandir(f"/proc/self/fd/{parent_fd}") as entries:
        for entry in entries:
            try:
                state = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if (
                stat.S_ISDIR(state.st_mode)
                and (state.st_dev, state.st_ino) == identity
            ):
                return entry.name
    return None


class _StagingOwner:
    def __init__(self, parent: _DirectoryLease, *, prefix: str) -> None:
        self.parent = parent
        self.prefix = prefix
        self._temporary = None
        self.lease = None
        self.name = None
        self.path = None
        self.published = False
        self.final_name = None

    def __enter__(self):
        parent_path = f"/proc/self/fd/{self.parent.fd}"
        # TemporaryDirectory owns its path before returning to this frame.  If
        # cancellation lands at CALL -> STORE_ATTR, its finalizer removes it.
        self._temporary = tempfile.TemporaryDirectory(
            prefix=self.prefix,
            dir=parent_path,
        )
        try:
            self.name = Path(self._temporary.name).name
            self.path = Path(parent_path) / self.name
            self.lease = _DirectoryLease(
                self.name,
                message="report staging directory changed",
                directory_fd=self.parent.fd,
            )
            self.lease.__enter__()
            return self
        except BaseException:
            self._temporary.cleanup()
            raise

    def detach_temporary_finalizer(self) -> None:
        self._temporary._finalizer.detach()

    def verify(self) -> None:
        self.lease.verify_entry(self.parent.fd, self.name)

    def prepare_publish(self, output_name: str) -> None:
        if not output_name or Path(output_name).name != output_name:
            raise ValueError("invalid final report directory name")
        self.final_name = output_name

    def mark_published(self, output_name: str) -> None:
        self.lease.verify_entry(self.parent.fd, output_name)
        self.published = True

    def __exit__(self, error_type, _error, _traceback):
        cleanup_error = None
        try:
            if not self.published:
                try:
                    _cleanup_staging(self)
                except BaseException as caught:
                    cleanup_error = caught
        finally:
            if self.lease is not None:
                self.lease.__exit__(None, None, None)
            temporary = self._temporary
            if temporary is not None:
                temporary._finalizer.detach()
        if error_type is None and cleanup_error is not None:
            raise cleanup_error
        return False


def _cleanup_staging(staging: _StagingOwner) -> None:
    lease = staging.lease
    owned_name = _find_owned_directory_name(
        staging.parent.fd, lease.identity
    )
    if owned_name is None or owned_name == staging.final_name:
        return
    _remove_directory_contents(lease.fd)
    owned_name = _find_owned_directory_name(
        staging.parent.fd, lease.identity
    )
    if owned_name is not None and owned_name != staging.final_name:
        os.rmdir(owned_name, dir_fd=staging.parent.fd)


def _publish_no_replace(
    staging_name: str, output_name: str, parent_fd: int
) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace report publication needs renameat2")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_fd,
        os.fsencode(staging_name),
        parent_fd,
        os.fsencode(output_name),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(output_name)
    raise OSError(error_number, os.strerror(error_number), output_name)


def _write_bundle(
    staging: Path,
    frame: pd.DataFrame,
    evaluation: Mapping,
    provenance: Mapping,
) -> None:
    figures = staging / "figures"
    figures.mkdir()

    frame.to_csv(
        staging / "per-image-scores.csv",
        index=False,
        lineterminator="\n",
    )
    pd.DataFrame(_metric_rows(evaluation)).to_csv(
        staging / "metrics.csv",
        index=False,
        lineterminator="\n",
    )
    pd.DataFrame(evaluation["bootstrap_comparisons"]).to_csv(
        staging / "bootstrap-comparisons.csv",
        index=False,
        lineterminator="\n",
    )
    summary = {
        "schema_version": 1,
        "provenance": provenance,
        "evaluation": evaluation,
    }
    (staging / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_figures(frame, evaluation, figures, provenance["config"])
    (staging / "report.md").write_text(
        render_report(frame, evaluation, provenance),
        encoding="utf-8",
    )
    if not _bundle_is_exact(staging):
        raise RuntimeError("report bundle is incomplete")


def _validated_report_inputs(rows, evaluation, provenance):
    normalized_provenance = _validate_provenance(provenance)
    frame = _validated_frame(
        rows,
        config=normalized_provenance["config"],
    )
    canonical_evaluation = _validate_evaluation(
        evaluation, frame, normalized_provenance
    )
    return normalized_provenance, frame, canonical_evaluation


def write_report(
    output,
    rows,
    evaluation,
    provenance,
    *,
    parent=None,
    _expected_content=None,
) -> None:
    """Atomically publish a report on Linux with procfs and renameat2.

    Directory identity pinning requires ``/proc/self/fd`` and publication
    requires the Linux ``renameat2(RENAME_NOREPLACE)`` contract.
    """
    normalized_provenance, frame, canonical_evaluation = (
        _validated_report_inputs(rows, evaluation, provenance)
    )
    if _expected_content is not None and (
        type(_expected_content) is not dict or _expected_content
    ):
        raise ValueError("expected report content capture must be an empty dict")
    if parent is None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output_name = output.name
        parent_context = _DirectoryLease(
            output.parent,
            message="report parent directory changed",
        )
    else:
        output_name = os.fspath(output)
        if (
            not isinstance(parent, _DirectoryLease)
            or parent.fd is None
            or not output_name
            or Path(output_name).name != output_name
        ):
            raise ValueError("report parent directory lease is not active")
        parent_context = nullcontext(parent)
    with parent_context as parent:
        with _StagingOwner(
            parent, prefix=f".{output_name}.staging-"
        ) as staging:
            staging.detach_temporary_finalizer()
            _write_bundle(
                staging.path,
                frame,
                canonical_evaluation,
                normalized_provenance,
            )
            if _expected_content is not None:
                _expected_content.update(
                    _bundle_bytes(
                        staging.path, message="intended report bundle changed"
                    )
                )
            parent.verify_path()
            staging.verify()
            anchored_output = (
                Path(f"/proc/self/fd/{parent.fd}") / output_name
            )
            try:
                os.stat(
                    output_name,
                    dir_fd=parent.fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                output_exists = False
            else:
                output_exists = True
            if output_exists:
                if _same_bundle(anchored_output, staging.path):
                    parent.verify_path()
                    return
                raise ValueError(
                    "existing report bundle differs; choose a new output "
                    "directory"
                )
            staging.prepare_publish(output_name)
            try:
                _publish_no_replace(staging.name, output_name, parent.fd)
            except FileExistsError:
                if _same_bundle(anchored_output, staging.path):
                    parent.verify_path()
                    return
                raise ValueError(
                    "existing report bundle differs; choose a new output "
                    "directory"
                ) from None
            staging.mark_published(output_name)
            parent.verify_path()
