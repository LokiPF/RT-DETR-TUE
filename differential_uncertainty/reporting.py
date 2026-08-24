from __future__ import annotations

import ctypes
import errno
import json
import math
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import SEVERITIES, summarize_series


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
    checkpoint = normalized.get("checkpoint_sha256")
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("provenance needs a nonempty checkpoint_sha256")

    config = normalized.get("config")
    if not isinstance(config, dict):
        raise ValueError("provenance needs a config mapping")
    for key in ("query_count", "k", "persistence_layer", "bootstrap_samples"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"provenance config {key} must be a positive integer")

    corruption = normalized.get("corruption")
    if not isinstance(corruption, dict):
        raise ValueError("provenance needs a corruption mapping")
    name = corruption.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("provenance corruption needs a nonempty name")
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
    return normalized


def _expected_gap(reference: float, responsive: float) -> float:
    total = reference + responsive
    return 0.0 if total == 0.0 else 2.0 * (responsive - reference) / total


def _validated_frame(rows, *, query_count: int) -> pd.DataFrame:
    rows = list(rows)
    if not rows:
        raise ValueError("report rows need at least one complete image")
    checked = []
    seen: set[tuple[str, int]] = set()
    by_image: dict[str, set[int]] = {}
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
        image_id = row["image_id"]
        if not isinstance(image_id, str) or not image_id.strip():
            raise ValueError("score row image_id must be a nonempty string")
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
                raise ValueError(f"score row {field} must be a non-negative integer")
            value = int(value)
            if value < 0:
                raise ValueError(f"score row {field} must be a non-negative integer")
            row[field] = value
        if row["padded_count"] + row["valid_count"] != query_count:
            raise ValueError("padded_count plus valid_count must equal query_count")
        if row["reference_count"] <= 0 or row["responsive_count"] <= 0:
            raise ValueError("reference and responsive groups must be nonempty")
        if (
            row["reference_count"] > row["valid_count"]
            or row["responsive_count"] > row["valid_count"]
        ):
            raise ValueError("group counts cannot exceed valid_count")

        for field in _SCORE_FIELDS:
            value = row[field]
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"score row {field} must be a finite real number")
            row[field] = float(value)
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

    return (
        pd.DataFrame(checked, columns=_SCORE_COLUMNS)
        .sort_values(["image_id", "severity"], kind="stable")
        .reset_index(drop=True)
    )


def _validate_evaluation(evaluation, frame: pd.DataFrame, provenance: dict) -> dict:
    if not isinstance(evaluation, Mapping):
        raise ValueError("evaluation must be a mapping")
    _require_finite_tree(evaluation, name="evaluation")
    normalized = _json_copy(evaluation, name="evaluation")
    if set(evaluation) != {"series", "bootstrap_comparisons"}:
        raise ValueError("evaluation must keep the exact Task 8 structure")
    series = evaluation["series"]
    if not isinstance(series, Mapping) or set(series) != set(_SERIES):
        raise ValueError("evaluation series must contain the four fixed scores")

    score_rows = frame.to_dict(orient="records")
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
            raise ValueError(f"evaluation series {name} has an invalid orientation")
        expected = summarize_series(score_rows, name, int(orientation))
        if _json_copy(supplied, name="evaluation") != _json_copy(
            expected, name="expected evaluation"
        ):
            raise ValueError(
                f"evaluation series {name} does not match supplied rows"
            )

    comparisons = evaluation["bootstrap_comparisons"]
    if not isinstance(comparisons, list) or len(comparisons) != len(_CONTROLS):
        raise ValueError("evaluation needs the three fixed bootstrap comparisons")
    config = provenance["config"]
    for index, (item, control) in enumerate(zip(comparisons, _CONTROLS)):
        if not isinstance(item, Mapping):
            raise ValueError(f"bootstrap comparison {index} must be a mapping")
        if item.get("candidate") != "persistence_relative_gap":
            raise ValueError("bootstrap candidate must be persistence_relative_gap")
        if item.get("control") != control:
            raise ValueError("bootstrap controls must keep their fixed order")
        for orientation_key, series_name in (
            ("candidate_orientation", "persistence_relative_gap"),
            ("control_orientation", control),
        ):
            if item.get(orientation_key) != series[series_name]["orientation"]:
                raise ValueError("bootstrap orientation does not match its series")
        for key in ("point_difference", "ci_low", "ci_high"):
            value = item.get(key)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"bootstrap {key} must be finite")
        if float(item["ci_low"]) > float(item["ci_high"]):
            raise ValueError("bootstrap interval bounds are reversed")
        expected_point = (
            float(series["persistence_relative_gap"]["macro_auroc"])
            - float(series[control]["macro_auroc"])
        )
        if not math.isclose(
            float(item["point_difference"]), expected_point,
            rel_tol=1e-12, abs_tol=1e-12,
        ):
            raise ValueError("bootstrap point difference does not match series")
        if item.get("samples") != config["bootstrap_samples"]:
            raise ValueError("bootstrap sample count does not match provenance")
        seed = item.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("bootstrap seed must be a non-negative integer")
    return normalized


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


def _write_figures(
    frame: pd.DataFrame, evaluation: Mapping, directory: Path
) -> None:
    severity = np.asarray(SEVERITIES)

    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.5), constrained_layout=True
    )
    for axis, prefix, title in (
        (axes[0], "persistence", "Layer-2 distance"),
        (axes[1], "confidence", "1 - maximum sigmoid confidence"),
    ):
        axis.plot(
            severity,
            _median_curve(frame, f"{prefix}_reference"),
            marker="o",
            label="90-100% reference",
        )
        axis.plot(
            severity,
            _median_curve(frame, f"{prefix}_responsive"),
            marker="o",
            label="50-60% responsive",
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
    corruption_name = provenance["corruption"]["name"].replace("_", " ")
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
and levels 1 through 5 add more corruption. The detector made
{config['query_count']} query guesses for every version of each image. Sometimes
a detector fills unused spaces by repeating its last guess. We removed those
exact repeated, padded guesses before doing any calculation.

For each image, we ranked the remaining guesses by their maximum **sigmoid**
class confidence. We compared two fixed groups: the middle 50-60% group, called
the responsive group, and the highest 90-100% group, called the reference
group. Both groups use the detector fingerprint from decoder layer
{config['persistence_layer']}.

For every selected query, we found its **{config['k']} nearest clean** bank
fingerprints and averaged their distances. Then we averaged those query
distances inside each group. Here is a small nearest-neighbor example. If a
query's five nearest distances are 0.10, 0.14, 0.17, 0.21, and 0.28, its score
is:

    (0.10 + 0.14 + 0.17 + 0.21 + 0.28) / 5 = 0.18

A larger distance means that the fingerprint looks less like the saved clean
fingerprints.

## How the persistence relative gap was calculated

For image `{example['image_id']}` at corruption level
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
from earlier blur tuning. Fresh numbers may differ from the historical run
because this clean workflow also removes padded queries from the reference
bank.

The complete per-image rows, metric tables, bootstrap comparisons, figures, and
reproducibility details are stored beside this report. Checkpoint SHA-256:
`{provenance['checkpoint_sha256']}`.
"""


def _observed_bundle(root: Path) -> tuple[set[str], set[str]] | None:
    try:
        metadata = root.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or root.is_symlink():
            return None
        files: set[str] = set()
        directories: set[str] = set()
        for path in root.rglob("*"):
            relative = str(path.relative_to(root))
            entry = path.lstat()
            if stat.S_ISLNK(entry.st_mode):
                return None
            if stat.S_ISDIR(entry.st_mode):
                directories.add(relative)
            elif stat.S_ISREG(entry.st_mode):
                files.add(relative)
            else:
                return None
        return files, directories
    except OSError:
        return None


def _same_bundle(existing: Path, intended: Path) -> bool:
    existing_entries = _observed_bundle(existing)
    intended_entries = _observed_bundle(intended)
    expected = (set(REPORT_FILES), {"figures"})
    if existing_entries != expected or intended_entries != expected:
        return False
    try:
        return all(
            (existing / relative).read_bytes()
            == (intended / relative).read_bytes()
            for relative in REPORT_FILES
        )
    except OSError:
        return False


def _cleanup_staging(staging: Path) -> None:
    try:
        metadata = staging.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
        shutil.rmtree(staging)
    else:
        staging.unlink()


def _publish_no_replace(staging: Path, output: Path) -> None:
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
        -100,
        os.fsencode(staging),
        -100,
        os.fsencode(output),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(output)
    raise OSError(error_number, os.strerror(error_number), output)


def _write_bundle(
    staging: Path,
    frame: pd.DataFrame,
    evaluation: Mapping,
    normalized_evaluation: Mapping,
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
        "evaluation": normalized_evaluation,
    }
    (staging / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_figures(frame, evaluation, figures)
    (staging / "report.md").write_text(
        render_report(frame, evaluation, provenance),
        encoding="utf-8",
    )
    if _observed_bundle(staging) != (set(REPORT_FILES), {"figures"}):
        raise RuntimeError("report bundle is incomplete")


def write_report(output, rows, evaluation, provenance) -> None:
    normalized_provenance = _validate_provenance(provenance)
    frame = _validated_frame(
        rows,
        query_count=normalized_provenance["config"]["query_count"],
    )
    normalized_evaluation = _validate_evaluation(
        evaluation, frame, normalized_provenance
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.staging-",
        dir=output.parent,
    ))
    try:
        _write_bundle(
            staging,
            frame,
            evaluation,
            normalized_evaluation,
            normalized_provenance,
        )
        if os.path.lexists(output):
            if _same_bundle(output, staging):
                return
            raise ValueError(
                "existing report bundle differs; choose a new output directory"
            )
        try:
            _publish_no_replace(staging, output)
        except FileExistsError:
            if _same_bundle(output, staging):
                return
            raise ValueError(
                "existing report bundle differs; choose a new output directory"
            ) from None
    finally:
        _cleanup_staging(staging)
