from __future__ import annotations

from collections import defaultdict
from numbers import Integral, Real

import numpy as np
from scipy.stats import rankdata, spearmanr

from .config import ExperimentConfig, FIXED_CONFIG


_BOOTSTRAP_ROW_CHUNK_SIZE = 256
SEVERITIES = tuple(range(6))


def _require_orientation(orientation: int) -> int:
    if isinstance(orientation, bool) or not isinstance(orientation, Integral):
        raise ValueError("orientation must be an integer equal to -1 or +1")
    orientation = int(orientation)
    if orientation not in (-1, 1):
        raise ValueError("orientation must be -1 or +1")
    return orientation


def trend_metrics(scores) -> dict:
    values = np.asarray(list(scores), dtype=float)
    if values.shape != (6,) or not bool(np.isfinite(values).all()):
        raise ValueError("trend metrics need exactly six finite scores")
    signed = (
        0.0
        if bool(np.all(values == values[0]))
        else float(spearmanr(np.arange(6), values).statistic)
    )
    return {
        "signed_spearman": signed,
        "absolute_spearman": abs(signed),
        "direction": (
            "increasing" if signed > 0 else "decreasing" if signed < 0 else "flat"
        ),
    }


def oriented_curve_metrics(scores, orientation: int) -> dict:
    orientation = _require_orientation(orientation)
    values = np.asarray(list(scores), dtype=float)
    if values.shape != (6,) or not bool(np.isfinite(values).all()):
        raise ValueError("curve metrics need exactly six finite scores")
    oriented = values * orientation
    return {
        "adjacent_consistency": float(np.mean(np.diff(oriented) >= 0)),
        "strongest_blur_above_clean": bool(oriented[5] > oriented[0]),
    }


def binary_auroc(clean_scores, corrupted_scores, *, orientation: int) -> float:
    orientation = _require_orientation(orientation)
    clean = np.asarray(clean_scores, dtype=float) * orientation
    corrupted = np.asarray(corrupted_scores, dtype=float) * orientation
    if (
        not clean.size
        or not corrupted.size
        or not bool(np.isfinite(clean).all())
        or not bool(np.isfinite(corrupted).all())
    ):
        raise ValueError("AUROC needs non-empty finite clean and corrupted groups")
    ranks = rankdata(np.concatenate((clean, corrupted)), method="average")
    positives = corrupted.size
    rank_sum = float(ranks[clean.size :].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (
        clean.size * positives
    )


def _curves(rows, field: str) -> dict[str, dict[int, float]]:
    curves: dict[str, dict[int, float]] = defaultdict(dict)
    required = ("image_id", "severity", field)
    for row_index, row in enumerate(rows):
        for key in required:
            if key not in row:
                raise ValueError(
                    f"score row {row_index} is missing required row key {key!r}"
                )
        image_id = row["image_id"]
        if not isinstance(image_id, str) or not image_id.strip():
            raise ValueError("score row image_id must be a nonempty string")
        raw_severity = row["severity"]
        if isinstance(raw_severity, bool) or not isinstance(raw_severity, Integral):
            raise ValueError(
                f"score row for {image_id} severity must be an integer"
            )
        severity = int(raw_severity)
        if severity in curves[image_id]:
            raise ValueError(f"duplicate score row for {image_id} severity {severity}")
        raw_value = row[field]
        if (
            isinstance(raw_value, (bool, np.bool_))
            or not isinstance(raw_value, Real)
        ):
            raise ValueError(
                f"score row for {image_id} severity {severity} score must be a real number"
            )
        value = float(raw_value)
        if not np.isfinite(value):
            raise ValueError(
                f"score row for {image_id} severity {severity} needs a finite score"
            )
        curves[image_id][severity] = value
    for image_id, curve in curves.items():
        if set(curve) != set(SEVERITIES):
            raise ValueError(f"image {image_id} does not have severities 0 through 5")
    return dict(curves)


def summarize_series(rows, field: str, orientation: int) -> dict:
    curves = _curves(rows, field)
    if not curves:
        raise ValueError("series summary needs at least one complete image")
    orientation = _require_orientation(orientation)
    ordered = sorted(curves)
    trends = [trend_metrics([curves[i][s] for s in SEVERITIES]) for i in ordered]
    oriented = [
        oriented_curve_metrics([curves[i][s] for s in SEVERITIES], orientation)
        for i in ordered
    ]
    clean = [curves[i][0] for i in ordered]
    aurocs = {
        severity: binary_auroc(
            clean,
            [curves[i][severity] for i in ordered],
            orientation=orientation,
        )
        for severity in SEVERITIES[1:]
    }
    severity_statistics = {}
    for severity in SEVERITIES:
        values = np.asarray([curves[i][severity] for i in ordered])
        q25, median, q75 = np.percentile(values, [25, 50, 75])
        severity_statistics[severity] = {
            "count": int(values.size),
            "mean": float(values.mean()),
            "median": float(median),
            "q25": float(q25),
            "q75": float(q75),
        }
    signed = [item["signed_spearman"] for item in trends]
    absolute = [item["absolute_spearman"] for item in trends]
    directions = [item["direction"] for item in trends]
    return {
        "field": field,
        "orientation": orientation,
        "image_count": len(ordered),
        "median_signed_spearman": float(np.median(signed)),
        "median_absolute_spearman": float(np.median(absolute)),
        "increasing_count": directions.count("increasing"),
        "decreasing_count": directions.count("decreasing"),
        "flat_count": directions.count("flat"),
        "oriented_adjacent_consistency": float(
            np.mean([item["adjacent_consistency"] for item in oriented])
        ),
        "strongest_corruption_above_clean_rate": float(
            np.mean([item["strongest_blur_above_clean"] for item in oriented])
        ),
        "auroc_by_severity": aurocs,
        "macro_auroc": float(np.mean(list(aurocs.values()))),
        "severity_statistics": severity_statistics,
    }


def _arrays(rows, field):
    curves = _curves(rows, field)
    ordered = sorted(curves)
    return ordered, {
        severity: np.asarray([curves[image_id][severity] for image_id in ordered])
        for severity in SEVERITIES
    }


def _bootstrap_macro(columns, draws, orientation):
    count = draws.shape[1]
    clean = orientation * columns[0][draws]
    total = np.zeros(draws.shape[0], dtype=float)
    for severity in SEVERITIES[1:]:
        corrupted = orientation * columns[severity][draws]
        ranks = rankdata(
            np.concatenate((clean, corrupted), axis=1),
            method="average",
            axis=1,
        )
        positive_rank_sum = ranks[:, count:].sum(axis=1)
        total += (positive_rank_sum - count * (count + 1) / 2) / (count * count)
    return total / 5


def paired_macro_bootstrap(
    rows,
    candidate_field: str,
    control_field: str,
    *,
    candidate_orientation: int,
    control_orientation: int,
    samples: int,
    seed: int,
) -> dict:
    candidate_orientation = _require_orientation(candidate_orientation)
    control_orientation = _require_orientation(control_orientation)
    if isinstance(samples, bool) or not isinstance(samples, Integral) or samples <= 0:
        raise ValueError("bootstrap sample count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("bootstrap seed must be an integer")
    if seed < 0:
        raise ValueError("bootstrap seed must be non-negative")
    samples = int(samples)
    seed = int(seed)
    rows = list(rows)
    if any(
        candidate_field not in row or control_field not in row for row in rows
    ):
        raise ValueError("paired bootstrap needs candidate and control on every row")
    candidate_ids, candidate = _arrays(rows, candidate_field)
    control_ids, control = _arrays(rows, control_field)
    if candidate_ids != control_ids:
        raise ValueError("candidate and control must cover the same image identities")
    count = len(candidate_ids)
    if count == 0 or samples <= 0:
        raise ValueError("bootstrap needs images and a positive sample count")
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, count, size=(samples, count))
    differences = np.empty(samples, dtype=float)
    for start in range(0, samples, _BOOTSTRAP_ROW_CHUNK_SIZE):
        stop = min(start + _BOOTSTRAP_ROW_CHUNK_SIZE, samples)
        batch = draws[start:stop]
        differences[start:stop] = _bootstrap_macro(
            candidate, batch, candidate_orientation
        ) - _bootstrap_macro(control, batch, control_orientation)
    identity = np.arange(count, dtype=int)[None, :]
    point = float(
        _bootstrap_macro(candidate, identity, candidate_orientation)[0]
        - _bootstrap_macro(control, identity, control_orientation)[0]
    )
    low, high = np.percentile(differences, [2.5, 97.5])
    return {
        "candidate": candidate_field,
        "control": control_field,
        "candidate_orientation": candidate_orientation,
        "control_orientation": control_orientation,
        "point_difference": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "samples": samples,
        "seed": seed,
    }


def evaluate_rows(rows, config: ExperimentConfig = FIXED_CONFIG) -> dict:
    rows = list(rows)
    orientations = {
        "persistence_relative_gap": config.persistence_orientation,
        "confidence_relative_gap": config.confidence_orientation,
        "persistence_responsive": config.raw_responsive_orientation,
        "persistence_reference": config.raw_reference_orientation,
    }
    series = {
        field: summarize_series(rows, field, orientation)
        for field, orientation in orientations.items()
    }
    comparisons = [
        paired_macro_bootstrap(
            rows,
            "persistence_relative_gap",
            control,
            candidate_orientation=config.persistence_orientation,
            control_orientation=orientations[control],
            samples=config.bootstrap_samples,
            seed=config.bootstrap_seed,
        )
        for control in (
            "confidence_relative_gap",
            "persistence_responsive",
            "persistence_reference",
        )
    ]
    return {"series": series, "bootstrap_comparisons": comparisons}
