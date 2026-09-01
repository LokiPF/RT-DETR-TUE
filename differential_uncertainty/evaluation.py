from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import numpy as np
from scipy.stats import rankdata


_METHODS = ("fingerprint", "confidence", "entropy")
_SEVERITIES = (0, 4, 5)
_SEVERITY_INDEX = {severity: index for index, severity in enumerate(_SEVERITIES)}
_ROW_KEYS = frozenset(("image_id", "corruption", "severity", *_METHODS))
_BOOTSTRAP_BATCH_SIZE = 16


def _score_vector(values, *, name: str) -> np.ndarray:
    try:
        source = values if isinstance(values, np.ndarray) else list(values)
        array = np.asarray(source, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a non-empty finite one-dimensional input") from error
    if array.ndim != 1 or not array.size or not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} must be a non-empty finite one-dimensional input")
    return array


def binary_auroc(clean, corrupted) -> float:
    """Return tie-correct AUROC, with larger values indicating corruption."""
    clean_values = _score_vector(clean, name="clean scores")
    corrupted_values = _score_vector(corrupted, name="corrupted scores")
    ranks = rankdata(np.concatenate((clean_values, corrupted_values)), method="average")
    positives = corrupted_values.size
    rank_sum = float(ranks[clean_values.size :].sum())
    return float(
        (rank_sum - positives * (positives + 1) / 2)
        / (clean_values.size * positives)
    )


def _validate_families(families) -> tuple[str, ...]:
    if isinstance(families, str):
        raise ValueError("families must be a non-empty sequence of unique strings")
    try:
        ordered = tuple(families)
    except TypeError as error:
        raise ValueError("families must be a non-empty sequence of unique strings") from error
    if (
        not ordered
        or len(set(ordered)) != len(ordered)
        or any(not isinstance(family, str) or not family.strip() for family in ordered)
    ):
        raise ValueError("families must be a non-empty sequence of unique strings")
    return ordered


def _validate_bootstrap_options(samples, seed) -> tuple[int, int]:
    if isinstance(samples, bool) or not isinstance(samples, Integral) or samples <= 0:
        raise ValueError("samples must be a positive non-boolean integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a non-negative non-boolean integer")
    return int(samples), int(seed)


def _panel_arrays(rows, families) -> dict[str, np.ndarray]:
    """Validate score rows and make method arrays shaped family/severity/image."""
    ordered_families = _validate_families(families)
    by_family: dict[str, dict[str, dict[int, dict]]] = {
        family: {} for family in ordered_families
    }
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != _ROW_KEYS:
            raise ValueError(f"score row {index} must have exactly the required keys")
        image_id = row["image_id"]
        family = row["corruption"]
        severity = row["severity"]
        if not isinstance(image_id, str) or not image_id.strip():
            raise ValueError("image_id must be a non-empty string")
        if not isinstance(family, str) or family not in by_family:
            raise ValueError("score row corruption must be one of the requested families")
        if isinstance(severity, bool) or not isinstance(severity, Integral) or severity not in _SEVERITY_INDEX:
            raise ValueError("score row severity must be exactly 0, 4, or 5")
        severity = int(severity)
        for method in _METHODS:
            value = row[method]
            if isinstance(value, bool) or not isinstance(value, (float, np.floating)):
                raise ValueError(f"score row {method} must be a finite float")
            if not np.isfinite(value):
                raise ValueError(f"score row {method} must be a finite float")
        image_rows = by_family[family].setdefault(image_id, {})
        if severity in image_rows:
            raise ValueError("duplicate family/image/severity score row")
        image_rows[severity] = row

    rosters = []
    for family in ordered_families:
        family_rows = by_family[family]
        if not family_rows:
            raise ValueError("every requested family needs a non-empty image roster")
        for image_id, severity_rows in family_rows.items():
            if set(severity_rows) != set(_SEVERITIES):
                raise ValueError(
                    f"family {family!r} image {image_id!r} must have exactly severities 0, 4, and 5"
                )
        rosters.append(set(family_rows))
    if any(roster != rosters[0] for roster in rosters[1:]):
        raise ValueError("all families must share the same image roster")

    image_ids = tuple(sorted(rosters[0]))
    return {
        method: np.asarray(
            [
                [
                    [float(by_family[family][image_id][severity][method]) for image_id in image_ids]
                    for severity in _SEVERITIES
                ]
                for family in ordered_families
            ],
            dtype=float,
        )
        for method in _METHODS
    }


def _macro_aurocs_for_draws(panel: dict[str, np.ndarray], draws: np.ndarray, method: str) -> np.ndarray:
    """Compute equal-weight task macro AUROCs for one shared draw per row."""
    columns = panel[method]
    image_count = draws.shape[1]
    clean = columns[:, 0, :][:, draws]
    total = np.zeros(draws.shape[0], dtype=float)
    for severity_index in (1, 2):
        corrupted = columns[:, severity_index, :][:, draws]
        ranks = rankdata(np.concatenate((clean, corrupted), axis=-1), method="average", axis=-1)
        rank_sum = ranks[..., image_count:].sum(axis=-1)
        aurocs = (rank_sum - image_count * (image_count + 1) / 2) / (image_count * image_count)
        total += aurocs.sum(axis=0)
    return total / (columns.shape[0] * 2)


def _paired_macro_differences(panel: dict[str, np.ndarray], draws: np.ndarray, left: str, right: str) -> np.ndarray:
    return _macro_aurocs_for_draws(panel, draws, left) - _macro_aurocs_for_draws(panel, draws, right)


def _comparison(panel, left: str, right: str, *, samples: int, seed: int, point: float) -> dict:
    image_count = panel[left].shape[-1]
    generator = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=float)
    for start in range(0, samples, _BOOTSTRAP_BATCH_SIZE):
        stop = min(start + _BOOTSTRAP_BATCH_SIZE, samples)
        draws = generator.integers(0, image_count, size=(stop - start, image_count))
        differences[start:stop] = _paired_macro_differences(panel, draws, left, right)
    low, high = np.percentile(differences, (2.5, 97.5))
    return {"point": float(point), "low": float(low), "high": float(high)}


def evaluate_scores(rows, families, *, samples, seed) -> dict:
    """Evaluate the complete fixed clean/L4/L5 corruption evidence panel."""
    ordered_families = _validate_families(families)
    samples, seed = _validate_bootstrap_options(samples, seed)
    panel = _panel_arrays(list(rows), ordered_families)
    tasks = []
    for family_index, family in enumerate(ordered_families):
        for severity in (4, 5):
            severity_index = _SEVERITY_INDEX[severity]
            task = {"corruption": family, "severity": severity}
            for method in _METHODS:
                task[f"{method}_auroc"] = binary_auroc(
                    panel[method][family_index, 0],
                    panel[method][family_index, severity_index],
                )
            tasks.append(task)
    aggregate = {
        method: float(np.mean([task[f"{method}_auroc"] for task in tasks]))
        for method in _METHODS
    }
    return {
        "tasks": tasks,
        "aggregate": aggregate,
        "comparisons": {
            "fingerprint_minus_confidence": _comparison(
                panel, "fingerprint", "confidence", samples=samples, seed=seed,
                point=aggregate["fingerprint"] - aggregate["confidence"],
            ),
            "fingerprint_minus_entropy": _comparison(
                panel, "fingerprint", "entropy", samples=samples, seed=seed,
                point=aggregate["fingerprint"] - aggregate["entropy"],
            ),
        },
    }
