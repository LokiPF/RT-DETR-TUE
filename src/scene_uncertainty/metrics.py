from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def jaccard_overlap(first, second) -> float:
    first_set = set(int(value) for value in first)
    second_set = set(int(value) for value in second)
    union = first_set | second_set
    if not union:
        return 1.0
    return len(first_set & second_set) / len(union)


def monotonicity_metrics(severities, scores) -> dict:
    severity = np.asarray(severities, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(values)
    severity = severity[valid]
    values = values[valid]
    order = np.argsort(severity, kind="stable")
    severity = severity[order]
    values = values[order]
    if values.size < 2:
        return {
            "spearman": float("nan"),
            "adjacent_monotonicity": float("nan"),
            "violation_magnitude": float("nan"),
            "endpoint_increase": False,
        }
    correlation = spearmanr(severity, values).statistic
    if not np.isfinite(correlation):
        correlation = 0.0
    differences = np.diff(values)
    observed_range = max(float(values.max() - values.min()), 1e-12)
    return {
        "spearman": float(correlation),
        "adjacent_monotonicity": float(np.mean(differences >= 0)),
        "violation_magnitude": float(np.maximum(-differences, 0).sum() / observed_range),
        "endpoint_increase": bool(values[-1] > values[0]),
    }


def has_class_switch(first: dict[int, int], second: dict[int, int]) -> bool:
    shared = set(first) & set(second)
    return any(int(first[annotation_id]) != int(second[annotation_id]) for annotation_id in shared)
