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
    """Describe how a scene score trends across blur severity, and over how many points.

    A scene has no score wherever its query selection came out empty, which is exactly what
    blur causes: confidence falls, a threshold policy selects nothing, and the severity is
    scored `nan`. Those points are dropped, so every statistic below describes only the
    severities that survived -- a policy that collapsed at high severity is graded on the
    low severities where it still worked, and reports the same `spearman` and
    `endpoint_increase` as a policy that really did rise the whole way.

    `finite_count` and `total_count` are reported so that collapse is detectable rather
    than invisible. They are recorded, never acted on: no survival fraction is used to
    suppress or blank the statistics, because choosing that cutoff is a research-design
    call for the study rather than one to bury here. Both counts appear on every return
    path, so a consumer can read them without first checking that they exist.
    """
    severity = np.asarray(severities, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    total_count = int(values.size)
    valid = np.isfinite(values)
    severity = severity[valid]
    values = values[valid]
    order = np.argsort(severity, kind="stable")
    severity = severity[order]
    values = values[order]
    counts = {"finite_count": int(values.size), "total_count": total_count}
    if values.size < 2:
        return {
            "spearman": float("nan"),
            "adjacent_monotonicity": float("nan"),
            "violation_magnitude": float("nan"),
            "endpoint_increase": False,
            **counts,
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
        **counts,
    }


def has_class_switch(first: dict[int, int], second: dict[int, int]) -> bool:
    shared = set(first) & set(second)
    return any(int(first[annotation_id]) != int(second[annotation_id]) for annotation_id in shared)
