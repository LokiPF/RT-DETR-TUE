from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def jaccard_overlap(first, second) -> float:
    """Overlap between two query selections, as intersection over union.

    Two empty selections score 1.0. That is set-theoretically right -- they really are the
    same selection -- but in a report it reads as "selection was perfectly stable across this
    severity step" when what happened is "the policy selected nothing at either severity, and
    neither scene has a score at all". Returning 0.0 or `nan` for that case instead would be
    no better, since 0.0 reads as "the selection changed completely"; the caller is the only
    one that can separate the two, from the `selected_count` and `valid` that
    `score_cached_record` reports alongside each selection.
    """
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
    """Report whether an annotation matched at both severities changed its predicted class.

    This does not detect detection collapse. Only annotations present in both mappings are
    compared, so a scene that matched ten objects at severity 0 and matched *none* at
    severity 5 returns `False` -- the same value as a scene that was perfectly stable. Read
    `False` strictly as "no annotation that was still detected changed its predicted class",
    never as "the detector was unaffected". Match loss is the dominant effect of blur and
    this function is blind to it by construction.

    Widening the comparison to the union, with a sentinel standing in for "unmatched", is the
    wrong fix: it would push detection loss and misclassification through one flag that cannot
    say which of them happened. What is missing is a sibling metric, and `matched_predictions`
    already carries what such a metric needs -- its key set shrinks as annotations go
    unmatched.
    """
    shared = set(first) & set(second)
    return any(int(first[annotation_id]) != int(second[annotation_id]) for annotation_id in shared)
