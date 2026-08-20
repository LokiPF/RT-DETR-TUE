from __future__ import annotations

import math

import torch
from torch import Tensor


TOPK = {"top10": 10, "top20": 20, "top50": 50}
THRESHOLDS = {"threshold_0.2": 0.2, "threshold_0.3": 0.3, "threshold_0.5": 0.5}
SMOOTH_EXPONENTS = {"smooth_1": 1.0, "smooth_2": 2.0}
ORACLE_POLICIES = ("oracle_matched", "oracle_background", "oracle_correct", "oracle_incorrect")


def _uniform_selection(indices: Tensor) -> dict:
    """Weight the selected queries equally, and mark an empty selection invalid.

    A scene that selects no query has no scene score. Reporting that here, rather than
    handing back weights nobody can normalise, keeps an empty selection distinguishable
    from a scene that really did score zero once it reaches the results table.
    """
    if indices.numel() == 0:
        return {"indices": indices, "weights": torch.empty(0), "valid": False}
    return {
        "indices": indices,
        "weights": torch.full((indices.numel(),), 1.0 / indices.numel()),
        "valid": True,
    }


def _selection_from_mask(mask: Tensor) -> dict:
    return _uniform_selection(torch.where(mask.cpu())[0])


def select_oracle_queries(record: dict, policy: str) -> dict:
    if policy == "oracle_matched":
        return _selection_from_mask(record["is_matched"])
    if policy == "oracle_background":
        return _selection_from_mask(~record["is_matched"])
    if policy == "oracle_correct":
        return _selection_from_mask(record["is_correct"])
    if policy == "oracle_incorrect":
        return _selection_from_mask(record["is_matched"] & ~record["is_correct"])
    raise ValueError(f"Unknown oracle query policy: {policy}")


def select_queries(
    logits: Tensor,
    policy: str,
    uniform_floor: float = 0.05,
    topk_override: int | None = None,
) -> dict:
    if policy in ORACLE_POLICIES:
        raise ValueError(f"{policy} reads ground truth; select_oracle_queries is its only entry point")
    confidence = logits.sigmoid().amax(dim=-1)
    query_count = logits.shape[0]
    if policy == "all":
        return _uniform_selection(torch.arange(query_count))
    if policy in TOPK:
        if topk_override is not None and topk_override < 0:
            raise ValueError(f"top-K query budget must not be negative: {topk_override}")
        count = TOPK[policy] if topk_override is None else topk_override
        order = torch.argsort(confidence, descending=True, stable=True).cpu()
        return _uniform_selection(order[:count])
    if policy in THRESHOLDS:
        return _selection_from_mask(confidence > THRESHOLDS[policy])
    if policy in SMOOTH_EXPONENTS:
        weights = uniform_floor + confidence.cpu().pow(SMOOTH_EXPONENTS[policy])
        return {
            "indices": torch.arange(query_count),
            "weights": weights / weights.sum(),
            "valid": query_count > 0,
        }
    raise ValueError(f"Unknown query policy: {policy}")


def aggregate_scores(scores: Tensor, method: str, weights: Tensor | None = None) -> Tensor:
    scores = scores.float()
    if scores.numel() == 0:
        return torch.tensor(float("nan"))
    if method == "mean":
        return scores.mean()
    if method == "median":
        return scores.median()
    if method == "q90":
        return torch.quantile(scores, 0.9)
    if method == "top20_mean":
        count = max(1, math.ceil(scores.numel() * 0.2))
        return scores.topk(count).values.mean()
    if method == "weighted_mean":
        if weights is None or weights.numel() != scores.numel():
            raise ValueError("weighted_mean requires one weight per score")
        return (scores * weights.to(scores)).sum()
    raise ValueError(f"Unknown aggregation method: {method}")
