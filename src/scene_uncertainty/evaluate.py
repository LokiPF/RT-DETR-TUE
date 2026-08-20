from __future__ import annotations

import torch

from .knn import mean_knn_distance
from .query_policy import ORACLE_POLICIES, aggregate_scores, select_oracle_queries, select_queries


def compute_query_distances(
    record: dict,
    banks: dict[int, torch.Tensor],
    k: int,
    bank_chunk_size: int = 8192,
) -> dict[int, torch.Tensor]:
    distances = {}
    for layer_id in sorted(banks):
        queries = record["layers"][layer_id].float().to(banks[layer_id].device)
        distances[layer_id] = mean_knn_distance(
            queries, banks[layer_id], k, bank_chunk_size
        ).cpu()
    return distances


def score_cached_record(
    record: dict,
    banks: dict[int, torch.Tensor],
    policy: str,
    aggregation: str,
    k: int,
    bank_chunk_size: int = 8192,
    uniform_floor: float = 0.05,
    query_distances: dict[int, torch.Tensor] | None = None,
    layer_score_scales: dict[int, dict[str, torch.Tensor]] | None = None,
) -> dict:
    if policy in ORACLE_POLICIES:
        selection = select_oracle_queries(record, policy)
    else:
        selection = select_queries(record["logits"].float(), policy, uniform_floor=uniform_floor)
    if policy.startswith("smooth_"):
        # A smoothed policy keeps every query and carries its whole signal in the per-query
        # weights, so a plain mean -- or worse a quantile -- would aggregate that signal away.
        # The aggregation is forced to the weighted mean, and the row is then relabelled with
        # the aggregation that actually ran: the results table groups on this label, so a
        # weighted mean filed under `q90` would publish a q90 curve no q90 ever produced, and
        # sweeping `mean` and `q90` over one smoothed policy would file a single number under
        # two contradictory names instead of collapsing onto one.
        aggregation = "weighted_mean"
    matched_predictions = {
        int(annotation_id): int(predicted_class)
        for annotation_id, predicted_class in zip(
            record["matched_annotation_id"].tolist(),
            record["predicted_class"].tolist(),
        )
        if int(annotation_id) >= 0
    }
    base = {
        "image_id": int(record["image_id"]),
        "severity": int(record.get("severity", 0)),
        "source_partition": record.get("source_partition", "unknown"),
        "policy": policy,
        "aggregation": aggregation,
        "selected_count": int(selection["indices"].numel()),
        "selected_query_ids": selection["indices"].tolist(),
        "matched_predictions": matched_predictions,
        "valid": bool(selection["valid"]),
    }
    if not selection["valid"]:
        return {
            **base,
            "raw_score": float("nan"),
            "layer_scores": {},
            "clean_scaled_layer_scores": {},
        }

    if query_distances is None:
        query_distances = compute_query_distances(record, banks, k, bank_chunk_size)
    layer_scores: dict[int, float] = {}
    for layer_id in sorted(query_distances):
        selected_scores = query_distances[layer_id].index_select(0, selection["indices"])
        score = aggregate_scores(selected_scores, aggregation, selection["weights"])
        layer_scores[layer_id] = float(score.item())
    if layer_score_scales is None:
        layer_score_scales = {
            layer_id: {"center": torch.tensor(0.0), "scale": torch.tensor(1.0)}
            for layer_id in layer_scores
        }
    clean_scaled_layers = {
        layer_id: (score - float(layer_score_scales[layer_id]["center"]))
        / float(layer_score_scales[layer_id]["scale"])
        for layer_id, score in layer_scores.items()
    }
    # `raw_score` is the mean of the *scaled* per-layer scores; `layer_scores` keeps the
    # unscaled ones because they are what the per-layer trend curves are drawn from.
    # `clean_scaled_layer_scores` is the per-row record of the standardisation and is
    # currently read by nothing downstream -- see `reporting._expanded_frame`.
    raw_score = sum(clean_scaled_layers.values()) / len(clean_scaled_layers)
    return {
        **base,
        "raw_score": raw_score,
        "layer_scores": layer_scores,
        "clean_scaled_layer_scores": clean_scaled_layers,
    }
