import torch

from src.scene_uncertainty.evaluate import compute_query_distances, score_cached_record


def make_record(logits):
    return {
        "image_id": 7,
        "severity": 0,
        "layers": {2: torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])},
        "logits": logits,
        "matched_annotation_id": torch.tensor([10, -1, 11]),
        "predicted_class": logits.argmax(dim=-1),
    }


def test_same_persistence_and_confidence_with_different_argmax_has_same_score():
    first_logits = torch.tensor([[3.0, 2.0], [1.0, 0.0], [4.0, 1.0]])
    second_logits = torch.tensor([[2.0, 3.0], [0.0, 1.0], [1.0, 4.0]])
    bank = {2: torch.tensor([[0.0, 0.0], [3.0, 3.0]])}
    first = score_cached_record(make_record(first_logits), bank, "all", "mean", k=1)
    second = score_cached_record(make_record(second_logits), bank, "all", "mean", k=1)
    assert first["raw_score"] == second["raw_score"]


def test_query_distances_are_computed_once_for_all_queries_and_layers():
    logits = torch.tensor([[3.0, 2.0], [1.0, 0.0], [4.0, 1.0]])
    bank = {2: torch.tensor([[0.0, 0.0], [3.0, 3.0]])}
    distances = compute_query_distances(make_record(logits), bank, k=1)
    assert distances[2].shape == (3,)
    result = score_cached_record(
        make_record(logits), bank, "top20", "mean", k=1,
        query_distances=distances,
    )
    assert result["selected_count"] == 3


def test_empty_threshold_is_explicitly_invalid():
    logits = torch.full((3, 2), -20.0)
    bank = {2: torch.tensor([[0.0, 0.0], [3.0, 3.0]])}
    result = score_cached_record(make_record(logits), bank, "threshold_0.5", "mean", k=1)
    assert result["valid"] is False
    assert result["selected_count"] == 0
