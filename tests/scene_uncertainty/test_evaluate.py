import math

import pytest
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


CONFIDENT_LOGITS = torch.tensor([[3.0, 2.0], [1.0, 0.0], [4.0, 1.0]])
THREE_LAYER_BANKS = {
    layer_id: torch.tensor([[0.0, 0.0], [10.0, 10.0]]) for layer_id in (0, 1, 2)
}
DEPLOYABLE_POLICIES = (
    "all", "top10", "top20", "top50",
    "threshold_0.2", "threshold_0.3", "threshold_0.5",
    "smooth_1", "smooth_2",
)


def make_layered_record(**overrides):
    """Three decoder layers whose queries sit at three separable distances from one bank."""
    record = {
        "image_id": 7,
        "severity": 3,
        "logits": CONFIDENT_LOGITS,
        "matched_annotation_id": torch.tensor([10, -1, 11]),
        "predicted_class": torch.tensor([0, 1, 1]),
        "layers": {
            0: torch.zeros(3, 2),
            1: torch.ones(3, 2),
            2: torch.full((3, 2), 2.0),
        },
    }
    record.update(overrides)
    return record


def test_each_layer_score_is_keyed_by_the_layer_it_was_computed_from():
    result = score_cached_record(make_layered_record(), THREE_LAYER_BANKS, "all", "mean", k=1)
    assert set(result["layer_scores"]) == {0, 1, 2}
    assert math.isclose(result["layer_scores"][0], 0.0, abs_tol=1e-6)
    assert math.isclose(result["layer_scores"][1], math.sqrt(2.0), rel_tol=1e-6)
    assert math.isclose(result["layer_scores"][2], 2.0 * math.sqrt(2.0), rel_tol=1e-6)
    assert math.isclose(result["raw_score"], math.sqrt(2.0), rel_tol=1e-6)


def test_clean_scaling_uses_each_layers_own_center_and_spread():
    scales = {
        0: {"center": torch.tensor(0.7), "scale": torch.tensor(2.0)},
        1: {"center": torch.tensor(1.0), "scale": torch.tensor(4.0)},
        2: {"center": torch.tensor(3.0), "scale": torch.tensor(5.0)},
    }
    result = score_cached_record(
        make_layered_record(), THREE_LAYER_BANKS, "all", "mean", k=1,
        layer_score_scales=scales,
    )
    expected = {
        0: (0.0 - 0.7) / 2.0,
        1: (math.sqrt(2.0) - 1.0) / 4.0,
        2: (2.0 * math.sqrt(2.0) - 3.0) / 5.0,
    }
    for layer_id, value in expected.items():
        assert math.isclose(result["clean_scaled_layer_scores"][layer_id], value, abs_tol=1e-6)
    assert math.isclose(result["raw_score"], sum(expected.values()) / 3.0, abs_tol=1e-6)
    assert result["layer_scores"][2] != result["clean_scaled_layer_scores"][2]


def test_a_missing_clean_scale_leaves_the_layer_scores_unscaled():
    fallback = score_cached_record(make_layered_record(), THREE_LAYER_BANKS, "all", "mean", k=1)
    explicit = score_cached_record(
        make_layered_record(), THREE_LAYER_BANKS, "all", "mean", k=1,
        layer_score_scales={
            layer_id: {"center": torch.tensor(0.0), "scale": torch.tensor(1.0)}
            for layer_id in (0, 1, 2)
        },
    )
    assert fallback["clean_scaled_layer_scores"] == fallback["layer_scores"]
    assert fallback == explicit
    assert math.isclose(
        fallback["raw_score"], sum(fallback["layer_scores"].values()) / 3.0, rel_tol=1e-9
    )


def test_an_empty_selection_reports_a_missing_score_rather_than_a_zero():
    logits = torch.full((3, 2), -20.0)
    bank = {2: torch.tensor([[0.0, 0.0], [3.0, 3.0]])}
    result = score_cached_record(make_record(logits), bank, "threshold_0.5", "mean", k=1)
    assert math.isnan(result["raw_score"])
    assert result["layer_scores"] == {}
    assert result["clean_scaled_layer_scores"] == {}
    assert result["selected_query_ids"] == []
    assert result["policy"] == "threshold_0.5"
    assert result["matched_predictions"] == {10: 0, 11: 0}


def test_oracle_policies_read_ground_truth_and_stay_labelled_as_oracles():
    record = make_layered_record(
        is_matched=torch.tensor([True, False, True]),
        is_correct=torch.tensor([True, False, False]),
    )
    matched = score_cached_record(record, THREE_LAYER_BANKS, "oracle_matched", "mean", k=1)
    background = score_cached_record(record, THREE_LAYER_BANKS, "oracle_background", "mean", k=1)
    correct = score_cached_record(record, THREE_LAYER_BANKS, "oracle_correct", "mean", k=1)
    incorrect = score_cached_record(record, THREE_LAYER_BANKS, "oracle_incorrect", "mean", k=1)
    assert matched["policy"] == "oracle_matched"
    assert matched["selected_query_ids"] == [0, 2]
    assert background["selected_query_ids"] == [1]
    assert correct["selected_query_ids"] == [0]
    assert incorrect["selected_query_ids"] == [2]


@pytest.mark.parametrize("policy", DEPLOYABLE_POLICIES)
def test_deployable_policies_score_a_record_that_carries_no_oracle_fields(policy):
    record = make_layered_record()
    assert "is_matched" not in record and "is_correct" not in record
    result = score_cached_record(record, THREE_LAYER_BANKS, policy, "mean", k=1)
    assert result["policy"] == policy
    assert result["valid"] is True


def test_a_deployable_score_does_not_move_with_the_ground_truth_labels():
    plain = score_cached_record(make_layered_record(), THREE_LAYER_BANKS, "top20", "mean", k=1)
    relabelled = score_cached_record(
        make_layered_record(
            matched_annotation_id=torch.tensor([-1, 44, 45]),
            predicted_class=torch.tensor([1, 0, 0]),
            is_matched=torch.tensor([False, True, True]),
            is_correct=torch.tensor([False, True, False]),
        ),
        THREE_LAYER_BANKS, "top20", "mean", k=1,
    )
    assert relabelled["selected_query_ids"] == plain["selected_query_ids"]
    assert relabelled["raw_score"] == plain["raw_score"]
    assert relabelled["matched_predictions"] != plain["matched_predictions"]


def test_an_unknown_policy_is_rejected_rather_than_scored():
    with pytest.raises(ValueError, match="Unknown query policy"):
        score_cached_record(make_layered_record(), THREE_LAYER_BANKS, "top1", "mean", k=1)


def test_matched_predictions_pair_each_annotation_with_the_query_that_matched_it():
    result = score_cached_record(make_layered_record(), THREE_LAYER_BANKS, "all", "mean", k=1)
    assert result["matched_predictions"] == {10: 0, 11: 1}


def test_scoring_one_record_twice_gives_the_same_numbers():
    scales = {
        layer_id: {"center": torch.tensor(0.3), "scale": torch.tensor(1.7)}
        for layer_id in (0, 1, 2)
    }
    first = score_cached_record(
        make_layered_record(), THREE_LAYER_BANKS, "smooth_1", "mean", k=1,
        layer_score_scales=scales,
    )
    second = score_cached_record(
        make_layered_record(), THREE_LAYER_BANKS, "smooth_1", "mean", k=1,
        layer_score_scales=scales,
    )
    assert first == second


def test_a_smoothed_policy_reports_the_weighted_mean_it_actually_computed():
    """The `aggregation` label has to name the aggregation that ran.

    A smoothed policy carries its whole signal in the per-query weights, so a plain mean
    or a quantile would throw that signal away and the aggregation is forced to the
    weighted mean. The results table groups on `aggregation`, so a weighted mean filed
    under `q90` publishes a q90 curve no q90 ever produced, and asking for `mean` and
    `q90` of one smoothed policy files a single number under two contradictory names.
    """
    record = make_layered_record(layers={0: torch.tensor([[0.0, 0.0], [3.0, 3.0], [9.0, 9.0]])})
    banks = {0: torch.tensor([[0.0, 0.0]])}
    distances = compute_query_distances(record, banks, k=1)
    weights = 0.05 + CONFIDENT_LOGITS.sigmoid().amax(dim=-1)
    expected = float((distances[0] * (weights / weights.sum())).sum())

    as_mean = score_cached_record(record, banks, "smooth_1", "mean", k=1)
    as_quantile = score_cached_record(record, banks, "smooth_1", "q90", k=1)
    assert as_mean["aggregation"] == "weighted_mean"
    assert as_quantile == as_mean
    assert math.isclose(as_mean["layer_scores"][0], expected, rel_tol=1e-6)
    assert not math.isclose(expected, float(distances[0].mean()), rel_tol=1e-6)
