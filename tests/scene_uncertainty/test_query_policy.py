import pytest
import torch

from src.scene_uncertainty.query_policy import (
    ORACLE_POLICIES,
    aggregate_scores,
    select_oracle_queries,
    select_queries,
)


def logits_from_confidences(confidences):
    values = torch.tensor(confidences).clamp(1e-5, 1 - 1e-5)
    logits = torch.logit(values)
    return torch.stack((logits, torch.full_like(logits, -20.0)), dim=1)


def oracle_record(is_matched, is_correct):
    return {"is_matched": torch.tensor(is_matched), "is_correct": torch.tensor(is_correct)}


def test_topk_returns_unique_queries_in_descending_confidence_order():
    logits = logits_from_confidences([0.1, 0.9, 0.4, 0.8])
    selection = select_queries(logits, "top20", topk_override=2)
    assert selection["indices"].tolist() == [1, 3]
    assert torch.allclose(selection["weights"], torch.tensor([0.5, 0.5]))


def test_threshold_uses_strict_greater_than_and_reports_empty():
    logits = logits_from_confidences([0.2, 0.3, 0.5])
    selected = select_queries(logits, "threshold_0.3")
    empty = select_queries(logits, "threshold_0.5")
    assert selected["indices"].tolist() == [2]
    assert empty["valid"] is False


def test_smooth_policy_keeps_every_query_with_positive_weight():
    logits = logits_from_confidences([0.01, 0.5, 0.99])
    selection = select_queries(logits, "smooth_2", uniform_floor=0.05)
    assert selection["indices"].tolist() == [0, 1, 2]
    assert torch.all(selection["weights"] > 0)
    assert torch.isclose(selection["weights"].sum(), torch.tensor(1.0))


def test_oracle_policies_use_matching_metadata_without_class_routing():
    record = {
        "matched_gt_class": torch.tensor([2, 4, -1, 7]),
        "is_matched": torch.tensor([True, True, False, True]),
        "is_correct": torch.tensor([True, False, False, False]),
    }
    assert select_oracle_queries(record, "oracle_matched")["indices"].tolist() == [0, 1, 3]
    assert select_oracle_queries(record, "oracle_background")["indices"].tolist() == [2]
    assert select_oracle_queries(record, "oracle_correct")["indices"].tolist() == [0]
    assert select_oracle_queries(record, "oracle_incorrect")["indices"].tolist() == [1, 3]


def test_scene_aggregations_have_expected_values():
    scores = torch.tensor([1.0, 2.0, 3.0, 100.0])
    assert aggregate_scores(scores, "mean").item() == 26.5
    assert aggregate_scores(scores, "median").item() == 2.0
    assert aggregate_scores(scores, "top20_mean").item() == 100.0
    weighted = aggregate_scores(scores, "weighted_mean", torch.tensor([0.1, 0.2, 0.3, 0.4]))
    assert torch.isclose(weighted, torch.tensor(41.4))


def test_all_policy_keeps_every_query_with_uniform_weight():
    logits = logits_from_confidences([0.1, 0.9, 0.4, 0.8])
    selection = select_queries(logits, "all")
    assert selection["valid"] is True
    assert selection["indices"].tolist() == [0, 1, 2, 3]
    assert torch.allclose(selection["weights"], torch.full((4,), 0.25))
    assert select_queries(torch.empty(0, 2), "all")["valid"] is False


def test_topk_breaks_confidence_ties_by_ascending_query_index():
    logits = logits_from_confidences([0.5, 0.9, 0.5, 0.5])
    first = select_queries(logits, "top20", topk_override=3)
    second = select_queries(logits, "top20", topk_override=3)
    assert first["indices"].tolist() == [1, 0, 2]
    assert torch.equal(first["indices"], second["indices"])


def test_topk_uses_locked_budgets_and_clamps_to_the_available_queries():
    logits = logits_from_confidences([0.05 * (index + 1) for index in range(12)])
    ten = select_queries(logits, "top10")
    assert ten["indices"].tolist() == [11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
    assert torch.allclose(ten["weights"], torch.full((10,), 0.1))
    fifty = select_queries(logits, "top50")
    assert fifty["indices"].tolist() == list(range(11, -1, -1))
    assert torch.allclose(fifty["weights"], torch.full((12,), 1.0 / 12))
    assert select_queries(logits, "top20", topk_override=0)["valid"] is False
    with pytest.raises(ValueError, match="must not be negative: -1"):
        select_queries(logits, "top20", topk_override=-1)


def test_threshold_boundary_excludes_queries_at_the_locked_thresholds():
    logits = logits_from_confidences([0.2, 0.25, 0.3, 0.5, 0.6])
    assert select_queries(logits, "threshold_0.2")["indices"].tolist() == [1, 2, 3, 4]
    assert select_queries(logits, "threshold_0.3")["indices"].tolist() == [3, 4]
    assert select_queries(logits, "threshold_0.5")["indices"].tolist() == [4]


def test_smooth_weights_add_the_uniform_floor_before_normalising():
    logits = logits_from_confidences([1e-4, 0.9])
    squared = select_queries(logits, "smooth_2", uniform_floor=0.05)
    assert torch.allclose(squared["weights"], torch.tensor([0.054945, 0.945055]), atol=1e-5)
    linear = select_queries(logits, "smooth_1", uniform_floor=0.05)
    assert torch.allclose(linear["weights"], torch.tensor([0.050095, 0.949905]), atol=1e-5)


def test_oracle_policies_report_an_empty_selection_as_invalid():
    record = oracle_record([False, False], [False, False])
    matched = select_oracle_queries(record, "oracle_matched")
    assert matched["valid"] is False
    assert matched["indices"].numel() == 0
    assert matched["weights"].numel() == 0
    assert select_oracle_queries(record, "oracle_correct")["valid"] is False
    assert select_oracle_queries(record, "oracle_incorrect")["valid"] is False
    assert select_oracle_queries(record, "oracle_background")["indices"].tolist() == [0, 1]


def test_oracle_policies_are_reachable_only_through_the_oracle_entry_point():
    logits = logits_from_confidences([0.1, 0.9])
    record = oracle_record([True, False], [True, False])
    assert ORACLE_POLICIES == ("oracle_matched", "oracle_background", "oracle_correct", "oracle_incorrect")
    for policy in ORACLE_POLICIES:
        with pytest.raises(ValueError, match="select_oracle_queries"):
            select_queries(logits, policy)
    for policy in ("all", "top20", "threshold_0.3", "smooth_2"):
        with pytest.raises(ValueError, match="Unknown oracle query policy"):
            select_oracle_queries(record, policy)


def test_unknown_policy_and_aggregation_names_raise():
    logits = logits_from_confidences([0.1, 0.9])
    with pytest.raises(ValueError, match="Unknown query policy"):
        select_queries(logits, "top99")
    with pytest.raises(ValueError, match="Unknown oracle query policy"):
        select_oracle_queries(oracle_record([True], [True]), "oracle_rare")
    with pytest.raises(ValueError, match="Unknown aggregation method"):
        aggregate_scores(torch.tensor([1.0]), "q99")


def test_weighted_mean_requires_one_weight_per_score():
    scores = torch.tensor([1.0, 2.0])
    with pytest.raises(ValueError, match="one weight per score"):
        aggregate_scores(scores, "weighted_mean")
    with pytest.raises(ValueError, match="one weight per score"):
        aggregate_scores(scores, "weighted_mean", torch.tensor([1.0]))


def test_every_aggregation_of_an_empty_selection_is_nan():
    empty = torch.empty(0)
    for method in ("mean", "median", "q90", "top20_mean", "weighted_mean"):
        assert torch.isnan(aggregate_scores(empty, method, torch.empty(0)))


def test_q90_interpolates_between_neighbouring_scores():
    scores = torch.tensor([1.0, 2.0, 3.0, 100.0])
    assert torch.isclose(aggregate_scores(scores, "q90"), torch.tensor(70.9))


def test_top20_mean_rounds_its_query_budget_up():
    assert aggregate_scores(torch.arange(1, 8, dtype=torch.float32), "top20_mean").item() == 6.5
    assert aggregate_scores(torch.arange(1, 301, dtype=torch.float32), "top20_mean").item() == 270.5


def test_aggregations_are_defined_for_one_score_and_for_identical_scores():
    single = torch.tensor([4.0])
    for method in ("mean", "median", "q90", "top20_mean"):
        assert aggregate_scores(single, method).item() == 4.0
    assert aggregate_scores(single, "weighted_mean", torch.tensor([1.0])).item() == 4.0
    identical = torch.full((5,), 7.0)
    for method in ("mean", "median", "q90", "top20_mean"):
        assert aggregate_scores(identical, method).item() == 7.0
    assert aggregate_scores(identical, "weighted_mean", torch.full((5,), 0.2)).item() == pytest.approx(7.0)
