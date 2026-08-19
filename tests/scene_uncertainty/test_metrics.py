import math

import pytest
from scipy.stats import ConstantInputWarning

from src.scene_uncertainty.metrics import has_class_switch, jaccard_overlap, monotonicity_metrics


def test_perfectly_increasing_curve_has_perfect_metrics():
    result = monotonicity_metrics([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6])
    assert result["spearman"] == 1.0
    assert result["adjacent_monotonicity"] == 1.0
    assert result["violation_magnitude"] == 0.0
    assert result["endpoint_increase"] is True


def test_input_order_does_not_change_adjacent_metrics():
    result = monotonicity_metrics([2, 0, 1], [2.0, 0.0, 1.0])
    assert result["adjacent_monotonicity"] == 1.0


def test_decrease_records_frequency_and_magnitude():
    result = monotonicity_metrics([0, 1, 2, 3], [0.0, 2.0, 1.0, 3.0])
    assert math.isclose(result["adjacent_monotonicity"], 2 / 3)
    assert result["violation_magnitude"] > 0


def test_jaccard_handles_empty_sets():
    assert jaccard_overlap([], []) == 1.0
    assert jaccard_overlap([1, 2], [2, 3]) == 1 / 3


def test_class_switch_uses_shared_matched_annotation_ids():
    first = {10: 2, 11: 7}
    second = {10: 5, 11: 7, 12: 3}
    assert has_class_switch(first, second) is True
    assert has_class_switch({10: 2}, {10: 2}) is False


def test_undefined_when_fewer_than_two_finite_scores():
    none_finite = monotonicity_metrics([0, 1, 2], [float("nan")] * 3)
    one_finite = monotonicity_metrics([0, 1], [1.0, float("nan")])
    for result in (none_finite, one_finite):
        assert math.isnan(result["spearman"])
        assert math.isnan(result["adjacent_monotonicity"])
        assert math.isnan(result["violation_magnitude"])
        assert result["endpoint_increase"] is False


def test_non_finite_scores_are_dropped_before_the_trend_is_measured():
    result = monotonicity_metrics([0, 1, 2, 3], [0.0, float("nan"), 2.0, 3.0])
    assert math.isclose(result["spearman"], 1.0)
    assert result["adjacent_monotonicity"] == 1.0
    assert result["violation_magnitude"] == 0.0
    assert result["endpoint_increase"] is True


def test_endpoint_increase_compares_the_surviving_endpoints():
    # The highest severity scored `nan`, so the endpoint comparison silently falls back to
    # the highest severity that still has a score.
    result = monotonicity_metrics([0, 1, 2], [1.0, 5.0, float("nan")])
    assert result["endpoint_increase"] is True


def test_flat_curve_reports_zero_correlation_and_no_violation():
    with pytest.warns(ConstantInputWarning):
        result = monotonicity_metrics([0, 1, 2, 3], [4.0, 4.0, 4.0, 4.0])
    assert result["spearman"] == 0.0
    assert result["adjacent_monotonicity"] == 1.0
    assert result["violation_magnitude"] == 0.0
    assert result["endpoint_increase"] is False


def test_sorting_by_severity_precedes_every_adjacent_computation():
    result = monotonicity_metrics([4, 0, 2], [3.0, 1.0, 2.0])
    assert result["adjacent_monotonicity"] == 1.0
    assert result["violation_magnitude"] == 0.0
    assert result["endpoint_increase"] is True


def test_violation_magnitude_sums_drops_relative_to_the_observed_range():
    result = monotonicity_metrics([0, 1, 2, 3, 4], [0.0, 3.0, 2.0, 5.0, 4.0])
    assert math.isclose(result["adjacent_monotonicity"], 0.5)
    assert math.isclose(result["violation_magnitude"], 2.0 / 5.0)


def test_monotone_decrease_reports_a_negative_correlation():
    result = monotonicity_metrics([0, 1, 2, 3, 4, 5], [6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    assert math.isclose(result["spearman"], -1.0)
    assert result["adjacent_monotonicity"] == 0.0
    assert math.isclose(result["violation_magnitude"], 1.0)
    assert result["endpoint_increase"] is False


def test_jaccard_ignores_duplicates_and_scores_disjoint_selections_zero():
    assert jaccard_overlap([1, 1, 2], [2, 2, 1]) == 1.0
    assert jaccard_overlap([1, 2], [3, 4]) == 0.0
    assert jaccard_overlap([], [1]) == 0.0


def test_class_switch_ignores_annotations_missing_from_either_severity():
    assert has_class_switch({10: 2, 11: 3}, {10: 2}) is False
    assert has_class_switch({10: 2}, {11: 3}) is False
    assert has_class_switch({}, {}) is False
