import math
import warnings

import pytest

from src.scene_uncertainty.contrast_scores import (
    FOLD_COUNT,
    SCORE_METHODS,
    assign_folds,
    clean_residual,
    contrast_score,
    raw_gap,
    raw_responsive,
    relative_gap,
    robust_line,
)


def test_method_names_are_the_four_declared_ones():
    assert SCORE_METHODS == ("raw_responsive", "raw_gap", "relative_gap", "clean_residual")


def test_raw_responsive_returns_the_responsive_input_exactly():
    assert raw_responsive(0.4, 0.9) == 0.9
    assert raw_responsive(9.9, 0.9) == 0.9


def test_raw_gap_is_signed_and_keeps_negatives():
    assert raw_gap(0.3, 0.8) == pytest.approx(0.5)
    assert raw_gap(0.8, 0.3) == pytest.approx(-0.5)


def test_relative_gap_is_scale_invariant():
    assert relative_gap(1.0, 2.0) == pytest.approx(relative_gap(10.0, 20.0))
    assert relative_gap(1.0, 2.0) == pytest.approx(2.0 / 3.0)


def test_equal_positive_inputs_give_zero_relative_gap():
    assert relative_gap(0.7, 0.7) == 0.0


def test_zero_plus_zero_gives_zero_relative_gap():
    assert relative_gap(0.0, 0.0) == 0.0


@pytest.mark.parametrize(
    "reference, responsive",
    [(0.0, 5.0), (5.0, 0.0), (1e-9, 4.0), (4.0, 1e-9), (2.5, 2.5)],
)
def test_relative_gap_stays_within_two(reference, responsive):
    value = relative_gap(reference, responsive)
    assert -2.0 <= value <= 2.0


def test_relative_gap_reaches_the_bounds_only_when_one_side_is_zero():
    assert relative_gap(0.0, 5.0) == pytest.approx(2.0)
    assert relative_gap(5.0, 0.0) == pytest.approx(-2.0)


def test_clean_residual_subtracts_the_predicted_clean_responsive():
    # expected clean responsive = 1.0 + 2.0 * 0.5 = 2.0; observed 2.75 -> +0.75
    assert clean_residual(0.5, 2.75, slope=2.0, offset=1.0) == pytest.approx(0.75)
    assert clean_residual(0.5, 1.25, slope=2.0, offset=1.0) == pytest.approx(-0.75)


@pytest.mark.parametrize("reference, responsive", [(-0.1, 1.0), (1.0, -0.1)])
def test_only_the_relative_gap_rejects_a_negative_input(reference, responsive):
    """The combined scope is a signed z-score, so three of the four methods must accept it."""
    with pytest.raises(ValueError, match="non-negative"):
        relative_gap(reference, responsive)
    assert raw_responsive(reference, responsive) == responsive
    assert raw_gap(reference, responsive) == pytest.approx(responsive - reference)
    assert clean_residual(
        reference, responsive, slope=1.0, offset=0.0
    ) == pytest.approx(responsive - reference)


def test_a_signed_combined_style_gap_is_computed_not_refused():
    """Real values from the completed run's combined-scope decile_90_100 arm."""
    assert raw_gap(-0.2599, -0.0230) == pytest.approx(0.2369)
    assert contrast_score("raw_gap", -0.2599, -0.0230) == pytest.approx(0.2369)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_distances_are_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        relative_gap(1.0, bad)


@pytest.mark.parametrize("method", SCORE_METHODS)
@pytest.mark.parametrize("bad_position", ["reference", "responsive"])
def test_every_method_rejects_a_non_finite_input(method, bad_position):
    """`raw_responsive` ignores the reference's value but must not ignore its corruption."""
    reference, responsive = (math.nan, 1.0) if bad_position == "reference" else (1.0, math.nan)
    with pytest.raises(ValueError, match="must be finite"):
        contrast_score(method, reference, responsive, line=(1.0, 0.0))


def test_contrast_score_dispatches_to_each_method():
    assert contrast_score("raw_responsive", 0.3, 0.8) == 0.8
    assert contrast_score("raw_gap", 0.3, 0.8) == pytest.approx(0.5)
    assert contrast_score("relative_gap", 0.3, 0.8) == pytest.approx(1.0 / 1.1)
    assert contrast_score(
        "clean_residual", 0.5, 2.75, line=(2.0, 1.0)
    ) == pytest.approx(0.75)


def test_contrast_score_refuses_a_residual_without_a_line():
    with pytest.raises(ValueError, match="clean_residual needs a fitted line"):
        contrast_score("clean_residual", 0.5, 2.75)


def test_contrast_score_refuses_an_unknown_method():
    with pytest.raises(ValueError, match="unknown"):
        contrast_score("median_gap", 0.5, 2.75)


def test_robust_line_recovers_a_hand_checkable_line():
    references = [0.0, 1.0, 2.0, 3.0, 4.0]
    responsives = [1.0, 3.0, 5.0, 7.0, 9.0]  # y = 2x + 1
    assert robust_line(references, responsives) == pytest.approx((2.0, 1.0))


def test_one_extreme_outlier_does_not_control_the_robust_fit():
    references = [float(index) for index in range(10)] + [4.0]
    responsives = [2.0 * index + 1.0 for index in range(10)] + [900.0]
    slope, offset = robust_line(references, responsives)
    assert slope == pytest.approx(2.0)
    assert offset == pytest.approx(1.0)


def test_repeated_reference_values_skip_undefined_pairwise_slopes():
    # the one pair with equal references contributes no slope; the five usable slopes
    # are 2, 2, 0, 1, 2 -> median 2, and offset = median([3, 5, 3, 3]) = 3
    assert robust_line([1.0, 1.0, 2.0, 3.0], [5.0, 7.0, 7.0, 9.0]) == pytest.approx((2.0, 3.0))


def test_an_undefined_pairwise_slope_is_dropped_not_divided_by_zero():
    """The same four points with the tied pair descending, which is where dropping bites.

    Kept, that pair divides -2 by 0 and contributes -inf, which drags the median of six from
    the true 2 down to 1.5; the ascending order of the test above contributes +inf instead and
    happens to leave the median unmoved, so that test does not pin the filter down on its own.
    The five usable slopes are 0, 1, 2, 2, 2 -> median 2, and offset = median([5, 3, 3, 3]) = 3.
    The division must not happen at all, so the warning it would raise is an error here.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        line = robust_line([1.0, 1.0, 2.0, 3.0], [7.0, 5.0, 7.0, 9.0])
    assert line == pytest.approx((2.0, 3.0))


def test_robust_line_drops_non_finite_pairs_before_fitting():
    """A single unusable clean row must not poison every residual fitted from the fold."""
    references = [0.0, 1.0, 2.0, 3.0, math.nan, 4.0]
    responsives = [1.0, 3.0, 5.0, 7.0, 100.0, math.inf]
    assert robust_line(references, responsives) == pytest.approx((2.0, 1.0))


def test_a_constant_reference_makes_the_line_unavailable():
    assert robust_line([1.0, 1.0, 1.0], [2.0, 3.0, 4.0]) is None


def test_a_single_point_makes_the_line_unavailable():
    assert robust_line([1.0], [2.0]) is None


def test_robust_line_refuses_mismatched_lengths():
    with pytest.raises(ValueError, match="one responsive value per reference"):
        robust_line([1.0, 2.0], [3.0])


def test_fold_assignment_is_deterministic_under_reordering():
    forward = assign_folds([1, 3, 5, 7, 10])
    backward = assign_folds([10, 7, 5, 3, 1])
    assert forward == backward == {1: 0, 3: 1, 5: 2, 7: 3, 10: 4}


def test_fold_assignment_wraps_at_the_fold_count():
    folds = assign_folds(range(1, 13))
    assert FOLD_COUNT == 5
    assert [folds[image_id] for image_id in range(1, 13)] == [
        0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1
    ]


def test_every_fold_is_non_empty_for_a_full_tuning_run():
    folds = assign_folds(range(1, 251))
    counts = [sum(1 for value in folds.values() if value == fold) for fold in range(FOLD_COUNT)]
    assert counts == [50, 50, 50, 50, 50]
