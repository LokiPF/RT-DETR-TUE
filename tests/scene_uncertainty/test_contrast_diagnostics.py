"""Hand arithmetic against `scene_uncertainty.contrast_diagnostics`, one claim per test.

Several fixtures here are chosen rather than obvious, because the obvious version of each
passes against broken code. The pattern behind every one of them is the same: a roster that is
symmetric, uniformly moving, or exactly collinear makes the right answer and the wrong answer
the same number.

`test_drift_is_measured_against_each_image_own_severity_zero` uses baselines `1, 2, 3, 10`.
Symmetric baselines -- `1, 2, 3, 4`, which is what several other fixtures use -- cannot see the
bug the test is named for: with a uniform drift the median of `value - group mean` equals the
median of `value - own clean value` exactly, so measuring the drift against the group's
severity-zero mean reproduces `0.2` and `0.5` unchanged. Asymmetry separates the mean variant,
and the absolute-drift assertion beside it separates the group *median* variant, which
asymmetry alone does not: the median of `value - median(value)` is zero for any roster.

`test_stability_to_spread_divides_drift_by_the_clean_interquartile_range` gives the four images
*different* per-severity steps. When every image drifts by the same amount the spread at
severity 2 is a translate of the spread at severity 0, so its interquartile range is also
`1.5`, and dividing by the wrong severity's spread returns the right answer anyway.

`test_every_fold_line_excludes_its_own_fold` uses four images, not ten. Theil-Sen is robust
enough that one dragged-off image out of ten contributes 9 of 45 pairwise slopes and the
median stays at `2.0` whether or not the fold is held out -- the mutation that fits each fold
on the whole roster survives that fixture untouched. A single outlier is visible at three and
four images and invisible from five onwards, so four is the largest roster that still binds.

`test_an_equal_error_line_is_not_predictive` alternates the responsive value between `5.0` and
`6.0` so that every fold's median pairwise slope is exactly zero and its intercept is exactly
the median of the training responsives -- which makes the fitted line and the constant
predictor the same predictor, and their two errors exactly equal. Nothing else in the file
distinguishes `crossfit_error < constant_error` from `<=`.

`one_scene_far_off_the_line` is the only fixture whose error lists are skewed. Everywhere else
the mean and the median of both error lists coincide, so "these are medians of absolute errors
rather than means", which is the most-argued sentence in `clean_relationship`, is invisible.
"""

import math

import pytest

from src.scene_uncertainty.contrast_diagnostics import (
    between_image_spread,
    clean_relationship,
    within_image_drift,
)
from src.scene_uncertainty.contrast_scores import FOLD_COUNT, assign_folds, robust_line

SEVERITIES = range(6)


def flat_curves():
    """Four images whose reference never moves: baselines 1.0, 2.0, 3.0, 4.0."""
    return {
        image_id: {severity: float(image_id) for severity in SEVERITIES}
        for image_id in (1, 2, 3, 4)
    }


def opposing_curves():
    """Two images whose references move the same distance in opposite directions."""
    return {
        1: {severity: 1.0 + 0.1 * severity for severity in SEVERITIES},
        2: {severity: 2.0 - 0.1 * severity for severity in SEVERITIES},
    }


def wandering_curves():
    """Four images that step once, by four different signed amounts, and then hold.

    The four drifts -- -0.5, -0.25, 0.25, 0.75 -- are quarters, so every quartile below is an
    exact binary fraction and can be asserted without a tolerance.
    """
    return {
        image_id: {
            severity: float(image_id) + (drift if severity else 0.0)
            for severity in SEVERITIES
        }
        for image_id, drift in ((1, -0.5), (2, -0.25), (3, 0.25), (4, 0.75))
    }


def one_scene_far_off_the_line():
    """Ten collinear clean images with image 1 dragged to 500.0, cross-fitted.

    Every fold's line is exactly `(2.0, 1.0)`: a training set holding the outlier still has 21
    of its 28 pairwise slopes at `2.0` and seven far below, so the median never leaves that
    block, and `median(y - 2x)` over seven `1.0`s and one `498.0` is `1.0`. That makes every
    number below hand-computable, and it makes both error lists lopsided -- which is the point
    of the fixture.
    """
    references = {image_id: float(image_id) for image_id in range(1, 11)}
    responsives = {image_id: 2.0 * float(image_id) + 1.0 for image_id in range(1, 11)}
    responsives[1] = 500.0
    return clean_relationship(references, responsives, folds=assign_folds(references))


def collapsed_line_relationship():
    """Ten images whose responsive alternates 6.0 / 5.0, cross-fitted.

    For every fold's eight-image training set the 28 pairwise slopes are 12 exact zeros from
    the same-parity pairs and 16 non-zero ones, and the 14th and 15th sorted values always sit
    inside the zero block, so the median slope is exactly `0.0`. A zero slope makes the
    intercept `median(y - 0 * x)`, which *is* the median of the training responsives -- the
    fitted line collapses onto the constant predictor and the two error lists become the same
    list.
    """
    references = {image_id: float(image_id) for image_id in range(1, 11)}
    responsives = {image_id: 5.0 if image_id % 2 == 0 else 6.0 for image_id in references}
    return clean_relationship(references, responsives, folds=assign_folds(references))


def test_a_flat_reference_has_zero_drift_everywhere():
    drift = within_image_drift(flat_curves())
    for severity in range(1, 6):
        row = drift["by_severity"][severity]
        assert row["median_signed_drift"] == 0.0
        assert row["median_absolute_drift"] == 0.0
        assert row["zero_drift_fraction"] == 1.0


def test_a_flat_reference_is_flat_not_unmeasured():
    drift = within_image_drift(flat_curves())
    for image_id in (1, 2, 3, 4):
        image = drift["by_image"][image_id]
        assert image["reference_range"] == 0.0
        assert image["signed_spearman"] == 0.0
        assert image["absolute_spearman"] == 0.0


def test_a_falling_reference_keeps_its_sign_in_the_per_image_trend():
    # The flat roster above has both correlations at 0.0, so it cannot tell the signed trend
    # from the absolute one. A reference that falls with blur is the case that can.
    curves = {
        1: {severity: 5.0 - severity for severity in SEVERITIES},
        2: {severity: 5.0 + severity for severity in SEVERITIES},
    }
    by_image = within_image_drift(curves)["by_image"]
    assert by_image[1]["signed_spearman"] == pytest.approx(-1.0)
    assert by_image[1]["absolute_spearman"] == pytest.approx(1.0)
    assert by_image[2]["signed_spearman"] == pytest.approx(1.0)


def test_reference_range_spans_the_whole_curve_not_just_its_endpoints():
    # A reference that rises and comes back is exactly the wandering anchor these diagnostics
    # exist to expose, and it is the only curve shape on which max-minus-min and
    # last-minus-first disagree.
    curves = {1: {0: 1.0, 1: 1.5, 2: 2.0, 3: 2.0, 4: 1.5, 5: 1.0}}
    assert within_image_drift(curves)["by_image"][1]["reference_range"] == pytest.approx(1.0)


def test_drift_is_measured_against_each_image_own_severity_zero():
    # Baselines 1, 2, 3, 10 -- deliberately not symmetric. Their mean, 4.0, and their median,
    # 2.5, both differ from every image's own clean value, so a drift taken against either
    # group anchor lands somewhere other than the uniform 0.1-per-severity step below.
    curves = {
        image_id: {severity: baseline + 0.1 * severity for severity in SEVERITIES}
        for image_id, baseline in ((1, 1.0), (2, 2.0), (3, 3.0), (4, 10.0))
    }
    drift = within_image_drift(curves)
    # every image drifts the same amount, so the median equals that amount exactly
    assert drift["by_severity"][2]["median_signed_drift"] == pytest.approx(0.2)
    assert drift["by_severity"][2]["median_absolute_drift"] == pytest.approx(0.2)
    assert drift["by_severity"][5]["median_signed_drift"] == pytest.approx(0.5)
    assert drift["by_severity"][3]["zero_drift_fraction"] == 0.0
    assert drift["by_image"][4]["reference_range"] == pytest.approx(0.5)


def test_zero_drift_fraction_counts_exact_zeros_not_near_zeros():
    # image 2 moves by two units in the last place -- 2**-50, about 8.9e-16, far below any
    # tolerance anyone would write down, and still a reference that did not hold still. The
    # median of one exact zero and that step is half of it.
    curves = {
        1: {severity: 1.0 for severity in SEVERITIES},
        2: {severity: 2.0 + (1e-15 if severity else 0.0) for severity in SEVERITIES},
    }
    row = within_image_drift(curves)["by_severity"][1]
    assert row["median_absolute_drift"] == 2.0**-51
    assert row["zero_drift_fraction"] == 0.5


def test_signed_and_absolute_drift_differ_when_images_move_opposite_ways():
    row = within_image_drift(opposing_curves())["by_severity"][4]
    assert row["median_signed_drift"] == pytest.approx(0.0)
    assert row["median_absolute_drift"] == pytest.approx(0.4)


def test_the_drift_quartiles_describe_the_signed_and_absolute_spreads_separately():
    row = within_image_drift(wandering_curves())["by_severity"][1]
    # signed drifts [-0.5, -0.25, 0.25, 0.75]; their absolute values [0.25, 0.25, 0.5, 0.75].
    # Four different numbers, so neither pair can be computed from the other array.
    assert row["signed_q25"] == -0.3125
    assert row["signed_q75"] == 0.375
    assert row["absolute_q25"] == 0.25
    assert row["absolute_q75"] == 0.5625
    assert row["median_signed_drift"] == 0.0
    assert row["median_absolute_drift"] == 0.375


def test_between_image_spread_reports_the_full_shape():
    spread = between_image_spread(flat_curves(), within_image_drift(flat_curves()))
    clean = spread[0]
    assert clean["count"] == 4
    assert clean["mean"] == pytest.approx(2.5)
    assert clean["variance"] == pytest.approx(1.25)  # population, not sample
    assert clean["median"] == pytest.approx(2.5)
    assert clean["q25"] == pytest.approx(1.75)
    assert clean["q75"] == pytest.approx(3.25)
    assert clean["iqr"] == pytest.approx(1.5)
    assert clean["mad"] == pytest.approx(1.0)
    assert clean["min"] == pytest.approx(1.0)
    assert clean["max"] == pytest.approx(4.0)


def test_the_spread_summary_separates_the_median_from_the_mean():
    # Baselines 1, 2, 3, 10: mean 4.0 against median 2.5. On the symmetric roster above the
    # two coincide, so an absolute deviation taken about the mean scores the same 1.0.
    curves = {
        image_id: {severity: baseline for severity in SEVERITIES}
        for image_id, baseline in ((1, 1.0), (2, 2.0), (3, 3.0), (4, 10.0))
    }
    clean = between_image_spread(curves, within_image_drift(curves))[0]
    assert clean["mean"] == pytest.approx(4.0)
    assert clean["median"] == pytest.approx(2.5)
    assert clean["variance"] == pytest.approx(12.5)  # population, not sample
    assert clean["mad"] == pytest.approx(1.0)  # about the median; about the mean it is 2.5


def test_severity_zero_has_no_stability_to_spread_ratio():
    spread = between_image_spread(flat_curves(), within_image_drift(flat_curves()))
    assert spread[0]["stability_to_spread"] is None


def test_stability_to_spread_divides_drift_by_the_clean_interquartile_range():
    # Uneven steps, so severity 2's own spread (interquartile range 1.8) is not the clean
    # spread (1.5) and the two possible denominators give different answers.
    curves = {
        image_id: {severity: baseline + step * severity for severity in SEVERITIES}
        for image_id, baseline, step in (
            (1, 1.0, 0.0),
            (2, 2.0, 0.1),
            (3, 3.0, 0.2),
            (4, 4.0, 0.3),
        )
    }
    spread = between_image_spread(curves, within_image_drift(curves))
    # median absolute drift at severity 2 is 0.3; the clean IQR of [1, 2, 3, 4] is 1.5
    assert spread[2]["iqr"] == pytest.approx(1.8)
    assert spread[2]["stability_to_spread"] == pytest.approx(0.3 / 1.5)


def test_stability_to_spread_uses_the_absolute_drift_not_the_signed_drift():
    # Two images drifting 0.4 apart in opposite directions: their signed drifts cancel to a
    # median of 0.0 and would read as a perfectly still anchor, which is the failure mode
    # these diagnostics exist to catch. The clean IQR of [1.0, 2.0] is 0.5.
    curves = opposing_curves()
    spread = between_image_spread(curves, within_image_drift(curves))
    assert spread[4]["stability_to_spread"] == pytest.approx(0.8)


def test_a_zero_clean_spread_makes_the_ratio_unavailable():
    curves = {
        image_id: {severity: 3.0 + 0.1 * severity for severity in SEVERITIES}
        for image_id in (1, 2, 3, 4)
    }
    spread = between_image_spread(curves, within_image_drift(curves))
    assert spread[3]["stability_to_spread"] is None


def test_a_predictive_clean_relationship_beats_the_constant_median():
    references = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
    responsives = {image_id: 2.0 * value + 1.0 for image_id, value in references.items()}
    result = clean_relationship(
        references, responsives, folds=assign_folds(references)
    )
    assert result["pearson"] == pytest.approx(1.0)
    assert result["spearman"] == pytest.approx(1.0)
    assert result["final_slope"] == pytest.approx(2.0)
    assert result["final_offset"] == pytest.approx(1.0)
    assert result["crossfit_median_absolute_error"] == pytest.approx(0.0)
    assert result["constant_median_absolute_error"] == pytest.approx(3.0)
    assert result["predictive"] is True


def test_pearson_and_spearman_are_two_different_numbers():
    # A parabola: perfectly monotone, so Spearman is 1.0, but the covariance of 25.0 over the
    # two deviation norms sqrt(5) and sqrt(129) leaves Pearson at 25 / sqrt(645). The linear
    # fixture above cannot see the difference, because there both are 1.0.
    references = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
    responsives = {1: 1.0, 2: 4.0, 3: 9.0, 4: 16.0}
    result = clean_relationship(references, responsives, folds=assign_folds(references))
    assert result["spearman"] == pytest.approx(1.0)
    assert result["pearson"] == pytest.approx(25.0 / math.sqrt(645.0))
    assert result["pearson"] < result["spearman"]


def test_a_non_predictive_clean_relationship_is_labelled_so():
    references = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
    responsives = {1: 5.0, 2: 5.1, 3: 4.9, 4: 5.0}
    result = clean_relationship(
        references, responsives, folds=assign_folds(references)
    )
    assert result["crossfit_median_absolute_error"] == pytest.approx(0.125)
    assert result["constant_median_absolute_error"] == pytest.approx(0.05)
    assert result["predictive"] is False


def test_both_errors_are_medians_of_absolute_errors_not_means():
    result = one_scene_far_off_the_line()
    # nine images land exactly on their fold's line and image 1 misses by 497, so the line's
    # errors are [497, 0 x 9]: median 0.0, mean 49.7. The constant's are
    # [487, 10, 7, 7, 5, 5, 3, 3, 0, 0]: median 5.0, mean 52.7.
    assert result["crossfit_median_absolute_error"] == 0.0
    assert result["constant_median_absolute_error"] == 5.0
    assert result["predictive"] is True


def test_the_residual_summary_is_taken_about_the_median():
    result = one_scene_far_off_the_line()
    # the same [497, 0 x 9] residuals. Taken about the mean of 49.7 all three of these read
    # 49.7, which would describe nine exactly fitted scenes as uniformly off by fifty.
    assert result["residual_median"] == 0.0
    assert result["residual_iqr"] == 0.0
    assert result["residual_mad"] == 0.0


def test_the_residual_spread_and_deviation_are_different_statistics():
    result = collapsed_line_relationship()
    # residuals are five at +0.5 and five at -0.5, so the interquartile range spans the full
    # 1.0 and the median absolute deviation is half of it.
    assert result["residual_median"] == 0.0
    assert result["residual_iqr"] == 1.0
    assert result["residual_mad"] == 0.5


def test_an_equal_error_line_is_not_predictive():
    result = collapsed_line_relationship()
    # every fold's line is the constant predictor wearing a slope of zero, so the two errors
    # are the same number rather than merely close, and only a strict comparison keeps the
    # line from claiming it predicts anything.
    assert result["fold_lines"][0] == [0.0, 5.5]
    assert result["crossfit_median_absolute_error"] == 0.5
    assert (
        result["crossfit_median_absolute_error"]
        == result["constant_median_absolute_error"]
    )
    assert result["predictive"] is False


def test_a_constant_reference_leaves_the_relationship_unavailable():
    references = {1: 2.0, 2: 2.0, 3: 2.0, 4: 2.0}
    responsives = {1: 5.0, 2: 6.0, 3: 7.0, 4: 8.0}
    result = clean_relationship(
        references, responsives, folds=assign_folds(references)
    )
    assert result["final_slope"] is None
    assert result["final_offset"] is None
    assert result["crossfit_median_absolute_error"] is None
    assert result["predictive"] is False
    assert result["pearson"] is None


def test_every_fold_line_excludes_its_own_fold():
    references = {image_id: float(image_id) for image_id in (1, 2, 3, 4)}
    responsives = {image_id: 2.0 * float(image_id) + 1.0 for image_id in (1, 2, 3, 4)}
    folds = assign_folds(references)
    # image 1 alone is dragged off the line. Its own fold is fitted on images 2, 3 and 4,
    # which are exactly collinear, so that fold's line must be (2.0, 1.0) exactly.
    responsives[1] = 0.0
    result = clean_relationship(references, responsives, folds=folds)
    assert result["fold_lines"][folds[1]] == pytest.approx([2.0, 1.0])
    # and the line that fold would have got had it seen image 1 is a different line, which is
    # what makes the assertion above able to fail at all
    assert robust_line([1.0, 2.0, 3.0, 4.0], [0.0, 5.0, 7.0, 9.0]) == pytest.approx(
        (2.5, -0.75)
    )


def test_the_final_line_is_fitted_on_every_clean_image():
    references = {image_id: float(image_id) for image_id in (1, 2, 3, 4)}
    responsives = {1: 0.0, 2: 5.0, 3: 7.0, 4: 9.0}
    result = clean_relationship(references, responsives, folds=assign_folds(references))
    # all six pairwise slopes -- [2, 2, 2, 3, 3.5, 5] -- with median 2.5, and an intercept of
    # median([-2.5, 0, -0.5, -1]) = -0.75. Drop any single image and it moves: without
    # image 1 it is (2.0, 1.0), which is what each of the fold lines is.
    assert result["final_slope"] == pytest.approx(2.5)
    assert result["final_offset"] == pytest.approx(-0.75)


def test_a_fold_outside_the_range_is_refused_rather_than_never_held_out():
    references = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
    responsives = {image_id: 2.0 * value + 1.0 for image_id, value in references.items()}
    folds = assign_folds(references)
    folds[4] = FOLD_COUNT  # one past the last fold the cross-fitting loop visits
    # unguarded, image 4 is never held out and never predicted: it stays in every training
    # set and the constant error median silently moves from 3.0 to 2.0.
    with pytest.raises(ValueError, match="every fold must lie in range"):
        clean_relationship(references, responsives, folds=folds)
