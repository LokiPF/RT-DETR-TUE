"""Hand arithmetic against `scene_uncertainty.contrast_diagnostics`, one claim per test.

Four fixtures here are chosen rather than obvious, because the obvious version of each passes
against broken code:

`test_drift_is_measured_against_each_image_own_severity_zero` uses baselines `1, 2, 3, 10`.
Symmetric baselines -- `1, 2, 3, 4`, which is what every other fixture in this file uses --
cannot see the bug the test is named for: with a uniform drift the median of
`value - group mean` equals the median of `value - own clean value` exactly, so measuring the
drift against the group's severity-zero mean reproduces `0.2` and `0.5` unchanged. Asymmetry
separates the two, and the absolute-drift assertion alongside it separates the group *median*
variant, which asymmetry alone does not.

`test_stability_to_spread_divides_drift_by_the_clean_interquartile_range` gives the four images
*different* per-severity steps. When every image drifts by the same amount the spread at
severity 2 is a translate of the spread at severity 0, so its interquartile range is also
`1.5`, and dividing by the wrong severity's spread returns the right answer anyway.

`test_every_fold_line_excludes_its_own_fold` uses four images, not ten. Theil-Sen is robust
enough that one dragged-off image out of ten contributes 9 of 45 pairwise slopes and the
median stays at `2.0` whether or not the fold is held out -- the mutation that fits each fold
on the whole roster survives that fixture untouched. With four images the held image
contributes 3 of 6 pairs, which does move the median.

`test_an_equal_error_line_is_not_predictive` alternates the responsive value between `5.0` and
`6.0` so that every fold's median pairwise slope is exactly zero and its intercept is exactly
the median of the training responsives -- which makes the fitted line and the constant
predictor the same predictor, and their two errors exactly equal. Nothing in the two fixtures
above it distinguishes `crossfit_error < constant_error` from `<=`.
"""

import pytest

from src.scene_uncertainty.contrast_diagnostics import (
    between_image_spread,
    clean_relationship,
    within_image_drift,
)
from src.scene_uncertainty.contrast_scores import assign_folds, robust_line

SEVERITIES = range(6)


def flat_curves():
    """Four images whose reference never moves: baselines 1.0, 2.0, 3.0, 4.0."""
    return {
        image_id: {severity: float(image_id) for severity in SEVERITIES}
        for image_id in (1, 2, 3, 4)
    }


def drifting_curves(step=0.1):
    """The same four baselines, each rising by `step` per severity."""
    return {
        image_id: {severity: image_id + step * severity for severity in SEVERITIES}
        for image_id in (1, 2, 3, 4)
    }


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
    # image 2 moves by two units in the last place -- about 8.9e-16, far below any tolerance
    # anyone would write down, and still a reference that did not hold still.
    curves = {
        1: {severity: 1.0 for severity in SEVERITIES},
        2: {severity: 2.0 + (1e-15 if severity else 0.0) for severity in SEVERITIES},
    }
    row = within_image_drift(curves)["by_severity"][1]
    assert row["median_absolute_drift"] != 0.0
    assert row["zero_drift_fraction"] == 0.5


def test_signed_and_absolute_drift_differ_when_images_move_opposite_ways():
    curves = {
        1: {severity: 1.0 + 0.1 * severity for severity in SEVERITIES},
        2: {severity: 2.0 - 0.1 * severity for severity in SEVERITIES},
    }
    row = within_image_drift(curves)["by_severity"][4]
    assert row["median_signed_drift"] == pytest.approx(0.0)
    assert row["median_absolute_drift"] == pytest.approx(0.4)


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


def test_a_non_predictive_clean_relationship_is_labelled_so():
    references = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
    responsives = {1: 5.0, 2: 5.1, 3: 4.9, 4: 5.0}
    result = clean_relationship(
        references, responsives, folds=assign_folds(references)
    )
    assert result["crossfit_median_absolute_error"] == pytest.approx(0.125)
    assert result["constant_median_absolute_error"] == pytest.approx(0.05)
    assert result["predictive"] is False


def test_an_equal_error_line_is_not_predictive():
    references = {image_id: float(image_id) for image_id in range(1, 11)}
    # Alternating responsives make every fold's median pairwise slope exactly zero, so each
    # fold's line collapses onto the median of its own training responsives -- which is the
    # constant predictor. The two errors are then equal by construction, at 0.5 each, and
    # only a strict comparison keeps the line from claiming it predicts anything.
    responsives = {image_id: 5.0 if image_id % 2 == 0 else 6.0 for image_id in references}
    result = clean_relationship(
        references, responsives, folds=assign_folds(references)
    )
    assert result["crossfit_median_absolute_error"] == pytest.approx(0.5)
    assert result["constant_median_absolute_error"] == pytest.approx(0.5)
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
