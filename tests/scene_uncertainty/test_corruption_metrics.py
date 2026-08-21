"""Four questions about one candidate's six-severity curve, and the ways each can be faked.

`corruption_metrics` is pure arithmetic over small arrays, so almost every test here is a
number checked by hand rather than a shape or a key set. What the tests are really guarding is
that the four questions stay *separate*:

* a signed trend and its absolute strength are two numbers, not one. Every trend test asserts
  both, and the decreasing curve is the one that fails if `absolute_spearman` were ever filled
  in from `signed_spearman` unchanged;
* a flat curve and a missing curve are two facts, not one. `test_flat_and_unmeasured_are_
  different_facts` runs both through and asserts one is `0.0` and the other is `None`, so an
  implementation that coerced a `nan` score into "no trend" fails rather than reporting a
  plausible zero;
* orientation is chosen once from a group and then applied unchanged. Two tests exist only to
  fail a mean-based chooser and a chooser that reads non-finite entries as zero, because both
  produce a legal direction from data that should have produced a different one -- or none;
* AUROC is a *ranking* statistic over two groups, and its tie handling is the part a wrong
  implementation still gets nearly right. `test_binary_auroc_averages_tied_ranks` pins 0.875
  on a case where `min` ranking gives 0.75 and `max`/`ordinal` ranking give 1.0.

`test_constant_curve_is_short_circuited_without_a_scipy_warning` turns `ConstantInputWarning`
into an error rather than merely reading the returned dictionary, because a constant curve
routed through `spearmanr` returns `nan` and would be reported as `flat` by any implementation
that mapped `nan` to zero -- the warning is the only visible difference between the two.
"""

import ast
import warnings
from pathlib import Path

import pytest
from scipy.stats import ConstantInputWarning

from src.scene_uncertainty import corruption_metrics as metrics_module
from src.scene_uncertainty.corruption_metrics import (
    binary_auroc,
    choose_orientation,
    complete_trend_metrics,
    oriented_curve_metrics,
    severity_aurocs,
)


# --- per-image trend: sign and strength ---------------------------------------------


def test_complete_trend_metrics_keeps_sign_and_absolute_strength():
    result = complete_trend_metrics(range(6), [0, 1, 2, 3, 4, 5])
    assert result == {
        "fully_measured": True,
        "signed_spearman": 1.0,
        "absolute_spearman": 1.0,
        "direction": "increasing",
    }


def test_decreasing_curve_keeps_a_negative_sign_and_a_positive_strength():
    """The one case where copying `signed_spearman` into `absolute_spearman` is visible."""
    result = complete_trend_metrics(range(6), [5, 4, 3, 2, 1, 0])
    assert result == {
        "fully_measured": True,
        "signed_spearman": -1.0,
        "absolute_spearman": 1.0,
        "direction": "decreasing",
    }


def test_tied_scores_use_tie_aware_spearman():
    """Two equal middle scores, so a strict-monotonic check would still read 1.0."""
    result = complete_trend_metrics(range(6), [0, 1, 1, 2, 3, 4])
    assert result["signed_spearman"] == pytest.approx(0.9856107606091623)
    assert result["absolute_spearman"] == pytest.approx(0.9856107606091623)
    assert result["direction"] == "increasing"
    assert result["fully_measured"] is True


# --- per-image trend: flat is a measurement, unmeasured is not ----------------------


def test_complete_trend_metrics_marks_constant_curve_flat():
    result = complete_trend_metrics(range(6), [3.0] * 6)
    assert result == {
        "fully_measured": True,
        "signed_spearman": 0.0,
        "absolute_spearman": 0.0,
        "direction": "flat",
    }


def test_constant_curve_is_short_circuited_without_a_scipy_warning():
    """Six identical finite scores must never reach `spearmanr`.

    Routed through it they would produce `nan` plus a `ConstantInputWarning`; the returned
    dictionary alone cannot tell that apart from the declared zero, so the warning is what is
    asserted on.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConstantInputWarning)
        result = complete_trend_metrics(range(6), [0.25] * 6)
    assert result["signed_spearman"] == 0.0
    assert result["direction"] == "flat"


def test_incomplete_curve_is_unmeasured_not_flat():
    result = complete_trend_metrics([0, 1, 2, 4, 5], [0, 1, 2, 4, 5])
    assert result["fully_measured"] is False
    assert result["signed_spearman"] is None
    assert result["absolute_spearman"] is None
    assert result["direction"] == "unmeasured"


@pytest.mark.parametrize("missing", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_score_is_unmeasured_not_flat(missing):
    result = complete_trend_metrics(range(6), [0.0, 1.0, 2.0, 3.0, 4.0, missing])
    assert result["fully_measured"] is False
    assert result["signed_spearman"] is None
    assert result["absolute_spearman"] is None
    assert result["direction"] == "unmeasured"


def test_flat_and_unmeasured_are_different_facts():
    """The same five leading scores; only the sixth separates a zero trend from no trend."""
    flat = complete_trend_metrics(range(6), [3.0] * 6)
    unmeasured = complete_trend_metrics(range(6), [3.0, 3.0, 3.0, 3.0, 3.0, float("nan")])
    assert (flat["fully_measured"], unmeasured["fully_measured"]) == (True, False)
    assert flat["absolute_spearman"] == 0.0
    assert unmeasured["absolute_spearman"] is None
    assert (flat["direction"], unmeasured["direction"]) == ("flat", "unmeasured")


# --- per-image trend: the severity axis is an identity, not a count -----------------


def test_severities_out_of_order_are_unmeasured():
    """A descending axis paired with a descending curve correlates +1 if the axis is trusted."""
    result = complete_trend_metrics([5, 4, 3, 2, 1, 0], [5, 4, 3, 2, 1, 0])
    assert result["fully_measured"] is False
    assert result["direction"] == "unmeasured"
    assert result["signed_spearman"] is None


def test_severities_outside_zero_through_five_are_unmeasured():
    result = complete_trend_metrics([1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5])
    assert result["fully_measured"] is False
    assert result["direction"] == "unmeasured"
    assert result["signed_spearman"] is None


def test_a_seventh_severity_is_unmeasured():
    """A longer ladder is refused for the same reason a shorter one is: it is not this ladder."""
    result = complete_trend_metrics(range(7), [0, 1, 2, 3, 4, 5, 6])
    assert result["fully_measured"] is False
    assert result["direction"] == "unmeasured"
    assert result["signed_spearman"] is None


# --- locked orientation --------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "expected"),
    [([0.8, 0.9], 1), ([-0.8, -0.2], -1), ([-0.8, 0.8], None)],
)
def test_choose_orientation_uses_one_group_median(values, expected):
    assert choose_orientation(values) == expected


def test_choose_orientation_uses_the_median_not_the_mean():
    """Three weak negatives and two strong positives: median -0.1, mean +0.24."""
    assert choose_orientation([-0.3, -0.2, -0.1, 0.9, 0.9]) == -1


def test_choose_orientation_drops_non_finite_values_rather_than_zeroing_them():
    """Two unmeasured images and one strong positive is a positive candidate.

    Reading the two as `0.0` would put the median at zero and call the candidate unorientable.
    """
    assert choose_orientation([float("nan"), float("nan"), 0.9]) == 1


@pytest.mark.parametrize(
    "values",
    [[], [float("nan")], [float("nan"), float("inf"), float("-inf")]],
)
def test_choose_orientation_returns_none_when_no_finite_value_remains(values):
    assert choose_orientation(values) is None


def test_choose_orientation_returns_none_for_a_zero_median():
    assert choose_orientation([0.0, 0.0, 0.0]) is None


# --- AUROC across scenes -------------------------------------------------------------


def test_binary_auroc_is_tie_aware_and_uses_locked_orientation():
    assert binary_auroc([0, 1], [2, 3], orientation=1) == pytest.approx(1.0)
    assert binary_auroc([0, 1], [2, 3], orientation=-1) == pytest.approx(0.0)
    assert binary_auroc([1, 1], [1, 1], orientation=1) == pytest.approx(0.5)


def test_binary_auroc_averages_tied_ranks():
    """One clean and one corrupted score are equal, so the tie must split.

    `min` ranking scores this 0.75 and both `max` and `ordinal` ranking score it 1.0, so the
    value below is only reachable with averaged ranks.
    """
    assert binary_auroc([0.0, 1.0], [1.0, 2.0], orientation=1) == pytest.approx(0.875)


def test_binary_auroc_handles_unequal_group_sizes():
    """One corrupted score sitting between two clean ones ranks above exactly half of them."""
    assert binary_auroc([0.0, 10.0], [5.0], orientation=1) == pytest.approx(0.5)
    assert binary_auroc([0.0], [1.0, 2.0, 3.0], orientation=1) == pytest.approx(1.0)


@pytest.mark.parametrize("orientation", [0, 2, -2, 0.5, None, "1"])
def test_binary_auroc_rejects_an_orientation_other_than_plus_or_minus_one(orientation):
    with pytest.raises(ValueError, match="orientation must be"):
        binary_auroc([0.0, 1.0], [2.0, 3.0], orientation=orientation)


@pytest.mark.parametrize(
    ("clean", "corrupted"),
    [([], [1.0, 2.0]), ([1.0, 2.0], []), ([], [])],
)
def test_binary_auroc_rejects_an_empty_group(clean, corrupted):
    with pytest.raises(ValueError, match="non-empty"):
        binary_auroc(clean, corrupted, orientation=1)


@pytest.mark.parametrize(
    ("clean", "corrupted"),
    [
        ([0.0, float("nan")], [1.0, 2.0]),
        ([0.0, 1.0], [2.0, float("inf")]),
        ([float("-inf")], [1.0]),
    ],
)
def test_binary_auroc_rejects_non_finite_scores(clean, corrupted):
    with pytest.raises(ValueError, match="must be finite"):
        binary_auroc(clean, corrupted, orientation=1)


# --- the oriented curve's deployment checks ------------------------------------------


def test_oriented_curve_metrics_report_deployment_checks():
    result = oriented_curve_metrics([5, 4, 3, 2, 1, 0], orientation=-1)
    assert result["adjacent_consistency"] == 1.0
    assert result["max_blur_above_clean"] is True


def test_oriented_curve_metrics_read_the_curve_against_the_locked_direction():
    """The same curve, the other direction: nothing about it is direction-free."""
    result = oriented_curve_metrics([5, 4, 3, 2, 1, 0], orientation=1)
    assert result["adjacent_consistency"] == 0.0
    assert result["max_blur_above_clean"] is False


def test_adjacent_consistency_is_a_fraction_of_steps_not_a_flag():
    """One step of five runs backwards, so neither 1.0 nor 0.0 is the answer."""
    forward = oriented_curve_metrics([0, 1, 0, 1, 2, 3], orientation=1)
    assert forward["adjacent_consistency"] == pytest.approx(0.8)
    assert forward["max_blur_above_clean"] is True
    backward = oriented_curve_metrics([0, 1, 0, 1, 2, 3], orientation=-1)
    assert backward["adjacent_consistency"] == pytest.approx(0.2)
    assert backward["max_blur_above_clean"] is False


@pytest.mark.parametrize("orientation", [-1, 1])
def test_a_flat_curve_never_steps_backwards_and_never_clears_clean(orientation):
    """`adjacent_consistency` counts a zero step as consistent; the endpoint check is strict."""
    result = oriented_curve_metrics([2.0] * 6, orientation=orientation)
    assert result["adjacent_consistency"] == 1.0
    assert result["max_blur_above_clean"] is False


@pytest.mark.parametrize(
    "scores",
    [
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [0.0, 1.0, 2.0, 3.0, 4.0, float("nan")],
        [float("inf"), 1.0, 2.0, 3.0, 4.0, 5.0],
    ],
)
def test_oriented_curve_metrics_report_none_for_an_unusable_curve(scores):
    assert oriented_curve_metrics(scores, orientation=1) == {
        "adjacent_consistency": None,
        "max_blur_above_clean": None,
    }


@pytest.mark.parametrize("orientation", [0, 2, -2, 0.5, None, "1"])
def test_oriented_curve_metrics_reject_an_orientation_other_than_plus_or_minus_one(orientation):
    with pytest.raises(ValueError, match="orientation must be"):
        oriented_curve_metrics([0, 1, 2, 3, 4, 5], orientation=orientation)


# --- the five severity AUROCs and their macro ----------------------------------------


SEPARATING_SCORES = {
    0: [0.0, 1.0],
    1: [0.0, 1.0],
    2: [0.5, 2.0],
    3: [2.0, 3.0],
    4: [4.0, 5.0],
    5: [6.0, 7.0],
}
"""Clean scores repeated at severity 1 and pulled apart from there.

Severity 1 is therefore chance-level and severities 3 to 5 are perfectly separated, which is
the shape the macro average is supposed to be unable to hide.
"""


def test_severity_aurocs_keep_every_component_and_average_them():
    values, macro = severity_aurocs(SEPARATING_SCORES, orientation=1)
    assert values == {
        1: pytest.approx(0.5),
        2: pytest.approx(0.75),
        3: pytest.approx(1.0),
        4: pytest.approx(1.0),
        5: pytest.approx(1.0),
    }
    assert macro == pytest.approx(0.85)
    assert macro == pytest.approx(sum(values.values()) / 5)


def test_severity_aurocs_pass_the_locked_orientation_through():
    """Reading the same scores backwards mirrors every component about chance level."""
    values, macro = severity_aurocs(SEPARATING_SCORES, orientation=-1)
    assert values == {
        1: pytest.approx(0.5),
        2: pytest.approx(0.25),
        3: pytest.approx(0.0),
        4: pytest.approx(0.0),
        5: pytest.approx(0.0),
    }
    assert macro == pytest.approx(0.15)


UNEVEN_COVERAGE_SCORES = {
    0: [0.0, 1.0, 2.0, 3.0, 4.0],
    1: [-10.0],
    2: [10.0, 11.0, 12.0, 13.0, 14.0],
    3: [10.0, 11.0, 12.0, 13.0, 14.0],
    4: [10.0, 11.0, 12.0, 13.0, 14.0],
    5: [10.0, 11.0, 12.0, 13.0, 14.0],
}
"""One thinly covered severity that reads the *other* way, and four fully covered perfect ones.

`_candidate_metrics` drops a non-finite score from its severity's group and keeps the rest of
that image's curve, so the five comparisons are routinely made over groups of different sizes
and this shape is reachable in published output rather than hypothetical. Severity 1 holds one
image against five clean ones and ranks below all of them, so its AUROC is `0.0` while the
other four are `1.0` -- the widest gap a weighting can move.
"""


def test_the_macro_average_weights_the_five_severities_equally():
    """The docstring's own claim, which every equal-coverage fixture leaves unpinned.

    "Weighting by group size would let whichever severity happened to retain the most scored
    images dominate the ranking." Every other fixture in this module gives all six severities
    the same number of scores, so a group-size-weighted average is numerically identical to the
    plain mean in all of them and the sentence is unenforced. Here the thin severity is the one
    holding the macro down: the plain mean is 0.8, weighting by the corrupted group sizes reads
    0.952 and weighting by clean-plus-corrupted reads 0.870, so a candidate that fails at mild
    blur would be published as a near-perfect detector.
    """
    values, macro = severity_aurocs(UNEVEN_COVERAGE_SCORES, orientation=1)

    assert values == {1: 0.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}
    assert macro == pytest.approx(0.8)
    assert macro == pytest.approx(sum(values.values()) / 5)
    sizes = [len(UNEVEN_COVERAGE_SCORES[severity]) for severity in range(1, 6)]
    assert len(set(sizes)) > 1, "the fixture must actually have uneven coverage"
    assert macro != pytest.approx(
        sum(value * size for value, size in zip(values.values(), sizes)) / sum(sizes)
    )


@pytest.mark.parametrize("severity", [0, 1, 5])
def test_severity_aurocs_require_every_severity_zero_through_five(severity):
    incomplete = {key: value for key, value in SEPARATING_SCORES.items() if key != severity}
    with pytest.raises(ValueError, match="severity 0 and corrupted severities"):
        severity_aurocs(incomplete, orientation=1)


def test_severity_aurocs_reject_an_unexpected_extra_severity():
    with pytest.raises(ValueError, match="severity 0 and corrupted severities"):
        severity_aurocs({**SEPARATING_SCORES, 6: [8.0, 9.0]}, orientation=1)


# --- a pure module --------------------------------------------------------------------


def test_the_metric_module_reaches_no_file_no_model_and_no_sibling():
    """These are arithmetic answers about arrays a caller already holds.

    Anything read from disk, from torch or from another `scene_uncertainty` module would make
    a number here depend on state the caller cannot see in its own inputs.
    """
    source = Path(metrics_module.__file__).read_text()
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "no relative import may reach a sibling module"
            imported.add(node.module or "")
    assert {name.split(".")[0] for name in imported} <= {
        "__future__", "collections", "numpy", "scipy",
    }
