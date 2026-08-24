import pytest

from differential_uncertainty.config import ExperimentConfig
from differential_uncertainty.evaluation import (
    binary_auroc,
    evaluate_rows,
    oriented_curve_metrics,
    paired_macro_bootstrap,
    summarize_series,
    trend_metrics,
)


def _rows(field, curves):
    return [
        {"image_id": image_id, "severity": severity, field: value}
        for image_id, curve in curves.items()
        for severity, value in enumerate(curve)
    ]


def test_trend_metrics_distinguish_increasing_constant_and_incomplete():
    assert trend_metrics([0, 1, 2, 3, 4, 5])["signed_spearman"] == 1.0
    assert trend_metrics([4, 4, 4, 4, 4, 4])["signed_spearman"] == 0.0
    with pytest.raises(ValueError, match="six finite"):
        trend_metrics([0, 1, 2])


def test_tie_aware_auroc_matches_the_pair_count_example():
    assert binary_auroc([0, 1], [1, 2], orientation=1) == 0.875
    assert binary_auroc([0, 1], [1, 2], orientation=-1) == 0.125


def test_oriented_curve_metrics_match_five_hand_counted_steps():
    assert oriented_curve_metrics([0, 2, 1, 3, 3, 4], orientation=1) == {
        "adjacent_consistency": 0.8,
        "strongest_blur_above_clean": True,
    }
    assert oriented_curve_metrics([5, 4, 4, 2, 3, 0], orientation=-1) == {
        "adjacent_consistency": 0.8,
        "strongest_blur_above_clean": True,
    }
    with pytest.raises(ValueError, match="orientation must be an integer"):
        oriented_curve_metrics([0, 1, 2, 3, 4, 5], orientation=1.0)


def test_binary_auroc_requires_an_integral_non_boolean_orientation():
    for orientation in (True, 1.0, "1"):
        with pytest.raises(ValueError, match="orientation must be an integer"):
            binary_auroc([0], [1], orientation=orientation)


def test_series_summary_reports_every_severity_and_curve_check():
    field = "score"
    rows = _rows(field, {"a": [0, 1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5, 6]})
    summary = summarize_series(rows, field, orientation=1)
    assert summary["median_signed_spearman"] == 1.0
    assert summary["oriented_adjacent_consistency"] == 1.0
    assert summary["strongest_blur_above_clean_rate"] == 1.0
    assert set(summary["auroc_by_severity"]) == {1, 2, 3, 4, 5}
    assert summary["auroc_by_severity"] == pytest.approx(
        {1: 0.875, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}
    )
    assert summary["macro_auroc"] == pytest.approx(0.975)
    assert summary["severity_statistics"][0] == {
        "count": 2,
        "mean": 0.5,
        "median": 0.5,
        "q25": 0.25,
        "q75": 0.75,
    }


@pytest.mark.parametrize("severity", [True, 1.0, "1"])
def test_series_rows_require_integral_non_boolean_severities(severity):
    rows = _rows("score", {"a": [0, 1, 2, 3, 4, 5]})
    rows[1]["severity"] = severity
    with pytest.raises(ValueError, match="severity must be an integer"):
        summarize_series(rows, "score", orientation=1)


def test_series_needs_exactly_six_unique_severities_for_every_image():
    complete = _rows("score", {"a": [0, 1, 2, 3, 4, 5]})
    with pytest.raises(ValueError, match="severities 0 through 5"):
        summarize_series(complete[:-1], "score", orientation=1)

    wrong_identity = [dict(row) for row in complete]
    wrong_identity[-1]["severity"] = 6
    with pytest.raises(ValueError, match="severities 0 through 5"):
        summarize_series(wrong_identity, "score", orientation=1)

    duplicate = [dict(row) for row in complete]
    duplicate.append(dict(complete[3]))
    with pytest.raises(ValueError, match="duplicate score row"):
        summarize_series(duplicate, "score", orientation=1)


def test_series_rejects_nonfinite_scores():
    rows = _rows("score", {"a": [0, 1, 2, float("nan"), 4, 5]})
    with pytest.raises(ValueError, match="finite score"):
        summarize_series(rows, "score", orientation=1)


def test_empty_series_is_an_explicit_error():
    with pytest.raises(ValueError, match="at least one complete image"):
        summarize_series([], "score", orientation=1)


def test_paired_bootstrap_of_a_series_against_itself_is_exactly_zero():
    rows = _rows("candidate", {"a": [0, 1, 2, 3, 4, 5], "b": [1, 1, 2, 2, 3, 3]})
    for row in rows:
        row["control"] = row["candidate"]
    result = paired_macro_bootstrap(
        rows,
        "candidate",
        "control",
        candidate_orientation=1,
        control_orientation=1,
        samples=100,
        seed=9,
    )
    assert result["point_difference"] == 0.0
    assert result["ci_low"] == result["ci_high"] == 0.0


@pytest.mark.parametrize(
    "candidate_orientation, control_orientation",
    [(1.0, 1), (1, True), (1, "-1")],
)
def test_paired_bootstrap_requires_integral_non_boolean_orientations(
    candidate_orientation, control_orientation
):
    rows = _rows("candidate", {"a": [0, 1, 2, 3, 4, 5]})
    for row in rows:
        row["control"] = row["candidate"]
    with pytest.raises(ValueError, match="orientation must be an integer"):
        paired_macro_bootstrap(
            rows,
            "candidate",
            "control",
            candidate_orientation=candidate_orientation,
            control_orientation=control_orientation,
            samples=10,
            seed=3,
        )


def test_paired_bootstrap_requires_both_finite_scores_on_every_row():
    rows = _rows("candidate", {"a": [0, 1, 2, 3, 4, 5]})
    for row in rows:
        row["control"] = row["candidate"]
    del rows[3]["control"]
    with pytest.raises(ValueError, match="candidate and control on every row"):
        paired_macro_bootstrap(
            rows, "candidate", "control",
            candidate_orientation=1, control_orientation=1,
            samples=10, seed=3,
        )

    rows[3]["control"] = float("inf")
    with pytest.raises(ValueError, match="finite score"):
        paired_macro_bootstrap(
            rows, "candidate", "control",
            candidate_orientation=1, control_orientation=1,
            samples=10, seed=3,
        )


@pytest.mark.parametrize(
    "samples, seed, message",
    [(True, 3, "sample count must be a positive integer"),
     (10.0, 3, "sample count must be a positive integer"),
     (10, False, "seed must be an integer"),
     (10, 3.0, "seed must be an integer")],
)
def test_paired_bootstrap_requires_exact_integer_samples_and_seed(samples, seed, message):
    rows = _rows("candidate", {"a": [0, 1, 2, 3, 4, 5]})
    for row in rows:
        row["control"] = row["candidate"]
    with pytest.raises(ValueError, match=message):
        paired_macro_bootstrap(
            rows, "candidate", "control",
            candidate_orientation=1, control_orientation=1,
            samples=samples, seed=seed,
        )


def test_paired_bootstrap_is_deterministic_and_reconciles_its_point_estimate():
    rows = _rows(
        "candidate",
        {
            "a": [0, 0, 1, 1, 2, 2],
            "b": [1, 1, 2, 2, 3, 4],
            "c": [2, 2, 2, 3, 4, 5],
        },
    )
    controls = {
        "a": [0, 0, 0, 0, 1, 1],
        "b": [1, 1, 1, 2, 2, 2],
        "c": [2, 2, 3, 3, 3, 3],
    }
    for row in rows:
        row["control"] = controls[row["image_id"]][row["severity"]]

    expected = (
        summarize_series(rows, "candidate", orientation=1)["macro_auroc"]
        - summarize_series(rows, "control", orientation=1)["macro_auroc"]
    )
    first = paired_macro_bootstrap(
        rows, "candidate", "control",
        candidate_orientation=1, control_orientation=1,
        samples=200, seed=81,
    )
    second = paired_macro_bootstrap(
        list(reversed(rows)), "candidate", "control",
        candidate_orientation=1, control_orientation=1,
        samples=200, seed=81,
    )

    assert first == second
    assert first["point_difference"] == pytest.approx(expected)


def test_evaluate_rows_keeps_two_relative_gaps_and_two_persistence_controls():
    curves = {
        "a": [0, 1, 2, 3, 4, 5],
        "b": [1, 2, 3, 4, 5, 6],
    }
    rows = _rows("persistence_relative_gap", curves)
    for row in rows:
        severity = row["severity"]
        row.update(
            {
                "confidence_relative_gap": float(5 - severity),
                "persistence_responsive": float(severity + 2),
                "persistence_reference": float(8 - severity),
                "confidence_responsive": 123.0,
                "confidence_reference": 456.0,
            }
        )
    config = ExperimentConfig.for_tests(bootstrap_samples=40, bootstrap_seed=7)

    result = evaluate_rows(rows, config)

    expected_fields = {
        "persistence_relative_gap",
        "confidence_relative_gap",
        "persistence_responsive",
        "persistence_reference",
    }
    assert set(result["series"]) == expected_fields
    assert "confidence_responsive" not in result["series"]
    assert "confidence_reference" not in result["series"]
    assert {
        field: summary["orientation"] for field, summary in result["series"].items()
    } == {
        "persistence_relative_gap": 1,
        "confidence_relative_gap": -1,
        "persistence_responsive": 1,
        "persistence_reference": -1,
    }
    comparisons = result["bootstrap_comparisons"]
    assert [item["candidate"] for item in comparisons] == [
        "persistence_relative_gap"
    ] * 3
    assert [item["control"] for item in comparisons] == [
        "confidence_relative_gap",
        "persistence_responsive",
        "persistence_reference",
    ]
