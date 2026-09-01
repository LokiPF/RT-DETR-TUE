import numpy as np
import pytest

import differential_uncertainty.evaluation as evaluation
from differential_uncertainty.corruptions import CORRUPTION_NAMES
from differential_uncertainty.evaluation import binary_auroc, evaluate_scores


FAMILIES = CORRUPTION_NAMES


def _rows(families=FAMILIES, image_ids=("a", "b", "c")):
    rows = []
    for family_index, family in enumerate(families):
        for image_index, image_id in enumerate(image_ids):
            for severity in (0, 4, 5):
                clean = float(image_index)
                corrupted = clean + (20.0 + family_index + severity if severity else 0.0)
                rows.append({
                    "image_id": image_id,
                    "corruption": family,
                    "severity": severity,
                    "fingerprint": corrupted,
                    "confidence": corrupted + 0.1,
                    "entropy": corrupted + 0.2,
                })
    return rows


def test_binary_auroc_is_tie_correct_and_uses_larger_as_corrupted():
    assert binary_auroc([0, 1], [1, 2]) == pytest.approx(0.875)
    assert binary_auroc([1, 2], [0, 1]) == pytest.approx(0.125)


@pytest.mark.parametrize("clean, corrupted", [
    ([], [1]), ([1], []), ([float("nan")], [1]), ([1], [float("inf")]),
    ([[1]], [2]), (1, [2]),
])
def test_binary_auroc_rejects_nonfinite_empty_or_non_vector_inputs(clean, corrupted):
    with pytest.raises(ValueError, match="one-dimensional"):
        binary_auroc(clean, corrupted)


def test_evaluate_scores_returns_all_tasks_in_supplied_family_and_severity_order():
    result = evaluate_scores(_rows(), FAMILIES, samples=20, seed=7)

    assert set(result) == {"tasks", "aggregate", "comparisons"}
    assert len(result["tasks"]) == 38
    assert [(task["corruption"], task["severity"]) for task in result["tasks"]] == [
        (family, severity) for family in FAMILIES for severity in (4, 5)
    ]
    assert all(set(task) == {
        "corruption", "severity", "fingerprint_auroc", "confidence_auroc", "entropy_auroc"
    } for task in result["tasks"])


@pytest.mark.parametrize("families", [
    FAMILIES[:1],
    (FAMILIES[1], FAMILIES[0], *FAMILIES[2:]),
])
def test_evaluate_scores_requires_the_exact_fixed_family_roster_in_order(families):
    with pytest.raises(ValueError, match="fixed corruption roster"):
        evaluate_scores(_rows(), families, samples=20, seed=7)


def test_evaluate_scores_accepts_integer_valued_real_scores():
    rows = _rows()
    for row in rows:
        for method in ("fingerprint", "confidence", "entropy"):
            row[method] = int(row[method])

    assert len(evaluate_scores(rows, FAMILIES, samples=10, seed=2)["tasks"]) == 38


def test_evaluate_scores_orients_every_method_as_larger_means_more_corrupted():
    result = evaluate_scores(_rows(), FAMILIES, samples=10, seed=2)
    for task in result["tasks"]:
        assert task["fingerprint_auroc"] == 1.0
        assert task["confidence_auroc"] == 1.0
        assert task["entropy_auroc"] == 1.0


def test_aggregate_is_an_equal_weight_task_mean_not_a_pooled_score():
    rows = _rows(image_ids=("a", "b"))
    for row in rows:
        if row["corruption"] not in FAMILIES[:9] and row["severity"] in (4, 5):
            row["fingerprint"] = float(row["image_id"] == "b")
    result = evaluate_scores(rows, FAMILIES, samples=20, seed=3)

    assert result["aggregate"]["fingerprint"] == pytest.approx(28 / 38)


@pytest.mark.parametrize("mutate, message", [
    (lambda rows: rows.append(dict(rows[0])), "duplicate"),
    (lambda rows: rows.pop(), "exactly severities"),
    (lambda rows: rows[0].update({"unexpected": 1}), "exactly"),
    (lambda rows: rows[0].pop("entropy"), "exactly"),
    (lambda rows: rows[0].update({"severity": True}), "severity"),
    (lambda rows: rows[0].update({"fingerprint": True}), "fingerprint"),
    (lambda rows: rows[0].update({"fingerprint": "1"}), "fingerprint"),
])
def test_evaluate_scores_validates_the_exact_row_contract(mutate, message):
    rows = _rows()
    mutate(rows)
    with pytest.raises(ValueError, match=message):
        evaluate_scores(rows, FAMILIES, samples=10, seed=0)


def test_evaluate_scores_requires_the_same_nonempty_image_roster_for_every_family():
    rows = _rows()
    for row in rows:
        if row["corruption"] == FAMILIES[1] and row["image_id"] == "c":
            row["image_id"] = "different"
    with pytest.raises(ValueError, match="same image roster"):
        evaluate_scores(rows, FAMILIES, samples=10, seed=0)


@pytest.mark.parametrize("samples, seed", [(True, 0), (2.0, 0), (1, True), (1, -1)])
def test_evaluate_scores_requires_positive_integer_samples_and_nonnegative_integer_seed(samples, seed):
    with pytest.raises(ValueError):
        evaluate_scores(_rows(), FAMILIES, samples=samples, seed=seed)


def test_paired_bootstrap_is_deterministic_and_comparison_point_matches_aggregates():
    rows = _rows(image_ids=("a", "b", "c", "d"))
    for row in rows:
        row["confidence"] = row["fingerprint"] - float(row["image_id"] in {"a", "b"})
        row["entropy"] = row["fingerprint"] - float(row["severity"] == 5)
    first = evaluate_scores(rows, FAMILIES, samples=101, seed=8)
    second = evaluate_scores(list(reversed(rows)), FAMILIES, samples=101, seed=8)

    assert first == second
    assert first["comparisons"]["fingerprint_minus_confidence"]["point"] == pytest.approx(
        first["aggregate"]["fingerprint"] - first["aggregate"]["confidence"]
    )
    assert first["comparisons"]["fingerprint_minus_entropy"]["point"] == pytest.approx(
        first["aggregate"]["fingerprint"] - first["aggregate"]["entropy"]
    )


def test_private_paired_draw_helper_uses_the_same_draw_for_both_methods():
    panel = evaluation._panel_arrays(_rows(), FAMILIES)
    draws = np.array([[0, 1, 1], [2, 0, 2]], dtype=int)
    differences = evaluation._paired_macro_differences(
        panel, draws, "fingerprint", "fingerprint"
    )
    assert np.array_equal(differences, np.zeros(2))
