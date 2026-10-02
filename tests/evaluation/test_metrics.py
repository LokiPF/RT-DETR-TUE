import numpy as np
import pytest

from degradation_monitor.evaluation import metrics as m


def test_aupr_is_one_for_perfect_separation_and_half_for_identical_scores():
    assert m.aupr([0.0, 1.0], [2.0, 3.0]) == pytest.approx(1.0)
    assert m.aupr([1.0, 1.0], [1.0, 1.0]) == pytest.approx(0.5)


def test_fpr_at_95_tpr_uses_the_threshold_that_keeps_95_percent_of_degraded():
    degraded = np.arange(100, 200, dtype=float)
    clean = np.array([150.0, 50.0, 104.0, 106.0])
    assert m.fpr_at_95_tpr(clean, degraded) == pytest.approx(0.5)


def test_condition_aurocs_match_binary_auroc_row_by_row():
    rng = np.random.default_rng(0)
    clean, degraded = rng.normal(0, 1, 50), rng.normal(0.5, 1, (3, 50))
    expected = [m.auroc(clean, row) for row in degraded]
    assert m.condition_aurocs(clean, degraded) == pytest.approx(expected)


def test_bootstrap_intervals_bracket_the_point_estimate():
    rng = np.random.default_rng(0)
    clean, degraded = rng.normal(0, 1, 200), rng.normal(1, 1, (2, 200))
    point = m.condition_aurocs(clean, degraded).mean()
    intervals = m.bootstrap(lambda idx: {"mean_auroc": m.condition_aurocs(clean[idx], degraded[:, idx]).mean()},
                            200, samples=200, seed=1)
    low, high = intervals["mean_auroc"]
    assert low <= point <= high


from degradation_monitor.evaluation.metrics import binary_auroc


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


def test_zscored_sum_standardises_each_stage_on_clean_statistics():
    clean = np.array([[1.0, 10.0], [3.0, 30.0], [2.0, 20.0]])
    mean, std = m.stage_zstats(clean)
    assert mean.tolist() == pytest.approx([2.0, 20.0])
    assert std.tolist() == pytest.approx([np.sqrt(2 / 3), 10 * np.sqrt(2 / 3)])
    assert m.zscored_sum(np.array([[3.0, 20.0]]), mean, std)[0] == pytest.approx(1 / np.sqrt(2 / 3))
    with pytest.raises(ValueError, match="spread"):
        m.stage_zstats(np.array([[1.0, 5.0], [1.0, 6.0]]))


def test_bootstrap_gives_the_same_intervals_with_worker_processes():
    rng = np.random.default_rng(2)
    clean, degraded = rng.normal(0, 1, 60), rng.normal(0.5, 1, (3, 60))

    def statistic(idx):
        return {"a": m.condition_aurocs(clean[idx], degraded[:, idx]).mean(), "b": float(clean[idx].mean())}

    serial = m.bootstrap(statistic, 60, samples=40, seed=7)
    assert m.bootstrap(statistic, 60, samples=40, seed=7, workers=3) == serial


def test_group_separation_averages_the_three_metrics_over_conditions():
    rng = np.random.default_rng(3)
    clean, degraded = rng.normal(0, 1, 50), rng.normal(1, 1, (4, 50))
    auroc, aupr, fpr95 = m.group_separation(clean, degraded)
    assert auroc == pytest.approx(np.mean([m.auroc(clean, row) for row in degraded]))
    assert aupr == pytest.approx(np.mean([m.aupr(clean, row) for row in degraded]))
    assert fpr95 == pytest.approx(np.mean([m.fpr_at_95_tpr(clean, row) for row in degraded]))
