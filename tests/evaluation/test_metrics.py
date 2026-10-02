import numpy as np
import pytest
import uq_detr

from degradation_monitor.evaluation import metrics as m
from differential_uncertainty.baselines import scores


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


def test_fit_lambda_from_parts_matches_uq_detr_fit_lambda():
    rng = np.random.default_rng(0)
    queries, reliability = [], []
    for _ in range(40):
        probs = np.full((20, 3), 1e-4)
        probs[:, 0] = rng.uniform(0, 1, 20)
        boxes = np.tile([0.5, 0.5, 0.2, 0.2], (20, 1))
        queries.append(uq_detr.Detections.from_cxcywh(boxes, probs, image_size=(100, 100)))
        reliability.append(rng.uniform(0, 1))
    reliability = np.array(reliability)
    reliability[3] = np.nan
    expected = uq_detr.fit_lambda(queries, reliability, method="threshold", param=0.3)
    conf_pos, conf_neg = scores.contrastive_parts(queries, 0.3)
    assert m.fit_lambda_from_parts(conf_pos, conf_neg, reliability) == pytest.approx(expected)


def test_cross_fit_lambda_never_uses_the_images_of_its_own_fold():
    rng = np.random.default_rng(3)
    conf_pos, conf_neg = rng.uniform(0, 1, 100), rng.uniform(0, 0.1, 100)
    folds = np.repeat([0, 1], 50)
    reliability = np.where(folds == 1, conf_pos - 20 * conf_neg, conf_pos) + rng.normal(0, 1e-3, 100)
    per_image, per_fold = m.cross_fit_lambda(conf_pos, conf_neg, reliability, folds)
    assert per_fold == {0: 20.0, 1: 0.0}
    assert set(per_image[folds == 0]) == {20.0} and set(per_image[folds == 1]) == {0.0}


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


def test_group_aurocs_average_the_common_and_the_extra_conditions_on_the_chosen_images():
    from degradation_monitor import corruptions
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(30, 96)) + np.linspace(0, 1, 96)[None]
    rows = np.array([0, 0, 3, 7, 12])
    common, extra = m.group_aurocs(scores, rows)
    assert common == pytest.approx(np.mean([m.auroc(scores[rows, 0], scores[rows, c]) for c in corruptions.COMMON_CONDITIONS]))
    assert extra == pytest.approx(np.mean([m.auroc(scores[rows, 0], scores[rows, c]) for c in corruptions.EXTRA_CONDITIONS]))
