import numpy as np
import pytest
import uq_detr

from differential_uncertainty.baselines import metrics as m
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


def test_spearman_drops_nan_pairs_and_needs_three_points():
    assert m.spearman([1, 2, 3, np.nan], [2, 4, 6, 1]) == pytest.approx(1.0)
    assert np.isnan(m.spearman([1, 2], [1, 2]))


def test_mean_within_condition_spearman_skips_conditions_without_defined_risk():
    delta_scores = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    delta_risks = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
    assert m.mean_within_condition_spearman(delta_scores, delta_risks) == pytest.approx(1.0)


def test_risk_coverage_keeps_lowest_scores_ignores_undefined_risk_and_oracle_is_best():
    scores_ = np.array([0.1, 0.2, 0.9, 0.8, 0.5])
    risks = np.array([0.0, 0.1, 1.0, 0.2, np.nan])
    _, kept = m.risk_coverage(scores_, risks, coverages=(1.0, 0.5))
    assert kept.tolist() == pytest.approx([np.nanmean(risks), 0.05])
    assert m.aurc(risks, risks, coverages=(1.0, 0.5)) <= m.aurc(scores_, risks, coverages=(1.0, 0.5))


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
