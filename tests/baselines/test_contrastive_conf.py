import numpy as np
import pytest
import uq_detr
from scipy.special import logit

from degradation_monitor.baselines import contrastive_conf as scores


def _outputs(max_probs, classes=3):
    probs = np.full((len(max_probs), classes), 1e-4)
    probs[:, 0] = max_probs
    boxes = np.tile([0.5, 0.5, 0.2, 0.4], (len(max_probs), 1))
    return logit(probs), boxes


def test_contrastive_parts_match_uq_detr_threshold_split():
    logits, boxes = _outputs([0.9, 0.5, 0.2, 0.1])
    queries = [scores.query_detections(logits, boxes, (100, 50))]
    conf_pos, conf_neg = scores.contrastive_parts(queries, theta=0.3)
    assert conf_pos[0] == pytest.approx(0.7)
    assert conf_neg[0] == pytest.approx(0.15)
    reference = uq_detr.contrastive_conf(queries, method="threshold", param=0.3, lambda_=5.0)[0]
    assert -scores.contrastive_degradation(conf_pos, conf_neg, 5.0)[0] == pytest.approx(reference)


def test_contrastive_parts_without_positive_queries_use_all_queries_as_negatives():
    logits, boxes = _outputs([0.2, 0.1])
    conf_pos, conf_neg = scores.contrastive_parts([scores.query_detections(logits, boxes, (10, 10))])
    assert conf_pos[0] == 0.0 and conf_neg[0] == pytest.approx(0.15)


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
    assert scores.fit_lambda_from_parts(conf_pos, conf_neg, reliability) == pytest.approx(expected)


def test_cross_fit_lambda_never_uses_the_images_of_its_own_fold():
    rng = np.random.default_rng(3)
    conf_pos, conf_neg = rng.uniform(0, 1, 100), rng.uniform(0, 0.1, 100)
    folds = np.repeat([0, 1], 50)
    reliability = np.where(folds == 1, conf_pos - 20 * conf_neg, conf_pos) + rng.normal(0, 1e-3, 100)
    per_image, per_fold = scores.cross_fit_lambda(conf_pos, conf_neg, reliability, folds)
    assert per_fold == {0: 20.0, 1: 0.0}
    assert set(per_image[folds == 0]) == {20.0} and set(per_image[folds == 1]) == {0.0}
