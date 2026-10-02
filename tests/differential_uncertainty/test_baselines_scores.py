import numpy as np
import pytest
import torch
import uq_detr
from scipy.special import logit

from differential_uncertainty.baselines import scores


def _outputs(max_probs, classes=3):
    probs = np.full((len(max_probs), classes), 1e-4)
    probs[:, 0] = max_probs
    boxes = np.tile([0.5, 0.5, 0.2, 0.4], (len(max_probs), 1))
    return logit(probs), boxes


def test_saod_uncertainty_averages_one_minus_confidence_of_the_m_best():
    top = np.array([0.1, 0.9, 0.8])
    assert scores.saod_uncertainty(top, 1) == pytest.approx(0.1)
    assert scores.saod_uncertainty(top, 3) == pytest.approx((0.1 + 0.2 + 0.9) / 3)
    for bad in (0, 4):
        with pytest.raises(ValueError):
            scores.saod_uncertainty(top, bad)


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


def test_knn_distances_are_sorted_kth_neighbour_distances_and_independent_of_chunking():
    angles = np.deg2rad([0, 10, 20, 30, 90])
    bank = torch.tensor(np.stack([np.cos(angles), np.sin(angles)], 1), dtype=torch.float32)
    query = torch.tensor([[2.0, 0.0]])
    full = scores.knn_distances(query, bank, k_max=3, chunk_size=100)
    chunked = scores.knn_distances(query, bank, k_max=3, chunk_size=2)
    expected = [2 * np.sin(np.deg2rad(d) / 2) for d in (0, 10, 20)]
    assert full[0].tolist() == pytest.approx(expected, abs=1e-3)
    assert torch.allclose(full, chunked)


def test_knn_distances_reject_zero_queries_and_too_large_k():
    bank = torch.eye(2)
    with pytest.raises(ValueError):
        scores.knn_distances(torch.zeros(1, 2), bank, k_max=1)
    with pytest.raises(ValueError):
        scores.knn_distances(torch.ones(1, 2), bank, k_max=3)
