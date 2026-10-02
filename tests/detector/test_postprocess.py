import numpy as np
import pytest
from scipy.special import logit

from degradation_monitor.detector.postprocess import TOP_K, top_detections


def test_top_detections_ranks_query_class_pairs_and_scales_boxes_to_pixels():
    logits = logit(np.array([[0.2, 0.9], [0.7, 0.1]]))
    boxes = np.array([[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.5, 0.5]])
    top, labels, xyxy = top_detections(logits, boxes, image_size=(100, 50), top_k=3)
    assert top == pytest.approx([0.9, 0.7, 0.2], abs=1e-6)
    assert labels.tolist() == [1, 0, 0]
    assert xyxy[0] == pytest.approx([40, 15, 60, 35])
    assert xyxy[1] == pytest.approx([0, 0, 50, 25])
    assert TOP_K == 100
