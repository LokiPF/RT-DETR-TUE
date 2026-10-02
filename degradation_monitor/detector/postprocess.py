"""RT-DETR post-processing: sigmoid scores and the top-k (query, class) pairs, with boxes in pixels."""
from __future__ import annotations

import numpy as np
from scipy.special import expit

TOP_K = 100


def checked_outputs(logits, boxes):
    logits = np.asarray(logits, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64)
    if logits.ndim != 2 or boxes.shape != (logits.shape[0], 4):
        raise ValueError("expected logits (queries, classes) and boxes (queries, 4)")
    if not (np.isfinite(logits).all() and np.isfinite(boxes).all()):
        raise ValueError("detector outputs must be finite")
    return logits, boxes


def top_detections(logits, boxes_cxcywh, image_size, top_k=TOP_K):
    """RT-DETR post-processing: sigmoid, then the top-k (query, class) pairs."""
    logits, boxes = checked_outputs(logits, boxes_cxcywh)
    width, height = image_size
    probs = expit(logits).reshape(-1)
    order = np.argsort(-probs, kind="stable")[: min(int(top_k), probs.size)]
    queries, labels = np.divmod(order, logits.shape[1])
    cx, cy, w, h = boxes[queries].T
    xyxy = np.stack(
        [(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height],
        axis=1,
    )
    return probs[order].astype(np.float32), labels.astype(np.int64), xyxy.astype(np.float32)
