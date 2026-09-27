"""SAOD, ContrastiveConf and kNN image scores from one frozen detector."""
from __future__ import annotations

import numpy as np
import torch
import uq_detr
from scipy.special import expit


def _checked(logits, boxes):
    logits = np.asarray(logits, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64)
    if logits.ndim != 2 or boxes.shape != (logits.shape[0], 4):
        raise ValueError("expected logits (queries, classes) and boxes (queries, 4)")
    if not (np.isfinite(logits).all() and np.isfinite(boxes).all()):
        raise ValueError("detector outputs must be finite")
    return logits, boxes


def top_detections(logits, boxes_cxcywh, image_size, top_k=100):
    """RT-DETR post-processing: sigmoid, then the top-k (query, class) pairs."""
    logits, boxes = _checked(logits, boxes_cxcywh)
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


def saod_uncertainty(top_scores, m: int) -> float:
    """SAOD image uncertainty: mean of (1 - p) over the m most confident detections."""
    ranked = np.sort(np.asarray(top_scores, dtype=np.float64))[::-1]
    if type(m) is not int or not 1 <= m <= ranked.size:
        raise ValueError("m must be between 1 and the number of detections")
    return float(np.mean(1.0 - ranked[:m]))


def query_detections(logits, boxes_cxcywh, image_size) -> uq_detr.Detections:
    """All queries of one image as uq-detr Detections (image_size is width, height)."""
    logits, boxes = _checked(logits, boxes_cxcywh)
    width, height = image_size
    return uq_detr.Detections.from_cxcywh(boxes, expit(logits), image_size=(height, width))


def contrastive_parts(query_sets, theta: float = 0.3):
    """Conf+ and Conf- from uq-detr's threshold split, recovered from two lambda values."""
    conf_pos = np.asarray(
        uq_detr.contrastive_conf(query_sets, method="threshold", param=theta, lambda_=0.0), float)
    difference = np.asarray(
        uq_detr.contrastive_conf(query_sets, method="threshold", param=theta, lambda_=1.0), float)
    return conf_pos, conf_pos - difference


def contrastive_degradation(conf_pos, conf_neg, lam: float) -> np.ndarray:
    """Higher means more likely degraded: minus ContrastiveConf."""
    return -(np.asarray(conf_pos, float) - lam * np.asarray(conf_neg, float))


def normalize_rows(features: torch.Tensor) -> torch.Tensor:
    features = features.float()
    norms = features.norm(dim=1, keepdim=True)
    if not bool(torch.isfinite(features).all()) or bool((norms == 0).any()):
        raise ValueError("features must be finite with nonzero norm")
    return features / norms


def knn_distances(queries: torch.Tensor, bank: torch.Tensor, k_max: int, chunk_size: int = 16384):
    """Ascending Euclidean distances from unit queries to their k_max nearest unit bank rows."""
    if queries.ndim != 2 or bank.ndim != 2 or queries.shape[1] != bank.shape[1]:
        raise ValueError("queries and bank must be (rows, dim) with the same dim")
    if not 1 <= k_max <= bank.shape[0]:
        raise ValueError("k_max must be between 1 and the bank size")
    q = normalize_rows(queries)
    best = None
    for start in range(0, bank.shape[0], chunk_size):
        rows = bank[start:start + chunk_size].to(device=q.device, dtype=torch.float32)
        squared = (2.0 - 2.0 * q @ rows.T).clamp_min_(0.0)
        local = squared.topk(min(k_max, rows.shape[0]), dim=1, largest=False).values
        merged = local if best is None else torch.cat([best, local], dim=1)
        best = merged.topk(min(k_max, merged.shape[1]), dim=1, largest=False).values
    return best.sqrt()
