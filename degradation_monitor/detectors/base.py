"""What every frozen detector gives the monitors from one forward pass over images of one size."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

TOP_K = 100  # detections kept per image, as for RT-DETR (degradation_monitor.detector.postprocess.TOP_K)
LEVELS = ("s1", "s2", "s3", "s4")  # the method's three scored levels, then its content key
# COCO's 80 category ids in sorted order: label i is COCO_CATEGORY_IDS[i], as for RT-DETR and YOLO
COCO_CATEGORY_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31,
                     32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                     58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87,
                     88, 89, 90)
LABEL_OF_CATEGORY = np.full(91, -1, dtype=np.int64)  # COCO category id -> label 0-79, -1 for the unused ids
LABEL_OF_CATEGORY[list(COCO_CATEGORY_IDS)] = np.arange(len(COCO_CATEGORY_IDS))


@dataclass
class Outputs:
    """One forward pass over N images of one size.

    levels: the method's maps by LEVELS name, (N, C, H, W) each. cdf: the activation CDFs' five maps, stem first.
    pooled: (N, D) features for the kNN baseline. Only when the heads ran: scores, labels and boxes, the TOP_K most
    confident detections ((N, TOP_K), (N, TOP_K) and (N, TOP_K, 4); labels 0-79 in COCO's category order, boxes xyxy
    in the original image's pixels, empty slots with score 0); and for DETR-type detectors query_logits (N, Q, 80)
    and query_boxes (N, Q, 4, cxcywh in [0, 1]) for ContrastiveConf, and decoder, the last decoder layer's queries
    (N, Q, D), for Hashemi et al.
    """
    levels: dict
    cdf: list
    pooled: torch.Tensor
    scores: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    boxes: Optional[np.ndarray] = None
    query_logits: Optional[np.ndarray] = None
    query_boxes: Optional[np.ndarray] = None
    decoder: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class Region:
    """Where the image lies in the network's input, in input pixels."""
    top: float
    left: float
    height: float
    width: float

    def crop(self, maps: torch.Tensor, stride: int) -> torch.Tensor:
        """The cells of a stride-`stride` map that lie wholly on the image (at least one row and one column)."""
        top, left = math.ceil(self.top / stride), math.ceil(self.left / stride)
        bottom = max(top + 1, math.floor((self.top + self.height) / stride))
        right = max(left + 1, math.floor((self.left + self.width) / stride))
        return maps[..., top:bottom, left:right]


def padded(scores, labels, boxes, k: int = TOP_K):
    """The k most confident detections; empty slots get score 0, label 0 and an empty box."""
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    order = np.argsort(-scores, kind="stable")[:k]
    out_scores, out_labels, out_boxes = np.zeros(k, np.float32), np.zeros(k, np.int64), np.zeros((k, 4), np.float32)
    out_scores[:len(order)], out_labels[:len(order)], out_boxes[:len(order)] = scores[order], labels[order], boxes[order]
    return out_scores, out_labels, out_boxes
