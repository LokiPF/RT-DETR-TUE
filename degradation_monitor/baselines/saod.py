"""SAOD image-level uncertainty (Oksuz et al., CVPR 2023): 1 - confidence of the most confident detections."""
from __future__ import annotations

import numpy as np


def saod_uncertainty(top_scores, m: int) -> float:
    """SAOD image uncertainty: mean of (1 - p) over the m most confident detections."""
    ranked = np.sort(np.asarray(top_scores, dtype=np.float64))[::-1]
    if type(m) is not int or not 1 <= m <= ranked.size:
        raise ValueError("m must be between 1 and the number of detections")
    return float(np.mean(1.0 - ranked[:m]))
