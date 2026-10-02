"""kNN out-of-distribution score (Sun et al., ICML 2022): distance to the k-th nearest clean train image, on the
L2-normalised, globally pooled last backbone stage of the same frozen detector."""
from __future__ import annotations

import torch

KNN_K = 100       # fixed in advance; the report also shows k = 1, 10, 50, 200
KNN_K_MAX = 200   # how many neighbour distances the detector pass stores per image


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
