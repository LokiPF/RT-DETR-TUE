from __future__ import annotations

import torch
from torch import Tensor


def _squared_distances(queries: Tensor, bank: Tensor) -> Tensor:
    """Squared Euclidean distances via the norm expansion, floored at zero.

    `||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b` cancels catastrophically for near-identical
    vectors: it subtracts two large, almost equal float32 numbers, so the result can land
    slightly below zero. The clamp is what keeps `sqrt` from turning that rounding noise
    into a NaN that would then propagate silently through every downstream scene score.
    """
    query_norm = queries.square().sum(dim=1, keepdim=True)
    bank_norm = bank.square().sum(dim=1).unsqueeze(0)
    return (query_norm + bank_norm - 2.0 * queries @ bank.T).clamp_min_(0.0)


def chunked_knn_distances(
    queries: Tensor,
    bank: Tensor,
    k: int,
    bank_chunk_size: int = 8192,
) -> Tensor:
    """Exact distances to the k nearest bank vectors, nearest first.

    The full `(query, bank)` distance matrix is never materialised: each bank chunk
    contributes only its own k smallest distances, which are merged into a running
    top-k. That merge is exact, because a global k-nearest neighbour is always among
    the k nearest within its own chunk.
    """
    queries = queries.float()
    bank = bank.float().to(queries.device)
    if k <= 0 or k > bank.shape[0]:
        raise ValueError(f"k must be in [1, {bank.shape[0]}], got {k}")
    best = torch.full((queries.shape[0], k), float("inf"), device=queries.device)
    for bank_chunk in bank.split(bank_chunk_size, dim=0):
        local = _squared_distances(queries, bank_chunk)
        local_k = min(k, local.shape[1])
        local_best = local.topk(local_k, dim=1, largest=False).values
        best = torch.cat((best, local_best), dim=1).topk(k, dim=1, largest=False).values
    return best.clamp_min(0.0).sqrt()


def mean_knn_distance(queries: Tensor, bank: Tensor, k: int, bank_chunk_size: int = 8192) -> Tensor:
    return chunked_knn_distances(queries, bank, k, bank_chunk_size).mean(dim=1)


def fit_clean_distance_scale(
    bank: Tensor,
    k: int,
    max_samples: int = 1024,
    seed: int = 42,
    bank_chunk_size: int = 8192,
) -> dict[str, Tensor]:
    """Robust centre and spread of the clean mean-kNN distance, for standardising scores.

    Sampled rows are scored against the bank they belong to, so column zero is that row's own
    zero-distance self match and is dropped by averaging neighbours `1..k`.

    That drop is *positional, not by identity*: it removes one near-zero distance, which is the
    self match only when the sampled row is unique in the bank. A row that has a near-duplicate
    twin elsewhere in the bank keeps the twin's near-zero distance in its average, which pulls
    `center` down and inflates `scale`. Measured on a 2000x64 bank with 20% of rows duplicated,
    `center` fell 4.6% while `scale` grew 2.4x -- and `scale` is the divisor every downstream
    scene score is standardised by, so the distortion is silent and plausible-looking. Whether
    near-duplicate bank rows belong in a clean-distance fit is a question about what `scale` is
    meant to measure, so this function does not decide it; it reports `min_neighbor_distance`,
    the smallest post-self distance any sampled row saw, so the assumption is visible in the
    artifacts. A value far below `center` means the bank holds near-duplicates and the fit should
    be revisited -- offline, from this same state, without re-extracting anything.

    The inter-quartile spread is floored, because a bank whose clean distances are all equal
    would otherwise hand every caller a division by zero.
    """
    if bank.shape[0] <= k:
        raise ValueError("Bank must contain more than k vectors for leave-self-out scaling")
    if max_samples <= 0:
        raise ValueError(f"max_samples must be positive, got {max_samples}")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(bank.shape[0], generator=generator)[:max_samples]
    sample = bank.index_select(0, indices.to(bank.device))
    neighbors = chunked_knn_distances(sample, bank, k + 1, bank_chunk_size)
    clean_scores = neighbors[:, 1:].mean(dim=1).cpu()
    center = clean_scores.median()
    scale = (
        torch.quantile(clean_scores, 0.75) - torch.quantile(clean_scores, 0.25)
    ).clamp_min(1e-6)
    return {
        "center": center,
        "scale": scale,
        "sample_count": torch.tensor(len(clean_scores)),
        "min_neighbor_distance": neighbors[:, 1].min().cpu(),
    }
