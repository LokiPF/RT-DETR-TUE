import numpy as np
import pytest
import torch

from degradation_monitor.baselines.knn import KNN_K, KNN_K_MAX, knn_distances


def test_knn_distances_are_sorted_kth_neighbour_distances_and_independent_of_chunking():
    angles = np.deg2rad([0, 10, 20, 30, 90])
    bank = torch.tensor(np.stack([np.cos(angles), np.sin(angles)], 1), dtype=torch.float32)
    query = torch.tensor([[2.0, 0.0]])
    full = knn_distances(query, bank, k_max=3, chunk_size=100)
    chunked = knn_distances(query, bank, k_max=3, chunk_size=2)
    expected = [2 * np.sin(np.deg2rad(d) / 2) for d in (0, 10, 20)]
    assert full[0].tolist() == pytest.approx(expected, abs=1e-3)
    assert torch.allclose(full, chunked)
    assert (KNN_K, KNN_K_MAX) == (100, 200)


def test_knn_distances_reject_zero_queries_and_too_large_k():
    bank = torch.eye(2)
    with pytest.raises(ValueError):
        knn_distances(torch.zeros(1, 2), bank, k_max=1)
    with pytest.raises(ValueError):
        knn_distances(torch.ones(1, 2), bank, k_max=3)
