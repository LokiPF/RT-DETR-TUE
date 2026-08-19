import math

import pytest
import torch

from src.scene_uncertainty.knn import (
    chunked_knn_distances,
    fit_clean_distance_scale,
    mean_knn_distance,
)


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")


def test_chunked_knn_matches_full_cdist():
    generator = torch.Generator().manual_seed(4)
    queries = torch.randn(7, 5, generator=generator)
    bank = torch.randn(31, 5, generator=generator)
    expected = torch.cdist(queries, bank).topk(3, largest=False).values
    actual = chunked_knn_distances(queries, bank, k=3, bank_chunk_size=8)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_clean_distance_scale_is_finite_and_positive():
    generator = torch.Generator().manual_seed(9)
    bank = torch.randn(40, 6, generator=generator)
    state = fit_clean_distance_scale(bank, k=3, max_samples=20, seed=2)
    assert torch.isfinite(state["center"])
    assert state["scale"] > 0


def test_knn_rejects_invalid_k():
    queries = torch.zeros(2, 3)
    bank = torch.zeros(1, 3)
    try:
        chunked_knn_distances(queries, bank, k=2)
    except ValueError as error:
        assert "k" in str(error)
    else:
        raise AssertionError("Expected invalid k to fail")


def test_chunked_knn_matches_an_unchunked_reference_at_every_chunk_size():
    """Chunk width must not change the answer, including widths that do not divide the bank.

    Agreement is to float32 rounding rather than bit-for-bit: a narrower chunk gives
    `queries @ bank.T` a narrower right-hand side, and the BLAS kernel then accumulates in
    a different order. That is worth ~1 ULP (measured 2.4e-7 here), while a chunk-merge
    bug would move a distance by order one.
    """
    generator = torch.Generator().manual_seed(11)
    queries = torch.randn(6, 9, generator=generator)
    bank = torch.randn(37, 9, generator=generator)
    unchunked = torch.cdist(queries, bank, compute_mode="donot_use_mm_for_euclid_dist")
    unchunked = unchunked.topk(4, dim=1, largest=False).values
    for chunk_size in (1, 2, 5, 8, 36, 37, 38, 8192):
        actual = chunked_knn_distances(queries, bank, k=4, bank_chunk_size=chunk_size)
        assert torch.allclose(actual, unchunked, rtol=0.0, atol=1e-5)


def test_a_fixed_chunk_size_is_bit_reproducible():
    generator = torch.Generator().manual_seed(12)
    queries = torch.randn(6, 9, generator=generator)
    bank = torch.randn(37, 9, generator=generator)
    first = chunked_knn_distances(queries, bank, k=4, bank_chunk_size=8)
    assert torch.equal(chunked_knn_distances(queries, bank, k=4, bank_chunk_size=8), first)


def test_knn_rejects_a_non_positive_k():
    queries = torch.zeros(2, 3)
    bank = torch.zeros(4, 3)
    for k in (0, -1):
        with pytest.raises(ValueError, match="k"):
            chunked_knn_distances(queries, bank, k=k)


def test_k_may_equal_the_bank_size():
    generator = torch.Generator().manual_seed(13)
    queries = torch.randn(3, 4, generator=generator)
    bank = torch.randn(5, 4, generator=generator)
    assert chunked_knn_distances(queries, bank, k=5, bank_chunk_size=2).shape == (3, 5)


def test_distances_are_returned_nearest_first():
    """`fit_clean_distance_scale` drops column zero as the self match, so the order is load-bearing."""
    generator = torch.Generator().manual_seed(17)
    queries = torch.randn(5, 4, generator=generator)
    bank = torch.randn(23, 4, generator=generator)
    distances = chunked_knn_distances(queries, bank, k=6, bank_chunk_size=7)
    assert torch.all(distances[:, 1:] >= distances[:, :-1])


def test_mean_knn_distance_averages_the_k_nearest_neighbours():
    generator = torch.Generator().manual_seed(19)
    queries = torch.randn(4, 5, generator=generator)
    bank = torch.randn(29, 5, generator=generator)
    distances = chunked_knn_distances(queries, bank, k=3, bank_chunk_size=6)
    assert torch.equal(mean_knn_distance(queries, bank, k=3, bank_chunk_size=6), distances.mean(dim=1))


def test_near_duplicate_vectors_never_produce_nan_or_negative_distances():
    """The squared-norm expansion cancels catastrophically here; the clamp is what stops the NaN.

    The unclamped expansion is recomputed inline at the same chunk width the kernel uses, so the
    test asserts its own relevance: `isfinite` and `>= 0` would both pass vacuously if a torch or
    BLAS change moved this fixture out of the regime where cancellation goes negative.
    """
    generator = torch.Generator().manual_seed(23)
    bank = torch.randn(64, 335, generator=generator) * 50.0
    queries = bank[:8] + torch.randn(8, 335, generator=generator) * 1e-5
    raw = torch.cat([
        queries.square().sum(dim=1, keepdim=True)
        + chunk.square().sum(dim=1).unsqueeze(0)
        - 2.0 * queries @ chunk.T
        for chunk in bank.split(16, dim=0)
    ], dim=1)
    assert (raw < 0).any(), "fixture no longer reaches the cancellation the clamp exists for"
    assert torch.isnan(raw.sqrt()).any()
    distances = chunked_knn_distances(queries, bank, k=3, bank_chunk_size=16)
    assert torch.isfinite(distances).all()
    assert (distances >= 0).all()


def test_duplicate_bank_rows_still_give_deterministic_distances():
    """Which duplicate wins the k-th slot is ambiguous, but the distance it reports is not."""
    generator = torch.Generator().manual_seed(29)
    unique = torch.randn(9, 4, generator=generator)
    bank = unique.repeat(4, 1)
    queries = torch.randn(5, 4, generator=generator)
    first = chunked_knn_distances(queries, bank, k=5, bank_chunk_size=7)
    assert torch.equal(chunked_knn_distances(queries, bank, k=5, bank_chunk_size=7), first)
    assert torch.allclose(chunked_knn_distances(queries, bank, k=5, bank_chunk_size=36), first,
                          rtol=0.0, atol=1e-5)
    assert torch.allclose(first[:, 0], first[:, 3], rtol=0.0, atol=1e-5)


def test_a_float64_bank_is_scored_in_float32():
    generator = torch.Generator().manual_seed(31)
    queries = torch.randn(3, 6, generator=generator, dtype=torch.float64)
    bank = torch.randn(20, 6, generator=generator, dtype=torch.float64)
    assert chunked_knn_distances(queries, bank, k=2).dtype == torch.float32


def test_clean_distance_scale_rejects_a_bank_no_larger_than_k():
    bank = torch.eye(3)
    with pytest.raises(ValueError, match="Bank must contain more than k vectors"):
        fit_clean_distance_scale(bank, k=3)


def test_clean_distance_scale_rejects_a_non_positive_max_samples():
    """`-1` is a natural "use everything" sentinel, and `randperm(n)[:-1]` would quietly honour it
    as "all but the last row" -- a fit over 39 of 40 rows that looks entirely plausible."""
    generator = torch.Generator().manual_seed(47)
    bank = torch.randn(40, 5, generator=generator)
    for max_samples in (0, -1, -3):
        with pytest.raises(ValueError, match="max_samples"):
            fit_clean_distance_scale(bank, k=3, max_samples=max_samples)


def test_clean_distance_scale_reports_the_nearest_non_self_distance():
    """The dropped-self-match assumption holds only for rows unique in the bank, so expose it.

    A sampled row with a twin keeps the twin's near-zero distance in its mean, biasing `center`
    down and `scale` up with nothing in the artifacts to show it happened. `min_neighbor_distance`
    collapsing towards zero is that signal.
    """
    generator = torch.Generator().manual_seed(53)
    bank = torch.randn(48, 6, generator=generator)
    clean = fit_clean_distance_scale(bank, k=3, max_samples=48, seed=1)
    twinned = fit_clean_distance_scale(torch.cat((bank, bank[:24])), k=3, max_samples=48, seed=1)
    nearest_non_self = chunked_knn_distances(bank, bank, k=2)[:, 1].min()
    assert clean["min_neighbor_distance"] == pytest.approx(float(nearest_non_self), abs=1e-6)
    assert twinned["min_neighbor_distance"] < 1e-3 < clean["min_neighbor_distance"]


def test_clean_distance_scale_floors_a_degenerate_spread():
    """Every clean score is identical here, so an unfloored inter-quartile range would be zero."""
    bank = torch.eye(8) * 2.0
    state = fit_clean_distance_scale(bank, k=3, max_samples=8, seed=5)
    assert state["scale"] == pytest.approx(1e-6)
    assert state["center"] == pytest.approx(8.0 ** 0.5, abs=1e-5)


def test_clean_distance_scale_averages_k_neighbours_after_dropping_the_self_match():
    """Points on a circle: every row sees the same neighbour distances, so the centre is exact."""
    angles = torch.arange(12, dtype=torch.float32) * (2.0 * math.pi / 12.0)
    bank = torch.stack((angles.cos(), angles.sin()), dim=1)
    first_chord = 2.0 * math.sin(math.pi / 12.0)
    second_chord = 2.0 * math.sin(2.0 * math.pi / 12.0)
    state = fit_clean_distance_scale(bank, k=3, max_samples=12, seed=3)
    assert state["center"] == pytest.approx((2.0 * first_chord + second_chord) / 3.0, abs=1e-5)


def test_clean_distance_scale_is_deterministic_and_chunk_invariant():
    generator = torch.Generator().manual_seed(37)
    bank = torch.randn(60, 7, generator=generator)
    first = fit_clean_distance_scale(bank, k=4, max_samples=25, seed=8)
    repeat = fit_clean_distance_scale(bank, k=4, max_samples=25, seed=8)
    reseeded = fit_clean_distance_scale(bank, k=4, max_samples=25, seed=9)
    assert not torch.equal(first["center"], reseeded["center"])
    rechunked = fit_clean_distance_scale(bank, k=4, max_samples=25, seed=8, bank_chunk_size=9)
    for key in ("center", "scale", "sample_count", "min_neighbor_distance"):
        assert torch.equal(first[key], repeat[key])
        assert torch.allclose(first[key].float(), rechunked[key].float(), rtol=0.0, atol=1e-5)
    assert int(first["sample_count"]) == 25


@needs_cuda
def test_cuda_and_cpu_agree_within_float_tolerance():
    generator = torch.Generator().manual_seed(41)
    queries = torch.randn(64, 335, generator=generator)
    bank = torch.randn(4096, 335, generator=generator)
    on_cpu = chunked_knn_distances(queries, bank, k=5, bank_chunk_size=1024)
    on_cuda = chunked_knn_distances(queries.cuda(), bank.cuda(), k=5, bank_chunk_size=1024)
    assert on_cuda.device.type == "cuda"
    assert torch.allclose(on_cuda.cpu(), on_cpu, atol=1e-4)


@needs_cuda
def test_a_cpu_bank_is_scored_against_cuda_queries():
    generator = torch.Generator().manual_seed(43)
    queries = torch.randn(4, 8, generator=generator).cuda()
    bank = torch.randn(50, 8, generator=generator)
    distances = chunked_knn_distances(queries, bank, k=3, bank_chunk_size=16)
    assert distances.device.type == "cuda"
