import numpy as np
import pytest
import torch

from differential_uncertainty.convtu import graph


def _random_layer(seed, c_in=3, c_out=2, height=5, width=6, zero_share=0.4):
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(c_in, height, width, generator=generator)
    x[torch.rand(c_in, height, width, generator=generator) < zero_share] = 0.0
    kernel = torch.rand(c_out, c_in, 3, 3, generator=generator)
    return x, kernel


def test_signed_tap_products_add_up_to_the_conv():
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(3, 5, 6, generator=generator)
    kernel = torch.randn(2, 3, 3, 3, generator=generator)
    total = sum(graph.tap_products(x, kernel, tap).sum(dim=1) for tap in range(9))
    expected = torch.nn.functional.conv2d(x[None], kernel, padding=1)[0]
    assert torch.allclose(total, expected, atol=1e-5)


def test_gathered_edges_skip_the_zero_padding():
    x, kernel = torch.ones(3, 5, 6), torch.ones(2, 3, 3, 3)
    edges = graph.edges_at_least(x, kernel, 0.5)
    assert edges.weights.numel() == 2 * 3 * (3 * 5 - 2) * (3 * 6 - 2)
    assert bool((edges.weights == 1).all())


def test_heaviest_weight_is_exact():
    x, kernel = _random_layer(5)
    weights = np.sort(np.concatenate([graph.tap_products(x, kernel, t).flatten().numpy() for t in range(9)]))[::-1]
    assert graph.heaviest_weight(x, kernel, 17) == float(weights[16])


def test_fingerprint_length_is_one_percent_of_the_nodes_for_large_graphs():
    assert graph.fingerprint_length(3_276_800) == 32_768
    assert graph.fingerprint_length(10_000) == 100
    assert graph.fingerprint_length(9_999) == 9_998
    with pytest.raises(ValueError):
        graph.fingerprint_length(1)


def test_folded_kernel_multiplies_by_the_norm_scale():
    conv = torch.nn.Conv2d(3, 2, 3, 1, padding=1, bias=False)
    norm = torch.nn.BatchNorm2d(2, eps=1.0)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([2.0, -6.0]))
        norm.running_var.copy_(torch.tensor([3.0, 8.0]))
    folded = graph.folded_kernel(conv, norm)
    assert torch.allclose(folded[0], conv.weight[0].abs() * 1.0)
    assert torch.allclose(folded[1], conv.weight[1].abs() * 2.0)
    with pytest.raises(ValueError, match="stride-1"):
        graph.folded_kernel(torch.nn.Conv2d(3, 2, 3, 2, padding=1), norm)


def _brute_force(x, kernel, k):
    """Every edge explicitly, then Kruskal from the heaviest, in float32 like the module."""
    x, kernel = x.float(), kernel.float()
    c_in, height, width = x.shape
    cells = height * width
    edges = []
    for d in range(kernel.shape[0]):
        for c in range(c_in):
            for ky in range(3):
                for kx in range(3):
                    for qy in range(height):
                        for qx in range(width):
                            py, px = qy + ky - 1, qx + kx - 1
                            if 0 <= py < height and 0 <= px < width:
                                weight = float(kernel[d, c, ky, kx] * x[c, py, px])
                                edges.append((weight, c * cells + py * width + px,
                                              c_in * cells + d * cells + qy * width + qx))
    edges.sort(key=lambda e: -e[0])
    parent = list(range((c_in + kernel.shape[0]) * cells))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    values = []
    for weight, a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            values.append(weight)
    heaviest = [e[0] for e in edges[:k]]
    heaviest += [0.0] * (k - len(heaviest))
    return np.array(values[:k], dtype=np.float32), np.array(heaviest, dtype=np.float32)


@pytest.mark.parametrize("seed", range(6))
def test_top_merges_match_brute_force_kruskal(seed):
    x, kernel = _random_layer(seed)
    k = 40
    expected, heaviest = _brute_force(x, kernel, k)
    start = graph.heaviest_weight(x, kernel, 8 * k)
    top = graph.conv_top_merges(x, kernel, k, start)
    np.testing.assert_array_equal(top.values, expected)
    np.testing.assert_array_equal(top.heaviest, heaviest)
    assert top.read <= top.gathered


def test_a_cut_above_every_edge_is_lowered_and_stays_exact():
    x, kernel = _random_layer(1)
    expected, _ = _brute_force(x, kernel, 25)
    top = graph.conv_top_merges(x, kernel, 25, cut=100.0)
    assert top.rounds > 1
    np.testing.assert_array_equal(top.values, expected)


def test_a_small_cap_raises_the_cut_and_stays_exact():
    x, kernel = _random_layer(2)
    expected, _ = _brute_force(x, kernel, 5)
    top = graph.conv_top_merges(x, kernel, 5, cut=1e-9, cap=60)
    assert top.gathered <= 60 and top.rounds > 1
    np.testing.assert_array_equal(top.values, expected)


def test_too_few_positive_edges_leave_exact_zeros():
    x = torch.zeros(3, 5, 6)
    kernel = torch.rand(2, 3, 3, 3, generator=torch.Generator().manual_seed(3))
    x[1, 2, 3] = 0.7  # one active cell: at most 2 * 9 positive edges
    expected, heaviest = _brute_force(x, kernel, 30)
    top = graph.conv_top_merges(x, kernel, 30, cut=1.0)
    np.testing.assert_array_equal(top.values, expected)
    np.testing.assert_array_equal(top.heaviest, heaviest)
    assert (top.values[18:] == 0).all()


def test_k_that_needs_more_than_the_cap_is_refused():
    x, kernel = _random_layer(4, zero_share=0.0)
    with pytest.raises(ValueError, match="need more than 3 edges"):
        graph.conv_top_merges(x, kernel, 10, cut=0.5, cap=3)
