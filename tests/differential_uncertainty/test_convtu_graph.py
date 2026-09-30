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
