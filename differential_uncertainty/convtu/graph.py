"""Activation graph of one stride-1 3x3 conv and the exact top of its 0-dimensional persistence diagram.

Nodes are the conv's input cells (c, p) and output cells (d, q). Every product inside the conv is one edge,
weighted by |K_eff[d, c, t]| * |x[c, q + offset(t)]|; edges into the zero padding do not exist. Kruskal's
algorithm reads the edges from the heaviest down, and the weights of the edges that join two groups are the
diagram's values. Whether an edge joins two groups depends only on heavier edges, so the edges at or above
any cut below the K-th value give the K largest values exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

FRACTION = 0.01             # K = 1% of the graph's nodes
SMALL_GRAPH_NODES = 10_000  # below this, keep the whole diagram (nodes - 1 values)
EDGE_CAP = 32_000_000       # at most this many edges are gathered at once
TAPS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))  # kernel index (dy + 1, dx + 1)


def fingerprint_length(nodes: int) -> int:
    """K: 1% of the nodes, or the whole diagram (nodes - 1 values) for graphs under 10,000 nodes."""
    if nodes < 2:
        raise ValueError("a graph needs at least two nodes")
    if nodes < SMALL_GRAPH_NODES:
        return nodes - 1
    return int(round(FRACTION * nodes))


def folded_kernel(conv: torch.nn.Conv2d, norm: torch.nn.Module) -> torch.Tensor:
    """|K| times the norm layer's per-output-channel scale |gamma| / sqrt(var + eps), as float32."""
    if (tuple(conv.kernel_size) != (3, 3) or tuple(conv.stride) != (1, 1) or tuple(conv.padding) != (1, 1)
            or tuple(conv.dilation) != (1, 1) or conv.groups != 1):
        raise ValueError("only stride-1, padding-1, 3x3 convs without dilation or groups are supported")
    scale = norm.weight.detach().abs() / torch.sqrt(norm.running_var.detach() + norm.eps)
    return (conv.weight.detach().abs() * scale[:, None, None, None]).float()


def _shifted(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """out[c, qy, qx] = x[c, qy + dy, qx + dx], and 0 where that cell is outside the map."""
    _, height, width = x.shape
    out = torch.zeros_like(x)
    ys, ye = max(0, -dy), min(height, height - dy)
    xs, xe = max(0, -dx), min(width, width - dx)
    out[:, ys:ye, xs:xe] = x[:, ys + dy:ye + dy, xs + dx:xe + dx]
    return out


def tap_products(x: torch.Tensor, kernel: torch.Tensor, tap: int) -> torch.Tensor:
    """(C_out, C_in, H, W): kernel[d, c, tap] * x[c, q + offset(tap)], 0 where the tap reads the padding."""
    dy, dx = TAPS[tap]
    return kernel[:, :, dy + 1, dx + 1][:, :, None, None] * _shifted(x, dy, dx)[None]


class TooManyEdges(ValueError):
    pass


@dataclass(frozen=True)
class Edges:
    weights: torch.Tensor  # (n,) float32, heaviest first
    inputs: torch.Tensor   # (n,) input-node ids c * H * W + p
    outputs: torch.Tensor  # (n,) output-node ids C_in * H * W + d * H * W + q


def edges_at_least(x_abs: torch.Tensor, kernel_abs: torch.Tensor, cut: float, cap: int = EDGE_CAP) -> Edges:
    """Every edge weighing at least `cut` (> 0), heaviest first; more than `cap` of them raises TooManyEdges."""
    if not cut > 0:
        raise ValueError("the cut must be positive")
    c_in, height, width = x_abs.shape
    cells = height * width
    weights, inputs, outputs = [], [], []
    total = 0
    for tap, (dy, dx) in enumerate(TAPS):
        products = tap_products(x_abs, kernel_abs, tap)
        d, c, qy, qx = torch.nonzero(products >= cut, as_tuple=True)
        total += d.numel()
        if total > cap:
            raise TooManyEdges(f"more than {cap} edges weigh at least {cut:g}")
        weights.append(products[d, c, qy, qx])
        inputs.append(c * cells + (qy + dy) * width + (qx + dx))
        outputs.append(c_in * cells + d * cells + qy * width + qx)
    weights = torch.cat(weights)
    order = torch.argsort(weights, descending=True, stable=True)
    return Edges(weights[order], torch.cat(inputs)[order], torch.cat(outputs)[order])


def count_positive(x_abs: torch.Tensor, kernel_abs: torch.Tensor) -> int:
    return sum(int((tap_products(x_abs, kernel_abs, tap) > 0).sum()) for tap in range(len(TAPS)))


def heaviest_weight(x_abs: torch.Tensor, kernel_abs: torch.Tensor, m: int) -> float:
    """The m-th largest edge weight, exactly; the smallest positive one if fewer than m edges are positive."""
    best = None
    for tap in range(len(TAPS)):
        flat = tap_products(x_abs, kernel_abs, tap).flatten()
        top = torch.topk(flat, min(m, flat.numel())).values
        merged = top if best is None else torch.cat([best, top])
        best = torch.topk(merged, min(m, merged.numel())).values
    positive = best[best > 0]
    if positive.numel() == 0:
        raise ValueError("every edge weighs zero")
    return float(positive[min(m, positive.numel()) - 1])
