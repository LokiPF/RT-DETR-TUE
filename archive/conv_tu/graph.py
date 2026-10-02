"""Activation graph of one stride-1 3x3 conv and the exact top of its 0-dimensional persistence diagram.

Nodes are the conv's input cells (c, p) and output cells (d, q). Every product inside the conv is one edge,
weighted by |K_eff[d, c, t]| * |x[c, q + offset(t)]|; edges into the zero padding do not exist. Kruskal's
algorithm reads the edges from the heaviest down, and the weights of the edges that join two groups are the
diagram's values. Whether an edge joins two groups depends only on heavier edges, so the edges at or above
any cut below the K-th value give the K largest values exactly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numba
import numpy as np
import torch

FRACTION = 0.01             # K = 1% of the graph's nodes
SMALL_GRAPH_NODES = 10_000  # below this, keep the whole diagram (nodes - 1 values)
EDGE_CAP = 32_000_000       # at most this many edges are gathered at once
TAPS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))  # kernel index (dy + 1, dx + 1)
MAX_ROUNDS = 40


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


@numba.njit(cache=True)
def _first_merges(inputs, outputs, weights, nodes, k):
    parent = np.arange(nodes)
    values = np.empty(k, dtype=np.float32)
    found = 0
    read = 0
    for i in range(weights.shape[0]):
        read = i + 1
        a = inputs[i]
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        b = outputs[i]
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        if a != b:
            parent[a] = b
            values[found] = weights[i]
            found += 1
            if found == k:
                break
    return values[:found], read


def first_merges(edges: Edges, nodes: int, k: int) -> tuple[np.ndarray, int]:
    """Kruskal over edges sorted heaviest first: the first k merge weights and how many edges it read."""
    return _first_merges(edges.inputs.cpu().numpy().astype(np.int64, copy=False),
                         edges.outputs.cpu().numpy().astype(np.int64, copy=False),
                         edges.weights.cpu().numpy().astype(np.float32, copy=False), int(nodes), int(k))


@dataclass(frozen=True)
class TopMerges:
    values: np.ndarray    # (k,) float32: the k largest diagram values, heaviest first
    heaviest: np.ndarray  # (k,) float32: the k heaviest edge weights
    cut: float            # the cut that produced them
    gathered: int         # edges at or above that cut
    read: int             # edges Kruskal read to make the k merges
    rounds: int           # cuts tried


def _padded(values: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros(k, dtype=np.float32)
    out[:values.size] = values
    return out


def conv_top_merges(x_abs: torch.Tensor, kernel_abs: torch.Tensor, k: int, cut: float,
                    cap: int = EDGE_CAP) -> TopMerges:
    """The k largest values of the layer's diagram, exactly.

    The cut moves until the edges at or above it fit under `cap` and make k merges: fourfold down while there
    are too few merges, fourfold up while there are too many edges, then by bisection once both are known.
    If every positive edge is gathered and still makes fewer than k merges, the remaining merges happen
    through zero-weight edges, so the remaining values are exactly 0.
    """
    if (x_abs.ndim != 3 or kernel_abs.ndim != 4 or kernel_abs.shape[1] != x_abs.shape[0]
            or tuple(kernel_abs.shape[2:]) != (3, 3)):
        raise ValueError("expected x of shape (C_in, H, W) and a (C_out, C_in, 3, 3) kernel")
    if bool((x_abs < 0).any()) or bool((kernel_abs < 0).any()):
        raise ValueError("pass absolute values")
    c_in, height, width = x_abs.shape
    nodes = (c_in + kernel_abs.shape[0]) * height * width
    if not 1 <= k < nodes:
        raise ValueError(f"k must be between 1 and {nodes - 1}")
    low = high = None  # low: too many edges at this cut; high: too few merges at this cut
    positive = None
    for rounds in range(1, MAX_ROUNDS + 1):
        try:
            edges = edges_at_least(x_abs, kernel_abs, cut, cap)
        except TooManyEdges:
            low = cut
        else:
            values, read = first_merges(edges, nodes, k)
            gathered = edges.weights.numel()
            heaviest = edges.weights[:k].cpu().numpy()
            if values.size == k:
                return TopMerges(values, heaviest, cut, gathered, read, rounds)
            if positive is None:
                positive = count_positive(x_abs, kernel_abs)
            if gathered == positive:
                return TopMerges(_padded(values, k), _padded(heaviest, k), cut, gathered, read, rounds)
            high = cut
        if low is not None and high is not None:
            if high / low < 1.0001:
                raise ValueError(f"{k} merges need more than {cap} edges")
            cut = math.sqrt(low * high)
        else:
            cut = cut * 4.0 if high is None else cut / 4.0
    raise ValueError(f"no workable cut found in {MAX_ROUNDS} rounds")
