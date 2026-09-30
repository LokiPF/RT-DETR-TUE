# Conv-layer Topological Uncertainty Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score every condition of 200 COCO val images with Topological Uncertainty on four backbone conv layers, with fingerprints of K = 1% of each graph's nodes. Report the result next to three controls and the nine existing baseline rows.

**Architecture:**
- **New subpackage `differential_uncertainty/convtu/`:**
  - `graph.py`: the conv's activation graph, and the exact top of its diagram. Edges are gathered above a cut, one kernel tap at a time, and a numba union–find runs over them.
  - `tap.py`: forward pre-hooks that capture the input of `res_layers[s].blocks[1].branch2a.conv`, plus the folded kernels.
  - `features.py`: the fingerprint (**mst**), its three controls (**edges**, **acts**, **means**), and a Euclidean kNN.
  - `pipeline.py`: five resumable phases.
  - `report.py`: the pilot tables.
- **Wiring.** The phases run through the existing `baselines-coco --phase` runner, with the same `Settings`, output folder and `run_config.json`. They only add the folders `convtu/`, `test_convtu/` and `results_convtu/`.

**Tech Stack:** Python 3.11, PyTorch 2.11, NumPy, numba 0.65, pytest; the existing `differential_uncertainty.baselines` package.

**Spec:**
- `docs/conv-tu-pilot-design.md`: the decisions of 30 September 2026 and the evidence gathered before the pilot.
- The page "Conv Graph Reductions", Idea 1, which argues why a cut keeps the top K values exact.

**Before you start:**
- Create the worktree: `git worktree add .worktrees/convtu-pilot -b convtu-pilot fingerprint_bank` (`.worktrees` is git-ignored).
- In the worktree, commit `docs/conv-tu-pilot-design.md` and this plan. Both are untracked in the main checkout; copy them over.
- Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU
  ```

## Global Constraints

- **Detector:** the frozen RT-DETRv2-R18 COCO checkpoint `/home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth`, with 640 × 640 input.
- **Layers:** `backbone.res_layers[s].blocks[1].branch2a.conv` for s = 0, 1, 2, 3. Each kernel is folded with that branch's norm scale |γ| / √(σ² + ε).
- **Fingerprint:** K = round(0.01 × nodes). Graphs under 10,000 nodes keep all nodes − 1 values. The values must be exact: the K largest Kruskal merge weights of the full conv graph.
- **Protocol:** unchanged.
  - `runs/coco-baselines/run_config.json` must not change; do not add anything to `Settings.experiment()`.
  - The corruptions come from `protocol.variants`, and their digests must match `runs/coco-baselines/test/*.npz`.
- **Clean data:** three disjoint seeded draws from COCO train2017 (seed 44): 200 for calibration, 2,000 for the bank, 500 for the z-statistics.
- **Pilot images:** the first 200 images of `evaluation(settings)` (the seed-44 order), with all 96 conditions.
- **Scoring:** mean Euclidean distance to the 5 nearest bank rows, not normalized. Per-layer scores are z-scored with the z-statistics images and summed over the four layers.
- **Reporting:**
  - The nine baseline rows use the stored full-run LRP threshold (`results/summary.json`, 0.55) and the stored per-fold λ of ContrastiveConf.
  - Paired bootstrap with 1,000 draws, seed 44.
- **GPU:** one RTX 5090, shared with the `explore` and `hunk-0927` sessions. Agree a window with them through SendMessage before any GPU phase. Tests never use the GPU: prefix every pytest run with `CUDA_VISIBLE_DEVICES=`.
- **Python:** `/home/yuchen/miniconda3/envs/UE/bin/python`.
- **Never commit** the Springer PDF `978-3-032-31654-7_9.pdf`, `can_you_make_it_more_foggy_at.mp4` or `output/`.
- **The existing suite stays green:** 228 passed, 2 skipped before this plan.

## Review Focus

1. **Noise-corrupted images, whose activations are much larger than clean ones.**
   - Risk: far more edges clear the clean cut.
   - Expected: the cut rises until the edges fit under the cap, and the values stay exact.
   - Pinned by `test_a_small_cap_raises_the_cut_and_stays_exact` (Task 2) and by the Task 8 gate on rounds and peak memory.
2. **Near-black or flat variants (brightness and contrast at severity 5, blank inputs).**
   - Risk: few products are positive.
   - Expected: the missing merges happen through zero-weight edges, so the values are exactly 0, and nothing crashes.
   - Pinned by `test_too_few_positive_edges_leave_exact_zeros` (Task 2) and `test_a_blank_input_gives_all_zero_features` (Task 4).
3. **A resumed scoring run after the calibration, bank or z-statistics changed.**
   - Expected: refusal instead of mixing fits.
   - Pinned by `test_scores_phase_refuses_changed_fits` (Task 6) and `test_zstats_refuse_a_bank_from_another_calibration` (Task 5).
4. **Corruptions regenerated differently from the detector pass.**
   - Expected: refusal.
   - Pinned by `test_scores_phase_detects_changed_corruptions` (Task 6).
5. **A crash in the middle of the scoring phase.** Other sessions can take GPU memory at any time.
   - Expected: finished images are kept, and a rerun continues with the rest.
   - Pinned by `test_scores_phase_writes_every_condition_and_resumes` (Task 6). `peak_gpu_gib` in the calibration output feeds the Task 8 gate.

---

### Task 1: Conv-graph edges

**Files:**
- Create: `differential_uncertainty/convtu/__init__.py`
- Create: `differential_uncertainty/convtu/graph.py`
- Test: `tests/differential_uncertainty/test_convtu_graph.py`

**Interfaces:**
- Consumes: nothing.
- Produces, in `differential_uncertainty.convtu.graph`:
  - `FRACTION = 0.01`, `SMALL_GRAPH_NODES = 10_000`, `EDGE_CAP = 32_000_000`, `TAPS`: 9 `(dy, dx)` pairs, with kernel index `(dy + 1, dx + 1)`.
  - `fingerprint_length(nodes: int) -> int`
  - `folded_kernel(conv: nn.Conv2d, norm: nn.Module) -> Tensor`: shape `(C_out, C_in, 3, 3)`, float32, ≥ 0.
  - `tap_products(x: Tensor, kernel: Tensor, tap: int) -> Tensor`: shape `(C_out, C_in, H, W)`.
  - `class TooManyEdges(ValueError)`
  - `@dataclass(frozen=True) Edges(weights: Tensor, inputs: Tensor, outputs: Tensor)`: heaviest first. Input nodes are numbered `c·H·W + p`, output nodes `C_in·H·W + d·H·W + q`.
  - `edges_at_least(x_abs, kernel_abs, cut: float, cap: int = EDGE_CAP) -> Edges`
  - `count_positive(x_abs, kernel_abs) -> int`
  - `heaviest_weight(x_abs, kernel_abs, m: int) -> float`

- [ ] **Step 1: Write the failing tests**

`tests/differential_uncertainty/test_convtu_graph.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_convtu_graph.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'differential_uncertainty.convtu'`.

- [ ] **Step 3: Write the implementation**

`differential_uncertainty/convtu/__init__.py`:

```python
"""Topological Uncertainty on the backbone's conv layers: exact top-K persistence fingerprints."""
```

`differential_uncertainty/convtu/graph.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_convtu_graph.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add differential_uncertainty/convtu/__init__.py differential_uncertainty/convtu/graph.py tests/differential_uncertainty/test_convtu_graph.py
git commit -m "feat: enumerate the edges of a conv layer's activation graph above a cut"
```

---

### Task 2: Exact top of the diagram

**Files:**
- Modify: `differential_uncertainty/convtu/graph.py`. Add the imports `math`, `numba` and `numpy as np`, then append after `heaviest_weight`.
- Test: `tests/differential_uncertainty/test_convtu_graph.py` (append).

**Interfaces:**
- Consumes: from Task 1, `Edges`, `edges_at_least`, `count_positive`, `TooManyEdges`, `EDGE_CAP`.
- Produces, in `differential_uncertainty.convtu.graph`:
  - `MAX_ROUNDS = 40`
  - `first_merges(edges: Edges, nodes: int, k: int) -> tuple[np.ndarray, int]`: the merge weights, then the number of edges read.
  - `@dataclass(frozen=True) TopMerges(values: np.ndarray, heaviest: np.ndarray, cut: float, gathered: int, read: int, rounds: int)`. `values` and `heaviest` are `(k,)` float32 arrays.
  - `conv_top_merges(x_abs, kernel_abs, k: int, cut: float, cap: int = EDGE_CAP) -> TopMerges`

- [ ] **Step 1: Write the failing tests** (append to `tests/differential_uncertainty/test_convtu_graph.py`)

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_convtu_graph.py -q`
Expected: the 10 new tests fail with `AttributeError: module 'differential_uncertainty.convtu.graph' has no attribute 'conv_top_merges'`; the 5 from Task 1 pass.

- [ ] **Step 3: Write the implementation**

At the top of `graph.py`, the imports become:

```python
import math
from dataclasses import dataclass

import numba
import numpy as np
import torch
```

Below `TAPS`, add `MAX_ROUNDS = 40`. Append after `heaviest_weight`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_convtu_graph.py -q`
Expected: `15 passed`. This exact code passed the same 15 tests as a prototype on 30 September.

- [ ] **Step 5: Commit**

```bash
git add differential_uncertainty/convtu/graph.py tests/differential_uncertainty/test_convtu_graph.py
git commit -m "feat: exact top-K persistence values of a conv graph from the edges above a cut"
```

---

### Task 3: Inputs of the four pilot convs

**Files:**
- Create: `differential_uncertainty/convtu/tap.py`
- Create: `tests/differential_uncertainty/convtu_fakes.py` (test helpers, not a test module)
- Test: `tests/differential_uncertainty/test_convtu_tap.py`

**Interfaces:**
- Consumes: `graph.folded_kernel` (Task 1).
- Produces:
  - In `differential_uncertainty.convtu.tap`:
    - `STAGES = 4`
    - `LAYER_NAMES = ("res_layers.0.blocks.1.branch2a.conv", …, "res_layers.3.blocks.1.branch2a.conv")`
    - `class ConvInputs(backbone)`:
      - `.kernels: list[Tensor]`, the folded `(C, C, 3, 3)` kernels on the backbone's device;
      - `__call__(batch: Tensor) -> list[Tensor]`, the four float32 conv inputs of shape `(N, C, H, W)`;
      - `close()`, plus context-manager use.
  - In `tests/differential_uncertainty/convtu_fakes.py`:
    - `FakeBackbone(seed=0)`: 640 × 640 input pooled to 16 × 16, then stages of 2, 3, 4 and 5 channels at 16, 8, 4 and 2 cells;
    - `FakeTap`, a detector stand-in for the baselines `test` phase.

- [ ] **Step 1: Write the test helpers and the failing tests**

`tests/differential_uncertainty/convtu_fakes.py`:

```python
"""Small stand-ins for the RT-DETRv2 backbone and the detector tap, for the conv-TU tests."""
import numpy as np
import torch
from torch import nn


class _Branch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        return torch.relu(self.norm(self.conv(x)))


class _Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branch2a = _Branch(channels)

    def forward(self, x):
        return self.branch2a(x)


class _Stage(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Conv2d(c_in, c_out, 3, stride, 1), _Block(c_out)])

    def forward(self, x):
        return self.blocks[1](torch.relu(self.blocks[0](x)))


class FakeBackbone(nn.Module):
    """640 x 640 input pooled to 16 x 16, then four stages of 2, 3, 4 and 5 channels at 16, 8, 4 and 2 cells."""

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        widths = (3, 2, 3, 4, 5)
        self.pool = nn.AvgPool2d(40)
        self.res_layers = nn.ModuleList([_Stage(widths[s], widths[s + 1], 1 if s == 0 else 2) for s in range(4)])
        self.eval()

    def forward(self, x):
        x = self.pool(x)
        outputs = []
        for stage in self.res_layers:
            x = stage(x)
            outputs.append(x)
        return outputs


class FakeTap:
    """Detector stand-in for the baselines `test` phase: fixed boxes, brightness-driven confidence."""

    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def run(self, arrays, batch_size=32):
        n = len(arrays)
        brightness = np.array([a.mean() / 255.0 for a in arrays])
        logits = np.full((n, 300, 80), -6.0)
        logits[:, :3, 0] = (4.0 * brightness)[:, None]
        boxes = np.full((n, 300, 4), 0.25)
        pooled = np.random.default_rng(0).normal(size=(n, 512)) + brightness[:, None]
        return logits, boxes, pooled
```

`tests/differential_uncertainty/test_convtu_tap.py`:

```python
from pathlib import Path

import pytest
import torch

from convtu_fakes import FakeBackbone
from differential_uncertainty.convtu.graph import folded_kernel
from differential_uncertainty.convtu.tap import ConvInputs, LAYER_NAMES
from differential_uncertainty.extraction import load_frozen_detector

CHECKPOINT = Path("/home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth")


def test_conv_inputs_are_the_tensors_entering_each_pilot_conv():
    backbone = FakeBackbone()
    batch = torch.rand(2, 3, 640, 640)
    with ConvInputs(backbone) as taps:
        inputs = taps(batch)
    with torch.no_grad():
        x = backbone.pool(batch)
        expected = []
        for stage in backbone.res_layers:
            entering = torch.relu(stage.blocks[0](x))
            expected.append(entering)
            x = stage.blocks[1](entering)
    assert [tuple(t.shape) for t in inputs] == [(2, 2, 16, 16), (2, 3, 8, 8), (2, 4, 4, 4), (2, 5, 2, 2)]
    for got, want in zip(inputs, expected):
        assert torch.allclose(got, want)


def test_kernels_are_folded_with_each_branch_norm():
    backbone = FakeBackbone()
    with ConvInputs(backbone) as taps:
        for stage, kernel in zip(backbone.res_layers, taps.kernels):
            branch = stage.blocks[1].branch2a
            assert torch.equal(kernel, folded_kernel(branch.conv, branch.norm))


def test_closing_removes_the_hooks():
    backbone = FakeBackbone()
    taps = ConvInputs(backbone)
    taps.close()
    assert all(len(stage.blocks[1].branch2a.conv._forward_pre_hooks) == 0 for stage in backbone.res_layers)
    assert LAYER_NAMES[0] == "res_layers.0.blocks.1.branch2a.conv"


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="needs the RT-DETRv2-R18 checkpoint")
def test_the_real_backbone_gives_the_four_pilot_layers():
    model = load_frozen_detector(CHECKPOINT, torch.device("cpu"))
    with ConvInputs(model.backbone) as taps:
        inputs = taps(torch.rand(1, 3, 640, 640))
        kernel_shapes = [tuple(k.shape) for k in taps.kernels]
    assert [tuple(t.shape) for t in inputs] == [(1, 64, 160, 160), (1, 128, 80, 80), (1, 256, 40, 40), (1, 512, 20, 20)]
    assert kernel_shapes == [(64, 64, 3, 3), (128, 128, 3, 3), (256, 256, 3, 3), (512, 512, 3, 3)]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_convtu_tap.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'differential_uncertainty.convtu.tap'`.

- [ ] **Step 3: Write the implementation**

`differential_uncertainty/convtu/tap.py`:

```python
"""Inputs of one stride-1 3x3 conv per backbone stage of the frozen RT-DETRv2-R18, with their folded kernels."""
from __future__ import annotations

from functools import partial

import torch

from .graph import folded_kernel

STAGES = 4
LAYER_NAMES = tuple(f"res_layers.{s}.blocks.1.branch2a.conv" for s in range(STAGES))


class ConvInputs:
    """Captures the input of the first conv in the second block of each backbone stage."""

    def __init__(self, backbone: torch.nn.Module):
        self.backbone = backbone
        branches = [backbone.res_layers[s].blocks[1].branch2a for s in range(STAGES)]
        self.kernels = [folded_kernel(branch.conv, branch.norm) for branch in branches]
        self._inputs: list = [None] * STAGES
        self._handles = [branch.conv.register_forward_pre_hook(partial(self._capture, index))
                         for index, branch in enumerate(branches)]

    def _capture(self, index, _module, inputs):
        self._inputs[index] = inputs[0]

    @torch.inference_mode()
    def __call__(self, batch: torch.Tensor) -> list[torch.Tensor]:
        self._inputs = [None] * STAGES
        self.backbone(batch)
        if any(value is None for value in self._inputs):
            raise RuntimeError("a pilot conv input was not captured")
        return [value.float() for value in self._inputs]

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_convtu_tap.py -q`
Expected: `4 passed`. The real-checkpoint test takes a few seconds on the CPU.

- [ ] **Step 5: Commit**

```bash
git add differential_uncertainty/convtu/tap.py tests/differential_uncertainty/convtu_fakes.py tests/differential_uncertainty/test_convtu_tap.py
git commit -m "feat: capture the inputs of one stride-1 conv per backbone stage"
```

---

### Task 4: Fingerprint, controls and kNN

**Files:**
- Create: `differential_uncertainty/convtu/features.py`
- Test: `tests/differential_uncertainty/test_convtu_features.py`

**Interfaces:**
- Consumes: `graph.conv_top_merges`, `graph.fingerprint_length`, `graph.EDGE_CAP` (Tasks 1–2).
- Produces, in `differential_uncertainty.convtu.features`:
  - `REPRESENTATIONS = ("mst", "edges", "acts", "means")`, `KNN_NEIGHBOURS = 5`
  - `@dataclass(frozen=True) LayerSpec(stage: int, c_in: int, c_out: int, height: int, width: int)`, with properties `nodes`, `k` and `name`. `name` is `"s1"` … `"s4"`.
  - `layer_specs(inputs: list[Tensor], kernels: list[Tensor]) -> list[LayerSpec]`
  - `layer_features(x: Tensor, kernel_abs: Tensor, spec: LayerSpec, cut: float, cap: int = EDGE_CAP) -> dict`, with keys:
    - `mst`, `edges`, `acts`: `(k,)` float32;
    - `means`: `(c_in,)` float32;
    - `tau_k`: float;
    - `read`, `gathered`, `rounds`: int.
  - `knn_scores(queries: Tensor, bank: Tensor, neighbours: int = KNN_NEIGHBOURS, chunk: int = 64) -> np.ndarray`: shape `(n,)`, float64.

- [ ] **Step 1: Write the failing tests**

`tests/differential_uncertainty/test_convtu_features.py`:

```python
import numpy as np
import pytest
import torch

from differential_uncertainty.convtu import features, graph


def test_layer_specs_give_one_percent_of_the_nodes_for_the_real_layers():
    inputs = [torch.empty(1, c, h, h) for c, h in ((64, 160), (128, 80), (256, 40), (512, 20))]
    kernels = [torch.empty(c, c, 3, 3) for c in (64, 128, 256, 512)]
    specs = features.layer_specs(inputs, kernels)
    assert [s.name for s in specs] == ["s1", "s2", "s3", "s4"]
    assert [s.nodes for s in specs] == [3_276_800, 1_638_400, 819_200, 409_600]
    assert [s.k for s in specs] == [32_768, 16_384, 8_192, 4_096]


def test_layer_features_hold_the_fingerprint_and_the_three_controls():
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(3, 5, 6, generator=generator)
    x[x < 0.3] = 0.0
    kernel = torch.rand(2, 3, 3, 3, generator=generator)
    spec = features.LayerSpec(1, 3, 2, 5, 6)  # 150 nodes: a small graph keeps all 149 values
    got = features.layer_features(x, kernel, spec, cut=0.2)
    top = graph.conv_top_merges(x, kernel, 149, 0.2)
    np.testing.assert_array_equal(got["mst"], top.values)
    np.testing.assert_array_equal(got["edges"], top.heaviest)
    expected_acts = np.zeros(149, dtype=np.float32)
    expected_acts[:90] = np.sort(x.flatten().numpy())[::-1]
    np.testing.assert_array_equal(got["acts"], expected_acts)
    np.testing.assert_allclose(got["means"], x.mean(dim=(1, 2)).numpy())
    assert got["tau_k"] == float(top.values[-1]) and got["rounds"] >= 1 and got["gathered"] >= got["read"] > 0


def test_a_blank_input_gives_all_zero_features():
    spec = features.LayerSpec(1, 3, 2, 5, 6)
    got = features.layer_features(torch.zeros(3, 5, 6), torch.rand(2, 3, 3, 3), spec, cut=0.1)
    assert all(not got[key].any() for key in ("mst", "edges", "acts", "means"))


def test_knn_scores_are_mean_euclidean_distances_to_the_nearest_rows():
    rng = np.random.default_rng(1)
    queries, bank = rng.normal(size=(7, 4)), rng.normal(size=(20, 4))
    distances = np.sqrt(((queries[:, None, :] - bank[None, :, :]) ** 2).sum(axis=2))
    expected = np.sort(distances, axis=1)[:, :5].mean(axis=1)
    got = features.knn_scores(torch.from_numpy(queries), torch.from_numpy(bank), chunk=3)
    np.testing.assert_allclose(got, expected, rtol=1e-5)
    with pytest.raises(ValueError, match="neighbours"):
        features.knn_scores(torch.from_numpy(queries), torch.from_numpy(bank[:4]))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_convtu_features.py -q`
Expected: collection error `ImportError: cannot import name 'features'`.

- [ ] **Step 3: Write the implementation**

`differential_uncertainty/convtu/features.py`:

```python
"""Per-layer fingerprint of the conv graph, three simpler summaries of the same layer, and their kNN scores.

mst: the K largest diagram values (the method). edges: the K heaviest edge weights, the same numbers without
the cycle rule. acts: the K largest activations, no kernel and no graph (zero-padded when K exceeds the input
cells). means: the mean absolute activation of each input channel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .graph import EDGE_CAP, conv_top_merges, fingerprint_length

REPRESENTATIONS = ("mst", "edges", "acts", "means")
KNN_NEIGHBOURS = 5


@dataclass(frozen=True)
class LayerSpec:
    stage: int
    c_in: int
    c_out: int
    height: int
    width: int

    @property
    def nodes(self) -> int:
        return (self.c_in + self.c_out) * self.height * self.width

    @property
    def k(self) -> int:
        return fingerprint_length(self.nodes)

    @property
    def name(self) -> str:
        return f"s{self.stage}"


def layer_specs(inputs: list[torch.Tensor], kernels: list[torch.Tensor]) -> list[LayerSpec]:
    """One spec per captured conv input of shape (N, C, H, W) and its (C_out, C, 3, 3) kernel."""
    return [LayerSpec(stage + 1, x.shape[1], kernel.shape[0], x.shape[2], x.shape[3])
            for stage, (x, kernel) in enumerate(zip(inputs, kernels))]


def layer_features(x: torch.Tensor, kernel_abs: torch.Tensor, spec: LayerSpec, cut: float,
                   cap: int = EDGE_CAP) -> dict:
    """The fingerprint and the three controls of one image's conv input x, shape (C, H, W)."""
    x_abs = x.abs()
    top = conv_top_merges(x_abs, kernel_abs, spec.k, cut, cap)
    count = min(spec.k, x_abs.numel())
    acts = np.zeros(spec.k, dtype=np.float32)
    acts[:count] = torch.topk(x_abs.flatten(), count).values.cpu().numpy()
    return {"mst": top.values, "edges": top.heaviest.astype(np.float32), "acts": acts,
            "means": x_abs.mean(dim=(1, 2)).cpu().numpy().astype(np.float32),
            "tau_k": float(top.values[-1]), "read": int(top.read), "gathered": int(top.gathered),
            "rounds": int(top.rounds)}


def knn_scores(queries: torch.Tensor, bank: torch.Tensor, neighbours: int = KNN_NEIGHBOURS,
               chunk: int = 64) -> np.ndarray:
    """Mean Euclidean distance from each query row to its nearest `neighbours` bank rows (no normalization)."""
    if queries.ndim != 2 or bank.ndim != 2 or queries.shape[1] != bank.shape[1]:
        raise ValueError("queries and bank must be (rows, dim) with the same dim")
    if not 1 <= neighbours <= bank.shape[0]:
        raise ValueError("neighbours must be between 1 and the bank size")
    bank = bank.float()
    out = []
    for start in range(0, queries.shape[0], chunk):
        rows = queries[start:start + chunk].to(bank.device, torch.float32)
        distances = torch.cdist(rows, bank, compute_mode="donot_use_mm_for_euclid_dist")
        out.append(distances.topk(neighbours, dim=1, largest=False).values.mean(dim=1))
    return torch.cat(out).cpu().numpy().astype(np.float64)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_convtu_features.py -q`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add differential_uncertainty/convtu/features.py tests/differential_uncertainty/test_convtu_features.py
git commit -m "feat: conv-TU fingerprint, its three controls and an unnormalized kNN score"
```

---

### Task 5: Clean-train phases (calibrate, bank, z-statistics)

**Files:**
- Create: `differential_uncertainty/convtu/pipeline.py`
- Modify: `differential_uncertainty/baselines/pipeline.py`. After the `PHASES = {...}` dict, register the conv-TU phases.
- Modify: `differential_uncertainty/cli.py`, the `--phase` choices of `baselines-coco`.
- Test: `tests/differential_uncertainty/test_convtu_pipeline.py`; `tests/differential_uncertainty/test_cli.py` (append).

**Interfaces:**
- Consumes:
  - `ConvInputs` (Task 3);
  - `layer_specs`, `layer_features`, `knn_scores`, `REPRESENTATIONS`, `KNN_NEIGHBOURS` (Task 4);
  - `heaviest_weight`, `conv_top_merges`, `EDGE_CAP`, `FRACTION` (Tasks 1–2);
  - from `baselines.pipeline`: `Settings`, `_atomic_json`, `_atomic_npz`, `_progress`, `_train_loader`;
  - `baselines.activation_cdf.stage_zstats`.
- Produces, in `differential_uncertainty.convtu.pipeline`:
  - `CALIBRATION_IMAGES = 200`, `BANK_IMAGES = 2000`, `ZSTAT_IMAGES = 500`, `CUT_MARGIN = 0.5`, `START_MULTIPLE = 8`
  - `train_splits(count: int, seed: int) -> dict[str, np.ndarray]`, with keys `calibration`, `bank`, `zstats`
  - `calibration_path(settings)`, `bank_path(settings)`, `zstats_path(settings) -> Path`, under `settings.output / "convtu"`
  - `features_of(inputs, kernels, specs, cuts) -> list[dict]`, with keys `f"{key}_{spec.name}"` for `key` in `(*REPRESENTATIONS, "tau_k", "read", "rounds")`
  - `load_bank(settings, device) -> dict[str, Tensor]`
  - `layer_scores(rows, bank, specs) -> dict[str, np.ndarray]`: `(images, layers)` per representation
  - `_cuts(settings) -> dict[str, float]`, `_conv_inputs(settings) -> ConvInputs`: tests replace the latter
  - `PHASES` with `convtu-calibrate`, `convtu-bank`, `convtu-zstats`
- Output formats:
  - `convtu/calibration.json`: `{"images", "seed", "fraction", "cut_margin", "edge_cap", "peak_gpu_gib", "layers": {"s1": {"conv", "input_shape", "nodes", "k", "cut", "tau_k", "tau_k_cv", "edges_read", "edges_gathered_at_cut", "rounds_at_cut_max", "ms_per_image"}, …}}`. The quantile entries are `{"min", "p50", "p90", "p99", "max"}`.
  - `convtu/bank.npz`: `calibration_sha1` plus `f"{rep}_{s}"` arrays of shape `(2000, k)`; `means_*` are `(2000, C)`.
  - `convtu/zstats.json`: `{"images", "layers", "neighbours", "stats": {rep: {"mean": [4], "std": [4]}}, "bank_sha1"}`.

- [ ] **Step 1: Write the failing tests**

`tests/differential_uncertainty/test_convtu_pipeline.py`:

```python
import json

import numpy as np
import pytest
from PIL import Image

from convtu_fakes import FakeBackbone
from differential_uncertainty.baselines import pipeline as baselines
from differential_uncertainty.convtu import pipeline as convtu
from differential_uncertainty.convtu.tap import ConvInputs


@pytest.fixture
def small(monkeypatch, tmp_path):
    monkeypatch.setattr(convtu, "CALIBRATION_IMAGES", 2)
    monkeypatch.setattr(convtu, "BANK_IMAGES", 6)
    monkeypatch.setattr(convtu, "ZSTAT_IMAGES", 3)
    monkeypatch.setattr(convtu, "_conv_inputs", lambda _settings: ConvInputs(FakeBackbone()))
    rng = np.random.default_rng(3)
    for folder, count in (("train", 12), ("val", 3)):
        (tmp_path / folder).mkdir()
        for index in range(count):
            Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)).save(tmp_path / folder / f"{index:04d}.jpg")
    return baselines.Settings(output=tmp_path / "out", checkpoint=tmp_path / "ckpt.pth",
                              train_images=tmp_path / "train", val_images=tmp_path / "val",
                              annotations=tmp_path / "ann.json", discopatch_root=tmp_path,
                              limit=2, workers=0, device="cpu", batch_size=4)


def test_train_splits_are_disjoint_seeded_and_sized(monkeypatch):
    monkeypatch.setattr(convtu, "CALIBRATION_IMAGES", 2)
    monkeypatch.setattr(convtu, "BANK_IMAGES", 3)
    monkeypatch.setattr(convtu, "ZSTAT_IMAGES", 2)
    splits = convtu.train_splits(10, seed=44)
    assert [len(splits[k]) for k in ("calibration", "bank", "zstats")] == [2, 3, 2]
    assert len(set(np.concatenate(list(splits.values())).tolist())) == 7
    again = convtu.train_splits(10, seed=44)
    assert all(np.array_equal(splits[k], again[k]) for k in splits)
    with pytest.raises(ValueError, match="needs 7 train images"):
        convtu.train_splits(6, seed=44)


def test_clean_phases_write_calibration_bank_and_zstats_and_resume(small):
    for phase in ("convtu-calibrate", "convtu-bank", "convtu-zstats"):
        baselines.run_phase(phase, small)
    calibration = json.loads(convtu.calibration_path(small).read_text())
    assert calibration["images"] == 2 and set(calibration["layers"]) == {"s1", "s2", "s3", "s4"}
    s1 = calibration["layers"]["s1"]
    assert s1["nodes"] == 1024 and s1["k"] == 1023 and s1["cut"] > 0
    assert s1["edges_read"]["max"] >= s1["edges_read"]["min"] > 0
    with np.load(convtu.bank_path(small)) as bank:
        assert bank["mst_s1"].shape == (6, 1023) and bank["means_s4"].shape == (6, 5)
    zstats = json.loads(convtu.zstats_path(small).read_text())
    assert zstats["images"] == 3 and set(zstats["stats"]) == {"mst", "edges", "acts", "means"}
    assert all(value > 0 for value in zstats["stats"]["mst"]["std"])
    stamp = convtu.bank_path(small).stat().st_mtime_ns
    baselines.run_phase("convtu-bank", small)
    assert convtu.bank_path(small).stat().st_mtime_ns == stamp


def test_bank_needs_the_calibration(small):
    with pytest.raises(ValueError, match="convtu-calibrate phase first"):
        baselines.run_phase("convtu-bank", small)


def test_zstats_refuse_a_bank_from_another_calibration(small):
    for phase in ("convtu-calibrate", "convtu-bank"):
        baselines.run_phase(phase, small)
    path = convtu.calibration_path(small)
    record = json.loads(path.read_text())
    record["layers"]["s1"]["cut"] *= 2
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="calibration changed"):
        baselines.run_phase("convtu-zstats", small)
```

Append to `tests/differential_uncertainty/test_cli.py`:

```python
def test_cli_accepts_the_convtu_phases(monkeypatch, tmp_path):
    import differential_uncertainty.baselines.pipeline as pipeline
    seen = []
    monkeypatch.setattr(pipeline, "run_phase", lambda phase, settings: seen.append(phase))
    base = ["--output", str(tmp_path / "out"), "--checkpoint", "c.pth", "--coco-train-images", "train",
            "--coco-val-images", "val", "--coco-annotations", "ann.json", "--discopatch-root", "dcp"]
    phases = ("convtu-calibrate", "convtu-bank", "convtu-zstats")
    for phase in phases:
        assert cli.main(["baselines-coco", "--phase", phase, *base]) == 0
    assert tuple(seen) == phases
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_convtu_pipeline.py tests/differential_uncertainty/test_cli.py -q`
Expected:
- `test_convtu_pipeline.py` fails to collect with `ImportError: cannot import name 'pipeline' from 'differential_uncertainty.convtu'`.
- `test_cli_accepts_the_convtu_phases` fails with exit code 2, from argparse's `invalid choice`.

- [ ] **Step 3: Write the implementation**

`differential_uncertainty/convtu/pipeline.py`:

```python
"""Resumable phases of the conv-TU pilot on the fixed COCO protocol; they run through `baselines-coco --phase`.

Clean COCO train images give each layer's cut (convtu-calibrate), the kNN bank (convtu-bank) and the
z-statistics (convtu-zstats), from three disjoint seeded draws.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import torch

from ..baselines import protocol
from ..baselines.activation_cdf import stage_zstats
from ..baselines.pipeline import Settings, _atomic_json, _atomic_npz, _progress, _train_loader
from ..extraction import load_frozen_detector
from .features import KNN_NEIGHBOURS, REPRESENTATIONS, knn_scores, layer_features, layer_specs
from .graph import EDGE_CAP, FRACTION, conv_top_merges, heaviest_weight
from .tap import LAYER_NAMES, ConvInputs

CALIBRATION_IMAGES = 200
BANK_IMAGES = 2000
ZSTAT_IMAGES = 500
CUT_MARGIN = 0.5    # each layer's cut: half the smallest K-th value over the calibration images
START_MULTIPLE = 8  # calibration starts each image at its (8 K)-th heaviest edge


def _folder(settings: Settings):
    return settings.output / "convtu"


def calibration_path(settings: Settings):
    return _folder(settings) / "calibration.json"


def bank_path(settings: Settings):
    return _folder(settings) / "bank.npz"


def zstats_path(settings: Settings):
    return _folder(settings) / "zstats.json"


def _sha1(path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def train_splits(count: int, seed: int) -> dict:
    """Disjoint seeded draws of train-image indices for calibration, the bank and the z-statistics."""
    needed = CALIBRATION_IMAGES + BANK_IMAGES + ZSTAT_IMAGES
    if count < needed:
        raise ValueError(f"the pilot needs {needed} train images, found {count}")
    order = np.random.default_rng(seed).permutation(count)
    first, second = CALIBRATION_IMAGES, CALIBRATION_IMAGES + BANK_IMAGES
    return {"calibration": np.sort(order[:first]), "bank": np.sort(order[first:second]),
            "zstats": np.sort(order[second:needed])}


def _conv_inputs(settings: Settings) -> ConvInputs:
    return ConvInputs(load_frozen_detector(settings.checkpoint, torch.device(settings.device)).backbone)


def _clean_batches(settings: Settings, split: str):
    paths = protocol.list_images(settings.train_images)
    return _train_loader(settings, [paths[i] for i in train_splits(len(paths), settings.seed)[split]])


def _sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def _quantiles(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {"min": float(values.min()), "p50": float(np.median(values)), "p90": float(np.quantile(values, 0.9)),
            "p99": float(np.quantile(values, 0.99)), "max": float(values.max())}


def _last_positive(values: np.ndarray) -> float:
    """The K-th value, or the last positive one when a small graph's diagram ends in zeros."""
    positive = values[values > 0]
    if positive.size == 0:
        raise ValueError("a calibration image has no positive diagram value")
    return float(positive[-1])


def phase_calibrate(settings: Settings) -> None:
    """Where the K-th merge falls on clean train images, how many edges it takes, and each layer's cut."""
    path = calibration_path(settings)
    if path.exists():
        return
    on_gpu = torch.device(settings.device).type == "cuda"
    tau, read, specs = [], [], None
    with _conv_inputs(settings) as taps:
        for batch in _clean_batches(settings, "calibration"):
            inputs = taps(batch.to(settings.device))
            specs = specs or layer_specs(inputs, taps.kernels)
            for image in range(batch.shape[0]):
                row_tau, row_read = [], []
                for layer, spec in enumerate(specs):
                    x_abs = inputs[layer][image].abs()
                    start = heaviest_weight(x_abs, taps.kernels[layer], START_MULTIPLE * spec.k)
                    top = conv_top_merges(x_abs, taps.kernels[layer], spec.k, start)
                    row_tau.append(_last_positive(top.values))
                    row_read.append(top.read)
                tau.append(row_tau)
                read.append(row_read)
        tau, read = np.array(tau), np.array(read)
        cuts = CUT_MARGIN * tau.min(axis=0)
        if on_gpu:
            torch.cuda.reset_peak_memory_stats()
        gathered, ms, rounds = [], [], []
        for batch in _clean_batches(settings, "calibration"):
            inputs = taps(batch.to(settings.device))
            for image in range(batch.shape[0]):
                row_gathered, row_ms, row_rounds = [], [], []
                for layer, spec in enumerate(specs):
                    _sync(settings.device)
                    started = time.perf_counter()
                    features = layer_features(inputs[layer][image], taps.kernels[layer], spec, float(cuts[layer]))
                    _sync(settings.device)
                    row_ms.append(1000.0 * (time.perf_counter() - started))
                    row_gathered.append(features["gathered"])
                    row_rounds.append(features["rounds"])
                gathered.append(row_gathered)
                ms.append(row_ms)
                rounds.append(row_rounds)
    gathered, ms, rounds = np.array(gathered), np.array(ms), np.array(rounds)
    layers = {}
    for layer, spec in enumerate(specs):
        layers[spec.name] = {
            "conv": LAYER_NAMES[layer], "input_shape": [spec.c_in, spec.height, spec.width],
            "nodes": spec.nodes, "k": spec.k, "cut": float(cuts[layer]),
            "tau_k": _quantiles(tau[:, layer]), "tau_k_cv": float(tau[:, layer].std() / tau[:, layer].mean()),
            "edges_read": _quantiles(read[:, layer]), "edges_gathered_at_cut": _quantiles(gathered[:, layer]),
            "rounds_at_cut_max": int(rounds[:, layer].max()), "ms_per_image": _quantiles(ms[:, layer]),
        }
    _atomic_json(path, {"images": int(tau.shape[0]), "seed": settings.seed, "fraction": FRACTION,
                        "cut_margin": CUT_MARGIN, "edge_cap": EDGE_CAP,
                        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if on_gpu else None,
                        "layers": layers})


def _cuts(settings: Settings) -> dict:
    path = calibration_path(settings)
    if not path.exists():
        raise ValueError("run the convtu-calibrate phase first")
    return {name: layer["cut"] for name, layer in json.loads(path.read_text())["layers"].items()}


def features_of(inputs, kernels, specs, cuts) -> list[dict]:
    """Per image of the batch: every representation of every layer, plus the per-layer diagnostics."""
    rows = []
    for image in range(inputs[0].shape[0]):
        row = {}
        for layer, spec in enumerate(specs):
            features = layer_features(inputs[layer][image], kernels[layer], spec, cuts[spec.name])
            for key in (*REPRESENTATIONS, "tau_k", "read", "rounds"):
                row[f"{key}_{spec.name}"] = features[key]
        rows.append(row)
    return rows


def phase_bank(settings: Settings) -> None:
    """The fingerprint and the three controls of 2,000 clean train images: the kNN bank."""
    path = bank_path(settings)
    if path.exists():
        return
    cuts = _cuts(settings)
    rows, specs, started = [], None, time.time()
    with _conv_inputs(settings) as taps:
        for batch in _clean_batches(settings, "bank"):
            inputs = taps(batch.to(settings.device))
            specs = specs or layer_specs(inputs, taps.kernels)
            rows += features_of(inputs, taps.kernels, specs, cuts)
            _progress("convtu-bank", len(rows), BANK_IMAGES, started)
    arrays = {f"{rep}_{spec.name}": np.stack([row[f"{rep}_{spec.name}"] for row in rows])
              for rep in REPRESENTATIONS for spec in specs}
    _atomic_npz(path, calibration_sha1=np.array(_sha1(calibration_path(settings))), **arrays)


def load_bank(settings: Settings, device) -> dict:
    path = bank_path(settings)
    if not path.exists():
        raise ValueError("run the convtu-bank phase first")
    with np.load(path, allow_pickle=False) as data:
        if str(data["calibration_sha1"]) != _sha1(calibration_path(settings)):
            raise ValueError("the calibration changed since the bank was built")
        return {key: torch.from_numpy(data[key]).to(device) for key in data.files if key != "calibration_sha1"}


def layer_scores(rows, bank, specs) -> dict:
    """Per representation, (images, layers) mean distances to the nearest bank rows."""
    out = {}
    for rep in REPRESENTATIONS:
        columns = [knn_scores(torch.from_numpy(np.stack([row[f"{rep}_{spec.name}"] for row in rows])),
                              bank[f"{rep}_{spec.name}"]) for spec in specs]
        out[rep] = np.stack(columns, axis=1)
    return out


def phase_zstats(settings: Settings) -> None:
    """Mean and spread of each layer's kNN score over 500 clean train images outside the bank."""
    path = zstats_path(settings)
    if path.exists():
        return
    cuts = _cuts(settings)
    bank = load_bank(settings, settings.device)
    rows, specs = [], None
    with _conv_inputs(settings) as taps:
        for batch in _clean_batches(settings, "zstats"):
            inputs = taps(batch.to(settings.device))
            specs = specs or layer_specs(inputs, taps.kernels)
            rows += features_of(inputs, taps.kernels, specs, cuts)
    stats = {}
    for rep, values in layer_scores(rows, bank, specs).items():
        mean, std = stage_zstats(values)
        stats[rep] = {"mean": mean.tolist(), "std": std.tolist()}
    _atomic_json(path, {"images": len(rows), "layers": [spec.name for spec in specs],
                        "neighbours": KNN_NEIGHBOURS, "stats": stats, "bank_sha1": _sha1(bank_path(settings))})


PHASES = {"convtu-calibrate": phase_calibrate, "convtu-bank": phase_bank, "convtu-zstats": phase_zstats}
```

In `differential_uncertainty/baselines/pipeline.py`, directly after the `PHASES = {...}` dict and before `def run_phase`:

```python
CONVTU_PHASES = ("convtu-calibrate", "convtu-bank", "convtu-zstats")


def _convtu_phase(name: str):
    def run(settings: Settings) -> None:
        from ..convtu import pipeline as convtu  # imported late: convtu.pipeline imports this module
        convtu.PHASES[name](settings)
    return run


PHASES.update({name: _convtu_phase(name) for name in CONVTU_PHASES})
```

In `differential_uncertainty/cli.py`, the choices list of `baselines-coco --phase` becomes:

```python
                           choices=["sanity", "bank", "test", "train-discopatch", "discopatch-scores",
                                    "hashemi-fit", "cdf-fit", "cdf-zstats", "activation-scores", "timing", "report",
                                    "convtu-calibrate", "convtu-bank", "convtu-zstats"])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_convtu_pipeline.py tests/differential_uncertainty/test_cli.py -q`
Expected: all pass (4 in the new file, and the CLI tests including the new one).

- [ ] **Step 5: Commit**

```bash
git add differential_uncertainty/convtu/pipeline.py differential_uncertainty/baselines/pipeline.py differential_uncertainty/cli.py tests/differential_uncertainty/test_convtu_pipeline.py tests/differential_uncertainty/test_cli.py
git commit -m "feat: calibrate the conv-TU cut and build its bank and z-statistics on clean train images"
```

---

### Task 6: Scoring phase over the pilot images

**Files:**
- Modify: `differential_uncertainty/convtu/pipeline.py` (append; add imports).
- Modify: `differential_uncertainty/baselines/pipeline.py`: `CONVTU_PHASES` gains `"convtu-scores"`.
- Modify: `differential_uncertainty/cli.py`: the choices gain `"convtu-scores"`.
- Test: `tests/differential_uncertainty/test_convtu_pipeline.py` (append); `tests/differential_uncertainty/test_cli.py`, where the tuple `phases` in `test_cli_accepts_the_convtu_phases` gains `"convtu-scores"`.

**Interfaces:**
- Consumes:
  - from Task 5: `features_of`, `layer_scores`, `load_bank`, `_cuts`, `_conv_inputs`, `calibration_path`, `bank_path`, `zstats_path`, `_sha1`;
  - from `baselines.pipeline`: `TEST_KEYS`, `_load_npz`, `_valid_existing`, `_variant_stream`, `evaluation`;
  - `protocol.digest`, `prepare_image`, `activation_cdf.zscored_sum`.
- Produces:
  - `PILOT_IMAGES = 200`, `IMAGE_SIZE = (640, 640)`
  - `SCORE_KEYS = ("mst", "edges", "acts", "means", "mst_layers", "edges_layers", "acts_layers", "means_layers", "tau_k", "read", "rounds")`
  - `pilot_images(settings) -> list[Path]`
  - `image_scores(taps, arrays, bank, zstats, cuts, batch_size) -> dict`
  - `phase_scores`, and `PHASES["convtu-scores"]`
- Output format: `test_convtu/{stem}.npz`:
  - `mst`, `edges`, `acts`, `means`: `(96,)` z-scored sums over the four layers;
  - `*_layers`: `(96, 4)` raw kNN distances;
  - `tau_k`, `read`, `rounds`: `(96, 4)`.

  Plus `test_convtu/fits.json`, which records the sha1 of the calibration, the bank and the z-statistics.

- [ ] **Step 1: Write the failing tests** (append to `tests/differential_uncertainty/test_convtu_pipeline.py`)

```python
import torch

from convtu_fakes import FakeTap


@pytest.fixture
def detector(monkeypatch):
    monkeypatch.setattr(baselines, "DetectorTap", FakeTap)
    monkeypatch.setattr(baselines, "_load_bank",
                        lambda _s, _d: torch.nn.functional.normalize(torch.randn(256, 512), dim=1))


def _pilot(settings):
    for phase in ("test", "convtu-calibrate", "convtu-bank", "convtu-zstats", "convtu-scores"):
        baselines.run_phase(phase, settings)


def test_scores_phase_writes_every_condition_and_resumes(small, detector, monkeypatch):
    monkeypatch.setattr(convtu, "PILOT_IMAGES", 2)
    _pilot(small)
    files = sorted((small.output / "test_convtu").glob("*.npz"))
    assert len(files) == 2
    with np.load(files[0]) as scores:
        assert scores["mst"].shape == (96,) and scores["mst_layers"].shape == (96, 4)
        assert scores["tau_k"].shape == (96, 4) and np.isfinite(scores["acts"]).all()
    stamp = files[0].stat().st_mtime_ns
    baselines.run_phase("convtu-scores", small)
    assert files[0].stat().st_mtime_ns == stamp


def test_scores_phase_detects_changed_corruptions(small, detector, monkeypatch):
    monkeypatch.setattr(convtu, "PILOT_IMAGES", 2)
    _pilot(small)
    stem = sorted((small.output / "test_convtu").glob("*.npz"))[0].stem
    (small.output / "test_convtu" / f"{stem}.npz").unlink()
    stored = dict(np.load(small.output / "test" / f"{stem}.npz"))
    stored["digests"] = stored["digests"].copy()
    stored["digests"][5] = "0" * 16
    np.savez(small.output / "test" / f"{stem}.npz", **stored)
    with pytest.raises(ValueError, match="corruptions differ"):
        baselines.run_phase("convtu-scores", small)


def test_scores_phase_refuses_changed_fits(small, detector, monkeypatch):
    monkeypatch.setattr(convtu, "PILOT_IMAGES", 2)
    _pilot(small)
    path = convtu.zstats_path(small)
    record = json.loads(path.read_text())
    record["images"] += 1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="changed since"):
        baselines.run_phase("convtu-scores", small)
```

In `tests/differential_uncertainty/test_cli.py`, change the tuple in `test_cli_accepts_the_convtu_phases` to:

```python
    phases = ("convtu-calibrate", "convtu-bank", "convtu-zstats", "convtu-scores")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_convtu_pipeline.py tests/differential_uncertainty/test_cli.py -q`
Expected:
- the three new pipeline tests fail with `ValueError: unknown phase 'convtu-scores'`;
- the CLI test fails with exit code 2.

- [ ] **Step 3: Write the implementation**

In `differential_uncertainty/convtu/pipeline.py`:
- add `from pathlib import Path` and `from PIL import Image`;
- change the two imports from the baselines package to:

```python
from ..baselines.activation_cdf import stage_zstats, zscored_sum
from ..baselines.pipeline import (TEST_KEYS, Settings, _atomic_json, _atomic_npz, _load_npz, _progress,
                                  _train_loader, _valid_existing, _variant_stream, evaluation)
from ..extraction import load_frozen_detector, prepare_image
```

Add below `START_MULTIPLE`:

```python
PILOT_IMAGES = 200  # the first images of the seed-44 evaluation order, all 96 conditions
IMAGE_SIZE = (640, 640)
SCORE_KEYS = (*REPRESENTATIONS, *(f"{rep}_layers" for rep in REPRESENTATIONS), "tau_k", "read", "rounds")
```

Append before `PHASES`, and extend `PHASES`:

```python
def pilot_images(settings: Settings) -> list:
    return evaluation(settings)[:PILOT_IMAGES]


def image_scores(taps, arrays, bank, zstats, cuts, batch_size) -> dict:
    """Every condition of one image: per-layer kNN distances, their z-scored sums and the diagnostics."""
    rows, specs = [], None
    for start in range(0, len(arrays), batch_size):
        batch = torch.stack([prepare_image(Image.fromarray(a), IMAGE_SIZE) for a in arrays[start:start + batch_size]])
        inputs = taps(batch.to(taps.kernels[0].device))
        specs = specs or layer_specs(inputs, taps.kernels)
        rows += features_of(inputs, taps.kernels, specs, cuts)
    out = {}
    for rep, values in layer_scores(rows, bank, specs).items():
        stats = zstats["stats"][rep]
        out[f"{rep}_layers"] = values
        out[rep] = zscored_sum(values, stats["mean"], stats["std"])
    for key in ("tau_k", "read", "rounds"):
        out[key] = np.array([[row[f"{key}_{spec.name}"] for spec in specs] for row in rows])
    if not all(np.isfinite(value).all() for value in out.values()):
        raise ValueError("conv-TU produced non-finite scores")
    return out


def phase_scores(settings: Settings) -> None:
    """The fingerprint and its three controls for every condition of the pilot images."""
    fits = {"convtu-calibrate": calibration_path(settings), "convtu-bank": bank_path(settings),
            "convtu-zstats": zstats_path(settings)}
    missing = [phase for phase, path in fits.items() if not path.exists()]
    if missing:
        raise ValueError("run the " + " and ".join(missing) + " phase first")
    folder = settings.output / "test_convtu"
    record = {phase: {"path": str(path), "sha1": _sha1(path)} for phase, path in fits.items()}
    marker = folder / "fits.json"
    if marker.exists():
        if json.loads(marker.read_text()) != record:
            raise ValueError(f"the calibration, bank or z-statistics changed since {marker} was written")
    else:
        _atomic_json(marker, record)
    pending = [p for p in pilot_images(settings) if not _valid_existing(folder / f"{p.stem}.npz", SCORE_KEYS)]
    absent = [p.name for p in pending if not (settings.output / "test" / f"{p.stem}.npz").exists()]
    if absent:
        raise ValueError(f"run the test phase first: {len(absent)} detector results are missing, e.g. {absent[0]}")
    if not pending:
        return
    cuts = _cuts(settings)
    bank = load_bank(settings, settings.device)
    zstats = json.loads(fits["convtu-zstats"].read_text())
    if zstats["bank_sha1"] != _sha1(bank_path(settings)):
        raise ValueError("the bank changed since the z-statistics were computed")
    started = time.time()
    with _conv_inputs(settings) as taps:
        for done, (name, arrays) in enumerate(_variant_stream(settings, pending), start=1):
            stem = Path(name).stem
            stored = _load_npz(settings.output / "test" / f"{stem}.npz", TEST_KEYS)["digests"]
            if list(stored) != [protocol.digest(a) for a in arrays]:
                raise ValueError(f"corruptions differ from the detector pass for {name}")
            _atomic_npz(folder / f"{stem}.npz", **image_scores(taps, arrays, bank, zstats, cuts, settings.batch_size))
            if done % 5 == 0:
                _progress("convtu-scores", done, len(pending), started)


PHASES = {"convtu-calibrate": phase_calibrate, "convtu-bank": phase_bank, "convtu-zstats": phase_zstats,
          "convtu-scores": phase_scores}
```

In `baselines/pipeline.py`: `CONVTU_PHASES = ("convtu-calibrate", "convtu-bank", "convtu-zstats", "convtu-scores")`. In `cli.py`, append `"convtu-scores"` to the choices.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_convtu_pipeline.py tests/differential_uncertainty/test_cli.py -q`
Expected: all pass. The three scoring tests build 96 corruptions of two small images each, so they take tens of seconds.

- [ ] **Step 5: Commit**

```bash
git add differential_uncertainty/convtu/pipeline.py differential_uncertainty/baselines/pipeline.py differential_uncertainty/cli.py tests/differential_uncertainty/test_convtu_pipeline.py tests/differential_uncertainty/test_cli.py
git commit -m "feat: score every condition of the pilot images with conv TU and its controls"
```

---

### Task 7: Pilot report

**Files:**
- Modify: `differential_uncertainty/baselines/report.py`: `_markdown` and `write_outputs` take `methods`, `labels` and `title`, with the current values as defaults.
- Create: `differential_uncertainty/convtu/report.py`
- Modify: `differential_uncertainty/convtu/pipeline.py`: add `phase_report`, plus a `PHASES` entry.
- Modify: `differential_uncertainty/baselines/pipeline.py`: `CONVTU_PHASES` gains `"convtu-report"`.
- Modify: `differential_uncertainty/cli.py`: the choices gain `"convtu-report"`.
- Test:
  - `tests/differential_uncertainty/test_baselines_report.py` (append);
  - `tests/differential_uncertainty/test_convtu_report.py`;
  - `tests/differential_uncertainty/test_cli.py`, where the tuple gains `"convtu-report"`.

**Interfaces:**
- Consumes:
  - from `baselines.report`: `method_scores`, `separation_rows`, `aggregate_rows`, `harm_rows`, `headline_numbers`, `write_outputs`, `_stack`, `_fmt`, `METHODS`, `LABELS`, `TEST_ARRAYS`, `ACTIVATION_ARRAYS`, `CORRUPTED`, `COMMON`, `EXTRA`;
  - `metrics.bootstrap`, `metrics.condition_aurocs`, `metrics.mean_within_condition_spearman`;
  - `coco_quality.CocoGroundTruth`, `coco_map`, `coco_results`, `image_lrp`;
  - `runs/coco-baselines/results/summary.json`, written by the full baseline report: `lrp_threshold`, `lambda_per_fold`, `lambda_folds_agree`.
- Produces:
  - `write_outputs(folder, tables, summary, methods=METHODS, labels=LABELS, title="# COCO baseline numbers")`
  - In `differential_uncertainty.convtu.report`: `METHODS`, `LABELS`, `BOOTSTRAP_SAMPLES = 1000`, `depth_rows(conv, lrp) -> list[dict]`, `build_pilot_report(settings, names) -> None`
  - `results_convtu/`: the CSV files `separation`, `aggregates`, `harm`, `aurc_pools`, `intervals`, `differences` and `depth`, plus `summary.json` and `report.md`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/differential_uncertainty/test_baselines_report.py`:

```python
def test_write_outputs_takes_other_methods_labels_and_title(tmp_path):
    scores = {"mine": np.random.default_rng(0).normal(size=(6, len(protocol.CONDITIONS)))}
    rows = report.separation_rows(scores, protocol.assign_folds(6))
    report.write_outputs(tmp_path, {"separation": rows, "aggregates": report.aggregate_rows(rows)}, {"x": 1},
                         methods=("mine",), labels={"mine": "My method"}, title="# Pilot")
    text = (tmp_path / "report.md").read_text()
    assert text.startswith("# Pilot") and "My method" in text
```

`tests/differential_uncertainty/test_convtu_report.py`:

```python
import json
from pathlib import Path

import numpy as np

from differential_uncertainty.baselines import pipeline as baselines
from differential_uncertainty.convtu import pipeline as convtu
from differential_uncertainty.convtu import report as pilot
from test_baselines_report import _tiny_run


def test_pilot_report_puts_the_fingerprint_next_to_the_baselines(tmp_path, monkeypatch):
    settings = _tiny_run(tmp_path, monkeypatch)   # 15 images through the fake test phase
    baselines.run_phase("report", settings)       # the full report: stored LRP threshold and per-fold lambda
    monkeypatch.setattr(convtu, "PILOT_IMAGES", 15)
    monkeypatch.setattr(pilot, "BOOTSTRAP_SAMPLES", 5)
    folder = settings.output / "test_convtu"
    folder.mkdir()
    rng = np.random.default_rng(0)
    for path in convtu.pilot_images(settings):
        layers = {rep: rng.normal(size=(96, 4)) + np.linspace(0, 2, 96)[:, None] for rep in ("mst", "edges", "acts", "means")}
        arrays = {f"{rep}_layers": values for rep, values in layers.items()}
        arrays.update({rep: values.sum(axis=1) for rep, values in layers.items()})
        np.savez(folder / f"{Path(path.name).stem}.npz", **arrays)
    (settings.output / "convtu").mkdir()
    (settings.output / "convtu" / "calibration.json").write_text(json.dumps(
        {"fraction": 0.01, "cut_margin": 0.5, "layers": {"s1": {"k": 32768}}}))

    baselines.run_phase("convtu-report", settings)

    results = settings.output / "results_convtu"
    summary = json.loads((results / "summary.json").read_text())
    assert summary["images"] == 15 and summary["lrp_threshold"] == json.loads(
        (settings.output / "results" / "summary.json").read_text())["lrp_threshold"]
    text = (results / "report.md").read_text()
    assert text.startswith("# Conv TU pilot") and "Conv TU: top 1% of the diagram" in text
    assert "Depth: one layer at a time" in text and "ContrastiveConf" in text
    assert len((results / "depth.csv").read_text().splitlines()) == 1 + 16
    assert "convtu_mst:auroc_common" in (results / "intervals.csv").read_text()
```

In `tests/differential_uncertainty/test_cli.py`, the tuple becomes:

```python
    phases = ("convtu-calibrate", "convtu-bank", "convtu-zstats", "convtu-scores", "convtu-report")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_baselines_report.py tests/differential_uncertainty/test_convtu_report.py tests/differential_uncertainty/test_cli.py -q`
Expected:
- the `write_outputs` test fails with `TypeError: write_outputs() got an unexpected keyword argument 'methods'`;
- the pilot test fails at collection with `ImportError: cannot import name 'report' from 'differential_uncertainty.convtu'`;
- the CLI test fails with exit code 2.

- [ ] **Step 3: Write the implementation**

In `differential_uncertainty/baselines/report.py`, change the two signatures and the three places that read the globals:

```python
def _markdown(tables: dict, summary: dict, methods=METHODS, labels=LABELS, title="# COCO baseline numbers") -> str:
    intervals = {r["quantity"]: r for r in tables.get("intervals", []) + tables.get("differences", [])}
    aggregates = tables.get("aggregates", [])
    present = [m for m in methods if any(r["method"] == m for r in aggregates)]
    lines = [title, "",
```

The rest of the header lines stay as they are. In the separation loop, replace `{LABELS[method]}` with `{labels[method]}`. In the harm loop, replace `{LABELS[m]}` with `{labels[m]}`. Then:

```python
def write_outputs(folder, tables: dict, summary: dict, methods=METHODS, labels=LABELS,
                  title="# COCO baseline numbers") -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        if rows:
            _write_csv(folder / f"{name}.csv", rows)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    (folder / "report.md").write_text(_markdown(tables, summary, methods, labels, title))
```

`differential_uncertainty/convtu/report.py`:

```python
"""Pilot tables: the conv-TU fingerprint and its three controls next to the nine baselines, on the pilot images.

The baselines are restricted to the same images and use the full run's stored LRP threshold and per-fold
ContrastiveConf lambda, so their rows differ from docs/coco-baseline-numbers.md only through the image subset.
"""
from __future__ import annotations

import json

import numpy as np

from ..baselines import metrics, protocol, report
from ..baselines.coco_quality import CocoGroundTruth, coco_map, coco_results, image_lrp
from .features import REPRESENTATIONS

METHODS = report.METHODS + tuple(f"convtu_{rep}" for rep in REPRESENTATIONS)
LABELS = {**report.LABELS,
          "convtu_mst": "Conv TU: top 1% of the diagram",
          "convtu_edges": "Control: top 1% heaviest edges",
          "convtu_acts": "Control: largest activations",
          "convtu_means": "Control: channel means"}
BOOTSTRAP_SAMPLES = 1000


def depth_rows(conv: dict, lrp: np.ndarray) -> list[dict]:
    """Per stage and representation: separation and harm tracking of that one layer's kNN distance."""
    delta_risk = lrp[:, report.CORRUPTED] - lrp[:, [0]]
    rows = []
    for rep in REPRESENTATIONS:
        values = conv[f"{rep}_layers"]
        for layer in range(values.shape[2]):
            v = values[:, :, layer]
            rows.append({"representation": rep, "stage": layer + 1,
                         "auroc_common": float(metrics.condition_aurocs(v[:, 0], v[:, report.COMMON].T).mean()),
                         "auroc_extra": float(metrics.condition_aurocs(v[:, 0], v[:, report.EXTRA].T).mean()),
                         "rho_within": metrics.mean_within_condition_spearman(
                             v[:, report.CORRUPTED] - v[:, [0]], delta_risk)})
    return rows


def _depth_markdown(rows: list[dict]) -> str:
    lines = ["## Depth: one layer at a time", "",
             "| Representation | Stage | AUROC common | AUROC extra | ρ(Δscore, ΔLRP) within |",
             "| --- | ---: | ---: | ---: | ---: |"]
    lines += [f"| {r['representation']} | {r['stage']} | {report._fmt(r['auroc_common'])} | "
              f"{report._fmt(r['auroc_extra'])} | {report._fmt(r['rho_within'])} |" for r in rows]
    return "\n".join(lines) + "\n"


def build_pilot_report(settings, names) -> None:
    names = list(names)
    folds = protocol.assign_folds(len(names))
    baseline = json.loads((settings.output / "results" / "summary.json").read_text())
    test = report._stack(settings.output / "test", names, report.TEST_ARRAYS)
    dcp_folder = settings.output / "test_dcp"
    dcp = report._stack(dcp_folder, names, ("dcp",))["dcp"] if dcp_folder.exists() else None
    activation_folder = settings.output / "test_activation"
    activation = (report._stack(activation_folder, names, report.ACTIVATION_ARRAYS)
                  if activation_folder.exists() else None)
    conv = report._stack(settings.output / "test_convtu", names,
                         (*REPRESENTATIONS, *(f"{rep}_layers" for rep in REPRESENTATIONS)))
    per_image_lambda = np.array([baseline["lambda_per_fold"][str(f)] for f in folds])
    per_fold_methods = () if baseline["lambda_folds_agree"] else ("contrastive",)
    ordered = report.method_scores(test, dcp, per_image_lambda, activation=activation)
    ordered.update({f"convtu_{rep}": conv[rep] for rep in REPRESENTATIONS})
    scores = {m: ordered[m] for m in METHODS if m in ordered}

    gt = CocoGroundTruth(settings.annotations)
    ids = [gt.image_id(n) for n in names]
    threshold = baseline["lrp_threshold"]
    lrp = np.full((len(names), len(protocol.CONDITIONS)), np.nan)
    for k, image_id in enumerate(ids):
        gt_boxes, gt_labels, crowd = gt.boxes(image_id)
        for c in range(len(protocol.CONDITIONS)):
            lrp[k, c] = image_lrp(test["det_scores"][k, c], test["det_labels"][k, c], test["det_boxes"][k, c],
                                  gt_boxes, gt_labels, crowd, threshold)
    condition_map = np.array([
        coco_map(gt, [r for k, i in enumerate(ids) for r in coco_results(
            i, test["det_scores"][k, c], test["det_labels"][k, c], test["det_boxes"][k, c], gt.category_ids)], ids)
        for c in range(len(protocol.CONDITIONS))])

    separation = report.separation_rows(scores, folds, per_fold_methods=per_fold_methods)
    harm, pools = report.harm_rows(scores, lrp, condition_map)
    point = report.headline_numbers(scores, lrp, folds, per_fold_methods)

    def statistic(draw):
        return report.headline_numbers({m: v[draw] for m, v in scores.items()}, lrp[draw], folds[draw],
                                       per_fold_methods)

    ranges = metrics.bootstrap(statistic, len(names), samples=BOOTSTRAP_SAMPLES, seed=settings.seed)
    interval_rows = [{"quantity": key, "point": point[key], "low": ranges[key][0], "high": ranges[key][1]}
                     for key in point]
    depth = depth_rows(conv, lrp)
    calibration = json.loads((settings.output / "convtu" / "calibration.json").read_text())
    summary = {"images": len(names), "chosen_as": "first images of the seed-44 evaluation order",
               "lrp_threshold": threshold, "lambda_per_fold": baseline["lambda_per_fold"],
               "contrastive_lambda": "stored per-fold values of the full run, not refitted per bootstrap draw",
               "bootstrap_samples": BOOTSTRAP_SAMPLES, "clean_map_of_these_images": float(condition_map[0]),
               "fraction": calibration["fraction"], "cut_margin": calibration["cut_margin"],
               "calibration": calibration["layers"]}
    folder = settings.output / "results_convtu"
    report.write_outputs(folder, {
        "separation": separation, "aggregates": report.aggregate_rows(separation), "harm": harm,
        "aurc_pools": pools, "intervals": [r for r in interval_rows if " - " not in r["quantity"]],
        "differences": [r for r in interval_rows if " - " in r["quantity"]], "depth": depth,
    }, summary, methods=METHODS, labels=LABELS, title="# Conv TU pilot")
    with (folder / "report.md").open("a") as handle:
        handle.write("\n" + _depth_markdown(depth))
```

In `differential_uncertainty/convtu/pipeline.py`, before `PHASES`:

```python
def phase_report(settings: Settings) -> None:
    from .report import build_pilot_report
    build_pilot_report(settings, [p.name for p in pilot_images(settings)])
```

`PHASES` gains `"convtu-report": phase_report`; `CONVTU_PHASES` in `baselines/pipeline.py` gains `"convtu-report"`; the CLI choices gain `"convtu-report"`.

- [ ] **Step 4: Run the tests, then the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_baselines_report.py tests/differential_uncertainty/test_convtu_report.py tests/differential_uncertainty/test_cli.py -q`
Expected: all pass.

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty -q`
Expected: every earlier test still passes (228 passed and 2 skipped before this plan) and every new test passes; `0 failed`.

- [ ] **Step 5: Commit**

```bash
git add differential_uncertainty/baselines/report.py differential_uncertainty/convtu/report.py differential_uncertainty/convtu/pipeline.py differential_uncertainty/baselines/pipeline.py differential_uncertainty/cli.py tests/differential_uncertainty/test_baselines_report.py tests/differential_uncertainty/test_convtu_report.py tests/differential_uncertainty/test_cli.py
git commit -m "feat: pilot report for conv TU next to the baselines on the same images"
```

---

### Task 8: Run the pilot and publish the numbers

**Files:**
- Create: `$WORKSPACE/convtu-launch.sh`, in the plan's `.superpowers/sdd/…` workspace, not in git.
- Create: `docs/conv-tu-pilot-results.md`
- Create: `docs/results/conv-tu-pilot/` (copies of `runs/coco-baselines/results_convtu/*` and `convtu/calibration.json`)

**Interfaces:**
- Consumes: the five phases (Tasks 5–7) through `baselines-coco --phase`.
- Produces: the published pilot numbers.

- [ ] **Step 1: Agree a GPU window**

List the sessions with ListAgents. Send `explore` and `hunk-0927` a SendMessage that asks for a window of about two hours on the RTX 5090. Say that peak use is expected under 6 GiB (the calibration peak plus the 1.5 GB bank), so the pilot can run next to a small job. Start nothing on the GPU until both have answered.

- [ ] **Step 2: Write the launch script**

`$WORKSPACE/convtu-launch.sh`:

```bash
#!/usr/bin/env bash
# usage: convtu-launch.sh PHASE WORKERS — runs one baselines-coco phase detached from the Claude session.
set -euo pipefail
PHASE=$1
WORKERS=$2
cd /home/yuchen/YuchenZ/UE/philip_sa/.worktrees/convtu-pilot
OUT=/home/yuchen/YuchenZ/UE/philip_sa/runs/coco-baselines
mkdir -p "$OUT/logs"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
setsid nohup /home/yuchen/miniconda3/envs/UE/bin/python -m differential_uncertainty baselines-coco \
  --phase "$PHASE" --workers "$WORKERS" --output "$OUT" \
  --checkpoint /home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \
  --coco-train-images /home/yuchen/YuchenZ/Datasets/coco/train2017 \
  --coco-val-images /home/yuchen/YuchenZ/Datasets/coco/val2017 \
  --coco-annotations /home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json \
  --discopatch-root /home/yuchen/YuchenZ/UE/DisCoPatch \
  >> "$OUT/logs/$PHASE.log" 2>&1 < /dev/null &
echo $! > "$OUT/logs/$PHASE.pid"
echo "started $PHASE pid $(cat "$OUT/logs/$PHASE.pid")"
```

Run `bash $WORKSPACE/convtu-launch.sh convtu-calibrate 9`. Wait on the PID in `logs/convtu-calibrate.pid` (not `pgrep -f`, which matches its own shell).
Expected: `runs/coco-baselines/convtu/calibration.json` exists, and `logs/convtu-calibrate.log` ends without a traceback.

- [ ] **Step 3: Check the calibration gate**

Read `convtu/calibration.json`. All four must hold:
- (a) `peak_gpu_gib` ≤ 4.5. The scoring phase adds the bank, about 1.5 GB, on top of this peak.
- (b) `rounds_at_cut_max` ≤ 2 in every layer;
- (c) `edges_gathered_at_cut.max` ≤ 32,000,000 in every layer;
- (d) the projected scoring time is at most 4 h. The projection is 1.5 × 19,200 × Σ over layers of `ms_per_image.p50` / 3.6e6 hours; the factor 1.5 covers the backbone pass and the kNN, which `ms_per_image` leaves out.

If (d) fails, lower `PILOT_IMAGES` to the largest multiple of 5 that fits into 4 h, and ledger it as `Task 8: Ruling: …`. Changing `PILOT_IMAGES` does not touch `run_config.json`. If (a)–(c) fail, stop and report the numbers to the user; do not change `CUT_MARGIN` or `EDGE_CAP` on your own.

- [ ] **Step 4: Bank, z-statistics and scores**

Run in order, waiting on each PID:
- `bash $WORKSPACE/convtu-launch.sh convtu-bank 9`
- `bash $WORKSPACE/convtu-launch.sh convtu-zstats 9`
- `bash $WORKSPACE/convtu-launch.sh convtu-scores 9`

Expected: `convtu/bank.npz`, `convtu/zstats.json`, 200 files in `test_convtu/` plus `fits.json`, and logs without tracebacks.

- [ ] **Step 5: Report**

Run `bash $WORKSPACE/convtu-launch.sh convtu-report 9`; it uses the CPU only.
Expected: `runs/coco-baselines/results_convtu/report.md`, with the separation, harm, differences and depth tables for 13 methods.

- [ ] **Step 6: Write up and commit**

Copy `results_convtu/*` and `convtu/calibration.json` into `docs/results/conv-tu-pilot/`. Write `docs/conv-tu-pilot-results.md` with:
- the clean-image checks from the calibration, per layer: K, the spread of the K-th value (`tau_k` quantiles and CV), the edges read, and the ms per image;
- the separation and harm tables for the four conv-TU rows and the nine baselines, from `report.md`;
- the depth table;
- the differences `convtu_mst − convtu_edges` and `convtu_mst − convtu_acts` with their intervals;
- a verdict against "What counts as a positive pilot" in `docs/conv-tu-pilot-design.md`.

```bash
git add docs/conv-tu-pilot-results.md docs/results/conv-tu-pilot
git commit -m "docs: conv-TU pilot numbers on 200 COCO val images"
```
