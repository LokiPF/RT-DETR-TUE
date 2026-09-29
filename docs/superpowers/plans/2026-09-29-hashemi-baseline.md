# Hashemi et al. Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the Gaussian neuron-interval runtime monitor of Hashemi, Křetínský, Rieder & Schmidt (FM 2023) as the fifth COCO baseline. It monitors the last decoder layer of the frozen RT-DETRv2-R18 (option A), and reports the hybrid encoder's output maps (option B) as a sensitivity row. Its numbers go into the existing COCO report and results doc.

**Architecture:**
- A small module `baselines/hashemi.py` does three things: it fits the per-neuron mean and standard deviation in a single streaming pass over clean COCO train images, stores them, and scores an image as the share of neurons outside μ ± 2σ.
- `DetectorTap` gains an optional hidden-layer capture.
- Two new resumable pipeline phases, `hashemi-fit` and `hashemi-scores`, follow the pattern of `bank` and `discopatch-scores`. The scoring pass regenerates the corrupted versions and checks their digests against the existing detector pass.
- The report treats the two scores like the other optional baseline (DisCoPatch).

**Tech Stack:** Python 3.11, PyTorch, NumPy, pytest; the existing `differential_uncertainty.baselines` package.

**Spec:**
- `docs/driving-benchmark-baselines-and-metrics.md`: the Hashemi row under "Baselines we keep", and "Open items → Hashemi et al. on RT-DETRv2: which layer to monitor", where option A was decided on 29 September 2026.
- `docs/literature-review-image-corruption-detection.md`, Section 12.
- The paper itself: arXiv 2212.07773, Sections 2.2–3.2.

**Before you start:** the spec edits (both docs above) and this plan are still uncommitted in the
main checkout on `fingerprint_bank`. Commit them there first, with the user's go-ahead. Then create
the worktree `.worktrees/hashemi-baseline` on a new branch `hashemi-baseline`, so it contains the
spec.

## Global Constraints

- **Detector:** the frozen RT-DETRv2-R18 COCO checkpoint `/home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth`, with 640 × 640 input.
- **Evaluation protocol is unchanged:** all 5,000 COCO val2017 images in the seed-44 order, 5 folds, 96 conditions (clean first). The existing `runs/coco-baselines/test/*.npz` stay valid, and their digests must match the regenerated corruptions.
- **`run_config.json` must not change.** Do not add anything to `Settings.experiment()`.
- **Fitting data:** all 118,287 clean COCO train2017 images; no labels, no corrupted data.
- **Monitor definition** (the paper's Eqs. 5–6):
  - class information is discarded;
  - k = 2 ("a value close to 2");
  - σ is the population standard deviation;
  - score = share of monitored neurons with |h − μ| > kσ (strict).
- **No conformal p-value.** Our metrics are threshold-free, and the p-value is a decreasing step function of the score, so we use the score directly.
- **Option A, the baseline:** all 300 × 256 neurons of the last decoder layer's output (`model.decoder.decoder.layers[-1]`, index 2), before the class and box heads.
- **Option B, the sensitivity row:** all neurons of the hybrid encoder's three output maps (256 channels at strides 8, 16 and 32, i.e. 80², 40² and 20²), pooled into one share.
- **Tests** run with `CUDA_VISIBLE_DEVICES=` and never touch the GPU. The suite command is:
  `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty -q`
- **Outputs** go to `/home/yuchen/YuchenZ/UE/philip_sa/runs/coco-baselines`, the main checkout's git-ignored `runs/`.
- **The GPU is shared.** Before any GPU step, message the `explore` session (tree-curves, OrthogonalDet) and agree on a gap. Its training holds about 30 GB of the 32 GB card.

## Review Focus

1. **A neuron that was constant on every training image (σ = 0).** A test value equal to its mean stays inside. Any other value counts as outside. Nothing may divide by σ. (Task 1: `test_outside_counts_use_a_strict_k_sigma_band_and_keep_constant_neurons_inside`.)
2. **Activations with a large offset relative to their spread** (for example 10⁶ ± 0.01). The streaming mean and standard deviation must match NumPy's; a naive sum-of-squares would lose all precision. (Task 1: `test_streaming_stats_match_numpy_even_with_a_large_offset`.)
3. **The same image scored alone or in a batch** must get the same score. (Task 1: `test_monitor_scores_each_image_alone_whatever_its_batch`.)
4. **Intervals refitted or edited after scoring started.** A resumed scoring pass must refuse to mix scores from two different fits. (Task 3: `test_hashemi_scores_cover_every_variant_and_stay_tied_to_one_fit`.)
5. **Activations of another shape** (a different input size, or a wrong layer) or only some of the three encoder maps. The monitor must raise a clear error, never broadcast silently. (Task 1: `test_outside_counts_refuse_activations_of_another_shape` and `test_monitor_pools_encoder_neurons_and_needs_all_three_maps`.)

---

### Task 1: The monitor itself (`hashemi.py`)

**Files:**
- Create: `differential_uncertainty/baselines/hashemi.py`
- Test: `tests/differential_uncertainty/test_baselines_hashemi.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces:
  - `K: float = 2.0`
  - `LAYERS = ("decoder", "encoder_s8", "encoder_s16", "encoder_s32")`
  - `class NeuronStats` with `.update(batch: torch.Tensor) -> None`, where the first dimension counts images, and `.result() -> tuple[np.ndarray, np.ndarray]` returning float32 (mean, population std)
  - `outside_counts(values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, k: float = K) -> torch.Tensor` returning per-image counts
  - `save_intervals(path, stats: dict[str, NeuronStats], images: int) -> None`, which writes an npz with keys `images`, `{layer}_mean` and `{layer}_std`
  - `class HashemiMonitor(path, device, k=K)`:
    - `.stats[layer] -> (mean, std)` tensors;
    - `.decoder_share(decoder: Tensor[n,300,256]) -> np.ndarray[n]`;
    - `.scores(decoder, encoder: list[Tensor]) -> tuple[np.ndarray, np.ndarray]`, giving (decoder share, pooled encoder share).

- [ ] **Step 1: Write the failing tests**

Create `tests/differential_uncertainty/test_baselines_hashemi.py`:

```python
import numpy as np
import pytest
import torch

from differential_uncertainty.baselines import hashemi


def test_streaming_stats_match_numpy_even_with_a_large_offset():
    rng = np.random.default_rng(0)
    data = 1e6 + rng.normal(0, 1e-2, size=(257, 3, 4))
    stats = hashemi.NeuronStats()
    for start in range(0, 257, 50):
        stats.update(torch.from_numpy(data[start:start + 50]))
    mean, std = stats.result()
    assert mean.dtype == np.float32 and std.dtype == np.float32
    assert np.allclose(mean, data.mean(axis=0), rtol=1e-6, atol=0)
    assert np.allclose(std, data.std(axis=0), rtol=1e-4, atol=0)


def test_stats_need_two_images_and_a_fixed_neuron_shape():
    stats = hashemi.NeuronStats()
    stats.update(torch.zeros(1, 2))
    with pytest.raises(ValueError, match="two images"):
        stats.result()
    with pytest.raises(ValueError, match="shape"):
        stats.update(torch.zeros(1, 3))


def test_outside_counts_use_a_strict_k_sigma_band_and_keep_constant_neurons_inside():
    mean = torch.tensor([0.0, 5.0, 1.0])
    std = torch.tensor([1.0, 0.0, 0.5])
    values = torch.tensor([[2.0, 5.0, 1.0],     # on the band edge, at the constant, at the mean: all inside
                           [2.1, 5.001, 2.1]])  # beyond the edge, off the constant, beyond the edge
    assert hashemi.outside_counts(values, mean, std, k=2.0).tolist() == [0, 3]


def test_outside_counts_refuse_activations_of_another_shape():
    with pytest.raises(ValueError, match="do not match"):
        hashemi.outside_counts(torch.zeros(2, 4), torch.zeros(3), torch.ones(3))


def _fitted(tmp_path, rng):
    shapes = {"decoder": (300, 256), "encoder_s8": (4, 4, 4), "encoder_s16": (4, 2, 2), "encoder_s32": (4, 1, 1)}
    stats = {name: hashemi.NeuronStats() for name in hashemi.LAYERS}
    for name, shape in shapes.items():
        stats[name].update(torch.from_numpy(rng.normal(size=(50, *shape))))
    path = tmp_path / "intervals.npz"
    hashemi.save_intervals(path, stats, images=50)
    return path, shapes


def test_monitor_scores_each_image_alone_whatever_its_batch(tmp_path):
    rng = np.random.default_rng(1)
    path, shapes = _fitted(tmp_path, rng)
    monitor = hashemi.HashemiMonitor(path, "cpu")
    decoder = torch.from_numpy(rng.normal(size=(3, *shapes["decoder"]))).float()
    decoder[2] += 10.0                                    # far outside every interval
    encoder = [torch.from_numpy(rng.normal(size=(3, *shapes[n]))).float() for n in hashemi.LAYERS[1:]]
    both_dec, both_enc = monitor.scores(decoder, encoder)
    alone_dec, alone_enc = monitor.scores(decoder[:1], [e[:1] for e in encoder])
    assert both_dec[0] == pytest.approx(alone_dec[0]) and both_enc[0] == pytest.approx(alone_enc[0])
    assert both_dec[2] == pytest.approx(1.0) and 0.0 < both_dec[0] < 0.2
    assert np.all((both_enc >= 0) & (both_enc <= 1))
    assert monitor.decoder_share(decoder[:1])[0] == pytest.approx(alone_dec[0])


def test_monitor_pools_encoder_neurons_and_needs_all_three_maps(tmp_path):
    rng = np.random.default_rng(2)
    path, shapes = _fitted(tmp_path, rng)
    monitor = hashemi.HashemiMonitor(path, "cpu")
    decoder = torch.zeros(1, *shapes["decoder"])
    encoder = [torch.zeros(1, *shapes[n]) for n in hashemi.LAYERS[1:]]
    encoder[0] += 100.0                                   # only the stride-8 map (64 of 84 neurons) is off
    _, share = monitor.scores(decoder, encoder)
    assert share[0] == pytest.approx(64 / 84)
    with pytest.raises(ValueError, match="three"):
        monitor.scores(decoder, encoder[:2])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_baselines_hashemi.py -q`
Expected: 6 errors or failures, with `ImportError: cannot import name 'hashemi'` or `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

Create `differential_uncertainty/baselines/hashemi.py`:

```python
"""Gaussian neuron-interval runtime monitor (Hashemi, Křetínský, Rieder & Schmidt, FM 2023).

Every monitored neuron gets the interval mu +- k*sigma from clean training images, with class
information discarded (their Sec. 3.1). An image's score is the share of monitored neurons outside
their interval (their Eq. 6). Their conformal p-value is a decreasing step function of this score,
so threshold-free metrics use the score directly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

K = 2.0  # "k is a value close to 2" (their Sec. 2.2)
LAYERS = ("decoder", "encoder_s8", "encoder_s16", "encoder_s32")


class NeuronStats:
    """Exact streaming mean and population standard deviation per neuron (Chan et al.), in float64."""

    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None

    def update(self, batch: torch.Tensor) -> None:
        batch = batch.detach().to(torch.float64)
        n = batch.shape[0]
        if n == 0:
            return
        if self.mean is not None and tuple(batch.shape[1:]) != tuple(self.mean.shape):
            raise ValueError(f"neuron shape changed from {tuple(self.mean.shape)} to {tuple(batch.shape[1:])}")
        batch_mean = batch.mean(dim=0)
        batch_m2 = ((batch - batch_mean) ** 2).sum(dim=0)
        if self.mean is None:
            self.count, self.mean, self.m2 = n, batch_mean, batch_m2
            return
        total = self.count + n
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (n / total)
        self.m2 = self.m2 + batch_m2 + delta ** 2 * (self.count * n / total)
        self.count = total

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count < 2:
            raise ValueError("need at least two images to estimate a standard deviation")
        return self.mean.float().cpu().numpy(), (self.m2 / self.count).sqrt().float().cpu().numpy()


def outside_counts(values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, k: float = K) -> torch.Tensor:
    """Neurons per image with |h - mu| > k*sigma; strict, so a constant neuron at its mean stays inside."""
    if tuple(values.shape[1:]) != tuple(mean.shape):
        raise ValueError(f"activations {tuple(values.shape[1:])} do not match the fitted neurons {tuple(mean.shape)}")
    outside = (values.float() - mean).abs() > k * std
    return outside.flatten(1).sum(dim=1)


def save_intervals(path, stats: dict, images: int) -> None:
    arrays = {}
    for name in LAYERS:
        arrays[f"{name}_mean"], arrays[f"{name}_std"] = stats[name].result()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, images=np.array(images), **arrays)
    temporary.replace(path)


class HashemiMonitor:
    def __init__(self, path, device, k: float = K):
        with np.load(path, allow_pickle=False) as data:
            self.stats = {name: (torch.from_numpy(data[f"{name}_mean"]).to(device),
                                 torch.from_numpy(data[f"{name}_std"]).to(device)) for name in LAYERS}
        self.k = k

    def decoder_share(self, decoder: torch.Tensor) -> np.ndarray:
        mean, std = self.stats["decoder"]
        return (outside_counts(decoder, mean, std, self.k).double() / mean.numel()).cpu().numpy()

    def scores(self, decoder: torch.Tensor, encoder) -> tuple[np.ndarray, np.ndarray]:
        """(decoder share, encoder share) per image; the encoder share pools all three maps' neurons."""
        if len(encoder) != len(LAYERS) - 1:
            raise ValueError(f"expected the encoder's three output maps, got {len(encoder)}")
        counts, total = None, 0
        for name, values in zip(LAYERS[1:], encoder):
            mean, std = self.stats[name]
            part = outside_counts(values, mean, std, self.k).double()
            counts = part if counts is None else counts + part
            total += mean.numel()
        return self.decoder_share(decoder), (counts / total).cpu().numpy()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_baselines_hashemi.py -q`
Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add differential_uncertainty/baselines/hashemi.py tests/differential_uncertainty/test_baselines_hashemi.py
git commit -m "feat: add the Hashemi et al. neuron-interval monitor"
```

---

### Task 2: Hidden-layer capture in `DetectorTap`

**Files:**
- Modify: `differential_uncertainty/baselines/detector.py` (the whole file is shown below)
- Test: `tests/differential_uncertainty/test_baselines_detector.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `DetectorTap(checkpoint_path, device, image_size=(640, 640), hidden=False)`. With `hidden=True`, `.forward_hidden(batch) -> (decoder: Tensor[n, 300, 256], encoder: list[Tensor])`, both float32 on the tap's device.
  - `.prepare`, `.forward` and `.run` are unchanged.
  - `DECODER_SHAPE = (300, 256)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/differential_uncertainty/test_baselines_detector.py`:

```python
class FakeHiddenDetector(nn.Module):
    """Last decoder layer output = 3 * level + 3, first encoder map = level (level = mean image value)."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Identity()
        self.encoder = nn.Identity()
        self.decoder = nn.Module()
        self.decoder.decoder = nn.Module()
        self.decoder.decoder.layers = nn.ModuleList([nn.Identity() for _ in range(3)])

    def forward(self, images):
        n = images.shape[0]
        level = images.mean(dim=(1, 2, 3))
        self.backbone([torch.zeros(n, 256, 4, 4), torch.ones(n, 512, 2, 2)])
        self.encoder([level.view(n, 1, 1, 1).expand(n, 256, 4, 4).clone(),
                      torch.zeros(n, 256, 2, 2), torch.zeros(n, 256, 1, 1)])
        queries = torch.zeros(n, 300, 256)
        for index, layer in enumerate(self.decoder.decoder.layers):
            queries = layer(queries + index + level.view(n, 1, 1))
        return {"pred_logits": torch.zeros(n, 300, 80), "pred_boxes": torch.full((n, 300, 4), 0.5)}


def test_hidden_tap_returns_last_decoder_layer_and_encoder_maps(monkeypatch):
    monkeypatch.setattr(detector, "load_frozen_detector", lambda _p, _d: FakeHiddenDetector())
    arrays = [np.full((20, 30, 3), 255, np.uint8), np.zeros((10, 10, 3), np.uint8)]
    with detector.DetectorTap("unused.pth", "cpu", image_size=(16, 16), hidden=True) as tap:
        decoder, encoder = tap.forward_hidden(tap.prepare(arrays))
        logits, _, _ = tap.run(arrays, batch_size=2)   # the normal path still works next to the extra hooks
    assert tuple(decoder.shape) == (2, 300, 256)
    assert [tuple(e.shape) for e in encoder] == [(2, 256, 4, 4), (2, 256, 2, 2), (2, 256, 1, 1)]
    assert decoder[0, 0, 0].item() == pytest.approx(6.0) and decoder[1, 0, 0].item() == pytest.approx(3.0)
    assert encoder[0][0, 0, 0, 0].item() == pytest.approx(1.0)
    assert logits.shape == (2, 300, 80)


def test_hidden_layers_need_a_hidden_tap(monkeypatch):
    monkeypatch.setattr(detector, "load_frozen_detector", lambda _p, _d: FakeHiddenDetector())
    with detector.DetectorTap("unused.pth", "cpu", image_size=(8, 8)) as tap:
        with pytest.raises(RuntimeError, match="hidden=True"):
            tap.forward_hidden(tap.prepare([np.zeros((8, 8, 3), np.uint8)]))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_baselines_detector.py -q`
Expected: 2 failed (`TypeError: ... unexpected keyword argument 'hidden'` and `AttributeError: 'DetectorTap' object has no attribute 'forward_hidden'`), 2 passed.

- [ ] **Step 3: Write the implementation**

Replace `differential_uncertainty/baselines/detector.py` with:

```python
"""Frozen RT-DETRv2 forward pass exposing logits, boxes, the pooled last backbone stage and, on request,
the last decoder layer's queries and the hybrid encoder's output maps."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from ..extraction import load_frozen_detector, prepare_image

QUERY_COUNT = 300
CLASS_COUNT = 80
POOLED_DIM = 512
DECODER_SHAPE = (QUERY_COUNT, 256)


class DetectorTap:
    def __init__(self, checkpoint_path, device, image_size=(640, 640), hidden=False):
        self.device = torch.device(device)
        self.image_size = tuple(image_size)
        self.model = load_frozen_detector(checkpoint_path, self.device)
        self._pooled = self._decoder = self._encoder = None
        self._handles = [self.model.backbone.register_forward_hook(self._capture)]
        self.hidden = hidden
        if hidden:
            self._handles.append(self.model.decoder.decoder.layers[-1].register_forward_hook(self._capture_decoder))
            self._handles.append(self.model.encoder.register_forward_hook(self._capture_encoder))

    def _capture(self, _module, _inputs, outputs):
        self._pooled = outputs[-1].mean(dim=(2, 3))

    def _capture_decoder(self, _module, _inputs, output):
        self._decoder = output

    def _capture_encoder(self, _module, _inputs, outputs):
        self._encoder = list(outputs)

    def prepare(self, arrays) -> torch.Tensor:
        return torch.stack([prepare_image(Image.fromarray(a), self.image_size) for a in arrays])

    @torch.inference_mode()
    def forward(self, batch: torch.Tensor):
        outputs = self.model(batch.to(self.device, non_blocking=True))
        pooled, self._pooled = self._pooled, None
        self._decoder = self._encoder = None
        if pooled is None:
            raise RuntimeError("the backbone hook did not run")
        logits = outputs["pred_logits"].float().cpu().numpy()
        boxes = outputs["pred_boxes"].float().cpu().numpy()
        pooled = pooled.float().cpu().numpy()
        n = batch.shape[0]
        if (logits.shape != (n, QUERY_COUNT, CLASS_COUNT) or boxes.shape != (n, QUERY_COUNT, 4)
                or pooled.shape != (n, POOLED_DIM)):
            raise ValueError("detector outputs do not have the expected shapes")
        if not (np.isfinite(logits).all() and np.isfinite(boxes).all() and np.isfinite(pooled).all()):
            raise ValueError("detector outputs must be finite")
        return logits, boxes, pooled

    @torch.inference_mode()
    def forward_hidden(self, batch: torch.Tensor):
        """Last decoder layer's queries (n, 300, 256) and the encoder's output maps, float32 on the device."""
        if not self.hidden:
            raise RuntimeError("create the tap with hidden=True to read hidden layers")
        self.model(batch.to(self.device, non_blocking=True))
        decoder, encoder = self._decoder, self._encoder
        self._pooled = self._decoder = self._encoder = None
        if decoder is None or encoder is None:
            raise RuntimeError("the decoder or encoder hook did not run")
        n = batch.shape[0]
        if tuple(decoder.shape) != (n, *DECODER_SHAPE):
            raise ValueError(f"decoder queries have shape {tuple(decoder.shape)}, expected {(n, *DECODER_SHAPE)}")
        decoder, encoder = decoder.float(), [e.float() for e in encoder]
        if not (bool(torch.isfinite(decoder).all()) and all(bool(torch.isfinite(e).all()) for e in encoder)):
            raise ValueError("hidden activations must be finite")
        return decoder, encoder

    def run(self, arrays, batch_size: int = 32):
        parts = [self.forward(self.prepare(arrays[s:s + batch_size]))
                 for s in range(0, len(arrays), batch_size)]
        return tuple(np.concatenate(values) for values in zip(*parts))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self.model = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_baselines_detector.py -q`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add differential_uncertainty/baselines/detector.py tests/differential_uncertainty/test_baselines_detector.py
git commit -m "feat: let the detector tap read the last decoder layer and encoder maps"
```

---

### Task 3: Pipeline phases `hashemi-fit` and `hashemi-scores`, timing and CLI

**Files:**
- Modify: `differential_uncertainty/baselines/pipeline.py` (imports, new helpers and phases, `phase_timing`, `PHASES`)
- Modify: `differential_uncertainty/cli.py` (the `--phase` choices)
- Test: `tests/differential_uncertainty/test_baselines_pipeline.py`, `tests/differential_uncertainty/test_cli.py`

**Interfaces:**
- Consumes:
  - Task 1: `K`, `LAYERS`, `NeuronStats`, `save_intervals`, `HashemiMonitor(.scores, .decoder_share)`.
  - Task 2: `DetectorTap(..., hidden=True).forward_hidden(batch)`.
- Produces:
  - `runs/.../hashemi/intervals.npz` and `hashemi/fit.json` (`images`, `k`, `std`, `layers`).
  - `runs/.../test_hashemi/{stem}.npz`, holding `hashemi_decoder` (96,) and `hashemi_encoder` (96,).
  - `test_hashemi/intervals.json`, with the sha1 of `intervals.npz`.
  - `pipeline.HASHEMI_KEYS = ("hashemi_decoder", "hashemi_encoder")`.
  - `timing.json` gains `detector_plus_hashemi_ms`.
  - CLI phases `hashemi-fit` and `hashemi-scores`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/differential_uncertainty/test_baselines_pipeline.py`:

```python
class HiddenFakeTap(FakeTap):
    """FakeTap plus the hidden-layer interface; activations shift with image brightness."""
    hidden_calls = 0

    def prepare(self, arrays):
        return torch.stack([torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 255.0
                            for a in arrays])

    def forward(self, batch):
        return self.run([(b.permute(1, 2, 0).numpy() * 255).astype(np.uint8) for b in batch])

    def forward_hidden(self, batch):
        HiddenFakeTap.hidden_calls += 1
        n = batch.shape[0]
        level = batch.float().mean(dim=(1, 2, 3))
        generator = torch.Generator().manual_seed(0)
        decoder = torch.randn(300, 256, generator=generator) + 4.0 * level.view(n, 1, 1)
        encoder = [level.view(n, 1, 1, 1).expand(n, 8, s, s).clone() + torch.randn(8, s, s, generator=generator)
                   for s in (4, 2, 1)]
        return decoder, encoder


def _train_images(tmp_path, count=3):
    folder = tmp_path / "train"
    folder.mkdir(exist_ok=True)
    rng = np.random.default_rng(3)
    for index in range(count):
        Image.fromarray(rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)).save(folder / f"{index:04d}.jpg")
    return folder


@pytest.fixture
def hidden_fakes(monkeypatch):
    HiddenFakeTap.calls = HiddenFakeTap.hidden_calls = 0
    monkeypatch.setattr(pipeline, "DetectorTap", HiddenFakeTap)
    monkeypatch.setattr(pipeline, "_load_bank", lambda _s, _d: torch.nn.functional.normalize(torch.randn(256, 512), dim=1))


def test_hashemi_fit_stores_intervals_for_every_monitored_layer_once(tmp_path, hidden_fakes):
    settings = _settings(tmp_path, train_images=_train_images(tmp_path))
    pipeline.run_phase("hashemi-fit", settings)
    with np.load(settings.output / "hashemi" / "intervals.npz") as data:
        assert data["decoder_mean"].shape == (300, 256) and data["encoder_s8_std"].shape == (8, 4, 4)
        assert data["encoder_s32_mean"].shape == (8, 1, 1) and int(data["images"]) == 3
    fit = json.loads((settings.output / "hashemi" / "fit.json").read_text())
    assert fit["k"] == 2.0 and fit["images"] == 3
    calls = HiddenFakeTap.hidden_calls
    pipeline.run_phase("hashemi-fit", settings)
    assert HiddenFakeTap.hidden_calls == calls


def test_hashemi_scores_cover_every_variant_and_stay_tied_to_one_fit(tmp_path, hidden_fakes):
    settings = _settings(tmp_path, train_images=_train_images(tmp_path))
    pipeline.run_phase("test", settings)
    with pytest.raises(ValueError, match="hashemi-fit"):
        pipeline.run_phase("hashemi-scores", settings)
    pipeline.run_phase("hashemi-fit", settings)
    pipeline.run_phase("hashemi-scores", settings)
    files = sorted((settings.output / "test_hashemi").glob("*.npz"))
    assert len(files) == 2
    with np.load(files[0]) as data:
        assert data["hashemi_decoder"].shape == (96,) and data["hashemi_encoder"].shape == (96,)
        assert np.all((data["hashemi_decoder"] >= 0) & (data["hashemi_decoder"] <= 1))
    intervals = settings.output / "hashemi" / "intervals.npz"
    intervals.write_bytes(intervals.read_bytes() + b"x")
    with pytest.raises(ValueError, match="intervals changed"):
        pipeline.run_phase("hashemi-scores", settings)


def test_hashemi_scores_detect_changed_corruptions(tmp_path, hidden_fakes):
    settings = _settings(tmp_path, train_images=_train_images(tmp_path))
    pipeline.run_phase("test", settings)
    pipeline.run_phase("hashemi-fit", settings)
    victim = sorted((settings.output / "test").glob("*.npz"))[0]
    data = dict(np.load(victim))
    data["digests"] = np.array(["0" * 16] * 96)
    np.savez(victim, **data)
    with pytest.raises(ValueError, match="corruptions differ"):
        pipeline.run_phase("hashemi-scores", settings)


def test_timing_reports_the_hashemi_monitor_once_it_is_fitted(tmp_path, hidden_fakes):
    settings = _settings(tmp_path, train_images=_train_images(tmp_path))
    pipeline.run_phase("hashemi-fit", settings)
    pipeline.run_phase("timing", settings)
    timing = json.loads((settings.output / "timing.json").read_text())
    assert {"detector_ms", "detector_plus_hashemi_ms"} <= set(timing) and "discopatch_ms" not in timing
```

Append to `tests/differential_uncertainty/test_cli.py`:

```python
def test_cli_accepts_the_hashemi_phases(monkeypatch, tmp_path):
    import differential_uncertainty.baselines.pipeline as pipeline
    seen = []
    monkeypatch.setattr(pipeline, "run_phase", lambda phase, settings: seen.append(phase))
    base = ["--output", str(tmp_path / "out"), "--checkpoint", "c.pth", "--coco-train-images", "train",
            "--coco-val-images", "val", "--coco-annotations", "ann.json", "--discopatch-root", "dcp"]
    for phase in ("hashemi-fit", "hashemi-scores"):
        assert cli.main(["baselines-coco", "--phase", phase, *base]) == 0
    assert seen == ["hashemi-fit", "hashemi-scores"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_baselines_pipeline.py tests/differential_uncertainty/test_cli.py -q`
Expected:
- The four new pipeline tests fail with `ValueError: unknown phase 'hashemi-fit'`. The timing test fails with a missing `detector_plus_hashemi_ms`, or with the unknown-phase error first.
- The CLI test fails with `SystemExit: 2` (invalid choice).
- All older tests pass.

- [ ] **Step 3: Write the implementation**

In `differential_uncertainty/baselines/pipeline.py`:

1. Change the module docstring's first line to `"""Resumable phases that compute the baseline scores on the fixed COCO protocol."""`.
2. Below `from .discopatch import DisCoPatchScorer, train_discopatch`, add:

```python
from .hashemi import K as HASHEMI_K
from .hashemi import LAYERS as HASHEMI_LAYERS
from .hashemi import HashemiMonitor, NeuronStats, save_intervals
```

3. Below the `TEST_KEYS = (...)` definition, add:

```python
HASHEMI_KEYS = ("hashemi_decoder", "hashemi_encoder")
```

4. Directly above `def phase_timing`, add:

```python
def _hashemi_intervals(settings: Settings) -> Path:
    return settings.output / "hashemi" / "intervals.npz"


def phase_hashemi_fit(settings: Settings) -> None:
    """Per-neuron mean and standard deviation over every clean COCO train image (Hashemi et al., Sec. 3.1)."""
    path = _hashemi_intervals(settings)
    if path.exists():
        return
    paths = protocol.list_images(settings.train_images)
    loader = DataLoader(_Prepared(paths), batch_size=settings.batch_size,
                        num_workers=settings.workers, pin_memory=True)
    stats = {name: NeuronStats() for name in HASHEMI_LAYERS}
    started = time.time()
    with DetectorTap(settings.checkpoint, settings.device, hidden=True) as tap:
        for index, batch in enumerate(loader):
            decoder, encoder = tap.forward_hidden(batch)
            if len(encoder) != len(HASHEMI_LAYERS) - 1:
                raise ValueError(f"expected the encoder's three output maps, got {len(encoder)}")
            stats["decoder"].update(decoder)
            for name, values in zip(HASHEMI_LAYERS[1:], encoder):
                stats[name].update(values)
            if index % 200 == 0:
                _progress("hashemi-fit", min((index + 1) * settings.batch_size, len(paths)), len(paths), started)
    save_intervals(path, stats, images=len(paths))
    _atomic_json(path.with_name("fit.json"), {"images": len(paths), "k": HASHEMI_K,
                                              "std": "population (ddof=0)", "layers": list(HASHEMI_LAYERS)})


def _hashemi_scores(tap, monitor, arrays, batch_size) -> dict:
    decoder_parts, encoder_parts = [], []
    for start in range(0, len(arrays), batch_size):
        decoder, encoder = tap.forward_hidden(tap.prepare(arrays[start:start + batch_size]))
        decoder_share, encoder_share = monitor.scores(decoder, encoder)
        decoder_parts.append(decoder_share)
        encoder_parts.append(encoder_share)
    return {"hashemi_decoder": np.concatenate(decoder_parts), "hashemi_encoder": np.concatenate(encoder_parts)}


def phase_hashemi_scores(settings: Settings) -> None:
    intervals = _hashemi_intervals(settings)
    if not intervals.exists():
        raise ValueError(f"run the hashemi-fit phase first: {intervals} is missing")
    folder = settings.output / "test_hashemi"
    record = {"intervals": str(intervals), "sha1": hashlib.sha1(intervals.read_bytes()).hexdigest(), "k": HASHEMI_K}
    marker = folder / "intervals.json"
    if marker.exists():
        if json.loads(marker.read_text()) != record:
            raise ValueError(f"the fitted intervals changed since {marker} was written")
    else:
        _atomic_json(marker, record)
    pending = [p for p in evaluation(settings) if not _valid_existing(folder / f"{p.stem}.npz", HASHEMI_KEYS)]
    missing = [p.name for p in pending if not (settings.output / "test" / f"{p.stem}.npz").exists()]
    if missing:
        raise ValueError(f"run the test phase first: {len(missing)} detector results are missing, e.g. {missing[0]}")
    if not pending:
        return
    monitor = HashemiMonitor(intervals, settings.device)
    started = time.time()
    with DetectorTap(settings.checkpoint, settings.device, hidden=True) as tap:
        for done, (name, arrays) in enumerate(_variant_stream(settings, pending), start=1):
            stem = Path(name).stem
            stored = _load_npz(settings.output / "test" / f"{stem}.npz", TEST_KEYS)["digests"]
            if list(stored) != [protocol.digest(a) for a in arrays]:
                raise ValueError(f"corruptions differ from the detector pass for {name}")
            values = _hashemi_scores(tap, monitor, arrays, settings.batch_size)
            if not all(np.isfinite(v).all() for v in values.values()):
                raise ValueError(f"the Hashemi monitor returned non-finite scores for {name}")
            _atomic_npz(folder / f"{stem}.npz", **values)
            if done % 25 == 0:
                _progress("hashemi", done, len(pending), started)
```

5. In `phase_timing`, directly before `_atomic_json(settings.output / "timing.json", result)`, add:

```python
    intervals = _hashemi_intervals(settings)
    if intervals.exists():
        monitor = HashemiMonitor(intervals, settings.device)
        # a separate tap, so the hidden-layer hooks never touch the other timings
        with DetectorTap(settings.checkpoint, settings.device, hidden=True) as hidden_tap:
            def hashemi_score(array):
                decoder, _ = hidden_tap.forward_hidden(hidden_tap.prepare([array]))
                monitor.decoder_share(decoder)

            result["detector_plus_hashemi_ms"] = timed(hashemi_score)
```

6. Replace the `PHASES` dict with:

```python
PHASES = {
    "sanity": phase_sanity, "bank": phase_bank, "test": phase_test,
    "train-discopatch": phase_train_discopatch, "discopatch-scores": phase_discopatch_scores,
    "hashemi-fit": phase_hashemi_fit, "hashemi-scores": phase_hashemi_scores,
    "timing": phase_timing, "report": phase_report,
}
```

In `differential_uncertainty/cli.py`, replace the `--phase` choices line with:

```python
                           choices=["sanity", "bank", "test", "train-discopatch", "discopatch-scores",
                                    "hashemi-fit", "hashemi-scores", "timing", "report"])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_baselines_pipeline.py tests/differential_uncertainty/test_cli.py -q`
Expected: all pass (the 4 new pipeline tests, the new CLI test, and every older test).

- [ ] **Step 5: Commit**

```bash
git add differential_uncertainty/baselines/pipeline.py differential_uncertainty/cli.py tests/differential_uncertainty/test_baselines_pipeline.py tests/differential_uncertainty/test_cli.py
git commit -m "feat: add hashemi-fit and hashemi-scores phases and their timing"
```

---

### Task 4: Hashemi scores in the report

**Files:**
- Modify: `differential_uncertainty/baselines/report.py` (imports, `METHODS`, `LABELS`, `method_scores`, `build_report`)
- Test: `tests/differential_uncertainty/test_baselines_report.py`

**Interfaces:**
- Consumes:
  - Task 1: `hashemi.K`.
  - Task 3: the `test_hashemi/{stem}.npz` files with `hashemi_decoder` and `hashemi_encoder`.
- Produces:
  - The methods `hashemi` (option A) and `hashemi_enc` (option B) in every table.
  - `summary.json` keys `hashemi_included` and `hashemi_k`.
  - `method_scores(test, dcp, per_image_lambda, k=KNN_K, hashemi=None)`.

- [ ] **Step 1: Write the failing tests**

In `tests/differential_uncertainty/test_baselines_report.py`:

1. Add `import csv` at the top, next to `import json`.
2. Replace the whole `test_build_report_end_to_end_on_a_tiny_fixture` function with the helper and two tests below. The helper's body is the old test's setup, unchanged.

```python
def _tiny_run(tmp_path, monkeypatch):
    """15 fixture images through the test phase with a fake detector; returns the settings."""
    import torch
    from PIL import Image
    from differential_uncertainty.baselines import pipeline

    val = tmp_path / "val"
    val.mkdir()
    rng = np.random.default_rng(7)
    images, annotations = [], []
    for index in range(15):  # enough that every bootstrap fold complement keeps >= 3 images
        name = f"{index:012d}.jpg"
        Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)).save(val / name)
        images.append({"id": index + 1, "file_name": name, "width": 64, "height": 48})
        annotations.append({"id": 100 + index, "image_id": index + 1, "category_id": 1,
                            "bbox": [8, 6, 24, 18], "area": 432, "iscrowd": 0})
    (tmp_path / "ann.json").write_text(json.dumps({
        "images": images, "annotations": annotations,
        "categories": [{"id": c, "name": f"c{c}"} for c in range(1, 81)]}))

    class FakeTap:
        def __init__(self, *_a, **_k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def run(self, arrays, batch_size=32):
            n = len(arrays)
            quality = np.array([1.0 - a.std() / 128.0 for a in arrays])
            logits = np.full((n, 300, 80), -6.0)
            logits[:, 0, 0] = 6.0 * quality - 2.0
            boxes = np.tile([0.3125, 0.3125, 0.375, 0.375], (n, 300, 1))
            pooled = np.random.default_rng(0).normal(size=(n, 512)) + quality[:, None]
            return logits, boxes, pooled

    monkeypatch.setattr(pipeline, "DetectorTap", FakeTap)
    monkeypatch.setattr(pipeline, "_load_bank", lambda _s, _d: torch.nn.functional.normalize(torch.randn(256, 512), dim=1))
    monkeypatch.setattr(report, "BOOTSTRAP_SAMPLES", 5)
    settings = pipeline.Settings(output=tmp_path / "out", checkpoint=tmp_path / "c.pth", train_images=tmp_path,
                                 val_images=val, annotations=tmp_path / "ann.json", discopatch_root=tmp_path,
                                 limit=15, workers=0, device="cpu")
    pipeline.run_phase("test", settings)
    return settings


def test_build_report_end_to_end_on_a_tiny_fixture(tmp_path, monkeypatch):
    from differential_uncertainty.baselines import pipeline
    settings = _tiny_run(tmp_path, monkeypatch)
    pipeline.run_phase("report", settings)

    results = settings.output / "results"
    summary = json.loads((results / "summary.json").read_text())
    assert summary["images"] == 15 and len(summary["lambda_per_fold"]) == 5
    assert summary["discopatch_included"] is False and summary["hashemi_included"] is False
    for name in ("separation", "aggregates", "harm", "aurc_pools", "conditions", "intervals", "differences", "knn_k"):
        assert (results / f"{name}.csv").exists(), name
    assert "ContrastiveConf" in (results / "report.md").read_text()
    header = (results / "conditions.csv").read_text().splitlines()[0].split(",")
    assert "images_undefined_lrp" in header


def test_build_report_includes_both_hashemi_variants_when_scored(tmp_path, monkeypatch):
    from differential_uncertainty.baselines import pipeline
    settings = _tiny_run(tmp_path, monkeypatch)
    folder = settings.output / "test_hashemi"
    folder.mkdir()
    rng = np.random.default_rng(8)
    for path in sorted((settings.output / "test").glob("*.npz")):
        np.savez(folder / path.name, hashemi_decoder=rng.uniform(0, 1, 96), hashemi_encoder=rng.uniform(0, 1, 96))
    pipeline.run_phase("report", settings)

    results = settings.output / "results"
    summary = json.loads((results / "summary.json").read_text())
    assert summary["hashemi_included"] is True and summary["hashemi_k"] == 2.0
    with (results / "harm.csv").open() as handle:
        methods = {row["method"] for row in csv.DictReader(handle)}
    assert {"hashemi", "hashemi_enc"} <= methods
    assert "Hashemi et al., decoder queries" in (results / "report.md").read_text()


def test_method_scores_add_both_hashemi_variants_only_when_given():
    test = {"saod_top3": np.zeros((2, 96)), "saod_min": np.zeros((2, 96)), "conf_pos": np.ones((2, 96)),
            "conf_neg": np.zeros((2, 96)), "knn": np.zeros((2, 96, 200))}
    hashemi = {"hashemi_decoder": np.full((2, 96), 0.1), "hashemi_encoder": np.full((2, 96), 0.2)}
    scores = report.method_scores(test, None, np.ones(2), hashemi=hashemi)
    assert scores["hashemi"][0, 0] == 0.1 and scores["hashemi_enc"][0, 0] == 0.2
    assert "hashemi" not in report.method_scores(test, None, np.ones(2))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_baselines_report.py -q`
Expected: 3 failed:
- the end-to-end test, with `KeyError: 'hashemi_included'`;
- the Hashemi report test, with `KeyError: 'hashemi_included'` or a missing `hashemi` method;
- `test_method_scores_...`, with `TypeError: ... unexpected keyword argument 'hashemi'`.

The other report tests pass.

- [ ] **Step 3: Write the implementation**

In `differential_uncertainty/baselines/report.py`:

1. Next to the other intra-package imports at the top, add `from .hashemi import K as HASHEMI_K`.
2. Replace the `METHODS` and `LABELS` definitions with:

```python
METHODS = ("saod_top3", "saod_min", "contrastive", "knn", "discopatch", "hashemi", "hashemi_enc")
LABELS = {"saod_top3": "SAOD, mean of top 3", "saod_min": "SAOD, min (1 − max confidence)",
          "contrastive": "ContrastiveConf", "knn": "kNN (k = 100)", "discopatch": "DisCoPatch",
          "hashemi": "Hashemi et al., decoder queries",
          "hashemi_enc": "Hashemi et al., encoder maps (sensitivity)"}
HASHEMI_ARRAYS = ("hashemi_decoder", "hashemi_encoder")
```

3. Replace `method_scores` with:

```python
def method_scores(test: dict, dcp, per_image_lambda, k: int = KNN_K, hashemi=None) -> dict:
    scores = {"saod_top3": test["saod_top3"], "saod_min": test["saod_min"],
              "contrastive": -(test["conf_pos"] - np.asarray(per_image_lambda)[:, None] * test["conf_neg"]),
              "knn": test["knn"][:, :, k - 1]}
    if dcp is not None:
        scores["discopatch"] = dcp
    if hashemi is not None:
        scores["hashemi"] = hashemi["hashemi_decoder"]
        scores["hashemi_enc"] = hashemi["hashemi_encoder"]
    return scores
```

4. In `build_report`, replace

```python
    dcp = _stack(dcp_folder, names, ("dcp",))["dcp"] if dcp_folder.exists() else None
```

with

```python
    dcp = _stack(dcp_folder, names, ("dcp",))["dcp"] if dcp_folder.exists() else None
    hashemi_folder = settings.output / "test_hashemi"
    hashemi = _stack(hashemi_folder, names, HASHEMI_ARRAYS) if hashemi_folder.exists() else None
```

and replace

```python
    ordered = {m: v for m, v in method_scores(test, dcp, per_image_lambda).items()}
```

with

```python
    ordered = {m: v for m, v in method_scores(test, dcp, per_image_lambda, hashemi=hashemi).items()}
```

5. In the `summary = {...}` dict of `build_report`, replace the line
`"bootstrap_samples": BOOTSTRAP_SAMPLES, "discopatch_included": dcp is not None,` with

```python
        "bootstrap_samples": BOOTSTRAP_SAMPLES, "discopatch_included": dcp is not None,
        "hashemi_included": hashemi is not None, "hashemi_k": HASHEMI_K,
```

- [ ] **Step 4: Run the tests, then the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_baselines_report.py -q`
Expected: all pass.

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty -q`
Expected: all pass. The previous 203 plus the 15 added in Tasks 1–4 make 218 passed, with 2 CUDA tests skipped.

- [ ] **Step 5: Commit**

```bash
git add differential_uncertainty/baselines/report.py tests/differential_uncertainty/test_baselines_report.py
git commit -m "feat: report both Hashemi et al. variants"
```

---

### Task 5: Run on the GPU and publish the numbers

**Files:**
- Create (outputs): `runs/coco-baselines/hashemi/*`, `runs/coco-baselines/test_hashemi/*` (git-ignored)
- Modify: `docs/results/coco-baselines/*` (copied from `runs/coco-baselines/results/`), `docs/coco-baseline-numbers.md`
- Create: `docs/results/coco-baselines/hashemi_fit.json`, `docs/results/coco-baselines/hashemi_intervals.json`

**Interfaces:**
- Consumes: the CLI phases from Task 3 and the report from Task 4.
- Produces: the published numbers.

Shared shell values (the session's worktree guard refuses `$VAR` command lines, so type the literal paths). The worktree is `/home/yuchen/YuchenZ/UE/philip_sa/.worktrees/hashemi-baseline`. The output folder is `OUT = /home/yuchen/YuchenZ/UE/philip_sa/runs/coco-baselines`. The data arguments are:

```
--checkpoint /home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth --coco-train-images /home/yuchen/YuchenZ/Datasets/coco/train2017 --coco-val-images /home/yuchen/YuchenZ/Datasets/coco/val2017 --coco-annotations /home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json --discopatch-root /home/yuchen/YuchenZ/UE/DisCoPatch
```

- [ ] **Step 1: Write the detached launcher**

Save this as `launch.sh` in this plan's workspace `.superpowers/sdd/2026-09-29-hashemi-baseline/`. `setsid`/`nohup` must live in a script, because the guard refuses them in plain commands.

```bash
#!/usr/bin/env bash
# usage: launch.sh PHASE WORKERS — runs one baselines-coco phase detached from the Claude session.
set -euo pipefail
PHASE=$1
WORKERS=$2
cd /home/yuchen/YuchenZ/UE/philip_sa/.worktrees/hashemi-baseline
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

- [ ] **Step 2: Agree a GPU window with `explore`**

Send `explore` a message via `SendMessage`: about 4 h at roughly 8 GB for fitting and scoring, then about 10 min with the GPU to ourselves for timing. Wait for the answer and start nothing before it. If the card has less than 8 GB free, wait for their announced pause.

- [ ] **Step 3: Fit the intervals (about 5–10 min)**

Run: `bash .superpowers/sdd/2026-09-29-hashemi-baseline/launch.sh hashemi-fit 6`, then wait until `OUT/logs/hashemi-fit.log` ends and `OUT/hashemi/intervals.npz` exists.
Check with:
`/home/yuchen/miniconda3/envs/UE/bin/python -c "import numpy as np; d=np.load('/home/yuchen/YuchenZ/UE/philip_sa/runs/coco-baselines/hashemi/intervals.npz'); print(int(d['images']), d['decoder_mean'].shape, d['encoder_s8_mean'].shape, d['encoder_s16_mean'].shape, d['encoder_s32_mean'].shape, float((d['decoder_std']==0).mean()))"`
Expected: `118287 (300, 256) (256, 80, 80) (256, 40, 40) (256, 20, 20)`, followed by a small share of zero-σ decoder neurons (close to 0.0).

- [ ] **Step 4: Smoke run on 20 images (about 5 min)**

```bash
mkdir -p /home/yuchen/YuchenZ/UE/philip_sa/runs/coco-baselines-smoke/hashemi
cp /home/yuchen/YuchenZ/UE/philip_sa/runs/coco-baselines/hashemi/intervals.npz /home/yuchen/YuchenZ/UE/philip_sa/runs/coco-baselines/hashemi/fit.json /home/yuchen/YuchenZ/UE/philip_sa/runs/coco-baselines-smoke/hashemi/
/home/yuchen/miniconda3/envs/UE/bin/python -m differential_uncertainty baselines-coco --phase hashemi-scores --limit 20 --workers 9 --output /home/yuchen/YuchenZ/UE/philip_sa/runs/coco-baselines-smoke --checkpoint /home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth --coco-train-images /home/yuchen/YuchenZ/Datasets/coco/train2017 --coco-val-images /home/yuchen/YuchenZ/Datasets/coco/val2017 --coco-annotations /home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json --discopatch-root /home/yuchen/YuchenZ/UE/DisCoPatch
/home/yuchen/miniconda3/envs/UE/bin/python -m differential_uncertainty baselines-coco --phase report --limit 20 --output /home/yuchen/YuchenZ/UE/philip_sa/runs/coco-baselines-smoke --checkpoint /home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth --coco-train-images /home/yuchen/YuchenZ/Datasets/coco/train2017 --coco-val-images /home/yuchen/YuchenZ/Datasets/coco/val2017 --coco-annotations /home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json --discopatch-root /home/yuchen/YuchenZ/UE/DisCoPatch
```

Expected: `runs/coco-baselines-smoke/results/report.md` has finite rows for "Hashemi et al., decoder queries" and "Hashemi et al., encoder maps (sensitivity)". In `conditions.csv`, the clean `mean_hashemi` is well below 0.5; for Gaussian neurons it would be near 0.05. These numbers are not reported.

- [ ] **Step 5: Full scoring pass (about 3 h; the corruptions are rebuilt on the CPU)**

Run: `bash .superpowers/sdd/2026-09-29-hashemi-baseline/launch.sh hashemi-scores 14` and wait for the process to exit.
Expected:
- `OUT/test_hashemi/` holds 5,000 `.npz` files plus `intervals.json`;
- the log ends with `[hashemi] 5000/5000`;
- there is no "corruptions differ", "intervals changed" or "non-finite" error.

If the run is interrupted, rerun the same command; it resumes.

- [ ] **Step 6: Runtime, with nothing else on the GPU (about 10 min)**

Confirm with `explore` that the card is free. Then run in the foreground:
`/home/yuchen/miniconda3/envs/UE/bin/python -m differential_uncertainty baselines-coco --phase timing --output /home/yuchen/YuchenZ/UE/philip_sa/runs/coco-baselines --checkpoint /home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth --coco-train-images /home/yuchen/YuchenZ/Datasets/coco/train2017 --coco-val-images /home/yuchen/YuchenZ/Datasets/coco/val2017 --coco-annotations /home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json --discopatch-root /home/yuchen/YuchenZ/UE/DisCoPatch`
Expected: `OUT/timing.json` contains `detector_plus_hashemi_ms`, plus fresh values for every earlier key. The earlier values may move by a few tenths of a millisecond; the published table uses the new file.

- [ ] **Step 7: Report (about 50 min)**

Run: `bash .superpowers/sdd/2026-09-29-hashemi-baseline/launch.sh report 9` and wait for the process to exit.
Expected: `OUT/results/summary.json` has `"hashemi_included": true` and `"hashemi_k": 2.0`. Then check that the earlier baselines did not move (same bootstrap seed and draws):

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -c "
import csv
old = {r['quantity']: r for r in csv.DictReader(open('/home/yuchen/YuchenZ/UE/philip_sa/.worktrees/hashemi-baseline/docs/results/coco-baselines/intervals.csv'))}
new = {r['quantity']: r for r in csv.DictReader(open('/home/yuchen/YuchenZ/UE/philip_sa/runs/coco-baselines/results/intervals.csv'))}
changed = [q for q in old if old[q] != new.get(q)]
print('unchanged earlier rows:', len(old) - len(changed), 'of', len(old), '| changed:', changed[:5])
"
```

Expected: `changed: []`.

- [ ] **Step 8: Publish the numbers**

Copy every CSV, `summary.json`, `report.md` and `timing.json` from `OUT/results/` and `OUT/` into `docs/results/coco-baselines/`. Then copy `OUT/hashemi/fit.json` to `docs/results/coco-baselines/hashemi_fit.json` and `OUT/test_hashemi/intervals.json` to `docs/results/coco-baselines/hashemi_intervals.json`.

Then edit `docs/coco-baseline-numbers.md`. Take every number from the new `report.md` and CSVs:
- **"The short version":** one bullet on where Hashemi et al. lands on separation (common and extra AUROC) and on harm (within-condition ρ and AURC), compared with the four earlier baselines.
- **Setup, baseline table:** a row "Hashemi et al. (FM 2023)", with the score = share of the last decoder layer's 300 × 256 neurons outside μ ± 2σ, fitted on 118,287 clean train images. Mention the encoder-map sensitivity row.
- **Separation tables (common and extra):** two rows, "Hashemi et al., decoder queries" and "Hashemi et al., encoder maps (sensitivity)", taken from `report.md`.
- **Per-family table:** one column "Hashemi (dec.)", AUROC at severity 3 / 5, from `separation.csv` (method `hashemi`). Update the bold best-at-severity-5 marks.
- **Harm table, per-severity ρ table and AURC severity-pool table:** rows or columns for both variants, from `harm.csv` and `aurc_pools.csv`.
- **Differences table:** add "Hashemi (dec.) − SAOD min", "Hashemi (dec.) − kNN" and "Hashemi (dec.) − DisCoPatch", from `differences.csv`.
- **Runtime table:** the `detector_plus_hashemi_ms` row and the refreshed earlier rows.
- **Deviations:** a new item listing the choices that differ from the paper:
  - the monitored layer (option A; the paper used PolyYOLO's last batch-norm or leaky-ReLU layer);
  - μ and σ from 118,287 images, not 500, as population values;
  - the raw score, not the conformal p-value, because all metrics are threshold-free;
  - option B reported as a sensitivity row.
- **Files:** list `hashemi_fit.json` and `hashemi_intervals.json`.

- [ ] **Step 9: Commit**

```bash
git add docs/results/coco-baselines docs/coco-baseline-numbers.md
git commit -m "docs: add Hashemi et al. to the COCO baseline numbers"
```

---

## Expected timeline

| Step | Time | Needs |
| --- | --- | --- |
| Tasks 1–4 (code) | about 1 h | CPU only |
| `hashemi-fit` | 5–10 min | GPU, about 8 GB |
| Smoke run | about 5 min | GPU |
| `hashemi-scores` | about 3 h | GPU about 5 GB, 14 CPU workers |
| Timing | about 10 min | GPU alone |
| Report | about 50 min | CPU |
