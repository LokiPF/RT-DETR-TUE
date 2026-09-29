# Activation-Monitor Baselines (Hashemi et al. and ICPR 2026) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two training-free, clean-only monitors that read the activations of the frozen RT-DETRv2-R18 as the fifth and sixth COCO baselines, and publish their numbers:
- the Gaussian neuron-interval runtime monitor of Hashemi, Křetínský, Rieder & Schmidt (FM 2023), on the last decoder layer (option A), with the encoder output maps (option B) as a sensitivity row;
- a reproduction of the activation-distribution monitor of Becker, Bayer, Hübner & Arens, "Operational Readiness for Object Detection" (ICPR 2026), on the backbone stages C1–C5.

**Architecture:**
- **Two small modules.** `baselines/hashemi.py` fits per-neuron mean and standard deviation in one streaming pass and scores the share of neurons outside μ ± 2σ. `baselines/activation_cdf.py` fits per-channel activation ranges and training-set histograms in two streaming passes, and scores the Earth Mover's distance between each channel's CDF and the training CDF.
- **The detector.** `DetectorTap` gains an optional hidden-layer capture, returning the decoder queries, the encoder maps and the backbone stages at once.
- **The pipeline.** Two fit phases (`hashemi-fit`, `cdf-fit`) and **one** scoring phase (`activation-scores`). The single scoring pass computes both monitors, so the three-hour rebuild of the corrupted images runs once. It checks the corruption digests against the existing detector pass.
- **The report** treats the three new scores like the other optional baseline (DisCoPatch).

**Tech Stack:** Python 3.11, PyTorch, NumPy, pytest; the existing `differential_uncertainty.baselines` package.

**Spec:**
- `docs/driving-benchmark-baselines-and-metrics.md`:
  - the Hashemi row under "Baselines we keep";
  - the activation-distribution monitor under "Candidate newer baselines". The user asked on 29 September 2026 to reproduce it ("you can try to reproduce the ICPR paper then").
  - "Open items → Hashemi et al. on RT-DETRv2", where option A was decided.
- `docs/literature-review-image-corruption-detection.md`, Section 12.
- The papers:
  - Hashemi et al.: arXiv 2212.07773, Sections 2.2–3.2.
  - Becker et al., ICPR 2026, LNCS 16814, pp. 121–135, Sections 3 and 4.2. The PDF is `978-3-032-31654-7_9.pdf` in the main checkout and is not in git.

**Before you start:** this revised plan must be committed on the branch (`hashemi-baseline`) before Task 1. The spec docs are already committed on `fingerprint_bank`.

## Global Constraints

- **Detector:** the frozen RT-DETRv2-R18 COCO checkpoint `/home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth`, with 640 × 640 input.
- **Evaluation protocol is unchanged:** all 5,000 COCO val2017 images in the seed-44 order, 5 folds, 96 conditions (clean first). The existing `runs/coco-baselines/test/*.npz` stay valid, and their digests must match the regenerated corruptions.
- **`run_config.json` must not change.** Do not add anything to `Settings.experiment()`.
- **Fitting data** for both monitors: all 118,287 clean COCO train2017 images; no labels, no corrupted data.
- **Hashemi et al.** (their Eqs. 5–6):
  - class information is discarded;
  - k = 2;
  - σ is the population standard deviation;
  - score = share of monitored neurons with |h − μ| > kσ (strict);
  - no conformal p-value, because our metrics are threshold-free and the p-value is a decreasing step function of the score.
- **Hashemi option A (the baseline):** all 300 × 256 neurons of the last decoder layer's output (`model.decoder.decoder.layers[-1]`, index 2).
- **Hashemi option B (the sensitivity row):** all neurons of the hybrid encoder's three output maps (256 channels at 80², 40² and 20²), pooled into one share.
- **ICPR 2026 activation-distribution monitor** (their Sect. 3 and 4.2, "CDFs backbone", their best variant on a ResNet backbone):
  - **Layers:** C1 = the stem output after max-pooling (64 × 160², read as the input of `backbone.res_layers[0]`), and C2–C5 = the outputs of the four residual stages (64 × 160², 128 × 80², 256 × 40², 512 × 20²). That is 1,024 channels.
  - **Histograms:** 1,000 bins per channel. The range is the training minimum and maximum, widened by 20% of their span on each side. A channel that was constant in training gets span 1. Values outside the range fall into the edge bins.
  - **Reference:** the histogram of every training image and position, summed, as a CDF.
  - **Distance:** the Earth Mover's distance between the image's per-channel CDF and the reference CDF, in units of the channel's range (the mean |ΔCDF| over bins).
  - **Score:** the sum over all 1,024 channels. There is no z-scoring, because z-scoring leaves a single score's AUROC unchanged.
- **The ICPR numbers cannot be reproduced exactly:**
  - their 10-level severity extension is not specified;
  - their detectors (Faster R-CNN R50, RT-DETR-l with HGNet) differ from ours.

  We reproduce the *method* on our detector and protocol, and compare with their numbers only in words.
- **Tests** run with `CUDA_VISIBLE_DEVICES=` and never touch the GPU. The suite command is:
  `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty -q`
- **Outputs** go to `/home/yuchen/YuchenZ/UE/philip_sa/runs/coco-baselines`, the main checkout's git-ignored `runs/`.
- **The GPU is shared.** Before any GPU step, message the `explore` session (tree-curves, OrthogonalDet) and agree on a gap. Its training holds about 30 GB of the 32 GB card.

## Review Focus

1. **A neuron that was constant on every training image (σ = 0) in Hashemi et al.** A test value equal to its mean stays inside. Any other value counts as outside. Nothing may divide by σ. (Task 1: `test_outside_counts_use_a_strict_k_sigma_band_and_keep_constant_neurons_inside`.)
2. **Activations with a large offset relative to their spread** (10⁶ ± 0.01). The streaming mean and standard deviation must match NumPy's. (Task 1: `test_streaming_stats_match_numpy_even_with_a_large_offset`.)
3. **Test activations outside the range seen in training, and channels that were constant in training** (CDF monitor). They must land in the edge bins, with no division by zero and no lost mass. (Task 2: `test_ranges_widen_by_the_margin_and_survive_a_constant_channel`, `test_histograms_count_every_activation_and_clamp_to_the_edge_bins`.)
4. **The same image scored alone or in a batch** must get the same score from both monitors. (Task 1: `test_monitor_scores_each_image_alone_whatever_its_batch`; Task 2: `test_monitor_scores_shifted_images_higher_and_each_image_alone`.)
5. **A fit replaced after scoring started.** A resumed scoring pass must refuse to mix scores from two fits. (Task 4: `test_activation_scores_cover_every_variant_and_stay_tied_to_their_fits`.)

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

### Task 2: The activation-distribution monitor (`activation_cdf.py`)

**Files:**
- Create: `differential_uncertainty/baselines/activation_cdf.py`
- Test: `tests/differential_uncertainty/test_baselines_activation_cdf.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces:
  - `BINS = 1000`, `MARGIN = 0.2`, `STAGES = ("C1", "C2", "C3", "C4", "C5")`
  - `class ChannelRanges` with `.update(values: Tensor[n, C, H, W])` and `.result(margin=MARGIN) -> (low, high)` as float64 arrays of shape (C,)
  - `channel_histograms(values, low, high, bins=BINS) -> Tensor[n, C, bins]` (counts)
  - `emd_to_reference(counts: Tensor[n, C, bins], reference_cdf: Tensor[C, bins]) -> Tensor[n]`
  - `class ReferenceHistograms(bounds: dict[stage, (low, high)], device, bins=BINS)` with `.update(stages: list[Tensor])` and `.save(path, images: int)`, which writes an npz with `images`, `bins`, `margin`, `{stage}_low`, `{stage}_high` and `{stage}_cdf`
  - `class CdfMonitor(path, device)` with `.scores(stages: list[Tensor]) -> np.ndarray[n]`

- [ ] **Step 1: Write the failing tests**

Create `tests/differential_uncertainty/test_baselines_activation_cdf.py`:

```python
import numpy as np
import pytest
import torch

from differential_uncertainty.baselines import activation_cdf as cdf


def test_ranges_widen_by_the_margin_and_survive_a_constant_channel():
    ranges = cdf.ChannelRanges()
    values = torch.zeros(2, 2, 3, 3)
    values[0, 0] = 10.0            # channel 0 spans [0, 10] over the two images
    values[:, 1] = 3.0             # channel 1 is constant
    ranges.update(values[:1])
    ranges.update(values[1:])
    low, high = ranges.result()
    assert low.tolist() == pytest.approx([-2.0, 2.8]) and high.tolist() == pytest.approx([12.0, 3.2])


def test_histograms_count_every_activation_and_clamp_to_the_edge_bins():
    values = torch.tensor([0.1, 0.3, 0.6, 0.99, -5.0, 7.0]).view(1, 1, 2, 3)
    counts = cdf.channel_histograms(values, torch.tensor([0.0]), torch.tensor([1.0]), bins=4)
    assert counts.shape == (1, 1, 4) and counts[0, 0].tolist() == [2.0, 1.0, 1.0, 2.0]


def test_emd_is_zero_for_the_reference_and_one_bin_width_per_shifted_channel():
    reference = torch.ones(2, 4)                                    # all mass in bin 0, two channels
    same = torch.tensor([[[5.0, 0, 0, 0], [2.0, 0, 0, 0]]])
    shifted = torch.tensor([[[0, 5.0, 0, 0], [2.0, 0, 0, 0]]])      # channel 0 moved by one bin of four
    assert cdf.emd_to_reference(same, reference).item() == pytest.approx(0.0)
    assert cdf.emd_to_reference(shifted, reference).item() == pytest.approx(0.25)


SHAPES = [(2, 4, 4), (2, 4, 4), (3, 2, 2), (3, 2, 2), (4, 1, 1)]


def _stages(rng, n, shift=0.0):
    return [torch.from_numpy(rng.normal(shift, 1.0, size=(n, *s))).float() for s in SHAPES]


def _reference(tmp_path, rng):
    training = [_stages(rng, 20) for _ in range(3)]
    ranges = {stage: cdf.ChannelRanges() for stage in cdf.STAGES}
    for batch in training:
        for stage, values in zip(cdf.STAGES, batch):
            ranges[stage].update(values)
    reference = cdf.ReferenceHistograms({s: r.result() for s, r in ranges.items()}, "cpu", bins=50)
    for batch in training:
        reference.update(batch)
    path = tmp_path / "reference.npz"
    reference.save(path, images=60)
    return path


def test_monitor_scores_shifted_images_higher_and_each_image_alone(tmp_path):
    rng = np.random.default_rng(0)
    monitor = cdf.CdfMonitor(_reference(tmp_path, rng), "cpu")
    clean, shifted = _stages(rng, 4), _stages(rng, 4, shift=2.0)
    clean_scores, shifted_scores = monitor.scores(clean), monitor.scores(shifted)
    assert shifted_scores.min() > clean_scores.max()
    assert monitor.scores([s[:1] for s in clean])[0] == pytest.approx(clean_scores[0])


def test_monitor_needs_all_five_backbone_stages_with_the_fitted_channels(tmp_path):
    monitor = cdf.CdfMonitor(_reference(tmp_path, np.random.default_rng(1)), "cpu")
    stages = _stages(np.random.default_rng(2), 1)
    with pytest.raises(ValueError, match="five"):
        monitor.scores(stages[:4])
    stages[0] = torch.zeros(1, 7, 4, 4)
    with pytest.raises(ValueError, match="channels"):
        monitor.scores(stages)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_baselines_activation_cdf.py -q`
Expected: collection error `ImportError: cannot import name 'activation_cdf'`.

- [ ] **Step 3: Write the implementation**

Create `differential_uncertainty/baselines/activation_cdf.py`:

```python
"""Activation-distribution monitor, reproduced from Becker, Bayer, Hübner & Arens (ICPR 2026).

Each channel of each monitored layer is summarised, per image, by a histogram of its activations over all
spatial positions. The histogram has 1,000 bins on a fixed per-channel range: the training minimum and
maximum, widened by 20% of their span. It becomes a CDF and is compared with the same channel's CDF pooled
over all training images, using the Earth Mover's distance in units of the channel's range. The image score
is the sum over all channels and layers. Their final z-scoring is left out: it does not change a single
score's AUROC.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

BINS = 1000
MARGIN = 0.2
STAGES = ("C1", "C2", "C3", "C4", "C5")


class ChannelRanges:
    """Streaming per-channel minimum and maximum over images and positions."""

    def __init__(self):
        self.low = None
        self.high = None

    def update(self, values: torch.Tensor) -> None:
        low = values.detach().amin(dim=(0, 2, 3)).double()
        high = values.detach().amax(dim=(0, 2, 3)).double()
        if self.low is None:
            self.low, self.high = low, high
            return
        if low.shape != self.low.shape:
            raise ValueError(f"channel count changed from {self.low.numel()} to {low.numel()}")
        self.low, self.high = torch.minimum(self.low, low), torch.maximum(self.high, high)

    def result(self, margin: float = MARGIN) -> tuple[np.ndarray, np.ndarray]:
        if self.low is None:
            raise ValueError("no activations were seen")
        span = self.high - self.low
        span = torch.where(span > 0, span, torch.ones_like(span))  # a constant channel gets span 1
        return (self.low - margin * span).cpu().numpy(), (self.high + margin * span).cpu().numpy()


def channel_histograms(values: torch.Tensor, low: torch.Tensor, high: torch.Tensor, bins: int = BINS) -> torch.Tensor:
    """Per image and channel, counts in `bins` equal bins on [low, high]; values outside go to the edge bins."""
    n, channels = values.shape[:2]
    if low.numel() != channels:
        raise ValueError(f"activations have {channels} channels, the fit has {low.numel()}")
    flat = values.detach().float().flatten(2)
    low = low.float().view(1, channels, 1)
    width = high.float().view(1, channels, 1) - low
    index = ((flat - low) / width * bins).floor_().clamp_(0, bins - 1).long()
    counts = torch.zeros(n, channels, bins, device=values.device)
    counts.scatter_add_(2, index, torch.ones_like(flat))
    return counts


def emd_to_reference(counts: torch.Tensor, reference_cdf: torch.Tensor) -> torch.Tensor:
    """Per image, the sum over channels of the EMD between its CDF and the reference CDF, in range units."""
    cdf = counts.cumsum(dim=2) / counts.sum(dim=2, keepdim=True)
    return (cdf - reference_cdf).abs().mean(dim=2).sum(dim=1)


class ReferenceHistograms:
    """Training-set histograms per stage and channel, on fixed ranges."""

    def __init__(self, bounds: dict, device, bins: int = BINS):
        self.bins = bins
        self.low = {s: torch.as_tensor(bounds[s][0], device=device) for s in STAGES}
        self.high = {s: torch.as_tensor(bounds[s][1], device=device) for s in STAGES}
        self.counts = {s: torch.zeros(self.low[s].numel(), bins, dtype=torch.float64, device=device) for s in STAGES}

    def update(self, stages) -> None:
        if len(stages) != len(STAGES):
            raise ValueError(f"expected the five backbone stages, got {len(stages)}")
        for stage, values in zip(STAGES, stages):
            self.counts[stage] += channel_histograms(values, self.low[stage], self.high[stage], self.bins).sum(dim=0).double()

    def save(self, path, images: int) -> None:
        arrays = {}
        for stage in STAGES:
            counts = self.counts[stage]
            if bool((counts.sum(dim=1) == 0).any()):
                raise ValueError(f"no training activations were counted for {stage}")
            arrays[f"{stage}_low"] = self.low[stage].double().cpu().numpy()
            arrays[f"{stage}_high"] = self.high[stage].double().cpu().numpy()
            arrays[f"{stage}_cdf"] = (counts.cumsum(dim=1) / counts.sum(dim=1, keepdim=True)).cpu().numpy()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, images=np.array(images), bins=np.array(self.bins), margin=np.array(MARGIN), **arrays)
        temporary.replace(path)


class CdfMonitor:
    def __init__(self, path, device):
        with np.load(path, allow_pickle=False) as data:
            self.bins = int(data["bins"])
            self.low = {s: torch.from_numpy(data[f"{s}_low"]).float().to(device) for s in STAGES}
            self.high = {s: torch.from_numpy(data[f"{s}_high"]).float().to(device) for s in STAGES}
            self.cdf = {s: torch.from_numpy(data[f"{s}_cdf"]).float().to(device) for s in STAGES}

    def scores(self, stages) -> np.ndarray:
        if len(stages) != len(STAGES):
            raise ValueError(f"expected the five backbone stages, got {len(stages)}")
        total = None
        for stage, values in zip(STAGES, stages):
            counts = channel_histograms(values, self.low[stage], self.high[stage], self.bins)
            part = emd_to_reference(counts, self.cdf[stage]).double()
            total = part if total is None else total + part
        return total.cpu().numpy()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_baselines_activation_cdf.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add differential_uncertainty/baselines/activation_cdf.py tests/differential_uncertainty/test_baselines_activation_cdf.py
git commit -m "feat: reproduce the ICPR 2026 activation-distribution monitor"
```

---

### Task 3: Hidden-layer capture in `DetectorTap`

**Files:**
- Modify: `differential_uncertainty/baselines/detector.py` (the whole file is shown below)
- Test: `tests/differential_uncertainty/test_baselines_detector.py`

**Interfaces:**
- Consumes: nothing from Tasks 1–2.
- Produces:
  - `DetectorTap(checkpoint_path, device, image_size=(640, 640), hidden=False)`. With `hidden=True`, `.forward_hidden(batch) -> dict` with:
    - `"decoder"`: Tensor[n, 300, 256];
    - `"encoder"`: a list of 3 Tensors;
    - `"backbone"`: a list of 5 Tensors, C1..C5.

    All are float32 on the tap's device.
  - `.prepare`, `.forward` and `.run` are unchanged.
  - `DECODER_SHAPE = (300, 256)`, `STAGE_COUNT = 5`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/differential_uncertainty/test_baselines_detector.py`:

```python
class FakeBackbone(nn.Module):
    """C1 (input of the first stage) = level + 1, and stage i adds 1 each time: C1..C5 = level + 1, 1, 2, 3, 4."""

    def __init__(self):
        super().__init__()
        self.res_layers = nn.ModuleList([nn.Identity() for _ in range(4)])

    def forward(self, images):
        n = images.shape[0]
        x = images.mean(dim=(1, 2, 3)).view(n, 1, 1, 1).expand(n, 4, 4, 4).clone()
        for stage in self.res_layers:
            x = stage(x + 1.0)
        return [torch.zeros(n, 256, 4, 4), torch.ones(n, 512, 2, 2)]


class FakeHiddenDetector(nn.Module):
    """Last decoder layer output = 3 * level + 3; first encoder map = level (level = mean image value)."""

    def __init__(self):
        super().__init__()
        self.backbone = FakeBackbone()
        self.encoder = nn.Identity()
        self.decoder = nn.Module()
        self.decoder.decoder = nn.Module()
        self.decoder.decoder.layers = nn.ModuleList([nn.Identity() for _ in range(3)])

    def forward(self, images):
        n = images.shape[0]
        level = images.mean(dim=(1, 2, 3))
        self.backbone(images)
        self.encoder([level.view(n, 1, 1, 1).expand(n, 256, 4, 4).clone(),
                      torch.zeros(n, 256, 2, 2), torch.zeros(n, 256, 1, 1)])
        queries = torch.zeros(n, 300, 256)
        for index, layer in enumerate(self.decoder.decoder.layers):
            queries = layer(queries + index + level.view(n, 1, 1))
        return {"pred_logits": torch.zeros(n, 300, 80), "pred_boxes": torch.full((n, 300, 4), 0.5)}


def test_hidden_tap_returns_decoder_encoder_and_backbone_stages(monkeypatch):
    monkeypatch.setattr(detector, "load_frozen_detector", lambda _p, _d: FakeHiddenDetector())
    arrays = [np.full((20, 30, 3), 255, np.uint8), np.zeros((10, 10, 3), np.uint8)]
    with detector.DetectorTap("unused.pth", "cpu", image_size=(16, 16), hidden=True) as tap:
        hidden = tap.forward_hidden(tap.prepare(arrays))
        logits, _, _ = tap.run(arrays, batch_size=2)   # the normal path still works next to the extra hooks
    decoder, encoder, backbone = hidden["decoder"], hidden["encoder"], hidden["backbone"]
    assert tuple(decoder.shape) == (2, 300, 256)
    assert [tuple(e.shape) for e in encoder] == [(2, 256, 4, 4), (2, 256, 2, 2), (2, 256, 1, 1)]
    assert decoder[0, 0, 0].item() == pytest.approx(6.0) and decoder[1, 0, 0].item() == pytest.approx(3.0)
    assert encoder[0][0, 0, 0, 0].item() == pytest.approx(1.0)
    assert [round(b[0, 0, 0, 0].item(), 4) for b in backbone] == [2.0, 2.0, 3.0, 4.0, 5.0]
    assert logits.shape == (2, 300, 80)


def test_hidden_layers_need_a_hidden_tap(monkeypatch):
    monkeypatch.setattr(detector, "load_frozen_detector", lambda _p, _d: FakeHiddenDetector())
    with detector.DetectorTap("unused.pth", "cpu", image_size=(8, 8)) as tap:
        with pytest.raises(RuntimeError, match="hidden=True"):
            tap.forward_hidden(tap.prepare([np.zeros((8, 8, 3), np.uint8)]))
```

What the backbone assertion checks: the white image has level 1. C1 is the input of the first stage, 1 + 1 = 2. The outputs of stages 1–4 are 2, 3, 4 and 5, because every stage input adds 1.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_baselines_detector.py -q`
Expected: 2 failed (`TypeError: ... unexpected keyword argument 'hidden'` and `AttributeError: ... 'forward_hidden'`), 2 passed.

- [ ] **Step 3: Write the implementation**

Replace `differential_uncertainty/baselines/detector.py` with:

```python
"""Frozen RT-DETRv2 forward pass exposing logits, boxes, the pooled last backbone stage and, on request,
the backbone stages C1-C5, the hybrid encoder's output maps and the last decoder layer's queries."""
from __future__ import annotations

from functools import partial

import numpy as np
import torch
from PIL import Image

from ..extraction import load_frozen_detector, prepare_image

QUERY_COUNT = 300
CLASS_COUNT = 80
POOLED_DIM = 512
DECODER_SHAPE = (QUERY_COUNT, 256)
STAGE_COUNT = 5  # C1 (stem output after max-pooling) and the four residual stages


class DetectorTap:
    def __init__(self, checkpoint_path, device, image_size=(640, 640), hidden=False):
        self.device = torch.device(device)
        self.image_size = tuple(image_size)
        self.model = load_frozen_detector(checkpoint_path, self.device)
        self._reset()
        self._handles = [self.model.backbone.register_forward_hook(self._capture)]
        self.hidden = hidden
        if hidden:
            stages = self.model.backbone.res_layers
            self._handles.append(stages[0].register_forward_pre_hook(self._capture_stem))
            self._handles += [stage.register_forward_hook(partial(self._capture_stage, index + 1))
                              for index, stage in enumerate(stages)]
            self._handles.append(self.model.decoder.decoder.layers[-1].register_forward_hook(self._capture_decoder))
            self._handles.append(self.model.encoder.register_forward_hook(self._capture_encoder))

    def _reset(self):
        self._pooled = self._decoder = self._encoder = None
        self._stages = [None] * STAGE_COUNT

    def _capture(self, _module, _inputs, outputs):
        self._pooled = outputs[-1].mean(dim=(2, 3))

    def _capture_stem(self, _module, inputs):
        self._stages[0] = inputs[0]

    def _capture_stage(self, index, _module, _inputs, output):
        self._stages[index] = output

    def _capture_decoder(self, _module, _inputs, output):
        self._decoder = output

    def _capture_encoder(self, _module, _inputs, outputs):
        self._encoder = list(outputs)

    def prepare(self, arrays) -> torch.Tensor:
        return torch.stack([prepare_image(Image.fromarray(a), self.image_size) for a in arrays])

    @torch.inference_mode()
    def forward(self, batch: torch.Tensor):
        outputs = self.model(batch.to(self.device, non_blocking=True))
        pooled = self._pooled
        self._reset()
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
    def forward_hidden(self, batch: torch.Tensor) -> dict:
        """Decoder queries (n, 300, 256), encoder output maps and backbone stages C1-C5, float32 on the device."""
        if not self.hidden:
            raise RuntimeError("create the tap with hidden=True to read hidden layers")
        self.model(batch.to(self.device, non_blocking=True))
        decoder, encoder, stages = self._decoder, self._encoder, self._stages
        self._reset()
        if decoder is None or encoder is None or any(s is None for s in stages):
            raise RuntimeError("a hidden-layer hook did not run")
        n = batch.shape[0]
        if tuple(decoder.shape) != (n, *DECODER_SHAPE):
            raise ValueError(f"decoder queries have shape {tuple(decoder.shape)}, expected {(n, *DECODER_SHAPE)}")
        hidden = {"decoder": decoder.float(), "encoder": [e.float() for e in encoder],
                  "backbone": [s.float() for s in stages]}
        if not all(bool(torch.isfinite(t).all()) for t in [hidden["decoder"], *hidden["encoder"], *hidden["backbone"]]):
            raise ValueError("hidden activations must be finite")
        return hidden

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
git commit -m "feat: let the detector tap read backbone stages, encoder maps and decoder queries"
```

---

### Task 4: Pipeline phases `hashemi-fit`, `cdf-fit` and `activation-scores`, timing and CLI

**Files:**
- Modify: `differential_uncertainty/baselines/pipeline.py` (imports, new helpers and phases, `phase_timing`, `PHASES`)
- Modify: `differential_uncertainty/cli.py` (the `--phase` choices)
- Test: `tests/differential_uncertainty/test_baselines_pipeline.py`, `tests/differential_uncertainty/test_cli.py`

**Interfaces:**
- Consumes:
  - Task 1: `K`, `LAYERS`, `NeuronStats`, `save_intervals`, `HashemiMonitor(.scores, .decoder_share)`.
  - Task 2: `BINS`, `MARGIN`, `STAGES`, `ChannelRanges`, `ReferenceHistograms`, `CdfMonitor`.
  - Task 3: `DetectorTap(..., hidden=True).forward_hidden(batch) -> {"decoder", "encoder", "backbone"}`.
- Produces:
  - `runs/.../hashemi/intervals.npz` and `hashemi/fit.json`.
  - `runs/.../cdf/reference.npz` and `cdf/fit.json`.
  - `runs/.../test_activation/{stem}.npz`, holding `hashemi_decoder`, `hashemi_encoder` and `cdf_backbone`, each of shape (96,).
  - `test_activation/fits.json`, with the sha1 of both fits.
  - `pipeline.ACTIVATION_KEYS = ("hashemi_decoder", "hashemi_encoder", "cdf_backbone")`.
  - `timing.json` gains `detector_plus_hashemi_ms` and `detector_plus_cdf_ms`.
  - CLI phases `hashemi-fit`, `cdf-fit` and `activation-scores`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/differential_uncertainty/test_baselines_pipeline.py`:

```python
class HiddenFakeTap(FakeTap):
    """FakeTap plus the hidden-layer interface; every activation shifts with image brightness."""
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

        def maps(channels, sizes):
            return [level.view(n, 1, 1, 1).expand(n, channels, s, s).clone()
                    + torch.randn(channels, s, s, generator=generator) for s in sizes]

        return {"decoder": torch.randn(300, 256, generator=generator) + 4.0 * level.view(n, 1, 1),
                "encoder": maps(8, (4, 2, 1)), "backbone": maps(3, (4, 4, 2, 2, 1))}


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


def test_cdf_fit_stores_ranges_and_reference_cdfs_for_the_five_stages_once(tmp_path, hidden_fakes):
    settings = _settings(tmp_path, train_images=_train_images(tmp_path))
    pipeline.run_phase("cdf-fit", settings)
    with np.load(settings.output / "cdf" / "reference.npz") as data:
        assert data["C1_cdf"].shape == (3, 1000) and data["C5_low"].shape == (3,)
        assert np.allclose(data["C3_cdf"][:, -1], 1.0) and int(data["images"]) == 3
    assert json.loads((settings.output / "cdf" / "fit.json").read_text())["bins"] == 1000
    assert HiddenFakeTap.hidden_calls == 2   # one ranges pass and one histogram pass over the single batch
    pipeline.run_phase("cdf-fit", settings)
    assert HiddenFakeTap.hidden_calls == 2


def test_activation_scores_cover_every_variant_and_stay_tied_to_their_fits(tmp_path, hidden_fakes):
    settings = _settings(tmp_path, train_images=_train_images(tmp_path))
    pipeline.run_phase("test", settings)
    pipeline.run_phase("hashemi-fit", settings)
    with pytest.raises(ValueError, match="cdf-fit"):
        pipeline.run_phase("activation-scores", settings)
    pipeline.run_phase("cdf-fit", settings)
    pipeline.run_phase("activation-scores", settings)
    files = sorted((settings.output / "test_activation").glob("*.npz"))
    assert len(files) == 2
    with np.load(files[0]) as data:
        assert all(data[key].shape == (96,) for key in pipeline.ACTIVATION_KEYS)
        assert np.all((data["hashemi_decoder"] >= 0) & (data["hashemi_decoder"] <= 1))
        assert np.all(data["cdf_backbone"] >= 0)
    reference = settings.output / "cdf" / "reference.npz"
    reference.write_bytes(reference.read_bytes() + b"x")
    with pytest.raises(ValueError, match="changed since"):
        pipeline.run_phase("activation-scores", settings)


def test_activation_scores_detect_changed_corruptions(tmp_path, hidden_fakes):
    settings = _settings(tmp_path, train_images=_train_images(tmp_path))
    pipeline.run_phase("test", settings)
    pipeline.run_phase("hashemi-fit", settings)
    pipeline.run_phase("cdf-fit", settings)
    victim = sorted((settings.output / "test").glob("*.npz"))[0]
    data = dict(np.load(victim))
    data["digests"] = np.array(["0" * 16] * 96)
    np.savez(victim, **data)
    with pytest.raises(ValueError, match="corruptions differ"):
        pipeline.run_phase("activation-scores", settings)


def test_timing_reports_both_activation_monitors_once_they_are_fitted(tmp_path, hidden_fakes):
    settings = _settings(tmp_path, train_images=_train_images(tmp_path))
    pipeline.run_phase("hashemi-fit", settings)
    pipeline.run_phase("cdf-fit", settings)
    pipeline.run_phase("timing", settings)
    timing = json.loads((settings.output / "timing.json").read_text())
    assert {"detector_ms", "detector_plus_hashemi_ms", "detector_plus_cdf_ms"} <= set(timing)
    assert "discopatch_ms" not in timing
```

Append to `tests/differential_uncertainty/test_cli.py`:

```python
def test_cli_accepts_the_activation_monitor_phases(monkeypatch, tmp_path):
    import differential_uncertainty.baselines.pipeline as pipeline
    seen = []
    monkeypatch.setattr(pipeline, "run_phase", lambda phase, settings: seen.append(phase))
    base = ["--output", str(tmp_path / "out"), "--checkpoint", "c.pth", "--coco-train-images", "train",
            "--coco-val-images", "val", "--coco-annotations", "ann.json", "--discopatch-root", "dcp"]
    phases = ("hashemi-fit", "cdf-fit", "activation-scores")
    for phase in phases:
        assert cli.main(["baselines-coco", "--phase", phase, *base]) == 0
    assert tuple(seen) == phases
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_baselines_pipeline.py tests/differential_uncertainty/test_cli.py -q`
Expected:
- The five new pipeline tests fail with `ValueError: unknown phase 'hashemi-fit'`, or `'cdf-fit'` for the CDF test.
- The CLI test fails with `SystemExit: 2` (invalid choice).
- All older tests pass.

- [ ] **Step 3: Write the implementation**

In `differential_uncertainty/baselines/pipeline.py`:

1. Change the module docstring's first line to `"""Resumable phases that compute the baseline scores on the fixed COCO protocol."""`.
2. Below `from . import discopatch, protocol`, add:

```python
from .activation_cdf import BINS as CDF_BINS
from .activation_cdf import MARGIN as CDF_MARGIN
from .activation_cdf import STAGES as CDF_STAGES
from .activation_cdf import CdfMonitor, ChannelRanges, ReferenceHistograms
```

   Below `from .discopatch import DisCoPatchScorer, train_discopatch`, add:

```python
from .hashemi import K as HASHEMI_K
from .hashemi import LAYERS as HASHEMI_LAYERS
from .hashemi import HashemiMonitor, NeuronStats, save_intervals
```

3. Below the `TEST_KEYS = (...)` definition, add:

```python
ACTIVATION_KEYS = ("hashemi_decoder", "hashemi_encoder", "cdf_backbone")
```

4. Directly above `def phase_timing`, add:

```python
def _hashemi_intervals(settings: Settings) -> Path:
    return settings.output / "hashemi" / "intervals.npz"


def _cdf_reference(settings: Settings) -> Path:
    return settings.output / "cdf" / "reference.npz"


def _train_loader(settings: Settings, paths) -> DataLoader:
    return DataLoader(_Prepared(paths), batch_size=settings.batch_size,
                      num_workers=settings.workers, pin_memory=True)


def phase_hashemi_fit(settings: Settings) -> None:
    """Per-neuron mean and standard deviation over every clean COCO train image (Hashemi et al., Sec. 3.1)."""
    path = _hashemi_intervals(settings)
    if path.exists():
        return
    paths = protocol.list_images(settings.train_images)
    stats = {name: NeuronStats() for name in HASHEMI_LAYERS}
    started = time.time()
    with DetectorTap(settings.checkpoint, settings.device, hidden=True) as tap:
        for index, batch in enumerate(_train_loader(settings, paths)):
            hidden = tap.forward_hidden(batch)
            if len(hidden["encoder"]) != len(HASHEMI_LAYERS) - 1:
                raise ValueError(f"expected the encoder's three output maps, got {len(hidden['encoder'])}")
            stats["decoder"].update(hidden["decoder"])
            for name, values in zip(HASHEMI_LAYERS[1:], hidden["encoder"]):
                stats[name].update(values)
            if index % 200 == 0:
                _progress("hashemi-fit", min((index + 1) * settings.batch_size, len(paths)), len(paths), started)
    save_intervals(path, stats, images=len(paths))
    _atomic_json(path.with_name("fit.json"), {"images": len(paths), "k": HASHEMI_K,
                                              "std": "population (ddof=0)", "layers": list(HASHEMI_LAYERS)})


def phase_cdf_fit(settings: Settings) -> None:
    """Per-channel ranges, then training histograms of backbone stages C1-C5 (Becker et al., ICPR 2026)."""
    path = _cdf_reference(settings)
    if path.exists():
        return
    paths = protocol.list_images(settings.train_images)
    ranges = {stage: ChannelRanges() for stage in CDF_STAGES}
    with DetectorTap(settings.checkpoint, settings.device, hidden=True) as tap:
        started = time.time()
        for index, batch in enumerate(_train_loader(settings, paths)):
            stages = tap.forward_hidden(batch)["backbone"]
            if len(stages) != len(CDF_STAGES):
                raise ValueError(f"expected the five backbone stages, got {len(stages)}")
            for stage, values in zip(CDF_STAGES, stages):
                ranges[stage].update(values)
            if index % 200 == 0:
                _progress("cdf-fit ranges", min((index + 1) * settings.batch_size, len(paths)), len(paths), started)
        reference = ReferenceHistograms({s: r.result() for s, r in ranges.items()}, settings.device)
        started = time.time()
        for index, batch in enumerate(_train_loader(settings, paths)):
            reference.update(tap.forward_hidden(batch)["backbone"])
            if index % 200 == 0:
                _progress("cdf-fit histograms", min((index + 1) * settings.batch_size, len(paths)), len(paths), started)
    reference.save(path, images=len(paths))
    _atomic_json(path.with_name("fit.json"), {"images": len(paths), "bins": CDF_BINS, "margin": CDF_MARGIN,
                                              "stages": list(CDF_STAGES), "passes": 2})


def _activation_scores(tap, hashemi_monitor, cdf_monitor, arrays, batch_size) -> dict:
    parts = {key: [] for key in ACTIVATION_KEYS}
    for start in range(0, len(arrays), batch_size):
        hidden = tap.forward_hidden(tap.prepare(arrays[start:start + batch_size]))
        decoder_share, encoder_share = hashemi_monitor.scores(hidden["decoder"], hidden["encoder"])
        parts["hashemi_decoder"].append(decoder_share)
        parts["hashemi_encoder"].append(encoder_share)
        parts["cdf_backbone"].append(cdf_monitor.scores(hidden["backbone"]))
    return {key: np.concatenate(values) for key, values in parts.items()}


def phase_activation_scores(settings: Settings) -> None:
    """Both activation monitors in one pass over every evaluation image and condition."""
    fits = {"hashemi": _hashemi_intervals(settings), "cdf": _cdf_reference(settings)}
    missing = [name for name, path in fits.items() if not path.exists()]
    if missing:
        raise ValueError("run the " + " and ".join(f"{name}-fit" for name in missing) + " phase first")
    folder = settings.output / "test_activation"
    record = {name: {"path": str(path), "sha1": hashlib.sha1(path.read_bytes()).hexdigest()}
              for name, path in fits.items()}
    record.update(hashemi_k=HASHEMI_K, cdf_bins=CDF_BINS)
    marker = folder / "fits.json"
    if marker.exists():
        if json.loads(marker.read_text()) != record:
            raise ValueError(f"the fitted intervals or reference changed since {marker} was written")
    else:
        _atomic_json(marker, record)
    pending = [p for p in evaluation(settings) if not _valid_existing(folder / f"{p.stem}.npz", ACTIVATION_KEYS)]
    absent = [p.name for p in pending if not (settings.output / "test" / f"{p.stem}.npz").exists()]
    if absent:
        raise ValueError(f"run the test phase first: {len(absent)} detector results are missing, e.g. {absent[0]}")
    if not pending:
        return
    hashemi_monitor = HashemiMonitor(fits["hashemi"], settings.device)
    cdf_monitor = CdfMonitor(fits["cdf"], settings.device)
    started = time.time()
    with DetectorTap(settings.checkpoint, settings.device, hidden=True) as tap:
        for done, (name, arrays) in enumerate(_variant_stream(settings, pending), start=1):
            stem = Path(name).stem
            stored = _load_npz(settings.output / "test" / f"{stem}.npz", TEST_KEYS)["digests"]
            if list(stored) != [protocol.digest(a) for a in arrays]:
                raise ValueError(f"corruptions differ from the detector pass for {name}")
            values = _activation_scores(tap, hashemi_monitor, cdf_monitor, arrays, settings.batch_size)
            if not all(np.isfinite(v).all() for v in values.values()):
                raise ValueError(f"an activation monitor returned non-finite scores for {name}")
            _atomic_npz(folder / f"{stem}.npz", **values)
            if done % 25 == 0:
                _progress("activation", done, len(pending), started)
```

5. In `phase_timing`, directly before `_atomic_json(settings.output / "timing.json", result)`, add:

```python
    intervals, reference = _hashemi_intervals(settings), _cdf_reference(settings)
    if intervals.exists() or reference.exists():
        # a separate tap, so the hidden-layer hooks never touch the other timings
        with DetectorTap(settings.checkpoint, settings.device, hidden=True) as hidden_tap:
            if intervals.exists():
                hashemi_monitor = HashemiMonitor(intervals, settings.device)

                def hashemi_score(array):
                    hashemi_monitor.decoder_share(hidden_tap.forward_hidden(hidden_tap.prepare([array]))["decoder"])

                result["detector_plus_hashemi_ms"] = timed(hashemi_score)
            if reference.exists():
                cdf_monitor = CdfMonitor(reference, settings.device)

                def cdf_score(array):
                    cdf_monitor.scores(hidden_tap.forward_hidden(hidden_tap.prepare([array]))["backbone"])

                result["detector_plus_cdf_ms"] = timed(cdf_score)
```

6. Replace the `PHASES` dict with:

```python
PHASES = {
    "sanity": phase_sanity, "bank": phase_bank, "test": phase_test,
    "train-discopatch": phase_train_discopatch, "discopatch-scores": phase_discopatch_scores,
    "hashemi-fit": phase_hashemi_fit, "cdf-fit": phase_cdf_fit, "activation-scores": phase_activation_scores,
    "timing": phase_timing, "report": phase_report,
}
```

In `differential_uncertainty/cli.py`, replace the `--phase` choices line with:

```python
                           choices=["sanity", "bank", "test", "train-discopatch", "discopatch-scores",
                                    "hashemi-fit", "cdf-fit", "activation-scores", "timing", "report"])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_baselines_pipeline.py tests/differential_uncertainty/test_cli.py -q`
Expected: all pass (the 5 new pipeline tests, the new CLI test, and every older test).

- [ ] **Step 5: Commit**

```bash
git add differential_uncertainty/baselines/pipeline.py differential_uncertainty/cli.py tests/differential_uncertainty/test_baselines_pipeline.py tests/differential_uncertainty/test_cli.py
git commit -m "feat: add fit and scoring phases for both activation monitors"
```

---

### Task 5: The activation monitors in the report

**Files:**
- Modify: `differential_uncertainty/baselines/report.py` (imports, `METHODS`, `LABELS`, `method_scores`, `build_report`)
- Test: `tests/differential_uncertainty/test_baselines_report.py`

**Interfaces:**
- Consumes:
  - Task 1: `hashemi.K`.
  - Task 2: `activation_cdf.BINS`.
  - Task 4: the `test_activation/{stem}.npz` files with `hashemi_decoder`, `hashemi_encoder` and `cdf_backbone`.
- Produces:
  - The methods `hashemi`, `hashemi_enc` and `cdf` in every table.
  - `summary.json` keys `activation_monitors_included`, `hashemi_k` and `cdf_bins`.
  - `method_scores(test, dcp, per_image_lambda, k=KNN_K, activation=None)`.

- [ ] **Step 1: Write the failing tests**

In `tests/differential_uncertainty/test_baselines_report.py`:

1. Add `import csv` at the top, next to `import json`.
2. Replace the whole `test_build_report_end_to_end_on_a_tiny_fixture` function with the helper and the tests below. The helper's body is the old test's setup, unchanged.

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
    assert summary["discopatch_included"] is False and summary["activation_monitors_included"] is False
    for name in ("separation", "aggregates", "harm", "aurc_pools", "conditions", "intervals", "differences", "knn_k"):
        assert (results / f"{name}.csv").exists(), name
    assert "ContrastiveConf" in (results / "report.md").read_text()
    header = (results / "conditions.csv").read_text().splitlines()[0].split(",")
    assert "images_undefined_lrp" in header


def test_build_report_includes_the_activation_monitors_when_scored(tmp_path, monkeypatch):
    from differential_uncertainty.baselines import pipeline
    settings = _tiny_run(tmp_path, monkeypatch)
    folder = settings.output / "test_activation"
    folder.mkdir()
    rng = np.random.default_rng(8)
    for path in sorted((settings.output / "test").glob("*.npz")):
        np.savez(folder / path.name, hashemi_decoder=rng.uniform(0, 1, 96), hashemi_encoder=rng.uniform(0, 1, 96),
                 cdf_backbone=rng.uniform(0, 50, 96))
    pipeline.run_phase("report", settings)

    results = settings.output / "results"
    summary = json.loads((results / "summary.json").read_text())
    assert summary["activation_monitors_included"] is True
    assert summary["hashemi_k"] == 2.0 and summary["cdf_bins"] == 1000
    with (results / "harm.csv").open() as handle:
        methods = {row["method"] for row in csv.DictReader(handle)}
    assert {"hashemi", "hashemi_enc", "cdf"} <= methods
    text = (results / "report.md").read_text()
    assert "Hashemi et al., decoder queries" in text and "Activation CDFs (Becker et al., ICPR 2026)" in text


def test_method_scores_add_the_activation_monitors_only_when_given():
    test = {"saod_top3": np.zeros((2, 96)), "saod_min": np.zeros((2, 96)), "conf_pos": np.ones((2, 96)),
            "conf_neg": np.zeros((2, 96)), "knn": np.zeros((2, 96, 200))}
    activation = {"hashemi_decoder": np.full((2, 96), 0.1), "hashemi_encoder": np.full((2, 96), 0.2),
                  "cdf_backbone": np.full((2, 96), 3.0)}
    scores = report.method_scores(test, None, np.ones(2), activation=activation)
    assert (scores["hashemi"][0, 0], scores["hashemi_enc"][0, 0], scores["cdf"][0, 0]) == (0.1, 0.2, 3.0)
    assert "hashemi" not in report.method_scores(test, None, np.ones(2))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_baselines_report.py -q`
Expected: 3 failed:
- the end-to-end test, with `KeyError: 'activation_monitors_included'`;
- the activation-monitor report test, with a `KeyError` or missing methods;
- `test_method_scores_...`, with `TypeError: ... unexpected keyword argument 'activation'`.

The other report tests pass.

- [ ] **Step 3: Write the implementation**

In `differential_uncertainty/baselines/report.py`:

1. Next to the other intra-package imports at the top, add:

```python
from .activation_cdf import BINS as CDF_BINS
from .hashemi import K as HASHEMI_K
```

2. Replace the `METHODS` and `LABELS` definitions with:

```python
METHODS = ("saod_top3", "saod_min", "contrastive", "knn", "discopatch", "hashemi", "hashemi_enc", "cdf")
LABELS = {"saod_top3": "SAOD, mean of top 3", "saod_min": "SAOD, min (1 − max confidence)",
          "contrastive": "ContrastiveConf", "knn": "kNN (k = 100)", "discopatch": "DisCoPatch",
          "hashemi": "Hashemi et al., decoder queries",
          "hashemi_enc": "Hashemi et al., encoder maps (sensitivity)",
          "cdf": "Activation CDFs (Becker et al., ICPR 2026)"}
ACTIVATION_ARRAYS = ("hashemi_decoder", "hashemi_encoder", "cdf_backbone")
```

3. Replace `method_scores` with:

```python
def method_scores(test: dict, dcp, per_image_lambda, k: int = KNN_K, activation=None) -> dict:
    scores = {"saod_top3": test["saod_top3"], "saod_min": test["saod_min"],
              "contrastive": -(test["conf_pos"] - np.asarray(per_image_lambda)[:, None] * test["conf_neg"]),
              "knn": test["knn"][:, :, k - 1]}
    if dcp is not None:
        scores["discopatch"] = dcp
    if activation is not None:
        scores["hashemi"] = activation["hashemi_decoder"]
        scores["hashemi_enc"] = activation["hashemi_encoder"]
        scores["cdf"] = activation["cdf_backbone"]
    return scores
```

4. In `build_report`, replace

```python
    dcp = _stack(dcp_folder, names, ("dcp",))["dcp"] if dcp_folder.exists() else None
```

with

```python
    dcp = _stack(dcp_folder, names, ("dcp",))["dcp"] if dcp_folder.exists() else None
    activation_folder = settings.output / "test_activation"
    activation = _stack(activation_folder, names, ACTIVATION_ARRAYS) if activation_folder.exists() else None
```

and replace

```python
    ordered = {m: v for m, v in method_scores(test, dcp, per_image_lambda).items()}
```

with

```python
    ordered = {m: v for m, v in method_scores(test, dcp, per_image_lambda, activation=activation).items()}
```

5. In the `summary = {...}` dict of `build_report`, replace the line
`"bootstrap_samples": BOOTSTRAP_SAMPLES, "discopatch_included": dcp is not None,` with

```python
        "bootstrap_samples": BOOTSTRAP_SAMPLES, "discopatch_included": dcp is not None,
        "activation_monitors_included": activation is not None, "hashemi_k": HASHEMI_K, "cdf_bins": CDF_BINS,
```

- [ ] **Step 4: Run the tests, then the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_baselines_report.py -q`
Expected: all pass.

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty -q`
Expected: all pass. The previous 203 plus the 21 added in Tasks 1–5 make 224 passed, with 2 CUDA tests skipped.

- [ ] **Step 5: Commit**

```bash
git add differential_uncertainty/baselines/report.py tests/differential_uncertainty/test_baselines_report.py
git commit -m "feat: report both activation monitors"
```

---

### Task 6: Run on the GPU and publish the numbers

**Files:**
- Create (outputs): `runs/coco-baselines/hashemi/*`, `runs/coco-baselines/cdf/*`, `runs/coco-baselines/test_activation/*` (git-ignored)
- Modify: `docs/results/coco-baselines/*` (copied from `runs/coco-baselines/results/`), `docs/coco-baseline-numbers.md`
- Create: `docs/results/coco-baselines/hashemi_fit.json`, `docs/results/coco-baselines/cdf_fit.json`, `docs/results/coco-baselines/activation_fits.json`

**Interfaces:**
- Consumes: the CLI phases from Task 4 and the report from Task 5.
- Produces: the published numbers.

Shared values (the session's worktree guard refuses `$VAR` command lines, so type the literal paths). The worktree is `/home/yuchen/YuchenZ/UE/philip_sa/.worktrees/hashemi-baseline`. The output folder is `OUT = /home/yuchen/YuchenZ/UE/philip_sa/runs/coco-baselines`. The data arguments are:

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

Send `explore` a message via `SendMessage`: about 4 h at roughly 8 GB for the fits and the scoring pass, then about 10 min with the GPU to ourselves for timing. Wait for the answer and start nothing before it. If less than 8 GB is free, wait for their announced pause.

- [ ] **Step 3: Fit both monitors (about 5–10 min each)**

Run `bash .superpowers/sdd/2026-09-29-hashemi-baseline/launch.sh hashemi-fit 6` and wait for the process to exit. Then do the same with `cdf-fit`.
Check with:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -c "
import numpy as np
h = np.load('/home/yuchen/YuchenZ/UE/philip_sa/runs/coco-baselines/hashemi/intervals.npz')
c = np.load('/home/yuchen/YuchenZ/UE/philip_sa/runs/coco-baselines/cdf/reference.npz')
print(int(h['images']), h['decoder_mean'].shape, h['encoder_s8_mean'].shape, h['encoder_s16_mean'].shape, h['encoder_s32_mean'].shape, float((h['decoder_std'] == 0).mean()))
print(int(c['images']), [c[f'{s}_cdf'].shape for s in ('C1', 'C2', 'C3', 'C4', 'C5')], float(np.abs(c['C5_cdf'][:, -1] - 1).max()))
"
```

Expected:
- first line: `118287 (300, 256) (256, 80, 80) (256, 40, 40) (256, 20, 20)`, then a share close to 0.0;
- second line: `118287 [(64, 1000), (64, 1000), (128, 1000), (256, 1000), (512, 1000)]`, then a value close to 0.0.

- [ ] **Step 4: Smoke run on 20 images (about 5 min)**

```bash
mkdir -p /home/yuchen/YuchenZ/UE/philip_sa/runs/coco-baselines-smoke/hashemi /home/yuchen/YuchenZ/UE/philip_sa/runs/coco-baselines-smoke/cdf
cp /home/yuchen/YuchenZ/UE/philip_sa/runs/coco-baselines/hashemi/intervals.npz /home/yuchen/YuchenZ/UE/philip_sa/runs/coco-baselines/hashemi/fit.json /home/yuchen/YuchenZ/UE/philip_sa/runs/coco-baselines-smoke/hashemi/
cp /home/yuchen/YuchenZ/UE/philip_sa/runs/coco-baselines/cdf/reference.npz /home/yuchen/YuchenZ/UE/philip_sa/runs/coco-baselines/cdf/fit.json /home/yuchen/YuchenZ/UE/philip_sa/runs/coco-baselines-smoke/cdf/
/home/yuchen/miniconda3/envs/UE/bin/python -m differential_uncertainty baselines-coco --phase activation-scores --limit 20 --workers 9 --output /home/yuchen/YuchenZ/UE/philip_sa/runs/coco-baselines-smoke --checkpoint /home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth --coco-train-images /home/yuchen/YuchenZ/Datasets/coco/train2017 --coco-val-images /home/yuchen/YuchenZ/Datasets/coco/val2017 --coco-annotations /home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json --discopatch-root /home/yuchen/YuchenZ/UE/DisCoPatch
/home/yuchen/miniconda3/envs/UE/bin/python -m differential_uncertainty baselines-coco --phase report --limit 20 --output /home/yuchen/YuchenZ/UE/philip_sa/runs/coco-baselines-smoke --checkpoint /home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth --coco-train-images /home/yuchen/YuchenZ/Datasets/coco/train2017 --coco-val-images /home/yuchen/YuchenZ/Datasets/coco/val2017 --coco-annotations /home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json --discopatch-root /home/yuchen/YuchenZ/UE/DisCoPatch
```

Expected:
- `runs/coco-baselines-smoke/results/report.md` has finite rows for "Hashemi et al., decoder queries", "Hashemi et al., encoder maps (sensitivity)" and "Activation CDFs (Becker et al., ICPR 2026)".
- In `conditions.csv`, the clean `mean_hashemi` is well below 0.5, and `mean_cdf` rises with severity for noise families.

These numbers are not reported.

- [ ] **Step 5: Full scoring pass (about 3 h; the corruptions are rebuilt on the CPU)**

Run `bash .superpowers/sdd/2026-09-29-hashemi-baseline/launch.sh activation-scores 14` and wait for the process to exit.
Expected:
- `OUT/test_activation/` holds 5,000 `.npz` files plus `fits.json`;
- the log ends with `[activation] 5000/5000`;
- there is no "corruptions differ", "changed since" or "non-finite" error.

If the run is interrupted, rerun the same command; it resumes.

- [ ] **Step 6: Runtime, with nothing else on the GPU (about 10 min)**

Confirm with `explore` that the card is free. Then run in the foreground:
`/home/yuchen/miniconda3/envs/UE/bin/python -m differential_uncertainty baselines-coco --phase timing --output /home/yuchen/YuchenZ/UE/philip_sa/runs/coco-baselines --checkpoint /home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth --coco-train-images /home/yuchen/YuchenZ/Datasets/coco/train2017 --coco-val-images /home/yuchen/YuchenZ/Datasets/coco/val2017 --coco-annotations /home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json --discopatch-root /home/yuchen/YuchenZ/UE/DisCoPatch`
Expected: `OUT/timing.json` contains `detector_plus_hashemi_ms` and `detector_plus_cdf_ms`, plus fresh values for every earlier key. The published runtime table uses this new file.

- [ ] **Step 7: Report (about 1 h)**

Run `bash .superpowers/sdd/2026-09-29-hashemi-baseline/launch.sh report 9` and wait for the process to exit.
Expected: `OUT/results/summary.json` has `"activation_monitors_included": true`. Then check that the earlier baselines did not move (same bootstrap seed and draws):

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

Copy every CSV, `summary.json`, `report.md` and `timing.json` from `OUT/results/` and `OUT/` into `docs/results/coco-baselines/`. Then copy:
- `OUT/hashemi/fit.json` to `docs/results/coco-baselines/hashemi_fit.json`;
- `OUT/cdf/fit.json` to `docs/results/coco-baselines/cdf_fit.json`;
- `OUT/test_activation/fits.json` to `docs/results/coco-baselines/activation_fits.json`.

Then edit `docs/coco-baseline-numbers.md`. Take every number from the new `report.md` and CSVs:
- **"The short version":** one bullet each for Hashemi et al. and the activation CDFs. Say where each lands on separation (common and extra AUROC) and on harm (within-condition ρ and AURC), compared with the four earlier baselines.
- **Setup, baseline table:**
  - a row "Hashemi et al. (FM 2023)": score = share of the last decoder layer's 300 × 256 neurons outside μ ± 2σ, fitted on 118,287 clean train images; mention the encoder-map sensitivity row;
  - a row "Activation CDFs (Becker et al., ICPR 2026), reproduced": score = sum over the 1,024 channels of backbone stages C1–C5 of the EMD between the image's activation CDF and the training CDF; 1,000 bins; ranges ±20%.
- **Separation tables (common and extra):** three rows from `report.md`.
- **Per-family table:** two columns, "Hashemi (dec.)" and "Act. CDFs", with AUROC at severity 3 / 5 from `separation.csv` (methods `hashemi` and `cdf`). Update the bold best-at-severity-5 marks.
- **Harm table, per-severity ρ table and AURC severity-pool table:** rows or columns for all three, from `harm.csv` and `aurc_pools.csv`.
- **Differences table:** add "Act. CDFs − DisCoPatch", "Act. CDFs − SAOD min", "Hashemi (dec.) − SAOD min" and "Hashemi (dec.) − kNN", from `differences.csv`.
- **Runtime table:** the two new rows and the refreshed earlier rows.
- **Deviations:** new items.
  - Hashemi et al.:
    - the monitored layer (option A; the paper used PolyYOLO's last batch-norm or leaky-ReLU layer);
    - μ and σ from 118,287 images, not 500, as population values;
    - the raw score, not the conformal p-value;
    - option B reported as a sensitivity row.
  - The ICPR 2026 monitor:
    - reproduced on RT-DETRv2-R18's ResNet-18 backbone, stages C1–C5, the analogue of their Faster R-CNN "CDFs backbone";
    - no code was released, so it was reimplemented from the paper;
    - the "20% margin" is read as 20% of the training span on each side;
    - EMD in units of each channel's range, summed over channels;
    - no final z-scoring, because it leaves a single score's AUROC unchanged;
    - standard severities 1–5, not their 10-level extension.
  - **Comparison with their numbers, in words only:**
    - their RT-DETR-l "CDFs backbone" reached 76.4 / 86.4 AUROC at *their* severities 3 / 5, pooled over the 19 corruptions;
    - give our mean AUROC over all 19 families at the standard severities 3 and 5, from `aggregates.csv`, group `all`;
    - say plainly that the severity scales and detectors differ.
- **Files:** list `hashemi_fit.json`, `cdf_fit.json` and `activation_fits.json`.

- [ ] **Step 9: Commit**

```bash
git add docs/results/coco-baselines docs/coco-baseline-numbers.md
git commit -m "docs: add Hashemi et al. and the ICPR 2026 activation monitor to the COCO baseline numbers"
```

---

## Expected timeline

| Step | Time | Needs |
| --- | --- | --- |
| Tasks 1–5 (code) | about 1.5 h | CPU only |
| `hashemi-fit`, `cdf-fit` | 5–10 min each | GPU, about 8 GB |
| Smoke run | about 5 min | GPU |
| `activation-scores` | about 3 h | GPU about 5 GB, 14 CPU workers |
| Timing | about 10 min | GPU alone |
| Report | about 1 h | CPU |
