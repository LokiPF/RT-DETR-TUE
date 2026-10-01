# Conv-TU Pilot Follow-up: Channel Statistics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer "how would the performance change if each channel were compared with its own training average?", and test two dev-log variants on the same 200 images, all next to the pilot's rows and the baselines. The variants are the per-channel top-1% means and 99th percentiles, and the 4 × 4-grid channel means.

**Architecture:**
- **New module** `differential_uncertainty/convtu/channels.py`: the four per-channel statistics, the own-average comparison, and an offline scorer that compares every statistic both ways.
- **One GPU phase,** `convtu-channels`. One backbone pass stores the statistics of the four pilot conv inputs for the pilot's 2,000 bank images, its 500 z-statistics images, and every condition of the 200 pilot images.
- **One CPU phase,** `convtu-channels-report`. It scores the stored statistics offline and writes the pilot report with eight extra rows. `build_pilot_report` gains optional extra rows.

**Tech Stack:** Python 3.11, PyTorch 2.11, NumPy, pytest; the existing `differential_uncertainty.convtu` and `baselines` packages.

**Spec:** `docs/conv-tu-pilot-design.md`, section "Follow-up 1 (1 October 2026): channel statistics", and `docs/dev-log.md`, entry 2026-10-01.

**Before you start:** work in the existing worktree `.worktrees/convtu-pilot` (branch `convtu-pilot`). Commit the design-note follow-up section and this plan there before Task 1. Commit messages end with:
```
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU
```

## Global Constraints

- **Same detector, layers and images as the pilot:**
  - the frozen RT-DETRv2-R18 checkpoint at 640 × 640;
  - the inputs of `backbone.res_layers[s].blocks[1].branch2a.conv` for s = 0…3;
  - the pilot's train splits (`train_splits(…)["bank"]`, 2,000 images; `["zstats"]`, 500 images; seed 44);
  - `pilot_images(settings)`, the first 200 images of the seed-44 order, with all 96 conditions and the corruption digests checked against `runs/coco-baselines/test/*.npz`.
- **Statistics are taken on |x|:**
  - means;
  - top: the mean of the k largest values per channel, with k = max(1, round(0.01 · H · W));
  - p99: the k-th largest value;
  - grid: 4 × 4 adaptive average pooling, channel-major.
- **Own-average score:** per layer, mean over dimensions of |v − μ| / σ, with μ and the population σ over the 2,000 bank images, and σ floored at 0.01 × the median of the layer's positive σ.
- **kNN score:** `features.knn_scores`: mean Euclidean distance to the 5 nearest bank rows, not normalized.
- **Layer combination:** z-score per layer with the 500 z-statistics images (`stage_zstats`), then sum over the four layers (`zscored_sum`).
- **Protocol unchanged:** `run_config.json` must not change; nothing is added to `Settings.experiment()`.
- **GPU:** shared with `explore` and `hunk-0927`. Agree a window through SendMessage first. The phase caps its allocator with `_cap_gpu_memory`. Tests never use the GPU (`CUDA_VISIBLE_DEVICES=`).
- **Python:** `/home/yuchen/miniconda3/envs/UE/bin/python`.
- **Suite:** must stay green; it stood at 264 passed and 2 skipped before this plan.

## Review Focus

1. **A channel that is dead on every clean image but active under a corruption.**
   - Expected: its σ is floored, so it adds a large but finite term instead of dividing by zero.
   - Pinned by `test_fit_own_average_floors_dead_dimensions` (Task 1).
2. **A layer where more than half the dimensions are dead.**
   - Expected: the floor uses the median of the positive spreads, and only an all-dead layer is refused.
   - Pinned by the same test.
3. **Corruptions regenerated differently from the detector pass.**
   - Expected: refusal.
   - Pinned by `test_channels_phase_detects_changed_corruptions` (Task 2).
4. **A crash in the middle of the channels phase.**
   - Expected: finished images are kept, and a rerun continues with the rest.
   - Pinned by `test_channels_phase_stores_every_statistic_and_resumes` (Task 2).
5. **The kNN-on-means row drifting from the pilot's channel-means row.**
   - Expected: it reproduces the pilot within 0.001 AUROC; otherwise the stored statistics or the scoring differ.
   - Checked in Task 4, Step 4.

---

### Task 1: Channel statistics and the own-average comparison

**Files:**
- Create: `differential_uncertainty/convtu/channels.py`
- Test: `tests/differential_uncertainty/test_convtu_channels.py`

**Interfaces:**
- Consumes: nothing new.
- Produces, in `differential_uncertainty.convtu.channels`:
  - `STATISTICS = ("means", "top", "p99", "grid")`, `COMPARISONS = ("knn", "own")`, `TOP_FRACTION = 0.01`, `GRID = 4`, `STD_FLOOR = 0.01`
  - `channel_statistics(x: Tensor) -> dict[str, np.ndarray]`. `x` has shape `(N, C, H, W)`. The output has keys `means`, `top` and `p99`, each `(N, C)`, and `grid`, `(N, 16·C)`, all float32.
  - `fit_own_average(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]`: per-dimension mean, and the floored population std.
  - `own_average_scores(values: np.ndarray, mean, std) -> np.ndarray`: shape `(rows,)`.

- [ ] **Step 1: Write the failing tests**

`tests/differential_uncertainty/test_convtu_channels.py`:

```python
import numpy as np
import pytest
import torch

from differential_uncertainty.convtu import channels


def test_channel_statistics_on_an_8_by_8_map():
    x = -torch.arange(2 * 8 * 8, dtype=torch.float32).reshape(1, 2, 8, 8)  # negative: statistics use |x|
    stats = channels.channel_statistics(x)
    values = np.abs(x.numpy())[0]
    np.testing.assert_allclose(stats["means"][0], values.mean(axis=(1, 2)))
    # 64 cells: k = max(1, round(0.64)) = 1, so top and p99 are the maximum
    np.testing.assert_allclose(stats["top"][0], values.max(axis=(1, 2)))
    np.testing.assert_allclose(stats["p99"][0], values.max(axis=(1, 2)))
    blocks = values.reshape(2, 4, 2, 4, 2).mean(axis=(2, 4)).reshape(-1)  # 2 x 2 blocks on a 4 x 4 grid
    np.testing.assert_allclose(stats["grid"][0], blocks)
    assert {key: value.shape for key, value in stats.items()} == {
        "means": (1, 2), "top": (1, 2), "p99": (1, 2), "grid": (1, 32)}


def test_top_and_p99_use_the_largest_one_percent():
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(3, 4, 20, 20, generator=generator)  # 400 cells: k = 4
    stats = channels.channel_statistics(x)
    ordered = np.sort(x.numpy().reshape(3, 4, -1), axis=2)[:, :, ::-1]
    np.testing.assert_allclose(stats["top"], ordered[:, :, :4].mean(axis=2), rtol=1e-6)
    np.testing.assert_allclose(stats["p99"], ordered[:, :, 3], rtol=1e-6)


def test_fit_own_average_floors_dead_dimensions():
    reference = np.array([[1.0, 5.0, 0.0, 0.0, 0.0],
                          [3.0, 9.0, 0.0, 0.0, 0.0],
                          [2.0, 7.0, 0.0, 0.0, 0.0]])  # three of five dimensions are dead
    mean, std = channels.fit_own_average(reference)
    np.testing.assert_allclose(mean, [2.0, 7.0, 0.0, 0.0, 0.0])
    positive = np.array([np.std([1.0, 3.0, 2.0]), np.std([5.0, 9.0, 7.0])])
    floor = 0.01 * np.median(positive)
    np.testing.assert_allclose(std, [positive[0], positive[1], floor, floor, floor])
    with pytest.raises(ValueError, match="no spread"):
        channels.fit_own_average(np.zeros((4, 3)))


def test_own_average_scores_are_mean_absolute_z():
    mean, std = np.array([1.0, 2.0]), np.array([0.5, 4.0])
    values = np.array([[1.0, 2.0], [2.0, -2.0]])
    np.testing.assert_allclose(channels.own_average_scores(values, mean, std), [0.0, (2.0 + 1.0) / 2])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_convtu_channels.py -q`
Expected: collection error `ImportError: cannot import name 'channels' from 'differential_uncertainty.convtu'`.

- [ ] **Step 3: Write the implementation**

`differential_uncertainty/convtu/channels.py`:

```python
"""Per-channel statistics of the four pilot conv inputs, and the two ways of comparing them with clean images.

means: each channel's mean |x|. top: the mean of each channel's largest 1% of |x|. p99: the smallest of those,
the nearest-rank 99th percentile. grid: each channel's mean |x| on a 4 x 4 grid of the map. kNN compares an
image's vector with its nearest clean images; "own" compares every dimension with its own clean average, in
units of its clean spread, the way the activation-distribution monitor compares every channel with itself.
"""
from __future__ import annotations

import numpy as np
import torch

STATISTICS = ("means", "top", "p99", "grid")
COMPARISONS = ("knn", "own")
TOP_FRACTION = 0.01
GRID = 4
STD_FLOOR = 0.01  # every dimension's spread is at least 1% of the median positive spread of its layer


@torch.inference_mode()
def channel_statistics(x: torch.Tensor) -> dict:
    """For a batch of conv inputs (N, C, H, W): every statistic as a float32 (N, dim) array."""
    if x.ndim != 4:
        raise ValueError("expected a batch of shape (N, C, H, W)")
    x_abs = x.abs().float()
    n, channel_count, height, width = x_abs.shape
    k = max(1, round(TOP_FRACTION * height * width))
    top = torch.topk(x_abs.reshape(n, channel_count, -1), k, dim=2).values
    grid = torch.nn.functional.adaptive_avg_pool2d(x_abs, GRID).reshape(n, channel_count * GRID * GRID)
    out = {"means": x_abs.mean(dim=(2, 3)), "top": top.mean(dim=2), "p99": top[:, :, -1], "grid": grid}
    return {key: value.cpu().numpy().astype(np.float32) for key, value in out.items()}


def fit_own_average(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension clean mean and population spread; the spread is floored so dead dimensions stay finite."""
    reference = np.asarray(reference, dtype=np.float64)
    if reference.ndim != 2 or reference.shape[0] < 2:
        raise ValueError("need at least two clean rows of shape (rows, dim)")
    mean, std = reference.mean(axis=0), reference.std(axis=0)
    positive = std[std > 0]
    if positive.size == 0:
        raise ValueError("the clean rows have no spread")
    return mean, np.maximum(std, STD_FLOOR * float(np.median(positive)))


def own_average_scores(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Mean over dimensions of |value - clean mean| / clean spread: one score per row."""
    return (np.abs(np.asarray(values, dtype=np.float64) - mean) / std).mean(axis=1)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_convtu_channels.py -q`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add differential_uncertainty/convtu/channels.py tests/differential_uncertainty/test_convtu_channels.py
git commit -m "feat: per-channel statistics of the pilot convs and the own-average comparison"
```

---

### Task 2: GPU phase that stores the channel statistics

**Files:**
- Modify: `differential_uncertainty/convtu/pipeline.py`. Add an import, constants and functions, and extend `PHASES`.
- Modify: `differential_uncertainty/baselines/pipeline.py`: `CONVTU_PHASES` gains `"convtu-channels"`.
- Modify: `differential_uncertainty/cli.py`: the choices gain `"convtu-channels"`.
- Test: `tests/differential_uncertainty/test_convtu_pipeline.py` (append); `tests/differential_uncertainty/test_cli.py`, where the tuple in `test_cli_accepts_the_convtu_phases` gains `"convtu-channels"`.

**Interfaces:**
- Consumes:
  - `channels.channel_statistics`, `channels.STATISTICS` (Task 1);
  - from `convtu.pipeline`: `_conv_inputs`, `_clean_batches`, `_cap_gpu_memory`, `pilot_images`, `IMAGE_SIZE`, `_folder`;
  - from `baselines.pipeline`: `TEST_KEYS`, `_load_npz`, `_valid_existing`, `_variant_stream`, `_atomic_npz`, `_progress`;
  - `features.layer_specs`.
- Produces, in `differential_uncertainty.convtu.pipeline`:
  - `CHANNELS_FOLDER = "test_convtu_channels"`
  - `CHANNEL_KEYS`: `f"{statistic}_s{stage}"` for each statistic in `STATISTICS` and stage 1–4
  - `channels_bank_path(settings)`, `channels_zstats_path(settings) -> Path`
  - `image_channel_statistics(taps, arrays, batch_size) -> dict[str, np.ndarray]`: shape `(96, dim)` per key
  - `phase_channels`, and `PHASES["convtu-channels"]`
- Output formats:
  - `convtu/channels_bank.npz`: `CHANNEL_KEYS`, each `(2000, dim)`;
  - `convtu/channels_zstats.npz`: `CHANNEL_KEYS`, each `(500, dim)`;
  - `test_convtu_channels/{stem}.npz`: `CHANNEL_KEYS`, each `(96, dim)`.

  The dims per stage are C for means, top and p99, and 16·C for grid.

- [ ] **Step 1: Write the failing tests** (append to `tests/differential_uncertainty/test_convtu_pipeline.py`)

```python
def _channels(settings):
    for phase in ("test", "convtu-channels"):
        baselines.run_phase(phase, settings)


def test_channels_phase_stores_every_statistic_and_resumes(small, detector, monkeypatch):
    monkeypatch.setattr(convtu, "PILOT_IMAGES", 2)
    capped = []
    monkeypatch.setattr(convtu, "_cap_gpu_memory", lambda device: capped.append(device))
    _channels(small)
    assert capped == ["cpu"]
    with np.load(convtu.channels_bank_path(small)) as bank:
        assert bank["means_s1"].shape == (6, 2) and bank["grid_s1"].shape == (6, 32)
        assert bank["top_s4"].shape == (6, 5) and bank["p99_s4"].shape == (6, 5)
    with np.load(convtu.channels_zstats_path(small)) as zstats:
        assert zstats["means_s2"].shape == (3, 3)
    files = sorted((small.output / convtu.CHANNELS_FOLDER).glob("*.npz"))
    assert len(files) == 2
    with np.load(files[0]) as stats:
        assert set(stats.files) == set(convtu.CHANNEL_KEYS)
        assert stats["means_s4"].shape == (96, 5) and stats["grid_s4"].shape == (96, 80)
        assert np.isfinite(stats["grid_s1"]).all()
    stamp = files[0].stat().st_mtime_ns
    baselines.run_phase("convtu-channels", small)
    assert files[0].stat().st_mtime_ns == stamp


def test_channels_phase_detects_changed_corruptions(small, detector, monkeypatch):
    monkeypatch.setattr(convtu, "PILOT_IMAGES", 2)
    _channels(small)
    stem = sorted((small.output / convtu.CHANNELS_FOLDER).glob("*.npz"))[0].stem
    (small.output / convtu.CHANNELS_FOLDER / f"{stem}.npz").unlink()
    stored = dict(np.load(small.output / "test" / f"{stem}.npz"))
    stored["digests"] = stored["digests"].copy()
    stored["digests"][7] = "0" * 16
    np.savez(small.output / "test" / f"{stem}.npz", **stored)
    with pytest.raises(ValueError, match="corruptions differ"):
        baselines.run_phase("convtu-channels", small)
```

In `tests/differential_uncertainty/test_cli.py`, the tuple becomes:

```python
    phases = ("convtu-calibrate", "convtu-bank", "convtu-zstats", "convtu-scores", "convtu-report", "convtu-channels")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_convtu_pipeline.py tests/differential_uncertainty/test_cli.py -q -k "channels or convtu_phases"`
Expected:
- the two new pipeline tests fail with `ValueError: unknown phase 'convtu-channels'`;
- the CLI test fails with exit code 2.

- [ ] **Step 3: Write the implementation**

In `differential_uncertainty/convtu/pipeline.py`, add `from .channels import STATISTICS, channel_statistics` to the imports. Add below `SCORE_KEYS`:

```python
CHANNELS_FOLDER = "test_convtu_channels"
CHANNEL_KEYS = tuple(f"{statistic}_s{stage}" for statistic in STATISTICS for stage in range(1, 5))
```

Append before `def phase_report`:

```python
def channels_bank_path(settings: Settings):
    return _folder(settings) / "channels_bank.npz"


def channels_zstats_path(settings: Settings):
    return _folder(settings) / "channels_zstats.npz"


def _batch_channel_statistics(taps, batch) -> dict:
    inputs = taps(batch.to(taps.kernels[0].device))
    out = {}
    for layer, spec in enumerate(layer_specs(inputs, taps.kernels)):
        for statistic, values in channel_statistics(inputs[layer]).items():
            out[f"{statistic}_{spec.name}"] = values
    return out


def _concatenated(parts: list) -> dict:
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def image_channel_statistics(taps, arrays, batch_size) -> dict:
    """Every channel statistic of every pilot layer for all conditions of one image, (96, dim) per key."""
    parts = []
    for start in range(0, len(arrays), batch_size):
        batch = torch.stack([prepare_image(Image.fromarray(a), IMAGE_SIZE) for a in arrays[start:start + batch_size]])
        parts.append(_batch_channel_statistics(taps, batch))
    out = _concatenated(parts)
    if not all(np.isfinite(value).all() for value in out.values()):
        raise ValueError("channel statistics are not finite")
    return out


def phase_channels(settings: Settings) -> None:
    """Channel statistics of the pilot layers: the bank and z-statistics images, then every pilot variant."""
    folder = settings.output / CHANNELS_FOLDER
    pending = [p for p in pilot_images(settings) if not _valid_existing(folder / f"{p.stem}.npz", CHANNEL_KEYS)]
    absent = [p.name for p in pending if not (settings.output / "test" / f"{p.stem}.npz").exists()]
    if absent:
        raise ValueError(f"run the test phase first: {len(absent)} detector results are missing, e.g. {absent[0]}")
    clean = {"bank": channels_bank_path(settings), "zstats": channels_zstats_path(settings)}
    if not pending and all(path.exists() for path in clean.values()):
        return
    _cap_gpu_memory(settings.device)
    started = time.time()
    with _conv_inputs(settings) as taps:
        for split, path in clean.items():
            if not path.exists():
                _atomic_npz(path, **_concatenated(
                    [_batch_channel_statistics(taps, batch) for batch in _clean_batches(settings, split)]))
        for done, (name, arrays) in enumerate(_variant_stream(settings, pending), start=1):
            stem = Path(name).stem
            stored = _load_npz(settings.output / "test" / f"{stem}.npz", TEST_KEYS)["digests"]
            if list(stored) != [protocol.digest(a) for a in arrays]:
                raise ValueError(f"corruptions differ from the detector pass for {name}")
            _atomic_npz(folder / f"{stem}.npz", **image_channel_statistics(taps, arrays, settings.batch_size))
            if done % 10 == 0:
                _progress("convtu-channels", done, len(pending), started)
```

`PHASES` gains `"convtu-channels": phase_channels`. In `baselines/pipeline.py`, `CONVTU_PHASES` gains `"convtu-channels"`. In `cli.py`, the choices gain `"convtu-channels"`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_convtu_pipeline.py tests/differential_uncertainty/test_cli.py -q`
Expected: all pass. The new tests build the corruptions of two small images, so they take tens of seconds.

- [ ] **Step 5: Commit**

```bash
git add differential_uncertainty/convtu/pipeline.py differential_uncertainty/baselines/pipeline.py differential_uncertainty/cli.py tests/differential_uncertainty/test_convtu_pipeline.py tests/differential_uncertainty/test_cli.py
git commit -m "feat: store per-channel statistics of the pilot convs for the clean splits and every pilot variant"
```

---

### Task 3: Offline scoring and the channel-statistics report

**Files:**
- Modify: `differential_uncertainty/convtu/channels.py`: append the scorer and the labels.
- Modify: `differential_uncertainty/convtu/report.py`: `depth_rows` takes a dict of per-layer scores; `build_pilot_report` takes optional extra rows, an output folder name and a title.
- Modify: `differential_uncertainty/convtu/pipeline.py`: add `phase_channels_report`, plus a `PHASES` entry.
- Modify: `differential_uncertainty/baselines/pipeline.py` and `differential_uncertainty/cli.py`: add `"convtu-channels-report"`.
- Test:
  - `tests/differential_uncertainty/test_convtu_channels.py` (append);
  - `tests/differential_uncertainty/test_convtu_report.py`: refactor its fixture into a helper, then append;
  - `tests/differential_uncertainty/test_cli.py`, where the tuple gains `"convtu-channels-report"`.

**Interfaces:**
- Consumes:
  - `features.knn_scores`;
  - `activation_cdf.stage_zstats`, `zscored_sum`;
  - from `baselines.report`: `_stack`;
  - from Task 2: `CHANNEL_KEYS`, `CHANNELS_FOLDER`, `channels_bank_path`, `channels_zstats_path`, `pilot_images`.
- Produces:
  - In `differential_uncertainty.convtu.channels`:
    - `method_name(statistic, comparison) -> str`, for example `"ch_means_own"`;
    - `LABELS: dict[str, str]`;
    - `channel_method_scores(bank, zstats, test, layers) -> tuple[dict, dict]`. The first dict maps method name to z-summed scores of shape `(images, conditions)`. The second maps method name to per-layer scores of shape `(images, conditions, layers)`.
  - In `differential_uncertainty.convtu.report`:
    - `depth_rows(layer_scores: dict[str, np.ndarray], lrp) -> list[dict]`
    - `build_pilot_report(settings, names, extra=None, extra_layers=None, extra_labels=None, folder_name="results_convtu", title="# Conv TU pilot") -> None`
  - The phase `convtu-channels-report`, which writes `results_convtu_channels/`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/differential_uncertainty/test_convtu_channels.py`:

```python
def _clean_and_test(rng, layers=("s1", "s2"), dims=(3, 4)):
    bank, zstats, test = {}, {}, {}
    for statistic in channels.STATISTICS:
        for layer, dim in zip(layers, dims):
            width = 16 * dim if statistic == "grid" else dim
            key = f"{statistic}_{layer}"
            bank[key] = rng.normal(size=(40, width))
            zstats[key] = rng.normal(size=(12, width))
            test[key] = np.stack([zstats[key], zstats[key] + 10.0])  # (2 images, 12 conditions, width)
    return bank, zstats, test


def test_channel_method_scores_compare_every_statistic_both_ways():
    bank, zstats, test = _clean_and_test(np.random.default_rng(0))
    summed, per_layer = channels.channel_method_scores(bank, zstats, test, ("s1", "s2"))
    expected = {channels.method_name(s, c) for s in channels.STATISTICS for c in channels.COMPARISONS}
    assert set(summed) == set(per_layer) == expected and set(channels.LABELS) == expected
    for name in expected:
        assert summed[name].shape == (2, 12) and per_layer[name].shape == (2, 12, 2)
        # image 0 is the z-statistics images themselves, so its layer z-scores average to 0
        assert abs(summed[name][0].mean()) < 1e-9
        # image 1 is shifted far from every clean row, so it scores higher everywhere
        assert (summed[name][1] > summed[name][0].max()).all()


def test_own_comparison_matches_the_per_dimension_formula():
    bank, zstats, test = _clean_and_test(np.random.default_rng(1))
    _, per_layer = channels.channel_method_scores(bank, zstats, test, ("s1", "s2"))
    mean, std = channels.fit_own_average(bank["top_s2"])
    np.testing.assert_allclose(per_layer["ch_top_own"][1, :, 1],
                               channels.own_average_scores(test["top_s2"][1], mean, std))
```

In `tests/differential_uncertainty/test_convtu_report.py`, move the fake-score setup of the existing test into a helper, and append a new test. The whole file becomes:

```python
import json
from pathlib import Path

import numpy as np

from differential_uncertainty.baselines import pipeline as baselines
from differential_uncertainty.convtu import channels
from differential_uncertainty.convtu import pipeline as convtu
from differential_uncertainty.convtu import report as pilot
from test_baselines_report import _tiny_run


def _fake_pilot(tmp_path, monkeypatch):
    """The tiny run's full report, then fake conv-TU scores and a fake calibration for its 15 images."""
    settings = _tiny_run(tmp_path, monkeypatch)
    baselines.run_phase("report", settings)
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
    return settings


def test_pilot_report_puts_the_fingerprint_next_to_the_baselines(tmp_path, monkeypatch):
    settings = _fake_pilot(tmp_path, monkeypatch)
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


def test_channels_report_adds_eight_rows(tmp_path, monkeypatch):
    settings = _fake_pilot(tmp_path, monkeypatch)
    rng = np.random.default_rng(1)
    widths = {f"{statistic}_s{stage}": (48 if statistic == "grid" else 3)
              for statistic in channels.STATISTICS for stage in range(1, 5)}
    np.savez(convtu.channels_bank_path(settings), **{k: rng.normal(size=(30, w)) for k, w in widths.items()})
    np.savez(convtu.channels_zstats_path(settings), **{k: rng.normal(size=(10, w)) for k, w in widths.items()})
    folder = settings.output / convtu.CHANNELS_FOLDER
    folder.mkdir()
    for path in convtu.pilot_images(settings):
        np.savez(folder / f"{Path(path.name).stem}.npz",
                 **{k: rng.normal(size=(96, w)) + np.linspace(0, 3, 96)[:, None] for k, w in widths.items()})

    baselines.run_phase("convtu-channels-report", settings)

    results = settings.output / "results_convtu_channels"
    text = (results / "report.md").read_text()
    assert text.startswith("# Conv TU pilot: channel statistics")
    assert "Channel means vs own training average" in text and "Conv TU: top 1% of the diagram" in text
    assert len((results / "depth.csv").read_text().splitlines()) == 1 + 16 + 32
    assert "ch_means_own:auroc_common" in (results / "intervals.csv").read_text()
```

In `tests/differential_uncertainty/test_cli.py`, the tuple becomes:

```python
    phases = ("convtu-calibrate", "convtu-bank", "convtu-zstats", "convtu-scores", "convtu-report",
              "convtu-channels", "convtu-channels-report")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_convtu_channels.py tests/differential_uncertainty/test_convtu_report.py tests/differential_uncertainty/test_cli.py -q`
Expected:
- the two new channel tests fail with `AttributeError: module 'differential_uncertainty.convtu.channels' has no attribute 'channel_method_scores'`;
- `test_channels_report_adds_eight_rows` fails with `ValueError: unknown phase 'convtu-channels-report'`;
- the CLI test fails with exit code 2;
- the refactored pilot-report test passes.

- [ ] **Step 3: Write the implementation**

Append to `differential_uncertainty/convtu/channels.py`, and add `from ..baselines.activation_cdf import stage_zstats, zscored_sum` and `from .features import knn_scores` to its imports:

```python
LABELS = {
    "ch_means_knn": "Channel means, kNN (check: the pilot's control)",
    "ch_means_own": "Channel means vs own training average",
    "ch_top_knn": "Per-channel top-1% means, kNN",
    "ch_top_own": "Per-channel top-1% means vs own training average",
    "ch_p99_knn": "Per-channel 99th percentiles, kNN",
    "ch_p99_own": "Per-channel 99th percentiles vs own training average",
    "ch_grid_knn": "4 × 4-grid channel means, kNN",
    "ch_grid_own": "4 × 4-grid channel means vs own training average",
}


def method_name(statistic: str, comparison: str) -> str:
    return f"ch_{statistic}_{comparison}"


def _float32(array) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


def channel_method_scores(bank: dict, zstats: dict, test: dict, layers) -> tuple[dict, dict]:
    """Every statistic compared both ways: z-summed (images, conditions) scores and the per-layer ones.

    bank and zstats map f"{statistic}_{layer}" to clean (rows, dim) arrays; test maps it to
    (images, conditions, dim). Per-layer scores are z-scored with the z-statistics images and summed over
    the layers, as in the pilot.
    """
    summed, per_layer = {}, {}
    for statistic in STATISTICS:
        for comparison in COMPARISONS:
            clean_columns, test_columns = [], []
            for layer in layers:
                key = f"{statistic}_{layer}"
                images, conditions, dim = test[key].shape
                flat = test[key].reshape(images * conditions, dim)
                if comparison == "knn":
                    reference = _float32(bank[key])
                    clean_columns.append(knn_scores(_float32(zstats[key]), reference))
                    test_columns.append(knn_scores(_float32(flat), reference))
                else:
                    mean, std = fit_own_average(bank[key])
                    clean_columns.append(own_average_scores(zstats[key], mean, std))
                    test_columns.append(own_average_scores(flat, mean, std))
            mean, std = stage_zstats(np.stack(clean_columns, axis=1))
            values = np.stack(test_columns, axis=1)
            name = method_name(statistic, comparison)
            per_layer[name] = values.reshape(images, conditions, len(layers))
            summed[name] = zscored_sum(values, mean, std).reshape(images, conditions)
    return summed, per_layer
```

In `differential_uncertainty/convtu/report.py`, replace `depth_rows` with:

```python
def depth_rows(layer_scores: dict, lrp: np.ndarray) -> list[dict]:
    """Per representation and stage: separation and harm tracking of that one layer's score."""
    delta_risk = lrp[:, report.CORRUPTED] - lrp[:, [0]]
    rows = []
    for name, values in layer_scores.items():
        for layer in range(values.shape[2]):
            v = values[:, :, layer]
            rows.append({"representation": name, "stage": layer + 1,
                         "auroc_common": float(metrics.condition_aurocs(v[:, 0], v[:, report.COMMON].T).mean()),
                         "auroc_extra": float(metrics.condition_aurocs(v[:, 0], v[:, report.EXTRA].T).mean()),
                         "rho_within": metrics.mean_within_condition_spearman(
                             v[:, report.CORRUPTED] - v[:, [0]], delta_risk)})
    return rows
```

`build_pilot_report` changes in four places:

1. **Signature:**

   ```python
   def build_pilot_report(settings, names, extra=None, extra_layers=None, extra_labels=None,
                          folder_name="results_convtu", title="# Conv TU pilot") -> None:
   ```

2. **The methods.** After `ordered.update({f"convtu_{rep}": conv[rep] for rep in REPRESENTATIONS})`, replace `scores = {m: ordered[m] for m in METHODS if m in ordered}` with:

   ```python
       ordered.update(extra or {})
       methods = METHODS + tuple(extra or {})
       labels = {**LABELS, **(extra_labels or {})}
       scores = {m: ordered[m] for m in methods if m in ordered}
   ```

3. **The depth rows.** Replace `depth = depth_rows(conv, lrp)` with:

   ```python
       layer_scores = {rep: conv[f"{rep}_layers"] for rep in REPRESENTATIONS}
       layer_scores.update(extra_layers or {})
       depth = depth_rows(layer_scores, lrp)
   ```

4. **The output.** Replace `folder = settings.output / "results_convtu"` with `folder = settings.output / folder_name`. In the `report.write_outputs(...)` call, replace `methods=METHODS, labels=LABELS, title="# Conv TU pilot"` with `methods=methods, labels=labels, title=title`.

In `differential_uncertainty/convtu/pipeline.py`, add before `PHASES`:

```python
def phase_channels_report(settings: Settings) -> None:
    """Score the stored channel statistics both ways and report them next to the pilot rows and the baselines."""
    from ..baselines import report as baseline_report
    from .channels import LABELS as CHANNEL_LABELS
    from .channels import channel_method_scores
    from .report import build_pilot_report
    clean = (channels_bank_path(settings), channels_zstats_path(settings))
    if not all(path.exists() for path in clean):
        raise ValueError("run the convtu-channels phase first")
    names = [p.name for p in pilot_images(settings)]
    with np.load(clean[0]) as bank, np.load(clean[1]) as zstats:
        bank, zstats = dict(bank), dict(zstats)
    test = baseline_report._stack(settings.output / CHANNELS_FOLDER, names, CHANNEL_KEYS)
    summed, per_layer = channel_method_scores(bank, zstats, test, [f"s{stage}" for stage in range(1, 5)])
    build_pilot_report(settings, names, extra=summed, extra_layers=per_layer, extra_labels=CHANNEL_LABELS,
                       folder_name="results_convtu_channels", title="# Conv TU pilot: channel statistics")
```

`PHASES` gains `"convtu-channels-report": phase_channels_report`. `CONVTU_PHASES` and the CLI choices gain `"convtu-channels-report"`.

- [ ] **Step 4: Run the tests, then the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_convtu_channels.py tests/differential_uncertainty/test_convtu_report.py tests/differential_uncertainty/test_cli.py -q`
Expected: all pass.

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty -q`
Expected: all earlier tests still pass, 2 skipped, `0 failed`.

- [ ] **Step 5: Commit**

```bash
git add differential_uncertainty/convtu/channels.py differential_uncertainty/convtu/report.py differential_uncertainty/convtu/pipeline.py differential_uncertainty/baselines/pipeline.py differential_uncertainty/cli.py tests/differential_uncertainty/test_convtu_channels.py tests/differential_uncertainty/test_convtu_report.py tests/differential_uncertainty/test_cli.py
git commit -m "feat: score the channel statistics both ways and report them next to the pilot rows"
```

---

### Task 4: Run and write up

**Files:**
- Modify: `docs/conv-tu-pilot-results.md` (append a section)
- Modify: `docs/dev-log.md` (new entry at the top)
- Create: `docs/results/conv-tu-pilot-channels/`, holding copies of `runs/coco-baselines/results_convtu_channels/*`

**Interfaces:**
- Consumes: the phases `convtu-channels` and `convtu-channels-report`, through `baselines-coco --phase`.
- Produces: the published numbers.

- [ ] **Step 1: Agree a GPU window.** Send `explore` and `hunk-0927` a SendMessage asking for about 30 minutes of GPU at under 3 GiB: one backbone pass, while the corruptions are rebuilt on the CPU. Start only after both answer.

- [ ] **Step 2: Run the GPU phase.** Write `$WORKSPACE/convtu-launch.sh`:

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
  --phase "$PHASE" --workers "$WORKERS" --batch-size 8 --output "$OUT" \
  --checkpoint /home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \
  --coco-train-images /home/yuchen/YuchenZ/Datasets/coco/train2017 \
  --coco-val-images /home/yuchen/YuchenZ/Datasets/coco/val2017 \
  --coco-annotations /home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json \
  --discopatch-root /home/yuchen/YuchenZ/UE/DisCoPatch \
  >> "$OUT/logs/$PHASE.log" 2>&1 < /dev/null &
echo $! > "$OUT/logs/$PHASE.pid"
echo "started $PHASE pid $(cat "$OUT/logs/$PHASE.pid")"
```

Run `bash $WORKSPACE/convtu-launch.sh convtu-channels 9` and wait on the PID with a background loop over `kill -0 PID` (not `pgrep -f`).
Expected:
- `convtu/channels_bank.npz` and `convtu/channels_zstats.npz` exist;
- `test_convtu_channels/` holds 200 files;
- the log ends without a traceback.

Then tell `explore` and `hunk-0927` that the GPU is free.

- [ ] **Step 3: Run the report on the CPU.** Run `bash $WORKSPACE/convtu-launch.sh convtu-channels-report 9` and wait on the PID.
Expected: `results_convtu_channels/report.md` with 21 rows: 9 baselines, 4 pilot rows and 8 new rows.

- [ ] **Step 4: Check the reproduction.** In `results_convtu_channels/intervals.csv`, `ch_means_knn:auroc_common` and `ch_means_knn:auroc_extra` must equal `convtu_means:auroc_common` and `convtu_means:auroc_extra` within 0.001. If they don't, stop and report it; do not publish.

- [ ] **Step 5: Write up and commit.**
  - Copy `results_convtu_channels/*` into `docs/results/conv-tu-pilot-channels/`.
  - Append a section "Follow-up: channel statistics (1 October 2026)" to `docs/conv-tu-pilot-results.md`, with:
    - a table of the eight new rows next to `convtu_means` and the activation-CDF monitor: AUROC common, AUROC extra, ρ within, AURC all, with intervals;
    - the differences `ch_means_own − ch_means_knn`, `ch_top_own − ch_means_own`, `ch_grid_knn − ch_means_knn` and `ch_means_own − cdf`, with intervals;
    - one paragraph that answers the user's question.
  - Add a dated entry at the top of `docs/dev-log.md` with the one-line answer and what it means for the next step.

```bash
git add docs/conv-tu-pilot-results.md docs/dev-log.md docs/results/conv-tu-pilot-channels
git commit -m "docs: channel statistics follow-up of the conv-TU pilot"
```
