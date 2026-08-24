# Clean Differential Uncertainty Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small, pretrained-inference-only workflow that reads generic image manifests, removes repeated RT-DETR query padding, computes the fixed layer-2 90--100% versus 50--60% relative gap, and publishes a resumable detailed corruption report.

**Architecture:** Add a focused `differential_uncertainty` package around an explicit RT-DETRv2-R18 model builder and a layer-2 persistence extractor. Each run has provenance-checked extraction, bank, score, and report stages under one directory; mathematical code stays pure and is checked against the archived blur result before the old experiment framework is deleted. Only after evaluator and detector parity pass do we prune training, COCO, alternative-model, deployment, and obsolete scene-uncertainty modules.

**Tech Stack:** Python 3.11, PyTorch, torchvision v2, NumPy, SciPy, pandas, Pillow, Matplotlib, pytest, Git worktrees

---

## Working context

- Worktree: `/home/yuchen/YuchenZ/UE/philip_sa/.worktrees/clean`
- Branch: `clean`
- Design: `docs/superpowers/specs/2026-08-23-clean-differential-uncertainty-design.md`
- Historical result bundle: `/home/yuchen/YuchenZ/UE/within-image-contrast-run-record/within_image_contrast/`
- Real checkpoint for local parity: `/home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth`
- Starting baseline: `1371 passed, 86 warnings in 300.98s`
- Starting Python inventory: `158` files and `51,884` lines

Run every command below from the clean worktree. Use this interpreter consistently:

```bash
UE_PY=/home/yuchen/miniconda3/envs/UE/bin/python
```

## Final file map

### New workflow package

| File | Responsibility |
|---|---|
| `differential_uncertainty/__init__.py` | Package version and public workflow name. |
| `differential_uncertainty/__main__.py` | `python -m differential_uncertainty` entry point. |
| `differential_uncertainty/cli.py` | One `run` command and runtime-only arguments. |
| `differential_uncertainty/config.py` | Frozen scientific constants and test-only configuration construction. |
| `differential_uncertainty/manifests.py` | Generic CSV parsing, canonical ordering, overlap checks, and digesting. |
| `differential_uncertainty/artifacts.py` | Atomic JSON/Torch writes, shard resume, provenance, and safe reads. |
| `differential_uncertainty/persistence.py` | Layer-2 hook and batched maximum-spanning-tree persistence. |
| `differential_uncertainty/extraction.py` | Fixed model builder, checkpoint loading, preprocessing, and record extraction. |
| `differential_uncertainty/bank.py` | Valid-query stream and exact 25,000-vector deterministic reservoir. |
| `differential_uncertainty/scoring.py` | Padding union, confidence deciles, kNN, relative gaps, and per-image rows. |
| `differential_uncertainty/evaluation.py` | Spearman, oriented curve checks, AUROC, macro AUROC, and paired bootstrap. |
| `differential_uncertainty/reporting.py` | CSV/JSON/Markdown bundle and four detailed figures. |
| `differential_uncertainty/corruptions/base.py` | Corruption protocol and severity descriptor. |
| `differential_uncertainty/corruptions/gaussian_blur.py` | The fixed six-level PIL Gaussian blur ladder. |

### Retained detector closure

Keep only these model files under `src/`:

```text
src/__init__.py
src/nn/__init__.py
src/nn/backbone/__init__.py
src/nn/backbone/common.py
src/nn/backbone/presnet.py
src/zoo/__init__.py
src/zoo/rtdetr/__init__.py
src/zoo/rtdetr/box_ops.py
src/zoo/rtdetr/denoising.py
src/zoo/rtdetr/hybrid_encoder.py
src/zoo/rtdetr/rtdetr.py
src/zoo/rtdetr/rtdetrv2_decoder.py
src/zoo/rtdetr/utils.py
```

`box_ops.py` and `denoising.py` stay because the checkpoint contains the decoder's denoising embedding and the fixed decoder imports its training-group helper even though the clean command always runs in evaluation mode. Their presence keeps checkpoint loading strict; they are not exposed as training functionality.

### New tests

Mirror package responsibilities under `tests/differential_uncertainty/`:

```text
test_config.py
test_manifests.py
test_corruptions.py
test_artifacts.py
test_persistence.py
test_extraction.py
test_bank.py
test_scoring.py
test_evaluation.py
test_legacy_parity.py
test_reporting.py
test_pipeline.py
test_cli.py
test_detector_parity.py
test_repository_surface.py
```

The detector-parity tests also retain the binary fixture
`tests/differential_uncertainty/fixtures/rtdetrv2_r18_layer2_golden.pt` after live
old-versus-new parity has proved how it was created.

## Commit discipline

Each task ends in a working commit. Do not combine the large deletion with replacement work. If a parity check fails, stop in that task and fix the mismatch before continuing; do not relax equality, change a direction, or update a golden value to make it pass.

## Task 1: Freeze the scientific configuration and generic input contract

**Files:**
- Create: `differential_uncertainty/__init__.py`
- Create: `differential_uncertainty/config.py`
- Create: `differential_uncertainty/manifests.py`
- Create: `differential_uncertainty/corruptions/__init__.py`
- Create: `differential_uncertainty/corruptions/base.py`
- Create: `differential_uncertainty/corruptions/gaussian_blur.py`
- Test: `tests/differential_uncertainty/test_config.py`
- Test: `tests/differential_uncertainty/test_manifests.py`
- Test: `tests/differential_uncertainty/test_corruptions.py`

- [ ] **Step 1: Write failing tests for every fixed scientific value**

```python
# tests/differential_uncertainty/test_config.py
from dataclasses import replace

import pytest

from differential_uncertainty.config import FIXED_CONFIG, ExperimentConfig


def test_public_configuration_is_the_one_approved_method():
    assert FIXED_CONFIG.image_size == (640, 640)
    assert FIXED_CONFIG.query_count == 300
    assert FIXED_CONFIG.persistence_layer == 2
    assert FIXED_CONFIG.persistence_dim == 335
    assert FIXED_CONFIG.bank_capacity == 25_000
    assert FIXED_CONFIG.bank_seed == 44
    assert FIXED_CONFIG.k == 5
    assert FIXED_CONFIG.reference_decile == 9
    assert FIXED_CONFIG.responsive_decile == 5
    assert FIXED_CONFIG.persistence_orientation == 1
    assert FIXED_CONFIG.confidence_orientation == -1
    assert FIXED_CONFIG.raw_responsive_orientation == 1
    assert FIXED_CONFIG.raw_reference_orientation == -1
    assert FIXED_CONFIG.bootstrap_samples == 10_000
    assert FIXED_CONFIG.bootstrap_seed == 20_260_821
    assert FIXED_CONFIG.blur_radii == (0.0, 1.0, 2.0, 4.0, 8.0, 12.0)


def test_invalid_internal_configuration_is_rejected():
    with pytest.raises(ValueError, match="six severities"):
        replace(FIXED_CONFIG, blur_radii=(0.0, 1.0))
    with pytest.raises(ValueError, match="ten deciles"):
        replace(FIXED_CONFIG, reference_decile=10)
    with pytest.raises(ValueError, match="bank_capacity"):
        replace(FIXED_CONFIG, bank_capacity=4)
    with pytest.raises(ValueError, match="bank_chunk_size"):
        replace(FIXED_CONFIG, bank_chunk_size=0)
    with pytest.raises(ValueError, match="bootstrap_samples"):
        replace(FIXED_CONFIG, bootstrap_samples=0)


def test_small_configuration_is_available_only_as_a_python_test_helper():
    small = ExperimentConfig.for_tests(bank_capacity=20, k=2, query_count=20, persistence_dim=7)
    assert small.bank_capacity == 20
    assert small.blur_radii == FIXED_CONFIG.blur_radii
```

- [ ] **Step 2: Run the configuration tests and verify the missing package failure**

Run:

```bash
$UE_PY -m pytest tests/differential_uncertainty/test_config.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'differential_uncertainty'`.

- [ ] **Step 3: Implement the immutable configuration**

```python
# differential_uncertainty/__init__.py
"""Fixed end-to-end differential corruption uncertainty."""

__version__ = "1.0.0"
```

```python
# differential_uncertainty/config.py
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ExperimentConfig:
    image_size: tuple[int, int] = (640, 640)
    class_count: int = 80
    query_count: int = 300
    persistence_layer: int = 2
    persistence_dim: int = 335
    bank_capacity: int = 25_000
    bank_seed: int = 44
    k: int = 5
    bank_chunk_size: int = 8_192
    reference_decile: int = 9
    responsive_decile: int = 5
    persistence_orientation: int = 1
    confidence_orientation: int = -1
    raw_responsive_orientation: int = 1
    raw_reference_orientation: int = -1
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20_260_821
    blur_radii: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0, 12.0)

    def __post_init__(self) -> None:
        if len(self.blur_radii) != 6 or self.blur_radii[0] != 0.0:
            raise ValueError("the fixed experiment needs six severities beginning with clean")
        if not 0 <= self.reference_decile < 10 or not 0 <= self.responsive_decile < 10:
            raise ValueError("reference and responsive indices must address ten deciles")
        if self.reference_decile == self.responsive_decile:
            raise ValueError("reference and responsive deciles must differ")
        if self.k <= 0 or self.bank_capacity < self.k:
            raise ValueError("bank_capacity must be at least k and k must be positive")
        if self.bank_chunk_size <= 0:
            raise ValueError("bank_chunk_size must be positive")
        if self.bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be positive")
        if self.query_count < 10:
            raise ValueError("query_count must be large enough to form ten deciles")
        if self.persistence_dim <= 0:
            raise ValueError("persistence_dim must be positive")
        for value in (
            self.persistence_orientation,
            self.confidence_orientation,
            self.raw_responsive_orientation,
            self.raw_reference_orientation,
        ):
            if value not in (-1, 1):
                raise ValueError("every score orientation must be -1 or +1")

    @classmethod
    def for_tests(cls, **changes) -> "ExperimentConfig":
        return replace(cls(), **changes)

    def scientific_dict(self) -> dict:
        return {
            "image_size": list(self.image_size),
            "class_count": self.class_count,
            "query_count": self.query_count,
            "persistence_layer": self.persistence_layer,
            "persistence_dim": self.persistence_dim,
            "bank_capacity": self.bank_capacity,
            "bank_seed": self.bank_seed,
            "k": self.k,
            "reference_decile": self.reference_decile,
            "responsive_decile": self.responsive_decile,
            "orientations": {
                "persistence_relative_gap": self.persistence_orientation,
                "confidence_relative_gap": self.confidence_orientation,
                "raw_responsive": self.raw_responsive_orientation,
                "raw_reference": self.raw_reference_orientation,
            },
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_seed": self.bootstrap_seed,
            "default_blur_radii": list(self.blur_radii),
            "feature_normalization": "raw",
        }


FIXED_CONFIG = ExperimentConfig()
```

> **Manifest contract amendment (2026-08-24):** After surrounding whitespace is
> stripped, an `image_id` must not begin with `=`, `+`, `-`, or `@`.
> Task 10 rejects these IDs during manifest loading, before creating run output,
> because exact CSV identity round-tripping would otherwise expose spreadsheet
> formula interpretation.

- [ ] **Step 4: Write failing manifest and corruption tests**

```python
# tests/differential_uncertainty/test_manifests.py
from pathlib import Path

import pytest
from PIL import Image

from differential_uncertainty.manifests import load_manifest, validate_disjoint


def _image(path: Path, color: int = 0) -> None:
    Image.new("RGB", (8, 6), (color, color, color)).save(path)


def test_manifest_paths_are_resolved_and_rows_are_canonicalized(tmp_path):
    _image(tmp_path / "b.png", 20)
    _image(tmp_path / "a.png", 10)
    manifest = tmp_path / "images.csv"
    manifest.write_text("image_id,image_path\nb,b.png\na,a.png\n", encoding="utf-8")
    entries = load_manifest(manifest)
    assert [entry.image_id for entry in entries] == ["a", "b"]
    assert entries[0].path == (tmp_path / "a.png").resolve()


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("image_id,image_path\na,missing.png\n", "does not exist"),
        ("image_id,image_path\na,x.png\na,y.png\n", "duplicate image_id"),
        ("wrong,image_path\na,x.png\n", "exactly image_id,image_path"),
    ],
)
def test_manifest_errors_are_explicit(tmp_path, text, message):
    _image(tmp_path / "x.png")
    _image(tmp_path / "y.png")
    path = tmp_path / "bad.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_manifest(path)


def test_reference_and_evaluation_must_be_disjoint(tmp_path):
    _image(tmp_path / "same.png")
    ref = tmp_path / "ref.csv"
    eva = tmp_path / "eval.csv"
    ref.write_text("image_id,image_path\nr,same.png\n", encoding="utf-8")
    eva.write_text("image_id,image_path\ne,same.png\n", encoding="utf-8")
    with pytest.raises(ValueError, match="same resolved image"):
        validate_disjoint(load_manifest(ref), load_manifest(eva))
```

```python
# tests/differential_uncertainty/test_corruptions.py
import pytest
from PIL import Image, ImageChops

from differential_uncertainty.corruptions.gaussian_blur import GaussianBlur


def test_gaussian_blur_declares_the_fixed_ladder():
    corruption = GaussianBlur()
    assert corruption.name == "gaussian_blur"
    assert [item.level for item in corruption.severities] == list(range(6))
    assert [item.parameter for item in corruption.severities] == [0.0, 1.0, 2.0, 4.0, 8.0, 12.0]


def test_level_zero_is_identity_and_blur_is_deterministic():
    image = Image.effect_noise((21, 21), 90).convert("RGB")
    corruption = GaussianBlur()
    assert ImageChops.difference(corruption.apply(image, 0), image).getbbox() is None
    first = corruption.apply(image, 3)
    second = corruption.apply(image, 3)
    assert first.tobytes() == second.tobytes()
    assert ImageChops.difference(first, image).getbbox() is not None


def test_unknown_levels_are_rejected_instead_of_using_python_negative_indexing():
    image = Image.new("RGB", (5, 5))
    corruption = GaussianBlur()
    for level in (-1, 6):
        with pytest.raises(ValueError, match="unknown gaussian_blur severity"):
            corruption.apply(image, level)
```

- [ ] **Step 5: Implement manifest parsing and the corruption interface**

```python
# differential_uncertainty/manifests.py
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, order=True)
class ManifestEntry:
    image_id: str
    path: Path


def load_manifest(value: str | Path) -> tuple[ManifestEntry, ...]:
    manifest = Path(value).resolve()
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["image_id", "image_path"]:
            raise ValueError("manifest columns must be exactly image_id,image_path")
        rows = list(reader)
    entries: list[ManifestEntry] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for number, row in enumerate(rows, start=2):
        image_id = row["image_id"].strip()
        if not image_id:
            raise ValueError(f"manifest row {number} has an empty image_id")
        if image_id in seen_ids:
            raise ValueError(f"duplicate image_id {image_id!r}")
        path = (manifest.parent / row["image_path"]).resolve()
        if not path.is_file():
            raise ValueError(f"image for {image_id!r} does not exist: {path}")
        if path in seen_paths:
            raise ValueError(f"manifest repeats resolved image: {path}")
        seen_ids.add(image_id)
        seen_paths.add(path)
        entries.append(ManifestEntry(image_id, path))
    if not entries:
        raise ValueError("manifest must contain at least one image")
    return tuple(sorted(entries))


def validate_disjoint(reference, evaluation) -> None:
    ids = {entry.image_id for entry in reference} & {entry.image_id for entry in evaluation}
    if ids:
        raise ValueError(f"reference and evaluation repeat image_id values: {sorted(ids)}")
    paths = {entry.path for entry in reference} & {entry.path for entry in evaluation}
    if paths:
        raise ValueError(f"reference and evaluation contain the same resolved image: {sorted(paths)}")


def manifest_digest(entries) -> str:
    payload = [{"image_id": item.image_id, "path": str(item.path)} for item in entries]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
```

```python
# differential_uncertainty/corruptions/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image


@dataclass(frozen=True)
class Severity:
    level: int
    parameter: float


class Corruption(Protocol):
    name: str
    severities: tuple[Severity, ...]

    def apply(self, image: Image.Image, level: int) -> Image.Image: ...
```

```python
# differential_uncertainty/corruptions/gaussian_blur.py
from PIL import Image, ImageFilter

from .base import Severity


class GaussianBlur:
    name = "gaussian_blur"
    severities = tuple(
        Severity(level, radius)
        for level, radius in enumerate((0.0, 1.0, 2.0, 4.0, 8.0, 12.0))
    )

    def apply(self, image: Image.Image, level: int) -> Image.Image:
        if level < 0 or level >= len(self.severities):
            raise ValueError(f"unknown gaussian_blur severity {level}")
        try:
            radius = self.severities[level].parameter
        except IndexError as error:
            raise ValueError(f"unknown gaussian_blur severity {level}") from error
        return image if radius == 0.0 else image.filter(ImageFilter.GaussianBlur(radius))
```

```python
# differential_uncertainty/corruptions/__init__.py
from .base import Corruption, Severity
from .gaussian_blur import GaussianBlur

__all__ = ["Corruption", "GaussianBlur", "Severity"]
```

- [ ] **Step 6: Run the task tests and commit**

Run:

```bash
$UE_PY -m pytest tests/differential_uncertainty/test_config.py tests/differential_uncertainty/test_manifests.py tests/differential_uncertainty/test_corruptions.py -q
```

Expected: all tests pass.

```bash
git add differential_uncertainty tests/differential_uncertainty
git commit -m "feat: freeze differential uncertainty inputs"
```

## Task 2: Add pure layer-2 persistence extraction

**Files:**
- Create: `differential_uncertainty/persistence.py`
- Test: `tests/differential_uncertainty/test_persistence.py`
- Reference: `src/misc/tue_utils.py:1-665`

- [ ] **Step 1: Write failing arithmetic and hook tests**

```python
# tests/differential_uncertainty/test_persistence.py
import torch
from torch import nn

from differential_uncertainty.persistence import Layer2Capture, batched_persistence
from src.misc.tue_utils import get_persistence_diagrams_batched


def test_batched_persistence_matches_the_legacy_implementation():
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(5, 8, generator=generator)
    inputs = torch.randn(13, 8, generator=generator)
    expected = get_persistence_diagrams_batched(weight, inputs, chunk_size=4)
    actual = batched_persistence(weight, inputs, chunk_size=4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (13, 12)


class _Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
        self.dec_score_head = nn.ModuleList([nn.Linear(4, 3) for _ in range(3)])

    def forward(self, values):
        for layer in self.decoder.layers:
            values = layer(values)
        return values


def test_layer_2_capture_records_features_and_score_weights_once():
    transformer = _Transformer()
    with Layer2Capture(transformer, layer=2) as capture:
        output = transformer(torch.ones(2, 5, 4))
        features, weight = capture.take()
        assert capture.handles
    assert not capture.handles
    torch.testing.assert_close(features, output)
    torch.testing.assert_close(weight, transformer.dec_score_head[2].weight)
```

- [ ] **Step 2: Run the focused test and verify the missing-module failure**

```bash
$UE_PY -m pytest tests/differential_uncertainty/test_persistence.py -q
```

Expected: collection fails because `differential_uncertainty.persistence` does not exist.

- [ ] **Step 3: Implement the exact batched maximum-spanning-tree calculation**

Create `differential_uncertainty/persistence.py` by moving only the following legacy inference logic, with names narrowed to this workflow:

```python
from __future__ import annotations

import torch
from torch import Tensor, nn


@torch.no_grad()
def _batched_prim(weight_matrix: Tensor, layer_inputs: Tensor) -> Tensor:
    if weight_matrix.ndim != 2 or layer_inputs.ndim != 2:
        raise ValueError("weight_matrix and layer_inputs must both be two-dimensional")
    device = layer_inputs.device
    weight = weight_matrix.detach().to(device=device, dtype=torch.float32)
    inputs = layer_inputs.detach().to(device=device, dtype=torch.float32)
    query_count, input_count = inputs.shape
    output_count, expected_inputs = weight.shape
    if input_count != expected_inputs:
        raise ValueError("persistence input and score-head dimensions do not match")
    if query_count == 0:
        return torch.empty((0, input_count + output_count - 1), device=device)
    edges = torch.abs(inputs[:, None, :] * weight[None, :, :])
    selected_inputs = torch.zeros((query_count, input_count), dtype=torch.bool, device=device)
    selected_outputs = torch.zeros((query_count, output_count), dtype=torch.bool, device=device)
    selected_inputs[:, 0] = True
    best_inputs = torch.full((query_count, input_count), -torch.inf, device=device)
    best_outputs = edges[:, :, 0].clone()
    result = torch.empty((query_count, input_count + output_count - 1), device=device)
    for step in range(result.shape[1]):
        candidates = torch.cat(
            [best_inputs.masked_fill(selected_inputs, -torch.inf),
             best_outputs.masked_fill(selected_outputs, -torch.inf)],
            dim=1,
        )
        chosen_weight, chosen_vertex = candidates.max(dim=1)
        result[:, step] = chosen_weight
        is_input = chosen_vertex < input_count
        input_index = chosen_vertex.clamp(max=input_count - 1)
        output_index = (chosen_vertex - input_count).clamp(min=0, max=output_count - 1)
        old_input = selected_inputs.gather(1, input_index[:, None])
        selected_inputs.scatter_(1, input_index[:, None], old_input | is_input[:, None])
        old_output = selected_outputs.gather(1, output_index[:, None])
        selected_outputs.scatter_(1, output_index[:, None], old_output | (~is_input)[:, None])
        from_input = edges.gather(
            2, input_index[:, None, None].expand(-1, output_count, 1)
        ).squeeze(2)
        best_outputs = torch.where(
            is_input[:, None], torch.maximum(best_outputs, from_input), best_outputs
        )
        from_output = edges.gather(
            1, output_index[:, None, None].expand(-1, 1, input_count)
        ).squeeze(1)
        best_inputs = torch.where(
            (~is_input)[:, None], torch.maximum(best_inputs, from_output), best_inputs
        )
    return torch.sort(result, dim=1, descending=True).values


@torch.no_grad()
def batched_persistence(
    weight_matrix: Tensor, layer_inputs: Tensor, chunk_size: int = 64
) -> Tensor:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if layer_inputs.ndim != 2 or weight_matrix.ndim != 2:
        raise ValueError("weight_matrix and layer_inputs must both be two-dimensional")
    if layer_inputs.shape[0] == 0:
        width = weight_matrix.shape[0] + weight_matrix.shape[1] - 1
        return torch.empty((0, width), device=layer_inputs.device)
    return torch.cat(
        [_batched_prim(weight_matrix, part) for part in layer_inputs.split(chunk_size)], dim=0
    )


class Layer2Capture:
    def __init__(self, transformer: nn.Module, layer: int = 2) -> None:
        if layer < 0 or layer >= len(transformer.decoder.layers):
            raise ValueError(f"decoder layer {layer} does not exist")
        head = transformer.dec_score_head[layer]
        if not isinstance(head, nn.Linear):
            raise TypeError("the selected decoder score head must be nn.Linear")
        self.head = head
        self.captured: Tensor | None = None
        self.handles = [transformer.decoder.layers[layer].register_forward_hook(self._hook)]

    def _hook(self, _module, _inputs, output: Tensor) -> None:
        self.captured = output.detach()

    def take(self) -> tuple[Tensor, Tensor]:
        if self.captured is None:
            raise RuntimeError("the detector did not execute decoder layer 2")
        features = self.captured
        self.captured = None
        return features, self.head.weight.detach()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
```

- [ ] **Step 4: Run exact legacy parity and commit**

```bash
$UE_PY -m pytest tests/differential_uncertainty/test_persistence.py -q
```

Expected: both tests pass with zero tolerance in the numerical comparison.

```bash
git add differential_uncertainty/persistence.py tests/differential_uncertainty/test_persistence.py
git commit -m "feat: isolate layer two persistence extraction"
```

## Task 3: Build and load the one retained detector

**Files:**
- Create: `differential_uncertainty/extraction.py`
- Test: `tests/differential_uncertainty/test_extraction.py`
- Test: `tests/differential_uncertainty/test_detector_parity.py`
- Reference: `src/scene_uncertainty/runtime.py`
- Reference: `configs/scene_uncertainty/rtdetrv2_r18vd_coco.yml`
- Reference: `configs/rtdetrv2/include/rtdetrv2_r50vd.yml`

- [ ] **Step 1: Write a failing model-structure parity test**

```python
# tests/differential_uncertainty/test_extraction.py
import pytest
import torch

from differential_uncertainty.extraction import build_fixed_detector, checkpoint_state
from src.core import YAMLConfig


def test_explicit_builder_has_the_same_state_contract_as_the_yaml_model():
    legacy = YAMLConfig("configs/scene_uncertainty/rtdetrv2_r18vd_coco.yml").model
    clean = build_fixed_detector()
    legacy_shapes = {name: tuple(value.shape) for name, value in legacy.state_dict().items()}
    clean_shapes = {name: tuple(value.shape) for name, value in clean.state_dict().items()}
    assert clean_shapes == legacy_shapes


def test_checkpoint_state_accepts_only_the_three_historical_shapes():
    tensor = torch.tensor([1.0])
    assert checkpoint_state({"ema": {"module": {"x": tensor}}}) == {"x": tensor}
    assert checkpoint_state({"model": {"x": tensor}}) == {"x": tensor}
    assert checkpoint_state({"x": tensor}) == {"x": tensor}
```

- [ ] **Step 2: Run the structure test and verify it fails on the missing builder**

```bash
$UE_PY -m pytest tests/differential_uncertainty/test_extraction.py -q
```

Expected: import fails because `differential_uncertainty.extraction` does not exist.

- [ ] **Step 3: Implement the fixed RT-DETRv2-R18 builder and strict checkpoint loader**

```python
# differential_uncertainty/extraction.py (first section)
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor, nn
from torchvision.transforms.v2 import functional as vision

from src.nn.backbone.presnet import PResNet
from src.zoo.rtdetr.hybrid_encoder import HybridEncoder
from src.zoo.rtdetr.rtdetr import RTDETR
from src.zoo.rtdetr.rtdetrv2_decoder import RTDETRTransformerv2

from .config import ExperimentConfig, FIXED_CONFIG
from .persistence import Layer2Capture, batched_persistence


def build_fixed_detector() -> RTDETR:
    spatial = [640, 640]
    backbone = PResNet(
        depth=18, variant="d", num_stages=4, return_idx=[1, 2, 3], act="relu",
        freeze_at=-1, freeze_norm=False, pretrained=False,
    )
    encoder = HybridEncoder(
        in_channels=[128, 256, 512], feat_strides=[8, 16, 32], hidden_dim=256,
        nhead=8, dim_feedforward=1024, dropout=0.0, enc_act="gelu",
        use_encoder_idx=[2], num_encoder_layers=1, pe_temperature=10_000,
        expansion=0.5, depth_mult=1.0, act="silu", eval_spatial_size=spatial,
        version="v2",
    )
    decoder = RTDETRTransformerv2(
        num_classes=80, hidden_dim=256, num_queries=300,
        feat_channels=[256, 256, 256], feat_strides=[8, 16, 32], num_levels=3,
        num_points=[4, 4, 4], nhead=8, num_layers=3, dim_feedforward=1024,
        dropout=0.0, activation="relu", num_denoising=100, label_noise_ratio=0.5,
        box_noise_scale=1.0, learn_query_content=False, eval_spatial_size=spatial,
        eval_idx=-1, eps=1e-2, aux_loss=True, cross_attn_method="default",
        query_select_method="default",
    )
    return RTDETR(backbone=backbone, encoder=encoder, decoder=decoder)


def checkpoint_state(checkpoint: dict) -> dict[str, Tensor]:
    if "ema" in checkpoint:
        return checkpoint["ema"]["module"]
    if "model" in checkpoint:
        return checkpoint["model"]
    if checkpoint and all(isinstance(value, Tensor) for value in checkpoint.values()):
        return checkpoint
    raise KeyError("checkpoint has neither ema.module nor model state")


def load_frozen_detector(checkpoint_path: str | Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_fixed_detector()
    incompatible = model.load_state_dict(checkpoint_state(checkpoint), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.to(device).eval().requires_grad_(False)
    return model
```

- [ ] **Step 4: Write preprocessing and extraction tests**

```python
# append to tests/differential_uncertainty/test_extraction.py
from PIL import Image
from torch import nn

from differential_uncertainty.config import ExperimentConfig
from differential_uncertainty.extraction import prepare_image, record_from_outputs


def test_prepare_image_is_rgb_640_float32_in_zero_one_range():
    image = Image.new("L", (13, 7), 128)
    tensor = prepare_image(image, (640, 640))
    assert tensor.shape == (3, 640, 640)
    assert tensor.dtype == torch.float32
    assert float(tensor.min()) == pytest.approx(128 / 255)
    assert float(tensor.max()) == pytest.approx(128 / 255)


def test_record_storage_matches_the_legacy_cache_dtypes():
    logits = torch.randn(20, 80)
    boxes = torch.randn(20, 4)
    diagrams = torch.randn(20, 7)
    record = record_from_outputs("scene", 3, logits, boxes, diagrams)
    assert record["image_id"] == "scene"
    assert record["severity"] == 3
    assert record["logits"].dtype == torch.float16
    assert record["boxes"].dtype == torch.float32
    assert record["persistence"].dtype == torch.float16
```

- [ ] **Step 5: Complete preprocessing and the reusable extractor object**

```python
# append to differential_uncertainty/extraction.py
def prepare_image(image: Image.Image, image_size: tuple[int, int]) -> Tensor:
    rgb = image.convert("RGB")
    resized = vision.resize(rgb, list(image_size), antialias=True)
    return vision.pil_to_tensor(resized).to(torch.float32).div_(255.0)


def record_from_outputs(
    image_id: str, severity: int, logits: Tensor, boxes: Tensor, persistence: Tensor
) -> dict:
    return {
        "image_id": image_id,
        "severity": int(severity),
        "logits": logits.detach().cpu().to(torch.float16),
        "boxes": boxes.detach().cpu().to(torch.float32),
        "persistence": persistence.detach().cpu().to(torch.float16),
    }


class RTDETRExtractor:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: torch.device,
        config: ExperimentConfig = FIXED_CONFIG,
    ) -> None:
        self.device = device
        self.config = config
        self.model = load_frozen_detector(checkpoint_path, device)
        self.capture = Layer2Capture(self.model.decoder, layer=config.persistence_layer)

    @torch.inference_mode()
    def extract_batch(self, identities: list[tuple[str, int]], samples: Tensor) -> list[dict]:
        outputs = self.model(samples.to(self.device))
        features, weight = self.capture.take()
        flat = features.reshape(-1, features.shape[-1])
        diagrams = batched_persistence(weight, flat).reshape(
            features.shape[0], features.shape[1], -1
        )
        expected = (
            len(identities), self.config.query_count, self.config.persistence_dim
        )
        if tuple(diagrams.shape) != expected:
            raise RuntimeError(f"unexpected persistence shape {tuple(diagrams.shape)}; expected {expected}")
        return [
            record_from_outputs(image_id, severity, logits, boxes, diagram)
            for (image_id, severity), logits, boxes, diagram in zip(
                identities, outputs["pred_logits"], outputs["pred_boxes"], diagrams
            )
        ]

    def close(self) -> None:
        self.capture.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
```

- [ ] **Step 6: Add a real checkpoint parity test before any legacy code is removed**

```python
# tests/differential_uncertainty/test_detector_parity.py
from pathlib import Path

import pytest
import torch

from differential_uncertainty.extraction import load_frozen_detector
from differential_uncertainty.persistence import Layer2Capture, batched_persistence
from src.misc.tue_utils import get_captured_persistence_diagrams, hook_decoder_layers
from src.scene_uncertainty.runtime import load_frozen_detector as load_legacy


CHECKPOINT = Path(
    "/home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/"
    "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
)
CONFIG = Path("configs/scene_uncertainty/rtdetrv2_r18vd_coco.yml")


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="local RT-DETRv2 checkpoint unavailable")
def test_explicit_model_and_layer_two_persistence_equal_the_legacy_path():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    samples = torch.rand((1, 3, 640, 640), generator=torch.Generator().manual_seed(19))
    legacy = load_legacy(CONFIG, CHECKPOINT, device)
    clean = load_frozen_detector(CHECKPOINT, device)
    with torch.inference_mode(), Layer2Capture(clean.decoder, 2) as clean_capture:
        clean_outputs = clean(samples.to(device))
        clean_features, clean_weight = clean_capture.take()
        clean_diagrams = batched_persistence(
            clean_weight, clean_features.reshape(-1, 256)
        ).reshape(1, 300, 335)
    captures, handles, layers = hook_decoder_layers(legacy.decoder, [2], "score")
    try:
        with torch.inference_mode():
            legacy_outputs = legacy(samples.to(device))
        selected = [torch.arange(300, device=device)]
        legacy_diagrams = torch.stack(
            list(get_captured_persistence_diagrams(captures, selected, layers)[2][0].values())
        )
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(clean_outputs["pred_logits"], legacy_outputs["pred_logits"], rtol=0, atol=0)
    torch.testing.assert_close(clean_outputs["pred_boxes"], legacy_outputs["pred_boxes"], rtol=0, atol=0)
    torch.testing.assert_close(clean_diagrams.cpu(), legacy_diagrams.cpu(), rtol=0, atol=0)
```

- [ ] **Step 7: Run unit and real parity tests, then commit**

```bash
$UE_PY -m pytest tests/differential_uncertainty/test_extraction.py tests/differential_uncertainty/test_detector_parity.py -q
```

Expected: all structure, dtype, checkpoint, logits, boxes, and persistence comparisons pass. On a host without the known checkpoint, only the real parity test is skipped; do not delete the legacy path on such a host.

```bash
git add differential_uncertainty/extraction.py tests/differential_uncertainty/test_extraction.py tests/differential_uncertainty/test_detector_parity.py
git commit -m "feat: add fixed pretrained detector extraction"
```

## Task 4: Add provenance-checked resumable artifacts

**Files:**
- Create: `differential_uncertainty/artifacts.py`
- Test: `tests/differential_uncertainty/test_artifacts.py`
- Reference: `src/scene_uncertainty/artifacts.py`

- [ ] **Step 1: Write failing tests for atomic publication, resume, and mismatch refusal**

```python
# tests/differential_uncertainty/test_artifacts.py
import json
from pathlib import Path

import pytest
import torch

from differential_uncertainty.artifacts import (
    ShardWriter,
    ensure_provenance,
    iter_records,
    sha256_file,
    source_digest,
)


def test_completed_shards_publish_an_atomic_manifest(tmp_path):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation", "input_id": "abc"}
    with ShardWriter(root, metadata, shard_size=2) as writer:
        writer.add({"image_id": "a", "severity": 0, "value": torch.tensor([1])})
        writer.add({"image_id": "a", "severity": 1, "value": torch.tensor([2])})
    assert (root / "manifest.json").is_file()
    assert [(r["image_id"], r["severity"]) for r in iter_records(root)] == [("a", 0), ("a", 1)]


def test_partial_shards_resume_from_their_record_keys(tmp_path):
    root = tmp_path / "cache"
    writer = ShardWriter(root, {"stage": "reference", "input_id": "abc"}, shard_size=1)
    writer.add({"image_id": "a", "severity": 0, "value": torch.tensor([1])})
    resumed = ShardWriter(root, {"stage": "reference", "input_id": "abc"}, shard_size=1)
    assert resumed.existing_keys() == {("a", 0)}
    resumed.add({"image_id": "b", "severity": 0, "value": torch.tensor([2])})
    resumed.close()
    assert len(list(iter_records(root))) == 2


def test_provenance_mismatch_refuses_to_mix_runs(tmp_path):
    run = tmp_path / "run"
    expected = {"schema_version": 1, "checkpoint_sha256": "first", "config": {"k": 5}}
    ensure_provenance(run, expected)
    ensure_provenance(run, expected)
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        ensure_provenance(run, {**expected, "checkpoint_sha256": "second"})


def test_sha256_file_reads_content_not_the_filename(tmp_path):
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    assert sha256_file(first) == sha256_file(second)


def test_source_digest_uses_repository_relative_names(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    (first_root / "module.py").write_text("VALUE = 1\n")
    (second_root / "module.py").write_text("VALUE = 1\n")
    assert source_digest([first_root / "module.py"], root=first_root) == source_digest(
        [second_root / "module.py"], root=second_root
    )
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

```bash
$UE_PY -m pytest tests/differential_uncertainty/test_artifacts.py -q
```

Expected: collection fails because `differential_uncertainty.artifacts` does not exist.

- [ ] **Step 3: Implement safe atomic artifact helpers**

```python
# differential_uncertainty/artifacts.py
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path

import torch


SCHEMA_VERSION = 1


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_file(value: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(value).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def source_digest(paths, *, root: str | Path) -> str:
    root = Path(root).resolve()
    digest = hashlib.sha256()
    for path in sorted(Path(item).resolve() for item in paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_json(value: Mapping, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def atomic_torch(value, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, target)


def ensure_provenance(run_directory: str | Path, expected: Mapping) -> Path:
    path = Path(run_directory) / "artifacts" / "provenance.json"
    if not path.exists():
        atomic_json(dict(expected), path)
        return path
    actual = json.loads(path.read_text(encoding="utf-8"))
    for key in sorted(set(actual) | set(expected)):
        if _canonical(actual.get(key)) != _canonical(expected.get(key)):
            raise ValueError(
                f"run provenance mismatch for {key}: actual={actual.get(key)!r}, "
                f"expected={expected.get(key)!r}; choose a new --output-dir"
            )
    return path


class ShardWriter:
    def __init__(self, directory: str | Path, metadata: Mapping, shard_size: int = 50) -> None:
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata)
        self.shard_size = int(shard_size)
        self.final = self.directory / "manifest.json"
        self.partial = self.directory / "partial_manifest.json"
        self.buffer: list[dict] = []
        if self.final.exists():
            raise FileExistsError(f"artifact is already complete: {self.final}")
        if self.partial.exists():
            state = json.loads(self.partial.read_text(encoding="utf-8"))
            for key, expected in self.metadata.items():
                if _canonical(state.get(key)) != _canonical(expected):
                    raise ValueError(f"partial artifact mismatch for {key}")
            if state["shard_size"] != self.shard_size:
                raise ValueError("cannot resume with a different shard_size")
            self.shards = list(state["shards"])
            self.record_count = int(state["record_count"])
        else:
            self.shards = []
            self.record_count = 0
            self._publish_partial()

    def _publish_partial(self) -> None:
        atomic_json(
            {**self.metadata, "schema_version": SCHEMA_VERSION, "shard_size": self.shard_size,
             "record_count": self.record_count, "shards": self.shards},
            self.partial,
        )

    def existing_keys(self) -> set[tuple[str, int]]:
        return {
            (str(record["image_id"]), int(record.get("severity", 0)))
            for shard in self.shards
            for record in torch.load(self.directory / shard, map_location="cpu", weights_only=True)
        }

    def add(self, record: dict) -> None:
        self.buffer.append(record)
        self.record_count += 1
        if len(self.buffer) >= self.shard_size:
            self._flush()

    def _flush(self) -> None:
        if not self.buffer:
            return
        name = f"shard_{len(self.shards):05d}.pt"
        atomic_torch(self.buffer, self.directory / name)
        self.shards.append(name)
        self.buffer = []
        self._publish_partial()

    def close(self) -> None:
        self._flush()
        atomic_json(
            {**self.metadata, "schema_version": SCHEMA_VERSION,
             "record_count": self.record_count, "shards": self.shards},
            self.final,
        )
        self.partial.unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_exc) -> None:
        if exc_type is None:
            self.close()


def load_manifest(directory: str | Path) -> dict:
    return json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))


def iter_records(directory: str | Path) -> Iterator[dict]:
    root = Path(directory)
    for shard in load_manifest(root)["shards"]:
        yield from torch.load(root / shard, map_location="cpu", weights_only=True)
```

- [ ] **Step 4: Run tests and commit**

```bash
$UE_PY -m pytest tests/differential_uncertainty/test_artifacts.py -q
git add differential_uncertainty/artifacts.py tests/differential_uncertainty/test_artifacts.py
git commit -m "feat: add resumable run artifacts"
```

Expected: all artifact tests pass, and no `.tmp` file remains after successful publication.

## Task 5: Implement padding-safe differential scoring

**Files:**
- Create: `differential_uncertainty/scoring.py`
- Test: `tests/differential_uncertainty/test_scoring.py`
- Reference: `src/scene_uncertainty/confidence_deciles.py`
- Reference: `src/scene_uncertainty/knn.py`
- Reference: `src/scene_uncertainty/contrast_scores.py`

- [ ] **Step 1: Write failing hand-arithmetic tests**

```python
# tests/differential_uncertainty/test_scoring.py
import math

import pytest
import torch

from differential_uncertainty.config import ExperimentConfig
from differential_uncertainty.scoring import (
    confidence_deciles,
    confidence_from_logits,
    detect_padded_tail,
    mean_knn_distance,
    relative_gap,
    score_image_records,
    union_padded_query_ids,
)


def _record(image_id="a", severity=0, query_count=20, width=7):
    generator = torch.Generator().manual_seed(100 + severity)
    return {
        "image_id": image_id,
        "severity": severity,
        "boxes": torch.randn(query_count, 4, generator=generator),
        "logits": torch.randn(query_count, 80, generator=generator),
        "persistence": torch.randn(query_count, width, generator=generator),
    }


def test_relative_gap_uses_the_symmetric_scale_independent_formula():
    assert relative_gap(2.0, 6.0) == 1.0
    assert relative_gap(20.0, 60.0) == 1.0
    assert relative_gap(0.0, 0.0) == 0.0
    with pytest.raises(ValueError, match="non-negative"):
        relative_gap(-1.0, 2.0)


def test_exact_repeated_suffix_is_padding_only_when_two_rows_repeat():
    record = _record()
    for field in ("boxes", "logits", "persistence"):
        record[field][-2] = record[field][-1]
    assert detect_padded_tail(record).tolist() == [18, 19]
    record["persistence"][-2, 0] += 1
    assert detect_padded_tail(record).numel() == 0


def test_padding_is_union_of_all_six_severities():
    records = [_record(severity=severity) for severity in range(6)]
    for field in ("boxes", "logits", "persistence"):
        records[1][field][-2] = records[1][field][-1]
        records[5][field][-3:] = records[5][field][-1]
    assert union_padded_query_ids(records).tolist() == [17, 18, 19]


def test_deciles_are_low_first_stable_and_break_ties_by_query_id():
    confidence = torch.tensor([0.5] * 20)
    bins = confidence_deciles(confidence, torch.arange(20))
    assert bins[0].tolist() == [0, 1]
    assert bins[5].tolist() == [10, 11]
    assert bins[9].tolist() == [18, 19]


def test_mean_knn_is_exact_on_small_vectors():
    queries = torch.tensor([[0.0], [4.0]])
    bank = torch.tensor([[0.0], [2.0], [6.0]])
    actual = mean_knn_distance(queries, bank, k=2, bank_chunk_size=2)
    torch.testing.assert_close(actual, torch.tensor([1.0, 2.0]))


def test_confidence_is_maximum_sigmoid_not_softmax():
    logits = torch.tensor([[0.0, math.log(3.0)], [0.0, 0.0]])
    torch.testing.assert_close(confidence_from_logits(logits), torch.tensor([0.75, 0.5]))
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

```bash
$UE_PY -m pytest tests/differential_uncertainty/test_scoring.py -q
```

Expected: collection fails because `differential_uncertainty.scoring` does not exist.

- [ ] **Step 3: Implement exact padding, confidence bins, kNN, and relative gap**

```python
# differential_uncertainty/scoring.py (pure primitives)
from __future__ import annotations

import math

import torch
from torch import Tensor

from .config import ExperimentConfig, FIXED_CONFIG


def _equal_to_last(values: Tensor) -> Tensor:
    if values.ndim < 2:
        raise ValueError("query fields need query and feature dimensions")
    return values.eq(values[-1]).reshape(values.shape[0], -1).all(dim=1)


def detect_padded_tail(record: dict) -> Tensor:
    fields = [record["boxes"], record["logits"], record["persistence"]]
    counts = {int(field.shape[0]) for field in fields}
    if len(counts) != 1:
        raise ValueError("query-count mismatch inside extraction record")
    count = counts.pop()
    if count == 0:
        return torch.empty(0, dtype=torch.long)
    repeated = torch.stack([_equal_to_last(field.cpu()) for field in fields]).all(dim=0)
    start = count - 1
    while start > 0 and bool(repeated[start - 1]):
        start -= 1
    if count - start < 2:
        return torch.empty(0, dtype=torch.long)
    return torch.arange(start, count, dtype=torch.long)


def union_padded_query_ids(records) -> Tensor:
    records = list(records)
    if not records:
        raise ValueError("cannot build a padding union from zero records")
    expected = {str(record["image_id"]) for record in records}
    if len(expected) != 1:
        raise ValueError("padding union needs records from exactly one image")
    masks = [detect_padded_tail(record) for record in records]
    return torch.cat(masks).unique() if masks else torch.empty(0, dtype=torch.long)


def confidence_from_logits(logits: Tensor) -> Tensor:
    if logits.ndim != 2:
        raise ValueError("logits must have shape query,class")
    return logits.float().sigmoid().amax(dim=-1).cpu()


def confidence_deciles(confidence: Tensor, valid_ids: Tensor) -> tuple[Tensor, ...]:
    scores = confidence.float().cpu()
    valid = torch.sort(valid_ids.long().cpu()).values
    if valid.numel() < 10:
        raise ValueError(f"ten confidence deciles need at least ten valid queries, got {valid.numel()}")
    if valid.unique().numel() != valid.numel():
        raise ValueError("valid query IDs must be unique")
    selected = scores.index_select(0, valid)
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("confidence must be finite")
    order = torch.argsort(selected, stable=True)
    return tuple(torch.tensor_split(valid.index_select(0, order), 10))


def _squared_distances(queries: Tensor, bank: Tensor) -> Tensor:
    return (
        queries.square().sum(1, keepdim=True)
        + bank.square().sum(1).unsqueeze(0)
        - 2.0 * queries @ bank.T
    ).clamp_min_(0.0)


def mean_knn_distance(
    queries: Tensor, bank: Tensor, k: int, bank_chunk_size: int = 8_192
) -> Tensor:
    queries = queries.float()
    bank = bank.float().to(queries.device)
    if k <= 0 or k > bank.shape[0]:
        raise ValueError(f"k must lie in [1, {bank.shape[0]}]")
    best = torch.full((queries.shape[0], k), float("inf"), device=queries.device)
    for chunk in bank.split(bank_chunk_size):
        local = _squared_distances(queries, chunk)
        local = local.topk(min(k, chunk.shape[0]), dim=1, largest=False).values
        best = torch.cat((best, local), dim=1).topk(k, dim=1, largest=False).values
    return best.sqrt().mean(dim=1)


def relative_gap(reference: float, responsive: float) -> float:
    reference, responsive = float(reference), float(responsive)
    if not math.isfinite(reference) or not math.isfinite(responsive):
        raise ValueError("relative-gap inputs must be finite")
    if reference < 0 or responsive < 0:
        raise ValueError("relative-gap inputs must be non-negative")
    total = reference + responsive
    return 0.0 if total == 0.0 else 2.0 * (responsive - reference) / total
```

- [ ] **Step 4: Add a complete six-severity scoring test**

```python
# append to tests/differential_uncertainty/test_scoring.py
def test_one_image_produces_six_complete_persistence_and_confidence_rows():
    config = ExperimentConfig.for_tests(bank_capacity=30, k=2, query_count=20, persistence_dim=7)
    records = [_record(severity=severity) for severity in range(6)]
    bank = torch.randn(30, 7, generator=torch.Generator().manual_seed(3))
    rows = score_image_records(records, bank, config)
    assert [row["severity"] for row in rows] == list(range(6))
    for row in rows:
        assert row["valid_count"] == 20
        assert row["reference_count"] == 2
        assert row["responsive_count"] == 2
        assert -2.0 <= row["persistence_relative_gap"] <= 2.0
        assert -2.0 <= row["confidence_relative_gap"] <= 2.0
```

- [ ] **Step 5: Implement the fixed per-image scoring path**

```python
# append to differential_uncertainty/scoring.py
def score_image_records(
    records, bank: Tensor, config: ExperimentConfig = FIXED_CONFIG
) -> list[dict]:
    records = sorted(records, key=lambda item: int(item["severity"]))
    if [int(item["severity"]) for item in records] != list(range(6)):
        raise ValueError("each image must have exactly severities 0 through 5")
    image_ids = {str(item["image_id"]) for item in records}
    if len(image_ids) != 1:
        raise ValueError("score_image_records needs exactly one image")
    padded = union_padded_query_ids(records)
    all_ids = torch.arange(config.query_count)
    keep = torch.ones(config.query_count, dtype=torch.bool)
    keep[padded] = False
    valid = all_ids[keep]
    if valid.numel() < 10:
        raise ValueError("padding leaves too few valid queries to form ten deciles")
    rows: list[dict] = []
    for record in records:
        confidence = confidence_from_logits(record["logits"])
        bins = confidence_deciles(confidence, valid)
        reference_ids = bins[config.reference_decile]
        responsive_ids = bins[config.responsive_decile]
        persistence = record["persistence"].float()
        reference_distance = float(mean_knn_distance(
            persistence.index_select(0, reference_ids), bank, config.k, config.bank_chunk_size
        ).mean())
        responsive_distance = float(mean_knn_distance(
            persistence.index_select(0, responsive_ids), bank, config.k, config.bank_chunk_size
        ).mean())
        uncertainty = 1.0 - confidence
        reference_uncertainty = float(uncertainty.index_select(0, reference_ids).mean())
        responsive_uncertainty = float(uncertainty.index_select(0, responsive_ids).mean())
        rows.append({
            "image_id": next(iter(image_ids)),
            "severity": int(record["severity"]),
            "padded_count": int(padded.numel()),
            "valid_count": int(valid.numel()),
            "reference_count": int(reference_ids.numel()),
            "responsive_count": int(responsive_ids.numel()),
            "persistence_reference": reference_distance,
            "persistence_responsive": responsive_distance,
            "persistence_relative_gap": relative_gap(reference_distance, responsive_distance),
            "confidence_reference": reference_uncertainty,
            "confidence_responsive": responsive_uncertainty,
            "confidence_relative_gap": relative_gap(reference_uncertainty, responsive_uncertainty),
        })
    return rows
```

- [ ] **Step 6: Run all scoring tests and commit**

```bash
$UE_PY -m pytest tests/differential_uncertainty/test_scoring.py -q
git add differential_uncertainty/scoring.py tests/differential_uncertainty/test_scoring.py
git commit -m "feat: score the fixed relative differential"
```

Expected: every hand calculation and complete-curve test passes.

## Task 6: Build the cleaned deterministic reference bank

**Files:**
- Create: `differential_uncertainty/bank.py`
- Test: `tests/differential_uncertainty/test_bank.py`

- [ ] **Step 1: Write failing tests for padding removal, exact capacity, and determinism**

```python
# tests/differential_uncertainty/test_bank.py
import pytest
import torch

from differential_uncertainty.bank import build_reference_bank
from differential_uncertainty.config import ExperimentConfig


def _record(image_id, offset):
    persistence = torch.arange(42, dtype=torch.float32).reshape(6, 7) + offset
    boxes = torch.arange(24, dtype=torch.float32).reshape(6, 4) + offset
    logits = torch.arange(480, dtype=torch.float32).reshape(6, 80) + offset
    for field in (persistence, boxes, logits):
        field[-2] = field[-1]
    return {"image_id": image_id, "severity": 0, "persistence": persistence,
            "boxes": boxes, "logits": logits}


def test_bank_removes_each_reference_image_padded_tail_and_is_order_invariant():
    config = ExperimentConfig.for_tests(bank_capacity=6, k=2, query_count=6, persistence_dim=7)
    records = [_record("b", 100), _record("a", 0)]
    first = build_reference_bank(records, config)
    second = build_reference_bank(reversed(records), config)
    assert first.shape == (6, 7)
    assert first.dtype == torch.float32
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not any(torch.equal(row, records[0]["persistence"][-1]) for row in first)


def test_bank_refuses_fewer_valid_vectors_than_the_fixed_capacity():
    config = ExperimentConfig.for_tests(bank_capacity=9, k=2, query_count=6, persistence_dim=7)
    with pytest.raises(ValueError, match="needs exactly 9 valid vectors"):
        build_reference_bank([_record("a", 0), _record("b", 100)], config)
```

- [ ] **Step 2: Run the test and verify the missing-module failure**

```bash
$UE_PY -m pytest tests/differential_uncertainty/test_bank.py -q
```

Expected: collection fails because `differential_uncertainty.bank` does not exist.

- [ ] **Step 3: Implement the valid-query deterministic reservoir**

```python
# differential_uncertainty/bank.py
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import Tensor

from .artifacts import atomic_torch
from .config import ExperimentConfig, FIXED_CONFIG
from .scoring import detect_padded_tail


def _valid_vectors(records) -> Iterable[Tensor]:
    for record in sorted(records, key=lambda item: str(item["image_id"])):
        if int(record.get("severity", 0)) != 0:
            raise ValueError("reference bank records must be clean severity 0")
        padded = set(detect_padded_tail(record).tolist())
        for query_id, vector in enumerate(record["persistence"]):
            if query_id not in padded:
                yield vector.detach().cpu().float().clone()


def build_reference_bank(
    records, config: ExperimentConfig = FIXED_CONFIG
) -> Tensor:
    rng = np.random.default_rng(config.bank_seed)
    reservoir: list[Tensor] = []
    seen = 0
    for seen, vector in enumerate(_valid_vectors(records), start=1):
        if len(reservoir) < config.bank_capacity:
            reservoir.append(vector)
        else:
            replacement = int(rng.integers(0, seen))
            if replacement < config.bank_capacity:
                reservoir[replacement] = vector
    if seen < config.bank_capacity:
        raise ValueError(
            f"reference bank needs exactly {config.bank_capacity} valid vectors; got {seen}"
        )
    return torch.stack(reservoir)


def save_reference_bank(bank: Tensor, path, metadata: dict) -> None:
    atomic_torch({"vectors": bank.cpu().float(), "metadata": metadata}, path)


def load_reference_bank(path) -> tuple[Tensor, dict]:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    return artifact["vectors"].float(), artifact["metadata"]
```

- [ ] **Step 4: Run the bank tests and commit**

```bash
$UE_PY -m pytest tests/differential_uncertainty/test_bank.py -q
git add differential_uncertainty/bank.py tests/differential_uncertainty/test_bank.py
git commit -m "feat: build a padding-clean reference bank"
```

Expected: exact-capacity and ordering tests pass.

## Task 7: Extract generic manifests into resumable reference and evaluation caches

**Files:**
- Modify: `differential_uncertainty/extraction.py`
- Test: `tests/differential_uncertainty/test_extraction.py`

- [ ] **Step 1: Write a failing fake-extractor test for resize-before-blur and resume**

```python
# append to tests/differential_uncertainty/test_extraction.py
from differential_uncertainty.artifacts import iter_records, load_manifest as load_artifact_manifest
from differential_uncertainty.corruptions.gaussian_blur import GaussianBlur
from differential_uncertainty.extraction import extract_manifest
from differential_uncertainty.manifests import ManifestEntry


class FakeExtractor:
    def __init__(self):
        self.calls = 0

    def extract_batch(self, identities, samples):
        self.calls += 1
        records = []
        for (image_id, severity), sample in zip(identities, samples):
            value = float(sample.mean()) + severity
            records.append({
                "image_id": image_id,
                "severity": severity,
                "boxes": torch.full((20, 4), value),
                "logits": torch.full((20, 80), value),
                "persistence": torch.full((20, 7), value),
            })
        return records


def test_evaluation_cache_has_six_records_and_completed_cache_is_reused(tmp_path):
    path = tmp_path / "image.png"
    Image.effect_noise((19, 11), 80).convert("RGB").save(path)
    entries = (ManifestEntry("scene", path.resolve()),)
    cache = tmp_path / "evaluation"
    extractor = FakeExtractor()
    metadata = {"stage": "evaluation", "input_id": "fixed"}
    extract_manifest(entries, cache, metadata, extractor, GaussianBlur(),
                     image_size=(640, 640), batch_size=2, shard_size=2)
    assert [(r["image_id"], r["severity"]) for r in iter_records(cache)] == [
        ("scene", 0), ("scene", 1), ("scene", 2), ("scene", 3), ("scene", 4), ("scene", 5)
    ]
    assert load_artifact_manifest(cache)["record_count"] == 6
    calls = extractor.calls
    extract_manifest(entries, cache, metadata, extractor, GaussianBlur(),
                     image_size=(640, 640), batch_size=2, shard_size=2)
    assert extractor.calls == calls
```

- [ ] **Step 2: Run the focused test and verify `extract_manifest` is missing**

```bash
$UE_PY -m pytest tests/differential_uncertainty/test_extraction.py::test_evaluation_cache_has_six_records_and_completed_cache_is_reused -q
```

Expected: import fails for `extract_manifest`.

- [ ] **Step 3: Split resize from tensor conversion and implement the staged extractor**

```python
# append/adjust differential_uncertainty/extraction.py
from .artifacts import ShardWriter, load_manifest as load_artifact_manifest
from .corruptions.base import Corruption, Severity
from .manifests import ManifestEntry


CLEAN_ONLY = (Severity(0, 0.0),)


def resize_image(image: Image.Image, image_size: tuple[int, int]) -> Image.Image:
    return vision.resize(image.convert("RGB"), list(image_size), antialias=True)


def image_tensor(resized: Image.Image) -> Tensor:
    return vision.pil_to_tensor(resized).to(torch.float32).div_(255.0)


def prepare_image(image: Image.Image, image_size: tuple[int, int]) -> Tensor:
    return image_tensor(resize_image(image, image_size))


def _pending_samples(entries, corruption, existing, image_size):
    for entry in entries:
        with Image.open(entry.path) as opened:
            resized = resize_image(opened, image_size)
            severities = corruption.severities if corruption is not None else CLEAN_ONLY
            for severity in severities:
                key = (entry.image_id, int(severity.level))
                if key in existing:
                    continue
                changed = resized if corruption is None else corruption.apply(resized, severity.level)
                yield key, image_tensor(changed)


def extract_manifest(
    entries: tuple[ManifestEntry, ...],
    directory,
    metadata: dict,
    extractor,
    corruption: Corruption | None,
    *,
    image_size: tuple[int, int],
    batch_size: int,
    shard_size: int,
) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    root = Path(directory)
    final = root / "manifest.json"
    severity_count = len(corruption.severities) if corruption is not None else 1
    expected_count = len(entries) * severity_count
    if final.exists():
        actual = load_artifact_manifest(root)
        for key, expected in metadata.items():
            if actual.get(key) != expected:
                raise ValueError(f"completed extraction mismatch for {key}")
        if actual["record_count"] != expected_count:
            raise RuntimeError(
                f"completed extraction has {actual['record_count']} records; expected {expected_count}"
            )
        return
    writer = ShardWriter(root, metadata, shard_size=shard_size)
    existing = writer.existing_keys()
    identities: list[tuple[str, int]] = []
    tensors: list[Tensor] = []

    def flush_batch() -> None:
        records = extractor.extract_batch(identities, torch.stack(tensors))
        actual_keys = [(str(item["image_id"]), int(item["severity"])) for item in records]
        if actual_keys != identities:
            raise RuntimeError(
                f"extractor returned keys {actual_keys!r}; expected {identities!r}"
            )
        for record in records:
            writer.add(record)

    for identity, tensor in _pending_samples(entries, corruption, existing, image_size):
        identities.append(identity)
        tensors.append(tensor)
        if len(tensors) == batch_size:
            flush_batch()
            identities, tensors = [], []
    if tensors:
        flush_batch()
    if writer.record_count != expected_count:
        raise RuntimeError(
            f"extraction wrote {writer.record_count} records; expected {expected_count}"
        )
    writer.close()
```

- [ ] **Step 4: Add strict record-count checks and rerun extraction tests**

Add assertions to the extraction test for the final manifest's `record_count`, and add a
malformed fake extractor case that returns the wrong identity. Require the record-count and
identity errors before a final `manifest.json` is published, leaving the partial cache resumable.

```python
# append to tests/differential_uncertainty/test_extraction.py
class WrongIdentityExtractor(FakeExtractor):
    def extract_batch(self, identities, samples):
        records = super().extract_batch(identities, samples)
        records[0]["image_id"] = "wrong"
        return records


def test_wrong_extractor_identity_never_publishes_a_complete_cache(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (9, 7)).save(path)
    entries = (ManifestEntry("scene", path.resolve()),)
    cache = tmp_path / "bad-reference"
    with pytest.raises(RuntimeError, match="extractor returned keys"):
        extract_manifest(
            entries, cache, {"stage": "reference"}, WrongIdentityExtractor(), None,
            image_size=(640, 640), batch_size=1, shard_size=1,
        )
    assert not (cache / "manifest.json").exists()
    assert (cache / "partial_manifest.json").exists()
```

Run:

```bash
$UE_PY -m pytest tests/differential_uncertainty/test_extraction.py -q
```

Expected: model, preprocessing, dtype, six-level, and resume tests pass.

- [ ] **Step 5: Commit the extraction stages**

```bash
git add differential_uncertainty/extraction.py tests/differential_uncertainty/test_extraction.py
git commit -m "feat: cache generic image extraction"
```

## Task 8: Evaluate the two scores and diagnostic controls

**Files:**
- Create: `differential_uncertainty/evaluation.py`
- Test: `tests/differential_uncertainty/test_evaluation.py`
- Test: `tests/differential_uncertainty/test_legacy_parity.py`
- Reference: `src/scene_uncertainty/corruption_metrics.py`
- Reference: `src/scene_uncertainty/contrast_controls.py`

- [ ] **Step 1: Write failing hand-arithmetic tests for Spearman and AUROC**

```python
# tests/differential_uncertainty/test_evaluation.py
import pytest

from differential_uncertainty.evaluation import (
    binary_auroc,
    paired_macro_bootstrap,
    summarize_series,
    trend_metrics,
)


def _rows(field, curves):
    return [
        {"image_id": image_id, "severity": severity, field: value}
        for image_id, curve in curves.items()
        for severity, value in enumerate(curve)
    ]


def test_trend_metrics_distinguish_increasing_constant_and_incomplete():
    assert trend_metrics([0, 1, 2, 3, 4, 5])["signed_spearman"] == 1.0
    assert trend_metrics([4, 4, 4, 4, 4, 4])["signed_spearman"] == 0.0
    with pytest.raises(ValueError, match="six finite"):
        trend_metrics([0, 1, 2])


def test_tie_aware_auroc_matches_the_pair_count_example():
    assert binary_auroc([0, 1], [1, 2], orientation=1) == 0.875
    assert binary_auroc([0, 1], [1, 2], orientation=-1) == 0.125


def test_series_summary_reports_every_severity_and_curve_check():
    field = "score"
    rows = _rows(field, {"a": [0, 1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5, 6]})
    summary = summarize_series(rows, field, orientation=1)
    assert summary["median_signed_spearman"] == 1.0
    assert summary["oriented_adjacent_consistency"] == 1.0
    assert summary["strongest_blur_above_clean_rate"] == 1.0
    assert set(summary["auroc_by_severity"]) == {1, 2, 3, 4, 5}


def test_empty_series_is_an_explicit_error():
    with pytest.raises(ValueError, match="at least one complete image"):
        summarize_series([], "score", orientation=1)


def test_paired_bootstrap_of_a_series_against_itself_is_exactly_zero():
    rows = _rows("candidate", {"a": [0, 1, 2, 3, 4, 5], "b": [1, 1, 2, 2, 3, 3]})
    for row in rows:
        row["control"] = row["candidate"]
    result = paired_macro_bootstrap(
        rows, "candidate", "control", candidate_orientation=1,
        control_orientation=1, samples=100, seed=9,
    )
    assert result["point_difference"] == 0.0
    assert result["ci_low"] == result["ci_high"] == 0.0
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

```bash
$UE_PY -m pytest tests/differential_uncertainty/test_evaluation.py -q
```

Expected: collection fails because `differential_uncertainty.evaluation` does not exist.

- [ ] **Step 3: Implement strict trend, curve, and AUROC metrics**

```python
# differential_uncertainty/evaluation.py (metrics section)
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import rankdata, spearmanr

from .config import ExperimentConfig, FIXED_CONFIG


SEVERITIES = tuple(range(6))


def trend_metrics(scores) -> dict:
    values = np.asarray(list(scores), dtype=float)
    if values.shape != (6,) or not bool(np.isfinite(values).all()):
        raise ValueError("trend metrics need exactly six finite scores")
    signed = 0.0 if bool(np.all(values == values[0])) else float(
        spearmanr(np.arange(6), values).statistic
    )
    return {
        "signed_spearman": signed,
        "absolute_spearman": abs(signed),
        "direction": "increasing" if signed > 0 else "decreasing" if signed < 0 else "flat",
    }


def oriented_curve_metrics(scores, orientation: int) -> dict:
    if orientation not in (-1, 1):
        raise ValueError("orientation must be -1 or +1")
    values = np.asarray(list(scores), dtype=float)
    if values.shape != (6,) or not bool(np.isfinite(values).all()):
        raise ValueError("curve metrics need exactly six finite scores")
    oriented = values * orientation
    return {
        "adjacent_consistency": float(np.mean(np.diff(oriented) >= 0)),
        "strongest_blur_above_clean": bool(oriented[5] > oriented[0]),
    }


def binary_auroc(clean_scores, corrupted_scores, *, orientation: int) -> float:
    if orientation not in (-1, 1):
        raise ValueError("orientation must be -1 or +1")
    clean = np.asarray(clean_scores, dtype=float) * orientation
    corrupted = np.asarray(corrupted_scores, dtype=float) * orientation
    if not clean.size or not corrupted.size or not np.isfinite(clean).all() or not np.isfinite(corrupted).all():
        raise ValueError("AUROC needs non-empty finite clean and corrupted groups")
    ranks = rankdata(np.concatenate((clean, corrupted)), method="average")
    positives = corrupted.size
    rank_sum = float(ranks[clean.size:].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (clean.size * positives)


def _curves(rows, field: str) -> dict[str, dict[int, float]]:
    curves: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        image_id, severity = str(row["image_id"]), int(row["severity"])
        if severity in curves[image_id]:
            raise ValueError(f"duplicate score row for {image_id} severity {severity}")
        curves[image_id][severity] = float(row[field])
    for image_id, curve in curves.items():
        if set(curve) != set(SEVERITIES):
            raise ValueError(f"image {image_id} does not have severities 0 through 5")
    return dict(curves)


def summarize_series(rows, field: str, orientation: int) -> dict:
    curves = _curves(rows, field)
    if not curves:
        raise ValueError("series summary needs at least one complete image")
    ordered = sorted(curves)
    trends = [trend_metrics([curves[i][s] for s in SEVERITIES]) for i in ordered]
    oriented = [oriented_curve_metrics([curves[i][s] for s in SEVERITIES], orientation) for i in ordered]
    clean = [curves[i][0] for i in ordered]
    aurocs = {
        severity: binary_auroc(clean, [curves[i][severity] for i in ordered], orientation=orientation)
        for severity in SEVERITIES[1:]
    }
    severity_statistics = {}
    for severity in SEVERITIES:
        values = np.asarray([curves[i][severity] for i in ordered])
        q25, median, q75 = np.percentile(values, [25, 50, 75])
        severity_statistics[severity] = {
            "count": int(values.size), "mean": float(values.mean()),
            "median": float(median), "q25": float(q25), "q75": float(q75),
        }
    signed = [item["signed_spearman"] for item in trends]
    absolute = [item["absolute_spearman"] for item in trends]
    directions = [item["direction"] for item in trends]
    return {
        "field": field,
        "orientation": orientation,
        "image_count": len(ordered),
        "median_signed_spearman": float(np.median(signed)),
        "median_absolute_spearman": float(np.median(absolute)),
        "increasing_count": directions.count("increasing"),
        "decreasing_count": directions.count("decreasing"),
        "flat_count": directions.count("flat"),
        "oriented_adjacent_consistency": float(np.mean([x["adjacent_consistency"] for x in oriented])),
        "strongest_blur_above_clean_rate": float(
            np.mean([x["strongest_blur_above_clean"] for x in oriented])
        ),
        "auroc_by_severity": aurocs,
        "macro_auroc": float(np.mean(list(aurocs.values()))),
        "severity_statistics": severity_statistics,
    }
```

- [ ] **Step 4: Implement the paired image-level macro-AUROC bootstrap**

```python
# append to differential_uncertainty/evaluation.py
def _arrays(rows, field):
    curves = _curves(rows, field)
    ordered = sorted(curves)
    return ordered, {severity: np.asarray([curves[i][severity] for i in ordered]) for severity in SEVERITIES}


def _bootstrap_macro(columns, draws, orientation):
    count = draws.shape[1]
    clean = orientation * columns[0][draws]
    total = np.zeros(draws.shape[0], dtype=float)
    for severity in SEVERITIES[1:]:
        corrupted = orientation * columns[severity][draws]
        ranks = rankdata(np.concatenate((clean, corrupted), axis=1), method="average", axis=1)
        positive_rank_sum = ranks[:, count:].sum(axis=1)
        total += (positive_rank_sum - count * (count + 1) / 2) / (count * count)
    return total / 5


def paired_macro_bootstrap(
    rows,
    candidate_field: str,
    control_field: str,
    *,
    candidate_orientation: int,
    control_orientation: int,
    samples: int,
    seed: int,
) -> dict:
    candidate_ids, candidate = _arrays(rows, candidate_field)
    control_ids, control = _arrays(rows, control_field)
    if candidate_ids != control_ids:
        raise ValueError("candidate and control must cover the same image identities")
    count = len(candidate_ids)
    if count == 0 or samples <= 0:
        raise ValueError("bootstrap needs images and a positive sample count")
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, count, size=(samples, count))
    differences = _bootstrap_macro(candidate, draws, candidate_orientation) - _bootstrap_macro(
        control, draws, control_orientation
    )
    identity = np.arange(count, dtype=int)[None, :]
    point = float(
        _bootstrap_macro(candidate, identity, candidate_orientation)[0]
        - _bootstrap_macro(control, identity, control_orientation)[0]
    )
    low, high = np.percentile(differences, [2.5, 97.5])
    return {
        "candidate": candidate_field,
        "control": control_field,
        "candidate_orientation": candidate_orientation,
        "control_orientation": control_orientation,
        "point_difference": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "samples": samples,
        "seed": seed,
    }


def evaluate_rows(rows, config: ExperimentConfig = FIXED_CONFIG) -> dict:
    orientations = {
        "persistence_relative_gap": config.persistence_orientation,
        "confidence_relative_gap": config.confidence_orientation,
        "persistence_responsive": config.raw_responsive_orientation,
        "persistence_reference": config.raw_reference_orientation,
    }
    series = {field: summarize_series(rows, field, direction) for field, direction in orientations.items()}
    comparisons = [
        paired_macro_bootstrap(
            rows, "persistence_relative_gap", control,
            candidate_orientation=config.persistence_orientation,
            control_orientation=orientations[control],
            samples=config.bootstrap_samples, seed=config.bootstrap_seed,
        )
        for control in ("confidence_relative_gap", "persistence_responsive", "persistence_reference")
    ]
    return {"series": series, "bootstrap_comparisons": comparisons}
```

- [ ] **Step 5: Run the arithmetic tests**

```bash
$UE_PY -m pytest tests/differential_uncertainty/test_evaluation.py -q
```

Expected: all tests pass, including the exact zero-width paired self-comparison.

- [ ] **Step 6: Add archived relative-gap golden parity**

```python
# tests/differential_uncertainty/test_legacy_parity.py
import csv
from pathlib import Path

import pytest

from differential_uncertainty.evaluation import summarize_series


ARCHIVE = Path(
    "/home/yuchen/YuchenZ/UE/within-image-contrast-run-record/"
    "within_image_contrast/per_scene_contrasts.csv"
)


def _archived_rows(signal):
    with ARCHIVE.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    return [
        {"image_id": row["image_id"], "severity": int(row["severity"]), "score": float(row["score"])}
        for row in source
        if row["arm"] == "decile_90_100__50_60"
        and row["aggregation"] == "mean"
        and row["method"] == "relative_gap"
        and row["signal"] == signal
    ]


@pytest.mark.skipif(not ARCHIVE.is_file(), reason="archived completed run unavailable")
def test_persistence_relative_gap_reproduces_the_completed_run():
    result = summarize_series(_archived_rows("persistence"), "score", orientation=1)
    assert result["median_signed_spearman"] == pytest.approx(0.8285714285714287)
    assert result["auroc_by_severity"] == pytest.approx({
        1: 0.510112, 2: 0.565776, 3: 0.6668, 4: 0.893072, 5: 0.961344,
    })
    assert result["macro_auroc"] == pytest.approx(0.7194208)


@pytest.mark.skipif(not ARCHIVE.is_file(), reason="archived completed run unavailable")
def test_matched_confidence_relative_gap_reproduces_the_completed_run():
    result = summarize_series(_archived_rows("confidence"), "score", orientation=-1)
    assert result["median_signed_spearman"] == pytest.approx(-0.9142857142857144)
    assert result["auroc_by_severity"] == pytest.approx({
        1: 0.521424, 2: 0.56312, 3: 0.633056, 4: 0.78968, 5: 0.881872,
    })
    assert result["macro_auroc"] == pytest.approx(0.6778304)
```

- [ ] **Step 7: Run archived parity and commit**

```bash
$UE_PY -m pytest tests/differential_uncertainty/test_evaluation.py tests/differential_uncertainty/test_legacy_parity.py -q
```

Expected: both exact archived Spearman values and all ten per-severity AUROCs pass. Do not continue to pruning if the archive is present and either test fails.

```bash
git add differential_uncertainty/evaluation.py tests/differential_uncertainty/test_evaluation.py tests/differential_uncertainty/test_legacy_parity.py
git commit -m "feat: evaluate differential corruption scores"
```

## Task 9: Publish the detailed report bundle

**Files:**
- Create: `differential_uncertainty/reporting.py`
- Test: `tests/differential_uncertainty/test_reporting.py`

- [ ] **Step 1: Write a failing exact-bundle and plain-language report test**

```python
# tests/differential_uncertainty/test_reporting.py
import json

import pandas as pd

from differential_uncertainty.config import ExperimentConfig
from differential_uncertainty.evaluation import evaluate_rows
from differential_uncertainty.reporting import REPORT_FILES, write_report


def _rows():
    rows = []
    for image_id, offset in (("a", 0.0), ("b", 0.2), ("c", -0.1)):
        for severity in range(6):
            persistence_reference = 2.0 - 0.1 * severity + offset
            persistence_responsive = 1.0 + 0.2 * severity + offset
            confidence_reference = 0.2 + 0.05 * severity
            confidence_responsive = 0.8 - 0.05 * severity
            rows.append({
                "image_id": image_id, "severity": severity,
                "padded_count": 0, "valid_count": 20,
                "reference_count": 2, "responsive_count": 2,
                "persistence_reference": persistence_reference,
                "persistence_responsive": persistence_responsive,
                "persistence_relative_gap": 2 * (persistence_responsive - persistence_reference)
                / (persistence_responsive + persistence_reference),
                "confidence_reference": confidence_reference,
                "confidence_responsive": confidence_responsive,
                "confidence_relative_gap": 2 * (confidence_responsive - confidence_reference)
                / (confidence_responsive + confidence_reference),
            })
    return rows


def test_report_writes_the_exact_dedicated_bundle(tmp_path):
    config = ExperimentConfig.for_tests(
        bank_capacity=20, k=5, query_count=20, persistence_dim=7,
        bootstrap_samples=100,
    )
    rows = _rows()
    evaluation = evaluate_rows(rows, config)
    output = tmp_path / "report"
    provenance = {
        "checkpoint_sha256": "abc",
        "config": config.scientific_dict(),
        "corruption": {
            "name": "gaussian_blur",
            "severities": [
                {"level": level, "parameter": radius}
                for level, radius in enumerate(config.blur_radii)
            ],
        },
    }
    write_report(output, rows, evaluation, provenance)
    actual = sorted(str(path.relative_to(output)) for path in output.rglob("*") if path.is_file())
    assert actual == sorted(REPORT_FILES)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["evaluation"]["series"]["persistence_relative_gap"]["orientation"] == 1
    text = (output / "report.md").read_text(encoding="utf-8").lower()
    for phrase in (
        "what was tested", "relative gap", "nearest clean", "sigmoid", "spearman",
        "3.5 / 4 = 0.875", "not a probability", "adjacent consistency",
        "paired bootstrap", "does not measure map",
    ):
        assert phrase in text
```

- [ ] **Step 2: Run the test and verify the missing-module failure**

```bash
$UE_PY -m pytest tests/differential_uncertainty/test_reporting.py -q
```

Expected: collection fails because `differential_uncertainty.reporting` does not exist.

- [ ] **Step 3: Implement tabular outputs and atomic report publication**

```python
# differential_uncertainty/reporting.py (writer and tables)
from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPORT_FILES = (
    "per-image-scores.csv",
    "metrics.csv",
    "bootstrap-comparisons.csv",
    "summary.json",
    "report.md",
    "figures/reference-and-responsive-distance.png",
    "figures/clean-reference-relationship.png",
    "figures/relative-gap-by-severity.png",
    "figures/auroc-by-corruption-severity.png",
)


def _metric_rows(evaluation):
    rows = []
    for name, values in evaluation["series"].items():
        row = {key: value for key, value in values.items() if key not in {"auroc_by_severity", "severity_statistics"}}
        row["series"] = name
        row.update({f"auroc_severity_{level}": score for level, score in values["auroc_by_severity"].items()})
        rows.append(row)
    return rows


def write_report(output, rows, evaluation, provenance) -> None:
    output = Path(output)
    if output.exists():
        existing = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        if existing.get("provenance") != provenance:
            raise ValueError("existing report provenance differs; choose a new output directory")
        return
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    figures = staging / "figures"
    figures.mkdir(parents=True)
    try:
        frame = pd.DataFrame(rows).sort_values(["image_id", "severity"])
        frame.to_csv(staging / "per-image-scores.csv", index=False)
        pd.DataFrame(_metric_rows(evaluation)).to_csv(staging / "metrics.csv", index=False)
        pd.DataFrame(evaluation["bootstrap_comparisons"]).to_csv(
            staging / "bootstrap-comparisons.csv", index=False
        )
        summary = {"schema_version": 1, "provenance": provenance, "evaluation": evaluation}
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_figures(frame, evaluation, figures)
        (staging / "report.md").write_text(
            render_report(frame, evaluation, provenance), encoding="utf-8"
        )
        observed = sorted(str(path.relative_to(staging)) for path in staging.rglob("*") if path.is_file())
        if observed != sorted(REPORT_FILES):
            raise RuntimeError(f"report bundle is incomplete: {observed}")
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
```

- [ ] **Step 4: Implement the four fixed detailed figures**

```python
# append to differential_uncertainty/reporting.py
def _median_curve(frame, field):
    return frame.groupby("severity")[field].median().reindex(range(6)).to_numpy()


def _write_figures(frame, evaluation, directory):
    severity = np.arange(6)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, prefix, title in (
        (axes[0], "persistence", "Layer-2 distance"),
        (axes[1], "confidence", "1 - maximum sigmoid confidence"),
    ):
        axis.plot(severity, _median_curve(frame, f"{prefix}_reference"), marker="o", label="90-100% reference")
        axis.plot(severity, _median_curve(frame, f"{prefix}_responsive"), marker="o", label="50-60% responsive")
        axis.set(title=title, xlabel="corruption level", ylabel="median group mean")
        axis.legend()
    fig.savefig(directory / "reference-and-responsive-distance.png", dpi=180)
    plt.close(fig)

    clean = frame[frame["severity"] == 0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].scatter(clean["persistence_reference"], clean["persistence_responsive"], s=14)
    axes[1].scatter(clean["confidence_reference"], clean["confidence_responsive"], s=14)
    for axis, title in zip(axes, ("Persistence on clean images", "Confidence uncertainty on clean images")):
        axis.set(title=title, xlabel="reference mean", ylabel="responsive mean")
    fig.savefig(directory / "clean-reference-relationship.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, field, title in (
        (axes[0], "persistence_relative_gap", "Persistence relative gap"),
        (axes[1], "confidence_relative_gap", "Matched confidence relative gap"),
    ):
        groups = [frame.loc[frame["severity"] == level, field].to_numpy() for level in range(6)]
        axis.boxplot(groups, positions=severity, showfliers=False)
        axis.set(title=title, xlabel="corruption level", ylabel="relative gap")
    fig.savefig(directory / "relative-gap-by-severity.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    labels = {
        "persistence_relative_gap": "persistence relative gap",
        "confidence_relative_gap": "matched confidence",
        "persistence_responsive": "raw responsive control",
        "persistence_reference": "raw reference control",
    }
    for name, label in labels.items():
        values = evaluation["series"][name]["auroc_by_severity"]
        axis.plot(list(values), list(values.values()), marker="o", label=label)
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set(
        xlabel="corruption level", ylabel="AUROC", ylim=(0, 1),
        title="Clean versus corrupted ranking",
    )
    axis.legend()
    fig.savefig(directory / "auroc-by-corruption-severity.png", dpi=180)
    plt.close(fig)
```

- [ ] **Step 5: Implement the plain-language Markdown with concrete run numbers**

````python
# append to differential_uncertainty/reporting.py
def render_report(frame, evaluation, provenance):
    primary = evaluation["series"]["persistence_relative_gap"]
    confidence = evaluation["series"]["confidence_relative_gap"]
    example = frame.sort_values(["image_id", "severity"]).iloc[0]
    r = example["persistence_reference"]
    m = example["persistence_responsive"]
    gap = example["persistence_relative_gap"]
    confidence_r = example["confidence_reference"]
    confidence_m = example["confidence_responsive"]
    confidence_gap = example["confidence_relative_gap"]
    config = provenance["config"]
    corruption_name = provenance["corruption"]["name"].replace("_", " ")
    auroc_lines = "\n".join(
        f"| {level} | {primary['auroc_by_severity'][level]:.3f} | "
        f"{confidence['auroc_by_severity'][level]:.3f} |"
        for level in range(1, 6)
    )
    bootstrap_lines = "\n".join(
        f"| {item['control'].replace('_', ' ')} | {item['point_difference']:+.3f} | "
        f"[{item['ci_low']:+.3f}, {item['ci_high']:+.3f}] |"
        for item in evaluation["bootstrap_comparisons"]
    )
    return f"""# Differential corruption uncertainty report

## What was tested

We tested **{corruption_name}** at six ordered levels, where level 0 is the clean image.
The detector made {config['query_count']} query guesses for each image. Exact repeated padded guesses were
removed. The remaining guesses were ranked by maximum **sigmoid** class confidence. We compared
the middle 50-60% group with the highest 90-100% group at decoder layer
{config['persistence_layer']}.

For every selected query, we found its **{config['k']} nearest clean** bank fingerprints and averaged
their distances. Then we averaged the query distances inside each group.

Here is a small kNN example. If one query's five nearest distances are 0.10, 0.14, 0.17,
0.21, and 0.28, its score is `(0.10 + 0.14 + 0.17 + 0.21 + 0.28) / 5 = 0.18`.

## A concrete relative-gap example

For image `{example['image_id']}` at blur level {int(example['severity'])}, the reference mean
was {r:.6f} and the responsive mean was {m:.6f}:

```text
2 x ({m:.6f} - {r:.6f}) / ({m:.6f} + {r:.6f}) = {gap:.6f}
```

This **relative gap** is scale independent: multiplying both inputs by the same number leaves
the answer unchanged. We chose it for that interpretation. The earlier run did not prove that
it was better than a raw subtraction.

## A concrete matched-confidence example

The confidence baseline uses the **same query IDs**. A query whose highest sigmoid class
confidence is 0.90 becomes uncertainty `1 - 0.90 = 0.10`. For example, confidences 0.90 and
0.80 give a mean uncertainty of 0.15, while confidences 0.60 and 0.50 give 0.45. Their relative
gap is `2 x (0.45 - 0.15) / (0.45 + 0.15) = 1.00`.

In this run's concrete row, the matched reference and responsive means were
{confidence_r:.6f} and {confidence_m:.6f}:

```text
2 x ({confidence_m:.6f} - {confidence_r:.6f})
    / ({confidence_m:.6f} + {confidence_r:.6f}) = {confidence_gap:.6f}
```

This is sometimes called “plain softmax” in conversation, but it is **not softmax**: the code
uses the maximum sigmoid class confidence. Its direction is frozen at -1 from the archived
study, so a smaller raw confidence gap ranks as more corrupted.

## Result

| Corruption level | Persistence AUROC | Matched confidence AUROC |
|---|---:|---:|
{auroc_lines}
| Average | {primary['macro_auroc']:.3f} | {confidence['macro_auroc']:.3f} |

Persistence median signed **Spearman** was {primary['median_signed_spearman']:+.3f}. Spearman
asks whether one image's six scores move in order as corruption grows. For a tiny example,
levels `[0, 1, 2, 3, 4, 5]` and scores `[10, 20, 30, 40, 50, 60]` have the same ranks, so
Spearman is +1. Reversing the score order gives -1; a completely flat curve is recorded as 0.

AUROC asks a different question: whether corrupted images rank above clean images across the
whole group. With clean scores `[0, 1]` and corrupted scores `[1, 2]`, three pairs are wins and
one is a tie worth half a win, so AUROC is `3.5 / 4 = 0.875`. AUROC is **not a probability**
that one image is corrupted. A value of 0.5 is chance ranking, and 1.0 is perfect ranking.

After applying its frozen direction, the persistence score moved the right way on
{primary['oriented_adjacent_consistency']:.1%} of adjacent steps. Its strongest-corruption
score was strictly above its clean score for {primary['strongest_blur_above_clean_rate']:.1%}
of images.

## Paired bootstrap comparisons

| Control | Persistence macro-AUROC minus control | 95% interval |
|---|---:|---:|
{bootstrap_lines}

The paired bootstrap redraws whole image IDs {config['bootstrap_samples']:,} times. Each redraw
keeps all six levels from an image together. The interval describes uncertainty on this supplied
set; it does not turn AUROC into a probability and it does not prove a universal effect.

## What this does not prove

The workflow has no object labels, so it **does not measure mAP**, detection accuracy, or
calibration. The bins, layer, formula, and directions came from earlier blur tuning. Fresh
numbers may differ from the historical run because this clean workflow also removes padded
queries from the reference bank.

Reproducibility details, complete rows, bootstrap comparisons, and figure data are stored beside
this report. Checkpoint SHA-256: `{provenance['checkpoint_sha256']}`.
"""
````

- [ ] **Step 6: Run reporting tests, inspect every generated PNG, and commit**

```bash
$UE_PY -m pytest tests/differential_uncertainty/test_reporting.py -q --basetemp=/tmp/differential-report-check
find /tmp/differential-report-check -name '*.png' -type f | sort
```

Expected: exact nine-file bundle test passes and `find` lists four PNGs. Open each listed image
through the local image viewer during implementation and verify titles, legends, axes, tick
labels, and lines or boxes are legible and not clipped.

```bash
git add differential_uncertainty/reporting.py tests/differential_uncertainty/test_reporting.py
git commit -m "feat: publish the detailed differential report"
```

## Task 10: Orchestrate the complete resumable run

**Files:**
- Modify: `differential_uncertainty/extraction.py`
- Create: `differential_uncertainty/cli.py`
- Create: `differential_uncertainty/__main__.py`
- Test: `tests/differential_uncertainty/test_pipeline.py`
- Test: `tests/differential_uncertainty/test_cli.py`

- [ ] **Step 1: Write a failing end-to-end test with a deterministic fake extractor**

```python
# tests/differential_uncertainty/test_pipeline.py
from contextlib import contextmanager
from pathlib import Path

import torch
from PIL import Image

from differential_uncertainty.cli import run_pipeline
from differential_uncertainty.config import ExperimentConfig


class FakeExtractor:
    calls = 0

    def __init__(self, _checkpoint, _device, config):
        self.config = config

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def extract_batch(self, identities, samples):
        type(self).calls += 1
        records = []
        for (image_id, severity), sample in zip(identities, samples):
            query = torch.arange(self.config.query_count, dtype=torch.float32)
            image_value = float(sample.mean())
            boxes = torch.stack((query, query + 1, query + 2, query + 3), dim=1)
            logits = query[:, None].repeat(1, 80) / 10 + image_value - severity / 20
            persistence = torch.stack(
                [query + image_value + severity * (query / self.config.query_count) ** power
                 for power in range(1, self.config.persistence_dim + 1)], dim=1
            )
            for field in (boxes, logits, persistence):
                field[-2] = field[-1]
            records.append({
                "image_id": image_id, "severity": severity,
                "boxes": boxes, "logits": logits, "persistence": persistence,
            })
        return records


def _manifest(root, name, entries):
    lines = ["image_id,image_path"]
    for image_id, color in entries:
        path = root / f"{image_id}.png"
        Image.new("RGB", (17, 13), (color, color, color)).save(path)
        lines.append(f"{image_id},{path.name}")
    manifest = root / name
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def test_pipeline_builds_every_stage_and_second_run_reuses_all_artifacts(tmp_path):
    reference = _manifest(tmp_path, "reference.csv", (("r1", 10), ("r2", 30)))
    evaluation = _manifest(tmp_path, "evaluation.csv", (("e1", 60), ("e2", 90), ("e3", 120)))
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fake checkpoint content")
    output = tmp_path / "run"
    config = ExperimentConfig.for_tests(
        bank_capacity=20, k=2, query_count=20, persistence_dim=7,
        bootstrap_samples=100,
    )
    FakeExtractor.calls = 0
    run_pipeline(
        reference, evaluation, checkpoint, output,
        device="cpu", batch_size=2, shard_size=2,
        config=config, extractor_factory=FakeExtractor,
    )
    first_calls = FakeExtractor.calls
    assert first_calls > 0
    assert (output / "artifacts" / "provenance.json").is_file()
    assert (output / "artifacts" / "reference-bank.pt").is_file()
    assert (output / "artifacts" / "scores.csv").is_file()
    assert (output / "report" / "report.md").is_file()
    run_pipeline(
        reference, evaluation, checkpoint, output,
        device="cpu", batch_size=2, shard_size=2,
        config=config, extractor_factory=FakeExtractor,
    )
    assert FakeExtractor.calls == first_calls


def test_changed_checkpoint_refuses_the_existing_run(tmp_path):
    reference = _manifest(tmp_path, "reference.csv", (("r1", 10), ("r2", 30)))
    evaluation = _manifest(tmp_path, "evaluation.csv", (("e1", 60), ("e2", 90)))
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"first")
    output = tmp_path / "run"
    config = ExperimentConfig.for_tests(
        bank_capacity=20, k=2, query_count=20, persistence_dim=7,
        bootstrap_samples=20,
    )
    run_pipeline(reference, evaluation, checkpoint, output, device="cpu", batch_size=2,
                 shard_size=2, config=config, extractor_factory=FakeExtractor)
    checkpoint.write_bytes(b"second")
    import pytest
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        run_pipeline(reference, evaluation, checkpoint, output, device="cpu", batch_size=2,
                     shard_size=2, config=config, extractor_factory=FakeExtractor)
```

- [ ] **Step 2: Run the end-to-end test and verify `run_pipeline` is missing**

```bash
$UE_PY -m pytest tests/differential_uncertainty/test_pipeline.py -q
```

Expected: collection fails because `differential_uncertainty.cli.run_pipeline` does not exist.

- [ ] **Step 3: Implement run provenance and stage helpers**

```python
# differential_uncertainty/cli.py (provenance and stage helpers)
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

from .artifacts import ensure_provenance, iter_records, sha256_file, source_digest
from .bank import build_reference_bank, load_reference_bank, save_reference_bank
from .config import ExperimentConfig, FIXED_CONFIG
from .corruptions.gaussian_blur import GaussianBlur
from .evaluation import evaluate_rows
from .extraction import RTDETRExtractor, extract_manifest
from .manifests import load_manifest, manifest_digest, validate_disjoint
from .reporting import write_report
from .scoring import score_image_records


def _source_files() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    workflow = list((root / "differential_uncertainty").rglob("*.py"))
    detector = [
        root / name for name in (
            "src/__init__.py", "src/nn/__init__.py", "src/nn/backbone/__init__.py",
            "src/nn/backbone/common.py", "src/nn/backbone/presnet.py",
            "src/zoo/__init__.py", "src/zoo/rtdetr/__init__.py",
            "src/zoo/rtdetr/box_ops.py", "src/zoo/rtdetr/denoising.py",
            "src/zoo/rtdetr/hybrid_encoder.py", "src/zoo/rtdetr/rtdetr.py",
            "src/zoo/rtdetr/rtdetrv2_decoder.py", "src/zoo/rtdetr/utils.py",
        )
    ]
    return workflow + detector


def _provenance(reference, evaluation, checkpoint, config, corruption):
    levels = [severity.level for severity in corruption.severities]
    if levels != list(range(6)):
        raise ValueError("the fixed evaluator needs corruption levels 0 through 5")
    return {
        "schema_version": 1,
        "reference_manifest_sha256": manifest_digest(reference),
        "evaluation_manifest_sha256": manifest_digest(evaluation),
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_sha256": source_digest(
            _source_files(), root=Path(__file__).resolve().parents[1]
        ),
        "config": config.scientific_dict(),
        "corruption": {
            "name": corruption.name,
            "severities": [
                {"level": severity.level, "parameter": severity.parameter}
                for severity in corruption.severities
            ],
        },
    }


def _atomic_score_csv(rows, path):
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    pd.DataFrame(rows).sort_values(["image_id", "severity"]).to_csv(temporary, index=False)
    os.replace(temporary, target)


def _load_score_csv(path):
    frame = pd.read_csv(path, dtype={"image_id": str})
    return frame.to_dict(orient="records")
```

- [ ] **Step 4: Implement the dependency-injected end-to-end orchestration**

```python
# append to differential_uncertainty/cli.py
def run_pipeline(
    reference_manifest,
    evaluation_manifest,
    checkpoint,
    output_dir,
    *,
    device: str,
    batch_size: int,
    shard_size: int,
    config: ExperimentConfig = FIXED_CONFIG,
    extractor_factory=RTDETRExtractor,
) -> Path:
    if batch_size <= 0 or shard_size <= 0:
        raise ValueError("batch-size and shard-size must be positive")
    reference = load_manifest(reference_manifest)
    evaluation = load_manifest(evaluation_manifest)
    validate_disjoint(reference, evaluation)
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint}")
    output = Path(output_dir).resolve()
    corruption = GaussianBlur()
    provenance = _provenance(reference, evaluation, checkpoint, config, corruption)
    ensure_provenance(output, provenance)
    artifacts = output / "artifacts"
    reference_cache = artifacts / "reference-extractions"
    evaluation_cache = artifacts / "evaluation-extractions"
    needs_extraction = not (reference_cache / "manifest.json").exists() or not (
        evaluation_cache / "manifest.json"
    ).exists()
    if needs_extraction:
        with extractor_factory(checkpoint, torch.device(device), config) as extractor:
            extract_manifest(
                reference, reference_cache,
                {"stage": "reference", "input_id": provenance["reference_manifest_sha256"],
                 "checkpoint_sha256": provenance["checkpoint_sha256"]},
                extractor, None, image_size=config.image_size,
                batch_size=batch_size, shard_size=shard_size,
            )
            extract_manifest(
                evaluation, evaluation_cache,
                {"stage": "evaluation", "input_id": provenance["evaluation_manifest_sha256"],
                 "checkpoint_sha256": provenance["checkpoint_sha256"],
                 "corruption": provenance["corruption"]},
                extractor, corruption, image_size=config.image_size,
                batch_size=batch_size, shard_size=shard_size,
            )
    bank_path = artifacts / "reference-bank.pt"
    if not bank_path.exists():
        bank = build_reference_bank(iter_records(reference_cache), config)
        save_reference_bank(bank, bank_path, {
            "reference_manifest_sha256": provenance["reference_manifest_sha256"],
            "capacity": config.bank_capacity, "seed": config.bank_seed,
            "padding_removed": True,
        })
    bank, bank_metadata = load_reference_bank(bank_path)
    if bank_metadata["reference_manifest_sha256"] != provenance["reference_manifest_sha256"]:
        raise ValueError("reference bank provenance does not match this run")
    score_path = artifacts / "scores.csv"
    if score_path.exists():
        rows = _load_score_csv(score_path)
    else:
        grouped = defaultdict(list)
        for record in iter_records(evaluation_cache):
            grouped[str(record["image_id"])].append(record)
        expected_ids = [entry.image_id for entry in evaluation]
        if sorted(grouped) != sorted(expected_ids):
            raise RuntimeError("evaluation cache image roster is incomplete")
        rows = [
            row
            for image_id in sorted(grouped)
            for row in score_image_records(grouped[image_id], bank, config)
        ]
        _atomic_score_csv(rows, score_path)
    evaluation_summary = evaluate_rows(rows, config)
    write_report(output / "report", rows, evaluation_summary, provenance)
    return output
```

- [ ] **Step 5: Run end-to-end and resume tests**

```bash
$UE_PY -m pytest tests/differential_uncertainty/test_pipeline.py -q
```

Expected: both tests pass, the second identical invocation performs no extraction, and the changed checkpoint is refused before any stage is reused.

- [ ] **Step 6: Write failing CLI surface tests**

```python
# tests/differential_uncertainty/test_cli.py
from differential_uncertainty.cli import build_parser, main


def test_cli_has_one_run_command_and_no_scientific_switches(capsys):
    parser = build_parser()
    help_text = parser.format_help()
    assert "run" in help_text
    run = next(action for action in parser._subparsers._group_actions).choices["run"]
    options = {option for action in run._actions for option in action.option_strings}
    assert {"--reference-manifest", "--evaluation-manifest", "--checkpoint", "--output-dir"} <= options
    assert not {"--k", "--layer", "--bins", "--score", "--orientation", "--blur-radii"} & options


def test_cli_reports_plain_errors_without_a_traceback(capsys, tmp_path):
    code = main([
        "run", "--reference-manifest", str(tmp_path / "missing.csv"),
        "--evaluation-manifest", str(tmp_path / "missing-too.csv"),
        "--checkpoint", str(tmp_path / "missing.pth"),
        "--output-dir", str(tmp_path / "run"),
    ])
    assert code == 2
    assert "error:" in capsys.readouterr().err
```

- [ ] **Step 7: Implement the one-command parser and module entry points**

```python
# append to differential_uncertainty/cli.py
def _positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description="Fixed differential corruption uncertainty")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run or resume the complete fixed workflow")
    run.add_argument("--reference-manifest", required=True)
    run.add_argument("--evaluation-manifest", required=True)
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--batch-size", type=_positive, default=1)
    run.add_argument("--shard-size", type=_positive, default=50)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        run_pipeline(
            args.reference_manifest, args.evaluation_manifest, args.checkpoint,
            args.output_dir, device=args.device, batch_size=args.batch_size,
            shard_size=args.shard_size,
        )
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

```python
# differential_uncertainty/__main__.py
from .cli import main

raise SystemExit(main())
```

- [ ] **Step 8: Run the workflow and CLI tests, then commit**

```bash
$UE_PY -m pytest tests/differential_uncertainty/test_pipeline.py tests/differential_uncertainty/test_cli.py -q
$UE_PY -m differential_uncertainty --help
$UE_PY -m differential_uncertainty.cli --help
```

Expected: tests pass and both help commands show only the `run` workflow plus runtime controls.

```bash
git add differential_uncertainty/cli.py differential_uncertainty/__main__.py tests/differential_uncertainty/test_pipeline.py tests/differential_uncertainty/test_cli.py
git commit -m "feat: run differential uncertainty end to end"
```

## Task 11: Freeze detector parity, then remove registry coupling

**Files:**

- Create temporarily: `tests/differential_uncertainty/record_detector_golden.py`
- Create: `tests/differential_uncertainty/fixtures/rtdetrv2_r18_layer2_golden.pt`
- Modify: `tests/differential_uncertainty/test_detector_parity.py`
- Modify: `src/__init__.py`
- Modify: `src/nn/__init__.py`
- Modify: `src/nn/backbone/__init__.py`
- Modify: `src/nn/backbone/presnet.py`
- Modify: `src/zoo/__init__.py`
- Modify: `src/zoo/rtdetr/__init__.py`
- Modify: `src/zoo/rtdetr/rtdetr.py`
- Modify: `src/zoo/rtdetr/hybrid_encoder.py`
- Modify: `src/zoo/rtdetr/rtdetrv2_decoder.py`

- [ ] **Step 1: Re-run live old-versus-new parity before removing any old path**

```bash
export UE_PY=/home/yuchen/miniconda3/envs/UE/bin/python
$UE_PY -m pytest tests/differential_uncertainty/test_detector_parity.py -q -rs
```

Expected: the test passes; it must neither skip nor fail while the legacy builder still exists.

- [ ] **Step 2: Record a deterministic golden fixture from the already-proven new path**

```python
# tests/differential_uncertainty/record_detector_golden.py
from pathlib import Path

import torch

from differential_uncertainty.extraction import load_frozen_detector
from differential_uncertainty.persistence import Layer2Capture, batched_persistence

CHECKPOINT = Path(
    "/home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/"
    "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
)
OUTPUT = Path("tests/differential_uncertainty/fixtures/rtdetrv2_r18_layer2_golden.pt")


def main():
    if not CHECKPOINT.is_file():
        raise SystemExit(f"checkpoint not found: {CHECKPOINT}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(19)
    sample = torch.rand((1, 3, 640, 640), generator=generator).to(device)
    model = load_frozen_detector(CHECKPOINT, device)
    with torch.inference_mode(), Layer2Capture(model.decoder, 2) as capture:
        outputs = model(sample)
        features, weight = capture.take()
        persistence = batched_persistence(
            weight, features.reshape(-1, features.shape[-1])
        ).reshape(1, 300, 335)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_seed": 19,
            "checkpoint_name": CHECKPOINT.name,
            "device_type": device.type,
            "pred_logits": outputs["pred_logits"].cpu(),
            "pred_boxes": outputs["pred_boxes"].cpu(),
            "layer_2_persistence": persistence.cpu(),
        },
        OUTPUT,
    )


if __name__ == "__main__":
    main()
```

```bash
$UE_PY tests/differential_uncertainty/record_detector_golden.py
test -s tests/differential_uncertainty/fixtures/rtdetrv2_r18_layer2_golden.pt
```

Expected: the fixture exists and contains logits, boxes, and layer-2 persistence for the fixed seeded tensor.

- [ ] **Step 3: Replace legacy-builder parity with a self-contained golden test**

```python
# replace tests/differential_uncertainty/test_detector_parity.py
from pathlib import Path

import pytest
import torch

from differential_uncertainty.extraction import build_fixed_detector, load_frozen_detector
from differential_uncertainty.persistence import Layer2Capture, batched_persistence

CHECKPOINT = Path(
    "/home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/"
    "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
)
GOLDEN = Path(__file__).parent / "fixtures" / "rtdetrv2_r18_layer2_golden.pt"


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="local pretrained checkpoint absent")
def test_fixed_builder_matches_recorded_detector_outputs_exactly():
    golden = torch.load(GOLDEN, map_location="cpu", weights_only=True)
    assert golden["checkpoint_name"] == CHECKPOINT.name
    generator = torch.Generator().manual_seed(golden["input_seed"])
    sample_cpu = torch.rand((1, 3, 640, 640), generator=generator)
    if golden["device_type"] == "cuda" and not torch.cuda.is_available():
        pytest.skip("golden fixture was recorded on CUDA, which is unavailable")
    device = torch.device(golden["device_type"])
    model = load_frozen_detector(CHECKPOINT, device)
    with torch.inference_mode(), Layer2Capture(model.decoder, 2) as capture:
        outputs = model(sample_cpu.to(device))
        features, weight = capture.take()
        persistence = batched_persistence(
            weight, features.reshape(-1, features.shape[-1])
        ).reshape(1, 300, 335)
    torch.testing.assert_close(outputs["pred_logits"].cpu(), golden["pred_logits"], rtol=0, atol=0)
    torch.testing.assert_close(outputs["pred_boxes"].cpu(), golden["pred_boxes"], rtol=0, atol=0)
    torch.testing.assert_close(persistence.cpu(), golden["layer_2_persistence"], rtol=0, atol=0)
```

```bash
$UE_PY -m pytest tests/differential_uncertainty/test_detector_parity.py -q -rs
git rm tests/differential_uncertainty/record_detector_golden.py
```

Expected: exact golden parity passes, then the one-use recording script is deleted so normal operation cannot rewrite the oracle.

- [ ] **Step 4: Narrow initializers and remove the unused registration decorators**

Use these exact public surfaces:

```python
# src/__init__.py
"""Minimal RT-DETRv2 inference components retained for differential uncertainty."""
```

```python
# src/nn/__init__.py
from .backbone.presnet import PResNet

__all__ = ["PResNet"]
```

```python
# src/nn/backbone/__init__.py
from .common import FrozenBatchNorm2d
from .presnet import PResNet

__all__ = ["FrozenBatchNorm2d", "PResNet"]
```

```python
# src/zoo/__init__.py
from .rtdetr import HybridEncoder, RTDETR, RTDETRTransformerv2

__all__ = ["HybridEncoder", "RTDETR", "RTDETRTransformerv2"]
```

```python
# src/zoo/rtdetr/__init__.py
from .hybrid_encoder import HybridEncoder
from .rtdetr import RTDETR
from .rtdetrv2_decoder import RTDETRTransformerv2

__all__ = ["HybridEncoder", "RTDETR", "RTDETRTransformerv2"]
```

Delete only `from ...core import register` (or its equivalent relative import) and `@register()` from these four files:

```text
src/nn/backbone/presnet.py
src/zoo/rtdetr/rtdetr.py
src/zoo/rtdetr/hybrid_encoder.py
src/zoo/rtdetr/rtdetrv2_decoder.py
```

- [ ] **Step 5: Replace YAML/registry structure coverage with fixed checkpoint-shape coverage**

Add this assertion to `test_detector_parity.py`:

```python
def test_fixed_builder_has_the_expected_checkpoint_shapes():
    model = build_fixed_detector()
    shapes = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    assert shapes["backbone.conv1.conv1_1.conv.weight"] == (32, 3, 3, 3)
    assert shapes["encoder.input_proj.0.conv.weight"] == (256, 128, 1, 1)
    assert shapes["decoder.dec_score_head.2.weight"] == (80, 256)
    assert shapes["decoder.denoising_class_embed.weight"] == (81, 256)
```

These exact keys and shapes were checked against the current YAML-built RT-DETRv2-R18 model on
2026-08-23; treat any mismatch as a builder regression rather than changing the assertion.

```bash
$UE_PY -m pytest tests/differential_uncertainty/test_detector_parity.py tests/differential_uncertainty/test_extraction.py -q -rs
```

Expected: fixed architecture, real checkpoint, logits, boxes, and layer-2 persistence are all covered without `src.core` or YAML configuration.

- [ ] **Step 6: Commit the isolated detector path**

```bash
git add src tests/differential_uncertainty/test_detector_parity.py tests/differential_uncertainty/fixtures/rtdetrv2_r18_layer2_golden.pt
git commit -m "refactor: isolate the fixed detector inference path"
```

## Task 12: Prove that later corruption types are plugins, not rewrites

**Files:**

- Modify: `tests/differential_uncertainty/test_pipeline.py`
- Modify: `differential_uncertainty/pipeline.py`

- [ ] **Step 1: Write a failing end-to-end test with a non-blur corruption**

```python
# add to tests/differential_uncertainty/test_pipeline.py
import csv
import json

from PIL import ImageOps

from differential_uncertainty.corruptions import Corruption, Severity


class InvertCorruption(Corruption):
    name = "invert"
    severities = tuple(Severity(level, float(level)) for level in range(6))

    def apply(self, image, level):
        if level not in range(6):
            raise ValueError(f"unknown invert severity {level}")
        return image.copy() if level == 0 else ImageOps.invert(image.convert("RGB"))


def test_pipeline_accepts_a_corruption_plugin_without_changing_scoring(tmp_path):
    reference = _manifest(tmp_path, "reference.csv", (("r1", 10), ("r2", 30)))
    evaluation = _manifest(tmp_path, "evaluation.csv", (("e1", 60), ("e2", 90)))
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fake checkpoint content")
    output = tmp_path / "invert-run"
    config = ExperimentConfig.for_tests(
        bank_capacity=20, k=2, query_count=20, persistence_dim=7,
        bootstrap_samples=20,
    )
    run_pipeline(
        reference, evaluation, checkpoint, output,
        device="cpu", batch_size=2, shard_size=2, config=config,
        extractor_factory=FakeExtractor, corruption=InvertCorruption(),
    )
    provenance = json.loads((output / "artifacts" / "provenance.json").read_text())
    assert provenance["corruption"]["name"] == "invert"
    assert [item["level"] for item in provenance["corruption"]["severities"]] == list(range(6))
    rows = list(csv.DictReader((output / "artifacts" / "scores.csv").open()))
    assert {int(row["severity"]) for row in rows} == set(range(6))
```

```bash
$UE_PY -m pytest tests/differential_uncertainty/test_pipeline.py::test_pipeline_accepts_a_corruption_plugin_without_changing_scoring -q
```

Expected: FAIL with an unexpected `corruption` argument; the first shipped command still fixes
Gaussian blur internally.

- [ ] **Step 2: Inject the plugin and record its identity in provenance**

Expose the already-generic internal corruption object as a Python-only dependency-injection
point. Change the pipeline signature and setup to:

```python
def run_pipeline(
    reference_manifest, evaluation_manifest, checkpoint, output_dir,
    *, device: str, batch_size: int, shard_size: int,
    config: ExperimentConfig = FIXED_CONFIG,
    extractor_factory=RTDETRExtractor, corruption=None,
):
    corruption = corruption or GaussianBlur()
    provenance = _provenance(reference, evaluation, checkpoint, config, corruption)
```

Keep passing `corruption` to `extract_manifest`. Do not add an invert-specific branch anywhere
outside the test plugin, and do not expose a corruption selector in the initial CLI.

- [ ] **Step 3: Run plugin, default-blur, and CLI tests**

```bash
$UE_PY -m pytest tests/differential_uncertainty/test_corruptions.py tests/differential_uncertainty/test_pipeline.py tests/differential_uncertainty/test_cli.py -q
```

Expected: every test passes; the default remains Gaussian blur at radii 0, 1, 2, 4, 8, and 12.

- [ ] **Step 4: Commit the extension point**

```bash
git add differential_uncertainty/pipeline.py tests/differential_uncertainty/test_pipeline.py
git commit -m "test: prove corruption plugins compose end to end"
```

## Task 13: Prune the repository to the fixed inference experiment

**Files:**

- Create: `tests/differential_uncertainty/test_repository_surface.py`
- Modify: `requirements.txt`
- Modify: `.gitignore`
- Delete: legacy training, dataset, solver, registry, old uncertainty, and unused detector files listed below

- [ ] **Step 1: Add an executable size-and-surface guard**

```python
# tests/differential_uncertainty/test_repository_surface.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RETAINED_SRC = {
    "src/__init__.py",
    "src/nn/__init__.py",
    "src/nn/backbone/__init__.py",
    "src/nn/backbone/common.py",
    "src/nn/backbone/presnet.py",
    "src/zoo/__init__.py",
    "src/zoo/rtdetr/__init__.py",
    "src/zoo/rtdetr/box_ops.py",
    "src/zoo/rtdetr/denoising.py",
    "src/zoo/rtdetr/hybrid_encoder.py",
    "src/zoo/rtdetr/rtdetr.py",
    "src/zoo/rtdetr/rtdetrv2_decoder.py",
    "src/zoo/rtdetr/utils.py",
}


def test_only_the_fixed_detector_closure_remains_under_src():
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src").glob("**/*.py")
    }
    assert actual == RETAINED_SRC


def test_training_and_annotation_dependencies_are_absent():
    requirements = (ROOT / "requirements.txt").read_text().lower()
    for removed in ("pycocotools", "pyyaml", "tensorboard", "supervisely"):
        assert removed not in requirements


def test_repository_has_at_most_45_python_files():
    python_files = [
        path for path in ROOT.glob("**/*.py")
        if ".git" not in path.parts and ".worktrees" not in path.parts
    ]
    assert len(python_files) <= 45
```

```bash
$UE_PY -m pytest tests/differential_uncertainty/test_repository_surface.py -q
```

Expected: FAIL on the old files and dependencies. This makes “substantially trimmed” a maintained property, not a one-time impression.

- [ ] **Step 2: Delete the exact obsolete directories and standalone modules**

```bash
git rm -r configs references src/core src/data src/misc src/optim src/scene_uncertainty src/solver tests/scene_uncertainty
git rm -r src/nn/arch src/nn/criterion src/nn/postprocessor
git rm src/nn/backbone/csp_darknet.py src/nn/backbone/csp_resnet.py src/nn/backbone/hgnetv2.py src/nn/backbone/test_resnet.py src/nn/backbone/timm_model.py src/nn/backbone/torchvision_model.py src/nn/backbone/utils.py
git rm src/zoo/rtdetr/conver_params.py src/zoo/rtdetr/matcher.py src/zoo/rtdetr/rtdetr_criterion.py src/zoo/rtdetr/rtdetr_decoder.py src/zoo/rtdetr/rtdetr_postprocessor.py src/zoo/rtdetr/rtdetrv2_criterion.py src/zoo/rtdetr/tue_rtdetr.py src/zoo/rtdetr/tue_rtdetrv2_decoder.py
git rm tools/audit_corruption_bundle.py tools/export_onnx.py tools/scene_uncertainty.py tools/train.py
```

Do not use a broad deletion at the repository root. These explicit paths preserve the fixed detector closure, the new package/tests, documentation, and shell helpers unless a dependency search proves a particular helper obsolete.

- [ ] **Step 3: Replace the dependency list with the inference/reporting closure**

```text
# requirements.txt
torch>=2.3.0
torchvision>=0.18.0
matplotlib>=3.5
numpy
pandas
pillow
scipy
```

Append `runs/` to `.gitignore` so generated experiment directories cannot be committed accidentally.

- [ ] **Step 4: Verify compilation, surface, parity, and removed imports**

```bash
$UE_PY -m compileall -q differential_uncertainty src
$UE_PY -m pytest tests/differential_uncertainty/test_repository_surface.py -q
$UE_PY -m pytest tests/differential_uncertainty/test_detector_parity.py tests/differential_uncertainty/test_legacy_parity.py -q -rs
$UE_PY -m pytest tests/differential_uncertainty -q
rg -n "src\\.(core|data|misc|optim|scene_uncertainty|solver)|pycocotools|yaml|tensorboard|supervisely" differential_uncertainty src tests/differential_uncertainty requirements.txt
```

Expected: compilation and tests pass. The final `rg` exits 1 with no matches, which is the successful no-match result.

- [ ] **Step 5: Commit the exact prune**

```bash
git add -A
git commit -m "refactor: prune to differential inference only"
```

## Task 14: Replace the README and record clean-branch verification

**Files:**

- Modify: `README.md`
- Create: `docs/clean-branch-verification.md`

- [ ] **Step 1: Replace the README with the one supported workflow**

````markdown
# Differential corruption uncertainty

This repository runs one pretrained RT-DETRv2-R18 experiment. It asks a simple
question: when an image is damaged more strongly, does the detector's uncertainty
score usually rise? The workflow performs inference only. It does not train a model
and it does not require COCO annotation files.

## Install

Create a Python environment with a suitable PyTorch build, then run:

```bash
python -m pip install -r requirements.txt
```

## Inputs

Both the clean reference set and evaluation set use the same CSV shape:

```csv
image_id,image_path
street_001,/data/images/street_001.jpg
street_002,/data/images/street_002.jpg
```

Image IDs must be unique inside each manifest. Paths may be absolute or relative to
the manifest. The checkpoint must be the fixed RT-DETRv2-R18 COCO checkpoint used by
this repository.

## Run or resume everything

```bash
python -m differential_uncertainty run \
  --reference-manifest reference.csv \
  --evaluation-manifest evaluation.csv \
  --checkpoint rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \
  --output-dir runs/blur-study
```

Only runtime controls are exposed: `--device`, `--batch-size`, and `--shard-size`.
Scientific choices are fixed in code: layer 2, 335 persistence features, a seeded
25,000-vector clean reference bank, 5-nearest-neighbour distance, responsive deciles
50--60, reference deciles 90--100, relative gap, and Gaussian blur radii 0, 1, 2, 4,
8, and 12.

The command safely resumes compatible partial extraction. It refuses to mix changed
manifests, checkpoint bytes, or source code into an existing run directory.

## Outputs

`<output-dir>/artifacts/` contains provenance, raw feature caches, the clean reference
bank, and per-image scores. `<output-dir>/report/` contains one plain-language Markdown
report, machine-readable metric tables, a JSON summary, and four detailed figures. Re-run
the same command after interruption; completed compatible stages are reused.

## What the score means

For each image, the method ranks valid detector queries by confidence. It compares the mean
layer-2 bank distance in the middle 50--60% confidence group with the mean distance in the
highest 90--100% confidence group:

```text
relative gap = 2 * (responsive mean - reference mean)
                   / (responsive mean + reference mean)
```

The primary score uses distance from the clean reference bank. The matched-confidence
baseline uses `1 - max(sigmoid(class logits))` on the same selected queries. AUROC asks
how often a randomly chosen corrupted image scores above a randomly chosen clean image;
it is a ranking measure, not a probability. Spearman correlation and adjacent-severity
consistency check whether scores rise as corruption becomes stronger. This workflow
does not calculate detector accuracy or mAP because the generic manifests have no
ground-truth boxes.
````

- [ ] **Step 2: Run the final checks and collect observed evidence**

```bash
export UE_PY=/home/yuchen/miniconda3/envs/UE/bin/python
git rev-parse HEAD
$UE_PY -m pytest tests/differential_uncertainty -q
$UE_PY -m pytest tests/differential_uncertainty/test_detector_parity.py tests/differential_uncertainty/test_legacy_parity.py -q -rs
find differential_uncertainty src tests/differential_uncertainty -name '*.py' -type f | sort | wc -l
find differential_uncertainty src tests/differential_uncertainty -name '*.py' -type f -print0 | sort -z | xargs -0 wc -l | tail -n 1
find /home/yuchen/YuchenZ/UE/within-image-contrast-run-record/within_image_contrast -maxdepth 1 -type f -printf '%f\n' | sort
```

Expected historical listing, unchanged and outside the pruned worktree:

```text
anchor_and_responsive_actual_distance.png
anchor_diagnostics.csv
auroc_by_blur_severity.png
candidate_metrics.csv
clean_anchor_relationship.png
contrast_scores_by_severity.png
easy-report.md
per_scene_contrasts.csv
summary.json
```

- [ ] **Step 3: Write the verification record using only the observed outputs**

Create `docs/clean-branch-verification.md` with the title “Clean branch verification.” Use
`apply_patch` and write the actual branch-point commit, the known pre-prune baseline of 1,371
passed tests and 86 warnings, the exact reduced-suite summary, the real-checkpoint detector
parity result, the archived-number parity result, retained Python file and line counts, and the
nine-file historical-bundle check. Include this caveat verbatim:

```text
The regenerated metrics are allowed to differ from the historical run because the new
reference bank intentionally excludes repeated padded detector queries. The archive test
protects the old published numbers; the detector test protects model fidelity.
```

Do not write or commit guessed results. Copy the actual terminal summaries into the record,
then verify that every claimed count and result still matches a fresh command.

- [ ] **Step 4: Commit the user-facing documentation and evidence**

```bash
git add README.md docs/clean-branch-verification.md
git commit -m "docs: explain and verify the clean workflow"
```

## Task 15: Run the acceptance audit and request review

**Files:**

- Verify only; modify the smallest relevant file if an audit exposes a defect

- [ ] **Step 1: Run focused acceptance groups**

```bash
export UE_PY=/home/yuchen/miniconda3/envs/UE/bin/python
$UE_PY -m pytest tests/differential_uncertainty/test_config.py tests/differential_uncertainty/test_manifests.py tests/differential_uncertainty/test_corruptions.py -q
$UE_PY -m pytest tests/differential_uncertainty/test_persistence.py tests/differential_uncertainty/test_extraction.py tests/differential_uncertainty/test_detector_parity.py -q -rs
$UE_PY -m pytest tests/differential_uncertainty/test_artifacts.py tests/differential_uncertainty/test_scoring.py tests/differential_uncertainty/test_bank.py tests/differential_uncertainty/test_extraction.py -q
$UE_PY -m pytest tests/differential_uncertainty/test_evaluation.py tests/differential_uncertainty/test_reporting.py tests/differential_uncertainty/test_legacy_parity.py -q
$UE_PY -m pytest tests/differential_uncertainty/test_pipeline.py tests/differential_uncertainty/test_cli.py tests/differential_uncertainty/test_repository_surface.py -q
```

Expected: all fixed-config, detector-fidelity, persistence, resume, metric, reporting, extension, CLI, and pruning checks pass.

- [ ] **Step 2: Search for the discarded methods and forbidden scope**

```bash
rg -n "raw gap|raw_gap|planned anchor|anchor test|anchor_score|training|COCO annotation" differential_uncertainty src tests/differential_uncertainty README.md docs
```

Expected: no implementation or public-interface matches for raw-gap or planned-anchor methods. “Training” and “COCO annotation” may appear only in plain-language statements that this workflow does not use them; archive tests may name historical fields solely to verify old numbers.

- [ ] **Step 3: Run the complete reduced suite and inspect repository state**

```bash
$UE_PY -m pytest -q
git diff --check
git status --short
git log --oneline --decorate 3f0775d..HEAD
```

Expected: the complete reduced suite passes, `git diff --check` is silent, and the worktree is clean after any audit fixes are committed.

- [ ] **Step 4: Request a code review of the full branch delta**

Invoke `superpowers:requesting-code-review` with base commit `3f0775d` and the current `HEAD`. Ask the reviewer to check:

```text
1. Fidelity to docs/superpowers/specs/2026-08-23-clean-differential-uncertainty-design.md
2. Exact layer-2 persistence and real-checkpoint detector parity
3. Padding removal, clean bank determinism, decile definitions, score orientations
4. AUROC, Spearman, adjacent-consistency, strongest-vs-clean, and paired-bootstrap math
5. Atomic resume/provenance refusal behavior
6. Report completeness and plain-language numeric examples
7. Whether the retained Python closure is truly inference-only
```

- [ ] **Step 5: Handle review findings with evidence, then re-verify**

If the reviewer raises issues, invoke `superpowers:receiving-code-review`, reproduce each issue, fix only confirmed defects, and re-run the nearest focused test plus the complete reduced suite. Commit confirmed fixes with a narrow message.

- [ ] **Step 6: Perform the final completion check**

Invoke `superpowers:verification-before-completion` and provide its required fresh command output. Do not call the branch complete or offer integration until that verification has passed.
