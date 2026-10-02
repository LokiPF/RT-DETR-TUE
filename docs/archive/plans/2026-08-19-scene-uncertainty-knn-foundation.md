# Class-Independent Scene-Uncertainty kNN Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, class-independent COCO persistence-feature cache and evaluate raw scene-level kNN uncertainty across Gaussian-blur severity, query-selection policies, and decoder layers without training an uncertainty head.

**Architecture:** A frozen RT-DETR extractor writes all 300 classification-head persistence vectors plus query/ground-truth metadata into immutable shards. Offline commands derive natural and coverage kNN banks, apply locked query policies, score cached blur sweeps with chunked exact search, and report monotonicity and class-switch diagnostics. Detector extraction and offline scoring are separated so bank size, normalization, query policy, and aggregation can change without another detector forward pass.

**Tech Stack:** Python 3, PyTorch/TorchVision, SciPy, pycocotools, NumPy, pandas, matplotlib, pytest, YAMLConfig, existing `src.misc.tue_utils` persistence hooks.

---

## Scope decomposition

This plan implements the common artifact pipeline plus the kNN comparison method. It produces independently useful end-to-end software and covers the primary baseline from the approved specification.

Two additional implementation plans should be written only after this foundation passes the pilot:

1. query-level statistical scorers: prototype, LOF, Mahalanobis, KDE, topology-native medoids, and persistence landscapes/images;
2. scene-set scorers: Chamfer/Hausdorff, MMD/energy, and Sinkhorn optimal transport.

Both consume the feature-cache interface defined in this plan and do not rerun RT-DETR.

## File structure

Create a focused package rather than adding more responsibilities to `src/solver/tue_engine.py`, which currently mixes confidence filtering, matching, bbox extraction, and class-conditioned Fréchet fitting.

```text
src/scene_uncertainty/
    __init__.py          public package exports
    runtime.py           frozen RT-DETR construction and checkpoint identity
    reference.py         rare-aware COCO and evaluation image selection
    blur.py              deterministic fixed-radius PIL Gaussian blur
    dataset.py           deterministic COCO loader used by extraction
    artifacts.py         schemas, shard writing, loading, compatibility checks
    extractor.py         all-query classification persistence extraction
    normalization.py     raw, robust-z, unit, and shape-plus-scale transforms
    bank.py              natural/coverage bank construction and sampling
    query_policy.py      all, top-K, threshold, smooth, and oracle selections
    knn.py               chunked exact nearest-neighbor distance
    evaluate.py          cached-record scoring and scene aggregation
    metrics.py           monotonicity, overlap, and class-switch metrics
    reporting.py         CSV, JSON, and PNG report generation
    cli.py               command implementations and argument parsing
    pipeline.py          end-to-end command orchestration and artifact checks

tools/scene_uncertainty.py
    thin executable wrapper around src.scene_uncertainty.cli

configs/scene_uncertainty/rtdetrv2_r18vd_coco.yml
    model-only deterministic R18 RT-DETR configuration

tests/scene_uncertainty/
    one focused test module per package responsibility
```

Generated artifacts live under `output/scene_uncertainty/` and are ignored by git.

### Task 1: Establish the package, model-only config, and verified runtime loader

**Files:**
- Create: `.gitignore`
- Create: `configs/scene_uncertainty/rtdetrv2_r18vd_coco.yml`
- Create: `src/scene_uncertainty/__init__.py`
- Create: `src/scene_uncertainty/runtime.py`
- Create: `tests/scene_uncertainty/test_runtime.py`

- [ ] **Step 1: Write the failing runtime tests**

```python
# tests/scene_uncertainty/test_runtime.py
from pathlib import Path

import torch

from src.core import YAMLConfig
from src.scene_uncertainty.runtime import checkpoint_sha256, load_frozen_detector


CONFIG = "configs/scene_uncertainty/rtdetrv2_r18vd_coco.yml"


def test_model_only_config_builds_plain_rtdetr():
    cfg = YAMLConfig(CONFIG)
    model = cfg.model
    assert type(model).__name__ == "RTDETR"
    assert len(model.decoder.decoder.layers) == 3
    assert model.decoder.num_queries == 300
    assert model.decoder.num_classes == 80


def test_checkpoint_sha256_is_content_addressed(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"detector")
    first = checkpoint_sha256(checkpoint)
    checkpoint.write_bytes(b"detector-changed")
    second = checkpoint_sha256(checkpoint)
    assert len(first) == 64
    assert first != second


def test_load_frozen_detector_accepts_model_state(tmp_path: Path):
    cfg = YAMLConfig(CONFIG)
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save({"model": cfg.model.state_dict()}, checkpoint)
    model = load_frozen_detector(CONFIG, checkpoint, torch.device("cpu"))
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
```

- [ ] **Step 2: Run the tests and verify the missing-package failure**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty/test_runtime.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'src.scene_uncertainty'`.

- [ ] **Step 3: Add the model-only config and runtime implementation**

```yaml
# configs/scene_uncertainty/rtdetrv2_r18vd_coco.yml
__include__:
  - ../runtime.yml
  - ../rtdetrv2/include/rtdetrv2_r50vd.yml

task: detection
model: RTDETR
num_classes: 80
use_focal_loss: true
eval_spatial_size: [640, 640]

PResNet:
  depth: 18
  freeze_at: -1
  freeze_norm: false
  pretrained: false

HybridEncoder:
  in_channels: [128, 256, 512]
  hidden_dim: 256
  expansion: 0.5

RTDETRTransformerv2:
  num_layers: 3
  num_queries: 300
```

```python
# src/scene_uncertainty/runtime.py
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from src.core import YAMLConfig


def checkpoint_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_state(checkpoint: dict) -> dict[str, torch.Tensor]:
    if "ema" in checkpoint:
        return checkpoint["ema"]["module"]
    if "model" in checkpoint:
        return checkpoint["model"]
    if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint
    raise KeyError("Checkpoint has neither ema.module nor model state")


def load_frozen_detector(
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: torch.device,
) -> nn.Module:
    cfg = YAMLConfig(str(config_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = _checkpoint_state(checkpoint)
    incompatible = cfg.model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model = cfg.model.to(device).eval()
    model.requires_grad_(False)
    return model
```

```python
# src/scene_uncertainty/__init__.py
from .runtime import checkpoint_sha256, load_frozen_detector

__all__ = ["checkpoint_sha256", "load_frozen_detector"]
```

```gitignore
# .gitignore
output/scene_uncertainty/
```

- [ ] **Step 4: Run the runtime tests**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty/test_runtime.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Run the real checkpoint smoke test**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -c "import torch; from src.scene_uncertainty.runtime import load_frozen_detector; m=load_frozen_detector('configs/scene_uncertainty/rtdetrv2_r18vd_coco.yml','/home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth',torch.device('cuda:0')); y=m(torch.zeros(1,3,640,640,device='cuda:0')); print(tuple(y['pred_logits'].shape), tuple(y['pred_boxes'].shape))"
```

Expected: `(1, 300, 80) (1, 300, 4)`.

- [ ] **Step 6: Commit**

```bash
git add .gitignore configs/scene_uncertainty/rtdetrv2_r18vd_coco.yml src/scene_uncertainty tests/scene_uncertainty/test_runtime.py
git commit -m "feat: add scene uncertainty runtime"
```

### Task 2: Add deterministic rare-aware reference and evaluation selection

**Files:**
- Create: `src/scene_uncertainty/reference.py`
- Create: `tests/scene_uncertainty/test_reference.py`

- [ ] **Step 1: Write failing selection tests with a small COCO-like index**

```python
# tests/scene_uncertainty/test_reference.py
from src.scene_uncertainty.reference import select_evaluation_ids, select_reference_ids


class FakeCoco:
    def __init__(self):
        self.imgs = {image_id: {} for image_id in range(1, 13)}
        self.cats = {1: {"name": "common"}, 2: {"name": "rare"}, 3: {"name": "joint"}}
        self.imgToAnns = {
            1: [{"category_id": 1}],
            2: [{"category_id": 1}],
            3: [{"category_id": 1}],
            4: [{"category_id": 1}],
            5: [{"category_id": 2}, {"category_id": 3}],
            6: [{"category_id": 2}, {"category_id": 3}],
            7: [{"category_id": 2}, {"category_id": 3}],
            8: [{"category_id": 1}],
            9: [{"category_id": 1}],
            10: [{"category_id": 1}],
            11: [{"category_id": 1}],
            12: [{"category_id": 1}],
        }


def test_reference_selection_is_deterministic_and_fills_rare_deficits():
    first = select_reference_ids(FakeCoco(), natural_count=4, augmentation_budget=2, quota=2, seed=7)
    second = select_reference_ids(FakeCoco(), natural_count=4, augmentation_budget=2, quota=2, seed=7)
    assert first == second
    assert len(first["natural_ids"]) == 4
    assert len(first["augmentation_ids"]) == 2
    assert set(first["natural_ids"]).isdisjoint(first["augmentation_ids"])
    assert first["category_image_counts"][2] >= 2
    assert first["category_image_counts"][3] >= 2


def test_evaluation_split_is_disjoint_and_keeps_requested_pilot_sizes():
    result = select_evaluation_ids(list(range(20)), seed=42, tuning_fraction=0.5, pilot_per_partition=4)
    assert set(result["tuning_ids"]).isdisjoint(result["test_ids"])
    assert len(result["tuning_pilot_ids"]) == 4
    assert len(result["test_pilot_ids"]) == 4
    assert set(result["tuning_pilot_ids"]) <= set(result["tuning_ids"])
    assert set(result["test_pilot_ids"]) <= set(result["test_ids"])
```

- [ ] **Step 2: Run the tests and verify the missing-function failures**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty/test_reference.py -v
```

Expected: import fails because `src.scene_uncertainty.reference` does not exist.

- [ ] **Step 3: Implement deterministic natural, augmentation, and evaluation selection**

```python
# src/scene_uncertainty/reference.py
from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


def _image_categories(coco) -> dict[int, frozenset[int]]:
    return {
        int(image_id): frozenset(
            int(annotation["category_id"])
            for annotation in coco.imgToAnns.get(image_id, [])
            if int(annotation.get("iscrowd", 0)) == 0
        )
        for image_id in coco.imgs
    }


def _counts(ids: Iterable[int], categories: dict[int, frozenset[int]]) -> Counter:
    result = Counter()
    for image_id in ids:
        result.update(categories[int(image_id)])
    return result


def select_reference_ids(
    coco,
    natural_count: int = 4000,
    augmentation_budget: int = 1000,
    quota: int = 50,
    seed: int = 42,
) -> dict:
    image_ids = np.asarray(sorted(int(image_id) for image_id in coco.imgs), dtype=np.int64)
    if natural_count + augmentation_budget > image_ids.size:
        raise ValueError("Reference request exceeds available COCO images")
    rng = np.random.default_rng(seed)
    natural = sorted(int(value) for value in rng.choice(image_ids, natural_count, replace=False))
    selected = set(natural)
    categories = _image_categories(coco)
    category_ids = sorted(int(category_id) for category_id in coco.cats)
    counts = _counts(natural, categories)
    augmentation: list[int] = []

    while len(augmentation) < augmentation_budget:
        under = {category_id for category_id in category_ids if counts[category_id] < quota}
        if not under:
            break
        candidates = [int(image_id) for image_id in image_ids if int(image_id) not in selected]
        best_id = min(
            candidates,
            key=lambda image_id: (
                -len(categories[image_id] & under),
                image_id,
            ),
        )
        if not (categories[best_id] & under):
            break
        augmentation.append(best_id)
        selected.add(best_id)
        counts.update(categories[best_id])

    remaining_budget = augmentation_budget - len(augmentation)
    if remaining_budget:
        remaining = np.asarray(
            [int(image_id) for image_id in image_ids if int(image_id) not in selected],
            dtype=np.int64,
        )
        fill = sorted(int(value) for value in rng.choice(remaining, remaining_budget, replace=False))
        augmentation.extend(fill)
        selected.update(fill)
        counts.update(_counts(fill, categories))

    return {
        "seed": seed,
        "natural_ids": natural,
        "augmentation_ids": sorted(augmentation),
        "category_image_counts": {category_id: counts[category_id] for category_id in category_ids},
        "unmet_category_quotas": {
            category_id: quota - counts[category_id]
            for category_id in category_ids
            if counts[category_id] < quota
        },
    }


def select_evaluation_ids(
    image_ids: Iterable[int],
    seed: int = 42,
    tuning_fraction: float = 0.5,
    pilot_per_partition: int = 250,
) -> dict:
    ordered = np.asarray(sorted(int(image_id) for image_id in image_ids), dtype=np.int64)
    shuffled = np.random.default_rng(seed).permutation(ordered)
    split = round(len(shuffled) * tuning_fraction)
    tuning_sequence = [int(value) for value in shuffled[:split]]
    test_sequence = [int(value) for value in shuffled[split:]]
    tuning = sorted(tuning_sequence)
    test = sorted(test_sequence)
    return {
        "seed": seed,
        "tuning_ids": tuning,
        "test_ids": test,
        "tuning_pilot_ids": sorted(tuning_sequence[: min(pilot_per_partition, len(tuning_sequence))]),
        "test_pilot_ids": sorted(test_sequence[: min(pilot_per_partition, len(test_sequence))]),
    }
```

- [ ] **Step 4: Run the selection tests**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty/test_reference.py -v
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/scene_uncertainty/reference.py tests/scene_uncertainty/test_reference.py
git commit -m "feat: add rare-aware COCO selection"
```

### Task 3: Preserve annotation identity and add deterministic post-resize blur loading

**Files:**
- Modify: `src/data/dataset/coco_dataset.py:63-104`
- Create: `src/scene_uncertainty/blur.py`
- Create: `src/scene_uncertainty/dataset.py`
- Create: `tests/scene_uncertainty/test_blur_dataset.py`

- [ ] **Step 1: Write failing tests for annotation IDs and exact severity zero**

```python
# tests/scene_uncertainty/test_blur_dataset.py
import numpy as np
import torch
from PIL import Image

from src.data.dataset.coco_dataset import ConvertCocoPolysToMask
from src.scene_uncertainty.blur import FixedGaussianBlur


def test_converter_preserves_filtered_annotation_ids():
    image = Image.new("RGB", (20, 20), color="white")
    target = {
        "image_id": 9,
        "annotations": [
            {"id": 101, "bbox": [1, 1, 5, 5], "category_id": 3, "area": 25, "iscrowd": 0},
            {"id": 102, "bbox": [2, 2, 0, 4], "category_id": 4, "area": 0, "iscrowd": 0},
        ],
    }
    _, converted = ConvertCocoPolysToMask(False)(image, target, category2label={3: 0, 4: 1})
    assert converted["annotation_ids"].tolist() == [101]


def test_blur_radius_zero_is_byte_exact():
    array = np.arange(12 * 12 * 3, dtype=np.uint8).reshape(12, 12, 3)
    image = Image.fromarray(array)
    output = FixedGaussianBlur(0)(image)
    assert np.array_equal(np.asarray(output), array)


def test_positive_blur_changes_a_sharp_edge():
    array = np.zeros((21, 21, 3), dtype=np.uint8)
    array[:, 10:] = 255
    image = Image.fromarray(array)
    output = FixedGaussianBlur(4)(image)
    assert not torch.equal(torch.from_numpy(np.asarray(output).copy()), torch.from_numpy(array))
```

- [ ] **Step 2: Run the tests and verify failures**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty/test_blur_dataset.py -v
```

Expected: import fails for `FixedGaussianBlur`; after test collection is repaired, the annotation-ID assertion fails.

- [ ] **Step 3: Add annotation IDs to converted COCO targets**

In `ConvertCocoPolysToMask.__call__`, create IDs alongside boxes and apply the same `keep` mask:

```python
        annotation_ids = torch.tensor(
            [int(obj["id"]) for obj in anno],
            dtype=torch.int64,
        )

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        annotation_ids = annotation_ids[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["annotation_ids"] = annotation_ids
```

- [ ] **Step 4: Implement fixed blur and deterministic loader construction**

```python
# src/scene_uncertainty/blur.py
from __future__ import annotations

from PIL import Image, ImageFilter
from torchvision.transforms.v2 import Transform


class FixedGaussianBlur(Transform):
    _transformed_types = (Image.Image,)

    def __init__(self, radius: float) -> None:
        super().__init__()
        if radius < 0:
            raise ValueError("Blur radius must be non-negative")
        self.radius = float(radius)

    def _transform(self, image: Image.Image, params: dict) -> Image.Image:
        if self.radius == 0:
            return image
        return image.filter(ImageFilter.GaussianBlur(radius=self.radius))

    def transform(self, image: Image.Image, params: dict) -> Image.Image:
        # The local transform stack calls the public torchvision-v2 hook.
        return self._transform(image, params)
```

```python
# src/scene_uncertainty/dataset.py
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from torch.utils.data import DataLoader

from src.data.dataloader import batch_image_collate_fn
from src.data.dataset import CocoDetection
from src.data.transforms import Compose, ConvertBoxes, ConvertPILImage, Resize, SanitizeBoundingBoxes

from .blur import FixedGaussianBlur


def make_coco_loader(
    image_root: str | Path,
    annotation_file: str | Path,
    image_ids: Sequence[int],
    blur_radius: float,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    transforms = Compose(
        [
            Resize(size=[640, 640]),
            FixedGaussianBlur(blur_radius),
            SanitizeBoundingBoxes(min_size=1),
            ConvertPILImage(dtype="float32", scale=True),
            ConvertBoxes(fmt="cxcywh", normalize=True),
        ]
    )
    dataset = CocoDetection(
        img_folder=str(image_root),
        ann_file=str(annotation_file),
        transforms=transforms,
        return_masks=False,
        remap_mscoco_category=True,
    )
    available = set(int(image_id) for image_id in dataset.ids)
    requested = [int(image_id) for image_id in image_ids]
    missing = sorted(set(requested) - available)
    if missing:
        raise KeyError(f"Requested COCO image IDs are missing: {missing[:10]}")
    dataset.ids = requested
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=batch_image_collate_fn,
        drop_last=False,
    )
```

- [ ] **Step 5: Run the blur and dataset tests**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty/test_blur_dataset.py -v
```

Expected: `3 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/data/dataset/coco_dataset.py src/scene_uncertainty/blur.py src/scene_uncertainty/dataset.py tests/scene_uncertainty/test_blur_dataset.py
git commit -m "feat: add deterministic blur dataset"
```

### Task 4: Add resumable immutable sharded artifacts and compatibility validation

**Files:**
- Create: `src/scene_uncertainty/artifacts.py`
- Create: `tests/scene_uncertainty/test_artifacts.py`

- [ ] **Step 1: Write failing round-trip and incompatibility tests**

```python
# tests/scene_uncertainty/test_artifacts.py
from pathlib import Path

import pytest
import torch

from src.scene_uncertainty.artifacts import ShardWriter, assert_compatible, iter_records, load_manifest


def test_shard_writer_round_trip(tmp_path: Path):
    metadata = {"checkpoint_sha256": "abc", "decoder_layers": [0, 1, 2], "query_count": 300}
    with ShardWriter(tmp_path, metadata, shard_size=2) as writer:
        writer.add({"image_id": 1, "layers": {0: torch.ones(300, 4)}})
        writer.add({"image_id": 2, "layers": {0: torch.zeros(300, 4)}})
        writer.add({"image_id": 3, "layers": {0: torch.full((300, 4), 2.0)}})
    manifest = load_manifest(tmp_path)
    assert manifest["record_count"] == 3
    assert len(manifest["artifact_id"]) == 64
    assert manifest["shards"] == ["shard_00000.pt", "shard_00001.pt"]
    assert [record["image_id"] for record in iter_records(tmp_path)] == [1, 2, 3]


def test_interrupted_writer_resumes_completed_shards(tmp_path: Path):
    metadata = {"checkpoint_sha256": "abc", "decoder_layers": [0], "query_count": 300}
    interrupted = ShardWriter(tmp_path, metadata, shard_size=2)
    interrupted.add({"image_id": 1, "severity": 0})
    interrupted.add({"image_id": 2, "severity": 0})
    resumed = ShardWriter(tmp_path, metadata, shard_size=2)
    assert resumed.existing_record_keys() == {(1, 0), (2, 0)}
    resumed.add({"image_id": 3, "severity": 0})
    resumed.close()
    assert [record["image_id"] for record in iter_records(tmp_path)] == [1, 2, 3]


def test_completed_artifact_is_immutable(tmp_path: Path):
    with ShardWriter(tmp_path, {"checkpoint_sha256": "abc"}, shard_size=2):
        pass
    with pytest.raises(FileExistsError, match="complete"):
        ShardWriter(tmp_path, {"checkpoint_sha256": "abc"}, shard_size=2)


def test_compatibility_rejects_checkpoint_mismatch():
    expected = {"checkpoint_sha256": "a", "decoder_layers": [0, 1, 2]}
    actual = {"checkpoint_sha256": "b", "decoder_layers": [0, 1, 2]}
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        assert_compatible(actual, expected, keys=("checkpoint_sha256", "decoder_layers"))
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty/test_artifacts.py -v
```

Expected: import fails for `src.scene_uncertainty.artifacts`.

- [ ] **Step 3: Implement atomic shards and manifests**

```python
# src/scene_uncertainty/artifacts.py
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

import torch


SCHEMA_VERSION = 1


def manifest_id(manifest: Mapping) -> str:
    payload = {key: value for key, value in manifest.items() if key != "artifact_id"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_torch_save(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json_save(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


class ShardWriter:
    def __init__(self, directory: str | Path, metadata: Mapping, shard_size: int = 50) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata)
        self.shard_size = int(shard_size)
        self.buffer: list[dict] = []
        self.final_manifest = self.directory / "manifest.json"
        self.partial_manifest = self.directory / "partial_manifest.json"
        if self.final_manifest.exists():
            raise FileExistsError(f"Artifact is already complete: {self.final_manifest}")
        if self.partial_manifest.exists():
            partial = json.loads(self.partial_manifest.read_text(encoding="utf-8"))
            assert_compatible(partial, self.metadata, keys=self.metadata)
            if partial["shard_size"] != self.shard_size:
                raise ValueError("Cannot resume with a different shard_size")
            self.shards = list(partial["shards"])
            self.record_count = int(partial["record_count"])
        else:
            self.shards = []
            self.record_count = 0
            self._write_partial_manifest()

    def __enter__(self):
        return self

    def add(self, record: dict) -> None:
        self.buffer.append(record)
        self.record_count += 1
        if len(self.buffer) >= self.shard_size:
            self._flush()

    def _write_partial_manifest(self) -> None:
        _atomic_json_save({
            "schema_version": SCHEMA_VERSION,
            **self.metadata,
            "shard_size": self.shard_size,
            "record_count": self.record_count,
            "shards": self.shards,
        }, self.partial_manifest)

    def existing_record_keys(self) -> set[tuple[int, int]]:
        keys = set()
        for shard in self.shards:
            records = torch.load(self.directory / shard, map_location="cpu", weights_only=False)
            keys.update((int(record["image_id"]), int(record.get("severity", 0))) for record in records)
        return keys

    def _flush(self) -> None:
        if not self.buffer:
            return
        name = f"shard_{len(self.shards):05d}.pt"
        _atomic_torch_save(self.buffer, self.directory / name)
        self.shards.append(name)
        self.buffer = []
        self._write_partial_manifest()

    def close(self) -> None:
        self._flush()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            **self.metadata,
            "record_count": self.record_count,
            "shards": self.shards,
        }
        manifest["artifact_id"] = manifest_id(manifest)
        _atomic_json_save(manifest, self.final_manifest)
        self.partial_manifest.unlink()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()


def load_manifest(directory: str | Path) -> dict:
    return json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))


def iter_records(directory: str | Path) -> Iterator[dict]:
    root = Path(directory)
    manifest = load_manifest(root)
    for shard in manifest["shards"]:
        yield from torch.load(root / shard, map_location="cpu", weights_only=False)


def assert_compatible(actual: Mapping, expected: Mapping, keys: Iterable[str]) -> None:
    for key in keys:
        if actual.get(key) != expected.get(key):
            raise ValueError(
                f"Artifact mismatch for {key}: actual={actual.get(key)!r}, expected={expected.get(key)!r}"
            )
```

- [ ] **Step 4: Run the artifact tests**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty/test_artifacts.py -v
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/scene_uncertainty/artifacts.py tests/scene_uncertainty/test_artifacts.py
git commit -m "feat: add persistence cache artifacts"
```

### Task 5: Extract all classification persistence vectors with matching metadata

**Files:**
- Create: `src/scene_uncertainty/extractor.py`
- Create: `tests/scene_uncertainty/test_extractor.py`

- [ ] **Step 1: Write failing tests for matching metadata and class independence**

```python
# tests/scene_uncertainty/test_extractor.py
import torch

from src.scene_uncertainty.extractor import build_match_metadata, stack_query_diagrams


def test_match_metadata_keeps_unmatched_and_incorrect_queries():
    logits = torch.tensor([[[5.0, -1.0], [-2.0, 4.0], [1.0, 0.0]]])
    targets = [{
        "labels": torch.tensor([0, 0]),
        "annotation_ids": torch.tensor([101, 102]),
    }]
    match_indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
    metadata = build_match_metadata(logits, targets, match_indices)[0]
    assert metadata["matched_annotation_id"].tolist() == [101, 102, -1]
    assert metadata["matched_gt_class"].tolist() == [0, 0, -1]
    assert metadata["is_matched"].tolist() == [True, True, False]
    assert metadata["is_correct"].tolist() == [True, False, False]


def test_stack_query_diagrams_preserves_query_order():
    diagrams = {2: torch.tensor([2.0]), 0: torch.tensor([0.0]), 1: torch.tensor([1.0])}
    stacked = stack_query_diagrams(diagrams, query_count=3)
    assert stacked[:, 0].tolist() == [0.0, 1.0, 2.0]
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty/test_extractor.py -v
```

Expected: import fails for `src.scene_uncertainty.extractor`.

- [ ] **Step 3: Implement pure metadata helpers and the hooked extractor**

```python
# src/scene_uncertainty/extractor.py
from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import Tensor, nn

from src.misc.tue_utils import get_captured_persistence_diagrams, hook_decoder_layers


def stack_query_diagrams(diagrams: dict[int, Tensor], query_count: int) -> Tensor:
    missing = sorted(set(range(query_count)) - set(diagrams))
    if missing:
        raise RuntimeError(f"Missing persistence diagrams for query IDs: {missing[:10]}")
    return torch.stack([diagrams[query_id] for query_id in range(query_count)])


def build_match_metadata(logits: Tensor, targets: Sequence[dict], match_indices) -> list[dict]:
    predicted_class = logits.argmax(dim=-1).cpu()
    confidence = logits.sigmoid().amax(dim=-1).cpu()
    batch_size, query_count = predicted_class.shape
    result: list[dict] = []
    for batch_id in range(batch_size):
        matched_annotation_id = torch.full((query_count,), -1, dtype=torch.int64)
        matched_gt_class = torch.full((query_count,), -1, dtype=torch.int64)
        query_indices, target_indices = match_indices[batch_id]
        target_indices_device = target_indices.to(targets[batch_id]["labels"].device)
        matched_annotation_id[query_indices] = targets[batch_id]["annotation_ids"][target_indices_device].cpu()
        matched_gt_class[query_indices] = targets[batch_id]["labels"][target_indices_device].cpu()
        is_matched = matched_gt_class >= 0
        result.append({
            "predicted_class": predicted_class[batch_id],
            "confidence": confidence[batch_id],
            "matched_annotation_id": matched_annotation_id,
            "matched_gt_class": matched_gt_class,
            "is_matched": is_matched,
            "is_correct": is_matched & (predicted_class[batch_id] == matched_gt_class),
        })
    return result


class ClassificationPersistenceExtractor:
    def __init__(self, model: nn.Module, matcher: nn.Module, decoder_layers: Iterable[int]) -> None:
        self.model = model
        self.matcher = matcher
        self.captures, self.handles, self.layers = hook_decoder_layers(
            transformer=model.decoder,
            decoder_layers=list(decoder_layers),
            head_task="score",
        )

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @torch.inference_mode()
    def extract(self, samples: Tensor, targets: Sequence[dict]) -> list[dict]:
        self.captures.clear()
        outputs = self.model(samples)
        query_count = outputs["pred_logits"].shape[1]
        query_indices = [
            torch.arange(query_count, device=samples.device)
            for _ in range(samples.shape[0])
        ]
        diagrams = get_captured_persistence_diagrams(
            captures=self.captures,
            query_indices=query_indices,
            decoder_layer_indices=self.layers,
        )
        matches = self.matcher(outputs, targets)["indices"]
        metadata = build_match_metadata(outputs["pred_logits"], targets, matches)
        records: list[dict] = []
        for batch_id, target in enumerate(targets):
            records.append({
                "image_id": int(target["image_id"].item()),
                "orig_size": target["orig_size"].cpu(),
                "layers": {
                    int(layer_id): stack_query_diagrams(diagrams[layer_id][batch_id], query_count).to(torch.float16)
                    for layer_id in self.layers
                },
                "logits": outputs["pred_logits"][batch_id].detach().cpu().to(torch.float16),
                "boxes": outputs["pred_boxes"][batch_id].detach().cpu().to(torch.float32),
                **metadata[batch_id],
            })
        return records
```

- [ ] **Step 4: Run the extractor unit tests**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty/test_extractor.py -v
```

Expected: `2 passed`.

- [ ] **Step 5: Add a hook-removal regression test**

Append a fake removable handle test:

```python
def test_close_removes_every_hook():
    class Handle:
        def __init__(self):
            self.removed = False

        def remove(self):
            self.removed = True

    extractor = object.__new__(__import__(
        "src.scene_uncertainty.extractor", fromlist=["ClassificationPersistenceExtractor"]
    ).ClassificationPersistenceExtractor)
    extractor.handles = [Handle(), Handle()]
    handles = list(extractor.handles)
    extractor.close()
    assert all(handle.removed for handle in handles)
    assert extractor.handles == []
```

- [ ] **Step 6: Run the extractor tests again**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty/test_extractor.py -v
```

Expected: `3 passed`.

- [ ] **Step 7: Commit**

```bash
git add src/scene_uncertainty/extractor.py tests/scene_uncertainty/test_extractor.py
git commit -m "feat: extract all-query persistence metadata"
```

### Task 6: Add feature normalization and natural/coverage bank construction

**Files:**
- Create: `src/scene_uncertainty/normalization.py`
- Create: `src/scene_uncertainty/bank.py`
- Create: `tests/scene_uncertainty/test_bank.py`

- [ ] **Step 1: Write failing normalization and bank tests**

```python
# tests/scene_uncertainty/test_bank.py
import torch

from src.scene_uncertainty.bank import deterministic_reservoir, make_coverage_bank, streaming_coverage_bank
from src.scene_uncertainty.normalization import fit_normalizer, transform_vectors


def test_reservoir_is_deterministic_and_capped():
    vectors = [torch.tensor([float(index), 0.0]) for index in range(100)]
    first = deterministic_reservoir(vectors, capacity=10, seed=3)
    second = deterministic_reservoir(vectors, capacity=10, seed=3)
    assert torch.equal(first, second)
    assert first.shape == (10, 2)


def test_coverage_bank_reserves_object_and_background_capacity():
    vectors = torch.arange(80, dtype=torch.float32).reshape(20, 4)
    classes = torch.tensor([0] * 4 + [1] * 4 + [-1] * 12)
    bank, selected = make_coverage_bank(vectors, classes, capacity=12, object_fraction=0.5, seed=5)
    assert bank.shape == (12, 4)
    assert (classes[selected] >= 0).sum().item() >= 6
    assert (classes[selected] < 0).sum().item() >= 6


def test_streaming_coverage_bank_is_capped_without_concatenating_the_cache():
    records = [
        {"layers": {2: torch.full((4, 2), float(index))},
         "matched_gt_class": torch.tensor([0, 1, -1, -1])}
        for index in range(20)
    ]
    bank, metadata = streaming_coverage_bank(
        iter(records), layer_id=2, capacity=20, object_fraction=0.5, class_count=2, seed=7
    )
    assert bank.shape == (20, 2)
    assert metadata["object_vectors"] == 10
    assert metadata["background_vectors"] == 10


def test_all_normalizers_are_finite_and_shape_scale_adds_one_coordinate():
    bank = torch.tensor([[1.0, 0.0], [2.0, 2.0], [4.0, 1.0]])
    for mode in ("raw", "robust_z", "unit"):
        state = fit_normalizer(bank, mode)
        output = transform_vectors(bank, state)
        assert output.shape == bank.shape
        assert torch.isfinite(output).all()
    state = fit_normalizer(bank, "shape_scale")
    output = transform_vectors(bank, state)
    assert output.shape == (3, 3)
    assert torch.isfinite(output).all()
```

- [ ] **Step 2: Run the tests and verify missing-module failures**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty/test_bank.py -v
```

Expected: imports fail for `bank` and `normalization`.

- [ ] **Step 3: Implement clean-only normalizers**

```python
# src/scene_uncertainty/normalization.py
from __future__ import annotations

import torch
from torch import Tensor


def _robust_state(values: Tensor) -> tuple[Tensor, Tensor]:
    median = values.median(dim=0).values
    q25 = torch.quantile(values, 0.25, dim=0)
    q75 = torch.quantile(values, 0.75, dim=0)
    scale = (q75 - q25).clamp_min(1e-6)
    return median, scale


def fit_normalizer(bank: Tensor, mode: str) -> dict:
    bank = bank.float()
    if mode == "raw":
        return {"mode": mode}
    if mode == "robust_z":
        center, scale = _robust_state(bank)
        return {"mode": mode, "center": center, "scale": scale}
    if mode == "unit":
        return {"mode": mode}
    if mode == "shape_scale":
        magnitudes = bank.norm(dim=1, keepdim=True).log1p()
        center, scale = _robust_state(magnitudes)
        return {"mode": mode, "magnitude_center": center, "magnitude_scale": scale}
    raise ValueError(f"Unknown normalization mode: {mode}")


def transform_vectors(vectors: Tensor, state: dict) -> Tensor:
    vectors = vectors.float()
    mode = state["mode"]
    if mode == "raw":
        return vectors
    if mode == "robust_z":
        return (vectors - state["center"]) / state["scale"]
    norms = vectors.norm(dim=1, keepdim=True).clamp_min(1e-12)
    shape = vectors / norms
    if mode == "unit":
        return shape
    if mode == "shape_scale":
        magnitude = (norms.log1p() - state["magnitude_center"]) / state["magnitude_scale"]
        return torch.cat((shape, magnitude), dim=1)
    raise ValueError(f"Unknown normalization mode: {mode}")
```

- [ ] **Step 4: Implement deterministic uniform and coverage sampling**

```python
# src/scene_uncertainty/bank.py
from __future__ import annotations

from collections.abc import Iterable

import math

import numpy as np
import torch
from torch import Tensor


def deterministic_reservoir(vectors: Iterable[Tensor], capacity: int, seed: int) -> Tensor:
    rng = np.random.default_rng(seed)
    reservoir: list[Tensor] = []
    for seen, vector in enumerate(vectors, start=1):
        value = vector.detach().cpu().float()
        if len(reservoir) < capacity:
            reservoir.append(value)
            continue
        replacement = int(rng.integers(0, seen))
        if replacement < capacity:
            reservoir[replacement] = value
    if not reservoir:
        raise ValueError("Cannot build a bank from zero vectors")
    return torch.stack(reservoir)


def _sample_indices(indices: Tensor, count: int, rng: np.random.Generator) -> list[int]:
    values = indices.cpu().numpy()
    chosen = rng.choice(values, size=min(count, len(values)), replace=False)
    return [int(value) for value in chosen]


def make_coverage_bank(
    vectors: Tensor,
    matched_gt_class: Tensor,
    capacity: int,
    object_fraction: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    if not 0.0 < object_fraction < 1.0:
        raise ValueError("object_fraction must lie between zero and one")
    rng = np.random.default_rng(seed)
    object_budget = round(capacity * object_fraction)
    background_budget = capacity - object_budget
    object_indices = torch.where(matched_gt_class >= 0)[0]
    background_indices = torch.where(matched_gt_class < 0)[0]
    selected: list[int] = []
    classes = sorted(int(value) for value in matched_gt_class[object_indices].unique().tolist())
    per_class = max(1, object_budget // max(1, len(classes)))
    for class_id in classes:
        class_indices = torch.where(matched_gt_class == class_id)[0]
        selected.extend(_sample_indices(class_indices, per_class, rng))
    selected = selected[:object_budget]
    selected.extend(_sample_indices(background_indices, background_budget, rng))
    if len(selected) < capacity:
        selected_set = set(selected)
        remaining = torch.tensor(
            [index for index in range(vectors.shape[0]) if index not in selected_set],
            dtype=torch.long,
        )
        selected.extend(_sample_indices(remaining, capacity - len(selected), rng))
    selected_tensor = torch.tensor(selected, dtype=torch.long)
    return vectors.index_select(0, selected_tensor).float(), selected_tensor


def streaming_coverage_bank(
    records: Iterable[dict],
    layer_id: int,
    capacity: int,
    object_fraction: float,
    class_count: int,
    seed: int,
) -> tuple[Tensor, dict]:
    if capacity < 2 or not 0.0 < object_fraction < 1.0:
        raise ValueError("Coverage bank needs capacity >= 2 and object_fraction between zero and one")
    object_budget = round(capacity * object_fraction)
    per_class_capacity = max(1, math.ceil(object_budget / class_count))
    class_vectors: dict[int, list[Tensor]] = {class_id: [] for class_id in range(class_count)}
    class_seen = {class_id: 0 for class_id in range(class_count)}
    background: list[Tensor] = []
    background_seen = 0
    generators = {class_id: np.random.default_rng(seed + class_id) for class_id in range(class_count)}
    background_rng = np.random.default_rng(seed + class_count + 1)

    def update(reservoir, seen, value, limit, rng):
        seen += 1
        if len(reservoir) < limit:
            reservoir.append(value.detach().cpu().float())
        else:
            replacement = int(rng.integers(0, seen))
            if replacement < limit:
                reservoir[replacement] = value.detach().cpu().float()
        return seen

    for record in records:
        for vector, class_id in zip(record["layers"][layer_id], record["matched_gt_class"]):
            class_value = int(class_id)
            if class_value < 0:
                background_seen = update(background, background_seen, vector, capacity, background_rng)
            else:
                class_seen[class_value] = update(
                    class_vectors[class_value], class_seen[class_value], vector,
                    per_class_capacity, generators[class_value],
                )
    objects = [vector for class_id in range(class_count) for vector in class_vectors[class_id]]
    if not objects or not background:
        raise ValueError("Coverage bank requires both matched-object and background vectors")
    object_bank = deterministic_reservoir(objects, min(object_budget, len(objects)), seed + 10_000)
    background_needed = min(capacity - len(object_bank), len(background))
    background_bank = deterministic_reservoir(background, background_needed, seed + 20_000)
    bank = torch.cat((object_bank, background_bank))
    return bank, {
        "object_vectors": int(len(object_bank)),
        "background_vectors": int(len(background_bank)),
        "per_class_available": class_seen,
    }
```

- [ ] **Step 5: Run the bank tests**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty/test_bank.py -v
```

Expected: `4 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/scene_uncertainty/normalization.py src/scene_uncertainty/bank.py tests/scene_uncertainty/test_bank.py
git commit -m "feat: build class-independent query banks"
```

### Task 7: Implement the locked query policies and scene aggregations

**Files:**
- Create: `src/scene_uncertainty/query_policy.py`
- Create: `tests/scene_uncertainty/test_query_policy.py`

- [ ] **Step 1: Write failing tests for every deployable policy and every annotated oracle diagnostic**

```python
# tests/scene_uncertainty/test_query_policy.py
import torch

from src.scene_uncertainty.query_policy import aggregate_scores, select_oracle_queries, select_queries


def logits_from_confidences(confidences):
    values = torch.tensor(confidences).clamp(1e-5, 1 - 1e-5)
    logits = torch.logit(values)
    return torch.stack((logits, torch.full_like(logits, -20.0)), dim=1)


def test_topk_returns_unique_queries_in_descending_confidence_order():
    logits = logits_from_confidences([0.1, 0.9, 0.4, 0.8])
    selection = select_queries(logits, "top20", topk_override=2)
    assert selection["indices"].tolist() == [1, 3]
    assert torch.allclose(selection["weights"], torch.tensor([0.5, 0.5]))


def test_threshold_uses_strict_greater_than_and_reports_empty():
    logits = logits_from_confidences([0.2, 0.3, 0.5])
    selected = select_queries(logits, "threshold_0.3")
    empty = select_queries(logits, "threshold_0.5")
    assert selected["indices"].tolist() == [2]
    assert empty["valid"] is False


def test_smooth_policy_keeps_every_query_with_positive_weight():
    logits = logits_from_confidences([0.01, 0.5, 0.99])
    selection = select_queries(logits, "smooth_2", uniform_floor=0.05)
    assert selection["indices"].tolist() == [0, 1, 2]
    assert torch.all(selection["weights"] > 0)
    assert torch.isclose(selection["weights"].sum(), torch.tensor(1.0))


def test_oracle_policies_use_matching_metadata_without_class_routing():
    record = {
        "matched_gt_class": torch.tensor([2, 4, -1, 7]),
        "is_matched": torch.tensor([True, True, False, True]),
        "is_correct": torch.tensor([True, False, False, False]),
    }
    assert select_oracle_queries(record, "oracle_matched")["indices"].tolist() == [0, 1, 3]
    assert select_oracle_queries(record, "oracle_background")["indices"].tolist() == [2]
    assert select_oracle_queries(record, "oracle_correct")["indices"].tolist() == [0]
    assert select_oracle_queries(record, "oracle_incorrect")["indices"].tolist() == [1, 3]


def test_scene_aggregations_have_expected_values():
    scores = torch.tensor([1.0, 2.0, 3.0, 100.0])
    assert aggregate_scores(scores, "mean").item() == 26.5
    assert aggregate_scores(scores, "median").item() == 2.0
    assert aggregate_scores(scores, "top20_mean").item() == 100.0
    weighted = aggregate_scores(scores, "weighted_mean", torch.tensor([0.1, 0.2, 0.3, 0.4]))
    assert torch.isclose(weighted, torch.tensor(41.4))
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty/test_query_policy.py -v
```

Expected: import fails for `query_policy`.

- [ ] **Step 3: Implement all, top-K, threshold, smooth, oracle, and aggregation policies**

```python
# src/scene_uncertainty/query_policy.py
from __future__ import annotations

import math

import torch
from torch import Tensor


TOPK = {"top10": 10, "top20": 20, "top50": 50}
THRESHOLDS = {"threshold_0.2": 0.2, "threshold_0.3": 0.3, "threshold_0.5": 0.5}
SMOOTH_EXPONENTS = {"smooth_1": 1.0, "smooth_2": 2.0}
ORACLE_POLICIES = ("oracle_matched", "oracle_background", "oracle_correct", "oracle_incorrect")


def _selection_from_mask(mask: Tensor) -> dict:
    indices = torch.where(mask.cpu())[0]
    if indices.numel() == 0:
        return {"indices": indices, "weights": torch.empty(0), "valid": False}
    return {
        "indices": indices,
        "weights": torch.full((indices.numel(),), 1.0 / indices.numel()),
        "valid": True,
    }


def select_oracle_queries(record: dict, policy: str) -> dict:
    if policy == "oracle_matched":
        return _selection_from_mask(record["is_matched"])
    if policy == "oracle_background":
        return _selection_from_mask(~record["is_matched"])
    if policy == "oracle_correct":
        return _selection_from_mask(record["is_correct"])
    if policy == "oracle_incorrect":
        return _selection_from_mask(record["is_matched"] & ~record["is_correct"])
    raise ValueError(f"Unknown oracle query policy: {policy}")


def select_queries(
    logits: Tensor,
    policy: str,
    uniform_floor: float = 0.05,
    topk_override: int | None = None,
) -> dict:
    confidence = logits.sigmoid().amax(dim=-1)
    query_count = logits.shape[0]
    if policy == "all":
        indices = torch.arange(query_count)
        weights = torch.full((query_count,), 1.0 / query_count)
    elif policy in TOPK:
        count = min(topk_override or TOPK[policy], query_count)
        indices = torch.argsort(confidence, descending=True, stable=True)[:count].cpu()
        weights = torch.full((count,), 1.0 / count)
    elif policy in THRESHOLDS:
        indices = torch.where(confidence > THRESHOLDS[policy])[0].cpu()
        if indices.numel() == 0:
            return {"indices": indices, "weights": torch.empty(0), "valid": False}
        weights = torch.full((indices.numel(),), 1.0 / indices.numel())
    elif policy in SMOOTH_EXPONENTS:
        indices = torch.arange(query_count)
        weights = uniform_floor + confidence.cpu().pow(SMOOTH_EXPONENTS[policy])
        weights = weights / weights.sum()
    else:
        raise ValueError(f"Unknown query policy: {policy}")
    return {"indices": indices, "weights": weights, "valid": True}


def aggregate_scores(scores: Tensor, method: str, weights: Tensor | None = None) -> Tensor:
    scores = scores.float()
    if scores.numel() == 0:
        return torch.tensor(float("nan"))
    if method == "mean":
        return scores.mean()
    if method == "median":
        return scores.median()
    if method == "q90":
        return torch.quantile(scores, 0.9)
    if method == "top20_mean":
        count = max(1, math.ceil(scores.numel() * 0.2))
        return scores.topk(count).values.mean()
    if method == "weighted_mean":
        if weights is None or weights.numel() != scores.numel():
            raise ValueError("weighted_mean requires one weight per score")
        return (scores * weights.to(scores)).sum()
    raise ValueError(f"Unknown aggregation method: {method}")
```

- [ ] **Step 4: Run the query-policy tests**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty/test_query_policy.py -v
```

Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/scene_uncertainty/query_policy.py tests/scene_uncertainty/test_query_policy.py
git commit -m "feat: add scene query policies"
```

### Task 8: Implement chunked exact kNN without materializing the full distance matrix

**Files:**
- Create: `src/scene_uncertainty/knn.py`
- Create: `tests/scene_uncertainty/test_knn.py`

- [ ] **Step 1: Write failing exactness and chunk-bound tests**

```python
# tests/scene_uncertainty/test_knn.py
import torch

from src.scene_uncertainty.knn import chunked_knn_distances, fit_clean_distance_scale


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
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty/test_knn.py -v
```

Expected: import fails for `knn`.

- [ ] **Step 3: Implement the streaming top-k merge**

```python
# src/scene_uncertainty/knn.py
from __future__ import annotations

import torch
from torch import Tensor


def _squared_distances(queries: Tensor, bank: Tensor) -> Tensor:
    query_norm = queries.square().sum(dim=1, keepdim=True)
    bank_norm = bank.square().sum(dim=1).unsqueeze(0)
    return (query_norm + bank_norm - 2.0 * queries @ bank.T).clamp_min_(0.0)


def chunked_knn_distances(
    queries: Tensor,
    bank: Tensor,
    k: int,
    bank_chunk_size: int = 8192,
) -> Tensor:
    queries = queries.float()
    bank = bank.float().to(queries.device)
    if k <= 0 or k > bank.shape[0]:
        raise ValueError(f"k must be in [1, {bank.shape[0]}], got {k}")
    best = torch.full((queries.shape[0], k), float("inf"), device=queries.device)
    for bank_chunk in bank.split(bank_chunk_size, dim=0):
        local = _squared_distances(queries, bank_chunk)
        local_k = min(k, local.shape[1])
        local_best = local.topk(local_k, dim=1, largest=False).values
        best = torch.cat((best, local_best), dim=1).topk(k, dim=1, largest=False).values
    return best.clamp_min(0.0).sqrt()


def mean_knn_distance(queries: Tensor, bank: Tensor, k: int, bank_chunk_size: int = 8192) -> Tensor:
    return chunked_knn_distances(queries, bank, k, bank_chunk_size).mean(dim=1)


def fit_clean_distance_scale(
    bank: Tensor,
    k: int,
    max_samples: int = 1024,
    seed: int = 42,
    bank_chunk_size: int = 8192,
) -> dict[str, Tensor]:
    if bank.shape[0] <= k:
        raise ValueError("Bank must contain more than k vectors for leave-self-out scaling")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(bank.shape[0], generator=generator)[:max_samples]
    sample = bank.index_select(0, indices.to(bank.device))
    neighbors = chunked_knn_distances(sample, bank, k + 1, bank_chunk_size)
    clean_scores = neighbors[:, 1:].mean(dim=1).cpu()
    center = clean_scores.median()
    scale = (
        torch.quantile(clean_scores, 0.75) - torch.quantile(clean_scores, 0.25)
    ).clamp_min(1e-6)
    return {"center": center, "scale": scale, "sample_count": torch.tensor(len(clean_scores))}
```

- [ ] **Step 4: Run kNN tests on CPU and CUDA**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty/test_knn.py -v
/home/yuchen/miniconda3/envs/UE/bin/python -c "import torch; from src.scene_uncertainty.knn import chunked_knn_distances; q=torch.randn(300,335,device='cuda'); b=torch.randn(25000,335,device='cuda'); print(chunked_knn_distances(q,b,5,4096).shape)"
```

Expected: `3 passed`, then `torch.Size([300, 5])` without an out-of-memory error.

- [ ] **Step 5: Commit**

```bash
git add src/scene_uncertainty/knn.py tests/scene_uncertainty/test_knn.py
git commit -m "feat: add chunked exact kNN scoring"
```

### Task 9: Score cached scenes across layers, policies, and aggregations

**Files:**
- Create: `src/scene_uncertainty/evaluate.py`
- Create: `tests/scene_uncertainty/test_evaluate.py`

- [ ] **Step 1: Write a failing test proving predicted class identity does not route the bank**

```python
# tests/scene_uncertainty/test_evaluate.py
import torch

from src.scene_uncertainty.evaluate import compute_query_distances, score_cached_record


def make_record(logits):
    return {
        "image_id": 7,
        "severity": 0,
        "layers": {2: torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])},
        "logits": logits,
        "matched_annotation_id": torch.tensor([10, -1, 11]),
        "predicted_class": logits.argmax(dim=-1),
    }


def test_same_persistence_and_confidence_with_different_argmax_has_same_score():
    first_logits = torch.tensor([[3.0, 2.0], [1.0, 0.0], [4.0, 1.0]])
    second_logits = torch.tensor([[2.0, 3.0], [0.0, 1.0], [1.0, 4.0]])
    bank = {2: torch.tensor([[0.0, 0.0], [3.0, 3.0]])}
    first = score_cached_record(make_record(first_logits), bank, "all", "mean", k=1)
    second = score_cached_record(make_record(second_logits), bank, "all", "mean", k=1)
    assert first["raw_score"] == second["raw_score"]


def test_query_distances_are_computed_once_for_all_queries_and_layers():
    logits = torch.tensor([[3.0, 2.0], [1.0, 0.0], [4.0, 1.0]])
    bank = {2: torch.tensor([[0.0, 0.0], [3.0, 3.0]])}
    distances = compute_query_distances(make_record(logits), bank, k=1)
    assert distances[2].shape == (3,)
    result = score_cached_record(
        make_record(logits), bank, "top20", "mean", k=1,
        query_distances=distances,
    )
    assert result["selected_count"] == 3


def test_empty_threshold_is_explicitly_invalid():
    logits = torch.full((3, 2), -20.0)
    bank = {2: torch.tensor([[0.0, 0.0], [3.0, 3.0]])}
    result = score_cached_record(make_record(logits), bank, "threshold_0.5", "mean", k=1)
    assert result["valid"] is False
    assert result["selected_count"] == 0
```

- [ ] **Step 2: Run the test and verify the missing-module failure**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty/test_evaluate.py -v
```

Expected: import fails for `evaluate`.

- [ ] **Step 3: Implement record scoring with one unified bank per layer**

```python
# src/scene_uncertainty/evaluate.py
from __future__ import annotations

import torch

from .knn import mean_knn_distance
from .query_policy import ORACLE_POLICIES, aggregate_scores, select_oracle_queries, select_queries


def compute_query_distances(
    record: dict,
    banks: dict[int, torch.Tensor],
    k: int,
    bank_chunk_size: int = 8192,
) -> dict[int, torch.Tensor]:
    distances = {}
    for layer_id in sorted(banks):
        queries = record["layers"][layer_id].float().to(banks[layer_id].device)
        distances[layer_id] = mean_knn_distance(
            queries, banks[layer_id], k, bank_chunk_size
        ).cpu()
    return distances


def score_cached_record(
    record: dict,
    banks: dict[int, torch.Tensor],
    policy: str,
    aggregation: str,
    k: int,
    bank_chunk_size: int = 8192,
    uniform_floor: float = 0.05,
    query_distances: dict[int, torch.Tensor] | None = None,
    layer_score_scales: dict[int, dict[str, torch.Tensor]] | None = None,
) -> dict:
    if policy in ORACLE_POLICIES:
        selection = select_oracle_queries(record, policy)
    else:
        selection = select_queries(record["logits"].float(), policy, uniform_floor=uniform_floor)
    matched_predictions = {
        int(annotation_id): int(predicted_class)
        for annotation_id, predicted_class in zip(
            record["matched_annotation_id"].tolist(),
            record["predicted_class"].tolist(),
        )
        if int(annotation_id) >= 0
    }
    base = {
        "image_id": int(record["image_id"]),
        "severity": int(record.get("severity", 0)),
        "source_partition": record.get("source_partition", "unknown"),
        "policy": policy,
        "aggregation": aggregation,
        "selected_count": int(selection["indices"].numel()),
        "selected_query_ids": selection["indices"].tolist(),
        "matched_predictions": matched_predictions,
        "valid": bool(selection["valid"]),
    }
    if not selection["valid"]:
        return {
            **base,
            "raw_score": float("nan"),
            "layer_scores": {},
            "clean_scaled_layer_scores": {},
        }

    if query_distances is None:
        query_distances = compute_query_distances(record, banks, k, bank_chunk_size)
    layer_scores: dict[int, float] = {}
    for layer_id in sorted(query_distances):
        selected_scores = query_distances[layer_id].index_select(0, selection["indices"])
        method = "weighted_mean" if policy.startswith("smooth_") else aggregation
        score = aggregate_scores(selected_scores, method, selection["weights"])
        layer_scores[layer_id] = float(score.item())
    if layer_score_scales is None:
        layer_score_scales = {
            layer_id: {"center": torch.tensor(0.0), "scale": torch.tensor(1.0)}
            for layer_id in layer_scores
        }
    clean_scaled_layers = {
        layer_id: (score - float(layer_score_scales[layer_id]["center"]))
        / float(layer_score_scales[layer_id]["scale"])
        for layer_id, score in layer_scores.items()
    }
    raw_score = sum(clean_scaled_layers.values()) / len(clean_scaled_layers)
    return {
        **base,
        "raw_score": raw_score,
        "layer_scores": layer_scores,
        "clean_scaled_layer_scores": clean_scaled_layers,
    }

```

- [ ] **Step 4: Run evaluation tests**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty/test_evaluate.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/scene_uncertainty/evaluate.py tests/scene_uncertainty/test_evaluate.py
git commit -m "feat: score cached persistence scenes"
```

### Task 10: Compute monotonicity, selection-overlap, and class-switch diagnostics

**Files:**
- Create: `src/scene_uncertainty/metrics.py`
- Create: `tests/scene_uncertainty/test_metrics.py`

- [ ] **Step 1: Write failing synthetic-curve tests**

```python
# tests/scene_uncertainty/test_metrics.py
import math

from src.scene_uncertainty.metrics import jaccard_overlap, monotonicity_metrics


def test_perfectly_increasing_curve_has_perfect_metrics():
    result = monotonicity_metrics([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6])
    assert result["spearman"] == 1.0
    assert result["adjacent_monotonicity"] == 1.0
    assert result["violation_magnitude"] == 0.0
    assert result["endpoint_increase"] is True


def test_input_order_does_not_change_adjacent_metrics():
    result = monotonicity_metrics([2, 0, 1], [2.0, 0.0, 1.0])
    assert result["adjacent_monotonicity"] == 1.0


def test_decrease_records_frequency_and_magnitude():
    result = monotonicity_metrics([0, 1, 2, 3], [0.0, 2.0, 1.0, 3.0])
    assert math.isclose(result["adjacent_monotonicity"], 2 / 3)
    assert result["violation_magnitude"] > 0


def test_jaccard_handles_empty_sets():
    assert jaccard_overlap([], []) == 1.0
    assert jaccard_overlap([1, 2], [2, 3]) == 1 / 3
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty/test_metrics.py -v
```

Expected: import fails for `metrics`.

- [ ] **Step 3: Implement terminal-readable metrics**

```python
# src/scene_uncertainty/metrics.py
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def jaccard_overlap(first, second) -> float:
    first_set = set(int(value) for value in first)
    second_set = set(int(value) for value in second)
    union = first_set | second_set
    if not union:
        return 1.0
    return len(first_set & second_set) / len(union)


def monotonicity_metrics(severities, scores) -> dict:
    severity = np.asarray(severities, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(values)
    severity = severity[valid]
    values = values[valid]
    order = np.argsort(severity, kind="stable")
    severity = severity[order]
    values = values[order]
    if values.size < 2:
        return {
            "spearman": float("nan"),
            "adjacent_monotonicity": float("nan"),
            "violation_magnitude": float("nan"),
            "endpoint_increase": False,
        }
    correlation = spearmanr(severity, values).statistic
    if not np.isfinite(correlation):
        correlation = 0.0
    differences = np.diff(values)
    observed_range = max(float(values.max() - values.min()), 1e-12)
    return {
        "spearman": float(correlation),
        "adjacent_monotonicity": float(np.mean(differences >= 0)),
        "violation_magnitude": float(np.maximum(-differences, 0).sum() / observed_range),
        "endpoint_increase": bool(values[-1] > values[0]),
    }
```

- [ ] **Step 4: Add a class-switch step helper test and implementation**

Append this test:

```python
from src.scene_uncertainty.metrics import has_class_switch


def test_class_switch_uses_shared_matched_annotation_ids():
    first = {10: 2, 11: 7}
    second = {10: 5, 11: 7, 12: 3}
    assert has_class_switch(first, second) is True
    assert has_class_switch({10: 2}, {10: 2}) is False
```

Append this implementation:

```python
def has_class_switch(first: dict[int, int], second: dict[int, int]) -> bool:
    shared = set(first) & set(second)
    return any(int(first[annotation_id]) != int(second[annotation_id]) for annotation_id in shared)
```

- [ ] **Step 5: Run metrics tests**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty/test_metrics.py -v
```

Expected: `5 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/scene_uncertainty/metrics.py tests/scene_uncertainty/test_metrics.py
git commit -m "feat: add blur trend metrics"
```

### Task 11: Generate lossless CSV, JSON, and class-switch-aware trend plots

**Files:**
- Create: `src/scene_uncertainty/reporting.py`
- Create: `tests/scene_uncertainty/test_reporting.py`

- [ ] **Step 1: Write failing report and structured-CSV round-trip tests**

```python
# tests/scene_uncertainty/test_reporting.py
from pathlib import Path

import json

from src.scene_uncertainty.reporting import read_result_csv, write_report, write_result_csv


def make_rows():
    rows = []
    for severity in range(6):
        rows.append({
            "image_id": 1,
            "severity": severity,
            "raw_score": float(severity),
            "layer_scores": {0: severity + 0.1, 1: severity + 0.2, 2: severity + 0.3},
            "valid": True,
            "source_partition": "tuning",
            "policy": "all",
            "aggregation": "mean",
            "selected_count": 2,
            "selected_query_ids": [0, 1],
            "matched_predictions": {10: 2 if severity < 3 else 5},
        })
    return rows


def test_structured_csv_fields_round_trip(tmp_path: Path):
    path = tmp_path / "results.csv"
    write_result_csv(make_rows(), path)
    loaded = read_result_csv(path)
    assert loaded[0]["selected_query_ids"] == [0, 1]
    assert loaded[0]["layer_scores"] == {"0": 0.1, "1": 0.2, "2": 0.3}
    assert loaded[0]["matched_predictions"] == {"10": 2}


def test_write_report_creates_raw_relative_and_switch_outputs(tmp_path: Path):
    write_report(make_rows(), tmp_path, run_metadata={"feature_cache_id": "cache", "bank_id": "bank"})
    for name in ("per_scene.csv", "summary.json", "run_metadata.json", "raw_trend.png", "relative_trend.png", "class_switch.png"):
        assert (tmp_path / name).exists()
    summary = json.loads((tmp_path / "summary.json").read_text())
    combined = next(group for group in summary["groups"] if group["score_scope"] == "combined")
    assert combined["median_spearman"] == 1.0
    assert combined["class_switch_step_count"] == 1
    assert combined["no_switch_step_count"] == 4
    assert summary["run_metadata"]["feature_cache_id"] == "cache"
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty/test_reporting.py -v
```

Expected: import fails for `reporting`.

- [ ] **Step 3: Implement JSON-safe CSV I/O, per-layer expansion, and diagnostics**

```python
# src/scene_uncertainty/reporting.py
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import has_class_switch, jaccard_overlap, monotonicity_metrics


STRUCTURED_COLUMNS = ("layer_scores", "clean_scaled_layer_scores", "selected_query_ids", "matched_predictions")


def write_result_csv(rows: list[dict], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    fields = sorted({key for row in rows for key in row})
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            for column in STRUCTURED_COLUMNS:
                if column in encoded:
                    encoded[column] = json.dumps(encoded[column], sort_keys=True)
            writer.writerow(encoded)
    os.replace(temporary, destination)


def read_result_csv(path: str | Path) -> list[dict]:
    frame = pd.read_csv(path)
    for column in STRUCTURED_COLUMNS:
        if column in frame:
            frame[column] = frame[column].map(json.loads)
    if frame["valid"].dtype == object:
        frame["valid"] = frame["valid"].map({"True": True, "False": False})
    return frame.to_dict(orient="records")


def _expanded_frame(rows: list[dict]) -> pd.DataFrame:
    layer_ids = sorted({str(layer_id) for row in rows for layer_id in row["layer_scores"]})
    expanded = []
    for row in rows:
        expanded.append({**row, "score_scope": "combined", "score": row["raw_score"]})
        scores = {str(layer_id): score for layer_id, score in row["layer_scores"].items()}
        for layer_id in layer_ids:
            expanded.append({
                **row,
                "score_scope": f"layer_{layer_id}",
                "score": scores.get(layer_id, float("nan")),
            })
    return pd.DataFrame(expanded)


def _adjacent_diagnostics(image_group: pd.DataFrame) -> list[dict]:
    ordered = image_group.sort_values("severity", kind="stable")
    score_range = max(float(ordered["score"].max() - ordered["score"].min()), 1e-12)
    result = []
    records = ordered.to_dict(orient="records")
    for previous, current in zip(records, records[1:]):
        change = float(current["score"] - previous["score"])
        result.append({
            "query_overlap": jaccard_overlap(previous["selected_query_ids"], current["selected_query_ids"]),
            "class_switch": has_class_switch(previous["matched_predictions"], current["matched_predictions"]),
            "nondecreasing": change >= 0,
            "normalized_downward_change": max(-change, 0.0) / score_range,
        })
    return result


def _safe_mean(values) -> float | None:
    finite = np.asarray(list(values), dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else None


def _safe_median(values) -> float | None:
    finite = np.asarray(list(values), dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else None


def _plot_trend(valid: pd.DataFrame, column: str, ylabel: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for keys, group in valid.groupby(["source_partition", "policy", "aggregation", "score_scope"], sort=True):
        severity = group.groupby("severity")[column]
        median, q25, q75 = severity.median(), severity.quantile(0.25), severity.quantile(0.75)
        axis.plot(median.index, median.values, marker="o", label=" / ".join(keys))
        axis.fill_between(median.index, q25.values, q75.values, alpha=0.12)
    axis.set_xlabel("Gaussian blur severity")
    axis.set_ylabel(ylabel)
    axis.legend(fontsize=6, ncol=2)
    figure.tight_layout()
    temporary = path.with_suffix(path.suffix + ".tmp")
    figure.savefig(temporary, format="png", dpi=160)
    plt.close(figure)
    os.replace(temporary, path)


def write_report(rows: list[dict], output_dir: str | Path, run_metadata: dict | None = None) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_result_csv(rows, output / "per_scene.csv")
    frame = _expanded_frame(rows)
    valid = frame[frame["valid"] & frame["score"].notna()].copy()
    clean = valid[valid["severity"] == 0][
        ["image_id", "source_partition", "policy", "aggregation", "score_scope", "score"]
    ].rename(columns={"score": "clean_score"})
    valid = valid.merge(clean, on=["image_id", "source_partition", "policy", "aggregation", "score_scope"], how="left")
    valid["clean_relative_score"] = valid["score"] - valid["clean_score"]

    groups = []
    key_columns = ["source_partition", "policy", "aggregation", "score_scope"]
    for keys, all_group in frame.groupby(key_columns, sort=True):
        group = valid[
            (valid["source_partition"] == keys[0])
            & (valid["policy"] == keys[1])
            & (valid["aggregation"] == keys[2])
            & (valid["score_scope"] == keys[3])
        ]
        image_metrics = []
        adjacent = []
        for _, image_group in group.groupby("image_id", sort=True):
            ordered = image_group.sort_values("severity", kind="stable")
            image_metrics.append(monotonicity_metrics(ordered["severity"], ordered["score"]))
            adjacent.extend(_adjacent_diagnostics(ordered))
        switched = [step for step in adjacent if step["class_switch"]]
        stable = [step for step in adjacent if not step["class_switch"]]
        groups.append({
            "source_partition": keys[0],
            "policy": keys[1],
            "aggregation": keys[2],
            "score_scope": keys[3],
            "image_count": len(image_metrics),
            "empty_selection_frequency": float((~all_group["valid"]).mean()),
            "mean_adjacent_query_overlap": _safe_mean(step["query_overlap"] for step in adjacent),
            "median_spearman": _safe_median(value["spearman"] for value in image_metrics),
            "mean_adjacent_monotonicity": _safe_mean(value["adjacent_monotonicity"] for value in image_metrics),
            "mean_violation_magnitude": _safe_mean(value["violation_magnitude"] for value in image_metrics),
            "endpoint_increase_rate": _safe_mean(value["endpoint_increase"] for value in image_metrics),
            "class_switch_step_count": len(switched),
            "class_switch_monotonicity": _safe_mean(step["nondecreasing"] for step in switched),
            "class_switch_violation_magnitude": _safe_mean(step["normalized_downward_change"] for step in switched),
            "no_switch_step_count": len(stable),
            "no_switch_monotonicity": _safe_mean(step["nondecreasing"] for step in stable),
            "no_switch_violation_magnitude": _safe_mean(step["normalized_downward_change"] for step in stable),
            "median_selected_count_by_severity": {
                str(int(severity)): float(values.median())
                for severity, values in all_group.groupby("severity")["selected_count"]
            },
        })

    summary_path = output / "summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"run_metadata": run_metadata or {}, "groups": groups}, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, summary_path)
    metadata_path = output / "run_metadata.json"
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(run_metadata or {}, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, metadata_path)
    _plot_trend(valid, "score", "Raw scene uncertainty", output / "raw_trend.png")
    _plot_trend(valid, "clean_relative_score", "Clean-relative scene uncertainty", output / "relative_trend.png")

    combined = pd.DataFrame([group for group in groups if group["score_scope"] == "combined"])
    figure, axis = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(combined))
    switch_values = [np.nan if value is None else value for value in combined["class_switch_monotonicity"]]
    stable_values = [np.nan if value is None else value for value in combined["no_switch_monotonicity"]]
    axis.bar(positions - 0.2, switch_values, width=0.4, label="class switch")
    axis.bar(positions + 0.2, stable_values, width=0.4, label="no switch")
    axis.set_xticks(positions, [
        f"{partition}/{policy}/{aggregation}"
        for partition, policy, aggregation in zip(
            combined["source_partition"], combined["policy"], combined["aggregation"]
        )
    ], rotation=90)
    axis.set_ylabel("Adjacent non-decrease rate")
    axis.legend()
    figure.tight_layout()
    temporary = output / "class_switch.png.tmp"
    figure.savefig(temporary, format="png", dpi=160)
    plt.close(figure)
    os.replace(temporary, output / "class_switch.png")
```

- [ ] **Step 4: Run the reporting tests**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty/test_reporting.py -v
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/scene_uncertainty/reporting.py tests/scene_uncertainty/test_reporting.py
git commit -m "feat: report scene uncertainty trends"
```

### Task 12: Add one CLI for selection, extraction, bank building, scoring, and reporting

**Files:**
- Create: `src/scene_uncertainty/cli.py`
- Create: `src/scene_uncertainty/pipeline.py`
- Create: `tools/scene_uncertainty.py`
- Create: `tests/scene_uncertainty/test_cli.py`

- [ ] **Step 1: Write a failing parser test covering every subcommand**

```python
# tests/scene_uncertainty/test_cli.py
from src.scene_uncertainty.cli import build_parser


def test_cli_exposes_complete_artifact_pipeline():
    parser = build_parser()
    subcommands = parser._subparsers._group_actions[0].choices
    assert set(subcommands) == {
        "select",
        "extract-reference",
        "extract-blur",
        "build-bank",
        "evaluate-knn",
        "report",
    }
```

- [ ] **Step 2: Run the parser test and verify the missing-module failure**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty/test_cli.py -v
```

Expected: import fails for `src.scene_uncertainty.cli`.

- [ ] **Step 3: Implement the parser with explicit artifact and dataset arguments**

```python
# src/scene_uncertainty/cli.py
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Class-independent persistence scene uncertainty")
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select")
    select.add_argument("--train-ann", required=True)
    select.add_argument("--val-ann", required=True)
    select.add_argument("--output", required=True)
    select.add_argument("--seed", type=int, default=42)

    for name in ("extract-reference", "extract-blur"):
        extract = subparsers.add_parser(name)
        extract.add_argument("--config", required=True)
        extract.add_argument("--checkpoint", required=True)
        extract.add_argument("--images", required=True)
        extract.add_argument("--annotations", required=True)
        extract.add_argument("--selection", required=True)
        extract.add_argument("--output", required=True)
        extract.add_argument("--device", default="cuda:0")
        extract.add_argument("--limit", type=int)
        extract.add_argument("--batch-size", type=int, default=1)
        extract.add_argument("--num-workers", type=int, default=4)
        extract.add_argument("--shard-size", type=int, default=50)

    build_bank = subparsers.add_parser("build-bank")
    build_bank.add_argument("--cache", required=True)
    build_bank.add_argument("--output", required=True)
    build_bank.add_argument("--population", choices=("natural", "coverage"), required=True)
    build_bank.add_argument("--capacity", type=int, required=True)
    build_bank.add_argument("--seed", type=int, default=42)

    evaluate = subparsers.add_parser("evaluate-knn")
    evaluate.add_argument("--cache", required=True)
    evaluate.add_argument("--bank", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--normalization", choices=("raw", "robust_z", "unit", "shape_scale"), required=True)
    evaluate.add_argument("--partition", choices=("tuning", "test", "all"), default="tuning")
    evaluate.add_argument("--policies", default="all,top10,top20,top50,threshold_0.2,threshold_0.3,threshold_0.5,smooth_1,smooth_2,oracle_matched,oracle_background,oracle_correct,oracle_incorrect")
    evaluate.add_argument("--aggregations", default="mean,median,q90,top20_mean")
    evaluate.add_argument("--k", type=int, default=5)
    evaluate.add_argument("--device", default="cuda:0")

    report = subparsers.add_parser("report")
    report.add_argument("--results", required=True)
    report.add_argument("--output", required=True)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    from .pipeline import COMMANDS
    COMMANDS[args.command](args)
```

```python
# tools/scene_uncertainty.py
from src.scene_uncertainty.cli import main


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add command implementations in a focused orchestration module**

Create `src/scene_uncertainty/pipeline.py` and wire these exact command responsibilities:

```python
# src/scene_uncertainty/pipeline.py
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import torch
from pycocotools.coco import COCO

from src.zoo.rtdetr.matcher import HungarianMatcher

from .artifacts import ShardWriter, assert_compatible, iter_records, load_manifest, manifest_id
from .bank import deterministic_reservoir, streaming_coverage_bank
from .dataset import make_coco_loader
from .evaluate import compute_query_distances, score_cached_record
from .extractor import ClassificationPersistenceExtractor
from .knn import fit_clean_distance_scale
from .normalization import fit_normalizer, transform_vectors
from .query_policy import ORACLE_POLICIES
from .reference import select_evaluation_ids, select_reference_ids
from .reporting import read_result_csv, write_report, write_result_csv
from .runtime import checkpoint_sha256, load_frozen_detector


BLUR_RADII = {0: 0.0, 1: 1.0, 2: 2.0, 3: 4.0, 4: 8.0, 5: 12.0}
DEPLOYABLE_POLICIES = ("all", "top10", "top20", "top50", "threshold_0.2", "threshold_0.3", "threshold_0.5", "smooth_1", "smooth_2")
POLICIES = DEPLOYABLE_POLICIES + ORACLE_POLICIES
AGGREGATIONS = ("mean", "median", "q90", "top20_mean")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def command_select(args) -> None:
    output = Path(args.output)
    train = COCO(args.train_ann)
    val = COCO(args.val_ann)
    _write_json(output / "reference.json", select_reference_ids(train, seed=args.seed))
    _write_json(output / "evaluation.json", select_evaluation_ids(val.getImgIds(), seed=args.seed))


def _matcher() -> HungarianMatcher:
    return HungarianMatcher(
        weight_dict={"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
        use_focal_loss=True,
        alpha=0.25,
        gamma=2.0,
    )


def _selected_image_groups(selection: dict, reference: bool, limit: int | None) -> dict[int, str]:
    if reference:
        groups = {
            **{int(image_id): "natural" for image_id in selection["natural_ids"]},
            **{int(image_id): "augmentation" for image_id in selection["augmentation_ids"]},
        }
        if limit is not None:
            groups = dict(list(groups.items())[:limit])
        return groups
    tuning = [int(value) for value in selection["tuning_pilot_ids"]]
    test = [int(value) for value in selection["test_pilot_ids"]]
    if limit is not None:
        tuning_count = min(len(tuning), (limit + 1) // 2)
        test_count = min(len(test), limit - tuning_count)
        tuning, test = tuning[:tuning_count], test[:test_count]
    return {
        **{image_id: "tuning" for image_id in tuning},
        **{image_id: "test" for image_id in test},
    }


def _extract(args, reference: bool) -> None:
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    groups = _selected_image_groups(selection, reference, args.limit)
    conditions = [(0, 0.0)] if reference else list(BLUR_RADII.items())
    device = torch.device(args.device)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = load_frozen_detector(args.config, args.checkpoint, device)
    image_ids = list(groups)
    metadata = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
        "model_config": str(Path(args.config).resolve()),
        "model_config_sha256": checkpoint_sha256(args.config),
        "annotation_file": str(Path(args.annotations).resolve()),
        "annotation_sha256": checkpoint_sha256(args.annotations),
        "image_root": str(Path(args.images).resolve()),
        "image_ids": image_ids,
        "split_seed": int(selection["seed"]),
        "source_kind": "reference" if reference else "evaluation",
        "preprocessing": "resize_640x640_then_pil_gaussian_blur_then_float_tensor",
        "corruption": {"type": "gaussian_blur", "radii": dict(conditions)},
        "persistence_extraction_version": 1,
        "classification_layers": ["decoder.score_head.0", "decoder.score_head.1", "decoder.score_head.2"],
        "decoder_layers": [0, 1, 2],
        "persistence_dim": 335,
        "query_count": 300,
    }
    output = Path(args.output)
    completed_manifest = output / "manifest.json"
    if completed_manifest.exists():
        assert_compatible(load_manifest(output), metadata, keys=metadata)
        return

    with ShardWriter(output, metadata, args.shard_size) as writer:
        completed = writer.existing_record_keys()
        with ClassificationPersistenceExtractor(model, _matcher(), [0, 1, 2]) as extractor:
            for severity, radius in conditions:
                pending_ids = [image_id for image_id in image_ids if (image_id, severity) not in completed]
                if not pending_ids:
                    continue
                loader = make_coco_loader(
                    args.images, args.annotations, pending_ids, radius,
                    args.batch_size, args.num_workers,
                )
                for samples, targets in loader:
                    samples = samples.to(device)
                    device_targets = [
                        {key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()}
                        for target in targets
                    ]
                    for record in extractor.extract(samples, device_targets):
                        for layer_id, vectors in record["layers"].items():
                            if tuple(vectors.shape) != (300, 335):
                                raise RuntimeError(
                                    f"Unexpected persistence shape at layer {layer_id}: {tuple(vectors.shape)}"
                                )
                        record["severity"] = int(severity)
                        record["blur_radius"] = float(radius)
                        record["corruption_type"] = "gaussian_blur"
                        record["source_partition"] = "reference" if reference else groups[record["image_id"]]
                        record["reference_group"] = groups[record["image_id"]] if reference else None
                        writer.add(record)
        writer.metadata["run_stats"] = {
            "wall_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
        }

def command_extract_reference(args) -> None:
    _extract(args, reference=True)


def command_extract_blur(args) -> None:
    _extract(args, reference=False)


def _atomic_torch_save(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _iter_layer_vectors(cache: str, layer_id: int, natural_only: bool):
    for record in iter_records(cache):
        if natural_only and record["reference_group"] != "natural":
            continue
        yield from record["layers"][layer_id]


def command_build_bank(args) -> None:
    started = time.perf_counter()
    output = Path(args.output)
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Bank artifact is already complete: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cache_manifest = load_manifest(args.cache)
    if cache_manifest["source_kind"] != "reference":
        raise ValueError("Banks must be built from a reference cache")
    bank_manifest = {
        "schema_version": 1,
        "artifact_type": "class_independent_query_bank",
        "source_cache_id": cache_manifest["artifact_id"],
        "source_image_ids": cache_manifest["image_ids"],
        "checkpoint_sha256": cache_manifest["checkpoint_sha256"],
        "decoder_layers": cache_manifest["decoder_layers"],
        "persistence_dim": cache_manifest["persistence_dim"],
        "query_count": cache_manifest["query_count"],
        "population": args.population,
        "capacity": args.capacity,
        "seed": args.seed,
        "class_independent": True,
        "layers": {},
    }
    for layer_id in cache_manifest["decoder_layers"]:
        if args.population == "coverage":
            bank, sampling = streaming_coverage_bank(
                iter_records(args.cache), layer_id, args.capacity,
                object_fraction=0.5, class_count=80, seed=args.seed + layer_id,
            )
        else:
            bank = deterministic_reservoir(
                _iter_layer_vectors(args.cache, layer_id, natural_only=True),
                args.capacity, args.seed + layer_id,
            )
            sampling = {"natural_vectors": int(bank.shape[0])}
        layer_path = output / f"layer_{layer_id}.pt"
        _atomic_torch_save({"vectors": bank, "sampling": sampling}, layer_path)
        bank_manifest["layers"][str(layer_id)] = {
            "path": layer_path.name,
            "vector_count": int(bank.shape[0]),
            "sampling": sampling,
        }
    bank_manifest["run_stats"] = {
        "wall_seconds": time.perf_counter() - started,
        "bank_bytes_excluding_manifest": sum(
            (output / layer["path"]).stat().st_size for layer in bank_manifest["layers"].values()
        ),
    }
    bank_manifest["artifact_id"] = manifest_id(bank_manifest)
    _write_json(output / "manifest.json", bank_manifest)


def _validate_evaluation_cache(cache_manifest: dict, bank_manifest: dict, records: list[dict]) -> None:
    assert_compatible(
        cache_manifest,
        bank_manifest,
        keys=("checkpoint_sha256", "decoder_layers", "persistence_dim", "query_count"),
    )
    overlap = set(cache_manifest["image_ids"]) & set(bank_manifest["source_image_ids"])
    if overlap:
        raise ValueError(f"Reference/evaluation image leakage: {sorted(overlap)[:10]}")
    severities: dict[int, set[int]] = {}
    for record in records:
        image_id = int(record["image_id"])
        severity = int(record["severity"])
        if tuple(record["logits"].shape) != (cache_manifest["query_count"], 80):
            raise ValueError(f"Query-count mismatch for image {image_id}, severity {severity}")
        if severity in severities.setdefault(image_id, set()):
            raise ValueError(f"Duplicate image/severity record: {(image_id, severity)}")
        severities[image_id].add(severity)
    incomplete = {image_id: sorted(values) for image_id, values in severities.items() if values != set(BLUR_RADII)}
    if incomplete:
        raise ValueError(f"Incomplete blur sweeps: {dict(list(incomplete.items())[:5])}")


def command_evaluate_knn(args) -> None:
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    bank_root = Path(args.bank)
    bank_manifest = json.loads((bank_root / "manifest.json").read_text(encoding="utf-8"))
    cache_manifest = load_manifest(args.cache)
    _validate_evaluation_cache(cache_manifest, bank_manifest, iter_records(args.cache))
    raw_banks = {
        int(layer_id): torch.load(
            bank_root / layer["path"], map_location="cpu", weights_only=False
        )["vectors"]
        for layer_id, layer in bank_manifest["layers"].items()
    }
    normalizers = {
        layer_id: fit_normalizer(bank, args.normalization)
        for layer_id, bank in raw_banks.items()
    }
    banks = {
        layer_id: transform_vectors(bank, normalizers[layer_id]).to(device)
        for layer_id, bank in raw_banks.items()
    }
    layer_score_scales = {
        layer_id: fit_clean_distance_scale(bank, args.k, seed=42 + layer_id)
        for layer_id, bank in banks.items()
    }
    policies = tuple(value.strip() for value in args.policies.split(",") if value.strip())
    aggregations_requested = tuple(value.strip() for value in args.aggregations.split(",") if value.strip())
    unknown_policies = set(policies) - set(POLICIES)
    unknown_aggregations = set(aggregations_requested) - set(AGGREGATIONS)
    if unknown_policies or unknown_aggregations:
        raise ValueError(
            f"Unknown policies={sorted(unknown_policies)}, aggregations={sorted(unknown_aggregations)}"
        )
    rows: list[dict] = []
    query_distance_rows: list[dict] = []
    for record in iter_records(args.cache):
        if args.partition != "all" and record["source_partition"] != args.partition:
            continue
        normalized = {**record, "layers": {
            layer_id: transform_vectors(record["layers"][layer_id], normalizers[layer_id])
            for layer_id in banks
        }}
        query_distances = compute_query_distances(normalized, banks, args.k)
        query_distance_rows.append({
            "image_id": int(record["image_id"]),
            "severity": int(record["severity"]),
            "source_partition": record["source_partition"],
            "query_scores_by_layer": {
                layer_id: values.to(torch.float16)
                for layer_id, values in query_distances.items()
            },
        })
        for policy in policies:
            aggregations = ("mean",) if policy.startswith("smooth_") else aggregations_requested
            for aggregation in aggregations:
                rows.append(score_cached_record(
                    normalized, banks, policy, aggregation, args.k,
                    query_distances=query_distances,
                    layer_score_scales=layer_score_scales,
                ))
    output = Path(args.output)
    write_result_csv(rows, output)
    query_distance_path = output.with_suffix(".query_distances.pt")
    _atomic_torch_save(query_distance_rows, query_distance_path)
    normalizer_path = output.with_suffix(".normalizers.pt")
    _atomic_torch_save({
        "feature_normalizers": normalizers,
        "layer_score_scales": layer_score_scales,
    }, normalizer_path)
    result_manifest = {
        "schema_version": 1,
        "artifact_type": "knn_scene_uncertainty_results",
        "feature_cache_id": cache_manifest["artifact_id"],
        "bank_id": bank_manifest["artifact_id"],
        "normalization": args.normalization,
        "source_partition": args.partition,
        "normalizer_path": normalizer_path.name,
        "combined_layer_score": "equal_mean_after_clean_reference_median_iqr_scaling",
        "query_distance_path": query_distance_path.name,
        "query_distance_record_count": len(query_distance_rows),
        "k": args.k,
        "policies": list(policies),
        "aggregations": list(aggregations_requested),
        "row_count": len(rows),
        "run_stats": {
            "wall_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
            "feature_cache": cache_manifest.get("run_stats", {}),
            "bank": bank_manifest.get("run_stats", {}),
        },
    }
    result_manifest["artifact_id"] = manifest_id(result_manifest)
    _write_json(output.with_suffix(".manifest.json"), result_manifest)


def command_report(args) -> None:
    results = Path(args.results)
    metadata_path = results.with_suffix(".manifest.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    write_report(read_result_csv(results), args.output, run_metadata=metadata)


COMMANDS = {
    "select": command_select,
    "extract-reference": command_extract_reference,
    "extract-blur": command_extract_blur,
    "build-bank": command_build_bank,
    "evaluate-knn": command_evaluate_knn,
    "report": command_report,
}
```

- [ ] **Step 5: Run the parser test and command help**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty/test_cli.py -v
/home/yuchen/miniconda3/envs/UE/bin/python tools/scene_uncertainty.py --help
```

Expected: `1 passed`; help lists all six subcommands.

- [ ] **Step 6: Commit**

```bash
git add src/scene_uncertainty/cli.py src/scene_uncertainty/pipeline.py tools/scene_uncertainty.py tests/scene_uncertainty/test_cli.py
git commit -m "feat: add scene uncertainty artifact CLI"
```

### Task 13: Run integration verification and the bounded pilot

**Files:**
- Create: `tests/scene_uncertainty/test_integration.py`
- Modify: `README.md`

- [ ] **Step 1: Add an integration test that uses tiny artifacts without downloading data**

```python
# tests/scene_uncertainty/test_integration.py
from pathlib import Path

import torch

from src.scene_uncertainty.artifacts import ShardWriter, iter_records
from src.scene_uncertainty.evaluate import score_cached_record
from src.scene_uncertainty.reporting import write_report


def test_cached_knn_report_pipeline(tmp_path: Path):
    cache = tmp_path / "cache"
    metadata = {"checkpoint_sha256": "test", "decoder_layers": [2], "query_count": 3}
    with ShardWriter(cache, metadata, shard_size=2) as writer:
        for severity in range(6):
            writer.add({
                "image_id": 1,
                "severity": severity,
                "layers": {2: torch.tensor([[severity, 0.0], [severity, 1.0], [severity, 2.0]])},
                "logits": torch.tensor([[4.0, 0.0], [3.0, 0.0], [2.0, 0.0]]),
                "matched_annotation_id": torch.tensor([10, -1, 11]),
                "predicted_class": torch.tensor([0, 0, 0]),
            })
    bank = {2: torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])}
    rows = [score_cached_record(record, bank, "all", "mean", k=1) for record in iter_records(cache)]
    report = tmp_path / "report"
    write_report(rows, report)
    assert (report / "summary.json").exists()
    assert rows[-1]["raw_score"] > rows[0]["raw_score"]
```

- [ ] **Step 2: Run all unit and integration tests**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty -v
```

Expected: every test passes with no import of `supervisely`.

- [ ] **Step 3: Add concise README commands**

Add a `Scene uncertainty pilot` section containing these commands and the statement that raw scores are not corruption probabilities:

```bash
UE_PY=/home/yuchen/miniconda3/envs/UE/bin/python
COCO=/home/yuchen/YuchenZ/Datasets/coco
CKPT=/home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth
CFG=configs/scene_uncertainty/rtdetrv2_r18vd_coco.yml
OUT=output/scene_uncertainty/pilot

$UE_PY tools/scene_uncertainty.py select \
  --train-ann $COCO/annotations/instances_train2017.json \
  --val-ann $COCO/annotations/instances_val2017.json \
  --output $OUT/splits

$UE_PY tools/scene_uncertainty.py extract-reference \
  --config $CFG --checkpoint $CKPT \
  --images $COCO/train2017 \
  --annotations $COCO/annotations/instances_train2017.json \
  --selection $OUT/splits/reference.json \
  --output $OUT/reference_cache

$UE_PY tools/scene_uncertainty.py extract-blur \
  --config $CFG --checkpoint $CKPT \
  --images $COCO/val2017 \
  --annotations $COCO/annotations/instances_val2017.json \
  --selection $OUT/splits/evaluation.json \
  --output $OUT/blur_cache

$UE_PY tools/scene_uncertainty.py build-bank \
  --cache $OUT/reference_cache \
  --population coverage --capacity 25000 \
  --output $OUT/bank_coverage_25k

$UE_PY tools/scene_uncertainty.py evaluate-knn \
  --cache $OUT/blur_cache --bank $OUT/bank_coverage_25k \
  --normalization raw --k 5 --partition tuning \
  --output $OUT/results/raw_k5.csv

$UE_PY tools/scene_uncertainty.py report \
  --results $OUT/results/raw_k5.csv \
  --output $OUT/reports/raw_k5
```

- [ ] **Step 4: Run an exact four-reference/four-evaluation smoke pipeline**

Run in a fresh output directory; `--limit 4` selects two tuning and two test images for the blur cache:

```bash
UE_PY=/home/yuchen/miniconda3/envs/UE/bin/python
COCO=/home/yuchen/YuchenZ/Datasets/coco
CKPT=/home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth
CFG=configs/scene_uncertainty/rtdetrv2_r18vd_coco.yml
SMOKE=output/scene_uncertainty/smoke

$UE_PY tools/scene_uncertainty.py select \
  --train-ann $COCO/annotations/instances_train2017.json \
  --val-ann $COCO/annotations/instances_val2017.json \
  --output $SMOKE/splits

$UE_PY tools/scene_uncertainty.py extract-reference \
  --config $CFG --checkpoint $CKPT --limit 4 \
  --batch-size 1 --num-workers 0 \
  --images $COCO/train2017 \
  --annotations $COCO/annotations/instances_train2017.json \
  --selection $SMOKE/splits/reference.json \
  --output $SMOKE/reference_cache

$UE_PY tools/scene_uncertainty.py extract-blur \
  --config $CFG --checkpoint $CKPT --limit 4 \
  --batch-size 1 --num-workers 0 \
  --images $COCO/val2017 \
  --annotations $COCO/annotations/instances_val2017.json \
  --selection $SMOKE/splits/evaluation.json \
  --output $SMOKE/blur_cache

$UE_PY tools/scene_uncertainty.py build-bank \
  --cache $SMOKE/reference_cache --population coverage --capacity 500 \
  --output $SMOKE/bank_coverage_500

$UE_PY tools/scene_uncertainty.py evaluate-knn \
  --cache $SMOKE/blur_cache --bank $SMOKE/bank_coverage_500 \
  --normalization raw --k 5 --partition all \
  --output $SMOKE/results/raw_k5.csv

$UE_PY tools/scene_uncertainty.py report \
  --results $SMOKE/results/raw_k5.csv \
  --output $SMOKE/reports/raw_k5

$UE_PY -c "from src.scene_uncertainty.artifacts import iter_records,load_manifest; r=load_manifest('$SMOKE/reference_cache'); b=load_manifest('$SMOKE/blur_cache'); rr=list(iter_records('$SMOKE/reference_cache')); br=list(iter_records('$SMOKE/blur_cache')); assert r['query_count']==300 and r['decoder_layers']==[0,1,2] and r['persistence_dim']==335; assert all(tuple(x['layers'][layer].shape)==(300,335) for x in rr for layer in (0,1,2)); assert len(br)==24 and all(sum(x['image_id']==image_id for x in br)==6 for image_id in set(x['image_id'] for x in br)); assert b['corruption']['radii']['0']==0.0; print('smoke artifact checks passed')"
```

Expected: all commands exit zero; `smoke artifact checks passed` prints; the report contains tuning and test curves for each requested policy, aggregation, and score scope. The exact severity-zero pixel equivalence is already enforced by Task 3's byte-equality test.

- [ ] **Step 5: Run the bounded 5,000-reference/500-evaluation pilot on tuning only**

Run the README commands without `--limit`; the default `--partition tuning` compares every policy on the 250 tuning images while the other 250 cached images remain held out. The report automatically writes `run_metadata.json` containing upstream wall time, peak CUDA memory, cache/bank identities, and bank bytes.

Verify these counts from the manifests:

```text
reference images: 5,000
cached evaluation images: 500
cached blur records: 3,000
scored tuning images: 250
queries per record: 300
decoder layers: 3
coverage bank vectors per layer: 25,000
```

After the decision checkpoint locks settings, run the held-out partition exactly once with explicit choices, replacing the example values with the selected policies and aggregations:

```bash
$UE_PY tools/scene_uncertainty.py evaluate-knn \
  --cache $OUT/blur_cache --bank $OUT/bank_coverage_25k \
  --normalization raw --k 5 --partition test \
  --policies top20,smooth_1 --aggregations mean,median \
  --output $OUT/results/final_raw_k5.csv
$UE_PY tools/scene_uncertainty.py report \
  --results $OUT/results/final_raw_k5.csv \
  --output $OUT/reports/final_raw_k5
```

- [ ] **Step 6: Run the final verification suite**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty -v
git diff --check
git status --short
```

Expected: all tests pass, `git diff --check` is silent, and status lists only the intended README and integration-test changes.

- [ ] **Step 7: Commit**

```bash
git add README.md tests/scene_uncertainty/test_integration.py
git commit -m "test: verify scene uncertainty pilot"
```

## Pilot decision checkpoint

Do not start the two scorer-extension plans until the pilot report answers these questions:

1. Does any raw kNN curve increase with blur on median across images?
2. Which query policy has the highest median Spearman correlation and adjacent monotonicity?
3. Do threshold policies lose queries or become empty as blur increases?
4. Does all-query or background-only oracle scoring contain useful global-blur signal?
5. Do class-switch steps have larger violations than non-switch steps despite the unified bank?
6. Does increasing the bank from 25,000 to 50,000 vectors materially change the ranking of query policies?

The checkpoint selects two or three query policies and one or two normalizations for the scorer-extension plans. It does not fit or calibrate an uncertainty head.
