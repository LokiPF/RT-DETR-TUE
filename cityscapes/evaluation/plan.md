# Cityscapes-C Numbers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute the two-axis score and every baseline on Cityscapes-C, for RT-DETRv2-R18 and YOLO11m fine-tuned on Cityscapes, and report the numbers as on COCO, with no pass/fail rule.

**Architecture:** The package learns a second benchmark.
- **Benchmark support:**
  - a `Cityscapes` dataset (city subfolders, 8 classes, no screening history), chosen by a `benchmark` setting;
  - an RT-DETR builder and tap that take any class count;
  - check stages that follow the benchmark;
  - reports that, for a benchmark without a screening history, give one image set and no decision lines.
- **The stages then run unchanged** on three configs in `cityscapes/evaluation/`: RT-DETR's own stages, the detectors stage (YOLO11m), and the IQA stage, which writes every detector's final report.

**Tech Stack:** Python 3.11, PyTorch, pycocotools, imagecorruptions, Ultralytics 8.3.235, pyiqa, pytest; the `UE` conda environment, `/home/yuchen/miniconda3/envs/UE/bin/python`.

**Spec:** `cityscapes/evaluation/design.md`. Plan 1's record of the two detectors: `cityscapes/models/README.md`.

## Global Constraints

- **No pass/fail rule.** Reports give the numbers with their intervals and no decision lines.
- **Detectors:**
  - RT-DETRv2-R18: `/home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_cityscapes_72e.pth`, resized to 640 × 640.
  - YOLO11m: `/home/yuchen/YuchenZ/lab/Detector_test/yolo11m_cityscapes_100e.pt`, letterboxed to 640 × 320.
  - Both use the taps and maps of COCO.
- **Classes:** ids 0–7, in the order person, rider, car, truck, bus, train, motorcycle, bicycle. Val annotations: `/home/yuchen/YuchenZ/Datasets/cityscape/annotations_coco/cityscapes_val_8cls.json`.
- **Clean references come from the 2,975 Cityscapes train images:**
  - the method: 2,000 + 500, the seed-44 splits;
  - kNN, the activation CDFs and Hashemi's statistics: all 2,975;
  - DisCoPatch: all 2,975, with the official settings (256 × 256, 48 patches, 65 epochs);
  - NIQE's refit and ARNIQA's prototype: all 2,975.
- **Evaluation:** all 500 val images in the seed-44 order, 5 folds, one image set (`all`).
- **Conditions:** the 96, applied at 2048 × 1024.
- **Metrics:** as on COCO.
- **The method's fixed choices are unchanged.**
- **Clean-AP floors:** RT-DETR 0.35 (0.382 − 0.03); YOLO11m 0.32 (Ultralytics' 0.356 − 0.03, rounded down).
- **COCO stays exactly as it is:**
  - `configs/coco*.toml` are unchanged;
  - COCO's protocols, summaries, reports and cross tables are byte-identical;
  - `tests/evaluation/test_report_golden.py` and `tests/test_equivalence.py` stay green.
- **Light bookkeeping:** nothing beyond what the package records already.
- **Where:**
  - The code is written in the worktree `/home/yuchen/YuchenZ/UE/philip_sa/.worktrees/cityscapes-eval`, on the branch `cityscapes-eval`, cut from `fingerprint_bank` after `iqa-baselines` is merged.
  - Configs and docs go in `cityscapes/evaluation/`.
  - Runs go to `/home/yuchen/YuchenZ/UE/philip_sa/runs/cityscapes*` (absolute paths, git-ignored).
- **Git:**
  - Commit only the paths a task names: `git add <paths> && git commit -m "…" -- <paths>`.
  - Never push. Never touch `IV_2027_Yuchen/`.
  - Every commit message ends with:

    ```
    Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
    Claude-Session: https://claude.ai/code/session_0167vRw3NLCumE8FSGAnF1Tc
    ```
- **Tests run on the CPU only:** `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`, from the worktree.
- **The GPU is shared.**
  - The controller, the session running this plan, asks the session "explore" via SendMessage before each GPU group (Tasks 8, 9 and 10), and sends "done" after each.
  - Each stage runs under its config's cap: RT-DETR stages 5.5 GiB, the detectors stage 8 GiB, the IQA stage 16 GiB
    (NIQE in float64 and CLIP on full 2048 × 1024 images). DisCoPatch's official training loop has no cap; its peak
    is measured in the smoke run and reported to explore.

## Before Task 1

1. **Confirm the IQA merge.** ue-implement messages when `iqa-baselines` has landed. Check it: `git -C /home/yuchen/YuchenZ/UE/philip_sa log --oneline fingerprint_bank -- degradation_monitor/stages/iqa.py | head -3` must list commits.
2. **Create the worktree** with superpowers:using-git-worktrees: `git -C /home/yuchen/YuchenZ/UE/philip_sa worktree add .worktrees/cityscapes-eval -b cityscapes-eval fingerprint_bank`.
3. **Run the whole suite in the worktree and note its count,** N tests collected. Every task's expected count below is relative to N.

## Review Focus

1. **A COCO run folder opened with Cityscapes settings** must be refused, naming `dataset` as the changed field, so COCO and Cityscapes results never mix. Test (Task 3): `test_a_coco_run_folder_refuses_cityscapes_settings`.
2. **City subfolders with a repeated image name** must be refused, because score files and the ground truth are keyed by name. Tests (Task 2): `test_repeated_image_names_are_refused` and `test_an_annotation_file_naming_one_image_twice_is_refused`.
3. **A checkpoint whose class count differs from the annotation file's** (e.g. the COCO checkpoint with the Cityscapes config) must be refused, naming both counts. Test (Task 4): `test_the_check_refuses_a_detector_whose_classes_differ_from_the_annotations`.
4. **COCO's reports must not change:** the same keys, decision lines and title. Test (Task 5): `test_a_coco_report_keeps_its_decisions_and_its_keys`. COCO's cross table keeps its untouched and decision columns: the existing `test_the_cross_detector_table_lists_every_detector` stays green.
5. **No Cityscapes output may mention untouched, held-out or screening images, or decisions:** not the report, not the cross table, not the IQA table. Tests (Tasks 5 and 6): `test_a_benchmark_without_a_screening_history_reports_one_image_set_and_no_decisions`, `test_the_cross_table_of_a_benchmark_without_a_screening_history_has_no_decisions` and `test_the_iqa_table_of_a_benchmark_without_a_screening_history_has_no_untouched_columns`.

---

### Task 1: The RT-DETR model and tap take any class count

**Files:**
- Modify: `degradation_monitor/detector/model.py` (`build_fixed_detector`, `load_frozen_detector`; new `COCO_CLASSES`, `SCORE_HEAD`, `checkpoint_classes`)
- Modify: `degradation_monitor/detector/taps.py` (the shape check; `CLASS_COUNT` goes)
- Test: `tests/detector/test_model.py`, `tests/detector/test_taps.py`

**Interfaces:**
- Produces:
  - `build_fixed_detector(num_classes: int = 80) -> RTDETR`;
  - `checkpoint_classes(state: Mapping[str, Tensor]) -> int`;
  - `load_frozen_detector(checkpoint_path, device)`, which builds as many classes as the checkpoint's score head has rows;
  - `DetectorTap.run` returns logits of shape (N, 300, classes) for any class count.

- [ ] **Step 1: Write the failing tests**

Append to `tests/detector/test_model.py`, and add `from pathlib import Path` to its imports:

```python
CITYSCAPES_CHECKPOINT = Path("/home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_cityscapes_72e.pth")


def test_checkpoint_classes_reads_the_decoder_score_head():
    assert extraction.checkpoint_classes({extraction.SCORE_HEAD: torch.zeros(8, 256)}) == 8
    assert extraction.checkpoint_classes({"x": torch.zeros(1)}) == 80


def test_the_fixed_detector_takes_a_class_count():
    model = extraction.build_fixed_detector(num_classes=8)
    assert model.decoder.dec_score_head[0].out_features == 8
    assert model.decoder.denoising_class_embed.num_embeddings == 9  # the classes plus the padding row


def test_the_cityscapes_checkpoint_loads_as_an_8_class_detector():
    if not CITYSCAPES_CHECKPOINT.exists():
        pytest.skip("the Cityscapes fine-tune is not available")
    model = load_frozen_detector(CITYSCAPES_CHECKPOINT, torch.device("cpu"))
    assert model.decoder.dec_score_head[0].out_features == 8
```

In the existing `test_load_frozen_detector_loads_cpu_state_and_freezes`, replace `lambda: nn.Linear(3, 2)` with `lambda num_classes=80: nn.Linear(3, 2)`.

Append to `tests/detector/test_taps.py`:

```python
class EightClassDetector(FakeDetector):
    """FakeDetector with the Cityscapes detector's 8 classes."""

    def forward(self, images):
        out = super().forward(images)
        return {**out, "pred_logits": out["pred_logits"][..., :8]}


def test_the_tap_takes_any_class_count(monkeypatch):
    monkeypatch.setattr(detector, "load_frozen_detector", lambda _p, _d: EightClassDetector())
    with detector.DetectorTap("unused.pth", "cpu", image_size=(16, 16)) as tap:
        logits, _, _ = tap.run([np.zeros((10, 10, 3), np.uint8)], batch_size=1)
    assert logits.shape == (1, 300, 8)
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/detector/test_model.py tests/detector/test_taps.py`
Expected: 4 failures:
- `AttributeError: … has no attribute 'checkpoint_classes'`;
- `TypeError: build_fixed_detector() got an unexpected keyword argument 'num_classes'`;
- a `RuntimeError` naming the size mismatch of the score heads;
- `ValueError: detector outputs do not have the expected shapes`.

- [ ] **Step 3: Implement**

In `degradation_monitor/detector/model.py`:
- **Module docstring:** change it to `"""Build the fixed RT-DETRv2-R18, load a frozen checkpoint (COCO or Cityscapes), and prepare images for it."""`.
- **Constants:** add after `IMAGE_SIZE`:

  ```python
  COCO_CLASSES = 80
  SCORE_HEAD = "decoder.dec_score_head.0.weight"  # (classes, 256) in every RT-DETRv2 checkpoint
  ```

- **The builder:** change its first line to `def build_fixed_detector(num_classes: int = COCO_CLASSES) -> RTDETR:`. In its `decoder = RTDETRTransformerv2(num_classes=80, …)` line, replace `num_classes=80` with `num_classes=num_classes` and leave every other argument as it is.
- **`checkpoint_classes`:** add after `checkpoint_state`:

  ```python
  def checkpoint_classes(state: Mapping[str, Tensor]) -> int:
      """How many classes a checkpoint detects: the rows of its decoder's first score head (COCO 80, Cityscapes 8)."""
      head = state.get(SCORE_HEAD)
      return COCO_CLASSES if head is None else int(head.shape[0])
  ```

- **The loader:** replace `load_frozen_detector` with:

  ```python
  def load_frozen_detector(checkpoint_path: str | Path, device: torch.device) -> nn.Module:
      checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
      state = checkpoint_state(checkpoint)
      model = build_fixed_detector(num_classes=checkpoint_classes(state))
      incompatible = model.load_state_dict(state, strict=False)
      if incompatible.missing_keys or incompatible.unexpected_keys:
          raise RuntimeError(f"checkpoint mismatch: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}")
      return model.to(device).eval().requires_grad_(False)
  ```

In `degradation_monitor/detector/taps.py`:
- delete the line `CLASS_COUNT = 80`;
- in `forward`, replace the condition `logits.shape != (n, QUERY_COUNT, CLASS_COUNT)` with `logits.ndim != 3 or logits.shape[:2] != (n, QUERY_COUNT)`. The class count is the checkpoint's; the check stage compares it with the annotations (Task 4).

- [ ] **Step 4: Run the tests to see them pass, then the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/detector`
Expected: PASS.

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: no failures, N + 4 collected.

- [ ] **Step 5: Commit**

```bash
git add degradation_monitor/detector/model.py degradation_monitor/detector/taps.py tests/detector/test_model.py tests/detector/test_taps.py
git commit -m "feat: the RT-DETR builder, loader and tap take the checkpoint's class count

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0167vRw3NLCumE8FSGAnF1Tc" -- degradation_monitor/detector/model.py degradation_monitor/detector/taps.py tests/detector/test_model.py tests/detector/test_taps.py
```

---

### Task 2: A Cityscapes dataset

**Files:**
- Create: `degradation_monitor/datasets/cityscapes.py`
- Modify: `degradation_monitor/datasets/coco.py`:
  - `list_images` and `evaluation_images` take `recursive`;
  - `CocoGroundTruth` finds images by base name;
  - `Coco` gets class attributes.
- Modify: `degradation_monitor/datasets/__init__.py` (the docstring)
- Test: `tests/datasets/test_cityscapes.py`

**Interfaces:**
- Produces:
  - `list_images(root, recursive: bool = False) -> list[Path]`, which raises `ValueError("image names repeat …")`;
  - `evaluation_images(val_root, *, seed, recursive=False)`;
  - `CocoGroundTruth.image_id(name)`, which takes a base file name;
  - `Coco`, with the class attributes `name = "coco"`, `classes = 80`, `min_clean_ap = 0.45`, `recursive = False` and `screened = True`;
  - `Cityscapes(Coco)`, with `name = "cityscapes"`, `classes = 8`, `min_clean_ap = 0.35`, `recursive = True` and `screened = False`;
  - `CLASSES` in `datasets.cityscapes`.

- [ ] **Step 1: Write the failing tests**

Create `tests/datasets/test_cityscapes.py`:

```python
import json
from pathlib import Path

import pytest
from PIL import Image

from degradation_monitor.datasets import coco
from degradation_monitor.datasets.cityscapes import CLASSES, Cityscapes
from degradation_monitor.datasets.coco import Coco, CocoGroundTruth, list_images

ROOT = Path("/home/yuchen/YuchenZ/Datasets/cityscape")


def _city_images(root, split, cities, per_city):
    for city in cities:
        folder = root / split / city
        folder.mkdir(parents=True)
        for index in range(per_city):
            Image.new("RGB", (8, 4)).save(folder / f"{city}_{index:06d}_000019_leftImg8bit.png")
    return root / split


def _annotations(path, file_names):
    path.write_text(json.dumps({
        "images": [{"id": i, "file_name": n, "width": 2048, "height": 1024} for i, n in enumerate(file_names)],
        "annotations": [{"id": 0, "image_id": 0, "category_id": 2, "bbox": [10, 20, 30, 40], "area": 1200,
                         "iscrowd": 0}],
        "categories": [{"id": i, "name": n} for i, n in enumerate(CLASSES)]}))
    return path


def test_cityscapes_lists_images_in_city_folders_in_a_seeded_order(tmp_path):
    val = _city_images(tmp_path, "val", ("frankfurt", "lindau"), 3)
    data = Cityscapes(tmp_path / "train", val, tmp_path / "ann.json")
    images = data.evaluation_images()
    assert sorted(p.name for p in images) == sorted(p.name for p in val.rglob("*.png"))
    assert images == Cityscapes(tmp_path / "train", val, tmp_path / "ann.json").evaluation_images()
    limited = Cityscapes(tmp_path / "train", val, tmp_path / "ann.json", limit=2).evaluation_images()
    assert limited == images[:2]
    assert data.folds().tolist() == [0, 1, 2, 3, 4, 0]


def test_cityscapes_splits_train_images_into_disjoint_reference_sets(tmp_path, monkeypatch):
    monkeypatch.setattr(coco, "SPLIT_IMAGES", (("reserved", 1), ("bank", 4), ("zstats", 2)))
    train = _city_images(tmp_path, "train", ("aachen", "bochum"), 4)
    data = Cityscapes(train, tmp_path / "val", tmp_path / "ann.json")
    splits = {name: data.reference_split(name) for name in ("reserved", "bank", "zstats")}
    assert [len(v) for v in splits.values()] == [1, 4, 2]
    assert len({p for v in splits.values() for p in v}) == 7


def test_cityscapes_ground_truth_finds_images_by_name_with_8_classes(tmp_path):
    path = _annotations(tmp_path / "ann.json", ["val/lindau/lindau_000000_000019_leftImg8bit.png"])
    gt = Cityscapes(tmp_path, tmp_path, path).ground_truth()
    boxes, labels, crowd = gt.boxes(gt.image_id("lindau_000000_000019_leftImg8bit.png"))
    assert boxes.tolist() == [[10, 20, 40, 60]] and labels.tolist() == [2] and crowd.size == 0
    assert gt.category_ids == tuple(range(8))


def test_an_annotation_file_naming_one_image_twice_is_refused(tmp_path):
    path = _annotations(tmp_path / "ann.json", ["val/a/x_leftImg8bit.png", "val/b/x_leftImg8bit.png"])
    with pytest.raises(ValueError, match="x_leftImg8bit.png appears twice"):
        CocoGroundTruth(path, expected_categories=8)


def test_repeated_image_names_are_refused(tmp_path):
    for city in ("a", "b"):
        (tmp_path / city).mkdir()
        Image.new("RGB", (4, 4)).save(tmp_path / city / "same.png")
    with pytest.raises(ValueError, match="image names repeat"):
        list_images(tmp_path, recursive=True)


def test_the_benchmarks_differ_in_layout_classes_and_history():
    assert (Coco.name, Coco.classes, Coco.recursive, Coco.screened) == ("coco", 80, False, True)
    assert (Cityscapes.name, Cityscapes.classes, Cityscapes.recursive, Cityscapes.screened) == (
        "cityscapes", 8, True, False)
    assert (Coco.min_clean_ap, Cityscapes.min_clean_ap) == (0.45, 0.35)


def test_the_real_cityscapes_folders_give_the_expected_images():
    annotations = ROOT / "annotations_coco" / "cityscapes_val_8cls.json"
    if not annotations.exists():
        pytest.skip("Cityscapes is not available")
    data = Cityscapes(ROOT / "leftImg8bit" / "train", ROOT / "leftImg8bit" / "val", annotations)
    assert len(data.train_images()) == 2975 and len(data.evaluation_images()) == 500
    assert len(data.reference_split("bank")) == 2000 and len(data.reference_split("zstats")) == 500
    gt = data.ground_truth()
    assert all(gt.image_id(p.name) is not None for p in data.evaluation_images())
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/datasets/test_cityscapes.py`
Expected: collection fails with `ModuleNotFoundError: No module named 'degradation_monitor.datasets.cityscapes'`.

- [ ] **Step 3: Implement**

In `degradation_monitor/datasets/coco.py`:
- **Imports:** change `from typing import Optional` to `from typing import ClassVar, Optional`.
- **Listing:** replace `list_images` and `evaluation_images` with:

  ```python
  def list_images(root, recursive: bool = False) -> list[Path]:
      """The images in root (and its subfolders when recursive), sorted; refused when two share a name, because result
      files and the ground truth are keyed by name."""
      root = Path(root)
      if not root.is_dir():
          raise ValueError(f"image directory does not exist: {root}")
      candidates = root.rglob("*") if recursive else root.iterdir()
      images = sorted(p for p in candidates if p.is_file() and p.suffix.lower() in _SUFFIXES)
      if len({p.stem for p in images}) != len(images):
          raise ValueError(f"image names repeat under {root}; result files are keyed by name")
      return images


  def evaluation_images(val_root, *, seed: int, recursive: bool = False) -> list[Path]:
      """All val images in the old benchmark's seeded shuffle order; the order defines the folds."""
      images = list_images(val_root, recursive)
      if not images:
          raise ValueError(f"no images found in {val_root}")
      np.random.default_rng(seed).shuffle(images)
      return images
  ```

- **Ground truth:** in `CocoGroundTruth.__init__`, replace the line `self._ids = {info["file_name"]: image_id for image_id, info in self.coco.imgs.items()}` with:

  ```python
          self._ids = {}
          for image_id, info in self.coco.imgs.items():
              name = Path(info["file_name"]).name  # Cityscapes stores split/city/name; images are looked up by name
              if name in self._ids:
                  raise ValueError(f"{name} appears twice in {annotation_file}")
              self._ids[name] = image_id
  ```

- **The `Coco` dataclass:** replace it with:

  ```python
  @dataclass(frozen=True)
  class Coco:
      """What a stage needs from the dataset: train images, reference splits, the evaluation order, folds, ground truth.

      The class attributes are what differs between benchmarks: the name, the ground truth's class count, the floor for
      the detector's clean val AP, whether the images sit in subfolders, and whether the method was screened on these
      images (then the report adds the screen, held-out and untouched sets and their pre-registered decisions).
      """
      train_root: Path
      val_root: Path
      annotations: Path
      seed: int = SEED
      limit: Optional[int] = None
      name: ClassVar[str] = "coco"
      classes: ClassVar[int] = 80
      min_clean_ap: ClassVar[float] = 0.45  # the COCO checkpoint's clean val AP is about 0.48
      recursive: ClassVar[bool] = False
      screened: ClassVar[bool] = True

      def train_images(self) -> list[Path]:
          return list_images(self.train_root, self.recursive)

      def reference_split(self, name: str) -> list[Path]:
          paths = self.train_images()
          return [paths[i] for i in train_splits(len(paths), self.seed)[name]]

      def evaluation_images(self) -> list[Path]:
          images = evaluation_images(self.val_root, seed=self.seed, recursive=self.recursive)
          return images[: self.limit] if self.limit else images

      def folds(self) -> np.ndarray:
          return assign_folds(len(self.evaluation_images()))

      def ground_truth(self) -> CocoGroundTruth:
          return CocoGroundTruth(self.annotations, expected_categories=self.classes)
  ```

Create `degradation_monitor/datasets/cityscapes.py`:

```python
"""Cityscapes (fine annotations, 8 instance classes) in the COCO-style form of datasets.coco.

The images sit in one folder per city, leftImg8bit/<split>/<city>/<name>_leftImg8bit.png. Their names are unique, so
result files and the ground truth are keyed by name, as for COCO. The annotation file is the val output of
cityscapes/models/rtdetrv2/cityscapes_converter.py: Detectron's conversion rules, category ids 0-7 in CLASSES order.
"""
from __future__ import annotations

from typing import ClassVar

from .coco import Coco

CLASSES = ("person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")


class Cityscapes(Coco):
    """Cityscapes train and val: city subfolders, 8 classes, and no screening history (one image set in the report)."""
    name: ClassVar[str] = "cityscapes"
    classes: ClassVar[int] = len(CLASSES)
    min_clean_ap: ClassVar[float] = 0.35  # the fine-tuned RT-DETR's clean val AP is 0.382
    recursive: ClassVar[bool] = True
    screened: ClassVar[bool] = False
```

Change the docstring of `degradation_monitor/datasets/__init__.py` to `"""Datasets: clean reference images, the evaluation order and folds, and ground truth (COCO, Cityscapes)."""`.

- [ ] **Step 4: Run the tests to see them pass, then the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/datasets`
Expected: PASS, the real-data test included.

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: no failures, N + 11 collected.

- [ ] **Step 5: Commit**

```bash
git add degradation_monitor/datasets/cityscapes.py degradation_monitor/datasets/coco.py degradation_monitor/datasets/__init__.py tests/datasets/test_cityscapes.py
git commit -m "feat: a Cityscapes dataset in the COCO-style form

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0167vRw3NLCumE8FSGAnF1Tc" -- degradation_monitor/datasets/cityscapes.py degradation_monitor/datasets/coco.py degradation_monitor/datasets/__init__.py tests/datasets/test_cityscapes.py
```

---

### Task 3: The settings choose the benchmark

**Files:**
- Modify: `degradation_monitor/settings.py`
- Create: `cityscapes/evaluation/cityscapes.toml`
- Test: `tests/test_settings.py`

**Interfaces:**
- Consumes: `Coco` and `Cityscapes` (Task 2).
- Produces:
  - `Settings.benchmark`, `"coco"` by default and the last field;
  - `Settings.dataset`, that benchmark's dataset;
  - `Settings.protocol()["dataset"] == settings.benchmark`;
  - `BENCHMARKS = {"coco": Coco, "cityscapes": Cityscapes}`;
  - the config `cityscapes/evaluation/cityscapes.toml`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_settings.py`, and add these imports: `from degradation_monitor.datasets.cityscapes import Cityscapes`, `from degradation_monitor.datasets.coco import Coco` and `from degradation_monitor.runs import Manifest`.

```python
def test_the_benchmark_chooses_the_dataset_and_names_the_protocol(tmp_path):
    (tmp_path / "ckpt.pth").write_bytes(b"weights")
    coco_settings = load_settings(_config(tmp_path))
    cityscapes_settings = load_settings(_config(tmp_path, CONFIG + 'benchmark = "cityscapes"\n'))
    assert type(coco_settings.dataset) is Coco and coco_settings.protocol()["dataset"] == "coco"
    assert type(cityscapes_settings.dataset) is Cityscapes and cityscapes_settings.protocol()["dataset"] == "cityscapes"
    with pytest.raises(ValueError, match="unknown benchmark 'kitti'"):
        load_settings(_config(tmp_path, CONFIG + 'benchmark = "kitti"\n'))


def test_a_coco_run_folder_refuses_cityscapes_settings(tmp_path):
    (tmp_path / "ckpt.pth").write_bytes(b"weights")
    coco_settings = load_settings(_config(tmp_path))
    Manifest(coco_settings.layout).check_protocol(coco_settings.protocol())
    cityscapes_settings = load_settings(_config(tmp_path, CONFIG + 'benchmark = "cityscapes"\n'))
    with pytest.raises(ValueError, match=r"another protocol \(dataset\)"):
        Manifest(cityscapes_settings.layout).check_protocol(cityscapes_settings.protocol())


def test_the_cityscapes_config_points_at_the_fine_tuned_detector():
    settings = load_settings(Path(__file__).resolve().parents[1] / "cityscapes" / "evaluation" / "cityscapes.toml")
    assert settings.benchmark == "cityscapes" and settings.run.name == "cityscapes"
    assert settings.checkpoint.name == "rtdetrv2_r18vd_cityscapes_72e.pth"
    assert settings.annotations.name == "cityscapes_val_8cls.json" and settings.gpu_memory_gib == 5.5
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/test_settings.py`
Expected: FAIL with `ValueError: unknown settings in …/coco.toml: benchmark`, and a missing `cityscapes.toml`.

- [ ] **Step 3: Implement**

In `degradation_monitor/settings.py`:
- **Module docstring, first line:** `"""Run settings: the benchmark, the machine's paths and run options, read from a TOML file such as configs/coco.toml.`
- **Imports and the benchmark table:**

  ```python
  from .datasets.cityscapes import Cityscapes
  from .datasets.coco import FOLDS, Coco
  ...
  BENCHMARKS = {"coco": Coco, "cityscapes": Cityscapes}
  ```

- **`Settings`:** add `benchmark: str = "coco"` as its last field, after `seed: int = 44`. Add a refusal, and make `dataset` follow it:

  ```python
      def __post_init__(self):
          if self.benchmark not in BENCHMARKS:
              raise ValueError(f"unknown benchmark {self.benchmark!r}; choose from {', '.join(sorted(BENCHMARKS))}")

      @property
      def dataset(self) -> Coco:
          return BENCHMARKS[self.benchmark](self.train_images, self.val_images, self.annotations, seed=self.seed,
                                            limit=self.limit)
  ```

- **`protocol()`:** replace `"dataset": "coco"` with `"dataset": self.benchmark`.

Create `cityscapes/evaluation/cityscapes.toml`:

```toml
# RT-DETRv2-R18 fine-tuned on Cityscapes, on Cityscapes-C (cityscapes/evaluation/design.md).
# Stages: python -m degradation_monitor <stage> --config cityscapes/evaluation/cityscapes.toml
benchmark = "cityscapes"
run = "/home/yuchen/YuchenZ/UE/philip_sa/runs/cityscapes"
checkpoint = "/home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_cityscapes_72e.pth"
train_images = "/home/yuchen/YuchenZ/Datasets/cityscape/leftImg8bit/train"
val_images = "/home/yuchen/YuchenZ/Datasets/cityscape/leftImg8bit/val"
annotations = "/home/yuchen/YuchenZ/Datasets/cityscape/annotations_coco/cityscapes_val_8cls.json"
discopatch_root = "/home/yuchen/YuchenZ/UE/DisCoPatch"
device = "cuda:0"
batch_size = 8
workers = 9
gpu_memory_gib = 5.5  # the GPU is shared with other people's jobs
epochs = 65  # DisCoPatch's official setting, as on COCO
seed = 44
```

- [ ] **Step 4: Run the tests to see them pass, then the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/test_settings.py`
Expected: PASS.

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: no failures, N + 14 collected.

- [ ] **Step 5: Commit**

```bash
git add degradation_monitor/settings.py cityscapes/evaluation/cityscapes.toml tests/test_settings.py
git commit -m "feat: the settings choose the benchmark; the Cityscapes-C config for RT-DETR

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0167vRw3NLCumE8FSGAnF1Tc" -- degradation_monitor/settings.py cityscapes/evaluation/cityscapes.toml tests/test_settings.py
```

---

### Task 4: The check stages follow the benchmark

**Files:**
- Modify: `degradation_monitor/stages/baselines.py` (`check`; `MIN_CLEAN_AP` goes)
- Modify: `degradation_monitor/stages/detectors.py` (`check`)
- Test: `tests/stages/test_baselines.py`, `tests/stages/test_detectors.py`

**Interfaces:**
- Consumes: `Settings.dataset`, with `name`, `recursive`, `min_clean_ap` and `ground_truth()` (Tasks 2 and 3).
- Produces:
  - `baselines.check(settings, manifest) -> float`:
    - lists the val images through the benchmark, so city subfolders are included;
    - refuses a class-count mismatch with `ValueError("the detector predicts N classes but <file> has M …")`;
    - uses the benchmark's floor;
    - keeps the manifest key `coco_val_ap`, which names COCO's AP@[.5:.95] metric.
  - `detectors.check(config)`, which lists the val images the same way.

- [ ] **Step 1: Write the failing tests**

Append to `tests/stages/test_baselines.py`, and add `from dataclasses import replace` to its imports:

```python
class EightClassTap(FakeTap):
    """FakeTap with the Cityscapes detector's 8 classes."""

    def run(self, arrays, batch_size=32):
        logits, boxes, pooled = super().run(arrays, batch_size)
        return logits[:, :, :8], boxes, pooled


def test_the_check_refuses_a_detector_whose_classes_differ_from_the_annotations(tmp_path, fakes):
    settings = replace(_settings(tmp_path), benchmark="cityscapes")
    names = sorted(p.name for p in settings.val_images.iterdir())
    settings.annotations.write_text(json.dumps({
        "images": [{"id": i, "file_name": f"val/x/{n}", "width": 64, "height": 48} for i, n in enumerate(names)],
        "annotations": [], "categories": [{"id": c, "name": str(c)} for c in range(8)]}))
    with pytest.raises(ValueError, match="predicts 80 classes but ann.json has 8"):
        run_stage("check", settings)


def test_the_check_reads_city_folders_and_holds_the_detector_to_the_benchmarks_floor(tmp_path, monkeypatch):
    monkeypatch.setattr(stage, "DetectorTap", EightClassTap)
    val, rng, entries = tmp_path / "cities", np.random.default_rng(1), []
    for city in ("aachen", "bochum"):
        (val / city).mkdir(parents=True)
        for index in range(2):
            name = f"{city}_{index:06d}_000019_leftImg8bit.png"
            Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)).save(val / city / name)
            entries.append({"id": len(entries), "file_name": f"val/{city}/{name}", "width": 64, "height": 48})
    settings = _settings(tmp_path, val_images=val, benchmark="cityscapes")
    settings.annotations.write_text(json.dumps({"images": entries, "annotations": [],
                                                "categories": [{"id": c, "name": str(c)} for c in range(8)]}))
    with pytest.raises(RuntimeError, match=r"clean cityscapes val AP is 0\.000; .* should reach 0\.35"):
        run_stage("check", settings)
    assert json.loads(settings.layout.manifest.read_text())["check"]["images"] == 4
```

Append to `tests/stages/test_detectors.py`:

```python
def test_the_check_reads_the_val_images_in_city_folders(tmp_path, monkeypatch):
    rng, entries = np.random.default_rng(2), []
    for city in ("aachen", "bochum"):
        (tmp_path / "val" / city).mkdir(parents=True)
        for index in range(2):
            name = f"{city}_{index:06d}_000019_leftImg8bit.png"
            Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)).save(tmp_path / "val" / city / name)
            entries.append({"id": len(entries), "file_name": f"val/{city}/{name}", "width": 64, "height": 48})
    (tmp_path / "ann.json").write_text(json.dumps({"images": entries, "annotations": [],
                                                   "categories": [{"id": c, "name": str(c)} for c in range(8)]}))
    for name in ("rtdetr.pth", "yolo.pt", "rfdetr.pth"):
        (tmp_path / name).write_bytes(name.encode())
    (tmp_path / "base.toml").write_text(BASE.format(root=tmp_path, images=tmp_path) + 'benchmark = "cityscapes"\n')
    (tmp_path / "detectors.toml").write_text(DETECTORS.format(root=tmp_path))
    monkeypatch.setattr(stage, "load_adapter", lambda name, weights, device: FakeAdapter(name))
    config = stage.load_config(tmp_path / "detectors.toml")
    stage.check(config)
    manifest = json.loads(config.settings("yolo11m").layout.manifest.read_text())
    assert manifest["check"]["images"] == 4 and manifest["protocol"]["dataset"] == "cityscapes"
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/stages/test_baselines.py tests/stages/test_detectors.py -k "classes_differ or city_folders"`
Expected: 3 failures:
- the class-count test fails with an `IndexError` or a different message, because the old check maps the fake's 80 labels onto 8 categories;
- the two city-folder tests find no images in the flat listing, so the image count is 0 and the AP message says COCO.

- [ ] **Step 3: Implement**

In `degradation_monitor/stages/baselines.py`, delete the line `MIN_CLEAN_AP = 0.45  # the checkpoint's clean COCO val AP is about 0.48`, and replace `check` with:

```python
def check(settings, manifest) -> float:
    """Every input exists, then the detector's clean AP (COCO's AP@[.5:.95]) on every val image of the benchmark.

    run_stage has already recorded the checkpoint's sha256 in the protocol. The detector must predict the annotation
    file's classes, and its AP must reach the benchmark's floor (COCO: 0.45, for a checkpoint AP of about 0.48).
    """
    for name in PATH_FIELDS:
        if name != "run" and not getattr(settings, name).exists():
            raise ValueError(f"{name} does not exist: {getattr(settings, name)}")
    dataset = settings.dataset
    gt = dataset.ground_truth()
    images = list_images(settings.val_images, dataset.recursive)
    results = []
    cap_gpu_memory(settings.device, settings.gpu_memory_gib)
    with DetectorTap(settings.checkpoint, settings.device) as tap:
        for start in range(0, len(images), settings.batch_size):
            chunk = images[start:start + settings.batch_size]
            arrays = [open_rgb(p) for p in chunk]
            logits, boxes, _ = tap.run(arrays, settings.batch_size)
            if logits.shape[-1] != len(gt.category_ids):
                raise ValueError(f"the detector predicts {logits.shape[-1]} classes but {settings.annotations.name} "
                                 f"has {len(gt.category_ids)}: check the checkpoint and the benchmark in the config")
            for path, array, l, b in zip(chunk, arrays, logits, boxes):
                s, labels, xyxy = top_detections(l, b, image_size(array), TOP_K)
                results += coco_results(gt.image_id(path.name), s, labels, xyxy, gt.category_ids)
    ap = coco_map(gt, results, [gt.image_id(p.name) for p in images])
    manifest.update(check={"coco_val_ap": ap, "images": len(images)})
    print(f"[check] clean {dataset.name} val AP = {ap:.4f} on {len(images)} images", flush=True)
    if ap < dataset.min_clean_ap:
        raise RuntimeError(f"clean {dataset.name} val AP is {ap:.3f}; this benchmark's checkpoint should reach "
                           f"{dataset.min_clean_ap}")
    return ap
```

In `degradation_monitor/stages/detectors.py`, `check`:
- **Docstring:** `"""Every detector's clean AP on every val image of the benchmark (COCO's AP@[.5:.95]); refuses one below its floor."""`.
- **The listing:** replace `images = list_images(settings.val_images)` with `images = list_images(settings.val_images, settings.dataset.recursive)`.
- **The print:** make it `print(f"[check] {name}: clean {settings.dataset.name} val AP = {ap:.4f} on {len(images)} images", flush=True)`.
- **The final error:** make it `raise RuntimeError("clean val AP below the floor: " + "; ".join(failed))`.

- [ ] **Step 4: Run the tests to see them pass, then the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/stages/test_baselines.py tests/stages/test_detectors.py`
Expected: PASS.

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: no failures, N + 17 collected.

- [ ] **Step 5: Commit**

```bash
git add degradation_monitor/stages/baselines.py degradation_monitor/stages/detectors.py tests/stages/test_baselines.py tests/stages/test_detectors.py
git commit -m "feat: the check stages read the benchmark's folders and floor, and refuse a class-count mismatch

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0167vRw3NLCumE8FSGAnF1Tc" -- degradation_monitor/stages/baselines.py degradation_monitor/stages/detectors.py tests/stages/test_baselines.py tests/stages/test_detectors.py
```

---

### Task 5: Reports for a benchmark without a screening history

**Files:**
- Modify: `degradation_monitor/evaluation/report.py` (`TITLES`; `build_tables` takes `sets` and `benchmark`; `markdown`)
- Modify: `degradation_monitor/stages/report.py` (`write_report`)
- Test: `tests/evaluation/test_report.py`, `tests/stages/test_report.py`

**Interfaces:**
- Consumes: `Settings.dataset`, with `name`, `screened` and `evaluation_images()` (Tasks 2 and 3).
- Produces:
  - `TITLES = {"coco": "COCO", "cityscapes": "Cityscapes-C"}` in `evaluation.report`;
  - `build_tables(scores, detector, ap, folds, condition_map, *, seed, samples=BOOTSTRAP_SAMPLES, workers=1, timing=None, sets=None, benchmark="coco")`:
    - with `sets=None`, COCO's four sets and both decisions, as before, byte-identical;
    - with a dict of sets, those sets only; the summary then has no `headline_decision`, `level_decision`, `screen_images` or `untouched_start`, and gains `"benchmark": benchmark`;
  - `write_report` writes to `reports/<benchmark>/` unless given `out`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/evaluation/test_report.py`. It already imports `numpy as np`, `corruptions`, `assign_folds` and `report`:

```python
def _plain_inputs(images=30):
    """Three scored rows with a severity signal, and the kNN distances the report's kNN table reads."""
    rng = np.random.default_rng(1)
    severity = np.array([s for _, s in corruptions.CONDITIONS], float)
    scores = {row: rng.normal(0, 1, (images, 1)) + strength * severity[None] + rng.normal(0, 1, (images, 96))
              for row, strength in (("two_axis", 0.5), ("level", 0.3), ("cdf", 0.1))}
    detector = {"knn": np.sort(rng.uniform(0.5, 1.5, (images, 96, 200)), axis=2).astype(np.float32)}
    return scores, detector, np.full(images, np.nan), 0.5 - 0.05 * severity


def test_a_benchmark_without_a_screening_history_reports_one_image_set_and_no_decisions():
    scores, detector, ap, condition_map = _plain_inputs()
    tables, summary = report.build_tables(scores, detector, ap, assign_folds(30), condition_map, seed=44, samples=10,
                                          workers=1, sets={"all": np.arange(30)}, benchmark="cityscapes")
    assert summary["image_sets"] == {"all": 30} and summary["benchmark"] == "cityscapes"
    assert not {"headline_decision", "level_decision", "screen_images", "untouched_start"} & set(summary)
    assert "two_axis - cdf:auroc_common" in summary["intervals"]["all"]
    text = report.markdown(summary, tables)
    assert text.startswith("# Corruption detection on Cityscapes-C: our method and ")
    assert not any(word in text.lower() for word in ("untouched", "held-out", "screen", "pre-registered"))


def test_a_coco_report_keeps_its_decisions_and_its_keys():
    scores, detector, ap, condition_map = _plain_inputs()
    tables, summary = report.build_tables(scores, detector, ap, assign_folds(30), condition_map, seed=44, samples=10,
                                          workers=1)
    assert "benchmark" not in summary
    assert {"headline_decision", "level_decision", "screen_images", "untouched_start"} <= set(summary)
    text = report.markdown(summary, tables)
    assert text.startswith("# Corruption detection on COCO: our method and ")
    assert "**Headline (two-axis score, all images and the untouched ones):**" in text
```

Append to `tests/stages/test_report.py`:

```python
class EightClassTap(QualityTap):
    """QualityTap with the Cityscapes detector's 8 classes."""

    def run(self, arrays, batch_size=32):
        logits, boxes, pooled = super().run(arrays, batch_size)
        return logits[:, :, :8], boxes, pooled


def test_the_report_stage_writes_a_cityscapes_report_with_one_image_set(tmp_path, monkeypatch):
    val = tmp_path / "val" / "city"
    val.mkdir(parents=True)
    rng = np.random.default_rng(3)
    images = []
    for index in range(IMAGES):
        name = f"city_{index:06d}_000019_leftImg8bit.png"
        Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)).save(val / name)
        images.append({"id": index, "file_name": f"val/city/{name}", "width": 64, "height": 48})
    (tmp_path / "ann.json").write_text(json.dumps({
        "images": images,
        "annotations": [{"id": i, "image_id": i, "category_id": 0, "bbox": [8, 6, 24, 18], "area": 432, "iscrowd": 0}
                        for i in range(IMAGES)],
        "categories": [{"id": c, "name": f"c{c}"} for c in range(8)]}))
    (tmp_path / "ckpt.pth").write_bytes(b"weights")
    monkeypatch.setattr(baseline_stages, "DetectorTap", EightClassTap)
    monkeypatch.setattr(tables, "BOOTSTRAP_SAMPLES", 5)
    settings = Settings(run=tmp_path / "run", checkpoint=tmp_path / "ckpt.pth", train_images=tmp_path,
                        val_images=tmp_path / "val", annotations=tmp_path / "ann.json", discopatch_root=tmp_path,
                        benchmark="cityscapes", limit=IMAGES, workers=0, device="cpu")
    settings.layout.knn_bank.parent.mkdir(parents=True)
    np.save(settings.layout.knn_bank, rng.normal(size=(256, 512)).astype(np.float16))
    run_stage("detector-pass", settings)
    run_stage("report", settings)
    summary = json.loads((settings.layout.report("cityscapes") / "summary.json").read_text())
    assert summary["image_sets"] == {"all": IMAGES} and summary["benchmark"] == "cityscapes"
    assert "headline_decision" not in summary and not settings.layout.report("coco").exists()
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/evaluation/test_report.py tests/stages/test_report.py -k "screening_history or keeps_its_decisions or cityscapes_report"`
Expected:
- `TypeError: build_tables() got an unexpected keyword argument 'sets'` for the first test;
- the COCO test passes already, because it pins today's behaviour;
- `FileNotFoundError` for `reports/cityscapes/summary.json` in the stage test.

- [ ] **Step 3: Implement**

In `degradation_monitor/evaluation/report.py`:
- **Module docstring:** add, after the paragraph that ends "It also gives the two pre-registered decisions, each condition's mAP and the kNN baseline's k.": `A benchmark without a screening history (Cityscapes-C) passes its own image sets instead and gets no decisions.`
- **Titles:** add after `NUMBER_WORDS`:

  ```python
  TITLES = {"coco": "COCO", "cityscapes": "Cityscapes-C"}  # report titles by benchmark
  ```

- **`build_tables`'s signature and docstring:**

  ```python
  def build_tables(scores: dict, detector: dict, ap, folds, condition_map, *, seed: int,
                   samples: int = BOOTSTRAP_SAMPLES, workers: int = 1, timing=None, sets=None,
                   benchmark: str = "coco") -> tuple[dict, dict]:
      """Every table of the report, and its summary.

      `scores` holds every present row except ContrastiveConf as (images, 96) arrays in evaluation order.
      ContrastiveConf is built for each image set, and in every draw, from the detector's Conf+ and Conf- and the
      clean images' AP. `sets` is None for COCO, whose screen, held-out and untouched sets and pre-registered decisions
      come from its history; a benchmark without that history passes its own sets and gets no decisions.
      """
  ```

- **Choosing the sets:** in its body, replace `sets = image_sets(len(folds))` with:

  ```python
      screened = sets is None
      sets = image_sets(len(folds)) if screened else sets
  ```

- **The summary:** after the `summary = {…}` literal, before `return tables, summary`, add:

  ```python
      if not screened:
          for key in ("screen_images", "untouched_start", "headline_decision", "level_decision"):
              del summary[key]
          summary["benchmark"] = benchmark
  ```

- **`_set_title`:** replace it with this version, which formats only the title asked for. The old dict built every set's title eagerly, so it read `untouched_start` and `screen_images` even for a benchmark that has neither, and raised `KeyError`. COCO's titles are unchanged, character for character.

  ```python
  def _set_title(summary: dict, name: str) -> str:
      n = summary["image_sets"][name]
      if name == "all":
          return f"All {n} images (the headline, on the same images as every baseline)"
      if name == "untouched":
          return (f"Untouched images ({n}, positions {summary['untouched_start']} and later, read by nobody "
                  "while the method was designed)")
      if name == "held_out":
          return (f"Held-out images ({n}, positions {summary['screen_images']} and later: the pre-registered "
                  "check of the level score)")
      return f"The {n} screening images"
  ```

- **`markdown`'s opening lines:** replace the first `lines = [...]` statement, from the title to the bootstrap sentence, with:

  ```python
      benchmark = summary.get("benchmark", "coco")
      lines = [f"# Corruption detection on {TITLES.get(benchmark, benchmark)}: our method and "
               f"{NUMBER_WORDS[families]} baseline{'' if families == 1 else 's'}", ""]
      if "headline_decision" in summary:
          lines += [f"**Headline (two-axis score, all images and the untouched ones):** {summary['headline_decision']}.",
                    "", f"**Pre-registered level score (held-out images):** {summary['level_decision']}.", ""]
      lines += ["Every score is oriented so that higher means more likely corrupted. ↑ higher is better, ↓ lower is "
                "better; an AUROC of 0.5 is chance. Brackets are 95% paired bootstrap intervals over images "
                f"({summary['bootstrap_samples']} draws, seed {summary['seed']}).", ""]
  ```

In `degradation_monitor/stages/report.py`:
- **Module docstring:** `"""The report stage: read every stored score and write the report to reports/<benchmark>/, or to another folder (out)."""`.
- **In `write_report`:**
  - replace `names = [p.name for p in settings.dataset.evaluation_images()]` with:

    ```python
        dataset = settings.dataset
        names = [p.name for p in dataset.evaluation_images()]
    ```

  - replace `gt = settings.dataset.ground_truth()` with `gt = dataset.ground_truth()`.
  - replace the `build_tables` call with:

    ```python
        sets = None if dataset.screened else {"all": np.arange(len(names))}
        report_tables, summary = tables.build_tables(scores, detector, ap, dataset.folds(), condition_map,
                                                     seed=settings.seed, samples=tables.BOOTSTRAP_SAMPLES,
                                                     workers=settings.workers, timing=timing, sets=sets,
                                                     benchmark=dataset.name)
    ```

  - replace its last two lines (`tables.write_outputs(...)` and the `print`) with:

    ```python
        tables.write_outputs(out or layout.report(dataset.name), report_tables, summary)
        if "headline_decision" in summary:
            print(f"[report] headline: {summary['headline_decision']}; level score: {summary['level_decision']}",
                  flush=True)
        else:
            print(f"[report] {dataset.name}: {summary['images']} images, one image set, no decision rule", flush=True)
    ```

- [ ] **Step 4: Run the tests to see them pass, then the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/evaluation tests/stages/test_report.py`
Expected: PASS, `test_report_golden.py` included, unchanged.

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: no failures, N + 20 collected.

- [ ] **Step 5: Commit**

```bash
git add degradation_monitor/evaluation/report.py degradation_monitor/stages/report.py tests/evaluation/test_report.py tests/stages/test_report.py
git commit -m "feat: a benchmark without a screening history gets one image set and no decisions

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0167vRw3NLCumE8FSGAnF1Tc" -- degradation_monitor/evaluation/report.py degradation_monitor/stages/report.py tests/evaluation/test_report.py tests/stages/test_report.py
```

---

### Task 6: The cross table and the IQA stage follow the benchmark

**Files:**
- Modify: `degradation_monitor/stages/detectors.py` (`cross_table`; imports)
- Modify: `degradation_monitor/stages/iqa.py` (`_manifest`, `iqa_table`)
- Test: `tests/stages/test_detectors.py`, `tests/stages/test_iqa.py`

**Interfaces:**
- Consumes: `TITLES` and `NUMBER_WORDS` from `evaluation.report` (Task 5); `Settings.benchmark` and `dataset.screened` (Tasks 2 and 3).
- Produces:
  - **`cross_table(config)`:** reads each report from `reports/<benchmark>/`. For an unscreened benchmark, it has no untouched or decision columns, in the csv or in `summary.md`. For COCO, the output is byte-identical.
  - **The IQA protocol** records `"dataset": base.benchmark`.
  - **`iqa_table(config)`:** for an unscreened benchmark, it has no untouched columns. For COCO, the output is byte-identical.

- [ ] **Step 1: Write the failing tests**

In `tests/stages/test_detectors.py`, replace `_write_summary` with this version, which can leave out the decisions:

```python
def _write_summary(folder, two_axis, cdf, subsets=("all", "untouched"), level="confirmed", decisions=True):
    head = {"two_axis": {"auroc_common": two_axis, "auroc_extra": two_axis - 0.05},
            "cdf": {"auroc_common": cdf, "auroc_extra": cdf - 0.02},
            "cdf_sum": {"auroc_common": cdf + 0.01, "auroc_extra": cdf},  # sensitivity rows, ahead of the CDFs
            "hashemi_enc": {"auroc_common": cdf + 0.02, "auroc_extra": cdf},
            "saod_top3": {"auroc_common": 0.6, "auroc_extra": 0.6}}
    folder.mkdir(parents=True, exist_ok=True)
    summary = {"images": 5000, "clean_map": 0.45, "headline": {s: head for s in subsets},
               "by_severity": {"all": {"two_axis": {"common": [two_axis - 0.1] * 5}}}}
    if decisions:
        summary.update(headline_decision="confirmed", level_decision=level)
    (folder / "summary.json").write_text(json.dumps(summary))
```

Append to `tests/stages/test_detectors.py`:

```python
def test_the_cross_table_of_a_benchmark_without_a_screening_history_has_no_decisions(config):
    config = replace(config, base=replace(config.base, benchmark="cityscapes"))
    _write_summary(RunLayout(config.reference_run).report("cityscapes"), 0.88, 0.80, subsets=("all",),
                   decisions=False)
    for name in config.detectors:
        _write_summary(config.settings(name).layout.report("cityscapes"), 0.85, 0.79, subsets=("all",),
                       decisions=False)
    rows = stage.cross_table(config)
    assert [r["detector"] for r in rows] == ["rtdetrv2_r18", "yolo11m", "rfdetr_m"]
    assert "headline_decision" not in rows[0] and "untouched_two_axis_auroc_common" not in rows[0]
    text = (config.run / "summary.md").read_text()
    assert text.startswith("# The two-axis score on three Cityscapes-C detectors")
    assert "ntouched" not in text and "Headline" not in text
    assert "| yolo11m | 5000 | 0.450 | 0.850 / 0.800 |" in text
```

Append to `tests/stages/test_iqa.py`:

```python
def test_the_iqa_protocol_names_the_benchmark(config):
    cityscapes = replace(config, base=replace(config.base, benchmark="cityscapes"))
    stage._manifest(cityscapes, FakeIqa("cpu"))
    assert json.loads(cityscapes.layout.manifest.read_text())["protocol"]["dataset"] == "cityscapes"


def test_the_iqa_table_of_a_benchmark_without_a_screening_history_has_no_untouched_columns(config):
    config = replace(config, base=replace(config.base, benchmark="cityscapes"))
    head = {row: {"auroc_common": 0.6, "auroc_extra": 0.55} for row in ROWS}
    cell = {"point": 0.3, "low": 0.28, "high": 0.32}
    intervals = {"all": {f"two_axis - {row}:auroc_{g}": cell for row in ROWS for g in ("common", "extra")}}
    for name in config.detectors:
        folder = config.layout.report(name)
        folder.mkdir(parents=True)
        (folder / "summary.json").write_text(json.dumps({"headline": {"all": head}, "intervals": intervals}))
    rows = stage.iqa_table(config)
    assert not any(key.startswith("untouched") for key in rows[0])
    lines = (config.run / "summary.md").read_text().splitlines()
    assert "untouched" not in "\n".join(lines)
    assert lines[4] == "| Detector | Row | AUROC all: common / extra | Two-axis − row, all: common | extra |"
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/stages/test_detectors.py tests/stages/test_iqa.py -k "screening_history or names_the_benchmark"`
Expected: 3 failures:
- `FileNotFoundError` for `reports/coco/summary.json` in the cross table;
- `protocol["dataset"] == "coco"`;
- the IQA rows still hold their `untouched_…` keys.

- [ ] **Step 3: Implement**

In `degradation_monitor/stages/detectors.py`:
- **Imports:** change `from ..evaluation.report import OURS` to `from ..evaluation.report import NUMBER_WORDS, OURS, TITLES`.
- **`cross_table`:** replace it with:

```python
def cross_table(config) -> list:
    """RT-DETR's and every detector's headline: the two-axis score, the CDFs, the best baseline and the clean mAP.

    A benchmark without a screening history has neither untouched images nor decisions: their columns are left out.
    """
    dataset = config.base.dataset
    subsets = ("all", "untouched") if dataset.screened else ("all",)
    sources = {"rtdetrv2_r18": RunLayout(config.reference_run).report(dataset.name) / "summary.json"}
    sources.update({name: config.settings(name).layout.report(dataset.name) / "summary.json"
                    for name in config.detectors})
    rows = []
    for name, path in sources.items():
        summary = json.loads(path.read_text())
        row = {"detector": name, "images": summary["images"], "clean_map": summary["clean_map"]}
        if dataset.screened:
            row.update(headline_decision=summary["headline_decision"], level_decision=summary["level_decision"])
        for subset in subsets:
            for method in ("two_axis", "cdf"):
                for group in ("common", "extra"):
                    row[f"{subset}_{method}_auroc_{group}"] = _auroc(summary, subset, method, group)
        row["two_axis_severity1_common"] = summary["by_severity"]["all"]["two_axis"]["common"][0]
        head = summary["headline"]["all"]
        best = max((m for m in head if m not in OURS + SENSITIVITY_ROWS), key=lambda m: head[m]["auroc_common"])
        row["best_baseline"], row["best_baseline_auroc_common"] = best, head[best]["auroc_common"]
        rows.append(row)
    config.run.mkdir(parents=True, exist_ok=True)
    with (config.run / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    count = NUMBER_WORDS[len(rows)] if len(rows) < len(NUMBER_WORDS) else str(len(rows))
    lines = [f"# The two-axis score on {count} {TITLES.get(dataset.name, dataset.name)} detectors", ""]
    if dataset.screened:
        lines += ["AUROC, common / extra families. Untouched: positions 1970 and later.", "",
                  "| Detector | Images | Clean mAP | Two-axis, all | Two-axis, untouched | Two-axis, severity 1 common "
                  "| CDFs, all | Best baseline, common | Headline | Level rule |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
    else:
        lines += ["AUROC, common / extra families, on all images.", "",
                  "| Detector | Images | Clean mAP | Two-axis | Two-axis, severity 1 common | CDFs "
                  "| Best baseline, common |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        cells = [r["detector"], str(r["images"]), _cell(r["clean_map"]),
                 f"{_cell(r['all_two_axis_auroc_common'])} / {_cell(r['all_two_axis_auroc_extra'])}"]
        if dataset.screened:
            cells.append(f"{_cell(r['untouched_two_axis_auroc_common'])} / "
                         f"{_cell(r['untouched_two_axis_auroc_extra'])}")
        cells += [_cell(r["two_axis_severity1_common"]),
                  f"{_cell(r['all_cdf_auroc_common'])} / {_cell(r['all_cdf_auroc_extra'])}",
                  f"{r['best_baseline']} {_cell(r['best_baseline_auroc_common'])}"]
        if dataset.screened:
            cells += [r["headline_decision"], r["level_decision"]]
        lines.append("| " + " | ".join(cells) + " |")
    (config.run / "summary.md").write_text("\n".join(lines) + "\n")
    return rows
```

The real COCO run has four rows, so its title stays `# The two-axis score on four COCO detectors`, and its lines are the old ones character for character.

In `degradation_monitor/stages/iqa.py`:
- **`_manifest`:** replace `"dataset": "coco",` with `"dataset": base.benchmark,`.
- **`iqa_table`:** replace it with:

```python
def iqa_table(config) -> list:
    """For every detector and row: the row's AUROC and the two-axis score minus it, on all images and, for a benchmark
    with a screening history, on the untouched ones."""
    subsets = ("all", "untouched") if config.base.dataset.screened else ("all",)
    rows = []
    for name in config.detectors:
        summary = json.loads((config.layout.report(name) / "summary.json").read_text())
        headline, intervals = summary["headline"], summary["intervals"]
        for row in ROWS:
            entry = {"detector": name, "row": row, "label": LABELS[row]}
            for subset in subsets:
                for group in ("common", "extra"):
                    entry[f"{subset}_auroc_{group}"] = headline.get(subset, {}).get(row, {}).get(f"auroc_{group}")
            for subset in subsets:
                for group in ("common", "extra"):
                    cell = intervals.get(subset, {}).get(f"two_axis - {row}:auroc_{group}")
                    key = f"{subset}_two_axis_minus_{group}"
                    entry[key] = None if cell is None else cell["point"]
                    entry[f"{key}_low"] = None if cell is None else cell["low"]
                    entry[f"{key}_high"] = None if cell is None else cell["high"]
            rows.append(entry)
    config.run.mkdir(parents=True, exist_ok=True)
    temporary = config.run / ".summary.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(config.run / "summary.csv")
    where = "on all images and on the untouched ones" if len(subsets) == 2 else "on all images"
    if len(subsets) == 2:
        header = ("| Detector | Row | AUROC all: common / extra | untouched: common / extra | Two-axis − row, all: "
                  "common | extra | untouched: common | extra |")
    else:
        header = "| Detector | Row | AUROC all: common / extra | Two-axis − row, all: common | extra |"
    lines = ["# The two-axis score against four image-quality baselines", "",
             f"Each row's AUROC, and the two-axis score minus it with 95% paired bootstrap intervals, {where}.", "",
             header, "|" + "---|" * (2 + 3 * len(subsets))]
    for r in rows:
        aurocs = [" / ".join("–" if r[f"{s}_auroc_{g}"] is None else f"{r[f'{s}_auroc_{g}']:.3f}"
                             for g in ("common", "extra")) for s in subsets]
        cells = [f"{_cell(r[k])} [{_cell(r[k + '_low'])}, {_cell(r[k + '_high'])}]"
                 for k in (f"{s}_two_axis_minus_{g}" for s in subsets for g in ("common", "extra"))]
        lines.append(f"| {r['detector']} | {SHORT_LABELS[r['row']]} | " + " | ".join(aurocs + cells) + " |")
    temporary = config.run / ".summary.md.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(config.run / "summary.md")
    return rows
```

For COCO the header, the separator (8 columns), the sentence and every row are the old ones, character for character.

- [ ] **Step 4: Run the tests to see them pass, then the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/stages/test_detectors.py tests/stages/test_iqa.py`
Expected: PASS, the existing COCO table tests included.

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: no failures, N + 23 collected.

- [ ] **Step 5: Commit**

```bash
git add degradation_monitor/stages/detectors.py degradation_monitor/stages/iqa.py tests/stages/test_detectors.py tests/stages/test_iqa.py
git commit -m "feat: the cross table and the IQA stage follow the benchmark

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0167vRw3NLCumE8FSGAnF1Tc" -- degradation_monitor/stages/detectors.py degradation_monitor/stages/iqa.py tests/stages/test_detectors.py tests/stages/test_iqa.py
```

---

### Task 7: The YOLO11m and IQA configs, and the old plan's pointer

**Files:**
- Create: `cityscapes/evaluation/cityscapes-detectors.toml`, `cityscapes/evaluation/cityscapes-iqa.toml`
- Modify: `docs/superpowers/plans/2026-10-04-cityscapes-c-evaluation.md` (one line at the top)
- Test: `tests/test_cityscapes_configs.py`

**Interfaces:**
- Consumes: `cityscapes/evaluation/cityscapes.toml` (Task 3); `detectors.load_config` and `iqa.load_config`.
- Produces: the two configs that Tasks 9 and 10 run.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cityscapes_configs.py`:

```python
"""The Cityscapes-C configs load and point at each other (cityscapes/evaluation/design.md)."""
from pathlib import Path

from degradation_monitor.stages import detectors, iqa

FOLDER = Path(__file__).resolve().parents[1] / "cityscapes" / "evaluation"


def test_the_detectors_config_runs_yolo11m_on_cityscapes_against_rtdetrs_run():
    config = detectors.load_config(FOLDER / "cityscapes-detectors.toml")
    assert config.base.benchmark == "cityscapes" and config.detectors == ("yolo11m",)
    assert config.weights["yolo11m"].name == "yolo11m_cityscapes_100e.pt" and config.floors == {"yolo11m": 0.32}
    assert config.reference_run == config.base.run and config.run.name == "cityscapes-detectors"


def test_the_iqa_config_scores_both_detectors_on_cityscapes():
    config = iqa.load_config(FOLDER / "cityscapes-iqa.toml")
    assert config.base.benchmark == "cityscapes" and list(config.detectors) == ["rtdetrv2_r18", "yolo11m"]
    assert config.run.name == "cityscapes-iqa" and config.reference_run == config.base.run
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/test_cityscapes_configs.py`
Expected: FAIL with `FileNotFoundError` for both configs.

- [ ] **Step 3: Write the configs and the pointer**

`cityscapes/evaluation/cityscapes-detectors.toml`:

```toml
# YOLO11m fine-tuned on Cityscapes, on Cityscapes-C (cityscapes/evaluation/design.md). Stages:
# python -m degradation_monitor.stages.detectors <check|fit|pass|report> --config cityscapes/evaluation/cityscapes-detectors.toml
base = "cityscapes.toml"  # the dataset, the seed, the device and the workers
run = "/home/yuchen/YuchenZ/UE/philip_sa/runs/cityscapes-detectors"  # one folder per detector
reference_run = "/home/yuchen/YuchenZ/UE/philip_sa/runs/cityscapes"  # RT-DETR's run: the digests and DisCoPatch's scores
gpu_memory_gib = 8.0  # the card is shared

[weights]
yolo11m = "/home/yuchen/YuchenZ/lab/Detector_test/yolo11m_cityscapes_100e.pt"

[clean_ap_floor]  # Ultralytics' val AP of the frozen weights (0.356) minus 0.03, rounded down
yolo11m = 0.32
```

`cityscapes/evaluation/cityscapes-iqa.toml`:

```toml
# The four image-quality baselines on Cityscapes-C (cityscapes/evaluation/design.md). Stages:
# python -m degradation_monitor.stages.iqa <fit|pass|report> --config cityscapes/evaluation/cityscapes-iqa.toml
base = "cityscapes.toml"  # the dataset, its train and evaluation images, the seed, the device and the workers
run = "/home/yuchen/YuchenZ/UE/philip_sa/runs/cityscapes-iqa"
reference_run = "/home/yuchen/YuchenZ/UE/philip_sa/runs/cityscapes"  # RT-DETR's run: the digests; also a detector run
detectors_config = "cityscapes-detectors.toml"  # YOLO11m's run folder
gpu_memory_gib = 16.0  # the card is shared; NIQE (float64) and CLIP read full 2048 x 1024 images
```

Insert as the first line of `docs/superpowers/plans/2026-10-04-cityscapes-c-evaluation.md`, followed by one empty line:

```markdown
> **Superseded on 5 October 2026** by `cityscapes/evaluation/design.md` and `cityscapes/evaluation/plan.md`: both detectors, the image-quality baselines, and no pass/fail rule.
```

- [ ] **Step 4: Run the tests to see them pass, then the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/test_cityscapes_configs.py`
Expected: `2 passed`.

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: no failures, N + 25 collected.

- [ ] **Step 5: Commit**

```bash
git add cityscapes/evaluation/cityscapes-detectors.toml cityscapes/evaluation/cityscapes-iqa.toml tests/test_cityscapes_configs.py docs/superpowers/plans/2026-10-04-cityscapes-c-evaluation.md
git commit -m "feat: the Cityscapes-C configs for YOLO11m and the image-quality baselines

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0167vRw3NLCumE8FSGAnF1Tc" -- cityscapes/evaluation/cityscapes-detectors.toml cityscapes/evaluation/cityscapes-iqa.toml tests/test_cityscapes_configs.py docs/superpowers/plans/2026-10-04-cityscapes-c-evaluation.md
```

---

### Task 8: RT-DETRv2-R18 on Cityscapes-C (controller; GPU and CPU)

**Files:** none in git. Run folders: `runs/cityscapes-smoke/` and `runs/cityscapes/`.

**Interfaces:**
- Consumes: Tasks 1–5 and `cityscapes.toml`.
- Produces:
  - `runs/cityscapes/`, holding 500 files in each of `scores/detector`, `scores/activations`, `scores/method` and `scores/discopatch`;
  - the clean references;
  - `reports/cityscapes/`.

  Tasks 9 and 10 use it as their reference run.

All commands run from the worktree, with `PY=/home/yuchen/miniconda3/envs/UE/bin/python` and `R=/home/yuchen/YuchenZ/UE/philip_sa/runs`.

To sample GPU memory during a run, start `nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 5 > <log> & SAMPLER=$!` and end it with `kill $SAMPLER`. Don't use a `pgrep -f` loop: its pattern matches its own command line, so it never stops.

- [ ] **Step 1: Ask explore for the GPU**

Send, via SendMessage:

> GPU request from ue-finetune (Cityscapes-C, plan 2): may I run a smoke run of RT-DETR's stages on 20 val images, about 30–60 min?
> - Every stage runs at 5.5 GiB or less, one at a time, except DisCoPatch's training, whose official loop has no cap; its peak will be measured.
> - The CPU runs 9 corruption workers.
> - I'll send "done" when it ends.

Wait for the reply.

- [ ] **Step 2: The smoke run on 20 val images**

A smoke report needs at least about 10 images: ContrastiveConf's λ is cross-fitted in every bootstrap draw, and fails on 4 images.

```bash
SMOKE="--config cityscapes/evaluation/cityscapes.toml --run $R/cityscapes-smoke --limit 20"
nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 5 > $R/cityscapes-smoke-gpu.log & SAMPLER=$!
for stage in check knn-bank detector-pass hashemi-fit cdf-fit cdf-zstats activation-pass method-reference method-pass; do
  /usr/bin/time -f "$stage %e s" $PY -m degradation_monitor $stage $SMOKE || break
done
$PY -m degradation_monitor discopatch-train $SMOKE --epochs 1 && $PY -m degradation_monitor discopatch-pass $SMOKE
$PY -m degradation_monitor report $SMOKE
kill $SAMPLER; sort -n $R/cityscapes-smoke-gpu.log | tail -1
```

Run it in the background.

Expected:
- `check` prints `clean cityscapes val AP = 0.37… on 500 images`, at least 0.35. The review measured 0.3784.
- `$R/cityscapes-smoke/scores/{detector,activations,method,discopatch}/` each hold 20 files.
- `$R/cityscapes-smoke/reports/cityscapes/report.md` starts with `# Corruption detection on Cityscapes-C`, has the one section `## All 20 images`, and no decision line.

From the `/usr/bin/time` lines, put each pass's seconds per image into the ledger, with its × 500 estimate. 20 images make about two rounds of the 9 corruption workers, so the estimate is fair.

- [ ] **Step 3: Tell explore it is done,** with the peak GPU memory and the full run's estimate.

- [ ] **Step 4: Ask explore for the full run,** with that estimate and the same caps. Wait for the reply.

- [ ] **Step 5: The full run up to DisCoPatch's training**

Run in the background:

```bash
C="--config cityscapes/evaluation/cityscapes.toml"
ok=1
for stage in check knn-bank detector-pass hashemi-fit cdf-fit cdf-zstats activation-pass method-reference method-pass; do
  $PY -m degradation_monitor $stage $C >> $R/cityscapes-rtdetr.log 2>&1 || { ok=0; break; }
done
if [ $ok = 1 ] && [ ! -e $R/cityscapes/reference/discopatch/discriminator.pt ]; then
  $PY -m degradation_monitor discopatch-train $C > $R/cityscapes-discopatch-train.log 2>&1
fi
```

The stages are resumable: after an interruption, rerun the block. DisCoPatch trains only after every other stage has succeeded, and only once, so a rerun keeps its loss log.

Expected: for each of `detector activations method`, `ls $R/cityscapes/scores/<folder> | grep -c npz` gives 500. `$R/cityscapes/reference/discopatch/discriminator.pt` exists.

- [ ] **Step 6: Check that DisCoPatch's training loss has levelled off, before its scores are used**

The official loop prints each epoch's mean losses once in its progress bar, as `Loss: G - D Loss: D` (DisCoPatch.py:453). Read them:

```bash
tr '\r' '\n' < $R/cityscapes-discopatch-train.log | grep -oE "Loss: [0-9.]+ - D Loss: [0-9.]+" | uniq > $R/cityscapes-discopatch-losses.txt
wc -l < $R/cityscapes-discopatch-losses.txt; tail -20 $R/cityscapes-discopatch-losses.txt
```

Expected: 65 lines, one per epoch. Two consecutive epochs with identical 4-decimal losses would merge into one line.

Call it levelled off when, for both losses, the mean of the last 10 epochs is within 10% of the mean of the 10 before:

```bash
$PY - <<'PYEOF'
import re
rows = [tuple(map(float, re.findall(r"[0-9.]+", line))) for line in open("/home/yuchen/YuchenZ/UE/philip_sa/runs/cityscapes-discopatch-losses.txt")]
for i, name in enumerate(("generator", "discriminator")):
    late, before = (sum(r[i] for r in part) / len(part) for part in (rows[-10:], rows[-20:-10]))
    print(name, f"{before:.4f} -> {late:.4f}", "levelled off" if abs(late - before) <= 0.1 * abs(before) else "STILL MOVING")
PYEOF
```

If either loss prints `STILL MOVING`, stop and report both curves to the user before Step 7.

- [ ] **Step 7: DisCoPatch's scores and RT-DETR's report**

```bash
$PY -m degradation_monitor discopatch-pass $C >> $R/cityscapes-rtdetr.log 2>&1 && $PY -m degradation_monitor report $C >> $R/cityscapes-rtdetr.log 2>&1
for f in detector activations method discopatch; do echo $f $(ls $R/cityscapes/scores/$f | grep -c npz); done
```

Expected:
- 500 for each of the four folders;
- `$R/cityscapes/reports/cityscapes/summary.json` has `"image_sets": {"all": 500}`, `"benchmark": "cityscapes"`, and no `headline_decision`.

- [ ] **Step 8: Tell explore the run is done.**

---

### Task 9: YOLO11m on Cityscapes-C (controller; GPU and CPU)

**Files:** none in git. Run folder: `runs/cityscapes-detectors/`.

**Interfaces:**
- Consumes: Task 8's `runs/cityscapes/` (the digests and DisCoPatch's scores); `cityscapes-detectors.toml` (Task 7).
- Produces:
  - `runs/cityscapes-detectors/yolo11m/`, with 500 files per score folder and `reports/cityscapes/`;
  - `runs/cityscapes-detectors/summary.md`, the cross table of both detectors.

With `D="--config cityscapes/evaluation/cityscapes-detectors.toml"`:

- [ ] **Step 1: Ask explore for the GPU:** the check (about 2 min), the fit (3 passes over 2,975 train images) and the pass. All run at 8 GiB or less. Give the time estimate from Task 8's per-image pass time.

- [ ] **Step 2: The check and the fit**

Run:

```bash
$PY -m degradation_monitor.stages.detectors check $D
$PY -m degradation_monitor.stages.detectors fit $D
```

Expected: `[check] yolo11m: clean cityscapes val AP = 0.3… on 500 images`, at least 0.32. The fit ends with a peak-memory line.

- [ ] **Step 3: Time the pass on 4 images, then run the rest**

```bash
/usr/bin/time -f "pass 18 images %e s" $PY -m degradation_monitor.stages.detectors pass $D --first 18
$PY -m degradation_monitor.stages.detectors pass $D > $R/cityscapes-detectors-pass.log 2>&1
```

The second command runs in the background and resumes after the first 18 images. 18 images are two rounds of the 9 corruption workers, so the time × 500 / 18 is a fair estimate.

Expected: 500 files in each of `$R/cityscapes-detectors/yolo11m/scores/{detector,activations,method}`.

- [ ] **Step 4: The report and the cross table**

Run: `$PY -m degradation_monitor.stages.detectors report $D`

Expected:
- `$R/cityscapes-detectors/yolo11m/reports/cityscapes/summary.json` has `"image_sets": {"all": 500}`;
- `$R/cityscapes-detectors/summary.md` starts with `# The two-axis score on two Cityscapes-C detectors`.

- [ ] **Step 5: Tell explore the run is done.**

---

### Task 10: The image-quality baselines on Cityscapes-C (controller; GPU)

**Files:** none in git. Run folder: `runs/cityscapes-iqa/`.

**Interfaces:**
- Consumes: Tasks 8 and 9's run folders; `cityscapes-iqa.toml` (Task 7).
- Produces:
  - `runs/cityscapes-iqa/reports/{rtdetrv2_r18,yolo11m}/`: each detector's final report, with every row;
  - `runs/cityscapes-iqa/summary.md`.

With `Q="--config cityscapes/evaluation/cityscapes-iqa.toml"`:

- [ ] **Step 1: Ask explore for the GPU:** the IQA fit (2,975 train images at 2048 × 1024) and pass (500 × 96 versions), at 16 GiB or less, for about 2–3 h.

- [ ] **Step 2: The fit, then the pass timed on 4 images**

```bash
$PY -m degradation_monitor.stages.iqa fit $Q
/usr/bin/time -f "iqa pass 18 images %e s" $PY -m degradation_monitor.stages.iqa pass $Q --first 18
```

Expected: the fit and the pass each end with a peak-memory line under the cap. The 18-image pass gives the estimate for the full pass (× 500 / 18), which goes into the ledger.

If either runs out of memory under the 16 GiB cap, stop and report to the user. A batch-size change is a code change this plan doesn't make.

- [ ] **Step 3: The pass, in the background, then the reports**

```bash
$PY -m degradation_monitor.stages.iqa pass $Q > $R/cityscapes-iqa-pass.log 2>&1
$PY -m degradation_monitor.stages.iqa report $Q
```

Expected:
- 500 files in `$R/cityscapes-iqa/scores/iqa`;
- `$R/cityscapes-iqa/reports/{rtdetrv2_r18,yolo11m}/report.md`, each starting with `# Corruption detection on Cityscapes-C` and holding the five IQA rows;
- each `report.md` holds a DisCoPatch row: `grep -c "DisCoPatch" $R/cityscapes-iqa/reports/*/report.md` gives at least 1 per file. An IQA report run before Task 9's report, which links DisCoPatch's scores into YOLO's run, would silently leave it out;
- `$R/cityscapes-iqa/summary.md` with no untouched columns.

- [ ] **Step 4: Tell explore the run is done.**

---

### Task 11: The results

**Files:**
- Create: `cityscapes/evaluation/results.md`
- Create: `cityscapes/evaluation/results/`, copies of the final reports and tables

**Interfaces:**
- Consumes: Tasks 8–10's reports.

- [ ] **Step 1: Copy the tables into git's folder**

```bash
mkdir -p cityscapes/evaluation/results/rtdetrv2_r18 cityscapes/evaluation/results/yolo11m
cp $R/cityscapes-iqa/reports/rtdetrv2_r18/* cityscapes/evaluation/results/rtdetrv2_r18/
cp $R/cityscapes-iqa/reports/yolo11m/* cityscapes/evaluation/results/yolo11m/
cp $R/cityscapes-iqa/summary.md cityscapes/evaluation/results/iqa-summary.md
cp $R/cityscapes-iqa/summary.csv cityscapes/evaluation/results/iqa-summary.csv
cp $R/cityscapes-detectors/summary.md cityscapes/evaluation/results/detectors-summary.md
cp $R/cityscapes-detectors/summary.csv cityscapes/evaluation/results/detectors-summary.csv
```

- [ ] **Step 2: Write `cityscapes/evaluation/results.md`**

Take every number from the copied `summary.json`, `report.md` and summary tables. Contents:
- **The setting,** in a few lines: the detectors with their inputs, the references, the corruptions at 2048 × 1024, and the 500 val images. Point to `design.md`.
- **For each detector, the headline table:** every row, with AUROC, AUPR and FPR95 on the common and the extra families, and their intervals. Mark each column's best row in bold.
- **The two-axis score minus each baseline,** with intervals: the `All 500 images` section of each `report.md`.
- **AUROC by severity:** the two-axis score, the level score, the activation CDFs and the strongest IQA row. That row is never the published NIQE, which is only a sensitivity row.
- **The families at severities 1, 3 and 5,** for the two-axis score.
- **The clean mAP and the mAP by severity,** as context.
- **DisCoPatch's training:** its loss check from Task 8, Step 6.
- **The caveats, from the spec:**
  - after the 3.2× downscale, noise, blur, pixelation and JPEG are milder than on COCO-C at the same severity;
  - the IQA models read the full 2048 × 1024 image, where on COCO they read about the detector's scale;
  - DisCoPatch reads 256 × 256, and trained for about 2,900 steps;
  - val has only 500 images.
- **How to rerun:** the stage commands of Tasks 8–10.

- [ ] **Step 3: Commit**

```bash
git add cityscapes/evaluation/results.md cityscapes/evaluation/results
git commit -m "results: Cityscapes-C, the two-axis score and every baseline on both Cityscapes detectors

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0167vRw3NLCumE8FSGAnF1Tc" -- cityscapes/evaluation/results.md cityscapes/evaluation/results
```
