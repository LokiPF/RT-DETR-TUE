> **Superseded on 5 October 2026** by `cityscapes/evaluation/design.md` and `cityscapes/evaluation/plan.md`: both detectors, the image-quality baselines, and no pass/fail rule.

# Cityscapes-C Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run our two-axis score and the six baselines on Cityscapes-C, with the RT-DETRv2-R18 fine-tuned on Cityscapes, and report them like the COCO results.

**Architecture:** The `degradation_monitor` package was built so that a second benchmark plugs in without touching the baselines or the method (clean-branch spec, row 7). This plan adds:
- a `Cityscapes` dataset next to `Coco`, which differs only in its class attributes;
- a `benchmark` setting that picks it;
- a detector builder that reads its class count from the checkpoint;
- a check stage and a report that follow the benchmark.

The stages then run unchanged on `configs/cityscapes.toml`.

Out of scope, each for a later plan:
- Foggy Cityscapes and ACDC, which still need downloads;
- the image-quality baselines (NIQE, ARNIQA, CLIP-IQA), an open question in `docs/paper-storyline.md`.

**Tech Stack:** Python 3.11, PyTorch 2.11, pycocotools, imagecorruptions, pytest; the `UE` conda environment at `/home/yuchen/miniconda3/envs/UE`.

**Spec:**
- `docs/driving-benchmark-baselines-and-metrics.md` (27 September): the setting, the baselines, the metrics and the fairness rules.
- `docs/superpowers/specs/2026-10-02-clean-branch-design.md`, row 7: a dataset-shaped layout, so that Cityscapes can be added later without touching the baselines or the method.

## Global Constraints

- **Detector.** RT-DETRv2-R18, fine-tuned from `rtdetrv2_r18vd_120e_coco_rerun_48.1.pth` on Cityscapes train with `RT-DETRv2-UE/configs/rtdetrv2/rtdetrv2_r18vd_cityscapes.yml`.
  - The 8 classes, in this order: person, rider, car, truck, bus, train, motorcycle, bicycle.
  - It is frozen at `/home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_cityscapes_72e.pth`. The package loads its EMA weights.
- **Clean reference data:** Cityscapes train images only (2,975), with no labels. The same images serve every method:
  - our bank and z-statistics images: the seed-44 splits of 2,000 and 500;
  - the kNN bank, Hashemi's per-neuron statistics, the activation CDFs' fit and z-statistics, and DisCoPatch's training: all 2,975.
- **Evaluation images:** all 500 Cityscapes val images, in the seed-44 shuffle order, with 5 folds.
- **Conditions:** the 96 conditions: clean, plus 15 common and 4 extra `imagecorruptions` families at severities 1–5.
  - Corruptions are applied by the package's seeded `corruptions.variants` at the full Cityscapes resolution of 2048 × 1024, as in the standard Cityscapes-C.
  - The detector sees 640 × 640.
- **Metrics:**
  - AUROC (main), AUPR and FPR95;
  - paired whole-image bootstrap, 1,000 draws, seed 44;
  - ContrastiveConf's λ cross-fitted on the 5 folds, from per-image AP against `cityscapes_val_8cls.json`.
- **Pre-registered rule, fixed on 4 October 2026 before any Cityscapes score is computed:** "confirmed" when, on all 500 images, the two-axis score beats the activation CDFs on the common and the extra families, every 95% interval excluding 0.
  - COCO's untouched and held-out sets and their two decisions do not apply to Cityscapes.
- **The method's fixed choices stay as they are:** k = 50, the stage-4 key, stages 1–3, the top 1%, 2,000 + 500 reference images.
- **COCO behaviour stays exactly as it is:**
  - `configs/coco.toml` is unchanged;
  - the COCO protocol dict is unchanged, so `runs/coco/` keeps working;
  - `tests/evaluation/test_report_golden.py` and `tests/test_equivalence.py` stay green.
- **Tests run on the CPU only:** `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`. The suite at the start: 159 passed, 1 skipped.
- **The GPU is shared** with the session "explore". Before Tasks 8 and 9:
  - ask explore via SendMessage, and wait for its answer;
  - every GPU stage runs under the config's `gpu_memory_gib = 5.5` cap, one stage at a time;
  - send explore "done" when finished.
- **Git:**
  - work on a new branch `cityscapes` cut from `fingerprint_bank`, and commit at the end of each task with both trailers in one `-m`: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU`;
  - never push;
  - `git add` only the files a task names. The working tree also holds uncommitted paper notes (`docs/paper-storyline.md`, `docs/results/paper-evidence/`, `scripts/paper/`), which must never be swept into a commit.

## Review Focus

1. **City subfolders whose image names repeat.** Result files and the ground truth are keyed by name, so a repeated name must be refused, not overwritten. Test: Task 3, `test_repeated_image_names_are_refused`.
2. **A checkpoint whose class count differs from the annotation file's**, for example the COCO checkpoint with the Cityscapes config. The check stage must refuse it with a message that names both counts. Test: Task 5, `test_the_check_refuses_a_detector_whose_classes_differ_from_the_annotations`.
3. **A run folder made with COCO settings, opened with Cityscapes settings.** It must be refused, with "dataset" named as the changed protocol field. Test: Task 4, `test_a_coco_run_folder_refuses_cityscapes_settings`.
4. **The Cityscapes report must not mention COCO's screen, held-out or untouched sets or their decisions.** It applies the pre-registered rule to all images. Tests: Task 6, `test_a_new_benchmark_reports_one_image_set_under_its_own_rule` and `test_the_report_stage_writes_a_cityscapes_report_with_one_image_set`.
5. **The real folders must give 2,975 train and 500 val images, and the reference splits must fit.** The city-folder listing must miss none. Test: Task 3, `test_the_real_cityscapes_folders_give_the_expected_images` (skipped when the dataset is absent).

---

### Task 1: Freeze the fine-tuned detector

The training ran from 16:03 to 17:50 on 4 October (1 h 48 min) and wrote `RT-DETRv2-UE/output/rtdetrv2_r18vd_cityscapes/`.
Its last epoch is also its best one: AP 0.382 and AP@50 0.590 on Cityscapes val, from the EMA weights.

**Files:**
- Create: `/home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_cityscapes_72e.pth` (a copy of `last.pth`)
- Create: `docs/cityscapes-finetune.md`

**Interfaces:**
- Produces: the checkpoint path above, which `configs/cityscapes.toml` (Task 4) and the guarded tests (Tasks 2 and 3) use; and the clean val AP, which Task 3 copies into `Cityscapes.min_clean_ap`.

- [ ] **Step 1: Confirm that training finished all 72 epochs**

Run: `grep -E "Epoch: \[71\] Total|Training time" /home/yuchen/YuchenZ/UE/RT-DETRv2-UE/output/rtdetrv2_r18vd_cityscapes.log`
Expected: both lines present.

- [ ] **Step 2: Copy the final checkpoint and record its hash**

```bash
cp /home/yuchen/YuchenZ/UE/RT-DETRv2-UE/output/rtdetrv2_r18vd_cityscapes/last.pth \
   /home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_cityscapes_72e.pth
sha256sum /home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_cityscapes_72e.pth
```

- [ ] **Step 3: Evaluate the final EMA weights on Cityscapes val**

The weights come from the last epoch, not the best epoch, so val stays unused for model selection. Run (GPU, about 10 s):

```bash
cd /home/yuchen/YuchenZ/UE/RT-DETRv2-UE
CUDA_VISIBLE_DEVICES=0 /home/yuchen/miniconda3/envs/UE/bin/python tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_cityscapes.yml -r pretrained_weights/rtdetrv2_r18vd_cityscapes_72e.pth \
  --test-only --output-dir output/rtdetrv2_r18vd_cityscapes_eval
```
Expected: the 12 COCO metrics are printed, and `output/rtdetrv2_r18vd_cityscapes_eval/eval.pth` is written.

- [ ] **Step 4: Per-class AP from `eval.pth`**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python - <<'EOF'
import numpy as np, torch
e = torch.load("/home/yuchen/YuchenZ/UE/RT-DETRv2-UE/output/rtdetrv2_r18vd_cityscapes_eval/eval.pth", weights_only=False)
p = np.asarray(e["precision"])  # (IoU 10, recall 101, classes 8, area 4, max dets 3)
names = ("person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")
for k, name in enumerate(names):
    ap = p[:, :, k, 0, 2]; ap50 = p[0, :, k, 0, 2]
    print(f"{name:<11} AP {ap[ap > -1].mean():.3f}  AP50 {ap50[ap50 > -1].mean():.3f}")
EOF
```
Expected: one line per class.

- [ ] **Step 5: Write `docs/cityscapes-finetune.md`**

Record:
- the recipe: the config file, 72 epochs, total batch 16 on one GPU, AdamW at 1e-4 with the backbone at 1e-5, EMA, AMP, multi-scale 480–800, 640 × 640 evaluation, class heads re-initialised by `-t`;
- the converter, `RT-DETRv2-UE/dataset_tools/cityscapes_converter.py` (Detectron's rules), and its box counts: train 50,347, val 9,792;
- the source: RT-DATR, arXiv 2504.09196;
- the training time;
- the clean val AP and AP@50, and the per-class table from Step 4;
- the checkpoint's sha256.

Add one line for it to `docs/README.md` under "Baselines and setting".

- [ ] **Step 6: Commit**

```bash
git checkout -b cityscapes
git add docs/cityscapes-finetune.md
git commit -m "docs: the Cityscapes fine-tune of RT-DETRv2-R18 and its clean val AP

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

`docs/README.md` holds uncommitted lines from 2 October (the storyline and paper-evidence entries). Commit it here only if the user has agreed to commit that work; otherwise leave it uncommitted and say so in the task report.

---

### Task 2: The detector's class count comes from its checkpoint

**Files:**
- Modify: `degradation_monitor/detector/model.py` (`build_fixed_detector`, `load_frozen_detector`; new `checkpoint_classes`, `COCO_CLASSES`, `SCORE_HEAD`)
- Test: `tests/detector/test_model.py`

**Interfaces:**
- Produces: `build_fixed_detector(num_classes: int = 80) -> RTDETR`; `checkpoint_classes(state: Mapping[str, Tensor]) -> int`; `load_frozen_detector(checkpoint_path, device)`, which builds as many classes as the checkpoint's decoder score head has rows.

- [ ] **Step 1: Write the failing tests**

Append to `tests/detector/test_model.py` (and add `from pathlib import Path` to its imports):

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

In the existing `test_load_frozen_detector_loads_cpu_state_and_freezes`, replace the monkeypatched builder `lambda: nn.Linear(3, 2)` with `lambda num_classes=80: nn.Linear(3, 2)`.

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/detector/test_model.py`
Expected: FAIL with `AttributeError: module 'degradation_monitor.detector.model' has no attribute 'checkpoint_classes'` (and `build_fixed_detector() got an unexpected keyword argument 'num_classes'`).

- [ ] **Step 3: Implement**

In `degradation_monitor/detector/model.py`, change the module docstring to `"""Build the fixed RT-DETRv2-R18, load a frozen checkpoint (COCO or Cityscapes), and prepare images for it."""`.

Add, after `IMAGE_SIZE`:

```python
COCO_CLASSES = 80
SCORE_HEAD = "decoder.dec_score_head.0.weight"  # (classes, 256) in every RT-DETRv2 checkpoint
```

Change the builder's signature and its decoder argument, leaving everything else in it as it is:

```python
def build_fixed_detector(num_classes: int = COCO_CLASSES) -> RTDETR:
    ...
    decoder = RTDETRTransformerv2(num_classes=num_classes, hidden_dim=256, ...)  # every other argument unchanged
```

Add, after `checkpoint_state`:

```python
def checkpoint_classes(state: Mapping[str, Tensor]) -> int:
    """How many classes a checkpoint detects: the rows of its decoder's first score head (COCO 80, Cityscapes 8)."""
    head = state.get(SCORE_HEAD)
    return COCO_CLASSES if head is None else int(head.shape[0])
```

Make `load_frozen_detector` build from the state:

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

- [ ] **Step 4: Run the tests to see them pass, then the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/detector/test_model.py`
Expected: PASS.

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: `162 passed, 1 skipped`.

- [ ] **Step 5: Commit**

```bash
git add degradation_monitor/detector/model.py tests/detector/test_model.py
git commit -m "feat: the detector's class count comes from its checkpoint

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

---

### Task 3: A Cityscapes dataset

**Files:**
- Create: `degradation_monitor/datasets/cityscapes.py`
- Modify: `degradation_monitor/datasets/coco.py`:
  - `list_images` and `evaluation_images` take `recursive`;
  - `CocoGroundTruth` finds images by base name;
  - `Coco` gets its class attributes and uses them.
- Modify: `degradation_monitor/datasets/__init__.py`, the docstring only.
- Test: `tests/datasets/test_cityscapes.py`

**Interfaces:**
- Consumes: the val AP measured in Task 1, Step 3.
- Produces:
  - `list_images(root, recursive: bool = False) -> list[Path]`, which raises `ValueError("image names repeat ...")` on a repeated stem;
  - `evaluation_images(val_root, *, seed, recursive=False)`;
  - `Coco` with class attributes `name = "coco"`, `classes = 80`, `min_clean_ap = 0.45`, `recursive = False`, `screened = True`;
  - `Cityscapes(Coco)` with `name = "cityscapes"`, `classes = 8`, `min_clean_ap` set from Task 1, `recursive = True`, `screened = False`;
  - `CocoGroundTruth.image_id(name)`, which takes the base file name.

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
    assert 0 < Cityscapes.min_clean_ap < 1 and Coco.min_clean_ap == 0.45


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
Expected: FAIL with `ModuleNotFoundError: No module named 'degradation_monitor.datasets.cityscapes'`.

- [ ] **Step 3: Implement**

In `degradation_monitor/datasets/coco.py`, add `ClassVar` to the `typing` import. Then replace `list_images` and `evaluation_images`:

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

In `CocoGroundTruth.__init__`, replace the `self._ids = {...}` line with:

```python
        self._ids = {}
        for image_id, info in self.coco.imgs.items():
            name = Path(info["file_name"]).name  # Cityscapes stores city/name; images are looked up by name
            if name in self._ids:
                raise ValueError(f"{name} appears twice in {annotation_file}")
            self._ids[name] = image_id
```

Replace the `Coco` dataclass with:

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

Create `degradation_monitor/datasets/cityscapes.py`. The floor is the fine-tune's clean val AP (0.382 at its last epoch, from RT-DETR's own evaluator) minus 0.03, rounded down to two decimals. If Task 1, Step 3 prints an AP that differs by more than 0.005, recompute the floor from it:

```python
"""Cityscapes (fine annotations, 8 instance classes) in the COCO-style form of datasets.coco.

The images sit in one folder per city, leftImg8bit/<split>/<city>/<name>_leftImg8bit.png. Their names are unique, so
result files and the ground truth are keyed by name, as for COCO. The annotation file is the val output of
RT-DETRv2-UE/dataset_tools/cityscapes_converter.py: Detectron's conversion rules, category ids 0-7 in CLASSES order.
"""
from __future__ import annotations

from typing import ClassVar

from .coco import Coco

CLASSES = ("person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")


class Cityscapes(Coco):
    """Cityscapes train and val: city subfolders, 8 classes, and no screening history (one image set in the report)."""
    name: ClassVar[str] = "cityscapes"
    classes: ClassVar[int] = len(CLASSES)
    min_clean_ap: ClassVar[float] = 0.35  # the fine-tuned detector's clean val AP is about 0.38
    recursive: ClassVar[bool] = True
    screened: ClassVar[bool] = False
```

Change the docstring of `degradation_monitor/datasets/__init__.py` to `"""Datasets: clean reference images, the evaluation order and folds, and ground truth (COCO, Cityscapes)."""`.

- [ ] **Step 4: Run the tests to see them pass, then the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/datasets`
Expected: PASS.

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: `169 passed, 1 skipped`.

- [ ] **Step 5: Commit**

```bash
git add degradation_monitor/datasets/cityscapes.py degradation_monitor/datasets/coco.py \
        degradation_monitor/datasets/__init__.py tests/datasets/test_cityscapes.py
git commit -m "feat: a Cityscapes dataset in the COCO-style form

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

---

### Task 4: The settings choose the benchmark

**Files:**
- Modify: `degradation_monitor/settings.py`
- Create: `configs/cityscapes.toml`
- Test: `tests/test_settings.py`

**Interfaces:**
- Consumes: `Coco`, `Cityscapes` (Task 3).
- Produces:
  - the `Settings.benchmark` field, `"coco"` by default, `"cityscapes"` for Cityscapes;
  - `Settings.dataset`, which returns that benchmark's dataset;
  - `Settings.protocol()["dataset"] == settings.benchmark`;
  - `BENCHMARKS = {"coco": Coco, "cityscapes": Cityscapes}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_settings.py`, and add `from degradation_monitor.datasets.cityscapes import Cityscapes`, `from degradation_monitor.datasets.coco import Coco` and `from degradation_monitor.runs import Manifest` to its imports:

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
    settings = load_settings(Path(__file__).resolve().parents[1] / "configs" / "cityscapes.toml")
    assert settings.benchmark == "cityscapes" and settings.run.name == "cityscapes"
    assert settings.checkpoint.name == "rtdetrv2_r18vd_cityscapes_72e.pth"
    assert settings.annotations.name == "cityscapes_val_8cls.json" and settings.gpu_memory_gib == 5.5
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/test_settings.py`
Expected: FAIL with `ValueError: unknown settings in .../coco.toml: benchmark`.

- [ ] **Step 3: Implement**

In `degradation_monitor/settings.py`, update the module docstring's first line to `"""Run settings: the benchmark, the machine's paths and run options, read from a TOML file such as configs/coco.toml.`, and make these changes:

```python
from .datasets.cityscapes import Cityscapes
from .datasets.coco import FOLDS, Coco
...
BENCHMARKS = {"coco": Coco, "cityscapes": Cityscapes}


@dataclass(frozen=True)
class Settings:
    run: Path
    checkpoint: Path
    train_images: Path
    val_images: Path
    annotations: Path
    discopatch_root: Path
    benchmark: str = "coco"
    device: str = "cuda:0"
    batch_size: int = 32
    workers: int = 9
    gpu_memory_gib: Optional[float] = None
    limit: Optional[int] = None
    epochs: int = 65
    seed: int = 44

    def __post_init__(self):
        if self.benchmark not in BENCHMARKS:
            raise ValueError(f"unknown benchmark {self.benchmark!r}; choose from {', '.join(sorted(BENCHMARKS))}")

    @property
    def layout(self) -> RunLayout:
        return RunLayout(self.run)

    @property
    def dataset(self) -> Coco:
        return BENCHMARKS[self.benchmark](self.train_images, self.val_images, self.annotations, seed=self.seed,
                                          limit=self.limit)
```

In `protocol()`, replace `"dataset": "coco"` with `"dataset": self.benchmark`.

Create `configs/cityscapes.toml`:

```toml
# This machine's paths and run options for Cityscapes-C. Stages: python -m degradation_monitor <stage> --config configs/cityscapes.toml
# The detector is the Cityscapes fine-tune of RT-DETRv2-R18 (docs/cityscapes-finetune.md); every reference is Cityscapes train.
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
epochs = 65
seed = 44
```

- [ ] **Step 4: Run the tests to see them pass, then the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/test_settings.py`
Expected: PASS.

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: `172 passed, 1 skipped`.

- [ ] **Step 5: Commit**

```bash
git add degradation_monitor/settings.py configs/cityscapes.toml tests/test_settings.py
git commit -m "feat: the settings choose the benchmark; a Cityscapes config

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

---

### Task 5: The check stage follows the benchmark

**Files:**
- Modify: `degradation_monitor/stages/baselines.py`:
  - remove `MIN_CLEAN_AP`;
  - `check` lists the benchmark's val images, compares the detector's classes with the ground truth, and uses the benchmark's floor.
- Modify: `README.md`, the `check` row of the stage table.
- Test: `tests/stages/test_baselines.py`

**Interfaces:**
- Consumes: `Settings.dataset` with `name`, `recursive`, `min_clean_ap` and `ground_truth()` (Tasks 3 and 4).
- Produces: `check(settings, manifest) -> float`. It keeps writing the manifest key `coco_val_ap` (the COCO AP@[.5:.95] metric on the val images), and raises `ValueError("the detector predicts N classes but <file> has M ...")` on a mismatch.

- [ ] **Step 1: Write the failing test**

Append to `tests/stages/test_baselines.py`, and add `from dataclasses import replace` to its imports:

```python
def test_the_check_refuses_a_detector_whose_classes_differ_from_the_annotations(tmp_path, fakes):
    settings = replace(_settings(tmp_path), benchmark="cityscapes")
    names = sorted(p.name for p in settings.val_images.iterdir())
    settings.annotations.write_text(json.dumps({
        "images": [{"id": i, "file_name": f"val/x/{n}", "width": 64, "height": 48} for i, n in enumerate(names)],
        "annotations": [], "categories": [{"id": c, "name": str(c)} for c in range(8)]}))
    with pytest.raises(ValueError, match="predicts 80 classes but ann.json has 8"):
        run_stage("check", settings)
```

- [ ] **Step 2: Run it to see it fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/stages/test_baselines.py -k classes_differ`
Expected: FAIL with `IndexError: tuple index out of range`. The old check maps the fake detector's 80 labels onto the 8 categories, instead of refusing the class-count mismatch.

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

In `README.md`, change the `check` row of the stage table to: `` | `check` | that every input exists, that the detector predicts the annotation file's classes, and the clean val AP (COCO about 0.48) | – | ``.

- [ ] **Step 4: Run the test to see it pass, then the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/stages/test_baselines.py`
Expected: PASS.

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: `173 passed, 1 skipped`.

- [ ] **Step 5: Commit**

```bash
git add degradation_monitor/stages/baselines.py README.md tests/stages/test_baselines.py
git commit -m "feat: the check stage follows the benchmark and refuses a class-count mismatch

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

---

### Task 6: The report follows the benchmark

**Files:**
- Modify: `degradation_monitor/evaluation/report.py`:
  - `build_tables` takes `sets` and `benchmark`;
  - new `transfer_decision`;
  - `markdown` uses a title per benchmark, and a headline without COCO's sets when unscreened.
- Modify: `degradation_monitor/stages/report.py`: `write_report` passes the dataset's sets and name, and writes to `reports/<benchmark>/`.
- Test: `tests/evaluation/test_report.py`, `tests/stages/test_report.py`

**Interfaces:**
- Consumes: `Settings.dataset` with `name`, `screened` and `evaluation_images()` (Tasks 3 and 4).
- Produces:
  - `build_tables(scores, detector, ap, folds, condition_map, *, seed, samples=BOOTSTRAP_SAMPLES, workers=1, timing=None, sets=None, benchmark="coco")`;
  - `sets=None` keeps COCO's four sets and both COCO decisions;
  - a dict of image sets gives those sets, `transfer_decision`, and `level_decision = "not applicable: ..."`;
  - the summary gains `benchmark` and `screened`;
  - `transfer_decision(intervals) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/evaluation/test_report.py`. It already imports `numpy as np`, `report` and `assign_folds`; add any that are missing:

```python
def _transfer_inputs(images=30):
    rng = np.random.default_rng(1)
    severity = np.array([s for _, s in corruptions.CONDITIONS], float)
    scores = {row: rng.normal(0, 1, (images, 1)) + strength * severity[None] + rng.normal(0, 1, (images, 96))
              for row, strength in (("two_axis", 0.5), ("level", 0.3), ("cdf", 0.1))}
    conf_pos = rng.uniform(0.2, 0.9, (images, 96))
    conf_neg = rng.uniform(0.0, 0.1, (images, 96))
    knn = np.sort(rng.uniform(0.5, 1.5, (images, 96, 200)), axis=2).astype(np.float32)
    ap = conf_pos[:, 0] - 3.0 * conf_neg[:, 0]
    return scores, {"conf_pos": conf_pos, "conf_neg": conf_neg, "knn": knn}, ap, 0.5 - 0.05 * severity


def test_the_transfer_decision_needs_both_groups_ahead_of_the_cdfs():
    ahead, behind = {"low": 0.01}, {"low": -0.01}
    both = {"all": {"two_axis - cdf:auroc_common": ahead, "two_axis - cdf:auroc_extra": ahead}}
    one = {"all": {"two_axis - cdf:auroc_common": ahead, "two_axis - cdf:auroc_extra": behind}}
    assert report.transfer_decision(both) == "confirmed"
    assert report.transfer_decision(one) == "not ahead of the activation CDFs"
    assert report.transfer_decision({}).startswith("unavailable")


def test_a_new_benchmark_reports_one_image_set_under_its_own_rule():
    scores, detector, ap, condition_map = _transfer_inputs()
    tables, summary = report.build_tables(scores, detector, ap, assign_folds(30), condition_map, seed=44, samples=10,
                                          workers=1, sets={"all": np.arange(30)}, benchmark="cityscapes")
    assert list(summary["image_sets"]) == ["all"] and summary["screened"] is False
    assert summary["headline_decision"] in {"confirmed", "not ahead of the activation CDFs"}
    assert summary["level_decision"].startswith("not applicable")
    text = report.markdown(summary, tables)
    assert text.startswith("# Corruption detection on Cityscapes-C")
    assert "untouched" not in text.lower() and "held-out" not in text.lower() and "screen" not in text.lower()
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
        "annotations": [{"id": i, "image_id": i, "category_id": 2, "bbox": [8, 6, 24, 18], "area": 432, "iscrowd": 0}
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
    assert list(summary["image_sets"]) == ["all"] and summary["benchmark"] == "cityscapes"
    assert not settings.layout.report("coco").exists()
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/evaluation/test_report.py tests/stages/test_report.py`
Expected: FAIL with `AttributeError: module 'degradation_monitor.evaluation.report' has no attribute 'transfer_decision'`, then `TypeError: build_tables() got an unexpected keyword argument 'sets'`.

- [ ] **Step 3: Implement**

In `degradation_monitor/evaluation/report.py`, add after `headline_decision`:

```python
TITLES = {"coco": "COCO", "cityscapes": "Cityscapes-C"}


def transfer_decision(intervals: dict) -> str:
    """The rule fixed for a new benchmark before any of its scores were computed (Cityscapes-C, 4 October 2026).

    Confirmed when, on all images, the two-axis score beats the activation CDFs on the common and the extra
    families, every interval excluding 0.
    """
    needed = [f"two_axis - cdf:auroc_{group}" for group in GROUPS]
    if not all(key in intervals.get("all", {}) for key in needed):
        return "unavailable: the two-axis or the activation-CDF scores are missing"
    if all(intervals["all"][key]["low"] > 0 for key in needed):
        return "confirmed"
    return "not ahead of the activation CDFs"
```

Change `build_tables`:

```python
def build_tables(scores: dict, detector: dict, ap, folds, condition_map, *, seed: int,
                 samples: int = BOOTSTRAP_SAMPLES, workers: int = 1, timing=None, sets=None,
                 benchmark: str = "coco") -> tuple[dict, dict]:
    """Every table of the report, and its summary.

    `scores` holds every present row except ContrastiveConf as (images, 96) arrays in evaluation order.
    ContrastiveConf is built for each image set, and in every draw, from the detector's Conf+ and Conf- and the
    clean images' AP. `sets` is None for COCO, whose screen, held-out and untouched sets and pre-registered decisions
    come from its history; another benchmark passes its own sets and is judged by transfer_decision.
    """
```

In its body:
- replace `sets = image_sets(len(folds))` with:

  ```python
      screened = sets is None
      sets = image_sets(len(folds)) if screened else sets
  ```

- in the `summary` dict, replace the two decision entries (`"headline_decision": headline_decision(intervals),` and `"level_decision": level_decision(intervals.get("held_out", {})),`) with:

  ```python
          "benchmark": benchmark, "screened": screened,
          "headline_decision": headline_decision(intervals) if screened else transfer_decision(intervals),
          "level_decision": (level_decision(intervals.get("held_out", {})) if screened
                             else "not applicable: the level score's rule was pre-registered for COCO only"),
  ```

In `markdown`, replace the first five list entries (the title, both decision lines and the blank lines between them) with:

```python
    benchmark = summary.get("benchmark", "coco")
    lines = [f"# Corruption detection on {TITLES.get(benchmark, benchmark)}: our method and six baselines", ""]
    if summary.get("screened", True):
        lines += [f"**Headline (two-axis score, all images and the untouched ones):** {summary['headline_decision']}.",
                  "", f"**Pre-registered level score (held-out images):** {summary['level_decision']}.", ""]
    else:
        lines += ["**Headline (two-axis score on all images, by the rule fixed before any score was computed):** "
                  f"{summary['headline_decision']}.", ""]
    lines += ["Every score is oriented so that higher means more likely corrupted. ↑ higher is better, ↓ lower is "
              "better; an AUROC of 0.5 is chance. Brackets are 95% paired bootstrap intervals over images "
              f"({summary['bootstrap_samples']} draws, seed {summary['seed']}).", ""]
```

In the `scalars` filter at the end of `markdown`, also exclude `screened`, so that the "Fixed choices" list stays free of the word "screen" for a new benchmark:

```python
    scalars = {k: v for k, v in summary.items()
               if not isinstance(v, dict) and k not in ("headline_decision", "level_decision", "screened")}
```

For an unscreened benchmark, also leave `screen_images` and `untouched_start` out of the summary. After the `summary = {...}` literal, add:

```python
    if not screened:
        del summary["screen_images"], summary["untouched_start"]
```

In `degradation_monitor/stages/report.py`:
- change the module docstring to `"""The report stage: read every stored score and write the report to reports/<benchmark>/."""`;
- in `write_report`, use the dataset object:

```python
    dataset = settings.dataset
    names = [p.name for p in dataset.evaluation_images()]
    ...
    report_tables, summary = tables.build_tables(scores, detector, ap, dataset.folds(), condition_map,
                                                 seed=settings.seed, samples=tables.BOOTSTRAP_SAMPLES,
                                                 workers=settings.workers, timing=timing,
                                                 sets=None if dataset.screened else {"all": np.arange(len(names))},
                                                 benchmark=dataset.name)
    ...
    tables.write_outputs(layout.report(dataset.name), report_tables, summary)
```

(`gt = dataset.ground_truth()` replaces `settings.dataset.ground_truth()` as well.)

- [ ] **Step 4: Run the tests to see them pass, then the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/evaluation tests/stages/test_report.py`
Expected: PASS, including `tests/evaluation/test_report_golden.py` unchanged.

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: `176 passed, 1 skipped`.

- [ ] **Step 5: Commit**

```bash
git add degradation_monitor/evaluation/report.py degradation_monitor/stages/report.py \
        tests/evaluation/test_report.py tests/stages/test_report.py
git commit -m "feat: the report follows the benchmark, with the transfer rule for a new benchmark

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

---

### Task 7: Document the Cityscapes runs

**Files:**
- Modify: `README.md`, in "Layout", "Running" and "Results"

- [ ] **Step 1: Edit `README.md`**

- In "Layout", under `configs/coco.toml`, add the line `configs/cityscapes.toml    the same for Cityscapes-C, with the Cityscapes fine-tune of the detector`.
- At the end of "Running", add:

```markdown
### Cityscapes-C

Pass `--config configs/cityscapes.toml` to every stage. The detector is the Cityscapes fine-tune
(`docs/cityscapes-finetune.md`), and every clean reference comes from the 2,975 Cityscapes train images. Corruptions
are applied at the full 2048 × 1024 resolution, so a pass takes longer than on COCO; glass blur dominates. The report
(`runs/cityscapes/reports/cityscapes/`) has one image set, all 500 val images, and judges the two-axis score by the
rule fixed before any Cityscapes score was computed: ahead of the activation CDFs on the common and the extra
families, every interval excluding 0.
```

- In "Results", change ``- `reports/coco/`: the report.`` to ``- `reports/coco/` or `reports/cityscapes/`: the report.``.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: how to run the Cityscapes-C benchmark

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

---

### Task 8: A smoke run on 4 val images

**Files:** none in the repository. The run folder is `runs/cityscapes-smoke/` (git-ignored).

- [ ] **Step 1: Ask explore for the GPU**

Use SendMessage to explore: about 20–40 minutes of short GPU stages, under the 5.5 GiB cap each, one at a time. Wait for its reply.

- [ ] **Step 2: Run every stage with `--limit 4` and one DisCoPatch epoch**

```bash
cd /home/yuchen/YuchenZ/UE/philip_sa
PY=/home/yuchen/miniconda3/envs/UE/bin/python
SMOKE="--config configs/cityscapes.toml --run runs/cityscapes-smoke --limit 4"
for stage in check knn-bank detector-pass hashemi-fit cdf-fit cdf-zstats activation-pass method-reference method-pass; do
  /usr/bin/time -f "$stage %e s" $PY -m degradation_monitor $stage $SMOKE || break
done
$PY -m degradation_monitor discopatch-train $SMOKE --epochs 1 && $PY -m degradation_monitor discopatch-pass $SMOKE
$PY -m degradation_monitor report $SMOKE
```

Expected:
- `check` prints the clean cityscapes val AP on 500 images, at least `Cityscapes.min_clean_ap`;
- each of `runs/cityscapes-smoke/scores/{detector,activations,method,discopatch}/` holds 4 files;
- `runs/cityscapes-smoke/reports/cityscapes/report.md` begins with `# Corruption detection on Cityscapes-C` and has one image-set section.

- [ ] **Step 3: Record the timing**

From the `/usr/bin/time` lines, note the seconds per image of each pass. Multiply by 500 to plan Task 9, and write the estimate into the task report.

- [ ] **Step 4: Tell explore the smoke run is done**

---

### Task 9: The full Cityscapes-C run and its results

**Files:**
- Create: `docs/results/cityscapes/` (a copy of `runs/cityscapes/reports/cityscapes/`)
- Create: `docs/cityscapes-c-results.md`
- Modify: `docs/README.md`, `docs/paper-storyline.md` (step 9, claim C10, the figures and tables list)

- [ ] **Step 1: Ask explore for the GPU, with Task 8's estimate of the total time**

- [ ] **Step 2: Run the stages in order on `runs/cityscapes/`**

The run is resumable; rerun a stage after an interruption.

```bash
cd /home/yuchen/YuchenZ/UE/philip_sa
PY=/home/yuchen/miniconda3/envs/UE/bin/python
for stage in check knn-bank detector-pass hashemi-fit cdf-fit cdf-zstats activation-pass method-reference method-pass \
             discopatch-train discopatch-pass report; do
  $PY -m degradation_monitor $stage --config configs/cityscapes.toml || break
done
```

Expected:
- each pass folder holds 500 files;
- `runs/cityscapes/reports/cityscapes/summary.json` has `"image_sets": {"all": 500}`;
- `summary.json` has a `headline_decision` of `confirmed`, `not ahead of the activation CDFs` or `unavailable: ...`.

- [ ] **Step 3: Tell explore the run is done**

- [ ] **Step 4: Keep the tables, and write the results**

```bash
mkdir -p docs/results/cityscapes && cp runs/cityscapes/reports/cityscapes/* docs/results/cityscapes/
```

Write `docs/cityscapes-c-results.md` with:
- the setting (detector, references, the full-resolution corruptions);
- the pre-registered rule and its outcome, quoted from `summary.json`;
- the headline table for all rows (AUROC, AUPR and FPR95, common / extra, with intervals);
- AUROC by severity, and the families at severities 1 / 3 / 5;
- the clean mAP and the mAP by severity;
- the caveats:
  - the corruptions are at 2048 × 1024 while the detector sees 640 × 640, which weakens noise and blur at a given severity;
  - DisCoPatch resizes to 256 × 256;
  - Cityscapes val is 500 images.

Add the file and the folder to `docs/README.md`. In `docs/paper-storyline.md`, replace the "(to make)" of step 9 and claim C10 with the numbers.

- [ ] **Step 5: Commit**

```bash
git add docs/results/cityscapes docs/cityscapes-c-results.md
git commit -m "results: Cityscapes-C, our method and six baselines on the Cityscapes fine-tune

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

`docs/README.md` and `docs/paper-storyline.md` join this commit only if the user agreed to commit the 2 October paper notes they also hold; otherwise they stay uncommitted and the task report says so.
