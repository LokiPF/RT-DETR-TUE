# Four Detectors on COCO-C Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show that the two-axis score works across detector designs. It runs, unchanged, on three more frozen COCO detectors (YOLO11m, Faster R-CNN R50-FPN v2 and RF-DETR-M) next to RT-DETRv2-R18. Each detector gets every baseline it supports, on the same 5,000 COCO val images and 96 conditions.

**Architecture:**
- **Adapters.** `degradation_monitor/detectors/` holds one adapter per detector. An adapter turns a batch of same-size RGB images into the `Outputs` every monitor needs, from one forward pass:
  - the method's four levels;
  - the activation CDFs' five maps;
  - pooled features for kNN;
  - the top 100 detections;
  - for DETR-type detectors, also every query's logits and box, and the last decoder layer.

  With `heads=False` it runs the backbone only, which is all the clean fits need.
- **Stages.** `degradation_monitor/stages/detectors.py` fits each detector's clean references in three resumable passes. It then makes one shared pass over the val images: each image's 96 corrupted versions are generated once and fed to all three detectors. Each detector writes into its own run folder, laid out like `runs/coco/`, so the existing report stage reads it unchanged except for rows a detector lacks.
- **RT-DETRv2-R18's results stay in `runs/coco/`.** That run supplies the corruption digests, the DisCoPatch scores and the fourth row of the final table.

**Tech Stack:** Python 3.11, PyTorch 2.11, torchvision 0.26, Ultralytics 8.3.235, rfdetr 1.11.2, pycocotools, imagecorruptions, pytest. The `UE` conda environment is `/home/yuchen/miniconda3/envs/UE`.

**Spec:**
- The decisions of 4 October 2026, written out under Global Constraints: the detectors, the tap rule, the baselines per detector, the floors and the pre-registered rule.
- `docs/driving-benchmark-baselines-and-metrics.md`: baselines, metrics and fairness rules.
- `docs/paper-storyline.md`: why the paper needs several detectors.

## Global Constraints

- **Detectors** (COCO-trained, frozen, used as released):
  - RT-DETRv2-R18: its results are in `runs/coco/` and are not re-run.
  - YOLO11m: `/home/yuchen/YuchenZ/lab/Detector_test/yolo11m.pt`, Ultralytics 8.3.235.
  - Faster R-CNN R50-FPN v2: torchvision's `COCO_V1` weights, `/home/yuchen/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth`.
  - RF-DETR-M: `/home/yuchen/.roboflow/models/rf-detr-medium.pth`, rfdetr 1.11.2.
- **Preprocessing:** each detector's own, as it is deployed.
  - YOLO11m: Ultralytics' letterbox for same-size images. The long side goes to 640, and the short side is padded with grey 114 to a multiple of 32, centred.
  - Faster R-CNN: the model's transform. ImageNet normalisation, then the short side to 800 with the long side at most 1333, then padding to a multiple of 32 at the bottom and right.
  - RF-DETR-M: resize to 576 × 576, bilinear without antialiasing, then ImageNet normalisation.
  - Padding cells never enter a statistic: every map is cropped to the cells that lie wholly on the image. Deeper cells still see the padding through their receptive fields, identically in all 96 versions of an image.
  - Float32 matmuls run at full precision in every stage (`"highest"`, PyTorch's default). Importing rfdetr switches the whole process to TF32 (`"high"`), so the RF-DETR adapter restores the previous setting, and each detector folder's protocol records the precision. Otherwise the fits and the shared pass would compute YOLO's attention, Faster R-CNN's box head and the kNN distances at different precisions.
- **The tap rule, fixed 4 October:** score the three earliest feature levels and use the deepest as the content key.
  - A level is a stage in a hierarchical backbone and a block in a plain ViT.
  - RT-DETR reads the output of each stage's first block.

  | Detector | s1, s2, s3 (scored) | s4 (key) |
  |---|---|---|
  | YOLO11m | layers 1, 3, 5: the first module of stages 1-3, a stride-2 downsampling conv (strides 4, 8, 16) | layer 7: stage 4's downsampling conv (stride 32) |
  | Faster R-CNN | `layer1[0]`, `layer2[0]`, `layer3[0]` (first bottleneck block of each stage) | `layer4[0]` |
  | RF-DETR-M | raw outputs of ViT blocks 1, 2, 3 | block 12 |

  - Each YOLO stage is a stride-2 downsampling conv followed by a C3k2 block: layers (1, 2), (3, 4), (5, 6) and (7, 8). Ultralytics numbers them separately; layer 0 is the stem, and SPPF (9) and C2PSA (10) end the backbone.
  - YOLO's taps were chosen on 4 October in a small exploration round, on 300 development images (positions 0–1969, seed 7). The scripts and logs of this round and of the ViT's are committed in `docs/results/coco-detectors/tap-exploration/` before Task 1. Two-axis AUROC, common / extra families:

    | Taps | AUROC | Minus the stage blocks (common / extra), 95% interval |
    |---|---|---|
    | 2, 4, 6 + key 8 (the stage blocks) | 0.881 / 0.840 | — |
    | 2, 4, 6 + key 10 | 0.868 / 0.828 | −0.014 / −0.012, both below 0 |
    | **1, 3, 5 + key 7 (chosen)** | **0.898 / 0.876** | +0.017 [+0.010, +0.024] / +0.035 [+0.030, +0.041] |
    | 1, 2, 3 + key 10 | 0.893 / 0.879 | +0.012 / +0.039 |
    | 0, 1, 2 + key 10 | 0.887 / 0.886 | +0.006 (interval includes 0) / +0.046 |

    The chosen set leads on the common families and mirrors RT-DETR's taps, which read each stage's first block.
  - The ViT's patch tokens are regathered from its attention windows, leaving out its class token.
- **The method is unchanged:** k = 50, the top 1%, the bank of 2,000 and the 500 z-statistics images (COCO train, the seed-44 splits), and the larger of the flattening and level arms.
- **Baselines for each detector:**
  - SAOD, from the top 100 detections.
  - kNN: k = 100 on L2-normalised pooled features. The bank is all 118,287 COCO train images.
  - Activation CDFs (Becker et al.), on five maps:
    - YOLO11m: the stem (layer 0, stride 2) and the four stage outputs, layers 2, 4, 6 and 10 (10 is the end of the backbone, after SPPF and C2PSA), as for the ResNets: what enters stage 1, then each stage's output;
    - Faster R-CNN: the stem after max pooling and the four stage outputs;
    - RF-DETR-M: the embeddings and blocks 1, 2, 3 and 12, the method's own levels plus the stem, because a ViT has no stage outputs.
  - DisCoPatch: RT-DETR's scores, hard-linked, because it reads the image, not the detector.
  - ContrastiveConf and Hashemi et al. (on the last decoder layer's queries): RF-DETR-M only. Both are defined for DETR-type detectors only.
  - kNN pooled features: YOLO11m's layer 10, Faster R-CNN's `layer4` output, and RF-DETR-M's block 12 after the backbone's LayerNorm.
- **Detections:** the top 100 per image, with labels 0–79 in COCO's sorted category order. Empty slots have score 0.
  - YOLO11m: NMS as Ultralytics' predict does it (one label per box), at confidence 0.001 and IoU 0.7, with no time limit.
  - Faster R-CNN: score threshold 0.001, 100 per image.
  - RF-DETR-M: the top 100 (query, class) pairs of sigmoid scores over the 80 COCO classes, as for RT-DETR.
- **Evaluation:** all 5,000 COCO val images in the seed-44 order, with the 5 folds and the 96 conditions. The corrupted images must equal `runs/coco/`'s; the shared pass checks each image's digests.
- **Pre-registered rules:** for each detector, the COCO headline rule and the level rule, as `headline_decision` and `level_decision` in `degradation_monitor/evaluation/report.py`, on that detector's scores. Every detector is reported, whatever its outcome.
  - The ViT's taps were chosen on positions 0–1969 only, and YOLO's exploration reads only those positions too, so the untouched positions 1970–4999 confirm for every new detector.
- **Clean AP floors**, the published COCO val AP minus 0.03: YOLO11m 0.485, Faster R-CNN v2 0.437, RF-DETR-M 0.517.
- **Run folders:**
  - each detector's manifest records a protocol: RT-DETR's protocol with the detector's weights, plus the detector's name and its adapter's `protocol` (taps, input size, thresholds), so a run folder made with other taps is refused;
  - each score folder records its inputs: the sha1 of the fits it was computed from.
- **The existing COCO pipeline stays as it is:**
  - `tests/evaluation/test_report_golden.py` and `tests/test_equivalence.py` stay green;
  - `runs/coco/` is only read, never written;
  - `configs/coco.toml` is unchanged.
- **Tests run on the CPU only:** `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`.
  - The suite at the start: 159 passed, 1 skipped (checked on 4 October).
  - Test folders have no `__init__.py`, so every test file needs a unique base name.
  - Tests that load real weights skip when the weights are absent.
- **Environment:**
  - `ultralytics==8.3.235` and `rfdetr==1.11.2` were installed into `UE` on 4 October, and `pip check` is clean.
  - `tests/test_package.py` requires `requirements.txt` to name exactly the libraries the package imports, so Tasks 2 and 4 pin each library as they first import it.
- **The GPU is shared** with the session "explore". Before each GPU task (7–10):
  - ask explore via SendMessage and wait for its answer;
  - the detectors config caps the process at `gpu_memory_gib = 8.0`;
  - send "done" when the task finishes.
- **Git and paths:**
  - all work happens in the worktree `/home/yuchen/YuchenZ/UE/philip_sa/.worktrees/detectors`, on branch `detectors`, which already exists, cut from `fingerprint_bank` at `e5e6050`; never `cd` to the main checkout, which stays on `fingerprint_bank`;
  - `runs/` is git-ignored and exists only in the main checkout, so every path into it is absolute, `/home/yuchen/YuchenZ/UE/philip_sa/runs/...`, as the configs' paths already are;
  - this plan and the tap explorations are committed on `detectors` before Task 1;
  - commit at the end of each task, with both trailers in one `-m`: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU`;
  - `git add` only the files a task names, and never push;
  - `docs/superpowers/plans/2026-10-04-cityscapes-c-evaluation.md` stays uncommitted in the main checkout; it will be revised after these results.

## Review Focus

1. **Padding inside a detector's input** (YOLO's letterbox rows, Faster R-CNN's padding to 32) must stay out of the channel statistics. Otherwise the bank and the test images are compared on different regions. Tests: Task 1 `test_region_crops_the_padding_away`, Task 2 `test_yolo_levels_and_cdf_maps_have_the_backbone_shapes` and `test_yolo_crops_an_odd_letterbox_padding_where_ultralytics_puts_the_image`, Task 3 `test_faster_rcnn_levels_have_the_resnet50_shapes_without_padding`.
2. **Label conventions:** YOLO uses 0–79, while torchvision and RF-DETR use COCO category ids 1–90. All must end as labels 0–79 in COCO's sorted order, or the mAP and the per-image AP are wrong. Tests: Task 1 `test_the_category_mapping_matches_the_coco_annotations`, and the detection tests of Tasks 2–4.
3. **The fits run the backbone alone** (`heads=False`), some before RF-DETR is loaded and the shared pass with it loaded. They must see the very maps the shared pass sees, at the same matmul precision, or the references describe other features than the test images. Tests: the `heads=False` assertions in Tasks 2, 3 and 4; Task 4's check that loading RF-DETR leaves the precision at `"highest"`; Task 6's check that the protocol records it.
4. **The shared pass must feed every detector RT-DETR's own corrupted images, and must refuse fits that changed after scores were written.** Tests: Task 6 `test_the_pass_refuses_corruptions_that_differ_from_the_reference_run` and `test_the_pass_refuses_scores_from_a_changed_fit`.
5. **A detector without ContrastiveConf or Hashemi still gets a full report, and an interrupted fit or pass redoes only what is missing.** Tests: Task 5 `test_the_report_stage_leaves_out_the_rows_a_detector_does_not_have`, Task 6 `test_an_interrupted_fit_redoes_only_the_missing_passes` and `test_the_shared_pass_writes_three_files_per_detector_and_resumes`.

---

### Task 1: The adapter types

**Files:**
- Create: `degradation_monitor/detectors/__init__.py`, `degradation_monitor/detectors/base.py`
- Test: `tests/detectors/test_base.py`

**Interfaces:**
- Produces:
  - `Outputs` (dataclass): required `levels: dict` ("s1"–"s4" → (N, C, H, W)), `cdf: list` (5 maps) and `pooled: Tensor (N, D)`; then, `None` unless the heads ran, `scores (N, 100)`, `labels (N, 100) int64`, `boxes (N, 100, 4)`, `query_logits (N, Q, 80)`, `query_boxes (N, Q, 4)` and `decoder (N, Q, D)`.
  - `Region(top, left, height, width)`, with `.crop(maps, stride)`.
  - `padded(scores, labels, boxes, k=100)`.
  - The constants `TOP_K = 100`, `LEVELS`, `COCO_CATEGORY_IDS`, `LABEL_OF_CATEGORY` and `DETECTORS`.
  - `adapter_class(name)`, which imports no detector library, and `load_adapter(name, weights, device)`.
  - The adapter contract, which Tasks 2–4 implement and Task 6 relies on:
    - `__call__(arrays, heads=True) -> Outputs`;
    - `close()`;
    - class attributes `name`, `batch_size`, `fit_batch_size`, `detr: bool` and `protocol: dict`.

- [ ] **Step 1: Write the failing tests**

Create `tests/detectors/test_base.py` (the folder gets no `__init__.py`, like the other test folders):

```python
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from degradation_monitor.detectors import COCO_CATEGORY_IDS, LABEL_OF_CATEGORY, Region, load_adapter, padded

ANNOTATIONS = Path("/home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json")


def test_padded_keeps_the_most_confident_and_fills_empty_slots():
    scores, labels, boxes = padded(np.array([0.2, 0.9, 0.5]), np.array([3, 1, 2]), np.arange(12.0).reshape(3, 4), k=5)
    assert scores.tolist() == pytest.approx([0.9, 0.5, 0.2, 0.0, 0.0])
    assert labels.tolist() == [1, 2, 3, 0, 0]
    assert boxes[0].tolist() == [4, 5, 6, 7] and not boxes[3:].any()


def test_region_crops_the_padding_away():
    maps = torch.arange(10 * 8, dtype=torch.float32).reshape(1, 1, 10, 8)  # a stride-4 map of a 40 x 32 input
    region = Region(top=4.0, left=0.0, height=28.0, width=30.0)  # 4 grey rows above, 8 below, 2 columns right
    cropped = region.crop(maps, stride=4)
    assert tuple(cropped.shape) == (1, 1, 7, 7) and cropped[0, 0, 0, 0] == maps[0, 0, 1, 0]


def test_the_category_mapping_matches_the_coco_annotations():
    assert len(COCO_CATEGORY_IDS) == 80 and LABEL_OF_CATEGORY[1] == 0 and LABEL_OF_CATEGORY[90] == 79
    assert LABEL_OF_CATEGORY[12] == -1
    if ANNOTATIONS.exists():
        ids = sorted(c["id"] for c in json.loads(ANNOTATIONS.read_text())["categories"])
        assert tuple(ids) == COCO_CATEGORY_IDS


def test_an_unknown_detector_is_refused():
    with pytest.raises(ValueError, match="unknown detector 'ssd'"):
        load_adapter("ssd", None, "cpu")
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/detectors/test_base.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'degradation_monitor.detectors'`.

- [ ] **Step 3: Implement**

Create `degradation_monitor/detectors/base.py`:

```python
"""What every frozen detector gives the monitors from one forward pass over images of one size."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

TOP_K = 100  # detections kept per image, as for RT-DETR (degradation_monitor.detector.postprocess.TOP_K)
LEVELS = ("s1", "s2", "s3", "s4")  # the method's three scored levels, then its content key
# COCO's 80 category ids in sorted order: label i is COCO_CATEGORY_IDS[i], as for RT-DETR and YOLO
COCO_CATEGORY_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31,
                     32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                     58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87,
                     88, 89, 90)
LABEL_OF_CATEGORY = np.full(91, -1, dtype=np.int64)  # COCO category id -> label 0-79, -1 for the unused ids
LABEL_OF_CATEGORY[list(COCO_CATEGORY_IDS)] = np.arange(len(COCO_CATEGORY_IDS))


@dataclass
class Outputs:
    """One forward pass over N images of one size.

    levels: the method's maps by LEVELS name, (N, C, H, W) each. cdf: the activation CDFs' five maps, stem first.
    pooled: (N, D) features for the kNN baseline. Only when the heads ran: scores, labels and boxes, the TOP_K most
    confident detections ((N, TOP_K), (N, TOP_K) and (N, TOP_K, 4); labels 0-79 in COCO's category order, boxes xyxy
    in the original image's pixels, empty slots with score 0); and for DETR-type detectors query_logits (N, Q, 80)
    and query_boxes (N, Q, 4, cxcywh in [0, 1]) for ContrastiveConf, and decoder, the last decoder layer's queries
    (N, Q, D), for Hashemi et al.
    """
    levels: dict
    cdf: list
    pooled: torch.Tensor
    scores: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    boxes: Optional[np.ndarray] = None
    query_logits: Optional[np.ndarray] = None
    query_boxes: Optional[np.ndarray] = None
    decoder: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class Region:
    """Where the image lies in the network's input, in input pixels."""
    top: float
    left: float
    height: float
    width: float

    def crop(self, maps: torch.Tensor, stride: int) -> torch.Tensor:
        """The cells of a stride-`stride` map that lie wholly on the image (at least one row and one column)."""
        top, left = math.ceil(self.top / stride), math.ceil(self.left / stride)
        bottom = max(top + 1, math.floor((self.top + self.height) / stride))
        right = max(left + 1, math.floor((self.left + self.width) / stride))
        return maps[..., top:bottom, left:right]


def padded(scores, labels, boxes, k: int = TOP_K):
    """The k most confident detections; empty slots get score 0, label 0 and an empty box."""
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    order = np.argsort(-scores, kind="stable")[:k]
    out_scores, out_labels, out_boxes = np.zeros(k, np.float32), np.zeros(k, np.int64), np.zeros((k, 4), np.float32)
    out_scores[:len(order)], out_labels[:len(order)], out_boxes[:len(order)] = scores[order], labels[order], boxes[order]
    return out_scores, out_labels, out_boxes
```

Create `degradation_monitor/detectors/__init__.py`:

```python
"""Frozen COCO detectors beyond RT-DETRv2-R18, each behind one adapter that gives every monitor its inputs.

An adapter is called on a list of same-size RGB uint8 arrays and returns Outputs; with heads=False it runs the
backbone only, which is all the clean fits need. Its class attributes: name; batch_size, the versions of one image per
forward pass; fit_batch_size, the clean train images per forward pass (1 when the input size follows the image's);
detr, whether it also gives query logits and decoder queries; and protocol, the taps, input size and thresholds its
results depend on, which its run folder's manifest records.
"""
from __future__ import annotations

from .base import COCO_CATEGORY_IDS, LABEL_OF_CATEGORY, LEVELS, TOP_K, Outputs, Region, padded

DETECTORS = ("yolo11m", "faster_rcnn_r50_fpn_v2", "rfdetr_m")


def adapter_class(name: str):
    """The adapter class of one detector; importing it loads no detector library."""
    if name == "yolo11m":
        from .yolo import Yolo11m
        return Yolo11m
    if name == "faster_rcnn_r50_fpn_v2":
        from .faster_rcnn import FasterRcnn
        return FasterRcnn
    if name == "rfdetr_m":
        from .rfdetr import RfDetrM
        return RfDetrM
    raise ValueError(f"unknown detector {name!r}; choose from {', '.join(DETECTORS)}")


def load_adapter(name: str, weights, device):
    """The adapter of one detector, with its weights loaded and frozen on the device."""
    return adapter_class(name)(weights, device)


__all__ = ["COCO_CATEGORY_IDS", "DETECTORS", "LABEL_OF_CATEGORY", "LEVELS", "TOP_K", "Outputs", "Region",
           "adapter_class", "load_adapter", "padded"]
```

- [ ] **Step 4: Run the tests to see them pass, then the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/detectors/test_base.py`
Expected: PASS.

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: `163 passed, 1 skipped`.

- [ ] **Step 5: Commit**

```bash
git add degradation_monitor/detectors/__init__.py degradation_monitor/detectors/base.py tests/detectors/test_base.py
git commit -m "feat: adapter types for detectors beyond RT-DETR

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

---

### Task 2: The YOLO11m adapter

**Files:**
- Create: `degradation_monitor/detectors/yolo.py`
- Modify: `requirements.txt` (add the line `ultralytics==8.3.235`)
- Test: `tests/detectors/test_yolo.py`

**Interfaces:**
- Consumes: `Outputs`, `Region`, `padded`, `TOP_K` (Task 1).
- Produces: `Yolo11m(weights, device)`, with:
  - class attributes `name = "yolo11m"`, `batch_size = 16`, `fit_batch_size = 1`, `detr = False` and `protocol`;
  - `__call__(arrays, heads=True) -> Outputs`;
  - `prepare(arrays) -> (batch, Region)`;
  - `close()`.

- [ ] **Step 1: Write the failing tests**

Create `tests/detectors/test_yolo.py`:

```python
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

WEIGHTS = Path("/home/yuchen/YuchenZ/lab/Detector_test/yolo11m.pt")
IMAGE = Path("/home/yuchen/YuchenZ/Datasets/coco/val2017/000000000139.jpg")  # 426 rows x 640 columns


@pytest.fixture(scope="module")
def adapter():
    if not (WEIGHTS.exists() and IMAGE.exists()):
        pytest.skip("YOLO11m's weights or the COCO image are not available")
    from degradation_monitor.detectors import load_adapter
    return load_adapter("yolo11m", WEIGHTS, "cpu")


def _image():
    return np.array(Image.open(IMAGE).convert("RGB"))  # writable, as the stages' images are


def test_yolo_levels_and_cdf_maps_have_the_backbone_shapes(adapter):
    image = _image()
    out = adapter([image, image])
    # 426 x 640 letterboxes to 448 x 640, 11 grey rows above and below; the crops keep only the image's cells
    assert {n: tuple(m.shape) for n, m in out.levels.items()} == {
        "s1": (2, 128, 106, 160), "s2": (2, 256, 52, 80), "s3": (2, 512, 26, 40), "s4": (2, 512, 12, 20)}
    assert tuple(out.cdf[0].shape) == (2, 64, 212, 320)  # the stride-2 stem, its 6 grey rows above cropped away
    assert [m.shape[1] for m in out.cdf] == [64, 256, 512, 512, 512] and tuple(out.pooled.shape) == (2, 512)
    assert out.scores.shape == (2, 100) and out.query_logits is None and out.decoder is None
    backbone = adapter([image], heads=False)  # what the clean fits see
    assert backbone.scores is None
    assert torch.allclose(backbone.levels["s1"], out.levels["s1"][:1], atol=1e-4)
    assert torch.allclose(backbone.pooled, out.pooled[:1], atol=1e-4)


def test_yolo_detections_match_ultralytics_predict(adapter):
    from ultralytics import YOLO

    image = _image()
    out = adapter([image])
    reference = YOLO(str(WEIGHTS)).predict(image[..., ::-1].copy(), conf=0.001, iou=0.7, max_det=100, imgsz=640,
                                           device="cpu", verbose=False)[0]  # Ultralytics takes numpy images as BGR
    confidences = reference.boxes.conf.numpy()
    expected = np.sort(confidences)[::-1]
    assert out.scores[0, :len(expected)] == pytest.approx(expected, abs=1e-3)
    best = int(np.argmax(confidences))
    assert out.labels[0, 0] == int(reference.boxes.cls[best])
    assert out.boxes[0, 0] == pytest.approx(reference.boxes.xyxy[best].numpy(), abs=1.0)


def test_yolo_crops_an_odd_letterbox_padding_where_ultralytics_puts_the_image(adapter):
    image = np.random.default_rng(0).integers(0, 256, (427, 640, 3), dtype=np.uint8)  # 21 grey rows: 10 above, 11 below
    out = adapter([image], heads=False)
    assert tuple(out.cdf[0].shape) == (1, 64, 213, 320)  # the stride-2 stem keeps the cell of rows 10-11
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/detectors/test_yolo.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'degradation_monitor.detectors.yolo'`.

- [ ] **Step 3: Implement**

Add the line `ultralytics==8.3.235` to `requirements.txt`; `tests/test_package.py` requires it once the package imports `ultralytics`.

Create `degradation_monitor/detectors/yolo.py`:

```python
"""YOLO11m (Ultralytics 8.3.235, COCO val AP 51.5): one-stage, anchor-free, a CSP backbone with SiLU, NMS.

The input is letterboxed as Ultralytics does for same-size images: the long side to 640, the short side padded with
grey 114 to a multiple of 32, centred. Each stage is a stride-2 downsampling conv followed by a C3k2 block: layers
(1, 2), (3, 4), (5, 6) and (7, 8); layer 0 is the stem, and SPPF (9) and C2PSA (10) end the backbone. Levels: each
stage's first module, as RT-DETR's taps read each stage's first block: layers 1, 3 and 5 (strides 4, 8, 16) scored
and layer 7 (stride 32) the key, chosen on 300 development images on 4 October over the stage blocks 2, 4, 6 + 8.
CDF maps: the stem and the four stage outputs, the last at the end of the backbone (layers 0, 2, 4, 6 and 10). kNN:
layer 10, mean-pooled. Detections: NMS as Ultralytics' predict does it (one label per box), at confidence 0.001 and
IoU 0.7 and with no time limit, the 100 most confident.
"""
from __future__ import annotations

import numpy as np
import torch

from .base import TOP_K, Outputs, Region, padded

LEVEL_LAYERS = {"s1": (1, 4), "s2": (3, 8), "s3": (5, 16), "s4": (7, 32)}  # level -> (layer index, stride)
CDF_LAYERS = ((0, 2), (2, 4), (4, 8), (6, 16), (10, 32))
POOLED_LAYER = (10, 32)
BACKBONE_LAYERS = 11  # layers 0-10 run in sequence; the neck and the head follow
CONFIDENCE, IOU, SIZE = 0.001, 0.7, 640


class Yolo11m:
    name = "yolo11m"
    batch_size = 16  # the 96 versions of one image, 16 at a time
    fit_batch_size = 1  # clean train images differ in size, and a letterboxed batch needs one size
    detr = False
    protocol = {"levels": LEVEL_LAYERS, "cdf": CDF_LAYERS, "pooled": POOLED_LAYER, "size": SIZE,
                "confidence": CONFIDENCE, "iou": IOU}

    def __init__(self, weights, device):
        from ultralytics import YOLO

        self.device = torch.device(device)
        self.model = YOLO(str(weights)).model.to(self.device).eval().requires_grad_(False)
        self._maps = {}
        layers = {i for i, _ in LEVEL_LAYERS.values()} | {i for i, _ in CDF_LAYERS} | {POOLED_LAYER[0]}
        self._handles = [self.model.model[i].register_forward_hook(self._keep(i)) for i in sorted(layers)]

    def _keep(self, index):
        def hook(_module, _inputs, output):
            self._maps[index] = output
        return hook

    def prepare(self, arrays) -> tuple[torch.Tensor, Region]:
        """The letterboxed batch (RGB in [0, 1]) and where the image lies in it."""
        from ultralytics.data.augment import LetterBox

        height, width = arrays[0].shape[:2]
        letterbox = LetterBox(new_shape=(SIZE, SIZE), auto=True, stride=32)
        boxed = [letterbox(image=np.ascontiguousarray(a)) for a in arrays]
        input_height, input_width = boxed[0].shape[:2]
        ratio = min(SIZE / height, SIZE / width)
        new_height, new_width = round(height * ratio), round(width * ratio)
        region = Region(top=(input_height - new_height) // 2, left=(input_width - new_width) // 2,
                        height=new_height, width=new_width)  # an odd grey row or column goes below or right
        batch = torch.from_numpy(np.stack(boxed)).permute(0, 3, 1, 2).float().div_(255.0)
        return batch, region

    @torch.inference_mode()
    def __call__(self, arrays, heads: bool = True) -> Outputs:
        batch, region = self.prepare(arrays)
        x = batch.to(self.device)
        self._maps.clear()
        if heads:
            prediction = self.model(x)
        else:
            for layer in self.model.model[:BACKBONE_LAYERS]:
                x = layer(x)
        out = Outputs(levels={n: region.crop(self._maps[i].float(), s) for n, (i, s) in LEVEL_LAYERS.items()},
                      cdf=[region.crop(self._maps[i].float(), s) for i, s in CDF_LAYERS],
                      pooled=region.crop(self._maps[POOLED_LAYER[0]].float(), POOLED_LAYER[1]).mean(dim=(2, 3)))
        self._maps.clear()  # a later call's peak memory must not include these maps
        if heads:
            out.scores, out.labels, out.boxes = self._detections(prediction, batch.shape[2:], arrays[0].shape[:2])
        return out

    def _detections(self, prediction, input_shape, image_shape):
        """The 100 most confident boxes after NMS, in the original image's pixels."""
        from ultralytics.utils import ops
        from ultralytics.utils.nms import non_max_suppression

        prediction = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
        # Ultralytics stops NMS after 2 + 0.05 x batch seconds and leaves the rest of the batch empty: no time limit
        kept = non_max_suppression(prediction, conf_thres=CONFIDENCE, iou_thres=IOU, max_det=TOP_K, max_time_img=1e3)
        detections = []
        for found in kept:
            boxes = ops.scale_boxes(tuple(input_shape), found[:, :4].clone(), image_shape)
            detections.append(padded(found[:, 4].cpu().numpy(), found[:, 5].long().cpu().numpy(),
                                     boxes.cpu().numpy()))
        return tuple(np.stack(values) for values in zip(*detections))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
```

- [ ] **Step 4: Run the tests to see them pass, then the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/detectors/test_yolo.py tests/test_package.py`
Expected: PASS.

If the detection test fails on a value, compare `prepare` and `_detections` with Ultralytics' `LetterBox` and `non_max_suppression` arguments before changing the test. The test's numbers come from Ultralytics' own `predict`.

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: `166 passed, 1 skipped`.

- [ ] **Step 5: Commit**

```bash
git add degradation_monitor/detectors/yolo.py requirements.txt tests/detectors/test_yolo.py
git commit -m "feat: the YOLO11m adapter

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

---

### Task 3: The Faster R-CNN adapter

**Files:**
- Create: `degradation_monitor/detectors/faster_rcnn.py`
- Test: `tests/detectors/test_faster_rcnn.py`

**Interfaces:**
- Consumes: `Outputs`, `Region`, `padded`, `TOP_K`, `LABEL_OF_CATEGORY` (Task 1).
- Produces: `FasterRcnn(weights, device)`, with:
  - class attributes `name = "faster_rcnn_r50_fpn_v2"`, `batch_size = 8`, `fit_batch_size = 1`, `detr = False` and `protocol`;
  - `model`;
  - `__call__(arrays, heads=True) -> Outputs`;
  - `close()`.

- [ ] **Step 1: Write the failing tests**

Create `tests/detectors/test_faster_rcnn.py`:

```python
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

WEIGHTS = Path("/home/yuchen/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth")
IMAGE = Path("/home/yuchen/YuchenZ/Datasets/coco/val2017/000000000139.jpg")


@pytest.fixture(scope="module")
def adapter():
    if not (WEIGHTS.exists() and IMAGE.exists()):
        pytest.skip("Faster R-CNN's weights or the COCO image are not available")
    from degradation_monitor.detectors import load_adapter
    return load_adapter("faster_rcnn_r50_fpn_v2", WEIGHTS, "cpu")


def _image():
    return np.array(Image.open(IMAGE).convert("RGB"))  # writable, as the stages' images are


def test_faster_rcnn_levels_have_the_resnet50_shapes_without_padding(adapter):
    image = _image()
    out = adapter([image])
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    height, width = adapter.model.transform([tensor])[0].image_sizes[0]  # inside the batch padded to 32
    expected = {"s1": (256, 4), "s2": (512, 8), "s3": (1024, 16), "s4": (2048, 32)}
    for name, (channels, stride) in expected.items():
        assert tuple(out.levels[name].shape) == (1, channels, height // stride, width // stride)
    assert [m.shape[1] for m in out.cdf] == [64, 256, 512, 1024, 2048] and tuple(out.pooled.shape) == (1, 2048)
    backbone = adapter([image], heads=False)  # what the clean fits see
    assert backbone.scores is None and torch.allclose(backbone.levels["s2"], out.levels["s2"], atol=1e-4)


def test_faster_rcnn_detections_match_the_model(adapter):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

    from degradation_monitor.detectors import LABEL_OF_CATEGORY

    image = _image()
    out = adapter([image])
    model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    model.roi_heads.score_thresh, model.roi_heads.detections_per_img = 0.001, 100
    with torch.inference_mode():
        result = model.eval()([torch.from_numpy(image).permute(2, 0, 1).float() / 255.0])[0]
    n = len(result["scores"])
    assert out.scores[0, :n] == pytest.approx(result["scores"].numpy(), abs=1e-4)
    assert out.labels[0, :n].tolist() == LABEL_OF_CATEGORY[result["labels"].numpy()].tolist()
    assert (out.labels[0] >= 0).all() and out.scores[0, n:].sum() == 0
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/detectors/test_faster_rcnn.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'degradation_monitor.detectors.faster_rcnn'`.

- [ ] **Step 3: Implement**

Create `degradation_monitor/detectors/faster_rcnn.py`:

```python
"""Faster R-CNN R50-FPN v2 (torchvision COCO_V1, box AP 46.7): two-stage, ResNet-50 with batch norm and an FPN.

The input goes through the model's own transform: the short side to 800 (the long side at most 1333), ImageNet
normalisation, padding to a multiple of 32 at the bottom and right. Levels: the first bottleneck block of each stage,
as RT-DETR's taps read the first block of each of its stages: layer1[0]-layer3[0] (strides 4, 8, 16) scored and
layer4[0] (stride 32) the key. CDF maps: the stem after max pooling and the four stage outputs. kNN: layer4's output,
mean-pooled. Detections: the model's own, with the score threshold lowered to 0.001 and 100 per image.
"""
from __future__ import annotations

import numpy as np
import torch

from .base import LABEL_OF_CATEGORY, TOP_K, Outputs, Region, padded

STRIDES = {1: 4, 2: 8, 3: 16, 4: 32}  # stage -> stride
SCORE_THRESHOLD = 0.001


class FasterRcnn:
    name = "faster_rcnn_r50_fpn_v2"
    batch_size = 8
    fit_batch_size = 1  # the crop needs one image size per batch
    detr = False
    protocol = {"levels": "layer1[0]-layer3[0], key layer4[0]", "cdf": "maxpool, layer1-layer4",
                "pooled": "layer4", "size": [800, 1333], "score_threshold": SCORE_THRESHOLD}

    def __init__(self, weights, device):
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

        self.device = torch.device(device)
        model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        model.roi_heads.score_thresh, model.roi_heads.detections_per_img = SCORE_THRESHOLD, TOP_K
        self.model = model.to(self.device).eval().requires_grad_(False)
        body, self._maps, self._sizes = self.model.backbone.body, {}, None
        self._handles = [body.maxpool.register_forward_hook(self._keep("stem")),
                         self.model.transform.register_forward_hook(self._keep_sizes)]
        for stage in STRIDES:
            layer = getattr(body, f"layer{stage}")
            self._handles += [layer[0].register_forward_hook(self._keep(f"s{stage}")),
                              layer.register_forward_hook(self._keep(f"C{stage + 1}"))]

    def _keep(self, name):
        def hook(_module, _inputs, output):
            self._maps[name] = output
        return hook

    def _keep_sizes(self, _module, _inputs, output):
        self._sizes = output[0].image_sizes  # the resized images inside the padded batch

    @torch.inference_mode()
    def __call__(self, arrays, heads: bool = True) -> Outputs:
        images = [torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float().div(255.0).to(self.device)
                  for a in arrays]
        self._maps.clear()
        if heads:
            results = self.model(images)
        else:
            self.model.backbone.body(self.model.transform(images)[0].tensors)
        height, width = self._sizes[0]
        region = Region(top=0.0, left=0.0, height=height, width=width)
        out = Outputs(levels={f"s{s}": region.crop(self._maps[f"s{s}"].float(), stride) for s, stride in STRIDES.items()},
                      cdf=[region.crop(self._maps["stem"].float(), 4)]
                          + [region.crop(self._maps[f"C{s + 1}"].float(), stride) for s, stride in STRIDES.items()],
                      pooled=region.crop(self._maps["C5"].float(), 32).mean(dim=(2, 3)))
        self._maps.clear()  # a later call's peak memory must not include these maps
        if heads:
            out.scores, out.labels, out.boxes = self._detections(results)
        return out

    @staticmethod
    def _detections(results):
        detections = []
        for result in results:
            labels = LABEL_OF_CATEGORY[result["labels"].cpu().numpy()]
            known = labels >= 0  # torchvision's 91-slot head never predicts the unused ids, but keep only COCO's 80
            detections.append(padded(result["scores"].cpu().numpy()[known], labels[known],
                                     result["boxes"].cpu().numpy()[known]))
        return tuple(np.stack(values) for values in zip(*detections))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
```

- [ ] **Step 4: Run the tests to see them pass, then the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/detectors/test_faster_rcnn.py`
Expected: PASS.

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: `168 passed, 1 skipped`.

- [ ] **Step 5: Commit**

```bash
git add degradation_monitor/detectors/faster_rcnn.py tests/detectors/test_faster_rcnn.py
git commit -m "feat: the Faster R-CNN R50-FPN v2 adapter

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

---

### Task 4: The RF-DETR-M adapter

**Files:**
- Create: `degradation_monitor/detectors/rfdetr.py`
- Modify: `requirements.txt` (add the line `rfdetr==1.11.2`)
- Test: `tests/detectors/test_rfdetr.py`

**Interfaces:**
- Consumes:
  - `Outputs`, `TOP_K`, `COCO_CATEGORY_IDS` (Task 1);
  - `degradation_monitor.detector.postprocess.top_detections(logits, boxes_cxcywh, image_size, top_k)`, with `image_size = (width, height)`.
- Produces: `RfDetrM(weights, device)`, with:
  - class attributes `name = "rfdetr_m"`, `batch_size = 32`, `fit_batch_size = 32`, `detr = True` and `protocol`;
  - `__call__(arrays, heads=True) -> Outputs`, which with the heads also fills `query_logits`, `query_boxes` and `decoder`;
  - `maps(tokens)`, `core`, `backbone`, `layernorm` and `close()`.

These attribute paths and shapes were checked on the CPU on 4 October:
- `RFDETRMedium(pretrain_weights=<absolute path>)` loads the file;
- `core = RFDETRMedium(...).model.model` gives `pred_logits` of shape (B, 300, 91), indexed by COCO category id;
- `core.backbone[0].encoder` returns the four LayerNorm-ed maps (B, 384, 36, 36), and its `.encoder` is the windowed ViT (patch 16, no registers, 12 blocks), each block returning a tuple whose first item is (4B, 325, 384);
- `core.transformer.decoder.layers[-1]` returns (B, 300, 256).

- [ ] **Step 1: Write the failing tests**

Create `tests/detectors/test_rfdetr.py`:

```python
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision.transforms.v2.functional as F
from PIL import Image

WEIGHTS = Path("/home/yuchen/.roboflow/models/rf-detr-medium.pth")
IMAGE = Path("/home/yuchen/YuchenZ/Datasets/coco/val2017/000000000139.jpg")


@pytest.fixture(scope="module")
def adapter():
    if not (WEIGHTS.exists() and IMAGE.exists()):
        pytest.skip("RF-DETR-M's weights or the COCO image are not available")
    from degradation_monitor.detectors import load_adapter
    return load_adapter("rfdetr_m", WEIGHTS, "cpu")


def _image():
    return np.array(Image.open(IMAGE).convert("RGB"))  # writable, as the stages' images are


def test_rfdetr_blocks_regather_into_the_detectors_own_feature_maps(adapter):
    from degradation_monitor.detectors.rfdetr import MEANS, RESOLUTION, STDS

    image = _image()
    out = adapter([image])
    assert {n: tuple(m.shape) for n, m in out.levels.items()} == {n: (1, 384, 36, 36) for n in ("s1", "s2", "s3", "s4")}
    assert len(out.cdf) == 5 and tuple(out.pooled.shape) == (1, 384)
    backbone = adapter([image], heads=False)  # what the clean fits see
    assert backbone.decoder is None and torch.allclose(backbone.levels["s4"], out.levels["s4"], atol=1e-5)
    tensor = F.resize(torch.from_numpy(image).permute(2, 0, 1).float() / 255.0, [RESOLUTION, RESOLUTION],
                      antialias=False)
    with torch.inference_mode():
        official = adapter.backbone(F.normalize(tensor[None], MEANS, STDS))  # LayerNorm-ed blocks 3, 6, 9, 12
        mine = adapter.maps(adapter.layernorm(adapter._tokens[12]))
    assert torch.allclose(mine, official[-1], atol=1e-5)


def test_rfdetr_outputs_have_the_detr_parts(adapter):
    assert torch.get_float32_matmul_precision() == "highest"  # loading RF-DETR left the other detectors' matmuls alone
    out = adapter([_image(), _image()])
    assert out.query_logits.shape == (2, 300, 80) and out.query_boxes.shape == (2, 300, 4)
    assert tuple(out.decoder.shape) == (2, 300, 256)
    best = 1 / (1 + np.exp(-out.query_logits[0].max()))
    assert out.scores[0, 0] == pytest.approx(best, abs=1e-5) and (out.labels >= 0).all()
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/detectors/test_rfdetr.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'degradation_monitor.detectors.rfdetr'`.

- [ ] **Step 3: Implement**

Add the line `rfdetr==1.11.2` to `requirements.txt`.

Create `degradation_monitor/detectors/rfdetr.py`:

```python
"""RF-DETR-M (rfdetr 1.11.2, COCO AP 54.7): a real-time DETR on a DINOv2 ViT-S backbone (12 blocks, patch 16).

The input is resized to 576 x 576 (bilinear, no antialiasing) and normalised, as its predict does. A plain ViT has
no strides, so its levels are blocks: the raw outputs of blocks 1, 2 and 3 scored and block 12 the key, their patch
tokens regathered from the attention windows (chosen on positions 0-1969, 4 October). CDF maps: the embeddings and
the same four blocks. kNN: block 12 after the backbone's LayerNorm, the feature the detector reads, mean-pooled.
Detections as RT-DETR's: the sigmoid of every query's logits for COCO's 80 categories, the top 100 (query, class)
pairs.
"""
from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms.v2.functional as F

from ..detector.postprocess import top_detections
from .base import COCO_CATEGORY_IDS, TOP_K, Outputs

BLOCKS = {"s1": 1, "s2": 2, "s3": 3, "s4": 12}
CDF_BLOCKS = (0, 1, 2, 3, 12)  # 0 is the embeddings
RESOLUTION, WINDOWS = 576, 2
MEANS, STDS = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class RfDetrM:
    name = "rfdetr_m"
    batch_size = 32
    fit_batch_size = 32  # every image is resized to 576 x 576, so a batch may mix image sizes
    detr = True
    protocol = {"levels": BLOCKS, "cdf": CDF_BLOCKS, "pooled": "layernorm(block 12)", "size": RESOLUTION,
                "classes": "the 80 COCO category columns of 91"}

    def __init__(self, weights, device):
        precision = torch.get_float32_matmul_precision()
        from rfdetr import RFDETRMedium

        self.device = torch.device(device)
        self.core = RFDETRMedium(pretrain_weights=str(weights)).model.model.to(self.device).eval().requires_grad_(False)
        torch.set_float32_matmul_precision(precision)  # importing rfdetr switched float32 matmuls to TF32 process-wide
        self.backbone = self.core.backbone[0].encoder  # returns the LayerNorm-ed maps the detector reads
        dino = self.backbone.encoder  # the windowed DINOv2 ViT-S
        self.layernorm = dino.layernorm
        self.grid = RESOLUTION // dino.config.patch_size
        self.skip = 1 + dino.config.num_register_tokens  # each window's class token and register tokens
        self._tokens = {}
        hooked = sorted(set(BLOCKS.values()) | set(CDF_BLOCKS))
        self._handles = [dino.embeddings.register_forward_hook(self._keep(0))]
        self._handles += [dino.encoder.layer[block - 1].register_forward_hook(self._keep(block, first=True))
                          for block in hooked if block > 0]
        self._handles.append(self.core.transformer.decoder.layers[-1].register_forward_hook(self._keep("decoder")))

    def _keep(self, key, first=False):
        def hook(_module, _inputs, output):
            self._tokens[key] = output[0] if first else output
        return hook

    def maps(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B * windows^2, skip + T, C) tokens -> (B, C, grid, grid), undoing the embeddings' window layout."""
        tokens = tokens[:, self.skip:]
        side = self.grid // WINDOWS
        batch = tokens.shape[0] // WINDOWS ** 2
        x = tokens.reshape(batch, WINDOWS, WINDOWS, side, side, -1)  # (B, window row, window column, h, w, C)
        return x.permute(0, 5, 1, 3, 2, 4).reshape(batch, -1, self.grid, self.grid)

    @torch.inference_mode()
    def __call__(self, arrays, heads: bool = True) -> Outputs:
        images = torch.stack([F.resize(torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float().div(255.0),
                                       [RESOLUTION, RESOLUTION], antialias=False) for a in arrays])
        x = F.normalize(images, MEANS, STDS).to(self.device)
        self._tokens.clear()
        predicted = self.core(x) if heads else self.backbone(x)
        out = Outputs(levels={name: self.maps(self._tokens[block]).float() for name, block in BLOCKS.items()},
                      cdf=[self.maps(self._tokens[block]).float() for block in CDF_BLOCKS],
                      pooled=self.maps(self.layernorm(self._tokens[12])).float().mean(dim=(2, 3)))
        if heads:
            logits = predicted["pred_logits"].float()[..., list(COCO_CATEGORY_IDS)].cpu().numpy()  # (N, 300, 80)
            boxes = predicted["pred_boxes"].float().cpu().numpy()  # (N, 300, 4), cxcywh in [0, 1]
            detections = [top_detections(l, b, (a.shape[1], a.shape[0]), TOP_K)
                          for l, b, a in zip(logits, boxes, arrays)]
            out.scores, out.labels, out.boxes = (np.stack(values) for values in zip(*detections))
            out.query_logits, out.query_boxes, out.decoder = logits, boxes, self._tokens["decoder"].float()
        self._tokens.clear()  # a later call's peak memory must not include these tokens
        return out

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
```

- [ ] **Step 4: Run the tests to see them pass, then the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/detectors/test_rfdetr.py tests/test_package.py`
Expected: PASS.

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: `170 passed, 1 skipped`.

- [ ] **Step 5: Commit**

```bash
git add degradation_monitor/detectors/rfdetr.py requirements.txt tests/detectors/test_rfdetr.py
git commit -m "feat: the RF-DETR-M adapter, with the ViT's blocks regathered from its windows

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

---

### Task 5: Hashemi on the decoder alone, and a report without the rows a detector lacks

**Files:**
- Modify: `degradation_monitor/baselines/hashemi.py`: `save_intervals` and `HashemiMonitor.__init__` handle the layers present.
- Modify: `degradation_monitor/stages/report.py`: optional detector and activation arrays.
- Modify: `degradation_monitor/evaluation/report.py`: `baseline_rows` and `build_tables` skip absent rows.
- Test: `tests/baselines/test_hashemi.py`, `tests/stages/test_report.py`

**Interfaces:**
- Produces:
  - `save_intervals(path, stats, images)` saves exactly the layers in `stats`;
  - `HashemiMonitor(path, device)` loads the layers the file holds;
  - `write_report(settings, manifest)` reads `conf_pos` and `conf_neg`, and `hashemi_decoder` and `hashemi_encoder`, only when the score files hold them;
  - `build_tables` leaves out ContrastiveConf when the detector arrays lack `conf_pos`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/baselines/test_hashemi.py`, which already imports `numpy as np`, `pytest`, `torch` and `from degradation_monitor.baselines import hashemi`:

```python
def test_hashemi_saves_and_reads_only_the_decoder(tmp_path):
    stats = hashemi.NeuronStats()
    stats.update(torch.zeros(3, 4, 2))
    stats.update(torch.ones(3, 4, 2))
    hashemi.save_intervals(tmp_path / "intervals.npz", {"decoder": stats}, images=6)
    monitor = hashemi.HashemiMonitor(tmp_path / "intervals.npz", "cpu")
    assert set(monitor.stats) == {"decoder"}
    assert monitor.decoder_share(torch.full((2, 4, 2), 10.0)).tolist() == [1.0, 1.0]
```

Append to `tests/stages/test_report.py`:

```python
def test_the_report_stage_leaves_out_the_rows_a_detector_does_not_have(run):
    """A CNN detector: no ContrastiveConf and no Hashemi; the activation CDFs alone among the activation monitors."""
    rng = np.random.default_rng(11)
    for path in sorted(run.layout.scores("detector").glob("*.npz")):
        with np.load(path) as data:
            kept = {k: data[k] for k in data.files if k not in ("conf_pos", "conf_neg")}
        atomic_npz(path, **kept)
        stages = rng.uniform(0, 1, (96, 5))
        atomic_npz(run.layout.score_file("activations", path.name), cdf_backbone=stages.sum(1),
                   cdf_backbone_z=stages.sum(1), cdf_stages=stages)
    run_stage("report", run)
    summary = json.loads((run.layout.report() / "summary.json").read_text())
    assert {"saod_top3", "saod_min", "knn", "cdf", "cdf_sum"} <= set(summary["rows"])
    assert not {"contrastive", "hashemi", "hashemi_enc"} & set(summary["rows"])
    assert summary["lambda_folds_agree"] == {}
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/baselines/test_hashemi.py tests/stages/test_report.py`
Expected: FAIL.
- The Hashemi test fails with `KeyError: 'encoder_s8'` in `save_intervals`.
- The report test fails with `KeyError: 'conf_pos is not a file in the archive'` in `stack`.

- [ ] **Step 3: Implement**

In `degradation_monitor/baselines/hashemi.py`, replace the loop of `save_intervals`:

```python
    arrays = {}
    for name in (n for n in LAYERS if n in stats):  # RT-DETR has all four layers, RF-DETR the decoder only
        arrays[f"{name}_mean"], arrays[f"{name}_std"] = stats[name].result()
```

In `HashemiMonitor.__init__`, load what the file holds:

```python
        with np.load(path, allow_pickle=False) as data:
            self.stats = {name: (torch.from_numpy(data[f"{name}_mean"]).to(device),
                                 torch.from_numpy(data[f"{name}_std"]).to(device))
                          for name in LAYERS if f"{name}_mean" in data.files}
```

In `degradation_monitor/stages/report.py`, replace the two array tuples:

```python
DETECTOR_ARRAYS = ("saod_min", "saod_top3", "knn", "det_scores", "det_labels", "det_boxes")
CONTRASTIVE_ARRAYS = ("conf_pos", "conf_neg")  # DETR-type detectors only
ACTIVATION_ARRAYS = ("cdf_backbone_z", "cdf_backbone")
HASHEMI_ARRAYS = ("hashemi_decoder", "hashemi_encoder")  # RT-DETR both, RF-DETR the decoder, CNN detectors neither
```

Add `from pathlib import Path` to the imports. Add, before `write_report`:

```python
def _stack_present(folder, names, required, optional) -> dict:
    """The required arrays, and those optional ones the score files hold: one pass writes every file alike."""
    first, present = folder / f"{Path(names[0]).stem}.npz", ()
    if first.exists():
        with np.load(first) as data:
            present = tuple(key for key in optional if key in data.files)
    return stack(folder, names, required + present)
```

In `write_report`, read the arrays through it:

```python
    detector = _stack_present(layout.scores("detector"), names, DETECTOR_ARRAYS, CONTRASTIVE_ARRAYS)
```

and

```python
    activations = _stack_present(layout.scores("activations"), names, ACTIVATION_ARRAYS, HASHEMI_ARRAYS) \
        if layout.scores("activations").exists() else None
```

In `degradation_monitor/evaluation/report.py`, make `baseline_rows` add the Hashemi rows only when present. Replace its `if activations is not None:` block and its return:

```python
    if activations is not None:
        rows.update(cdf=activations["cdf_backbone_z"], cdf_sum=activations["cdf_backbone"])
        if "hashemi_decoder" in activations:
            rows["hashemi"] = activations["hashemi_decoder"]
        if "hashemi_encoder" in activations:
            rows["hashemi_enc"] = activations["hashemi_encoder"]
    return rows
```

In `build_tables`, build ContrastiveConf only when its parts exist. Replace `rows_of`:

```python
    contrastive_ready = "conf_pos" in detector and "conf_neg" in detector

    def rows_of(chosen):
        """Every present row on the chosen images (repeats allowed), and ContrastiveConf's lambda per fold."""
        drawn, lam = {m: v[chosen] for m, v in scores.items()}, {}
        if contrastive_ready:
            drawn["contrastive"], lam = contrastive_scores(detector["conf_pos"][chosen], detector["conf_neg"][chosen],
                                                           ap[chosen], folds[chosen])
        return {m: drawn[m] for m in ROWS if m in drawn}, lam
```

In the same function, change `per_fold = () if len(set(lambdas[name].values())) == 1 else ("contrastive",)` to:

```python
        per_fold = () if len(set(lambdas[name].values())) <= 1 else ("contrastive",)
```

In the `summary` dict, change the `lambda_folds_agree` entry to:

```python
        "lambda_folds_agree": {n: len(set(v.values())) == 1 for n, v in lambdas.items() if v},
```

With ContrastiveConf present, every image set has its lambdas, so the golden numbers do not change.

- [ ] **Step 4: Run the tests to see them pass, then the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/baselines tests/stages tests/evaluation`
Expected: PASS. `tests/evaluation/test_report_golden.py` must pass unchanged.

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: `172 passed, 1 skipped`.

- [ ] **Step 5: Commit**

```bash
git add degradation_monitor/baselines/hashemi.py degradation_monitor/stages/report.py \
        degradation_monitor/evaluation/report.py tests/baselines/test_hashemi.py tests/stages/test_report.py
git commit -m "feat: Hashemi on the decoder alone; reports leave out rows a detector does not have

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

---

### Task 6: The detectors' config, fits, shared pass, reports and table

**Files:**
- Create: `degradation_monitor/stages/detectors.py`, `configs/coco-detectors.toml`
- Modify: `degradation_monitor/runs.py` (two package names), `README.md`
- Test: `tests/stages/test_detectors.py`

**Interfaces:**
- Consumes:
  - `adapter_class`, `load_adapter`, `DETECTORS`, `LEVELS`, `Outputs` and the adapter contract (Tasks 1–4);
  - `write_report` (Task 5).
- Existing helpers it uses:
  - `load_settings`;
  - from `runs`: `Manifest` (`check_protocol`, `check_inputs`, `record_environment`, `update`, `read`), `RunLayout`, `atomic_json`, `atomic_npz`, `load_npz`, `progress` and `sha1`;
  - `corruptions.digest`;
  - from `stages.common`: `variant_stream`, `cap_gpu_memory` and `image_size`;
  - `cli.positive_int`;
  - the baselines' functions, used as `stages/baselines.py` uses them.
- Produces:
  - `load_config(path, run=None) -> DetectorsConfig`, with `.base`, `.detectors`, `.settings(name)`, `.run`, `.reference_run`, `.weights`, `.floors` and `.gpu_memory_gib`;
  - the stages `check(config)`, `fit(config)`, `shared_pass(config, first=None)` and `report(config, first=None)`;
  - `image_files(adapter, references, arrays) -> dict`;
  - `cross_table(config) -> list[dict]`;
  - `main(argv)`, behind the CLI `python -m degradation_monitor.stages.detectors <check|fit|pass|report> [--config] [--run] [--first N]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/stages/test_detectors.py`:

```python
import json
import os
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image

from degradation_monitor import corruptions
from degradation_monitor.datasets import coco
from degradation_monitor.detectors import COCO_CATEGORY_IDS, Outputs
from degradation_monitor.method.statistics import KEYS
from degradation_monitor.runs import Manifest, RunLayout, atomic_npz
from degradation_monitor.stages import detectors as stage

BASE = """
run = "{root}/runs/coco"
checkpoint = "{root}/rtdetr.pth"
train_images = "{images}/train"
val_images = "{images}/val"
annotations = "{images}/ann.json"
discopatch_root = "{root}"
device = "cpu"
batch_size = 16
workers = 0
"""
DETECTORS = """
base = "base.toml"
run = "{root}/runs/coco-detectors"
reference_run = "{root}/runs/coco"
gpu_memory_gib = 8.0

[weights]
yolo11m = "{root}/yolo.pt"
rfdetr_m = "{root}/rfdetr.pth"

[clean_ap_floor]
yolo11m = 0.0
rfdetr_m = 0.0
"""


class FakeAdapter:
    """A detector stand-in whose maps brighten with the image; the DETR parts when it plays RF-DETR."""
    calls = 0

    def __init__(self, name):
        self.name, self.detr = name, name == "rfdetr_m"
        self.batch_size, self.fit_batch_size = 16, 4

    def __call__(self, arrays, heads=True):
        FakeAdapter.calls += 1
        n = len(arrays)
        level = torch.tensor([a.mean() / 255.0 for a in arrays], dtype=torch.float32)
        generator = torch.Generator().manual_seed(n)

        def maps(channels, side):
            return level.view(n, 1, 1, 1) + torch.rand(n, channels, side, side, generator=generator)

        out = Outputs(levels={"s1": maps(4, 8), "s2": maps(5, 4), "s3": maps(6, 2), "s4": maps(7, 2)},
                      cdf=[maps(3, 8), maps(3, 8), maps(3, 4), maps(3, 2), maps(3, 2)],
                      pooled=maps(16, 1).flatten(1) + 1.0)
        if heads:
            out.scores = np.tile(np.linspace(0.9, 0.0, 100, dtype=np.float32), (n, 1))
            out.labels = np.zeros((n, 100), np.int64)
            out.boxes = np.tile(np.float32([0, 0, 8, 8]), (n, 100, 1))
        if heads and self.detr:
            out.query_logits = np.full((n, 300, 80), -6.0)
            out.query_logits[:, :3, 0] = 2.0
            out.query_boxes = np.full((n, 300, 4), 0.25)
            out.decoder = maps(8, 1).flatten(1)[:, None, :].expand(n, 300, 8).contiguous()
        return out

    def close(self):
        pass


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    """24 clean train images, 3 val images without objects, and the 96 digests of each val image."""
    root = tmp_path_factory.mktemp("images")
    rng = np.random.default_rng(0)
    for folder, count in (("train", 24), ("val", 3)):
        (root / folder).mkdir()
        for index in range(count):
            Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)).save(
                root / folder / f"{index:012d}.jpg")
    (root / "ann.json").write_text(json.dumps({
        "images": [{"id": i + 1, "file_name": f"{i:012d}.jpg", "width": 64, "height": 48} for i in range(3)],
        "annotations": [], "categories": [{"id": c, "name": str(c)} for c in COCO_CATEGORY_IDS]}))
    digests = {}
    for path in sorted((root / "val").iterdir()):
        name, arrays = corruptions.load_variants(path)
        digests[name] = np.array([corruptions.digest(a) for a in arrays])
    return root, digests


@pytest.fixture
def config(tmp_path, images, monkeypatch):
    root, digests = images
    for name in ("rtdetr.pth", "yolo.pt", "rfdetr.pth"):
        (tmp_path / name).write_bytes(name.encode())
    (tmp_path / "base.toml").write_text(BASE.format(root=tmp_path, images=root))
    (tmp_path / "detectors.toml").write_text(DETECTORS.format(root=tmp_path))
    monkeypatch.setattr(coco, "SPLIT_IMAGES", (("reserved", 1), ("bank", 8), ("zstats", 4)))
    monkeypatch.setattr(stage, "KNN_K_MAX", 5)
    monkeypatch.setattr(stage, "CDF_ZSTAT_IMAGES", 6)
    monkeypatch.setattr(stage, "load_adapter", lambda name, weights, device: FakeAdapter(name))
    FakeAdapter.calls = 0
    reference = RunLayout(tmp_path / "runs" / "coco")  # RT-DETR's run: its digests and its DisCoPatch scores
    rng = np.random.default_rng(1)
    for name, values in digests.items():
        atomic_npz(reference.score_file("detector", name), digests=values)
        atomic_npz(reference.score_file("discopatch", name), dcp=rng.uniform(0, 1, 96))
    Manifest(reference).update(inputs={"discopatch": {"discriminator": "abc"}})
    return stage.load_config(tmp_path / "detectors.toml")


def _write_summary(folder, two_axis, cdf, subsets=("all", "untouched")):
    head = {"two_axis": {"auroc_common": two_axis, "auroc_extra": two_axis - 0.05},
            "cdf": {"auroc_common": cdf, "auroc_extra": cdf - 0.02},
            "saod_top3": {"auroc_common": 0.6, "auroc_extra": 0.6}}
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.json").write_text(json.dumps({
        "images": 5000, "clean_map": 0.45, "headline": {s: head for s in subsets}, "headline_decision": "confirmed",
        "by_severity": {"all": {"two_axis": {"common": [two_axis - 0.1] * 5}}}}))


def test_the_detectors_config_gives_one_settings_per_detector(config):
    assert config.detectors == ("yolo11m", "rfdetr_m")
    settings = config.settings("rfdetr_m")
    assert settings.run == config.run / "rfdetr_m" and settings.checkpoint.name == "rfdetr.pth"
    assert settings.gpu_memory_gib == 8.0 and settings.workers == 0


def test_the_check_records_each_clean_ap_and_refuses_a_detector_below_its_floor(config):
    stage.check(config)
    manifest = json.loads(config.settings("yolo11m").layout.manifest.read_text())
    assert manifest["check"]["images"] == 3 and manifest["check"]["coco_val_ap"] == 0.0
    with pytest.raises(RuntimeError, match=r"yolo11m 0\.000 \(floor 0\.5\)"):
        stage.check(replace(config, floors={"yolo11m": 0.5, "rfdetr_m": 0.0}))


def test_the_fit_writes_every_reference_and_records_the_adapters_protocol(config):
    stage.fit(config)
    calls = FakeAdapter.calls
    for name in config.detectors:
        layout = config.settings(name).layout
        assert np.load(layout.knn_bank).shape == (24, 16)
        assert layout.cdf_reference.exists() and layout.cdf_zstats.exists()
        assert layout.hashemi_intervals.exists() == (name == "rfdetr_m")
        with np.load(layout.method_bank) as bank, np.load(layout.method_zstats) as zstats:
            assert set(bank.files) == set(KEYS) and bank["means_s1"].shape == (8, 4)
            assert zstats["top_s4"].shape == (4, 7)
        manifest = json.loads(layout.manifest.read_text())
        assert manifest["protocol"]["detector"] == name and "levels" in manifest["protocol"]["adapter"]
        assert manifest["protocol"]["float32_matmul_precision"] == "highest"
        assert {"ultralytics", "rfdetr"} <= set(manifest["environment"]["packages"])
    stage.fit(config)
    assert FakeAdapter.calls == calls


def test_an_interrupted_fit_redoes_only_the_missing_passes(config):
    stage.fit(config)
    calls = FakeAdapter.calls
    layout = config.settings("yolo11m").layout
    layout.cdf_zstats.unlink()
    stage.fit(config)
    assert FakeAdapter.calls == calls + 2 and layout.cdf_zstats.exists()  # 6 sampled images, 4 per batch


def test_the_shared_pass_writes_three_files_per_detector_and_resumes(config):
    stage.fit(config)
    layouts = {name: config.settings(name).layout for name in config.detectors}
    stage.shared_pass(config, first=1)
    assert [len(list(l.scores("method").glob("*.npz"))) for l in layouts.values()] == [1, 1]
    stage.shared_pass(config)
    calls = FakeAdapter.calls
    for name, layout in layouts.items():
        files = sorted(p.name for p in layout.scores("method").glob("*.npz"))
        assert len(files) == 3
        with np.load(layout.score_file("detector", files[0])) as detector:
            assert ("conf_pos" in detector.files) == (name == "rfdetr_m") and detector["knn"].shape == (96, 5)
        with np.load(layout.score_file("activations", files[0])) as activations:
            assert ("hashemi_decoder" in activations.files) == (name == "rfdetr_m")
            assert activations["cdf_stages"].shape == (96, 5)
        with np.load(layout.score_file("method", files[0])) as method:
            assert set(method.files) == set(KEYS) and method["means_s4"].shape == (96, 7)
        assert set(json.loads(layout.manifest.read_text())["inputs"]) == {"detector", "activations", "method"}
    stage.shared_pass(config)
    assert FakeAdapter.calls == calls


def test_the_pass_refuses_corruptions_that_differ_from_the_reference_run(config):
    stage.fit(config)
    first = config.base.dataset.evaluation_images()[0].name
    atomic_npz(RunLayout(config.reference_run).score_file("detector", first), digests=np.array(["0" * 16] * 96))
    with pytest.raises(ValueError, match="corruptions differ from the reference run"):
        stage.shared_pass(config)


def test_the_pass_refuses_scores_from_a_changed_fit(config):
    stage.fit(config)
    stage.shared_pass(config, first=1)
    layout = config.settings("yolo11m").layout
    zstats = json.loads(layout.cdf_zstats.read_text())
    layout.cdf_zstats.write_text(json.dumps({**zstats, "mean": [m + 1.0 for m in zstats["mean"]]}))
    with pytest.raises(ValueError, match="inputs of scores/activations changed"):
        stage.shared_pass(config)


def test_the_report_stage_links_discopatch_and_writes_the_table(config, tmp_path, monkeypatch):
    reported = []

    def write_report(settings, manifest):
        reported.append((settings.run.name, settings.limit))
        _write_summary(settings.layout.report(), 0.9, 0.8, subsets=("all",))

    monkeypatch.setattr(stage, "write_report", write_report)
    _write_summary(RunLayout(config.reference_run).report(), 0.917, 0.821)
    assert stage.main(["report", "--config", str(tmp_path / "detectors.toml"), "--first", "2"]) == 0
    assert reported == [("yolo11m", 2), ("rfdetr_m", 2)]
    linked = sorted(config.settings("yolo11m").layout.scores("discopatch").glob("*.npz"))
    assert len(linked) == 3 and all(os.stat(p).st_nlink >= 2 for p in linked)
    assert "| yolo11m |" in (config.run / "summary.md").read_text()


def test_first_applies_to_the_pass_and_report_only(config, tmp_path):
    with pytest.raises(SystemExit):
        stage.main(["fit", "--config", str(tmp_path / "detectors.toml"), "--first", "2"])


def test_the_cross_detector_table_lists_every_detector(config):
    _write_summary(RunLayout(config.reference_run).report(), 0.917, 0.821)
    _write_summary(config.settings("yolo11m").layout.report(), 0.9, 0.8)
    _write_summary(config.settings("rfdetr_m").layout.report(), 0.86, 0.79, subsets=("all",))  # a smoke report
    rows = stage.cross_table(config)
    assert [r["detector"] for r in rows] == ["rtdetrv2_r18", "yolo11m", "rfdetr_m"]
    assert rows[1]["all_two_axis_auroc_common"] == 0.9 and rows[1]["best_baseline"] == "cdf"
    assert rows[2]["untouched_two_axis_auroc_common"] is None
    assert "| rfdetr_m |" in (config.run / "summary.md").read_text() and (config.run / "summary.csv").exists()
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/stages/test_detectors.py`
Expected: FAIL with `ImportError: cannot import name 'detectors' from 'degradation_monitor.stages'`.

- [ ] **Step 3: Implement**

In `degradation_monitor/runs.py`, add `"ultralytics"` and `"rfdetr"` to the end of `PACKAGES`, so each detector folder's manifest records their versions.

Create `configs/coco-detectors.toml`:

```toml
# Three more frozen COCO detectors on COCO-C (docs/superpowers/plans/2026-10-04-four-detectors-coco-c.md).
# Stages: python -m degradation_monitor.stages.detectors <check|fit|pass|report> --config configs/coco-detectors.toml
base = "coco.toml"  # the dataset, the seed, the device and the workers
run = "/home/yuchen/YuchenZ/UE/philip_sa/runs/coco-detectors"  # one folder per detector, laid out like runs/coco/
reference_run = "/home/yuchen/YuchenZ/UE/philip_sa/runs/coco"  # RT-DETR's run: the digests and DisCoPatch's scores
gpu_memory_gib = 8.0  # the card is shared

[weights]
yolo11m = "/home/yuchen/YuchenZ/lab/Detector_test/yolo11m.pt"
faster_rcnn_r50_fpn_v2 = "/home/yuchen/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
rfdetr_m = "/home/yuchen/.roboflow/models/rf-detr-medium.pth"

[clean_ap_floor]  # the published COCO val AP minus 0.03
yolo11m = 0.485
faster_rcnn_r50_fpn_v2 = 0.437
rfdetr_m = 0.517
```

Create `degradation_monitor/stages/detectors.py`:

```python
"""Three more frozen COCO detectors on COCO-C: their clean fits, one shared pass, their reports and a table.

    python -m degradation_monitor.stages.detectors <stage> [--config configs/coco-detectors.toml] [--first N]

Stages, each over every detector of the config:
- check: the detector's clean AP on every val image (COCO's AP@[.5:.95]) must reach its floor;
- fit: the clean references, in three passes over COCO train images, each skipped when its files exist: (1) the kNN
  bank, the CDF ranges, the method's bank and z-statistics and, for DETR-type detectors, Hashemi's decoder intervals;
  (2) the CDF histograms; (3) the CDFs' z-statistics;
- pass: every val image's 96 versions, generated once, through every detector: a detector, an activations and a
  method file per image and detector; --first N stops after N images, to time the pass;
- report: each detector's report (reports/coco/ in its folder), then the table of all four detectors in summary.md;
  --first N reports on the first N evaluation images only, a smoke test that the full report overwrites.

Each detector's folder (<run>/<detector>/) is laid out like runs/coco/. Its manifest's protocol adds the detector's
name and its adapter's protocol (taps, input size, thresholds) to RT-DETR's, with the detector's weights. DisCoPatch
reads the image, not the detector, so its scores are hard links to the reference run's. Every corrupted image must
match the reference run's digest.
"""
from __future__ import annotations

import argparse
import csv
import errno
import json
import os
import shutil
import sys
import time
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .. import corruptions
from ..baselines.activation_cdf import BINS as CDF_BINS
from ..baselines.activation_cdf import MARGIN as CDF_MARGIN
from ..baselines.activation_cdf import STAGES as CDF_STAGES
from ..baselines.activation_cdf import ZSTAT_IMAGES as CDF_ZSTAT_IMAGES
from ..baselines.activation_cdf import CdfMonitor, ChannelRanges, ReferenceHistograms
from ..baselines.contrastive_conf import THETA, contrastive_parts, query_detections
from ..baselines.hashemi import K as HASHEMI_K
from ..baselines.hashemi import HashemiMonitor, NeuronStats, save_intervals
from ..baselines.knn import KNN_K_MAX, knn_distances, normalize_rows
from ..baselines.saod import saod_uncertainty
from ..cli import positive_int
from ..datasets.coco import coco_map, coco_results, list_images
from ..detectors import DETECTORS, LEVELS, adapter_class, load_adapter
from ..evaluation.metrics import stage_zstats, zscored_sum
from ..evaluation.report import OURS
from ..method.statistics import KEYS, channel_statistics
from ..runs import Manifest, RunLayout, atomic_json, atomic_npz, load_npz, progress, sha1
from ..settings import load_settings
from .common import cap_gpu_memory, image_size, variant_stream
from .report import write_report

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "coco-detectors.toml"
CONFIG_KEYS = {"base", "run", "reference_run", "gpu_memory_gib", "weights", "clean_ap_floor"}
FOLDERS = ("detector", "activations", "method")
PROGRESS_IMAGES = 2000  # the fit reports its progress about every this many images


@dataclass(frozen=True)
class DetectorsConfig:
    base: object  # the base config's Settings: the dataset, the seed, the device and the workers
    run: Path
    reference_run: Path  # RT-DETR's run: the corruption digests and DisCoPatch's scores
    weights: dict
    floors: dict
    gpu_memory_gib: float

    @property
    def detectors(self) -> tuple:
        return tuple(self.weights)

    def settings(self, name: str):
        """One detector's Settings: its run folder, and its weights as the protocol's checkpoint."""
        return replace(self.base, run=self.run / name, checkpoint=Path(self.weights[name]),
                       gpu_memory_gib=self.gpu_memory_gib)


def load_config(path, run=None) -> DetectorsConfig:
    path = Path(path)
    values = tomllib.loads(path.read_text())
    unknown = sorted(set(values) - CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unknown settings in {path}: {', '.join(unknown)}")
    weights = {name: Path(value) for name, value in values["weights"].items()}
    strange = sorted(set(weights) - set(DETECTORS))
    if strange:
        raise ValueError(f"unknown detectors in {path}: {', '.join(strange)}; choose from {', '.join(DETECTORS)}")
    if set(values["clean_ap_floor"]) != set(weights):
        raise ValueError(f"{path}: every detector needs a clean AP floor, and only those")
    return DetectorsConfig(base=load_settings(path.parent / values["base"]), run=Path(run or values["run"]),
                           reference_run=Path(values["reference_run"]), weights=weights,
                           floors=dict(values["clean_ap_floor"]), gpu_memory_gib=float(values["gpu_memory_gib"]))


def _manifest(config, name) -> Manifest:
    """The detector folder's manifest, once its protocol is checked: the weights, the evaluation and the adapter's."""
    settings = config.settings(name)
    settings.layout.root.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(settings.layout)
    manifest.check_protocol({**settings.protocol(), "detector": name, "adapter": adapter_class(name).protocol,
                             "float32_matmul_precision": torch.get_float32_matmul_precision()})
    manifest.record_environment(settings.discopatch_root)
    return manifest


class _RgbImages(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as source:
            return np.asarray(source.convert("RGB"), dtype=np.uint8).copy()


def _batches(paths, size, workers):
    """Lists of `size` RGB arrays, in the order of `paths`."""
    return DataLoader(_RgbImages(paths), batch_size=size, num_workers=workers, collate_fn=list)


def _method_statistics(out) -> dict:
    """Each channel's level and top-1% mean in the four levels: KEYS -> float32 (N, C)."""
    return {f"{statistic}_{level}": values for level in LEVELS
            for statistic, values in channel_statistics(out.levels[level]).items()}


def _report_peak_memory(label, device) -> None:
    """The process's peak GPU memory so far, for the sessions that share the card."""
    if torch.device(device).type == "cuda":
        print(f"[{label}] peak GPU memory {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB", flush=True)


def check(config) -> None:
    """Every detector's clean AP on every COCO val image; refuses a detector below its floor."""
    failed = []
    for name in config.detectors:
        settings, manifest = config.settings(name), _manifest(config, name)
        gt = settings.dataset.ground_truth()
        images = list_images(settings.val_images)
        cap_gpu_memory(settings.device, settings.gpu_memory_gib)
        adapter = load_adapter(name, config.weights[name], settings.device)
        results = []
        for path, arrays in zip(images, _batches(images, 1, settings.workers)):  # val images differ in size
            out = adapter(arrays)
            results += coco_results(gt.image_id(path.name), out.scores[0], out.labels[0], out.boxes[0],
                                    gt.category_ids)  # an empty slot scores 0 with an empty box: it never matches
        adapter.close()
        ap = coco_map(gt, results, [gt.image_id(p.name) for p in images])
        manifest.update(check={"coco_val_ap": ap, "images": len(images), "floor": config.floors[name]})
        print(f"[check] {name}: clean COCO val AP = {ap:.4f} on {len(images)} images", flush=True)
        if ap < config.floors[name]:
            failed.append(f"{name} {ap:.3f} (floor {config.floors[name]})")
    if failed:
        raise RuntimeError("clean COCO val AP below the floor: " + "; ".join(failed))


def _ranges_path(layout) -> Path:
    return layout.reference("activation_cdf", "ranges.npz")


def _clean_pass_done(layout, detr: bool) -> bool:
    paths = [layout.knn_bank, layout.method_bank, layout.method_zstats, _ranges_path(layout)]
    return all(path.exists() for path in paths + ([layout.hashemi_intervals] if detr else []))


def _fitted(layout, detr: bool) -> bool:
    return _clean_pass_done(layout, detr) and layout.cdf_reference.exists() and layout.cdf_zstats.exists()


def _clean_pass(adapter, settings) -> None:
    """Pass 1 over every clean train image: the kNN bank, the CDF ranges, the method's bank and z-statistics, and for
    DETR-type detectors Hashemi's decoder intervals. The ranges are written last: they mark the pass as done."""
    layout, dataset = settings.layout, settings.dataset
    paths = dataset.train_images()
    splits = {split: [str(p) for p in dataset.reference_split(split)] for split in ("bank", "zstats")}
    wanted = {p for members in splits.values() for p in members}
    pooled, ranges, method = [], [ChannelRanges() for _ in CDF_STAGES], {}
    decoder = NeuronStats() if adapter.detr else None
    size, started = adapter.fit_batch_size, time.time()
    for index, arrays in enumerate(_batches(paths, size, settings.workers)):
        names = [str(p) for p in paths[index * size:(index + 1) * size]]
        out = adapter(arrays, heads=adapter.detr)  # only Hashemi reads past the backbone
        pooled.append(normalize_rows(out.pooled).cpu().numpy().astype(np.float16))
        for kept, maps in zip(ranges, out.cdf):
            kept.update(maps)
        if decoder is not None:
            decoder.update(out.decoder)
        if wanted.intersection(names):
            statistics = _method_statistics(out)
            method.update({p: {k: v[row] for k, v in statistics.items()} for row, p in enumerate(names) if p in wanted})
        if index % max(1, PROGRESS_IMAGES // size) == 0:
            progress(f"fit {adapter.name} 1/3", min((index + 1) * size, len(paths)), len(paths), started)
    layout.knn_bank.parent.mkdir(parents=True, exist_ok=True)
    temporary = layout.knn_bank.with_name(".bank.tmp.npy")
    np.save(temporary, np.concatenate(pooled))
    temporary.replace(layout.knn_bank)
    atomic_json(layout.knn_names, [p.name for p in paths])
    for split, path in (("bank", layout.method_bank), ("zstats", layout.method_zstats)):
        atomic_npz(path, **{key: np.stack([method[p][key] for p in splits[split]]) for key in KEYS})
    if decoder is not None:
        save_intervals(layout.hashemi_intervals, {"decoder": decoder}, images=len(paths))
        atomic_json(layout.hashemi_fit, {"images": len(paths), "k": HASHEMI_K, "std": "population (ddof=0)",
                                         "layers": ["decoder"]})
    bounds = {stage: kept.result() for stage, kept in zip(CDF_STAGES, ranges)}
    atomic_npz(_ranges_path(layout), **{f"{stage}_{bound}": values for stage, pair in bounds.items()
                                        for bound, values in zip(("low", "high"), pair)})


def _cdf_histograms(adapter, settings) -> None:
    """Pass 2: every clean train image's CDF maps, counted into histograms on the ranges of pass 1."""
    layout = settings.layout
    paths = settings.dataset.train_images()
    with np.load(_ranges_path(layout)) as data:
        bounds = {stage: (data[f"{stage}_low"], data[f"{stage}_high"]) for stage in CDF_STAGES}
    reference = ReferenceHistograms(bounds, settings.device)
    size, started = adapter.fit_batch_size, time.time()
    for index, arrays in enumerate(_batches(paths, size, settings.workers)):
        reference.update(adapter(arrays, heads=False).cdf)
        if index % max(1, PROGRESS_IMAGES // size) == 0:
            progress(f"fit {adapter.name} 2/3", min((index + 1) * size, len(paths)), len(paths), started)
    reference.save(layout.cdf_reference, images=len(paths))
    atomic_json(layout.cdf_fit, {"images": len(paths), "bins": CDF_BINS, "margin": CDF_MARGIN,
                                 "stages": list(CDF_STAGES), "passes": 2})


def _cdf_zstats(adapter, settings) -> None:
    """Pass 3: each CDF stage's mean and spread over a seeded sample of clean train images."""
    layout = settings.layout
    paths = settings.dataset.train_images()
    chosen = np.sort(np.random.default_rng(settings.seed).choice(len(paths), size=min(CDF_ZSTAT_IMAGES, len(paths)),
                                                                replace=False))
    monitor, values = CdfMonitor(layout.cdf_reference, settings.device), []
    for arrays in _batches([paths[i] for i in chosen], adapter.fit_batch_size, settings.workers):
        values.append(monitor.stage_scores(adapter(arrays, heads=False).cdf))
    mean, std = stage_zstats(np.concatenate(values))
    atomic_json(layout.cdf_zstats, {"images": len(chosen), "seed": settings.seed, "stages": list(CDF_STAGES),
                                    "mean": mean.tolist(), "std": std.tolist(),
                                    "reference_sha1": sha1(layout.cdf_reference)})


def fit(config) -> None:
    """Every detector's clean references, from the first pass whose files are missing onward."""
    for name in config.detectors:
        settings = config.settings(name)
        _manifest(config, name)
        layout, detr = settings.layout, adapter_class(name).detr
        passes = ((_clean_pass, _clean_pass_done(layout, detr)), (_cdf_histograms, layout.cdf_reference.exists()),
                  (_cdf_zstats, layout.cdf_zstats.exists()))
        start = next((i for i, (_, done) in enumerate(passes) if not done), None)
        if start is None:
            continue
        cap_gpu_memory(settings.device, settings.gpu_memory_gib)
        adapter = load_adapter(name, config.weights[name], settings.device)
        for step, _ in passes[start:]:  # a redone pass makes every later pass stale
            step(adapter, settings)
        adapter.close()
        _report_peak_memory(f"fit {name}", settings.device)


def _references(config, name) -> dict:
    settings = config.settings(name)
    layout, device = settings.layout, settings.device
    bank = normalize_rows(torch.from_numpy(np.load(layout.knn_bank)).float()).to(device).half()
    return {"knn_bank": bank, "cdf": CdfMonitor(layout.cdf_reference, device),
            "cdf_zstats": json.loads(layout.cdf_zstats.read_text()),
            "hashemi": HashemiMonitor(layout.hashemi_intervals, device) if adapter_class(name).detr else None}


def _inputs(layout, detr: bool) -> dict:
    """What each score folder is computed from: the manifest refuses a pass after a fit changed."""
    activations = {"cdf_reference": sha1(layout.cdf_reference), "cdf_zstats": sha1(layout.cdf_zstats),
                   "cdf_bins": CDF_BINS}
    if detr:
        activations.update(hashemi_intervals=sha1(layout.hashemi_intervals), hashemi_k=HASHEMI_K)
    return {"detector": {"knn_bank": sha1(layout.knn_bank)}, "activations": activations,
            "method": {"method_bank": sha1(layout.method_bank), "method_zstats": sha1(layout.method_zstats)}}


def image_files(adapter, references, arrays) -> dict:
    """The detector, activations and method arrays of one image's versions, from one forward pass per batch."""
    size = image_size(arrays[0])
    parts = {folder: [] for folder in FOLDERS}
    for start in range(0, len(arrays), adapter.batch_size):
        out = adapter(arrays[start:start + adapter.batch_size])
        detector = {"saod_min": np.array([saod_uncertainty(s, 1) for s in out.scores]),
                    "saod_top3": np.array([saod_uncertainty(s, 3) for s in out.scores]),
                    "knn": knn_distances(out.pooled.to(references["knn_bank"].device), references["knn_bank"],
                                         KNN_K_MAX).cpu().numpy().astype(np.float32),
                    "det_scores": out.scores, "det_labels": out.labels.astype(np.int16), "det_boxes": out.boxes}
        if out.query_logits is not None:
            detector["conf_pos"], detector["conf_neg"] = contrastive_parts(
                [query_detections(l, b, size) for l, b in zip(out.query_logits, out.query_boxes)], THETA)
        stage_scores, zstats = references["cdf"].stage_scores(out.cdf), references["cdf_zstats"]
        activations = {"cdf_backbone": stage_scores.sum(axis=1),
                       "cdf_backbone_z": zscored_sum(stage_scores, zstats["mean"], zstats["std"]),
                       "cdf_stages": stage_scores}
        if references["hashemi"] is not None:
            activations["hashemi_decoder"] = references["hashemi"].decoder_share(out.decoder)
        for folder, values in zip(FOLDERS, (detector, activations, _method_statistics(out))):
            parts[folder].append(values)
        del out  # its maps are views into the full feature maps: free them before the next forward pass
    files = {folder: {key: np.concatenate([p[key] for p in pieces]) for key in pieces[0]}
             for folder, pieces in parts.items()}
    for folder, values in files.items():
        if not all(np.isfinite(v).all() for v in values.values() if v.dtype.kind == "f"):
            raise ValueError(f"{adapter.name} gave non-finite {folder} values")
    files["detector"]["size"] = np.array(size)
    return files


def _complete(layout, image) -> bool:
    return all(layout.score_file(folder, image).exists() for folder in FOLDERS)


def shared_pass(config, first=None) -> None:
    """Every evaluation image's 96 versions, generated once, through every detector; resumable image by image."""
    names = config.detectors
    layouts = {name: config.settings(name).layout for name in names}
    for name in names:
        manifest, detr = _manifest(config, name), adapter_class(name).detr
        if not _fitted(layouts[name], detr):
            raise ValueError(f"run the fit stage first: {name}'s references are missing")
        for folder, inputs in _inputs(layouts[name], detr).items():
            manifest.check_inputs(folder, inputs)
    base = config.base
    pending = [p for p in base.dataset.evaluation_images() if not all(_complete(layouts[n], p) for n in names)]
    pending = pending[:first] if first else pending
    if not pending:
        return
    cap_gpu_memory(base.device, config.gpu_memory_gib)
    adapters = {name: load_adapter(name, config.weights[name], base.device) for name in names}
    references = {name: _references(config, name) for name in names}
    reference_run, started = RunLayout(config.reference_run), time.time()
    for done, (image, arrays) in enumerate(variant_stream(base, pending), start=1):
        digests = np.array([corruptions.digest(a) for a in arrays])
        if list(load_npz(reference_run.score_file("detector", image), ("digests",))["digests"]) != list(digests):
            raise ValueError(f"corruptions differ from the reference run for {image}")
        for name, adapter in adapters.items():
            if not _complete(layouts[name], image):
                files = image_files(adapter, references[name], arrays)
                files["detector"]["digests"] = digests
                for folder in FOLDERS:
                    atomic_npz(layouts[name].score_file(folder, image), **files[folder])
        if done % 25 == 0:
            progress("detectors pass", done, len(pending), started)
    for adapter in adapters.values():
        adapter.close()
    _report_peak_memory("pass", base.device)


def _link_discopatch(config, layout, manifest) -> None:
    """DisCoPatch's scores do not depend on the detector: hard links to the reference run's files."""
    reference = RunLayout(config.reference_run)
    inputs = Manifest(reference).read().get("inputs", {}).get("discopatch")
    if inputs is None:
        raise ValueError(f"the reference run records no DisCoPatch inputs: {reference.manifest}")
    manifest.check_inputs("discopatch", {**inputs, "reference_run": str(config.reference_run)})
    target = layout.scores("discopatch")
    target.mkdir(parents=True, exist_ok=True)
    for path in reference.scores("discopatch").glob("*.npz"):
        if not (target / path.name).exists():
            try:
                os.link(path, target / path.name)
            except OSError as error:  # a run root on another filesystem: copy instead
                if error.errno != errno.EXDEV:
                    raise
                shutil.copy2(path, target / path.name)


def _auroc(summary, subset, row, group):
    """A row's AUROC on one image set; None when a smoke report has no such set."""
    return summary["headline"].get(subset, {}).get(row, {}).get(f"auroc_{group}")


def _cell(value) -> str:
    return "–" if value is None else f"{value:.3f}"


def cross_table(config) -> list:
    """RT-DETR's and every detector's headline: the two-axis score, the CDFs, the best baseline and the clean mAP."""
    sources = {"rtdetrv2_r18": RunLayout(config.reference_run).report() / "summary.json"}
    sources.update({name: config.settings(name).layout.report() / "summary.json" for name in config.detectors})
    rows = []
    for name, path in sources.items():
        summary = json.loads(path.read_text())
        row = {"detector": name, "images": summary["images"], "clean_map": summary["clean_map"],
               "headline_decision": summary["headline_decision"]}
        for subset in ("all", "untouched"):
            for method in ("two_axis", "cdf"):
                for group in ("common", "extra"):
                    row[f"{subset}_{method}_auroc_{group}"] = _auroc(summary, subset, method, group)
        row["two_axis_severity1_common"] = summary["by_severity"]["all"]["two_axis"]["common"][0]
        head = summary["headline"]["all"]
        best = max((m for m in head if m not in OURS), key=lambda m: head[m]["auroc_common"])
        row["best_baseline"], row["best_baseline_auroc_common"] = best, head[best]["auroc_common"]
        rows.append(row)
    config.run.mkdir(parents=True, exist_ok=True)
    with (config.run / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# The two-axis score on four COCO detectors", "",
             "AUROC, common / extra families. Untouched: positions 1970 and later.", "",
             "| Detector | Images | Clean mAP | Two-axis, all | Two-axis, untouched | Two-axis, severity 1 common "
             "| CDFs, all | Best baseline, common | Headline |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['detector']} | {r['images']} | {_cell(r['clean_map'])} "
            f"| {_cell(r['all_two_axis_auroc_common'])} / {_cell(r['all_two_axis_auroc_extra'])} "
            f"| {_cell(r['untouched_two_axis_auroc_common'])} / {_cell(r['untouched_two_axis_auroc_extra'])} "
            f"| {_cell(r['two_axis_severity1_common'])} "
            f"| {_cell(r['all_cdf_auroc_common'])} / {_cell(r['all_cdf_auroc_extra'])} "
            f"| {r['best_baseline']} {_cell(r['best_baseline_auroc_common'])} | {r['headline_decision']} |")
    (config.run / "summary.md").write_text("\n".join(lines) + "\n")
    return rows


def report(config, first=None) -> None:
    """Each detector's report, then the table of all four; with `first`, a smoke report on the first images."""
    for name in config.detectors:
        settings, manifest = config.settings(name), _manifest(config, name)
        _link_discopatch(config, settings.layout, manifest)
        write_report(replace(settings, limit=first) if first else settings, manifest)
    cross_table(config)


STAGES = {"check": check, "fit": fit, "pass": shared_pass, "report": report}
WITH_FIRST = ("pass", "report")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m degradation_monitor.stages.detectors",
                                     description="Three more frozen COCO detectors on COCO-C: run one stage.")
    parser.add_argument("stage", choices=list(STAGES))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run", type=Path, help="the run root, instead of the config's")
    parser.add_argument("--first", type=positive_int,
                        help="pass: stop after N images; report: a smoke report on the first N evaluation images")
    args = parser.parse_args(argv)
    if args.first and args.stage not in WITH_FIRST:
        parser.error("--first applies to the pass and report stages only")
    try:
        config = load_config(args.config, run=args.run)
        STAGES[args.stage](config, **({"first": args.first} if args.stage in WITH_FIRST else {}))
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

In `README.md`:
- in "Layout", add the lines `detectors/                 one adapter per further detector: YOLO11m, Faster R-CNN, RF-DETR-M` and `configs/coco-detectors.toml  the further detectors' weights, run root and clean-AP floors`;
- at the end of "Running", add:

```markdown
### Three more detectors

`python -m degradation_monitor.stages.detectors <check|fit|pass|report> --config configs/coco-detectors.toml` runs
YOLO11m, Faster R-CNN R50-FPN v2 and RF-DETR-M on the same COCO-C images as RT-DETR. `fit` builds each detector's
clean references from COCO train images in three resumable passes. `pass` generates each val image's 96 versions
once, checks them against `runs/coco/`'s digests and feeds all three detectors; `--first N` stops after N images.
`report` writes each detector's report and `runs/coco-detectors/summary.md`, the table of all four; `--first N` is a
smoke report on the first N images. Each detector reads the method's three earliest feature levels and its deepest as
the key: stages for the CNNs, blocks 1-3 and 12 for RF-DETR's ViT. ContrastiveConf and Hashemi et al. exist for the
DETR-type detectors only.
```

- [ ] **Step 4: Run the tests to see them pass, then the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/stages/test_detectors.py`
Expected: PASS.

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: `182 passed, 1 skipped`.

- [ ] **Step 5: Commit**

```bash
git add degradation_monitor/stages/detectors.py configs/coco-detectors.toml degradation_monitor/runs.py README.md \
        tests/stages/test_detectors.py
git commit -m "feat: three more detectors on COCO-C: their fits, one shared pass, reports and a table

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

---

### Task 7: Check each detector's clean AP (GPU, about 15 minutes)

- [ ] **Step 1: Ask explore for the GPU** (up to 8 GiB, about 15 minutes), and wait for its answer.

- [ ] **Step 2: Run the check**

```bash
cd /home/yuchen/YuchenZ/UE/philip_sa/.worktrees/detectors
/home/yuchen/miniconda3/envs/UE/bin/python -m degradation_monitor.stages.detectors check --config configs/coco-detectors.toml
```

Expected: three `[check]` lines, each AP at or above its floor. The published values are about 0.515, 0.467 and 0.547. The 100-detection cap can cost a few thousandths, and YOLO11m's single-label NMS a little more: Ultralytics' validation uses multi-label NMS and 300 detections.

- [ ] **Step 3: Send explore "done", and record the three APs in the ledger.**

---

### Task 8: Fit every detector's clean references (GPU, about 2 hours)

- [ ] **Step 1: Ask explore for the GPU,** with this estimate. Each detector reads the 118,287 train images twice and 5,000 once. YOLO11m and Faster R-CNN take one image at a time; the passes run the backbone only, except RF-DETR's first pass, which needs its decoder.

- [ ] **Step 2: Run the fit in the background with a log** (it resumes from the first missing pass):

```bash
cd /home/yuchen/YuchenZ/UE/philip_sa/.worktrees/detectors
/home/yuchen/miniconda3/envs/UE/bin/python -m degradation_monitor.stages.detectors fit --config configs/coco-detectors.toml \
  > /home/yuchen/YuchenZ/UE/philip_sa/runs/coco-detectors-fit.log 2>&1
```

Expected:
- each `runs/coco-detectors/<detector>/reference/` holds the kNN bank (118,287 rows), the CDF ranges, reference, fit and z-statistics, and the method's bank (2,000) and z-statistics (500);
- `rfdetr_m` also holds Hashemi's intervals.

- [ ] **Step 3: Send explore "done".**

---

### Task 9: A smoke pass and report on 25 images (GPU, a few minutes)

- [ ] **Step 1: Ask explore for the GPU** for a few minutes.

- [ ] **Step 2: Run the pass on the first 25 images, then a smoke report on them**

These files are valid results under the full run's protocol, so the full pass keeps them. The smoke report is overwritten by Task 11's.

```bash
cd /home/yuchen/YuchenZ/UE/philip_sa/.worktrees/detectors
/home/yuchen/miniconda3/envs/UE/bin/python -m degradation_monitor.stages.detectors pass --config configs/coco-detectors.toml --first 25
/home/yuchen/miniconda3/envs/UE/bin/python -m degradation_monitor.stages.detectors report --config configs/coco-detectors.toml --first 25
```

Expected:
- each detector folder holds 25 files in each of `scores/detector`, `scores/activations` and `scores/method`;
- `reports/coco/report.md` is written in each;
- `runs/coco-detectors/summary.md` lists four detectors, with "–" for the untouched images.

The 25-image numbers mean nothing. Check that each detector's clean mAP in its `report.md` is plausible. Note the pass's time per image and its `peak GPU memory` line to plan Task 10. Position 19 is one of the largest Faster R-CNN inputs (1344 × 800), so the smoke run reaches the pass's peak memory.

- [ ] **Step 3: Send explore "done".**

---

### Task 10: The full shared pass (GPU, about 7–8 hours)

- [ ] **Step 1: Ask explore for the GPU,** with the estimate from Task 9.

- [ ] **Step 2: Run the pass in the background** (it resumes after an interruption):

```bash
cd /home/yuchen/YuchenZ/UE/philip_sa/.worktrees/detectors
/home/yuchen/miniconda3/envs/UE/bin/python -m degradation_monitor.stages.detectors pass --config configs/coco-detectors.toml \
  > /home/yuchen/YuchenZ/UE/philip_sa/runs/coco-detectors-pass.log 2>&1
```

Expected: 5,000 files in each of the nine score folders.

- [ ] **Step 3: Send explore "done".**

---

### Task 11: Reports, the table, and the results

**Files:**
- Create: `docs/results/coco-detectors/` (each detector's `reports/coco/` files, `summary.md`, `summary.csv` and `arms.csv`)
- Create: `scripts/paper/detector_arms.py`, `docs/coco-detectors-results.md`
- Modify: `docs/README.md`, `docs/paper-storyline.md`

- [ ] **Step 1: Write the reports and the table** (CPU, up to about 50 GiB of RAM)

The report holds every image's channel statistics in float64. Faster R-CNN's 3,840 channels need about 50 GiB at the peak. That compares with about 14 GiB for RT-DETR and about 20 GiB each for YOLO11m and RF-DETR-M. The detectors are reported one after another, so the peak is Faster R-CNN's. Check `free -g` first; at least 60 GiB should be available. The machine is shared, so tell explore before starting.

```bash
cd /home/yuchen/YuchenZ/UE/philip_sa/.worktrees/detectors
/home/yuchen/miniconda3/envs/UE/bin/python -m degradation_monitor.stages.detectors report --config configs/coco-detectors.toml
```

Expected:
- each detector's `summary.json` covers 5,000 images and gives both decisions;
- `/home/yuchen/YuchenZ/UE/philip_sa/runs/coco-detectors/summary.md` lists the four detectors.

- [ ] **Step 2: Keep the tables**

```bash
RUNS=/home/yuchen/YuchenZ/UE/philip_sa/runs/coco-detectors
mkdir -p docs/results/coco-detectors
cp $RUNS/summary.md $RUNS/summary.csv docs/results/coco-detectors/
for d in yolo11m faster_rcnn_r50_fpn_v2 rfdetr_m; do
  mkdir -p docs/results/coco-detectors/$d && cp $RUNS/$d/reports/coco/* docs/results/coco-detectors/$d/
done
```

- [ ] **Step 3: Each detector's two arms per condition**

The report has a row for the level score, whose AUROC equals the level arm's, but none for the signed flattening arm. `peak_share` is a different, unsigned statistic. Create `scripts/paper/detector_arms.py`:

```python
"""Each run's two-axis score and its two arms per corruption condition (AUROC): which arm catches which corruption.

    python scripts/paper/detector_arms.py RUN [RUN ...] --out docs/results/coco-detectors/arms.csv

Each RUN is a run folder laid out like runs/coco/: scores/method/ and reference/method/. The evaluation images are
configs/coco.toml's. The arms come from the package's own two_axis_scores, so they are exactly the two halves of the
reported two-axis score. Runs are read one after another; Faster R-CNN's needs up to about 50 GiB of RAM.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from degradation_monitor.corruptions import CONDITIONS  # noqa: E402
from degradation_monitor.evaluation.metrics import condition_aurocs  # noqa: E402
from degradation_monitor.method.scores import two_axis_scores  # noqa: E402
from degradation_monitor.method.statistics import KEYS  # noqa: E402
from degradation_monitor.runs import RunLayout, stack  # noqa: E402
from degradation_monitor.settings import load_settings  # noqa: E402


def arm_rows(run: Path, names: list) -> list:
    """Per corrupted condition, the AUROC of the two-axis score and of its flattening and level arms."""
    layout = RunLayout(run)
    statistics = stack(layout.scores("method"), names, KEYS)
    with np.load(layout.method_bank) as bank, np.load(layout.method_zstats) as zstats:
        bank, zstats = {k: bank[k] for k in KEYS}, {k: zstats[k] for k in KEYS}
    two_axis, arms = two_axis_scores(statistics, bank, zstats)
    rows = []
    for arm, values in (("two_axis", two_axis), ("flatter", arms["flatter"]), ("level", arms["level"])):
        aurocs = condition_aurocs(values[:, 0], values[:, 1:].T)
        rows += [{"run": run.name, "arm": arm, "family": family, "severity": int(severity), "auroc": float(value)}
                 for (family, severity), value in zip(CONDITIONS[1:], aurocs)]
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Each run's two-axis score and its two arms per condition.")
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    names = [p.name for p in load_settings(ROOT / "configs" / "coco.toml").dataset.evaluation_images()]
    rows = [row for run in args.runs for row in arm_rows(run, names)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
```

Run it on the four detectors, one after another:

```bash
R=/home/yuchen/YuchenZ/UE/philip_sa/runs
/home/yuchen/miniconda3/envs/UE/bin/python scripts/paper/detector_arms.py $R/coco $R/coco-detectors/yolo11m \
  $R/coco-detectors/faster_rcnn_r50_fpn_v2 $R/coco-detectors/rfdetr_m --out docs/results/coco-detectors/arms.csv
```

Expected:
- `wrote 1140 rows`: 4 runs × 3 rows (the two-axis score and its two arms) × 95 conditions; the rows labelled `coco` are RT-DETR's.
- Each run's `two_axis` rows equal the two-axis AUROCs on all images in its report's `separation.csv`; check one detector.
- RT-DETR's rows reproduce the storyline's C5: fog through the flattening arm, noise through the level arm.

- [ ] **Step 4: Write `docs/coco-detectors-results.md`**

Include:
- the four detectors and the tap rule, with YOLO's exploration round;
- the headline table: each detector's rows on all and on the untouched images, with intervals;
- each detector's pre-registered outcome, quoted from its `summary.json`;
- AUROC by severity;
- the families at severities 1, 3 and 5;
- the clean mAP;
- what each detector lacks: ContrastiveConf and Hashemi for the CNNs;
- for each detector, which arm catches fog and which catches noise, from `arms.csv`. The ViT's arms swapped roles on the development images.

Add the file and the folder to `docs/README.md`. In `docs/paper-storyline.md`:
- add each detector's fog and noise arms to C5 ("Fog flattens the channels; noise re-weights them");
- add the fixed choices carried unchanged to three more detectors to C7 ("The result is not tuned to the test images");
- add a "several detectors" item to the results part of the section-by-section outline.

- [ ] **Step 5: Commit**

```bash
git add docs/results/coco-detectors docs/coco-detectors-results.md docs/README.md docs/paper-storyline.md \
        scripts/paper/detector_arms.py
git commit -m "results: the two-axis score on four COCO detectors

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```
