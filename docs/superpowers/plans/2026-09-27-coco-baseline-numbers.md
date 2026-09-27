# COCO Baseline Numbers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute separation and harm numbers for the four kept baselines on all 5,000 COCO val images,
under the updated protocol:
- SAOD image-level uncertainty;
- ContrastiveConf;
- kNN on one pooled detector feature;
- DisCoPatch.

Our own method is out of scope for this run.

**Architecture:**
- **Code.** A new subpackage `differential_uncertainty/baselines/` holds focused modules: protocol, scores, detector tap, COCO detection quality, metrics, DisCoPatch adapter, pipeline and report. A new CLI subcommand `baselines-coco --phase …` runs them.
- **Phases.** Each phase is resumable and writes its results under `runs/coco-baselines/`.
- **Corruptions.** They are regenerated deterministically from `(image name, family, severity)`, so every phase sees bit-identical corrupted images without storing them.
- **Cross-fitting.** The only fitted quantity, ContrastiveConf's λ, is cross-fitted over five fixed folds of the evaluation images, so no image is ever scored with a λ fitted on itself.

**Tech Stack:** Python 3.11 (conda env `UE`), PyTorch 2.11 + CUDA (RTX 5090), `imagecorruptions` 1.1.2, `uq-detr` 0.1.1 (installed), `pycocotools`, `scipy`, `scikit-learn`, official DisCoPatch code at commit `ec36b85`.

**Spec:** `docs/driving-benchmark-baselines-and-metrics.md` (kept baselines, fairness rules), plus the run
decisions below, agreed with the user on 2026-09-27. A second review pass on the same day changed the
following:
- evaluate on all 5,000 val images;
- cross-fit λ;
- use AUPR instead of balanced accuracy;
- measure harm within each condition;
- add a paired bootstrap for every headline number;
- add runtime.

## Run decisions (the spec for this run)

1. **Setting:** COCO. Detector: frozen RT-DETRv2-R18, checkpoint
   `/home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth`.
2. **Evaluation images:**
   - All 5,000 COCO val2017 images, in the old benchmark's seeded shuffle (seed 44).
   - The image at shuffled position *i* belongs to fold *i* mod 5, giving five folds of 1,000.
   - No separate tuning split.
3. **Corruptions:**
   - All 19 `imagecorruptions` families, including the package's own Gaussian blur, at severities 1–5.
   - Clean + 95 corrupted versions per image.
   - Reported separately for the 15 common and the 4 extra families.
4. **Clean training data for baselines:** all 118,287 COCO train2017 images. They are used for the kNN
   bank and for DisCoPatch training.
5. **SAOD:**
   - RT-DETR post-processing: sigmoid over the 300 × 80 scores, then the top-100 (query, class) pairs.
   - Image uncertainty is the mean of (1 − p) over the *m* most confident, reported for m = 3 (SAOD's
     best) and m = 1.
   - Nothing is fitted.
6. **ContrastiveConf:**
   - `uq-detr`'s threshold split at θ = 0.3.
   - λ is cross-fitted: for fold *f*, λ is chosen on the clean images of the other four folds. The
     grid and objective are exactly those of `uq_detr.fit_lambda`: the first λ in its grid with the
     highest Pearson correlation with per-image COCO AP. Our re-implementation is tested for equality
     with `uq_detr.fit_lambda`.
   - If all five folds choose the same λ, the metrics are pooled over the 5,000 images. If they
     differ, the per-condition table averages over folds, and the report says so.
   - Degradation score = −ContrastiveConf.
7. **kNN (Sun et al. 2022):**
   - The last backbone stage (ResNet-18 layer4, 512-d), global-average-pooled and L2-normalised.
   - The score is the Euclidean distance to the k-th nearest bank row, with k = 100 fixed in advance
     (≈0.1% of the bank, as in Sun et al.).
   - The 200 nearest distances are stored, for a sensitivity table over k ∈ {1, 10, 50, 100, 200}.
8. **DisCoPatch:**
   - Training: the official code and the README's replication hyperparameters (batch 67 images,
     48 patches, hidden dims 128 256 512 1024, latent 1024, learning rate 8.5e-5, generator and
     reconstruction weights 1e-3, KL weight 1e-4, MSE loss).
   - Budget: 65 COCO epochs ≈ 7.7M image samples. This matches the paper's patch model: 30 epochs of
     one fifth of ImageNet each, the fifth being an ImageNet-only early break in their loop.
   - One training run, seed 44. The spread across training runs is a stated limitation.
   - Evaluation: 256 × 256 resize, Normalize(0.5, 0.5), 64 random 64 × 64 crops per image seeded by
     the image name, and per-image patch normalisation (their `Patchnorm2D`).
   - Degradation score = 1 − mean patch output. The final checkpoint is used, with no model selection.
9. **Metrics.** All scores are oriented so that higher means more degraded.
   - **Separation, threshold-free:**
     - AUROC.
     - AUPR: average precision with degraded images as the positive class. Each condition has equal
       numbers of clean and degraded images, so chance is 0.5.
   - **Separation, one operating point:** FPR95. Balanced accuracy is not reported.
   - **Harm, detection quality:** per-image LRP at a single confidence threshold, the LRP-optimal one
     on the 5,000 clean images (the optimal-LRP convention). The same threshold applies to all methods
     and conditions. Crowd regions are ignored.
   - **Harm, between conditions:** Spearman ρ between condition mean score and condition COCO mAP, and
     between condition mean score and condition mean LRP, over the 95 corrupted conditions.
   - **Harm, within a condition (primary per-image measure):**
     - For each condition, Spearman ρ over images between Δscore and ΔLRP, where Δ = corrupted − clean
       for the same image.
     - Then the mean over conditions, also reported by severity.
   - **Harm, risk–coverage (secondary):** AURC with per-image LRP as the risk, plus an oracle.
     - Pools: clean + one severity (all families); clean + one family (all severities); clean + all.
     - It also rewards flagging hard clean scenes, which is why it is secondary.
   - **Inference:**
     - A paired whole-image bootstrap with 1,000 draws, using the same draws for every method, with λ
       refitted inside each draw.
     - It covers aggregate AUROC, AUPR and FPR95, within-condition ρ, condition-level ρ with mean LRP,
       AURC, and every pairwise difference between methods.
     - There is no interval for ρ with mAP, because that would need 96 COCO evaluations per draw.
     - The 95 per-cell values are descriptive.
   - **Runtime:** median milliseconds per image at batch 1 on a warm GPU, including preprocessing, for
     the detector alone and for each baseline.
10. **Provenance (light):** `environment.json` records package versions, the DisCoPatch commit and the
    GPU name. It does not block resumes.

## Global Constraints

- Python: `/home/yuchen/miniconda3/envs/UE/bin/python`. Run tests from the repo root with
  `$PY -m pytest …`.
- Data:
  - COCO train: `/home/yuchen/YuchenZ/Datasets/coco/train2017`
  - COCO val: `/home/yuchen/YuchenZ/Datasets/coco/val2017`
  - COCO val annotations: `/home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json`
- DisCoPatch repo: `/home/yuchen/YuchenZ/UE/DisCoPatch` (commit `ec36b8529cef6c994a70237ca59cc0c8902d9aae`).
  Do not edit it; patch its module globals from our adapter instead.
- Outputs go under `runs/coco-baselines/`. `runs/` is already in `.gitignore`. Result tables that are
  kept go to `docs/results/coco-baselines/`.
- Do not change the existing benchmark path (`benchmark.py`, `scoring.py`, `corruptions/__init__.py`).
  The new protocol lives in `baselines/protocol.py`.
- Fixed constants:
  - seed 44;
  - all 5,000 val images, 5 folds;
  - top-k 100; θ = 0.3;
  - kNN k = 100, with k_max = 200;
  - DisCoPatch: 64 evaluation patches, 65 training epochs, one training run;
  - 1,000 bootstrap draws.
- Every score is oriented so that higher means more likely degraded.
- Commits: the harness only allows commits when the user asks. Each "Commit" step below is a
  checkpoint: propose the commit and wait for the user's go-ahead.

## Review Focus

1. **Grayscale COCO images.** Some val and train images are single-channel. Every path must convert
   them to RGB before corruption, detection and DisCoPatch. Pinned in Task 1 and Task 6.
2. **Images without objects.** Some val images have no annotations. Per-image LRP must be NaN (dropped
   and counted) when there are no ground-truth objects and no kept detections, and per-image AP must be
   NaN. They must also drop out of λ fitting and the within-condition correlations rather than become
   0. Pinned in Task 4 and Task 5.
3. **An interrupted or damaged result file.** It must stop the run with the file named, not be skipped
   or overwritten silently. Pinned in Task 7.
4. **The DisCoPatch pass seeing different corruptions from the detector pass.** A stored digest per
   variant must match, or the run stops. Pinned in Task 7.
5. **Cross-fitting leaking an image's own fold into its λ.** Fold *f*'s λ must be fitted only on the
   other folds. Pinned in Task 5.

Task 6 separately pins that DisCoPatch normalises each image on its own.

## File structure

| File | Responsibility |
| --- | --- |
| `differential_uncertainty/baselines/__init__.py` | Package marker |
| `differential_uncertainty/baselines/protocol.py` | Families and conditions, evaluation images and folds, seeded corruption, variant digests |
| `differential_uncertainty/baselines/scores.py` | SAOD, ContrastiveConf parts, kNN distances (pure functions) |
| `differential_uncertainty/baselines/detector.py` | Frozen RT-DETRv2 forward pass returning logits, boxes and the pooled layer4 feature |
| `differential_uncertainty/baselines/coco_quality.py` | COCO ground truth, per-image LRP, per-image AP, condition mAP, LRP threshold |
| `differential_uncertainty/baselines/metrics.py` | AUROC, AUPR, FPR95, Spearman, within-condition ρ, risk–coverage, λ fitting and cross-fitting, bootstrap |
| `differential_uncertainty/baselines/discopatch.py` | Imports the official DisCoPatch code, COCO patch dataset, training wrapper, per-image scorer |
| `differential_uncertainty/baselines/pipeline.py` | Settings, run config, environment record, phases `sanity`, `bank`, `test`, `train-discopatch`, `discopatch-scores`, `timing`, `report` |
| `differential_uncertainty/baselines/report.py` | Loads stored arrays, cross-fits λ, computes every table and interval, writes CSV, JSON and Markdown |
| `differential_uncertainty/cli.py`, `differential_uncertainty/__main__.py` (modify) | `baselines-coco` subcommand; `__main__` guard for spawned workers |
| `tests/differential_uncertainty/test_baselines_*.py` | One test file per module |

---

### Task 1: Protocol: conditions, evaluation images, folds and seeded corruptions

**Files:**
- Create: `differential_uncertainty/baselines/__init__.py`, `differential_uncertainty/baselines/protocol.py`
- Test: `tests/differential_uncertainty/test_baselines_protocol.py`

**Interfaces:**
- Consumes: `differential_uncertainty.corruptions.imagecorruptions.apply_imagecorruption(image, name, severity) -> PIL.Image`.
- Produces:
  - Constants: `COMMON_FAMILIES`, `EXTRA_FAMILIES`, `FAMILIES`, `SEVERITIES`, `FOLDS = 5`, and `CONDITIONS` (96 `(family, severity)` tuples, clean first as `("clean", 0)`).
  - `list_images(root) -> list[Path]`
  - `evaluation_images(val_root, *, seed) -> list[Path]` (all images, seeded shuffle)
  - `assign_folds(count, folds=FOLDS) -> ndarray`
  - `variant_seed(image_id, family, severity) -> int`
  - `corrupt(image, image_id, family, severity) -> PIL.Image`
  - `variants(image, image_id) -> list[np.ndarray]` (96 uint8 H×W×3 arrays)
  - `load_variants(path) -> tuple[str, list[np.ndarray]]`
  - `digest(array) -> str`

- [ ] **Step 1: Write the failing tests**

```python
# tests/differential_uncertainty/test_baselines_protocol.py
import numpy as np
import pytest
from PIL import Image

from differential_uncertainty import benchmark
from differential_uncertainty.baselines import protocol
from differential_uncertainty.corruptions import apply_corruption
from differential_uncertainty.corruptions.imagecorruptions import apply_imagecorruption


def _image(width=64, height=48, mode="RGB"):
    pixels = np.random.default_rng(0).integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB").convert(mode)


def test_conditions_are_clean_then_19_families_by_5_severities_in_package_order():
    from imagecorruptions import get_corruption_names
    assert protocol.FAMILIES == tuple(get_corruption_names("all"))
    assert protocol.COMMON_FAMILIES == tuple(get_corruption_names("common"))
    assert protocol.CONDITIONS[0] == ("clean", 0)
    assert len(protocol.CONDITIONS) == 96
    assert protocol.CONDITIONS[1:6] == tuple(("gaussian_noise", s) for s in range(1, 6))


def test_evaluation_images_use_the_old_benchmark_shuffle(tmp_path):
    for index in range(20):
        (tmp_path / f"{index:03d}.jpg").write_bytes(b"x")
    images = protocol.evaluation_images(tmp_path, seed=44)
    assert images == benchmark._select_images(tmp_path, 20, seed=44)
    assert images[:8] == benchmark._select_images(tmp_path, 8, seed=44)


def test_evaluation_images_reject_an_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="no images"):
        protocol.evaluation_images(tmp_path, seed=44)


def test_folds_are_balanced_and_follow_shuffled_position():
    assert np.bincount(protocol.assign_folds(5000)).tolist() == [1000] * 5
    assert protocol.assign_folds(7).tolist() == [0, 1, 2, 3, 4, 0, 1]


def test_corrupt_is_seeded_per_image_and_restores_global_numpy_state():
    image = _image()
    np.random.seed(123)
    expected_next = np.random.rand()
    np.random.seed(123)
    first = np.asarray(protocol.corrupt(image, "a.jpg", "gaussian_noise", 3))
    assert np.random.rand() == expected_next
    again = np.asarray(protocol.corrupt(image, "a.jpg", "gaussian_noise", 3))
    other = np.asarray(protocol.corrupt(image, "b.jpg", "gaussian_noise", 3))
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_every_condition_keeps_size_and_returns_rgb_for_grayscale_input():
    arrays = protocol.variants(_image(mode="L"), "gray.jpg")
    assert len(arrays) == 96
    assert all(a.shape == (48, 64, 3) and a.dtype == np.uint8 for a in arrays)


def test_gaussian_blur_uses_the_imagecorruptions_package_not_the_old_pil_blur():
    image = _image()
    ours = np.asarray(protocol.corrupt(image, "a.jpg", "gaussian_blur", 5))
    assert np.array_equal(ours, np.asarray(apply_imagecorruption(image, "gaussian_blur", 5)))
    assert not np.array_equal(ours, np.asarray(apply_corruption(image, "gaussian_blur", 5)))


@pytest.mark.parametrize("family, severity", [("rain", 3), ("fog", 0), ("fog", 6), ("fog", 2.0)])
def test_corrupt_rejects_unknown_family_or_severity(family, severity):
    with pytest.raises(ValueError):
        protocol.corrupt(_image(), "a.jpg", family, severity)


def test_load_variants_returns_file_name_and_stable_digests(tmp_path):
    path = tmp_path / "img.png"
    _image().save(path)
    name, arrays = protocol.load_variants(path)
    assert name == "img.png" and len(arrays) == 96
    assert protocol.digest(arrays[0]) == protocol.digest(arrays[0].copy())
    assert protocol.digest(arrays[0]) != protocol.digest(arrays[1])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/differential_uncertainty/test_baselines_protocol.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'differential_uncertainty.baselines'`.

- [ ] **Step 3: Write the implementation**

```python
# differential_uncertainty/baselines/__init__.py
"""Baseline image-level degradation scores on the fixed COCO protocol."""
```

```python
# differential_uncertainty/baselines/protocol.py
"""Fixed COCO protocol for the baselines: conditions, evaluation images, folds, seeded corruptions."""
from __future__ import annotations

import hashlib
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from ..corruptions.imagecorruptions import apply_imagecorruption

COMMON_FAMILIES = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
    "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
)
EXTRA_FAMILIES = ("speckle_noise", "gaussian_blur", "spatter", "saturate")
FAMILIES = COMMON_FAMILIES + EXTRA_FAMILIES
SEVERITIES = (1, 2, 3, 4, 5)
CONDITIONS = (("clean", 0),) + tuple(
    (family, severity) for family in FAMILIES for severity in SEVERITIES
)
FOLDS = 5
_SUFFIXES = {".jpg", ".jpeg", ".png"}


def list_images(root) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"image directory does not exist: {root}")
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _SUFFIXES)


def evaluation_images(val_root, *, seed: int) -> list[Path]:
    """All val images in the old benchmark's seeded shuffle order; the order defines the folds."""
    images = list_images(val_root)
    if not images:
        raise ValueError(f"no images found in {val_root}")
    np.random.default_rng(seed).shuffle(images)
    return images


def assign_folds(count: int, folds: int = FOLDS) -> np.ndarray:
    """Fold of each image from its shuffled position: 0, 1, ..., folds - 1, 0, 1, ..."""
    if count <= 0 or folds < 2:
        raise ValueError("count must be positive and folds at least 2")
    return np.arange(count) % folds


def variant_seed(image_id: str, family: str, severity: int) -> int:
    return zlib.crc32(f"{image_id}|{family}|{severity}".encode("utf-8")) & 0xFFFFFFFF


def corrupt(image: Image.Image, image_id: str, family: str, severity: int) -> Image.Image:
    if family not in FAMILIES:
        raise ValueError(f"unknown corruption family {family!r}")
    if type(severity) is not int or severity not in SEVERITIES:
        raise ValueError("severity must be an integer from 1 to 5")
    rgb = image.convert("RGB")
    saved = np.random.get_state()
    try:
        np.random.seed(variant_seed(image_id, family, severity))
        corrupted = apply_imagecorruption(rgb, family, severity)
    finally:
        np.random.set_state(saved)
    if corrupted.size != rgb.size or corrupted.mode != "RGB":
        raise ValueError(f"{family} severity {severity} changed size or mode of {image_id}")
    return corrupted


def variants(image: Image.Image, image_id: str) -> list[np.ndarray]:
    """All 96 versions of one image, in CONDITIONS order, as uint8 RGB arrays."""
    rgb = image.convert("RGB")
    arrays = [np.asarray(rgb, dtype=np.uint8).copy()]
    for family, severity in CONDITIONS[1:]:
        arrays.append(np.asarray(corrupt(rgb, image_id, family, severity), dtype=np.uint8).copy())
    return arrays


def load_variants(path) -> tuple[str, list[np.ndarray]]:
    """Worker entry point: open one image file and build its 96 versions."""
    path = Path(path)
    with Image.open(path) as source:
        rgb = source.convert("RGB")
    return path.name, variants(rgb, path.name)


def digest(array: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/differential_uncertainty/test_baselines_protocol.py -v`
Expected: 12 passed. The grayscale test runs all 95 corruptions on a 64 × 48 image and takes a few
seconds.

- [ ] **Step 5: Commit checkpoint**

```bash
git add differential_uncertainty/baselines/__init__.py differential_uncertainty/baselines/protocol.py tests/differential_uncertainty/test_baselines_protocol.py
git commit -m "feat: add fixed COCO protocol for baseline scores"
```

---

### Task 2: Baseline scores (SAOD, ContrastiveConf, kNN)

**Files:**
- Create: `differential_uncertainty/baselines/scores.py`
- Test: `tests/differential_uncertainty/test_baselines_scores.py`

**Interfaces:**
- Consumes: `uq_detr.Detections`, `uq_detr.contrastive_conf`.
- Produces:
  - `top_detections(logits, boxes_cxcywh, image_size, top_k=100) -> (scores float32 (k,), labels int64 (k,), boxes_xyxy float32 (k, 4))`. `image_size` is (width, height) of the original image.
  - `saod_uncertainty(top_scores, m) -> float`
  - `query_detections(logits, boxes_cxcywh, image_size) -> uq_detr.Detections`
  - `contrastive_parts(query_sets, theta=0.3) -> (conf_pos (n,), conf_neg (n,))`
  - `contrastive_degradation(conf_pos, conf_neg, lam) -> ndarray`
  - `normalize_rows(features: Tensor) -> Tensor`
  - `knn_distances(queries: Tensor, bank: Tensor, k_max: int, chunk_size=16384) -> Tensor (n, k_max)`, sorted ascending.

- [ ] **Step 1: Write the failing tests**

```python
# tests/differential_uncertainty/test_baselines_scores.py
import numpy as np
import pytest
import torch
import uq_detr
from scipy.special import logit

from differential_uncertainty.baselines import scores


def _outputs(max_probs, classes=3):
    probs = np.full((len(max_probs), classes), 1e-4)
    probs[:, 0] = max_probs
    boxes = np.tile([0.5, 0.5, 0.2, 0.4], (len(max_probs), 1))
    return logit(probs), boxes


def test_top_detections_ranks_query_class_pairs_and_scales_boxes_to_pixels():
    logits = logit(np.array([[0.2, 0.9], [0.7, 0.1]]))
    boxes = np.array([[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.5, 0.5]])
    top, labels, xyxy = scores.top_detections(logits, boxes, image_size=(100, 50), top_k=3)
    assert top == pytest.approx([0.9, 0.7, 0.2], abs=1e-6)
    assert labels.tolist() == [1, 0, 0]
    assert xyxy[0] == pytest.approx([40, 15, 60, 35])
    assert xyxy[1] == pytest.approx([0, 0, 50, 25])


def test_saod_uncertainty_averages_one_minus_confidence_of_the_m_best():
    top = np.array([0.1, 0.9, 0.8])
    assert scores.saod_uncertainty(top, 1) == pytest.approx(0.1)
    assert scores.saod_uncertainty(top, 3) == pytest.approx((0.1 + 0.2 + 0.9) / 3)
    for bad in (0, 4):
        with pytest.raises(ValueError):
            scores.saod_uncertainty(top, bad)


def test_contrastive_parts_match_uq_detr_threshold_split():
    logits, boxes = _outputs([0.9, 0.5, 0.2, 0.1])
    queries = [scores.query_detections(logits, boxes, (100, 50))]
    conf_pos, conf_neg = scores.contrastive_parts(queries, theta=0.3)
    assert conf_pos[0] == pytest.approx(0.7)
    assert conf_neg[0] == pytest.approx(0.15)
    reference = uq_detr.contrastive_conf(queries, method="threshold", param=0.3, lambda_=5.0)[0]
    assert -scores.contrastive_degradation(conf_pos, conf_neg, 5.0)[0] == pytest.approx(reference)


def test_contrastive_parts_without_positive_queries_use_all_queries_as_negatives():
    logits, boxes = _outputs([0.2, 0.1])
    conf_pos, conf_neg = scores.contrastive_parts([scores.query_detections(logits, boxes, (10, 10))])
    assert conf_pos[0] == 0.0 and conf_neg[0] == pytest.approx(0.15)


def test_knn_distances_are_sorted_kth_neighbour_distances_and_independent_of_chunking():
    angles = np.deg2rad([0, 10, 20, 30, 90])
    bank = torch.tensor(np.stack([np.cos(angles), np.sin(angles)], 1), dtype=torch.float32)
    query = torch.tensor([[2.0, 0.0]])
    full = scores.knn_distances(query, bank, k_max=3, chunk_size=100)
    chunked = scores.knn_distances(query, bank, k_max=3, chunk_size=2)
    expected = [2 * np.sin(np.deg2rad(d) / 2) for d in (0, 10, 20)]
    assert full[0].tolist() == pytest.approx(expected, abs=1e-3)
    assert torch.allclose(full, chunked)


def test_knn_distances_reject_zero_queries_and_too_large_k():
    bank = torch.eye(2)
    with pytest.raises(ValueError):
        scores.knn_distances(torch.zeros(1, 2), bank, k_max=1)
    with pytest.raises(ValueError):
        scores.knn_distances(torch.ones(1, 2), bank, k_max=3)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/differential_uncertainty/test_baselines_scores.py -v`
Expected: FAIL with `ImportError: cannot import name 'scores'`.

- [ ] **Step 3: Write the implementation**

```python
# differential_uncertainty/baselines/scores.py
"""SAOD, ContrastiveConf and kNN image scores from one frozen detector."""
from __future__ import annotations

import numpy as np
import torch
import uq_detr
from scipy.special import expit


def _checked(logits, boxes):
    logits = np.asarray(logits, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64)
    if logits.ndim != 2 or boxes.shape != (logits.shape[0], 4):
        raise ValueError("expected logits (queries, classes) and boxes (queries, 4)")
    if not (np.isfinite(logits).all() and np.isfinite(boxes).all()):
        raise ValueError("detector outputs must be finite")
    return logits, boxes


def top_detections(logits, boxes_cxcywh, image_size, top_k=100):
    """RT-DETR post-processing: sigmoid, then the top-k (query, class) pairs."""
    logits, boxes = _checked(logits, boxes_cxcywh)
    width, height = image_size
    probs = expit(logits).reshape(-1)
    order = np.argsort(-probs, kind="stable")[: min(int(top_k), probs.size)]
    queries, labels = np.divmod(order, logits.shape[1])
    cx, cy, w, h = boxes[queries].T
    xyxy = np.stack(
        [(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height],
        axis=1,
    )
    return probs[order].astype(np.float32), labels.astype(np.int64), xyxy.astype(np.float32)


def saod_uncertainty(top_scores, m: int) -> float:
    """SAOD image uncertainty: mean of (1 - p) over the m most confident detections."""
    ranked = np.sort(np.asarray(top_scores, dtype=np.float64))[::-1]
    if type(m) is not int or not 1 <= m <= ranked.size:
        raise ValueError("m must be between 1 and the number of detections")
    return float(np.mean(1.0 - ranked[:m]))


def query_detections(logits, boxes_cxcywh, image_size) -> uq_detr.Detections:
    """All queries of one image as uq-detr Detections (image_size is width, height)."""
    logits, boxes = _checked(logits, boxes_cxcywh)
    width, height = image_size
    return uq_detr.Detections.from_cxcywh(boxes, expit(logits), image_size=(height, width))


def contrastive_parts(query_sets, theta: float = 0.3):
    """Conf+ and Conf- from uq-detr's threshold split, recovered from two lambda values."""
    conf_pos = np.asarray(
        uq_detr.contrastive_conf(query_sets, method="threshold", param=theta, lambda_=0.0), float)
    difference = np.asarray(
        uq_detr.contrastive_conf(query_sets, method="threshold", param=theta, lambda_=1.0), float)
    return conf_pos, conf_pos - difference


def contrastive_degradation(conf_pos, conf_neg, lam: float) -> np.ndarray:
    """Higher means more likely degraded: minus ContrastiveConf."""
    return -(np.asarray(conf_pos, float) - lam * np.asarray(conf_neg, float))


def normalize_rows(features: torch.Tensor) -> torch.Tensor:
    features = features.float()
    norms = features.norm(dim=1, keepdim=True)
    if not bool(torch.isfinite(features).all()) or bool((norms == 0).any()):
        raise ValueError("features must be finite with nonzero norm")
    return features / norms


def knn_distances(queries: torch.Tensor, bank: torch.Tensor, k_max: int, chunk_size: int = 16384):
    """Ascending Euclidean distances from unit queries to their k_max nearest unit bank rows."""
    if queries.ndim != 2 or bank.ndim != 2 or queries.shape[1] != bank.shape[1]:
        raise ValueError("queries and bank must be (rows, dim) with the same dim")
    if not 1 <= k_max <= bank.shape[0]:
        raise ValueError("k_max must be between 1 and the bank size")
    q = normalize_rows(queries)
    best = None
    for start in range(0, bank.shape[0], chunk_size):
        rows = bank[start:start + chunk_size].to(device=q.device, dtype=torch.float32)
        squared = (2.0 - 2.0 * q @ rows.T).clamp_min_(0.0)
        local = squared.topk(min(k_max, rows.shape[0]), dim=1, largest=False).values
        merged = local if best is None else torch.cat([best, local], dim=1)
        best = merged.topk(min(k_max, merged.shape[1]), dim=1, largest=False).values
    return best.sqrt()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/differential_uncertainty/test_baselines_scores.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit checkpoint**

```bash
git add differential_uncertainty/baselines/scores.py tests/differential_uncertainty/test_baselines_scores.py
git commit -m "feat: add SAOD, ContrastiveConf and kNN image scores"
```

---

### Task 3: Detector tap

**Files:**
- Create: `differential_uncertainty/baselines/detector.py`
- Test: `tests/differential_uncertainty/test_baselines_detector.py`

**Interfaces:**
- Consumes: `differential_uncertainty.extraction.load_frozen_detector(path, device)` and `prepare_image(image, size) -> Tensor`.
- Produces:
  - `DetectorTap(checkpoint_path, device, image_size=(640, 640))`, a context manager.
  - `.prepare(arrays) -> Tensor`
  - `.forward(batch) -> (logits (n, 300, 80), boxes (n, 300, 4), pooled (n, 512))` as float32 numpy arrays.
  - `.run(arrays, batch_size=32) -> same triple`
  - `QUERY_COUNT = 300`, `CLASS_COUNT = 80`, `POOLED_DIM = 512`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/differential_uncertainty/test_baselines_detector.py
import numpy as np
import pytest
import torch
from torch import nn

from differential_uncertainty.baselines import detector


class FakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Identity()

    def forward(self, images):
        n = images.shape[0]
        self.backbone([torch.zeros(n, 256, 4, 4), torch.ones(n, 512, 2, 2) * images.mean()])
        return {"pred_logits": torch.zeros(n, 300, 80), "pred_boxes": torch.full((n, 300, 4), 0.5)}


def test_tap_returns_logits_boxes_and_pooled_last_backbone_stage(monkeypatch):
    monkeypatch.setattr(detector, "load_frozen_detector", lambda _p, _d: FakeDetector())
    arrays = [np.full((20, 30, 3), 255, np.uint8), np.zeros((10, 10, 3), np.uint8)]
    with detector.DetectorTap("unused.pth", "cpu", image_size=(16, 16)) as tap:
        logits, boxes, pooled = tap.run(arrays, batch_size=1)
    assert logits.shape == (2, 300, 80) and boxes.shape == (2, 300, 4) and pooled.shape == (2, 512)
    assert pooled[0, 0] == pytest.approx(1.0) and pooled[1, 0] == pytest.approx(0.0)


def test_tap_rejects_non_finite_outputs(monkeypatch):
    class Broken(FakeDetector):
        def forward(self, images):
            out = super().forward(images)
            out["pred_logits"][0, 0, 0] = float("nan")
            return out

    monkeypatch.setattr(detector, "load_frozen_detector", lambda _p, _d: Broken())
    with detector.DetectorTap("unused.pth", "cpu", image_size=(8, 8)) as tap:
        with pytest.raises(ValueError, match="finite"):
            tap.run([np.zeros((8, 8, 3), np.uint8)])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/differential_uncertainty/test_baselines_detector.py -v`
Expected: FAIL with `ImportError: cannot import name 'detector'`.

- [ ] **Step 3: Write the implementation**

```python
# differential_uncertainty/baselines/detector.py
"""Frozen RT-DETRv2 forward pass exposing logits, boxes and the pooled last backbone stage."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from ..extraction import load_frozen_detector, prepare_image

QUERY_COUNT = 300
CLASS_COUNT = 80
POOLED_DIM = 512


class DetectorTap:
    def __init__(self, checkpoint_path, device, image_size=(640, 640)):
        self.device = torch.device(device)
        self.image_size = tuple(image_size)
        self.model = load_frozen_detector(checkpoint_path, self.device)
        self._pooled = None
        self._handle = self.model.backbone.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, outputs):
        self._pooled = outputs[-1].mean(dim=(2, 3))

    def prepare(self, arrays) -> torch.Tensor:
        return torch.stack([prepare_image(Image.fromarray(a), self.image_size) for a in arrays])

    @torch.inference_mode()
    def forward(self, batch: torch.Tensor):
        outputs = self.model(batch.to(self.device, non_blocking=True))
        pooled, self._pooled = self._pooled, None
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

    def run(self, arrays, batch_size: int = 32):
        parts = [self.forward(self.prepare(arrays[s:s + batch_size]))
                 for s in range(0, len(arrays), batch_size)]
        return tuple(np.concatenate(values) for values in zip(*parts))

    def close(self) -> None:
        self._handle.remove()
        self.model = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/differential_uncertainty/test_baselines_detector.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit checkpoint**

```bash
git add differential_uncertainty/baselines/detector.py tests/differential_uncertainty/test_baselines_detector.py
git commit -m "feat: add frozen detector tap for baseline scores"
```

---

### Task 4: COCO detection quality (LRP, AP, mAP)

**Files:**
- Create: `differential_uncertainty/baselines/coco_quality.py`
- Test: `tests/differential_uncertainty/test_baselines_coco_quality.py`

**Interfaces:**
- Consumes: `uq_detr.Detections`, `uq_detr.GroundTruth`, `uq_detr.lrp`, `pycocotools`.
- Produces:
  - `CocoGroundTruth(annotation_file, expected_categories=80)` with:
    - `.category_ids` (sorted COCO ids; label i ↔ `category_ids[i]`);
    - `.image_id(file_name) -> int`;
    - `.boxes(image_id) -> (gt_xyxy (n, 4), gt_labels (n,), crowd_xyxy (m, 4))`.
  - `filter_detections(scores, labels, boxes, threshold, crowd_boxes) -> (scores, labels, boxes)`
  - `image_lrp(scores, labels, boxes, gt_boxes, gt_labels, crowd_boxes, threshold) -> float` (NaN when undefined)
  - `select_lrp_threshold(records, gt, grid=…) -> float`, where `records` is a list of `(image_id, scores, labels, boxes)`
  - `coco_results(image_id, scores, labels, boxes, category_ids) -> list[dict]`
  - `coco_map(gt, results, image_ids) -> float`
  - `per_image_ap(gt, results_by_image, image_ids) -> ndarray` (NaN when an image has no objects)

- [ ] **Step 1: Write the failing tests**

```python
# tests/differential_uncertainty/test_baselines_coco_quality.py
import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest

from differential_uncertainty.baselines import coco_quality as cq


def _gt(tmp_path):
    data = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100},
                   {"id": 2, "file_name": "b.jpg", "width": 100, "height": 100}],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 3, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
            {"id": 11, "image_id": 1, "category_id": 7, "bbox": [60, 60, 30, 30], "area": 900, "iscrowd": 1},
        ],
        "categories": [{"id": 3, "name": "car"}, {"id": 7, "name": "train"}],
    }
    path = tmp_path / "ann.json"
    path.write_text(json.dumps(data))
    return cq.CocoGroundTruth(path, expected_categories=None)


def test_ground_truth_splits_crowd_regions_and_maps_labels(tmp_path):
    gt = _gt(tmp_path)
    boxes, labels, crowd = gt.boxes(gt.image_id("a.jpg"))
    assert gt.category_ids == (3, 7)
    assert boxes.tolist() == [[10, 10, 30, 30]] and labels.tolist() == [0]
    assert crowd.tolist() == [[60, 60, 90, 90]]


def test_image_lrp_perfect_false_positive_undefined_and_crowd_cases(tmp_path):
    gt = _gt(tmp_path)
    boxes, labels, crowd = gt.boxes(1)
    good = np.array([[10, 10, 30, 30.0]])
    assert cq.image_lrp(np.array([0.9]), np.array([0]), good, boxes, labels, crowd, 0.5) == pytest.approx(0.0)
    in_crowd = np.array([[10, 10, 30, 30.0], [62, 62, 88, 88.0]])
    assert cq.image_lrp(np.array([0.9, 0.9]), np.array([0, 1]), in_crowd, boxes, labels, crowd, 0.5) == pytest.approx(0.0)
    empty = gt.boxes(2)
    stray = np.array([[0, 0, 5, 5.0]])
    assert math.isnan(cq.image_lrp(np.array([0.1]), np.array([0]), stray, *empty, 0.5))
    assert cq.image_lrp(np.array([0.9]), np.array([0]), stray, *empty, 0.5) == pytest.approx(1.0)


def test_image_lrp_counts_localisation_error_of_true_positives(tmp_path):
    gt = _gt(tmp_path)
    boxes, labels, crowd = gt.boxes(1)
    value = cq.image_lrp(np.array([0.9]), np.array([0]), np.array([[10, 10, 30, 25.0]]), boxes, labels, crowd, 0.5)
    assert value == pytest.approx((1 - 0.75) / 0.5)


def test_select_lrp_threshold_drops_low_confidence_false_positives(tmp_path):
    gt = _gt(tmp_path)
    records = [(1, np.array([0.9, 0.2]), np.array([0, 0]), np.array([[10, 10, 30, 30.0], [40, 40, 50, 50.0]]))]
    assert cq.select_lrp_threshold(records, gt, grid=(0.1, 0.5)) == 0.5


def test_coco_map_and_per_image_ap(tmp_path):
    gt = _gt(tmp_path)
    results = cq.coco_results(1, np.array([0.9]), np.array([0]), np.array([[10, 10, 30, 30.0]]), gt.category_ids)
    assert results[0]["bbox"] == pytest.approx([10, 10, 20, 20]) and results[0]["category_id"] == 3
    assert cq.coco_map(gt, results, [1, 2]) == pytest.approx(1.0)
    ap = cq.per_image_ap(gt, {1: results, 2: []}, [1, 2])
    assert ap[0] == pytest.approx(1.0) and math.isnan(ap[1])
    assert cq.coco_map(gt, [], [1, 2]) == 0.0


def test_label_order_matches_rtdetr_mscoco_mapping():
    source = Path("/home/yuchen/YuchenZ/RT-DETR/rtdetrv2_pytorch/src/data/dataset/coco_dataset.py")
    annotations = Path("/home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json")
    if not (source.exists() and annotations.exists()):
        pytest.skip("RT-DETR repository or COCO annotations not available")
    tree = ast.parse(source.read_text())
    mapping = next(
        ast.literal_eval(node.value) for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "mscoco_category2name"
    )
    assert tuple(mapping) == cq.CocoGroundTruth(annotations).category_ids
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/differential_uncertainty/test_baselines_coco_quality.py -v`
Expected: FAIL with `ImportError: cannot import name 'coco_quality'`.

- [ ] **Step 3: Write the implementation**

```python
# differential_uncertainty/baselines/coco_quality.py
"""Detection quality on COCO: per-image LRP (uq-detr), per-image AP and mAP (pycocotools)."""
from __future__ import annotations

import contextlib
import io
import math

import numpy as np
import uq_detr
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DEFAULT_LRP_GRID = tuple(np.round(np.arange(0.05, 0.951, 0.05), 2))


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


class CocoGroundTruth:
    def __init__(self, annotation_file, expected_categories=80):
        with _quiet():
            self.coco = COCO(str(annotation_file))
        self.category_ids = tuple(sorted(self.coco.getCatIds()))
        if expected_categories is not None and len(self.category_ids) != expected_categories:
            raise ValueError(f"expected {expected_categories} categories, found {len(self.category_ids)}")
        self.label_of = {c: i for i, c in enumerate(self.category_ids)}
        self._ids = {info["file_name"]: image_id for image_id, info in self.coco.imgs.items()}

    def image_id(self, file_name: str) -> int:
        try:
            return self._ids[file_name]
        except KeyError as error:
            raise KeyError(f"{file_name} is not in the annotation file") from error

    def boxes(self, image_id: int):
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image_id]))
        def xyxy(items):
            return np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2],
                              a["bbox"][1] + a["bbox"][3]] for a in items], dtype=np.float64).reshape(-1, 4)
        plain = [a for a in anns if not a.get("iscrowd", 0)]
        crowd = [a for a in anns if a.get("iscrowd", 0)]
        labels = np.array([self.label_of[a["category_id"]] for a in plain], dtype=np.int64)
        return xyxy(plain), labels, xyxy(crowd)


def _ioa(boxes, regions):
    """Intersection over each box's own area, shape (boxes, regions)."""
    x1 = np.maximum(boxes[:, None, 0], regions[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], regions[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], regions[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], regions[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    return inter / np.maximum(area[:, None], 1e-12)


def filter_detections(scores, labels, boxes, threshold, crowd_boxes, crowd_ioa=0.5):
    scores, labels = np.asarray(scores), np.asarray(labels)
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    keep = scores >= threshold
    if crowd_boxes.size and boxes.size:
        keep &= ~(_ioa(boxes, crowd_boxes) >= crowd_ioa).any(axis=1)
    return scores[keep], labels[keep], boxes[keep]


def image_lrp(scores, labels, boxes, gt_boxes, gt_labels, crowd_boxes, threshold) -> float:
    kept_scores, kept_labels, kept_boxes = filter_detections(scores, labels, boxes, threshold, crowd_boxes)
    if kept_scores.size == 0 and gt_labels.size == 0:
        return math.nan
    dets = uq_detr.Detections(boxes=kept_boxes, scores=kept_scores, labels=kept_labels)
    truth = uq_detr.GroundTruth(boxes=gt_boxes.reshape(-1, 4), labels=gt_labels)
    return float(uq_detr.lrp([dets], [truth], iou_threshold=0.5).score)


def _dataset_lrp(records, gt, threshold) -> float:
    dets, truths = [], []
    for image_id, scores, labels, boxes in records:
        gt_boxes, gt_labels, crowd = gt.boxes(image_id)
        s, l, b = filter_detections(scores, labels, boxes, threshold, crowd)
        dets.append(uq_detr.Detections(boxes=b, scores=s, labels=l))
        truths.append(uq_detr.GroundTruth(boxes=gt_boxes, labels=gt_labels))
    return float(uq_detr.lrp(dets, truths, iou_threshold=0.5).score)


def select_lrp_threshold(records, gt, grid=DEFAULT_LRP_GRID) -> float:
    """Confidence threshold with the lowest dataset LRP; ties go to the lowest threshold."""
    values = [(_dataset_lrp(records, gt, t), t) for t in grid]
    return float(min(values, key=lambda item: (item[0], item[1]))[1])


def coco_results(image_id, scores, labels, boxes, category_ids) -> list[dict]:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return [{"image_id": int(image_id), "category_id": int(category_ids[int(l)]),
             "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
             "score": float(s)} for s, l, b in zip(scores, labels, boxes)]


def _evaluate(gt, detections, image_ids):
    evaluator = COCOeval(gt.coco, detections, "bbox")
    evaluator.params.imgIds = list(image_ids)
    with _quiet():
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return float(evaluator.stats[0])


def coco_map(gt, results, image_ids) -> float:
    if not results:
        return 0.0
    with _quiet():
        detections = gt.coco.loadRes(list(results))
    return max(_evaluate(gt, detections, image_ids), 0.0)


def per_image_ap(gt, results_by_image, image_ids) -> np.ndarray:
    """COCO AP@[.5:.95] for each image separately; NaN for images without objects."""
    all_results = [r for image_id in image_ids for r in results_by_image.get(image_id, [])]
    has_objects = [bool(gt.boxes(i)[1].size) for i in image_ids]
    if not all_results:
        return np.array([0.0 if objects else math.nan for objects in has_objects])
    with _quiet():
        detections = gt.coco.loadRes(all_results)
    values = []
    for image_id, objects in zip(image_ids, has_objects):
        ap = _evaluate(gt, detections, [image_id]) if objects else -1.0
        values.append(ap if ap >= 0 else math.nan)
    return np.array(values)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/differential_uncertainty/test_baselines_coco_quality.py -v`
Expected: 6 passed. If the localisation test fails, print `uq_detr.compute_iou_matrix` for the two boxes
to check its pixel convention, then adjust the expected IoU in the test to match `uq-detr`. We use its
LRP as published.

- [ ] **Step 5: Commit checkpoint**

```bash
git add differential_uncertainty/baselines/coco_quality.py tests/differential_uncertainty/test_baselines_coco_quality.py
git commit -m "feat: add COCO detection quality for harm metrics"
```

---

### Task 5: Metrics, λ cross-fitting and bootstrap

**Files:**
- Create: `differential_uncertainty/baselines/metrics.py`
- Test: `tests/differential_uncertainty/test_baselines_metrics.py`

**Interfaces:**
- Consumes: `differential_uncertainty.evaluation.binary_auroc(clean, corrupted)`, `sklearn.metrics.average_precision_score`, and (in tests only) `scores.contrastive_parts` plus `uq_detr.fit_lambda`.
- Produces:
  - Constants: `COVERAGES`, `UQ_DETR_LAMBDA_GRID`.
  - Separation:
    - `auroc(clean, degraded)`
    - `aupr(clean, degraded)`
    - `fpr_at_95_tpr(clean, degraded)`
    - `condition_aurocs(clean (n,), degraded (c, n)) -> (c,)`
  - Correlation:
    - `spearman(x, y)`
    - `mean_within_condition_spearman(delta_scores (n, c), delta_risks (n, c))`
  - Risk–coverage:
    - `risk_coverage(scores, risks, coverages=COVERAGES) -> (coverages, kept_risk)`
    - `aurc(scores, risks, coverages=COVERAGES)`
  - λ fitting:
    - `fit_lambda_from_parts(conf_pos, conf_neg, reliability, grid=UQ_DETR_LAMBDA_GRID) -> (lam, pcc)`
    - `cross_fit_lambda(conf_pos, conf_neg, reliability, folds) -> (per_image_lambda (n,), per_fold dict)`
  - Inference:
    - `bootstrap(statistic, n_images, samples=1000, seed=44) -> dict[str, (low, high)]`, where `statistic(indices) -> dict[str, float]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/differential_uncertainty/test_baselines_metrics.py
import numpy as np
import pytest
import uq_detr

from differential_uncertainty.baselines import metrics as m
from differential_uncertainty.baselines import scores


def test_aupr_is_one_for_perfect_separation_and_half_for_identical_scores():
    assert m.aupr([0.0, 1.0], [2.0, 3.0]) == pytest.approx(1.0)
    assert m.aupr([1.0, 1.0], [1.0, 1.0]) == pytest.approx(0.5)


def test_fpr_at_95_tpr_uses_the_threshold_that_keeps_95_percent_of_degraded():
    degraded = np.arange(100, 200, dtype=float)
    clean = np.array([150.0, 50.0, 104.0, 106.0])
    assert m.fpr_at_95_tpr(clean, degraded) == pytest.approx(0.5)


def test_condition_aurocs_match_binary_auroc_row_by_row():
    rng = np.random.default_rng(0)
    clean, degraded = rng.normal(0, 1, 50), rng.normal(0.5, 1, (3, 50))
    expected = [m.auroc(clean, row) for row in degraded]
    assert m.condition_aurocs(clean, degraded) == pytest.approx(expected)


def test_spearman_drops_nan_pairs_and_needs_three_points():
    assert m.spearman([1, 2, 3, np.nan], [2, 4, 6, 1]) == pytest.approx(1.0)
    assert np.isnan(m.spearman([1, 2], [1, 2]))


def test_mean_within_condition_spearman_skips_conditions_without_defined_risk():
    delta_scores = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    delta_risks = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
    assert m.mean_within_condition_spearman(delta_scores, delta_risks) == pytest.approx(1.0)


def test_risk_coverage_keeps_lowest_scores_ignores_undefined_risk_and_oracle_is_best():
    scores_ = np.array([0.1, 0.2, 0.9, 0.8, 0.5])
    risks = np.array([0.0, 0.1, 1.0, 0.2, np.nan])
    _, kept = m.risk_coverage(scores_, risks, coverages=(1.0, 0.5))
    assert kept.tolist() == pytest.approx([np.nanmean(risks), 0.05])
    assert m.aurc(risks, risks, coverages=(1.0, 0.5)) <= m.aurc(scores_, risks, coverages=(1.0, 0.5))


def test_fit_lambda_from_parts_matches_uq_detr_fit_lambda():
    rng = np.random.default_rng(0)
    queries, reliability = [], []
    for _ in range(40):
        probs = np.full((20, 3), 1e-4)
        probs[:, 0] = rng.uniform(0, 1, 20)
        boxes = np.tile([0.5, 0.5, 0.2, 0.2], (20, 1))
        queries.append(uq_detr.Detections.from_cxcywh(boxes, probs, image_size=(100, 100)))
        reliability.append(rng.uniform(0, 1))
    reliability = np.array(reliability)
    reliability[3] = np.nan
    expected = uq_detr.fit_lambda(queries, reliability, method="threshold", param=0.3)
    conf_pos, conf_neg = scores.contrastive_parts(queries, 0.3)
    assert m.fit_lambda_from_parts(conf_pos, conf_neg, reliability) == pytest.approx(expected)


def test_cross_fit_lambda_never_uses_the_images_of_its_own_fold():
    rng = np.random.default_rng(3)
    conf_pos, conf_neg = rng.uniform(0, 1, 100), rng.uniform(0, 0.1, 100)
    folds = np.repeat([0, 1], 50)
    reliability = np.where(folds == 1, conf_pos - 20 * conf_neg, conf_pos) + rng.normal(0, 1e-3, 100)
    per_image, per_fold = m.cross_fit_lambda(conf_pos, conf_neg, reliability, folds)
    assert per_fold == {0: 20.0, 1: 0.0}
    assert set(per_image[folds == 0]) == {20.0} and set(per_image[folds == 1]) == {0.0}


def test_bootstrap_intervals_bracket_the_point_estimate():
    rng = np.random.default_rng(0)
    clean, degraded = rng.normal(0, 1, 200), rng.normal(1, 1, (2, 200))
    point = m.condition_aurocs(clean, degraded).mean()
    intervals = m.bootstrap(lambda idx: {"mean_auroc": m.condition_aurocs(clean[idx], degraded[:, idx]).mean()},
                            200, samples=200, seed=1)
    low, high = intervals["mean_auroc"]
    assert low <= point <= high
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/differential_uncertainty/test_baselines_metrics.py -v`
Expected: FAIL with `ImportError: cannot import name 'metrics'`.

- [ ] **Step 3: Write the implementation**

```python
# differential_uncertainty/baselines/metrics.py
"""Separation, harm and inference helpers; every score is oriented so higher means more degraded."""
from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.stats import ConstantInputWarning, pearsonr, rankdata, spearmanr
from sklearn.metrics import average_precision_score

from ..evaluation import binary_auroc

COVERAGES = tuple(np.round(np.linspace(1.0, 0.05, 20), 4))
UQ_DETR_LAMBDA_GRID = (0, 0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20)  # uq_detr.fit_lambda default


def auroc(clean, degraded) -> float:
    return binary_auroc(clean, degraded)


def aupr(clean, degraded) -> float:
    """Average precision with degraded images as the positive class."""
    clean, degraded = np.asarray(clean, float), np.asarray(degraded, float)
    labels = np.concatenate([np.zeros(clean.size), np.ones(degraded.size)])
    return float(average_precision_score(labels, np.concatenate([clean, degraded])))


def fpr_at_95_tpr(clean, degraded) -> float:
    degraded = np.sort(np.asarray(degraded, float))
    threshold = degraded[int(math.floor(0.05 * degraded.size))]
    return float(np.mean(np.asarray(clean, float) >= threshold))


def condition_aurocs(clean, degraded) -> np.ndarray:
    """AUROC of clean (n,) against each row of degraded (c, n), vectorised with average ranks."""
    clean, degraded = np.asarray(clean, float), np.asarray(degraded, float)
    n, m = clean.size, degraded.shape[1]
    joined = np.concatenate([np.broadcast_to(clean, (degraded.shape[0], n)), degraded], axis=1)
    ranks = rankdata(joined, method="average", axis=1)
    return (ranks[:, n:].sum(axis=1) - m * (m + 1) / 2) / (n * m)


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        return float(spearmanr(x[keep], y[keep]).statistic)


def mean_within_condition_spearman(delta_scores, delta_risks) -> float:
    """Mean over conditions (columns) of the per-image Spearman correlation."""
    values = [spearman(delta_scores[:, c], delta_risks[:, c]) for c in range(delta_scores.shape[1])]
    values = [v for v in values if not math.isnan(v)]
    return float(np.mean(values)) if values else math.nan


def risk_coverage(scores, risks, coverages=COVERAGES):
    """Mean risk of the images kept when the highest-scoring ones are rejected first."""
    scores, risks = np.asarray(scores, float), np.asarray(risks, float)
    keep = np.isfinite(risks)
    ranked = risks[keep][np.argsort(scores[keep], kind="stable")]
    kept = [ranked[: max(1, int(math.ceil(c * ranked.size)))].mean() for c in coverages]
    return np.asarray(coverages, float), np.asarray(kept, float)


def aurc(scores, risks, coverages=COVERAGES) -> float:
    return float(risk_coverage(scores, risks, coverages)[1].mean())


def fit_lambda_from_parts(conf_pos, conf_neg, reliability, grid=UQ_DETR_LAMBDA_GRID):
    """uq_detr.fit_lambda on precomputed Conf+/Conf-: first lambda with the highest Pearson r."""
    conf_pos, conf_neg, reliability = (np.asarray(v, float) for v in (conf_pos, conf_neg, reliability))
    valid = ~np.isnan(reliability)
    if valid.sum() < 3:
        raise ValueError("need at least 3 images with a defined reliability value")
    best_lambda, best_pcc = 0.0, -np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        for lam in grid:
            pcc = pearsonr(conf_pos[valid] - lam * conf_neg[valid], reliability[valid])[0]
            if pcc > best_pcc:
                best_lambda, best_pcc = float(lam), float(pcc)
    return best_lambda, best_pcc


def cross_fit_lambda(conf_pos, conf_neg, reliability, folds):
    """Fit lambda for each fold on the other folds only."""
    conf_pos, conf_neg, reliability, folds = (np.asarray(v) for v in (conf_pos, conf_neg, reliability, folds))
    per_fold = {}
    for fold in np.unique(folds):
        others = folds != fold
        per_fold[int(fold)] = fit_lambda_from_parts(conf_pos[others], conf_neg[others], reliability[others])[0]
    return np.array([per_fold[int(f)] for f in folds]), per_fold


def bootstrap(statistic, n_images: int, samples: int = 1000, seed: int = 44) -> dict:
    """Paired whole-image bootstrap: the same resampled images feed every quantity in `statistic`."""
    generator = np.random.default_rng(seed)
    draws = [statistic(generator.integers(0, n_images, n_images)) for _ in range(samples)]
    return {key: tuple(float(v) for v in np.nanpercentile([d[key] for d in draws], (2.5, 97.5)))
            for key in draws[0]}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/differential_uncertainty/test_baselines_metrics.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit checkpoint**

```bash
git add differential_uncertainty/baselines/metrics.py tests/differential_uncertainty/test_baselines_metrics.py
git commit -m "feat: add separation, harm and cross-fitting metrics for baselines"
```

---

### Task 6: DisCoPatch adapter (import, COCO training data, training, per-image scorer)

**Files:**
- Create: `differential_uncertainty/baselines/discopatch.py`
- Test: `tests/differential_uncertainty/test_baselines_discopatch.py`

**Interfaces:**
- Consumes: the official modules in `<root>/src/discopatch`:
  - `models.DisCoPatch.DisCoPatch(input_shape, input_channels, args)` with `.train_model(loader, loader)`;
  - `models.DisCoPatch.Discriminator(input_shape, input_channels, hidden_dims, lr, batch_size)`;
  - `models.DisCoPatch.Patchnorm2D(num_features, patches, eps)`.
- Produces:
  - `DEFAULT_ROOT`, `TRAIN_ARGS`
  - `import_discopatch(models_dir, root=DEFAULT_ROOT) -> module`
  - `CocoPatchDataset(paths, patches=48)`
  - `train_discopatch(paths, models_dir, *, epochs, num_workers, seed, overrides=None, root=DEFAULT_ROOT) -> Path` (the discriminator checkpoint)
  - `crop_positions(image_id, patches) -> ndarray (patches, 2)`
  - `DisCoPatchScorer(checkpoint, models_dir, device, patches=64, hidden_dims=TRAIN_ARGS["hidden_dims"], root=DEFAULT_ROOT, chunk_variants=16)` with `.score(arrays, image_id) -> ndarray (len(arrays),)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/differential_uncertainty/test_baselines_discopatch.py
import numpy as np
import pytest
import torch
from PIL import Image

from differential_uncertainty.baselines import discopatch as dp

pytestmark = pytest.mark.skipif(not dp.DEFAULT_ROOT.exists(), reason="DisCoPatch repository not available")


def test_crop_positions_are_seeded_by_image_and_stay_inside_the_256_square():
    first, again, other = dp.crop_positions("a.jpg", 64), dp.crop_positions("a.jpg", 64), dp.crop_positions("b.jpg", 64)
    assert first.shape == (64, 2) and np.array_equal(first, again) and not np.array_equal(first, other)
    assert first.min() >= 0 and first.max() < 192


def test_coco_patch_dataset_returns_normalised_64px_patches_from_grayscale_too(tmp_path):
    path = tmp_path / "x.jpg"
    Image.new("L", (320, 200), 255).save(path)
    patches, label = dp.CocoPatchDataset([path], patches=5)[0]
    assert patches.shape == (5, 3, 64, 64) and label == 0
    assert patches.max().item() == pytest.approx(1.0, abs=1e-2)


def test_scorer_normalises_each_image_separately(tmp_path):
    module = dp.import_discopatch(tmp_path, dp.DEFAULT_ROOT)
    torch.save(module.Discriminator(64, 3, [4, 8], 1e-4, 2).state_dict(), tmp_path / "disc.pt")
    scorer = dp.DisCoPatchScorer(tmp_path / "disc.pt", tmp_path, "cpu", patches=8, hidden_dims=[4, 8])
    rng = np.random.default_rng(0)
    a, b = (rng.integers(0, 256, (120, 160, 3), dtype=np.uint8) for _ in range(2))
    together, alone = scorer.score([a, b], "img.jpg"), scorer.score([a], "img.jpg")
    assert together.shape == (2,) and together[0] == pytest.approx(alone[0], abs=1e-6)
    assert np.all((together >= 0) & (together <= 1))
    assert not any(isinstance(layer, torch.nn.BatchNorm2d) for layer in scorer.discriminator.modules())


def test_training_wrapper_runs_one_tiny_epoch_and_writes_the_discriminator(tmp_path):
    paths = []
    for index in range(2):
        path = tmp_path / f"{index}.jpg"
        Image.new("RGB", (300, 260), (index * 100, 50, 200)).save(path)
        paths.append(path)
    checkpoint = dp.train_discopatch(
        paths, tmp_path / "models", epochs=1, num_workers=0, seed=0,
        overrides={"hidden_dims": [4, 8], "latent_dim": 8, "batch_size": 2, "patches": 2},
    )
    assert checkpoint.exists() and checkpoint.name == "Discriminator_coco.pt"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/differential_uncertainty/test_baselines_discopatch.py -v`
Expected: FAIL with `ImportError: cannot import name 'discopatch'`.

- [ ] **Step 3: Write the implementation**

```python
# differential_uncertainty/baselines/discopatch.py
"""Adapter around the official DisCoPatch code (Caetano et al., ICCV 2025) for COCO images."""
from __future__ import annotations

import importlib
import os
import sys
import types
import zlib
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

DEFAULT_ROOT = Path(os.environ.get("DISCOPATCH_ROOT", "/home/yuchen/YuchenZ/UE/DisCoPatch"))
IMAGE_SIDE = 256
PATCH = 64
TRAIN_ARGS = {  # README replication command for ImageNet-1K
    "batch_size": 67, "patches": 48, "hidden_dims": [128, 256, 512, 1024], "latent_dim": 1024,
    "lr": 8.5e-5, "gen_weight": 1e-3, "recon_weight": 1e-3, "kld_weight": 1e-4, "loss_type": "mse",
    "sample_and_save_frequency": 5, "no_wandb": True, "dataset": "coco", "num_samples": 16,
    "checkpoint": None, "discriminator_checkpoint": None,
}
_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIDE, IMAGE_SIDE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def import_discopatch(models_dir, root=DEFAULT_ROOT):
    """Import models.DisCoPatch without wandb and point its output folders at `models_dir`."""
    models_dir = Path(models_dir)
    (models_dir / "figures").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    if "wandb" not in sys.modules:
        stub = types.ModuleType("wandb")
        stub.log = stub.init = stub.finish = lambda *args, **kwargs: None
        sys.modules["wandb"] = stub
    source = str(Path(root) / "src" / "discopatch")
    if source not in sys.path:
        sys.path.insert(0, source)
    module = importlib.import_module("models.DisCoPatch")
    module.models_dir = str(models_dir)          # the repo's .env would otherwise win
    module.figures_dir = str(models_dir / "figures")
    return module


class CocoPatchDataset(Dataset):
    """Random 64x64 patches of 256x256-resized images, like the official ImageNet loader."""

    def __init__(self, paths, patches: int = 48):
        self.paths = list(paths)
        self.patches = patches

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as source:
            image = _TRANSFORM(source.convert("RGB"))
        corners = np.random.randint(0, IMAGE_SIDE - PATCH, size=(self.patches, 2))
        return torch.stack([image[:, x:x + PATCH, y:y + PATCH] for x, y in corners]), 0


def train_discopatch(paths, models_dir, *, epochs, num_workers, seed, overrides=None, root=DEFAULT_ROOT) -> Path:
    module = import_discopatch(models_dir, root)
    torch.manual_seed(seed)
    np.random.seed(seed)
    args = Namespace(**{**TRAIN_ARGS, **(overrides or {}), "n_epochs": epochs})
    loader = DataLoader(CocoPatchDataset(paths, patches=args.patches), batch_size=args.batch_size,
                        shuffle=True, pin_memory=True, num_workers=num_workers,
                        persistent_workers=num_workers > 0)
    model = module.DisCoPatch(input_shape=IMAGE_SIDE // 4, input_channels=3, args=args)
    model.train_model(loader, loader)
    return Path(models_dir) / "DisCoPatch" / "Discriminator_coco.pt"


def crop_positions(image_id: str, patches: int) -> np.ndarray:
    generator = np.random.default_rng(zlib.crc32(image_id.encode("utf-8")))
    return generator.integers(0, IMAGE_SIDE - PATCH, size=(patches, 2))


def _to_patchnorm(discriminator, patchnorm_class, patches):
    """Same replacement the official outlier_detection performs before scoring."""
    for block in discriminator.encoder:
        if isinstance(block, nn.Sequential):
            for index, layer in enumerate(block):
                if isinstance(layer, nn.BatchNorm2d):
                    replacement = patchnorm_class(layer.num_features, patches, eps=layer.eps, affine=True)
                    replacement.gamma = layer.weight
                    replacement.beta = layer.bias
                    block[index] = replacement


class DisCoPatchScorer:
    def __init__(self, checkpoint, models_dir, device, patches=64,
                 hidden_dims=TRAIN_ARGS["hidden_dims"], root=DEFAULT_ROOT, chunk_variants=16):
        module = import_discopatch(models_dir, root)
        discriminator = module.Discriminator(IMAGE_SIDE // 4, 3, list(hidden_dims),
                                             TRAIN_ARGS["lr"], TRAIN_ARGS["batch_size"])
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        discriminator.load_state_dict(state, strict=True)
        _to_patchnorm(discriminator, module.Patchnorm2D, patches)
        self.discriminator = discriminator.to(device).eval()
        self.device = torch.device(device)
        self.patches = patches
        self.chunk_variants = chunk_variants

    @torch.inference_mode()
    def score(self, arrays, image_id: str) -> np.ndarray:
        """Degradation score per array: 1 - mean patch output, patches normalised per image."""
        corners = crop_positions(image_id, self.patches)
        values = []
        for start in range(0, len(arrays), self.chunk_variants):
            patches = []
            for array in arrays[start:start + self.chunk_variants]:
                image = _TRANSFORM(Image.fromarray(array).convert("RGB"))
                patches.extend(image[:, x:x + PATCH, y:y + PATCH] for x, y in corners)
            output = self.discriminator(torch.stack(patches).to(self.device))
            values.append(1.0 - output.view(-1, self.patches).mean(dim=1).cpu().numpy())
        return np.concatenate(values)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/differential_uncertainty/test_baselines_discopatch.py -v`
Expected: 4 passed. The training smoke test takes under a minute.

- [ ] **Step 5: Commit checkpoint**

```bash
git add differential_uncertainty/baselines/discopatch.py tests/differential_uncertainty/test_baselines_discopatch.py
git commit -m "feat: add DisCoPatch adapter for COCO training and per-image scores"
```

---

### Task 7: Pipeline phases, CLI and entry-point guard

**Files:**
- Create: `differential_uncertainty/baselines/pipeline.py`
- Modify: `differential_uncertainty/cli.py`, which gains the `baselines-coco` subcommand.
- Modify: `differential_uncertainty/__main__.py`, which gains an `if __name__ == "__main__":` guard.
  The corruption workers use the `spawn` start method, which re-imports the entry module as
  `__mp_main__`. Without the guard, every worker would re-run the CLI.
- Test: `tests/differential_uncertainty/test_baselines_pipeline.py`

**Interfaces:**
- Consumes: Tasks 1–6.
- Produces:
  - `Settings` (a dataclass), `run_phase(phase: str, settings: Settings)`, `evaluation(settings) -> list[Path]`.
  - Phases: `sanity`, `bank`, `test`, `train-discopatch`, `discopatch-scores`, `timing`, `report`.
  - Files, all under `settings.output`:
    - `run_config.json` and `environment.json`
    - `sanity.json`
    - `bank/knn_bank.npy` (118,287 × 512 float16, unit rows)
    - `test/<stem>.npz`, with keys `TEST_KEYS`
    - `discopatch/DisCoPatch/Discriminator_coco.pt`
    - `test_dcp/<stem>.npz`, with key `dcp`
    - `timing.json`

- [ ] **Step 1: Write the failing tests**

```python
# tests/differential_uncertainty/test_baselines_pipeline.py
import json

import numpy as np
import pytest
import torch
from PIL import Image

from differential_uncertainty.baselines import pipeline


class FakeTap:
    calls = 0

    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def run(self, arrays, batch_size=32):
        FakeTap.calls += 1
        n = len(arrays)
        brightness = np.array([a.mean() / 255.0 for a in arrays])
        logits = np.full((n, 300, 80), -6.0)
        logits[:, :3, 0] = (4.0 * brightness)[:, None]
        boxes = np.full((n, 300, 4), 0.25)
        pooled = np.random.default_rng(0).normal(size=(n, 512)) + brightness[:, None]
        return logits, boxes, pooled


def _settings(tmp_path, **changes):
    val = tmp_path / "val"
    val.mkdir(exist_ok=True)
    rng = np.random.default_rng(1)
    for index in range(4):
        Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)).save(val / f"{index:04d}.jpg")
    values = dict(output=tmp_path / "out", checkpoint=tmp_path / "ckpt.pth", train_images=tmp_path,
                  val_images=val, annotations=tmp_path / "ann.json", discopatch_root=tmp_path,
                  limit=2, workers=0, device="cpu", batch_size=16)
    values.update(changes)
    return pipeline.Settings(**values)


@pytest.fixture
def fakes(monkeypatch):
    FakeTap.calls = 0
    monkeypatch.setattr(pipeline, "DetectorTap", FakeTap)
    monkeypatch.setattr(pipeline, "_load_bank", lambda _s, _d: torch.nn.functional.normalize(torch.randn(256, 512), dim=1))


def test_test_phase_writes_one_complete_file_per_image_and_resumes(tmp_path, fakes):
    settings = _settings(tmp_path)
    pipeline.run_phase("test", settings)
    files = sorted((settings.output / "test").glob("*.npz"))
    assert len(files) == 2
    data = np.load(files[0])
    assert set(pipeline.TEST_KEYS) <= set(data.files)
    assert data["saod_top3"].shape == (96,) and data["knn"].shape == (96, 200)
    assert data["det_boxes"].shape == (96, 100, 4) and data["digests"].shape == (96,)
    FakeTap.calls = 0
    pipeline.run_phase("test", settings)
    assert FakeTap.calls == 0


def test_evaluation_uses_the_seeded_shuffle_and_the_limit(tmp_path):
    settings = _settings(tmp_path)
    names = [p.name for p in pipeline.evaluation(settings)]
    assert len(names) == 2 and set(names) <= {f"{i:04d}.jpg" for i in range(4)}


def test_test_phase_refuses_a_damaged_result_file(tmp_path, fakes):
    settings = _settings(tmp_path)
    pipeline.run_phase("test", settings)
    victim = sorted((settings.output / "test").glob("*.npz"))[0]
    victim.write_bytes(b"not an npz")
    with pytest.raises(ValueError, match=victim.name):
        pipeline.run_phase("test", settings)


def test_run_config_mismatch_stops_the_run_and_environment_is_recorded(tmp_path, fakes):
    pipeline.run_phase("test", _settings(tmp_path))
    environment = json.loads((tmp_path / "out" / "environment.json").read_text())
    assert "torch" in environment["packages"]
    with pytest.raises(ValueError, match="run_config.json"):
        pipeline.run_phase("test", _settings(tmp_path, seed=45))


def test_discopatch_phase_detects_changed_corruptions(tmp_path, fakes, monkeypatch):
    settings = _settings(tmp_path)
    pipeline.run_phase("test", settings)
    checkpoint = settings.output / "discopatch" / "DisCoPatch" / "Discriminator_coco.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"x")

    class FakeScorer:
        def __init__(self, *_a, **_k):
            pass

        def score(self, arrays, _image_id):
            return np.array([a.mean() / 255.0 for a in arrays])

    monkeypatch.setattr(pipeline, "DisCoPatchScorer", FakeScorer)
    victim = sorted((settings.output / "test").glob("*.npz"))[0]
    data = dict(np.load(victim))
    data["digests"] = np.array(["0" * 16] * 96)
    np.savez(victim, **data)
    with pytest.raises(ValueError, match="corruptions differ"):
        pipeline.run_phase("discopatch-scores", settings)


def test_entry_module_does_not_run_the_cli_when_imported_by_a_spawned_worker():
    import runpy
    runpy.run_module("differential_uncertainty.__main__", run_name="__mp_main__")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/differential_uncertainty/test_baselines_pipeline.py -v`
Expected: FAIL with `ImportError: cannot import name 'pipeline'`. The entry-module test fails with
`SystemExit` until the guard is added.

- [ ] **Step 3: Write the implementation**

```python
# differential_uncertainty/baselines/pipeline.py
"""Resumable phases that compute the four baseline scores on the fixed COCO protocol."""
from __future__ import annotations

import json
import multiprocessing
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..extraction import prepare_image
from . import protocol
from .coco_quality import CocoGroundTruth, coco_map, coco_results
from .detector import DetectorTap
from .discopatch import DisCoPatchScorer, train_discopatch
from .scores import (contrastive_parts, knn_distances, normalize_rows, query_detections,
                     saod_uncertainty, top_detections)

TOP_K = 100
KNN_K = 100
KNN_K_MAX = 200
THETA = 0.3
TIMING_IMAGES = 100
TIMING_WARMUP = 10
TEST_KEYS = ("saod_min", "saod_top3", "conf_pos", "conf_neg", "knn",
             "det_scores", "det_labels", "det_boxes", "digests", "size")
_PACKAGES = ("torch", "torchvision", "numpy", "scipy", "scikit-image", "scikit-learn",
             "imagecorruptions", "uq-detr", "pycocotools", "Pillow")


@dataclass(frozen=True)
class Settings:
    output: Path
    checkpoint: Path
    train_images: Path
    val_images: Path
    annotations: Path
    discopatch_root: Path
    limit: Optional[int] = None
    seed: int = 44
    epochs: int = 65
    device: str = "cuda:0"
    batch_size: int = 32
    workers: int = 9

    def experiment(self) -> dict:
        return {
            "checkpoint": str(self.checkpoint), "train_images": str(self.train_images),
            "val_images": str(self.val_images), "annotations": str(self.annotations),
            "discopatch_root": str(self.discopatch_root), "limit": self.limit, "seed": self.seed,
            "epochs": self.epochs, "folds": protocol.FOLDS, "top_k": TOP_K, "knn_k": KNN_K,
            "knn_k_max": KNN_K_MAX, "theta": THETA,
            "conditions": [list(c) for c in protocol.CONDITIONS],
        }


def evaluation(settings: Settings) -> list[Path]:
    images = protocol.evaluation_images(settings.val_images, seed=settings.seed)
    return images[: settings.limit] if settings.limit else images


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def _load_npz(path: Path, keys) -> dict:
    try:
        with np.load(path, allow_pickle=False) as data:
            loaded = {key: data[key] for key in data.files}
    except Exception as error:
        raise ValueError(f"malformed result file {path.name}: {path}") from error
    missing = set(keys) - set(loaded)
    if missing:
        raise ValueError(f"result file {path.name} lacks {sorted(missing)}: {path}")
    return loaded


def ensure_run_config(settings: Settings) -> None:
    path = settings.output / "run_config.json"
    expected = settings.experiment()
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError(f"existing run_config.json does not match this run: {path}")
    else:
        _atomic_json(path, expected)


def _record_environment(settings: Settings) -> None:
    path = settings.output / "environment.json"
    if path.exists():
        return
    versions = {}
    for package in _PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    commit = subprocess.run(["git", "-C", str(settings.discopatch_root), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    _atomic_json(path, {"packages": versions, "discopatch_commit": commit, "gpu": gpu})


def _open(path) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(source.convert("RGB"), dtype=np.uint8).copy()


def _size(array) -> tuple[int, int]:
    return array.shape[1], array.shape[0]


def _load_bank(settings: Settings, device) -> torch.Tensor:
    path = settings.output / "bank" / "knn_bank.npy"
    if not path.exists():
        raise ValueError(f"run the bank phase first: {path} is missing")
    return normalize_rows(torch.from_numpy(np.load(path)).float()).to(device).half()


def _progress(label, done, total, started):
    rate = done / max(time.time() - started, 1e-9)
    remaining = (total - done) / max(rate, 1e-9)
    print(f"[{label}] {done}/{total} images, {rate:.2f}/s, about {remaining / 60:.0f} min left", flush=True)


def _bounded(pool, function, items, in_flight):
    """Ordered results with at most `in_flight` tasks queued, so memory stays bounded."""
    iterator = iter(items)
    # range first: zip stops on range without pulling (and losing) an extra item
    queue = deque(pool.apply_async(function, (item,)) for _, item in zip(range(in_flight), iterator))
    while queue:
        result = queue.popleft().get()
        following = next(iterator, None)
        if following is not None:
            queue.append(pool.apply_async(function, (following,)))
        yield result


def _variant_stream(settings: Settings, paths):
    if settings.workers == 0:
        for path in paths:
            yield protocol.load_variants(path)
        return
    context = multiprocessing.get_context("spawn")
    with context.Pool(settings.workers) as pool:
        yield from _bounded(pool, protocol.load_variants, paths, 2 * settings.workers)


def _detector_scores(tap, bank, arrays, batch_size, device) -> dict:
    logits, boxes, pooled = tap.run(arrays, batch_size)
    size = _size(arrays[0])
    detections = [top_detections(l, b, size, TOP_K) for l, b in zip(logits, boxes)]
    conf_pos, conf_neg = contrastive_parts([query_detections(l, b, size) for l, b in zip(logits, boxes)], THETA)
    knn = knn_distances(torch.from_numpy(pooled).to(device), bank, KNN_K_MAX).cpu().numpy()
    return {
        "saod_min": np.array([saod_uncertainty(d[0], 1) for d in detections]),
        "saod_top3": np.array([saod_uncertainty(d[0], 3) for d in detections]),
        "conf_pos": conf_pos, "conf_neg": conf_neg, "knn": knn.astype(np.float32),
        "det_scores": np.stack([d[0] for d in detections]),
        "det_labels": np.stack([d[1] for d in detections]).astype(np.int16),
        "det_boxes": np.stack([d[2] for d in detections]),
        "size": np.array(size),
    }


def phase_sanity(settings: Settings) -> float:
    gt = CocoGroundTruth(settings.annotations)
    images = protocol.list_images(settings.val_images)
    results = []
    with DetectorTap(settings.checkpoint, settings.device) as tap:
        for start in range(0, len(images), settings.batch_size):
            chunk = images[start:start + settings.batch_size]
            arrays = [_open(p) for p in chunk]
            logits, boxes, _ = tap.run(arrays, settings.batch_size)
            for path, array, l, b in zip(chunk, arrays, logits, boxes):
                s, labels, xyxy = top_detections(l, b, _size(array), TOP_K)
                results += coco_results(gt.image_id(path.name), s, labels, xyxy, gt.category_ids)
    ap = coco_map(gt, results, [gt.image_id(p.name) for p in images])
    _atomic_json(settings.output / "sanity.json", {"coco_val_ap": ap, "images": len(images)})
    print(f"[sanity] clean COCO val AP = {ap:.4f} on {len(images)} images", flush=True)
    if ap < 0.45:
        raise RuntimeError(f"clean COCO val AP is {ap:.3f}; expected about 0.48 for this checkpoint")
    return ap


class _Prepared(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as source:
            return prepare_image(source.convert("RGB"), (640, 640))


def phase_bank(settings: Settings) -> None:
    path = settings.output / "bank" / "knn_bank.npy"
    if path.exists():
        return
    paths = protocol.list_images(settings.train_images)
    loader = DataLoader(_Prepared(paths), batch_size=settings.batch_size,
                        num_workers=settings.workers, pin_memory=True)
    features, started = [], time.time()
    with DetectorTap(settings.checkpoint, settings.device) as tap:
        for index, batch in enumerate(loader):
            _, _, pooled = tap.forward(batch)
            features.append(normalize_rows(torch.from_numpy(pooled)).numpy().astype(np.float16))
            if index % 200 == 0:
                _progress("bank", min((index + 1) * settings.batch_size, len(paths)), len(paths), started)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".knn_bank.tmp.npy")
    np.save(temporary, np.concatenate(features))
    temporary.replace(path)
    _atomic_json(path.with_name("bank_names.json"), [p.name for p in paths])


def _valid_existing(path: Path, keys) -> bool:
    if not path.exists():
        return False
    _load_npz(path, keys)
    return True


def phase_test(settings: Settings) -> None:
    folder = settings.output / "test"
    pending = [p for p in evaluation(settings) if not _valid_existing(folder / f"{p.stem}.npz", TEST_KEYS)]
    if not pending:
        return
    bank = _load_bank(settings, settings.device)
    started = time.time()
    with DetectorTap(settings.checkpoint, settings.device) as tap:
        for done, (name, arrays) in enumerate(_variant_stream(settings, pending), start=1):
            values = _detector_scores(tap, bank, arrays, settings.batch_size, settings.device)
            values["digests"] = np.array([protocol.digest(a) for a in arrays])
            _atomic_npz(folder / f"{Path(name).stem}.npz", **values)
            if done % 25 == 0:
                _progress("test", done, len(pending), started)


def phase_train_discopatch(settings: Settings) -> None:
    train_discopatch(protocol.list_images(settings.train_images), settings.output / "discopatch",
                     epochs=settings.epochs, num_workers=settings.workers, seed=settings.seed,
                     root=settings.discopatch_root)


def _discopatch_checkpoint(settings: Settings) -> Path:
    return settings.output / "discopatch" / "DisCoPatch" / "Discriminator_coco.pt"


def phase_discopatch_scores(settings: Settings) -> None:
    checkpoint = _discopatch_checkpoint(settings)
    if not checkpoint.exists():
        raise ValueError(f"train DisCoPatch first: {checkpoint} is missing")
    scorer = DisCoPatchScorer(checkpoint, settings.output / "discopatch", settings.device,
                              root=settings.discopatch_root)
    folder = settings.output / "test_dcp"
    pending = [p for p in evaluation(settings) if not _valid_existing(folder / f"{p.stem}.npz", ("dcp",))]
    missing = [p.name for p in pending if not (settings.output / "test" / f"{p.stem}.npz").exists()]
    if missing:
        raise ValueError(f"run the test phase first: {len(missing)} detector results are missing, e.g. {missing[0]}")
    started = time.time()
    for done, (name, arrays) in enumerate(_variant_stream(settings, pending), start=1):
        stem = Path(name).stem
        stored = _load_npz(settings.output / "test" / f"{stem}.npz", TEST_KEYS)["digests"]
        if list(stored) != [protocol.digest(a) for a in arrays]:
            raise ValueError(f"corruptions differ from the detector pass for {name}")
        _atomic_npz(folder / f"{stem}.npz", dcp=scorer.score(arrays, name))
        if done % 25 == 0:
            _progress("discopatch", done, len(pending), started)


def phase_timing(settings: Settings) -> None:
    """Median ms per image at batch 1 on a warm GPU, preprocessing included."""
    images = [_open(p) for p in evaluation(settings)[:TIMING_IMAGES]]
    bank = _load_bank(settings, settings.device)
    on_gpu = torch.cuda.is_available() and settings.device.startswith("cuda")

    def timed(function):
        for array in images[:TIMING_WARMUP]:
            function(array)
        values = []
        for array in images:
            if on_gpu:
                torch.cuda.synchronize()
            start = time.perf_counter()
            function(array)
            if on_gpu:
                torch.cuda.synchronize()
            values.append(1000.0 * (time.perf_counter() - start))
        return float(np.median(values))

    result = {"images": len(images), "batch_size": 1, "device": settings.device}
    with DetectorTap(settings.checkpoint, settings.device) as tap:
        def detector(array):
            return tap.forward(tap.prepare([array]))

        def saod(array):
            logits, boxes, _ = detector(array)
            saod_uncertainty(top_detections(logits[0], boxes[0], _size(array), TOP_K)[0], 3)

        def contrastive(array):
            logits, boxes, _ = detector(array)
            contrastive_parts([query_detections(logits[0], boxes[0], _size(array))], THETA)

        def knn(array):
            _, _, pooled = detector(array)
            knn_distances(torch.from_numpy(pooled).to(settings.device), bank, KNN_K)

        # every *_plus_* figure includes the detector forward pass it depends on
        result.update(detector_ms=timed(detector), detector_plus_saod_ms=timed(saod),
                      detector_plus_contrastive_ms=timed(contrastive), detector_plus_knn_ms=timed(knn))
    checkpoint = _discopatch_checkpoint(settings)
    if checkpoint.exists():
        scorer = DisCoPatchScorer(checkpoint, settings.output / "discopatch", settings.device,
                                  root=settings.discopatch_root)
        result["discopatch_ms"] = timed(lambda array: scorer.score([array], "timing.jpg"))
    _atomic_json(settings.output / "timing.json", result)
    print(f"[timing] {result}", flush=True)


def phase_report(settings: Settings) -> None:
    from .report import build_report
    build_report(settings)


PHASES = {
    "sanity": phase_sanity, "bank": phase_bank, "test": phase_test,
    "train-discopatch": phase_train_discopatch, "discopatch-scores": phase_discopatch_scores,
    "timing": phase_timing, "report": phase_report,
}


def run_phase(phase: str, settings: Settings) -> None:
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}; choose from {sorted(PHASES)}")
    settings.output.mkdir(parents=True, exist_ok=True)
    ensure_run_config(settings)
    _record_environment(settings)
    PHASES[phase](settings)
```

Guard the entry point in `differential_uncertainty/__main__.py`:

```python
from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
```

Add the subcommand to `differential_uncertainty/cli.py`. Put this inside `build_parser()`, before `return parser`:

```python
    baselines = commands.add_parser("baselines-coco")
    baselines.add_argument("--phase", required=True,
                           choices=["sanity", "bank", "test", "train-discopatch", "discopatch-scores", "timing", "report"])
    baselines.add_argument("--output", required=True)
    baselines.add_argument("--checkpoint", required=True)
    baselines.add_argument("--coco-train-images", required=True)
    baselines.add_argument("--coco-val-images", required=True)
    baselines.add_argument("--coco-annotations", required=True)
    baselines.add_argument("--discopatch-root", required=True)
    baselines.add_argument("--limit", default=None, type=positive_int)
    baselines.add_argument("--device", default="cuda:0")
    baselines.add_argument("--batch-size", default=32, type=positive_int)
    baselines.add_argument("--workers", default=9, type=nonnegative_int)
    baselines.add_argument("--epochs", default=65, type=positive_int)
```

Then, in `main`, replace the body of the `try:` with a dispatch on `args.command`:

```python
    try:
        if args.command == "baselines-coco":
            from pathlib import Path
            from .baselines.pipeline import Settings, run_phase
            run_phase(args.phase, Settings(
                output=Path(args.output), checkpoint=Path(args.checkpoint),
                train_images=Path(args.coco_train_images), val_images=Path(args.coco_val_images),
                annotations=Path(args.coco_annotations), discopatch_root=Path(args.discopatch_root),
                limit=args.limit, device=args.device, batch_size=args.batch_size,
                workers=args.workers, epochs=args.epochs,
            ))
        else:
            run_coco_benchmark(
                args.checkpoint, args.coco_train_images, args.coco_val_images, args.output,
                reference_count=args.reference_count, evaluation_count=args.evaluation_count,
                device=args.device, batch_size=args.batch_size, seed=args.seed,
            )
    except (OSError, ValueError, RuntimeError, KeyError) as error:  # unchanged from the current cli.py
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/differential_uncertainty/test_baselines_pipeline.py tests/differential_uncertainty/test_cli.py -v`
Expected: all pass. The existing CLI tests still pass because `benchmark-coco` is unchanged.

- [ ] **Step 5: Commit checkpoint**

```bash
git add differential_uncertainty/baselines/pipeline.py differential_uncertainty/cli.py differential_uncertainty/__main__.py tests/differential_uncertainty/test_baselines_pipeline.py
git commit -m "feat: add resumable baseline pipeline and CLI"
```

---

### Task 8: Report (cross-fitted λ, tables, intervals)

**Files:**
- Create: `differential_uncertainty/baselines/report.py`
- Test: `tests/differential_uncertainty/test_baselines_report.py`

**Interfaces:**
- Consumes: the stored files from Task 7, and `metrics` and `coco_quality` from Tasks 4–5.
- Produces:
  - `METHODS`, `LABELS`
  - `method_scores(test: dict, dcp, per_image_lambda, k=100) -> dict[str, ndarray (N, 96)]`
  - `separation_rows(scores, folds, per_fold_methods=()) -> list[dict]`
  - `aggregate_rows(rows) -> list[dict]`
  - `harm_rows(scores, lrp (N, 96), condition_map (96,)) -> (harm_rows, pool_rows)`
  - `headline_numbers(scores, lrp, folds=None, per_fold_methods=()) -> dict[str, float]` (per method and pairwise differences; fold-averaged for `per_fold_methods`)
  - `write_outputs(folder, tables: dict, summary: dict) -> None`
  - `build_report(settings) -> None`
  - Files under `<output>/results/`:
    - `separation.csv`, `aggregates.csv`
    - `harm.csv`, `aurc_pools.csv`, `conditions.csv`
    - `intervals.csv`, `differences.csv`
    - `knn_k.csv`, `timing.csv`
    - `summary.json`, `report.md`

- [ ] **Step 1: Write the failing tests**

```python
# tests/differential_uncertainty/test_baselines_report.py
import json

import numpy as np
import pytest

from differential_uncertainty.baselines import protocol, report

SEVERITY = np.array([s for _, s in protocol.CONDITIONS], float)


def _scores(n=30):
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, (n, 1))
    return {m: base + SEVERITY[None, :] * 2.0 + rng.normal(0, 0.1, (n, 96)) for m in report.METHODS}


def test_separation_rows_cover_every_method_and_condition_with_the_right_pooling():
    folds = protocol.assign_folds(30)
    rows = report.separation_rows(_scores(), folds, per_fold_methods=("contrastive",))
    assert len(rows) == len(report.METHODS) * 95
    assert {r["pooling"] for r in rows if r["method"] == "contrastive"} == {"fold-averaged"}
    assert {r["pooling"] for r in rows if r["method"] == "knn"} == {"pooled"}
    assert all(r["auroc"] > 0.9 and r["aupr"] > 0.9 for r in rows if r["severity"] >= 3)


def test_aggregate_rows_average_common_and_extra_families_separately():
    rows = report.separation_rows(_scores(), protocol.assign_folds(30))
    keys = {(r["method"], r["group"], r["severity"]) for r in report.aggregate_rows(rows)}
    assert {("knn", "common", "all"), ("knn", "extra", 5), ("knn", "all", "all")} <= keys


def test_harm_rows_correlations_and_aurc_pools():
    scores = _scores()
    condition_map = 0.5 - 0.05 * SEVERITY
    lrp = np.tile(1.0 - condition_map, (30, 1)) + np.random.default_rng(2).normal(0, 0.01, (30, 96))
    harm, pools = report.harm_rows(scores, lrp, condition_map)
    knn = next(r for r in harm if r["method"] == "knn")
    assert knn["rho_condition_map"] < -0.9 and knn["rho_condition_lrp"] > 0.9
    assert len(pools) == len(report.METHODS) * (1 + 5 + 19)
    assert all(p["aurc"] >= p["aurc_oracle"] - 1e-12 for p in pools)


def test_headline_numbers_include_pairwise_differences():
    rng = np.random.default_rng(4)
    numbers = report.headline_numbers(_scores(), rng.uniform(0, 1, (30, 96)))
    difference = numbers["saod_top3 - knn:auroc_common"]
    assert difference == pytest.approx(numbers["saod_top3:auroc_common"] - numbers["knn:auroc_common"])


def test_headline_numbers_fold_average_the_methods_asked_for():
    scores, folds = _scores(), protocol.assign_folds(30)
    lrp = np.random.default_rng(5).uniform(0, 1, (30, 96))
    numbers = report.headline_numbers(scores, lrp, folds, per_fold_methods=("contrastive",))
    values = scores["contrastive"]
    expected = np.mean([report.metrics.condition_aurocs(values[folds == f, 0], values[folds == f][:, report.COMMON].T).mean()
                        for f in range(5)])
    assert numbers["contrastive:auroc_common"] == pytest.approx(expected)


def test_write_outputs_creates_csv_json_and_markdown(tmp_path):
    rows = report.separation_rows(_scores(), protocol.assign_folds(30))
    report.write_outputs(tmp_path, {"separation": rows, "aggregates": report.aggregate_rows(rows)},
                         {"lambda_per_fold": {"0": 5.0}, "lrp_threshold": 0.3})
    assert (tmp_path / "separation.csv").exists()
    assert json.loads((tmp_path / "summary.json").read_text())["lrp_threshold"] == 0.3
    text = (tmp_path / "report.md").read_text()
    assert "ContrastiveConf" in text and "DisCoPatch" in text and "AUPR" in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/differential_uncertainty/test_baselines_report.py -v`
Expected: FAIL with `ImportError: cannot import name 'report'`.

- [ ] **Step 3: Write the implementation**

```python
# differential_uncertainty/baselines/report.py
"""Turn stored baseline scores into separation, harm, interval and runtime tables."""
from __future__ import annotations

import csv
import json
import math
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np

from . import metrics, protocol
from .coco_quality import (CocoGroundTruth, coco_map, coco_results, image_lrp, per_image_ap,
                           select_lrp_threshold)

METHODS = ("saod_top3", "saod_min", "contrastive", "knn", "discopatch")
LABELS = {"saod_top3": "SAOD, mean of top 3", "saod_min": "SAOD, min (1 − max confidence)",
          "contrastive": "ContrastiveConf", "knn": "kNN (k = 100)", "discopatch": "DisCoPatch"}
KNN_K = 100
KNN_KS = (1, 10, 50, 100, 200)
SEPARATION = ("auroc", "aupr", "fpr95")
BOOTSTRAP_SAMPLES = 1000
DIFFERENCE_METRICS = ("auroc_common", "auroc_extra", "aupr_common", "fpr95_common",
                      "rho_within", "rho_condition_lrp", "aurc_all")
SEVERITY = np.array([s for _, s in protocol.CONDITIONS])
CORRUPTED = np.arange(1, len(protocol.CONDITIONS))
COMMON = np.array([c for c, (f, _) in enumerate(protocol.CONDITIONS) if f in protocol.COMMON_FAMILIES])
EXTRA = np.array([c for c, (f, _) in enumerate(protocol.CONDITIONS) if f in protocol.EXTRA_FAMILIES])
TEST_ARRAYS = ("saod_min", "saod_top3", "conf_pos", "conf_neg", "knn", "det_scores", "det_labels", "det_boxes")


def method_scores(test: dict, dcp, per_image_lambda, k: int = KNN_K) -> dict:
    scores = {"saod_top3": test["saod_top3"], "saod_min": test["saod_min"],
              "contrastive": -(test["conf_pos"] - np.asarray(per_image_lambda)[:, None] * test["conf_neg"]),
              "knn": test["knn"][:, :, k - 1]}
    if dcp is not None:
        scores["discopatch"] = dcp
    return scores


def _separation(clean, degraded) -> dict:
    return {"auroc": metrics.auroc(clean, degraded), "aupr": metrics.aupr(clean, degraded),
            "fpr95": metrics.fpr_at_95_tpr(clean, degraded)}


def separation_rows(scores: dict, folds, per_fold_methods=()) -> list[dict]:
    folds = np.asarray(folds)
    rows = []
    for method, values in scores.items():
        by_fold = method in per_fold_methods
        for c in CORRUPTED:
            family, severity = protocol.CONDITIONS[c]
            if by_fold:
                parts = [_separation(values[folds == f, 0], values[folds == f, c]) for f in np.unique(folds)]
                result = {key: float(np.mean([p[key] for p in parts])) for key in SEPARATION}
            else:
                result = _separation(values[:, 0], values[:, c])
            rows.append({"method": method, "family": family, "severity": int(severity),
                         "group": "common" if family in protocol.COMMON_FAMILIES else "extra",
                         "pooling": "fold-averaged" if by_fold else "pooled", **result})
    return rows


def aggregate_rows(rows: list[dict]) -> list[dict]:
    out = []
    for method in dict.fromkeys(r["method"] for r in rows):
        for group in ("common", "extra", "all"):
            for severity in (*protocol.SEVERITIES, "all"):
                chosen = [r for r in rows if r["method"] == method
                          and (group == "all" or r["group"] == group)
                          and (severity == "all" or r["severity"] == severity)]
                out.append({"method": method, "group": group, "severity": severity,
                            **{key: float(np.mean([r[key] for r in chosen])) for key in SEPARATION}})
    return out


def _pools():
    yield "all", np.arange(len(protocol.CONDITIONS))
    for s in protocol.SEVERITIES:
        yield f"severity {s}", np.r_[0, CORRUPTED[SEVERITY[CORRUPTED] == s]]
    for family in protocol.FAMILIES:
        yield family, np.r_[0, [c for c in CORRUPTED if protocol.CONDITIONS[c][0] == family]]


def _nanmean_columns(values):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=0)


def harm_rows(scores: dict, lrp: np.ndarray, condition_map: np.ndarray):
    mean_lrp = _nanmean_columns(lrp)
    delta_risk = lrp[:, CORRUPTED] - lrp[:, [0]]
    harm, pools = [], []
    for method, values in scores.items():
        means = values.mean(axis=0)
        delta_score = values[:, CORRUPTED] - values[:, [0]]
        row = {"method": method,
               "rho_condition_map": metrics.spearman(means[CORRUPTED], condition_map[CORRUPTED]),
               "rho_condition_lrp": metrics.spearman(means[CORRUPTED], mean_lrp[CORRUPTED]),
               "rho_within": metrics.mean_within_condition_spearman(delta_score, delta_risk)}
        for s in protocol.SEVERITIES:
            columns = SEVERITY[CORRUPTED] == s
            row[f"rho_within_sev{s}"] = metrics.mean_within_condition_spearman(
                delta_score[:, columns], delta_risk[:, columns])
        harm.append(row)
        for pool, columns in _pools():
            pooled_scores, pooled_risk = values[:, columns].ravel(), lrp[:, columns].ravel()
            pools.append({"method": method, "pool": pool, "aurc": metrics.aurc(pooled_scores, pooled_risk),
                          "aurc_oracle": metrics.aurc(pooled_risk, pooled_risk)})
    return harm, pools


def _group_separation(clean, degraded):
    """Mean AUROC, AUPR and FPR95 over the conditions (rows) of `degraded`."""
    return (float(metrics.condition_aurocs(clean, degraded).mean()),
            float(np.mean([metrics.aupr(clean, row) for row in degraded])),
            float(np.mean([metrics.fpr_at_95_tpr(clean, row) for row in degraded])))


def headline_numbers(scores: dict, lrp: np.ndarray, folds=None, per_fold_methods=()) -> dict:
    """Aggregates used for intervals, computed identically on the full set and on each draw.

    Methods in `per_fold_methods` get fold-averaged separation numbers, matching separation_rows.
    """
    out = {}
    delta_risk = lrp[:, CORRUPTED] - lrp[:, [0]]
    mean_lrp = _nanmean_columns(lrp)
    for method, values in scores.items():
        for group, columns in (("common", COMMON), ("extra", EXTRA)):
            if method in per_fold_methods:
                parts = [_group_separation(values[folds == f, 0], values[folds == f][:, columns].T)
                         for f in np.unique(folds)]
                auroc_value, aupr_value, fpr_value = np.mean(parts, axis=0)
            else:
                auroc_value, aupr_value, fpr_value = _group_separation(values[:, 0], values[:, columns].T)
            out[f"{method}:auroc_{group}"] = float(auroc_value)
            out[f"{method}:aupr_{group}"] = float(aupr_value)
            out[f"{method}:fpr95_{group}"] = float(fpr_value)
        out[f"{method}:rho_within"] = metrics.mean_within_condition_spearman(
            values[:, CORRUPTED] - values[:, [0]], delta_risk)
        out[f"{method}:rho_condition_lrp"] = metrics.spearman(values.mean(axis=0)[CORRUPTED], mean_lrp[CORRUPTED])
        out[f"{method}:aurc_all"] = metrics.aurc(values.ravel(), lrp.ravel())
    for a, b in combinations(scores, 2):
        for metric in DIFFERENCE_METRICS:
            out[f"{a} - {b}:{metric}"] = out[f"{a}:{metric}"] - out[f"{b}:{metric}"]
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value) -> str:
    return "–" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.3f}"


def _with_ci(intervals: dict, key: str) -> str:
    row = intervals.get(key)
    return "–" if row is None else f"{_fmt(row['point'])} [{_fmt(row['low'])}, {_fmt(row['high'])}]"


def _markdown(tables: dict, summary: dict) -> str:
    intervals = {r["quantity"]: r for r in tables.get("intervals", []) + tables.get("differences", [])}
    aggregates = tables.get("aggregates", [])
    present = [m for m in METHODS if any(r["method"] == m for r in aggregates)]
    lines = ["# COCO baseline numbers", "",
             "All scores are oriented so that higher means more likely degraded. AUROC and AUPR: higher is "
             "better (chance 0.5). FPR95: lower is better. Brackets are 95% paired bootstrap intervals over "
             "images.", ""]
    for group, title in (("common", "15 common families"), ("extra", "4 extra families")):
        lines += [f"## Separation, {title}", "",
                  "| Method | AUROC sev 1 | sev 2 | sev 3 | sev 4 | sev 5 | AUROC all | AUPR all | FPR95 all |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for method in present:
            cells = {r["severity"]: r for r in aggregates if r["method"] == method and r["group"] == group}
            by_severity = " | ".join(_fmt(cells[s]["auroc"]) if s in cells else "–" for s in protocol.SEVERITIES)
            lines.append(f"| {LABELS[method]} | {by_severity} | "
                         f"{_with_ci(intervals, f'{method}:auroc_{group}') if intervals else _fmt(cells['all']['auroc'])} | "
                         f"{_with_ci(intervals, f'{method}:aupr_{group}') if intervals else _fmt(cells['all']['aupr'])} | "
                         f"{_with_ci(intervals, f'{method}:fpr95_{group}') if intervals else _fmt(cells['all']['fpr95'])} |")
        lines.append("")
    harm = tables.get("harm", [])
    pools = {(r["method"], r["pool"]): r for r in tables.get("aurc_pools", [])}
    if harm:
        lines += ["## Harm alignment", "",
                  "| Method | ρ(score, mAP), conditions | ρ(score, LRP), conditions | ρ(Δscore, ΔLRP), within condition | AURC all (oracle) |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for row in harm:
            m = row["method"]
            pool = pools.get((m, "all"), {})
            lines.append(f"| {LABELS[m]} | {_fmt(row['rho_condition_map'])} | "
                         f"{_with_ci(intervals, f'{m}:rho_condition_lrp')} | {_with_ci(intervals, f'{m}:rho_within')} | "
                         f"{_with_ci(intervals, f'{m}:aurc_all')} ({_fmt(pool.get('aurc_oracle'))}) |")
        lines.append("")
    differences = tables.get("differences", [])
    if differences:
        lines += ["## Differences between methods", "",
                  "| Pair | Δ AUROC common | Δ ρ within condition | Δ AURC all |", "| --- | ---: | ---: | ---: |"]
        pairs = dict.fromkeys(r["quantity"].split(":")[0] for r in differences)
        for pair in pairs:
            lines.append(f"| {pair} | {_with_ci(intervals, f'{pair}:auroc_common')} | "
                         f"{_with_ci(intervals, f'{pair}:rho_within')} | {_with_ci(intervals, f'{pair}:aurc_all')} |")
        lines.append("")
    timing = tables.get("timing", [])
    if timing:
        lines += ["## Runtime (median ms per image, batch 1)", "", "| Part | ms |", "| --- | ---: |"]
        lines += [f"| {r['part']} | {_fmt(r['ms'])} |" for r in timing] + [""]
    lines += ["## Fixed choices", ""] + [f"- {key}: {value}" for key, value in sorted(summary.items())]
    return "\n".join(lines) + "\n"


def write_outputs(folder, tables: dict, summary: dict) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        if rows:
            _write_csv(folder / f"{name}.csv", rows)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    (folder / "report.md").write_text(_markdown(tables, summary))


def _stack(folder: Path, names, keys) -> dict:
    missing = [n for n in names if not (folder / f"{Path(n).stem}.npz").exists()]
    if missing:
        raise ValueError(f"{folder.name} is incomplete: {len(missing)} of {len(names)} missing, e.g. {missing[0]}")
    columns = {key: [] for key in keys}
    for name in names:
        with np.load(folder / f"{Path(name).stem}.npz") as item:
            for key in keys:
                columns[key].append(item[key])
    return {key: np.stack(values) for key, values in columns.items()}


def build_report(settings) -> None:
    from .pipeline import evaluation
    names = [p.name for p in evaluation(settings)]
    folds = protocol.assign_folds(len(names))
    test = _stack(settings.output / "test", names, TEST_ARRAYS)
    dcp_folder = settings.output / "test_dcp"
    dcp = _stack(dcp_folder, names, ("dcp",))["dcp"] if dcp_folder.exists() else None

    gt = CocoGroundTruth(settings.annotations)
    ids = [gt.image_id(n) for n in names]
    clean_records = [(i, test["det_scores"][k, 0], test["det_labels"][k, 0], test["det_boxes"][k, 0])
                     for k, i in enumerate(ids)]
    ap = per_image_ap(gt, {i: coco_results(i, s, l, b, gt.category_ids) for i, s, l, b in clean_records}, ids)
    clean_pos, clean_neg = test["conf_pos"][:, 0], test["conf_neg"][:, 0]
    per_image_lambda, per_fold = metrics.cross_fit_lambda(clean_pos, clean_neg, ap, folds)
    consistent = len(set(per_fold.values())) == 1
    ordered = {m: v for m, v in method_scores(test, dcp, per_image_lambda).items()}
    scores = {m: ordered[m] for m in METHODS if m in ordered}

    threshold = select_lrp_threshold(clean_records, gt)
    lrp = np.full((len(names), len(protocol.CONDITIONS)), np.nan)
    for k, image_id in enumerate(ids):
        gt_boxes, gt_labels, crowd = gt.boxes(image_id)
        for c in range(len(protocol.CONDITIONS)):
            lrp[k, c] = image_lrp(test["det_scores"][k, c], test["det_labels"][k, c], test["det_boxes"][k, c],
                                  gt_boxes, gt_labels, crowd, threshold)
    condition_map = np.array([
        coco_map(gt, [r for k, i in enumerate(ids) for r in coco_results(
            i, test["det_scores"][k, c], test["det_labels"][k, c], test["det_boxes"][k, c], gt.category_ids)], ids)
        for c in range(len(protocol.CONDITIONS))
    ])

    per_fold_methods = () if consistent else ("contrastive",)
    separation = separation_rows(scores, folds, per_fold_methods=per_fold_methods)
    harm, pools = harm_rows(scores, lrp, condition_map)
    point = headline_numbers(scores, lrp, folds, per_fold_methods)

    def statistic(draw):
        # Duplicated images stay in their own fold, so a fold's lambda never sees its own images.
        lam, _ = metrics.cross_fit_lambda(clean_pos[draw], clean_neg[draw], ap[draw], folds[draw])
        drawn = {m: v[draw] for m, v in scores.items()}
        drawn["contrastive"] = -(test["conf_pos"][draw] - lam[:, None] * test["conf_neg"][draw])
        return headline_numbers({m: drawn[m] for m in scores}, lrp[draw], folds[draw], per_fold_methods)

    ranges = metrics.bootstrap(statistic, len(names), samples=BOOTSTRAP_SAMPLES, seed=settings.seed)
    interval_rows = [{"quantity": key, "point": point[key], "low": ranges[key][0], "high": ranges[key][1]}
                     for key in point]
    conditions = [{"family": f, "severity": s, "map": float(condition_map[c]),
                   "mean_lrp": float(_nanmean_columns(lrp)[c]),
                   **{f"mean_{m}": float(v[:, c].mean()) for m, v in scores.items()}}
                  for c, (f, s) in enumerate(protocol.CONDITIONS)]
    knn_k = [{"k": k, "mean_auroc_common": float(metrics.condition_aurocs(
        test["knn"][:, 0, k - 1], test["knn"][:, COMMON, k - 1].T).mean())} for k in KNN_KS]
    timing_path = settings.output / "timing.json"
    timing = []
    if timing_path.exists():
        timing = [{"part": key, "ms": value} for key, value in json.loads(timing_path.read_text()).items()
                  if key.endswith("_ms")]
    summary = {
        "images": len(names), "folds": protocol.FOLDS, "lambda_per_fold": per_fold,
        "lambda_folds_agree": consistent, "images_with_ap": int(np.isfinite(ap).sum()),
        "lrp_threshold": threshold, "images_with_undefined_clean_lrp": int(np.isnan(lrp[:, 0]).sum()),
        "clean_map": float(condition_map[0]), "knn_k": KNN_K, "theta": 0.3,
        "bootstrap_samples": BOOTSTRAP_SAMPLES, "discopatch_included": dcp is not None,
    }
    write_outputs(settings.output / "results", {
        "separation": separation, "aggregates": aggregate_rows(separation), "harm": harm,
        "aurc_pools": pools, "conditions": conditions,
        "intervals": [r for r in interval_rows if " - " not in r["quantity"]],
        "differences": [r for r in interval_rows if " - " in r["quantity"]],
        "knn_k": knn_k, "timing": timing,
    }, summary)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/differential_uncertainty/test_baselines_report.py -v`
Expected: 6 passed.

- [ ] **Step 5: Run the whole suite**

Run: `$PY -m pytest tests/differential_uncertainty -q`
Expected: all tests pass. The DisCoPatch tests are skipped if its repository is missing.

- [ ] **Step 6: Commit checkpoint**

```bash
git add differential_uncertainty/baselines/report.py tests/differential_uncertainty/test_baselines_report.py
git commit -m "feat: add baseline report tables and intervals"
```

---

### Task 9: Run the baselines and publish the numbers

**Files:**
- Create (outputs): `runs/coco-baselines/**` (gitignored), `docs/results/coco-baselines/*.csv|json`,
  `docs/coco-baseline-numbers.md`

**Interfaces:**
- Consumes: the CLI `baselines-coco` from Task 7.
- Produces: the numbers, plus a plain-language results doc.

Shared shell setup for every step:

```bash
cd /home/yuchen/YuchenZ/UE/philip_sa
PY=/home/yuchen/miniconda3/envs/UE/bin/python
OUT=runs/coco-baselines
DATA="--checkpoint /home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \
  --coco-train-images /home/yuchen/YuchenZ/Datasets/coco/train2017 \
  --coco-val-images /home/yuchen/YuchenZ/Datasets/coco/val2017 \
  --coco-annotations /home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json \
  --discopatch-root /home/yuchen/YuchenZ/UE/DisCoPatch"
ARGS="--output $OUT $DATA"
mkdir -p $OUT/logs
```

- [ ] **Step 0: Freeze the protocol before any real run**

Ask the user to commit this plan and the code from Tasks 1–8, so the protocol and analysis are fixed,
with a timestamp, before any number exists. From here on, any change is recorded in the results doc as
a deviation, with the reason.

- [ ] **Step 1: Detector sanity check (about 5 min). Stop if it fails.**

Run: `$PY -m differential_uncertainty baselines-coco --phase sanity $ARGS`
Expected: `[sanity] clean COCO val AP = 0.47…–0.49…`. If AP < 0.45 the command exits with an error.
Check the label mapping (Task 4 test) and preprocessing before going on.

- [ ] **Step 2: Start DisCoPatch training in the background (the long pole). Stop if it projects over 36 h.**

Run:
```bash
nohup $PY -m differential_uncertainty baselines-coco --phase train-discopatch --workers 6 $ARGS \
  > $OUT/logs/train-discopatch.log 2>&1 &
echo $! > $OUT/logs/train-discopatch.pid
```
After the first epoch finishes, the `Loss:` progress bar in the log shows the epoch time. The
projection is epoch time × 65, and the expectation is about 20–35 h. If the projection is above 36 h,
stop the process and ask the user whether to lower `--epochs`.

- [ ] **Step 3: Build the kNN bank (about 10–20 min; can run during training)**

Run: `$PY -m differential_uncertainty baselines-coco --phase bank --workers 6 $ARGS 2>&1 | tee $OUT/logs/bank.log`
Expected: `runs/coco-baselines/bank/knn_bank.npy` with shape (118287, 512). Check with:
`$PY -c "import numpy as np; print(np.load('$OUT/bank/knn_bank.npy', mmap_mode='r').shape)"`.

- [ ] **Step 4: Smoke run on 20 images with the real detector (about 10 min)**

Run:
```bash
SMOKE=runs/coco-baselines-smoke
mkdir -p $SMOKE/bank && cp $OUT/bank/knn_bank.npy $OUT/bank/bank_names.json $SMOKE/bank/
$PY -m differential_uncertainty baselines-coco --phase test --limit 20 --workers 9 --output $SMOKE $DATA
$PY -m differential_uncertainty baselines-coco --phase report --limit 20 --output $SMOKE $DATA
```
Expected: `runs/coco-baselines-smoke/results/report.md` with finite AUROC and AUPR values for the
detector-based baselines, and λ values from the grid. These numbers are not reported; the step only
checks the plumbing.

- [ ] **Step 5: Full test pass for the detector-based baselines (about 2–3 h, sharing the CPU with training)**

Run: `$PY -m differential_uncertainty baselines-coco --phase test --workers 9 $ARGS 2>&1 | tee $OUT/logs/test.log`
Expected: 5,000 files in `runs/coco-baselines/test/`. If it is interrupted, rerun the same command; it
resumes.

- [ ] **Step 6: DisCoPatch scores, after training finishes (about 2 h)**

Wait until `$OUT/discopatch/DisCoPatch/Discriminator_coco.pt` exists and the training process has
exited. Then run:
`$PY -m differential_uncertainty baselines-coco --phase discopatch-scores --workers 14 $ARGS 2>&1 | tee $OUT/logs/discopatch-scores.log`
Expected: 5,000 files in `test_dcp/`, and no "corruptions differ" error.

- [ ] **Step 7: Runtime (about 5 min, with nothing else running on the GPU)**

Run: `$PY -m differential_uncertainty baselines-coco --phase timing $ARGS 2>&1 | tee $OUT/logs/timing.log`
Expected: `runs/coco-baselines/timing.json` with `detector_ms`, `detector_plus_saod_ms`,
`detector_plus_contrastive_ms`, `detector_plus_knn_ms` and `discopatch_ms`. DisCoPatch runs on its own,
without the detector, so it has no `detector_plus` entry.

- [ ] **Step 8: Report (about 3 h)**

Run: `$PY -m differential_uncertainty baselines-coco --phase report $ARGS 2>&1 | tee $OUT/logs/report.log`
The time goes on three things:
- per-image LRP for 480,000 variants: about 10 min;
- 96 COCO evaluations over 5,000 images: about 1–1.5 h;
- 1,000 bootstrap draws at about 4 s each: about 1 h.

Expected in `runs/coco-baselines/results/`:
- all the tables and `report.md`;
- in `summary.json`, a `clean_map` close to the sanity AP, and `lambda_per_fold` with five values;
- the report stating whether λ agreed across folds.

- [ ] **Step 9: Publish the numbers in docs**

Copy every CSV and `summary.json` from `runs/coco-baselines/results/` to `docs/results/coco-baselines/`.
Then write `docs/coco-baseline-numbers.md` in plain language:
- **Setup and fixed choices:** 5,000 images, 5 folds, λ per fold, the LRP threshold, k.
- **Separation:** the mean AUROC, AUPR and FPR95 tables for the 15 common and the 4 extra families by
  severity, with intervals.
- **Per family:** AUROC at severities 3 and 5.
- **Harm:** the table, with within-condition ρ as the primary per-image measure, and the AURC pools.
- **Differences** between methods, with intervals.
- **Runtime.**
- **Deviations** from this plan, if any.
- **Limitations to state:**
  - one detector, COCO only;
  - DisCoPatch trained once, with a budget inferred from the paper, and with its fixed 256-pixel resize
    and crop settings;
  - ContrastiveConf's λ fitted on labelled clean images;
  - LRP at one confidence threshold, with our own crowd-ignore rule (IoA ≥ 0.5);
  - equal weighting of families and severities, where severities are not comparable across families;
  - one corruption draw per image and condition;
  - balanced accuracy not reported;
  - no interval for ρ with mAP.

Also add a line to `docs/driving-benchmark-baselines-and-metrics.md`: "COCO numbers: see
docs/coco-baseline-numbers.md".

- [ ] **Step 10: Commit checkpoint**

```bash
git add docs/results/coco-baselines docs/coco-baseline-numbers.md docs/driving-benchmark-baselines-and-metrics.md
git commit -m "docs: add COCO baseline numbers"
```

---

## Expected timeline

| Stage | Wall clock | Notes |
| --- | --- | --- |
| Tasks 1–8 (code and tests) | a few hours | no GPU runs |
| Freeze, then sanity | ~5 min | gate: AP ≥ 0.45 |
| DisCoPatch training | ~20–35 h (estimate) | gate: projection ≤ 36 h after epoch 1 |
| Bank, smoke, test | ~3–4 h | runs while DisCoPatch trains |
| DisCoPatch scores, timing | ~2 h | after training |
| Report and docs | ~3–4 h | |

The measured corruption cost is 12 s per image for all 95 versions on one core. Glass blur is 1.9 s per
severity. With 9–14 worker processes, 5,000 images take roughly 1.2–2 h of corruption time per pass,
and there are two passes.
