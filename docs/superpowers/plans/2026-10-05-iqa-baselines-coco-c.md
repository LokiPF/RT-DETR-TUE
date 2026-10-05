# Image-Quality Baselines on COCO-C Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four detector-free image-quality (IQA) baselines to the COCO-C evaluation, scored once on the 5,000 val images × 96 conditions and reported in all four detectors' reports:
- NIQE, refitted on clean COCO train, plus the published model as a sensitivity row;
- ARNIQA's KADID-10k quality;
- ARNIQA's embedding against a clean prototype;
- zero-shot CLIP-IQA.

**Architecture:**
- **Models.** `degradation_monitor/baselines/iqa.py` wraps pyiqa 0.1.16's NIQE, ARNIQA and CLIP-IQA modules, with three deliberate changes, each pinned by a test:
  - a NIQE pristine-model refit, following the authors' recipe;
  - ARNIQA as its paper evaluates it (the official `test.py`): the centre and four corner crops, 224 × 224, of the image and of its half-size version (PIL's bicubic resize), each crop's `return_embedding`, averaged over the crops;
  - CLIP-IQA as its paper and official code define it: the official `CLIPIQAFixed.forward` with the prompt pair "Good photo." / "Bad photo." (pyiqa's wrapper averages five pairs) and fp16 weights on the GPU (pyiqa's wrapper uses fp32).
- **Stages.** `degradation_monitor/stages/iqa.py` has its own CLI and four stages:
  - **fit:** builds the clean references from the 118,287 train images;
  - **pass:** scores the 96 corrupted versions of each val image once, checked against `runs/coco/`'s digests;
  - **timing:** times each model;
  - **report:** writes each detector's report with the five IQA rows into `runs/coco-iqa/reports/<detector>/`, and writes a table of the two-axis score against each row.
- **Reports.** The report stage gains extra rows and an output folder. The detectors' own run folders, `runs/coco/` included, are only read.

**Tech Stack:** Python 3.11, PyTorch 2.11, torchvision 0.26, pyiqa 0.1.16 (installed with `--no-deps`, see Global Constraints), openai-clip 1.0.1, ftfy 6.3.1, imagecorruptions, pytest. The `UE` conda environment is `/home/yuchen/miniconda3/envs/UE`.

**Spec:** `docs/superpowers/specs/2026-10-05-iqa-baselines-request.md`: the paper session's request of 5 October 2026, word for word.

## Global Constraints

- **The four baselines, each scored so that higher means more degraded** (the spec):
  1. **NIQE (Mittal, Soundararajan & Bovik, IEEE SPL 2013).**
     - Main row `niqe`: its pristine model refitted on clean COCO train images.
     - Sensitivity row `niqe_default`: the published model (the authors' `modelparameters.mat`, pyiqa's `niqe`).
     - Both rows use the same test-image features.
  2. **ARNIQA quality (Agnolucci, Galteri, Bertini & Del Bimbo, WACV 2024)**, with the KADID-10k regressor: row `arniqa`, score = −quality.
  3. **ARNIQA embedding + clean prototype (Becker, Weiss, Hübner & Arens, arXiv 2602.18394)**: row `arniqa_proto`, score = 1 − cosine similarity between the image's ARNIQA embedding and the mean embedding of clean COCO train images. It shares the encoder pass with row 2.
  4. **CLIP-IQA (Wang, Chan & Loy, AAAI 2023), zero-shot**, with the prompts "Good photo." / "Bad photo.": row `clipiqa`, score = −quality. This is not CLIP-IQA+.
- **Model details, verified on 5 October 2026** (pyiqa 0.1.16, from its source and on the CPU):
  - **NIQE's features are pyiqa's:**
    - the luma, MATLAB's Y, in [0, 255], rounded, in float64;
    - 96 × 96 blocks, with partial blocks dropped;
    - 18 features at full size and 18 at half size (imresize with antialiasing).
  - **The refit follows the authors' `estimatemodelparam.m`:**
    - per image, keep the blocks whose sharpness exceeds 0.75 × the image's sharpest block (sharpness = the block's mean local standard deviation at full size);
    - over all kept blocks of all images, take the mean and the covariance (N − 1), leaving out blocks that hold a NaN.
  - **ARNIQA follows its paper's evaluation, as the official code runs it** (miccunifi/ARNIQA, branch `main`, checked on 5 October):
    - **The paper's protocol** (`test.py`, with `data/dataset_base_iqa.py` and `utils/utils_data.py`):
      - the half-size image is PIL's `img.resize((W // 2, H // 2))`, whose default filter for RGB is bicubic;
      - from the image and from its half-size version, the centre crop and the four corner crops, 224 × 224 (`center_corners_crop`), padded with zeros where the image is smaller than 224 px (at half size, every COCO image whose short side is under 448 px);
      - ImageNet normalisation, and the encoder under `torch.cuda.amp.autocast()`;
      - the KADID-10k regressor, a Ridge regression, was fitted on these crop features, and an image's score is the mean of its five crops' scores.
    - **The two other official recipes differ from it.** The README's torch.hub example scores the whole image, with the half-size image from torchvision's `transforms.Resize` (bilinear), and says "for simplicity … In the paper, we average the scores of the center and four corners crops". `single_image_inference.py` takes the five crops, but of a bilinear half-size image. On one COCO image's 96 versions, bilinear against bicubic moved the KADID quality by up to 0.215 (Spearman 0.956). So the choice matters, and the plan follows the paper.
    - **The embedding** is the official `return_embedding` of each crop: the ResNet-50 features of the crop and of its half-size counterpart, each L2-normalised and concatenated, 4096 values. An image's embedding is the mean over its five crops. The regressor is linear (a 1 × 4096 weight and a bias), so the quality of that mean embedding equals the mean of the five crops' qualities, as `test.py` averages them.
    - **Quality** is the KADID-10k regressor on the embedding, scaled from KADID-10k's rating range to [0, 1] as pyiqa scales it; it can fall slightly outside. pyiqa's own ARNIQA scores the whole image with a half-size image made without antialiasing, so only its weights, its regressor and its scaling are used.
    - **Precision:** on the GPU the encoder runs under autocast (fp16), as all three official recipes run it, and the regressor in fp32. On the CPU, which only the tests use, everything is fp32. Task 6 compares the two.
    - **Native size:** only the half-size version is resized; the crops keep the image's own scale. ARNIQA thus sees each image through five 224 × 224 windows at each scale, not whole: on a 640 × 480 image the full-size crops cover about 80 % of it, the half-size crops all of it.
    - **The prototype** is the mean embedding of all clean train images. Becker et al. z-score their final scores; that leaves AUROC unchanged, so it is left out. They do not say how they crop or resize the images for ARNIQA; the prototype row uses the same embedding as the quality row.
  - **CLIP-IQA follows its paper and its official code** (IceClear/CLIP-IQA, branch `v2-3.8`, checked on 5 October):
    - **The paper (Sec. 2.1–2.2):** antonym prompt pairing with ["Good photo.", "Bad photo."] for overall quality; a ResNet-50 CLIP; positional embedding removed, so the image keeps its own size ("resizing and cropping … may introduce additional distortions").
    - **The official zero-shot config** (`configs/clipiqa/clipiqa_attribute_test.py`) builds `CLIPIQAFixed` with `backbone_name='RN50'` and `classnames=[['Good photo.', 'Bad photo.']]`. Its test pipeline is: load as RGB, rescale to [0, 1], normalise with CLIP's mean and std, no resize.
    - **The official `CLIPIQAFixed.forward`** (`mmedit/models/backbones/sr_backbones/coopclipiqa.py`):
      - computes `logits_per_image, _ = clip_model(image, tokens, pos_embedding=False)`, where the logits are CLIP's `logit_scale.exp()` × the cosine similarity;
      - takes the softmax over the pair;
      - scores `probs[:, 0]`, the probability of "Good photo.".
    - **Precision:** the official `build_model` converts CLIP's weights to fp16 (`convert_weights`), and the image is cast to that dtype. pyiqa's CLIP module is the same modified CLIP, with the same `pos_embedding` switch, and its `load("RN50", device)` keeps the fp16 weights on the GPU. pyiqa's CLIP-IQA wrapper, though, loads on the CPU in fp32 and averages five prompt pairs, three about blur and noise. So the plan reimplements `CLIPIQAFixed.forward` on pyiqa's `load("RN50", device)`:
      - fp16 weights on the GPU, as the official code;
      - fp32 on the CPU, which only the tests use, where it equals pyiqa's port with the paper's pair.
    - **Logit scale:** the paper's Eq. 3 writes the softmax over the raw cosines; the code multiplies them by CLIP's logit scale first. The plan follows the code. AUROC is the same either way, since both are increasing functions of the cosine difference.
  - **Each model reads the corrupted image at its native size**, through its own preprocessing: NIQE and CLIP-IQA read the whole image, ARNIQA its paper's five crops at each scale.
- **Clean reference:** all 118,287 clean COCO train images, for both the NIQE refit and the ARNIQA prototype.
  - Measured on 5 October, the fit takes about 16 ms per image on the GPU at batch 1 (NIQE 12.6 ms; ARNIQA 3.5 ms on the whole image, a little more on its ten crops), about 35–45 minutes in all. So no subset is needed.
  - A train image without a whole 96 × 96 block cannot give NIQE features. It is left out of the NIQE refit only, and the count of such images is recorded.
- **Test images:** the 5,000 COCO val images in the seed-44 order, with the 5 folds and the 96 conditions, made with `degradation_monitor.corruptions`.
  - Every image's 96 digests must equal `runs/coco/`'s.
  - One set of scores serves all four detectors' reports.
- **Report.** For each detector (RT-DETRv2-R18 from `runs/coco/`; YOLO11m, Faster R-CNN R50-FPN v2 and RF-DETR-M from `runs/coco-detectors/`), its existing report is rebuilt with the five rows added. This covers every metric and interval the report already gives:
  - AUROC, AUPR and FPR95, common and extra;
  - results by severity and by family;
  - the paired bootstrap, 1,000 draws, seed 44;
  - "two-axis minus each row", with intervals, on all images and on the untouched ones.

  The IQA reports are written under `runs/coco-iqa/reports/<detector>/`. The detectors' own reports are not changed.
- **Timing:** ms per image of each model, measured as the existing timing stage measures the detector: the first 100 evaluation images, 10 warm-up, batch 1, preprocessing included, the card otherwise idle. They are reported next to the detector's 5.89 ms (`runs/coco/timing.json`).
- **What each model saw in training**, recorded in the results doc for the paper's fairness statement:
  - **ARNIQA's encoder (rows `arniqa` and `arniqa_proto`):** trained on synthetic distortions, including blur, noise, compression, brightness, colour, contrast and spatial distortions such as pixelation (ARNIQA's `utils/utils_data.py` and `utils/distortions.py`), but no weather.
  - **ARNIQA's KADID-10k regressor:** fitted on human ratings of KADID-10k's 25 synthetic distortion types. These COCO-C families appear among them: blur (gaussian, motion, and lens ≈ defocus), noise (white ≈ gaussian, impulse, and multiplicative ≈ speckle), JPEG, pixelate, contrast, brightness and saturation.
  - **CLIP-IQA:** saw no distortions.
  - **NIQE:** sees only clean images. The refit's clean images are COCO's JPEG-compressed photos, whereas the published model was fitted on 125 pristine photos; the results doc says so wherever the two NIQE rows differ.
- **Cityscapes reuse:**
  - The dataset, its train images and its evaluation images come from the base config, and the detector runs come from the configs.
  - A Cityscapes config can reuse every stage, with the reference taken from Cityscapes train.
  - Nothing Cityscapes-specific is built now.
- **pyiqa install (needs the user's approval):** `pip install --no-deps pyiqa==0.1.16 openai-clip==1.0.1 ftfy==6.3.1` into `UE`.
  - **Why not a plain `pip install pyiqa`:** pyiqa requires `opencv-python-headless`, and pip would install 5.0.0.93 of it. That package writes the same `cv2` module as the `opencv-python` 4.13.0.92 already in `UE`. `imagecorruptions` calls `cv2` for defocus blur, motion blur and frost, so the corrupted images, and with them every digest, could change.
  - **What the `--no-deps` install adds:** these three packages and nothing of any other package, checked with a dry run on 5 October. With them, pyiqa's NIQE, ARNIQA and CLIP-IQA all built and scored on the CPU, while `cv2` stayed 4.13.0 and the matmul precision stayed "highest".
  - **pyiqa's wheel is untidy:** besides `pyiqa/`, it puts four top-level folders into `site-packages`: `build/lib/pyiqa` (a second copy of its code, which nothing imports), `docs/`, `options/` and `ResultsCalibra/`. It also adds four test images to `site-packages/tests/`, a folder that ultralytics' wheel already put there. Nothing existing is overwritten, nothing imports these files, and `pip uninstall pyiqa` removes them.
  - **The cost:** `pip check` will then list pyiqa's unused optional dependencies (accelerate, datasets, facexlib, opencv-python-headless and others).
  - **Pins and docs:** `requirements.txt` pins `pyiqa==0.1.16`; `tests/test_package.py` matches it against the package's imports. The README explains the `--no-deps` install.
- **Weights:** pyiqa caches them under torch's hub folder.

  | Model | File | sha256 starts |
  |---|---|---|
  | ARNIQA encoder | `checkpoints/ARNIQA.pth` | `ad2022e59b1040d5` |
  | KADID-10k regressor | `pyiqa/regressor_kadid10k.pth` | `4315bf471d52eb7d` |
  | Published NIQE model | `pyiqa/niqe_modelparameters.mat` | `fa2aa4d12dd9595c` |
  | CLIP RN50 | `clip/RN50.pt` | `afeb0e10f9e5a86d` |

  The run's manifest records each file's full sha256. `ARNIQA()` also loads torchvision's ImageNet ResNet-50 first (`checkpoints/resnet50-0676ba61.pth`, cached), then overwrites every encoder weight with `ARNIQA.pth`. The scores do not depend on that file, so the manifest leaves it out, but the tests that load ARNIQA skip without it.
- **The GPU is shared** with the session "explore". Before each GPU task (5–7):
  - ask explore via SendMessage, and wait for its answer;
  - the config caps the process at `gpu_memory_gib = 8.0`. On 5 October the three models peaked at 1.4 GiB, at batch 16, with whole-image ARNIQA; Task 6 measures the peak again;
  - send "done" afterwards.

  The timing step also needs the card otherwise idle.
- **Tests and the existing pipeline:**
  - Tests run on the CPU only: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`. The suite at the start gives 193 passed, 1 skipped, 16 warnings. From Task 2 on it gives 19 warnings: loading pyiqa's ARNIQA regressor and CLIP archive adds four `torch.jit.load` deprecation warnings, and imagecorruptions' `pkg_resources` warning disappears, because pyiqa imports `pkg_resources` first with that warning filtered.
  - Tests that load real weights skip when the weights are absent.
  - `runs/coco/` is only read, never written.
  - `tests/evaluation/test_report_golden.py` and `tests/test_equivalence.py` stay green.
- **Git and paths:**
  - All work happens in the worktree `/home/yuchen/YuchenZ/UE/philip_sa/.worktrees/iqa-baselines`, on branch `iqa-baselines`. That branch already exists, cut from `fingerprint_bank` at `b54a196`. Never `cd` to the main checkout.
  - `runs/` exists only in the main checkout, so every path into it is absolute.
  - Commit at the end of each task, with both trailers in one `-m`: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU`.
  - `git add` only the files a task names. Never push.
  - The plan and its spec are committed before Task 1.
  - Never touch `IV_2027_Yuchen/` (the paper session writes there) or `docs/superpowers/plans/2026-10-04-cityscapes-c-evaluation.md` (it stays uncommitted).
- **Subagents:** every subagent, implementer and reviewer alike, runs on Opus.
- **Messages to the paper session:** send the plan's path once the plan is reviewed, and the results doc's path with the headline numbers after Task 8.

## Review Focus

1. **Installing pyiqa must not change a single corrupted image.** If `cv2` or anything `imagecorruptions` uses changed, the digests break. Tests: Task 1, Step 1, which checks one val image's 96 digests against `runs/coco/` after the install; and the pass's own digest check (Task 4, `test_the_pass_refuses_corruptions_that_differ_from_the_reference_run`).
2. **Every row must be oriented so that higher means more degraded.** NIQE's distance rises with degradation; ARNIQA's and CLIP-IQA's quality fall, so they are negated; the prototype distance rises. Test: Task 2, `test_the_scores_rise_with_strong_noise`, over all five rows.
3. **Each model must run as its paper's code runs it, not as pyiqa's port does.** CLIP-IQA takes the paper's single prompt pair, not pyiqa's five pairs that name blur and noise, which would carry distortion knowledge the spec rules out. ARNIQA takes the paper's five crops of the image and of a bicubic half-size version, not pyiqa's whole image with an unantialiased one. Tests: Task 2, `test_clipiqa_uses_the_papers_prompt_pair` and `test_arniqa_follows_the_papers_five_crops_and_regresses_as_pyiqa`.
4. **The detectors' run folders, `runs/coco/` above all, and their existing reports must stay untouched.** The run root must keep clear of all of them, and every report goes under the IQA run. Tests: Task 3, `test_the_report_adds_extra_rows_and_writes_where_it_is_told`; Task 4, `test_the_run_root_keeps_clear_of_every_detectors_run` and `test_the_report_writes_every_detectors_report_under_the_iqa_run_only`; and the listings compared in Tasks 6 and 8.
5. **NIQE must match pyiqa exactly for the published model, and must never stop the pass.** The refit must keep only sharp blocks, and an image too small for a block must not crash the fit. A corrupted version with fewer than two blocks NIQE can score (snow and frost can blank blocks) gets the pre-registered score instead of a NaN. Tests: Task 1, `test_niqe_matches_pyiqa_with_the_published_model`, `test_niqe_fit_keeps_the_sharp_blocks_and_drops_nan_rows` and `test_niqe_needs_one_whole_block`; Task 2, `test_niqe_scores_a_version_with_fewer_than_two_blocks_as_most_degraded`; Task 4, `test_the_fit_writes_the_refit_and_the_prototype_once`.

---

### Task 1: pyiqa, and NIQE with a refitted pristine model

**Files:**
- Modify: `requirements.txt` (add `pyiqa==0.1.16` under a comment), `README.md` (the `--no-deps` install, under "Setup")
- Create: `degradation_monitor/baselines/iqa.py`
- Test: `tests/baselines/test_iqa.py`

**Interfaces:**
- Produces, in `degradation_monitor.baselines.iqa`:
  - the constants `NIQE_BLOCK = 96`, `NIQE_SHARPNESS = 0.75` and `NIQE_FEATURES = 36`;
  - `to_unit_tensor(arrays, device) -> Tensor (N, 3, H, W)` in [0, 1];
  - `niqe_luma(images) -> Tensor (N, 1, H, W)` float64;
  - `niqe_block_features(luma) -> (features (N, B, 36), sharpness (N, B))`;
  - `niqe_test_model(features) -> (mu (N, 36), cov (N, 36, 36))`;
  - `niqe_distance(mu_pristine, cov_pristine, mu, cov) -> Tensor (N,)`;
  - `published_niqe() -> (mu (36,), cov (36, 36))`;
  - `NiqeFit`, with `.update(features, sharpness)`, `.images`, `.result() -> (mu, cov, blocks)` and `.dropped` (the blocks with a NaN that `result` left out).

- [ ] **Step 1: Install pyiqa without its dependencies, and check that the corrupted images did not change**

```bash
cd /home/yuchen/YuchenZ/UE/philip_sa/.worktrees/iqa-baselines
/home/yuchen/miniconda3/envs/UE/bin/pip install --no-deps pyiqa==0.1.16 openai-clip==1.0.1 ftfy==6.3.1
CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python - <<'EOF'
import cv2
from degradation_monitor import corruptions
from degradation_monitor.runs import RunLayout, load_npz
from degradation_monitor.settings import load_settings

settings = load_settings("configs/coco.toml")
path = settings.dataset.evaluation_images()[0]
name, arrays = corruptions.load_variants(path)
stored = load_npz(RunLayout(settings.run).score_file("detector", name), ("digests",))["digests"]
assert cv2.__version__ == "4.13.0", cv2.__version__
assert list(stored) == [corruptions.digest(a) for a in arrays], "the corruptions changed"
print("cv2", cv2.__version__, "- all 96 versions of", name, "match runs/coco")
EOF
```

Expected: `cv2 4.13.0 - all 96 versions of <name> match runs/coco`. If anything fails, stop and report: the install changed the environment.

- [ ] **Step 2: Write the failing tests**

Create `tests/baselines/test_iqa.py`:

```python
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from degradation_monitor.baselines import iqa

IMAGE = Path("/home/yuchen/YuchenZ/Datasets/coco/val2017/000000000139.jpg")
NIQE_MODEL = Path(torch.hub.get_dir()) / "pyiqa" / "niqe_modelparameters.mat"


def _unit(array):
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)[None].float() / 255.0


@pytest.mark.skipif(not (IMAGE.exists() and NIQE_MODEL.exists()), reason="the COCO image or NIQE's model is absent")
def test_niqe_matches_pyiqa_with_the_published_model():
    import pyiqa

    x = _unit(np.array(Image.open(IMAGE).convert("RGB")))
    mu, cov = iqa.niqe_test_model(iqa.niqe_block_features(iqa.niqe_luma(x))[0])
    ours = iqa.niqe_distance(*iqa.published_niqe(), mu, cov)
    with torch.inference_mode():
        theirs = pyiqa.create_metric("niqe", device="cpu")(x)
    assert ours.item() == pytest.approx(float(theirs), rel=1e-6)


def test_niqe_fit_keeps_the_sharp_blocks_and_drops_nan_rows():
    rng = np.random.default_rng(0)
    fit = iqa.NiqeFit()
    for _ in range(12):
        luma = rng.uniform(0, 255, (1, 1, 192, 288)).round()  # six blocks of noise ...
        luma[..., :96, :96] = 128.0  # ... one of them flat, so never among the sharp blocks
        features, sharpness = iqa.niqe_block_features(torch.from_numpy(luma))
        assert features.shape == (1, 6, 36) and sharpness.shape == (1, 6)
        fit.update(features, sharpness)
    mu, cov, blocks = fit.result()
    assert mu.shape == (36,) and cov.shape == (36, 36) and np.allclose(cov, cov.T)
    assert fit.images == 12 and 36 < blocks <= 12 * 5 and fit.dropped == 0
    fit.rows.append(np.full((2, 36), np.nan))
    assert fit.result()[2] == blocks and fit.dropped == 2


def test_niqe_needs_one_whole_block():
    with pytest.raises(ValueError, match="at least one whole 96 x 96 block"):
        iqa.niqe_block_features(torch.zeros(1, 1, 64, 300, dtype=torch.float64))
```

- [ ] **Step 3: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/baselines/test_iqa.py`
Expected: FAIL with `ImportError: cannot import name 'iqa' from 'degradation_monitor.baselines'`.

- [ ] **Step 4: Implement**

Add these two lines to `requirements.txt`; `tests/test_package.py` ignores the comment line:

```text
# pyiqa: install with --no-deps (README, Setup): its opencv-python-headless would replace the cv2 imagecorruptions uses
pyiqa==0.1.16
```

In `README.md`, under "Setup":
- change the first bullet to:

```markdown
- **Python packages:** Python 3.11 and the packages in `requirements.txt`. Install a PyTorch build for your machine
  first, then every package but pyiqa, `grep -v '^pyiqa' requirements.txt | pip install -r /dev/stdin`, then pyiqa
  as below.
```

- add after the DisCoPatch bullet:

```markdown
- **pyiqa** (the image-quality baselines): `pip install --no-deps pyiqa==0.1.16 openai-clip==1.0.1 ftfy==6.3.1`.
  A plain install would add `opencv-python-headless`, whose `cv2` replaces the one of `opencv-python` that
  imagecorruptions uses, and the corrupted images could change. `pip check` then lists pyiqa's unused optional
  dependencies; that is expected.
```

Create `degradation_monitor/baselines/iqa.py`:

```python
"""Four detector-free image-quality baselines, each scored so that higher means more likely degraded.

- NIQE (Mittal, Soundararajan & Bovik, IEEE SPL 2013): the distance between a multivariate Gaussian fitted to an
  image's block features and one fitted to pristine images. The main row's pristine model is refitted on clean train
  images with the authors' recipe (their estimatemodelparam.m: 96 x 96 blocks at two scales, keeping the blocks
  sharper than 0.75 times the image's sharpest); the sensitivity row uses the published model. Both rows share the
  test image's features, computed as pyiqa 0.1.16's NIQE does: the rounded luma, in float64.

Every model reads the uint8 RGB image at its own size.
"""
from __future__ import annotations

import numpy as np
import torch

NIQE_BLOCK = 96
NIQE_SHARPNESS = 0.75  # estimatemodelparam.m's sh_th: keep the blocks sharper than 0.75 times the sharpest
NIQE_FEATURES = 36  # 18 at full size, 18 at half size


def to_unit_tensor(arrays, device) -> torch.Tensor:
    """Same-size uint8 RGB arrays -> (N, 3, H, W) float32 in [0, 1] on the device."""
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(device).float().div_(255.0)


def niqe_luma(images: torch.Tensor) -> torch.Tensor:
    """(N, 3, H, W) RGB in [0, 1] -> (N, 1, H, W): the rounded luma in [0, 255], float64, as pyiqa's NIQE."""
    from pyiqa.archs.func_util import diff_round
    from pyiqa.utils.color_util import to_y_channel

    return diff_round(to_y_channel(images, 255, "yiq")).to(torch.float64)


def niqe_block_features(luma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Each whole 96 x 96 block's 36 features (18 at full size, 18 at half size) and its sharpness, the mean local
    standard deviation at full size: (N, B, 36) and (N, B). Partial blocks at the right and bottom are dropped."""
    from pyiqa.archs.func_util import normalize_img_with_gauss, safe_sqrt
    from pyiqa.archs.niqe_arch import compute_feature
    from pyiqa.matlab_utils import blockproc, fspecial, imfilter, imresize

    height, width = luma.shape[-2:]
    rows, cols = height // NIQE_BLOCK, width // NIQE_BLOCK
    if rows == 0 or cols == 0:
        raise ValueError(f"NIQE needs at least one whole {NIQE_BLOCK} x {NIQE_BLOCK} block, got {height} x {width}")
    image = luma[..., :rows * NIQE_BLOCK, :cols * NIQE_BLOCK]
    kernel = fspecial(7, 7.0 / 6, 1).to(image)  # the window normalize_img_with_gauss uses
    mu = imfilter(image, kernel, padding="replicate")
    sigma = safe_sqrt((imfilter(image ** 2, kernel, padding="replicate") - mu ** 2).abs())
    sharpness = blockproc(sigma, [NIQE_BLOCK, NIQE_BLOCK], fun=lambda blocks, _: blocks.mean(dim=(2, 3)))[..., 0]
    features = []
    for scale in (1, 2):
        normalized = normalize_img_with_gauss(image, padding="replicate")
        features.append(blockproc(normalized, [NIQE_BLOCK // scale, NIQE_BLOCK // scale], fun=compute_feature))
        if scale == 1:
            image = imresize(image / 255.0, scale=0.5, antialiasing=True) * 255.0
    return torch.cat(features, dim=-1), sharpness


def niqe_test_model(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Each image's Gaussian over its blocks: mean (N, 36) and covariance (N, 36, 36), blocks with a NaN left out."""
    from pyiqa.matlab_utils import nancov, nanmean

    return nanmean(features, dim=1), nancov(features)


def niqe_distance(mu_pristine, cov_pristine, mu: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    """NIQE's Eq. 10: the distance between the pristine Gaussian and each image's, (N,)."""
    mu_pristine = torch.as_tensor(mu_pristine).to(mu)
    cov_pristine = torch.as_tensor(cov_pristine).to(cov)
    diff = (mu_pristine - mu).unsqueeze(1)
    inverse = torch.linalg.pinv((cov_pristine + cov) / 2)
    return torch.bmm(torch.bmm(diff, inverse), diff.transpose(1, 2)).reshape(-1).sqrt()


def published_niqe() -> tuple[np.ndarray, np.ndarray]:
    """The authors' published pristine model (their modelparameters.mat), as pyiqa's 'niqe' metric loads it."""
    from pyiqa.archs.niqe_arch import NIQE

    model = NIQE(version="original")
    return model.mu_pris_param.numpy(), model.cov_pris_param.numpy()


class NiqeFit:
    """NIQE's pristine Gaussian from clean images' sharp blocks (estimatemodelparam.m)."""

    def __init__(self):
        self.rows, self.images, self.dropped = [], 0, 0

    def update(self, features: torch.Tensor, sharpness: torch.Tensor) -> None:
        for image_features, image_sharpness in zip(features, sharpness):
            keep = image_sharpness > NIQE_SHARPNESS * image_sharpness.max()
            self.rows.append(image_features[keep].double().cpu().numpy())
            self.images += 1

    def result(self) -> tuple[np.ndarray, np.ndarray, int]:
        """Mean (36,), covariance (36, 36) with N - 1, and the number of blocks. Blocks with a NaN are left out of
        both and counted in .dropped (MATLAB's nanmean would keep their other features in the mean; the sharp blocks
        kept hold no NaN in practice)."""
        rows = np.concatenate(self.rows) if self.rows else np.zeros((0, NIQE_FEATURES))
        nan = np.isnan(rows).any(axis=1)
        self.dropped, rows = int(nan.sum()), rows[~nan]
        if len(rows) <= NIQE_FEATURES:
            raise ValueError(f"NIQE's pristine model needs more than {NIQE_FEATURES} blocks, got {len(rows)}")
        return rows.mean(axis=0), np.cov(rows, rowvar=False), len(rows)
```

- [ ] **Step 5: Run the tests to see them pass, then the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/baselines/test_iqa.py tests/test_package.py`
Expected: PASS.

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: `196 passed, 1 skipped`, with the same 16 warnings as before. Report any new warning, as a finding.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt README.md degradation_monitor/baselines/iqa.py tests/baselines/test_iqa.py
git commit -m "feat: NIQE with a pristine model refitted on clean images, through pyiqa

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

---

### Task 2: ARNIQA, CLIP-IQA and the four models together

**Files:**
- Modify: `degradation_monitor/baselines/iqa.py` (append, and extend the module docstring)
- Test: `tests/baselines/test_iqa.py` (append)

**Interfaces:**
- Consumes: Task 1's functions.
- Produces, in `degradation_monitor.baselines.iqa`:
  - the constants `CLIP_PROMPTS`, `ARNIQA_REGRESSOR = "kadid"`, `ARNIQA_CROP = 224`, `NIQE_MIN_BLOCKS = 2`, `NIQE_UNSCORABLE = 1e6` and `ROWS = ("niqe", "niqe_default", "arniqa", "arniqa_proto", "clipiqa")`;
  - `arniqa_crops(image) -> list` of five PIL images, the official `center_corners_crop`;
  - `Arniqa(device)`, with `.embed(arrays) -> (N, 4096)` (uint8 RGB arrays of any sizes), `.quality(embedding) -> (N,)` and `.model`;
  - `ClipIqa(device)`, with `.quality(images) -> (N,)` (a tensor in [0, 1]), `.clip` (pyiqa's CLIP RN50) and `.tokens`;
  - `IqaModels(device, niqe_refit=None, prototype=None)`, with:
    - class attributes `batch_size = 16`, `fit_batch_size = 1` and `protocol`;
    - `.fit_features(arrays) -> (features or None, sharpness or None, embedding)`;
    - `.niqe_rows(arrays)`, `.arniqa_rows(arrays)` and `.clipiqa_rows(arrays)`, each taking uint8 RGB arrays of one size and returning a dict of (N,) tensors; `niqe_rows` also gives `niqe_blocks`, each version's blocks without a NaN;
    - `.scores(arrays) -> dict` of the five rows, as float64 numpy (N,), and `niqe_blocks`;
    - attributes `.niqe_refit`, `.prototype` and `.niqe_default`;
  - `weight_files() -> dict[str, Path]`;
  - `load_iqa_models(device, niqe_refit=None, prototype=None) -> IqaModels`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/baselines/test_iqa.py`:

```python
WEIGHTS = [Path(torch.hub.get_dir()) / p for p in ("checkpoints/ARNIQA.pth", "checkpoints/resnet50-0676ba61.pth",
                                                   "pyiqa/regressor_kadid10k.pth", "pyiqa/niqe_modelparameters.mat",
                                                   "clip/RN50.pt")]
needs_weights = pytest.mark.skipif(not (IMAGE.exists() and all(p.exists() for p in WEIGHTS)),
                                   reason="the COCO image or the IQA weights are absent")


@pytest.fixture(scope="module")
def models():
    if not (IMAGE.exists() and all(p.exists() for p in WEIGHTS)):
        pytest.skip("the COCO image or the IQA weights are absent")
    return iqa.load_iqa_models("cpu")


def _image():
    return np.array(Image.open(IMAGE).convert("RGB"))  # writable, as the stages' images are


@needs_weights
def test_arniqa_follows_the_papers_five_crops_and_regresses_as_pyiqa(models):
    import pyiqa
    import torch.nn.functional as F
    from torchvision import transforms
    from torchvision.transforms import functional as TVF

    image = Image.open(IMAGE).convert("RGB")
    embedding = models.arniqa.embed([np.array(image)])
    assert embedding.shape == (1, 4096)
    # ARNIQA's test pipeline as test.py runs it: the half-size image from PIL's default resize (bicubic), the centre
    # and the four corner crops of both, ToTensor, ImageNet normalisation; each crop's features L2-normalised
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def crops(img):
        w, h = img.size
        tops_lefts = [(h // 2 - 112, w // 2 - 112), (0, 0), (h - 224, 0), (0, w - 224), (h - 224, w - 224)]
        return torch.stack([normalize(transforms.ToTensor()(TVF.crop(img, t, l, 224, 224))) for t, l in tops_lefts])

    encoder = models.arniqa.model.encoder
    with torch.inference_mode():
        full = F.normalize(encoder(crops(image)).flatten(1), dim=1)
        half = F.normalize(encoder(crops(image.resize((image.size[0] // 2, image.size[1] // 2)))).flatten(1), dim=1)
    per_crop = torch.hstack((full, half))  # test.py's features: one row per crop
    assert torch.allclose(embedding[0], per_crop.mean(dim=0), atol=1e-5)
    mean_of_crops = models.arniqa.quality(per_crop).mean().item()  # test.py averages the five crops' scores
    assert models.arniqa.quality(embedding).item() == pytest.approx(mean_of_crops, abs=1e-5)
    x = _unit(np.array(image))  # the regressor and its scaling are pyiqa's own: the same score on pyiqa's features
    with torch.inference_mode():
        whole, small = models.arniqa.model._preprocess(x)
        theirs = torch.hstack((F.normalize(encoder(whole).flatten(1), dim=1),
                               F.normalize(encoder(small).flatten(1), dim=1)))
        metric = pyiqa.create_metric("arniqa-kadid", device="cpu")(x)
    assert models.arniqa.quality(theirs).item() == pytest.approx(float(metric), rel=1e-5)


@needs_weights
def test_clipiqa_uses_the_papers_prompt_pair(models):
    from pyiqa.archs.clip_imports import clip
    from pyiqa.archs.clipiqa_arch import CLIPIQA

    assert torch.equal(models.clipiqa.tokens, clip.tokenize(["Good photo.", "Bad photo."]))
    x = _unit(_image())
    one = models.clipiqa.quality(x)
    two = models.clipiqa.quality(torch.cat([x, x]))
    assert one.shape == (1,) and 0.0 < one.item() < 1.0 and torch.allclose(two, one.repeat(2), atol=1e-5)
    port = CLIPIQA(model_type="clipiqa").eval()  # pyiqa's port, fp32 on the CPU, given the paper's single pair
    port.prompt_pairs = clip.tokenize(["Good photo.", "Bad photo."])
    with torch.inference_mode():
        assert one.item() == pytest.approx(port(x).item(), abs=1e-5)


@needs_weights
def test_the_scores_rise_with_strong_noise(models):
    from degradation_monitor import corruptions

    clean = _image()
    noisy = np.array(corruptions.corrupt(Image.fromarray(clean), IMAGE.name, "gaussian_noise", 5))
    models.niqe_refit = iqa.published_niqe()  # stand-ins for the clean references the fit stage builds
    models.prototype = models.arniqa.embed([clean])[0].numpy()
    scores = models.scores([clean, noisy])
    assert set(scores) == {*iqa.ROWS, "niqe_blocks"} and (scores["niqe_blocks"] >= iqa.NIQE_MIN_BLOCKS).all()
    assert all(scores[row].shape == (2,) and np.isfinite(scores[row]).all() for row in iqa.ROWS)
    for row in iqa.ROWS:
        assert scores[row][1] > scores[row][0], row
    assert scores["arniqa_proto"][0] == pytest.approx(0.0, abs=1e-5)


@needs_weights
@pytest.mark.filterwarnings(r"ignore:cov\(\)")  # torch.cov warns on the one-block version, as it should
def test_niqe_scores_a_version_with_fewer_than_two_blocks_as_most_degraded(models, monkeypatch):
    models.niqe_refit, models.prototype = iqa.published_niqe(), np.full(4096, 1 / 64.0)
    features = torch.from_numpy(np.random.default_rng(1).uniform(0.1, 1.0, (3, 4, 36)))
    features[1, 1:, 0] = float("nan")  # one block left without a NaN
    features[2, :, 5] = float("nan")  # none left
    monkeypatch.setattr(iqa, "niqe_block_features", lambda luma: (features, None))
    rows = models.niqe_rows([_image()] * 3)
    assert rows["niqe_blocks"].tolist() == [4, 1, 0]
    assert 0 < rows["niqe"][0] < iqa.NIQE_UNSCORABLE and 0 < rows["niqe_default"][0] < iqa.NIQE_UNSCORABLE
    assert rows["niqe"][1:].tolist() == rows["niqe_default"][1:].tolist() == [iqa.NIQE_UNSCORABLE] * 2


@needs_weights
def test_the_scores_need_the_clean_references(models):
    models.niqe_refit = None
    with pytest.raises(ValueError, match="run the fit stage first"):
        models.scores([_image()])
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/baselines/test_iqa.py`
Expected: FAIL with `AttributeError: module 'degradation_monitor.baselines.iqa' has no attribute 'load_iqa_models'`, raised in the fixture.

- [ ] **Step 3: Implement**

In the module docstring of `degradation_monitor/baselines/iqa.py`, add after the NIQE paragraph:

```text
- ARNIQA (Agnolucci, Galteri, Bertini & Del Bimbo, WACV 2024), as its paper evaluates it (the official test.py): the
  centre and the four corner crops, 224 x 224, of the image and of its half-size version (PIL's bicubic resize); per
  crop, the ResNet-50 features at both sizes, each L2-normalised and concatenated (return_embedding, 4096 values);
  averaged over the five crops. On the GPU the encoder runs under autocast, as the official code runs it. Quality:
  the KADID-10k regressor on that embedding; the regressor is linear, so this is the mean of the crops' qualities, as
  test.py averages them. The prototype row (Becker, Weiss, Hübner & Arens, arXiv 2602.18394): 1 - cosine similarity
  to the mean embedding of clean train images.
- CLIP-IQA (Wang, Chan & Loy, AAAI 2023), zero-shot, as its official code computes it (IceClear/CLIP-IQA, the
  CLIPIQAFixed model of configs/clipiqa/clipiqa_attribute_test.py): CLIP RN50 on the image at its own size with the
  positional embedding removed, the softmax of its logits for "Good photo." against "Bad photo.", the first
  probability. On the GPU the weights are fp16, as the official build_model makes them; on the CPU, fp32.
- NIQE in the scores: a version with fewer than two blocks without a NaN has no covariance (snow and frost can
  blank blocks). Both NIQE rows give it NIQE_UNSCORABLE, above every real distance, so it counts as degraded;
  niqe_blocks records each version's count.
```

Add to the imports:

```python
import contextlib
from importlib import metadata
from pathlib import Path

import torch.nn.functional as F
from PIL import Image
```

Append:

```python
CLIP_PROMPTS = ("Good photo.", "Bad photo.")
ARNIQA_REGRESSOR = "kadid"
ARNIQA_CROP = 224  # the paper's crops: the centre and the four corners, of the image and of its half-size version
NIQE_MIN_BLOCKS = 2  # with fewer blocks without a NaN, a version's covariance is undefined
NIQE_UNSCORABLE = 1e6  # such a version's NIQE score: above every real distance, so it counts as degraded
ROWS = ("niqe", "niqe_default", "arniqa", "arniqa_proto", "clipiqa")


def arniqa_crops(image: Image.Image) -> list:
    """ARNIQA's center_corners_crop (utils/utils_data.py): the centre and the four corners of a PIL image, 224 x 224,
    in that order, padded with zeros where the image is smaller, as torchvision's crop of a PIL image pads."""
    width, height, size = image.width, image.height, ARNIQA_CROP
    corners = [(width // 2 - size // 2, height // 2 - size // 2), (0, 0), (0, height - size), (width - size, 0),
               (width - size, height - size)]
    return [image.crop((left, top, left + size, top + size)) for left, top in corners]


class Arniqa:
    """ARNIQA's encoder and KADID-10k regressor, on the paper's five crops of the image and of its half-size version."""

    def __init__(self, device):
        from pyiqa.archs.arniqa_arch import ARNIQA

        self.device = torch.device(device)
        self.model = ARNIQA(regressor_dataset=ARNIQA_REGRESSOR).to(self.device).eval().requires_grad_(False)
        self.mean = self.model.default_mean.to(self.device)
        self.std = self.model.default_std.to(self.device)

    def _features(self, crops) -> torch.Tensor:
        """(C, 2048): the encoder's features of PIL crops, each L2-normalised. On the GPU the encoder runs under
        autocast, as the official code runs it."""
        x = torch.from_numpy(np.stack([np.asarray(crop) for crop in crops])).permute(0, 3, 1, 2).to(self.device)
        x = (x.float().div_(255.0) - self.mean) / self.std
        precision = torch.autocast("cuda", dtype=torch.float16) if self.device.type == "cuda" \
            else contextlib.nullcontext()
        with precision:
            features = self.model.encoder(x)
        return F.normalize(features.float().flatten(1), dim=1)

    @torch.inference_mode()
    def embed(self, arrays) -> torch.Tensor:
        """uint8 RGB arrays (H, W, 3), of any sizes -> (N, 4096): for each image, test.py's features of its five crops,
        the full- and half-size features each L2-normalised and concatenated, averaged over the crops."""
        full, half = [], []
        for array in arrays:
            image = Image.fromarray(array)
            full += arniqa_crops(image)
            # the official resize_crop: PIL's resize, whose default filter for RGB is bicubic
            half += arniqa_crops(image.resize((image.width // 2, image.height // 2), Image.Resampling.BICUBIC))
        crops = torch.cat([self._features(full), self._features(half)], dim=1)
        return crops.view(len(arrays), 5, -1).mean(dim=1)  # five crops per image

    @torch.inference_mode()
    def quality(self, embedding: torch.Tensor) -> torch.Tensor:
        """The KADID-10k regressor's quality, scaled from KADID-10k's rating range to about [0, 1] as pyiqa scales it,
        higher is better: (N,). The regressor is linear, so on a mean embedding it gives the mean quality."""
        return self.model._scale_score(self.model.regressor(embedding)).reshape(-1)


class ClipIqa:
    """Zero-shot CLIP-IQA: the official CLIPIQAFixed.forward on pyiqa's copy of the modified CLIP."""

    def __init__(self, device):
        from pyiqa.archs.clip_imports import clip
        from pyiqa.archs.clip_model import load
        from pyiqa.archs.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

        self.device = torch.device(device)
        # fp16 weights on the GPU, as the official build_model's convert_weights; load() keeps fp32 on the CPU only
        self.clip = load("RN50", self.device).eval().requires_grad_(False)
        self.tokens = clip.tokenize(list(CLIP_PROMPTS)).to(self.device)
        self.mean = torch.tensor(OPENAI_CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(OPENAI_CLIP_STD, device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def quality(self, images: torch.Tensor) -> torch.Tensor:
        """The probability of "Good photo." against "Bad photo.", (N,): the official test pipeline's normalisation,
        then CLIPIQAFixed.forward; the model casts the image to its own dtype."""
        logits_per_image, _ = self.clip((images - self.mean) / self.std, self.tokens, pos_embedding=False)
        return logits_per_image.softmax(dim=-1)[:, 0].float()


def weight_files() -> dict:
    """The model files the scores depend on; pyiqa caches them under torch's hub folder."""
    hub = Path(torch.hub.get_dir())
    return {"arniqa_encoder": hub / "checkpoints" / "ARNIQA.pth",
            "arniqa_regressor": hub / "pyiqa" / "regressor_kadid10k.pth",
            "niqe_published": hub / "pyiqa" / "niqe_modelparameters.mat",
            "clip_rn50": hub / "clip" / "RN50.pt"}


class IqaModels:
    """The four baselines on one batch of same-size images. The scores need the clean references, the refitted
    NIQE model and ARNIQA's prototype, from the fit stage."""
    batch_size = 16  # one image's 96 versions, 16 at a time
    fit_batch_size = 1  # clean train images differ in size
    protocol = {
        "niqe": {"block": NIQE_BLOCK, "sharpness": NIQE_SHARPNESS, "luma": "rounded Y in [0, 255], float64",
                 "refit": "estimatemodelparam.m", "default": "modelparameters.mat",
                 "unscorable": f"fewer than {NIQE_MIN_BLOCKS} blocks without a NaN: {NIQE_UNSCORABLE:g}"},
        "arniqa": {"regressor": "kadid10k", "source": "miccunifi/ARNIQA test.py",
                   "crops": f"centre and four corners, {ARNIQA_CROP} x {ARNIQA_CROP}, zero-padded, at both sizes",
                   "half_size": "PIL bicubic to (W // 2, H // 2)",
                   "embedding": "per crop, both sizes each L2-normalised, 4096; the mean over the crops",
                   "precision": "encoder under autocast (fp16) on the GPU, fp32 on the CPU",
                   "prototype": "mean clean embedding; 1 - cosine"},
        "clipiqa": {"backbone": "RN50", "prompts": list(CLIP_PROMPTS), "positional_embedding": False,
                    "score": "softmax of logit_scale x cosine; probability of the first prompt",
                    "weights": "fp16 on the GPU (official build_model), fp32 on the CPU",
                    "source": "IceClear/CLIP-IQA v2-3.8, CLIPIQAFixed"},
        "packages": {name: metadata.version(name) for name in ("pyiqa", "openai-clip", "ftfy")},
    }

    def __init__(self, device, niqe_refit=None, prototype=None):
        self.device = torch.device(device)
        self.arniqa, self.clipiqa = Arniqa(device), ClipIqa(device)
        self.niqe_default = published_niqe()
        self.niqe_refit, self.prototype = niqe_refit, prototype

    @torch.inference_mode()
    def fit_features(self, arrays):
        """For the clean references: NIQE's block features and sharpness (None for an image without a whole block)
        and ARNIQA's embeddings."""
        features = sharpness = None
        if min(arrays[0].shape[:2]) >= NIQE_BLOCK:
            features, sharpness = niqe_block_features(niqe_luma(to_unit_tensor(arrays, self.device)))
        return features, sharpness, self.arniqa.embed(arrays)

    def _references(self):
        if self.niqe_refit is None or self.prototype is None:
            raise ValueError("the scores need the clean references: run the fit stage first")

    @torch.inference_mode()
    def niqe_rows(self, arrays) -> dict:
        """Both NIQE rows, and niqe_blocks: each version's blocks without a NaN. A version with fewer than
        NIQE_MIN_BLOCKS of them has no covariance; both rows give it NIQE_UNSCORABLE."""
        self._references()
        features = niqe_block_features(niqe_luma(to_unit_tensor(arrays, self.device)))[0]
        blocks = (~features.isnan().any(dim=2)).sum(dim=1)
        mu, cov = niqe_test_model(features)
        unscorable = blocks < NIQE_MIN_BLOCKS
        mu = torch.where(unscorable[:, None], 0.0, mu)  # finite stand-ins, so that pinv never sees a NaN
        cov = torch.where(unscorable[:, None, None], 0.0, cov)
        rows = {name: torch.where(unscorable, NIQE_UNSCORABLE, niqe_distance(*model, mu, cov))
                for name, model in (("niqe", self.niqe_refit), ("niqe_default", self.niqe_default))}
        return {**rows, "niqe_blocks": blocks}

    @torch.inference_mode()
    def arniqa_rows(self, arrays) -> dict:
        self._references()
        embedding = self.arniqa.embed(arrays)
        prototype = torch.as_tensor(self.prototype, dtype=embedding.dtype, device=self.device)
        return {"arniqa": -self.arniqa.quality(embedding),
                "arniqa_proto": 1.0 - F.cosine_similarity(embedding, prototype[None], dim=1)}

    @torch.inference_mode()
    def clipiqa_rows(self, arrays) -> dict:
        return {"clipiqa": -self.clipiqa.quality(to_unit_tensor(arrays, self.device))}

    def scores(self, arrays) -> dict:
        """The five rows of one batch of same-size images, float64 (N,) each, higher meaning more likely degraded;
        and niqe_blocks."""
        self._references()
        rows = {**self.niqe_rows(arrays), **self.arniqa_rows(arrays), **self.clipiqa_rows(arrays)}
        return {**{key: rows[key].double().cpu().numpy() for key in ROWS},
                "niqe_blocks": rows["niqe_blocks"].cpu().numpy()}


def load_iqa_models(device, niqe_refit=None, prototype=None) -> IqaModels:
    return IqaModels(device, niqe_refit=niqe_refit, prototype=prototype)
```

- [ ] **Step 4: Run the tests to see them pass, then the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/baselines/test_iqa.py`
Expected: PASS.
- If the noise test fails on a row, report which row and its two values. Never flip a sign to make it pass.
- If the ARNIQA test fails, report the largest difference: the crops or the half-size image then differ from the official pipeline.

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: `201 passed, 1 skipped, 19 warnings`, the warnings as the Global Constraints explain. Report any other new warning, as a finding.

- [ ] **Step 5: Commit**

```bash
git add degradation_monitor/baselines/iqa.py tests/baselines/test_iqa.py
git commit -m "feat: ARNIQA's quality and prototype, zero-shot CLIP-IQA, and the four models together

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

---

### Task 3: Reports with extra rows, written where they are told

**Files:**
- Modify: `degradation_monitor/runs.py`: add the `iqa` score folder.
- Modify: `degradation_monitor/evaluation/report.py`: the five rows, their labels, their families, number words up to ten, and `EXTRA_ROWS`.
- Modify: `degradation_monitor/stages/report.py`: `write_report` takes `out`, `extra_rows` and `extra_inputs`.
- Test: `tests/stages/test_report.py`, `tests/evaluation/test_report.py`, and `tests/test_runs.py` (its layout test names every score folder)

**Interfaces:**
- Consumes: `ROWS` from Task 2 (the five row names). Use them as literal strings here, not by importing them from `baselines/iqa.py`, which imports pyiqa.
- Produces:
  - `SCORE_KEYS["iqa"]`;
  - the report rows `niqe`, `niqe_default`, `arniqa`, `arniqa_proto` and `clipiqa`, which are also `tables.EXTRA_ROWS`, the only rows a report takes from outside its run folder;
  - `write_report(settings, manifest, out=None, extra_rows=None, extra_inputs=None)`. It adds `extra_rows` ((images, 96) arrays in evaluation order, named from `EXTRA_ROWS`) to the report's rows, writes to `out` (default `settings.layout.report()`), and merges `extra_inputs` into `summary["inputs"]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/stages/test_report.py`, which already has the `run` fixture, `_statistics`, `atomic_npz` and `method_reference`:

```python
def test_the_report_adds_extra_rows_and_writes_where_it_is_told(run, tmp_path, monkeypatch):
    from degradation_monitor.runs import Manifest
    from degradation_monitor.stages.report import write_report

    monkeypatch.setattr(method_reference, "NEIGHBOURS", 3)
    rng = np.random.default_rng(9)
    for path in sorted(run.layout.scores("detector").glob("*.npz")):
        atomic_npz(run.layout.score_file("method", path.name), **_statistics(rng, 96))
    atomic_npz(run.layout.method_bank, **_statistics(rng, 6))
    atomic_npz(run.layout.method_zstats, **_statistics(rng, 4))
    extra = {"niqe": rng.uniform(0, 1, (IMAGES, 96)), "clipiqa": rng.uniform(0, 1, (IMAGES, 96))}
    out = tmp_path / "iqa-report"
    write_report(run, Manifest(run.layout), out=out, extra_rows=extra, extra_inputs={"iqa": {"scores": "here"}})
    summary = json.loads((out / "summary.json").read_text())
    assert {"two_axis", "niqe", "clipiqa"} <= set(summary["rows"])
    assert "two_axis - niqe:auroc_common" in summary["intervals"]["all"]
    assert summary["inputs"]["iqa"] == {"scores": "here"}
    assert not run.layout.report().exists()  # the run folder itself gets no report


def test_the_report_refuses_extra_rows_it_cannot_take(run, tmp_path):
    from degradation_monitor.runs import Manifest
    from degradation_monitor.stages.report import write_report

    for rows, message in (({"psnr": np.zeros((IMAGES, 96))}, "cannot take from elsewhere: psnr"),
                          ({"knn": np.zeros((IMAGES, 96))}, "cannot take from elsewhere: knn"),
                          ({"niqe": np.zeros((IMAGES - 1, 96))}, "one per image and condition: niqe")):
        with pytest.raises(ValueError, match=message):
            write_report(run, Manifest(run.layout), out=tmp_path, extra_rows=rows)
```

In `tests/test_runs.py`, the layout test names every score folder; change its assertion to include the new one:

```python
    assert set(SCORE_KEYS) == {"detector", "activations", "discopatch", "method", "iqa"}
```

Append to `tests/evaluation/test_report.py`, after `test_the_report_title_counts_the_baseline_families_present`:

```python
def test_the_report_title_counts_the_image_quality_families(small_sets):
    rows = ("two_axis", "level", "global_level", "saod_top3", "saod_min", "knn", "discopatch", "cdf", "cdf_sum",
            "hashemi", "hashemi_enc", "niqe", "niqe_default", "arniqa", "arniqa_proto", "clipiqa")
    tables, summary = _tables(_scores(40, rows), 40, samples=2)  # and ContrastiveConf: ten families
    assert report.markdown(summary, tables).startswith("# Corruption detection on COCO: our method and ten baselines\n")
    assert report.LABELS["arniqa_proto"] in report.markdown(summary, tables)
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/stages/test_report.py tests/evaluation/test_report.py tests/test_runs.py`
Expected: FAIL.
- `write_report` raises `TypeError: write_report() got an unexpected keyword argument 'out'`, in both new stage tests.
- The title test fails its assertion: `build_tables` drops the rows it does not know, so the title still says "six baselines".
- The layout test fails its assertion: `SCORE_KEYS` has no `iqa` yet.

- [ ] **Step 3: Implement**

In `degradation_monitor/runs.py`, add to `SCORE_KEYS`:

```python
    "iqa": ("niqe", "niqe_default", "arniqa", "arniqa_proto", "clipiqa", "niqe_blocks", "digests"),
```

In `degradation_monitor/evaluation/report.py`:
- extend `BASELINES` with the five rows, in this order, after `"cdf_sum"`: `"niqe", "niqe_default", "arniqa", "arniqa_proto", "clipiqa"`;
- after `BASELINES`, add `EXTRA_ROWS = ("niqe", "niqe_default", "arniqa", "arniqa_proto", "clipiqa")`, with the comment `# the rows a report takes from outside its run folder: the image-quality baselines`;
- add to `LABELS`:

```python
    "niqe": "NIQE, pristine model refitted on clean train images",
    "niqe_default": "NIQE, published pristine model (sensitivity)",
    "arniqa": "ARNIQA quality (KADID-10k regressor)",
    "arniqa_proto": "ARNIQA embedding vs the clean prototype (Becker et al.)",
    "clipiqa": "CLIP-IQA, zero-shot (\"Good photo.\" / \"Bad photo.\")",
```

- add to `BASELINE_FAMILIES`: `"NIQE": ("niqe", "niqe_default"), "ARNIQA": ("arniqa",), "ARNIQA prototype": ("arniqa_proto",), "CLIP-IQA": ("clipiqa",)`;
- extend `NUMBER_WORDS` to `("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")`.

In `degradation_monitor/stages/report.py`, change `write_report`:
- Its signature and docstring:

```python
def write_report(settings, manifest, out=None, extra_rows=None, extra_inputs=None) -> None:
    """Every table of the report, from the stored scores of every pass that ran.

    extra_rows adds rows computed elsewhere, the image-quality baselines (EXTRA_ROWS): (images, 96) arrays in
    evaluation order. out writes the report to another folder, so that a read-only run can get one. extra_inputs is merged into
    the summary's inputs.
    """
```

- After `scores = tables.baseline_rows(...)`:

```python
    if extra_rows:
        refused = sorted(set(extra_rows) - set(tables.EXTRA_ROWS))
        if refused:
            raise ValueError(f"rows a report cannot take from elsewhere: {', '.join(refused)}")
        shape = (len(names), len(corruptions.CONDITIONS))
        wrong = sorted(k for k, v in extra_rows.items() if np.shape(v) != shape)
        if wrong:
            raise ValueError(f"extra rows need {shape[0]} x {shape[1]} values, one per image and condition: "
                             f"{', '.join(wrong)}")
        scores.update(extra_rows)
```

- The `summary["inputs"]` line and the write:

```python
    summary["inputs"] = {**manifest.read().get("inputs", {}), **({"method": reference} if reference else {}),
                         **(extra_inputs or {})}
    tables.write_outputs(out or layout.report(), report_tables, summary)
```

- [ ] **Step 4: Run the tests to see them pass, then the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/stages tests/evaluation`
Expected: PASS. `test_report_golden.py` must pass unchanged.

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: `204 passed, 1 skipped, 19 warnings`.

- [ ] **Step 5: Commit**

```bash
git add degradation_monitor/runs.py degradation_monitor/evaluation/report.py degradation_monitor/stages/report.py \
        tests/stages/test_report.py tests/evaluation/test_report.py tests/test_runs.py
git commit -m "feat: reports take extra rows and an output folder; the five image-quality rows

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

---

### Task 4: The image-quality stages, their config and README

**Files:**
- Create: `degradation_monitor/stages/iqa.py`, `configs/coco-iqa.toml`
- Modify:
  - `degradation_monitor/stages/common.py`: gains `rgb_batches` and `report_peak_memory`, moved from `stages/detectors.py`;
  - `degradation_monitor/stages/detectors.py`: uses it;
  - `README.md`.
- Test: `tests/stages/test_iqa.py`

**Interfaces:**
- Consumes:
  - `IqaModels`, `load_iqa_models`, `NiqeFit` (with `.dropped`), `ROWS`, `NIQE_BLOCK`, `NIQE_SHARPNESS` and `weight_files` (Tasks 1–2); the models' `*_rows` and `scores` take uint8 RGB arrays, and `scores` also gives `niqe_blocks`;
  - `write_report(settings, manifest, out, extra_rows, extra_inputs)` and `LABELS` (Task 3);
  - `stages.detectors.load_config` (`.run`, `.detectors`, `.settings(name)`);
  - `load_settings`;
  - from `runs`: `Manifest`, `RunLayout`, `atomic_json`, `atomic_npz`, `load_npz`, `stack`, `progress`, `sha1` and `sha256`;
  - `corruptions.digest`;
  - from `stages.common`: `variant_stream`, `cap_gpu_memory` and `open_rgb`;
  - from `stages.baselines`: `TIMING_IMAGES` and `TIMING_WARMUP`;
  - `cli.positive_int`.
- Produces:
  - `stages.common.rgb_batches(paths, size, workers)` and `stages.common.report_peak_memory(label, device)`;
  - in `stages.iqa`:
    - `load_config(path, run=None) -> IqaConfig`, with `.base`, `.run`, `.layout`, `.reference_run`, `.detectors` (an ordered dict, name → Settings) and `.gpu_memory_gib`. It refuses a run root that is, holds or lies inside the detectors' run root or any detector's run folder;
    - the stages `fit(config)`, `iqa_pass(config, first=None)`, `timing(config)` and `report(config, first=None)`;
    - `iqa_table(config) -> list[dict]`;
    - `main(argv)`, behind the CLI `python -m degradation_monitor.stages.iqa <fit|pass|timing|report> [--config] [--run] [--first N]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/stages/test_iqa.py`:

```python
import json
import os
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image

from degradation_monitor import corruptions
from degradation_monitor.detectors import COCO_CATEGORY_IDS
from degradation_monitor.runs import Manifest, RunLayout, atomic_npz
from degradation_monitor.stages import iqa as stage

ROWS = ("niqe", "niqe_default", "arniqa", "arniqa_proto", "clipiqa")
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

[clean_ap_floor]
yolo11m = 0.0
"""
IQA = """
base = "base.toml"
run = "{root}/runs/coco-iqa"
reference_run = "{root}/runs/coco"
detectors_config = "detectors.toml"
gpu_memory_gib = 8.0
"""


class FakeIqa:
    """An image-quality stand-in: rows that brighten with the image, and features that make a valid NIQE fit."""
    batch_size, fit_batch_size = 16, 4
    protocol = {"fake": True}
    calls = 0

    def __init__(self, device, niqe_refit=None, prototype=None):
        self.niqe_refit, self.prototype = niqe_refit, prototype

    def fit_features(self, arrays):
        FakeIqa.calls += 1
        n = len(arrays)
        generator = torch.Generator().manual_seed(n)
        features = torch.rand(n, 4, 36, dtype=torch.float64, generator=generator)
        sharpness = torch.tensor([[1.0, 1.0, 1.0, 0.1]] * n)  # three sharp blocks per image
        embedding = torch.tensor([[a.mean() / 255.0] * 8 for a in arrays], dtype=torch.float64)
        if arrays[0].shape[0] < 40:  # the fixture's small images stand in for images without a whole NIQE block
            features = sharpness = None
        return features, sharpness, embedding

    def _rows(self, arrays):
        level = torch.tensor([a.mean() / 255.0 for a in arrays], dtype=torch.float64)
        return {row: level + k for k, row in enumerate(ROWS)}

    def niqe_rows(self, arrays):
        return {**{k: v for k, v in self._rows(arrays).items() if k.startswith("niqe")},
                "niqe_blocks": torch.full((len(arrays),), 4)}

    def arniqa_rows(self, arrays):
        return {k: v for k, v in self._rows(arrays).items() if k.startswith("arniqa")}

    def clipiqa_rows(self, arrays):
        return {"clipiqa": self._rows(arrays)["clipiqa"]}

    def scores(self, arrays):
        FakeIqa.calls += 1
        return {**{k: v.numpy() for k, v in self._rows(arrays).items()}, "niqe_blocks": np.full(len(arrays), 4)}


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    """24 clean train images (four of them too small for a NIQE block), 3 val images and their 96 digests."""
    root = tmp_path_factory.mktemp("images")
    rng = np.random.default_rng(0)
    for folder, count in (("train", 24), ("val", 3)):
        (root / folder).mkdir()
        for index in range(count):
            shape = (32, 64, 3) if folder == "train" and index < 4 else (48, 64, 3)
            Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8)).save(root / folder / f"{index:012d}.jpg")
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
    for name in ("rtdetr.pth", "yolo.pt"):
        (tmp_path / name).write_bytes(name.encode())
    weights = {}
    for name in ("arniqa_encoder", "arniqa_regressor", "niqe_published", "clip_rn50"):
        weights[name] = tmp_path / f"{name}.bin"
        weights[name].write_bytes(name.encode())
    (tmp_path / "base.toml").write_text(BASE.format(root=tmp_path, images=root))
    (tmp_path / "detectors.toml").write_text(DETECTORS.format(root=tmp_path))
    (tmp_path / "iqa.toml").write_text(IQA.format(root=tmp_path))
    monkeypatch.setattr(stage, "load_iqa_models", lambda device, niqe_refit=None, prototype=None:
                        FakeIqa(device, niqe_refit, prototype))
    monkeypatch.setattr(stage, "weight_files", lambda: weights)
    FakeIqa.calls = 0
    reference = RunLayout(tmp_path / "runs" / "coco")  # RT-DETR's run: its digests, read only
    for name, values in digests.items():
        atomic_npz(reference.score_file("detector", name), digests=values)
    return stage.load_config(tmp_path / "iqa.toml")


def _listing(folder):
    return sorted((str(p.relative_to(folder)), p.stat().st_mtime_ns) for p in folder.rglob("*") if p.is_file())


def test_the_iqa_config_lists_every_detectors_run(config):
    assert list(config.detectors) == ["rtdetrv2_r18", "yolo11m"]
    assert config.detectors["rtdetrv2_r18"].run == config.reference_run
    assert config.detectors["yolo11m"].run.name == "yolo11m" and config.gpu_memory_gib == 8.0


def test_the_run_root_keeps_clear_of_every_detectors_run(config, tmp_path):
    yolo = config.detectors["yolo11m"].run
    for root in (config.reference_run, config.reference_run / "inner", yolo, yolo.parent, tmp_path / "runs"):
        with pytest.raises(ValueError, match="must lie outside the detectors' run folders"):
            stage.load_config(tmp_path / "iqa.toml", run=root)
    assert stage.main(["report", "--config", str(tmp_path / "iqa.toml"), "--run", str(yolo.parent)]) == 2


def test_the_fit_writes_the_refit_and_the_prototype_once(config):
    stage.fit(config)
    refit = config.layout.reference("iqa", "niqe_refit.npz")
    with np.load(refit) as data:
        assert data["mu"].shape == (36,) and data["cov"].shape == (36, 36) and int(data["blocks"]) == 20 * 3
    assert np.load(config.layout.reference("iqa", "arniqa_prototype.npy")).shape == (8,)
    record = json.loads(config.layout.reference("iqa", "fit.json").read_text())
    assert record["images"] == 24 and record["niqe_images"] == 20 and record["niqe_skipped"] == 4
    assert record["niqe_dropped_blocks"] == 0
    manifest = json.loads(config.layout.manifest.read_text())
    assert manifest["protocol"]["models"] == {"fake": True} and set(manifest["protocol"]["weights"]) == {
        "arniqa_encoder", "arniqa_regressor", "niqe_published", "clip_rn50"}
    calls = FakeIqa.calls
    stage.fit(config)
    assert FakeIqa.calls == calls


def test_the_pass_writes_the_five_rows_and_resumes(config):
    stage.fit(config)
    stage.iqa_pass(config, first=1)
    assert len(list(config.layout.scores("iqa").glob("*.npz"))) == 1
    stage.iqa_pass(config)
    calls = FakeIqa.calls
    files = sorted(config.layout.scores("iqa").glob("*.npz"))
    assert len(files) == 3
    with np.load(files[0]) as data:
        assert set(data.files) == {*ROWS, "niqe_blocks", "digests"} and all(data[row].shape == (96,) for row in ROWS)
    assert set(json.loads(config.layout.manifest.read_text())["inputs"]) == {"iqa"}
    stage.iqa_pass(config)
    assert FakeIqa.calls == calls


def test_the_pass_refuses_corruptions_that_differ_from_the_reference_run(config):
    stage.fit(config)
    first = config.base.dataset.evaluation_images()[0].name
    atomic_npz(RunLayout(config.reference_run).score_file("detector", first), digests=np.array(["0" * 16] * 96))
    with pytest.raises(ValueError, match="corruptions differ from the reference run"):
        stage.iqa_pass(config)


def test_the_pass_refuses_scores_from_a_changed_fit(config):
    stage.fit(config)
    stage.iqa_pass(config, first=1)
    np.save(config.layout.reference("iqa", "arniqa_prototype.npy"), np.ones(8))
    with pytest.raises(ValueError, match="inputs of scores/iqa changed"):
        stage.iqa_pass(config)


def test_the_report_writes_every_detectors_report_under_the_iqa_run_only(config, monkeypatch):
    stage.fit(config)
    stage.iqa_pass(config)
    reported = []

    def write_report(settings, manifest, out=None, extra_rows=None, extra_inputs=None):
        reported.append((settings.run, out, sorted(extra_rows), extra_rows["niqe"].shape, settings.limit))
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps({"headline": {}, "intervals": {}}))

    monkeypatch.setattr(stage, "write_report", write_report)
    before = _listing(config.reference_run)
    stage.report(config, first=2)
    assert [r[0] for r in reported] == [config.reference_run, config.detectors["yolo11m"].run]
    assert [r[1] for r in reported] == [config.layout.report("rtdetrv2_r18"), config.layout.report("yolo11m")]
    assert all(r[2] == sorted(ROWS) and r[3] == (2, 96) and r[4] == 2 for r in reported)
    assert _listing(config.reference_run) == before  # the reference run is only read
    assert (config.run / "summary.md").exists() and (config.run / "summary.csv").exists()


def test_the_report_refuses_a_detector_scored_on_other_images(config):
    stage.fit(config)
    stage.iqa_pass(config)
    yolo = replace(config.detectors["yolo11m"], limit=2)
    with pytest.raises(ValueError, match="evaluation images of yolo11m differ"):
        stage.report(replace(config, detectors={**config.detectors, "yolo11m": yolo}))
    assert not config.layout.report("rtdetrv2_r18").exists()  # checked before any report is written


def test_the_iqa_table_gives_the_two_axis_score_minus_each_row(config):
    head = {row: {"auroc_common": 0.6, "auroc_extra": 0.55} for row in ROWS}
    head["two_axis"] = {"auroc_common": 0.9, "auroc_extra": 0.85}
    cell = {"point": 0.3, "low": 0.28, "high": 0.32}
    intervals = {s: {f"two_axis - {row}:auroc_{g}": cell for row in ROWS for g in ("common", "extra")}
                 for s in ("all", "untouched")}
    for name in config.detectors:
        folder = config.layout.report(name)
        folder.mkdir(parents=True)
        (folder / "summary.json").write_text(json.dumps({"headline": {"all": head, "untouched": head},
                                                         "intervals": intervals}))
    rows = stage.iqa_table(config)
    assert len(rows) == 2 * 5 and rows[0]["detector"] == "rtdetrv2_r18" and rows[0]["row"] == "niqe"
    assert rows[0]["all_auroc_common"] == 0.6 and rows[0]["untouched_auroc_extra"] == 0.55
    assert rows[0]["all_two_axis_minus_common"] == 0.3 and rows[0]["untouched_two_axis_minus_extra_low"] == 0.28
    lines = (config.run / "summary.md").read_text().splitlines()
    labels = [line.split(" | ")[1] for line in lines if line.startswith("| rtdetrv2_r18 |")]
    assert len(labels) == len(set(labels)) == 5  # the two NIQE rows told apart


def test_timing_records_each_model_and_the_detector(config):
    stage.fit(config)
    (config.reference_run / "timing.json").write_text(json.dumps({"detector_ms": 5.9}))
    stage.timing(config)
    result = json.loads(config.layout.timing.read_text())
    assert {"niqe_ms", "arniqa_ms", "clipiqa_ms"} <= set(result) and result["detector_ms"] == 5.9
    assert result["batch_size"] == 1 and result["images"] == 3


def test_first_applies_to_the_pass_and_report_only(config, tmp_path):
    with pytest.raises(SystemExit):
        stage.main(["fit", "--config", str(tmp_path / "iqa.toml"), "--first", "2"])
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/stages/test_iqa.py`
Expected: FAIL with `ImportError: cannot import name 'iqa' from 'degradation_monitor.stages'`.

- [ ] **Step 3: Implement**

**Move the image loader.** Move `_RgbImages` and `_batches` from `degradation_monitor/stages/detectors.py` into `degradation_monitor/stages/common.py`, as `_RgbImages` and `rgb_batches`:

```python
class _RgbImages(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as source:
            return np.asarray(source.convert("RGB"), dtype=np.uint8).copy()


def rgb_batches(paths, size, workers):
    """Lists of `size` RGB arrays, in the order of `paths`."""
    return DataLoader(_RgbImages(paths), batch_size=size, num_workers=workers, collate_fn=list)


def report_peak_memory(label, device) -> None:
    """The process's peak GPU memory so far, for the sessions that share the card."""
    if torch.device(device).type == "cuda":
        print(f"[{label}] peak GPU memory {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB", flush=True)
```

Move `_report_peak_memory` there too, as `report_peak_memory`, as shown at the end of the block above.

In `stages/detectors.py`:
- delete the three definitions;
- import `report_peak_memory` and `rgb_batches` from `.common`, next to `cap_gpu_memory`;
- replace the four `_batches(` calls with `rgb_batches(`, and the two `_report_peak_memory(` calls with `report_peak_memory(`;
- remove `Image`, `DataLoader` and `Dataset` from its imports if nothing else there uses them.

Create `configs/coco-iqa.toml`:

```toml
# Four detector-free image-quality baselines on COCO-C (docs/superpowers/plans/2026-10-05-iqa-baselines-coco-c.md).
# Stages: python -m degradation_monitor.stages.iqa <fit|pass|timing|report> --config configs/coco-iqa.toml
base = "coco.toml"  # the dataset, its train and evaluation images, the seed, the device and the workers
run = "/home/yuchen/YuchenZ/UE/philip_sa/runs/coco-iqa"
reference_run = "/home/yuchen/YuchenZ/UE/philip_sa/runs/coco"  # RT-DETR's run: the digests; also a detector run
detectors_config = "coco-detectors.toml"  # the other detectors' run folders
gpu_memory_gib = 8.0  # the card is shared
```

Create `degradation_monitor/stages/iqa.py`:

```python
"""Four detector-free image-quality baselines on COCO-C: their clean references, one pass over the corrupted val
images, their timing, and every detector's report with their five rows.

    python -m degradation_monitor.stages.iqa <stage> [--config configs/coco-iqa.toml] [--first N]

Stages:
- fit: from every clean train image, NIQE's pristine model, refitted (estimatemodelparam.m), and ARNIQA's clean
  prototype, the mean embedding;
- pass: every val image's 96 versions, checked against the reference run's digests, through the four models: one
  file per image with the five rows and NIQE's block counts; --first N stops after N images;
- timing: ms per image of each model at batch 1, on the first evaluation images, as the timing stage measures the
  detector;
- report: every detector's report with the five rows added, written under <run>/reports/<detector>/ (the detectors'
  run folders, the read-only reference run included, are only read), then the table of the two-axis score against
  each row in summary.md; --first N reports on the first N evaluation images only.

The models read the image, not the detector, so one set of scores serves every detector. The dataset and its train
and evaluation images come from the base config, so a Cityscapes config can reuse every stage.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from .. import corruptions
from ..baselines.iqa import NIQE_BLOCK, NIQE_SHARPNESS, ROWS, NiqeFit, load_iqa_models, weight_files
from ..cli import positive_int
from ..corruptions import CONDITIONS
from ..datasets.coco import FOLDS
from ..evaluation.report import LABELS
from ..runs import Manifest, RunLayout, atomic_json, atomic_npz, load_npz, progress, sha1, sha256, stack
from ..settings import load_settings
from .baselines import TIMING_IMAGES, TIMING_WARMUP
from .common import cap_gpu_memory, open_rgb, report_peak_memory, rgb_batches, variant_stream
from .detectors import load_config as load_detectors_config
from .report import write_report

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "coco-iqa.toml"
CONFIG_KEYS = {"base", "run", "reference_run", "detectors_config", "gpu_memory_gib"}
REFERENCE_NAME = "rtdetrv2_r18"  # the reference run is RT-DETR's
PROGRESS_IMAGES = 2000
SHORT_LABELS = {"niqe": "NIQE (refit)", "niqe_default": "NIQE (published, sensitivity)", "arniqa": "ARNIQA quality",
                "arniqa_proto": "ARNIQA prototype", "clipiqa": "CLIP-IQA"}  # summary.md's row names


@dataclass(frozen=True)
class IqaConfig:
    base: object  # the base config's Settings: the dataset, the seed, the device and the workers
    run: Path
    reference_run: Path  # RT-DETR's run: the corruption digests, and one of the detectors
    detectors: dict  # detector name -> its Settings, whose run folder holds its scores
    gpu_memory_gib: float

    @property
    def layout(self) -> RunLayout:
        return RunLayout(self.run)


def load_config(path, run=None) -> IqaConfig:
    path = Path(path)
    values = tomllib.loads(path.read_text())
    unknown = sorted(set(values) - CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unknown settings in {path}: {', '.join(unknown)}")
    base = load_settings(path.parent / values["base"])
    reference_run = Path(values["reference_run"])
    if base.run.resolve() != reference_run.resolve():
        raise ValueError(f"{path}: the base config's run must be the reference run")
    others = load_detectors_config(path.parent / values["detectors_config"])
    detectors = {REFERENCE_NAME: base, **{name: others.settings(name) for name in others.detectors}}
    root = Path(run or values["run"])
    for folder in (others.run, *(settings.run for settings in detectors.values())):
        if root.resolve().is_relative_to(folder.resolve()) or folder.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"the run root {root} must lie outside the detectors' run folders and hold none of them, "
                             f"but {folder} is one: choose another run root (--run DIR)")
    return IqaConfig(base=base, run=root, reference_run=reference_run, detectors=detectors,
                     gpu_memory_gib=float(values["gpu_memory_gib"]))


def _manifest(config, models) -> Manifest:
    """The run's manifest, once its protocol is checked: the evaluation, the models' choices and their weights."""
    missing = [name for name, path in weight_files().items() if not Path(path).exists()]
    if missing:
        raise ValueError(f"the model files are missing ({', '.join(missing)}): load the models once (the fit stage)")
    base = config.base
    config.layout.root.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(config.layout)
    manifest.check_protocol({
        "dataset": "coco", "seed": base.seed, "limit": base.limit, "folds": FOLDS,
        "conditions": [list(c) for c in CONDITIONS], "models": models.protocol,
        "weights": {name: sha256(path) for name, path in weight_files().items()},
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32})
    manifest.record_environment(base.discopatch_root)
    return manifest


def _paths_of_references(layout) -> tuple:
    return layout.reference("iqa", "niqe_refit.npz"), layout.reference("iqa", "arniqa_prototype.npy")


def _references(layout) -> tuple:
    refit, prototype = _paths_of_references(layout)
    if not (refit.exists() and prototype.exists()):
        raise ValueError("run the fit stage first: the NIQE refit or the ARNIQA prototype is missing")
    return refit, prototype


def fit(config) -> None:
    """NIQE's refitted pristine model and ARNIQA's clean prototype, from every clean train image."""
    layout = config.layout
    if all(path.exists() for path in (*_paths_of_references(layout), layout.reference("iqa", "fit.json"))):
        return
    base = config.base
    cap_gpu_memory(base.device, config.gpu_memory_gib)
    models = load_iqa_models(base.device)
    _manifest(config, models)
    paths = base.dataset.train_images()
    niqe, total, count, skipped = NiqeFit(), None, 0, 0
    size, started = models.fit_batch_size, time.time()
    for index, arrays in enumerate(rgb_batches(paths, size, base.workers)):
        features, sharpness, embedding = models.fit_features(arrays)
        if features is None:
            skipped += len(arrays)  # no whole NIQE block: left out of the NIQE refit only
        else:
            niqe.update(features, sharpness)
        summed = embedding.double().sum(dim=0)
        total = summed if total is None else total + summed
        count += len(arrays)
        if index % max(1, PROGRESS_IMAGES // size) == 0:
            progress("iqa fit", min((index + 1) * size, len(paths)), len(paths), started)
    mu, cov, blocks = niqe.result()
    refit, prototype = _paths_of_references(layout)
    atomic_npz(refit, mu=mu, cov=cov, images=np.array(niqe.images), blocks=np.array(blocks))
    temporary = prototype.with_name(".arniqa_prototype.tmp.npy")
    np.save(temporary, (total / count).cpu().numpy())
    temporary.replace(prototype)
    atomic_json(layout.reference("iqa", "fit.json"), {
        "images": count, "niqe_images": niqe.images, "niqe_skipped": skipped, "niqe_blocks": blocks,
        "niqe_dropped_blocks": niqe.dropped, "niqe_block": NIQE_BLOCK, "niqe_sharpness": NIQE_SHARPNESS,
        "prototype": "mean of the clean embeddings"})
    report_peak_memory("iqa fit", base.device)


def _loaded_models(config):
    refit, prototype = _references(config.layout)
    with np.load(refit) as data:
        niqe_refit = (data["mu"], data["cov"])
    return load_iqa_models(config.base.device, niqe_refit=niqe_refit, prototype=np.load(prototype)), refit, prototype


def iqa_pass(config, first=None) -> None:
    """Every evaluation image's 96 versions through the four models; resumable image by image."""
    layout, base = config.layout, config.base
    refit, prototype = _references(layout)
    pending = [p for p in base.dataset.evaluation_images() if not layout.score_file("iqa", p).exists()]
    pending = pending[:first] if first else pending
    if not pending:
        return  # before the models load: the card is shared
    cap_gpu_memory(base.device, config.gpu_memory_gib)
    models, _, _ = _loaded_models(config)
    manifest = _manifest(config, models)
    manifest.check_inputs("iqa", {"niqe_refit": sha1(refit), "arniqa_prototype": sha1(prototype)})
    reference, started = RunLayout(config.reference_run), time.time()
    for done, (image, arrays) in enumerate(variant_stream(base, pending), start=1):
        digests = np.array([corruptions.digest(a) for a in arrays])
        if list(load_npz(reference.score_file("detector", image), ("digests",))["digests"]) != list(digests):
            raise ValueError(f"corruptions differ from the reference run for {image}")
        parts = [models.scores(arrays[start:start + models.batch_size])
                 for start in range(0, len(arrays), models.batch_size)]
        rows = {key: np.concatenate([part[key] for part in parts]) for key in (*ROWS, "niqe_blocks")}
        if not all(np.isfinite(rows[row]).all() for row in ROWS):
            raise ValueError(f"an image-quality score is not finite for {image}")
        atomic_npz(layout.score_file("iqa", image), digests=digests, **rows)
        if done % 25 == 0:
            progress("iqa pass", done, len(pending), started)
    report_peak_memory("iqa pass", base.device)


def timing(config) -> None:
    """Median ms per image of each model at batch 1, preprocessing included, as the timing stage measures."""
    base = config.base
    cap_gpu_memory(base.device, config.gpu_memory_gib)
    models, _, _ = _loaded_models(config)
    _manifest(config, models)
    images = [open_rgb(p) for p in base.dataset.evaluation_images()[:TIMING_IMAGES]]
    on_gpu = torch.cuda.is_available() and str(base.device).startswith("cuda")

    def timed(part):
        for array in images[:TIMING_WARMUP]:
            part([array])
        values = []
        for array in images:
            if on_gpu:
                torch.cuda.synchronize()
            start = time.perf_counter()
            part([array])
            if on_gpu:
                torch.cuda.synchronize()
            values.append(1000.0 * (time.perf_counter() - start))
        return float(np.median(values))

    detector = RunLayout(config.reference_run).timing
    result = {"images": len(images), "batch_size": 1, "device": str(base.device),
              "niqe_ms": timed(models.niqe_rows), "arniqa_ms": timed(models.arniqa_rows),
              "clipiqa_ms": timed(models.clipiqa_rows),
              "detector_ms": json.loads(detector.read_text()).get("detector_ms") if detector.exists() else None}
    atomic_json(config.layout.timing, result)
    print(f"[iqa timing] {result}", flush=True)
    report_peak_memory("iqa timing", base.device)


def _cell(value) -> str:
    return "–" if value is None else f"{value:+.3f}"


def iqa_table(config) -> list:
    """For every detector and row: the row's AUROC and the two-axis score minus it, on all and untouched images."""
    rows = []
    for name in config.detectors:
        summary = json.loads((config.layout.report(name) / "summary.json").read_text())
        headline, intervals = summary["headline"], summary["intervals"]
        for row in ROWS:
            entry = {"detector": name, "row": row, "label": LABELS[row]}
            for subset in ("all", "untouched"):
                for group in ("common", "extra"):
                    entry[f"{subset}_auroc_{group}"] = headline.get(subset, {}).get(row, {}).get(f"auroc_{group}")
            for subset in ("all", "untouched"):
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
    lines = ["# The two-axis score against four image-quality baselines", "",
             "Each row's AUROC, and the two-axis score minus it with 95% paired bootstrap intervals, on all images and on "
             "the untouched ones.", "",
             "| Detector | Row | AUROC all: common / extra | untouched: common / extra | Two-axis − row, all: common | "
             "extra | untouched: common | extra |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        aurocs = [" / ".join("–" if r[f"{s}_auroc_{g}"] is None else f"{r[f'{s}_auroc_{g}']:.3f}"
                             for g in ("common", "extra")) for s in ("all", "untouched")]
        cells = [f"{_cell(r[k])} [{_cell(r[k + '_low'])}, {_cell(r[k + '_high'])}]"
                 for k in ("all_two_axis_minus_common", "all_two_axis_minus_extra",
                           "untouched_two_axis_minus_common", "untouched_two_axis_minus_extra")]
        lines.append(f"| {r['detector']} | {SHORT_LABELS[r['row']]} | " + " | ".join(aurocs + cells) + " |")
    temporary = config.run / ".summary.md.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(config.run / "summary.md")
    return rows


def report(config, first=None) -> None:
    """Every detector's report with the five rows, under <run>/reports/<detector>/, then the table."""
    layout = config.layout
    recorded = Manifest(layout).read()
    if "protocol" not in recorded:
        raise ValueError("run the pass stage first: this run folder records no protocol")
    settings_first = replace(config.base, limit=first) if first else config.base
    names = [p.name for p in settings_first.dataset.evaluation_images()]
    rows = stack(layout.scores("iqa"), names, ROWS)
    inputs = {"iqa": {**recorded.get("inputs", {}).get("iqa", {}), "models": recorded["protocol"]["models"],
                      "weights": recorded["protocol"]["weights"], "run": str(config.run)}}
    chosen = {name: replace(settings, limit=first) if first else settings
              for name, settings in config.detectors.items()}
    for name, settings in chosen.items():  # all checked before any report is written
        if [p.name for p in settings.dataset.evaluation_images()] != names:
            raise ValueError(f"the evaluation images of {name} differ from the image-quality pass's")
    for name, settings in chosen.items():
        write_report(settings, Manifest(settings.layout), out=layout.report(name), extra_rows=rows,
                     extra_inputs=inputs)
    iqa_table(config)


STAGES = {"fit": fit, "pass": iqa_pass, "timing": timing, "report": report}
WITH_FIRST = ("pass", "report")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m degradation_monitor.stages.iqa",
                                     description="Four detector-free image-quality baselines on COCO-C: run one stage.")
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

One note on the code above. `report` does not call `_manifest`: it needs no models, and it must not re-check weight files that a later machine may lack. It reads the protocol that the fit and pass stages recorded, and refuses a run folder that records none.

In `README.md`:
- in "Layout", add `configs/coco-iqa.toml  the image-quality baselines' run root and reference run`;
- after the "Three more detectors" subsection of "Running", add:

```markdown
### Image-quality baselines

`python -m degradation_monitor.stages.iqa <fit|pass|timing|report> --config configs/coco-iqa.toml` scores four
detector-free baselines on the same COCO-C images: NIQE (refitted on clean train, and the published model as a
sensitivity row), ARNIQA's KADID-10k quality, ARNIQA's embedding against a clean prototype, and zero-shot CLIP-IQA.
`fit` builds the clean references from all train images, `pass` checks each val image's 96 versions against
`runs/coco/`'s digests and scores them once, `timing` measures each model at batch 1, and `report` writes each
detector's report with the five rows under `runs/coco-iqa/reports/<detector>/` plus `runs/coco-iqa/summary.md`, the
two-axis score against each row. The detectors' own run folders are only read.
```

- [ ] **Step 4: Run the tests to see them pass, then the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/stages/test_iqa.py tests/stages/test_detectors.py`
Expected: PASS. The detectors' tests must still pass after `rgb_batches` moves.

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: `215 passed, 1 skipped, 19 warnings`.

- [ ] **Step 5: Commit**

```bash
git add degradation_monitor/stages/iqa.py configs/coco-iqa.toml degradation_monitor/stages/common.py \
        degradation_monitor/stages/detectors.py README.md tests/stages/test_iqa.py
git commit -m "feat: the image-quality baselines' fit, pass, timing and reports

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

---

### Task 5: Fit the clean references (GPU, about 35–45 minutes)

- [ ] **Step 1: Record the detectors' run folders,** so that Tasks 6 and 8 can show they did not change:

```bash
W=/home/yuchen/YuchenZ/UE/philip_sa/.worktrees/iqa-baselines/.superpowers/sdd/2026-10-05-iqa-baselines-coco-c
find /home/yuchen/YuchenZ/UE/philip_sa/runs/coco /home/yuchen/YuchenZ/UE/philip_sa/runs/coco-detectors -type f \
  -printf '%p %T@ %s\n' | sort > $W/runs-before.txt
wc -l $W/runs-before.txt
```

- [ ] **Step 2: Ask explore for the GPU,** capped at 8 GiB, for about 45 minutes (a peak of about 1–2 GiB is expected). Wait for its answer.

- [ ] **Step 3: Run the fit.** It does not resume: an interruption restarts it from the first image.

```bash
cd /home/yuchen/YuchenZ/UE/philip_sa/.worktrees/iqa-baselines
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/yuchen/miniconda3/envs/UE/bin/python -m degradation_monitor.stages.iqa fit \
  --config configs/coco-iqa.toml > /home/yuchen/YuchenZ/UE/philip_sa/runs/coco-iqa-fit.log 2>&1
```

Expected:
- `runs/coco-iqa/reference/iqa/` holds `niqe_refit.npz` (mean (36,), covariance (36, 36)), `arniqa_prototype.npy` (4096,) and `fit.json`;
- `fit.json` gives 118,287 images and the NIQE counts, including any skipped images;
- the log ends with the peak GPU memory.

- [ ] **Step 4: Send explore "done",** with the peak.

---

### Task 6: A smoke run of 25 images, and the timing (GPU, a few minutes; the card otherwise idle)

- [ ] **Step 1: Ask explore for the GPU** for about 10 minutes, and ask it to keep the card free during the timing.

- [ ] **Step 2: Run the pass on the first 25 images, the timing, and a smoke report**

```bash
cd /home/yuchen/YuchenZ/UE/philip_sa/.worktrees/iqa-baselines
P=/home/yuchen/miniconda3/envs/UE/bin/python; C=configs/coco-iqa.toml
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $P -m degradation_monitor.stages.iqa pass --config $C --first 25
$P -m degradation_monitor.stages.iqa timing --config $C
CUDA_VISIBLE_DEVICES= $P -m degradation_monitor.stages.iqa report --config $C --first 25
```

Expected:
- 25 files in `runs/coco-iqa/scores/iqa/`;
- `runs/coco-iqa/timing.json` with `niqe_ms`, `arniqa_ms`, `clipiqa_ms` and `detector_ms` 5.89;
- four reports under `runs/coco-iqa/reports/`;
- `runs/coco-iqa/summary.md` with 20 rows, "–" for the untouched images.

Also check the pass's time per image and its peak GPU memory, to plan Task 7.

- [ ] **Step 3: Send explore "done".**

- [ ] **Step 4: Compare the GPU's precision with fp32** (CPU, a few minutes). On the GPU, ARNIQA's encoder runs under autocast and CLIP's weights are fp16, as their official code runs them; on the CPU everything is fp32. Rescore the first three smoke images on the CPU and compare them with the pass's files:

```bash
cd /home/yuchen/YuchenZ/UE/philip_sa/.worktrees/iqa-baselines
CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python - <<'EOF'
from dataclasses import replace

import numpy as np
from scipy.stats import kendalltau

from degradation_monitor.baselines.iqa import ROWS
from degradation_monitor.runs import load_npz
from degradation_monitor.stages.common import variant_stream
from degradation_monitor.stages.iqa import _loaded_models, load_config

config = load_config("configs/coco-iqa.toml")
config = replace(config, base=replace(config.base, device="cpu"))
models, _, _ = _loaded_models(config)
gpu, cpu = {row: [] for row in ROWS}, {row: [] for row in ROWS}
for image, arrays in variant_stream(config.base, config.base.dataset.evaluation_images()[:3]):
    stored = load_npz(config.layout.score_file("iqa", image), ROWS)
    parts = [models.scores(arrays[start:start + 16]) for start in range(0, len(arrays), 16)]
    for row in ROWS:
        gpu[row].append(stored[row])
        cpu[row].append(np.concatenate([part[row] for part in parts]))
for row in ROWS:
    a, b = np.concatenate(gpu[row]), np.concatenate(cpu[row])
    print(f"{row:13s} max |GPU - CPU| {np.abs(a - b).max():.1e}, Kendall's tau {kendalltau(a, b).statistic:.4f}")
EOF
```

Expected: Kendall's τ above 0.99 for every row. Note each row's largest difference and τ in the ledger, for the results doc. If a τ is below 0.99, stop and report: the official precision then changes the ranking, and the results doc must discuss it.

- [ ] **Step 5: Check that the detectors' run folders did not change:**

```bash
W=/home/yuchen/YuchenZ/UE/philip_sa/.worktrees/iqa-baselines/.superpowers/sdd/2026-10-05-iqa-baselines-coco-c
find /home/yuchen/YuchenZ/UE/philip_sa/runs/coco /home/yuchen/YuchenZ/UE/philip_sa/runs/coco-detectors -type f \
  -printf '%p %T@ %s\n' | sort | diff $W/runs-before.txt - && echo "the detectors' run folders are unchanged"
```

Expected: `the detectors' run folders are unchanged`. If anything differs, stop and report.

---

### Task 7: The full pass (GPU, about 2.5 hours)

- [ ] **Step 1: Ask explore for the GPU,** with the estimate from Task 6.

- [ ] **Step 2: Run the pass in the background** (it resumes after an interruption):

```bash
cd /home/yuchen/YuchenZ/UE/philip_sa/.worktrees/iqa-baselines
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/yuchen/miniconda3/envs/UE/bin/python -m degradation_monitor.stages.iqa pass \
  --config configs/coco-iqa.toml > /home/yuchen/YuchenZ/UE/philip_sa/runs/coco-iqa-pass.log 2>&1
```

Expected: 5,000 files in `runs/coco-iqa/scores/iqa/`. The 2.5 hours are an estimate from the corruption workers' rate; Task 6's measured rate replaces it.

- [ ] **Step 3: Send explore "done".**

---

### Task 8: Reports, tables and the results

**Files:**
- Create:
  - `docs/results/coco-iqa/`: each detector's report folder, `summary.md`, `summary.csv`, `timing.json`, `fit.json` and `niqe-blocks.csv`;
  - `docs/coco-iqa-results.md`.
- Modify: `docs/README.md`

- [ ] **Step 1: Write the reports and the table** (CPU, up to about 50 GiB of RAM)

The four reports run one after another, about 1.5 hours in all (the three detectors' reports took 57 minutes, with a peak of 52 GB), and Faster R-CNN's peaks near 50 GiB. Check `free -g` first: its `available` column should show at least 60 GiB. Tell explore before starting, because the machine is shared.

```bash
cd /home/yuchen/YuchenZ/UE/philip_sa/.worktrees/iqa-baselines
CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m degradation_monitor.stages.iqa report --config configs/coco-iqa.toml
```

Expected:
- each report's `summary.json` covers 5,000 images, and its rows include the five IQA rows;
- `runs/coco-iqa/summary.md` lists 20 rows (four detectors × five rows), with intervals on all and on the untouched images.

- [ ] **Step 2: Check the reports against the detectors' own, and the detectors' run folders**

Every number the IQA reports share with a detector's own report must be the same, since only rows were added:

```bash
cd /home/yuchen/YuchenZ/UE/philip_sa/.worktrees/iqa-baselines
/home/yuchen/miniconda3/envs/UE/bin/python - <<'EOF'
import json

from degradation_monitor.stages.iqa import load_config

config = load_config("configs/coco-iqa.toml")
for name, settings in config.detectors.items():
    ours = json.loads((config.layout.report(name) / "summary.json").read_text())
    theirs = json.loads((settings.layout.report() / "summary.json").read_text())
    differing = [f"headline {subset} {row}" for subset, rows in theirs["headline"].items() for row, value in rows.items()
                 if ours["headline"].get(subset, {}).get(row) != value]
    differing += [f"interval {subset} {key}" for subset, cells in theirs["intervals"].items()
                  for key, value in cells.items() if ours["intervals"].get(subset, {}).get(key) != value]
    print(name, "matches its own report" if not differing else f"differs in {len(differing)}: {differing[:5]}")
EOF
```

Expected: `matches its own report` for every detector. RT-DETR's own report in `runs/coco/reports/coco/` dates from 2 October, before the four-detector work changed the report code; if it differs, find which rows and why before writing the results doc.

Then run Task 6's Step 5 again. Expected: `the detectors' run folders are unchanged`.

- [ ] **Step 3: Keep the tables**

```bash
R=/home/yuchen/YuchenZ/UE/philip_sa/runs/coco-iqa
mkdir -p docs/results/coco-iqa
cp $R/summary.md $R/summary.csv $R/timing.json $R/reference/iqa/fit.json docs/results/coco-iqa/
for d in rtdetrv2_r18 yolo11m faster_rcnn_r50_fpn_v2 rfdetr_m; do
  mkdir -p docs/results/coco-iqa/$d && cp $R/reports/$d/* docs/results/coco-iqa/$d/
done
CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python - <<'EOF'
import csv

import numpy as np
from PIL import Image

from degradation_monitor.baselines.iqa import NIQE_BLOCK, NIQE_MIN_BLOCKS
from degradation_monitor.corruptions import CONDITIONS
from degradation_monitor.runs import stack
from degradation_monitor.stages.iqa import load_config

config = load_config("configs/coco-iqa.toml")
paths = config.base.dataset.evaluation_images()
blocks = stack(config.layout.scores("iqa"), [p.name for p in paths], ("niqe_blocks",))["niqe_blocks"]
whole = np.array([(height // NIQE_BLOCK) * (width // NIQE_BLOCK) for width, height in (Image.open(p).size for p in paths)])
with open("docs/results/coco-iqa/niqe-blocks.csv", "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["corruption", "severity", "versions_with_a_nan_block", "versions_not_scored"])
    for k, (corruption, severity) in enumerate(CONDITIONS):
        writer.writerow([corruption, severity, int((blocks[:, k] < whole).sum()),
                         int((blocks[:, k] < NIQE_MIN_BLOCKS).sum())])
print("versions NIQE could not score:", int((blocks < NIQE_MIN_BLOCKS).sum()), "of", blocks.size)
EOF
```

- [ ] **Step 4: Write `docs/coco-iqa-results.md`**

Quote every number from the files, with its interval where the report gives one. Include:
- **What each baseline is,** with the deliberate choices and why:
  - NIQE's refit recipe; the counts of skipped train images and dropped blocks (`fit.json`); and that the refit's clean images are JPEG-compressed COCO photos while the published model's were 125 pristine photos, said wherever the two NIQE rows differ;
  - how many corrupted versions NIQE could not score, by condition (`niqe-blocks.csv`), and that they count as degraded. If there are any, say that `conditions.csv`'s mean NIQE for their conditions includes the 1e6 score; the AUROCs only rank them as degraded;
  - ARNIQA run as its paper evaluates it (five crops of the image and of a bicubic half-size version, the encoder under autocast), and how that differs from the README's whole-image example and from pyiqa's port;
  - CLIP-IQA's single prompt pair against pyiqa's five, and its fp16 weights;
  - Task 6's comparison of the GPU's precision with fp32.
- **What each model saw in training,** for the fairness statement, from the Global Constraints: the COCO-C families that also appear among KADID-10k's distortion types, blur (gaussian, motion, and lens ≈ defocus), noise (white ≈ gaussian, impulse, and multiplicative ≈ speckle), JPEG, pixelate, contrast, brightness and saturation; and pixelation among the spatial distortions ARNIQA's encoder was trained on.
- **Per detector, each IQA row's AUROC,** common and extra, on all images and on the untouched ones.
- **The two-axis score minus each row,** with intervals, on all images and on the untouched ones.
- **AUROC by severity, and the families at severities 1, 3 and 5** for the strongest IQA row.
- **The timing table:** each model's ms per image next to the detector's 5.89 ms.

Write each finding in plain words, and say where an IQA row leads. Add the file and the folder to `docs/README.md`.

- [ ] **Step 5: Commit**

```bash
git add docs/results/coco-iqa docs/coco-iqa-results.md docs/README.md
git commit -m "results: four image-quality baselines against the two-axis score on COCO-C

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

- [ ] **Step 6: Message the paper session** with the results doc's path and the headline numbers. For each detector, give the two-axis score minus each IQA row on the common and extra families, with intervals, on all images and on the untouched ones; also give the timing table.
