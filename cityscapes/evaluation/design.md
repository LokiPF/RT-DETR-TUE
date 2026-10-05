# Cityscapes-C numbers for both detectors: design

5 October 2026. This is plan 2 of 2. Plan 1 (`../models/`) trained the two detectors.

## Goal

The numbers of the two-axis score and of every baseline on Cityscapes-C, for RT-DETRv2-R18 and YOLO11m fine-tuned on Cityscapes, reported as on COCO.

## Decided

- **No pass/fail rule** (the user, 5 October). The report gives the numbers with their intervals and no decision lines.
  - This drops the rule pre-registered on 4 October in `docs/superpowers/plans/2026-10-04-cityscapes-c-evaluation.md`.
  - That old plan is replaced by this folder, and gets a one-line pointer here.
- **Detectors:** both are frozen, and each keeps its own preprocessing (`../models/README.md`).
  - RT-DETRv2-R18: `RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_cityscapes_72e.pth`, resized to 640 × 640.
  - YOLO11m: `lab/Detector_test/yolo11m_cityscapes_100e.pt`, letterboxed to 640 × 320.
  - Their taps and maps are the ones used on COCO (`docs/coco-detectors-results.md`).
- **Rows:**
  - **Both detectors:**
    - the two-axis score, with its arms and the pilot's control rows;
    - SAOD (top-3 and min) and kNN;
    - DisCoPatch;
    - the activation CDFs, with the plain-sum sensitivity row;
    - the 5 IQA rows: NIQE refitted, NIQE published (a sensitivity row), ARNIQA quality, ARNIQA prototype, CLIP-IQA.
  - **RT-DETR also has** ContrastiveConf and Hashemi et al. (decoder queries, with the encoder maps as a sensitivity row). YOLO11m has neither, because both are DETR-only, as on COCO.
  - **DisCoPatch and the IQA models read the image, not the detector,** so one set of their scores serves both detectors.
- **Clean references:** all come from the 2,975 Cityscapes train images, without labels.
  - **The method:** the bank of 2,000 and the z-statistics of 500, the seed-44 splits.
  - **kNN, the activation CDFs (fit and z-statistics) and Hashemi's statistics:** all 2,975.
  - **DisCoPatch:** trained on all 2,975 with the same official settings as on COCO: 256 × 256, 48 patches per image, 65 epochs.
    - That is about 2,900 optimiser steps, against about 115,000 on COCO.
    - Before its scores are used, check that its training loss has levelled off.
  - **IQA:**
    - NIQE's refit and ARNIQA's prototype come from all 2,975 images.
    - ARNIQA's KADID-10k regressor and zero-shot CLIP-IQA stay as published.
- **Evaluation images:** all 500 Cityscapes val images, in the seed-44 order.
  - **Folds:** 5. Only ContrastiveConf's λ uses them, cross-fitted from the AP per image against `cityscapes_val_8cls.json`.
  - **One image set,** all 500. There are no screen, untouched or held-out sets, because nothing was chosen on Cityscapes.
- **Conditions:** the 96 used on COCO: clean, plus 15 common and 4 extra families at severities 1–5.
  - They are applied at 2048 × 1024 by the package's seeded `corruptions`, as in the standard Cityscapes-C.
  - Every method then reads its own input: the detectors 640 × 640 or 640 × 320, DisCoPatch 256 × 256, and the IQA models the full 2048 × 1024 image through their own preprocessing.
- **Metrics,** as on COCO:
  - AUROC on the common and the extra families (the main metric), AUPR and FPR95;
  - paired whole-image bootstrap intervals, 1,000 draws, seed 44;
  - the same by severity and by family;
  - the two-axis score minus each baseline, with intervals;
  - mAP per condition, as context.
- **The method's fixed choices stay as they are:** k = 50, the stage-4 key, stages 1–3, the top 1%, 2,000 + 500 reference images.
- **Check floors:** each detector's clean val AP must reach its toolkit's val AP minus 0.03, as on COCO.
  - RT-DETR: 0.382, so the floor is 0.35.
  - YOLO11m: the floor is set from Ultralytics' val of the frozen weights.
- **COCO stays as it is:** `configs/coco*.toml`, the COCO protocols and the run folders under `runs/coco*` don't change, and the golden and equivalence tests stay green.
- **Little bookkeeping:** nothing beyond what the package records already.

## Code changes, in `degradation_monitor/`

1. **The RT-DETR builder takes a class count,** and the loader reads it from the checkpoint: 80 for COCO, 8 for Cityscapes.
2. **A Cityscapes dataset:**
   - the images sit in one folder per city and are listed recursively; their names are unique, and a repeated name is refused;
   - the ground truth is found by base name;
   - it has 8 classes, its own clean-AP floor, and no screening history.
3. **A `benchmark` setting** chooses the dataset. The protocols' `"dataset"` field follows it, in `Settings.protocol()` and in the IQA stage's own protocol.
4. **The check stages follow the benchmark:**
   - RT-DETR's check and the detectors stage's check list the val images through the dataset, so city subfolders are included;
   - they compare the detector's classes with the annotation file's, and use the benchmark's floor.
5. **Reports for a benchmark without a screening history:**
   - one image set (`all`), no decision lines, and a title that names the benchmark;
   - the detectors' cross table and the IQA summary table lose their untouched-image columns.

## Configs, in `cityscapes/evaluation/`

- **`cityscapes.toml`:** RT-DETR's run in `runs/cityscapes/`, with `benchmark = "cityscapes"`, the Cityscapes checkpoint, the Cityscapes train and val images and the val annotations.
- **`cityscapes-detectors.toml`:**
  - base `cityscapes.toml`;
  - run `runs/cityscapes-detectors/`, with reference run `runs/cityscapes/`;
  - the Cityscapes YOLO11m and its floor.
- **`cityscapes-iqa.toml`:**
  - base `cityscapes.toml`;
  - run `runs/cityscapes-iqa/`, with reference run `runs/cityscapes/`;
  - detectors config `cityscapes-detectors.toml`.

## Runs

- **Order:**
  1. RT-DETR's stages: `check`, `knn-bank`, `detector-pass`, `hashemi-fit`, `cdf-fit`, `cdf-zstats`, `activation-pass`, `method-reference`, `method-pass`, `discopatch-train`, `discopatch-pass`, `report`.
  2. YOLO11m's stages: `check`, `fit`, `pass`, `report`.
  3. The IQA stages: `fit`, `pass`, `report`.
- **First, a smoke run on a few val images,** to time each pass.
- **Cost:** every pass regenerates the corruptions. That happens about 6 times, at roughly 1 h of CPU each, for 500 images at 2048 × 1024 (88 s per image on one core). With the fits, DisCoPatch's training and the IQA pass (about 2 h), the total is roughly 8–12 h, mostly on the CPU.
- **GPU:** every stage runs under a memory cap. Ask explore, and ue-implement while its work runs, before each GPU job, and send "done" after.

## Results

- **`cityscapes/evaluation/results.md`:**
  - the setting;
  - for each detector, the headline table: AUROC, AUPR and FPR95 for the common and the extra families, with intervals, for every row;
  - AUROC by severity, and the families at severities 1, 3 and 5;
  - the two-axis score minus each baseline;
  - the mAP by severity;
  - the caveats:
    - after the 3.2× downscale, noise, blur, pixelation and JPEG are milder than on COCO-C at the same severity;
    - the IQA models read the full 2048 × 1024 image, where on COCO they read about the detector's scale;
    - DisCoPatch reads 256 × 256, and trained for about 2,900 steps;
    - val has only 500 images.
- **`cityscapes/evaluation/results/`:** each report folder, copied from the runs.

## Where and when

- **The package changes go in a worktree branch,** cut from `fingerprint_bank` after `iqa-baselines` is merged, and merged back at the end:
  - plan 2 builds on the IQA stage;
  - the paper scripts import the package from the main checkout, so half-finished edits mustn't land there.
- **The run folders `runs/cityscapes*/`** sit in the main checkout's `runs/`, git-ignored, like COCO's.

## Not in this plan

- Foggy Cityscapes and ACDC, which still need downloads.
- The runtime of the method and of the baselines on Cityscapes.
