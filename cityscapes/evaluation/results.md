# Cityscapes-C: the two-axis score and every baseline on two Cityscapes detectors

Run on 5 and 6 October 2026, following `plan.md`, as `design.md` specifies. The tables are in `results/`. There is no pass/fail rule: this page gives the numbers.

## Setting

- **Detectors:** RT-DETRv2-R18 and YOLO11m, each fine-tuned on Cityscapes train and frozen (`../models/README.md`). Each uses its own input:
  - RT-DETR resizes to 640 × 640;
  - YOLO11m letterboxes to 640 × 320.
- **Clean val AP:** our check stage measured 0.378 for RT-DETR and 0.337 for YOLO11m, on all 500 val images and through our adapters. The floors are 0.35 and 0.32. The manifest stores this value under the key `coco_val_ap`, which names COCO's AP metric.
- **Evaluation images:** the 500 Cityscapes val images, in the seed-44 order, in all 96 conditions: clean, plus 15 common and 4 extra `imagecorruptions` families at severities 1–5. The corruptions are applied at 2048 × 1024, as in the standard Cityscapes-C.
- **The method is unchanged:** k = 50, the stage-4 key, stages 1–3 scored, the top 1%. Its 2,000 + 500 reference images are Cityscapes train images (the seed-44 splits).
- **Every baseline's clean reference** comes from the 2,975 Cityscapes train images, without labels:
  - kNN, the activation CDFs and Hashemi et al. use all of them;
  - DisCoPatch was trained on all of them, with the official settings: 256 × 256, 48 patches, 65 epochs;
  - NIQE's pristine model was refitted, and ARNIQA's prototype computed, from all of them.
  - ARNIQA's KADID-10k regressor, zero-shot CLIP-IQA and NIQE's published model are used as published.
- **Rows per detector:**
  - RT-DETR has every row.
  - YOLO11m has no ContrastiveConf and no Hashemi et al., which are DETR-only.
  - DisCoPatch and the image-quality models read the image, not the detector, so their rows are the same for both detectors.
- **Metrics:** AUROC on the common and the extra families, with AUPR and FPR95. Brackets are 95% paired bootstrap intervals over images (1,000 draws, seed 44). There is one image set, all 500 images: nothing was chosen on Cityscapes, so there are no screen, held-out or untouched sets.

## What the numbers show

- **Against every baseline that reads the detector, the two-axis score is ahead on both detectors and both family groups,** and every interval is above 0. The strongest of those baselines is the activation CDFs:
  - RT-DETR: +0.027 [+0.024, +0.031] common and +0.036 [+0.032, +0.040] extra;
  - YOLO11m: +0.036 [+0.032, +0.040] common and +0.036 [+0.033, +0.039] extra.
- **The level score alone is about as good as the two-axis score here.** The flattening arm adds little on Cityscapes:
  - RT-DETR: the two-axis score minus the level score is +0.005 [+0.003, +0.006] common and −0.001 [−0.002, +0.000] extra;
  - YOLO11m: it is −0.002 [−0.003, −0.001] common and −0.003 [−0.004, −0.002] extra, so the level score is slightly ahead.
- **ARNIQA's prototype is the strongest row overall,** at 0.982 / 0.990. It is ahead of the two-axis score on both detectors: by 0.033 / 0.021 on RT-DETR and by 0.061 / 0.047 on YOLO11m, with every interval excluding 0.
  - The same ordering held on COCO-C, where both ARNIQA rows beat the two-axis score (`docs/coco-iqa-results.md`).
  - On Cityscapes, ARNIQA reads 224 × 224 crops of the full 2048 × 1024 image, while the detectors see a 3.2× downscale (see the caveats).
- **The other image-quality rows split by family group:**
  - **NIQE refitted:** on the common families it ties the two-axis score on RT-DETR (−0.002 [−0.005, +0.001]) and is ahead of it on YOLO11m (−0.030 [−0.034, −0.026]). On the extra families it trails by 0.099 and 0.073.
  - **ARNIQA quality:** it ties the two-axis score on the common families on YOLO11m (+0.001 [−0.005, +0.007]) and trails on RT-DETR (+0.029). It trails by 0.10–0.13 on the extra families.
  - **CLIP-IQA is below chance,** at 0.460 / 0.256: zero-shot CLIP-IQA rates the corrupted Cityscapes images as better than the clean ones. Why was not examined.
- **Where the detector-based scores stay weak:**
  - elastic transform: 0.55 / 0.60 / 0.67 on RT-DETR and about 0.51 on YOLO11m, at severities 1 / 3 / 5;
  - pixelate at low severity: 0.59 and 0.53 at severity 1;
  - brightness, glass blur, Gaussian blur and spatter at severity 1.

  From severity 3 on, the two-axis score is at or near 1.00 on most families, on both detectors.
- **Cityscapes-C is easier than COCO-C for the detector-based scores.** RT-DETR's two-axis score reaches 0.949 / 0.970 here, against 0.917 / 0.858 on COCO-C (`docs/coco-detectors-results.md`). Cityscapes' val images and its clean reference are street scenes of one size. Whether that explains the difference was not tested.

## Headline tables

Δ is the two-axis score minus the row: a positive Δ means the two-axis score is ahead.

### RT-DETRv2-R18 (clean mAP 0.378)

| Row | AUROC common ↑ | AUROC extra ↑ | AUPR common ↑ | FPR95 common ↓ |
|---|---|---|---|---|
| **Two-axis (ours)** | 0.949 [0.946, 0.951] | 0.970 [0.967, 0.972] | 0.941 | 0.118 |
| Peak share vs similar scenes | 0.941 [0.939, 0.943] | 0.965 [0.962, 0.968] | 0.935 | 0.127 |
| Level vs similar scenes | 0.944 [0.941, 0.946] | 0.971 [0.968, 0.973] | 0.937 | 0.121 |
| Level vs the average clean image | 0.927 [0.923, 0.930] | 0.954 [0.950, 0.957] | 0.918 | 0.159 |
| Channel means, kNN (control) | 0.944 [0.941, 0.946] | 0.976 [0.974, 0.978] | 0.937 | 0.121 |
| Channel means vs own average (control) | 0.925 [0.921, 0.928] | 0.948 [0.943, 0.952] | 0.914 | 0.166 |
| SAOD, top-3 | 0.746 [0.736, 0.756] | 0.668 [0.659, 0.678] | 0.714 | 0.598 |
| SAOD, min | 0.742 [0.733, 0.751] | 0.661 [0.653, 0.670] | 0.715 | 0.595 |
| ContrastiveConf | 0.583 [0.531, 0.616] | 0.549 [0.519, 0.571] | 0.587 | 0.875 |
| kNN (k = 100) | 0.882 [0.875, 0.888] | 0.850 [0.843, 0.857] | 0.863 | 0.265 |
| DisCoPatch | 0.709 [0.700, 0.719] | 0.733 [0.722, 0.744] | 0.683 | 0.606 |
| Hashemi et al., decoder queries | 0.392 [0.378, 0.405] | 0.359 [0.346, 0.369] | 0.447 | 0.953 |
| Hashemi et al., encoder maps (sensitivity) | 0.596 [0.586, 0.605] | 0.562 [0.554, 0.571] | 0.578 | 0.790 |
| Activation CDFs (Becker et al., ICPR 2026) | 0.921 [0.918, 0.925] | 0.934 [0.929, 0.938] | 0.911 | 0.167 |
| Activation CDFs, plain sum (sensitivity) | 0.906 [0.901, 0.911] | 0.899 [0.892, 0.905] | 0.892 | 0.210 |
| NIQE, refitted | 0.950 [0.948, 0.953] | 0.871 [0.869, 0.873] | 0.943 | 0.112 |
| NIQE, published (sensitivity) | 0.930 [0.929, 0.932] | 0.804 [0.803, 0.806] | 0.939 | 0.128 |
| ARNIQA quality | 0.920 [0.915, 0.925] | 0.843 [0.837, 0.848] | 0.913 | 0.220 |
| **ARNIQA prototype** | **0.982** [0.980, 0.984] | **0.990** [0.989, 0.992] | **0.976** | **0.048** |
| CLIP-IQA | 0.460 [0.449, 0.470] | 0.256 [0.246, 0.266] | 0.522 | 0.803 |

| Two-axis minus | Δ AUROC common | Δ AUROC extra |
|---|---|---|
| Level vs similar scenes | +0.005 [+0.003, +0.006] | −0.001 [−0.002, +0.000] |
| Activation CDFs | +0.027 [+0.024, +0.031] | +0.036 [+0.032, +0.040] |
| kNN (k = 100) | +0.067 [+0.061, +0.073] | +0.120 [+0.113, +0.126] |
| SAOD, min | +0.207 [+0.198, +0.216] | +0.309 [+0.300, +0.317] |
| DisCoPatch | +0.240 [+0.230, +0.248] | +0.237 [+0.226, +0.247] |
| ContrastiveConf | +0.366 [+0.332, +0.417] | +0.421 [+0.399, +0.451] |
| Hashemi et al., decoder queries | +0.556 [+0.543, +0.570] | +0.611 [+0.601, +0.624] |
| NIQE, refitted | −0.002 [−0.005, +0.001] | +0.099 [+0.096, +0.102] |
| ARNIQA quality | +0.029 [+0.024, +0.034] | +0.127 [+0.122, +0.134] |
| ARNIQA prototype | −0.033 [−0.036, −0.030] | −0.021 [−0.023, −0.018] |
| CLIP-IQA | +0.488 [+0.477, +0.499] | +0.714 [+0.705, +0.724] |

### YOLO11m (clean mAP 0.337)

| Row | AUROC common ↑ | AUROC extra ↑ | AUPR common ↑ | FPR95 common ↓ |
|---|---|---|---|---|
| **Two-axis (ours)** | 0.921 [0.917, 0.925] | 0.944 [0.939, 0.947] | 0.908 | 0.164 |
| Peak share vs similar scenes | 0.916 [0.912, 0.920] | 0.939 [0.934, 0.943] | 0.904 | 0.177 |
| Level vs similar scenes | 0.923 [0.919, 0.927] | 0.947 [0.942, 0.951] | 0.910 | 0.158 |
| Level vs the average clean image | 0.899 [0.893, 0.905] | 0.921 [0.916, 0.925] | 0.876 | 0.201 |
| Channel means, kNN (control) | 0.911 [0.904, 0.917] | 0.948 [0.942, 0.953] | 0.879 | 0.171 |
| Channel means vs own average (control) | 0.888 [0.880, 0.895] | 0.909 [0.903, 0.915] | 0.853 | 0.220 |
| SAOD, top-3 | 0.749 [0.744, 0.755] | 0.728 [0.722, 0.734] | 0.738 | 0.555 |
| SAOD, min | 0.734 [0.728, 0.740] | 0.713 [0.708, 0.718] | 0.725 | 0.569 |
| kNN (k = 100) | 0.780 [0.774, 0.787] | 0.760 [0.756, 0.764] | 0.777 | 0.464 |
| DisCoPatch | 0.709 [0.700, 0.719] | 0.733 [0.722, 0.744] | 0.683 | 0.606 |
| Activation CDFs (Becker et al., ICPR 2026) | 0.885 [0.878, 0.891] | 0.908 [0.902, 0.913] | 0.859 | 0.230 |
| Activation CDFs, plain sum (sensitivity) | 0.831 [0.823, 0.838] | 0.828 [0.822, 0.834] | 0.810 | 0.366 |
| NIQE, refitted | 0.950 [0.948, 0.953] | 0.871 [0.869, 0.873] | 0.943 | 0.112 |
| NIQE, published (sensitivity) | 0.930 [0.929, 0.932] | 0.804 [0.803, 0.806] | 0.939 | 0.128 |
| ARNIQA quality | 0.920 [0.915, 0.925] | 0.843 [0.837, 0.848] | 0.913 | 0.220 |
| **ARNIQA prototype** | **0.982** [0.980, 0.984] | **0.990** [0.989, 0.992] | **0.976** | **0.048** |
| CLIP-IQA | 0.460 [0.449, 0.470] | 0.256 [0.246, 0.266] | 0.522 | 0.803 |

| Two-axis minus | Δ AUROC common | Δ AUROC extra |
|---|---|---|
| Level vs similar scenes | −0.002 [−0.003, −0.001] | −0.003 [−0.004, −0.002] |
| Activation CDFs | +0.036 [+0.032, +0.040] | +0.036 [+0.033, +0.039] |
| kNN (k = 100) | +0.141 [+0.134, +0.147] | +0.184 [+0.178, +0.189] |
| SAOD, min | +0.187 [+0.181, +0.193] | +0.231 [+0.225, +0.236] |
| DisCoPatch | +0.212 [+0.203, +0.221] | +0.211 [+0.201, +0.220] |
| NIQE, refitted | −0.030 [−0.034, −0.026] | +0.073 [+0.069, +0.077] |
| ARNIQA quality | +0.001 [−0.005, +0.007] | +0.101 [+0.094, +0.108] |
| ARNIQA prototype | −0.061 [−0.065, −0.057] | −0.047 [−0.051, −0.043] |
| CLIP-IQA | +0.460 [+0.451, +0.472] | +0.688 [+0.679, +0.698] |

Each detector's `report.md` and `summary.json` hold every difference, together with AUPR and FPR95 for both family groups.

## AUROC by severity

Common and extra families, severities 1–5, all 500 images. The image-quality row shown is the strongest one, ARNIQA's prototype; the published NIQE is a sensitivity row and is never chosen.

| Detector | Row | Common | Extra |
|---|---|---|---|
| RT-DETRv2-R18 | Two-axis | 0.905 / 0.939 / 0.955 / 0.968 / 0.975 | 0.856 / 0.998 / 0.995 / 1.000 / 1.000 |
| | Level vs similar scenes | 0.898 / 0.932 / 0.949 / 0.965 / 0.975 | 0.860 / 0.998 / 0.996 / 1.000 / 1.000 |
| | Activation CDFs | 0.854 / 0.904 / 0.935 / 0.951 / 0.963 | 0.753 / 0.947 / 0.968 / 1.000 / 1.000 |
| YOLO11m | Two-axis | 0.863 / 0.909 / 0.933 / 0.944 / 0.955 | 0.780 / 0.978 / 0.960 / 1.000 / 1.000 |
| | Level vs similar scenes | 0.868 / 0.911 / 0.934 / 0.945 / 0.956 | 0.791 / 0.980 / 0.963 / 1.000 / 1.000 |
| | Activation CDFs | 0.814 / 0.863 / 0.897 / 0.917 / 0.933 | 0.715 / 0.914 / 0.924 / 0.989 / 0.997 |
| Both (detector-free) | ARNIQA prototype | 0.974 / 0.984 / 0.984 / 0.984 / 0.983 | 0.963 / 1.000 / 0.989 / 1.000 / 1.000 |

## AUROC by family at severities 1 / 3 / 5

The two-axis score on each detector, and ARNIQA's prototype, which reads no detector. `*` marks the extra families.

| Family | Two-axis, RT-DETRv2-R18 | Two-axis, YOLO11m | ARNIQA prototype |
|---|---|---|---|
| gaussian noise | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| shot noise | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| impulse noise | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| defocus blur | 0.97 / 1.00 / 1.00 | 0.90 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| glass blur | 0.84 / 0.99 / 1.00 | 0.70 / 0.94 / 1.00 | 1.00 / 1.00 / 1.00 |
| motion blur | 0.87 / 1.00 / 1.00 | 0.80 / 0.99 / 1.00 | 1.00 / 1.00 / 1.00 |
| zoom blur | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| snow | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| frost | 1.00 / 1.00 / 1.00 | 0.99 / 1.00 / 1.00 | 0.99 / 1.00 / 1.00 |
| fog | 0.99 / 1.00 / 1.00 | 0.98 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| brightness | 0.79 / 1.00 / 1.00 | 0.64 / 0.95 / 1.00 | 0.82 / 0.99 / 1.00 |
| contrast | 0.98 / 1.00 / 1.00 | 0.95 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| elastic transform | 0.55 / 0.60 / 0.67 | 0.51 / 0.51 / 0.52 | 0.80 / 0.77 / 0.75 |
| pixelate | 0.59 / 0.75 / 0.96 | 0.53 / 0.62 / 0.82 | 1.00 / 1.00 / 1.00 |
| jpeg compression | 0.99 / 1.00 / 1.00 | 0.95 / 0.99 / 1.00 | 1.00 / 1.00 / 1.00 |
| speckle noise * | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| gaussian blur * | 0.77 / 1.00 / 1.00 | 0.68 / 0.99 / 1.00 | 1.00 / 1.00 / 1.00 |
| spatter * | 0.69 / 1.00 / 1.00 | 0.61 / 1.00 / 1.00 | 0.86 / 1.00 / 1.00 |
| saturate * | 0.96 / 0.98 / 1.00 | 0.83 / 0.85 / 1.00 | 1.00 / 0.96 / 1.00 |

## Detector mAP by severity

The mAP of the clean condition, and the mean over each family group at severities 1–5. It is context only: the goal is to tell corrupted images from clean ones.

| Detector | Clean | Common, severities 1–5 | Extra, severities 1–5 |
|---|---|---|---|
| RT-DETRv2-R18 | 0.378 | 0.307 / 0.270 / 0.228 / 0.183 / 0.152 | 0.367 / 0.321 / 0.284 / 0.259 / 0.187 |
| YOLO11m | 0.337 | 0.221 / 0.192 / 0.164 / 0.142 / 0.116 | 0.293 / 0.226 / 0.176 / 0.143 / 0.092 |

## DisCoPatch's training

DisCoPatch trained for 65 epochs on the 2,975 train images, from 05:19 to 06:10 on 6 October. That is about 2,900 optimiser steps, against about 115,000 on COCO's 118k images.

Its per-epoch losses, read from its progress bar, levelled off. Over the last 20 epochs:
- the generator loss went from 0.0463 to 0.0455 (the mean of epochs 46–55 against that of 56–65);
- the discriminator loss went from 0.1094 to 0.1142. It is noisy from epoch to epoch (0.06–0.14), but its mean is steady.

## Caveats

- **The corruptions are applied before the detectors' downscale.** At 2048 × 1024, the 3.2× downscale to the detectors' input makes noise, blur, pixelation and JPEG milder than on COCO-C at the same severity. Fog, snow, frost, brightness, contrast and saturate barely change. Per-severity comparisons with COCO-C are therefore inexact; comparisons between methods on these images are not affected.
- **The image-quality models read the full 2048 × 1024 image,** through their own preprocessing. ARNIQA's crops and NIQE see the corruption at full strength. On COCO they read about the detector's scale, so this advantage is new on Cityscapes. A deployed image-quality monitor would get the camera image too.
- **DisCoPatch reads 256 × 256, and trained for about 2,900 steps** (above).
- **Val has only 500 images.** The intervals account for the sampling of images, but not for the narrow scene distribution of one city dataset.
- **ContrastiveConf's λ is cross-fitted on the 5 folds** from the AP of each clean image. Its interval on RT-DETR (0.583 [0.531, 0.616]) is the widest of any row.

## How to rerun

From the repository root, after `cityscapes/models/` (both detectors frozen):

```bash
PY=/home/yuchen/miniconda3/envs/UE/bin/python
C="--config cityscapes/evaluation/cityscapes.toml"
for stage in check knn-bank detector-pass hashemi-fit cdf-fit cdf-zstats activation-pass method-reference method-pass \
             discopatch-train discopatch-pass report; do $PY -m degradation_monitor $stage $C || break; done
D="--config cityscapes/evaluation/cityscapes-detectors.toml"
for stage in check fit pass report; do $PY -m degradation_monitor.stages.detectors $stage $D || break; done
Q="--config cityscapes/evaluation/cityscapes-iqa.toml"
for stage in fit pass report; do $PY -m degradation_monitor.stages.iqa $stage $Q || break; done
```

The run folders are `runs/cityscapes/`, `runs/cityscapes-detectors/` and `runs/cityscapes-iqa/`. The final reports, with every row, are under `runs/cityscapes-iqa/reports/`, and copies are in `results/`.

## Files in `results/`

- `rtdetrv2_r18/` and `yolo11m/`: each detector's final report, with every row: `report.md`, `summary.json`, and the separation, aggregate, interval, condition and kNN-by-k tables.
- `detectors-summary.md` and `.csv`: the cross table of both detectors.
- `iqa-summary.md` and `.csv`: the two-axis score against each image-quality row.
