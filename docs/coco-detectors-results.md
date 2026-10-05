# The two-axis score on four COCO detectors

Run on 4 and 5 October 2026, following `docs/superpowers/plans/2026-10-04-four-detectors-coco-c.md`. The method ran unchanged on three more frozen COCO detectors, next to RT-DETRv2-R18. The tables are in `docs/results/coco-detectors/`. RT-DETRv2-R18's numbers are those of the 2 October confirmation (`docs/conv-tu-conditioned-results.md`), as the clean branch reports them in `docs/results/coco/`.

**What was scored:**
- **Detectors:** RT-DETRv2-R18, YOLO11m, Faster R-CNN R50-FPN v2 and RF-DETR-M. All four are trained on COCO and used frozen, as released, each with its own preprocessing.
- **Images:** all 5,000 COCO val2017 images, in the seed-44 order, in all 96 conditions. The three new detectors saw RT-DETR's corrupted images: one shared pass made each image's 96 versions, checked them against `runs/coco/`'s digests and fed them to all three.
- **The method, unchanged:** k = 50 neighbours, the top 1%, a bank of 2,000 and 500 z-statistics images (COCO train, the seed-44 splits), the three earliest feature levels scored and the deepest as the content key, and the larger of the flattening and level arms.
- **Baselines:** every one each detector supports, on the same images (below).
- **Rules:** RT-DETR's two pre-registered rules, applied to each detector's own scores: the headline rule and the level rule (`headline_decision` and `level_decision` in `degradation_monitor/evaluation/report.py`). Every detector is reported, whatever its outcome.

## How to read the tables

↑ means higher is better and ↓ means lower is better.

| Column | What it measures | Reference values |
|---|---|---|
| AUROC common ↑ | How well the score tells a corrupted image from its clean version, averaged over the 75 conditions of the 15 common families | 0.5 = chance, 1 = perfect |
| AUROC extra ↑ | The same, over the 20 conditions of the 4 extra families | 0.5 = chance, 1 = perfect |
| FPR95 ↓ | The share of clean images flagged when the threshold catches 95% of the corrupted ones | 0 = perfect |

- Brackets are 95% paired bootstrap intervals over images (1,000 draws, seed 44). In a difference (first row minus second), a positive Δ AUROC means the first row is better.
- The untouched images are the 3,030 at positions 1970 and later. Nobody read them while the method or the new taps were chosen.
- The tables by severity, by family and by arm have no intervals.

## Verdict

The pre-registered outcomes, as each detector's `summary.json` gives them:

| Detector | Headline: the two-axis score | The level score |
|---|---|---|
| RT-DETRv2-R18 | confirmed | confirmed |
| YOLO11m | confirmed | confirmed |
| Faster R-CNN R50-FPN v2 | ahead of the activation CDFs, but the flattening adds nothing over the level score on the common families | confirmed |
| RF-DETR-M | confirmed | confirmed |

- **The two-axis score beats the strongest baseline on every detector.** That baseline is the activation CDFs (Becker et al., ICPR 2026) on all four. The gain holds on the common and the extra families, on all images and on the untouched ones, with every interval above 0.
- **The margin is smaller off RT-DETR.** On the common families it is +0.096 on RT-DETR, against +0.055, +0.048 and +0.044 on YOLO11m, Faster R-CNN and RF-DETR-M.
- **The flattening arm adds over the level score on three of the four detectors.** On Faster R-CNN it adds +0.001 [−0.002, +0.003] on the common families on all images, and +0.001 [−0.003, +0.004] on the untouched ones. Both intervals include 0, so the rule's last clause fails and the headline is partial there. The section on Faster R-CNN below shows why.
- **The level score's rule holds on all four.**
- **RT-DETR's split of the arms does not carry over as such.** On RT-DETR fog flattens the channels and noise re-weights them, and YOLO11m behaves the same way. Faster R-CNN's flattening arm is weak, and on RF-DETR-M's ViT the two arms swap roles. The two-axis score still catches fog and noise on every detector.

## The four detectors

| | RT-DETRv2-R18 | YOLO11m | Faster R-CNN R50-FPN v2 | RF-DETR-M |
|---|---|---|---|---|
| Design | real-time DETR | one-stage, anchor-free, with NMS | two-stage, with an FPN | real-time DETR |
| Backbone | ResNet-18 (vd) | CSP backbone with SiLU | ResNet-50 | DINOv2 ViT-S, 12 blocks, patch 16 |
| Weights | `rtdetrv2_r18vd_120e_coco_rerun_48.1.pth` | `yolo11m.pt`, Ultralytics 8.3.235 | torchvision's `COCO_V1` | `rf-detr-medium.pth`, rfdetr 1.11.2 |
| Input | resized to 640 × 640 | letterboxed: long side 640, short side padded to a multiple of 32 | short side 800, long side at most 1333, padded to a multiple of 32 | resized to 576 × 576 |
| Scored levels s1, s2, s3 | first block of stages 1–3 | layers 1, 3, 5 | `layer1[0]`, `layer2[0]`, `layer3[0]` | blocks 1, 2, 3 |
| Key s4 | first block of stage 4 | layer 7 | `layer4[0]` | block 12 |
| Channels s1 / s2 / s3 / key | 64 / 128 / 256 / 512 | 128 / 256 / 512 / 512 | 256 / 512 / 1,024 / 2,048 | 384 / 384 / 384 / 384 |

Padding never enters a statistic: every map is cropped to the cells that lie wholly on the image.

**The tap rule,** fixed on 4 October before any run: score the three earliest feature levels, and use the deepest as the content key. A level is a stage in a hierarchical backbone and a block in a plain ViT. RT-DETR reads the output of each stage's first block, and the CNNs copy that:
- Faster R-CNN reads the first bottleneck block of each stage. The rule set these taps, without an exploration.
- YOLO11m reads each stage's first module, its stride-2 downsampling conv: strides 4, 8 and 16, and 32 for the key.
- RF-DETR-M reads the raw outputs of ViT blocks 1, 2 and 3, before the backbone's LayerNorm, with block 12 as the key. Its patch tokens are regathered from the attention windows, without the class token.

**YOLO's and the ViT's taps were chosen on development images.** Both explorations ran on 4 October, on the same 300 images, drawn with seed 7 from positions 0–1969. The untouched images stayed unread. The scripts and logs are in `docs/results/coco-detectors/tap-exploration/`. For YOLO11m, the two-axis AUROC on those 300 images, as the log gives it:

| YOLO11m taps | AUROC common / extra | Minus the stage blocks, common / extra |
|---|---|---|
| 2, 4, 6 + key 8 (the stage blocks, the plan's first choice) | 0.8813 / 0.8404 | — |
| 2, 4, 6 + key 10 | 0.8678 / 0.8281 | −0.0135 [−0.0188, −0.0081] / −0.0122 [−0.0168, −0.0079] |
| **1, 3, 5 + key 7 (chosen)** | **0.8979 / 0.8755** | +0.0166 [+0.0095, +0.0236] / +0.0352 [+0.0295, +0.0408] |
| 1, 2, 3 + key 10 | 0.8932 / 0.8790 | +0.0119 [+0.0032, +0.0198] / +0.0386 [+0.0316, +0.0452] |
| 0, 1, 2 + key 10 | 0.8869 / 0.8862 | +0.0057 [−0.0059, +0.0158] / +0.0458 [+0.0364, +0.0543] |

The chosen set leads on the common families, and it mirrors RT-DETR's taps, which read each stage's first block. On the extra families the last two sets are ahead of it.

For the ViT, blocks 1, 2 and 3 with key 12 beat blocks 3, 6 and 9 with key 12: 0.8591 / 0.8371 against 0.7880 / 0.7394 on the raw block outputs. Blocks 3, 6 and 9 minus blocks 1, 2 and 3 is −0.0711 [−0.0850, −0.0566] common and −0.0977 [−0.1099, −0.0845] extra. The layer-normed outputs give the same order: 0.8482 / 0.8431 against 0.7716 / 0.7460.

**The baselines read these maps:**
- **Activation CDFs, five maps per detector.**
  - YOLO11m: the stem (layer 0, stride 2) and the four stage outputs, layers 2, 4, 6 and 10. Layer 10 ends the backbone, after SPPF and C2PSA.
  - Faster R-CNN: the stem after max pooling and the four stage outputs.
  - RF-DETR-M: the embeddings and blocks 1, 2, 3 and 12, because a ViT has no stage outputs.
- **kNN:** YOLO11m's layer 10, Faster R-CNN's `layer4` output and RF-DETR-M's block 12 after the backbone's LayerNorm, each mean-pooled and L2-normalised. The bank is all 118,287 COCO train images.
- **SAOD:** each detector's top 100 detections.

### What each detector lacks

| Baseline | RT-DETRv2-R18 | YOLO11m | Faster R-CNN | RF-DETR-M |
|---|---|---|---|---|
| SAOD, kNN, DisCoPatch, activation CDFs | yes | yes | yes | yes |
| ContrastiveConf | yes | no | no | yes |
| Hashemi et al., decoder queries | yes | no | no | yes |
| Hashemi et al., encoder maps (a sensitivity row) | yes | no | no | no |

- **ContrastiveConf and Hashemi et al. exist for DETR-type detectors only,** so YOLO11m and Faster R-CNN have neither. Their reports compare the method with four baselines: SAOD, kNN, DisCoPatch and the activation CDFs. RF-DETR-M's report has six.
- **RF-DETR-M has no encoder maps,** so its Hashemi row reads the decoder only, and the encoder sensitivity row is missing.
- **DisCoPatch reads the image, not the detector.** Its scores are RT-DETR's, hard-linked, so its row is the same for every detector.
- **No runtime** was measured for the three new detectors.

## Headline: all images and the untouched ones

| Detector | Row | All, common ↑ | All, extra ↑ | Untouched, common ↑ | Untouched, extra ↑ | FPR95 all, common ↓ |
|---|---|---|---|---|---|---|
| RT-DETRv2-R18 | **Two-axis (headline)** | **0.917** [0.914, 0.920] | 0.858 [0.856, 0.861] | **0.916** [0.912, 0.920] | 0.858 [0.854, 0.862] | **0.226** |
|  | Peak share vs similar scenes | 0.898 [0.895, 0.902] | 0.861 [0.858, 0.864] | 0.899 [0.895, 0.903] | 0.861 [0.857, 0.865] | 0.250 |
|  | Level vs similar scenes | 0.841 [0.837, 0.844] | **0.866** [0.863, 0.869] | 0.840 [0.835, 0.845] | **0.865** [0.861, 0.869] | 0.361 |
|  | Level vs the average clean image | 0.800 [0.796, 0.803] | 0.830 [0.827, 0.832] | 0.799 [0.795, 0.804] | 0.829 [0.825, 0.833] | 0.439 |
|  | SAOD, top-3 | 0.686 [0.682, 0.690] | 0.625 [0.622, 0.628] | 0.683 [0.678, 0.688] | 0.622 [0.618, 0.626] | 0.729 |
|  | SAOD, min | 0.731 [0.727, 0.734] | 0.664 [0.660, 0.667] | 0.731 [0.727, 0.736] | 0.664 [0.660, 0.669] | 0.714 |
|  | ContrastiveConf | 0.568 [0.562, 0.574] | 0.554 [0.550, 0.558] | 0.569 [0.562, 0.578] | 0.554 [0.550, 0.560] | 0.869 |
|  | kNN (k = 100) | 0.594 [0.590, 0.598] | 0.622 [0.619, 0.625] | 0.595 [0.590, 0.600] | 0.623 [0.619, 0.627] | 0.847 |
|  | DisCoPatch | 0.764 [0.760, 0.767] | 0.751 [0.747, 0.755] | 0.761 [0.757, 0.766] | 0.748 [0.744, 0.753] | 0.567 |
|  | Hashemi et al., decoder queries | 0.428 [0.424, 0.432] | 0.437 [0.434, 0.441] | 0.430 [0.425, 0.435] | 0.438 [0.434, 0.443] | 0.959 |
|  | Activation CDFs (Becker et al.) | 0.821 [0.817, 0.824] | 0.807 [0.804, 0.811] | 0.822 [0.817, 0.826] | 0.808 [0.804, 0.812] | 0.413 |
| YOLO11m | **Two-axis (headline)** | **0.890** [0.887, 0.893] | 0.870 [0.867, 0.872] | **0.891** [0.887, 0.894] | 0.870 [0.867, 0.873] | **0.259** |
|  | Peak share vs similar scenes | 0.887 [0.884, 0.890] | 0.850 [0.847, 0.853] | 0.887 [0.883, 0.891] | 0.850 [0.846, 0.854] | 0.267 |
|  | Level vs similar scenes | 0.871 [0.867, 0.874] | **0.872** [0.869, 0.875] | 0.869 [0.865, 0.874] | **0.871** [0.868, 0.875] | 0.290 |
|  | Level vs the average clean image | 0.835 [0.831, 0.838] | 0.838 [0.835, 0.841] | 0.834 [0.829, 0.838] | 0.837 [0.834, 0.841] | 0.354 |
|  | SAOD, top-3 | 0.693 [0.689, 0.696] | 0.624 [0.621, 0.627] | 0.691 [0.687, 0.695] | 0.623 [0.620, 0.627] | 0.742 |
|  | SAOD, min | 0.715 [0.712, 0.719] | 0.646 [0.643, 0.649] | 0.714 [0.710, 0.719] | 0.646 [0.642, 0.650] | 0.735 |
|  | kNN (k = 100) | 0.747 [0.743, 0.750] | 0.721 [0.718, 0.723] | 0.745 [0.741, 0.750] | 0.720 [0.717, 0.724] | 0.697 |
|  | DisCoPatch | 0.764 [0.760, 0.767] | 0.751 [0.747, 0.755] | 0.761 [0.757, 0.766] | 0.748 [0.744, 0.753] | 0.567 |
|  | Activation CDFs (Becker et al.) | 0.835 [0.831, 0.838] | 0.814 [0.811, 0.817] | 0.836 [0.831, 0.840] | 0.814 [0.811, 0.818] | 0.395 |
| Faster R-CNN R50-FPN v2 | **Two-axis (headline)** | 0.863 [0.859, 0.867] | 0.852 [0.849, 0.854] | 0.863 [0.858, 0.867] | 0.851 [0.848, 0.855] | 0.348 |
|  | Peak share vs similar scenes | **0.888** [0.885, 0.892] | 0.854 [0.851, 0.857] | **0.888** [0.884, 0.893] | 0.854 [0.850, 0.858] | **0.256** |
|  | Level vs similar scenes | 0.862 [0.859, 0.866] | **0.871** [0.868, 0.873] | 0.862 [0.857, 0.867] | **0.870** [0.866, 0.873] | 0.313 |
|  | Level vs the average clean image | 0.823 [0.819, 0.828] | 0.821 [0.819, 0.825] | 0.823 [0.817, 0.828] | 0.821 [0.817, 0.825] | 0.384 |
|  | SAOD, top-3 | 0.674 [0.670, 0.678] | 0.608 [0.604, 0.611] | 0.673 [0.668, 0.679] | 0.608 [0.603, 0.612] | 0.774 |
|  | SAOD, min | 0.694 [0.690, 0.697] | 0.625 [0.622, 0.628] | 0.693 [0.688, 0.697] | 0.624 [0.621, 0.628] | 0.820 |
|  | kNN (k = 100) | 0.804 [0.800, 0.808] | 0.749 [0.746, 0.753] | 0.804 [0.799, 0.809] | 0.749 [0.744, 0.754] | 0.496 |
|  | DisCoPatch | 0.764 [0.760, 0.767] | 0.751 [0.747, 0.755] | 0.761 [0.757, 0.766] | 0.748 [0.744, 0.753] | 0.567 |
|  | Activation CDFs (Becker et al.) | 0.815 [0.811, 0.819] | 0.796 [0.793, 0.799] | 0.815 [0.810, 0.820] | 0.796 [0.792, 0.800] | 0.421 |
| RF-DETR-M | **Two-axis (headline)** | **0.844** [0.841, 0.847] | **0.823** [0.820, 0.826] | **0.844** [0.840, 0.848] | **0.823** [0.819, 0.827] | **0.359** |
|  | Peak share vs similar scenes | 0.783 [0.778, 0.787] | 0.752 [0.747, 0.756] | 0.784 [0.779, 0.790] | 0.752 [0.747, 0.758] | 0.453 |
|  | Level vs similar scenes | 0.808 [0.805, 0.812] | 0.814 [0.811, 0.818] | 0.809 [0.804, 0.814] | 0.815 [0.811, 0.820] | 0.400 |
|  | Level vs the average clean image | 0.763 [0.759, 0.768] | 0.785 [0.782, 0.789] | 0.763 [0.757, 0.769] | 0.785 [0.780, 0.790] | 0.475 |
|  | SAOD, top-3 | 0.647 [0.644, 0.650] | 0.594 [0.591, 0.596] | 0.645 [0.641, 0.649] | 0.592 [0.589, 0.595] | 0.802 |
|  | SAOD, min | 0.668 [0.664, 0.671] | 0.612 [0.609, 0.615] | 0.666 [0.662, 0.671] | 0.612 [0.608, 0.616] | 0.819 |
|  | ContrastiveConf | 0.571 [0.567, 0.575] | 0.547 [0.545, 0.550] | 0.572 [0.567, 0.577] | 0.548 [0.544, 0.552] | 0.886 |
|  | kNN (k = 100) | 0.664 [0.661, 0.667] | 0.627 [0.624, 0.629] | 0.663 [0.660, 0.667] | 0.626 [0.623, 0.630] | 0.780 |
|  | DisCoPatch | 0.764 [0.760, 0.767] | 0.751 [0.747, 0.755] | 0.761 [0.757, 0.766] | 0.748 [0.744, 0.753] | 0.567 |
|  | Hashemi et al., decoder queries | 0.442 [0.438, 0.445] | 0.482 [0.479, 0.486] | 0.445 [0.440, 0.450] | 0.485 [0.481, 0.489] | 0.945 |
|  | Activation CDFs (Becker et al.) | 0.800 [0.796, 0.804] | 0.788 [0.785, 0.792] | 0.801 [0.796, 0.806] | 0.789 [0.785, 0.793] | 0.437 |

Bold is each detector's best row in a column. The pilot's two control rows (channel means by kNN, and against their own average) and the two sensitivity rows (the CDFs' plain sum, Hashemi on the encoder) are left out here; each detector's `report.md` has them.

**The headline rule** asks for six intervals above 0 per detector: the two-axis score minus the CDFs on both family groups, and minus the level score on the common families, each on all images and on the untouched ones.

| Detector | Image set | Two-axis − CDFs, common | Two-axis − CDFs, extra | Two-axis − level, common |
|---|---|---|---|---|
| RT-DETRv2-R18 | all | +0.096 [+0.093, +0.100] | +0.051 [+0.048, +0.054] | +0.076 [+0.073, +0.079] |
|  | untouched | +0.095 [+0.090, +0.099] | +0.050 [+0.046, +0.054] | +0.076 [+0.073, +0.080] |
| YOLO11m | all | +0.055 [+0.053, +0.058] | +0.056 [+0.054, +0.058] | +0.019 [+0.017, +0.022] |
|  | untouched | +0.055 [+0.051, +0.058] | +0.056 [+0.053, +0.059] | +0.021 [+0.018, +0.024] |
| Faster R-CNN R50-FPN v2 | all | +0.048 [+0.045, +0.052] | +0.056 [+0.053, +0.059] | **+0.001 [−0.002, +0.003]** |
|  | untouched | +0.048 [+0.043, +0.053] | +0.056 [+0.052, +0.059] | **+0.001 [−0.003, +0.004]** |
| RF-DETR-M | all | +0.044 [+0.042, +0.048] | +0.035 [+0.032, +0.038] | +0.036 [+0.034, +0.038] |
|  | untouched | +0.043 [+0.039, +0.047] | +0.034 [+0.031, +0.037] | +0.035 [+0.032, +0.038] |

**The level rule,** on the 4,800 held-out images (positions 200 and later): the level against the 50 most similar clean scenes minus each comparison.

| Detector | − average clean image, common | − average clean image, extra | − CDFs, common | − CDFs, extra |
|---|---|---|---|---|
| RT-DETRv2-R18 | +0.041 [+0.039, +0.044] | +0.037 [+0.035, +0.039] | +0.020 [+0.017, +0.024] | +0.059 [+0.056, +0.062] |
| YOLO11m | +0.036 [+0.033, +0.038] | +0.034 [+0.032, +0.036] | +0.035 [+0.032, +0.039] | +0.058 [+0.055, +0.061] |
| Faster R-CNN R50-FPN v2 | +0.039 [+0.036, +0.042] | +0.049 [+0.047, +0.051] | +0.048 [+0.044, +0.051] | +0.075 [+0.072, +0.077] |
| RF-DETR-M | +0.046 [+0.043, +0.048] | +0.029 [+0.027, +0.032] | +0.009 [+0.006, +0.012] | +0.026 [+0.023, +0.029] |

**What else the tables show:**
- **The untouched images give the same numbers.** On every detector, the two-axis score on the untouched images is within 0.001 of its value on all images.
- **On the extra families the two-axis score is below the level score on three detectors:** RT-DETR −0.008 [−0.010, −0.005], YOLO11m −0.002 [−0.004, −0.001] and Faster R-CNN −0.019 [−0.020, −0.017]. On RF-DETR-M it is above: +0.009 [+0.007, +0.011]. The rule compares with the level score on the common families only.
- **Hashemi et al. is below chance on RF-DETR-M too** (0.442 / 0.482), as on RT-DETR (0.428 / 0.437).

## Faster R-CNN: which clause failed, and why

**The clause.** The headline is confirmed when, on all images and on the untouched ones, the two-axis score beats the activation CDFs on both family groups and beats the level score on the common families, every interval excluding 0. On Faster R-CNN (`summary.json`, `intervals`):
- **Two-axis − CDFs passes.** +0.048 [+0.045, +0.052] common and +0.056 [+0.053, +0.059] extra on all images; +0.048 [+0.043, +0.053] and +0.056 [+0.052, +0.059] on the untouched ones.
- **Two-axis − level, common, fails.** +0.0006 [−0.0018, +0.0031] on all images and +0.0009 [−0.0026, +0.0040] on the untouched ones. Both intervals include 0.

So the two scores are tied on AUROC. On the other two metrics they split: the two-axis score is ahead on AUPR (common: +0.009 [+0.007, +0.012]) and behind on FPR95 (0.348 against 0.313; +0.035 [+0.029, +0.041]).

**Why, from `arms.csv`.** The two-axis score is the larger of its two arms. Mean AUROC of each arm on its own, over the 75 common and the 20 extra conditions (all images):

| Detector | Flattening arm, common / extra | Level arm, common / extra | Two-axis, common / extra |
|---|---|---|---|
| RT-DETRv2-R18 | 0.698 / 0.570 | 0.841 / 0.866 | 0.917 / 0.858 |
| YOLO11m | 0.862 / 0.735 | 0.871 / 0.872 | 0.890 / 0.870 |
| Faster R-CNN R50-FPN v2 | **0.550 / 0.495** | 0.862 / 0.871 | 0.863 / 0.852 |
| RF-DETR-M | 0.820 / 0.779 | 0.808 / 0.814 | 0.844 / 0.823 |

The level arm's AUROC equals the report's level row in every condition. On RT-DETR the flattening arm is weak on average too (0.698), but it is strong exactly where the level misses. On Faster R-CNN it is not:
- **Where the level is weak, the flattening arm is weak too.** At severity 1, fog gives 0.75 for the flattening arm and 0.69 for the level arm; on RT-DETR it gives 0.97 and 0.51. Averaged over the five severities, Faster R-CNN's level arm reaches 0.732 on fog and 0.877 on contrast, against 0.567 and 0.686 on RT-DETR, so it leaves less to add. On zoom blur both levels are weak (0.703 and 0.707), but Faster R-CNN's flattening arm reaches only 0.59–0.71 across the severities, against 0.93–0.97 on RT-DETR.
- **On the blurs, the flattening arm falls below chance as the blur grows.** It gives 0.60 / 0.47 / 0.30 on defocus blur at severities 1 / 3 / 5, 0.59 / 0.43 / 0.19 on Gaussian blur, and 0.73 / 0.69 / 0.43 on contrast. Below 0.5 the arm points the wrong way: relative to their neighbours, the corrupted images' channels come out *peakier* than the clean images' do, not flatter. On RT-DETR the same arm rises with the blur (defocus 0.96 / 0.99 / 0.99).
- **So the larger arm gains on some families and loses on others.** Two-axis minus level, averaged over the five severities:
  - gains: fog +0.090, zoom blur +0.064, contrast +0.032, elastic transform +0.013, JPEG +0.011;
  - losses: brightness −0.080, frost −0.036, glass blur −0.033, motion blur −0.024, defocus blur −0.021, snow −0.007; the other four families are within ±0.001;
  - over the 75 common conditions this nets +0.001. On RT-DETR the same comparison gains +0.405 on fog, +0.293 on contrast and +0.223 on zoom blur, and nets +0.076.
- **The peak share does move; its signed average over channels does not.** The unsigned peak-share row (the mean |change| over channels) is Faster R-CNN's best row on the common families: 0.888 [0.885, 0.892], ahead of the two-axis score by +0.025 [+0.022, +0.029]. The signed flattening arm, which averages the same changes with their sign, reaches 0.550. So under these corruptions Faster R-CNN's peak shares change, but not in one direction across channels.

Choosing the peak-share row for Faster R-CNN now would be a choice made after reading the test images. The pre-registered headline stays the two-axis score, and its outcome on Faster R-CNN stays partial. The data do not show why Faster R-CNN's blurred channels get peakier, or which channels flatten and which sharpen; that was not measured.

## AUROC by severity

All images. Common families:

| Detector | Row | Sev 1 | Sev 2 | Sev 3 | Sev 4 | Sev 5 |
|---|---|---|---|---|---|---|
| RT-DETRv2-R18 | Two-axis | **0.863** | 0.905 | 0.926 | 0.940 | 0.951 |
|  | Level vs similar scenes | 0.761 | 0.815 | 0.850 | 0.878 | 0.899 |
|  | Activation CDFs | 0.707 | 0.781 | 0.836 | 0.877 | 0.901 |
| YOLO11m | Two-axis | **0.825** | 0.874 | 0.902 | 0.918 | 0.931 |
|  | Level vs similar scenes | 0.799 | 0.849 | 0.883 | 0.903 | 0.920 |
|  | Activation CDFs | 0.729 | 0.801 | 0.853 | 0.886 | 0.906 |
| Faster R-CNN R50-FPN v2 | Two-axis | **0.794** | 0.838 | 0.872 | 0.895 | 0.916 |
|  | Level vs similar scenes | 0.788 | 0.840 | 0.873 | 0.897 | 0.913 |
|  | Activation CDFs | 0.701 | 0.777 | 0.830 | 0.870 | 0.895 |
| RF-DETR-M | Two-axis | **0.786** | 0.835 | 0.854 | 0.869 | 0.878 |
|  | Level vs similar scenes | 0.741 | 0.797 | 0.816 | 0.837 | 0.851 |
|  | Activation CDFs | 0.703 | 0.778 | 0.811 | 0.841 | 0.865 |

Extra families:

| Detector | Row | Sev 1 | Sev 2 | Sev 3 | Sev 4 | Sev 5 |
|---|---|---|---|---|---|---|
| RT-DETRv2-R18 | Two-axis | 0.705 | 0.849 | 0.859 | 0.924 | 0.955 |
|  | Level vs similar scenes | 0.683 | 0.858 | 0.881 | 0.943 | 0.965 |
|  | Activation CDFs | 0.610 | 0.762 | 0.834 | 0.896 | 0.935 |
| YOLO11m | Two-axis | 0.704 | 0.875 | 0.866 | 0.938 | 0.966 |
|  | Level vs similar scenes | 0.695 | 0.864 | 0.877 | 0.950 | 0.974 |
|  | Activation CDFs | 0.602 | 0.759 | 0.836 | 0.915 | 0.957 |
| Faster R-CNN R50-FPN v2 | Two-axis | 0.670 | 0.817 | 0.867 | 0.938 | 0.967 |
|  | Level vs similar scenes | 0.690 | 0.838 | 0.891 | 0.957 | 0.977 |
|  | Activation CDFs | 0.583 | 0.720 | 0.825 | 0.900 | 0.950 |
| RF-DETR-M | Two-axis | 0.680 | 0.796 | 0.841 | 0.879 | 0.920 |
|  | Level vs similar scenes | 0.647 | 0.755 | 0.829 | 0.899 | 0.941 |
|  | Activation CDFs | 0.603 | 0.701 | 0.807 | 0.890 | 0.941 |

- **Mild corruption.** At severity 1 of the common families the two-axis score leads the CDFs on every detector: 0.863 / 0.825 / 0.794 / 0.786 against 0.707 / 0.729 / 0.701 / 0.703.
- **The lead narrows as severity grows.** At severity 5 it is 0.951 / 0.931 / 0.916 / 0.878 against 0.901 / 0.906 / 0.895 / 0.865.
- **On Faster R-CNN the two-axis and the level score are within 0.006 of each other** at every severity of the common families.
- **On the extra families the level score is ahead at severities 4 and 5 on every detector,** and at every severity on Faster R-CNN. On RF-DETR-M the CDFs are ahead of the two-axis score at severities 4 and 5 there (0.890 / 0.941 against 0.879 / 0.920).

## Families at severities 1, 3 and 5

The two-axis score's AUROC at severities 1 / 3 / 5, all images. `*` marks the extra families.

| Family | RT-DETRv2-R18 | YOLO11m | Faster R-CNN R50-FPN v2 | RF-DETR-M |
|---|---|---|---|---|
| gaussian noise | 0.93 / 1.00 / 1.00 | 0.97 / 1.00 / 1.00 | 0.99 / 1.00 / 1.00 | 0.94 / 1.00 / 1.00 |
| shot noise | 0.91 / 1.00 / 1.00 | 0.97 / 1.00 / 1.00 | 0.98 / 1.00 / 1.00 | 0.93 / 0.99 / 1.00 |
| impulse noise | 0.99 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 0.98 / 1.00 / 1.00 |
| defocus blur | 0.95 / 0.98 / 0.99 | 0.92 / 0.99 / 1.00 | 0.84 / 0.93 / 0.97 | 0.92 / 0.98 / 0.99 |
| glass blur | 0.91 / 0.96 / 0.98 | 0.84 / 0.96 / 0.99 | 0.75 / 0.86 / 0.93 | 0.81 / 0.90 / 0.97 |
| motion blur | 0.87 / 0.95 / 0.97 | 0.84 / 0.96 / 0.99 | 0.80 / 0.93 / 0.97 | 0.76 / 0.92 / 0.96 |
| zoom blur | 0.90 / 0.93 / 0.95 | 0.85 / 0.89 / 0.91 | 0.74 / 0.77 / 0.80 | 0.90 / 0.93 / 0.94 |
| snow | 0.92 / 0.97 / 0.98 | 0.90 / 0.95 / 0.96 | 0.92 / 0.97 / 0.98 | 0.82 / 0.87 / 0.89 |
| frost | 0.72 / 0.89 / 0.92 | 0.58 / 0.76 / 0.80 | 0.61 / 0.78 / 0.82 | 0.70 / 0.90 / 0.92 |
| fog | 0.95 / 0.98 / 0.99 | 0.84 / 0.91 / 0.91 | 0.80 / 0.83 / 0.84 | 0.89 / 0.95 / 0.96 |
| brightness | 0.53 / 0.60 / 0.71 | 0.50 / 0.59 / 0.72 | 0.54 / 0.63 / 0.78 | 0.52 / 0.59 / 0.67 |
| contrast | 0.94 / 0.99 / 1.00 | 0.86 / 0.96 / 1.00 | 0.82 / 0.92 / 0.98 | 0.91 / 0.99 / 1.00 |
| elastic transform | 0.68 / 0.76 / 0.84 | 0.54 / 0.61 / 0.70 | 0.60 / 0.67 / 0.74 | 0.54 / 0.52 / 0.50 |
| pixelate | 0.85 / 0.96 / 0.99 | 0.83 / 0.97 / 1.00 | 0.69 / 0.86 / 0.97 | 0.61 / 0.67 / 0.74 |
| jpeg compression | 0.89 / 0.94 / 0.97 | 0.94 / 0.99 / 1.00 | 0.84 / 0.93 / 0.99 | 0.57 / 0.60 / 0.61 |
| speckle noise * | 0.80 / 0.98 / 1.00 | 0.92 / 0.99 / 1.00 | 0.91 / 0.99 / 1.00 | 0.87 / 0.97 / 0.99 |
| gaussian blur * | 0.85 / 0.98 / 0.99 | 0.78 / 0.99 / 1.00 | 0.71 / 0.94 / 0.98 | 0.79 / 0.98 / 1.00 |
| spatter * | 0.57 / 0.94 / 1.00 | 0.53 / 0.94 / 1.00 | 0.55 / 0.95 / 1.00 | 0.50 / 0.89 / 0.93 |
| saturate * | 0.60 / 0.54 / 0.84 | 0.59 / 0.54 / 0.87 | 0.50 / 0.58 / 0.89 | 0.56 / 0.52 / 0.76 |

Each detector's `report.md` has the same table with the peak share, the level, the activation CDFs and DisCoPatch.

**Where a baseline leads by more than 0.03** (each `summary.json`'s `by_family`):
- **At severity 1 the activation CDFs never lead,** on any family or detector. DisCoPatch and kNN do, on frost, elastic transform, spatter and saturate, and on RF-DETR-M's JPEG. The leading baseline reaches only 0.60–0.66 there.
- **Brightness and saturate stay weak on every detector,** as on RT-DETR: 0.50–0.54 and 0.50–0.60 at severity 1.
- **YOLO11m and Faster R-CNN, frost and elastic transform.** DisCoPatch leads on frost at every severity (0.66 / 0.83 / 0.87 at severities 1 / 3 / 5). On elastic transform kNN reaches 0.88–0.89 at severity 5, against 0.70–0.74.
- **Faster R-CNN, zoom blur:** 0.74 / 0.77 / 0.80, against kNN's 0.82 at severity 3 and SAOD min's 0.87 at severity 5.
- **RF-DETR-M, elastic transform, JPEG and pixelate.**
  - The two-axis score stays near chance on elastic transform (0.54 / 0.52 / 0.50) and JPEG (0.57 / 0.60 / 0.61), and reaches only 0.74 on pixelate at severity 5.
  - DisCoPatch, SAOD min (0.74 and 0.81 at severity 5) and kNN (0.92 on pixelate at severity 5) lead there.
  - The activation CDFs, which read the method's blocks plus the embeddings, are near chance on elastic transform and JPEG too: 0.49 / 0.49 / 0.51 and 0.52 / 0.52 / 0.57.

## Which arm catches fog and which catches noise

From `arms.csv`, all images. AUROC at severities 1 / 3 / 5:

| Detector | Corruption | Flattening arm | Level arm | Two-axis |
|---|---|---|---|---|
| RT-DETRv2-R18 | fog | **0.97** / 0.99 / 1.00 | 0.51 / 0.58 / 0.62 | 0.95 / 0.98 / 0.99 |
|  | contrast | **0.96** / 0.99 / 1.00 | 0.53 / 0.66 / 0.89 | 0.94 / 0.99 / 1.00 |
|  | gaussian noise | 0.20 / 0.03 / 0.00 | **0.96** / 1.00 / 1.00 | 0.93 / 1.00 / 1.00 |
| YOLO11m | fog | **0.89** / 0.94 / 0.95 | 0.74 / 0.82 / 0.87 | 0.84 / 0.91 / 0.91 |
|  | contrast | **0.90** / 0.98 / 1.00 | 0.76 / 0.90 / 0.97 | 0.86 / 0.96 / 1.00 |
|  | gaussian noise | 0.76 / 0.95 / 1.00 | **0.99** / 1.00 / 1.00 | 0.97 / 1.00 / 1.00 |
| Faster R-CNN R50-FPN v2 | fog | **0.75** / 0.74 / 0.77 | 0.69 / 0.75 / 0.75 | 0.80 / 0.83 / 0.84 |
|  | contrast | 0.73 / 0.69 / 0.43 | **0.75** / 0.89 / 0.98 | 0.82 / 0.92 / 0.98 |
|  | gaussian noise | 0.46 / 0.08 / 0.01 | **0.99** / 1.00 / 1.00 | 0.99 / 1.00 / 1.00 |
| RF-DETR-M | fog | 0.67 / 0.76 / 0.85 | **0.90** / 0.96 / 0.96 | 0.89 / 0.95 / 0.96 |
|  | contrast | 0.68 / 0.84 / 1.00 | **0.91** / 0.99 / 1.00 | 0.91 / 0.99 / 1.00 |
|  | gaussian noise | **0.96** / 0.99 / 0.99 | 0.89 / 1.00 / 1.00 | 0.94 / 1.00 / 1.00 |

Bold marks the stronger arm at severity 1.

- **RT-DETRv2-R18: fog through the flattening arm, noise through the level arm.** Under noise the flattening arm points the wrong way (0.20 at severity 1). This reproduces C5 of `docs/paper-storyline.md`.
- **YOLO11m: the same split, less sharp.** The level arm also sees fog (0.74 at severity 1). Unlike RT-DETR's, YOLO's flattening arm also rises under noise, to 0.95 at severity 3.
- **Faster R-CNN: noise through the level arm.** The flattening arm is near chance under noise at severity 1 (0.46) and points the wrong way from severity 2 on (0.26, then 0.08). Fog moves both arms only moderately (0.75 and 0.69), and the two-axis score (0.80) is above either arm alone.
- **RF-DETR-M: the roles swap.** Fog goes through the level arm (0.90 against 0.67), and noise at severity 1 through the flattening arm (0.96 against 0.89). From severity 3 both arms are at 0.99–1.00 under noise.
  - The swap showed before the run, on the 300 development images. On the raw block outputs at severity 1, fog gave 0.676 for the flattening arm and 0.916 for the level arm, and Gaussian noise 0.981 and 0.902 (`tap-exploration/vit_taps.log`).
- **Contrast splits like fog** on RT-DETR, YOLO11m and RF-DETR-M. On Faster R-CNN both arms are near 0.75 at severity 1, and the flattening arm falls to 0.43 at severity 5.
- **The other three noises split like Gaussian noise** at severity 1 on each detector: shot, impulse and speckle noise. On RF-DETR-M, impulse noise is within 0.01 between the arms (0.98 and 0.97).

**What carries over is not which arm a corruption moves, but that it moves one.** At severity 1 the two-axis score reaches 0.95 / 0.84 / 0.80 / 0.89 on fog and 0.93 / 0.97 / 0.99 / 0.94 on Gaussian noise (RT-DETRv2-R18, YOLO11m, Faster R-CNN, RF-DETR-M). Why the ViT's roles swap was not tested.

## Clean mAP

| Detector | Check stage, COCO val AP | Floor | Published | The report's clean mAP | mAP at severities 1 / 3 / 5, common |
|---|---|---|---|---|---|
| RT-DETRv2-R18 | 0.4791 | – | 48.1 (the checkpoint's name) | 0.479 | 0.387 / 0.270 / 0.151 |
| YOLO11m | 0.5046 | 0.485 | 51.5 | 0.505 | 0.399 / 0.285 / 0.188 |
| Faster R-CNN R50-FPN v2 | 0.4694 | 0.437 | 46.7 | 0.469 | 0.361 / 0.244 / 0.141 |
| RF-DETR-M | 0.5452 | 0.517 | 54.7 | 0.545 | 0.483 / 0.386 / 0.243 |

- **The check stage** runs each detector on the 5,000 val image files (`check` in each `manifest.json`). The floor is the published COCO val AP minus 0.03, and every detector is above it.
- **The report's clean mAP** comes from the pass's stored detections of the clean condition (`clean_map` in each `summary.json`). It agrees with the check to within 0.0002.
- **YOLO11m is 0.010 below its published AP.** The plan expected a shortfall. Our NMS keeps one label per box and the top 100 detections, while Ultralytics' validation uses multi-label NMS and 300 detections. The shortfall was not measured separately.
- **The mAP by severity** is in each report's "Detector mAP" table. It is context only: the goal is to tell clean images from corrupted ones, not to predict the drop in mAP.

## Checks

- **`arms.csv` reproduces the reports.**
  - Each run's `two_axis` rows equal the two-axis AUROCs on all images in its report's `separation.csv` exactly, on all four runs.
  - Each run's `level` rows equal its report's level row exactly.
  - RT-DETR's rows (labelled `coco`) equal `docs/results/paper-evidence/arms_by_family.csv` to within 5e-5, that table's rounding.
- **The corrupted images are RT-DETR's.** The shared pass compares every image's 96 digests with `runs/coco/`'s before it scores the image, and it covered all 5,000 images.
- **One stop in the full pass, and no result affected.**
  - The first full-pass run stopped early, with 56–57 images written per detector. It hit its own 8 GiB GPU cap through allocator fragmentation: 5.51 GiB was allocated and 2.07 GiB reserved but unallocated when a 526 MiB request failed.
  - The pass resumed with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, the same cap and the same batch sizes, and peaked at 6.02 GiB.
  - Result files are written atomically, a resumed pass redoes only the missing ones, and the outputs do not depend on the batch.
- **Precision.** Each detector's protocol records float32 matmuls at `"highest"`. Importing rfdetr switches a process to TF32, so the RF-DETR adapter restores the setting.
- **Memory.** The three reports took 57 minutes on the CPU, with a peak of 49.7 GiB. The arms script took 3.5 minutes, with a peak of 53.8 GiB. Faster R-CNN's method scores, with 3,840 channels, are the largest: 14 GiB on disk, against 3.5–5.6 GiB for the other three.

## Not yet shown

- **Why Faster R-CNN's flattening arm is weak.** Its peak shares move under corruption, but not in one direction across channels, and under strong blur its flattening arm points the wrong way. Which channels flatten and which sharpen was not measured.
- **Why the ViT's arms swap roles.** Fog moves the levels of RF-DETR-M's blocks 1–3, and noise flattens them. No pure-operation test was run on the ViT.
- **Why the margin over the CDFs is smaller off RT-DETR.** The method's choices were made on RT-DETR. These data do not say whether that is the reason.
- **The method's runtime** on the three new detectors.
- **Real fog and driving data.** All four detectors are tested on synthetic corruptions of COCO images, with a COCO clean bank.

## Files

Everything in `docs/results/coco-detectors/`:
- `summary.md` and `summary.csv`: the table of all four detectors.
- `yolo11m/`, `faster_rcnn_r50_fpn_v2/` and `rfdetr_m/`: each detector's report.
  - `report.md`: the generated report.
  - `summary.json`: both decisions and every number above.
  - `separation.csv`, `aggregates.csv` and `intervals.csv`: per condition, per group and the bootstrap intervals.
  - `conditions.csv`: each condition's mAP and mean scores.
  - `knn_k.csv`: the kNN baseline by k.
- `arms.csv`: per run and condition, the AUROC of the two-axis score and of its flattening and level arms. It is written by `scripts/paper/detector_arms.py`; the rows labelled `coco` are RT-DETR's.
- `tap-exploration/`: the two tap explorations.

RT-DETRv2-R18's report is in `docs/results/coco/`. The per-image scores stay in `runs/coco-detectors/` in the main checkout, which is not in git.
