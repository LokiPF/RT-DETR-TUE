# COCO baseline numbers

These are the numbers for the four baselines we compare against on COCO: SAOD, ContrastiveConf, kNN
and DisCoPatch. Our own method is not in this document. The plan is
`docs/superpowers/plans/2026-09-27-coco-baseline-numbers.md`, and the baseline choices are in
`docs/driving-benchmark-baselines-and-metrics.md`. Every table here comes from the files in
`docs/results/coco-baselines/`.

## The short version

- **DisCoPatch is best at telling a degraded image from a clean one.** Its mean AUROC is 0.764 on the
  15 common corruption families and 0.751 on the 4 extra ones, and it has the lowest FPR95 (0.567). It
  is almost perfect on the three common noise families (AUROC 0.97–1.00 at severities 3 and 5) and the
  best on snow, frost and fog.
- **But DisCoPatch does not tell us when the detector is hurt.** Within one corruption condition,
  images whose DisCoPatch score rises more lose slightly *less* detection quality (ρ = −0.051). Its
  risk–coverage curve is the worst of the five (AURC 0.665, against 0.698 for a random order). It is
  also weak on JPEG compression (AUROC 0.59 at severity 5) and pixelation (0.75), which destroy the
  detector (mAP 0.108 and 0.037 at severity 5).
- **SAOD's min score is the strongest detector-based baseline.** Mean AUROC is 0.731 (common) and 0.664
  (extra). Across the 95 conditions, its mean score follows the detector's mAP almost perfectly
  (ρ = −0.98).
- **No baseline predicts per-image harm well.** The best within-condition correlation between the
  change in score and the change in detection error is 0.257 (SAOD min), then 0.230 (ContrastiveConf).
  ContrastiveConf has the best risk–coverage curve (AURC 0.540; the best possible is 0.449).
- **Every monitor is cheap.** The detector takes 5.9 ms per image, and each score adds 0.3–1.3 ms on
  top. DisCoPatch runs on its own in 2.4 ms.

So "is this image degraded?" and "will the detector fail on this image?" are different questions, and
the existing baselines each answer only part of one of them.

## Setup

| Item | Value |
| --- | --- |
| Detector | RT-DETRv2-R18, the COCO checkpoint `rtdetrv2_r18vd_120e_coco_rerun_48.1.pth`, frozen |
| Clean COCO val mAP (our pipeline) | 0.479 (checkpoint name: 48.1) |
| Test images | all 5,000 COCO val2017 images, shuffled with seed 44, in 5 folds of 1,000 |
| Corruptions | all 19 `imagecorruptions` families × severities 1–5 = 95 conditions, plus clean; one seeded draw per image and condition |
| Family groups | 15 common families (COCO-C) and 4 extra ones (speckle noise, Gaussian blur, spatter, saturate), reported separately |
| Training data for kNN and DisCoPatch | all 118,287 COCO train2017 images, clean |

The four baselines, all scored so that higher means "more likely degraded":

| Baseline | Score | Fixed choices |
| --- | --- | --- |
| SAOD (Oksuz et al., CVPR 2023) | 1 − confidence over the top 100 detections: the mean of the top 3 values ("top-3"), or 1 − the highest confidence ("min") | none |
| ContrastiveConf (Park et al., TPAMI 2026, `uq-detr`) | −(Conf⁺ − λ·Conf⁻) | θ = 0.3; λ cross-fitted: for each fold, fitted on the other 4 folds' clean images against per-image AP. λ per fold = 7, 8, 7, 7, 7 |
| kNN (Sun et al., ICML 2022) | distance to the 100th nearest train image, on L2-normalised 512-d backbone features (ResNet-18 layer 4, global average pooled) | k = 100, fixed in advance (see the k check below) |
| DisCoPatch (ICCV 2025, official code) | 1 − mean discriminator output over 64 random 64-px patches of a 256² resize, normalised per image | README hyperparameters, 65 COCO epochs, one training run |

Detection harm uses per-image LRP. The confidence threshold is 0.55, chosen once for the optimal
average LRP on clean images, and detections that are at least 50% inside a crowd box are ignored. In
32 clean images and 14–38 images per corrupted condition, LRP is undefined: there is no object and no
kept detection. Those images drop out of the harm numbers, and `conditions.csv` gives the count for
each condition.

Intervals are 95% paired bootstrap intervals over images. There are 1,000 draws, and each redraw keeps
the image across all 96 conditions and refits λ.

## Separation: clean versus degraded

For each condition, the 5,000 clean images are the negatives and the same 5,000 images, corrupted, are
the positives. AUROC and AUPR: higher is better, and 0.5 is chance. FPR95 is the share of clean images
flagged when 95% of the degraded ones are caught: lower is better. "All" is the mean over the
families' conditions. ContrastiveConf numbers are averaged over the 5 folds, because its λ differs
per fold.

### 15 common families

| Method | AUROC sev 1 | sev 2 | sev 3 | sev 4 | sev 5 | AUROC all | AUPR all | FPR95 all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SAOD, top-3 | 0.580 | 0.625 | 0.683 | 0.745 | 0.798 | 0.686 [0.682, 0.690] | 0.644 [0.638, 0.649] | 0.729 [0.722, 0.736] |
| SAOD, min | 0.608 | 0.665 | 0.733 | 0.799 | 0.848 | 0.731 [0.727, 0.734] | 0.717 [0.711, 0.723] | 0.714 [0.707, 0.722] |
| ContrastiveConf | 0.527 | 0.545 | 0.569 | 0.595 | 0.605 | 0.568 [0.562, 0.574] | 0.532 [0.527, 0.539] | 0.869 [0.864, 0.875] |
| kNN (k = 100) | 0.546 | 0.559 | 0.590 | 0.625 | 0.650 | 0.594 [0.590, 0.598] | 0.561 [0.557, 0.565] | 0.847 [0.842, 0.852] |
| **DisCoPatch** | **0.678** | **0.733** | **0.777** | **0.805** | 0.825 | **0.764** [0.760, 0.767] | **0.729** [0.724, 0.734] | **0.567** [0.558, 0.576] |

### 4 extra families

| Method | AUROC sev 1 | sev 2 | sev 3 | sev 4 | sev 5 | AUROC all | AUPR all | FPR95 all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SAOD, top-3 | 0.531 | 0.577 | 0.618 | 0.667 | 0.731 | 0.625 [0.622, 0.628] | 0.590 [0.586, 0.595] | 0.820 [0.814, 0.825] |
| SAOD, min | 0.543 | 0.607 | 0.660 | 0.720 | 0.787 | 0.664 [0.660, 0.667] | 0.652 [0.646, 0.657] | 0.811 [0.804, 0.817] |
| ContrastiveConf | 0.507 | 0.531 | 0.548 | 0.582 | 0.600 | 0.554 [0.550, 0.558] | 0.530 [0.527, 0.535] | 0.895 [0.890, 0.899] |
| kNN (k = 100) | 0.518 | 0.587 | 0.605 | 0.673 | 0.725 | 0.622 [0.619, 0.625] | 0.590 [0.587, 0.594] | 0.802 [0.796, 0.807] |
| **DisCoPatch** | **0.633** | **0.755** | **0.756** | **0.787** | **0.824** | **0.751** [0.747, 0.755] | **0.717** [0.712, 0.723] | **0.580** [0.572, 0.588] |

At severity 5 on the common families, SAOD min (0.848) and DisCoPatch (0.825) are close. DisCoPatch's
lead comes from the mild severities.

### Per family: AUROC at severity 3 / severity 5

Bold is the best method at severity 5. `*` marks the extra families.

| Family | SAOD top-3 | SAOD min | ContrastiveConf | kNN | DisCoPatch |
| --- | ---: | ---: | ---: | ---: | ---: |
| gaussian noise | 0.69 / 0.90 | 0.75 / 0.95 | 0.62 / 0.70 | 0.48 / 0.58 | **0.99 / 1.00** |
| shot noise | 0.67 / 0.86 | 0.74 / 0.93 | 0.60 / 0.69 | 0.49 / 0.62 | **0.97 / 0.99** |
| impulse noise | 0.70 / 0.89 | 0.76 / 0.95 | 0.61 / 0.71 | 0.55 / 0.58 | **0.99 / 1.00** |
| defocus blur | 0.73 / 0.85 | **0.80 / 0.92** | 0.58 / 0.64 | 0.65 / 0.73 | 0.80 / 0.85 |
| glass blur | 0.84 / 0.89 | **0.90 / 0.95** | 0.56 / 0.58 | 0.58 / 0.63 | 0.88 / 0.94 |
| motion blur | 0.71 / 0.85 | **0.78 / 0.91** | 0.59 / 0.68 | 0.66 / 0.73 | 0.74 / 0.83 |
| zoom blur | 0.82 / 0.87 | **0.88 / 0.92** | 0.62 / 0.63 | 0.59 / 0.57 | 0.82 / 0.85 |
| snow | 0.67 / 0.73 | 0.73 / 0.80 | 0.63 / 0.62 | 0.73 / 0.66 | **0.84 / 0.84** |
| frost | 0.66 / 0.69 | 0.71 / 0.75 | 0.55 / 0.57 | 0.55 / 0.56 | **0.83 / 0.87** |
| fog | 0.54 / 0.57 | 0.55 / 0.58 | 0.49 / 0.50 | 0.45 / 0.52 | **0.72 / 0.84** |
| brightness | 0.52 / 0.56 | **0.53 / 0.58** | 0.52 / 0.53 | 0.50 / 0.55 | 0.50 / 0.52 |
| contrast | 0.57 / 0.78 | **0.58 / 0.85** | 0.50 / 0.59 | 0.46 / 0.41 | 0.61 / 0.78 |
| elastic transform | 0.64 / 0.72 | 0.68 / 0.74 | 0.50 / 0.48 | **0.73 / 0.85** | 0.68 / 0.72 |
| pixelate | 0.81 / 0.95 | **0.85 / 0.97** | 0.56 / 0.49 | 0.69 / 0.83 | 0.64 / 0.75 |
| jpeg compression | 0.67 / 0.86 | 0.74 / 0.92 | 0.60 / 0.66 | **0.76 / 0.93** | 0.65 / 0.59 |
| speckle noise * | 0.64 / 0.74 | 0.69 / 0.81 | 0.57 / 0.61 | 0.49 / 0.56 | **0.94 / 0.97** |
| gaussian blur * | 0.71 / 0.88 | **0.78 / 0.94** | 0.56 / 0.65 | 0.64 / 0.77 | 0.80 / 0.87 |
| spatter * | 0.62 / 0.74 | 0.66 / 0.81 | 0.55 / 0.60 | **0.77 / 0.93** | 0.81 / 0.89 |
| saturate * | 0.51 / 0.57 | 0.51 / 0.59 | 0.51 / 0.53 | **0.52 / 0.64** | 0.47 / 0.56 |

What stands out:

- DisCoPatch wins on noise and weather.
- SAOD min wins on blur, contrast and pixelation, because it sees what the detector sees.
- kNN wins on JPEG, elastic transform and spatter.
- Brightness, fog and saturate are hard for everyone. They are also mild for the detector: at
  severity 5, mAP only falls from 0.479 to 0.39–0.42 (`conditions.csv`).

## Harm: does the score follow detection quality?

This is the part that matters for a monitor.

| Method | ρ(score, mAP), 96 conditions | ρ(score, LRP), 96 conditions | **ρ(Δscore, ΔLRP), within condition** | AURC, all (oracle 0.449) |
| --- | ---: | ---: | ---: | ---: |
| SAOD, top-3 | −0.986 | 0.988 [0.986, 0.989] | 0.175 [0.162, 0.187] | 0.638 [0.631, 0.645] |
| SAOD, min | −0.981 | 0.983 [0.981, 0.985] | **0.257** [0.246, 0.268] | 0.560 [0.552, 0.568] |
| ContrastiveConf | −0.729 | 0.744 [0.688, 0.787] | 0.230 [0.216, 0.243] | **0.540** [0.532, 0.549] |
| kNN (k = 100) | −0.534 | 0.542 [0.522, 0.554] | 0.061 [0.049, 0.072] | 0.638 [0.631, 0.645] |
| DisCoPatch | −0.636 | 0.640 [0.628, 0.653] | −0.051 [−0.066, −0.034] | 0.665 [0.658, 0.672] |

How to read the columns:

- **Between conditions (the first two columns).** Each condition's mean score is compared with that
  condition's mAP, or its mean LRP. This says whether a method ranks corruption *types and strengths*
  like the detector does. mAP is one number per condition, so that column has no interval.
- **Within condition (the primary per-image measure).** For each image, take how much its score
  changed from clean (Δscore) and how much its LRP changed (ΔLRP). Compute Spearman ρ over the images
  of one condition, then average over the 95 conditions. This asks: *among images with the same
  corruption, does the score rise more where the detector suffers more?*
- **AURC.** Reject the highest-scoring images first, and record the mean LRP of the images kept, at 20
  coverage levels. Lower is better. The oracle rejects by the true LRP (0.449). A random order gives
  about 0.698.

Within-condition ρ at each severity:

| Method | sev 1 | sev 2 | sev 3 | sev 4 | sev 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SAOD, top-3 | 0.109 | 0.181 | 0.212 | 0.208 | 0.165 |
| SAOD, min | 0.152 | 0.226 | 0.284 | 0.318 | 0.308 |
| ContrastiveConf | 0.102 | 0.157 | 0.231 | 0.297 | 0.364 |
| kNN | 0.036 | 0.045 | 0.068 | 0.076 | 0.077 |
| DisCoPatch | −0.011 | −0.027 | −0.050 | −0.074 | −0.091 |

AURC per severity pool. Each pool is the clean images plus that severity's 19 conditions. Bold is the
best method:

| Pool | SAOD top-3 | SAOD min | ContrastiveConf | kNN | DisCoPatch | oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all conditions | 0.638 | 0.560 | **0.540** | 0.638 | 0.665 | 0.449 |
| severity 1 | 0.581 | 0.490 | **0.430** | 0.545 | 0.596 | 0.352 |
| severity 2 | 0.609 | 0.521 | **0.472** | 0.584 | 0.627 | 0.389 |
| severity 3 | 0.639 | 0.558 | **0.531** | 0.634 | 0.661 | 0.441 |
| severity 4 | 0.674 | 0.603 | **0.600** | 0.689 | 0.703 | 0.503 |
| severity 5 | 0.709 | **0.651** | 0.668 | 0.745 | 0.749 | 0.569 |

The family pools are in `aurc_pools.csv`. ContrastiveConf is best in 16 of the 19, and SAOD min wins
the other three: glass blur, zoom blur and pixelate.

SAOD top-3 and kNN have the same AURC over all conditions (0.638) only by coincidence. The family pools
tell them apart, for example on fog (0.558 vs 0.511).

## Differences between methods

Each cell is the first method minus the second, with a paired 95% interval. The full set, including
AUPR and FPR95 for both family groups, is in `differences.csv`.

| Pair | Δ AUROC, common | Δ ρ within condition | Δ AURC, all |
| --- | ---: | ---: | ---: |
| SAOD min − SAOD top-3 | 0.044 [0.040, 0.049] | 0.082 [0.068, 0.096] | −0.078 [−0.084, −0.072] |
| DisCoPatch − SAOD min | 0.033 [0.028, 0.038] | −0.308 [−0.330, −0.287] | 0.105 [0.100, 0.110] |
| SAOD min − ContrastiveConf | 0.162 [0.156, 0.170] | 0.027 [0.011, 0.044] | 0.020 [0.015, 0.024] |
| SAOD min − kNN | 0.137 [0.131, 0.142] | 0.197 [0.180, 0.213] | −0.078 [−0.084, −0.072] |
| ContrastiveConf − kNN | −0.026 [−0.033, −0.019] | 0.170 [0.151, 0.187] | −0.098 [−0.103, −0.091] |
| DisCoPatch − ContrastiveConf | 0.195 [0.188, 0.203] | −0.281 [−0.303, −0.257] | 0.125 [0.118, 0.131] |
| DisCoPatch − kNN | 0.170 [0.164, 0.175] | −0.111 [−0.132, −0.091] | 0.027 [0.021, 0.033] |
| SAOD top-3 − kNN | 0.092 [0.086, 0.098] | 0.115 [0.095, 0.132] | 0.000 [−0.006, 0.006] |

With 5,000 images the intervals are narrow. Every difference in the table excludes zero except the
SAOD top-3 vs kNN AURC.

## kNN: how much does k matter?

k was fixed at 100 before any number existed. This check is for the record, not for tuning.

| k | 1 | 10 | 50 | **100** | 200 |
| --- | ---: | ---: | ---: | ---: | ---: |
| mean AUROC, common | 0.646 | 0.623 | 0.605 | **0.594** | 0.582 |

A smaller k would help kNN, but even k = 1 stays well below SAOD min and DisCoPatch.

## Runtime

This is the median time per image at batch size 1 on an RTX 5090, over 100 images after 10 warm-up
images, with nothing else on the GPU. Preprocessing is included.

| Part | ms per image |
| --- | ---: |
| Detector alone | 5.88 |
| Detector + ContrastiveConf | 6.14 |
| Detector + kNN (search over 118k train features) | 6.78 |
| Detector + SAOD | 7.14 |
| DisCoPatch, standalone | 2.44 |

SAOD comes out slower than kNN because of how we implemented it, not because of the method. Our SAOD
post-processing runs a full numpy sort of all 300 × 80 (query, class) scores on the CPU.

## Deviations from the plan

1. **DisCoPatch training numerics.** The README batch (67 images × 48 patches) needs about 55 GB in
   fp32; the authors used a 94 GB H100. On our 32 GB RTX 5090:
   - Training recomputes activations during the backward pass, which gives the same arithmetic.
   - The conv layers run under bf16 autocast. The discriminator's last layer and sigmoid, the VAE's
     latent heads and the loss stay in fp32, because a bf16 sigmoid rounds to exactly 1.0.
   - Every README hyperparameter is kept, including the batch. The paper shows the discriminator
     depends on batch statistics.
   - Peak memory was 21.7 GB, and training took 30.3 h for 65 epochs.
   - Before the run, a 150-step check with the same seed matched fp32 closely: discriminator loss
     0.0866 vs 0.0847, and AUROC on noisy vs clean images 0.754 vs 0.757.
   - The alternative was fp32 at 44 images per batch, projected at about 52 h, over the plan's 36 h
     limit. The user chose to keep bf16.
   - `discopatch_training.json` records the setup, and `discopatch_checkpoint.json` holds the sha1 of
     the discriminator that was scored.
2. **DisCoPatch score precision.** The score 1 − mean(patch outputs) is computed in float64. In
   float32, images whose 64 patch outputs are all below about 6e-8 rounded to exactly 1.0 and tied.
   The scoring pass that had this problem was stopped after 78 images and rerun from scratch.
3. **Extra outputs.** These were not in the plan's first version:
   - pairwise differences for AUPR and FPR95 on the extra families;
   - the number of images with undefined LRP for every condition, not only clean.

## Limitations to state in the paper

- One detector (RT-DETRv2-R18), and COCO only. The driving-dataset numbers come next.
- DisCoPatch was trained once, for 65 epochs. That budget is inferred from the paper's ImageNet setup,
  and training used bf16 as described above. It keeps the official fixed 256-px resize and 64 × 64
  crops. The official `outlier_detection` function defaults to `patches=1`; we score 64 patches per
  image, as the paper describes.
- ContrastiveConf's λ is fitted on labelled clean images. It is cross-fitted over 5 folds, so no image
  is scored with a λ fitted on itself. The folds did not choose exactly the same λ (7 or 8), and the
  per-image scores use their own fold's λ.
- LRP uses one global confidence threshold (0.55) and our own crowd-ignore rule (IoA ≥ 0.5).
- Families and severities are weighted equally in every mean, although severities are not comparable
  across families.
- There is one corruption draw per image and condition.
- Balanced accuracy is not reported. A single threshold does not describe a monitor well, so the
  threshold-free AUROC, AUPR and FPR95 are used instead.
- The between-condition ρ with mAP has no interval, because mAP is a single number per condition.
- Separation and harm are measured on the same 5,000 images. Nothing is tuned on them except
  ContrastiveConf's cross-fitted λ.

## Files

Everything in `docs/results/coco-baselines/`:

- `separation.csv` gives AUROC, AUPR and FPR95 for each method × condition.
- `aggregates.csv` gives the group means.
- `harm.csv` gives the correlations.
- `aurc_pools.csv` gives AURC and the oracle for each pool.
- `conditions.csv` gives each condition's mAP, mean LRP, undefined-LRP count and mean scores.
- `intervals.csv` and `differences.csv` hold the bootstrap intervals.
- `knn_k.csv`, `timing.csv` and `timing.json` hold the k check and the runtimes.
- Provenance: `summary.json`, `run_config.json`, `environment.json`, `sanity.json`,
  `discopatch_training.json` and `discopatch_checkpoint.json`.
- `report.md` is the generated report.

The per-image scores (about 1.5 GB) stay in `runs/coco-baselines/` in the main checkout. That folder
is not in git.
