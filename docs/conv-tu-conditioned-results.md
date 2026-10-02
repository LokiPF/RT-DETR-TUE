# Corruption detection from the early channels: 5,000-image confirmation

Run on 2 October 2026, following `docs/superpowers/plans/2026-10-01-content-conditioned-confirmation.md` and its amendment 2. The raw tables, `summary.json` and the generated `report.md` are in `docs/results/conv-tu-conditioned/`.

**What was scored:**
- **Images:** all 5,000 COCO val2017 images, in the seed-44 order, in all 96 conditions (480,000 variants). These are the same images as the baseline numbers in `docs/coco-baseline-numbers.md`.
- **Statistics:** for every channel of backbone stages 1–4, the level (mean |activation|) and the mean of its strongest 1% of positions.
- **Reference:** 2,000 clean COCO train images, plus 500 more for the z-statistics.
- **Rows:** the method's rows are described in `docs/dev-log.md`, entries of 1 October (evening and night).

## How to read the tables

↑ means higher is better and ↓ means lower is better.

| Column | What it measures | Reference values |
|---|---|---|
| AUROC common ↑ | How well the score tells a corrupted image from its clean version, averaged over the 75 conditions of the 15 common families | 0.5 = chance, 1 = perfect |
| AUROC extra ↑ | The same, over the 20 conditions of the 4 extra families | 0.5 = chance, 1 = perfect |
| FPR95 ↓ | The share of clean images flagged when the threshold catches 95% of the corrupted ones | 0 = perfect |

Brackets are 95% paired bootstrap intervals over images (1,000 draws, seed 44). In a difference (first row minus second), a positive Δ AUROC means the first row is better.

## Verdict

Both pre-registered decisions are **confirmed**.

**The headline: the two-axis score** (the larger of a "flatter peaks" arm and a "shifted level" arm, each judged against the 50 most similar clean scenes).
- **All 5,000 images:** AUROC 0.917 [0.914, 0.920] on the common families and 0.858 [0.856, 0.861] on the extra ones.
- **Against the activation CDFs** (Becker et al., ICPR 2026), the strongest baseline: +0.096 [+0.093, +0.100] common and +0.051 [+0.048, +0.054] extra.
- **Against the level score alone:** +0.076 [+0.073, +0.079] on the common families. This is what the "flatter" arm adds.
- **The untouched images** (the 3,030 at positions 1970 and later, which nobody read while the method was designed) give the same result:
  - AUROC 0.916 / 0.858;
  - minus the CDFs: +0.095 [+0.090, +0.099] / +0.050 [+0.046, +0.054];
  - minus the level score: +0.076 [+0.073, +0.080] common.

**The pre-registered level score, on the 4,800 held-out images.** "Level vs the 50 most similar clean scenes" beats both comparisons on both family groups:
- the same level against the average of all clean images: +0.041 [+0.039, +0.044] common and +0.037 [+0.035, +0.039] extra. So choosing similar scenes as the reference is what helps;
- the activation CDFs: +0.020 [+0.017, +0.024] common and +0.059 [+0.056, +0.062] extra.

**No sign that reading part of the data inflated the numbers.** The two-axis score's AUROC common / extra on each image set:

| Image set | AUROC common | AUROC extra |
|---|---|---|
| The 200 screening images | 0.922 | 0.862 |
| The panel's held-out positions 1059–1969 (dev log) | 0.922 | 0.861 |
| All 5,000 | 0.917 | 0.858 |
| The 3,030 untouched | 0.916 | 0.858 |

## All 5,000 images

| Row | AUROC common ↑ | AUROC extra ↑ | FPR95 common ↓ | FPR95 extra ↓ |
|---|---|---|---|---|
| **Two-axis vs the 50 most similar clean scenes (headline)** | **0.917** [0.914, 0.920] | 0.858 [0.856, 0.861] | **0.226** | 0.360 |
| Peak share vs the 50 most similar clean scenes | 0.898 [0.895, 0.902] | 0.861 [0.858, 0.864] | 0.250 | 0.342 |
| Level vs the 50 most similar clean scenes | 0.841 [0.837, 0.844] | **0.866** [0.863, 0.869] | 0.361 | **0.327** |
| Level vs the average of all clean images | 0.800 [0.796, 0.803] | 0.830 [0.827, 0.832] | 0.439 | 0.397 |
| Channel means, kNN, 4 stages | 0.801 [0.797, 0.805] | 0.843 [0.839, 0.847] | 0.440 | 0.381 |
| Channel means vs own average, 4 stages | 0.782 | 0.808 | 0.478 | 0.442 |
| Activation CDFs (Becker et al., ICPR 2026) | 0.821 [0.817, 0.824] | 0.807 [0.804, 0.811] | 0.413 | 0.429 |
| DisCoPatch | 0.764 [0.760, 0.767] | 0.751 [0.747, 0.755] | 0.567 | 0.580 |
| SAOD, min (1 − max confidence) | 0.731 | 0.664 | 0.714 | 0.811 |
| kNN (k = 100) | 0.594 | 0.622 | 0.847 | 0.802 |
| Hashemi et al., decoder queries | 0.428 | 0.437 | 0.959 | 0.954 |

The remaining baselines are in `docs/coco-baseline-numbers.md`, on the same images: SAOD top-3 0.686 / 0.625, ContrastiveConf 0.568 / 0.554.

## Where it is strong, and where it is weak

**Mild corruptions.** Severity 1 of the common families:

| Row | AUROC, severity 1 common |
|---|---|
| Two-axis | 0.863 |
| Peak share | 0.816 |
| Level vs similar scenes | 0.761 |
| Activation CDFs | 0.707 |
| DisCoPatch | 0.678 |

**Corruptions that remove structure.** Fog, contrast and the blurs flatten the peaks, and the flatter arm catches them. At severities 1 / 3 / 5:

| Family | Two-axis | Activation CDFs | DisCoPatch |
|---|---|---|---|
| Fog | 0.95 / 0.98 / 0.99 | 0.64 / 0.73 / 0.75 | 0.60 / 0.72 / 0.84 |
| Contrast | 0.94 / 0.99 / 1.00 | 0.67 / 0.85 / 0.99 | 0.54 / 0.61 / 0.78 |
| Defocus blur | 0.95 / 0.98 / 0.99 | 0.80 / 0.94 / 0.98 | 0.66 / 0.80 / 0.85 |

**Weak spots:**
- **The extra families.** There the two-axis score is slightly behind the level score: −0.008 [−0.010, −0.005] on all images. The headline rule only asked for a gain on the common families, where it is +0.076.
- **Photometric changes.** Brightness is 0.53 / 0.60 / 0.71 and saturate 0.60 / 0.54 / 0.84. Every row is weak there: the detector is trained to ignore these changes.
- **Elastic transform and frost at severity 1:** 0.68 and 0.72.

The full per-family table is in `docs/results/conv-tu-conditioned/report.md`.

## Checks

- **The statistics match the pilot's.** For the 200 screening images, the channel statistics of this pass equal those of the pilot pass exactly: the largest relative difference is 0 for every stage and statistic.
- **The screen is reproduced.** The level row gives 0.841 / 0.870 on the screening images, as in the screen (tolerance 0.003).
- **The baselines match.** The baseline rows equal `docs/coco-baseline-numbers.md` (for example, activation CDFs 0.821 / 0.807), because both read the same stored scores.
- **The fixed choices,** frozen before the run: k = 50 neighbours, the stage-4 content key, scored stages 1–3, and the top 1%.

## Not yet shown

**Real fog.** These are synthetic corruptions of COCO images, with a COCO clean bank. The deciding test is Foggy Cityscapes, with a Cityscapes clean bank (dev log, 1 October, night).
