# Conv-layer Topological Uncertainty: pilot results

Run on 30 September 2026, following `docs/superpowers/plans/2026-09-30-conv-tu-pilot.md` and the design in `docs/conv-tu-pilot-design.md`. The raw tables, `summary.json`, the calibration and the z-statistics are in `docs/results/conv-tu-pilot/`.

What was scored: the first 200 COCO val2017 images of the seed-44 order, each in all 96 conditions (19,200 variants). The layers were the four stride-1 3 × 3 convs `res_layers[s].blocks[1].branch2a.conv`, and the fingerprint was the exact top K = 1% of each conv graph's persistence diagram. The nine baseline rows are recomputed on the same 200 images.

## Verdict

At K = 1%, the pilot is negative. Neither criterion of the design note is met.

- **The topology adds nothing measurable.** The fingerprint (**mst**) is indistinguishable from the same number of heaviest edges without the cycle rule (**edges**), and from the largest activations (**acts**). Every difference is within ±0.012 AUROC, and every interval includes 0.
- **It trails the strongest baselines.** Mean AUROC over the 15 common families is 0.661, against 0.825 for the ICPR activation-distribution monitor, 0.760 for DisCoPatch and 0.742 for SAOD min.
- **Harm tracking is modest.** ρ(Δscore, ΔLRP) within conditions is 0.100 [0.040, 0.160]. That is above DisCoPatch (−0.054) and the ICPR monitor (0.021), and below ContrastiveConf (0.336) and SAOD min (0.263).
- **The simplest control is the strongest separator.** The mean |activation| per channel (**means**) reaches AUROC 0.798 on the common families and 0.847 on the extra ones. The extra-family value is the best of all 13 rows. It tracks harm weakly, at 0.074.

The top of the diagram behaves this way because at K = 1% the heaviest edges rarely close a loop. Before the pilot, the mean gap between mst and edges was 0.4% at stage 1 and 7.3% at stage 4. So the top of the diagram is almost an unfiltered list of the largest products, and those follow the largest activations.

## Clean-image checks (calibration on 200 clean train images)

| Stage | K | Cut | K-th value, median [min, max] | CV | Edges read, median (max) | Edges at the cut, median (max) | ms per image |
|---|---:|---:|---|---:|---:|---:|---:|
| 1 | 32,768 | 0.148 | 0.400 [0.296, 0.574] | 0.13 | 33,693 (37,932) | 1.56 M (3.47 M) | 15.9 |
| 2 | 16,384 | 0.096 | 0.262 [0.193, 0.356] | 0.11 | 18,338 (21,564) | 0.90 M (1.70 M) | 14.0 |
| 3 | 8,192 | 0.048 | 0.142 [0.097, 0.192] | 0.11 | 8,939 (10,828) | 0.65 M (1.49 M) | 13.2 |
| 4 | 4,096 | 0.038 | 0.119 [0.076, 0.173] | 0.15 | 4,646 (10,096) | 0.49 M (1.07 M) | 12.5 |

- **The K-th value is stable across clean images.** Its coefficient of variation is 0.11–0.15.
- **The K merges need only 3–13% more edges than K** at the median. That is at most 0.004% of a layer's ~0.9 billion edges.
- **Every variant needed one round.** All 19,200 pilot variants, corrupted ones included, were exact on the first try. The lowest K-th value under any corruption (0.262 / 0.170 / 0.067 / 0.048 for stages 1–4) stayed above the clean cut.
- **Cost:**
  - peak GPU memory was 1.6 GiB reserved in calibration and about 3.6 GiB while scoring, bank included;
  - scoring the 19,200 variants took 19 minutes, next to another session's evaluation job.

## Separation and harm on the 200 images

Higher AUROC is better (chance 0.5); higher ρ is better; lower AURC is better. Brackets are 95% paired bootstrap intervals over images (1,000 draws).

| Method | AUROC common | AUROC extra | ρ(Δscore, ΔLRP) within | ρ(score, LRP) conditions | AURC all |
|---|---|---|---|---|---|
| Conv TU: top 1% of the diagram | 0.661 [0.634, 0.687] | 0.615 [0.595, 0.633] | 0.100 [0.040, 0.160] | 0.582 [0.538, 0.616] | 0.667 [0.636, 0.698] |
| Control: top 1% heaviest edges | 0.658 [0.631, 0.685] | 0.609 [0.588, 0.630] | 0.096 [0.035, 0.155] | 0.572 [0.527, 0.607] | 0.666 [0.634, 0.697] |
| Control: largest activations | 0.671 [0.647, 0.695] | 0.603 [0.584, 0.621] | 0.099 [0.043, 0.158] | 0.717 [0.684, 0.746] | 0.671 [0.642, 0.699] |
| Control: channel means | 0.798 [0.780, 0.816] | 0.847 [0.829, 0.864] | 0.074 [0.021, 0.123] | 0.500 [0.462, 0.527] | 0.649 [0.620, 0.679] |
| Activation CDFs (Becker et al., ICPR 2026) | 0.825 [0.807, 0.845] | 0.808 [0.791, 0.824] | 0.021 [−0.036, 0.075] | 0.677 [0.649, 0.708] | 0.664 [0.635, 0.692] |
| Activation CDFs, plain channel sum | 0.785 [0.765, 0.806] | 0.758 [0.739, 0.776] | 0.058 [−0.000, 0.114] | 0.763 [0.734, 0.790] | 0.662 [0.633, 0.690] |
| DisCoPatch | 0.760 [0.743, 0.779] | 0.747 [0.731, 0.765] | −0.054 [−0.130, 0.026] | 0.616 [0.578, 0.657] | 0.657 [0.624, 0.688] |
| SAOD, min (1 − max confidence) | 0.742 [0.726, 0.760] | 0.669 [0.655, 0.685] | 0.263 [0.206, 0.316] | 0.981 [0.970, 0.984] | 0.562 [0.526, 0.598] |
| SAOD, mean of top 3 | 0.682 [0.663, 0.702] | 0.622 [0.607, 0.639] | 0.123 [0.055, 0.191] | 0.988 [0.977, 0.988] | 0.634 [0.605, 0.663] |
| kNN (k = 100) | 0.593 [0.573, 0.614] | 0.628 [0.612, 0.645] | 0.118 [0.058, 0.172] | 0.544 [0.456, 0.595] | 0.637 [0.604, 0.669] |
| ContrastiveConf | 0.579 [0.558, 0.603] | 0.557 [0.543, 0.578] | 0.336 [0.285, 0.382] | 0.779 [0.692, 0.848] | 0.551 [0.511, 0.591] |
| Hashemi et al., decoder queries | 0.435 [0.411, 0.456] | 0.440 [0.420, 0.457] | 0.125 [0.060, 0.186] | −0.393 [−0.527, −0.201] | 0.633 [0.594, 0.672] |
| Hashemi et al., encoder maps | 0.348 [0.330, 0.365] | 0.395 [0.377, 0.412] | 0.018 [−0.041, 0.078] | −0.663 [−0.700, −0.617] | 0.718 [0.687, 0.750] |

## Differences that decide the pilot (fingerprint minus the other row)

| Pair | Δ AUROC common | Δ AUROC extra | Δ ρ within | Δ AURC all |
|---|---|---|---|---|
| mst − edges | +0.003 [−0.004, +0.009] | +0.006 [−0.001, +0.013] | +0.004 [−0.011, +0.018] | +0.001 [−0.000, +0.003] |
| mst − acts | −0.010 [−0.028, +0.007] | +0.012 [−0.006, +0.028] | +0.001 [−0.039, +0.042] | −0.004 [−0.012, +0.005] |
| mst − means | −0.138 [−0.171, −0.107] | −0.232 [−0.258, −0.208] | +0.026 [−0.044, +0.100] | +0.019 [+0.001, +0.038] |
| mst − Activation CDFs | −0.165 [−0.192, −0.138] | −0.193 [−0.216, −0.171] | +0.079 [+0.003, +0.151] | +0.003 [−0.009, +0.015] |
| mst − DisCoPatch | −0.099 [−0.132, −0.068] | −0.132 [−0.158, −0.108] | +0.154 [+0.036, +0.274] | +0.010 [−0.005, +0.027] |
| mst − ContrastiveConf | +0.082 [+0.049, +0.110] | +0.057 [+0.029, +0.080] | −0.236 [−0.306, −0.153] | +0.117 [+0.094, +0.140] |

## Depth: one layer at a time

Each layer's own kNN distance, without the z-scored sum.

| Stage | mst AUROC common / extra / ρ within | edges | acts | means |
|---|---|---|---|---|
| 1 | 0.678 / 0.663 / 0.038 | 0.684 / 0.665 / 0.046 | 0.667 / 0.653 / −0.007 | 0.741 / 0.839 / 0.040 |
| 2 | 0.658 / 0.662 / 0.070 | 0.667 / 0.666 / 0.070 | 0.663 / 0.670 / 0.090 | 0.824 / 0.859 / 0.047 |
| 3 | 0.582 / 0.516 / 0.063 | 0.577 / 0.509 / 0.065 | 0.645 / 0.519 / 0.059 | 0.825 / 0.822 / 0.069 |
| 4 | 0.600 / 0.504 / 0.079 | 0.583 / 0.505 / 0.060 | 0.596 / 0.499 / 0.100 | 0.535 / 0.594 / 0.099 |

The fingerprint's largest edge over the heaviest edges is at stage 4, the layer where the cycle rule acts most: +0.017 AUROC common and +0.019 ρ. It is small and has no interval. The channel means separate best at stages 2 and 3.

## Caveats

- **Sample size.** These are 200 images, not the 5,000 of `docs/coco-baseline-numbers.md`, so the intervals are wider. The baseline rows differ from that document only through the image subset: they use the full run's stored LRP threshold (0.55) and per-fold λ.
- **ContrastiveConf intervals** leave out λ uncertainty. The pilot keeps the full run's per-fold λ fixed in every bootstrap draw, while the full report refits it (final review, finding M6).
- **One setting of K.** The pilot tested K = 1% of nodes only. Deeper into the diagram, the gap between mst and edges grows: 10–25% at stages 3–4 for K = 25–50% on one clean image.
