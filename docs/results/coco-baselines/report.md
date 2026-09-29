# COCO baseline numbers

All scores are oriented so that higher means more likely degraded. AUROC and AUPR: higher is better (chance 0.5). FPR95: lower is better. Brackets are 95% paired bootstrap intervals over images.

## Separation, 15 common families

| Method | AUROC sev 1 | sev 2 | sev 3 | sev 4 | sev 5 | AUROC all | AUPR all | FPR95 all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SAOD, mean of top 3 | 0.580 | 0.625 | 0.683 | 0.745 | 0.798 | 0.686 [0.682, 0.690] | 0.644 [0.638, 0.649] | 0.729 [0.722, 0.736] |
| SAOD, min (1 − max confidence) | 0.608 | 0.665 | 0.733 | 0.799 | 0.848 | 0.731 [0.727, 0.734] | 0.717 [0.711, 0.723] | 0.714 [0.707, 0.722] |
| ContrastiveConf | 0.527 | 0.545 | 0.569 | 0.595 | 0.605 | 0.568 [0.562, 0.574] | 0.532 [0.527, 0.539] | 0.869 [0.864, 0.875] |
| kNN (k = 100) | 0.546 | 0.559 | 0.590 | 0.625 | 0.650 | 0.594 [0.590, 0.598] | 0.561 [0.557, 0.565] | 0.847 [0.842, 0.852] |
| DisCoPatch | 0.678 | 0.733 | 0.777 | 0.805 | 0.825 | 0.764 [0.760, 0.767] | 0.729 [0.724, 0.734] | 0.567 [0.558, 0.576] |

## Separation, 4 extra families

| Method | AUROC sev 1 | sev 2 | sev 3 | sev 4 | sev 5 | AUROC all | AUPR all | FPR95 all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SAOD, mean of top 3 | 0.531 | 0.577 | 0.618 | 0.667 | 0.731 | 0.625 [0.622, 0.628] | 0.590 [0.586, 0.595] | 0.820 [0.814, 0.825] |
| SAOD, min (1 − max confidence) | 0.543 | 0.607 | 0.660 | 0.720 | 0.787 | 0.664 [0.660, 0.667] | 0.652 [0.646, 0.657] | 0.811 [0.804, 0.817] |
| ContrastiveConf | 0.507 | 0.531 | 0.548 | 0.582 | 0.600 | 0.554 [0.550, 0.558] | 0.530 [0.527, 0.535] | 0.895 [0.890, 0.899] |
| kNN (k = 100) | 0.518 | 0.587 | 0.605 | 0.673 | 0.725 | 0.622 [0.619, 0.625] | 0.590 [0.587, 0.594] | 0.802 [0.796, 0.807] |
| DisCoPatch | 0.633 | 0.755 | 0.756 | 0.787 | 0.824 | 0.751 [0.747, 0.755] | 0.717 [0.712, 0.723] | 0.580 [0.572, 0.588] |

## Harm alignment

| Method | ρ(score, mAP), conditions | ρ(score, LRP), conditions | ρ(Δscore, ΔLRP), within condition | AURC all (oracle) |
| --- | ---: | ---: | ---: | ---: |
| SAOD, mean of top 3 | -0.986 | 0.988 [0.986, 0.989] | 0.175 [0.162, 0.187] | 0.638 [0.631, 0.645] (0.449) |
| SAOD, min (1 − max confidence) | -0.981 | 0.983 [0.981, 0.985] | 0.257 [0.246, 0.268] | 0.560 [0.552, 0.568] (0.449) |
| ContrastiveConf | -0.729 | 0.744 [0.688, 0.787] | 0.230 [0.216, 0.243] | 0.540 [0.532, 0.549] (0.449) |
| kNN (k = 100) | -0.534 | 0.542 [0.522, 0.554] | 0.061 [0.049, 0.072] | 0.638 [0.631, 0.645] (0.449) |
| DisCoPatch | -0.636 | 0.640 [0.628, 0.653] | -0.051 [-0.066, -0.034] | 0.665 [0.658, 0.672] (0.449) |

## Differences between methods

| Pair | Δ AUROC common | Δ ρ within condition | Δ AURC all |
| --- | ---: | ---: | ---: |
| saod_top3 - saod_min | -0.044 [-0.049, -0.040] | -0.082 [-0.096, -0.068] | 0.078 [0.072, 0.084] |
| saod_top3 - contrastive | 0.118 [0.111, 0.126] | -0.055 [-0.074, -0.035] | 0.098 [0.090, 0.105] |
| saod_top3 - knn | 0.092 [0.086, 0.098] | 0.115 [0.095, 0.132] | 0.000 [-0.006, 0.006] |
| saod_top3 - discopatch | -0.077 [-0.082, -0.072] | 0.226 [0.205, 0.245] | -0.027 [-0.033, -0.021] |
| saod_min - contrastive | 0.162 [0.156, 0.170] | 0.027 [0.011, 0.044] | 0.020 [0.015, 0.024] |
| saod_min - knn | 0.137 [0.131, 0.142] | 0.197 [0.180, 0.213] | -0.078 [-0.084, -0.072] |
| saod_min - discopatch | -0.033 [-0.038, -0.028] | 0.308 [0.287, 0.330] | -0.105 [-0.110, -0.100] |
| contrastive - knn | -0.026 [-0.033, -0.019] | 0.170 [0.151, 0.187] | -0.098 [-0.103, -0.091] |
| contrastive - discopatch | -0.195 [-0.203, -0.188] | 0.281 [0.257, 0.303] | -0.125 [-0.131, -0.118] |
| knn - discopatch | -0.170 [-0.175, -0.164] | 0.111 [0.091, 0.132] | -0.027 [-0.033, -0.021] |

## Runtime (median ms per image, batch 1)

| Part | ms |
| --- | ---: |
| detector_ms | 5.879 |
| detector_plus_contrastive_ms | 6.144 |
| detector_plus_knn_ms | 6.784 |
| detector_plus_saod_ms | 7.143 |
| discopatch_ms | 2.445 |

## Fixed choices

- bootstrap_samples: 1000
- clean_map: 0.4790545764060369
- discopatch_included: True
- folds: 5
- images: 5000
- images_with_ap: 4952
- images_with_undefined_clean_lrp: 32
- knn_k: 100
- lambda_folds_agree: False
- lambda_per_fold: {0: 7.0, 1: 8.0, 2: 7.0, 3: 7.0, 4: 7.0}
- lrp_threshold: 0.55
- theta: 0.3
