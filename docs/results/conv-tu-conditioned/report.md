# Corruption detection from a frozen detector's early channels: 5,000-image confirmation

**Headline (two-axis score; all images and the untouched ones):** confirmed.

**Pre-registered level score (held-out images):** confirmed.

↑ higher is better, ↓ lower is better. Brackets are 95% paired bootstrap intervals over images (1000 draws, seed 44).

## All 5000 images (the headline; the same images as the baselines)

| Row | AUROC common ↑ | AUROC extra ↑ | FPR95 common ↓ | FPR95 extra ↓ |
|---|---|---|---|---|
| Two-axis: flatter or shifted vs the 50 most similar clean scenes (headline) | 0.917 [0.914, 0.920] | 0.858 [0.856, 0.861] | 0.226 | 0.360 |
| Peak share vs the 50 most similar clean scenes | 0.898 [0.895, 0.902] | 0.861 [0.858, 0.864] | 0.250 | 0.342 |
| Level vs the 50 most similar clean scenes (stage-4 key) | 0.841 [0.837, 0.844] | 0.866 [0.863, 0.869] | 0.361 | 0.327 |
| Level vs the average of all clean images (stages 1–3) | 0.800 [0.796, 0.803] | 0.830 [0.827, 0.832] | 0.439 | 0.397 |
| Channel means, kNN, 4 stages (the pilot's control) | 0.801 [0.797, 0.805] | 0.843 [0.839, 0.847] | 0.440 | 0.381 |
| Channel means vs own average, 4 stages | 0.782 | 0.808 | 0.478 | 0.442 |
| Activation CDFs (Becker et al., ICPR 2026) | 0.821 [0.817, 0.824] | 0.807 [0.804, 0.811] | 0.413 | 0.429 |
| DisCoPatch | 0.764 [0.760, 0.767] | 0.751 [0.747, 0.755] | 0.567 | 0.580 |
| SAOD, min (1 − max confidence) | 0.731 | 0.664 | 0.714 | 0.811 |
| kNN (k = 100) | 0.594 | 0.622 | 0.847 | 0.802 |
| Hashemi et al., decoder queries | 0.428 | 0.437 | 0.959 | 0.954 |

Differences, two-axis: flatter or shifted vs the 50 most similar clean scenes minus each other row (positive Δ: it is better):

| Other row | Δ AUROC common ↑ | Δ AUROC extra ↑ |
|---|---|---|
| Peak share vs the 50 most similar clean scenes | +0.019 [+0.016, +0.021] | −0.002 [−0.005, +0.000] |
| Level vs the 50 most similar clean scenes (stage-4 key) | +0.076 [+0.073, +0.079] | −0.008 [−0.010, −0.005] |
| Level vs the average of all clean images (stages 1–3) | +0.117 [+0.114, +0.120] | +0.029 [+0.026, +0.031] |
| Channel means, kNN, 4 stages (the pilot's control) | +0.116 [+0.112, +0.119] | +0.015 [+0.013, +0.018] |
| Activation CDFs (Becker et al., ICPR 2026) | +0.096 [+0.093, +0.100] | +0.051 [+0.048, +0.054] |
| DisCoPatch | +0.153 [+0.149, +0.158] | +0.107 [+0.103, +0.112] |

## Untouched images (3030, positions 1970 and later, read by nobody while the method was designed)

| Row | AUROC common ↑ | AUROC extra ↑ | FPR95 common ↓ | FPR95 extra ↓ |
|---|---|---|---|---|
| Two-axis: flatter or shifted vs the 50 most similar clean scenes (headline) | 0.916 [0.912, 0.920] | 0.858 [0.854, 0.862] | 0.227 | 0.362 |
| Peak share vs the 50 most similar clean scenes | 0.899 [0.895, 0.903] | 0.861 [0.857, 0.865] | 0.249 | 0.343 |
| Level vs the 50 most similar clean scenes (stage-4 key) | 0.840 [0.835, 0.845] | 0.865 [0.861, 0.869] | 0.363 | 0.328 |
| Level vs the average of all clean images (stages 1–3) | 0.799 [0.795, 0.804] | 0.829 [0.825, 0.833] | 0.439 | 0.399 |
| Channel means, kNN, 4 stages (the pilot's control) | 0.801 [0.797, 0.806] | 0.843 [0.839, 0.848] | 0.441 | 0.381 |
| Channel means vs own average, 4 stages | 0.782 | 0.808 | 0.476 | 0.441 |
| Activation CDFs (Becker et al., ICPR 2026) | 0.822 [0.817, 0.826] | 0.808 [0.804, 0.812] | 0.412 | 0.429 |
| DisCoPatch | 0.761 [0.757, 0.766] | 0.748 [0.744, 0.753] | 0.572 | 0.585 |
| SAOD, min (1 − max confidence) | 0.731 | 0.664 | 0.716 | 0.812 |
| kNN (k = 100) | 0.595 | 0.623 | 0.847 | 0.804 |
| Hashemi et al., decoder queries | 0.430 | 0.438 | 0.956 | 0.952 |

Differences, two-axis: flatter or shifted vs the 50 most similar clean scenes minus each other row (positive Δ: it is better):

| Other row | Δ AUROC common ↑ | Δ AUROC extra ↑ |
|---|---|---|
| Peak share vs the 50 most similar clean scenes | +0.017 [+0.014, +0.021] | −0.003 [−0.007, −0.000] |
| Level vs the 50 most similar clean scenes (stage-4 key) | +0.076 [+0.073, +0.080] | −0.008 [−0.011, −0.004] |
| Level vs the average of all clean images (stages 1–3) | +0.117 [+0.113, +0.121] | +0.029 [+0.025, +0.032] |
| Channel means, kNN, 4 stages (the pilot's control) | +0.115 [+0.111, +0.119] | +0.014 [+0.011, +0.018] |
| Activation CDFs (Becker et al., ICPR 2026) | +0.095 [+0.090, +0.099] | +0.050 [+0.046, +0.054] |
| DisCoPatch | +0.155 [+0.150, +0.160] | +0.110 [+0.105, +0.114] |

## Held-out images (4800, positions 200 and later: the pre-registered check of the level score)

| Row | AUROC common ↑ | AUROC extra ↑ | FPR95 common ↓ | FPR95 extra ↓ |
|---|---|---|---|---|
| Two-axis: flatter or shifted vs the 50 most similar clean scenes (headline) | 0.917 [0.914, 0.920] | 0.858 [0.856, 0.861] | 0.226 | 0.360 |
| Peak share vs the 50 most similar clean scenes | 0.898 [0.895, 0.901] | 0.861 [0.858, 0.863] | 0.250 | 0.342 |
| Level vs the 50 most similar clean scenes (stage-4 key) | 0.841 [0.837, 0.844] | 0.866 [0.863, 0.869] | 0.360 | 0.327 |
| Level vs the average of all clean images (stages 1–3) | 0.799 [0.796, 0.803] | 0.829 [0.826, 0.832] | 0.440 | 0.397 |
| Channel means, kNN, 4 stages (the pilot's control) | 0.801 [0.797, 0.805] | 0.843 [0.839, 0.847] | 0.440 | 0.381 |
| Channel means vs own average, 4 stages | 0.781 | 0.808 | 0.478 | 0.442 |
| Activation CDFs (Becker et al., ICPR 2026) | 0.820 [0.817, 0.824] | 0.807 [0.804, 0.810] | 0.414 | 0.428 |
| DisCoPatch | 0.764 [0.760, 0.767] | 0.751 [0.747, 0.755] | 0.567 | 0.580 |
| SAOD, min (1 − max confidence) | 0.730 | 0.663 | 0.714 | 0.811 |
| kNN (k = 100) | 0.594 | 0.622 | 0.847 | 0.802 |
| Hashemi et al., decoder queries | 0.428 | 0.437 | 0.959 | 0.954 |

Differences, level vs the 50 most similar clean scenes minus each other row (positive Δ: it is better):

| Other row | Δ AUROC common ↑ | Δ AUROC extra ↑ |
|---|---|---|
| Peak share vs the 50 most similar clean scenes | −0.058 [−0.061, −0.055] | +0.005 [+0.003, +0.008] |
| Level vs the average of all clean images (stages 1–3) | +0.041 [+0.039, +0.044] | +0.037 [+0.035, +0.039] |
| Channel means, kNN, 4 stages (the pilot's control) | +0.039 [+0.037, +0.042] | +0.023 [+0.021, +0.025] |
| Activation CDFs (Becker et al., ICPR 2026) | +0.020 [+0.017, +0.024] | +0.059 [+0.056, +0.062] |
| DisCoPatch | +0.077 [+0.072, +0.082] | +0.115 [+0.111, +0.119] |

## The 200 screening images

| Row | AUROC common ↑ | AUROC extra ↑ | FPR95 common ↓ | FPR95 extra ↓ |
|---|---|---|---|---|
| Two-axis: flatter or shifted vs the 50 most similar clean scenes (headline) | 0.922 | 0.862 | 0.229 | 0.358 |
| Peak share vs the 50 most similar clean scenes | 0.897 | 0.859 | 0.253 | 0.352 |
| Level vs the 50 most similar clean scenes (stage-4 key) | 0.841 | 0.870 | 0.368 | 0.317 |
| Level vs the average of all clean images (stages 1–3) | 0.811 | 0.837 | 0.431 | 0.394 |
| Channel means, kNN, 4 stages (the pilot's control) | 0.798 | 0.847 | 0.441 | 0.378 |
| Channel means vs own average, 4 stages | 0.794 | 0.818 | 0.467 | 0.431 |
| Activation CDFs (Becker et al., ICPR 2026) | 0.825 | 0.808 | 0.403 | 0.428 |
| DisCoPatch | 0.760 | 0.747 | 0.554 | 0.567 |
| SAOD, min (1 − max confidence) | 0.742 | 0.669 | 0.704 | 0.806 |
| kNN (k = 100) | 0.593 | 0.628 | 0.847 | 0.781 |
| Hashemi et al., decoder queries | 0.435 | 0.440 | 0.953 | 0.946 |

## AUROC by severity (all images)

| Row | Common, severities 1–5 ↑ | Extra, severities 1–5 ↑ |
|---|---|---|
| Two-axis: flatter or shifted vs the 50 most similar clean scenes (headline) | 0.863 / 0.905 / 0.926 / 0.940 / 0.951 | 0.705 / 0.849 / 0.859 / 0.924 / 0.955 |
| Peak share vs the 50 most similar clean scenes | 0.816 / 0.877 / 0.913 / 0.935 / 0.950 | 0.679 / 0.848 / 0.872 / 0.938 / 0.966 |
| Level vs the 50 most similar clean scenes (stage-4 key) | 0.761 / 0.815 / 0.850 / 0.878 / 0.899 | 0.683 / 0.858 / 0.881 / 0.943 / 0.965 |
| Level vs the average of all clean images (stages 1–3) | 0.700 / 0.765 / 0.809 / 0.849 / 0.876 | 0.624 / 0.803 / 0.859 / 0.915 / 0.946 |
| Channel means, kNN, 4 stages (the pilot's control) | 0.721 / 0.775 / 0.815 / 0.838 / 0.855 | 0.649 / 0.813 / 0.862 / 0.936 / 0.956 |
| Channel means vs own average, 4 stages | 0.672 / 0.740 / 0.793 / 0.837 / 0.867 | 0.602 / 0.774 / 0.839 / 0.895 / 0.932 |
| Activation CDFs (Becker et al., ICPR 2026) | 0.707 / 0.781 / 0.836 / 0.877 / 0.901 | 0.610 / 0.762 / 0.834 / 0.896 / 0.935 |
| DisCoPatch | 0.678 / 0.733 / 0.777 / 0.805 / 0.825 | 0.633 / 0.755 / 0.756 / 0.787 / 0.824 |
| SAOD, min (1 − max confidence) | 0.608 / 0.665 / 0.733 / 0.799 / 0.848 | 0.543 / 0.607 / 0.660 / 0.720 / 0.787 |
| kNN (k = 100) | 0.546 / 0.559 / 0.590 / 0.625 / 0.650 | 0.518 / 0.587 / 0.605 / 0.673 / 0.725 |
| Hashemi et al., decoder queries | 0.443 / 0.425 / 0.421 / 0.422 / 0.430 | 0.471 / 0.443 / 0.416 / 0.432 / 0.426 |

## AUROC by family at severities 1 / 3 / 5 (all images)

| Family | Two-axis | Peak share | Level (similar scenes) | Activation CDFs | DisCoPatch |
|---|---|---|---|---|---|
| gaussian noise | 0.93 / 1.00 / 1.00 | 0.95 / 1.00 / 1.00 | 0.96 / 1.00 / 1.00 | 0.89 / 0.99 / 1.00 | 0.88 / 0.99 / 1.00 |
| shot noise | 0.91 / 1.00 / 1.00 | 0.93 / 0.99 / 1.00 | 0.95 / 1.00 / 1.00 | 0.88 / 0.99 / 1.00 | 0.87 / 0.97 / 0.99 |
| impulse noise | 0.99 / 1.00 / 1.00 | 0.98 / 0.99 / 1.00 | 0.99 / 1.00 / 1.00 | 0.97 / 0.99 / 1.00 | 0.92 / 0.99 / 1.00 |
| defocus blur | 0.95 / 0.98 / 0.99 | 0.93 / 0.99 / 1.00 | 0.87 / 0.92 / 0.94 | 0.80 / 0.94 / 0.98 | 0.66 / 0.80 / 0.85 |
| glass blur | 0.91 / 0.96 / 0.98 | 0.85 / 0.96 / 0.99 | 0.76 / 0.92 / 0.95 | 0.68 / 0.90 / 0.96 | 0.68 / 0.88 / 0.94 |
| motion blur | 0.87 / 0.95 / 0.97 | 0.84 / 0.96 / 0.99 | 0.86 / 0.94 / 0.97 | 0.71 / 0.89 / 0.96 | 0.58 / 0.74 / 0.83 |
| zoom blur | 0.90 / 0.93 / 0.95 | 0.90 / 0.91 / 0.92 | 0.73 / 0.68 / 0.65 | 0.79 / 0.81 / 0.83 | 0.75 / 0.82 / 0.85 |
| snow | 0.92 / 0.97 / 0.98 | 0.92 / 0.97 / 0.98 | 0.95 / 0.98 / 0.99 | 0.83 / 0.91 / 0.93 | 0.76 / 0.84 / 0.84 |
| frost | 0.72 / 0.89 / 0.92 | 0.65 / 0.85 / 0.89 | 0.66 / 0.82 / 0.84 | 0.54 / 0.73 / 0.78 | 0.66 / 0.83 / 0.87 |
| fog | 0.95 / 0.98 / 0.99 | 0.81 / 0.91 / 0.93 | 0.51 / 0.58 / 0.62 | 0.64 / 0.73 / 0.75 | 0.60 / 0.72 / 0.84 |
| brightness | 0.53 / 0.60 / 0.71 | 0.53 / 0.65 / 0.79 | 0.54 / 0.69 / 0.82 | 0.51 / 0.59 / 0.70 | 0.49 / 0.50 / 0.52 |
| contrast | 0.94 / 0.99 / 1.00 | 0.83 / 0.97 / 1.00 | 0.53 / 0.66 / 0.89 | 0.67 / 0.85 / 0.99 | 0.54 / 0.61 / 0.78 |
| elastic transform | 0.68 / 0.76 / 0.84 | 0.58 / 0.69 / 0.80 | 0.60 / 0.75 / 0.87 | 0.52 / 0.61 / 0.72 | 0.61 / 0.68 / 0.72 |
| pixelate | 0.85 / 0.96 / 0.99 | 0.73 / 0.93 / 0.99 | 0.68 / 0.89 / 0.98 | 0.57 / 0.81 / 0.97 | 0.55 / 0.64 / 0.75 |
| jpeg compression | 0.89 / 0.94 / 0.97 | 0.83 / 0.93 / 0.98 | 0.81 / 0.93 / 0.98 | 0.63 / 0.79 / 0.94 | 0.63 / 0.65 / 0.59 |
| speckle noise * | 0.80 / 0.98 / 1.00 | 0.84 / 0.97 / 0.99 | 0.89 / 0.99 / 1.00 | 0.79 / 0.96 / 0.99 | 0.80 / 0.94 / 0.97 |
| gaussian blur * | 0.85 / 0.98 / 0.99 | 0.78 / 0.99 / 1.00 | 0.73 / 0.94 / 0.96 | 0.64 / 0.94 / 0.99 | 0.56 / 0.80 / 0.87 |
| spatter * | 0.57 / 0.94 / 1.00 | 0.52 / 0.92 / 0.99 | 0.54 / 0.96 / 1.00 | 0.49 / 0.87 / 0.98 | 0.52 / 0.81 / 0.89 |
| saturate * | 0.60 / 0.54 / 0.84 | 0.58 / 0.61 / 0.88 | 0.57 / 0.64 / 0.91 | 0.52 / 0.56 / 0.78 | 0.65 / 0.47 / 0.56 |

`*` marks the extra families.

## Checks

- Screen images' channel statistics vs the pilot's stored ones, largest relative difference: means_s1 0.0e+00, means_s2 0.0e+00, means_s3 0.0e+00, means_s4 0.0e+00, top_s1 0.0e+00, top_s2 0.0e+00, top_s3 0.0e+00, top_s4 0.0e+00.
- Screen AUROC of the level row: 0.841 / 0.870 (the screen: 0.841 / 0.870).
- Neighbours k = 50, key s4, scored stages s1, s2, s3.
