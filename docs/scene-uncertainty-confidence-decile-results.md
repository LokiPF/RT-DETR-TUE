# Confidence-Decile Blur Experiment

Recomputed from saved artifacts over 250 images at 6 blur levels in the `tuning` partition. No detector was run and no nearest-neighbour search was performed: every number below comes from the cached features and the saved query distances.

Provenance: feature cache `df79fddf24088e441fa867569875d775df53c0eb21b55e4bca31bad722bfbddd`, kNN result `c686df05cf6609ed7a93b3003181e6848b93f75d077c4af5382e29ef0bb3b90c`, clean bank `353a59922f4bfc9f53856f847749cf1fe267e132f1fb7bb8144ab647ac151921`, k=5, normalization `raw`. 523500 scored rows in 349 groups, 33 of which cleared the deployable gate.

## Short answer

**Which confidence range worked best?** `decile_50_60`, using `dynamic` membership, the `top20_mean` scene summary and persistence at `layer_2`, with the padded decoder queries removed. Its median per-image Spearman against blur severity is +0.6286, its adjacent non-decrease rate 0.618, its maximum-blur-above-clean rate 0.880, and it scored 1500 of the 1500 image-severity pairs it swept. It came first of 33 candidates that cleared the deployable gate -- which also means it was chosen on the same images every number in this report is measured over, so what follows is a selection and not a hypothesis test of anything.

**Did it beat confidence alone?** On the same selected queries and the same scene summary, persistence out-trends its confidence control: median Spearman +0.6286 against -0.5429, a difference of +1.1714. Image by image, it wins 176, ties 9 and loses 65 of the 250 images both sides measured -- a win rate of 0.704 over all 250, 0.730 over the 241 it decided.

**Did it beat the existing all-query benchmark?** The existing all-query benchmark reproduces from these artifacts at +0.6000 (`q90`, `layer_2`, all queries kept). The candidate's median is +0.6286, a difference of +0.0286 -- one step of the 1/35 = 0.0286 grid a median over an even number of images can land on, which is the smallest difference this metric can express. Paired image by image it out-trends the benchmark: it wins 127, ties 39 and loses 84 of the 250 images both sides measured -- a win rate of 0.508 over all 250, 0.602 over the 211 it decided.

**Does padding or bin movement explain the result?** Padding is not tested on this selection, and bin movement is not shown to explain it. `decile_50_60` keeps 0.0587 of its severity-zero membership from severity 1 onward, against 0.0526 for two unrelated memberships -- so its queries are almost entirely reselected at every blur level: whatever the trend is, it belongs to the confidence range and not to any particular queries. Holding that bin's membership fixed at severity zero scores +0.6000 instead of +0.6286 -- a difference of +0.0286, which the primary metric resolves to within 0.0286 at best. Paired image by image the same selection against its own frozen twin: it wins 127, ties 30 and loses 93 of the 250 images both sides measured -- a win rate of 0.508 over all 250, 0.577 over the 220 it decided -- rebuilding the bins at every severity does better than freezing them image by image, so on these images the movement is not costing this selection its trend. That frozen number is a diagnostic and not an alternative method, and the two are never combined into one score. The padding control is scoped to `decile_00_10` alone, so it does not test `decile_50_60` directly. Where padding can be measured -- the every-query selection -- removing the repeated decoder placeholders moves the trend from +0.6000 to +0.5429, so on this run the placeholders were adding to the old benchmark rather than to this bin. 66 of 250 images carried any padding at all.

## Best confidence range

The ranking puts `decile_50_60` (`dynamic`, `top20_mean`) first at +0.6286.
No other selection matches that median: the 2 ranked rows that hold it are this same selection at 2 scene summaries, which the caption on every table here calls views of one selection rather than independent measurements of it.
The same bin across every scene summary it was ranked at: `mean` +0.6286, `q90` +0.6000, `top20_mean` +0.6286. These are 3 views of one bin on the same images, not 3 independent confirmations of it.

Against the all-query benchmark, one row per scene summary the candidate was scored at. The difference of medians is what the design ranks on; the paired columns are what make a difference of zero readable, because a median over an even number of images can only move in steps of 0.0286. The benchmark is published at `q90` alone here, so its column below is one measurement repeated rather than three.

| candidate summary | benchmark summary | candidate | benchmark | difference | W/T/L | win rate, all paired | decided, over N |
|---|---|---|---|---|---|---|---|
| `mean` | `q90` | +0.6286 | +0.6000 | +0.0286 | 122/40/88 | 0.488 | 0.581 (210) |
| `q90` | `q90` | +0.6000 | +0.6000 | +0.0000 | 122/36/92 | 0.488 | 0.570 (214) |
| `top20_mean` | `q90` | +0.6286 | +0.6000 | +0.0286 | 127/39/84 | 0.508 | 0.602 (211) |

Across the ranked field the paired comparison against the benchmark comes out in the candidate's favour 5 times, against it 28 times, and exactly even 0 times, so it is not a procedure that favours whatever it is handed.
The clearest counter-example is `decile_40_50` at `mean`, which matches or beats the benchmark's median (+0.6000 against +0.6000) and still comes out 103 wins to 112 losses with 35 ties -- 0.479 of the 215 images it decided. At counts that close the comparison is **undecided**: it has not established the candidate, and it has not established the benchmark either.

The deployable ranking, best first. Only candidates with the padding union removed, at `layer_2`, on a membership a single image can rebuild, and at full coverage, appear in it at all.

| # | membership | bin | summary | median Spearman | adjacent non-decrease | max blur above clean | overlap vs severity 0 | coverage |
|---|---|---|---|---|---|---|---|---|
| 1 | `dynamic` | `decile_50_60` | `top20_mean` | +0.6286 | 0.618 | 0.880 | 0.0587 | 1500/1500 |
| 2 | `dynamic` | `decile_50_60` | `mean` | +0.6286 | 0.615 | 0.844 | 0.0587 | 1500/1500 |
| 3 | `dynamic` | `decile_50_60` | `q90` | +0.6000 | 0.614 | 0.884 | 0.0587 | 1500/1500 |
| 4 | `dynamic` | `decile_40_50` | `mean` | +0.6000 | 0.608 | 0.840 | 0.0593 | 1500/1500 |
| 5 | `dynamic` | `decile_30_40` | `top20_mean` | +0.6000 | 0.607 | 0.864 | 0.0623 | 1500/1500 |
| 6 | `dynamic` | `decile_30_40` | `mean` | +0.6000 | 0.606 | 0.828 | 0.0623 | 1500/1500 |
| 7 | `dynamic` | `decile_40_50` | `q90` | +0.6000 | 0.605 | 0.856 | 0.0593 | 1500/1500 |
| 8 | `dynamic` | `decile_30_40` | `q90` | +0.6000 | 0.594 | 0.852 | 0.0623 | 1500/1500 |
| 9 | `dynamic` | `decile_40_50` | `top20_mean` | +0.6000 | 0.593 | 0.864 | 0.0593 | 1500/1500 |
| 10 | `dynamic` | `decile_60_70` | `top20_mean` | +0.5714 | 0.584 | 0.816 | 0.0570 | 1500/1500 |
| 11 | `dynamic` | `decile_20_30` | `q90` | +0.5429 | 0.598 | 0.836 | 0.0609 | 1500/1500 |
| 12 | `shared` | `all_valid` | `q90` | +0.5429 | 0.596 | 0.820 | 1.0000 (arithmetic) | 1500/1500 |
| 13 | `dynamic` | `decile_20_30` | `top20_mean` | +0.5429 | 0.596 | 0.828 | 0.0609 | 1500/1500 |
| 14 | `dynamic` | `decile_60_70` | `q90` | +0.5429 | 0.583 | 0.800 | 0.0570 | 1500/1500 |
| 15 | `dynamic` | `decile_60_70` | `mean` | +0.5429 | 0.574 | 0.776 | 0.0570 | 1500/1500 |

18 further candidates are in `summary.json` under `ranked_layer_2_persistence`.

## Which decoder layer this is about

Every number above is persistence scored at `layer_2`. The same rows were scored at `combined`, `layer_0` and `layer_1` as well, and those columns are summarised but never ranked. What they show is not the same result.

The winning selection is `decile_50_60` under `dynamic` at `top20_mean`; the benchmark column is the all-query benchmark at `q90`, the summary it is published at.

| scope | winning selection | all-query benchmark | groups at this scope |
|---|---|---|---|
| `combined` | +0.4286 | +0.4286 | 70 |
| `layer_0` | -0.3143 | -0.4286 | 70 |
| `layer_1` | -0.2857 | -0.6000 | 70 |
| `layer_2` | +0.6286 | +0.6000 | 70 |

At `layer_0` and `layer_1` the winning selection's median has the opposite sign: the score there moves against blur rather than with it. So what this report establishes is about `layer_2` and not about persistence in general, and a reader carrying any of it forward is carrying a statement about one decoder layer.

`layer_2` is a constant of this analysis and not a value chosen from these rows: the deployable ranking admits it alone, so all 33 ranked candidates are at it and no candidate could be promoted here by scoring better at another layer.

## Persistence versus confidence alone

On the same selected queries and the same scene summary, persistence out-trends its confidence control: median Spearman +0.6286 against -0.5429, a difference of +1.1714. Image by image, it wins 176, ties 9 and loses 65 of the 250 images both sides measured -- a win rate of 0.704 over all 250, 0.730 over the 241 it decided.

Every pair below shares one selection and one scene summary, which is what makes the comparison fair; the two scores are in unrelated units, so only their trends are ever compared. Both denominators are given because a per-image Spearman over 6 severities lands on a coarse grid and exact ties are common: a win rate over every paired image counts each tie as a non-win, while a rate over the decided images alone hides how much of the run could not be separated.

**Read the difference column with the control's own column beside it.** Over every valid query the confidence control itself trends +0.1143 at `mean`, -0.6286 at `q90` and -0.6000 at `top20_mean`. It *falls* as blur rises at `q90` and `top20_mean` and *rises* at `mean`, so how much of a positive difference belongs to the control is a different answer at different scene summaries, and the row a reader wants is the one at the summary they are reading. Where the control is anti-correlated, a large positive difference is partly a statement about the control and only partly about persistence. The clearest case is `decile_00_10` at `q90`: it out-trends its control by +0.5143 while its own persistence median is -0.0286.

Slice: persistence at `layer_2` against its matched control, with the padding union removed, at every scene summary. Rows that differ only in the summary are views of one selection on the same images, not independent measurements of it.

| membership | bin | summary | persistence | confidence | difference | W/T/L | win rate, all paired | decided, over N |
|---|---|---|---|---|---|---|---|---|
| `dynamic` | `decile_00_10` | `mean` | -0.2000 | -0.6000 | +0.4000 | 145/12/93 | 0.580 | 0.609 (238) |
| `dynamic` | `decile_00_10` | `q90` | -0.0286 | -0.5429 | +0.5143 | 150/8/92 | 0.600 | 0.620 (242) |
| `dynamic` | `decile_00_10` | `top20_mean` | -0.0857 | -0.4857 | +0.4000 | 154/11/85 | 0.616 | 0.644 (239) |
| `frozen` (diagnostic) | `decile_00_10` | `mean` | +0.3714 | -0.6571 | +1.0286 | 223/8/19 | 0.892 | 0.921 (242) |
| `frozen` (diagnostic) | `decile_00_10` | `q90` | +0.4857 | -0.7714 | +1.2571 | 229/1/20 | 0.916 | 0.920 (249) |
| `frozen` (diagnostic) | `decile_00_10` | `top20_mean` | +0.5429 | -0.7714 | +1.3143 | 227/2/21 | 0.908 | 0.915 (248) |
| `dynamic` | `decile_10_20` | `mean` | +0.3714 | -0.6000 | +0.9714 | 178/10/62 | 0.712 | 0.742 (240) |
| `dynamic` | `decile_10_20` | `q90` | +0.4286 | -0.6000 | +1.0286 | 185/6/59 | 0.740 | 0.758 (244) |
| `dynamic` | `decile_10_20` | `top20_mean` | +0.4286 | -0.6000 | +1.0286 | 179/8/63 | 0.716 | 0.740 (242) |
| `frozen` (diagnostic) | `decile_10_20` | `mean` | +0.4857 | -0.5429 | +1.0286 | 218/6/26 | 0.872 | 0.893 (244) |
| `frozen` (diagnostic) | `decile_10_20` | `q90` | +0.6000 | -0.4286 | +1.0286 | 206/3/41 | 0.824 | 0.834 (247) |
| `frozen` (diagnostic) | `decile_10_20` | `top20_mean` | +0.6000 | -0.3714 | +0.9714 | 198/7/45 | 0.792 | 0.815 (243) |
| `dynamic` | `decile_20_30` | `mean` | +0.4857 | -0.6000 | +1.0857 | 187/6/57 | 0.748 | 0.766 (244) |
| `dynamic` | `decile_20_30` | `q90` | +0.5429 | -0.6571 | +1.2000 | 186/10/54 | 0.744 | 0.775 (240) |
| `dynamic` | `decile_20_30` | `top20_mean` | +0.5429 | -0.6000 | +1.1429 | 192/6/52 | 0.768 | 0.787 (244) |
| `frozen` (diagnostic) | `decile_20_30` | `mean` | +0.6000 | -0.6000 | +1.2000 | 211/5/34 | 0.844 | 0.861 (245) |
| `frozen` (diagnostic) | `decile_20_30` | `q90` | +0.6000 | -0.1429 | +0.7429 | 187/6/57 | 0.748 | 0.766 (244) |
| `frozen` (diagnostic) | `decile_20_30` | `top20_mean` | +0.6000 | -0.0857 | +0.6857 | 185/7/58 | 0.740 | 0.761 (243) |
| `dynamic` | `decile_30_40` | `mean` | +0.6000 | -0.5714 | +1.1714 | 181/9/60 | 0.724 | 0.751 (241) |
| `dynamic` | `decile_30_40` | `q90` | +0.6000 | -0.6000 | +1.2000 | 178/10/62 | 0.712 | 0.742 (240) |
| `dynamic` | `decile_30_40` | `top20_mean` | +0.6000 | -0.6000 | +1.2000 | 185/7/58 | 0.740 | 0.761 (243) |
| `frozen` (diagnostic) | `decile_30_40` | `mean` | +0.6000 | -0.5429 | +1.1429 | 196/8/46 | 0.784 | 0.810 (242) |
| `frozen` (diagnostic) | `decile_30_40` | `q90` | +0.6000 | +0.0571 | +0.5429 | 171/7/72 | 0.684 | 0.704 (243) |
| `frozen` (diagnostic) | `decile_30_40` | `top20_mean` | +0.6000 | +0.0857 | +0.5143 | 170/6/74 | 0.680 | 0.697 (244) |
| `dynamic` | `decile_40_50` | `mean` | +0.6000 | -0.6000 | +1.2000 | 178/8/64 | 0.712 | 0.736 (242) |
| `dynamic` | `decile_40_50` | `q90` | +0.6000 | -0.6000 | +1.2000 | 179/8/63 | 0.716 | 0.740 (242) |
| `dynamic` | `decile_40_50` | `top20_mean` | +0.6000 | -0.6000 | +1.2000 | 179/7/64 | 0.716 | 0.737 (243) |
| `frozen` (diagnostic) | `decile_40_50` | `mean` | +0.6000 | -0.5429 | +1.1429 | 185/8/57 | 0.740 | 0.764 (242) |
| `frozen` (diagnostic) | `decile_40_50` | `q90` | +0.6000 | +0.0857 | +0.5143 | 164/8/78 | 0.656 | 0.678 (242) |
| `frozen` (diagnostic) | `decile_40_50` | `top20_mean` | +0.6000 | +0.1143 | +0.4857 | 169/8/73 | 0.676 | 0.698 (242) |
| `dynamic` | `decile_50_60` | `mean` | +0.6286 | -0.5143 | +1.1429 | 177/12/61 | 0.708 | 0.744 (238) |
| `dynamic` | `decile_50_60` | `q90` | +0.6000 | -0.5429 | +1.1429 | 179/6/65 | 0.716 | 0.734 (244) |
| `dynamic` | `decile_50_60` | `top20_mean` | +0.6286 | -0.5429 | +1.1714 | 176/9/65 | 0.704 | 0.730 (241) |
| `frozen` (diagnostic) | `decile_50_60` | `mean` | +0.5429 | -0.4286 | +0.9714 | 174/12/64 | 0.696 | 0.731 (238) |
| `frozen` (diagnostic) | `decile_50_60` | `q90` | +0.6000 | +0.0857 | +0.5143 | 177/5/68 | 0.708 | 0.722 (245) |
| `frozen` (diagnostic) | `decile_50_60` | `top20_mean` | +0.6000 | +0.0857 | +0.5143 | 180/8/62 | 0.720 | 0.744 (242) |
| `dynamic` | `decile_60_70` | `mean` | +0.5429 | -0.4571 | +1.0000 | 161/7/82 | 0.644 | 0.663 (243) |
| `dynamic` | `decile_60_70` | `q90` | +0.5429 | -0.4286 | +0.9714 | 162/10/78 | 0.648 | 0.675 (240) |
| `dynamic` | `decile_60_70` | `top20_mean` | +0.5714 | -0.4286 | +1.0000 | 161/11/78 | 0.644 | 0.674 (239) |
| `frozen` (diagnostic) | `decile_60_70` | `mean` | +0.5714 | -0.1143 | +0.6857 | 151/7/92 | 0.604 | 0.621 (243) |
| `frozen` (diagnostic) | `decile_60_70` | `q90` | +0.5429 | +0.0857 | +0.4571 | 163/7/80 | 0.652 | 0.671 (243) |
| `frozen` (diagnostic) | `decile_60_70` | `top20_mean` | +0.6000 | +0.0857 | +0.5143 | 172/6/72 | 0.688 | 0.705 (244) |
| `dynamic` | `decile_70_80` | `mean` | +0.4286 | -0.3429 | +0.7714 | 149/11/90 | 0.596 | 0.623 (239) |
| `dynamic` | `decile_70_80` | `q90` | +0.4857 | -0.4000 | +0.8857 | 157/9/84 | 0.628 | 0.651 (241) |
| `dynamic` | `decile_70_80` | `top20_mean` | +0.4857 | -0.4000 | +0.8857 | 162/8/80 | 0.648 | 0.669 (242) |
| `frozen` (diagnostic) | `decile_70_80` | `mean` | +0.4857 | +0.0857 | +0.4000 | 140/12/98 | 0.560 | 0.588 (238) |
| `frozen` (diagnostic) | `decile_70_80` | `q90` | +0.4857 | +0.0857 | +0.4000 | 162/5/83 | 0.648 | 0.661 (245) |
| `frozen` (diagnostic) | `decile_70_80` | `top20_mean` | +0.5429 | +0.1143 | +0.4286 | 159/4/87 | 0.636 | 0.646 (246) |
| `dynamic` | `decile_80_90` | `mean` | +0.0000 | +0.0857 | -0.0857 | 123/6/121 | 0.492 | 0.504 (244) |
| `dynamic` | `decile_80_90` | `q90` | +0.0286 | -0.1143 | +0.1429 | 133/4/113 | 0.532 | 0.541 (246) |
| `dynamic` | `decile_80_90` | `top20_mean` | +0.1429 | -0.1429 | +0.2857 | 130/6/114 | 0.520 | 0.533 (244) |
| `frozen` (diagnostic) | `decile_80_90` | `mean` | +0.3714 | +0.3143 | +0.0571 | 118/6/126 | 0.472 | 0.484 (244) |
| `frozen` (diagnostic) | `decile_80_90` | `q90` | +0.4857 | +0.0857 | +0.4000 | 143/6/101 | 0.572 | 0.586 (244) |
| `frozen` (diagnostic) | `decile_80_90` | `top20_mean` | +0.4857 | +0.0857 | +0.4000 | 146/8/96 | 0.584 | 0.603 (242) |
| `dynamic` | `decile_90_100` | `mean` | -0.8286 | +0.8286 | -1.6571 | 26/5/219 | 0.104 | 0.106 (245) |
| `dynamic` | `decile_90_100` | `q90` | -0.7714 | +0.3143 | -1.0857 | 70/9/171 | 0.280 | 0.290 (241) |
| `dynamic` | `decile_90_100` | `top20_mean` | -0.7714 | +0.3143 | -1.0857 | 67/9/174 | 0.268 | 0.278 (241) |
| `frozen` (diagnostic) | `decile_90_100` | `mean` | -0.6571 | +0.9429 | -1.6000 | 20/1/229 | 0.080 | 0.080 (249) |
| `frozen` (diagnostic) | `decile_90_100` | `q90` | -0.4286 | +0.2286 | -0.6571 | 52/10/188 | 0.208 | 0.217 (240) |
| `frozen` (diagnostic) | `decile_90_100` | `top20_mean` | -0.5429 | +0.2571 | -0.8000 | 47/4/199 | 0.188 | 0.191 (246) |
| `shared` | `all_valid` | `mean` | +0.4857 | +0.1143 | +0.3714 | 136/3/111 | 0.544 | 0.551 (247) |
| `shared` | `all_valid` | `q90` | +0.5429 | -0.6286 | +1.1714 | 191/6/53 | 0.764 | 0.783 (244) |
| `shared` | `all_valid` | `top20_mean` | +0.4857 | -0.6000 | +1.0857 | 188/9/53 | 0.752 | 0.780 (241) |

## Dynamic versus frozen queries

`dynamic` rebuilds the ten bins from the blurred image itself, which is something a single image can do. `frozen` reuses the bins built from the clean image, which a naturally corrupted image cannot: there is no paired clean version of it. So a strong frozen result is a diagnostic -- it says how far the fingerprints moved once membership is held still -- and never a method. The two are reported side by side and are never combined into one score.

Slice: persistence at `layer_2`, with the padding union removed, at every scene summary. Rows that differ only in the summary are views of one selection on the same images, not independent measurements of it.

| bin | summary | dynamic | frozen (diagnostic) | dynamic W/T/L vs frozen | decided win rate, over N | dynamic overlap vs severity 0 |
|---|---|---|---|---|---|---|
| `decile_00_10` | `mean` | -0.2000 | +0.3714 | 50/8/192 | 0.207 (242) | 0.0647 |
| `decile_00_10` | `q90` | -0.0286 | +0.4857 | 48/11/191 | 0.201 (239) | 0.0647 |
| `decile_00_10` | `top20_mean` | -0.0857 | +0.5429 | 53/14/183 | 0.225 (236) | 0.0647 |
| `decile_10_20` | `mean` | +0.3714 | +0.4857 | 78/28/144 | 0.351 (222) | 0.0642 |
| `decile_10_20` | `q90` | +0.4286 | +0.6000 | 71/21/158 | 0.310 (229) | 0.0642 |
| `decile_10_20` | `top20_mean` | +0.4286 | +0.6000 | 73/22/155 | 0.320 (228) | 0.0642 |
| `decile_20_30` | `mean` | +0.4857 | +0.6000 | 97/30/123 | 0.441 (220) | 0.0609 |
| `decile_20_30` | `q90` | +0.5429 | +0.6000 | 92/28/130 | 0.414 (222) | 0.0609 |
| `decile_20_30` | `top20_mean` | +0.5429 | +0.6000 | 86/29/135 | 0.389 (221) | 0.0609 |
| `decile_30_40` | `mean` | +0.6000 | +0.6000 | 102/48/100 | 0.505 (202) | 0.0623 |
| `decile_30_40` | `q90` | +0.6000 | +0.6000 | 99/34/117 | 0.458 (216) | 0.0623 |
| `decile_30_40` | `top20_mean` | +0.6000 | +0.6000 | 104/26/120 | 0.464 (224) | 0.0623 |
| `decile_40_50` | `mean` | +0.6000 | +0.6000 | 122/24/104 | 0.540 (226) | 0.0593 |
| `decile_40_50` | `q90` | +0.6000 | +0.6000 | 107/36/107 | 0.500 (214) | 0.0593 |
| `decile_40_50` | `top20_mean` | +0.6000 | +0.6000 | 115/30/105 | 0.523 (220) | 0.0593 |
| `decile_50_60` | `mean` | +0.6286 | +0.5429 | 117/55/78 | 0.600 (195) | 0.0587 |
| `decile_50_60` | `q90` | +0.6000 | +0.6000 | 124/39/87 | 0.588 (211) | 0.0587 |
| `decile_50_60` | `top20_mean` | +0.6286 | +0.6000 | 127/30/93 | 0.577 (220) | 0.0587 |
| `decile_60_70` | `mean` | +0.5429 | +0.5714 | 122/31/97 | 0.557 (219) | 0.0570 |
| `decile_60_70` | `q90` | +0.5429 | +0.5429 | 107/31/112 | 0.489 (219) | 0.0570 |
| `decile_60_70` | `top20_mean` | +0.5714 | +0.6000 | 106/37/107 | 0.498 (213) | 0.0570 |
| `decile_70_80` | `mean` | +0.4286 | +0.4857 | 106/26/118 | 0.473 (224) | 0.0618 |
| `decile_70_80` | `q90` | +0.4857 | +0.4857 | 101/25/124 | 0.449 (225) | 0.0618 |
| `decile_70_80` | `top20_mean` | +0.4857 | +0.5429 | 94/32/124 | 0.431 (218) | 0.0618 |
| `decile_80_90` | `mean` | +0.0000 | +0.3714 | 72/21/157 | 0.314 (229) | 0.0779 |
| `decile_80_90` | `q90` | +0.0286 | +0.4857 | 74/13/163 | 0.312 (237) | 0.0779 |
| `decile_80_90` | `top20_mean` | +0.1429 | +0.4857 | 61/33/156 | 0.281 (217) | 0.0779 |
| `decile_90_100` | `mean` | -0.8286 | -0.6571 | 63/45/142 | 0.307 (205) | 0.2480 |
| `decile_90_100` | `q90` | -0.7714 | -0.4286 | 46/44/160 | 0.223 (206) | 0.2480 |
| `decile_90_100` | `top20_mean` | -0.7714 | -0.5429 | 51/55/144 | 0.262 (195) | 0.2480 |

The `W/T/L` column is the paired comparison the difference of medians cannot make: one selection against its own frozen twin, image by image, over the images both measured. Freezing the membership at severity zero is the only manipulation in this design that removes the movement of queries between bins and leaves everything else, so this column -- and not the gap between the two medians -- is what says whether that movement explains a bin's trend. It is a comparison and never a combination: neither column is added to, averaged with, or subtracted from the other in anything this report ranks.

**Why the paired column and not the gap between the medians.** 9 of the rows above have two medians that are *exactly equal* -- a dead heat on the statistic the design ranks by -- and the per-image comparison puts them on opposite sides of even: `decile_50_60` at `q90` has dynamic ahead 124 images to 87, a decided rate of 0.588 over the 211 it decided, while `decile_70_80` at `q90` has frozen ahead 124 images to 101, a decided rate of 0.449 over the 225 it decided. Identical on the primary metric, opposite answers image by image, which is what a difference of two medians on a 36-valued statistic cannot see.

**The shape of that column across the 10 bins.** At `top20_mean` the decided rate runs from 0.225 at `decile_00_10` to 0.577 at `decile_50_60` and back to 0.262 at `decile_90_100`, one peak with no second rise. The peak is the bin the ranking selected. Its neighbours -- `decile_40_50` at 0.523 and `decile_60_70` at 0.498 -- straddle even, so this is a short plateau of near-even bins and not one bin standing apart. 2 of the 10 bins are above even at this summary.

**And how much of that survives the other denominator.** Over the 30 rows of this slice, 7 clear 0.5 on the images the comparison decided and 1 clear it over every paired image. Quoting the first alone overstates the effect and quoting the second alone understates it, which is why both are here.

**What the frozen column says about the field the ranking chose from.** At `top20_mean` the frozen medians are identical at +0.6000 across 6 neighbouring bins, `decile_10_20` through `decile_60_70` -- a run that touches neither end of the confidence range, so a middle bin of almost any kind scores about +0.6000 here once the membership is held still.

**What the ranked field says about the selection.** The ranking is choosing among near-equals: 3 of the 11 ranked selections sit within one 1/35 = 0.0286 step of the top median (9 ranked rows, the same selections counted once per scene summary), and no other selection matches the top median exactly -- the 2 ranked rows that hold it are the winner's own scene summaries. The margin over the best of the others is one step of the 1/35 = 0.0286 grid a median over an even number of images can land on, which is the smallest difference this metric can express. Set against that, the same selection is never behind at any scene summary: it holds the top median alone at `mean` and `top20_mean`, and ties for it at `q90`.

Two unrelated decile memberships would overlap at 0.0526, so a bin sitting close to that line is being rebuilt almost from scratch at every severity. A `frozen` or `shared` row would print 1.000 in that column by construction, which is arithmetic rather than a measurement, and it is left out of the column above for that reason.

## Effect of padded queries

66 of 250 images carry a repeated decoder tail; together they contribute 6586 padded query slots, and the largest single image loses 257 of them. Of those 66 padded images, 0 have the same detected tail at all six severities -- the tail wanders with blur rather than growing, which is why one mask is taken per image and reused at every severity instead of one per severity. On the remaining 184 images there is nothing to remove, so the control is a no-op there by construction and any median taken over all 250 images is diluted by them.

Every scene summary the control was scored at. Persistence rows are at `layer_2`; the confidence control has no decoder-layer scope.

| bin | membership | signal | summary | filtered | unfiltered | difference | score changed on | decided win rate on those, over N |
|---|---|---|---|---|---|---|---|---|
| `all_valid` | `shared` | `persistence` | `q90` | +0.5429 | +0.6000 | +0.0571 | 66 | 0.820 (50) |
| `decile_00_10` | `dynamic` | `confidence` | `mean` | -0.6000 | -0.6000 | +0.0000 | 66 | 0.568 (44) |
| `decile_00_10` | `dynamic` | `confidence` | `q90` | -0.5429 | -0.4286 | +0.1143 | 64 | 0.808 (52) |
| `decile_00_10` | `dynamic` | `confidence` | `top20_mean` | -0.4857 | -0.4857 | +0.0000 | 65 | 0.738 (42) |
| `decile_00_10` | `dynamic` | `persistence` | `mean` | -0.2000 | -0.0857 | +0.1143 | 66 | 0.767 (60) |
| `decile_00_10` | `dynamic` | `persistence` | `q90` | -0.0286 | +0.1143 | +0.1429 | 64 | 0.764 (55) |
| `decile_00_10` | `dynamic` | `persistence` | `top20_mean` | -0.0857 | +0.0571 | +0.1429 | 65 | 0.772 (57) |
| `decile_00_10` | `frozen` (diagnostic) | `confidence` | `mean` | -0.6571 | -0.6571 | +0.0000 | 65 | 0.289 (38) |
| `decile_00_10` | `frozen` (diagnostic) | `confidence` | `q90` | -0.7714 | -0.7714 | +0.0000 | 65 | 0.468 (47) |
| `decile_00_10` | `frozen` (diagnostic) | `confidence` | `top20_mean` | -0.7714 | -0.7714 | +0.0000 | 65 | 0.510 (49) |
| `decile_00_10` | `frozen` (diagnostic) | `persistence` | `mean` | +0.3714 | +0.4286 | +0.0571 | 65 | 0.712 (52) |
| `decile_00_10` | `frozen` (diagnostic) | `persistence` | `q90` | +0.4857 | +0.5429 | +0.0571 | 65 | 0.696 (56) |
| `decile_00_10` | `frozen` (diagnostic) | `persistence` | `top20_mean` | +0.5429 | +0.6000 | +0.0571 | 65 | 0.768 (56) |

The `score changed on` column is a **lower bound** on the images the padding mask reached. It counts the images whose scene score moved, and a changed selection can still produce a bit-identical score because a scene summary is a many-to-one map -- the same selection pair reports different counts under different summaries, which a set of images a mask reached could not do. It must not be read as the images the padding changed.

## Metrics in plain language

- **Median per-image Spearman** is the headline. For one image, rank its six scene scores against the six blur levels and correlate the ranks; +1 means the score rose at every step, 0 means no relation, -1 means it fell throughout. The median is taken across images. Over 6 severities this can only land on 36 distinct values, spaced 0.0571 apart, so two configurations can tie here while differing on most images. That is why every comparison is also made image by image.
- **Adjacent non-decrease rate** is the share of the five steps between neighbouring blur levels on which the score did not fall. **Violation magnitude** is how far it fell when it did, as a share of that image's own score range.
- **Maximum-blur-above-clean rate** is the share of images whose worst blur level scored above their clean one -- the weakest thing a useful signal must do.
- **Coverage** is scored image-severity pairs against the pairs the sweep should have produced. A configuration that collapsed at high blur and one that rose the whole way can publish the same median, so full coverage is required rather than noted.
- **Overlap vs severity 0** is the Jaccard overlap between a bin's membership at a blur level and the same bin on the clean image, averaged over severities 1 to 5. Two unrelated memberships would score 0.0526. A 1.000 on a `frozen` or `shared` row is arithmetic, not evidence: those selections are the same query set at every severity by construction.
- **W/T/L and the two win rates.** Ties are common at this resolution, so a rate over every paired image and a rate over the decided ones alone can fall on opposite sides of 0.5. Both are always given; neither is a significance test.

**Neither score is a probability.** Persistence uncertainty is a distance from an image's decoder fingerprints to the nearest clean ones, and the confidence control is one minus the detector's largest class score. That subtraction only reverses direction -- nothing here is trained or calibrated, and a value of 0.8 does not mean an image is 80 percent corrupted. Distances and `1 - confidence` share no unit, so the two signals are only ever compared through their trends and never through their raw magnitudes.

## What this does not prove

- **Selecting and reporting on the same images.** Any candidate named above was chosen best of 33 ranked candidates on the same images the numbers describe. Its counts are conditioned on that choice and are not a hypothesis test of anything.
- **Three scene summaries of one bin are one result.** They are computed from the same selections on the same images; agreement between them is arithmetic, not replication.
- **The ranking did not choose the padding rule.** Every candidate it admits has the padding union removed, so the choice between removing and keeping the decoder placeholders is not settled by the ranking. The sensitivity table above is the evidence for that half of the decision, and on a run where the unfiltered rows trend higher, what that shows is that the placeholders carry a blur signal of their own -- not that the method should keep them.
- **A frozen result is not a method.** It needs a paired clean image, which a naturally corrupted one does not have. Frozen rows are excluded from the ranking by design and are never added to or averaged with dynamic ones.
- **`score changed on` is a lower bound**, not the set of images the padding mask reached.
- **An undecided comparison is not a loss.** A pair that comes out close to even, with many ties, has not established anything in either direction.

## Next decision

The tuning run is allowed to fix at most one confidence bin, one membership rule, one scene summary and one padding rule. On this run that is `decile_50_60` / `dynamic` / `top20_mean` for the first three; the fourth is not decided by the ranking and has to be argued from the sensitivity table.

That choice is a proposal to be reviewed before anything is scored on the held-out images. Until that review and that run happen, every number in this report describes the images it was selected on.

The held-out test images were not used.
