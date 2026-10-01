# Content-conditioned reference: 5,000-image confirmation

**Goal:** confirm on all 5,000 COCO val images the screen recorded in `docs/dev-log.md` (2026-10-01, evening): stages 1–3 channel means judged against the clean scenes nearest in stage-4 channel means. All choices are fixed below, before any held-out number exists.

**Spec:** `docs/dev-log.md`, entry "The back of the detector as a content key". The user's goal is detecting corruption, so separation is the criterion; harm is not reported.

**Authorization:** the user asked for this run on 1 October ("you probably can run the 5000 image confirmation") while the detection roundtable runs.

## Global constraints

- **GPU.** The card is shared. Explore's training peaks at 31.5 of 32.6 GB, so nothing may run beside it. The means pass runs only in explore's gaps, when explore uses 2.6 GiB: about 16:35–18:30, 22:00–23:50 and 03:25–05:30, and freely after about 05:30.
  - Each part runs for at most 85 minutes (`timeout`).
  - A watchdog stops the part at once if the card's total compute memory exceeds 6 GiB.
  - Explore pings at the start of each gap. The pass caps itself at 1.5 GiB (`MEANS_GPU_CAP_GIB`).
- **Same reference as the screen:** the pilot's 2,000-image clean bank and its 500 z-statistics images (`runs/coco-baselines/convtu/channels_bank.npz`, `channels_zstats.npz`). Nothing is refitted.
- Tests run with `CUDA_VISIBLE_DEVICES=`.

## Amendment (16:35, before the pass started)

The pass also stores each channel's top-1% mean (`top_s1..s4`, from `channel_statistics`, so it is identical to the pilot's).
- **Why:** roundtable panelist C asked for it. C's candidate applies the same conditioning to log(top-1% mean / mean). On the 200 screen images it reached 0.897 common / 0.859 extra, with the statistic picked among about 7 tried there.
- **What it does not change:** the decision below, which concerns the means score only. Any result for C's statistic is reported separately, as a second candidate whose choices were fixed on the 200 screen images.
- **Cost:** no extra pass, about 1.8 GB more on disk.

## Amendment 2 (19:30, before positions 1970 and later were read): the headline

The user chose the two-axis score as the paper's headline. It is the "flatten or shift" row of the dev log, 2026-10-01 (night). It is evaluated on all 5,000 images, so that it stays comparable with the six baselines.

- **Headline row** (`conditioned.two_axis_scores`): the larger of two arms, each re-z-scored on the z-statistics images.
  - The flatter arm averages −(peak share − neighbours' peak share)/sd over the channels of s1–s3.
  - The level arm is the level score below.
  - The bank, key, k = 50 and stages are the same as below.
  - The peak share alone (`conditioned.peak_share_scores`) is the ablation row.
- **Images:**
  - all 5,000 images, for the headline;
  - positions 1970–4999, as the check: 3,030 images that nobody read while the method was designed. The roundtable read positions 200–1969 of the running pass.
- **Headline rule** (`confirmation.headline_decision`): confirmed when, on both image sets, every 95% interval of these differences excludes 0:
  - two-axis minus the activation CDFs, on AUROC common and on AUROC extra;
  - two-axis minus the level score, on AUROC common.
- **The level score's pre-registered rule below is unchanged.** It is still reported on the 4,800 held-out images.

## Pre-registered method (exactly the screen)

1. **Key.** Stage-4 channel means, each channel standardised with the bank's mean and population spread; Euclidean distance; the k = 50 nearest bank images.
2. **Score each of stages s1, s2 and s3:**
   - take the mean over channels of |v − μ_nb| / σ;
   - μ_nb is the neighbours' mean;
   - σ is the bank's per-channel population spread, floored as in `channels.fit_own_average` (no dimension was floored in the pilot).
3. **Combine.** Z-score each stage with the 500 z-statistics images, scored the same way with neighbours from the bank, and add the three.

## Rows reported

- **The conditioned score:** the method above.
- **Stages 1–3 vs the global clean average:** the same as the conditioned score, with μ_nb replaced by the bank's mean. This isolates the conditioning.
- **Channel means, kNN:** the pilot's control, all 4 stages, mean Euclidean distance to the 5 nearest bank rows, z-scored with the z-statistics images and summed.
- **Channel means vs own average, all 4 stages:** the follow-up's row.
- **Baselines** from the stored 5,000-image scores:
  - activation CDFs (headline, z-scored stages);
  - DisCoPatch;
  - SAOD min;
  - kNN (k = 100);
  - Hashemi et al. (decoder).

## Evaluation

- **Images:**
  - **Primary:** the 4,800 images not used by the screen, positions 200–4,999 of the seed-44 order.
  - **Also:** all 5,000 images, and the 200 screen images.
  - **Reproduction check:** the screen images must reproduce the screen's 0.841 / 0.870 to within 0.003. Their stage means must equal the pilot's stored means to within 1e-4 relative, on the same GPU settings.
- **Metrics:**
  - mean AUROC over the 75 common and the 20 extra conditions;
  - AUPR and FPR95;
  - AUROC by severity, and by family at severities 3 and 5.
- **Intervals:** 95% paired bootstrap over images, with 1,000 draws and seed 44.

## Decision rule (on the 4,800 held-out images)

- **Confirmed:** both of the following hold, each with a 95% interval excluding 0 on both AUROC common and AUROC extra:
  - the conditioned score minus the activation CDFs is above 0;
  - the conditioned score minus stages 1–3 vs the global average is above 0.
- **Conditioning confirmed, not ahead of the CDFs:** only the second condition holds.
- **Not confirmed:** the second condition fails on either group.

## Tasks

1. **The `convtu-means` phase** (done, `bff2bef`): `means_s1..s4` for the 96 conditions of every evaluation image. It is resumable per image and checked against the detector pass's corruption digests. The tests cover:
   - coverage beyond the pilot images;
   - resume;
   - exact equality with the channel-statistics means;
   - changed corruptions.
2. **Run it in explore's gaps:** `run_means.sh` in the session scratchpad, with the 85-minute timeout and the 6 GiB watchdog.
3. **`differential_uncertainty/convtu/conditioned.py`**, test first:
   - `nearest_rows(queries, reference, k)` (exact, chunked);
   - `conditioned_scores(test, bank, zstats, k)`;
   - `global_scores(test, bank, zstats)`.

   The tests cover:
   - agreement with a brute-force neighbour search;
   - a synthetic bank with two content clusters, where a clean image of the second cluster scores high against the global average but low against its neighbours;
   - a shifted early stage scoring high either way.
4. **The `convtu-conditioned-report` phase:** the rows, metrics, intervals, decision and reproduction checks above, written to `runs/coco-baselines/results_convtu_conditioned/`. Its test runs on the fake backbone.
5. **Results:** `docs/conv-tu-conditioned-results.md` with the tables in `docs/results/conv-tu-conditioned/`, a dev-log line, then commit.
