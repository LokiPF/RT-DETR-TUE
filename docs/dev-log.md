# Dev log

Dated observations and decisions that don't belong in a results document. Newest first.

## 2026-10-02: The 5,000-image confirmation passes

Both pre-registered decisions are confirmed. The details are in `docs/conv-tu-conditioned-results.md`.
- **The headline, the two-axis score, on all 5,000 images:** AUROC 0.917 common and 0.858 extra.
  - It beats the activation CDFs by +0.096 [+0.093, +0.100] and +0.051 [+0.048, +0.054].
  - It beats the level score by +0.076 [+0.073, +0.079] on the common families.
  - The 3,030 untouched images give the same result: 0.916 / 0.858.
- **The level score, on the 4,800 held-out images:** it beats the global average by +0.041 common and +0.037 extra, and the CDFs by +0.020 and +0.059. Every interval is above 0.
- **The cost of the flatter arm:** on the extra families, the two-axis score is 0.008 [0.005, 0.010] below the level score.
- **The equivalence reference:** `docs/results/conv-tu-conditioned/summary.json` is what the clean branch must reproduce.

## 2026-10-01 (night): Corruptions flatten or shift the early channels

**Source.** The detection-only roundtable proposed this; all five panelists gave it their first vote. The record is in `docs/roundtable-2026-10-01-detection/`. I re-implemented the two leading rows from their written spec (`verify_panel_rows.py` in that folder) and reproduced the panel's point estimates exactly.

**Two numbers per channel, at stages 1–3:**
- **Level** m: the channel's mean |activation| over all positions, as in the previous entry.
- **Peak share** π = log(t + 1e-6) − log(m + 1e-6), where t is the mean of the channel's strongest 1% of positions. A high π means the pattern appears in a few places, strongly. A low π means the responses are flat across the image.

Both are judged against the 50 clean bank images nearest in standardised stage-4 channel means, the content key of the previous entry.

**The two leading rows:**
- **AC, the shape row** (frozen in round 1):
  - per stage, take the mean over channels of |π − mean_N π| / sd_bank(π);
  - z-score each stage with the 500 z-statistics images;
  - sum over stages 1–3.
- **M1, the two-axis row** (frozen in round 2): the larger of two arms, each re-z-scored on the z-statistics images.
  - The signed "flatter" arm is the mean over channels of −(π − mean_N π)/sd.
  - The level arm is the previous entry's score.

**The panel's account of why it works** (after the confrontation; measured on the screen or the clean bank):
- **Structure removal lowers the peaks.** Fog, contrast and the blurs lower a channel's strongest responses much more than its mean. At severity 1, an s1 channel's log ceiling drops 4–14 times more than its log level (fog: −0.280 vs −0.061). 77–84% of s1 channels flatten, against 52–53% under noise.
- **The flattening is shared across channels.** The direction in which all channels move together holds 12% of the clean variation, but 40–59% of the shift under fog, contrast and the blurs. A signed average over channels adds it up.
- **Noise, snow and spatter add responses instead.** They mainly move the level.
- **Content still matters for the peak share.** The stage-4 key predicts 60–69% of its clean variation (69–77% for the level), so the similar-scene reference still helps.

**Numbers** on positions 1059–1969 (911 images, read only after both rows were frozen):

| Row | AUROC common ↑ | AUROC extra ↑ | Severity 1, common ↑ |
|---|---|---|---|
| M1 (two axes) | **0.922** | 0.861 | **0.869** |
| AC (peak share) | 0.901 | 0.862 | 0.818 |
| Level vs similar scenes (previous entry) | 0.842 | **0.868** | 0.763 |
| Activation CDFs (Becker et al., ICPR 2026) | 0.819 | 0.808 | 0.705 |

- **AC also held** on the 859 images before them (positions 200–1058): 0.893 / 0.858.
- **Severity-1 fog / contrast:** M1 0.959 / 0.947, CDFs 0.633 / 0.661.
- **FPR95, common families:** M1 0.216, CDFs 0.417.
- **The panel's bootstrap intervals** (not re-run by me):
  - AC − CDFs: +0.082 [+0.075, +0.089] common, +0.053 [+0.048, +0.059] extra;
  - M1 − AC: +0.021 [+0.015, +0.027] common, −0.001 [−0.007, +0.004] extra.

**Ruled out by the panel:**
- topology: the Euler characteristic adds +0.004 [−0.016, +0.022] over the means;
- whitening the peak shares: −0.034 held out, because it divides out the shared flattening;
- comparing small image cells: −0.112;
- other content keys: no gain;
- NAP's published form: 0.302.

**Weak spots.**
- Brightness (about 0.53 at severity 1), saturate and spatter stay weak for every row. The detector is trained to ignore photometric changes (L10 of the briefing).
- M1 loses there against the level row: brightness −0.075, saturate −0.057.

**Real fog is unproven.**
- **Synthetic test:** 24 Cityscapes frames with fog synthesised from a guessed depth map, each AUROC about ±0.08. The plain level tied the new rows.
- **Bank domain:** with the COCO bank, every content-conditioned row fell to 0.54–0.80. So the clean bank must come from the deployment camera.
- **The deciding test** is Foggy Cityscapes: clean and foggy versions of the same scenes, with a Cityscapes clean bank.

**Data caveat.** For their checks, the panelists read positions 200–1969 of the running 5,000-image pass.
- Positions 1970–4999 (about 3,030 images) are unread by anyone, so they are the clean confirmation set for these rows.
- The pre-registered decision on the level row is unaffected, because its rule was fixed beforehand. It will also be reported on 1970–4999.

**Decided (user, 19:30, before 1970–4999 were read).**
- **The headline:** M1, the two-axis score.
- **The images:** all 5,000 val images, so that it is comparable with the baselines.
- **The check:** the 3,030 untouched images. The rule is in the plan's amendment 2. AC is the ablation.
- **The clean bank stays COCO train only:** 2,000 images, plus 500 for the z-statistics, the same source as the baselines' references.

**Still open (user):**
- the downloads of Foggy Cityscapes and ACDC;
- the framing, "corruptions flatten or shift the early channels".

**Credits:**
- A: the mechanism and the row;
- C: the same row, found independently, and the negative result for topology;
- D: the two-axis framing and the driving plan;
- E: the level arm and the same-scene oracle;
- B: the ratio-artefact critique and the whitened baseline.

## 2026-10-01 (evening): The back of the detector as a content key

**Goal (user).** "I want to detect the corruption of the images, not how the performance will drop."
- From now on, separation (AUROC) is the criterion. Harm numbers are context at most.
- `docs/driving-benchmark-baselines-and-metrics.md` still lists harm as goal (b).

**Why plain channel means are not enough.**
- **Already published.** Comparing activation means with the training means is Neural Mean Discrepancy (Dong et al., CVPR 2022, arXiv 2104.11408). It takes the training means from the BatchNorm layers as a "free lunch", and reports that the means of out-of-distribution inputs deviate more.
- **On clean images, the early stages' channel means vary mostly with scene content.** A ridge map from the stage-4 means predicts 73%, 77% and 82% of that variation at stages 1, 2 and 3. That is held-out R² on the 500 clean z-statistics images, with the map fitted on the 2,000-image bank.
- That content variation is what hides mild corruptions.

**Idea.** The back of the detector barely reacts to corruption: the pilot's stage-4 channel means separate at only 0.535 AUROC on the common families. But they still describe the scene, so use them as a content key.
1. **Find the neighbours.** For each image, find the k = 50 clean bank images nearest to it in stage-4 channel means. Each channel is standardised with the bank's mean and spread, and distance is Euclidean.
2. **Score each early stage.** At stages 1–3, the score is the mean over channels of |v − μ_nb| / σ:
   - v is the image's channel mean;
   - μ_nb is the mean of the neighbours;
   - σ is the bank's spread for that channel.
3. **Combine the stages.** Z-score each stage's score with the 500 clean z-statistics images, scored the same way with neighbours from the bank, and add the three.

**Screen.** CPU only, on the stored channel means of the pilot's 200 images × 96 conditions. Brackets are 95% paired bootstrap intervals (1,000 draws, seed 44). They reproduce the pilot report's intervals for the rows they share.

| Method | AUROC common ↑ | AUROC extra ↑ | Severity 1, common ↑ |
|---|---|---|---|
| Stages 1–3 vs the 50 most similar clean scenes | **0.841** [0.824, 0.858] | **0.870** [0.855, 0.884] | **0.763** |
| Stages 1–3 vs the average of all clean images | 0.811 [0.794, 0.826] | 0.837 [0.823, 0.851] | 0.713 |
| Channel means, kNN (the pilot's control, all 4 stages) | 0.798 [0.780, 0.816] | 0.847 [0.829, 0.864] | 0.719 |
| Activation CDFs (Becker et al., ICPR 2026) | 0.825 [0.807, 0.845] | 0.808 [0.791, 0.824] | – |
| DisCoPatch | 0.760 [0.743, 0.779] | 0.747 [0.731, 0.765] | – |

Differences, the content-conditioned row minus each other row. A positive Δ means the conditioned row is better:

| Other row | Δ AUROC common ↑ | Δ AUROC extra ↑ |
|---|---|---|
| Stages 1–3 vs the average of all clean images | +0.031 [+0.018, +0.045] | +0.033 [+0.022, +0.044] |
| Channel means, kNN | +0.043 [+0.033, +0.053] | +0.023 [+0.015, +0.031] |
| Activation CDFs | +0.016 [−0.004, +0.035] | +0.062 [+0.045, +0.078] |
| DisCoPatch | +0.081 [+0.060, +0.102] | +0.123 [+0.102, +0.142] |

**Reading.**
- **Conditioning on the scene helps most where content hides the corruption.** By severity 1–5 on the common families, the conditioned score reaches 0.763 / 0.816 / 0.850 / 0.879 / 0.900. The average of all clean images reaches 0.713 / 0.777 / 0.820 / 0.859 / 0.884. So the gain is +0.050 at severity 1 and +0.016 at severity 5.
- **The number of neighbours hardly matters.** k = 10, 50 and 200 give 0.836, 0.841 and 0.840 on the common families.
- **The neighbourhood itself matters, not just regression on the key.**
  - A linear version does worse: 0.789 common and 0.811 extra. It predicts stages 1–3 from stage 4 with the same ridge map and standardises the residual by the bank's residual spread.
  - A guess at the reason, untested: a linear map carries the key's own small corruption shift into its prediction, while the neighbours are always clean images.

**Candidate thesis.** The detector's own invariance gives a free content anchor.
- The back says *what* the scene is, and the front says *how it looks*.
- A corruption is the front disagreeing with clean scenes like this one.

It keeps the three lessons of this log: the front, not the back, sees the corruption; position is ignored; channel identity is kept, with no sorting.

**Caveats.**
- **A 200-image screen.** The three values of k were tried on the same images.
- **Novelty not checked.** The nearest relative known so far is Lee et al.'s (2018) class-conditional Mahalanobis distance, which conditions on the predicted class within one layer.
- **The key still moves a little under corruption.** Stage 4 separates at 0.535 (common) and 0.594 (extra). A heavy corruption could change which clean scenes are picked.

**Next (not yet run).**
- **CPU, on stored data:**
  - per-family results;
  - the neighbours' own spread instead of the bank's;
  - a key from stages 3 + 4;
  - adding stage 4's own unconditioned score;
  - how often an image's neighbours change between its clean and corrupted versions.
- **GPU** (each needs a written plan and an agreed window first):
  - channel means for all 5,000 images × 96 conditions, to confirm against the 5,000-image baselines;
  - the CDF monitor with the same neighbourhood reference;
  - the driving data.
- **Literature:** check conditional and cross-layer references for OOD and corruption detection.

**Code.** The screen ran from scratch scripts outside the repository. The 5,000-image confirmation reimplements the score in the package; its plan is `docs/superpowers/plans/2026-10-01-content-conditioned-confirmation.md`.

**Note.** A first paper-idea roundtable, run the same afternoon, was briefed with harm as its primary metric. It answered the wrong question, and was discarded at the user's request. A second roundtable on detection only is running.

## 2026-10-01 (later): Own training average vs kNN on channel means

**Answer** (details in `docs/conv-tu-pilot-results.md`, section "Follow-up: channel statistics"):
- **Separation is about the same as kNN.** Comparing each channel with its own clean average gives 0.794 against 0.798 AUROC on the common families, and 0.818 against 0.847 on the extra ones.
- **Ranking conditions by harm is much better:** ρ(score, LRP) across conditions is 0.68 against 0.50, the same as the full-CDF monitor.
- **One mean per channel comes close to the full CDF.** It gets within 0.03 AUROC of the full per-channel CDF monitor, and tracks within-condition harm slightly better.
- **Tails and position don't help on COCO.** Per-channel tails (top-1% means, 99th percentiles) and a 4 × 4 grid are all worse.

**What it means for the next step:**
- The separation signal lives in per-channel activation levels at stages 1–3, and a single mean per channel captures most of it.
- Every activation monitor tracks harm weakly within a condition: at most 0.13 (Hashemi et al. on the decoder queries), against 0.26 for SAOD min and 0.34 for ContrastiveConf.
- So a combination of a channel-level shift score with an output-based harm score is the natural next thing to try. A coarse grid stays worth testing on the fixed-camera driving data.

## 2026-10-01: Channel identity, not edge identity

**Observation from the conv-TU pilot** (`docs/conv-tu-pilot-results.md`). Three of the four representations are sorted top-K lists: the diagram's top 1%, the heaviest edges, and the largest activations. All three separate at about 0.66 AUROC. The unsorted channel means (mean |x| per input channel, in a fixed channel order) reach 0.798 on the common families and 0.847 on the extra ones. The likely reason: a sorted top-K list mixes channels and positions and sees only the extreme tail of the map. The channel means keep which channel is which and average over the whole map, and noise or blur raise or lower specific channels.

**Proposal (user).** Carry the identity idea over to the edges. Find the edges that are heavy on clean training images, read exactly those edges at inference, unsorted and in the training-defined order, and compare them with the clean bank.

**Why a fixed edge list brings back the wrong identity.**
- Each edge weight is |K_eff[d,c,t]| · |x[c,p]|. For a fixed edge list, the kernel factor of each entry is a constant, so the vector is just the activations at a fixed set of (channel, position) cells, each times a constant. The kernel and the graph only rescale it, so it is a weighted crop of the activation map.
- On COCO, a fixed cell holds different content in every image, a dog's ear in one and sky in another. Two clean images already differ a lot at the same cell, so the content variation would drown out the corruption signal. The edges that are heavy on average come mostly from channels that respond everywhere, plus the image border.
- Grouping edges by connection and summing over positions gives |K| × the channel mean. The maximum over positions gives |K| × the per-channel maximum, which is Zheng et al.'s summary. So identity helps at the level of the channel, not the position.

**Variants worth testing** (not yet run; each needs a short plan and about 30 minutes of GPU):
1. **The fixed heavy-edge list as proposed:** the edges heaviest on average over clean training images, which is |K| × the clean mean activation map.
2. **Per-channel heaviest values in a fixed channel order:** the mean of each channel's top 1% activations, or its 99th percentile. This keeps the focus on strong responses and drops the position.
3. **Channel means on a coarse grid** such as 4 × 4. Position could matter on the driving benchmark, where a fixed camera puts the road at the bottom and the sky at the top, and fog hits the distant upper part harder.

**Related.** The ICPR 2026 monitor (Becker et al.) already works per channel. It builds the full distribution of each channel's activations over all positions, as a CDF, and compares it with the same channel's training CDF. The channel mean is one summary of that distribution, and a per-channel top 1% is another.
