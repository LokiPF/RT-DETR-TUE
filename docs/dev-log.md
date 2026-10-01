# Dev log

Dated observations and decisions that don't belong in a results document. Newest first.

## 2026-10-01 (evening): The back of the detector as a content key

**Goal (user).** "I want to detect the corruption of the images, not how the performance will drop."
- From now on, separation (AUROC) is the criterion. Harm numbers are context at most.
- `docs/driving-benchmark-baselines-and-metrics.md` still lists harm as goal (b).

**Why plain channel means are not enough.**
- **Already published.** Comparing each channel's mean with its training mean is Neural Mean Discrepancy (Dong et al., CVPR 2022), which takes the training means from the BatchNorm layers. This was verified during the paper-idea roundtable (`docs/roundtable-2026-10-01/record.md`).
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

**Scripts.** `content_conditioned.py` and `content_conditioned_bootstrap.py`, kept with the roundtable record in `docs/roundtable-2026-10-01/scripts/`. They read the tables stacked by `build_tables.py` in the same folder.

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
