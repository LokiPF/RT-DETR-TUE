# Dev log

Dated observations and decisions that don't belong in a results document. Newest first.

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
