# Detection roundtable record, 1 October 2026

## A (Mechanism): Shape, not level: corruption breaks the level-peakiness trade-off of a frozen detector's early channels

**Thesis:** Scenes change how much of the image a detector's early filters fire on, while corruptions change how sharply they fire. A training-free corruption detector should therefore read each early channel's peak-to-mean ratio against content-matched clean images, not its mean.

### Round 1 proposal

## Thesis
Scenes change how much of the image a detector's early filters fire on; corruptions change how sharply they fire. So read each early channel's peak-to-mean ratio, not its mean.

## Insight
**Mechanism.** In clean photos, each early channel's ceiling (the mean of its top 1% of positions) is nearly fixed by exposure and optics. Its clean spread is only 0.54–0.71 of the level's spread at s1–s3. Content mainly decides how much of the frame is textured.
- So across clean images, level and peak-to-mean ratio are strongly anti-correlated: the median per-channel r is −0.74 / −0.85 / −0.88 at s1 / s2 / s3. A scene gives either many weak responses or a few strong ones.
- Gain and blur corruptions break this trade-off by lowering the ceiling itself.
  - Blur spreads an edge's response: the peak falls while the response integrated across the edge is conserved.
  - Contrast and fog shrink the structured part against fixed offsets: the preserved mean intensity and the frozen BatchNorm shifts. Fog follows Koschmieder's I = J·t + A(1−t), and the package's fog has the same form.
- At severity 1, 35–52% of s1 channels lose level and peakiness together under fog, contrast and every blur. A move with the clean correlation would give about 12%. Noise, snow, spatter and brightness stay at 4–8%.

**Why it is not obvious.**
- L4, L5 and L8 point the other way. The tail alone loses to the bulk (top-1% 0.792 vs means 0.811), and "magnitude must be kept" for fog and contrast. Yet the scale-free ratio catches contrast best (0.941), because a network with offsets turns a gain into a shape change.
- It also explains why Section 4 fails on exactly these families (fog 0.546, contrast 0.681, zoom blur 0.700).
  - Fog also lowers the stage-4 key, so the neighbour search recruits low-level clean scenes, and the level shift is absorbed. The s1 score d′ at severity 1 is −0.06 for fog, +0.03 for contrast, +0.14 for zoom.
  - The same neighbours are peakier, so the shape deviation grows instead: d′ +1.90 / +2.19 / +1.85.
- I first guessed that the shape moves more, or is less shared across channels. Both are refuted: the RMS shifts are equal at s2–s3, and the effective dimension is the same (13.0 vs 13.2 at s1).

**Supported by:** L1 (front stages), L2 (position-free), L3 (channel identity kept; only exchangeable positions are sorted), L9 (content is the nuisance), and L10, which explains the remaining cap: brightness and saturate barely move either axis.

**Could kill it:** L11 (these are the same 200 images that produced Section 4), and a key that still moves with the corruption.

## Method
1. **Taps.** The post-ReLU inputs of `res_layers[s].blocks[1].branch2a.conv`, stages s1–s4. The hooks already exist.
2. **Per channel k at s1–s3:**
   - level m = mean over positions of a(u);
   - ceiling t = mean of the largest 1% of a(u);
   - peakiness π = log(t + ε) − log(m + ε), with ε = 1e-6.
3. **Content key.** Stage-4 channel means, standardised on the bank. N(x) is the set of the 50 nearest of 2,000 clean COCO-train images.
4. **Stage deviation.** D_s(x) = mean_k |π_{s,k}(x) − mean_{j∈N(x)} π_{s,k}(j)| / σ_{s,k}, where σ is the bank's per-channel spread of π.
5. **Score.** S(x) = Σ_s (D_s − μ_s)/σ_s, with μ_s and σ_s from 500 other clean images scored the same way.

The method uses clean images only, trains nothing and needs no labels. The only change from Section 4 is the statistic: mean → peak-to-mean ratio.

## Novelty
- **Becker et al. (ICPR 2026):** EMD between absolute-unit CDFs and a global training CDF, so the level dominates. Planned ablation: their EMD on a/m.
- **NMD:** the first moment, which is exactly the level we discard.
- **Mahalanobis and Gram:** pooled means or cross-channel products against training statistics. Neither reads the spatial shape within a channel.
- **kNN-OOD:** the pooled penultimate vector.
- **DisCoPatch:** a trained discriminator.
- **Section 4:** the same reference, with the statistic that content absorbs.
- **ASH (ICLR 2023) and SCALE (ICLR 2024)**, both checked: the same algebra, Σa / Σ_top-p a, but taken across the channels of the pooled penultimate vector, to rescale logits for semantic OOD. Ours is taken across the positions of each early channel, compared with clean content-matched images, for covariate corruption.
- **Ancestors:** BRISQUE (Mittal, Moorthy & Bovik, TIP 2012) and NIQE use the distribution shape of hand-crafted normalised-luminance coefficients against pristine statistics. MDFS (Ni et al., TMM 2024) uses the mean and std of deep features with a pristine Gaussian, scored against human ratings (MOS). Deep Channel Prior (arXiv 2404.01703) uses channel-correlation matrices for feature enhancement, not detection.
- **Our claim:** natural-scene statistics inside the detector, with its own filters, one shape number per channel, and a content key from its own deep stage.
- **Recollection, unverified:** natural images have heavy-tailed, sparse filter responses (Field 1987; Olshausen & Field 1996).

## Decisive experiment
Score 5,000 images × 96 conditions with the pre-registered row exactly as above, and report the 4,800 held-out images separately.

**Paper-making result:**
- AUROC common ≥ 0.87 and extra ≥ 0.83, with paired 95% intervals against the activation CDFs (0.821 / 0.807) excluding 0;
- severity-1 common ≥ 0.77 (CDFs 0.707);
- FPR95 common ≤ 0.30 (CDFs 0.413);
- fog, contrast and zoom blur at severity 1 each at least 0.08 above the CDFs.

**Secondary row** (exploratory origin disclosed): the level score plus a signed "flatter than neighbours" term.

**Kill criteria:**
- **K1:** the difference against the CDFs on common has an interval containing 0.
- **K2:** the difference against the level row (Section 4, same images) does not exclude 0 on common. The thesis dies and Section 4 stands.
- **K3:** the average improves, but the gain does not sit in fog, contrast and the blurs, or the clean anti-correlation is weaker than −0.5. Then the mechanism is wrong.

## Cheapest first test
Run on the stored pilot (200 × 96). This is a screen, not proof. No parameter was tuned, but I chose the statistic after seeing Section 4 fail on fog on these same images.

| Row | AUROC common ↑ | AUROC extra ↑ | FPR95 common / extra ↓ |
|---|---|---|---|
| Peakiness, conditioned (the row) | **0.897** [0.881, 0.911] | **0.859** [0.844, 0.871] | **0.253** / 0.352 |
| Section 4 (level, conditioned) | 0.841 | 0.870 | 0.368 / 0.317 |
| Activation CDFs | 0.825 | 0.808 | 0.403 / 0.428 |
| Peakiness, unconditioned | 0.849 | 0.805 | – |
| Peakiness from p99 instead of top-1% | 0.886 | 0.853 | – |
| Oracle: clean version's neighbours | 0.927 | 0.879 | – |

- **Severities 1–5 (common):** 0.814 / 0.876 / 0.912 / 0.935 / 0.949.
- **Against the CDFs:** +0.072 [+0.056, +0.087] common, +0.051 [+0.038, +0.063] extra, +0.098 [+0.079, +0.115] at severity 1.
- **Against Section 4:** +0.056 [+0.044, +0.069] common, but −0.011 [−0.022, −0.000] extra. Saturate, spatter, speckle and snow favour the level.
- **Per family (mean AUROC):** fog 0.875 (Section 4 0.546, CDFs 0.725), contrast 0.941 (0.681, 0.851), zoom blur 0.905 (0.700, 0.830).
- **Still weak:** brightness 0.660 and saturate 0.705.

## Figure 1
- **Left:** one s1 channel. 2,000 clean images in the (log level, log peakiness) plane form a descending band (r ≈ −0.74). Arrows show one image's 19 corruptions at severity 3: blurs, fog, contrast and pixelate leave the band down-left; noise slides far along it; brightness and saturate barely move.
- **Right:** the same over all channels, as a "corruption compass" of family-mean shifts.
- **Caption:** clean scenes trade level for peakiness; corruptions break the trade.

## Fit for IEEE IV
Fog is IV's canonical degradation, and Koschmieder fog is a textbook lowering of the ceiling.
- **Pre-registered prediction:** Foggy Cityscapes and ACDC fog separate on shape, while monitors that read the level struggle.
- Night combines noise and low contrast, so both axes move. Lens soiling and droplets act as local blur.

**Driving-data plan:**
- Cityscapes-C from Cityscapes val, with the same seeds; the bank and z-statistics images come from train.
- Foggy Cityscapes and ACDC, downloaded. ACDC's normal-condition reference images are an unverified recollection.
- nuImages, if its scene tags allow.

The detector stays frozen. The statistic is position-free, so the fixed-camera caveat (L2) does not apply. The added cost is one top-k per channel, expected well under 1 ms (L12 measured 0.2–1.2 ms for all monitors).

## Risks and cost
- **GPU:** one pass of 5,000 × 96 saving means and top-1% at s1–s4, about 2.5–3 h and limited by corruption generation on the CPU. The bank and z-statistics images already store both statistics. The pass can be shared with the running confirmation if `top` is added to it. Storage is about 3.7 GB.
- **New code:** about 40 lines.
- **What could go wrong:**
  - optimism from the screen (L11);
  - the loss on the extra families (fallback: level + shape, 0.890 / 0.874);
  - the L10 cap on brightness and saturate;
  - a residual 0.03 gap to the oracle, because the key still moves with the corruption.

**Quick check:** All checks ran on CPU on the stored pilot (200 images × 96 conditions) against the 2,000-image bank and the 500 z-statistics images. Scripts are in /tmp/claude-1000/-home-yuchen-YuchenZ-UE-philip-sa/3d52b916-b3ff-4806-a501-7f70cde31a29/scratchpad/detection/roundtable/panelist-A/: coherence.py, explain_away.py, shape.py, shape_confirm.py, shape_direction.py, shape_metrics.py, shift_size.py, why_shape.py and tradeoff.py.

Caveats: this is a screen, not proof. No parameter was tuned (top-1%, k = 50 and the stage-4 key are inherited). I chose the shape statistic after seeing Section 4 fail on fog and contrast on these same images.

(1) Reproduction. Section 4 gives 0.841 common / 0.870 extra, as in the briefing.

(2) Negative: level coherence. The common-mode share of the paired level shift is 0.01–0.14, against 0.02 for clean residuals. A log-gain 3-vector reaches only 0.792 / 0.790.

(3) Explaining away. At severity 3, only 14–72% of an image's neighbours survive the corruption, for most families. With the clean version's neighbours (oracle), Section 4 would reach 0.898 / 0.895. Gain-free keys do not fix it (cosine 0.835 / 0.868, log-pattern 0.844 / 0.872). Section 4 per family: fog 0.546, contrast 0.681, zoom blur 0.700. The CDFs reach 0.725 / 0.851 / 0.830 on these.

(4) Main result. Log peakiness, log(top-1%) − log(mean), conditioned, s1–s3:
- AUROC common 0.897 [0.881, 0.911], extra 0.859 [0.844, 0.871], severity-1 common 0.814 [0.794, 0.835];
- severities 1–5: 0.814 / 0.876 / 0.912 / 0.935 / 0.949;
- AUPR common 0.874; FPR95 common 0.253, extra 0.352. The CDFs give FPR95 0.403 / 0.428; Section 4 gives 0.368 / 0.317.

Paired bootstrap, 1,000 draws:
- against the CDFs: +0.072 [+0.056, +0.087] common, +0.051 [+0.038, +0.063] extra, +0.098 [+0.079, +0.115] at severity 1;
- against Section 4: +0.056 [+0.044, +0.069] common, −0.011 [−0.022, −0.000] extra, +0.050 [+0.036, +0.066] at severity 1.

Variants and per family:
- unconditioned: 0.849 / 0.805;
- p99 instead of top-1%: 0.886 / 0.853;
- oracle neighbours: 0.927 / 0.879;
- per-stage conditioned AUROC common: s1 0.873, s2 0.878, s3 0.874, s4 0.812;
- per family (mean AUROC): fog 0.875, contrast 0.941, zoom 0.905, defocus 0.979, pixelate 0.878, frost 0.802, JPEG 0.922;
- still weak: brightness 0.660, saturate 0.705, elastic 0.681.

(5) Mechanism.
- Clean per-channel correlation of level and peakiness: −0.738 / −0.847 / −0.876 at s1 / s2 / s3.
- The ceiling's spread is 0.71 / 0.57 / 0.54 of the level's.
- Share of s1 channels where both fall at severity 1: fog 0.52, contrast 0.49, zoom 0.45, blurs 0.35–0.40, pixelate 0.39. Noise gives 0.04–0.07, snow 0.05, spatter 0.04, brightness 0.08. A move with the clean correlation would give about 0.12.
- s1 score d′ at severity 1, level vs shape: fog −0.06 vs +1.90, contrast +0.03 vs +2.19, zoom +0.14 vs +1.85.
- My first two explanations were refuted:
  - RMS shift sizes are about equal (fog s1, level vs shape: 1.02 vs 1.54; equal at s2–s3);
  - the effective dimension is about equal (13.0 vs 13.2 at s1; clean CV 0.301 vs 0.309).

(6) Exploratory, chosen after looking. A signed "flatter than neighbours" term alone gives 0.692 / 0.566, because it misses noise. Its severity-1 AUROC is 0.969 on fog and 0.957 on contrast. Added to the level score it gives 0.916 / 0.831, severity-1 0.856, FPR95 0.256.

**Runners-up:**
- Black point from the stem: Brightness (HSV value + c) lifts every pixel's value by at least c, so no pixel stays truly dark, and the maxima of the stem's darkness-tuned channels should fall. This targets the only weak spot the shape statistic leaves (brightness 0.660, saturate 0.705), like a dark-channel prior inside the detector (my unverified recollection of He et al.), and needs one stem-level pass.
- A corruption-invariant content key: With the clean version's neighbours the same scores would reach 0.927 / 0.879 (shape) and 0.898 / 0.895 (level). At severity 3 only 14–72% of the stage-4 neighbours survive the corruption, and gain-free keys did not help (cosine 0.835, log-pattern 0.844 common), so the search is for a key inside the detector that the corruption cannot move.

### Round 2: critiques written by A

**On B** (serious)

- Steelman: Log channel energies, whitened by their clean covariance, turn each corruption's small residue in the quiet directions into evidence, with no key and no neighbours, and B pre-registered a held-out test and passed it (0.882/0.858 on 308 images; I get 0.874/0.853 on 859).
- Objection: Whitening is the right geometry for the level but the wrong one for the shape, and the shape is what carries the gain and blur families. So 'invisible per channel, obvious jointly' is only half true.
- The shape's corruption shift is coherent. At s1, severity 1, 67-74% of the paired shift energy of fog, contrast, zoom and defocus lies on the first principal direction of the clean residuals (PC1), which holds only 19% of clean variance. The shift is nearly uniform across channels (|cos| with the all-channels direction 0.65-0.75). For the level, PC1 holds only 7-10%.
- Whitening divides that direction by its large clean variance. Whitened log peakiness scores 0.848/0.826, against 0.897/0.859 for the same statistic averaged over channels against content-matched neighbours: -0.049 [-0.059, -0.039].
- B's joint [log mean, log peakiness] whitening scores -0.014 [-0.020, -0.008] below B's own primary. So B's finding that adding the tail hurt follows from the geometry; it is not evidence that the tail carries nothing.
- The cost falls on IV's flagship degradation. On 859 held-out images B matches the CDFs at severity 1 on fog (0.632 vs 0.634) and contrast (0.670 vs 0.665), where the shape row reaches 0.799 and 0.821. Overall A - B = +0.019 [+0.014, +0.023] common, +0.005 [+0.001, +0.009] extra, +0.029 [+0.023, +0.035] at severity 1.
- Novelty: the method is Rippel et al. 2020's Ledoit-Wolf Gaussian (verified by E) plus a log, a risk B names itself.
- Failure scenario: An IEEE IV reviewer runs light fog from ACDC or Foggy Cityscapes. The fog shift lies in the content directions that whitening discounts, so B flags mild fog no better than the CDFs it claims to beat. The paper's weakest family is then the road's canonical degradation.
- Evidence: - r2_mechanism.log: the coherence and PC1 shares.
- r2_screen.log: whitened shape 0.848/0.826; joint whitening -0.014.
- r2_heldout.log: 859 images, positions 200-1058, rows frozen in PREREGISTRATION_R2.md before any scoring.
- L5 and L8.
- Caveat against my own point: in my depth-fog probe on 12 Cityscapes images, Koschmieder fog moves the s1 level -0.80 SD at beta = 0.01, against -0.48 for uniform fog. B's fog weakness may therefore partly be an artefact of the package's fog.
- Would change my mind: Either result would change my mind: B >= A on Foggy Cityscapes or ACDC fog at beta = 0.005-0.01 with a Cityscapes bank; or, on the 4,800 held-out images, a whitening that keeps the uniform direction (for example whitening only the complement of the clean PC1) beating the channel-averaged shape.

**On C** (fixable)

- Steelman: C independently reached my row (content-conditioned log(top-1%/mean) at s1-s3, k = 50, two-sided), gave an honest topology negative and found NAP; the row now passes a pre-registered held-out test with almost no shrinkage (859 images: 0.893/0.858; screen 0.897/0.859).
- Objection: C's explanation of why the row works is wrong. 'Content scales both ends together, so the ratio cancels much of the content' fails on the bank:
- SD(log peak share)/SD(log mean) is 0.89/0.79/0.70 at s1/s2/s3.
- A ridge from the stage-4 key, scored on the 500 z-statistics images, explains 60/69/68% of the shape's variance, against 69/77/77% for the level.
- After kNN conditioning the leftover spread is the same: shape 0.78/0.72/0.73, level 0.81/0.73/0.70.
So the ratio is about as content-driven as the level. Two things actually make it work:
- Gain and blur families drop all s1 channels' peakiness together (PC1 share 0.67-0.74).
- Conditioning recruits peakier neighbours for a dimmed image (clean conditional corr(level residual, shape residual) -0.73/-0.84/-0.84).
'Corruption makes firing flatter' is also not universal:
- NAP's one-sided global ratio, the same intuition, is inverted on this data (0.302/0.343 at s1-s3).
- Under depth-dependent fog, s3 gets peakier (+0.17/+0.38 SD at beta = 0.01/0.02).
- Failure scenario: A reviewer checks the paper's central explanation with the R-squared numbers above and finds it false. Then, on Foggy Cityscapes, the paper predicts flattening at every stage while s3 turns peakier.
- Evidence: - r2_mechanism.log: SD ratios, R-squared, conditional spreads, coherence.
- r2_x1.log: conditional correlations.
- r2_nap.log: NAP inverted.
- r2_depthfog.log: s3 peakier under depth fog.
- r2_heldout.log: the shared row passes K1-K3 and every paper threshold.
- Would change my mind: A content-predictability measure under which the shape is much less content-driven than the level (for example a ridge R-squared below half the level's), or a key for which conditioning helps the level as much as it helps the shape.

**On D** (fixable)

- Steelman: Reading a level and a crest per channel keeps the level's sensitivity to noise and spatter while adding the crest's sensitivity to visibility loss; D has the most concrete IV plan (a Foggy Cityscapes kill at beta = 0.005/0.01 and an upper/lower band split), and its frozen row beats the CDFs on held-out images (+0.055 [+0.048, +0.062] common, +0.062 extra).
- Objection: Both of D's design choices cost signal, and the cost lands on D's flagship family.
- The raw ratio: D's crest alone (0.876/0.849) loses to the log ratio (0.897/0.859) by +0.022 [+0.017, +0.026] / +0.010 [+0.006, +0.013].
- The equal-weight sum with a level that is blind to fog (the level's s1 d' at severity-1 fog is -0.06) dilutes the crest. Fog falls from 0.825 for D's own crest alone to 0.739 for D's row.
- D therefore fails its own pre-registered bar 'fog >= 0.75' on the screen (0.739) and on 859 held-out images (0.747). Its severity-1 fog of 0.670 is only +0.036 above the CDFs.
- On held-out images the log version of the same sum (my E2) beats D on every headline: +0.011 [+0.010, +0.012] common, +0.003 [+0.002, +0.004] extra, +0.013 at severity 1. The shape alone beats D by +0.020 [+0.017, +0.023] common (fog 0.874 vs 0.747, contrast 0.936 vs 0.862).
- D's mechanism ('content mostly moves the level, the peaks are stable') overstates: the log crest varies 70-89% as much as the level and is 60-69% predictable from the content key.
- Failure scenario: The paper sells 'catches the fog and contrast loss that level-only monitors miss', but the scored row gives up 0.13 of fog AUROC to the crest it contains. A reviewer who runs the crest alone, or its log, gets a better fog monitor than the paper's method.
- Evidence: - r2_screen.log: D reproduced exactly (0.876/0.870).
- r2_heldout.log and r2_heldout_families.log.
- D's own visibility_stats: conditioned crest fog 0.825.
- My Round-1 d': level -0.06 vs shape +1.90 at s1 for severity-1 fog.
- L8: adding a component that does not separate only adds noise.
- Would change my mind: Level + crest beating crest alone on Foggy Cityscapes or ACDC. My depth-fog probe makes this plausible: under depth-dependent fog the s1 level moves -0.80 to -1.03 SD, more than the shape (-0.50). If that holds on real fog, D is right to keep the level for IV, and only the raw ratio and the plain sum need fixing (log ratio, max instead of sum).

**On E** (serious)

- Steelman: E asks the right diagnostic question (is a monitor blind because the detector is invariant, or because scene variation hides the corruption?) and answers it with a same-scene oracle backed by a placebo, a lambda sweep and a blend curve, which reframes L10.
- Objection: The split into aliasing and blindness depends on which statistic is read. A single-image row contradicts the headline 'a single image cannot tell a hazy scene from a fogged one'.
- At severity 1 on the same 200 images, the conditioned peakiness closes 62% of E's fog gap (0.558 -> 0.800; oracle 0.949), 67% of the contrast gap (0.581 -> 0.823; oracle 0.941) and 59% of the zoom-blur gap. Held out it keeps 0.799 and 0.821.
- My Round-1 pre-registered secondary E4 (signed 'flatter than neighbours' term + level) reaches 0.936 on fog and 0.928 on contrast at severity 1 on 859 held-out images. That is close to the oracle's screen values (0.949/0.941), from one image.
- So what E calls aliasing for fog, contrast and zoom is mostly the cost of reading the level instead of the shape.
- E's diagnosis survives where the shape closes nothing: frost (-0.22), elastic (-0.24), saturate (+0.03), brightness (-0.09), spatter (-0.29).
The deployable rows are weak:
- Row (a) is the lowest of the five frozen rows on held-out images (0.846/0.851; A - E = +0.046 [+0.039, +0.054]).
- Row (b) needs a clean view of the same scene. That breaks the one-image rule, and no such view exists for fog present from the first frame.
- E's own blend curve shows the advantage decaying with scene change (0.977 at 25% other scene, 0.868 at 100%), and frames from a moving car are exactly that kind of changed scene.
- Failure scenario: The paper's Figure 1 shows a large aliasing bar for fog and contrast. A reviewer adds the conditioned shape row, two thirds of the bar disappears with no same-scene reference, and the thesis becomes 'the authors read the wrong statistic'.
- Evidence: - r2_egap.log: my row against E's own figure1_table.csv columns, same images.
- r2_heldout.log and r2_heldout_families.log: A and E4 held out.
- E's oracle_checks.log: the blend curve.
- Table A: severity-1 CDFs fog 0.638, contrast 0.665.
- Would change my mind: On the 4,800 held-out images the shape closes under a third of the fog and contrast gap, or the same-scene reference (b) beats single-image rows at matched false-alarm rates on real driving onsets with ego-motion.

**Defense of own:** I expect three attacks on my proposal. Here is each, with my answer, then what I change.

**Attack 1, L11: I chose the statistic after seeing Section 4 fail on fog on the same 200 images.**
- Answer: a held-out test whose rows I froze in PREREGISTRATION_R2.md before scoring anything.
  - It used all 859 confirmation images at positions 200-1058. Each of these files stores top-1% means since 16:35.
  - The confirmation files equal pilot200.npz exactly on the 200 screen images.
- My unchanged row scores 0.893 [0.884, 0.902] common, 0.858 [0.850, 0.866] extra, 0.810 at severity 1, FPR95 0.251/0.335. The screen gave 0.897/0.859/0.814/0.253, so it barely shrank.
- Against the CDFs: +0.075 [+0.068, +0.082] common, +0.053 [+0.047, +0.058] extra, +0.105 [+0.095, +0.114] at severity 1.
- Against Section 4: +0.052 [+0.046, +0.059] common, -0.008 [-0.013, -0.002] extra.
- K1, K2 and K3 all pass:
  - fog, contrast and the four blurs carry +0.851 of the +0.785 total per-family gain (other families net negative);
  - the held-out clean anti-correlation is -0.711.
- Every paper-making threshold passes. The severity-1 margins over the CDFs are fog +0.165, contrast +0.155, zoom +0.102.

**Attack 2, novelty against NAP (Wan et al., arXiv 2402.18162, verified from the paper).**
- NAP uses the same peak-to-mean intuition: (1/C) sum (max/mean)^2, at the penultimate layer, one-sided.
- As published it points the wrong way for corruption here: 0.302/0.343 at s1-s3 and 0.313/0.380 at s4.
- Its two-sided form against the bank reaches 0.835/0.805. A per-channel log, standardisation and content conditioning add +0.062 [+0.050, +0.075] on top of that.
- So the contribution is not the ratio. It is where it is read (early stages), how (per-channel log, two-sided, content-matched) and why.

**Attack 3, D's 'real fog spares the peaks'.**
- My CPU probe partly confirms it. I applied Koschmieder fog with a flat-ground depth proxy to 12 Cityscapes images:
  - the s1 shape still falls (-0.50 to -0.64 SD);
  - the level falls more at beta >= 0.01 (-0.80 and -1.03 SD);
  - s3 turns peakier (+0.17/+0.38 SD).
- It also shows that my Round-1 sentence 'the package's fog has the same form as Koschmieder' was wrong. The package's plasma term exaggerates the shape drop: -0.93 SD, against -0.66 for uniform airlight at a similar transmission.

**What I change.**
1. **The mechanism.** The shape is not less content-driven: ridge R-squared 0.60-0.69, against 0.69-0.77 for the level. It works for two reasons:
   - gain, blur and compression drop all early channels' peakiness together (PC1 share 0.67-0.74 at s1), which channel averaging accumulates and whitening suppresses;
   - the content key recruits peakier neighbours for dimmed images (conditional correlation -0.73 to -0.84).
2. **The IV claim.** I withdraw 'level monitors fail on real fog'. On Foggy Cityscapes I pre-register both arms and per-stage numbers, with a kill if the shape arm does not beat the level arm at beta = 0.005 and 0.01.
3. **A Round-3 candidate, M1 = max(flatter arm, level arm).** It is frozen in PREREGISTRATION_R2_M1.md and judged only on positions above 1058.
   - The idea: a corruption either flattens early channels relative to content-matched clean scenes, or moves their level.
   - Screen: 0.922/0.862, severity 1 0.869, FPR95 0.229; M1 - A = +0.024 [+0.013, +0.038] common.
   - Its summed sibling E4 was pre-registered in Round 1 and scored 0.916/0.836 held out (severity 1 0.856).
   - M1 replaces my row only if it beats A on common with an interval above 0 and does not lose on extra.

**Negative results I keep.**
- A 2x2 per-channel trade-off whitening: -0.007 [-0.012, -0.003].
- The B + A fusion: -0.002 [-0.004, +0.001] held out.
- The L10 cap remains: brightness 0.653, saturate 0.713.

**Best other proposal:** If mine were off the table I would bet on C, because it is the same row and is now validated held out; I would keep its row and replace its explanation with mine. Among the genuinely different ideas I would bet on B:
- it is the strongest row that needs no content key (0.874/0.853 on 859 held-out images);
- it has its own pre-registered pass;
- its 'residue in the quiet directions' is the right account of the level arm.
I would not bet on whitening the shape, or on fusing B's score into the shape row.

**Merge suggestion:** Merge A and C into one paper, and take one part from each of B, D and E.

**Thesis:** corruptions that remove contrast or detail make a frozen detector's early channels flatter than clean scenes with the same content, while noise-like corruptions move their level.

**Method:** one shape number and one level number per channel, read against the detector's own stage-4 content key. The primary is the held-out-validated shape row, or M1 if it passes on positions above 1058.

**From B:** the geometry explanation and the ablation.
- Level shifts are incoherent residues, which whitening finds.
- Shape shifts are coherent common-mode drops, which channel averaging accumulates and whitening suppresses (whitened shape 0.848 vs 0.897).
- Fusing B's score adds nothing (-0.002 held out), so it stays an ablation, not a component.

**From D:** the IV protocol, meaning the Foggy Cityscapes kill at beta = 0.005/0.01 and the upper/lower band split. My depth probe now motivates the split, since s3 turns peakier under depth-dependent fog.

**From E:** the oracle decomposition, recomputed for the merged row, as the analysis figure.
- The shape closes 59-67% of the fog, contrast and zoom gaps from one image.
- Frost, elastic, saturate, brightness and spatter stay aliased. Those are the honest limits.

**From C:** the topology negative as one paragraph, and NAP as the closest prior, with its sign inverted on corruption (0.30).

**Process:**
- The final 4,800-image report should keep positions 200-1058, used here for frozen rows, apart from positions above 1058. Only the latter are clean for any Round-3 revision.
- None of the five proposals drifts into predicting performance drop; all are judged on detection AUROC and FPR95.

**Ranking:**
1. A: The only proposal validated on held-out images by its own pre-registered rules with every threshold met (859 images: 0.893/0.858, severity 1 0.810, FPR95 0.251). It beats B, D and E on common and at severity 1, fixes fog, contrast and zoom, and its mechanism is now checked, though its IV fog claim is weakened.
2. C: The same validated row plus an honest topology negative, but its explanation (the ratio cancels content) is refuted by the R-squared and spread checks.
3. B: The strongest row with no content key, with its own pre-registered pass (0.874/0.853 here) and the right account of the level arm. But it loses to A by +0.019 [+0.014, +0.023], matches the CDFs on mild fog and contrast, and its novelty over Rippel 2020 is a log.
4. D: The log version of its own row (my E2) beats it on held-out images (+0.011 [+0.010, +0.012] common), and it fails its own fog bar (0.747 < 0.75). It has the best IV plan, and its level arm may matter for real fog.
5. E: A valuable diagnostic whose main claim depends on the statistic: the shape closes about two thirds of the fog and contrast gap from one image. Its deployable row is the weakest held out (0.846/0.851), and the same-scene variant breaks the one-image rule.

**Checks run:** All checks ran on the CPU, niced, with BLAS capped at 4 threads. Scripts, logs and pre-registrations are in /tmp/claude-1000/-home-yuchen-YuchenZ-UE-philip-sa/3d52b916-b3ff-4806-a501-7f70cde31a29/scratchpad/detection/roundtable/panelist-A/r2/. The repository was not modified and the GPU was not used.

(0) Drift check: none of B-E predicts performance drop.

(1) r2_screen.py, the 200 pilot images (a screen).
- Every frozen row reproduces its reported numbers exactly: A 0.897/0.859 (FPR95 0.253), Section 4 0.841/0.870, B 0.876/0.856, D 0.876/0.870, D crest 0.876/0.849, E 0.855/0.860, A-E2 0.890/0.874, A-E4 0.916/0.831, CDFs 0.825/0.808.
- Whitened log peakiness, no key: 0.848/0.826; vs A -0.049 [-0.059, -0.039]; vs B -0.028 [-0.036, -0.021].
- Retrieval + whitened log peakiness: 0.858/0.832; vs A -0.040 [-0.049, -0.031].
- Joint whitened [log mean, log peakiness]: 0.863/0.837; vs B -0.014 [-0.020, -0.008].
- Conditioned log ceiling alone: 0.853/0.818; A minus it +0.044 [+0.033, +0.056].
- Conditioned log level: 0.871/0.874; A minus it +0.026 [+0.019, +0.035] common, -0.015 [-0.022, -0.008] extra. Its fog is 0.638 and contrast 0.804.
- A minus D's raw crest: +0.022 [+0.017, +0.026] / +0.010 [+0.006, +0.013].
- F (B + A): 0.894/0.862; F - A = -0.003 [-0.008, +0.001].

(2) r2_heldout.py, PRE-REGISTERED (PREREGISTRATION_R2.md written before scoring).
- 859 images at positions 200-1058. On the screen images, max |confirmation - pilot| = 0.
- Rows: A 0.893 [0.884, 0.902] / 0.858 [0.850, 0.866], severity 1 0.810, FPR95 0.251/0.335; Section 4 0.841/0.866; E2 0.884/0.870; E4 0.916/0.836 (severity 1 0.856, FPR95 0.237/0.397); B 0.874/0.853; D 0.873/0.867; E 0.846/0.851; F 0.891/0.860; CDFs 0.818/0.805.
- A minus others:
  - A - CDFs +0.075 [+0.068, +0.082] / +0.053 [+0.047, +0.058] / +0.105 [+0.095, +0.114] at severity 1;
  - A - Section 4 +0.052 [+0.046, +0.059] / -0.008 [-0.013, -0.002];
  - A - B +0.019 [+0.014, +0.023] / +0.005 [+0.001, +0.009] / +0.029 [+0.023, +0.035];
  - A - D +0.020 [+0.017, +0.023] / -0.009 [-0.012, -0.007];
  - A - E +0.046 [+0.039, +0.054] / +0.007 [+0.001, +0.013].
- Other pairs:
  - E2 - D +0.011 [+0.010, +0.012] / +0.003 [+0.002, +0.004];
  - E4 - A +0.024 [+0.016, +0.031] / -0.022 [-0.029, -0.015];
  - F - A -0.002 [-0.004, +0.001].
- Against the CDFs:
  - B +0.056 [+0.049, +0.063] / +0.048 [+0.042, +0.053];
  - D +0.055 [+0.048, +0.062] / +0.062 [+0.056, +0.068];
  - E +0.029 [+0.018, +0.038] / +0.046 [+0.037, +0.054];
  - Section 4 +0.023 [+0.014, +0.031] / +0.061 [+0.053, +0.068]. This is partial and not the confirmation's decision.
- K1, K2 and K3 pass.
- Severity-1 fog / contrast: A 0.799/0.821, B 0.632/0.670, D 0.670/0.698, E 0.567/0.582, E4 0.936/0.928, CDFs 0.634/0.665. D's mean fog is 0.747.

(3) r2_mechanism.py, content and coherence.
- SD(log peakiness)/SD(log mean): 0.89/0.79/0.70.
- Ridge R-squared: level 0.69/0.77/0.77, shape 0.60/0.69/0.68.
- Conditional spread: level 0.81/0.73/0.70, shape 0.78/0.72/0.73.
- PC1 share / |cos uniform| at s1, severity 1:
  - shape: fog 0.69/0.75, contrast 0.74/0.70, zoom 0.73/0.70, defocus 0.67/0.65;
  - level: 0.10/0.46, 0.09/0.50, 0.07/0.34, 0.08/0.22.

(4) r2_egap.py, share of E's severity-1 oracle gap closed by A.
- Fog +0.62, contrast +0.67, zoom +0.59, defocus +0.49, glass +0.32.
- Frost -0.22, elastic -0.24, saturate +0.03, brightness -0.09, spatter -0.29.
- Over E's six aliased families, 30%.

(5) r2_nap.py, NAP baselines.
- NAP one-sided: 0.313/0.380 at s4, 0.302/0.343 at s1-s3.
- Two-sided raw ratio against the bank: 0.835/0.805. A minus it: +0.062 [+0.050, +0.075] / +0.054 [+0.042, +0.066].

(6) r2_depthfog.py, 12 Cityscapes val images, CPU forward pass (137 s).
- s1 (level shift, shape shift) in clean conditional SD units:
  - package fog severity 1: (-0.43, -0.93);
  - depth fog at beta 0.005 / 0.01 / 0.02: (-0.43, -0.64) / (-0.80, -0.51) / (-1.03, -0.50);
  - uniform fog matched to beta 0.01 / 0.02: (-0.48, -0.66) / (-0.61, -0.76).
- s3 shape under depth fog: +0.17 / +0.38 at beta 0.01 / 0.02; the share of channels moving along the clean band is 0.52 / 0.58.

(7) r2_x1.py, exploratory: the 2x2 trade-off score gives 0.890/0.849, -0.007 [-0.012, -0.003] against A. The conditional correlation of the level and shape residuals is -0.733/-0.841/-0.837.

(8) r2_m1.py, exploratory, chosen after seeing held-out numbers, screened on the pilot only.
- M1: 0.922/0.862, severity 1 0.869, FPR95 0.229/0.358.
- M1 - A: +0.024 [+0.013, +0.038] / +0.003 [-0.010, +0.017] / +0.055 [+0.037, +0.074].
- Severity-1 fog 0.949, contrast 0.941.
- Frozen in PREREGISTRATION_R2_M1.md for positions above 1058.

### Round 3: rebuttal by A

- **To B** (partially concede): (1) Real Koschmieder fog depends on depth. Near, high-contrast edges keep t of about 0.9 and may hold the top 1% while the far bulk is erased, so the peak share could rise. That would invert Figure 1 and the IV prediction. (2) The 'level-peakiness trade-off' is the spurious correlation of a ratio with its own denominator, so K3's correlation clause can never fire. — (2) Conceded in full.
- corr(log level, log ratio) < 0 means only that, on clean images, the log-log slope of the ceiling on the level is below 1.
- On the bank that slope is 0.38/0.37/0.41 at s1/s2/s3, below 1 in 98-100% of channels, and corr(log level, log ceiling) is +0.56/+0.64/+0.78. So there is no trade-off.
- I drop the word, the left panel of Figure 1 and K3's correlation clause.
- The replacement lives in the (log level, log ceiling) plane, where no ratio is formed. At severity 1, the families that remove structure lower the s1 ceiling 4-14x more than the level, in log units: fog -0.280 vs -0.061, contrast -0.283 vs -0.057, zoom -0.268 vs -0.047, defocus -0.231 vs -0.027. They do so in 77-84% of channels at once.

(1) Partly conceded. Your failure scenario's direction (median s1 delta-log-pi >= 0) did not occur on either depth-fog proxy.
- D's probe: median s1 shift of the log share -0.50/-0.77/-1.20 SD at beta 0.005/0.01/0.02, falling in 70-73% of (image, channel) pairs.
- My rerun of D's exact fog model on the same 24 frames: 70-74% of s1 channels flatten, 61-63% at s2, 49-50% at s3.
- You are right about s3, and about size: the shift is about half of package fog's.
- With a Cityscapes reference the two-sided shape ties the level: 0.776 vs 0.792 at beta 0.005, 0.894 vs 0.908 at 0.01 (+/-0.08).
- So I withdraw 'a textbook lowering of the ceiling' and the IV arrows for real fog. I keep the s1-s2 direction and pre-register it on Foggy Cityscapes with your kill: a median s1 delta-pi >= 0 at beta 0.01.
- Caveat: both probes use flat-ground depth from gtFine labels, not Foggy Cityscapes itself.
- **To C** (concede): The trade-off, and Figure 1's descending band, are an artefact of dividing by the level, so K3 cannot fire. 'The ceiling is nearly fixed by exposure and optics' is overstated: the 50-scene reference explains as much of the ceiling's clean variance as of the level's. NAP is not cited. — Conceded on all three.
- Neither of your tests for a genuine trade-off passes on the bank. corr(log level, log ceiling) is positive (+0.56/+0.64/+0.78), and the observed corr(level, ratio) is no stronger than under independence.
- Your content-share numbers refute 'fixed by optics': ceiling 0.46/0.51/0.47 against level 0.35/0.47/0.50. I withdraw that sentence.
- What survives is your own shift-size result: at s1, severity 1, the ceiling moves -1.89/-1.92/-2.04/-1.96 residual SDs and the level -0.45/-0.24/-0.20/-0.12 under fog/contrast/zoom/defocus. My direction and coherence numbers add to it: content rises along a slope of about 0.4, while these corruptions drop the ceiling almost vertically.
- The new Figure 1 plots log level against log ceiling with no ratio. Your reviewer's redraw with independent variables therefore cannot produce its arrows.
- NAP (Wan et al., arXiv 2402.18162, verified in Round 2) is now cited with numbers:
  - one-sided here 0.302/0.343;
  - two-sided against the bank 0.835/0.805;
  - per-channel log, standardisation and conditioning add +0.062 [+0.050, +0.075].
- Your frozen row equals mine to three decimals, so I merge with C.
- **To D** (concede): The IV prediction fails on a depth-fog driving probe. With a Cityscapes reference the plain level gives 0.78/0.90 at beta 0.005/0.01 and the shape 0.76/0.88. Depth fog moves the shape about 30% less per unit of level lost, and not at all at s3, so 'shape, not level' looks like a property of COCO's content. K3's correlation is arithmetic, and NAP is uncited. — Conceded, and replicated.
- With your fog model, your 24 frames and a 24-frame Cityscapes-train reference: level 0.792/0.908/0.976, two-sided shape 0.776/0.894/0.979 at beta 0.005/0.01/0.02.
- I withdraw 'shape, not level' and 'level monitors struggle on fog'. On COCO, the level's blindness to fog came partly from the stage-4 key absorbing the level shift (s1 d' -0.06). A global driving reference does not absorb it.
- A new finding is worse for every conditioned row. With the COCO bank and key they collapse on Cityscapes: level 0.573/0.542/0.632, shape 0.635/0.684/0.804, max rule 0.644/0.693/0.693. The bank must come from the deployment domain.
- The merged proposal adopts your two-axis thesis. The max-of-arms row was the best at the two thin-fog levels (0.826/0.920), but within +/-0.08 on 24 frames.
- I pre-register your official test (Foggy Cityscapes and ACDC, 2,000-frame Cityscapes-train bank) with a weaker claim: the fixed row is not below the conditioned level row. Your >= +0.03 bar is reported, not claimed.
- K3's correlation clause is dropped, and NAP is cited.
- **To E** (concede): 'Read the ratio, not its mean' is falsified family by family. The common gain over the level comes from fog, contrast and zoom, while the plain level is significantly better on 9 of 19 families and on extra. The defensible claim is level AND shape. 'Corruption lowers the ceiling' holds only for the typical channel, since aggregated NAP-style ratios point the wrong way. — Conceded. Your test fails for my row on 753 images no panelist had read (positions 1059-1811).
- Shape alone beats level + shape on common, +0.010 [+0.007, +0.014], but loses on extra, -0.011 [-0.014, -0.009].
- Its per-family losses to Section 4 exclude 0 on 9 families: elastic -0.056, spatter -0.034, brightness -0.024, speckle -0.019, saturate -0.018, snow -0.012, shot -0.007, gaussian -0.004, impulse -0.004.
- So 'not its mean' is withdrawn, and the merged thesis reads both.

The two-arm row is my frozen M1 (max of a signed flatter arm and the level arm): 0.924/0.863, severity-1 0.872, FPR95 0.212/0.351.
- M1 - A: +0.021 [+0.015, +0.028] common, -0.0006 [-0.007, +0.006] extra.
- My frozen rule needs an extra point estimate >= 0, so A stays for now. The formal decision comes at completion.
- Against Section 4, M1 turns the elastic and spatter losses into +0.025 and -0.003. But it loses more on brightness (-0.075) and saturate (-0.057).
- Your preferred E2 has the best extra (0.875) and is reported.

I adopt the typical-channel qualifier: at severity 1, 77-84% of s1 channels flatten under the families that remove structure, 52-53% under noise and 41% under brightness.

**Status:** merged (merged with: C, D)

**Final proposal:**

# Merged proposal A + C + D: Peaks fall, bulk stays

**Status: merged.**
- A and C froze the same row.
- D's two-axis thesis and driving plan are adopted.
- E's per-family critique added the level, and E's oracle decomposition becomes the analysis of what remains.
- B's critique removed the "trade-off".
- No new row was introduced in Round 3. A, E2 and Section 4 were frozen in Round 1, M1 in Round 2.

## Thesis
- Across clean scenes, an early detector channel's peak response rises with its bulk, at about 0.4 of its rate.
- Corruptions that remove image structure (blur, haze, contrast loss, compression) instead lower the peak while the bulk barely moves, in most channels at once.
- So each early channel's log peak-to-mean ratio, compared with clean scenes of the same content, detects corruption better than activation magnitudes or full activation distributions. The channel level adds the corruptions that add energy.

*Short form:* content raises the peaks with the bulk; corruption drops the peaks below it.

## Withdrawn since Round 1
- **The "level-peakiness trade-off"** (B, C, D). The negative correlation only says that the ceiling grows more slowly than the level. It is gone from the mechanism, from Figure 1 and from K3.
- **"The ceiling is fixed by exposure and optics"** (C). The 50-scene reference explains as much of the ceiling's clean variance as of the level's: 0.46/0.51/0.47 vs 0.35/0.47/0.50.
- **"Shape, not level"** (E, D). The level wins on 9 of 19 families, and it ties the shape on depth-fog driving frames.
- **"Fog is a textbook lowering of the ceiling" and "level monitors fail on real fog"** (B, D).

## Insight: the mechanism, with no ratio
Everything below is read per channel in the (log level, log ceiling) plane.

1. **The content direction.**
   - The 2,000 clean scenes rise along a slope beta = 0.38/0.37/0.41 at s1/s2/s3 (median over channels).
   - beta < 1 in 98-100% of channels; r = +0.56/+0.64/+0.78.
   - A richer scene raises both numbers, but the bulk faster.
2. **The corruption direction.** At severity 1, paired on the same image, the families that remove structure lower the s1 ceiling 4-14x more than the level (log units):

   | Family | Level | Ceiling |
   |---|---|---|
   | fog | -0.061 | -0.280 |
   | contrast | -0.057 | -0.283 |
   | zoom blur | -0.047 | -0.268 |
   | defocus | -0.027 | -0.231 |
   | glass | -0.016 | -0.156 |

   In content-residual SD units (C's numbers), the ceiling moves -1.89 to -2.04 and the level -0.12 to -0.45.
3. **Coherence.**
   - Under these families, 77-84% of s1 channels flatten.
   - 16-39% of the ratio shift's energy lies on the all-channels direction. For noise it is 3-4%, close to the 1/C expected without a common component.
   - Averaging over channels accumulates such a shift, and a *signed* average is its matched filter.
4. **The other way.**
   - Noise, snow and spatter do not flatten (52-58% of channels); they move the level.
   - Brightness and saturate move neither (L10).
5. **Why level-only references miss package fog (kept).**
   - Fog lowers the stage-4 key, which then recruits lower-level neighbours and absorbs the level shift (s1 d' -0.06).
   - Those neighbours have higher ratios, so the shape deviation grows (d' +1.90).

The ratio, rather than the ceiling alone, is the shape arm because it also catches noise: the conditioned log ceiling gives 0.853/0.819, against the ratio's 0.897/0.859 (C).

## Method (three sentences)
1. **Read two numbers per channel.** At s1-s3 (the post-ReLU inputs of `res_layers[s].blocks[1].branch2a.conv`), take each channel's level m = mean |a| and its shape pi = log(t + 1e-6) - log(m + 1e-6), where t is the mean of its top 1% of positions.
2. **Compare with content-matched clean scenes.** N is the 50 of 2,000 clean COCO-train images nearest in bank-standardised stage-4 channel means.
   - Shape arm, per stage: mean_c |pi - mean_N pi| / sd_bank(pi).
   - M1 adds a signed flatter arm, mean_c -(pi - mean_N pi)/sd, and the level arm, mean_c |m - mean_N m|/sd (Section 4).
3. **Combine.** Z-score each stage on 500 other clean images and sum over s1-s3.
   - The score is the shape arm: A, the primary since Round 1.
   - If M1 passes its frozen rule at completion, the score is instead M1 = max(flatter arm, level arm), each arm re-z-scored.

The method is training-free, clean-only and label-free, with one score per image. E2 (level arm + shape arm) is reported as the balanced secondary.

## Evidence: frozen rows on held-out images
Positions 1059-1811 (753 images, unread by any panelist before Round 3):

| Row | AUROC common [95%] | AUROC extra [95%] | Sev-1 | FPR95 common / extra |
|---|---|---|---|---|
| A (shape) | 0.902 [0.895, 0.910] | 0.864 [0.857, 0.871] | 0.820 | 0.246 / 0.335 |
| M1 (max of arms) | 0.924 [0.917, 0.930] | 0.863 [0.856, 0.869] | 0.872 | 0.212 / 0.351 |
| E2 (sum of arms) | 0.892 [0.884, 0.901] | 0.875 [0.869, 0.882] | 0.807 | 0.267 / 0.304 |
| Section 4 (level) | 0.844 | 0.870 | 0.765 | 0.355 / 0.318 |
| B (log-whitened) | 0.886 | 0.862 | 0.792 | 0.272 / 0.325 |
| Activation CDFs | 0.821 | 0.811 | 0.709 | 0.414 / 0.425 |

**Paired differences:**
- A - CDFs: +0.081 [+0.074, +0.089] common, +0.053 [+0.047, +0.059] extra, +0.111 at severity 1.
- M1 - CDFs: +0.102 common, +0.053 extra, +0.164 at severity 1.
- A - Section 4: +0.058 [+0.052, +0.066] common, -0.006 [-0.012, -0.000] extra.

**The first held-out set** (positions 200-1058, 859 images): A gives 0.893/0.858, which is +0.075/+0.053 over the CDFs.

**Per family, A / M1 / CDFs:**
- fog 0.891 / 0.979 / 0.711;
- contrast 0.945 / 0.983 / 0.844;
- zoom blur 0.925 / 0.943 / 0.822;
- brightness 0.658 / 0.606 / 0.600;
- saturate 0.714 / 0.675 / 0.626.

## Novelty
- **Becker et al. (ICPR 2026).** Per-channel CDFs in absolute units against one global training CDF, so the level dominates, and there is no content reference.
- **NMD.** The level against the BatchNorm means, globally: one arm, unconditioned.
- **NAP** (Wan et al., arXiv 2402.18162; verified).
  - It computes (1/C) sum (max/mean)^2 at the penultimate layer: one-sided, with no reference, for semantic OOD.
  - Moved to s1-s3 it is below chance here (0.302/0.343), because sparse channels sharpen.
  - Two-sided against the bank it gives 0.835/0.805. A per-channel log, standardisation and content conditioning add +0.062 [+0.050, +0.075] (screen).
- **ASH and SCALE.** The same algebra, but across the channels of the pooled penultimate vector, used to rescale logits. Ours runs across the positions of each early channel.
- **Mahalanobis, Rippel et al. 2020 and Gram matrices.** Pooled means or products, with no within-channel shape. B's log-whitened means are -0.019 [-0.023, -0.014] below A on common (Round-2 held-out).
- **SPADE** (verified by E). Retrieve the nearest normal images, then compare finer features. Our reference is SPADE-like, and we claim nothing for the retrieval. The claim is *what* is compared, the move off the content direction, and why it works.
- **Image-quality ancestors** (partly from memory). BRISQUE and NIQE compare shape statistics of normalised luminance with pristine images. Natural images have sparse, heavy-tailed filter responses (Field 1987; Olshausen & Field 1996; unverified). Ours works inside the frozen detector, with its own content key.
- **Unverified beyond Rounds 1-2:** the session's web-search budget is exhausted.

## Decisive experiment
**COCO** (no new GPU: the running pass stores means and top-1% for all 5,000 x 96):
- Score the frozen rows with the frozen bank and z-statistics images.
- Report positions 200-4999, and separately 1812-4999, which no panelist has read.
- Apply the M1 rule on 1059-4999. This must happen before any driving data are scored, so that the driving row is fixed in advance.
- **The paper-making result** keeps the Round-1 thresholds, all met by A on both held-out sets so far:
  - AUROC common >= 0.87 and extra >= 0.83;
  - severity-1 common >= 0.77;
  - FPR95 common <= 0.30;
  - severity-1 fog, contrast and zoom blur each >= the CDFs + 0.08.

**Driving** (one agreed GPU window, after COCO):
- **Data.** One backbone-only pass storing means and top-1%:
  - 2,000 + 500 Cityscapes-train frames (bank and z-statistics);
  - Cityscapes val, clean and as Cityscapes-C (19 x 5);
  - Foggy Cityscapes at beta = 0.005/0.01/0.02 and ACDC (fog, night, rain, snow, and normal), both to download.
- **Pre-registered predictions.**
  - P1: on Foggy Cityscapes, the median paired delta-pi is < 0 at s1 and s2 for every beta, and s3 is about 0.
  - P2: the fixed row is not below the conditioned level row (Section 4 with the Cityscapes bank) at beta 0.005 and 0.01. D's >= +0.03 bar is reported, not claimed.
  - P3: on Cityscapes-C, the fixed row minus Section 4 on the common families has an interval above 0.
  - P4: the COCO bank on Cityscapes is reported as a domain control.
- **Analyses:** D's split into upper and lower image bands, and E's same-scene oracle on ACDC.

## Kill criteria
- **K1 (dead):** the interval of A - CDFs on AUROC common contains 0.
- **K2 (dead; Section 4 stands):** the interval of A - Section 4 on common contains 0.
- **K3 (revised, ratio-free; the mechanism is wrong).** Judged on 1812-4999, it fires if either holds:
  - less than half of the summed per-family gain over Section 4 sits in fog, contrast and the four blurs;
  - fewer than 70% of s1 channels flatten at severity 1 under fog, contrast, zoom, defocus or glass. The screen gives 0.81-0.84.
- **The M1 rule (frozen).** On 1059-4999, M1 replaces A only if M1 - A on common is > 0 with its interval excluding 0, and the extra point estimate is >= 0. Interim: +0.021 [+0.015, +0.028] and -0.0006, so A stays for now.
- **IV-K1 (B's scenario; the mechanism is dead on real fog):** the median paired s1 delta-pi on Foggy Cityscapes is >= 0 at beta = 0.01.
- **IV-K2 (the IV paper reverts to Section 4):** the fixed row is below the conditioned level row at beta 0.005 or 0.01, with an interval excluding 0.

## Figure 1
- **(a) One s1 channel's (log level, log ceiling) plane.**
  - The 2,000 clean scenes form a band rising at a slope of about 0.4.
  - Arrows lead from 20 images to their severity-3 versions. Blur, fog and contrast drop almost vertically, noise and snow run along the band, and brightness and saturate barely move.
  - No ratio is plotted.
- **(b) Coherence by family.** The share of s1 channels that flatten (x) against the share of the shift on the all-channels direction (y). The families that remove structure sit near (0.8, 0.16-0.39); noise sits at (0.52, 0.04).
- **(c) E's decomposition at severity 1.** Three rows per family: the best single-image means row, the shape row and the same-scene oracle.
  - The shape closes 59-67% of the gap on fog, contrast and zoom.
  - Frost, elastic, saturate, brightness and spatter remain open.

## Fit for IEEE IV
- **Runtime.** A camera monitor on features the detector already computes: one top-k per channel at s1-s3 plus a 2,000 x 512 neighbour search. This fits inside L12's 0.2-1.2 ms, next to the detector's 5.9 ms.
- **Coverage.** The shape covers visibility loss (fog, haze, defocus, a smeared windshield). The level covers sensor noise and soiling overlays.
- **Real fog, claimed cautiously.** On two depth-fog proxies the s1 shape still drops in 70-74% of channels, but by about half the package-fog shift, and the level ties the shape there.
- **Calibration.** The bank comes from the deployment camera.

## Risks and cost
- **Domain.** With the COCO bank, every conditioned row collapses on Cityscapes depth fog: AUROC 0.54-0.80 on 24 frames.
- **The signed arm needs content conditioning.** Against a global driving reference it saturates: d' 1.00/1.50/1.82, against 1.15/2.15/4.05 for the two-sided arm.
- **The L10 cap.** Brightness and saturate stay weak, and M1's max costs there: -0.052 and -0.039 against A.
- **M1 vs A is undecided until completion.** The interim extra difference is -0.0006.
- **False alarms.** Soft or hazy clean photos, and night frames with point lights.
- **Key drift.** The key still moves with the corruption, leaving a gap to the oracle: 0.927 vs 0.897 on the screen.
- **Cost.**
  - COCO: none.
  - Driving: about 1 h of GPU, plus downloads.
  - Code: it already exists (r2common.py).

## Credits
- **C:** the identical row, the ladder, the ceiling-shift numbers and the topology negative.
- **D:** the two-axis thesis, the Koschmieder fog model, the driving plan and the band split.
- **E:** the level arm and the oracle decomposition.
- **B:** the ratio-artefact critique and the log-whitening baseline.

**Vote:** first A, second B. First, A (the merged A + C + D proposal).
- Its frozen row is the only one that has passed every pre-registered criterion on two disjoint held-out sets: 859 and 753 images.
- Over the activation CDFs it gains +0.075 / +0.081 on common, +0.053 / +0.053 on extra, and +0.105 / +0.111 at severity 1.
- Its mechanism is now stated without the ratio artefact.
- Its IV claims are cut back to what the depth-fog probes support.

Second, B.
- It is the strongest independent method: log-whitened energies give 0.886 / 0.862 on the newest held-out images, +0.065 / +0.051 over the CDFs.
- It is closed-form and cheap, and it is the natural fallback if the shape statistic fails at 5,000 images.
- But whitening divides out the coherent flattening (fog 0.71), and its novelty is "Rippel et al. plus a log".

C and D are merged into A, so a vote for them would count the same row twice. E's deployable row is the weakest held out (0.857 / 0.860).

**First experiment:** Score the frozen rows on the completed 5,000-image confirmation.
- **Cost:** no extra GPU. The pass already stores per-channel means and top-1% means for every image under all 96 conditions.
- **Timing:** it must finish before any driving data are scored, because it fixes the row for the driving test (A, or M1 under its frozen rule).
- **Report:** A on positions 200-4999, and separately on 1812-4999, which no panelist has read. Apply the M1 rule on 1059-4999.
- **Kill:** the 95% paired interval of A minus the activation CDFs on AUROC common contains 0 (K1). Or the interval of A minus Section 4 on common contains 0 (K2), in which case Section 4 stands.
- **Next, in the first agreed GPU window:** a Cityscapes-train bank plus Foggy Cityscapes, scored with that fixed row. Kill if the median paired s1 change in log(peak/mean) at beta 0.01 is >= 0.

**Checks run:** All checks ran on CPU (niced, CUDA hidden). Scripts and logs are in /tmp/claude-1000/-home-yuchen-YuchenZ-UE-philip-sa/3d52b916-b3ff-4806-a501-7f70cde31a29/scratchpad/detection/roundtable/panelist-A/r3/.

(1) r3_interim.py (149 s). An interim look at the frozen M1 test on 753 held-out images (positions 1059-1811) that no panelist had read. The pass had stored 1,812 of 5,000 images.

Rows (AUROC common / extra, severity-1 common, FPR95 common / extra):
- A: 0.902 [0.895, 0.910] / 0.864 [0.857, 0.871]; severity-1 0.820; FPR95 0.246 / 0.335.
- M1: 0.924 [0.917, 0.930] / 0.863 [0.856, 0.869]; severity-1 0.872; FPR95 0.212 / 0.351.
- E2: 0.892 / 0.875; severity-1 0.807; FPR95 0.267 / 0.304.
- E4 0.923 / 0.839; B 0.886 / 0.862; D 0.880 / 0.872; E 0.857 / 0.860; F 0.902 / 0.868.
- Section 4 0.844 / 0.870; CDFs 0.821 / 0.811.

Paired differences (common | extra | severity 1):
- M1 - A: +0.021 [+0.015, +0.028] | -0.0006 [-0.007, +0.006] | +0.052 [+0.043, +0.061]. By my frozen rule A stays; the formal decision comes at completion.
- A - CDFs: +0.081 [+0.074, +0.089] | +0.053 [+0.047, +0.059] | +0.111 [+0.102, +0.121].
- A - Section 4: +0.058 [+0.052, +0.066] | -0.006 [-0.012, -0.000].
- A - E2: +0.010 [+0.007, +0.014] | -0.011 [-0.014, -0.009].
- M1 - E2: +0.031 | -0.012.
- M1 - CDFs: +0.102 [+0.093, +0.112] | +0.053 [+0.044, +0.060] | +0.164.

Per family:
- A - Section 4 has intervals entirely below 0 on 9 families: elastic -0.056, spatter -0.034, brightness -0.024, speckle -0.019, saturate -0.018, snow -0.012, shot -0.007, gaussian -0.004, impulse -0.004.
- M1 - Section 4: fog +0.407, contrast +0.297, zoom +0.230, frost +0.087, pixelate +0.086, elastic +0.025, spatter -0.003, brightness -0.075, saturate -0.057, speckle -0.028.

(2) r3_depthfog_arms.py (206 s, slightly over the 3-minute guide; a screen). D's Koschmieder fog model on D's 24 Cityscapes val frames, at beta 0.005 / 0.01 / 0.02.
- With the COCO bank and stage-4 key: level 0.573/0.542/0.632, shape 0.635/0.684/0.804, signed flatter 0.644/0.693/0.693, M1 0.644/0.693/0.693, E2 0.613/0.635/0.766.
- With a 24-frame Cityscapes-train global reference (each AUROC +/-0.08): level 0.792/0.908/0.976, shape 0.776/0.894/0.979, signed flatter 0.760/0.814/0.840, M1-style max 0.826/0.920/0.976, E2-style sum 0.786/0.899/0.979.
- Paired, the share of channels that flatten: s1 0.74/0.73/0.70, s2 0.61/0.62/0.63, s3 0.49/0.50/0.49. The median s1 shift is -0.19/-0.30/-0.45 bank SDs; at s3 it is 0.00.

(3) r3_direction.py (about 20 s; the bank plus the 200-image screen; descriptive).
- On the clean bank, the slope of log ceiling on log level is 0.38/0.37/0.41 (IQR 0.28-0.52 at s1); r = +0.56/+0.64/+0.78; the slope is below 1 in 1.00/0.98/1.00 of channels.
- Severity-1 paired shifts at s1, as level / ceiling / share of channels flattening / share of energy on the uniform direction:
  - fog -0.061 / -0.280 / 0.84 / 0.39;
  - contrast -0.057 / -0.283 / 0.81 / 0.29;
  - zoom -0.047 / -0.268 / 0.83 / 0.35;
  - defocus -0.027 / -0.231 / 0.81 / 0.27;
  - glass -0.016 / -0.156 / 0.84 / 0.26.
- Share flattening / uniform-direction share for the other families:
  - pixelate 0.78 / 0.26;
  - elastic 0.78 / 0.17;
  - JPEG 0.70 / 0.13;
  - snow 0.58 / 0.03;
  - spatter 0.56 / 0.04;
  - noise 0.52-0.53 / 0.03-0.04;
  - brightness 0.41 / 0.06.

(4) r3_digits.py: the exact M1 - A difference on extra is -0.00063.

(5) progress.py: 1,716 images were stored at 17:32. Web searches were not possible: the session budget is exhausted.

## B (Statistics): Invisible per channel, obvious jointly: whitened log channel energies for corruption detection

**Thesis:** Corruptions are invisible channel by channel but obvious jointly: whitening a frozen detector's early log channel energies by their clean covariance beats comparing each channel's full activation distribution.

### Round 1 proposal

## Thesis
Corruptions are invisible channel by channel but obvious jointly: whitening a frozen detector's early log channel energies by their clean covariance beats comparing each channel's full activation distribution.

## Insight
**Why it is true.**
- **Content moves the early channel energies together.** At s2, in standardised log energies, the top 16 of 128 clean directions hold 84% of the clean variance.
- **Most of a corruption's shift looks like content.** 64–83% of each family's squared shift lies along those same directions, and there it drowns in content variation. A per-channel monitor must tolerate each channel's full marginal spread, and most of that spread is shared content.
- **The evidence is a residue.** Every corruption also moves the energies a little in directions where clean images barely vary. Whitening divides each direction by its clean spread and turns that residue into the score.
- **The split of the evidence.** On the common families, the quiet directions alone reach 0.879 AUROC. The 16 content directions, whitened, reach 0.832, and they matter more on the extra families (0.870).
- **One example.** At 5% clean false alarms on s2, a per-channel max-|z| monitor misses 94% of severity-1 JPEG images. The whitened score flags 44% of those.

**Why logs.**
- Channel energies are positive and vary multiplicatively (texture, contrast, haze). The log turns a joint gain into a linear direction that a covariance can model. It also cuts the skew at s3 from 1.12 to 0.53.
- The gain is not marginal Gaussianisation. On the screen (AUROC common), log whitening beats a rank-Gaussian copula by +0.012 [+0.005, +0.017], an L2 normalisation in the style of Mahalanobis++ by +0.023 [+0.013, +0.033], and raw whitening by +0.029 [+0.018, +0.040].
- Neither ingredient works alone. Logs without the covariance give 0.823, and removing only the gain gives 0.824.

**Why it is not obvious.**
- Every monitor published in this setting compares channels separately: the activation CDFs (Becker et al.), the clean-only NMD score, and Hashemi's per-neuron intervals.
- Becker et al.'s own joint baseline, RealNVP flows on pooled features of several layers, scored below their per-channel CDFs: 61.8 vs 68.0 at severity 3 on Faster R-CNN, as recorded in our literature review.
- L5 said one number per channel is nearly the whole distribution. This idea says the missing information lies in the correlations, not in the distribution.

**What it explains.**
- **Section 4 works mostly as a partial whitening.** The same kNN reference keyed on the stage itself, with no back, gets 0.836/0.862, against 0.841/0.870 when keyed on s4.
- **Once whitened, the back adds little.** It adds +0.008 [+0.006, +0.010].
- **The closed-form conditional on the back loses.** D²(s1–s4) − D²(s4) is 0.009 [−0.012, −0.006] below whitening the front alone. Once whitened, the back is not invariant: its own log-Mahalanobis separates at 0.744/0.703, while the unnormalised kNN reading gave 0.535.
- **L7 and L9 follow.** Unnormalised distances are dominated by the high-variance content channels (L7). Content is the nuisance, and whitening removes it without any key (L9).

**Lessons.**
- **Support it:** L2; L3 (channel identity, no sorting); L4/L5 (means, since adding top-1%/p99 to the whitened vector hurt: −0.039 extra); L8 (magnitude kept, s4 left out, per-stage z-scores); L7 and L9.
- **Could kill it:** L10, since brightness and saturate at severity 1 stay at 0.53. L11, since about 40 variants were screened; the held-out test below answers that.

## Method
In three sentences: take the log of each early channel's mean activation. Whiten each stage's vector with its clean covariance. The score is the squared norm, z-scored per stage on clean images and summed.
1. **Stages s1–s3:** the inputs of `res_layers[s].blocks[1].branch2a.conv` for s = 0..2 (post-ReLU; 64, 128 and 256 channels).
2. **Log energies:** y_c = log(mean over positions of a_c + 10⁻³). Offsets from 10⁻⁴ to 10⁻² give 0.874–0.877.
3. **Clean reference:** from N clean train images, standardise each channel (u), and fit the Ledoit–Wolf mean and covariance (μ̂_s, Σ̂_s). The shrinkage is analytic, so there is no knob.
4. **Stage score:** T_s = (u − μ̂_s)ᵀ Σ̂_s⁻¹ (u − μ̂_s) = Σ_c (u_c − E[u_c | u_<c])² / Var(u_c | u_<c). Each channel is judged against what the image's own other channels predict for a clean image, so the image is its own reference.
5. **Image score:** S = Σ_s (T_s − a_s)/b_s, with a_s and b_s from M = 500 clean images. The threshold is the clean (1 − α) quantile, which gives conformal false-alarm control.

**Optional ablation:** subtract the mean of the 50 clean scenes nearest in log s4 energies before whitening.

**Cost:** about 0.17 MFLOP (roughly 10 µs) and 0.35 MB per detector. The method needs no training, no labels and no corrupted data.

## Novelty
- **Becker et al., ICPR 2026:** per-channel CDF/EMD scores summed with the channels treated independently, and their own joint density lost to it. We supply the missing step (log geometry plus a shrunk Gaussian on the early stages), and it reverses their ordering.
- **NMD (Dong et al., CVPR 2022; verified):** correlations enter only through an LR or MLP trained on OOD or pixel-permuted pseudo-OOD samples. Ours is clean-only and closed form.
- **Mahalanobis (Lee et al., NeurIPS 2018; verified):** class-conditional, with a tied covariance on raw pooled features (low-level layers included), and layer weights fitted on in- and out-of-distribution validation data. Our raw single-class ablation is the clean-only form of that baseline, and the log beats it by +0.042 on held-out images.
- **Mahalanobis++ (Müller et al., ICML 2025; verified):** L2-normalises penultimate features. Its analogue here loses by 0.023.
- **Kamoi & Kobayashi 2020; PRISM (Krumpl et al., 2026; verified):** minor-subspace and residual energy for semantic or cross-domain OOD in classifiers. Our minor-subspace variant is a special case: 0.853–0.880 common, but only 0.837–0.850 extra.
- **Other methods:**
  - Gram: entrywise and unwhitened, and needs the full maps.
  - kNN-OOD: 0.798 on the screen.
  - DisCoPatch: trained, 0.760.
  - NIQE (a recollection): a Gaussian on hand-made pixel statistics of pristine images. Ours fits the detector's own energies.
- **Section 4:** explained above, and beaten on common (+0.036 held out). It stays ahead on extra (−0.013, from spatter and saturate).
- **Unchecked:** our searches found no log-energy Gaussian used for corruption detection, but they stopped at the session's search budget, so this is unverified.

## Decisive experiment
COCO, on the 4,800 held-out images from the confirmation's stored means. It needs no extra GPU, the bank and z-statistics stay fixed, and the rules are frozen in `panelist-B/PREREGISTRATION.md`.
- **The paper:**
  - primary − CDFs ≥ +0.03 on both groups, with intervals excluding 0 (about ±0.004 at that size);
  - the largest gains at severities 1–2;
  - FPR95 common ≤ 0.33 (the CDFs reach 0.413);
  - gains on frost, brightness, saturate and elastic.
- **Kill:** primary − CDFs includes 0 on AUROC common.
- **Geometry claim withdrawn:** log − raw whitening includes 0. The paper would then be Lee-style Mahalanobis, which is not paper-level.

The driving sets follow, under the same rules.

## Cheapest first test
**Run on the screen** (200 images, 95% paired bootstrap): 0.876/0.856.
- By severity: 0.782 / 0.849 / 0.893 / 0.920 / 0.938.
- FPR95: 0.287.
- Against the CDFs: +0.051 [+0.035, +0.066] common and +0.048 [+0.035, +0.060] extra.
- A 250-image bank gives 0.873/0.858.

**Pre-registered held-out test**, on the 308 images at positions 200–507 that the confirmation had stored. The confirmation's means for the 200 screen images equal the pilot's bit for bit. Result: 0.882/0.858.
- By severity: 0.791 / 0.857 / 0.897 / 0.923 / 0.941.
- Against the CDFs (0.824/0.810): +0.058 [+0.046, +0.069] common and +0.048 [+0.039, +0.057] extra.
- Against raw whitening: +0.042 [+0.034, +0.052] and +0.008 [+0.001, +0.016].
- Against Section 4 (0.845/0.870): +0.036 [+0.027, +0.047] and −0.013 [−0.021, −0.004].
- The back-kNN variant reaches 0.890/0.862.

**Losses (screen):** fog 0.706 vs 0.725, and zoom blur 0.816 vs 0.830.

## Figure 1
- **(a) The joint view:** two correlated s2 log energies over the 2,000 clean images, with their marginal histograms. A severity-1 JPEG image sits inside both marginals but off the clean ellipse. The caption: the per-channel monitor misses 94%, and whitening flags 44% of those.
- **(b) The spectrum:** clean variance and corruption shift per eigen-direction, showing the content-like bulk and the quiet residue.
- **(c) AUROC by severity** on 5,000 images: CDFs, Section 4 and ours.

## Fit for IEEE IV
- **A runtime camera monitor** on features the detector already computes, adding about 10 µs to 5.9 ms.
- **Calibration:** a few hundred clean frames per camera (250 suffice on the screen), with online covariance updates and conformal false-alarm control.
- **Driving plan:**
  - bank from Cityscapes train (2,975 frames);
  - tested on Cityscapes-C (19 × 5), Foggy Cityscapes and ACDC (to download), and nuImages;
  - report clean false alarms per scene type (night, tunnel);
  - test the L2 caveat with a whitened coarse grid on the fixed camera.

## Risks and cost
- **GPU:** none for COCO, since the confirmation's means are reused. Driving needs one backbone-only means pass per dataset, under an hour in total, in agreed windows.
- **New code:** about 40 lines plus tests.
- **What could go wrong:**
  - the held-out gain shrinks at 4,800 images;
  - the bank must come from the deployment domain;
  - on night frames with near-zero energies, the log offset matters;
  - the "just Mahalanobis" objection, which the geometry ablations and the reversal of Becker's flow result answer;
  - fog and zoom blur stay slightly below the CDFs, and brightness and saturate at severity 1 stay near chance (L10).

## Brainstorm, stress-tested
1. **Log-whitened energies:** chosen.
2. **Ancillary back, D²(front | back):** the self key nearly matches the s4 key, the linear form leaks (−0.009), and kNN adds +0.008. Runner-up.
3. **Matched-subspace residual:** needs r and is known (Kamoi; PRISM). A special case of 1.
4. **Whitened quantile functions (EMD plus covariance):** whitening the stored top-1%/p99 proxies hurt (−0.075 raw). Killed in that form.
5. **Cross-scale self-reference:** a second pass on a 2× downsampled copy. Blind to low-frequency fog and brightness. Runner-up.
6. **Conditioning on 4×4 cells:** the image means are already the coherent average of the cells (L2). Not pursued.
7. **Higher-criticism aggregation for sparse shifts:** needs decorrelated channels, so it can only be an add-on to 1.
8. **CUSUM over frames:** a system layer that cannot be tested on COCO. An IV extension.


**Quick check:** Everything ran on CPU over the stored 200-image pilot tables (200 images × 96 conditions; the 2,000-image clean bank and 500 z-statistics images). The scripts are check1–check13 in roundtable/panelist-B/. Numbers are AUROC common / extra, with 95% paired bootstrap intervals (1,000 draws). About 40 variants were screened on the same 200 images, so the screen numbers carry selection risk. The primary was frozen in PREREGISTRATION.md at 16:27, before any held-out image existed.

(1) The Section 4 controls:
- s4-keyed kNN: 0.841/0.870;
- the same kNN keyed on the stage itself (no back): 0.836/0.862;
- global own average: 0.811/0.837;
- PCA residual of the stage itself (no key, r=16): 0.862/0.849.

(2) Raw whitening (Ledoit–Wolf Mahalanobis per stage, s1–s3 z-summed): 0.847/0.857. Minus Section 4: +0.006 [−0.004, +0.015] / −0.013 [−0.021, −0.005].

(3) PRIMARY, log whitening: 0.876/0.856.
- Severities 1–5: 0.782/0.849/0.893/0.920/0.938; FPR95 common 0.287.
- Minus the CDFs (0.825/0.808 on the same images): +0.051 [+0.035, +0.066] / +0.048 [+0.035, +0.060].
- Minus Section 4: +0.035 [+0.023, +0.047] / −0.014 [−0.024, −0.004].
- Minus raw whitening: +0.029 [+0.018, +0.040] / −0.001 [−0.010, +0.008].
- Log offset 1e-4 / 1e-3 / 1e-2: 0.877 / 0.876 / 0.874 common.
- Log without the covariance: 0.823/0.831. Centred log-ratio (gain removed) without the covariance: 0.824/0.818.

(4) Controls:
- L2-normalised (Mahalanobis++-style): 0.853/0.855; log minus it +0.023 [+0.013, +0.033] / +0.001 [−0.007, +0.009].
- Rank-Gaussian copula: 0.865/0.850; log minus it +0.012 [+0.005, +0.017] / +0.006 [+0.001, +0.011].
- Square root: 0.860/0.850.
- Bank of 250 / 500 / 1,000 images: 0.873 / 0.873 / 0.876 common.
- Stage s2 alone: 0.877/0.856.

(5) The back:
- kNN on log s4 before whitening: 0.884/0.860, i.e. +0.008 [+0.006, +0.010] / +0.005 [+0.003, +0.006].
- Closed-form D²(s1–s4) − D²(s4): 0.870/0.845, against the joint D²(s1–s3) at 0.879/0.852: −0.009 [−0.012, −0.006] / −0.007 [−0.009, −0.004].
- The back's own log-Mahalanobis separates at 0.744/0.703.

(6) Negative: whitening [mean, top-1%, p99] per channel gives 0.773/0.812 raw and 0.847/0.817 in logs. In logs, against the means alone: −0.001 [−0.017, +0.014] / −0.039 [−0.053, −0.027].

(7) Per family, CDFs → primary (mean over severities):
- gains: frost 0.699→0.808, brightness 0.604→0.694, saturate 0.626→0.706, elastic 0.622→0.688, pixelate 0.786→0.910, JPEG 0.807→0.922;
- losses: fog 0.725→0.706, zoom blur 0.830→0.816;
- at severity 3/5: brightness 0.60/0.71→0.68/0.87, saturate 0.58/0.77→0.61/0.92;
- brightness and saturate at severity 1 stay at 0.530.

(8) Mechanism (s2, log):
- The top 16 clean directions hold 84% of clean variance and 64–83% of each family's squared shift.
- The quiet directions alone reach 0.879/0.850; the 16 content directions, whitened, reach 0.832/0.870.
- At 5% clean false alarms, a per-channel max-|z| monitor misses 73.5% of severity-1 images, and whitening flags 16.2% of those. For JPEG at severity 1: 94% missed, 44% of those flagged.

(9) HELD OUT, pre-registered: 308 images at positions 200–507 from runs/coco-baselines/test_convtu_means. Its 200 screen images equal the pilot's stored means bit for bit.
- Primary: 0.882/0.858; severities 0.791/0.857/0.897/0.923/0.941.
- Minus the CDFs (0.824/0.810): +0.058 [+0.046, +0.069] / +0.048 [+0.039, +0.057], so SUPPORTED by the pre-registered rule.
- Minus raw whitening: +0.042 [+0.034, +0.052] / +0.008 [+0.001, +0.016].
- Minus Section 4 (0.845/0.870): +0.036 [+0.027, +0.047] / −0.013 [−0.021, −0.004].
- Back-kNN variant: 0.890/0.862, i.e. +0.008 [+0.007, +0.010] / +0.005 [+0.004, +0.006].
- Section 4 minus the CDFs: +0.021 [+0.006, +0.036] / +0.060 [+0.048, +0.072].

These are 308 images, not 4,800, so the intervals are about ±0.01.

**Runners-up:**
- Ancillary back by kNN, after whitening: Subtract the mean of the 50 clean scenes nearest in log stage-4 energies before whitening the front: +0.008 [+0.007, +0.010] common held out (0.890/0.862). The closed-form Gaussian version leaks, because the whitened back moves under corruption (0.744), so this is an ablation, not a thesis.
- Cross-scale self-reference: Also run the backbone on a 2x-downsampled copy, and whiten the joint log energies of both scales against clean images. Fine-scale corruptions (noise, JPEG, pixelate) fade at half resolution, so the image supplies its own corruption-attenuated reference. It costs one quarter-size pass, needs the GPU, is untested, and is blind to low-frequency fog and brightness.

### Round 2: critiques written by B

**On A** (serious)

- Steelman: Fog, contrast and blur remove visibility. They lower each early channel's ceiling relative to its bulk, and content mostly does not. So the log peak-to-mean ratio, read against content-matched clean scenes, catches exactly the families that level monitors miss. On 678 unseen images it is the best deployable row (0.889/0.854, severity-1 0.806, FPR95 0.257), and it passes every criterion A pre-registered.
- Objection: Two objections. (1) Serious for the IV story. The mechanism assumes the transmission is the same everywhere in the image. That is what imagecorruptions' fog does: one global contrast scale plus a fractal overlay. Real Koschmieder fog depends on depth, t = exp(-beta d). The strongest 1% of s1 responses sit on near, high-contrast edges such as cars and lane markings, which keep t of about 0.9 at beta = 0.01 and 10 m. The far texture, which makes up the bulk, is erased. Under real fog the peak share can therefore rise, not fall. That would invert A's Figure-1 arrows, the claim that fog is 'a textbook lowering of the ceiling', and A's pre-registered prediction that Foggy Cityscapes separates on shape while level monitors struggle. The frozen two-sided |delta| row may still separate, but only by the size of the change, not by its stated direction. (2) Fixable. The 'level-peakiness trade-off' is the spurious correlation of a ratio with its own denominator (Pearson 1897, from memory, not verified), not a property of scenes. If log top were independent of log mean, corr(log m, log pi) would be -0.817/-0.870/-0.879 at s1/s2/s3. The observed values are weaker, -0.738/-0.847/-0.876, because top and mean are positively coupled (rho = 0.56/0.64/0.78). So the data show only that the ceiling varies less than the level. For the same reason, K3 ('an anti-correlation weaker than -0.5 kills the mechanism') cannot fire while sd(log top) < sd(log mean). A's explaining-away argument survives, because it needs only the correlation, whatever causes it.
- Failure scenario: On Foggy Cityscapes at beta = 0.005-0.01, the median s1 delta-log-pi of each fogged frame against its own clean frame is zero or positive, because near edges dominate the top 1%. One-sided 'flatter' variants then fail. The two-sided row separates only as far as the drop in the bulk level allows. The paper's mechanism figure would be contradicted on the very dataset chosen to show it off.
- Evidence: My check on 678 held-out images, positions 200-877 (pre-registered at 17:08 in panelist-B/r2/PREREG_R2.md, before any of these rows was computed). A reaches 0.889/0.854; severities 1-5 0.806/0.867/0.904/0.927/0.943; FPR95 0.257/0.342. Against the CDFs: +0.072 [+0.064, +0.081] / +0.051 [+0.044, +0.059], severity-1 +0.100 [+0.090, +0.111]. Against Section 4: +0.050 [+0.042, +0.058] / -0.010 [-0.016, -0.003]. At severity 1, fog/contrast/zoom reach 0.795/0.816/0.883 against the CDFs' 0.638/0.668/0.787, i.e. +0.157/+0.148/+0.096, all at least +0.08. K1 and K2 do not fire, and the screen-to-held-out shrinkage is only 0.897 to 0.889, which answers the L11 worry. Ratio check on the bank: observed -0.738/-0.847/-0.876 against -0.817/-0.870/-0.879 under independence. The real-fog risk is untested: it needs Foggy Cityscapes, which is not on disk.
- Would change my mind: For (1): compare Foggy Cityscapes and ACDC fog with the same clean frames. If the median s1-s3 delta-log-pi is negative, and the frozen row's AUROC at beta = 0.005 is at least the level row's, objection (1) is dead. For (2): a clean-image test where the ratio's anti-correlation is stronger than its value under independence would show a genuine trade-off.

**On C** (fixable)

- Steelman: One gain-invariant number per channel, the log ratio of the top-1% mean to the mean, compared with the same channel in content-matched clean scenes, beats full activation CDFs by +0.072/+0.051 on unseen images. C's honest topology negatives show why the right invariant here is a ratio, not a Betti number.
- Objection: C is the same row as A, and C's mechanism sentence is contradicted by the data. The claim is 'content scales both ends together, so the ratio cancels much of the content'. In fact the clean spread of log pi is 0.89/0.79/0.70 of log m's at s1/s2/s3, so the ratio removes only 11-30% of the clean spread. The s4 key still predicts 62/67/67% of the shape's held-out clean variance, against 73-82% for the level. The ratio therefore cancels little content, and the conditioning is still needed (+0.048 on the screen, global 0.849 against conditioned 0.897). The gain comes from the signal side: corruption moves the shape coherently across channels. At s1, severity 1, the squared cosine between the mean shift and the all-ones direction is 0.59/0.52/0.52 for fog/contrast/zoom with the shape, against 0.22/0.26/0.11 with the level. If the thesis says the ratio cancels content, the reviewer will ask why it then needs 2,000 neighbours.
- Failure scenario: On a fixed-camera driving set such as Cityscapes, a reviewer asks for the global peak share as the headline, since content varies less (C's plan reports both). If C's mechanism were right, global and conditioned would be about equal. On COCO the global row loses 0.048, so the claimed invariance is not what makes the method work, and a paper built on it overclaims.
- Evidence: Held out: C reaches 0.889/0.855 and A 0.889/0.854, identical to three decimals, so eps does not matter. C's own kill tests pass: against Section 4 on common, +0.050 [+0.042, +0.058] (the threshold is +0.015); against the CDFs on extra, +0.051 [+0.044, +0.059]. C's pre-registered secondary, gain + shape, reaches 0.885/0.864, severity-1 0.799, FPR95 0.269. Against shape alone it is -0.005 [-0.007, -0.003] on common and +0.010 [+0.008, +0.011] on extra. Mechanism numbers: sd ratio 0.888/0.794/0.698; shape R^2 from s4 0.615/0.666/0.669; squared cosines as above (panelist-B/r2/mechanism.log and shape_oracle.log).
- Would change my mind: If on Cityscapes (fixed camera) the gap between conditioned and global peak share shrinks to 0.01 or less, the 'cancels content' framing would be fine for IV.

**On D** (fixable)

- Steelman: Level and crest read two different physical effects: how much of the frame is textured, and how peaked the response is. Judging both against content-matched clean scenes catches visibility loss (fog, contrast, blur) without giving up the level's strength on noise, spatter and saturate. D is also the only proposal that flagged the real-fog risk to every shape statistic.
- Objection: The implementation uses the wrong geometry, and a trivially different variant dominates it. D scores the crest as a raw ratio and the level in raw units. Both are multiplicative quantities, yet they are scored with additive spreads. On the 678 held-out images, the raw crest alone gives 0.872/0.847 against 0.889/0.854 for the log ratio. C's pre-registered gain + shape uses the log level and the log ratio with the same key and the same aggregation, and it beats D's level + crest by +0.014 [+0.012, +0.017] on common and ties on extra (-0.000 [-0.002, +0.001]). D's own criterion 'fog >= 0.75' fails on held-out (0.743), while its kill test passes (+0.031 [+0.027, +0.036] over level only). The equal weight on three level terms also halves the crest's weight, which is what costs D on common.
- Failure scenario: A reviewer replaces t/m with log(t/m), and the raw level with its log, in one line, and gains +0.014 on common at no cost on extra. D's paper would then report a dominated row as its method.
- Evidence: Held out (positions 200-877): D reaches 0.870/0.865, severity-1 0.784, FPR95 0.294/0.327. Against the CDFs: +0.053 [+0.045, +0.063] / +0.061 [+0.054, +0.069]. Against Section 4: +0.031 [+0.027, +0.036] / +0.000 [-0.004, +0.005]. Family means: fog 0.743, contrast 0.859. Crest alone: 0.872/0.847. C's gain + shape minus D: +0.014 [+0.012, +0.017] common, -0.000 [-0.002, +0.001] extra, +0.015 [+0.012, +0.018] at severity 1. Logs also lift the conditioned level: 0.868 against Section 4's 0.839 on common.
- Would change my mind: If on Cityscapes-C or Foggy Cityscapes the raw crest beats the log ratio, for example because night frames with near-zero means make the log unstable, the raw geometry would be justified for IV.

**On E** (serious)

- Steelman: Given the same scene's clean look, the detector's early channel means separate severity-1 fog and contrast at 0.93 on unseen images. So the weak spots of every single-image monitor are aliasing with natural scene variation, not blindness. This is a quantified ceiling: it shows the field where the headroom is, and that only brightness and spatter (oracle about 0.65) are truly blind.
- Objection: The decisive number needs the corrupted image's clean twin, so it measures a paired change detector, not a corruption detector under the paper's rules. 'A1, leak-free' is an oracle too, because its neighbour is found with the clean twin's key. The deployable row (a) is the weakest proposal row on held-out: 0.846/0.850, only +0.006 [+0.000, +0.012] over Section 4 on common, -0.014 [-0.019, -0.010] on extra, and 0.043 below the shape row. E's practical claim, that a single image cannot tell a hazy scene from a fogged one, holds for channel means but not for every statistic. The shape statistic on a single image reaches 0.795 at severity-1 fog, against 0.564 for (a). Given the clean twin, the shape is nearly as informative as the means: 0.968 against 0.976 on common, and 0.963 against 0.933 on fog. Choosing the right statistic therefore halves the aliasing gap (0.369 to 0.168 at severity-1 fog, 0.130 to 0.079 on common), so 'blindness' should be measured with the best statistic. The driving remedy (b) fails for conditions present from the first frame: driving into fog, at night, or with a lens soiled at start-up, which are IV's core cases.
- Failure scenario: At IV, the reviewer asks for the deployable single-frame result. E's method row is Rippel-style Mahalanobis plus SPADE retrieval, both prior work as E says, at 0.846, and the shape statistic beats it by 0.043. The video variant scores a car that leaves the garage into fog only against fogged frames, so its reference is already corrupted and it never alarms.
- Evidence: Held out (positions 200-877), E's criteria pass. (c) reaches 0.976/0.948, with severity-1 fog 0.933 and contrast 0.925 against a kill threshold of 0.80. (c) - (a) at severity 1: fog +0.369, contrast +0.348, saturate +0.298, zoom +0.298, frost +0.272, all at least 0.20. A1 recovers 0.07 of the gap for fog and 0.03 for contrast, under a third, but 0.49 for zoom blur. (a) - CDFs: +0.029 [+0.017, +0.040] / +0.047 [+0.037, +0.057]. The same decomposition for the shape (shape_oracle_r2.py): single image 0.889/0.854, severity-1 0.806; ridge oracle on the twin 0.968/0.930/0.929; kNN on the twin 0.917/0.873.
- Would change my mind: Either of two results: a deployable single-frame reference that closes at least half of the shape statistic's remaining gap (0.889 to at least 0.93 on common), or an onset test on driving video in which the previous-frame reference keeps clean false alarms at the single-frame level.

**Defense of own:** Held-out verdict on 678 images under my pre-registered rule: supported, but dominated.

My primary reaches 0.873/0.851; severities 1-5 0.779/0.846/0.888/0.915/0.934; FPR95 0.289/0.343.
- Against the CDFs: +0.055 [+0.047, +0.064] / +0.048 [+0.040, +0.055]. The rule (at least +0.03 on both, with intervals excluding 0) is met.
- Log minus raw whitening: +0.037 [+0.031, +0.043], so the geometry claim stands.
- The predicted gains appear, against the CDFs: frost 0.806 vs 0.689, brightness 0.692 vs 0.596, saturate 0.705 vs 0.620, elastic 0.681 vs 0.612.
- My Round-1 number on 308 images (0.882/0.858) shrank to 0.873/0.851 with more images.

The strongest attack is not "just Mahalanobis" but this:
- The shape row beats mine by +0.017 [+0.012, +0.022] on common and +0.027 [+0.020, +0.034] at severity 1, and ties on extra (+0.004 [-0.000, +0.009]).
- Adding my score to the shape buys nothing on common (M1 - A = -0.001 [-0.004, +0.002]) and +0.003 [+0.000, +0.005] on extra.
- Whitening the conditioned shape hurts (M2 - A = -0.034 [-0.039, -0.029]). So "invisible per channel, obvious jointly" is not a general law: it holds for the level only.

I accept this, and B should not be the lead. What survives, and what I would change:
1. **A statistical design rule** that explains the shape proposals and my own results: whiten what content moves incoherently, and average what corruption moves coherently.
   - The level's corruption residue is incoherent. The squared cosine between the shift and the all-ones direction is at most 0.26, and the all-ones direction carries at most 0.08 of the Mahalanobis signal. So whitening or conditioning is needed: 0.873 whitened and 0.868 for the conditioned log level, both held out; the global average gave 0.811 on the screen.
   - The shape shift is coherent at s1, with squared cosines of 0.40-0.59 for fog, contrast and the blurs. So the channel average beats whitening by 0.034.
2. **Logs for every multiplicative statistic.** The log ratio beats D's raw crest by 0.017, and the log level beats raw Section 4 by 0.029 (0.868 vs 0.839). The log is not specific to whitening.
3. **Conformal false-alarm control and held-out discipline.** Positions 200-877 have now been seen for every frozen row, so any Round-3 revision must be judged on positions 878 and above only.

I would withdraw B as a standalone paper and contribute points 1-3 to the shape paper.

**Best other proposal:** A (Mechanism): the content-conditioned log peak-to-mean ratio at s1-s3. C's row is identical to three decimals. On 678 unseen images it is the best deployable row: 0.889/0.854, severity-1 0.806, FPR95 0.257, +0.072/+0.051 over the CDFs and +0.050 over Section 4 on common. It passes every criterion A pre-registered, including severity-1 fog, contrast and zoom blur each more than 0.08 above the CDFs.

**Merge suggestion:** Scope check: all five proposals stay on corruption detection, and none drifts into predicting performance drop.

Merge A, C and D into one paper, with E's analysis and B's design rule.

**Thesis.** "Corruption flattens every early channel at once; scenes do not." This is A, C and D's shape idea in D's two-axis framing. Drop the "trade-off" and "cancels content" wording, which the data contradict.

**Method.** Frozen as A = C:
- conditioned log peak share at s1-s3;
- s4-means key, k = 50;
- channel-averaged |delta|/sigma, z-scored per stage and summed.

The pre-registered secondary is C's gain + shape (log level + log shape), the best-balanced row on held-out: 0.885/0.864, FPR95 0.269. It is there for the extra families.

**Analysis figure (E).** With the clean twin, the detector sees mild fog through both statistics (about 0.97). On a single image, the level aliases with scene content (0.846). The shape escapes half of that aliasing (0.889; the severity-1 fog gap falls from 0.369 to 0.168). Brightness, spatter and saturate stay near blind even with the twin (0.59-0.80), which quantifies L10.

**Design rationale (B).**
- Average shifts that move every channel the same way; whiten shifts that hide among content directions.
- Use logs for every multiplicative statistic.
- Report M2 (-0.034) as the negative control.

**Decisive tests.**
1. COCO positions 878-4,999: about 4,100 images, roughly ±0.004. Do not use the 678 already seen. The running pass already stores top, so this needs no new GPU time.
2. Foggy Cityscapes, scored two-sided, with the sign of the peak-share change registered as an open question (D's real-fog risk), plus ACDC fog and night.

The held-out scores of all 14 rows are in panelist-B/r2/heldout_scores.npz.

**Ranking:**
1. A: Best deployable row on unseen images (0.889/0.854, severity-1 0.806, FPR95 0.257). It passes every criterion it pre-registered, and its explaining-away analysis explains why Section 4 fails on fog.
2. C: Same row as A (0.889/0.855). Its pre-registered gain + shape is the best-balanced row (0.885/0.864), and its topology negatives are honest. The mechanism wording needs fixing.
3. D: Same insight, wrong geometry: the log version dominates it by 0.014 on common, and it misses its own fog >= 0.75 criterion (0.743). It alone flagged the real-fog risk.
4. B: Supported on held-out (+0.055/+0.048 over the CDFs, +0.037 for the log), but the shape dominates it (-0.017 on common, -0.027 at severity 1) and it adds nothing to the shape. Its thesis holds for the level only.
5. E: Its decomposition holds on held-out and makes the best analysis figure, but its deployable row is the weakest (0.846/0.850), its headline needs the clean twin, and the shape halves the aliasing E calls unavoidable.

**Checks run:** All checks ran on CPU (niced, 4 threads); scripts are in panelist-B/r2/.

**Check 0 (check_top_equal.py).** All 878 confirmation files carry top_s1-s4. On the 200 screen images, the confirmation's means and top equal the pilot's bit for bit (worst relative difference 0.0).

**Smoke (heldout_r2.py --smoke).** Run on the 200 screen images, every frozen row reproduces its author's screen numbers:
- A and C 0.897/0.859; D 0.876/0.870; crest 0.876/0.849;
- E(a) 0.855/0.860; raw whitening 0.847/0.857; E(c) 0.982/0.958; A1 0.868/0.860;
- B 0.876/0.856; B2 0.884/0.860; Section 4 0.841/0.870; CDFs 0.825/0.808.

**Held-out run.** Positions 200-877, 678 images, pre-registered in r2/PREREG_R2.md at 17:08 before any of these rows was computed. AUROC common/extra, severity-1, FPR95 common:

| Row | Common / extra | Severity 1 | FPR95 common |
|---|---|---|---|
| A | 0.889/0.854 | 0.806 | 0.257 |
| C | 0.889/0.855 | 0.806 | 0.257 |
| D level + crest | 0.870/0.865 | 0.784 | 0.294 |
| D crest alone | 0.872/0.847 | 0.784 | 0.285 |
| E(a) | 0.846/0.850 | 0.768 | 0.337 |
| Raw whitening | 0.836/0.846 | 0.756 | 0.355 |
| E(c) oracle | 0.976/0.948 | 0.941 | 0.071 |
| E A1 | 0.867/0.859 | 0.777 | 0.403 |
| B | 0.873/0.851 | 0.779 | 0.289 |
| B2 | 0.881/0.855 | 0.791 | 0.271 |
| M1 = z(B) + z(A) | 0.888/0.857 | 0.803 | 0.253 |
| M2 (whitened shape) | 0.856/0.827 | 0.763 | 0.320 |
| Section 4 | 0.839/0.864 | 0.760 | 0.360 |
| CDFs | 0.817/0.803 | 0.706 | 0.416 |

**Paired bootstrap** (1,000 draws, seed 44; common / extra):
- A - CDFs: +0.072 [+0.064, +0.081] / +0.051 [+0.044, +0.059]
- A - Section 4: +0.050 [+0.042, +0.058] / -0.010 [-0.016, -0.003]
- A - B: +0.017 [+0.012, +0.022] / +0.004 [-0.000, +0.009]
- D - Section 4: +0.031 [+0.027, +0.036] / +0.000 [-0.004, +0.005]
- B - CDFs: +0.055 [+0.047, +0.064] / +0.048 [+0.040, +0.055]
- B - raw whitening: +0.037 [+0.031, +0.043] / +0.005 [-0.000, +0.010]
- M1 - A: -0.001 [-0.004, +0.002] / +0.003 [+0.000, +0.005]
- M2 - A: -0.034 [-0.039, -0.029] / -0.028 [-0.032, -0.024]
- E(a) - Section 4: +0.006 [+0.000, +0.012] / -0.014 [-0.019, -0.010]

**C's pre-registered gain + shape** (mechanism_r2.py) reaches 0.885/0.864, severity-1 0.799, FPR95 0.269. Paired differences:
- minus shape alone: -0.005 [-0.007, -0.003] / +0.010 [+0.008, +0.011]
- minus D: +0.014 [+0.012, +0.017] / -0.000 [-0.002, +0.001]

**Ratio artefact** (bank, s1/s2/s3):
- corr(log m, log pi): observed -0.738/-0.847/-0.876; if log top were independent of log mean, -0.817/-0.870/-0.879;
- corr(log m, log t): 0.561/0.644/0.776;
- sd(log pi)/sd(log m): 0.888/0.794/0.698.

**Shift geometry** (s1, severity 1, shape vs level):
- squared cosine with the all-ones direction: fog 0.59 vs 0.22, contrast 0.52 vs 0.26, zoom 0.52 vs 0.11, defocus 0.43 vs 0.04;
- share of the Mahalanobis signal on the all-ones direction: 0.13-0.23 for the shape, at most 0.08 for the level everywhere.

**Shape decomposition** (shape_oracle_r2.py):
- R^2 of the shape from the s4 key: 0.615/0.666/0.669;
- single image (A): 0.889/0.854, severity-1 0.806;
- ridge from its own key: 0.847/0.811;
- ridge oracle from the clean twin: 0.968/0.930, severity-1 0.929, with fog 0.963 vs 0.933 for the means oracle;
- kNN oracle on the twin: 0.917/0.873.

**Contamination.** Positions 200-877 have now been seen for all of these rows.

### Round 3: rebuttal by B

- **To A** (concede): Whitening suits the level but not the shape. The shape's corruption shift is coherent: it lies along the high-variance clean first principal direction (PC1), and whitening divides that direction away (whitened shape 0.848/0.826 against 0.897/0.859; joint [level, shape] whitening -0.014). B ties the CDFs at severity-1 fog and contrast. A - B = +0.019 common / +0.005 extra / +0.029 at severity 1. B is Rippel 2020 plus a log. — Conceded, and now tested on images no panelist had used.

(1) The geometry is right.
- My own Round-2 mechanism check found the same thing. At s1, the shape's squared cosine with the all-ones direction is 0.40-0.59 for fog, contrast and the blurs, against at most 0.26 for the level.
- On the clean bank, the all-ones direction holds only 12% of the conditioned shape residual's variance at s1 (r3_content_share.py).

(2) My Round-1 negative, 'adding top-1%/p99 to the whitened vector hurt', was this geometry artefact, not evidence against the tail. I withdraw it as support for L4/L5.

(3) Whitening only the complement of PC1 is not the right remedy: the uniform direction should be tested, not whitened.
- Your signed flatter arm is that one-degree-of-freedom directed test (O'Brien, Biometrics 1984, abstract verified).
- Test set: 911 images at positions 1059-1969, pre-registered in panelist-B/r3/PREREG_R3.md before any of these rows was scored on them.
- On them, M1 = max(flatter, level) minus A primary is +0.021 [+0.015, +0.027] common, -0.001 [-0.007, +0.004] extra and +0.051 at severity 1. Fog at severity 1 reaches 0.959 and contrast 0.947 (CDFs 0.633 / 0.661).
- The other place whitening could go also fails. Within-channel whitening (W2) minus A is -0.002 [-0.004, +0.001] common and -0.011 [-0.014, -0.009] extra.

(4) Rippel et al. is the right citation; leaving it out was my mistake.

I withdraw B and back your M1.
- **To C** (concede): C's per-channel conditioned log level (the X row) matches B: -0.005 common and +0.020 extra on 308 images. So per-channel monitors failed because of their global reference, not because they score channels separately, and the log is the active ingredient. Whitening suppresses the coherent fog, contrast and blur shift: whitening the shape (WSH) costs -0.040. B is Rippel plus a log. — Conceded: 'invisible per channel' is false.

- On the 911 fresh images, whitening and your per-channel conditioned log level trade off rather than one beating the other: B - X = +0.010 [+0.007, +0.014] common and -0.013 [-0.017, -0.010] extra. Your condition (whitening ahead on both) fails.
- The correct statement is 'invisible against a global reference'. Whitening and the kNN mean are two estimates of the same conditional reference: one conditions on the image's other channels, the other on the back.
- The log is real for the level on its own: X - S4 = +0.032 [+0.027, +0.036] common and +0.007 [+0.003, +0.011] extra.
- Inside the max it washes out: MX - M1 = +0.003 [-0.001, +0.007] common and -0.001 [-0.004, +0.002] extra.
- On fog: whitening the fog-carrying statistic within each channel lifts severity-1 fog above the two-sided share (W2 0.890 vs A 0.807). But the directed arm does far better (M1 0.959), and W2 costs 0.011 on extra. So whitening does not earn a place.
- **To D** (concede): Gain and blur flatten all channels along the dominant clean direction, which whitening divides out. Whitening the log share costs 0.04-0.05. Held out, B is beaten by the conditioned shape and adds nothing over the CDFs at severity-1 fog and contrast. The novelty reduces to Rippel plus a log. — Conceded.

- No whitened variant that keeps the common mode beats the A/C row on unseen images:
  - W2 - A: -0.002 [-0.004, +0.001] common, -0.011 [-0.014, -0.009] extra;
  - FW2 = max(flatter, W2), minus A: +0.014 [+0.010, +0.018] common, but -0.027 [-0.032, -0.022] extra.
- What keeps the common-mode signal is a directed test, and A's flatter arm is one.
- The fog probes are why I back the max of a flatter arm and a level arm rather than the shape alone:
  - yours: level 0.903 vs share 0.877 at beta = 0.01;
  - A's: s1 level -1.03 vs shape -0.50 SD at beta = 0.02.
- Your two-axis framing is the structure of that design. Your Foggy Cityscapes kill is the IV test I propose to run first.
- **To E** (concede): Without the log, B is Rippel et al. 2020, and B does not cite it. The thesis is contradicted on held-out data: a per-channel diagonal score of the right statistic beats B, and whitening that statistic makes it worse. — Conceded on both counts.

- Without the log, B is Rippel et al. 2020 applied at the detector's stages. I failed to cite it.
- The thesis is contradicted. On 911 fresh images the per-channel shape still beats B: A - B = +0.016 [+0.011, +0.020] common, +0.001 [-0.003, +0.004] extra.
- Your condition fails. Whitened [log level, log peak share], taken within each channel, minus level + shape (C2) is +0.004 [+0.001, +0.008] common and -0.020 [-0.023, -0.017] extra.
- Used as the level arm inside the max, whitening only trades extra for common: +0.006 / -0.008, exploratory, because its arms were already seen.

**Status:** merged (merged with: A, D)

**Final proposal:**

**Status: merged with A and D.** I withdraw my Round-1 thesis (whitened log energies). The paper I back is A's shape idea in A's own frozen max form (M1, `panelist-A/r2/PREREGISTRATION_R2_M1.md`), with D's two-axis framing and IV kill, and with the statistics that explain why it works. Nothing in M1 is retuned.

# Flatter all at once: a directed test for corruption inside a frozen detector

## Thesis
A corruption makes a frozen detector's early channels flatter all at once, while scene content moves them in scattered directions. So one signed average, "flatter than clean scenes like this one", is the most sensitive single-image corruption test, and a level test catches the corruptions that add structure instead.

## What I withdraw, and what survives
**Withdrawn: "invisible per channel, obvious jointly."** The test used 911 images that no panelist had used (positions 1059–1969, pre-registered in `panelist-B/r3/PREREG_R3.md`).
- The per-channel conditioned log level ties whitening: B − X = +0.010 [+0.007, +0.014] common, −0.013 [−0.017, −0.010] extra.
- Whitening never helps the shape:
  - within each channel, W2 − A is −0.002 [−0.004, +0.001] common and −0.011 [−0.014, −0.009] extra;
  - across channels it cost −0.034 in Round 2.
- As the level arm inside the max, whitening only trades extra for common: +0.006 [+0.001, +0.010] / −0.008 [−0.012, −0.004]. This is exploratory, because its arms had already been seen.

**Survives, as the design rule of the merged paper.**
- Test a coherent shift with a directed test, and an incoherent one with an omnibus or dense test (O'Brien, *Biometrics* 40:1079–87, 1984; abstract verified).
- My Round-1 whitening was the omnibus test: right for the level, wrong for the shape.

**Also survives:**
- logs for multiplicative statistics (the peak share is a log ratio);
- conformal false-alarm control, now checked;
- held-out discipline.

## Insight
**Corruption moves the shape along one direction.**
- For the conditioned log peak share at s1, the all-ones direction holds 12% of the clean (content) residual variance (5% at s2, 3% at s3).
- It holds 40–59% of the squared paired shift of fog, contrast, zoom, defocus and motion blur at severity 1 (Round 2, positions 200–877).
- At s1 and severity 1, the channel average moves +2.84 / +2.72 / +2.52 / +3.21 clean SDs under fog / contrast / zoom / defocus (fresh images).
- The physics (A): blur spreads an edge's response, and fog and contrast shrink the structured part against the fixed BatchNorm offsets. Both lower every channel's ceiling relative to its bulk.

**So the test must be directed.**
- A one-degree-of-freedom test along that direction spends its power where the shift is. The omnibus (Mahalanobis, Rippel) test spreads it over 64–256 directions.
- Across 38 (family, severity) cells on the fresh images, the predicted power advantage of the directed test over the omnibus test ranks the observed AUROC advantage with Spearman ρ = +0.78 (p ≈ 1e−8).
- The ranking holds, but the Gaussian calibration does not: sign agreement is only 0.50, because the omnibus loses more in practice than a χ² calculation predicts.
- The share of the omnibus signal that lies on the all-ones direction (λ₁/λ_W) is 0.20–0.29 at s1 for fog, contrast, zoom and defocus, and ≤ 0.01 for noise, snow and spatter.

**Two arms, because noise adds peaks.**
- Noise moves the channel average the other way (−0.60 to −1.01 SD), so the one-sided arm alone scores only 0.702 on the common families.
- The noise families, snow, spatter and saturate move the level instead.
- Each family moves mainly one arm, so the arms are combined by a max, not a sum: max minus sum = +0.024 [+0.018, +0.029] on extra (MX − E4).

**The reference.**
- Both arms subtract the mean of the 50 clean scenes nearest in stage-4 means (Section 4).
- A's explaining-away mechanism applies: a dimmed image recruits darker neighbours, which absorbs the level shift but exposes the shape.

**Why it is not obvious.**
- None of the monitors in Section 6 tests a direction: they sum per-channel distances (CDFs, NMD) or use a joint distance (Mahalanobis, Rippel, Gram, kNN).
- NAP's one-sided global max/mean ratio, the nearest statistic, is below chance here (0.30; A, E). The direction appears only against content-matched clean scenes.
- L4 and L5 hold for unnormalised statistics against a global reference.

## Method (three sentences)
1. For each channel of s1–s3 (the post-ReLU inputs of `res_layers[s].blocks[1].branch2a.conv`), compute the mean m and the log peak share π = log(top-1% mean + 1e-6) − log(m + 1e-6), and find the 50 clean bank images nearest in bank-standardised stage-4 means.
2. The flatter arm is, per stage, the channel average of −(π − neighbours' mean)/sd_bank(π); the level arm is Section 4's channel average of |m − neighbours' mean|/sd_bank(m); each stage is z-scored on 500 clean images, the stages are summed, and each arm is re-z-scored.
3. The score is the larger of the two arms, and the alarm threshold is the clean calibration images' (1 − α) quantile, which gives conformal false-alarm control.

The method is training-free, clean-only and label-free. Its cost equals A's row: one top-k per channel plus a 2,000 × 512 neighbour search.

## Novelty
- **Becker et al., ICPR 2026:** a per-channel EMD against a global CDF, not a directed test. On the fresh images, M1 − CDFs = +0.103 [+0.094, +0.111] common, +0.052 [+0.045, +0.059] extra and +0.164 [+0.153, +0.174] at severity 1.
- **Lee et al. 2018 and Rippel et al. 2020 (verified by E):** omnibus Gaussians, shown here to be the wrong test for a coherent shift.
- **NAP (verified by C and D), ASH and SCALE (verified by A and D):** the same ratio, but global, one-sided or taken across channels, for semantic OOD. None has a reference or a directed test across channels.
- **SPADE (verified by E):** the retrieval step; Section 4 is its form inside the detector.
- **O'Brien 1984:** the directed-test principle, from multiple-endpoint trials. Treating channels as the endpoints, against content-matched clean scenes, for corruption detection: our searches found no prior use, but they stopped at the session's search budget.

## Decisive experiment
COCO positions above 1969 once the running pass ends (about 3,030 images, unseen by me), with the same bank and z-statistics images. These rules are new, written after the interim look, and they apply only to those positions.
- **K1, kill:** M1 − CDFs on common has an interval containing 0.
- **K2, the directed arm adds nothing:** M1 − A primary on common has a lower bound below +0.010. A primary then leads.
- **Non-inferiority on extra:** M1 − A primary on extra must have a lower bound above −0.010. Otherwise report both rows, with A primary leading.
- **Paper bars:**
  - common ≥ 0.90 and severity-1 common ≥ 0.85;
  - FPR95 common ≤ 0.25;
  - fog and contrast at severity 1 ≥ 0.90;
  - clean false alarms at α = 0.05 within [0.035, 0.065].
- **Secondaries, not for selection:** MX, MFB, W2, E4, C2.
- **IV:** Foggy Cityscapes, with D's kill (b); see the first experiment.

## Cheapest first test (done, CPU, 40 s)
911 fresh images, positions 1059–1969. The confirmation's files equal the pilot bit for bit on the screen images.

| Row | Common | Extra | Sev. 1 | FPR95 common |
|---|---|---|---|---|
| M1 = max(flatter, level) | **0.922** | 0.861 | **0.869** | **0.216** |
| MX (log level arm) | 0.925 | 0.860 | 0.875 | 0.202 |
| A primary (shape) | 0.901 | 0.862 | 0.818 | 0.251 |
| C2 (log level + log share) | 0.895 | 0.871 | 0.808 | 0.264 |
| B Round-1 primary | 0.885 | 0.861 | 0.790 | 0.276 |
| Section 4 | 0.842 | 0.868 | 0.763 | 0.358 |
| Activation CDFs | 0.819 | 0.808 | 0.705 | 0.417 |

**M1's other numbers:**
- by severity 1–5: 0.869 / 0.910 / 0.931 / 0.944 / 0.955;
- M1 − A primary: +0.021 [+0.015, +0.027] common, −0.001 [−0.007, +0.004] extra, +0.051 at severity 1.

**Severity 1, M1 / A primary / CDFs:**

| Family | M1 | A primary | CDFs |
|---|---|---|---|
| fog | 0.959 | 0.807 | 0.633 |
| contrast | 0.947 | 0.829 | 0.661 |
| glass blur | 0.926 | 0.848 | 0.672 |
| JPEG | 0.899 | 0.831 | 0.630 |
| pixelate | 0.861 | 0.734 | 0.565 |
| frost | 0.721 | 0.644 | 0.541 |
| elastic | 0.687 | 0.574 | 0.520 |

Still capped at severity 1: brightness 0.528, saturate 0.602, spatter 0.567.

**Conformal check:**
- Thresholds from the 500 clean COCO-train images give 0.088 / 0.047 / 0.012 false alarms on clean COCO-val images at α = 0.10 / 0.05 / 0.01.
- At α = 0.05, the monitor flags 45 / 64 / 71 / 77 / 80% of corrupted images at severities 1–5, 75% of severity-1 fog and 70% of severity-1 contrast.

**Pre-registered decisions:**
- MX does not replace A primary (extra −0.002).
- The log level arm ties, so M1 keeps priority.
- Whitening stays an ablation.
- The bar is met.

By A's own M1 rule, this interim look also misses on the extra point (−0.001). That is why the decisive experiment now uses a non-inferiority margin, judged on unseen positions only.

## Figure 1
- **(a)** D's (Δlevel, Δshape) plane at s1, conditioned, in clean SDs. Gain and blur move down the shape axis together, noise moves along the level, and brightness stays put.
- **(b)** Directed versus omnibus: per family, the coherent share of the corruption shift against the directed arm's AUROC gain over the whitened arm (ρ = 0.78).
- **(c)** AUROC by severity on 5,000 images for the CDFs, Section 4, A primary and M1.

## Fit for IEEE IV
- **The families the flatter arm fixes are the road's visibility failures:** fog, haze, low contrast, blur and smear.
- **Real fog shifts the balance between the arms.**
  - A's depth-fog probe on 12 Cityscapes frames: at s1, shape −0.64 vs level −0.43 SD at β = 0.005, and shape −0.50 vs level −1.03 SD at β = 0.02.
  - D's probe, with a global reference: level 0.903 vs share 0.877 at β = 0.01.
  - The max keeps both arms.
- **Glare and night lights may make the channels peakier,** which the one-sided arm cannot see. A two-sided directed arm is the driving-data variant to report.
- **Deployment:**
  - calibration on 500 clean frames per camera, with a conformal threshold;
  - latency as for A's row, about 0.2 ms estimated next to the detector's 5.9 ms.

## Risks and cost
- **COCO:** no GPU. Scoring takes about a minute when the pass ends.
- **IV:** one backbone-only Cityscapes pass (bank, z-statistics, clean and foggy val) in an agreed GPU window, plus the Foggy Cityscapes download.
- **Risks:**
  - **Non-uniform fog shrinks the shape signal.** On E's 20 images with half or ramp fog, the share's AUROC falls to 0.54–0.62, although 82–88% of the s1 channels still flatten.
  - **The L10 cap:** brightness and saturate stay near chance at severity 1.
  - **The extra families:** M1 ties A and trails C2 there (0.861 vs 0.871).
  - **Selection:** A chose M1 after seeing positions 200–1058. This test is its first look on unseen images, and positions ≥ 1970 confirm it.

**Vote:** first A, second D. A first. A's peak-share idea, in A's own frozen max form (M1), is the strongest single-image monitor on 911 images no panelist had used. It scores 0.922/0.861 against the CDFs' 0.819/0.808, and 0.959 on severity-1 fog against 0.633. Its mechanism (coherent flattening, explaining away) is what the directed-test statistics predict. D second. D's thesis that level and crest are complementary axes is the structure M1 implements. D's Koschmieder probe and Foggy Cityscapes kill are the IV test the paper needs. D's own raw-crest row is dominated, so this vote is for the framing and the IV plan, not the row. Not chosen: C is A's row with a refuted mechanism sentence. E's deployable row is the weakest, though its oracle is a useful headroom figure. I withdrew my own Round-1 proposal.

**First experiment:** Foggy Cityscapes, paired by scene.

**Data.** The 500 Cityscapes val scenes, clean and rendered at beta = 0.005, 0.01 and 0.02. These beta values are a recollection, unverified, and the set must be downloaded. The fallback is Koschmieder fog rendered from Cityscapes disparity.

**Reference.** Rebuilt on Cityscapes train: 2,000 bank frames and 500 z-statistics frames.

**Run.** One backbone-only pass storing channel means and top-1% means at s1-s4. It runs after the COCO pass ends, in an agreed GPU window, and only after the user approves a written plan.

**Scoring.** Score M1, its two arms, A primary and Section 4 on clean against fogged val scenes, with a paired bootstrap over scenes.

**Kill.** At both beta = 0.005 and beta = 0.01, the flatter arm alone fails to beat the level arm with a 95% interval above 0. Then 'flatter all at once' is a property of the package's uniform fog, not of real fog. The IV claim for the shape dies, and on driving data M1 survives only as a level monitor.

**Free in the meantime.** Score M1 on COCO positions 1970 and above when the running confirmation completes. Kill if M1 - CDFs on common has an interval containing 0. If the lower bound of M1 - A primary on common is below +0.010, A primary leads.

**Checks run:** All checks ran on CPU (niced, 4 threads), each under a minute. Scripts and logs are in panelist-B/r3/.

**(1) Code check on the 200-image screen** (r3_test.py --smoke; reproduces other panelists' frozen rows exactly):
- M1 0.922/0.862, severity 1 0.869, FPR95 0.229/0.358;
- A primary 0.897/0.859; E4 0.916/0.831; C2 0.891/0.869; X 0.871/0.874; S4 0.841/0.870; B 0.876/0.856; CDFs 0.825/0.808.
- New rows, seen here for the first time: MX 0.922/0.858; W2 0.894/0.844; FW2 0.913/0.830.

**(2) Fresh test** (r3_test.py): 911 images, positions 1059-1969, pre-registered in PREREG_R3.md at 17:41 before any of these rows was scored on them. The screen images matched the pilot bit for bit.

AUROC common / extra (severity 1, FPR95 common):
- MX 0.925/0.860 (0.875, 0.202)
- M1 0.922/0.861 (0.869, 0.216)
- W2 0.899/0.851 (0.820)
- FW2 0.915/0.834
- A 0.901/0.862 (0.818, 0.251)
- E4 0.921/0.836
- C2 0.895/0.871
- X 0.874/0.875
- S4 0.842/0.868
- B 0.885/0.861
- CDFs 0.819/0.808 (0.705, 0.417)

Paired differences, common / extra (severity 1 where given):
- MX - A: +0.025 [+0.020, +0.029] / -0.002 [-0.007, +0.003] / +0.058
- MX - M1: +0.003 [-0.001, +0.007] / -0.001 [-0.004, +0.002]
- M1 - A: +0.021 [+0.015, +0.027] / -0.001 [-0.007, +0.004] / +0.051 [+0.043, +0.060]
- M1 - CDFs: +0.103 [+0.094, +0.111] / +0.052 [+0.045, +0.059] / +0.164 [+0.153, +0.174]
- W2 - C2: +0.004 [+0.001, +0.008] / -0.020 [-0.023, -0.017]
- W2 - A: -0.002 [-0.004, +0.001] / -0.011 [-0.014, -0.009]
- FW2 - MX: -0.011 [-0.013, -0.008] / -0.025 [-0.028, -0.023]
- MX - E4: +0.004 [-0.001, +0.009] / +0.024 [+0.018, +0.029]
- X - S4: +0.032 [+0.027, +0.036] / +0.007 [+0.003, +0.011]
- A - B: +0.016 [+0.011, +0.020] / +0.001 [-0.003, +0.004]

Pre-registered decisions:
- MX does not replace A (extra -0.002).
- Log level arm: a tie, so M1 keeps priority.
- Whitening: an ablation only.
- Bar against the CDFs: met.

**(3) Analysis on the same images** (r3_extra.py; nothing selected):
- Flatter arm alone 0.702 common; whitened shape 0.868.
- Share of the omnibus signal on the all-ones direction (lambda_1/lambda_W), s1, severity 1: fog 0.29, contrast 0.27, zoom 0.24, defocus 0.20; noise 0.01 or less.
- Channel-average shift at s1: fog +2.84, contrast +2.72, zoom +2.52, defocus +3.21 SD; noise -0.60 to -1.01 SD.
- Predicted power of the directed minus the omnibus test, against the observed AUROC difference over 38 cells: Spearman rho +0.78 (p 9.6e-9); sign agreement 0.50.
- Conformal: clean false alarms 0.088 / 0.047 / 0.012 at alpha 0.10 / 0.05 / 0.01.
- At alpha 0.05, flagged by severity 1-5: 0.45 / 0.64 / 0.71 / 0.77 / 0.80; severity-1 fog 0.75, contrast 0.70.

**(4) Whitening against the per-channel log level** (r3_bx.py):
- B - X: +0.010 [+0.007, +0.014] / -0.013 [-0.017, -0.010].
- B - S4: +0.043 [+0.036, +0.049] / -0.007 [-0.012, -0.002].

**(5) Clean bank only** (r3_content_share.py): the all-ones direction holds this share of the conditioned residual variance at s1 / s2 / s3:
- shape 0.124 / 0.051 / 0.026;
- level 0.035 / 0.042 / 0.039.

**(6) Exploratory, arms already seen** (r3_mfb.py): MFB = max(flatter, whitened log level).
- Fresh: 0.928/0.853. MFB - M1: +0.006 [+0.001, +0.010] / -0.008 [-0.012, -0.004].
- Screen: +0.001 / -0.013.

**(7) Citation:** O'Brien PC, Biometrics 40(4):1079-87, 1984. Title, journal and abstract verified through NCBI E-utilities.

## C (Topology): Content scales, corruption flattens: one dimensionless peak share per channel, judged against scenes like this one

**Thesis:** Scene content mostly sets how strongly a detector's early channels fire, while corruption makes that firing flatter; so each channel's dimensionless peak share, compared with clean scenes like this one, detects corruption far better than activation magnitudes or full activation distributions.

### Round 1 proposal

## Thesis
Scene content mostly sets how strongly a detector's early channels fire, while corruption makes that firing flatter. So one dimensionless number per channel separates corrupted from clean images far better than magnitudes or full activation distributions. That number is the channel's peak share: the mean of its strongest 1% of responses divided by its mean, compared with the same channel in clean scenes like this one.

## Insight
**First, plainly, the topology verdict from my seat: topology does not earn its place.** I ran three tests, each against a control.
1. **Conv-graph persistence.** The top of the diagram is the heaviest edges (L6).
2. **Excursion sets, today on the CPU.** 40 pilot images × 18 severity-2 conditions (glass blur left out), with a 150-image clean bank. I computed the Euler characteristic and perimeter of every channel's excursion sets at s1–s3.
   - The signal is real. Euler reaches 0.777 / 0.784 (common / extra), and still 0.774 / 0.764 when per-image quantile thresholds remove the magnitudes. It beats total variation by +0.073 [+0.042, +0.104].
   - But it is redundant. Means + Euler minus means: +0.004 [−0.016, +0.022] / −0.015 [−0.037, +0.006]. Area + perimeter + Euler minus area alone: −0.010 [−0.029, +0.006].
3. **Neighbourhood-graph topology of the clean bank, today on stored data.** The question: do the front and the back agree, by rank only, about which scenes are alike? This reaches 0.657–0.674 on the common families, against 0.841 for Section 4's distance-based version. Adding it hurts (0.815).

Topology aimed at the right thing: an invariant that ignores the nuisance. But here the nuisance is a per-channel gain, not a deformation. So the invariant is a ratio, not a Betti number.

**Why the ratio works.**
- **Content scales both ends together.** A channel's bulk and its peaks rise and fall with the scene: on the 2,000 clean bank images, the median correlation of log mean with log top-1% mean is 0.56 / 0.64 / 0.78 at s1 / s2 / s3. The ratio therefore cancels much of the content (L9).
- **Corruption moves the ratio, almost always downward.** At severity 1, the typical channel's peak share drops at all three stages for 14 of 19 families, that is, all except brightness and the four noise families.
- **How each family flattens:** blur widens peaks; haze and low contrast lower them against the BatchNorm floor; overlays and compression add activity to the bulk.
- **Noise is mixed,** because it makes its own peaks. But noise is easy.

**Why it is not obvious.**
- L4 says tails lose to the bulk, and L5 says the full CDF adds about 0.03 over the mean. Both hold for unnormalised statistics, where the gain dominates.
- What the CDF monitor's extra information buys sits in one scale-free coordinate, which its EMD buries under scale.
- **The ladder** (same 200 images, s1–s3, AUROC common / extra):

  | Step | AUROC common / extra |
  |---|---|
  | per-channel top-1% means | 0.792 / 0.749 |
  | peak share | 0.849 / 0.805 |
  | peak share against the 50 nearest clean scenes | 0.897 / 0.859 |
- It also redeems TU's starting point. The top of the diagram measured the strongest responses (0.661). That is useless sorted and unnormalised, but strong per channel and relative to the channel's own bulk.

**Lessons.**
- **For:** L1 (s1–s3); L2 (no positions); L3 (per channel, unsorted); L9 (Section 4's key).
- **Against:**
  - L11: the statistic was picked among about 7 on these 200 images.
  - L10: brightness and saturate stay at 0.53 / 0.57 at severity 1.
  - L8: the ratio drops the magnitude, costing −0.011 on the extra families against Section 4.

## Method
Training-free, clean-only and label-free, with one score per image. The taps are those of Table C.
1. **Peak share.** For channel c of stage s ∈ {1, 2, 3} (post-ReLU, n_s positions):
   u_{s,c} = log(T_{s,c} + ε) − log(M_{s,c} + ε).
   - M is the channel's mean over positions.
   - T is the mean of its ⌈0.01·n_s⌉ largest values.
   - ε = 1e-4. Offsets from 1e-6 to 1e-2 give the same AUROC.
2. **Content key.** Exactly Section 4's key: the stage-4 channel means, standardised with the bank. N(x) is the k = 50 nearest of the 2,000 clean COCO-train bank images (Euclidean).
3. **Stage score.** S_s(x) = (1/C_s) Σ_c |u_{s,c}(x) − mean_{n∈N(x)} u_{s,c}(n)| / σ_{s,c}, where σ is the bank's spread of u.
4. **Combine.** Score = Σ_s (S_s − μ_s)/τ_s. μ_s and τ_s come from 500 other clean images, scored the same way.

**In three sentences:**
- Measure each early channel's peak share.
- Compare it with the same channel in the 50 clean scenes that the detector's back finds most similar.
- Average over channels, z-score each stage on clean images, and add the stages.

**Cost:** a top-k per channel plus a kNN over 2,000 keys. My estimate is about 0.2 ms, next to the detector's 5.9 ms.

## Novelty
**Against the named methods:**
- **Becker et al. (ICPR 2026):** EMD between unnormalised per-channel CDFs and one global training CDF, so scale dominates; no content reference. On the screen, the peak share beats it by +0.072 [+0.057, +0.086] common and +0.051 [+0.038, +0.063] extra.
- **NMD:** the mean against the BatchNorm mean is the gain alone.
- **Mahalanobis / Gram:** moments of pooled features against class-conditional statistics. Even Gram's higher powers stay scale-homogeneous, so neither is scale-free per channel.
- **kNN-OOD:** L2-normalising the pooled penultimate vector removes one global gain, not each channel's.
- **DisCoPatch:** a trained patch discriminator.
- **Section 4:** the same reference, but on magnitudes. Swapping in the peak share gives +0.056 [+0.044, +0.069] common and −0.011 [−0.022, −0.000] extra.

**The closest statistic: Neural Activation Prior** (Wan et al., arXiv 2402.18162, preprint; I read the abstract and §1–3).
- **NAP:** the max over the mean of each channel before global pooling at the penultimate layer. Raw and one-sided (in-distribution images are more peaked), with no reference, for semantic OOD.
- **Ours:** the early stages, a robust top-1% mean, and a two-sided comparison with content-matched clean scenes, for corruption.

**Other related work:**
- SCALE (ICLR 2024, verified) and ASH (ICLR 2023, unverified) rescale logits by activation-sum ratios.
- DAVIS and Catalyst (2026, seen only in search results) use channel max and variance at the penultimate layer.
- TopoAD (MDPI Sustainability 2026; abstract only) combines Euler-characteristic curves with energy for semantic OOD. Test 2 above finds the Euler characteristic redundant for corruption.
- **IQA:** BRISQUE/NIQE fit scale-free shape parameters to contrast-normalised pixels (from memory). That is the pixel-level ancestor; ours lives inside the detector and takes its reference from the detector's own invariant back.
- Liu et al. (IEEE T-IV 2024, read) observe content-independent channel statistics in degraded features, but train an enhancement module.

## Decisive experiment
**Setup:**
- 5,000 images × 96 conditions, with frozen choices: s1–s3, top 1%, k = 50, two-sided, and the pilot's bank and z-statistics (whose top-1% means are already stored).
- The other 4,800 images are unseen.
- Paired bootstrap against the activation CDFs (0.821 / 0.807) and against Section 4 on the means from the same pass, plus the ladder.

**What makes the paper:**
- AUROC common ≥ 0.85 and extra ≥ 0.83.
- Intervals above 0 against the CDFs on both, and against Section 4 on the common families.
- Severity-1 common ≥ 0.78 (CDFs: 0.707).
- Fog, contrast and zoom blur no worse than the CDFs.

**Kill:** either of these ends the idea.
- Δ against Section 4 on the common families below +0.015, or an interval touching 0. The shape is then not the insight, and the paper is Section 4.
- An interval against the CDFs on the extra families that includes 0.

**Pre-registered secondaries, not for selection:** gain + shape (screen 0.891 / 0.869) and a one-sided flattening score (0.921 / 0.823).

## Cheapest first test
Done on the stored pilot tables in about 20 s. It is a screen, and the statistic was chosen on these same images.
- **Headline:** 0.897 / 0.859, FPR95 common 0.253.

  | Row | AUROC common / extra | FPR95 common |
  |---|---|---|
  | Peak share, conditioned | 0.897 / 0.859 | 0.253 |
  | Section 4 | 0.841 / 0.870 | 0.368 |
  | Activation CDFs | 0.825 / 0.808 | 0.403 |
- **By severity (common, 1–5):** 0.814 / 0.876 / 0.912 / 0.935 / 0.949, against 0.716 / 0.787 / 0.841 / 0.880 / 0.903 for the CDFs.
- **Section 4's weak spots,** at severity 1 / 3 / 5:

  | Family | Peak share | Section 4 |
  |---|---|---|
  | fog | 0.80 / 0.90 / 0.92 | 0.49 / 0.55 / 0.60 |
  | contrast | 0.82 / 0.97 / 1.00 | 0.51 / 0.65 / 0.89 |
  | zoom blur | 0.88 / 0.89 / 0.90 | 0.73 / 0.67 / 0.64 |
- **Stability:**
  - Split halves: +0.057 and +0.055 over Section 4.
  - k = 10 / 200: 0.893 / 0.889.
  - p99 over the mean instead of the top 1%: 0.887 / 0.855.
  - Each stage alone: 0.873–0.878.
- **Not a padding artefact.** On 12 images, leaving out the 2-cell zero-padding frame keeps the median s1 log peak share: clean 1.613 vs 1.615, fog 1.340 vs 1.339.

## Figure 1
**(a)** One s2 channel, with the 2,000 clean scenes plotted as log mean against log peak share; content spreads them along the gain axis. Arrows lead from 20 pilot images to their fog, blur and contrast versions, and all point down (flatter). Brightness barely moves.

**(b)** The ladder by severity: TU top of diagram → per-channel top 1% → peak share → conditioned peak share. The CDFs and Section 4 are reference lines.

## Fit for IEEE IV
A camera-health monitor inside the frozen detector: about 0.2 ms, one score per frame, and no corrupted data.

**Driving-data plan:**
- **Synthetic corruptions:** the clean bank from Cityscapes train; Cityscapes val with the same 19 × 5 imagecorruptions.
- **Matched content:** Foggy Cityscapes, the same val images at three fog densities.
- **Real conditions:** ACDC fog, night, rain and snow against its clean reference drives (from memory; both need downloading), and nuImages night and rain through metadata.
- **Global and conditioned:** driving content is more uniform, so report the global and the conditioned peak share side by side.
- **Position (L2):** the fixed camera reopens it, so an upper/lower-band peak share is a pre-registered secondary.

## Risks and cost
**GPU passes:**
- **COCO:** one pass storing per-channel top-1% means for the 5,000 × 96 variants. That takes about 2.5–3 h, limited by the CPU, or nothing extra if the means pass that is about to start stores them too. I flagged this to the main session.
- **Cityscapes:** about 1 h.

**New code:** about 30 lines: a top-k per channel, then Section 4's scorer.

**What could go wrong:**
- On held-out images, the gain shrinks because the statistic was selected on the screen.
- Brightness and saturate stay capped (L10).
- Noise-type extra families stay slightly below the means; the gain + shape secondary covers that.
- The mechanism leans on BatchNorm + ReLU sparsity, so a claim of generality needs a second backbone.
- Soft or hazy clean photos become false positives.
- The stage-4 key drifts under heavy corruption: the stage-4 level falls to ×0.50–0.95 of clean at severity 5.

**Quick check:** All runs were on the CPU, niced, each under 3 min. Scripts and outputs are in /tmp/claude-1000/-home-yuchen-YuchenZ-UE-philip-sa/3d52b916-b3ff-4806-a501-7f70cde31a29/scratchpad/detection/roundtable/panelist-C/. Results on 200 or 40 images are screens.

(1) TOPOLOGY, CPU forward pass (extract.py, score.py; negative result).
- Setup: 40 pilot images × 18 severity-2 conditions (glass blur left out), 150-image clean train bank, 50 z-statistics images, own-average scoring at s1–s3. My CPU channel means match the pilot's stored GPU means to 4–9e-4 L1-relative per image.
- AUROC common / extra: means 0.789 / 0.831; area at 4 clean thresholds 0.788 / 0.776; Euler characteristic 0.777 / 0.784; Euler at per-image quantiles 0.774 / 0.764; total variation 0.704 / 0.697; area + perimeter + Euler 0.779 / 0.774; means + Euler 0.793 / 0.816. The stored activation CDFs on the same images and conditions reach 0.785 / 0.754.
- Paired differences:
  - means + Euler − means: +0.004 [−0.016, +0.022] / −0.015 [−0.037, +0.006];
  - Minkowski − area: −0.010 [−0.029, +0.006] / −0.002 [−0.021, +0.019];
  - Euler − total variation: +0.073 [+0.042, +0.104] / +0.088 [+0.053, +0.129].

(2) Stored data, 200 × 96 (neighbourhood_agreement.py; negative result).
- Rank-only neighbourhood-graph agreement between the s4 key and s1–s3: overlap 0.657 / 0.629, rank 0.674 / 0.647, distance ratio 0.441 / 0.379.
- Section 4 + overlap 0.815 / 0.833, against Section 4 alone 0.841 / 0.870 (reproduced exactly).

(3) Stored data (key_gain.py; no real gain).
- A gain-free stage-4 key (log, own mean removed) gives 0.843 / 0.872. Fog moves from 0.55 to 0.63, contrast from 0.69 to 0.75, and elastic falls from 0.74 to 0.66.
- The s4 key level falls under corruption, to ×0.50–0.95 of clean at severity 5.

(4) CPU forward pass, scale self-reference r = log m(320) − log m(640) (extract_scale.py, score_scale.py).
- r alone 0.798 / 0.837, with pixelate 0.92 and JPEG 0.93.
- Section 4 + r − Section 4: +0.002 [−0.018, +0.019] / −0.006 [−0.029, +0.019].

(5) THE PROPOSAL, stored data, 200 × 96 (conditioned_shape.py, shape_check.py, shape_robustness.py, gain_plus_shape.py).
- Conditioned log(top-1% mean / mean), s1–s3, k = 50: 0.897 / 0.859, FPR95 common 0.253. Section 4 reaches 0.368 FPR95 and the CDFs 0.403.
- Common AUROC by severity 1–5: 0.814 / 0.876 / 0.912 / 0.935 / 0.949.
- Paired differences:
  - vs Section 4: +0.056 [+0.044, +0.069] / −0.011 [−0.022, −0.000];
  - vs the activation CDFs (0.825 / 0.808 on these images): +0.072 [+0.057, +0.086] / +0.051 [+0.038, +0.063];
  - vs the global peak share (0.849 / 0.805): +0.048 [+0.037, +0.060] / +0.053 [+0.043, +0.064].
- Other rows: conditioned log means 0.871 / 0.874; global top-1% means 0.792 / 0.749; conditioned p99/mean 0.887 / 0.855; k = 10 / 200 gives 0.893 / 0.889; s1–s4 gives 0.894 / 0.850; each stage alone 0.873–0.878.
- Split halves: 0.893 / 0.852 and 0.902 / 0.866, against Section 4's 0.836 / 0.864 and 0.847 / 0.876.
- Log offsets from 1e-6 to 1e-2 change nothing.
- Not selected: gain + shape 0.891 / 0.869; one-sided 0.921 / 0.823.
- Per family, severity 1 / 3 / 5: fog 0.80 / 0.90 / 0.92 (Section 4 0.49 / 0.55 / 0.60, CDFs 0.65 / 0.75 / 0.77); contrast 0.82 / 0.97 / 1.00; zoom 0.88 / 0.89 / 0.90; brightness 0.53 / 0.66 / 0.79; saturate 0.57 / 0.61 / 0.86; elastic 0.58 / 0.69 / 0.79.
- Clean correlation of log mean with log top: 0.56 / 0.64 / 0.78 at s1 / s2 / s3. At severity 1 the typical channel's peak share drops at all three stages for 14 of 19 families.
- Caveat: the statistic was picked among about 7 per-channel statistics on these same 200 images (L11).

(6) CPU, 12 images (border_check.py).
- 15% of the s1 top-1% positions lie in the 2-cell zero-padding frame, which holds 4.9% of positions.
- Dropping the frame leaves the median s1 log peak share almost unchanged: clean 1.613 vs 1.615, fog 1.340 vs 1.339, contrast 1.311 vs 1.297. So the signal is not a padding artefact.

**Runners-up:**
- Scale self-reference: the image at half resolution as its own content reference: Per channel, log m(320) − log m(640) reaches 0.798 / 0.837 alone on the 40-image CPU screen, with 0.92 on pixelate and 0.93 on JPEG, but adds only +0.002 [−0.018, +0.019] on top of Section 4.
- Minkowski functionals (area, perimeter, Euler characteristic) of per-channel excursion sets: The best topological candidate: real signal (0.777 / 0.784, +0.073 over total variation) but redundant with the means (+0.004 [−0.016, +0.022]), so it is worth a paragraph on why topology fails, not a paper.

### Round 2: critiques written by C

**On A** (fixable)

- Steelman: A's row, the content-conditioned log peak-to-mean ratio at s1-s3, is the strongest single-image detector any panelist found. It replicates on held-out images (0.898 / 0.862 on 308 images). A's mechanism, that corruption lowers each channel's ceiling, correctly predicts where the gain sits: fog, contrast and the blurs.
- Objection: A's main evidence for the mechanism is an artefact of dividing by the level. That evidence is the clean 'level-peakiness trade-off' (r = -0.74 / -0.85 / -0.88) and the descending band in the left panel of Figure 1. Peakiness is log t - log m, so it is anti-correlated with log m by construction whenever the ceiling varies less than the level. Suppose the ceiling were independent of the level. Then corr(log m, log t - log m) = -sd_m / sqrt(sd_m^2 + sd_t^2), which gives -0.82 / -0.87 / -0.88 (medians over channels). The observed values are equal to that or weaker. So K3 ('anti-correlation weaker than -0.5') can never fire, and it is not a kill criterion. The claim that the ceiling is 'nearly fixed by exposure and optics' while content sets the level is also overstated: the 50-scene reference explains about the same share of the ceiling's clean variance as of the level's (s1 0.46 vs 0.35; s2 0.51 vs 0.47; s3 0.47 vs 0.50). What does hold is the size of the shift. At severity 1, s1, the conditioned log ceiling moves -1.89 / -1.92 / -2.04 / -1.96 clean residual SDs under fog / contrast / zoom / defocus. The log level moves -0.45 / -0.24 / -0.20 / -0.12. A minor gap: A's novelty section omits Neural Activation Prior (Wan et al., arXiv 2402.18162; I read it in Round 1). NAP divides each channel's max by its mean at the penultimate layer, with no reference, for semantic OOD, and is the closest published statistic.
- Failure scenario: A reviewer redraws the left panel of Figure 1 with two independent positive random variables of the same spreads. The same descending band appears, with r of about -0.8. The mechanism section collapses, even though the AUROC is real.
- Evidence: mechanism_r2.py (screen; clean bank): observed against independence-null correlations; content share of variance; severity-1 paired shifts. The conditioned log ceiling alone reaches 0.853 / 0.819 overall, but fog 0.91, contrast 0.96 and zoom 0.93 (mean over severities). Held-out interim (308 images): A's row (eps 1e-6) gives 0.898 / 0.862. Every pre-registered threshold of A's passes: common 0.898 >= 0.87; extra 0.862 >= 0.83; severity-1 0.818 >= 0.77; FPR95 0.246 <= 0.30; severity-1 fog / contrast / zoom above the CDFs by +0.18 / +0.16 / +0.11. K2 also passes: vs Section 4, +0.053 [+0.042, +0.063].
- Would change my mind: A trade-off measure that is immune to the ratio artefact and still shows a scene effect. For example, the clean conditioned residuals of (log level, log ceiling) form a negatively sloped band, or the observed correlation clearly exceeds the independence null.

**On B** (serious)

- Steelman: B's log-whitened channel means need no tuning, are closed form (about 10 µs, no key, no kNN) and were properly pre-registered. They are confirmed on 308 held-out images (0.882 / 0.858; +0.058 / +0.048 over the CDFs). That makes B the strongest baseline any shape method must beat, and the simplest to deploy.
- Objection: A per-channel control contradicts the thesis 'invisible per channel, obvious jointly'. The conditioned log means score each channel alone, with no covariance, against Section 4's key. On the same held-out images they reach 0.877 / 0.877, which is -0.005 [-0.011, +0.002] common and +0.020 [+0.014, +0.024] extra against B's primary. Per-channel monitors failed because of their global reference, not because they score channels separately. Whitening is one way to remove content, the kNN reference is another, and the log is the active ingredient (it adds +0.032 to Section 4). Second, whitening suppresses exactly the corruption that matters for IV. Fog, contrast and blur shift the channels coherently, along high-variance clean directions that whitening divides away. On held-out, B's fog is 0.72 (severity 1: 0.64), against 0.88 (0.81) for the peak share and 0.71 for the CDFs. On the screen, B-style whitening of the conditioned log share loses -0.040 [-0.049, -0.031] common and -0.027 [-0.035, -0.019] extra against the plain share. So 'whiten everything' is not a general principle. As a method, B is a Rippel-style Gaussian plus a log, which is a strong baseline but not a paper.
- Failure scenario: On Foggy Cityscapes, B matches the CDFs on fog (as on COCO: 0.72 vs 0.71) while the shape monitor wins. The headline method of an IV paper then fails on IV's canonical degradation.
- Evidence: heldout_all_rows.py, 308 held-out images: X (conditioned log means) - B as above. C - B: +0.017 [+0.009, +0.024] common, +0.005 [-0.002, +0.011] extra, +0.027 [+0.018, +0.037] at severity 1. Per family, B vs C (mean over severities): fog 0.72 vs 0.88; contrast 0.86 vs 0.94; zoom 0.84 vs 0.92. merge_hints.py (screen): whitened share 0.858 / 0.832. I also verified B's literature claim in docs/literature-review: Becker's best normalizing flow scored 61.8 vs 68.0 for the CDFs at severity 3.
- Would change my mind: Whitening beating the per-channel conditioned log means on both common and extra, with intervals excluding 0, on the 4,800 images. Or whitening rescuing fog when applied to a statistic that carries fog; it did not (WSH, -0.040).

**On D** (fixable)

- Steelman: D's level + crest is the most balanced row. It keeps magnitude (L8), adds shape, and beats Section 4 on common without losing on extra: held-out +0.036 [+0.030, +0.043] / +0.003 [-0.002, +0.009]. It also has the sharpest IV kill (Foggy Cityscapes at fixed fog density β) and names the right risk: real fog may spare the peaks.
- Objection: D is the A/C idea in a weaker parameterisation. On the same held-out images, log versions of the raw crest t/m and raw level do better. The log share alone beats D by +0.017 [+0.012, +0.022] common (-0.011 [-0.015, -0.007] extra). My pre-registered log gain + log share (C2) reaches 0.894 / 0.872, against D's 0.881 / 0.873. D's mechanism ('content mostly moves the level; the peak only 0.20 / 0.19 / 0.25') is half true. The ceiling varies less in absolute log units, but content explains about the same fraction of its variance as of the level's (0.46 / 0.51 / 0.47 vs 0.35 / 0.47 / 0.50). The serious open issue is shared by A, C and D. The package's fog compresses contrast uniformly. Real fog's transmission e^(-βd) spares near-field edges, and in driving scenes those may be exactly the top 1% (car edges, poles, near lane marks). The ceiling would then hold while the mean falls with the far field, and the crest rises. The two-sided score still reacts, but the shift may fall inside the clean spread.
- Failure scenario: Foggy Cityscapes at β = 0.005: the near-field top 1% survives, the far field loses texture, and the crest moves up by less than its clean spread. D's kill (b) fires, and the whole A/C/D family loses its IV story.
- Evidence: heldout_all_rows.py (308 images): D 0.881 / 0.873, FPR95 0.282, severity 1 0.797. C - D as above. C2 0.894 / 0.872. All of D's interim criteria pass: vs CDFs +0.057 / +0.063; severity 1 0.797; FPR95 0.282; fog 0.76 and contrast 0.87 (means over severities). Content shares from mechanism_r2.py.
- Would change my mind: The raw crest beating the log share on held-out common. Or a concrete deployment reason for raw ratios, such as night frames with near-zero means where the log offset misbehaves.

**On E** (serious)

- Steelman: E's same-scene oracle is a genuinely new diagnostic. When the content is known, the early means separate mild fog, contrast and zoom blur at 0.94-0.99, so the weak spots are not blindness. The 'slide' finding also matters: stage-4 typicality stays at chance while 34-72% of the neighbours turn over. That explains the residual gap of every conditioned method, including mine (oracle neighbours 0.927 vs 0.897).
- Objection: The aliasing/blindness split is a property of the channel mean, not of the detector or of single images. A single-image statistic that reads each channel's ceiling closes most of E's aliasing gap with no same-scene view. On the screen at severity 1, fog / contrast / zoom reach 0.84 / 0.86 / 0.89 for the conditioned log ceiling and 0.80 / 0.82 / 0.88 for the peak share. E's row (a) gets 0.56 / 0.58 / 0.69 and the oracle 0.949 / 0.941 / 0.993. So the ceiling closes about 66-78% of the gap and the share about 62-66%. On held-out, the share reaches 0.81 / 0.83 / 0.90. This contradicts the 'so' clause of the thesis ('single-image monitors fail because scene variation hides the corruption'). It also contradicts E's IV message that haze, contrast and zoom 'need a same-scene reference'. E's method is then one of two things. (a) is Rippel + SPADE, the weakest row on held-out: 0.850 / 0.854; C - E = +0.048 [+0.037, +0.060]. (b) is a change detector that needs a clean frame of the same scene. It misses conditions present from the first frame and gradual onsets, such as thickening fog or accumulating soiling, where the reference is already degraded. That detects changes, not corruption.
- Failure scenario: A car leaves a garage into fog. No clean same-scene frame exists, so (b) never fires. (a) reaches only 0.57 AUROC at severity-1 fog, while the single-image ceiling separates it.
- Evidence: heldout_all_rows.py: per-family E column vs C. merge_hints.py (screen): CEIL and C at severity 1. E's own oracle_decomposition.log gives the oracle numbers. All of E's interim single-image criteria still pass: (a) - CDFs = +0.026 [+0.010, +0.042] / +0.044 / +0.063 at severity 1; FPR95 0.330.
- Would change my mind: An oracle computed for the ceiling or share that still leaves a large gap at severity 1 on 4,800 images (say 0.99 vs 0.80), together with a deployable same-scene reference that beats the single-image ceiling, for example ACDC's same-place images. E's frame would then be right for an IV extension.

**Defense of own:** I expect two attacks: selection on the 200 screen images (L11; three of us picked the best of about 7 stored statistics there), and my mechanism.

(1) Selection. I wrote a pre-registration in the docstring of heldout_all_rows.py at 17:06, before any score at positions 200+ existed. I then ran every frozen Round-1 row on the 308 held-out images that B had already used (positions 200-507), so no fresh held-out image was spent. A smoke run on the screen reproduced every author's numbers exactly first. My row on held-out:
- 0.898 / 0.862 (screen 0.897 / 0.859);
- severities 0.818 / 0.877 / 0.913 / 0.934 / 0.949;
- FPR95 0.245 / 0.337;
- vs Section 4: +0.053 [+0.042, +0.063] common, -0.008 [-0.017, +0.001] extra;
- vs CDFs: +0.074 [+0.062, +0.087] common, +0.052 [+0.043, +0.062] extra, +0.107 at severity 1.
Nothing shrank. Every paper criterion I pre-registered passes and no kill fires. It is still an interim look (308 images, about ±0.01); the 4,800 images decide.

(2) Mechanism. My Round-1 claim was 'content scales both ends; the ratio cancels much of the content'. My own check refutes it: the 50-scene reference explains 0.35 / 0.46 / 0.40 of the clean variance of level / ceiling / share at s1 (0.47 / 0.51 / 0.47 at s2; 0.50 / 0.47 / 0.45 at s3). I withdraw that claim. What survives, and is A's better story:
- Visibility loss lowers each channel's ceiling 4-16 times more than its level. At severity 1, s1, in clean residual SDs: fog -1.89 vs -0.45, contrast -1.92 vs -0.24, zoom -2.04 vs -0.20, defocus -1.96 vs -0.12.
- Noise moves the bulk.
- The ceiling alone wins fog / contrast / zoom (0.91 / 0.96 / 0.93) but reaches only 0.853 / 0.819 overall. The level alone reaches 0.871 / 0.874. The ratio keeps both: 0.897 / 0.859.
- This qualifies L4: once scale and content are removed, the tail carries visibility loss.

(3) Extra families. My pre-registered secondary C2 (log gain + log share) recovers extra: +0.010 [+0.008, +0.013] for -0.004 common. I will not swap primaries on 308 images.

(4) Real fog. The risk that real fog spares near-field peaks is real and shared with A and D. I would pre-register D's band split and a sign prediction for Foggy Cityscapes.

Changes:
- New thesis: 'Visibility loss lives in each early channel's ceiling, noise in its bulk; their ratio, against clean scenes like this one, detects both.'
- Figure 1 becomes (Δ log level, Δ log ceiling) in conditioned residual-SD units, not the arithmetic level-share band.
- Report positions 508-4999, untouched by any panelist, separately.

The topology verdict stands: topology does not earn its place. The Euler characteristic adds +0.004 [-0.016, +0.022] over the means, and neighbourhood-graph agreement reaches only 0.657-0.674 and lowers Section 4 to 0.815.

**Best other proposal:** A. It is the same row as mine and passes all its stricter thresholds on the interim held-out set (0.898 / 0.862; FPR95 0.246; severity-1 fog / contrast / zoom +0.18 / +0.16 / +0.11 over the CDFs). My check also supports A's mechanism, that corruption lowers each channel's ceiling, better than my own Round-1 story. A only needs to drop the trade-off evidence, which is an artefact of the ratio. If the whole A/C/D family were off the table, I would bet on B as a method (held-out 0.882 / 0.858, closed form, pre-registered) but not on its thesis.

**Merge suggestion:** Merge A, C and D into one paper on the thesis 'the ceiling carries visibility loss'.

Note first why the three of us converged: the stored data hold only three per-channel statistics (mean, top-1%, p99). So the ratio is the best function of what we stored, not necessarily the best statistic. Sparsity-type statistics such as ‖x‖₄/‖x‖₁ or the active fraction need a GPU pass.

Contents of the merged paper:
- **Primary, frozen:** conditioned log peak share (s1-s3, top 1%, k = 50, stage-4 key), scored on the 4,800 images. Report the 4,492 untouched images (positions 508+) separately.
- **No extra GPU pass needed.** The confirmation's files store top_s1..s4, bit-identical to the pilot on the 200 screen images.
- **Mechanism:** A's framing, supported by the paired-shift table instead of the trade-off band.
- **IV plan:** D's plan, with the Foggy Cityscapes kill at fixed fog density, the upper/lower-band split and a pre-registered sign prediction.
- **Secondary:** C2 (log gain + log share) for the extra families.
- **Baseline and ablation:** B's log-whitened means (and B2) as the strongest baseline. The whitening ablation (-0.040) shows that visibility shifts are coherent across channels.
- **Analysis:** E's oracle, computed for the means, the ceiling and the share. It shows how much single-image detection leaves, and that the key drift costs about 0.03 (oracle neighbours 0.927).

Merges of statistics did not help on the screen (exploratory):
- C + B2: -0.000 common, +0.005 extra;
- level + ceiling: -0.014 [-0.020, -0.007] common;
- whitened share: -0.040.
So nothing beyond C2.

**Ranking:**
1. A: The same winning row (held-out 0.898 / 0.862) with the right mechanism (corruption lowers the ceiling) and the strictest pre-registered thresholds, all passing on the interim. Its trade-off evidence is a ratio artefact, which is fixable.
2. C: The identical row and the same held-out replication, plus a tested negative verdict on topology. My Round-1 mechanism ('the ratio cancels content') is refuted by my own check and must take A's ceiling story.
3. D: Same family in a weaker raw parameterisation (-0.017 common vs the log share on held-out), but the most balanced row (no extra loss vs Section 4: +0.003) and the best IV plan and kill.
4. B: The strongest distinct method, properly pre-registered (0.882 / 0.858 held-out). But the per-channel conditioned log-means control matches it (-0.005 common, +0.020 extra), which contradicts its thesis, and it does not fix fog (0.72 vs 0.88).
5. E: A valuable analysis, but the single-image ceiling closes about two thirds to three quarters of its 'aliasing' gap, which contradicts the thesis. Its single-image row is the weakest on held-out (0.850 / 0.854), and its deployable variant detects changes rather than corruption.

**Checks run:** All checks ran on CPU (niced). Scripts and logs are in /tmp/claude-1000/-home-yuchen-YuchenZ-UE-philip-sa/3d52b916-b3ff-4806-a501-7f70cde31a29/scratchpad/detection/roundtable/panelist-C/r2/. No proposal drifts into predicting performance drop.

(0) count_and_top_check.py. At 17:06 the confirmation had 863 files, including the 200 screen images and all 308 images at positions 200-507. Its means_s1..4 and top_s1..4 equal the pilot's stored values bit for bit (worst relative difference 0.0) on the 200 screen images. A, C and D therefore need no extra GPU pass.

(1) heldout_all_rows.py. Pre-registered in the docstring before scoring. It runs every frozen Round-1 row on the 308 images B had already used (positions 200-507), so no fresh held-out image was spent. A smoke run on the screen first reproduced every author's numbers exactly: C 0.897 / 0.859, D 0.876 / 0.870, B 0.876 / 0.856, B2 0.884 / 0.860, E(a) 0.855 / 0.860, Section 4 0.841 / 0.870, CDFs 0.825 / 0.808.

Held-out results (common / extra; FPR95 common; severity 1):

| Row | Common / extra | FPR95 common | Sev 1 |
|---|---|---|---|
| C = A | 0.898 / 0.862 | 0.245 | 0.818 |
| D | 0.881 / 0.873 | 0.282 | 0.797 |
| B | 0.882 / 0.858 | 0.278 | – |
| B2 | 0.890 / 0.862 | 0.259 | – |
| E(a) | 0.850 / 0.854 | 0.330 | – |
| Section 4 | 0.845 / 0.870 | 0.353 | – |
| CDFs | 0.824 / 0.810 | 0.411 | – |
| C2 | 0.894 / 0.872 | – | – |
| one-sided | 0.912 / 0.824 | – | – |
| X (conditioned log means) | 0.877 / 0.877 | – | – |

Paired differences (common | extra | severity 1):
- C - Section 4: +0.053 [+0.042, +0.063] | -0.008 [-0.017, +0.001] | +0.049
- C - CDFs: +0.074 [+0.062, +0.087] | +0.052 [+0.043, +0.062] | +0.107
- C - D: +0.017 [+0.012, +0.022] | -0.011 [-0.015, -0.007]
- C - B: +0.017 [+0.009, +0.024] | +0.005 [-0.002, +0.011]
- C - B2: +0.008 [+0.001, +0.016] | -0.000
- C - E: +0.048 [+0.037, +0.060] | +0.008
- D - Section 4: +0.036 [+0.030, +0.043] | +0.003
- B - CDFs: +0.058 [+0.046, +0.069] | +0.048 (identical to B's report)
- E - CDFs: +0.026 [+0.010, +0.042] | +0.044
- X - B: -0.005 [-0.011, +0.002] | +0.020 [+0.014, +0.024]
- C2 - C: -0.004 | +0.010

Severity-1 fog / contrast / zoom by row: C 0.81 / 0.83 / 0.90; D 0.68 / 0.71 / 0.83; B 0.64 / 0.68 / 0.81; E 0.57 / 0.58 / 0.69; Section 4 0.52 / 0.54 / 0.75; CDFs 0.63 / 0.67 / 0.79.

(2) mechanism_r2.py, screen; descriptive, nothing selected.
- corr(level, share), observed vs the null with an independent ceiling: -0.74 vs -0.82, -0.85 vs -0.87, -0.88 vs -0.88 at s1 / s2 / s3.
- Content share of clean variance, level / ceiling / share: 0.35 / 0.46 / 0.40, 0.47 / 0.51 / 0.47, 0.50 / 0.47 / 0.45.
- Conditioned rows, common / extra (fog, contrast, zoom as means over severities): log level 0.871 / 0.874 (0.64, 0.80, 0.84); log ceiling 0.853 / 0.819 (0.91, 0.96, 0.93); log share 0.897 / 0.859 (0.88, 0.94, 0.90).
- Severity-1 s1 paired shifts in residual SDs, level / ceiling: fog -0.45 / -1.89, contrast -0.24 / -1.92, zoom -0.20 / -2.04, defocus -0.12 / -1.96, brightness about 0.

(3) merge_hints.py, screen; exploratory, carries L11 risk. Severity-1 fog / contrast / zoom in brackets.
- ceiling 0.853 / 0.819 (0.84 / 0.86 / 0.89);
- level + ceiling 0.884 / 0.861, i.e. -0.014 [-0.020, -0.007] vs C;
- C + B2 0.897 / 0.864, i.e. -0.000 [-0.005, +0.004] / +0.005 [+0.001, +0.009];
- whitened share 0.858 / 0.832, i.e. -0.040 [-0.049, -0.031] / -0.027 [-0.035, -0.019].

(4) I confirmed B's literature claim in docs/literature-review-image-corruption-detection.md: the best flow scored 61.8 vs 68.0 for the CDFs at severity 3.

### Round 3: rebuttal by C

- **To A** (partially concede): My mechanism is false on the bank. 'Content scales both ends together, so the ratio cancels much of the content' fails: SD(log share)/SD(log mean) is 0.89/0.79/0.70, a ridge from the stage-4 key explains 60/69/68% of the shape against 69/77/77% of the level, and kNN conditioning leaves the same spread. What actually works is (1) a coherent drop in s1 peakiness and (2) conditioning recruiting peakier neighbours. 'Corruption flattens' is not universal either: NAP's one-sided ratio is inverted (0.302/0.343), and depth-dependent fog makes s3 peakier. — I concede the content-cancellation claim. I had already withdrawn it in Round 2 with my own numbers: the 50-scene reference explains 0.35/0.46/0.40 of the clean variance of level, ceiling and share at s1. A's two mechanisms now enter the merged proposal as Facts 3 and 4.

I concede the depth-fog point. On A's 12 and D's 24 Cityscapes images, s3 does not flatten (A: +0.17/+0.38; D: 0.00/+0.03). The IV sign prediction is therefore limited to s1-s2.

I reject the reading that NAP's inversion refutes flattening. It is an artefact of averaging raw ratios. The ladder in nap_ladder.py (a screen) goes NAP 0.302, then 0.402 without the square, then 0.678 in logs (+0.276 [+0.263, +0.289]). That log row uses no reference and already gives fog 0.947 and contrast 0.943.
- At s2, 69-75% of channels' ratios fall under fog, contrast, zoom and defocus blur.
- Yet NAP's average rises in 81-100% of images, because the 5% of channels with the largest clean ratio (mean r^2 1,360 vs median 49) carry 43-58% of the change, and those channels sharpen.
- Across channels, the paired change of u rises with clean sparsity: Spearman +0.38 to +0.62 for fog and contrast.
- The densest fifth flattens by 0.93-1.98 bank SDs and the sparsest fifth sharpens by +0.02 to +0.69. That holds in 14 of 14 cells each way, on the screen and on 308 held-out images.

So 'flatter' is true for the dense majority and false for the sparse minority, and the thesis now says 'opposite moves'.
- **To B** (concede): C is the same row as A, and C's mechanism sentence is contradicted: the ratio removes only 11-30% of the clean spread, and the s4 key still predicts 62/67/67% of the shape. The gain is on the signal side, from a coherent shape shift (cos^2 with the all-ones direction 0.59/0.52/0.52, against 0.22/0.26/0.11 for the level). If the ratio cancelled content, why would it need 2,000 neighbours? On Cityscapes, global and conditioned would then be about equal. — I concede both points.
- My row is A's row: B finds 0.889/0.855 for both on 678 held-out images. So I merge into A, with D.
- The 'cancels content' sentence is gone. The merged mechanism is B's and A's signal-side story, plus the opposite moves of dense and sparse channels.

The content reference is needed because content sets the share almost as much as the level. Conditioning adds +0.048 [+0.036, +0.062] to the two-sided row and +0.037 [+0.029, +0.046] to the flattening arm (screen).

B's Cityscapes test, global and conditioned side by side, becomes a pre-registered secondary in the driving plan. It measures how much a fixed camera needs the key; it no longer supports a claim I make.

B's held-out numbers also show that my Round-1 kill tests pass: +0.050 [+0.042, +0.058] against Section 4, and +0.051 [+0.044, +0.059] against the CDFs on extra.
- **To D** (partially concede): The thesis overclaims. 'Far better than magnitudes' holds only on COCO's common families: held out, the shape is below the level on the extra families, and on driving fog it ties the level. Its one-sided reading is NAP's prior, which fails here. The one-sided secondary is an IV hazard, because glare and spatter push the share up. D would change their mind if C restated the thesis as two-sided and content-conditioned, reported NAP as a baseline, and showed the one-sided score holding on ACDC night and on spatter. — I concede the overclaim. On 859 held-out images the share is +0.052 [+0.046, +0.059] against Section 4 on common but -0.008 [-0.013, -0.002] on extra. Against conditioned log means it is +0.021 on common and -0.015 on extra (308 images). I withdraw 'far better than magnitudes'.

The merged challenger C4 = max(flattening arm, Section 4) keeps the level for the families where it wins. On the screen it reaches 0.931/0.871, which is +0.034 [+0.025, +0.044] and +0.013 [+0.004, +0.022] over the frozen row. It is registered for positions above 1058.

I also concede the restatement: the thesis is now per channel, in logs, and read against content-matched scenes. NAP is reported as a baseline (0.302/0.343), with the ladder that explains its inversion.

The Koschmieder probe is a tie within noise: 24 vs 24 images, AUROC 0.877 against 0.903, while d' is 2.05 against 1.88 in favour of the share. It does show that the level is not blind on homogeneous content, which is a further reason for the max design.

I reject the spatter half of the hazard on COCO. On 308 held-out images the one-sided C3 scores 0.576 on severity-1 spatter (two-sided 0.522, Section 4 0.547), and 0.867 over all severities (0.845 and 0.880).

Glare and headlights are untested: ACDC is not on disk and needs a GPU pass. The test is pre-registered (C4 must not score below C on ACDC night), and C4 never uses the one-sided arm alone.
- **To E** (partially concede): C is not a second idea: it is identical to A on the screen and held out. Its thesis inherits A's family problem and, in aggregate, points the wrong way (NAP-style 0.302/0.343). Section 4 beats the share on nine families held out. The paper rests on the per-channel, two-sided reading, which C uses but never argues for. E would change their mind if the one-sided 'flatter than neighbours' score held on a held-out family set (real fog, ACDC night). — I concede that it is not a second idea, and I merge into A, with D.

I concede the family split. Nine families favour the level; the largest gaps are elastic (-0.058) and spatter (-0.041). That is why the merged challenger C4 adds the level as a second arm.

I reject 'points the wrong way in aggregate'. Only the average of raw ratios does. The log average points the right way (0.678; fog 0.947, contrast 0.943), and the per-channel readings reach 0.897 two-sided and 0.921 rectified.

The argument for the per-channel reading, which E rightly says was missing, is now the thesis. Dense channels flatten while sparse ones sharpen: Spearman +0.38 to +0.62 for fog and contrast, and 14 of 14 cells each way on the screen and held out. Any signed average therefore cancels the two groups (the signed per-channel mean reaches only 0.695). A two-sided or rectified per-channel reading does not cancel them: rectifying alone adds +0.227 [+0.215, +0.236].

E's requested evidence, in part:
- The flattening row C3, frozen in Round 1, holds on 308 held-out images: 0.912/0.824 (screen 0.921/0.823).
- At severity 1 it matches E's own same-scene oracle from a single image: fog 0.947 vs 0.930 (+0.017 [-0.004, +0.037]), contrast 0.946 vs 0.925 (+0.021 [-0.001, +0.042]).

Real fog has only been probed: A and D find that s1-s2 flatten and s3 does not. Foggy Cityscapes and ACDC night are pre-registered, not done.

**Status:** merged (merged with: A, D)

**Final proposal:**

# Merged proposal A + C + D: Opposite moves. Corruption flattens a frozen detector's dense early channels and sharpens its sparse ones

*Merged from three seats:*
- *A: the row, the ceiling-versus-level mechanism, the explaining-away of the level, and the thresholds.*
- *C: the topology verdict, the ladder from TU, the decomposition of NAP, the opposite moves, the flattening arm, and the registrations.*
- *D: the level as a second axis, the driving plan, the Koschmieder probe, and the band split.*

*The frozen primary is A's and C's Round-1 row, unchanged.*

## Thesis
A corruption moves a frozen detector's early channels in opposite directions: under blur, haze, contrast loss and noise, the dense channels flatten while the sparse ones sharpen. Peakiness averaged over channels therefore points the wrong way. Each channel's log peak share, read against clean scenes like this one, detects corruption from a single image.

## Insight
**The quantity.** For each channel of stages s1-s3:
- the level m, the mean over positions;
- the ceiling t, the mean of its strongest 1%;
- the log peak share u = log t - log m.

Positions are pooled (L2) and channel identity is kept (L3).

**1. Opposite moves (new).** Screen, severity 3, each corrupted image paired with its own clean version.
- **The change of u rises with the channel's clean sparsity** (its bank median u). Spearman across channels:
  - fog +0.38 / +0.61 / +0.50 at s1 / s2 / s3;
  - contrast +0.42 / +0.62 / +0.56;
  - the five blurs +0.23 to +0.52;
  - the four noises +0.21 to +0.41 at s2-s3.
- **Under the seven visibility-loss families,** the densest fifth of channels flattens by 0.93-1.98 bank SDs at s1-s2, and the sparsest fifth sharpens by +0.02 to +0.69. That holds in 14 of 14 cells each way, on the screen and on 308 held-out images.
- **The sparse channels are not a cap artefact.** Their clean t/m is about 12-30, against the cap of 100 reached when only 1% of positions fire, and their 99th percentile is never 0.
- **Proposed mechanism** (not yet checked against the BatchNorm parameters):
  - A dense channel sits above its ReLU threshold, so lost contrast pulls its peaks toward its BatchNorm offset.
  - A sparse channel sits mostly below it, so lost contrast silences its weak responses first and its strongest ones survive.
  - This extends A's "fixed offsets" account to the sparse channels.
- **Exceptions.** Overlays and compression fill the sparse channels' bulk and flatten them (Spearman at s2: snow -0.20, elastic -0.33, JPEG -0.43, spatter -0.43). Brightness and saturate barely move (L10).

**2. Why averaged peakiness points the wrong way.** The Neural Activation Prior (NAP) averages each channel's (max/mean)^2, on the premise that clean images are more peaked. Here it scores 0.302 / 0.343, below chance.
- At s2, 69-75% of channels' ratios fall under fog, contrast, zoom and defocus blur, yet NAP's average rises in 81-100% of images.
- The 5% of channels with the largest clean ratio (mean r^2 1,360, against a median of 49) carry 43-58% of the change, and those channels sharpen.

One ingredient at a time (screen, AUROC common / extra):

| Step | AUROC common / extra |
|---|---|
| NAP, -mean r^2 (top-1% mean in place of the max) | 0.302 / 0.343 |
| without the square | 0.402 / 0.394 |
| log, -mean u, no reference | 0.678 / 0.569 (fog 0.947, contrast 0.943; noise 0.04-0.05) |
| per channel, standardised, conditioned, signed | 0.695 / 0.569 |
| the same, flattening side only (C3) | **0.921** / 0.823 |
| the same, two-sided (the frozen row C) | 0.897 / **0.859** |

- The log flips the sign: +0.276 [+0.263, +0.289].
- Reading each channel by magnitude, or by its flattening side alone, stops the two groups of channels cancelling: rectifying the signed mean adds +0.227 [+0.215, +0.236].

**3. Visibility loss lives in the ceiling, noise in the bulk (A).**
- At severity 1, s1, in conditioned residual SDs, fog / contrast / zoom blur / defocus blur move:
  - the ceiling by -1.89 / -1.92 / -2.04 / -1.96;
  - the level by -0.45 / -0.24 / -0.20 / -0.12.
- On the screen, the level alone reaches 0.871 / 0.874, the ceiling alone 0.853 / 0.819, and their ratio 0.897 / 0.859.
- The shape shift is coherent across s1 channels: 67-74% of its energy lies on the first clean principal direction. Averaging the per-channel deviation therefore beats whitening it, by 0.034 [0.029, 0.039] on held-out images (B's M2).

**4. Why the content reference is needed (A, B).**
- Content predicts the share almost as well as the level (ridge R^2 0.60-0.69, against 0.69-0.77).
- Conditioning adds +0.048 [+0.036, +0.062] to the two-sided row and +0.037 [+0.029, +0.046] to the flattening arm.
- Fog lowers the stage-4 key, so the search recruits dimmer clean scenes. Those are peakier: the clean conditional correlation of level and shape is -0.73 / -0.84 / -0.84.
- So the level's deviation is explained away (Section 4 on fog: 0.546), while the shape's deviation grows (0.876 two-sided, 0.979 flattening).

**5. A single image is enough for mild fog and contrast.**
- On 308 held-out images at severity 1, C3 (frozen in Round 1) reaches fog 0.947 and contrast 0.946.
- E's same-scene oracle, which uses the clean version's key, reaches 0.930 and 0.925. The differences are +0.017 [-0.004, +0.037] and +0.021 [-0.001, +0.042].
- The CDFs reach 0.633 and 0.665.
- The oracle stays far ahead on frost (-0.205), saturate (-0.171), elastic (-0.143) and brightness (-0.128). There, E's aliasing diagnosis holds.

**Topology, plainly (C's seat): it does not earn its place.**
- **Conv-graph persistence:** the top of the diagram is the heaviest edges (L6; 0.661).
- **Euler characteristic of the excursion sets** (40 images x 18 conditions): a real signal (0.777 / 0.784; +0.073 over total variation), but redundant with the means (+0.004 [-0.016, +0.022]).
- **New this round, with a content reference:** the Euler characteristic gains +0.034 [+0.015, +0.051] (to 0.808 / 0.809). Added to the frozen row it still lowers common AUROC by 0.022 [0.006, 0.037]; extra changes by +0.008 [-0.012, +0.028]. Fog stays at 0.63.
- **Neighbourhood-graph agreement:** 0.657-0.674, and it lowers Section 4 to 0.815.
- **Why:** content sets a per-channel gain, not a deformation. The invariant is a log ratio, and the signal is its direction in each channel, which no Betti number sees.
- **The ladder from TU:** top of the diagram 0.661, per-channel top-1% means 0.792, log peak share 0.849, the same against clean scenes like this one 0.897.

**Why it is not obvious.**
- L4 (tails lose to the bulk) holds for raw tails. Relative to the channel's own bulk and to similar scenes, the tail is where visibility loss shows.
- NAP's published prior, transplanted here, scores below chance. The same ratio, read per channel, in logs and against similar scenes, is the strongest single-image monitor in this study.
- E's aliasing gap for fog and contrast is the cost of reading the level, not of reading a single image.

**Lessons.**
- For: L1, L2, L3, L8 (the magnitude is kept in C4's second arm) and L9.
- Against: L10 (brightness and saturate at 0.53 / 0.57 at severity 1) and L11 (C4 was chosen on the screen, so it is registered below).

## Method
Training-free, clean-only and label-free: one forward pass and one score per image.
1. **Taps.** The post-ReLU inputs of `res_layers[s].blocks[1].branch2a.conv`, s1-s3. Per channel, u = log(t + 1e-4) - log(m + 1e-4); an offset of 1e-6 gives the same AUROCs to three decimals.
2. **Content key.** The stage-4 channel means, standardised with the 2,000-image clean bank. N(x) is the 50 nearest bank images (Euclidean). The deviation is e_c = (u_c - mean over N(x) of u_c) / bank sd of u_c.
3. **Primary C** (frozen since Round 1). Per stage, average |e_c| over channels. Z-score each stage with 500 other clean images scored the same way, and add s1-s3.
4. **Registered challenger C4** = max(z(F), z(L)):
   - F is C with max(-e_c, 0) in place of |e_c| (flattening only; this is C3);
   - L is Section 4 (raw means, same neighbours, two-sided);
   - z() standardises each arm with the 500 clean images.

**In three sentences:**
- Read each early channel's log ratio of its strongest responses to its mean.
- Compare it, channel by channel, with the 50 clean scenes the detector's back finds most similar.
- Average the deviations over channels and stages in both directions (C). Or count only the flattening, and flag the image if either that or the channel level departs from those scenes (C4).

**Cost:** one top-k per channel and a kNN over 2,000 x 512, within the 0.2-1.2 ms monitor budget (L12).

## Novelty
**Against the named methods:**
- **Becker et al. (ICPR 2026):** per-channel CDF EMD against one global CDF; scale dominates and there is no content reference. On 859 held-out images C beats it by +0.075 [+0.068, +0.082] common, +0.053 [+0.047, +0.058] extra and +0.105 [+0.095, +0.114] at severity 1.
- **NAP (Wan et al., arXiv 2402.18162; verified by A, C and D):** the same ratio, but raw, squared, averaged over channels, one-sided, at the penultimate layer, for semantic OOD. Here it scores below chance; the opposite moves explain why, and the per-channel log reading reverses it.
- **ASH (ICLR 2023, via SCALE's description) and SCALE (ICLR 2024, verified):** activation-sum ratios across the pooled penultimate channels, used to rescale logits.
- **The retrieval step is not claimed.** It is SPADE's (Cohen & Hoshen 2020; verified by E) and contextual anomaly detection's (Song et al. TKDE 2007; ROCOD, CIKM 2016; E, abstracts).
- **Section 4:** the same reference, read on the level.
- **NMD:** the level against the BatchNorm means.
- **Mahalanobis:** without classes, with Ledoit-Wolf, it is Rippel et al. (ICPR 2020; verified by E). C beats B's log-whitened version by +0.019 [+0.014, +0.023] on held-out images.
- **kNN-OOD and DisCoPatch:** as in the briefing.
- **IQA:** BRISQUE (Mittal, Moorthy & Bovik, TIP 2012; cited by A) and NIQE (my recollection, unverified) fit shape statistics of normalised pixels against pristine images. Ours lives inside the detector, per channel, conditioned on the detector's own content, and read through the opposite moves.

**Claimed:**
- (i) dense and sparse channels move in opposite directions, which explains NAP's inversion;
- (ii) the per-channel two-sided or flattening reading;
- (iii) from a single image, mild fog and contrast are detected as well as with a clean view of the same scene.

## Decisive experiment
Registered in `panelist-C/r3/PREREGISTRATION_R3.md`; the thresholds are A's.
- **Data:** the confirmation's stored means and top-1% means at seed-44 positions 1059-4999, about 3,940 images that no panelist has scored. Positions 200-1058 are reported separately as interim; the pilot's bank and z-statistics images are used.
- **Rows:** C, C2 (log level + C), C3 and C4. For reference: Section 4, the CDFs, A's M1, B's primary and E's (a).
- **Paper:** all of
  - common >= 0.87 and extra >= 0.83;
  - severity-1 common >= 0.77;
  - FPR95 common <= 0.30;
  - fog, contrast and zoom blur at severity 1 each at least 0.08 above the CDFs.
- **Kill:** either of
  - C - Section 4 on common is below +0.015, or its interval touches 0;
  - C - CDFs on extra has an interval that includes 0.
- **Swap:** C4 becomes the headline only if C4 - C on common is above 0 with its interval excluding 0, and C4 - C on extra is >= 0 (point estimate).
- **Mechanism kill:** at severity 3, the densest fifth must flatten in at least 13 of 14 cells and the sparsest fifth sharpen in at least 12 of 14 (7 families x s1-s2). Otherwise the thesis falls back to "visibility loss lives in the ceiling".
- **Single-image claim:** it dies if C3 - E's oracle at severity-1 fog or contrast is below -0.03 with its interval excluding 0.

## Cheapest first test (done, on CPU; screens labelled)
- **Held out, C** (859 images): 0.893 [0.884, 0.902] / 0.858 [0.850, 0.866]; severity 1 0.810; FPR95 0.251 / 0.335. It passes K1-K3 and every one of A's thresholds. Against Section 4: +0.052 [+0.046, +0.059] common, -0.008 [-0.013, -0.002] extra.
- **Held out, C3** (308 images): 0.912 / 0.824; severity 1 0.850; FPR95 0.216 / 0.400. At severity 1: fog 0.947, contrast 0.946, zoom 0.946.
- **Screen, C4** (selected there): 0.931 / 0.871; severity 1 0.874; FPR95 0.186 / 0.327.
  - Against C: +0.034 [+0.025, +0.044] common, +0.013 [+0.004, +0.022] extra, +0.060 [+0.047, +0.074] at severity 1.
  - Against A's M1, which uses a signed flattening arm: +0.010 [+0.001, +0.017] / +0.010 [+0.001, +0.018].

## Figure 1
- **(a) Opposite moves.** Each s2 channel's change of u under severity-3 fog, against its clean sparsity (Spearman +0.61). Each dot is sized by the channel's weight in NAP's average; that weight sits on the sparse channels, which sharpen.
- **(b) The ladder.** NAP 0.302, then the log 0.678, then per channel and conditioned (two-sided 0.897, flattening 0.921), then C4 0.931 (screen), with held-out bars.
- **(c) Severity 1.** Single image (C3) against the same-scene oracle against the CDFs: fog 0.947 / 0.930 / 0.633; contrast 0.946 / 0.925 / 0.665. Frost, saturate, elastic and brightness show where the oracle stays ahead.

## Fit for IEEE IV
A camera-health monitor inside the frozen detector, with one score per frame. The driving plan is D's:
1. **Cityscapes-C:** bank and z-statistics images from train, the 19 x 5 corruptions on val. Global and conditioned scores side by side, as B's test of how much a fixed camera needs the key.
2. **Foggy Cityscapes** (to download), at beta = 0.005 / 0.01 / 0.02.
   - Predicted signs: flattening at s1-s2 and none at s3. In D's Koschmieder probe the s1 share fell in 70-73% of image-channel pairs, with medians of -0.50 / -0.77 / -1.20 SD; s3 moved 0.00 / +0.03 (A's probe: +0.17 / +0.38).
   - Kill: the flattening arm does not beat the level at beta = 0.005 and 0.01. D's 24-image probe is a tie: 0.877 against 0.903, with d' 2.05 against 1.88.
3. **ACDC** fog, night, rain and snow (to download), against its normal-condition references. Night glare is the hazard for the flattening arm, so C4 must not score below C there.
4. **nuImages** night and rain, selected through metadata.
5. **Upper- and lower-band shares** as a secondary (the L2 caveat, Koschmieder).
6. **Video:** an exponential average of the per-frame scores.

## Risks and cost
**Cost:**
- No GPU for COCO, since the confirmation stores means and top-1% means; scoring takes about 5 CPU minutes once it ends.
- Driving data: about 1 h of GPU for Cityscapes-C, plus minutes for Foggy Cityscapes and ACDC, in agreed windows.
- About 40 lines of new code.

**What could go wrong:**
- C4 shrinks out of sample, because it was selected on the screen.
- Brightness and saturate stay capped (L10).
- The mechanism relies on BatchNorm and ReLU, so a claim of generality needs a second backbone.
- The bank must come from the deployment domain.
- Soft or hazy clean photos become false positives.
- The stage-4 key drifts under heavy corruption: its level falls to x0.50-0.95 of clean at severity 5.
- On homogeneous driving content the level may already see fog, which narrows the shape's margin.

**Vote:** first A, second E. First: A, as merged with C and D in this proposal. It is the only row that passes every author's pre-registered criteria on up to 859 held-out images:
- +0.075 [+0.068, +0.082] over the CDFs on common, +0.053 on extra and +0.105 at severity 1;
- +0.052 over Section 4 on common.

Its mechanism is now checked: dense and sparse channels move in opposite directions, which also explains why NAP is inverted. It also carries a registered challenger, C4.

Second: E. Its same-scene oracle is the right yardstick. Measured against the best single-image statistic, it shows exactly what is left: at severity 1 the oracle leads by 0.205 on frost, 0.171 on saturate, 0.143 on elastic and 0.128 on brightness, while fog and contrast are already matched from one image.

B's thesis fails the per-channel control: conditioned log means reach 0.877 / 0.877, against B's 0.882 / 0.858. Its method is Rippel et al. plus a log, so it belongs in the paper's table as the whitened baseline.

**First experiment:** Once the confirmation pass finishes, score the registered rows (C, C2, C3, C4, with Section 4, the CDFs, A's M1, B's primary and E's (a) for reference). Use the stored means and top-1% means at seed-44 positions 1059-4999: about 3,940 images that no panelist has scored. It runs on CPU in about 5 minutes and needs no extra GPU.

Kill: either of these ends the idea.
- C - Section 4 on AUROC common is below +0.015, or its 95% interval touches 0.
- C - CDFs on AUROC extra has an interval that includes 0.

Swap: C4 = max(flattening arm, Section 4) becomes the headline only if C4 - C on common is above 0 with its interval excluding 0, and C4 - C on extra is >= 0.

Mechanism kill: the opposite-moves thesis is dead if, at severity 3 across the seven visibility-loss families at s1-s2, the densest fifth of channels flattens in fewer than 13 of 14 cells, or the sparsest fifth sharpens in fewer than 12 of 14.

**Checks run:** All checks ran on CPU. Scripts and logs are in panelist-C/r3/.

(1) nap_ladder.py (screen, 200 images): a ladder from NAP to the frozen row.
- Rows, AUROC common / extra:
  - NAP: 0.302 / 0.343
  - without the square: 0.402 / 0.394
  - log: 0.678 / 0.569
  - median: 0.691 / 0.562
  - standardised, global: 0.706 / 0.581
  - conditioned, signed: 0.695 / 0.569
  - flattening only: 0.921 / 0.823 (global reference: 0.885 / 0.781)
  - two-sided: 0.897 / 0.859 (global reference: 0.849 / 0.805)
  - sharpening only: 0.616 / 0.737
- Paired differences:
  - log minus raw: +0.276 [+0.263, +0.289]
  - rectified minus signed: +0.227 [+0.215, +0.236]
  - two-sided minus rectified: -0.024 [-0.037, -0.014] / +0.036 [+0.022, +0.047]
  - conditioning: +0.048 [+0.036, +0.062] (two-sided) and +0.037 [+0.029, +0.046] (rectified)
- At s2, under fog, contrast, zoom and defocus blur: 69-75% of channels' r^2 fall, NAP's average rises in 81-100% of images, and the top-5% channels carry 43-58% of the change.

(2) two_populations.py (screen).
- Spearman(clean sparsity, change of u) at severity 3:
  - fog +0.38 / +0.61 / +0.50; contrast +0.42 / +0.62 / +0.56;
  - blurs +0.23 to +0.52; noises at s2-s3 +0.21 to +0.41;
  - at s2: snow -0.20, elastic -0.33, JPEG -0.43, spatter -0.43.
- Densest fifth -0.93 to -1.93 SD and sparsest fifth +0.02 to +0.68 SD at s1-s2: 14 of 14 cells each way.
- Per family, flattening arm: fog 0.979, contrast 0.987, frost 0.906.

(3) two_arm_screen.py (screen, selected).
- C4 = max(z(C3), z(Section 4)): 0.931 / 0.871, severity 1 0.874, FPR95 0.186 / 0.327.
- C4 minus C: +0.034 [+0.025, +0.044] / +0.013 [+0.004, +0.022] / +0.060 [+0.047, +0.074].
- A's M1 reproduced at 0.922 / 0.862 / 0.869; C4 minus M1: +0.010 [+0.001, +0.017] / +0.010 [+0.001, +0.018].

(4) topology_conditioned.py (40 images x 18 conditions, my CPU data, 150-image bank, s3 key).
- Euler characteristic at per-image quantiles: global 0.774 / 0.764, conditioned 0.808 / 0.809 (+0.034 [+0.015, +0.051]).
- Frozen row plus conditioned Euler, minus the frozen row: -0.022 [-0.037, -0.006] / +0.008 [-0.012, +0.028].
- Fog stays at 0.63.

(5) sparsity_proxy.py (bank only): the sparsest fifth has a median u of 2.45 / 3.02 / 3.40, against the cap of 4.61, and its 99th percentile is never 0.

(6) heldout_c3_families.py (308 held-out images, positions 200-507, frozen rows only).
- C: 0.898 / 0.862.
- C3: 0.912 / 0.824, severity 1 0.850; at severity 1, fog 0.947, contrast 0.946, spatter 0.576.
- Opposite-moves signs: 14 of 14 and 14 of 14 cells.

(7) c3_vs_oracle.py (the same 308 images), C3 minus E's same-scene oracle at severity 1:
- fog +0.017 [-0.004, +0.037]; contrast +0.021 [-0.001, +0.042];
- frost -0.205, saturate -0.171, elastic -0.143, brightness -0.128.

I wrote PREREGISTRATION_R3.md before any Round-3 row was computed on a held-out image. C4 has not been computed at any position from 200 on.

## D (Vehicles): Level and crest: content-conditioned peak-to-level ratios of early detector channels

**Thesis:** Content sets how strongly a scene drives a detector's early filters but corruption changes how peaked that drive is, so judging each early channel's crest (top-1% mean ÷ mean) alongside its level, against clean scenes with the same late-layer content, detects corruption better than full activation distributions, and catches the fog and contrast loss that level-only monitors miss.

### Round 1 proposal

## Thesis

Content sets how strongly a scene drives the detector's early filters; corruption changes how *peaked* that drive is. So judge each early channel's **crest** (top-1% mean ÷ mean |x|), not only its **level**, against clean scenes with the same content.

## Insight

**Why it is true.** Two numbers per channel separate two things.
- **Content mostly moves the level.** On the 2,000 clean bank images, the across-image std of log mean |x| is 0.32 / 0.34 / 0.47 at s1 / s2 / s3. For the log peak (top-1% mean) it is only 0.20 / 0.19 / 0.25, and the two co-vary (r = 0.55–0.73). How much of a frame is textured varies a lot; every natural image has some strong edges.
- **Visibility loss cuts exactly those peaks.** At s1 and severity 3, in clean-SD units, the crest moves 1.25 under fog against 0.63 for the level. Contrast: 1.63 vs 0.77. Zoom blur: 1.23 vs 0.49. Defocus: 1.69 vs 0.97.
- **Noise and spatter move the level more:** 2.28 vs 1.63, and 0.64 vs 0.51. The two axes are complementary.

**Why it is not obvious.** L4 and L5 say tails lose and that one number per channel is nearly everything. That holds for *raw* tails: the top-1% mean against the global average reaches only 0.792 / 0.749. It does not hold for the tail *relative to the level*. Also, the idea to beat reads levels only, and it is near chance on fog (0.546; unconditioned 0.511) and weak on contrast (0.681). The s1–s3 levels barely see fog at all.

**Lessons.**
- Supported by L1 (front only), L3 (per channel, channel identity kept), L8 (magnitude kept in the level term; the crest is scale-free by design) and L9 (content conditioning).
- Could be killed by L11 (the crest was the best of 10 rows on the screen) and L10 (brightness and saturate stay capped).

**Candidates I stress-tested:**
1. **Local, per-cell content conditioning (my first bet). Tested; it fails.** It reaches 0.729 / 0.799, −0.112 [−0.125, −0.100] against image-level conditioning. Fog falls to 0.351 and contrast to 0.263, below chance: textureless clean cells (sky, walls) explain foggy cells away. The anchor must stay image-level.
2. **A magnitude-free key. Killed by a drift check:** under fog, the key's direction moves as much as the raw key.
3. **Level + crest. Chosen.**
4. **Position as a free key on fixed cameras.** On COCO, position-only cells reach 0.699 (L2); kept as an IV ablation.
5. **Coherent temporal integration.** Runner-up.
6. **Re-corruption probes (re-blur, re-JPEG).** Need extra passes and knowledge of the families; dropped.
7. **Input photometry conditioned on content.** Runner-up.
8. **Max instead of sum over stages.** Fog peaks at C1 and JPEG at C4 in the 5,000-image CDF stages. This is a better row, not an idea.

## Method

Training-free, clean-only, label-free; one forward pass.

1. **Read each early channel twice.** At s1–s3 (the inputs of `res_layers[s].blocks[1].branch2a.conv`, post-ReLU), compute for each channel c:
   - the level m_c = mean |x| over positions;
   - the peak t_c = mean of the largest 1% of |x|;
   - the crest r_c = t_c / m_c.
2. **Find the content neighbours.** The key is the s4 channel-mean vector, standardised with the 2,000-image clean bank. N(x) is the set of the k = 50 nearest bank images (Euclidean). This step is identical to the idea to beat.
3. **Score each stage.**
   - Level: L_s = mean_c |m_c − μ_N(m_c)| / σ_bank(m_c).
   - Crest: C_s = mean_c |r_c − μ_N(r_c)| / σ_bank(r_c).
4. **Combine.** Z-score the six stage scores with the 500 clean z-statistics images, scored the same way. The image score is S(x) = Σ_s z(L_s) + Σ_s z(C_s), with equal weights and nothing tuned.

In three sentences: read each early channel twice, its level and its crest. Find the 50 clean training scenes whose late-layer content is closest. Add how far both numbers fall from those scenes, in clean units.

## Novelty

- **Becker et al. ICPR 2026** compares 1,000-bin per-channel CDFs with one *global* training CDF. Its EMD mixes level and shape.
  - At matched stages (C2–C4) it reaches 0.830 / 0.808 on the screen; level + crest reaches 0.876 / 0.870.
  - The crest alone, *unconditioned*, matches the full CDF monitor (0.835 / 0.805) with one number per channel.
- **Neural Mean Discrepancy** compares levels with BatchNorm means: global, no shape.
- **Mahalanobis** (Lee 2018) uses class-conditional Gaussians on pooled levels.
- **Gram matrices** (Sastry & Oore 2020) use higher-order Gram matrices with per-class bounds. This detail is my recollection and unverified. They are not scale-normalised, and they condition on class, not content.
- **kNN-OOD** normalises the pooled vector *across* channels; it reaches 0.594 here.
- **SCALE** (Xu et al., ICLR 2024; verified) and **ASH** (as SCALE describes it) use r = total activation ÷ activation above the p-th percentile.
  - Their ratio is taken over the D channels of the pooled penultimate vector, rescales logits, and targets semantic OOD.
  - Ours is per channel over spatial positions, at early layers, for covariate shift, against content-matched clean scenes.
- **Sparsity as a Key** (Oh et al., CVPR 2026; checked at abstract level) trains a top-k sparse autoencoder on ViT [CLS] tokens, for semantic OOD.
- **Image-quality assessment:** kurtosis- and sparsity-based sharpness measures in the pixel, DCT and wavelet domains are classical (checked at title level). We move the principle into the frozen detector's own channels and condition it on the detector's own content.
- **The idea to beat** has the same key and neighbours; we compare a second number. On the screen:
  - +0.035 [+0.026, +0.043] common, +0.000 [−0.007, +0.007] extra, +0.027 at severity 1;
  - fog 0.546 → 0.739, contrast 0.681 → 0.863.
- **DisCoPatch** is trained, and reaches 0.760 / 0.747.

## Decisive experiment

**The run.** One GPU pass stores levels and top-1% means at s1–s4 for the 5,000 × 96 variants, the bank and the z-statistics images. Every choice is frozen now: k = 50, the s4-means key, s1–s3, the top 1%, equal weights. The result is judged on the 4,800 images outside the screen.

**What makes the paper:**
- level + crest beats the activation CDFs (0.821 / 0.807) on both headline numbers, with paired intervals excluding 0. The screen predicts about +0.05: +0.051 [+0.034, +0.068] common and +0.062 [+0.049, +0.075] extra;
- severity-1 common ≥ 0.76 (CDFs 0.707);
- FPR95 common ≤ 0.33 (CDFs 0.413);
- fog ≥ 0.75 and contrast ≥ 0.85 (CDFs 0.708 / 0.842).

**What kills it:**
- (a) Level + crest minus level-only, on the same 4,800 images and the common families, gains less than +0.015 or has an interval touching 0. The crest then adds nothing, and we revert to the idea to beat.
- (b) The IV kill: on Foggy Cityscapes, the crest term does not beat the level term at β = 0.005 and 0.01.

## Cheapest first test

Run today on CPU, on the stored 200-image screen. The crest was the best of the 10 rows I tried, so this is a screen, not a proof.

| Row | Common ↑ | Extra ↑ | Sev 1 ↑ | FPR95 ↓ |
|---|---|---|---|---|
| Level + crest, conditioned | 0.876 [0.859, 0.892] | 0.870 [0.856, 0.883] | 0.790 | 0.295 |
| Crest, conditioned | 0.876 | 0.849 | 0.786 | 0.288 |
| Level only (idea to beat) | 0.841 | 0.870 | 0.763 | 0.368 |
| Activation CDFs | 0.825 | 0.808 | 0.716 | 0.404 |

**Robustness:**
- The p99/mean twin gives 0.872 / 0.852.
- Each stage alone reaches 0.851–0.861.
- k = 10 / 50 / 200 gives 0.869 / 0.876 / 0.870.
- Each half of the screen gains about the same over the idea to beat: +0.036 and +0.034.
- Conditioning adds +0.041 [+0.029, +0.052] to the crest.

## Figure 1

The (Δlevel, Δcrest) plane at s1, both content-conditioned and in clean-SD units:
- **Clean val images** form a tight cloud at the origin.
- **Each family is an arrow** from severity 1 to 5. Noise, JPEG and spatter run along the level axis; fog, contrast and blur run along the crest axis. Brightness and saturate barely move (L10).
- **Inset:** one s1 channel on a clean and a fogged street image. The mean is the same; the peaks are flattened.

## Fit for IEEE IV

**What it rescues.** The families it fixes (fog, contrast loss, defocus, zoom blur) are the road's visibility failures: fog, haze, veiling glare, a smeared windshield, an out-of-focus camera.

**Driving-data plan.** The bank is 2,000 clean Cityscapes train images.
1. Cityscapes-C on val, paired.
2. Foggy Cityscapes at three β, paired. Its fog is physics-based and depth-dependent, which makes this the decisive IV test.
3. ACDC fog, night, rain and snow, against ACDC's normal-condition reference images of the same routes.
4. The nuImages night and rain splits, if the metadata allows.

**L2's caveat** becomes one ablation: an upper-half and a lower-half crest per channel, because fog attenuates far structure first (Koschmieder's law).

**Latency.** One top-k per stage plus a 2,000 × 512 neighbour search fits inside the 0.2–1.2 ms monitor budget (L12). A sort-free ‖x‖₄/‖x‖₁ ratio is the embedded variant to validate.

**Video.** The per-frame score feeds a short exponential average.

## Risks and cost

**Cost:**
- GPU: about 2.5–3 h for the 5,000 images, bound by corruption generation on the CPU. If the running confirmation stores only channel means, the crest needs this pass of its own.
- About 40 min for Cityscapes-C, and minutes for Foggy Cityscapes and ACDC.
- New code: about 40 lines. `channels.py` already computes the top-1% statistic, and `content_conditioned.py` already has the scorer.

**What could go wrong:**
- **Real fog may spare the peaks.** Near edges keep their contrast (t = e^{−βd} ≈ 1), so the crest may move less than under the package's fractal fog. The band split hedges against this.
- **Night glare raises peaks.** The two-sided |Δ| catches it, but a lit clean street may look corrupted.
- **Brightness and saturate stay weak** (0.686 / 0.728).
- **The s3 top 1% is noisy:** it is only 16 positions.
- **Selection on the screen (L11):** only the 4,800 held-out images settle it.

**Quick check:** All checks ran on CPU on the stored data (the 200-image pilot plus the stored 4×4 grids), with k = 50 and the idea to beat's recipe. Scripts are in /tmp/claude-1000/-home-yuchen-YuchenZ-UE-philip-sa/3d52b916-b3ff-4806-a501-7f70cde31a29/scratchpad/detection/roundtable/panelist-D/.

1. **Local content conditioning, my first bet: negative** (local_content_screen.py, using the 4×4 grids of the bank, the z-statistics images and the tests).
   - My reproduction of the idea to beat gives 0.841 / 0.870, matching the briefing.
   - Per-cell keys with 32,000 bank cells give 0.729 / 0.799, −0.112 [−0.125, −0.100] common against image-level conditioning.
   - Fog falls to 0.351 and contrast to 0.263, below chance.
   - Variants: compositional 0.806 / 0.849; position + content 0.732 / 0.804; position only 0.699 / 0.799; max over cells 0.655 / 0.720.
2. **Key drift** (key_drift.py). Under fog the s4 key moves 0.43 / 0.57 / 0.72 nearest-neighbour distances at severities 1 / 3 / 5. A magnitude-free key moves about the same (0.40 / 0.56 / 0.71). The unconditioned level is also blind to fog (0.511).
3. **Statistics screen** (visibility_stats.py): 5 statistics × {global, conditioned} = 10 rows; selection caveat applies.
   - Conditioned crest (top/mean): 0.876 / 0.849, fog 0.825, contrast 0.917.
   - Conditioned level: 0.841 / 0.870, fog 0.546, contrast 0.681.
   - Raw top-1% global: 0.792 / 0.749.
4. **Bootstrap** (crest_bootstrap.py; 1,000 paired draws, seed 44).
   - Level + crest: 0.876 [0.859, 0.892] / 0.870 [0.856, 0.883]; severity 1: 0.790; FPR95 0.295 (my implementation).
   - Level + crest − idea to beat: +0.035 [+0.026, +0.043] common, +0.000 [−0.007, +0.007] extra, +0.027 [+0.017, +0.037] at severity 1.
   - Level + crest − CDFs: +0.051 [+0.034, +0.068] common, +0.062 [+0.049, +0.075] extra.
   - Crest − CDFs: +0.050 / +0.041.
   - k = 10 / 50 / 200: 0.869 / 0.876 / 0.870. Each stage alone: 0.851–0.861. The p99/mean twin: 0.872 / 0.852.
5. **Mechanism** (crest_mechanism.py).
   - Clean across-image std of log level vs log peak: 0.32 / 0.20 at s1, 0.34 / 0.19 at s2, 0.47 / 0.25 at s3; corr(log level, log peak) = 0.55 / 0.65 / 0.73.
   - Severity-3 effect sizes at s1, crest vs level: fog 1.25 vs 0.63; contrast 1.63 vs 0.77; zoom blur 1.23 vs 0.49; noise 1.63 vs 2.28.
6. **CDFs at matched stages** (cdf_matched_stages.py). C2–C4 z-sum: 0.830 / 0.808; best post-hoc subset C1–C4: 0.839 / 0.826.
7. **Split halves** (split_half.py). Level + crest beats the idea to beat by +0.036 and +0.034 on common in the two halves.

The repository was not modified, and the GPU was not used.

**Runners-up:**
- Coherent temporal integration (IV extension): On driving video, average the content-conditioned residual vectors (not the scores) over a short window of frames: a persistent degradation (fog, dirt, darkness) adds up while content residuals cancel as the scene moves. It needs sequences such as Cityscapes snippets or nuImages sweeps, so it extends the paper rather than carrying the COCO headline.
- Content-conditioned input photometry: For the families the detector is trained to ignore (brightness, saturate; L10, still 0.686 / 0.728 here), compare the input image's black point and colourfulness with the same 50 content-matched clean scenes; it costs almost nothing, but it reads pixels rather than the detector.

### Round 2: critiques written by D

**On A** (serious)

- Steelman: One scale-free number per early channel, log(top-1% mean / mean), judged against the 50 content-matched clean scenes, fixes exactly the families where Section 4 fails (fog, contrast, zoom blur). A also explains why: the key that absorbs fog's level shift recruits neighbours that make the shape deviation larger.
- Objection: A's IV prediction does not hold on my driving-data probe. It said Foggy Cityscapes and ACDC fog separate on shape while monitors that read the level struggle.
- Depth-dependent (Koschmieder) fog does lower the ceiling, so the direction survives at s1–s2.
- But per unit of level lost it moves the shape about 30% less than the package fog does, and it does not move s3 at all.
- On homogeneous driving content the plain level is no longer blind. 'Shape, not level' looks like a property of COCO's content diversity, not of fog.

Secondary points:
- The clean level–peakiness anti-correlation (−0.74 to −0.88) follows arithmetically from sd(log top) < sd(log level) with ρ ≈ 0.55: −0.78 comes out of the marginals that C and D report. So K3's correlation clause is not independent evidence.
- A does not cite the Neural Activation Prior (NAP). I verified it: the same per-channel max/mean statistic.
- Failure scenario: On Foggy Cityscapes with a Cityscapes-train bank, the plain level scores 0.78 / 0.90 at β = 0.005 / 0.01 and the shape 0.76 / 0.88. The paper's IV headline ('level monitors struggle on fog') is then refuted on the venue's canonical dataset. A reviewer who also runs NAP calls the COCO result 'NAP at early layers with kNN'.
- Evidence: Held out (773 images, positions 200–972, pre-registered in panelist-D/r2/PREREGISTRATION_R2.md before scoring). The A/C row:
- 0.891 [0.881, 0.899] common / 0.856 [0.848, 0.864] extra, severity-1 0.808, FPR95 0.255.
- Against the CDFs: +0.075 [+0.067, +0.083] / +0.054 [+0.047, +0.059].
- Against S4: +0.051 [+0.045, +0.058].
- Severity-1 fog / contrast / zoom: +0.164 / +0.153 / +0.102 over the CDFs.
- Every criterion A set passes, K1–K3 included.

Koschmieder probe (koschmieder_probe.py; 24 Cityscapes val images against a 48-image train global reference, ±0.07):
- Level 0.781 / 0.903 / 0.976 vs log share 0.760 / 0.877 / 0.976 at β = 0.005 / 0.01 / 0.02.
- s1 shift Δlevel / Δshare: −0.38 / −0.77 σ at β = 0.01, against −0.81 / −2.28 for package fog at severity 1.
- s3 share: +0.00 to +0.03.

NAP (Wan et al., arXiv 2402.18162), read from the HTML: S = mean_j (max/mean)², penultimate layer, semantic OOD.
- Would change my mind: Official Foggy Cityscapes and ACDC fog, with a 2,000-image Cityscapes-train bank, where A's row beats both the plain and the conditioned level rows by ≥ 0.03 at β = 0.005 and 0.01, with paired intervals excluding 0.

**On B** (serious)

- Steelman: A key-free, closed-form, pre-registered monitor: log channel energies, whitened per stage with a Ledoit–Wolf covariance. It costs about 10 µs and passes its own held-out rule.
- Objection: Whitening is the wrong geometry for the strongest corruption signal, so 'the evidence is a residue in quiet directions' does not generalise.
- Gain and blur corruptions flatten all channels together, which is along the dominant clean direction. Whitening divides exactly that direction out.
- Whitening the log peak share costs 0.04–0.05 AUROC and most of the fog signal.
- Held out, B is beaten by the per-channel conditioned shape, and it adds nothing over the CDFs on severity-1 fog and contrast, the IV-canonical degradations.
- What novelty remains is 'Rippel et al. 2020 plus a log'.
- Failure scenario: An IV reviewer checks severity-1 fog and contrast: B scores 0.632 / 0.668, the CDFs 0.634 / 0.665. The new monitor is no better than the baseline on haze, and 'log-Mahalanobis on pooled channel means' reads as incremental next to a shape row that beats it by +0.028 at severity 1.
- Evidence: Held out (773):
- B 0.873 [0.865, 0.882] / 0.852 [0.845, 0.859], severity-1 0.780, FPR95 0.288.
- B − CDFs +0.058 [+0.050, +0.065] / +0.049 [+0.043, +0.055], so B's own rule is supported.
- A/C − B: +0.018 [+0.013, +0.023] common, +0.004 [+0.000, +0.009] extra, +0.028 [+0.021, +0.035] at severity 1.
- Severity-1 fog / contrast: B 0.632 / 0.668, CDFs 0.634 / 0.665, A/C 0.798 / 0.818.

Screen (merge_screen.py, 200 images):
- Whitened log share: 0.848 / 0.826 without a key, 0.858 / 0.832 with a key and leave-one-out residual covariance, against 0.897 / 0.859 for the per-channel (diagonal) average.
- Fog at severity 1: 0.635 / 0.660 against 0.800.
- A/C + B summed: 0.894 / 0.862, fog at severity 1 0.739.
- Would change my mind: A whitened variant that keeps the common-mode shape signal (for example, whitening only after removing the first principal direction) and beats the A/C row on unseen images. Or B beating the A/C row on Cityscapes-C or Foggy Cityscapes, where my probe suggests level-type statistics are stronger.

**On C** (fixable)

- Steelman: The same conditioned log peak share as A, reached honestly. Three controlled topology tests show topological invariants are redundant (+0.004 [−0.016, +0.022]). A ladder from TU's 'top of the diagram' to the conditioned share explains why the project's first idea failed.
- Objection: The thesis overclaims, and its one-sided reading is NAP's prior, which fails on these data.
- 'Far better than activation magnitudes' holds only on COCO's common families. Held out, the shape is below the level on the extra families, and on driving-data fog it ties the level in my probe.
- 'Corruption flattens', read one-sided, is NAP's premise. NAP's published score, transplanted here, is below chance: the mean of raw ratios gets peakier under corruption.
- The fact that actually works is the two-sided, log, per-channel deviation from content-matched scenes.
- C's pre-registered one-sided secondary (0.921 / 0.823) is an IV hazard: glare and spatter push the share up.
- Otherwise C is identical to A, to three decimals.
- Failure scenario: On night driving, headlights and street lamps raise each channel's peaks. Rain spatter on the lens does the same. The one-sided 'flatter' score rates these frames cleaner than clean. A reviewer also shows NAP's one-sided score at 0.30 AUROC and asks why 'corruption flattens' is the thesis.
- Evidence: Held out (773), the A/C row passes all of C's criteria:
- vs S4 common +0.051 [+0.045, +0.058], above C's +0.015 bar;
- vs the CDFs on extra +0.054 [+0.047, +0.059];
- severity-1 0.808;
- but vs S4 on extra −0.008 [−0.014, −0.003];
- ε = 1e-4 and 1e-6 give an identical 0.891 / 0.856.

Screen (nap_baseline.py), NAP one-sided with the top-1% mean standing in for the max:
- s1–s3 z-summed 0.302 / 0.343;
- per stage: s1 0.396, s2 0.342, s3 0.279, s4 0.313.

Koschmieder probe: level 0.903 vs share 0.877 at β = 0.01.
- Would change my mind: C restates the thesis as the two-sided, content-conditioned deviation, reports NAP as a baseline, and shows the one-sided secondary holding on ACDC night and on spatter.

**On E** (serious)

- Steelman: The same-scene oracle shows the early channels do register mild fog, contrast and zoom blur (held out: 0.975 / 0.948, severity-1 fog 0.931). So the cap lies in the reference, not in the detector, and splitting each failure into aliasing and blindness tells IV engineers which ones need context.
- Objection: 1. Aliasing depends on the statistic, so the headline that a single image cannot tell a hazy scene from a fogged one is refuted. From one image, the log peak share recovers about two thirds of the severity-1 fog, contrast and zoom gap that E attributes to the scene.
2. The deployable same-scene references fail in exactly the IV cases:
   - A recent frame is degraded too when fog or lens dirt is there from the first frame, and it drifts along with slow onsets (thickening fog, accumulating dirt).
   - A same-place ACDC image differs in parked cars, traffic and season. E's own blend curve falls from 0.977 to 0.868 as the reference moves to another scene.

Also:
- The 'leak-free other-scene key' chooses its scene with the clean version's key, so it is an oracle row too.
- E's single-image row is, by E's own account, Rippel + SPADE, and it is the weakest held-out primary.
- Failure scenario: A car drives into slowly thickening fog, or the lens gathers dirt over an hour. The recent-frame reference degrades along with the image, the residual stays near zero, and the monitor never fires. Only the single-image score, which E deprioritises, can catch it.
- Evidence: Held out (773), severity-1 fog / contrast / zoom:
- oracle on means: 0.931 / 0.923 / 0.988;
- E's single-image row (a): 0.567 / 0.581 / 0.692;
- A/C single image: 0.798 / 0.818 / 0.885, which recovers 63% / 69% / 65% of the gap;
- oracle on the log share: 0.963 / 0.965 / 0.994.

E's row (a):
- 0.846 / 0.850, FPR95 0.335;
- against the CDFs +0.031 [+0.021, +0.041] / +0.048 [+0.039, +0.056], and +0.066 [+0.055, +0.077] at severity 1, so E's single-image rule passes;
- A/C − E is about +0.045 common (point estimate).

In E's final_rows.py, `other` is chosen from the clean version's key.
- Would change my mind: A deployable reference that reaches ≥ 0.90 on severity-1 fog and contrast without the clean version, for example the frames before a synthetic slow onset in driving video, or ACDC pairs. Or an oracle gap that stays ≥ 0.30 for the shape statistic too.

**Defense of own:** The strongest attack on D is that it is a strictly worse variant of A and C. I concede it. Held out (773 images), my frozen row (level + raw crest):
- 0.871 [0.862, 0.880] / 0.865 [0.858, 0.873], severity-1 0.785, FPR95 0.291.
- It passes my kill (a): +0.032 [+0.028, +0.036] over Section 4.
- It beats the CDFs by +0.056 [+0.048, +0.064] / +0.063 [+0.057, +0.069].
- It misses my own fog target: 0.746 < 0.75.
- The A/C row dominates it on common: A/C − D = +0.020 [+0.016, +0.023] common, −0.009 [−0.012, −0.007] extra.

Two of my choices were wrong:
- **The raw ratio.** Near-dead channels blow up top/mean, and the log ratio beats my raw crest by +0.017 [+0.015, +0.019].
- **Summing a fog-blind level.** It dilutes fog. Severity-1 fog falls from 0.757 for the crest alone to 0.670 with the level added; level + log shape gives 0.705, against 0.798 for the A/C row.

I withdraw my statistic and adopt the A/C row as the single primary. A per-stage max with B's level adds nothing on the screen (0.897 / 0.860, FPR95 0.242 vs 0.253), so the level is reported as a separate axis, not summed.

What survives from D is the IV programme, and the probe shows it is not a formality:
1. **Depth fog keeps the direction but loses size.** Under depth-dependent fog the shape still falls at s1–s2: in 72% of image-channel pairs at β = 0.005–0.02, against 83% for package fog. Per unit of level lost it falls about 30% less, and s3 does not respond.
2. **Driving content is homogeneous, so the plain level stops being blind.** On Cityscapes, package fog at severity 1 is caught by the plain level at 0.960, against about 0.51 on COCO. The shape's fog advantage therefore has to be shown, not assumed. My own IV kill (b) is close to triggering on the probe: level 0.781 / 0.903 vs shape 0.760 / 0.877.
3. **The score must stay two-sided.** Glare, spatter and noise raise the share, and NAP's one-sided form scores 0.302 here.

What I would change: D becomes the pre-registered IV section of the merged paper.
- Datasets: Cityscapes-C with a 2,000-image Cityscapes-train bank; official Foggy Cityscapes at β = 0.005 / 0.01 / 0.02; ACDC fog, night, rain and snow against its normal-condition images.
- Baselines: the plain level, Section 4, B's whitened log level, the CDFs and NAP.
- Kill: if the shape does not beat the best level baseline on Foggy Cityscapes at β ≤ 0.01 with intervals excluding 0, the claim is restricted to diverse content, and the paper says so.
- Also kept: the latency budget (one top-k per stage) and a per-frame exponential moving average.

**Best other proposal:** A. If D were off the table I would bet on A's write-up of the A/C row.
- On 773 held-out images it passes every pre-registered criterion A set: 0.891 / 0.856, +0.075 / +0.054 over the CDFs, severity-1 0.808, FPR95 0.255, and severity-1 fog / contrast / zoom each at least 0.10 above the CDFs.
- A also has the best mechanism: the key that absorbs fog's level shift makes the shape deviation larger.

C is the same method; A is the better framing. A's IV prediction needs the driving tests with level baselines before it can be claimed.

**Merge suggestion:** No proposal drifts into predicting performance drop; all five are judged on AUROC and FPR95.

The merge is one paper built on the A/C row (conditioned log peak share, s1–s3, k = 50, stage-4 key):
- **Primary row:** the A/C row, unchanged.
- **Mechanism (A):** gain against the frozen BatchNorm offsets lowers each channel's ceiling. This also explains why Section 4's key absorbs fog.
- **Origin paragraph (C):** the ladder from TU's 'top of the diagram' to the conditioned share, and the topology negative.
- **Analysis figure (E):** the aliasing/blindness decomposition, computed for both statistics. Held out, the shape cuts the severity-1 fog gap from 0.364 to 0.165 and the contrast gap from 0.342 to 0.147, which turns E's scepticism into the paper's argument.
- **Strongest level baseline (B):** B's log-whitened means, plus the negative result that whitening the shape costs 0.04–0.05, because the shape signal is common-mode.
- **NAP as a baseline:** cited (verified); its one-sided published form is below chance here (0.302).
- **IV section (D):** Cityscapes-C, official Foggy Cityscapes and ACDC, with the plain level and B as baselines. Fixed-camera driving content makes the level much stronger, so the shape's IV advantage has to be shown.

Positions 200–972 are now seen by the panel. Any Round-3 revision, such as level + shape or a per-stage max, must be judged only on positions ≥ 973 of the finished confirmation. The frozen rows' 4,800-image test stays valid.

**Ranking:**
1. A: Its frozen row passes every one of its own pre-registered criteria on 773 held-out images (0.891 / 0.856, +0.075 / +0.054 over the CDFs, severity-1 fog +0.164), and it explains why Section 4 fails on fog.
2. C: The same row, with the same held-out pass and an honest topology negative. It ranks below A because its thesis overclaims against magnitudes (−0.008 on extra, a tie on driving-data fog) and its one-sided reading is NAP's prior, which is below chance here.
3. B: A distinct, pre-registered idea confirmed held out (+0.058 / +0.049 over the CDFs). But the shape row beats it (+0.018 common, +0.028 at severity 1), it adds nothing on severity-1 fog or contrast, and whitening destroys the shape signal.
4. D: Mine. It passes its own kill held out (+0.032 [+0.028, +0.036] over Section 4), but the A/C row dominates it on common (−0.020), it misses its own fog target (0.746), and my probe nearly triggers its IV kill. What survives is its IV programme.
5. E: A valuable diagnostic, but not a deployable detector. Its single-image row is the weakest held-out primary (0.846 / 0.850) and not novel, and about two thirds of its 'aliasing' gap is a property of the means statistic.

**Checks run:** All checks ran on CPU only: niced, thread-limited, no GPU, repository untouched. Scripts and logs are in /tmp/claude-1000/-home-yuchen-YuchenZ-UE-philip-sa/3d52b916-b3ff-4806-a501-7f70cde31a29/scratchpad/detection/roundtable/panelist-D/r2/. Two checks ran past the ~3-minute guideline (308 s and 358 s) because the machine was at load ~22 on 16 cores.

**(1) Held-out test of the frozen Round-1 primaries** (heldout_frozen_rows.py; pre-registered in PREREGISTRATION_R2.md before scoring).
- **Data:** the running confirmation (runs/coco-baselines/test_convtu_means) now stores means and top-1% means at s1–s4. Its first 200 images equal the pilot's stored statistics exactly (max |diff| = 0.0). Held out: 773 images, positions 200–972. Bootstrap: 1,000 paired draws, seed 44.
- **Rows** (common / extra, severity-1 common, FPR95 common):
  - S4 (Section 4): 0.840 / 0.864, 0.761, 0.358.
  - A/C row: 0.891 / 0.856, 0.808, 0.255.
  - D (mine): 0.871 / 0.865, 0.785, 0.291.
  - D's raw crest alone: 0.873 / 0.849.
  - Level + log shape: 0.882 / 0.868.
  - Log gain + log shape: 0.886 / 0.865.
  - Log share without key: 0.847 / 0.808.
  - B: 0.873 / 0.852, 0.780, 0.288.
  - E (a): 0.846 / 0.850, 0.769, 0.335.
  - CDFs: 0.815 / 0.802, 0.703, 0.419.
  - Oracle on means: 0.975 / 0.948.
  - Oracle on log shape: 0.968 / 0.930.
- **Paired differences** (common / extra):
  - A/C − CDFs: +0.075 [+0.067, +0.083] / +0.054 [+0.047, +0.059]; +0.104 at severity 1.
  - A/C − S4: +0.051 [+0.045, +0.058] / −0.008 [−0.014, −0.003].
  - D − S4: +0.032 [+0.028, +0.036] / +0.001.
  - A/C − D: +0.020 [+0.016, +0.023] / −0.009 [−0.012, −0.007].
  - Level + log shape − A/C: −0.008 / +0.012.
  - B − CDFs: +0.058 [+0.050, +0.065] / +0.049 [+0.043, +0.055].
  - A/C − B: +0.018 [+0.013, +0.023] / +0.004; +0.028 at severity 1.
  - E − CDFs: +0.031 [+0.021, +0.041] / +0.048 [+0.039, +0.056].
  - S4 − CDFs: +0.024 [+0.015, +0.034] / +0.062.
- **Severity-1 fog / contrast / zoom blur:**
  - S4: 0.528 / 0.540 / 0.740.
  - A/C: 0.798 / 0.818 / 0.885.
  - D: 0.670 / 0.696 / 0.818.
  - B: 0.632 / 0.668 / 0.799.
  - E: 0.567 / 0.581 / 0.692.
  - CDFs: 0.634 / 0.665 / 0.783.
  - Oracle on means: 0.931 / 0.923 / 0.988.
  - Oracle on shape: 0.963 / 0.965 / 0.994.
- **Verdicts by each proposal's own rules:** A and C pass all of their criteria; B is supported; E's single-image rule passes; D passes kill (a) but misses its fog target (0.746).

**(2) Koschmieder probe** (koschmieder_probe.py, 358 s). 24 Cityscapes val images; depth from a flat-ground, vertical-object model built from gtFine labels, with camera parameters from my unverified recollection; 48 train images as a global reference.
- Median transmission: 0.91 / 0.82 / 0.68 at β = 0.005 / 0.01 / 0.02.
- s1 shifts, Δlog level / Δlog top / Δlog share in σ units: −0.38 / −1.34 / −0.77 at β = 0.01, against −0.81 / −3.24 / −2.28 for package fog at severity 1.
- s3 share: about 0.
- AUROC, level vs log share: 0.781 vs 0.760, 0.903 vs 0.877, 0.976 vs 0.976 at the three β; package fog at severity 1: 0.960 vs 0.986.

**(3) Merge screen** (merge_screen.py, 200 images, selection risk), common / extra and fog at severity 1:
- A/C row: 0.897 / 0.859, fog 0.800.
- Whitened log share, no key: 0.848 / 0.826, fog 0.635.
- Whitened log share with key: 0.858 / 0.832, fog 0.660.
- B: 0.876 / 0.856, fog 0.626.
- A/C + B summed: 0.894 / 0.862, fog 0.739.
- A/C + S4: 0.890 / 0.874, fog 0.702.
- Per-stage max of A/C and B: 0.897 / 0.860, fog 0.776, FPR95 0.242.

**(4) NAP baseline** (nap_baseline.py, screen). NAP was verified from arXiv 2402.18162. One-sided, with the top-1% mean standing in for the max:
- s1 0.396, s2 0.342, s3 0.279, s4 0.313;
- s1–s3 z-summed: 0.302 / 0.343, below chance.

### Round 3: rebuttal by D

- **To A** (concede): My two design choices both cost signal, and the cost lands on fog. The raw ratio loses to the log ratio. The equal-weight sum with a level that cannot see fog dilutes the crest. My row misses my own bar of fog >= 0.75 (0.739 on the screen, 0.747 on 859 held-out images), and the log version of the same sum (E2) beats it on every headline. My mechanism ('content mostly moves the level') also overstates: the log crest varies 70-89% as much as the level and is 60-69% predictable from the content key. — I concede every part. I pre-registered a check on 806 images that no Round-2 panelist had scored (positions 1059-1864; panelist-D/r3/PREREGISTRATION_R3.md).
- The log share beats my raw crest by +0.018 [+0.016, +0.019] on common and +0.006 [+0.005, +0.008] on extra.
- My frozen row's fog family mean is 0.749. That is the third miss of my 0.75 bar: 0.739 on the screen, 0.746 on 773 held-out images, 0.749 here.

On the mechanism:
- My number described the raw peak. Its clean SD of 0.20, against 0.32 for the log level at s1, is true.
- But the ratio inherits the level's content variation through its denominator, so my sentence said nothing useful about the crest. What works is on the signal side.

A new screen check (drift_mechanism.py) makes the signal side concrete, and it also corrects part of A's own story. At severity-1 fog, s1:
- The image's channel means move -0.27 bank-SD, while its log share moves -0.84.
- With the clean twin's neighbours there is no key drift, yet the level still reaches only 0.545 (0.490 with its own neighbours).
- The drifted neighbours' level moves -0.06, about a fifth of the image's shift at s1 and half at s2-s3. Their share moves -0.06, so the neighbours come out flatter, not peakier.
- Explaining away accounts for zoom blur (level 0.726 -> 0.906 with oracle neighbours), not for fog.

On what would change A's mind (the level helping on real fog): my 24-image depth-fog probe says it may (level 0.781 vs share 0.760 at beta = 0.005). The right way to keep the level is A's own M1, a max that does not dilute: severity-1 fog 0.960 on the 806 never-read images. My sum is not that way.
- **To B** (concede): The geometry is wrong: multiplicative quantities are scored with additive spreads. C's log gain + log shape dominates my row (+0.014 common, a tie on extra), and my row misses its fog bar (0.743 on held-out images). B would change its mind only for a deployment reason to prefer raw ratios, such as night frames with near-zero means. — I concede. I have no deployment reason for raw ratios, and the case of near-zero means argues the other way:
- For near-dead channels at night, top / max(mean, 1e-6) is unbounded.
- log(t + eps) - log(m + eps) tends to 0 as both vanish.

On the 806 never-read images:
- C's GS reaches 0.895 / 0.871, against my 0.879 / 0.871. The point differences are +0.016 / +0.000; I did not bootstrap this pair.
- The A/C row reaches 0.901 / 0.863.

The merged plan freezes eps at A's 1e-6. It also reports 1e-4 on ACDC night, because B named the log offset as a risk on near-zero energies.
- **To C** (partially concede): D is the A/C idea in a weaker parameterisation (the log share beats it by +0.017 common), and its mechanism is half true: content explains about as much of the ceiling's variance as of the level's. The shared risk is that real fog spares near-field edges. The crest would then move by less than its clean spread on Foggy Cityscapes at beta = 0.005, and D's IV kill would fire. — I concede the parameterisation and the mechanism. My claim concerned the raw peak's spread, which is smaller; C shows that the share of variance content explains is similar.

I partially reject the strong form of the fog scenario, that real fog makes the crest rise:
- In my depth-fog probe (24 Cityscapes val images, label-based Koschmieder fog, beta = 0.005-0.02), the s1 share still falls in 70-73% of image-channel pairs, against 81-83% for the package's fog.
- The median fall is 0.50 to 1.20 reference SDs.
- E's ramp fog shows no sign flip.
- The weak stage is s3: 50% of pairs fall, median shift 0.00, and A's probe even finds +0.17 / +0.38 SD.

So the risk is to the size of the shift, not its sign, but size alone can fire the kill. In that probe the plain level matched the share: 0.781 vs 0.760 at beta = 0.005, and 0.903 vs 0.877 at beta = 0.01.

I keep the kill exactly as C states it. I add M1 as a pre-registered IV row, because its level arm is the hedge. With 500 pairs, a single fog density gives a paired interval of about +/-0.02 (iv_power.py), so a true advantage under about 0.03 would count as a failure. That is why the plan adds the 475 spare train images as extra pairs.
- **To E** (concede): My row is dominated by its own log twin and misses my own bar, fog >= 0.75. E would change its mind only if, on never-read images, the linear crest matched the log version within about 0.003 on both groups and cleared fog >= 0.75. — I concede. I ran E's exact test on the 806 never-read images (positions 1059-1864):
- The log share minus the raw crest is +0.018 [+0.016, +0.019] on common and +0.006 [+0.005, +0.008] on extra. Both exceed 0.003, so E's objection stands.
- The crest alone does clear fog (0.844), but my summed row does not (0.749).

E's ramp-fog result agrees with my depth probe: the direction is kept at s1-s2.

One point goes back to E's own thesis, from the same images. A's M1, a single-image row, reaches severity-1 fog 0.960 and contrast 0.948. That is at the level of E's same-scene oracle (0.931 / 0.923 on my 773 held-out images). What stays aliased is frost, elastic, saturate, brightness and spatter, so E's decomposition survives as a diagnostic figure for those families.

**Status:** merged (merged with: A, C)

**Final proposal:**

# Merged proposal (A + C, with D's driving-data section): Content sets the level, corruption flattens the peaks

## Thesis
Corruptions that remove image structure barely move a frozen detector's early channel means once the scene is accounted for, but they flatten the peaks of most early channels at once. So each early channel's log peak share, judged against the clean training scenes that the detector's own deep stage finds most similar, detects corruption better than activation levels and far better than full activation distributions, most of all at mild severity.

## Insight: what survived three rounds
1. **The level barely moves; the peaks fall.**
   - At severity-1 fog, s1, an image's channel means move -0.27 bank-SD; its log peak share moves -0.84 (D, Round 3, screen).
   - In clean conditioned-residual units, the log ceiling moves -1.89 / -1.92 / -2.04 / -1.96 under fog / contrast / zoom / defocus. The log level moves -0.45 / -0.24 / -0.20 / -0.12 (C, Round 2).
   - A's causal reading, not yet tested: early responses sit on fixed offsets (mean intensity, frozen BatchNorm shifts), so losing structure lowers peaks before it lowers means.
2. **The flattening is coherent: the corruption has a direction.**
   - 67-74% of the shape's shift under fog, contrast, zoom and defocus lies on one clean direction (A).
   - That direction is close to all-ones: cos^2 0.52-0.59 at s1 for fog, contrast and zoom (B).
3. **There are two kinds of corruption.**
   - Structure-removing families (fog, contrast, the blurs, pixelate, JPEG, frost) lower the share.
   - Structure-adding families (noise, snow, spatter) move the level, and can raise the share at s2-s3.
   - So the level is a second arm, never summed into the first: summing dilutes fog (D's frozen row reaches only 0.749 on fog on never-read images).
4. **Content conditioning is still needed.**
   - The share is 62-67% predictable from the stage-4 key (B).
   - The content reference adds +0.048 on the screen (0.849 -> 0.897).

**Corrected, and no longer claimed:**
- "The ratio cancels content" (C). Its clean spread is 0.70-0.89 of the level's.
- "Clean scenes trade level for peakiness" (A). That correlation comes from the ratio's own denominator (B, C), so A's K3 loses its correlation clause.
- "Key drift recruits peakier neighbours" (A). Under severity-1 fog, the recruited neighbours' s1 share moves -0.06 bank-SD, so they are flatter. The level with the clean twin's neighbours still reaches only 0.545 at severity-1 fog. Drift explains zoom blur (level 0.726 -> 0.906 with oracle neighbours), not fog.
- "Raw crest + raw level" (D). It is dominated.
- "A single image cannot tell a hazy scene from a fogged one" (E). Shown false for fog and contrast below.

## Method
**Primary row AC**, frozen in Round 1 by A and C (identical to three decimals). Training-free, clean-only, one forward pass.
1. At the post-ReLU inputs of `res_layers[s].blocks[1].branch2a.conv`, for s1-s3, compute each channel's pi = log(top-1% mean + 1e-6) - log(mean + 1e-6).
2. Find the 50 of 2,000 clean COCO-train bank images nearest in bank-standardised stage-4 channel means (Euclidean).
3. Score each stage by the mean over channels of |pi - the neighbours' mean pi| / the bank SD of pi. Z-score each stage with 500 other clean images, scored the same way, and sum s1-s3.

**Lead secondary M1** (A's Round-2 addendum, frozen before any position above 1058 was scored). M1 is the maximum of two arms, each re-z-scored on the clean z-statistics images:
- the signed flattening arm: per stage, the mean over channels of -(pi - pi_nb) / sigma, z-scored, then summed over stages;
- the level arm: Section 4.

M1 replaces AC if and only if, on all positions above 1058 of the completed pass:
- M1 - AC on common is above 0, with an interval excluding 0, and
- the point estimate of M1 - AC on extra is at least 0 (A's rule).

**Other pre-registered secondaries:**
- C's log gain + log share (GS);
- level + share (LS).

## Evidence
Paired bootstrap, 1,000 draws. Every row was frozen before each look. AUROC is given as common / extra.

| Row | Screen (200) | Held out, positions 200-972 (773) | Never read in Round 2, positions 1059-1864 (806) | Severity-1 common (806) | FPR95 common (806) |
|---|---|---|---|---|---|
| AC (primary) | 0.897 / 0.859 | 0.891 / 0.856 | **0.901 / 0.863** | 0.818 | 0.249 |
| M1 (lead secondary) | 0.922 / 0.862 | not scored (chosen after) | **0.922 / 0.862** | 0.870 | 0.215 |
| GS: log gain + log share | 0.891 / 0.869 | 0.886 / 0.865 | 0.895 / 0.871 | 0.809 | 0.262 |
| B: log-whitened means | 0.876 / 0.856 | 0.873 / 0.852 | 0.885 / 0.862 | 0.791 | 0.274 |
| Section 4 (level) | 0.841 / 0.870 | 0.840 / 0.864 | 0.842 / 0.868 | 0.763 | 0.356 |
| Activation CDFs | 0.825 / 0.808 | 0.815 / 0.802 | 0.820 / 0.810 | 0.707 | 0.416 |

**On the 806 never-read images (paired differences):**
- AC - CDFs: +0.081 [+0.073, +0.089] common, +0.053 [+0.047, +0.059] extra, +0.112 at severity 1.
- AC - Section 4: +0.059 [+0.052, +0.066] common, -0.006 [-0.011, -0.000] extra.
- M1 - AC: +0.021 [+0.015, +0.028] common, -0.001 [-0.007, +0.006] extra, +0.052 [+0.043, +0.061] at severity 1.
- By A's rule, AC stays the primary on this look (extra point estimate -0.001 < 0). The completed pass decides.

**Single image vs the same-scene oracle:**
- At severity-1 fog / contrast: M1 0.960 / 0.948; AC 0.807 / 0.830; Section 4 0.515 / 0.531; CDFs 0.637 / 0.665.
- E's same-scene oracle reached 0.931 / 0.923 on the 773 held-out images, so a single image suffices for mild fog and contrast.
- Still aliased, M1 vs the oracle (E, 712 images): frost 0.723 vs 0.962; elastic 0.689 vs 0.841; saturate 0.602 vs 0.803; brightness 0.526 vs 0.650; spatter 0.569 vs 0.646.

**Still capped by L10** (family means, M1 / AC): brightness 0.606 / 0.656, saturate 0.674 / 0.713.

## Novelty
- **Becker et al., ICPR 2026.** Per-channel EMD of absolute-unit CDFs against one global training CDF. The level dominates, and there is no content reference. AC beats it by +0.081 / +0.053.
- **NMD.** The first moment against BatchNorm means, which is the axis that barely moves.
- **Neural Activation Prior** (Wan et al., arXiv 2402.18162; verified by C and D). Per-channel max/mean at the penultimate layer, one-sided, with no reference, for semantic OOD. Transplanted to s1-s3 with raw ratios, it scores 0.302 / 0.343, below chance. The log, the early stages and the content reference are what make the statistic work.
- **ASH / SCALE** (SCALE verified). The same ratio algebra, but across the channels of the pooled penultimate vector, to rescale logits.
- **Mahalanobis and Rippel et al. 2020** (verified by B and E). Whitening the share costs 0.040-0.049, because the corruption's direction is the dominant clean direction.
- **SPADE** (verified by E). Retrieval is not new; the statistic and its direction are.
- **BRISQUE / NIQE** (recollection, unverified). Shape statistics of normalised pixels.
- **Topology (C, a negative result).** The Euler characteristic adds +0.004 [-0.016, +0.022] over the means. C's ladder runs from TU's top of the diagram (0.661) through per-channel top-1% (0.792) and the share (0.849) to the conditioned share (0.897).

## Decisive experiment (COCO; no new GPU)
Score the frozen rows on the completed 5,000-image pass, which stores means and top-1% means for every variant. Report positions above 1058, and separately the never-read remainder above 1864.
- **Kills:**
  - AC - CDFs has an interval including 0 on common or on extra (A's K1, C).
  - AC - Section 4 on common is below +0.015, or its interval touches 0 (A's K2, C).
- **M1 vs AC:** decided by A's rule.
- **Paper bars:** common >= 0.87; extra >= 0.83; severity-1 common >= 0.77; FPR95 common <= 0.30; severity-1 fog, contrast and zoom each >= the CDFs + 0.08. On the 806 images, AC clears all of them (+0.170 / +0.165 / +0.121 on the last).

## Figure 1
- **(a) Corruption compass at s1.** Per family, the conditioned (Delta log level, Delta log share) shift, drawn as an arrow from severity 1 to 5.
- **(b)** C's ladder.
- **(c)** E's dumbbells at severity 1: the CDFs, a single image (AC and M1), and the same-scene oracle.

## Fit for IEEE IV (D's section, pre-registered now)
1. **Data.**
   - Cityscapes-C: 500 val images x 19 families x 5 severities, same seeds.
   - Official Foggy Cityscapes, beta = 0.005 / 0.01 / 0.02, paired with the same images. Downloading it needs the user's Cityscapes account. The 475 train images outside the bank and z-statistics sets add test pairs.
   - ACDC fog / night / rain / snow against its normal-condition images.
   - nuImages night and rain, if the metadata allows.
   - Bank: 2,000 clean Cityscapes train images; z-statistics: 500 others.
2. **Rows.** AC, M1, Section 4, B's whitened level, CDFs with a Cityscapes reference, the global level, and global M1. Global M1 is the fixed-camera ablation; with no key it reaches 0.903 / 0.827 on the COCO screen.
3. **Probe predictions.**
   - Depth fog keeps the direction but loses size: the s1 share falls in 70-73% of image-channel pairs (package fog: 81-83%), and s3 does not respond.
   - The plain level is competitive: 0.781 vs 0.760 at beta = 0.005, and 0.903 vs 0.877 at beta = 0.01.
4. **IV kill.**
   - If, at both beta = 0.005 and beta = 0.01, the paired interval of the primary minus the better level baseline (Section 4 or B) includes 0 or lies below it, the shape claim is restricted to diverse content.
   - Separately, the primary must beat the CDFs on Cityscapes-C, on common and on extra.
   - With 500 pairs, a single condition's interval is about +/-0.02.
5. **Two-sidedness.**
   - M1's flattening arm is one-sided. On COCO, the level arm covers the cases where the share rises (severity-1 spatter: M1 0.569, AC 0.516).
   - On ACDC night, report M1 beside the two-sided AC, with eps = 1e-6 and 1e-4.
6. **L2.** An upper-band and a lower-band share as a secondary.
7. **Latency.** One top-k per channel at three stages, plus a 2,000 x 512 neighbour search, inside the 0.2-1.2 ms monitor budget. It will be timed in the run.
8. **Video.** A per-frame exponential moving average. For frost, spatter, brightness and saturate, average the residual vectors over frames.

## Risks and cost
- **M1 selection risk.** M1 was chosen after the Round-2 looks, so only positions above 1058 count.
- **L10 cap.** Brightness and saturate stay weak.
- **s3 noise.** The s3 top 1% covers only 16 positions.
- **Generality.** The mechanism leans on BatchNorm and ReLU, so a second backbone is needed.
- **Real fog.** It shrinks the shape's signal.
- **Cost.**
  - COCO: no new GPU.
  - Cityscapes: one backbone pass, about 40-60 min and bound by the CPU, in an agreed GPU window, after the user approves a written plan.
  - Code: about 40 new lines.

**Vote:** first A, second C. A's proposal now holds both the pre-registered primary and the best row on images no Round-2 panelist had read.
- The primary, AC, passed every kill on those 806 images: 0.901 / 0.863. It beats the CDFs by +0.081 / +0.053 and Section 4 by +0.059 on common.
- M1 reaches 0.922 / 0.862, with severity-1 common at 0.870. From a single image it gets severity-1 fog to 0.960, at the same-scene oracle's level.
- A also has the coherence finding, which explains why the signed arm works.
- A's thesis must change from 'shape, not level' to 'level and flattening' if M1 wins.

C has the identical primary and the best narrative: the ladder from TU to the conditioned share, and an honest topology negative. But its mechanism ('the ratio cancels content') is refuted.

Not D: my statistic is dominated (+0.018 / +0.006 for the log share). B and E are strong baselines and diagnostics, not the paper.

**First experiment:** Run the Cityscapes IV test as soon as the running 5,000-image pass frees the GPU, and after the user approves a written plan in an agreed GPU window. Scoring that pass's frozen rows is already fixed by A's rule.

**Setup:**
- One backbone pass with the frozen COCO recipes: k = 50, the stage-4 key, s1-s3, the top 1%.
- Bank: 2,000 clean Cityscapes train images; z-statistics: 500 other train images.
- Test: the 500 val images clean, as Cityscapes-C (19 families x 5 severities), and as official Foggy Cityscapes at beta = 0.005 / 0.01 / 0.02. The 475 spare train images add foggy pairs. Start the download now.
- Rows: AC, M1, Section 4, B's whitened level, CDFs with a Cityscapes reference, the global level, and global M1.

**Kill criteria:**
1. If, at both beta = 0.005 and beta = 0.01, the paired 95% interval of the primary (AC, or M1 if it wins at 5,000) minus the better level baseline (Section 4 or B) includes 0 or lies below it, the shape claim is restricted to diverse content, and the IV paper says so.
2. If the primary minus the CDFs on Cityscapes-C has an interval including 0 on common or on extra, the COCO headline does not transfer to driving data.

On 500 pairs, a single condition's interval is about +/-0.02.

**Checks run:** All checks ran on CPU (niced). The repository was not modified and the GPU was not used. Scripts and logs are in panelist-D/r3/.

**(1) PREREGISTRATION_R3.md, then fresh_rows.py (81 s).**
- **Data:** frozen rows on the 806 images at positions 1059-1864, which no Round-2 panelist had scored.
- **Smoke test** on the 200 screen images first. Every known number reproduced exactly:
  - AC 0.897 / 0.859, M1 0.922 / 0.862, E4 0.916 / 0.831;
  - LS 0.890 / 0.874, GS 0.891 / 0.869, S4 0.841 / 0.870;
  - B 0.876 / 0.856, D 0.876 / 0.870, Dc 0.876 / 0.849, CDF 0.825 / 0.808.
- **Fresh results:**
  - AC 0.901 / 0.863; severity-1 common 0.818; FPR95 0.249.
  - M1 0.922 / 0.862; severity-1 0.870; FPR95 0.215.
  - E4 0.923 / 0.838; LS 0.891 / 0.874; GS 0.895 / 0.871.
  - S4 0.842 / 0.868; B 0.885 / 0.862.
  - D 0.879 / 0.871 (fog 0.749); Dc 0.884 / 0.856 (fog 0.844).
  - CDF 0.820 / 0.810.
- **Paired differences:**
  - M1 - AC: +0.021 [+0.015, +0.028] / -0.001 [-0.007, +0.006]; severity 1 +0.052.
  - AC - S4: +0.059 [+0.052, +0.066] / -0.006 [-0.011, -0.000].
  - AC - CDF: +0.081 [+0.073, +0.089] / +0.053 [+0.047, +0.059].
  - AC - B: +0.016 [+0.012, +0.020].
  - AC - Dc: +0.018 [+0.016, +0.019] / +0.006 [+0.005, +0.008].
  - D - S4: +0.037 [+0.032, +0.041].
- **Severity-1 fog / contrast:** M1 0.960 / 0.948; AC 0.807 / 0.830; S4 0.515 / 0.531; CDF 0.637 / 0.665.
- **Decisions:**
  - A's M1 rule fails on the extra point estimate, so AC stays.
  - A's and C's kills pass.
  - E's objection to D stands.

**(2) drift_mechanism.py (screen, 15 s).** Each row with its own neighbours / the clean twin's (oracle) / no key.
- Severity-1 fog:
  - level 0.490 / 0.545 / 0.470;
  - share 0.800 / 0.831 / 0.709;
  - flattening arm 0.969 / 0.974 / 0.935;
  - M1 0.949 / 0.957 / 0.912.
- Zoom, level: 0.726 / 0.906.
- Global M1 headline: 0.903 / 0.827.
- Severity-1 fog shifts at s1 (bank-SD units): image level -0.27, neighbours' level -0.06; image share -0.84, neighbours' share -0.06.

**(3) iv_power.py (14 s).** Half-widths of single-condition paired 95% intervals:
- 0.016-0.019 on 806 images;
- 0.020-0.024 on 500 images.

## E (Skeptic): Aliased, not blind: a frozen detector sees mild corruptions, the scene hides them

**Thesis:** A frozen detector is not blind to mild corruptions: judged against the same scene's clean look, its early-layer channel means separate severity-1 fog, contrast and zoom blur from clean at 0.94–0.99 AUROC, so single-image monitors fail because natural scene variation hides the corruption, not because the detector ignores it.

### Round 1 proposal

## Thesis
A frozen detector is not blind to mild corruptions: judged against the same scene's clean look, its early-layer channel means separate severity-1 fog, contrast and zoom blur from clean at 0.94–0.99 AUROC, so single-image monitors fail because natural scene variation hides the corruption, not because the detector ignores it.

## Insight
All numbers are a 200-image CPU screen on the stored channel means (brackets: 95% paired bootstrap). Many rows were tried (L11), so only the pre-registered held-out test below counts.

- **The same-scene oracle.** Predict stages 1–3 from the *clean* version's stage-4 means (the bank's ridge map, λ = 10), then score the variant's residual.
  - It reaches 0.982 / 0.958; 0.952 at severity 1; FPR95 0.058. The best single-image reference reaches 0.855 / 0.860.
  - It is not an artifact. Another image's clean key gives 0.707, below the global average (0.811), and λ = 1 to 1000 all give 0.979–0.983.
- **The weak spots are aliasing, not blindness.** At severity 1, oracle vs best single-image row:
  - fog 0.95 vs 0.56; contrast 0.94 vs 0.58; saturate 0.85 vs 0.49;
  - zoom blur 0.99 vs 0.69; frost 0.98 vs 0.71; elastic 0.87 vs 0.63.
  - Only mild brightness and spatter stay partly blind (0.67 and 0.66).
  - This contradicts the invariance reading of L10 and quantifies L9. L4/L5 hold: one mean per channel carries 0.98.
- **The back cannot anchor itself.** This goes against Section 4's premise and the usual reading of L1.
  - Stage-4 typicality stays at chance: AUROC 0.476 at severity 1 and 0.530 at severity 5.
  - Yet its 50 nearest clean scenes turn over: 34% are replaced at severity 1, 72% at severity 5, and 90% for severity-5 Gaussian noise.
  - So corruption slides the deep representation *along* the clean manifold: the detector re-describes the image as another plausible scene.
  - Hence a stage-3 key works as well as stage 4 (Δ −0.000 [−0.009, +0.007]), and a ridge on the image's own key leaks (0.789).
  - A scale-free "composition" key, which I pre-registered as the fix, failed (−0.008 [−0.011, −0.004]).
- **The gap is the scene's own look, not only the leak.**
  - A leak-free key from the most similar *other* clean scene gives 0.868 (0.781 at severity 1). It recovers 0.01 of fog's 0.39 gap and none of contrast's 0.36.
  - Blending the key back toward the same scene recovers it smoothly: 0.919, 0.958 and 0.977 at 25%, 50% and 75% same scene.
  - So a single image cannot tell a hazy scene from a fogged one, but a nearby clean view of the same scene can.
- **Why it is not obvious.** The field's axis of progress has been richer statistics (CDFs, tails, topology: L4–L6), and the weak spots were blamed on the detector's photometric invariance.
- **What kills it.** Either of these on held-out images:
  - the oracle falls below 0.8 on mild fog and contrast, so the detector is blind after all;
  - a leak-free retrieval recovers most of the gap, so it is a key problem and an improved Section 4 wins.

## Method
Training-free, clean-only, label-free; one score per image.
1. **Features.** v_s ∈ R^{C_s} is the mean |activation| per channel at the inputs of `res_layers[s].blocks[1].branch2a.conv`, s = 1, 2, 3. Standardise with the 2,000-image clean bank: z_s = (v_s − μ_s)/σ_s. The key u is the standardised stage-4 means.
2. **Scene reference m_s, from the best available source:**
   - **(a) single image.** m_s = the mean z_s of the k = 50 bank images nearest in u; r_s = z_s − m_s. Score d_s = r_sᵀ Σ_s⁻¹ r_s, where Σ_s is the Ledoit-Wolf covariance of leave-one-out bank residuals.
   - **(b) same scene (driving).** m_s = W_s u(x′), where x′ is a clean view of the same scene (a recent frame of the same camera, or a same-place normal-condition image) and W_s is the ridge map fitted on the bank. Score d_s = mean_c |r_{s,c}|/τ_{s,c}, where τ is the bank's ridge-residual spread.
   - **(c) oracle (analysis only).** As (b), with x′ = the clean version of x.
3. **Combine.** Z-score each d_s with the 500 clean z-statistics images, scored the same way, and sum over s1–s3.
4. **Decompose,** per family and severity:
   - aliasing = AUROC(c) − AUROC(a);
   - blindness = 1 − AUROC(c).

In three sentences: read one mean per channel at stages 1–3. Predict them from the best available clean view of this scene, and whiten the residual. An oracle view splits every family's failure into aliasing and blindness.

## Novelty
- **Becker et al., ICPR 2026.** Per-channel CDFs against one global training CDF, with no cross-channel structure and no scene. On the screen, (a) beats it by +0.030 [+0.008, +0.052] common, +0.052 [+0.034, +0.069] extra and +0.063 [+0.039, +0.088] at severity 1. The decomposition also explains its weak spots.
- **NMD.** This is the global diagonal reference (0.811 here).
- **Mahalanobis / Gram.**
  - (a) without retrieval *is* Rippel et al. (ICPR 2020; verified): class-free Gaussians on pooled multi-level features, Ledoit-Wolf, summed distances. MahaAD (Doorenbos et al., ECCV 2022) also uses Mahalanobis as a strong baseline. I claim nothing for this part.
  - Lee et al. add class conditioning and OOD-tuned layer weights. Gram matrices need full feature maps.
- **Section 4 is not new as a method.**
  - Retrieving the K nearest normal images by pooled deep features, then comparing finer features only with them, is SPADE (Cohen & Hoshen, arXiv 2020; verified).
  - Judging behaviour given context, by neighbours or by regression, is conditional/contextual anomaly detection: Song et al. (TKDE 2007) and ROCOD (CIKM 2016), which couples both.
  - On the screen, retrieval adds only +0.008 [+0.005, +0.010] over whitening.
- **kNN-OOD.** Its L2 normalisation discards magnitude, which carries the signal (0.587 on s1–s3).
- **DisCoPatch.** Trained, and weaker.
- **Reduced-reference IQA** (Wang & Simoncelli, SPIE 2005; verified). It compares a few statistics with the reference image's. Ours borrows the detector's own 512-number summary of another view, and scores detection AUROC.
- **New:** the aliasing/blindness decomposition, the slide finding, and the same-scene semantic reference. My searches did not find them, which is not proof.

## Decisive experiment
It runs on the 4,800 held-out images and needs no GPU: the running confirmation stores means for s1–s4 for all 5,000 × 96 variants. The rows are fixed now, before any held-out number is read: (a), whitening alone, (c), the leak-free other-scene key, Section 4, and the stored CDFs.

**What makes the paper (all three):**
- (a) − CDFs > 0, with intervals excluding 0 on common, extra and severity 1 (CDFs: 0.821 / 0.807), and FPR95 below 0.413;
- (c) − (a) ≥ 0.20 at severity 1 on fog, contrast, saturate, zoom blur and frost (screen: 0.27–0.39);
- the other-scene key recovers under a third of that gap on fog and contrast.

**Kill criteria:**
- (c) < 0.80 at severity 1 on fog and contrast: the detector is blind and the thesis is dead;
- the other-scene key recovers ≥ two thirds of the gap: the problem is the key's leak, so pursue a leak-free retrieval instead;
- the (a) − CDFs interval includes 0: no claim of a single-image method.

## Cheapest first test
Already run: six CPU scripts, about 5 minutes in total, giving all the numbers above. It is a screen, since rows were tried on the same 200 images, and the composition-key hypothesis I pre-registered failed. The held-out test costs about 10 CPU minutes once the confirmation finishes.

## Figure 1
"The detector sees it; the scene hides it."
- One dumbbell per family at severity 1, sorted by aliasing.
- Three dots per family: activation CDFs, the single-image reference (a), and the same-scene oracle (c).
- The bar from (a) to (c) is aliasing; the gap from (c) to 1 is blindness.
- Inset: AUROC against how far the reference view is from the same scene: 0.982 → 0.977 → 0.958 → 0.919 → 0.868.

## Fit for IEEE IV
- **Runtime.** Channel means, three small matrix products and one 2,000 × 512 neighbour search. That fits within the 0.2–1.2 ms of the existing monitors, beside a 5.9 ms detector (L12).
- **Engineering message:**
  - one frame is enough for noise, snow, JPEG and motion blur (≥ 0.84 at severity 1);
  - haze, contrast, colour, zoom blur and elastic need a same-scene reference (single image 0.49–0.69, oracle 0.85–0.99). These are the weather and lighting cases.
- **Driving data:**
  1. Cityscapes-C, single image plus the oracle. Prediction: a fixed camera means less content variation, so a smaller aliasing gap. Also test L2's position caveat.
  2. Synthetic onsets in driving video (Cityscapes demoVideo or KITTI raw; to download, sizes unchecked). Negatives are clean frames scored against their predecessor, which tests (b) symmetrically.
  3. ACDC (verified: every fog, night, rain or snow image comes with a same-scene normal-condition image), for real conditions.
  4. Foggy Cityscapes.

## Risks and cost
- **GPU.** None for the decisive test. Cityscapes-C needs about 1–1.5 h and the video onsets about 1–2 h, each with a written plan and an agreed window.
- **Code.** About 150 lines, plus tests.
- **What could go wrong:**
  - the margin over the CDFs on the common families is thin (lower bound +0.008);
  - the CDFs beat every means reference at severity-1 fog and contrast (0.65 / 0.68 vs ≤ 0.58);
  - reviewers may call the oracle trivial; the answer is the placebo, the blend curve, and that it reads only 512 semantic numbers of the clean view;
  - (b) leaves the single-image rule, and it misses conditions present from the first frame;
  - ACDC has no second clean capture per place, so its negatives need care;
  - clean COCO contains naturally hazy or blurred images, so some false positives may be real degradations.

**Quick check:** Six CPU scripts ran on pilot200.npz: 200 images × 96 conditions, the stored channel means s1–s4, the 2,000-image bank and the 500 z-statistics images. This is a screen, and many rows were tried. I used no GPU and read none of the running confirmation's partial output. Scripts and logs are in /tmp/claude-1000/-home-yuchen-YuchenZ-UE-philip-sa/3d52b916-b3ff-4806-a501-7f70cde31a29/scratchpad/detection/roundtable/panelist-E/. Brackets are 95% paired bootstraps (1,000 draws, seed 44). Numbers are AUROC common / extra unless stated.

**(1) key_necessity.py**
- **Reproduction:** own average s1–s3 gives 0.811 / 0.837, and Section 4 gives 0.841 / 0.870, both exactly as in the briefing.
- **Controls with no key:**
  - per-stage Ledoit-Wolf Mahalanobis: 0.847 / 0.857, FPR95 0.332. Section 4 minus it: −0.006 [−0.015, +0.004] common, +0.013 [+0.005, +0.021] extra;
  - PCA residual, r = 4 / 8 / 16 / 32: 0.832 / 0.830 / 0.864 / 0.845 common (not used);
  - standardised kNN: 0.819 / 0.863;
  - kNN with L2 normalisation: 0.587 / 0.598.
- **Content-anchor tests:**
  - clean-key oracle with kNN: 0.898 / 0.895, which is +0.056 [+0.052, +0.061] over the actual key;
  - random neighbours: 0.801;
  - mismatched key: 0.754;
  - ridge with its own key: 0.789 / 0.811, against the ridge oracle at 0.982 / 0.958.
- **Neighbour stability:** the mean Jaccard of the 50-neighbour sets, clean vs variant, is 0.303 (0.05 at severity-5 noise).
- **Key sweep, scoring s1–s2:** the s3 key gives 0.825 / 0.869 and the s4 key 0.825 / 0.873; s4 minus s3 is −0.000 [−0.009, +0.007].
- **Conditioning other statistics:** top-1% gains +0.034 / +0.068 and p99 gains +0.035 / +0.071.

**(2) composition_key.py**
- The pre-registered hypothesis H1 failed: the s4 composition key gives 0.834 / 0.869 with Jaccard 0.311, a difference of −0.008 [−0.011, −0.004].
- Whitened residual after retrieval: 0.855 / 0.860, which is +0.008 [+0.005, +0.010] / +0.003 [+0.001, +0.005] over whitening alone.
- Gaussian conditional s1–s3 given s4: 0.833 / 0.840. Joint Mahalanobis over s1–s4: 0.750 / 0.782.

**(3) cross_depth.py**
- Cross-depth retrieval disagreement is weak: the best pair (s3, s4 sets) gives 0.689 / 0.655. Negative result.
- The stage-4 slide, severities 1 to 5: typicality AUROC 0.476 / 0.474 / 0.491 / 0.514 / 0.530, while the neighbour Jaccard is 0.494 / 0.364 / 0.283 / 0.210 / 0.164.

**(4) oracle_decomposition.py**
- Severity 1, oracle vs whitened: fog 0.949 vs 0.520; contrast 0.941 vs 0.547; saturate 0.848 vs 0.479; zoom blur 0.993 vs 0.678; frost 0.978 vs 0.689; elastic 0.869 vs 0.623; brightness 0.675 vs 0.539.
- Oracle, mean over the common families by severity 1–5: 0.952 / 0.977 / 0.990 / 0.994 / 0.997.

**(5) oracle_checks.py**
- Placebo (another image's clean key): 0.707 / 0.741, 0.624 at severity 1.
- Ridge penalty: the oracle stays at 0.979–0.983 for λ = 1 to 1000.
- Blend toward the nearest other clean scene, a = 0.25 / 0.5 / 0.75 / 1: 0.977 / 0.958 / 0.919 / 0.868.
- Severity 1, oracle minus whitened: fog +0.429 [+0.396, +0.461], contrast +0.394 [+0.362, +0.424].
- Severity 1, other-scene key minus whitened: fog +0.052 [+0.018, +0.088], contrast +0.027 [−0.007, +0.063], brightness +0.003 [−0.009, +0.015], zoom blur +0.167 [+0.127, +0.206].

**(6) final_rows.py**
- Primary row (a): 0.855 / 0.860, AUPR common 0.808, FPR95 common 0.315 (CDFs: 0.403). By severity 1–5: 0.779 / 0.832 / 0.867 / 0.890 / 0.907.
- (a) minus CDFs: +0.030 [+0.008, +0.052] common, +0.052 [+0.034, +0.069] extra, +0.063 [+0.039, +0.088] at severity 1.
- Whitening alone minus CDFs: +0.022 [+0.001, +0.044] / +0.049 [+0.031, +0.066].
- (a) minus Section 4: +0.013 [+0.004, +0.023] / −0.010 [−0.017, −0.003].
- The per-family, per-severity table for Figure 1 is in figure1_table.csv.

**Novelty sources I verified:**
- SPADE and Rippel et al.: PDFs read.
- Doorenbos et al. (ECCV 2022), Song et al. (TKDE 2007), ROCOD (CIKM 2016), ACDC (ICCV 2021), Wang & Simoncelli (SPIE 2005): abstracts and listings.
- Guglielmo & Masana (CVWW 2025): read; not a threat.

**Runners-up:**
- Whitened channel means (the baseline every proposal must beat): Per-stage Ledoit-Wolf Mahalanobis on the s1–s3 channel means has no free parameters and screens at 0.847 / 0.857, +0.022 [+0.001, +0.044] / +0.049 [+0.031, +0.066] over the CDFs and level with Section 4 on the common families (−0.006 [−0.015, +0.004]); it is Rippel et al. (ICPR 2020) moved to detector stages, so it is not a paper, but every proposal must beat it.
- Scene-referenced activation CDFs: Replace Becker's single global training CDF with the mean CDF of the 50 retrieved clean scenes, because the CDF statistic beats every means-based reference at severity-1 fog and contrast (0.65 / 0.68 against at most 0.58); this needs per-channel histograms for the bank, one GPU pass.

### Round 2: critiques written by E

**On A** (serious)

- Steelman: A's content-conditioned log peak share is the strongest single training-free monitor on the table, and held-out data now backs it. On 712 unseen images (positions 200-911 of the running confirmation) it clears every bar A pre-registered: 0.890/0.855, FPR95 0.256, severity-1 common 0.807, and fog, contrast and zoom blur at severity 1 each +0.10 to +0.16 above the CDFs. It also has the best mechanism account: the clean trade-off between level and peakiness, and the level being explained away by the leaky key.
- Objection: The thesis sentence, 'read the peak-to-mean ratio ... not its mean', is falsified family by family on held-out data. The average gain over the level comes almost entirely from the three families that motivated the statistic, while the plain level is significantly better on 9 of the 19 families. This is the 'better row for fog, contrast and zoom' a reviewer will name. The defensible claim is level AND shape, and A's own disclosed fallback (level + log shape) is the row to pre-register.
- Failure scenario: On the 4,800-image table a reviewer computes A minus Section 4 per family.
- Fog (+0.293), contrast (+0.243) and zoom blur (+0.188) supply 0.048 of the +0.050 common gain.
- Against that sit losses whose intervals exclude 0: elastic -0.058, spatter -0.041, brightness -0.031, speckle -0.026, saturate -0.022, snow -0.019 and all three noises. On extra the difference is -0.010 [-0.015, -0.003].
- On a driving set dominated by sensor noise or spatter-like soiling, the shape-only monitor would then be worse than the plain level it claims to replace.
- A smaller problem: aggregated NAP-style (mean over channels of (top/mean)^2, one-sided 'flatter = corrupted'), the same ratio scores 0.302/0.343 at s1-s3, below chance, because a minority of sparse channels sharpen. 'Corruption lowers the ceiling' therefore has to be stated for the typical channel; about 80% of s1 channels flatten in my CPU check.
- Evidence: - Held-out check, pre-registered in panelist-E/R2_PREREGISTRATION.md before any held-out top-1% value was read, plus r2_family_deltas.py, both on 712 images.
- The level + log shape row reaches 0.882/0.868: +0.043 [+0.038, +0.047] common and +0.003 [+0.000, +0.007] extra over Section 4, and +0.066/+0.065 over the CDFs. Its worst per-family loss to the level is -0.022 (elastic), against -0.058 for A.
- Against A, that row is -0.008 [-0.011, -0.004] common and +0.013 [+0.010, +0.016] extra.
- NAP verified on the arXiv 2402.18162 HTML: S = mean_c (max/(mean+eps))^2 at the penultimate layer, one-sided, semantic OOD only.
- The NAP-style rows are from the 200-image screen.
- Would change my mind: On the never-read images (positions >= 912), or on a held-out family set such as Foggy Cityscapes or ACDC fog: shape alone matches or beats level + shape on both groups, and its per-family losses to the level shrink to within about 0.01.

**On B** (serious)

- Steelman: B is the cheapest strong monitor on the table: closed form, knob-free, about 10 µs, and supported on held-out images. Log channel energies whitened by a Ledoit-Wolf clean covariance beat the CDFs by +0.057 [+0.049, +0.065] common and +0.049 [+0.042, +0.055] extra on 712 unseen images, with FPR95 0.290. B also gains where the shape rows are weak: frost 0.807, brightness 0.692, pixelate 0.909.
- Objection: Both the novelty and the thesis fail.
- Without the log, the method is Rippel et al. (ICPR 2020; I read the PDF in Round 1): class-free Gaussians with Ledoit-Wolf shrinkage on pooled features from several levels of a pre-trained network, with the Mahalanobis distances summed. B does not cite it, so the contribution reduces to 'a log before Rippel'.
- The thesis, 'invisible per channel, obvious jointly', is contradicted on held-out data: a per-channel diagonal score of the right statistic beats B, and whitening that statistic makes it worse.
- Failure scenario: A reviewer cites Rippel et al. and reads the paper as one transform on a 2020 baseline (+0.042 over raw whitening on B's 308 images).
- In the same Table 1, the per-channel peak-share row is ahead by +0.017 [+0.012, +0.022] common and +0.027 [+0.021, +0.034] at severity 1 (712 held-out images).
- Whitening that statistic drops it from 0.897 to 0.858 when conditioned, and to 0.848 with no key (screen). Adding B to it changes nothing (0.894/0.862).
- B is also at 0.709 on fog, IV's flagship degradation, against 0.871 for the shape row.
- Evidence: - r2_heldout.py (712 images): B scores 0.873/0.851.
- A - B: +0.017 [+0.012, +0.022] common, +0.004 [-0.000, +0.008] extra, +0.027 [+0.021, +0.034] at severity 1.
- B - Section 4 on extra: -0.013 [-0.019, -0.008].
- Per family, A - B: fog +0.162, zoom +0.084, contrast +0.080; but brightness -0.041 and pixelate -0.036.
- Merges (a) to (c) are from r2_screen_checks.py on the screen.
- Rippel et al. was verified in Round 1.
- Would change my mind: A held-out row where whitening improves the best per-channel statistic, for example whitened [log level, log peak share] beating level + shape on both groups. Or a driving set where B matches the shape rows on fog.

**On C** (fixable)

- Steelman: C reached the peak share by the most honest route.
- Topology was tested against controls and dropped: Euler adds +0.004 [-0.016, +0.022] over the means.
- C has the cleanest ablation ladder (raw top-1% 0.792, then peak share 0.849, then conditioned 0.897) and a check that the signal is not a padding artefact.
- Every kill criterion C pre-registered passes on 712 held-out images: +0.050 [+0.043, +0.057] against Section 4, and +0.052 [+0.046, +0.059] against the CDFs on extra.
- Objection: C is not a second idea. Its frozen row is A's row: identical to three decimals on the screen and on the held-out images (0.890/0.855 with the 1e-4 and the 1e-6 offset). Its thesis, 'content scales, corruption flattens ... far better than magnitudes', inherits A's family problem, and in aggregate it points the wrong way.
- Failure scenario: A reviewer who knows NAP aggregates the same ratio over channels (NAP-style, one-sided) and gets 0.302/0.343 at s1-s3, below chance, so 'corruption flattens' reads as reversed. Section 4's magnitudes beat the peak share outright on nine families on held-out data. The paper then rests on the per-channel, two-sided reading, which C uses but never argues for.
- Evidence: - r2_heldout.py: C's row is identical to A's.
- r2_family_deltas.py: nine families favour the level, with intervals excluding 0.
- r2_screen_checks.py: NAP-style one-sided 0.302/0.343, two-sided aggregate 0.689/0.676.
- Would change my mind: C's one-sided 'flatter than neighbours' secondary holding up on a held-out family set (real fog, ACDC night). That would show the direction itself generalises.

**On D** (fixable)

- Steelman: D states the thesis that the held-out family data supports: level and crest are complementary axes. Noise, spatter and elastic move along the level; fog, contrast and blur move along the crest.
- D's frozen row is the only shape-based one with no loss to Section 4 on extra (+0.000 [-0.003, +0.004]), while it gains +0.031 [+0.027, +0.036] on common.
- D's IV plan is the most realistic: Foggy Cityscapes as the decisive test, a band split, and the warning that real fog may spare near peaks.
- Objection: The row is dominated by its own log twin, and it misses one of D's own bars. The linear crest t/m measures a multiplicative quantity on an additive scale.
- On the same 712 images, level + log peak share scores 0.882/0.868 against D's 0.871/0.865.
- D's fog family mean is 0.745, short of its pre-registered fog >= 0.75.
- Failure scenario: The 4,800-image report shows D's row behind its log version on both groups and short on fog. A reviewer asks why the method is not the log version. D's distinct content then shrinks to the IV plan and the per-cell negative (-0.112).
- Evidence: - r2_heldout.py (712 images): D 0.871/0.865, FPR95 0.292, severity-1 common 0.784; fog 0.745, contrast 0.860. Level + log shape scores 0.882/0.868.
- D - CDFs: +0.055 [+0.046, +0.063] common and +0.062 [+0.056, +0.069] extra.
- On D's risk that real fog spares the peaks, my CPU check with uneven fog found no sign flip. With ramp-shaped fog the s1 log peak share falls by 0.079-0.085, against 0.091 for uniform fog of the same mean strength. 82-84% of s1 channels flatten, against 88%.
- So the risk looks smaller than D feared. Depth-dependent fog is still untested.
- Would change my mind: On the never-read images, the linear crest matching the log version within about 0.003 on both groups and clearing fog >= 0.75.

**Defense of own:** Every criterion I pre-registered holds on the 712 held-out images, but my single-image row loses to A, B and D, so I withdraw it as the method.

What held:
- (a) - CDFs: +0.030 [+0.019, +0.041] common, +0.048 [+0.039, +0.057] extra, +0.065 [+0.053, +0.076] at severity 1. FPR95 is 0.336.
- The same-scene oracle at severity 1 reaches 0.933 on fog and 0.925 on contrast, so the detector is not blind there.
- The oracle beats (a) at severity 1 by 0.369 on fog, 0.348 on contrast, 0.299 on saturate, 0.297 on zoom and 0.272 on frost. Each clears the 0.20 bar.
- The leak-free other-scene key recovers only 8% of the fog gap and 3% of the contrast gap.

What lost:
- A - E is +0.044 [+0.036, +0.052] on common, and B - E is +0.027.

The strongest attack is: "the oracle uses the clean image, so it is full-reference IQA, and the deployable row is dominated."

My answer is that the oracle is a measurement, not a method. It reads only the clean view's 512 stage-4 means; another image's clean key falls to 0.707 (screen). It bounds what any reference can reach on these features. On held-out images that measurement now explains three things the merged paper needs:
- **Why the shape works: content aliases it less.** At severity 1, fog's gap is 0.37 with the level (0.564 against an oracle of 0.933) and 0.17 with the shape (0.797 against 0.963). Contrast's falls from 0.35 to 0.15.
- **What stays out of reach.** Brightness and spatter at severity 1 are blind for both statistics: oracle 0.650/0.646 on the level and 0.591/0.597 on the shape. This is L10 measured on held-out images.
- **A correction to the briefing.** Saturate is aliased, not blind (oracle 0.803), so the photometric-invariance reading does not apply to it.

My thesis needs one correction. "The scene hides it" holds for the level, and only about half holds for the shape.

What I would change:
- Drop row (a) and adopt A's statistic as the single-image row.
- Turn my proposal into the merged paper's analysis section.
- Keep the IV extension: a same-scene reference from a recent clean frame. The oracle sets its ceiling at 0.976/0.948 on the level and 0.968/0.930 on the shape.

**Best other proposal:** A, with the level put back in. Keep A's statistic and mechanism, but make A's own disclosed fallback, level + log peak share, the primary row. It scores 0.890/0.874 on the screen. On 712 held-out images:
- 0.882/0.868, with FPR95 0.274/0.318 and severity-1 common 0.798;
- +0.066/+0.065 over the CDFs and +0.043/+0.003 over Section 4;
- it beats both Section 4 and D's row on common and extra.

**Merge suggestion:** A, C and D are one paper. E's decomposition becomes its analysis section, and B is the baseline it must beat.

- **Thesis:** D's, with A's mechanism. Content sets how much of the frame a channel fires on, and corruption changes how sharply it fires. So read both, against content-matched clean scenes.
- **Primary row:** level + log peak share, k = 50, stage-4 key, s1-s3, equal z-sum.
- **Secondaries:**
  - A's shape-only row;
  - B's log-whitening, as the strongest knob-free joint baseline. It should be presented and cited as Rippel et al. plus a log.
- **Ablation ladder:** C's, extended. It also answers the NAP novelty question:
  - NAP-style one-sided: 0.30 (screen, below chance);
  - two-sided aggregate: 0.69;
  - per-channel global: 0.849;
  - conditioned: 0.897;
  - plus the level.
- **Analysis:** E's split of each family's gap into aliasing and blindness, for each statistic.
- **IV:** D's plan, including kill (b) on Foggy Cityscapes, plus E's same-scene video reference as the extension.

Process: I have read positions 200-911 of the running confirmation, and B read 200-507. Freeze the primary now. Make the 5,000-image decision on positions >= 912 (4,088 images), and also report all 4,800 held-out images.

No proposal drifts into predicting performance drop. All five score separation only.

**Ranking:**
1. A: Best single row, and it passes every bar it pre-registered on 712 held-out images. Its fallback row is the best balanced row. It needs 'and level' added to its thesis.
2. D: Its thesis is the one the per-family held-out data supports, and its IV plan is the best. Its linear row is dominated by the log version and misses its fog bar by 0.005.
3. C: Same row as A, with the most honest negative result (topology) and the cleanest ladder, but no independent idea.
4. B: Supported on held-out images and the cheapest monitor, but it is Rippel et al. plus a log. The per-channel shape row beats it, and whitening hurts that shape statistic, which contradicts B's thesis.
5. E: Mine. Every pre-registered criterion passes, but my single-image row is the weakest on held-out images. The decomposition belongs in the merged paper as analysis, not as a paper of its own.

**Checks run:** All scripts and logs are in roundtable/panelist-E/. Two checks ran over the 3-minute guide (314 s and 285 s), because the confirmation's 10 workers load the CPU.

(1) r2_heldout.py: every proposal's frozen row on 712 held-out images (positions 200-911 of the running confirmation).
- Rows were frozen in R2_PREREGISTRATION.md before reading. The confirmation's 200 screen images equal pilot200.npz bit for bit.
- AUROC common/extra and FPR95 common/extra:
  - A = C: 0.890/0.855, FPR95 0.256/0.342; severities 1-5: 0.807/0.867/0.905/0.928/0.943;
  - D: 0.871/0.865, 0.292/0.324;
  - level + log shape: 0.882/0.868, 0.274/0.318;
  - B: 0.873/0.851, 0.290/0.343;
  - E: 0.846/0.850, 0.336/0.338;
  - Section 4: 0.840/0.864, 0.358/0.326;
  - CDFs: 0.816/0.802, 0.419/0.435.
- Oracles:
  - same-scene, level: 0.976/0.948;
  - same-scene, log peak share: 0.968/0.930;
  - shape with the clean version's neighbours: 0.917/0.873;
  - level with the clean version's neighbours: 0.892/0.888;
  - other-scene key: 0.866/0.858.
- Paired differences, common | extra | severity 1:
  - A - CDFs: +0.074 [+0.066, +0.082] | +0.052 [+0.046, +0.059] | +0.102 [+0.092, +0.113];
  - A - S4: +0.050 [+0.043, +0.057] | -0.010 [-0.015, -0.003] | +0.046 [+0.038, +0.054];
  - D - S4: +0.031 [+0.027, +0.036] | +0.000 [-0.003, +0.004] | +0.024;
  - D - CDFs: +0.055 | +0.062 | +0.080;
  - B - CDFs: +0.057 [+0.049, +0.065] | +0.049 [+0.042, +0.055] | +0.075;
  - B - S4: +0.033 | -0.013 [-0.019, -0.008] | +0.018;
  - E - CDFs: +0.030 [+0.019, +0.041] | +0.048 | +0.065;
  - E - S4: +0.006 | -0.014 | +0.008;
  - A - B: +0.017 [+0.012, +0.022] | +0.004 [-0.000, +0.008] | +0.027 [+0.021, +0.034];
  - A - D: +0.019 | -0.010 | +0.022;
  - level + log shape - A: -0.008 [-0.011, -0.004] | +0.013 [+0.010, +0.016] | -0.009;
  - level + log shape - S4: +0.043 [+0.038, +0.047] | +0.003 [+0.000, +0.007] | +0.037;
  - level + log shape - CDFs: +0.066 | +0.065 | +0.094.
- Severity 1, as A / S4 / CDFs / level oracle / shape oracle:
  - fog: 0.797/0.526/0.636/0.933/0.963;
  - contrast: 0.818/0.537/0.666/0.925/0.965;
  - zoom: 0.884/0.740/0.784/0.988/0.994;
  - brightness: 0.527/0.539/0.510/0.650/0.591;
  - saturate: 0.578/0.575/0.519/0.803/0.747;
  - spatter: 0.517/0.544/0.487/0.646/0.597.
- Every proposal passes its own pre-registered bars except D's fog >= 0.75 (0.745).

(2) r2_family_deltas.py: per-family differences on the same 712 images.
- A - S4 gains: fog +0.293, contrast +0.243, zoom +0.188, defocus +0.062, gaussian blur +0.050, glass +0.049, pixelate +0.031, frost +0.018.
- A - S4 losses, every interval excluding 0: elastic -0.058, spatter -0.041, brightness -0.031, speckle -0.026, saturate -0.022, snow -0.019, shot -0.010, impulse -0.007, gaussian noise -0.006.
- Level + log shape - S4, worst losses: elastic -0.022, spatter -0.015, brightness -0.011.

(3) r2_screen_checks.py and r2_ls_screen.py: the 200-image screen, nothing tuned.
- NAP-style one-sided: s4 0.313/0.380, s1-s3 0.302/0.343. NAP-style two-sided aggregate: 0.689/0.676.
- Whitened log peak share, no key: 0.848/0.826. Whitened conditioned peak share: 0.858/0.832.
- A/C + B summed: 0.894/0.862.
- Level + log shape: 0.890/0.874, which is A's fallback.

(4) r2_heterogeneous.py: a CPU forward pass on 20 pilot images, fog and defocus at severity 2. The CPU features lie within 1.4e-3 of the stored GPU features.
- s1 median change in log peak share, uniform half / ramp-top / ramp-bottom: fog -0.091/-0.079/-0.085 (full -0.197).
- Share of s1 channels that flatten: 0.88/0.84/0.82.
- AUROC, fog: A/C 0.588/0.542/0.623; S4 0.450/0.405/0.487; B 0.530/0.505/0.588.
- AUROC, defocus: A/C 0.695/0.625/0.708; S4 0.680/0.600/0.688.
- My hypothesis that uneven degradation flips the sign is refuted.

(5) NAP formula, checked on the arXiv 2402.18162 HTML: S = (1/C) sum_j (Max(A_j)/(Mean(A_j)+eps))^2, at the penultimate layer, one-sided, semantic OOD only.

### Round 3: rebuttal by E

- **To A** (concede): The aliasing/blindness split depends on which statistic is read. The conditioned shape closes 59–67% of E's fog, contrast and zoom gaps, and A's E4 reaches 0.936 / 0.928 at severity-1 fog / contrast from one image, so 'a single image cannot tell a hazy scene from a fogged one' is refuted. E's deployable rows are weak: (a) is the lowest held-out row, (b) breaks the one-image rule and fails for fog present from the first frame, and E's own blend curve decays with scene change. — Conceded, and my Round-3 check makes the point stronger than A stated it.

r3_matched_decomposition.py scores each statistic against its own same-scene oracle, on 859 held-out images (positions 200–1058; nothing above 1058 read). At severity 1, E4 sits within:
- +0.054 [+0.046, +0.062] of its oracle on fog;
- +0.058 [+0.050, +0.067] on contrast;
- +0.064 [+0.054, +0.075] on zoom blur.
The level row leaves gaps of 0.41 / 0.39 / 0.25 on the same three families.

Fog's gap runs from Section 4 (0.525) to the ceiling (0.990). The scene accounts for only 12% of it (contrast: 13%). The rest is the statistic and the test:
- the statistic (level to log share): +0.27;
- the test (two-sided to signed): +0.11 to +0.14. A mean of |deviations| has zero slope at zero shift, so a shift of μ clean SDs raises it by about 0.4μ², while a signed mean rises by μ.

The signed test even closes 35–40% of the gap on families where A said the shape closes nothing:
- frost: 0.662 → 0.764 (oracle 0.956);
- elastic: 0.598 → 0.716 (oracle 0.885).

What survives is narrower:
- frost, saturate and elastic still sit +0.193 [+0.180, +0.206], +0.170 [+0.156, +0.184] and +0.169 [+0.155, +0.182] below E4's own ceiling, which is 59–74% of their gap;
- brightness and spatter are near-blind (best oracle 0.649 / 0.674).

I withdraw (a) and (b). The oracle stays only as the merged paper's analysis.
- **To B** (partially concede): The decisive number needs the clean twin, so it measures a paired change detector. The 'leak-free' A1 row is also an oracle, because its neighbour is found with the clean twin's key. The deployable (a) is the weakest held-out row. The shape halves the aliasing gap, so blindness must be measured with the best statistic. (b) fails for fog, night or soiling present from the first frame. — Conceded:
- (a) is withdrawn, and (b) leaves the claims.
- A1 is an oracle retrieval. In final_rows.py and r2_heldout.py, `other` is the bank image nearest to the clean version's key. I relabel it 'oracle retrieval of the nearest other scene'.
- Blindness must be measured with the best statistic. The Round-3 check does this over four statistics. At severity 1 the best oracle is ≥ 0.96 on 15 families, 0.806 on saturate, 0.885 on elastic, 0.649 on brightness and 0.674 on spatter.

Rejected: that the oracle being a change detector counts against it as a measurement.
- I never offered it as a detector. It measures what the early channels carry once content is fixed.
- It reads only 512 numbers of the clean view. Another image's key drops it to 0.707 (screen). B's own steelman calls it a quantified ceiling.
- The relabelled A1 is an upper bound on retrieving another scene, so its failure still shows that retrieval cannot close the level's gap.
- Round 3 confirms this for each statistic. At severity-1 common, the key's leak costs 0.03–0.05 and the reference's imprecision 0.06–0.13, for the level, the share, LS and E4. Zoom blur is the only exception.

B's change-of-mind bar (a single-frame row going from 0.889 to ≥ 0.93 common) is not met: E4 reaches 0.916, and M1 0.922 on the screen.
- **To C** (partially concede): The split is a property of the channel mean, not of the detector. The single-image ceiling or share closes about two thirds of E's fog / contrast / zoom gap, which contradicts the thesis's 'so' clause and the IV message that haze, contrast and zoom 'need a same-scene reference'. E's method is either Rippel + SPADE (the weakest held-out row) or a change detector that misses conditions present from the start and gradual onsets. — Conceded: the thesis's 'so' clause, the IV message, and both methods.

One correction: the split is not a property of the mean alone. For the log share with the two-sided rule, the share's own oracle still beats the share at severity 1 (R3, 859 images) by:
- 0.165 on fog and 0.145 on contrast;
- 0.269 on frost;
- 0.184 on gaussian blur and 0.173 on pixelate.

So the split depends on both the statistic and the test. With the signed test (E4) the gap falls to ≤ 0.10 everywhere except frost, saturate and elastic. The honest headline is that residual list, not the level's gap.

C's bar (a share-oracle gap like 0.99 vs 0.80) is met only on frost (0.920 vs 0.651) and gaussian blur (0.955 vs 0.771), not on fog or contrast. No deployable same-scene reference was tested. So C's mind should not change, and mine has.

C's log-ceiling decomposition (ceiling −1.9 SD vs level −0.2 to −0.45 SD) is the mechanism the merged paper uses.
- **To D** (concede): Aliasing depends on the statistic: the log peak share recovers about two thirds of the severity-1 fog, contrast and zoom gap from one image. The deployable same-scene references fail in the IV cases: fog or dirt present from the first frame, slow onsets, and ACDC same-place images that differ in traffic and season (as E's own blend curve predicts). The 'leak-free' key is an oracle. (a) is Rippel + SPADE and the weakest held-out primary. — Conceded on all four points.
1. Round 3 strengthens this point: with the signed flattening arm, the residual at severity-1 fog and contrast is only 0.054 / 0.058.
2. A recent-frame reference degrades with the image when the condition is there from the start or arrives slowly. My blend curve (0.977 → 0.868 as the reference moves to another scene) predicts that ACDC's same-place images lose most of the oracle's advantage. So (b) leaves the paper.
3. A1 is an oracle row; I checked the code and relabelled it.
4. (a) is withdrawn.

D's two change-of-mind conditions:
- ≥ 0.90 on severity-1 fog and contrast without the clean version: met, but by A's single-image E4 (0.936 / 0.928). That supports the merged single-image paper, not my reference.
- A shape-oracle gap ≥ 0.30: not met (0.165 / 0.145 two-sided, 0.054 / 0.058 signed).

D's Koschmieder probe is now the merged paper's main IV risk, and Foggy Cityscapes its decisive driving test.

**Status:** merged (merged with: A, C, D)

**Final proposal:**

=== Final proposal E (merged with A, C, D) ===

**Title:** Flattened or shifted: two readings of a frozen detector's early channels detect mild corruption, and a same-scene ceiling shows what is left

**Status:**
- **Withdrawn:** my Round-1 method (retrieval + whitening = Rippel + SPADE) and its driving variant (a recent-frame reference, which is a change detector).
- **Kept:** the same-scene oracle, recomputed per statistic in Round 3, as this paper's analysis.
- **The method** is A's statistic, with C's flattening direction and D's level-and-crest structure.

## Thesis
A corruption either flattens a frozen detector's early channels or shifts their level:
- "flattens": each channel's peaks fall relative to its level, coherently across channels.

A signed test of the first and a content-matched test of the second detect mild corruption far better than full activation distributions. A same-scene ceiling then shows two things:
- the detector is blind only to mild brightness and spatter;
- what else it misses is hidden by the scene: mild frost, saturation and elastic distortion.

## Who contributes what
- **A:** the log peak share; the coherence evidence; the frozen rows E4 and M1; the never-peeked boundary (positions above 1058).
- **C:**
  - "corruption flattens": at severity 1 the share falls at all three stages for 14 of 19 families;
  - the pre-registered one-sided secondary and the log-ceiling evidence;
  - the topology negative result (one paragraph that closes L6).
- **D:**
  - "judge the crest alongside the level", the two-arm structure that survives family by family;
  - the driving plan, with Foggy Cityscapes as the decisive IV test;
  - the Koschmieder probe, which found the main IV risk.
- **E:** the per-statistic same-scene ceiling; the split of each family's gap; the leak/imprecision split; the slide.
- **B:** log-whitened channel energies as the strongest key-free baseline, and the evidence that whitening divides out the coherent flattening.

## Insight
Numbers are held out (859 images, positions 200–1058, already seen by the panel) unless marked "screen" (200 images).

1. **Visibility loss is a coherent flattening that the level cannot see.**
   - At s1, severity 1, the conditioned log ceiling falls 1.89 / 1.92 / 2.04 clean residual SDs under fog / contrast / zoom blur. The level moves only 0.45 / 0.24 / 0.20 (C, screen).
   - 67–74% of the share's shift energy lies on one clean-residual direction, which holds 19% of clean variance. For the level the figure is 7–10% (A, screen).
2. **For a coherent one-sided shift, a signed test beats a two-sided one.**
   - A mean of |deviations| has zero slope at zero shift: a shift of μ clean SDs raises it by about 0.4μ², while a signed mean rises by μ.
   - Under severity-1 fog, s2–s3 shift by 0.2–0.5 SD. There the per-channel d′ differs 3–8× (independent-Gaussian approximation).
   - At severity 1, two-sided share vs signed flattening + level (E4): fog 0.799 vs 0.936; contrast 0.821 vs 0.928; zoom 0.887 vs 0.932.
   - The log is what makes the flattening direction work (screen, AUROC common):
     - the raw max/mean ratio with no reference, NAP's form, is inverted: 0.402;
     - the log alone gives 0.674;
     - per-channel standardisation gives 0.702;
     - content conditioning, as the arm alone, gives 0.692.
   - The content key adds little to this arm: M1 with a key-free flattening arm is −0.007 [−0.010, −0.003] (screen).
3. **Additive corruptions move the level instead.**
   - The conditioned level beats the share on 9 of 19 families (712 held-out images): the three noises, snow, brightness, elastic, speckle, spatter and saturate.
   - Noise makes channels peakier, so a summed signed arm penalises it. E4 scores 0.746 on severity-1 gaussian noise, below the CDFs' 0.888.
   - So the arms are joined by a max (A's M1), which recovers 0.929 on the screen.
4. **The ceiling** is the same-scene oracle: a ridge (λ = 10) from the clean version's 512 stage-4 means, used for analysis only.
   - Per statistic (common / extra / severity 1): level 0.976 / 0.949 / 0.942; share 0.969 / 0.931 / 0.930; E4 0.978 / 0.930 / 0.953.
   - At severity 1 the best oracle is ≥ 0.96 on 15 families, 0.806 on saturate and 0.885 on elastic.
   - It is only 0.649 on brightness and 0.674 on spatter. This is L10, measured.
5. **Where each family's severity-1 gap goes** (from Section 4 to E4's own ceiling):

   | family | Section 4 (level) | A (share, two-sided) | E4 (signed + level) | ceiling | share of the gap left to the scene |
   |---|---|---|---|---|---|
   | fog | 0.525 | 0.799 | 0.936 | 0.990 | 12% |
   | contrast | 0.538 | 0.821 | 0.928 | 0.986 | 13% |
   | zoom blur | 0.739 | 0.887 | 0.932 | 0.996 | 25% |
   | frost | 0.662 | 0.651 | 0.764 | 0.956 | 65% |
   | saturate | 0.575 | 0.579 | 0.636 | 0.806 | 74% |
   | elastic | 0.598 | 0.573 | 0.716 | 0.885 | 59% |
   | brightness | 0.539 | 0.527 | 0.529 | 0.592 | blind |
   | spatter | 0.545 | 0.520 | 0.588 | 0.674 | blind |

   Paired bootstrap, ceiling minus E4:
   - fog +0.054 [+0.046, +0.062]; contrast +0.058 [+0.050, +0.067];
   - frost +0.193 [+0.180, +0.206]; saturate +0.170 [+0.156, +0.184]; elastic +0.169 [+0.155, +0.182].
6. **Why a better key will not close the rest.**
   - At severity-1 common, two costs appear for every statistic:
     - the key's leak (kNN on the clean version's key, minus the row): 0.03–0.05;
     - the reference's imprecision (ridge ceiling minus kNN oracle): 0.06–0.13.
   - Zoom blur is the exception: there the leak dominates (level 0.161 vs 0.088).
   - A precise reference built from the image's own key is worse than kNN: level 0.789 vs 0.841 (screen), share 0.847 vs 0.889 (B, held out).
   - The reason is that stage 4 slides along the clean manifold under corruption. Its typicality AUROC stays at 0.476 at severity 1, yet 34% of its 50 neighbours are replaced (screen).
   - So the headroom on frost, saturate and elastic needs a reference that is both precise and corruption-invariant. The detector's own late features cannot give one.

**Why it is not obvious.**
- Published monitors are two-sided or quadratic: CDF EMD, Mahalanobis, kNN distance, Section 4. The weak spots were blamed on invariance (L10).
- L4 says the tail loses to the bulk. That holds for raw tails. A log share relative to the level, read with a signed test, reverses it.
- The evidence: for fog and contrast the cause was the statistic and the test, not the detector or the scene. Only brightness and spatter are blind.

## Method (three sentences)
For each channel at s1–s3, read the level m (mean |x|) and the log peak share π = log(top-1% mean + 1e-6) − log(m + 1e-6). Find the 50 clean bank images nearest in bank-standardised stage-4 means, and express each channel's deviation from their average in bank SDs. Score a flattening arm (the signed mean over channels of −Δπ) and a level arm (the mean over channels of |Δm|), z-score each stage on 500 clean images, sum the stages, re-z-score each arm, and alarm on the larger (M1, exactly A's r2_m1.py).

- **Pre-registered fallback:** LS, the symmetric sum of the level arm and the two-sided share (A's E2 = C's C2).
- **Primary rule, fixed now:** M1 is the primary if, on the never-peeked images, M1 − LS > 0 with 95% intervals excluding 0 on common and on severity-1 common; otherwise LS.
- **Analysis (never a detector):** each row is recomputed with (i) the clean version's neighbours and (ii) the ridge from the clean version's own key.
  - leak = (i) − row;
  - imprecision = (ii) − (i);
  - blindness = 1 − (ii).
- **Rules kept:** training-free, frozen detector, clean data only, no labels, one score per image. The flattening direction is a physical prior, not corrupted data.

## Novelty
- **Becker et al. (ICPR 2026):** an EMD between each channel's full CDF and one global training CDF. It is two-sided, dominated by the level, and blind to content.
  - LS beats it on all 19 families at severity 1 (held out).
  - A − CDFs: +0.075 [+0.068, +0.082] common, +0.053 [+0.047, +0.058] extra.
- **NMD:** the level against BatchNorm means, which is our level arm without content.
- **NAP** (Wan et al., arXiv 2402.18162; verified by C and D, not by me): per-channel max/mean at the penultimate layer, one-sided, no reference, for semantic OOD.
  - Transplanted raw, it is inverted here (0.402, screen).
  - What flips it is the log and per-channel standardisation at early stages, plus the level arm for noise.
- **ASH / SCALE** (verified by A and D): ratios across the channels of the pooled penultimate vector, used to rescale logits.
- **Mahalanobis / Rippel et al.** (ICPR 2020, verified): quadratic, so second-order in a small shift. Whitening divides out the coherent flattening: the whitened conditioned share scores 0.856 vs 0.889 (B, held out).
- **SPADE** (verified) **and conditional anomaly detection** (Song et al., TKDE 2007; abstract): retrieval conditioning is known, and we claim nothing for it. It adds about 0.007 to the flattening arm (screen).
- **IQA:** BRISQUE / NIQE (recollection, unverified) and kurtosis or crest sharpness measures (D, title level) are pixel-level ancestors.
- **Claimed as new:**
  1. the signed, early-layer log-flattening test, combined with the level arm by a logical OR;
  2. the per-statistic same-scene ceiling, splitting each family's gap into statistic, test, scene and blindness;
  3. the slide of the detector's late features under corruption.
  My searches did not find these, which is not proof.

## Decisive experiment (CPU, no GPU, no refit)
- **Images:** the never-peeked seed-44 positions 1059–4999 (3,941 when the confirmation completes). Bank and z-statistics come from pilot200.npz.
- **Frozen rows:** M1, LS, A primary, E4, Section 4, B primary, D primary, and the stored CDFs.
- **Ablation, declared now:** M1g, with the flattening arm referenced to the bank mean (no key; r3_global_flatter.py).
- **Analysis:** r3_matched_decomposition.py with the position cap moved.
- **The paper needs (primary row):**
  - primary − CDFs > 0, with intervals excluding 0, on common, extra and severity-1 common;
  - common ≥ 0.88, extra ≥ 0.84, severity 1 ≥ 0.79, FPR95 common ≤ 0.28;
  - family means above the CDFs on all 19 families.
- **Kill criteria:**
  - **K1:** the primary − CDFs interval includes 0 on any of the three. Then there is no single-image claim.
  - **K2:** LS − Section 4 on common is below +0.015, or its interval touches 0. Then the shape adds nothing, and the paper is Section 4 plus the analysis.
  - **K3:** the best oracle is below 0.80 at severity-1 fog or contrast. Then the detector is blind, and the ceiling claim is dropped.
  - **K4:** the primary's residual (its own oracle minus the row) at severity 1 is below 0.10 on all of frost, saturate and elastic. Then "the scene hides the rest" is dead, and the analysis shrinks to "blind: brightness, spatter".
- **Ablation rule:** if M1g − M1 on common has an interval containing 0, or lies above 0, the flattening arm needs no content key. The conditioning claim then covers the level arm only.

## Cheapest first test (done)
- **Held out (Round 2, 859 images):**
  - E4: 0.916 / 0.836, severity 1 0.856, FPR95 0.237;
  - LS: 0.884 / 0.870, severity 1 0.800; LS − Section 4: +0.044 [+0.040, +0.047] / +0.004 [+0.001, +0.008];
  - CDFs: 0.818 / 0.805.
- **Screen:**
  - M1: 0.922 / 0.862, severity 1 0.869, FPR95 0.229;
  - M1 − LS: +0.032 [+0.020, +0.045] common, −0.012 [−0.023, +0.000] extra, +0.063 [+0.046, +0.080] severity 1.
- **Round-3 analysis:** 205 s of CPU on the same 859 images; nothing above position 1058 read.
- **Remaining:** about 10 CPU minutes once the confirmation completes. It held about 1,750 of 5,000 files at 17:35.

## Figure 1
- **(a) The (Δ level, Δ log share) plane at s1,** content-conditioned, in clean residual SDs.
  - Clean images sit at the origin; each family is an arrow over severities 1–5.
  - Fog, contrast, the blurs and JPEG point down (flattened).
  - Noise, snow and spatter run along the level axis (shifted).
  - Brightness and saturate barely move.
- **(b) The ladder at severity 1, per family.**
  - Dots: the CDFs, Section 4, the two-sided share, the primary, and its same-scene ceiling.
  - The bar from the primary to its ceiling is what the scene hides; the gap from the ceiling to 1 is what the detector misses.
  - Families are sorted by the share of the gap left to the scene (the table above).

## Fit for IEEE IV
- **The arms map to road degradations:**
  - flattened: fog, haze, low contrast, defocus, motion blur, a smeared lens, compression;
  - shifted: sensor noise at night, snow, droplets;
  - blind: exposure and white-balance shifts, which need an input-level cue (A's runner-up: a stem black point).
- **Runtime:** a per-channel mean and top-1% at s1–s3 (the top 256 / 64 / 16 positions), plus one 2,000 × 512 neighbour search. This should fit the 0.2–1.2 ms monitor budget (L12; an estimate). D's sort-free ‖x‖₄/‖x‖₁ variant is the embedded option to validate.
- **Driving plan** (bank from Cityscapes train):
  1. Cityscapes-C, single image plus the same-scene ceiling. Prediction: less of the gap is left to the scene on homogeneous driving content.
  2. Foggy Cityscapes (depth-dependent fog), the decisive IV test.
     - Prediction: M1 ≥ the level arm alone, and the flattening arm adds at β = 0.005.
     - Kill: the M1 − level-arm interval includes 0 at β = 0.005 and 0.01. The flattening claim is then a property of the package's fog, and the IV claim reduces to the level.
  3. ACDC fog, night, rain and snow, against its normal-condition images.
  4. nuImages night and rain, through the metadata.
- **No temporal reference is claimed.** A recent-frame reference is a change detector, blind to conditions present from the start (objections from A, B, C and D).

## Risks and cost
- **Selection:** M1 and E4 were chosen after seeing data. Only positions ≥ 1059 count.
- **Depth-dependent fog:**
  - it moves the share less, and s3 gets peakier (A: +0.17 / +0.38 SD at β = 0.01 / 0.02);
  - D's 24-image Cityscapes probe, with a global reference, finds level and share tied (0.903 vs 0.877 at β = 0.01);
  - M1's level arm is the hedge.
- **Partial fog:** fog over half the image cuts the share row from 0.882 to 0.588 (E, 20 images). A two-band version is a pre-registered IV secondary (D's L2 ablation).
- **The max lifts the clean tail:** under M1, brightness and saturate lose 0.03–0.04 against A primary (family means, screen).
- **Novelty margin over NAP:** the log, the early stages, the standardisation, the level arm and the analysis. A reviewer may still call the method "NAP with a log".
- **Cost:**
  - no GPU for COCO;
  - about 60 lines of code (A's r2common.py and r2_m1.py, plus my oracle script);
  - Cityscapes passes take about 1–1.5 h each, with a written plan the user reviews and an agreed window on the shared GPU.

**Vote:** first A, second D. **A first**, on one condition: the primary must be a two-arm row (M1, or LS as the pre-registered fallback), not the share alone, which loses to the level on 9 of 19 families.
- A's statistic and frozen rows carry the merged paper: A − CDFs is +0.075 [+0.068, +0.082] on common, held out.
- E4 reaches 0.916 / 0.836, with 0.856 at severity 1 and FPR95 0.237.
- Round 3 puts E4 within 0.054–0.064 of its own same-scene ceiling on severity-1 fog, contrast and zoom.
- A froze M1 and the never-peeked boundary (1058) before anyone looked at those images.

**D second.**
- Its thesis, judging the crest alongside the level, is the structure that survived family by family: LS beats the CDFs on all 19 families at severity 1.
- Its Koschmieder probe found the main IV risk.
- Its raw ratio is dominated by the log twin (+0.011 [+0.010, +0.012]); that is a fix, not a flaw.

**Not the others.**
- C: the same row as A, and its mechanism (the ratio cancels content) is refuted. Its one-sided secondary still deserves credit for the flattening direction.
- B: whitening divides out the coherent flattening (severity-1 fog 0.632).
- Mine: withdrawn as a method.

**First experiment:** Score the frozen rows on the never-peeked seed-44 positions 1059–4999 of the running 5,000-image confirmation, once it completes.
- **Cost:** about 10 CPU minutes; no GPU, no refit. Bank and z-statistics come from pilot200.npz.
- **Rows:**
  - M1 (A's r2_m1.py), LS (A's E2 = C's C2), A primary, E4, Section 4, B, D and the stored CDFs;
  - the ablation M1g (key-free flattening arm, declared now);
  - the per-statistic same-scene oracles (r3_matched_decomposition.py with the position cap moved).
- **Primary:** M1 if M1 − LS > 0 with 95% intervals excluding 0 on common and on severity-1 common; otherwise LS.
- **Kill criteria:**
  - the primary − CDFs interval includes 0 on common, extra or severity-1 common: no single-image claim;
  - LS − Section 4 on common is below +0.015 or its interval touches 0: the shape adds nothing, and the paper is Section 4 plus the analysis;
  - the best oracle is below 0.80 at severity-1 fog or contrast: the detector is blind;
  - the primary's residual (its own oracle minus the row) is below 0.10 at severity 1 on all of frost, saturate and elastic: "the scene hides the rest" is dead.

**Checks run:** No GPU was used, no repository file was changed, and no image above position 1058 was read.

**(1) r3_matched_decomposition.py.** CPU, 205 s, about 25 s over the 3-minute guideline.
- Held-out positions 200–1058 only (n = 859, already seen by the panel).
- It reproduces A's rows exactly: Section 4 0.841 / 0.866, share 0.893 / 0.858, LS 0.884 / 0.870, E4 0.916 / 0.836, CDFs 0.818 / 0.805.
- Same-scene ridge oracles, common / extra / severity 1:
  - level 0.976 / 0.949 / 0.942;
  - share 0.969 / 0.931 / 0.930;
  - LS 0.976 / 0.943 / 0.941;
  - E4 0.978 / 0.930 / 0.953.
- Oracles built from the clean version's neighbours:
  - level 0.893 / 0.889 / 0.809;
  - share 0.919 / 0.876 / 0.843;
  - LS 0.918 / 0.888 / 0.839;
  - E4 0.942 / 0.867 / 0.891.
- Own oracle minus row at severity 1 (1,000-draw paired bootstrap):
  - LS: common +0.141 [+0.133, +0.149]; fog +0.255 [+0.240, +0.270]; contrast +0.233 [+0.219, +0.247]; frost +0.288; saturate +0.209; elastic +0.223.
  - E4: common +0.097 [+0.090, +0.105]; fog +0.054 [+0.046, +0.062]; contrast +0.058 [+0.050, +0.067]; zoom +0.064 [+0.054, +0.075].
  - E4: frost +0.193 [+0.180, +0.206]; saturate +0.170 [+0.156, +0.184]; elastic +0.169 [+0.155, +0.182].
  - E4: gaussian blur +0.094; pixelate +0.098; brightness +0.064 (oracle 0.592); spatter +0.086 (oracle 0.674).
- Best oracle at severity 1: ≥ 0.96 on 15 families; saturate 0.806; elastic 0.885; brightness 0.649; spatter 0.674.
- Leak / imprecision at severity-1 common: level 0.047 / 0.133; share 0.033 / 0.087; LS 0.039 / 0.102; E4 0.036 / 0.062. Zoom blur is the only family where the leak dominates (level 0.161 / 0.088).

**(2) r3_nap_flip.py** (screen, 200 images): the one-sided flattening arm at s1–s3, AUROC common / extra / severity-1 fog.
- raw ratio, no reference (NAP's form): 0.402 / 0.394 / 0.632;
- log ratio, no reference: 0.674 / 0.566 / 0.914;
- log, standardised against the bank: 0.702 / 0.578 / 0.935;
- log, standardised, conditioned: 0.692 / 0.566 / 0.969.

**(3) r3_global_flatter.py** (screen):
- M1: 0.922 / 0.862, severity 1 0.869.
- M1 with a key-free flattening arm: 0.915 / 0.858, severity 1 0.855. The difference is −0.007 [−0.010, −0.003] / −0.004 [−0.008, +0.000] / −0.014 [−0.020, −0.008].
- M1 − LS: +0.032 [+0.020, +0.045] / −0.012 [−0.023, +0.000] / +0.063 [+0.046, +0.080].

**(4) Code reading.** In final_rows.py and r2_heldout.py, the "leak-free other-scene key" is the bank image nearest to the clean version's key. It is therefore an oracle retrieval.

Scripts and logs are in /tmp/claude-1000/-home-yuchen-YuchenZ-UE-philip-sa/3d52b916-b3ff-4806-a501-7f70cde31a29/scratchpad/detection/roundtable/panelist-E/.
