# Roundtable briefing: a paper on detecting image corruption from a frozen detector

## 0. The roundtable, and the one rule that matters most

**The goal is detecting corruption.** The user's words: "I want to detect the corruption of the images, not how the performance will drop."
- The score must tell a corrupted image from a clean one.
- Predicting the detector's performance drop, LRP or mAP is **out of scope**. Do not optimise for it, propose it, or judge ideas by it.
- A previous roundtable drifted into harm prediction and was discarded for that reason.

**The process:**
- **Five panelists (A–E),** each with a seat that gives a starting angle. A seat is not a cage, but each panelist brings that angle to the critique.
- **Round 1, alone:** brainstorm widely, then propose one paper-level idea, plus at most two runners-up.
- **Round 2, confrontation:** read all five proposals, attack the other four as hard as the evidence allows, and defend your own.
- **Round 3, rebuttal:** answer the attacks on your proposal, then revise it, merge it or withdraw it, and vote.
- **A sixth agent summarizes** for the user.

The user wants an elegant, paper-level idea. The plain monitor, per-channel mean activations compared with clean images, is "too plain": it is effectively published (Section 6).

## 1. The project

- **The paper:** for IEEE IV (Intelligent Vehicles Symposium). It proposes an **image-level corruption detector built into a frozen object detector**, RT-DETRv2-R18 (COCO checkpoint, clean val mAP 0.479).
- **Fixed design rules:** training-free; detector frozen; clean reference images only; no labels; no corrupted data at any point; one score per image.
- **Origin:** the project started from Topological Uncertainty (TU; Lacombe et al., 2021), applied first to decoder queries (August 2026) and then to backbone conv layers (30 September 2026). Neither beat simple statistics.
- **Current best idea, now under confirmation on 5,000 images:** a content-conditioned reference (Section 4).

## 2. Protocol and metrics

**Data:**
- **COCO val2017:** all 5,000 images, shuffled with seed 44. The 200-image pilot uses the first 200.
- **96 conditions per image:** clean, plus 19 `imagecorruptions` families × severities 1–5, with one seeded draw each.
  - **15 "common" families (COCO-C):** gaussian/shot/impulse noise; defocus/glass/motion/zoom blur; snow, frost, fog; brightness, contrast; elastic; pixelate, JPEG.
  - **4 "extra" families:** speckle noise, gaussian blur, spatter, saturate.
- **Clean reference:** COCO train2017 (118,287 images).
- **Input size:** always a 640 × 640 resize, in training and at inference.

**Metrics (all ↑ higher is better, except FPR95):**
- **AUROC per condition:** the clean images are the negatives, and the same images corrupted are the positives.
- **The headline numbers:** mean AUROC over the 75 common conditions, and over the 20 extra conditions. 0.5 is chance.
- **Also reported:** per severity, per family, AUPR, and FPR95 ↓ (the share of clean images flagged when 95% of corrupted ones are caught).
- **95% paired bootstrap intervals** over images, with 1,000 draws. On 200 images that is about ±0.02 AUROC; on 5,000, about ±0.004.

## 3. The numbers

### Table A: separation on all 5,000 images (`docs/coco-baseline-numbers.md`)

| Method (what it reads) | AUROC common ↑ | AUROC extra ↑ | FPR95 common ↓ |
|---|---|---|---|
| Activation CDFs, Becker et al. ICPR 2026: per-channel EMD between the image's activation CDF and the training CDF, backbone C1–C5, stage sums z-scored and added | **0.821** | **0.807** | **0.413** |
| DisCoPatch (ICCV 2025): a patch discriminator, trained by us on clean COCO | 0.764 | 0.751 | 0.567 |
| SAOD min (CVPR 2023): 1 − max detection confidence | 0.731 | 0.664 | 0.714 |
| SAOD top-3 | 0.686 | 0.625 | 0.729 |
| kNN, Sun et al. ICML 2022: layer-4 GAP, L2-normalised, k = 100 (k = 1 gives 0.646) | 0.594 | 0.622 | 0.847 |
| ContrastiveConf (TPAMI 2026) | 0.568 | 0.554 | 0.869 |
| Hashemi et al. (FM 2023): share of the last decoder layer's neurons outside μ ± 2σ | 0.428 | 0.437 | 0.959 |
| Hashemi et al., encoder output maps | 0.346 | 0.401 | 0.974 |

**The activation CDFs by severity (common families):** 0.707 / 0.781 / 0.836 / 0.877 / 0.901. On the extra families: 0.610 / 0.762 / 0.834 / 0.896 / 0.935.

**AUROC at severity 3 / 5 for the three strongest methods.** `*` marks the extra families.

| Family | Act. CDFs | DisCoPatch | SAOD min |
|---|---|---|---|
| gaussian noise | 0.99 / 1.00 | 0.99 / 1.00 | 0.75 / 0.95 |
| shot noise | 0.99 / 1.00 | 0.97 / 0.99 | 0.74 / 0.93 |
| impulse noise | 0.99 / 1.00 | 0.99 / 1.00 | 0.76 / 0.95 |
| defocus blur | 0.94 / 0.98 | 0.80 / 0.85 | 0.80 / 0.92 |
| glass blur | 0.90 / 0.96 | 0.88 / 0.94 | 0.90 / 0.95 |
| motion blur | 0.89 / 0.96 | 0.74 / 0.83 | 0.78 / 0.91 |
| zoom blur | 0.81 / 0.83 | 0.82 / 0.85 | 0.88 / 0.92 |
| snow | 0.91 / 0.93 | 0.84 / 0.84 | 0.73 / 0.80 |
| frost | 0.73 / 0.78 | 0.83 / 0.87 | 0.71 / 0.75 |
| fog | 0.73 / 0.75 | 0.72 / 0.84 | 0.55 / 0.58 |
| brightness | 0.59 / 0.70 | 0.50 / 0.52 | 0.53 / 0.58 |
| contrast | 0.85 / 0.99 | 0.61 / 0.78 | 0.58 / 0.85 |
| elastic transform | 0.61 / 0.72 | 0.68 / 0.72 | 0.68 / 0.74 |
| pixelate | 0.81 / 0.97 | 0.64 / 0.75 | 0.85 / 0.97 |
| jpeg compression | 0.79 / 0.94 | 0.65 / 0.59 | 0.74 / 0.92 |
| speckle noise * | 0.96 / 0.99 | 0.94 / 0.97 | 0.69 / 0.81 |
| gaussian blur * | 0.94 / 0.99 | 0.80 / 0.87 | 0.78 / 0.94 |
| spatter * | 0.87 / 0.98 | 0.81 / 0.89 | 0.66 / 0.81 |
| saturate * | 0.56 / 0.78 | 0.47 / 0.56 | 0.51 / 0.59 |

**The weak spots of the best method:**
- **Brightness and saturate.** Plausibly the detector is invariant to them, because RT-DETRv2-R18 trains with RandomPhotometricDistort (brightness/contrast/saturation/hue, p = 0.5) for 117 of its 120 epochs. A brightness shift is also largely a DC change, which zero-mean edge filters ignore.
- **Elastic, fog, frost and zoom blur.**
- **Severity 1 in general.** At severity 1 the activation CDFs reach only these AUROCs:

  | Family | AUROC |
  |---|---|
  | fog | 0.638 |
  | brightness | 0.510 |
  | contrast | 0.665 |
  | elastic | 0.521 |
  | pixelate | 0.568 |
  | JPEG | 0.632 |
  | saturate | 0.523 |

### Table B: per-stage separation of the activation-CDF monitor, all 5,000 images (from the stored per-stage scores)

| Layer | AUROC common ↑ | AUROC extra ↑ | FPR95 common ↓ |
|---|---|---|---|
| C1 (stem output after max-pool, 64 ch) | 0.790 | 0.811 | 0.447 |
| C2 (stage 1, 64 ch) | 0.811 | 0.813 | 0.418 |
| C3 (stage 2, 128 ch) | **0.821** | **0.814** | **0.411** |
| C4 (stage 3, 256 ch) | 0.793 | 0.755 | 0.487 |
| C5 (stage 4, 512 ch) | 0.656 | 0.601 | 0.748 |

### Table C: the 200-image pilot (`docs/conv-tu-pilot-results.md`)

**The layers:** the inputs of `res_layers[s].blocks[1].branch2a.conv`, which are post-ReLU block outputs.

| Stage | Shape |
|---|---|
| s1 | (64, 160, 160) |
| s2 | (128, 80, 80) |
| s3 | (256, 40, 40) |
| s4 | (512, 20, 20) |

Stages s1–s4 roughly correspond to C2–C5.

**The pilot's scoring:** each layer is scored by its mean Euclidean distance to the 5 nearest of 2,000 clean train images (unnormalised). The four layer scores are z-scored with 500 other clean images and added.

| Row | AUROC common ↑ | AUROC extra ↑ |
|---|---|---|
| Conv TU: top 1% of the 0-dim persistence diagram of the conv graph | 0.661 | 0.615 |
| Control: the same number of heaviest edges (no cycle rule) | 0.658 | 0.609 |
| Control: the same number of largest activations | 0.671 | 0.603 |
| Channel means (mean \|x\| per channel), kNN, 4 stages | 0.798 | 0.847 |
| Channel means vs own clean average (mean over channels of \|v − μ\|/σ), 4 stages | 0.794 | 0.818 |
| Per-channel top-1% means, kNN / own average | 0.637 / 0.765 | 0.728 / 0.721 |
| Per-channel 99th percentiles, kNN / own average | 0.633 / 0.751 | 0.729 / 0.726 |
| 4 × 4-grid channel means, kNN / own average | 0.510 / 0.646 | 0.606 / 0.774 |
| Activation CDFs (same 200 images) | 0.825 | 0.808 |
| DisCoPatch (same 200) | 0.760 | 0.747 |

**Per-layer channel means, kNN (AUROC common / extra):**

| Stage | AUROC common / extra |
|---|---|
| s1 | 0.741 / 0.839 |
| s2 | 0.824 / 0.859 |
| s3 | 0.825 / 0.822 |
| s4 | 0.535 / 0.594 |

Against the own clean average, the per-layer AUROC common is 0.780 at s1 and 0.612 at s4. kNN is ahead at s2–s3, where the own average reaches 0.800 and 0.793.

## 4. The idea to beat: a content-conditioned reference (screened today, now confirming on 5,000 images)

**Observation.** On clean images, most of the variation of the early stages' channel means is scene content. A ridge map from the stage-4 channel means predicts 73%, 77% and 82% of it at s1, s2 and s3. (Held-out R² on 500 clean images; map fitted on the 2,000-image bank.) That content variation is what hides mild corruptions from a global clean reference.

**Method:**
1. **Find the neighbours.** For each image, find the k = 50 clean bank images nearest in stage-4 channel means (each channel standardised with the bank's mean and std, Euclidean). Stage 4 barely reacts to corruption (AUROC 0.535), but it still describes the scene.
2. **Score the early stages.** At s1–s3, score the image's channel means by the mean over channels of |v − μ_nb| / σ_bank. Here μ_nb is the neighbours' mean, and σ_bank is the bank's per-channel std.
3. **Combine.** Z-score each stage's score with the 500 clean z-statistics images, scored the same way, and add the three.

**Screen** (200 pilot images, 95% paired bootstrap):

| Row | AUROC common ↑ | AUROC extra ↑ | Common, severities 1–5 |
|---|---|---|---|
| Content-conditioned (k = 50) | **0.841** [0.824, 0.858] | **0.870** [0.855, 0.884] | 0.763 / 0.816 / 0.850 / 0.879 / 0.900 |
| Same stages vs the global clean average | 0.811 | 0.837 | 0.713 / 0.777 / 0.820 / 0.859 / 0.884 |
| Channel means kNN (4 stages) | 0.798 | 0.847 | 0.719 / 0.771 / 0.812 / 0.836 / 0.854 |
| Activation CDFs | 0.825 | 0.808 | – |

**Paired differences, the conditioned row minus each other row:**

| Other row | Δ AUROC common | Δ AUROC extra |
|---|---|---|
| Activation CDFs | +0.016 [−0.004, +0.035] | +0.062 [+0.045, +0.078] |
| The global clean average | +0.031 [+0.018, +0.045] | +0.033 [+0.022, +0.044] |

**Sensitivity:**
- k = 10, 50 and 200 give 0.836, 0.841 and 0.840 on the common families.
- A *linear* version is worse: 0.789 common and 0.811 extra. It predicts s1–s3 from s4 with the ridge map, then standardises the residual. So the neighbourhood itself matters.

**Candidate thesis:** "the detector's own invariance gives a free content anchor". The back says *what* the scene is, and the front says *how it looks*. A corruption is the front disagreeing with clean scenes like this one.

**Status:**
- **Novelty is unchecked.** The nearest relative known so far is Lee et al.'s (2018) class-conditional Mahalanobis, which conditions on the predicted class within one layer.
- **The 5,000-image confirmation runs in parallel with this roundtable.** It has fixed choices: k = 50, the stage-4 key, s1–s3 scored. The 4,800 images outside the screen are held out.
- **Panelists may improve, generalise or replace this idea, or attack it.** It is the idea to beat, not the answer.

## 5. Lessons so far (for detection)

**L1. The front-to-middle of the detector sees corruption; the back does not.**
- Separation peaks at C2–C3 and collapses at C5 (Table B).
- The decoder and the encoder output point the wrong way: Hashemi et al. score 0.43 and 0.35, below chance, because degraded images give *fewer* extreme late activations.
- The August decoder-query TU fingerprints *shrank* under blur, toward the typical, low-magnitude background region of the clean bank.

The user's reading: "the latter is more tuned to the detection head". The invariance of the back is a nuisance for detection, but also an asset (Section 4).

**L2. Position should be ignored on COCO.**
- The 4 × 4 grid of channel means falls to chance with kNN (0.510).
- A fixed spatial cell holds different content in every image.
- Caveat: on a fixed-camera driving dataset (sky at the top, road at the bottom), coarse position might carry signal. This is untested.

**L3. Sorting destroys the cue: keep channel identity.**
- All three sorted top-K lists reach about 0.66: the persistence diagram's top 1%, the heaviest edges, and the largest activations.
- The unsorted per-channel means reach 0.80 / 0.85.
- Positions are exchangeable samples; channels are coordinates with identity.

**L4. The bulk of each channel's distribution carries the signal, not its tail.** Per-channel top-1% means and 99th percentiles lose to the plain means.

**L5. One number per channel is almost the whole distribution.** The plain channel means come within 0.03 AUROC of the full per-channel CDF/EMD monitor on the common families, and beat it on the extra families.

**L6. Topology added nothing at the top of the diagram.**
- At K = 1% of nodes, the top of the 0-dim persistence diagram of the conv graph equals the heaviest edges almost exactly, because the cycle rule rarely acts.
- The edge weight |K_eff|·|x| is dominated by |x|.
- Deeper into the diagram (K = 25–50%) the cycle rule matters more. That is untested and costly.

**L7. Global kNN typicality and per-channel standardised deviation separate about equally** (0.798 vs 0.794). Unnormalised kNN distances are dominated by high-variance channels.

**L8. Magnitude matters; combine layers with care.**
- Fog, contrast and brightness change activation size, so unnormalised statistics must keep it.
- Z-scoring each stage before adding helps: +0.044 AUROC for the CDF monitor over the plain sum, where C5's 512 content-driven channels dominate.
- Adding a layer that does not separate (C5, s4) only adds noise. Leaving s4 out of the own-average raises it from 0.794 to 0.811.

**L9. Content variation is the main nuisance.**
- Clean images differ a lot in their early-stage statistics because their content differs.
- The early-stage means are 73–82% predictable from stage 4 on clean images (Section 4).
- That is why mild corruptions are hard for any global reference.

**L10. Some corruptions are close to natural variation.**
- Mild brightness and saturation changes look like ordinary lighting.
- The detector was trained to be invariant to photometric changes (L1, Table A).
- Any monitor that reads only the detector's features may be capped on these families. Earlier layers, or the input itself, keep more of that information.

**L11. Process lessons:**
- **Selection on the screening images is fragile.** Improvements of one coarse step on the same 250 images were not evidence.
- **Pre-registered decision rules once stopped a false positive.**
- **Dead or padded query slots contaminated a bank,** and confidence thresholds caused survivor bias. Keep full coverage.

**L12. Engineering facts:**
- RT-DETRv2 fixes the input at 640 × 640, so feature-map sizes are constant.
- Every monitor is cheap: the detector takes 5.9 ms per image, and the monitors add 0.2–1.2 ms.
- A 200-image × 96-condition screen reproduces the 5,000-image ordering (CDFs 0.825 vs 0.821).

## 6. Related work to distinguish from (details: `docs/literature-review-image-corruption-detection.md`)

**Neural Mean Discrepancy** (Dong et al., CVPR 2022, arXiv 2104.11408; verified):
- It compares activation means with the training means, taking those from the BatchNorm layers as a "free lunch".
- It reports that the means of out-of-distribution inputs deviate more.
- This is the published ancestor of the "plain" channel-mean monitor.

**Becker, Bayer, Hübner & Arens, "Operational Readiness for Object Detection" (ICPR 2026):** the closest published work.
- Frozen detectors, including RT-DETR-l; clean COCO only; per-channel activation CDFs vs the training CDF via EMD; COCO-C AUROC per severity.
- No code; we reproduced it (Table A).

**Becker et al., "degradation manifolds" (2026 preprint):** trains a monitoring branch on synthetically damaged images, so it is not training-free.

**Other methods:**
- Mahalanobis (Lee et al., NeurIPS 2018; class-conditional, multi-layer, pooled features);
- Gram matrices (Sastry & Oore, ICML 2020);
- kNN-OOD (Sun et al., ICML 2022);
- DisCoPatch (ICCV 2025);
- Hashemi et al. (FM 2023);
- SAOD (CVPR 2023);
- Topological Uncertainty (Lacombe et al., 2021; no detector application found).

**Image-quality models:** ARNIQA (an embedding + clean prototype reached 85.7 AUROC at severity 5 in Becker et al.'s test), QualiCLIP/CLIP-IQA, NIQE (natural-scene statistics, clean-only).

**Foundation-model density:** Keser et al., BMVC 2025, segmentation on Cityscapes → Foggy/ACDC.

**Degradation classifiers trained with labels** (soiling, weather) are a different setting.

**Possibly relevant, check yourself:** conditional or contextual anomaly detection; reconstruction of shallow features from deep ones (reverse distillation, trained); test-time BatchNorm adaptation (TENT).

## 7. Assets and constraints

**Code:** `/home/yuchen/YuchenZ/UE/philip_sa/.worktrees/convtu-pilot` (branch `convtu-pilot`).
- `differential_uncertainty/convtu/`: hooks on the backbone convs (`tap.py`), the exact persistence code (`graph.py`), kNN and channel statistics (`features.py`, `channels.py`) and the phases (`pipeline.py`).
- `differential_uncertainty/baselines/`: the six baselines, the protocol (`protocol.py`) and metrics and reports (`metrics.py`, `report.py`).
- `src/`: the RT-DETRv2 model.

**Stored data you can analyse on CPU now:** `/tmp/claude-1000/-home-yuchen-YuchenZ-UE-philip-sa/3d52b916-b3ff-4806-a501-7f70cde31a29/scratchpad/detection/data/`.
- Its `README.md` describes every key, and shows how to compute the headline metrics with `report.headline_numbers` (use the `auroc_*`, `aupr_*` and `fpr95_*` keys; ignore the harm keys).
- **`coco5000.npz`:** every baseline score for 5,000 images × 96 conditions, including the CDF monitor's per-stage scores.
- **`pilot200.npz`:** for the 200 pilot images × 96 conditions, the per-channel means, top-1% means and 99th percentiles at s1–s4, plus the 2,000-image clean bank and 500 z-statistics images.
- **The screen's code:** `content_conditioned.py` and `content_conditioned_bootstrap.py` in the parent folder.

**Raw per-image files:** `/home/yuchen/YuchenZ/UE/philip_sa/runs/coco-baselines/`:
- `test_convtu_channels/`: 4 × 4 grids for the 200 pilot images;
- `cdf/reference.npz`: the clean CDF references;
- `hashemi/intervals.npz`: per-neuron μ/σ for the decoder and encoder maps.

**Not stored:** full feature maps, per-channel histograms of test images, decoder queries, encoder features, and anything at the input-pixel level. Those need a GPU pass.

**The detector:**
- RT-DETRv2-R18 = ResNet-18-vd backbone → hybrid encoder (three 256-channel maps at strides 8/16/32) → decoder (3 layers, 300 queries) → heads.
- The backbone BatchNorms were trained with the detector (`freeze_norm=False`), so their running statistics are COCO-train statistics.
- Checkpoint: `/home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth`.

**Datasets on disk** (`/home/yuchen/YuchenZ/Datasets/`):
- COCO; Cityscapes (`leftImg8bit` + `gtFine`); nuImages;
- Fishyscapes, StreetHazards, SegmentMeIfYouCan, VOC, Visual Genome, OWOD splits.

Foggy Cityscapes and ACDC are not on disk, but can be downloaded.

**Compute:**
- One RTX 5090, shared, and in use by other sessions and by the 5,000-image confirmation. **Panelists must not use the GPU.**
- A new statistic on the 200-image screen costs about 6 minutes of GPU. On all 5,000 images, about 2.5–3 hours, limited by corruption generation on the CPU.

## 8. Rules for panelists

**Read-only in the repository.** Write scratch files only under `/tmp/claude-1000/-home-yuchen-YuchenZ-UE-philip-sa/3d52b916-b3ff-4806-a501-7f70cde31a29/scratchpad/detection/roundtable/panelist-<your letter>/`.

**CPU checks on the stored data are encouraged when they settle a claim:**
- Keep each under about 3 minutes.
- Run with `CUDA_VISIBLE_DEVICES="" nice -n 10 /home/yuchen/miniconda3/envs/UE/bin/python <script file>`.
- Write scripts to files; inline heredocs may be refused by the command guard.
- Report the exact numbers.
- A check on the 200 pilot images is a screen, not a proof. If you tune anything on them, say so.

**Literature:** web search is allowed for novelty checks. Cite only what you verified, and mark recollections as unverified.

**Be concrete:** give lesson numbers, numbers and failure scenarios.

## 9. What "paper-level and elegant" means here

1. **A one-sentence thesis about detecting corruption that a reviewer remembers.** It must be a non-obvious insight, not "a better row".
2. **A method stated in three sentences.** Every design choice should follow from the lessons, and none should contradict them.
3. **Clear novelty against the closest work:** Becker et al. ICPR 2026, Neural Mean Discrepancy, Mahalanobis/Gram, kNN-OOD, DisCoPatch, IQA models, and TU.
4. **A decisive experiment on separation,** with a kill criterion, and the cheapest first test, ideally on the stored data today.

   The bar: beat the activation CDFs (0.821 / 0.807 on 5,000 images) with intervals excluding 0, ideally most at mild severities and on the current weak spots.
5. **A fit for IEEE IV:** runtime monitoring of camera perception, driving data (Cityscapes-C, Foggy Cityscapes, ACDC, nuImages), and low latency.
6. **Honesty.** Topology is welcome only where it earns its place (L6). Harm prediction is out of scope (Section 0).
