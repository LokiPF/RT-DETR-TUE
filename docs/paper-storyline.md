# Paper storyline: corruption detection from a frozen detector's early channels

Notes for writing the IEEE IV paper, 2 October 2026. The aim is a paper in which every step reads as the obvious next move, because each one answers a question that the step before it raised. The numbers below come from three sources, all on the 5,000 COCO val images unless marked:
- `docs/conv-tu-conditioned-results.md`;
- `docs/results/paper-evidence/`, the evidence tables made for this storyline (evening of 2 October). The forward-pass tables there use 200 images;
- `docs/dev-log.md`.

Items marked **(to make)** still need to be computed or written; **(open)** marks a decision for the authors.

**Scope (user, 2 October).** The paper presents only the final method, the two-axis score, as ours.
- **Ditched approaches stay out.** These are the decoder-query fingerprint and the conv-TU topology pilot.
- **The exception: two that pointed the way.** Each gets one sentence at most, where it set a direction (steps 2 and 3 below).
- **Everything else in the tables** is either a published baseline or an ablation of the final method.

## The story in one sentence

A frozen detector already has what it needs to notice that its input is corrupted:
- its deep layers say *what* the scene is, and react least to the corruption;
- its early layers say *how the scene looks*;
- a corruption either *flattens* the early channels or *re-weights* them, compared with clean scenes like this one.

## The chain of questions (the skeleton of the paper)

Each step follows one pattern: a question, then an observation with its evidence, then the design choice it forces. The design choice in turn raises the next question.

1. **The problem: can a deployed detector tell that its camera image is degraded?**
   - **Why it matters.** Fog, blur, noise and compression degrade detection silently. A vehicle needs a flag: "this input is not like the images I was built for".
   - **Requirements.** Each one fixes a design choice later.
     - (R1) **Keep the detector frozen.** Nothing is retrained and no second network is added, so we read the features it already computes.
     - (R2) **No corrupted data.** Field corruptions are unknown in advance, so the reference is clean images only.
     - (R3) **One cheap score per image,** so we use per-channel summaries, not per-pixel models.
     - (R4) **Catch mild corruption.** Severity 1 is where a degraded image is easiest to miss, so it is reported throughout.
   - **The goal is detecting the condition, not predicting the mAP drop.** One sentence in the introduction is enough.
2. **Where in the detector should a monitor look?**
   - **Observation (claim C1 below):** a simple statistic separates corrupted images less well the deeper it is read (Figure 3). The detector is trained to give the same answer despite nuisance changes, so its deep features keep the scene and lose much of the corruption, while the early features keep both.
     - Each stage's channel levels against the clean average separate at 0.760 / 0.794 / 0.786 at stages 1–3, and at 0.599 at stage 4 (common families).
     - The peak share falls the same way: 0.842 / 0.828 / 0.811, then 0.739. So do the activation CDFs at stage 4 (0.793 → 0.656).
     - After the encoder and on the decoder queries, the published monitor of Hashemi et al. falls below chance: 0.34–0.38 and 0.428 / 0.437. A corrupted image trips fewer of its clean intervals than a clean image does.
     - The fall is learned. A random-weight backbone of the same architecture separates as well at stage 4 as at stage 1 (0.762 against 0.786), while the trained one falls from 0.759 to 0.602 on the same 200 images.
     - **At most one sentence on our own dead end:** our first attempt read the decoder queries and did no better than the detector's own confidence. That is what turned us to the early layers.
   - **Choice:** read stages 1–3.
   - **Bridge:** what should we measure there?
3. **What should we measure in the early stages?**
   - **Observation:** every image has a different layout, so a statistic tied to a position mostly measures content.
   - **Observation, one sentence from an earlier pilot:** summaries that sort the strongest responses of a layer mix its channels together. They reach about 0.66 AUROC, against 0.80 for one mean per channel. So the signal lives in which channel moves, not in the extreme values.
     - Keep it to that sentence. Do not describe the pilot's topological method.
   - **Choice:** per channel, with position ignored. The simplest such summary is the channel's **level**, its mean |activation|.
   - **Starting point:** this is close to Neural Mean Discrepancy (Dong et al., CVPR 2022), which compares channel means with the training means. Present it as the established idea to improve on.
   - **Bridge:** the level against the average clean image reaches 0.800 / 0.830 (common / extra). What holds it back?
4. **Why is the plain level not enough? Content.**
   - **Observation:** on clean images, the early channel levels vary mainly with what is in the scene. A linear map from the stage-4 means predicts 67 / 75 / 81% of their clean variance at stages 1 / 2 / 3, on held-out clean images. A mild corruption hides inside this content variation: at severity 1, fog moves a stage-1 channel by 0.59 of its clean spread on average.
   - **Choice:** compare each image with clean images of *similar content*, not with the average clean image.
   - **Where content comes from, for free:** the deep stage that reacts least to corruption (step 2) still describes the scene. The image's 50 nearest bank images share 2.5 times as many COCO labels with it as random bank images do, and their object counts follow its own (Spearman 0.73). Its weak reaction to corruption, which makes it a poor detector, makes it a good content key.
   - **Method:** the 50 clean COCO-train images nearest in stage-4 channel means are the reference.
   - **Why neighbours and not the linear map.** The map is the better content model, but it follows the corruption:
     - from the corrupted image's own key, its prediction moves with the corruption, and explains most of it away. At severity-1 fog it moves −0.24 of the image's −0.30 clean SD at stage 1;
     - the neighbours are clean images, so their mean moves only −0.06;
     - so the map scores 0.783 / 0.798, below even the plain level, against 0.841 / 0.866 for the neighbours.
   - **Evidence:**
     - level against similar scenes: 0.841 / 0.866, against 0.800 / 0.830. On the 4,800 held-out images the gain is +0.041 / +0.037, both intervals above 0;
     - the gain is largest at severity 1: +0.061 at severity 1, falling to +0.023 at severity 5 (common), every interval above 0;
     - k hardly matters: 20 to 100 neighbours are within 0.002 of k = 50. With 1 neighbour the score loses 0.06–0.07;
     - the bank could be smaller: 1,000 images lose 0.001–0.003, 250 images about 0.01.
   - **Bridge:** which corruptions does the level still miss, and why?
5. **The level misses corruptions that remove structure.**
   - **Observation:** the level is near chance on fog (0.51 at severity 1) and contrast (0.53), and weak on zoom blur (0.73). These corruptions weaken edges and texture. A channel's strongest responses fall much more than its average (Figure 1):
     - at severity-1 fog, the summed level of the stage-1 channels falls 5%, but their summed top-1% falls 23%. Per channel, the log of the top falls 2.6 times as much as the log level (−0.276 against −0.106);
     - 81–84% of stage-1 channels get flatter under fog, contrast, defocus, glass and zoom blur, against 48–52% under the noises;
     - they flatten together: 45–51% of each image's change lies along the direction in which all channels move together, which holds 14% of the clean variation;
     - the flattening fades with depth: under fog, 84 / 75 / 62 / 48% of the channels flatten at stages 1–4.
   - **Choice:** a second, scale-free number per channel, the **peak share**: log(mean of the strongest 1% of positions) − log(mean). It is judged against the same 50 similar clean scenes. The flattening is shared across channels, so its arm averages the *signed* change over the channels, and the change adds up.
     - **Why a log ratio.** The raw ratio costs 0.018 / 0.013. The top-1% mean alone, with or without the log, costs 0.018–0.019 on common and 0.035 at severity 1, though it is 0.002–0.003 better on extra.
   - **Observation, the other half: noise re-weights the channels.**
     - Noise moves the stage-1 levels the most of any family: by 1.27 clean SD per channel at severity 1, against 0.59 for fog.
     - It moves them in both directions: 36% of the channels rise by more than 10% and 43% fall by more than 10%, while the summed level stays at ×0.99.
     - It makes the channels slightly *peakier*, so the flattening arm points the wrong way (0.20). The level arm, an unsigned deviation per channel, catches it (0.96).
   - **Choice: the two-axis score.** It takes the larger of two scores:
     - the flattening arm: the signed drop in peak share, averaged over channels, so only a shared flattening counts;
     - the level arm: the unsigned deviation of each channel's level, averaged over channels, so that changes in both directions add up.

     Each arm is standardised on 500 clean images. A corruption moves along one axis or the other, so the larger arm catches either. The flattening arm is the larger on 98–99% of fogged images, on 0% of images with Gaussian noise, and on 53% of clean images.
   - **Evidence, the ladder (all 5,000 images, common / extra):**

     | Score | Common | Extra |
     |---|---|---|
     | Level vs average clean | 0.800 | 0.830 |
     | Level vs similar scenes | 0.841 | 0.866 |
     | Peak share vs similar scenes | 0.898 | 0.861 |
     | Two-axis | 0.917 | 0.858 |

     Fog at severity 1 goes from 0.51 to 0.95.
   - **Bridge:** is this better than the published monitors, and was it tested fairly?
6. **A fair test.**
   - **Data:** all 5,000 COCO val images under 19 imagecorruptions families × 5 severities, one seeded corruption per image and condition.
   - **Reference:** clean COCO-train images only, 2,000 for the bank and 500 for the z-statistics.
   - **Baselines:** six published monitors on the same frozen RT-DETRv2-R18 and the same images.
   - **Pre-registration:**
     - the level score's rule was fixed before the 5,000-image run;
     - the two-axis headline's rule was fixed before the last 3,030 images had been read by anyone;
     - those untouched images are reported separately.
7. **Results.**
   - **Headline:** two-axis 0.917 / 0.858 against the strongest baseline, the activation CDFs (Becker et al., ICPR 2026), at 0.821 / 0.807. The gain is +0.096 / +0.051, with every interval above 0.
   - **The untouched 3,030 images give the same result** (0.916 / 0.858), so the design was not tuned to the test set.
   - **Mild corruption:** severity 1 is 0.863 against 0.707 for the CDFs; the gap closes as severity grows.
   - **Low false-alarm rate:** FPR95 0.226 against 0.413 for the CDFs.
8. **Where it fails, and why (claim C8 below).**
   - Brightness (0.53 at severity 1, 0.61 over all severities) and saturate (0.60, 0.67) stay weak for every method. Spatter is weak only at severity 1 (0.57). From severity 2 on it is 0.87–1.00.
   - **The features barely move.** At severity 1, brightness shifts a stage-1 channel by 0.16 of its clean spread (fog 0.59, noise 1.27). Only 7% of the channels change by more than 10%.
   - **The reference is not to blame.** With the clean twin's neighbours, brightness stays at 0.53 at severity 1 (0.64 over all severities), saturate at 0.62 and spatter at 0.59. The signal is missing from the features, not from our reference.
   - **Why: these changes keep the image's structure; the training's jitter is not the reason.** Pure operations tell the two explanations apart:
     - losses of contrast *inside* the training's jitter range are detected (×0.75: 0.76; ×0.5: 0.94);
     - hue shifts far *outside* it stay invisible (0.52–0.56);
     - what stays invisible is what keeps the luminance structure: hue, moderate saturation, and offsets until they clip. Two thirds of the stem's first filters are almost blind to a uniform offset.
   - **A second blind spot, by design.** The flattening arm is one-sided, so a change that makes the channels peakier, such as more contrast or sharpening, is seen only by the level arm. Contrast ×1.5 gives the two-axis score 0.45.
   - **The larger arm costs a little where neither arm moves much.** Over all severities, brightness is 0.61 against 0.68 for the level arm alone, and saturate 0.67 against 0.73. On the extra families, the two-axis score is 0.008 below the level alone.
9. **Does it transfer to driving? (the deciding test for IV)**
   - **Bank:** the reference must come from the deployment camera. With a COCO bank, the content-matched rows fell to 0.54–0.80 on synthetically fogged Cityscapes frames.
   - **Plan:** a Cityscapes-train bank, scored on Foggy Cityscapes, ACDC (real fog, night, rain, snow) and Cityscapes-C **(to make, needs downloads and GPU)**.
   - **Kill test:** under real fog, the peak share must still fall at stage 1, and the headline must not lose to the level.

## Claims, the reasons behind them, and the evidence

Every "because" in the story is a claim a reader can question. Each claim below gives:
- **Why:** the reason we believe it;
- **In hand:** the evidence we already have;
- **To make:** the evidence still to produce;
- **Status.**

**Sources:**
- *results*: `docs/conv-tu-conditioned-results.md`, all 5,000 images;
- *roundtable*: `docs/roundtable-2026-10-01-detection/`;
- *screen*: the first 200 images;
- *911 held out*: positions 1059–1969.

**Cost of the evidence still to make:**
- **CPU, stored:** minutes, from the stored statistics in `runs/coco/`.
- **CPU, forward:** about an hour, running the backbone on a few hundred images on the CPU.
- **GPU:** needs a window agreed with the session that uses the card.
- **Reading:** code or literature.

Numbers measured on the screen or on part of the images are recomputed on all 5,000 images before they go in the paper.

### C1. The deep stage reacts least to corruption but still describes the scene; the early stages keep both

- **Why:**
  - The detection loss rewards features that tell objects apart whatever the nuisance.
  - Training adds photometric jitter, zoom-out and crops. RT-DETRv2 uses torchvision's `RandomPhotometricDistort` with p = 0.5 until epoch 71 of 120 (`configs/rtdetrv2/include/dataloader.yml` in the training repository).
  - Deep units pool over large receptive fields.
  - Early units are generic local filters (edges, texture; Zeiler & Fergus 2014, Yosinski et al. 2014), which a corruption changes directly.
- **In hand** (all 5,000 images; the tables are in `docs/results/paper-evidence/`):
  - **Separability falls with depth** (`depth_curve.csv`, Figure 3). Each stage's own score against the bank's mean, common / extra:
    - level: 0.760 / 0.826 at stage 1, 0.794 / 0.825 at stage 2, 0.786 / 0.770 at stage 3, then 0.599 / 0.585 at stage 4;
    - peak share: 0.842 / 0.795, 0.828 / 0.799 and 0.811 / 0.779, then 0.739 / 0.684;
    - the activation CDFs (common): 0.790 at the stem, 0.811, 0.821 and 0.793, then 0.656 at stage 4;
    - Hashemi et al.: 0.343–0.377 after the encoder and 0.428 / 0.437 on the decoder queries. Below chance means that a corrupted image trips *fewer* of the monitor's clean intervals than a clean image does.
  - **The key still describes the scene** (`key_categories.csv`, `stored.json`):
    - the 50 neighbours share 2.5 times as many COCO panoptic labels with the image as 50 random bank images do. The Jaccard index is 0.217 against 0.085: 0.208 / 0.078 for objects and 0.234 / 0.091 for background classes ("stuff");
    - their object count follows the image's: Spearman 0.73, against −0.02 for random images;
    - if the image shows sky, 70% of its neighbours do, against 22% if it does not and 35% of the bank. The same holds for a person (80% / 44% / 55%), a road (40% / 8% / 11%) and a table (40% / 9% / 15%);
    - the key linearly predicts 67 / 75 / 81% of the clean variance of the levels at stages 1 / 2 / 3, and 61 / 67 / 67% of the peak share's. The map is a ridge (λ = 10) fitted on the bank and tested on the 500 z-statistics images.
  - **The key moves, but to similar clean scenes** (`key_stability.csv`):
    - on average, 63% of the 50 neighbours survive a severity-1 corruption, falling to 50 / 40 / 31 / 24% at severities 2–5;
    - the key moves by half its neighbourhood's radius at severity 1, and by one radius at severity 5 (medians);
    - yet the reference barely moves. Under severity-1 fog, the neighbours' mean shifts −0.06 bank SD at stage 1, in both level and peak share, while the image shifts −0.30 and −0.83.
  - **What the drift costs** (`twin_key.csv`): with the clean twin's neighbours, the two-axis score would reach 0.936 / 0.877, against 0.917 / 0.858. Zoom blur loses the most: 0.902 against 0.955 at severity 1.
  - **The invariance is learned** (`untrained_backbone.csv`, `untrained_key_categories.json`; 200 images, clean and the 19 families at severities 1, 3 and 5). The control is a random-weight backbone of the same architecture, with its batch norm re-estimated on 200 clean train images:
    - its stage 4 separates as well as its stage 1: 0.762 against 0.786 (level against the bank mean, common). On the same images, the trained backbone falls from 0.759 to 0.602;
    - its key finds images that look alike rather than show the same scene: label Jaccard 0.123, against 0.221 for the trained key and 0.083 for random bank images;
    - its key moves more under corruption: 33% of the neighbours survive severity 1, against 63%;
    - the whole method on the random backbone reaches only 0.727 / 0.705, against 0.930 / 0.851 for the trained one on the same images and conditions;
    - its stage-4 neighbours explain *more* of its early levels' clean variance (71%, against 36–47%), because every stage of a random network carries the same low-level statistics. So variance explained does not by itself measure scene content; the label overlap does.
- **To make:** nothing on COCO.
- **Status:** supported, with a correction. Stage 4 reacts *less* than stages 1–3 (0.60 / 0.74 against about 0.8), not "barely", and the random-weight control shows that training made it so. Its job is the key, and the key's drift costs 0.02.

### C2. The signal is in which channel moves, not where in the image or in the extreme values

- **Why:** every image has a different layout, so a fixed position holds different content in every image. Sorting the strongest responses mixes the channels. A channel is a feature detector, and its average activation says how much of that feature the image has.
- **In hand** (dev log, 1 October; screen):
  - summaries that sort the strongest responses reach about 0.66, against 0.80 for one mean per channel;
  - a fixed list of heavy cells is a weighted crop of the activation map, so it measures content;
  - a 4 × 4 grid and per-channel tails alone were worse on COCO;
  - one mean per channel comes within 0.03 AUROC of the full per-channel CDF.
- **To make:** nothing. This is a one-sentence pointer in the paper (Scope).
- **Status:** supported, in one sentence.

### C3. On clean images, the early levels vary mainly with content, and that variation hides mild corruption

- **Why:** early filters respond to edges and texture, and how much texture a scene has depends on what is in it, such as a busy street or an empty sky.
- **In hand:**
  - The stage-4 key linearly predicts 67–81% of the clean variance of the levels at stages 1–3 (C1). "Content" here is the scene and its look together. In a random-weight backbone, the key's neighbours explain 71% of that variance (the trained key's neighbours: 36–47%) by matching appearance alone (C1).
  - The shifts are small against that variation (`effect_sizes.csv`). At severity 1 a stage-1 channel moves, on average, by:
    - 0.59 of its clean spread under fog, 0.55 under contrast and 0.67 under zoom blur;
    - 1.27 under Gaussian noise;
    - 0.16 under brightness and 0.06 under spatter.
  - Around the neighbours' mean, the clean spread is 0.80 / 0.75 / 0.72 of the bank's at stages 1 / 2 / 3, as a median over channels (`spread_ratio.csv`). So the neighbours remove 36 / 44 / 48% of each channel's clean variance, and the same shift becomes 1.3 times as large: fog's 0.59 becomes 0.77.
- **To make:** nothing.
- **Status:** supported.

### C4. Comparing with similar clean scenes removes that variation, and helps most at mild severity

- **Why:**
  - Among similar scenes each channel's clean spread is smaller, so the same shift is a larger deviation (C3).
  - The neighbours are always clean images, so their mean cannot follow the corruption. A linear map from the key extrapolates from the corrupted key, so its prediction does follow it.
- **In hand:**
  - Level vs similar scenes against level vs the average clean image: +0.041 / +0.037 on the 4,800 held-out images, both intervals above 0 (results).
  - **The gain is largest at severity 1** (`ladder_by_severity.csv`; all images, paired 1,000-draw intervals):
    - common: +0.061 [+0.058, +0.063] at severity 1, then +0.050, +0.041, +0.029, and +0.023 at severity 5;
    - extra: +0.059 [+0.056, +0.062] at severity 1, falling to +0.019 at severity 5;
    - every interval is above 0.
  - **Neighbours against a linear map** (`linear_vs_neighbours.csv`):
    - the ridge map is the better content model: 67–81% of the clean variance, against 36–48% for the neighbours' mean;
    - but from the image's own key it scores only 0.783 / 0.798, below even the global level (0.800 / 0.830);
    - its prediction follows the corruption. At severity-1 fog it moves −0.24 bank SD at stage 1, of the image's −0.30, while the neighbours' mean moves −0.06. Under contrast: −0.22 of −0.28, against −0.06. Under zoom blur: −0.24 of −0.20, against −0.11;
    - from the clean twin's key, the same map reaches 0.976 / 0.950. So the content model is not what fails; the drift of the key is.
  - **k** (`ablation_k.csv`; two-axis, common / extra):
    - 0.851 / 0.801 at k = 1, 0.902 / 0.849 at k = 5 and 0.911 / 0.856 at k = 10;
    - 0.915–0.917 / 0.857–0.859 from k = 20 to 100;
    - 0.914 / 0.853 at k = 200 and 0.909 / 0.844 at k = 500.
  - **Bank size** (`ablation_bank.csv`; k = 50, three seeded subsets per size):
    - 100 images: 0.901–0.906 / 0.830–0.841;
    - 250: 0.907–0.911 / 0.845–0.847;
    - 500: 0.912–0.913 / 0.848–0.853;
    - 1,000: 0.914–0.917 / 0.856–0.857;
    - 2,000: 0.917 / 0.858. Going from 1,000 to 2,000 images gains at most 0.003.
  - **The ceiling** (`twin_key.csv`): with the clean twin's neighbours the score reaches 0.936 / 0.877. So the neighbours of the corrupted key get within 0.02 of a drift-free reference.
- **To make:** the 5,000-image bank. **GPU.** The bank-size curve flattens, so it will likely add little.
- **Status:** supported.

### C5. Fog flattens the channels; noise re-weights them

This is the core analysis behind the two axes.

- **Why:**
  - A loss of contrast or of sharpness weakens the edges, where a channel's strongest responses sit. The channel's average also counts the many weak responses, so the peaks fall more than the average, and the channel gets *flatter*.
  - One reading of why the average holds up: a convolution scales with the image's contrast, but the batch norm's fixed offsets do not. This last step is still a hypothesis.
  - Noise adds fine structure everywhere. Some channels fire more and others less, so the *levels* move, in both directions.
- **In hand, on all 5,000 images** (`flatten_or_shift.csv`, `arms_by_family.csv`, `second_axis_statistic.csv`; Figure 1):
  - **Fog lowers the peaks, not the level.** At severity 1, the summed level of the stage-1 channels falls 5% (×0.946), while their summed top-1% mean falls 23% (×0.766).
    - Per channel, the log of the top falls 2.6 times as much as the log level: −0.276 against −0.106.
    - Contrast (×0.950 / ×0.762) and the blurs (×0.97–0.99 / ×0.75–0.85) do the same.
  - **Most channels flatten.** The share of stage-1 channels that get flatter is:
    - 81–84% under fog, contrast, defocus, glass and zoom blur;
    - 75–80% under motion and Gaussian blur, and 70–78% under JPEG, pixelate and elastic;
    - against 48–52% under the four noises.
  - **They flatten together.** Under fog, contrast and zoom blur, 45–51% of an image's peak-share change lies along the direction in which all channels move together. That direction holds 13.8% of the clean variation at stage 1, and 2–3% of the change under the noises.
    - So a signed average over channels adds the flattening up, while whitening divides it away: whitening the peak share costs 0.034 on held-out images (roundtable, B).
    - When every channel shifts the same way, a one-direction test beats a whitened distance (O'Brien 1984).
  - **The flattening is an early-stage effect.** Under fog, 84 / 75 / 62 / 48% of the channels flatten at stages 1–4, and the peak share moves −0.83 / −0.42 / −0.15 / +0.02 bank SD.
  - **Noise re-weights the channels:**
    - at severity 1, Gaussian noise moves the stage-1 levels by 1.27 bank SD per channel, the most of any family (fog: 0.59);
    - but in both directions: 36% of the channels rise by more than 10% and 43% fall by more than 10%, while the summed level stays at ×0.99. Shot, impulse and speckle noise behave the same way;
    - the channels get slightly peakier (+0.17 bank SD), so the flattening arm points the wrong way (0.20 at severity 1), while the level arm catches the noise (0.96).
  - **Each axis needs the other.** At severity 1, fog is 0.51 with the level alone, 0.81 with the peak share's |deviation|, 0.97 with the flattening arm alone and 0.95 with the two-axis score. Averaged over severities, the level beats the peak share on 9 of the 19 families (C6).
  - **The peak share is a log ratio on purpose:**
    - the raw ratio costs 0.018 / 0.013 (two-axis 0.899 / 0.845);
    - the top-1% mean alone, with or without the log, costs 0.018–0.019 on common and 0.035 at severity 1, though it is 0.002–0.003 better on extra;
    - as a |deviation| row, the top-1% mean is below the plain level: 0.825 / 0.817 against 0.841 / 0.866;
    - so the scale-free ratio carries what the level does not.
  - **The peak share still depends on content:** the key predicts 61–67% of its clean variance (67–81% for the level), so it is judged against similar scenes too.
- **In hand, from pure operations** (200 images, `dose_response.csv`, `figures/dose_response.pdf`):
  - **Every loss of contrast flattens.** A gain (x·g), a contrast factor towards the mean grey and a white veil each flatten 76–87% of the stage-1 channels. The flattening arm carries the detection. The level arm stays near chance (0.45–0.58) for gains and contrasts down to ×0.4, and reaches only 0.69 for the strongest veil. Two-axis AUROC:
    - gain ×0.75 / ×0.5 / ×0.3: 0.67 / 0.85 / 0.95;
    - contrast ×0.75 / ×0.5 / ×0.3: 0.76 / 0.94 / 0.99;
    - veil 0.1 / 0.3 / 0.6: 0.62 / 0.82 / 0.94.
  - **Blur flattens too.** σ = 0.5 / 1 / 2 px gives 0.63 / 0.89 / 0.98. The stage-1 top falls 5 / 16 / 27%, while the summed level stays (×0.995–0.999).
  - **Noise re-weights.** σ = 0.02 / 0.04 / 0.08 / 0.12 gives 0.47 / 0.63 / 0.94 / 0.99, all from the level arm (0.56 / 0.79 / 0.98 / 1.00). The share of channels that rise or fall by more than 10% grows from 11% / 20% to 41% / 44%.
  - **Fog is a loss of contrast.** The benchmark's fog is clip(g·x + g·c·P), with a gain g = 0.40 at severity 1 and a plasma veil P. Its parts, at severity 1:
    - the gain alone: 0.906;
    - the gain with a uniform veil of the same mean: 0.979;
    - the full fog: 0.974;
    - the veil's spatial structure alone, with zero mean and no gain: 0.498.

    So the early channels see the benchmark's fog as a uniform loss of contrast, and the plasma's shape adds nothing.
  - **The flattening arm is one-sided on purpose, and that has a cost.** *Raising* the contrast or the gain makes the channels peakier (flattening arm 0.08–0.40). Only the level arm sees it, and weakly (contrast ×1.5: 0.63; gain ×2: 0.76). The two-axis score even falls below chance (contrast ×1.5: 0.45). No benchmark family raises contrast, but sharpening or a gain boost would.
- **Status:** measured on all images. The pure operations support the mechanism: losses of contrast and of sharpness flatten, noise re-weights, and the veil's shape does nothing. Why the peaks fall more than the average (the batch norm's fixed offsets) is still a hypothesis.

### C6. One axis or the other, so the score takes the larger arm

- **Why:** the two arms answer different questions, and a corruption moves mostly along one of them (C5).
- **In hand** (all images; `arms_by_family.csv`, `max_vs_sum.csv`):
  - **Which arm is the larger, image by image:**
    - the flattening arm on 98–99% of fogged images, 96–99% under contrast, 88–96% under defocus and zoom blur, 85–90% under glass blur and 82–95% under Gaussian blur;
    - the level arm on 94–100% of the images under the four noises, and on 92–100% under snow;
    - frost leans to the flattening arm (68–85%). Motion blur, elastic, pixelate, JPEG, brightness, saturate and spatter are mixed, and change with severity;
    - on clean images the flattening arm is the larger on 53%, so the two standardised arms are balanced.
  - **Each arm alone:** flattening 0.698 / 0.570, level 0.841 / 0.866. Averaged over severities, the level beats the peak share on 9 of the 19 families, and the flattening arm beats the level on 10.
  - **The larger arm against the sum** (paired 1,000-draw intervals):
    - all images: common +0.0001 [−0.0020, +0.0021], extra +0.025 [+0.023, +0.027];
    - untouched images: +0.0007 [−0.0018, +0.0032] and +0.026 [+0.023, +0.029];
    - FPR95: 0.226 against 0.237 on common, and 0.360 against 0.407 on extra.
  - So the larger arm equals the sum where both arms help (common), and wins where one arm only adds noise. Under the noises the flattening arm points the wrong way: 0.07–0.15, averaged over severities.
- **Status:** supported on all images.

### C7. The result is not tuned to the test images

- **In hand:**
  - The rules were fixed before the run: `docs/superpowers/plans/2026-10-01-content-conditioned-confirmation.md` and its amendment 2.
  - The 3,030 untouched images give 0.916 / 0.858, against 0.917 / 0.858 on all images.
  - The two-axis score was frozen after the screen and positions 200–1058 had been read. It then gave 0.922 / 0.861 on positions 1059–1969, which were read only afterwards (roundtable).
- **Status:** supported.

### C8. Brightness, saturation and hue barely move the early channels, because they keep the image's structure

- **Why (two candidates, which predict different things):**
  - **(a) Structure:** brightness, saturation and hue change the colours point by point but keep the edges and texture, which is what the early filters respond to.
  - **(b) Training:** changes inside the augmentation range are what the detector learned to ignore. That range is brightness ×0.875–1.125, contrast ×0.5–1.5, saturation ×0.5–1.5 and hue ±0.05, each with p = 0.5, until epoch 71.
- **In hand:**
  - **The features barely move** (`effect_sizes.csv`, `flatten_or_shift.csv`). At severity 1, brightness:
    - shifts the stage-1 levels by 0.16 bank SD per channel (fog: 0.59);
    - changes only 7% of the channels by more than 10%;
    - leaves the summed level and top at ×1.000 and ×1.003.
  - **The reference is not to blame** (`twin_key.csv`). With the clean twin's neighbours, brightness reaches 0.53 at severity 1 (0.64 over all severities), saturate 0.62 and spatter 0.59. Spatter is weak only at severity 1: from severity 2 on it is 0.87–1.00.
  - **The pure operations decide between (a) and (b)** (`dose_response.csv`):
    - **against (b): contrast losses *inside* the training range are detected,** ×0.75 at 0.76 and ×0.5 at 0.94. The early channels did not learn to ignore them;
    - **against (b): hue shifts far *outside* the range stay invisible,** 0.52–0.56 for shifts of 0.1–0.5. The last one turns every colour into its opposite;
    - **for (a): every operation that keeps the luminance structure stays near chance.** That covers hue, saturation between ×0.5 and ×2 (0.50–0.56), and offsets until clipping removes structure. An RGB offset of +0.1 gives 0.56; an HSV value offset of +0.1, the benchmark's brightness at severity 1, gives 0.53;
    - **for (a): a gain's detectability grows smoothly with the loss of contrast,** ×0.875 at 0.58, ×0.75 at 0.67 and ×0.5 at 0.85. There is no step at the edge of the jitter range;
    - saturation becomes visible only at its extremes, which do change the structure. Grey (×0) removes the colour edges (0.69), and ×5 clips (0.80). The benchmark's saturate follows the same curve: hardest at ×2 (0.54), easier at ×0.3 (0.60) and ×20 (0.84).
  - **The first convolution explains the offsets** (`first_conv_filters.csv`; the stem's 32 first filters):
    - a uniform offset of 0.1 moves the median filter's output by 3% of its spread on the image. A random filter's output moves by 32%. Fog's loss of contrast at severity 1 changes that spread by 60%;
    - 21 filters are almost blind to the offset (at most 10% of their spread): 14 luminance filters with more than 80% of their energy along R = G = B, and 7 colour-opponent ones with almost none;
    - the other 11, mostly mixed colour filters, pass the offset at 24–42%;
    - a random filter bank has no luminance filters, and only 3 of its 32 filters are blind to the offset; the median passes it at 32%.
  - Brightness in imagecorruptions adds 0.1–0.5 to the HSV value channel, close to an additive offset with clipping. That is not the training's multiplicative jitter.
- **Status:** supported, for (a), structure. The training's jitter explains neither the detected contrast losses inside its range nor the invisible hue shifts outside it.

### C9. It costs almost nothing

- **Why:** the statistics are means and top-1% means of maps the detector computes anyway, plus a search for the 50 nearest of 2,000 images in 512 dimensions.
- **In hand:** the detector alone takes 5.9 ms per image.
- **To make:** the method's added milliseconds. **GPU.**

### C10. The clean reference must come from the deployment camera

- **Why:** both the content key and the levels depend on the camera and its scenes.
- **In hand** (weak):
  - On 24 Cityscapes frames fogged synthetically from a guessed depth map, a COCO bank brought the content-matched rows down to 0.54–0.80.
  - On the same frames, the stage-1 peak share still fell in 70–74% of channels, by about half the shift seen with the package's fog, and stage 3 did not flatten (roundtable).
- **To make:** the driving experiment with a Cityscapes bank. **GPU, downloads.**
- **Status:** the deciding test for IV.

### C11. The two ideas are new

- **In hand:**
  - The nearest relatives: Lee et al. 2018 (class-conditional, within one layer), NMD (Dong et al., 2022; channel means against training means), Rippel et al. 2020 (whitened Gaussian features), and the activation CDFs.
  - NAP's published form scores 0.302 here.
- **To make:** a targeted search **(Reading)** for:
  - references conditioned on content or class for OOD and corruption detection;
  - one layer conditioning another;
  - peak-to-mean statistics such as the crest factor or Hoyer sparsity in activation monitoring.

### Claims we must not make

The roundtable refuted these, or found them to be artefacts:
- **No "trade-off between level and peakiness on clean images".** Any ratio is correlated with its own denominator: r = −0.74 to −0.88, against −0.82 to −0.88 expected under independence.
- **No "the ratio cancels content".** The key still predicts 61–67% of the peak share's clean variance.
- **No "the reference explains the shift away".** The neighbours move only −0.06 SD while the image moves −0.83 (C1).
- **No "dense channels flatten while sparse ones sharpen" ("opposite moves").** Checked on 2 October (`opposite_moves.csv`), it holds only as a tendency, and mostly at stages 2–3:
  - under fog at severity 1, the densest third of the stage-1 channels flatten on 92% of the images and the peakiest third on 66%;
  - at stages 2 and 3 the peakiest third splits about evenly (51% and 47%);
  - 9 of the 64 stage-1 channels sharpen on most images, but 86 of the 256 stage-3 channels do.

  Say "most channels flatten, the densest ones most reliably".

The evidence tables of 2 October rule out these as well:
- **No "the deep layers ignore corruption".** Stage 4's own score still separates at 0.60 (level) and 0.74 (peak share). Say that it reacts least, and let the stability of the reference carry the argument (C1).
- **No "the key is stable".** Only 63% of the neighbours survive a severity-1 corruption. What is stable is the reference, their mean (C1).
- **No "noise adds responses everywhere" or "noise raises the average".** Under noise, the summed stage-1 level stays at ×0.99, and the channel levels move in both directions (C5).
- **No "the neighbours explain most of the content variation".** They explain 36–48% per channel. The linear map explains 67–81%, but it follows the corruption (C4).
- **No "spatter is invisible".** It is weak only at severity 1 (C8).
- **No "the training's jitter taught the detector to ignore brightness".** Contrast losses inside the jitter range are detected, and hue shifts far outside it are not (C8).
- **No "the veil's pattern is what the detector sees in fog".** The plasma's zero-mean pattern alone scores 0.498; the loss of contrast does the work (C5).

### Evidence plan, by cost

- **CPU, stored: done** (2 October, `docs/results/paper-evidence/`):
  - the depth curve;
  - neighbour stability and category overlap;
  - effect sizes and the ratio of spreads;
  - the twin ceiling;
  - the flatten-or-shift figure;
  - which arm wins, and max against sum;
  - brightness and saturate per stage;
  - the per-severity gain;
  - the k ablation, and bank sizes up to 2,000;
  - added on the way: the linear map against the neighbours, the second axis's statistic, the fixed threshold, and the check of "opposite moves".
- **CPU, forward: done** (200 images, `docs/results/paper-evidence/`):
  - the untrained backbone;
  - the dose-response with pure operations, and fog split into its parts;
  - photometric factors inside and outside the training range;
  - the first convolution's filters (planned as reading).
- **GPU** (needs an agreed window):
  - the driving data;
  - the 5,000-image bank;
  - other top fractions on all images;
  - a second backbone;
  - the runtime.
- **Reading:** the novelty search.

## Section-by-section outline

**Title (open).**
- "Fog Flattens, Noise Shifts: Detecting Image Corruption in a Frozen Object Detector"
- "Fog Flattens, Noise Re-weights: …", which is closer to C5: noise moves the channel levels in both directions and leaves their sum alone. "Shifts" still holds if it means that the levels move.
- "Training-Free Corruption Detection from a Detector's Early Channels"

**Abstract (five sentences).**
1. The problem and the requirements (frozen, clean-only, one score, mild corruption).
2. The two observations: deep layers know the scene but react least to corruption; early channels show it, but mixed with content.
3. The method: compare early-channel level and peak share with the most similar clean scenes; take the larger of the two arms.
4. The result on 5,000 COCO images against six baselines, including severity 1 and the untouched images.
5. Driving results **(to make)** and the limitation: photometric changes.

**1. Introduction**
- Motivation: degraded camera input and silent detection failure.
- The requirements R1–R4.
- Why existing monitors fall short:
  - confidence-based ones react late (SAOD 0.731 / 0.664);
  - feature-statistics ones compare with one global clean average (the activation CDFs, 0.821 / 0.807).
- Our two observations, then the method in one paragraph.
- Contributions, three bullets:
  - content-matched clean reference from the detector's own deep features;
  - the flatten-or-shift two-axis statistic;
  - a pre-registered evaluation with six baselines.

**2. Related work.** Grouped by what the monitor reads:
- **Outputs:** SAOD (Oksuz et al., CVPR 2023), ContrastiveConf (Park et al., TPAMI 2026).
- **Features against clean statistics:**
  - kNN (Sun et al., ICML 2022);
  - Mahalanobis (Lee et al., 2018), Gaussian features (Rippel et al., 2020) and NMD (Dong et al., 2022);
  - the runtime monitor of Hashemi et al. (FM 2023);
  - the activation CDFs (Becker et al., ICPR 2026).
- **Separate shift detectors:** DisCoPatch (Caetano et al., ICCV 2025).
- **Image-quality models (open):** NIQE, ARNIQA, CLIP-IQA.
- **The gap:** none conditions the clean reference on scene content, and none separates flattening from shifting.

**3. Observations that shape the method** (a short "design study" on clean and corrupted COCO images; steps 2–5 above, one figure each):
- separability by stage (`figures/depth_curve.pdf`, Figure 3);
- how much content explains: the key's R², and what the neighbours remove (C3);
- the flatten-or-shift picture: per family at severity 1, the change of the stage-1 levels against the change of the peak share (`figures/flatten_or_shift.pdf`, Figure 1);
- the mechanism, from pure operations (`figures/dose_response.pdf`).

**4. Method**
- Notation: stage s, channel c, map x.
- The level m and the peak share π.
- The content key and the 50 neighbours.
- Per-channel deviations scaled by the bank's spread.
- The two arms, standardised on 500 clean images; the score is the larger arm.
- The fixed choices: k = 50, the stage-4 key, stages 1–3, the top 1%, 2,000 + 500 clean images.
- Cost: one pass of the early stages, already computed by the detector, plus a 50-nearest-neighbour search in a 512-dimensional key **(runtime to make)**.

**5. Experimental protocol**
- Detector: RT-DETRv2-R18, COCO, AP 0.479.
- Corruptions: 15 common + 4 extra families, severities 1–5, one seeded draw per image.
- Metrics: AUROC (main), AUPR and FPR95; paired image bootstrap, 1,000 draws.
- Image sets: all 5,000, the 3,030 untouched, the 4,800 held out.
- Pre-registered rules.
- Baselines as their authors define them, on the same detector.

**6. Results**
- **Table 1:** all rows on all images and on the untouched images.
- **Figure:** AUROC by severity.
- **Table:** families at severities 1 / 3 / 5.
- **Ablations:**
  - the ladder: global level → similar-scene level → peak share → two-axis;
  - k and bank size (`ablation_k.csv`, `ablation_bank.csv`, `figures/ablation.pdf`);
  - the second axis's statistic: log ratio, raw ratio, top-1% mean (`second_axis_statistic.csv`);
  - key choice;
  - top fraction;
  - the larger of the two arms vs their sum.
- **False-alarm rate at a fixed threshold** set on the 500 clean train images (`fixed_threshold.csv`):
  - on the 5,000 clean val images: 0.111 / 0.055 / 0.015 at α = 0.10 / 0.05 / 0.01;
  - at α = 0.05 it flags 46% of the images at severity 1, 71% at severity 3 and 80% at severity 5 (common).
  - Under extra it flags 17% at severity 1 and 83% at severity 5.

**7. Driving data (to make).** Foggy Cityscapes, ACDC and Cityscapes-C with a Cityscapes bank, plus the kill test.

**8. Limitations**
- Brightness, saturation and hue changes keep the spatial structure, and barely move the early channels (claim C8).
- The flattening arm is one-sided: a change that makes the channels peakier, such as more contrast or sharpening, is seen only by the level arm (C5).
- The reference must match the deployment camera.
- One backbone so far (R50 **(to make)**).
- Synthetic corruptions on COCO.

**9. Conclusion.** Restate the two observations and the one-sentence story. The monitor costs almost nothing, because it reuses what the detector already computes.

## Figures and tables

| Item | Status |
|---|---|
| Figure 1: flatten or shift, per family at severity 1 | draft: `docs/results/paper-evidence/figures/flatten_or_shift.pdf` |
| Figure 2: method diagram (frozen detector, key → 50 neighbours, early channels → two arms → max) | to draw |
| Figure 3: separability by stage | draft: `figures/depth_curve.pdf`, with the untrained backbone |
| Figure 5: pure operations, dose-response | draft: `figures/dose_response.pdf` |
| Figure 4: AUROC by severity, ours vs baselines | data in `docs/results/coco/summary.json` |
| Table 1: main results, all and untouched images | data in `docs/results/coco/` |
| Table 2: ablation ladder | data in hand, with k and bank size (`figures/ablation.pdf`) |
| Table 3: families at severities 1 / 3 / 5 | data in hand |
| Table 4: driving data | to make |
| Runtime (ms per image) | to make |

## Writing rules for a natural read

- **Observation first.** Introduce every design choice after the observation that forces it: "because X, we do Y". Never write "we propose Y" without the X.
- **One running example:** fog at severity 1. The level misses it (0.51), the peak share catches it, and the reader can picture why.
- **Plain meanings.** Each quantity gets a one-line plain meaning when it first appears:
  - level: how strongly a channel fires on average;
  - peak share: how much of that firing sits in its strongest spots.
- **Show the ladder.** Present the method as two improvements on an established idea, with a number at each rung. Readers trust increments they can see.
- **Make the invariance do double duty.** The deep layers' invariance is first the reason to look early, then the source of the content key.
  - The limitation does not close that loop. C8's tests point to the image's structure, not to the training's invariance, so state the blind spot as what the early filters can see.
- **No "why" without evidence.** Every "because" in the paper points to a number, a figure or a citation. The claims section above lists them, and says which are still hypotheses.
- **Only the final method is ours.** A ditched approach appears only where it pointed the way, in one sentence: the decoder-query attempt (step 2) and the sorted-response pilot (step 3). No section, figure or table row is given to them.
- **Ablations are of the final method only:**
  - its parts: the global vs the similar-scene reference, each arm alone, the larger of the two arms vs their sum;
  - its settings: k, bank size, key and top fraction.
- **The pilot's leftovers stay out of the paper:** its control rows (kNN on channel means, means against their own average) can stay in our own report.
- **Keep severity 1 in every results view.** That is where the method differs most, and where it matters most.
- **Use pre-registration once, briefly.** State what was fixed when, and show the untouched images. Don't dwell on it.

## Likely reviewer questions, and where the paper answers them

| Question | Answer |
|---|---|
| "COCO is not driving." | Section 7, driving data (to make) |
| "Synthetic corruptions only." | Foggy Cityscapes and ACDC (to make) |
| "Isn't this just NMD or Mahalanobis on channel means?" | The ladder: content matching adds +0.04, the second axis +0.08 on common |
| "Tuned on the test set?" | Pre-registered rules; the untouched 3,030 images give the same result |
| "Sensitive to k or the bank?" | k ablation (flat on the screen); bank-size ablation (to make) |
| "Why not an image-quality model?" | Image-quality baselines (open) |
| "What does it cost?" | Runtime next to the detector's 5.9 ms (to make) |
| "Why does it fail on brightness?" | Section 8 and claim C8. Brightness, saturation and hue keep the luminance structure. Pure operations show that contrast losses inside the training's jitter range are detected, and hue shifts outside it are not, so the jitter is not the reason |
| "Why would deep layers ignore corruption but keep the scene?" | Claim C1: the depth curve (stage 4 reacts least), category overlap, the stability of the reference, and an untrained backbone as the control |
| "Why do the peaks fall more than the mean?" | Claim C5. Pure contrast, gain, veil and blur operations flatten, and noise re-weights. The benchmark's fog acts as a loss of contrast: its veil's shape does nothing |
| "Why neighbours, not a regression from the key?" | Claim C4: the regression is the better content model, but it follows the corrupted key (0.783 / 0.798) |
| "Why the larger of the two arms?" | Claim C6: which arm wins per family, and max vs sum |

## Open questions for the authors

1. **Driving data.** Should the paper include Foggy Cityscapes and ACDC? They need Cityscapes downloads and a GPU window. For IV I would say yes. Also: with the COCO checkpoint, or with a detector fine-tuned on Cityscapes, as the decision record planned?
2. **Image-quality baselines.** The decision record listed NIQE, ARNIQA and CLIP-IQA or QualiCLIP. Run them for the paper, on COCO, on driving data, or not at all?
3. **Format and title.** The 6-page IV format? And the "Fog flattens, noise shifts" framing, or a plainer title?

Settled: ditched methods stay out, except a one-sentence pointer where one set the direction (user, 2 October; see Scope at the top).
