# Four image-quality baselines against the two-axis score on COCO-C

Run on 5 October 2026, following `docs/superpowers/plans/2026-10-05-iqa-baselines-coco-c.md`. Four detector-free image-quality (IQA) baselines, in five rows, were scored once on the COCO-C images of the four-detector study (`docs/coco-detectors-results.md`). Each detector's report was then rebuilt with the five rows added. The tables are in `docs/results/coco-iqa/`.

**Both ARNIQA rows are ahead of the two-axis score on all four detectors.** Their AUROC is higher on the common and on the extra families, on all images and on the untouched ones, and every one of these 32 intervals excludes 0. On AUROC, NIQE with its refitted model is ahead in one place, RF-DETR-M's common families, and NIQE with the published model nowhere. The two NIQE models differ in their clean images: JPEG-compressed COCO train photos for the refit, 125 pristine photos for the published model. The two-axis score is ahead of CLIP-IQA and of the published NIQE everywhere on AUROC.

| Detector | Two-axis − ARNIQA quality, common | Two-axis − ARNIQA quality, extra | Two-axis − ARNIQA prototype, common | Two-axis − ARNIQA prototype, extra |
|---|---|---|---|---|
| RT-DETRv2-R18 | −0.010 [−0.013, −0.007] | −0.047 [−0.050, −0.043] | −0.029 [−0.032, −0.026] | −0.021 [−0.024, −0.019] |
| YOLO11m | −0.037 [−0.040, −0.034] | −0.035 [−0.038, −0.032] | −0.056 [−0.059, −0.053] | −0.010 [−0.013, −0.008] |
| Faster R-CNN R50-FPN v2 | −0.064 [−0.068, −0.060] | −0.053 [−0.056, −0.050] | −0.083 [−0.086, −0.079] | −0.028 [−0.031, −0.025] |
| RF-DETR-M | −0.083 [−0.086, −0.080] | −0.082 [−0.085, −0.078] | −0.102 [−0.105, −0.099] | −0.057 [−0.060, −0.054] |

All images; a negative Δ means the ARNIQA row is ahead. The untouched images give the same signs, every interval below 0 (the full table is under "Headline" below).

**What was scored:**
- **Images:** all 5,000 COCO val2017 images, in the seed-44 order, in all 96 conditions: the clean image and 19 families at 5 severities. Before scoring an image, the pass checked its 96 versions against `runs/coco/`'s digests, so the IQA models saw exactly the images the detectors saw.
- **The five rows,** each oriented so that higher means more likely degraded:
  - **NIQE (refit):** NIQE (Mittal, Soundararajan & Bovik, IEEE SPL 2013) with its pristine model refitted on clean COCO train images. This is the main NIQE row.
  - **NIQE (published, sensitivity):** NIQE with the authors' published pristine model.
  - **ARNIQA quality:** ARNIQA (Agnolucci, Galteri, Bertini & Del Bimbo, WACV 2024) with its KADID-10k regressor; score = −quality.
  - **ARNIQA prototype:** ARNIQA's embedding against the mean embedding of clean COCO train images (Becker, Weiss, Hübner & Arens, arXiv 2602.18394); score = 1 − cosine similarity.
  - **CLIP-IQA:** zero-shot CLIP-IQA (Wang, Chan & Loy, AAAI 2023) with "Good photo." / "Bad photo."; score = −quality. This is not CLIP-IQA+.
- **One set of scores for all four detectors.** The rows read the image, not the detector, so each row's numbers are the same in all four reports. Only the two-axis score, and so the difference, changes with the detector.
- **The reports:** the reports of RT-DETRv2-R18, YOLO11m, Faster R-CNN R50-FPN v2 and RF-DETR-M, rebuilt with the five rows added. They give every metric and interval the detectors' own reports give, with the same paired bootstrap. All other numbers are unchanged (Checks).

## How to read the tables

↑ means higher is better and ↓ means lower is better.

| Column | What it measures | Reference values |
|---|---|---|
| AUROC common ↑ | How well the score tells a corrupted image from its clean version, averaged over the 75 conditions of the 15 common families | 0.5 = chance, 1 = perfect |
| AUROC extra ↑ | The same, over the 20 conditions of the 4 extra families | 0.5 = chance, 1 = perfect |
| FPR95 ↓ | The share of clean images flagged when the threshold catches 95% of the corrupted ones | 0 = perfect |

- Brackets are 95% paired bootstrap intervals over images (1,000 draws, seed 44). In "two-axis − row", a positive ΔAUROC means the two-axis score is better and a negative one that the row is better. For ΔFPR95 it is the reverse, since a lower FPR95 is better.
- The untouched images are the 3,030 at positions 1970 and later. Nobody read them while the two-axis method was chosen. The IQA rows' settings come from their papers and official code, and were fixed before the pass.
- An IQA row's numbers are the same for every detector, so the tables give them once.
- The tables by severity and by family have no intervals.

## In short

- **Both ARNIQA rows are ahead on every detector, with every interval below 0** (table above). ARNIQA quality reaches 0.927 [0.925, 0.929] on the common families and 0.905 [0.903, 0.907] on the extra ones; the prototype 0.946 [0.944, 0.947] and 0.880 [0.878, 0.882].
- **The margin grows as the two-axis score weakens.** The IQA rows' AUROC is fixed, while the two-axis score's falls from 0.917 [0.914, 0.920] on RT-DETRv2-R18 to 0.844 [0.841, 0.847] on RF-DETR-M (common families). Against ARNIQA quality, the gap on the common families runs from −0.010 on RT-DETRv2-R18 to −0.083 on RF-DETR-M.
- **On AUROC, NIQE (refit) is ahead only on RF-DETR-M's common families:** two-axis − NIQE (refit) is −0.010 [−0.013, −0.007] on all images and −0.010 [−0.014, −0.007] on the untouched ones. Everywhere else the two-axis score is ahead of it, every interval above 0. The smallest margin is on Faster R-CNN's common families: +0.009 [+0.005, +0.013], and +0.008 [+0.003, +0.013] on the untouched images. The published NIQE is ahead nowhere on AUROC. The two NIQE models differ in their clean images: JPEG-compressed COCO photos for the refit, 125 pristine photos for the published model.
- **On AUROC, the two-axis score is ahead of CLIP-IQA and of the published NIQE in every cell,** by +0.043 to +0.199 and +0.027 to +0.118 on all images, every interval above 0. On FPR95, two NIQE cells go the other way (the FPR95 bullets under "Headline").
- **Where ARNIQA leads:** averaged over the four detectors, most on saturate and brightness. There the two-axis score gives only 0.50–0.60 and 0.50–0.54 at severity 1. On RF-DETR-M, JPEG and pixelate are also among ARNIQA's largest leads. It leads on frost and fog on every detector, and by wide margins on blur, JPEG and pixelate where a detector's two-axis score is weak on them. **Where the two-axis score leads ARNIQA:**
  - spatter, on every detector;
  - elastic transform, against the quality row on all four detectors and against the prototype on three;
  - JPEG, against the quality row on three detectors;
  - gaussian and impulse noise, against the quality row on two detectors, by less than 0.005.
- **By severity:** on the common families both ARNIQA rows are above the two-axis score at every severity on every detector, by the most at severity 1. On the extra families the two-axis score passes ARNIQA quality at high severity on three detectors and the prototype on all four. RF-DETR-M's two-axis score stays below ARNIQA quality at every severity.
- **What the models saw in training is context, not a measured cause.** ARNIQA's encoder was trained on synthetic distortions (blur, noise, compression, brightness, colour, contrast and pixelation, but no weather). Its KADID-10k regressor was fitted on ratings of distortion types that include 11 of the 19 COCO-C families, some only approximately. The two-axis score and NIQE are fitted on clean images only, and CLIP-IQA saw no distortions. ARNIQA also leads on frost and fog, which are weather and so not among its training distortions.
- **Cost:** NIQE takes 20.72 ms per image, ARNIQA 5.03 ms and CLIP-IQA 4.82 ms, against the detector's 5.89 ms.

## What each baseline is

### NIQE

- **The features are pyiqa 0.1.16's:**
  - the luma (MATLAB's Y) in [0, 255], rounded, in float64;
  - 96 × 96 blocks, with partial blocks dropped;
  - 18 features at full size and 18 at half size (imresize with antialiasing).
- **Both NIQE rows use the same test-image features.** Only the pristine model differs. With the published model, the score equals pyiqa's `niqe` metric on the test's COCO image, to a relative 1e-6 (`test_niqe_matches_pyiqa_with_the_published_model`).
- **Why a refit:** the paper session's request (`docs/superpowers/specs/2026-10-05-iqa-baselines-request.md`) asked for the same clean reference as the other COCO baselines, all 118,287 COCO train images, for both the NIQE refit and the ARNIQA prototype. The published model stays as a sensitivity row.
- **The refit follows the authors' `estimatemodelparam.m`:**
  - per image, keep the blocks sharper than 0.75 × the image's sharpest block (sharpness = the block's mean local standard deviation at full size);
  - over all kept blocks of all images, take the mean and the covariance (N − 1), leaving out blocks that hold a NaN.
- **The refit's counts** (`fit.json`): all 118,287 clean COCO train images were read. 2 of them have no whole 96 × 96 block, so they were left out of the NIQE refit only, which used 118,285 images. It kept 610,748 blocks and dropped none for a NaN.
- **The two models' clean images differ.** The refit's are COCO train photos, which are JPEG-compressed. The published model's are 125 pristine photos. The section on the two NIQE rows shows where their results differ.
- **No version went unscored.** A version keeps pyiqa's own NIQE score as long as one of its blocks is free of NaN (`NIQE_MIN_BLOCKS` = 1):
  - pyiqa leaves the NaN values out of the version's mean and the blocks that hold one out of its covariance;
  - with one NaN-free block, pyiqa's covariance is zero and its score finite;
  - only a version with no NaN-free block would get `NIQE_UNSCORABLE` = 1e6, above every real distance, so that it would count as degraded and the pass would not stop;
  - so the rule replaces only scores that pyiqa cannot give.

  `niqe-blocks.csv` gives 0 of the 480,000 versions (5,000 images × 96 conditions, the clean ones included) without a score. So the 1e6 rule never applied, and `conditions.csv`'s mean NIQE includes no such score: across the 96 conditions it runs from 2.467 to 11.087 for the refit and from 3.204 to 21.970 for the published model.
- **7,140 versions had at least one block that holds a NaN** (`niqe-blocks.csv`); each still had a NaN-free block. By condition, at severities 1–5:
  - snow: 265 / 449 / 448 / 550 / 791;
  - JPEG: 98 / 166 / 206 / 291 / 449;
  - frost: 270 / 174 / 153 / 109 / 86;
  - contrast: 46 / 49 / 59 / 96 / 241;
  - brightness: 65 / 70 / 79 / 84 / 93;
  - the clean images: 40;
  - gaussian noise, impulse noise and fog: none;
  - every other condition: 0–59.
- **A floor of 1 rather than 2 changed no score.** The plan first set it at 2 NaN-free blocks. No version had fewer than 2, so either floor scores every version the same. This minimum comes from the per-image scores in `runs/coco-iqa/scores/iqa/`, which are not in git.

### ARNIQA

- **Run as its paper evaluates it** (miccunifi/ARNIQA, `test.py`, with `data/dataset_base_iqa.py` and `utils/utils_data.py`):
  - the half-size image is PIL's `img.resize((W // 2, H // 2))`, whose default filter for RGB is bicubic;
  - from the image and from its half-size version, the centre crop and the four corner crops, 224 × 224, padded with zeros where the image is smaller than 224 px (at half size, every COCO image whose short side is under 448 px);
  - ImageNet normalisation, and the encoder under autocast (fp16) on the GPU.
- **The embedding** is the official `return_embedding` of each crop: the ResNet-50 features of the crop and of its half-size counterpart, each L2-normalised and concatenated, 4,096 values. An image's embedding is the mean over its five crops.
- **Quality** is the KADID-10k regressor on that embedding, scaled from KADID-10k's rating range to about [0, 1] as pyiqa scales it. The regressor is linear, so the quality of the mean embedding equals the mean of the five crops' qualities, as `test.py` averages them.
- **The prototype** is the mean embedding of all 118,287 clean train images, from the same embedding as the quality row. Becker et al. z-score their final scores; that leaves AUROC unchanged, so it is left out. They do not say how they crop or resize the images for ARNIQA.
- **Native size:** only the half-size version is resized; the crops keep the image's own scale. ARNIQA thus sees each image through five 224 × 224 windows at each scale, not whole. On a 640 × 480 image the full-size crops cover about 80% of it, the half-size crops all of it.
- **How this differs from the other recipes:**
  - The README's torch.hub example scores the whole image, with a bilinear half-size image (torchvision's `transforms.Resize`). It says "for simplicity … In the paper, we average the scores of the center and four corners crops".
  - `single_image_inference.py` takes the five crops, but of a bilinear half-size image.
  - pyiqa's own ARNIQA scores the whole image, with a half-size image made without antialiasing. Only its weights, its regressor and its scaling are used here.
  - The choice matters. While the plan was written, on one COCO image's 96 versions, bilinear against bicubic moved the KADID quality by up to 0.215 (Spearman 0.956). That check is recorded in the plan, not in the result files.
- **Pinned by a test** (`test_arniqa_follows_the_papers_five_crops_and_regresses_as_pyiqa`): the embedding equals the official pipeline written out step by step, and the quality on pyiqa's own features equals pyiqa's `arniqa-kadid` metric.

### CLIP-IQA

- **It follows the paper (Sec. 2.1–2.2) and the official zero-shot model** (IceClear/CLIP-IQA, branch `v2-3.8`, `CLIPIQAFixed` from `configs/clipiqa/clipiqa_attribute_test.py`):
  - a ResNet-50 CLIP, with the positional embedding removed, so the image keeps its own size;
  - one antonym pair, "Good photo." / "Bad photo.";
  - the softmax over the pair of CLIP's logit scale × the cosine similarity, and the probability of "Good photo.".
- **pyiqa's wrapper differs in two ways:**
  - it averages five prompt pairs, three of them about blur and noise ("Sharp image" / "blurry image", "sharp edges" / "blurry edges", "Noise-free image" / "noisy image");
  - it loads CLIP on the CPU in fp32.

  So the baseline reimplements `CLIPIQAFixed.forward` on pyiqa's CLIP RN50, the same modified CLIP.
- **Weights:** fp16 on the GPU, as the official `build_model` converts them, and this run used them. On the CPU, which only the tests use, the weights are fp32, and the score equals pyiqa's port given the paper's single pair (`test_clipiqa_uses_the_papers_prompt_pair`).
- **Logit scale:** the paper's Eq. 3 writes the softmax over the raw cosines; the code multiplies them by CLIP's logit scale first. The baseline follows the code. AUROC is the same either way.

### The GPU's precision against fp32

On the GPU, ARNIQA's encoder runs under autocast and CLIP-IQA with fp16 weights, as their official code runs them. The plan's precision check (Task 6) compared two sets of scores, all 96 versions of the first three evaluation images, 288 values per row (`precision-check.txt`):
- the GPU scores stored by the pass;
- the same versions scored again on the CPU in fp32.

| Row | Largest absolute difference, GPU against CPU | Kendall's τ | Tied values of 288, GPU / CPU |
|---|---|---|---|
| NIQE (refit) | 8.3e-03 | 0.9999 | 3 / 3 |
| NIQE (published, sensitivity) | 3.5e-02 | 0.9997 | 3 / 3 |
| ARNIQA quality | 9.1e-04 | 0.9992 | 3 / 3 |
| ARNIQA prototype | 9.7e-04 | 0.9993 | 3 / 3 |
| CLIP-IQA | 6.7e-03 | 0.9977 | 61 / 3 |

- **The ranks barely move:** Kendall's τ is 0.9977 or more for every row.
- **The 3 ties on both devices are identical versions.** In one of the three images, `000000040083.jpg`, the clean version and saturate at severities 1–3 are the same image: their digests match. The digests are stored with the per-image scores in `runs/coco-iqa/`, which is not in git.
- **CLIP-IQA's fp16 scores tie more often:** 61 of 288 values, against 3 in fp32. How much this costs its AUROC was not measured.
- NIQE's features are float64 on both devices. Its two rows differ only in the pristine model: JPEG-compressed COCO photos for the refit, 125 pristine photos for the published model. Where their small differences come from was not traced.

## What each model saw in training

This is context for the comparison, for the paper's fairness statement. None of it was measured here.

- **ARNIQA's encoder (both ARNIQA rows)** was trained on synthetic distortions, including blur, noise, compression, brightness, colour, contrast and spatial distortions such as pixelation (ARNIQA's `utils/utils_data.py` and `utils/distortions.py`), but no weather.
- **ARNIQA's KADID-10k regressor (the quality row only)** was fitted on human ratings of KADID-10k's 25 synthetic distortion types. These COCO-C families appear among them, some only approximately: blur (gaussian, motion, and lens ≈ defocus), noise (white ≈ gaussian, impulse, and multiplicative ≈ speckle), JPEG, pixelate, contrast, brightness and saturation. That is 11 of the 19 families: 8 of the 15 common ones and 3 of the 4 extra ones.
- **The prototype row** uses ARNIQA's encoder but not the regressor. Its reference, the mean embedding, comes from clean COCO train images.
- **CLIP-IQA** saw no distortions.
- **NIQE** sees only clean images: the refit's 118,285 COCO train photos, which are JPEG-compressed, and the published model's 125 pristine photos.
- **The two-axis score** is fitted on clean COCO train images only: its bank of 2,000 images and its 500 z-statistics images.
  - Its settings were chosen before anyone read the untouched images (positions 1970 and later). On those images its AUROC is within 0.001 of its value on all images, on every detector.
  - The detectors themselves were trained on COCO train images with their own augmentations, which this doc does not list.

**Next to this, the results show (measured; whether the training overlap explains them was not tested):**
- Averaged over the four detectors, ARNIQA leads most on saturate and brightness. On RF-DETR-M, JPEG and pixelate are also among its largest leads. All four families are among the 11 above.
- It also leads on frost and fog on every detector, though its encoder saw no weather.
- The two-axis score leads both ARNIQA rows on spatter on every detector. On elastic transform it leads both on three detectors; on RF-DETR-M it leads the quality row but not the prototype. Neither family is among the 11 above.
- The two-axis score also leads the quality row on JPEG on three detectors, though JPEG is among the 11.

## Headline: all images and the untouched ones

**The IQA rows,** the same in all four detectors' reports:

| Row | All, common ↑ | All, extra ↑ | Untouched, common ↑ | Untouched, extra ↑ | FPR95 all, common ↓ | FPR95 all, extra ↓ |
|---|---|---|---|---|---|---|
| NIQE (refit) | 0.854 [0.852, 0.857] | 0.765 [0.763, 0.767] | 0.854 [0.851, 0.857] | 0.765 [0.763, 0.768] | 0.338 [0.334, 0.342] | 0.479 [0.474, 0.483] |
| NIQE (published, sensitivity) | 0.799 [0.796, 0.801] | 0.796 [0.794, 0.798] | 0.799 [0.796, 0.802] | 0.796 [0.793, 0.799] | 0.393 [0.389, 0.397] | 0.408 [0.404, 0.412] |
| ARNIQA quality | 0.927 [0.925, 0.929] | **0.905** [0.903, 0.907] | 0.927 [0.924, 0.929] | **0.904** [0.901, 0.907] | 0.178 [0.174, 0.183] | **0.242** [0.238, 0.247] |
| ARNIQA prototype | **0.946** [0.944, 0.947] | 0.880 [0.878, 0.882] | **0.946** [0.944, 0.948] | 0.881 [0.878, 0.884] | **0.128** [0.125, 0.131] | 0.251 [0.247, 0.255] |
| CLIP-IQA | 0.801 [0.797, 0.805] | 0.671 [0.667, 0.674] | 0.803 [0.797, 0.809] | 0.672 [0.667, 0.678] | 0.472 [0.465, 0.479] | 0.676 [0.670, 0.682] |

**The two-axis score,** per detector:

| Detector | All, common ↑ | All, extra ↑ | Untouched, common ↑ | Untouched, extra ↑ | FPR95 all, common ↓ | FPR95 all, extra ↓ |
|---|---|---|---|---|---|---|
| RT-DETRv2-R18 | 0.917 [0.914, 0.920] | 0.858 [0.856, 0.861] | 0.916 [0.912, 0.920] | 0.858 [0.854, 0.862] | 0.226 [0.221, 0.233] | 0.360 [0.355, 0.366] |
| YOLO11m | 0.890 [0.887, 0.893] | 0.870 [0.867, 0.872] | 0.891 [0.887, 0.894] | 0.870 [0.867, 0.873] | 0.259 [0.253, 0.264] | 0.315 [0.310, 0.320] |
| Faster R-CNN R50-FPN v2 | 0.863 [0.859, 0.867] | 0.852 [0.849, 0.854] | 0.863 [0.858, 0.867] | 0.851 [0.848, 0.855] | 0.348 [0.342, 0.355] | 0.365 [0.359, 0.370] |
| RF-DETR-M | 0.844 [0.841, 0.847] | 0.823 [0.820, 0.826] | 0.844 [0.840, 0.848] | 0.823 [0.819, 0.827] | 0.359 [0.354, 0.364] | 0.438 [0.431, 0.445] |

Bold is the best value in a column over both tables. The two NIQE rows differ in their pristine model's clean images: COCO's JPEG-compressed train photos for the refit, 125 pristine photos for the published model (see the section on the two NIQE rows).

**The two-axis score minus each row, ΔAUROC:**

| Row | Detector | All, common | All, extra | Untouched, common | Untouched, extra |
|---|---|---|---|---|---|
| NIQE (refit) | RT-DETRv2-R18 | +0.063 [+0.059, +0.066] | +0.093 [+0.090, +0.097] | +0.062 [+0.058, +0.066] | +0.092 [+0.088, +0.097] |
|  | YOLO11m | +0.036 [+0.033, +0.039] | +0.105 [+0.102, +0.107] | +0.036 [+0.033, +0.040] | +0.105 [+0.101, +0.108] |
|  | Faster R-CNN R50-FPN v2 | +0.009 [+0.005, +0.013] | +0.087 [+0.083, +0.090] | +0.008 [+0.003, +0.013] | +0.086 [+0.081, +0.090] |
|  | RF-DETR-M | **−0.010 [−0.013, −0.007]** | +0.058 [+0.055, +0.062] | **−0.010 [−0.014, −0.007]** | +0.057 [+0.054, +0.061] |
| NIQE (published, sensitivity) | RT-DETRv2-R18 | +0.118 [+0.115, +0.122] | +0.063 [+0.059, +0.066] | +0.118 [+0.113, +0.122] | +0.062 [+0.058, +0.066] |
|  | YOLO11m | +0.091 [+0.089, +0.095] | +0.074 [+0.072, +0.076] | +0.092 [+0.088, +0.096] | +0.074 [+0.071, +0.077] |
|  | Faster R-CNN R50-FPN v2 | +0.065 [+0.061, +0.068] | +0.056 [+0.053, +0.059] | +0.064 [+0.059, +0.069] | +0.055 [+0.051, +0.059] |
|  | RF-DETR-M | +0.046 [+0.042, +0.049] | +0.027 [+0.025, +0.031] | +0.045 [+0.041, +0.049] | +0.027 [+0.023, +0.031] |
| ARNIQA quality | RT-DETRv2-R18 | **−0.010 [−0.013, −0.007]** | **−0.047 [−0.050, −0.043]** | **−0.010 [−0.014, −0.006]** | **−0.047 [−0.051, −0.042]** |
|  | YOLO11m | **−0.037 [−0.040, −0.034]** | **−0.035 [−0.038, −0.032]** | **−0.036 [−0.040, −0.032]** | **−0.034 [−0.038, −0.031]** |
|  | Faster R-CNN R50-FPN v2 | **−0.064 [−0.068, −0.060]** | **−0.053 [−0.056, −0.050]** | **−0.064 [−0.069, −0.059]** | **−0.053 [−0.057, −0.049]** |
|  | RF-DETR-M | **−0.083 [−0.086, −0.080]** | **−0.082 [−0.085, −0.078]** | **−0.083 [−0.087, −0.078]** | **−0.082 [−0.086, −0.077]** |
| ARNIQA prototype | RT-DETRv2-R18 | **−0.029 [−0.032, −0.026]** | **−0.021 [−0.024, −0.019]** | **−0.029 [−0.033, −0.026]** | **−0.023 [−0.027, −0.019]** |
|  | YOLO11m | **−0.056 [−0.059, −0.053]** | **−0.010 [−0.013, −0.008]** | **−0.055 [−0.058, −0.052]** | **−0.011 [−0.014, −0.007]** |
|  | Faster R-CNN R50-FPN v2 | **−0.083 [−0.086, −0.079]** | **−0.028 [−0.031, −0.025]** | **−0.083 [−0.088, −0.078]** | **−0.029 [−0.033, −0.025]** |
|  | RF-DETR-M | **−0.102 [−0.105, −0.099]** | **−0.057 [−0.060, −0.054]** | **−0.102 [−0.106, −0.098]** | **−0.058 [−0.062, −0.054]** |
| CLIP-IQA | RT-DETRv2-R18 | +0.116 [+0.111, +0.121] | +0.188 [+0.183, +0.192] | +0.114 [+0.108, +0.120] | +0.186 [+0.180, +0.192] |
|  | YOLO11m | +0.089 [+0.084, +0.094] | +0.199 [+0.195, +0.203] | +0.088 [+0.082, +0.094] | +0.198 [+0.192, +0.203] |
|  | Faster R-CNN R50-FPN v2 | +0.062 [+0.057, +0.067] | +0.181 [+0.177, +0.186] | +0.060 [+0.053, +0.067] | +0.179 [+0.173, +0.185] |
|  | RF-DETR-M | +0.043 [+0.038, +0.048] | +0.153 [+0.148, +0.157] | +0.041 [+0.034, +0.048] | +0.151 [+0.144, +0.157] |

Bold marks the cells where the row is ahead of the two-axis score. Every interval in the table excludes 0. As above, the two NIQE rows differ in their pristine model's clean images: JPEG-compressed COCO photos for the refit, 125 pristine photos for the published model.

**What else the tables show:**
- **The untouched images give nearly the same numbers (within 0.002).** The two-axis score is within 0.001 of its value on all images, on every detector, and so is every IQA row except CLIP-IQA, which is within 0.002.
- **FPR95 agrees with AUROC in all but two cells.**
  - The two ARNIQA rows have the lower FPR95 on every detector. On the common families they give 0.178 [0.174, 0.183] (quality) and 0.128 [0.125, 0.131] (prototype), against the two-axis score's 0.226 [0.221, 0.233] on RT-DETRv2-R18 to 0.359 [0.354, 0.364] on RF-DETR-M.
  - On Faster R-CNN's common families, NIQE (refit) has the lower FPR95, though the lower AUROC: 0.338 [0.334, 0.342] against 0.348 [0.342, 0.355]. The paired difference, two-axis minus NIQE (refit), is +0.011 [+0.004, +0.018].
  - On RF-DETR-M's extra families, the published NIQE has the lower FPR95, though the lower AUROC: 0.408 [0.404, 0.412] against 0.438 [0.431, 0.445]. The paired difference is +0.029 [+0.023, +0.038].
  - These two NIQE rows differ in their pristine model's clean images: JPEG-compressed COCO photos for the refit, 125 pristine photos for the published model.
  - In every other cell, FPR95 favours the same side as AUROC, and its interval excludes 0.
- **AUPR** is in each detector's `report.md`, next to the AUROC and FPR95 of every row.

## AUROC by severity

All images. Common families:

| Row | Sev 1 | Sev 2 | Sev 3 | Sev 4 | Sev 5 |
|---|---|---|---|---|---|
| NIQE (refit) | 0.809 | 0.843 | 0.859 | 0.876 | 0.884 |
| NIQE (published, sensitivity) | 0.751 | 0.786 | 0.800 | 0.823 | 0.833 |
| ARNIQA quality | 0.877 | 0.918 | 0.937 | 0.950 | 0.953 |
| ARNIQA prototype | 0.924 | 0.942 | 0.950 | 0.955 | 0.958 |
| CLIP-IQA | 0.755 | 0.796 | 0.808 | 0.822 | 0.824 |
| Two-axis, RT-DETRv2-R18 | 0.863 | 0.905 | 0.926 | 0.940 | 0.951 |
| Two-axis, YOLO11m | 0.825 | 0.874 | 0.902 | 0.918 | 0.931 |
| Two-axis, Faster R-CNN R50-FPN v2 | 0.794 | 0.838 | 0.872 | 0.895 | 0.916 |
| Two-axis, RF-DETR-M | 0.786 | 0.835 | 0.854 | 0.869 | 0.878 |

Extra families:

| Row | Sev 1 | Sev 2 | Sev 3 | Sev 4 | Sev 5 |
|---|---|---|---|---|---|
| NIQE (refit) | 0.676 | 0.744 | 0.807 | 0.774 | 0.824 |
| NIQE (published, sensitivity) | 0.700 | 0.782 | 0.826 | 0.814 | 0.856 |
| ARNIQA quality | 0.846 | 0.911 | 0.884 | 0.937 | 0.948 |
| ARNIQA prototype | 0.845 | 0.890 | 0.860 | 0.894 | 0.911 |
| CLIP-IQA | 0.629 | 0.627 | 0.679 | 0.691 | 0.727 |
| Two-axis, RT-DETRv2-R18 | 0.705 | 0.849 | 0.859 | 0.924 | 0.955 |
| Two-axis, YOLO11m | 0.704 | 0.875 | 0.866 | 0.938 | 0.966 |
| Two-axis, Faster R-CNN R50-FPN v2 | 0.670 | 0.817 | 0.867 | 0.938 | 0.967 |
| Two-axis, RF-DETR-M | 0.680 | 0.796 | 0.841 | 0.879 | 0.920 |

- **Common families: both ARNIQA rows are above the two-axis score at every severity on every detector.**
  - The gap is widest at severity 1: 0.877 (quality) and 0.924 (prototype), against 0.863 / 0.825 / 0.794 / 0.786 (RT-DETRv2-R18, YOLO11m, Faster R-CNN, RF-DETR-M).
  - It is narrower at severity 5: 0.953 and 0.958, against 0.951 / 0.931 / 0.916 / 0.878.
- **Extra families: the two-axis score passes ARNIQA quality at high severity on three detectors, and the prototype on all four.**
  - Against ARNIQA quality, it is ahead at severity 5 on RT-DETRv2-R18 (0.955 against 0.948). On YOLO11m and Faster R-CNN it is ahead at severities 4 and 5 (0.938 and 0.966, and 0.938 and 0.967, against 0.937 and 0.948).
  - Against the prototype, it is ahead from severity 3 on YOLO11m and Faster R-CNN, and from severity 4 on RT-DETRv2-R18.
  - RF-DETR-M's two-axis score stays below ARNIQA quality at every severity. It passes the prototype only at severity 5 (0.920 against 0.911).
  - At severity 1, both ARNIQA rows lead by the most: 0.846 and 0.845, against 0.705 / 0.704 / 0.670 / 0.680.
- **The two NIQE rows split.** They differ in their pristine model's clean images: JPEG-compressed COCO photos for the refit, 125 pristine photos for the published model.
  - On the common families, NIQE (refit) is above RF-DETR-M's two-axis score at every severity: 0.809 / 0.843 / 0.859 / 0.876 / 0.884, against 0.786 / 0.835 / 0.854 / 0.869 / 0.878. It is also above Faster R-CNN's at severities 1 and 2 (0.809 and 0.843, against 0.794 and 0.838).
  - The published NIQE is below the two-axis score at every severity of the common families, on every detector.
  - On the extra families at severity 1, both NIQE rows are above Faster R-CNN's two-axis score, 0.676 and 0.700 against 0.670. The published one is also above RF-DETR-M's, 0.700 against 0.680.
- **CLIP-IQA is below the two-axis score at every severity, on both family groups and on every detector.**

## Families at severities 1, 3 and 5

AUROC at severities 1 / 3 / 5, all images, from each report's `by_family`. `*` marks the extra families. ARNIQA quality is the strongest IQA row on the extra families and the prototype on the common ones, so the table gives both. They read the image only, so one column each serves all four detectors.

| Family | Two-axis, RT-DETRv2-R18 | Two-axis, YOLO11m | Two-axis, Faster R-CNN | Two-axis, RF-DETR-M | ARNIQA quality | ARNIQA prototype |
|---|---|---|---|---|---|---|
| gaussian noise | 0.93 / 1.00 / 1.00 | 0.97 / 1.00 / 1.00 | 0.99 / 1.00 / 1.00 | 0.94 / 1.00 / 1.00 | 0.97 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| shot noise | 0.91 / 1.00 / 1.00 | 0.97 / 1.00 / 1.00 | 0.98 / 1.00 / 1.00 | 0.93 / 0.99 / 1.00 | 0.98 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| impulse noise | 0.99 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 0.98 / 1.00 / 1.00 | 0.99 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| defocus blur | 0.95 / 0.98 / 0.99 | 0.92 / 0.99 / 1.00 | 0.84 / 0.93 / 0.97 | 0.92 / 0.98 / 0.99 | 0.99 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| glass blur | 0.91 / 0.96 / 0.98 | 0.84 / 0.96 / 0.99 | 0.75 / 0.86 / 0.93 | 0.81 / 0.90 / 0.97 | 0.96 / 0.99 / 1.00 | 1.00 / 1.00 / 1.00 |
| motion blur | 0.87 / 0.95 / 0.97 | 0.84 / 0.96 / 0.99 | 0.80 / 0.93 / 0.97 | 0.76 / 0.92 / 0.96 | 0.95 / 1.00 / 1.00 | 0.99 / 1.00 / 1.00 |
| zoom blur | 0.90 / 0.93 / 0.95 | 0.85 / 0.89 / 0.91 | 0.74 / 0.77 / 0.80 | 0.90 / 0.93 / 0.94 | 0.95 / 0.93 / 0.90 | 0.98 / 0.96 / 0.96 |
| snow | 0.92 / 0.97 / 0.98 | 0.90 / 0.95 / 0.96 | 0.92 / 0.97 / 0.98 | 0.82 / 0.87 / 0.89 | 0.94 / 0.99 / 1.00 | 0.91 / 0.98 / 0.99 |
| frost | 0.72 / 0.89 / 0.92 | 0.58 / 0.76 / 0.80 | 0.61 / 0.78 / 0.82 | 0.70 / 0.90 / 0.92 | 0.93 / 0.95 / 0.94 | 0.88 / 0.93 / 0.93 |
| fog | 0.95 / 0.98 / 0.99 | 0.84 / 0.91 / 0.91 | 0.80 / 0.83 / 0.84 | 0.89 / 0.95 / 0.96 | 0.97 / 0.98 / 0.98 | 0.98 / 0.99 / 0.99 |
| brightness | 0.53 / 0.60 / 0.71 | 0.50 / 0.59 / 0.72 | 0.54 / 0.63 / 0.78 | 0.52 / 0.59 / 0.67 | 0.63 / 0.91 / 0.98 | 0.60 / 0.83 / 0.94 |
| contrast | 0.94 / 0.99 / 1.00 | 0.86 / 0.96 / 1.00 | 0.82 / 0.92 / 0.98 | 0.91 / 0.99 / 1.00 | 0.96 / 0.99 / 1.00 | 0.97 / 1.00 / 1.00 |
| elastic transform | 0.68 / 0.76 / 0.84 | 0.54 / 0.61 / 0.70 | 0.60 / 0.67 / 0.74 | 0.54 / 0.52 / 0.50 | 0.43 / 0.46 / 0.50 | 0.57 / 0.56 / 0.57 |
| pixelate | 0.85 / 0.96 / 0.99 | 0.83 / 0.97 / 1.00 | 0.69 / 0.86 / 0.97 | 0.61 / 0.67 / 0.74 | 0.90 / 0.98 / 1.00 | 0.99 / 1.00 / 1.00 |
| jpeg compression | 0.89 / 0.94 / 0.97 | 0.94 / 0.99 / 1.00 | 0.84 / 0.93 / 0.99 | 0.57 / 0.60 / 0.61 | 0.60 / 0.88 / 1.00 | 0.99 / 1.00 / 1.00 |
| speckle noise * | 0.80 / 0.98 / 1.00 | 0.92 / 0.99 / 1.00 | 0.91 / 0.99 / 1.00 | 0.87 / 0.97 / 0.99 | 0.97 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| gaussian blur * | 0.85 / 0.98 / 0.99 | 0.78 / 0.99 / 1.00 | 0.71 / 0.94 / 0.98 | 0.79 / 0.98 / 1.00 | 0.92 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| spatter * | 0.57 / 0.94 / 1.00 | 0.53 / 0.94 / 1.00 | 0.55 / 0.95 / 1.00 | 0.50 / 0.89 / 0.93 | 0.53 / 0.76 / 0.80 | 0.51 / 0.71 / 0.66 |
| saturate * | 0.60 / 0.54 / 0.84 | 0.59 / 0.54 / 0.87 | 0.50 / 0.58 / 0.89 | 0.56 / 0.52 / 0.76 | 0.96 / 0.78 / 0.99 | 0.88 / 0.73 / 0.98 |

**How often each row is above the two-axis score,** counted over the 95 conditions (19 families × 5 severities), all images. No condition is tied.

| Row | RT-DETRv2-R18 | YOLO11m | Faster R-CNN R50-FPN v2 | RF-DETR-M |
|---|---|---|---|---|
| NIQE (refit) | 34 | 30 | 37 | 39 |
| NIQE (published, sensitivity) | 41 | 34 | 38 | 55 |
| ARNIQA quality | 75 | 75 | 77 | 81 |
| ARNIQA prototype | 84 | 86 | 81 | 90 |
| CLIP-IQA | 18 | 26 | 34 | 35 |

The published NIQE is above the two-axis score in more conditions than the refit on every detector, though its AUROC is lower on the common families. More of its conditions are on the extra families: 7 to 10 of the 20, against 5 for the refit. The two NIQE rows differ in their pristine model's clean images: JPEG-compressed COCO photos for the refit, 125 pristine photos for the published model.

In the rest of this section, a family's lead is the mean, over severities 1–5, of the row's AUROC minus the two-axis score's, from `by_family`. These are point estimates, with no intervals.

**Where the ARNIQA rows lead.** ARNIQA quality has a positive lead on 14 to 17 of the 19 families, depending on the detector, and the prototype on 17 or 18.
- **Saturate and brightness, by the widest margins averaged over the four detectors.**
  - Averaged over the detectors, ARNIQA quality leads by 0.277 on saturate and 0.246 on brightness. The prototype leads by 0.235 and 0.192, then by 0.153 on pixelate.
  - They are both rows' two largest leads on RT-DETRv2-R18 and YOLO11m, and the quality row's on Faster R-CNN.
  - On Faster R-CNN, the prototype's brightness lead (0.158) comes sixth, after saturate (0.232), zoom blur (0.204), frost (0.168), fog (0.165) and pixelate (0.161).
  - On RF-DETR-M, JPEG and pixelate join them. The quality row's four largest leads are saturate (0.320), pixelate (0.283), brightness (0.263) and JPEG (0.256). The prototype's are JPEG (0.400), pixelate (0.322), saturate (0.278) and brightness (0.209).
  - At severity 1 the two-axis score gives only 0.50–0.54 on brightness and 0.50–0.60 on saturate, as `docs/coco-detectors-results.md` noted. ARNIQA quality gives 0.63 and 0.96.
- **Frost and fog, on every detector.**
  - On frost, ARNIQA quality is ahead by 0.088 to 0.221 and the prototype by 0.060 to 0.194.
  - On fog, ARNIQA quality is ahead by 0.003 to 0.154 and the prototype by 0.014 to 0.165. The smallest leads are on RT-DETRv2-R18, whose two-axis score gives 0.95 / 0.98 / 0.99 there: only 0.003 (quality) and 0.014 (prototype).
- **Blur, JPEG and pixelation, by more where the two-axis score is weaker.** The largest of these leads, for ARNIQA quality and for the prototype:
  - on Faster R-CNN, zoom blur (0.167 and 0.204) and glass blur (0.138 and 0.153);
  - on RF-DETR-M, JPEG (0.256 and 0.400) and pixelate (0.283 and 0.322). There RF-DETR-M's two-axis score stays at 0.57–0.61 and 0.61–0.74 at severities 1 / 3 / 5.
- **Noise is close to a tie on average, but not at severity 1.**
  - Averaged over the severities, both ARNIQA rows are within 0.024 of the two-axis score on gaussian, shot and impulse noise, on every detector. The largest gap is 0.0234: the prototype on shot noise, on RT-DETRv2-R18.
  - The average is held down by severities 3–5, where these AUROCs are at or near 1.00.
  - At severity 1 the gaps are wider. On RT-DETRv2-R18's shot noise, the prototype leads by 0.091 (0.999 against 0.908) and the quality row by 0.070 (0.977 against 0.908).

**How much of each group's lead one family carries** (point estimates, no intervals):
- A group's lead is the mean of its families' leads. It equals the headline difference with its sign turned.
- A family's share is its lead divided by the number of families in the group (15 common, 4 extra). So the shares of a group add up to its lead; rounding can leave 0.001.

| Row | Detector | Common: lead | Brightness's share | The other 14 families' share | Extra: lead | Saturate's share | The other 3 families' share |
|---|---|---|---|---|---|---|---|
| ARNIQA quality | RT-DETRv2-R18 | +0.010 | +0.017 | −0.006 | +0.047 | +0.067 | −0.020 |
|  | YOLO11m | +0.037 | +0.017 | +0.020 | +0.035 | +0.062 | −0.026 |
|  | Faster R-CNN R50-FPN v2 | +0.064 | +0.014 | +0.050 | +0.053 | +0.069 | −0.015 |
|  | RF-DETR-M | +0.083 | +0.018 | +0.065 | +0.082 | +0.080 | +0.002 |
| ARNIQA prototype | RT-DETRv2-R18 | +0.029 | +0.013 | +0.016 | +0.021 | +0.056 | −0.035 |
|  | YOLO11m | +0.056 | +0.014 | +0.042 | +0.010 | +0.051 | −0.041 |
|  | Faster R-CNN R50-FPN v2 | +0.083 | +0.011 | +0.072 | +0.028 | +0.058 | −0.030 |
|  | RF-DETR-M | +0.102 | +0.014 | +0.088 | +0.057 | +0.069 | −0.013 |

- **Common families:** on RT-DETRv2-R18, brightness's share of ARNIQA quality's lead (+0.017) is larger than the whole lead (+0.010), and the other 14 families' share is −0.006. In the seven other comparisons the other 14 families' share is positive.
- **Extra families:** saturate's share is larger than the whole lead in seven of the eight comparisons, all but ARNIQA quality on RF-DETR-M. In those seven, the other three families' share is negative, from −0.041 to −0.013.

**Where the two-axis score leads ARNIQA.** These are all the families where an ARNIQA row's lead is negative:
- **Spatter, on every detector, against both rows.**
  - Averaged over the severities, the two-axis score is ahead of ARNIQA quality by 0.080 to 0.168, and of the prototype by 0.162 to 0.250.
  - At severity 5 it gives 1.00 on three detectors and 0.93 on RF-DETR-M, against 0.80 and 0.66.
- **Elastic transform, against the quality row on all four detectors and the prototype on three.**
  - On RT-DETRv2-R18, YOLO11m and Faster R-CNN, it is ahead of the quality row by 0.296, 0.153 and 0.206, and of the prototype by 0.192, 0.049 and 0.103.
  - On RF-DETR-M, whose two-axis score is near chance there (0.54 / 0.52 / 0.50), it is ahead of the quality row by 0.062 and behind the prototype by 0.041.
  - ARNIQA quality itself gives only 0.43 / 0.46 / 0.50 on elastic transform.
- **JPEG, against ARNIQA quality, on three detectors.**
  - It is ahead by 0.086 on RT-DETRv2-R18, 0.128 on YOLO11m and 0.073 on Faster R-CNN. ARNIQA quality starts at 0.60 at severity 1.
  - The prototype leads on JPEG on all four detectors (0.99 / 1.00 / 1.00).
- **Gaussian and impulse noise, against ARNIQA quality on YOLO11m and Faster R-CNN,** by less than 0.005.
- **Ahead of both ARNIQA rows at all five severities:** elastic transform and spatter on RT-DETRv2-R18 and Faster R-CNN, spatter on YOLO11m, and no family on RF-DETR-M.

**NIQE and CLIP-IQA by family:**

| Family | NIQE (refit) | NIQE (published) | CLIP-IQA |
|---|---|---|---|
| gaussian noise | 0.91 / 0.99 / 1.00 | 0.99 / 1.00 / 1.00 | 0.80 / 0.80 / 0.65 |
| shot noise | 0.87 / 0.98 / 1.00 | 0.99 / 1.00 / 1.00 | 0.78 / 0.73 / 0.68 |
| impulse noise | 0.99 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 0.62 / 0.70 / 0.68 |
| defocus blur | 1.00 / 1.00 / 1.00 | 0.97 / 1.00 / 1.00 | 0.90 / 0.95 / 0.98 |
| glass blur | 0.98 / 0.99 / 1.00 | 0.82 / 0.94 / 0.99 | 0.94 / 0.91 / 0.91 |
| motion blur | 0.97 / 0.99 / 0.99 | 0.90 / 0.96 / 0.98 | 0.93 / 0.98 / 0.99 |
| zoom blur | 0.89 / 0.89 / 0.85 | 0.75 / 0.74 / 0.67 | 0.96 / 0.94 / 0.92 |
| snow | 0.88 / 0.94 / 0.98 | 0.85 / 0.92 / 0.98 | 0.69 / 0.86 / 0.92 |
| frost | 0.48 / 0.69 / 0.75 | 0.32 / 0.32 / 0.34 | 0.74 / 0.80 / 0.80 |
| fog | 0.64 / 0.73 / 0.77 | 0.62 / 0.69 / 0.71 | 0.62 / 0.68 / 0.66 |
| brightness | 0.48 / 0.45 / 0.47 | 0.49 / 0.49 / 0.51 | 0.60 / 0.71 / 0.77 |
| contrast | 0.63 / 0.78 / 0.98 | 0.61 / 0.76 / 0.98 | 0.62 / 0.76 / 0.96 |
| elastic transform | 0.56 / 0.52 / 0.50 | 0.30 / 0.31 / 0.37 | 0.58 / 0.51 / 0.49 |
| pixelate | 0.95 / 0.97 / 1.00 | 0.93 / 0.97 / 0.99 | 0.84 / 0.95 / 0.99 |
| jpeg compression | 0.90 / 0.96 / 0.99 | 0.74 / 0.89 / 0.98 | 0.70 / 0.84 / 0.96 |
| speckle noise * | 0.75 / 0.93 / 0.98 | 0.95 / 0.99 / 1.00 | 0.74 / 0.67 / 0.59 |
| gaussian blur * | 0.98 / 1.00 / 1.00 | 0.87 / 1.00 / 1.00 | 0.81 / 0.93 / 0.99 |
| spatter * | 0.50 / 0.85 / 0.88 | 0.50 / 0.86 / 0.92 | 0.51 / 0.53 / 0.63 |
| saturate * | 0.47 / 0.45 / 0.44 | 0.48 / 0.46 / 0.50 | 0.46 / 0.59 / 0.70 |

- **NIQE (refit) on RF-DETR-M** (point estimates, no intervals): its common-family lead comes from JPEG and pixelate.
  - Averaged over the severities, it leads by 0.362 on JPEG and 0.298 on pixelate.
  - In the shares defined above, its lead of +0.010 splits into +0.024 from JPEG, +0.020 from pixelate and −0.034 from the other 13 common families.
- **CLIP-IQA** leads on brightness (by 0.057 to 0.108), motion blur (0.036 to 0.084) and zoom blur (0.017 to 0.179) on every detector. Its other leads, which complete the list:
  - Faster R-CNN: pixelate (0.098), glass blur (0.076), frost (0.034), defocus blur (0.026) and gaussian blur (0.013);
  - RF-DETR-M: pixelate (0.259), JPEG (0.244) and glass blur (0.023);
  - YOLO11m: frost (0.060);
  - RT-DETRv2-R18: pixelate (0.001).
- **The two-axis score leads CLIP-IQA on the four noise families,** by 0.222 to 0.322 on every detector, averaged over the severities. There CLIP-IQA gives 0.59–0.80 at severities 1 / 3 / 5, and less at severity 5 than at severity 1 on three of the four.
- The two NIQE rows differ in their pristine model's clean images: JPEG-compressed COCO photos for the refit, 125 pristine photos for the published model. The next section compares them.

## The two NIQE rows

Both rows score the same test-image features; only the pristine model differs. **The two models' clean images differ: the refit's are 118,285 COCO train photos, which are JPEG-compressed, and the published model's are 125 pristine photos.** Which of these differences drives which change below was not measured.

- **The refit is ahead on the common families and behind on the extra ones.** On all images, it gives 0.854 [0.852, 0.857] against 0.799 [0.796, 0.801] on the common families, and 0.765 [0.763, 0.767] against 0.796 [0.794, 0.798] on the extra ones. The intervals do not overlap; the reports give no interval for the difference.
- **By family** (table above), averaged over the severities, the refit minus the published model:
  - the refit is ahead on frost (+0.330), elastic transform (+0.204), zoom blur (+0.140), JPEG (+0.077) and glass blur (+0.063);
  - it is behind on speckle noise (−0.097);
  - the other families differ by 0.037 or less.
- **The published model is below chance on frost (0.32 / 0.32 / 0.34) and elastic transform (0.30 / 0.31 / 0.37).** More often than not, it puts those versions closer to its pristine model than the clean images. The refit gives 0.48 / 0.69 / 0.75 and 0.56 / 0.52 / 0.50 there.
- **On JPEG the refit is ahead,** 0.90 / 0.96 / 0.99 against 0.74 / 0.89 / 0.98, although its own clean images are JPEG-compressed.
- **The extra families' gap comes mostly from speckle noise:** 0.75 / 0.93 / 0.98 for the refit, against 0.95 / 0.99 / 1.00.
- **Both rows are near chance on brightness and saturate,** at 0.44–0.51 at the severities shown.

## Timing

| Model | Rows it gives | ms per image |
|---|---|---|
| NIQE | NIQE (refit) and NIQE (published), from the same features | 20.72 |
| ARNIQA | ARNIQA quality and prototype, sharing one embedding of the ten crops (the encoder runs once on the five full-size crops and once on the five half-size ones) | 5.03 |
| CLIP-IQA | CLIP-IQA | 4.82 |
| RT-DETRv2-R18, the detector alone | — | 5.89 |

- **Measured as the detector was** (`timing.json`): the median time per image at batch 1, over the first 100 evaluation images after 10 warm-up calls, preprocessing included, on one RTX 5090. Both timing stages call the same loop, `median_ms` in `degradation_monitor/stages/common.py`.
- **The detector's 5.89 ms** comes from RT-DETR's run (`runs/coco/timing.json`), measured the same way with nothing else on the GPU (`docs/coco-baseline-numbers.md`). The card was also otherwise idle during the IQA timing. That is recorded in the run log, not in `timing.json`.
- **NIQE takes 3.5 times the detector's time;** its features are float64 on the GPU. ARNIQA and CLIP-IQA each take a little less than the detector.
- **Each IQA model runs on the image in addition to the detector.** The two-axis score reads the detector's own features; its own added time has not been measured yet (`docs/todo.md`).

## Checks

- **The rebuilt reports change nothing but the five rows.** Every other number in each IQA report's `summary.json` equals the detector's own report: `docs/results/coco/` for RT-DETRv2-R18 and `docs/results/coco-detectors/` for the other three. That covers the headline, the intervals, the tables by severity and by family, the clean mAP and both pre-registered decisions.
  - The decisions are therefore unchanged. The headline is confirmed on RT-DETRv2-R18, YOLO11m and RF-DETR-M. On Faster R-CNN it reads "ahead of the activation CDFs, but the flattening adds nothing over the level score on the common families". The level score is confirmed on all four. The IQA rows enter neither rule.
- **The IQA rows are identical in all four reports:** the headline, the intervals, and the tables by severity and by family.
- **The corrupted images are the detectors'.** The pass compares every image's 96 digests with `runs/coco/`'s before it scores the image, and it wrote a file for each of the 5,000 images. It refuses a score that is not finite.
- **NIQE scored every version** (`niqe-blocks.csv`), and the GPU precision check found Kendall's τ of 0.9977 or more for every row (`precision-check.txt`).
- **Run facts, recorded in no result file:**
  - the plan's own check (Task 8, Step 2) compared the headline and the intervals with the detectors' reports in `runs/`, and found them identical on all four detectors;
  - `runs/coco/` and `runs/coco-detectors/` were listed before the IQA runs, and the listing was the same after them, so the detectors' run folders were left unchanged;
  - the fit took 23 minutes on the GPU, at 86 images/s, with a peak of 0.37 GiB;
  - the smoke pass covered 25 images, with a peak of 1.13 GiB;
  - the full pass ran from 16:24 to 19:09, about 2 h 45 min, at 0.50 images/s, with a peak of 1.20 GiB;
  - the four reports took 1 h 45 min on the CPU;
  - all GPU work ran on one RTX 5090, shared with other sessions, under the config's 8 GiB cap (`configs/coco-iqa.toml`).

## Not yet shown

- **Whether ARNIQA's lead comes from its training distortions.** It leads on frost and fog, which are not among them. The quality row trails on JPEG, which is among them, on three detectors. No experiment separates the training overlap from the rest.
- **Whether an IQA row and the two-axis score combine well.** They lead on different families: ARNIQA most on saturate and brightness, the two-axis score on spatter and elastic transform. No combination was tested.
- **How much CLIP-IQA's fp16 ties cost its AUROC.**
- **Why the published NIQE falls below chance on frost and elastic transform,** and why the refit does better there and on JPEG. The two models differ in their clean images (JPEG-compressed COCO photos against 125 pristine photos), but which difference matters was not measured.
- **The two-axis score's own runtime** next to these models.
- **The IQA rows on real fog, driving data or Cityscapes-C.** A Cityscapes config can reuse every stage, but nothing was run.

## Files

Everything in `docs/results/coco-iqa/`:
- `summary.md` and `summary.csv`: the two-axis score against each IQA row, on all four detectors.
- `rtdetrv2_r18/`, `yolo11m/`, `faster_rcnn_r50_fpn_v2/` and `rfdetr_m/`: each detector's report with the five rows.
  - `report.md`: the generated report.
  - `summary.json`: both decisions, and the AUROC, FPR95 and intervals behind every table and comparison in this doc. The timing, the fit counts, NIQE's block counts and the precision check come from the files listed below, and the mean NIQE from `conditions.csv`.
  - `separation.csv`, `aggregates.csv` and `intervals.csv`: per condition, per group and the bootstrap intervals.
  - `conditions.csv`: each condition's mAP and mean scores.
  - `knn_k.csv`: the kNN baseline by k.
  - `timing.csv`, in `rtdetrv2_r18/` only: the detector's and the baselines' runtimes, as RT-DETR's own report gives them.
- `timing.json`: the IQA models' ms per image, with the detector's.
- `fit.json`: the counts of the NIQE refit and of the ARNIQA prototype.
- `niqe-blocks.csv`: per condition, the versions with a block that holds a NaN and the versions NIQE could not score.
- `precision-check.txt`: the GPU's scores against fp32 on the CPU, on three images.

The per-image scores stay in `runs/coco-iqa/` in the main checkout, which is not in git.
