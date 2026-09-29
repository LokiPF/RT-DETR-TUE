# Driving benchmark: baselines and metrics

*Decision record, 27 September 2026. The sources and reasoning are in
[literature-review-image-corruption-detection.md](literature-review-image-corruption-detection.md).*

COCO numbers: see [coco-baseline-numbers.md](coco-baseline-numbers.md).

## What the monitor must do

The monitor is built into a frozen object detector and needs no training. It warns when the input
image is degraded, and its score should rise as the degradation harms the detector. It is not a
mAP predictor and not a generic image-quality model. See the literature review, Sections 8–9, for
the framing.

## Setting

- **Detector:** RT-DETRv2-R18, fine-tuned from the COCO checkpoint on **Cityscapes train**, using the
  8 usual detection classes (person, rider, car, truck, bus, train, motorcycle, bicycle).
  - Training uses the official [RT-DETR repository](https://github.com/lyuwenyu/RT-DETR), because
    this repo only contains inference code.
  - With 8 classes the fingerprint has 256 + 8 − 1 = 263 values instead of 335.
- **Clean reference:** Cityscapes train images only, with no labels. The detector and the bank share
  the same "normal" conditions: clear daytime.
- **Evaluation data:**

| Set | Clean | Degraded | Paired? |
| --- | --- | --- | --- |
| Cityscapes-C | Cityscapes val | `imagecorruptions` families × severities 1–5 | yes |
| [Foggy Cityscapes](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/) | Cityscapes val | 3 fog densities | yes |
| [ACDC](https://arxiv.org/abs/2104.13395) | normal-condition reference images | real fog, night, rain, snow | roughly |

## Rule for choosing baselines

A baseline must produce **one score per image, as defined by its authors**, and have **public
code or weights**. Methods that score individual queries or objects would need an aggregation we
invent, so they are not baselines. They appear only as ablations inside our pipeline, where the
aggregation stays fixed.

Paper sentence: *"We compare against methods that natively produce a single score per image;
methods that score individual queries are evaluated only within our ablations, where the
aggregation is held fixed."*

## Baselines we keep

| Group | Baseline | Native output | Data needed: corrupted, or clean only? | Code | Notes |
| --- | --- | --- | --- | --- | --- |
| Detector outputs | **SAOD image-level uncertainty** (Oksuz et al., CVPR 2023), 1 − confidence with *min* and *top-3 mean* aggregation | one score per image by definition | **Nothing is fitted.** Its scores are used directly; we report no thresholded metric. | [fiveai/saod](https://github.com/fiveai/saod) | Our old "1 − max confidence" baseline is SAOD's *min* variant, so cite it as SAOD. Entropy only in SAOD's own form, or not at all. |
| Detector outputs | **ContrastiveConf** (Park et al., IEEE TPAMI 2026) | one score per image, designed for DETR | **Clean only, but labelled.** Nothing is trained. The split is the fixed threshold θ = 0.3; λ is cross-fitted on labelled clean evaluation images (5 folds; each fold's λ comes from the other four). | [azizanlab/uq-detr](https://github.com/azizanlab/uq-detr), `pip install uq-detr`, MIT | The paper already uses Cityscapes and Foggy Cityscapes (see "Open items"). |
| Detector features | **kNN** (Sun et al., ICML 2022) on **one pooled image-level feature** from the same frozen detector | one distance per image | **Clean only, no labels.** A bank of clean Cityscapes train images. | [deeplearning-wisc/knn-ood](https://github.com/deeplearning-wisc/knn-ood) | Uses the same clean bank images as ours. Choose and fix the pooled layer before running. |
| Detector features | **Gaussian neuron-interval runtime monitor** (Hashemi, Křetínský, Rieder & Schmidt, FM 2023). *Added 29 September 2026 as the fifth baseline.* | one score per image: the share of monitored neurons outside μ ± kσ, with k = 2. Their conformal p-value is a decreasing function of this score, so it changes none of our threshold-free metrics. | **Clean only, no labels.** μ and σ per neuron come from clean train images, ignoring classes. The calibration set only sets a threshold, which we do not use. | none found; the method is fully described on one page, so we reimplement it | The only published monitor with exactly our assumptions: frozen detector, clean data only, image-level, class-agnostic. It comes from the runtime-verification community. The monitored layer for RT-DETRv2 is still to be fixed (see "Open items"). Paper: arXiv 2212.07773. |
| External image quality | **ARNIQA quality score** (Agnolucci et al., WACV 2024) | one score per image | **Corrupted, in its authors' training.** The encoder was trained on synthetically distorted images, and the `kadid10k` regressor on human ratings of synthetically distorted images. We train nothing. | [miccunifi/ARNIQA](https://github.com/miccunifi/ARNIQA); `pyiqa` (`arniqa`) | |
| External image quality | **ARNIQA embedding + clean prototype**, following Becker et al. (2026) | one cosine distance per image | **Corrupted, in its authors' training** (same encoder). We add only a clean prototype from Cityscapes train images. | ARNIQA repo (`return_embedding`) | The prototype is the mean embedding of the same clean Cityscapes train images. Becker et al. have no code, but the protocol is fully described. |
| External image quality | **QualiCLIP** or **CLIP-IQA** score (pick one) | one score per image | **QualiCLIP: corrupted**; its authors fine-tuned CLIP on synthetically degraded images. **CLIP-IQA (zero-shot): neither**; plain CLIP with text prompts, no distortion training. (CLIP-IQA+ adds prompts learned from human ratings of real-world distorted photos.) We train nothing. | `pyiqa` (`qualiclip`, `clipiqa`); [miccunifi/QualiCLIP](https://github.com/miccunifi/QualiCLIP); [IceClear/CLIP-IQA](https://github.com/IceClear/CLIP-IQA) | |
| External image quality | **NIQE** (Mittal et al., 2013) | one score per image | **Clean only.** It fits a statistical model to clean (pristine) images and never sees distorted ones. | `pyiqa` (`niqe`) | Training-free. Refitting its clean model on Cityscapes train images is still standard NIQE. |
| General shift detector | **DisCoPatch** (Caetano et al., ICCV 2025) | one score per image: the mean discriminator score over its patches | **Clean only.** We train it ourselves on the same clean Cityscapes train images; no corrupted or OOD data. | [caetas/DisCoPatch](https://github.com/caetas/DisCoPatch), MIT | The newest general detector of shifted images (95.5% AUROC on ImageNet-C). Use the authors' default settings. Its 256 × 256 resize and 64-pixel patches may hide fine noise or JPEG damage in 2048 × 1024 images, so report that as a caveat. |

All external models are available through [IQA-PyTorch (`pyiqa`)](https://github.com/chaofengc/IQA-PyTorch).

**For comparison, ours:** clean images only (the Cityscapes train bank), with no labels and no
corrupted data. All detector-based methods, ours included, read the same detector, fine-tuned for
detection on clean, labelled Cityscapes train images. The only baselines that saw corrupted data
are the IQA models, and only in their authors' pretraining. State this in the paper, since it
favours them. DisCoPatch is the only baseline we train ourselves, on clean images only.

## Not kept as baselines

| Method | Why not |
| --- | --- |
| Original Topological Uncertainty (Lacombe et al., 2021) | Gives one score per query. We build on it and say so, and its class-wise reference becomes an ablation. |
| Energy score (Liu et al., 2020) | One score per query in a detector. Ablation only. |
| kNN on the 300 queries | This is our pipeline without the topological step. Ablation only. |
| Mahalanobis (Lee et al., 2018) | Per query: ablation. On a pooled feature it would be a possible extra baseline, but its code was not checked. |
| DA-CLIP | Has no "clean" class, so a clean-vs-degraded score would be our invention. |
| LIQE distortion head | "1 − p(others)" would be our invention. Its quality score could be used, but was not chosen. |
| Becker et al. (2026, arXiv preprint, "degradation manifolds") | No code, and it trains a monitoring branch on damaged images. Discussed as close concurrent work. Their ICPR 2026 paper is a separate, training-free method; it is a candidate below. |
| Rahman / Yatbaz / Keser & Knoll (LFA) runtime monitors | No public code found for Rahman; not checked for Yatbaz and Keser & Knoll. A reimplemented "Rahman-style" supervised predictor is still an open decision (see below). |
| Foundation-model density (Keser et al., BMVC 2025) | No public code, and it relies on an external foundation model (CLIP, DINOv2 or Grounding DINO). **Related work:** the closest new work, with the same Cityscapes → Foggy Cityscapes / ACDC setting, but segmentation only. |
| MA-CLIP (Liao et al., AAAI 2026) | Relies on an external pretrained model, CLIP (RN50 image and text encoders by default; RN101, ViT-B/32 and ViT-L/14 also supported). **Related work:** the newest zero-shot image-quality model; it needs no training data. |

COCO run plan: `docs/superpowers/plans/2026-09-27-coco-baseline-numbers.md`.

## Candidate newer baselines (2025–2026), still undecided

From the literature review, Section 11. Already decided: DisCoPatch is kept; Keser et al. and MA-CLIP
are related work only.

| Baseline | Native output | Data needed: corrupted, or clean only? | Code | Notes |
| --- | --- | --- | --- | --- |
| **Activation-distribution monitor** (Becker, Bayer, Hübner & Arens, "Operational Readiness for Object Detection", ICPR 2026). *Recommended; awaiting decision.* | one score per image: the Earth Mover's distance between each channel's activation CDF and a training-set reference CDF, summed over channels and layers | **Clean only, no labels.** Reference CDFs come from clean train images through the frozen detector. | none found; fully described (1,000 bins per channel; ranges from the training minimum and maximum plus a 20% margin) | The closest *published* work, in our exact setting: a frozen detector (RT-DETR-l among others), clean COCO, COCO-C, AUROC. Reviewers will expect it. Their best variant reads the backbone layers. On the same pass as Hashemi et al. it adds little cost. Literature review, Section 12. |
| MaRS (Di Salvo et al., arXiv 2026) | one score per image | **Clean only, no labels.** | [francescodisalvo05/mars](https://github.com/francescodisalvo05/mars), Apache-2.0 | Relies on an external foundation model (ViT or DINOv3). Never tested on corruptions. |
| Cumulative Consensus Score (Manoharan et al., arXiv 2025) | one score per image | **Neither.** Box consistency under test-time augmentation. | not found | The only new score built on detector outputs, but it has no code, was never tested under degradation, and needs about 9 extra forward passes. |

## Ablations (our aggregation held fixed)

- Reference: the TU class-wise Fréchet mean vs our kNN bank.
- Representation: the raw decoder query embedding vs the topological fingerprint.
- Per-query score: energy vs fingerprint distance.
- Aggregation: confidence-weighted mean vs plain mean vs max vs top-k.
- Decoder layer (0/1/2), k, and bank size.
- Early vs late features: a kNN bank on pooled early backbone features of the same detector.

## Metrics we compute

**Separation: does the score tell degraded from clean?**
- **AUROC**, clean vs degraded, per Cityscapes-C family and severity, per Foggy Cityscapes density,
  and per ACDC condition. Report aggregates over the 15 common families and over all families
  separately.
- **AUPR:** average precision with degraded images as the positive class. Like AUROC it needs no
  threshold: it is the monitor's counterpart to detection AP. Chance is 0.5, because each condition
  has as many degraded images as clean ones.
- **FPR95:** the share of clean images flagged when 95% of degraded images are caught. This is the one
  standard operating point in OOD papers.
- **Balanced accuracy is not reported.** A single threshold says little on its own, and AUROC and AUPR
  already cover every threshold.
- **Paired bootstrap intervals,** resampling whole images, with the same draws for every method. They
  cover every headline number and every pairwise difference between methods.

**Harm alignment: does the score follow the damage to the detector?**
- **Per condition:** rank correlation between each condition's mean score and the detector's mean
  LRP (or mAP), across all Cityscapes-C conditions and the 3 fog densities.
- **Per image, within each condition (primary):** for each condition, the rank correlation between an
  image's score increase and its LRP increase (corrupted vs clean), then the mean over conditions.
  Doing this within conditions keeps the between-condition effect out. Images whose LRP is undefined
  (no objects and no detections) are dropped and counted.
- **LRP threshold:** one confidence threshold for all methods and conditions: the LRP-optimal one on
  the clean evaluation images.
- **Risk–coverage (secondary):** reject the highest-scoring images, then plot the per-image LRP of
  the images kept, with an oracle curve for reference. Summarised as the area under the curve, for
  pools by severity, by family and overall. It also rewards flagging hard clean scenes, which is why it
  is secondary.

**Context:**
- The detector's mAP and LRP drop per condition, which shows how much each degradation hurts.
- Runtime per image for the detector alone, and for the detector plus each monitor.

**Not computed:** SAOD's LaECE, IDQ, IDQ_T and DAQ. They score a calibrated self-aware detector, not
a monitor.

LRP is available in `uq-detr` (`uq_detr.lrp()`).

## Fairness rules

1. **Same clean reference data.** Every method that needs clean images uses the same Cityscapes train
   images: our bank, the kNN bank, the ARNIQA prototype, DisCoPatch's training set, and NIQE if
   refitted.
2. **Same tuning data.** The only fitted quantity is ContrastiveConf's λ. It is cross-fitted over 5
   fixed folds of the evaluation images, so no image is scored with a λ fitted on itself. The paper
   states that ContrastiveConf is the only method that uses labels. Ours and the other baselines use
   none.
3. **Same images, corruptions and seeds** for every method.

## Open items

- **Hashemi et al. on RT-DETRv2: which layer to monitor.** *Decided 29 September 2026: option A.*
  They monitor all neurons of the last hidden layer before the output (PolyYOLO's last batch-norm or
  leaky-ReLU layer), and cite earlier work that last layers work best. The two RT-DETRv2 analogues:
  - **A (the baseline): the last decoder layer's query embeddings** (300 queries × 256 channels),
    right before the class and box heads. This is the literal "last hidden layer". The queries are
    ordered by encoder score, so neuron (j, c) is channel c of the j-th proposal.
  - **B (sensitivity check only): the hybrid encoder's output maps** (256 channels at strides 8, 16
    and 32 for a 640 × 640 input). This is the last *spatial* layer, closest to PolyYOLO's feature
    map.

  μ and σ are fitted on all clean COCO train images, as for the other baselines.

- **External-model rule.** Decide whether "relies on an external pretrained model" also rules out
  ARNIQA and QualiCLIP/CLIP-IQA. Both are external pretrained models, and CLIP-IQA and QualiCLIP use
  CLIP, just like MA-CLIP.
- **ACDC.** Check which images come with labels, and whether instance labels give usable boxes for
  the harm metrics. AUROC needs no labels.
- **LRP citation.** Cite the original LRP paper (Oksuz et al.), which is not yet in the verified
  reference list.
- **Supervised reference.** Decide whether to reimplement one for the leave-families-out
  generalization test. It would use pooled backbone features and an MLP, "Rahman-style".
- **COCO setting.** Decide whether to keep the COCO-trained detector on COCO-C as a second setting for
  comparability with the literature.
- **Protocol fixes before any run:**
  - use the package's Gaussian blur;
  - evaluate severities 1–5;
  - report the 15 common and 4 extra families separately.
