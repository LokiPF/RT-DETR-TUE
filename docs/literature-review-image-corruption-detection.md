# Related work: telling corrupted images apart using a detector's own signals

*Literature search, 27 September 2026. Written with AI assistance (Claude). Every paper below was
checked against its arXiv, proceedings, or publisher page. Datasets and metrics were read from each
paper's experiments section unless a note says otherwise. Section 10 explains how the search was
done and what it may have missed. Section 6 on image-quality assessment and Section 11 on 2025–2026 work were added
later. Section 12, a second search on image degradation with object detectors, was added on 29 September 2026.*

The question behind this search: **who else uses a trained object detector's outputs or internal
features to decide whether an input image is corrupted or shifted, and which datasets and metrics
do they use?** Our own benchmark is described in the [README](../README.md): a frozen RT-DETRv2-R18,
a bank of topological fingerprints from clean COCO train images, and AUROC for clean versus corrupted
COCO val images at severities 4 and 5.

## The short version

- **One paper is very close to ours.** Becker et al. (2026, preprint) use COCO-pretrained detectors,
  including RT-DETR. They build a reference from clean COCO train images, score each image by cosine
  distance to that reference, and report AUROC for clean versus corrupted COCO val images at each
  severity. The main differences:
  - They *train* a separate monitoring branch (a small head plus fine-tuned backbone features) on
    synthetically damaged COCO train images, while the detector itself stays unchanged.
  - For RT-DETR they read only the CNN backbone, not the transformer or the queries.
  - They use one reference point instead of a nearest-neighbour bank.
  - They give each image one corruption.

  With RT-DETR-l they reach 95.6 AUROC at severity 4 and 97.1 at severity 5. Their confidence
  baseline, taken from a probabilistic RetinaNet, reaches 69.7 and 77.7.
- **A second paper by the same group is now the closest *published* work** (Section 12). Becker, Bayer,
  Hübner and Arens, "Operational Readiness for Object Detection" (ICPR 2026):
  - Frozen detectors, including RT-DETR-l, and clean COCO train images only.
  - An image-level score: per-channel activation distributions compared with a training reference.
  - AUROC for clean versus corrupted COCO val (19 corruptions, 10 severities).

  Like every paper we found, it measures only whether the score separates corrupted from clean
  images. It does not check whether the score follows the harm to the detector.
- **Two other detector papers make image-level decisions but ask a different question.** SAOD (Oksuz
  et al., CVPR 2023) wants the detector to *accept* mildly corrupted images and reject only images
  that contain no known objects. Park et al. (TPAMI 2026) build an image-level reliability score from
  a frozen DETR's queries, but they measure how well it tracks per-image AP, not AUROC.
- **We found no paper that applies Topological Uncertainty to an object detector.** The original
  method and its follow-ups use small image classifiers (MNIST, CIFAR-10 and similar). The two
  follow-ups that detect corruptions do it for *batches* of images with a statistical test, not for
  single images.
- **Our datasets and main metric match the field.** In object detection, corrupted test sets are almost
  always made by running the `imagecorruptions` package (Michaelis et al., 2019) on COCO val2017.
  AUROC is the standard metric for image-level detection. Out-of-distribution (OOD) papers usually
  also report FPR95.
- **Image-quality assessment (IQA) supplies baselines that need no detector.** IQA papers report
  correlation with human quality scores, not AUROC. In Becker et al.'s COCO test, IQA quality scores
  reached only about 53–73 AUROC. ARNIQA's embedding compared with a clean-COCO reference reached 85.7
  at severity 5, so the "embedding plus clean reference" recipe matters more than the quality score.
  Section 6 covers this.
- **Two points to fix or state in our write-up.** Our `gaussian_blur` is not the package's
  Gaussian blur; it is twice as strong. We also use all 19 families, while the standard COCO-C test
  set uses 15. Section 8 lists these and a few cheap additions.

## 1. Which problem, exactly?

Many papers use corrupted images, but they ask different questions. Two papers can look alike and
still not be comparable. This table sorts the work we found by the question it asks.

| Question the paper asks | What the score should flag | Typical metrics | Examples |
| --- | --- | --- | --- |
| **Is this image corrupted or shifted?** (our question) | a corrupted image | AUROC, FPR95 | Becker 2026; Lee & AlRegib 2020; Viviers 2024 |
| Should the detector refuse this image? | an image with no known objects; mild corruptions should be *accepted* | AUROC, balanced accuracy, DAQ | SAOD (Oksuz 2023) |
| Will the detector do badly on this image or frame? | a frame where detector accuracy drops below a threshold | AUROC, FPR95, F1; correlation with per-image AP | Rahman 2021a/b; Yatbaz 2024; Park 2026 |
| Has the incoming data shifted? | a *batch* of images | power of a statistical test | Rabanser 2019; MAGDiff 2024; DGP 2023 |
| How much accuracy do we lose on corrupted images? | nothing is flagged | mAP, mPC, rPC | Michaelis 2019 |
| Are the confidences still honest on corrupted images? | nothing is flagged | D-ECE, LaECE, NLL | Cal-DETR 2023; Kuzucu 2024 |
| Is this *object* from a class the detector never saw? | an object from an unknown class | FPR95, AUROC (per object) | VOS, SIREN, SAFE |

Our benchmark belongs in the first row. Several large benchmarks use the *opposite* convention:
OpenOOD v1.5, Full-Spectrum OOD and SAOD count corrupted images as normal data that a model should
accept. Our write-up should say plainly that the target is detecting strong corruption, and that this
is a choice.

## 2. The closest papers

### Becker et al. (2026): Self-Aware Object Detection via Degradation Manifolds (preprint)

- **What they do.** They start from COCO-pretrained Ultralytics detectors: YOLOv9-m, YOLOv10-m,
  YOLOv11-m and RT-DETR-l, all at 640×640 input.
  - **Features.** A monitoring branch reads feature maps from several backbone depths. For RT-DETR-l
    these are CNN backbone layers 0, 2, 4, 8 and 9, before the transformer.
  - **Embedding.** Each map is shrunk by a 1×1 convolution and a learned attention pooling. The
    results are concatenated and passed through a small MLP into a unit-length embedding.
  - **Training.** The branch is trained contrastively, as in ARNIQA. Two *different* images damaged
    with the same random chain of degradations form a positive pair. A half-size crop scaled back up
    serves as a hard negative, which teaches sensitivity to lost resolution.
  - **Two paths.** The branch's backbone features are fine-tuned, but the detector used for detection
    stays unchanged. Their appendix shows that fine-tuning the detector's own backbone this way harms
    detection.
  - **Score.** One minus the cosine similarity to a "pristine" prototype: a running average of the
    embeddings of clean COCO train images.
- **Training data.**
  - COCO train2017 only, damaged on the fly with ARNIQA-style chains: blur, noise, compression,
    brightness, colour, contrast/sharpness and spatial distortion.
  - A weather variant adds Hendrycks & Dietterich weather corruptions.
  - The detectors themselves are the standard COCO train2017 weights.
- **Evaluation data.**
  - **COCO val.** Clean images against the same images run through the Michaelis et al. corruption
    suite at severities 1–5. In the main table each image gets one corruption, assigned round-robin
    and kept the same across severities. Per-corruption results are in the appendix.
  - **Other datasets.** Synthetically damaged KITTI, VisDrone, DETRAC, UAVDT and FLIR, plus mixed
    COCO-plus-other-dataset pools.
  - **Real weather.** Seeing Through Fog (clear vs dense fog, heavy snow, heavy rain) and BDD100K
    (clear vs heavy rain), with ambiguous images removed by hand. The paper gives no image counts.
- **Two inconsistencies in the paper.**
  - *Datasets.* The cross-dataset section says the monitor is trained on COCO only. The text next to
    Figure 5 says BDD and KITTI were "not part of the training pool, whereas the remaining datasets
    were".
  - *Degradations.* The Figure 4 caption says training used only ARNIQA-style degradations. The
    appendix says training used degradations from both ARNIQA and Hendrycks & Dietterich, and the
    evaluation corruptions are built on Hendrycks & Dietterich.
  - Either way, their "zero-shot" claims are weaker than stated.
- **Metric.** AUROC, with degraded images as the positive class.
- **Baselines.**
  - Uncertainty outputs of probabilistic detectors (RetinaNet, Faster R-CNN, DETR variants), each
    averaged over the top-3 detections.
  - Normalizing flows on detector features.
  - Image-quality-assessment models: CLIPIQA, MANIQA, QualiCLIP, ARNIQA.
- **Results (RT-DETR-l, severities 1–5).** 83.9, 88.3, 90.4, 95.6, 97.1 AUROC. In an ablation with
  YOLOv10-m at severity 1, reading only the last backbone layer dropped AUROC from 88.6 to 81.9.
  Early layers matter.
- **Why it matters for us.** It is the nearest published protocol. Its numbers are not directly
  comparable to ours: they train the monitor, mix corruptions across images, and take the confidence
  baseline from different detectors. They also argue that detection training makes late features
  ignore damage. Our fingerprint comes from a late decoder layer, so this is worth testing rather
  than assuming.

### Oksuz et al. (2023): Self-Aware Object Detectors, SAOD (CVPR 2023)

- **What they do.** A detector should decide, per image, whether to accept or reject it, and then give
  calibrated detections. The image-level uncertainty is the mean of (1 − confidence) over the three
  most confident detections. This "top-3" aggregation worked best among the options they compared:
  sum, mean, top-k and min. The "min" option equals our confidence baseline, 1 − max confidence.
- **Data.**
  - The general setting trains on COCO. The in-distribution test set is Obj45K, a COCO-class subset of
    Objects365.
  - The corrupted set Obj45K-C applies 15 ImageNet-C-style corruptions at severities 1, 3 and 5, one
    corruption per image.
  - The images to reject, SiNObj110K-OOD, contain no COCO objects and come from Objects365,
    iNaturalist and SVHN.
  - A driving setting trains on nuImages and tests on BDD45K from BDD100K.
- **Detectors.** Faster R-CNN, Rank & Sort R-CNN, ATSS and Deformable DETR.
- **Metrics.** AUROC and balanced accuracy for accept/reject; LaECE for calibration; LRP for accuracy;
  DAQ, one summary number combining them.
- **Key difference.** Corrupted images at severities 1 and 3 must be *accepted*. At severity 5 either
  choice is allowed, because such images "might not contain enough cues".

### Park, Sobolewski and Azizan (2026): uncertainty in detection transformers (IEEE TPAMI)

- **What they do.** They study how DETR-family detectors use their queries. Typically one query per
  object is confident and well calibrated, and the rest are pushed down. They propose an image-level
  score: the mean confidence of the kept queries minus a weighted mean confidence of the discarded
  ones.
- **Detectors.** UP-DETR, Deformable-DETR, Cal-DETR, DINO; all frozen.
- **Data.** COCO val2017 as normal data, Cityscapes as mild shift, Foggy Cityscapes as strong shift.
- **Metrics.** Pearson correlation between the image score and that image's AP; a new object-level
  calibration error (OCE), D-ECE, LaECE, AP. There is no AUROC.
- **Why it matters for us.** It is the only paper we found that aggregates a *frozen DETR's queries*
  into an image-level score. It also suggests the low-confidence queries carry information, which
  our confidence weighting mostly ignores.

### Frame-level failure monitors for driving

This line of work trains a small classifier on detector features to predict frames where the
detector will do badly. The label comes from the detector's own accuracy, such as per-frame mAP below
a threshold, not from whether the image was corrupted.

- **Rahman et al., WACV Workshops 2021.**
  - Faster R-CNN on KITTI and BDD, including cross-dataset tests.
  - Metrics: AUROC, AUPRC, true warning rate.
- **Rahman et al., IROS 2021.**
  - Faster R-CNN and RetinaNet; KITTI, BDD, Waymo; windows of 10 frames.
  - Metrics: AUROC, TPR at 5% FPR, FPR at 95% TPR, MAE, RMSE.
- **Yatbaz et al., IEEE T-IV 2024** (and an ICCV 2023 workshop precursor).
  - FCOS, YOLOv8, Faster R-CNN; KITTI, BDD100K.
  - Metrics: AUROC, F1, false-negative rate.
- **Keser and Knoll, 2026 (preprint).**
  - Faster R-CNN and DETR; KITTI, BDD100K.
  - Metrics: AUROC, F1, false-negative rate.
- **Qutub et al., CVPR Workshops 2024.** This one is closer to OOD detection.
  - A retrained DINO-DETR scores each image by disagreement between two box heads.
  - Normal data: KITTI or BDD100K. Shifted data: BDD100K, Cityscapes, Lyft (near) and COCO (far).
  - Metrics: AUROC, AUPR, FPR95.

### Estimating detector accuracy without labels

These papers predict a detector's mAP on a whole unlabeled dataset, so the decision is made per
dataset, not per image.

- **BoS (Yang et al., ICLR 2024).**
  - Measures how stable boxes stay under feature dropout.
  - Data: leave-one-out over 10 vehicle and 9 pedestrian datasets.
  - Metrics: RMSE of predicted mAP, R², Spearman correlation.
- **PCR (Yoo et al., ICCV 2025).**
  - Adds a corruption meta-dataset: 10 corruption families at severities 1–5.
  - Metrics: RMSE, Pearson and Spearman correlation.

## 3. The Topological Uncertainty family

Our fingerprint follows Topological Uncertainty (TU) by Lacombe et al. (IJCAI 2021). TU treats a
dense layer as a graph whose edge weights are |input × weight|, keeps the maximum spanning tree, and
uses its sorted edge weights as a summary. It then compares that summary with an average summary
computed from training samples of the same predicted class.

- **Lacombe et al. (2021), the original TU paper.**
  - Models: small CNNs and MLPs.
  - OOD tests: CIFAR-10 versus Fashion-MNIST, MNIST, SVHN, DTD and noise images, measured with FPR95
    and AUC.
  - On CIFAR-10 the AUC was moderate: 65.8 against SVHN, 57.3 against DTD, 86.4 against
    Fashion-MNIST. The authors say it is less effective on images than on graph data.
  - Shift is shown only as plots: pixel corruption and blur on MNIST. There is no AUROC for shift.
- **MAGDiff (Arnal et al., TMLR 2024), by the same group.**
  - Detects covariate shift from activation-graph matrices, without persistence.
  - Data: MNIST, Fashion-MNIST, CIFAR-10, SVHN, Imagenette; Gaussian noise, Gaussian blur and affine
    shifts at six intensities.
  - The decision is per batch, using statistical tests. The metric is test power, meaning how often a
    real shift is detected.
- **Deep Graph Persistence, DGP (Girrbach et al., TMLR 2023).**
  - Applies whole-network persistence to MLPs on MNIST, Fashion-MNIST and CIFAR-10, with noise,
    dropout and blur corruptions.
  - Batch-level, compared against TU and MAGDiff. The metric is the share of corrupted batches
    detected.
  - Their finding relevant to us: comparing with *all* class means beat comparing with only the
    predicted class. This partly anticipates our class-agnostic bank.
- **Goibert et al. (NeurIPS 2022 ML Safety Workshop).**
  - Same kind of |activation × weight| graph and 0-dimensional persistence, fed to a trained SVM.
  - Target is adversarial inputs, not corruptions: MNIST, Fashion-MNIST, SVHN, CIFAR-10; metric AUC.
- **Pollano et al. (IJCAI 2024 AISafety Workshop).**
  - Topological features of BERT attention maps, scored by nearest-neighbour distance to a bank of
    in-distribution features.
  - This is the only kNN-bank-of-topological-features design we found, and it is for text OOD:
    HuffPost news versus CNN/DailyMail and IMDB. Metrics: AUROC, FPR95.
- **Also found, further away:**
  - A Bayesian extension of TU (Yeh & Yang, 2025 preprint), batch-level on MNIST-scale data.
  - Early persistent-homology adversarial detection (Gebhart & Schrater, 2017 preprint).
  - Neural Persistence (Rieck et al., ICLR 2019), which looks at weights only.

**Takeaway.** Nobody we found has taken TU to object detectors, DETR queries, or a standard
corruption suite scored per image.

## 4. General OOD and covariate-shift detection

### Where our score and baselines come from

These methods are almost always tested on *new-class* OOD, not corruptions. The usual setup is
CIFAR-10/100 or ImageNet as normal data against SVHN, LSUN, iSUN, Textures, Places365, iNaturalist or
SUN.

- **kNN (Sun et al., ICML 2022).**
  - Uses the k-th nearest-neighbour distance on L2-normalised penultimate features, with k from 50 to
    1,000.
  - Metrics: FPR95, AUROC.
  - Normalising the features was a large part of the gain.
- **Mahalanobis (Lee et al., NeurIPS 2018).**
  - Metrics: TNR at 95% TPR, AUROC, AUPR, detection accuracy.
- **Maximum softmax probability (Hendrycks & Gimpel, ICLR 2017).**
  - The source of our confidence baseline. Metrics: AUROC, AUPR.
  - A footnote notes that an entropy-like score behaved almost the same as max probability.
- **Energy score (Liu et al., NeurIPS 2020).**
  - The standard logit-based alternative to max probability. Metrics: FPR95, AUROC, AUPR.

### Papers that, like us, treat corrupted images as the thing to flag

- **Lee & AlRegib (ICIP 2020).** Re-read on 29 September 2026.
  - **Data:** CIFAR-10 versus CIFAR-10-C, and CURE-TSR traffic signs. Only **8 of the 19** CIFAR-10-C
    types are reported, chosen to match CURE-TSR (noise, lens blur, Gaussian blur, dirty lens,
    exposure, snow, haze, decolour), at levels 1–5.
  - **Metric:** AUROC for each type at each level, with no averages. Their OOD experiments also report
    detection accuracy and AUPR.
  - **Training:** the classifier sees only clean images. The detector is a 2-layer network on gradient
    features, trained on clean *and* corrupted test images (40/40/20 split). So it sees the
    corruptions, unlike ours.
- **Lee et al. (IEEE Access 2023).** Re-read on 29 September 2026.
  - **Data:** all 19 CIFAR-10-C families at levels 1–5 (95 cases), plus CURE-TSR (12 × 5).
  - **Metric:** *detection accuracy* per corruption and level, and averaged per corruption. This is the
    best accuracy over all thresholds, max_δ {0.5·P_in(q ≤ δ) + 0.5·P_out(q > δ)}; for their own
    detector δ is fixed at 0.5.
  - **Why not AUROC:** AUROC was "highly saturated", and their significance test (a corrected
    repeated k-fold cross-validation paired t-test, k = 5, r = 2, p = 0.05) needs predictions.
  - **Training:** the same kind of detector, trained on clean and corrupted examples, with 5-fold
    cross-validation repeated with 2 seeds.
- **Viviers et al. (ECCV 2024 Workshops).** Re-read on 29 September 2026.
  - **Models:** generative models (VAEs, normalizing flows, diffusion), trained only on clean
    training images.
  - **Data:** CIFAR-10-C (19 families × 5 levels = 95 test sets) and ImageNet200-C (15 × 5 = 75),
    each against its clean 10k test set, following OpenOOD.
  - **Metrics:** AUROC and FPR95 per condition. The main table gives the mean AUROC over all corruptions
    at each severity, plus an overall mean AUROC and FPR95. Per-corruption tables are in the
    supplement.
  - They argue that only the overall performance matters, because the type of degradation cannot be
    predicted. This is the closest reporting template to ours.
- **Tian et al. (2021 preprint; short version at a NeurIPS 2021 workshop).**
  - CIFAR-10/100-C, 15 families × 5 levels. Metrics: AUROC, TNR at 95% TPR.
  - Includes entropy-based baselines.
- **Ferreira et al. (IEEE PRDC 2021).**
  - Safety monitors for classifiers on CIFAR-10 and GTSRB, tested with 19 corruptions at levels 1–5.
  - Metrics: MCC, F1, false-positive and false-negative rates.
  - Several monitors were no better than random.

### Papers that treat corrupted images as normal

- **Full-Spectrum OOD (Yang et al., IJCV 2023)** and **OpenOOD v1.5 (Zhang et al., DMLR 2024).**
  - Count CIFAR-10-C or ImageNet-C as normal data that should be accepted. Metrics: AUROC, FPR95,
    AUPR.
  - OpenOOD samples one 10,000-image set across all 15 × 5 ImageNet-C combinations.
  - Full-Spectrum OOD also found that simple low-level feature statistics react strongly to
    corruption.
- **ImageNet-OOD (Yang, Zhang & Russakovsky, ICLR 2024).**
  - Modern OOD detectors react more to covariate shift than to new classes.
  - Even with randomly initialised networks, blurred ImageNet-C images were scored as more unusual
    than clean ones, while noisy images were scored as *less* unusual.
  - So the direction of a feature-based score can depend on the corruption family.

### Uncertainty under dataset shift

- **Ovadia et al. (NeurIPS 2019).**
  - Classifier calibration on CIFAR-10-C and ImageNet-C: 16 corruption types × 5 intensities.
  - Metrics: accuracy, ECE, Brier score, NLL, plotted per intensity.
- **Rabanser et al. (NeurIPS 2019).**
  - Dataset-level two-sample tests on MNIST and CIFAR-10 with noise and affine shifts.
  - Metric: detection rate against batch size. Tests on softmax outputs worked best.
- **Jaeger et al. (ICLR 2023).**
  - Failure detection, where the target is misclassification, on CIFAR-10/100-C and other shifts.
  - Metric: AURC. Plain softmax confidence was a strong baseline.

## 5. Detector uncertainty and calibration under corruption

These papers study corrupted COCO but measure detection quality or calibration, not whether the
image is corrupted.

- **Harakeh & Waslander (ICLR 2021).**
  - RetinaNet, Faster R-CNN and DETR on COCO val with 18 corruptions at levels 1, 3 and 5, plus
    OpenImages.
  - Metrics: NLL, energy score, Brier score, mAP, calibration error.
- **Munir et al.: TCD (NeurIPS 2022), BPC (CVPR 2023), Cal-DETR (NeurIPS 2023).**
  - Calibration under shift, largely with Deformable-DETR, UP-DETR and DINO.
  - "CorCOCO" gives each COCO val image one randomly chosen corruption out of 19, at a random severity
    from 1 to 5. Also Foggy Cityscapes and BDD100K.
  - Metrics: D-ECE, AP.
- **Kuzucu et al. (ECCV 2024).**
  - Calibration of many detectors, including D-DETR, DINO and Co-DETR.
  - Data: COCO-C, Obj45K, Cityscapes-C, Foggy Cityscapes, LVIS-C.
  - Metrics: LaECE, LaACE, LRP, D-ECE, AP.
  - Simple post-hoc calibration beat training-time methods.
- **Probabilistic detection.**
  - Hall et al. (WACV 2020) introduced the PDQ metric.
  - Feng et al. (IEEE T-ITS) compared MC-dropout and ensemble detectors from BDD100K to KITTI and
    Lyft, using mAP, PDQ, NLL and ECE.
  - GroupEnsemble (Yang et al., 2026 preprint) uses extra DETR query groups as a cheap ensemble on
    Cityscapes → Foggy Cityscapes, measured with mAP, PDQ and D-ECE.
- **Object-level new-class OOD: VOS (ICLR 2022), SIREN (NeurIPS 2022), SAFE (ICCV 2023).**
  - Normal data: PASCAL VOC and BDD100K. OOD data: COCO and OpenImages subsets with no known classes.
  - Metrics: FPR95, AUROC, mAP.
  - SIREN is relevant to our design. It scores Deformable-DETR decoder embeddings by kNN distance.
    It reports that a Mahalanobis score on plain Deformable-DETR embeddings was near chance (about 50%
    AUROC), because the embeddings are not Gaussian.

## 6. Image-quality assessment: detecting degradation without a detector

Image-quality assessment (IQA) is the field whose job is judging how damaged an image is. Most IQA
models predict a human quality rating, the mean opinion score (MOS). They are judged by rank
correlation (SRCC) and linear correlation (PLCC) with human scores, not by AUROC. For us the field
provides two things: baselines that need no detector, and ideas for damage-aware features.

### Self-supervised damage encoders: the best fit as baselines

- **ARNIQA (Agnolucci et al., WACV 2024).**
  - A contrastive "distortion manifold": two different images with the same chain of damage are
    positives.
  - Trained on 140,000 clean KADIS-700k images with the 24 KADID distortion types × 5 levels, in
    chains of up to four. There are no weather types.
  - Evaluated with SRCC and PLCC on LIVE, CSIQ, TID2013, KADID-10k, FLIVE and SPAQ.
  - Public weights through torch.hub and `pyiqa` (`arniqa`).
  - Becker et al. copied its training recipe. In their COCO test, its embedding compared with a
    clean-COCO prototype was the best IQA baseline: 73.6 AUROC at severity 1 and 85.7 at severity 5.
- **CONTRIQUE (Madhusudana et al., IEEE TIP 2022) and Re-IQA (Saha et al., CVPR 2023).**
  - Contrastive training on distortion type × level (25 × 5), plus unlabeled real photos.
  - Public checkpoints, but not in `pyiqa`.
  - Re-IQA keeps a separate "quality" encoder and "content" encoder, which is the split we want.
  - Caveat: both list COCO (about 330,000 images) among their unlabeled training images, so they may
    have seen COCO val.
- **QualiCLIP (Agnolucci et al., preprint).**
  - CLIP fine-tuned to rank increasingly damaged crops, without human scores. In `pyiqa` as
    `qualiclip`.
  - In Becker et al.: 72.5 AUROC at severity 5 as a score, but only about 56 as an embedding.

### Off-the-shelf quality scorers

- **CLIP-IQA (Wang et al., AAAI 2023).**
  - Zero-shot CLIP with the prompts "Good photo." and "Bad photo.". Attribute prompts such as
    "Blurry photo." or "Noisy photo." could give per-family scores.
  - In Becker et al.: 69.7 AUROC at severity 5 as a score. As an embedding it was worse than chance
    (45.8).
- **MANIQA (Yang et al., CVPR Workshops 2022).**
  - A ViT trained on human scores, with PIPAL, KADID-10k and KonIQ-10k checkpoints.
  - In Becker et al.: 61.6 as a score and 77.3 as an embedding, at severity 5.
  - The KADID-10k checkpoint, trained on synthetic damage, may suit us better than the default KonIQ
    one. That is untested.
- **LIQE (Zhang et al., CVPR 2023).**
  - CLIP-based. It also predicts one of 11 distortion types, and its "others" class includes clean
    images.
  - One minus p(others) could serve directly as a clean-vs-damaged score. This is an untested idea.
- **DBCNN (Zhang et al., IEEE TCSVT 2020).** One branch is pre-trained to classify synthetic
  distortion type and level.
- **TOPIQ, MUSIQ, HyperIQA and Q-Align** are strong human-score regressors. Q-Align is a large
  multimodal model and expensive to run.
- **NIQE (Mittal et al., IEEE SPL 2013).**
  - Classical and training-free. It fits a Gaussian to natural-scene statistics of clean images and
    scores a test image by its distance to that fit.
  - It is the conceptual ancestor of "clean bank plus distance".
- **Tooling.** Most of these models are in IQA-PyTorch (`pyiqa`, Chen & Mo), which makes them easy to
  run. Most default weights there are trained on KonIQ-10k, which contains authentic rather than
  synthetic damage.

### Telling clean images from damaged ones

- **Waterloo Exploration Database (Ma et al., IEEE TIP 2017).**
  - 4,744 clean images and 94,880 damaged ones: JPEG, JPEG2000, white noise and Gaussian blur at five
    levels.
  - It introduced the **D-test**: the best balanced accuracy for separating clean from damaged images
    over all thresholds. This is the IQA field's closest analogue to our AUROC.
  - DBCNN reports D = 0.96. MEON (Ma et al., IEEE TIP 2018) has an explicit "pristine" class in its
    distortion classifier.
- **Bianco, Celona & Napoletano (Pattern Recognition Letters 2021).**
  - Frozen ImageNet CNN features, with PCA and a nearest-neighbour classifier, separate distortion
    types and levels without any training.
  - The best layers are early ones, such as ResNet-50 layer1. Accuracy reaches 92% for distortion type
    on TID2008.
  - This supports trying early layers of our frozen detector.

### Damage encoders from image restoration

- **DA-CLIP (Luo et al., ICLR 2024).**
  - CLIP made damage-aware.
  - It classifies 10 damage types zero-shot, including haze, rain, snow, raindrops and low light,
    which the KADID-style IQA encoders lack.
  - It reports classification accuracy: near-perfect, except 91.6% for blur. Public weights.
- **DASR (Wang et al., CVPR 2021) and AirNet (Li et al., CVPR 2022).**
  - They learn damage embeddings to guide restoration.
  - They report only PSNR/SSIM and t-SNE plots, with no detection metrics, so they are idea sources
    only.

### Does image quality predict detector performance?

- **MIQA (Wang, Zhang & Lin, 2025 preprint).**
  - 5,000 COCO val and 5,000 ImageNet val images with ImageNet-C-like damage at five severities, in
    three spatial modes (whole image, objects only, background only).
  - Labels come from 75 models, including 20 detectors such as DETR, Deformable DETR, DINO and
    YOLOv9–v11.
  - Human-perception IQA predicts detector quality poorly: MANIQA reaches an SRCC of 0.46, against
    0.81 for their machine-trained model. Metrics: SRCC, PLCC, KRCC, RMSE.
- **Dremin et al. (2024 preprint).** About 40 IQA metrics correlate only about 0.2–0.3 (SRCC) with
  YOLOv5s performance under compression, on a COCO subset, WIDER and CCPD.
- **Beniwal, Mantini & Shah (VISAPP 2022).** A full-reference quality metric built from Faster R-CNN
  backbone features tracks detection AP better than PSNR or SSIM, on surveillance video.
- **Venkataramanan et al. (IS&T Electronic Imaging 2022).** Predicts per-image YOLOv3 accuracy from
  no-reference quality features, on LIVE-RoadImpairs.
- **Kees et al. (2026 preprint).**
  - Fits a normalizing flow to handcrafted quality features of clean BDD100K images and uses the
    likelihood as a quality score.
  - It then lowers the YOLOv11/DETR confidence threshold on poor-quality images. There is no AUROC.

### Takeaways from IQA

- No IQA paper we found reports clean-vs-damaged AUROC. The closest protocol is the Waterloo D-test.
  Becker et al. is the only work that computes AUROC for IQA models on COCO.
- In that test, "embedding plus clean reference" (ARNIQA, 85.7) beat every quality *score* (at most
  72.5). This is the same recipe as ours, with a detector-free encoder.
- Perceived quality and detector-relevant quality are only weakly related (MIQA, Dremin et al.). A
  detector-based signal can therefore reasonably add something beyond IQA.

## 7. Datasets and metrics at a glance

### Closest work: image-, frame- or batch-level decisions

| Paper | Decision | Model | Normal data | Shifted data | Metrics |
| --- | --- | --- | --- | --- | --- |
| **Ours** | image | RT-DETRv2-R18 (frozen) | COCO train (bank), COCO val | COCO val, 19 `imagecorruptions` families, severities 4 and 5, every image × every family | AUROC per family and severity; mean over 38 tasks |
| Becker 2026 (preprint) | image | YOLOv9/10/11-m, RT-DETR-l (separate monitor branch trained) | COCO train (training and prototype), COCO val | COCO val, Michaelis suite, severities 1–5, one family per image; VisDrone, KITTI, DETRAC, UAVDT, FLIR; fog and rain sets | AUROC |
| MIQA 2025 (preprint) | image (quality score) | ViT-small trained on labels from 75 models | COCO val, ImageNet val | ImageNet-C-like damage, 5 severities × 3 spatial modes | SRCC, PLCC, KRCC, RMSE |
| Waterloo Exploration 2017 (TIP) | image | 20 classical IQA models | 4,744 clean images | JPEG, JPEG2000, noise, blur × 5 levels | D-test (best balanced accuracy), L-test, P-test |
| SAOD 2023 (CVPR) | image | Faster R-CNN, RS R-CNN, ATSS, D-DETR | COCO → Obj45K; nuImages → BDD45K | 15 corruptions at severities 1, 3, 5 (to *accept*); no-object images (to reject) | AUROC, balanced accuracy, DAQ, LaECE, LRP |
| Park 2026 (TPAMI) | image | UP-DETR, D-DETR, Cal-DETR, DINO | COCO val2017 | Cityscapes, Foggy Cityscapes | Pearson correlation with per-image AP; OCE, D-ECE, LaECE, AP |
| Rahman 2021b (IROS) | 10-frame window | Faster R-CNN, RetinaNet | KITTI, BDD, Waymo | cross-dataset | AUROC, TPR at 5% FPR, FPR at 95% TPR, MAE, RMSE |
| Yatbaz 2024 (T-IV) | frame | FCOS, YOLOv8, Faster R-CNN | KITTI, BDD100K | none (hard frames) | AUROC, F1, FNR |
| Qutub 2024 (CVPRW) | image | DINO-DETR (retrained) | KITTI or BDD100K | BDD100K, Cityscapes, Lyft, COCO | AUROC, AUPR, FPR95 |
| PCR 2025 (ICCV) | dataset | RetinaNet, Faster R-CNN | vehicle and pedestrian datasets | + 10 corruptions × severities 1–5 | RMSE of predicted mAP, Pearson, Spearman |
| TU 2021 (IJCAI) | image | small CNNs, MLP | MNIST, CIFAR-10 | new-class OOD; MNIST corruption (plots only) | FPR95, AUC |
| MAGDiff 2024 (TMLR) | batch | CNNs, ResNet-18 | MNIST, FMNIST, CIFAR-10, SVHN, Imagenette | noise, blur, affine; 6 intensities | test power |
| DGP 2023 (TMLR) | batch | MLPs | MNIST, FMNIST, CIFAR-10 | noise, dropout, blur | share of batches detected |
| Lee & AlRegib 2020 (ICIP) | image | ResNet-18 | CIFAR-10 | CIFAR-10-C, CURE-TSR, levels 1–5 | AUROC per corruption × level |
| Viviers 2024 (ECCVW) | image | generative models | CIFAR-10, ImageNet200 | CIFAR-10-C (19 × 5), ImageNet200-C (15 × 5) | AUROC, FPR95 per severity |

### How corrupted COCO is usually built

- **Standard:** run `imagecorruptions` on COCO val2017. Michaelis et al.'s COCO-C benchmark uses the
  15 "common" families at severities 1–5 and summarises detector accuracy with mPC and rPC. The
  package and the original ImageNet-C paper keep four more families (speckle noise, Gaussian blur,
  spatter, saturate) as *validation* corruptions, not test corruptions.
- **Variants we saw:**
  - Harakeh & Waslander: 18 families at 1, 3, 5.
  - SAOD: 15 families, one per image, at 1, 3, 5.
  - CorCOCO: one random family out of 19 per image, random severity.
  - PCR: 10 families × 1–5.
  - Becker: round-robin, one family per image, 1–5.
- **Ours:** all 19 families at severities 4 and 5 only. Every image gets every family, which makes the
  comparisons paired, and results are reported per family. Reporting per family *and* per severity
  is less common. We saw it in Lee & AlRegib, Lee et al., the Viviers supplement and the Becker
  appendix.

### What the metrics mean

- **AUROC:** pick one corrupted and one clean image at random. AUROC is the chance the corrupted one
  gets the higher score. 0.5 is a coin flip and 1.0 is perfect. It needs no threshold.
- **FPR95 (FPR at 95% TPR):** set the threshold so 95% of corrupted images are caught. FPR95 is the
  share of clean images wrongly flagged at that threshold. Lower is better. TNR at 95% TPR is
  1 − FPR95.
- **AUPR:** like AUROC but built from precision. It depends on the ratio of positives to negatives.
- **Balanced accuracy:** average accuracy on the two classes at a fixed threshold. DAQ is SAOD's single
  number combining accept/reject accuracy with detection quality and calibration.
- **mPC and rPC:** detector AP averaged over corruptions and severities, and that average divided by
  clean AP.
- **D-ECE, LaECE, OCE:** calibration errors for detectors, measuring how far confidences are from
  actual precision.
- **Test power:** for batch-level shift detection, how often a statistical test detects a real shift
  from *n* images.
- **Correlation with per-image AP (Park) or RMSE of predicted mAP (BoS, PCR):** how well a score
  predicts detection accuracy.
- **SRCC and PLCC:** the standard IQA metrics. They measure how well a model's quality score follows
  human ratings: SRCC by rank order, PLCC linearly.
- **D-test:** the IQA field's clean-vs-damaged measure. It is the best balanced accuracy over all
  thresholds, which makes it a close cousin of AUROC.

## 8. What this means for our benchmark

1. **Keep the core design.** COCO val with `imagecorruptions` and AUROC, with corrupted images as
   positives, matches the closest paper (Becker 2026) and standard COCO-C practice.
2. **Fix or clearly label `gaussian_blur`.**
   - `differential_uncertainty/corruptions/__init__.py` uses Pillow's Gaussian blur with radius 8 at
     severity 4 and 12 at severity 5.
   - Pillow's radius is the standard deviation. The `imagecorruptions` package (version 1.1.2, as
     pinned in `requirements.txt`) uses σ = 4 and σ = 6.
   - So our "Gaussian blur" is twice as strong as everyone else's, and its AUROC is not comparable to
     other papers' Gaussian blur numbers. This looks like a leftover from the earlier blur-only
     workflow.
3. **Report a 15-family average next to the 19-family one.** The 15 common families are the COCO-C
   test set; the other four are validation corruptions by convention. This can be computed from
   `results.csv`.
4. **Add FPR95 next to AUROC.** Most OOD papers report both. It can be computed from
   `per_image_scores.csv` without rerunning the detector.
5. **Add a mixed-corruption number for comparison with Becker et al.** Assign one family per image,
   round-robin, and compute one AUROC per severity. This can also be computed from the existing
   per-image scores. The result still won't be directly comparable, because their method is trained.
6. **Consider stronger standard baselines.**
   - SAOD's top-3 mean confidence is the standard image-level detector baseline, and Becker uses it.
     Ours is its top-1 ("min") variant.
   - An energy score is the usual logit-based OOD baseline.
   - Both need the raw logits, which the score files do not store, so they need a code change and a
     rerun of evaluation.
   - A kNN score on the raw decoder query embeddings, without the topological step, would show
     whether persistence adds anything. SIREN found that kNN works on Deformable-DETR decoder
     embeddings.
7. **Frame the task clearly.**
   - Call it "detecting strongly corrupted images". Note that SAOD, OpenOOD and Full-Spectrum OOD
     treat mild corruption as data to accept.
   - Explain why only severities 4 and 5 are used. SAOD's own reasoning, that severity-5 images may
     lack the cues needed for detection, supports this.
8. **Expect family-dependent results.** ImageNet-OOD found that blur and noise can move a
   feature-based score in opposite directions. Our own earlier 250-image pilot, with a different and
   older score ([results](coco-imagecorruptions-250-pilot-results.md)), already showed near-chance
   AUROC for brightness, saturate, fog and elastic transform. Per-family reporting will make this
   visible, and it is more detailed than most prior work.
9. **Add baselines that need no detector.**
   - The fairest one: ARNIQA embeddings with *our* clean COCO-train bank and the same 5-nearest-
     neighbour cosine score. That isolates "which features" from "which scoring rule".
   - Cheap extras: QualiCLIP and CLIP-IQA scores, and NIQE, all in `pyiqa`.
   - These need only the images. The corrupted images are not saved, so this means a separate pass
     that regenerates the same corruptions.
10. **Try early features of our own frozen detector.** Becker et al.'s ablation and Bianco et al. both
    find that early layers carry most of the damage signal. A nearest-neighbour bank on pooled early
    backbone features of RT-DETRv2 would test whether our late-layer fingerprint is missing it.

## 9. What we did not find

- Topological Uncertainty or any activation-graph persistence applied to an object detector,
  including DETR-family decoder queries.
- A training-free, image-level corruption score built from a *frozen* detector's decoder queries with
  a nearest-neighbour bank. Becker et al. train and fine-tune; Park et al. use frozen DETR queries but
  target per-image AP.
- Confidence-weighted pooling of per-query distances into an image score.
- For detectors, per-corruption *and* per-severity AUROC as the main result. Becker et al. report it
  only in an appendix.
- An IQA paper that evaluates clean-vs-damaged separability with AUROC. IQA reports correlation with
  human scores, or the Waterloo D-test.

"Not found" means these searches did not turn it up. It does not prove the work does not exist.

*Update, 29 September 2026 (Section 12):*
- A training-free, image-level corruption score from a frozen detector now exists: Becker et al.,
  ICPR 2026. It uses activation distributions of backbone layers, not decoder queries, and no
  nearest-neighbour bank. So the second bullet above still holds for queries.
- That paper reports AUROC per severity, pooled over corruptions, but not per corruption.

## 10. How this search was done, and its limits

- **Method.** Four parallel search strands:
  - topological and activation-graph uncertainty;
  - image- and frame-level shift and failure monitoring for detectors;
  - detector uncertainty and calibration under corruption;
  - general OOD and covariate-shift detection.

  A fifth strand, on no-reference image-quality assessment and damage estimation, was added later.
  The Becker et al. paper was then read in full. Sources were arXiv, OpenReview, CVF open access, PMLR, NeurIPS/ICLR/IJCAI proceedings, ACL
  Anthology and publisher pages.
- **Inclusion rule.** A paper was included only if a fetched page confirmed its title, authors and
  venue. Datasets and metrics were taken from the experiments sections and tables. The key numbers
  for Becker et al., SAOD and the COCO-C benchmark were checked a second time against the papers.
- **Limits.**
  - This is a narrative review, not a systematic (PRISMA) review.
  - CVF and Springer pages were sometimes blocked, so some venues were confirmed through arXiv
    comments or other listings.
  - The citation sweep of the TU paper (Semantic Scholar, 28 citing papers) hit rate limits and may
    be incomplete.
  - Becker et al. (2026), Keser & Knoll (2026), GroupEnsemble (2026), Tian et al. (2021), Yeh &
    Yang (2025), QualiCLIP, MIQA (2025), Dremin et al. (2024) and Kees et al. (2026) are preprints
    without peer review.
  - The IQA strand focused on no-reference models with public weights. It did not cover
    full-reference metrics or video quality.
  - "Benchmarking Object Detection Robustness against Real-World Corruptions" (reported as IJCV 2024)
    could not be verified and was left out.
  - Work after September 2026 is not covered.

## 11. Newer work, 2025–2026: methods that need only clean images

A later search looked only at work whose first version appeared in 2025 or 2026. It kept methods
that need no degraded training data and no failure labels. It ran as two strands: monitors for object
detectors, and general detectors of shifted or degraded images. Both hit the session's web-search
limit partway through. They then switched to arXiv search and to title lists from CVF (CVPR 2025/2026,
ICCV 2025, WACV 2025/2026, and the SAIAD workshops). IEEE Xplore could not be searched directly.

### Usable as baselines

- **Keser et al. (BMVC 2025, arXiv January 2025).** The closest new work to our setting.
  - **Method:** frozen foundation-model features (CLIP, DINO, DINOv2, Grounding DINO) of clean
    Cityscapes train images, modelled by a density model. The best are a GMM, with the number of
    components chosen by AIC, and a Real-NVP normalizing flow. The image score is the log-density.
  - **Data:** evaluated on Foggy Cityscapes, ACDC (fog, night, rain, snow), BRAVO synthetic rain and
    flare, and several semantic-shift sets.
  - **Metrics:** AUROC, AUPR, FPR95.
  - **Limits:** segmentation only (DeepLabV3+), no detector, and no public code. The method is fully
    described, so it can be reimplemented. The first author is at TUM and Continental.
- **DisCoPatch (Caetano et al., ICCV 2025, arXiv January 2025).**
  - **Method:** an adversarially trained VAE discriminator scores patches of an image, trained on
    in-distribution images only. The image score is the mean patch score.
  - **Results:** 95.5% AUROC on ImageNet-C covariate shift, with per-corruption and per-severity
    results in the appendix.
  - **Code:** public, MIT licence (training and evaluation scripts, an ImageNet checkpoint).
  - **Caveats:** it would need retraining on clean Cityscapes images. Its 256 × 256 resize and
    64-pixel patches may hide fine noise or JPEG artefacts in 2048 × 1024 images.
- **MA-CLIP (Liao et al., AAAI 2026, arXiv November 2025).**
  - **Method:** zero-shot image-quality scoring from CLIP. It combines similarity to "good photo" /
    "bad photo" prompts with the length of the CLIP feature vector.
  - **Data:** needs no data at all.
  - **Code:** `pip install Maclip`.
  - **Use:** a modern successor to CLIP-IQA as the zero-shot quality baseline.
- **MaRS (Di Salvo et al., arXiv June 2026; MICCAI 2026 according to its code repository).**
  - **Method:** Mahalanobis distance on the reconstruction residuals of an autoencoder fitted to
    clean foundation-model features. One score per image, public code.
  - **Limits:** tested only on medical data, never on corruptions.
- **Cumulative Consensus Score (Manoharan et al., arXiv September 2025).**
  - **Method:** a training-free image score from a detector's own outputs: how consistent its boxes
    stay under nine photometric augmentations. It correlates with F1 on clean data (Spearman about
    0.81).
  - **Limits:** never tested under degradation, no code, and about nine extra forward passes per
    image.

### Cite and discuss, don't run

- **PCR (Yoo et al., ICCV 2025).** Has code, but maps scores to mAP with a regression fitted on
  labelled, corrupted data. Its per-image score also relies on boxes before non-maximum suppression,
  which a DETR does not have.
- **Knowledge-Guided Failure Prediction (Zimmermann et al., CVPR 2026 SAIAD workshop).**
  - Image-level, with no degraded training data.
  - Needs failure labels derived from ground truth on clean COCO images.
  - Built for YOLOv8.
- **Query2Uncertainty (Beemelmanns et al., CVPR 2026).**
  - Fits a normalizing-flow density to a DETR-style detector's object queries on clean training data.
    That is the closest methodological relative of our query bank.
  - But it works per object, needs labels (true-positive queries), is 3D (nuScenes), and its code is
    not yet released.
- **Aher (arXiv 2026, under review at IEEE T-IV).**
  - A camera-health monitor for driver-assistance systems, trained on 12 synthetic degradation types.
  - Its assumption is the opposite of ours, but IV reviewers may know it.
  - It reports correlation with YOLOv8 mAP and uses BRISQUE and NIQE as baselines.
- **Wu et al. (IEEE TPAMI, arXiv March 2025).** New-class OOD for YOLO, Faster R-CNN and RT-DETR, with
  retraining. It also argues that existing OOD benchmarks for detection are flawed.
- **HiRQA (Ramesh et al., Machine Vision and Applications, arXiv August 2025).** A newer quality
  model in the ARNIQA family, trained on synthetic distortions.
- **Heng & Soh (ICLR 2025 workshop).** Covariate-shift detection with CLIP and a new ImageNet-CS
  benchmark. Only the abstract could be read.

### One finding to keep in mind

A single-author 2026 preprint (Bhuyan) reports that DINOv2 and CLIP features barely separate CIFAR-10-C
corruptions from clean images: AUROC 0.57 for DINOv2, and CLIP "nearly blind". ImageNet ResNet-50
features reach 0.98 on the same test. Keser et al., by contrast, find foundation-model densities
near-perfect on Foggy Cityscapes. Which features react to which degradations is an open question, and
our per-family results could help answer it. The evidence here is weak (CIFAR only, one author), so
cite it with care.

### What still seems open

We found no 2025–2026 work that builds a training-free, clean-only, image-level monitor from a DETR's
own queries or features, evaluates it on weather or corruption, and releases code. (The ICPR 2026
paper in Section 12 comes closest: it has everything except a DETR-query signal and released code.)

## 12. Image degradation with object detectors: a second search

*Added 29 September 2026.* The question was narrower than in Sections 2–11: **which papers detect or
quantify the degradation of the input image, with or for an object detector, excluding mAP or
accuracy prediction?** It ran as three parallel searches:
- image-level monitors built on detectors;
- driving-camera degradation and camera health;
- detectors that estimate their own input's degradation, and runtime monitors tested on corrupted
  images.

The two papers marked "read in full" were read end to end. The rest were checked against a fetched
page and read at abstract and experiments level, unless a note says otherwise.

### The short version

- **One new paper is the closest published work:** Becker, Bayer, Hübner and Arens, "Operational
  Readiness for Object Detection" (ICPR 2026). It comes from the same group as the Becker et al. 2026
  preprint (Section 2), but uses a different method.
- **Everything else falls into one of three groups:**
  - degradation classifiers trained on labelled degraded images (soiling, weather, glare), often
    sharing the detector's backbone;
  - a small monitor on a frozen detector with a very small evaluation (Hashemi et al., FM 2023);
  - degradation signals measured over sequences, not single images (Hildebrand et al., 2023).
- **None of them checks, image by image, whether the score follows the harm to the detector.** The
  ICPR 2026 paper says so directly: "Because dense error labeling under shift is impractical, we use
  ID vs. OOD separability as a proxy." Our per-image LRP makes exactly that measurement.

### Operational Readiness for Object Detection (Becker et al., ICPR 2026), read in full

- **Question.** An image-level score for "is the input inside the detector's operating range?". No
  unsafe or corrupted samples are used to build the monitor; the paper cites Guérin et al. (AAAI
  2023) for this setting.
- **Method (activation distributions).** For chosen layers of a frozen detector:
  - Build a histogram of each channel's activations. There are 1,000 bins, and each channel's range
    is its training minimum and maximum plus a 20% margin.
  - Turn each histogram into a cumulative distribution function (CDF).
  - Compare the image's CDFs with CDFs aggregated over the whole training set, using the Earth
    Mover's distance. Sum over channels and layers.

  The idea comes from Visual DNA (Ramtoula et al., CVPR 2023, cited there; not checked by us).
- **Baselines:**
  - RealNVP normalizing flows on globally average-pooled features from several layers, either one
    flow for all layers or one per layer.
  - SAOD-style top-3 aggregation of detection uncertainties from a probabilistic Faster R-CNN
    (variance networks trained with the energy score). The scores are confidence, classification
    entropy, and the trace, determinant and entropy of the box distribution.
- **Data:**
  - Reference: COCO train. In-distribution: clean COCO val.
  - Shifted: COCO val under the 19 Hendrycks corruptions at **10 severity levels** ("extending the
    standard five by intensifying parameters"). The paper does not say whether its levels 3 and 5
    equal the standard levels 3 and 5.
- **Detectors:**
  - Faster R-CNN R50, read at backbone stages C1–C5 and at the ROI head.
  - RT-DETR-l, YOLOv9-m, YOLOv10-m and YOLOv11, read at early convolutions and FPN stages.
- **Metrics.** AUROC for clean versus corrupted at severities 3, 5, 8 and 10, pooled over the
  corruptions, as mean ± std over 3 runs, after z-score normalization. mAP is used only to show that
  the corruptions hurt the detector. There is no per-family table, no FPR95 or AUPR, no harm
  measure, and no code.
- **Results (AUROC at severities 3 / 5 / 8 / 10):**

  | Method | 3 | 5 | 8 | 10 |
  | --- | ---: | ---: | ---: | ---: |
  | Faster R-CNN, CDFs of all backbone layers | 68.0 | 80.3 | 86.8 | 87.4 |
  | Faster R-CNN, low layers C1–C3 only | 71.9 | 79.5 | 84.6 | 86.7 |
  | Best normalizing flow | 61.8 | 76.5 | 77.1 | 79.3 |
  | Top-3 classification entropy | 62.8 | 77.6 | 81.2 | 81.7 |
  | Top-3 confidence | 58.8 | 72.6 | 75.6 | 78.3 |
  | RT-DETR-l, CDFs of backbone layers | 76.4 | 86.4 | 91.4 | 92.8 |

  Adding head layers never helps, and the low-level layers carry most of the signal.
- **What it means for us:**
  - It is the closest published competitor and uses our setting: a frozen detector including RT-DETR,
    clean COCO only, COCO-C, AUROC. Reviewers will expect a comparison. The method is fully
    described and simple to reimplement on RT-DETRv2.
  - Its separation-only evaluation is exactly the gap our harm columns (within-condition ρ, AURC)
    fill.
  - Its finding that low-level backbone activations carry most of the shift signal matches
    Full-Spectrum OOD (Section 4) and Bianco et al. (Section 6). That matters for a method built on
    decoder queries, which are high-level.

### Hashemi, Křetínský, Rieder and Schmidt (FM 2023), read in full

- **Method.**
  - A frozen PolyYOLO (YOLOv3-based) trained on Cityscapes.
  - One layer is monitored: the last batch-norm layer or the last leaky-ReLU layer before the
    output. For each of its neurons they compute the mean μ and standard deviation σ over 500 clean
    training images, ignoring classes.
  - The score is the share of the layer's neurons that fall outside μ ± kσ, with k ≈ 2.
  - Inductive conformal anomaly detection turns the score into a p-value against 100 clean
    calibration images. An image is flagged below 5%.
- **Data.**
  - 100 Cityscapes test images with Gaussian noise (variance 0.02, 0.04, 0.06), impulse noise (0.03,
    0.06) and FGSM attacks.
  - KITTI and A2D2 serve as other-dataset shift.
- **Metric: images flagged out of 100.**
  - 3–6 clean images.
  - 7–9 at noise variance 0.02, 91–92 at 0.04, and all 100 at 0.06 and for impulse noise.
  - Every KITTI and A2D2 image.

  There is no AUROC and no code.
- **A hint towards harm alignment.** They note that the variance-0.02 noise that fools the monitor "is
  not as critical as large objects are still detected", but they do not measure this.
- **What it means for us.** It is the only published monitor with exactly our assumptions: frozen
  detector, clean data only, one score per image, no classes. It also comes from the
  runtime-verification community. It is now our fifth baseline (see the decision record). Our metrics
  are threshold-free, so the conformal step does not change any result: the p-value is a
  decreasing step function of the score.

### Other work on the detector side

- **Hildebrand, Brown, Brown and Waslander (IEEE Access 2023).**
  - A probabilistic Faster R-CNN (camera and LiDAR), tested under Waymo rain and night, simulated
    fog, lens spatter and pixel dropout, and CADC snow.
  - Kernel densities of box uncertainties, fitted on normal data, label each detection as a likely
    true or false positive.
  - The ratio of predicted false to true positives, averaged over a sequence, rises as AP falls: 2.1×
    in rain with an obscured lens, and 1.4× in fog with −19% AP.
  - This is measured over sequences, not single images, and there is no clean-vs-degraded AUROC.
- **Yoo, Lee, Chung, Kim and Kwak (CVPR 2024).**
  - Test-time adaptation of Faster R-CNN, from COCO to COCO-C and SHIFT.
  - A "when to adapt" trigger, the KL divergence between Gaussians fitted to training features and to
    running test features, peaks at every domain change.
  - It works on streams and is never scored as a detector.
- **Khandal and Vidyarthi (COMSNETS 2022 workshop).** Confidence of YOLOv3, Faster R-CNN and SSD under
  noise, blur, fog, glare and snow. Exploratory; there is no monitor.
- **Yuhas and Easwaran (ITSC 2023).**
  - A separate β-VAE flags heavy rain in CARLA.
  - The in-distribution limit, 10% rain, is set where YOLOv7-tiny starts to degrade.
  - Code is public.

### Degradation estimated next to a detector, using degradation labels

- **Soiling.**
  - SoilingNet (Uřičář et al., ITSC 2019) and TiledSoilingNet (Das et al., ITSC 2020), both on
    WoodScape, put soiling heads on the encoder they share with a YOLOv2 detector. TiledSoilingNet
    trains its head on the *frozen* encoder.
  - Both are supervised on 76k–106k real soiled frames and scored by accuracy or per-class RMSE.
    Neither measures the effect on detection.
- **Soiling severity (Yang, Duan, Li and Zhang, *Sensors* 2026).**
  - Tile-level soiling classes, plus an image-level severity score smoothed over time.
  - A pedestrian detector's detection rate falls steadily with severity, from 88.7% at level 1 to
    56.9% at level 3.
  - Supervised on WoodScape plus synthetic sequences. It reports Spearman 0.79 and MAE.
- **Weather and light classifiers inside detectors, with the estimate evaluated.**
  - DTRDNet (Huang et al., *Sensors* 2024): clean, haze, rain or snow, read from the YOLOX encoder;
    99% precision.
  - AW-MoE (preprint 2026): a weather router for 3D detection, about 99% accurate.
  - Illumination-aware Faster R-CNN (Li et al., *Pattern Recognition* 2019): day/night gating for
    RGB-thermal pedestrian detection.
- **Weather and light used only internally**, with only mAP reported: IA-YOLO, GDIP, RDMNet, MTW-DETR
  and similar.
- **Reliability maps inside multi-sensor 3D detectors:** RAF (Park, Jeong and Yoon, ECCV 2026) and
  Seeing Through Fog (Bijelic et al., CVPR 2020).

### Driving-camera degradation without a detector

These papers classify the condition, but none links its output to a detector's accuracy:
- soiling: SoildNet, and "Let's Get Dirty" (WACV 2021);
- lens dirt and raindrops: Einecke et al., ITSC 2014;
- overexposure: ITSC 2018;
- weather and light level: Dhananjaya et al., ITSC 2021;
- fog visibility: Hautière et al., 2006;
- sun glare: a NeurIPS 2021 ML4AD workshop paper, IV 2021 and IV 2026.

The two IV glare papers were read at abstract level only. Beránek et al. report data leakage and
annotation errors in WoodScape's soiling set and release cleaned splits, which matters if WoodScape
is used.

### Borderline: the label is detector failure

- **Sezgin et al. (IV 2023).** A weather-chamber monitor built from image sharpness, brightness and
  contrast, labelled by YOLOv5 confidence and IoU. A useful IV precedent.
- **Dario et al. (arXiv 2026, vision-based aircraft landing).** Corruption-aware runtime monitors on
  YOLOv5, with 5 corruptions × 3 severities. The ground truth is detector failure (IoU < 0.7). Code
  is public.

### Resources worth borrowing

- **Degradation taxonomy.** Becker, Weiss, Hübner and Arens (arXiv 2026), "A causally grounded
  taxonomy for image degradation robustness evaluation". Severity is quantified by PSNR, SSIM and
  LPIPS.
- **DRIVE-C** (Aher, arXiv 2026). 12 degradations × 5 severities, as 600 driving clips aligned
  pixel-for-pixel with clean versions.
- **ImageNet-ES** (Baek et al., CVPR 2024). Real covariate shifts from sensor and environment
  settings; existing OOD detectors fail on them.
- **TCSR-Monitor** (Pham et al., arXiv 2026, surgical segmentation). Explicitly tests whether a
  monitor predicts failure rather than merely detecting corruption. That is the distinction between
  our separation and harm columns.

### Judging a monitor by the failures it catches

- **Guérin, Delmas, Ferreira and Guiochet (AAAI 2023), "Out-of-distribution detection is not all you
  need".**
  - Runtime monitors should be judged by how well they discard incorrect predictions, not by OOD
    detection. They call this "out-of-model-scope detection".
  - Classifiers only. The ICPR 2026 paper above adopts their setting but falls back to ID-vs-OOD
    separability.
- **Geifman, Uziel and El-Yaniv (ICLR 2019).**
  - Introduce AURC, the area under the risk–coverage curve of a selective classifier.
  - Our AURC applies it to detection, with per-image LRP as the risk.

### What this search did not find, and its limits

- **Not found:** a paper combining all of the following.
  - A frozen modern (DETR-family) detector trained on clean data only.
  - An image-level score with AUROC over the full corruption suite.
  - A per-image check that the score follows the harm to the detector.

  The ICPR 2026 paper has everything except the last.
- **Not found either:** a follow-up to Hashemi et al. on detection with more corruptions or with AUROC.
  Their RV 2024 follow-up uses classification; a 2025 one uses segmentation.
- **Limits:**
  - Semantic Scholar rate-limited almost every request, and the arXiv API throttled phrase queries.
  - MDPI, ScienceDirect, Springer, Wiley and IEEE Xplore pages were often blocked.
  - Read at title or metadata level only: Reddy Mure et al. (IV 2026), Yoneda et al. (IV 2021) and
    Johansen et al. (*Applied Sciences* 2023).

## References

Surnames only; links point to the page used for checking.

- Agnolucci, Galteri, Bertini (2025). Quality-aware image-text alignment for opinion-unaware image quality assessment (QualiCLIP). *arXiv preprint*. https://arxiv.org/abs/2403.11176
- Agnolucci, Galteri, Bertini, Del Bimbo (2024). ARNIQA: Learning distortion manifold for image quality assessment. *WACV*. https://arxiv.org/abs/2310.14918 (code: https://github.com/miccunifi/ARNIQA)
- Aher (2026). DRIVE-C: A controlled corruption dataset for autonomous driving. *arXiv preprint*. https://arxiv.org/abs/2605.09774
- Aher (2026). Safety-critical camera reliability monitoring for ADAS via degradation-aware uncertainty pattern analysis. *arXiv preprint*. https://arxiv.org/abs/2605.05439
- Arnal, Hensel, Carrière, Lacombe, Kurihara, Ike, Chazal (2024). MAGDiff: Covariate data set shift detection via activation graphs of neural networks. *TMLR*. https://arxiv.org/abs/2305.13271
- Baek, Park, Kim, Kim (2024). Unexplored faces of robustness and out-of-distribution: Covariate shifts in environment and sensor domains (ImageNet-ES). *CVPR*. https://arxiv.org/abs/2404.15882
- Becker, Bayer, Hübner, Arens (2026). Operational readiness for object detection. *ICPR 2026*, LNCS 16814, pp. 121–135 (Springer, 2027). https://doi.org/10.1007/978-3-032-31654-7_9 (read in full)
- Becker, Weiss, Hübner, Arens (2026). A causally grounded taxonomy for image degradation robustness evaluation. *arXiv preprint*. https://arxiv.org/abs/2605.15906
- Becker, Weiss, Hübner, Arens (2026). Self-aware object detection via degradation manifolds. *arXiv preprint*. https://arxiv.org/abs/2602.18394
- Beemelmanns, Nekrasov, Vilceanu, et al. (2026). Query2Uncertainty: Robust uncertainty quantification and calibration for 3D object detection under distribution shift. *CVPR*. https://arxiv.org/abs/2605.05328
- Beniwal, Mantini, Shah (2022). Image quality assessment using deep features for object detection. *VISAPP*. https://www.scitepress.org/Papers/2022/109170/109170.pdf
- Beránek, Diviš, Gruber (2025). Soiling detection for advanced driver assistance systems. *arXiv preprint* (reported as ICMV 2024; venue not checked). https://arxiv.org/abs/2511.09740
- Bhuyan (2026). Tippett-minimum fusion of representation-space diffusion models for multi-encoder out-of-distribution detection. *arXiv preprint*. https://arxiv.org/abs/2605.20502
- Bianco, Celona, Napoletano (2021). Disentangling image distortions in deep feature space. *Pattern Recognition Letters*, 148. https://arxiv.org/abs/2002.11409
- Caetano, Viviers, Zavala-Mondragón, et al. (2025). DisCoPatch: Taming adversarially-driven batch statistics for improved out-of-distribution detection. *ICCV*. https://arxiv.org/abs/2501.08005 (code: https://github.com/caetas/DisCoPatch)
- Chen, Mo (2022). IQA-PyTorch: PyTorch toolbox for image quality assessment (`pyiqa`). *Software*. https://github.com/chaofengc/IQA-PyTorch
- Chen, Mo, Hou, et al. (2024). TOPIQ: A top-down approach from semantics to distortions for image quality assessment. *IEEE TIP*, 33. https://arxiv.org/abs/2308.03060
- Dario, Chenevier, Delmas, Guérin, Guiochet (2026). Unifying runtime monitoring approaches for safety-critical machine learning: Application to vision-based landing. *arXiv preprint* (reported as ICPR 2026). https://arxiv.org/abs/2604.26411
- Das, Křížek, Sistu, et al. (2020). TiledSoilingNet: Tile-level soiling detection on automotive surround-view cameras using coverage metric. *IEEE ITSC*. https://arxiv.org/abs/2007.00801
- Dhananjaya, Kumar, Yogamani (2021). Weather and light level classification for autonomous driving: Dataset, baseline and active learning. *IEEE ITSC*. https://arxiv.org/abs/2104.14042
- Di Salvo, Doerrich, Ledig (2026). MaRS: Robust out-of-distribution detection via Mahalanobis residual scoring. *arXiv preprint* (MICCAI 2026 according to the code repository). https://arxiv.org/abs/2606.22649 (code: https://github.com/francescodisalvo05/mars)
- Dremin, Kozhemyakov, Molodetskikh, et al. (2024). Machine vision-aware quality metrics for compressed image and video assessment. *arXiv preprint*. https://arxiv.org/abs/2411.06776
- Du, Gozum, Ming, Li (2022). SIREN: Shaping representations for detecting out-of-distribution objects. *NeurIPS*. https://proceedings.neurips.cc/paper_files/paper/2022/hash/804dbf8d3b8eee1ef875c6857efc64eb-Abstract-Conference.html
- Du, Wang, Cai, Li (2022). VOS: Learning what you don't know by virtual outlier synthesis. *ICLR*. https://arxiv.org/abs/2202.01197
- Feng, Harakeh, Waslander, Dietmayer. A review and comparative study on probabilistic object detection in autonomous driving. *IEEE T-ITS*. https://arxiv.org/abs/2011.10671
- Ferreira, Arlat, Guiochet, Waeselynck (2021). Benchmarking safety monitors for image classifiers with machine learning. *IEEE PRDC*. https://arxiv.org/abs/2110.01232
- Gebhart, Schrater (2017). Adversary detection in neural networks via persistent homology. *arXiv preprint*. https://arxiv.org/abs/1711.10056
- Geifman, Uziel, El-Yaniv (2019). Bias-reduced uncertainty estimation for deep neural classifiers (introduces AURC). *ICLR*. https://openreview.net/forum?id=SJfb5jCqKm
- Girrbach, Christensen, Winther, Akata, Koepke (2023). Addressing caveats of neural persistence with deep graph persistence. *TMLR*. https://arxiv.org/abs/2307.10865
- Goibert, Ricatte, Dohmatob (2022). An adversarial robustness perspective on the topology of neural networks. *NeurIPS ML Safety Workshop*. https://arxiv.org/abs/2211.02675
- Guérin, Delmas, Ferreira, Guiochet (2023). Out-of-distribution detection is not all you need. *AAAI*, 37(12), 14829–14837. https://arxiv.org/abs/2211.16158
- Hall, Dayoub, Skinner, et al. (2020). Probabilistic object detection: Definition and evaluation. *WACV*. https://arxiv.org/abs/1811.10800
- Harakeh, Waslander (2021). Estimating and evaluating regression predictive uncertainty in deep object detectors. *ICLR*. https://arxiv.org/abs/2101.05036
- Hashemi, Křetínský, Rieder, Schmidt (2023). Runtime monitoring for out-of-distribution detection in object detection neural networks. *FM 2023*, LNCS. https://arxiv.org/abs/2212.07773 (read in full)
- Hendrycks, Dietterich (2019). Benchmarking neural network robustness to common corruptions and perturbations. *ICLR*. https://arxiv.org/abs/1903.12261
- Hendrycks, Gimpel (2017). A baseline for detecting misclassified and out-of-distribution examples in neural networks. *ICLR*. https://arxiv.org/abs/1610.02136
- Heng, Soh (2025). Detecting covariate shifts with vision-language foundation models. *ICLR 2025 Workshop on Foundation Models in the Wild*. https://mlanthology.org/iclrw/2025/heng2025iclrw-detecting/
- Hildebrand, Brown, Brown, Waslander (2023). Assessing distribution shift in probabilistic object detection under adverse weather. *IEEE Access*, 11. https://doi.org/10.1109/ACCESS.2023.3270447
- Huang, Wang, Teng, He, Chen (2024). Degradation type-aware image restoration for effective object detection in adverse weather (DTRDNet). *Sensors*, 24. https://pmc.ncbi.nlm.nih.gov/articles/PMC11478636/
- Jaeger, Lüth, Klein, Bungert (2023). A call to reflect on evaluation practices for failure detection in image classification. *ICLR*. https://arxiv.org/abs/2211.15259
- Ke, Wang, Wang, Milanfar, Yang (2021). MUSIQ: Multi-scale image quality transformer. *ICCV*. (CVF open access)
- Kees, Hoemann, Köster, Hallerbach (2026). Image quality dependent degradation for AI systems. *arXiv preprint*. https://arxiv.org/abs/2607.25736
- Keser, Knoll (2026). LFA: Layer feature attention for run-time introspection of 2D object detectors in automated driving. *arXiv preprint*. https://arxiv.org/abs/2606.00372
- Keser, Orhan, Amini-Naieni, Schwalbe, Knoll, Rottmann (2025). Benchmarking vision foundation models for input monitoring in autonomous driving. *BMVC*. https://arxiv.org/abs/2501.08083
- Khandal, Vidyarthi (2022). Exploring credibility scoring metrics of perception systems for autonomous driving. *COMSNETS 2022 workshop*. https://arxiv.org/abs/2112.11643
- Kuzucu, Oksuz, Sadeghi, Dokania (2024). On calibration of object detectors: Pitfalls, evaluation and baselines. *ECCV*. https://arxiv.org/abs/2405.20459
- Lacombe, Ike, Carrière, Chazal, Glisse, Umeda (2021). Topological uncertainty: Monitoring trained neural networks through persistence of activation graphs. *IJCAI*. https://arxiv.org/abs/2105.04404
- Lee, AlRegib (2020). Gradients as a measure of uncertainty in neural networks. *IEEE ICIP*. https://arxiv.org/abs/2008.08030
- Lee, Lee, Lee, Shin (2018). A simple unified framework for detecting out-of-distribution samples and adversarial attacks. *NeurIPS*. https://arxiv.org/abs/1807.03888
- Lee, Lehman, Prabhushankar, AlRegib (2023). Probing the purview of neural networks via gradient analysis. *IEEE Access*, 11. https://arxiv.org/abs/2304.02834
- Li, Liu, Hu, Wu, Lv, Peng (2022). All-in-one image restoration for unknown corruption (AirNet). *CVPR*. https://github.com/XLearning-SCU/2022-CVPR-AirNet
- Liao, Wu, Shi, et al. (2026). Beyond cosine similarity: Magnitude-aware CLIP for no-reference image quality assessment (MA-CLIP). *AAAI*. https://arxiv.org/abs/2511.09948 (code: https://github.com/zhix000/MA-CLIP)
- Liu, Wang, Owens, Li (2020). Energy-based out-of-distribution detection. *NeurIPS*. https://arxiv.org/abs/2010.03759
- Luo, Gustafsson, Zhao, Sjölund, Schön (2024). Controlling vision-language models for multi-task image restoration (DA-CLIP). *ICLR*. https://arxiv.org/abs/2310.01018
- Ma, Duanmu, Wu, Wang, Yong, Li, Zhang (2017). Waterloo Exploration Database: New challenges for image quality assessment models. *IEEE TIP*, 26(2). https://kedema.org/paper/17_TIP_EXPLORATION.pdf
- Ma, Liu, Zhang, Duanmu, Wang, Zuo (2018). End-to-end blind image quality assessment using deep neural networks (MEON). *IEEE TIP*, 27(3). https://kedema.org/paper/18_TIP_MEON.pdf
- Madhusudana, Birkbeck, Wang, Adsumilli, Bovik (2022). Image quality assessment using contrastive learning (CONTRIQUE). *IEEE TIP*, 31. https://arxiv.org/abs/2110.13266
- Manoharan, Yin, Helm, Cheng (2025). Cumulative consensus score: Label-free and model-agnostic evaluation of object detectors in deployment. *arXiv preprint*. https://arxiv.org/abs/2509.12871
- Michaelis, Mitzkus, Geirhos, Rusak, Bringmann, Ecker, Bethge, Brendel (2019). Benchmarking robustness in object detection: Autonomous driving when winter is coming. *NeurIPS 2019 Workshop on Machine Learning for Autonomous Driving*. https://arxiv.org/abs/1907.07484 (package: https://github.com/bethgelab/imagecorruptions)
- Mittal, Soundararajan, Bovik (2013). Making a "completely blind" image quality analyzer (NIQE). *IEEE Signal Processing Letters*, 20.
- Munir, Khan, Khan, Ali, Khan (2023). Cal-DETR: Calibrated detection transformer. *NeurIPS*. https://arxiv.org/abs/2311.03570
- Munir, Khan, Khan, Khan (2023). Bridging precision and confidence: A train-time loss for calibrating object detection. *CVPR*. https://arxiv.org/abs/2303.14404
- Munir, Khan, Sarfraz, Ali (2022). Towards improving calibration in object detection under domain shift. *NeurIPS*. https://arxiv.org/abs/2209.07601
- Oksuz, Joy, Dokania (2023). Towards building self-aware object detectors via reliable uncertainty quantification and calibration. *CVPR*. https://arxiv.org/abs/2307.00934
- Ovadia, Fertig, Ren, et al. (2019). Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. *NeurIPS*. https://arxiv.org/abs/1906.02530
- Park, Jeong, Yoon (2026). RAF: Reliability-aware fusion of camera, LiDAR, and 4D RADAR for robust 3D object detection in adverse weather. *ECCV*. https://arxiv.org/abs/2607.04587
- Park, Sobolewski, Azizan (2026). Uncertainty quantification in detection transformers: Object-level calibration and image-level reliability. *IEEE TPAMI*. https://arxiv.org/abs/2412.01782
- Pham, Cao, Huynh, et al. (2026). Beyond uncertainty: Generalizable failure monitoring for surgical segmentation under acquisition degradation (TCSR-Monitor). *arXiv preprint*. https://arxiv.org/abs/2608.16748
- Pollano, Chaudhuri, Simmons (2024). Detecting out-of-distribution text using topological features of transformer-based language models. *IJCAI 2024 AISafety Workshop (CEUR-WS Vol-3856)*. https://arxiv.org/abs/2311.13102
- Qutub, Paulitsch, Scholl, et al. (2024). Situation monitor: Diversity-driven zero-shot out-of-distribution detection using budding ensemble architecture for object detection. *CVPR Workshops (SAIAD)*. https://arxiv.org/abs/2406.03188
- Rabanser, Günnemann, Lipton (2019). Failing loudly: An empirical study of methods for detecting dataset shift. *NeurIPS*. https://arxiv.org/abs/1810.11953
- Rahman, Sünderhauf, Dayoub (2021a). Per-frame mAP prediction for continuous performance monitoring of object detection during deployment. *WACV Workshops*. https://arxiv.org/abs/2009.08650
- Rahman, Sünderhauf, Dayoub (2021b). Online monitoring of object detection performance during deployment. *IROS*. https://arxiv.org/abs/2011.07750
- Ramesh, Wang, Islam (2025). HiRQA: Hierarchical ranking and quality alignment for opinion-unaware image quality assessment. *Machine Vision and Applications* (accepted). https://arxiv.org/abs/2508.15130
- Rieck, Togninalli, Bock, et al. (2019). Neural persistence: A complexity measure for deep neural networks using algebraic topology. *ICLR*. https://arxiv.org/abs/1812.09764
- Saha, Mishra, Bovik (2023). Re-IQA: Unsupervised learning for image quality assessment in the wild. *CVPR*. https://arxiv.org/abs/2304.00451
- Sezgin, Vriesman, Steinhauser, Lugner, Brandmeier (2023). Safe autonomous driving in adverse weather: Sensor evaluation and performance monitoring. *IEEE IV*. https://arxiv.org/abs/2305.01336
- Su, Yan, Zhu, et al. (2020). Blindly assess image quality in the wild guided by a self-adaptive hyper network (HyperIQA). *CVPR*. (code: https://github.com/SSL92/hyperIQA)
- Sun, Ming, Zhu, Li (2022). Out-of-distribution detection with deep nearest neighbors. *ICML*. https://arxiv.org/abs/2204.06507
- Tian, Hsu, Shen, Jin, Kira (2021). Exploring covariate and concept shift for detection and calibration of out-of-distribution data. *arXiv preprint* (short version at NeurIPS 2021 DistShift Workshop). https://arxiv.org/abs/2110.15231
- Uřičář, Křížek, Sistu, Yogamani (2019). SoilingNet: Soiling detection on automotive surround-view cameras. *IEEE ITSC*. https://arxiv.org/abs/1905.01492
- Venkataramanan, Facktor, Gupta, Bovik (2022). Assessing the impact of image quality on object-detection algorithms. *IS&T Electronic Imaging*. https://doi.org/10.2352/EI.2022.34.9.IQSP-334
- Viviers, Valiuddin, Caetano, et al. (2024). Can your generative model detect out-of-distribution covariate shift? *ECCV 2024 Workshops*. https://arxiv.org/abs/2409.03043
- Wang, Chan, Loy (2023). Exploring CLIP for assessing the look and feel of images (CLIP-IQA). *AAAI*. https://arxiv.org/abs/2207.12396
- Wang, Wang, Dong, et al. (2021). Unsupervised degradation representation learning for blind super-resolution (DASR). *CVPR*. https://arxiv.org/abs/2104.00416
- Wang, Zhang, Lin (2025). Image quality assessment for machines: Paradigm, large-scale database, and models (MIQA). *arXiv preprint*. https://arxiv.org/abs/2508.19850
- Wilson, Fischer, Dayoub, Miller, Sünderhauf (2023). SAFE: Sensitivity-aware features for out-of-distribution object detection. *ICCV*. https://arxiv.org/abs/2208.13930
- Wu, He, Cheng, Huang, Bensalem (2025). Revisiting out-of-distribution detection in real-time object detection: From benchmark pitfalls to a new mitigation paradigm. *IEEE TPAMI* (accepted). https://arxiv.org/abs/2503.07330
- Wu, Zhang, Zhang, et al. (2024). Q-Align: Teaching LMMs for visual scoring via discrete text-defined levels. *ICML*. https://arxiv.org/abs/2312.17090
- Yang, Duan, Li, Zhang (2026). A static-to-temporal framework for interpretable camera lens soiling severity estimation in autonomous driving. *Sensors*, 26(11), 3533. https://doi.org/10.3390/s26113533
- Yang, Popović, Wiederer, et al. (2026). GroupEnsemble: Efficient uncertainty estimation for DETR-based object detection. *arXiv preprint*. https://arxiv.org/abs/2603.01847
- Yang, Wang, Chen, Dai, Zheng (2024). Bounding box stability against feature dropout reflects detector generalization across environments. *ICLR*. https://arxiv.org/abs/2403.13803
- Yang, Wu, Shi, et al. (2022). MANIQA: Multi-dimension attention network for no-reference image quality assessment. *CVPR Workshops*. https://arxiv.org/abs/2204.08958
- Yang, Zhang, Russakovsky (2024). ImageNet-OOD: Deciphering modern out-of-distribution detection algorithms. *ICLR*. https://arxiv.org/abs/2310.01755
- Yang, Zhou, Liu (2023). Full-spectrum out-of-distribution detection. *IJCV*, 131. https://arxiv.org/abs/2204.05306
- Yatbaz, Dianati, Koufos, Woodman (2024). Run-time introspection of 2D object detection in automated driving systems using learning representations. *IEEE Transactions on Intelligent Vehicles*. https://arxiv.org/abs/2403.01172
- Yatbaz, Dianati, Woodman (2024). Introspection of DNN-based perception functions in automated driving systems: State-of-the-art and open research challenges. *IEEE T-ITS*, 25(2). https://wrap.warwick.ac.uk/id/eprint/179419/
- Yeh, Yang (2025). Uncertainty of network topology with applications to out-of-distribution detection. *arXiv preprint*. https://arxiv.org/abs/2511.18813
- Yoo, Kwon, Hwang, Lee (2025). Automated model evaluation for object detection via prediction consistency and reliability. *ICCV*. https://arxiv.org/abs/2508.12082
- Yoo, Lee, Chung, Kim, Kwak (2024). What, how, and when should object detectors update in continually changing test domains? *CVPR*. https://arxiv.org/abs/2312.08875
- Yuhas, Easwaran (2023). Co-design of out-of-distribution detectors for autonomous emergency braking systems. *IEEE ITSC*. https://arxiv.org/abs/2307.13419
- Zhang, Ma, Yan, Deng, Wang (2020). Blind image quality assessment using a deep bilinear convolutional neural network (DBCNN). *IEEE TCSVT*, 30. https://arxiv.org/abs/1907.02665
- Zhang, Yang, Wang, et al. (2024). OpenOOD v1.5: Enhanced benchmark for out-of-distribution detection. *DMLR*. https://arxiv.org/abs/2306.09301
- Zhang, Zhai, Wei, Yang, Ma (2023). Blind image quality assessment via vision-language correspondence: A multitask learning perspective (LIQE). *CVPR*. https://arxiv.org/abs/2303.14968
- Zimmermann, Holzbach, Lerch (2026). Knowledge-guided failure prediction: Detecting when object detectors miss safety-critical objects. *CVPR Workshops (SAIAD)*. https://arxiv.org/abs/2603.25499


## Appendix: reading list if we only care about degradation

If new-class OOD (unknown objects) is out of scope and only image degradation matters, these are
the papers to read, roughly in order.

*Update, 29 September 2026:* read Becker, Bayer, Hübner and Arens, "Operational Readiness for Object
Detection" (ICPR 2026) first. It is the closest published work. Then read Hashemi et al. (FM 2023),
now our fifth baseline. Both are summarised in Section 12.

1. Becker et al. 2026, "Self-Aware Object Detection via Degradation Manifolds" (preprint, arXiv 2602.18394). This is the closest paper.
   - The task is purely degradation. Positives are corrupted COCO val images at severities 1–5 against clean ones, scored with AUROC, and there are no unknown objects.
   - They also test real bad weather (Seeing Through Fog, BDD100K rain) and other datasets (KITTI, VisDrone, DETRAC, UAVDT, FLIR).
   - Their baselines are exactly the kinds of scores we'd compare against:
     - detector confidence and entropy
     - normalizing flows on detector features
     - image-quality models (ARNIQA, CLIPIQA, MANIQA, QualiCLIP)
2. Park, Sobolewski & Azizan 2026 (TPAMI). Use it for the frozen-DETR-queries part.
   - They pool a frozen DETR's queries into one image score and test it on Foggy Cityscapes, which is a degradation.
   - Their target is per-image AP, not "is it degraded?", so read it for the query-pooling idea rather than the evaluation.
3. SAOD (Oksuz et al., CVPR 2023). Read only Sections 3–4, for two things.
   - The top-3 confidence aggregation, which is the standard image-level detector baseline.
   - The corrupted set Obj45K-C: 15 corruptions at severities 1, 3, 5.
   - Skip its main task, because it counts corrupted images as data to accept.
4. Lee & AlRegib 2020 (ICIP) and Lee et al. 2023 (IEEE Access).
   - Clean vs corrupted CIFAR-10-C, reported per corruption and level; the 2023 paper covers all 19
     families.
   - The closest in reporting style to ours, but their detector is trained on corrupted examples.
5. Viviers et al. 2024 (ECCV Workshops). Despite "OOD" in the title, it is only about degradation.
   - CIFAR-10-C (19 × 5) and ImageNet200-C (15 × 5), with AUROC and FPR95 per severity.
   - A good template for the evaluation.
6. Tian et al. 2021 (preprint). Splits an OOD score into a covariate (degradation) part and a concept
   (new-class) part. Useful for arguing why confidence-based scores mix the two.
7. ImageNet-OOD (Yang et al., ICLR 2024), covariate-shift part only. Blur and noise push
   feature-based scores in opposite directions, which matters when reading per-family results.
8. MAGDiff (TMLR 2024) and DGP (TMLR 2023). The only activation-graph or topological papers aimed at
   degradation.
   - Both decide per batch, not per image, with small classifiers.
   - DGP's "compare with all classes" finding supports our class-agnostic bank.
9. Topological Uncertainty (Lacombe et al., IJCAI 2021), Section 4.3 on shift only. It is
   descriptive, with no AUROC.
10. Hendrycks & Dietterich 2019 (ImageNet-C) and Michaelis et al. 2019 (COCO-C and
    `imagecorruptions`). For corruption definitions, the 15 + 4 split and severity conventions.
11. Image-quality assessment (Section 6):
    - ARNIQA: Becker et al.'s training recipe and their best detector-free baseline.
    - NIQE: the classical "clean reference plus distance" method.
    - The Waterloo D-test: the IQA field's clean-vs-damaged evaluation.
    - Bianco et al.: frozen features separate damage types, and early layers work best.
    - MIQA: quality for machines on COCO val. Human-perceived quality is a weak proxy for detector
      quality.

Skip for this purpose:
- new-class OOD (VOS, SIREN, SAFE, the Qutub situation monitor, most of OpenOOD);
- failure predictors (Rahman, Yatbaz, Keser), whose label is detector accuracy, not degradation;
- calibration and robustness papers (Cal-DETR, TCD, BPC, Kuzucu, Harakeh & Waslander);
- dataset-level accuracy estimators (BoS, PCR).

SIREN is worth one line: kNN distance works on Deformable-DETR decoder embeddings, while
Mahalanobis distance is at chance.
