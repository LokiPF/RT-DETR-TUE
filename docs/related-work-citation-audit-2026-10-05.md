# IV 2027 Yuchen related work citation audit

Reviewed on 5 October 2026 against `IV_2027_Yuchen/chapters/03_RelatedWorks.tex` and `IV_2027_Yuchen/root.bib`. This is a review report; the manuscript and bibliography have not been edited.

The section has 28 active citation keys, all present in the bibliography, plus one commented citation. Most references are relevant, but several descriptions need correction or qualification. The most consequential issues concern the image-monitoring taxonomy, the claim that all image monitors require separate feature extraction, and the scope of the final comparison with prior feature monitors.

## Scope and evidence

The current LaTeX source is authoritative for this review; an existing compiled PDF may be stale. The review covers every active citation in Related Work, its surrounding claim, and the final uncited comparison. The commented Geissler sentence is checked separately. Unused bibliography entries and citations moved into comments in Methods are outside scope.

Evidence comes from original proceedings papers, author manuscripts, publisher records, and institutional repositories. Page references identify the linked version; author manuscripts can have different pagination from published papers. A supported verdict means the specific claim is supported, not that a monitor guarantees safe driving. Access limitations are stated explicitly.

Reviewed source SHA-256: `2434b75341b3bd52260e22579edcf81bf16958ed897a0505136dc711da8a2517`.

## Opening paragraph

All entries below concern line 4 of the reviewed LaTeX file.

| Citation | Verdict | Evidence and action |
| --- | --- | --- |
| `iso21448` | Supported within the stated scope | ISO 21448 addresses hazards from functional and performance insufficiencies in intended vehicle functions, including sensor-dependent situational awareness. The motivation is appropriate. Do not imply that the standard prescribes this particular runtime monitor. **Access:** checked the official public scope, not the complete paid standard. [ISO scope and publication record](https://committee.iso.org/cms/live/live/en/sites/isoorg/contents/data/standard/07/74/77490.html?browse=tc). |
| `rahman2021survey` | Supported | Section 1, author PDF pp. 1–2, explains why test-set performance does not establish deployment reliability and motivates runtime monitoring and fallback. Section 3 also supports organizing monitors by their location in the perception pipeline; it allows combinations of locations. [Original author manuscript](https://arxiv.org/pdf/2101.01364); [published metadata](https://digital.library.adelaide.edu.au/items/f9977cac-3c22-4a11-a38b-9e486d48a730). |
| `yatbaz2024survey` | Supported | Introduction, printed pp. 1–2 of the accepted manuscript, connects incomplete training coverage to perception errors and the need for additional monitoring. It supports the motivation, not a guarantee of safe stopping. [Accepted manuscript](https://wrap.warwick.ac.uk/id/eprint/179419/1/WRAP-Introspection-DNN-based-perception-functions-driving-state-art-open-research-23.pdf); [published metadata](https://wrap.warwick.ac.uk/id/eprint/179419/). |
| `du2022vos` | Supported with narrower terminology | VOS studies detecting objects from categories outside the detector's training label space; see the introduction and problem setup. This supports **semantic OOD object detection**, rather than a universal definition of all OOD detection. [Original paper, ICLR 2022 version](https://arxiv.org/pdf/2202.01197v2). |
| `du2022siren` | Supported with narrower terminology | Introduction and Section 3 concern unknown-category objects and object-level OOD scores. The cited example is correct; qualify the sentence as semantic OOD object detection. [NeurIPS original paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/804dbf8d3b8eee1ef875c6857efc64eb-Paper-Conference.pdf). |
| `feng2022review` | Supported | Sections II–III survey predictive uncertainty and probabilistic object detection, including category and localization uncertainty. The paper supports uncertainty as information about prediction reliability, not an automatic correctness certificate. [Author manuscript](https://arxiv.org/pdf/2011.10671); [publication DOI](https://doi.org/10.1109/TITS.2021.3096854). |
| `harakeh2021estimating` | Supported | Sections 3–4 estimate and evaluate bounding-box predictive distributions, calibration, and uncertainty under dataset shift. This is an appropriate citation for object-level predictive uncertainty. Its regression energy scoring rule is distinct from Liu et al.'s logit-derived OOD energy score. [Original ICLR manuscript](https://arxiv.org/pdf/2101.05036). |
| `rahman2021perframe` | Needs a scope qualification | Section 3, published p. 154, trains a binary monitor to predict whether **per-frame mAP** falls below a threshold using pooled backbone activations. It does not estimate the reliability of each individual detection. Change the umbrella wording to cover both predictions and frames. [Final WACV workshop paper](https://openaccess.thecvf.com/content/WACV2021W/AVV/papers/Rahman_Per-Frame_mAP_Prediction_for_Continuous_Performance_Monitoring_of_Object_Detection_WACVW_2021_paper.pdf). |
| `yatbaz2024runtime` | Needs a scope qualification | Figure 1 and Section III explicitly address **frame-level** error prediction from processed backbone activations; training labels derive from thresholded per-frame mAP. Broaden “a given prediction” to “individual detections or whole frames.” [Accepted full paper](https://wrap.warwick.ac.uk/id/eprint/184585/1/WRAP-run-time-introspection-2D-object-detection-automated-driving-systems-using-learning-representations-2024.pdf); [publication metadata](https://wrap.warwick.ac.uk/id/eprint/184585/). |
| `yang2024generalized` | Supported with terminology qualification | Section 2, author PDF p. 3, distinguishes changes in input distribution with a fixed label space from semantic changes. Your corruption examples fit its usage. “Content stays familiar” is an informal description, not a requirement that scene content be identical. The three opening monitoring objectives can overlap. [Full author manuscript](https://arxiv.org/pdf/2110.11334); [journal record](https://link.springer.com/article/10.1007/s11263-024-02117-4). |
| `hendrycks2019benchmarking` | Supported | Section 4.1 and Figure 1 define ImageNet-C corruption families, including noise, blur, weather, and digital corruption. Appropriate support for the listed examples; it is a robustness benchmark, not a proposed runtime detector. [ICLR paper](https://arxiv.org/pdf/1903.12261). |
| `michaelis2019benchmarking` | Supported | Section 3 and Figure 3 transfer common-corruption evaluation to object detection using PASCAL-C, COCO-C, and Cityscapes-C. Appropriate support for corruptions affecting object detectors. The bibliography's 2019 arXiv form is legitimate; a NeurIPS 2019 workshop version also exists. [Full author manuscript](https://arxiv.org/pdf/1907.07484); [workshop paper](https://ml4ad.github.io/files/papers/Benchmarking%20Robustness%20in%20Object~Detection%3A%20Autonomous%20Driving%20when%20Winter%20is%20Coming.pdf). |

Suggested opening clarification:

> Semantic OOD object detection identifies objects from classes outside the detector's training label space. Uncertainty estimation and failure prediction assess the reliability of individual detections or whole frames. Here, we focus on input corruptions that alter image appearance while retaining the familiar object categories. These objectives can overlap; we organize the following methods by whether they use predictions, image information, or internal features.

The prediction/image/features organization is useful, but should not imply mutually exclusive architectures: some methods combine these sources.

## Monitoring the prediction

All entries below concern line 8.

| Citation | Verdict | Evidence and action |
| --- | --- | --- |
| `hendrycks2017baseline` | Supported as motivation | Section 3, PDF p. 3, uses maximum softmax probability to identify classification errors and OOD examples. The paper also cautions that softmax is not automatically calibrated confidence. Your “Ideally” wording is defensible; avoid asserting that unfamiliar inputs necessarily yield low confidence. [Original paper](https://arxiv.org/pdf/1610.02136). |
| `oksuz2023saod` | Supported for the selected baseline | Section 4/Table 2, p. 9266, aggregates the three lowest uncertainties. With its chosen uncertainty of one minus confidence, these are the three most confident detections. Section 6, p. 9269, selects mean(top-3). **Do not mark the current top-three statement as incorrect.** Calling this the “SAOD baseline” is more precise. [CVPR paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Oksuz_Towards_Building_Self-Aware_Object_Detectors_via_Reliable_Uncertainty_Quantification_and_CVPR_2023_paper.pdf). |
| `park2026uncertainty` | Needs aggregation precision | Section VI-B, author v4 p. 11: ContrastiveConf subtracts a scaled mean maximum foreground confidence of discarded predictions from the corresponding mean of retained predictions. The retained set uses postprocessing selected by validation OCE. “Margin” is broadly descriptive but hides the weighting and averaging. This is an image-reliability score, not by itself evidence of corruption detection. [Full author text](https://arxiv.org/html/2412.01782v4#S6.SS2). |
| `liu2020energy` | Score correct; clarify task scope | Equation (4), p. 3, defines energy from classifier logits; Section 4.1 evaluates classifiers. It does not establish an image-level aggregation for bounding-box object detectors or detector corruption monitoring. Introduce it explicitly as a related OOD classification score. [NeurIPS paper](https://papers.neurips.cc/paper/2020/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf). |
| `hendrycks2022scaling` | Score correct; clarify task scope | Section 3, p. 8763, defines negative maximum logit as an anomaly score. Sections 4–5 study multi-label classification and semantic anomaly segmentation, not bounding-box detection. Use that scope rather than implying that this citation establishes the score for detector-image monitoring. [ICML paper](https://proceedings.mlr.press/v162/hendrycks22a/hendrycks22a.pdf). |

Suggested replacements for the two places needing precision:

> ContrastiveConf subtracts a scaled mean foreground confidence of discarded DETR predictions from that of retained predictions.

> Related OOD scores computed from logits include energy for classification and maximum-logit scoring for classification and anomaly segmentation.

Keep the existing corresponding citation keys on these sentences.

## Monitoring the image

All entries below concern line 11. Full methods text was obtained for all eight papers.

| Citation | Verdict | Evidence and action |
| --- | --- | --- |
| `mittal2013niqe` | Essentially supported | Section II-A–E, author PDF pp. 2–3, models locally normalized luminance and neighboring-coefficient statistics, then compares fitted distributions. It needs pristine natural images, not distorted examples or opinion scores. “Spatial natural-scene statistics” is more precise than “local contrast statistics.” [Author-hosted paper](https://utw10503.utweb.utexas.edu/publications/2013/mittal2013.pdf). |
| `caetano2025discopatch` | Mechanism broadly correct; taxonomy incorrect | Sections 3.1–3.3, pp. 2901–2903, train on real ID patches and negative examples from both VAE reconstructions and randomly generated patches. Patch discriminator scores are aggregated per image. This is an **OOD detector**, evaluated for covariate and semantic shifts, rather than an IQA predictor of human quality. Its generated negatives also complicate the clean/degraded dichotomy. [Official ICCV paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Caetano_DisCoPatch_Taming_Adversarially-driven_Batch_Statistics_for_Improved_Out-of-Distribution_Detection_ICCV_2025_paper.pdf). |
| `uricar2019soilingnet` | Local claim supported; final architectural claim contradicted | Sections II-C/III-A use annotated opaque and transparent lens-soiling examples. Crucially, Section III-C, PDF p. 4/Figure 4, shares an encoder with object detection and segmentation; Section IV/Table II evaluates multitask training. This directly contradicts the claim that all listed approaches leave detector features unused and run a separate model. [Original author paper](https://arxiv.org/pdf/1905.01492). |
| `pavlic2012image` | Supported | Sections IV-B–C and V, PDF pp. 4–6, classify daytime fog using spectral features, PCA, and an SVM trained on labeled foggy/fog-free driving images. This is condition classification rather than scalar perceptual IQA. [Institutional full paper](https://mediatum.ub.tum.de/doc/1137870/document.pdf). |
| `dhananjaya2021weather` | Local claim supported; universal architecture claim too strong | Section III, PDF p. 3, supports annotated rain/snow classification and active learning. Section III-B discusses adding the task to an existing perception model using a shared encoder. Rain/snow wording is safe; the source is less consistent about its complete class list. [Original author paper](https://arxiv.org/pdf/2104.14042). |
| `mittal2012brisque` | Supported but supervision omitted | Sections III-B/IV-A, author PDF pp. 7–8, train an SVR from spatial natural-scene statistics to **human quality scores**, using synthetically distorted LIVE images and DMOS labels. Clarify that distortion exposure alone does not supply its training target. [Author-hosted paper](https://live.ece.utexas.edu/research/quality/brisque_journal.pdf). |
| `agnolucci2024arniqa` | Needs distinction between learning stages | Sections 3.1–3.2 learn representations using synthetic distortions and self-supervision. Sections 4.1–4.3, p. 194, then fit a ridge regressor to human MOS with a frozen encoder. Regression datasets include authentic as well as synthetic distortions. Do not describe the whole pipeline as simply learning quality scores from synthetic distortions. [Official WACV paper](https://openaccess.thecvf.com/content/WACV2024/papers/Agnolucci_ARNIQA_Learning_Distortion_Manifold_for_Image_Quality_Assessment_WACV_2024_paper.pdf). |
| `wang2023clipiqa` | Zero-shot scoring correct; “learns from neither” misleading | Sections 1/2.1–2.2 explain CLIP image–text pretraining and prompt/image similarity scoring. The base CLIP-IQA needs no additional task-specific IQA training; it is not untrained. CLIP-IQA+ is a separate prompt-trained variant. Repair the malformed closing quotation on “Bad photo.” [Full author paper](https://arxiv.org/pdf/2207.12396); [AAAI record](https://ojs.aaai.org/index.php/AAAI/article/view/25353). |

The clean/degraded/neither taxonomy conflates training data, supervision, and pretraining. Condition classifiers, perceptual IQA models, and OOD detectors can remain in this subsection, but should be introduced as distinct ways of analyzing images.

Suggested opening:

> Image-based monitoring includes perceptual quality assessment, condition classification, and OOD detection. These approaches differ in their training data and supervision: some model pristine-image statistics, some use labeled conditions or quality scores, and others apply pretrained representations without task-specific training.

Suggested method clarifications:

> DisCoPatch detects OOD inputs using a discriminator trained to distinguish ID patches from VAE reconstructions and generated patches, aggregating its patch scores per image.

> BRISQUE maps spatial natural-scene statistics to human quality scores. ARNIQA learns a distortion representation through self-supervised training on synthetically degraded images, then fits a linear regressor to human quality scores.

> CLIP-IQA uses a pretrained vision–language model without task-specific quality training, scoring image similarity to the paired prompts “Good photo.” and “Bad photo.”

Replace the final universal claim with:

> Many image-based approaches require separate feature extraction, although SoilingNet demonstrates shared-encoder integration with object detection and segmentation.

If your intended distinction is reusing an **already-trained, frozen** detector without additional representation training, state that narrower distinction explicitly.

## Monitoring the detector features

The active citations occur on line 15; the comparison is on line 20.

| Citation | Verdict | Evidence and action |
| --- | --- | --- |
| `hashemi2023runtime` | Needs correction and calibration detail | Sections 2.2–3.2, author PDF pp. 4–6, use neuron bounds of mean plus/minus **k** standard deviations, with k described as close to two. The fraction of violations becomes a nonconformity score; a separate ID calibration set supplies conformal p-values for thresholding. The paper discards class information when constructing the detector monitor. Exact “two” and a simple count-only decision omit important qualifications. [Full paper](https://arxiv.org/pdf/2212.07773). |
| `becker2026operational` | Supported provisionally; full paper unavailable | The original conference poster's “Approach” and “Key design requirements” confirm a frozen detector and comparisons of channel activation CDFs against training-reference CDFs. This supports your high-level summary. **The full paper is paywalled:** exact distance, aggregation, and complete calibration/data requirements were not verified. [Original poster](https://icpr2026orgteam.github.io/PosterPres/266.pdf); [publisher record](https://link.springer.com/chapter/10.1007/978-3-032-31654-7_9); [institutional record](https://publica.fraunhofer.de/handle/publica/521897). |
| `becker2026degradation` | Material qualification needed | Sections 3.1–3.3 support synthetic degradation training and cosine distance from a pristine prototype. However, Section 4.2 jointly fine-tunes a backbone and embedding head; Section 3.4 describes an auxiliary monitoring path for the main experiments. It is misleading to imply that only a small head is trained over unchanged detector features. The method explicitly tries to reduce content dependence, so a blanket scene-confounding criticism should not include it. [Full original text](https://arxiv.org/html/2602.18394v1). |

Suggested Hashemi replacement:

> Hashemi et al. estimate per-neuron activation bounds from in-distribution images and use the fraction of bound violations as a nonconformity score, calibrated through conformal prediction.

Suggested degradation-manifold replacement:

> Becker et al. learn degradation-sensitive embeddings by jointly fine-tuning a detector backbone and an embedding head on synthetically degraded images. Their main experiments use an auxiliary monitoring path, with cosine distance to a clean-image prototype providing the degradation score.

### Final comparison and claims about this paper

The final paragraph needs three distinctions:

1. **Identify the comparator.** “Becker et al.” could refer to either Becker paper. Cite `becker2026operational` explicitly when comparing reference-distribution methods. The degradation-manifold approach does not have the same training requirements.
2. **Separate mechanism from hypothesis.** “Reference statistics pooled across scenes” is more accurate than “single global reference,” since there are many neuron/channel-specific statistics. The claim that pooling hides mild corruption is a plausible motivation, but it was not established as a demonstrated failure mode by the inspected sources. Label it a hypothesis and test it with a global-versus-conditioned reference ablation.
3. **Support your own empirical claims.** “Same content” overstates what nearest-neighbor retrieval establishes. Use “semantically similar scenes.” The deepest stage “reacts least to corruption” needs your own detector-, dataset-, and corruption-specific depth evidence. Likewise, the flattening/re-weighting explanation needs a definition and evidence in Methods/Experiments. These claims cannot be validated merely by verifying the related-work citations.

Suggested replacement for the comparison's opening:

> Our monitor uses a frozen detector and clean reference images. Hashemi et al. and the activation-distribution monitor of Becker et al. compare inputs with reference statistics pooled across scenes. We hypothesize that this pooling can make mild corruption difficult to distinguish from normal variation in scene content. We therefore condition the reference on semantically similar clean scenes retrieved using a deep-stage representation.

Attach `hashemi2023runtime` and `becker2026operational` to the named methods. Do not extend the clean-only equivalence to every detail of Operational Readiness until its full paper is available.

### Commented citation

`geissler2023low`, line 18, is not active in the compiled section. If restored, its sentence is broadly correct but omits the defining representation: Section 4, PDF pp. 6–7, uses **quantiles across channel-wise spatial activation sums**, normalized against fault-free bounds, before decision-tree classification. Section 3 includes image corruptions and memory faults. Suggested wording: “Geissler et al. train a decision tree on quantiles of channel-wise activation sums, normalized against fault-free reference bounds, to detect errors induced by image corruptions and memory faults.” [Full paper](https://arxiv.org/pdf/2310.20349).

## Bibliography findings and verification limits

No missing active citation keys were found, and the cited works were identified. The main problems are descriptions and comparisons, rather than nonexistent references. This is not a claim that every registry field of every entry was independently certified.

- **Keep Becker Operational Readiness's 2027 bibliography year.** The publisher specifies 2027, LNCS 16814, pp. 121–135, while its online publication date is 4 August 2026 and the conference is ICPR 2026. A key named `becker2026operational` need not match the publication year. [Publisher record](https://link.springer.com/chapter/10.1007/978-3-032-31654-7_9); [publisher-deposited metadata](https://api.crossref.org/works/10.1007/978-3-032-31654-7_9).
- **Keep Park's TPAMI 2026 entry.** The title, three authors, DOI, and pp. 1–14 match publisher-deposited metadata. No volume or issue was deposited in the checked record. Technical checking used the complete author v4 manuscript because the publisher PDF was inaccessible. [Metadata](https://api.crossref.org/works/10.1109/TPAMI.2026.3686715); [author manuscript](https://arxiv.org/pdf/2412.01782v4).
- **NIQE's volume 20 is correct.** Do not replace it with the inconsistent volume on an author laboratory index. The current SPL 20(3), pp. 209–212, 2013 entry matches the publication metadata.
- **DisCoPatch's proceedings title is appropriate.** An earlier arXiv title differs. Method checking used the final ICCV text, including its final normalization description.
- **ARNIQA has a pagination discrepancy between primary records.** The official CVF paper/proceedings use pp. **189–198**, matching your bibliography, while IEEE-deposited Crossref metadata for the same DOI reports **188–197**. The title, authors, venue, and DOI identify the same work. Retain the CVF pagination when citing that version; record this as a publisher metadata discrepancy rather than a verified error in your bibliography. [CVF record](https://openaccess.thecvf.com/content/WACV2024/html/Agnolucci_ARNIQA_Learning_Distortion_Manifold_for_Image_Quality_Assessment_WACV_2024_paper.html); [DOI metadata](https://api.crossref.org/works/10.1109/WACV57701.2024.00026).
- **DisCoPatch DOI verification succeeded on retry.** The deposited title, five authors, ICCV 2025 venue, and pp. 2898–2908 match the entry. [DOI metadata](https://api.crossref.org/works/10.1109/ICCV51701.2025.00278).
- **ISO:** verification is limited to its official public scope and bibliographic record. No claim is made to have audited all clauses of the paid standard.
- **Operational Readiness:** its original poster and metadata were checked, but the full article was unavailable. Obtain the full text before treating its detailed calibration requirements and clean-data-only equivalence as fully verified. A local copy or accessible link was requested during the review.

## Priority of revisions

1. Correct the claim that all image methods use separate models and cannot reuse detector features.
2. Separate perceptual IQA, condition recognition, and OOD detection; fix CLIP-IQA's pretraining description and ARNIQA/BRISQUE supervision.
3. Correct the frozen-feature implication for degradation manifolds and identify the intended Becker comparator explicitly.
4. Add Hashemi's conformal calibration and avoid an unsupported exact two-standard-deviation claim.
5. Clarify ContrastiveConf's weighted averaging and the original task scope of energy/MaxLogit.
6. Qualify the opening definitions and distinguish per-detection uncertainty from frame-level failure prediction.
7. Present scene-confounding as a tested hypothesis and link this paper's depth/content claims to its own experiments.

The proposed wording is review material only. None of it has been applied to the manuscript.
