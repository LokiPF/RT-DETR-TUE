# RT-DETR Query-Level Topological Uncertainty

## Detailed implementation and paper TODO

### Project objective

Build a real-time-capable RT-DETR variant that produces, for each retained detection:

- A calibrated probability that the detection is correct.
- A mean bounding box.
- Aleatoric localization uncertainty from a heteroscedastic bbox head.
- A query-level topological novelty score used as an epistemic uncertainty proxy.
- A conformally calibrated 90% or 95% joint bbox prediction region.

The intended interpretation is:

```math
\text{predictive localization uncertainty}
=
\text{aleatoric uncertainty}
+
\text{TU-conditioned excess uncertainty}.
```

Topological uncertainty (TU) is not claimed to be a Bayesian posterior variance. It is an activation-novelty score that must be calibrated against observed classification and localization errors.

---

## 0. Freeze the research questions and terminology

- [ ] Define the main research question:
  - Can query-level activation-graph topology predict RT-DETR classification and localization failures?
- [ ] Define the real-time question:
  - Can it improve uncertainty quality while preserving the target FPS/latency?
- [ ] Define the calibration question:
  - Does TU produce tighter bbox prediction regions at the same empirical coverage?
- [ ] Select the target RT-DETR implementation and record:
  - Repository URL.
  - Exact commit.
  - Model variant.
  - Input resolution.
  - Number of decoder layers.
  - Number of object queries.
- [ ] Define the deployment hardware and runtime:
  - GPU/accelerator.
  - PyTorch, ONNX Runtime, or TensorRT.
  - Batch size.
  - FP32, FP16, or INT8.
- [ ] Define the real-time budget before implementation:
  - Minimum acceptable FPS.
  - Maximum per-image latency.
  - Maximum permitted latency increase, preferably 5–10%.
- [ ] Use consistent terminology:
  - `TU_cls`: topological novelty used for classification calibration.
  - `TU_box`: topological novelty used for localization calibration.
  - `Sigma_alea`: predicted aleatoric bbox covariance.
  - `Sigma_epi_TU`: TU-conditioned excess covariance.
  - `Sigma_total`: total calibrated predictive covariance.
  - “95% bbox prediction region,” not “95% epistemic confidence box.”
- [ ] Explicitly separate:
  - Classification uncertainty.
  - Localization uncertainty.
  - Detection/existence uncertainty, including false positives and misses.

**Exit criterion:** a one-page specification states the model, hardware, datasets, uncertainty definitions, and target latency.

---

## 1. Reproduce and lock the RT-DETR baseline

- [ ] Create an isolated development branch.
- [ ] Reproduce the pretrained baseline metrics:
  - AP.
  - AP50.
  - AP75.
  - APS/APM/APL.
- [ ] Benchmark baseline deployment performance:
  - Warm-up iterations.
  - At least 500 timed iterations.
  - GPU synchronization around timing.
  - Median latency.
  - p95 latency.
  - FPS.
  - Peak GPU memory.
- [ ] Save the exact evaluation configuration.
- [ ] Save the model checkpoint hash.
- [ ] Confirm deterministic evaluation within an acceptable tolerance.
- [ ] Add a regression test that fails if AP or latency changes unexpectedly.

**Exit criterion:** baseline accuracy and runtime are reproducible and recorded.

---

## 2. Design the per-query data flow

For decoder layer $\ell$ and query $q$:

```math
h_{\ell,q}
\xrightarrow{\text{decoder FFN}}
z_{\ell,q}
\xrightarrow{\text{bbox MLP}}
\Delta b_{\ell,q}.
```

The FFN and bbox-head weights are shared across queries. Query-specific topology comes from the different activation vectors.

- [ ] Confirm tensor shapes in the selected implementation:
  - Decoder output: `[batch, queries, hidden_dim]`.
  - Decoder FFN hidden activation.
  - Bbox-head hidden activations.
  - Classification logits.
  - Predicted bbox coordinates.
- [ ] Confirm parameter sharing:
  - Same decoder-layer FFN weights for all queries.
  - Same bbox-head weights for all queries within a decoder layer.
  - Different weights across decoder layers if heads/layers are cloned.
- [ ] Decide the first implementation scope:
  - Final decoder layer only.
  - Retained inference queries only.
  - Bbox MLP for `TU_box`.
  - Final decoder FFN and/or classification head for `TU_cls`.
- [ ] Add an optional instrumentation flag:
  - Disabled during ordinary training and baseline inference.
  - Returns or caches only the activations required for TU.
- [ ] Preserve the mapping between:
  - Decoder query index.
  - Predicted class.
  - Predicted bbox.
  - Postprocessing top-$K$ result.
- [ ] Avoid assuming that a query index has a fixed semantic meaning.
- [ ] Add shape and index-alignment assertions.

**Exit criterion:** every retained detection can be traced back to the correct query activations without changing baseline predictions.

---

## 3. Implement aleatoric bbox uncertainty

### 3.1 Output parameterization

- [ ] Change each deployed bbox head from four outputs to eight:

    ```math
    (\mu_{cx},\mu_{cy},\mu_w,\mu_h,s_{cx},s_{cy},s_w,s_h).
    ```

- [ ] Convert raw scale values into positive Laplace scales:

    ```math
    b_j=\operatorname{softplus}(s_j)+\epsilon.
    ```

- [ ] Choose and document:
  - `epsilon`.
  - Minimum scale.
  - Maximum scale or raw-logit clamp.
- [ ] Initialize scale outputs so initial uncertainty is finite and conservative.
- [ ] Apply the same change to auxiliary decoder heads if auxiliary losses are used.
- [ ] Ensure denoising-query training remains compatible.

### 3.2 Loss

- [ ] Keep Hungarian matching based on the mean bbox only.
- [ ] Do not initially include predicted variance in the matching cost.
- [ ] Replace or augment L1 with Laplace negative log-likelihood:

    ```math
    \mathcal L_{\text{Laplace}}
    =
    \sum_j\left(
    \frac{|y_j-\mu_j|}{b_j}
    +\log(2b_j)
    \right).
    ```

- [ ] Retain GIoU on the mean bbox.
- [ ] Tune the relative weights of:
  - Laplace NLL.
  - GIoU.
  - Classification loss.
- [ ] Log separately:
  - Mean absolute residual.
  - Mean predicted scale.
  - Laplace NLL.
  - GIoU.
- [ ] Check for the failure modes:
  - Scale collapse.
  - Scale explosion.
  - Variance inflation used to avoid learning the mean.
  - FP16 overflow or underflow.

### 3.3 Covariance representation

- [ ] Start with diagonal aleatoric covariance:

    ```math
    \Sigma_{\text{alea}}
    =
    \operatorname{diag}(2b^2).
    ```

- [ ] Keep covariance in normalized `cxcywh` coordinates during training and calibration.
- [ ] Implement a Jacobian transform to `xyxy` covariance for output:

    ```math
    \Sigma_{xyxy}=J\Sigma_{cxcywh}J^\top.
    ```

- [ ] Add tests verifying:
  - Positive scales.
  - Positive semidefinite covariance.
  - Correct coordinate transformation.
  - Finite gradients.

**Exit criterion:** the probabilistic head trains stably, preserves competitive AP, and produces aleatoric scales correlated with localization residuals.

---

## 4. Implement query-level activation graphs

### 4.1 Faithful activation-graph construction

For a linear transition with input activation $x_q$ and shared weights $W$, define:

```math
e_{ij,q}=|W_{ij}x_{q,i}|.
```

- [ ] Implement activation-graph extraction for a generic `Linear` layer.
- [ ] Exclude bias initially to match the original TU definition.
- [ ] Use identical graph combinatorics for:
  - Prototype construction.
  - Calibration.
  - Test inference.
- [ ] Build a maximum spanning tree for each selected graph.
- [ ] Store its sorted edge-weight vector as the one-dimensional persistence diagram.
- [ ] Implement the diagram distance:

    ```math
    d(D,D')
    =
    \sqrt{\frac{1}{N}\sum_{i=1}^{N}(w_i-w'_i)^2}.
    ```

- [ ] Verify the implementation against a small reference graph where the MST can be checked manually.

### 4.2 Candidate transitions

- [ ] Implement independent switches for:
  - Final decoder FFN `linear1`.
  - Final decoder FFN `linear2`.
  - Bbox MLP first hidden transition.
  - Bbox MLP second hidden transition.
  - Bbox MLP output transition.
  - Classification output transition.
- [ ] Use bbox MLP transitions as the initial `TU_box`.
- [ ] Use decoder FFN and/or classification transition as the initial `TU_cls`.
- [ ] Do not average distances from different transitions until each distance has been normalized or calibrated.

### 4.3 Normalization

- [ ] Record the training distribution of diagram distances for every transition.
- [ ] Compare:
  - Raw distances.
  - Mean/standard-deviation normalization.
  - Median/MAD normalization.
  - Rank or empirical-CDF normalization.
- [ ] Select one normalization using calibration data only.

**Exit criterion:** two detections from the same image can receive different reproducible TU scores, and scores are invariant to batching/order.

---

## 5. Construct query-level TU prototype banks

### 5.1 Select prototype samples

- [ ] Run the trained detector over the training-prototype split.
- [ ] Match predictions to ground truth using the normal detector matching/evaluation procedure.
- [ ] Retain matched positive queries.
- [ ] For the cleanest initial prototype bank, retain detections that:
  - Match a ground-truth object.
  - Predict the correct class.
  - Meet a minimum IoU threshold.
- [ ] Record both predicted and ground-truth class.
- [ ] Use ground-truth class for prototype construction.
- [ ] Use predicted class to select a prototype at inference.

### 5.2 Condition prototypes

- [ ] Build a first prototype bank conditioned on class.
- [ ] Add object-size buckets:
  - Small.
  - Medium.
  - Large.
- [ ] Optionally test aspect-ratio or occlusion buckets.
- [ ] Define a minimum sample count per bucket.
- [ ] Add deterministic fallbacks:
  1. Class-and-size prototype.
  2. Class-only prototype.
  3. Global prototype.
- [ ] Compute a Fréchet mean diagram by averaging corresponding sorted MST weights.
- [ ] Store metadata:
  - Model/checkpoint hash.
  - Layer and transition name.
  - Class.
  - Size bucket.
  - Sample count.
  - Graph sparsity configuration.
  - Normalization statistics.

### 5.3 Leakage prevention

- [ ] Use training data for prototype construction.
- [ ] Use a separate calibration split for uncertainty mappings and conformal quantiles.
- [ ] Use untouched ID and shifted test sets for final reporting.

**Exit criterion:** prototype lookup is deterministic, versioned, and has no calibration/test leakage.

---

## 6. Define per-query TU scores

- [ ] Compute localization TU:

    ```math
    TU^{box}_q
    =
    \frac{1}{|L_{box}|}
    \sum_{\ell\in L_{box}}
    \widetilde d(D_{\ell,q},\bar D_{\ell,c(q),s(q)}).
    ```

- [ ] Compute classification TU:

    ```math
    TU^{cls}_q
    =
    \frac{1}{|L_{cls}|}
    \sum_{\ell\in L_{cls}}
    \widetilde d(D_{\ell,q},\bar D_{\ell,c(q)}).
    ```

- [ ] Keep `TU_box` and `TU_cls` separate.
- [ ] Log the individual transition distances in addition to the aggregate.
- [ ] Test whether using a learned weighted combination improves over a simple mean.
- [ ] If learning weights, fit them only on calibration data and regularize strongly.
- [ ] Measure whether TU is confounded by:
  - Class frequency.
  - Object size.
  - Aspect ratio.
  - Confidence.
  - Image brightness or corruption severity.

**Exit criterion:** `TU_cls` and `TU_box` are stable scalar features with documented conditioning and fallbacks.

---

## 7. Build essential non-topological baselines

TU must outperform simpler novelty measures to justify its complexity.

- [ ] Implement Euclidean distance from the final query embedding to a class prototype.
- [ ] Implement class-conditional Mahalanobis distance:

    ```math
    u_q=(h_q-\bar h_c)^\top C_c^{-1}(h_q-\bar h_c).
    ```

- [ ] Implement k-nearest-neighbour distance in query-embedding space.
- [ ] Implement confidence-only calibration.
- [ ] Implement aleatoric-only bbox intervals.
- [ ] Implement aleatoric + conformal bbox intervals.
- [ ] If feasible, implement last-layer Laplace as a single-pass epistemic baseline.
- [ ] Use MC dropout and a small deep ensemble as slower quality references.
- [ ] Give every uncertainty feature the same calibration protocol and data budget.

**Go/no-go gate:** continue positioning TU as the main method only if it improves localization-error ranking, OOD detection, interval efficiency, or another clearly motivated property over Mahalanobis/kNN.

---

## 8. Create the calibration dataset

For every retained prediction, store:

- [ ] Image identifier.
- [ ] Query index and decoder layer.
- [ ] Predicted class and raw logits.
- [ ] Raw confidence.
- [ ] Predicted mean bbox.
- [ ] Predicted aleatoric scale/covariance.
- [ ] Ground-truth match, if one exists.
- [ ] Ground-truth class and bbox.
- [ ] Coordinate residual.
- [ ] IoU/GIoU.
- [ ] `TU_cls`.
- [ ] `TU_box`.
- [ ] Mahalanobis/kNN baselines.
- [ ] Object-size bucket.
- [ ] Corruption/domain label.

Define classification correctness before fitting:

- [ ] Decide the matching IoU threshold.
- [ ] Decide whether correctness requires:
  - A matched object only.
  - Correct class and matched object.
  - Correct class and IoU above a specified threshold.
- [ ] Treat unmatched predictions as false positives.
- [ ] Evaluate missed ground-truth objects separately; they cannot receive query-level bbox covariance.

**Exit criterion:** calibration records can reproduce all calibrator training and evaluation without rerunning RT-DETR.

---

## 9. Calibrate classification confidence

- [ ] Establish baseline calibrators:
  - Temperature scaling.
  - Isotonic regression.
  - Logistic calibration.
- [ ] Fit a TU-aware calibrator:

    ```math
    \hat p_q
    =
    g_{\text{cls}}(
    \text{raw logit or margin},
    TU^{cls}_q
    ).
    ```

- [ ] Start with a low-capacity monotonic or logistic model.
- [ ] Test whether `TU_box` adds useful classification information.
- [ ] Avoid a large neural calibrator unless the calibration set is large.
- [ ] Compare global and class-conditional calibration.
- [ ] Evaluate:
  - NLL.
  - Brier score.
  - ECE and detection-specific calibration error.
  - Reliability diagrams.
  - Risk-coverage/AURC.
  - ID versus shifted/OOD performance.
- [ ] Verify that improved ECE is not caused by destroying detection ranking or AP.

**Exit criterion:** the TU-aware calibrator improves at least one proper scoring rule and remains robust across relevant shifts.

---

## 10. Map `TU_box` to excess localization uncertainty

### 10.1 Scalar covariance inflation

Start with:

```math
\Sigma_{\text{total},q}
=
\alpha(TU^{box}_q,c_q,s_q)\Sigma_{\text{alea},q},
\qquad \alpha\ge 1.
```

```math
\Sigma_{\text{epi-TU},q}
=
(\alpha-1)\Sigma_{\text{alea},q}.
```

- [ ] Compute normalized squared residuals:

    ```math
    m_i
    =
    r_i^\top\Sigma_{\text{alea},i}^{-1}r_i.
    ```

- [ ] Fit a low-capacity monotonic mapping from `TU_box` to $\alpha$.
- [ ] Compare:
  - Isotonic regression.
  - Monotonic spline.
  - Binned estimate $\mathbb E[m\mid TU]/4$.
  - Direct Gaussian quasi-NLL optimization.
- [ ] Constrain $\alpha\ge1$ for the initial interpretation.
- [ ] Compare global, per-class, and class-and-size mappings.
- [ ] Use hierarchical fallbacks for sparse classes.

### 10.2 More expressive alternatives

Only attempt these after the scalar model:

- [ ] Coordinate-specific inflation:

    ```math
    \Sigma_{\text{total}}
    =
    \operatorname{diag}(
    \alpha_j(TU)\sigma^2_{\text{alea},j}
    ).
    ```

- [ ] Class/size residual-shape template:

    ```math
    \Sigma_{\text{epi-TU}}
    =
    g(TU)C_{c,s}.
    ```

- [ ] Full covariance using a Cholesky parameterization.
- [ ] Confirm every covariance remains positive semidefinite.

**Exit criterion:** higher TU corresponds to larger empirical localization error, and TU-conditioned scaling improves NLL or interval efficiency.

---

## 11. Produce conformal bbox prediction regions

### 11.1 Choose the desired output

- [ ] Decide whether the product needs:
  - A four-dimensional ellipsoidal bbox region.
  - Joint intervals for `cx, cy, w, h`.
  - Joint intervals for `x1, y1, x2, y2`.
  - A conservative drawable outer envelope.
- [ ] Prefer joint `xyxy` edge intervals for simple visualization and downstream use.

### 11.2 Joint axis-aligned conformal intervals

For calibration item $i$, define:

```math
s_i
=
\max_j
\frac{|y_{ij}-\mu_{ij}|}
{\sqrt{\Sigma_{\text{total},i,jj}}}.
```

- [ ] Calculate the finite-sample conformal quantile using the corrected rank:

    ```math
    k=\left\lceil(n+1)(1-\delta)\right\rceil.
    ```

- [ ] Store $q_{0.90}$, $q_{0.95}$, or other required levels.
- [ ] At inference, return:

    ```math
    \mu_{qj}
    \pm
    q_{1-\delta}
    \sqrt{\Sigma_{\text{total},q,jj}}.
    ```

- [ ] Evaluate empirical joint coverage, not only coordinate-wise coverage.
- [ ] Test Mondrian conformal calibration by:
  - Class.
  - Object-size bucket.
  - Relevant deployment domain.
- [ ] Use fallbacks where calibration groups are too small.
- [ ] Evaluate the effect of clipping normalized coordinates to image bounds.
- [ ] Report coverage both before and after clipping.
- [ ] Enforce valid coordinate ordering carefully and re-check coverage afterward.

### 11.3 Coverage interpretation

- [ ] State that conformal coverage is marginal under exchangeability.
- [ ] Do not claim exact conditional coverage for each class, size, or TU level.
- [ ] Do not claim preserved coverage under arbitrary deployment shift.
- [ ] Report stratified coverage to expose failures hidden by aggregate coverage.

**Exit criterion:** the requested empirical coverage is achieved on untouched ID test data, with competitive interval width.

---

## 12. Optimize TU for real-time inference

### 12.1 Establish the expensive operations

- [ ] Profile separately:
  - Activation capture.
  - Edge-weight construction.
  - MST computation.
  - Diagram sorting.
  - Prototype lookup.
  - Diagram-distance computation.
  - Calibration and conformal postprocessing.
- [ ] Profile CPU/GPU synchronization caused by topology code.
- [ ] Ensure timing includes the entire deployed uncertainty pipeline.

### 12.2 Reduce topology cost

- [ ] Compute TU only for retained top-$K$ detections.
- [ ] Begin with only the final decoder layer.
- [ ] Begin with only the bbox MLP transition that gives the best quality/latency ratio.
- [ ] Compare full graphs with fixed sparse graphs.
- [ ] If sparsifying:
  - Create the mask once from model weights or a fixed rule.
  - Use the same mask for prototypes and inference.
  - Ensure the graph remains connected.
  - Document that this is an approximation to the original TU method.
- [ ] Batch edge-weight computation across retained queries.
- [ ] Avoid Python loops on the inference-critical path.
- [ ] Avoid transferring activations to CPU unless profiling shows it is acceptable.
- [ ] Cache prototype tensors on the execution device.
- [ ] Consider a separate asynchronous postprocessor only if output latency permits it.
- [ ] If ONNX/TensorRT cannot express the topology operations:
  - Export RT-DETR normally.
  - Export selected query features.
  - Run TU as a compact deployment-side postprocessor.

### 12.3 Runtime modes

- [ ] Implement switchable modes:
  - Baseline.
  - Aleatoric only.
  - Confidence TU only.
  - Bbox TU only.
  - Full uncertainty.
- [ ] Report latency for every mode.

**Exit criterion:** full uncertainty satisfies the predefined real-time budget or the paper clearly presents the accuracy/uncertainty/latency Pareto frontier.

---

## 13. Define the inference output contract

- [ ] Return a stable per-detection record containing:

    ```text
    box_mean_xyxy
    class_id
    score_raw
    score_calibrated
    tu_cls
    tu_box
    aleatoric_covariance
    tu_excess_covariance
    total_covariance
    box_interval_90
    box_interval_95
    prototype_fallback_level
    ```

- [ ] Include an uncertainty-version identifier tied to:
  - Model checkpoint.
  - Prototype bank.
  - Calibrator.
  - Conformal quantiles.
- [ ] Decide behaviour when:
  - No valid prototype exists.
  - TU computation fails.
  - Covariance is non-finite.
  - A prediction is outside the calibrated support.
- [ ] Prefer a conservative fallback rather than silently returning a narrow interval.

---

## 14. Unit and integration tests

### Model tests

- [ ] Shared FFN weights produce different query activation graphs when query activations differ.
- [ ] Reordering queries reorders TU scores identically.
- [ ] Batching does not change individual TU values.
- [ ] Instrumentation does not change predicted logits or boxes.
- [ ] Top-$K$ indices select the correct cached activations.

### Topology tests

- [ ] MST matches a reference implementation on small graphs.
- [ ] Sorted persistence vectors have the expected length.
- [ ] Diagram distance is zero for identical diagrams.
- [ ] Prototype means match manually computed examples.
- [ ] Sparse graph masks are fixed and connected.

### Uncertainty tests

- [ ] Aleatoric scales are positive and finite.
- [ ] Covariances are symmetric and positive semidefinite.
- [ ] Coordinate transformations are correct.
- [ ] Inflation mapping is monotonic and $\alpha\ge1$.
- [ ] Conformal quantile indexing is correct for small synthetic datasets.
- [ ] Requested joint coverage is recovered on simulated calibrated data.

### Deployment tests

- [ ] FP16 results remain numerically stable.
- [ ] Batch size one meets latency requirements.
- [ ] Exported runtime matches PyTorch outputs within tolerance.

---

## 15. Experimental protocol

### 15.1 Data

- [ ] Select at least:
  - One in-distribution object-detection dataset.
  - One synthetic corruption suite.
  - One cross-domain or naturally shifted dataset with compatible labels.
- [ ] Keep prototype, calibration, validation, and final test roles separate.
- [ ] Document class mappings for cross-domain evaluation.
- [ ] Include shifts relevant to the intended deployment:
  - Blur.
  - Noise.
  - Illumination.
  - Weather.
  - Compression.
  - Occlusion.
  - Camera/domain changes.

### 15.2 Baselines

- [ ] Deterministic RT-DETR.
- [ ] Heteroscedastic RT-DETR.
- [ ] Heteroscedastic + conformal.
- [ ] Confidence/embedding Mahalanobis + conformal.
- [ ] kNN query novelty + conformal.
- [ ] Last-layer Laplace + conformal, if feasible.
- [ ] MC dropout.
- [ ] Deep ensemble as an offline quality ceiling.
- [ ] Proposed query-TU + aleatoric + conformal.

### 15.3 Metrics

Detection performance:

- [ ] AP, AP50, AP75.
- [ ] APS, APM, APL.
- [ ] Precision and recall at relevant operating points.

Classification calibration:

- [ ] NLL.
- [ ] Brier score.
- [ ] ECE/detection calibration error.
- [ ] Reliability plots.
- [ ] AURC.

Localization uncertainty:

- [ ] Localization NLL.
- [ ] Joint 90% and 95% coverage.
- [ ] Coverage error.
- [ ] Mean interval width.
- [ ] Median interval width.
- [ ] Region volume or generalized variance.
- [ ] Coverage stratified by class and object size.

Failure and OOD ranking:

- [ ] AUROC/AUPR for detecting IoU below chosen thresholds.
- [ ] Spearman correlation with $1-\mathrm{IoU}$.
- [ ] Risk-coverage curves.
- [ ] OOD/corruption AUROC where labels permit.

Efficiency:

- [ ] Median and p95 latency.
- [ ] FPS.
- [ ] Peak GPU memory.
- [ ] Model size.
- [ ] Prototype-bank size.
- [ ] Postprocessing time.

### 15.4 Statistics

- [ ] Run multiple training seeds where training changes.
- [ ] Bootstrap confidence intervals over images, not individual boxes alone.
- [ ] Use paired bootstrap comparisons for interval width and error ranking.
- [ ] Report both aggregate and stratified results.

---

## 16. Required ablation studies

- [ ] Global image TU versus query-level TU.
- [ ] Decoder FFN versus bbox MLP versus both.
- [ ] Individual decoder layers.
- [ ] Individual bbox-head transitions.
- [ ] Raw versus normalized diagram distances.
- [ ] Full versus sparse activation graphs.
- [ ] Number of retained queries receiving TU.
- [ ] Class-only versus class-and-size prototypes.
- [ ] Prototype sample count.
- [ ] Correct high-IoU prototypes versus all matched positive prototypes.
- [ ] Aleatoric only versus TU only versus combined.
- [ ] Scalar versus coordinate-wise covariance inflation.
- [ ] TU mapping without conformalization versus with conformalization.
- [ ] Standard versus Mondrian conformal calibration.
- [ ] TU versus Mahalanobis versus kNN at matched latency.
- [ ] ID versus corruption versus cross-domain performance.

**Critical ablation:** show whether persistent topology adds information beyond the raw decoder embedding and raw confidence.

---

## 17. Recommended implementation order and gates

### Milestone A — feasibility

- [ ] Instrument the final bbox MLP.
- [ ] Compute full-graph `TU_box` offline for matched detections.
- [ ] Compare `TU_box`, confidence, Mahalanobis distance, and $1-\mathrm{IoU}$.
- [ ] Plot localization error by TU decile.

**Gate A:** stop or redesign if TU has no relationship with localization error.

### Milestone B — basic uncertainty system

- [ ] Train the aleatoric bbox head.
- [ ] Fit scalar TU covariance inflation.
- [ ] Produce conformal 95% joint bbox intervals.
- [ ] Compare interval coverage and width against aleatoric-only conformal.

**Gate B:** proceed if TU produces tighter intervals, better stratified coverage, or better shifted-data behaviour.

### Milestone C — classification and localization

- [ ] Add `TU_cls`.
- [ ] Fit TU-aware confidence calibration.
- [ ] Evaluate joint confidence and localization failure ranking.

**Gate C:** confirm `TU_cls` and `TU_box` provide distinguishable, task-relevant information.

### Milestone D — real-time system

- [ ] Restrict TU to the best transition(s).
- [ ] Add fixed graph sparsification if needed.
- [ ] Vectorize the critical path.
- [ ] Benchmark end-to-end deployment latency.

**Gate D:** satisfy the predefined latency budget or present a clear Pareto improvement.

### Milestone E — paper-scale evaluation

- [ ] Complete all baselines.
- [ ] Complete ID, corruption, and cross-domain evaluation.
- [ ] Complete ablations.
- [ ] Run statistical comparisons.
- [ ] Freeze the method before final test-set evaluation.

---

## 18. Paper framing

### Defensible central claim

> Query-conditioned activation-graph topology provides a single-pass novelty signal for transformer object detectors that can improve confidence calibration and adapt bounding-box prediction regions under distribution shift while retaining real-time capability.

### Avoid unsupported claims

- [ ] Do not claim TU is an exact Bayesian epistemic variance.
- [ ] Do not claim conformal coverage under arbitrary distribution shift.
- [ ] Do not claim conditional 95% coverage without appropriate evidence.
- [ ] Do not claim topology is beneficial without comparison to embedding distances.
- [ ] Do not claim real-time performance based only on neural-network forward time.

### Candidate contributions

- [ ] First query-level adaptation of persistence-of-activation-graphs TU to DETR-style set prediction, subject to final literature verification.
- [ ] Separate topology signals for semantic confidence and bbox localization.
- [ ] TU-conditioned aleatoric covariance and conformal bbox regions.
- [ ] Efficient retained-query topology computation for real-time deployment.
- [ ] Evaluation of uncertainty quality under corruption and domain shift.

### Core figures

- [ ] Method overview from query activations to calibrated confidence and bbox region.
- [ ] Example detections showing raw box, aleatoric interval, and TU-expanded interval.
- [ ] Localization error versus TU decile.
- [ ] Coverage-versus-width Pareto plot.
- [ ] Risk-coverage curves.
- [ ] Uncertainty quality versus latency.

### Core tables

- [ ] ID accuracy and calibration.
- [ ] Shift/OOD uncertainty results.
- [ ] 90%/95% coverage and interval width.
- [ ] Runtime and memory.
- [ ] Layer/prototype/topology ablations.

---

## 19. Reproducibility and release

- [ ] Save all configurations and random seeds.
- [ ] Version:
  - Detector checkpoint.
  - Prototype bank.
  - Confidence calibrator.
  - Bbox inflation calibrator.
  - Conformal quantiles.
- [ ] Provide scripts for:
  - Prototype extraction.
  - Calibration-record generation.
  - Calibrator fitting.
  - Conformal fitting.
  - ID evaluation.
  - Shift/OOD evaluation.
  - Latency benchmarking.
- [ ] Include a minimal inference example.
- [ ] Include pretrained uncertainty artifacts if licensing permits.
- [ ] Document hardware and software versions.
- [ ] Release raw aggregate metrics needed to reproduce plots.

---

## Suggested module layout

Adapt names to the existing repository:

```text
uncertainty/
  aleatoric_bbox_head.py
  activation_capture.py
  activation_graph.py
  mst_backend.py
  prototype_bank.py
  query_tu.py
  embedding_baselines.py
  confidence_calibration.py
  bbox_calibration.py
  conformal_bbox.py
  output_schema.py

tools/
  build_tu_prototypes.py
  collect_calibration_records.py
  fit_confidence_calibrator.py
  fit_bbox_calibrator.py
  fit_conformal_quantiles.py
  evaluate_uncertainty.py
  benchmark_uncertainty_runtime.py

tests/
  test_activation_alignment.py
  test_activation_graph.py
  test_mst.py
  test_prototype_bank.py
  test_aleatoric_bbox.py
  test_bbox_covariance.py
  test_conformal_bbox.py
  test_runtime_modes.py
```

---

## Minimum viable paper result

The project is paper-ready only if the final evidence supports all of the following:

- [ ] Query-level TU predicts localization failure better than raw confidence.
- [ ] TU adds measurable value beyond Mahalanobis or kNN query distance.
- [ ] TU-aware prediction regions are tighter at matched coverage, or more robust under meaningful shifts.
- [ ] Calibrated confidence improves a proper scoring rule.
- [ ] Detection AP remains competitive.
- [ ] End-to-end uncertainty inference remains within the declared real-time budget.
- [ ] Limitations and coverage assumptions are reported clearly.
