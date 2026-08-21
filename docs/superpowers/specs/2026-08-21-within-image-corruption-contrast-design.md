# Within-Image Corruption Contrast Design

Date: 2026-08-21

## Goal

Create a deployment-oriented corruption score from one image at one timestamp. The score uses
one confidence-percentile range as an internal reference for another range whose persistence
distance responds strongly to blur. It must not require a paired clean version of the deployed
image.

The experiment asks whether a within-image contrast removes scene-to-scene changes in absolute
persistence distance and ranks corrupted images more reliably than the responsive percentile's
raw distance.

## Decisions already made

- Deployment receives one image only. There is no clean partner from the same scene or timestamp.
- Both the reference and responsive scores are computed from the current image.
- The primary reference is the dynamic, padding-filtered 0--10 percent confidence decile.
- The primary responsive range is the dynamic, padding-filtered 50--60 percent confidence
  decile.
- A wider secondary comparison uses the 0--20 percent and 40--60 percent confidence quintiles.
- Persistence distance at decoder layer 2 is the signal. Earlier layers and the combined score
  are outside this experiment.
- Apply the same scene summary to both sides of a contrast. Test `mean`, `q90`, and
  `top20_mean`.
- Compare four signals: raw responsive distance, signed raw gap, symmetric relative gap, and a
  clean-predicted residual.
- The residual may store one robust slope and one robust offset calculated from clean tuning
  images. This is a fixed mathematical correction, not a trained uncertainty head.
- Use the existing clean fingerprint bank to obtain query distances. "No reference at
  inference" means no paired clean image; it does not remove the already-deployed clean bank.
- Use the 250 tuning images and six Gaussian-blur severities only. Held-out images remain
  untouched.
- Treat outputs as corruption-ranking scores, not calibrated probabilities.

## Why the low Spearman result is not enough

The dynamic 0--10 percent decile has median per-image Spearman close to zero at layer 2 under
`q90`. That says the score has no dependable increasing or decreasing order across the six blur
levels. It does not prove that the score is numerically stable.

A near-zero Spearman can describe three very different anchors:

1. a score that barely changes within each image but differs greatly between scenes;
2. a score that is nearly identical for every image;
3. a score that jumps randomly within each image.

Only the first behavior is the desired internal reference. It can explain image-specific
baseline distance while remaining largely unaffected by corruption. The second adds little, and
the third injects noise into every contrast.

The reference is also a dynamic percentile population, not a fixed collection of query IDs. The
previous experiment found that the dynamic low decile retained only about 6.5 percent of its
severity-zero membership. This follow-up therefore evaluates stability of the aggregate
percentile score. It never claims that particular low-confidence queries remain fixed.

## Inference interpretation

At inference, both inputs come from the current image:

```text
current image
  -> remove padded query slots
  -> rank valid queries by current detector confidence
  -> summarize layer-2 persistence distance in the reference range
  -> summarize layer-2 persistence distance in the responsive range
  -> combine the two summaries into one corruption score
```

The reference score is used as a proxy for scene-specific baseline distance. It is not treated
as the responsive range's clean value. The experiment must first establish the normal clean
relationship between the two ranges.

For example, if clean scenes normally place the responsive range 0.30 distance units above the
reference range, an observed gap of 0.68 is evidence that the responsive range moved unusually
far. The deployment calculation observes only the current reference and responsive values; the
normal relationship is stored from tuning data.

## Source artifacts and architecture

Add a reporting-only command:

```text
analyze-within-image-contrast
  --source <completed analyze-corruption-sensitivity output directory>
  --output <new output directory>
```

The source directory must contain the completed `per_scene.csv` and `summary.json` produced by
`analyze-corruption-sensitivity`. The new command reads scene scores and provenance from those
files. It must not import or run DETR, persistence extraction, the comparison bank, query kNN, or
feature-cache iteration.

The command writes a separate output directory atomically and refuses to overwrite an existing
one. The completed corruption-sensitivity directory remains unchanged.

### Input requirements

The source must prove all of the following before any contrast is calculated:

- source partition is `tuning`;
- image count is 250;
- severities are exactly 0, 1, 2, 3, 4, and 5;
- every selected source candidate covers all 250 images at all six severities;
- membership is `dynamic`;
- padding mode is `filtered`;
- signal is `persistence`;
- score scope is `layer_2`;
- aggregation is one of `mean`, `q90`, and `top20_mean`;
- the source summary's provenance agrees with every source row;
- no source row key is missing or duplicated.

Refuse a source containing held-out rows rather than filtering them silently.

## Fixed bucket pairs

The experiment has two predeclared pairs:

| Pair name | Scheme | Reference range | Responsive range | Role |
|---|---|---|---|---|
| `decile_00_10__50_60` | decile | `decile_00_10` | `decile_50_60` | primary |
| `quintile_00_20__40_60` | quintile | `quintile_00_20` | `quintile_40_60` | wider stability check |

Do not search over every possible bucket pair. The purpose is to test the approved internal
reference hypothesis, not to manufacture a winner from a large contrast sweep.

For each pair, use matched `mean`, `q90`, and `top20_mean` summaries. Never subtract a mean from a
q90 or otherwise mix scene summaries.

## Four score methods

Let `reference` be the current image's reference-bucket scene score and `responsive` the current
image's responsive-bucket scene score. Both are non-negative query-to-clean-bank persistence
distance summaries in the same layer and units.

### 1. Raw responsive control

```text
score = responsive
```

This is the matched control. It shows whether either contrast improves over using the responsive
distance alone.

### 2. Signed raw gap

```text
score = responsive - reference
```

This removes an additive baseline when the two ranges rise and fall together between scenes. It
is signed, not an absolute value. A negative value remains negative.

### 3. Symmetric relative gap

```text
score = 2 times (responsive - reference), divided by (responsive + reference)
```

This removes multiplicative scale. Two scenes whose raw distances differ tenfold receive the
same value when the responsive-to-reference relationship is proportional.

Because both inputs are non-negative, the denominator cannot be negative. When both are exactly
zero, define the relative gap as zero. Do not add a tunable epsilon. For finite non-negative
inputs with a positive denominator, the score lies between negative two and positive two.

### 4. Clean-predicted residual

```text
expected clean responsive = offset + slope times reference
score = responsive - expected clean responsive
```

The reference predicts what the responsive score would normally be for a clean image with that
baseline. A positive residual means the observed responsive distance is higher than that clean
prediction.

The line is fitted using clean severity-zero tuning rows only. Corrupted rows and corruption
labels never determine its slope or offset.

## Robust clean relationship

Fit a separate relationship for each bucket pair and scene summary. Do not share one slope across
deciles, quintiles, or aggregations.

Use a deterministic robust line:

1. calculate every pairwise slope between clean images whose reference values differ;
2. take the median pairwise slope;
3. take the median of `responsive - slope times reference` as the offset.

With 250 images, the pairwise calculation is small enough to do directly. If fewer than two
distinct finite reference values exist, mark the residual unavailable. Do not substitute a zero
slope.

### Leakage-safe tuning evaluation

Use five deterministic image folds:

- sort the 250 integer image IDs;
- assign the image at sorted position `p` to fold `p modulo 5`;
- keep all six severities of an image in its assigned fold;
- for one held-out fold, fit slope and offset from severity-zero rows in the other four folds;
- apply that line to all six severities of every image in the held-out fold;
- concatenate the five held-out predictions to obtain cross-fitted tuning scores.

Here, "held-out fold" means one fold temporarily excluded inside the tuning partition. It is not
the separate held-out test partition, which this command must never read.

The raw responsive, raw gap, and relative gap need no fitting. The residual metrics and plots use
cross-fitted scores so an image never helps construct its own expected clean value.

After cross-fitted evaluation, fit one final line for each of the six pair-and-summary
configurations on all 250 severity-zero tuning images. Store the five fold-specific lines and
the final all-clean line in `summary.json`. The figures may show these final lines, but the final
lines are never used to report tuning residual performance. If a residual is ultimately chosen,
only its one final slope and offset become deployment constants.

## Anchor diagnostics

Evaluate whether the reference behaves like an internal anchor before interpreting a contrast.

### Within-image corruption drift

For each pair, summary, and severity, calculate the reference score minus the same image's
severity-zero reference score. Report across images:

- median signed drift;
- median absolute drift;
- 25th and 75th percentiles of signed drift;
- 25th and 75th percentiles of absolute drift;
- fraction whose absolute drift is exactly zero.

For each image, also report its reference range across all six severities, signed Spearman, and
absolute Spearman. Constant complete curves are flat with both Spearman values equal to zero;
missing or non-finite curves remain unmeasured.

### Between-image spread

At every severity, report the reference's count, mean, population variance, median, 25th
percentile, 75th percentile, interquartile range, median absolute deviation, minimum, and maximum.

For severities 1 through 5, also report a stability-to-spread ratio:

```text
median absolute within-image drift at this severity
divided by
the severity-zero reference interquartile range
```

Lower values are better. If the clean interquartile range is zero, mark the ratio unavailable;
do not divide by a replacement constant.

### Clean cross-bucket relationship

For each pair and summary, report on severity-zero rows:

- Pearson correlation between reference and responsive distance;
- Spearman correlation between reference and responsive distance;
- final robust slope and offset;
- cross-fitted median absolute prediction error;
- cross-fitted median absolute error from a fold-specific constant predictor that always uses
  the other folds' median clean responsive score;
- cross-fitted residual median, interquartile range, and median absolute deviation.

High between-image reference spread is useful only when the reference also predicts the normal
responsive level and remains stable under corruption. Do not call the anchor useful from any one
diagnostic alone.

Call the robust clean line predictive only when its cross-fitted median absolute error is lower
than the fold-specific constant-median predictor's error. Otherwise label the clean relationship
non-predictive, even if its in-sample correlation looks strong.

## Per-image trend metrics

For each image, bucket pair, aggregation, and score method, use all six severity scores to
calculate:

- signed Spearman;
- absolute Spearman;
- direction: increasing, decreasing, flat, or unmeasured;
- adjacent-step consistency after candidate orientation;
- maximum-blur-versus-clean success after candidate orientation.

As in the preceding deployment analysis, a complete finite constant curve is explicitly flat
with signed and absolute Spearman equal to zero. An incomplete or non-finite curve is unmeasured,
not zero.

The group sensitivity is the median of per-image absolute Spearman values. Never replace it with
the absolute value of median signed Spearman.

## Candidate orientation and AUROC

Each pair, aggregation, and score method is one candidate. Choose one orientation for that
candidate from its median signed tuning Spearman:

- positive median: use the score unchanged;
- negative median: multiply by negative one;
- zero median: mark the candidate unorientable.

Never choose orientation separately by image, severity, fold, or AUROC comparison.

For every orientable full-coverage candidate, calculate clean severity 0 versus each corrupted
severity 1 through 5 AUROC using the one locked orientation. Macro AUROC is the ordinary mean of
the five values. AUROC measures ranking, not probability.

## Ranking and matched comparisons

A candidate is eligible for deployment ranking only when it has:

- all 250 images and all six severities;
- finite scores everywhere;
- one nonzero locked orientation;
- one of the two fixed pairs;
- one of the three matched summaries;
- one of the four declared score methods.

Rank eligible candidates by:

1. higher macro AUROC;
2. higher severity-1 AUROC;
3. higher median per-image absolute Spearman;
4. higher dominant-direction fraction, including flat images in the denominator;
5. higher oriented adjacent-step consistency;
6. deterministic pair, aggregation, and method name.

Every contrast must also be compared with the raw responsive control from the same bucket pair
and aggregation. Report the difference in all five severity AUROCs and macro AUROC. A contrast is
"promising" only when its macro AUROC exceeds the matched raw control and it retains complete
coverage. A higher point estimate is tuning evidence, not confirmation.

### Paired image bootstrap

For each contrast-versus-control comparison, perform a paired bootstrap over image IDs:

- use random seed `20260821`;
- draw 2,000 samples of 250 image IDs with replacement;
- apply one sampled ID list to both methods and all six severities;
- recompute the five AUROCs and macro AUROC difference;
- report the 2.5th and 97.5th percentiles of the macro difference.

Call a positive point estimate whose interval crosses zero `inconclusive on tuning`. Call a
positive point estimate whose interval remains above zero `supported on tuning`. Neither label is
a held-out claim because the methods and ranges were selected using this tuning program.

## Figures

Produce four figures.

### `anchor_and_responsive_actual_distance.png`

Use six panels: two bucket pairs by three summaries. Each panel shows raw, un-oriented median and
interquartile bands over severities 0 through 5 for the reference and responsive scores. The two
lines in one panel share an axis because they have the same units. Do not force unrelated scene
summaries onto one shared numerical scale.

### `clean_anchor_relationship.png`

Use the same six-panel layout. Each panel shows the 250 severity-zero reference/responsive points
and the final robust clean line. Axes say which bucket is the reference and which is responsive.
The caption states that cross-fitted residual metrics use fold-specific lines even though the
figure displays the final deployment line.

### `contrast_scores_by_severity.png`

Show raw responsive, raw gap, relative gap, and cross-fitted residual in separate panels because
their units and ranges differ. Each panel uses a median line and interquartile band across images.
Plots show raw candidate direction; do not orient curves for presentation.

### `auroc_by_blur_severity.png`

For each pair and aggregation, show all four methods on one AUROC plot with corruption severity 1
through 5 on the x-axis and AUROC from 0 to 1 on the y-axis. Include a horizontal chance line at
0.5. These panels can share a y-axis because AUROC has the same meaning for every method.

Record all plotted values, panel selections, and axis limits in `summary.json`.

## Outputs

The output directory contains exactly:

- `per_scene_contrasts.csv`: one row per image, severity, pair, aggregation, and method, with raw
  inputs, derived score, fold, fit provenance, and per-image trend fields;
- `anchor_diagnostics.csv`: anchor drift, between-image spread, and clean-relationship metrics;
- `candidate_metrics.csv`: one row per pair, aggregation, and method with orientation, five
  AUROCs, macro AUROC, trend metrics, coverage, and matched-control comparison;
- `summary.json`: source provenance, fixed configuration, validation counts, fold assignments,
  fold-specific fits, final fits, anchor diagnostics, candidate metrics, bootstrap intervals,
  figure data, axis limits, and ranking;
- `anchor_and_responsive_actual_distance.png`;
- `clean_anchor_relationship.png`;
- `contrast_scores_by_severity.png`;
- `auroc_by_blur_severity.png`;
- `easy-report.md`.

The easy report uses plain language and leads with:

1. whether the low-confidence range is numerically stable enough to act as an anchor;
2. whether the anchor explains scene-to-scene raw-distance variation;
3. whether raw gap, relative gap, or clean residual beats the matched raw responsive distance;
4. performance at every blur severity, especially severity 1;
5. whether the bootstrap comparison is supported or inconclusive on tuning;
6. the final slope and offset if the residual wins;
7. the statement that no held-out images were used and the result is not a calibrated
   probability.

## Error handling

Fail before writing a final output directory when:

- the source directory or required file does not exist;
- the source artifact is unfinished or has an unknown schema;
- source provenance disagrees between CSV and JSON;
- any selected row is outside the tuning partition;
- image count is not 250 or the exact six severities are not complete;
- a required pair, summary, or layer-2 persistence row is missing;
- a source or derived row key is duplicated;
- a source score is negative, non-finite, or otherwise invalid for a persistence distance;
- the output directory already exists.

A constant clean reference does not invalidate raw responsive, raw gap, or relative gap. It makes
only the residual unavailable, records the reason, and excludes residual candidates from ranking.
The relative gap's exact zero-plus-zero case is valid and equals zero.

Write all tables, figures, JSON, and Markdown into a sibling staging directory. Publish the
directory only after every expected file has been written and validated. Remove the staging
directory on failure.

## Testing strategy

### Pure score tests

- raw responsive returns the responsive input exactly;
- raw gap is signed responsive minus reference;
- symmetric relative gap is scale-invariant;
- equal positive inputs give zero relative gap;
- zero plus zero gives zero relative gap;
- positive inputs keep the relative gap between negative two and positive two;
- non-finite and negative distances are rejected.

### Robust-fit and fold tests

- a hand-checkable straight line recovers its slope and offset;
- one extreme clean outlier does not control the robust fit;
- repeated reference values skip undefined pairwise slopes;
- a constant reference makes the residual unavailable;
- deterministic fold assignment is stable under input row reordering;
- every image's fit excludes that image and every severity derived from it;
- only severity-zero rows influence slope and offset;
- the final all-clean fit is stored but never substituted for cross-fitted tuning residuals.

### Synthetic behavior tests

- additive scene offsets make the raw gap outperform raw responsive distance;
- multiplicative scene scaling makes relative gap outperform raw gap;
- a non-unit clean relationship makes the clean residual outperform a unit raw gap;
- a randomly jumping reference makes contrast methods worse and is reported honestly;
- a stable but constant reference provides no scene-specific normalization benefit;
- positive and negative corruption responses receive correct locked orientations;
- complete flat curves are zero-sensitivity flats, while incomplete curves are unmeasured.

### Metric and reporting tests

- five severity AUROCs and macro AUROC match hand calculations with ties;
- ranking uses severity-1 AUROC as the second criterion;
- every contrast is matched to the correct raw control;
- paired bootstrap resamples image IDs identically for candidate and control;
- bootstrap output is deterministic under seed `20260821`;
- all figure data and axis limits are present in `summary.json`;
- the output bundle contains exactly nine declared files;
- failure leaves no completed or staging output;
- the easy report contains no probability claim and states that held-out data was unused.

### Integration isolation tests

Run the command end to end on a synthetic completed corruption-sensitivity directory. Patch model
loading, cache iteration, bank loading, persistence extraction, and kNN entry points to raise if
called. The command must succeed without reaching any of them.

## Success criteria

The experiment succeeds operationally when it produces all nine declared outputs from the saved
tuning scene-score artifacts, passes the validation and isolation tests, and leaves existing
analysis outputs unchanged.

The internal-reference hypothesis is supported on tuning only when all of the following hold:

- for at least one fixed pair and summary, the anchor's stability-to-spread ratio is below 1.0
  at every corrupted severity 1 through 5;
- one contrast on that pair and summary has complete coverage and macro AUROC above its matched
  raw responsive control;
- the paired bootstrap's 95 percent interval for that macro AUROC improvement remains above zero;
- the contrast's severity-1 AUROC is at least as high as the matched raw control's severity-1
  AUROC, so severe blur does not hide a mild-blur loss;
- if the winning contrast is the clean-predicted residual, its cross-fitted median absolute
  prediction error is lower than the fold-specific constant-median predictor's error;
- the report preserves uncertainty from the paired bootstrap and does not describe tuning
  selection as held-out validation.

If those conditions fail, the correct result is that the 0--10 percent range is not a reliable
within-image reference under this design. Do not proceed to threshold selection or probability
calibration from a negative or inconclusive result.

## Non-goals

- No paired clean image at deployment.
- No rerun of DETR, persistence extraction, bank construction, or nearest-neighbour search.
- No learned neural head or multivariate classifier.
- No search over arbitrary bucket pairs or mixed scene summaries.
- No use of earlier decoder layers or combined-layer scores.
- No binary deployment threshold in this phase.
- No corruption probability calibration.
- No held-out evaluation until the tuning conclusion and one final candidate are reviewed and
  frozen.
