# Within-Image Corruption Contrast Design

Date: 2026-08-21

## Goal

Create a deployment-oriented corruption score from one image at one timestamp. The score
combines two confidence-percentile ranges measured on that same image. It must not require a
paired clean version of the deployed image.

The experiment asks whether a within-image contrast removes scene-to-scene changes in absolute
persistence distance and ranks corrupted images more reliably than the responsive percentile's
raw distance.

Two contrast families are tested. An anchored contrast pairs a range that barely responds to
blur with a range that responds strongly, and uses the first as a scene-baseline reference. A
differential contrast pairs two ranges that both respond to blur in opposite directions, so the
subtraction adds the two responses instead of normalizing one against a flat baseline. Both
families use identical arithmetic. They differ only in what the first range is expected to do,
and therefore in which diagnostics decide whether the result means anything.

## Decisions already made

- Deployment receives one image only. There is no clean partner from the same scene or timestamp.
- Both the reference and responsive scores are computed from the current image.
- The primary anchored reference is the dynamic, padding-filtered 0--10 percent confidence
  decile.
- The primary responsive range is the dynamic, padding-filtered 50--60 percent confidence
  decile.
- A wider secondary comparison uses the 0--20 percent and 40--60 percent confidence quintiles.
- A differential comparison pairs the dynamic, padding-filtered 90--100 percent confidence
  decile with the same 50--60 percent responsive range. The completed deployment analysis
  measured those two ranges responding to blur in opposite directions.
- Persistence distance at decoder layer 2 is the signal for every arm. The differential arm is
  additionally evaluated at the combined-layer score, because the completed analysis measured
  the 90--100 percent decile ranking better there than at layer 2. No other layer enters this
  experiment.
- Every candidate is compared against a confidence-only twin built from the same two ranges, so
  a persistence result that merely restates detector confidence is exposed rather than claimed.
- The differential arm and its combined-layer variant were chosen after reading tuning results.
  Their tuning numbers are selection estimates, not unbiased ones, and they cannot support a
  claim without held-out confirmation. The two original pairs were declared before the data and
  carry no such debt.
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

## Evidence from the completed deployment analysis

The `analyze-corruption-sensitivity` tuning run over 250 images and six blur severities measured
every confidence bucket separately. The numbers below are its dynamic, padding-filtered, layer-2,
`mean`-summary persistence candidates. They are the reason this design carries a differential arm
in addition to the anchored one.

| Range | Locked orientation | Median signed Spearman | AUROC severity 1 | AUROC severity 5 | Macro AUROC |
|---|---|---|---|---|---|
| `decile_00_10` | -1 | -0.200 | 0.502 | 0.548 | 0.553 |
| `decile_50_60` | +1 | +0.629 | 0.474 | 0.843 | 0.610 |
| `decile_90_100` | -1 | -0.829 | 0.530 | 0.887 | 0.685 |

Three findings follow.

First, the anchored hypothesis survives. The 0--10 percent decile is close to flat, at signed
Spearman -0.029 under `q90`, -0.086 under `top20_mean`, and -0.200 under `mean`. That is what an
internal reference is supposed to look like. The 90--100 percent decile cannot replace it: a range
moving at signed Spearman -0.829 is not an anchor.

One range is flatter still. The 80--90 percent decile reaches signed Spearman +0.029 under `q90`
and exactly zero under `mean`, and its macro AUROC sits at 0.502, which is what a range carrying
no corruption information looks like. Flatness alone does not make an anchor: the reference must
also spread between scenes and predict the responsive range's normal level, and the completed
analysis measured neither of those. The 80--90 percent decile is recorded here as an observation
for a later round. Adding it now would be exactly the open-ended pair search this design forbids.

Second, the 50--60 and 90--100 percent deciles move in opposite directions. The responsive range
rises with blur and the top decile falls. Their difference therefore adds two responses rather
than normalizing one against a flat baseline, which is a different and better-powered
construction than the anchored pair can produce. The 90--100 percent decile was also the
strongest single mover among the persistence candidates in the completed sweep, at the largest
absolute median signed Spearman of any of them.

Third, the top decile carries a redundancy risk that the responsive range does not. The matched
confidence-only control gives the following, on the same dynamic, padding-filtered, `mean`
configuration:

| Range | Locked orientation | Median signed Spearman | Macro AUROC |
|---|---|---|---|
| `decile_00_10` | -1 | -0.600 | 0.557 |
| `decile_50_60` | -1 | -0.514 | 0.534 |
| `decile_90_100` | +1 | +0.829 | 0.642 |

At the 90--100 percent decile, persistence and detector confidence have the same magnitude of
median signed Spearman, 0.829, with opposite signs, and confidence alone already reaches macro
AUROC 0.642. That pattern is what a restatement of confidence looks like. At the 50--60 percent
decile the two disagree in both sign and strength, so persistence there carries information
confidence does not. The confidence-only twin is mandatory for this reason.

Two further measurements shape the design:

- The 90--100 percent decile scores macro AUROC 0.723 at the combined-layer scope against 0.685
  at layer 2. The layer-2 restriction was declared before that was known, so the differential arm
  is evaluated at both scopes.
- The 90--100 percent decile retains about 20 percent of its severity-zero dynamic membership at
  severity 5, against about 5 percent for the 0--10 and 50--60 percent deciles. Its aggregate
  population is the most stable of the three, which matters because every range here is a dynamic
  percentile population rather than a fixed set of query IDs.

Finally, the regime that matters is not where the completed sweep looks strongest. Of the 510
candidates it evaluated, 44 passed its deployability gate, and every one of those sits near chance
at severities 1 and 2. The best severity-1 AUROC anywhere in the sweep is 0.538 and the best
severity-2 is 0.570, both from `quintile_80_100`. The winning candidate reaches only 0.530 and
0.569. This experiment exists to attack that mild-blur blindness. Macro AUROC will improve for
reasons unrelated to the hypothesis, so severity-1 and severity-2 behavior decides the outcome.

## Inference interpretation

At inference, both inputs come from the current image:

```text
current image
  -> remove padded query slots
  -> rank valid queries by current detector confidence
  -> summarize persistence distance in the reference range at the arm's scope
  -> summarize persistence distance in the responsive range at the same scope
  -> combine the two summaries into one corruption score
```

In an anchored arm the reference score is used as a proxy for scene-specific baseline distance.
It is not treated as the responsive range's clean value. The experiment must first establish the
normal clean relationship between the two ranges.

In a differential arm the first range is not a baseline proxy at all. It is a second corruption
response that happens to move the other way, and the same subtraction accumulates both responses.
The arithmetic below is shared, so the word "reference" continues to name the first slot in every
formula, but only anchored arms are entitled to the baseline-removal interpretation.

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
- persistence rows are present at score scope `layer_2` for every declared range, and at score
  scope `combined` for the ranges the combined-layer differential arm uses;
- confidence rows are present at score scope `confidence` for every declared range under the same
  membership, padding mode, and aggregations, so every candidate has a confidence-only twin;
- aggregation is one of `mean`, `q90`, and `top20_mean`;
- the source summary's provenance agrees with every source row;
- no source row key is missing or duplicated.

Refuse a source containing held-out rows rather than filtering them silently.

## Fixed arms

The experiment has four predeclared arms. An arm is one bucket pair at one persistence scope.

| Arm name | Family | Scheme | Scope | Reference range | Responsive range | Role |
|---|---|---|---|---|---|---|
| `decile_00_10__50_60` | anchored | decile | `layer_2` | `decile_00_10` | `decile_50_60` | primary |
| `quintile_00_20__40_60` | anchored | quintile | `layer_2` | `quintile_00_20` | `quintile_40_60` | wider stability check |
| `decile_90_100__50_60` | differential | decile | `layer_2` | `decile_90_100` | `decile_50_60` | opposite-direction contrast |
| `decile_90_100__50_60__combined` | differential | decile | `combined` | `decile_90_100` | `decile_50_60` | same contrast at its stronger scope |

Do not search beyond these four. The purpose is to test two predeclared hypotheses, not to
manufacture a winner from a large contrast sweep. Four arms by three summaries by four score
methods is 48 candidates, every one of them named before the command runs.

For each arm, use matched `mean`, `q90`, and `top20_mean` summaries. Never subtract a mean from a
q90 or otherwise mix scene summaries. Never mix scopes inside one arm: both sides of a contrast
come from the same persistence scope.

The two anchored arms were declared before any tuning result was read. The two differential arms
were not: they were added after the completed deployment analysis showed the 90--100 percent
decile to be the strongest mover in the sweep and the 50--60 percent decile to move against it.
Record that provenance in `summary.json` as a per-arm `declared_before_data` boolean, carry it
into `candidate_metrics.csv`, and state it in the easy report. A differential arm's tuning macro
AUROC is a selection estimate. It may motivate a held-out test. It may not stand in for one.

## Four score methods

Let `reference` be the current image's reference-bucket scene score and `responsive` the current
image's responsive-bucket scene score. Both are non-negative query-to-clean-bank persistence
distance summaries at the arm's scope, in the same units.

The four methods apply unchanged to every arm and to every confidence-only twin. In a
differential arm, read "reference" as the first slot rather than as a baseline.

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

Fit a separate relationship for each arm, scene summary, and signal. Do not share one slope
across deciles, quintiles, scopes, aggregations, or signals.

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

After cross-fitted evaluation, fit one final line for each configuration on all 250
severity-zero tuning images. There are 21 configurations: twelve persistence ones, being four
arms by three summaries, and nine confidence ones, being three bucket pairs by three summaries.
The confidence count is lower because confidence has a single scope, so the two differential arms
share one confidence-only twin. Store the five fold-specific lines and the final all-clean line
for every configuration in `summary.json`. The figures may show these final lines, but the final
lines are never used to report tuning residual performance. If a residual is ultimately chosen,
only its one final slope and offset become deployment constants.

## Anchor diagnostics

Evaluate whether the reference behaves like an internal anchor before interpreting a contrast.

Run every diagnostic in this section on all four arms, and report all of it. The diagnostics gate
only the anchored arms. A differential arm is expected to fail them: its reference range responds
to blur by construction, at median signed Spearman -0.829 in the completed analysis, so it will
show large within-image drift and a stability-to-spread ratio well above 1.0. That is the
declared behavior of a differential arm, not a defect and not a reason to exclude it. Record the
failure, label the arm `differential`, and judge it on the contrast results and the
confidence-only twin instead. Never report a differential arm as a validated internal reference.

### Within-image corruption drift

For each arm, summary, and severity, calculate the reference score minus the same image's
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

For each arm and summary, report on severity-zero rows:

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

For each image, arm, aggregation, and score method, use all six severity scores to
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

Each arm, aggregation, and score method is one candidate. Choose one orientation for that
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
- one of the four declared arms;
- one of the three matched summaries;
- one of the four declared score methods;
- a computable confidence-only twin.

Rank eligible candidates by:

1. higher macro AUROC;
2. higher severity-1 AUROC;
3. higher severity-2 AUROC;
4. higher median per-image absolute Spearman;
5. higher dominant-direction fraction, including flat images in the denominator;
6. higher oriented adjacent-step consistency;
7. deterministic arm, aggregation, and method name.

Severity 1 and severity 2 sit high in this order on purpose. The completed analysis left every
candidate near chance there, so a macro gain driven entirely by severities 4 and 5 repeats a
result already in hand.

Every contrast must also be compared with the raw responsive control from the same arm and
aggregation. Report the difference in all five severity AUROCs and macro AUROC. A contrast is
"promising" only when its macro AUROC exceeds the matched raw control and it retains complete
coverage. A higher point estimate is tuning evidence, not confirmation.

### Raw reference control

A contrast must beat both of the ranges it is built from, not just the responsive one. A
differential arm subtracts two responsive ranges, so a gap that beats the 50--60 percent range
while losing to the 90--100 percent range has discovered nothing except that the top decile is the
stronger input.

For every arm and aggregation, calculate the raw reference range's own five severity AUROCs and
macro AUROC, using an orientation locked from the reference range's own median signed tuning
Spearman. This is a reported control, not a ranked candidate: it adds no rows to the candidate
list and does not change the count of four declared score methods.

Report each contrast's macro and per-severity differences against both the raw responsive control
and the raw reference control. A contrast that fails to beat either input is reported as failing,
whatever its absolute macro AUROC.

### Confidence-only redundancy control

A persistence contrast is worth reporting only when it beats the same contrast built from
detector confidence. Confidence is available for free at deployment; persistence is not.

For every candidate, build a confidence-only twin: the same bucket pair, the same aggregation,
the same score method, and the same fold-cross-fitted procedure, with the source's confidence
scene scores at score scope `confidence` substituted for the persistence scene scores on both
sides. Fit the twin's robust line from its own clean rows. Lock the twin's orientation from its
own median signed tuning Spearman, never from the persistence candidate's.

Report for every candidate:

- the twin's five severity AUROCs and macro AUROC;
- the persistence-minus-confidence difference at each severity and at macro;
- a paired image bootstrap of the macro difference, using the same seed and procedure as the
  control comparison below;
- a `confidence_redundant` boolean, true when the persistence candidate's macro AUROC does not
  exceed its twin's.

A candidate marked `confidence_redundant` may still be ranked and plotted, but the easy report
must say plainly that it adds nothing over the detector's own confidence. This control exists
because the completed analysis measured persistence and confidence at the 90--100 percent decile
with equal-magnitude, opposite-sign trends, which is exactly the shape a redundant signal takes.

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

Use twelve panels: four arms by three summaries. Each panel shows raw, un-oriented median and
interquartile bands over severities 0 through 5 for the reference and responsive scores. The two
lines in one panel share an axis because they have the same units. In the differential panels the
two lines are expected to diverge rather than track each other. Do not force unrelated scene
summaries onto one shared numerical scale.

### `clean_anchor_relationship.png`

Use the same twelve-panel layout. Each panel shows the 250 severity-zero reference/responsive
points and the final robust clean line. Axes say which bucket is the reference and which is
responsive.
The caption states that cross-fitted residual metrics use fold-specific lines even though the
figure displays the final deployment line.

### `contrast_scores_by_severity.png`

Show raw responsive, raw gap, relative gap, and cross-fitted residual in separate panels because
their units and ranges differ. Each panel uses a median line and interquartile band across images.
Plots show raw candidate direction; do not orient curves for presentation.

### `auroc_by_blur_severity.png`

For each arm and aggregation, show all four methods on one AUROC plot with corruption severity 1
through 5 on the x-axis and AUROC from 0 to 1 on the y-axis. Draw each method's confidence-only
twin as a dashed line in the same color as its persistence candidate, so redundancy is visible
without a separate figure. Include a horizontal chance line at 0.5. These panels can share a
y-axis because AUROC has the same meaning for every method.

Record all plotted values, panel selections, and axis limits in `summary.json`.

## Outputs

The output directory contains exactly:

- `per_scene_contrasts.csv`: one row per image, severity, arm, signal, aggregation, and method,
  with raw inputs, derived score, fold, fit provenance, and per-image trend fields;
- `anchor_diagnostics.csv`: anchor drift, between-image spread, and clean-relationship metrics for
  all four arms, each row carrying its arm family so differential failures are not misread;
- `candidate_metrics.csv`: one row per arm, aggregation, and method with orientation, five AUROCs,
  macro AUROC, trend metrics, coverage, the raw responsive and raw reference control
  comparisons, the confidence-only twin's AUROCs and macro difference, the `confidence_redundant`
  flag, and `declared_before_data`;
- `summary.json`: source provenance, fixed configuration, arm table with family and
  `declared_before_data`, validation counts, fold assignments, fold-specific fits, final fits,
  anchor diagnostics, candidate metrics, raw reference control metrics, confidence-twin metrics,
  bootstrap intervals, figure data, axis limits, and ranking;
- `anchor_and_responsive_actual_distance.png`;
- `clean_anchor_relationship.png`;
- `contrast_scores_by_severity.png`;
- `auroc_by_blur_severity.png`;
- `easy-report.md`.

The easy report uses plain language and leads with:

1. whether the low-confidence range is numerically stable enough to act as an anchor;
2. whether the anchor explains scene-to-scene raw-distance variation;
3. whether raw gap, relative gap, or clean residual beats both of its inputs, meaning the matched
   raw responsive distance and the matched raw reference distance;
4. whether each reported candidate beats its confidence-only twin, stated plainly in words, and
   named as adding nothing over detector confidence when it does not;
5. performance at every blur severity, with severities 1 and 2 given first because that is the
   regime the completed analysis could not separate;
6. whether the bootstrap comparison is supported or inconclusive on tuning;
7. the anchored-versus-differential distinction, and for any differential result the explicit
   statement that the arm was chosen after reading tuning data and needs held-out confirmation;
8. the final slope and offset if the residual wins;
9. the statement that no held-out images were used and the result is not a calibrated
   probability.

## Error handling

Fail before writing a final output directory when:

- the source directory or required file does not exist;
- the source artifact is unfinished or has an unknown schema;
- source provenance disagrees between CSV and JSON;
- any selected row is outside the tuning partition;
- image count is not 250 or the exact six severities are not complete;
- a required arm, summary, or persistence row is missing at that arm's scope;
- a confidence row needed for a candidate's confidence-only twin is missing;
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
- the final all-clean fit is stored but never substituted for cross-fitted tuning residuals;
- exactly 21 final fits are stored, twelve persistence and nine confidence;
- a persistence fit and its confidence twin's fit are independent, so perturbing one leaves the
  other unchanged.

### Synthetic behavior tests

- additive scene offsets make the raw gap outperform raw responsive distance;
- multiplicative scene scaling makes relative gap outperform raw gap;
- a non-unit clean relationship makes the clean residual outperform a unit raw gap;
- a randomly jumping reference makes contrast methods worse and is reported honestly;
- a stable but constant reference provides no scene-specific normalization benefit;
- positive and negative corruption responses receive correct locked orientations;
- complete flat curves are zero-sensitivity flats, while incomplete curves are unmeasured;
- a reference and responsive range moving in opposite directions make the signed raw gap
  outperform both raw ranges, which is the differential arm's premise;
- a differential arm whose anchor diagnostics fail is still ranked, still plotted, and labelled
  `differential` rather than dropped;
- a synthetic persistence signal that is an exact monotone function of confidence is marked
  `confidence_redundant`, and one carrying independent information is not.

### Metric and reporting tests

- five severity AUROCs and macro AUROC match hand calculations with ties;
- ranking uses severity-1 AUROC as the second criterion and severity-2 AUROC as the third, proven
  by two candidates that tie on macro and separate only on severity 2;
- every contrast is matched to the correct raw responsive control from its own arm and
  aggregation, and to the correct raw reference control;
- the raw reference control's orientation is locked from the reference range's own Spearman, and
  a contrast that beats the responsive input but loses to the reference input is reported as
  failing;
- every candidate is matched to the correct confidence-only twin, and the two differential arms
  resolve to the same twin because confidence has one scope;
- a twin's orientation is locked from the twin's own Spearman, proven by a case where candidate
  and twin lock to opposite orientations;
- paired bootstrap resamples image IDs identically for candidate and control, and for candidate
  and confidence twin;
- bootstrap output is deterministic under seed `20260821`;
- all figure data and axis limits are present in `summary.json`;
- every arm carries the correct `declared_before_data` value into both `candidate_metrics.csv`
  and `summary.json`, and the easy report states it for any differential result it reports;
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

- for at least one anchored arm and summary, the anchor's stability-to-spread ratio is below 1.0
  at every corrupted severity 1 through 5;
- one contrast on that arm and summary has complete coverage and macro AUROC above both its
  matched raw responsive control and its matched raw reference control;
- the paired bootstrap's 95 percent interval for that macro AUROC improvement remains above zero;
- the contrast's severity-1 AUROC is at least as high as the matched raw control's severity-1
  AUROC, so severe blur does not hide a mild-blur loss;
- the contrast's macro AUROC exceeds its confidence-only twin's;
- if the winning contrast is the clean-predicted residual, its cross-fitted median absolute
  prediction error is lower than the fold-specific constant-median predictor's error;
- the report preserves uncertainty from the paired bootstrap and does not describe tuning
  selection as held-out validation.

If those conditions fail, the correct result is that the 0--10 percent range is not a reliable
within-image reference under this design. Do not proceed to threshold selection or probability
calibration from a negative or inconclusive result.

The differential hypothesis is a separate verdict with its own bar. It is worth carrying to a
held-out test only when all of the following hold on a differential arm:

- the signed raw gap or relative gap has complete coverage and macro AUROC above both its matched
  raw responsive control and its matched raw reference control, so the contrast beats each of its
  two inputs rather than inheriting the stronger one;
- its macro AUROC exceeds its confidence-only twin's, and the paired bootstrap interval for that
  difference remains above zero;
- its severity-1 and severity-2 AUROCs both exceed the best severity-1 and severity-2 values the
  completed deployment analysis achieved anywhere, 0.538 and 0.570, because separating mild blur
  is the only outcome this design adds. Those two maxima come from the `quintile_80_100` range,
  which no arm here uses, so they are a genuine outside bar rather than a restatement of this
  design's own inputs.

A differential arm meeting that bar is a candidate for held-out evaluation and nothing more. It
was selected on tuning data, so its tuning macro AUROC may not be reported as its performance.
Anchored and differential verdicts are reported separately. Neither substitutes for the other,
and a differential success does not rescue an anchored failure.

## Non-goals

- No paired clean image at deployment.
- No rerun of DETR, persistence extraction, bank construction, or nearest-neighbour search.
- No learned neural head or multivariate classifier.
- No search over arbitrary bucket pairs, mixed scopes, or mixed scene summaries.
- No use of decoder layers other than layer 2, and no combined-layer score outside the one
  declared differential arm.
- No held-out claim from any differential arm on the strength of tuning numbers alone.
- No binary deployment threshold in this phase.
- No corruption probability calibration.
- No held-out evaluation until the tuning conclusion and one final candidate are reviewed and
  frozen.
