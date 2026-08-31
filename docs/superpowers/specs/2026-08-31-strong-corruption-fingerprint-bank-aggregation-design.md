# Strong-Corruption Fingerprint Bank and Aggregation Selection Design

Date: 2026-08-31

Status: Approved design

## Purpose

Determine whether the layer-2 persistent-graph fingerprint provides useful
information for detecting strongly corrupted images, and select one fixed
reference-bank, query-distance, and query-to-image aggregation configuration
without confusing the effects of those choices.

The scientific target is image-corruption detection. It is not detector mAP,
per-image detection accuracy, calibration, localization quality, or the number
of failed detections in a scene.

## Scope

The development study uses the existing disjoint COCO reference and evaluation
rosters:

- 1,000 clean reference images for bank construction;
- 250 evaluation images;
- all 19 configured corruption families; and
- only severity 0, severity 4, and severity 5.

Severity 0 supplies the clean negative class. Severities 4 and 5 are evaluated
separately and then averaged with equal weight. Severities 1 through 3 do not
participate in cache loading, nearest-neighbour scoring, configuration
selection, or the final strong-corruption report. Existing caches may contain
them, but the strong-corruption workflow never opens those records.

The intended final experiment is the untouched 2,500-reference/2,500-evaluation
run. The final bank is rebuilt from its 2,500 clean reference images using the
fully frozen bank algorithm, capacity, metric, and seed. No design choice may be
changed after final evaluation begins.

## Non-goals

- No learned query-pooling model.
- No learned image-level aggregation model.
- No claim about weak or moderate corruptions.
- No selection based only on Gaussian blur.
- No class-specific bank routing at evaluation time.
- No claim that persistence is better than the raw decoder embedding. The raw
  embedding is not stored in the current cache, so that comparison requires a
  separate extraction ablation.

## Why the experiment must separate its axes

The existing results compare banks that changed reference-image count, query
composition, capacity, compression, and image aggregation at the same time.
Those comparisons cannot identify which choice caused a score to improve,
weaken, or reverse direction.

This study treats the complete scored method as three explicit stages:

```text
clean reference fingerprints
    -> reference-bank policy

evaluation query fingerprint + bank
    -> query-to-bank score

all valid query scores for one image
    -> deterministic image-level score
```

Each stage has a named policy and provenance. Candidate comparisons either
hold the other axes fixed or cross them explicitly and retain the complete
audit slice. Final selection chooses the complete policy tuple because the
stages may interact.

## Fixed detector and fingerprint representation

The study reuses the frozen RT-DETRv2-R18 detector, checkpoint, preprocessing,
decoder layer 2, 300 detector queries, 80 classes, 335-dimensional persistence
fingerprint, and union-padding removal policy already recorded by the current
benchmark artifacts.

The fingerprint remains the descending sorted maximum-spanning-tree edge
weight vector formed from the layer-2 query activation and the layer-2
classification-head weights. Larger query-to-bank distance always has the
fixed interpretation "more evidence of corruption." Orientation is never
flipped after results are observed.

For each evaluation image and corruption family, the valid-query mask is the
complement of the union of repeated padded tails detected in that group's
severity-0, severity-4, and severity-5 records. Exactly the same query IDs are
then used for all three scores in that group. A clean score can therefore be
family-specific when the union mask differs between families. This paired-mask
policy must be recorded in the report and is not presented as a deployment-time
padding policy.

## Reference-query matching

True matched and background variants use COCO annotations on the clean
reference images. They do not substitute a detector-confidence threshold for
ground-truth matching.

Matching reuses the historical RT-DETR Hungarian policy:

```text
classification cost weight: 2
box L1 cost weight:          5
generalized-IoU weight:      2
focal-loss matching:         enabled
focal alpha:                 0.25
focal gamma:                 2.0
```

For a clean reference record, its own repeated padded tail is removed before
matching. A valid query assigned to a COCO object is `matched`; every other
valid query is `background`. Here `background` means unmatched by this policy,
not necessarily that the query describes literal visual background. The
matching artifact stores image ID, query ID, matched object ID or `-1`, matched
class ID or `-1`, query confidence, and the source annotation digest.

Matching labels are used only to construct and diagnose clean reference banks.
Evaluation images need no annotations for query selection or scoring.

## Stage 1: reference-query composition

The first screen fixes the bank at 2,000 rows, uses reservoir seed 44, uses raw
Euclidean query distance, averages the five nearest distances, and tests the
approved image aggregators. It compares five candidate populations:

| Bank name | Candidate population | 2,000-row construction |
| --- | --- | --- |
| `matched` | True Hungarian-matched valid queries | 2,000-query reservoir |
| `background` | True unmatched valid queries | 2,000-query reservoir |
| `balanced` | Equal-weight matched and background mixture | Two independent 1,000-query reservoirs |
| `all_valid` | Every valid, unpadded query | 2,000-query reservoir |
| `confidence_0_5` | Valid queries with maximum sigmoid confidence at least 0.5 | 2,000-query reservoir |

The `confidence_0_5` arm is the current working-tree policy. It is named
confidence-filtered, never matched-only: it can include confident false
positives and exclude low-confidence matched objects.

Candidate iteration is deterministic in string image-ID order and then query-ID
order. Reservoir construction uses standard Algorithm R. Independent balanced
reservoir streams derive their random-number state from the declared bank seed
and the literal population labels `matched` and `background`. Specifically,
take the first eight bytes of SHA-256 over `base_seed:population_label` as an
unsigned big-endian integer and reduce it modulo `2^63`. Each reservoir
records the candidate count, retained matched/background counts, seed, and
manifest digest. A composition with fewer
candidates than its required capacity is marked infeasible rather than being
silently padded or given a smaller bank.

For this screen, each composition is evaluated with all five selectable image
aggregators. Its screening score is the score of the aggregator selected on the
inner training data, not an aggregator chosen from the held-out data. All 25
composition-by-aggregation results remain in the audit table, so a bank that is
useful only under one aggregation rule is visible rather than reported as a
general bank effect.

## Stage 2: construction and capacity sensitivity

Only the two strongest Stage-1 compositions advance. For each advancing
composition, compare:

- a 2,000-row seeded reservoir; and
- 2,000 global k-means centroids built from the complete eligible candidate
  population.

Global k-means uses the current exact Lloyd implementation: distinct candidate
indices sampled for initialization, at most 50 iterations,
maximum-centroid-shift tolerance `1e-4`, and deterministic reseeding of empty
clusters from the farthest currently assigned candidates. Distance ties use
canonical candidate order. For `balanced`, run two independent global k-means
problems over the complete matched and complete background populations, using
the same label-derived seed rule as the reservoirs. Give each problem half the
requested centroids and concatenate matched centroids before background
centroids. This preserves the declared 50/50 mixture instead of allowing the
much larger background population to dominate.

Within each advancing composition, the winning construction receives a
capacity sensitivity check at 1,000, 2,000, and 5,000 rows. The result is one
construction-and-capacity policy for each of the two advancing compositions. A
capacity is reported as infeasible when any required subpopulation cannot
fill its allocation; it is never fabricated by duplication. Capacity
sensitivity describes robustness and does not allow a configuration to bypass
nested selection.

The five bank-sensitivity seeds are 42, 43, 44, 45, and 46. Selection uses the
mean result across seeds, never the best seed. The final frozen run uses seed
44. K-means metadata includes initialization seed, candidate count, iteration
count, tolerance, cluster counts, and empty-cluster handling. Construction and
capacity comparisons allow all five image aggregators and choose an aggregator
only on inner data.

## Stage 3: query-to-bank score

Stage 1 uses the current query score as its fixed screen:

```text
mean_5_euclidean = mean Euclidean distance to the five nearest bank rows
```

For the two advancing bank policies, evaluate this four-score panel:

1. `mean_5_euclidean`: mean raw Euclidean distance to neighbours 1 through 5.
2. `fifth_neighbor_euclidean`: raw Euclidean distance to neighbour 5.
3. `mean_5_standardized_euclidean`: mean Euclidean distance to neighbours 1
   through 5 after coordinate-wise clean-reference standardization.
4. `mean_5_cosine`: mean cosine distance to neighbours 1 through 5.

Nearest-neighbour distance alone is not a separate Stage-3 arm.

Standardization statistics come only from the complete eligible clean-reference
candidate population. For `balanced`, the matched and background populations
each receive total weight one half when calculating the moments. Variance is
the population variance under those weights. A coordinate with zero reference
variance makes only the standardized-distance arm infeasible; evaluation data
never alter a mean or scale. Reservoir rows are transformed with those fixed
statistics. K-means centroids for the standardized metric are fit in
standardized space.

For cosine scoring, L2-normalize query and bank rows at scoring time; k-means
construction otherwise remains unchanged. Cosine scoring marks that metric arm
infeasible if it encounters a zero-norm fingerprint or bank row instead of
silently assigning an arbitrary similarity. All searches remain exact and
chunked; the implementation never materializes
the complete query-by-bank distance matrix.

After selecting the metric, perform a final neighbour-count sensitivity check
at 1, 5, 10, and 20. For a mean-distance winner, average neighbours 1 through
`k`; for a radius winner, use the distance to neighbour `k`. The selected `k`
must come from inner development data and is frozen before final evaluation.
The `k=1` point is a noise-sensitivity diagnostic and cannot win; selectable
values are 5, 10, and 20.

Distance-derived entropy is excluded. Normalized neighbour weights would need
an additional temperature and measure how evenly neighbours are weighted, not
whether every neighbour is far away. Detector class-logit entropy remains a
separate output baseline.

## Stage 4: deterministic query-to-image aggregation

Every selectable aggregator starts from exactly the same union-unpadded valid
query set for one image. The five candidates are:

1. `mean_all`: arithmetic mean of every valid query distance.
2. `q90_all`: nearest-rank 90th percentile of valid query distances. With `N`
   valid queries sorted in ascending order, select zero-based index
   `ceil(0.90 * N) - 1`; do not interpolate.
3. `top20_mean_all`: mean of the largest `ceil(0.20 * N)` query distances.
4. `top_confidence_query`: distance of the valid query with the largest maximum
   sigmoid class confidence. A confidence tie selects the lowest query ID.
5. `confidence_weighted_mean`: sum of `confidence * distance` divided by the
   sum of confidence over valid queries. It uses raw maximum-sigmoid confidence
   with no exponent, temperature, threshold, or fitted parameter. A nonpositive
   or nonfinite weight sum is an error.

Minimum and maximum query distance are diagnostics and cannot win selection.
The existing 50--60 versus 90--100 relative-gap score is retained as a legacy
comparator and cannot win the new selection.

## Strong-corruption evaluation target

For each corruption family, calculate two separate AUROCs:

```text
level_4_auroc = clean severity 0 versus severity 4
level_5_auroc = clean severity 0 versus severity 5

family_strong_score = average(level_4_auroc, level_5_auroc)
```

The primary configuration-selection score is the equal-weight mean of the 19
`family_strong_score` values. Each family therefore contributes equally,
regardless of its number of images or score variance. Level 4 and level 5 are
never pooled into one positive-class sample before AUROC calculation.

The report also records the median of the 19 family scores, the mean level-4
AUROC, and the mean level-5 AUROC. An overall mean is a selection summary, not a
replacement for the 19 per-family results. Using the same bootstrap image-ID
draws for all methods, it also reports paired AUROC differences between the
fingerprint and each detector-output baseline. Every interval uses 10,000
percentile-bootstrap draws with fixed seed 20,260,821 and endpoints at the
2.5th and 97.5th percentiles.

## Information beyond direct confidence

Standalone AUROC can reward a fingerprint score that merely reproduces direct
model confidence. The scientific complementarity check therefore measures
fingerprint concordance only among clean-corrupted image pairs with similar
direct maximum confidence.

At each validation split, fit the confidence strata separately for severity 4
and severity 5 without using the held-out image IDs or held-out corruption
family:

1. In the training partition, form one clean-corrupted pair for every available
   image and corruption family at the given severity. The clean value is
   repeated once per family, giving clean and corrupted values equal total
   weight.
2. Pool and sort the direct maximum-confidence values from both members of
   those training pairs. For decile `j` from 1 through 9, place the boundary
   halfway between sorted positions `ceil(j * M / 10) - 1` and
   `ceil(j * M / 10)`, where `M` is the pooled count.
3. Collapse repeated boundaries and record the resulting number of
   nonempty-width strata. Assign a boundary-valued score to the higher bin,
   equivalent to `searchsorted(boundaries, score, side="right")`.
4. Carry those boundaries unchanged to the held-out family and image IDs.
5. Compare a clean and corrupted held-out image only when both fall in the same
   confidence stratum.
6. Count a correctly ordered fingerprint pair as 1, a tie as 0.5, and an
   incorrectly ordered pair as 0.

Each family/severity task score is the mean over its eligible pairs. A family
score averages its level-4 and level-5 task scores, and the aggregate gives each
of the 19 families equal weight. The report includes contributing-pair counts,
collapsed boundaries, and empty strata. A configuration cannot support the
conditional-information claim if any of the 38 family/severity tasks has no
eligible pair.

A paired bootstrap resamples complete image IDs so all corruptions, methods,
and both strong severities for one image move together. The final untouched
run uses severity-specific boundaries fitted once on all development families
and freezes those boundaries before final scores are inspected.

This is a diagnostic of conditional information, not a deployable learned
pooling model. No learned query aggregator is introduced.

## Nested development selection

The development estimate crosses five fixed image-ID group folds with
leave-one-corruption-family-out evaluation. Each image ID belongs to one fold,
and each of the 95 outer cells therefore holds out both complete image
identities and one complete corruption family. Within an outer training
partition, inner validation crosses the remaining four image groups with the
remaining 18 corruption families.

Inside each outer training partition:

1. Screen all five reservoir bank compositions at capacity 2,000 and seed 44
   with `mean_5_euclidean`, crossing each with the five aggregators.
2. For each composition, retain its inner-selected aggregator and rank the five
   compositions; advance the two strongest compositions.
3. Within each advancing composition, compare reservoir versus global k-means
   at capacity 2,000 over seeds 42 through 46, then compare capacities 1,000,
   2,000, and 5,000 for that composition's chosen construction.
4. Cross the resulting two bank policies with the four query-to-bank scores and
   all five selectable image aggregators, using the mean across the five bank
   seeds.
5. Apply the neighbour-count check to the leading metric families, recording
   `k=1` as diagnostic-only.
6. Select one complete policy tuple without reading the outer cell.

At every stage, the primary ranking statistic is the equal-family mean strong
AUROC. Exact ties are broken by higher confidence-conditioned concordance, then
higher minimum-seed strong AUROC, then smaller bank capacity, then
lexicographic configuration ID. Every held-out prediction is produced by a
choice made without that prediction. A candidate that is infeasible in any
required inner cell cannot advance.

The 95 outer cells collectively estimate the performance of this selection
procedure; they are not presented as 95 independent estimates of one fixed
tuple. After nested evaluation is complete, repeat the staged selection on all
development data using the 95 crossed held-out cells as validation predictions.
Apply the qualification gates to each finalist's cross-validated predictions,
select one final tuple, and then freeze it before running the untouched final
experiment once.

A configuration qualifies for final selection only when:

- the lower bound of its 95% paired image-bootstrap interval for the aggregate
  family strong AUROC is above 0.5;
- the lower bound of its 95% paired image-bootstrap interval for
  confidence-conditioned concordance is above 0.5;
- its aggregate level-4 AUROC and aggregate level-5 AUROC are each above 0.5;
- the median of its 19 family strong scores is above 0.5; and
- every bank-seed strong-score point estimate remains above 0.5.

Among qualifying configurations, select the highest mean family strong score.
Use the same deterministic tie breakers defined above. No direction, family
weight, threshold, or candidate list changes after seeing outer or final
results. If no configuration qualifies, the experiment reports that no
fingerprint-bank design demonstrated complementary strong-corruption
information; it does not promote the least-bad configuration.

The report keeps four evidence statements separate:

| Question | Required evidence |
| --- | --- |
| Does the fingerprint detect strong corruption at all? | Aggregate family strong-AUROC interval is above 0.5 |
| Does it retain information beyond maximum confidence? | Confidence-conditioned concordance interval is above 0.5 |
| Is it a better standalone score than maximum confidence? | Paired fingerprint-minus-confidence AUROC interval is above 0 |
| Is its bank result stable? | Every seed point estimate is above 0.5 and the seed spread is reported |

A fingerprint may retain complementary information without beating maximum
confidence as a standalone score. Because learned fusion is outside scope, that
outcome is reported as additional signal, not as an improved deployed detector.

## Baselines

The final comparison freezes these detector-output baselines:

- `direct_confidence_max`: maximum query maximum-sigmoid confidence over the
  same union-unpadded query set, with lower confidence oriented as more
  corrupted;
- `softmax_entropy_top_confidence_query`: normalized softmax Shannon entropy of
  the maximum-confidence query, with larger entropy oriented as more
  corrupted; and
- the legacy persistence relative gap, reported but not selectable.

The direct-confidence and entropy baselines are recomputed from the exact final
evaluation roster. No development AUROC such as 0.712 or 0.714 is assumed to be
the final baseline.

For the entropy baseline, first choose the query with the largest maximum
sigmoid confidence, breaking a tie by lowest query ID. Apply softmax across its
80 class logits and calculate `-sum(p * log(p)) / log(80)`. This fixes both the
query and normalization conventions.

## Final reporting

The human-readable report contains two tables with one row for every
corruption family:

| Corruption | Fingerprint L4 | Fingerprint L5 | Confidence L4 | Confidence L5 | Entropy L4 | Entropy L5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |

| Corruption | Fingerprint strong mean | Confidence strong mean | Entropy strong mean | Fingerprint minus confidence | Fingerprint minus entropy |
| --- | ---: | ---: | ---: | ---: | ---: |

Every family row includes paired image-bootstrap intervals in the detailed
artifact; the two difference columns use paired intervals from identical
bootstrap draws. The report also contains:

- confidence-conditioned concordance and its interval for every family and
  severity, plus the per-family strong mean;
- a controlled Stage-1 bank-composition table for every aggregator, followed by
  construction, capacity, distance, neighbour-count, and aggregation ablation
  tables for every candidate that advanced;
- selected-bank seed mean, standard deviation, minimum, and maximum;
- selected policy definitions and score orientations;
- candidate counts, matching composition, and k-means cluster counts when
  applicable;
- the complete nested-selection trace;
- overall equal-weight mean and median as final summary rows; and
- an explicit statement that levels 1 through 3 were not evaluated.

Machine-readable CSV and JSON artifacts contain every method, corruption,
severity, point estimate, interval, seed, configuration identifier, and
provenance digest. The overall summary can never replace or suppress a weak
per-corruption result.

## Diagnostic attribution

The report attributes performance changes only through controlled comparisons:

- Bank composition is informative when matched, background, balanced,
  all-valid, and confidence-filtered banks differ under the same construction,
  distance, seed, and aggregator. If their intervals overlap broadly and their
  per-family ordering is inconsistent, the bank labels have not demonstrated
  useful information.
- Construction or capacity is responsible when reservoir versus k-means, or
  1,000 versus 2,000 versus 5,000 rows, changes held-out results while the other
  policies remain fixed. Large seed spread indicates sampling or clustering
  instability rather than a dependable fingerprint effect.
- Query geometry is responsible when Euclidean, standardized Euclidean,
  cosine, or neighbour-radius scoring changes results for the same bank and
  aggregator.
- Scene aggregation is responsible when `mean_all`, `q90_all`,
  `top20_mean_all`, `top_confidence_query`, or `confidence_weighted_mean`
  changes results for the same query scores. Strong mean-based results support
  scene-wide degradation; a result confined to `top_confidence_query` supports
  a one-query effect.
- The fingerprint itself has not demonstrated useful corruption information
  when no policy passes the frozen AUROC and conditional-concordance gates,
  regardless of the best observed point estimate.

These are diagnostic associations within the tested panel, not causal claims
outside it.

## Components and data flow

The implementation separates five responsibilities:

- `BankPolicy`: candidate composition, matching provenance, construction,
  capacity, and seed.
- `QueryDistancePolicy`: feature transform, distance metric, neighbour count,
  and neighbour reduction.
- `ImageAggregationPolicy`: one of the five deterministic aggregators.
- `StrongCorruptionEvaluator`: levels 0, 4, and 5, per-family AUROC,
  confidence-conditioned concordance, and paired bootstrap.
- `StrongCorruptionReporter`: validation matrix, per-family final tables,
  machine-readable artifacts, and provenance.

The data flow is:

```text
clean reference cache + COCO annotations
    -> matching metadata
    -> candidate population
    -> bank artifact

cached evaluation records for levels 0, 4, and 5
    -> exact query-to-bank distances
    -> deterministic image scores
    -> per-family metrics and intervals
    -> nested-selection and final reports
```

The bank, query-distance, and aggregation interfaces are independent. Changing
one policy does not alter the stored detector outputs or silently change
another policy.

## Validation and failure behavior

The workflow refuses to continue when:

- reference and evaluation image IDs overlap;
- a reference image lacks required COCO annotations;
- matching metadata does not bind to the reference manifest, checkpoint, and
  annotation file;
- any query, bank, transformed feature, distance, weight sum, image score, or
  metric is nonfinite;
- an evaluation image-family scoring group does not resolve to exactly one
  severity-0, one severity-4, and one severity-5 record;
- a corruption family is missing, duplicated, or outside the fixed roster of
  19; or
- a resumed artifact differs in any scientific policy or provenance digest.

An individual candidate arm is instead marked infeasible, recorded, and
excluded from selection when its required population cannot fill its declared
capacity, a balanced allocation cannot be filled exactly, a standardized
coordinate has zero reference variance, or a cosine-scored vector has zero
norm. These conditions do not erase valid results from other candidate arms.

Levels 1 through 3 are not opened by the strong-corruption scorer. Existing
caches may contain them, but their presence cannot affect a result.

## Test strategy

Unit tests use hand-calculated tensors and cover:

- exact Hungarian match metadata on a small synthetic assignment;
- matched, background, balanced, all-valid, and confidence-filter masks;
- exact reservoir counts, ordering independence, seed reproducibility, and
  insufficient-candidate failure;
- k-means capacity, cluster-count metadata, deterministic initialization,
  empty-cluster handling, and separate balanced-population clustering;
- raw Euclidean, standardized Euclidean, cosine, nearest, mean-neighbour, and
  radius-neighbour scores;
- zero-variance, zero-norm, nonfinite, and mismatched-dimension failures;
- nearest-rank q90, top-20% rounding, confidence ties, weighted mean, and
  diagnostic-only aggregator exclusion;
- exact level-4, level-5, and strong-mean AUROC on toy scores;
- confidence-stratum boundaries fitted only on inner data;
- collapsed confidence boundaries and an empty conditional-comparison task;
- paired image-ID bootstrap behavior;
- outer image and corruption-family isolation;
- staged selection, deterministic tie breaking, diagnostic-only `k=1`, and
  all four evidence-statement outcomes;
- the fact that levels 1 through 3 are never read;
- exactly 19 family rows in each final per-family table, plus mean and median
  summary rows; and
- complete CSV/JSON round-trip and provenance validation.

An integration test uses a tiny synthetic reference cache and two synthetic
corruption families to exercise bank construction, scoring, nested selection,
reporting, resume validation, and deterministic reruns without detector
inference.

## Interpretation boundaries

A successful result supports only this statement: the selected layer-2
persistence-bank configuration helps rank strongly corrupted images under the
tested corruption families, and retains information after approximately
conditioning on direct maximum confidence.

It does not show that:

- the score detects mild corruption;
- the score predicts detector accuracy or localization error;
- a fingerprint distance is a calibrated probability;
- the selected bank is optimal outside the tested design panel;
- entropy or confidence is universally weaker; or
- the topological transformation adds value beyond the raw decoder activation.

Because the benchmark applies a union padding mask across paired severity
records, it also does not establish a standalone deployment-time padding
policy. That requires a separate per-image masking validation.

The last claim requires storing raw layer-2 decoder activations and repeating
the same bank, distance, and aggregation comparison with that representation.
