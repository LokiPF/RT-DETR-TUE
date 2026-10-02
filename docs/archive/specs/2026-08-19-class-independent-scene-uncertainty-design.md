# Class-Independent Scene Uncertainty from Persistence Features

Date: 2026-08-19

Status: Approved design for implementation planning

## Objective

Measure scene-level image-corruption uncertainty from the classification-head
persistence representation without training an uncertainty head. The first
experiment uses progressively stronger Gaussian blur and tests whether the raw
uncertainty rises smoothly as blur severity increases.

The uncertainty path must not use a predicted class to choose a reference.
Consequently, a detector class change such as `car -> truck` cannot switch the
comparison ruler.

## Scope

The first phase covers:

- RT-DETR classification-head persistence features only;
- the last decoder layer and, separately, all three decoder layers;
- class-independent clean reference banks built from COCO train images;
- ten non-neural mathematical comparison methods;
- query-to-scene aggregation without a learned head;
- a six-level Gaussian-blur sweep;
- calibration-free trend and class-switch diagnostics; and
- reusable feature and bank artifacts so the detector is not rerun when a
  scoring method changes.

The first phase does not include:

- bbox-head persistence features;
- a trained uncertainty or calibration head;
- a literal posterior probability that the image is corrupted;
- fog, rain, noise, or non-Gaussian blur corruptions; or
- changing detector predictions or reranking detections using uncertainty.

## Output interpretation

The primary output is a raw anomaly score: larger means less compatible with
the clean persistence-feature distribution. It is not called a corruption
probability.

For plots that compare images with different raw baselines, the experiment also
reports:

```text
clean-relative score at severity s
    = raw score at severity s - raw score for the clean image
```

This translation does not impose monotonicity. It only aligns each curve to
start at zero.

## High-level data flow

```text
COCO train reference images
    -> frozen detector
    -> classification-head persistence extraction
    -> reusable per-image feature cache
    -> class-independent banks, one per decoder layer

COCO val evaluation image
    -> clean plus five Gaussian-blur variants
    -> frozen detector
    -> reusable persistence-feature cache
    -> selected comparison method
    -> per-query raw scores
    -> mathematical scene aggregation
    -> raw scene uncertainty
    -> trend and class-switch evaluation
```

Detector logits and labels may be saved for diagnostics, but they are not an
input to bank lookup or distance calculation.

## Dataset partitioning and leakage prevention

Use COCO train 2017 only for clean reference-bank construction. Use COCO val
2017 for blur-method development and evaluation.

Before generating any blurred images, deterministically partition COCO val by
image ID with seed 42:

- the first 50 percent form the tuning partition;
- the remaining 50 percent form the final test partition.

All variants of one original image remain in the same partition. Bank images,
tuning images, and test images are disjoint. A clean evaluation image is never
looked up against its own persistence features.

The tuning partition selects non-neural settings such as neighbor count,
normalization, distance, aggregation, and transport bandwidth. The final test
partition is evaluated once using the chosen settings.

## Efficient extraction and caching

The detector forward pass and persistence construction are the expensive
steps. Run them once and save their outputs. Do not save only a single derived
prototype: later comparison methods require the original per-query persistence
features.

### Per-image feature cache

Store one record per source image and corruption level. Each record contains:

- COCO image ID;
- source partition: `reference`, `tuning`, or `test`;
- corruption type and severity;
- persistence tensor for every selected classification decoder layer;
- tensor shape and dtype;
- the fixed number of decoder queries;
- detector logits, boxes, and confidence for diagnostics only; and
- extraction metadata needed to reject stale or incompatible caches.

Feature files are sharded instead of stored as one monolithic file. Persistence
tensors may be stored as float16 to reduce disk use, but all distances and
summary statistics are computed in float32.

### Required manifest metadata

The cache manifest records:

- git commit;
- model checkpoint identifier and file hash;
- model and dataset config identifiers;
- COCO annotation file identifier;
- preprocessing and resize settings;
- persistence extraction version;
- classification layer names and decoder-layer indices;
- persistence-vector dimension;
- number of decoder queries;
- data split seed and image IDs;
- corruption parameters; and
- shard paths and record counts.

An existing cache is reused only when all compatibility fields match. A partial
run resumes at the first missing image instead of overwriting completed shards.

### Pilot and scale-up

The first useful pilot uses:

- 5,000 deterministic COCO train images for reference extraction;
- 500 deterministic COCO val images for the blur sweep;
- all decoder queries; and
- classification decoder layers 0, 1, and 2.

This pilot establishes whether any method has a useful trend before spending
the cost of full COCO extraction. If the signal is promising, scale the
reference extraction while using deterministic reservoir sampling to cap each
per-layer query bank at 100,000 vectors. Scene-level methods retain a separate
bounded set of complete clean-scene records.

## Class-independent bank construction

Build one bank for each decoder layer. Never partition a bank by predicted or
ground-truth class.

The raw bank contains clean query-persistence vectors plus image provenance.
Method-specific artifacts are derived from the same raw cache:

- robust prototype or medoid;
- nearest-neighbor index;
- local-density statistics;
- robust center and shrinkage covariance;
- kernel-density bandwidth;
- persistence-diagram medoids or landscapes;
- clean scene sets;
- MMD reference samples; and
- optimal-transport reference scenes or barycenters.

Bank reduction must remain soft at scoring time. For example, several medoids
are combined using nearest distance or a smooth distance combination rather
than routing through a predicted semantic class.

## Feature normalization variants

Evaluate the following variants independently:

1. raw persistence vector;
2. coordinate-wise robust standardization from clean-bank median and IQR;
3. unit-normalized vector; and
4. unit-normalized shape with original vector magnitude retained as a separate
   scalar.

Magnitude is not discarded by default because activation scale may carry blur
information. Normalization statistics come only from the clean reference
partition.

## Comparison methods

All methods implement the same conceptual interface:

```text
build(clean feature cache) -> method-specific clean artifact
score(test scene features, artifact) -> raw query and scene scores
```

The first experiment supports these ten methods:

1. distance from a robust clean prototype;
2. mean distance to the k nearest clean query vectors;
3. Local Outlier Factor relative to clean neighborhoods;
4. global robust Mahalanobis distance with shrinkage covariance;
5. negative clean kernel-density score;
6. topology-native distance to clean diagram medoids;
7. deviation from clean persistence-landscape or persistence-image bands;
8. nearest-clean-scene symmetric Chamfer or Hausdorff distance;
9. MMD or energy distance between the test query set and clean query sets; and
10. entropically regularized optimal-transport distance between test and clean
    query sets.

No method fits a neural uncertainty head.

## Query and scene aggregation

Use every fixed decoder query. Do not filter queries using a confidence
threshold because threshold crossings would introduce another discontinuity.

For methods that first produce per-query anomaly scores, report these scene
aggregations separately:

- mean;
- median;
- 90th percentile;
- mean of the highest 20 percent; and
- fraction above a threshold derived only from clean reference scores.

Mean and median target scene-wide corruption. Tail aggregations check whether a
smaller subset of queries contains the useful blur signal. Maximum is excluded
from the default set because it is dominated by a single unstable query.

## Multiple decoder layers

First evaluate decoder layers 0, 1, and 2 independently. Do not concatenate raw
layer vectors for the initial comparison.

For an optional combined score, normalize each layer using clean-reference
location and scale, then take the unweighted mean of the three layer scores.
Also evaluate a trajectory variant that compares clean and test changes from
layer 0 to 1 and from layer 1 to 2. The trajectory reference is still
class-independent.

## Gaussian-blur sweep

Resize images to the detector's standard 640 by 640 input size, then apply PIL
Gaussian blur before tensor conversion. The six blur radii are:

```text
severity: 0   1   2   3   4   5
radius:   0   1   2   4   8  12
```

Severity 0 follows the existing clean inference preprocessing exactly. All six
variants use the same detector checkpoint, clean bank, comparison settings, and
scene aggregation.

## Trend evaluation

For each method, layer, normalization, and aggregation combination, compute:

1. median raw score by blur severity with the 25th and 75th percentiles;
2. median clean-relative score by severity;
3. per-image Spearman rank correlation between severity and raw uncertainty;
4. adjacent monotonicity rate: the fraction of adjacent severity steps whose
   uncertainty does not decrease;
5. monotonicity violation magnitude: the sum of downward changes, normalized by
   that image's observed score range;
6. endpoint separation: the fraction of images for which severity 5 exceeds
   severity 0; and
7. runtime, peak memory, artifact size, and query-bank size.

Monotonic regression is not applied to the scores. Monotonicity is an observed
property used to compare methods, not a property forced after measurement.

## Class-switch diagnostic

Class changes are diagnostic annotations only. Because COCO annotations remain
valid under blur, associate predictions with the same ground-truth object at
each severity using the detector's existing matching logic. Mark adjacent
severity steps where the matched predicted class changes.

Report monotonicity rate and violation magnitude separately for:

- steps containing at least one matched-object class switch; and
- steps with no matched-object class switch.

Also include a regression check in which identical persistence features are
paired with different argmax labels. Their uncertainty must be identical.

## Artifacts and reports

Each experiment run produces:

- an immutable extraction manifest;
- sharded persistence-feature files;
- a method-specific bank artifact and manifest;
- per-image and per-severity raw results in a tabular file;
- a summary JSON containing all trend metrics;
- raw-score and clean-relative trend plots; and
- class-switch diagnostic plots and summaries.

Every report records the exact feature-cache and bank-manifest identifiers so a
result can be reproduced without rerunning the detector.

## Failure handling

- Reject caches and banks whose checkpoint, layer, feature dimension,
  preprocessing, or split metadata do not match the requested run.
- Fail if an evaluation image ID appears in the reference manifest.
- Fail if the query count changes between severity levels of one image.
- Record but do not discard images with missing detections; persistence scoring
  uses the fixed decoder queries rather than post-threshold detections.
- Write shards and reports atomically so interrupted runs can resume safely.

## Verification

Unit tests cover:

- deterministic, disjoint image splitting;
- exact severity-0 preprocessing equivalence;
- blur-level ordering and reproducibility;
- class-independent bank lookup;
- cache compatibility and stale-cache rejection;
- raw-to-clean-relative score conversion;
- monotonicity metrics on increasing, flat, and decreasing synthetic curves;
- scene aggregations with fixed query counts;
- independent per-layer scoring; and
- identical uncertainty when only the predicted argmax label changes.

A small integration test extracts a bank from a few images, runs all six blur
levels for separate images, evaluates at least prototype and kNN scoring, and
checks that every artifact contains the expected image, layer, and severity
metadata.

## Acceptance criteria

The first phase is complete when:

- clean persistence features are extracted once and reused across scoring
  methods;
- the same fixed, class-independent bank scores every blur level;
- no uncertainty computation reads a predicted class or hard confidence mask;
- raw per-query, per-layer, and per-scene values are saved;
- all trend and class-switch metrics are reported;
- at least the pilot data sizes complete reproducibly; and
- the report makes no unsupported claim that a raw anomaly score is a
  corruption probability.
