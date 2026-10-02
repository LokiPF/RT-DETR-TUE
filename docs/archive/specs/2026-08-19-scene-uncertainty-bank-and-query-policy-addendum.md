# Scene-Uncertainty Bank and Query-Policy Addendum

Date: 2026-08-19

Status: Approved design amendment

This document amends the bank construction and query-selection sections of
`2026-08-19-class-independent-scene-uncertainty-design.md`. Where the two
documents differ, this addendum takes precedence.

## Reference-image coverage

Construct the 5,000-image COCO train reference sample in two deterministic
parts:

- a natural core of 4,000 images sampled uniformly with seed 42; and
- a rare-category augmentation of at most 1,000 additional images selected
  greedily from the remaining training images.

The augmentation targets at least 50 reference images containing each COCO
category. At each step it chooses the image that fills the largest number of
remaining category deficits, with deterministic image-ID ordering for ties.
The manifest records the natural or augmentation origin of every image and the
achieved per-category counts. If the budget cannot satisfy every quota, the
selector reports the unmet quotas without silently increasing the sample size.

Derive two class-independent bank populations from the same raw cache:

- a natural bank from the uniformly sampled core for density methods; and
- a coverage bank from all reference images for kNN, medoid, and other
  coverage-oriented methods.

Neither bank is partitioned or selected by the predicted class at inference.

## Saved query metadata

Save all 300 query persistence vectors for every selected decoder layer. Attach
the following metadata to each query:

- image ID and query ID;
- decoder layer;
- Hungarian-matched ground-truth object and class IDs, or `-1` for an
  unmatched/background query;
- predicted class, maximum class confidence, and predicted box; and
- matched/unmatched and correct/incorrect status.

Matching and class metadata support reference coverage, oracle diagnostics,
and interpretation only. They never route a test query to a class-specific
reference.

## Locked inference query policies

Use the final decoder layer to compute one selection score per unique query:
the maximum sigmoid classification score across its 80 classes. Use the same
selected query IDs for all decoder layers. Maximum raw logit gives the same
ranking, but fixed thresholds are defined on the sigmoid score.

Evaluate these policies:

1. all 300 queries with equal weight;
2. top 10, 20, and 50 queries by maximum classification score;
3. queries whose maximum classification score is greater than 0.2, 0.3, or
   0.5; and
4. all 300 queries with smooth confidence weights using exponents 1 and 2 and
   a nonzero uniform floor.

Also report ground-truth matched-only, background-only, correct-only, and
incorrect-only oracle results on annotated evaluation images. These oracle
policies are diagnostic and are not deployable inference policies.

Hard thresholds have a variable query count and may suffer survivor bias under
blur: low-confidence anomalous queries can disappear while only clean-looking
queries remain. Therefore, every threshold result reports selected-query count,
empty-selection frequency, adjacent query-set overlap, and monotonicity. An
empty threshold selection is reported as missing rather than silently assigned
a low uncertainty; the all-query and smooth-weight policies always provide the
complete scene score.

Top-K has a fixed count but may change membership. Report adjacent query-set
overlap for K equal to 10, 20, and 50 and compare uncertainty changes at
membership switches.

## Efficient bank comparison

Keep the complete raw feature cache, but derive indexed banks capped at 10,000,
25,000, 50,000, and 100,000 vectors per decoder layer. Start policy comparison
with the 25,000-vector bank, then evaluate bank-size sensitivity for the best
two or three query policies.

Until an approximate-neighbor dependency is deliberately added, perform exact
GPU search in bank chunks and retain only the nearest distances. Never
materialize the full query-by-bank distance tensor. If approximate search is
added later, measure its neighbor recall against exact search on a fixed tuning
subset.

## Additional verification

Tests and reports must cover:

- deterministic natural and rare-category image selection;
- achieved category coverage;
- natural and coverage bank reproducibility;
- unique-query ranking rather than flattened query-class ranking;
- final-layer selection reused across decoder layers;
- fixed top-K membership and overlap metrics;
- fixed-threshold query counts and empty selections;
- smooth confidence weights and their uniform floor; and
- identical comparison-bank lookup when only the predicted class identity
  changes.
