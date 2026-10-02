# Deployment-Oriented Corruption-Sensitivity Evaluation

Date: 2026-08-21
Status: approved design

## Goal

Evaluate whether the existing persistence-distance signal can distinguish a single corrupted
image from a clean image at deployment time, regardless of whether the raw signal rises or
falls as corruption increases.

The experiment remains training-free. It uses the existing 250 tuning images at six Gaussian
blur levels, reuses saved query distances and detector outputs, and does not score the held-out
250-image test partition.

## Decisions already made

- Treat a consistently increasing or decreasing signal as useful corruption evidence.
- Use deployment separation, measured by clean-versus-blurred AUROC, as the primary ranking
  criterion.
- Use median per-image absolute Spearman as the main corruption-sensitivity diagnostic.
- Preserve the signed Spearman and report the proportions of upward, downward, and flat
  image-level trends.
- Add five 20-percent confidence buckets beside the existing ten 10-percent buckets; do not
  replace the existing deciles.
- Plot actual scene distances, not changes relative to each image's clean counterpart.
- Use small multiples with blur severity on the x-axis and one shared y-axis scale.
- Plot a median line and the middle 50 percent of images rather than using colour to encode
  variance.
- Do not train a head or calibrate the score into a corruption probability.

## Main questions

1. Can an actual persistence distance from one image separate clean and blurred images with
   unrelated scene content?
2. Which detector-confidence range gives the best deployment separation?
3. Do wider 20-percent buckets give a more stable result than 10-percent buckets?
4. Does a candidate react strongly to blur even when its raw direction is negative?
5. Is the direction consistent enough to choose once on tuning data and lock for later use?
6. Does persistence outperform detector confidence alone on the same queries?

## Non-goals

- Do not rerun DETR, persistence extraction, bank construction, or nearest-neighbour search.
- Do not alter the clean comparison bank, normalizer, query fingerprints, or kNN definition.
- Do not train an uncertainty or corruption-classification head.
- Do not choose a different sign for each image, severity, or held-out example.
- Do not choose a deployment threshold or report a probability of corruption.
- Do not use the held-out test partition for metric design, candidate selection, orientation,
  plotting decisions, or calibration.
- Do not overwrite or reinterpret the completed confidence-decile experiment artifacts.

## Why absolute Spearman is not enough

Absolute Spearman measures whether a candidate changes monotonically within the same scene as
blur rises. It treats a reliable downward response as strongly as a reliable upward response.
That is desirable for discovering a corruption-sensitive feature.

It does not prove that the raw score can classify a new image. Different clean scenes may have
very different baseline distances. A candidate can therefore have excellent per-image absolute
Spearman while the raw distances of clean and corrupted images overlap heavily. Deployment
separation must be measured directly from the actual scores.

The report must keep these questions separate:

- absolute Spearman: does corruption move the feature within an image?;
- AUROC: does the oriented actual score rank corrupted images above clean images across scenes?

## Existing artifacts to reuse

Reuse the same tuning feature cache and raw-kNN query-distance artifacts validated by the
confidence-decile analysis. They contain, for each image and severity:

- logits for all emitted queries;
- predicted boxes;
- persistence fingerprints for decoder layers 0, 1, and 2;
- saved query-to-clean-bank distances;
- tuning/test partition metadata and artifact provenance.

The new analysis must repeat the existing checks for artifact identity, record keys, image IDs,
severities, partition, layers, and query counts before joining inputs.

## Architecture

Add a new `analyze-corruption-sensitivity` cache-only command for the deployment-oriented
evaluation. The command reuses the existing safe loaders, padding detector, membership logic,
query-score aggregators, and provenance checks. It writes a separate self-contained output
directory.

Keep the existing `analyze-confidence-deciles` command and its output contract unchanged. Shared
code may be generalized behind its current public interfaces, but the old decile results and
field meanings must reproduce exactly.

The new analysis has five bounded responsibilities:

1. construct deterministic decile and quintile memberships;
2. calculate matched persistence and confidence-control scene scores;
3. orient each candidate once from tuning trends;
4. calculate deployment separation and corruption-sensitivity metrics;
5. produce machine-readable results, shared-axis plots, and a plain-language report.

## Query validity and confidence

Use the existing stable padding-union rule. For each image, remove the union of padded query IDs
found across all six severities before building primary bucket memberships. This keeps the valid
query set fixed across severity.

For each valid query, confidence is the largest sigmoid classification score. Sort from lowest
to highest confidence and use ascending query ID as the deterministic tie-breaker.

The primary analysis is padding-filtered. Repeat the lowest decile and lowest quintile without
padding removal only as sensitivity diagnostics. Unfiltered results are never deployment-ranked.

## Confidence bucket schemes

### Deciles

Preserve the existing ten equal-count confidence bins:

- `decile_00_10` through `decile_90_100`.

### Quintiles

Add five equal-count bins over the same ordered valid queries:

- `quintile_00_20`;
- `quintile_20_40`;
- `quintile_40_60`;
- `quintile_60_80`;
- `quintile_80_100`.

Bin sizes may differ by at most one query when the query count is not divisible by the number of
bins. Because a normal record must support both schemes, fewer than ten valid queries remains a
hard input error.

Each output row must carry an explicit `bucket_scheme` and `confidence_bin`; names alone must not
be parsed to infer the scheme.

## Dynamic and frozen memberships

Build both schemes under the existing membership rules:

- `dynamic`: rebuild the confidence ordering and bins from the image being scored;
- `frozen`: reuse the severity-zero query IDs across severities.

Only dynamic, padding-filtered candidates are deployable and eligible for the main ranking.
Frozen memberships remain diagnostics for separating query movement from fingerprint movement.
The two modes must never be combined into one score.

## Signals and scene summaries

Calculate both signals on identical selected query IDs:

- persistence: saved mean distance to the five nearest clean-bank fingerprints;
- confidence control: one minus the maximum query class confidence.

Apply the same three query-to-scene summaries to both signals:

- `mean`;
- `q90`;
- `top20_mean`.

Persistence is evaluated at `combined`, `layer_0`, `layer_1`, and `layer_2`. Layer 2 remains the
primary scope. Confidence has no decoder-layer scope.

The deployment plots are deliberately narrower than the metric sweep: they show dynamic,
padding-filtered `q90`, with persistence at layer 2 and its matched confidence control. This
holds the score recipe fixed while the confidence bucket changes.

## Per-image trend measurements

For each image and candidate, use its six severity scores to calculate:

- signed Spearman;
- absolute Spearman;
- oriented adjacent-step consistency;
- strongest-blur-versus-clean success after orientation.

The group-level corruption-sensitivity value is the median of the per-image absolute Spearman
values. It must not be calculated as the absolute value of the median signed Spearman.

A complete curve whose six finite values are all identical contains no trend. Record it
explicitly as `flat`, assign documented signed and absolute sensitivity values of zero, and
include it in denominators. This is a declared metric convention, not silent missing-data
replacement. A curve with missing or non-finite severity scores remains unmeasured and must not
be converted to zero.

For every candidate, report the fraction and count of fully measured images whose signed trend
is:

- positive;
- negative;
- flat.

## Candidate orientation

Choose one orientation for each candidate from tuning data only:

- positive median signed Spearman: use the actual score unchanged;
- negative median signed Spearman: multiply the actual score by negative one;
- zero median signed Spearman: mark the candidate unorientable and exclude it from deployment
  ranking.

For one signal candidate, orientation is chosen once across all six severities. It must not be
optimized separately for an individual severity, image, or metric. Persistence and its matched
confidence control are separate signal candidates: each receives one tuning orientation of its
own, and neither is re-oriented for a particular comparison.

Store the chosen orientation and the original signed metrics. Plots show the original actual
distance direction rather than the oriented score, so a useful decreasing signal remains visibly
decreasing.

## Deployment separation

For each orientable candidate, calculate five clean-versus-corrupted AUROCs:

- severity 0 versus severity 1;
- severity 0 versus severity 2;
- severity 0 versus severity 3;
- severity 0 versus severity 4;
- severity 0 versus severity 5.

Each comparison contains the same number of clean and corrupted images when coverage is complete.
Use the one locked candidate orientation for all five comparisons. Never replace a severity's
AUROC with its direction-reversed value.

The primary deployment metric is macro AUROC: the ordinary mean of the five per-severity AUROCs.
Publish all five components so strong separation at severe blur cannot hide failure at mild blur.

Interpretation in the easy report:

- 0.5 is chance-level ranking;
- 1.0 is perfect ranking;
- a value below 0.5 after locked orientation means that severity behaves against the selected
  direction.

AUROC measures ranking, not probability calibration.

## Supporting metrics

Retain and report:

- median signed Spearman;
- median absolute Spearman;
- positive, negative, and flat trend counts and fractions;
- oriented adjacent-step consistency;
- oriented strongest-blur-versus-clean rate;
- score coverage by image and severity;
- selected-query counts;
- dynamic overlap with severity-zero membership;
- padded-query counts and lowest-bin padding sensitivity;
- matched persistence-versus-confidence comparisons.

Mean and variance of the actual scene scores by severity must be saved for inspection. The plots
use median and interquartile range because they are less sensitive to a few unusually large
distances.

## Deployment ranking

A candidate enters the deployment ranking only when it is:

- dynamic;
- padding-filtered;
- fully covered for all 250 images and six severities;
- persistence at layer 2;
- orientable from its median signed tuning trend.

Rank eligible candidates by:

1. higher macro AUROC;
2. higher median per-image absolute Spearman;
3. higher dominant-direction fraction, counting flat images in the denominator;
4. higher oriented adjacent-step consistency;
5. deterministic bucket-scheme, confidence-bin, and summary names.

This is a tuning ranking, not a hypothesis test. The report must say that every candidate and its
orientation were selected on the same 250 tuning images used to describe them.

Dominant-direction fraction is the larger of the positive-trend count and negative-trend count,
divided by all fully measured images, including flat images in that denominator.

Keep the completed experiment's original signed-Spearman ranking in its existing artifacts. The
new ranking applies only to the new deployment-oriented output.

## Actual-distance figures

Produce four small-multiple figures:

1. persistence deciles;
2. persistence quintiles;
3. confidence-control deciles;
4. confidence-control quintiles.

Within every panel:

- x-axis: Gaussian-blur severity 0 through 5;
- y-axis: actual un-oriented scene score;
- solid line: median across images;
- shaded band: 25th through 75th percentile across images.

All persistence panels in both persistence figures must use exactly the same y-axis limits.
All confidence panels in both confidence figures must use exactly the same y-axis limits.
Persistence and confidence use separate scales because their units are unrelated.

Derive a signal's shared limits jointly from the minimum plotted 25th percentile and maximum
plotted 75th percentile across both bucket schemes. Add a margin equal to 5 percent of that span
on each side. If the span is zero, use a margin equal to the larger of 5 percent of the absolute
value and `1e-6`. Do not autoscale individual panels. Record the final limits in `summary.json`
so the figures are reproducible and testable.

Use colour to distinguish the signal and its band, not to encode variance. Titles must state the
bucket boundaries and a tuning trend label. Label a positive median signed Spearman `increasing`,
a negative median `decreasing`, an all-flat group `flat`, and any other zero-median group `mixed`.
Figures use q90, dynamic membership, padding filtering, and layer 2 for persistence.

## Outputs

Write a new analysis directory without modifying the completed decile output. It contains:

- `per_scene.csv`: one row per image, severity, signal, bucket scheme, confidence bin,
  membership mode, padding mode, scene summary, and score scope;
- `candidate_metrics.csv`: one row per candidate with orientation, five severity AUROCs, macro
  AUROC, signed and absolute trend metrics, direction counts, supporting metrics, and coverage;
- `summary.json`: the same results plus provenance, configuration, shared axis limits, validation
  counts, and ranking;
- `persistence_actual_distance_deciles.png`;
- `persistence_actual_distance_quintiles.png`;
- `confidence_actual_distance_deciles.png`;
- `confidence_actual_distance_quintiles.png`;
- `easy-report.md`: a plain-language deployment-oriented explanation.

The easy report must lead with:

1. whether the best actual score separates clean from corrupted images;
2. its five per-severity AUROCs and macro AUROC;
3. its absolute trend strength and dominant direction;
4. whether a quintile is more stable or deployable than the best decile;
5. whether persistence beats its matched confidence-only control;
6. the statement that the result is not a calibrated probability and the held-out images were
   not used.

## Data flow and resource use

Reuse the existing streaming cache loader. Retain only the confidence vector, padding metadata,
record identity, and fields required to validate the cached fingerprints. Join these with the
saved per-query distance artifact by image, severity, and partition.

Calculate the sorted valid query order once per record. Slice both decile and quintile memberships
from that one order, then reuse memberships across signals, layers, and summaries. No GPU work,
model inference, bank loading, or nearest-neighbour search is required.

## Validation and error handling

Stop with a clear error when:

- input provenance or artifact keys disagree;
- a key is missing or duplicated;
- an image lacks severity zero or any expected severity;
- a record is outside the tuning partition;
- layer IDs or query counts disagree;
- fewer than ten valid queries remain;
- a bucket scheme, summary, membership mode, padding mode, or scope is unknown;
- result keys would collide;
- an AUROC comparison lacks both clean and corrupted examples;
- output would overwrite a completed analysis directory.

Missing data and non-finite scores remain explicit. Record metric-specific numerators and
denominators. Do not silently drop failed images while displaying the full dataset size.

## Testing strategy

Implementation follows test-driven development. Tests must cover:

- deterministic quintile boundaries and stable confidence tie-breaking;
- quintile sizes for divisible and non-divisible query counts;
- the existing decile command interface and calculations remain unchanged, and shared new-command
  decile score rows match the old command on every common field;
- both schemes reuse the same padding-filtered valid query order;
- dynamic and frozen memberships retain their existing meanings;
- persistence and confidence use identical selected query IDs;
- the lowest-quintile padding diagnostic is excluded from deployment ranking;
- median of per-image absolute Spearman is not absolute median signed Spearman;
- a complete constant curve is explicitly counted as flat with zero sensitivity;
- a missing curve remains unmeasured rather than becoming zero;
- positive and negative synthetic trends receive equal absolute sensitivity;
- orientation is selected once and reused across all severity AUROCs;
- a negative raw trend becomes a useful oriented deployment score;
- unorientable zero-median candidates cannot enter the deployment ranking;
- per-severity AUROCs and macro AUROC match hand-checkable synthetic examples;
- ranking gates full coverage before comparing metrics;
- all panels of one signal share identical y-axis limits;
- persistence and confidence are never placed on a shared numerical scale;
- existing decile analysis tests and the full scene-uncertainty suite continue to pass.

An integration test creates a small six-severity cache and saved-distance artifact, runs the new
command, and verifies both bucket schemes, locked orientations, CSV/JSON consistency, rankings,
all four figures, and the plain-language report.

## Experiment procedure

1. Implement and test the cache-only analysis.
2. Run it on the existing 250-image tuning partition and six blur severities.
3. Confirm the existing decile signed metrics reproduce before interpreting new metrics.
4. Compare actual-distance AUROC and absolute Spearman across deciles and quintiles.
5. Select at most one candidate and lock its bucket scheme, bin, summary, padding rule, layer,
   and orientation.
6. Review the tuning conclusion before authorizing any held-out test run or probability
   calibration.

## Success criteria

A promising candidate must do more than react to paired blur. It must have complete coverage,
consistent orientation, useful per-severity clean-versus-corrupted AUROC from actual distances,
and a strong median absolute Spearman. It should also outperform the matched confidence-only
control.

A strong absolute Spearman with chance-level AUROC is not deployable: it means the feature moves
within a scene but unrelated scene baselines hide that movement. Conversely, strong AUROC with a
weak blur trend may separate this synthetic dataset for the wrong reason and must not be selected
without further diagnosis.
