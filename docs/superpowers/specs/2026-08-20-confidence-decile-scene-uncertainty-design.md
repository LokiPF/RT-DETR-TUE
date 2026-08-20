# Confidence-Decile Scene-Uncertainty Experiment

Date: 2026-08-20
Status: approved design

## Goal

Determine which detector-confidence ranges carry the most useful persistence-distance signal for image blur, and compare that signal fairly with uncertainty obtained directly from the detector's confidence.

The experiment remains training-free. It uses the existing 250 tuning images at six Gaussian-blur levels and does not score the held-out 250-image test partition.

## Main questions

1. Do low- or medium-confidence queries carry a stronger blur signal than high-confidence queries?
2. Does persistence distance add information beyond the detector's confidence alone?
3. How much of the result comes from queries changing confidence rank as blur increases?
4. Do invalid padded tail queries create a false low-confidence signal?

## Non-goals

- Do not train an uncertainty head or any other learned component.
- Do not calibrate the score into a probability of corruption.
- Do not select a final method on the held-out test partition.
- Do not rerun detector or persistence-feature extraction for the first experiment.
- Do not change the clean comparison bank or feature normalization in this experiment.

## Existing artifacts to reuse

The blur feature cache already contains, for each image and severity:

- logits for all 300 queries;
- predicted boxes;
- persistence fingerprints for decoder layers 0, 1, and 2;
- the tuning/test partition label.

The existing raw-kNN result artifact also stores every query's distance to its five nearest clean-bank fingerprints at all three decoder layers. The new analysis will consume those saved distances rather than recomputing kNN.

The analysis must verify that cache IDs, image IDs, severities, partitions, layers, and query counts agree before joining artifacts.

## Architecture

Implement this as a separate confidence-decile analysis command and supporting modules, instead of forcing cross-severity state into the current single-record query-policy function.

The command accepts the existing evaluation cache and raw-kNN result artifact, then writes a new self-contained analysis directory. It performs no model inference and no kNN search.

The implementation has four small responsibilities:

1. detect invalid padded query tails;
2. construct dynamic and clean-frozen confidence bins;
3. calculate matched persistence and confidence-only scene scores;
4. summarize trends and produce tables and plots.

## Padded-tail detection

Some cached images contain a consecutive run of invalid query slots ending at query 299. They are decoder placeholders, not object or background hypotheses.

For each cached record, detect the longest suffix whose entries are exactly identical to the final query in all of these fields:

- predicted box;
- complete logit vector;
- persistence fingerprint at every cached decoder layer.

Only a repeated suffix containing at least two queries is treated as padding. Identical entries elsewhere in the query sequence are not removed.

For a given image, take the union of padded query IDs detected across its six severities. This creates one stable valid-query mask for all severities and prevents the validity rule itself from changing dynamic or frozen membership.

Record the padded count by image and severity, the image-level union count, and whether the detected tail was identical at all severities.

The primary analysis excludes the union mask. A sensitivity control repeats only the lowest-confidence-bin analysis without filtering, so the report can show how much padding changes the answer.

## Confidence definition and bins

For each query, confidence is the largest sigmoid class score from its classification logits. Ties are resolved by ascending query ID for deterministic behavior.

Sort valid queries from lowest to highest confidence and split them into ten equal-count bins:

- `decile_00_10`: lowest-confidence 10 percent;
- `decile_10_20` through `decile_80_90`;
- `decile_90_100`: highest-confidence 10 percent.

Bin sizes may differ by at most one query when the valid-query count is not divisible by ten. Fewer than ten valid queries is an invalid record and must produce a clear error rather than empty or misleading bins.

Also define `all_valid`, containing every non-padded query, as a direct comparison with the existing all-query method.

## Dynamic and clean-frozen membership

### Dynamic bins

Re-sort queries and rebuild all ten bins independently at every blur severity. This represents a deployable policy on a single image, but membership may change as confidence changes.

### Clean-frozen bins

Build the ten bins from severity zero and reuse those query IDs at severities one through five. Confidence and persistence values still change; only membership is frozen.

This is a diagnostic rather than a deployable policy because a naturally corrupted image has no paired clean version. It separates movement of the query fingerprints from movement of queries between confidence bins.

Report Jaccard overlap between each dynamic bin and its severity-zero membership at every severity.

## Two matched uncertainty signals

For every selected query, calculate two signals.

### Persistence uncertainty

Use its saved mean distance to the five nearest clean-bank persistence fingerprints. Larger means farther from clean.

### Confidence uncertainty control

Use one minus the query's maximum class confidence. Larger means less confident.

The confidence transformation only reverses direction; it does not train or calibrate anything. A value such as 0.8 must not be described as an 80-percent probability of corruption.

## Scene summaries

Apply the same three summaries to both per-query signals:

- `mean`;
- `q90`;
- `top20_mean`, the mean of the largest 20 percent of selected values.

Using identical memberships and summaries makes the persistence-versus-confidence comparison fair.

Persistence scores are available separately for decoder layers 0, 1, and 2. Layer 2 is the primary scope because it was strongest in the pilot. Layers 0 and 1 and the existing clean-scaled equal-layer combination are secondary diagnostics.

Confidence uncertainty has no decoder-layer scope. In comparison tables, the same confidence result is paired with each persistence scope but stored only once in the underlying result table.

## Benchmarks

The report must include:

1. the existing all-300-query persistence benchmark: q90, layer 2, median Spearman 0.600;
2. the new `all_valid` persistence result, showing the effect of removing padding;
3. `all_valid` confidence-only uncertainty with the same scene summaries;
4. the matched confidence-only result for every confidence bin and membership mode;
5. filtered versus unfiltered results for the lowest-confidence bin.

The saved artifacts, not a hard-coded constant, are the source for recomputing the existing benchmark. The value 0.600 is a regression expectation for the current pilot artifacts.

## Metrics

Compute metrics per signal, confidence bin, membership mode, scene summary, and persistence scope where applicable.

### Primary metric

Median per-image Spearman correlation between blur severity and scene uncertainty. Positive is desired; larger is better.

### Supporting metrics

- adjacent non-decrease rate;
- normalized downward-violation magnitude;
- maximum-blur-above-clean rate;
- scored image/severity coverage;
- selected query count by severity;
- dynamic-bin overlap with the clean bin;
- padded-query counts and padding sensitivity.

For each matched persistence/confidence pair, report the difference in median Spearman. Also report the fraction of images where persistence has a higher per-image Spearman than its confidence control. Scale-dependent raw score magnitudes must not be compared directly.

Full coverage is required for a candidate to be ranked as deployable. Rank primarily by median Spearman, then by adjacent non-decrease rate, then by smaller violation magnitude.

## Outputs

Write an analysis directory containing:

- `per_scene.csv`: one row per image, severity, signal, membership mode, confidence bin, summary, and scope;
- `summary.json`: metrics, provenance, validation counts, and exact configuration;
- `confidence_decile_heatmap.png`: median Spearman by confidence bin and signal;
- `blur_curves.png`: clean-relative severity curves for persistence and confidence;
- `dynamic_vs_frozen.png`: query movement versus feature movement;
- `padding_sensitivity.png`: filtered and unfiltered lowest-bin results;
- `easy-report.md`: a plain-language explanation and ranked results.

The easy report must lead with which confidence range worked best, whether it beat confidence alone, whether it beat the existing all-query benchmark, and whether padding or bin movement explains the result.

## Data flow and memory

Stream the feature cache once. For each record, retain only its artifact key, confidence vector, and detected padding mask; discard persistence tensors after checking the suffix. This avoids holding the multi-gigabyte feature cache in memory.

Load the much smaller saved query-distance artifact, join it to the retained metadata by image ID, severity, and partition, and calculate all selections and summaries from the joined records.

Calculate each selection once and reuse it across persistence layers, persistence summaries, and confidence controls.

## Validation and error handling

Stop with a clear error when:

- artifact provenance does not match;
- a record key is duplicated or missing from either input;
- an image lacks severity zero or any of the six expected severities;
- a record is not in the requested tuning partition;
- layer IDs or query counts disagree;
- fewer than ten valid queries remain;
- a requested summary or score scope is unknown;
- output result keys would be duplicated.

Missing data must never be silently replaced with zero. The first experiment must reject the held-out test partition.

## Testing strategy

Implementation will follow test-driven development. Unit tests must cover:

- repeated identical tail queries are removed;
- identical non-tail queries are retained;
- a single final query is not automatically called padding;
- image-level union masks remain fixed across severities;
- confidence computation and stable tie-breaking;
- ten-bin sizes and boundaries for divisible and non-divisible query counts;
- dynamic membership changes when confidence ranking changes;
- frozen membership remains equal to severity zero;
- confidence uncertainty is exactly one minus confidence;
- persistence and confidence use identical selected IDs and summaries;
- unfiltered bottom-bin control retains padded queries;
- artifact alignment and partition failures are rejected;
- duplicate output keys are rejected.

An integration test will create a tiny six-severity cache and query-distance artifact, run the command, and verify the CSV, summary, benchmark rows, and plots. The existing scene-uncertainty suite must continue to pass.

## Experiment procedure

1. Implement and test the analysis command.
2. Run it on the existing 250-image tuning partition and six blur severities.
3. Verify the current all-query q90 layer-2 benchmark is reproduced from artifacts.
4. Produce the decile comparison and easy report.
5. Select at most one confidence bin, membership rule, scene summary, and padding rule using tuning data.
6. Review the tuning conclusion before any held-out test run.

## Expected interpretation

A useful result is not merely a positive persistence trend. It should have full coverage, remain understandable after padding is removed, and outperform the matched confidence-only control. Beating the existing all-query q90 layer-2 benchmark would be stronger evidence, but a weaker decile can still explain where that all-query signal originates.

Dynamic and frozen results answer different questions and must not be combined into one score. A strong frozen-only result shows useful diagnostic information but does not by itself define a deployable single-image method.
