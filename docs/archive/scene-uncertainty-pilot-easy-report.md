# Scene-Uncertainty Pilot and Confidence-Decile Follow-up

Date: 2026-08-21

## The short answer

The implementation is faithful to the approved experiment and well tested. All 774 current scene-uncertainty tests pass. I also recalculated the main numbers directly from the saved 523,500-row table rather than trusting the generated summary.

The work now has two phases:

- **Phase 1** found a useful blur signal by keeping all 300 queries and taking the 90th percentile of their last-layer persistence distances. Its median Spearman score was **+0.6000**.
- **Phase 2** sorted each image's valid queries into ten confidence buckets. The strongest bucket was the middle, from the 50th to the 60th confidence percentile. Its median Spearman was **+0.6286**.
- On exactly the same middle-confidence queries, uncertainty from detector confidence alone had Spearman **-0.5429**. Persistence therefore contains information that the raw confidence control does not provide.
- The new result is only **+0.0286** above the old all-query result. That is the smallest difference this median statistic can show, and the candidate was selected on the same tuning images used to report it.
- The highest-confidence queries still behave badly. Their persistence distance usually falls as blur increases.
- The current score is an anomaly score, not a probability that an image is corrupted.
- Only the 250 tuning images were scored. The 250 held-out test images remain unused.

The current tuning candidate is:

```text
last decoder layer
+ remove repeated padded queries
+ sort the remaining queries by detector confidence
+ keep the 50th-to-60th-percentile bucket
+ rebuild that bucket independently for each image
+ distance to the five nearest clean query fingerprints
+ average the largest 20% of the selected distances
```

This candidate is promising, not final. Its advantage over the old result is small, the clean comparison bank still contains padded slots, and the held-out test set has not confirmed it.

## What was tested

The clean reference library used 5,000 COCO training images. Each image produced 300 query fingerprints at each of three decoder layers.

The blur study cached 500 COCO validation images at six blur levels:

```text
blur level:  0   1   2   3   4   5
blur radius: 0   1   2   4   8  12
```

The 500 images were split into:

- 250 tuning images, used for this report;
- 250 held-out test images, saved but not scored yet.

The clean comparison bank contained 25,000 query fingerprints per decoder layer. It was built from clean images only. It was not split by predicted class, so a prediction changing from car to truck does not change the comparison ruler.

Phase 2 reused the cached logits and the previously saved per-query distances. It did not rerun the detector, rebuild the bank, recompute nearest neighbors, or train a new head. It produced 523,500 score rows: 250 images times six blur levels across 349 signal configurations.

The complete Phase 2 tables and provenance remain in [the detailed confidence-decile report](scene-uncertainty-confidence-decile-results.md). This cumulative report keeps the main findings and points to that document for every row.

## What is the signal?

RT-DETR makes 300 possible object guesses, called queries. A query can describe an object, part of an object, background, or sometimes a dead/padded slot.

For each query, the code looks inside the detector's classification layer. It builds a graph from:

- the query's activation values; and
- the classification layer's weights.

It keeps the strongest connections needed to join the graph together. The 335 sorted connection strengths form the query's persistence vector. It is helpful to think of this vector as a fingerprint of what the classification head was doing for that query.

The main uncertainty signal is then:

> How far is this query fingerprint from similar fingerprints made by clean images?

A large distance means “this query looks unusual compared with clean data.” A small distance means “this query looks familiar.”

## How one scene score was calculated

The calculation has six simple steps.

1. Make one 335-number persistence fingerprint for every query.
2. Find the five closest fingerprints in the clean bank.
3. Average those five distances. This is the query's unusualness score.
4. Decide which queries to keep, or how much weight to give each one.
5. Combine the query scores into one scene score.
6. Do this separately for decoder layers 0, 1, and 2. The optional combined score first puts the three layer scores onto clean-reference scales and then averages them.

No uncertainty head was trained. No predicted class was used to select a class-specific bank.

## Phase 1: Original query-policy pilot


### Query choices

Nine choices can be used at normal inference time:

- `all`: keep all 300 queries;
- `top10`, `top20`, `top50`: keep the queries with the highest detector confidence;
- `threshold_0.2`, `threshold_0.3`, `threshold_0.5`: keep queries above a confidence cutoff;
- `smooth_1`, `smooth_2`: keep every query but give more confident queries more weight.

Four extra choices use ground truth and are only diagnostics:

- matched queries;
- background or unmatched queries;
- correctly classified matched queries;
- incorrectly classified matched queries.

These four cannot be used on a new unlabeled image. They are included to help explain where the signal comes from.

### Ways to combine query scores

- `mean`: average every selected score;
- `median`: use the middle score;
- `q90`: use the score near the high end, where 90 percent of query scores are below it;
- `top20_mean`: average the highest-scoring 20 percent of selected queries;
- `weighted_mean`: used for the two smooth-confidence choices.

### Decoder-layer choices

- layer 0: early decoder output;
- layer 1: middle decoder output;
- layer 2: last decoder output;
- combined: clean-scale the three layer scores and average them equally.

Together, the run reported 184 combinations. That is 46 policy-and-summary recipes viewed at four score scopes.

### The 10 most effective usable signals

The table below excludes ground-truth-only oracle policies. It also excludes recipes that failed to produce a score at every image and blur level. The ranking uses median per-image Spearman correlation: larger positive values are better for the desired rule, “more blur should mean more uncertainty.”

“Up steps” is the fraction of neighboring blur changes where the score did not fall. “Blurred ended higher” is the fraction of images whose strongest blur scored above their clean version.

| Rank | Recipe | Median Spearman | Up steps | Blurred ended higher | Plain-language reading |
|---:|---|---:|---:|---:|---|
| 1 | All queries, q90, layer 2 | 0.600 | 61.0% | 84.8% | Best usable signal. Look at the high end of all last-layer query distances. |
| 2 | All queries, top 20% mean, layer 2 | 0.486 | 58.4% | 84.0% | Average the most unusual fifth of last-layer queries. |
| 3 | All queries, mean, layer 2 | 0.486 | 57.9% | 78.8% | The whole last layer carries some scene-wide signal. |
| 4 | All queries, median, layer 2 | 0.429 | 57.5% | 71.6% | The middle last-layer query rises, but less reliably. |
| 5 | All queries, q90, three-layer combined | 0.429 | 51.1% | 70.8% | Positive overall, but layers 0 and 1 weaken layer 2. |
| 6 | All queries, top 20% mean, combined | 0.371 | 49.3% | 72.4% | Endpoint is useful, but step-by-step behavior is almost random. |
| 7 | Smooth confidence weight, layer 2 | 0.086 | 48.6% | 61.6% | Very weak. Confidence weighting hides much of the useful tail. |
| 8 | All queries, mean, combined | 0.086 | 47.8% | 60.0% | Almost flat after combining the disagreeing layers. |
| 9 | Stronger smooth confidence weight, layer 2 | -0.029 | 45.4% | 51.2% | Essentially no useful upward trend. |
| 10 | All queries, median, combined | -0.143 | 46.2% | 47.6% | Weak and in the wrong direction. Included because only eight full-coverage deployable recipes were nonnegative. |

Within Phase 1, the first four rows told a consistent story: keep all queries, use the last layer, and focus on the upper part of the distance distribution. Phase 2 later refined the query-selection part of that conclusion.

### A useful diagnostic result

The ground-truth-only background signal was slightly stronger than most deployable signals:

- background q90 at layer 2: Spearman 0.600;
- background top-20-percent mean at layer 2: Spearman 0.600;
- background mean at layer 2: Spearman 0.543.

This cannot be deployed because deciding which queries are truly background requires labels. However, it explains that much of the positive blur signal is in background-like queries. The deployable all-query q90 result reaches the same 0.600 without needing labels.

### Independent check of the strongest result

I recalculated the best usable result directly from the saved per-scene CSV instead of trusting the summary table.

For all queries, q90, layer 2, the median score at each blur level was:

```text
level 0: 0.0985
level 1: 0.0978
level 2: 0.0967
level 3: 0.0974
level 4: 0.1099
level 5: 0.1254
```

The recalculated results exactly matched the summary:

- median Spearman: 0.600;
- adjacent non-decrease rate: 61.0%;
- strongest blur above clean: 84.8%;
- complete coverage: all 1,500 image-and-severity records had scores.

This is a real but imperfect signal. It gets slightly smaller for the first two blur steps, then rises clearly at stronger blur.

## Phase 2: Confidence-decile follow-up

### Why divide queries by confidence?

Phase 1 showed that taking only the highest-confidence queries was a poor strategy. But that did not tell us whether low-confidence or middle-confidence queries carried useful information.

For Phase 2, the software did this separately for each image and blur level:

1. Detect repeated decoder slots at the end of the query list.
2. Remove the union of those padded query IDs across all six blur levels.
3. Calculate each remaining query's confidence as its largest sigmoid class score.
4. Sort the valid queries from lowest to highest confidence.
5. Divide them into ten equal-count buckets, called deciles.
6. Calculate persistence uncertainty and confidence uncertainty on exactly the same query IDs.

A dynamic bucket is rebuilt from the image being scored, so it can be used at normal inference time. A frozen bucket reuses IDs selected from the clean image. Frozen results are useful diagnostics, but they cannot be used when a naturally corrupted image has no clean partner.

The confidence control is simply one minus the detector's confidence. This only reverses the direction; it does not turn the value into a probability of corruption.

### Main comparison

| Measurement | Median Spearman | Up steps | Maximum blur above clean | Coverage |
|---|---:|---:|---:|---:|
| Middle-confidence candidate: 50%-60%, dynamic, top-20% mean, layer 2 | **+0.6286** | 61.8% | 88.0% | 1,500/1,500 |
| Confidence-only control on the same queries | **-0.5429** | 46.5% | 40.8% | 1,500/1,500 |
| Original all-300 benchmark: q90, layer 2 | **+0.6000** | 61.1% | 84.8% | 1,500/1,500 |
| All valid queries after padding removal: q90, layer 2 | **+0.5429** | 59.6% | 82.0% | 1,500/1,500 |

Against confidence alone, the persistence candidate won on 176 images, tied on 9, and lost on 65. Against the original all-300 benchmark, it won on 127, tied on 39, and lost on 84. That is encouraging, but its median advantage over the benchmark is only +0.0286: one step on the coarse Spearman grid created by six blur levels.

### Confidence-range map

The table below holds everything fixed at dynamic membership, q90, layer 2, and padded-query filtering. Only the confidence range changes.

| Confidence percentile | Median Spearman | Plain-language reading |
|---|---:|---|
| 0%-10% | -0.0286 | The lowest-confidence bucket is nearly flat. |
| 10%-20% | +0.4286 | A useful positive trend begins. |
| 20%-30% | +0.5429 | Stronger positive trend. |
| 30%-40% | +0.6000 | Matches the old benchmark. |
| 40%-50% | +0.6000 | Matches the old benchmark. |
| 50%-60% | +0.6000 | Strong middle-confidence range. |
| 60%-70% | +0.5429 | Still useful, but weaker. |
| 70%-80% | +0.4857 | Moderate positive trend. |
| 80%-90% | +0.0286 | Almost flat. |
| 90%-100% | -0.7714 | Strongly moves in the wrong direction. |

This is the main new result: the blur information has a hill-shaped relationship with confidence. It is strongest in the middle and poor at both extremes, especially among the most confident queries.

### The 10 strongest Phase 2 candidates

Every row below uses dynamic membership, removes padded queries, scores persistence at layer 2, and covers all 1,500 image-severity pairs.

| Rank | Confidence percentile | Scene summary | Median Spearman | Up steps | Maximum blur above clean |
|---:|---|---|---:|---:|---:|
| 1 | 50%-60% | top-20% mean | **+0.6286** | 61.8% | 88.0% |
| 2 | 50%-60% | mean | **+0.6286** | 61.5% | 84.4% |
| 3 | 50%-60% | q90 | +0.6000 | 61.4% | 88.4% |
| 4 | 40%-50% | mean | +0.6000 | 60.8% | 84.0% |
| 5 | 30%-40% | top-20% mean | +0.6000 | 60.7% | 86.4% |
| 6 | 30%-40% | mean | +0.6000 | 60.6% | 82.8% |
| 7 | 40%-50% | q90 | +0.6000 | 60.5% | 85.6% |
| 8 | 30%-40% | q90 | +0.6000 | 59.4% | 85.2% |
| 9 | 40%-50% | top-20% mean | +0.6000 | 59.3% | 86.4% |
| 10 | 60%-70% | top-20% mean | +0.5714 | 58.4% | 81.6% |

These rows are not ten independent experiments. Several are different summaries of the same selected queries, all chosen and evaluated on the same tuning images.

### What query movement tells us

The winning dynamic bucket kept only 0.0587 of its clean membership at blur levels 1 through 5. Two unrelated ten-percent buckets would overlap by about 0.0526. In other words, the bucket is almost completely rebuilt as blur changes.

When the winning bucket's clean query IDs were frozen, its median Spearman was +0.6000 rather than +0.6286. Image by image, dynamic membership won 127, tied 30, and lost 93 against its frozen twin.

This means the result belongs to a middle-confidence range, not to a stable set of particular query IDs. Query movement is not shown to be the cause of the positive trend, but it is large enough that dynamic and frozen results must remain separate.

### What padding changed

Repeated padded tails appeared in 66 of the 250 tuning images. Across images, the stable union masks contained 6,586 padded slots, and one image lost 257 of its 300 emitted queries. None of the 66 padded images had the same detected tail at every blur level.

For the old all-query q90 result, removing padded queries changed the median Spearman from +0.6000 to +0.5429. For the dynamic lowest-confidence q90 bucket, keeping padding changed it from -0.0286 to +0.1143.

That does not make padded slots useful evidence. It shows that placeholders carry a blur-related signal of their own and can artificially improve a score. The Phase 2 ranking therefore removes them. However, the padding sensitivity run covered only the lowest bucket and the all-query control; it did not directly repeat the winning middle bucket without filtering. The clean comparison bank also still contains padded fingerprints.

### Decoder layers still disagree

For the winning 50%-60% selection with top-20% mean:

- combined three-layer score: +0.4286;
- layer 0: -0.3143;
- layer 1: -0.2857;
- layer 2: +0.6286.

The useful result remains specific to the final decoder layer. It should not be described as evidence that every persistence feature rises under blur.

## Lessons shared by both phases

### Confident-query selection mostly points backward

In Phase 1, top-10 and top-20 with a mean scene score both had median Spearman -0.886. Phase 2 confirmed the same pattern from another angle: the highest-confidence ten-percent bucket had Spearman -0.8286 with a mean summary. These are strong relationships with the wrong sign: stronger blur produced a smaller clean-bank distance.

The report tested whether this was caused only by different query IDs entering the top 20. It froze each image's clean top-20 query IDs and reused those same IDs at every blur level. The trend was still strongly negative, with Spearman -0.771.

That means most of the problem is not the query list changing. The query fingerprints themselves shrink under blur and move toward a dense, background-heavy part of the clean bank. Query switching explains only about 13 to 29 percent of the drop.

So the raw Euclidean distance is acting like this:

```text
more blur
  -> smaller activation/persistence magnitude
  -> closer to common low-magnitude background fingerprints
  -> smaller distance
```

This is why simply choosing the most confident detections is not recommended.

### Confidence thresholds lose the hardest images

At the 0.5 threshold, the median number of selected queries fell from five at clean severity to zero at maximum blur. Only 89 of 250 images still had any score at maximum blur.

That creates survivor bias: the report sees only the images and queries that survived. The code correctly records these missing scores instead of pretending they are low uncertainty.

### The decoder layers disagree

For all-query q90:

- layer 0 Spearman: -0.429;
- layer 1 Spearman: -0.600;
- layer 2 Spearman: +0.600;
- equal three-layer combination: +0.429.

The last layer sees the useful trend. Averaging it equally with the earlier layers weakens it.

### Some bank entries are padded detector slots

About 1.6 percent of bank rows were exact duplicates. Investigation showed that these were not caused by float16 storage. They were contiguous dead or padded query slots at the end of some images' 300-query output.

The current bank treats unmatched slots as background, so those padded slots enter the background half of the bank. This is methodologically questionable even though the implementation matches its written rules. A future bank should compare “all emitted queries” against “live, valid query slots only.”

## Metrics used in the report

### Median score by blur level

For each blur level, take the middle scene score across the 250 tuning images. The report also stores the lower and upper quarters to show how much images differ.

### Clean-relative score

Subtract each image's clean score from its blurred score. Every image then starts at zero. This makes curves easier to compare but does not force them to rise.

### Spearman correlation

This asks whether larger blur levels usually get larger uncertainty scores for the same image.

- +1 means perfectly rising;
- 0 means no dependable order;
- -1 means perfectly falling.

The report computes this separately for every image, then takes the median across images.

### Adjacent monotonicity

There are five steps between the six blur levels. This metric counts how often the next step stays level or rises. A perfect score is 100 percent. About 50 percent is coin-flip behavior.

### Violation magnitude

When a score falls, this measures how large the fall was compared with that image's total score range. Smaller is better.

### Endpoint increase rate

The fraction of images where maximum blur scores above the clean image. This can look good even when the middle blur levels wobble, so it should not be used alone.

### Coverage and empty selections

These count how many images and blur levels actually received a score. They are essential for threshold and oracle policies. A high trend score is not trustworthy if most difficult cases disappeared.

### Query-set overlap

Phase 1 checks how similar selected query IDs are between neighboring blur levels. Phase 2 compares each dynamic confidence bucket with that image's severity-zero bucket. A value of one means the same set; zero means no overlap. Two unrelated ten-percent buckets overlap by about 0.0526, which gives the Phase 2 values a useful reference.

### Matched persistence-versus-confidence comparison

Persistence and confidence are scored over the same query IDs with the same scene summary. Their raw values are not compared because a distance and a confidence score have different units. Instead, the report compares their per-image Spearman trends and counts how many images persistence wins, ties, or loses.

### Dynamic-versus-frozen comparison

Dynamic buckets can be built from one incoming image. Frozen buckets require a paired clean image and are diagnostics only. Comparing them helps separate movement of query fingerprints from movement of query IDs between confidence buckets. They are never averaged into one score.

### Class-switch diagnostics

The report separates steps where a still-matched object changed predicted class from steps where no such class change occurred. This does not detect objects that vanished completely, so “no switch” does not mean the detector was stable.

### Bank-size stability

The 25,000- and 50,000-vector banks gave almost the same policy ranking: Spearman 0.950 and Kendall 0.894. Doubling the bank is therefore unlikely to fix the direction problem.

### Near-duplicate diagnostics

The clean-distance fit reports the nearest non-self distance and the fraction of sampled bank rows with a near duplicate. This exposed the dead/padded query tails.

## Implementation audit

### What is solid

- All 774 current scene-uncertainty tests pass.
- The comparison bank is class-independent.
- All 300 query persistence vectors are cached once for three layers.
- k-nearest-neighbor search is exact and chunked, so it avoids a giant distance matrix.
- Query distances are calculated once and reused across query policies and scene summaries.
- Empty selections remain missing instead of becoming fake zero uncertainty.
- Train reference images and validation evaluation images are checked for overlap.
- Artifacts record checkpoint, configuration, layers, dimensions, image IDs, and content identifiers.
- The report rejects duplicate result keys that would otherwise inflate statistics.
- There is no hidden sign inversion: larger distance really means farther from the clean bank.
- The confidence-decile loader safely reads existing artifacts and has no inference or nearest-neighbor path.
- It verifies cache and result provenance, partition, layers, query counts, record keys, and all six severities before producing scores.
- Persistence and confidence controls share the exact selected query IDs and scene summaries.
- The committed detailed report is byte-identical to the generated remote report, and its headline statistics were independently reproduced from the saved CSV.

### Small software caveats

- The report contents are fully staged before publication, but the seven final renames are sequential. A process killed after `summary.json` is renamed can leave an incomplete directory that the CLI subsequently treats as finished. This did not happen in the completed run.
- The original implementation plan asked plot helpers to reject a missing confidence bucket. The final implementation instead shows that cell as `n/a`. The real run contained every expected group, so this did not affect its plots or statistics.

### What limits the scientific conclusion

- Only raw feature normalization was tested.
- The 50%-60% candidate was selected and measured on the same 250 tuning images, so its advantage is not held-out evidence.
- The held-out 250-image test partition remains unused.
- Padded evaluation queries were removed from Phase 2, but padded query slots remain in the clean bank.
- Padding sensitivity was measured directly for the lowest bucket and the all-query benchmark, not for the winning middle bucket.
- The combined score gives equal weight to decoder layers that show opposite trends.
- Oracle results are explanatory tools, not deployable methods.
- Class-switch analysis does not measure lost detections.
- The raw score has not been calibrated into a corruption probability.

## Recommended next experiment

Do not spend the held-out test set quite yet.

The confidence-decile tuning run proposes three locked choices:

```text
+ confidence bucket: 50%-60%
+ membership: dynamic
+ scene summary: top-20% mean
```

Before the held-out run, resolve the remaining methodological issue rather than searching more confidence buckets:

1. Build a clean comparison bank that excludes exact repeated padded tails.
2. Keep padded evaluation queries filtered, because they are placeholders rather than predictions.
3. On tuning data only, compare the proposed middle-confidence candidate and the old all-query benchmark under `raw`, `robust_z`, `unit`, and `shape_scale` normalization.
4. Choose one bank rule and one normalization, then freeze the complete recipe.
5. Run that recipe once on the 250 held-out images without changing it afterward.

If the held-out trend remains useful, calibration into a probability of corruption can be studied afterward. Calibration should not begin from the tuning scores alone.

## Final verdict

The implementation is reliable enough to trust both phases of the experiment.

Phase 1 established that last-layer persistence contains a useful upper-tail blur signal, while the highest-confidence selections move strongly in the wrong direction. Phase 2 refined that conclusion: the strongest deployable signal is concentrated in middle-confidence queries, especially the 50%-60% bucket, rather than in all queries equally.

The new candidate reaches +0.6286 instead of the old +0.6000, beats its matched confidence control on 176 of 250 images, and has full coverage. But the improvement over the old benchmark is one coarse metric step, it was selected on tuning data, and its clean bank still includes padded fingerprints.

So the right conclusion is not "the problem is solved." It is:

> Middle-confidence, last-layer persistence is the best current lead. Clean up the comparison bank, lock the recipe, and ask the untouched test set whether the improvement is real.

Neither the persistence score nor `1 - confidence` is currently a probability that the image is corrupted.
