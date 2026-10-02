# Clean Differential Uncertainty: End-to-End Design

**Date:** 2026-08-23
**Status:** Approved in conversation; implementation has not started
**Target branch:** `clean`

## Purpose

Create a substantially smaller, inference-only codebase that answers one question:

> Can the scale-independent difference between a detector image's middle-confidence queries
> and highest-confidence queries identify image corruption?

The retained method is the later differential idea from the within-image corruption study. It
uses the detector's layer-2 persistence fingerprints, a clean comparison bank, dynamic
confidence deciles, and one symmetric relative-gap score. The cleaned project must run this
method end to end from ordinary image manifests through a detailed result bundle.

The work will be developed in an isolated worktree on a new `clean` branch. The current
`dev_tue` branch and its historical implementation remain intact while the clean branch is
built and verified.

## Scope

### Retained

- Pretrained RT-DETRv2-R18 inference through the current detector implementation.
- The decoder persistence hook and layer-2 persistence fingerprints.
- A class-independent, natural-query clean comparison bank.
- Gaussian blur as the first corruption, behind a small corruption interface.
- Exact padded-query detection and removal.
- Dynamic confidence deciles rebuilt for each image and corruption severity.
- The 50--60% responsive decile and 90--100% reference decile.
- Mean aggregation, five-nearest-neighbour distance, and the symmetric relative gap.
- A matched confidence-only relative-gap baseline.
- Per-severity AUROC, macro AUROC, per-image Spearman, curve-shape checks, paired bootstrap
  comparisons, diagnostics, figures, JSON, CSV, and a plain-language Markdown report.
- Resumable intermediate artifacts with strict provenance checks.

### Removed

- Detector training, calibration, and deployment code.
- Alternative detector families and alternative model configurations.
- COCO annotation loading, Hungarian ground-truth matching, and coverage-bank construction.
- The planned-anchor test and every anchored candidate.
- Raw-gap, raw-responsive, residual, combined-layer, alternative-layer, frozen-membership,
  unfiltered-padding, quantile-summary, and candidate-search paths as selectable methods.
- General experiment ranking, tuning gates, and method-selection machinery.
- Unrelated uncertainty experiments and obsolete command paths.

Raw responsive and reference values remain visible as diagnostic controls. They are not
selectable detector methods.

## Fixed scientific configuration

The clean workflow deliberately has no scientific-method switches.

| Setting | Fixed value |
|---|---|
| Detector | RT-DETRv2 with PResNet-18 backbone |
| Checkpoint use | Frozen pretrained-model inference only |
| Detector queries | 300 before padding removal |
| Persistence scope | Decoder layer 2 only |
| Persistence width | 335 values per query |
| Bank population | Natural valid queries, class independent |
| Bank capacity | 25,000 vectors |
| Bank reservoir seed | 44, matching base seed 42 plus layer 2 |
| kNN | Exact five-nearest-neighbour search |
| Feature normalization | Raw fingerprints; no centering, scaling, or fitted normalization |
| Query distance | Mean of the five neighbour distances |
| Reference range | Dynamic confidence decile 90--100% |
| Responsive range | Dynamic confidence decile 50--60% |
| Within-range aggregation | Arithmetic mean |
| Score | Symmetric relative gap |
| Persistence orientation | `+1`, fixed from the completed blur tuning run |
| Matched-confidence orientation | `-1`, fixed from the completed blur tuning run |
| Initial corruption | Gaussian blur |
| Blur radii | `0, 1, 2, 4, 8, 12` for levels `0..5` |

The persistence and matched-confidence orientations must never be reselected on a new
evaluation manifest or separately for each corruption severity. New corruption plugins inherit
these directions unless a separate, explicitly designed tuning study replaces them.

## Inputs and command

The project exposes one orchestration command:

```bash
python -m differential_uncertainty.cli run \
  --reference-manifest data/reference.csv \
  --evaluation-manifest data/evaluation.csv \
  --checkpoint models/rtdetrv2_r18vd.pth \
  --output-dir runs/blur-test
```

Runtime controls such as device, batch size, worker count, and cache shard size may be exposed.
They must not change the scientific result. There are no command-line switches for choosing
layers, bins, aggregations, score formulae, bank populations, k, orientations, or blur radii.

Each input manifest is a CSV with exactly the required semantic fields:

```csv
image_id,image_path
scene_001,images/001.jpg
scene_002,images/002.jpg
```

Relative image paths resolve from the directory holding the manifest. Image IDs must be unique
within a manifest. The reference and evaluation manifests must not overlap by image ID or
resolved file identity. COCO annotations are not accepted or required.

## Architecture

The experiment-specific code becomes one compact package:

```text
differential_uncertainty/
|-- cli.py
|-- config.py
|-- manifests.py
|-- extraction.py
|-- bank.py
|-- scoring.py
|-- evaluation.py
|-- reporting.py
`-- corruptions/
    |-- base.py
    `-- gaussian_blur.py
```

The minimum dependency closure needed to construct and execute the existing RT-DETRv2-R18
model remains under `src/`. Its pretrained checkpoint interpretation, decoder implementation,
and persistence calculation are retained rather than reimplemented. The general training and
multi-model configuration system is reduced to the single model path after parity is proven.

The command is a thin orchestrator. Numerical functions remain pure where practical so their
results can be checked with hand-built arrays. File formats and provenance validation stay at
the package boundary rather than being mixed into the mathematics.

## End-to-end data flow

### 1. Validate and fingerprint the run

Before inference, the runner:

1. parses both manifests;
2. resolves and verifies every image path;
3. rejects duplicate or overlapping inputs;
4. verifies the checkpoint can construct the fixed model;
5. records manifest content digests, checkpoint digest, fixed configuration, corruption
   identity, relevant code identity, software versions, and runtime controls.

An existing output directory is resumable only when its recorded scientific inputs match.
A mismatch is an error and never causes old and new artifacts to be mixed.

### 2. Extract the clean reference population

Load every image as RGB, resize it to 640 by 640 pixels, and convert it to a float32 tensor
scaled to `[0, 1]`, matching the current inference path. Run the frozen detector in evaluation
and inference mode on each reference image at level 0. Save boxes as float32, full logits as
float16, and layer-2 persistence fingerprints as float16, matching the current cache semantics.

For each reference image, detect the longest suffix of at least two query slots whose box,
complete logit vector, and complete layer-2 persistence fingerprint are bit-for-bit identical
to the final query. Remove that suffix. A single final query never counts as padding.

Stream all remaining layer-2 vectors through a deterministic reservoir and store exactly
25,000 float32 vectors. Fewer than 25,000 valid source vectors is an error. Reference image
order is canonicalized before sampling so a manifest row reordering cannot alter the bank.

### 3. Extract evaluation images at every severity

For each evaluation image, first resize its RGB PIL image to 640 by 640 pixels, matching the
current loader. Produce the clean image and apply PIL Gaussian blur at radii `1, 2, 4, 8, 12`
to the resized image before float32 tensor conversion. Run the same frozen detector and
persistence hook on all six versions.

Detect the repeated suffix independently at each severity, then take the union of the padded
query IDs across all six records for that image. Remove that same union at every severity. This
keeps the query population fixed: a decoder placeholder appearing or disappearing with blur
cannot masquerade as corruption sensitivity.

### 4. Build dynamic confidence groups

Convert each stored logit tensor to float32, apply sigmoid, and take the largest class value for
each query as its confidence. This is recomputed from the stored logits rather than read from a
separately rounded confidence field.

At each image and severity, rank the valid queries by this confidence and split them into ten
dynamic percentile groups. Select:

- the 90--100% group as the reference; and
- the 50--60% group as the responsive group.

The ordering is lowest confidence first. Exact confidence ties are broken by ascending query
ID through a stable sort. If the valid count is not divisible by ten, `tensor_split` semantics
put the extra queries into the lower bins. These details are fixed because float16 logits make
ties common and a different tie rule can change membership at a bin boundary.

Both the persistence score and matched confidence baseline use exactly these query IDs.

### 5. Calculate the two retained scores

For every selected query, calculate its exact Euclidean distances to the clean bank, take its
five nearest distances, and average those five. Then average the resulting query distances
within each selected decile.

Let `R` be the reference-decile mean and `M` the responsive-decile mean. The persistence score
is:

```text
relative_gap(R, M) = 2 * (M - R) / (M + R)
```

If both inputs are exactly zero, the score is defined as zero. Negative or non-finite inputs
are rejected because the symmetric relative-gap interpretation requires non-negative inputs.

For the matched confidence baseline, turn each selected query confidence `c` into uncertainty
`1 - c`, average those uncertainties within the same two deciles, and apply the identical
relative-gap formula. This baseline is sometimes called "plain softmax" in discussion, but the
implementation uses maximum sigmoid class confidence and must say so in every report.

## Metrics

### Per-image trend

For every image and each of the two score series, compute Spearman correlation between ordered
severity `[0, 1, 2, 3, 4, 5]` and the six raw scores. A complete constant curve receives signed
Spearman `0`; a missing or non-finite curve is an error for this strict workflow.

Report the median signed Spearman, median absolute Spearman, and counts of increasing,
decreasing, and flat images. The signed values remain raw evidence about direction. They are
not replaced by their absolute values.

After applying the score's fixed orientation, also report:

- adjacent consistency: the fraction of five consecutive steps that do not move backwards;
- strongest-corruption-above-clean: whether oriented level 5 is strictly above oriented level 0.

### Per-severity AUROC

For each severity `s` from 1 through 5, severity-0 scores from all evaluation images form the
negative group and severity-`s` scores form the positive group. Multiply all values by the
series' one fixed orientation before ranking them.

AUROC is the tie-aware Mann--Whitney ranking fraction. A positive score above a negative score
earns one point, a tie earns half a point, and a lower positive earns zero. The result is the
mean over all clean-versus-corrupted pairs. It is a ranking score, not a probability of
corruption.

Macro AUROC is the unweighted arithmetic mean of the five per-severity AUROCs. No severity is
dropped or weighted by how many rows survived.

### Controls and uncertainty

The raw responsive and raw reference persistence means are reported as diagnostic controls.
Where their AUROCs are compared, their directions are also frozen from the archived tuning run:
raw responsive `+1` and raw reference `-1`. They are not exposed as alternative deployable
scores.

Paired bootstrap comparisons resample complete image identities with replacement, preserving
all six severities of a sampled image. Ten thousand draws use a fixed seed. Report persistence
macro AUROC minus the matched confidence baseline and diagnostic controls, including the 2.5th
and 97.5th percentiles. The bootstrap is descriptive evidence on the supplied evaluation set;
it does not turn a tuning set into held-out validation.

## Output contract

One output directory owns both reusable work and human-facing results:

```text
<run>/
|-- artifacts/
|   |-- provenance.json
|   |-- reference-extractions/
|   |-- evaluation-extractions/
|   |-- reference-bank.pt
|   `-- scores.csv
`-- report/
    |-- per-image-scores.csv
    |-- metrics.csv
    |-- bootstrap-comparisons.csv
    |-- summary.json
    |-- report.md
    `-- figures/
        |-- reference-and-responsive-distance.png
        |-- clean-reference-relationship.png
        |-- relative-gap-by-severity.png
        `-- auroc-by-corruption-severity.png
```

Artifacts are written by resumable stages. A completed stage is published atomically, so an
interruption cannot make a partial shard look complete. The final report bundle is also
published atomically after all strict completeness checks pass.

The Markdown report uses plain language that a twelve-year-old can follow, includes concrete
arithmetic examples for kNN averaging, relative gap, matched confidence, Spearman, and AUROC,
and states both what was measured and what was not measured.

The existing nine-file historical within-image contrast bundle remains preserved together at
`/home/yuchen/YuchenZ/UE/within-image-contrast-run-record/within_image_contrast/`. It is
read-only evidence for parity and is never overwritten or mixed with a new run.

## Corruption extension point

The initial build implements only Gaussian blur. A corruption plugin supplies:

- a stable name;
- six ordered severity identities including level 0;
- serializable parameters for provenance; and
- a deterministic `apply(image, severity)` operation.

Scoring, caching, padding union, metrics, and reporting depend only on this interface. Adding a
new corruption type should require one new plugin and tests, not edits to detector extraction or
score mathematics. The single initial CLI does not expose a menu of unimplemented corruptions.

## Failure behavior

The run stops with a stage-specific, plain-language error for:

- malformed manifests, duplicate IDs, overlap, or missing/unreadable images;
- a checkpoint that does not match the fixed detector;
- unexpected query or persistence dimensions;
- inconsistent query counts across cached fields;
- a padding rule that removes too many queries to form ten non-empty groups;
- missing severities, non-finite values, or incomplete score curves;
- a stale or mismatched cache;
- an empty or undersized bank; or
- an existing final report whose provenance differs.

There is no silent row dropping, missing-value imputation, direction retuning, or destructive
overwrite. Completed matching stages remain resumable after a later stage fails.

## Faithfulness and test strategy

Implementation follows test-driven development. Required test layers are:

1. **Hand arithmetic:** exact small examples for five-neighbour averaging, confidence
   conversion, relative gap including the zero case, tie-aware AUROC, Spearman, oriented curve
   checks, and paired bootstrap grouping.
2. **Padding:** exact repeated-tail detection, a one-query non-tail, disagreement in any field,
   all-identical rejection, and the union across severities.
3. **Manifests and provenance:** path resolution, duplicate/overlap rejection, canonical order,
   cache hits, and every relevant cache invalidation.
4. **Corruptions:** clean identity, the fixed blur ladder, deterministic application, and a
   synthetic second plugin proving the interface does not require pipeline edits.
5. **Legacy evaluator parity:** feed archived or frozen score rows through old and new
   implementations and require the retained relative-gap rows, trends, orientations, AUROCs,
   and macro values to match exactly.
6. **Detector parity:** when the current checkpoint is available, run a small fixed image set
   through both paths and require equal logits and layer-2 fingerprints before padding removal.
7. **End to end:** a lightweight deterministic detector fixture builds a bank, extracts six
   levels, resumes without repeating completed inference, and publishes the exact output tree.
8. **Report:** all numeric cells reconcile with `summary.json`, all figures render, and the
   Markdown explanations name sigmoid rather than softmax and include the required caveats.
9. **Reduced repository:** every remaining Python file imports or compiles, the complete reduced
   test suite passes, and no retained entry point imports a removed training or experiment
   module.

Deletion happens only after the new vertical path passes parity and end-to-end tests. The final
handoff records the Python file count and line count before and after cleanup, the exact test
commands and results, and the detector modules that remain.

## Interpretation limits

- Historical numbers were produced with a bank that still contained some repeated padded query
  slots. Fresh clean-branch numbers may therefore differ; the change is intentional and must be
  disclosed rather than described as a regression.
- The selected bins, layer, aggregation, formula, and orientations came from earlier blur tuning.
  Results on that same tuning population remain tuning evidence.
- The workflow measures corruption ranking. Without ground-truth annotations, it does not
  measure object-detection accuracy, COCO mAP, calibration, or a probability of corruption.
- The retained relative gap was chosen for its scale-independent interpretation. The archived
  blur run did not prove it superior to raw gap; their macro AUROCs were nearly equal and no
  direct raw-versus-relative bootstrap was performed.

## Acceptance criteria

The design is complete when the `clean` branch:

1. runs the fixed pretrained detector path from two generic image manifests;
2. removes repeated padding from both the bank and evaluation images;
3. computes only the fixed layer-2 differential relative gap and its matched confidence
   baseline as complete score series;
4. supports the fixed blur ladder through a reusable corruption interface;
5. resumes safely from provenance-matched artifacts;
6. produces the complete dedicated report bundle;
7. passes arithmetic, parity, detector, end-to-end, reporting, and import tests; and
8. contains no Python dependency that is unnecessary for this vertical path.
