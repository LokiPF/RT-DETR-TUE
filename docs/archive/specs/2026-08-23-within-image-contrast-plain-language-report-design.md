# Plain-Language Within-Image Contrast Results Report Design

**Date:** 2026-08-23

## Goal

Create `docs/scene-uncertainty-within-image-contrast-results.md`, a standalone report that
explains what the within-image corruption contrast experiment tested and what the real 250-image
run found. A reader around age twelve should be able to understand the main idea, result, and
limitations without knowing machine-learning terminology.

## Evidence source

The report will use the archived real-run bundle at
`/home/yuchen/YuchenZ/UE/within-image-contrast-run-record/within_image_contrast`. It will not use
synthetic test results as scientific evidence. The reviewed verdict rules in
`src/scene_uncertainty/contrast_reporting.py` will be applied to the archived diagnostics and
candidate records.

## Structure

The report will use a layered structure:

1. A short answer stating the two verdicts immediately.
2. A simple explanation of the question and the two confidence ranges compared within one image.
3. A description of the data, blur levels, score combinations, and fairness controls.
4. A small table of the key results.
5. Separate explanations of the anchored and differential findings.
6. Worked examples showing exactly how the strongest Test 1 and Test 2 scores are calculated.
7. A plain-language explanation of Spearman correlation, AUROC, and resampling.
8. A limitations section and a clear next decision.

The main text will favor short sentences, familiar words, and concrete analogies. Technical names
will appear only when they let a reader trace a statement back to the result files.

## Scientific conclusions to preserve

- The experiment used 250 tuning images at six blur levels, from clean through severity five.
- It used saved scene scores only; it did not rerun the detector, comparison bank, or nearest-
  neighbour search.
- The pre-declared anchored hypothesis is supported on tuning. Seven anchored candidates meet the
  complete composite gate.
- The strongest anchored candidate has macro AUROC about 0.648. It is the
  `decile_00_10__50_60` mean `relative_gap` candidate.
- The strongest result overall is the post-selected differential mean `raw_gap`, with macro AUROC
  about 0.721.
- That differential result does not pass the mild-blur decision bars: its severity-one AUROC is
  about 0.512 versus the required 0.538, and severity two is about 0.569 versus the required
  0.570. No differential candidate qualifies for a held-out test.
- The differential section must show Test 2 and its matched confidence-only baseline at all five
  corrupted blur levels. Label the baseline `Plain confidence baseline ("plain softmax"
  shorthand)` and state that it uses the same two confidence ranges as Test 2, rather than an
  unmatched whole-image score.
- The five Test 2 AUROCs are 0.512, 0.569, 0.669, 0.893, and 0.961. The corresponding plain
  softmax AUROCs are 0.522, 0.567, 0.645, 0.806, and 0.895. Their macro AUROCs are 0.721 and
  0.687, respectively.
- The table must show Test 2 minus softmax differences of -0.010, +0.002, +0.024, +0.087, and
  +0.066, plus their +0.034 macro difference in the average row. It must say plainly that
  softmax was slightly better at blur level 1.
- The requested “plain softmax” name is shorthand rather than the implementation's literal
  operation. The detector confidence is the maximum sigmoid class score. Its uncertainty input
  is `1 - confidence`, and the baseline then uses the same two confidence ranges, mean summary,
  and raw-gap calculation as Test 2. The longer label above keeps the report familiar and
  technically accurate.
- Only blur levels 1 and 2 have decision targets. Levels 3 through 5 are marked `Reported only`,
  not as passes, because they were not part of the differential gate.
- The anchored diagnostics show 28 of 30 measured stability ratios below 1.0, and all six
  anchored clean relationships beat their constant baselines. Counts that include differential
  arms must not be described as evidence about anchors.
- Twenty of 33 derived contrasts beat both raw input ranges. Eight of 45 persistence candidates
  fail the confidence-only control.
- Explain the strongest Test 1 score as
  `2 * (responsive - reference) / (responsive + reference)` and the strongest Test 2 score as
  `responsive - reference`. Use archived image 885 at clean severity 0 and blur severity 5 as a
  concrete example, making clear that these are illustrations from one image rather than the
  whole result.
- Explain that the confidence baseline first summarizes `1 - confidence` within the same two
  query ranges and then applies Test 2's raw gap. Its direction is negative, so a smaller raw gap
  is read as stronger evidence of blur after that one direction is locked.
- AUROC is computed separately for each blur level by comparing all 250 oriented scores at that
  level with all 250 clean scores: 62,500 clean-versus-blurred pairs. A win contributes one, a
  tie one half, and a loss zero. Macro AUROC is the ordinary mean of the five level AUROCs.
- AUROC was not the only measurement. Report the top candidates' median per-image signed
  Spearman correlations, adjacent-step consistency, and maximum-blur-above-clean rates. Also
  explain the anchor stability and clean-relationship diagnostics, raw-input and confidence
  controls, full-data coverage, and paired bootstrap intervals.

## Language and interpretation rules

- Explain confidence-percentile ranges as groups of detector guesses sorted by confidence.
- Explain the anchor as a ruler or reference mark inside the same image.
- Explain AUROC as how often a blurred image receives a stronger corruption score than a clean
  image; 0.5 is guessing and 1.0 is perfect ordering.
- Explain Spearman correlation as checking whether one image's six scores generally move in
  order as blur increases. `+1` means a perfectly rising order, `-1` a perfectly falling order,
  and `0` no consistent ranked direction. Explain that the median signed Spearman locks one
  direction per candidate before AUROC is calculated; the direction is not re-chosen for each
  blur level.
- Do not describe any score as the probability that an image is corrupted.
- Keep the anchored and differential verdicts separate. The differential arms were chosen after
  reading tuning results and cannot be presented as confirmed performance.
- State prominently that no held-out images were used.
- Describe bootstrap intervals as a stability check made by repeatedly resampling the same tuning
  images, not as proof about new images.

## Verification

Before delivery:

- cross-check every reported number against `summary.json` or `easy-report.md` in the archived
  real-run bundle;
- recompute the two composite verdicts with the reviewed implementation;
- scan for unexplained jargon and probability claims;
- recompute each worked score from its displayed reference and responsive values and verify that
  it matches `per_scene_contrasts.csv` after rounding;
- cross-check the Spearman, direction-consistency, and endpoint values against
  `candidate_metrics.csv`;
- verify the AUROC example by enumerating its four clean-versus-blurred comparisons and verify
  that the five Test 2 AUROCs average to its reported macro AUROC;
- confirm the report says that the anchored result is tuning-only and the differential result
  should not advance to held-out testing;
- run `git diff --check` and inspect the rendered Markdown structure as plain text.

## Out of scope

The report will not rerun the experiment, alter the archived output bundle, claim held-out
generalization, choose a deployment threshold, or reproduce every candidate row from the
technical result files.
