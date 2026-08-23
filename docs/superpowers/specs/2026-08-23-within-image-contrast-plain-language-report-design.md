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
6. A plain-language explanation of AUROC and resampling.
7. A limitations section and a clear next decision.

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
  corrupted blur levels. Label the baseline `Plain softmax (confidence-only)` and state that it
  uses the same two confidence ranges as Test 2, rather than an unmatched whole-image score.
- The five Test 2 AUROCs are 0.512, 0.569, 0.669, 0.893, and 0.961. The corresponding plain
  softmax AUROCs are 0.522, 0.567, 0.645, 0.806, and 0.895. Their macro AUROCs are 0.721 and
  0.687, respectively.
- The table must show Test 2 minus softmax differences of -0.010, +0.002, +0.024, +0.087, and
  +0.066, plus their +0.034 macro difference in the average row. It must say plainly that
  softmax was slightly better at blur level 1.
- Only blur levels 1 and 2 have decision targets. Levels 3 through 5 are marked `Reported only`,
  not as passes, because they were not part of the differential gate.
- The anchored diagnostics show 28 of 30 measured stability ratios below 1.0, and all six
  anchored clean relationships beat their constant baselines. Counts that include differential
  arms must not be described as evidence about anchors.
- Twenty of 33 derived contrasts beat both raw input ranges. Eight of 45 persistence candidates
  fail the confidence-only control.

## Language and interpretation rules

- Explain confidence-percentile ranges as groups of detector guesses sorted by confidence.
- Explain the anchor as a ruler or reference mark inside the same image.
- Explain AUROC as how often a blurred image receives a stronger corruption score than a clean
  image; 0.5 is guessing and 1.0 is perfect ordering.
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
- confirm the report says that the anchored result is tuning-only and the differential result
  should not advance to held-out testing;
- run `git diff --check` and inspect the rendered Markdown structure as plain text.

## Out of scope

The report will not rerun the experiment, alter the archived output bundle, claim held-out
generalization, choose a deployment threshold, or reproduce every candidate row from the
technical result files.
