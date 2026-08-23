# Plain-Language Within-Image Contrast Results Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Publish a standalone Markdown report that lets a twelve-year-old understand what the real within-image corruption experiment tested, what it found, and what it did not prove.

**Architecture:** Treat the archived real-run bundle as the only scientific evidence source. Lead with the two separate verdicts, explain the experiment through a same-image “ruler” analogy, retain only decision-relevant numbers, and finish with tuning-only limitations and the next decision.

**Tech Stack:** Markdown, `jq`, repository Python environment, `rg`, Git

---

### Task 1: Establish the report evidence ledger

**Files:**
- Read: `/home/yuchen/YuchenZ/UE/within-image-contrast-run-record/within_image_contrast/summary.json`
- Read: `/home/yuchen/YuchenZ/UE/within-image-contrast-run-record/within_image_contrast/easy-report.md`
- Read: `src/scene_uncertainty/contrast_reporting.py`

- [x] **Step 1: Confirm the archived bundle is complete**

Run:

```bash
find /home/yuchen/YuchenZ/UE/within-image-contrast-run-record/within_image_contrast \
  -maxdepth 1 -type f -printf '%f\n' | sort
```

Expected: exactly the four PNG figures, three CSV tables, `easy-report.md`, and `summary.json`.

- [x] **Step 2: Confirm the run provenance and table sizes**

Run:

```bash
jq '{provenance, row_counts, bootstrap}' \
  /home/yuchen/YuchenZ/UE/within-image-contrast-run-record/within_image_contrast/summary.json
```

Expected evidence:

- 250 tuning images;
- severities 0 through 5;
- 54,000 retained source rows;
- 121,500 published per-scene contrast rows;
- 81 total candidates, of which 45 are ranked persistence candidates;
- 2,000 paired bootstrap resamples with seed `20260821`.

- [x] **Step 3: Recompute the reviewed composite verdicts**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -c 'import json; from pathlib import Path; from src.scene_uncertainty.contrast_reporting import _hypothesis_verdicts; p=Path("/home/yuchen/YuchenZ/UE/within-image-contrast-run-record/within_image_contrast/summary.json"); s=json.loads(p.read_text()); print(json.dumps(_hypothesis_verdicts(s["anchor_diagnostics"], s["candidates"]), indent=2))'
```

Expected:

- anchored: `supported on tuning`, with seven qualifying candidates;
- differential: `not worth carrying to a held-out test`, with zero qualifying candidates.

- [x] **Step 4: Confirm the headline candidates and mild-blur numbers**

Run:

```bash
jq '.candidates[] | select(.signal=="persistence" and ((.arm=="decile_00_10__50_60" and .aggregation=="mean" and .method=="relative_gap") or (.arm=="decile_90_100__50_60" and .aggregation=="mean" and .method=="raw_gap"))) | {arm, aggregation, method, macro_auroc, auroc_by_severity, beats_both_inputs, confidence_redundant}' \
  /home/yuchen/YuchenZ/UE/within-image-contrast-run-record/within_image_contrast/summary.json
```

Expected:

- strongest anchored candidate: macro AUROC `0.647872`;
- strongest overall differential candidate: macro AUROC `0.7209056`;
- differential severity-one AUROC `0.512216`, below bar `0.538`;
- differential severity-two AUROC `0.56916`, below bar `0.570`.

### Task 2: Write the layered plain-language report

**Files:**
- Create: `docs/scene-uncertainty-within-image-contrast-results.md`
- Reference: `docs/scene-uncertainty-confidence-decile-results.md`
- Reference: `docs/superpowers/specs/2026-08-23-within-image-contrast-plain-language-report-design.md`

- [x] **Step 1: Create the report with the approved layers**

Use these exact section headings and content boundaries:

```markdown
# Within-Image Corruption Contrast Experiment

## Short answer
State the anchored verdict and differential verdict in separate paragraphs.

## The question we tested
Explain the same-image ruler idea and the difference between an anchor and responsive range.

## How the test worked
Explain 250 tuning images, six blur levels, four combination methods, and three controls.

## Results at a glance
Give one compact table for the key counts and headline AUROCs.

## Result 1: The planned anchor idea worked on the tuning images
Explain the anchored diagnostics, seven qualifying candidates, and strongest anchored result.

## Result 2: The later differential idea did not pass its test
Explain why the highest overall macro AUROC still fails both pre-set mild-blur bars.

## What the measurements mean
Explain AUROC, 0.5, macro AUROC, and bootstrap resampling without probability language.

## What this result does not prove
State tuning-only selection, no held-out images, no calibrated probabilities, and no threshold.

## Bottom line
State that the anchor idea is promising on tuning and the differential idea stops here.
```

- [x] **Step 2: Keep the vocabulary accessible**

Use these translations consistently:

| Technical term | Plain-language wording |
|---|---|
| confidence percentile | a group of detector guesses sorted by confidence |
| reference or anchor range | a ruler inside the same image |
| responsive range | the group expected to change more under blur |
| AUROC | how often the score puts a blurred image above a clean image |
| macro AUROC | the average of that ordering score over blur levels 1–5 |
| bootstrap interval | a stability check made by repeatedly resampling the same images |
| held-out data | new test images that were not used to choose the method |

Keep identifiers such as `decile_00_10__50_60` in a traceability note after the plain-language
description, not as the first words a reader encounters.

- [x] **Step 3: Preserve the scientific distinction between the verdicts**

The anchored section must say:

- the two anchor ideas were chosen before reading this run;
- 28 of 30 anchor stability checks were below 1.0;
- all six anchored clean relationships beat their constant baselines;
- seven complete anchored candidates passed every control;
- the strongest anchored macro AUROC was about 0.648.

The differential section must say:

- the differential ideas were chosen after reading earlier tuning results;
- the best one had the highest overall macro AUROC, about 0.721;
- severity one was about 0.512 and severity two about 0.569;
- the pre-set bars were 0.538 and 0.570;
- therefore no differential candidate qualifies for held-out testing.

- [x] **Step 4: Include the shared supporting results without overstating them**

State that 20 of 33 derived contrasts beat both raw input ranges and that 8 of 45 persistence
candidates failed the confidence-only comparison. Explain that a failure means the complex score
did not add information beyond confidence already available from the detector.

### Task 3: Verify and commit the report

**Files:**
- Verify: `docs/scene-uncertainty-within-image-contrast-results.md`

- [x] **Step 1: Check required conclusions and limitations**

Run:

```bash
rg -n 'supported on the tuning|did not pass|250|six blur|seven|0\.648|0\.721|0\.512|0\.569|0\.538|0\.570|no held-out images|not probabilities' \
  docs/scene-uncertainty-within-image-contrast-results.md
```

Expected: every concept appears in a sentence that matches the evidence ledger.

- [x] **Step 2: Scan for misleading probability language and unexplained jargon**

Run:

```bash
rg -ni 'is [0-9]+(\.[0-9]+)? percent corrupted|probability (is|of corruption equals)|statistically significant|proves.*new images' \
  docs/scene-uncertainty-within-image-contrast-results.md
```

Expected: no matches.

Read the complete report once and confirm that `AUROC`, `bootstrap`, `anchor`, `responsive`,
`tuning`, and `held-out` are each explained at first use.

- [x] **Step 3: Check Markdown and repository cleanliness**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; the new report and the corrected design and plan are the only
changes.

- [x] **Step 4: Commit the report and plan**

Run:

```bash
git add docs/scene-uncertainty-within-image-contrast-results.md \
  docs/superpowers/specs/2026-08-23-within-image-contrast-plain-language-report-design.md \
  docs/superpowers/plans/2026-08-23-within-image-contrast-plain-language-report.md
git commit -m "docs: explain within-image contrast results plainly"
```

Expected: one documentation commit containing the denominator correction and the new report.

### Task 4: Explain the score and evaluation arithmetic

**Files:**
- Modify: `docs/scene-uncertainty-within-image-contrast-results.md`
- Modify: `docs/superpowers/specs/2026-08-23-within-image-contrast-plain-language-report-design.md`
- Modify: `docs/superpowers/plans/2026-08-23-within-image-contrast-plain-language-report.md`
- Read: `src/scene_uncertainty/contrast_scores.py`
- Read: `src/scene_uncertainty/corruption_metrics.py`
- Read: `src/scene_uncertainty/decile_scoring.py`
- Read: `/home/yuchen/YuchenZ/UE/within-image-contrast-run-record/within_image_contrast/per_scene_contrasts.csv`
- Read: `/home/yuchen/YuchenZ/UE/within-image-contrast-run-record/within_image_contrast/candidate_metrics.csv`

- [x] **Step 1: Add the complete Test 2 comparison table**

Replace the two-row mild-blur table with all five blur levels plus an average row:

```markdown
| Blur level | Test 2 AUROC | Plain confidence baseline | Difference | Decision target |
|---|---:|---:|---:|---|
| 1 | 0.512 | 0.522 | -0.010 | Above 0.538: No |
| 2 | 0.569 | 0.567 | +0.002 | Above 0.570: No |
| 3 | 0.669 | 0.645 | +0.024 | Reported only |
| 4 | 0.893 | 0.806 | +0.087 | Reported only |
| 5 | 0.961 | 0.895 | +0.066 | Reported only |
| Average | 0.721 | 0.687 | +0.034 | Not a separate target |
```

Immediately explain that the baseline is the same two-range calculation made from
`1 - maximum sigmoid class confidence`; “plain softmax” is only shorthand, not the literal
operation in the implementation. State that the baseline is slightly better at level 1.

- [x] **Step 2: Add real worked examples for the two scores**

Show image 885 as an illustration, using enough decimal places to reproduce the stored result:

```text
Test 1, clean: 2 * (0.083252 - 0.085921) / (0.083252 + 0.085921) = -0.03156
Test 1, blur 5: 2 * (0.083908 - 0.078284) / (0.083908 + 0.078284) = +0.06935
Test 2, clean: 0.083252 - 0.102030 = -0.01878
Test 2, blur 5: 0.083908 - 0.077775 = +0.00613
Confidence baseline, clean: 0.958912 - 0.577506 = 0.38141
Confidence baseline, blur 5: 0.912514 - 0.739659 = 0.17286
```

Explain `reference`, `responsive`, the relative gap's scale adjustment, and the confidence
baseline's negative locked direction.

- [x] **Step 3: Explain Spearman and AUROC with concrete arithmetic**

Explain that Spearman ranks each image's six values from clean through blur level 5. Its median
signed value locks one candidate-wide direction, once. Explain AUROC with oriented clean scores
`[0, 1]` and blurred scores `[1, 2]`: three wins and one tie produce
`(1 + 0.5 + 1 + 1) / 4 = 0.875`. State that each real per-level AUROC uses all
`250 * 250 = 62,500` clean-versus-blurred comparisons and that macro AUROC is the mean of five
per-level AUROCs.

- [x] **Step 4: Report the other recorded measurements**

Include this compact table:

```markdown
| Measurement | Test 1 | Test 2 | Plain confidence baseline |
|---|---:|---:|---:|
| Median signed Spearman | +0.657 | +0.829 | -0.914 |
| Images moving in the locked direction | 86.8% | 96.8% | 86.0% |
| Adjacent blur steps moving the locked way | 62.6% | 73.4% | 74.1% |
| Blur level 5 stronger than clean after direction adjustment | 86.4% | 97.2% | 89.6% |
```

Explain why negative Spearman for the confidence baseline means its raw gap usually falls with
blur, not that the baseline is automatically poor. Briefly list the additional decision checks:
complete six-level coverage, anchor stability, clean relationship, both raw inputs, matched
confidence control, and paired bootstrap intervals.

- [x] **Step 5: Verify every displayed calculation and claim**

Run:

```bash
python3 -c 'from statistics import mean; assert round(2*(.083251953125-.08592122048139572)/(.083251953125+.08592122048139572),5)==-.03156; assert round(2*(.0839080810546875-.07828368991613388)/(.0839080810546875+.07828368991613388),5)==.06935; assert round(.083251953125-.1020304337143898,5)==-.01878; assert round(.0839080810546875-.07777506858110428,5)==.00613; assert round(.9589120149612427-.5775061249732971,5)==.38141; assert round(.9125139713287354-.739658772945404,5)==.17286; assert mean([.512216,.56916,.669056,.893264,.960832])==.7209056; assert (1+.5+1+1)/4==.875; print("worked calculations verified")'
rg -n 'relative gap|raw gap|sigmoid|Spearman|62,500|0\.875|0\.657|0\.829|-0\.914|62\.6%|73\.4%|74\.1%|86\.4%|97\.2%|89\.6%' docs/scene-uncertainty-within-image-contrast-results.md
git diff --check
```

Expected: `worked calculations verified`, every required explanation and value is present, and
there are no whitespace errors.

- [x] **Step 6: Commit the expanded explanation**

Run:

```bash
git add docs/scene-uncertainty-within-image-contrast-results.md \
  docs/superpowers/specs/2026-08-23-within-image-contrast-plain-language-report-design.md \
  docs/superpowers/plans/2026-08-23-within-image-contrast-plain-language-report.md
git commit -m "docs: explain contrast score calculations"
```

Expected: one documentation commit with the expanded calculation and metric explanations.
