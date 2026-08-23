# Plain-Language Within-Image Contrast Results Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a standalone Markdown report that lets a twelve-year-old understand what the real within-image corruption experiment tested, what it found, and what it did not prove.

**Architecture:** Treat the archived real-run bundle as the only scientific evidence source. Lead with the two separate verdicts, explain the experiment through a same-image “ruler” analogy, retain only decision-relevant numbers, and finish with tuning-only limitations and the next decision.

**Tech Stack:** Markdown, `jq`, repository Python environment, `rg`, Git

---

### Task 1: Establish the report evidence ledger

**Files:**
- Read: `/home/yuchen/YuchenZ/UE/within-image-contrast-run-record/within_image_contrast/summary.json`
- Read: `/home/yuchen/YuchenZ/UE/within-image-contrast-run-record/within_image_contrast/easy-report.md`
- Read: `src/scene_uncertainty/contrast_reporting.py`

- [ ] **Step 1: Confirm the archived bundle is complete**

Run:

```bash
find /home/yuchen/YuchenZ/UE/within-image-contrast-run-record/within_image_contrast \
  -maxdepth 1 -type f -printf '%f\n' | sort
```

Expected: exactly the four PNG figures, three CSV tables, `easy-report.md`, and `summary.json`.

- [ ] **Step 2: Confirm the run provenance and table sizes**

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

- [ ] **Step 3: Recompute the reviewed composite verdicts**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -c 'import json; from pathlib import Path; from src.scene_uncertainty.contrast_reporting import _hypothesis_verdicts; p=Path("/home/yuchen/YuchenZ/UE/within-image-contrast-run-record/within_image_contrast/summary.json"); s=json.loads(p.read_text()); print(json.dumps(_hypothesis_verdicts(s["anchor_diagnostics"], s["candidates"]), indent=2))'
```

Expected:

- anchored: `supported on tuning`, with seven qualifying candidates;
- differential: `not worth carrying to a held-out test`, with zero qualifying candidates.

- [ ] **Step 4: Confirm the headline candidates and mild-blur numbers**

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

- [ ] **Step 1: Create the report with the approved layers**

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

- [ ] **Step 2: Keep the vocabulary accessible**

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

- [ ] **Step 3: Preserve the scientific distinction between the verdicts**

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

- [ ] **Step 4: Include the shared supporting results without overstating them**

State that 20 of 45 derived contrasts beat both raw input ranges and that 8 of 45 persistence
candidates failed the confidence-only comparison. Explain that a failure means the complex score
did not add information beyond confidence already available from the detector.

### Task 3: Verify and commit the report

**Files:**
- Verify: `docs/scene-uncertainty-within-image-contrast-results.md`

- [ ] **Step 1: Check required conclusions and limitations**

Run:

```bash
rg -n 'supported on the tuning|did not pass|250|six blur|seven|0\.648|0\.721|0\.512|0\.569|0\.538|0\.570|no held-out images|not probabilities' \
  docs/scene-uncertainty-within-image-contrast-results.md
```

Expected: every concept appears in a sentence that matches the evidence ledger.

- [ ] **Step 2: Scan for misleading probability language and unexplained jargon**

Run:

```bash
rg -ni 'percent corrupted|chance that.*corrupt|probability of corruption|statistically significant|proves.*new images' \
  docs/scene-uncertainty-within-image-contrast-results.md
```

Expected: no matches.

Read the complete report once and confirm that `AUROC`, `bootstrap`, `anchor`, `responsive`,
`tuning`, and `held-out` are each explained at first use.

- [ ] **Step 3: Check Markdown and repository cleanliness**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only the new report and this implementation plan are uncommitted.

- [ ] **Step 4: Commit the report and plan**

Run:

```bash
git add docs/scene-uncertainty-within-image-contrast-results.md \
  docs/superpowers/plans/2026-08-23-within-image-contrast-plain-language-report.md
git commit -m "docs: explain within-image contrast results plainly"
```

Expected: one documentation commit containing the plan and the new report.
