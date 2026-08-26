# Scene Reliability Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a plain-language scene-reliability reference, a reproducible evidence CSV, and a publication-ready plot for construction and agricultural vehicle design teams.

**Architecture:** Keep the reference and its two evidence assets in `docs/`. Derive every plotted value from the recorded within-image contrast report, preserve planned-versus-exploratory provenance in the labels and caption, and validate both file links and rendered output.

**Tech Stack:** Markdown, CSV, Python 3 standard library, Matplotlib, Pillow.

---

## File structure

- `docs/scene-reliability-reference.md`: standalone, reader-facing explanation with embedded local plot and compact evidence table.
- `docs/scene-reliability-group-evidence.csv`: canonical, machine-readable values copied from the recorded 250-image Gaussian-blur tuning study.
- `docs/scene-reliability-group-reaction.png`: PNG line chart generated only from the CSV values.
- `tools/generate_scene_reliability_group_figure.py`: small reproducible generator that validates the CSV schema and emits the PNG.

### Task 1: Create the canonical evidence table

**Files:**
- Create: `docs/scene-reliability-group-evidence.csv`
- Test: inline Python CSV assertions

- [ ] Write the five blur-level rows for both comparisons: planned 0–10% versus 50–60% relative gap (0.478, 0.515, 0.600, 0.802, 0.845) and exploratory 90–100% versus 50–60% raw gap (0.512, 0.569, 0.669, 0.893, 0.961).
- [ ] Verify the CSV has exactly ten data rows, blur levels 1–5 for each named comparison, and every AUROC lies in `[0, 1]`.
- [ ] Commit the CSV with message `Add group-reaction evidence data`.

### Task 2: Generate and verify the publication-ready figure

**Files:**
- Create: `tools/generate_scene_reliability_group_figure.py`
- Create: `docs/scene-reliability-group-reaction.png`
- Test: script validation and PNG inspection

- [ ] Write the generator to require the CSV columns `comparison_id`, `comparison_label`, `provenance`, `blur_level`, and `auroc`; reject duplicate comparison/level pairs and missing levels 1–5.
- [ ] Plot both AUROC curves on a 0–1 y-axis with a chance-level line at 0.5, an explicit Gaussian-blur/250-image-tuning subtitle, and a legend that marks the latter curve exploratory.
- [ ] Run `python tools/generate_scene_reliability_group_figure.py docs/scene-reliability-group-evidence.csv docs/scene-reliability-group-reaction.png` and verify the PNG opens, is at least 1200 pixels wide, and has nonzero dimensions.
- [ ] Commit the generator and PNG with message `Add group-reaction evidence figure`.

### Task 3: Write and check the design-team reference

**Files:**
- Create: `docs/scene-reliability-reference.md`
- Test: Markdown link and terminology checks

- [ ] Write the reference in the approved scene-reliability-first order: off-road motivation; minimum query background; fingerprints; clean bank; query distances; empirical group discovery; relative-gap score; interpretation boundaries; paired construction/agriculture examples; glossary.
- [ ] Include the local PNG with `![Recorded group-reaction evidence](scene-reliability-group-reaction.png)` and a compact table matching the CSV values.
- [ ] State that rain, fog, dust, glare, darkness, and lens contamination are motivating deployment conditions, while the recorded evidence is controlled Gaussian blur only.
- [ ] State that the current implementation fixes the 90–100% reference and 50–60% responsive groups with a relative gap, and distinguish this from the exploratory historical raw-gap curve.
- [ ] Verify every local Markdown link resolves, the formula is `2 × (responsive − reference) / (responsive + reference)`, both application domains have worked examples, and no unsupported deployment claim remains.
- [ ] Commit the reference with message `Add scene reliability reference`.

### Task 4: Final evidence review

**Files:**
- Verify: all four files above

- [ ] Run the figure generator again from the committed CSV and confirm the resulting PNG is byte-stable or inspect the regenerated figure for identical visual content.
- [ ] Run `git diff --check`, inspect `git status --short`, and confirm only the planned files changed.
- [ ] Read the complete reference in rendered Markdown form and visually inspect the PNG before reporting completion.
- [ ] Commit any verification-only corrections with message `Polish scene reliability reference`.
