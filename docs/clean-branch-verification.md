# Clean branch verification

This record was made on 2026-08-24 from the `clean` worktree. It separates
historical evidence from commands run on the reduced repository.

## Commits checked

The full branch-point commit is:

```text
3f0775dbe6ee8c8ea72e9f09646f2a0ef193457b
```

The code commit tested immediately before this documentation change is:

```text
bac41c132fd38df95c4869d34fd0b4abb1b5394f
```

## Historical baseline

Before pruning, the known test baseline was **1,371 passed, 86 warnings**. This
is a historical result from the larger repository, not the result of a current
test run.

## Current reduced test suite

The complete retained suite was run with:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty -q
```

Its exact summary was:

```text
764 passed in 134.40s (0:02:14)
```

The two parity files were also run together with skip reasons enabled:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/differential_uncertainty/test_detector_parity.py \
  tests/differential_uncertainty/test_legacy_parity.py -q -rs
```

Their exact combined summary was:

```text
14 passed in 3.63s
```

There was no skipped-test summary. The parity checks ran rather than being
hidden behind an unavailable-checkpoint or incompatible-platform skip.

### Detector parity

The detector test was run separately and reported:

```text
11 passed in 2.91s
```

`nvidia-smi` identified the local GPU as:

```text
NVIDIA GeForce RTX 5090, 580.173.02, 32607 MiB
```

This test binds the golden fixture to the expected checkpoint name and SHA-256,
then checks exact detector logits, boxes, and layer-2 persistence features on
the recorded CUDA platform. It protects the fixed detector inference path.

### Archived-number parity

The archived-number test was run separately and reported:

```text
3 passed in 0.80s
```

This test reads the historical `per_scene_contrasts.csv`, checks the complete
250-image by six-severity roster, and reproduces the published persistence and
matched-confidence Spearman and AUROC values. It protects the old published
numbers; it does not require the new workflow to regenerate those same numbers.

```text
The regenerated metrics are allowed to differ from the historical run because the new
reference bank intentionally excludes repeated padded detector queries. The archive test
protects the old published numbers; the detector test protects model fidelity.
```

## Retained Python inventory

The retained workflow, detector closure, and tests were counted with:

```bash
find differential_uncertainty src tests/differential_uncertainty \
  -name '*.py' -type f | sort | wc -l
```

Observed result:

```text
43
```

Their line count was measured with:

```bash
find differential_uncertainty src tests/differential_uncertainty \
  -name '*.py' -type f -print0 | sort -z | xargs -0 wc -l | tail -n 1
```

Observed result:

```text
21424 total
```

That is **43 Python files and 21,424 lines** in the retained surface.

## Historical report bundle

The historical result bundle remains outside the pruned worktree at
`/home/yuchen/YuchenZ/UE/within-image-contrast-run-record/within_image_contrast`.
The read-only listing command was:

```bash
find /home/yuchen/YuchenZ/UE/within-image-contrast-run-record/within_image_contrast \
  -maxdepth 1 -type f -printf '%f\n' | sort
```

It returned exactly these nine files:

```text
anchor_and_responsive_actual_distance.png
anchor_diagnostics.csv
auroc_by_blur_severity.png
candidate_metrics.csv
clean_anchor_relationship.png
contrast_scores_by_severity.png
easy-report.md
per_scene_contrasts.csv
summary.json
```
