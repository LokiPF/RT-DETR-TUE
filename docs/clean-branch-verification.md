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
cc6b0f6c88498dd1fccf81e2b41b1ce2b68a7bf5
```

## Historical baseline

Before pruning, the known test baseline was **1,371 passed, 86 warnings**. This
is a historical result from the larger repository, not the result of a current
test run.

## Current reduced test suite

The complete retained suite was run with:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty -q -rs
```

Its exact summary was:

```text
777 passed in 141.26s (0:02:21)
```

This full run includes the image-fingerprint, runtime-regime, practical corruption
interface, cache and artifact integrity, report reconciliation, final input and
artifact checks, and real-checkpoint GPU batch-resume tests. The real GPU check
performed detector inference at batch size 1 and verified that a batch size 2
resume was refused; it does not assert batch invariance.

The corruption checks validate a name, six ordered severities, and
`apply(image, level)`. They refuse cache reuse when the recorded name or severity
table changes. Resume otherwise assumes the same corruption code and hidden
settings. The workflow intentionally does not inspect arbitrary Python dependencies
or in-memory state, so code or hidden-setting changes require a new output folder.
Output folders created with the old strict corruption-implementation schema should
not be reused after this code change; start a new output directory.

The final checks still perform full input-image validation, audit the published
artifacts, and finish with lightweight input signatures.

The two parity files were also run together with skip reasons enabled:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/differential_uncertainty/test_detector_parity.py \
  tests/differential_uncertainty/test_legacy_parity.py -q -rs
```

Their exact combined summary was:

```text
14 passed in 3.60s
```

There was no skipped-test summary. The parity checks ran rather than being
hidden behind an unavailable-checkpoint or incompatible-platform skip.

### Detector parity

The detector test was run separately and reported:

```text
11 passed in 2.96s
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
3 passed in 0.82s
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
22230 total
```

That is **43 Python files and 22,230 lines** in the retained surface.

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

## Environment and documentation evidence

The tests recorded here target code commit
`cc6b0f6c88498dd1fccf81e2b41b1ce2b68a7bf5`, not the later documentation
commit. This documentation update changes no Python source, so the tested code
commit and retained Python counts recorded above remain the same.

A fresh environment query returned:

```text
Python 3.11.15
PyTorch 2.11.0+cu128
torchvision 0.26.0+cu128
CUDA runtime 12.8
cuDNN 91900 (version 9.19.0)
CUDA available True
GPU NVIDIA GeForce RTX 5090
compute capability (12, 0)
```

A direct parser check returned the documented runtime defaults:

```text
device cuda:0
batch_size 1
shard_size 50
```

On the tested host, `findmnt` reported `proc /proc`, `/proc/self/fd` was a
directory, and the C library exposed the `renameat2` symbol. The report tests
exercise the no-replace publication path.

A HEAD-only request returned HTTP 200 for the
[official repository](https://github.com/lyuwenyu/RT-DETR) and for the final
asset behind the [official checkpoint link](https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth).
This checked the links without downloading the 81,198,974-byte body. The
official repository's [`hubconf.py`](https://github.com/lyuwenyu/RT-DETR/blob/main/hubconf.py)
mapped `rtdetrv2_r18vd` to that exact release URL.

The existing local official checkpoint was checked independently:

```text
2ace52184b620204004509b72752ac7bfe64aadaf7fc1d076b18df8ab5a5c77e  rtdetrv2_r18vd_120e_coco_rerun_48.1.pth
```
