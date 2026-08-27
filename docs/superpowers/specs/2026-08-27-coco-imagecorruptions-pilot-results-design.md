# COCO ImageCorruptions Pilot Results Summary Design

## Purpose

Create a concise, research-facing Markdown summary of the completed COCO-val
ImageCorruptions pilot. It should answer whether the primary persistence
relative-gap score remains useful across 19 corruption families when compared
with the matched-confidence control and direct model-confidence baselines.

## Evidence source

The document will derive every reported number from the completed 250/250
pilot leaderboard in the `feat/imagecorruptions-coco-benchmark` worktree:

- `runs/coco-imagecorruptions-250/benchmark-report/corruption-metrics.csv`
- `runs/coco-imagecorruptions-250/benchmark-report/summary.json`

It will not recompute metrics or introduce results from any 2,500/2,500 run.

## Document structure

The new `docs/coco-imagecorruptions-250-pilot-results.md` will contain:

1. A scope statement: COCO-val, 250 clean-reference images, 250 evaluation
   images, 19 corruption families, and the fact that this is not the final
   2,500/2,500 experiment.
2. A headline method table with mean and median macro-AUROC for the primary
   persistence relative gap, matched-confidence relative-gap control, direct
   mean confidence, and direct maximum confidence.
3. A compact family table that highlights primary-score strengths and
   weaknesses alongside direct-maximum confidence, rather than presenting a
   visually overwhelming 19-by-6 table.
4. Plain-language interpretation: direct maximum confidence is the strongest
   overall baseline in this pilot; the primary differential score ranks second
   overall and exceeds the matched-confidence control and direct mean
   confidence.
5. Limitations: corruption detection/ranking only (not detector mAP or
   accuracy); pilot scale; and ImageCorruptions' stochastic pixels are not
   expected to reproduce exactly across separate runs at the same named
   corruption/severity.

## Integrity and presentation rules

- Use macro-AUROC values as published by the root leaderboard; do not pool raw
  severity values across corruption families.
- State score orientation for direct confidence: lower raw confidence is
  treated as stronger corruption before AUROC calculation.
- Use explicit labels for pilot-only inferences and avoid claims about the
  unstarted final-scale experiment.
- Link readers to the CSV and JSON artifacts in the feature worktree for the
  full per-family/six-method values.
