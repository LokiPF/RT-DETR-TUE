# Docs

The documents behind the paper. Older work is in `archive/`.

## The method and its evidence

- `conv-tu-conditioned-results.md`: the 5,000-image confirmation of the two-axis score (2 October). Tables:
  `results/conv-tu-conditioned/`.
- `results/coco/`: the clean branch's report. It covers every row on all four image sets, with intervals.
- `coco-detectors-results.md`: the two-axis score, unchanged, on three more frozen COCO detectors next to
  RT-DETRv2-R18: YOLO11m, Faster R-CNN R50-FPN v2 and RF-DETR-M (5 October). Plan:
  `superpowers/plans/2026-10-04-four-detectors-coco-c.md`.
- `results/coco-detectors/`: each new detector's report, the table of all four (`summary.md`, `summary.csv`), each
  detector's two arms per condition (`arms.csv`, from `../scripts/paper/detector_arms.py`) and the tap explorations.
- `coco-iqa-results.md`: four detector-free image-quality baselines against the two-axis score on all four
  detectors: NIQE (refitted and published), ARNIQA's quality, ARNIQA's clean prototype and zero-shot CLIP-IQA
  (5 October). Plan: `superpowers/plans/2026-10-05-iqa-baselines-coco-c.md`.
- `results/coco-iqa/`: each detector's report with the five image-quality rows, the table of all four (`summary.md`,
  `summary.csv`), the timing (`timing.json`), the NIQE refit's counts (`fit.json`), NIQE's block counts per
  condition (`niqe-blocks.csv`) and the GPU precision check (`precision-check.txt`).
- `roundtable-2026-10-01-detection/`: the detection roundtable that proposed the two-axis score. Its
  `verify_panel_rows.py` imports the old `differential_uncertainty` package, so run it at the commit named in
  `../archive/README.md`.
- `superpowers/plans/2026-10-01-content-conditioned-confirmation.md`: the confirmation's pre-registered plan, with
  its amendments.
- `paper-storyline.md`: the paper's storyline. It gives the chain of observations that leads to the method, a
  section-by-section outline, and the figures and tables still to make.
- `results/paper-evidence/`: the evidence tables and draft figures behind the storyline's claims (2 October). The
  scripts are in `../scripts/paper/`.
- `dev-log.md`: dated observations and decisions, newest first.
- `todo.md`: the open ablations and experiments.

## Baselines and setting

- `driving-benchmark-baselines-and-metrics.md`: the decision record for baselines and metrics (27 September).
- `coco-baseline-numbers.md`: the six baselines on COCO (29 September). Tables: `results/coco-baselines/`.
- `literature-review-image-corruption-detection.md`: related work (27 September).

## The conv-TU pilot (retired)

- `conv-tu-pilot-design.md` and `conv-tu-pilot-results.md`: topology on the conv layers. Its channel-means control
  led to the current method. Tables: `results/conv-tu-pilot/`, `results/conv-tu-pilot-channels/`. Code:
  `../archive/conv_tu/`.

## This branch

- `superpowers/specs/2026-10-02-clean-branch-design.md` and `superpowers/plans/2026-10-02-clean-branch.md`: the
  design and the plan of the clean branch.
