# Docs

The documents behind the paper. Older work is in `archive/`.

## The method and its evidence

- `conv-tu-conditioned-results.md`: the 5,000-image confirmation of the two-axis score (2 October). Tables:
  `results/conv-tu-conditioned/`.
- `results/coco/`: the clean branch's report. It covers every row on all four image sets, with intervals.
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
