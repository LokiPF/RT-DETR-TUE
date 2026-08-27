# COCO ImageCorruptions Per-Severity Results Design

## Purpose

Extend the 250/250 COCO ImageCorruptions pilot summary so readers can inspect
AUROC separately at corruption levels 1 through 5, rather than seeing only
the five-level macro-AUROC.

## Evidence source

Derive values only from the tracked pilot summary artifact:

`docs/results/coco-imagecorruptions-250/summary.json`

For every corruption family, use
`evaluation.series.<method>.auroc_by_severity["1".."5"]`. Each entry compares
the clean level-0 distribution with that one corruption level across the 250
evaluation images. It is not a paired-only AUROC and it does not pool severity
values across families.

## Presentation

Add a `## Per-severity AUROC` section immediately after the headline method
comparison in `docs/coco-imagecorruptions-250-pilot-results.md`.

The section will state the clean-versus-each-corruption-level comparison and include two 19-row tables, each with corruption family
and columns for levels 1, 2, 3, 4, and 5:

1. Persistence relative gap, whose larger score is oriented as more corrupted.
2. Direct confidence maximum, whose lower raw confidence is oriented as more
   corrupted before AUROC calculation.

Values will display to six decimal places and rows will use the benchmark's
fixed corruption order. Existing macro-AUROC and interpretation sections stay
unchanged.

## Validation

The generated table values must reconcile to all 19 x 5 entries for both
methods in the tracked JSON artifact. The document must retain portable
relative links and pass `git diff --check`.
