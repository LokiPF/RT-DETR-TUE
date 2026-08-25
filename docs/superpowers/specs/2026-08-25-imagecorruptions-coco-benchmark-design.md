# ImageCorruptions COCO benchmark design

**Status:** Approved for implementation on 2026-08-25

## Purpose

Extend the Gaussian-blur-only differential-corruption uncertainty workflow to a
reproducible COCO validation benchmark. The pilot builds one clean fingerprint
bank from 250 images and evaluates the method on a different deterministic set of
250 images. It compares the primary persistence relative-gap score with the
existing matched-confidence control and two direct model-confidence baselines.

The pilot is strictly limited to 250 reference and 250 evaluation images. It must
not launch a 2,500/2,500 run. A later larger run uses the same command with new
counts and a new output directory.

## Corruption scope

The matrix contains 19 corruptions: the existing project `gaussian_blur` and the
other 18 named routines exposed by `imagecorruptions` 1.1.2. The upstream package
calls its first 15 routines “common” and its final four “validation” corruptions.
Because its Gaussian blur is one of the final four, this selection means 14
upstream common routines plus four upstream validation routines.

Every adapter keeps the public contract:

```python
class Corruption(Protocol):
    name: str
    severities: tuple[Severity, ...]

    def apply(self, image: Image.Image, level: int) -> Image.Image: ...
```

Level 0 returns the exact source image. Levels 1 through 5 convert RGB Pillow
pixels to a `uint8` NumPy array and call `imagecorruptions.corrupt` with that
named corruption and severity. Reported parameters are ordinal levels `0..5`, not
invented common physical units for family-specific settings.

Some package corruptions are stochastic. The benchmark uses the upstream routine
at the named corruption and requested severity without requiring repeat calls to
produce identical pixels. An interrupted run reuses its completed extracted
artifacts, while an independently started run may receive a different random draw
at the same corruption/severity. This is acceptable for the requested robustness
comparison; the report identifies the corruption and ordinal severity but does
not claim pixel-for-pixel replayability.

## COCO inputs and split

`benchmark-coco` receives a COCO 2017 validation image directory, its
`instances_val2017.json` file, the official RT-DETRv2-R18 checkpoint, and a
benchmark output directory. Detection labels are used only to map COCO image IDs
to filenames; the workflow remains inference-only and reports no mAP or accuracy.

The command sorts COCO image identities, applies a fixed recorded split seed, and
chooses the first 250 identities for reference and the next 250 for evaluation.
It writes canonical CSV manifests plus `benchmark-manifest.json` under the output
directory. The manifest records the annotation digest, image directory, split
seed, requested counts, selected IDs, manifest digests, and checkpoint digest.
Existing manifest checks enforce that reference and evaluation images remain
disjoint.

The command exposes count controls for a later 2,500/2,500 run but defaults to
250/250. Provenance rejects an attempt to reuse a pilot directory with changed
counts, split, or checkpoint.

## Execution and cache layout

The benchmark root owns one shared clean reference cache and one reference bank.
Each corruption owns an independently resumable evaluation cache, score table,
provenance, and report bundle:

```text
benchmark-root/
├── inputs/
│   ├── reference.csv
│   ├── evaluation.csv
│   └── benchmark-manifest.json
├── artifacts/
│   ├── reference-extractions/
│   └── reference-bank.pt
├── corruptions/<corruption-name>/
│   ├── artifacts/evaluation-extractions/
│   ├── artifacts/scores.csv
│   └── report/
└── benchmark-report/
    ├── corruption-metrics.csv
    ├── summary.json
    └── report.md
```

The current pipeline is factored at the clean-reference boundary: it prepares or
resumes the reference artifacts once, then evaluates, scores, reports, and audits
one corruption against those fixed artifacts. Corruption provenance identifies the
shared reference artifacts. Interruptions reuse valid reference work and completed
corruption shards; incompatible output can never be mistaken for another
corruption.

## Scores and statistics

The primary and current controls do not change:

- `persistence_relative_gap` remains the primary differential score;
- `confidence_relative_gap` remains the decile-matched confidence control;
- `persistence_responsive` and `persistence_reference` remain raw controls.

Every score row adds direct model confidence computed across all valid query IDs
after the current union-of-padded-tails filter and before confidence deciles:

- `direct_confidence_mean`: mean maximum-sigmoid class confidence;
- `direct_confidence_max`: maximum maximum-sigmoid class confidence.

Both direct confidence fields use orientation `-1`: lower confidence ranks as
more corrupted. Each receives the current per-severity AUROC, macro-AUROC,
per-image Spearman trend, adjacent-severity consistency, and strongest-level rate.
Paired bootstrap comparison extends the primary score's existing comparisons with
both direct-confidence baseline differences.

## Reporting

Each corruption report shows its exact name, level table, all score metrics,
per-severity AUROCs, macro-AUROC, trends, and paired bootstrap comparisons. Its
tables and figures include the two direct baselines alongside the existing scores.

After all 19 reports have passed their audits, the root-level report produces a
19-row leaderboard with every corruption's macro-AUROC for the primary score,
matched-confidence control, raw persistence controls, direct-confidence mean, and
direct-confidence max. It also gives every method's median and mean macro-AUROC
over corruptions. The report calls this uncertainty/corruption-detection
performance, never mAP, accuracy, or general detector robustness.

## Dependency and CLI surface

Add `imagecorruptions==1.1.2` and its required runtime dependencies. The
`benchmark-coco` command accepts COCO paths, checkpoint, output directory, device,
batch/shard sizes, and reference/evaluation counts. Existing `run` retains its
Gaussian-blur-only behavior and compatibility.

Existing practical provenance boundaries remain: changes to arbitrary corruption
code or hidden settings require a new output directory. Stochastic corruptions
are identified by name and severity rather than by an exact generated-image
fingerprint.

## Tests and acceptance

Test-first implementation covers:

- all 18 adapters' six-level contract, exact level-0 pixels, invalid-level
  rejection, and RGB output shape;
- direct mean/max confidence calculation over valid queries and their
  lower-confidence evaluation orientation;
- report/artifact inclusion of both new score fields and comparisons;
- deterministic, disjoint 250/250 COCO split creation and malformed/missing/
  insufficient/incompatible input rejection;
- one reference extraction/bank for a 19-corruption matrix, resumability, and
  top-level leaderboard reconciliation;
- benchmark CLI defaults, changed-provenance refusal, and unchanged legacy `run`;
- the full retained test suite, detector parity, cache safety, and artifact audits.

Acceptance requires a fresh test-suite pass, a real all-19-corruption COCO
250/250 pilot, and an audited report of primary and direct-baseline metrics. The
2,500/2,500 command must not be run during this work.
