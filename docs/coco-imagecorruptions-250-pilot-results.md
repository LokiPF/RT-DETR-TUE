# COCO ImageCorruptions 250/250 pilot results

## Scope

This pilot evaluates exactly **250 clean COCO-val reference images** and **250 COCO-val evaluation images** across **19 corruption families**. It is explicitly **not** a 2,500/2,500 evaluation. The per-family macro-AUROC measurements are recorded in [corruption-metrics.csv](results/coco-imagecorruptions-250/corruption-metrics.csv), with provenance and aggregates in [summary.json](results/coco-imagecorruptions-250/summary.json).

## Headline comparison

Scores are macro AUROC aggregated over the 19 corruption families (higher is better).

| Method | Mean AUROC | Median AUROC |
| --- | ---: | ---: |
| Persistence relative gap (primary) | 0.660734 | 0.673542 |
| Confidence relative gap | 0.600815 | 0.597216 |
| Direct confidence mean | 0.478847 | 0.488675 |
| Direct confidence max | 0.712078 | 0.742261 |

The primary persistence-relative-gap score ranks second overall: direct confidence max is first, while the primary score is ahead of the matched confidence-relative-gap and direct-confidence-mean baselines. For the direct-confidence baselines, lower raw direct confidence is oriented as more corrupted before calculating AUROC.

## Where the primary score is strongest and weakest

The primary score is `persistence_relative_gap`; direct confidence max is included for context.

| Group | Corruption family | Primary AUROC | Direct-max AUROC |
| --- | --- | ---: | ---: |
| Strongest | zoom blur | 0.843568 | 0.859846 |
| Strongest | gaussian blur | 0.745453 | 0.794773 |
| Strongest | impulse noise | 0.742291 | 0.775893 |
| Strongest | pixelate | 0.738150 | 0.782562 |
| Weaker | fog | 0.532016 | 0.555211 |
| Weaker | elastic transform | 0.538922 | 0.670794 |
| Weaker | brightness | 0.512301 | 0.541214 |
| Weaker | saturate | 0.517248 | 0.543485 |

## Interpretation

The primary signal separates several blur, impulse-noise, and pixelation cases well in this small pilot, but is close to chance for the listed weather, geometric, and photometric cases. Direct confidence max has the strongest aggregate result here; the persistence-relative-gap score nonetheless provides a stronger aggregate separation than the matched confidence-relative-gap and direct-confidence-mean comparisons.

## Limitations and next step

This is a 250/250 pilot, so it does not support an mAP or task-accuracy conclusion. The final 2,500/2,500 pilot has not yet started. ImageCorruptions pixel generation is stochastic, so standalone reruns do not reproduce exactly the same corrupted pixels. The next step is to run that final pilot with fixed, recorded evaluation artifacts and report uncertainty alongside the family-level AUROCs.
