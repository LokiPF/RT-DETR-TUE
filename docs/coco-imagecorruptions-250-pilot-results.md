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

## Per-severity AUROC

Each cell below compares the 250 clean scores with the 250 scores at that one corruption level. Higher is better after the method-specific direction is applied; for direct confidence max, this means lower raw confidence is treated as more corrupted. These are separate comparisons for levels 1 through 5, not a pool across levels.

### Persistence relative gap

| Corruption family | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| gaussian blur | 0.540480 | 0.597520 | 0.700784 | 0.919856 | 0.968624 |
| gaussian noise | 0.549040 | 0.593072 | 0.706080 | 0.858528 | 0.927792 |
| shot noise | 0.542784 | 0.608928 | 0.712208 | 0.846480 | 0.895616 |
| impulse noise | 0.598832 | 0.638816 | 0.687792 | 0.849504 | 0.936512 |
| defocus blur | 0.579888 | 0.597856 | 0.653344 | 0.719856 | 0.804000 |
| glass blur | 0.554224 | 0.608464 | 0.777312 | 0.821776 | 0.855344 |
| motion blur | 0.533536 | 0.598256 | 0.674576 | 0.780560 | 0.848016 |
| zoom blur | 0.735328 | 0.817392 | 0.855664 | 0.905056 | 0.904400 |
| snow | 0.585616 | 0.679264 | 0.693920 | 0.758288 | 0.757088 |
| frost | 0.547632 | 0.582112 | 0.644624 | 0.630976 | 0.680320 |
| fog | 0.529248 | 0.526320 | 0.540976 | 0.526928 | 0.536608 |
| brightness | 0.497088 | 0.507952 | 0.510128 | 0.522976 | 0.523360 |
| contrast | 0.520512 | 0.532320 | 0.552080 | 0.622992 | 0.784304 |
| elastic transform | 0.513936 | 0.531376 | 0.547152 | 0.550512 | 0.551632 |
| pixelate | 0.554640 | 0.611616 | 0.752016 | 0.870976 | 0.901504 |
| jpeg compression | 0.557216 | 0.598960 | 0.635664 | 0.745248 | 0.830624 |
| speckle noise | 0.521824 | 0.550976 | 0.639648 | 0.698160 | 0.768176 |
| spatter | 0.500736 | 0.560272 | 0.577968 | 0.738448 | 0.776496 |
| saturate | 0.506144 | 0.515792 | 0.495184 | 0.522496 | 0.546624 |

### Direct confidence max

| Corruption family | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| gaussian blur | 0.573896 | 0.667720 | 0.793632 | 0.953560 | 0.985056 |
| gaussian noise | 0.577336 | 0.656704 | 0.751896 | 0.868256 | 0.953704 |
| shot noise | 0.582744 | 0.649360 | 0.742592 | 0.873512 | 0.929720 |
| impulse noise | 0.635480 | 0.695872 | 0.734328 | 0.864472 | 0.949312 |
| defocus blur | 0.624272 | 0.670888 | 0.737248 | 0.810624 | 0.868272 |
| glass blur | 0.621720 | 0.708664 | 0.877832 | 0.902408 | 0.925416 |
| motion blur | 0.577728 | 0.642896 | 0.721824 | 0.828128 | 0.889200 |
| zoom blur | 0.770264 | 0.837584 | 0.875800 | 0.902920 | 0.912664 |
| snow | 0.641640 | 0.733872 | 0.743224 | 0.834376 | 0.813000 |
| frost | 0.582576 | 0.663264 | 0.715720 | 0.735312 | 0.760576 |
| fog | 0.532144 | 0.546152 | 0.555760 | 0.562296 | 0.579704 |
| brightness | 0.514608 | 0.528648 | 0.537584 | 0.548704 | 0.576528 |
| contrast | 0.534624 | 0.555560 | 0.594592 | 0.691656 | 0.851256 |
| elastic transform | 0.581816 | 0.625464 | 0.671816 | 0.714432 | 0.760440 |
| pixelate | 0.578784 | 0.636256 | 0.813424 | 0.916488 | 0.967856 |
| jpeg compression | 0.600488 | 0.664976 | 0.708392 | 0.848768 | 0.939400 |
| speckle noise | 0.538736 | 0.585352 | 0.696416 | 0.758112 | 0.822904 |
| spatter | 0.508728 | 0.641968 | 0.685424 | 0.759200 | 0.825480 |
| saturate | 0.510280 | 0.526408 | 0.525904 | 0.550872 | 0.603960 |

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
