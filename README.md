# Fixed COCO corruption benchmark

This repository runs one inference-only experiment with a pretrained RT-DETRv2-R18
checkpoint. It builds a fingerprint bank from clean COCO training images, then tests
whether three image-level scores rank corrupted COCO validation images above their
clean versions. Nothing is trained.

The evaluation covers all 19 configured corruption families at severities 4 and 5:
Gaussian blur, Gaussian noise, shot noise, impulse noise, defocus blur, glass blur,
motion blur, zoom blur, snow, frost, fog, brightness, contrast, elastic transform,
pixelate, JPEG compression, speckle noise, spatter, and saturate. Levels 1 through 3
are not evaluated.

## Method

The fixed fingerprint method is:

1. On the selected clean training images, keep non-padded detector queries whose
   maximum sigmoid confidence is at least 0.5.
2. Build a 2,000-row bank with seeded Algorithm R reservoir sampling.
3. For each validation query, take the mean cosine distance to its five nearest bank
   rows.
4. Form the image fingerprint score as the confidence-weighted mean across the
   non-padded scene queries.

The two fixed baselines are one minus the maximum sigmoid confidence and normalized
Shannon entropy of the highest-confidence non-padded query.

## Install and inputs

Use Python 3.10 or newer. Install a PyTorch build appropriate for your machine, then
install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Provide a compatible pretrained RT-DETRv2-R18 checkpoint. The official RT-DETR
project publishes the `rtdetrv2_r18vd_120e_coco_rerun_48.1.pth` COCO checkpoint.

The image inputs are flat directories, normally COCO `train2017` and `val2017`.
Images must be directly inside those directories. The command does not use COCO
annotations or image manifests.

## Run or resume

```bash
python -m differential_uncertainty benchmark-coco \
  --checkpoint /path/to/rtdetrv2_checkpoint.pth \
  --coco-train-images /path/to/coco/train2017 \
  --coco-val-images /path/to/coco/val2017 \
  --reference-count 2500 \
  --evaluation-count 2500 \
  --output runs/fingerprint-coco-2500 \
  --device cuda:0 \
  --batch-size 4 \
  --seed 44
```

The value 2,500 is an example, not a default. Both `--reference-count` and
`--evaluation-count` are required, so choose them explicitly for each experiment.
The optional controls default to `--device cuda:0`, `--batch-size 1`, and `--seed 44`.

Selection is repeatable for a given seed. To resume an interruption, run the same
command with the same output directory. The working files are deliberately simple:

- `run_config.json` records the small fixed run configuration.
- `bank_progress.pt` holds partial reservoir state and is replaced by `bank.pt` when
  bank construction finishes.
- `scores/` contains one JSON score file per completed validation image.
- `evaluation_progress.pt` records the next image and NumPy random state needed to
  continue stochastic corruptions.

## Results

A completed run writes five final result files in the output directory:

- `per_image_scores.csv` contains clean, level-4, and level-5 scores.
- `results.csv` contains every per-corruption AUROC at levels 4 and 5.
- `summary.json` contains the same task results, aggregate values, and comparisons.
- `corruption_auroc_bars.png` plots all per-corruption results.
- `report.md` presents every corruption separately and ends with the bar chart.

The aggregate is the equal-weight mean across all 38 corruption-and-severity tasks.
Paired whole-image bootstrap comparisons describe the fingerprint macro-AUROC
difference from each baseline on this evaluation set. They are descriptive stability
intervals, not population confidence intervals.

This benchmark detects and ranks image corruption. It does not evaluate detector box
accuracy, mAP, or task performance.
