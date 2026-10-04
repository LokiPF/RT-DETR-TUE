# Corruption detection inside a frozen object detector

This repository detects image corruption (fog, blur, noise, compression and more) from inside a frozen
RT-DETRv2-R18 trained on COCO. Nothing is trained for it. The repository holds our method and the six published
baselines we compare it with. All of them are evaluated on COCO val2017 under 19 imagecorruptions families at five
severities.

**Our method, the two-axis score.**
- **Two numbers per channel.** In the backbone's stages 1–3, each channel gives its level (mean |activation|) and its
  peak share (how far its strongest 1% of positions stand out).
- **A reference of similar scenes.** Both numbers are compared with the 50 clean COCO train scenes most like the image,
  found by the stage-4 channel means.
- **Flatten or shift.** A corruption either flattens the peaks (fog, contrast, blur) or shifts the level (noise). The
  score is the larger of the two deviations.

On all 5,000 COCO val images it reaches AUROC 0.917 on the 15 common families and 0.858 on the 4 extra ones. The
strongest baseline, the activation CDFs of Becker et al. (ICPR 2026), reaches 0.821 and 0.807. The details are in
`docs/conv-tu-conditioned-results.md`.

## Layout

```
degradation_monitor/
  cli.py, settings.py, runs.py, corruptions.py
  datasets/coco.py         COCO: train images, the seed-44 val order and folds, ground truth
  detector/                the frozen RT-DETRv2-R18: vendored code (rtdetrv2/), loader, post-processing, hooks
  detectors/               one adapter per further detector: YOLO11m, Faster R-CNN, RF-DETR-M
  baselines/               SAOD, ContrastiveConf, kNN, DisCoPatch, Hashemi et al., activation CDFs: one file each
  method/                  our method: channel statistics, the clean reference, the scores and the ablation rows
  evaluation/              separation metrics, the paired bootstrap and the report
  stages/                  the runnable, resumable steps
configs/coco.toml          this machine's paths and run options
configs/coco-detectors.toml  the further detectors' weights, run root and clean-AP floors
scripts/convert_runs.py    the one-time conversion of the old run folder
archive/                   retired methods, read only (archive/README.md)
docs/                      results, decisions and the dev log (docs/README.md)
tests/                     mirrors degradation_monitor/
```

## Setup

- **Python packages:** Python 3.11 and the packages in `requirements.txt`. Install a PyTorch build for your machine
  first.
- **The detector checkpoint:** RT-DETRv2-R18 trained on COCO, `rtdetrv2_r18vd_120e_coco` (48.1 AP). The vendored model
  code comes from github.com/lyuwenyu/RT-DETR (Apache-2.0).
- **COCO 2017:** `train2017`, `val2017` and `annotations/instances_val2017.json`.
- **DisCoPatch** (Caetano et al., ICCV 2025): a clone of github.com/caetas/DisCoPatch, with its own requirements. Its
  commit is recorded in each run's `manifest.json`.

Then edit the paths in `configs/coco.toml`.

## Running

The four passes (`detector-pass`, `discopatch-pass`, `activation-pass` and `method-pass`) resume image by image. The
bank and the fits write their result once and are skipped when it exists; DisCoPatch's training refuses to overwrite a
finished discriminator. An interrupted fit or training starts again. Every stage refuses a run folder made with another
protocol; the later passes refuse corrupted images that differ from the detector pass's, and the passes refuse fitted
references that changed since their results were written.

Use one run folder per protocol — a limited smoke run gets its own folder with `--run DIR` — and run one stage at a time
per run folder, because the stages share its manifest.

```bash
python -m degradation_monitor <stage> [--config configs/coco.toml] [--device cuda:0] [--limit N]
                                      [--batch-size N] [--workers N] [--gpu-memory-gib X] [--epochs N] [--run DIR]
```

| Stage | What it computes | Needs |
|---|---|---|
| `check` | that every input exists, and the clean COCO val AP (about 0.48) | – |
| `knn-bank` | the kNN baseline's bank: pooled last-stage features of every COCO train image | – |
| `detector-pass` | for every val image under all 96 conditions: SAOD, ContrastiveConf's parts, kNN distances, detections | `knn-bank` |
| `discopatch-train` | DisCoPatch's discriminator, with the official code and settings | – |
| `discopatch-pass` | DisCoPatch's scores | `detector-pass`, `discopatch-train` |
| `hashemi-fit`, `cdf-fit`, `cdf-zstats` | the activation monitors' clean statistics | `cdf-fit` before `cdf-zstats` |
| `activation-pass` | the scores of Hashemi et al. and of the activation CDFs | `detector-pass` and the three fits |
| `method-reference` | our method's clean reference: 2,000 bank and 500 z-statistics train images | – |
| `method-pass` | our method's channel statistics | `detector-pass` |
| `report` | every table, both pre-registered decisions, per-condition mAP | `detector-pass`; the others when present |
| `timing` | milliseconds per image for the detector and each baseline; our method's added cost is not timed yet | `knn-bank` |

Run the tests with `python -m pytest -q`; pytest is not in `requirements.txt`.

### Three more detectors

`python -m degradation_monitor.stages.detectors <check|fit|pass|report> --config configs/coco-detectors.toml` runs
YOLO11m, Faster R-CNN R50-FPN v2 and RF-DETR-M on the same COCO-C images as RT-DETR. `fit` builds each detector's
clean references from COCO train images in three resumable passes. `pass` generates each val image's 96 versions
once, checks them against `runs/coco/`'s digests and feeds all three detectors; `--first N` stops after N images.
`report` writes each detector's report and `runs/coco-detectors/summary.md`, the table of all four; `--first N` is a
smoke report on the first N images. Each detector reads the method's three earliest feature levels and its deepest as
the key: stages for the CNNs, blocks 1-3 and 12 for RF-DETR's ViT. ContrastiveConf and Hashemi et al. exist for the
DETR-type detectors only.

## Results

- **The run folder** (`run` in the config):
  - `manifest.json`: the protocol, the environment and the inputs of every score folder;
  - `reference/`: what the clean train images provide;
  - `scores/`: one file per val image for each pass;
  - `reports/coco/`: the report.
- **Hard links:** the converted `runs/coco/` shares its per-image files and fitted references with
  `runs/coco-baselines/` through hard links, so an in-place edit of either copy changes both. The stages are safe,
  because they write a new file and rename it.
- **The tables kept in the repository:** `docs/results/coco/`. The documents that explain them are listed in
  `docs/README.md`.
