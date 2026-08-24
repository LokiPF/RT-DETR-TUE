# Differential corruption uncertainty

This repository runs one pretrained RT-DETRv2-R18 experiment. It asks a simple
question: when an image is damaged more strongly, does the detector's uncertainty
score usually rise? The workflow performs inference only. It does not train a model
and it does not require COCO annotation files.

## Install

Create a Python environment with a suitable PyTorch build, then run:

```bash
python -m pip install -r requirements.txt
```

## Inputs

Both the clean reference set and evaluation set use the same CSV shape:

```csv
image_id,image_path
street_001,/data/images/street_001.jpg
street_002,/data/images/street_002.jpg
```

Image IDs must be unique inside each manifest. Paths may be absolute or relative to
the manifest. The checkpoint must be the fixed RT-DETRv2-R18 COCO checkpoint used by
this repository.

## Run or resume everything

```bash
python -m differential_uncertainty run \
  --reference-manifest reference.csv \
  --evaluation-manifest evaluation.csv \
  --checkpoint rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \
  --output-dir runs/blur-study
```

Only runtime controls are exposed: `--device`, `--batch-size`, and `--shard-size`.
Scientific choices are fixed in code: layer 2, 335 persistence features, a seeded
25,000-vector clean reference bank, 5-nearest-neighbour distance, responsive deciles
50--60, reference deciles 90--100, relative gap, and Gaussian blur radii 0, 1, 2, 4,
8, and 12.

The command safely resumes compatible partial extraction. It refuses to mix changed
manifests, checkpoint bytes, or source code into an existing run directory.

## Outputs

`<output-dir>/artifacts/` contains provenance, raw feature caches, the clean reference
bank, and per-image scores. `<output-dir>/report/` contains one plain-language Markdown
report, machine-readable metric tables, a JSON summary, and four detailed figures. Re-run
the same command after interruption; completed compatible stages are reused.

## What the score means

For each image, the method ranks valid detector queries by confidence. It compares the mean
layer-2 bank distance in the middle 50--60% confidence group with the mean distance in the
highest 90--100% confidence group:

```text
relative gap = 2 * (responsive mean - reference mean)
                   / (responsive mean + reference mean)
```

The primary score uses distance from the clean reference bank. The matched-confidence
baseline uses `1 - max(sigmoid(class logits))` on the same selected queries. AUROC asks
how often a randomly chosen corrupted image ranks above a randomly chosen clean image
after the score's fixed direction is applied; it is a ranking measure, not a probability.
Spearman correlation and adjacent-severity consistency check whether scores move as
corruption becomes stronger. This workflow does not calculate detector accuracy or mAP
because the generic manifests have no ground-truth boxes.
