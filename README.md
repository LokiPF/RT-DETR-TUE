# Differential corruption uncertainty

This repository runs one pretrained RT-DETRv2-R18 experiment. It asks a simple
question: when an image is damaged more strongly, does the detector's uncertainty
score usually rise? The workflow performs inference only. It does not train a model
and it does not require COCO annotation files.

## Requirements and install

Use Python 3.10 or newer. The workflow currently requires Linux with `/proc`
mounted and `renameat2` support; those operating-system features let it pin output
directories safely and publish a complete report without replacing an existing one.

The workflow was verified here with Python 3.11.15, PyTorch 2.11.0+cu128,
torchvision 0.26.0+cu128, CUDA 12.8, cuDNN 9.19.0, and an NVIDIA GeForce RTX
5090. These are tested versions, not the only suitable versions. Install a PyTorch
build suitable for your computer, then run:

```bash
python -m pip install -r requirements.txt
```

## Checkpoint

The model comes from the [official RT-DETR repository](https://github.com/lyuwenyu/RT-DETR).
Use its [official RT-DETRv2-R18 COCO checkpoint](https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth).
The upstream [`hubconf.py`](https://github.com/lyuwenyu/RT-DETR/blob/main/hubconf.py)
maps `rtdetrv2_r18vd` to that release file.

The expected filename is `rtdetrv2_r18vd_120e_coco_rerun_48.1.pth`. Its expected
SHA-256 is `2ace52184b620204004509b72752ac7bfe64aadaf7fc1d076b18df8ab5a5c77e`.
From the directory containing the downloaded file, verify both the bytes and the
filename before running:

```bash
printf '%s  %s\n' \
  '2ace52184b620204004509b72752ac7bfe64aadaf7fc1d076b18df8ab5a5c77e' \
  'rtdetrv2_r18vd_120e_coco_rerun_48.1.pth' | sha256sum --check --strict
```

A successful check prints `rtdetrv2_r18vd_120e_coco_rerun_48.1.pth: OK`.
The run command records the SHA-256 of whichever checkpoint file you supply, but
it does not enforce this official identity for a new run. The command uses the
recorded digest to stop a resumed run from mixing different checkpoint bytes.

## Inputs

Both the clean reference set and evaluation set use the same CSV shape:

```csv
image_id,image_path
street_001,/data/images/street_001.jpg
street_002,/data/images/street_002.jpg
```

Each CSV must follow these rules:

- The header must be exactly `image_id,image_path`, in that order, with no extra
  columns.
- The manifest must contain at least one image.
- After surrounding spaces are removed, every image ID must be nonempty and
  unique within that manifest. An ID may contain at most 256 characters, may
  not contain any Unicode category-C character (such as a control or format
  character), and may not begin with `=`, `+`, `-`, or `@`.
- Every image path must resolve to an existing file. Relative paths are resolved
  from the directory containing the manifest, not from the current terminal
  directory.
- Two rows in one manifest may not resolve to the same image file, including
  through different hard-link paths to the same file identity.
- The reference and evaluation manifests must be separate sets: they may not
  share an image ID, a resolved image path, or hard links to the same file.
- Preflight records each canonical path, content SHA-256, device/inode identity,
  size, mode, and modification/change timestamps. Extraction reopens and fully
  loads that exact fingerprinted file; concurrent replacement or mutation is
  refused.

## Run or resume everything

```bash
python -m differential_uncertainty run \
  --reference-manifest reference.csv \
  --evaluation-manifest evaluation.csv \
  --checkpoint rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \
  --output-dir runs/blur-study
```

The runtime defaults are `--device cuda:0`, `--batch-size 1`, and
`--shard-size 50`. These are the only optional runtime controls. To run without
CUDA, use the same command with the CPU control:

```bash
python -m differential_uncertainty run \
  --reference-manifest reference.csv \
  --evaluation-manifest evaluation.csv \
  --checkpoint rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \
  --output-dir runs/blur-study-cpu \
  --device cpu
```

CPU inference is supported but will normally be much slower than GPU inference.
Scientific choices are fixed in code: layer 2, 335 persistence features, a seeded
25,000-vector clean reference bank, 5-nearest-neighbour distance, responsive deciles
50--60, reference deciles 90--100, relative gap, and Gaussian blur radii 0, 1, 2, 4,
8, and 12.

The command safely resumes compatible partial extraction. It refuses to mix changed
image bytes or file identities, manifests, checkpoint bytes, source code, corruption
plugin implementation, or runtime regime into an existing run directory. The runtime
regime includes the resolved device and index, batch and shard sizes, Python and library
versions, and CUDA/cuDNN/GPU details when CUDA is used. Therefore a partial run created
with one batch size cannot be resumed with another batch size.

## Outputs

`<output-dir>/artifacts/` contains provenance, raw feature caches, the clean reference
bank, and per-image scores. `<output-dir>/report/` contains one plain-language Markdown
report, machine-readable metric tables, a JSON summary, and four detailed figures. The
provenance and report record the runtime and corruption implementation identity. Re-run
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
