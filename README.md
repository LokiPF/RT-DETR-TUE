# Differential corruption uncertainty

## Repository map

This tree shows the active workflow and what each part is responsible for. Historical result
reports and planning documents are grouped so the runtime path stays easy to see.

```text
.
|-- differential_uncertainty/          # End-to-end experiment package
|   |-- __main__.py                     # `python -m differential_uncertainty` entry point
|   |-- cli.py                          # Validates inputs and coordinates run/benchmark resume/audits
|   |-- config.py                       # Fixed scientific settings and score directions
|   |-- benchmark.py                    # Deterministic COCO split and 19-corruption coordinator
|   |-- manifests.py                    # Reads, fingerprints, and separates image manifests
|   |-- extraction.py                   # Runs pretrained RT-DETR and creates query records
|   |-- persistence.py                  # Computes layer-2 topological persistence features
|   |-- bank.py                         # Builds the clean natural-query comparison bank
|   |-- scoring.py                      # Forms confidence groups and relative-gap scores
|   |-- evaluation.py                   # Computes AUROC, Spearman, curve, and bootstrap metrics
|   |-- artifacts.py                    # Validates and atomically stores resumable artifacts
|   |-- reporting.py                    # Writes CSV, JSON, Markdown, and figure outputs
|   |-- corruptions/
|   |   |-- base.py                     # Small corruption interface and Severity record
|   |   |-- gaussian_blur.py            # Legacy six-level Gaussian blur implementation
|   |   `-- imagecorruptions.py         # Eighteen ImageCorruptions adapters
|   `-- __init__.py                     # Package version marker
|-- src/                                # Minimal upstream detector closure kept for inference
|   |-- nn/backbone/
|   |   |-- common.py                   # Frozen batch-normalization layer
|   |   `-- presnet.py                  # RT-DETR ResNet backbone
|   `-- zoo/rtdetr/
|       |-- box_ops.py                  # Bounding-box conversion helpers
|       |-- denoising.py                # Dormant decoder compatibility helper
|       |-- hybrid_encoder.py           # Multi-scale feature encoder
|       |-- rtdetr.py                   # RT-DETR model assembly
|       |-- rtdetrv2_decoder.py         # RT-DETRv2 transformer decoder
|       `-- utils.py                    # Detector tensor and initialization utilities
|-- tests/differential_uncertainty/     # Tests for the retained workflow
|   |-- fixtures/                       # Small frozen detector-parity reference
|   |-- test_benchmark.py               # COCO inputs, benchmark resume, and leaderboard
|   |-- test_artifacts.py               # Atomic storage, cache, and filesystem safety
|   |-- test_bank.py                    # Clean-bank construction
|   |-- test_cli.py                     # Public command-line surface
|   |-- test_config.py                  # Fixed scientific configuration
|   |-- test_corruptions.py             # Corruption interface and Gaussian blur
|   |-- test_detector_parity.py         # Retained detector versus frozen reference
|   |-- test_evaluation.py              # AUROC, Spearman, curve, and bootstrap calculations
|   |-- test_extraction.py              # Image loading, detector inference, and clean level 0
|   |-- test_legacy_parity.py           # New evaluator versus archived blur results
|   |-- test_manifests.py               # Manifest validation and image fingerprints
|   |-- test_persistence.py             # Layer-2 persistence calculation
|   |-- test_pipeline.py                # End-to-end execution, resume, and final audits
|   |-- test_reporting.py               # Report values, figures, and safe publication
|   |-- test_repository_surface.py      # Enforces the intentionally small repository
|   `-- test_scoring.py                 # Deciles, nearest neighbours, and relative gaps
|-- docs/
|   |-- clean-branch-verification.md    # Reproducible verification record for this codebase
|   |-- scene-uncertainty-*.md          # Historical plain-language experiment reports
|   `-- superpowers/                    # Approved design notes and implementation plans
|-- pretrained_weights/                 # Legacy statistics; not used by this clean workflow
|-- requirements.txt                    # Python dependencies
|-- RT-DETR_Topological_Uncertainty_TODO.md # Historical project notes
`-- README.md                            # Setup, usage, outputs, and interpretation
```

This repository runs one pretrained RT-DETRv2-R18 inference experiment. It asks a
simple question: when an image is damaged more strongly, does the detector's
uncertainty score usually rise? The generic workflow accepts its own image manifests;
the fixed COCO benchmark additionally uses COCO annotations to construct its split.
Neither workflow trains a model.

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
8, and 12. This generic `run` command is the legacy Gaussian-blur workflow. The
complete 19-corruption matrix is run by `benchmark-coco` below.

The command safely resumes compatible partial extraction. It refuses to mix changed
image bytes or file identities, manifests, checkpoint bytes, workflow or detector source
code, or runtime regimes in one output directory. A corruption provides only a `name`,
six ordered severities, and `apply(image, level)`. The recorded name and severity
table must still match before a completed cache is reused. Severity 0 must return a
pixel-equivalent clean image. Before any evaluation inference begins, the pipeline
prevalidates that contract for every pending level-0 image.

Resume assumes that the corruption code and its hidden settings are unchanged. The
workflow intentionally does not inspect arbitrary Python dependencies or in-memory
state, so it cannot detect every code-only or settings change. If either changes, use a
new output folder rather than resuming the old one. This is a practical resume boundary,
not a claim of strict corruption-implementation identity. Output folders created before
this practical contract should not be reused after the code change; start a new output
directory. The built-in Gaussian blur needs no extra configuration.

The runtime regime includes the resolved device and index, batch and shard sizes,
Python and library versions, and CUDA/cuDNN/GPU details when CUDA is used. Therefore a
partial run created with one batch size cannot be resumed with another batch size.

## COCO ImageCorruptions 250/250 pilot

The current COCO command is a **250-reference / 250-evaluation-image pilot**. It is
explicitly **not** the final 2,500/2,500 experiment; do not add
`--reference-count 2500` or `--evaluation-count 2500` to this run.

```bash
python -m differential_uncertainty benchmark-coco --coco-annotations /home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json --coco-images /home/yuchen/YuchenZ/Datasets/coco/val2017 --checkpoint /home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth --output-dir runs/coco-imagecorruptions-250 --device cuda:0 --batch-size 1 --shard-size 50
```

The command selects a deterministic, disjoint COCO-val split and runs a matrix of the
legacy Gaussian blur plus 18 routines from ImageCorruptions. Every routine has clean
level 0 and package severities 1--5. Some package routines are stochastic: a corrupted
image is identified by its corruption name and severity, but the workflow does not
promise identical corrupted pixels on a later rerun.

The pilot prepares one shared clean-reference cache and fingerprint bank, then stores
each corruption independently under `<output-dir>/corruptions/<corruption-name>/` so a
compatible interruption can resume per corruption. The deterministic input manifests
are under `<output-dir>/inputs/`, the shared reference artifacts are under
`<output-dir>/reference-artifacts/`, and the reconciled leaderboard is under
`<output-dir>/benchmark-report/`. That leaderboard contains `corruption-metrics.csv`,
`summary.json`, and `report.md`.

For every corruption, the leaderboard reports macro-AUROC: the average of separate
clean-versus-severity-1 through -5 AUROCs. Its primary method is the persistence
relative gap. The matched-confidence relative gap is a control on the same
decile-selected queries, while direct global confidence mean and maximum are baselines
over all valid detector queries; lower raw direct confidence is treated as more
corrupted before ranking. Compare the primary method with both the matched-confidence
control and the direct mean/maximum confidence baselines to determine whether any gain
is specific to the fingerprint score rather than ordinary detector confidence.

This benchmark evaluates how well each score ranks corrupted images ahead of clean
images. It does not measure detector mAP, box quality, or detector accuracy.

## Outputs

For the generic Gaussian-blur command, `<output-dir>/artifacts/` contains provenance,
raw feature caches, the clean reference bank, and per-image scores.
`<output-dir>/report/` contains one plain-language Markdown report, machine-readable
metric tables, a JSON summary, and four detailed figures. The provenance and report
record the runtime, corruption name, and ordered severity table. Re-run the same command
after interruption; completed compatible stages are reused when the recorded experiment
settings match. The COCO benchmark layout and resume behavior are described above.

## What the score means

For each image, the method ranks valid detector queries by confidence. It compares the mean
layer-2 bank distance in the middle 50--60% confidence group with the mean distance in the
highest 90--100% confidence group:

```text
relative gap = 2 * (responsive mean - reference mean)
                   / (responsive mean + reference mean)
```

The primary score uses distance from the clean reference bank. The matched-confidence
control uses `1 - max(sigmoid(class logits))` on the same selected queries. Direct
confidence baselines use the mean or maximum of `max(sigmoid(class logits))` over all
valid, non-padded queries, with lower confidence oriented as more corrupted. AUROC asks
how often a randomly chosen corrupted image ranks above a randomly chosen clean image
after the score's fixed direction is applied; it is a ranking measure, not a probability.
Spearman correlation and adjacent-severity consistency check whether scores move as
corruption becomes stronger. This workflow does not calculate detector accuracy or mAP.
