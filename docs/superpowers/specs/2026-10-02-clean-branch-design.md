# Clean branch: design

Agreed with the user on 2 October 2026, in a brainstorming session (questions 1–7 and design parts 1–5).

## Purpose

After two months of development, the codebase mixes four generations of work:
- the decoder-query fingerprint method of August;
- the conv-TU pilot;
- the six COCO baselines;
- the current method: the two-axis "flatten or shift" score of `docs/dev-log.md`, 2026-10-01 (night).

The user wants a new branch `clean` that keeps **only the six baselines and the current method**, in a neat folder organization. All baselines go under one `baselines/` folder.

## Success criteria

1. **Only maintained code remains** in the package. The ablation rows are those listed under `method/scores.py` below.
   - the six baselines (SAOD, ContrastiveConf, kNN, DisCoPatch, Hashemi et al., the activation CDFs of Becker et al.);
   - the current method and its ablation rows;
   - the shared parts they need.
2. **Earlier work is archived.** The conv-TU pilot and the decoder-query fingerprint method sit in `archive/` as a read-only snapshot.
3. **Refactoring only:** no computation changes. From the converted run folder, the clean branch reproduces:
   - the six baselines' per-condition AUROC, AUPR and FPR95 in `docs/results/coco-baselines/separation.csv`, to within about 1e-12;
   - the confirmation report that the old code writes from the same stored files (its `summary.json`), to within float rounding.
4. **Every maintained module is tested.** The suite is green at every commit, and a fresh reviewer checks the whole branch at the end.
5. **No GPU re-run is needed.** The GPU is booked by another session for about two days.

## Decisions

| # | Question | Decision |
|---|---|---|
| 1 | What counts as "our current method" | The two-axis score and its ablation rows are maintained code. The conv-TU pilot and the decoder-query fingerprint method move to an untested `archive/`. |
| 2 | Metrics | Detection metrics only: AUROC, AUPR and FPR95 per condition, severity and family, with paired bootstrap intervals, plus each condition's mAP as context. LRP, ρ within and AURC are removed. Per-image AP stays only because ContrastiveConf fits its λ on it. |
| 3 | Stored outputs | A tidy run-folder layout, built from today's `runs/coco-baselines/` by a one-time, verified conversion. |
| 4 | Package name | `degradation_monitor` (replaces `differential_uncertainty`). |
| 5 | Docs | Paper-relevant docs stay in a flat `docs/` with an index; old ones move to `docs/archive/`. |
| 6 | The code archive | A frozen snapshot, copied as it was, never run or maintained. Its README gives the `fingerprint_bank` commit where it last ran with passing tests. |
| 7 | Datasets | A dataset-shaped layout with COCO implemented, so a Cityscapes module can be added later without touching the baselines or the method. No Cityscapes code now. |
| – | Approach | Move with `git mv` what is already clean, and rewrite what grew tangled (the phase files, the report builders, the CLI). |

## Package layout

```
degradation_monitor/
  cli.py                  python -m degradation_monitor <stage> --config configs/coco.toml [options]
  settings.py             paths and run options read from the config; the method's fixed choices are NOT here
  runs.py                 run-folder layout, atomic writes, resume checks, the manifest
  corruptions.py          19 families × 5 severities, seeded variants, digests
  datasets/
    coco.py               clean train splits, the seed-44 val order, folds, ground truth
                          (per-image AP for ContrastiveConf, mAP per condition as context)
  detector/
    rtdetrv2/             vendored model code (today's src/; internal imports are already relative)
    model.py              build RT-DETRv2-R18, load the frozen checkpoint, preprocessing
    taps.py               hooks: detector outputs, decoder queries, encoder maps, backbone stages, early-channel inputs
  baselines/
    saod.py  contrastive_conf.py  knn.py  discopatch.py  hashemi.py  activation_cdf.py
  method/
    statistics.py         per-channel level and top-1% mean at the four taps
    reference.py          clean bank, z-statistics, the stage-4 content key and its neighbours
    scores.py             level, peak share, two-axis, and the ablation rows: the level against the average of all
                          clean images (stages 1–3), kNN on channel means (4 stages, the pilot's control), and
                          channel means against their own clean average (4 stages, the Neural Mean Discrepancy-style row)
  evaluation/
    metrics.py            AUROC, AUPR, FPR95, per-condition AUROCs, paired bootstrap, stage z-scoring
    report.py             one report for all methods: tables, intervals, both decisions, markdown
  stages/                 the runnable, resumable steps
    baselines.py  method.py  report.py
configs/coco.toml         the machine's paths and run options
archive/                  conv_tu/, decoder_fingerprint/, README.md
scripts/convert_runs.py   the one-time conversion
tests/                    mirrors degradation_monitor/
```

**Where today's code goes:**
- **Baselines:** each of the six baselines gets its own file, out of `baselines/scores.py`, `metrics.py` and `detector.py`. ContrastiveConf's λ fitting goes to `contrastive_conf.py`.
- **`protocol.py`:** the corruption parts go to `corruptions.py`, and the COCO order and folds go to `datasets/coco.py`.
- **`extraction.py`:** it keeps only its loader and preprocessing, as `detector/model.py`.
- **`convtu/tap.py`:** it becomes the early-channel hooks in `detector/taps.py`, without the folded kernels that only topology needed.
- **The method:** `convtu/channels.py` and `conditioned.py` become `method/`, and `confirmation.py` becomes part of `evaluation/report.py`.
- **Rewritten:** `baselines/pipeline.py` and `convtu/pipeline.py` become `stages/`. `baselines/report.py`, `convtu/report.py`'s shared parts and `confirmation.py` become `evaluation/report.py`. The CLI is rewritten too.
- **Removed:** LRP, ρ within and AURC (decision 2); the 4 × 4-grid and 99th-percentile channel statistics, which no planned ablation uses.
- **Archived:**
  - the conv-TU code (`graph.py`, the topology parts of `features.py`, `convtu/report.py`, all of `convtu/pipeline.py`);
  - the decoder-query fingerprint method (`benchmark.py`, `bank.py`, `scoring.py`, `persistence.py`, `reporting.py`, `config.py`, `evaluation.py`, the old `extraction.py` and `corruptions/`);
  - `pretrained_weights/` (the Fréchet means of the old method);
  - `RT-DETR_Topological_Uncertainty_TODO.md`.

Pure computation (`baselines/`, `method/`, `evaluation/`) is kept apart from running over the dataset (`stages/`). Each scorer is testable on small arrays without the detector.

## Stages and command line

`configs/coco.toml` holds:
- the checkpoint and the COCO train, val and annotation paths;
- the DisCoPatch code root and the run folder;
- batch size, workers and the GPU memory cap.

The method's fixed choices stay constants in `method/`, so a config edit cannot change them: k = 50, the stage-4 key, stages 1–3, the top 1%, and the 2,000 bank and 500 z-statistics images. The later ablations pass other values explicitly.

Every stage is resumable per image. It refuses inputs whose corruption digests or fitted references changed.

| Stage | Was | Does |
|---|---|---|
| `check` | sanity | paths, checkpoint hash, data present |
| `knn-bank` | bank | layer-4 features of the 118,287 clean train images |
| `detector-pass` | test | per val image × 96 conditions: confidences, ContrastiveConf parts, kNN distances, detections, digests |
| `discopatch-train`, `discopatch-pass` | same names | train on clean train patches, then score |
| `hashemi-fit`, `cdf-fit`, `cdf-zstats` | same names | clean-train statistics for the activation monitors |
| `activation-pass` | activation-scores | Hashemi and CDF scores per image |
| `method-reference` | convtu-channels, clean part | level and top-1% mean of the 2,000 bank and 500 z-statistics images |
| `method-pass` | convtu-means | level and top-1% mean per val image × 96 conditions |
| `report` | report + convtu-conditioned-report | separation of every method on four image sets, with intervals, both decisions, per-condition mAP and ContrastiveConf's cross-fitted λ |
| `timing` | timing | milliseconds per image for the detector and each monitor |

The four image sets of the `report` stage:
- all 5,000 images;
- the 3,030 untouched images;
- the pre-registered held-out 4,800 images;
- the 200 screen images.

The ablations of `docs/todo.md` are not built now. The method's functions take the bank, k and the reference as arguments, so each ablation can later be a thin stage.

## Run folder and conversion

```
runs/coco/
  manifest.json        checkpoint hash, protocol (seed 44, 96 conditions, 5,000 images), every reference's sha1,
                       and the conversion record
  reference/
    knn/               bank.npy, image_names.json                 (from bank/)
    activation_cdf/    reference.npz, fit.json, zstats.json       (from cdf/)
    hashemi/           intervals.npz, fit.json                    (from hashemi/)
    discopatch/        discriminator.pt, training.json            (only the scored checkpoint)
    method/            bank.npz, zstats.npz                       (level and top-1% mean only)
  scores/
    detector/<image>.npz      (from test/)
    activations/<image>.npz   (from test_activation/)
    discopatch/<image>.npz    (from test_dcp/)
    method/<image>.npz        (from test_convtu_means/)
  reports/coco/        written by the report stage
  logs/
```

`scripts/convert_runs.py`:
- **Hard links:** it hard-links every per-image file. The array names stay, there is no extra disk and the originals are untouched.
- **Rewrites:**
  - the method's bank and z-statistics files, dropping the grid and 99th-percentile arrays;
  - the scattered markers (`fits.json`, `checkpoint.json`), folded into `manifest.json` with sha1 hashes and paths relative to the run folder.
- **Checks:** it verifies that every rewritten array equals its source exactly, and that every linked file keeps its sha1.
- **Refusals:** it refuses to run if `runs/coco/` exists, or if the method pass is incomplete.

`runs/coco-baselines/` stays untouched. It still holds the archive's data, the old harm reports and DisCoPatch's per-epoch checkpoints, and only the user deletes it.

## Archive and docs

**`archive/`** is a read-only snapshot, copied exactly as it was:
- `conv_tu/` (code and tests);
- `decoder_fingerprint/` (code, tests, the `.pth` files);
- `README.md`: what each part was, the result that retired it, and the last commit where it ran.

**`docs/`** is flat, with a `docs/README.md` index.
- **It keeps:**
  - the dev log and the to-do list;
  - the literature review and the decision record;
  - the COCO baseline numbers;
  - the conv-TU pilot design and results, with a note that the code is in `archive/conv_tu/`;
  - the detection roundtable record and the confirmation plan;
  - this spec and its plan;
  - the result tables these cite.
- **Dated notes, nothing rewritten:**
  - the decision record: since 1 October, the goal is detecting corruption, and harm is not an objective;
  - the baseline-numbers doc: its harm sections are superseded, and the clean report reproduces its separation numbers.
- **`docs/archive/` gets:**
  - the August scene-uncertainty and fingerprint reports;
  - the meeting notes and the old branch verification;
  - the old TU to-do file and the fingerprint chart;
  - the 250-image pilot results;
  - the plans for finished or archived work.

The top-level `README.md` is rewritten: what the project is, the layout, how to run each stage, and where results are.

## Testing and equivalence

- **Unit tests:** every maintained module keeps tests ported from today's suite.
  - Stages are tested on the fake backbone and fake detector.
  - The conversion script is tested on a small fake old-style run folder.
  - The real-checkpoint tests skip when the checkpoint is absent.
- **Equivalence tests on the real data** (they skip when the run folder is absent):
  1. The clean report on `runs/coco/` reproduces every per-condition AUROC, AUPR and FPR95 of the six baselines in `docs/results/coco-baselines/separation.csv`.
  2. The clean report reproduces the old code's confirmation output. That output is run on `fingerprint_bank` once the method pass finishes, and its `summary.json` is stored as the reference.

## Process

- **Ordering:** first, the confirmation report runs on `fingerprint_bank`, and its results are committed there and reported to the user. The branch `clean` is cut from `fingerprint_bank` after that, in a new worktree `.worktrees/clean`.
- **Implementation:** it follows a written plan (superpowers:writing-plans), test first, with the suite green at every commit and one fresh whole-branch review at the end.
- **Dependencies:** `requirements.txt` lists exactly what the maintained code imports: torch, torchvision, numpy, pillow, scipy, scikit-learn, pycocotools, uq-detr and imagecorruptions 1.1.2.
  - It drops numba and matplotlib, which only the archive uses.
  - It adds scikit-learn, pycocotools and uq-detr. The baselines import them, but today's file does not list them.
- **Never without asking the user:** merging, pushing, or deleting `runs/coco-baselines/`.

## Out of scope

- New experiments: the ablations, the driving data and the second backbone, all listed in `docs/todo.md`.
- Any change to a computation.
- Any GPU run.
