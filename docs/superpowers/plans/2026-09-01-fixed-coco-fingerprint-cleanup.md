# Fixed COCO Fingerprint Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the research framework with one restartable experiment that builds the selected fingerprint bank on COCO train and evaluates corruption detection on COCO val.

**Architecture:** Process images directly through the retained RT-DETR persistence extractor. Keep one confidence-filtered reservoir bank, one cosine/confidence-weighted fingerprint score, confidence and entropy baselines, simple progress files, and direct report output.

**Tech Stack:** Python, PyTorch, torchvision, Pillow, NumPy, SciPy, imagecorruptions, matplotlib, pytest.

---

## Boundaries

Retain these focused modules:

- `config.py`: fixed constants.
- `persistence.py`: layer-2 persistence calculation.
- `extraction.py`: detector loading and query extraction.
- `corruptions/`: the fixed 19 corruption functions.
- `bank.py`: Algorithm R reservoir.
- `scoring.py`: the three fixed image scores.
- `evaluation.py`: per-task AUROC and paired macro intervals.
- `reporting.py`: CSV, JSON, Markdown, and PNG output.
- `benchmark.py`: train-bank/val-evaluation orchestration and restart.
- `cli.py`: one `benchmark-coco` command.

Delete the generic artifact store, manifests, experimental study engine, legacy `run`, k-means, policy searches, confidence-conditioned analysis, exact-platform fixture, provenance, locks, and transactional bundles.

The current untracked meeting files, analysis scripts, and historical result documents are user files. Leave them untouched and unstaged. The dirty tracked k-means/provenance edits are inside files this approved rewrite replaces.

Use `/home/yuchen/miniconda3/envs/UE/bin/python -m pytest` for all tests.

### Task 1: Replace the scientific core with the selected method

**Files:**
- Replace: `differential_uncertainty/config.py`
- Replace: `differential_uncertainty/corruptions/__init__.py`
- Simplify: `differential_uncertainty/corruptions/imagecorruptions.py`
- Replace: `differential_uncertainty/extraction.py`
- Keep: `differential_uncertainty/persistence.py`
- Replace: `differential_uncertainty/bank.py`
- Replace: `differential_uncertainty/scoring.py`
- Replace: `tests/differential_uncertainty/test_config.py`
- Replace: `tests/differential_uncertainty/test_corruptions.py`
- Replace: `tests/differential_uncertainty/test_extraction.py`
- Keep and simplify: `tests/differential_uncertainty/test_persistence.py`
- Replace: `tests/differential_uncertainty/test_bank.py`
- Replace: `tests/differential_uncertainty/test_scoring.py`

- [ ] **Step 1: Write focused failing tests**

Test these exact behaviors:

- Configuration contains capacity 2,000, confidence threshold 0.5, five neighbors, layer 2, and levels `(4, 5)`; there are no decile, k-means, orientation, or seed-sweep fields.
- The corruption roster is exactly the 19 approved names. Gaussian blur uses radii 8 and 12 at levels 4 and 5; every other family delegates to imagecorruptions; any other level is rejected.
- Extraction returns only `logits`, `persistence`, and `padded_ids`. Padding is decided from boxes, logits, and persistence before boxes are discarded.
- Algorithm R produces the same bank when restored halfway from `state_dict`.
- Bank rows exclude padded queries and queries below confidence 0.5.
- Scoring uses the union padding mask, mean distance to five nearest cosine neighbors, confidence-weighted query aggregation, `1 - max confidence`, and normalized top-query entropy.

The reservoir-resume test must use explicit vectors:

```python
vectors = torch.arange(36, dtype=torch.float32).reshape(12, 3)
full = Reservoir(capacity=5, dimension=3, seed=44)
full.add(vectors)
partial = Reservoir(capacity=5, dimension=3, seed=44)
partial.add(vectors[:6])
resumed = Reservoir.from_state_dict(partial.state_dict())
resumed.add(vectors[6:])
assert resumed.seen == full.seen
assert torch.equal(resumed.bank(), full.bank())
```

- [ ] **Step 2: Run the focused tests and verify they fail against the old framework**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/differential_uncertainty/test_config.py \
  tests/differential_uncertainty/test_corruptions.py \
  tests/differential_uncertainty/test_extraction.py \
  tests/differential_uncertainty/test_persistence.py \
  tests/differential_uncertainty/test_bank.py \
  tests/differential_uncertainty/test_scoring.py -q
```

Expected: failures for the new fixed APIs.

- [ ] **Step 3: Implement only the retained core APIs**

Keep the configuration dataclass this small:

```python
@dataclass(frozen=True)
class ExperimentConfig:
    image_size: tuple[int, int] = (640, 640)
    class_count: int = 80
    query_count: int = 300
    persistence_layer: int = 2
    persistence_dim: int = 335
    bank_capacity: int = 2_000
    bank_confidence_threshold: float = 0.5
    neighbors: int = 5
    bank_chunk_size: int = 512
    bootstrap_samples: int = 2_000
    levels: tuple[int, int] = (4, 5)

```

The public call signatures are `apply_corruption(image, name, severity)`, `RTDETRExtractor.extract_batch(images)`, `Reservoir.add(vectors)`, `Reservoir.add_record(record, threshold)`, `Reservoir.bank()`, `Reservoir.state_dict()`, `Reservoir.from_state_dict(state)`, `normalize_bank(bank)`, `mean_five_cosine(queries, normalized_bank, chunk_size)`, and `score_triplet(records, normalized_bank, config=FIXED_CONFIG)`.

Keep the current detector constructor, checkpoint-layout support, image resize behavior, layer-2 capture, and persistence kernel. Remove all artifact/manifest code and stop storing boxes after computing `padded_ids`.

`Reservoir` uses a preallocated CPU float16 tensor and one `random.Random(seed)`. Its state contains capacity, dimension, retained vectors, size, eligible-vector count, and `rng.getstate()`—nothing else.

`score_triplet` accepts exactly severities 0, 4, and 5 for one image/family and returns three rows with only `image_id`, `corruption`, `severity`, `fingerprint`, `confidence`, and `entropy`.

- [ ] **Step 4: Run the focused tests**

Expected: all six focused files pass.

- [ ] **Step 5: Commit the scientific core**

```bash
git add differential_uncertainty/config.py differential_uncertainty/corruptions \
  differential_uncertainty/extraction.py differential_uncertainty/persistence.py \
  differential_uncertainty/bank.py differential_uncertainty/scoring.py \
  tests/differential_uncertainty/test_config.py \
  tests/differential_uncertainty/test_corruptions.py \
  tests/differential_uncertainty/test_extraction.py \
  tests/differential_uncertainty/test_persistence.py \
  tests/differential_uncertainty/test_bank.py \
  tests/differential_uncertainty/test_scoring.py
git commit -m "refactor: retain fixed fingerprint method"
```

### Task 2: Replace evaluation and reporting

**Files:**
- Replace: `differential_uncertainty/evaluation.py`
- Replace: `differential_uncertainty/reporting.py`
- Replace: `tests/differential_uncertainty/test_evaluation.py`
- Replace: `tests/differential_uncertainty/test_reporting.py`

- [ ] **Step 1: Write focused failing tests**

Test that:

- Each of 19 families produces separate level-4 and level-5 AUROC rows: 38 tasks total.
- All three scores treat larger values as more corrupted.
- Aggregate AUROC is the equal-weight mean of the 38 tasks.
- The paired bootstrap resamples the same image IDs for fingerprint and its baseline, is deterministic under seed 44, and reports only fingerprint-minus-confidence and fingerprint-minus-entropy.
- Output consists of `per_image_scores.csv`, `results.csv`, `summary.json`, `report.md`, and `corruption_auroc_bars.png`.
- `results.csv` has 38 rows, the report table contains all 19 families, and the chart embed is the final nonblank line:
- The report does not claim a fingerprint advantage when the corresponding paired interval includes zero.

```python
assert report.read_text().splitlines()[-1] == (
    "![Per-corruption AUROC](corruption_auroc_bars.png)"
)
```

- [ ] **Step 2: Run the focused tests and verify they fail**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/differential_uncertainty/test_evaluation.py \
  tests/differential_uncertainty/test_reporting.py -q
```

Expected: failures because the current code reports the legacy five-level experiment and uses transactional bundles.

- [ ] **Step 3: Implement the fixed result structure**

Expose `binary_auroc(clean, corrupted)`, `evaluate_scores(rows, families, samples, seed)`, and `write_results(output, score_rows, evaluation, families)`.

`evaluate_scores` returns three keys. `tasks` contains 38 dictionaries with corruption, severity, and one AUROC for each method. `aggregate` contains the equal-weight fingerprint, confidence, and entropy means. `comparisons` contains only fingerprint-minus-confidence and fingerprint-minus-entropy, each with point, low, and high values.

Use rank-based AUROC, shared paired image resampling, direct CSV/JSON writes, and a matplotlib `Figure`. Do not retain trends, orientations, conditional concordance, HTML, pandas, staging directories, locks, or bundle comparison.

- [ ] **Step 4: Run the focused tests**

Expected: both files pass.

- [ ] **Step 5: Commit evaluation and reporting**

```bash
git add differential_uncertainty/evaluation.py differential_uncertainty/reporting.py \
  tests/differential_uncertainty/test_evaluation.py \
  tests/differential_uncertainty/test_reporting.py
git commit -m "refactor: report fixed corruption evidence"
```

### Task 3: Replace the runner, restart support, and CLI

**Files:**
- Replace: `differential_uncertainty/benchmark.py`
- Replace: `differential_uncertainty/cli.py`
- Keep: `differential_uncertainty/__main__.py`
- Replace: `tests/differential_uncertainty/test_benchmark.py`
- Replace: `tests/differential_uncertainty/test_cli.py`

- [ ] **Step 1: Write focused failing tests**

Use a fake extractor with a capacity-five configuration and tiny image directories. Test:

- Sorted, seeded selection from separate train and val roots.
- Required `--reference-count` and `--evaluation-count` flags.
- Absence of annotation, manifest, shard, and legacy `run` flags.
- Direct rejection when a count exceeds available images or an existing `run_config.json` differs.
- Clear rejection of missing roots/checkpoints, malformed progress files, too few eligible bank vectors, and non-finite extracted tensors or scores.
- An interrupted bank build resumes to the exact same bank as an uninterrupted build.
- An interrupted validation run resumes to the exact same score files, including stochastic corruptions.
- There is one score JSON per completed validation image.

- [ ] **Step 2: Run the focused tests and verify they fail**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/differential_uncertainty/test_benchmark.py \
  tests/differential_uncertainty/test_cli.py -q
```

Expected: failures for the new train/val and lightweight-restart interfaces.

- [ ] **Step 3: Implement the direct runner**

Expose `run_coco_benchmark(checkpoint, coco_train_images, coco_val_images, output, *, reference_count, evaluation_count, device="cuda:0", batch_size=1, seed=44, config=FIXED_CONFIG, extractor_factory=RTDETRExtractor, corruption_fn=apply_corruption) -> Path` and use these fixed progress schemas:

```python
bank_progress = {
    "next_image": 17,
    "reservoir": reservoir.state_dict(),
}
evaluation_progress = {
    "next_image": 17,
    "numpy_random_state": np.random.get_state(),
}
```

At startup set Python, NumPy, PyTorch, and CUDA seeds once. Enumerate flat COCO image directories, sort paths, shuffle each split using the supplied seed, and select the required counts. Save supplied paths, counts, seed, runtime options, and fixed constants directly in `run_config.json`; compare dictionaries on restart and do not hash anything.

Bank construction processes clean training images and replaces `bank_progress.pt` after each completed image. Validation extracts one clean image plus 38 corrupted variants, reuses the clean record across 19 family triplets, writes 57 score rows to `scores/<image-name>.json`, then advances `evaluation_progress.pt`. Restore its NumPy state so stochastic corruptions remain identical after restart. Use only a temporary file plus `Path.replace` for each progress/result write.

After all validation images finish, combine score files, evaluate them, and write the five report outputs.

- [ ] **Step 4: Replace the CLI with one command**

```python
benchmark = commands.add_parser("benchmark-coco")
benchmark.add_argument("--checkpoint", required=True)
benchmark.add_argument("--coco-train-images", required=True)
benchmark.add_argument("--coco-val-images", required=True)
benchmark.add_argument("--reference-count", required=True, type=positive_int)
benchmark.add_argument("--evaluation-count", required=True, type=positive_int)
benchmark.add_argument("--output", required=True)
benchmark.add_argument("--device", default="cuda:0")
benchmark.add_argument("--batch-size", default=1, type=positive_int)
benchmark.add_argument("--seed", default=44, type=nonnegative_int)
```

Prefix ordinary input/runtime failures with `error:` and return exit code 2.

- [ ] **Step 5: Run the focused tests**

Expected: both files pass.

- [ ] **Step 6: Commit the runner and CLI**

```bash
git add differential_uncertainty/benchmark.py differential_uncertainty/cli.py \
  differential_uncertainty/__main__.py \
  tests/differential_uncertainty/test_benchmark.py \
  tests/differential_uncertainty/test_cli.py
git commit -m "feat: run fixed COCO train-to-val benchmark"
```

### Task 4: Delete the old framework and update documentation

**Files:**
- Delete: `differential_uncertainty/artifacts.py`
- Delete: `differential_uncertainty/manifests.py`
- Delete: `differential_uncertainty/strong_corruption_study.py`
- Delete: `differential_uncertainty/corruptions/base.py`
- Delete: `differential_uncertainty/corruptions/gaussian_blur.py`
- Delete: `tests/differential_uncertainty/test_artifacts.py`
- Delete: `tests/differential_uncertainty/test_manifests.py`
- Delete: `tests/differential_uncertainty/test_strong_corruption_study.py`
- Delete: `tests/differential_uncertainty/test_legacy_parity.py`
- Delete: `tests/differential_uncertainty/test_repository_surface.py`
- Delete: `tests/differential_uncertainty/test_pipeline.py`
- Delete: `tests/differential_uncertainty/test_detector_parity.py`
- Delete: `tests/differential_uncertainty/fixtures/rtdetrv2_r18_layer2_golden.pt`
- Replace relevant section: `README.md`

- [ ] **Step 1: Confirm no retained module imports an obsolete module**

```bash
rg -n "artifacts|manifests|strong_corruption_study|GaussianBlur|Severity|global_kmeans|confidence_conditioned|reference_decile|responsive_decile" \
  differential_uncertainty tests/differential_uncertainty README.md
```

Expected: matches occur only in files being deleted and obsolete README text.

- [ ] **Step 2: Delete the listed framework files and tests**

Use `apply_patch` for text files. Delete the tracked binary fixture explicitly. Preserve every untracked user file.

- [ ] **Step 3: Replace the README workflow**

Document the fixed method, train/val separation, restart files, outputs, and this example:

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

State that 2,500 is an example, not a default, and levels 1 through 3 are not evaluated.

- [ ] **Step 4: Run the retained package tests**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty -q
```

Expected: all retained tests pass.

- [ ] **Step 5: Commit deletion and documentation**

```bash
git add README.md differential_uncertainty tests/differential_uncertainty
git commit -m "refactor: remove obsolete uncertainty framework"
```

Verify `git diff --cached --name-only` contains no untracked meeting, analysis, or historical-document files.

### Task 5: Verify the finished cleanup

**Files:**
- Modify only if a retained-path defect is found.

- [ ] **Step 1: Check formatting and prove removed machinery is absent**

```bash
git diff --check
rg -n "global_kmeans|confidence_conditioned|reference-manifest|coco-annotations|shard-size|source_digest|ensure_provenance|ShardWriter|fcntl|ctypes" \
  differential_uncertainty README.md tests/differential_uncertainty
```

Expected: both commands are silent.

- [ ] **Step 2: Verify the CLI**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m differential_uncertainty --help
/home/yuchen/miniconda3/envs/UE/bin/python -m differential_uncertainty benchmark-coco --help
```

Expected: only `benchmark-coco` exists, and both count flags are required.

- [ ] **Step 3: Run all retained package tests**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty -q
```

Expected: all pass.

- [ ] **Step 4: Run the repository test suite**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q
```

Expected: all pass except existing environment-dependent skips.

- [ ] **Step 5: Inspect the local handoff**

```bash
git status --short
git log --oneline -8
```

Expected: cleanup commits are local; pre-existing untracked user files remain unstaged. Do not push or remove the old feature worktree without a separate request.
