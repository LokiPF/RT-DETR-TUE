# Frozen Strong-Corruption Final 2,500/2,500 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the untouched 2,500-reference/2,500-evaluation experiment once with the development-selected fingerprint tuple frozen in advance.

**Architecture:** Add a frozen mode to the existing self-contained study module. It loads one committed policy file, builds one bank, writes one distance cache, evaluates every final image without selection, and reuses the existing AUROC/bootstrap helpers for the same three methods and per-corruption report.

**Tech Stack:** Python, PyTorch, NumPy, existing shard/manifests APIs, pytest.

---

## Frozen decision

The development run at commit `82e2166a712230f3b6822f7e666a6f6cc6b56cb6` selected this tuple. None of these values may change after final benchmark extraction begins.

```text
bank population       confidence_0_5
bank rule             maximum sigmoid confidence >= 0.5
bank capacity         2000
bank construction     Algorithm R reservoir
bank seed             44
query distance        mean_5_cosine
image aggregation     confidence_weighted_mean
levels                0, 4, 5
methods               fingerprint, direct max confidence, top-query entropy
bootstrap             2000 complete-image draws, seed 20260821
```

The saved confidence boundaries come from the 150-image development selection split and remain frozen. The final run does not refit them, select a policy, compare alternative banks, or inspect levels 1 through 3.

## Files

| Path | Responsibility |
| --- | --- |
| `configs/strong-corruption-final-2500.json` | Immutable selected tuple, development provenance, confidence boundaries, and final counts. |
| `differential_uncertainty/strong_corruption_study.py` | Add one frozen-policy loader and one frozen final-run path. |
| `tests/differential_uncertainty/test_strong_corruption_study.py` | Frozen-mode leakage, cache-count, roster, and report tests. |
| `runs/coco-imagecorruptions-2500-final/` | Ignored resumable final detector artifacts. |
| `runs/strong-corruption-final-2500x2500/` | Ignored final fingerprint cache and three reports. |

### Task 1: Commit the frozen policy before final data exists

**Files:**
- Create: `configs/strong-corruption-final-2500.json`
- Modify: `tests/differential_uncertainty/test_strong_corruption_study.py`

- [ ] **Step 1: Write the frozen policy file exactly**

```json
{
  "schema_version": 1,
  "mode": "frozen_final",
  "development_summary_sha256": "44a6a97874258de6621572993daf09a131d6cc9856fcf6b92b79fab2319c3d72",
  "development_manifest_sha256": {
    "reference": "9b2f6acce62cf1fcb72bc43c4d100aa3c89dfc507f631c4ffd36d38ee7cc4533",
    "evaluation": "49ca0562ab3eadcc539f6e19a47f00c4bd31b600b645d845ae80da0212dc2021"
  },
  "policy": {
    "bank": "confidence_0_5",
    "confidence_threshold": 0.5,
    "bank_capacity": 2000,
    "seed": 44,
    "distance": "mean_5_cosine",
    "neighbors": 5,
    "aggregation": "confidence_weighted_mean"
  },
  "levels": [0, 4, 5],
  "confidence_decile_boundaries": {
    "4": [0.5984295606613159, 0.758331686258316, 0.8537908792495728, 0.8958876132965088, 0.9209320545196533, 0.9374402165412903, 0.9462003111839294, 0.9558166265487671, 0.9641867876052856],
    "5": [0.5070948600769043, 0.6617589592933655, 0.8128668963909149, 0.873214840888977, 0.9108630418777466, 0.9352282881736755, 0.9447912573814392, 0.9539660811424255, 0.9633687138557434]
  },
  "bootstrap_draws": 2000,
  "bootstrap_seed": 20260821,
  "benchmark_split_seed": 20260825,
  "reference_count": 2500,
  "evaluation_count": 2500,
  "families": ["gaussian_blur", "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression", "speckle_noise", "spatter", "saturate"]
}
```

- [ ] **Step 2: Add and run a literal freeze test**

```python
def test_final_policy_is_frozen():
    payload = json.loads(Path("configs/strong-corruption-final-2500.json").read_text())
    assert payload["mode"] == "frozen_final"
    assert payload["policy"] == {
        "bank": "confidence_0_5", "confidence_threshold": 0.5,
        "bank_capacity": 2_000, "seed": 44,
        "distance": "mean_5_cosine", "neighbors": 5,
        "aggregation": "confidence_weighted_mean",
    }
    assert payload["levels"] == [0, 4, 5]
    assert (payload["reference_count"], payload["evaluation_count"]) == (2_500, 2_500)
    assert set(payload["confidence_decile_boundaries"]) == {"4", "5"}
```

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/differential_uncertainty/test_strong_corruption_study.py::test_final_policy_is_frozen -q
```

Expected: pass.

- [ ] **Step 3: Commit the freeze**

```bash
git add configs/strong-corruption-final-2500.json \
  tests/differential_uncertainty/test_strong_corruption_study.py
git commit -m "config: freeze strong corruption final policy"
```

### Task 2: Add one-policy final evaluation mode

**Files:**
- Modify: `differential_uncertainty/strong_corruption_study.py`
- Modify: `tests/differential_uncertainty/test_strong_corruption_study.py`

- [ ] **Step 1: Add failing final-mode tests**

Extend the existing tiny on-disk fixture with two development manifests whose IDs are contained in its reference roster and absent from its evaluation roster:

```python
write_manifest(tmp_path / "development-reference.csv", (1,))
write_manifest(tmp_path / "development-evaluation.csv", (2,))
```

Return those two paths from `tiny_study`. Add a `tiny_frozen_policy` fixture by loading `configs/strong-corruption-final-2500.json`, overriding only counts to 2/4, capacity to 6, families to `fog/snow`, and the two development manifest digests, then writing the payload under `tmp_path`.

Patch `select_policy` and `confidence_decile_boundaries` to raise if called, then run:

```python
result = run_final_study(
    tiny_study.reference_root,
    tiny_study.evaluation_root,
    tiny_study.annotations,
    tmp_path / "final-output",
    frozen_policy=tiny_frozen_policy,
    development_reference_manifest=tiny_study.development_reference_manifest,
    development_evaluation_manifest=tiny_study.development_evaluation_manifest,
    device="cpu",
)
assert result.selected_policy == Policy(
    "confidence_0_5", "mean_5_cosine", "confidence_weighted_mean", 44
)
assert len(list((result.output_dir / "cache").glob("*.pt"))) == 1
```

Also assert:

```python
summary = json.loads((result.output_dir / "summary.json").read_text())
rows = list(csv.DictReader((result.output_dir / "results.csv").open()))
assert summary["mode"] == "frozen_final"
assert "selection_ranking" not in summary
assert set(row["row_type"] for row in rows) == {
    "selected_method", "paired_difference", "conditional"
}
assert all(
    row["lower"] and row["upper"]
    for row in rows
    if row["row_type"] in {"selected_method", "paired_difference"}
)
assert all(
    (row["lower"] and row["upper"]) or int(row["count"]) == 0
    for row in rows
    if row["row_type"] == "conditional"
)
```

Add one roster test where a final evaluation ID occurs in either development manifest; `run_final_study` must raise before scoring.

Run the three new tests. Expected: fail because `run_final_study` does not exist.

- [ ] **Step 2: Implement the narrow final path**

Expose `run_final_study(reference_run, evaluation_benchmark, annotations, output_dir, *, frozen_policy, development_reference_manifest, development_evaluation_manifest, device="cpu") -> StudyRunResult`.

The implementation performs these operations in order:

1. Load and validate the committed JSON, including the exact policy names, counts, levels, 19-family roster, boundary ordering, development manifest digests, and bootstrap settings.
2. Load both development manifests and the final manifests. Require the final evaluation ID set to be disjoint from the union of development IDs, and require that union to be contained in the final 2,500-reference set.
3. Reuse the existing annotation, shard, metadata, candidate, group-preparation, bank, and cache helpers. Build only `confidence_0_5`, capacity 2,000, seed 44; open only `confidence_0_5-cosine-seed44.pt`.
4. Score only `(mean_5_cosine, confidence_weighted_mean)`. Mark every final row as `split="validation"`. Never call `select_policy`, `confidence_decile_boundaries`, `_score_primary_arms`, or `_score_sensitivity_seeds`.
5. Pass the frozen level-4/level-5 boundaries to `paired_validation_bootstrap` for the aggregate and all 38 family/severity tasks.
6. Write `results.csv`, `summary.json`, and `report.md`. Let `_selected_csv_rows` accept `seed_scores=None` and skip its seed loop in final mode; reuse its selected-method, difference, and conditional interval rows. Do not write candidate, controlled-comparison, or seed-sensitivity rows. The Markdown report retains both 19-row tables, aggregate intervals, conditional evidence, the frozen tuple/provenance, and the final sentence `Levels 1 through 3 were not evaluated.`

Add CLI arguments without changing the production package CLI:

```text
--frozen-final-policy PATH
--development-reference-manifest PATH
--development-evaluation-manifest PATH
```

When `--frozen-final-policy` is absent, preserve the existing development command unchanged.

- [ ] **Step 3: Run tests and commit**

```bash
CUDA_VISIBLE_DEVICES='' /home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/differential_uncertainty/test_strong_corruption_study.py \
  tests/differential_uncertainty/test_repository_surface.py -q
CUDA_VISIBLE_DEVICES='' /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q
git diff --check
git add differential_uncertainty/strong_corruption_study.py \
  tests/differential_uncertainty/test_strong_corruption_study.py
git commit -m "feat: evaluate frozen strong corruption policy"
```

Expected: focused and full CPU-only suites pass; GPU-only tests skip.

### Task 3: Materialize the untouched 2,500/2,500 benchmark

**Files:**
- Produce (ignored): `runs/coco-imagecorruptions-2500-final/`

- [ ] **Step 1: Check storage and GPU ownership**

The 250/250 benchmark occupies 6.9 GB, so reserve at least 80 GB for the 10-times-larger artifacts and final cache. Current free space was 1,009 GB on 2026-08-31. Run `nvidia-smi`; do not start detector extraction while another process leaves insufficient memory, and do not kill another user's process.

- [ ] **Step 2: Create manifests and prove the final roster is untouched**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python - <<'PY'
from pathlib import Path
from differential_uncertainty.benchmark import create_coco_manifests
from differential_uncertainty.manifests import load_manifest

root = Path("runs/coco-imagecorruptions-2500-final")
split = create_coco_manifests(
    "/home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json",
    "/home/yuchen/YuchenZ/Datasets/coco/val2017",
    root / "inputs",
    reference_count=2500,
    evaluation_count=2500,
)
dev_reference = load_manifest(
    "/home/yuchen/YuchenZ/UE/philip_sa/runs/coco-offline-bank-1000/inputs/reference-manifest.csv"
)
dev_evaluation = load_manifest(
    "/home/yuchen/YuchenZ/UE/philip_sa/.worktrees/imagecorruptions-coco-benchmark/runs/coco-imagecorruptions-250/inputs/evaluation-manifest.csv"
)
development_ids = {row.image_id for row in (*dev_reference, *dev_evaluation)}
final_reference_ids = set(split.reference_ids)
final_evaluation_ids = set(split.evaluation_ids)
assert len(development_ids) == 1250
assert development_ids <= {str(value) for value in final_reference_ids}
assert development_ids.isdisjoint({str(value) for value in final_evaluation_ids})
print("untouched final evaluation:", len(final_evaluation_ids))
PY
```

Expected: `untouched final evaluation: 2500`.

- [ ] **Step 3: Run or resume extraction**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m differential_uncertainty benchmark-coco \
  --coco-annotations /home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json \
  --coco-images /home/yuchen/YuchenZ/Datasets/coco/val2017 \
  --checkpoint /home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \
  --output-dir runs/coco-imagecorruptions-2500-final \
  --device cuda:0 --batch-size 1 --shard-size 50 \
  --reference-count 2500 --evaluation-count 2500
```

This command is resumable. Do not inspect its benchmark metrics to change the frozen fingerprint policy.

### Task 4: Run the final evaluation once and report every corruption

**Files:**
- Produce (ignored): `runs/strong-corruption-final-2500x2500/`

- [ ] **Step 1: Run the frozen command**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m differential_uncertainty.strong_corruption_study \
  --reference-run runs/coco-imagecorruptions-2500-final \
  --evaluation-benchmark runs/coco-imagecorruptions-2500-final \
  --annotations /home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json \
  --output-dir runs/strong-corruption-final-2500x2500 \
  --device cuda:0 \
  --frozen-final-policy configs/strong-corruption-final-2500.json \
  --development-reference-manifest /home/yuchen/YuchenZ/UE/philip_sa/runs/coco-offline-bank-1000/inputs/reference-manifest.csv \
  --development-evaluation-manifest /home/yuchen/YuchenZ/UE/philip_sa/.worktrees/imagecorruptions-coco-benchmark/runs/coco-imagecorruptions-250/inputs/evaluation-manifest.csv
```

Expected: one distance cache and the three top-level report files.

- [ ] **Step 2: Verify the frozen result shape**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python - <<'PY'
import csv, json
from pathlib import Path

root = Path("runs/strong-corruption-final-2500x2500")
rows = list(csv.DictReader((root / "results.csv").open()))
summary = json.loads((root / "summary.json").read_text())
selected = [row for row in rows if row["row_type"] == "selected_method"]
assert summary["mode"] == "frozen_final"
assert summary["selected_policy"]["policy_id"] == "confidence_0_5-mean_5_cosine-confidence_weighted_mean-44"
assert "selection_ranking" not in summary
assert len({row["corruption"] for row in selected}) == 19
assert {int(row["severity"]) for row in selected} == {4, 5}
assert {row["method"] for row in selected} == {
    "fingerprint", "direct_confidence_max", "softmax_entropy_top_confidence_query"
}
assert all(
    row["lower"] and row["upper"]
    for row in rows
    if row["row_type"] in {"selected_method", "paired_difference"}
)
assert all(
    (row["lower"] and row["upper"]) or int(row["count"]) == 0
    for row in rows
    if row["row_type"] == "conditional"
)
assert len(list((root / "cache").glob("*.pt"))) == 1
assert (root / "report.md").read_text().rstrip().endswith(
    "Levels 1 through 3 were not evaluated."
)
print(summary["conclusions"])
PY
```

- [ ] **Step 3: Interpret once, without revising the tuple**

Report the final aggregate points and intervals, all 19 family rows at levels 4 and 5, both paired baseline differences, all 38 conditional tasks, and the three evidence conclusions. Explicitly compare development and final results as replication evidence, not as another selection step. A weak or negative final result is reported as-is; it does not trigger a new policy search on these 2,500 evaluation images.
