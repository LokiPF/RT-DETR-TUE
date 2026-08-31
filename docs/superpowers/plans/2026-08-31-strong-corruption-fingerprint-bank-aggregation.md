# Lean Strong-Corruption Fingerprint Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether a persistent-fingerprint bank helps detect severity-4 and severity-5 corruption, identify which tested bank/distance/aggregation combination is most informative, and report every corruption family beside freshly computed confidence and Shannon-entropy baselines.

**Architecture:** Add one self-contained experiment module that reads the existing authenticated 1,000-reference and 250-evaluation caches, evaluates a fixed panel on a deterministic 150/100 image split, and writes one CSV, one JSON file, and one terminal-safe Markdown report. Reuse existing cache readers and scoring helpers; do not build a new cache format, production CLI subsystem, nested selector, or final 2,500/2,500 workflow in this phase.

**Tech Stack:** Python 3.11, PyTorch, NumPy, SciPy Hungarian matching, scikit-learn AUROC, pytest, and the repository's existing cache readers.

---

## Scope

This plan covers the development study only. If its held-out result is promising, the untouched 2,500/2,500 experiment gets a separate short plan.

This is the lean first subproject of the broader approved design. Reuse that document's fingerprint, matching, masking, distance, aggregation, and baseline definitions, but follow this plan's narrower execution scope: the nested selector, qualification system, cache production, and final run are intentionally not part of this implementation.

```text
reference images:  1,000 clean COCO images
evaluation images: 250 disjoint COCO images
selection split:   150 evaluation image IDs
validation split:  100 evaluation image IDs
families:          all 19 configured corruption families
levels used:       0, 4, and 5
bank construction: reservoir, capacity 2,000, primary seed 44
bank variants:     matched, background, balanced, all_valid, confidence_0_5
query distances:   mean_5_euclidean, fifth_neighbor_euclidean,
                   mean_5_standardized_euclidean, mean_5_cosine
aggregators:       mean_all, q90_all, top20_mean_all,
                   top_confidence_query, confidence_weighted_mean
sensitivity:       selected tuple only, seeds 42, 43, 44, 45, 46
bootstrap:         2,000 paired validation-image draws, seed 20,260,821
```

The primary panel attempts exactly 100 configurations: five bank populations, four query distances, and five image aggregators. Construction, capacity, and seed stay fixed so that bank composition is not confounded with other choices. If a reference population contains fewer than 2,000 eligible queries, its configurations are reported as infeasible and excluded rather than silently resized.

Select one tuple using only the 150-image selection split. The 100-image validation split supplies its reported results. Validation scores for other candidates may appear as explicitly exploratory ablations, but they cannot alter the selected tuple.

Keep the same union-unpadded query IDs for fingerprint, confidence, and entropy. Calculate clean-versus-level-4 and clean-versus-level-5 AUROC separately for every family. Do not pool positive severities, assume a previous confidence AUROC, introduce learned pooling, or hide a weak family behind an overall average.

Defer global k-means, capacity search, additional neighbor counts, nested leave-one-family-out selection, qualification gates, strong-only cache generation, production CLI integration, adversarial artifact hardening, and the final experiment.

When implementation encounters complicated existing code, first check its callers with `rg`. Simplify it only when the study directly needs that path, the replacement is smaller, and existing focused tests preserve its observable behavior. Otherwise bypass it from the self-contained study module. Do not perform unrelated cleanup in `artifacts.py`, `cli.py`, or `reporting.py` merely because those files are complex.

## Files

| Path | Responsibility |
| --- | --- |
| `differential_uncertainty/strong_corruption_study.py` | Policies, cache loading, reference matching, reservoir banks, scoring, selection, evaluation, simple caching, reporting, and a module-local command. |
| `tests/differential_uncertainty/test_strong_corruption_study.py` | Focused unit and tiny end-to-end tests. |
| `tests/differential_uncertainty/test_repository_surface.py` | Add exactly the two new Python files to the protected allowlist. |

The experiment writes only three report files plus an ordinary reusable cache:

```text
<output>/
  cache/
  results.csv
  summary.json
  report.md
```

### Task 0: Isolate the work and verify the baseline

**Files:**
- Read: `.gitignore`
- Read: `docs/superpowers/specs/2026-08-31-strong-corruption-fingerprint-bank-aggregation-design.md`

- [ ] **Step 1: Create an ignored worktree from this committed plan**

Read `/home/yuchen/.agents/skills/superpowers/using-git-worktrees/SKILL.md`, then run:

```bash
git check-ignore -q .worktrees
git worktree add .worktrees/strong-corruption-lean -b strong-corruption-lean HEAD
cd .worktrees/strong-corruption-lean
git status --short
```

Expected: the worktree is clean. Do not copy or stage changes from the original dirty checkout.

- [ ] **Step 2: Run the existing suite**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q
```

Expected: the existing suite passes. Record its exact count.

### Task 1: Build the five fixed reference-bank populations

**Files:**
- Create: `differential_uncertainty/strong_corruption_study.py`
- Create: `tests/differential_uncertainty/test_strong_corruption_study.py`
- Read: `differential_uncertainty/bank.py`
- Read: `differential_uncertainty/scoring.py:21-69`

- [ ] **Step 1: Write fixed-panel, split, matching, and reservoir tests**

Create `tests/differential_uncertainty/test_strong_corruption_study.py`:

```python
import pytest
import torch

from differential_uncertainty.strong_corruption_study import (
    AGGREGATIONS, BANK_VARIANTS, DISTANCES, Bank, CandidateRows, StudyConfig,
    build_bank, match_reference_queries, reservoir_indices, split_image_ids,
)


def test_public_panel_is_fixed_and_contains_100_candidates():
    config = StudyConfig()
    assert (config.reference_count, config.evaluation_count) == (1_000, 250)
    assert (config.selection_count, config.validation_count) == (150, 100)
    assert config.levels == (0, 4, 5)
    assert config.bank_capacity == 2_000 and config.primary_seed == 44
    assert len(BANK_VARIANTS) * len(DISTANCES) * len(AGGREGATIONS) == 100


def test_split_is_order_independent_and_disjoint():
    ids = tuple(f"image-{index}" for index in range(10))
    left = split_image_ids(ids, selection_count=6)
    right = split_image_ids(reversed(ids), selection_count=6)
    assert left == right
    assert len(left.selection) == 6 and len(left.validation) == 4
    assert set(left.selection).isdisjoint(left.validation)


def test_algorithm_r_has_a_fixed_oracle():
    assert reservoir_indices(10, capacity=4, seed=44).tolist() == [5, 1, 7, 4]


def test_hungarian_matching_marks_only_the_assigned_query():
    record = {
        "image_id": "1",
        "logits": torch.tensor([[8.0, -8.0], [-8.0, 8.0], [-8.0, -8.0]]),
        "boxes": torch.tensor([[.1, .1, .1, .1], [.5, .5, .2, .2], [.9, .9, .1, .1]]),
        "persistence": torch.arange(12, dtype=torch.float32).reshape(3, 4),
    }
    annotations = [{"id": 17, "category_id": 5, "bbox": [40, 40, 20, 20], "iscrowd": 0}]
    result = match_reference_queries(
        record, annotations, valid_query_ids=torch.tensor([0, 1, 2]),
        width=100, height=100, category_ids=(3, 5),
    )
    assert result.tolist() == [False, True, False]


@pytest.mark.parametrize("variant", BANK_VARIANTS)
def test_all_bank_variants_have_exact_capacity(reference_candidates, variant):
    bank = build_bank(
        reference_candidates, variant=variant,
        distance="mean_5_euclidean", capacity=4, seed=44,
    )
    assert bank.vectors.shape == (4, 2)
    if variant == "balanced":
        assert (bank.matched_count, bank.background_count) == (2, 2)
```

The `reference_candidates` fixture contains at least four matched rows, four background rows, four rows with confidence at least 0.5, and different numeric locations for matched and background vectors.

- [ ] **Step 2: Run the tests and observe the missing module**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_strong_corruption_study.py -q
```

Expected: collection fails because `strong_corruption_study` does not exist.

- [ ] **Step 3: Implement the fixed contracts**

Define these constants and small types in `strong_corruption_study.py`:

```python
BANK_VARIANTS = ("matched", "background", "balanced", "all_valid", "confidence_0_5")
DISTANCES = (
    "mean_5_euclidean", "fifth_neighbor_euclidean",
    "mean_5_standardized_euclidean", "mean_5_cosine",
)
AGGREGATIONS = (
    "mean_all", "q90_all", "top20_mean_all",
    "top_confidence_query", "confidence_weighted_mean",
)
FAMILIES = ("gaussian_blur", *ADDITIONAL_IMAGECORRUPTIONS)


@dataclass(frozen=True)
class StudyConfig:
    reference_count: int = 1_000
    evaluation_count: int = 250
    selection_count: int = 150
    validation_count: int = 100
    levels: tuple[int, ...] = (0, 4, 5)
    bank_capacity: int = 2_000
    primary_seed: int = 44
    sensitivity_seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    bootstrap_draws: int = 2_000
    bootstrap_seed: int = 20_260_821
    families: tuple[str, ...] = FAMILIES


@dataclass(frozen=True)
class CandidateRows:
    vectors: torch.Tensor
    matched: torch.Tensor
    confidence: torch.Tensor


@dataclass(frozen=True)
class ImageSplit:
    selection: tuple[str, ...]
    validation: tuple[str, ...]


@dataclass(frozen=True)
class Policy:
    bank: str
    distance: str
    aggregation: str
    seed: int

    @property
    def policy_id(self):
        return "-".join((self.bank, self.distance, self.aggregation, str(self.seed)))


@dataclass(frozen=True)
class Bank:
    vectors: torch.Tensor
    mean: torch.Tensor | None
    scale: torch.Tensor | None
    matched_count: int
    background_count: int
```

`split_image_ids` sorts unique string IDs by SHA-256 of `strong-study:<image-id>`, takes the first requested count for selection, and uses the rest for validation.

`match_reference_queries` uses the historical focal Hungarian cost: class weight 2, box-L1 weight 5, generalized-IoU weight 2, alpha 0.25, and gamma 2. Ignore crowd annotations and return one Boolean per valid query.

Build the canonical reference candidates by sorting clean reference records by string image ID, removing each repeated padded tail with `detect_padded_tail`, and storing finite CPU float32 persistence, the Hungarian matched flag, and maximum-sigmoid confidence.

Use standard Algorithm R. The eligible populations are exactly:

```text
matched        = matched rows
background     = unmatched rows
balanced       = half matched and half background
all_valid      = every valid row
confidence_0_5 = rows whose maximum sigmoid confidence is at least 0.5
```

For balanced banks, derive independent reservoir seeds from SHA-256 of `seed:matched` and `seed:background`. For standardized Euclidean, calculate mean and population standard deviation from the complete eligible reference population, then transform both sampled bank rows and evaluation queries with those same moments before computing distances. Balanced moments give matched and background total weight one half each. Record an infeasible candidate when a population cannot fill its allocation.

- [ ] **Step 4: Run and commit**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_strong_corruption_study.py -q
git add differential_uncertainty/strong_corruption_study.py tests/differential_uncertainty/test_strong_corruption_study.py
git commit -m "feat: define lean strong-corruption bank panel"
```

Expected: all tests from this task pass before the commit.

### Task 2: Score queries and aggregate images

**Files:**
- Modify: `differential_uncertainty/strong_corruption_study.py`
- Modify: `tests/differential_uncertainty/test_strong_corruption_study.py`
- Read: `differential_uncertainty/scoring.py:21-145`

- [ ] **Step 1: Add hand-calculated tests**

Append tests with these exact oracles:

```python
from differential_uncertainty.strong_corruption_study import (
    aggregate_queries, query_distances, score_image_group, top_query_entropy,
)


def test_distance_and_aggregation_oracles(bank_fixture):
    bank = bank_fixture([[1.0], [3.0], [8.0], [9.0], [10.0]])
    query = torch.tensor([[0.0]])
    assert query_distances(query, bank, "mean_5_euclidean").item() == pytest.approx(6.2)
    assert query_distances(query, bank, "fifth_neighbor_euclidean").item() == 10.0

    distances = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0])
    confidence = torch.tensor([.1, .9, .3, .2, .4])
    query_ids = torch.arange(5)
    assert aggregate_queries(distances, confidence, query_ids, "mean_all") == 22.0
    assert aggregate_queries(distances, confidence, query_ids, "q90_all") == 100.0
    assert aggregate_queries(distances, confidence, query_ids, "top20_mean_all") == 100.0
    assert aggregate_queries(distances, confidence, query_ids, "top_confidence_query") == 2.0
    assert aggregate_queries(
        distances, confidence, query_ids, "confidence_weighted_mean"
    ) == pytest.approx(43.6 / 1.9)


def test_entropy_uses_softmax_on_the_highest_confidence_query():
    logits = torch.tensor([[2.0, 0.0], [1.0, 1.0]])
    probability = logits[0].softmax(0)
    expected = -torch.xlogy(probability, probability).sum() / torch.log(torch.tensor(2.0))
    assert top_query_entropy(logits, torch.tensor([0, 1])) == pytest.approx(float(expected))


def test_one_union_mask_is_reused_for_fingerprint_confidence_and_entropy(strong_group, bank_fixture):
    bank = bank_fixture([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    rows = score_image_group(strong_group, bank, "mean_5_euclidean")
    assert {row.severity for row in rows} == {0, 4, 5}
    assert len({row.valid_query_ids for row in rows}) == 1
    assert all(row.confidence_score == 1.0 - row.raw_confidence for row in rows)
```

Define `bank_fixture` as a callable pytest fixture that converts the supplied vectors to CPU float32 and returns `Bank(vectors, None, None, 0, len(vectors))`. Define `strong_group` with severity-0/4/5 records whose final two query rows repeat, so the union-mask assertion exercises real padding removal.

- [ ] **Step 2: Implement the four distance rules and five aggregators**

Use chunked exact distance computation and retain five nearest bank rows. Raw Euclidean returns their mean or fifth distance according to the policy. Standardized Euclidean transforms evaluation queries with the bank's clean-reference mean and scale. Cosine L2-normalizes query and bank vectors. Reject nonfinite values, dimension mismatches, zero-norm cosine rows, and banks smaller than five.

Implement aggregation exactly:

```python
def aggregate_queries(distances, confidence, query_ids, name):
    if name == "mean_all":
        return float(distances.mean())
    if name == "q90_all":
        return float(distances.sort().values[int(np.ceil(.9 * len(distances))) - 1])
    if name == "top20_mean_all":
        return float(distances.topk(int(np.ceil(.2 * len(distances)))).values.mean())
    if name == "top_confidence_query":
        index = min(range(len(distances)), key=lambda i: (-float(confidence[i]), int(query_ids[i])))
        return float(distances[index])
    if name == "confidence_weighted_mean":
        return float((distances * confidence).sum() / confidence.sum())
    raise ValueError(f"unknown aggregation: {name}")
```

For entropy, select the retained query with greatest maximum-sigmoid confidence, break ties by lowest query ID, apply softmax across class logits, and return normalized Shannon entropy with `torch.xlogy`.

For each image/family group, take the complement of the union of repeated padded tails across exactly levels 0, 4, and 5. Reuse those IDs for all methods. Higher fingerprint distance, `1 - maximum confidence`, and entropy always mean more corrupted.

Return one compact row per severity:

```python
@dataclass(frozen=True)
class ImageScore:
    image_id: str
    family: str
    severity: int
    valid_query_ids: tuple[int, ...]
    fingerprint_scores: dict[str, float]
    raw_confidence: float
    confidence_score: float
    entropy_score: float
```

- [ ] **Step 3: Run and commit**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_strong_corruption_study.py -q
git add differential_uncertainty/strong_corruption_study.py tests/differential_uncertainty/test_strong_corruption_study.py
git commit -m "feat: score strong corruption queries and images"
```

Expected: all study tests pass before the commit.

### Task 3: Select on 150 images and evaluate on 100 images

**Files:**
- Modify: `differential_uncertainty/strong_corruption_study.py`
- Modify: `tests/differential_uncertainty/test_strong_corruption_study.py`
- Read: `differential_uncertainty/evaluation.py:55-253`

- [ ] **Step 1: Add separation and evaluation tests**

```python
from differential_uncertainty.strong_corruption_study import (
    confidence_conditioned_concordance, per_family_aurocs, select_policy,
)


def test_level_four_and_five_are_not_pooled():
    rows = synthetic_scores(clean=[0, 1], level4=[2, 3], level5=[.5, 1.5])
    result = per_family_aurocs(rows, method="fingerprint")["fog"]
    assert (result.level4, result.level5, result.strong) == (1.0, .75, .875)


def test_validation_values_cannot_change_selection():
    first = synthetic_candidate_table(selection_winner="matched", validation_offset=0)
    changed = synthetic_candidate_table(selection_winner="matched", validation_offset=10_000)
    assert select_policy(first).policy_id == select_policy(changed).policy_id


def test_empty_confidence_conditioned_task_is_inconclusive():
    result = confidence_conditioned_concordance(
        synthetic_cross_stratum_pair(), family="fog", severity=4, boundaries=(.5,)
    )
    assert result.pair_count == 0
    assert result.point is None
```

`synthetic_candidate_table` keeps its selection rows identical in both calls and reverses the validation ranking when `validation_offset` is nonzero. The test therefore fails if selection reads even one validation score.

- [ ] **Step 2: Implement selection, held-out metrics, and paired bootstrap**

Use these result types:

```python
@dataclass(frozen=True)
class FamilyResult:
    family: str
    level4: float
    level5: float
    strong: float


@dataclass(frozen=True)
class ConditionalResult:
    family: str
    severity: int
    point: float | None
    pair_count: int
```

For every candidate, calculate clean-versus-level-4 and clean-versus-level-5 AUROC per family, then their family strong mean. On the selection split, rank configurations by equal-family mean strong AUROC, then family median, then lexicographic policy ID. Freeze one policy before reading validation metrics.

Fit confidence-decile boundaries separately for levels 4 and 5 using selection pairs only. On validation, retain a clean/corrupted pair only when both maximum-confidence values fall in the same frozen stratum. Count correct fingerprint ordering as 1, a tie as 0.5, and reversal as 0. Return `None` with pair count zero instead of failing the whole study.

Use 2,000 paired bootstrap draws of complete validation image IDs. Each sampled ID brings all families, both severities, and all methods. Use identical draws for fingerprint, confidence, entropy, fingerprint-minus-confidence, fingerprint-minus-entropy, and confidence-conditioned concordance intervals.

After freezing the selected tuple, rebuild only that bank at seeds 42 through 46 and record validation mean, minimum, maximum, and standard deviation. Seed stability is descriptive, not a selection gate.

- [ ] **Step 3: Run and commit**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_strong_corruption_study.py -q
git add differential_uncertainty/strong_corruption_study.py tests/differential_uncertainty/test_strong_corruption_study.py
git commit -m "feat: select and evaluate strong corruption policy"
```

Expected: all study tests pass before the commit.

### Task 4: Load existing caches and write the three-file report

**Files:**
- Modify: `differential_uncertainty/strong_corruption_study.py`
- Modify: `tests/differential_uncertainty/test_strong_corruption_study.py`
- Modify: `tests/differential_uncertainty/test_repository_surface.py`
- Read: `differential_uncertainty/artifacts.py:1845-1875`
- Read: `differential_uncertainty/manifests.py:161-285`
- Read: `differential_uncertainty/benchmark.py:678-780`

- [ ] **Step 1: Add a tiny real-cache integration test**

Use `ShardWriter` to build two reference records with enough matched/background queries to fill a six-row bank, plus four evaluation image IDs for two families with levels 0 through 5. Override the test configuration to reference count 2, evaluation count 4, selection 2, validation 2, families `(fog, snow)`, bank capacity 6, and 50 bootstrap draws.

```python
from differential_uncertainty.strong_corruption_study import run_study


def test_tiny_study_filters_levels_and_reports_each_family(tiny_study, tmp_path):
    result = run_study(
        tiny_study.reference_root, tiny_study.evaluation_root,
        tiny_study.annotations, tmp_path / "output",
        config=tiny_study.config, device="cpu",
    )
    assert result.levels_read == {0, 4, 5}
    assert result.reported_families == {"fog", "snow"}
    assert {path.name for path in (tmp_path / "output").iterdir() if path.is_file()} == {
        "results.csv", "summary.json", "report.md",
    }
```

- [ ] **Step 2: Implement the minimal loader and cache**

Return the orchestration result as:

```python
@dataclass(frozen=True)
class StudyRunResult:
    selected_policy: Policy
    levels_read: frozenset[int]
    reported_families: frozenset[str]
    output_dir: Path
```

Load the reference and evaluation CSV manifests with `manifests.load_manifest`, require configured counts and disjoint IDs, and read clean records from `reference-artifacts/reference-extractions`. Read the benchmark corruption roster and stream `corruptions/<family>/artifacts/evaluation-extractions`, retaining exactly one record per image at levels 0, 4, and 5. Check only the scientifically relevant shared metadata: checkpoint digest, query count, class count, persistence layer, and persistence dimension.

Save one ordinary Torch cache per bank, distance geometry, and seed:

```text
cache/<bank>-<geometry>-seed<seed>.pt
```

The three geometries are raw Euclidean, standardized Euclidean, and cosine. Raw `mean_5_euclidean` and `fifth_neighbor_euclidean` share the same sorted five-neighbor tensor. Cache metadata contains the geometry, bank policy, source manifest digests, image IDs, and families. Reuse it when those fields match; otherwise recompute and overwrite it. All five aggregators also reuse these query distances. Materialize up to 15 feasible primary bank-geometry caches, then only four extra seeds for the selected bank and selected geometry.

- [ ] **Step 3: Write CSV, JSON, Markdown, and one module-local command**

`results.csv` uses:

```text
row_type,split,policy_id,bank,distance,aggregation,seed,corruption,
severity,method,point,lower,upper,count
```

It uses exactly these row types: `candidate_family`, `selected_method`, `paired_difference`, `conditional`, and `seed`. It includes all candidate-family rows, selected validation method rows for every family and level, paired family differences, conditional rows with pair counts, and five seed rows.

`summary.json` contains configuration, source paths and manifest digests, selected policy, selection ranking, aggregate validation metrics, seed summary, and three independent evidence conclusions. `detects_corruption` is `supported` when the aggregate validation fingerprint strong-AUROC interval has lower bound above 0.5, `not_supported` when its upper bound is at most 0.5, and otherwise `inconclusive`. `better_than_confidence` applies the same rule at zero to the paired fingerprint-minus-confidence interval. `information_after_confidence` applies the same rule at 0.5 to the confidence-conditioned concordance interval; it is `inconclusive` when no eligible pairs exist. Missing or incomplete required evidence also yields `inconclusive`. Seed stability remains a numeric mean, standard deviation, minimum, and maximum rather than a thresholded conclusion. None of these statuses suppresses output.

`report.md` is plain terminal-safe text with two 19-row tables:

```text
Corruption | Fingerprint L4 | Fingerprint L5 | Confidence L4 |
Confidence L5 | Entropy L4 | Entropy L5

Corruption | Fingerprint strong | Confidence strong | Entropy strong |
Fingerprint minus confidence | Fingerprint minus entropy
```

Follow them with mean and median summaries, the selected tuple, conditional diagnostic, seed range, and three controlled comparisons: banks at the selected distance/aggregation, distances at the selected bank/aggregation, and aggregators at the selected bank/distance. Show both selection and exploratory validation values for these comparisons. End with `Levels 1 through 3 were not evaluated.` Never include a presumed AUROC such as 0.714.

Expose only:

```bash
python -m differential_uncertainty.strong_corruption_study \
  --reference-run PATH --evaluation-benchmark PATH \
  --annotations PATH --output-dir PATH --device cuda:0
```

Do not modify the package's production CLI.

- [ ] **Step 4: Update the protected surface, run tests, and commit**

Add only the new module and test paths to `RETAINED_PYTHON`, changing its count from 46 to 48. Then run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/differential_uncertainty/test_strong_corruption_study.py \
  tests/differential_uncertainty/test_repository_surface.py -q
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q
git diff --check
```

Expected: focused and full tests pass; diff check has no output.

```bash
git add differential_uncertainty/strong_corruption_study.py \
  tests/differential_uncertainty/test_strong_corruption_study.py \
  tests/differential_uncertainty/test_repository_surface.py
git commit -m "feat: run lean strong corruption study"
```

### Task 5: Run the 1,000/250 study and inspect every corruption

**Files:**
- Produce (ignored): `runs/strong-corruption-lean-1000x250/`

- [ ] **Step 1: Run or resume the study**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m differential_uncertainty.strong_corruption_study \
  --reference-run /home/yuchen/YuchenZ/UE/philip_sa/runs/coco-offline-bank-1000 \
  --evaluation-benchmark /home/yuchen/YuchenZ/UE/philip_sa/.worktrees/imagecorruptions-coco-benchmark/runs/coco-imagecorruptions-250 \
  --annotations /home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json \
  --output-dir runs/strong-corruption-lean-1000x250 \
  --device cuda:0
```

Expected: all 100 primary configurations are attempted, infeasible populations are reported, only levels 0/4/5 contribute, selection uses 150 image IDs, and the selected feasible tuple is reported on the other 100 image IDs.

- [ ] **Step 2: Verify report completeness**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -c 'import csv,json,pathlib; root=pathlib.Path("runs/strong-corruption-lean-1000x250"); rows=list(csv.DictReader((root/"results.csv").open())); summary=json.loads((root/"summary.json").read_text()); chosen=[r for r in rows if r["row_type"]=="selected_method"]; assert len({r["corruption"] for r in chosen})==19; assert {int(r["severity"]) for r in chosen}=={4,5}; assert {r["method"] for r in chosen}=={"fingerprint","direct_confidence_max","softmax_entropy_top_confidence_query"}; print(summary["selected_policy"], 19)'
```

Expected: the selected tuple and `19` are printed.

- [ ] **Step 3: Interpret the report before planning anything else**

Read `runs/strong-corruption-lean-1000x250/report.md` and answer:

1. Which fixed bank population is most informative?
2. Does a scene-wide aggregator such as mean or q90 work, or only the top-confidence query?
3. Does the fingerprint detect held-out strong corruption at all?
4. Does it beat confidence or retain ordering within comparable-confidence pairs?
5. Which individual corruption families fail despite any favorable average?
6. Is the selected tuple stable across seeds 42 through 46?

If promising, write a separate short plan that freezes this tuple for the untouched final experiment. If not, report the negative result and stop.
