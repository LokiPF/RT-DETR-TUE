# Strong-Corruption Fingerprint Bank and Aggregation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable, leakage-safe study that selects a fixed persistent-fingerprint bank, query-to-bank distance, and deterministic image aggregator for detecting severity-4 and severity-5 corruption, then reports every corruption family beside direct-confidence and Shannon-entropy baselines.

**Architecture:** Add an offline `strong_*` analysis path over the authenticated detector-extraction caches; leave the legacy benchmark and score path unchanged. Separate immutable policies, reference matching and bank construction, cache access, scoring, evaluation, nested selection, reporting, and orchestration so each scientific axis has explicit provenance and can be tested independently. The public workflow evaluates only severities 0, 4, and 5 and never learns an image pooling model.

**Tech Stack:** Python 3.11, PyTorch, NumPy, SciPy (`linear_sum_assignment`), scikit-learn metrics, pytest, the repository's JSON/Torch artifact helpers, and the existing RT-DETRv2 COCO cache format.

**Approved design:** [`docs/superpowers/specs/2026-08-31-strong-corruption-fingerprint-bank-aggregation-design.md`](../specs/2026-08-31-strong-corruption-fingerprint-bank-aggregation-design.md)

---

## Execution precondition

The source checkout already contains unrelated user edits. Commit only this plan, then execute it in a dedicated ignored worktree rooted at that clean plan commit, which must descend from approved-design commit `e3599b6`; do not stage, discard, or copy the dirty source checkout's changes. All commands below assume the repository root and `/home/yuchen/miniconda3/envs/UE/bin/python`.

The fixed public study constants are:

```text
development: 1,000 reference images, 250 evaluation images
final:       2,500 reference images, 2,500 evaluation images
families:    gaussian_blur plus the 18 configured imagecorruptions families
levels read: 0, 4, 5 only
bank compositions: matched, background, balanced, all_valid, confidence_0_5
bank constructions: reservoir, global_kmeans
capacities: 1,000, 2,000, 5,000
sensitivity seeds: 42, 43, 44, 45, 46; final seed: 44
distance screen: mean_5_euclidean
distance panel: mean_5_euclidean, fifth_neighbor_euclidean,
                mean_5_standardized_euclidean, mean_5_cosine
neighbor diagnostic: k=1; selectable k: 5, 10, 20
aggregators: mean_all, q90_all, top20_mean_all,
             top_confidence_query, confidence_weighted_mean
selection target: equal-family mean of separate level-4 and level-5 AUROCs
bootstrap: 10,000 paired image-ID draws, seed 20,260,821
```

Do not encode or cite a presumed confidence AUROC such as `0.714`. Recompute confidence and entropy from the same roster, query mask, family, and severity records as the fingerprint.

## File structure

The package stays flat because the repository uses a flat `differential_uncertainty/` layout and its repository-surface test protects that topology.

| Path | Responsibility |
| --- | --- |
| `differential_uncertainty/strong_policies.py` | Frozen policy dataclasses, candidate panels, stable IDs, feasibility/error types, and public development/final constants. |
| `differential_uncertainty/strong_matching.py` | Historical focal Hungarian matching from cached clean detector outputs to COCO objects. |
| `differential_uncertainty/strong_bank.py` | Canonical candidate pools, Algorithm-R reservoirs, weighted moments, deterministic Lloyd k-means, bank artifacts, and safe serialization. |
| `differential_uncertainty/strong_data.py` | Authenticated reference/evaluation cache discovery and level-0/4/5 group streaming. |
| `differential_uncertainty/strong_scoring.py` | Exact chunked query-to-bank distances, neighbor reductions, five deterministic image aggregators, and confidence/entropy baselines. |
| `differential_uncertainty/strong_evaluation.py` | Per-family AUROC, confidence-conditioned concordance, fixed image-ID folds, and paired bootstrap intervals. |
| `differential_uncertainty/strong_selection.py` | Leakage-safe staged selection, tie breakers, qualification gates, frozen-policy records, and the 95-cell audit trace. |
| `differential_uncertainty/strong_reporting.py` | Validated CSV/JSON/Markdown tables, evidence statements, provenance, and atomic publication. |
| `differential_uncertainty/strong_pipeline.py` | Resumable development/final orchestration and lazy materialization of only the arms a stage needs. |
| `differential_uncertainty/extraction.py` | Preserve the legacy six-level default while allowing the fixed strong cache writer to request exactly levels 0, 4, and 5. |
| `differential_uncertainty/cli.py` | Add `strong-coco-develop` and `strong-coco-final`; expose runtime paths/device only, not post-hoc scientific switches. |
| `README.md` | Document cache prerequisites, development command, qualification gate, frozen final command, and output interpretation. |
| `tests/differential_uncertainty/test_strong_*.py` | Focused unit and integration coverage for each new module. |
| `tests/differential_uncertainty/conftest.py` | Shared safe synthetic cache, bank, observation, provider, and report factories used across the strong-workflow tests. |
| `tests/differential_uncertainty/test_cli.py` | Parser and dispatch coverage for strong cache, development, and final commands. |
| `tests/differential_uncertainty/test_extraction.py` | Regression coverage for the fixed strong-only extraction subset and unchanged legacy default. |
| `tests/differential_uncertainty/test_repository_surface.py` | Extend the exact Python-file allowlist from 46 to 65 without changing allowed top-level directories. |

Use this artifact layout. Each directory after `artifacts/` is named by an immutable SHA-256 identity derived from source manifests, annotations, checkpoint/configuration, and the complete policy object; resuming a path with different provenance is an error.

```text
<output>/
  development.json
  final.json                         # final-mode output only
  frozen-policy.json                 # absent when no candidate qualifies
  artifacts/
    matching/<artifact-id>/matching.pt
    candidate-pool/<artifact-id>/candidates.pt
    banks/<artifact-id>/bank.pt
    query-scores/<artifact-id>/<family>.pt
    image-scores/<artifact-id>/<family>.json
    validation/<artifact-id>/cell-000.json through cell-094.json
  reports/
    strong-corruption-report.md
    per-corruption-levels.csv
    per-corruption-strong.csv
    conditional-concordance.csv
    paired-differences.csv
    bank-composition-ablation.csv
    construction-ablation.csv
    capacity-ablation.csv
    distance-ablation.csv
    neighbor-count-ablation.csv
    aggregation-ablation.csv
    seed-sensitivity.csv
    selection-trace.json
    provenance.json
```

### Task 0: Create the isolated implementation worktree and prove the baseline

**Files:**
- Read: `.gitignore`
- Read: `docs/superpowers/specs/2026-08-31-strong-corruption-fingerprint-bank-aggregation-design.md`

- [ ] **Step 1: Load the worktree instructions**

Read `/home/yuchen/.agents/skills/superpowers/using-git-worktrees/SKILL.md` completely and follow it for this task.

- [ ] **Step 2: Verify the worktree root is ignored**

Run:

```bash
git check-ignore -q .worktrees
```

Expected: exit status `0` and no output. If it is not ignored, stop before creation and add only `.worktrees/` to `.gitignore` in a separate reviewed commit.

- [ ] **Step 3: Create the feature worktree from the committed plan HEAD**

Run:

```bash
git merge-base --is-ancestor e3599b6 HEAD
git worktree add .worktrees/strong-corruption-selection -b strong-corruption-selection HEAD
cd .worktrees/strong-corruption-selection
git status --short
```

Expected: the ancestry check exits `0`, worktree creation succeeds, this plan exists in the worktree, and `git status --short` prints nothing.

- [ ] **Step 4: Run the full baseline suite**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q
```

Expected: all baseline tests pass. Record the exact count in execution notes; stop and diagnose any failure before implementing new behavior.

### Task 1: Define immutable study policies and legal candidate panels

**Files:**
- Create: `differential_uncertainty/strong_policies.py`
- Create: `tests/differential_uncertainty/test_strong_policies.py`

- [ ] **Step 1: Write failing policy tests**

Create `tests/differential_uncertainty/test_strong_policies.py`:

```python
import pytest

from differential_uncertainty.strong_policies import (
    AGGREGATION_NAMES, BANK_CAPACITIES, BANK_COMPOSITIONS, BANK_SEEDS,
    QUERY_DISTANCE_PANEL, CompletePolicy, ImageAggregationPolicy,
    QueryDistancePolicy, STRONG_DEVELOPMENT, STRONG_FINAL,
)


def test_public_study_constants_are_frozen():
    assert (STRONG_DEVELOPMENT.reference_count, STRONG_DEVELOPMENT.evaluation_count) == (1_000, 250)
    assert (STRONG_FINAL.reference_count, STRONG_FINAL.evaluation_count) == (2_500, 2_500)
    assert STRONG_DEVELOPMENT.levels == (0, 4, 5)
    assert len(STRONG_DEVELOPMENT.families) == 19
    assert BANK_COMPOSITIONS == ("matched", "background", "balanced", "all_valid", "confidence_0_5")
    assert BANK_CAPACITIES == (1_000, 2_000, 5_000)
    assert BANK_SEEDS == (42, 43, 44, 45, 46)
    assert AGGREGATION_NAMES == (
        "mean_all", "q90_all", "top20_mean_all",
        "top_confidence_query", "confidence_weighted_mean",
    )


def test_query_panel_has_four_named_arms_and_k1_cannot_be_selected():
    assert tuple(policy.name for policy in QUERY_DISTANCE_PANEL) == (
        "mean_5_euclidean", "fifth_neighbor_euclidean",
        "mean_5_standardized_euclidean", "mean_5_cosine",
    )
    with pytest.raises(ValueError, match="diagnostic-only"):
        QueryDistancePolicy("euclidean", "mean", 1)


def test_complete_policy_round_trips_with_a_stable_id():
    policy = CompletePolicy.stage_one("matched", "q90_all")
    restored = CompletePolicy.from_dict(policy.to_dict())
    assert restored == policy
    assert restored.policy_id == policy.policy_id
    assert len(policy.policy_id) == 64


def test_unknown_policy_value_is_rejected():
    with pytest.raises(ValueError, match="aggregation"):
        ImageAggregationPolicy("maximum")
```

- [ ] **Step 2: Run the tests and verify the import fails**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_strong_policies.py -q
```

Expected: collection fails with `ModuleNotFoundError: differential_uncertainty.strong_policies`.

- [ ] **Step 3: Implement the policy schema and canonical serialization**

Create `differential_uncertainty/strong_policies.py` with the following public contract. Keep `for_tests` so small synthetic tests can reduce counts/families; the CLI must always choose one of the two fixed public instances.

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json

from .corruptions.imagecorruptions import ADDITIONAL_IMAGECORRUPTIONS

BANK_COMPOSITIONS = ("matched", "background", "balanced", "all_valid", "confidence_0_5")
BANK_CONSTRUCTIONS = ("reservoir", "global_kmeans")
BANK_CAPACITIES = (1_000, 2_000, 5_000)
BANK_SEEDS = (42, 43, 44, 45, 46)
AGGREGATION_NAMES = (
    "mean_all", "q90_all", "top20_mean_all",
    "top_confidence_query", "confidence_weighted_mean",
)
STRONG_LEVELS = (0, 4, 5)
STRONG_FAMILIES = ("gaussian_blur", *ADDITIONAL_IMAGECORRUPTIONS)


class InfeasiblePolicy(RuntimeError):
    """One candidate arm cannot satisfy its declared scientific policy."""


def canonical_id(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class StrongStudyConfig:
    reference_count: int
    evaluation_count: int
    families: tuple[str, ...] = STRONG_FAMILIES
    levels: tuple[int, ...] = STRONG_LEVELS
    bootstrap_draws: int = 10_000
    bootstrap_seed: int = 20_260_821
    testing: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.reference_count) is not int
            or type(self.evaluation_count) is not int
            or self.reference_count <= 0
            or self.evaluation_count <= 0
        ):
            raise ValueError("study counts must be positive")
        if not self.families or len(set(self.families)) != len(self.families):
            raise ValueError("families must be nonempty and unique")
        if self.levels != STRONG_LEVELS:
            raise ValueError("strong-corruption levels must be exactly (0, 4, 5)")
        if type(self.bootstrap_draws) is not int or self.bootstrap_draws <= 0:
            raise ValueError("bootstrap draws must be positive")
        if type(self.bootstrap_seed) is not int or self.bootstrap_seed < 0:
            raise ValueError("bootstrap seed must be nonnegative")
        if type(self.testing) is not bool:
            raise ValueError("testing must be boolean")

    @classmethod
    def for_tests(cls, **changes: object) -> "StrongStudyConfig":
        return replace(STRONG_DEVELOPMENT, testing=True, **changes)


@dataclass(frozen=True)
class BankPolicy:
    composition: str
    construction: str
    capacity: int
    seed: int

    def __post_init__(self) -> None:
        if self.composition not in BANK_COMPOSITIONS:
            raise ValueError(f"unknown bank composition: {self.composition}")
        if self.construction not in BANK_CONSTRUCTIONS:
            raise ValueError(f"unknown bank construction: {self.construction}")
        if type(self.capacity) is not int or self.capacity <= 0:
            raise ValueError("bank capacity must be positive")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("bank seed must be nonnegative")


@dataclass(frozen=True)
class QueryDistancePolicy:
    metric: str
    reduction: str
    k: int
    diagnostic: bool = False

    def __post_init__(self) -> None:
        if self.metric not in {"euclidean", "standardized_euclidean", "cosine"}:
            raise ValueError(f"unknown query metric: {self.metric}")
        if self.reduction not in {"mean", "radius"}:
            raise ValueError(f"unknown neighbor reduction: {self.reduction}")
        if self.k not in {1, 5, 10, 20}:
            raise ValueError(f"unsupported neighbor count: {self.k}")
        if self.k == 1 and not self.diagnostic:
            raise ValueError("k=1 is diagnostic-only")

    @property
    def name(self) -> str:
        if self.metric == "euclidean" and self.reduction == "radius" and self.k == 5:
            return "fifth_neighbor_euclidean"
        return f"{self.reduction}_{self.k}_{self.metric}"


@dataclass(frozen=True)
class ImageAggregationPolicy:
    name: str

    def __post_init__(self) -> None:
        if self.name not in AGGREGATION_NAMES:
            raise ValueError(f"unknown aggregation: {self.name}")


@dataclass(frozen=True)
class CompletePolicy:
    bank: BankPolicy
    query: QueryDistancePolicy
    aggregation: ImageAggregationPolicy

    @classmethod
    def stage_one(cls, composition: str, aggregation: str) -> "CompletePolicy":
        return cls(
            BankPolicy(composition, "reservoir", 2_000, 44),
            QueryDistancePolicy("euclidean", "mean", 5),
            ImageAggregationPolicy(aggregation),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "CompletePolicy":
        return cls(
            BankPolicy(**value["bank"]),
            QueryDistancePolicy(**value["query"]),
            ImageAggregationPolicy(**value["aggregation"]),
        )

    @property
    def policy_id(self) -> str:
        return canonical_id(self.to_dict())


STRONG_DEVELOPMENT = StrongStudyConfig(1_000, 250)
STRONG_FINAL = StrongStudyConfig(2_500, 2_500)
QUERY_DISTANCE_PANEL = (
    QueryDistancePolicy("euclidean", "mean", 5),
    QueryDistancePolicy("euclidean", "radius", 5),
    QueryDistancePolicy("standardized_euclidean", "mean", 5),
    QueryDistancePolicy("cosine", "mean", 5),
)
```

- [ ] **Step 4: Run policy tests**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_strong_policies.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Commit the policy contract**

```bash
git add differential_uncertainty/strong_policies.py tests/differential_uncertainty/test_strong_policies.py
git commit -m "feat: define strong corruption study policies"
```

### Task 2: Reproduce historical focal Hungarian matching

**Files:**
- Create: `differential_uncertainty/strong_matching.py`
- Create: `tests/differential_uncertainty/test_strong_matching.py`
- Read: `differential_uncertainty/scoring.py:15-60`

- [ ] **Step 1: Write exact cost and assignment tests**

Create `tests/differential_uncertainty/test_strong_matching.py`:

```python
import math
import torch
import pytest

from differential_uncertainty.strong_matching import (
    ReferenceMatches, focal_hungarian_cost, match_reference_record,
)


def test_focal_hungarian_cost_matches_historical_formula():
    logits = torch.tensor([[0.0]], dtype=torch.float64)
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float64)
    result = focal_hungarian_cost(logits, boxes, torch.tensor([0]), boxes.clone())
    class_cost = 0.25 * 0.25 * math.log(2) - 0.75 * 0.25 * math.log(2)
    assert result.item() == pytest.approx(2 * class_cost - 2.0)


def test_match_records_matched_and_background_valid_queries():
    record = {
        "image_id": "0002",
        "logits": torch.tensor([[8.0, -8.0], [-8.0, 8.0], [-8.0, -8.0]]),
        "boxes": torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2], [0.9, 0.9, 0.1, 0.1]]),
    }
    annotations = [{"id": 17, "category_id": 5, "bbox": [40, 40, 20, 20], "iscrowd": 0}]
    result = match_reference_record(
        record, annotations, valid_query_ids=torch.tensor([0, 1]),
        width=100, height=100, category_ids=(3, 5),
    )
    assert isinstance(result, ReferenceMatches)
    assert result.query_ids.tolist() == [0, 1]
    assert result.annotation_ids.tolist() == [-1, 17]
    assert result.class_ids.tolist() == [-1, 1]
    assert torch.allclose(result.confidences, record["logits"][:2].sigmoid().amax(1))
```

```python
def test_all_crowd_annotations_leave_every_valid_query_unmatched():
    record = {
        "image_id": "x", "logits": torch.zeros(2, 2),
        "boxes": torch.tensor([[.2, .2, .1, .1], [.8, .8, .1, .1]]),
    }
    result = match_reference_record(
        record, [{"id": 1, "category_id": 3, "bbox": [1, 1, 2, 2], "iscrowd": 1}],
        valid_query_ids=torch.tensor([0, 1]), width=10, height=10,
        category_ids=(3, 5),
    )
    assert result.annotation_ids.tolist() == [-1, -1]


def test_nonfinite_matching_cost_is_rejected():
    with pytest.raises(ValueError, match="nonfinite Hungarian cost"):
        focal_hungarian_cost(
            torch.tensor([[float("nan")]]), torch.tensor([[.5, .5, .2, .2]]),
            torch.tensor([0]), torch.tensor([[.5, .5, .2, .2]]),
        )
```

- [ ] **Step 2: Run the matching tests and observe the missing module**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_strong_matching.py -q
```

Expected: collection fails with `ModuleNotFoundError: differential_uncertainty.strong_matching`.

- [ ] **Step 3: Implement the exact matcher**

Create `differential_uncertainty/strong_matching.py`. Convert COCO `xywh` boxes to normalized `cxcywh`, map sorted COCO category IDs to zero-based detector class IDs, ignore `iscrowd=1`, and call SciPy only after finite validation:

```python
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou

MATCH_CLASS_WEIGHT = 2.0
MATCH_L1_WEIGHT = 5.0
MATCH_GIOU_WEIGHT = 2.0
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
FOCAL_EPSILON = 1e-8


@dataclass(frozen=True)
class ReferenceMatches:
    image_id: str
    query_ids: torch.Tensor
    annotation_ids: torch.Tensor
    class_ids: torch.Tensor
    confidences: torch.Tensor

    @property
    def matched(self) -> torch.Tensor:
        return self.annotation_ids.ge(0)


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, width, height = boxes.unbind(-1)
    return torch.stack((cx-width/2, cy-height/2, cx+width/2, cy+height/2), dim=-1)


def focal_hungarian_cost(logits, boxes, target_classes, target_boxes):
    probability = logits.sigmoid()
    negative = (1-FOCAL_ALPHA) * probability.pow(FOCAL_GAMMA) * -torch.log(
        1-probability+FOCAL_EPSILON
    )
    positive = FOCAL_ALPHA * (1-probability).pow(FOCAL_GAMMA) * -torch.log(
        probability+FOCAL_EPSILON
    )
    class_cost = positive[:, target_classes] - negative[:, target_classes]
    l1_cost = torch.cdist(boxes, target_boxes, p=1)
    giou_cost = -generalized_box_iou(_cxcywh_to_xyxy(boxes), _cxcywh_to_xyxy(target_boxes))
    result = 2.0 * class_cost + 5.0 * l1_cost + 2.0 * giou_cost
    if not torch.isfinite(result).all():
        raise ValueError("nonfinite Hungarian cost")
    return result


def match_reference_record(
    record: Mapping[str, object],
    annotations: Sequence[Mapping[str, object]],
    *, valid_query_ids: torch.Tensor, width: int, height: int,
    category_ids: tuple[int, ...],
) -> ReferenceMatches:
    query_ids = torch.sort(valid_query_ids.long().cpu()).values
    if query_ids.unique().numel() != query_ids.numel():
        raise ValueError("valid query IDs must be unique")
    logits = torch.as_tensor(record["logits"])[query_ids]
    boxes = torch.as_tensor(record["boxes"])[query_ids]
    confidence = logits.float().sigmoid().amax(dim=1).cpu()
    kept = sorted(
        (a for a in annotations if not a.get("iscrowd", 0)),
        key=lambda a: int(a["id"]),
    )
    annotation_ids = torch.full((len(query_ids),), -1, dtype=torch.long)
    class_ids = torch.full((len(query_ids),), -1, dtype=torch.long)
    if kept:
        category_to_class = {category_id: index for index, category_id in enumerate(category_ids)}
        targets = torch.tensor([
            [(float(a["bbox"][0])+float(a["bbox"][2])/2)/width,
             (float(a["bbox"][1])+float(a["bbox"][3])/2)/height,
             float(a["bbox"][2])/width, float(a["bbox"][3])/height]
            for a in kept
        ], dtype=boxes.dtype)
        classes = torch.tensor(
            [category_to_class[int(a["category_id"])] for a in kept], dtype=torch.long
        )
        row, column = linear_sum_assignment(
            focal_hungarian_cost(logits, boxes, classes, targets).detach().cpu().numpy()
        )
        for query_row, target_column in zip(row.tolist(), column.tolist(), strict=True):
            annotation_ids[query_row] = int(kept[target_column]["id"])
            class_ids[query_row] = int(classes[target_column])
    return ReferenceMatches(str(record["image_id"]), query_ids, annotation_ids, class_ids, confidence)
```

The serialized matching artifact must additionally hold annotation-file, reference-manifest, checkpoint, and detector-configuration digests. Task 10 binds those fields through the common artifact identity before matching is reused.

- [ ] **Step 4: Run matching tests**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_strong_matching.py -q
```

Expected: all matching tests pass.

- [ ] **Step 5: Commit exact matching**

```bash
git add differential_uncertainty/strong_matching.py tests/differential_uncertainty/test_strong_matching.py
git commit -m "feat: add reference query matching"
```

### Task 3: Build one canonical candidate pool and exact reservoir compositions

**Files:**
- Create: `differential_uncertainty/strong_bank.py`
- Create: `tests/differential_uncertainty/test_strong_bank.py`
- Read: `differential_uncertainty/bank.py:19-105`
- Read: `differential_uncertainty/scoring.py:15-60`

- [ ] **Step 1: Write candidate-population and Algorithm-R tests**

Start `tests/differential_uncertainty/test_strong_bank.py` with deterministic oracles:

```python
import torch
import pytest

from differential_uncertainty.strong_bank import (
    CandidatePool, algorithm_r_indices, build_candidate_pool, composition_indices,
    derive_population_seed,
)
from differential_uncertainty.strong_matching import ReferenceMatches
from differential_uncertainty.strong_policies import InfeasiblePolicy


def pool() -> CandidatePool:
    return CandidatePool(
        vectors=torch.arange(40, dtype=torch.float32).reshape(10, 4),
        image_ids=("a", "a", "b", "b", "c", "c", "d", "d", "e", "e"),
        query_ids=torch.tensor([0, 1] * 5),
        matched=torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.bool),
        confidences=torch.tensor([.9, .8, .7, .6, .5, .4, .3, .2, .1, .0]),
        annotation_ids=torch.tensor([10, -1, 11, -1, 12, -1, 13, -1, 14, -1]),
        source_digest="a" * 64,
    )


def test_algorithm_r_has_a_fixed_numpy_generator_oracle():
    assert algorithm_r_indices(torch.arange(10), 4, seed=44).tolist() == [5, 1, 7, 4]


def test_population_seed_is_derived_from_the_literal_label():
    assert derive_population_seed(44, "matched") == 7490089924146745319
    assert derive_population_seed(44, "background") == 535838926718841778


def test_composition_masks_and_balanced_capacity_are_exact():
    assert composition_indices(pool(), "matched", 4, 44).numel() == 4
    assert composition_indices(pool(), "background", 4, 44).numel() == 4
    assert composition_indices(pool(), "all_valid", 4, 44).numel() == 4
    assert set(composition_indices(pool(), "confidence_0_5", 4, 44).tolist()) <= {0, 1, 2, 3, 4}
    balanced = composition_indices(pool(), "balanced", 4, 44)
    assert pool().matched[balanced].sum().item() == 2
    assert (~pool().matched[balanced]).sum().item() == 2


def test_insufficient_population_is_infeasible_not_padded():
    with pytest.raises(InfeasiblePolicy, match="matched.*6.*5"):
        composition_indices(pool(), "matched", 6, 44)
```

```python
def test_candidate_pool_is_canonical_and_removes_each_clean_padded_tail():
    def record(image_id, offset):
        repeated_box = torch.tensor([.9, .9, .1, .1])
        repeated_logit = torch.tensor([-9.0, -9.0])
        repeated_fingerprint = torch.tensor([9.0, 9.0, 9.0])
        return {
            "image_id": image_id, "severity": 0,
            "boxes": torch.stack((torch.tensor([.1, .1, .1, .1]), torch.tensor([.5, .5, .2, .2]), repeated_box, repeated_box)),
            "logits": torch.stack((torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), repeated_logit, repeated_logit)),
            "persistence": torch.stack((torch.tensor([offset, 1.0, 2.0]), torch.tensor([offset, 3.0, 4.0]), repeated_fingerprint, repeated_fingerprint)),
        }
    matches = {
        image_id: ReferenceMatches(
            image_id, torch.tensor([0, 1]), torch.tensor([10, -1]),
            torch.tensor([0, -1]), torch.tensor([.8, .2]),
        )
        for image_id in ("a", "b")
    }
    left = build_candidate_pool(
        [record("b", 20.0), record("a", 10.0)], matches, "a" * 64
    )
    right = build_candidate_pool(
        [record("a", 10.0), record("b", 20.0)], matches, "a" * 64
    )
    assert tuple(zip(left.image_ids, left.query_ids.tolist())) == (
        ("a", 0), ("a", 1), ("b", 0), ("b", 1)
    )
    assert left.vectors.dtype == torch.float32
    assert torch.equal(left.vectors, right.vectors)
    assert left.matched.tolist() == [True, False, True, False]
```

- [ ] **Step 2: Run the focused bank tests and verify failure**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_strong_bank.py -q
```

Expected: collection fails with `ModuleNotFoundError: differential_uncertainty.strong_bank`.

- [ ] **Step 3: Implement canonical pool validation and composition selection**

Create `differential_uncertainty/strong_bank.py` with these types and exact sampling functions:

```python
from dataclasses import dataclass
import hashlib
import json
import numpy as np
import torch

from .scoring import detect_padded_tail
from .strong_policies import InfeasiblePolicy


@dataclass(frozen=True)
class CandidatePool:
    vectors: torch.Tensor
    image_ids: tuple[str, ...]
    query_ids: torch.Tensor
    matched: torch.Tensor
    confidences: torch.Tensor
    annotation_ids: torch.Tensor
    source_digest: str

    def __post_init__(self) -> None:
        size = self.vectors.shape[0]
        if self.vectors.ndim != 2 or self.vectors.dtype != torch.float32:
            raise ValueError("candidate vectors must be a float32 matrix")
        if not torch.isfinite(self.vectors).all():
            raise ValueError("candidate vectors must be finite")
        if not all(len(value) == size for value in (
            self.image_ids, self.query_ids, self.matched,
            self.confidences, self.annotation_ids,
        )):
            raise ValueError("candidate metadata length mismatch")
        keys = tuple(zip(self.image_ids, self.query_ids.tolist()))
        if keys != tuple(sorted(keys)) or len(set(keys)) != size:
            raise ValueError("candidates must have unique canonical image/query order")

    @classmethod
    def for_tests(cls, *, vectors, matched=None):
        vectors = torch.as_tensor(vectors, dtype=torch.float32)
        size = vectors.shape[0]
        labels = (
            torch.ones(size, dtype=torch.bool)
            if matched is None else torch.as_tensor(matched, dtype=torch.bool)
        )
        return cls(
            vectors=vectors,
            image_ids=tuple(f"image-{index:04d}" for index in range(size)),
            query_ids=torch.zeros(size, dtype=torch.long),
            matched=labels,
            confidences=torch.linspace(1.0, 0.0, size),
            annotation_ids=torch.where(
                labels, torch.arange(size, dtype=torch.long),
                torch.full((size,), -1, dtype=torch.long),
            ),
            source_digest="f" * 64,
        )


def derive_population_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def algorithm_r_indices(eligible: torch.Tensor, capacity: int, *, seed: int) -> torch.Tensor:
    eligible = eligible.long().cpu()
    if eligible.numel() < capacity:
        raise InfeasiblePolicy(
            f"population capacity {capacity} exceeds candidate count {eligible.numel()}"
        )
    reservoir = eligible[:capacity].tolist()
    generator = np.random.default_rng(seed)
    for seen in range(capacity, eligible.numel()):
        replacement = int(generator.integers(0, seen + 1))
        if replacement < capacity:
            reservoir[replacement] = int(eligible[seen])
    return torch.tensor(reservoir, dtype=torch.long)


def _eligible(pool: CandidatePool, composition: str) -> torch.Tensor:
    if composition == "matched":
        mask = pool.matched
    elif composition == "background":
        mask = ~pool.matched
    elif composition == "all_valid":
        mask = torch.ones(len(pool.image_ids), dtype=torch.bool)
    elif composition == "confidence_0_5":
        mask = pool.confidences >= 0.5
    else:
        raise ValueError(f"composition {composition} needs separate populations")
    return mask.nonzero(as_tuple=False).flatten()


def composition_indices(pool, composition, capacity, seed):
    if composition != "balanced":
        return algorithm_r_indices(_eligible(pool, composition), capacity, seed=seed)
    if capacity % 2:
        raise InfeasiblePolicy("balanced capacity must be even")
    half = capacity // 2
    matched = algorithm_r_indices(
        _eligible(pool, "matched"), half,
        seed=derive_population_seed(seed, "matched"),
    )
    background = algorithm_r_indices(
        _eligible(pool, "background"), half,
        seed=derive_population_seed(seed, "background"),
    )
    return torch.cat((matched, background))
```

Implement `build_candidate_pool(records, matches_by_image, source_digest)` in the same module with this exact sequence:

```text
1. Validate every record is severity 0 and has finite `logits`, `boxes`, and `persistence`.
2. Reject duplicate/string-colliding image IDs; iterate records by string image ID.
3. Compute `padded = detect_padded_tail(record)` and retain the complement in ascending query-ID order.
4. Require `matches_by_image[image_id].query_ids` to equal that retained ID tensor exactly.
5. Convert retained persistence to detached contiguous CPU float32; never threshold it.
6. Concatenate vectors and matching metadata, validate `CandidatePool`, and record population counts.
```

- [ ] **Step 4: Run the reservoir/candidate tests**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_strong_bank.py -q
```

Expected: all tests added in this task pass.

- [ ] **Step 5: Commit canonical candidate construction**

```bash
git add differential_uncertainty/strong_bank.py tests/differential_uncertainty/test_strong_bank.py
git commit -m "feat: add strong bank candidate populations"
```

### Task 4: Add metric-aware reservoirs, global k-means, and bank artifacts

**Files:**
- Modify: `differential_uncertainty/strong_bank.py`
- Modify: `tests/differential_uncertainty/test_strong_bank.py`
- Read: `differential_uncertainty/bank.py:105-230`
- Read: `differential_uncertainty/artifacts.py:100-180`

- [ ] **Step 1: Add failing transform and construction tests**

Append these cases to `tests/differential_uncertainty/test_strong_bank.py`:

```python
from differential_uncertainty.strong_bank import (
    BankArtifact, build_bank, population_moments,
    save_bank_artifact, load_bank_artifact,
)
from differential_uncertainty.strong_policies import BankPolicy


def test_balanced_moments_give_each_population_half_the_weight():
    values = torch.tensor([[0.0], [2.0], [100.0], [102.0], [104.0]])
    matched = torch.tensor([True, True, False, False, False])
    candidates = CandidatePool.for_tests(vectors=values, matched=matched)
    mean, scale = population_moments(candidates, "balanced")
    assert mean.item() == pytest.approx(51.5)
    expected_variance = 0.5 * (((values[:2]-51.5)**2).mean()) + 0.5 * (((values[2:]-51.5)**2).mean())
    assert scale.square().item() == pytest.approx(expected_variance.item())


def test_global_kmeans_finds_two_cluster_centers_deterministically():
    candidates = CandidatePool.for_tests(
        vectors=torch.tensor([[1.0], [2.0], [101.0], [102.0]]),
        matched=torch.tensor([True, True, True, True]),
    )
    policy = BankPolicy("matched", "global_kmeans", capacity=2, seed=44)
    first = build_bank(candidates, policy, metric="euclidean", device="cpu")
    second = build_bank(candidates, policy, metric="euclidean", device="cpu")
    assert torch.allclose(first.vectors.sort(dim=0).values, torch.tensor([[1.5], [101.5]]))
    assert torch.equal(first.vectors, second.vectors)
    assert first.metadata["cluster_counts"] == second.metadata["cluster_counts"]


def test_standardized_zero_variance_and_cosine_zero_norm_are_arm_infeasibilities():
    constant = CandidatePool.for_tests(vectors=torch.ones(6, 2))
    with pytest.raises(InfeasiblePolicy, match="zero reference variance"):
        build_bank(constant, BankPolicy("all_valid", "reservoir", 2, 44), metric="standardized_euclidean")
    zero = CandidatePool.for_tests(vectors=torch.zeros(6, 2))
    with pytest.raises(InfeasiblePolicy, match="zero-norm"):
        build_bank(zero, BankPolicy("all_valid", "reservoir", 2, 44), metric="cosine")


def test_bank_artifact_round_trip_rejects_metadata_or_tensor_tampering(tmp_path):
    artifact = build_bank(pool(), BankPolicy("all_valid", "reservoir", 4, 44), metric="euclidean")
    path = tmp_path / "bank.pt"
    save_bank_artifact(artifact, path)
    loaded = load_bank_artifact(path, expected_identity=artifact.artifact_id)
    assert loaded.policy == artifact.policy
    assert loaded.metadata == artifact.metadata
    assert loaded.artifact_id == artifact.artifact_id
    torch.testing.assert_close(loaded.vectors, artifact.vectors)
    with pytest.raises(ValueError, match="artifact identity"):
        load_bank_artifact(path, expected_identity="0" * 64)
```

```python
def test_balanced_kmeans_preserves_half_capacity_per_population():
    candidates = CandidatePool.for_tests(
        vectors=torch.tensor([[0.0], [2.0], [4.0], [6.0], [100.0], [102.0], [104.0], [106.0], [108.0], [110.0]]),
        matched=torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.bool),
    )
    artifact = build_bank(
        candidates, BankPolicy("balanced", "global_kmeans", 4, 44),
        metric="euclidean", device="cpu",
    )
    assert artifact.metadata["centroid_populations"] == [
        "matched", "matched", "background", "background"
    ]
    assert artifact.vectors[:2].max() < artifact.vectors[2:].min()


def test_empty_cluster_reseed_uses_lowest_canonical_farthest_index():
    from differential_uncertainty.strong_bank import _farthest_reseed_indices
    assert _farthest_reseed_indices(
        torch.tensor([4.0, 4.0, 0.0]), count=1
    ).tolist() == [0]
```

- [ ] **Step 2: Run the new tests and verify the missing APIs**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_strong_bank.py -q
```

Expected: failures name `population_moments`, `build_bank`, and artifact serialization.

- [ ] **Step 3: Implement metric transforms and deterministic construction**

Add these interfaces to `strong_bank.py`:

```python
@dataclass(frozen=True)
class BankArtifact:
    vectors: torch.Tensor              # already standardized when required
    policy: BankPolicy
    metric: str
    reference_mean: torch.Tensor | None
    reference_scale: torch.Tensor | None
    metadata: dict
    artifact_id: str


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _eligible_mask(pool, composition):
    if composition == "matched":
        return pool.matched
    if composition == "background":
        return ~pool.matched
    if composition == "all_valid" or composition == "balanced":
        return torch.ones(len(pool.image_ids), dtype=torch.bool)
    if composition == "confidence_0_5":
        return pool.confidences >= 0.5
    raise ValueError(f"unknown bank composition: {composition}")


def population_moments(pool, composition):
    vectors, matched = pool.vectors, pool.matched
    if composition == "balanced":
        left, right = vectors[matched], vectors[~matched]
        if not len(left) or not len(right):
            raise InfeasiblePolicy("balanced moments need both populations")
        mean = 0.5 * left.mean(0) + 0.5 * right.mean(0)
        variance = 0.5 * (left-mean).square().mean(0) + 0.5 * (right-mean).square().mean(0)
    else:
        selected = vectors[_eligible_mask(pool, composition)]
        mean = selected.mean(0)
        variance = (selected-mean).square().mean(0)
    if (variance == 0).any():
        raise InfeasiblePolicy("standardized distance has zero reference variance")
    return mean, variance.sqrt()


@dataclass(frozen=True)
class KMeansResult:
    centroids: torch.Tensor
    iterations: int
    cluster_counts: tuple[int, ...]
    empty_replacements: int


def _assign_clusters(candidates, centroids, chunk_size):
    labels, squared = [], []
    for chunk in candidates.split(chunk_size):
        distance = torch.cdist(chunk, centroids, p=2.0, compute_mode="use_mm_for_euclid_dist")
        minimum, assigned = distance.min(dim=1)
        labels.append(assigned)
        squared.append(minimum.square())
    return torch.cat(labels), torch.cat(squared)


def _farthest_reseed_indices(squared_distances, *, count):
    return torch.argsort(
        squared_distances, descending=True, stable=True
    )[:count]


def global_kmeans(
    candidates, capacity, seed, device, *, chunk_size=8192,
    max_iterations=50, tolerance=1e-4,
):
    if candidates.shape[0] < capacity:
        raise InfeasiblePolicy(
            f"k-means capacity {capacity} exceeds candidate count {candidates.shape[0]}"
        )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial = torch.randperm(candidates.shape[0], generator=generator)[:capacity]
    candidates = candidates.to(device=device, dtype=torch.float32)
    centroids = candidates.index_select(0, initial.to(candidates.device)).clone()
    previous_labels = None
    empty_replacements = 0
    for iteration in range(1, max_iterations + 1):
        labels, squared = _assign_clusters(candidates, centroids, chunk_size)
        if previous_labels is not None and torch.equal(labels, previous_labels):
            break
        sums = torch.zeros_like(centroids)
        sums.index_add_(0, labels, candidates)
        counts = torch.bincount(labels, minlength=capacity)
        populated = counts > 0
        updated = centroids.clone()
        updated[populated] = sums[populated] / counts[populated, None]
        empty = (~populated).nonzero(as_tuple=False).flatten()
        if empty.numel():
            farthest = _farthest_reseed_indices(squared, count=empty.numel())
            updated[empty] = candidates.index_select(0, farthest)
            empty_replacements += int(empty.numel())
        shift = torch.linalg.vector_norm(updated-centroids, dim=1).amax()
        centroids = updated
        previous_labels = labels
        if float(shift) <= tolerance:
            break
    final_labels, _ = _assign_clusters(candidates, centroids, chunk_size)
    final_counts = torch.bincount(final_labels, minlength=capacity)
    result = centroids.detach().cpu().float().contiguous()
    if not torch.isfinite(result).all():
        raise ValueError("global k-means produced nonfinite centroids")
    return KMeansResult(
        result, iteration, tuple(int(value) for value in final_counts.cpu()),
        empty_replacements,
    )
```

This is the exact Lloyd loop: distinct seeded initialization, chunked exact assignment, first-centroid tie handling, stable canonical farthest reseeding, unchanged-label/maximum-shift stopping, and finite CPU output.

Implement `build_bank(pool, policy, metric, device)` as follows:

```text
reservoir/raw or cosine: select canonical indices with `composition_indices`.
reservoir/standardized: compute moments from the complete eligible population,
                        then transform selected rows.
k-means/raw or cosine: cluster the complete eligible population.
k-means/standardized: compute complete-population moments, transform all eligible
                      rows, then cluster in standardized space.
balanced k-means: derive `matched` and `background` seeds, cluster each complete
                  population for capacity/2, concatenate matched then background.
cosine: reject any zero-norm candidate or bank row for that arm; normalization
        itself remains in the scorer.
```

Record policy, source digest, metric, eligible count, matched/background retained counts, initialization seed(s), iteration count(s), tolerance, cluster counts, centroid-population labels, and empty replacements in metadata. Compute the identity exactly as:

```python
moments_identity = None if mean is None else {
    "mean_sha256": tensor_sha256(mean),
    "scale_sha256": tensor_sha256(scale),
}
artifact_id = canonical_id({
    "source_digest": pool.source_digest,
    "policy": asdict(policy),
    "metric": metric,
    "moments": moments_identity,
})
```

Use `atomic_torch` for publication. Load only through:

```python
with _open_regular_file(path, error_message="bank artifact must be a regular file") as handle:
    payload = torch.load(handle, weights_only=True, map_location="cpu")
```

Validate exact keys, tensor shapes/dtypes/finiteness, metadata, and expected artifact identity before returning an artifact.

- [ ] **Step 4: Run all strong-bank tests and legacy-bank regression tests**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/differential_uncertainty/test_strong_bank.py \
  tests/differential_uncertainty/test_bank.py -q
```

Expected: both files pass; legacy `bank.py` behavior is unchanged.

- [ ] **Step 5: Commit bank construction and artifacts**

```bash
git add differential_uncertainty/strong_bank.py tests/differential_uncertainty/test_strong_bank.py
git commit -m "feat: construct strong fingerprint bank variants"
```

### Task 5: Load authenticated benchmark caches as 0/4/5 groups

**Files:**
- Create: `differential_uncertainty/strong_data.py`
- Create: `tests/differential_uncertainty/test_strong_data.py`
- Create: `tests/differential_uncertainty/conftest.py`
- Read: `differential_uncertainty/artifacts.py:1845-1915`
- Read: `differential_uncertainty/benchmark.py:620-790`
- Read: `differential_uncertainty/manifests.py`

- [ ] **Step 1: Write cache-discovery and severity-isolation tests**

Create `tests/differential_uncertainty/conftest.py` with shared fixtures named `strong_cache`, `bank_artifact`, `strong_group`, `observations`, `synthetic_provider`, `counters`, `final_cache`, `qualified_development`, `coco_fixture`, `completed_study`, `report_result`, and `tiny_strong_benchmark`. Each fixture returns fresh paths/objects under `tmp_path`; no production module imports test code. Start its cache writer with this exact primitive:

```python
def write_record_cache(path, records, *, metadata):
    with ShardWriter(path, metadata, shard_size=3) as writer:
        for record in records:
            writer.add(record)
    loaded = list(iter_records(path))
    assert [(row["image_id"], row["severity"]) for row in loaded] == [
        (row["image_id"], row["severity"]) for row in records
    ]


def extraction_record(image_id, severity, *, query_count=12, offset=0.0):
    base = torch.arange(query_count, dtype=torch.float32)[:, None]
    return {
        "image_id": str(image_id), "severity": int(severity),
        "logits": (base + offset).repeat(1, 80),
        "boxes": (base + offset).repeat(1, 4),
        "persistence": (base + offset).repeat(1, 335),
    }
```

`strong_cache(...)` writes four distinct regular image files (two disjoint reference and two evaluation images), canonical reference/evaluation CSV manifests, a COCO annotation JSON with image dimensions/categories/objects, reference and evaluation extraction caches through `write_record_cache`, and exact legacy/strong provenance JSON. Its returned object exposes `reference_root`, `evaluation_root`, `weak_evaluation_root`, `changed_evaluation_root`, `annotations`, and `config`; its callable form creates requested family/level/fault variants. Extend this same fixture module in later tasks with the named factory methods shown in their tests.

Keep the later shared-fixture contracts explicit:

- `bank_artifact` returns a fresh, finite CPU `float32` artifact with four two-dimensional bank vectors, known weighted moments, a valid SHA-256 identity, and `.with_metric(...)` / `.with_vectors(...)` immutable-copy helpers.
- `observations` is a callable factory for `ScoreObservation` rows. Calling it with `clean`, `level4`, and `level5` creates aligned image IDs and fingerprint rows; `.paired_confidence(...)`, `.conditional_pairs(...)`, and `.two_identical_methods(...)` create exactly the paired rows named by the evaluation tests.
- `synthetic_provider` returns a fresh logging `ScoreProvider`. Its deterministic score table makes `balanced-cosine-q90` the winner by default, supports the three requested fault/offset switches, and separately records every inner and outer request.
- `counters` records matching, pool, bank, family, and severity materializations. Setting `raise_after_family` makes its next matching family callback raise `RuntimeError("injected interruption")` once.
- `qualified_development` writes a semantically valid 19-family development/frozen pair, exposes both paths and the frozen confidence boundaries, and `.with_fault(...)` writes fresh mutated copies without changing the originals.
- `coco_fixture` writes four tiny real image files, minimal COCO annotations, a regular dummy checkpoint, two corruption callables, and a fake extractor that emits valid tensors while recording requested levels.
- `completed_study` runs the miniature development fixture once and returns its output path; `report_result` builds a complete schema-valid public report payload, accepts independent `fingerprint_lower`, `conditional_lower`, `confidence_delta_lower`, and `seed_scores` evidence overrides, and returns deep copies for every fault mutation.
- `final_cache` uses the guarded test factories specified in Task 10; `tiny_strong_benchmark` uses real cache shards and the two-family test policy without mocking any scientific operation.

Create `tests/differential_uncertainty/test_strong_data.py`. The evaluation fixture is parameterized with records ordered as both six levels per image and only `(0, 4, 5)` per image.

```python
from differential_uncertainty.strong_data import (
    StrongRecordGroup, open_strong_benchmark,
)
from differential_uncertainty.strong_policies import StrongStudyConfig


@pytest.mark.parametrize("stored_levels", [(0, 1, 2, 3, 4, 5), (0, 4, 5)])
def test_loader_yields_only_one_zero_four_five_group(stored_levels, strong_cache):
    reference_run, evaluation_benchmark = strong_cache(
        families=("fog",), levels=stored_levels
    )
    config = StrongStudyConfig.for_tests(
        reference_count=2, evaluation_count=2, families=("fog",)
    )
    inputs = open_strong_benchmark(
        reference_run, evaluation_benchmark, strong_cache.annotations, config
    )
    groups = list(inputs.iter_family_groups("fog"))
    assert [(g.image_id, tuple(g.records)) for g in groups] == [
        ("eval-0", (0, 4, 5)), ("eval-1", (0, 4, 5))
    ]
    assert all(isinstance(group, StrongRecordGroup) for group in groups)


def test_mild_record_values_cannot_change_strong_groups(strong_cache):
    first_reference, first_evaluation = strong_cache(
        families=("fog",), levels=(0, 1, 2, 3, 4, 5), mild_offset=0
    )
    second_reference, second_evaluation = strong_cache(
        families=("fog",), levels=(0, 1, 2, 3, 4, 5), mild_offset=10_000
    )
    config = StrongStudyConfig.for_tests(reference_count=2, evaluation_count=2, families=("fog",))
    left = [g.tensor_digest() for g in open_strong_benchmark(
        first_reference, first_evaluation, strong_cache.annotations, config
    ).iter_family_groups("fog")]
    right = [g.tensor_digest() for g in open_strong_benchmark(
        second_reference, second_evaluation, strong_cache.annotations, config
    ).iter_family_groups("fog")]
    assert left == right


def test_loader_rejects_overlap_missing_family_and_bad_group(strong_cache):
    config = StrongStudyConfig.for_tests(reference_count=2, evaluation_count=2, families=("fog",))
    with pytest.raises(ValueError, match="overlap"):
        open_strong_benchmark(*strong_cache(overlap=True), strong_cache.annotations, config)
    with pytest.raises(ValueError, match="family roster"):
        open_strong_benchmark(*strong_cache(families=("snow",)), strong_cache.annotations, config)
    bad_reference, bad_evaluation = strong_cache(families=("fog",), levels=(0, 4))
    with pytest.raises(ValueError, match="exactly.*0.*4.*5"):
        list(open_strong_benchmark(
            bad_reference, bad_evaluation, strong_cache.annotations, config
        ).iter_family_groups("fog"))
```

The fixture writes the real paths shown below and valid lower-case SHA-256 values in `benchmark-manifest.json`, `corruption-roster.json`, and cache manifests. Give it a `fault` keyword and add this exact validation matrix:

```python
@pytest.mark.parametrize(("fault", "message"), [
    ("annotation_digest", "annotation digest"),
    ("checkpoint_digest", "checkpoint"),
    ("persistence_dimension", "extraction configuration"),
    ("duplicate_image_severity", "duplicate image/severity"),
    ("reference_count", "reference count"),
    ("evaluation_count", "evaluation count"),
    ("unsafe_family", "unsafe family"),
    ("extra_family", "family roster"),
])
def test_loader_rejects_each_provenance_or_roster_fault(strong_cache, fault, message):
    reference, evaluation = strong_cache(fault=fault)
    config = StrongStudyConfig.for_tests(
        reference_count=2, evaluation_count=2, families=("fog",)
    )
    with pytest.raises(ValueError, match=message):
        inputs = open_strong_benchmark(
            reference, evaluation, strong_cache.annotations, config
        )
        list(inputs.iter_family_groups("fog"))
```

- [ ] **Step 2: Run the loader tests and verify the import failure**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_strong_data.py -q
```

Expected: collection fails with `ModuleNotFoundError: differential_uncertainty.strong_data`.

- [ ] **Step 3: Implement immutable input discovery and grouped streaming**

Create `differential_uncertainty/strong_data.py` with these exact paths:

```python
REFERENCE_MANIFEST = "inputs/reference-manifest.csv"
REFERENCE_BINDING = "reference-artifacts/reference-bank-binding.json"
REFERENCE_CACHE = "reference-artifacts/reference-extractions"
STRONG_CACHE_MANIFEST = "strong-cache-manifest.json"
EVALUATION_MANIFEST = "inputs/evaluation-manifest.csv"
BENCHMARK_MANIFEST = "inputs/benchmark-manifest.json"
CORRUPTION_ROSTER = "corruption-roster.json"
EVALUATION_CACHE = "corruptions/{family}/artifacts/evaluation-extractions"
```

Define the immutable boundary objects:

```python
@dataclass(frozen=True)
class StrongRecordGroup:
    image_id: str
    family: str
    records: dict[int, dict]

    def __post_init__(self) -> None:
        if tuple(sorted(self.records)) != (0, 4, 5):
            raise ValueError("strong group needs exactly severities 0, 4, and 5")
        if any(str(record["image_id"]) != self.image_id for record in self.records.values()):
            raise ValueError("strong group image ID mismatch")

    def tensor_digest(self) -> str:
        return safe_tensor_mapping_digest(self.records)


def safe_tensor_mapping_digest(records):
    digest = hashlib.sha256()
    for severity in (0, 4, 5):
        record = records[severity]
        digest.update(str(record["image_id"]).encode())
        digest.update(str(severity).encode())
        for field in ("logits", "boxes", "persistence"):
            tensor = torch.as_tensor(record[field]).detach().cpu().contiguous()
            digest.update(field.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
            digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class StrongBenchmarkInputs:
    reference_root: Path
    evaluation_root: Path
    annotations_path: Path
    reference_manifest: tuple[ManifestEntry, ...]
    evaluation_manifest: tuple[ManifestEntry, ...]
    reference_cache: Path
    evaluation_caches: dict[str, Path]
    provenance: dict

    def iter_reference_records(self):
        yield from iter_records(self.reference_cache)

    def iter_family_groups(self, family: str):
        if family not in self.evaluation_caches:
            raise ValueError(f"unknown family: {family}")
        grouped: dict[str, dict[int, dict]] = {}
        for record in iter_records(self.evaluation_caches[family]):
            image_id, level = str(record["image_id"]), int(record["severity"])
            if level not in (0, 4, 5):
                continue
            if level in grouped.setdefault(image_id, {}):
                raise ValueError("duplicate image/severity in strong group")
            grouped[image_id][level] = record
        expected_ids = tuple(str(entry.image_id) for entry in self.evaluation_manifest)
        if set(grouped) != set(expected_ids):
            raise ValueError("strong cache image roster does not match evaluation manifest")
        for image_id in expected_ids:
            yield StrongRecordGroup(image_id, family, grouped[image_id])
```

`open_strong_benchmark(reference_root, evaluation_root, annotations_path, config)` must:

```text
1. Resolve regular files/directories without following an unsafe replacement.
2. Load the reference manifest/cache and either its legacy reference binding or
   its completed `strong-cache-manifest.json` from `reference_root`; load the
   evaluation manifest/family caches plus either the legacy corruption roster
   or completed strong-cache manifest from `evaluation_root`, validate their
   exact requested counts, and call `validate_disjoint` across roots.
3. Parse the COCO JSON, require every reference ID to have image width/height,
   and require the annotation file's sorted category IDs to match 80 classes.
4. Validate the legacy reference binding/evaluation roster or strong-cache
   manifest binds the respective
   manifests and the same checkpoint plus extraction-relevant detector fields:
   image size, class count, query count, persistence layer, and persistence
   dimension. Record both source digests; do not reject the existing pair merely
   because downstream legacy bank-capacity fields differ.
5. Require exactly `config.families` in order. A legacy benchmark roster must
   declare `(0, 1, 2, 3, 4, 5)`; a completed strong-cache manifest must declare
   `selected_levels=(0, 4, 5)`. In both cases the analysis panel is `(0, 4, 5)`.
6. Validate reference cache count `reference_count` and each evaluation cache
   count as either `evaluation_count*3` or `evaluation_count*6`.
7. Build provenance containing all source digests plus the annotation SHA-256.
```

For an existing six-level Torch shard, `iter_records` necessarily deserializes its shard container, but no level-1/2/3 record may cross `StrongRecordGroup` or be inspected by matching, distance, aggregation, selection, or reporting. Strong-only three-level caches avoid even that compatibility read. The invariance test above is the executable guarantee that mild tensors cannot affect a result.

- [ ] **Step 4: Run data tests and existing artifact/benchmark tests**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/differential_uncertainty/test_strong_data.py \
  tests/differential_uncertainty/test_artifacts.py \
  tests/differential_uncertainty/test_benchmark.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the authenticated strong-data boundary**

```bash
git add differential_uncertainty/strong_data.py tests/differential_uncertainty/test_strong_data.py tests/differential_uncertainty/conftest.py
git commit -m "feat: load strong corruption cache groups"
```

### Task 5A: Allow a fixed 0/4/5 extraction cache without changing the legacy default

**Files:**
- Modify: `differential_uncertainty/extraction.py:270-315,684-735`
- Modify: `tests/differential_uncertainty/test_extraction.py`

- [ ] **Step 1: Write selected-level extraction tests**

Append:

```python
def test_extract_manifest_can_materialize_only_fixed_strong_levels(cache_inputs, tmp_path):
    extract_manifest(
        cache_inputs.entries, tmp_path / "cache", {"purpose": "strong"},
        cache_inputs.extractor, cache_inputs.corruption,
        image_size=(640, 640), batch_size=2, shard_size=3,
        selected_levels=(0, 4, 5),
    )
    records = list(iter_records(tmp_path / "cache"))
    assert [(row["image_id"], row["severity"]) for row in records] == [
        (entry.image_id, level)
        for entry in cache_inputs.entries for level in (0, 4, 5)
    ]
    assert cache_inputs.corruption.applied_levels == [4, 5] * len(cache_inputs.entries)


def test_extract_manifest_legacy_default_still_uses_all_six_levels(cache_inputs, tmp_path):
    extract_manifest(
        cache_inputs.entries, tmp_path / "cache", {"purpose": "legacy"},
        cache_inputs.extractor, cache_inputs.corruption,
        image_size=(640, 640), batch_size=2, shard_size=6,
    )
    assert {row["severity"] for row in iter_records(tmp_path / "cache")} == set(range(6))


@pytest.mark.parametrize("levels", [(4, 5), (0, 5, 4), (0, 1, 4, 5), [0, 4, 5]])
def test_selected_levels_accepts_only_the_fixed_tuple(cache_inputs, tmp_path, levels):
    with pytest.raises(ValueError, match="selected levels must be exactly"):
        extract_manifest(
            cache_inputs.entries, tmp_path / "cache", {}, cache_inputs.extractor,
            cache_inputs.corruption, image_size=(640, 640), batch_size=1,
            shard_size=3, selected_levels=levels,
        )
```

Also resume the exact three-level cache and assert no extra extractor calls; try to resume it with the legacy default and assert a metadata/record-roster mismatch rather than appending levels 1 through 3.

- [ ] **Step 2: Run the focused tests and verify the keyword is rejected**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/differential_uncertainty/test_extraction.py -k 'selected_levels or strong_levels' -q
```

Expected: failures report `unexpected keyword argument 'selected_levels'`.

- [ ] **Step 3: Add one validated optional keyword at the extraction boundary**

Change only the extraction boundary; leave corruption definitions at all six severities:

```python
def _selected_severities(corruption, selected_levels):
    severities = _validated_severities(corruption)
    if selected_levels is None:
        return severities
    if type(selected_levels) is not tuple or selected_levels != (0, 4, 5):
        raise ValueError("selected levels must be exactly the tuple (0, 4, 5)")
    if corruption is None:
        raise ValueError("clean-only extraction does not accept selected levels")
    by_level = {severity.level: severity for severity in severities}
    return tuple(by_level[level] for level in selected_levels)


def extract_manifest(
    entries, directory, metadata, extractor, corruption, *, image_size,
    batch_size, shard_size, anchored_directory=False,
    selected_levels: tuple[int, ...] | None = None,
):
    # Existing validation remains unchanged up to severity resolution.
    severities = _selected_severities(corruption, selected_levels)
    # Existing canonical-prefix, resume, extraction, and finalize logic follows.
```

Add `"selected_levels": [0, 4, 5]` to the strong materializer's metadata in Task 10. The existing callers omit the keyword and therefore preserve six-level validation byte for byte.

- [ ] **Step 4: Run the complete extraction and legacy pipeline tests**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/differential_uncertainty/test_extraction.py \
  tests/differential_uncertainty/test_pipeline.py \
  tests/differential_uncertainty/test_benchmark.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the strong-only cache primitive**

```bash
git add differential_uncertainty/extraction.py tests/differential_uncertainty/test_extraction.py
git commit -m "feat: support fixed strong-level extraction caches"
```

### Task 6: Compute exact chunked query-to-bank distances

**Files:**
- Create: `differential_uncertainty/strong_scoring.py`
- Create: `tests/differential_uncertainty/test_strong_scoring.py`
- Modify: `tests/differential_uncertainty/conftest.py`
- Read: `differential_uncertainty/scoring.py:90-145`

- [ ] **Step 1: Write hand-calculated distance tests**

Extend `tests/differential_uncertainty/conftest.py` with:

```python
@pytest.fixture
def bank_artifact():
    def make(vectors, *, metric, mean=None, scale=None):
        from differential_uncertainty.strong_bank import BankArtifact
        from differential_uncertainty.strong_policies import BankPolicy, canonical_id
        vectors = torch.as_tensor(vectors, dtype=torch.float32)
        policy = BankPolicy("all_valid", "reservoir", len(vectors), 44)
        identity = canonical_id({
            "fixture": True, "metric": metric,
            "vectors": vectors.tolist(),
            "mean": None if mean is None else torch.as_tensor(mean).tolist(),
            "scale": None if scale is None else torch.as_tensor(scale).tolist(),
        })
        return BankArtifact(
            vectors=vectors, policy=policy, metric=metric,
            reference_mean=None if mean is None else torch.as_tensor(mean).float(),
            reference_scale=None if scale is None else torch.as_tensor(scale).float(),
            metadata={"fixture": True}, artifact_id=identity,
        )
    return make
```

Create `tests/differential_uncertainty/test_strong_scoring.py`:

```python
import torch
import pytest

from differential_uncertainty.strong_scoring import (
    exact_neighbor_distances, reduce_neighbor_distances, score_queries,
)
from differential_uncertainty.strong_policies import (
    AGGREGATION_NAMES, InfeasiblePolicy, QueryDistancePolicy,
)


def test_raw_euclidean_neighbors_are_sorted_and_chunk_exact(bank_artifact):
    bank = bank_artifact(torch.tensor([[1.0], [3.0], [8.0]]), metric="euclidean")
    queries = torch.tensor([[0.0], [10.0]])
    expected = torch.tensor([[1.0, 3.0, 8.0], [2.0, 7.0, 9.0]])
    assert torch.allclose(exact_neighbor_distances(queries, bank, max_k=3, chunk_size=1), expected)
    assert torch.allclose(exact_neighbor_distances(queries, bank, max_k=3, chunk_size=2), expected)


def test_standardized_queries_use_reference_statistics(bank_artifact):
    bank = bank_artifact(
        torch.tensor([[0.0, 0.0], [2.0, 2.0]]),
        metric="standardized_euclidean",
        mean=torch.tensor([10.0, 20.0]), scale=torch.tensor([2.0, 4.0]),
    )
    result = exact_neighbor_distances(torch.tensor([[10.0, 20.0]]), bank, max_k=2)
    assert torch.allclose(result, torch.tensor([[0.0, 8.0**0.5]]), atol=1e-6)


def test_cosine_distance_and_zero_norm_failure(bank_artifact):
    bank = bank_artifact(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), metric="cosine")
    assert torch.allclose(
        exact_neighbor_distances(torch.tensor([[1.0, 1.0]]), bank, max_k=2),
        torch.tensor([[1-2**-0.5, 1-2**-0.5]]), atol=1e-6,
    )
    with pytest.raises(InfeasiblePolicy, match="zero-norm"):
        exact_neighbor_distances(torch.zeros(1, 2), bank, max_k=2)


def test_neighbor_reductions_distinguish_mean_from_radius():
    distances = torch.tensor([[1.0, 2.0, 5.0, 8.0, 9.0]])
    assert reduce_neighbor_distances(distances, QueryDistancePolicy("euclidean", "mean", 5)).item() == 5.0
    assert reduce_neighbor_distances(distances, QueryDistancePolicy("euclidean", "radius", 5)).item() == 9.0
```

```python
def test_distance_validation_rejects_bad_shapes_counts_and_values(bank_artifact):
    bank = bank_artifact(torch.ones(3, 2), metric="euclidean")
    with pytest.raises(ValueError, match="neighbor"):
        exact_neighbor_distances(torch.ones(1, 2), bank, max_k=4)
    with pytest.raises(ValueError, match="finite"):
        exact_neighbor_distances(torch.tensor([[float("nan"), 0.0]]), bank, max_k=1)
    with pytest.raises(ValueError, match="dimensions"):
        exact_neighbor_distances(torch.ones(1, 3), bank, max_k=1)
    with pytest.raises(ValueError, match="chunk"):
        exact_neighbor_distances(torch.ones(1, 2), bank, max_k=1, chunk_size=0)


def test_score_queries_rejects_policy_bank_metric_mismatch(bank_artifact):
    bank = bank_artifact(torch.ones(5, 2), metric="cosine")
    with pytest.raises(ValueError, match="metric"):
        score_queries(
            torch.ones(1, 2), bank,
            QueryDistancePolicy("euclidean", "mean", 5),
        )
```

- [ ] **Step 2: Run the scoring tests and verify the missing module**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_strong_scoring.py -q
```

Expected: collection fails with `ModuleNotFoundError: differential_uncertainty.strong_scoring`.

- [ ] **Step 3: Implement metric transforms, exact top-k merging, and reductions**

Create `differential_uncertainty/strong_scoring.py`:

```python
def _scoring_space(queries: torch.Tensor, bank: BankArtifact):
    queries = queries.detach().float()
    vectors = bank.vectors.to(queries.device)
    if bank.metric == "standardized_euclidean":
        queries = (queries - bank.reference_mean.to(queries.device)) / bank.reference_scale.to(queries.device)
    if bank.metric == "cosine":
        query_norm = torch.linalg.vector_norm(queries, dim=1)
        bank_norm = torch.linalg.vector_norm(vectors, dim=1)
        if (query_norm == 0).any() or (bank_norm == 0).any():
            raise InfeasiblePolicy("cosine distance encountered a zero-norm vector")
        queries = queries / query_norm[:, None]
        vectors = vectors / bank_norm[:, None]
    return queries, vectors


def exact_neighbor_distances(queries, bank, *, max_k, chunk_size=8192):
    queries, vectors = _scoring_space(queries, bank)
    if queries.ndim != 2 or vectors.ndim != 2 or queries.shape[1] != vectors.shape[1]:
        raise ValueError("query and bank dimensions do not match")
    if not torch.isfinite(queries).all() or not torch.isfinite(vectors).all():
        raise ValueError("query and bank vectors must be finite")
    if max_k <= 0 or max_k > vectors.shape[0] or chunk_size <= 0:
        raise ValueError("invalid neighbor or chunk count")
    best = torch.full((queries.shape[0], max_k), float("inf"), device=queries.device)
    for chunk in vectors.split(chunk_size):
        if bank.metric == "cosine":
            local = (1.0 - queries @ chunk.T).clamp(0.0, 2.0)
        else:
            local = torch.cdist(queries, chunk, p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
        best = torch.cat((best, local), dim=1).topk(max_k, dim=1, largest=False, sorted=True).values
    if not torch.isfinite(best).all():
        raise ValueError("query-to-bank distances must be finite")
    return best.cpu()


def reduce_neighbor_distances(neighbors, policy):
    if neighbors.ndim != 2 or neighbors.shape[1] < policy.k:
        raise ValueError("neighbor matrix does not cover requested k")
    selected = neighbors[:, :policy.k]
    return selected.mean(1) if policy.reduction == "mean" else selected[:, -1]
```

Validate that `policy.metric == bank.metric` in a public `score_queries(queries, bank, policy, chunk_size=8192)` wrapper. Compute up to 20 neighbors once per query/bank and derive the k=1/5/10/20 reductions from that single tensor. Mark `k=1` rows `diagnostic=True`; selection code must reject them even when their score is numerically best.

- [ ] **Step 4: Run exact-distance tests**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_strong_scoring.py -q
```

Expected: all distance tests pass.

- [ ] **Step 5: Commit query-to-bank scoring**

```bash
git add differential_uncertainty/strong_scoring.py tests/differential_uncertainty/test_strong_scoring.py tests/differential_uncertainty/conftest.py
git commit -m "feat: score exact strong bank distances"
```

### Task 7: Aggregate query scores and recompute detector-output baselines

**Files:**
- Modify: `differential_uncertainty/strong_scoring.py`
- Modify: `tests/differential_uncertainty/test_strong_scoring.py`
- Modify: `tests/differential_uncertainty/conftest.py`
- Read: `differential_uncertainty/scoring.py:136-225`

- [ ] **Step 1: Add failing aggregation and baseline tests**

Extend `tests/differential_uncertainty/conftest.py` with:

```python
@pytest.fixture
def strong_group():
    class Factory:
        @staticmethod
        def with_valid_query_count(valid_count):
            from differential_uncertainty.strong_data import StrongRecordGroup
            records = {}
            for severity in (0, 4, 5):
                valid = torch.arange(valid_count, dtype=torch.float32)[:, None]
                repeated = torch.full((2, 1), 99.0 + severity)
                scalar = torch.cat((valid + severity, repeated))
                records[severity] = {
                    "image_id": "scene", "severity": severity,
                    "logits": scalar.repeat(1, 2),
                    "boxes": scalar.repeat(1, 4),
                    "persistence": scalar,
                }
            return StrongRecordGroup("scene", "fog", records)
    return Factory()
```

Append:

```python
from differential_uncertainty.strong_scoring import (
    aggregate_image_score, normalized_top_query_entropy, score_strong_group,
)


def test_all_five_aggregators_have_hand_calculated_values():
    query_ids = torch.arange(5)
    distances = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0])
    confidence = torch.tensor([.1, .9, .3, .2, .4])
    assert aggregate_image_score(distances, confidence, query_ids, "mean_all") == pytest.approx(22.0)
    assert aggregate_image_score(distances, confidence, query_ids, "q90_all") == 100.0
    assert aggregate_image_score(distances, confidence, query_ids, "top20_mean_all") == 100.0
    assert aggregate_image_score(distances, confidence, query_ids, "top_confidence_query") == 2.0
    assert aggregate_image_score(distances, confidence, query_ids, "confidence_weighted_mean") == pytest.approx(43.6 / 1.9)


def test_top_confidence_tie_uses_lowest_query_id():
    assert aggregate_image_score(
        torch.tensor([7.0, 3.0]), torch.tensor([.8, .8]), torch.tensor([9, 2]),
        "top_confidence_query",
    ) == 3.0


def test_entropy_uses_softmax_on_the_max_sigmoid_query():
    logits = torch.tensor([[0.0, 0.0], [10.0, -10.0]])
    assert normalized_top_query_entropy(logits, torch.tensor([0, 1])) < 0.001
    assert normalized_top_query_entropy(torch.zeros(2, 2), torch.tensor([0, 1])) == pytest.approx(1.0)


def test_group_uses_one_union_mask_for_zero_four_and_five(strong_group, bank_artifact):
    group = strong_group.with_valid_query_count(10)
    bank = bank_artifact(torch.arange(20, dtype=torch.float32)[:, None], metric="euclidean")
    rows = score_strong_group(group, bank, QueryDistancePolicy("euclidean", "mean", 5))
    assert {row.severity for row in rows} == {0, 4, 5}
    assert {row.aggregation for row in rows} == set(AGGREGATION_NAMES)
    assert len({tuple(row.valid_query_ids) for row in rows}) == 1
    assert all(row.direct_confidence_score == pytest.approx(1-row.direct_confidence_raw) for row in rows)
```

```python
@pytest.mark.parametrize(("distances", "confidence", "message"), [
    (torch.tensor([]), torch.tensor([]), "nonempty"),
    (torch.tensor([float("nan")]), torch.tensor([.5]), "finite"),
    (torch.tensor([1.0]), torch.tensor([float("inf")]), "finite"),
])
def test_aggregation_rejects_empty_or_nonfinite_inputs(distances, confidence, message):
    with pytest.raises(ValueError, match=message):
        aggregate_image_score(distances, confidence, torch.arange(len(distances)), "mean_all")


def test_confidence_weighted_mean_requires_positive_weight_sum():
    with pytest.raises(ValueError, match="positive and finite"):
        aggregate_image_score(
            torch.tensor([1.0, 2.0]), torch.tensor([0.0, 0.0]),
            torch.tensor([0, 1]), "confidence_weighted_mean",
        )


def test_group_requires_ten_valid_queries_for_legacy_diagnostic(strong_group, bank_artifact):
    with pytest.raises(ValueError, match="fewer than ten valid queries"):
        score_strong_group(
            strong_group.with_valid_query_count(9),
            bank_artifact(torch.arange(10, dtype=torch.float32)[:, None], metric="euclidean"),
            QueryDistancePolicy("euclidean", "mean", 5),
        )


def test_minimum_maximum_and_legacy_gap_stay_diagnostic(strong_group, bank_artifact):
    rows = score_strong_group(
        strong_group.with_valid_query_count(10),
        bank_artifact(torch.arange(20, dtype=torch.float32)[:, None], metric="euclidean"),
        QueryDistancePolicy("euclidean", "mean", 5),
    )
    assert all(row.minimum_query_distance <= row.maximum_query_distance for row in rows)
    assert all(math.isfinite(row.legacy_relative_gap) for row in rows)
    assert {"minimum", "maximum", "legacy_relative_gap"}.isdisjoint(AGGREGATION_NAMES)
```

- [ ] **Step 2: Run the new tests and verify the APIs fail**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_strong_scoring.py -q
```

Expected: failures identify the aggregation, entropy, and group-scoring APIs.

- [ ] **Step 3: Implement the five fixed aggregators and baseline conventions**

Add:

```python
def aggregate_image_score(distances, confidence, query_ids, name):
    distances, confidence, query_ids = distances.float(), confidence.float(), query_ids.long()
    if distances.ndim != 1 or confidence.shape != distances.shape or query_ids.shape != distances.shape:
        raise ValueError("query score metadata shape mismatch")
    if not len(distances) or not torch.isfinite(distances).all() or not torch.isfinite(confidence).all():
        raise ValueError("query scores must be nonempty and finite")
    if name == "mean_all":
        value = distances.mean()
    elif name == "q90_all":
        index = math.ceil(0.90 * len(distances)) - 1
        value = distances.sort().values[index]
    elif name == "top20_mean_all":
        count = math.ceil(0.20 * len(distances))
        value = distances.topk(count).values.mean()
    elif name == "top_confidence_query":
        order = sorted(range(len(distances)), key=lambda i: (-float(confidence[i]), int(query_ids[i])))
        value = distances[order[0]]
    elif name == "confidence_weighted_mean":
        total = confidence.sum()
        if not torch.isfinite(total) or float(total) <= 0:
            raise ValueError("confidence weight sum must be positive and finite")
        value = (confidence * distances).sum() / total
    else:
        raise ValueError(f"unknown aggregation: {name}")
    return float(value)


def normalized_top_query_entropy(logits, valid_query_ids):
    confidence = confidence_from_logits(logits)
    top = sorted(valid_query_ids.tolist(), key=lambda q: (-float(confidence[q]), q))[0]
    probability = logits[top].float().softmax(dim=0)
    return float(-torch.xlogy(probability, probability).sum() / math.log(logits.shape[1]))
```

Define `ImageScore` with `image_id`, `family`, `severity`, `bank_artifact_id`, `query_policy`, `aggregation`, `fingerprint_score`, `direct_confidence_raw`, `direct_confidence_score`, `entropy_score`, `valid_query_ids`, `minimum_query_distance`, `maximum_query_distance`, and `legacy_relative_gap`.

`score_strong_group(group, bank, query_policy)` must call `union_padded_query_ids` on exactly the group's severity-0/4/5 records, sort the retained query IDs, require at least ten, and reuse that one ID tensor for every severity and every method. For each severity, compute query distances once, emit five selectable `ImageScore` rows, set `direct_confidence_raw` to the maximum retained maximum-sigmoid confidence, set the corruption-oriented confidence score to `1-direct_confidence_raw`, and compute entropy with the exact top-query convention. Use the existing `confidence_deciles` with deciles 9 and 5 plus `relative_gap` for the diagnostic comparator.

- [ ] **Step 4: Run all new and legacy scoring tests**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/differential_uncertainty/test_strong_scoring.py \
  tests/differential_uncertainty/test_scoring.py \
  tests/differential_uncertainty/test_legacy_parity.py -q
```

Expected: all selected tests pass and legacy six-level scoring remains unchanged.

- [ ] **Step 5: Commit deterministic image scoring**

```bash
git add differential_uncertainty/strong_scoring.py tests/differential_uncertainty/test_strong_scoring.py tests/differential_uncertainty/conftest.py
git commit -m "feat: aggregate strong corruption image scores"
```

### Task 8: Evaluate per-family AUROC, conditional information, and paired intervals

**Files:**
- Create: `differential_uncertainty/strong_evaluation.py`
- Create: `tests/differential_uncertainty/test_strong_evaluation.py`
- Modify: `tests/differential_uncertainty/conftest.py`
- Read: `differential_uncertainty/evaluation.py`

- [ ] **Step 1: Write hand-calculated AUROC and conditioning tests**

Create `tests/differential_uncertainty/test_strong_evaluation.py` around a small `ScoreObservation` factory, with `from dataclasses import replace` and `import pytest`:

```python
from differential_uncertainty.strong_evaluation import (
    _stratum, bootstrap_image_draws, confidence_boundaries,
    conditional_concordance, per_family_aurocs, paired_image_bootstrap,
    summarize_strong,
)


def test_level_four_and_five_are_separate_before_the_strong_mean(observations):
    rows = observations(
        family="fog", clean=[0.0, 1.0], level4=[2.0, 3.0], level5=[0.5, 1.5]
    )
    result = per_family_aurocs(rows, method="fingerprint")["fog"]
    assert result.level4 == 1.0
    assert result.level5 == 0.75
    assert result.strong == 0.875


def test_macro_average_weights_families_not_samples(observations):
    rows = observations(family="fog", clean=[0, 0], level4=[1, 1], level5=[1, 1])
    rows += observations(family="snow", clean=[1] * 20, level4=[0] * 20, level5=[0] * 20)
    summary = summarize_strong(per_family_aurocs(rows, method="fingerprint"))
    assert summary.mean_strong == 0.5
    assert summary.median_strong == 0.5


def test_decile_boundaries_use_training_values_only(observations):
    training = observations.paired_confidence(
        clean=list(range(0, 20, 2)), corrupted=list(range(1, 20, 2)),
    )
    boundaries = confidence_boundaries(training, severity=4)
    assert boundaries == tuple(value + 0.5 for value in range(1, 18, 2))
    held_out = observations.paired_confidence(
        clean=[10_000], corrupted=[20_000],
    )
    assert confidence_boundaries(training, severity=4) == boundaries
    assert held_out  # deliberately never passed to the fitter


def test_conditional_concordance_keeps_only_same_stratum_pairs(observations):
    rows = observations.conditional_pairs(
        clean_confidence=[.11, .19, .50], corrupted_confidence=[.12, .31, .50],
        clean_fingerprint=[1.0, 2.0, 3.0], corrupted_fingerprint=[2.0, 1.0, 3.0],
    )
    result = conditional_concordance(rows, family="fog", severity=4, boundaries=(.2, .4))
    assert result.pair_count == 2
    assert result.concordance == 0.75  # one correct plus one tie


def test_empty_conditional_task_is_recorded_not_imputed(observations):
    rows = observations.conditional_pairs(
        clean_confidence=[.1], corrupted_confidence=[.9],
        clean_fingerprint=[0.0], corrupted_fingerprint=[1.0],
    )
    result = conditional_concordance(rows, family="fog", severity=4, boundaries=(.5,))
    assert result.pair_count == 0
    assert result.concordance is None


def test_paired_bootstrap_of_identical_methods_has_zero_delta(observations):
    rows = observations.two_identical_methods(image_count=8, families=("fog", "snow"))
    interval = paired_image_bootstrap(rows, "left", "right", draws=200, seed=7)
    assert interval.point == 0.0
    assert interval.lower == 0.0
    assert interval.upper == 0.0
```

```python
def test_auroc_ties_count_as_half(observations):
    rows = observations(
        family="fog", clean=[0.0, 1.0], level4=[0.0, 1.0], level5=[0.0, 1.0]
    )
    result = per_family_aurocs(rows, method="fingerprint")["fog"]
    assert result.level4 == .5 and result.level5 == .5


def test_duplicate_missing_class_and_nonfinite_rows_are_rejected(observations):
    rows = observations(family="fog", clean=[0.0, 1.0], level4=[2.0, 3.0], level5=[2.0, 3.0])
    with pytest.raises(ValueError, match="duplicate"):
        per_family_aurocs(rows + [rows[0]], method="fingerprint")
    with pytest.raises(ValueError, match="severity"):
        per_family_aurocs([row for row in rows if row.severity == 0], method="fingerprint")
    with pytest.raises(ValueError, match="finite"):
        per_family_aurocs([replace(rows[0], score=float("nan")), *rows[1:]], method="fingerprint")


def test_repeated_boundaries_collapse_and_boundary_value_enters_higher_bin(observations):
    rows = observations.paired_confidence(
        clean=[.1] * 10, corrupted=[.1] * 5 + [.5] * 5,
    )
    boundaries = confidence_boundaries(rows, severity=4)
    assert boundaries == tuple(dict.fromkeys(boundaries))
    assert _stratum((.2, .4), .4) == 2


def test_bootstrap_draw_moves_complete_image_identity_together():
    draws = bootstrap_image_draws(("a", "b", "c"), draws=4, seed=9)
    assert draws.shape == (4, 3)
    assert set(draws.flatten()) <= {"a", "b", "c"}
    # The bootstrap consumer indexes all family/level/method rows by each value
    # in one draw; it never samples individual observations.
```

- [ ] **Step 2: Run the evaluator tests and verify the import failure**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_strong_evaluation.py -q
```

Expected: collection fails with `ModuleNotFoundError: differential_uncertainty.strong_evaluation`.

- [ ] **Step 3: Implement explicit observation and metric result types**

Create `differential_uncertainty/strong_evaluation.py`:

```python
@dataclass(frozen=True)
class ScoreObservation:
    image_id: str
    family: str
    severity: int
    method: str
    score: float
    direct_confidence_raw: float
    configuration_id: str
    bank_seed: int | None


@dataclass(frozen=True)
class FamilyAuroc:
    family: str
    level4: float
    level5: float
    strong: float


@dataclass(frozen=True)
class StrongSummary:
    mean_level4: float
    mean_level5: float
    mean_strong: float
    median_strong: float


@dataclass(frozen=True)
class ConditionalTask:
    family: str
    severity: int
    concordance: float | None
    pair_count: int
    boundaries: tuple[float, ...]
    empty_strata: tuple[int, ...]


@dataclass(frozen=True)
class Interval:
    point: float
    lower: float
    upper: float
```

Implement strict uniqueness/finite validation first. Then implement AUROC without pooling levels:

```python
def per_family_aurocs(rows, method):
    selected = validate_observations(rows, method=method)
    result = {}
    for family in sorted({row.family for row in selected}):
        family_rows = [row for row in selected if row.family == family]
        clean = [row.score for row in family_rows if row.severity == 0]
        level4 = [row.score for row in family_rows if row.severity == 4]
        level5 = [row.score for row in family_rows if row.severity == 5]
        auc4 = roc_auc_score([0] * len(clean) + [1] * len(level4), clean + level4)
        auc5 = roc_auc_score([0] * len(clean) + [1] * len(level5), clean + level5)
        result[family] = FamilyAuroc(family, auc4, auc5, (auc4 + auc5) / 2)
    return result


def summarize_strong(by_family):
    values = tuple(by_family.values())
    return StrongSummary(
        float(np.mean([v.level4 for v in values])),
        float(np.mean([v.level5 for v in values])),
        float(np.mean([v.strong for v in values])),
        float(np.median([v.strong for v in values])),
    )
```

- [ ] **Step 4: Implement training-only strata and conditional concordance**

Use these exact boundary rules:

```python
def confidence_boundaries(training_rows, severity):
    pairs = paired_clean_corrupted(training_rows, severity)
    pooled = sorted(
        [pair.clean.direct_confidence_raw for pair in pairs]
        + [pair.corrupted.direct_confidence_raw for pair in pairs]
    )
    size = len(pooled)
    boundaries = []
    for decile in range(1, 10):
        lower = math.ceil(decile * size / 10) - 1
        upper = math.ceil(decile * size / 10)
        boundaries.append((pooled[lower] + pooled[upper]) / 2)
    return tuple(dict.fromkeys(boundaries))


def _stratum(boundaries, value):
    return int(np.searchsorted(boundaries, value, side="right"))
```

`conditional_concordance` pairs identical image IDs within one family and severity, retains a pair only when clean and corrupted direct confidence map to the same frozen stratum, counts fingerprint ordering as `1`, ties as `0.5`, and reversals as `0`. Return `None` rather than `0.5` when no pair contributes; record all empty strata. Aggregate by first averaging level 4 and level 5 within each family, then equally averaging the 19 families. Never fit a boundary on an outer/final held-out row.

- [ ] **Step 5: Implement paired image-ID bootstrap**

```python
def bootstrap_image_draws(image_ids, *, draws, seed):
    image_ids = np.asarray(tuple(sorted(image_ids)), dtype=object)
    if image_ids.size == 0 or draws <= 0:
        raise ValueError("bootstrap needs images and a positive draw count")
    indices = np.random.default_rng(seed).integers(
        0, image_ids.size, size=(draws, image_ids.size)
    )
    return image_ids[indices]
```

`paired_image_bootstrap` must use this matrix, bring along every selected method, family, and severity occurrence of each sampled ID, and recompute the requested equal-family statistic. Use the same matrix for fingerprint, confidence, entropy, and both differences. Report the original point estimate and `np.percentile(sampled_statistics, [2.5, 97.5])`; reject a draw lacking either class instead of substituting a value. Public development/final calls use exactly 10,000 draws and seed `20_260_821`; tests may pass a smaller explicit draw count.

- [ ] **Step 6: Run evaluator and legacy-evaluation tests**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/differential_uncertainty/test_strong_evaluation.py \
  tests/differential_uncertainty/test_evaluation.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit strong-corruption evaluation**

```bash
git add differential_uncertainty/strong_evaluation.py tests/differential_uncertainty/test_strong_evaluation.py tests/differential_uncertainty/conftest.py
git commit -m "feat: evaluate strong corruption evidence"
```

### Task 9: Implement leakage-safe staged nested selection and qualification gates

**Files:**
- Create: `differential_uncertainty/strong_selection.py`
- Create: `tests/differential_uncertainty/test_strong_selection.py`
- Modify: `tests/differential_uncertainty/conftest.py`
- Read: `differential_uncertainty/strong_policies.py`
- Read: `differential_uncertainty/strong_evaluation.py`

- [ ] **Step 1: Write fold, isolation, staging, and gate tests**

Create `tests/differential_uncertainty/test_strong_selection.py`:

```python
from collections import Counter
from dataclasses import replace
import pytest

from differential_uncertainty.strong_evaluation import ConditionalTask, Interval
from differential_uncertainty.strong_policies import (
    AGGREGATION_NAMES, BANK_COMPOSITIONS, BankPolicy, CompletePolicy,
    ImageAggregationPolicy, QueryDistancePolicy, STRONG_DEVELOPMENT,
)
from differential_uncertainty.strong_selection import (
    CandidateResult, SelectionPanel, build_outer_cells, fixed_image_folds,
    qualify_finalist, run_nested_selection, rank_key, seed_summary,
)


def test_fixed_folds_are_order_independent_and_balanced():
    image_ids = tuple(f"image-{i}" for i in range(13))
    first = fixed_image_folds(image_ids, fold_count=5)
    second = fixed_image_folds(reversed(image_ids), fold_count=5)
    assert first == second
    assert sorted(Counter(first.values()).values()) == [2, 2, 3, 3, 3]


def test_public_outer_matrix_has_95_family_image_cells():
    cells = build_outer_cells(tuple(f"i{i}" for i in range(250)), STRONG_DEVELOPMENT.families)
    assert len(cells) == 5 * 19
    assert {(cell.image_fold, cell.held_out_family) for cell in cells} == {
        (fold, family) for fold in range(5) for family in STRONG_DEVELOPMENT.families
    }


def test_outer_scores_cannot_change_inner_policy_choice(synthetic_provider):
    panel = SelectionPanel.for_tests()
    base_provider = synthetic_provider()
    changed_provider = synthetic_provider(outer_score_offset=1_000_000)
    base = run_nested_selection(base_provider, panel)
    changed = run_nested_selection(changed_provider, panel)
    assert [cell.selected_policy for cell in base.outer_cells] == [
        cell.selected_policy for cell in changed.outer_cells
    ]
    assert base_provider.requested_outer_rows_during_inner_selection == set()
    assert changed_provider.requested_outer_rows_during_inner_selection == set()


def test_staged_selector_recovers_known_winner_and_records_every_screen_arm(synthetic_provider):
    provider = synthetic_provider(winner="balanced-cosine-q90")
    result = run_nested_selection(provider, SelectionPanel.for_tests())
    assert result.final_decision.selected_policy.policy_id == provider.winner_id
    assert len(result.stage1_audit) == 5 * 5
    assert {row.composition for row in result.stage1_audit} == set(BANK_COMPOSITIONS)
    assert {row.aggregation for row in result.stage1_audit} == set(AGGREGATION_NAMES)


@pytest.fixture
def candidate_result():
    def make(
        name, *, strong=.7, strong_lower=.55, conditional=.7,
        conditional_lower=.55, minimum_seed=.6, capacity=2000,
        level4=.7, level5=.7, median=.7,
        seed_scores=None, diagnostic=False, empty_task=False,
        missing_task=False,
    ):
        policy = CompletePolicy(
            BankPolicy("all_valid", "reservoir", capacity, 44),
            QueryDistancePolicy("euclidean", "mean", 5),
            ImageAggregationPolicy("mean_all"),
        )
        tasks = tuple(
            ConditionalTask(
                family=f"family-{family}", severity=severity,
                concordance=None if empty_task and family == 0 and severity == 4 else conditional,
                pair_count=0 if empty_task and family == 0 and severity == 4 else 10,
                boundaries=(), empty_strata=(),
            )
            for family in range(19) for severity in (4, 5)
        )
        if missing_task:
            tasks = tasks[:-1]
        scores = seed_scores or (minimum_seed, strong, strong, strong, strong)
        return CandidateResult(
            configuration_id=name, policy=policy,
            mean_level4=level4, mean_level5=level5,
            mean_strong=strong, median_strong=median,
            conditional=conditional,
            strong_interval=Interval(strong, strong_lower, min(1.0, strong + .1)),
            conditional_interval=Interval(
                conditional, conditional_lower, min(1.0, conditional + .1)
            ) if conditional is not None else None,
            seed_scores=tuple(scores), conditional_tasks=tasks,
            diagnostic=diagnostic, feasible=True, validation_cells=(),
        )
    return make


def test_rank_ties_use_conditional_then_seed_floor_capacity_and_id(candidate_result):
    ordered = sorted([
        candidate_result("z", strong=.7, conditional=.6, minimum_seed=.55, capacity=1000),
        candidate_result("a", strong=.7, conditional=.7, minimum_seed=.51, capacity=5000),
        candidate_result("b", strong=.7, conditional=.7, minimum_seed=.60, capacity=5000),
        candidate_result("c", strong=.7, conditional=.7, minimum_seed=.60, capacity=2000),
    ], key=rank_key)
    assert [row.configuration_id for row in ordered] == ["c", "b", "a", "z"]


def test_k1_and_failed_gates_can_never_produce_a_frozen_policy(candidate_result):
    diagnostic = candidate_result("k1", diagnostic=True, strong=.99, conditional=.99)
    assert qualify_finalist(diagnostic).qualified is False
    failed = candidate_result(
        "weak", strong=.62, strong_lower=.51, conditional=.55,
        conditional_lower=.49, level4=.63, level5=.61, median=.62,
        seed_scores=(.61, .62, .63, .64, .65),
    )
    assert qualify_finalist(failed).failed_gates == ("conditional_interval",)
```

```python
@pytest.mark.parametrize(("changes", "gate"), [
    ({"strong_lower": .50}, "strong_interval"),
    ({"level4": .50}, "level4"),
    ({"level5": .50}, "level5"),
    ({"median": .50}, "family_median"),
    ({"seed_scores": (.7, .7, .5, .7, .7)}, "seed_stability"),
    ({"empty_task": True}, "empty_conditional_task"),
    ({"missing_task": True}, "conditional_task_count"),
])
def test_each_qualification_gate_is_independently_enforced(candidate_result, changes, gate):
    decision = qualify_finalist(candidate_result("candidate", **changes))
    assert decision.qualified is False
    assert gate in decision.failed_gates


def test_seed_summary_uses_all_five_points_not_the_best_seed():
    unstable = seed_summary((.99, .49, .49, .49, .49))
    stable = seed_summary((.60, .60, .60, .60, .60))
    assert unstable.mean < stable.mean
    assert unstable.maximum > stable.maximum


def test_infeasible_inner_cell_and_no_qualifier_are_never_promoted(synthetic_provider):
    infeasible = run_nested_selection(
        synthetic_provider(infeasible_cell=(0, "fog")), SelectionPanel.for_tests()
    )
    assert all(
        row.configuration_id not in infeasible.promoted_configuration_ids
        for row in infeasible.infeasible_rows
    )
    weak = run_nested_selection(
        synthetic_provider(all_candidates_below_chance=True), SelectionPanel.for_tests()
    )
    assert weak.final_decision.selected_policy is None
    assert weak.final_decision.status == "no_qualifying_policy"
```

- [ ] **Step 2: Run the selector tests and verify the missing module**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_strong_selection.py -q
```

Expected: collection fails with `ModuleNotFoundError: differential_uncertainty.strong_selection`.

- [ ] **Step 3: Define deterministic folds, cells, and injectable panels**

Create `differential_uncertainty/strong_selection.py`:

```python
@dataclass(frozen=True)
class ValidationCell:
    image_fold: int
    held_out_family: str
    training_image_ids: tuple[str, ...]
    held_out_image_ids: tuple[str, ...]
    training_families: tuple[str, ...]


def fixed_image_folds(image_ids, fold_count=5):
    normalized = tuple(str(value) for value in image_ids)
    unique = tuple(sorted(set(normalized)))
    if len(unique) != len(normalized):
        raise ValueError("image IDs must be unique")
    ordered = sorted(
        unique,
        key=lambda value: (hashlib.sha256(f"strong-fold:{value}".encode()).digest(), value),
    )
    return {image_id: index % fold_count for index, image_id in enumerate(ordered)}


def build_outer_cells(image_ids, families, fold_count=5):
    folds = fixed_image_folds(image_ids, fold_count)
    cells = []
    for fold in range(fold_count):
        for family in families:
            cells.append(ValidationCell(
                fold, family,
                tuple(sorted(i for i in folds if folds[i] != fold)),
                tuple(sorted(i for i in folds if folds[i] == fold)),
                tuple(item for item in families if item != family),
            ))
    return tuple(cells)
```

Define the injectable panel and provider contract:

```python
@dataclass(frozen=True)
class SelectionPanel:
    families: tuple[str, ...]
    fold_count: int
    screen_capacity: int
    screen_seed: int
    compositions: tuple[str, ...]
    constructions: tuple[str, ...]
    capacities: tuple[int, ...]
    query_policies: tuple[QueryDistancePolicy, ...]
    neighbor_counts: tuple[int, ...]
    aggregations: tuple[str, ...]
    seeds: tuple[int, ...]

    @classmethod
    def public(cls):
        return cls(
            families=STRONG_FAMILIES, fold_count=5,
            screen_capacity=2_000, screen_seed=44,
            compositions=BANK_COMPOSITIONS,
            constructions=BANK_CONSTRUCTIONS,
            capacities=BANK_CAPACITIES,
            query_policies=QUERY_DISTANCE_PANEL,
            neighbor_counts=(1, 5, 10, 20),
            aggregations=AGGREGATION_NAMES, seeds=BANK_SEEDS,
        )

    @classmethod
    def for_tests(cls):
        return cls(
            families=("fog", "snow"), fold_count=2,
            screen_capacity=5, screen_seed=44,
            compositions=BANK_COMPOSITIONS,
            constructions=BANK_CONSTRUCTIONS,
            capacities=(5, 10), query_policies=QUERY_DISTANCE_PANEL,
            neighbor_counts=(1, 5, 10),
            aggregations=AGGREGATION_NAMES, seeds=(42, 44),
        )


class ScoreProvider(Protocol):
    def observations(
        self, policy: CompletePolicy, *, image_ids: tuple[str, ...],
        families: tuple[str, ...], bank_seeds: tuple[int, ...],
    ) -> tuple[ScoreObservation, ...]:
        raise NotImplementedError
```

The production provider in Task 10 will lazily materialize any requested legal arm; the synthetic provider logs each request so isolation is testable.

- [ ] **Step 4: Implement one split evaluator and the deterministic ranking key**

For each candidate, concatenate only held-out observations from all inner cells, while fitting confidence boundaries from that inner cell's training rows. Produce `CandidateResult` with equal-family level-4/level-5/strong statistics, conditional concordance and exactly `2 * len(panel.families)` task rows (38 public), per-seed strong scores, feasibility, and the complete validation-cell list.

```python
@dataclass(frozen=True)
class CandidateResult:
    configuration_id: str
    policy: CompletePolicy
    mean_level4: float
    mean_level5: float
    mean_strong: float
    median_strong: float
    conditional: float | None
    strong_interval: Interval
    conditional_interval: Interval | None
    seed_scores: tuple[float, ...]
    conditional_tasks: tuple[ConditionalTask, ...]
    diagnostic: bool
    feasible: bool
    validation_cells: tuple[ValidationCell, ...]


@dataclass(frozen=True)
class Qualification:
    qualified: bool
    failed_gates: tuple[str, ...]


@dataclass(frozen=True)
class FinalDecision:
    status: str
    selected_policy: CompletePolicy | None
    qualification: Qualification
    confidence_boundaries: dict[int, tuple[float, ...]]


@dataclass(frozen=True)
class OuterCellResult:
    cell: ValidationCell
    selected_policy: CompletePolicy
    held_out_observations: tuple[ScoreObservation, ...]


@dataclass(frozen=True)
class StageAuditRow:
    stage: str
    composition: str
    aggregation: str
    candidate: CandidateResult


@dataclass(frozen=True)
class NestedSelectionResult:
    outer_cells: tuple[OuterCellResult, ...]
    stage1_audit: tuple[StageAuditRow, ...]
    infeasible_rows: tuple[CandidateResult, ...]
    promoted_configuration_ids: tuple[str, ...]
    final_decision: FinalDecision
    report_payload: dict
    selection_trace: dict


@dataclass(frozen=True)
class SeedSummary:
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float


def seed_summary(values):
    values = np.asarray(tuple(values), dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("seed summary needs a nonempty finite vector")
    return SeedSummary(
        float(values.mean()), float(values.std(ddof=0)),
        float(values.min()), float(values.max()),
    )


def rank_key(result):
    conditional = -math.inf if result.conditional is None else result.conditional
    seed_floor = -math.inf if not result.seed_scores else min(result.seed_scores)
    return (
        -result.mean_strong,
        -conditional,
        -seed_floor,
        result.policy.bank.capacity,
        result.configuration_id,
    )
```

Never call `ScoreProvider.observations` with an outer cell's held-out image IDs or held-out family until the inner selector returns a complete frozen policy for that cell. Then request only that policy's outer predictions and append them to the nested held-out record.

- [ ] **Step 5: Implement the exact staged candidate transition**

For each outer training partition—and later once more over all nested held-out development predictions—execute these immutable transitions:

```text
Stage 1: 5 compositions x 5 aggregators; reservoir, panel.screen_capacity
         (2,000 public), panel.screen_seed (44 public), mean-5 raw Euclidean.
         Keep all 25 audit rows. Select the best
         aggregator within each composition, then advance the best 2 compositions.

Stage 2a: for each advancing composition, reservoir versus global k-means at
          panel.screen_capacity over panel.seeds (2,000 and 42..46 public);
          allow all 5 aggregators and rank
          seed means, never individual best seeds.

Stage 2b: for the winning construction of each composition, compare capacities
          in panel.capacities (1,000, 2,000, 5,000 public) over panel.seeds and
          all 5 aggregators. Keep one
          construction/capacity result for each of the 2 compositions.

Stage 3: cross those 2 bank policies with panel.query_policies (the 4 fixed
         public query-distance policies), panel.seeds, and all 5 aggregators.
         Keep the best complete result for
         each bank policy, then rank the two.

Stage 4: for each leading metric/reduction family, evaluate panel.neighbor_counts
         (k=1,5,10,20 public) using the same bank policies, seeds, and aggregators.
         Record k=1 but remove it before ranking; public k=5,10,20 remain
         selectable. Select one tuple.
```

A candidate infeasible in any required inner cell is recorded with the exact reason and cannot advance. Use `rank_key` for every selection. Retain the complete list of evaluated IDs, metrics, folds, feasibility reasons, promotions, and tie-break fields in `SelectionTrace`; lazy backfill any legal arm required by the all-development replay so final selection does not depend on which arms happened to advance in a particular outer cell.

- [ ] **Step 6: Implement qualification and separate evidence statements**

```python
def qualify_finalist(result, *, expected_seed_count=5, expected_task_count=38):
    failures = []
    if result.diagnostic:
        failures.append("diagnostic_only")
    if result.strong_interval.lower <= 0.5:
        failures.append("strong_interval")
    if result.conditional_interval is None or result.conditional_interval.lower <= 0.5:
        failures.append("conditional_interval")
    if result.mean_level4 <= 0.5:
        failures.append("level4")
    if result.mean_level5 <= 0.5:
        failures.append("level5")
    if result.median_strong <= 0.5:
        failures.append("family_median")
    if len(result.seed_scores) != expected_seed_count or min(result.seed_scores) <= 0.5:
        failures.append("seed_stability")
    if len(result.conditional_tasks) != expected_task_count:
        failures.append("conditional_task_count")
    if any(task.pair_count == 0 for task in result.conditional_tasks):
        failures.append("empty_conditional_task")
    return Qualification(not failures, tuple(failures))
```

Among qualifiers, use `rank_key`; if none qualifies, return `selected_policy=None` and never create a frozen-policy artifact. Record four independent booleans/statements: fingerprint AUROC interval above chance; conditional-concordance interval above chance; paired fingerprint-minus-confidence interval above zero; and every seed above chance with spread. Do not make beating confidence a qualification gate and do not merge these claims.

`run_nested_selection` passes `expected_seed_count=len(panel.seeds)` and `expected_task_count=2 * len(panel.families)`; the public panel therefore requires exactly five seeds and 38 conditional tasks, while miniature tests exercise the same logic with two seeds and four tasks.

- [ ] **Step 7: Run selector tests**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/differential_uncertainty/test_strong_selection.py \
  tests/differential_uncertainty/test_strong_evaluation.py -q
```

Expected: all selection and evaluation tests pass.

- [ ] **Step 8: Commit nested selection**

```bash
git add differential_uncertainty/strong_selection.py tests/differential_uncertainty/test_strong_selection.py tests/differential_uncertainty/conftest.py
git commit -m "feat: select strong corruption policy without leakage"
```

### Task 10: Orchestrate resumable development and frozen final runs

**Files:**
- Create: `differential_uncertainty/strong_pipeline.py`
- Create: `tests/differential_uncertainty/test_strong_pipeline.py`
- Modify: `tests/differential_uncertainty/conftest.py`
- Read: `differential_uncertainty/artifacts.py:480-620`
- Read: `differential_uncertainty/artifacts.py:1845-1915`
- Read: `differential_uncertainty/cli.py:200-440`

- [ ] **Step 1: Write lazy-materialization and resume tests**

Create `tests/differential_uncertainty/test_strong_pipeline.py` using the two-family cache fixture from Task 5 and an injected miniature `SelectionPanel`:

```python
import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch

from differential_uncertainty.artifacts import iter_records
from differential_uncertainty.strong_pipeline import (
    develop_strong_coco, evaluate_frozen_strong_coco,
    materialize_strong_coco_cache,
)
from differential_uncertainty.strong_policies import STRONG_FINAL
from differential_uncertainty.strong_selection import SelectionPanel


def file_sha256_and_mtime(root):
    root = Path(root)
    return {
        str(path.relative_to(root)): (
            hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns,
        )
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def test_development_builds_matching_once_and_only_requested_score_arms(strong_cache, counters, tmp_path):
    result = develop_strong_coco(
        strong_cache.reference_root, strong_cache.evaluation_root,
        strong_cache.annotations, tmp_path / "study",
        device="cpu", config=strong_cache.config, panel=SelectionPanel.for_tests(),
        hooks=counters,
    )
    assert counters.matching_builds == 1
    assert counters.candidate_pool_builds == 1
    assert counters.scored_levels == {0, 4, 5}
    assert counters.scored_levels.isdisjoint({1, 2, 3})
    assert result.development_path.is_file()


def test_strong_cache_materializer_never_extracts_mild_levels(coco_fixture, tmp_path):
    output = materialize_strong_coco_cache(
        coco_fixture.annotations, coco_fixture.images, coco_fixture.checkpoint,
        tmp_path / "cache", mode="development", device="cpu",
        batch_size=2, shard_size=3,
        study_config=coco_fixture.study_config,
        detector_config=coco_fixture.detector_config,
        corruptions=coco_fixture.corruptions,
        extractor_factory=coco_fixture.extractor_factory,
    )
    for cache in output.glob("corruptions/*/artifacts/evaluation-extractions"):
        records = list(iter_records(cache))
        assert {record["severity"] for record in records} == {0, 4, 5}
        assert len(records) == 3 * coco_fixture.study_config.evaluation_count


def test_exact_resume_reuses_artifacts_and_policy_mismatch_refuses(strong_cache, counters, tmp_path):
    output = tmp_path / "study"
    first = develop_strong_coco(
        strong_cache.reference_root, strong_cache.evaluation_root,
        strong_cache.annotations, output,
        device="cpu", config=strong_cache.config, panel=SelectionPanel.for_tests(), hooks=counters,
    )
    snapshots = file_sha256_and_mtime(output / "artifacts")
    second = develop_strong_coco(
        strong_cache.reference_root, strong_cache.evaluation_root,
        strong_cache.annotations, output,
        device="cpu", config=strong_cache.config, panel=SelectionPanel.for_tests(), hooks=counters,
    )
    assert first == second
    assert file_sha256_and_mtime(output / "artifacts") == snapshots
    with pytest.raises(ValueError, match="provenance mismatch"):
        develop_strong_coco(
            strong_cache.reference_root, strong_cache.changed_evaluation_root,
            strong_cache.annotations, output,
            device="cpu", config=strong_cache.config, panel=SelectionPanel.for_tests(), hooks=counters,
        )


def test_failed_qualification_never_writes_frozen_policy(strong_cache, tmp_path):
    result = develop_strong_coco(
        strong_cache.reference_root, strong_cache.weak_evaluation_root,
        strong_cache.annotations, tmp_path / "study",
        device="cpu", config=strong_cache.config, panel=SelectionPanel.for_tests(),
    )
    assert result.selected_policy is None
    assert not (tmp_path / "study" / "frozen-policy.json").exists()


def test_final_rebuilds_seed44_bank_and_reuses_development_boundaries(final_cache, qualified_development, tmp_path):
    result = evaluate_frozen_strong_coco(
        final_cache.reference_root, final_cache.evaluation_root,
        final_cache.annotations, tmp_path / "final",
        frozen_policy_path=qualified_development.frozen_policy,
        development_path=qualified_development.summary,
        device="cpu", config=final_cache.config,
        inputs_factory=final_cache.inputs_factory,
        provider_factory=final_cache.provider_factory,
    )
    payload = json.loads((result.output / "final.json").read_text())
    assert result.selected_policy.bank.seed == 44
    assert payload["study"]["reference_count"] == 2_500
    assert payload["study"]["evaluation_count"] == 2_500
    assert {
        int(level): tuple(values)
        for level, values in payload["frozen"]["confidence_boundaries"].items()
    } == qualified_development.confidence_boundaries
    assert final_cache.requested_bank_seeds == [44]


def test_public_final_configuration_rejects_test_factories(final_cache, qualified_development, tmp_path):
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="test-only"):
        evaluate_frozen_strong_coco(
            final_cache.reference_root, final_cache.evaluation_root,
            final_cache.annotations, output,
            frozen_policy_path=qualified_development.frozen_policy,
            development_path=qualified_development.summary,
            device="cpu", config=STRONG_FINAL,
            inputs_factory=final_cache.inputs_factory,
            provider_factory=final_cache.provider_factory,
        )
    assert not output.exists()
```

The `final_cache` fixture is manifest-only, declares the fixed counts, exposes explicit `inputs_factory` and `provider_factory` test seams, records every requested bank seed, and returns a complete 19-family observation roster from the provider, so it does not allocate 5,000 images. Its config comes from `StrongStudyConfig.for_tests(reference_count=2_500, evaluation_count=2_500, families=STRONG_FAMILIES)`, which is the only reason non-default factories are accepted. Add these concrete failure/resume cases:

```python
def test_interrupted_family_scoring_resumes_completed_artifacts(strong_cache, counters, tmp_path):
    output = tmp_path / "study"
    counters.raise_after_family = "fog"
    with pytest.raises(RuntimeError, match="injected interruption"):
        develop_strong_coco(
            strong_cache.reference_root, strong_cache.evaluation_root,
            strong_cache.annotations, output, device="cpu",
            config=strong_cache.config, panel=SelectionPanel.for_tests(), hooks=counters,
        )
    completed = file_sha256_and_mtime(output / "artifacts")
    counters.raise_after_family = None
    develop_strong_coco(
        strong_cache.reference_root, strong_cache.evaluation_root,
        strong_cache.annotations, output, device="cpu",
        config=strong_cache.config, panel=SelectionPanel.for_tests(), hooks=counters,
    )
    assert completed.items() <= file_sha256_and_mtime(output / "artifacts").items()


def test_tampered_score_tensor_fails_digest_validation(strong_cache, completed_study):
    path = next((completed_study / "artifacts/query-scores").rglob("*.pt"))
    payload = torch.load(path, weights_only=True)
    payload["distances"][0, 0] += 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="digest"):
        develop_strong_coco(
            strong_cache.reference_root, strong_cache.evaluation_root,
            strong_cache.annotations, completed_study, device="cpu",
            config=strong_cache.config, panel=SelectionPanel.for_tests(),
        )


@pytest.mark.parametrize("fault", ["not_qualified", "policy", "confidence_boundaries"])
def test_final_refuses_development_or_frozen_mismatch_before_output(
    final_cache, qualified_development, tmp_path, fault,
):
    frozen, development = qualified_development.with_fault(fault)
    output = tmp_path / "final"
    with pytest.raises(ValueError, match="frozen|qualified|boundaries"):
        evaluate_frozen_strong_coco(
            final_cache.reference_root, final_cache.evaluation_root,
            final_cache.annotations, output,
            frozen_policy_path=frozen, development_path=development,
            device="cpu", config=final_cache.config,
            inputs_factory=final_cache.inputs_factory,
            provider_factory=final_cache.provider_factory,
        )
    assert not output.exists()
```

- [ ] **Step 2: Run pipeline tests and verify the missing module**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_strong_pipeline.py -q
```

Expected: collection fails with `ModuleNotFoundError: differential_uncertainty.strong_pipeline`.

- [ ] **Step 3: Implement immutable artifact identities and the lazy score provider**

Create `differential_uncertainty/strong_pipeline.py`. Define:

```python
@dataclass(frozen=True)
class StrongRunResult:
    output: Path
    development_path: Path | None
    frozen_policy_path: Path | None
    report_directory: Path
    selected_policy: CompletePolicy | None
    provenance: dict


def artifact_identity(kind, source_provenance, policy=None, extra=None):
    return canonical_id({
        "schema_version": 1,
        "kind": kind,
        "source": source_provenance,
        "policy": None if policy is None else policy.to_dict(),
        "extra": extra or {},
    })
```

The root provenance passed to `ensure_provenance` must contain schema version, mode (`development` or `final`), both source roots, fixed study config, reference/evaluation manifest digests, annotation digest, checkpoint digest, extraction-relevant detector scientific config, both historical source digests, family roster, level tuple `(0,4,5)`, and runtime device/chunk settings. Also expose `reference_count`, `evaluation_count`, `families`, and `levels` as validated top-level fields; final provenance additionally exposes `bank_seed`, `policy_id`, and `development_identity`. These are the exact fields copied to the report's `provenance.json`. Bind runtime to computed artifact identities so a resume cannot mix CPU/GPU numerics, while keeping the scientific `CompletePolicy.policy_id` independent of runtime.

Implement `LazyBankScoreProvider(ScoreProvider)` with these memoized layers:

```text
matching:       reference source + annotation digest + historical cost policy
candidate pool: matching identity + reference cache manifest digest
bank:           candidate identity + BankPolicy + scoring metric
query scores:   bank identity + QueryDistancePolicy + family cache digest
image scores:   query-score identity + all five fixed aggregators + union-mask policy
```

At each layer: calculate the target identity first; if the artifact exists, safely load and validate its complete metadata; if absent, compute once and atomically publish; if a path exists with any mismatch, raise rather than overwrite. The provider returns `ScoreObservation` rows and caches all five aggregators from one query-distance pass. It catches only `InfeasiblePolicy`, records the arm/reason, and propagates corruption, I/O, provenance, or nonfinite errors.

- [ ] **Step 4: Implement the fixed strong-only cache materializer**

Add:

```python
def materialize_strong_coco_cache(
    coco_annotations, coco_images, checkpoint, output_dir, *, mode,
    device, batch_size, shard_size, study_config=None,
    detector_config=FIXED_CONFIG, corruptions=None,
    extractor_factory=RTDETRExtractor,
):
    if mode not in {"development", "final"}:
        raise ValueError("strong cache mode must be development or final")
    fixed = STRONG_DEVELOPMENT if mode == "development" else STRONG_FINAL
    config = fixed if study_config is None else study_config
    split = create_coco_manifests(
        coco_annotations, coco_images, Path(output_dir) / "inputs",
        reference_count=config.reference_count,
        evaluation_count=config.evaluation_count,
    )
    reference = load_manifest(split.reference_manifest)
    evaluation = load_manifest(split.evaluation_manifest)
    validate_disjoint(reference, evaluation)
    families = tuple(benchmark_corruptions()) if corruptions is None else tuple(corruptions)
    if tuple(item.name for item in families) != config.families:
        raise ValueError("strong cache family roster mismatch")
    metadata = strong_cache_provenance(
        split, checkpoint, coco_annotations, config, detector_config,
        device=device, batch_size=batch_size, shard_size=shard_size,
    )
    ensure_provenance(output_dir, metadata)
    with extractor_factory(checkpoint, device, detector_config) as extractor:
        extract_manifest(
            reference,
            Path(output_dir) / "reference-artifacts/reference-extractions",
            {**metadata, "stage": "reference"}, extractor, None,
            image_size=detector_config.image_size, batch_size=batch_size,
            shard_size=shard_size,
        )
        for corruption in families:
            extract_manifest(
                evaluation,
                Path(output_dir) / "corruptions" / corruption.name
                / "artifacts/evaluation-extractions",
                {**metadata, "stage": "evaluation", "corruption": corruption.name,
                 "selected_levels": [0, 4, 5]},
                extractor, corruption,
                image_size=detector_config.image_size, batch_size=batch_size,
                shard_size=shard_size, selected_levels=(0, 4, 5),
            )
    publish_exact_json(
        Path(output_dir) / "strong-cache-manifest.json",
        completed_strong_cache_manifest(output_dir, metadata),
    )
    return Path(output_dir)
```

Reuse the existing coordinated-directory and regular-file validation helpers around the skeleton above; authenticate every completed cache manifest/shard before publishing `strong-cache-manifest.json`. On resume, require exact source manifests, image fingerprints, checkpoint, detector configuration, family order, runtime, and selected levels. The public mode fixes counts and never accepts a level list from CLI.

- [ ] **Step 5: Implement development orchestration and the frozen record**

```python
def validated_development_record(selection, provenance, config):
    decision = selection.final_decision
    selected = decision.selected_policy
    return {
        "schema_version": 1,
        "mode": "development",
        "study": {
            "reference_count": config.reference_count,
            "evaluation_count": config.evaluation_count,
            "families": list(config.families),
            "levels": list(config.levels),
            "outer_cell_count": len(selection.outer_cells),
            "bootstrap_draws": config.bootstrap_draws,
            "bootstrap_seed": config.bootstrap_seed,
        },
        "provenance": provenance,
        "selection": {
            "status": decision.status,
            "selected_policy_id": None if selected is None else selected.policy_id,
            "selected_policy": None if selected is None else selected.to_dict(),
            "qualification": asdict(decision.qualification),
            "confidence_boundaries": {
                str(level): list(values)
                for level, values in decision.confidence_boundaries.items()
            },
        },
        "results": selection.report_payload,
        "selection_trace": selection.selection_trace,
    }


def develop_strong_coco(
    reference_run, evaluation_benchmark, coco_annotations, output_dir, *, device,
    config=STRONG_DEVELOPMENT, panel=None, hooks=None,
):
    output = Path(output_dir)
    inputs = open_strong_benchmark(
        reference_run, evaluation_benchmark, coco_annotations, config
    )
    provenance = development_provenance(inputs, config)
    ensure_provenance(output, provenance)
    provider = LazyBankScoreProvider(inputs, output, device=device, hooks=hooks)
    selection = run_nested_selection(provider, panel or SelectionPanel.public())
    development = validated_development_record(selection, provenance, config)
    publish_exact_json(output / "development.json", development)
    if selection.final_decision.selected_policy is not None:
        frozen = {
            "schema_version": 1,
            "qualified": True,
            "development_identity": canonical_id(development),
            "policy": replace(
                selection.final_decision.selected_policy,
                bank=replace(selection.final_decision.selected_policy.bank, seed=44),
            ).to_dict(),
            "confidence_boundaries": selection.final_decision.confidence_boundaries,
            "qualification": asdict(selection.final_decision.qualification),
        }
        publish_exact_json(output / "frozen-policy.json", frozen)
    write_strong_report(output / "reports", development)
    frozen_path = output / "frozen-policy.json"
    return StrongRunResult(
        output=output,
        development_path=output / "development.json",
        frozen_policy_path=frozen_path if frozen_path.is_file() else None,
        report_directory=output / "reports",
        selected_policy=selection.final_decision.selected_policy,
        provenance=provenance,
    )
```

`publish_exact_json` uses `atomic_json` when absent; on resume it safely reads and requires JSON-semantic equality. A no-qualifier run writes `development.json` and the full negative report, but never an empty or fallback `frozen-policy.json`.

- [ ] **Step 6: Implement the one-shot untouched final evaluation**

Implement this fixed path; `validate_frozen_pair` requires the frozen record's `development_identity` to equal `canonical_id(development)`, all gates and 38 conditional tasks to be valid, and bank seed 44:

```python
def evaluate_frozen_strong_coco(
    reference_run, evaluation_benchmark, coco_annotations, output_dir, *,
    frozen_policy_path, development_path, device, config=STRONG_FINAL,
    inputs_factory=open_strong_benchmark,
    provider_factory=LazyBankScoreProvider,
):
    output = Path(output_dir)
    if not config.testing and (
        inputs_factory is not open_strong_benchmark
        or provider_factory is not LazyBankScoreProvider
    ):
        raise ValueError("dependency factories are test-only")
    frozen = load_exact_json(frozen_policy_path, label="frozen policy")
    development = load_exact_json(development_path, label="development summary")
    validate_frozen_pair(frozen, development)
    policy = CompletePolicy.from_dict(frozen["policy"])
    if policy.bank.seed != 44:
        raise ValueError("final bank seed must be 44")
    inputs = inputs_factory(
        reference_run, evaluation_benchmark, coco_annotations, config
    )
    provenance = final_provenance(
        inputs, config, development_identity=frozen["development_identity"],
        policy_id=policy.policy_id,
    )
    ensure_provenance(output, provenance)
    provider = provider_factory(inputs, output, device=device)
    observations = provider.observations(
        policy,
        image_ids=tuple(str(entry.image_id) for entry in inputs.evaluation_manifest),
        families=config.families,
        bank_seeds=(44,),
    )
    results = evaluate_frozen_policy(
        observations, policy=policy,
        confidence_boundaries={
            int(level): tuple(values)
            for level, values in frozen["confidence_boundaries"].items()
        },
        bootstrap_draws=config.bootstrap_draws,
        bootstrap_seed=config.bootstrap_seed,
    )
    final = {
        "schema_version": 1, "mode": "final",
        "study": {
            "reference_count": config.reference_count,
            "evaluation_count": config.evaluation_count,
            "families": list(config.families), "levels": list(config.levels),
            "bootstrap_draws": config.bootstrap_draws,
            "bootstrap_seed": config.bootstrap_seed,
        },
        "provenance": provenance,
        "frozen": frozen,
        "results": results,
    }
    publish_exact_json(output / "final.json", final)
    write_strong_report(output / "reports", final)
    return StrongRunResult(
        output=output, development_path=None, frozen_policy_path=Path(frozen_policy_path),
        report_directory=output / "reports", selected_policy=policy,
        provenance=provenance,
    )
```

The public CLI supplies neither factory argument and uses `STRONG_FINAL`, so the manifests must contain exactly 2,500 reference and 2,500 evaluation IDs. The two factories are dependency seams for unit tests only; reject non-default factories unless `config.testing` is true. Rebuild the bank from those final clean candidates using only the frozen tuple. This function must not import or call `run_nested_selection`, enumerate alternatives, refit confidence boundaries, or inspect final results before policy construction.

- [ ] **Step 7: Run pipeline tests and all upstream strong tests**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/differential_uncertainty/test_strong_policies.py \
  tests/differential_uncertainty/test_strong_matching.py \
  tests/differential_uncertainty/test_strong_bank.py \
  tests/differential_uncertainty/test_strong_data.py \
  tests/differential_uncertainty/test_strong_scoring.py \
  tests/differential_uncertainty/test_strong_evaluation.py \
  tests/differential_uncertainty/test_strong_selection.py \
  tests/differential_uncertainty/test_strong_pipeline.py -q
```

Expected: all strong-workflow tests pass.

- [ ] **Step 8: Commit resumable orchestration**

```bash
git add differential_uncertainty/strong_pipeline.py tests/differential_uncertainty/test_strong_pipeline.py tests/differential_uncertainty/conftest.py
git commit -m "feat: orchestrate strong corruption study"
```

### Task 11: Publish complete per-corruption and ablation reports

**Files:**
- Create: `differential_uncertainty/strong_reporting.py`
- Create: `tests/differential_uncertainty/test_strong_reporting.py`
- Modify: `tests/differential_uncertainty/conftest.py`
- Read: `differential_uncertainty/reporting.py:1088-1680`

- [ ] **Step 1: Write exact report-schema and content tests**

Create `tests/differential_uncertainty/test_strong_reporting.py`:

```python
import csv
import hashlib
import json
from pathlib import Path
import pytest

from differential_uncertainty.strong_policies import (
    AGGREGATION_NAMES, BANK_COMPOSITIONS, STRONG_FAMILIES,
)
from differential_uncertainty.strong_reporting import (
    STRONG_REPORT_FILES, write_strong_report,
)


def file_sha256_and_mtime(root):
    root = Path(root)
    return {
        str(path.relative_to(root)): (
            hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns,
        )
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def test_report_has_every_family_level_and_baseline(report_result, tmp_path):
    output = tmp_path / "report"
    write_strong_report(output, report_result(families=STRONG_FAMILIES))
    assert {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()} == set(STRONG_REPORT_FILES)
    rows = list(csv.DictReader((output / "per-corruption-levels.csv").open()))
    family_rows = [row for row in rows if row["row_type"] == "family"]
    assert len(family_rows) == 19 * 3 * 2
    assert {row["corruption"] for row in family_rows} == set(STRONG_FAMILIES)
    assert {int(row["severity"]) for row in family_rows} == {4, 5}
    assert {row["method"] for row in family_rows} == {
        "fingerprint", "direct_confidence_max", "softmax_entropy_top_confidence_query"
    }


def test_report_never_hides_a_weak_corruption_behind_the_average(report_result, tmp_path):
    output = tmp_path / "report"
    write_strong_report(output, report_result(weak_family="glass_blur", weak_score=.31))
    rows = list(csv.DictReader((output / "per-corruption-strong.csv").open()))
    glass = next(row for row in rows if row["corruption"] == "glass_blur")
    assert float(glass["fingerprint_strong_point"]) == .31
    markdown = (output / "strong-corruption-report.md").read_text()
    assert "glass_blur" in markdown
    assert "0.310000" in markdown


def test_confidence_is_recomputed_not_assumed(report_result, tmp_path):
    output = tmp_path / "report"
    write_strong_report(output, report_result(confidence_point=.612345))
    markdown = (output / "strong-corruption-report.md").read_text()
    assert "0.612345" in markdown
    assert "0.714" not in markdown


@pytest.mark.parametrize(("overrides", "failed_label"), [
    ({"fingerprint_lower": .49}, "Detects strong corruption at all"),
    ({"conditional_lower": .49}, "Adds information after confidence stratification"),
    ({"confidence_delta_lower": -.01}, "Better standalone AUROC than maximum confidence"),
    ({"seed_scores": (.7, .7, .49, .7, .7)}, "Stable across bank seeds"),
])
def test_four_evidence_statements_are_independent(report_result, tmp_path, overrides, failed_label):
    labels = (
        "Detects strong corruption at all",
        "Adds information after confidence stratification",
        "Better standalone AUROC than maximum confidence",
        "Stable across bank seeds",
    )
    output = tmp_path / failed_label.replace(" ", "-")
    write_strong_report(output, report_result(**overrides))
    markdown = (output / "strong-corruption-report.md").read_text()
    for label in labels:
        status = "not supported" if label == failed_label else "supported"
        assert f"{label}: {status}" in markdown


def test_terminal_report_uses_plain_labels_not_math_markup(report_result, tmp_path):
    output = tmp_path / "report"
    write_strong_report(output, report_result())
    markdown = (output / "strong-corruption-report.md").read_text()
    assert "$" not in markdown
    assert "\\(" not in markdown
    assert "Levels 1 through 3 were not evaluated." in markdown


def test_stage_one_audit_keeps_all_bank_aggregation_cells(report_result, tmp_path):
    output = tmp_path / "report"
    write_strong_report(output, report_result())
    rows = list(csv.DictReader((output / "bank-composition-ablation.csv").open()))
    assert len(rows) == 25
    assert {(r["composition"], r["aggregation"]) for r in rows} == {
        (bank, aggregation) for bank in BANK_COMPOSITIONS for aggregation in AGGREGATION_NAMES
    }
```

```python
@pytest.mark.parametrize(("fault", "message"), [
    ("missing_family", "family roster"),
    ("duplicate_family", "duplicate"),
    ("missing_level", "severity"),
    ("missing_method", "method"),
    ("mild_level", "severity"),
    ("nonfinite_point", "finite"),
    ("nonfinite_interval", "finite"),
    ("reversed_interval", "interval"),
    ("missing_conditional_task", "38"),
    ("missing_seed", "seed"),
    ("provenance_digest", "provenance"),
])
def test_report_rejects_each_incomplete_or_invalid_schema(report_result, tmp_path, fault, message):
    with pytest.raises(ValueError, match=message):
        write_strong_report(tmp_path / fault, report_result(fault=fault))


def test_report_exact_resume_succeeds_but_changed_content_refuses(report_result, tmp_path):
    output = tmp_path / "report"
    value = report_result(confidence_point=.61)
    write_strong_report(output, value)
    snapshot = file_sha256_and_mtime(output)
    write_strong_report(output, value)
    assert file_sha256_and_mtime(output) == snapshot
    with pytest.raises(ValueError, match="existing strong report differs"):
        write_strong_report(output, report_result(confidence_point=.62))
```

- [ ] **Step 2: Run report tests and verify the missing module**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_strong_reporting.py -q
```

Expected: collection fails with `ModuleNotFoundError: differential_uncertainty.strong_reporting`.

- [ ] **Step 3: Define the exact bundle and validate before rendering**

Create `differential_uncertainty/strong_reporting.py`:

```python
STRONG_REPORT_FILES = (
    "strong-corruption-report.md",
    "per-corruption-levels.csv",
    "per-corruption-strong.csv",
    "conditional-concordance.csv",
    "paired-differences.csv",
    "bank-composition-ablation.csv",
    "construction-ablation.csv",
    "capacity-ablation.csv",
    "distance-ablation.csv",
    "neighbor-count-ablation.csv",
    "aggregation-ablation.csv",
    "seed-sensitivity.csv",
    "selection-trace.json",
    "provenance.json",
)
METHODS = (
    "fingerprint", "direct_confidence_max",
    "softmax_entropy_top_confidence_query",
)
```

`validate_report_result(result, expected_families)` must require exactly one point and interval for every `(family, method, severity)` in `families x METHODS x (4,5)`, one strong row per family, paired fingerprint-minus-confidence and fingerprint-minus-entropy intervals from the same bootstrap identity, all family/severity conditional tasks, every required ablation arm that was feasible or an explicit infeasibility record, all five selected-bank seeds, the full selection trace, fixed score orientations, and finite JSON-compatible values. Summary mean/median rows are derived inside the renderer and cannot replace family rows.

- [ ] **Step 4: Render two readable per-family tables and all machine tables**

Build the Markdown in plain terminal-safe text. The first table has one row per family and these columns:

```text
Corruption | Fingerprint L4 | Fingerprint L5 | Confidence L4 |
Confidence L5 | Entropy L4 | Entropy L5
```

The second has:

```text
Corruption | Fingerprint strong mean | Confidence strong mean |
Entropy strong mean | Fingerprint minus confidence | Fingerprint minus entropy
```

Append equal-family mean and median summary rows after all 19 family rows. Put point estimates in the readable tables at six decimals; put `point`, `lower`, `upper`, bootstrap draw count, bootstrap seed, configuration ID, and bootstrap identity in the detailed CSVs. Render separate controlled bank composition, construction, capacity, distance, neighbor-count, aggregation, and seed sections. Include candidate population counts, matched/background retained counts, k-means iteration/cluster counts, exact policy definitions, orientations, padding-mask caveat, full qualification results, and these four separately labeled statements:

```text
Detects strong corruption at all: supported / not supported
Adds information after confidence stratification: supported / not supported
Better standalone AUROC than maximum confidence: supported / not supported
Stable across bank seeds: supported / not supported
```

Use the actual intervals to fill each statement, including a negative conclusion when a gate fails. Explicitly say `Levels 1 through 3 were not evaluated.` and `No learned query pooling was used.` Do not emit TeX delimiters or unexplained symbols.

- [ ] **Step 5: Atomically publish or exactly resume the bundle**

Have `render_strong_report(result)` return a `dict[str, bytes]` whose key set equals `STRONG_REPORT_FILES`. Reuse the hardened directory ownership and no-replace primitives `_DirectoryLease`, `_StagingOwner`, and `_publish_no_replace` from `reporting.py`: write all intended bytes into the owned staging directory, fsync files and directory, verify the exact key set and bytes, then atomically rename without replacement. If the final directory exists, compare all regular-file bytes and return only on exact equality; otherwise raise `ValueError("existing strong report differs; choose a new output directory")`.

- [ ] **Step 6: Run strong and legacy reporting tests**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/differential_uncertainty/test_strong_reporting.py \
  tests/differential_uncertainty/test_reporting.py -q
```

Expected: all selected tests pass and the existing report bundle is unchanged.

- [ ] **Step 7: Commit complete reporting**

```bash
git add differential_uncertainty/strong_reporting.py tests/differential_uncertainty/test_strong_reporting.py tests/differential_uncertainty/conftest.py
git commit -m "feat: report every strong corruption result"
```

### Task 12: Expose fixed cache, development, and final CLI commands

**Files:**
- Modify: `differential_uncertainty/cli.py:2048-2105`
- Modify: `tests/differential_uncertainty/test_cli.py`

- [ ] **Step 1: Write parser and dispatch tests**

Append to `tests/differential_uncertainty/test_cli.py`:

```python
def test_strong_cache_command_fixes_levels_and_mode_counts(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        cli, "materialize_strong_coco_cache",
        lambda *args, **kwargs: captured.update(args=args, kwargs=kwargs),
    )
    assert cli.main([
        "strong-coco-cache", "--coco-annotations", "instances.json",
        "--coco-images", "val2017", "--checkpoint", "model.pth",
        "--output-dir", "cache", "--mode", "final", "--device", "cpu",
        "--batch-size", "4", "--shard-size", "50",
    ]) == 0
    assert captured["args"] == ("instances.json", "val2017", "model.pth", "cache")
    assert captured["kwargs"] == {
        "mode": "final", "device": "cpu", "batch_size": 4, "shard_size": 50
    }


def test_strong_development_command_has_paths_and_runtime_only(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "develop_strong_coco", lambda *args, **kwargs: captured.update(args=args, kwargs=kwargs))
    assert cli.main([
        "strong-coco-develop", "--reference-run", "reference",
        "--evaluation-benchmark", "evaluation",
        "--coco-annotations", "instances.json", "--output-dir", "study",
        "--device", "cpu",
    ]) == 0
    assert captured == {
        "args": ("reference", "evaluation", "instances.json", "study"),
        "kwargs": {"device": "cpu"},
    }


def test_strong_final_command_requires_frozen_development_inputs(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "evaluate_frozen_strong_coco", lambda *args, **kwargs: captured.update(args=args, kwargs=kwargs))
    assert cli.main([
        "strong-coco-final", "--reference-run", "final-reference",
        "--evaluation-benchmark", "final-evaluation",
        "--coco-annotations", "instances.json", "--output-dir", "final-study",
        "--frozen-policy", "development/frozen-policy.json",
        "--development-summary", "development/development.json", "--device", "cpu",
    ]) == 0
    assert captured["kwargs"]["frozen_policy_path"] == "development/frozen-policy.json"
    assert captured["kwargs"]["development_path"] == "development/development.json"


@pytest.mark.parametrize("flag", ["--bank-composition", "--aggregation", "--levels", "--bank-seed"])
def test_cli_does_not_expose_post_hoc_scientific_switches(flag):
    with pytest.raises(SystemExit) as error:
        cli.build_parser().parse_args([
            "strong-coco-develop", "--reference-run", "reference",
            "--evaluation-benchmark", "evaluation",
            "--coco-annotations", "instances.json", "--output-dir", "study",
            flag, "anything",
        ])
    assert error.value.code == 2
```

```python
def test_strong_dispatch_reports_validation_error(monkeypatch, capsys):
    def fail(*_args, **_kwargs):
        raise ValueError("frozen policy does not match development")
    monkeypatch.setattr(cli, "evaluate_frozen_strong_coco", fail)
    code = cli.main([
        "strong-coco-final", "--reference-run", "reference",
        "--evaluation-benchmark", "evaluation",
        "--coco-annotations", "instances.json", "--output-dir", "final",
        "--frozen-policy", "frozen.json",
        "--development-summary", "development.json",
    ])
    assert code == 2
    assert capsys.readouterr().err == "error: frozen policy does not match development\n"
```

- [ ] **Step 2: Run CLI tests and verify command parsing fails**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty/test_cli.py -q
```

Expected: new tests fail because all three commands are unknown.

- [ ] **Step 3: Add all three parsers and explicit dispatch branches**

Import `materialize_strong_coco_cache`, `develop_strong_coco`, and `evaluate_frozen_strong_coco` from `strong_pipeline`. Extend `build_parser`:

```python
cache = commands.add_parser(
    "strong-coco-cache", help="materialize fixed severity-0/4/5 COCO caches",
)
cache.add_argument("--coco-annotations", required=True)
cache.add_argument("--coco-images", required=True)
cache.add_argument("--checkpoint", required=True)
cache.add_argument("--output-dir", required=True)
cache.add_argument("--mode", choices=("development", "final"), required=True)
cache.add_argument("--device", default="cuda:0")
cache.add_argument("--batch-size", type=_positive, default=1)
cache.add_argument("--shard-size", type=_positive, default=50)

develop = commands.add_parser(
    "strong-coco-develop",
    help="select a fingerprint policy on fixed severity-4/5 development data",
)
develop.add_argument("--reference-run", required=True)
develop.add_argument("--evaluation-benchmark", required=True)
develop.add_argument("--coco-annotations", required=True)
develop.add_argument("--output-dir", required=True)
develop.add_argument("--device", default="cuda:0")

final = commands.add_parser(
    "strong-coco-final",
    help="evaluate one qualified frozen policy on untouched severity-4/5 data",
)
final.add_argument("--reference-run", required=True)
final.add_argument("--evaluation-benchmark", required=True)
final.add_argument("--coco-annotations", required=True)
final.add_argument("--output-dir", required=True)
final.add_argument("--frozen-policy", required=True)
final.add_argument("--development-summary", required=True)
final.add_argument("--device", default="cuda:0")
```

Replace the current two-way `if/else` dispatch with explicit branches:

```python
if args.command == "run":
    run_pipeline(
        args.reference_manifest, args.evaluation_manifest, args.checkpoint,
        args.output_dir, device=args.device, batch_size=args.batch_size,
        shard_size=args.shard_size,
    )
elif args.command == "benchmark-coco":
    run_coco_benchmark(
        args.coco_annotations, args.coco_images, args.checkpoint,
        args.output_dir, device=args.device, batch_size=args.batch_size,
        shard_size=args.shard_size, reference_count=args.reference_count,
        evaluation_count=args.evaluation_count,
    )
elif args.command == "strong-coco-cache":
    materialize_strong_coco_cache(
        args.coco_annotations, args.coco_images, args.checkpoint,
        args.output_dir, mode=args.mode, device=args.device,
        batch_size=args.batch_size, shard_size=args.shard_size,
    )
elif args.command == "strong-coco-develop":
    develop_strong_coco(
        args.reference_run, args.evaluation_benchmark,
        args.coco_annotations, args.output_dir,
        device=args.device,
    )
elif args.command == "strong-coco-final":
    evaluate_frozen_strong_coco(
        args.reference_run, args.evaluation_benchmark,
        args.coco_annotations, args.output_dir,
        frozen_policy_path=args.frozen_policy,
        development_path=args.development_summary,
        device=args.device,
    )
else:
    raise AssertionError(f"unhandled command: {args.command}")
```

Retain the existing caught error types and return codes.

- [ ] **Step 4: Run CLI and pipeline tests**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/differential_uncertainty/test_cli.py \
  tests/differential_uncertainty/test_strong_pipeline.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit CLI integration**

```bash
git add differential_uncertainty/cli.py tests/differential_uncertainty/test_cli.py
git commit -m "feat: expose strong corruption study commands"
```

### Task 13: Prove end-to-end determinism and update the protected Python surface

**Files:**
- Modify: `tests/differential_uncertainty/test_strong_pipeline.py`
- Modify: `tests/differential_uncertainty/test_repository_surface.py:20-78,174-175`

- [ ] **Step 1: Add a true tiny-cache integration test**

Append one integration case that uses real cache shards, two reference images, ten queries, two evaluation images, and two corruption families. Inject `StrongStudyConfig.for_tests` and `SelectionPanel.for_tests`, whose miniature capacities are `(5,10)`, neighbor counts are `(1,5,10)`, and seeds are `(42,44)`; do not mock matching, reservoir construction, distances, union masking, aggregation, AUROC, conditional concordance, selection, or report serialization.

```python
def semantic_tree_digest(root):
    digest = hashlib.sha256()
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_tiny_cache_runs_matching_through_report_deterministically(tiny_strong_benchmark, tmp_path):
    first = develop_strong_coco(
        tiny_strong_benchmark.reference_root,
        tiny_strong_benchmark.evaluation_root,
        tiny_strong_benchmark.annotations,
        tmp_path / "first", device="cpu",
        config=tiny_strong_benchmark.config, panel=tiny_strong_benchmark.panel,
    )
    second = develop_strong_coco(
        tiny_strong_benchmark.reference_root,
        tiny_strong_benchmark.evaluation_root,
        tiny_strong_benchmark.annotations,
        tmp_path / "second", device="cpu",
        config=tiny_strong_benchmark.config, panel=tiny_strong_benchmark.panel,
    )
    assert semantic_tree_digest(first.output) == semantic_tree_digest(second.output)
    levels = read_csv(first.report_directory / "per-corruption-levels.csv")
    assert {int(row["severity"]) for row in levels if row["row_type"] == "family"} == {4, 5}
    assert {row["corruption"] for row in levels if row["row_type"] == "family"} == {"fog", "snow"}
```

Make each synthetic severity record deliberately change padding-tail length so the test also proves the union mask applies identical query IDs to clean, level 4, and level 5. Give mild records invalid sentinel tensors that would fail if a scorer accessed them.

- [ ] **Step 2: Run the integration test and verify the surface guard fails**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/differential_uncertainty/test_strong_pipeline.py::test_tiny_cache_runs_matching_through_report_deterministically \
  tests/differential_uncertainty/test_repository_surface.py::test_repository_python_surface_is_exact -q
```

Expected: integration passes; repository-surface test fails because 19 intentional Python files are not yet allowlisted.

- [ ] **Step 3: Extend only the exact Python allowlist**

Add these source files to `RETAINED_PYTHON`:

```text
differential_uncertainty/strong_bank.py
differential_uncertainty/strong_data.py
differential_uncertainty/strong_evaluation.py
differential_uncertainty/strong_matching.py
differential_uncertainty/strong_pipeline.py
differential_uncertainty/strong_policies.py
differential_uncertainty/strong_reporting.py
differential_uncertainty/strong_scoring.py
differential_uncertainty/strong_selection.py
```

Add the corresponding nine `tests/differential_uncertainty/test_strong_*.py` paths plus:

```text
tests/differential_uncertainty/conftest.py
```

Then change only:

```python
assert len(RETAINED_PYTHON) == 65
```

Do not change `RETAINED_IMPORT_DIRECTORIES`, `RETAINED_SRC`, symlink checks, dependency exclusions, or artifact-suffix checks.

- [ ] **Step 4: Run integration and repository-surface tests**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/differential_uncertainty/test_strong_pipeline.py \
  tests/differential_uncertainty/test_repository_surface.py -q
```

Expected: all selected tests pass and the retained count is exactly 65.

- [ ] **Step 5: Commit integration coverage and surface closure**

```bash
git add tests/differential_uncertainty/test_strong_pipeline.py tests/differential_uncertainty/test_repository_surface.py
git commit -m "test: cover strong corruption workflow end to end"
```

### Task 14: Document and run the 1,000-reference/250-evaluation development study

**Files:**
- Modify: `README.md`
- Produce (ignored run artifact): `runs/strong-corruption-development-1000x250/`

- [ ] **Step 1: Document the three commands and interpretation boundary**

Add a `Strong-corruption fingerprint study` section to `README.md` containing these commands:

```bash
# Optional: make a new cache that extracts only levels 0, 4, and 5.
/home/yuchen/miniconda3/envs/UE/bin/python -m differential_uncertainty strong-coco-cache \
  --mode development \
  --coco-annotations /home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json \
  --coco-images /home/yuchen/YuchenZ/Datasets/coco/val2017 \
  --checkpoint /home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \
  --output-dir runs/strong-corruption-development-cache \
  --device cuda:0 --batch-size 1 --shard-size 50

# Select on the existing authenticated 1,000-reference and 250-evaluation caches.
/home/yuchen/miniconda3/envs/UE/bin/python -m differential_uncertainty strong-coco-develop \
  --reference-run /home/yuchen/YuchenZ/UE/philip_sa/runs/coco-offline-bank-1000 \
  --evaluation-benchmark /home/yuchen/YuchenZ/UE/philip_sa/.worktrees/imagecorruptions-coco-benchmark/runs/coco-imagecorruptions-250 \
  --coco-annotations /home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json \
  --output-dir runs/strong-corruption-development-1000x250 \
  --device cuda:0
```

Explain that a legacy six-level cache may be reused, but only records 0/4/5 enter scoring; newly materialized caches contain only those three levels. State that every corruption is reported separately, confidence and entropy are recomputed on the identical mask/roster, no learned pooling is used, and `frozen-policy.json` exists only if every qualification gate passes.

- [ ] **Step 2: Verify help text and README commands**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m differential_uncertainty strong-coco-cache --help
/home/yuchen/miniconda3/envs/UE/bin/python -m differential_uncertainty strong-coco-develop --help
/home/yuchen/miniconda3/envs/UE/bin/python -m differential_uncertainty strong-coco-final --help
rg -n "strong-coco-(cache|develop|final)|Levels 1 through 3|learned pooling" README.md
```

Expected: each help command exits 0; the README contains all commands and interpretation statements.

- [ ] **Step 3: Commit documentation before producing results**

```bash
git add README.md
git commit -m "docs: explain strong corruption study"
```

- [ ] **Step 4: Preflight the actual split development caches**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -c 'from differential_uncertainty.strong_data import open_strong_benchmark; from differential_uncertainty.strong_policies import STRONG_DEVELOPMENT; x=open_strong_benchmark("/home/yuchen/YuchenZ/UE/philip_sa/runs/coco-offline-bank-1000", "/home/yuchen/YuchenZ/UE/philip_sa/.worktrees/imagecorruptions-coco-benchmark/runs/coco-imagecorruptions-250", "/home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json", STRONG_DEVELOPMENT); print(f"reference={len(x.reference_manifest)} evaluation={len(x.evaluation_manifest)} families={len(x.evaluation_caches)} levels={STRONG_DEVELOPMENT.levels}")'
```

Expected exactly:

```text
reference=1000 evaluation=250 families=19 levels=(0, 4, 5)
```

This also validates zero overlap, identical checkpoint digest, compatible extraction fields, and the current annotation digest. Do not substitute the available 250-reference cache.

- [ ] **Step 5: Run or resume development selection**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m differential_uncertainty strong-coco-develop \
  --reference-run /home/yuchen/YuchenZ/UE/philip_sa/runs/coco-offline-bank-1000 \
  --evaluation-benchmark /home/yuchen/YuchenZ/UE/philip_sa/.worktrees/imagecorruptions-coco-benchmark/runs/coco-imagecorruptions-250 \
  --coco-annotations /home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json \
  --output-dir runs/strong-corruption-development-1000x250 \
  --device cuda:0
```

Expected: the command is resumable, scores only levels 0/4/5, completes all 95 outer cells, and exits 0 with either `qualified` or `no_qualifying_policy`. Do not alter the panel after observing partial results.

- [ ] **Step 6: Reconcile every development report row**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -c 'import csv,json,pathlib; root=pathlib.Path("runs/strong-corruption-development-1000x250"); d=json.loads((root/"development.json").read_text()); levels=list(csv.DictReader((root/"reports/per-corruption-levels.csv").open())); strong=list(csv.DictReader((root/"reports/per-corruption-strong.csv").open())); families={r["corruption"] for r in strong if r["row_type"]=="family"}; assert d["study"]["levels"]==[0,4,5]; assert d["study"]["outer_cell_count"]==95; assert {int(r["severity"]) for r in levels if r["row_type"]=="family"}=={4,5}; assert {r["method"] for r in levels if r["row_type"]=="family"}=={"fingerprint","direct_confidence_max","softmax_entropy_top_confidence_query"}; assert len(families)==19; print(d["selection"]["status"], len(families))'
```

Expected: prints either `qualified 19` or `no_qualifying_policy 19`. Open `runs/strong-corruption-development-1000x250/reports/strong-corruption-report.md` and verify both tables show all 19 corruption names, level-4 and level-5 numbers, both baselines, paired differences, conditional concordance, seed spread, and every ablation—not only averages.

### Task 15: Run the untouched 2,500/2,500 final study only after qualification

**Files:**
- Produce (ignored cache): `runs/strong-corruption-final-cache-2500x2500/`
- Produce (ignored run artifact): `runs/strong-corruption-final-2500x2500/`

- [ ] **Step 1: Enforce the qualification checkpoint**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -c 'import json,pathlib; root=pathlib.Path("runs/strong-corruption-development-1000x250"); d=json.loads((root/"development.json").read_text()); assert d["selection"]["status"]=="qualified", "development produced no qualifying fingerprint policy"; assert (root/"frozen-policy.json").is_file(); print(d["selection"]["selected_policy_id"])'
sha256sum runs/strong-corruption-development-1000x250/frozen-policy.json
```

Expected: a 64-character policy ID and a frozen-file digest. If the assertion fails, stop: the completed outcome is the negative development report, and no final cache or evaluation is allowed by the design.

- [ ] **Step 2: Materialize only 0/4/5 for the untouched final roster**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m differential_uncertainty strong-coco-cache \
  --mode final \
  --coco-annotations /home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json \
  --coco-images /home/yuchen/YuchenZ/Datasets/coco/val2017 \
  --checkpoint /home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \
  --output-dir runs/strong-corruption-final-cache-2500x2500 \
  --device cuda:0 --batch-size 1 --shard-size 50
```

Expected: 2,500 clean reference records and `2,500 * 3` evaluation records for each of 19 families; no level-1/2/3 record exists.

- [ ] **Step 3: Evaluate the frozen tuple exactly once**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m differential_uncertainty strong-coco-final \
  --reference-run runs/strong-corruption-final-cache-2500x2500 \
  --evaluation-benchmark runs/strong-corruption-final-cache-2500x2500 \
  --coco-annotations /home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json \
  --frozen-policy runs/strong-corruption-development-1000x250/frozen-policy.json \
  --development-summary runs/strong-corruption-development-1000x250/development.json \
  --output-dir runs/strong-corruption-final-2500x2500 \
  --device cuda:0
```

Expected: the evaluator builds only the frozen seed-44 bank and one fingerprint scoring tuple, recomputes confidence/entropy, uses frozen development confidence boundaries, and never enters nested selection.

- [ ] **Step 4: Prove the frozen file and complete per-family output**

Run:

```bash
sha256sum runs/strong-corruption-development-1000x250/frozen-policy.json
/home/yuchen/miniconda3/envs/UE/bin/python -c 'import csv,json,pathlib; root=pathlib.Path("runs/strong-corruption-final-2500x2500"); p=json.loads((root/"reports/provenance.json").read_text()); levels=list(csv.DictReader((root/"reports/per-corruption-levels.csv").open())); strong=list(csv.DictReader((root/"reports/per-corruption-strong.csv").open())); family_rows=[r for r in strong if r["row_type"]=="family"]; assert p["reference_count"]==2500 and p["evaluation_count"]==2500; assert p["levels"]==[0,4,5]; assert p["bank_seed"]==44; assert len(family_rows)==19; assert {int(r["severity"]) for r in levels if r["row_type"]=="family"}=={4,5}; print("verified", len(family_rows), "families")'
```

Expected: the frozen-policy SHA-256 is unchanged from Step 1 and the script prints `verified 19 families`. The deliverable report is `runs/strong-corruption-final-2500x2500/reports/strong-corruption-report.md`; it must retain every weak family and not replace the table with an overall average.

### Task 16: Final verification and branch review

**Files:**
- Review: all files changed since `e3599b6`

- [ ] **Step 1: Load the completion-verification instructions**

Read `/home/yuchen/.agents/skills/superpowers/verification-before-completion/SKILL.md` completely and follow it before making any passing/completion claim.

- [ ] **Step 2: Run syntax and full regression verification**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m compileall -q differential_uncertainty tests/differential_uncertainty
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q
git diff --check e3599b6..HEAD
git status --short
```

Expected: compile succeeds; every test passes; diff check has no output; status is clean except ignored run artifacts.

- [ ] **Step 3: Audit fixed scientific invariants in source**

Run:

```bash
rg -n "0\.714|learned.*pool|severity.*[123]|selected_levels|BANK_COMPOSITIONS|AGGREGATION_NAMES|BANK_SEEDS" differential_uncertainty tests/differential_uncertainty README.md
git diff --stat e3599b6..HEAD
git diff --name-only e3599b6..HEAD
```

Expected: no hard-coded `0.714`; no learned pooler; every occurrence of levels 1-3 is a rejection, legacy-compatibility, or explicit non-evaluation statement; the fixed panels are centralized; only planned files changed.

- [ ] **Step 4: Request code review and resolve findings**

Read `/home/yuchen/.agents/skills/superpowers/requesting-code-review/SKILL.md` completely. Review the complete diff against the approved design, paying special attention to outer-fold leakage, same-mask baselines, matched/background semantics, mild-level isolation, resume identity, and all 19 report rows. Fix each confirmed finding with a failing regression test first, rerun the focused test, then rerun Step 2.

- [ ] **Step 5: Hand off the verified branch and reports**

Report the branch name, commit range, full-test count, development qualification status, selected tuple or explicit no-qualifier result, and clickable paths to the Markdown plus both per-corruption CSVs. If Task 15 ran, report final numbers rather than development numbers and state that the final policy file hash remained unchanged.
