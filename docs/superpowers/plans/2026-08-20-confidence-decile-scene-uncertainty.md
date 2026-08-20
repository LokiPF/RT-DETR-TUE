# Confidence-Decile Scene-Uncertainty Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a training-free confidence-decile analysis that finds which detector-confidence ranges carry blur-sensitive persistence information and compares every persistence score with the detector's own confidence uncertainty.

**Architecture:** Add pure modules for padded-tail detection, confidence-bin membership, and matched signal scoring; add a streaming artifact orchestrator that reuses cached logits and per-query kNN distances; add a dedicated reporter for paired metrics, plots, and a plain-language result. Expose the workflow through one tuning-only CLI command so the held-out test partition cannot be spent accidentally.

**Tech Stack:** Python 3.11, PyTorch, NumPy, pandas, SciPy, Matplotlib, pytest, and the existing `src.scene_uncertainty` artifact and metric helpers.

---

Design specification: `docs/superpowers/specs/2026-08-20-confidence-decile-scene-uncertainty-design.md`

## File map

- Create `src/scene_uncertainty/confidence_deciles.py`: padded-tail detection and deterministic dynamic/frozen decile membership.
- Create `src/scene_uncertainty/decile_scoring.py`: matched persistence and confidence-only score rows.
- Create `src/scene_uncertainty/decile_analysis.py`: safe artifact loading, validation, cross-severity orchestration, and controls.
- Create `src/scene_uncertainty/decile_reporting.py`: CSV/JSON output, paired metrics, plots, and plain-language report.
- Modify `src/scene_uncertainty/cli.py` and `pipeline.py`: add a tuning-only command.
- Create `tests/scene_uncertainty/decile_test_utils.py` plus five focused test modules; modify `test_cli.py`.
- Modify `README.md`: document the cache-only experiment.

## Task 1: Detect invalid padded query tails

**Files:**
- Create: `src/scene_uncertainty/confidence_deciles.py`
- Create: `tests/scene_uncertainty/test_confidence_deciles.py`

- [ ] **Step 1: Write the failing tests**

```python
import torch

from src.scene_uncertainty.confidence_deciles import detect_padded_tail, union_padded_query_ids


def record(boxes, logits, layer_0, layer_1, image_id=1, severity=0):
    return {
        "image_id": image_id,
        "severity": severity,
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "logits": torch.tensor(logits, dtype=torch.float16),
        "layers": {
            0: torch.tensor(layer_0, dtype=torch.float16),
            1: torch.tensor(layer_1, dtype=torch.float16),
        },
    }


def test_detects_only_the_repeated_identical_suffix():
    item = record(
        [[0, 0], [1, 1], [9, 9], [9, 9]],
        [[1, 0], [0, 1], [-5, -5], [-5, -5]],
        [[1, 0], [0, 1], [7, 7], [7, 7]],
        [[2, 0], [0, 2], [8, 8], [8, 8]],
    )
    assert detect_padded_tail(item).tolist() == [2, 3]


def test_identical_non_tail_queries_are_retained():
    item = record(
        [[9, 9], [9, 9], [1, 1], [2, 2]],
        [[-5, -5], [-5, -5], [1, 0], [0, 1]],
        [[7, 7], [7, 7], [1, 0], [0, 1]],
        [[8, 8], [8, 8], [2, 0], [0, 2]],
    )
    assert detect_padded_tail(item).numel() == 0


def test_one_final_query_is_not_called_padding():
    item = record(
        [[0, 0], [1, 1], [9, 9]],
        [[1, 0], [0, 1], [-5, -5]],
        [[1, 0], [0, 1], [7, 7]],
        [[2, 0], [0, 2], [8, 8]],
    )
    assert detect_padded_tail(item).numel() == 0


def test_union_mask_is_fixed_across_severities():
    clean = record(
        [[0, 0], [1, 1], [2, 2], [3, 3]],
        [[1, 0], [0, 1], [1, 1], [2, 2]],
        [[1, 0], [0, 1], [1, 1], [2, 2]],
        [[2, 0], [0, 2], [2, 2], [3, 3]], severity=0,
    )
    blurred = record(
        [[0, 0], [1, 1], [9, 9], [9, 9]],
        [[1, 0], [0, 1], [-5, -5], [-5, -5]],
        [[1, 0], [0, 1], [7, 7], [7, 7]],
        [[2, 0], [0, 2], [8, 8], [8, 8]], severity=1,
    )
    assert union_padded_query_ids([clean, blurred]).tolist() == [2, 3]
```

- [ ] **Step 2: Run the test and confirm red**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty/test_confidence_deciles.py -v
```

Expected: collection fails because `confidence_deciles` does not exist.

- [ ] **Step 3: Implement the pure detector**

```python
from __future__ import annotations

from collections.abc import Iterable
import torch
from torch import Tensor


DECILE_NAMES = tuple(f"decile_{lower:02d}_{lower + 10:02d}" for lower in range(0, 100, 10))


def _rows_equal_to_last(values: Tensor) -> Tensor:
    if values.ndim < 2:
        raise ValueError(f"query tensor needs query and feature dimensions, got {tuple(values.shape)}")
    return values.eq(values[-1]).reshape(values.shape[0], -1).all(dim=1)


def detect_padded_tail(record: dict) -> Tensor:
    fields = [record["boxes"], record["logits"]]
    fields.extend(record["layers"][layer_id] for layer_id in sorted(record["layers"]))
    counts = {int(field.shape[0]) for field in fields}
    if len(counts) != 1:
        raise ValueError(f"query-count mismatch inside record: {sorted(counts)}")
    query_count = counts.pop()
    if query_count == 0:
        return torch.empty(0, dtype=torch.long)
    repeated = torch.stack([_rows_equal_to_last(field.cpu()) for field in fields]).all(dim=0)
    start = query_count - 1
    while start > 0 and bool(repeated[start - 1]):
        start -= 1
    if query_count - start < 2:
        return torch.empty(0, dtype=torch.long)
    return torch.arange(start, query_count, dtype=torch.long)


def union_padded_query_ids(records: Iterable[dict]) -> Tensor:
    records = list(records)
    if not records:
        raise ValueError("cannot build a padding union from zero records")
    image_ids = {int(record["image_id"]) for record in records}
    if len(image_ids) != 1:
        raise ValueError(f"padding union needs one image, got {sorted(image_ids)}")
    counts = {int(record["logits"].shape[0]) for record in records}
    if len(counts) != 1:
        raise ValueError(f"query-count mismatch across severities: {sorted(counts)}")
    padded = sorted({int(index) for record in records for index in detect_padded_tail(record)})
    return torch.tensor(padded, dtype=torch.long)
```

- [ ] **Step 4: Run the test and confirm green**

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/scene_uncertainty/confidence_deciles.py tests/scene_uncertainty/test_confidence_deciles.py
git commit -m "feat: detect padded decoder query tails"
```

## Task 2: Build deterministic dynamic and frozen confidence deciles

**Files:**
- Modify: `src/scene_uncertainty/confidence_deciles.py`
- Modify: `tests/scene_uncertainty/test_confidence_deciles.py`

- [ ] **Step 1: Add failing tests for confidence, ties, sizes, and frozen membership**

```python
import pytest
from src.scene_uncertainty.confidence_deciles import (
    confidence_deciles, confidence_from_logits, memberships_by_severity,
)


def test_confidence_is_largest_sigmoid_class_score():
    actual = confidence_from_logits(torch.tensor([[0.0, 2.0], [-2.0, -1.0]]))
    assert torch.allclose(actual, torch.tensor([2.0, -1.0]).sigmoid())


def test_equal_confidence_ties_follow_query_id():
    bins = confidence_deciles(torch.tensor([0.5] * 20), torch.arange(20))
    assert [values.tolist() for values in bins.values()] == [
        [0, 1], [2, 3], [4, 5], [6, 7], [8, 9],
        [10, 11], [12, 13], [14, 15], [16, 17], [18, 19],
    ]


def test_non_divisible_count_differs_by_at_most_one():
    bins = confidence_deciles(torch.arange(23, dtype=torch.float32), torch.arange(23))
    sizes = [len(values) for values in bins.values()]
    assert sum(sizes) == 23
    assert max(sizes) - min(sizes) == 1


def test_frozen_ids_stay_clean_while_dynamic_ids_move():
    records = {
        0: {"confidence": torch.arange(20, dtype=torch.float32)},
        1: {"confidence": torch.arange(19, -1, -1, dtype=torch.float32)},
    }
    result = memberships_by_severity(records, torch.empty(0, dtype=torch.long))
    assert result[1]["frozen"]["decile_00_10"].tolist() == [0, 1]
    assert result[1]["dynamic"]["decile_00_10"].tolist() == [19, 18]
    assert result[0]["all_valid"].tolist() == list(range(20))


def test_fewer_than_ten_valid_queries_is_rejected():
    with pytest.raises(ValueError, match="at least ten"):
        confidence_deciles(torch.arange(9, dtype=torch.float32), torch.arange(9))
```

- [ ] **Step 2: Run the file and confirm the new imports fail**

Use the Task 1 pytest command.

- [ ] **Step 3: Implement confidence ranking and membership**

```python
def confidence_from_logits(logits: Tensor) -> Tensor:
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape (query, class), got {tuple(logits.shape)}")
    return logits.float().sigmoid().amax(dim=-1).cpu()


def confidence_deciles(confidence: Tensor, valid_indices: Tensor) -> dict[str, Tensor]:
    valid = torch.sort(valid_indices.long().cpu()).values
    if valid.numel() < len(DECILE_NAMES):
        raise ValueError(f"confidence deciles need at least ten valid queries, got {valid.numel()}")
    if valid.unique().numel() != valid.numel():
        raise ValueError("valid query indices must be unique")
    if int(valid.min()) < 0 or int(valid.max()) >= confidence.numel():
        raise ValueError("valid query index lies outside the confidence vector")
    order = torch.argsort(confidence.index_select(0, valid), stable=True)
    chunks = torch.tensor_split(valid.index_select(0, order), len(DECILE_NAMES))
    return {name: chunk for name, chunk in zip(DECILE_NAMES, chunks)}


def memberships_by_severity(records_by_severity: dict[int, dict], padded: Tensor) -> dict[int, dict]:
    if 0 not in records_by_severity:
        raise ValueError("clean-frozen bins require severity zero")
    counts = {int(record["confidence"].numel()) for record in records_by_severity.values()}
    if len(counts) != 1:
        raise ValueError(f"query-count mismatch across severities: {sorted(counts)}")
    keep = torch.ones(counts.pop(), dtype=torch.bool)
    keep[padded.long()] = False
    valid = torch.where(keep)[0]
    frozen = confidence_deciles(records_by_severity[0]["confidence"], valid)
    return {
        severity: {
            "dynamic": confidence_deciles(record["confidence"], valid),
            "frozen": {name: indices.clone() for name, indices in frozen.items()},
            "all_valid": valid.clone(),
        }
        for severity, record in sorted(records_by_severity.items())
    }
```

- [ ] **Step 4: Run and confirm all pure tests pass**

Expected: `9 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/scene_uncertainty/confidence_deciles.py tests/scene_uncertainty/test_confidence_deciles.py
git commit -m "feat: build confidence decile memberships"
```

## Task 3: Score persistence and confidence with identical selections

**Files:**
- Create: `src/scene_uncertainty/decile_scoring.py`
- Create: `tests/scene_uncertainty/test_decile_scoring.py`

- [ ] **Step 1: Write failing matched-score tests**

```python
import pytest
import torch
from src.scene_uncertainty.decile_scoring import score_selection


SCALES = {
    0: {"center": torch.tensor(1.0), "scale": torch.tensor(2.0)},
    1: {"center": torch.tensor(2.0), "scale": torch.tensor(4.0)},
}


def test_matched_signals_use_same_ids_and_mean():
    rows = score_selection(
        image_id=7, severity=2, source_partition="tuning",
        membership_mode="dynamic", confidence_bin="decile_00_10",
        padding_mode="filtered", indices=torch.tensor([1, 3]),
        confidence=torch.tensor([0.9, 0.8, 0.7, 0.6]),
        query_scores_by_layer={
            0: torch.tensor([0.0, 3.0, 0.0, 5.0]),
            1: torch.tensor([0.0, 6.0, 0.0, 10.0]),
        },
        layer_score_scales=SCALES, clean_overlap=0.5,
        aggregations=("mean",),
    )
    by_key = {(row["signal"], row["score_scope"]): row for row in rows}
    assert by_key[("confidence", "confidence")]["score"] == pytest.approx(0.3)
    assert by_key[("persistence", "layer_0")]["score"] == 4.0
    assert by_key[("persistence", "layer_1")]["score"] == 8.0
    assert by_key[("persistence", "combined")]["score"] == pytest.approx(1.5)
    assert {tuple(row["selected_query_ids"]) for row in rows} == {(1, 3)}


def test_q90_and_top20_mean_are_applied_to_both_signals():
    rows = score_selection(
        image_id=1, severity=0, source_partition="tuning",
        membership_mode="frozen", confidence_bin="decile_10_20",
        padding_mode="filtered", indices=torch.arange(10),
        confidence=torch.linspace(0.1, 1.0, 10),
        query_scores_by_layer={0: torch.arange(10, dtype=torch.float32)},
        layer_score_scales={0: {"center": torch.tensor(0.0), "scale": torch.tensor(1.0)}},
        clean_overlap=1.0, aggregations=("q90", "top20_mean"),
    )
    assert {(row["signal"], row["aggregation"]) for row in rows} == {
        ("confidence", "q90"), ("confidence", "top20_mean"),
        ("persistence", "q90"), ("persistence", "top20_mean"),
    }


def test_empty_selection_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        score_selection(
            image_id=1, severity=0, source_partition="tuning",
            membership_mode="dynamic", confidence_bin="decile_00_10",
            padding_mode="filtered", indices=torch.empty(0, dtype=torch.long),
            confidence=torch.ones(3), query_scores_by_layer={0: torch.ones(3)},
            layer_score_scales={0: {"center": torch.tensor(0.0), "scale": torch.tensor(1.0)}},
            clean_overlap=1.0, aggregations=("mean",),
        )
```

- [ ] **Step 2: Run and confirm the module is missing**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty/test_decile_scoring.py -v
```

- [ ] **Step 3: Implement the row builder**

```python
from __future__ import annotations

from torch import Tensor
from .query_policy import aggregate_scores

DECILE_AGGREGATIONS = ("mean", "q90", "top20_mean")


def score_selection(
    *, image_id: int, severity: int, source_partition: str,
    membership_mode: str, confidence_bin: str, padding_mode: str,
    indices: Tensor, confidence: Tensor, query_scores_by_layer: dict[int, Tensor],
    layer_score_scales: dict[int, dict[str, Tensor]], clean_overlap: float,
    aggregations: tuple[str, ...] = DECILE_AGGREGATIONS,
    include_confidence: bool = True,
) -> list[dict]:
    indices = indices.long().cpu()
    if indices.numel() == 0:
        raise ValueError("cannot score an empty confidence-bin selection")
    query_count = int(confidence.numel())
    if any(int(values.numel()) != query_count for values in query_scores_by_layer.values()):
        raise ValueError("persistence and confidence query count disagree")
    if set(query_scores_by_layer) != set(layer_score_scales):
        raise ValueError("persistence layers and clean-distance scales disagree")
    base = {
        "image_id": int(image_id), "severity": int(severity),
        "source_partition": source_partition, "membership_mode": membership_mode,
        "confidence_bin": confidence_bin, "padding_mode": padding_mode,
        "selected_count": int(indices.numel()), "selected_query_ids": indices.tolist(),
        "clean_overlap": float(clean_overlap),
    }
    rows = []
    confidence_uncertainty = 1.0 - confidence.float().index_select(0, indices)
    for aggregation in aggregations:
        if include_confidence:
            rows.append({
                **base, "signal": "confidence", "score_scope": "confidence",
                "aggregation": aggregation,
                "score": float(aggregate_scores(confidence_uncertainty, aggregation)),
            })
        layer_scores = {
            layer_id: float(aggregate_scores(values.float().index_select(0, indices), aggregation))
            for layer_id, values in sorted(query_scores_by_layer.items())
        }
        scaled = [
            (score - float(layer_score_scales[layer_id]["center"]))
            / float(layer_score_scales[layer_id]["scale"])
            for layer_id, score in layer_scores.items()
        ]
        for layer_id, score in layer_scores.items():
            rows.append({
                **base, "signal": "persistence", "score_scope": f"layer_{layer_id}",
                "aggregation": aggregation, "score": score,
            })
        rows.append({
            **base, "signal": "persistence", "score_scope": "combined",
            "aggregation": aggregation, "score": sum(scaled) / len(scaled),
        })
    return rows
```

- [ ] **Step 4: Run and confirm green**

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/scene_uncertainty/decile_scoring.py tests/scene_uncertainty/test_decile_scoring.py
git commit -m "feat: score matched persistence and confidence signals"
```

## Task 4: Load and validate cached artifacts without recomputing kNN

**Files:**
- Create: `src/scene_uncertainty/decile_analysis.py`
- Create: `tests/scene_uncertainty/test_decile_analysis.py`
- Create: `tests/scene_uncertainty/decile_test_utils.py`

- [ ] **Step 1: Build a six-severity fixture and write failing validation tests**

Use `ShardWriter` to create one tuning image with severities zero through five, 20 queries, boxes, logits, and three persistence layers. Write sibling query-distance, normalizer, and manifest files using safe tensor/JSON types. Put the reusable artifact builder in `decile_test_utils.py`, import it into `test_decile_analysis.py`, then add:

```python
def test_load_inputs_joins_six_tuning_records_by_key(decile_artifacts):
    loaded = load_decile_inputs(decile_artifacts["cache"], decile_artifacts["results"])
    assert set(loaded.records_by_image) == {11}
    assert set(loaded.records_by_image[11]) == set(range(6))
    assert set(loaded.distances) == {(11, severity, "tuning") for severity in range(6)}
    assert set(loaded.layer_score_scales) == {0, 1, 2}


def test_load_inputs_refuses_another_cache(decile_artifacts):
    path = decile_artifacts["results"].with_suffix(".manifest.json")
    manifest = json.loads(path.read_text())
    manifest["feature_cache_id"] = "another-cache"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DecileAnalysisError, match="feature_cache_id"):
        load_decile_inputs(decile_artifacts["cache"], decile_artifacts["results"])


def test_load_inputs_refuses_test_results(decile_artifacts):
    path = decile_artifacts["results"].with_suffix(".manifest.json")
    manifest = json.loads(path.read_text())
    manifest["source_partition"] = "test"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DecileAnalysisError, match="tuning"):
        load_decile_inputs(decile_artifacts["cache"], decile_artifacts["results"])


@pytest.mark.parametrize("mutation,message", [
    ("drop_distance", "missing"),
    ("duplicate_distance", "duplicate"),
    ("wrong_query_count", "query count"),
    ("drop_severity", "six blur severities"),
])
def test_load_inputs_rejects_misalignment(decile_artifacts, mutation, message):
    mutate_decile_artifacts(decile_artifacts, mutation)
    with pytest.raises(DecileAnalysisError, match=message):
        load_decile_inputs(decile_artifacts["cache"], decile_artifacts["results"])
```

- [ ] **Step 2: Run and confirm the module is missing**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty/test_decile_analysis.py -v
```

- [ ] **Step 3: Implement safe, streaming input loading**

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import torch

from .artifacts import iter_records, load_manifest
from .confidence_deciles import confidence_from_logits, detect_padded_tail

EXPECTED_SEVERITIES = frozenset(range(6))


class DecileAnalysisError(ValueError):
    pass


@dataclass
class DecileInputs:
    records_by_image: dict[int, dict[int, dict]]
    distances: dict[tuple[int, int, str], dict[int, torch.Tensor]]
    layer_score_scales: dict[int, dict[str, torch.Tensor]]
    run_metadata: dict


def _unique_rows(rows: list[dict], label: str) -> dict[tuple[int, int, str], dict]:
    result = {}
    for row in rows:
        key = (int(row["image_id"]), int(row["severity"]), row["source_partition"])
        if key in result:
            raise DecileAnalysisError(f"duplicate {label} record for {key}")
        result[key] = row
    return result


def load_decile_inputs(cache_value: str | Path, results_value: str | Path) -> DecileInputs:
    cache, results = Path(cache_value), Path(results_value)
    cache_manifest = load_manifest(cache)
    manifest_path = results.with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise DecileAnalysisError(f"missing result manifest: {manifest_path}")
    result_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if result_manifest.get("feature_cache_id") != cache_manifest.get("artifact_id"):
        raise DecileAnalysisError("result feature_cache_id does not match the cache")
    if result_manifest.get("source_partition") != "tuning":
        raise DecileAnalysisError("confidence-decile analysis accepts tuning results only")
    distance_path = results.parent / result_manifest["query_distance_path"]
    normalizer_path = results.parent / result_manifest["normalizer_path"]
    distance_rows = torch.load(distance_path, map_location="cpu", weights_only=True)
    distance_index = _unique_rows(distance_rows, "query-distance")
    normalizers = torch.load(normalizer_path, map_location="cpu", weights_only=True)
    scales = {int(key): value for key, value in normalizers["layer_score_scales"].items()}
    if set(scales) != set(cache_manifest["decoder_layers"]):
        raise DecileAnalysisError("cache decoder layers and clean-distance scales disagree")

    records_by_image: dict[int, dict[int, dict]] = {}
    seen = set()
    for record in iter_records(cache):
        if record["source_partition"] != "tuning":
            continue
        key = (int(record["image_id"]), int(record["severity"]), "tuning")
        if key in seen:
            raise DecileAnalysisError(f"duplicate feature-cache record for {key}")
        seen.add(key)
        distance_row = distance_index.get(key)
        if distance_row is None:
            raise DecileAnalysisError(f"missing query-distance record for {key}")
        by_layer = {int(layer): values for layer, values in distance_row["query_scores_by_layer"].items()}
        query_count = int(record["logits"].shape[0])
        if set(by_layer) != set(scales):
            raise DecileAnalysisError(f"layer mismatch for {key}")
        if any(int(values.numel()) != query_count for values in by_layer.values()):
            raise DecileAnalysisError(f"query count mismatch for {key}")
        records_by_image.setdefault(key[0], {})[key[1]] = {
            "confidence": confidence_from_logits(record["logits"]),
            "padded_query_ids": detect_padded_tail(record),
            "query_count": query_count,
        }
    extra = set(distance_index) - seen
    if extra:
        raise DecileAnalysisError(f"query-distance artifact has unmatched keys: {sorted(extra)[:5]}")
    for image_id, records in records_by_image.items():
        if set(records) != EXPECTED_SEVERITIES:
            raise DecileAnalysisError(
                f"image {image_id} does not contain all six blur severities: {sorted(records)}"
            )
    return DecileInputs(
        records_by_image,
        {key: {int(layer): value for layer, value in row["query_scores_by_layer"].items()}
         for key, row in distance_index.items()},
        scales,
        {
            "artifact_type": "confidence_decile_scene_uncertainty",
            "feature_cache_id": cache_manifest["artifact_id"],
            "source_result_id": result_manifest["artifact_id"],
            "normalization": result_manifest["normalization"],
            "k": result_manifest["k"],
            "source_partition": "tuning",
            "severities": sorted(EXPECTED_SEVERITIES),
        },
    )
```

- [ ] **Step 4: Run and confirm validation tests pass**

- [ ] **Step 5: Commit**

```bash
git add src/scene_uncertainty/decile_analysis.py \
  tests/scene_uncertainty/decile_test_utils.py \
  tests/scene_uncertainty/test_decile_analysis.py
git commit -m "feat: load confidence-decile inputs safely"
```

## Task 5: Construct all experiment and control rows

**Files:**
- Modify: `src/scene_uncertainty/decile_analysis.py`
- Modify: `tests/scene_uncertainty/test_decile_analysis.py`

- [ ] **Step 1: Add failing orchestration tests**

```python
def test_analysis_emits_deciles_and_controls(decile_artifacts):
    rows, _ = analyze_deciles(load_decile_inputs(
        decile_artifacts["cache"], decile_artifacts["results"]
    ))
    labels = {(r["membership_mode"], r["confidence_bin"], r["padding_mode"]) for r in rows}
    assert {("dynamic", name, "filtered") for name in DECILE_NAMES} <= labels
    assert {("frozen", name, "filtered") for name in DECILE_NAMES} <= labels
    assert ("shared", "all_valid", "filtered") in labels
    assert ("legacy", "all_300", "unfiltered") in labels
    assert ("dynamic", "decile_00_10", "unfiltered") in labels
    assert ("frozen", "decile_00_10", "unfiltered") in labels


def test_frozen_ids_stay_fixed_and_dynamic_overlap_is_bounded(decile_artifacts):
    rows, _ = analyze_deciles(load_decile_inputs(
        decile_artifacts["cache"], decile_artifacts["results"]
    ))
    selected = {
        (r["severity"], r["membership_mode"]): r["selected_query_ids"]
        for r in rows
        if r["confidence_bin"] == "decile_00_10" and r["padding_mode"] == "filtered"
        and r["signal"] == "confidence" and r["aggregation"] == "mean"
    }
    assert selected[(0, "frozen")] == selected[(5, "frozen")]
    assert all(0.0 <= row["clean_overlap"] <= 1.0 for row in rows)


def test_padding_control_is_the_only_bottom_bin_that_keeps_padded_ids(decile_artifacts):
    rows, diagnostics = analyze_deciles(load_decile_inputs(
        decile_artifacts["cache"], decile_artifacts["results"]
    ))
    padded = set(diagnostics["images"]["11"]["union_padded_query_ids"])
    assert all(
        not padded.intersection(row["selected_query_ids"])
        for row in rows if row["padding_mode"] == "filtered"
    )
    assert any(
        padded.intersection(row["selected_query_ids"])
        for row in rows
        if row["padding_mode"] == "unfiltered" and row["confidence_bin"] == "decile_00_10"
    )


def test_legacy_is_persistence_q90_only(decile_artifacts):
    rows, _ = analyze_deciles(load_decile_inputs(
        decile_artifacts["cache"], decile_artifacts["results"]
    ))
    legacy = [row for row in rows if row["confidence_bin"] == "all_300"]
    assert {row["signal"] for row in legacy} == {"persistence"}
    assert {row["aggregation"] for row in legacy} == {"q90"}
```

- [ ] **Step 2: Run and confirm `analyze_deciles` is missing**

- [ ] **Step 3: Implement cross-severity construction**

Import `confidence_deciles`, `memberships_by_severity`, `score_selection`, and `jaccard_overlap`, then add:

```python
def _valid_indices(query_count: int, padded: torch.Tensor) -> torch.Tensor:
    keep = torch.ones(query_count, dtype=torch.bool)
    keep[padded.long()] = False
    return torch.where(keep)[0]


def analyze_deciles(inputs: DecileInputs) -> tuple[list[dict], dict]:
    rows, images = [], {}
    for image_id, records in sorted(inputs.records_by_image.items()):
        padded = torch.tensor(sorted({
            int(query_id) for record in records.values()
            for query_id in record["padded_query_ids"]
        }), dtype=torch.long)
        query_count = records[0]["query_count"]
        valid = _valid_indices(query_count, padded)
        memberships = memberships_by_severity(records, padded)
        clean_dynamic = memberships[0]["dynamic"]
        all_indices = torch.arange(query_count)
        unfiltered_clean = confidence_deciles(records[0]["confidence"], all_indices)
        tails = {
            str(severity): record["padded_query_ids"].tolist()
            for severity, record in sorted(records.items())
        }
        images[str(image_id)] = {
            "union_padded_query_ids": padded.tolist(),
            "union_padded_count": int(padded.numel()),
            "padded_query_ids_by_severity": tails,
            "padded_count_by_severity": {key: len(value) for key, value in tails.items()},
            "tail_identical_across_severities": len({tuple(value) for value in tails.values()}) == 1,
        }
        for severity, record in sorted(records.items()):
            distances = inputs.distances[(image_id, severity, "tuning")]
            for mode in ("dynamic", "frozen"):
                for name, indices in memberships[severity][mode].items():
                    rows.extend(score_selection(
                        image_id=image_id, severity=severity, source_partition="tuning",
                        membership_mode=mode, confidence_bin=name, padding_mode="filtered",
                        indices=indices, confidence=record["confidence"],
                        query_scores_by_layer=distances,
                        layer_score_scales=inputs.layer_score_scales,
                        clean_overlap=jaccard_overlap(clean_dynamic[name], indices),
                    ))
            rows.extend(score_selection(
                image_id=image_id, severity=severity, source_partition="tuning",
                membership_mode="shared", confidence_bin="all_valid", padding_mode="filtered",
                indices=valid, confidence=record["confidence"],
                query_scores_by_layer=distances, layer_score_scales=inputs.layer_score_scales,
                clean_overlap=1.0,
            ))
            dynamic_all = confidence_deciles(record["confidence"], all_indices)["decile_00_10"]
            for mode, indices in (
                ("dynamic", dynamic_all),
                ("frozen", unfiltered_clean["decile_00_10"]),
            ):
                rows.extend(score_selection(
                    image_id=image_id, severity=severity, source_partition="tuning",
                    membership_mode=mode, confidence_bin="decile_00_10",
                    padding_mode="unfiltered", indices=indices,
                    confidence=record["confidence"], query_scores_by_layer=distances,
                    layer_score_scales=inputs.layer_score_scales,
                    clean_overlap=jaccard_overlap(unfiltered_clean["decile_00_10"], indices),
                ))
            rows.extend(score_selection(
                image_id=image_id, severity=severity, source_partition="tuning",
                membership_mode="legacy", confidence_bin="all_300", padding_mode="unfiltered",
                indices=all_indices, confidence=record["confidence"],
                query_scores_by_layer=distances, layer_score_scales=inputs.layer_score_scales,
                clean_overlap=1.0, aggregations=("q90",), include_confidence=False,
            ))
    return rows, {"images": images}
```

- [ ] **Step 4: Run all analysis tests**

- [ ] **Step 5: Commit**

```bash
git add src/scene_uncertainty/decile_analysis.py tests/scene_uncertainty/test_decile_analysis.py
git commit -m "feat: construct confidence-decile controls"
```

## Task 6: Summarize trends and paired performance

**Files:**
- Create: `src/scene_uncertainty/decile_reporting.py`
- Create: `tests/scene_uncertainty/test_decile_reporting.py`

- [ ] **Step 1: Write failing summary tests**

Create two synthetic images over six severities. Persistence rises for both; confidence uncertainty rises for one and stays flat for one. Assert:

```python
def test_summary_reports_complete_trends_and_paired_advantage():
    summary = summarize_decile_rows(synthetic_rows(), {"source_partition": "tuning"})
    persistence = group(summary, "persistence", "layer_2")
    confidence = group(summary, "confidence", "confidence")
    comparison = summary["comparisons"][0]
    assert persistence["median_spearman"] == 1.0
    assert persistence["scored_severity_count"] == persistence["total_severity_count"] == 12
    assert comparison["persistence_minus_confidence_spearman"] == pytest.approx(
        persistence["median_spearman"] - confidence["median_spearman"]
    )
    assert comparison["persistence_image_win_rate"] == 0.5


def test_ranking_requires_complete_deployable_groups():
    summary = summarize_decile_rows(rows_with_incomplete_group(), {})
    ranked = summary["ranked_layer_2_persistence"]
    assert all(row["scored_severity_count"] == row["total_severity_count"] for row in ranked)
    assert all(row["membership_mode"] in {"dynamic", "shared"} for row in ranked)


def test_duplicate_score_key_is_rejected():
    rows = synthetic_rows()
    with pytest.raises(ValueError, match="duplicate"):
        summarize_decile_rows(rows + [rows[0]], {})
```

- [ ] **Step 2: Run and confirm the reporting module is missing**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty/test_decile_reporting.py -v
```

- [ ] **Step 3: Implement grouping, per-image metrics, pairs, and ranking**

Use `GROUP_KEYS = (signal, membership_mode, confidence_bin, aggregation, score_scope, padding_mode)` and `ROW_KEYS = (image_id, severity, *GROUP_KEYS)`. For each group, call existing `monotonicity_metrics` per image and publish:

```python
group = {
    **dict(zip(GROUP_KEYS, keys)),
    "image_count": len(image_metrics),
    "scored_image_count": len(measured),
    "scored_severity_count": sum(value["finite_count"] for value in image_metrics),
    "total_severity_count": sum(value["total_count"] for value in image_metrics),
    "median_spearman": finite_median(value["spearman"] for value in measured),
    "mean_adjacent_monotonicity": finite_mean(
        value["adjacent_monotonicity"] for value in measured
    ),
    "mean_violation_magnitude": finite_mean(
        value["violation_magnitude"] for value in measured
    ),
    "endpoint_increase_rate": finite_mean(
        value["endpoint_increase"] for value in measured
    ),
    "mean_clean_overlap": finite_mean(group_frame["clean_overlap"]),
    "mean_clean_overlap_by_severity": {
        str(int(severity)): float(values.mean())
        for severity, values in group_frame.groupby("severity")["clean_overlap"]
    },
    "median_selected_count_by_severity": {
        str(int(severity)): float(values.median())
        for severity, values in group_frame.groupby("severity")["selected_count"]
    },
}
```

Pair layer-2 persistence and confidence on membership mode, confidence bin, aggregation, padding mode, and image ID. Report the difference between group median Spearman values and the fraction of paired images where persistence Spearman is larger.

Rank filtered, full-coverage layer-2 persistence groups whose membership is `dynamic` or `shared`. Sort by median Spearman descending, adjacent non-decrease descending, then violation magnitude ascending. Convert NumPy scalars to Python values and non-finite summary values to `None` before JSON serialization.

- [ ] **Step 4: Run and confirm summary tests pass**

- [ ] **Step 5: Commit**

```bash
git add src/scene_uncertainty/decile_reporting.py tests/scene_uncertainty/test_decile_reporting.py
git commit -m "feat: compare decile persistence with confidence"
```

## Task 7: Write artifacts, plots, and a plain-language report

**Files:**
- Modify: `src/scene_uncertainty/decile_reporting.py`
- Modify: `tests/scene_uncertainty/test_decile_reporting.py`

- [ ] **Step 1: Add failing artifact tests**

```python
def test_write_report_creates_every_declared_artifact(tmp_path):
    write_decile_report(
        synthetic_rows(), tmp_path,
        run_metadata={"source_partition": "tuning"}, diagnostics={"images": {}},
    )
    expected = {
        "per_scene.csv", "summary.json", "confidence_decile_heatmap.png",
        "blur_curves.png", "dynamic_vs_frozen.png", "padding_sensitivity.png",
        "easy-report.md",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    text = (tmp_path / "summary.json").read_text()
    assert "NaN" not in text
    assert json.loads(text)["run_metadata"]["source_partition"] == "tuning"


def test_easy_report_names_winner_controls_and_test_status(tmp_path):
    write_decile_report(synthetic_rows(), tmp_path, {"source_partition": "tuning"}, {})
    report = (tmp_path / "easy-report.md").read_text()
    assert "Best confidence range" in report
    assert "confidence alone" in report
    assert "all-query benchmark" in report
    assert "held-out test images were not used" in report
```

- [ ] **Step 2: Run and confirm `write_decile_report` is missing**

Use the Task 6 pytest command.

- [ ] **Step 3: Implement atomic output helpers and the writer**

```python
def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _save_figure(figure, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    figure.tight_layout()
    figure.savefig(temporary, format="png", dpi=160)
    plt.close(figure)
    os.replace(temporary, path)


def write_decile_report(rows, output_dir, run_metadata, diagnostics) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize_decile_rows(rows, run_metadata, diagnostics)
    csv_path = output / "per_scene.csv"
    csv_temporary = csv_path.with_suffix(".csv.tmp")
    pd.DataFrame(rows).drop(columns=["selected_query_ids"]).to_csv(csv_temporary, index=False)
    os.replace(csv_temporary, csv_path)
    _atomic_text(
        output / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False),
    )
    _write_heatmap(summary, output / "confidence_decile_heatmap.png")
    _write_blur_curves(rows, output / "blur_curves.png")
    _write_dynamic_frozen(summary, output / "dynamic_vs_frozen.png")
    _write_padding_sensitivity(summary, output / "padding_sensitivity.png")
    _atomic_text(output / "easy-report.md", _easy_report(summary))
```

- [ ] **Step 4: Implement four plots with fixed meanings**

- Heatmap: filtered q90, with layer-2 persistence and matched confidence rows and ten ordered decile columns.
- Blur curves: filtered dynamic q90; layer-2 persistence and confidence in separate panels; median clean-relative score per severity.
- Dynamic versus frozen: filtered layer-2 q90 median Spearman, paired bars per decile.
- Padding sensitivity: lowest decile q90, showing persistence and confidence with filtering on and off.

Each plot helper must assert all expected deciles or control rows exist. Import `DECILE_NAMES` and use it for order rather than alphabetical sorting. Save every figure through `_save_figure`.

- [ ] **Step 5: Implement deterministic easy Markdown**

`_easy_report(summary)` must locate the top ranked deployable row, its matched confidence comparison, the frozen version, the unfiltered bottom-bin control, `all_valid`, and legacy `all_300/q90/layer_2`. Emit:

```text
# Confidence-Decile Blur Experiment
## Short answer
## Best confidence range
## Persistence versus confidence alone
## Dynamic versus frozen queries
## Effect of padded queries
## Metrics in plain language
## What this does not prove
## Next decision
```

Include Spearman, adjacent rate, endpoint rate, confidence difference, per-image win rate, legacy benchmark, padding count, and coverage. Never call either score a probability. Finish with “The held-out test images were not used.”

- [ ] **Step 6: Run reporting tests and confirm JSON and PNG output**

- [ ] **Step 7: Commit**

```bash
git add src/scene_uncertainty/decile_reporting.py tests/scene_uncertainty/test_decile_reporting.py
git commit -m "feat: report confidence-decile experiment"
```

## Task 8: Expose the tuning-only CLI and add an integration test

**Files:**
- Modify: `src/scene_uncertainty/cli.py`
- Modify: `src/scene_uncertainty/pipeline.py`
- Modify: `tests/scene_uncertainty/test_cli.py`
- Create: `tests/scene_uncertainty/test_decile_integration.py`

- [ ] **Step 1: Write failing parser tests**

Add `analyze-confidence-deciles` to every expected command set in `test_cli.py`, then add:

```python
def test_confidence_decile_command_has_only_cache_results_and_output():
    args = build_parser().parse_args([
        "analyze-confidence-deciles",
        "--cache", "cache", "--results", "raw_k5.csv", "--output", "report",
    ])
    assert vars(args) == {
        "command": "analyze-confidence-deciles",
        "cache": "cache", "results": "raw_k5.csv", "output": "report",
    }
```

No `--partition` argument is intentional: the loader accepts only a tuning manifest.

- [ ] **Step 2: Write the failing six-severity integration test**

Call `write_decile_artifacts(tmp_path)` from `decile_test_utils.py`; it creates the 20-query, three-layer, six-severity tuning cache, identical tail queries 18 and 19, matching query distances, and identity layer scales. Invoke:

```python
assert main([
    "analyze-confidence-deciles",
    "--cache", str(cache), "--results", str(results), "--output", str(output),
]) == 0
```

Then assert:

```python
summary = json.loads((output / "summary.json").read_text())
assert summary["run_metadata"]["source_partition"] == "tuning"
assert summary["diagnostics"]["images"]["11"]["union_padded_query_ids"] == [18, 19]
legacy = next(
    group for group in summary["groups"]
    if group["membership_mode"] == "legacy" and group["score_scope"] == "layer_2"
)
assert legacy["scored_severity_count"] == legacy["total_severity_count"] == 6
assert {row["confidence_bin"] for row in summary["ranked_layer_2_persistence"]} >= {
    "all_valid", "decile_00_10", "decile_90_100",
}
```

- [ ] **Step 3: Run focused tests and confirm the command is unknown**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/scene_uncertainty/test_cli.py tests/scene_uncertainty/test_decile_integration.py -v
```

- [ ] **Step 4: Add parser help and arguments**

Update the CLI module description to seven commands. Define an epilog that states this command performs no inference or kNN, accepts tuning results only, and refuses a finished output. Add:

```python
deciles = add(
    "analyze-confidence-deciles",
    "Compare persistence and model confidence across confidence deciles.",
    _DECILE_EPILOG,
)
deciles.add_argument("--cache", required=True, help="six-severity evaluation feature cache")
deciles.add_argument("--results", required=True, help="tuning CSV written by evaluate-knn")
deciles.add_argument("--output", required=True, help="new confidence-decile report directory")
```

- [ ] **Step 5: Add the pipeline wrapper**

```python
def command_analyze_confidence_deciles(args) -> None:
    cache = _existing_artifact(args.cache, "--cache")
    results = _existing_path(args.results, "--results")
    output = Path(args.output)
    if (output / "summary.json").exists():
        raise FileExistsError(f"Confidence-decile report is already complete: {output}")
    try:
        inputs = load_decile_inputs(cache, results)
        rows, diagnostics = analyze_deciles(inputs)
        write_decile_report(rows, output, inputs.run_metadata, diagnostics)
    except ValueError as error:
        raise PipelineError(f"Cannot analyze confidence deciles: {error}") from error
    _report(f"analyze-confidence-deciles: summarized {len(rows)} score rows into {output}")
```

Import the three called functions and add the command to `COMMANDS`.

- [ ] **Step 6: Run CLI and integration tests**

Expected: command help works, the seven outputs exist, invalid provenance returns exit code 2, and there is no test-partition switch.

- [ ] **Step 7: Commit**

```bash
git add src/scene_uncertainty/cli.py src/scene_uncertainty/pipeline.py \
  tests/scene_uncertainty/test_cli.py tests/scene_uncertainty/test_decile_integration.py
git commit -m "feat: add confidence-decile analysis command"
```

## Task 9: Document and verify the implementation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add the cache-only tuning command after the existing report command**

```bash
$UE_PY tools/scene_uncertainty.py analyze-confidence-deciles \
  --cache $OUT/blur_cache \
  --results $OUT/results/raw_k5.csv \
  --output $OUT/reports/confidence_deciles_raw_k5
```

Explain that it reuses logits, boxes, fingerprints, query distances, and layer scales. It does no detector pass, bank build, kNN, training, calibration, or test scoring.

- [ ] **Step 2: Run all new and directly affected tests**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/scene_uncertainty/test_confidence_deciles.py \
  tests/scene_uncertainty/test_decile_scoring.py \
  tests/scene_uncertainty/test_decile_analysis.py \
  tests/scene_uncertainty/test_decile_reporting.py \
  tests/scene_uncertainty/test_decile_integration.py \
  tests/scene_uncertainty/test_cli.py -v
```

- [ ] **Step 3: Run the complete suite**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty -q
```

Expected: the existing 267 tests plus every new test pass with zero failures.

- [ ] **Step 4: Verify formatting and safe artifact reads**

```bash
git diff --check
rg -n "weights_only=False" \
  src/scene_uncertainty/confidence_deciles.py \
  src/scene_uncertainty/decile_scoring.py \
  src/scene_uncertainty/decile_analysis.py \
  src/scene_uncertainty/decile_reporting.py
```

Expected: the diff check prints nothing, and no new production loader uses unsafe unpickling.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md
git commit -m "docs: explain confidence-decile analysis"
```

## Task 10: Run the real tuning experiment and verify the result

**Files:**
- Create after the run: `docs/scene-uncertainty-confidence-decile-results.md`

- [ ] **Step 1: Sync the verified commit to alienware2 using the existing rsync exclusions**

Record the exact source commit with `git rev-parse HEAD`. Do not copy, overwrite, or delete any existing pilot output.

- [ ] **Step 2: Run only the cache analysis**

From `/home/alienware2/YuchenZ/UE/philip_sa-scene-unc`:

```bash
export OUT=output/scene_uncertainty/pilot
export UE_PY=/home/alienware2/miniconda3/envs/UE/bin/python
$UE_PY tools/scene_uncertainty.py analyze-confidence-deciles \
  --cache $OUT/blur_cache \
  --results $OUT/results/raw_k5.csv \
  --output $OUT/reports/confidence_deciles_raw_k5
```

Expected: no GPU extraction and no kNN progress, only one completion line.

- [ ] **Step 3: Verify invariants before reading the winner**

```python
import json
from pathlib import Path

root = Path("output/scene_uncertainty/pilot/reports/confidence_deciles_raw_k5")
summary = json.loads((root / "summary.json").read_text())
assert summary["run_metadata"]["source_partition"] == "tuning"
assert summary["run_metadata"]["k"] == 5
assert summary["run_metadata"]["normalization"] == "raw"
assert set(summary["run_metadata"]["severities"]) == set(range(6))
assert all(
    group["scored_severity_count"] == group["total_severity_count"]
    for group in summary["ranked_layer_2_persistence"]
)
legacy = next(
    group for group in summary["groups"]
    if group["signal"] == "persistence"
    and group["membership_mode"] == "legacy"
    and group["confidence_bin"] == "all_300"
    and group["aggregation"] == "q90"
    and group["score_scope"] == "layer_2"
)
assert legacy["image_count"] == 250
assert legacy["total_severity_count"] == 1500
assert abs(legacy["median_spearman"] - 0.6) < 1e-12
```

If the legacy assertion fails, stop and diagnose artifact alignment or float conversion before interpreting any decile.

- [ ] **Step 4: Copy and inspect the easy report locally**

Copy remote `easy-report.md` to `docs/scene-uncertainty-confidence-decile-results.md`. Confirm it names the winning bin and aggregation, persistence and confidence metrics, their difference and per-image win rate, dynamic/frozen behavior, padding sensitivity, the all-query benchmark, full coverage, and the untouched test partition.

- [ ] **Step 5: Verify and commit only the result report**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/scene_uncertainty -q
git diff --check
git add docs/scene-uncertainty-confidence-decile-results.md
git commit -m "docs: report confidence-decile blur experiment"
```

## Completion checklist

- [ ] Every behavior was introduced by a failing test.
- [ ] Padding requires repeated suffix identity across boxes, logits, and all layers.
- [ ] Primary bins exclude the image-level union of padded IDs.
- [ ] Dynamic and frozen memberships remain separate.
- [ ] Persistence and `1 - confidence` use identical IDs and summaries.
- [ ] `all_valid`, legacy all-300, and unfiltered bottom-decile controls exist.
- [ ] Layer 2 is primary; other scopes remain diagnostic.
- [ ] Rankings exclude incomplete and frozen-only groups from deployable winners.
- [ ] Inputs are safe-loaded and provenance-checked.
- [ ] The CLI cannot select the test partition.
- [ ] The real run reproduces legacy layer-2 q90 Spearman 0.600 before ranking.
- [ ] The report compares confidence fairly and never calls a score a probability.
- [ ] The held-out 250-image test partition remains unused.

## Exact test helpers referenced by the tasks

Task 4 creates this shared artifact builder so Task 8 does not invent a second synthetic artifact format:

```python
# tests/scene_uncertainty/decile_test_utils.py
from __future__ import annotations

import json
from pathlib import Path
import torch

from src.scene_uncertainty.artifacts import ShardWriter, load_manifest


LAYERS = (0, 1, 2)
QUERY_COUNT = 20
PERSISTENCE_DIM = 4


def write_decile_artifacts(root: Path, severities=range(6)) -> dict[str, Path]:
    cache = root / "cache"
    metadata = {
        "source_kind": "evaluation", "image_ids": [11],
        "checkpoint_sha256": "checkpoint", "decoder_layers": list(LAYERS),
        "query_count": QUERY_COUNT, "persistence_dim": PERSISTENCE_DIM,
    }
    with ShardWriter(cache, metadata, shard_size=20) as writer:
        for severity in severities:
            confidence = torch.linspace(0.05, 0.95, QUERY_COUNT).sub(severity * 0.005).clamp(0.002, 0.998)
            logits = torch.full((QUERY_COUNT, 80), -20.0)
            logits[:, 0] = torch.logit(confidence)
            logits[-2:, 0] = torch.logit(torch.tensor(0.002))
            boxes = torch.arange(QUERY_COUNT * 4, dtype=torch.float32).reshape(QUERY_COUNT, 4) / 100
            boxes[-1] = boxes[-2]
            logits[-1] = logits[-2]
            layers = {
                layer_id: (
                    torch.arange(QUERY_COUNT * PERSISTENCE_DIM, dtype=torch.float32)
                    .reshape(QUERY_COUNT, PERSISTENCE_DIM)
                    .add(layer_id + severity)
                )
                for layer_id in LAYERS
            }
            for values in layers.values():
                values[-1] = values[-2]
            writer.add({
                "image_id": 11, "severity": int(severity), "source_partition": "tuning",
                "boxes": boxes, "logits": logits.to(torch.float16),
                "layers": {layer: values.to(torch.float16) for layer, values in layers.items()},
            })
    results = root / "results" / "raw_k5.csv"
    results.parent.mkdir(parents=True)
    results.write_text("source result rows are not read by this command\n", encoding="utf-8")
    distance_path = results.with_suffix(".query_distances.pt")
    distance_rows = [
        {
            "image_id": 11, "severity": int(severity), "source_partition": "tuning",
            "query_scores_by_layer": {
                layer: torch.arange(QUERY_COUNT, dtype=torch.float16).mul(0.01).add(
                    severity * 0.02 + layer * 0.001
                )
                for layer in LAYERS
            },
        }
        for severity in severities
    ]
    torch.save(distance_rows, distance_path)
    normalizer_path = results.with_suffix(".normalizers.pt")
    torch.save({
        "layer_score_scales": {
            layer: {"center": torch.tensor(0.0), "scale": torch.tensor(1.0)}
            for layer in LAYERS
        }
    }, normalizer_path)
    manifest = {
        "artifact_id": "synthetic-results", "artifact_type": "knn_scene_uncertainty_results",
        "feature_cache_id": load_manifest(cache)["artifact_id"],
        "source_partition": "tuning", "normalization": "raw", "k": 5,
        "query_distance_path": distance_path.name,
        "normalizer_path": normalizer_path.name,
    }
    results.with_suffix(".manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return {"cache": cache, "results": results}


def mutate_decile_artifacts(artifacts: dict[str, Path], mutation: str) -> None:
    results = artifacts["results"]
    distance_path = results.with_suffix(".query_distances.pt")
    rows = torch.load(distance_path, map_location="cpu", weights_only=True)
    if mutation == "drop_distance":
        rows.pop()
    elif mutation == "duplicate_distance":
        rows.append(dict(rows[0]))
    elif mutation == "wrong_query_count":
        rows[0]["query_scores_by_layer"][2] = rows[0]["query_scores_by_layer"][2][:-1]
    elif mutation == "drop_severity":
        rows = [row for row in rows if int(row["severity"]) != 5]
        manifest = load_manifest(artifacts["cache"])
        shard = artifacts["cache"] / manifest["shards"][0]
        records = torch.load(shard, map_location="cpu", weights_only=True)
        torch.save([record for record in records if int(record["severity"]) != 5], shard)
    else:
        raise ValueError(f"unknown fixture mutation: {mutation}")
    torch.save(rows, distance_path)
```

At the top of `test_decile_analysis.py`, define the fixture exactly as:

```python
@pytest.fixture
def decile_artifacts(tmp_path):
    return write_decile_artifacts(tmp_path)
```

Task 6 defines its row helpers in `test_decile_reporting.py`:

```python
def synthetic_rows():
    rows = []
    for image_id in (1, 2):
        for severity in range(6):
            common = {
                "image_id": image_id, "severity": severity,
                "source_partition": "tuning", "membership_mode": "dynamic",
                "confidence_bin": "decile_00_10", "aggregation": "mean",
                "padding_mode": "filtered", "selected_count": 2,
                "clean_overlap": 1.0,
            }
            rows.append({
                **common, "signal": "persistence", "score_scope": "layer_2",
                "score": float(severity),
            })
            rows.append({
                **common, "signal": "confidence", "score_scope": "confidence",
                "score": float(severity if image_id == 1 else 0.0),
            })
    return rows


def group(summary, signal, scope):
    return next(
        value for value in summary["groups"]
        if value["signal"] == signal and value["score_scope"] == scope
    )


def rows_with_incomplete_group():
    rows = synthetic_rows()
    incomplete = []
    for row in rows:
        if row["signal"] != "persistence":
            continue
        copy = {**row, "confidence_bin": "decile_10_20"}
        if copy["image_id"] == 1 and copy["severity"] == 5:
            copy["score"] = float("nan")
        incomplete.append(copy)
    return rows + incomplete
```
