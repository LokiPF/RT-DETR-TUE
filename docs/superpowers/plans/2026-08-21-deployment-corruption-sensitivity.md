# Deployment Corruption Sensitivity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a cache-only evaluation that compares persistence distance with the model's own confidence as signals of image corruption across six blur levels, while showing confidence quintiles beside the existing deciles and leaving the existing decile command unchanged.

**Architecture:** Preserve the existing cache loader and query scorer as the trusted data path, generalize only their binning/provenance seams, and put the new experiment in four focused modules: row construction, deployment metrics, plots, and reporting. Each candidate is oriented once from tuning-set signed Spearman, then evaluated with per-severity clean-versus-corrupted AUROC. The command consumes only the saved blur cache and saved query-distance CSV; it must never import or run DETR, the comparison bank, or kNN.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, SciPy, pandas, Matplotlib, pytest, the existing `scene_uncertainty` CLI and artifact formats.

**Approved design:** [`docs/superpowers/specs/2026-08-21-deployment-corruption-sensitivity-design.md`](../specs/2026-08-21-deployment-corruption-sensitivity-design.md)

---

## Ground rules

- Work in a dedicated worktree created with `superpowers:using-git-worktrees` before changing production code.
- Use `/home/yuchen/miniconda3/envs/UE/bin/python` for every Python and pytest command.
- Follow red-green-refactor for every task: add one focused failing test, run it and inspect the expected failure, make the smallest production change, then rerun the focused tests.
- Do not alter the files or output of `analyze-confidence-deciles`. New fields may appear only when the new command explicitly requests them.
- The tuning partition is the only allowed input. Do not add a `--partition` option and do not run the held-out partition.
- Never turn AUROC into a probability claim. The output is a corruption-ranking score until a later calibration experiment exists.
- Do not add scikit-learn. Use SciPy's tie-aware `rankdata` for AUROC.

## Required row and candidate contracts

Every new raw score row must contain these identifying fields:

```python
ROW_KEY = (
    "image_id",
    "severity",
    "signal",
    "bucket_scheme",
    "confidence_bin",
    "membership_mode",
    "padding_mode",
    "aggregation",
    "score_scope",
)

CANDIDATE_KEY = (
    "signal",
    "bucket_scheme",
    "confidence_bin",
    "membership_mode",
    "padding_mode",
    "aggregation",
    "score_scope",
)
```

The two schemes and their legal new-command selections are:

| Scheme | Filtered dynamic/frozen | Unfiltered diagnostic |
|---|---:|---:|
| decile | all ten bins | `decile_00_10` only |
| quintile | all five bins | `quintile_00_20` only |

For each selected query set, emit persistence signals for `combined`, `layer_0`, `layer_1`, and `layer_2`, plus the matched `confidence` signal. Emit `mean`, `q90`, and `top20_mean` for both signals. Layer 2 is the primary persistence scope for ranking and the report.

## Task 1: Generalize deterministic confidence partitions

**Files:**

- Modify: `src/scene_uncertainty/confidence_deciles.py:11,211-370`
- Test: `tests/scene_uncertainty/test_confidence_deciles.py`

- [ ] **Step 1: Add failing tests for quintile membership and compatibility**

Add imports for `QUINTILE_NAMES`, `confidence_buckets`, and `memberships_by_scheme_severity`, then add tests covering:

```python
def test_confidence_quintiles_split_twenty_queries_lowest_first() -> None:
    confidence = torch.tensor([
        0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.0,
        0.91, 0.11, 0.81, 0.21, 0.71, 0.31, 0.61, 0.41, 0.51, 0.01,
    ])
    bins = confidence_buckets(
        confidence,
        torch.arange(20),
        names=QUINTILE_NAMES,
        noun="confidence quintiles",
    )

    assert tuple(bins) == QUINTILE_NAMES
    assert [len(bins[name]) for name in QUINTILE_NAMES] == [4, 4, 4, 4, 4]
    assert torch.equal(torch.cat(list(bins.values())).sort().values, torch.arange(20))


def test_confidence_buckets_break_ties_by_ascending_query_id() -> None:
    confidence = torch.ones(10)
    bins = confidence_buckets(
        confidence,
        torch.tensor([9, 3, 7, 1, 5, 0, 8, 2, 6, 4]),
        names=QUINTILE_NAMES,
        noun="confidence quintiles",
    )

    assert [values.tolist() for values in bins.values()] == [
        [0, 1], [2, 3], [4, 5], [6, 7], [8, 9],
    ]


def test_legacy_confidence_deciles_match_generic_partition() -> None:
    confidence = torch.linspace(0.0, 1.0, 23)
    valid = torch.tensor([22, 0, 5, 12, 3, 7, 18, 2, 9, 14, 4, 8, 17, 6, 1])

    legacy = confidence_deciles(confidence, valid, label="sample")
    generic = confidence_buckets(
        confidence,
        valid,
        names=DECILE_NAMES,
        noun="confidence deciles",
        label="sample",
    )

    assert legacy.keys() == generic.keys()
    assert all(torch.equal(legacy[name], generic[name]) for name in DECILE_NAMES)
```

Also test that five valid queries are enough for quintiles, four are rejected, duplicate/out-of-range/non-finite inputs retain the current strict validation, and `memberships_by_scheme_severity(..., names=QUINTILE_NAMES)` returns dynamic, frozen, all-valid, and clean-overlap data for all six severities.

- [ ] **Step 2: Run the focused tests and confirm they fail because the generic API does not exist**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/scene_uncertainty/test_confidence_deciles.py -q
```

Expected: import errors for the new names, while all pre-existing tests still collect.

- [ ] **Step 3: Add the generic partition primitive and compatibility wrappers**

Add these constants and function shape:

```python
DECILE_NAMES = tuple(f"decile_{lower:02d}_{lower + 10:02d}" for lower in range(0, 100, 10))
QUINTILE_NAMES = tuple(
    f"quintile_{lower:02d}_{lower + 20:02d}" for lower in range(0, 100, 20)
)


def confidence_buckets(
    confidence: Tensor,
    valid_indices: Tensor,
    *,
    names: tuple[str, ...],
    noun: str,
    label: str = "",
) -> dict[str, Tensor]:
    if not names or len(set(names)) != len(names):
        raise ValueError("confidence bucket names must be non-empty and unique")
    scores = confidence.float().cpu()
    if scores.ndim != 1:
        raise ValueError(
            f"confidence must be one score per query, got shape {tuple(confidence.shape)}"
        )
    _require_index_dtype(valid_indices, "valid query indices")
    valid = torch.sort(valid_indices.reshape(-1).long().cpu()).values
    named = f" for {label}" if label else ""
    if valid.numel() < len(names):
        raise ValueError(
            f"{noun} need at least {len(names)} valid queries, got {valid.numel()}{named}"
        )
    if valid.unique().numel() != valid.numel():
        raise ValueError(f"valid query indices must be unique{named}")
    if int(valid.min()) < 0 or int(valid.max()) >= scores.numel():
        raise ValueError(f"valid query index lies outside the confidence vector{named}")
    selected = scores.index_select(0, valid)
    if not bool(torch.isfinite(selected).all()):
        raise ValueError(f"confidence must be finite to rank queries{named}")
    order = torch.argsort(selected, stable=True)
    chunks = torch.tensor_split(valid.index_select(0, order), len(names))
    return {name: chunk for name, chunk in zip(names, chunks)}


def confidence_deciles(
    confidence: Tensor, valid_indices: Tensor, label: str = ""
) -> dict[str, Tensor]:
    return confidence_buckets(
        confidence,
        valid_indices,
        names=DECILE_NAMES,
        noun="confidence deciles",
        label=label,
    )
```

Preserve the current `confidence_deciles` error text exactly by special-casing `noun == "confidence deciles"` to say “at least ten” rather than “at least 10” if the existing tests assert the literal text.

Rename the body of `memberships_by_severity` to `memberships_by_scheme_severity` and add keyword-only `names` and `noun` parameters. Replace its two calls to `confidence_deciles` with `confidence_buckets(..., names=names, noun=noun)`. Keep the old entry point as an exact wrapper:

```python
def memberships_by_severity(
    records_by_severity: dict[int, dict], padded: Tensor
) -> dict[int, dict]:
    return memberships_by_scheme_severity(
        records_by_severity,
        padded,
        names=DECILE_NAMES,
        noun="confidence deciles",
    )
```

- [ ] **Step 4: Run focused and regression tests**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/scene_uncertainty/test_confidence_deciles.py \
  tests/scene_uncertainty/test_decile_analysis.py -q
```

Expected: all pass, including every old decile validation-message test.

- [ ] **Step 5: Commit**

```bash
git add src/scene_uncertainty/confidence_deciles.py \
  tests/scene_uncertainty/test_confidence_deciles.py
git commit -m "feat: add deterministic confidence quintiles"
```

## Task 2: Add opt-in bucket provenance to the shared scorer

**Files:**

- Modify: `src/scene_uncertainty/decile_scoring.py:11-30,185-320`
- Test: `tests/scene_uncertainty/test_decile_scoring.py`

- [ ] **Step 1: Add failing compatibility and scheme-validation tests**

Use the existing scorer fixture/helper to assert:

```python
def test_score_selection_keeps_legacy_rows_byte_compatible() -> None:
    rows = score_selection(**selection_arguments())
    assert all("bucket_scheme" not in row for row in rows)


def test_score_selection_records_explicit_quintile_scheme() -> None:
    arguments = selection_arguments()
    arguments["confidence_bin"] = "quintile_00_20"
    rows = score_selection(**arguments, bucket_scheme="quintile")
    assert {row["bucket_scheme"] for row in rows} == {"quintile"}


@pytest.mark.parametrize(
    ("scheme", "name"),
    [("decile", "quintile_00_20"), ("quintile", "decile_00_10"), ("quartile", "x")],
)
def test_score_selection_rejects_scheme_bin_mismatch(scheme: str, name: str) -> None:
    arguments = selection_arguments()
    arguments["confidence_bin"] = name
    with pytest.raises(ValueError, match="bucket scheme"):
        score_selection(**arguments, bucket_scheme=scheme)
```

- [ ] **Step 2: Run the scorer tests and confirm the new keyword fails**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/scene_uncertainty/test_decile_scoring.py -q
```

Expected: the new tests fail with an unexpected `bucket_scheme` argument or missing validation.

- [ ] **Step 3: Implement optional, validated provenance**

Add:

```python
from .confidence_deciles import DECILE_NAMES, QUINTILE_NAMES

BUCKET_NAMES = {
    "decile": frozenset(DECILE_NAMES),
    "quintile": frozenset(QUINTILE_NAMES),
}
```

Add `bucket_scheme: str | None = None` as the final keyword-only parameter of `score_selection`. Keep the current legacy `CONFIDENCE_BINS` validation when it is `None`. Otherwise validate the exact scheme/name pair:

```python
if bucket_scheme is None:
    if confidence_bin not in CONFIDENCE_BINS:
        raise ValueError(
            f"unknown confidence bin {confidence_bin!r}, expected {list(CONFIDENCE_BINS)}"
        )
else:
    if bucket_scheme not in BUCKET_NAMES:
        raise ValueError(
            f"unknown bucket scheme {bucket_scheme!r}, expected {sorted(BUCKET_NAMES)}"
        )
    if confidence_bin not in BUCKET_NAMES[bucket_scheme]:
        raise ValueError(
            f"confidence bin {confidence_bin!r} does not belong to bucket scheme "
            f"{bucket_scheme!r}"
        )
```

After constructing the existing provenance dictionary, add the field only for an explicit scheme:

```python
if bucket_scheme is not None:
    provenance["bucket_scheme"] = bucket_scheme
```

This conditional is the compatibility boundary: old rows remain exactly the same dictionaries.

- [ ] **Step 4: Run scorer and old integration tests**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/scene_uncertainty/test_decile_scoring.py \
  tests/scene_uncertainty/test_decile_integration.py -q
```

Expected: all pass and the existing command still writes its old schema.

- [ ] **Step 5: Commit**

```bash
git add src/scene_uncertainty/decile_scoring.py \
  tests/scene_uncertainty/test_decile_scoring.py
git commit -m "feat: record opt-in confidence bucket scheme"
```

## Task 3: Build both schemes from saved tuning artifacts

**Files:**

- Modify: `src/scene_uncertainty/decile_analysis.py:415-579`
- Create: `src/scene_uncertainty/corruption_analysis.py`
- Create: `tests/scene_uncertainty/test_corruption_analysis.py`
- Reuse: `tests/scene_uncertainty/decile_test_utils.py`

- [ ] **Step 1: Add failing loader and row-contract tests**

Using `write_decile_artifacts`, test all of these facts:

```python
def test_corruption_analysis_emits_both_schemes_and_only_requested_controls(tmp_path: Path) -> None:
    cache, results = write_decile_artifacts(tmp_path)
    inputs = load_corruption_inputs(cache, results)
    rows, diagnostics = analyze_corruption_sensitivity(inputs)

    assert inputs.run_metadata["artifact_type"] == "scene_corruption_sensitivity_inputs"
    assert len(rows) == 3060  # 34 selections x 15 signal/summary rows x 6 severities
    assert {row["bucket_scheme"] for row in rows} == {"decile", "quintile"}
    assert {row["severity"] for row in rows} == set(range(6))
    assert all(row["source_partition"] == "tuning" for row in rows)

    unfiltered = [row for row in rows if row["padding_mode"] == "unfiltered"]
    assert {row["confidence_bin"] for row in unfiltered} == {
        "decile_00_10", "quintile_00_20",
    }
    assert {row["membership_mode"] for row in unfiltered} == {"dynamic", "frozen"}
    assert len({tuple(row[field] for field in ROW_KEY) for row in rows}) == len(rows)
    assert diagnostics["images"]["1"]["union_padded_count"] == 2
```

Add a regression test that calls both old and new analyzers and, for filtered decile rows common to both, removes only `bucket_scheme` from the new row before comparing complete dictionaries. This protects the established numbers, selected query IDs, and overlap values.

Also assert that concatenating the quintile memberships and concatenating the decile memberships produces the same confidence-ranked valid-query sequence, and that persistence and confidence rows with the same selection keys carry exactly equal `selected_query_ids`.

Add static isolation assertions similar to the existing integration test: `corruption_analysis.py` may import the safe loader/scorer modules, but source text must not contain model loading, bank loading, `query_knn`, `torchvision`, or DETR symbols.

- [ ] **Step 2: Run the new test and confirm missing imports fail**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/scene_uncertainty/test_corruption_analysis.py -q
```

Expected: collection fails because `corruption_analysis` does not exist.

- [ ] **Step 3: Refactor the existing loader without changing its public behavior**

Rename the current `load_decile_inputs` body to this private helper:

```python
def _load_scene_query_inputs(
    cache_value: str | Path,
    results_value: str | Path,
    *,
    artifact_type: str,
) -> DecileInputs:
```

Inside its existing `run_metadata`, replace only:

```python
"artifact_type": artifact_type,
```

Then restore the old wrapper exactly:

```python
def load_decile_inputs(cache_value: str | Path, results_value: str | Path) -> DecileInputs:
    return _load_scene_query_inputs(
        cache_value,
        results_value,
        artifact_type=ANALYSIS_ARTIFACT_TYPE,
    )
```

Do not change validation order, exception classes, messages, partition filtering, streamed record shape, or the returned `DecileInputs` type.

- [ ] **Step 4: Implement the cache-only analyzer**

Create `corruption_analysis.py` with these public boundaries:

```python
from pathlib import Path

import torch

from .confidence_deciles import (
    DECILE_NAMES,
    QUINTILE_NAMES,
    bin_overlap,
    confidence_buckets,
    memberships_by_scheme_severity,
    union_query_ids,
)
from .decile_analysis import (
    DecileInputs,
    _load_scene_query_inputs,
    _padding_diagnostics,
)
from .decile_scoring import score_selection

CORRUPTION_INPUT_ARTIFACT_TYPE = "scene_corruption_sensitivity_inputs"
SCHEMES = {
    "decile": DECILE_NAMES,
    "quintile": QUINTILE_NAMES,
}


def load_corruption_inputs(cache_value: str | Path, results_value: str | Path) -> DecileInputs:
    return _load_scene_query_inputs(
        cache_value,
        results_value,
        artifact_type=CORRUPTION_INPUT_ARTIFACT_TYPE,
    )
```

Implement `analyze_corruption_sensitivity(inputs)` by iterating images, then schemes, then severities. For each scheme:

1. Build filtered dynamic/frozen memberships with `memberships_by_scheme_severity`.
2. Build clean filtered and clean unfiltered references separately.
3. Score every filtered bucket for both modes.
4. Score only the lowest bucket unfiltered for both modes.
5. Pass the same `indices`, `query_confidence`, aggregations, and `clean_overlap` into one `score_selection` call so persistence and confidence remain matched.
6. Pass `bucket_scheme=scheme` on every call.

Use this exact loop structure so no benchmark row slips into the new experiment:

```python
def analyze_corruption_sensitivity(inputs: DecileInputs) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    images: dict[str, dict] = {}
    for image_id, records in sorted(inputs.records_by_image.items()):
        padded = union_query_ids(
            [record["padded_query_ids"] for _, record in sorted(records.items())]
        )
        images[str(image_id)] = _padding_diagnostics(records, padded)
        query_count = int(records[0]["query_count"])
        all_indices = torch.arange(query_count, dtype=torch.long)

        for scheme, names in SCHEMES.items():
            noun = "confidence deciles" if scheme == "decile" else "confidence quintiles"
            memberships = memberships_by_scheme_severity(
                records, padded, names=names, noun=noun
            )
            clean_filtered = memberships[0]["dynamic"]
            clean_unfiltered = confidence_buckets(
                records[0]["query_confidence"],
                all_indices,
                names=names,
                noun=noun,
                label=f"image {image_id} severity 0, unfiltered",
            )
            lowest = names[0]

            for severity, record in sorted(records.items()):
                common = {
                    "image_id": image_id,
                    "severity": severity,
                    "source_partition": record["source_partition"],
                    "query_confidence": record["query_confidence"],
                    "query_scores_by_layer": inputs.distances[
                        (image_id, severity, record["source_partition"])
                    ],
                    "layer_score_scales": inputs.layer_score_scales,
                    "bucket_scheme": scheme,
                }
                for mode in ("dynamic", "frozen"):
                    bins = memberships[severity][mode]
                    overlap = bin_overlap(bins, clean_filtered)
                    for name in names:
                        rows.extend(score_selection(
                            **common,
                            membership_mode=mode,
                            confidence_bin=name,
                            padding_mode="filtered",
                            indices=bins[name],
                            clean_overlap=overlap[name],
                        ))

                    unfiltered = confidence_buckets(
                        record["query_confidence"],
                        all_indices,
                        names=names,
                        noun=noun,
                        label=f"image {image_id} severity {severity}, unfiltered",
                    )
                    unfiltered_overlap = bin_overlap(unfiltered, clean_unfiltered)
                    chosen = unfiltered if mode == "dynamic" else clean_unfiltered
                    rows.extend(score_selection(
                        **common,
                        membership_mode=mode,
                        confidence_bin=lowest,
                        padding_mode="unfiltered",
                        indices=chosen[lowest],
                        clean_overlap=(
                            unfiltered_overlap[lowest] if mode == "dynamic" else 1.0
                        ),
                    ))

    return rows, {"run": inputs.run_metadata, "images": images}
```

Important correction while implementing the loop: frozen unfiltered membership must use `clean_unfiltered`, as shown, while its overlap is `1.0`. Dynamic unfiltered uses the current severity's bins and measured overlap. Compute `unfiltered` once per severity and scheme, outside the mode loop, during refactoring to avoid duplicate work without changing output.

- [ ] **Step 5: Run focused tests and old loader regression tests**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/scene_uncertainty/test_corruption_analysis.py \
  tests/scene_uncertainty/test_decile_analysis.py \
  tests/scene_uncertainty/test_decile_integration.py -q
```

Expected: all pass; new fixture output has 3,060 unique rows and old output is unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/scene_uncertainty/decile_analysis.py \
  src/scene_uncertainty/corruption_analysis.py \
  tests/scene_uncertainty/test_corruption_analysis.py
git commit -m "feat: analyze deciles and quintiles from saved distances"
```

## Task 4: Implement per-image trends, locked orientation, and AUROC

**Files:**

- Create: `src/scene_uncertainty/corruption_metrics.py`
- Create: `tests/scene_uncertainty/test_corruption_metrics.py`

- [ ] **Step 1: Write exhaustive failing unit tests**

Cover increasing, decreasing, constant, incomplete, non-finite, tied, and reversed curves. Required assertions:

```python
def test_complete_trend_metrics_keeps_sign_and_absolute_strength() -> None:
    result = complete_trend_metrics(range(6), [0, 1, 2, 3, 4, 5])
    assert result == {
        "fully_measured": True,
        "signed_spearman": 1.0,
        "absolute_spearman": 1.0,
        "direction": "increasing",
    }


def test_complete_trend_metrics_marks_constant_curve_flat() -> None:
    result = complete_trend_metrics(range(6), [3.0] * 6)
    assert result == {
        "fully_measured": True,
        "signed_spearman": 0.0,
        "absolute_spearman": 0.0,
        "direction": "flat",
    }


def test_incomplete_curve_is_unmeasured_not_flat() -> None:
    result = complete_trend_metrics([0, 1, 2, 4, 5], [0, 1, 2, 4, 5])
    assert result["fully_measured"] is False
    assert result["signed_spearman"] is None
    assert result["absolute_spearman"] is None
    assert result["direction"] == "unmeasured"


@pytest.mark.parametrize(
    ("values", "expected"),
    [([0.8, 0.9], 1), ([-0.8, -0.2], -1), ([-0.8, 0.8], None)],
)
def test_choose_orientation_uses_one_group_median(values, expected) -> None:
    assert choose_orientation(values) == expected


def test_binary_auroc_is_tie_aware_and_uses_locked_orientation() -> None:
    assert binary_auroc([0, 1], [2, 3], orientation=1) == pytest.approx(1.0)
    assert binary_auroc([0, 1], [2, 3], orientation=-1) == pytest.approx(0.0)
    assert binary_auroc([1, 1], [1, 1], orientation=1) == pytest.approx(0.5)


def test_oriented_curve_metrics_report_deployment_checks() -> None:
    result = oriented_curve_metrics([5, 4, 3, 2, 1, 0], orientation=-1)
    assert result["adjacent_consistency"] == 1.0
    assert result["max_blur_above_clean"] is True
```

Also assert `choose_orientation` rejects no finite measured values, `binary_auroc` rejects empty/non-finite classes and orientations other than `-1`/`1`, and macro AUROC is the arithmetic mean of severities 1 through 5.

- [ ] **Step 2: Run and confirm module-not-found failure**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/scene_uncertainty/test_corruption_metrics.py -q
```

- [ ] **Step 3: Implement the pure metric functions**

Use this complete numerical contract:

```python
from collections.abc import Iterable, Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr

EXPECTED_SEVERITIES = tuple(range(6))


def complete_trend_metrics(severities: Iterable[int], scores: Iterable[float]) -> dict:
    severity = np.asarray(list(severities), dtype=int)
    values = np.asarray(list(scores), dtype=float)
    complete = (
        severity.shape == (6,)
        and values.shape == (6,)
        and tuple(severity.tolist()) == EXPECTED_SEVERITIES
        and bool(np.isfinite(values).all())
    )
    if not complete:
        return {
            "fully_measured": False,
            "signed_spearman": None,
            "absolute_spearman": None,
            "direction": "unmeasured",
        }
    if bool(np.all(values == values[0])):
        signed = 0.0
    else:
        signed = float(spearmanr(severity, values).statistic)
    direction = "increasing" if signed > 0 else "decreasing" if signed < 0 else "flat"
    return {
        "fully_measured": True,
        "signed_spearman": signed,
        "absolute_spearman": abs(signed),
        "direction": direction,
    }


def choose_orientation(signed_spearman: Iterable[float]) -> int | None:
    values = np.asarray(list(signed_spearman), dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    median = float(np.median(values))
    return 1 if median > 0 else -1 if median < 0 else None


def oriented_curve_metrics(scores: Sequence[float], orientation: int) -> dict:
    if orientation not in (-1, 1):
        raise ValueError("orientation must be -1 or 1")
    values = np.asarray(scores, dtype=float)
    if values.shape != (6,) or not bool(np.isfinite(values).all()):
        return {"adjacent_consistency": None, "max_blur_above_clean": None}
    oriented = values * orientation
    return {
        "adjacent_consistency": float(np.mean(np.diff(oriented) >= 0)),
        "max_blur_above_clean": bool(oriented[5] > oriented[0]),
    }


def binary_auroc(
    clean_scores: Sequence[float],
    corrupted_scores: Sequence[float],
    *,
    orientation: int,
) -> float:
    if orientation not in (-1, 1):
        raise ValueError("orientation must be -1 or 1")
    clean = np.asarray(clean_scores, dtype=float) * orientation
    corrupted = np.asarray(corrupted_scores, dtype=float) * orientation
    if not clean.size or not corrupted.size:
        raise ValueError("AUROC needs non-empty clean and corrupted groups")
    if not bool(np.isfinite(clean).all() and np.isfinite(corrupted).all()):
        raise ValueError("AUROC scores must be finite")
    ranks = rankdata(np.concatenate([clean, corrupted]), method="average")
    positive_count = corrupted.size
    positive_rank_sum = float(ranks[clean.size:].sum())
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2
    ) / (clean.size * positive_count)


def severity_aurocs(
    scores_by_severity: dict[int, Sequence[float]], orientation: int
) -> tuple[dict[int, float], float]:
    if set(scores_by_severity) != set(EXPECTED_SEVERITIES):
        raise ValueError("AUROC needs clean severity 0 and corrupted severities 1 through 5")
    clean = scores_by_severity[0]
    values = {
        severity: binary_auroc(clean, scores_by_severity[severity], orientation=orientation)
        for severity in EXPECTED_SEVERITIES[1:]
    }
    return values, float(np.mean(list(values.values())))
```

- [ ] **Step 4: Run the metric tests**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/scene_uncertainty/test_corruption_metrics.py -q
```

Expected: all pass without warnings for constant curves.

- [ ] **Step 5: Commit**

```bash
git add src/scene_uncertainty/corruption_metrics.py \
  tests/scene_uncertainty/test_corruption_metrics.py
git commit -m "feat: add corruption trend and ranking metrics"
```

## Task 5: Summarize candidates and rank only deployable ones

**Files:**

- Create: `src/scene_uncertainty/corruption_reporting.py`
- Create: `tests/scene_uncertainty/test_corruption_reporting.py`

- [ ] **Step 1: Add failing tests for aggregation semantics and gates**

Build small in-memory row sets and assert:

1. `median_absolute_spearman` is the median of each image's absolute coefficient, not `abs(median_signed_spearman)`.
2. Persistence and confidence candidates choose their orientation independently.
3. One orientation is reused for all five AUROCs.
4. Flat complete curves count in the dominant-direction denominator.
5. Missing or non-finite curves count as unmeasured and never as flat.
6. The ranking table contains only dynamic, filtered, 250-image, six-severity, persistence-`layer_2` candidates.
7. Sorting is macro AUROC descending, median absolute Spearman descending, dominant-direction fraction descending, oriented adjacent consistency descending, then candidate fields ascending.
8. Per-severity raw scene summaries use population variance (`ddof=0`).
9. Duplicate `ROW_KEY` rows are rejected before any grouping.

The key semantic test must use opposite-signed strong images:

```python
def test_group_strength_is_median_of_per_image_absolute_spearman() -> None:
    rows = candidate_rows(
        curves={1: [0, 1, 2, 3, 4, 5], 2: [5, 4, 3, 2, 1, 0]}
    )
    _, metrics, ranking = summarize_candidates(rows, expected_image_count=2)
    metric = metrics[0]

    assert metric["median_signed_spearman"] == pytest.approx(0.0)
    assert metric["median_absolute_spearman"] == pytest.approx(1.0)
    assert metric["orientation"] is None
    assert metric["deployable"] is False
    assert ranking == []
```

- [ ] **Step 2: Run the tests and confirm the reporter is missing**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/scene_uncertainty/test_corruption_reporting.py -q
```

- [ ] **Step 3: Implement validation and per-image curves**

Create constants matching `ROW_KEY` and `CANDIDATE_KEY` at the top of `corruption_reporting.py`. Implement:

```python
def validate_rows(rows: list[dict]) -> None:
    seen: set[tuple] = set()
    for index, row in enumerate(rows):
        missing = [field for field in ROW_KEY if field not in row]
        if missing:
            raise ValueError(f"score row {index} is missing fields {missing}")
        key = tuple(row[field] for field in ROW_KEY)
        if key in seen:
            raise ValueError(f"duplicate corruption score row key: {key}")
        seen.add(key)
```

Group rows by candidate then image, sort by severity, and call `complete_trend_metrics`. Build one `per_scene.csv` row per candidate/image/severity, as required by the approved output contract. Repeat the image-level trend fields on its six severity rows so every row remains independently readable:

```python
{
    **{field: score_row[field] for field in ROW_KEY},
    "source_partition": score_row["source_partition"],
    "score": score_row["score"],
    "selected_count": score_row["selected_count"],
    "clean_overlap": score_row["clean_overlap"],
    "fully_measured": trend["fully_measured"],
    "signed_spearman": trend["signed_spearman"],
    "absolute_spearman": trend["absolute_spearman"],
    "direction": trend["direction"],
}
```

Do not serialize `selected_query_ids` into a CSV list cell. Validate matching selected IDs before this transformation, then retain selection size and overlap as ordinary numeric columns. The resulting real `per_scene.csv` has the same 765,000 rows as the raw score table, not 127,500 six-severity-wide rows.

- [ ] **Step 4: Implement group metrics, raw severity statistics, and ranking**

For each candidate:

```python
measured = [row for row in scene_rows if row["fully_measured"]]
signed = np.asarray([row["signed_spearman"] for row in measured], dtype=float)
orientation = choose_orientation(signed)
positive_count = int(np.sum(signed > 0))
negative_count = int(np.sum(signed < 0))
flat_count = int(np.sum(signed == 0))
measured_count = len(measured)
dominant = (
    max(positive_count, negative_count) / measured_count if measured_count else None
)
```

For an orientable candidate, compute each measured image's oriented adjacent consistency and max-blur check, then compute the five AUROCs by collecting raw scores across all finite images at severity 0 and the target severity. Never drop an image from only one side silently: record `clean_count` and `corrupted_count` for each severity and include them in `coverage_by_severity`.

Also aggregate raw scores by severity with:

```python
severity_statistics[str(severity)] = {
    "count": int(values.size),
    "mean": float(np.mean(values)),
    "variance": float(np.var(values, ddof=0)),
    "median": float(np.median(values)),
    "q25": float(np.quantile(values, 0.25, method="linear")),
    "q75": float(np.quantile(values, 0.75, method="linear")),
}
```

Mark `deployable` only when every gate is true:

```python
FULL_TUNING_IMAGE_COUNT = 250

deployable = (
    expected_image_count == FULL_TUNING_IMAGE_COUNT
    and candidate["membership_mode"] == "dynamic"
    and candidate["padding_mode"] == "filtered"
    and candidate["signal"] == "persistence"
    and candidate["score_scope"] == "layer_2"
    and measured_count == FULL_TUNING_IMAGE_COUNT
    and all(
        len(score_groups[severity]) == FULL_TUNING_IMAGE_COUNT
        for severity in range(6)
    )
    and orientation is not None
)
```

Return all candidate metrics in deterministic candidate-key order and a second ranking list containing only deployable rows, sorted by:

```python
key=lambda row: (
    -row["macro_auroc"],
    -row["median_absolute_spearman"],
    -row["dominant_direction_fraction"],
    -row["oriented_adjacent_consistency"],
    *(row[field] for field in CANDIDATE_KEY),
)
```

Matched confidence controls remain in `candidate_metrics.csv` and `summary.json`; they are excluded only from the persistence ranking gate.

For every candidate, also flatten these supporting diagnostics into `candidate_metrics.csv` and retain their per-severity form in `summary.json`:

- measured, missing, positive, negative, and flat image counts and fractions;
- minimum, median, and maximum selected-query count by severity;
- median dynamic overlap with the clean membership by severity;
- finite score coverage by severity;
- oriented adjacent consistency and strongest-blur-versus-clean rate;
- the matching filtered-versus-unfiltered lowest-bucket score difference for padding sensitivity.

Build the padding-sensitivity comparison only between candidates whose signal, scheme, lowest bucket, membership mode, aggregation, and scope match and whose only differing field is `padding_mode`. Never let unfiltered candidates enter the deployment ranking.

- [ ] **Step 5: Run reporter and metric tests**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/scene_uncertainty/test_corruption_metrics.py \
  tests/scene_uncertainty/test_corruption_reporting.py -q
```

- [ ] **Step 6: Commit**

```bash
git add src/scene_uncertainty/corruption_reporting.py \
  tests/scene_uncertainty/test_corruption_reporting.py
git commit -m "feat: summarize deployable corruption candidates"
```

## Task 6: Plot raw scene distance with comparable axes

**Files:**

- Create: `src/scene_uncertainty/corruption_plots.py`
- Create: `tests/scene_uncertainty/test_corruption_plots.py`

- [ ] **Step 1: Add failing plot tests**

Use five/quintile and ten/decile synthetic candidates with deliberately different ranges. Assert:

- Exactly four PNGs are created with the required names.
- Decile figures have ten axes in a `2 x 5` layout; quintile figures have five axes in a `1 x 5` layout.
- Both persistence figures share the same numeric y-limits.
- Both confidence figures share a separate common y-limit.
- Every panel has x ticks `0,1,2,3,4,5`, one median line, one IQR collection, and a direction label.
- The values plotted are raw, un-oriented q90 scores; a decreasing candidate remains visually decreasing.

Required filenames:

```python
PLOT_FILENAMES = {
    ("persistence", "decile"): "persistence_actual_distance_deciles.png",
    ("persistence", "quintile"): "persistence_actual_distance_quintiles.png",
    ("confidence", "decile"): "confidence_actual_distance_deciles.png",
    ("confidence", "quintile"): "confidence_actual_distance_quintiles.png",
}
```

- [ ] **Step 2: Run and confirm the plot module is absent**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/scene_uncertainty/test_corruption_plots.py -q
```

- [ ] **Step 3: Implement shared limits and small multiples**

Select only dynamic, filtered, q90 candidates, and persistence `layer_2` for persistence plots. Confidence's `score_scope` is `confidence`. Compute limits separately per signal, jointly across its decile and quintile q25/q75 values:

```python
def shared_limits(candidates: list[dict], signal: str) -> tuple[float, float]:
    chosen = [row for row in candidates if row["signal"] == signal]
    lower = min(
        stats["q25"]
        for row in chosen
        for stats in row["severity_statistics"].values()
    )
    upper = max(
        stats["q75"]
        for row in chosen
        for stats in row["severity_statistics"].values()
    )
    span = upper - lower
    margin = 0.05 * span if span > 0 else max(0.05 * abs(lower), 1e-6)
    return lower - margin, upper + margin
```

For each subplot:

```python
severity = np.arange(6)
median = [candidate["severity_statistics"][str(s)]["median"] for s in severity]
q25 = [candidate["severity_statistics"][str(s)]["q25"] for s in severity]
q75 = [candidate["severity_statistics"][str(s)]["q75"] for s in severity]
axis.plot(severity, median, marker="o", linewidth=2)
axis.fill_between(severity, q25, q75, alpha=0.22)
axis.set_xlim(0, 5)
axis.set_xticks(severity)
axis.set_ylim(*signal_limits)
```

Use the un-oriented `median`, `q25`, and `q75`; orientation belongs to AUROC only. Title each panel with the bucket name and the group median-signed-trend label:

- `increasing` if median signed Spearman is positive,
- `decreasing` if median signed Spearman is negative,
- `flat` if all fully measured images are flat,
- `mixed` for any other zero-median group.

Put “Blur severity” on the x-axis and “Raw q90 persistence distance” or “Raw q90 confidence uncertainty (1 - confidence)” on the y-axis. Call `figure.tight_layout()` and save at 160 DPI.

- [ ] **Step 4: Run plot tests**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/scene_uncertainty/test_corruption_plots.py -q
```

- [ ] **Step 5: Commit**

```bash
git add src/scene_uncertainty/corruption_plots.py \
  tests/scene_uncertainty/test_corruption_plots.py
git commit -m "feat: plot comparable raw corruption signals"
```

## Task 7: Write an atomic result bundle and plain-language report

**Files:**

- Modify: `src/scene_uncertainty/corruption_reporting.py`
- Modify: `tests/scene_uncertainty/test_corruption_reporting.py`

- [ ] **Step 1: Add failing bundle and report tests**

Assert that `write_corruption_report(...)`:

- refuses an existing output directory;
- leaves no final or staging directory when a forced write fails;
- creates exactly these eight files on success:

```python
EXPECTED_FILES = {
    "per_scene.csv",
    "candidate_metrics.csv",
    "summary.json",
    "persistence_actual_distance_deciles.png",
    "persistence_actual_distance_quintiles.png",
    "confidence_actual_distance_deciles.png",
    "confidence_actual_distance_quintiles.png",
    "easy-report.md",
}
```

- writes deterministic CSV column order and JSON with `sort_keys=True`;
- stores the exact persistence and confidence shared y-axis limits in `summary.json`;
- stores input provenance, fixed configuration, validation counts, padding diagnostics, all candidate metrics, and the eligible ranking in `summary.json`;
- leads `easy-report.md` with the best deployable layer-2 result and explains AUROC as ordering, not probability;
- reports all five per-severity AUROCs plus macro AUROC;
- explains absolute Spearman together with positive/negative/flat counts;
- compares quintile with decile and persistence with confidence;
- says calibration is absent and held-out data was not used;
- contains no LaTeX or terminal-unfriendly math notation.
- handles a valid synthetic or diagnostic run with fewer than 250 images by writing an empty deployment ranking and saying no candidate passed the full tuning-coverage gate, rather than crashing or relaxing the gate.

- [ ] **Step 2: Run and observe missing writer failures**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/scene_uncertainty/test_corruption_reporting.py -q
```

- [ ] **Step 3: Implement atomic directory publication**

Use a sibling staging directory and publish only after every CSV, JSON, PNG, and Markdown file succeeds:

```python
import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd


def write_corruption_report(
    output_value: str | Path,
    *,
    score_rows: list[dict],
    diagnostics: dict,
) -> dict:
    output = Path(output_value)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        per_scene, candidates, ranking = summarize_candidates(
            score_rows,
            expected_image_count=int(diagnostics["run"]["image_count"]),
        )
        pd.DataFrame(per_scene).to_csv(staging / "per_scene.csv", index=False)
        pd.DataFrame(candidates).to_csv(staging / "candidate_metrics.csv", index=False)
        axis_limits = write_corruption_plots(staging, candidates)
        summary = build_summary(
            diagnostics,
            candidates,
            ranking,
            axis_limits=axis_limits,
        )
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / "easy-report.md").write_text(
            render_easy_report(summary), encoding="utf-8"
        )
        actual = {path.name for path in staging.iterdir()}
        if actual != EXPECTED_FILES:
            raise RuntimeError(
                f"report bundle contains {sorted(actual)}, expected {sorted(EXPECTED_FILES)}"
            )
        os.replace(staging, output)
        return summary
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
```

Keep nested dictionaries out of CSV cells. Flatten the five AUROC columns to `auroc_severity_1` through `auroc_severity_5`, raw severity statistics to named columns, and coverage to named columns. Preserve the nested version in `summary.json` for readability.

Make `write_corruption_plots` return `{"persistence": [lower, upper], "confidence": [lower, upper]}` after applying those exact values to every corresponding axis. `build_summary` must record that object verbatim, alongside `diagnostics["run"]`, the fixed schemes/modes/summaries/scopes, raw/accepted/duplicate row counts, image padding diagnostics, every candidate, and the ordered deployable ranking.

- [ ] **Step 4: Render the easy report in plain language**

Use short sections in this order:

1. “What should we deploy?” — best candidate's selection, query count range, macro and five AUROCs.
2. “Does it react steadily to blur?” — median absolute Spearman, signed median, positive/negative/flat counts, adjacent consistency, maximum-blur comparison.
3. “Did 20% buckets help?” — best quintile versus best decile under the same ranking rule.
4. “Did persistence beat model confidence?” — matched control with the same scheme, bucket, mode, padding, aggregation, and selected IDs.
5. “What the files contain” — explain the four actual-distance plots and their shared axes.
6. “What this does not prove” — no calibrated corruption probability and no held-out evaluation.

Spell out the interpretation in child-friendly language, for example:

```text
An AUROC of 0.80 means that when we randomly pick one clean image and one blurred image,
the score puts the blurred image higher about 80 times out of 100. It does not mean the
image has an 80% chance of being corrupted.
```

- [ ] **Step 5: Run reporter tests and inspect generated fixture images**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/scene_uncertainty/test_corruption_reporting.py \
  tests/scene_uncertainty/test_corruption_plots.py -q
```

Open the four test PNGs and confirm labels are not clipped and the two persistence plots visibly use one y-scale while the two confidence plots use their own one y-scale.

- [ ] **Step 6: Commit**

```bash
git add src/scene_uncertainty/corruption_reporting.py \
  tests/scene_uncertainty/test_corruption_reporting.py
git commit -m "feat: write corruption sensitivity report bundle"
```

## Task 8: Add the cache-only CLI and end-to-end regression coverage

**Files:**

- Modify: `src/scene_uncertainty/cli.py:1-270`
- Modify: `src/scene_uncertainty/pipeline.py:726-785`
- Modify: `tests/scene_uncertainty/test_cli.py`
- Create: `tests/scene_uncertainty/test_corruption_integration.py`
- Modify: `README.md:253-275`

- [ ] **Step 1: Add failing parser tests**

Add `analyze-corruption-sensitivity` to the expected command lists. Test its exact arguments:

```python
def test_analyze_corruption_sensitivity_accepts_only_saved_artifacts_and_output() -> None:
    args = build_parser().parse_args([
        "analyze-corruption-sensitivity",
        "--cache", "cache",
        "--results", "results/raw_k5.csv",
        "--output", "reports/corruption",
    ])

    assert vars(args) == {
        "command": "analyze-corruption-sensitivity",
        "cache": Path("cache"),
        "results": Path("results/raw_k5.csv"),
        "output": Path("reports/corruption"),
    }
```

Also assert `--partition`, bank/model/device/worker options, and an omitted required argument are rejected by argparse.

- [ ] **Step 2: Add a failing synthetic end-to-end test**

Mirror `test_decile_integration.py` with `write_decile_artifacts`, run `main([...])`, and assert:

- the completion message names the new command and output;
- all eight output files exist;
- `summary.json` says tuning, six severities, and one image;
- both schemes appear in `candidate_metrics.csv`;
- the old command run against the same inputs still has its old exact file set and schema;
- monkeypatched model, comparison-bank, and kNN entry points raise if called, proving the new path is cache-only;
- rerunning into an existing output fails without overwriting it.

- [ ] **Step 3: Run CLI/integration tests and confirm expected failures**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/scene_uncertainty/test_cli.py \
  tests/scene_uncertainty/test_corruption_integration.py -q
```

- [ ] **Step 4: Wire the command**

In `cli.py`, add a parser with only the three paths:

```python
corruption = subparsers.add_parser(
    "analyze-corruption-sensitivity",
    help="compare decile/quintile persistence and confidence corruption signals",
)
corruption.add_argument("--cache", required=True, type=Path)
corruption.add_argument("--results", required=True, type=Path)
corruption.add_argument("--output", required=True, type=Path)
```

In `pipeline.py`, add imports and:

```python
def command_analyze_corruption_sensitivity(args) -> None:
    inputs = load_corruption_inputs(args.cache, args.results)
    rows, diagnostics = analyze_corruption_sensitivity(inputs)
    write_corruption_report(
        args.output,
        score_rows=rows,
        diagnostics=diagnostics,
    )
    _report(
        f"analyze-corruption-sensitivity: summarized {len(rows)} score rows "
        f"into {args.output}"
    )
```

Register it in `COMMANDS`. Do not import any model/cache-building module into the new analysis/reporting modules.

- [ ] **Step 5: Document the exact operator command**

Add beside the decile experiment in `README.md`:

```bash
SCENE_OUT=output/scene_uncertainty
/home/yuchen/miniconda3/envs/UE/bin/python tools/scene_uncertainty.py \
  analyze-corruption-sensitivity \
  --cache "$SCENE_OUT/blur_cache" \
  --results "$SCENE_OUT/results/raw_k5.csv" \
  --output "$SCENE_OUT/reports/corruption_sensitivity_raw_k5"
```

Explain that this reuses saved tuning query distances, creates decile and quintile actual-distance plots, treats confidence as the matched control, and reports ranking AUROC rather than calibrated probability.

- [ ] **Step 6: Run the integration and old-command regression tests**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/scene_uncertainty/test_cli.py \
  tests/scene_uncertainty/test_corruption_integration.py \
  tests/scene_uncertainty/test_decile_integration.py -q
```

- [ ] **Step 7: Commit**

```bash
git add src/scene_uncertainty/cli.py \
  src/scene_uncertainty/pipeline.py \
  tests/scene_uncertainty/test_cli.py \
  tests/scene_uncertainty/test_corruption_integration.py \
  README.md
git commit -m "feat: add corruption sensitivity CLI"
```

## Task 9: Verify the full suite and run the six-level tuning experiment

**Files:**

- Verify: all modified source and tests
- Generate, without committing large artifacts: `output/scene_uncertainty/reports/corruption_sensitivity_raw_k5/`

- [ ] **Step 1: Run formatting/static checks already used by the repository**

First inspect the repository's configured commands in `pyproject.toml`, `setup.cfg`, `tox.ini`, or CI. Run every configured formatter/linter/type checker that applies. Do not introduce a new tool solely for this change.

- [ ] **Step 2: Run the complete test suite**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q
```

Expected: the prior 774 tests plus the new corruption-sensitivity tests pass; record the actual final count rather than assuming it remains 774 plus a fixed number.

- [ ] **Step 3: Audit forbidden dependencies and compatibility**

```bash
rg -n "load_model|build_model|torchvision|query_knn|load_bank|comparison_bank" \
  src/scene_uncertainty/corruption_analysis.py \
  src/scene_uncertainty/corruption_metrics.py \
  src/scene_uncertainty/corruption_plots.py \
  src/scene_uncertainty/corruption_reporting.py
```

Expected: no matches.

Run the old decile integration test one more time and compare its fixture schema with the baseline asserted in the test. This is the release gate for “old command unchanged.”

- [ ] **Step 4: Run the real tuning experiment over the existing six blur levels**

Only run this after confirming the two source artifacts exist:

```bash
test -d output/scene_uncertainty/blur_cache
test -f output/scene_uncertainty/results/raw_k5.csv
```

Then run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python tools/scene_uncertainty.py \
  analyze-corruption-sensitivity \
  --cache output/scene_uncertainty/blur_cache \
  --results output/scene_uncertainty/results/raw_k5.csv \
  --output output/scene_uncertainty/reports/corruption_sensitivity_raw_k5
```

If those artifacts live in the experiment worktree on the GPU host instead, use these already-established absolute paths without copying or rebuilding them:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python tools/scene_uncertainty.py \
  analyze-corruption-sensitivity \
  --cache /home/alienware2/YuchenZ/UE/philip_sa-deciles/output/scene_uncertainty/blur_cache \
  --results /home/alienware2/YuchenZ/UE/philip_sa-deciles/output/scene_uncertainty/results/raw_k5.csv \
  --output /home/alienware2/YuchenZ/UE/philip_sa-deciles/output/scene_uncertainty/reports/corruption_sensitivity_raw_k5
```

Do not run feature extraction, bank construction, kNN, or held-out evaluation.

- [ ] **Step 5: Independently audit the generated tables**

Run a short read-only verification script that checks:

- row count is `250 images x 34 selections x 15 signal/scope/aggregation rows x 6 severities = 765,000`;
- every candidate has exactly 250 images and each measured image has severities 0 through 5;
- `candidate_metrics.csv` has no duplicate candidate key;
- deployable ranking rows are dynamic, filtered, persistence, layer 2, fully covered, and orientable;
- each macro AUROC equals the mean of its five per-severity AUROCs;
- each absolute Spearman is non-negative and equals the absolute value of that same image's signed Spearman;
- population variance values recomputed from raw rows agree with the saved values;
- persistence figures share one y-range and confidence figures share their own y-range.

If the actual cache manifest does not contain 250 tuning images, compute the expected raw-row count as `image_count x 3060`, report the manifest count, and do not force the 250-image deployability gate to pass.

- [ ] **Step 6: Read the result as a deployment experiment**

Use `easy-report.md`, `candidate_metrics.csv`, and the four plots to answer, in order:

1. Which layer-2 persistence candidate has the highest macro AUROC?
2. Is its direction consistent enough to use one locked orientation?
3. At which blur severity does it first separate clean images well?
4. Did a quintile beat the best decile under the same gate and ranking rule?
5. Did persistence beat its exactly matched `1 - confidence` control?
6. Do the raw plots reveal smooth change, saturation, reversal, or unusually wide image-to-image spread?

Do not select a candidate by absolute Spearman alone. It is a sensitivity diagnostic; macro AUROC is the primary deployment ranking metric.

- [ ] **Step 7: Final verification and commit any test/document fixes**

If the real run exposes a deterministic reporting bug, add a failing synthetic regression test before fixing it, rerun the focused test, rerun the full suite, then commit only source/tests/docs. Leave the generated 250-image CSVs and PNGs uncommitted unless the repository's existing artifact policy explicitly tracks them.

```bash
git status --short
git diff --check
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q
```

Expected: no whitespace errors, every test passes, and `git status` contains no unintended generated artifacts.

## Final acceptance checklist

- [ ] `analyze-confidence-deciles` behavior, schema, and tests are unchanged.
- [ ] `analyze-corruption-sensitivity` accepts exactly cache, results, and output paths.
- [ ] The new path is cache-only and tuning-only.
- [ ] Deciles and quintiles use stable confidence sorting with query-ID tie breaks.
- [ ] Persistence and confidence use exactly matched selected query IDs and aggregations.
- [ ] Signed and absolute per-image Spearman are both retained; flat and unmeasured are distinct.
- [ ] Each candidate gets one tuning-derived orientation, never a per-severity or per-image orientation.
- [ ] Five clean-versus-blur AUROCs and their macro mean are reported as ranking metrics.
- [ ] Ranking gates enforce dynamic, filtered, full coverage, persistence layer 2, and orientability.
- [ ] Four raw q90 median/IQR plots exist and use comparable signal-specific y-limits.
- [ ] The report bundle is atomically published and contains exactly eight files.
- [ ] The easy report uses plain language and explicitly says the score is not a calibrated probability.
- [ ] The real run uses six blur levels and does not touch held-out data.
