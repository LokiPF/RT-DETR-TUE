# Within-Image Corruption Contrast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reporting-only `analyze-within-image-contrast` command that scores corruption from one image at one timestamp by combining two confidence-percentile ranges measured on that same image, and that reports two separate verdicts — an anchored hypothesis and a differential hypothesis — each against its own controls.

**Architecture:** The command reads a *completed* `analyze-corruption-sensitivity` output directory and nothing else. It never touches DETR, the comparison bank, kNN, or the feature cache — every number it needs is already in `per_scene.csv`. Four predeclared arms (bucket pair at one persistence scope) cross four score methods cross three matched scene summaries give 45 named candidates — 45 and not 48 because the `combined` arm is a signed z-score and has no symmetric relative gap. Each candidate is oriented once from its median signed tuning Spearman, evaluated with per-severity clean-versus-corrupted AUROC, and compared against three controls: the raw responsive range, the raw reference range, and a confidence-only twin. The clean-predicted residual is cross-fitted over five deterministic image folds so no image helps construct its own expected clean value.

**Tech Stack:** Python 3.10+, NumPy, SciPy (`rankdata` only), pandas, Matplotlib, pytest, the existing `scene_uncertainty` CLI and artifact formats.

**Spec:** [`docs/superpowers/specs/2026-08-21-within-image-corruption-contrast-design.md`](../specs/2026-08-21-within-image-corruption-contrast-design.md)

---

## Global Constraints

Every task's requirements implicitly include this section. Values are copied verbatim from the spec.

- **Cache-only architectural boundary.** The new modules must never import or run DETR, persistence extraction, the comparison bank, query kNN, or feature-cache iteration. Task 9 proves this with patched entry points that raise if called.
- **Tuning partition only.** The source must be `tuning`. Do not add a `--partition` flag. Do not read the held-out partition. Refuse a source containing held-out rows rather than filtering them silently.
- **Exactly four predeclared arms.** Do not search over other bucket pairs, scopes, or mixed scene summaries. The arm table is closed before the command runs.
- **Matched summaries only.** Both sides of a contrast use the same scene summary (`mean`, `q90`, or `top20_mean`) and the same persistence scope. Never subtract a mean from a q90. Never mix scopes inside one arm.
- **One locked orientation per candidate**, chosen from that candidate's median signed tuning Spearman: positive median → use unchanged; negative median → multiply by negative one; zero median → unorientable. Never choose orientation per image, per severity, per fold, or by AUROC comparison.
- **Cross-fitting is mandatory for the residual.** Five deterministic folds; sorted image position `p` → fold `p % 5`; all six severities of an image stay in its fold. Reported tuning residual performance always uses cross-fitted scores, never the final all-clean line.
- **No probability claims.** AUROC measures ranking. The easy report must contain no calibrated-probability language and must state that no held-out images were used.
- **`declared_before_data` travels with every arm** into `candidate_metrics.csv`, `summary.json`, and the easy report. The two differential arms are `False`.
- **Deterministic bootstrap.** Seed `20260821`, 2,000 samples, image IDs resampled identically for both methods in a comparison.
- **No scikit-learn.** Use SciPy's tie-aware `rankdata` through the existing `corruption_metrics.binary_auroc`.
- **Atomic publication.** Write into a staging directory beside the destination and `os.replace` it into place. Refuse an existing `--output`. Leave nothing behind on failure.
- **Nine output files exactly.** Adding a tenth is a spec change, not an implementation decision.

## Ground rules

- Work in a dedicated worktree created with `superpowers:using-git-worktrees` before changing production code. Branch from `dev_tue` at `928faf3` or later, **not** from `origin/main` — `origin/main` has neither the corruption-sensitivity work this builds on nor the spec.
- Local Python for every command: `/home/yuchen/miniconda3/envs/UE/bin/python`. This plan writes it as `$UE_PY`.
- `tools/scene_uncertainty.py` has a pre-existing `ModuleNotFoundError: No module named 'src'` when run directly. Prefix with `PYTHONPATH=$PWD`. **Do not fix it** — it is outside this plan's scope and affects the already-documented commands equally.
- Follow red-green-refactor for every task: add one focused failing test, run it and read the failure text, make the smallest production change, rerun.
- **Mutation testing is required from Task 2 onward.** Before committing a task, break the production code in a plausible way (flip a comparison, drop a term, return the input unchanged, remove a filter) and confirm a test fails with a message that names the real problem. A test that still passes is not a test. The previous plan on this codebase shipped six assertions that looked binding and were not — a test that sorted away the ordering it compared, a `match=` string satisfied by two different error branches, a column checked against a constant derived from itself, a parametrization with zero discriminating power, a fixture where `mean == median`, and a constraint guarded only by a missing dictionary key.
- Do not alter the behaviour or output of `analyze-corruption-sensitivity` or `analyze-confidence-deciles`. This command is a pure consumer of the first one's finished bundle.
- Every module in this plan gets docstrings in the surrounding style: say *why* a choice was made and what the rejected alternative would have broken, not what the code does.

## Test commands

```bash
UE_PY=/home/yuchen/miniconda3/envs/UE/bin/python

# focused
$UE_PY -m pytest tests/scene_uncertainty/test_contrast_scores.py -v

# whole suite (local GPU may be occupied; see Task 10)
$UE_PY -m pytest tests/scene_uncertainty -q
```

## The four arms

An **arm** is one bucket pair at one persistence scope. This table is the closed set.

| Arm name | Family | Scheme | Scope | Reference range | Responsive range | `declared_before_data` |
|---|---|---|---|---|---|---|
| `decile_00_10__50_60` | anchored | decile | `layer_2` | `decile_00_10` | `decile_50_60` | `True` |
| `quintile_00_20__40_60` | anchored | quintile | `layer_2` | `quintile_00_20` | `quintile_40_60` | `True` |
| `decile_90_100__50_60` | differential | decile | `layer_2` | `decile_90_100` | `decile_50_60` | `False` |
| `decile_90_100__50_60__combined` | differential | decile | `combined` | `decile_90_100` | `decile_50_60` | `False` |

The four score methods are `raw_responsive`, `raw_gap`, `relative_gap`, `clean_residual`.
The three matched summaries are `mean`, `q90`, `top20_mean`.

**Scope decides sign, and the `combined` arm is signed.** A `layer_N` score is a raw mean-kNN
distance and is non-negative. The `combined` score is a robust z-score — `score_selection`
builds it as the mean over layers of `(score - center) / scale` with `center` the clean median —
so it goes negative whenever the selected queries sit below that median. The completed run
confirms it: 548 negative `combined` per-severity statistics against none at `layer_2`, and both
of the combined arm's own bins go negative under `mean`. Two consequences bind every task:
validation is scope-aware (non-negative required for `layer_N` persistence and for confidence,
finite-only for `combined`), and **`relative_gap` is unavailable at the `combined` scope**,
because its scale invariance and its ±2 bounds both rest on non-negative inputs.

3 layer_2 arms × 3 summaries × 4 methods, plus the combined arm × 3 summaries × 3 methods =
**45 persistence candidates**, every one named before the command runs.

Confidence-only twins are keyed by *bucket pair*, not arm, because confidence has a single scope: the two differential arms share one twin. 3 bucket pairs × 3 summaries × 4 methods = **36 twins** — unaffected by the sign rule, since confidence is non-negative. Total candidates: **81**.

Robust clean lines are fitted per (arm, summary, signal): 4 × 3 = 12 persistence plus 3 × 3 = 9 confidence = **21 final fits**.

## What the source bundle must contain

The source is a finished `analyze-corruption-sensitivity` output directory. `per_scene.csv` has this header, and the new command reads a strict subset of its 765,000 rows:

```text
image_id,severity,signal,bucket_scheme,confidence_bin,membership_mode,padding_mode,
aggregation,score_scope,source_partition,score,selected_count,clean_overlap,
fully_measured,signed_spearman,absolute_spearman,direction
```

Required series, all at `membership_mode="dynamic"` and `padding_mode="filtered"`, all three aggregations. **Rows outside those two modes are discarded, not refused** — the producer writes a `frozen` twin for every bin and an `unfiltered` control for the first bin of each scheme, so a loader that refused them would refuse every real bundle:

| Signal | `confidence_bin` | `score_scope` |
|---|---|---|
| `persistence` | `decile_00_10` | `layer_2` |
| `persistence` | `decile_50_60` | `layer_2`, `combined` |
| `persistence` | `decile_90_100` | `layer_2`, `combined` |
| `persistence` | `quintile_00_20` | `layer_2` |
| `persistence` | `quintile_40_60` | `layer_2` |
| `confidence` | all five bins above | `confidence` |

That is 12 `(signal, bin, scope)` series × 3 aggregations × 250 images × 6 severities = **54,000 retained rows**. Every one of these was verified to exist in the completed run at `~/YuchenZ/UE/philip_sa-scene-unc/output/scene_uncertainty/pilot/reports/corruption_sensitivity_raw_k5/` on alienware2 before this plan was written.

## File structure

| File | Responsibility |
|---|---|
| `src/scene_uncertainty/contrast_inputs.py` | The arm table, and loading/validating a finished source bundle into a scene-score lookup. |
| `src/scene_uncertainty/contrast_scores.py` | Pure maths: four score methods, deterministic folds, robust clean line. No I/O, no arm knowledge. |
| `src/scene_uncertainty/contrast_diagnostics.py` | Anchor diagnostics: within-image drift, between-image spread, clean cross-bucket relationship. |
| `src/scene_uncertainty/contrast_analysis.py` | Per-scene contrast rows with cross-fitted residuals; per-candidate trends, orientation, AUROC, ranking. |
| `src/scene_uncertainty/contrast_controls.py` | Raw reference control, confidence-only twin, paired image bootstrap. |
| `src/scene_uncertainty/contrast_plots.py` | The four figures and the axis limits they applied. |
| `src/scene_uncertainty/contrast_reporting.py` | `summary.json`, the three CSVs, the easy report, atomic bundle publication. |
| `src/scene_uncertainty/cli.py` (modify) | `analyze-within-image-contrast` subcommand. |
| `src/scene_uncertainty/pipeline.py` (modify) | `command_analyze_within_image_contrast` handler. |
| `README.md` (modify) | Documented invocation using the existing `$UE_PY`/`$OUT` convention. |

Tests mirror the modules: `tests/scene_uncertainty/test_contrast_<name>.py`, plus a shared fixture builder at `tests/scene_uncertainty/contrast_test_utils.py` and an end-to-end `test_contrast_integration.py`.

---

## Task 1: Load and validate a finished corruption-sensitivity bundle

> **Amended after review, 2026-08-22.** The code block below was written against two false
> premises and both were confirmed false against the completed run. Apply these corrections;
> everything else in the block stands.
>
> 1. **Rows outside `dynamic`/`filtered` are discarded, not refused.** The producer writes a
>    `frozen` twin for every bin and an `unfiltered` control for the first bin of each scheme,
>    and those rows carry `(signal, confidence_bin, score_scope)` triples that are in
>    `REQUIRED_SERIES`. The `raise` below therefore fires on every real bundle. Move the
>    membership/padding test up beside the `entry not in wanted` filter as a `continue`, and
>    rewrite the `SCORE_KEY` docstring, which currently documents the bug as a design decision.
> 2. **Negativity is scope-aware.** `layer_N` persistence and confidence are non-negative;
>    `combined` is a robust z-score built as the mean over layers of `(score - center) / scale`
>    against the clean median, and it is negative for any selection below that median — 548 of
>    the completed run's published `combined` statistics are, against none at `layer_2`. Refuse
>    a negative `layer_N` or confidence score; accept a negative `combined` one. Add
>    `COMBINED_SCOPE = "combined"` as an exported constant; Task 4 imports it.


**Files:**
- Create: `src/scene_uncertainty/contrast_inputs.py`
- Create: `tests/scene_uncertainty/contrast_test_utils.py`
- Create: `tests/scene_uncertainty/test_contrast_inputs.py`

**Interfaces:**
- Consumes: nothing from earlier tasks. Reads `per_scene.csv` and `summary.json` written by `corruption_reporting.write_corruption_report`.
- Produces: `Arm`, `ARMS`, `ContrastInputError`, `ContrastInputs`, `load_contrast_inputs(source, *, expected_image_count=FULL_TUNING_IMAGE_COUNT)`, `REQUIRED_SERIES`, and the test helper `write_source_bundle(...)`. Every later task imports the arm table from here.

**Why `expected_image_count` is a keyword argument and not a bare `250` literal:** the spec requires the real command to refuse anything but 250 images, and it does — that is the default. But a test that must materialise 250 images × 6 severities × 36 series to exercise a one-line validation branch is a test nobody will run often enough to trust. The existing `corruption_reporting.summarize_candidates` already takes `expected_image_count` for exactly this reason. The CLI never passes it, so production keeps the hard 250.

- [ ] **Step 1: Write the fixture builder**

Create `tests/scene_uncertainty/contrast_test_utils.py`:

```python
"""A finished corruption-sensitivity bundle, small enough to write in a test.

`write_source_bundle` produces the two files `load_contrast_inputs` reads and nothing else.
It is deliberately not a call into `corruption_reporting`: a fixture built by the producer
under test cannot catch a producer/consumer disagreement, and the whole point of the loader's
validation is to notice when the upstream bundle is not what this command expects.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

PER_SCENE_HEADER = (
    "image_id", "severity", "signal", "bucket_scheme", "confidence_bin",
    "membership_mode", "padding_mode", "aggregation", "score_scope",
    "source_partition", "score", "selected_count", "clean_overlap",
    "fully_measured", "signed_spearman", "absolute_spearman", "direction",
)

SERIES = (
    ("persistence", "decile", "decile_00_10", "layer_2"),
    ("persistence", "decile", "decile_50_60", "layer_2"),
    ("persistence", "decile", "decile_50_60", "combined"),
    ("persistence", "decile", "decile_90_100", "layer_2"),
    ("persistence", "decile", "decile_90_100", "combined"),
    ("persistence", "quintile", "quintile_00_20", "layer_2"),
    ("persistence", "quintile", "quintile_40_60", "layer_2"),
    ("confidence", "decile", "decile_00_10", "confidence"),
    ("confidence", "decile", "decile_50_60", "confidence"),
    ("confidence", "decile", "decile_90_100", "confidence"),
    ("confidence", "quintile", "quintile_00_20", "confidence"),
    ("confidence", "quintile", "quintile_40_60", "confidence"),
)

AGGREGATIONS = ("mean", "q90", "top20_mean")


def default_score(
    image_id: int, severity: int, confidence_bin: str, signal: str, scope: str
) -> float:
    """A score that differs along every axis the loader keys on.

    Every term is needed. Without the `image_id` term two images share a curve and a
    between-image spread test measures nothing. Without the `severity` term the trend is flat
    and every orientation test passes vacuously. Without the bin term the reference and
    responsive series are identical and every contrast is exactly zero. Without the `scope`
    term the `layer_2` and `combined` differential arms are byte-identical, and a mutation that
    read the wrong scope would pass every test in the suite.
    """
    base = 1.0 + 0.01 * image_id + (0.5 if scope == "combined" else 0.0)
    lift = {"decile_00_10": 0.0, "quintile_00_20": 0.0,
            "decile_50_60": 0.30, "quintile_40_60": 0.30,
            "decile_90_100": 0.60}[confidence_bin]
    slope = {"decile_00_10": 0.00, "quintile_00_20": 0.00,
             "decile_50_60": 0.05, "quintile_40_60": 0.05,
             "decile_90_100": -0.04}[confidence_bin]
    if signal == "confidence":
        return round(0.90 + 0.001 * image_id - 0.002 * severity, 6)
    return round(base + lift + slope * severity, 6)


def write_source_bundle(
    directory: Path,
    *,
    image_ids=range(1, 7),
    partition: str = "tuning",
    severities=range(6),
    membership_mode: str = "dynamic",
    padding_mode: str = "filtered",
    score=default_score,
    drop=(),
    extra_rows=(),
) -> Path:
    """Write `per_scene.csv` and `summary.json` into `directory` and return it.

    `drop` removes `(signal, confidence_bin, score_scope)` series so a test can prove the
    loader refuses an incomplete source. `extra_rows` appends raw dictionaries so a test can
    prove it refuses duplicates and held-out rows.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    image_ids = [int(value) for value in image_ids]
    rows = []
    for signal, scheme, confidence_bin, scope in SERIES:
        if (signal, confidence_bin, scope) in drop:
            continue
        for aggregation in AGGREGATIONS:
            for image_id in image_ids:
                for severity in severities:
                    rows.append({
                        "image_id": image_id, "severity": severity, "signal": signal,
                        "bucket_scheme": scheme, "confidence_bin": confidence_bin,
                        "membership_mode": membership_mode, "padding_mode": padding_mode,
                        "aggregation": aggregation, "score_scope": scope,
                        "source_partition": partition,
                        "score": score(image_id, severity, confidence_bin, signal, scope),
                        "selected_count": 30, "clean_overlap": 0.2,
                        "fully_measured": True, "signed_spearman": 0.6,
                        "absolute_spearman": 0.6, "direction": "increasing",
                    })
    rows.extend(dict(row) for row in extra_rows)
    with (directory / "per_scene.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PER_SCENE_HEADER))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "run": {
            "source_partition": partition,
            "image_count": len(image_ids),
            "severities": list(severities),
            "membership_modes": [membership_mode],
            "padding_modes": [padding_mode],
        },
        "per_scene_row_count": len(rows),
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return directory
```

- [ ] **Step 2: Write the failing tests**

Create `tests/scene_uncertainty/test_contrast_inputs.py`:

```python
import json

import pytest

from src.scene_uncertainty.contrast_inputs import (
    ARMS,
    ContrastInputError,
    load_contrast_inputs,
)

from tests.scene_uncertainty.contrast_test_utils import write_source_bundle


def test_arm_table_is_the_four_declared_arms():
    assert [arm.name for arm in ARMS] == [
        "decile_00_10__50_60",
        "quintile_00_20__40_60",
        "decile_90_100__50_60",
        "decile_90_100__50_60__combined",
    ]
    assert [arm.family for arm in ARMS] == [
        "anchored", "anchored", "differential", "differential"
    ]
    assert [arm.declared_before_data for arm in ARMS] == [True, True, False, False]
    assert [arm.score_scope for arm in ARMS] == [
        "layer_2", "layer_2", "layer_2", "combined"
    ]
    # the two differential arms share one bucket pair, and therefore one confidence twin
    assert [arm.pair_name for arm in ARMS] == [
        "decile_00_10__50_60",
        "quintile_00_20__40_60",
        "decile_90_100__50_60",
        "decile_90_100__50_60",
    ]
    assert len({arm.pair_name for arm in ARMS}) == 3


def test_loads_every_required_series(tmp_path):
    source = write_source_bundle(tmp_path / "source")
    inputs = load_contrast_inputs(source, expected_image_count=6)
    assert inputs.image_ids == (1, 2, 3, 4, 5, 6)
    # 12 series x 3 aggregations x 6 images x 6 severities
    assert len(inputs.scores) == 12 * 3 * 6 * 6
    assert inputs.scores[(1, 0, "persistence", "decile_50_60", "mean", "layer_2")] == 1.31
    assert inputs.provenance["source_partition"] == "tuning"


def test_refuses_a_held_out_source(tmp_path):
    source = write_source_bundle(tmp_path / "source", partition="held_out")
    with pytest.raises(ContrastInputError, match="tuning"):
        load_contrast_inputs(source, expected_image_count=6)


def test_refuses_a_missing_required_series(tmp_path):
    source = write_source_bundle(
        tmp_path / "source", drop=(("persistence", "decile_90_100", "combined"),)
    )
    with pytest.raises(ContrastInputError, match="decile_90_100.*combined"):
        load_contrast_inputs(source, expected_image_count=6)


def test_refuses_a_wrong_image_count(tmp_path):
    source = write_source_bundle(tmp_path / "source")
    with pytest.raises(ContrastInputError, match="250"):
        load_contrast_inputs(source)


def test_refuses_a_duplicated_row_key(tmp_path):
    source = write_source_bundle(tmp_path / "source")
    rows = (source / "per_scene.csv").read_text().splitlines()
    (source / "per_scene.csv").write_text("\n".join(rows + [rows[1]]) + "\n")
    with pytest.raises(ContrastInputError, match="duplicate"):
        load_contrast_inputs(source, expected_image_count=6)


def test_refuses_provenance_disagreement_between_csv_and_json(tmp_path):
    source = write_source_bundle(tmp_path / "source")
    summary = json.loads((source / "summary.json").read_text())
    summary["run"]["image_count"] = 99
    (source / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ContrastInputError, match="image_count"):
        load_contrast_inputs(source, expected_image_count=6)


def test_refuses_a_negative_persistence_distance(tmp_path):
    def negative(image_id, severity, confidence_bin, signal, scope):
        if image_id == 3 and confidence_bin == "decile_50_60" and signal == "persistence":
            return -0.5
        return 1.0

    source = write_source_bundle(tmp_path / "source", score=negative)
    with pytest.raises(ContrastInputError, match="negative"):
        load_contrast_inputs(source, expected_image_count=6)


@pytest.mark.parametrize("missing", ["per_scene.csv", "summary.json"])
def test_refuses_an_unfinished_source(tmp_path, missing):
    source = write_source_bundle(tmp_path / "source")
    (source / missing).unlink()
    with pytest.raises(ContrastInputError, match=missing):
        load_contrast_inputs(source, expected_image_count=6)
```

- [ ] **Step 3: Run the tests and read the failures**

```bash
$UE_PY -m pytest tests/scene_uncertainty/test_contrast_inputs.py -v
```

Expected: every test fails at import — `ModuleNotFoundError: No module named 'scene_uncertainty.contrast_inputs'`.

- [ ] **Step 4: Implement the loader**

Create `src/scene_uncertainty/contrast_inputs.py`:

```python
"""The four declared arms, and the only door a finished bundle comes through.

This module is the whole of the command's contact with the filesystem's input side. Everything
downstream sees a `ContrastInputs`, which is a flat dictionary of numbers plus provenance --
so no later module has to know that the source was a CSV, and no later module can quietly
widen the set of rows the experiment is allowed to see.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

SOURCE_FILES = ("per_scene.csv", "summary.json")
SOURCE_PARTITION = "tuning"
FULL_TUNING_IMAGE_COUNT = 250
EXPECTED_SEVERITIES = tuple(range(6))
MEMBERSHIP_MODE = "dynamic"
PADDING_MODE = "filtered"
PERSISTENCE_SIGNAL = "persistence"
CONFIDENCE_SIGNAL = "confidence"
CONFIDENCE_SCOPE = "confidence"
AGGREGATIONS = ("mean", "q90", "top20_mean")

SCORE_KEY = ("image_id", "severity", "signal", "confidence_bin", "aggregation", "score_scope")
"""What makes two source scores measurements of different things, once the fixed fields are gone.

`membership_mode` and `padding_mode` are absent because they are *validated* to single values
rather than varied: a source carrying a frozen or unfiltered row for a required series is
refused, not filed under a longer key. `bucket_scheme` is absent because `confidence_bin`
already determines it -- `decile_00_10` and `quintile_00_20` cannot collide -- and a redundant
key field is a second place for two rows to be told apart, which is one more than there should
be.
"""


class ContrastInputError(ValueError):
    """A source bundle this command will not read.

    A `ValueError` so `pipeline.command_analyze_within_image_contrast` turns it into one line
    on stderr through the same `except ValueError` its siblings use, rather than a traceback
    through frames the operator did not write.
    """


@dataclass(frozen=True)
class Arm:
    """One bucket pair at one persistence scope, with the provenance of how it was chosen.

    `pair_name` is the bucket pair without the scope, and it is a stored field rather than a
    string derived from `name` because it is what the confidence-only twin is keyed on.
    Confidence has no decoder layer, so the two differential arms -- `layer_2` and `combined`
    over the same two buckets -- have one twin between them, and that identity has to be
    something the code states rather than something a suffix-stripping rule infers.
    """

    name: str
    pair_name: str
    family: str
    bucket_scheme: str
    score_scope: str
    reference_bin: str
    responsive_bin: str
    declared_before_data: bool


ARMS = (
    Arm("decile_00_10__50_60", "decile_00_10__50_60", "anchored", "decile", "layer_2",
        "decile_00_10", "decile_50_60", True),
    Arm("quintile_00_20__40_60", "quintile_00_20__40_60", "anchored", "quintile", "layer_2",
        "quintile_00_20", "quintile_40_60", True),
    Arm("decile_90_100__50_60", "decile_90_100__50_60", "differential", "decile", "layer_2",
        "decile_90_100", "decile_50_60", False),
    Arm("decile_90_100__50_60__combined", "decile_90_100__50_60", "differential", "decile",
        "combined", "decile_90_100", "decile_50_60", False),
)
"""The closed set. Four arms, and adding a fifth is a spec change.

`declared_before_data` is a field rather than a comment because it is the difference between a
result and a hypothesis: the two anchored arms were named before any tuning number was read,
and the two differential arms were named after the completed deployment analysis showed the
90--100 percent decile moving hardest and against the 50--60 percent range. A tuning macro
AUROC from an arm with `declared_before_data=False` is a selection estimate. Carrying the flag
all the way to the easy report is what stops it being quoted as performance.
"""


def required_series() -> tuple[tuple[str, str, str], ...]:
    """Every `(signal, confidence_bin, score_scope)` the four arms need, deduplicated.

    Derived from `ARMS` rather than listed, so an arm cannot be added without the loader
    demanding its rows. The confidence entries collapse across scope on purpose: confidence has
    no decoder layer, so the two differential arms need one confidence series between them and
    that is also why they share a single confidence-only twin downstream.
    """
    series: list[tuple[str, str, str]] = []
    for arm in ARMS:
        for confidence_bin in (arm.reference_bin, arm.responsive_bin):
            for entry in (
                (PERSISTENCE_SIGNAL, confidence_bin, arm.score_scope),
                (CONFIDENCE_SIGNAL, confidence_bin, CONFIDENCE_SCOPE),
            ):
                if entry not in series:
                    series.append(entry)
    return tuple(series)


REQUIRED_SERIES = required_series()


@dataclass(frozen=True)
class ContrastInputs:
    """The retained scores, the image roster, and where they came from."""

    scores: dict[tuple, float]
    image_ids: tuple[int, ...]
    provenance: dict


def _read_summary(source: Path) -> dict:
    try:
        return json.loads((source / "summary.json").read_text())
    except FileNotFoundError as error:
        raise ContrastInputError(
            f"source is not a finished corruption-sensitivity bundle: summary.json "
            f"is missing from {source}"
        ) from error
    except json.JSONDecodeError as error:
        raise ContrastInputError(f"summary.json in {source} is not valid JSON: {error}") from error


def load_contrast_inputs(
    source_value: str | Path, *, expected_image_count: int = FULL_TUNING_IMAGE_COUNT
) -> ContrastInputs:
    """Every score the four arms need, and a refusal if the source cannot supply them all.

    The checks run in this order because each one makes the next one's message readable. The
    partition is checked first: a held-out bundle is refused outright rather than filtered,
    because silently dropping rows from a held-out source is how a held-out claim gets made by
    accident. Provenance agreement between the CSV and the JSON comes next, because a
    disagreement means one of the two is describing a different run and every count after this
    point would be measuring the wrong thing. Only then are the rows filed, and only then can a
    missing series be reported as "this bundle does not contain what this experiment needs"
    rather than as a `KeyError` five frames away.

    Rows outside `REQUIRED_SERIES` are discarded, not refused. The source legitimately holds
    765,000 rows across ten deciles, five quintiles, four scopes, two membership modes and two
    padding modes; this command reads 54,000 of them. Refusing the rest would refuse every real
    bundle.
    """
    source = Path(source_value)
    summary = _read_summary(source)
    run = summary.get("run", {})
    if run.get("source_partition") != SOURCE_PARTITION:
        raise ContrastInputError(
            f"within-image contrast reads the tuning partition only; {source} reports "
            f"source_partition={run.get('source_partition')!r}"
        )

    path = source / "per_scene.csv"
    try:
        handle = path.open(newline="")
    except FileNotFoundError as error:
        raise ContrastInputError(
            f"source is not a finished corruption-sensitivity bundle: per_scene.csv "
            f"is missing from {source}"
        ) from error

    wanted = set(REQUIRED_SERIES)
    scores: dict[tuple, float] = {}
    images: set[int] = set()
    seen_series: set[tuple[str, str, str]] = set()
    with handle:
        for index, row in enumerate(csv.DictReader(handle)):
            entry = (row["signal"], row["confidence_bin"], row["score_scope"])
            if entry not in wanted:
                continue
            if row["source_partition"] != SOURCE_PARTITION:
                raise ContrastInputError(
                    f"per_scene.csv row {index} is outside the tuning partition: "
                    f"source_partition={row['source_partition']!r}"
                )
            if row["membership_mode"] != MEMBERSHIP_MODE or row["padding_mode"] != PADDING_MODE:
                raise ContrastInputError(
                    f"per_scene.csv row {index} for {entry} is "
                    f"{row['membership_mode']}/{row['padding_mode']}; this experiment reads "
                    f"{MEMBERSHIP_MODE}/{PADDING_MODE} only"
                )
            if row["aggregation"] not in AGGREGATIONS:
                continue
            score = float(row["score"])
            if row["signal"] == PERSISTENCE_SIGNAL and score < 0.0:
                raise ContrastInputError(
                    f"per_scene.csv row {index} has a negative persistence distance: {score}"
                )
            if score != score:
                raise ContrastInputError(f"per_scene.csv row {index} has a non-finite score")
            key = (
                int(row["image_id"]), int(row["severity"]), row["signal"],
                row["confidence_bin"], row["aggregation"], row["score_scope"],
            )
            if key in scores:
                raise ContrastInputError(f"duplicate source row key: {key}")
            scores[key] = score
            images.add(key[0])
            seen_series.add(entry)

    missing = [entry for entry in REQUIRED_SERIES if entry not in seen_series]
    if missing:
        described = ", ".join(f"{signal}/{name}/{scope}" for signal, name, scope in missing)
        raise ContrastInputError(f"source is missing required series: {described}")

    image_ids = tuple(sorted(images))
    if len(image_ids) != expected_image_count:
        raise ContrastInputError(
            f"within-image contrast needs {expected_image_count} tuning images; "
            f"{source} has {len(image_ids)}"
        )
    if run.get("image_count") != len(image_ids):
        raise ContrastInputError(
            f"summary.json reports image_count={run.get('image_count')} but per_scene.csv "
            f"contains {len(image_ids)} images"
        )
    if tuple(run.get("severities", ())) != EXPECTED_SEVERITIES:
        raise ContrastInputError(
            f"within-image contrast needs severities {list(EXPECTED_SEVERITIES)}; "
            f"{source} reports {run.get('severities')}"
        )

    expected = len(REQUIRED_SERIES) * len(AGGREGATIONS) * len(image_ids) * len(EXPECTED_SEVERITIES)
    if len(scores) != expected:
        raise ContrastInputError(
            f"source coverage is incomplete: expected {expected} retained rows, found "
            f"{len(scores)}; every required series must cover every image at every severity"
        )

    return ContrastInputs(
        scores=scores,
        image_ids=image_ids,
        provenance={
            "source": str(source),
            "source_partition": SOURCE_PARTITION,
            "image_count": len(image_ids),
            "severities": list(EXPECTED_SEVERITIES),
            "membership_mode": MEMBERSHIP_MODE,
            "padding_mode": PADDING_MODE,
            "aggregations": list(AGGREGATIONS),
            "required_series": [list(entry) for entry in REQUIRED_SERIES],
            "retained_row_count": len(scores),
        },
    )
```

- [ ] **Step 5: Run the tests and verify they pass**

```bash
$UE_PY -m pytest tests/scene_uncertainty/test_contrast_inputs.py -v
```

Expected: 10 passed.

- [ ] **Step 6: Mutation check**

Make each of these breaks one at a time, confirm a *named* test fails, then revert:

1. Change `if key in scores:` to `if False:` → `test_refuses_a_duplicated_row_key` must fail.
2. Delete the `run.get("image_count") != len(image_ids)` block → `test_refuses_provenance_disagreement_between_csv_and_json` must fail.
3. Change `missing = [...]` to `missing = []` → `test_refuses_a_missing_required_series` must fail.
4. Change `score < 0.0` to `score < -1e9` → `test_refuses_a_negative_persistence_distance` must fail.

If any break leaves the suite green, the test is not binding — fix the test before moving on.

- [ ] **Step 7: Commit**

```bash
git add src/scene_uncertainty/contrast_inputs.py \
        tests/scene_uncertainty/contrast_test_utils.py \
        tests/scene_uncertainty/test_contrast_inputs.py
git commit -m "feat: load and validate a finished corruption-sensitivity bundle"
```

---

## Task 2: Pure score methods, deterministic folds, and the robust clean line

**Files:**
- Create: `src/scene_uncertainty/contrast_scores.py`
- Create: `tests/scene_uncertainty/test_contrast_scores.py`

**Interfaces:**
- Consumes: nothing. This module is pure arithmetic — no I/O, no arm knowledge, no pandas.
- Produces: `SCORE_METHODS`, `FOLD_COUNT`, `raw_responsive`, `raw_gap`, `relative_gap`, `clean_residual`, `contrast_score`, `assign_folds`, `robust_line`. Tasks 4, 5 and 6 call all of these.

- [ ] **Step 1: Write the failing tests**

Create `tests/scene_uncertainty/test_contrast_scores.py`:

```python
import math

import pytest

from src.scene_uncertainty.contrast_scores import (
    FOLD_COUNT,
    SCORE_METHODS,
    assign_folds,
    clean_residual,
    contrast_score,
    raw_gap,
    raw_responsive,
    relative_gap,
    robust_line,
)


def test_method_names_are_the_four_declared_ones():
    assert SCORE_METHODS == ("raw_responsive", "raw_gap", "relative_gap", "clean_residual")


def test_raw_responsive_returns_the_responsive_input_exactly():
    assert raw_responsive(0.4, 0.9) == 0.9
    assert raw_responsive(9.9, 0.9) == 0.9


def test_raw_gap_is_signed_and_keeps_negatives():
    assert raw_gap(0.3, 0.8) == pytest.approx(0.5)
    assert raw_gap(0.8, 0.3) == pytest.approx(-0.5)


def test_relative_gap_is_scale_invariant():
    assert relative_gap(1.0, 2.0) == pytest.approx(relative_gap(10.0, 20.0))
    assert relative_gap(1.0, 2.0) == pytest.approx(2.0 / 3.0)


def test_equal_positive_inputs_give_zero_relative_gap():
    assert relative_gap(0.7, 0.7) == 0.0


def test_zero_plus_zero_gives_zero_relative_gap():
    assert relative_gap(0.0, 0.0) == 0.0


@pytest.mark.parametrize(
    "reference, responsive",
    [(0.0, 5.0), (5.0, 0.0), (1e-9, 4.0), (4.0, 1e-9), (2.5, 2.5)],
)
def test_relative_gap_stays_within_two(reference, responsive):
    value = relative_gap(reference, responsive)
    assert -2.0 <= value <= 2.0


def test_relative_gap_reaches_the_bounds_only_when_one_side_is_zero():
    assert relative_gap(0.0, 5.0) == pytest.approx(2.0)
    assert relative_gap(5.0, 0.0) == pytest.approx(-2.0)


def test_clean_residual_subtracts_the_predicted_clean_responsive():
    # expected clean responsive = 1.0 + 2.0 * 0.5 = 2.0; observed 2.75 -> +0.75
    assert clean_residual(0.5, 2.75, slope=2.0, offset=1.0) == pytest.approx(0.75)
    assert clean_residual(0.5, 1.25, slope=2.0, offset=1.0) == pytest.approx(-0.75)


@pytest.mark.parametrize("reference, responsive", [(-0.1, 1.0), (1.0, -0.1)])
def test_only_the_relative_gap_rejects_a_negative_input(reference, responsive):
    """The combined scope is a signed z-score, so three of the four methods must accept it."""
    with pytest.raises(ValueError, match="non-negative"):
        relative_gap(reference, responsive)
    assert raw_responsive(reference, responsive) == responsive
    assert raw_gap(reference, responsive) == pytest.approx(responsive - reference)
    assert clean_residual(
        reference, responsive, slope=1.0, offset=0.0
    ) == pytest.approx(responsive - reference)


def test_a_signed_combined_style_gap_is_computed_not_refused():
    """Real values from the completed run's combined-scope decile_90_100 arm."""
    assert raw_gap(-0.2599, -0.0230) == pytest.approx(0.2369)
    assert contrast_score("raw_gap", -0.2599, -0.0230) == pytest.approx(0.2369)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_distances_are_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        relative_gap(1.0, bad)


def test_contrast_score_dispatches_to_each_method():
    assert contrast_score("raw_responsive", 0.3, 0.8) == 0.8
    assert contrast_score("raw_gap", 0.3, 0.8) == pytest.approx(0.5)
    assert contrast_score("relative_gap", 0.3, 0.8) == pytest.approx(1.0 / 1.1)
    assert contrast_score(
        "clean_residual", 0.5, 2.75, line=(2.0, 1.0)
    ) == pytest.approx(0.75)


def test_contrast_score_refuses_a_residual_without_a_line():
    with pytest.raises(ValueError, match="clean_residual"):
        contrast_score("clean_residual", 0.5, 2.75)


def test_contrast_score_refuses_an_unknown_method():
    with pytest.raises(ValueError, match="unknown"):
        contrast_score("median_gap", 0.5, 2.75)


def test_robust_line_recovers_a_hand_checkable_line():
    references = [0.0, 1.0, 2.0, 3.0, 4.0]
    responsives = [1.0, 3.0, 5.0, 7.0, 9.0]  # y = 2x + 1
    assert robust_line(references, responsives) == pytest.approx((2.0, 1.0))


def test_one_extreme_outlier_does_not_control_the_robust_fit():
    references = [float(index) for index in range(10)] + [4.0]
    responsives = [2.0 * index + 1.0 for index in range(10)] + [900.0]
    slope, offset = robust_line(references, responsives)
    assert slope == pytest.approx(2.0)
    assert offset == pytest.approx(1.0)


def test_repeated_reference_values_skip_undefined_pairwise_slopes():
    # the one pair with equal references contributes no slope; the five usable slopes
    # are 2, 2, 0, 1, 2 -> median 2, and offset = median([3, 5, 3, 3]) = 3
    assert robust_line([1.0, 1.0, 2.0, 3.0], [5.0, 7.0, 7.0, 9.0]) == pytest.approx((2.0, 3.0))


def test_a_constant_reference_makes_the_line_unavailable():
    assert robust_line([1.0, 1.0, 1.0], [2.0, 3.0, 4.0]) is None


def test_a_single_point_makes_the_line_unavailable():
    assert robust_line([1.0], [2.0]) is None


def test_robust_line_refuses_mismatched_lengths():
    with pytest.raises(ValueError, match="one responsive value per reference"):
        robust_line([1.0, 2.0], [3.0])


def test_fold_assignment_is_deterministic_under_reordering():
    forward = assign_folds([1, 3, 5, 7, 10])
    backward = assign_folds([10, 7, 5, 3, 1])
    assert forward == backward == {1: 0, 3: 1, 5: 2, 7: 3, 10: 4}


def test_fold_assignment_wraps_at_the_fold_count():
    folds = assign_folds(range(1, 13))
    assert FOLD_COUNT == 5
    assert [folds[image_id] for image_id in range(1, 13)] == [
        0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1
    ]


def test_every_fold_is_non_empty_for_a_full_tuning_run():
    folds = assign_folds(range(1, 251))
    counts = [sum(1 for value in folds.values() if value == fold) for fold in range(FOLD_COUNT)]
    assert counts == [50, 50, 50, 50, 50]
```

- [ ] **Step 2: Run the tests and read the failures**

```bash
$UE_PY -m pytest tests/scene_uncertainty/test_contrast_scores.py -v
```

Expected: collection error — `ModuleNotFoundError: No module named 'scene_uncertainty.contrast_scores'`.

- [ ] **Step 3: Implement the pure module**

Create `src/scene_uncertainty/contrast_scores.py`:

```python
"""The four contrast methods, the fold rule, and the robust clean line.

Everything here takes numbers and returns numbers. Nothing here knows what an arm is, which
bucket a score came from, or that a filesystem exists -- which is what makes the whole module
testable by hand arithmetic, and what stops a change of experiment design from needing a change
of maths.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

SCORE_METHODS = ("raw_responsive", "raw_gap", "relative_gap", "clean_residual")
FOLD_COUNT = 5


def _validated(
    reference: float, responsive: float, *, non_negative: bool = False
) -> tuple[float, float]:
    """Both inputs as finite floats, and non-negative too when the caller needs that.

    Non-finite is always refused, for the reason `complete_trend_metrics` refuses it: a `nan`
    propagates through subtraction into a score that is neither a measurement nor an absence,
    and lands in a CSV cell that reads as neither.

    Non-negativity is *not* always refused, and that is the correction the completed run
    forced. A `layer_N` score is a raw mean-kNN distance and cannot be negative. A `combined`
    score is a robust z-score against the clean median and goes negative for any selection
    below it -- 548 of the completed run's published `combined` statistics are, against none at
    `layer_2`. So the check belongs to the one method that mathematically requires it rather
    than to every method: `relative_gap`'s scale invariance and its bounds both collapse on a
    signed input, while a raw gap and a residual are perfectly well defined on one.
    """
    reference = float(reference)
    responsive = float(responsive)
    if not (math.isfinite(reference) and math.isfinite(responsive)):
        raise ValueError(
            f"contrast inputs must be finite: reference={reference}, responsive={responsive}"
        )
    if non_negative and (reference < 0.0 or responsive < 0.0):
        raise ValueError(
            f"the symmetric relative gap needs non-negative inputs: reference={reference}, "
            f"responsive={responsive}; a signed scope must not declare this method"
        )
    return reference, responsive


def raw_responsive(reference: float, responsive: float) -> float:
    """The responsive range alone: the control every contrast has to beat.

    Takes `reference` it does not use, so that the four methods share one signature and the
    caller's dispatch table cannot pair a method with the wrong arguments. Validation still
    runs on both, because a control computed from a row whose reference is corrupt is a control
    over a different population than the contrast it is being compared with.
    """
    _, responsive = _validated(reference, responsive)
    return responsive


def raw_gap(reference: float, responsive: float) -> float:
    """`responsive - reference`, signed.

    Signed rather than absolute. An absolute value would map "the responsive range moved
    unusually far above its baseline" and "the responsive range collapsed below it" onto the
    same number, and those are opposite events. The candidate's locked orientation is what
    turns a consistently negative score into a usable one, and it can only do that if the sign
    survives to it.
    """
    reference, responsive = _validated(reference, responsive)
    return responsive - reference


def relative_gap(reference: float, responsive: float) -> float:
    """`2 * (responsive - reference) / (responsive + reference)`, the scale-free contrast.

    Two scenes whose raw distances differ tenfold get the same value when the
    responsive-to-reference relationship is proportional, which is the multiplicative
    counterpart to what `raw_gap` removes additively.

    This method alone requires non-negative inputs, and refuses a negative one rather than
    accommodating it: an arm whose scope can go negative has no business declaring this method,
    and Task 4 excludes it there. Given non-negative inputs the denominator cannot be negative
    and the only degenerate case is both being exactly zero -- defined as zero, because
    "neither range moved at all" is an absence of contrast and not an undefined one. No epsilon is added. An epsilon would be a
    tunable constant sitting inside a score that is otherwise entirely determined by the data,
    and it would make the near-zero region's values a function of a number nobody chose on
    evidence.
    """
    reference, responsive = _validated(reference, responsive, non_negative=True)
    total = responsive + reference
    if total == 0.0:
        return 0.0
    return 2.0 * (responsive - reference) / total


def clean_residual(
    reference: float, responsive: float, *, slope: float, offset: float
) -> float:
    """How far the responsive range sits above what a clean image with this baseline would show.

    `slope` and `offset` come from clean severity-zero rows only, so the line encodes the normal
    clean relationship and nothing about corruption. A positive residual therefore means "higher
    than clean-normal for this scene", which is the claim the score is making.
    """
    reference, responsive = _validated(reference, responsive)
    return responsive - (offset + slope * reference)


def contrast_score(
    method: str,
    reference: float,
    responsive: float,
    *,
    line: tuple[float, float] | None = None,
) -> float:
    """One score by name, with the residual's line supplied rather than looked up.

    `line` is an argument instead of module state because the residual is cross-fitted: the
    same `(arm, summary, signal)` uses five different lines across the five folds, and a
    module-level cache would hand one image the line its own fold fitted. Passing it in makes
    the fold the caller's responsibility, which is where the fold assignment already lives.

    A missing line for `clean_residual` raises rather than returning `None`. The caller knows
    before it starts whether `robust_line` returned a line, and a residual candidate whose line
    is unavailable is excluded with a recorded reason -- not silently filled with a value that
    would read as a measurement.
    """
    if method == "raw_responsive":
        return raw_responsive(reference, responsive)
    if method == "raw_gap":
        return raw_gap(reference, responsive)
    if method == "relative_gap":
        return relative_gap(reference, responsive)
    if method == "clean_residual":
        if line is None:
            raise ValueError(
                "clean_residual needs a fitted line; a candidate whose clean relationship is "
                "unavailable must be excluded, not scored"
            )
        slope, offset = line
        return clean_residual(reference, responsive, slope=slope, offset=offset)
    raise ValueError(f"unknown contrast score method: {method!r}")


def assign_folds(image_ids: Iterable[int]) -> dict[int, int]:
    """Sorted image position modulo five, as `{image_id: fold}`.

    Sorted rather than input-ordered, so the assignment is a property of the image roster and
    not of whatever order the rows happened to arrive in. Two runs over the same 250 images
    produce the same five folds whether the CSV was written by severity or by candidate, which
    is what makes a stored fold assignment worth comparing across runs at all.

    All six severities of an image share its fold -- enforced by keying on `image_id` alone,
    which is the only place that rule can be enforced once. Assigning per row would let an
    image's severity-3 measurement help fit the line that predicts its severity-0 value.
    """
    ordered = sorted({int(image_id) for image_id in image_ids})
    return {image_id: index % FOLD_COUNT for index, image_id in enumerate(ordered)}


def robust_line(
    references: Sequence[float], responsives: Sequence[float]
) -> tuple[float, float] | None:
    """Median pairwise slope, then median intercept. `None` when the reference is constant.

    A Theil-Sen line rather than least squares, and deterministic rather than sampled. Least
    squares would let one clean scene with an extreme baseline set the slope that every other
    scene's residual is measured against; the median of the pairwise slopes needs more than
    half the pairs to move before it does. With 250 clean images the 31,125 pairs are cheap
    enough to enumerate outright, so there is no sampling step and therefore no seed and no
    run-to-run variation in a number that becomes a deployment constant.

    Pairs with equal reference values are dropped rather than counted: their slope is a
    division by zero, and treating it as infinite or as zero would both be inventing a
    measurement from two points that say nothing about slope. Fewer than two distinct finite
    reference values leaves nothing to fit, and that returns `None` -- never a zero slope,
    which would look like "the ranges are unrelated" rather than "this could not be measured".
    """
    reference = np.asarray(references, dtype=float)
    responsive = np.asarray(responsives, dtype=float)
    if reference.shape != responsive.shape:
        raise ValueError(
            "robust line needs one responsive value per reference value: "
            f"{reference.shape} against {responsive.shape}"
        )
    finite = np.isfinite(reference) & np.isfinite(responsive)
    reference = reference[finite]
    responsive = responsive[finite]
    if np.unique(reference).size < 2:
        return None
    left, right = np.triu_indices(reference.size, k=1)
    run = reference[right] - reference[left]
    usable = run != 0.0
    slopes = (responsive[right][usable] - responsive[left][usable]) / run[usable]
    slope = float(np.median(slopes))
    offset = float(np.median(responsive - slope * reference))
    return slope, offset
```

- [ ] **Step 4: Run the tests and verify they pass**

```bash
$UE_PY -m pytest tests/scene_uncertainty/test_contrast_scores.py -v
```

Expected: 31 passed.

- [ ] **Step 5: Mutation check**

Break each, confirm the *named* test fails, revert:

1. `relative_gap` → drop the `2.0 *` factor. `test_relative_gap_reaches_the_bounds_only_when_one_side_is_zero` must fail. (`test_relative_gap_is_scale_invariant` alone would still pass — that is exactly the "looks binding, is not" shape this step exists to catch.)
2. `raw_gap` → return `abs(responsive - reference)`. `test_raw_gap_is_signed_and_keeps_negatives` must fail.
3. `robust_line` → use `np.mean(slopes)` instead of `np.median`. `test_one_extreme_outlier_does_not_control_the_robust_fit` must fail.
4. `robust_line` → drop the `usable` filter (expect a divide-by-zero warning and a `nan` slope). `test_repeated_reference_values_skip_undefined_pairwise_slopes` must fail.
5. `robust_line` → return `(0.0, 0.0)` instead of `None` for a constant reference. `test_a_constant_reference_makes_the_line_unavailable` must fail.
6. `assign_folds` → drop the `sorted(...)`, iterating the input order. `test_fold_assignment_is_deterministic_under_reordering` must fail.
7. `_validated` → drop the `non_negative` guard. `test_only_the_relative_gap_rejects_a_negative_input` must fail.
8. `relative_gap` → call `_validated` without `non_negative=True`. The same test must fail.
9. `raw_gap` → pass `non_negative=True`. `test_a_signed_combined_style_gap_is_computed_not_refused` must fail.

- [ ] **Step 6: Commit**

```bash
git add src/scene_uncertainty/contrast_scores.py tests/scene_uncertainty/test_contrast_scores.py
git commit -m "feat: add pure contrast score methods, folds, and robust clean line"
```

---

## Task 3: Anchor diagnostics

**Files:**
- Create: `src/scene_uncertainty/contrast_diagnostics.py`
- Create: `tests/scene_uncertainty/test_contrast_diagnostics.py`

**Interfaces:**
- Consumes: `contrast_scores.robust_line`, `contrast_scores.assign_folds` (Task 2); `corruption_metrics.complete_trend_metrics` (existing).
- Produces: `within_image_drift(curves)`, `between_image_spread(curves, drift)`, `clean_relationship(references, responsives, *, folds)`. Task 5 calls all three once per `(arm, summary, signal)`; Task 7 writes their output to `anchor_diagnostics.csv`.

**These diagnostics gate only the anchored arms.** They run on all four and are reported in full, but a differential arm is *expected* to fail them — its reference range responds to blur by design. This module computes; it does not judge. The `arm_family` label that decides how a number is read is attached in Task 5, and the pass/fail rule lives in Task 8's success reporting.

- [ ] **Step 1: Write the failing tests**

Create `tests/scene_uncertainty/test_contrast_diagnostics.py`:

```python
import pytest

from src.scene_uncertainty.contrast_diagnostics import (
    between_image_spread,
    clean_relationship,
    within_image_drift,
)
from src.scene_uncertainty.contrast_scores import assign_folds

SEVERITIES = range(6)


def flat_curves():
    """Four images whose reference never moves: baselines 1.0, 2.0, 3.0, 4.0."""
    return {
        image_id: {severity: float(image_id) for severity in SEVERITIES}
        for image_id in (1, 2, 3, 4)
    }


def drifting_curves(step=0.1):
    """The same four baselines, each rising by `step` per severity."""
    return {
        image_id: {severity: image_id + step * severity for severity in SEVERITIES}
        for image_id in (1, 2, 3, 4)
    }


def test_a_flat_reference_has_zero_drift_everywhere():
    drift = within_image_drift(flat_curves())
    for severity in range(1, 6):
        row = drift["by_severity"][severity]
        assert row["median_signed_drift"] == 0.0
        assert row["median_absolute_drift"] == 0.0
        assert row["zero_drift_fraction"] == 1.0


def test_a_flat_reference_is_flat_not_unmeasured():
    drift = within_image_drift(flat_curves())
    for image_id in (1, 2, 3, 4):
        image = drift["by_image"][image_id]
        assert image["reference_range"] == 0.0
        assert image["signed_spearman"] == 0.0
        assert image["absolute_spearman"] == 0.0


def test_drift_is_measured_against_each_image_own_severity_zero():
    drift = within_image_drift(drifting_curves())
    # every image drifts the same amount, so the median equals that amount exactly
    assert drift["by_severity"][2]["median_signed_drift"] == pytest.approx(0.2)
    assert drift["by_severity"][5]["median_signed_drift"] == pytest.approx(0.5)
    assert drift["by_severity"][3]["zero_drift_fraction"] == 0.0
    assert drift["by_image"][1]["reference_range"] == pytest.approx(0.5)


def test_signed_and_absolute_drift_differ_when_images_move_opposite_ways():
    curves = {
        1: {severity: 1.0 + 0.1 * severity for severity in SEVERITIES},
        2: {severity: 2.0 - 0.1 * severity for severity in SEVERITIES},
    }
    row = within_image_drift(curves)["by_severity"][4]
    assert row["median_signed_drift"] == pytest.approx(0.0)
    assert row["median_absolute_drift"] == pytest.approx(0.4)


def test_between_image_spread_reports_the_full_shape():
    spread = between_image_spread(flat_curves(), within_image_drift(flat_curves()))
    clean = spread[0]
    assert clean["count"] == 4
    assert clean["mean"] == pytest.approx(2.5)
    assert clean["variance"] == pytest.approx(1.25)  # population, not sample
    assert clean["median"] == pytest.approx(2.5)
    assert clean["q25"] == pytest.approx(1.75)
    assert clean["q75"] == pytest.approx(3.25)
    assert clean["iqr"] == pytest.approx(1.5)
    assert clean["mad"] == pytest.approx(1.0)
    assert clean["min"] == pytest.approx(1.0)
    assert clean["max"] == pytest.approx(4.0)


def test_severity_zero_has_no_stability_to_spread_ratio():
    spread = between_image_spread(flat_curves(), within_image_drift(flat_curves()))
    assert spread[0]["stability_to_spread"] is None


def test_stability_to_spread_divides_drift_by_the_clean_interquartile_range():
    curves = drifting_curves()
    spread = between_image_spread(curves, within_image_drift(curves))
    # median absolute drift at severity 2 is 0.2; the clean IQR of [1, 2, 3, 4] is 1.5
    assert spread[2]["stability_to_spread"] == pytest.approx(0.2 / 1.5)


def test_a_zero_clean_spread_makes_the_ratio_unavailable():
    curves = {
        image_id: {severity: 3.0 + 0.1 * severity for severity in SEVERITIES}
        for image_id in (1, 2, 3, 4)
    }
    spread = between_image_spread(curves, within_image_drift(curves))
    assert spread[3]["stability_to_spread"] is None


def test_a_predictive_clean_relationship_beats_the_constant_median():
    references = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
    responsives = {image_id: 2.0 * value + 1.0 for image_id, value in references.items()}
    result = clean_relationship(
        references, responsives, folds=assign_folds(references)
    )
    assert result["pearson"] == pytest.approx(1.0)
    assert result["spearman"] == pytest.approx(1.0)
    assert result["final_slope"] == pytest.approx(2.0)
    assert result["final_offset"] == pytest.approx(1.0)
    assert result["crossfit_median_absolute_error"] == pytest.approx(0.0)
    assert result["constant_median_absolute_error"] == pytest.approx(3.0)
    assert result["predictive"] is True


def test_a_non_predictive_clean_relationship_is_labelled_so():
    references = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
    responsives = {1: 5.0, 2: 5.1, 3: 4.9, 4: 5.0}
    result = clean_relationship(
        references, responsives, folds=assign_folds(references)
    )
    assert result["crossfit_median_absolute_error"] == pytest.approx(0.125)
    assert result["constant_median_absolute_error"] == pytest.approx(0.05)
    assert result["predictive"] is False


def test_a_constant_reference_leaves_the_relationship_unavailable():
    references = {1: 2.0, 2: 2.0, 3: 2.0, 4: 2.0}
    responsives = {1: 5.0, 2: 6.0, 3: 7.0, 4: 8.0}
    result = clean_relationship(
        references, responsives, folds=assign_folds(references)
    )
    assert result["final_slope"] is None
    assert result["final_offset"] is None
    assert result["crossfit_median_absolute_error"] is None
    assert result["predictive"] is False
    assert result["pearson"] is None


def test_every_fold_line_excludes_its_own_fold():
    references = {image_id: float(image_id) for image_id in range(1, 11)}
    responsives = {image_id: 2.0 * float(image_id) + 1.0 for image_id in range(1, 11)}
    folds = assign_folds(references)
    # image 1 alone gets an extreme responsive value; if its own fold's line saw it,
    # that fold's slope would move away from 2.0
    responsives[1] = 500.0
    result = clean_relationship(references, responsives, folds=folds)
    assert result["fold_lines"][folds[1]][0] == pytest.approx(2.0)
```

- [ ] **Step 2: Run the tests and read the failures**

```bash
$UE_PY -m pytest tests/scene_uncertainty/test_contrast_diagnostics.py -v
```

Expected: collection error — module does not exist.

- [ ] **Step 3: Implement the diagnostics**

Create `src/scene_uncertainty/contrast_diagnostics.py`:

```python
"""Whether the reference range behaves like an anchor, measured three ways.

Three questions, deliberately not collapsed into one score. Does the reference stay still
within an image as blur increases (`within_image_drift`)? Does it differ enough between scenes
to be worth subtracting (`between_image_spread`)? And does it actually predict the responsive
range's clean level, better than a constant would (`clean_relationship`)? A reference can pass
any one of these and be useless: a perfectly stable reference that is identical for every
scene explains no scene-specific baseline at all, and a reference with enormous between-scene
spread that wanders under blur injects that wander into every contrast built on it.

Nothing here decides anything. The differential arms are expected to fail the drift and
stability checks by construction, and the module that reads these numbers is the one that
knows which family an arm belongs to.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .contrast_scores import FOLD_COUNT, robust_line
from .corruption_metrics import EXPECTED_SEVERITIES, complete_trend_metrics


def _statistics(values: np.ndarray) -> dict:
    """Ten numbers describing one severity's spread of reference scores.

    Population variance (`ddof=0`), not sample variance. These 250 images are the tuning
    partition entire, not a sample drawn from it, and the question being asked is how widely
    this reference actually spread over the images it was measured on.
    """
    q25, median, q75 = (float(value) for value in np.percentile(values, [25, 50, 75]))
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "variance": float(np.var(values)),
        "median": median,
        "q25": q25,
        "q75": q75,
        "iqr": q75 - q25,
        "mad": float(np.median(np.abs(values - median))),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def within_image_drift(curves: dict[int, dict[int, float]]) -> dict:
    """How far each image's reference moves from its own severity-zero value.

    Measured against the image's own clean reference, never against a group mean. The whole
    premise of an internal anchor is that its level is scene-specific, so a drift measured
    against anything but that scene's own starting point would mostly be measuring
    between-scene spread instead.

    `zero_drift_fraction` counts exact zeros rather than near-zeros. A tolerance would be a
    threshold nobody chose on evidence, and the number it is there to expose -- a reference
    that is bit-identical across severities because the dynamic percentile happened to select
    the same population -- is exactly zero when it happens.

    Per-image Spearman comes from `complete_trend_metrics`, so a flat complete curve is flat
    with both correlations zero and an incomplete one is `unmeasured` with both `None`. That
    distinction is the difference between "this anchor holds still", which is the desired
    result, and "this anchor produced no curve", which is not a result at all.
    """
    by_severity: dict[int, dict] = {}
    for severity in EXPECTED_SEVERITIES:
        signed = np.array(
            [curve[severity] - curve[0] for curve in curves.values()], dtype=float
        )
        absolute = np.abs(signed)
        signed_q25, signed_q75 = (float(value) for value in np.percentile(signed, [25, 75]))
        absolute_q25, absolute_q75 = (
            float(value) for value in np.percentile(absolute, [25, 75])
        )
        by_severity[severity] = {
            "median_signed_drift": float(np.median(signed)),
            "median_absolute_drift": float(np.median(absolute)),
            "signed_q25": signed_q25,
            "signed_q75": signed_q75,
            "absolute_q25": absolute_q25,
            "absolute_q75": absolute_q75,
            "zero_drift_fraction": float(np.mean(absolute == 0.0)),
        }

    by_image: dict[int, dict] = {}
    for image_id, curve in curves.items():
        severities = sorted(curve)
        scores = [curve[severity] for severity in severities]
        trend = complete_trend_metrics(severities, scores)
        finite = [value for value in scores if math.isfinite(value)]
        by_image[image_id] = {
            "reference_range": (max(finite) - min(finite)) if finite else None,
            "signed_spearman": trend["signed_spearman"],
            "absolute_spearman": trend["absolute_spearman"],
        }
    return {"by_severity": by_severity, "by_image": by_image}


def between_image_spread(curves: dict[int, dict[int, float]], drift: dict) -> dict:
    """The reference's spread across images at each severity, and drift relative to it.

    `stability_to_spread` is the median absolute within-image drift at this severity divided
    by the *clean* interquartile range. Lower is better: it asks whether the anchor moves less
    under corruption than it differs between scenes, which is the only sense in which
    subtracting it removes more signal than it adds noise. The denominator is severity zero's
    spread rather than this severity's, so the yardstick is fixed and five severities' ratios
    can be compared with each other.

    Severity zero has no ratio. Its drift is identically zero by definition, so a ratio there
    would be a guaranteed 0.0 that reads like a passing score.

    A zero clean interquartile range makes the ratio `None`, never a substituted constant. A
    reference identical across every scene has no scene-specific baseline to explain, and
    dividing by an epsilon would turn that finding into a very large number that looks like a
    measurement of instability instead.
    """
    result: dict[int, dict] = {}
    clean_iqr = None
    for severity in EXPECTED_SEVERITIES:
        values = np.array([curve[severity] for curve in curves.values()], dtype=float)
        statistics = _statistics(values)
        if severity == 0:
            clean_iqr = statistics["iqr"]
            statistics["stability_to_spread"] = None
        elif clean_iqr == 0.0:
            statistics["stability_to_spread"] = None
        else:
            statistics["stability_to_spread"] = (
                drift["by_severity"][severity]["median_absolute_drift"] / clean_iqr
            )
        result[severity] = statistics
    return result


def _correlation(reference: np.ndarray, responsive: np.ndarray) -> tuple:
    """Pearson and Spearman, or `None` when either side is constant.

    SciPy returns `nan` with a `ConstantInputWarning` for a constant input, and a `nan` in
    `summary.json` is not writable JSON. `None` says "undefined here" in a way a reader and a
    serialiser both understand.
    """
    if np.unique(reference).size < 2 or np.unique(responsive).size < 2:
        return None, None
    return float(pearsonr(reference, responsive).statistic), float(
        spearmanr(reference, responsive).statistic
    )


def clean_relationship(
    references: dict[int, float], responsives: dict[int, float], *, folds: dict[int, int]
) -> dict:
    """The normal clean relationship between the two ranges, cross-fitted and challenged.

    Two errors are reported and they are the point of the function. The first is what the
    robust line achieves on images it never saw. The second is what a fold-specific constant
    achieves -- always predicting the other folds' median clean responsive score, ignoring the
    reference entirely. The line earns the word "predictive" only by beating that constant,
    because a line that cannot is not explaining scene-specific baseline; it is reproducing the
    group average with extra steps, and an in-sample correlation of 0.9 can sit on top of
    exactly that.

    Both errors are medians of absolute errors rather than means. One clean scene with an
    extreme baseline would otherwise decide which of the two predictors wins.

    `final_slope` and `final_offset` are fitted on all clean images and are what a deployment
    would store, but they never touch the two error numbers above -- an image must not help
    construct the line that predicts it.
    """
    image_ids = sorted(references)
    reference = np.array([references[image_id] for image_id in image_ids], dtype=float)
    responsive = np.array([responsives[image_id] for image_id in image_ids], dtype=float)
    pearson, spearman = _correlation(reference, responsive)
    final = robust_line(reference, responsive)

    fold_lines: dict[int, tuple[float, float] | None] = {}
    line_errors: list[float] = []
    constant_errors: list[float] = []
    residuals: list[float] = []
    for fold in range(FOLD_COUNT):
        held = [image_id for image_id in image_ids if folds[image_id] == fold]
        trained = [image_id for image_id in image_ids if folds[image_id] != fold]
        if not held:
            continue
        train_reference = np.array([references[i] for i in trained], dtype=float)
        train_responsive = np.array([responsives[i] for i in trained], dtype=float)
        line = robust_line(train_reference, train_responsive) if trained else None
        fold_lines[fold] = line
        constant = float(np.median(train_responsive)) if trained else None
        for image_id in held:
            if constant is not None:
                constant_errors.append(abs(responsives[image_id] - constant))
            if line is not None:
                slope, offset = line
                residual = responsives[image_id] - (offset + slope * references[image_id])
                residuals.append(residual)
                line_errors.append(abs(residual))

    crossfit_error = float(np.median(line_errors)) if line_errors else None
    constant_error = float(np.median(constant_errors)) if constant_errors else None
    residual_array = np.array(residuals, dtype=float)
    residual_median = float(np.median(residual_array)) if residuals else None
    if residuals:
        q25, q75 = (float(value) for value in np.percentile(residual_array, [25, 75]))
        residual_iqr = q75 - q25
        residual_mad = float(np.median(np.abs(residual_array - residual_median)))
    else:
        residual_iqr = residual_mad = None

    return {
        "pearson": pearson,
        "spearman": spearman,
        "final_slope": final[0] if final else None,
        "final_offset": final[1] if final else None,
        "fold_lines": {
            fold: list(line) if line else None for fold, line in fold_lines.items()
        },
        "crossfit_median_absolute_error": crossfit_error,
        "constant_median_absolute_error": constant_error,
        "residual_median": residual_median,
        "residual_iqr": residual_iqr,
        "residual_mad": residual_mad,
        "predictive": bool(
            crossfit_error is not None
            and constant_error is not None
            and crossfit_error < constant_error
        ),
    }
```

- [ ] **Step 4: Run the tests and verify they pass**

```bash
$UE_PY -m pytest tests/scene_uncertainty/test_contrast_diagnostics.py -v
```

Expected: 12 passed.

- [ ] **Step 5: Mutation check**

Break each, confirm the *named* test fails, revert:

1. `within_image_drift` → measure drift against the group's severity-zero mean instead of `curve[0]`. `test_drift_is_measured_against_each_image_own_severity_zero` must fail.
2. `within_image_drift` → return `median_absolute_drift` for both the signed and absolute keys. `test_signed_and_absolute_drift_differ_when_images_move_opposite_ways` must fail.
3. `_statistics` → `np.var(values, ddof=1)`. `test_between_image_spread_reports_the_full_shape` must fail.
4. `between_image_spread` → use this severity's own IQR as the denominator. `test_stability_to_spread_divides_drift_by_the_clean_interquartile_range` must fail.
5. `between_image_spread` → substitute `1e-9` for a zero clean IQR. `test_a_zero_clean_spread_makes_the_ratio_unavailable` must fail.
6. `clean_relationship` → change `crossfit_error < constant_error` to `<=`. `test_a_non_predictive_clean_relationship_is_labelled_so` must still fail? **No** — check it by hand: with `<=` the non-predictive fixture still has `0.125 > 0.05`, so this mutation is *not* caught. Add a third fixture where the two errors are exactly equal and assert `predictive is False`, then re-run.
7. `clean_relationship` → fit each fold's line on all images including the held fold. `test_every_fold_line_excludes_its_own_fold` must fail.

Mutation 6 is deliberate: it is an example of a mutation the first draft of a test suite does not catch, and finding it is the job. Add the equal-error fixture before committing.

- [ ] **Step 6: Commit**

```bash
git add src/scene_uncertainty/contrast_diagnostics.py \
        tests/scene_uncertainty/test_contrast_diagnostics.py
git commit -m "feat: add anchor drift, spread, and clean-relationship diagnostics"
```

---

## Task 4: Contrast rows with cross-fitted residuals, and assembled anchor diagnostics

**Files:**
- Create: `src/scene_uncertainty/contrast_analysis.py`
- Create: `tests/scene_uncertainty/test_contrast_analysis.py`

**Interfaces:**
- Consumes: `contrast_inputs.ARMS`, `ContrastInputs`, `AGGREGATIONS`, `CONFIDENCE_SCOPE`, `PERSISTENCE_SIGNAL`, `CONFIDENCE_SIGNAL` (Task 1); `contrast_scores.SCORE_METHODS`, `assign_folds`, `robust_line`, `contrast_score` (Task 2); `contrast_diagnostics.within_image_drift`, `between_image_spread`, `clean_relationship` (Task 3).
- Produces: `ContrastAnalysisError`, `CONTRAST_ROW_KEY`, `CONTRAST_ROW_FIELDS`, `build_contrast_rows(inputs) -> (rows, fits)`, `build_anchor_diagnostics(inputs) -> list[dict]`. Task 5 consumes `rows`; Tasks 7 and 8 consume `fits` and the diagnostics list.

**Row identity.** `CONTRAST_ROW_KEY = ("image_id", "severity", "arm", "signal", "aggregation", "method")`. For `signal="persistence"` the `arm` field holds the arm name; for `signal="confidence"` it holds the **pair name**, because confidence has one scope and the two differential arms share one twin. Emitting the twin twice under two arm names would double every twin row and make the twin's own orientation depend on which copy a grouping happened to see first.

- [ ] **Step 1: Write the failing tests**

Create `tests/scene_uncertainty/test_contrast_analysis.py`:

```python
import pytest

from src.scene_uncertainty.contrast_analysis import (
    build_anchor_diagnostics,
    build_contrast_rows,
)
from src.scene_uncertainty.contrast_inputs import ARMS, load_contrast_inputs
from src.scene_uncertainty.contrast_scores import SCORE_METHODS, assign_folds

from tests.scene_uncertainty.contrast_test_utils import default_score, write_source_bundle

IMAGES = 6
SEVERITIES = 6


def loaded(tmp_path, **kwargs):
    source = write_source_bundle(tmp_path / "source", **kwargs)
    return load_contrast_inputs(source, expected_image_count=IMAGES)


def crossfit_score(image_id, severity, confidence_bin, signal, scope):
    """A reference/responsive pair whose fold-0 line differs sharply from the final line.

    reference = image_id; responsive = image_id + [0, 5, 5, 5, 0, 0][image_id - 1].
    Fold 0 holds images 1 and 6, so the four training images carry three of the three lifts
    and the fitted slope moves from 1 to 1/6.
    """
    if signal != "persistence" or scope != "layer_2":
        return default_score(image_id, severity, confidence_bin, signal, scope)
    if confidence_bin == "decile_00_10":
        return float(image_id)
    if confidence_bin == "decile_50_60":
        return float(image_id) + [0.0, 5.0, 5.0, 5.0, 0.0, 0.0][image_id - 1]
    return default_score(image_id, severity, confidence_bin, signal, scope)


def index(rows):
    return {
        (row["image_id"], row["severity"], row["arm"], row["signal"],
         row["aggregation"], row["method"]): row
        for row in rows
    }


def test_every_arm_summary_and_method_produces_a_full_grid(tmp_path):
    rows, _ = build_contrast_rows(loaded(tmp_path))
    persistence = [row for row in rows if row["signal"] == "persistence"]
    confidence = [row for row in rows if row["signal"] == "confidence"]
    # three layer_2 arms at four methods plus the combined arm at three: fifteen combinations
    assert len(persistence) == 15 * 3 * IMAGES * SEVERITIES
    assert len(confidence) == 3 * 3 * len(SCORE_METHODS) * IMAGES * SEVERITIES
    assert len(index(rows)) == len(rows)  # no duplicate row keys


def test_the_relative_gap_is_unavailable_at_the_combined_scope(tmp_path):
    """A signed z-score has no symmetric relative gap; the reason is recorded, not silent."""
    rows, fits = build_contrast_rows(loaded(tmp_path))
    combined = {
        row["method"] for row in rows
        if row["arm"] == "decile_90_100__50_60__combined"
    }
    assert combined == {"raw_responsive", "raw_gap", "clean_residual"}
    layered = {
        row["method"] for row in rows if row["arm"] == "decile_90_100__50_60"
        and row["signal"] == "persistence"
    }
    assert layered == set(SCORE_METHODS)
    fit = fits[("decile_90_100__50_60__combined", "persistence", "mean")]
    assert fit["relative_gap_available"] is False
    assert "signed z-score" in fit["relative_gap_unavailable_reason"]
    assert fits[("decile_90_100__50_60", "persistence", "mean")][
        "relative_gap_available"
    ] is True


def test_the_two_differential_arms_share_one_confidence_twin(tmp_path):
    rows, _ = build_contrast_rows(loaded(tmp_path))
    twins = {row["arm"] for row in rows if row["signal"] == "confidence"}
    assert twins == {
        "decile_00_10__50_60", "quintile_00_20__40_60", "decile_90_100__50_60"
    }
    assert "decile_90_100__50_60__combined" not in twins


def test_persistence_rows_carry_their_arm_scope_and_provenance(tmp_path):
    rows, _ = build_contrast_rows(loaded(tmp_path))
    by_key = index(rows)
    combined = by_key[
        (1, 0, "decile_90_100__50_60__combined", "persistence", "mean", "raw_gap")
    ]
    assert combined["score_scope"] == "combined"
    assert combined["arm_family"] == "differential"
    assert combined["declared_before_data"] is False
    anchored = by_key[(1, 0, "decile_00_10__50_60", "persistence", "mean", "raw_gap")]
    assert anchored["score_scope"] == "layer_2"
    assert anchored["arm_family"] == "anchored"
    assert anchored["declared_before_data"] is True


def test_the_two_differential_arms_read_different_scopes(tmp_path):
    rows, _ = build_contrast_rows(loaded(tmp_path))
    by_key = index(rows)
    layer = by_key[(2, 1, "decile_90_100__50_60", "persistence", "q90", "raw_responsive")]
    combined = by_key[
        (2, 1, "decile_90_100__50_60__combined", "persistence", "q90", "raw_responsive")
    ]
    assert layer["score"] != combined["score"]
    assert combined["score"] == pytest.approx(layer["score"] + 0.5)


def test_raw_responsive_and_raw_gap_match_their_inputs(tmp_path):
    rows, _ = build_contrast_rows(loaded(tmp_path))
    by_key = index(rows)
    key = (3, 2, "decile_00_10__50_60", "persistence", "mean")
    control = by_key[(*key, "raw_responsive")]
    gap = by_key[(*key, "raw_gap")]
    assert control["score"] == control["responsive"]
    assert gap["score"] == pytest.approx(gap["responsive"] - gap["reference"])
    assert control["reference"] == gap["reference"]


def test_all_six_severities_of_an_image_share_its_fold(tmp_path):
    rows, _ = build_contrast_rows(loaded(tmp_path))
    folds = assign_folds(range(1, IMAGES + 1))
    for row in rows:
        assert row["fold"] == folds[row["image_id"]]


def test_residual_uses_its_own_fold_line_and_not_the_final_line(tmp_path):
    rows, fits = build_contrast_rows(loaded(tmp_path, score=crossfit_score))
    key = ("decile_00_10__50_60", "persistence", "mean")
    fit = fits[key]
    fold_slope, fold_offset = fit["fold_lines"][0]
    final_slope, final_offset = fit["final_line"]
    assert (fold_slope, fold_offset) != (final_slope, final_offset)

    row = index(rows)[
        (1, 0, "decile_00_10__50_60", "persistence", "mean", "clean_residual")
    ]
    reference, responsive = row["reference"], row["responsive"]
    # image 1 is in fold 0, so its residual is -6.25 from the fold line, not -2.5
    assert row["score"] == pytest.approx(
        responsive - (fold_offset + fold_slope * reference)
    )
    assert row["score"] != pytest.approx(
        responsive - (final_offset + final_slope * reference)
    )
    assert row["fit_slope"] == pytest.approx(fold_slope)
    assert row["fit_offset"] == pytest.approx(fold_offset)


def test_only_severity_zero_rows_influence_a_fit(tmp_path):
    """Changing corrupted severities alone must leave every fitted line untouched."""
    def corrupted_only(image_id, severity, confidence_bin, signal, scope):
        value = default_score(image_id, severity, confidence_bin, signal, scope)
        if severity > 0 and signal == "persistence":
            return value + 7.0
        return value

    _, baseline = build_contrast_rows(loaded(tmp_path / "a"))
    _, shifted = build_contrast_rows(loaded(tmp_path / "b", score=corrupted_only))
    assert baseline.keys() == shifted.keys()
    for key in baseline:
        assert baseline[key]["final_line"] == shifted[key]["final_line"]
        assert baseline[key]["fold_lines"] == shifted[key]["fold_lines"]


def test_twenty_one_final_lines_are_stored(tmp_path):
    _, fits = build_contrast_rows(loaded(tmp_path))
    persistence = [key for key in fits if key[1] == "persistence"]
    confidence = [key for key in fits if key[1] == "confidence"]
    assert len(persistence) == 12  # four arms x three summaries
    assert len(confidence) == 9  # three bucket pairs x three summaries
    assert len(fits) == 21


def test_a_constant_reference_makes_the_residual_unavailable(tmp_path):
    def constant_reference(image_id, severity, confidence_bin, signal, scope):
        if signal == "persistence" and confidence_bin == "decile_00_10":
            return 2.0
        return default_score(image_id, severity, confidence_bin, signal, scope)

    rows, fits = build_contrast_rows(loaded(tmp_path, score=constant_reference))
    fit = fits[("decile_00_10__50_60", "persistence", "mean")]
    assert fit["final_line"] is None
    assert fit["residual_available"] is False
    assert "constant" in fit["unavailable_reason"]
    residuals = [
        row for row in rows
        if row["arm"] == "decile_00_10__50_60"
        and row["signal"] == "persistence"
        and row["method"] == "clean_residual"
    ]
    assert residuals == []
    # the other three methods survive a constant reference
    survivors = {
        row["method"] for row in rows
        if row["arm"] == "decile_00_10__50_60" and row["signal"] == "persistence"
    }
    assert survivors == {"raw_responsive", "raw_gap", "relative_gap"}


def test_anchor_diagnostics_cover_every_arm_and_summary(tmp_path):
    diagnostics = build_anchor_diagnostics(loaded(tmp_path))
    keys = {(row["arm"], row["signal"], row["aggregation"]) for row in diagnostics}
    assert len(keys) == 21
    families = {row["arm"]: row["arm_family"] for row in diagnostics}
    assert families["decile_90_100__50_60__combined"] == "differential"
    assert families["decile_00_10__50_60"] == "anchored"


def test_a_differential_arm_reports_its_anchor_failure_rather_than_hiding_it(tmp_path):
    diagnostics = build_anchor_diagnostics(loaded(tmp_path))
    row = next(
        item for item in diagnostics
        if item["arm"] == "decile_90_100__50_60"
        and item["signal"] == "persistence"
        and item["aggregation"] == "mean"
    )
    # the fixture's 90-100 reference falls with blur, so it drifts and is not an anchor
    assert row["arm_family"] == "differential"
    assert row["drift"]["by_severity"][5]["median_absolute_drift"] > 0.0
    assert row["spread"][5]["stability_to_spread"] is not None
```

- [ ] **Step 2: Run the tests and read the failures**

```bash
$UE_PY -m pytest tests/scene_uncertainty/test_contrast_analysis.py -v
```

Expected: collection error — module does not exist.

- [ ] **Step 3: Implement row construction**

Create `src/scene_uncertainty/contrast_analysis.py`:

```python
"""Source scores turned into contrast rows, one per image, severity, arm, summary and method.

This is where the experiment's shape lives: which arms exist, which signals each produces, and
which fold's line each residual is allowed to see. Everything numeric it does is a call into
`contrast_scores` or `contrast_diagnostics`; everything structural it does is here and nowhere
else, so a change to the arm table changes one file.
"""
from __future__ import annotations

from .contrast_diagnostics import between_image_spread, clean_relationship, within_image_drift
from .contrast_inputs import (
    AGGREGATIONS,
    ARMS,
    COMBINED_SCOPE,
    CONFIDENCE_SCOPE,
    CONFIDENCE_SIGNAL,
    EXPECTED_SEVERITIES,
    PERSISTENCE_SIGNAL,
    ContrastInputs,
)
from .contrast_scores import FOLD_COUNT, SCORE_METHODS, assign_folds, contrast_score, robust_line


class ContrastAnalysisError(ValueError):
    """A source that loaded cleanly but cannot be turned into the declared candidates."""


CONTRAST_ROW_KEY = ("image_id", "severity", "arm", "signal", "aggregation", "method")
"""What makes two contrast rows measurements of different things.

`arm` carries an arm name on a persistence row and a *pair* name on a confidence row, and that
asymmetry is deliberate rather than sloppy: confidence has no decoder layer, so the `layer_2`
and `combined` differential arms have one twin between them. Keying the twin by arm would
produce two identical copies whose separately-locked orientations could disagree by nothing but
which copy a grouping saw first.
"""

CONTRAST_ROW_FIELDS = (
    *CONTRAST_ROW_KEY,
    "arm_family", "declared_before_data", "score_scope",
    "reference_bin", "responsive_bin", "reference", "responsive", "score",
    "fold", "fit_slope", "fit_offset",
)


def _signal_plan(arm) -> tuple[tuple[str, str, str], ...]:
    """The `(row arm label, signal, source scope)` entries one arm contributes.

    Two for most arms: its persistence rows under its own name and scope, and its confidence
    twin under the pair name. The `combined` differential arm contributes only the persistence
    entry, because its twin is already produced by the `layer_2` arm that shares its pair.
    """
    entries = [(arm.name, PERSISTENCE_SIGNAL, arm.score_scope)]
    first_with_pair = next(other for other in ARMS if other.pair_name == arm.pair_name)
    if first_with_pair.name == arm.name:
        entries.append((arm.pair_name, CONFIDENCE_SIGNAL, CONFIDENCE_SCOPE))
    return tuple(entries)


def _curve(inputs: ContrastInputs, confidence_bin, aggregation, scope, signal) -> dict:
    """`{image_id: {severity: score}}` for one source series."""
    return {
        image_id: {
            severity: inputs.scores[
                (image_id, severity, signal, confidence_bin, aggregation, scope)
            ]
            for severity in EXPECTED_SEVERITIES
        }
        for image_id in inputs.image_ids
    }


def _fit(references: dict, responsives: dict, folds: dict, scope: str) -> dict:
    """The final all-clean line, the five fold lines, and whether the residual is usable.

    The final line is what a deployment would store and is never used to score a tuning image;
    the fold lines are what every reported residual is built from. Both are kept because the
    figures show the first and the numbers come from the second, and a bundle that published
    only one of them would leave a reader unable to tell which the picture was drawn from.
    """
    clean_reference = {image: curve[0] for image, curve in references.items()}
    clean_responsive = {image: curve[0] for image, curve in responsives.items()}
    image_ids = sorted(clean_reference)
    final = robust_line(
        [clean_reference[image] for image in image_ids],
        [clean_responsive[image] for image in image_ids],
    )
    fold_lines: dict[int, tuple[float, float] | None] = {}
    for fold in range(FOLD_COUNT):
        trained = [image for image in image_ids if folds[image] != fold]
        held = [image for image in image_ids if folds[image] == fold]
        if not held:
            continue
        fold_lines[fold] = robust_line(
            [clean_reference[image] for image in trained],
            [clean_responsive[image] for image in trained],
        )
    unavailable = [fold for fold, line in fold_lines.items() if line is None]
    if final is None:
        reason = "the clean reference is constant, so no line can be fitted"
    elif unavailable:
        reason = f"folds {sorted(unavailable)} have a constant clean reference"
    else:
        reason = None
    return {
        "final_line": list(final) if final else None,
        "fold_lines": {fold: list(line) if line else None
                       for fold, line in fold_lines.items()},
        "residual_available": final is not None and not unavailable,
        "unavailable_reason": reason,
        "relative_gap_available": scope != COMBINED_SCOPE,
        "relative_gap_unavailable_reason": (
            None if scope != COMBINED_SCOPE else
            "the combined scope is a signed z-score, and the symmetric relative gap's "
            "scale invariance and bounds both require non-negative inputs"
        ),
    }


def build_contrast_rows(inputs: ContrastInputs) -> tuple[list[dict], dict]:
    """Every contrast row the four arms declare, and the 21 fits behind their residuals.

    Returns `(rows, fits)` where `fits` is keyed `(arm-or-pair, signal, aggregation)`. Residual
    rows are omitted entirely when the fit is unavailable rather than written with a `None`
    score: a candidate missing its residual is not a candidate with a gap in it, and the
    recorded `unavailable_reason` is what a reader gets instead of a column of blanks.
    """
    folds = assign_folds(inputs.image_ids)
    rows: list[dict] = []
    fits: dict[tuple, dict] = {}
    for arm in ARMS:
        for label, signal, scope in _signal_plan(arm):
            for aggregation in AGGREGATIONS:
                references = _curve(inputs, arm.reference_bin, aggregation, scope, signal)
                responsives = _curve(inputs, arm.responsive_bin, aggregation, scope, signal)
                fit = _fit(references, responsives, folds, scope)
                fits[(label, signal, aggregation)] = fit
                for method in SCORE_METHODS:
                    if method == "clean_residual" and not fit["residual_available"]:
                        continue
                    if method == "relative_gap" and scope == COMBINED_SCOPE:
                        continue
                    for image_id in inputs.image_ids:
                        fold = folds[image_id]
                        line = fit["fold_lines"].get(fold)
                        for severity in EXPECTED_SEVERITIES:
                            reference = references[image_id][severity]
                            responsive = responsives[image_id][severity]
                            rows.append({
                                "image_id": image_id, "severity": severity, "arm": label,
                                "signal": signal, "aggregation": aggregation,
                                "method": method, "arm_family": arm.family,
                                "declared_before_data": arm.declared_before_data,
                                "score_scope": scope,
                                "reference_bin": arm.reference_bin,
                                "responsive_bin": arm.responsive_bin,
                                "reference": reference, "responsive": responsive,
                                "score": contrast_score(
                                    method, reference, responsive,
                                    line=tuple(line) if line else None,
                                ),
                                "fold": fold,
                                "fit_slope": line[0] if (
                                    method == "clean_residual" and line
                                ) else None,
                                "fit_offset": line[1] if (
                                    method == "clean_residual" and line
                                ) else None,
                            })
    if not rows:
        raise ContrastAnalysisError("no contrast rows were produced from this source")
    return rows, fits


def build_anchor_diagnostics(inputs: ContrastInputs) -> list[dict]:
    """Drift, spread and clean relationship for all 21 arm-summary-signal configurations.

    All four arms are measured, including the two whose reference is not an anchor. Reporting
    only the arms expected to pass would leave a reader unable to see how far a differential
    reference moves, which is the number that says the two families are different experiments
    rather than one experiment with a weak member.
    """
    folds = assign_folds(inputs.image_ids)
    diagnostics: list[dict] = []
    for arm in ARMS:
        for label, signal, scope in _signal_plan(arm):
            for aggregation in AGGREGATIONS:
                references = _curve(inputs, arm.reference_bin, aggregation, scope, signal)
                responsives = _curve(inputs, arm.responsive_bin, aggregation, scope, signal)
                drift = within_image_drift(references)
                diagnostics.append({
                    "arm": label,
                    "arm_family": arm.family,
                    "declared_before_data": arm.declared_before_data,
                    "signal": signal,
                    "aggregation": aggregation,
                    "score_scope": scope,
                    "reference_bin": arm.reference_bin,
                    "responsive_bin": arm.responsive_bin,
                    "drift": drift,
                    "spread": between_image_spread(references, drift),
                    "relationship": clean_relationship(
                        {image: curve[0] for image, curve in references.items()},
                        {image: curve[0] for image, curve in responsives.items()},
                        folds=folds,
                    ),
                })
    return diagnostics
```

- [ ] **Step 4: Run the tests and verify they pass**

```bash
$UE_PY -m pytest tests/scene_uncertainty/test_contrast_analysis.py -v
```

Expected: 13 passed.

- [ ] **Step 5: Mutation check**

Break each, confirm the *named* test fails, revert:

0. `build_contrast_rows` → drop the `relative_gap` / `COMBINED_SCOPE` skip. `test_the_relative_gap_is_unavailable_at_the_combined_scope` must fail, and so must the row count in `test_every_arm_summary_and_method_produces_a_full_grid`.
1. `_signal_plan` → always append the confidence entry. `test_the_two_differential_arms_share_one_confidence_twin` must fail (twin rows double).
2. `_fit` → fit fold lines on all images instead of `trained`. `test_residual_uses_its_own_fold_line_and_not_the_final_line` must fail.
3. `_fit` → use `curve[severity]` for every severity rather than `curve[0]`. `test_only_severity_zero_rows_influence_a_fit` must fail.
4. `build_contrast_rows` → pass `fit["final_line"]` as `line` instead of the fold's. `test_residual_uses_its_own_fold_line_and_not_the_final_line` must fail.
5. `build_contrast_rows` → emit residual rows with `score=None` when unavailable instead of skipping. `test_a_constant_reference_makes_the_residual_unavailable` must fail.
6. `assign_folds` call → replace with `{image: 0 for image in ...}`. `test_all_six_severities_of_an_image_share_its_fold` must still pass (all rows agree with the mutated map) — **this mutation is not caught.** Add an assertion that the fold values across all images are `{0, 1, 2, 3, 4}`, then re-run.

Mutation 6 is deliberate. `test_all_six_severities_of_an_image_share_its_fold` compares the rows against the same function the production code calls, so it proves consistency and not correctness. Add the distinctness assertion before committing.

- [ ] **Step 6: Commit**

```bash
git add src/scene_uncertainty/contrast_analysis.py \
        tests/scene_uncertainty/test_contrast_analysis.py
git commit -m "feat: build contrast rows with cross-fitted residuals and anchor diagnostics"
```

---

## Task 5: Per-candidate trends, locked orientation, and per-severity AUROC

**Files:**
- Modify: `src/scene_uncertainty/contrast_analysis.py`
- Modify: `tests/scene_uncertainty/test_contrast_analysis.py`

**Interfaces:**
- Consumes: `CONTRAST_ROW_KEY`, the rows from `build_contrast_rows` (Task 4); `corruption_metrics.complete_trend_metrics`, `choose_orientation`, `oriented_curve_metrics`, `severity_aurocs` (existing, unchanged).
- Produces: `CONTRAST_CANDIDATE_KEY`, `DEPLOYABLE_SIGNAL`, `summarize_contrast_candidates(rows, *, expected_image_count) -> list[dict]`. Task 6 attaches controls to these dictionaries and ranks them; Task 7 plots them; Task 8 writes them.

**No `deployable` flag and no ranking here.** The spec's eligibility list includes "a computable confidence-only twin", which this task cannot see. Task 6 owns the gate and the order. This task owns everything measurable from one candidate's own rows.

- [ ] **Step 1: Write the failing tests (append to `test_contrast_analysis.py`)**

```python
from src.scene_uncertainty.contrast_analysis import (
    CONTRAST_CANDIDATE_KEY,
    summarize_contrast_candidates,
)


def candidate_rows(curves, *, arm="decile_00_10__50_60", signal="persistence",
                   aggregation="mean", method="raw_gap"):
    """Rows for a single candidate from `{image_id: [six scores]}`."""
    return [
        {
            "image_id": image_id, "severity": severity, "arm": arm, "signal": signal,
            "aggregation": aggregation, "method": method, "arm_family": "anchored",
            "declared_before_data": True, "score_scope": "layer_2",
            "reference_bin": "decile_00_10", "responsive_bin": "decile_50_60",
            "reference": 1.0, "responsive": 1.0 + score, "score": score,
            "fold": image_id % 5, "fit_slope": None, "fit_offset": None,
        }
        for image_id, scores in curves.items()
        for severity, score in enumerate(scores)
    ]


def test_candidate_key_is_arm_signal_aggregation_method():
    assert CONTRAST_CANDIDATE_KEY == ("arm", "signal", "aggregation", "method")


def test_every_declared_candidate_appears(tmp_path):
    rows, _ = build_contrast_rows(loaded(tmp_path))
    candidates = summarize_contrast_candidates(rows, expected_image_count=IMAGES)
    persistence = [item for item in candidates if item["signal"] == "persistence"]
    confidence = [item for item in candidates if item["signal"] == "confidence"]
    # 45, not 48: the combined arm has no relative gap (signed z-score)
    assert len(persistence) == 45
    assert len(confidence) == 36
    assert len(candidates) == 81
    assert not [
        item for item in persistence
        if item["arm"].endswith("__combined") and item["method"] == "relative_gap"
    ]


def test_a_rising_candidate_locks_to_plus_one():
    rows = candidate_rows({image: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] for image in range(1, 5)})
    candidate = summarize_contrast_candidates(rows, expected_image_count=4)[0]
    assert candidate["median_signed_spearman"] == pytest.approx(1.0)
    assert candidate["orientation"] == 1
    assert candidate["macro_auroc"] == pytest.approx(1.0)


def test_a_falling_candidate_locks_to_minus_one_and_is_read_that_way():
    rows = candidate_rows({image: [5.0, 4.0, 3.0, 2.0, 1.0, 0.0] for image in range(1, 5)})
    candidate = summarize_contrast_candidates(rows, expected_image_count=4)[0]
    assert candidate["median_signed_spearman"] == pytest.approx(-1.0)
    assert candidate["orientation"] == -1
    # read in its locked direction a falling candidate separates perfectly, not at chance
    assert candidate["macro_auroc"] == pytest.approx(1.0)
    assert candidate["auroc_by_severity"][5] == pytest.approx(1.0)


def test_a_candidate_with_no_agreed_direction_is_unorientable():
    rising = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    falling = list(reversed(rising))
    rows = candidate_rows({1: rising, 2: rising, 3: falling, 4: falling})
    candidate = summarize_contrast_candidates(rows, expected_image_count=4)[0]
    assert candidate["median_signed_spearman"] == pytest.approx(0.0)
    assert candidate["orientation"] is None
    assert candidate["macro_auroc"] is None
    assert candidate["orientable"] is False


def test_a_flat_curve_is_flat_and_an_incomplete_one_is_unmeasured():
    flat = [2.0] * 6
    rows = candidate_rows({1: flat, 2: flat, 3: flat, 4: flat})
    candidate = summarize_contrast_candidates(rows, expected_image_count=4)[0]
    assert candidate["flat_count"] == 4
    assert candidate["measured_count"] == 4
    assert candidate["missing_count"] == 0
    assert candidate["median_absolute_spearman"] == pytest.approx(0.0)

    partial = candidate_rows({1: flat, 2: flat})
    partial = [row for row in partial if not (row["image_id"] == 2 and row["severity"] == 3)]
    incomplete = summarize_contrast_candidates(partial, expected_image_count=2)[0]
    assert incomplete["measured_count"] == 1
    assert incomplete["missing_count"] == 1
    assert incomplete["complete"] is False


def test_per_severity_auroc_is_computed_severity_by_severity():
    rows = candidate_rows({
        1: [0.0, 0.5, 1.0, 1.0, 1.0, 10.0],
        2: [1.0, 1.5, 2.0, 2.0, 2.0, 11.0],
        3: [2.0, 2.5, 3.0, 3.0, 3.0, 12.0],
        4: [3.0, 3.5, 4.0, 4.0, 4.0, 13.0],
    })
    candidate = summarize_contrast_candidates(rows, expected_image_count=4)[0]
    # 10 of the 16 clean/corrupted pairs separate at severity 1; all 16 at severity 5
    assert candidate["auroc_by_severity"][1] == pytest.approx(0.625)
    assert candidate["auroc_by_severity"][5] == pytest.approx(1.0)
    assert candidate["macro_auroc"] == pytest.approx(
        sum(candidate["auroc_by_severity"][severity] for severity in range(1, 6)) / 5
    )


def test_macro_auroc_is_never_a_stand_in_for_severity_one():
    """A candidate strong only at severe blur must show it, not hide behind the mean."""
    rows = candidate_rows({
        1: [0.0, 0.0, 0.0, 0.0, 0.0, 9.0],
        2: [1.0, 1.0, 1.0, 1.0, 1.0, 9.5],
        3: [2.0, 2.0, 2.0, 2.0, 2.0, 10.0],
        4: [3.0, 3.0, 3.0, 3.0, 3.0, 10.5],
    })
    candidate = summarize_contrast_candidates(rows, expected_image_count=4)[0]
    assert candidate["auroc_by_severity"][1] == pytest.approx(0.5)
    assert candidate["auroc_by_severity"][5] == pytest.approx(1.0)
    assert candidate["macro_auroc"] > candidate["auroc_by_severity"][1]


def test_severity_statistics_are_published_for_the_plots():
    rows = candidate_rows({image: [float(image)] * 6 for image in range(1, 5)})
    candidate = summarize_contrast_candidates(rows, expected_image_count=4)[0]
    clean = candidate["severity_statistics"][0]
    assert clean["count"] == 4
    assert clean["mean"] == pytest.approx(2.5)
    assert clean["median"] == pytest.approx(2.5)
    assert clean["q25"] == pytest.approx(1.75)
    assert clean["q75"] == pytest.approx(3.25)


def test_candidate_provenance_survives_from_the_rows(tmp_path):
    rows, _ = build_contrast_rows(loaded(tmp_path))
    candidates = summarize_contrast_candidates(rows, expected_image_count=IMAGES)
    combined = next(
        item for item in candidates
        if item["arm"] == "decile_90_100__50_60__combined"
        and item["method"] == "raw_gap" and item["aggregation"] == "mean"
    )
    assert combined["arm_family"] == "differential"
    assert combined["declared_before_data"] is False
    assert combined["score_scope"] == "combined"


def test_a_duplicated_row_key_is_refused():
    rows = candidate_rows({1: [0.0] * 6})
    with pytest.raises(ContrastAnalysisError, match="duplicate"):
        summarize_contrast_candidates(rows + rows[:1], expected_image_count=1)
```

Add `ContrastAnalysisError` to the module's import line at the top of the test file.

- [ ] **Step 2: Run the tests and read the failures**

```bash
$UE_PY -m pytest tests/scene_uncertainty/test_contrast_analysis.py -v -k "candidate or orient or auroc or severity_statistics or duplicated"
```

Expected: `ImportError: cannot import name 'summarize_contrast_candidates'`.

- [ ] **Step 3: Implement candidate summarisation (append to `contrast_analysis.py`)**

```python
import numpy as np

from .corruption_metrics import (
    choose_orientation,
    complete_trend_metrics,
    oriented_curve_metrics,
    severity_aurocs,
)

CONTRAST_CANDIDATE_KEY = ("arm", "signal", "aggregation", "method")
"""A row key with the scene coordinates taken out.

Note what stays separate: a persistence candidate and its confidence twin differ in `signal`,
so they are two candidates with two independently chosen orientations. That independence is
the whole point of the twin -- a twin forced to share its candidate's orientation would be
measuring how well the candidate's direction happens to suit confidence, not how well
confidence does on its own terms.
"""

DEPLOYABLE_SIGNAL = PERSISTENCE_SIGNAL
"""Only persistence candidates are ranked. Confidence twins are summarised in full and
published in full; it is the ranking they stay out of, because a control that can win the
comparison it exists to lose is not a control."""

_CANDIDATE_PROVENANCE = (
    "arm_family", "declared_before_data", "score_scope", "reference_bin", "responsive_bin",
)


def _severity_statistics(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"count": 0, "mean": None, "variance": None,
                "median": None, "q25": None, "q75": None}
    q25, median, q75 = (float(value) for value in np.percentile(values, [25, 50, 75]))
    return {
        "count": int(values.size), "mean": float(np.mean(values)),
        "variance": float(np.var(values)), "median": median, "q25": q25, "q75": q75,
    }


def summarize_contrast_candidates(
    rows: list[dict], *, expected_image_count: int
) -> list[dict]:
    """One dictionary per candidate: its trend, its one locked orientation, and its five AUROCs.

    Orientation is chosen from the median signed Spearman of this candidate's own tuning
    images, once, before any AUROC is computed -- so no direction can be picked because it
    scored better. An unorientable candidate gets `orientation=None` and `macro_auroc=None`
    rather than a defaulted `+1`: "these images do not agree which way this moves" is a
    finding, and a defaulted direction would publish it as a measurement of roughly 0.5.

    `expected_image_count` is an argument rather than a count of the rows, for the reason the
    corruption-sensitivity command gives: the rows cannot tell a run of 240 images apart from a
    run of 250 that lost ten before scoring, and only the first of those is a smaller run.
    """
    index: dict[tuple, dict] = {}
    provenance: dict[tuple, dict] = {}
    seen: set[tuple] = set()
    for position, row in enumerate(rows):
        missing = [field for field in CONTRAST_ROW_KEY if field not in row]
        if missing:
            raise ContrastAnalysisError(f"contrast row {position} is missing {missing}")
        key = tuple(row[field] for field in CONTRAST_ROW_KEY)
        if key in seen:
            raise ContrastAnalysisError(f"duplicate contrast row key: {key}")
        seen.add(key)
        candidate_key = tuple(row[field] for field in CONTRAST_CANDIDATE_KEY)
        index.setdefault(candidate_key, {}).setdefault(row["image_id"], {})[
            row["severity"]
        ] = row["score"]
        provenance.setdefault(
            candidate_key, {field: row.get(field) for field in _CANDIDATE_PROVENANCE}
        )

    candidates: list[dict] = []
    for candidate_key in sorted(index, key=lambda key: tuple(str(part) for part in key)):
        by_image = index[candidate_key]
        trends = {}
        for image_id, curve in by_image.items():
            severities = sorted(curve)
            trends[image_id] = complete_trend_metrics(
                severities, [curve[severity] for severity in severities]
            )
        signed = [
            trend["signed_spearman"] for trend in trends.values()
            if trend["signed_spearman"] is not None
        ]
        absolute = [
            trend["absolute_spearman"] for trend in trends.values()
            if trend["absolute_spearman"] is not None
        ]
        directions = [trend["direction"] for trend in trends.values()]
        orientation = choose_orientation(signed)
        measured = sum(1 for trend in trends.values() if trend["fully_measured"])
        complete = measured == len(by_image) == expected_image_count

        oriented = [
            oriented_curve_metrics(
                [by_image[image_id][severity] for severity in sorted(by_image[image_id])],
                orientation,
            )
            for image_id in by_image
            if orientation in (-1, 1) and trends[image_id]["fully_measured"]
        ]

        auroc_by_severity = macro = None
        if orientation in (-1, 1) and complete:
            scores_by_severity = {
                severity: [by_image[image_id][severity] for image_id in sorted(by_image)]
                for severity in EXPECTED_SEVERITIES
            }
            auroc_by_severity, macro = severity_aurocs(scores_by_severity, orientation)

        candidate = dict(zip(CONTRAST_CANDIDATE_KEY, candidate_key))
        candidate.update(provenance[candidate_key])
        candidate.update({
            "expected_image_count": expected_image_count,
            "image_count": len(by_image),
            "measured_count": measured,
            "missing_count": len(by_image) - measured,
            "positive_count": directions.count("increasing"),
            "negative_count": directions.count("decreasing"),
            "flat_count": directions.count("flat"),
            "median_signed_spearman": float(np.median(signed)) if signed else None,
            "median_absolute_spearman": float(np.median(absolute)) if absolute else None,
            "orientation": orientation,
            "orientable": orientation in (-1, 1),
            "complete": complete,
            "oriented_adjacent_consistency": (
                float(np.mean([item["adjacent_consistency"] for item in oriented]))
                if oriented else None
            ),
            "max_blur_above_clean_rate": (
                float(np.mean([item["max_blur_above_clean"] for item in oriented]))
                if oriented else None
            ),
            "auroc_by_severity": auroc_by_severity,
            "macro_auroc": macro,
            "severity_statistics": {
                severity: _severity_statistics(
                    np.array(
                        [
                            by_image[image_id][severity]
                            for image_id in by_image
                            if severity in by_image[image_id]
                        ],
                        dtype=float,
                    )
                )
                for severity in EXPECTED_SEVERITIES
            },
        })
        total = len(by_image) or 1
        candidate["measured_fraction"] = measured / total
        candidate["missing_fraction"] = candidate["missing_count"] / total
        candidate["positive_fraction"] = candidate["positive_count"] / total
        candidate["negative_fraction"] = candidate["negative_count"] / total
        candidate["flat_fraction"] = candidate["flat_count"] / total
        candidate["dominant_direction_fraction"] = max(
            candidate["positive_fraction"],
            candidate["negative_fraction"],
            candidate["flat_fraction"],
        )
        candidates.append(candidate)
    return candidates
```

- [ ] **Step 4: Run the tests and verify they pass**

```bash
$UE_PY -m pytest tests/scene_uncertainty/test_contrast_analysis.py -v
```

Expected: 22 passed.

- [ ] **Step 5: Mutation check**

Break each, confirm the *named* test fails, revert:

1. `choose_orientation(signed)` → hard-code `1`. `test_a_falling_candidate_locks_to_minus_one_and_is_read_that_way` must fail (a falling candidate read upward gives macro 0.0, not 1.0), and `test_a_candidate_with_no_agreed_direction_is_unorientable` must fail.
2. Compute orientation from the AUROC that scores better instead of from the median Spearman. Both tests above must fail.
3. `severity_aurocs(..., orientation)` → pass `1` instead of `orientation`. `test_a_falling_candidate_locks_to_minus_one_and_is_read_that_way` must fail.
4. `macro` → return `auroc_by_severity[5]` instead of the mean. `test_per_severity_auroc_is_computed_severity_by_severity` must fail.
5. `median_absolute_spearman` → return `abs(median_signed_spearman)`. Check by hand against `test_a_candidate_with_no_agreed_direction_is_unorientable`: its median signed is 0.0 and its median absolute is 1.0, so add an assertion there that `median_absolute_spearman == 1.0` — **the current test does not catch this mutation.** Add the assertion, then re-run.
6. `complete` → drop the `== expected_image_count` clause. `test_a_flat_curve_is_flat_and_an_incomplete_one_is_unmeasured` must fail.
7. `dominant_direction_fraction` → divide by `measured` instead of `total`. Add a fixture with one unmeasured image and assert the flat images do not become a larger fraction of a smaller denominator.

Mutations 5 and 7 are deliberate gaps in the first draft. Close them before committing.

- [ ] **Step 6: Commit**

```bash
git add src/scene_uncertainty/contrast_analysis.py \
        tests/scene_uncertainty/test_contrast_analysis.py
git commit -m "feat: summarize contrast candidates with locked orientation and AUROC"
```

---

## Task 6: Raw reference control, confidence-only twin, paired bootstrap, and ranking

**Files:**
- Create: `src/scene_uncertainty/contrast_controls.py`
- Create: `tests/scene_uncertainty/test_contrast_controls.py`

**Interfaces:**
- Consumes: `contrast_inputs.ARMS`, `FULL_TUNING_IMAGE_COUNT` (Task 1); `contrast_analysis.CONTRAST_CANDIDATE_KEY`, `DEPLOYABLE_SIGNAL`, `summarize_contrast_candidates` (Task 5); `corruption_metrics.EXPECTED_SEVERITIES` (existing).
- Produces: `BOOTSTRAP_SEED`, `BOOTSTRAP_SAMPLES`, `reference_control_rows(rows)`, `attach_controls(candidates, controls, rows)`, `paired_macro_bootstrap(...)`, `rank_contrast_candidates(candidates)`. Task 8 writes every field this attaches; Task 7 draws the twin AUROCs.

**Why the reference control reuses the candidate machinery.** The raw reference range needs its own locked orientation, its own five AUROCs and its own coverage gate — exactly what `summarize_contrast_candidates` already computes. Re-implementing that here would be a second place for orientation to be chosen, and the two would drift.

- [ ] **Step 1: Write the failing tests**

Create `tests/scene_uncertainty/test_contrast_controls.py`:

```python
import numpy as np
import pytest

from src.scene_uncertainty.contrast_analysis import (
    build_contrast_rows,
    summarize_contrast_candidates,
)
from src.scene_uncertainty.contrast_controls import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    attach_controls,
    index_curves,
    paired_macro_bootstrap,
    rank_contrast_candidates,
    reference_control_rows,
)
from src.scene_uncertainty.contrast_inputs import FULL_TUNING_IMAGE_COUNT, load_contrast_inputs

from tests.scene_uncertainty.contrast_test_utils import write_source_bundle

IMAGES = 6


def loaded(tmp_path, **kwargs):
    source = write_source_bundle(tmp_path / "source", **kwargs)
    return load_contrast_inputs(source, expected_image_count=IMAGES)


def prepared(tmp_path, **kwargs):
    rows, _ = build_contrast_rows(loaded(tmp_path, **kwargs))
    candidates = summarize_contrast_candidates(rows, expected_image_count=IMAGES)
    controls = summarize_contrast_candidates(
        reference_control_rows(rows), expected_image_count=IMAGES
    )
    attach_controls(candidates, controls, rows)
    return candidates, controls, rows


def stub(**overrides):
    """A minimal candidate dictionary for ranking tests."""
    candidate = {
        "arm": "decile_00_10__50_60", "signal": "persistence", "aggregation": "mean",
        "method": "raw_gap", "macro_auroc": 0.7,
        "auroc_by_severity": {1: 0.6, 2: 0.6, 3: 0.7, 4: 0.8, 5: 0.8},
        "median_absolute_spearman": 0.5, "dominant_direction_fraction": 0.9,
        "oriented_adjacent_consistency": 0.8, "orientable": True, "complete": True,
        "expected_image_count": FULL_TUNING_IMAGE_COUNT, "twin_macro_auroc": 0.5,
    }
    candidate.update(overrides)
    return candidate


def test_the_seed_and_sample_count_are_the_declared_ones():
    assert BOOTSTRAP_SEED == 20260821
    assert BOOTSTRAP_SAMPLES == 2000


def test_a_reference_control_exists_for_every_arm_summary_and_signal(tmp_path):
    _, controls, _ = prepared(tmp_path)
    keys = {(item["arm"], item["signal"], item["aggregation"]) for item in controls}
    assert len(keys) == 21
    assert {item["method"] for item in controls} == {"raw_reference"}


def test_the_reference_control_scores_the_reference_range_itself(tmp_path):
    rows, _ = build_contrast_rows(loaded(tmp_path))
    control_rows = reference_control_rows(rows)
    assert len(control_rows) == len(
        [row for row in rows if row["method"] == "raw_responsive"]
    )
    for row in control_rows:
        assert row["score"] == row["reference"]
        assert row["method"] == "raw_reference"


def test_the_reference_control_locks_its_own_orientation():
    """A reference that falls while the responsive rises must lock to -1, not inherit +1."""
    rows = []
    for image_id in range(1, 5):
        for severity in range(6):
            rows.append({
                "image_id": image_id, "severity": severity,
                "arm": "a", "signal": "persistence", "aggregation": "mean",
                "method": "raw_responsive", "arm_family": "differential",
                "declared_before_data": False, "score_scope": "layer_2",
                "reference_bin": "r", "responsive_bin": "p",
                "reference": 10.0 + image_id - severity,
                "responsive": float(image_id) + severity,
                "score": float(image_id) + severity,
                "fold": image_id % 5, "fit_slope": None, "fit_offset": None,
            })
    candidate = summarize_contrast_candidates(rows, expected_image_count=4)[0]
    control = summarize_contrast_candidates(
        reference_control_rows(rows), expected_image_count=4
    )[0]
    assert candidate["orientation"] == 1
    assert control["orientation"] == -1


def test_both_differential_arms_resolve_to_the_same_confidence_twin(tmp_path):
    candidates, _, _ = prepared(tmp_path)
    layer = next(
        item for item in candidates
        if item["arm"] == "decile_90_100__50_60"
        and item["signal"] == "persistence"
        and item["method"] == "raw_gap" and item["aggregation"] == "mean"
    )
    combined = next(
        item for item in candidates
        if item["arm"] == "decile_90_100__50_60__combined"
        and item["signal"] == "persistence"
        and item["method"] == "raw_gap" and item["aggregation"] == "mean"
    )
    assert layer["twin_arm"] == combined["twin_arm"] == "decile_90_100__50_60"
    assert layer["twin_macro_auroc"] == combined["twin_macro_auroc"]


def test_a_candidate_that_only_restates_confidence_is_flagged_redundant(tmp_path):
    candidates, _, _ = prepared(tmp_path)
    for candidate in candidates:
        if candidate["signal"] != "persistence" or candidate["macro_auroc"] is None:
            continue
        expected = not (candidate["macro_auroc"] > candidate["twin_macro_auroc"])
        assert candidate["confidence_redundant"] is expected


def test_a_candidate_is_compared_with_both_of_its_inputs(tmp_path):
    candidates, _, _ = prepared(tmp_path)
    candidate = next(
        item for item in candidates
        if item["signal"] == "persistence" and item["macro_auroc"] is not None
    )
    assert candidate["responsive_control_macro_difference"] == pytest.approx(
        candidate["macro_auroc"] - candidate["responsive_control_macro_auroc"]
    )
    assert candidate["reference_control_macro_difference"] == pytest.approx(
        candidate["macro_auroc"] - candidate["reference_control_macro_auroc"]
    )
    assert candidate["beats_both_inputs"] is (
        candidate["responsive_control_macro_difference"] > 0
        and candidate["reference_control_macro_difference"] > 0
    )


def test_beating_one_input_while_losing_to_the_other_is_not_beating_both():
    candidate = stub(
        macro_auroc=0.70,
        responsive_control_macro_auroc=0.60,
        reference_control_macro_auroc=0.85,
    )
    candidate["responsive_control_macro_difference"] = 0.10
    candidate["reference_control_macro_difference"] = -0.15
    from src.scene_uncertainty.contrast_controls import beats_both_inputs

    assert beats_both_inputs(candidate) is False


def test_the_paired_bootstrap_is_deterministic_under_its_seed():
    rng_images = list(range(1, 21))
    candidate = {image: {severity: image + severity for severity in range(6)}
                 for image in rng_images}
    control = {image: {severity: image + 0.5 * severity for severity in range(6)}
               for image in rng_images}
    first = paired_macro_bootstrap(
        rng_images, candidate, control,
        candidate_orientation=1, control_orientation=1, samples=200,
    )
    second = paired_macro_bootstrap(
        rng_images, candidate, control,
        candidate_orientation=1, control_orientation=1, samples=200,
    )
    assert first == second
    assert first["low"] <= first["macro_difference"] <= first["high"]


def test_the_paired_bootstrap_resamples_both_methods_identically():
    """A method compared with itself must give an interval of exactly zero width."""
    images = list(range(1, 21))
    scores = {image: {severity: image + severity for severity in range(6)}
              for image in images}
    result = paired_macro_bootstrap(
        images, scores, scores,
        candidate_orientation=1, control_orientation=1, samples=200,
    )
    assert result["macro_difference"] == pytest.approx(0.0)
    assert result["low"] == pytest.approx(0.0)
    assert result["high"] == pytest.approx(0.0)
    assert result["verdict"] == "not supported on tuning"


def test_a_positive_interval_above_zero_is_supported_on_tuning():
    images = list(range(1, 41))
    strong = {image: {severity: float(severity) for severity in range(6)}
              for image in images}
    flat = {image: {severity: float(image) for severity in range(6)}
            for image in images}
    result = paired_macro_bootstrap(
        images, strong, flat,
        candidate_orientation=1, control_orientation=1, samples=200,
    )
    assert result["macro_difference"] > 0
    assert result["low"] > 0
    assert result["verdict"] == "supported on tuning"


def test_the_full_two_thousand_sample_bootstrap_runs_in_reasonable_time():
    """The production sample count on a small image set, so the batched path is exercised.

    Task 9's integration tests patch `BOOTSTRAP_SAMPLES` down to keep the suite fast; without
    this test nothing would ever run the real number and a per-draw regression would only show
    up on alienware2.
    """
    images = list(range(1, 21))
    candidate = {image: {severity: image + severity for severity in range(6)}
                 for image in images}
    control = {image: {severity: image + 0.5 * severity for severity in range(6)}
               for image in images}
    result = paired_macro_bootstrap(
        images, candidate, control, candidate_orientation=1, control_orientation=1
    )
    assert result["samples"] == BOOTSTRAP_SAMPLES == 2000


def test_row_curves_are_indexed_in_one_pass(tmp_path):
    rows, _ = build_contrast_rows(loaded(tmp_path))
    curves = index_curves(rows)
    assert len(curves) == 45 + 36
    one = curves[("decile_00_10__50_60", "persistence", "mean", "raw_gap")]
    assert sorted(one) == list(range(1, IMAGES + 1))
    assert sorted(one[1]) == list(range(6))


def test_ranking_uses_severity_two_as_the_third_criterion():
    weak = stub(method="raw_gap", auroc_by_severity={1: 0.6, 2: 0.55, 3: 0.7, 4: 0.8, 5: 0.8})
    strong = stub(
        method="relative_gap", auroc_by_severity={1: 0.6, 2: 0.65, 3: 0.7, 4: 0.8, 5: 0.8}
    )
    ranked = rank_contrast_candidates([weak, strong])
    assert [item["method"] for item in ranked] == ["relative_gap", "raw_gap"]


def test_ranking_prefers_macro_then_severity_one():
    low_macro = stub(method="raw_gap", macro_auroc=0.60)
    high_macro = stub(method="relative_gap", macro_auroc=0.80)
    assert [item["method"] for item in rank_contrast_candidates([low_macro, high_macro])] == [
        "relative_gap", "raw_gap"
    ]
    tied_a = stub(method="raw_gap", auroc_by_severity={1: 0.50, 2: 0.9, 3: 0.7, 4: 0.8, 5: 0.8})
    tied_b = stub(
        method="relative_gap", auroc_by_severity={1: 0.70, 2: 0.5, 3: 0.7, 4: 0.8, 5: 0.8}
    )
    assert [item["method"] for item in rank_contrast_candidates([tied_a, tied_b])] == [
        "relative_gap", "raw_gap"
    ]


def test_confidence_twins_and_reference_controls_are_never_ranked():
    twin = stub(signal="confidence", macro_auroc=0.99)
    control = stub(method="raw_reference", macro_auroc=0.99)
    real = stub(macro_auroc=0.55)
    ranked = rank_contrast_candidates([twin, control, real])
    assert [item["method"] for item in ranked] == ["raw_gap"]
    assert twin["deployable"] is False
    assert control["deployable"] is False


def test_a_run_short_of_two_hundred_and_fifty_images_is_not_deployable():
    short = stub(expected_image_count=249)
    assert rank_contrast_candidates([short]) == []
    assert short["deployable"] is False


def test_an_unorientable_or_incomplete_candidate_is_not_deployable():
    for broken in (stub(orientable=False, macro_auroc=None), stub(complete=False)):
        assert rank_contrast_candidates([broken]) == []
        assert broken["deployable"] is False


def test_a_candidate_without_a_computable_twin_is_not_deployable():
    orphan = stub(twin_macro_auroc=None)
    assert rank_contrast_candidates([orphan]) == []
    assert orphan["deployable"] is False
```

- [ ] **Step 2: Run the tests and read the failures**

```bash
$UE_PY -m pytest tests/scene_uncertainty/test_contrast_controls.py -v
```

Expected: collection error — module does not exist.

- [ ] **Step 3: Implement the controls**

Create `src/scene_uncertainty/contrast_controls.py`:

```python
"""The three things a contrast has to beat, and the order the survivors are read in.

A contrast is only interesting relative to something. Three somethings, and each rules out a
different way of being fooled:

* the **raw responsive** range alone -- does combining two ranges beat using one?
* the **raw reference** range alone -- a differential arm subtracts two responsive ranges, so a
  gap that beats the 50--60 percent range while losing to the 90--100 percent range has found
  nothing except which of its two inputs was stronger;
* the **confidence-only twin** -- the same contrast built from detector confidence, which is
  free at deployment where persistence is not.

The completed deployment analysis is why the third exists: at the 90--100 percent decile,
persistence and confidence showed equal-magnitude, opposite-sign trends, which is the shape a
restatement of confidence takes.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from .contrast_analysis import DEPLOYABLE_SIGNAL, summarize_contrast_candidates
from .contrast_inputs import ARMS, FULL_TUNING_IMAGE_COUNT
from .corruption_metrics import EXPECTED_SEVERITIES

BOOTSTRAP_SEED = 20260821
BOOTSTRAP_SAMPLES = 2000
REFERENCE_CONTROL_METHOD = "raw_reference"

_PAIR_BY_ARM = {arm.name: arm.pair_name for arm in ARMS}


def reference_control_rows(rows: list[dict]) -> list[dict]:
    """The reference range as if it were a candidate, so it gets its own orientation and AUROCs.

    Built only from the `raw_responsive` rows, because all four methods of one
    `(arm, signal, aggregation)` carry the same `reference` value and taking them all would
    produce four identical copies whose row keys collide.
    """
    return [
        {**row, "method": REFERENCE_CONTROL_METHOD, "score": row["reference"]}
        for row in rows
        if row["method"] == "raw_responsive"
    ]


def index_curves(rows: list[dict]) -> dict[tuple, dict[int, dict[int, float]]]:
    """`{candidate key: {image_id: {severity: score}}}`, built in one pass over the rows.

    One pass, not one pass per candidate. The real table is 126,000 rows and every candidate
    needs two lookups for its bootstraps, so a per-candidate scan is twelve million row
    comparisons for a dictionary that could have been built once.
    """
    curves: dict[tuple, dict[int, dict[int, float]]] = {}
    for row in rows:
        key = (row["arm"], row["signal"], row["aggregation"], row["method"])
        curves.setdefault(key, {}).setdefault(row["image_id"], {})[
            row["severity"]
        ] = row["score"]
    return curves


def _macro_from_draws(
    clean: np.ndarray, by_severity: dict[int, np.ndarray], draws: np.ndarray, orientation: int
) -> np.ndarray:
    """Macro AUROC for every bootstrap draw at once.

    Ranked in batch with `rankdata(..., axis=1)` rather than one draw at a time. The paired
    bootstrap needs 2,000 draws for each of 45 candidates against each of two controls, and a
    per-draw loop turns a half-minute report into a quarter-hour one. `method="average"` is the
    same tie handling `corruption_metrics.binary_auroc` uses, so a bootstrap median and the
    point estimate cannot disagree about what a tie is worth.
    """
    count = clean.size
    sign = float(orientation)
    clean_draws = sign * clean[draws]
    totals = np.zeros(draws.shape[0], dtype=float)
    for severity in range(1, 6):
        corrupted_draws = sign * by_severity[severity][draws]
        joint = np.concatenate([clean_draws, corrupted_draws], axis=1)
        ranks = rankdata(joint, method="average", axis=1)
        corrupted_rank_sum = ranks[:, count:].sum(axis=1)
        totals += (corrupted_rank_sum - count * (count + 1) / 2) / (count * count)
    return totals / 5.0


def paired_macro_bootstrap(
    image_ids,
    candidate_scores: dict[int, dict[int, float]],
    control_scores: dict[int, dict[int, float]],
    *,
    candidate_orientation: int,
    control_orientation: int,
    seed: int = BOOTSTRAP_SEED,
    samples: int | None = None,
) -> dict:
    """How far apart two methods' macro AUROCs are, and how much of that survives resampling.

    One sampled list of image IDs is applied to *both* methods and all six severities. Drawing
    separately would compare one method on one set of scenes against the other on a different
    set, and most of the resulting spread would be scene sampling rather than the difference
    between the methods.

    The verdict wording is the spec's. Neither label is a held-out claim, because the arms and
    methods were selected using this same tuning programme.
    """
    samples = BOOTSTRAP_SAMPLES if samples is None else samples
    ordered = sorted(image_ids)
    count = len(ordered)
    candidate_clean = np.array([candidate_scores[i][0] for i in ordered], dtype=float)
    control_clean = np.array([control_scores[i][0] for i in ordered], dtype=float)
    candidate_by_severity = {
        severity: np.array([candidate_scores[i][severity] for i in ordered], dtype=float)
        for severity in EXPECTED_SEVERITIES
    }
    control_by_severity = {
        severity: np.array([control_scores[i][severity] for i in ordered], dtype=float)
        for severity in EXPECTED_SEVERITIES
    }
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, count, size=(samples, count))
    differences = _macro_from_draws(
        candidate_clean, candidate_by_severity, draws, candidate_orientation
    ) - _macro_from_draws(control_clean, control_by_severity, draws, control_orientation)
    identity = np.arange(count).reshape(1, count)
    point = float(
        _macro_from_draws(candidate_clean, candidate_by_severity, identity,
                          candidate_orientation)[0]
        - _macro_from_draws(control_clean, control_by_severity, identity,
                            control_orientation)[0]
    )
    low, high = (float(value) for value in np.percentile(differences, [2.5, 97.5]))
    if point > 0 and low > 0:
        verdict = "supported on tuning"
    elif point > 0:
        verdict = "inconclusive on tuning"
    else:
        verdict = "not supported on tuning"
    return {
        "macro_difference": point, "low": low, "high": high,
        "verdict": verdict, "seed": seed, "samples": samples,
    }


def beats_both_inputs(candidate: dict) -> bool:
    """True only when the contrast is ahead of the responsive *and* the reference range.

    Both strictly. A contrast level with one of its inputs has not improved on it, and a
    differential arm level with its stronger input has only rediscovered which input that was.
    """
    responsive = candidate.get("responsive_control_macro_difference")
    reference = candidate.get("reference_control_macro_difference")
    return bool(
        responsive is not None and reference is not None
        and responsive > 0 and reference > 0
    )


def _by_key(items: list[dict]) -> dict[tuple, dict]:
    return {
        (item["arm"], item["signal"], item["aggregation"], item["method"]): item
        for item in items
    }


def attach_controls(
    candidates: list[dict],
    controls: list[dict],
    rows: list[dict],
    *,
    samples: int | None = None,
) -> None:
    """Write every control comparison onto the candidate dictionaries, in place.

    In place rather than returning new dictionaries, so a reader that has a candidate has all
    of it: the ranking, the plots, the CSV and the report each hold the same object, and there
    is no second copy that could quote a different macro AUROC for one candidate.

    Confidence twins are looked up by *pair*, so the `layer_2` and `combined` differential arms
    share one. Their persistence numbers differ; their twin does not, and that is correct --
    there is only one confidence measurement of that bucket pair to be redundant with.
    """
    indexed = _by_key(candidates)
    curves = index_curves(rows)
    control_index = {
        (item["arm"], item["signal"], item["aggregation"]): item for item in controls
    }
    for candidate in candidates:
        if candidate["signal"] != DEPLOYABLE_SIGNAL:
            continue
        arm, aggregation, method = (
            candidate["arm"], candidate["aggregation"], candidate["method"]
        )
        responsive = indexed.get((arm, DEPLOYABLE_SIGNAL, aggregation, "raw_responsive"))
        reference = control_index.get((arm, DEPLOYABLE_SIGNAL, aggregation))
        twin_arm = _PAIR_BY_ARM[arm]
        twin = indexed.get((twin_arm, "confidence", aggregation, method))

        candidate["twin_arm"] = twin_arm
        candidate["twin_macro_auroc"] = twin["macro_auroc"] if twin else None
        candidate["twin_auroc_by_severity"] = twin["auroc_by_severity"] if twin else None
        candidate["twin_orientation"] = twin["orientation"] if twin else None
        candidate["responsive_control_macro_auroc"] = (
            responsive["macro_auroc"] if responsive else None
        )
        candidate["reference_control_macro_auroc"] = (
            reference["macro_auroc"] if reference else None
        )

        for label, other in (
            ("responsive_control", responsive),
            ("reference_control", reference),
            ("twin", twin),
        ):
            if candidate["macro_auroc"] is None or other is None or other["macro_auroc"] is None:
                candidate[f"{label}_macro_difference"] = None
                candidate[f"{label}_auroc_difference_by_severity"] = None
                continue
            candidate[f"{label}_macro_difference"] = (
                candidate["macro_auroc"] - other["macro_auroc"]
            )
            candidate[f"{label}_auroc_difference_by_severity"] = {
                severity: candidate["auroc_by_severity"][severity]
                - other["auroc_by_severity"][severity]
                for severity in range(1, 6)
            }

        candidate["beats_both_inputs"] = beats_both_inputs(candidate)
        candidate["confidence_redundant"] = not (
            candidate["macro_auroc"] is not None
            and candidate["twin_macro_auroc"] is not None
            and candidate["macro_auroc"] > candidate["twin_macro_auroc"]
        )

        candidate["responsive_control_bootstrap"] = None
        candidate["twin_bootstrap"] = None
        if candidate["macro_auroc"] is None:
            continue
        own = curves[(arm, DEPLOYABLE_SIGNAL, aggregation, method)]
        image_ids = sorted(own)
        if responsive is not None and responsive["macro_auroc"] is not None:
            candidate["responsive_control_bootstrap"] = paired_macro_bootstrap(
                image_ids, own,
                curves[(arm, DEPLOYABLE_SIGNAL, aggregation, "raw_responsive")],
                candidate_orientation=candidate["orientation"],
                control_orientation=responsive["orientation"],
                samples=samples,
            )
        if twin is not None and twin["macro_auroc"] is not None:
            candidate["twin_bootstrap"] = paired_macro_bootstrap(
                image_ids, own,
                curves[(twin_arm, "confidence", aggregation, method)],
                candidate_orientation=candidate["orientation"],
                control_orientation=twin["orientation"],
                samples=samples,
            )


def _sort_key(candidate: dict) -> tuple:
    auroc = candidate["auroc_by_severity"]
    return (
        -candidate["macro_auroc"],
        -auroc[1],
        -auroc[2],
        -(candidate["median_absolute_spearman"] or 0.0),
        -(candidate["dominant_direction_fraction"] or 0.0),
        -(candidate["oriented_adjacent_consistency"] or 0.0),
        candidate["arm"], candidate["aggregation"], candidate["method"],
    )


def rank_contrast_candidates(candidates: list[dict]) -> list[dict]:
    """Mark every candidate `deployable` or not, and return the deployable ones in order.

    Severities 1 and 2 sit second and third on purpose. The completed deployment analysis left
    every candidate it produced near chance at mild blur, so a macro gain driven entirely by
    severities 4 and 5 repeats a result already in hand and is not what this experiment is for.

    The returned list holds the same dictionaries, not copies -- a reader that edits one sees
    the other, which is what keeps a report from quoting two different values for one candidate.
    """
    for candidate in candidates:
        candidate["deployable"] = bool(
            candidate["signal"] == DEPLOYABLE_SIGNAL
            and candidate["method"] != REFERENCE_CONTROL_METHOD
            and candidate.get("complete")
            and candidate.get("orientable")
            and candidate.get("macro_auroc") is not None
            and candidate.get("twin_macro_auroc") is not None
            and candidate.get("expected_image_count") == FULL_TUNING_IMAGE_COUNT
        )
    return sorted(
        [candidate for candidate in candidates if candidate["deployable"]], key=_sort_key
    )
```

- [ ] **Step 4: Run the tests and verify they pass**

```bash
$UE_PY -m pytest tests/scene_uncertainty/test_contrast_controls.py -v
```

Expected: 18 passed.

- [ ] **Step 5: Mutation check**

Break each, confirm the *named* test fails, revert:

1. `attach_controls` → look the twin up by `arm` instead of `_PAIR_BY_ARM[arm]`. `test_both_differential_arms_resolve_to_the_same_confidence_twin` must fail (the `combined` arm finds no twin).
2. `attach_controls` → pass `candidate["orientation"]` as `control_orientation` for the twin. `test_the_reference_control_locks_its_own_orientation` covers the reference control; add the twin equivalent — build a fixture where candidate and twin lock to opposite orientations and assert `twin_orientation != orientation`. **The first draft does not cover this.** Add it.
3. `paired_macro_bootstrap` → draw a second `draws` matrix for the control. `test_the_paired_bootstrap_resamples_both_methods_identically` must fail (a method against itself stops being exactly zero).
4. `paired_macro_bootstrap` → drop the `seed` argument and use fresh entropy. `test_the_paired_bootstrap_is_deterministic_under_its_seed` must fail.
5. `_sort_key` → remove the `-auroc[2]` term. `test_ranking_uses_severity_two_as_the_third_criterion` must fail.
6. `beats_both_inputs` → use `or` instead of `and`. `test_beating_one_input_while_losing_to_the_other_is_not_beating_both` must fail.
7. `rank_contrast_candidates` → drop the `twin_macro_auroc is not None` clause. `test_a_candidate_without_a_computable_twin_is_not_deployable` must fail.
8. `confidence_redundant` → use `>=` instead of `>`. Build a candidate whose macro exactly equals its twin's and assert `confidence_redundant is True`. **Not covered by the first draft.** Add it.

- [ ] **Step 6: Commit**

```bash
git add src/scene_uncertainty/contrast_controls.py \
        tests/scene_uncertainty/test_contrast_controls.py
git commit -m "feat: add reference control, confidence twin, paired bootstrap, and ranking"
```

---

## Task 7: The four figures

**Files:**
- Create: `src/scene_uncertainty/contrast_plots.py`
- Create: `tests/scene_uncertainty/test_contrast_plots.py`

**Interfaces:**
- Consumes: candidates and controls from Tasks 5–6, contrast rows from Task 4, `fits` from Task 4.
- Produces: `PLOT_FILENAMES`, `PLOT_DPI`, `write_contrast_plots(directory, *, candidates, controls, rows, fits) -> dict`. Task 8 records the returned axis limits in `summary.json` verbatim and never recomputes them.

**Panel layouts** (the spec fixes these):

| File | Panels | Content |
|---|---|---|
| `anchor_and_responsive_actual_distance.png` | 12 = 4 arms × 3 summaries | Raw un-oriented reference and responsive median + IQR band over severities 0–5, sharing one axis per panel |
| `clean_anchor_relationship.png` | 12 = 4 arms × 3 summaries | Severity-zero reference/responsive scatter plus the **final** robust line |
| `contrast_scores_by_severity.png` | 4 = one per score method | Median line and IQR band across images; separate panels because the four methods' units differ |
| `auroc_by_blur_severity.png` | 12 = 4 arms × 3 summaries | Four methods solid, their confidence twins dashed in the same colour, chance line at 0.5, shared y-axis 0–1 |

- [ ] **Step 1: Write the failing tests**

Create `tests/scene_uncertainty/test_contrast_plots.py`:

```python
import json

import pytest

from src.scene_uncertainty.contrast_analysis import (
    build_contrast_rows,
    summarize_contrast_candidates,
)
from src.scene_uncertainty.contrast_controls import (
    attach_controls,
    reference_control_rows,
)
from src.scene_uncertainty.contrast_inputs import load_contrast_inputs
from src.scene_uncertainty.contrast_plots import PLOT_FILENAMES, write_contrast_plots

from tests.scene_uncertainty.contrast_test_utils import write_source_bundle

IMAGES = 6


@pytest.fixture
def prepared(tmp_path):
    source = write_source_bundle(tmp_path / "source")
    inputs = load_contrast_inputs(source, expected_image_count=IMAGES)
    rows, fits = build_contrast_rows(inputs)
    candidates = summarize_contrast_candidates(rows, expected_image_count=IMAGES)
    controls = summarize_contrast_candidates(
        reference_control_rows(rows), expected_image_count=IMAGES
    )
    attach_controls(candidates, controls, rows, samples=20)
    return {"candidates": candidates, "controls": controls, "rows": rows, "fits": fits}


def test_the_four_declared_figures_are_written(tmp_path, prepared):
    write_contrast_plots(tmp_path, **prepared)
    written = sorted(path.name for path in tmp_path.glob("*.png"))
    assert written == sorted(PLOT_FILENAMES.values())
    assert len(written) == 4
    for name in written:
        assert (tmp_path / name).stat().st_size > 0


def test_the_figure_names_are_the_spec_names():
    assert sorted(PLOT_FILENAMES.values()) == [
        "anchor_and_responsive_actual_distance.png",
        "auroc_by_blur_severity.png",
        "clean_anchor_relationship.png",
        "contrast_scores_by_severity.png",
    ]


def test_axis_limits_are_returned_for_every_figure(tmp_path, prepared):
    limits = write_contrast_plots(tmp_path, **prepared)
    assert set(limits) == set(PLOT_FILENAMES)
    for key, value in limits.items():
        assert isinstance(value, list) and len(value) == 2
        assert value[0] <= value[1]
    # AUROC panels always span the full interval, because that is what makes them comparable
    assert limits["auroc"] == [0.0, 1.0]


def test_axis_limits_survive_a_json_round_trip(tmp_path, prepared):
    limits = write_contrast_plots(tmp_path, **prepared)
    assert json.loads(json.dumps(limits)) == limits


def test_the_panel_counts_are_the_declared_ones(tmp_path, prepared):
    from src.scene_uncertainty.contrast_plots import panel_plan

    plan = panel_plan(prepared["candidates"])
    assert len(plan["anchor"]) == 12
    assert len(plan["relationship"]) == 12
    assert len(plan["contrast"]) == 4
    assert len(plan["auroc"]) == 12


def test_a_missing_twin_leaves_a_labelled_empty_panel_rather_than_crashing(tmp_path, prepared):
    for candidate in prepared["candidates"]:
        candidate["twin_auroc_by_severity"] = None
        candidate["macro_auroc"] = None
        candidate["auroc_by_severity"] = None
    write_contrast_plots(tmp_path, **prepared)
    assert (tmp_path / PLOT_FILENAMES["auroc"]).stat().st_size > 0


def test_writing_into_a_missing_directory_leaves_no_open_figures(tmp_path, prepared):
    import matplotlib.pyplot as plt

    with pytest.raises(OSError):
        write_contrast_plots(tmp_path / "absent", **prepared)
    assert plt.get_fignums() == []
```

- [ ] **Step 2: Run the tests and read the failures**

```bash
$UE_PY -m pytest tests/scene_uncertainty/test_contrast_plots.py -v
```

Expected: collection error — module does not exist.

- [ ] **Step 3: Implement the figures**

Create `src/scene_uncertainty/contrast_plots.py`. Follow `corruption_plots.py`'s conventions exactly: pin the backend before importing `pyplot`, build every figure before writing any file, and close them all in a `finally`.

```python
"""Four figures, and the axis ranges they were actually drawn on.

Panels are grouped by what shares a meaning, not by what fits. The anchor panels put reference
and responsive on one axis because they are the same quantity in the same units; the contrast
panels give each score method its own panel because a relative gap and a raw distance are not
comparable numbers; the AUROC panels share one 0-to-1 axis because AUROC means the same thing
everywhere, which is what makes a dashed confidence twin readable against its solid candidate.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

# Drawn on a headless box, so the backend is pinned before pyplot is imported.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .contrast_analysis import DEPLOYABLE_SIGNAL  # noqa: E402
from .contrast_inputs import AGGREGATIONS, ARMS  # noqa: E402
from .contrast_scores import SCORE_METHODS  # noqa: E402
from .corruption_metrics import EXPECTED_SEVERITIES  # noqa: E402

PLOT_FILENAMES = {
    "anchor": "anchor_and_responsive_actual_distance.png",
    "relationship": "clean_anchor_relationship.png",
    "contrast": "contrast_scores_by_severity.png",
    "auroc": "auroc_by_blur_severity.png",
}
PLOT_DPI = 160
AUROC_LIMITS = [0.0, 1.0]
CHANCE = 0.5
METHOD_COLOURS = dict(
    zip(SCORE_METHODS, ("tab:blue", "tab:orange", "tab:green", "tab:red"))
)


def panel_plan(candidates: list[dict]) -> dict[str, list[tuple]]:
    """Which panels each figure has, derived from the arm table rather than from the data.

    Derived, so a figure cannot quietly shrink when a candidate fails to produce numbers: an
    arm with nothing to draw gets an empty panel that says so, and a reader counting twelve
    panels is counting the experiment rather than counting its successes.
    """
    arm_panels = [(arm.name, aggregation) for arm in ARMS for aggregation in AGGREGATIONS]
    return {
        "anchor": arm_panels,
        "relationship": arm_panels,
        "contrast": [(method,) for method in SCORE_METHODS],
        "auroc": arm_panels,
    }
```

Then implement four private figure builders and the public writer. Requirements each one must meet:

- **`_anchor_figure`** — 4×3 grid. For each `(arm, aggregation)` take the `raw_responsive` candidate's `severity_statistics` for the responsive line and the matching `raw_reference` control's for the reference line. Plot `median` with an IQR band from `q25`/`q75` against `EXPECTED_SEVERITIES`. Do **not** orient: the caption is about raw distance. Return the figure and `[min, max]` over every value drawn.
- **`_relationship_figure`** — 4×3 grid. For each `(arm, aggregation)` scatter the severity-zero `reference` against `responsive` from `rows` (persistence rows, `raw_responsive` method only, so each image contributes one point), then draw `fits[(arm, "persistence", aggregation)]["final_line"]` across the reference range. Axis labels name which bucket is which. The caption states that the cross-fitted residual metrics use fold-specific lines even though this figure shows the final deployment line.
- **`_contrast_figure`** — 1×4 grid, one panel per method, every arm/aggregation drawn as its own line. Median and IQR band from `severity_statistics`. Raw direction, never oriented.
- **`_auroc_figure`** — 4×3 grid. Per panel, plot each method's `auroc_by_severity` for severities 1–5 solid in `METHOD_COLOURS[method]`, and its `twin_auroc_by_severity` dashed in the same colour. Horizontal chance line at `CHANCE`. `set_ylim(*AUROC_LIMITS)`.
- **Empty panels** — when a candidate has `None` where numbers should be, call a shared `_absent(axis, message)` that writes a short centred message and removes the ticks. Never skip the panel.
- **`write_contrast_plots(directory, *, candidates, controls, rows, fits)`** — build all four figures first, then `savefig` each at `PLOT_DPI`, closing every figure in a `finally`. Return `{key: [lower, upper]}` for all four keys, with `"auroc": list(AUROC_LIMITS)`.

- [ ] **Step 4: Run the tests and verify they pass**

```bash
$UE_PY -m pytest tests/scene_uncertainty/test_contrast_plots.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Look at the figures**

```bash
mkdir -p /tmp/contrast-preview
PYTHONPATH=$PWD $UE_PY - <<'PY'
import sys
from pathlib import Path

from src.scene_uncertainty.contrast_analysis import (
    build_contrast_rows,
    summarize_contrast_candidates,
)
from src.scene_uncertainty.contrast_controls import attach_controls, reference_control_rows
from src.scene_uncertainty.contrast_inputs import load_contrast_inputs
from src.scene_uncertainty.contrast_plots import write_contrast_plots

sys.path.insert(0, str(Path.cwd() / "tests" / "scene_uncertainty"))
from contrast_test_utils import write_source_bundle  # noqa: E402

out = Path("/tmp/contrast-preview")
source = write_source_bundle(out / "source", image_ids=range(1, 21))
inputs = load_contrast_inputs(source, expected_image_count=20)
rows, fits = build_contrast_rows(inputs)
candidates = summarize_contrast_candidates(rows, expected_image_count=20)
controls = summarize_contrast_candidates(
    reference_control_rows(rows), expected_image_count=20
)
attach_controls(candidates, controls, rows, samples=50)
print(write_contrast_plots(out, candidates=candidates, controls=controls,
                           rows=rows, fits=fits))
PY
xdg-open /tmp/contrast-preview/auroc_by_blur_severity.png
```

Run it from the repository root. Twenty images rather than six, because six produce interquartile bands too narrow to judge by eye.

Confirm by eye: twelve panels where twelve are declared, no overlapping labels, the dashed twin visibly distinguishable from its solid candidate, and the chance line present on every AUROC panel. A figure test that only checks `st_size > 0` has not looked at the figure.

- [ ] **Step 6: Mutation check**

1. `panel_plan` → derive `arm_panels` from the candidates present rather than from `ARMS`. `test_the_panel_counts_are_the_declared_ones` must fail once a candidate is removed — add that removal to the test.
2. `write_contrast_plots` → return a recomputed limit for `"auroc"` instead of `AUROC_LIMITS`. `test_axis_limits_are_returned_for_every_figure` must fail.
3. `write_contrast_plots` → drop the `finally` that closes figures. `test_writing_into_a_missing_directory_leaves_no_open_figures` must fail.
4. `_auroc_figure` → draw the twin solid rather than dashed. Not catchable by file size; this is what Step 5 is for.

- [ ] **Step 7: Commit**

```bash
git add src/scene_uncertainty/contrast_plots.py tests/scene_uncertainty/test_contrast_plots.py
git commit -m "feat: draw anchor, relationship, contrast, and AUROC figures"
```

---

## Task 8: The atomic bundle and the plain-language report

**Files:**
- Create: `src/scene_uncertainty/contrast_reporting.py`
- Create: `tests/scene_uncertainty/test_contrast_reporting.py`

**Interfaces:**
- Consumes: everything from Tasks 4–7.
- Produces: `EXPECTED_FILES`, `SECTION_TITLES`, `build_contrast_summary(...)`, `render_contrast_report(summary)`, `write_contrast_report(output_value, *, inputs, rows, fits, diagnostics, candidates, controls, ranking)`. Task 9's pipeline handler calls only `write_contrast_report`.

**The bundle is exactly nine files:**

```text
per_scene_contrasts.csv
anchor_diagnostics.csv
candidate_metrics.csv
summary.json
anchor_and_responsive_actual_distance.png
clean_anchor_relationship.png
contrast_scores_by_severity.png
auroc_by_blur_severity.png
easy-report.md
```

**CSV projection rule.** A CSV cell cannot hold a dictionary. Every per-severity map is published twice: nested in `summary.json` (which the plots and a programmatic reader use) and flattened into scalar columns in the CSV. The flattening is a single rule — `{"auroc_by_severity": {1: 0.6}}` becomes `auroc_severity_1` — declared once as a mapping and applied by the writer, so the CSV projection stays "drop every dictionary-valued key, after expanding the declared ones".

| Nested key | Column prefix |
|---|---|
| `auroc_by_severity` | `auroc_severity` |
| `twin_auroc_by_severity` | `twin_auroc_severity` |
| `responsive_control_auroc_difference_by_severity` | `responsive_control_auroc_difference_severity` |
| `reference_control_auroc_difference_by_severity` | `reference_control_auroc_difference_severity` |
| `twin_auroc_difference_by_severity` | `twin_auroc_difference_severity` |
| `severity_statistics` | `score_count`, `score_mean`, `score_variance`, `score_median`, `score_q25`, `score_q75` (one column per statistic per severity) |
| `responsive_control_bootstrap` | `responsive_control_bootstrap_macro_difference`, `_low`, `_high`, `_verdict` |
| `twin_bootstrap` | `twin_bootstrap_macro_difference`, `_low`, `_high`, `_verdict` |

- [ ] **Step 1: Write the failing tests**

Create `tests/scene_uncertainty/test_contrast_reporting.py`:

```python
import csv
import json

import pytest

from src.scene_uncertainty.contrast_analysis import (
    build_anchor_diagnostics,
    build_contrast_rows,
    summarize_contrast_candidates,
)
from src.scene_uncertainty.contrast_controls import (
    attach_controls,
    rank_contrast_candidates,
    reference_control_rows,
)
from src.scene_uncertainty.contrast_inputs import load_contrast_inputs
from src.scene_uncertainty.contrast_reporting import (
    EXPECTED_FILES,
    SECTION_TITLES,
    render_contrast_report,
    write_contrast_report,
)

from tests.scene_uncertainty.contrast_test_utils import write_source_bundle

IMAGES = 6


@pytest.fixture
def bundle_inputs(tmp_path):
    source = write_source_bundle(tmp_path / "source")
    inputs = load_contrast_inputs(source, expected_image_count=IMAGES)
    rows, fits = build_contrast_rows(inputs)
    candidates = summarize_contrast_candidates(rows, expected_image_count=IMAGES)
    controls = summarize_contrast_candidates(
        reference_control_rows(rows), expected_image_count=IMAGES
    )
    attach_controls(candidates, controls, rows, samples=20)
    ranking = rank_contrast_candidates(candidates)
    return {
        "inputs": inputs, "rows": rows, "fits": fits,
        "diagnostics": build_anchor_diagnostics(inputs),
        "candidates": candidates, "controls": controls, "ranking": ranking,
    }


def test_the_bundle_holds_exactly_nine_declared_files(tmp_path, bundle_inputs):
    output = tmp_path / "report"
    write_contrast_report(output, **bundle_inputs)
    assert sorted(path.name for path in output.iterdir()) == sorted(EXPECTED_FILES)
    assert len(EXPECTED_FILES) == 9


def test_an_existing_output_directory_is_refused_before_anything_is_computed(
    tmp_path, bundle_inputs
):
    output = tmp_path / "report"
    output.mkdir()
    with pytest.raises(FileExistsError):
        write_contrast_report(output, **bundle_inputs)
    assert list(output.iterdir()) == []


def test_a_failed_run_leaves_neither_output_nor_staging(tmp_path, bundle_inputs, monkeypatch):
    import src.scene_uncertainty.contrast_reporting as reporting

    def explode(*args, **kwargs):
        raise RuntimeError("figure failure")

    monkeypatch.setattr(reporting, "write_contrast_plots", explode)
    output = tmp_path / "report"
    with pytest.raises(RuntimeError, match="figure failure"):
        write_contrast_report(output, **bundle_inputs)
    assert not output.exists()
    assert list(tmp_path.iterdir()) == [tmp_path / "source"]


def test_candidate_metrics_csv_holds_no_dictionary_cells(tmp_path, bundle_inputs):
    output = tmp_path / "report"
    write_contrast_report(output, **bundle_inputs)
    with (output / "candidate_metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    for row in rows:
        for value in row.values():
            assert not value.startswith("{")
    assert "auroc_severity_1" in rows[0]
    assert "twin_auroc_severity_5" in rows[0]
    assert "declared_before_data" in rows[0]
    assert "confidence_redundant" in rows[0]
    assert "arm_family" in rows[0]


def test_per_scene_contrasts_has_one_row_per_contrast_row(tmp_path, bundle_inputs):
    output = tmp_path / "report"
    write_contrast_report(output, **bundle_inputs)
    with (output / "per_scene_contrasts.csv").open() as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == len(bundle_inputs["rows"])


def test_anchor_diagnostics_csv_covers_all_twenty_one_configurations(tmp_path, bundle_inputs):
    output = tmp_path / "report"
    write_contrast_report(output, **bundle_inputs)
    with (output / "anchor_diagnostics.csv").open() as handle:
        written = list(csv.DictReader(handle))
    assert len({(row["arm"], row["signal"], row["aggregation"]) for row in written}) == 21
    assert "stability_to_spread_severity_3" in written[0]
    assert "arm_family" in written[0]


def test_summary_records_the_axis_limits_the_figures_used(tmp_path, bundle_inputs):
    output = tmp_path / "report"
    write_contrast_report(output, **bundle_inputs)
    summary = json.loads((output / "summary.json").read_text())
    assert set(summary["axis_limits"]) == {
        "anchor", "relationship", "contrast", "auroc"
    }
    assert summary["axis_limits"]["auroc"] == [0.0, 1.0]


def test_summary_records_the_arm_table_with_its_provenance(tmp_path, bundle_inputs):
    output = tmp_path / "report"
    write_contrast_report(output, **bundle_inputs)
    summary = json.loads((output / "summary.json").read_text())
    arms = {arm["name"]: arm for arm in summary["arms"]}
    assert len(arms) == 4
    assert arms["decile_00_10__50_60"]["declared_before_data"] is True
    assert arms["decile_90_100__50_60__combined"]["declared_before_data"] is False
    assert arms["decile_90_100__50_60__combined"]["family"] == "differential"


def test_summary_records_twenty_one_final_fits(tmp_path, bundle_inputs):
    output = tmp_path / "report"
    write_contrast_report(output, **bundle_inputs)
    summary = json.loads((output / "summary.json").read_text())
    assert len(summary["fits"]) == 21
    entry = summary["fits"][0]
    assert "final_line" in entry and "fold_lines" in entry


def test_the_report_leads_with_the_declared_sections(tmp_path, bundle_inputs):
    text = render_contrast_report(_summary_for(tmp_path, bundle_inputs))
    for title in SECTION_TITLES:
        assert title in text
    # the order is part of the contract: severities 1 and 2 come before the bootstrap
    positions = [text.index(title) for title in SECTION_TITLES]
    assert positions == sorted(positions)


def test_the_summary_is_json_serialisable_without_a_fallback(tmp_path, bundle_inputs):
    """No `default=str`. A summary that only serialises through a coercion hook is a summary
    holding a numpy scalar or a `nan`, and both reach a reader as something they are not."""
    summary = _summary_for(tmp_path, bundle_inputs)
    encoded = json.dumps(summary)  # raises TypeError on a numpy scalar
    assert json.loads(encoded)["axis_limits"]["auroc"] == [0.0, 1.0]
    # json.dumps writes bare NaN and Infinity, which are not valid JSON for any other reader
    assert "NaN" not in encoded
    assert "Infinity" not in encoded


def _summary_for(tmp_path, bundle_inputs):
    from src.scene_uncertainty.contrast_reporting import build_contrast_summary
    from src.scene_uncertainty.contrast_plots import write_contrast_plots

    directory = tmp_path / "figures"
    directory.mkdir()
    limits = write_contrast_plots(
        directory,
        candidates=bundle_inputs["candidates"],
        controls=bundle_inputs["controls"],
        rows=bundle_inputs["rows"],
        fits=bundle_inputs["fits"],
    )
    return build_contrast_summary(
        inputs=bundle_inputs["inputs"], rows=bundle_inputs["rows"],
        fits=bundle_inputs["fits"], diagnostics=bundle_inputs["diagnostics"],
        candidates=bundle_inputs["candidates"], controls=bundle_inputs["controls"],
        ranking=bundle_inputs["ranking"], axis_limits=limits,
    )


def test_the_report_makes_no_probability_claim(tmp_path, bundle_inputs):
    output = tmp_path / "report"
    write_contrast_report(output, **bundle_inputs)
    text = (output / "easy-report.md").read_text().lower()
    for forbidden in ("probability of corruption", "calibrated", "% chance", "likelihood that"):
        assert forbidden not in text
    assert "held-out" in text
    assert "no held-out images were used" in text


def test_the_report_states_a_differential_result_was_selected_on_tuning(
    tmp_path, bundle_inputs
):
    output = tmp_path / "report"
    write_contrast_report(output, **bundle_inputs)
    text = (output / "easy-report.md").read_text()
    assert "chosen after reading tuning" in text
    assert "differential" in text


def test_the_report_names_a_redundant_candidate_as_adding_nothing(tmp_path, bundle_inputs):
    for candidate in bundle_inputs["candidates"]:
        if candidate["signal"] == "persistence":
            candidate["confidence_redundant"] = True
    output = tmp_path / "report"
    write_contrast_report(output, **bundle_inputs)
    text = (output / "easy-report.md").read_text()
    assert "adds nothing over the detector's own confidence" in text
```

- [ ] **Step 2: Run the tests and read the failures**

```bash
$UE_PY -m pytest tests/scene_uncertainty/test_contrast_reporting.py -v
```

Expected: collection error — module does not exist.

- [ ] **Step 3: Implement the writer**

Create `src/scene_uncertainty/contrast_reporting.py`. Copy the atomic-publication mechanics from `corruption_reporting.write_corruption_report` — they are already reviewed, already correct, and already documented:

- refuse an existing `--output` **before** computing anything;
- write everything into a `mkdtemp` staging directory created **beside** the destination, so `os.replace` stays within one filesystem and is atomic;
- `chmod` the staging directory to the umask-derived mode, so the published bundle is not left `0o700`;
- check the produced file set against `EXPECTED_FILES` **before** the rename, never after;
- clean up on `BaseException`, not `Exception`, so a `KeyboardInterrupt` leaves nothing half-written.

**One deliberate difference from the sibling.** `corruption_reporting` imports its plot module *inside* the writing function because `corruption_plots` imports `corruption_reporting` back, and a module-scope import would be a cycle that fails at interpreter start-up. `contrast_plots` imports nothing from `contrast_reporting`, so there is no cycle here — import `write_contrast_plots` at module scope. It also has to be a module global for `test_a_failed_run_leaves_neither_output_nor_staging` to patch it; a function-local import would rebind the name after the patch and the test would silently stop testing anything.

Declare:

```python
EXPECTED_FILES = (
    "per_scene_contrasts.csv",
    "anchor_diagnostics.csv",
    "candidate_metrics.csv",
    "summary.json",
    "anchor_and_responsive_actual_distance.png",
    "clean_anchor_relationship.png",
    "contrast_scores_by_severity.png",
    "auroc_by_blur_severity.png",
    "easy-report.md",
)

SEVERITY_MAP_COLUMNS = {
    "auroc_by_severity": "auroc_severity",
    "twin_auroc_by_severity": "twin_auroc_severity",
    "responsive_control_auroc_difference_by_severity":
        "responsive_control_auroc_difference_severity",
    "reference_control_auroc_difference_by_severity":
        "reference_control_auroc_difference_severity",
    "twin_auroc_difference_by_severity": "twin_auroc_difference_severity",
}

SEVERITY_STATISTIC_COLUMNS = {
    "count": "score_count", "mean": "score_mean", "variance": "score_variance",
    "median": "score_median", "q25": "score_q25", "q75": "score_q75",
}

BOOTSTRAP_COLUMNS = ("macro_difference", "low", "high", "verdict")

SECTION_TITLES = (
    "## Is the low-confidence range steady enough to be an anchor?",
    "## Does the anchor explain how scenes differ?",
    "## Does any contrast beat both of the ranges it is built from?",
    "## Does any contrast beat the detector's own confidence?",
    "## How well does it separate each blur level?",
    "## How much of this survives resampling?",
    "## Anchored and differential are two separate verdicts",
    "## What a deployment would store",
    "## What this result is not",
)
```

`build_contrast_summary(*, inputs, rows, fits, diagnostics, candidates, controls, ranking, axis_limits)` returns a dictionary holding: `provenance` (from `inputs.provenance`), `arms` (name, pair_name, family, scheme, scope, reference/responsive bins, `declared_before_data`), `fold_assignment`, `fits` (a list of 21 entries, each with its key, `final_line`, `fold_lines`, `residual_available`, `unavailable_reason`), `anchor_diagnostics`, `candidates`, `reference_controls`, `ranking` (candidate keys only, so one candidate is never serialised twice with different numbers), `axis_limits` recorded verbatim, `row_counts`, and `bootstrap` settings (seed and sample count).

`render_contrast_report(summary)` writes the nine sections in `SECTION_TITLES` order. Requirements the tests pin:

- Section 4 must contain the literal phrase **"adds nothing over the detector's own confidence"** for any reported candidate whose `confidence_redundant` is true.
- Section 5 gives severities 1 and 2 before 3, 4 and 5, and says in words that these are the levels the completed deployment analysis could not separate.
- Section 7 must contain **"chosen after reading tuning"** whenever it reports a differential result, and must state the anchored and differential verdicts separately.
- Section 9 must contain **"no held-out images were used"** and must not contain probability language.
- When nothing is deployable, every section still renders and says so; a report that omits a section because it had no winner is a report that reads as if the question was not asked.
- Sections 3, 4 and 5 read from every candidate that produced a macro AUROC, not only from `ranking`. `ranking` names the winner; it is not the set of results. A report whose control comparisons vanish on a run where nothing cleared the 250-image gate would hide exactly the numbers a reader needs to see why nothing did.
- Section 7 states the anchored/differential distinction and the selection debt whether or not a differential arm won anything, because the distinction is a property of the design and not of the outcome.

- [ ] **Step 4: Run the tests and verify they pass**

```bash
$UE_PY -m pytest tests/scene_uncertainty/test_contrast_reporting.py -v
```

Expected: 15 passed.

- [ ] **Step 5: Mutation check**

1. Rename one entry of `EXPECTED_FILES` → `test_the_bundle_holds_exactly_nine_declared_files` must fail.
2. Move the existing-output check to after the staging build → `test_an_existing_output_directory_is_refused_before_anything_is_computed` must fail on the leftover staging directory.
3. Catch `Exception` instead of `BaseException` in the cleanup → not caught by these tests; add a test that raises `KeyboardInterrupt` from the patched plot writer and asserts no staging directory survives.
4. Recompute `axis_limits` inside `build_contrast_summary` instead of taking the argument → `test_summary_records_the_axis_limits_the_figures_used` still passes, because both computations agree today. Change the test to pass a deliberately wrong `axis_limits` and assert the summary carries it verbatim. **Do this before committing** — a recorded limit that is re-derived is not a record of what the figure used.
5. Drop the `declared_before_data` column from the CSV projection → `test_candidate_metrics_csv_holds_no_dictionary_cells` must fail.

- [ ] **Step 6: Commit**

```bash
git add src/scene_uncertainty/contrast_reporting.py \
        tests/scene_uncertainty/test_contrast_reporting.py
git commit -m "feat: publish the within-image contrast bundle and plain-language report"
```

---

## Task 9: The cache-only CLI, and proof that it is cache-only

**Files:**
- Modify: `src/scene_uncertainty/cli.py`
- Modify: `src/scene_uncertainty/pipeline.py`
- Modify: `README.md`
- Create: `tests/scene_uncertainty/test_contrast_integration.py`
- Modify: `tests/scene_uncertainty/test_cli.py`

**Interfaces:**
- Consumes: `contrast_inputs.load_contrast_inputs`, `contrast_analysis.build_contrast_rows` / `build_anchor_diagnostics` / `summarize_contrast_candidates`, `contrast_controls.reference_control_rows` / `attach_controls` / `rank_contrast_candidates`, `contrast_reporting.write_contrast_report`.
- Produces: the `analyze-within-image-contrast` subcommand and `pipeline.command_analyze_within_image_contrast`.

- [ ] **Step 1: Write the failing tests**

Create `tests/scene_uncertainty/test_contrast_integration.py`:

```python
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from src.scene_uncertainty.cli import main
from tests.scene_uncertainty.contrast_test_utils import write_source_bundle
from tests.scene_uncertainty.test_corruption_integration import (
    DETONATION_SITES,
    _module,
)

CONTRAST_REPORT_FILES = (
    "anchor_and_responsive_actual_distance.png",
    "anchor_diagnostics.csv",
    "auroc_by_blur_severity.png",
    "candidate_metrics.csv",
    "clean_anchor_relationship.png",
    "contrast_scores_by_severity.png",
    "easy-report.md",
    "per_scene_contrasts.csv",
    "summary.json",
)


def _run(source: Path, output: Path) -> int:
    return main([
        "analyze-within-image-contrast",
        "--source", str(source),
        "--output", str(output),
    ])


@pytest.fixture(autouse=True)
def fast_bootstrap(monkeypatch):
    """Twenty draws instead of two thousand, for every test in this module.

    The real count is exercised by `test_the_full_two_thousand_sample_bootstrap_runs_in_
    reasonable_time` in `test_contrast_controls.py` and by the real run in Task 10. Here it
    would add roughly three minutes per test for no additional coverage: these tests are about
    what the command publishes and what it refuses, not about resampling.

    Patching the module constant works because `attach_controls` and `paired_macro_bootstrap`
    both take `samples=None` and resolve it at call time. A default of `BOOTSTRAP_SAMPLES`
    evaluated at definition time would ignore this patch silently.
    """
    monkeypatch.setattr(
        "src.scene_uncertainty.contrast_controls.BOOTSTRAP_SAMPLES", 20
    )


def _full_source(tmp_path: Path) -> Path:
    """250 images, because the deployability gate accepts no other run size."""
    return write_source_bundle(tmp_path / "source", image_ids=range(1, 251))


def test_the_command_publishes_every_declared_artifact(tmp_path: Path, capsys):
    output = tmp_path / "contrast"
    assert _run(_full_source(tmp_path), output) == 0
    assert sorted(path.name for path in output.iterdir()) == list(CONTRAST_REPORT_FILES)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["provenance"]["image_count"] == 250
    assert len(summary["arms"]) == 4
    assert capsys.readouterr().err == ""


def test_no_model_bank_or_knn_entry_point_is_reachable(tmp_path: Path, monkeypatch):
    """The command's reason to exist, checked instead of written down.

    Every number this command publishes was already computed by
    `analyze-corruption-sensitivity`. A rewrite that re-extracted fingerprints or re-searched
    the bank would still publish nine plausible files and every other assertion here would
    still hold -- and it would publish different numbers.
    """
    def detonator(site: str):
        def explode(*args, **kwargs):
            raise AssertionError(f"analyze-within-image-contrast called {site}")
        return explode

    for site in DETONATION_SITES:
        module_name, _, name = site.partition(".")
        monkeypatch.setattr(_module(module_name), name, detonator(site))

    output = tmp_path / "contrast"
    assert _run(_full_source(tmp_path), output) == 0
    assert sorted(path.name for path in output.iterdir()) == list(CONTRAST_REPORT_FILES)


def test_a_bad_source_is_one_line_on_stderr_and_no_directory(tmp_path: Path, capsys):
    output = tmp_path / "contrast"
    assert _run(tmp_path / "absent", output) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(captured.err.strip().splitlines()) == 1
    assert "analyze-within-image-contrast" in captured.err
    assert not output.exists()


def test_a_held_out_source_is_refused(tmp_path: Path, capsys):
    source = write_source_bundle(
        tmp_path / "source", image_ids=range(1, 251), partition="held_out"
    )
    output = tmp_path / "contrast"
    assert _run(source, output) == 2
    assert "tuning" in capsys.readouterr().err
    assert not output.exists()


def test_a_wrong_sized_source_is_refused(tmp_path: Path, capsys):
    source = write_source_bundle(tmp_path / "source", image_ids=range(1, 200))
    output = tmp_path / "contrast"
    assert _run(source, output) == 2
    assert "250" in capsys.readouterr().err
    assert not output.exists()


def test_a_finished_bundle_is_refused_and_left_untouched(tmp_path: Path, capsys):
    source = _full_source(tmp_path)
    output = tmp_path / "contrast"
    assert _run(source, output) == 0
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    assert _run(source, output) == 2
    after = {path.name: path.read_bytes() for path in output.iterdir()}
    assert before == after
    assert "exists" in capsys.readouterr().err.lower()


def test_the_source_bundle_is_left_byte_identical(tmp_path: Path):
    source = _full_source(tmp_path)
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    assert _run(source, tmp_path / "contrast") == 0
    after = {path.name: path.read_bytes() for path in source.iterdir()}
    assert before == after


def test_the_published_corruption_command_still_writes_its_own_files(tmp_path: Path):
    """Adding a consumer must not change the producer."""
    from tests.scene_uncertainty.test_corruption_integration import (
        CORRUPTION_REPORT_FILES,
        _analyze,
    )
    from tests.scene_uncertainty.decile_test_utils import write_decile_artifacts

    artifacts = write_decile_artifacts(tmp_path)
    output = tmp_path / "corruption"
    assert _analyze(artifacts, output) == 0
    assert sorted(path.name for path in output.iterdir()) == list(CORRUPTION_REPORT_FILES)
```

Add to `tests/scene_uncertainty/test_cli.py`:

```python
def test_the_contrast_subcommand_requires_source_and_output(capsys):
    from src.scene_uncertainty.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["analyze-within-image-contrast", "--source", "a"])
    with pytest.raises(SystemExit):
        parser.parse_args(["analyze-within-image-contrast", "--output", "b"])
    args = parser.parse_args(
        ["analyze-within-image-contrast", "--source", "a", "--output", "b"]
    )
    assert args.command == "analyze-within-image-contrast"
    assert str(args.source) == "a"
    assert str(args.output) == "b"


def test_the_contrast_subcommand_has_no_partition_flag():
    """The tuning partition is the only legal input, so there is nothing to choose."""
    from src.scene_uncertainty.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "analyze-within-image-contrast", "--source", "a",
            "--output", "b", "--partition", "held_out",
        ])
```

- [ ] **Step 2: Run the tests and read the failures**

```bash
$UE_PY -m pytest tests/scene_uncertainty/test_contrast_integration.py -v
```

Expected: `argparse` rejects the unknown subcommand and `main` exits 2 with a usage error.

- [ ] **Step 3: Add the subcommand**

In `cli.py`, beside the existing `corruption = add(...)` block:

```python
    contrast = add(
        "analyze-within-image-contrast",
        "Score corruption from one image using one percentile range against another.",
        _CONTRAST_EPILOG,
    )
    contrast.add_argument("--source", required=True, type=Path,
                          help="completed analyze-corruption-sensitivity output directory")
    contrast.add_argument("--output", required=True, type=Path,
                          help="within-image contrast report directory; must not exist")
```

Write `_CONTRAST_EPILOG` in the voice of `_CORRUPTION_EPILOG`: what the command reads, that it reads nothing else, that the tuning partition is the only input, that anchored and differential are two verdicts, and that the output ranks rather than calibrates.

- [ ] **Step 4: Add the pipeline handler**

In `pipeline.py`:

```python
def command_analyze_within_image_contrast(args) -> None:
    """The corruption-sensitivity bundle read back as within-image contrasts.

    `except ValueError` for the reason the sibling above gives: `ContrastInputError` and
    `ContrastAnalysisError` are both `ValueError` and neither is a `PipelineError`, and so are
    the plain `ValueError`s `contrast_scores` raises for a negative or non-finite distance.
    Catching `ValueError` is what turns all of them into the one line `cli.main` prints.

    What that deliberately does not swallow: `FileExistsError` is an `OSError`, so the rerun
    refusal passes through to the handler in `cli.main` that already knows it, and
    `write_contrast_report`'s file-set check raises `RuntimeError`, which also passes through --
    a bundle that staged the wrong files is a bug in this package, not an operator's mistake.
    """
    try:
        inputs = load_contrast_inputs(args.source)
        rows, fits = build_contrast_rows(inputs)
        diagnostics = build_anchor_diagnostics(inputs)
        candidates = summarize_contrast_candidates(
            rows, expected_image_count=inputs.provenance["image_count"]
        )
        controls = summarize_contrast_candidates(
            reference_control_rows(rows),
            expected_image_count=inputs.provenance["image_count"],
        )
        attach_controls(candidates, controls, rows)
        ranking = rank_contrast_candidates(candidates)
        write_contrast_report(
            args.output, inputs=inputs, rows=rows, fits=fits, diagnostics=diagnostics,
            candidates=candidates, controls=controls, ranking=ranking,
        )
    except ValueError as error:
        raise PipelineError(f"Cannot analyze within-image contrast: {error}") from error
```

Register it in `COMMANDS` under `"analyze-within-image-contrast"`.

**Do not omit the `except` clause.** The sibling command shipped without one in an earlier draft of the previous plan, and a `DecileAnalysisError` — a `ValueError` that is not a `PipelineError` — escaped as a twenty-frame traceback with exit code 1 instead of one line and exit code 2.

- [ ] **Step 5: Run the tests and verify they pass**

```bash
$UE_PY -m pytest tests/scene_uncertainty/test_contrast_integration.py tests/scene_uncertainty/test_cli.py -v
```

Expected: 10 new tests passed, and every existing `test_cli.py` test still passing.

- [ ] **Step 6: Document the command in `README.md`**

Add a block beside the existing `analyze-corruption-sensitivity` one, using the same `$UE_PY`/`$OUT` convention the file already establishes (`OUT=output/scene_uncertainty/pilot`, README.md:189 — keep the `pilot` segment; a block that drops it does not match the pipeline the rest of the file documents):

```bash
PYTHONPATH=$PWD $UE_PY tools/scene_uncertainty.py analyze-within-image-contrast \
  --source $OUT/reports/corruption_sensitivity_raw_k5 \
  --output $OUT/reports/within_image_contrast
```

State in prose: it reads only a finished corruption-sensitivity bundle, it runs on a CPU in minutes, `--output` must not exist, and the result ranks corruption rather than estimating its probability. `PYTHONPATH=$PWD` is required because `tools/scene_uncertainty.py` cannot import `src` on its own — that is a pre-existing repository bug affecting the already-documented commands equally, and **fixing it is outside this plan**.

- [ ] **Step 7: Mutation check**

1. Delete the `except ValueError` clause → `test_a_bad_source_is_one_line_on_stderr_and_no_directory` must fail (exit 1, traceback, multi-line stderr).
2. Change `except ValueError` to `except ContrastInputError` → `test_a_wrong_sized_source_is_refused` must still pass, but a negative-distance source must now traceback. Add that source to the test file and confirm.
3. Remove one entry from `DETONATION_SITES`'s use here → `test_no_model_bank_or_knn_entry_point_is_reachable` still passes. That is expected: this test proves reachability, not coverage. The coverage claim is `test_the_detonators_reach_the_definition_and_every_binding_of_it` in the sibling module, which this one imports from rather than re-deriving.

- [ ] **Step 8: Commit**

```bash
git add src/scene_uncertainty/cli.py src/scene_uncertainty/pipeline.py README.md \
        tests/scene_uncertainty/test_contrast_integration.py \
        tests/scene_uncertainty/test_cli.py
git commit -m "feat: add the analyze-within-image-contrast command"
```

---

## Task 10: Full suite, and the real 250-image run

**Files:** none created. This task runs the experiment the plan exists to enable.

**Interfaces:** consumes the finished command from Task 9.

**Run every test on alienware2.** The local GPU is frequently at 96–98% memory from another process, which makes `tests/scene_uncertainty/test_knn.py` fail with `CUBLAS_STATUS_ALLOC_FAILED` at `knn.py:17` for reasons that have nothing to do with this branch. Those failures are environmental and must not be reported as results. alienware2's GPU is free.

| | Local | alienware2 |
|---|---|---|
| Repo | `/home/yuchen/YuchenZ/UE/philip_sa` | `~/YuchenZ/UE/philip_sa-corruption-run` (on `dev_tue`) |
| Python | `/home/yuchen/miniconda3/envs/UE/bin/python` | `/home/alienware2/miniconda3/envs/UE_yuchen/bin/python` |
| Source bundle | — | `~/YuchenZ/UE/philip_sa-scene-unc/output/scene_uncertainty/pilot/reports/corruption_sensitivity_raw_k5` |

- [ ] **Step 1: Run the whole suite locally and note what fails**

```bash
$UE_PY -m pytest tests/scene_uncertainty -q 2>&1 | tail -20
```

Record the result. Any failure outside `test_knn.py` is a real failure and must be fixed before continuing. `git diff --stat dev_tue -- src/scene_uncertainty/knn.py tests/scene_uncertainty/test_knn.py` must be empty; if it is not, the kNN failures are yours.

- [ ] **Step 2: Ship the branch to alienware2**

```bash
BRANCH=$(git branch --show-current)
git bundle create /tmp/contrast.bundle dev_tue..$BRANCH
scp /tmp/contrast.bundle alienware2:/tmp/contrast.bundle
ssh alienware2 "cd ~/YuchenZ/UE/philip_sa-corruption-run && \
  git fetch /tmp/contrast.bundle '+refs/heads/*:refs/remotes/local/*' && \
  git checkout -B $BRANCH local/$BRANCH && git log --oneline -1"
```

- [ ] **Step 3: Run the whole suite on alienware2**

```bash
ssh alienware2 "cd ~/YuchenZ/UE/philip_sa-corruption-run && \
  /home/alienware2/miniconda3/envs/UE_yuchen/bin/python -m pytest tests/scene_uncertainty -q 2>&1 | tail -20"
```

Expected: every test passes, including `test_knn.py`. If a test fails here, it is a real failure — fix it and repeat from Step 1.

- [ ] **Step 4: Run the real experiment**

```bash
ssh alienware2 "cd ~/YuchenZ/UE/philip_sa-corruption-run && \
  PYTHONPATH=\$PWD /home/alienware2/miniconda3/envs/UE_yuchen/bin/python \
    tools/scene_uncertainty.py analyze-within-image-contrast \
    --source ~/YuchenZ/UE/philip_sa-scene-unc/output/scene_uncertainty/pilot/reports/corruption_sensitivity_raw_k5 \
    --output ~/YuchenZ/UE/philip_sa-corruption-run/output/within_image_contrast \
    && echo EXIT_OK"
```

Expected: exit 0, nine files, no stderr. If the paired bootstrap makes this take more than about fifteen minutes, check that `_macro_from_draws` is batching with `rankdata(..., axis=1)` and not looping per draw.

- [ ] **Step 5: Read the result and record it**

```bash
ssh alienware2 "cat ~/YuchenZ/UE/philip_sa-corruption-run/output/within_image_contrast/easy-report.md"
mkdir -p ~/YuchenZ/UE/within-image-contrast-run-record
scp -r alienware2:~/YuchenZ/UE/philip_sa-corruption-run/output/within_image_contrast/ \
    ~/YuchenZ/UE/within-image-contrast-run-record/
```

Read the report and answer, in the ledger, each of these separately. Do not merge them into one verdict.

**Anchored verdict.** Is any anchored arm's stability-to-spread ratio below 1.0 at every severity 1–5? Does a contrast on that arm beat both its inputs and its confidence twin, with a bootstrap interval above zero, and a severity-1 AUROC at least matching its raw responsive control?

**Differential verdict.** Does the signed raw gap or relative gap on a differential arm beat both inputs and its twin, with a bootstrap interval above zero — and are its severity-1 and severity-2 AUROCs above **0.538** and **0.570**, the best the completed deployment analysis achieved anywhere? Those two maxima come from `quintile_80_100`, a range no arm here uses, so they are an outside bar rather than a restatement of this design's own inputs.

A differential arm that clears its bar is a candidate for a held-out test and nothing more. Its tuning macro AUROC was selected on this data and may not be reported as its performance.

- [ ] **Step 6: Final acceptance checklist**

Confirm each, with the command or file that shows it:

1. The whole suite passes on alienware2, including `test_knn.py`.
2. `analyze-within-image-contrast` publishes exactly the nine declared files and nothing else.
3. `summary.json` names four arms with correct `family` and `declared_before_data` values.
4. `summary.json` holds 21 final fits — twelve persistence, nine confidence.
5. `candidate_metrics.csv` holds 45 persistence candidates and 36 confidence twins — 45 because the `combined` arm has no relative gap — and no cell holds a dictionary.
6. `anchor_diagnostics.csv` covers all 21 configurations, differential arms included.
7. Every persistence candidate carries a raw responsive comparison, a raw reference comparison, a twin comparison, and a `confidence_redundant` flag.
8. Both differential arms report the same `twin_arm`.
9. The bootstrap is reproducible: rerunning the command into a second `--output` gives byte-identical `candidate_metrics.csv`.
10. The easy report contains "no held-out images were used", contains no probability language, and states the selection debt for any differential result it reports.
11. The source corruption-sensitivity bundle is byte-identical before and after the run.
12. `analyze-corruption-sensitivity` and `analyze-confidence-deciles` still publish their own file sets unchanged.
13. The detonation test passes — no model, bank, extractor, or kNN entry point is reachable.

For item 9:

```bash
ssh alienware2 "cd ~/YuchenZ/UE/philip_sa-corruption-run && \
  PYTHONPATH=\$PWD /home/alienware2/miniconda3/envs/UE_yuchen/bin/python \
    tools/scene_uncertainty.py analyze-within-image-contrast \
    --source ~/YuchenZ/UE/philip_sa-scene-unc/output/scene_uncertainty/pilot/reports/corruption_sensitivity_raw_k5 \
    --output /tmp/contrast_rerun && \
  cmp output/within_image_contrast/candidate_metrics.csv /tmp/contrast_rerun/candidate_metrics.csv \
  && echo IDENTICAL"
```

- [ ] **Step 7: Finish the branch**

Use `superpowers:finishing-a-development-branch`. The base branch is `dev_tue`.

---

## Self-review notes

Recorded so an executor knows these were checked rather than assumed.

**Spec coverage.** Every spec section maps to a task: Goal and Decisions → Task 1's arm table; Evidence → Task 10's acceptance bars; Inference interpretation → Task 4's row construction; Source artifacts and Input requirements → Task 1; Fixed arms → Task 1; Four score methods → Task 2; Robust clean relationship and Leakage-safe tuning evaluation → Tasks 2 and 4; Anchor diagnostics → Task 3, assembled in Task 4; Per-image trend metrics, Candidate orientation and AUROC → Task 5; Ranking, Raw reference control, Confidence-only redundancy control, Paired image bootstrap → Task 6; Figures → Task 7; Outputs and Error handling → Task 8; Testing strategy → distributed, with Integration isolation in Task 9; Success criteria → Task 10 Step 5.

**Known first-draft test gaps, each with a step that closes it.** Task 3 mutation 6 (an equal-error fixture for `predictive`), Task 5 mutations 5 and 7 (`median_absolute_spearman` and the `dominant_direction_fraction` denominator), Task 6 mutations 2 and 8 (opposite twin orientation, and `confidence_redundant` on an exact tie), Task 7 mutation 1 (panel plan derived from data), Task 8 mutations 3 and 4 (`KeyboardInterrupt` cleanup, and `axis_limits` recorded rather than recomputed). These are written into the plan as work, not as warnings: a mutation step that finds nothing has not been run properly.

**Type consistency.** `Arm.pair_name` (Task 1) is what Task 4's `_signal_plan` and Task 6's `_PAIR_BY_ARM` both key on. `CONTRAST_ROW_KEY` (Task 4) is what Task 5's `summarize_contrast_candidates` validates and what Task 6's `_curves` filters on. `severity_statistics` (Task 5) is what Task 7's anchor and contrast panels read and what Task 8's `SEVERITY_STATISTIC_COLUMNS` flattens. `write_contrast_plots`'s return value (Task 7) is what Task 8 records verbatim as `axis_limits`.

**One thing this plan does not do.** It does not touch the held-out partition, and no task produces a held-out number. If the differential arm clears its bar, the held-out evaluation is a separate spec, a separate plan, and a decision to make with the tuning conclusion in hand.
