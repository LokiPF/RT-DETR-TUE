# Clean Branch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a branch `clean` whose package `degradation_monitor` holds only the six COCO baselines and the two-axis method, with the retired work in a read-only `archive/`. It reproduces the existing numbers from a converted run folder.

**Architecture:** a strangler refactor.
- First, archive the retired methods.
- Then move each clean module into the new package with `git mv`, rewriting its importers in the same commit, so the old orchestration keeps working and the suite stays green.
- Then write the new run-folder layer, the stages, the report and the CLI beside the old orchestration.
- Delete the old orchestration only when the new one passes.
- Last, convert the stored outputs and prove equivalence against the old results.

**Tech Stack:** Python 3.11 (`tomllib`), PyTorch, NumPy, SciPy, scikit-learn, pycocotools, uq-detr, imagecorruptions 1.1.2, pytest.

**Spec:** `docs/superpowers/specs/2026-10-02-clean-branch-design.md`

## Global Constraints

- **Refactoring only:** no computation changes.
  - From the converted run folder, the clean branch reproduces the six baselines' per-condition AUROC, AUPR and FPR95 in `docs/results/coco-baselines/separation.csv` to within about 1e-12.
  - It also reproduces the old confirmation report (`docs/results/conv-tu-conditioned/summary.json`) to within float rounding.
- **Package name:** `degradation_monitor`.
- **Metrics:** detection only. AUROC, AUPR and FPR95 per condition, severity and family, with paired bootstrap intervals, plus each condition's mAP as context. LRP, ρ within and AURC are removed. Per-image AP stays only because ContrastiveConf fits its λ on it.
- **Fixed method choices** are constants in `degradation_monitor/method/`, never in the config: k = 50, the stage-4 key, stages 1–3, the top 1%, and 2,000 bank + 500 z-statistics images.
- **Archive:** a read-only snapshot. Nothing in `archive/` is imported, run or collected by pytest.
- **`requirements.txt`** lists exactly torch, torchvision, numpy, pillow, scipy, scikit-learn, pycocotools, uq-detr and imagecorruptions==1.1.2.
- **Never without asking the user:** merge, push, or delete `runs/coco-baselines/`.
- **No GPU run.** Run every test with `CUDA_VISIBLE_DEVICES=` (empty).
- **Commit endings:** end every commit message with these two lines, as their own `-m` arguments:
  - `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
  - `Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU`
- **Test command** for every task: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`, run from the worktree root. It collects `tests/` only (Task 1's `pyproject.toml`).

## Review Focus

1. **A stage run against a run folder made with another protocol** (another seed, limit or checkpoint) must refuse, not mix results. Test: `test_a_run_folder_refuses_another_protocol` in Task 9.
2. **The conversion must refuse an incomplete old folder** (method pass not finished, or a score folder missing images), and must refuse when `runs/coco/` already exists. Tests: `test_conversion_refuses_an_incomplete_method_pass` and `test_conversion_refuses_an_existing_target` in Task 14.
3. **A config file with a misspelled or missing key** must fail with the key's name, not be silently ignored. Test: `test_settings_reject_unknown_and_missing_keys` in Task 9.
4. **The report on a run folder where a baseline was never scored** (no DisCoPatch or no activation folder) must still report the rest. The headline decision then says "unavailable". Test: `test_report_without_the_activation_monitors_says_the_headline_is_unavailable` in Task 12.
5. **A parallel bootstrap must give exactly the same intervals as the serial one.** Test: `test_bootstrap_gives_the_same_intervals_with_worker_processes` in Task 6.

## File structure (end state)

```
degradation_monitor/
  __init__.py, __main__.py, cli.py, settings.py, runs.py, corruptions.py
  datasets/__init__.py, datasets/coco.py
  detector/__init__.py, detector/model.py, detector/postprocess.py, detector/taps.py, detector/rtdetrv2/ (vendored, from src/)
  baselines/__init__.py, saod.py, contrastive_conf.py, knn.py, discopatch.py, hashemi.py, activation_cdf.py
  method/__init__.py, statistics.py, reference.py, scores.py
  evaluation/__init__.py, metrics.py, report.py
  stages/__init__.py, common.py, baselines.py, method.py, report.py
configs/coco.toml
scripts/convert_runs.py
archive/README.md, archive/decoder_fingerprint/..., archive/conv_tu/...
tests/fakes.py, tests/test_package.py, tests/test_corruptions.py, tests/test_runs.py, tests/test_settings.py,
tests/test_cli.py, tests/test_convert_runs.py, tests/test_equivalence.py,
tests/datasets/test_coco.py, tests/detector/{test_model,test_postprocess,test_taps}.py,
tests/baselines/{test_saod,test_contrastive_conf,test_knn,test_discopatch,test_hashemi,test_activation_cdf}.py,
tests/method/{test_statistics,test_reference,test_scores}.py, tests/evaluation/{test_metrics,test_report}.py,
tests/stages/{test_baselines,test_method,test_report}.py
docs/README.md, docs/archive/...
pyproject.toml (pytest configuration only), requirements.txt, README.md
```

**Where each old file goes** (`du` = `differential_uncertainty`):

| Old | New | How |
|---|---|---|
| `src/` | `degradation_monitor/detector/rtdetrv2/` | `git mv` (Task 4) |
| `du/extraction.py` | `detector/model.py` | `git mv` and strip the extractor; full old copy into the archive (Tasks 2, 4) |
| `du/baselines/detector.py` | `detector/taps.py` | `git mv`, plus the early-channel taps from `du/convtu/tap.py` (Task 4) |
| `du/baselines/scores.py` | `baselines/contrastive_conf.py`, plus new `saod.py`, `knn.py`, `detector/postprocess.py` | `git mv` and split (Tasks 4, 7) |
| `du/baselines/protocol.py` | `corruptions.py` (+ `du/corruptions/imagecorruptions.py`), with the COCO parts in `datasets/coco.py` | `git mv` (Task 5) |
| `du/baselines/coco_quality.py` | `datasets/coco.py` | `git mv`, LRP removed (Tasks 3, 5) |
| `du/baselines/metrics.py` | `evaluation/metrics.py` | `git mv`; harm removed, λ fitting to `contrastive_conf.py` (Tasks 3, 6, 7) |
| `du/baselines/{discopatch,hashemi,activation_cdf}.py` | `baselines/` | `git mv` (Task 7) |
| `du/convtu/channels.py` | `method/statistics.py` (+ `reference.py`, `scores.py`) | `git mv` and split (Task 8) |
| `du/convtu/conditioned.py` | `method/scores.py` | `git mv` (Task 8) |
| `du/baselines/report.py` + `du/convtu/confirmation.py` | `evaluation/report.py` | rewritten (Task 12) |
| `du/baselines/pipeline.py` + `du/convtu/pipeline.py` | `stages/` | rewritten; old files deleted (Tasks 10, 11, 13) |
| `du/cli.py`, `du/__main__.py` | `degradation_monitor/cli.py`, `__main__.py` | rewritten (Task 13) |
| `du/{benchmark,bank,scoring,persistence,reporting,config,evaluation}.py`, `du/corruptions/__init__.py`, `pretrained_weights/` | `archive/decoder_fingerprint/` | `git mv` (Task 2) |
| `du/convtu/{graph,features,report,tap}.py` and the old `du/convtu/pipeline.py` | `archive/conv_tu/` | `git mv` or snapshot copy (Task 2) |

---

### Task 1: Test configuration and the empty package

The suite must collect only `tests/`, and must import tests whose basenames repeat in different folders. During the transition, the old tests in `tests/differential_uncertainty/` still import their helper `convtu_fakes` by name.

**Files:**
- Create: `pyproject.toml`
- Create: `degradation_monitor/__init__.py`, and an empty package marker `__init__.py` in each of `degradation_monitor/datasets/`, `detector/`, `baselines/`, `method/`, `evaluation/`, `stages/`
- Test: `tests/test_package.py`

**Interfaces:**
- Produces: the importable package `degradation_monitor` with `__version__ = "2.0.0"`, and the pytest configuration that later tasks rely on.

- [ ] **Step 1: Create the worktree** (superpowers:using-git-worktrees).

  Base it on `fingerprint_bank` after the confirmation results are committed there: `git worktree add .worktrees/clean -b clean fingerprint_bank`, then `cd .worktrees/clean`. Every later command runs from this worktree's root.

- [ ] **Step 2: Write the failing test** `tests/test_package.py`:

```python
import degradation_monitor


def test_the_package_is_importable_and_versioned():
    assert degradation_monitor.__version__ == "2.0.0"
```

- [ ] **Step 3: Run it to see it fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/test_package.py`
Expected: FAIL, `ModuleNotFoundError: No module named 'degradation_monitor'`.

- [ ] **Step 4: Create `pyproject.toml`**:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--import-mode=importlib"
# "tests/differential_uncertainty" is removed again in Task 13, with the last old test.
pythonpath = [".", "tests", "tests/differential_uncertainty"]
```

- [ ] **Step 5: Create the package.**

`degradation_monitor/__init__.py`:

```python
"""Image-level corruption detection inside a frozen RT-DETRv2-R18.

Six published baselines and the two-axis score of the early backbone channels: corruptions either flatten the
channels (fog, contrast, blur) or shift their level (noise), judged against clean training scenes like the image.
"""

__version__ = "2.0.0"
```

Each sub-package marker gets one docstring line:
- `datasets/__init__.py`: `"""Datasets: clean reference images, the evaluation order and folds, and ground truth. Only COCO so far."""`
- `detector/__init__.py`: `"""The frozen RT-DETRv2-R18: building, loading, preprocessing, post-processing and forward hooks."""`
- `baselines/__init__.py`: `"""The six published baselines, one module each."""`
- `method/__init__.py`: `"""Our method: per-channel level and peak share of the early stages, judged against similar clean scenes."""`
- `evaluation/__init__.py`: `"""Separation metrics, bootstrap intervals and the report."""`
- `stages/__init__.py`: `"""The runnable, resumable steps."""` (Task 10 fills it in.)

- [ ] **Step 6: Run the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: every test passes and 2 are skipped, as on `fingerprint_bank`, plus the new one: "297 passed, 2 skipped" (296 old + 1 new). The import-mode change must not break the old tests. If any old test fails under `--import-mode=importlib`, the cause is a helper import, so add its folder to `pythonpath`.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml degradation_monitor tests/test_package.py
git commit -q -m "build: the degradation_monitor package and a pytest configuration that collects tests/ only" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

### Task 2: Archive the decoder-query fingerprint method

The August method (persistence fingerprints of the decoder queries) is retired. Its modules move into `archive/decoder_fingerprint/` unchanged. The three maintained files that still touch it get trimmed, and the archive keeps their full old versions:
- `extraction.py` (its `RTDETRExtractor`);
- `baselines/metrics.py` (its `binary_auroc` import from `evaluation.py`);
- `cli.py` (its `benchmark-coco` command).

The conv-TU pilot is archived later, in Task 13, together with the old orchestration that still imports it.

**Files:**
- Move to `archive/decoder_fingerprint/`:
  - `differential_uncertainty/{benchmark,bank,scoring,persistence,reporting,config,evaluation}.py`;
  - `differential_uncertainty/corruptions/__init__.py`, as `corruptions/__init__.py`;
  - `pretrained_weights/rtdetrv2_r18vd_120e_coco_tue/`, as `pretrained_weights/rtdetrv2_r18vd_120e_coco_tue/`;
  - `tests/differential_uncertainty/{test_bank,test_benchmark,test_config,test_evaluation,test_persistence,test_reporting,test_scoring,test_corruptions}.py`, as `tests/`.
- Snapshot copies into `archive/decoder_fingerprint/`, taken before any edit:
  - `extraction.py`, `cli.py`, `corruptions/imagecorruptions.py`;
  - `tests/test_extraction.py`, `tests/test_cli.py`.
- Modify: `differential_uncertainty/extraction.py`, `differential_uncertainty/baselines/metrics.py`, `differential_uncertainty/cli.py`.
- Modify tests: `tests/differential_uncertainty/test_extraction.py`, `test_cli.py`, `test_baselines_metrics.py`.
- Create: `archive/README.md`.

**Interfaces:**
- Produces:
  - `differential_uncertainty.baselines.metrics.binary_auroc(clean, corrupted) -> float`, now defined there, with the same behaviour as before;
  - `differential_uncertainty.extraction` with only `build_fixed_detector`, `checkpoint_state`, `load_frozen_detector`, `resize_image`, `prepare_image`;
  - `differential_uncertainty.cli` with only `baselines-coco`.

- [ ] **Step 1: Record the starting commit and take the snapshots**

```bash
git rev-parse --short HEAD                     # note this hash: archive/README.md names it in Step 7
mkdir -p archive/decoder_fingerprint/corruptions archive/decoder_fingerprint/tests archive/decoder_fingerprint/pretrained_weights
git show HEAD:differential_uncertainty/extraction.py > archive/decoder_fingerprint/extraction.py
git show HEAD:differential_uncertainty/cli.py > archive/decoder_fingerprint/cli.py
git show HEAD:differential_uncertainty/corruptions/imagecorruptions.py > archive/decoder_fingerprint/corruptions/imagecorruptions.py
git show HEAD:tests/differential_uncertainty/test_extraction.py > archive/decoder_fingerprint/tests/test_extraction.py
git show HEAD:tests/differential_uncertainty/test_cli.py > archive/decoder_fingerprint/tests/test_cli.py
```

- [ ] **Step 2: Move the retired modules, weights and tests**

```bash
for m in benchmark bank scoring persistence reporting config evaluation; do git mv differential_uncertainty/$m.py archive/decoder_fingerprint/$m.py; done
git mv differential_uncertainty/corruptions/__init__.py archive/decoder_fingerprint/corruptions/__init__.py
git mv pretrained_weights/rtdetrv2_r18vd_120e_coco_tue archive/decoder_fingerprint/pretrained_weights/rtdetrv2_r18vd_120e_coco_tue
for t in test_bank test_benchmark test_config test_evaluation test_persistence test_reporting test_scoring test_corruptions; do git mv tests/differential_uncertainty/$t.py archive/decoder_fingerprint/tests/$t.py; done
```

`differential_uncertainty/corruptions/` is now a namespace package holding only `imagecorruptions.py`. `baselines/protocol.py` still imports it as `..corruptions.imagecorruptions`, which works without an `__init__.py`.

- [ ] **Step 3: Write the failing test.**

Append to `tests/differential_uncertainty/test_baselines_metrics.py`. These two tests move here from the archived `test_evaluation.py`:

```python
from differential_uncertainty.baselines.metrics import binary_auroc


def test_binary_auroc_is_tie_correct_and_uses_larger_as_corrupted():
    assert binary_auroc([0, 1], [1, 2]) == pytest.approx(0.875)
    assert binary_auroc([1, 2], [0, 1]) == pytest.approx(0.125)


@pytest.mark.parametrize("clean, corrupted", [
    ([], [1]), ([1], []), ([float("nan")], [1]), ([1], [float("inf")]),
    ([[1]], [2]), (1, [2]),
])
def test_binary_auroc_rejects_nonfinite_empty_or_non_vector_inputs(clean, corrupted):
    with pytest.raises(ValueError, match="one-dimensional"):
        binary_auroc(clean, corrupted)
```

If the file has no `import pytest` at its top, add it.

- [ ] **Step 4: Run it to see it fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/differential_uncertainty/test_baselines_metrics.py`
Expected: an ERROR at collection, `ModuleNotFoundError: No module named 'differential_uncertainty.evaluation'`. `metrics.py` still imports the moved module.

- [ ] **Step 5: Define `binary_auroc` in `metrics.py`.**

In `differential_uncertainty/baselines/metrics.py`, replace the line `from ..evaluation import binary_auroc` with the two functions, copied verbatim from the archived `evaluation.py`:

```python
def _score_vector(values, *, name: str) -> np.ndarray:
    try:
        source = values if isinstance(values, np.ndarray) else list(values)
        array = np.asarray(source, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a non-empty finite one-dimensional input") from error
    if array.ndim != 1 or not array.size or not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} must be a non-empty finite one-dimensional input")
    return array


def binary_auroc(clean, corrupted) -> float:
    """Return tie-correct AUROC, with larger values indicating corruption."""
    clean_values = _score_vector(clean, name="clean scores")
    corrupted_values = _score_vector(corrupted, name="corrupted scores")
    ranks = rankdata(np.concatenate((clean_values, corrupted_values)), method="average")
    positives = corrupted_values.size
    rank_sum = float(ranks[clean_values.size :].sum())
    return float(
        (rank_sum - positives * (positives + 1) / 2)
        / (clean_values.size * positives)
    )
```

Place them after the constants and before `def auroc`. `rankdata` is already imported from `scipy.stats`.

- [ ] **Step 6: Trim `extraction.py` and `cli.py`, and their tests.**

Replace `differential_uncertainty/extraction.py` with exactly its first 78 lines, without the two imports of the retired modules:

```python
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn
from torchvision.transforms.v2 import functional as vision

from src.nn.backbone.presnet import PResNet
from src.zoo.rtdetr.hybrid_encoder import HybridEncoder
from src.zoo.rtdetr.rtdetr import RTDETR
from src.zoo.rtdetr.rtdetrv2_decoder import RTDETRTransformerv2
```

The rest of that header stays as it is: `build_fixed_detector`, `_tensor_state`, `checkpoint_state`, `load_frozen_detector`, `resize_image` and `prepare_image`, each byte for byte. Delete `_padded_ids` and `class RTDETRExtractor` to the end of the file.

Replace `tests/differential_uncertainty/test_extraction.py` with its first four tests, and change its imports to:

```python
import pytest
import torch
from PIL import Image
from torch import nn

import differential_uncertainty.extraction as extraction
from differential_uncertainty.extraction import checkpoint_state, load_frozen_detector, prepare_image
```

The four kept tests, verbatim:
- `test_checkpoint_state_accepts_ema_model_and_direct_tensor_states`;
- `test_load_frozen_detector_loads_cpu_state_and_freezes`;
- `test_prepare_image_returns_rgb_float32_nchw_values`;
- `test_prepare_image_keeps_an_already_exact_rgb_image_open_until_tensorized`.

Delete everything from `class _InnerDecoder` to the end.

Replace `differential_uncertainty/cli.py` with:

```python
from __future__ import annotations

import argparse
import sys


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from error
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="COCO baselines and the conv-TU phases")
    commands = parser.add_subparsers(dest="command", required=True)
    baselines = commands.add_parser("baselines-coco")
    baselines.add_argument("--phase", required=True,
                           choices=["sanity", "bank", "test", "train-discopatch", "discopatch-scores",
                                    "hashemi-fit", "cdf-fit", "cdf-zstats", "activation-scores", "timing", "report",
                                    "convtu-calibrate", "convtu-bank", "convtu-zstats", "convtu-scores",
                                    "convtu-report", "convtu-channels",
                                    "convtu-channels-report", "convtu-means",
                                    "convtu-conditioned-report"])
    baselines.add_argument("--output", required=True)
    baselines.add_argument("--checkpoint", required=True)
    baselines.add_argument("--coco-train-images", required=True)
    baselines.add_argument("--coco-val-images", required=True)
    baselines.add_argument("--coco-annotations", required=True)
    baselines.add_argument("--discopatch-root", required=True)
    baselines.add_argument("--limit", default=None, type=positive_int)
    baselines.add_argument("--device", default="cuda:0")
    baselines.add_argument("--batch-size", default=32, type=positive_int)
    baselines.add_argument("--workers", default=9, type=nonnegative_int)
    baselines.add_argument("--epochs", default=65, type=positive_int)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from pathlib import Path
        from .baselines.pipeline import Settings, run_phase
        run_phase(args.phase, Settings(
            output=Path(args.output), checkpoint=Path(args.checkpoint),
            train_images=Path(args.coco_train_images), val_images=Path(args.coco_val_images),
            annotations=Path(args.coco_annotations), discopatch_root=Path(args.discopatch_root),
            limit=args.limit, device=args.device, batch_size=args.batch_size,
            workers=args.workers, epochs=args.epochs,
        ))
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

In `tests/differential_uncertainty/test_cli.py`:
- Delete `_benchmark_parser`, `test_cli_exposes_only_the_fixed_benchmark_contract` and `test_cli_forwards_values_and_defaults`.
- Replace `test_cli_rejects_invalid_numbers` and `test_cli_prints_plain_runtime_errors` with their `baselines-coco` versions:

```python
BASE = ["baselines-coco", "--phase", "test", "--output", "out", "--checkpoint", "c.pth", "--coco-train-images", "train",
        "--coco-val-images", "val", "--coco-annotations", "ann.json", "--discopatch-root", "dcp"]


@pytest.mark.parametrize(("option", "value"), (("--batch-size", "0"), ("--workers", "-1")))
def test_cli_rejects_invalid_numbers(option, value, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main([*BASE, option, value])
    assert error.value.code == 2
    assert "must be" in capsys.readouterr().err


def test_cli_prints_plain_runtime_errors(monkeypatch, capsys):
    import differential_uncertainty.baselines.pipeline as pipeline
    monkeypatch.setattr(pipeline, "run_phase",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad input")))
    code = cli.main(BASE)
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert captured.err == "error: bad input\n"
```

- [ ] **Step 6b: Unhook the protocol tests from the archived modules.**

`tests/differential_uncertainty/test_baselines_protocol.py` imported the archived `benchmark` module and the old `apply_corruption`. Snapshot it first:

```bash
git show HEAD:tests/differential_uncertainty/test_baselines_protocol.py > archive/decoder_fingerprint/tests/test_baselines_protocol.py
```

Then:
- Delete its imports `from differential_uncertainty import benchmark` and `from differential_uncertainty.corruptions import apply_corruption`.
- Add `from PIL import ImageFilter`. `Image` is already imported from PIL.
- Replace the two tests that compared against them with direct checks of the same facts:

```python
def test_evaluation_images_use_the_old_benchmark_shuffle(tmp_path):
    for index in range(20):
        (tmp_path / f"{index:03d}.jpg").write_bytes(b"x")
    expected = sorted(tmp_path.iterdir())
    np.random.default_rng(44).shuffle(expected)
    images = protocol.evaluation_images(tmp_path, seed=44)
    assert images == expected  # the archived benchmark selected its images with this same shuffle


def test_gaussian_blur_uses_the_imagecorruptions_package_not_the_old_pil_blur():
    image = _image()
    ours = np.asarray(protocol.corrupt(image, "a.jpg", "gaussian_blur", 5))
    assert np.array_equal(ours, np.asarray(apply_imagecorruption(image, "gaussian_blur", 5)))
    old_pil_blur = np.asarray(image.convert("RGB").filter(ImageFilter.GaussianBlur(12)))  # the archived level-5 blur
    assert not np.array_equal(ours, old_pil_blur)
```

The test count is unchanged.

- [ ] **Step 7: Write `archive/README.md`**, using the hash noted in Step 1:

```markdown
# Archive: retired methods, read only

Nothing in this folder is imported, run, tested or maintained. The files are a snapshot, copied exactly as they last
were, so the project's earlier work stays readable next to the current code. To run any of it, check out the commit
named for it below, where its tests passed.

## `decoder_fingerprint/`: persistence fingerprints of the decoder queries (August 2026)

- **What it was.** Topological Uncertainty on the last decoder layer. It built a persistence fingerprint (335 sorted
  values) of the classification head for each of the 300 queries, kept a clean bank of fingerprints, and took
  each query's distance to its nearest clean fingerprints. A confidence-weighted mean turned those into one score
  per image.
  - **Code:** `benchmark.py`, `bank.py`, `scoring.py`, `persistence.py`, `reporting.py`, `config.py` and
    `evaluation.py`, the old `extraction.py` (its `RTDETRExtractor`), `corruptions/`, and the old `cli.py` (its
    `benchmark-coco` command).
  - **Data:** `pretrained_weights/` holds its class-wise Fréchet means.
- **Why it was retired.** On 100 held-out images at severities 4 and 5, its mean AUROC was 0.822. That equals top-query
  entropy (0.822) and is barely above 1 − max confidence (0.813), so it added nothing over the detector's own
  confidence.
- **Last run.** Branch `fingerprint_bank` at commit `<hash from Step 1>`: the full suite passed there (296 passed, 2
  skipped).
```

Replace `<hash from Step 1>` with the literal hash printed in Step 1.

- [ ] **Step 8: Run the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: "170 passed, 1 skipped": 171 tests in all. That is 299, less the 131 archived tests, less 2 extractor tests, less 2 CLI tests, plus the 7 `binary_auroc` tests. One of the archived tests was the CUDA-only persistence test, so one test stays skipped: DisCoPatch's CUDA-only autocast check. No test imports `differential_uncertainty.{benchmark,bank,scoring,persistence,reporting,config,evaluation}` any more: check with `grep -rn "differential_uncertainty\.\(benchmark\|bank\|scoring\|persistence\|reporting\|config\|evaluation\)\b" differential_uncertainty tests`, which must print nothing.

- [ ] **Step 9: Commit**

```bash
git add -A archive differential_uncertainty tests pretrained_weights
git commit -q -m "refactor: archive the decoder-query fingerprint method" -m "Its modules, tests and Fréchet means move to archive/decoder_fingerprint/ unchanged. extraction.py keeps only the detector loader and preprocessing, the CLI only baselines-coco, and binary_auroc moves into baselines/metrics.py." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

### Task 3: Detection metrics only

Remove per-image LRP, ρ within and the risk–coverage AURC (spec, decision 2). Per-condition mAP stays as context, and per-image AP stays for ContrastiveConf's λ. The conv-TU pilot's own report (`convtu/report.py`) computed LRP through these functions, so it is archived now, with its two phases.

**Files:**
- Modify:
  - `differential_uncertainty/baselines/coco_quality.py`;
  - `differential_uncertainty/baselines/metrics.py`;
  - `differential_uncertainty/baselines/report.py`;
  - `differential_uncertainty/convtu/pipeline.py`, removing the `convtu-report` and `convtu-channels-report` phases;
  - `differential_uncertainty/baselines/pipeline.py` (`CONVTU_PHASES`) and `differential_uncertainty/cli.py` (choices).
- Move: `differential_uncertainty/convtu/report.py` to `archive/conv_tu/report.py`, and `tests/differential_uncertainty/test_convtu_report.py` to `archive/conv_tu/tests/test_convtu_report.py`.
- Modify tests: `test_baselines_metrics.py`, `test_baselines_coco_quality.py`, `test_baselines_report.py`, `test_cli.py`.

**Interfaces:**
- Produces:
  - `headline_numbers(scores: dict, folds=None, per_fold_methods=()) -> dict`, with no `lrp` argument and separation keys only;
  - `DIFFERENCE_METRICS = ("auroc_common", "auroc_extra", "aupr_common", "aupr_extra", "fpr95_common", "fpr95_extra")`;
  - a `build_report` that writes `separation`, `aggregates`, `conditions`, `intervals`, `differences`, `knn_k` and `timing`, without `harm` or `aurc_pools`.

- [ ] **Step 1: Write the failing test.**

In `tests/differential_uncertainty/test_baselines_report.py`, replace `test_headline_numbers_include_pairwise_differences` with the version below. It calls `headline_numbers` without LRP and requires that no harm key remains:

```python
def test_headline_numbers_include_pairwise_differences_and_no_harm():
    numbers = report.headline_numbers(_scores())
    difference = numbers["saod_top3 - knn:auroc_common"]
    assert difference == pytest.approx(numbers["saod_top3:auroc_common"] - numbers["knn:auroc_common"])
    assert not any(key.endswith((":rho_within", ":rho_condition_lrp", ":aurc_all")) for key in numbers)
```

- [ ] **Step 2: Run it to see it fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/differential_uncertainty/test_baselines_report.py -k no_harm`
Expected: FAIL, `TypeError: headline_numbers() missing 1 required positional argument: 'lrp'`.

- [ ] **Step 3: Remove harm from `metrics.py` and `coco_quality.py`.**

**`metrics.py`:**
- Delete `COVERAGES`, `spearman`, `mean_within_condition_spearman`, `risk_coverage` and `aurc`.
- Change the scipy import to `from scipy.stats import ConstantInputWarning, pearsonr, rankdata`.
- The module docstring becomes `"""Separation and inference helpers; every score is oriented so higher means more degraded."""`.

**`coco_quality.py`:**
- Delete `DEFAULT_LRP_GRID`, `_ioa`, `filter_detections`, `image_lrp`, `_dataset_lrp` and `select_lrp_threshold`, and the `import uq_detr` and `import math` lines if nothing else uses them. `per_image_ap` uses `math.nan`, so keep `import math`.
- The docstring becomes `"""Detection quality on COCO: per-image AP and mAP (pycocotools)."""`.

- [ ] **Step 4: Remove harm from `report.py`.**

In `differential_uncertainty/baselines/report.py`:
- Change the `coco_quality` import to `from .coco_quality import CocoGroundTruth, coco_map, coco_results, per_image_ap`.
- Set `DIFFERENCE_METRICS = ("auroc_common", "auroc_extra", "aupr_common", "aupr_extra", "fpr95_common", "fpr95_extra")`.
- Delete `_pools`, `_nanmean_columns` and `harm_rows`.
- Replace `headline_numbers` with:

```python
def headline_numbers(scores: dict, folds=None, per_fold_methods=()) -> dict:
    """Separation aggregates used for intervals, computed identically on the full set and on each draw.

    Methods in `per_fold_methods` get fold-averaged separation numbers, matching separation_rows.
    """
    out = {}
    for method, values in scores.items():
        for group, columns in (("common", COMMON), ("extra", EXTRA)):
            if method in per_fold_methods:
                parts = [_group_separation(values[folds == f, 0], values[folds == f][:, columns].T)
                         for f in np.unique(folds)]
                auroc_value, aupr_value, fpr_value = np.mean(parts, axis=0)
            else:
                auroc_value, aupr_value, fpr_value = _group_separation(values[:, 0], values[:, columns].T)
            out[f"{method}:auroc_{group}"] = float(auroc_value)
            out[f"{method}:aupr_{group}"] = float(aupr_value)
            out[f"{method}:fpr95_{group}"] = float(fpr_value)
    for a, b in combinations(scores, 2):
        for metric in DIFFERENCE_METRICS:
            out[f"{a} - {b}:{metric}"] = out[f"{a}:{metric}"] - out[f"{b}:{metric}"]
    return out
```

In `_markdown`, delete the whole `harm = ...` / `pools = ...` / `if harm:` block. Replace the differences block with:

```python
    differences = tables.get("differences", [])
    if differences:
        lines += ["## Differences between methods", "",
                  "| Pair | Δ AUROC common | Δ AUROC extra | Δ FPR95 common |", "| --- | ---: | ---: | ---: |"]
        pairs = dict.fromkeys(r["quantity"].split(":")[0] for r in differences)
        for pair in pairs:
            lines.append(f"| {pair} | {_with_ci(intervals, f'{pair}:auroc_common')} | "
                         f"{_with_ci(intervals, f'{pair}:auroc_extra')} | {_with_ci(intervals, f'{pair}:fpr95_common')} |")
        lines.append("")
```

In `build_report`, delete everything about LRP:
- the line `threshold = select_lrp_threshold(...)`;
- the `lrp = np.full(...)` array and its double loop;
- `harm, pools = harm_rows(...)`.

Then:
- **Point estimates:** call `point = headline_numbers(scores, folds, per_fold_methods)`.
- **Bootstrap statistic:** it returns `headline_numbers({m: drawn[m] for m in scores}, folds[draw], per_fold_methods)`.
- **`conditions` rows:** drop `mean_lrp` and `images_undefined_lrp`. They become:

```python
    conditions = [{"family": f, "severity": s, "map": float(condition_map[c]),
                   **{f"mean_{m}": float(v[:, c].mean()) for m, v in scores.items()}}
                  for c, (f, s) in enumerate(protocol.CONDITIONS)]
```

- **`summary`:** drop `lrp_threshold` and `images_with_undefined_clean_lrp`.
- **`write_outputs`:** drop `"harm"` and `"aurc_pools"` from its tables.

- [ ] **Step 5: Archive the pilot report and its two phases**

```bash
mkdir -p archive/conv_tu/tests
git mv differential_uncertainty/convtu/report.py archive/conv_tu/report.py
git mv tests/differential_uncertainty/test_convtu_report.py archive/conv_tu/tests/test_convtu_report.py
```

In `differential_uncertainty/convtu/pipeline.py`:
- Delete `phase_report` and `phase_channels_report`.
- Remove their two entries, `"convtu-report"` and `"convtu-channels-report"`, from `PHASES`.

Remove the same two names from `CONVTU_PHASES` in `differential_uncertainty/baselines/pipeline.py`, from the `--phase` choices in `differential_uncertainty/cli.py`, and from the phase tuple in `test_cli_accepts_the_convtu_phases`.

- [ ] **Step 6: Update the remaining tests.**

**Delete:**
- from `test_baselines_metrics.py`: `test_spearman_drops_nan_pairs_and_needs_three_points`, `test_mean_within_condition_spearman_skips_conditions_without_defined_risk` and `test_risk_coverage_keeps_lowest_scores_ignores_undefined_risk_and_oracle_is_best`;
- from `test_baselines_coco_quality.py`: the three `image_lrp` / `select_lrp_threshold` tests;
- from `test_baselines_report.py`: `test_harm_rows_correlations_and_aurc_pools`.

**Edit in `test_baselines_report.py`:**
- `test_headline_numbers_fold_average_the_methods_asked_for`: drop the `lrp` line and call `report.headline_numbers(scores, folds, per_fold_methods=("contrastive",))`.
- `test_differences_cover_every_separation_metric_for_both_family_groups`: call `report.headline_numbers(_scores())`.
- `test_write_outputs_creates_csv_json_and_markdown`: pass the summary `{"lambda_per_fold": {"0": 5.0}, "clean_map": 0.48}` and assert `...["clean_map"] == 0.48`.
- `test_build_report_end_to_end_on_a_tiny_fixture`: the loop checks `("separation", "aggregates", "conditions", "intervals", "differences", "knn_k")`, and the last assertion becomes `assert "map" in header and "images_undefined_lrp" not in header`.
- `test_build_report_includes_the_activation_monitors_when_scored`: read `separation.csv` instead of `harm.csv`.

- [ ] **Step 7: Run the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: "160 passed, 1 skipped". That is 170, less 3 metrics tests, 3 LRP tests, 1 harm test and the 3 archived pilot-report tests. `grep -rn "lrp\|aurc\|rho_within" differential_uncertainty` prints nothing.

- [ ] **Step 8: Commit**

```bash
git add -A differential_uncertainty tests archive
git commit -q -m "refactor: detection metrics only" -m "Per-image LRP, rho within and AURC are removed from the baselines' report and metrics. Per-condition mAP stays as context, per-image AP for ContrastiveConf's lambda. The conv-TU pilot's own report, which used them, is archived with its two phases." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

### Task 4: The detector package

These move into `degradation_monitor/detector/`:
- the vendored RT-DETRv2 code;
- the loader and preprocessing;
- the forward hooks;
- RT-DETR's post-processing.

The hooks gain `EarlyChannelTaps`, the method's taps without the topology-only kernels. The old `convtu/tap.py` (with kernels) stays for the conv-TU pilot until Task 13.

**Files:**
- Move: `src/` to `degradation_monitor/detector/rtdetrv2/`.
- Move: `differential_uncertainty/extraction.py` to `degradation_monitor/detector/model.py`.
- Move: `differential_uncertainty/baselines/detector.py` to `degradation_monitor/detector/taps.py`, and add `EarlyChannelTaps`.
- Create: `degradation_monitor/detector/postprocess.py`.
- Modify:
  - `differential_uncertainty/baselines/scores.py`;
  - `differential_uncertainty/baselines/pipeline.py`;
  - `differential_uncertainty/convtu/pipeline.py`.
- Move tests:
  - `tests/differential_uncertainty/test_extraction.py` to `tests/detector/test_model.py`;
  - `tests/differential_uncertainty/test_baselines_detector.py` to `tests/detector/test_taps.py`;
  - `tests/differential_uncertainty/convtu_fakes.py` to `tests/fakes.py`.
- Create test: `tests/detector/test_postprocess.py`.

**Interfaces:**
- Produces:
  - `degradation_monitor.detector.model`: `IMAGE_SIZE = (640, 640)`, `build_fixed_detector()`, `checkpoint_state(checkpoint)`, `load_frozen_detector(checkpoint_path, device) -> nn.Module`, `resize_image(image, image_size)`, `prepare_image(image, image_size) -> Tensor (3, H, W) float32`.
  - `degradation_monitor.detector.postprocess`: `TOP_K = 100`, `checked_outputs(logits, boxes) -> (ndarray, ndarray)`, `top_detections(logits, boxes_cxcywh, image_size, top_k=TOP_K) -> (scores float32 (k,), labels int64 (k,), xyxy float32 (k, 4))`.
  - `degradation_monitor.detector.taps`:
    - `DetectorTap(checkpoint_path, device, image_size=(640, 640), hidden=False)`, with `prepare`, `forward`, `forward_hidden`, `run`, `close` and context-manager use, unchanged;
    - `QUERY_COUNT`, `CLASS_COUNT`, `POOLED_DIM`, `DECODER_SHAPE`, `STAGE_COUNT`;
    - `EARLY_STAGES = 4`, `EARLY_LAYERS`;
    - `EarlyChannelTaps(backbone)`. Calling it on a batch returns the four post-ReLU maps as float32 on the backbone's device. It also has `close()` and context-manager use.
  - `tests/fakes.py`: `FakeBackbone`, `FakeTap`, unchanged.

- [ ] **Step 1: Move the files**

```bash
mkdir -p degradation_monitor/detector tests/detector
git mv src degradation_monitor/detector/rtdetrv2
git mv differential_uncertainty/extraction.py degradation_monitor/detector/model.py
git mv differential_uncertainty/baselines/detector.py degradation_monitor/detector/taps.py
git mv tests/differential_uncertainty/test_extraction.py tests/detector/test_model.py
git mv tests/differential_uncertainty/test_baselines_detector.py tests/detector/test_taps.py
git mv tests/differential_uncertainty/convtu_fakes.py tests/fakes.py
```

- [ ] **Step 2: Point the moved code at its new home**

In `degradation_monitor/detector/model.py`:
- Add the module docstring `"""Build the fixed RT-DETRv2-R18, load its frozen COCO checkpoint, and prepare images for it."""` as the first line.
- Replace the four `from src....` imports with:

```python
from .rtdetrv2.nn.backbone.presnet import PResNet
from .rtdetrv2.zoo.rtdetr.hybrid_encoder import HybridEncoder
from .rtdetrv2.zoo.rtdetr.rtdetr import RTDETR
from .rtdetrv2.zoo.rtdetr.rtdetrv2_decoder import RTDETRTransformerv2

IMAGE_SIZE = (640, 640)  # RT-DETRv2 fixes its input size in training and at inference
```

Further edits:
- **`degradation_monitor/detector/rtdetrv2/__init__.py`:** its docstring becomes `"""Vendored RT-DETRv2 inference code (github.com/lyuwenyu/RT-DETR, Apache-2.0), trimmed to what the frozen detector needs."""`.
- **`degradation_monitor/detector/taps.py`:**
  - Replace `from ..extraction import load_frozen_detector, prepare_image` with `from .model import load_frozen_detector, prepare_image`.
  - Replace the module docstring with `"""Forward hooks on the frozen RT-DETRv2: logits, boxes and the pooled last stage; the hidden layers the activation monitors read; and the early-channel inputs our method reads."""`.
- **`tests/fakes.py`:** its docstring becomes `"""Small stand-ins for the RT-DETRv2 backbone and the detector tap."""`.

Rewrite the imports of the old package and the tests:

```bash
sed -i 's/^from \.\.extraction import prepare_image$/from degradation_monitor.detector.model import prepare_image/; s/^from \.detector import DetectorTap$/from degradation_monitor.detector.taps import DetectorTap/' differential_uncertainty/baselines/pipeline.py
sed -i 's/^from \.\.extraction import load_frozen_detector, prepare_image$/from degradation_monitor.detector.model import load_frozen_detector, prepare_image/' differential_uncertainty/convtu/pipeline.py
sed -i 's/^import differential_uncertainty\.extraction as extraction$/import degradation_monitor.detector.model as extraction/; s/^from differential_uncertainty\.extraction import /from degradation_monitor.detector.model import /' tests/detector/test_model.py
sed -i 's/^from differential_uncertainty\.baselines import detector$/from degradation_monitor.detector import taps as detector/' tests/detector/test_taps.py
sed -i 's/^from convtu_fakes import /from fakes import /' tests/differential_uncertainty/*.py
sed -i 's/^from differential_uncertainty\.extraction import /from degradation_monitor.detector.model import /' tests/differential_uncertainty/*.py
grep -rn "differential_uncertainty.extraction\|from \.\.extraction\|convtu_fakes\|from src\." differential_uncertainty tests degradation_monitor
```

Expected: the last `grep` prints nothing.

- [ ] **Step 3: Run the moved tests**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/detector`
Expected: all pass (4 model tests, 4 tap tests).

- [ ] **Step 4: Write the failing tests** for post-processing and the early-channel taps.

`tests/detector/test_postprocess.py`, moved from `test_baselines_scores.py`:

```python
import numpy as np
import pytest
from scipy.special import logit

from degradation_monitor.detector.postprocess import TOP_K, top_detections


def test_top_detections_ranks_query_class_pairs_and_scales_boxes_to_pixels():
    logits = logit(np.array([[0.2, 0.9], [0.7, 0.1]]))
    boxes = np.array([[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.5, 0.5]])
    top, labels, xyxy = top_detections(logits, boxes, image_size=(100, 50), top_k=3)
    assert top == pytest.approx([0.9, 0.7, 0.2], abs=1e-6)
    assert labels.tolist() == [1, 0, 0]
    assert xyxy[0] == pytest.approx([40, 15, 60, 35])
    assert xyxy[1] == pytest.approx([0, 0, 50, 25])
    assert TOP_K == 100
```

Delete `test_top_detections_ranks_query_class_pairs_and_scales_boxes_to_pixels` from `tests/differential_uncertainty/test_baselines_scores.py`.

Add these imports at the top of `tests/detector/test_taps.py`:

```python
from pathlib import Path

from fakes import FakeBackbone
from degradation_monitor.detector.model import load_frozen_detector
from degradation_monitor.detector.taps import EARLY_LAYERS, EarlyChannelTaps

CHECKPOINT = Path("/home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth")
```

Then append these tests:

```python
def test_early_channel_taps_capture_the_tensors_entering_each_stage_conv():
    backbone = FakeBackbone()
    batch = torch.rand(2, 3, 640, 640)
    with EarlyChannelTaps(backbone) as taps:
        inputs = taps(batch)
    with torch.no_grad():
        x = backbone.pool(batch)
        expected = []
        for stage in backbone.res_layers:
            entering = torch.relu(stage.blocks[0](x))
            expected.append(entering)
            x = stage.blocks[1](entering)
    assert [tuple(t.shape) for t in inputs] == [(2, 2, 16, 16), (2, 3, 8, 8), (2, 4, 4, 4), (2, 5, 2, 2)]
    for got, want in zip(inputs, expected):
        assert torch.allclose(got, want)


def test_closing_the_early_channel_taps_removes_their_hooks():
    backbone = FakeBackbone()
    taps = EarlyChannelTaps(backbone)
    taps.close()
    assert all(len(stage.blocks[1].branch2a.conv._forward_pre_hooks) == 0 for stage in backbone.res_layers)
    assert EARLY_LAYERS[0] == "res_layers.0.blocks.1.branch2a.conv"


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="needs the RT-DETRv2-R18 checkpoint")
def test_the_real_backbone_gives_the_four_early_channel_maps():
    model = load_frozen_detector(CHECKPOINT, torch.device("cpu"))
    with EarlyChannelTaps(model.backbone) as taps:
        inputs = taps(torch.rand(1, 3, 640, 640))
    assert [tuple(t.shape) for t in inputs] == [(1, 64, 160, 160), (1, 128, 80, 80), (1, 256, 40, 40), (1, 512, 20, 20)]
```

- [ ] **Step 5: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/detector`
Expected: ERRORS at collection, `ModuleNotFoundError: No module named 'degradation_monitor.detector.postprocess'` and `ImportError: cannot import name 'EARLY_LAYERS'`.

- [ ] **Step 6: Implement.**

Create `degradation_monitor/detector/postprocess.py`. The two function bodies are verbatim from `differential_uncertainty/baselines/scores.py`:

```python
"""RT-DETR post-processing: sigmoid scores and the top-k (query, class) pairs, with boxes in pixels."""
from __future__ import annotations

import numpy as np
from scipy.special import expit

TOP_K = 100


def checked_outputs(logits, boxes):
    logits = np.asarray(logits, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64)
    if logits.ndim != 2 or boxes.shape != (logits.shape[0], 4):
        raise ValueError("expected logits (queries, classes) and boxes (queries, 4)")
    if not (np.isfinite(logits).all() and np.isfinite(boxes).all()):
        raise ValueError("detector outputs must be finite")
    return logits, boxes


def top_detections(logits, boxes_cxcywh, image_size, top_k=TOP_K):
    """RT-DETR post-processing: sigmoid, then the top-k (query, class) pairs."""
    logits, boxes = checked_outputs(logits, boxes_cxcywh)
    width, height = image_size
    probs = expit(logits).reshape(-1)
    order = np.argsort(-probs, kind="stable")[: min(int(top_k), probs.size)]
    queries, labels = np.divmod(order, logits.shape[1])
    cx, cy, w, h = boxes[queries].T
    xyxy = np.stack(
        [(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height],
        axis=1,
    )
    return probs[order].astype(np.float32), labels.astype(np.int64), xyxy.astype(np.float32)
```

In `differential_uncertainty/baselines/scores.py`, delete `_checked` and `top_detections`, and add after the imports:

```python
from degradation_monitor.detector.postprocess import checked_outputs as _checked, top_detections
```

`query_detections` keeps calling `_checked`, and `pipeline.py`'s `from .scores import (... top_detections)` keeps working.

Append to `degradation_monitor/detector/taps.py`:

```python
EARLY_STAGES = 4
EARLY_LAYERS = tuple(f"res_layers.{s}.blocks.1.branch2a.conv" for s in range(EARLY_STAGES))


class EarlyChannelTaps:
    """The inputs of the first conv in the second block of each backbone stage: the four post-ReLU maps our method reads.

    For a 640 x 640 image they have shapes (64, 160, 160), (128, 80, 80), (256, 40, 40) and (512, 20, 20).
    """

    def __init__(self, backbone: torch.nn.Module):
        self.backbone = backbone
        self.device = next(backbone.parameters()).device
        convs = [backbone.res_layers[s].blocks[1].branch2a.conv for s in range(EARLY_STAGES)]
        self._inputs: list = [None] * EARLY_STAGES
        self._handles = [conv.register_forward_pre_hook(partial(self._capture, index))
                         for index, conv in enumerate(convs)]

    def _capture(self, index, _module, inputs):
        self._inputs[index] = inputs[0]

    @torch.inference_mode()
    def __call__(self, batch: torch.Tensor) -> list[torch.Tensor]:
        self._inputs = [None] * EARLY_STAGES
        self.backbone(batch.to(self.device))
        if any(value is None for value in self._inputs):
            raise RuntimeError("an early-channel input was not captured")
        return [value.float() for value in self._inputs]

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
```

`partial` is already imported in `taps.py`.

- [ ] **Step 7: Run the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: "163 passed, 1 skipped" (160 + 3 tap tests + 1 post-processing test − 1 moved scores test).

- [ ] **Step 8: Commit**

```bash
git add -A degradation_monitor differential_uncertainty tests src
git commit -q -m "refactor: the detector package: vendored RT-DETRv2, loader, post-processing and forward hooks" -m "src/ moves to degradation_monitor/detector/rtdetrv2/, the loader to detector/model.py, the detector tap to detector/taps.py, which gains the method's early-channel taps without the topology-only kernels, and top_detections to detector/postprocess.py." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

### Task 5: The corruption protocol and the COCO dataset

`baselines/protocol.py` mixes two things:
- the corruption protocol, which is the same for any dataset;
- COCO's file order and folds.

The corruption half becomes `degradation_monitor/corruptions.py`, together with the `imagecorruptions` compatibility wrapper. The COCO half joins the ground-truth code in `degradation_monitor/datasets/coco.py`, which also gets the method's train splits and a small `Coco` class: the seam a later Cityscapes module mirrors.

**Files:**
- Move: `differential_uncertainty/baselines/protocol.py` to `degradation_monitor/corruptions.py`.
- Delete: `differential_uncertainty/corruptions/imagecorruptions.py`. Its code is inlined into `corruptions.py`, and the archive has the snapshot from Task 2.
- Move: `differential_uncertainty/baselines/coco_quality.py` to `degradation_monitor/datasets/coco.py`.
- Modify:
  - `differential_uncertainty/baselines/pipeline.py`;
  - `differential_uncertainty/baselines/report.py`;
  - `differential_uncertainty/convtu/pipeline.py`;
  - `differential_uncertainty/convtu/confirmation.py`.
- Move tests:
  - `tests/differential_uncertainty/test_baselines_protocol.py` to `tests/test_corruptions.py`;
  - `tests/differential_uncertainty/test_baselines_coco_quality.py` to `tests/datasets/test_coco.py`.
- Modify tests: `test_baselines_report.py`, `test_baselines_pipeline.py`.

**Interfaces:**
- Produces:
  - `degradation_monitor.corruptions`:
    - constants `COMMON_FAMILIES`, `EXTRA_FAMILIES`, `FAMILIES`, `SEVERITIES`, `CONDITIONS` (96 pairs, clean first);
    - functions `apply_imagecorruption(image, name, severity)`, `variant_seed(image_id, family, severity)`, `corrupt(image, image_id, family, severity)`, `variants(image, image_id) -> list[ndarray]`, `load_variants(path) -> (name, list[ndarray])`, `digest(array) -> str`.
  - `degradation_monitor.datasets.coco`:
    - constants `FOLDS = 5`, `SEED = 44`, `SPLIT_IMAGES = (("reserved", 200), ("bank", 2000), ("zstats", 500))`;
    - `list_images(root)`, `evaluation_images(val_root, *, seed)`, `assign_folds(count, folds=FOLDS)`;
    - `train_splits(count, seed=SEED) -> {"reserved", "bank", "zstats": sorted index arrays}`;
    - `CocoGroundTruth`, `coco_results`, `coco_map`, `per_image_ap`;
    - `Coco(train_root, val_root, annotations, seed=SEED, limit=None)`, with `.train_images()`, `.reference_split(name)`, `.evaluation_images()`, `.folds()` and `.ground_truth()`.

- [ ] **Step 1: Move the files**

```bash
mkdir -p degradation_monitor/datasets tests/datasets
git mv differential_uncertainty/baselines/protocol.py degradation_monitor/corruptions.py
git mv differential_uncertainty/baselines/coco_quality.py degradation_monitor/datasets/coco.py
git rm -q differential_uncertainty/corruptions/imagecorruptions.py
git mv tests/differential_uncertainty/test_baselines_protocol.py tests/test_corruptions.py
git mv tests/differential_uncertainty/test_baselines_coco_quality.py tests/datasets/test_coco.py
```

- [ ] **Step 2: Write the failing tests.**

Move these three tests out of `tests/test_corruptions.py` and into `tests/datasets/test_coco.py`, replacing `protocol.` with `coco.` in them:
- `test_evaluation_images_use_the_old_benchmark_shuffle`;
- `test_evaluation_images_reject_an_empty_directory`;
- `test_folds_are_balanced_and_follow_shuffled_position`.

In `tests/test_corruptions.py`:
- Set the imports to:

```python
import numpy as np
import pytest
from PIL import Image, ImageFilter

from degradation_monitor import corruptions as protocol
from degradation_monitor.corruptions import apply_imagecorruption
```

- The alias keeps the moved tests' bodies unchanged.

In `tests/datasets/test_coco.py`:
- Replace `from differential_uncertainty.baselines import coco_quality as cq` with:

```python
from degradation_monitor.datasets import coco
from degradation_monitor.datasets import coco as cq
```

- Append:

```python
def test_train_splits_are_disjoint_seeded_and_sized(monkeypatch):
    monkeypatch.setattr(coco, "SPLIT_IMAGES", (("reserved", 2), ("bank", 3), ("zstats", 2)))
    splits = coco.train_splits(10, seed=44)
    assert [len(splits[k]) for k in ("reserved", "bank", "zstats")] == [2, 3, 2]
    assert len(set(np.concatenate(list(splits.values())).tolist())) == 7
    again = coco.train_splits(10, seed=44)
    assert all(np.array_equal(splits[k], again[k]) for k in splits)
    with pytest.raises(ValueError, match="needs 7 train images"):
        coco.train_splits(6, seed=44)


def test_train_splits_reproduce_the_stored_bank_draw():
    # the pilot drew calibration, bank and z-statistics images as order[:200], order[200:2200], order[2200:2700]
    order = np.random.default_rng(44).permutation(5000)
    splits = coco.train_splits(5000, seed=44)
    assert np.array_equal(splits["bank"], np.sort(order[200:2200]))
    assert np.array_equal(splits["zstats"], np.sort(order[2200:2700]))


def test_the_coco_dataset_gives_the_limited_seeded_order_its_folds_and_the_reference_splits(tmp_path, monkeypatch):
    for folder, count in (("train", 9), ("val", 6)):
        (tmp_path / folder).mkdir()
        for index in range(count):
            (tmp_path / folder / f"{index:03d}.jpg").write_bytes(b"x")
    monkeypatch.setattr(coco, "SPLIT_IMAGES", (("reserved", 2), ("bank", 4), ("zstats", 3)))
    dataset = coco.Coco(tmp_path / "train", tmp_path / "val", tmp_path / "ann.json", limit=4)
    assert dataset.evaluation_images() == coco.evaluation_images(tmp_path / "val", seed=44)[:4]
    assert dataset.folds().tolist() == [0, 1, 2, 3]
    bank = dataset.reference_split("bank")
    assert len(bank) == 4 and set(bank) <= set(dataset.train_images())
    assert not set(bank) & set(dataset.reference_split("zstats"))
```

The file already imports `numpy as np` and `pytest`.

- [ ] **Step 3: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/test_corruptions.py tests/datasets/test_coco.py`
Expected: FAIL or ERROR.
- `corruptions.py` still imports `..corruptions.imagecorruptions`, which no longer exists.
- `coco` has no `evaluation_images`, `train_splits` or `Coco`.

- [ ] **Step 4: Implement `corruptions.py`**

Its new header:

```python
"""The corruption protocol: clean plus 19 imagecorruptions families x severities 1-5, one seeded draw per image and condition."""
from __future__ import annotations

import hashlib
import inspect
import zlib
from pathlib import Path

import numpy as np
from PIL import Image
```

Keep, verbatim:
- `COMMON_FAMILIES`, `EXTRA_FAMILIES`, `FAMILIES`, `SEVERITIES` and `CONDITIONS`;
- `variant_seed`, `corrupt`, `variants`, `load_variants` and `digest`.

Delete `FOLDS`, `_SUFFIXES`, `list_images`, `evaluation_images` and `assign_folds`; they move to `coco.py` below. Insert the deleted wrapper's code before `variant_seed`, verbatim from `archive/decoder_fingerprint/corruptions/imagecorruptions.py`: `_NumpyCompatibility`, `_prepare_compatibility` and `apply_imagecorruption`.

- [ ] **Step 5: Implement `datasets/coco.py`**

Its new header:

```python
"""COCO 2017: clean train images, the seed-44 val order and its folds, and the ground truth the baselines need."""
from __future__ import annotations

import contextlib
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

FOLDS = 5
SEED = 44
# Disjoint seeded draws of train images. "reserved" was the archived conv-TU pilot's calibration set; the slot is kept
# so that the bank and z-statistics images stay exactly those the stored references were computed from.
SPLIT_IMAGES = (("reserved", 200), ("bank", 2000), ("zstats", 500))
_SUFFIXES = {".jpg", ".jpeg", ".png"}


def list_images(root) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"image directory does not exist: {root}")
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _SUFFIXES)


def evaluation_images(val_root, *, seed: int) -> list[Path]:
    """All val images in the old benchmark's seeded shuffle order; the order defines the folds."""
    images = list_images(val_root)
    if not images:
        raise ValueError(f"no images found in {val_root}")
    np.random.default_rng(seed).shuffle(images)
    return images


def assign_folds(count: int, folds: int = FOLDS) -> np.ndarray:
    """Fold of each image from its shuffled position: 0, 1, ..., folds - 1, 0, 1, ..."""
    if count <= 0 or folds < 2:
        raise ValueError("count must be positive and folds at least 2")
    return np.arange(count) % folds


def train_splits(count: int, seed: int = SEED) -> dict:
    """Sorted, disjoint, seeded draws of train-image indices: reserved, bank and z-statistics."""
    needed = sum(size for _, size in SPLIT_IMAGES)
    if count < needed:
        raise ValueError(f"the reference needs {needed} train images, found {count}")
    order = np.random.default_rng(seed).permutation(count)
    out, start = {}, 0
    for name, size in SPLIT_IMAGES:
        out[name] = np.sort(order[start:start + size])
        start += size
    return out
```

Keep, verbatim:
- `_quiet`, `CocoGroundTruth`, `coco_results`, `_evaluate`, `coco_map` and `per_image_ap`.

Then append:

```python
@dataclass(frozen=True)
class Coco:
    """What a stage needs from the dataset; a Cityscapes module would offer the same five methods."""
    train_root: Path
    val_root: Path
    annotations: Path
    seed: int = SEED
    limit: Optional[int] = None

    def train_images(self) -> list[Path]:
        return list_images(self.train_root)

    def reference_split(self, name: str) -> list[Path]:
        paths = self.train_images()
        return [paths[i] for i in train_splits(len(paths), self.seed)[name]]

    def evaluation_images(self) -> list[Path]:
        images = evaluation_images(self.val_root, seed=self.seed)
        return images[: self.limit] if self.limit else images

    def folds(self) -> np.ndarray:
        return assign_folds(len(self.evaluation_images()))

    def ground_truth(self) -> CocoGroundTruth:
        return CocoGroundTruth(self.annotations)
```

- [ ] **Step 6: Rewrite the importers**

```bash
for f in differential_uncertainty/baselines/pipeline.py differential_uncertainty/baselines/report.py differential_uncertainty/convtu/pipeline.py differential_uncertainty/convtu/confirmation.py tests/differential_uncertainty/test_baselines_report.py tests/differential_uncertainty/test_baselines_pipeline.py; do
  sed -i 's/protocol\.list_images/coco.list_images/g; s/protocol\.evaluation_images/coco.evaluation_images/g; s/protocol\.assign_folds/coco.assign_folds/g; s/protocol\.FOLDS/coco.FOLDS/g; s/protocol\./corruptions./g' "$f"
done
sed -i 's/^from \. import discopatch, protocol$/from degradation_monitor import corruptions\nfrom degradation_monitor.datasets import coco\nfrom . import discopatch/; s/^from \.coco_quality import /from degradation_monitor.datasets.coco import /' differential_uncertainty/baselines/pipeline.py
sed -i 's/^from \. import metrics, protocol$/from degradation_monitor import corruptions\nfrom degradation_monitor.datasets import coco\nfrom . import metrics/; s/^from \.coco_quality import /from degradation_monitor.datasets.coco import /' differential_uncertainty/baselines/report.py
sed -i 's/^from \.\.baselines import protocol$/from degradation_monitor import corruptions\nfrom degradation_monitor.datasets import coco/' differential_uncertainty/convtu/pipeline.py
sed -i 's/^from \.\.baselines import metrics, protocol$/from degradation_monitor import corruptions\nfrom degradation_monitor.datasets import coco\nfrom ..baselines import metrics/' differential_uncertainty/convtu/confirmation.py
sed -i 's/^from differential_uncertainty\.baselines import protocol, report$/from degradation_monitor import corruptions\nfrom degradation_monitor.datasets import coco\nfrom differential_uncertainty.baselines import report/' tests/differential_uncertainty/test_baselines_report.py
grep -rn "protocol\|coco_quality\|\.\.corruptions" differential_uncertainty tests --include=*.py | grep -v "^tests/test_corruptions.py"
```

Expected: the last `grep` prints nothing, except:
- the `from degradation_monitor import corruptions as protocol` alias in `tests/test_corruptions.py`, which it excludes anyway;
- words like "COCO protocol" inside docstrings, which are fine.

Then, in `test_baselines_pipeline.py`, check that `pipeline.corruptions.digest` replaced `pipeline.protocol.digest`.

- [ ] **Step 7: Run the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: "166 passed, 1 skipped" (163 + 3 new COCO tests).

- [ ] **Step 8: Commit**

```bash
git add -A degradation_monitor differential_uncertainty tests
git commit -q -m "refactor: the corruption protocol and the COCO dataset" -m "baselines/protocol.py splits into degradation_monitor/corruptions.py, which absorbs the imagecorruptions wrapper, and datasets/coco.py, which joins the ground truth with the val order, folds, the method's train splits and a small Coco class." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

### Task 6: Evaluation metrics

The separation metrics, the per-stage z-scoring and the bootstrap move into `degradation_monitor/evaluation/metrics.py`.
- The bootstrap can evaluate its draws in forked worker processes. The draws still come from one seeded generator in the same order, so the intervals are bit-identical. The full report then takes minutes rather than an hour.
- The three-metric and two-group helpers of the old report and confirmation become shared functions.
- The condition index arrays move into `corruptions.py`.

**Files:**
- Move: `differential_uncertainty/baselines/metrics.py` to `degradation_monitor/evaluation/metrics.py`.
- Modify:
  - `degradation_monitor/corruptions.py` (index arrays);
  - `differential_uncertainty/baselines/activation_cdf.py` (drop `stage_zstats` and `zscored_sum`);
  - the importers: `differential_uncertainty/baselines/{pipeline,report}.py` and `differential_uncertainty/convtu/{channels,conditioned,pipeline,confirmation}.py`.
- Move test: `tests/differential_uncertainty/test_baselines_metrics.py` to `tests/evaluation/test_metrics.py`.
- Modify test: `tests/differential_uncertainty/test_baselines_activation_cdf.py`. Its z-score test moves out.

**Interfaces:**
- Produces:
  - `degradation_monitor.corruptions`: `CORRUPTED_CONDITIONS`, `COMMON_CONDITIONS`, `EXTRA_CONDITIONS` (column index arrays into the 96 conditions).
  - `degradation_monitor.evaluation.metrics`:
    - `binary_auroc`, `auroc`, `aupr`, `fpr_at_95_tpr`;
    - `condition_aurocs(clean (n,), degraded (c, n)) -> (c,)`;
    - `group_separation(clean (n,), degraded (c, n)) -> (auroc, aupr, fpr95)`;
    - `group_aurocs(scores (images, 96), rows) -> (auroc_common, auroc_extra)`;
    - `bootstrap(statistic, n_images, samples=1000, seed=44, workers=1) -> {key: (low, high)}`;
    - `stage_zstats(stage_scores) -> (mean, std)`, `zscored_sum(stage_scores, mean, std)`;
    - until Task 7: `UQ_DETR_LAMBDA_GRID`, `fit_lambda_from_parts`, `cross_fit_lambda`.

- [ ] **Step 1: Move the module and its test**

```bash
mkdir -p tests/evaluation
git mv differential_uncertainty/baselines/metrics.py degradation_monitor/evaluation/metrics.py
git mv tests/differential_uncertainty/test_baselines_metrics.py tests/evaluation/test_metrics.py
sed -i 's/^from differential_uncertainty\.baselines import metrics as m$/from degradation_monitor.evaluation import metrics as m/; s/^from differential_uncertainty\.baselines\.metrics import binary_auroc$/from degradation_monitor.evaluation.metrics import binary_auroc/' tests/evaluation/test_metrics.py
```

- [ ] **Step 2: Write the failing tests.**

Move `test_zscored_sum_standardises_each_stage_on_clean_statistics` out of `tests/differential_uncertainty/test_baselines_activation_cdf.py` into `tests/evaluation/test_metrics.py`, replacing `cdf.` with `m.` in it. Then append:

```python
def test_bootstrap_gives_the_same_intervals_with_worker_processes():
    rng = np.random.default_rng(2)
    clean, degraded = rng.normal(0, 1, 60), rng.normal(0.5, 1, (3, 60))

    def statistic(idx):
        return {"a": m.condition_aurocs(clean[idx], degraded[:, idx]).mean(), "b": float(clean[idx].mean())}

    serial = m.bootstrap(statistic, 60, samples=40, seed=7)
    assert m.bootstrap(statistic, 60, samples=40, seed=7, workers=3) == serial


def test_group_separation_averages_the_three_metrics_over_conditions():
    rng = np.random.default_rng(3)
    clean, degraded = rng.normal(0, 1, 50), rng.normal(1, 1, (4, 50))
    auroc, aupr, fpr95 = m.group_separation(clean, degraded)
    assert auroc == pytest.approx(np.mean([m.auroc(clean, row) for row in degraded]))
    assert aupr == pytest.approx(np.mean([m.aupr(clean, row) for row in degraded]))
    assert fpr95 == pytest.approx(np.mean([m.fpr_at_95_tpr(clean, row) for row in degraded]))


def test_group_aurocs_average_the_common_and_the_extra_conditions_on_the_chosen_images():
    from degradation_monitor import corruptions
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(30, 96)) + np.linspace(0, 1, 96)[None]
    rows = np.array([0, 0, 3, 7, 12])
    common, extra = m.group_aurocs(scores, rows)
    assert common == pytest.approx(np.mean([m.auroc(scores[rows, 0], scores[rows, c]) for c in corruptions.COMMON_CONDITIONS]))
    assert extra == pytest.approx(np.mean([m.auroc(scores[rows, 0], scores[rows, c]) for c in corruptions.EXTRA_CONDITIONS]))
```

- [ ] **Step 3: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/evaluation/test_metrics.py`
Expected: FAIL, with `AttributeError`s for `stage_zstats`, `group_separation`, `group_aurocs`, `COMMON_CONDITIONS`, and `TypeError: bootstrap() got an unexpected keyword argument 'workers'`.

- [ ] **Step 4: Implement**

Append to `degradation_monitor/corruptions.py`, after `CONDITIONS`:

```python
CORRUPTED_CONDITIONS = np.arange(1, len(CONDITIONS))
COMMON_CONDITIONS = np.array([c for c, (f, _) in enumerate(CONDITIONS) if f in COMMON_FAMILIES])
EXTRA_CONDITIONS = np.array([c for c, (f, _) in enumerate(CONDITIONS) if f in EXTRA_FAMILIES])
```

In `degradation_monitor/evaluation/metrics.py`:
- Add `import multiprocessing` and `from ..corruptions import COMMON_CONDITIONS, EXTRA_CONDITIONS`.
- Move `stage_zstats` and `zscored_sum` here verbatim, deleting them from `differential_uncertainty/baselines/activation_cdf.py`.
- Add after `condition_aurocs`:

```python
def group_separation(clean, degraded) -> tuple[float, float, float]:
    """Mean AUROC, AUPR and FPR95 over the conditions (rows) of `degraded`."""
    return (float(condition_aurocs(clean, degraded).mean()),
            float(np.mean([aupr(clean, row) for row in degraded])),
            float(np.mean([fpr_at_95_tpr(clean, row) for row in degraded])))


def group_aurocs(scores, rows) -> tuple[float, float]:
    """Mean AUROC over the common and over the extra conditions, on the given images."""
    values = np.asarray(scores, dtype=np.float64)[np.asarray(rows)]
    clean = values[:, 0]
    return (float(condition_aurocs(clean, values[:, COMMON_CONDITIONS].T).mean()),
            float(condition_aurocs(clean, values[:, EXTRA_CONDITIONS].T).mean()))
```

Replace `bootstrap` with:

```python
_STATISTIC = None  # the statistic forked bootstrap workers evaluate


def _evaluate(draw):
    return _STATISTIC(draw)


def bootstrap(statistic, n_images: int, samples: int = 1000, seed: int = 44, workers: int = 1) -> dict:
    """Paired whole-image bootstrap: the same resampled images feed every quantity in `statistic`.

    The draws come from one seeded generator in a fixed order, so `workers` (forked processes) changes only the speed.
    """
    global _STATISTIC
    generator = np.random.default_rng(seed)
    draws = [generator.integers(0, n_images, n_images) for _ in range(samples)]
    if workers > 1:
        _STATISTIC = statistic
        try:
            with multiprocessing.get_context("fork").Pool(workers) as pool:
                values = pool.map(_evaluate, draws, chunksize=max(1, samples // (4 * workers)))
        finally:
            _STATISTIC = None
    else:
        values = [statistic(draw) for draw in draws]
    return {key: tuple(float(v) for v in np.nanpercentile([d[key] for d in values], (2.5, 97.5)))
            for key in values[0]}
```

The docstring becomes `"""Separation metrics, per-stage z-scoring and the paired bootstrap; every score is oriented so higher means more degraded."""`.

- [ ] **Step 5: Rewrite the importers**

Rewrite the import lines:

```bash
sed -i 's/^from \. import metrics$/from degradation_monitor.evaluation import metrics/' differential_uncertainty/baselines/report.py
sed -i 's/^from \.\.baselines import metrics$/from degradation_monitor.evaluation import metrics/' differential_uncertainty/convtu/confirmation.py
sed -i 's/^from \.\.baselines\.activation_cdf import stage_zstats, zscored_sum$/from degradation_monitor.evaluation.metrics import stage_zstats, zscored_sum/' differential_uncertainty/convtu/channels.py differential_uncertainty/convtu/conditioned.py differential_uncertainty/convtu/pipeline.py
```

In `differential_uncertainty/baselines/pipeline.py`:
- Change `from .activation_cdf import CdfMonitor, ChannelRanges, ReferenceHistograms, stage_zstats, zscored_sum` to `from .activation_cdf import CdfMonitor, ChannelRanges, ReferenceHistograms`.
- Add `from degradation_monitor.evaluation.metrics import stage_zstats, zscored_sum`.

In `differential_uncertainty/baselines/report.py`:
- Delete `_group_separation`.
- Replace its two calls with `metrics.group_separation`.

In `differential_uncertainty/convtu/confirmation.py`:
- Replace the body of `group_aurocs` with `return metrics.group_aurocs(scores, rows)`. Its tests still call `confirmation.group_aurocs`.

Check:

```bash
grep -rn "baselines import metrics\|baselines\.metrics\|activation_cdf import.*zstats\|_group_separation" differential_uncertainty tests
```

Expected: prints nothing.

- [ ] **Step 6: Run the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: "169 passed, 1 skipped" (166 + 3 new metrics tests; the z-score test only moved).

- [ ] **Step 7: Commit**

```bash
git add -A degradation_monitor differential_uncertainty tests
git commit -q -m "refactor: evaluation metrics, with a bootstrap that can fork workers" -m "Separation metrics, stage z-scoring and the paired bootstrap move to degradation_monitor/evaluation/metrics.py; the bootstrap draws stay one seeded sequence, so worker processes give identical intervals." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

### Task 7: The six baselines

Each baseline gets its own module under `degradation_monitor/baselines/`. `baselines/scores.py` held three of them, so it splits:
- ContrastiveConf keeps the moved file and gains its λ fitting from `metrics.py`;
- SAOD and kNN get new files.

DisCoPatch, Hashemi et al. and the activation CDFs move unchanged.

**Files:**
- Move: `differential_uncertainty/baselines/scores.py` to `degradation_monitor/baselines/contrastive_conf.py`.
- Create: `degradation_monitor/baselines/saod.py`, `degradation_monitor/baselines/knn.py`.
- Move: `differential_uncertainty/baselines/{discopatch,hashemi,activation_cdf}.py` to `degradation_monitor/baselines/`.
- Modify: `degradation_monitor/evaluation/metrics.py` (λ fitting out), `differential_uncertainty/baselines/{pipeline,report}.py`.
- Move tests:
  - `tests/differential_uncertainty/test_baselines_scores.py` to `tests/baselines/test_contrastive_conf.py`;
  - `test_baselines_{discopatch,hashemi,activation_cdf}.py` to `tests/baselines/test_{discopatch,hashemi,activation_cdf}.py`.
- Create tests: `tests/baselines/test_saod.py`, `tests/baselines/test_knn.py`.

**Interfaces:**
- Produces:
  - `degradation_monitor.baselines.saod.saod_uncertainty(top_scores, m) -> float`.
  - `degradation_monitor.baselines.contrastive_conf`:
    - `THETA = 0.3`, `UQ_DETR_LAMBDA_GRID`;
    - `query_detections(logits, boxes, image_size)`, `contrastive_parts(query_sets, theta=0.3) -> (conf_pos, conf_neg)`, `contrastive_degradation(conf_pos, conf_neg, lam)`;
    - `fit_lambda_from_parts(conf_pos, conf_neg, reliability, grid=UQ_DETR_LAMBDA_GRID) -> (lambda, pcc)`;
    - `cross_fit_lambda(conf_pos, conf_neg, reliability, folds) -> (per_image ndarray, per_fold dict)`.
  - `degradation_monitor.baselines.knn`: `KNN_K = 100`, `KNN_K_MAX = 200`, `normalize_rows(features)`, `knn_distances(queries, bank, k_max, chunk_size=16384)`.
  - `degradation_monitor.baselines.activation_cdf`: unchanged names, plus `ZSTAT_IMAGES = 5000`.
  - `degradation_monitor.baselines.discopatch`, `degradation_monitor.baselines.hashemi`: unchanged.

- [ ] **Step 1: Move the files**

```bash
mkdir -p tests/baselines
git mv differential_uncertainty/baselines/scores.py degradation_monitor/baselines/contrastive_conf.py
for b in discopatch hashemi activation_cdf; do git mv differential_uncertainty/baselines/$b.py degradation_monitor/baselines/$b.py; git mv tests/differential_uncertainty/test_baselines_$b.py tests/baselines/test_$b.py; done
git mv tests/differential_uncertainty/test_baselines_scores.py tests/baselines/test_contrastive_conf.py
sed -i 's/^from differential_uncertainty\.baselines import /from degradation_monitor.baselines import /' tests/baselines/test_discopatch.py tests/baselines/test_hashemi.py tests/baselines/test_activation_cdf.py
```

- [ ] **Step 2: Write the failing tests**

`tests/baselines/test_saod.py`:

```python
import numpy as np
import pytest

from degradation_monitor.baselines.saod import saod_uncertainty


def test_saod_uncertainty_averages_one_minus_confidence_of_the_m_best():
    top = np.array([0.1, 0.9, 0.8])
    assert saod_uncertainty(top, 1) == pytest.approx(0.1)
    assert saod_uncertainty(top, 3) == pytest.approx((0.1 + 0.2 + 0.9) / 3)
    for bad in (0, 4):
        with pytest.raises(ValueError):
            saod_uncertainty(top, bad)
```

`tests/baselines/test_knn.py`:

```python
import numpy as np
import pytest
import torch

from degradation_monitor.baselines.knn import KNN_K, KNN_K_MAX, knn_distances


def test_knn_distances_are_sorted_kth_neighbour_distances_and_independent_of_chunking():
    angles = np.deg2rad([0, 10, 20, 30, 90])
    bank = torch.tensor(np.stack([np.cos(angles), np.sin(angles)], 1), dtype=torch.float32)
    query = torch.tensor([[2.0, 0.0]])
    full = knn_distances(query, bank, k_max=3, chunk_size=100)
    chunked = knn_distances(query, bank, k_max=3, chunk_size=2)
    expected = [2 * np.sin(np.deg2rad(d) / 2) for d in (0, 10, 20)]
    assert full[0].tolist() == pytest.approx(expected, abs=1e-3)
    assert torch.allclose(full, chunked)
    assert (KNN_K, KNN_K_MAX) == (100, 200)


def test_knn_distances_reject_zero_queries_and_too_large_k():
    bank = torch.eye(2)
    with pytest.raises(ValueError):
        knn_distances(torch.zeros(1, 2), bank, k_max=1)
    with pytest.raises(ValueError):
        knn_distances(torch.ones(1, 2), bank, k_max=3)
```

In `tests/baselines/test_contrastive_conf.py`:
- Delete the SAOD and kNN tests: `test_saod_uncertainty_averages_one_minus_confidence_of_the_m_best`, `test_knn_distances_are_sorted_kth_neighbour_distances_and_independent_of_chunking` and `test_knn_distances_reject_zero_queries_and_too_large_k`.
- Replace `from differential_uncertainty.baselines import scores` with `from degradation_monitor.baselines import contrastive_conf as scores`.
- Move `test_fit_lambda_from_parts_matches_uq_detr_fit_lambda` and `test_cross_fit_lambda_never_uses_the_images_of_its_own_fold` here from `tests/evaluation/test_metrics.py`, replacing `m.` with `scores.` in them.
- In `tests/evaluation/test_metrics.py`, delete the now-unused `from differential_uncertainty.baselines import scores` and `import uq_detr` lines.

- [ ] **Step 3: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/baselines`
Expected: ERRORS. There is no `degradation_monitor.baselines.saod` or `degradation_monitor.baselines.knn`, and `contrastive_conf` has no `fit_lambda_from_parts`.

- [ ] **Step 4: Implement**

`degradation_monitor/baselines/saod.py`:

```python
"""SAOD image-level uncertainty (Oksuz et al., CVPR 2023): 1 - confidence of the most confident detections."""
from __future__ import annotations

import numpy as np


def saod_uncertainty(top_scores, m: int) -> float:
    """SAOD image uncertainty: mean of (1 - p) over the m most confident detections."""
    ranked = np.sort(np.asarray(top_scores, dtype=np.float64))[::-1]
    if type(m) is not int or not 1 <= m <= ranked.size:
        raise ValueError("m must be between 1 and the number of detections")
    return float(np.mean(1.0 - ranked[:m]))
```

`degradation_monitor/baselines/knn.py`. Both function bodies are verbatim from the old `scores.py`:

```python
"""kNN out-of-distribution score (Sun et al., ICML 2022): distance to the k-th nearest clean train image, on the
L2-normalised, globally pooled last backbone stage of the same frozen detector."""
from __future__ import annotations

import torch

KNN_K = 100       # fixed in advance; the report also shows k = 1, 10, 50, 200
KNN_K_MAX = 200   # how many neighbour distances the detector pass stores per image


def normalize_rows(features: torch.Tensor) -> torch.Tensor:
    features = features.float()
    norms = features.norm(dim=1, keepdim=True)
    if not bool(torch.isfinite(features).all()) or bool((norms == 0).any()):
        raise ValueError("features must be finite with nonzero norm")
    return features / norms


def knn_distances(queries: torch.Tensor, bank: torch.Tensor, k_max: int, chunk_size: int = 16384):
    """Ascending Euclidean distances from unit queries to their k_max nearest unit bank rows."""
    if queries.ndim != 2 or bank.ndim != 2 or queries.shape[1] != bank.shape[1]:
        raise ValueError("queries and bank must be (rows, dim) with the same dim")
    if not 1 <= k_max <= bank.shape[0]:
        raise ValueError("k_max must be between 1 and the bank size")
    q = normalize_rows(queries)
    best = None
    for start in range(0, bank.shape[0], chunk_size):
        rows = bank[start:start + chunk_size].to(device=q.device, dtype=torch.float32)
        squared = (2.0 - 2.0 * q @ rows.T).clamp_min_(0.0)
        local = squared.topk(min(k_max, rows.shape[0]), dim=1, largest=False).values
        merged = local if best is None else torch.cat([best, local], dim=1)
        best = merged.topk(min(k_max, merged.shape[1]), dim=1, largest=False).values
    return best.sqrt()
```

**`degradation_monitor/baselines/contrastive_conf.py`:**
- Delete `saod_uncertainty`, `normalize_rows` and `knn_distances`, and the `import torch`.
- Replace the import `from degradation_monitor.detector.postprocess import checked_outputs as _checked, top_detections` with `from ..detector.postprocess import checked_outputs as _checked`.
- Add `import math`, `import warnings` and `from scipy.stats import ConstantInputWarning, pearsonr`.
- Add `THETA = 0.3  # uq-detr's threshold split between positive and negative queries`.
- Move `UQ_DETR_LAMBDA_GRID`, `fit_lambda_from_parts` and `cross_fit_lambda` here verbatim, deleting them from `degradation_monitor/evaluation/metrics.py`.
- The docstring becomes `"""ContrastiveConf (Park, Sobolewski & Azizan, TPAMI 2026): Conf+ - lambda Conf- over a frozen DETR's queries, with lambda cross-fitted on labelled clean images."""`.

In `metrics.py`, drop `pearsonr` from the scipy import if nothing else uses it.

Add to `degradation_monitor/baselines/activation_cdf.py`, after `STAGES`: `ZSTAT_IMAGES = 5000  # seeded sample of clean train images for the per-stage z-statistics`.

- [ ] **Step 5: Rewrite the importers**

In `differential_uncertainty/baselines/pipeline.py`, replace the baseline imports with:

```python
from degradation_monitor.baselines import discopatch
from degradation_monitor.baselines.activation_cdf import BINS as CDF_BINS
from degradation_monitor.baselines.activation_cdf import MARGIN as CDF_MARGIN
from degradation_monitor.baselines.activation_cdf import STAGES as CDF_STAGES
from degradation_monitor.baselines.activation_cdf import CdfMonitor, ChannelRanges, ReferenceHistograms
from degradation_monitor.baselines.contrastive_conf import contrastive_parts, query_detections
from degradation_monitor.baselines.discopatch import DisCoPatchScorer, train_discopatch
from degradation_monitor.baselines.hashemi import K as HASHEMI_K
from degradation_monitor.baselines.hashemi import LAYERS as HASHEMI_LAYERS
from degradation_monitor.baselines.hashemi import HashemiMonitor, NeuronStats, save_intervals
from degradation_monitor.baselines.knn import knn_distances, normalize_rows
from degradation_monitor.baselines.saod import saod_uncertainty
from degradation_monitor.detector.postprocess import top_detections
```

That replaces the `from . import discopatch` line, the five `from .activation_cdf ...` / `from .discopatch ...` / `from .hashemi ...` lines, and the `from .scores import (...)` block.

In `differential_uncertainty/baselines/report.py`:
- `from .activation_cdf import BINS as CDF_BINS` becomes `from degradation_monitor.baselines.activation_cdf import BINS as CDF_BINS`.
- `from .hashemi import K as HASHEMI_K` becomes `from degradation_monitor.baselines.hashemi import K as HASHEMI_K`.
- Add `from degradation_monitor.baselines.contrastive_conf import cross_fit_lambda`, and replace both `metrics.cross_fit_lambda(` calls with `cross_fit_lambda(`.

Then check:

```bash
grep -rn "from \.scores\|from \.activation_cdf\|from \.hashemi\|from \.discopatch\|baselines import scores\|metrics.cross_fit_lambda\|metrics.fit_lambda" differential_uncertainty tests
```

Expected: prints nothing.

- [ ] **Step 6: Run the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: "169 passed, 1 skipped". Tests only moved between files: SAOD 1, kNN 2, λ 2.

- [ ] **Step 7: Commit**

```bash
git add -A degradation_monitor differential_uncertainty tests
git commit -q -m "refactor: one module per baseline under degradation_monitor/baselines/" -m "scores.py splits into saod.py, contrastive_conf.py (which gains the lambda cross-fit) and knn.py; DisCoPatch, Hashemi et al. and the activation CDFs move unchanged." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

### Task 8: The method

The method moves into `degradation_monitor/method/`:
- **`statistics.py`:** per-channel level and top-1% mean.
- **`reference.py`:** the clean spread and the content key's neighbour search.
- **`scores.py`:** level, peak share and two-axis scores, and the three ablation rows.

`conditioned.py` moves with `git mv`, as `scores.py`, and its functions get clearer names. The old `convtu/channels.py` stays as it is: the old pipeline still writes its four statistics until Task 13, which archives it. So `statistics.py` and `reference.py` start as new files. A transitional test pins the new kNN and own-average rows to the old `channel_method_scores`, bit for bit.

**Files:**
- Move: `differential_uncertainty/convtu/conditioned.py` to `degradation_monitor/method/scores.py`.
- Create: `degradation_monitor/method/statistics.py`, `degradation_monitor/method/reference.py`.
- Modify: `differential_uncertainty/convtu/confirmation.py`.
- Move test: `tests/differential_uncertainty/test_convtu_conditioned.py` to `tests/method/test_scores.py`.
- Create tests: `tests/method/test_statistics.py`, `tests/method/test_reference.py`.
- Modify tests: `tests/differential_uncertainty/test_convtu_channels.py` (the transitional equality test), `tests/differential_uncertainty/test_convtu_pipeline.py` (the monkeypatched `NEIGHBOURS`).

**Interfaces:**
- Produces:
  - `degradation_monitor.method.statistics`: `TOP_FRACTION = 0.01`, `STATISTICS = ("means", "top")`, `KEYS = ("means_s1", ..., "means_s4", "top_s1", ..., "top_s4")`, `channel_statistics(x (N, C, H, W)) -> {"means", "top": float32 (N, C)}`.
  - `degradation_monitor.method.reference`: `STD_FLOOR = 0.01`, `KEY_LAYER = "s4"`, `SCORED_LAYERS = ("s1", "s2", "s3")`, `NEIGHBOURS = 50`, `CHUNK = 2048`, `fit_own_average(reference) -> (mean, floored std)`, `nearest_rows(queries, reference, k, chunk=CHUNK) -> (n, k) indices`.
  - `degradation_monitor.method.scores`. Each takes `(test, bank, zstats, ...)`, where `test` maps keys to `(..., C)` arrays and `bank`/`zstats` map them to `(rows, C)`:
    - `EPS`, `KNN_NEIGHBOURS = 5`;
    - `peak_share(values, layer)`;
    - `level_scores(test, bank, zstats, key=KEY_LAYER, scored=SCORED_LAYERS, k=NEIGHBOURS) -> (summed, per_stage)`;
    - `peak_share_scores(...) -> (summed, per_stage)`;
    - `two_axis_scores(...) -> (score, {"flatter", "level"})`;
    - `global_level_scores(test, bank, zstats, scored=SCORED_LAYERS) -> (summed, per_stage)`;
    - `knn_scores(queries, bank, neighbours=KNN_NEIGHBOURS, chunk=64)`;
    - `own_average_scores(values, mean, std)`;
    - `means_knn_scores(test, bank, zstats) -> (summed, per_stage)` and `means_own_scores(test, bank, zstats) -> (summed, per_stage)`, both over all four stages.

- [ ] **Step 1: Move the file and its test**

```bash
mkdir -p tests/method
git mv differential_uncertainty/convtu/conditioned.py degradation_monitor/method/scores.py
git mv tests/differential_uncertainty/test_convtu_conditioned.py tests/method/test_scores.py
```

- [ ] **Step 2: Write the failing tests.**

`tests/method/test_statistics.py`:

```python
import numpy as np
import pytest
import torch

from degradation_monitor.method.statistics import KEYS, STATISTICS, TOP_FRACTION, channel_statistics


def test_channel_statistics_give_the_mean_and_the_top_one_percent_mean_of_each_channel():
    x = torch.zeros(1, 2, 10, 10)
    x[0, 0] = -torch.arange(100, dtype=torch.float32).view(10, 10)  # |x| = 0..99; the top 1% is the single 99
    x[0, 1] = 2.0
    stats = channel_statistics(x)
    assert set(stats) == set(STATISTICS) == {"means", "top"} and TOP_FRACTION == 0.01
    assert stats["means"].dtype == np.float32 and stats["means"].shape == (1, 2)
    assert stats["means"][0].tolist() == pytest.approx([49.5, 2.0])
    assert stats["top"][0].tolist() == pytest.approx([99.0, 2.0])
    assert KEYS == tuple(f"{s}_s{l}" for s in ("means", "top") for l in range(1, 5))
    with pytest.raises(ValueError):
        channel_statistics(torch.zeros(2, 10, 10))


def test_channel_statistics_equal_the_old_pilot_statistics():
    from differential_uncertainty.convtu.channels import channel_statistics as old
    x = torch.randn(3, 4, 20, 20)
    new, before = channel_statistics(x), old(x)
    assert np.array_equal(new["means"], before["means"]) and np.array_equal(new["top"], before["top"])
```

The second test is transitional. Task 13 deletes it with the old module.

`tests/method/test_reference.py`:

```python
import numpy as np
import pytest

from degradation_monitor.method import reference


def test_fit_own_average_floors_dead_dimensions():
    clean = np.array([[1.0, 0.0, 5.0], [3.0, 0.0, 7.0]])
    mean, std = reference.fit_own_average(clean)
    assert mean.tolist() == [2.0, 0.0, 6.0]
    assert std.tolist() == pytest.approx([1.0, reference.STD_FLOOR * 1.0, 1.0])
    with pytest.raises(ValueError):
        reference.fit_own_average(np.zeros((1, 3)))


def test_nearest_rows_match_a_brute_force_search():
    rng = np.random.default_rng(0)
    bank, queries = rng.normal(size=(40, 6)), rng.normal(size=(9, 6))
    got = reference.nearest_rows(queries, bank, 5, chunk=4)
    distances = ((queries[:, None] - bank[None]) ** 2).sum(-1)
    assert [set(row) for row in got] == [set(row) for row in np.argsort(distances, axis=1)[:, :5]]
    with pytest.raises(ValueError, match="between 1 and"):
        reference.nearest_rows(queries, bank, 41)


def test_the_fixed_choices_are_the_preregistered_ones():
    assert (reference.KEY_LAYER, reference.SCORED_LAYERS, reference.NEIGHBOURS) == ("s4", ("s1", "s2", "s3"), 50)
```

In `tests/method/test_scores.py`:
- Replace `from differential_uncertainty.convtu import conditioned` with `from degradation_monitor.method import scores as conditioned`.
- Rename the calls: `conditioned.conditioned_scores(` becomes `conditioned.level_scores(`, and `conditioned.global_scores(` becomes `conditioned.global_level_scores(`.
- Delete `test_nearest_rows_match_a_brute_force_search`; it now lives in `test_reference.py`.
- Append:

```python
def test_knn_scores_are_mean_euclidean_distances_to_the_nearest_rows():
    import torch
    bank = torch.tensor([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    queries = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    assert conditioned.knn_scores(queries, bank, neighbours=2).tolist() == pytest.approx([2.5, 2.5])


def test_own_average_scores_are_mean_absolute_z():
    values = np.array([[3.0, 1.0], [1.0, 1.0]])
    got = conditioned.own_average_scores(values, np.array([1.0, 1.0]), np.array([2.0, 1.0]))
    assert got.tolist() == pytest.approx([0.5, 0.0])


def test_the_four_stage_rows_keep_the_test_shape_and_score_the_zstatistics_images_near_zero():
    rng = np.random.default_rng(4)
    widths = {"s1": 3, "s2": 4, "s3": 5, "s4": 6}
    bank = {f"means_{l}": rng.random((30, c)) for l, c in widths.items()}
    zstats = {f"means_{l}": rng.random((12, c)) for l, c in widths.items()}
    test = {f"means_{l}": rng.random((4, 7, c)) for l, c in widths.items()}
    for function in (conditioned.means_knn_scores, conditioned.means_own_scores):
        summed, per_stage = function(test, bank, zstats)
        assert summed.shape == (4, 7) and per_stage.shape == (4, 7, 4)
        clean, _ = function({k: v[None] for k, v in zstats.items()}, bank, zstats)
        assert abs(clean.mean()) < 1e-6
```

Append to `tests/differential_uncertainty/test_convtu_channels.py` (transitional; Task 13 deletes the file):

```python
def test_the_new_four_stage_rows_equal_the_old_channel_method_scores_bit_for_bit():
    from degradation_monitor.method import scores as method_scores
    rng = np.random.default_rng(9)
    widths = {"s1": 3, "s2": 4, "s3": 5, "s4": 6}
    bank = {f"means_{l}": rng.random((30, c)).astype(np.float32) for l, c in widths.items()}
    zstats = {f"means_{l}": rng.random((12, c)).astype(np.float32) for l, c in widths.items()}
    test = {f"means_{l}": rng.random((4, 7, c)).astype(np.float32) for l, c in widths.items()}
    old, _ = channels.channel_method_scores(bank, zstats, test, list(widths), statistics=("means",))
    assert np.array_equal(method_scores.means_knn_scores(test, bank, zstats)[0], old["ch_means_knn"])
    assert np.array_equal(method_scores.means_own_scores(test, bank, zstats)[0], old["ch_means_own"])
```

- [ ] **Step 3: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/method tests/differential_uncertainty/test_convtu_channels.py`
Expected: ERRORS and FAILs. `degradation_monitor.method.statistics` and `degradation_monitor.method.reference` do not exist, and `scores` has no `level_scores`, `knn_scores` or `means_knn_scores`.

- [ ] **Step 4: Implement `statistics.py` and `reference.py`**

`degradation_monitor/method/statistics.py`. The arithmetic is the old `channel_statistics`, restricted to its two kept outputs:

```python
"""Per-channel statistics of the four early backbone maps: the level (mean |x|) and the mean of the top 1% of |x|."""
from __future__ import annotations

import numpy as np
import torch

TOP_FRACTION = 0.01
STATISTICS = ("means", "top")
KEYS = tuple(f"{statistic}_s{stage}" for statistic in STATISTICS for stage in range(1, 5))  # one per stored array


@torch.inference_mode()
def channel_statistics(x: torch.Tensor) -> dict:
    """For a batch of maps (N, C, H, W): each channel's mean |x| and the mean of its largest 1% of |x|, float32 (N, C)."""
    if x.ndim != 4:
        raise ValueError("expected a batch of shape (N, C, H, W)")
    x_abs = x.abs().float()
    n, channel_count, height, width = x_abs.shape
    k = max(1, round(TOP_FRACTION * height * width))
    top = torch.topk(x_abs.reshape(n, channel_count, -1), k, dim=2).values
    out = {"means": x_abs.mean(dim=(2, 3)), "top": top.mean(dim=2)}
    return {key: value.cpu().numpy().astype(np.float32) for key, value in out.items()}
```

`degradation_monitor/method/reference.py`. `fit_own_average` is verbatim from `convtu/channels.py`, and `nearest_rows` verbatim from the moved `scores.py`, which deletes its own copy:

```python
"""The clean reference: per-dimension spreads, and the stage-4 content key's search for the most similar clean scenes.

The fixed choices below were pre-registered before the 5,000-image confirmation; ablations pass other values explicitly.
"""
from __future__ import annotations

import numpy as np

STD_FLOOR = 0.01  # every dimension's spread is at least 1% of the median positive spread of its layer
KEY_LAYER = "s4"
SCORED_LAYERS = ("s1", "s2", "s3")
NEIGHBOURS = 50
CHUNK = 2048


def fit_own_average(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension clean mean and population spread; the spread is floored so dead dimensions stay finite."""
    reference = np.asarray(reference, dtype=np.float64)
    if reference.ndim != 2 or reference.shape[0] < 2:
        raise ValueError("need at least two clean rows of shape (rows, dim)")
    mean, std = reference.mean(axis=0), reference.std(axis=0)
    positive = std[std > 0]
    if positive.size == 0:
        raise ValueError("the clean rows have no spread")
    return mean, np.maximum(std, STD_FLOOR * float(np.median(positive)))


def nearest_rows(queries: np.ndarray, reference: np.ndarray, k: int, chunk: int = CHUNK) -> np.ndarray:
    """Indices of each query's k nearest reference rows (Euclidean), in no particular order."""
    queries, reference = np.asarray(queries, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if not 1 <= k <= len(reference):
        raise ValueError("k must be between 1 and the number of reference rows")
    reference_sq = (reference ** 2).sum(axis=1)
    out = np.empty((len(queries), k), dtype=np.int64)
    for start in range(0, len(queries), chunk):
        rows = queries[start:start + chunk]
        distances = (rows ** 2).sum(axis=1)[:, None] + reference_sq[None] - 2 * rows @ reference.T
        out[start:start + chunk] = np.argpartition(distances, k - 1, axis=1)[:, :k]
    return out
```

- [ ] **Step 5: Implement `scores.py`**

In `degradation_monitor/method/scores.py`:
- **Imports:** change the imports to:

```python
import numpy as np
import torch

from ..evaluation.metrics import stage_zstats, zscored_sum
from .reference import CHUNK, KEY_LAYER, NEIGHBOURS, SCORED_LAYERS, fit_own_average, nearest_rows

EPS = 1e-6  # keeps the logarithm finite for a channel that is silent on an image
KNN_NEIGHBOURS = 5
```

- **Delete** its own `KEY_LAYER`, `SCORED_LAYERS`, `NEIGHBOURS`, `CHUNK`, `EPS` and `nearest_rows`.
- **Rename:** `conditioned_scores` becomes `level_scores`, and `global_scores` becomes `global_level_scores`. Both bodies are unchanged.
- **Keep** `peak_share_scores` and `two_axis_scores` unchanged.
- **Docstring:** the module docstring's first line becomes `"""Our scores: level, peak share and the two-axis score, all judged against similar clean scenes, plus three ablation rows."""`. The rest of the docstring stays.

Then append the three ablation helpers. `knn_scores` is verbatim from `convtu/features.py`, and `own_average_scores` verbatim from `convtu/channels.py`. The two row functions repeat the old `channel_method_scores` arithmetic for the means statistic:

```python
def knn_scores(queries: torch.Tensor, bank: torch.Tensor, neighbours: int = KNN_NEIGHBOURS,
               chunk: int = 64) -> np.ndarray:
    """Mean Euclidean distance from each query row to its nearest `neighbours` bank rows (no normalization)."""
    if queries.ndim != 2 or bank.ndim != 2 or queries.shape[1] != bank.shape[1]:
        raise ValueError("queries and bank must be (rows, dim) with the same dim")
    if not 1 <= neighbours <= bank.shape[0]:
        raise ValueError("neighbours must be between 1 and the bank size")
    bank = bank.float()
    out = []
    for start in range(0, queries.shape[0], chunk):
        rows = queries[start:start + chunk].to(bank.device, torch.float32)
        distances = torch.cdist(rows, bank, compute_mode="donot_use_mm_for_euclid_dist")
        out.append(distances.topk(neighbours, dim=1, largest=False).values.mean(dim=1))
    return torch.cat(out).cpu().numpy().astype(np.float64)


def own_average_scores(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Mean over dimensions of |value - clean mean| / clean spread: one score per row."""
    return (np.abs(np.asarray(values, dtype=np.float64) - mean) / std).mean(axis=1)


FOUR_LAYERS = ("s1", "s2", "s3", "s4")


def _float32(array) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


def _four_stage_rows(column_of, test: dict, zstats: dict) -> tuple[np.ndarray, np.ndarray]:
    leading = _flat(test["means_s1"])[1]
    clean = np.stack([column_of(layer, zstats[f"means_{layer}"]) for layer in FOUR_LAYERS], axis=1)
    mean, std = stage_zstats(clean)
    values = np.stack([column_of(layer, test[f"means_{layer}"].reshape(-1, test[f"means_{layer}"].shape[-1]))
                       for layer in FOUR_LAYERS], axis=1)
    return zscored_sum(values, mean, std).reshape(leading), values.reshape(*leading, len(FOUR_LAYERS))


def means_knn_scores(test: dict, bank: dict, zstats: dict) -> tuple[np.ndarray, np.ndarray]:
    """Ablation: kNN (5 nearest, unnormalised) on the channel means of all four stages; the pilot's control row."""
    def column(layer, values):
        return knn_scores(_float32(values), _float32(bank[f"means_{layer}"]))
    return _four_stage_rows(column, test, zstats)


def means_own_scores(test: dict, bank: dict, zstats: dict) -> tuple[np.ndarray, np.ndarray]:
    """Ablation: each channel mean against its own clean average, all four stages (Neural Mean Discrepancy-style)."""
    def column(layer, values):
        mean, std = fit_own_average(bank[f"means_{layer}"])
        return own_average_scores(values, mean, std)
    return _four_stage_rows(column, test, zstats)
```

Old `channel_method_scores` reshaped `test[key]` as `(images * conditions, dim)` and kept every dimension, as the rows above do. The transitional test in Step 2 proves bit equality.

- [ ] **Step 6: Point the old confirmation at the new names**

In `differential_uncertainty/convtu/confirmation.py`:
- Replace `from . import conditioned` with `from degradation_monitor.method import reference as method_reference` and `from degradation_monitor.method import scores as method_scores`.
- Delete `from .channels import channel_method_scores`.
- In `build_confirmation_report`:
  - `k = conditioned.NEIGHBOURS` becomes `k = method_reference.NEIGHBOURS`.
  - Each `conditioned.X_scores(` call becomes `method_scores.X_scores(`, under the new names: `two_axis_scores`, `peak_share_scores`, `level_scores` for the old `conditioned_scores`, and `global_level_scores` for the old `global_scores`.
  - The two lines that called `channel_method_scores` become:

```python
    scores.update({"ch_means_knn": method_scores.means_knn_scores(means, bank, zstats)[0],
                   "ch_means_own": method_scores.means_own_scores(means, bank, zstats)[0]})
```

  - Also in the summary dict: `conditioned.KEY_LAYER` becomes `method_reference.KEY_LAYER`, and `conditioned.SCORED_LAYERS` becomes `method_reference.SCORED_LAYERS`.

In `tests/differential_uncertainty/test_convtu_pipeline.py`, inside `test_conditioned_report_writes_every_row_and_checks_the_screen_images`:
- Replace `from differential_uncertainty.convtu import conditioned, confirmation` with `from differential_uncertainty.convtu import confirmation` and `from degradation_monitor.method import reference as conditioned`.
- The `monkeypatch.setattr(conditioned, "NEIGHBOURS", 3)` line then patches the new home.

- [ ] **Step 7: Run the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: "177 passed, 1 skipped". That is 169, plus:
- 2 statistics tests;
- 3 reference tests, of which 1 moved from `test_scores`, so 2 more in all;
- 3 new score tests;
- 1 transitional equality test.

`grep -rn "convtu import conditioned\|conditioned\.\(conditioned\|global\)_scores\|channel_method_scores" differential_uncertainty/convtu/confirmation.py` prints nothing.

- [ ] **Step 8: Commit**

```bash
git add -A degradation_monitor differential_uncertainty tests
git commit -q -m "refactor: the method package: statistics, reference and scores" -m "conditioned.py moves to degradation_monitor/method/scores.py with clearer names and gains the two four-stage ablation rows; the level and top-1% statistics and the clean reference are new small modules. A transitional test pins the new rows to the old channel_method_scores bit for bit." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

### Task 9: The run folder and the settings

There are two new modules. Nothing uses them until Task 10.
- `runs.py` names every path of the new run folder, writes files atomically, refuses damaged results, and keeps the manifest: the protocol, the environment, and the inputs each score folder was computed from.
- `settings.py` reads the machine's paths and run options from a TOML file.

**Files:**
- Create: `degradation_monitor/runs.py`, `degradation_monitor/settings.py`, `configs/coco.toml`.
- Test: `tests/test_runs.py`, `tests/test_settings.py`.

**Interfaces:**
- Consumes:
  - `degradation_monitor.datasets.coco.{Coco, FOLDS}`, `degradation_monitor.corruptions.CONDITIONS`;
  - `degradation_monitor.detector.postprocess.TOP_K`, `degradation_monitor.baselines.knn.{KNN_K, KNN_K_MAX}`, `degradation_monitor.baselines.contrastive_conf.THETA`.
- Produces:
  - `degradation_monitor.runs`:
    - `SCORE_KEYS: dict[str, tuple]` for `detector`, `activations`, `discopatch` and `method`;
    - `RunLayout(root)`. Its properties are `knn_bank`, `knn_names`, `cdf_reference`, `cdf_fit`, `cdf_zstats`, `hashemi_intervals`, `hashemi_fit`, `discopatch_dir`, `discopatch_checkpoint`, `discopatch_training`, `method_bank`, `method_zstats`, `manifest` and `timing`. Its methods are `reference(*parts)`, `scores(name)`, `score_file(name, image)` and `report(name="coco")`;
    - `atomic_json(path, value)`, `atomic_npz(path, **arrays)`, `load_npz(path, keys)`, `valid_existing(path, keys)`, `stack(folder, names, keys)`, `sha1(path)`, `sha256(path)`, `progress(label, done, total, started)`;
    - `Manifest(layout)`, with `read()`, `update(**sections)`, `check_protocol(protocol)`, `check_inputs(folder, inputs)` and `record_environment(discopatch_root)`.
  - `degradation_monitor.settings`:
    - `Settings(run, checkpoint, train_images, val_images, annotations, discopatch_root, device="cuda:0", batch_size=32, workers=9, gpu_memory_gib=None, limit=None, epochs=65, seed=44)`, with `.layout`, `.dataset` and `.protocol()`;
    - `load_settings(path, **overrides) -> Settings`;
    - `PATH_FIELDS`.

- [ ] **Step 1: Write the failing tests**

`tests/test_runs.py`:

```python
import json

import numpy as np
import pytest

from degradation_monitor.runs import SCORE_KEYS, Manifest, RunLayout, atomic_npz, load_npz, stack, valid_existing


def test_the_layout_names_every_reference_and_score_folder(tmp_path):
    layout = RunLayout(tmp_path)
    assert layout.knn_bank == tmp_path / "reference" / "knn" / "bank.npy"
    assert layout.method_bank == tmp_path / "reference" / "method" / "bank.npz"
    assert layout.discopatch_checkpoint == tmp_path / "reference" / "discopatch" / "discriminator.pt"
    assert layout.score_file("method", "val/000000000139.jpg") == tmp_path / "scores" / "method" / "000000000139.npz"
    assert layout.report() == tmp_path / "reports" / "coco"
    assert set(SCORE_KEYS) == {"detector", "activations", "discopatch", "method"}
    with pytest.raises(ValueError, match="unknown score folder"):
        layout.scores("test")


def test_result_files_are_written_atomically_and_damaged_ones_are_refused(tmp_path):
    path = tmp_path / "scores" / "detector" / "a.npz"
    atomic_npz(path, x=np.arange(3))
    assert load_npz(path, ("x",))["x"].tolist() == [0, 1, 2]
    assert not list(path.parent.glob(".*.tmp"))
    assert valid_existing(path, ("x",)) and not valid_existing(tmp_path / "missing.npz", ("x",))
    with pytest.raises(ValueError, match="lacks"):
        load_npz(path, ("x", "y"))
    path.write_bytes(b"not an npz")
    with pytest.raises(ValueError, match="malformed"):
        valid_existing(path, ("x",))


def test_stack_reads_every_named_image_and_names_a_missing_one(tmp_path):
    for name in ("a", "b"):
        atomic_npz(tmp_path / f"{name}.npz", v=np.full(2, ord(name)))
    assert stack(tmp_path, ["a.jpg", "b.jpg"], ("v",))["v"].shape == (2, 2)
    with pytest.raises(ValueError, match="1 of 3 missing, e.g. c.jpg"):
        stack(tmp_path, ["a.jpg", "b.jpg", "c.jpg"], ("v",))


def test_a_run_folder_refuses_another_protocol(tmp_path):
    manifest = Manifest(RunLayout(tmp_path))
    manifest.check_protocol({"seed": 44, "limit": None, "conditions": [("clean", 0), ("fog", 1)]})
    manifest.check_protocol({"seed": 44, "limit": None, "conditions": [["clean", 0], ["fog", 1]]})  # tuples == lists
    with pytest.raises(ValueError, match=r"another protocol \(limit, seed\)"):
        manifest.check_protocol({"seed": 45, "limit": 10, "conditions": [["clean", 0], ["fog", 1]]})


def test_a_score_folder_refuses_inputs_other_than_its_own(tmp_path):
    manifest = Manifest(RunLayout(tmp_path))
    manifest.check_inputs("activations", {"cdf_reference": "aa", "hashemi_k": 2.0})
    manifest.check_inputs("activations", {"cdf_reference": "aa", "hashemi_k": 2.0})
    with pytest.raises(ValueError, match="inputs of scores/activations changed .*cdf_reference"):
        manifest.check_inputs("activations", {"cdf_reference": "bb", "hashemi_k": 2.0})
    assert json.loads((tmp_path / "manifest.json").read_text())["inputs"]["activations"]["cdf_reference"] == "aa"


def test_the_environment_is_recorded_once(tmp_path):
    manifest = Manifest(RunLayout(tmp_path))
    manifest.record_environment(tmp_path)
    first = manifest.read()["environment"]
    assert "torch" in first["packages"] and first["discopatch_commit"] is None
    manifest.update(environment={"packages": {}, "discopatch_commit": "x", "gpu": None})
    manifest.record_environment(tmp_path)
    assert manifest.read()["environment"]["discopatch_commit"] == "x"
```

`tests/test_settings.py`:

```python
import hashlib
from pathlib import Path

import pytest

from degradation_monitor.settings import load_settings

CONFIG = """
run = "{root}/runs/coco"
checkpoint = "{root}/ckpt.pth"
train_images = "{root}/train"
val_images = "{root}/val"
annotations = "{root}/ann.json"
discopatch_root = "{root}/dcp"
batch_size = 8
gpu_memory_gib = 5.5
"""


def _config(tmp_path, text=CONFIG):
    path = tmp_path / "coco.toml"
    path.write_text(text.format(root=tmp_path))
    return path


def test_settings_load_paths_and_options_and_take_command_line_overrides(tmp_path):
    settings = load_settings(_config(tmp_path), device="cpu", limit=10, workers=None)
    assert settings.run == tmp_path / "runs" / "coco" and isinstance(settings.checkpoint, Path)
    assert (settings.batch_size, settings.gpu_memory_gib, settings.device, settings.limit) == (8, 5.5, "cpu", 10)
    assert settings.workers == 9  # an override of None keeps the file's value, here the default
    assert settings.layout.root == settings.run
    assert settings.dataset.limit == 10 and settings.dataset.val_root == tmp_path / "val"


def test_settings_reject_unknown_and_missing_keys(tmp_path):
    with pytest.raises(ValueError, match="unknown settings .*: bach_size"):
        load_settings(_config(tmp_path, CONFIG + "bach_size = 4\n"))
    with pytest.raises(ValueError, match="needs annotations"):
        load_settings(_config(tmp_path, CONFIG.replace('annotations = "{root}/ann.json"\n', "")))


def test_the_protocol_names_the_checkpoint_by_its_hash_and_fixes_the_benchmark(tmp_path):
    (tmp_path / "ckpt.pth").write_bytes(b"weights")
    protocol = load_settings(_config(tmp_path)).protocol()
    assert protocol["checkpoint_sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert (protocol["dataset"], protocol["seed"], protocol["limit"], protocol["folds"]) == ("coco", 44, None, 5)
    assert len(protocol["conditions"]) == 96 and protocol["conditions"][0] == ["clean", 0]
    assert (protocol["top_k"], protocol["knn_k"], protocol["knn_k_max"], protocol["theta"]) == (100, 100, 200, 0.3)


def test_the_repository_config_names_every_path():
    settings = load_settings(Path(__file__).parents[1] / "configs" / "coco.toml")
    assert settings.run.is_absolute() and settings.run.name == "coco"
    assert settings.gpu_memory_gib == 5.5 and settings.seed == 44
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/test_runs.py tests/test_settings.py`
Expected: ERRORS, `ModuleNotFoundError: No module named 'degradation_monitor.runs'` and `...settings`.

- [ ] **Step 3: Implement `degradation_monitor/runs.py`**

```python
"""The run folder: where every stage reads and writes, atomic writes, resume checks and the manifest.

  manifest.json                    the protocol, the environment, and the inputs each score folder was computed from
  reference/<name>/                what the clean train images provide: knn, activation_cdf, hashemi, discopatch, method
  scores/<pass>/<image stem>.npz   one file per evaluation image, 96 conditions each: detector, activations,
                                   discopatch, method
  reports/<name>/                  the report stage's tables
  timing.json                      the timing stage's measurements
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np

SCORE_KEYS = {
    "detector": ("saod_min", "saod_top3", "conf_pos", "conf_neg", "knn", "det_scores", "det_labels", "det_boxes",
                 "digests", "size"),
    "activations": ("hashemi_decoder", "hashemi_encoder", "hashemi_encoder_maps", "cdf_backbone", "cdf_backbone_z",
                    "cdf_stages"),
    "discopatch": ("dcp",),
    "method": tuple(f"{statistic}_s{stage}" for statistic in ("means", "top") for stage in range(1, 5)),
}
PACKAGES = ("torch", "torchvision", "numpy", "scipy", "scikit-image", "scikit-learn", "imagecorruptions", "uq-detr",
            "pycocotools", "Pillow")


@dataclass(frozen=True)
class RunLayout:
    root: Path

    def reference(self, *parts: str) -> Path:
        return self.root.joinpath("reference", *parts)

    @property
    def knn_bank(self) -> Path:
        return self.reference("knn", "bank.npy")

    @property
    def knn_names(self) -> Path:
        return self.reference("knn", "image_names.json")

    @property
    def cdf_reference(self) -> Path:
        return self.reference("activation_cdf", "reference.npz")

    @property
    def cdf_fit(self) -> Path:
        return self.reference("activation_cdf", "fit.json")

    @property
    def cdf_zstats(self) -> Path:
        return self.reference("activation_cdf", "zstats.json")

    @property
    def hashemi_intervals(self) -> Path:
        return self.reference("hashemi", "intervals.npz")

    @property
    def hashemi_fit(self) -> Path:
        return self.reference("hashemi", "fit.json")

    @property
    def discopatch_dir(self) -> Path:
        return self.reference("discopatch")

    @property
    def discopatch_checkpoint(self) -> Path:
        return self.reference("discopatch", "discriminator.pt")

    @property
    def discopatch_training(self) -> Path:
        return self.reference("discopatch", "training.json")

    @property
    def method_bank(self) -> Path:
        return self.reference("method", "bank.npz")

    @property
    def method_zstats(self) -> Path:
        return self.reference("method", "zstats.npz")

    def scores(self, name: str) -> Path:
        if name not in SCORE_KEYS:
            raise ValueError(f"unknown score folder {name!r}; choose from {sorted(SCORE_KEYS)}")
        return self.root / "scores" / name

    def score_file(self, name: str, image) -> Path:
        return self.scores(name) / f"{Path(image).stem}.npz"

    def report(self, name: str = "coco") -> Path:
        return self.root / "reports" / name

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def timing(self) -> Path:
        return self.root / "timing.json"


def atomic_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_npz(path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def load_npz(path, keys) -> dict:
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as data:
            loaded = {key: data[key] for key in data.files}
    except Exception as error:
        raise ValueError(f"malformed result file {path.name}: {path}") from error
    missing = set(keys) - set(loaded)
    if missing:
        raise ValueError(f"result file {path.name} lacks {sorted(missing)}: {path}")
    return loaded


def valid_existing(path, keys) -> bool:
    """False for a missing file; a damaged or incomplete one is refused rather than silently redone."""
    if not Path(path).exists():
        return False
    load_npz(path, keys)
    return True


def stack(folder, names, keys) -> dict:
    """The given arrays of every named image's file, stacked along a new first axis."""
    folder = Path(folder)
    missing = [n for n in names if not (folder / f"{Path(n).stem}.npz").exists()]
    if missing:
        raise ValueError(f"{folder.name} is incomplete: {len(missing)} of {len(names)} missing, e.g. {missing[0]}")
    columns = {key: [] for key in keys}
    for name in names:
        with np.load(folder / f"{Path(name).stem}.npz") as item:
            for key in keys:
                columns[key].append(item[key])
    return {key: np.stack(values) for key, values in columns.items()}


def sha1(path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def progress(label, done, total, started) -> None:
    rate = done / max(time.time() - started, 1e-9)
    remaining = (total - done) / max(rate, 1e-9)
    print(f"[{label}] {done}/{total} images, {rate:.2f}/s, about {remaining / 60:.0f} min left", flush=True)


def _plain(value):
    """The value as JSON gives it back, so that tuples and lists compare equal."""
    return json.loads(json.dumps(value))


class Manifest:
    """manifest.json: the run's protocol, its environment, and the inputs each score folder was computed from."""

    def __init__(self, layout: RunLayout):
        self.path = layout.manifest

    def read(self) -> dict:
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def update(self, **sections) -> None:
        data = self.read()
        data.update(_plain(sections))
        atomic_json(self.path, data)

    def check_protocol(self, protocol: dict) -> None:
        """Record the protocol of a new run folder; refuse a run folder made with another protocol."""
        protocol = _plain(protocol)
        recorded = self.read().get("protocol")
        if recorded is None:
            self.update(protocol=protocol)
        elif recorded != protocol:
            changed = sorted(k for k in set(recorded) | set(protocol) if recorded.get(k) != protocol.get(k))
            raise ValueError(f"this run folder was made with another protocol ({', '.join(changed)}): {self.path}")

    def check_inputs(self, folder: str, inputs: dict) -> None:
        """Record what a score folder is computed from; refuse to add to it if it was computed from something else."""
        inputs = _plain(inputs)
        data = self.read()
        recorded = data.get("inputs", {}).get(folder)
        if recorded is None:
            data.setdefault("inputs", {})[folder] = inputs
            atomic_json(self.path, data)
        elif recorded != inputs:
            changed = sorted(k for k in set(recorded) | set(inputs) if recorded.get(k) != inputs.get(k))
            raise ValueError(f"the inputs of scores/{folder} changed since its files were written "
                             f"({', '.join(changed)}): {self.path}")

    def record_environment(self, discopatch_root) -> None:
        """Package versions, the DisCoPatch commit and the GPU, recorded once."""
        if "environment" in self.read():
            return
        import torch
        versions = {}
        for package in PACKAGES:
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                versions[package] = None
        commit = subprocess.run(["git", "-C", str(discopatch_root), "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip() or None
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        self.update(environment={"packages": versions, "discopatch_commit": commit, "gpu": gpu})
```

- [ ] **Step 4: Implement `degradation_monitor/settings.py` and `configs/coco.toml`**

```python
"""Run settings: the machine's paths and run options, read from a TOML file such as configs/coco.toml.

The method's fixed choices (k = 50 neighbours, the stage-4 key, stages 1-3, the top 1%, the 2,000 + 500 reference
images) are constants in degradation_monitor.method, so that a config edit cannot change them.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

from .corruptions import CONDITIONS
from .datasets.coco import FOLDS, Coco
from .runs import RunLayout, sha256

PATH_FIELDS = ("run", "checkpoint", "train_images", "val_images", "annotations", "discopatch_root")


@dataclass(frozen=True)
class Settings:
    run: Path
    checkpoint: Path
    train_images: Path
    val_images: Path
    annotations: Path
    discopatch_root: Path
    device: str = "cuda:0"
    batch_size: int = 32
    workers: int = 9
    gpu_memory_gib: Optional[float] = None
    limit: Optional[int] = None
    epochs: int = 65
    seed: int = 44

    @property
    def layout(self) -> RunLayout:
        return RunLayout(self.run)

    @property
    def dataset(self) -> Coco:
        return Coco(self.train_images, self.val_images, self.annotations, seed=self.seed, limit=self.limit)

    def protocol(self) -> dict:
        """What every result in the run folder depends on; a stage refuses a run folder made with another one."""
        from .baselines.contrastive_conf import THETA
        from .baselines.knn import KNN_K, KNN_K_MAX
        from .detector.postprocess import TOP_K
        return {"dataset": "coco", "seed": self.seed, "limit": self.limit, "folds": FOLDS,
                "conditions": [list(c) for c in CONDITIONS], "checkpoint_sha256": sha256(self.checkpoint),
                "top_k": TOP_K, "knn_k": KNN_K, "knn_k_max": KNN_K_MAX, "theta": THETA}


def load_settings(path, **overrides) -> Settings:
    """Settings from a TOML file. Keyword overrides (from the command line) win; an override of None keeps the file's value."""
    path = Path(path)
    values = tomllib.loads(path.read_text())
    unknown = sorted(set(values) - {f.name for f in fields(Settings)})
    if unknown:
        raise ValueError(f"unknown settings in {path}: {', '.join(unknown)}")
    missing = [name for name in PATH_FIELDS if name not in values]
    if missing:
        raise ValueError(f"{path} needs {', '.join(missing)}")
    values.update({key: value for key, value in overrides.items() if value is not None})
    for name in PATH_FIELDS:
        values[name] = Path(values[name])
    return Settings(**values)
```

`configs/coco.toml`:

```toml
# This machine's paths and the run options. Stages: python -m degradation_monitor <stage> --config configs/coco.toml
# The method's fixed choices are constants in degradation_monitor/method/, not settings.
run = "/home/yuchen/YuchenZ/UE/philip_sa/runs/coco"
checkpoint = "/home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
train_images = "/home/yuchen/YuchenZ/Datasets/coco/train2017"
val_images = "/home/yuchen/YuchenZ/Datasets/coco/val2017"
annotations = "/home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json"
discopatch_root = "/home/yuchen/YuchenZ/UE/DisCoPatch"
device = "cuda:0"
batch_size = 8
workers = 9
gpu_memory_gib = 5.5  # the GPU is shared with other people's jobs
epochs = 65
seed = 44
```

- [ ] **Step 5: Run the tests**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/test_runs.py tests/test_settings.py`
Expected: 10 passed.

- [ ] **Step 6: Run the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: "187 passed, 1 skipped".

- [ ] **Step 7: Commit**

```bash
git add degradation_monitor/runs.py degradation_monitor/settings.py configs/coco.toml tests/test_runs.py tests/test_settings.py
git commit -q -m "feat: the run-folder layout, its manifest, and TOML settings" -m "runs.py names every reference and score path, writes atomically, refuses damaged results and keeps the manifest (protocol, environment, the inputs of each score folder); settings.py reads configs/coco.toml with command-line overrides and refuses unknown or missing keys." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

### Task 10: The baselines' stages

This rewrites the baselines' half of the 565-line `baselines/pipeline.py` as stages on the new run folder. The arithmetic is unchanged; what changes is where files go and how inputs are checked.
- `stages/common.py` holds what every stage shares.
- `stages/__init__.py` gets the registry and `run_stage`, which checks the protocol and records the environment before any stage runs. Tasks 11 and 12 add their stages to it.
- The old pipeline stays until Task 13.

**Files:**
- Create: `degradation_monitor/stages/common.py`, `degradation_monitor/stages/baselines.py`.
- Modify: `degradation_monitor/stages/__init__.py`.
- Test: `tests/stages/test_baselines.py`.

**Interfaces:**
- Consumes:
  - `Settings` and `Manifest` (Task 9);
  - `DetectorTap`, `top_detections`, `TOP_K` (Task 4);
  - `corruptions.load_variants` and `corruptions.digest` (Task 5);
  - `coco_map`, `coco_results`, `list_images` (Task 5);
  - the six baselines (Task 7);
  - `stage_zstats` and `zscored_sum` (Task 6).
- Produces:
  - `degradation_monitor.stages.common`:
    - `cap_gpu_memory(device, gib)`, `open_rgb(path)`, `image_size(array) -> (width, height)`;
    - `PreparedImages(paths)`, `clean_loader(settings, paths)`;
    - `bounded(pool, function, items, in_flight)`, `variant_stream(settings, paths)`;
    - `check_digests(layout, name, arrays)`, `pending_images(settings, folder) -> list[Path]`.
  - `degradation_monitor.stages.baselines`. Each stage takes `(settings, manifest)`:
    - `check`, `knn_bank`, `detector_pass`;
    - `discopatch_train`, `discopatch_pass`;
    - `hashemi_fit`, `cdf_fit`, `cdf_zstats`, `activation_pass`;
    - `timing`;
    - plus the helpers `detector_scores(tap, bank, arrays, batch_size, device)` and `activation_scores(tap, hashemi_monitor, cdf_monitor, zstats, arrays, batch_size)`.
  - `degradation_monitor.stages`: `STAGES: dict[str, callable]` and `run_stage(name, settings)`.

- [ ] **Step 1: Write the failing tests** in `tests/stages/test_baselines.py`. They are ported from `tests/differential_uncertainty/test_baselines_pipeline.py` to the new layout:

```python
import hashlib
import json

import numpy as np
import pytest
import torch
from PIL import Image

from degradation_monitor import corruptions
from degradation_monitor.runs import SCORE_KEYS, Manifest
from degradation_monitor.settings import Settings
from degradation_monitor.stages import baselines as stage
from degradation_monitor.stages import common, run_stage


class FakeTap:
    """Detector stand-in: fixed boxes, brightness-driven confidence and pooled features."""
    calls = 0

    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def run(self, arrays, batch_size=32):
        FakeTap.calls += 1
        n = len(arrays)
        brightness = np.array([a.mean() / 255.0 for a in arrays])
        logits = np.full((n, 300, 80), -6.0)
        logits[:, :3, 0] = (4.0 * brightness)[:, None]
        boxes = np.full((n, 300, 4), 0.25)
        pooled = np.random.default_rng(0).normal(size=(n, 512)) + brightness[:, None]
        return logits, boxes, pooled

    def prepare(self, arrays):
        return torch.stack([torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 255.0
                            for a in arrays])

    def forward(self, batch):
        return self.run([(b.permute(1, 2, 0).numpy() * 255).astype(np.uint8) for b in batch])

    hidden_calls = 0

    def forward_hidden(self, batch):
        FakeTap.hidden_calls += 1
        n = batch.shape[0]
        level = batch.float().mean(dim=(1, 2, 3))
        generator = torch.Generator().manual_seed(0)

        def maps(channels, sizes):
            return [level.view(n, 1, 1, 1).expand(n, channels, s, s).clone()
                    + torch.randn(channels, s, s, generator=generator) for s in sizes]

        return {"decoder": torch.randn(300, 256, generator=generator) + 4.0 * level.view(n, 1, 1),
                "encoder": maps(8, (4, 2, 1)), "backbone": maps(3, (4, 4, 2, 2, 1))}


def _images(folder, count, shape, seed):
    folder.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    for index in range(count):
        Image.fromarray(rng.integers(0, 256, (*shape, 3), dtype=np.uint8)).save(folder / f"{index:04d}.jpg")
    return folder


def _settings(tmp_path, **changes):
    (tmp_path / "ckpt.pth").write_bytes(b"weights")
    values = dict(run=tmp_path / "run", checkpoint=tmp_path / "ckpt.pth",
                  train_images=_images(tmp_path / "train", 3, (40, 50), 3),
                  val_images=_images(tmp_path / "val", 4, (48, 64), 1), annotations=tmp_path / "ann.json",
                  discopatch_root=tmp_path, limit=2, workers=0, device="cpu", batch_size=16)
    values.update(changes)
    return Settings(**values)


def _bank(settings):
    path = settings.layout.knn_bank
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.random.default_rng(0).normal(size=(256, 512)).astype(np.float16))


@pytest.fixture
def fakes(monkeypatch):
    FakeTap.calls = FakeTap.hidden_calls = 0
    monkeypatch.setattr(stage, "DetectorTap", FakeTap)


def _detector(settings):
    _bank(settings)
    run_stage("detector-pass", settings)


def _fake_scorer(monkeypatch, score):
    class FakeScorer:
        def __init__(self, *_a, **_k):
            pass

        def score(self, arrays, _image_id):
            return score(arrays)

    monkeypatch.setattr(stage, "DisCoPatchScorer", FakeScorer)


def _discriminator(settings, content=b"x"):
    path = settings.layout.discopatch_checkpoint
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_the_detector_pass_writes_one_complete_file_per_image_and_resumes(tmp_path, fakes):
    settings = _settings(tmp_path)
    _detector(settings)
    files = sorted(settings.layout.scores("detector").glob("*.npz"))
    assert len(files) == 2
    with np.load(files[0]) as data:
        assert set(SCORE_KEYS["detector"]) <= set(data.files)
        assert data["saod_top3"].shape == (96,) and data["knn"].shape == (96, 200)
        assert data["det_boxes"].shape == (96, 100, 4) and data["digests"].shape == (96,)
    inputs = Manifest(settings.layout).read()["inputs"]["detector"]
    assert inputs == {"knn_bank": hashlib.sha1(settings.layout.knn_bank.read_bytes()).hexdigest()}
    FakeTap.calls = 0
    run_stage("detector-pass", settings)
    assert FakeTap.calls == 0


def test_the_detector_pass_refuses_a_damaged_result_file(tmp_path, fakes):
    settings = _settings(tmp_path)
    _detector(settings)
    victim = sorted(settings.layout.scores("detector").glob("*.npz"))[0]
    victim.write_bytes(b"not an npz")
    with pytest.raises(ValueError, match=victim.name):
        run_stage("detector-pass", settings)


def test_a_run_folder_made_with_another_seed_is_refused_and_the_environment_is_recorded(tmp_path, fakes):
    _detector(_settings(tmp_path))
    manifest = Manifest(_settings(tmp_path).layout).read()
    assert "torch" in manifest["environment"]["packages"] and manifest["protocol"]["seed"] == 44
    with pytest.raises(ValueError, match="another protocol"):
        run_stage("detector-pass", _settings(tmp_path, seed=45))


def test_a_later_pass_needs_the_detector_pass_first(tmp_path, fakes, monkeypatch):
    settings = _settings(tmp_path)
    _discriminator(settings)
    _fake_scorer(monkeypatch, lambda arrays: np.zeros(len(arrays)))
    with pytest.raises(ValueError, match="run the detector-pass stage first"):
        run_stage("discopatch-pass", settings)


def test_the_discopatch_pass_detects_changed_corruptions(tmp_path, fakes, monkeypatch):
    settings = _settings(tmp_path)
    _detector(settings)
    _discriminator(settings)
    _fake_scorer(monkeypatch, lambda arrays: np.array([a.mean() / 255.0 for a in arrays]))
    victim = sorted(settings.layout.scores("detector").glob("*.npz"))[0]
    data = dict(np.load(victim))
    data["digests"] = np.array(["0" * 16] * 96)
    np.savez(victim, **data)
    with pytest.raises(ValueError, match="corruptions differ"):
        run_stage("discopatch-pass", settings)


def test_the_worker_pool_returns_every_image_in_order_with_identical_corruptions(tmp_path):
    paths = sorted(_settings(tmp_path).val_images.glob("*.jpg"))[:3]
    in_process = list(common.variant_stream(_settings(tmp_path, workers=0), paths))
    pooled = list(common.variant_stream(_settings(tmp_path, workers=1), paths))  # 2 in flight < 3 images
    assert [name for name, _ in pooled] == [p.name for p in paths]
    for (_, expected), (_, actual) in zip(in_process, pooled):
        assert [corruptions.digest(a) for a in actual] == [corruptions.digest(a) for a in expected]


def test_training_refuses_to_overwrite_a_finished_discriminator(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    calls = []
    monkeypatch.setattr(stage, "train_discopatch", lambda *a, **k: calls.append(k))
    _discriminator(settings)
    with pytest.raises(ValueError, match="already exists"):
        run_stage("discopatch-train", settings)
    assert calls == []


def test_a_lower_epoch_budget_keeps_the_run_folder_and_the_trained_discriminator_is_linked(tmp_path, fakes,
                                                                                            monkeypatch):
    def fake_train(paths, models_dir, **kwargs):
        trained = models_dir / "DisCoPatch" / "Discriminator_coco.pt"
        trained.parent.mkdir(parents=True)
        trained.write_bytes(b"trained")
        return trained

    monkeypatch.setattr(stage, "train_discopatch", fake_train)
    _detector(_settings(tmp_path))
    settings = _settings(tmp_path, epochs=30)  # the epoch budget is not part of the protocol
    run_stage("discopatch-train", settings)
    assert settings.layout.discopatch_checkpoint.read_bytes() == b"trained"
    training = json.loads(settings.layout.discopatch_training.read_text())
    assert training["epochs"] == 30 and training["numerics"] == stage.discopatch.TRAINING_NUMERICS


def test_discopatch_scores_stay_tied_to_one_checkpoint(tmp_path, fakes, monkeypatch):
    settings = _settings(tmp_path)
    _detector(settings)
    checkpoint = _discriminator(settings, b"first")
    _fake_scorer(monkeypatch, lambda arrays: np.array([a.mean() / 255.0 for a in arrays]))
    run_stage("discopatch-pass", settings)
    recorded = Manifest(settings.layout).read()["inputs"]["discopatch"]
    assert recorded == {"discriminator": hashlib.sha1(b"first").hexdigest()}
    checkpoint.write_bytes(b"second")
    with pytest.raises(ValueError, match="changed since"):
        run_stage("discopatch-pass", settings)


def test_discopatch_scores_refuse_non_finite_values(tmp_path, fakes, monkeypatch):
    settings = _settings(tmp_path)
    _detector(settings)
    _discriminator(settings)
    _fake_scorer(monkeypatch, lambda arrays: np.full(len(arrays), np.nan))
    with pytest.raises(ValueError, match="non-finite"):
        run_stage("discopatch-pass", settings)
    assert not list(settings.layout.scores("discopatch").glob("*.npz"))


def test_the_hashemi_fit_stores_intervals_for_every_monitored_layer_once(tmp_path, fakes):
    settings = _settings(tmp_path)
    run_stage("hashemi-fit", settings)
    with np.load(settings.layout.hashemi_intervals) as data:
        assert data["decoder_mean"].shape == (300, 256) and data["encoder_s8_std"].shape == (8, 4, 4)
        assert data["encoder_s32_mean"].shape == (8, 1, 1) and int(data["images"]) == 3
    fit = json.loads(settings.layout.hashemi_fit.read_text())
    assert fit["k"] == 2.0 and fit["images"] == 3
    calls = FakeTap.hidden_calls
    run_stage("hashemi-fit", settings)
    assert FakeTap.hidden_calls == calls


def test_the_cdf_fit_stores_ranges_and_reference_cdfs_for_the_five_stages_once(tmp_path, fakes):
    settings = _settings(tmp_path)
    run_stage("cdf-fit", settings)
    with np.load(settings.layout.cdf_reference) as data:
        assert data["C1_cdf"].shape == (3, 1000) and data["C5_low"].shape == (3,)
        assert np.allclose(data["C3_cdf"][:, -1], 1.0) and int(data["images"]) == 3
    assert json.loads(settings.layout.cdf_fit.read_text())["bins"] == 1000
    assert FakeTap.hidden_calls == 2   # one ranges pass and one histogram pass over the single batch
    run_stage("cdf-fit", settings)
    assert FakeTap.hidden_calls == 2


def test_the_cdf_zstats_standardise_each_stage_on_clean_train_images_once(tmp_path, fakes):
    settings = _settings(tmp_path)
    with pytest.raises(ValueError, match="cdf-fit"):
        run_stage("cdf-zstats", settings)
    run_stage("cdf-fit", settings)
    run_stage("cdf-zstats", settings)
    stats = json.loads(settings.layout.cdf_zstats.read_text())
    assert stats["images"] == 3 and len(stats["mean"]) == 5 and len(stats["std"]) == 5
    assert all(s > 0 for s in stats["std"])
    calls = FakeTap.hidden_calls
    run_stage("cdf-zstats", settings)
    assert FakeTap.hidden_calls == calls


def test_the_activation_pass_covers_every_variant_and_stays_tied_to_its_fits(tmp_path, fakes):
    settings = _settings(tmp_path)
    _detector(settings)
    run_stage("hashemi-fit", settings)
    with pytest.raises(ValueError, match="cdf-fit"):
        run_stage("activation-pass", settings)
    run_stage("cdf-fit", settings)
    with pytest.raises(ValueError, match="cdf-zstats"):
        run_stage("activation-pass", settings)
    run_stage("cdf-zstats", settings)
    run_stage("activation-pass", settings)
    files = sorted(settings.layout.scores("activations").glob("*.npz"))
    assert len(files) == 2
    with np.load(files[0]) as data:
        assert set(data.files) == set(SCORE_KEYS["activations"])
        assert all(data[key].shape == (96,) for key in ("hashemi_decoder", "hashemi_encoder", "cdf_backbone",
                                                         "cdf_backbone_z"))
        assert data["hashemi_encoder_maps"].shape == (96, 3) and data["cdf_stages"].shape == (96, 5)
        assert np.all((data["hashemi_decoder"] >= 0) & (data["hashemi_decoder"] <= 1))
        assert np.all(data["cdf_backbone"] >= 0)
        assert np.allclose(data["cdf_stages"].sum(axis=1), data["cdf_backbone"])
    reference = settings.layout.cdf_reference
    reference.write_bytes(reference.read_bytes() + b"x")
    with pytest.raises(ValueError, match="changed since"):
        run_stage("activation-pass", settings)


def test_the_activation_pass_detects_changed_corruptions(tmp_path, fakes):
    settings = _settings(tmp_path)
    _detector(settings)
    for name in ("hashemi-fit", "cdf-fit", "cdf-zstats"):
        run_stage(name, settings)
    victim = sorted(settings.layout.scores("detector").glob("*.npz"))[0]
    data = dict(np.load(victim))
    data["digests"] = np.array(["0" * 16] * 96)
    np.savez(victim, **data)
    with pytest.raises(ValueError, match="corruptions differ"):
        run_stage("activation-pass", settings)


def test_timing_reports_both_activation_monitors_once_they_are_fitted(tmp_path, fakes):
    settings = _settings(tmp_path)
    _bank(settings)
    run_stage("hashemi-fit", settings)
    run_stage("cdf-fit", settings)
    run_stage("timing", settings)
    timing = json.loads(settings.layout.timing.read_text())
    assert {"detector_ms", "detector_plus_hashemi_ms", "detector_plus_cdf_ms"} <= set(timing)
    assert "discopatch_ms" not in timing


def test_the_check_names_a_missing_input_before_loading_the_detector(tmp_path, fakes):
    settings = _settings(tmp_path)  # writes no annotation file
    with pytest.raises(ValueError, match="annotations does not exist"):
        run_stage("check", settings)
    assert FakeTap.calls == 0


def test_the_gpu_cap_limits_the_process_to_its_share_of_the_card(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda device: type("Properties", (), {"total_memory": 32 * 2**30})())
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction",
                        lambda fraction, device=None: calls.append((fraction, device)))
    common.cap_gpu_memory("cuda:0", 5.5)
    common.cap_gpu_memory("cuda:0", None)
    common.cap_gpu_memory("cpu", 5.5)
    assert calls == [(5.5 / 32, torch.device("cuda:0"))]
```

`tests/stages/` needs no `__init__.py`, because pytest runs with `--import-mode=importlib`.

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/stages/test_baselines.py`
Expected: ERROR at collection, `ImportError: cannot import name 'baselines' from 'degradation_monitor.stages'`.

- [ ] **Step 3: Implement `degradation_monitor/stages/common.py`**

`bounded` and the loader are verbatim from the old pipeline:

```python
"""What every stage shares: the GPU memory cap, image loading, the stream of corrupted variants and the input checks."""
from __future__ import annotations

import multiprocessing
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .. import corruptions
from ..detector.model import IMAGE_SIZE, prepare_image
from ..runs import SCORE_KEYS, load_npz, valid_existing


def cap_gpu_memory(device, gib) -> None:
    """Make this process run out of memory itself before it squeezes other jobs on a shared GPU."""
    device = torch.device(device)
    if gib is None or device.type != "cuda":
        return
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, gib * 2**30 / total), device)


def open_rgb(path) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(source.convert("RGB"), dtype=np.uint8).copy()


def image_size(array) -> tuple[int, int]:
    """(width, height) of an RGB array."""
    return array.shape[1], array.shape[0]


class PreparedImages(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as source:
            return prepare_image(source.convert("RGB"), IMAGE_SIZE)


def clean_loader(settings, paths) -> DataLoader:
    return DataLoader(PreparedImages(paths), batch_size=settings.batch_size, num_workers=settings.workers,
                      pin_memory=True)


def bounded(pool, function, items, in_flight):
    """Ordered results with at most `in_flight` tasks queued, so memory stays bounded."""
    iterator = iter(items)
    # range first: zip stops on range without pulling (and losing) an extra item
    queue = deque(pool.apply_async(function, (item,)) for _, item in zip(range(in_flight), iterator))
    while queue:
        result = queue.popleft().get()
        following = next(iterator, None)
        if following is not None:
            queue.append(pool.apply_async(function, (following,)))
        yield result


def variant_stream(settings, paths):
    """(file name, its 96 variants) for each image, in order, built by `settings.workers` processes."""
    if settings.workers == 0:
        for path in paths:
            yield corruptions.load_variants(path)
        return
    context = multiprocessing.get_context("spawn")
    with context.Pool(settings.workers) as pool:
        yield from bounded(pool, corruptions.load_variants, paths, 2 * settings.workers)


def check_digests(layout, name, arrays) -> None:
    """A later pass must see exactly the corruptions the detector pass saw."""
    stored = load_npz(layout.score_file("detector", name), SCORE_KEYS["detector"])["digests"]
    if list(stored) != [corruptions.digest(a) for a in arrays]:
        raise ValueError(f"corruptions differ from the detector pass for {name}")


def pending_images(settings, folder: str) -> list[Path]:
    """Evaluation images without a valid file in scores/<folder>; a later pass needs the detector pass's files."""
    layout = settings.layout
    pending = [p for p in settings.dataset.evaluation_images()
               if not valid_existing(layout.score_file(folder, p), SCORE_KEYS[folder])]
    if folder != "detector":
        absent = [p.name for p in pending if not layout.score_file("detector", p).exists()]
        if absent:
            raise ValueError(f"run the detector-pass stage first: {len(absent)} detector results are missing, "
                             f"e.g. {absent[0]}")
    return pending
```

- [ ] **Step 4: Implement `degradation_monitor/stages/baselines.py`**

```python
"""The baselines' stages: the clean references they need, and their per-image scores on every condition."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from .. import corruptions
from ..baselines import discopatch
from ..baselines.activation_cdf import BINS as CDF_BINS
from ..baselines.activation_cdf import MARGIN as CDF_MARGIN
from ..baselines.activation_cdf import STAGES as CDF_STAGES
from ..baselines.activation_cdf import ZSTAT_IMAGES as CDF_ZSTAT_IMAGES
from ..baselines.activation_cdf import CdfMonitor, ChannelRanges, ReferenceHistograms
from ..baselines.contrastive_conf import THETA, contrastive_parts, query_detections
from ..baselines.discopatch import DisCoPatchScorer, train_discopatch
from ..baselines.hashemi import K as HASHEMI_K
from ..baselines.hashemi import LAYERS as HASHEMI_LAYERS
from ..baselines.hashemi import HashemiMonitor, NeuronStats, save_intervals
from ..baselines.knn import KNN_K, KNN_K_MAX, knn_distances, normalize_rows
from ..baselines.saod import saod_uncertainty
from ..datasets.coco import coco_map, coco_results, list_images
from ..detector.postprocess import TOP_K, top_detections
from ..detector.taps import DetectorTap
from ..evaluation.metrics import stage_zstats, zscored_sum
from ..runs import SCORE_KEYS, atomic_json, atomic_npz, progress, sha1
from ..settings import PATH_FIELDS
from .common import (cap_gpu_memory, check_digests, clean_loader, image_size, open_rgb, pending_images,
                     variant_stream)

TIMING_IMAGES = 100
TIMING_WARMUP = 10
MIN_CLEAN_AP = 0.45  # the checkpoint's clean COCO val AP is about 0.48


def _load_bank(settings, device) -> torch.Tensor:
    path = settings.layout.knn_bank
    if not path.exists():
        raise ValueError(f"run the knn-bank stage first: {path} is missing")
    return normalize_rows(torch.from_numpy(np.load(path)).float()).to(device).half()


def detector_scores(tap, bank, arrays, batch_size, device) -> dict:
    """SAOD's confidences, ContrastiveConf's parts, kNN distances and the top detections of every array."""
    logits, boxes, pooled = tap.run(arrays, batch_size)
    size = image_size(arrays[0])
    detections = [top_detections(l, b, size, TOP_K) for l, b in zip(logits, boxes)]
    conf_pos, conf_neg = contrastive_parts([query_detections(l, b, size) for l, b in zip(logits, boxes)], THETA)
    knn = knn_distances(torch.from_numpy(pooled).to(device), bank, KNN_K_MAX).cpu().numpy()
    return {
        "saod_min": np.array([saod_uncertainty(d[0], 1) for d in detections]),
        "saod_top3": np.array([saod_uncertainty(d[0], 3) for d in detections]),
        "conf_pos": conf_pos, "conf_neg": conf_neg, "knn": knn.astype(np.float32),
        "det_scores": np.stack([d[0] for d in detections]),
        "det_labels": np.stack([d[1] for d in detections]).astype(np.int16),
        "det_boxes": np.stack([d[2] for d in detections]),
        "size": np.array(size),
    }


def check(settings, manifest) -> float:
    """Every input exists, then the detector's clean AP on every COCO val image: about 0.48 for this checkpoint.

    run_stage has already recorded the checkpoint's sha256 in the protocol.
    """
    for name in PATH_FIELDS:
        if name != "run" and not getattr(settings, name).exists():
            raise ValueError(f"{name} does not exist: {getattr(settings, name)}")
    gt = settings.dataset.ground_truth()
    images = list_images(settings.val_images)
    results = []
    cap_gpu_memory(settings.device, settings.gpu_memory_gib)
    with DetectorTap(settings.checkpoint, settings.device) as tap:
        for start in range(0, len(images), settings.batch_size):
            chunk = images[start:start + settings.batch_size]
            arrays = [open_rgb(p) for p in chunk]
            logits, boxes, _ = tap.run(arrays, settings.batch_size)
            for path, array, l, b in zip(chunk, arrays, logits, boxes):
                s, labels, xyxy = top_detections(l, b, image_size(array), TOP_K)
                results += coco_results(gt.image_id(path.name), s, labels, xyxy, gt.category_ids)
    ap = coco_map(gt, results, [gt.image_id(p.name) for p in images])
    manifest.update(check={"coco_val_ap": ap, "images": len(images)})
    print(f"[check] clean COCO val AP = {ap:.4f} on {len(images)} images", flush=True)
    if ap < MIN_CLEAN_AP:
        raise RuntimeError(f"clean COCO val AP is {ap:.3f}; expected about 0.48 for this checkpoint")
    return ap


def knn_bank(settings, manifest) -> None:
    """The kNN baseline's bank: L2-normalised pooled last-stage features of every clean train image, float16."""
    layout = settings.layout
    if layout.knn_bank.exists():
        return
    paths = settings.dataset.train_images()
    features, started = [], time.time()
    cap_gpu_memory(settings.device, settings.gpu_memory_gib)
    with DetectorTap(settings.checkpoint, settings.device) as tap:
        for index, batch in enumerate(clean_loader(settings, paths)):
            _, _, pooled = tap.forward(batch)
            features.append(normalize_rows(torch.from_numpy(pooled)).numpy().astype(np.float16))
            if index % 200 == 0:
                progress("knn-bank", min((index + 1) * settings.batch_size, len(paths)), len(paths), started)
    layout.knn_bank.parent.mkdir(parents=True, exist_ok=True)
    temporary = layout.knn_bank.with_name(".bank.tmp.npy")
    np.save(temporary, np.concatenate(features))
    temporary.replace(layout.knn_bank)
    atomic_json(layout.knn_names, [p.name for p in paths])


def detector_pass(settings, manifest) -> None:
    """Every evaluation image under all 96 conditions through the detector: SAOD, ContrastiveConf, kNN, detections."""
    layout = settings.layout
    pending = pending_images(settings, "detector")
    if not pending:
        return
    bank = _load_bank(settings, settings.device)
    manifest.check_inputs("detector", {"knn_bank": sha1(layout.knn_bank)})
    cap_gpu_memory(settings.device, settings.gpu_memory_gib)
    started = time.time()
    with DetectorTap(settings.checkpoint, settings.device) as tap:
        for done, (name, arrays) in enumerate(variant_stream(settings, pending), start=1):
            values = detector_scores(tap, bank, arrays, settings.batch_size, settings.device)
            values["digests"] = np.array([corruptions.digest(a) for a in arrays])
            atomic_npz(layout.score_file("detector", name), **values)
            if done % 25 == 0:
                progress("detector-pass", done, len(pending), started)


def discopatch_train(settings, manifest) -> None:
    """Train DisCoPatch's discriminator on clean train patches with the official code and README settings."""
    layout = settings.layout
    # The official loop saves no optimiser state, so a finished discriminator is never trained over.
    if layout.discopatch_checkpoint.exists():
        raise ValueError(f"{layout.discopatch_checkpoint} already exists; remove it to train again")
    atomic_json(layout.discopatch_training,
                {"epochs": settings.epochs, "seed": settings.seed, "numerics": discopatch.TRAINING_NUMERICS})
    trained = train_discopatch(settings.dataset.train_images(), layout.discopatch_dir / "training",
                               epochs=settings.epochs, num_workers=settings.workers, seed=settings.seed,
                               root=settings.discopatch_root)
    os.link(trained, layout.discopatch_checkpoint)


def discopatch_pass(settings, manifest) -> None:
    layout = settings.layout
    checkpoint = layout.discopatch_checkpoint
    if not checkpoint.exists():
        raise ValueError(f"run the discopatch-train stage first: {checkpoint} is missing")
    manifest.check_inputs("discopatch", {"discriminator": sha1(checkpoint)})
    pending = pending_images(settings, "discopatch")
    if not pending:
        return
    cap_gpu_memory(settings.device, settings.gpu_memory_gib)
    scorer = DisCoPatchScorer(checkpoint, layout.discopatch_dir / "training", settings.device,
                              root=settings.discopatch_root)
    started = time.time()
    for done, (name, arrays) in enumerate(variant_stream(settings, pending), start=1):
        check_digests(layout, name, arrays)
        dcp = scorer.score(arrays, name)
        if not np.isfinite(dcp).all():
            raise ValueError(f"DisCoPatch returned non-finite scores for {name}")
        atomic_npz(layout.score_file("discopatch", name), dcp=dcp)
        if done % 25 == 0:
            progress("discopatch-pass", done, len(pending), started)


def hashemi_fit(settings, manifest) -> None:
    """Per-neuron mean and standard deviation over every clean COCO train image (Hashemi et al., Sec. 3.1)."""
    layout = settings.layout
    if layout.hashemi_intervals.exists():
        return
    paths = settings.dataset.train_images()
    stats = {name: NeuronStats() for name in HASHEMI_LAYERS}
    cap_gpu_memory(settings.device, settings.gpu_memory_gib)
    started = time.time()
    with DetectorTap(settings.checkpoint, settings.device, hidden=True) as tap:
        for index, batch in enumerate(clean_loader(settings, paths)):
            hidden = tap.forward_hidden(batch)
            if len(hidden["encoder"]) != len(HASHEMI_LAYERS) - 1:
                raise ValueError(f"expected the encoder's three output maps, got {len(hidden['encoder'])}")
            stats["decoder"].update(hidden["decoder"])
            for name, values in zip(HASHEMI_LAYERS[1:], hidden["encoder"]):
                stats[name].update(values)
            if index % 200 == 0:
                progress("hashemi-fit", min((index + 1) * settings.batch_size, len(paths)), len(paths), started)
    save_intervals(layout.hashemi_intervals, stats, images=len(paths))
    atomic_json(layout.hashemi_fit, {"images": len(paths), "k": HASHEMI_K, "std": "population (ddof=0)",
                                     "layers": list(HASHEMI_LAYERS)})


def cdf_fit(settings, manifest) -> None:
    """Per-channel ranges, then training histograms of backbone stages C1-C5 (Becker et al., ICPR 2026)."""
    layout = settings.layout
    if layout.cdf_reference.exists():
        return
    paths = settings.dataset.train_images()
    ranges = {stage: ChannelRanges() for stage in CDF_STAGES}
    cap_gpu_memory(settings.device, settings.gpu_memory_gib)
    with DetectorTap(settings.checkpoint, settings.device, hidden=True) as tap:
        started = time.time()
        for index, batch in enumerate(clean_loader(settings, paths)):
            stages = tap.forward_hidden(batch)["backbone"]
            if len(stages) != len(CDF_STAGES):
                raise ValueError(f"expected the five backbone stages, got {len(stages)}")
            for stage, values in zip(CDF_STAGES, stages):
                ranges[stage].update(values)
            if index % 200 == 0:
                progress("cdf-fit ranges", min((index + 1) * settings.batch_size, len(paths)), len(paths), started)
        reference = ReferenceHistograms({s: r.result() for s, r in ranges.items()}, settings.device)
        started = time.time()
        for index, batch in enumerate(clean_loader(settings, paths)):
            reference.update(tap.forward_hidden(batch)["backbone"])
            if index % 200 == 0:
                progress("cdf-fit histograms", min((index + 1) * settings.batch_size, len(paths)), len(paths), started)
    reference.save(layout.cdf_reference, images=len(paths))
    atomic_json(layout.cdf_fit, {"images": len(paths), "bins": CDF_BINS, "margin": CDF_MARGIN,
                                 "stages": list(CDF_STAGES), "passes": 2})


def cdf_zstats(settings, manifest) -> None:
    """Mean and spread of each stage's EMD over a seeded sample of clean train images, for the z-scored sum."""
    layout = settings.layout
    if layout.cdf_zstats.exists():
        return
    if not layout.cdf_reference.exists():
        raise ValueError(f"run the cdf-fit stage first: {layout.cdf_reference} is missing")
    paths = settings.dataset.train_images()
    chosen = np.sort(np.random.default_rng(settings.seed).choice(len(paths), size=min(CDF_ZSTAT_IMAGES, len(paths)),
                                                                replace=False))
    monitor = CdfMonitor(layout.cdf_reference, settings.device)
    values = []
    cap_gpu_memory(settings.device, settings.gpu_memory_gib)
    with DetectorTap(settings.checkpoint, settings.device, hidden=True) as tap:
        for batch in clean_loader(settings, [paths[i] for i in chosen]):
            values.append(monitor.stage_scores(tap.forward_hidden(batch)["backbone"]))
    mean, std = stage_zstats(np.concatenate(values))
    atomic_json(layout.cdf_zstats, {"images": len(chosen), "seed": settings.seed, "stages": list(CDF_STAGES),
                                    "mean": mean.tolist(), "std": std.tolist(),
                                    "reference_sha1": sha1(layout.cdf_reference)})


def activation_scores(tap, hashemi_monitor, cdf_monitor, zstats, arrays, batch_size) -> dict:
    """Both activation monitors' scores for every array, from one hidden-layer pass."""
    parts = {key: [] for key in SCORE_KEYS["activations"]}
    for start in range(0, len(arrays), batch_size):
        hidden = tap.forward_hidden(tap.prepare(arrays[start:start + batch_size]))
        decoder_share, encoder_share = hashemi_monitor.scores(hidden["decoder"], hidden["encoder"])
        stage_scores = cdf_monitor.stage_scores(hidden["backbone"])
        parts["hashemi_decoder"].append(decoder_share)
        parts["hashemi_encoder"].append(encoder_share)
        parts["hashemi_encoder_maps"].append(hashemi_monitor.encoder_shares(hidden["encoder"]))
        parts["cdf_backbone"].append(stage_scores.sum(axis=1))
        parts["cdf_backbone_z"].append(zscored_sum(stage_scores, zstats["mean"], zstats["std"]))
        parts["cdf_stages"].append(stage_scores)
    return {key: np.concatenate(values) for key, values in parts.items()}


def activation_pass(settings, manifest) -> None:
    """Both activation monitors in one pass over every evaluation image and condition."""
    layout = settings.layout
    fits = {"hashemi-fit": layout.hashemi_intervals, "cdf-fit": layout.cdf_reference, "cdf-zstats": layout.cdf_zstats}
    missing = [stage for stage, path in fits.items() if not path.exists()]
    if missing:
        raise ValueError("run the " + " and ".join(missing) + " stage first")
    manifest.check_inputs("activations", {"hashemi_intervals": sha1(layout.hashemi_intervals),
                                          "cdf_reference": sha1(layout.cdf_reference),
                                          "cdf_zstats": sha1(layout.cdf_zstats),
                                          "hashemi_k": HASHEMI_K, "cdf_bins": CDF_BINS})
    pending = pending_images(settings, "activations")
    if not pending:
        return
    hashemi_monitor = HashemiMonitor(layout.hashemi_intervals, settings.device)
    cdf_monitor = CdfMonitor(layout.cdf_reference, settings.device)
    zstats = json.loads(layout.cdf_zstats.read_text())
    cap_gpu_memory(settings.device, settings.gpu_memory_gib)
    started = time.time()
    with DetectorTap(settings.checkpoint, settings.device, hidden=True) as tap:
        for done, (name, arrays) in enumerate(variant_stream(settings, pending), start=1):
            check_digests(layout, name, arrays)
            values = activation_scores(tap, hashemi_monitor, cdf_monitor, zstats, arrays, settings.batch_size)
            if not all(np.isfinite(v).all() for v in values.values()):
                raise ValueError(f"an activation monitor returned non-finite scores for {name}")
            atomic_npz(layout.score_file("activations", name), **values)
            if done % 25 == 0:
                progress("activation-pass", done, len(pending), started)


def timing(settings, manifest) -> None:
    """Median ms per image at batch 1 on a warm GPU, preprocessing included."""
    layout = settings.layout
    images = [open_rgb(p) for p in settings.dataset.evaluation_images()[:TIMING_IMAGES]]
    bank = _load_bank(settings, settings.device)
    on_gpu = torch.cuda.is_available() and settings.device.startswith("cuda")

    def timed(function):
        for array in images[:TIMING_WARMUP]:
            function(array)
        values = []
        for array in images:
            if on_gpu:
                torch.cuda.synchronize()
            start = time.perf_counter()
            function(array)
            if on_gpu:
                torch.cuda.synchronize()
            values.append(1000.0 * (time.perf_counter() - start))
        return float(np.median(values))

    result = {"images": len(images), "batch_size": 1, "device": settings.device}
    with DetectorTap(settings.checkpoint, settings.device) as tap:
        def detector(array):
            return tap.forward(tap.prepare([array]))

        def saod(array):
            logits, boxes, _ = detector(array)
            saod_uncertainty(top_detections(logits[0], boxes[0], image_size(array), TOP_K)[0], 3)

        def contrastive(array):
            logits, boxes, _ = detector(array)
            contrastive_parts([query_detections(logits[0], boxes[0], image_size(array))], THETA)

        def knn(array):
            _, _, pooled = detector(array)
            knn_distances(torch.from_numpy(pooled).to(settings.device), bank, KNN_K)

        # every *_plus_* figure includes the detector forward pass it depends on
        result.update(detector_ms=timed(detector), detector_plus_saod_ms=timed(saod),
                      detector_plus_contrastive_ms=timed(contrastive), detector_plus_knn_ms=timed(knn))
    if layout.discopatch_checkpoint.exists():
        scorer = DisCoPatchScorer(layout.discopatch_checkpoint, layout.discopatch_dir / "training", settings.device,
                                  root=settings.discopatch_root)
        result["discopatch_ms"] = timed(lambda array: scorer.score([array], "timing.jpg"))
    if layout.hashemi_intervals.exists() or layout.cdf_reference.exists():
        # a separate tap, so the hidden-layer hooks never touch the other timings
        with DetectorTap(settings.checkpoint, settings.device, hidden=True) as hidden_tap:
            if layout.hashemi_intervals.exists():
                hashemi_monitor = HashemiMonitor(layout.hashemi_intervals, settings.device)

                def hashemi_score(array):
                    hashemi_monitor.decoder_share(hidden_tap.forward_hidden(hidden_tap.prepare([array]))["decoder"])

                result["detector_plus_hashemi_ms"] = timed(hashemi_score)
            if layout.cdf_reference.exists():
                cdf_monitor = CdfMonitor(layout.cdf_reference, settings.device)

                def cdf_score(array):
                    cdf_monitor.scores(hidden_tap.forward_hidden(hidden_tap.prepare([array]))["backbone"])

                result["detector_plus_cdf_ms"] = timed(cdf_score)
    atomic_json(layout.timing, result)
    print(f"[timing] {result}", flush=True)
```

- [ ] **Step 5: Implement the registry** in `degradation_monitor/stages/__init__.py`:

```python
"""The runnable, resumable steps. Each refuses inputs that changed since its results were written."""
from __future__ import annotations

from . import baselines

STAGES = {
    "check": baselines.check,
    "knn-bank": baselines.knn_bank,
    "detector-pass": baselines.detector_pass,
    "discopatch-train": baselines.discopatch_train,
    "discopatch-pass": baselines.discopatch_pass,
    "hashemi-fit": baselines.hashemi_fit,
    "cdf-fit": baselines.cdf_fit,
    "cdf-zstats": baselines.cdf_zstats,
    "activation-pass": baselines.activation_pass,
    "timing": baselines.timing,
}


def run_stage(name: str, settings) -> None:
    """Check the run folder's protocol, record the environment once, then run one stage."""
    from ..runs import Manifest
    if name not in STAGES:
        raise ValueError(f"unknown stage {name!r}; choose from {sorted(STAGES)}")
    settings.layout.root.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(settings.layout)
    manifest.check_protocol(settings.protocol())
    manifest.record_environment(settings.discopatch_root)
    STAGES[name](settings, manifest)
```

- [ ] **Step 6: Run the tests**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/stages/test_baselines.py`
Expected: 18 passed.

- [ ] **Step 7: Run the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: "205 passed, 1 skipped" (187 + 18).

- [ ] **Step 8: Commit**

```bash
git add degradation_monitor/stages tests/stages/test_baselines.py
git commit -q -m "feat: the baselines' stages on the new run folder" -m "The baselines' half of the old pipeline becomes stages/baselines.py on the new layout, with the same arithmetic; stages/common.py holds the GPU cap, the variant stream and the input checks, and run_stage checks the protocol before any stage." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

### Task 11: The method's stages

The method has two stages. Both store each channel's level and top-1% mean, and nothing else. The report compares the stored values with the reference, so the ablations of `docs/todo.md` (bank size, k, the average instead of similar scenes) need no new pass.
- `method-reference`: the statistics of the 2,000 bank and 500 z-statistics images, the seeded disjoint draws of COCO train from Task 5.
- `method-pass`: the statistics of every evaluation image under all 96 conditions, resumable image by image.

A transitional test runs the old `convtu-channels` and `convtu-means` phases and the new stages on the same fake backbone and images. It requires the stored arrays to be equal bit for bit. This is the premise of Task 14, which converts the stored arrays instead of recomputing them.

**Files:**
- Create: `degradation_monitor/stages/method.py`.
- Modify: `degradation_monitor/stages/__init__.py`.
- Test: `tests/stages/test_method.py`, plus a transitional test in `tests/differential_uncertainty/test_convtu_pipeline.py`.

**Interfaces:**
- Consumes:
  - `EarlyChannelTaps`, `load_frozen_detector`, `prepare_image`, `IMAGE_SIZE` (Task 4);
  - `Coco.reference_split(name)` (Task 5);
  - `channel_statistics`, `KEYS` (Task 8);
  - `RunLayout.method_bank`, `RunLayout.method_zstats`, `atomic_npz`, `progress` (Task 9);
  - `cap_gpu_memory`, `check_digests`, `clean_loader`, `pending_images`, `variant_stream` and the registry (Task 10).
- Produces: `degradation_monitor.stages.method`:
  - `early_taps(settings) -> EarlyChannelTaps`;
  - `batch_statistics(taps, batch) -> {KEYS: float32 (N, C)}`;
  - `image_statistics(taps, arrays, batch_size) -> {KEYS: float32 (96, C)}`;
  - the stages `method_reference(settings, manifest)` and `method_pass(settings, manifest)`, registered as `method-reference` and `method-pass`.

- [ ] **Step 1: Write the failing tests**

`tests/stages/test_method.py`:

```python
import numpy as np
import pytest
import torch
from PIL import Image

from fakes import FakeBackbone, FakeTap
from degradation_monitor.datasets import coco
from degradation_monitor.detector.taps import EarlyChannelTaps
from degradation_monitor.method.statistics import KEYS
from degradation_monitor.settings import Settings
from degradation_monitor.stages import baselines as baseline_stages
from degradation_monitor.stages import common, run_stage
from degradation_monitor.stages import method as stage

WIDTHS = (2, 3, 4, 5)  # the fake backbone's four stages


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setattr(coco, "SPLIT_IMAGES", (("reserved", 2), ("bank", 6), ("zstats", 3)))
    monkeypatch.setattr(stage, "early_taps", lambda _settings: EarlyChannelTaps(FakeBackbone()))
    monkeypatch.setattr(baseline_stages, "DetectorTap", FakeTap)
    rng = np.random.default_rng(3)
    for folder, count in (("train", 12), ("val", 3)):
        (tmp_path / folder).mkdir()
        for index in range(count):
            Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)).save(tmp_path / folder / f"{index:04d}.jpg")
    (tmp_path / "ckpt.pth").write_bytes(b"weights")
    result = Settings(run=tmp_path / "run", checkpoint=tmp_path / "ckpt.pth", train_images=tmp_path / "train",
                      val_images=tmp_path / "val", annotations=tmp_path / "ann.json", discopatch_root=tmp_path,
                      limit=2, workers=0, device="cpu", batch_size=4)
    result.layout.knn_bank.parent.mkdir(parents=True)
    np.save(result.layout.knn_bank, rng.normal(size=(256, 512)).astype(np.float16))
    return result


def _statistics_of(paths) -> dict:
    prepared = common.PreparedImages(paths)
    with EarlyChannelTaps(FakeBackbone()) as taps:
        return stage.batch_statistics(taps, torch.stack([prepared[i] for i in range(len(prepared))]))


def test_the_method_reference_stores_the_seeded_bank_and_zstatistics_images_once(settings):
    run_stage("method-reference", settings)
    paths = sorted(settings.train_images.glob("*.jpg"))
    splits = coco.train_splits(len(paths), seed=44)
    for split, path, rows in (("bank", settings.layout.method_bank, 6), ("zstats", settings.layout.method_zstats, 3)):
        expected = _statistics_of([paths[i] for i in splits[split]])
        with np.load(path) as stored:
            assert set(stored.files) == set(KEYS)
            assert [stored[f"top_s{s}"].shape for s in range(1, 5)] == [(rows, c) for c in WIDTHS]
            assert all(stored[key].dtype == np.float32 for key in KEYS)
            for key in KEYS:
                np.testing.assert_allclose(stored[key], expected[key], rtol=1e-6)
    stamp = settings.layout.method_bank.stat().st_mtime_ns
    run_stage("method-reference", settings)
    assert settings.layout.method_bank.stat().st_mtime_ns == stamp


def test_the_method_pass_covers_every_evaluation_image_and_resumes(settings):
    with pytest.raises(ValueError, match="run the detector-pass stage first"):
        run_stage("method-pass", settings)
    run_stage("detector-pass", settings)
    run_stage("method-pass", settings)
    files = sorted(settings.layout.scores("method").glob("*.npz"))
    assert [f.stem for f in files] == sorted(p.stem for p in settings.dataset.evaluation_images())
    with np.load(files[0]) as stored:
        assert set(stored.files) == set(KEYS)
        for statistic in ("means", "top"):
            assert [stored[f"{statistic}_s{s}"].shape for s in range(1, 5)] == [(96, c) for c in WIDTHS]
        assert all(stored[key].dtype == np.float32 for key in KEYS)
    stamp = files[0].stat().st_mtime_ns
    run_stage("method-pass", settings)
    assert files[0].stat().st_mtime_ns == stamp


def test_an_images_clean_condition_gets_the_statistics_the_reference_stage_would_give_it(settings):
    run_stage("detector-pass", settings)
    run_stage("method-pass", settings)
    path = settings.dataset.evaluation_images()[0]
    expected = _statistics_of([path])
    with np.load(settings.layout.score_file("method", path)) as stored:
        for key in KEYS:
            np.testing.assert_allclose(stored[key][0], expected[key][0], rtol=1e-6)


def test_the_method_pass_detects_changed_corruptions(settings):
    run_stage("detector-pass", settings)
    victim = sorted(settings.layout.scores("detector").glob("*.npz"))[0]
    stored = dict(np.load(victim))
    stored["digests"] = stored["digests"].copy()
    stored["digests"][11] = "0" * 16
    np.savez(victim, **stored)
    with pytest.raises(ValueError, match="corruptions differ"):
        run_stage("method-pass", settings)
```

Append to `tests/differential_uncertainty/test_convtu_pipeline.py`. The test is transitional, and Task 13 deletes the file:

```python
def test_the_new_method_stages_store_exactly_what_the_old_phases_stored(small, detector, monkeypatch, tmp_path):
    from degradation_monitor.datasets import coco
    from degradation_monitor.detector.taps import EarlyChannelTaps
    from degradation_monitor.method.statistics import KEYS
    from degradation_monitor.settings import Settings
    from degradation_monitor.stages import baselines as baseline_stages
    from degradation_monitor.stages import method as method_stages
    from degradation_monitor.stages import run_stage
    monkeypatch.setattr(convtu, "PILOT_IMAGES", 2)
    _channels(small)
    baselines.run_phase("convtu-means", small)
    monkeypatch.setattr(coco, "SPLIT_IMAGES", (("reserved", 2), ("bank", 6), ("zstats", 3)))  # the fixture's sizes
    monkeypatch.setattr(method_stages, "early_taps", lambda _settings: EarlyChannelTaps(FakeBackbone()))
    monkeypatch.setattr(baseline_stages, "DetectorTap", FakeTap)
    small.checkpoint.write_bytes(b"weights")
    settings = Settings(run=tmp_path / "run", checkpoint=small.checkpoint, train_images=small.train_images,
                        val_images=small.val_images, annotations=small.annotations, discopatch_root=tmp_path,
                        limit=2, workers=0, device="cpu", batch_size=small.batch_size)
    settings.layout.knn_bank.parent.mkdir(parents=True)
    np.save(settings.layout.knn_bank, np.random.default_rng(0).normal(size=(256, 512)).astype(np.float16))
    for name in ("detector-pass", "method-reference", "method-pass"):
        run_stage(name, settings)
    pairs = [(convtu.channels_bank_path(small), settings.layout.method_bank),
             (convtu.channels_zstats_path(small), settings.layout.method_zstats)]
    pairs += [(small.output / convtu.MEANS_FOLDER / p.name, p)
              for p in sorted(settings.layout.scores("method").glob("*.npz"))]
    assert len(pairs) == 4
    for old, new in pairs:
        with np.load(old) as before, np.load(new) as after:
            for key in KEYS:
                np.testing.assert_array_equal(after[key], before[key])
```

`FakeTap` there is `tests/fakes.py`'s, which the file already imports. Its `run` is all the detector pass calls.

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/stages/test_method.py tests/differential_uncertainty/test_convtu_pipeline.py`
Expected:
- an ERROR at collection of `test_method.py`: `ImportError: cannot import name 'method' from 'degradation_monitor.stages'`;
- the transitional test FAILs with the same `ImportError`;
- the old pipeline tests pass.

- [ ] **Step 3: Implement `degradation_monitor/stages/method.py`**

```python
"""Our method's stages: the clean reference statistics, then those of every evaluation image under all 96 conditions.

Both store each channel's level and top-1% mean. The report compares them with the reference, so an ablation of the
bank size, k or the key needs no new pass.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from PIL import Image

from ..detector.model import IMAGE_SIZE, load_frozen_detector, prepare_image
from ..detector.taps import EarlyChannelTaps
from ..method.statistics import KEYS, channel_statistics
from ..runs import atomic_npz, progress
from .common import cap_gpu_memory, check_digests, clean_loader, pending_images, variant_stream


def early_taps(settings) -> EarlyChannelTaps:
    return EarlyChannelTaps(load_frozen_detector(settings.checkpoint, torch.device(settings.device)).backbone)


def batch_statistics(taps, batch) -> dict:
    """Each channel's level and top-1% mean in the four early maps of a prepared batch: KEYS -> float32 (N, C)."""
    out = {}
    for stage, maps in enumerate(taps(batch), start=1):
        for statistic, values in channel_statistics(maps).items():
            out[f"{statistic}_s{stage}"] = values
    return out


def _concatenated(parts: list) -> dict:
    return {key: np.concatenate([part[key] for part in parts]) for key in KEYS}


def image_statistics(taps, arrays, batch_size) -> dict:
    """The statistics of every condition of one image: KEYS -> float32 (96, C)."""
    parts = []
    for start in range(0, len(arrays), batch_size):
        batch = torch.stack([prepare_image(Image.fromarray(a), IMAGE_SIZE) for a in arrays[start:start + batch_size]])
        parts.append(batch_statistics(taps, batch))
    out = _concatenated(parts)
    if not all(np.isfinite(value).all() for value in out.values()):
        raise ValueError("channel statistics are not finite")
    return out


def method_reference(settings, manifest) -> None:
    """The statistics of the bank and z-statistics images: seeded, disjoint draws of clean COCO train images."""
    layout = settings.layout
    pending = {split: path for split, path in (("bank", layout.method_bank), ("zstats", layout.method_zstats))
               if not path.exists()}
    if not pending:
        return
    cap_gpu_memory(settings.device, settings.gpu_memory_gib)
    with early_taps(settings) as taps:
        for split, path in pending.items():
            loader = clean_loader(settings, settings.dataset.reference_split(split))
            atomic_npz(path, **_concatenated([batch_statistics(taps, batch) for batch in loader]))


def method_pass(settings, manifest) -> None:
    """The statistics of every evaluation image under all 96 conditions, resumable image by image."""
    layout = settings.layout
    pending = pending_images(settings, "method")
    if not pending:
        return
    cap_gpu_memory(settings.device, settings.gpu_memory_gib)
    started = time.time()
    with early_taps(settings) as taps:
        for done, (name, arrays) in enumerate(variant_stream(settings, pending), start=1):
            check_digests(layout, name, arrays)
            atomic_npz(layout.score_file("method", name), **image_statistics(taps, arrays, settings.batch_size))
            if done % 25 == 0:
                progress("method-pass", done, len(pending), started)
```

- [ ] **Step 4: Register the two stages** in `degradation_monitor/stages/__init__.py`:
- Change `from . import baselines` to `from . import baselines, method`.
- Add two entries after `"activation-pass": baselines.activation_pass,`:

```python
    "method-reference": method.method_reference,
    "method-pass": method.method_pass,
```

- [ ] **Step 5: Run the tests**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/stages/test_method.py tests/differential_uncertainty/test_convtu_pipeline.py`
Expected: all pass, including the transitional test.

- [ ] **Step 6: Run the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: "210 passed, 1 skipped" (205 + 4 method stage tests + 1 transitional test).

- [ ] **Step 7: Commit**

```bash
git add degradation_monitor/stages tests/stages/test_method.py tests/differential_uncertainty/test_convtu_pipeline.py
git commit -q -m "feat: the method's reference and pass stages" -m "method-reference stores the level and top-1% mean of the 2,000 bank and 500 z-statistics images, and method-pass those of every evaluation image under all 96 conditions. A transitional test shows that both store exactly what the old convtu-channels and convtu-means phases stored." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

### Task 12: The report

The two old reports become one:
- the baselines' report (`differential_uncertainty/baselines/report.py`);
- the confirmation (`differential_uncertainty/convtu/confirmation.py`).

The new report scores every row on the four image sets: all, untouched, held_out and screen. Then:
- **Intervals:** every non-screen set gets paired bootstrap intervals. ContrastiveConf's λ is cross-fitted again on each set and in every draw.
- **Decisions:** it states both pre-registered decisions.
- **Context:** per-condition mAP, the kNN k table and the timing.

The pure computation goes in `evaluation/report.py`. The stage that loads the run folder goes in `stages/report.py`.

**Why the old numbers come back exactly.** A bootstrap interval of a quantity depends only on that quantity's values over the draws. The draws are `default_rng(44).integers(0, n, n)` for each of the 1,000 samples, as in both old reports, and each quantity is computed as in both old reports. So the new report reproduces both old reports' intervals, even though each draw now evaluates more quantities.

Three old rows are renamed:
- `conditioned` becomes `level`;
- `global_s123` becomes `global_level`;
- `ch_means_knn` and `ch_means_own` become `means_knn` and `means_own`.

The level decision's key `decision` becomes `level_decision`. Task 15 checks the equivalence on the real data.

**Files:**
- Create:
  - `degradation_monitor/evaluation/report.py`;
  - `degradation_monitor/stages/report.py`.
- Modify: `degradation_monitor/stages/__init__.py`.
- Test:
  - `tests/evaluation/test_report.py`;
  - `tests/stages/test_report.py`.

**Interfaces:**
- Consumes:
  - `corruptions.{CONDITIONS, CORRUPTED_CONDITIONS, COMMON_CONDITIONS, EXTRA_CONDITIONS, COMMON_FAMILIES, EXTRA_FAMILIES, FAMILIES, SEVERITIES}` (Tasks 5, 6);
  - `coco_map`, `coco_results`, `per_image_ap`, `FOLDS` (Task 5);
  - `auroc`, `aupr`, `fpr_at_95_tpr`, `condition_aurocs`, `group_separation`, `bootstrap(..., workers)` (Task 6);
  - `contrastive_degradation`, `cross_fit_lambda`, `THETA`, `KNN_K`, `hashemi.K`, `activation_cdf.BINS` (Task 7);
  - the method's score functions and `reference.{NEIGHBOURS, KEY_LAYER, SCORED_LAYERS}` (Task 8);
  - `stack`, `sha1`, `atomic_json`, `RunLayout.report()`, `RunLayout.timing` (Task 9);
  - the registry (Task 10).
- Produces:
  - `degradation_monitor.evaluation.report`:
    - constants `OURS`, `BASELINES`, `ROWS`, `LABELS`, `REFERENCES = ("two_axis", "level")`, `METRICS`, `GROUPS`, `QUANTITIES`, `BOOTSTRAP_SAMPLES = 1000`, `SCREEN_IMAGES = 200`, `UNTOUCHED_START = 1970`, `KNN_KS`;
    - `image_sets(count) -> {name: positions}`;
    - `method_rows(statistics, bank, zstats) -> dict` and `baseline_rows(detector, discopatch=None, activations=None) -> dict`;
    - `contrastive_scores(conf_pos, conf_neg, ap, folds) -> (scores, per_fold)`;
    - `separation_rows(scores, folds, per_fold=())`, `aggregate_rows(rows)`;
    - `headline_numbers(scores, folds, per_fold=(), references=REFERENCES) -> dict`;
    - `level_decision(intervals) -> str`, `headline_decision(intervals) -> str`;
    - `build_tables(scores, detector, ap, folds, condition_map, *, seed, samples=BOOTSTRAP_SAMPLES, workers=1, timing=None) -> (tables, summary)`;
    - `markdown(summary, tables) -> str`, `write_outputs(folder, tables, summary)`.
  - `degradation_monitor.stages.report`:
    - `DETECTOR_ARRAYS`, `ACTIVATION_ARRAYS`;
    - `condition_maps(gt, ids, detector, workers) -> (96,)`;
    - the stage `write_report(settings, manifest)`, registered as `report`.

- [ ] **Step 1: Write the failing tests**

`tests/evaluation/test_report.py`:

```python
import json

import numpy as np
import pytest

from degradation_monitor import corruptions
from degradation_monitor.baselines.contrastive_conf import contrastive_degradation, cross_fit_lambda
from degradation_monitor.datasets.coco import assign_folds
from degradation_monitor.evaluation import metrics, report
from degradation_monitor.method import reference as method_reference
from degradation_monitor.method import scores as method_scores

SEVERITY = np.array([s for _, s in corruptions.CONDITIONS], float)


def _scores(n=30, rows=report.ROWS):
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, (n, 1))
    return {m: base + SEVERITY[None, :] * 2.0 + rng.normal(0, 0.1, (n, 96)) for m in rows}


def _detector(n, seed=1):
    rng = np.random.default_rng(seed)
    knn = np.sort(rng.uniform(0.5, 1.5, (n, 96, 200)), axis=2) + 0.01 * SEVERITY[None, :, None]
    return {"conf_pos": rng.uniform(0.2, 0.9, (n, 96)) - 0.004 * SEVERITY[None],
            "conf_neg": rng.uniform(0.0, 0.1, (n, 96)) + 0.002 * SEVERITY[None],
            "knn": knn.astype(np.float32), "saod_top3": rng.uniform(0, 1, (n, 96)), "saod_min": rng.uniform(0, 1, (n, 96))}


def _tables(scores, n, workers=1, samples=20):
    ap = np.random.default_rng(3).uniform(0, 1, n)
    return report.build_tables(scores, _detector(n), ap, assign_folds(n), 0.5 - 0.05 * SEVERITY, seed=44,
                               samples=samples, workers=workers)


def _interval(low, high):
    return {"low": low, "high": high}


@pytest.fixture
def small_sets(monkeypatch):
    monkeypatch.setattr(report, "SCREEN_IMAGES", 10)
    monkeypatch.setattr(report, "UNTOUCHED_START", 20)


def test_separation_rows_cover_every_row_and_condition_with_the_right_pooling():
    rows = report.separation_rows(_scores(), assign_folds(30), per_fold=("contrastive",))
    assert len(rows) == len(report.ROWS) * 95
    assert {r["pooling"] for r in rows if r["method"] == "contrastive"} == {"fold-averaged"}
    assert {r["pooling"] for r in rows if r["method"] == "knn"} == {"pooled"}
    assert all(r["auroc"] > 0.9 and r["aupr"] > 0.9 for r in rows if r["severity"] >= 3)


def test_aggregate_rows_average_common_and_extra_families_separately():
    rows = report.separation_rows(_scores(), assign_folds(30))
    keys = {(r["method"], r["group"], r["severity"]) for r in report.aggregate_rows(rows)}
    assert {("knn", "common", "all"), ("knn", "extra", 5), ("knn", "all", "all")} <= keys


def test_headline_numbers_give_six_quantities_per_row_and_each_reference_minus_the_others():
    scores = _scores(rows=("two_axis", "level", "cdf"))
    numbers = report.headline_numbers(scores, assign_folds(30))
    for quantity in report.QUANTITIES:
        expected = numbers[f"two_axis:{quantity}"] - numbers[f"cdf:{quantity}"]
        assert numbers[f"two_axis - cdf:{quantity}"] == pytest.approx(expected)
    assert "level - cdf:auroc_common" in numbers and "two_axis - level:auroc_common" in numbers
    assert "level - two_axis:auroc_common" not in numbers and "cdf - level:auroc_common" not in numbers
    clean, degraded = scores["cdf"][:, 0], scores["cdf"][:, corruptions.COMMON_CONDITIONS].T
    assert numbers["cdf:fpr95_common"] == pytest.approx(np.mean([metrics.fpr_at_95_tpr(clean, r) for r in degraded]))


def test_headline_numbers_fold_average_the_rows_asked_for():
    scores, folds = _scores(), assign_folds(30)
    numbers = report.headline_numbers(scores, folds, per_fold=("contrastive",))
    values = scores["contrastive"]
    expected = np.mean([metrics.condition_aurocs(values[folds == f, 0],
                                                 values[folds == f][:, corruptions.COMMON_CONDITIONS].T).mean()
                        for f in range(5)])
    assert numbers["contrastive:auroc_common"] == pytest.approx(expected)


def test_the_image_sets_follow_the_evaluation_order():
    sets = report.image_sets(5000)
    assert {name: len(rows) for name, rows in sets.items()} == {"all": 5000, "untouched": 3030, "held_out": 4800,
                                                                 "screen": 200}
    assert sets["untouched"][0] == 1970 and sets["held_out"][0] == 200 and sets["screen"][-1] == 199
    assert set(report.image_sets(150)) == {"all", "screen"}  # a limited run has no images after the screen


def test_contrastive_scores_fit_lambda_on_the_given_images_only():
    detector, folds = _detector(30), assign_folds(30)
    ap = np.random.default_rng(3).uniform(0, 1, 30)
    scores, per_fold = report.contrastive_scores(detector["conf_pos"], detector["conf_neg"], ap, folds)
    lam, expected = cross_fit_lambda(detector["conf_pos"][:, 0], detector["conf_neg"][:, 0], ap, folds)
    assert per_fold == expected
    np.testing.assert_array_equal(scores, contrastive_degradation(detector["conf_pos"], detector["conf_neg"],
                                                                  lam[:, None]))


def test_baseline_rows_add_discopatch_and_the_activation_monitors_only_when_given():
    detector = _detector(2)
    activations = {"hashemi_decoder": np.full((2, 96), 0.1), "hashemi_encoder": np.full((2, 96), 0.2),
                   "cdf_backbone": np.full((2, 96), 3.0), "cdf_backbone_z": np.full((2, 96), -1.5)}
    rows = report.baseline_rows(detector, discopatch=np.ones((2, 96)), activations=activations)
    assert (rows["hashemi"][0, 0], rows["hashemi_enc"][0, 0], rows["cdf"][0, 0], rows["cdf_sum"][0, 0]) == \
        (0.1, 0.2, -1.5, 3.0)
    assert np.array_equal(rows["knn"], detector["knn"][:, :, 99]) and rows["discopatch"][0, 0] == 1.0
    assert set(report.baseline_rows(detector)) == {"saod_top3", "saod_min", "knn"}


def test_method_rows_score_our_six_rows_from_the_statistics(monkeypatch):
    monkeypatch.setattr(method_reference, "NEIGHBOURS", 3)
    rng = np.random.default_rng(5)
    widths = {1: 2, 2: 3, 3: 4, 4: 5}

    def statistics(*shape):
        return {f"{s}_s{l}": rng.uniform(0.1, 1.0, (*shape, c)) for s in ("means", "top") for l, c in widths.items()}

    test, bank, zstats = statistics(4, 96), statistics(8), statistics(5)
    rows = report.method_rows(test, bank, zstats)
    assert list(rows) == list(report.OURS) and all(v.shape == (4, 96) for v in rows.values())
    np.testing.assert_array_equal(rows["two_axis"], method_scores.two_axis_scores(test, bank, zstats, k=3)[0])


def test_the_level_decision_follows_the_preregistered_rule():
    ahead = {f"level - {other}:auroc_{group}": _interval(0.01, 0.03)
             for other in ("cdf", "global_level") for group in report.GROUPS}
    assert report.level_decision(ahead) == "confirmed"
    behind_cdf = dict(ahead, **{"level - cdf:auroc_common": _interval(-0.004, 0.02)})
    assert report.level_decision(behind_cdf) == "conditioning confirmed, not ahead of the activation CDFs"
    behind = dict(ahead, **{"level - global_level:auroc_extra": _interval(-0.01, 0.002)})
    assert report.level_decision(behind) == "not confirmed"
    assert report.level_decision({k: v for k, v in ahead.items() if "cdf" not in k}) == \
        "unavailable: the activation-CDF scores are missing"
    assert report.level_decision({}) == "unavailable: our method's scores are missing"


def _headline(low_cdf_common=0.05, low_cdf_extra=0.02, low_level=0.01):
    return {"two_axis - cdf:auroc_common": _interval(low_cdf_common, 0.1),
            "two_axis - cdf:auroc_extra": _interval(low_cdf_extra, 0.1),
            "two_axis - level:auroc_common": _interval(low_level, 0.1)}


def test_the_headline_decision_needs_both_image_sets():
    good = _headline()
    assert report.headline_decision({"all": good, "untouched": good}) == "confirmed"
    assert report.headline_decision({"all": good, "untouched": _headline(low_cdf_extra=-0.01)}) == \
        "ahead of the activation CDFs on all images, but not on the untouched images"
    level = _headline(low_level=-0.002)
    assert report.headline_decision({"all": level, "untouched": level}) == \
        "ahead of the activation CDFs, but the flattening adds nothing over the level score on the common families"
    behind = _headline(low_cdf_common=-0.01)
    assert report.headline_decision({"all": behind, "untouched": good}) == "not ahead of the activation CDFs"
    assert report.headline_decision({"all": {}, "untouched": {}}) == \
        "unavailable: the two-axis or the activation-CDF scores are missing"


def test_the_tables_give_every_present_row_on_every_image_set_and_intervals_after_the_screen(small_sets):
    rows = ("two_axis", "level", "global_level", "saod_top3", "knn", "cdf")
    tables, summary = _tables(_scores(40, rows), 40)
    assert summary["image_sets"] == {"all": 40, "untouched": 20, "held_out": 30, "screen": 10}
    assert set(summary["intervals"]) == {"all", "untouched", "held_out"}
    assert set(summary["headline"]["screen"]) == {*rows, "contrastive"}
    assert {"two_axis - cdf:fpr95_extra", "level - global_level:auroc_common",
            "contrastive:aupr_common"} <= set(summary["intervals"]["all"])
    assert summary["headline_decision"] == report.headline_decision(summary["intervals"])
    assert summary["level_decision"] == report.level_decision(summary["intervals"]["held_out"])
    assert set(summary["lambda_per_fold"]) == {"all", "untouched", "held_out", "screen"}
    assert {r["subset"] for r in tables["separation"]} == {"all", "untouched", "held_out", "screen"}
    assert len(tables["conditions"]) == 96 and tables["conditions"][0]["map"] == pytest.approx(0.5)
    assert [r["k"] for r in tables["knn_k"]] == list(report.KNN_KS)


def test_worker_processes_give_the_same_report(small_sets):
    scores = _scores(40, ("two_axis", "level", "cdf"))
    assert _tables(scores, 40, workers=2)[1] == _tables(scores, 40, workers=1)[1]


def test_report_without_the_activation_monitors_says_the_headline_is_unavailable(small_sets, tmp_path):
    rows = ("two_axis", "peak_share", "level", "global_level", "saod_top3", "saod_min", "knn")
    scores = _scores(40, rows)
    scores["global_level"] = np.random.default_rng(9).normal(size=(40, 96))  # no signal, so level is clearly ahead
    tables, summary = _tables(scores, 40)
    assert summary["headline_decision"] == "unavailable: the two-axis or the activation-CDF scores are missing"
    assert summary["level_decision"] == "unavailable: the activation-CDF scores are missing"
    assert {r["method"] for r in tables["separation"]} == {*rows, "contrastive"}
    report.write_outputs(tmp_path, tables, summary)
    text = (tmp_path / "report.md").read_text()
    assert "unavailable" in text and report.LABELS["two_axis"] in text and report.LABELS["cdf"] not in text


def test_write_outputs_creates_csv_json_and_markdown(small_sets, tmp_path):
    tables, summary = _tables(_scores(40, ("two_axis", "level", "global_level", "cdf", "discopatch")), 40)
    report.write_outputs(tmp_path, tables, summary)
    for name in ("separation", "aggregates", "intervals", "conditions", "knn_k"):
        assert (tmp_path / f"{name}.csv").exists(), name
    assert not (tmp_path / "timing.csv").exists()  # no timing was measured
    assert json.loads((tmp_path / "summary.json").read_text())["images"] == 40
    text = (tmp_path / "report.md").read_text()
    assert text.startswith("# Corruption detection on COCO") and "DisCoPatch" in text and "AUPR common" in text
```

`tests/stages/test_report.py`, ported from the old `_tiny_run` fixture:

```python
import json

import numpy as np
import pytest
from PIL import Image

from degradation_monitor.evaluation import report as tables
from degradation_monitor.method import reference as method_reference
from degradation_monitor.runs import atomic_npz
from degradation_monitor.settings import Settings
from degradation_monitor.stages import baselines as baseline_stages
from degradation_monitor.stages import run_stage

IMAGES = 20
WIDTHS = (2, 3, 4, 5)


class QualityTap:
    """Detector stand-in: one box whose confidence falls as the image gets noisier."""

    def __init__(self, *_a, **_k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def run(self, arrays, batch_size=32):
        n = len(arrays)
        quality = np.array([1.0 - a.std() / 128.0 for a in arrays])
        logits = np.full((n, 300, 80), -6.0)
        logits[:, 0, 0] = 6.0 * quality - 2.0
        boxes = np.tile([0.3125, 0.3125, 0.375, 0.375], (n, 300, 1))
        pooled = np.random.default_rng(0).normal(size=(n, 512)) + quality[:, None]
        return logits, boxes, pooled


@pytest.fixture
def run(tmp_path, monkeypatch):
    """IMAGES fixture images with one annotated object each, after the detector pass."""
    val = tmp_path / "val"
    val.mkdir()
    rng = np.random.default_rng(7)
    images, annotations = [], []
    for index in range(IMAGES):
        name = f"{index:012d}.jpg"
        Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)).save(val / name)
        images.append({"id": index + 1, "file_name": name, "width": 64, "height": 48})
        annotations.append({"id": 100 + index, "image_id": index + 1, "category_id": 1,
                            "bbox": [8, 6, 24, 18], "area": 432, "iscrowd": 0})
    (tmp_path / "ann.json").write_text(json.dumps({
        "images": images, "annotations": annotations,
        "categories": [{"id": c, "name": f"c{c}"} for c in range(1, 81)]}))
    (tmp_path / "ckpt.pth").write_bytes(b"weights")
    monkeypatch.setattr(baseline_stages, "DetectorTap", QualityTap)
    monkeypatch.setattr(tables, "BOOTSTRAP_SAMPLES", 5)
    monkeypatch.setattr(tables, "SCREEN_IMAGES", 5)
    monkeypatch.setattr(tables, "UNTOUCHED_START", 8)
    settings = Settings(run=tmp_path / "run", checkpoint=tmp_path / "ckpt.pth", train_images=tmp_path,
                        val_images=val, annotations=tmp_path / "ann.json", discopatch_root=tmp_path,
                        limit=IMAGES, workers=0, device="cpu")
    settings.layout.knn_bank.parent.mkdir(parents=True)
    np.save(settings.layout.knn_bank, rng.normal(size=(256, 512)).astype(np.float16))
    run_stage("detector-pass", settings)
    return settings


def _statistics(rng, *shape):
    return {f"{s}_s{l}": rng.uniform(0.1, 1.0, (*shape, c)).astype(np.float32)
            for s in ("means", "top") for l, c in zip(range(1, 5), WIDTHS)}


def test_the_report_stage_writes_every_table_for_a_run_with_only_the_detector_pass(run):
    run_stage("report", run)
    folder = run.layout.report()
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["images"] == IMAGES
    assert summary["image_sets"] == {"all": 20, "untouched": 12, "held_out": 15, "screen": 5}
    assert len(summary["lambda_per_fold"]["all"]) == 5
    assert summary["rows"] == ["saod_top3", "saod_min", "contrastive", "knn"]
    assert summary["headline_decision"] == "unavailable: the two-axis or the activation-CDF scores are missing"
    assert summary["level_decision"] == "unavailable: our method's scores are missing"
    for name in ("separation", "aggregates", "intervals", "conditions", "knn_k"):
        assert (folder / f"{name}.csv").exists(), name
    header = (folder / "conditions.csv").read_text().splitlines()[0].split(",")
    assert "map" in header and "mean_contrastive" in header
    assert "ContrastiveConf" in (folder / "report.md").read_text()


def test_the_report_stage_adds_the_activation_monitors_and_our_method_when_they_were_run(run, monkeypatch):
    monkeypatch.setattr(method_reference, "NEIGHBOURS", 3)
    rng = np.random.default_rng(8)
    for path in sorted(run.layout.scores("detector").glob("*.npz")):
        atomic_npz(run.layout.score_file("activations", path.name), hashemi_decoder=rng.uniform(0, 1, 96),
                   hashemi_encoder=rng.uniform(0, 1, 96), hashemi_encoder_maps=rng.uniform(0, 1, (96, 3)),
                   cdf_backbone=rng.uniform(0, 50, 96), cdf_backbone_z=rng.normal(0, 3, 96),
                   cdf_stages=rng.uniform(0, 10, (96, 5)))
        atomic_npz(run.layout.score_file("method", path.name), **_statistics(rng, 96))
    atomic_npz(run.layout.method_bank, **_statistics(rng, 6))
    atomic_npz(run.layout.method_zstats, **_statistics(rng, 4))
    run_stage("report", run)
    summary = json.loads((run.layout.report() / "summary.json").read_text())
    assert {"two_axis", "level", "means_own", "hashemi", "hashemi_enc", "cdf", "cdf_sum"} <= set(summary["rows"])
    assert set(summary["inputs"]["method"]) == {"method_bank", "method_zstats"}
    assert summary["headline_decision"] == tables.headline_decision(summary["intervals"])
    assert summary["level_decision"] == tables.level_decision(summary["intervals"]["held_out"])
    text = (run.layout.report() / "report.md").read_text()
    assert tables.LABELS["hashemi"] in text and tables.LABELS["two_axis"] in text


def test_the_report_stage_needs_the_method_reference_when_the_method_pass_ran(run):
    rng = np.random.default_rng(8)
    for path in sorted(run.layout.scores("detector").glob("*.npz")):
        atomic_npz(run.layout.score_file("method", path.name), **_statistics(rng, 96))
    with pytest.raises(ValueError, match="run the method-reference stage first"):
        run_stage("report", run)
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/evaluation/test_report.py tests/stages/test_report.py`
Expected: two ERRORs at collection, `ImportError: cannot import name 'report' from 'degradation_monitor.evaluation'` and `...from 'degradation_monitor.stages'`. The stage test imports `report` through `run_stage("report", ...)`, which fails with `unknown stage 'report'` once the evaluation module exists.

- [ ] **Step 3: Implement `degradation_monitor/evaluation/report.py`**

`separation_rows` and `aggregate_rows` are the old baselines report's, verbatim except for the names of their inputs. `_headline`, `_by_severity` and `_by_family` are the old confirmation's. `headline_decision` and `level_decision` (the old `decision`) are the old confirmation's, with the renamed rows:

```python
"""The report: separation of our method and the six baselines on four image sets.

Every score is oriented so that higher means more likely corrupted. For each image set the report gives:
- every row's AUROC, AUPR and FPR95 per condition;
- their means over the 15 common and the 4 extra families;
- 95% paired bootstrap intervals over images, ContrastiveConf's lambda being cross-fitted again in every draw.

It also gives the two pre-registered decisions, each condition's mAP and the kNN baseline's k.

The image sets, by position in the seed-44 evaluation order:
- all: every evaluation image (5,000), the headline, on the same images as every baseline;
- untouched: positions UNTOUCHED_START and later (3,030), which nobody read while the method was designed;
- held_out: positions SCREEN_IMAGES and later (4,800), the pre-registered check of the level score;
- screen: the first SCREEN_IMAGES (200), on which the method was screened. It gets no intervals.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from .. import corruptions
from ..baselines.activation_cdf import BINS as CDF_BINS
from ..baselines.contrastive_conf import THETA, contrastive_degradation, cross_fit_lambda
from ..baselines.hashemi import K as HASHEMI_K
from ..baselines.knn import KNN_K
from ..datasets.coco import FOLDS
from ..method import reference as method_reference
from ..method import scores as method_scores
from ..runs import atomic_json
from .metrics import aupr, auroc, bootstrap, condition_aurocs, fpr_at_95_tpr, group_separation

OURS = ("two_axis", "peak_share", "level", "global_level", "means_knn", "means_own")
BASELINES = ("saod_top3", "saod_min", "contrastive", "knn", "discopatch", "hashemi", "hashemi_enc", "cdf", "cdf_sum")
ROWS = OURS + BASELINES
LABELS = {
    "two_axis": "Two-axis: flatter or shifted vs the 50 most similar clean scenes (headline)",
    "peak_share": "Peak share vs the 50 most similar clean scenes",
    "level": "Level vs the 50 most similar clean scenes (stage-4 key)",
    "global_level": "Level vs the average of all clean images (stages 1–3)",
    "means_knn": "Channel means, kNN, 4 stages",
    "means_own": "Channel means vs own average, 4 stages",
    "saod_top3": "SAOD, mean of top 3",
    "saod_min": "SAOD, min (1 − max confidence)",
    "contrastive": "ContrastiveConf",
    "knn": "kNN (k = 100)",
    "discopatch": "DisCoPatch",
    "hashemi": "Hashemi et al., decoder queries",
    "hashemi_enc": "Hashemi et al., encoder maps (sensitivity)",
    "cdf": "Activation CDFs (Becker et al., ICPR 2026)",
    "cdf_sum": "Activation CDFs, plain channel sum (sensitivity)",
}
REFERENCES = ("two_axis", "level")  # rows whose differences with every other row get intervals
METRICS = ("auroc", "aupr", "fpr95")
GROUPS = ("common", "extra")
QUANTITIES = tuple(f"{metric}_{group}" for metric in METRICS for group in GROUPS)
BOOTSTRAP_SAMPLES = 1000
SCREEN_IMAGES = 200      # the first images of the evaluation order; the method was screened on them
UNTOUCHED_START = 1970   # the roundtable read positions 200-1969 of the running pass; nobody read these
KNN_KS = (1, 10, 50, 100, 200)
SECTION_REFERENCE = {"all": "two_axis", "untouched": "two_axis", "held_out": "level"}  # differences shown per set
FAMILY_ROWS = {"two_axis": "Two-axis", "peak_share": "Peak share", "level": "Level (similar scenes)",
               "cdf": "Activation CDFs", "discopatch": "DisCoPatch"}  # short column names for the family table


def image_sets(count: int) -> dict:
    """Positions of each image set in the evaluation order; a set without images (a limited run) is left out."""
    screen = min(SCREEN_IMAGES, count)
    sets = {"all": np.arange(count), "untouched": np.arange(min(UNTOUCHED_START, count), count),
            "held_out": np.arange(screen, count), "screen": np.arange(screen)}
    return {name: rows for name, rows in sets.items() if len(rows)}


def method_rows(statistics: dict, bank: dict, zstats: dict) -> dict:
    """Our six rows (images, 96) from the stored channel statistics and the clean reference's."""
    k = method_reference.NEIGHBOURS
    return {"two_axis": method_scores.two_axis_scores(statistics, bank, zstats, k=k)[0],
            "peak_share": method_scores.peak_share_scores(statistics, bank, zstats, k=k)[0],
            "level": method_scores.level_scores(statistics, bank, zstats, k=k)[0],
            "global_level": method_scores.global_level_scores(statistics, bank, zstats)[0],
            "means_knn": method_scores.means_knn_scores(statistics, bank, zstats)[0],
            "means_own": method_scores.means_own_scores(statistics, bank, zstats)[0]}


def baseline_rows(detector: dict, discopatch=None, activations=None) -> dict:
    """The baselines' rows (images, 96), except ContrastiveConf, whose lambda is fitted on each image set."""
    rows = {"saod_top3": detector["saod_top3"], "saod_min": detector["saod_min"],
            "knn": detector["knn"][:, :, KNN_K - 1]}
    if discopatch is not None:
        rows["discopatch"] = discopatch
    if activations is not None:
        rows.update(hashemi=activations["hashemi_decoder"], hashemi_enc=activations["hashemi_encoder"],
                    cdf=activations["cdf_backbone_z"], cdf_sum=activations["cdf_backbone"])
    return rows


def contrastive_scores(conf_pos, conf_neg, ap, folds) -> tuple[np.ndarray, dict]:
    """ContrastiveConf on the given images; each fold's lambda is fitted on the other folds' clean images (UQ-DETR)."""
    lam, per_fold = cross_fit_lambda(conf_pos[:, 0], conf_neg[:, 0], ap, folds)
    return contrastive_degradation(conf_pos, conf_neg, lam[:, None]), per_fold


def _separation(clean, degraded) -> dict:
    return {"auroc": auroc(clean, degraded), "aupr": aupr(clean, degraded), "fpr95": fpr_at_95_tpr(clean, degraded)}


def separation_rows(scores: dict, folds, per_fold=()) -> list[dict]:
    """Each row's separation of every corrupted condition from the clean images; rows in `per_fold` fold-averaged."""
    folds = np.asarray(folds)
    out = []
    for method, values in scores.items():
        by_fold = method in per_fold
        for c in corruptions.CORRUPTED_CONDITIONS:
            family, severity = corruptions.CONDITIONS[c]
            if by_fold:
                parts = [_separation(values[folds == f, 0], values[folds == f, c]) for f in np.unique(folds)]
                result = {key: float(np.mean([p[key] for p in parts])) for key in METRICS}
            else:
                result = _separation(values[:, 0], values[:, c])
            out.append({"method": method, "family": family, "severity": int(severity),
                        "group": "common" if family in corruptions.COMMON_FAMILIES else "extra",
                        "pooling": "fold-averaged" if by_fold else "pooled", **result})
    return out


def aggregate_rows(rows: list[dict]) -> list[dict]:
    out = []
    for method in dict.fromkeys(r["method"] for r in rows):
        for group in ("common", "extra", "all"):
            for severity in (*corruptions.SEVERITIES, "all"):
                chosen = [r for r in rows if r["method"] == method
                          and (group == "all" or r["group"] == group)
                          and (severity == "all" or r["severity"] == severity)]
                out.append({"method": method, "group": group, "severity": severity,
                            **{key: float(np.mean([r[key] for r in chosen])) for key in METRICS}})
    return out


def headline_numbers(scores: dict, folds, per_fold=(), references=REFERENCES) -> dict:
    """Each row's six separation quantities, and each reference row minus every other row.

    Rows in `per_fold` are averaged over folds, as in separation_rows. A pair of two references appears once, as the
    earlier reference minus the later one.
    """
    folds = np.asarray(folds)
    out = {}
    for method, values in scores.items():
        for group, columns in (("common", corruptions.COMMON_CONDITIONS), ("extra", corruptions.EXTRA_CONDITIONS)):
            if method in per_fold:
                numbers = np.mean([group_separation(values[folds == f, 0], values[folds == f][:, columns].T)
                                   for f in np.unique(folds)], axis=0)
            else:
                numbers = group_separation(values[:, 0], values[:, columns].T)
            for metric, value in zip(METRICS, numbers):
                out[f"{method}:{metric}_{group}"] = float(value)
    present = [reference for reference in references if reference in scores]
    for index, reference in enumerate(present):
        for other in scores:
            if other != reference and other not in present[:index]:
                for quantity in QUANTITIES:
                    out[f"{reference} - {other}:{quantity}"] = out[f"{reference}:{quantity}"] - out[f"{other}:{quantity}"]
    return out


def level_decision(intervals: dict) -> str:
    """The rule pre-registered for the level score, applied to the held-out images' intervals."""
    needed = {other: [f"level - {other}:auroc_{group}" for group in GROUPS] for other in ("global_level", "cdf")}
    if not all(key in intervals for key in needed["global_level"]):
        return "unavailable: our method's scores are missing"
    if not all(intervals[key]["low"] > 0 for key in needed["global_level"]):
        return "not confirmed"
    if not all(key in intervals for key in needed["cdf"]):
        return "unavailable: the activation-CDF scores are missing"
    if all(intervals[key]["low"] > 0 for key in needed["cdf"]):
        return "confirmed"
    return "conditioning confirmed, not ahead of the activation CDFs"


def headline_decision(intervals: dict) -> str:
    """The headline rule, fixed before the untouched images were read.

    Confirmed when, on all images and on the untouched ones, the two-axis score beats the activation CDFs on the
    common and the extra families, and beats the level score on the common families, every interval excluding 0.
    """
    sets = ("all", "untouched")
    needed = [f"two_axis - cdf:auroc_{group}" for group in GROUPS] + ["two_axis - level:auroc_common"]
    if not all(key in intervals.get(name, {}) for name in sets for key in needed):
        return "unavailable: the two-axis or the activation-CDF scores are missing"
    ahead = {name: all(intervals[name][f"two_axis - cdf:auroc_{group}"]["low"] > 0 for group in GROUPS)
             for name in sets}
    if not ahead["all"]:
        return "not ahead of the activation CDFs"
    if not ahead["untouched"]:
        return "ahead of the activation CDFs on all images, but not on the untouched images"
    if not all(intervals[name]["two_axis - level:auroc_common"]["low"] > 0 for name in sets):
        return "ahead of the activation CDFs, but the flattening adds nothing over the level score on the common families"
    return "confirmed"


def _headline(aggregates: list) -> dict:
    out = {}
    for row in aggregates:
        if row["severity"] == "all" and row["group"] in GROUPS:
            for metric in METRICS:
                out.setdefault(row["method"], {})[f"{metric}_{row['group']}"] = row[metric]
    return out


def _by_severity(aggregates: list) -> dict:
    out = {}
    for row in aggregates:
        if row["severity"] != "all" and row["group"] in GROUPS:
            out.setdefault(row["method"], {}).setdefault(row["group"], []).append(row["auroc"])
    return out


def _by_family(separation: list) -> dict:
    out = {}
    for row in separation:
        out.setdefault(row["method"], {}).setdefault(row["family"], {})[str(row["severity"])] = row["auroc"]
    return out


def build_tables(scores: dict, detector: dict, ap, folds, condition_map, *, seed: int,
                 samples: int = BOOTSTRAP_SAMPLES, workers: int = 1, timing=None) -> tuple[dict, dict]:
    """Every table of the report, and its summary.

    `scores` holds every present row except ContrastiveConf as (images, 96) arrays in evaluation order.
    ContrastiveConf is built for each image set, and in every draw, from the detector's Conf+ and Conf- and the
    clean images' AP.
    """
    folds, ap = np.asarray(folds), np.asarray(ap)

    def rows_of(chosen):
        """Every present row on the chosen images (repeats allowed), and ContrastiveConf's lambda per fold."""
        contrastive, lam = contrastive_scores(detector["conf_pos"][chosen], detector["conf_neg"][chosen], ap[chosen],
                                              folds[chosen])
        drawn = {**{m: v[chosen] for m, v in scores.items()}, "contrastive": contrastive}
        return {m: drawn[m] for m in ROWS if m in drawn}, lam

    sets = image_sets(len(folds))
    tables = {"separation": [], "aggregates": [], "intervals": []}
    headline, by_severity, by_family, intervals, lambdas = {}, {}, {}, {}, {}
    for name, rows in sets.items():
        point_scores, lambdas[name] = rows_of(rows)
        if name == "all":
            all_scores = point_scores
        per_fold = () if len(set(lambdas[name].values())) == 1 else ("contrastive",)
        separation = separation_rows(point_scores, folds[rows], per_fold)
        aggregates = aggregate_rows(separation)
        tables["separation"] += [{"subset": name, **row} for row in separation]
        tables["aggregates"] += [{"subset": name, **row} for row in aggregates]
        headline[name], by_severity[name] = _headline(aggregates), _by_severity(aggregates)
        by_family[name] = _by_family(separation)
        if name == "screen":
            continue

        def statistic(draw):
            drawn, _ = rows_of(rows[draw])
            return headline_numbers(drawn, folds[rows[draw]], per_fold)

        point = headline_numbers(point_scores, folds[rows], per_fold)
        ranges = bootstrap(statistic, len(rows), samples=samples, seed=seed, workers=workers)
        intervals[name] = {key: {"point": point[key], "low": low, "high": high} for key, (low, high) in ranges.items()}
        tables["intervals"] += [{"subset": name, "quantity": key, **value} for key, value in intervals[name].items()]
    knn = detector["knn"]
    tables["conditions"] = [{"family": f, "severity": s, "map": float(condition_map[c]),
                             **{f"mean_{m}": float(v[:, c].mean()) for m, v in all_scores.items()}}
                            for c, (f, s) in enumerate(corruptions.CONDITIONS)]
    tables["knn_k"] = [{"k": k, "mean_auroc_common": float(condition_aurocs(
        knn[:, 0, k - 1], knn[:, corruptions.COMMON_CONDITIONS, k - 1].T).mean())} for k in KNN_KS]
    tables["timing"] = [{"part": key, "ms": value} for key, value in (timing or {}).items() if key.endswith("_ms")]
    summary = {
        "images": len(folds), "image_sets": {name: len(rows) for name, rows in sets.items()},
        "screen_images": SCREEN_IMAGES, "untouched_start": UNTOUCHED_START, "folds": FOLDS,
        "lambda_per_fold": lambdas, "lambda_folds_agree": {n: len(set(v.values())) == 1 for n, v in lambdas.items()},
        "images_with_ap": int(np.isfinite(ap).sum()), "clean_map": float(condition_map[0]),
        "rows": list(all_scores), "knn_k": KNN_K, "theta": THETA, "hashemi_k": HASHEMI_K, "cdf_bins": CDF_BINS,
        "neighbours": method_reference.NEIGHBOURS, "key": method_reference.KEY_LAYER,
        "scored": list(method_reference.SCORED_LAYERS), "bootstrap_samples": samples, "seed": seed,
        "headline_row": "two_axis", "headline_decision": headline_decision(intervals),
        "level_decision": level_decision(intervals.get("held_out", {})),
        "headline": headline, "by_severity": by_severity, "by_family": by_family, "intervals": intervals,
        "labels": {m: LABELS[m] for m in all_scores},
    }
    return tables, summary


def _number(value) -> str:
    return "–" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.3f}"


def _signed(value) -> str:
    return f"{value:+.3f}".replace("-", "−")


def _cell(point, interval=None, show=_number) -> str:
    return show(point) + (f" [{show(interval['low'])}, {show(interval['high'])}]" if interval else "")


def _set_title(summary: dict, name: str) -> str:
    n = summary["image_sets"][name]
    return {"all": f"All {n} images (the headline, on the same images as every baseline)",
            "untouched": f"Untouched images ({n}, positions {summary['untouched_start']} and later, read by nobody "
                         "while the method was designed)",
            "held_out": f"Held-out images ({n}, positions {summary['screen_images']} and later: the pre-registered "
                        "check of the level score)",
            "screen": f"The {n} screening images"}[name]


def _set_section(summary: dict, name: str) -> list:
    head, intervals = summary["headline"][name], summary["intervals"].get(name, {})
    lines = [f"## {_set_title(summary, name)}", "",
             "| Row | AUROC common ↑ | AUROC extra ↑ | AUPR common ↑ | AUPR extra ↑ | FPR95 common ↓ | FPR95 extra ↓ |",
             "|---|---|---|---|---|---|---|"]
    for row in (r for r in ROWS if r in head):
        cells = [_cell(head[row][q], intervals.get(f"{row}:{q}")) for q in QUANTITIES]
        lines.append(f"| {LABELS[row]} | " + " | ".join(cells) + " |")
    reference = SECTION_REFERENCE.get(name)
    others = [r for r in ROWS if f"{reference} - {r}:auroc_common" in intervals]
    if others:
        lines += ["", f"{LABELS[reference]} minus each other row (Δ AUROC > 0 or Δ FPR95 < 0: it is better):", "",
                  "| Other row | Δ AUROC common | Δ AUROC extra | Δ FPR95 common | Δ FPR95 extra |",
                  "|---|---|---|---|---|"]
        for other in others:
            cells = [_cell(intervals[f"{reference} - {other}:{q}"]["point"], intervals[f"{reference} - {other}:{q}"],
                           _signed) for q in ("auroc_common", "auroc_extra", "fpr95_common", "fpr95_extra")]
            lines.append(f"| {LABELS[other]} | " + " | ".join(cells) + " |")
    return lines + [""]


def markdown(summary: dict, tables: dict) -> str:
    lines = ["# Corruption detection on COCO: our method and six baselines", "",
             f"**Headline (two-axis score, all images and the untouched ones):** {summary['headline_decision']}.", "",
             f"**Pre-registered level score (held-out images):** {summary['level_decision']}.", "",
             "Every score is oriented so that higher means more likely corrupted. ↑ higher is better, ↓ lower is "
             "better; an AUROC of 0.5 is chance. Brackets are 95% paired bootstrap intervals over images "
             f"({summary['bootstrap_samples']} draws, seed {summary['seed']}).", ""]
    for name in summary["headline"]:
        lines += _set_section(summary, name)
    severity = summary["by_severity"]["all"]
    lines += ["## AUROC by severity (all images)", "",
              "| Row | Common, severities 1–5 ↑ | Extra, severities 1–5 ↑ |", "|---|---|---|"]
    for row in (r for r in ROWS if r in severity):
        lines.append(f"| {LABELS[row]} | " + " / ".join(f"{v:.3f}" for v in severity[row]["common"]) + " | "
                     + " / ".join(f"{v:.3f}" for v in severity[row]["extra"]) + " |")
    family = summary["by_family"]["all"]
    shown = [r for r in FAMILY_ROWS if r in family]
    if shown:
        lines += ["", "## AUROC by family at severities 1 / 3 / 5 (all images)", "",
                  "| Family | " + " | ".join(FAMILY_ROWS[r] for r in shown) + " |", "|---|" + "---|" * len(shown)]
        for name in corruptions.FAMILIES:
            marker = " *" if name in corruptions.EXTRA_FAMILIES else ""
            cells = [" / ".join(f"{family[r][name][s]:.2f}" for s in ("1", "3", "5")) for r in shown]
            lines.append(f"| {name.replace('_', ' ')}{marker} | " + " | ".join(cells) + " |")
        lines += ["", "`*` marks the extra families."]
    conditions = tables.get("conditions", [])
    if conditions:
        lines += ["", "## Detector mAP (all images)", "", f"Clean: {conditions[0]['map']:.3f}.", "",
                  "| Severity | Common families | Extra families |", "|---|---|---|"]
        for level in corruptions.SEVERITIES:
            means = [np.mean([r["map"] for r in conditions if r["severity"] == level
                              and (r["family"] in corruptions.COMMON_FAMILIES) == (group == "common")])
                     for group in GROUPS]
            lines.append(f"| {level} | {means[0]:.3f} | {means[1]:.3f} |")
    if tables.get("knn_k"):
        lines += ["", "## kNN baseline by k (all images)", "", "| k | AUROC common |", "|---|---|"]
        lines += [f"| {r['k']} | {r['mean_auroc_common']:.3f} |" for r in tables["knn_k"]]
    if tables.get("timing"):
        lines += ["", "## Runtime (median ms per image, batch 1)", "", "| Part | ms |", "|---|---|"]
        lines += [f"| {r['part']} | {_number(r['ms'])} |" for r in tables["timing"]]
    scalars = {k: v for k, v in summary.items()
               if not isinstance(v, dict) and k not in ("headline_decision", "level_decision")}
    lines += ["", "## Fixed choices", ""] + [f"- {key}: {value}" for key, value in sorted(scalars.items())]
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(folder, tables: dict, summary: dict) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        if rows:
            _write_csv(folder / f"{name}.csv", rows)
    atomic_json(folder / "summary.json", summary)
    (folder / "report.md").write_text(markdown(summary, tables))
```

- [ ] **Step 4: Implement `degradation_monitor/stages/report.py`**

The mAP computation is the old report's, run in forked worker processes. It runs before any torch work, so the workers fork from a process that has not started torch's threads yet. The bootstrap's workers fork later, but they never call torch.

```python
"""The report stage: read every stored score and write the report to reports/coco/."""
from __future__ import annotations

import json
import multiprocessing

import numpy as np

from .. import corruptions
from ..datasets.coco import coco_map, coco_results, per_image_ap
from ..evaluation import report as tables
from ..method.statistics import KEYS
from ..runs import sha1, stack

DETECTOR_ARRAYS = ("saod_min", "saod_top3", "conf_pos", "conf_neg", "knn", "det_scores", "det_labels", "det_boxes")
ACTIVATION_ARRAYS = ("hashemi_decoder", "hashemi_encoder", "cdf_backbone_z", "cdf_backbone")
_MAP_INPUTS = None  # what the forked mAP workers read


def _results(gt, ids, detector, condition) -> list:
    """COCO result records of every image under one condition."""
    return [r for k, i in enumerate(ids)
            for r in coco_results(i, detector["det_scores"][k, condition], detector["det_labels"][k, condition],
                                  detector["det_boxes"][k, condition], gt.category_ids)]


def _condition_map(condition: int) -> float:
    gt, ids, detector = _MAP_INPUTS
    return coco_map(gt, _results(gt, ids, detector, condition), ids)


def condition_maps(gt, ids, detector, workers: int) -> np.ndarray:
    """Each condition's mAP over the evaluation images, in forked worker processes when workers > 1."""
    global _MAP_INPUTS
    _MAP_INPUTS = (gt, ids, detector)
    try:
        conditions = range(len(corruptions.CONDITIONS))
        if workers > 1:
            with multiprocessing.get_context("fork").Pool(workers) as pool:
                return np.array(pool.map(_condition_map, conditions))
        return np.array([_condition_map(c) for c in conditions])
    finally:
        _MAP_INPUTS = None


def _method_reference(layout):
    """The sha1 of the method's reference, or None when the method pass never ran; refuses a pass without it."""
    if not layout.scores("method").exists():
        return None
    missing = [path.name for path in (layout.method_bank, layout.method_zstats) if not path.exists()]
    if missing:
        raise ValueError(f"run the method-reference stage first: {', '.join(missing)} missing")
    return {"method_bank": sha1(layout.method_bank), "method_zstats": sha1(layout.method_zstats)}


def _method_rows(layout, names) -> dict:
    """Our six rows, from the stored statistics and the reference's."""
    statistics = stack(layout.scores("method"), names, KEYS)
    with np.load(layout.method_bank) as bank, np.load(layout.method_zstats) as zstats:
        bank, zstats = {k: bank[k] for k in KEYS}, {k: zstats[k] for k in KEYS}
    return tables.method_rows(statistics, bank, zstats)


def write_report(settings, manifest) -> None:
    """Every table of the report, from the stored scores of every pass that ran."""
    layout = settings.layout
    reference = _method_reference(layout)  # checked first: the mAP below takes minutes on all 5,000 images
    names = [p.name for p in settings.dataset.evaluation_images()]
    detector = stack(layout.scores("detector"), names, DETECTOR_ARRAYS)
    discopatch = stack(layout.scores("discopatch"), names, ("dcp",))["dcp"] if layout.scores("discopatch").exists() \
        else None
    activations = stack(layout.scores("activations"), names, ACTIVATION_ARRAYS) \
        if layout.scores("activations").exists() else None
    scores = tables.baseline_rows(detector, discopatch=discopatch, activations=activations)
    gt = settings.dataset.ground_truth()
    ids = [gt.image_id(n) for n in names]
    condition_map = condition_maps(gt, ids, detector, settings.workers)
    clean = {i: coco_results(i, detector["det_scores"][k, 0], detector["det_labels"][k, 0], detector["det_boxes"][k, 0],
                             gt.category_ids) for k, i in enumerate(ids)}
    ap = per_image_ap(gt, clean, ids)
    if reference:
        scores.update(_method_rows(layout, names))
    timing = json.loads(layout.timing.read_text()) if layout.timing.exists() else {}
    report_tables, summary = tables.build_tables(scores, detector, ap, settings.dataset.folds(), condition_map,
                                                 seed=settings.seed, samples=tables.BOOTSTRAP_SAMPLES,
                                                 workers=settings.workers, timing=timing)
    summary["inputs"] = {**manifest.read().get("inputs", {}), **({"method": reference} if reference else {})}
    tables.write_outputs(layout.report(), report_tables, summary)
    print(f"[report] headline: {summary['headline_decision']}; level score: {summary['level_decision']}", flush=True)
```

`samples=tables.BOOTSTRAP_SAMPLES` is passed at call time, so a test that sets the constant takes effect. A default argument would have been bound when the module was imported.

- [ ] **Step 5: Register the stage** in `degradation_monitor/stages/__init__.py`:
- Change the import to `from . import baselines, method, report`.
- Add `"report": report.write_report,` after `"method-pass": method.method_pass,`.

- [ ] **Step 6: Run the tests**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/evaluation/test_report.py tests/stages/test_report.py`
Expected: 17 passed.

- [ ] **Step 7: Run the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: "227 passed, 1 skipped" (210 + 14 + 3).

- [ ] **Step 8: Commit**

```bash
git add degradation_monitor/evaluation/report.py degradation_monitor/stages tests/evaluation/test_report.py tests/stages/test_report.py
git commit -q -m "feat: one report for our method and the six baselines" -m "evaluation/report.py scores every row on all, untouched, held-out and screen images, with paired bootstrap intervals in which ContrastiveConf's lambda is cross-fitted again, and states both pre-registered decisions; stages/report.py loads the run folder and computes per-condition mAP in worker processes. It replaces both the baselines' report and the confirmation." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

### Task 13: The command line, and the old package retired

Every stage now has a replacement, so the old package goes:
- **The new CLI:** `python -m degradation_monitor <stage> [--config configs/coco.toml] [overrides]` replaces `python -m differential_uncertainty baselines-coco --phase ...`.
- **The archive:** the conv-TU pilot's code and tests go to `archive/conv_tu/`. They are copied from the commit the branch was cut from, so the archive holds them exactly as they last ran, before this branch rewrote their imports. `convtu/report.py` is already there from Task 3.
- **Deletions:** the rest of `differential_uncertainty/` and `tests/differential_uncertainty/`. The superseded files are the old pipelines, the two old reports, the confirmation and the old CLI.

**Files:**
- Create:
  - `degradation_monitor/cli.py`, `degradation_monitor/__main__.py`;
  - `archive/conv_tu/{graph,features,tap,channels,pipeline}.py`;
  - `archive/conv_tu/tests/{test_convtu_graph,test_convtu_features,test_convtu_tap,test_convtu_channels,test_convtu_pipeline,convtu_fakes}.py`.
- Delete: `differential_uncertainty/`, `tests/differential_uncertainty/`.
- Modify: `pyproject.toml`, `archive/README.md`.
- Test: `tests/test_cli.py`.

**Interfaces:**
- Consumes:
  - `load_settings(path, **overrides)`, `PATH_FIELDS` (Task 9);
  - `STAGES`, `run_stage` (Tasks 10–12).
- Produces:
  - `degradation_monitor.cli`:
    - `DEFAULT_CONFIG`;
    - `positive_int`, `nonnegative_int`, `positive_float`;
    - `build_parser()`, `main(argv=None) -> int`.
  - `degradation_monitor.__main__`.

- [ ] **Step 1: Write the failing tests** in `tests/test_cli.py`:

```python
import runpy

import pytest

from degradation_monitor import cli, stages

CONFIG = """
run = "{root}/runs/coco"
checkpoint = "{root}/ckpt.pth"
train_images = "{root}/train"
val_images = "{root}/val"
annotations = "{root}/ann.json"
discopatch_root = "{root}/dcp"
batch_size = 8
"""


def _config(tmp_path):
    path = tmp_path / "coco.toml"
    path.write_text(CONFIG.format(root=tmp_path))
    return path


def _recorded(monkeypatch):
    calls = []
    monkeypatch.setattr(stages, "run_stage", lambda name, settings: calls.append((name, settings)))
    return calls


def test_every_stage_is_a_command():
    assert set(stages.STAGES) == {"check", "knn-bank", "detector-pass", "discopatch-train", "discopatch-pass",
                                  "hashemi-fit", "cdf-fit", "cdf-zstats", "activation-pass", "method-reference",
                                  "method-pass", "report", "timing"}
    parser = cli.build_parser()
    assert all(parser.parse_args([name]).stage == name for name in stages.STAGES)


def test_the_cli_forwards_the_config_and_its_overrides_to_the_stage(tmp_path, monkeypatch):
    calls = _recorded(monkeypatch)
    code = cli.main(["report", "--config", str(_config(tmp_path)), "--device", "cpu", "--limit", "10",
                     "--workers", "0", "--gpu-memory-gib", "1.5", "--run", str(tmp_path / "other")])
    assert code == 0
    (name, settings), = calls
    assert name == "report" and settings.run == tmp_path / "other"
    assert (settings.device, settings.limit, settings.workers, settings.gpu_memory_gib) == ("cpu", 10, 0, 1.5)
    assert settings.batch_size == 8 and settings.epochs == 65  # from the file and the default


@pytest.mark.parametrize(("option", "value"), (("--batch-size", "0"), ("--workers", "-1"), ("--gpu-memory-gib", "0")))
def test_the_cli_rejects_invalid_numbers(option, value, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["report", option, value])
    assert error.value.code == 2
    assert "must be" in capsys.readouterr().err


def test_the_cli_rejects_a_retired_phase(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["convtu-means"])
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_the_cli_prints_plain_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(stages, "run_stage", lambda *_args: (_ for _ in ()).throw(ValueError("bad input")))
    code = cli.main(["report", "--config", str(_config(tmp_path))])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert captured.err == "error: bad input\n"


def test_a_missing_config_is_a_plain_error(tmp_path, capsys):
    assert cli.main(["report", "--config", str(tmp_path / "missing.toml")]) == 2
    assert "missing.toml" in capsys.readouterr().err


def test_the_entry_module_does_not_run_the_cli_when_imported_by_a_spawned_worker():
    runpy.run_module("degradation_monitor.__main__", run_name="__mp_main__")
```

The last test moves here from the old `test_baselines_pipeline.py`. The image workers start with `spawn`, and spawn imports the entry module under that name.

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/test_cli.py`
Expected: an ERROR at collection, `ImportError: cannot import name 'cli' from 'degradation_monitor'`.

- [ ] **Step 3: Implement `degradation_monitor/cli.py` and `degradation_monitor/__main__.py`**

`degradation_monitor/cli.py`:

```python
"""Command line: python -m degradation_monitor <stage> [--config configs/coco.toml] [overrides]."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "coco.toml"


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from error
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive number") from error
    if not number > 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return number


def build_parser() -> argparse.ArgumentParser:
    from .stages import STAGES
    parser = argparse.ArgumentParser(prog="python -m degradation_monitor",
                                     description="Corruption detection inside a frozen RT-DETRv2-R18: run one stage.")
    parser.add_argument("stage", choices=list(STAGES))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="paths and run options (TOML)")
    parser.add_argument("--run", type=Path, help="the run folder, instead of the config's")
    parser.add_argument("--device")
    parser.add_argument("--limit", type=positive_int, help="only the first N evaluation images")
    parser.add_argument("--batch-size", type=positive_int)
    parser.add_argument("--workers", type=nonnegative_int)
    parser.add_argument("--gpu-memory-gib", type=positive_float, help="this process's share of a shared GPU")
    parser.add_argument("--epochs", type=positive_int, help="DisCoPatch's training epochs")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from . import stages
        from .settings import load_settings
        settings = load_settings(args.config, run=args.run, device=args.device, limit=args.limit,
                                 batch_size=args.batch_size, workers=args.workers,
                                 gpu_memory_gib=args.gpu_memory_gib, epochs=args.epochs)
        stages.run_stage(args.stage, settings)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0
```

`degradation_monitor/__main__.py`:

```python
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the CLI tests**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/test_cli.py`
Expected: 9 passed.

- [ ] **Step 5: Archive the conv-TU pilot, exactly as it last ran**

```bash
BASE=$(git merge-base HEAD fingerprint_bank)
git rev-parse --short "$BASE"                   # note this hash: archive/README.md names it in Step 7
mkdir -p archive/conv_tu/tests
for f in graph features tap channels pipeline; do
  git show "$BASE:differential_uncertainty/convtu/$f.py" > "archive/conv_tu/$f.py"
done
for t in test_convtu_graph test_convtu_features test_convtu_tap test_convtu_channels test_convtu_pipeline convtu_fakes; do
  git show "$BASE:tests/differential_uncertainty/$t.py" > "archive/conv_tu/tests/$t.py"
done
git show "$BASE:differential_uncertainty/convtu/report.py" | cmp - archive/conv_tu/report.py && echo "report.py is the base version"
```

Expected: the last line prints `report.py is the base version`.

- [ ] **Step 6: Delete the old package and its tests**

```bash
git rm -r -q differential_uncertainty tests/differential_uncertainty
```

Also delete the transitional `test_channel_statistics_equal_the_old_pilot_statistics` from `tests/method/test_statistics.py` (Task 8). It compares with the deleted `convtu/channels.py`, whose snapshot is now in the archive.

In `pyproject.toml`, delete the comment line `# "tests/differential_uncertainty" is removed again in Task 13, with the last old test.`, and set `pythonpath = [".", "tests"]`.

Check:

```bash
grep -rn "differential_uncertainty" degradation_monitor tests configs pyproject.toml
```

Expected: prints nothing.

- [ ] **Step 7: Describe the conv-TU archive.** Append to `archive/README.md`, using the hash from Step 5:

```markdown

## `conv_tu/`: Topological Uncertainty on the backbone's conv layers (30 September – 1 October 2026)

- **What it was.** For each of the four stride-1 3 × 3 convs `res_layers[s].blocks[1].branch2a.conv`, a graph whose
  edge weights are the products of input activations and kernel weights. Its fingerprint was the top 1% of the
  graph's persistence diagram (a maximum spanning tree), compared with a clean bank by kNN.
  - **Code:** `graph.py` (the diagram), `features.py` (fingerprints and kNN), `tap.py` (the conv inputs with their
    folded kernels), `channels.py` (the follow-up's per-channel statistics), `pipeline.py` (the phases) and
    `report.py`.
  - **Results:** `docs/conv-tu-pilot-results.md`, `docs/results/conv-tu-pilot/` and `docs/results/conv-tu-pilot-channels/`.
- **Why it was retired.** The topology added nothing measurable. On the pilot's 200 images, its AUROC on the common
  families was 0.661, against 0.658 for the same number of heaviest edges without the cycle rule. The plain mean
  |activation| per channel reached 0.798.
- **What it led to.** That channel-means control became the current method in `degradation_monitor/method/`: each
  channel's level and peak share, judged against similar clean scenes.
- **Last run.** Branch `fingerprint_bank` at commit `<hash from Step 5>`: the full suite passed there (296 passed, 2
  skipped).
```

Replace `<hash from Step 5>` with the literal hash printed in Step 5.

- [ ] **Step 8: Run the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: "145 passed, 1 skipped". That is 228 collected, less the 90 tests in `tests/differential_uncertainty/` and the transitional statistics test, plus the 9 CLI tests. The 90 are:

| File | Tests |
|---|---|
| `test_baselines_pipeline.py` | 17 |
| `test_baselines_report.py` | 10 |
| `test_cli.py` | 6 |
| `test_convtu_channels.py` | 9 |
| `test_convtu_confirmation.py` | 5 |
| `test_convtu_features.py` | 4 |
| `test_convtu_graph.py` | 15 |
| `test_convtu_pipeline.py` | 20 |
| `test_convtu_tap.py` | 4 |

The skipped test is DisCoPatch's CUDA-only autocast check.

- [ ] **Step 9: Commit**

```bash
git add -A degradation_monitor tests archive pyproject.toml differential_uncertainty
git commit -q -m "refactor: the new command line, and the old package retired" -m "python -m degradation_monitor <stage> replaces baselines-coco --phase. The conv-TU pilot's code and tests are archived exactly as they last ran on fingerprint_bank; the rest of differential_uncertainty, whose every phase now has a stage, is deleted with its tests." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

### Task 14: Convert the stored run folder

`scripts/convert_runs.py` turns `runs/coco-baselines/` into `runs/coco/` once. Nothing is recomputed.
- **Hard links:** every per-image file and every fitted reference is hard-linked. A hard link is the same file under a second name, so no disk is used and the old folder's files keep their content.
- **Rewrites:** the method's bank and z-statistics keep only the level and top-1% arrays.
- **The manifest:** it records the old run's markers (`run_config.json`, `fits.json`, `checkpoint.json`, `environment.json`, `sanity.json`), each checked against the files first. Each input is named by its layout name, such as `knn_bank`, and that name fixes its path in the run folder. So the manifest stores sha1 hashes without paths, exactly as the stages record them.
- **Atomic:** the new folder is built under a temporary name and renamed only after every check has passed. A failed conversion leaves nothing behind.

**Files:**
- Create: `scripts/convert_runs.py`.
- Test: `tests/test_convert_runs.py`.

**Interfaces:**
- Consumes:
  - `load_settings`, `Settings.protocol()`, `RunLayout` and its properties;
  - `Manifest`, `SCORE_KEYS`, `atomic_npz`, `sha1`, `sha256` (Task 9);
  - `KEYS` (Task 8);
  - `hashemi.K`, `activation_cdf.BINS` (Task 7);
  - `DEFAULT_CONFIG` (Task 13);
  - the stages, whose `check_inputs` records the conversion must match (Tasks 10, 11).
- Produces: `scripts.convert_runs`:
  - `SCORE_FOLDERS`, `LINKED`, `REWRITTEN`, `PROTOCOL_FIELDS`;
  - `check_protocol(old, protocol)`, `score_files(old, names)`, `inputs(old)`, `rewrite_reference(source, target)`;
  - `convert(old, settings) -> record`, `main(argv=None)`.

- [ ] **Step 1: Write the failing tests** in `tests/test_convert_runs.py`:

```python
import json
import os

import numpy as np
import pytest
from PIL import Image

from degradation_monitor.method.statistics import KEYS
from degradation_monitor.runs import SCORE_KEYS, Manifest, sha1
from degradation_monitor.settings import Settings
from degradation_monitor.stages import run_stage
from scripts import convert_runs

WIDTHS = (2, 3, 4, 5)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _statistics(rows, seed):
    rng = np.random.default_rng(seed)
    return {key: rng.random((rows, WIDTHS[int(key[-1]) - 1])).astype(np.float32) for key in KEYS}


@pytest.fixture
def old(tmp_path):
    """A complete old-layout run folder for three evaluation images, and the clean branch's settings."""
    val = tmp_path / "val"
    val.mkdir()
    for index in range(3):
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(val / f"{index:04d}.jpg")
    (tmp_path / "ckpt.pth").write_bytes(b"weights")
    settings = Settings(run=tmp_path / "runs" / "coco", checkpoint=tmp_path / "ckpt.pth", train_images=tmp_path,
                        val_images=val, annotations=tmp_path / "ann.json", discopatch_root=tmp_path, device="cpu",
                        workers=0)
    folder = tmp_path / "runs" / "coco-baselines"
    protocol = settings.protocol()
    _write_json(folder / "run_config.json", {"checkpoint": str(settings.checkpoint), "val_images": str(val),
                                             **{field: protocol[field] for field in convert_runs.PROTOCOL_FIELDS}})
    _write_json(folder / "environment.json", {"packages": {"torch": "2.8"}, "discopatch_commit": "abc", "gpu": "RTX"})
    _write_json(folder / "sanity.json", {"coco_val_ap": 0.479, "images": 5000})
    for source in convert_runs.LINKED.values():
        (folder / source).parent.mkdir(parents=True, exist_ok=True)
        (folder / source).write_bytes(source.encode())  # each file's own name, so every sha1 differs
    for index, source in enumerate(convert_runs.REWRITTEN.values()):
        np.savez(folder / source, **_statistics(4, index), grid_s1=np.ones((4, 32)), p99_s1=np.ones((4, 2)))
    for name, source in convert_runs.SCORE_FOLDERS.items():
        (folder / source).mkdir()
        for image in settings.dataset.evaluation_images():
            arrays = _statistics(96, 7) if name == "method" else {key: np.zeros(96) for key in SCORE_KEYS[name]}
            np.savez(folder / source / f"{image.stem}.npz", **arrays)
    _write_json(folder / "test_activation" / "fits.json", {
        "hashemi-fit": {"path": "x", "sha1": sha1(folder / convert_runs.LINKED["hashemi_intervals"])},
        "cdf-fit": {"path": "x", "sha1": sha1(folder / convert_runs.LINKED["cdf_reference"])},
        "cdf-zstats": {"path": "x", "sha1": sha1(folder / convert_runs.LINKED["cdf_zstats"])},
        "hashemi_k": 2.0, "cdf_bins": 1000})
    _write_json(folder / "test_dcp" / "checkpoint.json",
                {"checkpoint": "x", "sha1": sha1(folder / convert_runs.LINKED["discopatch_checkpoint"])})
    return folder, settings


def test_the_conversion_links_every_file_rewrites_the_method_reference_and_writes_the_manifest(old):
    folder, settings = old
    convert_runs.convert(folder, settings)
    layout = settings.layout
    for name, source in convert_runs.SCORE_FOLDERS.items():
        for path in sorted((folder / source).glob("*.npz")):
            assert os.path.samefile(path, layout.score_file(name, path))
    for attribute, source in convert_runs.LINKED.items():
        assert os.path.samefile(folder / source, getattr(layout, attribute))
    for attribute, source in convert_runs.REWRITTEN.items():
        with np.load(getattr(layout, attribute)) as new, np.load(folder / source) as before:
            assert set(new.files) == set(KEYS)
            assert all(np.array_equal(new[k], before[k]) and new[k].dtype == before[k].dtype for k in KEYS)
    manifest = Manifest(layout).read()
    assert manifest["protocol"] == json.loads(json.dumps(settings.protocol()))
    assert manifest["inputs"]["detector"] == {"knn_bank": sha1(layout.knn_bank)}
    assert manifest["check"]["coco_val_ap"] == 0.479 and manifest["environment"]["gpu"] == "RTX"
    assert manifest["conversion"]["linked"]["method"] == 3
    assert manifest["conversion"]["rewritten"]["method_bank"]["dropped"] == ["grid_s1", "p99_s1"]
    assert (settings.run / "logs").is_dir() and not (settings.run.parent / ".coco.converting").exists()


def test_the_new_stages_accept_the_converted_folder_as_complete(old):
    folder, settings = old
    convert_runs.convert(folder, settings)
    for stage in ("detector-pass", "discopatch-pass", "activation-pass", "method-pass"):
        run_stage(stage, settings)  # nothing is pending, and every recorded input matches its file
    assert len(list(settings.layout.scores("method").glob("*.npz"))) == 3


def test_conversion_refuses_an_existing_target(old):
    folder, settings = old
    settings.run.mkdir(parents=True)
    with pytest.raises(ValueError, match="already exists"):
        convert_runs.convert(folder, settings)


def test_conversion_refuses_an_incomplete_method_pass(old):
    folder, settings = old
    sorted((folder / "test_convtu_means").glob("*.npz"))[0].unlink()
    with pytest.raises(ValueError, match="test_convtu_means is incomplete: 1 of 3"):
        convert_runs.convert(folder, settings)
    assert not settings.run.exists() and not (settings.run.parent / ".coco.converting").exists()


def test_conversion_refuses_a_score_file_without_its_arrays(old):
    folder, settings = old
    victim = sorted((folder / "test_convtu_means").glob("*.npz"))[0]
    np.savez(victim, means_s1=np.zeros((96, 2)))
    with pytest.raises(ValueError, match=f"{victim.name} lacks"):
        convert_runs.convert(folder, settings)


def test_conversion_refuses_an_old_run_with_another_protocol(old):
    folder, settings = old
    config = json.loads((folder / "run_config.json").read_text())
    (folder / "run_config.json").write_text(json.dumps({**config, "seed": 45}))
    with pytest.raises(ValueError, match=r"another protocol \(seed\)"):
        convert_runs.convert(folder, settings)


def test_conversion_refuses_scores_computed_from_other_fits(old):
    folder, settings = old
    fits = json.loads((folder / "test_activation" / "fits.json").read_text())
    fits["cdf-fit"]["sha1"] = "0" * 40
    (folder / "test_activation" / "fits.json").write_text(json.dumps(fits))
    with pytest.raises(ValueError, match="other fits"):
        convert_runs.convert(folder, settings)
    assert not settings.run.exists()
```

`from scripts import convert_runs` works because `.` is on pytest's `pythonpath` and `scripts/` is a namespace package. It needs no `__init__.py`.

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/test_convert_runs.py`
Expected: an ERROR at collection, `ImportError: cannot import name 'convert_runs' from 'scripts'` (or `ModuleNotFoundError: No module named 'scripts'`).

- [ ] **Step 3: Implement `scripts/convert_runs.py`**

```python
"""One-time conversion of the old run folder (runs/coco-baselines/) into the clean layout (runs/coco/).

    python scripts/convert_runs.py --old runs/coco-baselines [--config configs/coco.toml]

- Hard links: every per-image file and every fitted reference. A hard link is the same file under a second name, so
  nothing is copied, its content cannot differ from the source, and the old folder's files are untouched.
- Rewritten: the method's bank and z-statistics, keeping only the level and top-1% arrays. Each kept array is checked
  to equal its source exactly.
- The manifest: the protocol, checked against the old run's configuration; the old environment and sanity check; the
  inputs each score folder was computed from, checked against the old markers; and this conversion's record.
- Refused: an existing target, an old run with another protocol, an incomplete or malformed score folder, and scores
  computed from other fits than the stored ones. Every check runs before anything is written. The new folder is
  built under a temporary name and renamed only when it is complete.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from degradation_monitor.baselines.activation_cdf import BINS as CDF_BINS  # noqa: E402
from degradation_monitor.baselines.hashemi import K as HASHEMI_K  # noqa: E402
from degradation_monitor.cli import DEFAULT_CONFIG  # noqa: E402
from degradation_monitor.method.statistics import KEYS  # noqa: E402
from degradation_monitor.runs import SCORE_KEYS, Manifest, RunLayout, atomic_npz, sha1, sha256  # noqa: E402
from degradation_monitor.settings import load_settings  # noqa: E402

SCORE_FOLDERS = {"detector": "test", "activations": "test_activation", "discopatch": "test_dcp",
                 "method": "test_convtu_means"}
LINKED = {  # RunLayout property: the old file, relative to the old run folder
    "knn_bank": "bank/knn_bank.npy", "knn_names": "bank/bank_names.json",
    "cdf_reference": "cdf/reference.npz", "cdf_fit": "cdf/fit.json", "cdf_zstats": "cdf/zstats.json",
    "hashemi_intervals": "hashemi/intervals.npz", "hashemi_fit": "hashemi/fit.json",
    "discopatch_checkpoint": "discopatch/DisCoPatch/Discriminator_coco.pt",
    "discopatch_training": "discopatch/training.json", "timing": "timing.json",
}
REWRITTEN = {"method_bank": "convtu/channels_bank.npz", "method_zstats": "convtu/channels_zstats.npz"}
MARKERS = ("run_config.json", "environment.json", "sanity.json", "test_activation/fits.json",
           "test_dcp/checkpoint.json")
PROTOCOL_FIELDS = ("seed", "limit", "folds", "conditions", "top_k", "knn_k", "knn_k_max", "theta")


def _json(path: Path):
    return json.loads(path.read_text())


def check_protocol(old: Path, protocol: dict) -> None:
    """The old run's configuration must be the protocol the clean branch records, checkpoint included."""
    config = _json(old / "run_config.json")
    differ = [field for field in PROTOCOL_FIELDS if config.get(field) != protocol[field]]
    checkpoint = Path(config["checkpoint"])
    if not checkpoint.exists() or sha256(checkpoint) != protocol["checkpoint_sha256"]:
        differ.append("checkpoint")
    if differ:
        raise ValueError(f"the old run used another protocol ({', '.join(differ)}): {old / 'run_config.json'}")


def score_files(old: Path, names: list) -> dict:
    """Every evaluation image's file in each old score folder; refuses a missing or incomplete one."""
    out = {}
    for folder, source in SCORE_FOLDERS.items():
        paths = [old / source / f"{Path(n).stem}.npz" for n in names]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            raise ValueError(f"{source} is incomplete: {len(missing)} of {len(names)} images missing, "
                             f"e.g. {missing[0]}")
        for path in paths:
            with np.load(path) as data:
                lacking = set(SCORE_KEYS[folder]) - set(data.files)
            if lacking:
                raise ValueError(f"{source}/{path.name} lacks {sorted(lacking)}")
        out[folder] = paths
    return out


def inputs(old: Path) -> dict:
    """What each score folder was computed from, checked against the old run's markers."""
    activations = {"hashemi_intervals": sha1(old / LINKED["hashemi_intervals"]),
                   "cdf_reference": sha1(old / LINKED["cdf_reference"]), "cdf_zstats": sha1(old / LINKED["cdf_zstats"]),
                   "hashemi_k": HASHEMI_K, "cdf_bins": CDF_BINS}
    fits = _json(old / "test_activation" / "fits.json")
    recorded = {"hashemi_intervals": fits["hashemi-fit"]["sha1"], "cdf_reference": fits["cdf-fit"]["sha1"],
                "cdf_zstats": fits["cdf-zstats"]["sha1"], "hashemi_k": fits["hashemi_k"], "cdf_bins": fits["cdf_bins"]}
    if recorded != activations:
        raise ValueError("the activation scores were computed from other fits than the stored ones")
    discopatch = {"discriminator": sha1(old / LINKED["discopatch_checkpoint"])}
    if _json(old / "test_dcp" / "checkpoint.json")["sha1"] != discopatch["discriminator"]:
        raise ValueError("the DisCoPatch scores were computed with another discriminator than the stored one")
    return {"detector": {"knn_bank": sha1(old / LINKED["knn_bank"])}, "activations": activations,
            "discopatch": discopatch}


def rewrite_reference(source: Path, target: Path) -> dict:
    """The method's reference with only the level and top-1% arrays, each equal to its source exactly."""
    with np.load(source) as data:
        missing = set(KEYS) - set(data.files)
        if missing:
            raise ValueError(f"{source} lacks {sorted(missing)}")
        kept = {key: data[key] for key in KEYS}
        dropped = sorted(set(data.files) - set(KEYS))
    atomic_npz(target, **kept)
    with np.load(target) as written:
        for key in KEYS:
            if written[key].dtype != kept[key].dtype or not np.array_equal(written[key], kept[key]):
                raise ValueError(f"{target.name}: {key} was not written exactly")
    return {"dropped": dropped, "sha1": sha1(target)}


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, target)


def convert(old, settings) -> dict:
    """Build settings.run from the old run folder; returns the conversion record."""
    old, target = Path(old), settings.run
    staging = target.with_name(f".{target.name}.converting")
    for path in (target, staging):
        if path.exists():
            raise ValueError(f"{path} already exists; remove it to convert again")
    missing = [s for s in (*MARKERS, *LINKED.values(), *REWRITTEN.values()) if not (old / s).exists()]
    if missing:
        raise ValueError(f"the old run folder lacks {', '.join(missing)}: {old}")
    protocol = settings.protocol()
    check_protocol(old, protocol)
    files = score_files(old, [p.name for p in settings.dataset.evaluation_images()])
    recorded_inputs = inputs(old)
    layout = RunLayout(staging)
    try:
        for attribute, source in LINKED.items():
            _link(old / source, getattr(layout, attribute))
        for folder, paths in files.items():
            for path in paths:
                _link(path, layout.score_file(folder, path))
        rewritten = {attribute: {"source": source, **rewrite_reference(old / source, getattr(layout, attribute))}
                     for attribute, source in REWRITTEN.items()}
        (staging / "logs").mkdir()
        record = {"source": str(old.resolve()), "converted": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "linked": {**{folder: len(paths) for folder, paths in files.items()},
                             "references": sorted(LINKED.values())},
                  "rewritten": rewritten}
        Manifest(layout).update(protocol=protocol, environment=_json(old / "environment.json"),
                                check=_json(old / "sanity.json"), inputs=recorded_inputs, conversion=record)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)  # links and rewritten files only; the old files stay
        raise
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Convert the old run folder into the clean layout, once.")
    parser.add_argument("--old", type=Path, required=True, help="the old run folder, e.g. runs/coco-baselines")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="names the new run folder")
    args = parser.parse_args(argv)
    record = convert(args.old, load_settings(args.config))
    print(json.dumps({"linked": record["linked"], "rewritten": record["rewritten"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`shutil.rmtree` on the staging folder removes only the new names. A hard-linked file's data stays, because the old name still points to it.

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/test_convert_runs.py`
Expected: 7 passed.

- [ ] **Step 5: Run the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: "152 passed, 1 skipped".

- [ ] **Step 6: Commit**

```bash
git add scripts/convert_runs.py tests/test_convert_runs.py
git commit -q -m "feat: a one-time, checked conversion of the old run folder" -m "scripts/convert_runs.py hard-links every per-image file and fitted reference of runs/coco-baselines/ into the new layout, rewrites the method's bank and z-statistics with only the level and top-1% arrays, and folds the old markers into the manifest. Every check runs before anything is written." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

- [ ] **Step 7: Convert the real run folder**

It writes only `runs/coco/`. `runs/coco-baselines/` is only read, and its files gain a second name.

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python scripts/convert_runs.py --old /home/yuchen/YuchenZ/UE/philip_sa/runs/coco-baselines`
Expected:
- `linked` shows `detector`, `activations`, `discopatch` and `method` at 5000 each, plus the ten references;
- `rewritten` drops `grid_s1`–`grid_s4` and `p99_s1`–`p99_s4` from both method files.

Then check:

```bash
ls /home/yuchen/YuchenZ/UE/philip_sa/runs/coco/scores/method | wc -l
stat -c '%h %n' /home/yuchen/YuchenZ/UE/philip_sa/runs/coco/scores/method/000000000139.npz
```

Expected: `5000`, then `2 ...` (two names, one file).

### Task 15: The report on the converted folder, and equivalence with the old code

This task proves success criterion 3 of the spec on the real data, in three tests:
- the baselines' per-condition separation, against `docs/results/coco-baselines/separation.csv`;
- the baselines' separation intervals, against `docs/results/coco-baselines/intervals.csv`, whose quantities are computed exactly as before;
- the old confirmation, against `docs/results/conv-tu-conditioned/summary.json`, with the renamed rows.

The tests skip when the report of the converted folder is absent. The report runs on the CPU, with 9 workers, in about 30–40 minutes.

**Files:**
- Test: `tests/test_equivalence.py`.
- Create: `docs/results/coco/`, a copy of the new report's tables.

**Interfaces:**
- Consumes:
  - `load_settings` (Task 9);
  - `RunLayout.report()` (Task 9);
  - the `report` stage and its output (Task 12);
  - `QUANTITIES` (Task 12).
- Produces: nothing new.

- [ ] **Step 1: Write the tests** in `tests/test_equivalence.py`:

```python
"""The clean report on the converted run folder reproduces the old code's numbers (spec, success criterion 3)."""
import csv
import json
from pathlib import Path

import pytest

from degradation_monitor.evaluation.report import QUANTITIES
from degradation_monitor.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]
REPORT = load_settings(ROOT / "configs" / "coco.toml").layout.report()
OLD_BASELINES = ROOT / "docs" / "results" / "coco-baselines"
OLD_CONFIRMATION = ROOT / "docs" / "results" / "conv-tu-conditioned" / "summary.json"
RENAMED = {"conditioned": "level", "global_s123": "global_level", "ch_means_knn": "means_knn",
           "ch_means_own": "means_own"}
pytestmark = pytest.mark.skipif(not (REPORT / "summary.json").exists(),
                                reason="needs the report of the converted run folder (Task 15)")


def _rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _renamed(quantity: str) -> str:
    """An old quantity name with the renamed rows, e.g. 'two_axis - conditioned:auroc_common'."""
    rows, _, metric = quantity.partition(":")
    return " - ".join(RENAMED.get(row, row) for row in rows.split(" - ")) + ":" + metric


def _summary() -> dict:
    return json.loads((REPORT / "summary.json").read_text())


def _flat(value, path=()) -> dict:
    """{path: number} of a nested dict or list; pytest.approx compares only flat mappings."""
    if isinstance(value, dict):
        return {k: v for key, item in value.items() for k, v in _flat(item, (*path, key)).items()}
    if isinstance(value, list):
        return {k: v for index, item in enumerate(value) for k, v in _flat(item, (*path, index)).items()}
    return {path: value}


def test_the_baselines_per_condition_separation_is_reproduced():
    old = {(r["method"], r["family"], r["severity"]): r for r in _rows(OLD_BASELINES / "separation.csv")}
    new = {(r["method"], r["family"], r["severity"]): r for r in _rows(REPORT / "separation.csv")
           if r["subset"] == "all"}
    assert len(old) == 9 * 95
    for key, row in old.items():
        assert new[key]["pooling"] == row["pooling"], key
        for metric in ("auroc", "aupr", "fpr95"):
            assert float(new[key][metric]) == pytest.approx(float(row[metric]), abs=1e-12), (key, metric)


def test_the_baselines_separation_intervals_are_reproduced():
    new = _summary()["intervals"]["all"]
    old = [r for r in _rows(OLD_BASELINES / "intervals.csv") if r["quantity"].split(":")[1] in QUANTITIES]
    assert len(old) == 9 * len(QUANTITIES)
    for row in old:
        for field in ("point", "low", "high"):
            assert new[row["quantity"]][field] == pytest.approx(float(row[field]), abs=1e-12), (row["quantity"], field)


def test_the_confirmation_is_reproduced():
    old, new = json.loads(OLD_CONFIRMATION.read_text()), _summary()
    assert (new["headline_decision"], new["level_decision"]) == (old["headline_decision"], old["decision"])
    assert new["image_sets"] == {"all": old["images"], "untouched": old["untouched_images"],
                                 "held_out": old["held_out_images"], "screen": old["screen_images"]}
    for part in ("headline", "by_severity", "by_family"):
        for subset, rows in old[part].items():
            for row, values in rows.items():
                got = _flat(new[part][subset][RENAMED.get(row, row)])
                assert got == pytest.approx(_flat(values), abs=1e-12), (part, subset, row)
    for subset, quantities in old["intervals"].items():
        for quantity, bounds in quantities.items():
            got = new["intervals"][subset][_renamed(quantity)]
            assert (got["low"], got["high"]) == pytest.approx((bounds["low"], bounds["high"]), abs=1e-12), \
                (subset, quantity)
```

`pytest.approx` raises a `TypeError` on nested dicts, and `by_family` is nested (family, then severity). So `_flat` turns each side into a flat `{path: number}` mapping first. A missing or extra path then fails the comparison as well.

- [ ] **Step 2: Run them, expecting skips**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/test_equivalence.py`
Expected: "3 skipped". The report does not exist yet.

- [ ] **Step 3: Run the report on the converted folder** (CPU only)

```bash
CUDA_VISIBLE_DEVICES= nice -n 5 /home/yuchen/miniconda3/envs/UE/bin/python -m degradation_monitor report --device cpu > /home/yuchen/YuchenZ/UE/philip_sa/runs/coco/logs/report.log 2>&1
tail -2 /home/yuchen/YuchenZ/UE/philip_sa/runs/coco/logs/report.log
```

Expected: the last line is `[report] headline: confirmed; level score: confirmed`.

- [ ] **Step 4: Run the equivalence tests**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/test_equivalence.py`
Expected: 3 passed.

If one fails, do not loosen the tolerance. Find the first quantity that differs, and trace it back to the computation that changed (superpowers:systematic-debugging). The spec allows no computation change.

- [ ] **Step 5: Keep the new report's tables**

```bash
mkdir -p docs/results/coco
cp /home/yuchen/YuchenZ/UE/philip_sa/runs/coco/reports/coco/{summary.json,report.md,separation.csv,aggregates.csv,intervals.csv,conditions.csv,knn_k.csv,timing.csv} docs/results/coco/
```

- [ ] **Step 6: Run the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: "155 passed, 1 skipped".

- [ ] **Step 7: Commit**

```bash
git add tests/test_equivalence.py docs/results/coco
git commit -q -m "test: the clean report reproduces the old code's numbers" -m "On the converted run folder, the new report reproduces the baselines' per-condition separation and intervals and the 5,000-image confirmation (decisions, tables and intervals) to within 1e-12. Its tables are kept in docs/results/coco/." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```

### Task 16: Docs, README and requirements

This task tidies what is left: the requirements, the docs and the README.
- **Requirements:** `requirements.txt` names exactly the packages the code imports. A test now checks this.
- **Docs:** they become flat, with an index. Reports, notes and plans of retired work move to `docs/archive/` unchanged. The decision record, the baseline-numbers doc and the pilot docs get dated notes.
- **README:** it is rewritten for the new package.

**Files:**
- Modify:
  - `requirements.txt`, `README.md`, `tests/test_package.py`;
  - `docs/driving-benchmark-baselines-and-metrics.md`, `docs/coco-baseline-numbers.md`;
  - `docs/conv-tu-pilot-design.md`, `docs/conv-tu-pilot-results.md`;
  - `docs/literature-review-image-corruption-detection.md` (one link path).
- Create: `docs/README.md`, `docs/archive/README.md`.
- Move to `docs/archive/`:
  - the August reports: `docs/scene-uncertainty-{confidence-decile-results,pilot-easy-report,within-image-contrast-results}.md`;
  - the 250-image pilot: `docs/coco-imagecorruptions-250-pilot-results.md` and `docs/results/coco-imagecorruptions-250/`, as `results/coco-imagecorruptions-250/`;
  - the meeting notes: `docs/aug28meeting/`;
  - the fingerprint chart: `docs/assets/fingerprint-method-selection-pilot-auroc.png`, as `assets/`;
  - the old branch verification: `docs/clean-branch-verification.md`;
  - the old TU to-do: `RT-DETR_Topological_Uncertainty_TODO.md`, from the repository root;
  - plans and specs of finished or retired work: every file in `docs/superpowers/plans/` and `docs/superpowers/specs/` except the confirmation plan and this branch's spec and plan. They go to `plans/` and `specs/`.

**Interfaces:**
- Consumes: the stage names (Tasks 10–12), the CLI (Task 13) and the result folders (Task 15).
- Produces: nothing new.

- [ ] **Step 1: Write the failing test.** Append to `tests/test_package.py`:

```python
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DISTRIBUTIONS = {"PIL": "pillow", "sklearn": "scikit-learn", "uq_detr": "uq-detr"}  # import name -> pip name


def test_the_requirements_name_exactly_the_packages_the_code_imports():
    imported = set()
    for path in (ROOT / "degradation_monitor").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                imported.add(node.module.split(".")[0])
    third_party = {DISTRIBUTIONS.get(name, name) for name in imported
                   if name not in sys.stdlib_module_names and name != "degradation_monitor"}
    lines = [line.strip() for line in (ROOT / "requirements.txt").read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    assert {re.split(r"[<>=]", line)[0] for line in lines} == third_party
```

DisCoPatch's own code, which `baselines/discopatch.py` imports by name at run time, has its own requirements. The README says so.

- [ ] **Step 2: Run it to see it fail**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/test_package.py`
Expected: FAIL. The old file names `matplotlib` and `numba`, which nothing imports any more, and lacks `scikit-learn`, `pycocotools` and `uq-detr`.

- [ ] **Step 3: Write `requirements.txt`**

```text
torch>=2.3.0
torchvision>=0.18.0
numpy
pillow
scipy
scikit-learn
pycocotools
uq-detr
imagecorruptions==1.1.2
```

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/test_package.py`
Expected: 2 passed.

If the vendored `detector/rtdetrv2/` imports a package this list lacks, add that package to the list rather than to the test.

- [ ] **Step 4: Move the archived docs**

```bash
mkdir -p docs/archive/results docs/archive/assets docs/archive/plans docs/archive/specs
git mv docs/scene-uncertainty-confidence-decile-results.md docs/scene-uncertainty-pilot-easy-report.md docs/scene-uncertainty-within-image-contrast-results.md docs/coco-imagecorruptions-250-pilot-results.md docs/clean-branch-verification.md docs/aug28meeting docs/archive/
git mv docs/results/coco-imagecorruptions-250 docs/archive/results/coco-imagecorruptions-250
git mv docs/assets/fingerprint-method-selection-pilot-auroc.png docs/archive/assets/
git mv RT-DETR_Topological_Uncertainty_TODO.md docs/archive/
for f in docs/superpowers/plans/*.md; do
  case "$(basename "$f")" in 2026-10-01-content-conditioned-confirmation.md|2026-10-02-clean-branch.md) ;; *) git mv "$f" docs/archive/plans/ ;; esac
done
for f in docs/superpowers/specs/*.md; do
  case "$(basename "$f")" in 2026-10-02-clean-branch-design.md) ;; *) git mv "$f" docs/archive/specs/ ;; esac
done
ls docs/superpowers/plans docs/superpowers/specs
```

Expected: the last command lists only:
- `2026-10-01-content-conditioned-confirmation.md` and `2026-10-02-clean-branch.md` (plans);
- `2026-10-02-clean-branch-design.md` (specs).

In `docs/literature-review-image-corruption-detection.md`, change the one link `[results](coco-imagecorruptions-250-pilot-results.md)` to `[results](archive/coco-imagecorruptions-250-pilot-results.md)`. Then check:

```bash
grep -rn "coco-imagecorruptions-250-pilot-results\|scene-uncertainty-\|aug28meeting\|clean-branch-verification\|fingerprint-method-selection" docs/*.md docs/roundtable-2026-10-01-detection
```

Expected: only the changed link, which now points into `archive/`.

- [ ] **Step 5: Write `docs/archive/README.md`**

```markdown
# Archived docs

Reports, notes and plans of retired work, moved here unchanged on 2 October 2026. Paths inside them are as they were
written: a link to `docs/<name>` now means `docs/archive/<name>`. Their code is in `archive/` at the repository root.

- `scene-uncertainty-*.md`: the August scene-uncertainty experiments on the decoder queries.
- `coco-imagecorruptions-250-pilot-results.md` and `results/coco-imagecorruptions-250/`: the 250-image pilot of the
  decoder-query fingerprint.
- `assets/fingerprint-method-selection-pilot-auroc.png`: the fingerprint's method-selection chart.
- `aug28meeting/`: the notes for the meeting of 28 August.
- `clean-branch-verification.md`: the verification of the August `clean` worktree.
- `RT-DETR_Topological_Uncertainty_TODO.md`: the August to-do list of the query-level topological method.
- `plans/` and `specs/`: the plans and designs of finished or retired work, including the COCO baselines, the
  Hashemi baseline and the conv-TU pilot.
```

- [ ] **Step 6: Add the dated notes.** Insert each note after the doc's title line, with one blank line before and after it.

In `docs/driving-benchmark-baselines-and-metrics.md`:

```markdown
> **Note, 2 October 2026.** Since 1 October the goal is detecting corruption, not predicting how much it harms the
> detector (dev log, 1 October). The harm metrics below are no longer computed. The report computes AUROC, AUPR and
> FPR95, with each condition's mAP as context.
```

In `docs/coco-baseline-numbers.md`:

```markdown
> **Note, 2 October 2026.** The harm sections are superseded: harm is no longer an objective. The clean branch's
> report reproduces this doc's separation numbers exactly (`tests/test_equivalence.py`), and its tables are in
> `docs/results/coco/`.
```

In both `docs/conv-tu-pilot-design.md` and `docs/conv-tu-pilot-results.md`:

```markdown
> **Note, 2 October 2026.** The pilot's code is archived in `archive/conv_tu/`, and its plan in `docs/archive/plans/`.
```

- [ ] **Step 7: Write `docs/README.md`**

```markdown
# Docs

The documents behind the paper. Older work is in `archive/`.

## The method and its evidence

- `conv-tu-conditioned-results.md`: the 5,000-image confirmation of the two-axis score (2 October). Tables:
  `results/conv-tu-conditioned/`.
- `results/coco/`: the clean branch's report. It covers every row on all four image sets, with intervals.
- `roundtable-2026-10-01-detection/`: the detection roundtable that proposed the two-axis score. Its
  `verify_panel_rows.py` imports the old `differential_uncertainty` package, so run it at the commit named in
  `../archive/README.md`.
- `superpowers/plans/2026-10-01-content-conditioned-confirmation.md`: the confirmation's pre-registered plan, with
  its amendments.
- `dev-log.md`: dated observations and decisions, newest first.
- `todo.md`: the open ablations and experiments.

## Baselines and setting

- `driving-benchmark-baselines-and-metrics.md`: the decision record for baselines and metrics (27 September).
- `coco-baseline-numbers.md`: the six baselines on COCO (29 September). Tables: `results/coco-baselines/`.
- `literature-review-image-corruption-detection.md`: related work (27 September).

## The conv-TU pilot (retired)

- `conv-tu-pilot-design.md` and `conv-tu-pilot-results.md`: topology on the conv layers. Its channel-means control
  led to the current method. Tables: `results/conv-tu-pilot/`, `results/conv-tu-pilot-channels/`. Code:
  `../archive/conv_tu/`.

## This branch

- `superpowers/specs/2026-10-02-clean-branch-design.md` and `superpowers/plans/2026-10-02-clean-branch.md`: the
  design and the plan of the clean branch.
```

- [ ] **Step 8: Rewrite `README.md`**

````markdown
# Corruption detection inside a frozen object detector

This repository detects image corruption (fog, blur, noise, compression and more) from inside a frozen
RT-DETRv2-R18 trained on COCO. Nothing is trained for it. The repository holds our method and the six published
baselines we compare it with. All of them are evaluated on COCO val2017 under 19 imagecorruptions families at five
severities.

**Our method, the two-axis score.**
- **Two numbers per channel.** In the backbone's stages 1–3, each channel gives its level (mean |activation|) and its
  peak share (how far its strongest 1% of positions stand out).
- **A reference of similar scenes.** Both numbers are compared with the 50 clean COCO train scenes most like the image,
  found by the stage-4 channel means.
- **Flatten or shift.** A corruption either flattens the peaks (fog, contrast, blur) or shifts the level (noise). The
  score is the larger of the two deviations.

On all 5,000 COCO val images it reaches AUROC 0.917 on the 15 common families and 0.858 on the 4 extra ones. The
strongest baseline, the activation CDFs of Becker et al. (ICPR 2026), reaches 0.821 and 0.807. The details are in
`docs/conv-tu-conditioned-results.md`.

## Layout

```
degradation_monitor/
  cli.py, settings.py, runs.py, corruptions.py
  datasets/coco.py         COCO: train images, the seed-44 val order and folds, ground truth
  detector/                the frozen RT-DETRv2-R18: vendored code (rtdetrv2/), loader, post-processing, hooks
  baselines/               SAOD, ContrastiveConf, kNN, DisCoPatch, Hashemi et al., activation CDFs: one file each
  method/                  our method: channel statistics, the clean reference, the scores and the ablation rows
  evaluation/              separation metrics, the paired bootstrap and the report
  stages/                  the runnable, resumable steps
configs/coco.toml          this machine's paths and run options
scripts/convert_runs.py    the one-time conversion of the old run folder
archive/                   retired methods, read only (archive/README.md)
docs/                      results, decisions and the dev log (docs/README.md)
tests/                     mirrors degradation_monitor/
```

## Setup

- **Python packages:** Python 3.11 and the packages in `requirements.txt`. Install a PyTorch build for your machine
  first.
- **The detector checkpoint:** RT-DETRv2-R18 trained on COCO, `rtdetrv2_r18vd_120e_coco` (48.1 AP). The vendored model
  code comes from github.com/lyuwenyu/RT-DETR (Apache-2.0).
- **COCO 2017:** `train2017`, `val2017` and `annotations/instances_val2017.json`.
- **DisCoPatch** (Caetano et al., ICCV 2025): a clone of github.com/caetas/DisCoPatch, with its own requirements. Its
  commit is recorded in each run's `manifest.json`.

Then edit the paths in `configs/coco.toml`.

## Running

Each stage is resumable image by image. Each refuses a run folder made with another protocol, and inputs that changed
since its results were written.

```bash
python -m degradation_monitor <stage> [--config configs/coco.toml] [--device cuda:0] [--limit N]
                                      [--batch-size N] [--workers N] [--gpu-memory-gib X] [--run DIR]
```

| Stage | What it computes | Needs |
|---|---|---|
| `check` | that every input exists, and the clean COCO val AP (about 0.48) | – |
| `knn-bank` | the kNN baseline's bank: pooled last-stage features of every COCO train image | – |
| `detector-pass` | for every val image under all 96 conditions: SAOD, ContrastiveConf's parts, kNN distances, detections | `knn-bank` |
| `discopatch-train` | DisCoPatch's discriminator, with the official code and settings | – |
| `discopatch-pass` | DisCoPatch's scores | `detector-pass`, `discopatch-train` |
| `hashemi-fit`, `cdf-fit`, `cdf-zstats` | the activation monitors' clean statistics | `cdf-fit` before `cdf-zstats` |
| `activation-pass` | the scores of Hashemi et al. and of the activation CDFs | `detector-pass` and the three fits |
| `method-reference` | our method's clean reference: 2,000 bank and 500 z-statistics train images | – |
| `method-pass` | our method's channel statistics | `detector-pass` |
| `report` | every table, both pre-registered decisions, per-condition mAP | `detector-pass`; the others when present |
| `timing` | milliseconds per image for the detector and each monitor | `knn-bank` |

Run the tests with `python -m pytest -q`.

## Results

- **The run folder** (`run` in the config):
  - `manifest.json`: the protocol, the environment and the inputs of every score folder;
  - `reference/`: what the clean train images provide;
  - `scores/`: one file per val image for each pass;
  - `reports/coco/`: the report.
- **The tables kept in the repository:** `docs/results/coco/`. The documents that explain them are listed in
  `docs/README.md`.
````

- [ ] **Step 9: Run the whole suite**

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: "156 passed, 1 skipped".

- [ ] **Step 10: Commit**

```bash
git add -A docs README.md requirements.txt tests/test_package.py RT-DETR_Topological_Uncertainty_TODO.md
git commit -q -m "docs: an index, the archive of retired docs, dated notes and a new README" -m "Reports, notes and plans of retired work move to docs/archive/ unchanged; the decision record, the baseline numbers and the pilot docs get dated notes; the README describes the package, the stages and where results are; requirements.txt names exactly the packages the code imports, which a test now checks." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01QRPQSdbvyV8DvaCNdMqjgU"
```
