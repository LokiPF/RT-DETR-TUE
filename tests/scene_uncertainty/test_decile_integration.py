"""`analyze-confidence-deciles` end to end, over a synthetic six-severity artifact pair.

This is the one subcommand that runs no model: it reads a blur feature cache and the three
siblings of a results CSV, and writes seven files. So what is covered here is the *wiring* --
which loader is called on which flag, in what order, what is handed to the reporter, what the
process exit status is and what a reader sees on stderr -- and not the analysis, which
`test_decile_analysis.py` and `test_decile_reporting.py` cover against the same fixture.

The fixture is `write_decile_artifacts`, shared with those two modules rather than rebuilt
here: its cache goes through the real `ShardWriter` and its result manifest carries a real
content address, so the provenance rules below are exercised against artifacts that satisfy
them for the same reason the pilot's do, not because the fixture was shaped to pass.

Two assertions are deliberately whole-value rather than membership, and both are aimed at a
mutant that a looser assertion would let through:

* the output directory is compared as a complete sorted listing, so a staged `.tmp` file left
  unpublished, or a missing artifact, fails here rather than in whichever later reader tripped
  over it;
* stderr is compared in full on every refusal, because every refusal path in the wrapper
  reaches exit status 2 and the status alone cannot tell `--cache is not a finished artifact`
  from the loader's own complaint about the manifest it then failed to find.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.scene_uncertainty import pipeline
from src.scene_uncertainty.cli import build_parser, main
from src.scene_uncertainty.decile_analysis import ALL_QUERY_BENCHMARK

from tests.scene_uncertainty.decile_test_utils import (
    mutate_decile_artifacts,
    write_decile_artifacts,
)


DECILE_REPORT_FILES = (
    "blur_curves.png",
    "confidence_decile_heatmap.png",
    "dynamic_vs_frozen.png",
    "easy-report.md",
    "padding_sensitivity.png",
    "per_scene.csv",
    "summary.json",
)
"""The seven artifacts spec:163-171 names, sorted. Written out rather than imported so that a
rename inside `decile_reporting` fails a test about the command's output instead of agreeing
with itself."""

PADDED_IMAGE = "11"
PADDED_QUERY_IDS = [18, 19]
SEVERITY_COUNT = 6


def _analyze(artifacts: dict[str, Path], output: Path) -> int:
    return main([
        "analyze-confidence-deciles",
        "--cache", str(artifacts["cache"]),
        "--results", str(artifacts["results"]),
        "--output", str(output),
    ])


def _stderr_line(capsys) -> str:
    captured = capsys.readouterr()
    lines = captured.err.strip().splitlines()
    assert "Traceback" not in captured.err
    assert len(lines) == 1, lines
    return lines[0]


# --------------------------------------------------------------------------------------
# the happy path
# --------------------------------------------------------------------------------------


def test_the_command_publishes_every_confidence_decile_artifact(tmp_path: Path, capsys):
    """One command, two existing artifacts in, seven files out, exit status 0.

    The four content assertions are each aimed at one hand-off the wrapper makes and nothing
    else can make for it:

    * `source_partition` proves `inputs.run_metadata` reached the reporter -- an empty dict
      there produces the same seven files with an anonymous run;
    * the padded query ids prove `diagnostics` reached it as well, since `analyze_deciles`
      returns them as its second value and dropping that argument leaves `summary["padding"]`
      an empty rollup that still serialises;
    * the all-query benchmark is pinned on all four group keys plus its signal, because
      `confidence_bin` and `score_scope` alone match many rows and `next()` would take
      whichever the producer emitted first;
    * the reported row count is compared against `per_scene.csv`, which is the table the
      summary describes, so the count on stderr cannot be some other length that happens to
      be handy.
    """
    artifacts = write_decile_artifacts(tmp_path)
    output = tmp_path / "report"

    assert _analyze(artifacts, output) == 0

    assert sorted(path.name for path in output.iterdir()) == list(DECILE_REPORT_FILES)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["run_metadata"]["source_partition"] == "tuning"
    assert summary["padding"]["images"][PADDED_IMAGE]["union_padded_query_ids"] == PADDED_QUERY_IDS
    benchmark = next(
        group for group in summary["groups"]
        if (group["membership_mode"], group["confidence_bin"], group["padding_mode"])
        == ALL_QUERY_BENCHMARK
        and group["signal"] == "persistence"
        and group["score_scope"] == "layer_2"
    )
    assert benchmark["scored_severity_count"] == benchmark["total_severity_count"] == SEVERITY_COUNT
    assert {row["confidence_bin"] for row in summary["ranked_layer_2_persistence"]} >= {
        "all_valid", "decile_00_10", "decile_90_100",
    }

    scored_rows = len(pd.read_csv(output / "per_scene.csv"))
    assert _stderr_line(capsys) == (
        f"analyze-confidence-deciles: summarized {scored_rows} score rows into {output}"
    )


# Everything in `pipeline` that needs a GPU, a checkpoint, a COCO tree or a bank. The list is
# the module's own expensive surface rather than a guess: if a name is added to it that this
# command legitimately needs, the failure names which one and why.
EXPENSIVE_PIPELINE_NAMES = (
    "load_frozen_detector", "make_coco_loader", "streaming_coverage_bank",
    "deterministic_reservoir", "compute_query_distances", "score_cached_record",
    "fit_clean_distance_scale", "fit_normalizer", "_device",
)


def test_the_command_reaches_for_no_detector_and_no_knn(tmp_path: Path, monkeypatch):
    """The help's central claim, checked instead of only written down.

    "Runs no detector forward pass and no kNN search" is why this command is worth having:
    it is minutes on a CPU against artifacts someone else spent GPU hours producing, and it
    can be rerun against a second result set over the same cache. A rewrite that recomputed
    the distances rather than reading the saved ones would still publish seven plausible
    files, and every other assertion in this module would still hold.

    Scoped honestly: the detonators are installed in `pipeline`'s namespace, so what this
    proves is that `command_analyze_confidence_deciles` reaches for none of them -- not that
    `decile_analysis` contains no arithmetic of its own, which it does.
    """
    def detonator(name):
        def explode(*args, **kwargs):
            raise AssertionError(f"analyze-confidence-deciles called {name}")
        return explode

    for name in EXPENSIVE_PIPELINE_NAMES:
        assert hasattr(pipeline, name), name
        monkeypatch.setattr(pipeline, name, detonator(name))

    artifacts = write_decile_artifacts(tmp_path)
    output = tmp_path / "report"

    assert _analyze(artifacts, output) == 0

    assert sorted(path.name for path in output.iterdir()) == list(DECILE_REPORT_FILES)


def test_an_unfinished_output_directory_is_completed_rather_than_refused(tmp_path: Path):
    """The refusal is `summary.json`, not the directory: an interrupted run must be rerunnable.

    `write_decile_report` publishes all seven files or none, so a directory holding some of
    them is a run that died mid-publication, and the only way forward is to run it again onto
    the same path. A guard on `output.exists()` would pass every other test in this file --
    they all point at a path that does not exist yet -- and would leave that state unrecoverable
    without a manual `rm -rf`.
    """
    artifacts = write_decile_artifacts(tmp_path)
    output = tmp_path / "report"
    output.mkdir()
    leftover = "a half-published run left this behind\n"
    (output / "per_scene.csv").write_text(leftover, encoding="utf-8")

    assert _analyze(artifacts, output) == 0

    assert sorted(path.name for path in output.iterdir()) == list(DECILE_REPORT_FILES)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["run_metadata"]["source_partition"] == "tuning"
    assert (output / "per_scene.csv").read_text(encoding="utf-8") != leftover


# --------------------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------------------


def test_a_finished_report_is_refused_and_left_byte_identical(tmp_path: Path, capsys):
    artifacts = write_decile_artifacts(tmp_path)
    output = tmp_path / "report"
    assert _analyze(artifacts, output) == 0
    published = {path.name: path.read_bytes() for path in output.iterdir()}
    capsys.readouterr()

    assert _analyze(artifacts, output) == 2

    assert _stderr_line(capsys) == (
        "scene_uncertainty analyze-confidence-deciles: error: "
        f"Confidence-decile report is already complete: {output}"
    )
    assert {path.name: path.read_bytes() for path in output.iterdir()} == published


@pytest.mark.parametrize("mutation,expected", [
    ("test_partition_row", "is in the 'test' partition"),
    ("forged_result_manifest", "does not match its own content address"),
    ("drop_severity", "does not contain all six blur severities"),
])
def test_artifacts_that_cannot_be_proved_leave_no_report_directory(
    tmp_path: Path, capsys, mutation, expected,
):
    """A refusal from the analysis layer is one line on stderr, status 2, and no directory.

    Two claims, and the mutations are chosen to reach the wrapper by two different routes.
    `DecileAnalysisError` is a `ValueError` but not a `PipelineError`, so without the
    `except ValueError` conversion it escapes `main` as a traceback and the process exits
    non-zero for the wrong reason -- which is why the prefix is compared and not merely the
    status. And `load_decile_inputs` runs before `write_decile_report` creates anything, so a
    refusal must leave no directory at all for a later reader to mistake for a partial report.
    """
    artifacts = write_decile_artifacts(tmp_path)
    mutate_decile_artifacts(artifacts, mutation)
    output = tmp_path / "report"

    assert _analyze(artifacts, output) == 2

    message = _stderr_line(capsys)
    assert message.startswith(
        "scene_uncertainty analyze-confidence-deciles: error: "
        "Cannot analyze confidence deciles: "
    )
    assert expected in message
    assert not output.exists()


def test_a_cache_without_a_manifest_names_the_flag_that_is_wrong(tmp_path: Path, capsys):
    """`--cache` is validated as a finished artifact here, not left to the loader.

    Both routes reach status 2, so the message is the whole difference: `_existing_artifact`
    says which flag was wrong, while the loader's own `missing feature-cache manifest` names a
    path the operator never typed and no flag at all.
    """
    artifacts = write_decile_artifacts(tmp_path)
    (artifacts["cache"] / "manifest.json").unlink()
    output = tmp_path / "report"

    assert _analyze(artifacts, output) == 2

    assert _stderr_line(capsys) == (
        "scene_uncertainty analyze-confidence-deciles: error: --cache is not a finished "
        f"artifact, no manifest.json in: {artifacts['cache']}"
    )
    assert not output.exists()


def test_a_missing_results_path_is_named_before_the_analysis_is_entered(tmp_path: Path, capsys):
    """Unwrapped, so the message stays the flag's own.

    Both existence checks raise `PipelineError`, which is a `ValueError`; moving either inside
    the `try` would re-wrap it as `Cannot analyze confidence deciles: --results does not
    exist`, blaming the analysis for an argument that never reached it.
    """
    artifacts = write_decile_artifacts(tmp_path)
    absent = tmp_path / "absent.csv"
    output = tmp_path / "report"

    assert main([
        "analyze-confidence-deciles",
        "--cache", str(artifacts["cache"]),
        "--results", str(absent),
        "--output", str(output),
    ]) == 2

    assert _stderr_line(capsys) == (
        f"scene_uncertainty analyze-confidence-deciles: error: --results does not exist: {absent}"
    )
    assert not output.exists()


# --------------------------------------------------------------------------------------
# help
# --------------------------------------------------------------------------------------


def test_the_command_help_states_what_it_does_not_do(capsys):
    """The three facts that decide whether an operator reaches for this command at all.

    It is cheap because it runs no model, it cannot be pointed at the held-out test partition,
    and it will not overwrite a finished report. None of the three is visible from the argument
    list -- the absence of `--partition` in particular reads as an oversight rather than a rule
    unless the help says otherwise.
    """
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["analyze-confidence-deciles", "--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "runs no detector forward pass and no kNN search" in help_text
    assert "Tuning results only." in help_text
    assert "no `--partition` switch" in help_text
    assert "refused when it already holds a finished report" in help_text
