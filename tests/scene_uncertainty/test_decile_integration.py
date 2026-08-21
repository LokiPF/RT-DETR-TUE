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

import ast
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


FORBIDDEN_MODULE_TAILS = (".knn", ".evaluate", ".extractor", ".bank", ".dataset", ".runtime")
"""Where a detector pass, a bank build or a kNN search can come from. `pipeline` imports all of
them legitimately for the other six commands, which is why the module-level scan
`test_decile_analysis.py` runs over `decile_analysis` cannot be run over this module -- and why
the scan below is scoped to one function body instead."""


def _wrapper_syntax_tree() -> ast.FunctionDef:
    source = Path(pipeline.__file__).read_text(encoding="utf-8")
    return next(
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
        and node.name == "command_analyze_confidence_deciles"
    )


def test_the_wrapper_names_no_detector_or_knn_path_anywhere_in_its_body():
    """The static half of the no-inference guard, closing what monkeypatching cannot see.

    `monkeypatch.setattr(pipeline, name, ...)` rebinds module globals, so it is blind in two
    directions that both matter. A name resolved another way never passes through those
    globals -- `from .knn import fit_clean_distance_scale` written *inside* the function is the
    obvious one, and it is the shape someone reintroducing a computation reaches for as readily
    as a global. And a global reached only on a branch the detonator fixture does not execute,
    such as the `except` handler, is never called while the detonators are installed at all.

    So two assertions over the wrapper's own syntax tree, which has neither blind spot:

    * it contains no `import` statement of any kind. Not a blocklist -- a blocklist is dodged
      by a module name nobody thought of, whereas this function has no legitimate reason to
      import anything at run time, its three analysis entry points being module-level imports
      that the scan below then vouches for;
    * every free name it mentions that resolves to a `pipeline` global comes from somewhere
      other than the detector, bank, dataset, extraction, scoring and kNN modules -- the
      `vars(module)` idiom `test_decile_analysis.py:308` uses, narrowed from the module to the
      names this one function actually mentions, which is what makes it applicable here.

    Together with the runtime detonators this is genuinely two-sided: the detonators catch a
    reach made *indirectly*, through a helper the wrapper calls, which no scan of this one
    function body can see.
    """
    wrapper = _wrapper_syntax_tree()

    imports = [
        node for node in ast.walk(wrapper)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert imports == [], [ast.unparse(node) for node in imports]

    reached = {}
    for node in ast.walk(wrapper):
        if isinstance(node, ast.Name):
            value = vars(pipeline).get(node.id)
            if value is not None:
                origin = getattr(value, "__module__", "") or getattr(value, "__name__", "")
                reached[node.id] = origin
    assert reached, "the wrapper mentions no module global at all, so this scan proves nothing"
    assert not {
        name for name, origin in reached.items() if origin.endswith(FORBIDDEN_MODULE_TAILS)
    }, reached


def test_the_command_reaches_for_no_detector_and_no_knn(tmp_path: Path, monkeypatch):
    """The help's central claim, checked instead of only written down.

    "Runs no detector forward pass and no kNN search" is why this command is worth having:
    it is minutes on a CPU against artifacts someone else spent GPU hours producing, and it
    can be rerun against a second result set over the same cache. A rewrite that recomputed
    the distances rather than reading the saved ones would still publish seven plausible
    files, and every other assertion in this module would still hold.

    The limit is about *name binding*, not about module boundaries, and it is narrower than it
    looks: these detonators sit in `pipeline`'s module globals, so they fire for any reach
    resolved through those globals -- including one made indirectly by a helper the wrapper
    calls, which is the case the static scan above cannot see -- and for nothing else. A local
    import inside the wrapper, or a global touched only on a branch this fixture does not
    execute, passes straight through. `test_the_wrapper_names_no_detector_or_knn_path_anywhere
    _in_its_body` exists for exactly those two, and neither test subsumes the other.

    Neither says anything about `decile_analysis`, which does plenty of arithmetic of its own
    on tensors already in memory.
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


LEFTOVER = b"a half-published run left this behind\n"


def test_an_unfinished_output_directory_is_completed_rather_than_refused(tmp_path: Path):
    """The refusal is `summary.json`, not the directory: an interrupted run must be rerunnable.

    `write_decile_report` publishes all seven files or none, so a directory holding some of
    them is a run that died mid-publication, and the only way forward is to run it again onto
    the same path. A guard on `output.exists()` would pass every other test in this file --
    they all point at a path that does not exist yet -- and would leave that state unrecoverable
    without a manual `rm -rf`.

    All six of the other artifacts are staged, not just one. The epilog documents the guard as
    "a directory containing `summary.json`", and a guard keyed on any *other* one of the seven
    is equally consistent with a single stale file and with the finished-report test below,
    which publishes all seven. Six here and one there pin the file's identity between them.
    """
    artifacts = write_decile_artifacts(tmp_path)
    output = tmp_path / "report"
    output.mkdir()
    unfinished = [name for name in DECILE_REPORT_FILES if name != "summary.json"]
    for name in unfinished:
        (output / name).write_bytes(LEFTOVER)

    assert _analyze(artifacts, output) == 0

    assert sorted(path.name for path in output.iterdir()) == list(DECILE_REPORT_FILES)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["run_metadata"]["source_partition"] == "tuning"
    assert [name for name in unfinished if (output / name).read_bytes() == LEFTOVER] == []


def test_the_finished_report_guard_reads_summary_json_and_does_not_parse_it(
    tmp_path: Path, capsys,
):
    """The other half of the file's identity, on a directory holding nothing else.

    `summary.json` is the last thing a reader of a finished report would want silently
    replaced, and it is the file the epilog names. Its content is deliberately `{}`: the guard
    is an existence check on the finished run's marker, not a validity check on it, so a
    corrupt summary must still be refused rather than parsed and overwritten.
    """
    artifacts = write_decile_artifacts(tmp_path)
    output = tmp_path / "report"
    output.mkdir()
    (output / "summary.json").write_text("{}", encoding="utf-8")

    assert _analyze(artifacts, output) == 2

    assert _stderr_line(capsys) == (
        "scene_uncertainty analyze-confidence-deciles: error: "
        f"Confidence-decile report is already complete: {output}"
    )
    assert [path.name for path in output.iterdir()] == ["summary.json"]
    assert (output / "summary.json").read_text(encoding="utf-8") == "{}"


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
    # The tuning-only rule at the manifest, re-sealed so the content address cannot mask it,
    # and under both names a non-tuning run can carry. `"all"` is the one that matters most:
    # it is what `evaluate-knn --partition all` writes, it holds every held-out image, and the
    # rule that refuses it is `!= "tuning"` -- which nothing else in the suite distinguishes
    # from `== "test"`.
    ("result_partition_test", "was built for the 'test' partition"),
    ("result_partition_all", "was built for the 'all' partition"),
    # Not provenance at all, and that is the point: `decile_scoring` refuses a collapsed
    # clean-distance fit with a plain `ValueError`, from inside the `try`. It is the reachable
    # input that separates `except ValueError` from `except DecileAnalysisError` -- narrowed,
    # this one escapes `main` as a traceback while the five above still give exit 2.
    ("zero_layer_scale", "clean-distance scale must be positive"),
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
