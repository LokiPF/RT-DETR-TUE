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


FORBIDDEN_MODULE_TAILS = (
    ".knn", ".evaluate", ".extractor", ".bank", ".dataset", ".runtime", ".normalization",
)
"""Where a detector pass, a bank build, a kNN search or a re-fitted normalizer can come from.

`pipeline` imports all seven legitimately for the other six commands, which is why the
module-level scan `test_decile_analysis.py:308` runs over `decile_analysis` cannot be run over
this module -- and why the scan below is scoped to one function body instead. `.normalization`
belongs here for the same reason as `.knn`: this command reads the layer score scales the
result artifact already carries, and re-fitting them would relabel every score with a scale the
saved distances were never divided by.
"""

EXPENSIVE_PIPELINE_NAMES = (
    "load_frozen_detector", "make_coco_loader", "streaming_coverage_bank",
    "deterministic_reservoir", "compute_query_distances", "score_cached_record",
    "fit_clean_distance_scale", "fit_normalizer",
)
"""Everything in `pipeline` that needs a GPU, a checkpoint, a COCO tree or a bank, and that
`pipeline` imports from somewhere else. Every one of these must have an origin in
`FORBIDDEN_MODULE_TAILS`, and the scan below asserts exactly that rather than trusting it: two
lists naming the same rule are two lists that drift, and the drift is silent in the direction
that matters -- a name declared expensive here whose module is missing from the tails above is
caught by neither guard on any branch."""

PIPELINE_LOCAL_GPU_NAMES = ("_device",)
"""The exception that makes the name check necessary rather than redundant.

`_device` validates a CUDA request; this command has no `--device` and must never ask for one.
But it is defined in `pipeline` itself, so it has no foreign origin for the tail check to
recognise, and only its *name* identifies it."""

DETONATED_NAMES = EXPENSIVE_PIPELINE_NAMES + PIPELINE_LOCAL_GPU_NAMES


def _wrapper_syntax_tree() -> ast.FunctionDef:
    source = Path(pipeline.__file__).read_text(encoding="utf-8")
    return next(
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
        and node.name == "command_analyze_confidence_deciles"
    )


def _origin(value) -> str:
    return getattr(value, "__module__", "") or getattr(value, "__name__", "")


def test_the_wrapper_body_holds_no_import_and_no_expensive_global_name():
    """A static scan of the wrapper's own syntax tree. Named for its mechanism, not its hope.

    Three assertions, in the order they earn their place:

    * **the two name lists cannot drift apart.** Every `EXPENSIVE_PIPELINE_NAMES` entry must
      have an origin in `FORBIDDEN_MODULE_TAILS`. Without this the detonator list could grow a
      name from a module the tails do not mention -- `fit_normalizer` was exactly that -- and
      the resulting hole is invisible from either list read alone;
    * **no `import` statement of any kind in the body.** Not a blocklist of module names, which
      is dodged by a module nobody thought of. This function has no legitimate reason to import
      anything at run time, so the absolute rule is both simpler and stronger;
    * **no free name that is expensive by name or foreign by origin.** The `vars(module)` idiom
      from `test_decile_analysis.py:308`, narrowed from a module to the names one function
      mentions, which is what makes it applicable to a module that legitimately imports all
      seven forbidden ones. The name half covers `PIPELINE_LOCAL_GPU_NAMES`, which has no
      foreign origin; the origin half covers a name nobody enumerated.

    What this mechanism does **not** see, stated because an unstated limit is a false claim:
    `importlib.import_module("...")` and `exec` carry no import node and bind no free name that
    resolves in `pipeline`; a `functools.partial` alias carries no `__module__`; and raw
    `torch.cdist(...).topk(...)` reaches no project module at all, so nothing here objects to
    it. It also says nothing about code outside this one function body -- for which see the
    runtime detonators below, which walk every path instead of every line.
    """
    origins = {name: _origin(vars(pipeline)[name]) for name in EXPENSIVE_PIPELINE_NAMES}
    assert not {
        name for name, origin in origins.items()
        if not origin.endswith(FORBIDDEN_MODULE_TAILS)
    }, origins

    wrapper = _wrapper_syntax_tree()
    imports = [
        node for node in ast.walk(wrapper)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert imports == [], [ast.unparse(node) for node in imports]

    mentioned = {node.id for node in ast.walk(wrapper) if isinstance(node, ast.Name)}
    assert not mentioned & set(DETONATED_NAMES), sorted(mentioned & set(DETONATED_NAMES))

    reached = {
        name: _origin(vars(pipeline)[name]) for name in mentioned if name in vars(pipeline)
    }
    assert reached, "the wrapper mentions no module global at all, so this scan proves nothing"
    assert not {
        name for name, origin in reached.items() if origin.endswith(FORBIDDEN_MODULE_TAILS)
    }, reached


# The five paths through `command_analyze_confidence_deciles`. Enumerated rather than sampled:
# detonators only fire on code that runs, so a fixture exercising one path says nothing about
# the other four, and the four that refuse are precisely the ones a happy-path fixture misses.
WRAPPER_PATHS = (
    "cache_refused",      # `_existing_artifact` rejects --cache
    "results_refused",    # `_existing_path` rejects --results
    "already_complete",   # the finished-report guard raises FileExistsError
    "published",          # the try body succeeds
    "analysis_refused",   # the except ValueError handler runs
)


@pytest.mark.parametrize("path", WRAPPER_PATHS)
def test_no_path_through_the_command_reaches_an_expensive_entry_point(
    tmp_path: Path, monkeypatch, path,
):
    """The help's central claim, checked on every branch instead of only written down.

    "Runs no detector forward pass and no kNN search" is why this command is worth having: it
    is minutes on a CPU against artifacts someone else spent GPU hours producing. A rewrite
    that recomputed the distances rather than reading the saved ones would still publish seven
    plausible files, and every other assertion in this module would still hold.

    Detonators in `pipeline`'s module globals fire for any reach resolved through those
    globals -- including one made *indirectly*, by a helper the wrapper calls, which no scan of
    the wrapper's own body can see. That is the half the static test above cannot cover, and it
    is why the two do not subsume each other.

    Parametrized over all five paths because a detonator is only as wide as the code that runs
    under it. A single happy-path fixture leaves four branches uncovered, and they are not
    exotic: the `except` handler and the finished-report guard both run on ordinary operator
    mistakes, and a computation reintroduced on either is invisible to a test that never
    reaches it.

    The remaining limit is name binding: a name resolved outside `pipeline`'s globals -- a
    local import, `importlib` -- passes straight through, which is the static scan's half.
    Neither says anything about `decile_analysis`, which does plenty of arithmetic of its own
    on tensors already in memory.
    """
    def detonator(name):
        def explode(*args, **kwargs):
            raise AssertionError(f"analyze-confidence-deciles called {name}")
        return explode

    for name in DETONATED_NAMES:
        assert hasattr(pipeline, name), name
        monkeypatch.setattr(pipeline, name, detonator(name))

    artifacts = write_decile_artifacts(tmp_path)
    output = tmp_path / "report"
    results = artifacts["results"]
    if path == "cache_refused":
        (artifacts["cache"] / "manifest.json").unlink()
    elif path == "results_refused":
        results = tmp_path / "absent.csv"
    elif path == "already_complete":
        output.mkdir()
        (output / "summary.json").write_text("{}", encoding="utf-8")
    elif path == "analysis_refused":
        mutate_decile_artifacts(artifacts, "drop_severity")

    status = main([
        "analyze-confidence-deciles",
        "--cache", str(artifacts["cache"]),
        "--results", str(results),
        "--output", str(output),
    ])

    assert status == (0 if path == "published" else 2)
    if path == "published":
        assert sorted(entry.name for entry in output.iterdir()) == list(DECILE_REPORT_FILES)


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
