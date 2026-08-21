"""`analyze-corruption-sensitivity` end to end, over the same synthetic artifact pair.

The second cache-only subcommand, and the wiring is what is covered here: which loader is
called on which flag, what is handed to the reporter, what the process exit status is and what
a reader sees on stderr. The analysis and the bundle themselves are covered against the same
fixture by `test_corruption_analysis.py`, `test_corruption_metrics.py`,
`test_corruption_plots.py` and `test_corruption_reporting.py`.

The fixture is `write_decile_artifacts`, shared with the decile tests rather than rebuilt, so
the artifacts this command reads satisfy the loader's provenance rules for the same reason the
pilot's do. Its one image caches 20 queries at six severities, which is the fewer-than-250
image path -- fine for wiring, and the reason nothing here asserts on the prose of
`easy-report.md`, whose small-run sentences are a separate concern from what this module tests.

Three of the assertions below are load-bearing and none of them is about a number this command
produced:

* **cache-only is proved by detonation, not by reading the source.** Every model, comparison
  bank and kNN entry point is replaced with a function that raises, at its definition *and* at
  every module attribute that a call could resolve through, and the run still has to succeed.
  `test_the_detonators_reach_the_definition_and_every_binding_of_it` pins that site list, since
  a monkeypatch on a name the call path never looks up passes whatever the command does.
* **the published decile command is compared against its own old output**, file set and column
  schema both, because the two commands now share a loader, a scorer and a row shape -- and the
  one field that separates the shapes, `bucket_scheme`, is exactly what a leak would add to the
  older command's `per_scene.csv` without changing anything else about it.
* **a finished bundle is compared byte for byte after the refusal.** Exit status 2 alone cannot
  tell "refused" from "rewrote it and then failed".
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.scene_uncertainty.cli import main

from tests.scene_uncertainty.decile_test_utils import (
    mutate_decile_artifacts,
    write_decile_artifacts,
)


CORRUPTION_REPORT_FILES = (
    "candidate_metrics.csv",
    "confidence_actual_distance_deciles.png",
    "confidence_actual_distance_quintiles.png",
    "easy-report.md",
    "per_scene.csv",
    "persistence_actual_distance_deciles.png",
    "persistence_actual_distance_quintiles.png",
    "summary.json",
)
"""The eight artifacts a published bundle holds, sorted. Written out rather than imported from
`corruption_reporting.EXPECTED_FILES` so that a rename there fails a test about this command's
output instead of agreeing with itself."""

DECILE_REPORT_FILES = (
    "blur_curves.png",
    "confidence_decile_heatmap.png",
    "dynamic_vs_frozen.png",
    "easy-report.md",
    "padding_sensitivity.png",
    "per_scene.csv",
    "summary.json",
)
"""The published command's seven, copied from `test_decile_integration.py` for the same reason
that module spells them out: a shared constant would let both commands drift together."""

DECILE_PER_SCENE_COLUMNS = (
    "image_id", "severity", "source_partition", "membership_mode", "confidence_bin",
    "padding_mode", "selected_count", "clean_overlap", "signal", "score_scope",
    "aggregation", "score",
)
"""The twelve columns README:299-302 documents for `analyze-confidence-deciles`, in order.

`summary_frame` builds them from first-seen row-key order, so an extra field on a scored row
becomes an extra column here with nothing else to notice it -- and `score_selection` now takes
an optional `bucket_scheme` that this command must never pass."""

SEVERITY_COUNT = 6
IMAGE_COUNT = 1
SCORE_ROWS_PER_IMAGE = 3060
"""34 selections x 15 signal/summary rows x 6 severities, as `corruption_analysis` derives it."""

CORRUPTION_INPUT_ARTIFACT_TYPE = "scene_corruption_sensitivity_inputs"
DECILE_INPUT_ARTIFACT_TYPE = "confidence_decile_scene_uncertainty"
"""What each command records as the type of the join it read. Spelled here rather than imported
so that the two commands cannot be relabelled into each other by one edit."""


def _analyze(artifacts: dict[str, Path], output: Path) -> int:
    return main([
        "analyze-corruption-sensitivity",
        "--cache", str(artifacts["cache"]),
        "--results", str(artifacts["results"]),
        "--output", str(output),
    ])


def _analyze_deciles(artifacts: dict[str, Path], output: Path) -> int:
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


def test_the_command_publishes_every_corruption_sensitivity_artifact(tmp_path: Path, capsys):
    """One command, two existing artifacts in, eight files out, exit status 0.

    The content assertions are each aimed at one hand-off the wrapper makes and nothing else
    can make for it:

    * `run` proves `diagnostics` reached the reporter with the loaded run metadata inside it,
      and the three fields checked are the ones that say *which* artifacts were read -- the
      tuning partition, all six severities, and the images the analysis actually walked;
    * both `bucket_scheme` values in `candidate_metrics.csv` prove the two resolutions were
      analysed and summarised as separate candidates, which is the whole comparison this
      command exists to make; a wrapper that scored one scheme would publish eight perfectly
      well-formed files;
    * the reported row count is compared against `per_scene.csv`, the table the summary
      describes, so the count on stderr cannot be some other length that happens to be handy,
      and against the count `corruption_analysis` documents for one image.
    """
    artifacts = write_decile_artifacts(tmp_path)
    output = tmp_path / "corruption"

    assert _analyze(artifacts, output) == 0

    assert sorted(path.name for path in output.iterdir()) == list(CORRUPTION_REPORT_FILES)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["run"]["artifact_type"] == CORRUPTION_INPUT_ARTIFACT_TYPE
    assert summary["run"]["source_partition"] == "tuning"
    assert summary["run"]["severities"] == list(range(SEVERITY_COUNT))
    assert summary["run"]["image_count"] == IMAGE_COUNT

    candidates = pd.read_csv(output / "candidate_metrics.csv")
    assert sorted(set(candidates["bucket_scheme"])) == ["decile", "quintile"]

    scored_rows = len(pd.read_csv(output / "per_scene.csv"))
    assert scored_rows == SCORE_ROWS_PER_IMAGE * IMAGE_COUNT
    assert _stderr_line(capsys) == (
        f"analyze-corruption-sensitivity: summarized {scored_rows} score rows into {output}"
    )


# --------------------------------------------------------------------------------------
# cache-only
# --------------------------------------------------------------------------------------


EXPENSIVE_ENTRY_POINTS = {
    "runtime": ("load_frozen_detector",),
    "extractor": ("ClassificationPersistenceExtractor",),
    "dataset": ("make_coco_loader",),
    "bank": ("deterministic_reservoir", "make_coverage_bank", "streaming_coverage_bank"),
    "knn": ("chunked_knn_distances", "mean_knn_distance", "fit_clean_distance_scale"),
    "evaluate": ("compute_query_distances", "score_cached_record"),
    "normalization": ("fit_normalizer", "transform_vectors"),
}
"""Every way into a detector forward pass, a comparison bank or a kNN search, keyed by the
module that defines it.

`normalization` is in here for the reason `knn` is: this command reads the layer score scales
the result artifact already carries, and re-fitting or re-applying them would relabel every
score with a scale the saved distances were never divided by.
"""

CANDIDATE_READERS = (
    "pipeline",
    "corruption_analysis", "corruption_reporting", "corruption_metrics", "corruption_plots",
    "decile_analysis", "decile_scoring", "decile_reporting", "confidence_deciles",
    "artifacts", "blur", "metrics", "reporting",
)
"""The modules scanned for a `from`-imported copy of one of the names above: this command's own
call path, and the rest of the package's analysis and reporting layer.

A `from`-import copies the function object into the importing module's globals, so patching the
definition alone leaves that copy live -- `pipeline` holds ten of them -- while patching
`pipeline` alone leaves a late `knn.mean_knn_distance(...)` lookup live. Both halves are
patched. The list is explicit rather than a walk of `sys.modules` because a walk would cover a
different set depending on what an earlier test in the session happened to import."""

DETONATION_SITES = (
    "runtime.load_frozen_detector",
    "pipeline.load_frozen_detector",
    "extractor.ClassificationPersistenceExtractor",
    "pipeline.ClassificationPersistenceExtractor",
    "dataset.make_coco_loader",
    "pipeline.make_coco_loader",
    "bank.deterministic_reservoir",
    "pipeline.deterministic_reservoir",
    "bank.make_coverage_bank",
    "bank.streaming_coverage_bank",
    "pipeline.streaming_coverage_bank",
    "knn.chunked_knn_distances",
    "knn.mean_knn_distance",
    "knn.fit_clean_distance_scale",
    "pipeline.fit_clean_distance_scale",
    "evaluate.compute_query_distances",
    "pipeline.compute_query_distances",
    "evaluate.score_cached_record",
    "pipeline.score_cached_record",
    "normalization.fit_normalizer",
    "pipeline.fit_normalizer",
    "normalization.transform_vectors",
    "pipeline.transform_vectors",
)
"""The module attributes the detonation below replaces, in the order `_detonation_sites` finds
them. Pinned as a literal so that the coverage is visible from the test rather than derived by
the same code that installs it."""


def _module(name: str):
    return importlib.import_module(f"src.scene_uncertainty.{name}")


def _detonation_sites() -> list[str]:
    """Each entry point's definition, plus every module global that still holds that object."""
    sites = []
    for home_name, names in EXPENSIVE_ENTRY_POINTS.items():
        home = _module(home_name)
        for name in names:
            entry_point = getattr(home, name)
            sites.append(f"{home_name}.{name}")
            sites.extend(
                f"{reader}.{name}" for reader in CANDIDATE_READERS
                if getattr(_module(reader), name, None) is entry_point
            )
    return sites


def test_the_detonators_reach_the_definition_and_every_binding_of_it():
    """What makes the detonation below a test rather than a decoration.

    A monkeypatch on a name the call path never resolves passes no matter what the command
    does, and a `from`-imported copy is exactly such a name: patching `knn.mean_knn_distance`
    does nothing to `pipeline.mean_knn_distance`, and vice versa. So the site list is compared
    whole. It fails if an entry point stops being defined where this module says it is, and it
    fails if a module on the call path starts or stops importing one -- either of which changes
    what the detonation covers without changing whether it passes.
    """
    assert _detonation_sites() == list(DETONATION_SITES)


def test_no_model_bank_or_knn_entry_point_is_reachable_from_the_new_command(
    tmp_path: Path, monkeypatch,
):
    """The command's reason to exist, checked instead of written down.

    This command is minutes on a CPU because it reads artifacts someone else spent GPU hours
    producing. A rewrite that re-extracted the fingerprints or re-searched the bank would still
    publish eight plausible files and every other assertion in this module would still hold --
    and it would publish *different* numbers, since the comparison between the two bucket
    schemes rests on both reading one set of saved distances.

    The limit, stated because an unstated limit is a false claim: this fires on reaches
    resolved through a module global. A local `import`, an `importlib.import_module`, or raw
    `torch.cdist(...).topk(...)` reaches no patched name and passes straight through. What it
    does cover that no scan of the wrapper's source can is an indirect reach, made several
    frames down by a helper the wrapper called.
    """
    def detonator(site: str):
        def explode(*args, **kwargs):
            raise AssertionError(f"analyze-corruption-sensitivity called {site}")
        return explode

    for site in _detonation_sites():
        module_name, _, name = site.partition(".")
        monkeypatch.setattr(_module(module_name), name, detonator(site))

    artifacts = write_decile_artifacts(tmp_path)
    output = tmp_path / "corruption"

    assert _analyze(artifacts, output) == 0

    assert sorted(path.name for path in output.iterdir()) == list(CORRUPTION_REPORT_FILES)


# --------------------------------------------------------------------------------------
# the published command is not disturbed
# --------------------------------------------------------------------------------------


def test_the_published_decile_command_still_writes_its_own_seven_files(tmp_path: Path, capsys):
    """Both commands over one artifact pair, and the older one has to be untouched.

    They now share a loader, a scorer and a row shape, so the ways the new one could reach into
    the old one's output are not hypothetical. The two leaks checked here are the two that would
    show up nowhere else: an eighth file in the decile bundle, and a `bucket_scheme` column in
    its `per_scene.csv` -- which `summary_frame` would add silently, because it builds its
    columns from whatever keys the scored rows carry.

    `artifact_type` separates the two runs at the other end. Both commands load through
    `_load_scene_query_inputs` and differ only in the label they ask it to record, so it is the
    one field that says which experiment a directory is, and a wrapper that called the wrong
    loader would otherwise be invisible here.
    """
    artifacts = write_decile_artifacts(tmp_path)
    deciles = tmp_path / "deciles"
    corruption = tmp_path / "corruption"

    assert _analyze_deciles(artifacts, deciles) == 0
    assert _analyze(artifacts, corruption) == 0

    assert sorted(path.name for path in deciles.iterdir()) == list(DECILE_REPORT_FILES)
    frame = pd.read_csv(deciles / "per_scene.csv")
    assert list(frame.columns) == list(DECILE_PER_SCENE_COLUMNS)
    summary = json.loads((deciles / "summary.json").read_text(encoding="utf-8"))
    assert summary["run_metadata"]["artifact_type"] == DECILE_INPUT_ARTIFACT_TYPE
    assert summary["run_metadata"]["source_partition"] == "tuning"

    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 2, lines
    assert lines[0].startswith("analyze-confidence-deciles: summarized ")
    assert lines[1] == (
        f"analyze-corruption-sensitivity: summarized {SCORE_ROWS_PER_IMAGE} score rows "
        f"into {corruption}"
    )


# --------------------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------------------


# One mutation per failure class the design's validation list names, chosen so that the two
# routes into the wrapper are both walked. Every entry but the last raises `DecileAnalysisError`
# from the loader; `zero_layer_scale` raises a *plain* `ValueError` from `decile_scoring`, at the
# point of division, several frames inside the analysis -- so a wrapper narrowed to
# `except DecileAnalysisError` would pass every row above it and fail only on that one.
UNPROVABLE_ARTIFACTS = [
    ("missing_cache_manifest", "missing feature-cache manifest"),
    ("forged_result_manifest", "does not match its own content address"),
    ("result_partition_all", "was built for the 'all' partition"),
    ("test_partition_row", "is in the 'test' partition"),
    ("drop_severity", "does not contain all six blur severities"),
    ("cache_layer_missing", "persistence layers"),
    ("cache_query_count", "query count"),
    ("everything_padded", "ten valid queries"),
    ("zero_layer_scale", "clean-distance scale must be positive"),
]


@pytest.mark.parametrize("mutation,expected", UNPROVABLE_ARTIFACTS)
def test_artifacts_that_cannot_be_proved_are_one_line_and_no_directory(
    tmp_path: Path, capsys, mutation, expected,
):
    """A refusal an operator caused reads as one line on stderr, not as a torch traceback.

    These are the likely mistakes in real use -- a mistyped `--cache`, last week's results CSV,
    a manifest from the wrong bank -- and they are exactly the failures the design requires this
    command to stop on with a clear error. `DecileAnalysisError` is a `ValueError` but not a
    `PipelineError`, and neither are the plain `ValueError`s that `confidence_deciles`,
    `decile_scoring` and `corruption_reporting` raise for the same class of problem, so without
    the `except ValueError` conversion in the wrapper none of them is caught by `cli.main` and
    every one of these inputs prints twenty frames of torch instead of a sentence.

    The prefix is compared and not merely the status, because a traceback also exits non-zero:
    the status alone cannot tell a converted refusal from an uncaught one. And nothing may be
    left on disk -- the analysis refuses before `write_corruption_report` stages anything, so a
    directory here would be a partial bundle a later reader could mistake for a result.
    """
    artifacts = write_decile_artifacts(tmp_path)
    mutate_decile_artifacts(artifacts, mutation)
    output = tmp_path / "corruption"

    assert _analyze(artifacts, output) == 2

    message = _stderr_line(capsys)
    assert message.startswith(
        "scene_uncertainty analyze-corruption-sensitivity: error: "
        "Cannot analyze corruption sensitivity: "
    )
    assert expected in message
    assert not output.exists()


def test_a_finished_bundle_is_refused_and_left_byte_identical(tmp_path: Path, capsys):
    """Rerunning onto a published bundle must not merge a second run into it.

    `write_corruption_report` refuses the directory itself rather than a marker file inside it,
    which is stricter than the decile command's `summary.json` guard and is what the eight-file
    atomic publication buys: a failed run leaves no directory at all, so a directory that exists
    is a finished bundle.

    Compared byte for byte rather than by file set, because exit status 2 and the same eight
    names are equally consistent with a rerun that republished the directory and then failed.
    """
    artifacts = write_decile_artifacts(tmp_path)
    output = tmp_path / "corruption"
    assert _analyze(artifacts, output) == 0
    published = {path.name: path.read_bytes() for path in output.iterdir()}
    capsys.readouterr()

    assert _analyze(artifacts, output) == 2

    assert _stderr_line(capsys) == (
        "scene_uncertainty analyze-corruption-sensitivity: error: "
        f"output already exists: {output}"
    )
    assert {path.name: path.read_bytes() for path in output.iterdir()} == published
