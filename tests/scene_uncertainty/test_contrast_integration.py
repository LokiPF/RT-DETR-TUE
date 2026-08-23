"""The command end to end, and the proof that it read nothing but the cache.

The detonation test is this module's reason to exist. Every number
`analyze-within-image-contrast` publishes was already computed by
`analyze-corruption-sensitivity`; a rewrite that re-extracted fingerprints or re-searched the
bank would still publish nine plausible files, every other assertion here would still hold, and
the numbers would be different ones. So the expensive entry points are replaced with charges
that raise, and the command is required to complete without touching any of them.
"""

from __future__ import annotations

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
"""The nine names in sorted order, written out rather than imported from the module under test."""


def _run(source: Path, output: Path) -> int:
    return main([
        "analyze-within-image-contrast",
        "--source", str(source),
        "--output", str(output),
    ])


@pytest.fixture(autouse=True)
def fast_bootstrap(monkeypatch):
    """Twenty draws instead of two thousand, for every test in this module.

    The real count is exercised by `test_contrast_controls`'s timing test and by the real run in
    Task 10. Here it would add minutes per test for no additional coverage: these tests are
    about what the command publishes and what it refuses, not about resampling.

    Patching the module constant works because `attach_controls` and `paired_macro_bootstrap`
    both take `samples=None` and resolve it at call time. A default of `BOOTSTRAP_SAMPLES`
    evaluated at definition time would ignore this patch silently.
    """
    monkeypatch.setattr("src.scene_uncertainty.contrast_controls.BOOTSTRAP_SAMPLES", 20)


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
    assert summary["bootstrap"]["samples"] == 20
    assert capsys.readouterr().err == ""

    # A full roster is the only run size the deployability gate can admit anything on, so this
    # is the one place the ranking is reachable at all -- every other fixture in the branch is
    # six images and ranks nothing. Without this, dropping the ranking call entirely publishes
    # a bundle whose report says "Nothing is ranked" and passes every other assertion here.
    assert summary["row_counts"]["ranked"] == len(summary["ranking"]) > 0
    assert all(item["signal"] == "persistence" for item in summary["ranking"])

    # The reference controls are summarised from the control rows, not from every row. Handing
    # the whole row set over produces one "control" per candidate -- a superset that includes
    # the contrasts themselves, so each candidate would be compared against a control that is
    # partly itself.
    assert summary["row_counts"]["reference_controls"] == 21
    assert {item["method"] for item in summary["reference_controls"]} == {"raw_reference"}
    assert len(summary["candidates"]) == 81


def test_no_model_bank_or_knn_entry_point_is_reachable(tmp_path: Path, monkeypatch):
    """The command's reason to exist, checked instead of written down."""
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


def test_the_detonation_covers_every_site_the_sibling_command_guards(tmp_path: Path):
    """The charges are shared with `test_corruption_integration`, so they cannot drift apart.

    A site added there because a new expensive entry point appeared is a site this command must
    also be shown not to reach; importing the tuple rather than restating it is what makes that
    automatic. Twenty-three of them, and a list that silently shrank would weaken both tests at
    once.
    """
    assert len(DETONATION_SITES) == 23
    assert len(set(DETONATION_SITES)) == len(DETONATION_SITES)
    for site in DETONATION_SITES:
        module_name, _, name = site.partition(".")
        assert hasattr(_module(module_name), name), site


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
    capsys.readouterr()

    assert _run(source, output) == 2

    after = {path.name: path.read_bytes() for path in output.iterdir()}
    assert before == after
    assert "exists" in capsys.readouterr().err.lower()


def test_the_source_bundle_is_left_byte_identical(tmp_path: Path):
    """Reporting-only, in the strongest sense available: the input is not written to at all."""
    source = _full_source(tmp_path)
    before = {path.name: path.read_bytes() for path in source.iterdir()}

    assert _run(source, tmp_path / "contrast") == 0

    after = {path.name: path.read_bytes() for path in source.iterdir()}
    assert before == after


def test_two_runs_over_one_source_publish_identical_bundles(tmp_path: Path):
    """Deterministic, and it is the bootstrap that makes this worth asserting.

    The resampling draws from a seeded generator, so two runs must agree to the byte. A run
    that seeded from the clock would publish two different sets of intervals from one input and
    nothing in a single run could show it.
    """
    source = _full_source(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"

    assert _run(source, first) == 0
    assert _run(source, second) == 0

    for name in CONTRAST_REPORT_FILES:
        if name.endswith(".png"):
            continue  # PNG bytes carry a creation date; the numbers behind them are compared
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


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


def test_any_value_error_from_the_analysis_becomes_one_line_and_not_a_traceback(
    tmp_path: Path, capsys, monkeypatch
):
    """`except ValueError`, and not `except ContrastInputError`.

    No source file can demonstrate the difference. Every operator-caused problem is refused by
    `load_contrast_inputs`, which raises `ContrastInputError`, and the plain `ValueError`s that
    `contrast_scores` raises for a negative or non-finite distance are unreachable from here
    because that same loader has already refused both -- so the narrower clause catches
    everything a real run can produce. The plan expected a negative-distance source to
    demonstrate it; that source cannot exist, which is a property of the layering worth
    recording rather than a gap to fill with a fixture that lies.

    What can be bound is the handler's contract: any `ValueError` escaping the analysis becomes
    one line and exit 2. Patching one step to raise a bare `ValueError` is the honest way to
    ask that, and it is what the narrower clause fails.
    """
    import src.scene_uncertainty.pipeline as pipeline

    def explode(*args, **kwargs):
        raise ValueError("a plain value error from inside the analysis")

    monkeypatch.setattr(pipeline, "build_contrast_rows", explode)
    output = tmp_path / "contrast"

    assert _run(_full_source(tmp_path), output) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(captured.err.strip().splitlines()) == 1
    assert "a plain value error from inside the analysis" in captured.err
    assert not output.exists()


def test_a_bug_in_this_package_is_not_disguised_as_an_operator_error(
    tmp_path: Path, monkeypatch
):
    """The mirror of the test above, and the reason the clause is `ValueError` and not `Exception`.

    A staged bundle with the wrong file set is a bug here, not an argument the operator got
    wrong. It raises `RuntimeError`, which passes through `cli.main` untouched and looks like
    what it is.
    """
    import src.scene_uncertainty.contrast_reporting as reporting

    real = reporting.write_contrast_plots

    def plots_plus_a_stray(directory, **kwargs):
        spans = real(directory, **kwargs)
        (Path(directory) / "stray.png").write_text("not a declared file")
        return spans

    monkeypatch.setattr(reporting, "write_contrast_plots", plots_plus_a_stray)

    with pytest.raises(RuntimeError, match="stray.png"):
        _run(_full_source(tmp_path), tmp_path / "contrast")
