"""The published bundle: nine files, or nothing at all.

Two things this file is careful about, both of them lessons this branch paid for earlier.

A constant compared against itself binds nothing. `EXPECTED_FILES` and `SECTION_TITLES` are the
two constants a mutation would reach for here, and a test that reads either on both sides of its
own comparison passes on any value whatever -- which is how a swapped `PLOT_FILENAMES` and a
`[0, 1]` `EMPTY_SPAN` both survived a full suite in Task 7. So both are pinned against literals
written out here, and the cost -- renaming a file means editing two places -- is the point.

A fixture that cannot tell two things apart cannot bind either. `figure_spans` is recorded rather
than recomputed, and on any real bundle the recorded and the recomputed value are the same
number, so the only fixture that can tell them apart is one carrying a span the figures did not
draw. `test_the_summary_records_the_spans_it_was_handed` passes an absurd one for that reason.
"""

import csv
import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.scene_uncertainty.contrast_analysis import (
    DEPLOYABLE_SIGNAL,
    RESIDUAL_METHOD,
    build_anchor_diagnostics,
    build_contrast_rows,
    summarize_contrast_candidates,
)
from src.scene_uncertainty.contrast_controls import (
    BOOTSTRAP_PERCENTILES,
    FULL_TUNING_IMAGE_COUNT,
    attach_controls,
    rank_contrast_candidates,
    reference_control_rows,
)
from src.scene_uncertainty.contrast_inputs import ARMS, load_contrast_inputs
from src.scene_uncertainty.contrast_plots import PLOT_FILENAMES, write_contrast_plots
from src.scene_uncertainty.contrast_reporting import (
    EXPECTED_FILES,
    SECTION_TITLES,
    build_contrast_summary,
    render_contrast_report,
    write_contrast_report,
)

from tests.scene_uncertainty.contrast_test_utils import write_source_bundle

IMAGES = 6

BUNDLE_FILES = (
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
"""The nine names, written out rather than read from the module they are asserted about."""

DECLARED_SECTIONS = (
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
"""The nine questions, in the spec's order, written out for the same reason."""

WRONG_SPANS = {
    "anchor": [-9999.0, -9998.0],
    "relationship": [-9997.0, -9996.0],
    "contrast": [-9995.0, -9994.0],
    "auroc": [-9993.0, -9992.0],
}
"""Spans no figure in this bundle could have drawn, for the one test that needs them.

Every number here is negative and enormous, and the AUROC entry is outside 0 to 1. A summary
that recomputed its spans instead of recording the ones it was handed cannot return any of
them, and a summary that carries them proves the recording is a recording.
"""


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


@pytest.fixture
def ranked_inputs(bundle_inputs):
    """The same bundle with the roster gate lifted, so the winner paths are reached.

    Six images is what makes this fixture cheap and what makes nothing in it deployable, and
    every section that reports a winner would otherwise be exercised only in its empty form.
    Raising `expected_image_count` to the declared roster is the same device
    `test_contrast_controls.test_a_full_roster_of_the_same_candidates_is_deployable` uses: the
    summaries are unchanged, so the gate is the only thing that moves.
    """
    for candidate in bundle_inputs["candidates"]:
        candidate["expected_image_count"] = FULL_TUNING_IMAGE_COUNT
    bundle_inputs["ranking"] = rank_contrast_candidates(bundle_inputs["candidates"])
    assert bundle_inputs["ranking"], "the lifted gate must produce a winner"
    return bundle_inputs


def _summary_for(tmp_path, bundle_inputs, figure_spans=None):
    """A summary built the way the writer builds one, with the real spans unless told otherwise."""
    if figure_spans is None:
        directory = tmp_path / "figures"
        directory.mkdir()
        figure_spans = write_contrast_plots(
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
        ranking=bundle_inputs["ranking"], figure_spans=figure_spans,
    )


def _read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


# --- the bundle ----------------------------------------------------------------------------------


def test_the_bundle_holds_exactly_nine_declared_files(tmp_path, bundle_inputs):
    output = tmp_path / "report"
    write_contrast_report(output, **bundle_inputs)

    assert sorted(path.name for path in output.iterdir()) == sorted(BUNDLE_FILES)
    # The constant itself, against the literal rather than against what was written. Both
    # halves are needed: the line above alone passes if `EXPECTED_FILES` and the writer drift
    # together, and this line alone passes if the writer produced something else entirely.
    assert sorted(EXPECTED_FILES) == sorted(BUNDLE_FILES)
    assert len(EXPECTED_FILES) == 9
    # The four figures are named once, in `contrast_plots`. A bundle that declared a fifth name
    # or renamed one of the four would publish a report describing a picture that is not there.
    assert set(PLOT_FILENAMES.values()) <= set(EXPECTED_FILES)


def test_an_existing_output_directory_is_refused_before_anything_is_computed(
    tmp_path, bundle_inputs
):
    output = tmp_path / "report"
    output.mkdir()

    with pytest.raises(FileExistsError):
        write_contrast_report(output, **bundle_inputs)

    assert list(output.iterdir()) == []
    # And no staging directory beside it: a refusal that has already built the bundle has not
    # refused before computing anything, it has merely declined to publish.
    assert sorted(path.name for path in tmp_path.iterdir()) == ["report", "source"]


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


def test_an_interrupt_leaves_neither_output_nor_staging(tmp_path, bundle_inputs, monkeypatch):
    """`BaseException`, not `Exception`, and this is the test that tells the two apart.

    A `KeyboardInterrupt` part-way through leaves exactly the half-written staging directory a
    `ValueError` does, and it is the likelier of the two on a run an operator is watching. A
    cleanup that catches `Exception` passes every other test in this file.
    """
    import src.scene_uncertainty.contrast_reporting as reporting

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(reporting, "write_contrast_plots", interrupt)
    output = tmp_path / "report"

    with pytest.raises(KeyboardInterrupt):
        write_contrast_report(output, **bundle_inputs)

    assert not output.exists()
    assert list(tmp_path.iterdir()) == [tmp_path / "source"]


def test_an_existing_output_is_refused_before_the_figures_are_drawn(
    tmp_path, bundle_inputs, monkeypatch
):
    """"Refused before anything is computed" needs something that would notice being computed.

    A refusal placed anywhere in the writer -- even just before the rename, after every table
    and figure has been built into staging -- passes an assertion that only checks the output
    directory is still empty and no staging survives, because the cleanup removes staging on
    the way out either way. Patching the figure writer to explode makes the ordering visible:
    if the refusal comes first, this raises `FileExistsError`; if anything is computed first,
    it raises `RuntimeError` instead.
    """
    import src.scene_uncertainty.contrast_reporting as reporting

    def explode(*args, **kwargs):
        raise RuntimeError("the figures were drawn")

    monkeypatch.setattr(reporting, "write_contrast_plots", explode)
    output = tmp_path / "report"
    output.mkdir()

    with pytest.raises(FileExistsError):
        write_contrast_report(output, **bundle_inputs)


def test_a_bundle_with_the_wrong_file_set_is_not_published(
    tmp_path, bundle_inputs, monkeypatch
):
    """The check that runs after publication has already published.

    Nothing in a passing run can produce a wrong file set, so the only way to bind the check is
    to make one. A tenth file is the easier direction to stage and the same guard catches
    both: a missing figure is a bundle whose report describes a picture that is not there, and
    an extra file is one nothing in `summary.json` accounts for.
    """
    import src.scene_uncertainty.contrast_reporting as reporting

    real = reporting.write_contrast_plots

    def plots_plus_a_stray(directory, **kwargs):
        spans = real(directory, **kwargs)
        (Path(directory) / "stray.png").write_text("not a declared file")
        return spans

    monkeypatch.setattr(reporting, "write_contrast_plots", plots_plus_a_stray)
    output = tmp_path / "report"

    with pytest.raises(RuntimeError, match="stray.png"):
        write_contrast_report(output, **bundle_inputs)

    assert not output.exists()
    assert list(tmp_path.iterdir()) == [tmp_path / "source"]


def test_a_non_finite_number_is_refused_rather_than_written_as_a_bare_token(
    tmp_path, bundle_inputs
):
    """`allow_nan=False`, and a fixture carrying a `nan` is the only thing that can see it.

    Left to its default, `json.dumps` writes the bare token `NaN`, which is valid for Python
    and for no other JSON reader. The file would then fail somewhere far from this run, in
    whatever read it next, with nothing pointing back here.
    """
    bundle_inputs["candidates"][0]["oriented_adjacent_consistency"] = float("nan")
    output = tmp_path / "report"

    with pytest.raises(ValueError):
        write_contrast_report(output, **bundle_inputs)

    assert not output.exists()
    assert list(tmp_path.iterdir()) == [tmp_path / "source"]


def test_two_rows_disagreeing_about_a_fold_are_refused(tmp_path, bundle_inputs):
    """One image, one fold. Two rows disagreeing is not one cross-fitting partition.

    Taking the last row's answer would publish a fold map that describes neither, and the
    residuals built under the two assignments are not comparable with each other.
    """
    rows = [dict(row) for row in bundle_inputs["rows"]]
    victim = rows[0]["image_id"]
    for row in rows:
        if row["image_id"] == victim:
            row["fold"] = (row["fold"] + 1) % 5
            break
    bundle_inputs["rows"] = rows

    with pytest.raises(RuntimeError, match=f"image {victim}"):
        _summary_for(tmp_path, bundle_inputs)


def test_two_values_claiming_one_column_are_refused(tmp_path, bundle_inputs, monkeypatch):
    """A collision publishes one measurement under another's heading, and loses the other.

    Nothing collides today -- the drift fields are `signed_q25` where the spread fields are
    `q25` -- but "nothing collides today" is a property of the current field names and not of
    the flattening, and the next field added is what tests it.
    """
    import src.scene_uncertainty.contrast_reporting as reporting

    collided = dict(reporting.SEVERITY_MAP_COLUMNS)
    collided["twin_auroc_by_severity"] = collided["auroc_by_severity"]
    monkeypatch.setattr(reporting, "SEVERITY_MAP_COLUMNS", collided)
    output = tmp_path / "report"

    with pytest.raises(RuntimeError, match="auroc_severity_1"):
        write_contrast_report(output, **bundle_inputs)

    assert not output.exists()


def test_the_candidate_named_strongest_is_the_one_with_the_highest_macro_auroc(
    tmp_path, bundle_inputs
):
    """Three sections name a strongest candidate, and none of them said which way it was picked.

    Reversing the comparison names the worst candidate in the run as its headline result, in a
    report whose every other number stays correct.
    """
    summary = _summary_for(tmp_path, bundle_inputs)
    text = render_contrast_report(summary)

    measured = [
        candidate for candidate in summary["candidates"]
        if candidate["signal"] == DEPLOYABLE_SIGNAL
        and candidate.get("macro_auroc") is not None
    ]
    best = max(measured, key=lambda item: item["macro_auroc"])
    worst = min(measured, key=lambda item: item["macro_auroc"])
    assert best["macro_auroc"] > worst["macro_auroc"], "the fixture must separate the two"

    expected = f"`{best['arm']}` / {best['aggregation']} / {best['method']}"
    unwanted = f"`{worst['arm']}` / {worst['aggregation']} / {worst['method']}"
    for index in (2, 4):
        section = text.split(DECLARED_SECTIONS[index], 1)[1].split("\n## ", 1)[0]
        assert expected in section, DECLARED_SECTIONS[index]
        assert unwanted not in section, DECLARED_SECTIONS[index]
    assert f"macro AUROC {best['macro_auroc']:.3f}" in text


def test_the_published_directory_is_readable_by_more_than_its_owner(tmp_path, bundle_inputs):
    """`mkdtemp` makes a `0o700` directory and `os.replace` carries the mode onto the bundle.

    The one property of a temporary directory the published result must not inherit: the files
    inside it are whatever the umask made them, and an operator who is not the owner would be
    left unable to enter the directory holding them.
    """
    reference = tmp_path / "reference"
    reference.mkdir()
    output = tmp_path / "report"
    write_contrast_report(output, **bundle_inputs)

    assert output.stat().st_mode & 0o777 == reference.stat().st_mode & 0o777
    assert output.stat().st_mode & 0o700 != 0


# --- the tables ----------------------------------------------------------------------------------


def test_candidate_metrics_csv_holds_no_dictionary_cells(tmp_path, bundle_inputs):
    output = tmp_path / "report"
    write_contrast_report(output, **bundle_inputs)
    written = _read_csv(output / "candidate_metrics.csv")

    assert len(written) == len(bundle_inputs["candidates"])
    for row in written:
        for column, value in row.items():
            assert not value.startswith("{"), column
    for column in (
        "auroc_severity_1", "twin_auroc_severity_5", "declared_before_data",
        "confidence_redundant", "arm_family", "score_median_severity_0",
        "responsive_control_bootstrap_verdict", "twin_bootstrap_macro_difference",
        "reference_control_auroc_difference_severity_3",
    ):
        assert column in written[0], column
    # The dictionaries are expanded, not merely absent. A projection that dropped them without
    # expanding them passes every assertion above and loses five numbers per candidate.
    source = {
        (item["arm"], item["signal"], item["aggregation"], item["method"]): item
        for item in bundle_inputs["candidates"]
    }
    for row in written:
        candidate = source[
            (row["arm"], row["signal"], row["aggregation"], row["method"])
        ]
        if candidate["auroc_by_severity"] is not None:
            assert float(row["auroc_severity_1"]) == pytest.approx(
                candidate["auroc_by_severity"][1]
            )
        else:
            assert row["auroc_severity_1"] == ""
        assert float(row["score_median_severity_0"]) == pytest.approx(
            candidate["severity_statistics"][0]["median"]
        )


def test_an_undeclared_nested_field_is_refused_rather_than_dropped(tmp_path, bundle_inputs):
    """"Drop every dictionary" and "drop every dictionary I decided to drop" are different rules.

    The spec's projection rule is the first; this is the second, and it is deliberately the
    stricter of the two. A nested field added to a candidate later would otherwise vanish from
    the CSV with nothing anywhere saying so -- the reader sees a complete-looking table, and the
    only trace of the missing measurement is its absence.
    """
    bundle_inputs["candidates"][0]["something_new_by_severity"] = {1: 0.5}
    output = tmp_path / "report"

    with pytest.raises(RuntimeError, match="something_new_by_severity"):
        write_contrast_report(output, **bundle_inputs)

    assert not output.exists()


def test_per_scene_contrasts_has_one_row_per_contrast_row(tmp_path, bundle_inputs):
    output = tmp_path / "report"
    write_contrast_report(output, **bundle_inputs)
    written = _read_csv(output / "per_scene_contrasts.csv")

    assert len(written) == len(bundle_inputs["rows"])
    assert written
    for column in ("image_id", "severity", "arm", "reference", "responsive", "score",
                   "fold", "fit_slope", "fit_offset", "arm_family", "declared_before_data"):
        assert column in written[0], column


def test_anchor_diagnostics_csv_covers_all_twenty_one_configurations(tmp_path, bundle_inputs):
    output = tmp_path / "report"
    write_contrast_report(output, **bundle_inputs)
    written = _read_csv(output / "anchor_diagnostics.csv")

    assert len({(row["arm"], row["signal"], row["aggregation"]) for row in written}) == 21
    assert len(written) == 21
    for column in ("stability_to_spread_severity_3", "arm_family",
                   "median_absolute_drift_severity_1", "relationship_spearman",
                   "relationship_predictive"):
        assert column in written[0], column
    for row in written:
        for column, value in row.items():
            assert not value.startswith("{"), column
    # Values, not only column names. A flattening that read the wrong severity produces a
    # complete table of numbers that are each somebody else's.
    source = {
        (item["arm"], item["signal"], item["aggregation"]): item
        for item in bundle_inputs["diagnostics"]
    }
    for row in written:
        entry = source[(row["arm"], row["signal"], row["aggregation"])]
        for severity in (1, 3, 5):
            assert float(row[f"stability_to_spread_severity_{severity}"]) == pytest.approx(
                entry["spread"][severity]["stability_to_spread"]
            )
        # Severity zero has no ratio and its cell is empty rather than 0.0. Its drift is
        # identically zero by definition, so a ratio there would be a guaranteed 0.0 that reads
        # like the best score in the column.
        assert row["stability_to_spread_severity_0"] == ""
        assert entry["spread"][0]["stability_to_spread"] is None


# --- the summary ---------------------------------------------------------------------------------


def test_summary_records_the_spans_the_figures_were_drawn_on(tmp_path, bundle_inputs):
    output = tmp_path / "report"
    write_contrast_report(output, **bundle_inputs)
    summary = json.loads((output / "summary.json").read_text())

    assert set(summary["figure_spans"]) == {"anchor", "relationship", "contrast", "auroc"}
    assert summary["figure_spans"]["auroc"] == [0.0, 1.0]
    # `figure_spans`, not `axis_limits`. Three of the four are the range of data an autoscaled
    # panel drew and are not limits imposed on anything; the spec names the key for that reason
    # and the report is forbidden from calling the three of them limits.
    assert "axis_limits" not in summary


def test_the_summary_records_the_spans_it_was_handed(tmp_path, bundle_inputs):
    """Recorded verbatim, never recomputed -- and the fixture is what makes the two separable.

    On any real bundle a recomputation agrees with the recording to the last bit, so a summary
    that recomputed would pass every other assertion in this file. `WRONG_SPANS` is a set of
    numbers no figure here could have drawn: a recomputing summary cannot return them.
    """
    summary = _summary_for(tmp_path, bundle_inputs, figure_spans=WRONG_SPANS)

    assert summary["figure_spans"] == WRONG_SPANS
    assert summary["figure_spans"]["auroc"] == [-9993.0, -9992.0]


def test_summary_records_the_arm_table_with_its_provenance(tmp_path, bundle_inputs):
    output = tmp_path / "report"
    write_contrast_report(output, **bundle_inputs)
    summary = json.loads((output / "summary.json").read_text())

    arms = {arm["name"]: arm for arm in summary["arms"]}
    assert len(arms) == 4
    assert arms["decile_00_10__50_60"]["declared_before_data"] is True
    assert arms["decile_90_100__50_60__combined"]["declared_before_data"] is False
    assert arms["decile_90_100__50_60__combined"]["family"] == "differential"
    assert arms["decile_00_10__50_60"]["family"] == "anchored"
    # Every field of every arm, against the table itself: the two differential arms differ only
    # in `score_scope`, and a summary that dropped that field would describe them identically.
    for arm in ARMS:
        recorded = arms[arm.name]
        assert recorded["pair_name"] == arm.pair_name
        assert recorded["score_scope"] == arm.score_scope
        assert recorded["bucket_scheme"] == arm.bucket_scheme
        assert recorded["reference_bin"] == arm.reference_bin
        assert recorded["responsive_bin"] == arm.responsive_bin
    assert arms["decile_90_100__50_60"]["score_scope"] == "layer_2"
    assert arms["decile_90_100__50_60__combined"]["score_scope"] == "combined"


def test_summary_records_twenty_one_final_fits(tmp_path, bundle_inputs):
    output = tmp_path / "report"
    write_contrast_report(output, **bundle_inputs)
    summary = json.loads((output / "summary.json").read_text())

    assert len(summary["fits"]) == 21
    entry = summary["fits"][0]
    for field in ("arm", "signal", "aggregation", "final_line", "fold_lines",
                  "residual_available", "unavailable_reason"):
        assert field in entry, field
    assert len({(item["arm"], item["signal"], item["aggregation"]) for item in summary["fits"]}) == 21


def test_summary_records_the_ranking_as_keys_and_not_as_second_copies(
    tmp_path, ranked_inputs
):
    """One candidate, one set of numbers. A ranking serialised as whole candidate dictionaries
    gives every winner a second copy in the same file, and two copies can disagree."""
    summary = _summary_for(tmp_path, ranked_inputs)

    assert summary["ranking"]
    assert len(summary["ranking"]) == len(ranked_inputs["ranking"])
    for entry in summary["ranking"]:
        assert list(entry) == ["arm", "signal", "aggregation", "method"]
    keys = {tuple(entry.values()) for entry in summary["ranking"]}
    assert keys <= {
        (item["arm"], item["signal"], item["aggregation"], item["method"])
        for item in summary["candidates"]
    }


def test_summary_records_the_bootstrap_settings_that_were_actually_run(tmp_path, bundle_inputs):
    """20 samples, because that is what this fixture asked for -- not 2000, the constant.

    A summary reporting the module default describes a run nobody performed whenever a caller
    passes `samples`, which Task 9's integration tests do precisely so they finish.
    """
    summary = _summary_for(tmp_path, bundle_inputs)

    assert summary["bootstrap"]["samples"] == 20
    assert summary["bootstrap"]["percentiles"] == list(BOOTSTRAP_PERCENTILES)
    assert summary["bootstrap"]["seed"] is not None


def test_summary_records_the_row_counts_of_every_table_it_published(tmp_path, bundle_inputs):
    output = tmp_path / "report"
    write_contrast_report(output, **bundle_inputs)
    summary = json.loads((output / "summary.json").read_text())

    assert summary["row_counts"] == {
        "per_scene_contrasts": len(_read_csv(output / "per_scene_contrasts.csv")),
        "anchor_diagnostics": len(_read_csv(output / "anchor_diagnostics.csv")),
        "candidate_metrics": len(_read_csv(output / "candidate_metrics.csv")),
        "reference_controls": len(bundle_inputs["controls"]),
        "ranked": len(bundle_inputs["ranking"]),
    }
    assert summary["row_counts"]["per_scene_contrasts"] == len(bundle_inputs["rows"])


def test_the_summary_is_json_serialisable_without_a_fallback(tmp_path, bundle_inputs):
    """No `default=str`. A summary that only serialises through a coercion hook is a summary
    holding a numpy scalar or a `nan`, and both reach a reader as something they are not."""
    summary = _summary_for(tmp_path, bundle_inputs)

    encoded = json.dumps(summary, allow_nan=False)  # raises on a numpy scalar or a nan
    assert json.loads(encoded)["figure_spans"]["auroc"] == [0.0, 1.0]
    # json.dumps writes the bare tokens NaN and Infinity, which are not valid JSON anywhere else
    assert "NaN" not in encoded
    assert "Infinity" not in encoded


# --- the report ----------------------------------------------------------------------------------


def test_the_report_leads_with_the_declared_sections(tmp_path, bundle_inputs):
    text = render_contrast_report(_summary_for(tmp_path, bundle_inputs))

    assert list(SECTION_TITLES) == list(DECLARED_SECTIONS)
    for title in DECLARED_SECTIONS:
        assert title in text, title
    # The order is part of the contract: the anchor questions come before the contrast ones,
    # and the two things the report may not be read as saying come last.
    positions = [text.index(title) for title in DECLARED_SECTIONS]
    assert positions == sorted(positions)


def test_every_section_renders_when_nothing_is_deployable(tmp_path, bundle_inputs):
    """A section omitted for want of a winner reads as a question that was never asked.

    This bundle is the empty case by construction -- six images against a declared roster of
    250 -- so this is the path the whole suite takes unless a test lifts the gate.
    """
    assert bundle_inputs["ranking"] == []
    text = render_contrast_report(_summary_for(tmp_path, bundle_inputs))

    for title in DECLARED_SECTIONS:
        assert title in text, title
        body = text.split(title, 1)[1].split("\n## ", 1)[0].strip()
        assert body, title
    assert "no candidate cleared" in text.lower()


def test_the_report_reports_control_comparisons_even_with_an_empty_ranking(
    tmp_path, bundle_inputs
):
    """Sections 3 to 5 read every candidate that produced a macro AUROC, not `ranking`.

    `ranking` names the winner; it is not the set of results. A report whose control
    comparisons vanished on a run where nothing cleared the roster gate would hide exactly the
    numbers a reader needs in order to see why nothing did.
    """
    assert bundle_inputs["ranking"] == []
    text = render_contrast_report(_summary_for(tmp_path, bundle_inputs))

    measured = [
        candidate for candidate in bundle_inputs["candidates"]
        if candidate["signal"] == DEPLOYABLE_SIGNAL
        and candidate.get("macro_auroc") is not None
    ]
    assert measured
    beat_both = text.split(DECLARED_SECTIONS[2], 1)[1].split("\n## ", 1)[0]
    assert str(len(measured)) in beat_both
    severities = text.split(DECLARED_SECTIONS[4], 1)[1].split("\n## ", 1)[0]
    assert "severity 1" in severities and "severity 5" in severities


def test_the_severity_section_gives_the_mild_blur_levels_first(tmp_path, ranked_inputs):
    """Severities 1 and 2 first, because that is the regime the completed analysis could not
    separate, and a macro gain driven by severities 4 and 5 alone repeats a result in hand."""
    text = render_contrast_report(_summary_for(tmp_path, ranked_inputs))
    section = text.split(DECLARED_SECTIONS[4], 1)[1].split("\n## ", 1)[0]

    positions = [section.index(f"severity {severity}") for severity in (1, 2, 3, 4, 5)]
    assert positions == sorted(positions)
    assert "could not separate" in section


def test_the_report_makes_no_probability_claim(tmp_path, bundle_inputs):
    output = tmp_path / "report"
    write_contrast_report(output, **bundle_inputs)
    text = (output / "easy-report.md").read_text().lower()

    for forbidden in ("probability of corruption", "calibrated", "% chance", "likelihood that"):
        assert forbidden not in text, forbidden
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
    assert "anchored" in text


def test_the_selection_debt_is_stated_whether_or_not_a_differential_arm_won(
    tmp_path, bundle_inputs
):
    """The anchored/differential distinction is a property of the design, not of the outcome.

    Stated on a run with no winner and on a run with one, because a sentence that appears only
    when a differential arm wins is a sentence a reader learns to read as a result.

    The ranked half is built here rather than through the `ranked_inputs` fixture: that fixture
    mutates `bundle_inputs` in place and returns it, so asking for both in one signature gives
    the same object under two names and the "empty" half would not be empty.
    """
    (tmp_path / "a").mkdir()
    empty = render_contrast_report(_summary_for(tmp_path / "a", bundle_inputs))
    assert bundle_inputs["ranking"] == []

    for candidate in bundle_inputs["candidates"]:
        candidate["expected_image_count"] = FULL_TUNING_IMAGE_COUNT
    bundle_inputs["ranking"] = rank_contrast_candidates(bundle_inputs["candidates"])
    assert bundle_inputs["ranking"]
    (tmp_path / "b").mkdir()
    ranked = render_contrast_report(_summary_for(tmp_path / "b", bundle_inputs))

    for text in (empty, ranked):
        assert "chosen after reading tuning" in text
        assert DECLARED_SECTIONS[6] in text
    assert "Nothing is ranked" in empty
    assert "Nothing is ranked" not in ranked


def _make_both_hypotheses_pass(bundle_inputs):
    """Isolate one qualifying candidate per family from the shared measured bundle."""
    prepared = deepcopy(bundle_inputs)
    for candidate in prepared["candidates"]:
        if candidate["signal"] != DEPLOYABLE_SIGNAL:
            continue
        candidate["complete"] = False
        candidate["beats_both_inputs"] = False
        candidate["confidence_redundant"] = True
        candidate["responsive_control_bootstrap"]["low"] = -1.0
        candidate["reference_control_bootstrap"]["low"] = -1.0
        candidate["twin_bootstrap"]["low"] = -1.0

    for entry in prepared["diagnostics"]:
        if entry["signal"] != DEPLOYABLE_SIGNAL:
            continue
        for severity in range(1, 6):
            entry["spread"][severity]["stability_to_spread"] = 2.0
        entry["relationship"]["predictive"] = False

    anchored = next(
        candidate for candidate in prepared["candidates"]
        if candidate["arm"] == "decile_00_10__50_60"
        and candidate["signal"] == DEPLOYABLE_SIGNAL
        and candidate["aggregation"] == "mean"
        and candidate["method"] == "raw_gap"
    )
    anchored.update({
        "complete": True,
        "expected_image_count": FULL_TUNING_IMAGE_COUNT,
        "image_count": FULL_TUNING_IMAGE_COUNT,
        "beats_both_inputs": True,
        "confidence_redundant": False,
    })
    anchored["responsive_control_bootstrap"]["low"] = 0.01
    anchored["reference_control_bootstrap"]["low"] = 0.02
    anchored["responsive_control_auroc_difference_by_severity"][1] = 0.0
    anchored_diagnostic = next(
        entry for entry in prepared["diagnostics"]
        if entry["arm"] == anchored["arm"]
        and entry["signal"] == DEPLOYABLE_SIGNAL
        and entry["aggregation"] == anchored["aggregation"]
    )
    for severity in range(1, 6):
        anchored_diagnostic["spread"][severity]["stability_to_spread"] = 0.5

    differential = next(
        candidate for candidate in prepared["candidates"]
        if candidate["arm"] == "decile_90_100__50_60"
        and candidate["signal"] == DEPLOYABLE_SIGNAL
        and candidate["aggregation"] == "mean"
        and candidate["method"] == "raw_gap"
    )
    differential.update({
        "complete": True,
        "expected_image_count": FULL_TUNING_IMAGE_COUNT,
        "image_count": FULL_TUNING_IMAGE_COUNT,
        "beats_both_inputs": True,
        "confidence_redundant": False,
    })
    differential["twin_bootstrap"]["low"] = 0.01
    differential["auroc_by_severity"][1] = 0.539
    differential["auroc_by_severity"][2] = 0.571
    return prepared, anchored, differential


def test_summary_and_report_publish_the_two_composite_hypothesis_verdicts(
    tmp_path, bundle_inputs
):
    prepared, anchored, differential = _make_both_hypotheses_pass(bundle_inputs)
    summary = _summary_for(tmp_path, prepared, figure_spans=WRONG_SPANS)

    verdicts = summary["hypothesis_verdicts"]
    assert verdicts["anchored"]["supported_on_tuning"] is True
    assert verdicts["anchored"]["qualifying_candidates"] == [{
        "arm": anchored["arm"],
        "aggregation": anchored["aggregation"],
        "method": anchored["method"],
    }]
    assert verdicts["differential"]["carry_to_held_out"] is True
    assert verdicts["differential"]["qualifying_candidates"] == [{
        "arm": differential["arm"],
        "aggregation": differential["aggregation"],
        "method": differential["method"],
    }]

    section = render_contrast_report(summary).split(
        DECLARED_SECTIONS[6], 1
    )[1].split("\n## ", 1)[0]
    assert "Anchored verdict: supported on tuning" in section
    assert "Differential verdict: worth carrying to a held-out test" in section
    assert "0.538" in section and "0.570" in section


def test_the_report_names_a_redundant_candidate_as_adding_nothing(tmp_path, bundle_inputs):
    for candidate in bundle_inputs["candidates"]:
        if candidate["signal"] == DEPLOYABLE_SIGNAL:
            candidate["confidence_redundant"] = True
    output = tmp_path / "report"
    write_contrast_report(output, **bundle_inputs)
    text = (output / "easy-report.md").read_text()

    assert "adds nothing over the detector's own confidence" in text


def test_a_candidate_that_beats_its_twin_is_not_called_redundant(tmp_path, bundle_inputs):
    """The mirror. A report that printed the redundancy sentence unconditionally passes the
    test above on every run, including the ones where the control was cleared."""
    for candidate in bundle_inputs["candidates"]:
        if candidate["signal"] == DEPLOYABLE_SIGNAL:
            candidate["confidence_redundant"] = False
    output = tmp_path / "report"
    write_contrast_report(output, **bundle_inputs)
    text = (output / "easy-report.md").read_text()

    assert "adds nothing over the detector's own confidence" not in text


def _deployment_section(summary):
    text = render_contrast_report(summary)
    return text.split(DECLARED_SECTIONS[7], 1)[1].split("\n## ", 1)[0]


def test_the_deployment_section_prints_the_line_a_residual_winner_would_store(
    tmp_path, ranked_inputs
):
    """Slope and offset, from the arm that won -- and the cross-fitting caveat beside them."""
    ranked_inputs["ranking"] = [
        item for item in ranked_inputs["ranking"] if item["method"] == RESIDUAL_METHOD
    ]
    assert ranked_inputs["ranking"], "the fixture must rank at least one residual"
    summary = _summary_for(tmp_path, ranked_inputs)
    section = _deployment_section(summary)

    winner = summary["ranking"][0]
    fit = next(
        item for item in summary["fits"]
        if (item["arm"], item["signal"], item["aggregation"])
        == (winner["arm"], winner["signal"], winner["aggregation"])
    )
    assert fit["final_line"] is not None
    assert f"{fit['final_line'][0]:.3f}" in section
    assert f"{fit['final_line'][1]:.3f}" in section
    assert "would store" in section
    assert "cross-fitted" in section
    assert "uses no fitted line" not in section


def test_a_winner_that_uses_no_fitted_line_is_not_offered_one(tmp_path, ranked_inputs):
    """A clean line is published for every configuration, and only `clean_residual` uses one.

    The other three methods are functions of the two raw ranges alone. Printing the fitted
    slope and offset under "what a deployment would store" for one of them hands an operator a
    number their chosen method has nowhere to put, and implies the two are connected. The line
    is still reachable -- it is in `fits` -- but it is not this candidate's to store.
    """
    ranked_inputs["ranking"] = [
        item for item in ranked_inputs["ranking"] if item["method"] != RESIDUAL_METHOD
    ]
    assert ranked_inputs["ranking"]
    summary = _summary_for(tmp_path, ranked_inputs)
    section = _deployment_section(summary)

    winner = summary["ranking"][0]
    fit = next(
        item for item in summary["fits"]
        if (item["arm"], item["signal"], item["aggregation"])
        == (winner["arm"], winner["signal"], winner["aggregation"])
    )
    # The line exists; the section still must not offer it as this candidate's.
    assert fit["final_line"] is not None
    assert "uses no fitted line" in section
    assert f"{fit['final_line'][0]:.3f}" not in section
    # What it does store is named instead, so the section is not merely a refusal.
    assert "percentile ranges" in section
    assert winner["aggregation"] in section


def test_the_report_is_built_from_the_summary_and_nothing_else(tmp_path, bundle_inputs):
    """Two renders of one summary agree, and a summary read back from disk renders the same.

    Which is the whole claim `summary.json` beside the report rests on: every sentence carrying
    a number can be checked against the file next to it.
    """
    output = tmp_path / "report"
    write_contrast_report(output, **bundle_inputs)
    published = (output / "easy-report.md").read_text()
    reloaded = json.loads((output / "summary.json").read_text())

    assert render_contrast_report(reloaded) == published
    assert render_contrast_report(reloaded) == render_contrast_report(reloaded)
