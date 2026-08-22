"""Four figures, and the ways a complete-looking set of PNGs quietly stops being the experiment.

Every failure this file is written against produces four well-labelled files that a reader
would not question. `st_size > 0` sees none of them, so almost nothing here is asserted on a
file: the figures are built through the same private entry point `write_contrast_plots` renders
and saves, and what a test reads off an axis is what the file holds.

* **a grid sized to the data.** Twelve panels is the experiment -- four arms times three scene
  summaries -- and nine drawn panels in a nine-panel grid reads as a nine-cell experiment.
  `panel_plan` is asserted against the arm table with candidates *removed*, one dimension at a
  time and then all of them, because a plan derived from the candidates is right on a complete
  run and wrong on exactly the run that needs the empty panels;
* **a panel drawn from the wrong cell.** The shared fixture cannot see this. `default_score`
  takes no aggregation, so a run's three summaries are numerically identical and a figure that
  read `mean` into all three panels of a row would pass every assertion made on real rows. The
  synthetic candidates give every `(arm, summary, method)` cell its own level, so a transposed
  grid, a repeated column or an off-by-one panel is a numeric mismatch;
* **a reference and a responsive curve swapped.** They are the same quantity in the same units,
  so nothing about the drawing says which is which. The synthetic reference is the negative of
  its responsive, and the panel's line order is asserted;
* **an oriented curve.** A contrast that falls as blur rises is a finding. A falling synthetic
  candidate carries `orientation: -1`, and its panel must fall;
* **a twin drawn solid.** A dashed confidence twin against its solid candidate is the whole
  readability claim of the AUROC figure, and it is not visible in a file comparison against a
  re-render of the same code. It is read off the line styles instead;
* **an AUROC axis fitted to its data.** Two panels on two ranges cannot be compared, which is
  what the twelve-panel grid exists for. The fixture's AUROCs span neither 0 nor 1, so a
  recomputed range is a different pair of numbers and the assertion can tell them apart;
* **a severity nobody scored, drawn as a straight line across the hole.** `count == 0` is a
  fact about the run, and a panel that joins severity 1 to severity 3 through it publishes an
  interpolation as a measurement. It is a gap in the line, and the panel is still a panel;
* **a repair written into the candidate.** These dictionaries are the same objects the ranking,
  the CSV and the report hold, so a plotting-time default would be published by three writers
  that never asked for it. Nothing here may be mutated, and the whole input is compared against
  a deep copy of itself.

Two properties are asserted about the files themselves and cannot be asserted anywhere else:
that each name holds the figure its name promises, which no per-figure assertion can see, and
that the saved image is the declared DPI rather than matplotlib's default 100.

Two properties are *not* asserted here, and the report for this task says so rather than
implying coverage: that the figures are legible -- no overlapping text, a visible dashed twin,
bands wide enough to read -- and that the layout is pleasant. Those were checked by rendering
the twenty-image bundle and looking at the four PNGs. A test that only counts artists has not
looked at the figure.
"""

from __future__ import annotations

import copy
import json
import struct
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from matplotlib import pyplot as plt

from src.scene_uncertainty import contrast_plots as plots_module
from src.scene_uncertainty.contrast_analysis import (
    DEPLOYABLE_SIGNAL,
    build_contrast_rows,
    summarize_contrast_candidates,
)
from src.scene_uncertainty.contrast_controls import (
    CORRUPTED_SEVERITIES,
    REFERENCE_CONTROL_METHOD,
    RESPONSIVE_CONTROL_METHOD,
    attach_controls,
    rank_contrast_candidates,
    reference_control_rows,
)
from src.scene_uncertainty.contrast_inputs import AGGREGATIONS, ARMS, load_contrast_inputs
from src.scene_uncertainty.contrast_plots import (
    AUROC_LIMITS,
    CHANCE,
    EMPTY_SPAN,
    METHOD_COLOURS,
    PLOT_DPI,
    PLOT_FILENAMES,
    panel_plan,
    write_contrast_plots,
)
from src.scene_uncertainty.contrast_scores import SCORE_METHODS
from src.scene_uncertainty.corruption_metrics import EXPECTED_SEVERITIES

from tests.scene_uncertainty.contrast_test_utils import write_source_bundle

IMAGES = 6

FAST_SAMPLES = 20
"""What `prepared` bootstraps with. Nothing in this file reads a bootstrap interval -- the
figures do not draw one -- so the sample count only decides how long `attach_controls` takes.
`twin_auroc_by_severity` is what these figures need from it, and that is not resampled."""

ARM_NAMES = tuple(arm.name for arm in ARMS)


# --- the shared fixture -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _pipeline(tmp_path_factory):
    """The whole run, built once. Handed to no test directly; `prepared` copies it.

    Built once because building it is the expensive part of this file -- 2,916 rows, 81
    candidates and three bootstraps apiece -- and handed out as a deep copy because several
    tests write into what they are given and one of them
    (`test_drawing_does_not_write_into_the_candidates_controls_rows_or_fits`) is *about* what
    writes into it. A shared mutable would make that test pass or fail on the order pytest
    happened to collect the file in.
    """
    source = write_source_bundle(tmp_path_factory.mktemp("bundle") / "source")
    inputs = load_contrast_inputs(source, expected_image_count=IMAGES)
    rows, fits = build_contrast_rows(inputs)
    candidates = summarize_contrast_candidates(rows, expected_image_count=IMAGES)
    controls = summarize_contrast_candidates(
        reference_control_rows(rows), expected_image_count=IMAGES
    )
    attach_controls(candidates, controls, rows, samples=FAST_SAMPLES)
    return {"candidates": candidates, "controls": controls, "rows": rows, "fits": fits}


@pytest.fixture
def prepared(_pipeline):
    return copy.deepcopy(_pipeline)


# --- the synthetic candidates -------------------------------------------------------------------


def level(arm: str, aggregation: str, method: str) -> float:
    """A number no other `(arm, summary, method)` cell produces.

    The shared fixture cannot bind which panel drew which cell. `default_score` takes no
    aggregation argument, so a run's `mean`, `q90` and `top20_mean` curves are the same six
    numbers, and a figure that drew the `mean` candidate into all three panels of a row would
    match every assertion made against real rows. Hundreds for the arm, tens for the summary
    and units for the method, so a mismatch names which coordinate went wrong.
    """
    return float(
        100 * (ARM_NAMES.index(arm) + 1)
        + 10 * (list(AGGREGATIONS).index(aggregation) + 1)
        + (list(SCORE_METHODS).index(method) + 1 if method in SCORE_METHODS else 9)
    )


def statistics(values, *, unscored=()):
    """A six-severity `severity_statistics`, with `count == 0` at the severities named.

    The quartiles are deliberately asymmetric about the median -- one below and two above -- so
    a band drawn from `q75`/`q25` the wrong way round, or from the mean, is a different polygon
    and not merely a differently sized one.
    """
    return {
        severity: (
            {"count": 0, "mean": None, "variance": None,
             "median": None, "q25": None, "q75": None}
            if severity in unscored
            else {"count": IMAGES, "mean": value, "variance": 1.0,
                  "median": value, "q25": value - 1.0, "q75": value + 2.0}
        )
        for severity, value in zip(EXPECTED_SEVERITIES, values)
    }


def curve(base: float, *, falling: bool = False):
    return [base - severity if falling else base + severity for severity in EXPECTED_SEVERITIES]


def aurocs(base: float, *, offset: float = 0.0):
    """Five AUROCs inside the unit interval, distinct per severity and per base."""
    return {
        severity: round(0.30 + base / 2000.0 + 0.01 * severity + offset, 6)
        for severity in CORRUPTED_SEVERITIES
    }


def synthetic(arm, aggregation, method, *, signal=DEPLOYABLE_SIGNAL, values=None,
              unscored=(), auroc="derive", twin="derive", **extra):
    base = level(arm, aggregation, method)
    if auroc == "derive":
        auroc = aurocs(base)
    if twin == "derive":
        twin = aurocs(base, offset=-0.1)
    candidate = {
        "arm": arm, "signal": signal, "aggregation": aggregation, "method": method,
        "severity_statistics": statistics(
            curve(base) if values is None else values, unscored=unscored
        ),
        "auroc_by_severity": auroc,
        "twin_auroc_by_severity": twin,
    }
    candidate.update(extra)
    return candidate


def synthetic_candidates(**overrides):
    return [
        synthetic(arm, aggregation, method, **overrides)
        for arm in ARM_NAMES
        for aggregation in AGGREGATIONS
        for method in SCORE_METHODS
    ]


def synthetic_controls(**overrides):
    """The reference controls, drawn from the negative of their arm's level.

    Negative on purpose: a reference and a responsive curve are the same quantity in the same
    units and nothing about the drawing distinguishes them, so the fixture puts them on opposite
    sides of zero and the panel's line order becomes checkable.
    """
    return [
        synthetic(
            arm, aggregation, REFERENCE_CONTROL_METHOD,
            values=[-value for value in curve(level(arm, aggregation, REFERENCE_CONTROL_METHOD))],
            **overrides,
        )
        for arm in ARM_NAMES
        for aggregation in AGGREGATIONS
    ]


def synthetic_rows():
    """Two clean images per arm and summary, with reference and responsive far apart."""
    return [
        {
            "image_id": image_id, "severity": severity, "arm": arm,
            "signal": DEPLOYABLE_SIGNAL, "aggregation": aggregation, "method": method,
            "reference": level(arm, aggregation, method) + image_id,
            "responsive": -level(arm, aggregation, method) - image_id,
        }
        for arm in ARM_NAMES
        for aggregation in AGGREGATIONS
        for method in SCORE_METHODS
        for image_id in (2, 1)
        for severity in EXPECTED_SEVERITIES
    ]


def synthetic_fits():
    return {
        (arm, DEPLOYABLE_SIGNAL, aggregation): {
            "final_line": [2.0 + ARM_NAMES.index(arm), 5.0 + list(AGGREGATIONS).index(aggregation)]
        }
        for arm in ARM_NAMES
        for aggregation in AGGREGATIONS
    }


def synthetic_bundle(**overrides):
    bundle = {
        "candidates": synthetic_candidates(),
        "controls": synthetic_controls(),
        "rows": synthetic_rows(),
        "fits": synthetic_fits(),
    }
    bundle.update(overrides)
    return bundle


# --- reading the figures ------------------------------------------------------------------------


@contextmanager
def drawn(bundle):
    """The four live figures and the ranges they were drawn with.

    They come from the same call `write_contrast_plots` renders and saves, so what a test reads
    off an axis is what the file holds. Closed on the way out: matplotlib warns once more than
    twenty figures are open, and this file opens four per test.
    """
    figures, limits = plots_module._contrast_figures(
        bundle["candidates"], bundle["controls"], bundle["rows"], bundle["fits"]
    )
    try:
        yield figures, limits
    finally:
        for figure in figures.values():
            plt.close(figure)


def panel(figure, arm: str, aggregation: str):
    """The axis the plan puts this arm and summary on, found through the plan and not counted."""
    plan = panel_plan([])["anchor"]
    return figure.axes[plan.index((arm, aggregation))]


def message(axis) -> str:
    return "\n".join(text.get_text() for text in axis.texts)


def curves(axis):
    """The data lines of an AUROC panel, with the chance line left out.

    The chance line is an `axhline`, so it is a `Line2D` in `get_lines()` like any other; it is
    told apart by having two vertices at one height rather than one per corrupted severity.
    """
    return [
        line for line in axis.get_lines()
        if len(line.get_xdata()) == len(CORRUPTED_SEVERITIES)
    ]


def chance_lines(axis):
    return [
        line for line in axis.get_lines()
        if len(line.get_ydata()) == 2 and set(line.get_ydata()) == {CHANCE}
    ]


def band_bounds(collection, severities=EXPECTED_SEVERITIES):
    """A `fill_between` band's lower and upper edge at each severity, read off the polygons.

    One closed path runs along one boundary and back along the other, so the two edges are the
    smallest and the largest vertex at each severity rather than whichever half came first. A
    band with a `nan` in it is split into several paths, so every path is pooled.
    """
    vertices = np.concatenate([path.vertices for path in collection.get_paths()])
    lower, upper = {}, {}
    for severity in severities:
        at = vertices[np.isclose(vertices[:, 0], severity)][:, 1]
        if at.size:
            lower[severity], upper[severity] = float(at.min()), float(at.max())
    return lower, upper


def every_stat(bundle, key="candidates", *, methods=None, signal=DEPLOYABLE_SIGNAL):
    """Every median and quartile the named collection holds, for a span recomputed by hand."""
    return [
        item["severity_statistics"][severity][field]
        for item in bundle[key]
        if item["signal"] == signal and (methods is None or item["method"] in methods)
        for severity in EXPECTED_SEVERITIES
        for field in ("median", "q25", "q75")
        if item["severity_statistics"][severity][field] is not None
    ]


# --- the four files ------------------------------------------------------------------------------


def test_the_figure_names_are_the_spec_names():
    assert sorted(PLOT_FILENAMES.values()) == [
        "anchor_and_responsive_actual_distance.png",
        "auroc_by_blur_severity.png",
        "clean_anchor_relationship.png",
        "contrast_scores_by_severity.png",
    ]
    assert set(PLOT_FILENAMES) == {"anchor", "relationship", "contrast", "auroc"}
    # The pairing, and not only the two sets. Two swapped names leave both sets identical, and
    # `test_each_file_holds_the_figure_its_name_promises` reads the same mapping on both sides
    # of its comparison, so the swap is invisible to it as well: the anchor figure would be
    # written into the file named for the relationship and both would agree that it belongs
    # there. This is the only assertion in the file that says which name is whose.
    assert PLOT_FILENAMES == {
        "anchor": "anchor_and_responsive_actual_distance.png",
        "relationship": "clean_anchor_relationship.png",
        "contrast": "contrast_scores_by_severity.png",
        "auroc": "auroc_by_blur_severity.png",
    }


def test_the_four_declared_figures_are_written(tmp_path, prepared):
    write_contrast_plots(tmp_path, **prepared)

    written = sorted(path.name for path in tmp_path.glob("*.png"))
    assert written == sorted(PLOT_FILENAMES.values())
    assert len(written) == 4
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(written)
    for name in written:
        assert (tmp_path / name).stat().st_size > 0
        assert (tmp_path / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_each_file_holds_the_figure_its_name_promises(tmp_path):
    """Kills a swapped pair of filenames, which no per-figure assertion can see.

    Matplotlib's Agg output is byte-deterministic for one figure in one process, so each name's
    file is compared against the figure that name is supposed to carry rather than against a
    property the four figures happen not to share.
    """
    bundle = synthetic_bundle()
    write_contrast_plots(tmp_path, **bundle)

    with drawn(synthetic_bundle()) as (figures, _):
        for key, name in PLOT_FILENAMES.items():
            buffer = BytesIO()
            figures[key].savefig(buffer, format="png", dpi=PLOT_DPI)
            assert (tmp_path / name).read_bytes() == buffer.getvalue(), name


def test_the_figures_are_saved_at_the_declared_dpi(tmp_path):
    """A figure saved at matplotlib's default 100 DPI is a smaller image of the same drawing."""
    assert PLOT_DPI == 160
    bundle = synthetic_bundle()
    write_contrast_plots(tmp_path, **bundle)

    with drawn(synthetic_bundle()) as (figures, _):
        for key, name in PLOT_FILENAMES.items():
            width, height = struct.unpack(">II", (tmp_path / name).read_bytes()[16:24])
            inches = figures[key].get_size_inches()
            assert (width, height) == (
                round(inches[0] * PLOT_DPI), round(inches[1] * PLOT_DPI)
            ), name


# --- the panel plan ------------------------------------------------------------------------------


def test_the_panel_plan_is_the_arm_table_in_order():
    plan = panel_plan([])

    expected = [(arm.name, aggregation) for arm in ARMS for aggregation in AGGREGATIONS]
    assert plan["anchor"] == expected
    assert plan["relationship"] == expected
    assert plan["auroc"] == expected
    assert plan["contrast"] == [(method,) for method in SCORE_METHODS]
    assert expected[0] == ("decile_00_10__50_60", "mean")
    assert expected[-1] == ("decile_90_100__50_60__combined", "top20_mean")
    # Three lists, not three names for one: a caller editing the plan it was handed for one
    # figure must not be editing the other two figures' plans.
    assert plan["anchor"] is not plan["relationship"] is not plan["auroc"]


def test_the_panel_counts_are_the_declared_ones(tmp_path, prepared):
    from src.scene_uncertainty.contrast_plots import panel_plan

    plan = panel_plan(prepared["candidates"])
    assert len(plan["anchor"]) == 12
    assert len(plan["relationship"]) == 12
    assert len(plan["contrast"]) == 4
    assert len(plan["auroc"]) == 12


@pytest.mark.parametrize(
    "description, remove",
    [
        ("one arm gone", lambda item: item["arm"] == "quintile_00_20__40_60"),
        ("one summary gone", lambda item: item["aggregation"] == "q90"),
        ("one method gone", lambda item: item["method"] == "relative_gap"),
        ("the persistence signal gone", lambda item: item["signal"] == DEPLOYABLE_SIGNAL),
        ("everything gone", lambda item: True),
    ],
)
def test_the_panel_plan_ignores_which_candidates_exist(prepared, description, remove):
    """One dimension removed at a time, because a plan can be derived from any one of them.

    An arm-derived plan and a summary-derived plan are different mistakes and a fixture missing
    only an arm binds only the first. The last case is the whole candidate list, which is the
    run this function exists for: nothing measured, twelve panels that say so.
    """
    remaining = [item for item in prepared["candidates"] if not remove(item)]
    assert len(remaining) < len(prepared["candidates"]), description

    plan = panel_plan(remaining)
    assert [len(plan[key]) for key in ("anchor", "relationship", "contrast", "auroc")] == [
        12, 12, 4, 12
    ]
    assert plan == panel_plan(prepared["candidates"])


def test_the_grids_are_the_shapes_the_panel_counts_ask_for(prepared):
    """Twelve axes in a row is still twelve axes, so the shape is read off the gridspec."""
    with drawn(prepared) as (figures, _):
        for key in ("anchor", "relationship", "auroc"):
            assert len(figures[key].axes) == 12, key
            for axis in figures[key].axes:
                assert axis.get_subplotspec().get_gridspec().get_geometry() == (4, 3), key
        assert len(figures["contrast"].axes) == 4
        for axis in figures["contrast"].axes:
            assert axis.get_subplotspec().get_gridspec().get_geometry() == (1, 4)


def test_every_panel_names_the_arm_and_summary_the_plan_put_there(prepared):
    with drawn(prepared) as (figures, _):
        for key in ("anchor", "relationship", "auroc"):
            plan = panel_plan([])[key]
            for axis, (arm_name, aggregation) in zip(figures[key].axes, plan):
                title = axis.get_title()
                assert title.splitlines()[0] == arm_name, key
                assert aggregation == title.splitlines()[1].split(" | ")[0], key
        assert [axis.get_title() for axis in figures["contrast"].axes] == list(SCORE_METHODS)


def test_a_panel_title_carries_the_family_and_the_scope_that_make_two_panels_incomparable():
    with drawn(synthetic_bundle()) as (figures, _):
        titles = {axis.get_title() for axis in figures["anchor"].axes}
        assert "decile_00_10__50_60\nmean | anchored | layer_2" in titles
        assert "decile_90_100__50_60__combined\nq90 | differential | combined" in titles


def test_the_grid_figures_name_their_units_on_the_edge_panels(prepared):
    """Both axis labels, and only on the edges: twelve copies of two strings is ink."""
    with drawn(prepared) as (figures, _):
        for key, xlabel, ylabel in (
            ("anchor", "Blur severity", "Raw un-oriented distance"),
            ("auroc", "Blur severity (corrupted only)",
             "AUROC (oriented, clean vs this severity)"),
        ):
            axes = figures[key].axes
            assert [axes[index].get_xlabel() for index in range(9, 12)] == [xlabel] * 3
            assert [axes[index].get_xlabel() for index in range(0, 9)] == [""] * 9
            assert [axes[index].get_ylabel() for index in (0, 3, 6, 9)] == [ylabel] * 4
            assert [
                axes[index].get_ylabel()
                for index in range(12) if index % 3
            ] == [""] * 8
        contrast = figures["contrast"].axes
        assert contrast[0].get_ylabel() == "Raw un-oriented contrast score"
        assert [axis.get_xlabel() for axis in contrast] == ["Blur severity"] * 4


def test_the_panels_are_laid_out_to_fit_rather_than_left_on_the_default_margins(prepared):
    """`tight_layout` is what stops twelve two-line titles overwriting the panels above them.

    Not visible in a file-against-file comparison, since both sides re-render the same code, so
    it is read off the geometry. Two things are asserted, and the second is the one that is
    about the reader: matplotlib's default left margin is `0.125` of the figure and a laid-out
    grid has pulled the panels out of it, and no panel -- ticks, labels and two-line title
    included -- reaches into the caption above it or the legend below it.
    """
    with drawn(prepared) as (figures, _):
        for key, figure in figures.items():
            assert figure.axes[0].get_position().x0 < 0.125, key
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            into_figure = figure.transFigure.inverted()
            caption = figure._suptitle.get_window_extent(renderer).transformed(into_figure)
            legend = figure.legends[0].get_window_extent(renderer).transformed(into_figure)
            for axis in figure.axes:
                box = axis.get_tightbbox(renderer).transformed(into_figure)
                assert box.y1 <= caption.y0, key
                assert box.y0 >= legend.y1, key


# --- the returned ranges -------------------------------------------------------------------------


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


def test_the_anchor_range_is_the_smallest_and_largest_value_it_drew(tmp_path, prepared):
    """Recomputed by hand from the two series the anchor figure draws, and from nothing else.

    Not `min`/`max` over every candidate: the anchor figure draws the `raw_responsive`
    candidates and the `raw_reference` controls, and a range stretched to a `relative_gap`
    candidate it never plots would be a number no panel is a claim about.
    """
    limits = write_contrast_plots(tmp_path, **prepared)

    values = every_stat(prepared, methods={RESPONSIVE_CONTROL_METHOD}) + every_stat(
        prepared, "controls"
    )
    assert limits["anchor"] == [min(values), max(values)]
    assert limits["anchor"] != limits["contrast"]


def test_the_contrast_range_is_the_smallest_and_largest_value_it_drew(tmp_path, prepared):
    limits = write_contrast_plots(tmp_path, **prepared)

    values = every_stat(prepared)
    assert limits["contrast"] == [min(values), max(values)]


def test_the_relationship_range_covers_both_axes_of_the_scatter(tmp_path, prepared):
    """Both axes, because a reference and a responsive distance are one quantity in one unit."""
    limits = write_contrast_plots(tmp_path, **prepared)

    scattered = [
        row[field]
        for row in prepared["rows"]
        if row["severity"] == 0
        and row["signal"] == DEPLOYABLE_SIGNAL
        and row["method"] == RESPONSIVE_CONTROL_METHOD
        for field in ("reference", "responsive")
    ]
    assert limits["relationship"][0] <= min(scattered)
    assert limits["relationship"][1] >= max(scattered)
    with drawn(prepared) as (figures, live):
        assert live["relationship"] == limits["relationship"]
        drawn_values = [
            value
            for axis in figures["relationship"].axes
            for line in axis.get_lines()
            for value in line.get_ydata()
        ] + scattered
        assert limits["relationship"] == [min(drawn_values), max(drawn_values)]


def test_the_auroc_range_is_the_declared_interval_and_not_the_span_of_its_curves(
    tmp_path, prepared
):
    """Kills a recomputed AUROC range, which looks identical in the file and is not comparable.

    The fixture's AUROCs reach neither 0 nor 1 -- asserted, because a fixture that happened to
    touch both would make the two answers the same number and the assertion vacuous.
    """
    limits = write_contrast_plots(tmp_path, **prepared)

    plotted = [
        value
        for candidate in prepared["candidates"]
        if candidate["signal"] == DEPLOYABLE_SIGNAL
        for vector in (candidate["auroc_by_severity"], candidate["twin_auroc_by_severity"])
        if vector is not None
        for value in vector.values()
    ]
    assert min(plotted) > AUROC_LIMITS[0]
    assert max(plotted) < AUROC_LIMITS[1]
    assert limits["auroc"] == AUROC_LIMITS
    assert limits["auroc"] is not AUROC_LIMITS

    with drawn(prepared) as (figures, _):
        for axis in figures["auroc"].axes:
            assert axis.get_ylim() == tuple(limits["auroc"])


def test_a_figure_that_drew_nothing_reports_an_empty_range_rather_than_a_plausible_one():
    empty = {"candidates": [], "controls": [], "rows": [], "fits": {}}
    with drawn(empty) as (figures, limits) :
        assert limits["anchor"] == EMPTY_SPAN
        assert limits["relationship"] == EMPTY_SPAN
        assert limits["contrast"] == EMPTY_SPAN
        assert limits["auroc"] == AUROC_LIMITS
        for key in ("anchor", "relationship", "auroc"):
            assert len(figures[key].axes) == 12, key
            for axis in figures[key].axes:
                assert not axis.get_lines(), key
                assert message(axis), key


# --- the anchor figure ---------------------------------------------------------------------------


def test_the_anchor_panel_draws_the_reference_control_and_the_responsive_candidate():
    """Exact values, in a fixed order, with the band read off its polygon.

    `raw_responsive` is the responsive range itself -- the method returns the responsive value
    unchanged -- and `raw_reference` is the control built the same way from the other bucket, so
    a panel drawing a `raw_gap` or a `clean_residual` under either label is a different curve.
    """
    with drawn(synthetic_bundle()) as (figures, _):
        for arm in ARM_NAMES:
            for aggregation in AGGREGATIONS:
                axis = panel(figures["anchor"], arm, aggregation)
                responsive = curve(level(arm, aggregation, RESPONSIVE_CONTROL_METHOD))
                reference = [
                    -value
                    for value in curve(level(arm, aggregation, REFERENCE_CONTROL_METHOD))
                ]
                lines = axis.get_lines()
                assert len(lines) == 2
                assert list(lines[0].get_ydata()) == reference
                assert list(lines[1].get_ydata()) == responsive
                assert list(lines[0].get_xdata()) == list(EXPECTED_SEVERITIES)
                assert len(axis.collections) == 2
                for collection, medians in zip(axis.collections, (reference, responsive)):
                    lower, upper = band_bounds(collection)
                    assert [lower[s] for s in EXPECTED_SEVERITIES] == [
                        value - 1.0 for value in medians
                    ]
                    assert [upper[s] for s in EXPECTED_SEVERITIES] == [
                        value + 2.0 for value in medians
                    ]


def test_the_anchor_panels_are_not_three_copies_of_one_summary():
    """The one property the shared fixture cannot bind, because its three summaries are equal."""
    with drawn(synthetic_bundle()) as (figures, _):
        for arm in ARM_NAMES:
            drawn_by_summary = {
                aggregation: tuple(
                    panel(figures["anchor"], arm, aggregation).get_lines()[1].get_ydata()
                )
                for aggregation in AGGREGATIONS
            }
            assert len(set(drawn_by_summary.values())) == len(AGGREGATIONS), arm
        for aggregation in AGGREGATIONS:
            drawn_by_arm = {
                arm: tuple(panel(figures["anchor"], arm, aggregation).get_lines()[1].get_ydata())
                for arm in ARM_NAMES
            }
            assert len(set(drawn_by_arm.values())) == len(ARM_NAMES), aggregation


def test_a_candidate_whose_distance_falls_is_drawn_falling():
    """A falling contrast is a finding, and an orientation applied here would draw it rising."""
    falling = [
        synthetic(
            arm, aggregation, method,
            values=curve(level(arm, aggregation, method), falling=True),
            orientation=-1, median_signed_spearman=-1.0,
        )
        for arm in ARM_NAMES
        for aggregation in AGGREGATIONS
        for method in SCORE_METHODS
    ]
    with drawn(synthetic_bundle(candidates=falling)) as (figures, limits):
        axis = panel(figures["anchor"], ARM_NAMES[0], "mean")
        responsive = list(axis.get_lines()[1].get_ydata())
        assert responsive == curve(
            level(ARM_NAMES[0], "mean", RESPONSIVE_CONTROL_METHOD), falling=True
        )
        assert responsive[-1] < responsive[0]


def test_the_anchor_panels_carry_the_severity_ladder_and_no_shared_range():
    """Panels must not share a range: three arms are `layer_2` distances and one is a z-score."""
    with drawn(synthetic_bundle()) as (figures, _):
        applied = set()
        for axis in figures["anchor"].axes:
            assert axis.get_xlim() == (0, 5)
            assert list(axis.get_xticks()) == list(EXPECTED_SEVERITIES)
            applied.add(axis.get_ylim())
        assert len(applied) == 12


def test_the_anchor_legend_says_which_colour_is_which_range():
    with drawn(synthetic_bundle()) as (figures, _):
        labels = [text.get_text() for text in figures["anchor"].legends[0].get_texts()]
        assert labels == [
            "reference range: median and interquartile band",
            "responsive range: median and interquartile band",
        ]


@pytest.mark.parametrize(
    "drop, expected",
    [
        ("both", "no reference candidate; no responsive candidate"),
        ("reference", "no reference candidate"),
        ("responsive", "no responsive candidate"),
    ],
)
def test_an_anchor_panel_says_which_of_its_two_series_never_arrived(drop, expected):
    target = (ARM_NAMES[1], "q90")

    def keep(item, method):
        return (item["arm"], item["aggregation"]) != target or item["method"] != method

    candidates = [
        item for item in synthetic_candidates()
        if drop == "reference" or keep(item, RESPONSIVE_CONTROL_METHOD)
    ]
    controls = [
        item for item in synthetic_controls()
        if drop == "responsive" or keep(item, REFERENCE_CONTROL_METHOD)
    ]
    bundle = synthetic_bundle(candidates=candidates, controls=controls)

    with drawn(bundle) as (figures, _):
        axis = panel(figures["anchor"], *target)
        assert expected in message(axis)
        assert len(axis.get_lines()) == (0 if drop == "both" else 1)
        # The panel keeps its place: its neighbours are untouched and still complete.
        neighbour = panel(figures["anchor"], ARM_NAMES[1], "mean")
        assert len(neighbour.get_lines()) == 2
        assert not message(neighbour)


def test_a_series_that_scored_no_image_is_said_differently_from_one_that_never_arrived():
    """A pipeline gap and an empty measurement are two findings, and one message hides one."""
    target = (ARM_NAMES[0], "mean")
    candidates = [
        synthetic(*target, method, unscored=EXPECTED_SEVERITIES)
        if (item["arm"], item["aggregation"], item["method"])
        == (*target, RESPONSIVE_CONTROL_METHOD)
        else item
        for item in synthetic_candidates()
        for method in [item["method"]]
    ]
    with drawn(synthetic_bundle(candidates=candidates)) as (figures, _):
        axis = panel(figures["anchor"], *target)
        assert "responsive scored no image at any severity" in message(axis)
        assert "no responsive candidate" not in message(axis)
        assert len(axis.get_lines()) == 1


def test_a_severity_no_image_scored_is_a_gap_in_the_line_and_not_a_missing_panel():
    """`count == 0` at one severity is a hole in a curve; it is not an absent candidate.

    A dropped point would join severity 1 to severity 3 with a straight segment through a
    severity nothing was measured at, which publishes an interpolation as a measurement.
    """
    target = (ARM_NAMES[2], "top20_mean")
    candidates = [
        synthetic(*target, item["method"], unscored=(2,))
        if (item["arm"], item["aggregation"]) == target
        else item
        for item in synthetic_candidates()
    ]
    with drawn(synthetic_bundle(candidates=candidates)) as (figures, limits):
        axis = panel(figures["anchor"], *target)
        assert len(axis.get_lines()) == 2
        assert message(axis) == ""
        responsive = list(axis.get_lines()[1].get_ydata())
        assert np.isnan(responsive[2])
        assert not np.isnan(np.array(responsive[:2] + responsive[3:])).any()
        assert list(axis.get_xticks()) == list(EXPECTED_SEVERITIES)
        # A `nan` never reaches the returned range.
        assert all(np.isfinite(value) for value in limits["anchor"])


# --- the relationship figure ---------------------------------------------------------------------


def test_the_relationship_panel_scatters_severity_zero_reference_against_responsive(prepared):
    with drawn(prepared) as (figures, _):
        for arm in ARM_NAMES:
            for aggregation in AGGREGATIONS:
                expected = sorted(
                    (row["reference"], row["responsive"])
                    for row in prepared["rows"]
                    if row["severity"] == 0
                    and row["arm"] == arm
                    and row["signal"] == DEPLOYABLE_SIGNAL
                    and row["aggregation"] == aggregation
                    and row["method"] == RESPONSIVE_CONTROL_METHOD
                )
                assert len(expected) == IMAGES
                axis = panel(figures["relationship"], arm, aggregation)
                offsets = axis.collections[0].get_offsets()
                assert sorted(map(tuple, offsets.tolist())) == expected


def test_each_image_contributes_one_point_and_not_one_per_method(prepared):
    """All four methods carry the same two ranges, so taking every method quadruples the cloud."""
    with drawn(prepared) as (figures, _):
        for axis in figures["relationship"].axes:
            assert len(axis.collections[0].get_offsets()) == IMAGES


def test_the_relationship_line_is_the_final_fit_drawn_across_the_reference_range():
    bundle = synthetic_bundle()
    with drawn(bundle) as (figures, _):
        for arm in ARM_NAMES:
            for aggregation in AGGREGATIONS:
                axis = panel(figures["relationship"], arm, aggregation)
                slope, offset = bundle["fits"][(arm, DEPLOYABLE_SIGNAL, aggregation)][
                    "final_line"
                ]
                references = [
                    level(arm, aggregation, RESPONSIVE_CONTROL_METHOD) + image_id
                    for image_id in (1, 2)
                ]
                line = axis.get_lines()[0]
                assert list(line.get_xdata()) == [min(references), max(references)]
                assert list(line.get_ydata()) == pytest.approx(
                    [offset + slope * value for value in (min(references), max(references))]
                )


def test_the_relationship_line_is_not_the_same_line_in_every_panel():
    """Twelve fits, and a lookup that ignored the arm or the summary would draw one of them."""
    with drawn(synthetic_bundle()) as (figures, _):
        drawn_lines = {
            (arm, aggregation): tuple(
                panel(figures["relationship"], arm, aggregation).get_lines()[0].get_ydata()
            )
            for arm in ARM_NAMES
            for aggregation in AGGREGATIONS
        }
        assert len(set(drawn_lines.values())) == 12


def test_the_relationship_axes_name_which_bucket_is_which(prepared):
    with drawn(prepared) as (figures, _):
        for arm in ARMS:
            for aggregation in AGGREGATIONS:
                axis = panel(figures["relationship"], arm.name, aggregation)
                assert axis.get_xlabel() == f"{arm.reference_bin} (reference)"
                assert axis.get_ylabel() == f"{arm.responsive_bin} (responsive)"
        # The two differential arms take the 90-100 decile as their reference, which is the
        # swap a symmetric label would hide.
        differential = panel(figures["relationship"], "decile_90_100__50_60", "mean")
        assert differential.get_xlabel() == "decile_90_100 (reference)"
        assert differential.get_ylabel() == "decile_50_60 (responsive)"


def test_the_relationship_caption_says_the_published_residuals_used_fold_lines(prepared):
    with drawn(prepared) as (figures, _):
        caption = figures["relationship"]._suptitle.get_text()
        assert "cross-fitted" in caption
        assert "fold line" in caption
        assert "final" in caption


def test_a_panel_with_no_fitted_line_still_scatters_and_says_the_line_is_missing():
    fits = synthetic_fits()
    fits[(ARM_NAMES[3], DEPLOYABLE_SIGNAL, "mean")] = {"final_line": None}
    with drawn(synthetic_bundle(fits=fits)) as (figures, _):
        axis = panel(figures["relationship"], ARM_NAMES[3], "mean")
        assert len(axis.collections[0].get_offsets()) == 2
        assert len(axis.get_lines()) == 0
        assert "no final clean line was fitted" in message(axis)


def test_a_panel_with_no_clean_rows_keeps_its_place_and_says_it_is_empty():
    rows = [
        row for row in synthetic_rows()
        if (row["arm"], row["aggregation"]) != (ARM_NAMES[0], "q90")
    ]
    with drawn(synthetic_bundle(rows=rows)) as (figures, _):
        axis = panel(figures["relationship"], ARM_NAMES[0], "q90")
        assert len(axis.collections) == 0
        assert "no severity-zero" in message(axis)
        assert axis.get_xticks().size == 0
        assert len(figures["relationship"].axes) == 12
        assert panel(figures["relationship"], ARM_NAMES[0], "mean").collections


# --- the contrast figure -------------------------------------------------------------------------


def test_each_contrast_panel_holds_one_line_per_arm_and_summary(prepared):
    with drawn(prepared) as (figures, _):
        plan = panel_plan([])["contrast"]
        for axis, (method,) in zip(figures["contrast"].axes, plan):
            expected = sum(
                1 for candidate in prepared["candidates"]
                if candidate["signal"] == DEPLOYABLE_SIGNAL and candidate["method"] == method
            )
            assert len(axis.get_lines()) == expected, method
            assert len(axis.collections) == expected, method
        # `relative_gap` is excluded at the signed `combined` scope, so that panel is short by
        # exactly one arm's three summaries and the other three panels are full.
        counts = {
            axis.get_title(): len(axis.get_lines()) for axis in figures["contrast"].axes
        }
        assert counts == {
            "raw_responsive": 12, "raw_gap": 12, "relative_gap": 9, "clean_residual": 12
        }


def test_the_contrast_panels_draw_the_raw_median_and_its_interquartile_band():
    with drawn(synthetic_bundle()) as (figures, _):
        axis = figures["contrast"].axes[SCORE_METHODS.index("raw_gap")]
        drawn_curves = {tuple(line.get_ydata()) for line in axis.get_lines()}
        assert len(drawn_curves) == 12
        assert tuple(curve(level(ARM_NAMES[0], "mean", "raw_gap"))) in drawn_curves
        assert tuple(curve(level(ARM_NAMES[3], "top20_mean", "raw_gap"))) in drawn_curves
        lower, upper = band_bounds(axis.collections[0])
        first = curve(level(ARM_NAMES[0], "mean", "raw_gap"))
        assert [lower[s] for s in EXPECTED_SEVERITIES] == [value - 1.0 for value in first]
        assert [upper[s] for s in EXPECTED_SEVERITIES] == [value + 2.0 for value in first]


def test_the_contrast_panels_draw_only_the_persistence_signal():
    """The confidence twin is what the AUROC figure compares against, not a thirteenth line."""
    twins = [
        synthetic(arm, aggregation, method, signal="confidence")
        for arm in ARM_NAMES
        for aggregation in AGGREGATIONS
        for method in SCORE_METHODS
    ]
    bundle = synthetic_bundle(candidates=synthetic_candidates() + twins)
    with drawn(bundle) as (figures, _):
        for axis in figures["contrast"].axes:
            assert len(axis.get_lines()) == 12


def test_a_method_no_arm_produced_gets_a_labelled_empty_panel():
    candidates = [
        item for item in synthetic_candidates() if item["method"] != "clean_residual"
    ]
    with drawn(synthetic_bundle(candidates=candidates)) as (figures, _):
        axis = figures["contrast"].axes[SCORE_METHODS.index("clean_residual")]
        assert axis.get_lines() == []
        assert "clean_residual" in message(axis)
        assert axis.get_xticks().size == 0
        assert len(figures["contrast"].axes) == 4
        assert len(figures["contrast"].axes[0].get_lines()) == 12


# --- the AUROC figure ----------------------------------------------------------------------------


def test_each_auroc_panel_draws_each_method_solid_and_its_twin_dashed_in_one_colour():
    """The dashed twin is the readability claim of this figure, and it is testable.

    A twin drawn solid, or in its own colour, produces a figure a reader would not question and
    could not read: eight curves in eight styles rather than four pairs.
    """
    with drawn(synthetic_bundle()) as (figures, _):
        for arm in ARM_NAMES:
            for aggregation in AGGREGATIONS:
                axis = panel(figures["auroc"], arm, aggregation)
                lines = curves(axis)
                assert len(lines) == 2 * len(SCORE_METHODS)
                for index, method in enumerate(SCORE_METHODS):
                    own, twin = lines[2 * index], lines[2 * index + 1]
                    base = level(arm, aggregation, method)
                    assert own.get_linestyle() == "-"
                    assert twin.get_linestyle() == "--"
                    assert own.get_color() == METHOD_COLOURS[method]
                    assert twin.get_color() == METHOD_COLOURS[method]
                    assert list(own.get_xdata()) == list(CORRUPTED_SEVERITIES)
                    assert list(own.get_ydata()) == [
                        aurocs(base)[severity] for severity in CORRUPTED_SEVERITIES
                    ]
                    assert list(twin.get_ydata()) == [
                        aurocs(base, offset=-0.1)[severity]
                        for severity in CORRUPTED_SEVERITIES
                    ]


def test_the_four_method_colours_are_four_different_colours():
    """A shared colour is what pairs a twin with its method; four shared colours pair nothing."""
    assert len(set(METHOD_COLOURS.values())) == len(SCORE_METHODS)
    assert set(METHOD_COLOURS) == set(SCORE_METHODS)


def test_every_auroc_panel_carries_the_chance_line_at_one_half(prepared):
    assert CHANCE == 0.5
    with drawn(prepared) as (figures, _):
        for axis in figures["auroc"].axes:
            marks = chance_lines(axis)
            assert len(marks) == 1
            assert list(marks[0].get_ydata()) == [CHANCE, CHANCE]
            assert marks[0].get_linestyle() == ":"


def test_every_auroc_panel_is_given_the_declared_interval_explicitly(prepared):
    """Not `sharey`: the shared range is the figure's claim, and letting matplotlib propagate
    one panel's limits would make it true of the drawing while saying nothing about the code."""
    with drawn(prepared) as (figures, _):
        for axis in figures["auroc"].axes:
            assert axis.get_ylim() == (AUROC_LIMITS[0], AUROC_LIMITS[1])
            assert list(axis.get_xticks()) == list(CORRUPTED_SEVERITIES)
            assert axis.get_xlim() == (CORRUPTED_SEVERITIES[0], CORRUPTED_SEVERITIES[-1])


def test_the_auroc_panels_start_at_severity_one_because_clean_is_the_comparison_group():
    with drawn(synthetic_bundle()) as (figures, _):
        assert list(CORRUPTED_SEVERITIES) == [1, 2, 3, 4, 5]
        for axis in figures["auroc"].axes:
            for line in curves(axis):
                assert 0 not in list(line.get_xdata())


def test_a_none_auroc_vector_leaves_a_labelled_panel_rather_than_crashing(tmp_path, prepared):
    for candidate in prepared["candidates"]:
        candidate["twin_auroc_by_severity"] = None
        candidate["macro_auroc"] = None
        candidate["auroc_by_severity"] = None
    write_contrast_plots(tmp_path, **prepared)
    assert (tmp_path / PLOT_FILENAMES["auroc"]).stat().st_size > 0

    with drawn(prepared) as (figures, _):
        assert len(figures["auroc"].axes) == 12
        for axis in figures["auroc"].axes:
            assert axis.get_lines() == []
            assert "no AUROC and no confidence twin" in message(axis)
            assert axis.get_xticks().size == 0


def test_a_candidate_with_an_auroc_and_no_twin_still_draws_its_own_curve():
    """The dimension a run of complete twins cannot bind: half a pair is not a missing panel."""
    candidates = [
        {**item, "twin_auroc_by_severity": None} for item in synthetic_candidates()
    ]
    with drawn(synthetic_bundle(candidates=candidates)) as (figures, _):
        for axis in figures["auroc"].axes:
            lines = curves(axis)
            assert len(lines) == len(SCORE_METHODS)
            assert {line.get_linestyle() for line in lines} == {"-"}
            assert len(chance_lines(axis)) == 1
            assert "no twin" in message(axis)


def test_a_twin_with_no_candidate_of_its_own_is_still_drawn_dashed():
    candidates = [{**item, "auroc_by_severity": None} for item in synthetic_candidates()]
    with drawn(synthetic_bundle(candidates=candidates)) as (figures, _):
        for axis in figures["auroc"].axes:
            lines = curves(axis)
            assert len(lines) == len(SCORE_METHODS)
            assert {line.get_linestyle() for line in lines} == {"--"}
            assert "no AUROC" in message(axis)


def test_a_method_with_no_candidate_at_all_is_named_in_the_panel_it_is_missing_from(prepared):
    """`relative_gap` is excluded at the `combined` scope, and the panel says so rather than
    quietly drawing three pairs where the other rows draw four."""
    with drawn(prepared) as (figures, _):
        for aggregation in AGGREGATIONS:
            axis = panel(figures["auroc"], "decile_90_100__50_60__combined", aggregation)
            assert "relative_gap: no candidate" in message(axis)
            assert len(curves(axis)) == 2 * (len(SCORE_METHODS) - 1)
        untouched = panel(figures["auroc"], "decile_90_100__50_60", "mean")
        assert message(untouched) == ""
        assert len(curves(untouched)) == 2 * len(SCORE_METHODS)


def test_the_auroc_legend_names_the_methods_the_twin_and_the_chance_line():
    with drawn(synthetic_bundle()) as (figures, _):
        labels = [text.get_text() for text in figures["auroc"].legends[0].get_texts()]
        assert labels == list(SCORE_METHODS) + [
            "confidence twin of the same method", "chance (0.5)"
        ]


# --- what the figures may not do -----------------------------------------------------------------


def test_drawing_does_not_write_into_the_candidates_controls_rows_or_fits(tmp_path, prepared):
    """A candidate dictionary is the one object the ranking, the CSV and the report all hold.

    Run twice, and the second run is the one that binds it. A pipeline candidate already carries
    `orientation`, `macro_auroc` and the seventeen fields `attach_controls` writes, so a
    plotting-time `setdefault` for any of them is a no-op on it and a deep-copy comparison sees
    nothing -- a fixture that does not vary a dimension cannot bind it. The minimal candidates
    carry only the seven fields these figures read, so anything written into them shows.
    """
    for label, bundle in (("a whole run", prepared), ("minimal dictionaries", synthetic_bundle())):
        before = copy.deepcopy(bundle)

        write_contrast_plots(tmp_path, **bundle)

        assert bundle == before, label
        assert bundle["candidates"][0] is not before["candidates"][0], label
    assert set(synthetic_candidates()[0]) == {
        "arm", "signal", "aggregation", "method",
        "severity_statistics", "auroc_by_severity", "twin_auroc_by_severity",
    }


def test_the_figures_are_drawn_although_nothing_in_the_run_is_deployable(tmp_path, prepared):
    """Six images is not 250, so this run's ranking is empty -- and the figures are not.

    A figure filtered on `deployable` would be twelve empty panels on every fixture in this
    suite and on any run smaller than the full tuning partition, which is exactly the run an
    operator draws figures to look at.
    """
    assert rank_contrast_candidates(prepared["candidates"]) == []
    assert not any(candidate["deployable"] for candidate in prepared["candidates"])

    limits = write_contrast_plots(tmp_path, **prepared)
    assert limits["contrast"] != EMPTY_SPAN
    with drawn(prepared) as (figures, _):
        for key in ("anchor", "relationship", "auroc"):
            assert all(axis.get_lines() for axis in figures[key].axes), key


def test_writing_the_figures_leaves_none_of_them_open(tmp_path, prepared):
    before = plt.get_fignums()
    write_contrast_plots(tmp_path, **prepared)
    assert plt.get_fignums() == before


def test_writing_into_a_missing_directory_leaves_no_open_figures(tmp_path, prepared):
    import matplotlib.pyplot as plt

    with pytest.raises(OSError):
        write_contrast_plots(tmp_path / "absent", **prepared)
    assert plt.get_fignums() == []


def test_a_builder_that_raises_part_way_closes_the_figures_already_built(monkeypatch, prepared):
    """Three open figures in no collection the caller could close is a warning hours later."""
    before = plt.get_fignums()

    def explode(*args, **kwargs):
        raise RuntimeError("no AUROC figure today")

    monkeypatch.setattr(plots_module, "_auroc_figure", explode)
    with pytest.raises(RuntimeError):
        plots_module._contrast_figures(
            prepared["candidates"], prepared["controls"], prepared["rows"], prepared["fits"]
        )
    assert plt.get_fignums() == before


def test_the_backend_is_pinned_before_pyplot_is_imported():
    """There is no display on the box these figures are drawn on.

    Asserted on the source order rather than only on `get_backend()`: by the time this runs,
    another module may already have pinned Agg, and this one has to pin it itself.
    """
    assert matplotlib.get_backend().lower() == "agg"
    source = Path(plots_module.__file__).read_text()
    assert source.index('matplotlib.use("Agg")') < source.index("import matplotlib.pyplot")
