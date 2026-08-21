"""Four small multiples, and the seven ways a comparable figure quietly stops being comparable.

`corruption_plots` draws one panel per confidence bucket and, across four figures, exactly two
y-ranges. Every failure this file is written against produces a complete, well-labelled set of
PNGs that a reader would not question:

* **per-panel autoscaling.** Ten panels each fitted to their own bucket make every bucket look
  equally responsive, which is the one conclusion the figure exists to prevent. Killed by
  asserting the *numeric* limits of all fifteen persistence panels are one value, not merely
  that each panel has some limit;
* **one scale for both signals.** A persistence distance and `1 - confidence` share no unit, so
  a single range makes one of them a flat line at the bottom of its panel. The fixture puts the
  two signals well over an order of magnitude apart so a shared range would be unmissable --
  and the test asserts the two ranges differ rather than just that each exists;
* **limits taken from one scheme.** The decile and quintile fixtures deliberately reach
  different extremes in *both* directions -- persistence quintiles run lower, persistence
  deciles run higher -- so a limit computed from either scheme alone clips the other's band.
  `test_the_shared_range_reaches_the_extremes_of_both_schemes` pins the exact numbers;
* **orienting the plotted values.** A candidate whose distance falls as blur rises is a finding.
  `decile_30_40` falls from 27 to 22 and its panel must fall; multiplying by the candidate's
  orientation would draw it rising and the reader would never know. Every synthetic candidate
  carries a real `orientation` key for this reason -- without one the requirement is enforced
  by `KeyError` alone, and `candidate.get("orientation", 1)` walks past that. The one test on
  real reporter rows checks the confidence control too, which is the candidate the reporter
  actually locks at `-1`;
* **a layout left on the default margins.** `tight_layout` cannot be seen by comparing a saved
  file against a re-render of the same code, so it is read off the axes geometry instead;
* **a layout that is not the one specified.** Ten axes in a row is still ten axes, so the
  row and column counts are read off the gridspec rather than counted;
* **a bucket silently dropped.** Nine drawn panels in a ten-panel grid look like a complete
  ten-bucket measurement if the tenth is simply not there. The absent bucket keeps its own
  position and says why it is empty.

The fixture gives every bucket of one signal and scheme the *same* range on purpose. The ranges
that have to be reconciled here are scheme against scheme and signal against signal, and making
the buckets identical is what lets a test remove one bucket without also moving the shared
limits it is checking.

`test_the_figures_read_real_candidate_metrics` is the one test that does not use the synthetic
dictionaries. Every other test here would still pass if this module and `corruption_plots`
agreed on a field name that `corruption_reporting` does not actually write, so one test builds
its candidates by running the real reporter over real scored rows.
"""

from __future__ import annotations

import json
import struct
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator

from src.scene_uncertainty import corruption_plots as plots_module
from src.scene_uncertainty.confidence_deciles import DECILE_NAMES, QUINTILE_NAMES
from src.scene_uncertainty.corruption_plots import (
    PLOT_DPI,
    PLOT_FILENAMES,
    shared_limits,
    write_corruption_plots,
)
from src.scene_uncertainty.corruption_reporting import summarize_candidates


SIGNAL_SCOPES = {"persistence": "layer_2", "confidence": "confidence"}
"""The scope each signal is drawn at. Persistence has a decoder layer; its matched confidence
control does not, and `confidence` is the scope `score_selection` writes for it."""

RECIPE = {"membership_mode": "dynamic", "padding_mode": "filtered", "aggregation": "q90"}

PERSISTENCE_DECILE = [22.0 + severity for severity in range(6)]
PERSISTENCE_DECILE_FALLING = [27.0 - severity for severity in range(6)]
PERSISTENCE_QUINTILE = [4.0 + 3.0 * severity for severity in range(6)]
CONFIDENCE_DECILE = [0.30 + 0.01 * severity for severity in range(6)]
CONFIDENCE_QUINTILE = [0.10 + 0.02 * severity for severity in range(6)]
CONFIDENCE_CONSTANT = [0.15] * 6

PERSISTENCE_SPREAD = 0.5
CONFIDENCE_SPREAD = 0.005

# Persistence: quintiles reach down to 3.5 and deciles up to 27.5, so neither scheme's own
# extremes are the pair's. Confidence: quintiles down to 0.095, deciles up to 0.355.
PERSISTENCE_LIMITS = (3.5 - 0.05 * 24.0, 27.5 + 0.05 * 24.0)
CONFIDENCE_LIMITS = (0.095 - 0.05 * 0.26, 0.355 + 0.05 * 0.26)

FALLING_BIN = "decile_30_40"
MIXED_BIN = "quintile_60_80"
FLAT_BIN = "quintile_80_100"


def candidate(
    signal, scheme, confidence_bin, medians, *,
    spread=PERSISTENCE_SPREAD, trend=1.0, flat_count=0, measured_count=10, **overrides,
):
    """One candidate dictionary carrying only the fields the figures read.

    Deliberately not the full 125-key row `corruption_reporting` publishes: a fixture that
    carried every field would hide a plot module reaching for a field it has no business
    reading, and `test_the_figures_read_real_candidate_metrics` covers the real shape.

    `mean` and `variance` are present and deliberately disagree with `median` and the
    quartiles, because they are the two fields this figure is defined by *not* drawing. So is
    `orientation`, for the same reason and more sharply: a fixture without it protects the
    anti-orientation requirement by `KeyError` alone, which an implementation spelled
    `candidate.get("orientation", 1)` walks straight past. It is set the way
    `corruption_metrics.choose_orientation` would set it, `None` included for a zero median.
    """
    statistics = {
        str(severity): {
            "count": measured_count,
            # The mean sits a whole unit above the median and the variance is four times the
            # half-width of the band, which is the skew the figure exists to be robust to. It
            # is also what makes a line drawn from the mean, or a band drawn from the variance,
            # visibly the wrong curve rather than the same one under another name.
            "mean": float(median) + 1.0,
            "variance": 4.0,
            "median": float(median),
            "q25": float(median) - spread,
            "q75": float(median) + spread,
        }
        for severity, median in enumerate(medians)
    }
    return {
        "signal": signal,
        "bucket_scheme": scheme,
        "confidence_bin": confidence_bin,
        "score_scope": SIGNAL_SCOPES[signal],
        **RECIPE,
        "severity_statistics": statistics,
        "median_signed_spearman": trend,
        "orientation": None if not trend else (1 if trend > 0 else -1),
        "measured_count": measured_count,
        "flat_count": flat_count,
        **overrides,
    }


def full_candidates():
    """Thirty candidates: both signals, both schemes, every bucket, four trend directions."""
    built = []
    for name in DECILE_NAMES:
        falling = name == FALLING_BIN
        built.append(candidate(
            "persistence", "decile", name,
            PERSISTENCE_DECILE_FALLING if falling else PERSISTENCE_DECILE,
            trend=-1.0 if falling else 1.0,
        ))
        built.append(candidate(
            "confidence", "decile", name, CONFIDENCE_DECILE, spread=CONFIDENCE_SPREAD,
        ))
    for name in QUINTILE_NAMES:
        built.append(candidate(
            "persistence", "quintile", name, PERSISTENCE_QUINTILE,
            # A group whose images disagree: the median signed trend is zero and none of the
            # images is flat, so the panel may not be captioned `flat`.
            **({"trend": 0.0, "flat_count": 0} if name == MIXED_BIN else {}),
        ))
        constant = name == FLAT_BIN
        built.append(candidate(
            "confidence", "quintile", name,
            CONFIDENCE_CONSTANT if constant else CONFIDENCE_QUINTILE,
            spread=CONFIDENCE_SPREAD,
            **({"trend": 0.0, "flat_count": 10} if constant else {}),
        ))
    return built


def decoys():
    """Five candidates outside the figures' fixed recipe, each with a range that would show.

    A million is far enough outside every real band that any of these reaching the limits
    collapses all thirty real panels onto one line -- so a recipe filter that misses one of the
    five cannot fail quietly.
    """
    wild = [1_000_000.0] * 6
    first = DECILE_NAMES[0]
    return [
        candidate("persistence", "decile", first, wild, membership_mode="frozen"),
        candidate("persistence", "decile", first, wild, padding_mode="unfiltered"),
        candidate("persistence", "decile", first, wild, aggregation="mean"),
        candidate("persistence", "decile", first, wild, score_scope="layer_1"),
        candidate("confidence", "decile", first, wild, score_scope="layer_2"),
    ]


def without(candidates, **match):
    return [
        row for row in candidates
        if not all(row[field] == value for field, value in match.items())
    ]


@contextmanager
def drawn(candidates):
    """The four live figures and the limits they were drawn with.

    The figures come from the same call `write_corruption_plots` renders and saves, so what a
    test reads off an axis is what the file holds. Closed on the way out: matplotlib warns once
    more than twenty figures are open, and this file opens four per test.
    """
    figures, limits = plots_module._corruption_figures(candidates)
    try:
        yield figures, limits
    finally:
        for figure in figures.values():
            plt.close(figure)


def panels(figure):
    """The axes of one small multiple, in the order the buckets are meant to appear."""
    return list(figure.axes)


def drawn_panels(figure):
    """The panels that got a curve, which are the only ones a y-range is a claim about.

    A bucket with nothing to draw never receives `set_ylim`, so it keeps matplotlib's default
    `(0.0, 1.0)` -- which is correct, since it carries no curve and no ticks, but it means
    "every axis of this figure reports one range" is an invariant that quietly stops holding
    the first time a run is missing a bucket. The shared-range tests are scoped to the drawn
    panels and assert how many they expected, so the scoping cannot pass by drawing nothing.
    """
    return [axis for axis in figure.axes if axis.get_lines()]


def band_bounds(axis):
    """The interquartile band's lower and upper edge at each severity, read off the polygon.

    `fill_between` emits one closed path that runs along one boundary and back along the other,
    so the two edges are recovered as the smallest and largest vertex at each severity rather
    than by assuming which half of the path came first.
    """
    vertices = axis.collections[0].get_paths()[0].vertices
    lower, upper = [], []
    for severity in range(6):
        at = vertices[np.isclose(vertices[:, 0], severity)][:, 1]
        lower.append(float(at.min()))
        upper.append(float(at.max()))
    return lower, upper


# --- the four files ---------------------------------------------------------------------------


def test_the_four_figures_are_named_by_signal_and_scheme():
    assert PLOT_FILENAMES == {
        ("persistence", "decile"): "persistence_actual_distance_deciles.png",
        ("persistence", "quintile"): "persistence_actual_distance_quintiles.png",
        ("confidence", "decile"): "confidence_actual_distance_deciles.png",
        ("confidence", "quintile"): "confidence_actual_distance_quintiles.png",
    }


def test_exactly_four_pngs_are_written_and_nothing_else(tmp_path):
    write_corruption_plots(tmp_path, full_candidates())

    written = sorted(path.name for path in tmp_path.iterdir())
    assert written == sorted(PLOT_FILENAMES.values())
    for name in written:
        assert (tmp_path / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_each_file_holds_the_figure_its_name_promises(tmp_path):
    """Kills a swapped pair of filenames, which no per-figure assertion can see.

    Matplotlib's Agg PNG output is byte-deterministic for one figure in one process, so the
    file each name holds is compared against the figure that name is supposed to carry rather
    than against a property the four figures happen not to share.
    """
    write_corruption_plots(tmp_path, full_candidates())

    with drawn(full_candidates()) as (figures, _):
        for key, name in PLOT_FILENAMES.items():
            buffer = BytesIO()
            figures[key].savefig(buffer, format="png", dpi=PLOT_DPI)
            assert (tmp_path / name).read_bytes() == buffer.getvalue(), name


def test_the_figures_are_saved_at_the_declared_dpi(tmp_path):
    """A figure saved at the default 100 DPI is a smaller image of the same drawing."""
    assert PLOT_DPI == 160
    write_corruption_plots(tmp_path, full_candidates())

    with drawn(full_candidates()) as (figures, _):
        for key, name in PLOT_FILENAMES.items():
            width, height = struct.unpack(">II", (tmp_path / name).read_bytes()[16:24])
            inches = figures[key].get_size_inches()
            assert (width, height) == (
                round(inches[0] * PLOT_DPI), round(inches[1] * PLOT_DPI)
            )


def test_the_panels_are_laid_out_to_fit_rather_than_left_on_the_default_margins():
    """`tight_layout` is what stops ten two-line titles and the suptitle overwriting each other.

    Not visible in a file-against-file comparison, since both sides re-render the same code, so
    it is read off the geometry: the default left margin is `0.125` of the figure and the
    default top of the axes is `0.88`, and a laid-out figure has pulled the panels out into the
    first and down out of the second to make room for the suptitle.
    """
    with drawn(full_candidates()) as (figures, _):
        for figure in figures.values():
            first = figure.axes[0].get_position()
            assert first.x0 < 0.125
            assert first.y1 < 0.88


def test_writing_the_figures_leaves_none_of_them_open(tmp_path):
    before = plt.get_fignums()
    write_corruption_plots(tmp_path, full_candidates())

    assert plt.get_fignums() == before


def test_a_failed_write_closes_the_figures_it_had_already_built(tmp_path):
    """The four figures exist before the first `savefig`, and one of them can still fail.

    Nothing here creates the directory, so an absent one raises part-way through the loop. If
    the remaining figures were left open, a caller that retried -- or a test session that ran
    several of these -- would walk into matplotlib's too-many-open-figures warning, whose whole
    point is that it fires long after the code that caused it.
    """
    before = plt.get_fignums()
    with pytest.raises(FileNotFoundError):
        write_corruption_plots(tmp_path / "not-created", full_candidates())

    assert plt.get_fignums() == before


# --- the layout -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scheme, geometry, names",
    [("decile", (2, 5), DECILE_NAMES), ("quintile", (1, 5), QUINTILE_NAMES)],
)
def test_each_scheme_is_drawn_in_the_layout_its_bucket_count_asks_for(scheme, geometry, names):
    """Ten axes in a row is also ten axes, so the row and column counts are read directly."""
    with drawn(full_candidates()) as (figures, _):
        for signal in ("persistence", "confidence"):
            figure = figures[(signal, scheme)]
            assert len(panels(figure)) == len(names)
            for axis in panels(figure):
                assert axis.get_subplotspec().get_gridspec().get_geometry() == geometry


@pytest.mark.parametrize("scheme, names", [("decile", DECILE_NAMES), ("quintile", QUINTILE_NAMES)])
def test_the_panels_run_in_ascending_confidence_order(scheme, names):
    with drawn(full_candidates()) as (figures, _):
        for signal in ("persistence", "confidence"):
            titles = [axis.get_title() for axis in panels(figures[(signal, scheme)])]
            assert [title.splitlines()[0] for title in titles] == list(names)


# --- one range per signal, shared across both of that signal's figures -------------------------


@pytest.mark.parametrize(
    "signal, expected",
    [("persistence", PERSISTENCE_LIMITS), ("confidence", CONFIDENCE_LIMITS)],
)
def test_both_figures_of_one_signal_share_one_numeric_y_range(signal, expected):
    with drawn(full_candidates()) as (figures, limits):
        curves = [
            axis
            for scheme in ("decile", "quintile")
            for axis in drawn_panels(figures[(signal, scheme)])
        ]
        applied = {axis.get_ylim() for axis in curves}

        assert len(curves) == len(DECILE_NAMES) + len(QUINTILE_NAMES)
        assert len(applied) == 1
        assert applied.pop() == pytest.approx(expected)
        assert limits[signal] == pytest.approx(expected)


def test_the_two_signals_never_share_a_scale():
    """A distance to clean-bank fingerprints and `1 - confidence` have unrelated units.

    One range covering both would flatten every confidence panel against the axis floor, which
    is why the fixture puts the two signals well over an order of magnitude apart.
    """
    with drawn(full_candidates()) as (_, limits):
        assert limits["persistence"] != limits["confidence"]
        assert max(limits["confidence"]) < min(limits["persistence"])


def test_the_shared_range_reaches_the_extremes_of_both_schemes():
    """Neither scheme's own extremes are the pair's, in either direction, for either signal."""
    chosen = [row for row in full_candidates() if row["signal"] == "persistence"]
    deciles = [row for row in chosen if row["bucket_scheme"] == "decile"]
    quintiles = [row for row in chosen if row["bucket_scheme"] == "quintile"]

    joint = shared_limits(chosen, "persistence")
    assert joint == pytest.approx(PERSISTENCE_LIMITS)
    assert shared_limits(deciles, "persistence") != pytest.approx(joint)
    assert shared_limits(quintiles, "persistence") != pytest.approx(joint)
    # The quintile band bottoms out below the decile-only floor and the decile band tops out
    # above the quintile-only ceiling, so either scheme alone would clip the other.
    assert joint[0] < shared_limits(deciles, "persistence")[0]
    assert joint[1] > shared_limits(quintiles, "persistence")[1]


def test_the_margin_never_collapses_onto_a_flat_candidate():
    """A candidate that never moves has a zero span, and `set_ylim(x, x)` is not a panel."""
    flat = [candidate("persistence", "decile", DECILE_NAMES[0], [4.0] * 6, spread=0.0)]
    lower, upper = shared_limits(flat, "persistence")

    assert lower == pytest.approx(4.0 - 0.2)
    assert upper == pytest.approx(4.0 + 0.2)

    at_zero = [candidate("persistence", "decile", DECILE_NAMES[0], [0.0] * 6, spread=0.0)]
    lower, upper = shared_limits(at_zero, "persistence")
    assert (lower, upper) == pytest.approx((-1e-6, 1e-6))


def test_only_the_dynamic_filtered_q90_recipe_reaches_the_figures():
    with drawn(full_candidates() + decoys()) as (figures, limits):
        assert limits["persistence"] == pytest.approx(PERSISTENCE_LIMITS)
        assert limits["confidence"] == pytest.approx(CONFIDENCE_LIMITS)
        for figure in figures.values():
            for axis in panels(figure):
                assert len(axis.get_lines()) == 1


def test_the_limits_the_call_returns_are_the_limits_the_panels_use(tmp_path):
    """Task 7 records this return in `summary.json` beside the figures it describes."""
    returned = write_corruption_plots(tmp_path, full_candidates() + decoys())

    assert set(returned) == {"persistence", "confidence"}
    assert json.loads(json.dumps(returned)) == returned
    with drawn(full_candidates() + decoys()) as (figures, _):
        for (signal, _scheme), figure in figures.items():
            for axis in panels(figure):
                assert axis.get_ylim() == pytest.approx(tuple(returned[signal]))


# --- what one panel draws ---------------------------------------------------------------------


def test_every_panel_carries_the_severity_ladder_one_median_line_and_one_band():
    """The six severities are named, not left to a locator that happens to agree with them.

    An autolocator over the range 0 to 5 picks the same six numbers today, so the tick values
    alone cannot tell the two apart -- but a severity is an identity on this axis and not a
    position, and a panel whose ticks are inferred prints whatever the locator prefers the
    moment anything about the range changes. Both facts are asserted.
    """
    with drawn(full_candidates()) as (figures, _):
        for figure in figures.values():
            for axis in panels(figure):
                assert list(axis.get_xticks()) == [0, 1, 2, 3, 4, 5]
                assert isinstance(axis.xaxis.get_major_locator(), FixedLocator)
                assert axis.get_xlim() == (0, 5)
                assert len(axis.get_lines()) == 1
                assert len(axis.collections) == 1


def test_the_line_is_the_raw_median_and_the_band_is_its_interquartile_range():
    with drawn(full_candidates()) as (figures, _):
        axis = panels(figures[("persistence", "quintile")])[0]
        line = axis.get_lines()[0]

        assert list(line.get_xdata()) == [0, 1, 2, 3, 4, 5]
        assert list(line.get_ydata()) == pytest.approx(PERSISTENCE_QUINTILE)
        lower, upper = band_bounds(axis)
        assert lower == pytest.approx([m - PERSISTENCE_SPREAD for m in PERSISTENCE_QUINTILE])
        assert upper == pytest.approx([m + PERSISTENCE_SPREAD for m in PERSISTENCE_QUINTILE])


def test_a_candidate_whose_distance_falls_is_drawn_falling():
    """Orientation belongs to AUROC. A signal that drops under blur is a finding, not a sign.

    `decile_30_40` runs 27 down to 22 with a median signed Spearman of -1. Multiplying the
    plotted values by that orientation would draw the same curve rising, and the panel would
    say the opposite of what was measured.
    """
    with drawn(full_candidates()) as (figures, _):
        axis = panels(figures[("persistence", "decile")])[DECILE_NAMES.index(FALLING_BIN)]
        drawn_values = list(axis.get_lines()[0].get_ydata())

        assert drawn_values == pytest.approx(PERSISTENCE_DECILE_FALLING)
        assert all(later < earlier for earlier, later in zip(drawn_values, drawn_values[1:]))
        lower, upper = band_bounds(axis)
        assert lower[0] > lower[-1] and upper[0] > upper[-1]


@pytest.mark.parametrize(
    "signal, scheme, names, bucket, label",
    [
        ("persistence", "decile", DECILE_NAMES, "decile_00_10", "increasing"),
        ("persistence", "decile", DECILE_NAMES, FALLING_BIN, "decreasing"),
        ("persistence", "quintile", QUINTILE_NAMES, MIXED_BIN, "mixed"),
        ("confidence", "quintile", QUINTILE_NAMES, FLAT_BIN, "flat"),
    ],
)
def test_each_panel_names_its_bucket_and_the_group_trend(signal, scheme, names, bucket, label):
    """`flat` and `mixed` are both zero-median groups and mean opposite things.

    `mixed` is a candidate whose scenes disagree about the direction; `flat` is one where every
    measured scene found no trend at all. The median alone cannot separate them, so a title
    derived from it would caption a disagreement as a non-result.
    """
    with drawn(full_candidates()) as (figures, _):
        title = panels(figures[(signal, scheme)])[names.index(bucket)].get_title()

    assert bucket in title
    assert label in title
    others = {"increasing", "decreasing", "flat", "mixed"} - {label}
    assert not [other for other in others if other in title]


def test_a_group_with_nothing_measured_is_not_captioned_as_a_measurement():
    """No fully measured image means no group trend -- neither `flat` nor `mixed` is true.

    `corruption_metrics` already calls a curve it could not read `unmeasured`; the group label
    reuses that word rather than borrowing one of the four that assert a finding.
    """
    nothing = candidate(
        "persistence", "decile", DECILE_NAMES[0], PERSISTENCE_DECILE,
        trend=None, measured_count=0, flat_count=0,
    )
    with drawn(full_candidates()[1:] + [nothing]) as (figures, _):
        title = panels(figures[("persistence", "decile")])[0].get_title()

    assert "unmeasured" in title
    assert "flat" not in title and "mixed" not in title


def test_the_axis_labels_name_the_unit_each_signal_is_measured_in():
    with drawn(full_candidates()) as (figures, _):
        for scheme in ("decile", "quintile"):
            persistence = figures[("persistence", scheme)]
            confidence = figures[("confidence", scheme)]
            assert "Raw q90 persistence distance" in {
                axis.get_ylabel() for axis in panels(persistence)
            }
            assert "Raw q90 confidence uncertainty (1 - confidence)" in {
                axis.get_ylabel() for axis in panels(confidence)
            }
            for figure in (persistence, confidence):
                assert "Blur severity" in {axis.get_xlabel() for axis in panels(figure)}


def test_the_confidence_panels_are_not_captioned_with_a_decoder_layer():
    """The confidence control has no layer scope; a panel of `1 - confidence` labelled
    `layer_2` states something that was never measured."""
    with drawn(full_candidates()) as (figures, _):
        confidence = figures[("confidence", "decile")]
        assert "layer_2" not in confidence.get_suptitle()
        assert "layer_2" not in "".join(axis.get_ylabel() for axis in panels(confidence))
        assert "layer_2" in figures[("persistence", "decile")].get_suptitle()


# --- what the figures do when a bucket has nothing to draw -------------------------------------


def test_a_bucket_with_no_candidate_keeps_its_place_and_says_it_is_empty():
    """Nine curves in a ten-panel grid read as a complete ten-bucket measurement.

    The missing bucket keeps its position, so every other bucket stays under its own name, and
    the panel carries the reason instead of an empty pair of axes -- which is indistinguishable
    from a measurement that came out flat.
    """
    missing = "decile_50_60"
    thin = without(full_candidates(), signal="persistence", confidence_bin=missing)
    with drawn(thin) as (figures, _):
        axes = panels(figures[("persistence", "decile")])
        assert len(axes) == len(DECILE_NAMES)
        empty = axes[DECILE_NAMES.index(missing)]

        assert not empty.get_lines() and not empty.collections
        assert missing in " ".join(text.get_text() for text in empty.texts)
        assert [axis.get_title().splitlines()[0] for axis in axes] == list(DECILE_NAMES)
        assert len(axes[DECILE_NAMES.index(missing) + 1].get_lines()) == 1

        # The nine that remain are still on the one range, and the range is still the one the
        # full table produces -- a missing bucket must not rescale the buckets around it.
        assert len(drawn_panels(figures[("persistence", "decile")])) == len(DECILE_NAMES) - 1
        applied = {
            axis.get_ylim()
            for scheme in ("decile", "quintile")
            for axis in drawn_panels(figures[("persistence", scheme)])
        }
        assert len(applied) == 1
        assert applied.pop() == pytest.approx(PERSISTENCE_LIMITS)


def test_a_candidate_missing_a_severity_is_neither_drawn_nor_scaled_to():
    """A `None` statistic is `corruption_reporting` saying no image scored that severity.

    It is not a zero and it is not a gap in a curve; there is no band to draw across it, and a
    limit computed over it either raises or silently rescales every panel of its signal.
    """
    blanked = candidate("persistence", "quintile", QUINTILE_NAMES[0], PERSISTENCE_QUINTILE)
    blanked["severity_statistics"]["3"] = {
        "count": 0, "mean": None, "variance": None, "median": None, "q25": None, "q75": None,
    }
    blanked["severity_statistics"]["0"]["q75"] = 1_000_000.0
    thin = without(
        full_candidates(), signal="persistence", bucket_scheme="quintile",
        confidence_bin=QUINTILE_NAMES[0],
    )
    with drawn(thin + [blanked]) as (figures, limits):
        axes = panels(figures[("persistence", "quintile")])

        assert limits["persistence"] == pytest.approx(PERSISTENCE_LIMITS)
        assert not axes[0].get_lines()
        assert [len(axis.get_lines()) for axis in axes[1:]] == [1, 1, 1, 1]


def test_a_signal_with_no_candidate_at_all_is_refused_rather_than_drawn_empty():
    persistence_only = [row for row in full_candidates() if row["signal"] == "persistence"]

    with pytest.raises(ValueError, match="confidence"):
        plots_module._corruption_figures(persistence_only)


# --- the real candidate rows -------------------------------------------------------------------


SCENE_OFFSETS = {1: 0.0, 2: 1.0, 3: 8.0}
"""Three scenes at unrelated baseline distances, the third far above the other two.

That skew is the reason the figure draws a median and not a mean, and it is what makes the
integration test discriminate: the reporter publishes both, and here they sit two whole units
apart at every severity, so a line drawn from either is a different curve.
"""


def scored_rows():
    """Three images scored at six severities for every bucket of both schemes, both signals.

    Rising persistence and falling confidence, so the two signals cannot be confused for each
    other in the figures, and one selected-query list per selection so the reporter's own
    `validate_selected_queries` accepts the table.
    """
    rows = []
    for scheme, names in (("decile", DECILE_NAMES), ("quintile", QUINTILE_NAMES)):
        for confidence_bin in names:
            for image_id in SCENE_OFFSETS:
                for severity in range(6):
                    shared = {
                        "image_id": image_id,
                        "severity": severity,
                        "bucket_scheme": scheme,
                        "confidence_bin": confidence_bin,
                        **RECIPE,
                        "source_partition": "tuning",
                        "selected_count": 4,
                        "clean_overlap": 0.5,
                        "selected_query_ids": [0, 1, 2, 3],
                    }
                    rows.append({
                        **shared, "signal": "persistence", "score_scope": "layer_2",
                        "score": 2.0 + severity + SCENE_OFFSETS[image_id],
                    })
                    rows.append({
                        **shared, "signal": "confidence", "score_scope": "confidence",
                        "score": 0.9 - 0.05 * severity,
                    })
    return rows


def test_the_figures_read_real_candidate_metrics(tmp_path):
    """The synthetic dictionaries above agree with `corruption_reporting`, not just themselves."""
    _, candidates, _ = summarize_candidates(
        scored_rows(), expected_image_count=len(SCENE_OFFSETS)
    )
    limits = write_corruption_plots(tmp_path, candidates)

    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(PLOT_FILENAMES.values())
    with drawn(candidates) as (figures, _):
        axis = panels(figures[("persistence", "decile")])[0]
        published = next(
            row for row in candidates
            if row["signal"] == "persistence" and row["bucket_scheme"] == "decile"
            and row["confidence_bin"] == DECILE_NAMES[0]
        )
        clean = published["severity_statistics"]["0"]
        assert clean["mean"] != pytest.approx(clean["median"])
        assert list(axis.get_lines()[0].get_ydata()) == pytest.approx(
            [published["severity_statistics"][str(s)]["median"] for s in range(6)]
        )
        assert "increasing" in axis.get_title()

        # The candidate the anti-orientation requirement actually turns on: the reporter locks
        # `-1` here, and every published median is positive, so a drawn value that had been
        # multiplied by the orientation would be the negation of the one below.
        control = next(
            row for row in candidates
            if row["signal"] == "confidence" and row["bucket_scheme"] == "decile"
            and row["confidence_bin"] == DECILE_NAMES[0]
        )
        control_axis = panels(figures[("confidence", "decile")])[0]
        assert control["orientation"] == -1
        published_medians = [
            control["severity_statistics"][str(s)]["median"] for s in range(6)
        ]
        assert min(published_medians) > 0
        assert list(control_axis.get_lines()[0].get_ydata()) == pytest.approx(
            published_medians
        )
        assert "decreasing" in control_axis.get_title()
    assert min(limits["confidence"]) < 0.9
    assert min(limits["persistence"]) > 0.9


# --- the environment the figures are drawn in ---------------------------------------------------


def test_the_backend_is_pinned_before_pyplot_is_imported():
    """There is no display on the box these figures are drawn on.

    Asserted on the source order rather than only on `get_backend()`: by the time this runs,
    another module may already have pinned Agg, and this one has to pin it itself.
    """
    assert matplotlib.get_backend().lower() == "agg"
    source = Path(plots_module.__file__).read_text()
    assert source.index('matplotlib.use("Agg")') < source.index("import matplotlib.pyplot")
