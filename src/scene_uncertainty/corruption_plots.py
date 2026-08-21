"""Four small multiples of the raw scene score, drawn on two y-ranges and no more.

`corruption_reporting` publishes a number for every candidate; this module publishes the shape
behind four of those numbers. One panel per confidence bucket, six severities across each
panel, ten panels for the decile cut and five for the quintile cut, twice over -- once for
persistence and once for its matched confidence control.

Four decisions here look like presentation choices and are not.

**The plotted values are raw and un-oriented.** `corruption_metrics.choose_orientation` picks a
single `+1` or `-1` per candidate so that AUROC has one direction to rank in, and nothing in
this module ever applies it. A candidate whose distance *falls* as blur rises is a real finding
-- the feature moves, and a deployment could read it downwards -- and a figure that multiplied
by the orientation would draw that candidate rising, which is the opposite of what was
measured. The reader has to be able to see it fall. `median_signed_spearman` reaches the panel
as a word in the title and never as a factor on the data.

**All panels of one signal share exactly one y-range, and the two signals never share one.**
Without a shared range, per-panel autoscaling fits each bucket to its own spread and every
bucket looks equally responsive; the comparison the small multiple exists to support -- this
bucket moves further than that one -- becomes unreadable, because the only thing varying
between panels is the axis. So `shared_limits` is computed once per signal, jointly across that
signal's decile and quintile figures, and the same pair is handed to every one of its fifteen
panels. It is computed per *signal* because a persistence distance to clean-bank fingerprints
and `1 - confidence` are unrelated quantities: one range covering both would flatten whichever
signal is smaller against the floor of its own panels, and would invite exactly the
magnitude-against-magnitude comparison the design forbids.

**The band is the interquartile range around the median, not a standard deviation around the
mean.** The distribution at one severity pools scenes whose baseline distances are very
different from each other -- `corruption_metrics` names exactly that spread as the reason a
within-image Spearman and an across-scene AUROC can disagree -- so the handful of scenes
furthest out pull a mean and inflate a variance far more than they move a median or a quartile.
`corruption_reporting` publishes `mean` and `variance` beside `median`, `q25` and `q75` for
inspection and this module draws the second three; the first two are not what the figure is
for. The limits come from `q25` and `q75` for the same reason they are what is drawn -- a
range fitted to the drawn band, so no band is ever clipped and no panel is mostly empty.

**The slice is narrower than the metric sweep on purpose.** Dynamic membership, padding-filtered
queries, the `q90` scene summary, persistence at `layer_2` and the confidence control at its own
scope. Everything except the confidence bucket is held fixed, so a reader comparing two panels
is comparing two buckets and nothing else -- which is the only comparison a grid of panels
invites anyone to make. Three of the five held-fixed fields are the ones
`corruption_reporting`'s deployability gate also requires (`dynamic`, `filtered`, `layer_2`);
the aggregation is not one of them, and is fixed here for the figure's own reason. The frozen
twins, the unfiltered padding controls, the other aggregations and the other decoder layers are
all summarised in the bundle and none of them is drawn here.

Nothing in this module reads or joins an artifact. It takes the candidate dictionaries
`corruption_reporting.summarize_candidates` returns and writes PNGs into a directory the caller
already owns; staging that directory and recording the returned limits beside the figures
belongs to the bundle writer above it, not here.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# These figures are drawn on a headless box, so the backend is pinned before pyplot is imported
# rather than left to whatever matplotlib would autodetect.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .confidence_deciles import DECILE_NAMES, QUINTILE_NAMES  # noqa: E402
from .corruption_metrics import EXPECTED_SEVERITIES  # noqa: E402
from .corruption_reporting import (  # noqa: E402
    DEPLOYABLE_MEMBERSHIP_MODE,
    DEPLOYABLE_SCORE_SCOPE,
    DEPLOYABLE_SIGNAL,
    FILTERED_PADDING_MODE,
)
from .decile_scoring import CONFIDENCE_SCOPE  # noqa: E402


CONFIDENCE_SIGNAL = "confidence"
"""The name `score_selection` writes for the matched control. Spelled here rather than reused
from `CONFIDENCE_SCOPE`, which happens to be the same string for a different reason: the
control has no decoder layer to be scoped to, so its scope field carries its signal name."""

PLOT_AGGREGATION = "q90"
"""The one scene summary the figures draw. Not a deployability gate: `corruption_reporting`
ranks all three summaries against each other and gates on none of them.

Held fixed here for the figure's own reason -- with the recipe constant, two panels differ only
in their confidence bucket, which is the comparison a grid of panels invites. `q90` rather than
`mean` or `top20_mean` because it is the summary `decile_reporting` slices its figures on, so
these panels are read against the published experiment's and not beside it. The other two are
summarised in the bundle in full and neither is drawn.
"""

PLOT_SCOPES = {
    DEPLOYABLE_SIGNAL: DEPLOYABLE_SCORE_SCOPE,
    CONFIDENCE_SIGNAL: CONFIDENCE_SCOPE,
}
"""Which score scope each drawn signal is taken at.

A mapping rather than a list of allowed scopes, so a row's scope is checked against its own
signal: persistence at `confidence` and the confidence control at `layer_2` are both undrawable
here, and a signal this module does not draw at all has no scope to match. The iteration order
is the order the returned limits come back in."""

SIGNAL_AXIS_LABELS = {
    DEPLOYABLE_SIGNAL: f"Raw {PLOT_AGGREGATION} persistence distance",
    CONFIDENCE_SIGNAL: f"Raw {PLOT_AGGREGATION} confidence uncertainty (1 - confidence)",
}
"""Both labels say `Raw`, because that is the property a reader has to carry into the panel:
these are un-oriented scores, so a falling curve is a falling measurement. The aggregation is
interpolated rather than typed, so a label cannot go on claiming `q90` of a different summary.
"""

SEVERITY_AXIS_LABEL = "Blur severity"

SCHEME_BUCKET_NAMES = {"decile": DECILE_NAMES, "quintile": QUINTILE_NAMES}
"""Each scheme's buckets *in ascending confidence order*, and the panel order of its figure.

The names drive the panels rather than the candidates that happen to be present, which is what
keeps a bucket with no candidate in its own position instead of letting the buckets after it
shift one panel to the left under someone else's name.
"""

SCHEME_LAYOUTS = {"decile": (2, 5), "quintile": (1, 5)}
"""Rows and columns. Ten panels in a single row is also ten panels, and at one figure width it
halves each panel and doubles its height -- a narrow sliver in which the six severities sit
almost on top of each other, which is the one axis these figures need to stay readable.
"""

SCHEME_FIGURE_SIZES = {"decile": (16.0, 6.4), "quintile": (16.0, 3.6)}
"""Inches. Both grids are five columns wide at the same figure width, so one bucket's panel is
the same width in the decile figure as in the quintile one and the two can be read against each
other rather than each against itself."""

PLOT_FILENAMES = {
    ("persistence", "decile"): "persistence_actual_distance_deciles.png",
    ("persistence", "quintile"): "persistence_actual_distance_quintiles.png",
    ("confidence", "decile"): "confidence_actual_distance_deciles.png",
    ("confidence", "quintile"): "confidence_actual_distance_quintiles.png",
}
"""`actual_distance` in every name: these are the figures of the score as measured, as opposed
to the oriented AUROCs and the rank correlations the bundle publishes beside them."""

PLOT_DPI = 160

FIGURE_SLICE = (
    f"{DEPLOYABLE_MEMBERSHIP_MODE} membership, {FILTERED_PADDING_MODE} queries, "
    f"{PLOT_AGGREGATION} scene summary"
)
"""The recipe held fixed across every panel, printed on every figure. A small multiple read
without it looks like a statement about all the rows rather than about one slice of them."""


def shared_limits(candidates: list[dict], signal: str) -> tuple[float, float]:
    """One y-range for every panel of one signal, across both of its bucket schemes.

    `candidates` is the drawn slice -- see `_plot_candidates` -- so filtering it by `signal`
    alone is what selects the decile and the quintile figure together. That joint span is the
    point: a range taken from the deciles alone clips whatever the quintiles reach past it, and
    two figures on two ranges cannot be compared to each other at all, which leaves the reader
    with two pictures and no way to tell whether five wide buckets move as far as ten narrow
    ones.

    The extremes are `q25` and `q75` rather than the smallest and largest scores, because the
    band drawn from those quartiles is the thing that must fit; a range stretched to a single
    outlying scene would squash all fifteen panels into the middle of the axis.

    The 5% margin keeps the band off the frame. When the span is zero -- a candidate that never
    moves, which is a finding and not an error -- there is no span to take a fraction of, so the
    margin falls back to 5% of the level itself, and to `1e-6` at a level of exactly zero, since
    `set_ylim(x, x)` is not an axis.
    """
    chosen = [row for row in candidates if row["signal"] == signal]
    lower = min(
        stats["q25"]
        for row in chosen
        for stats in row["severity_statistics"].values()
    )
    upper = max(
        stats["q75"]
        for row in chosen
        for stats in row["severity_statistics"].values()
    )
    span = upper - lower
    margin = 0.05 * span if span > 0 else max(0.05 * abs(lower), 1e-6)
    return lower - margin, upper + margin


def _has_complete_statistics(candidate: dict) -> bool:
    """Whether this candidate has a median and both quartiles at all six severities.

    `corruption_reporting._statistics` writes `None` for a severity no image produced a finite
    score at. That absence is neither a zero nor a gap a line can be drawn across: there is no
    band at that severity, and a limit computed over the `None` would either raise or -- worse,
    if it were coerced -- rescale every panel of the signal to a number nobody measured. A
    candidate holding one is shown as absent rather than drawn or silently repaired.
    """
    statistics = candidate["severity_statistics"]
    return all(
        str(severity) in statistics
        and all(
            statistics[str(severity)][field] is not None
            for field in ("median", "q25", "q75")
        )
        for severity in EXPECTED_SEVERITIES
    )


def _plot_candidates(candidates: list[dict]) -> list[dict]:
    """The fixed slice the four figures draw, out of every candidate the run summarised.

    One recipe, one scope per signal, and only candidates with something to draw at all six
    severities. Applied once and handed to both `shared_limits` and the panels, so the range an
    axis carries is a range over exactly the curves its signal draws -- no wider, which would
    leave every panel with empty margins, and no narrower, which would clip a band.
    """
    return [
        candidate for candidate in candidates
        if candidate["membership_mode"] == DEPLOYABLE_MEMBERSHIP_MODE
        and candidate["padding_mode"] == FILTERED_PADDING_MODE
        and candidate["aggregation"] == PLOT_AGGREGATION
        and candidate["score_scope"] == PLOT_SCOPES.get(candidate["signal"])
        and _has_complete_statistics(candidate)
    ]


def _direction_label(candidate: dict) -> str:
    """Which way this bucket's scenes moved, as one word for the panel title.

    Read from the group's median signed Spearman when that median has a sign. A median of
    exactly zero is where the counts are needed, because two different groups produce it and
    they mean opposite things: one where every measured scene found no trend at all, and one
    where the scenes disagreed about the direction. Calling the second `flat` would report a
    disagreement as a non-result, so `flat` is claimed only when every measured scene is flat.

    A candidate with no fully measured scene has no group trend to label. None of the four words
    is true of it -- `flat` would be vacuously "all zero of them" and `mixed` would assert a
    disagreement that no scene took part in -- so it takes `corruption_metrics`' own word for a
    curve that could not be read, and says `unmeasured`.
    """
    median = candidate["median_signed_spearman"]
    if median is not None:
        if median > 0:
            return "increasing"
        if median < 0:
            return "decreasing"
    measured_count = candidate["measured_count"]
    if not measured_count:
        return "unmeasured"
    if candidate["flat_count"] == measured_count:
        return "flat"
    return "mixed"


def _absent(axis, message: str) -> None:
    """Say a panel had nothing to draw, rather than drawing an empty one.

    Follows the convention `decile_reporting` sets for the same hazard. An empty axis carrying
    the shared y-range and no curve is indistinguishable from a bucket whose score never moved;
    this prints the reason across the panel and removes the ticks, so the only thing the panel
    can be read as saying is the only thing that is true of it.
    """
    axis.text(
        0.5, 0.5, message, ha="center", va="center", fontsize=7, color="0.3",
        transform=axis.transAxes, wrap=True,
    )
    axis.set_xticks([])
    axis.set_yticks([])


def _panel(axis, candidate: dict, signal_limits: tuple[float, float]) -> None:
    """One bucket's raw median curve and interquartile band, on the signal's shared range."""
    statistics = candidate["severity_statistics"]
    severity = np.asarray(EXPECTED_SEVERITIES)
    median = [statistics[str(s)]["median"] for s in EXPECTED_SEVERITIES]
    q25 = [statistics[str(s)]["q25"] for s in EXPECTED_SEVERITIES]
    q75 = [statistics[str(s)]["q75"] for s in EXPECTED_SEVERITIES]
    axis.plot(severity, median, marker="o", linewidth=2)
    axis.fill_between(severity, q25, q75, alpha=0.22)
    axis.set_xlim(0, 5)
    axis.set_xticks(severity)
    axis.set_ylim(*signal_limits)


def _signal_figure(
    candidates: list[dict], signal: str, scheme: str, signal_limits: tuple[float, float]
):
    """One small multiple: every bucket of one scheme, for one signal, on one y-range.

    The axis labels go on the bottom row and the left column rather than on all ten panels.
    Every panel of the grid carries the same two axes, so ten copies of the same two strings is
    ink that pushes the panels smaller without telling the reader anything the edge labels do
    not.

    Deliberately not `sharey=True`. The shared range is this figure's central claim, and
    letting matplotlib propagate one axis's limits to the rest would make the claim true of the
    drawing while saying nothing about the code: each panel is given the range explicitly, so a
    panel that did not get it is a panel that shows it.
    """
    names = SCHEME_BUCKET_NAMES[scheme]
    rows, columns = SCHEME_LAYOUTS[scheme]
    figure, _ = plt.subplots(rows, columns, figsize=SCHEME_FIGURE_SIZES[scheme])
    drawn = {
        candidate["confidence_bin"]: candidate
        for candidate in candidates
        if candidate["signal"] == signal and candidate["bucket_scheme"] == scheme
    }
    # `figure.axes` rather than the array `subplots` returns: it is flat and row-major for
    # both layouts, so the bucket order does not need a branch on the number of rows.
    for axis, name in zip(figure.axes, names):
        axis.tick_params(labelsize=7)
        candidate = drawn.get(name)
        if candidate is None:
            axis.set_title(name, fontsize=8)
            _absent(axis, f"no drawable {scheme} {signal} candidate for {name}")
            continue
        _panel(axis, candidate, signal_limits)
        axis.set_title(f"{name}\n{_direction_label(candidate)}", fontsize=8)
    for axis in figure.axes[-columns:]:
        axis.set_xlabel(SEVERITY_AXIS_LABEL, fontsize=8)
    for axis in figure.axes[::columns]:
        axis.set_ylabel(SIGNAL_AXIS_LABELS[signal], fontsize=8)
    scope_note = f" at {DEPLOYABLE_SCORE_SCOPE}" if signal == DEPLOYABLE_SIGNAL else ""
    figure.suptitle(
        f"{SIGNAL_AXIS_LABELS[signal]}{scope_note} against blur severity, "
        f"one panel per confidence {scheme}\n"
        f"{FIGURE_SLICE} -- median and interquartile band of the raw, un-oriented score; "
        f"the y range is shared with the other {signal} figure",
        fontsize=8,
    )
    figure.tight_layout()
    return figure


def _corruption_figures(
    candidates: list[dict],
) -> tuple[dict[tuple[str, str], object], dict[str, list[float]]]:
    """The four figures, keyed as `PLOT_FILENAMES` is, and the two ranges they were drawn on.

    Every figure is built before any of them is written, so a signal with nothing drawable
    raises here with all four files still unwritten rather than after two of them exist.

    A signal with no drawable candidate is refused rather than given four empty panels. There
    is no range to put on the other signal's figures relative to, and a set of figures in which
    one signal is a grid of empty boxes is a bundle that looks complete and describes half a
    measurement.
    """
    chosen = _plot_candidates(candidates)
    limits: dict[str, list[float]] = {}
    for signal in PLOT_SCOPES:
        if not any(candidate["signal"] == signal for candidate in chosen):
            raise ValueError(
                f"no drawable {signal} candidate: the figures need "
                f"{DEPLOYABLE_MEMBERSHIP_MODE} {FILTERED_PADDING_MODE} {PLOT_AGGREGATION} "
                f"rows at score scope {PLOT_SCOPES[signal]!r}, with statistics at all six "
                "severities"
            )
        limits[signal] = list(shared_limits(chosen, signal))
    figures = {
        (signal, scheme): _signal_figure(chosen, signal, scheme, tuple(limits[signal]))
        for signal, scheme in PLOT_FILENAMES
    }
    return figures, limits


def write_corruption_plots(
    directory: str | Path, candidates: list[dict]
) -> dict[str, list[float]]:
    """Write the four figures into `directory` and return the y-ranges they were drawn on.

    The return value is `{"persistence": [lower, upper], "confidence": [lower, upper]}` -- the
    limits *applied* to the panels, not a second computation of them. The bundle writer records
    it in `summary.json`, where a reader compares a bucket's published numbers against the
    picture; if the recorded range were re-derived rather than carried, the two could disagree
    about which range the figures actually used and nothing would say so. Lists rather than
    tuples because that is what survives a JSON round trip unchanged.

    `directory` is written into as it is found: no staging and no atomic rename here. Staging
    belongs to whichever writer has the whole bundle to make appear at once; a second, per-file
    notion of "published" underneath it would only add a way for four PNGs to land one at a
    time in a directory that already claims to hold a finished result.
    """
    directory = Path(directory)
    figures, limits = _corruption_figures(candidates)
    for key, figure in figures.items():
        figure.savefig(directory / PLOT_FILENAMES[key], format="png", dpi=PLOT_DPI)
        plt.close(figure)
    return limits
