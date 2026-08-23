"""Four figures, and the span of what each one actually drew.

`figure_spans` is the key `summary.json` records these under, and `span` rather than `limit` is
the spec's word for three of the four: only the AUROC figure pins its panels, and the range
returned for the other three describes the data drawn rather than an axis imposed on it. The
one that *is* a limit is named `AUROC_LIMITS` and says so.


Panels are grouped by what shares a meaning, not by what fits. The anchor panels put reference
and responsive on one axis because they are the same quantity in the same units; the contrast
panels give each score method its own panel because a relative gap and a raw distance are not
comparable numbers; the AUROC panels share one 0-to-1 axis because AUROC means the same thing
everywhere, which is what makes a dashed confidence twin readable against its solid candidate.

Four decisions here look like presentation choices and are not.

**The plotted scores are raw and un-oriented.** `summarize_contrast_candidates` locks one `+1`
or `-1` per candidate so its AUROC has a single direction to rank in, and nothing in this
module ever applies it. A contrast that *falls* as blur rises is a real finding -- the feature
moves, and a deployment could read it downwards -- and a figure that multiplied by the
orientation would draw that candidate rising, which is the opposite of what was measured. The
one figure whose numbers are oriented is the AUROC one, because an AUROC is already a number
about a direction; the three that draw scores draw them as measured.

**The twelve panels come from the arm table, not from the candidates.** `panel_plan` is built
out of `ARMS` and `AGGREGATIONS`, so a run whose fourth arm produced nothing still has twelve
panels and the fourth row of each grid says what is missing. A grid sized to whatever arrived
would shrink to nine panels and read as a nine-cell experiment, which is the one thing a
reader counting panels must not be told.

**No range is shared across panels except the AUROC one.** Three of the four arms are scored
at `layer_2`, a raw mean-kNN distance, and the fourth at `combined`, a robust z-score against
the clean median; those are unrelated quantities, and one range covering both would flatten
whichever is smaller against the floor of its own panels. So each anchor, relationship and
contrast panel autoscales to what it holds, and the sharing the anchor figure does claim --
reference against responsive -- is sharing *within* a panel, which is exactly where the two
series are the same quantity in the same units. AUROC is the exception because it is unitless
and means the same thing in all twelve panels, so `AUROC_LIMITS` is applied to every one of
them and a twin at 0.55 in one panel is the same claim as a twin at 0.55 in another.

**An absence is drawn, never skipped.** A severity no image scored becomes a break in the line
rather than a segment drawn across it, and a panel with nothing at all to draw gets `_absent`'s
centred message instead of an empty frame -- an empty frame carrying ticks is indistinguishable
from a candidate that never moved. `severity_statistics` always has six entries, so a gap is
found by reading `count == 0` rather than by a key being absent; `auroc_by_severity` is `None`
as a whole for an unorientable candidate rather than a dictionary of `None`s, so it is tested
for as a whole.

Nothing here reads an artifact, and nothing here writes into the dictionaries it draws. It
takes the candidate, control and row structures Tasks 4 to 6 produce and writes PNGs into a
directory the caller already owns; staging that directory and recording the returned ranges
beside the figures belongs to the bundle writer above it, not here.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

# Drawn on a headless box, so the backend is pinned before pyplot is imported.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .contrast_analysis import CONTRAST_CANDIDATE_KEY, DEPLOYABLE_SIGNAL  # noqa: E402
from .contrast_controls import (  # noqa: E402
    CORRUPTED_SEVERITIES,
    REFERENCE_CONTROL_METHOD,
    RESPONSIVE_CONTROL_METHOD,
)
from .contrast_inputs import AGGREGATIONS, ARMS  # noqa: E402
from .contrast_scores import SCORE_METHODS  # noqa: E402
from .corruption_metrics import EXPECTED_SEVERITIES  # noqa: E402

PLOT_FILENAMES = {
    "anchor": "anchor_and_responsive_actual_distance.png",
    "relationship": "clean_anchor_relationship.png",
    "contrast": "contrast_scores_by_severity.png",
    "auroc": "auroc_by_blur_severity.png",
}
"""`actual_distance` in the first name for the reason `corruption_plots` uses it: that figure
draws the score as measured, as opposed to the oriented AUROCs the bundle publishes beside it.
The iteration order is the order the figures are built and written in."""

PLOT_DPI = 160

AUROC_LIMITS = [0.0, 1.0]
"""The one range this module applies rather than measures.

Applied to all twelve AUROC panels and returned verbatim. A range fitted to the AUROCs a run
happened to produce would make 0.62 look impressive in a figure whose worst candidate reached
0.60 and unremarkable in one whose best reached 0.95, and the chance line would sit in a
different place in every panel. `write_contrast_plots` returns `list(AUROC_LIMITS)` and not a
recomputation of the drawn values, so the number Task 8 records is the number the panels carry.
"""

CHANCE = 0.5
"""Where a detector that cannot tell clean from corrupted lands. Drawn on every AUROC panel
that drew anything, because a curve at 0.55 and a curve at 0.45 are the same distance from
useless and only a marked axis says so."""

METHOD_COLOURS = dict(
    zip(SCORE_METHODS, ("tab:blue", "tab:orange", "tab:green", "tab:red"))
)
"""One colour per score method, shared by the method's own curve and its confidence twin.

Shared on purpose: the twin is the control the method is measured against, and giving it its
own colour would turn eight curves into eight unrelated series instead of four pairs.
`CANDIDATE_STYLE` and `TWIN_STYLE` are what tell the two halves of a pair apart."""

ARM_COLOURS = dict(
    zip((arm.name for arm in ARMS), ("tab:blue", "tab:orange", "tab:green", "tab:red"))
)
AGGREGATION_STYLES = dict(zip(AGGREGATIONS, ("-", "--", ":")))
"""The contrast figure's twelve lines, encoded as four colours times three dashes rather than
as twelve legend entries. A twelve-row legend on a four-panel figure is taller than the panels.
"""

CANDIDATE_STYLE = "-"
TWIN_STYLE = "--"
CHANCE_STYLE = ":"
CHANCE_COLOUR = "0.35"

ANCHOR_COLOURS = {"reference": "0.35", "responsive": "tab:blue"}
ANCHOR_SERIES = ("reference", "responsive")
"""Reference first, so a panel that drew both has its reference at line index 0 whatever the
run contained. The reference is the greyer of the two because it is the baseline being cleared,
not the measurement."""

SCATTER_COLOUR = "tab:blue"
FIT_COLOUR = "tab:red"

GRID_FIGURE_SIZE = (13.5, 14.5)
"""Inches, for the three 4x3 grids. Four arms down and three summaries across, rather than
twelve panels in a strip: at one figure width a strip gives each panel a twelfth of the page
and the six severities land on top of each other, which is the axis these figures need."""

CONTRAST_FIGURE_SIZE = (16.0, 5.0)

EMPTY_SPAN = [0.0, 0.0]
"""What a figure that drew nothing reports. Not a range any panel was drawn on -- no panel was
drawn -- and deliberately degenerate rather than a plausible-looking `[0, 1]`, so a reader of
`summary.json` sees an empty measurement instead of a unit interval nobody measured."""

SEVERITY_AXIS_LABEL = "Blur severity"
AUROC_AXIS_LABEL = "AUROC (oriented, clean vs this severity)"
ANCHOR_AXIS_LABEL = "Raw un-oriented distance"

RELATIONSHIP_CAPTION = (
    "Severity-zero reference against responsive, one point per image, with the final robust "
    "line fitted on all clean images.\nThe published residual metrics are cross-fitted: every "
    "reported clean_residual comes from its image's fold line, not from this final line, which "
    "is the line a deployment would store."
)
"""Printed on the relationship figure because the figure shows the one line the numbers beside
it were *not* computed from. Without the sentence a reader measures a point's vertical distance
to the drawn line and gets a residual that is smaller than the published one by exactly the
amount of that image's own influence on the fit."""

ARM_TABLE = {arm.name: arm for arm in ARMS}
"""The arm table by name, for panel labels. Labels come from here rather than from the
`arm_family` and `score_scope` a candidate carries, so a panel cannot end up captioned with one
arm's provenance and drawn from another's -- and an empty panel still gets its full caption."""


def panel_plan(candidates: list[dict]) -> dict[str, list[tuple]]:
    """Which panels each figure has, derived from the arm table rather than from the data.

    Derived, so a figure cannot quietly shrink when a candidate fails to produce numbers: an
    arm with nothing to draw gets an empty panel that says so, and a reader counting twelve
    panels is counting the experiment rather than counting its successes.

    `candidates` is accepted and deliberately never read. It is in the signature because the
    plan is a property of *this run's* figures and a caller should not have to know that the
    run cannot change it; reading it is the mutation this function exists to prevent, and
    `test_the_panel_plan_ignores_which_candidates_exist` hands over an empty list to say so.

    Panels are tuples in every entry, including the one-field contrast entry, so a caller can
    zip a plan against `figure.axes` without branching on which figure it holds. Each figure
    gets its own list rather than three references to one, so a caller that edits the plan it
    was handed cannot edit the other two figures' plans as a side effect.
    """
    def arm_panels() -> list[tuple[str, str]]:
        return [(arm.name, aggregation) for arm in ARMS for aggregation in AGGREGATIONS]

    return {
        "anchor": arm_panels(),
        "relationship": arm_panels(),
        "contrast": [(method,) for method in SCORE_METHODS],
        "auroc": arm_panels(),
    }


def _by_key(items: list[dict]) -> dict[tuple, dict]:
    """Candidates or controls indexed by `(arm, signal, aggregation, method)`.

    The controls are indexed on the same four fields as the candidates, method included, so a
    caller that handed the candidate list over twice gets empty reference panels rather than a
    responsive curve drawn under a reference label: a candidate list holds no `raw_reference`
    method at all, which is precisely how `rank_contrast_candidates` keeps the control out of
    its ranking.
    """
    return {tuple(item[field] for field in CONTRAST_CANDIDATE_KEY): item for item in items}


def _scored(statistics: dict) -> bool:
    """Whether any severity of this candidate has an image in it.

    `count` rather than `median is not None`, because `count == 0` is the fact being read: a
    severity no image reached. The two agree on everything `summarize_contrast_candidates`
    builds, and `count` is the one that says why.
    """
    return any(statistics[severity]["count"] for severity in EXPECTED_SEVERITIES)


def _series(statistics: dict) -> tuple[list[float], list[float], list[float]]:
    """The median and both quartiles across the six severities, unscored severities as `nan`.

    `nan` rather than a dropped point, and this is the whole distinction between a gap and a
    missing panel. A dropped point would let matplotlib join severity 2 to severity 4 with a
    straight segment through a severity nothing was measured at; `nan` breaks the line there
    and leaves the rest of the curve in place, so the panel says "measured here, not there"
    instead of interpolating across the hole or vanishing entirely.
    """
    def value(entry: dict, field: str) -> float:
        number = entry[field]
        return float("nan") if entry["count"] == 0 or number is None else float(number)

    return tuple(
        [value(statistics[severity], field) for severity in EXPECTED_SEVERITIES]
        for field in ("median", "q25", "q75")
    )


def _span(values) -> list[float]:
    """`[min, max]` over everything a figure drew, ignoring the gaps.

    A list rather than a tuple because that is what survives a JSON round trip unchanged, and
    Task 8 records it verbatim. `nan` and `None` are dropped rather than propagated: they mark
    severities nothing was drawn at, and a span of `nan` would be a range no reader could use
    and no axis could carry.
    """
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite:
        return list(EMPTY_SPAN)
    return [min(finite), max(finite)]


def _absent(axis, message: str) -> None:
    """Say a panel had nothing to draw, rather than drawing an empty one.

    Follows the convention `corruption_plots` and `decile_reporting` set for the same hazard.
    An empty axis carrying ticks is indistinguishable from a candidate whose score never moved;
    this prints the reason across the panel and removes the ticks, so the only thing the panel
    can be read as saying is the only thing that is true of it.
    """
    axis.text(
        0.5, 0.5, message, ha="center", va="center", fontsize=6.5, color="0.3",
        transform=axis.transAxes, wrap=True,
    )
    axis.set_xticks([])
    axis.set_yticks([])


def _panel_title(arm_name: str, aggregation: str) -> str:
    """The panel's own identity: which arm, which summary, and the two facts that explain it.

    The family and the scope are on the panel rather than in the caption because they are what
    make two panels incomparable. A `differential` arm subtracts two responsive ranges, and a
    `combined` scope is a signed z-score rather than a distance -- a reader comparing the height
    of two curves needs both words in front of them, not once at the top of the page.
    """
    arm = ARM_TABLE[arm_name]
    return f"{arm_name}\n{aggregation} | {arm.family} | {arm.score_scope}"


def _grid(plan: list[tuple]):
    """A 4x3 figure and its axes paired with the plan, row-major and in the plan's order.

    `figure.axes` rather than the array `subplots` returns: it is already flat and row-major, so
    the pairing does not depend on the grid shape and cannot silently transpose.
    """
    figure, _ = plt.subplots(len(ARMS), len(AGGREGATIONS), figsize=GRID_FIGURE_SIZE)
    return figure, list(zip(figure.axes, plan))


def _severity_axis(axis) -> None:
    axis.set_xlim(EXPECTED_SEVERITIES[0], EXPECTED_SEVERITIES[-1])
    axis.set_xticks(list(EXPECTED_SEVERITIES))


def _edge_labels(figure, xlabel: str, ylabel: str) -> None:
    """One x label along the bottom row and one y label down the left column.

    Twelve copies of two strings is ink that pushes the panels smaller without telling the
    reader anything the edge labels do not.
    """
    columns = len(AGGREGATIONS)
    for axis in figure.axes[-columns:]:
        axis.set_xlabel(xlabel, fontsize=8)
    for axis in figure.axes[::columns]:
        axis.set_ylabel(ylabel, fontsize=8)


def _absence_note(available: dict, arm_name: str, aggregation: str) -> str:
    """Which of the two anchor series is missing, and whether it is absent or merely unscored.

    Two different absences, said differently. A series with no candidate at all never reached
    the summariser -- a fit that could not be produced, or an arm the run did not build -- while
    a series whose six severities all hold `count == 0` was summarised and found empty. Printing
    one message for both would leave a reader unable to tell a pipeline gap from a measurement
    that came back with nothing in it.
    """
    parts = []
    for name in ANCHOR_SERIES:
        item = available[name]
        if item is None:
            parts.append(f"no {name} candidate")
        elif not _scored(item["severity_statistics"]):
            parts.append(f"{name} scored no image at any severity")
    return f"{arm_name} / {aggregation}: " + "; ".join(parts)


def _anchor_figure(candidates: list[dict], controls: list[dict]):
    """The raw reference and responsive ranges of each arm, on one axis per panel.

    The reference comes from the `raw_reference` control and the responsive from the
    `raw_responsive` candidate, and neither is a contrast: `contrast_scores.raw_responsive`
    returns the responsive value unchanged, and `reference_control_rows` scores the reference
    value unchanged, so these two candidates' `severity_statistics` *are* the two ranges'
    statistics. Nothing is subtracted here, which is what makes this the figure that shows
    whether the anchor was worth subtracting.

    Both series share one axis per panel because they are the same quantity in the same units,
    and a reader has to be able to see that the responsive range moved further than the
    reference one -- which is unreadable if each series is fitted to its own scale. Panels do
    not share a range with each other; see this module's docstring for why.
    """
    indexed = _by_key(candidates)
    references = _by_key(controls)
    figure, panels = _grid(panel_plan(candidates)["anchor"])
    severities = np.asarray(EXPECTED_SEVERITIES, dtype=float)
    drawn: list[float] = []
    for axis, (arm_name, aggregation) in panels:
        axis.tick_params(labelsize=7)
        axis.set_title(_panel_title(arm_name, aggregation), fontsize=7)
        available = {
            "reference": references.get(
                (arm_name, DEPLOYABLE_SIGNAL, aggregation, REFERENCE_CONTROL_METHOD)
            ),
            "responsive": indexed.get(
                (arm_name, DEPLOYABLE_SIGNAL, aggregation, RESPONSIVE_CONTROL_METHOD)
            ),
        }
        drawable = {
            name: item for name, item in available.items()
            if item is not None and _scored(item["severity_statistics"])
        }
        if not drawable:
            _absent(axis, _absence_note(available, arm_name, aggregation))
            continue
        for name in ANCHOR_SERIES:
            item = drawable.get(name)
            if item is None:
                continue
            median, q25, q75 = _series(item["severity_statistics"])
            axis.plot(
                severities, median, marker="o", markersize=3, linewidth=1.6,
                color=ANCHOR_COLOURS[name], label=name,
            )
            axis.fill_between(
                severities, q25, q75, alpha=0.20, linewidth=0, color=ANCHOR_COLOURS[name],
            )
            drawn.extend(median + q25 + q75)
        if len(drawable) < len(available):
            axis.text(
                0.02, 0.02, _absence_note(available, arm_name, aggregation),
                transform=axis.transAxes, fontsize=5.5, color="0.3", va="bottom",
            )
        _severity_axis(axis)
    _edge_labels(figure, SEVERITY_AXIS_LABEL, ANCHOR_AXIS_LABEL)
    figure.legend(
        handles=[
            Line2D([], [], color=ANCHOR_COLOURS[name], marker="o", markersize=3,
                   label=f"{name} range: median and interquartile band")
            for name in ANCHOR_SERIES
        ],
        loc="lower center", ncol=2, fontsize=8, frameon=False,
    )
    figure.suptitle(
        "Raw un-oriented reference and responsive distance against blur severity, "
        "one panel per arm and scene summary\n"
        "Both ranges share one axis per panel; panels do not share a range with each other, "
        "because layer_2 distances and combined z-scores are different quantities. "
        "A break in a line is a severity no image scored.",
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 0.955))
    return figure, _span(drawn)


def _clean_points(rows: list[dict]) -> dict[tuple[str, str], tuple[list[float], list[float]]]:
    """Severity-zero reference and responsive per image, keyed by arm and scene summary.

    `raw_responsive` rows only, and that is what makes each image contribute exactly one point.
    All four methods of one series carry the same `reference` and `responsive` columns -- the
    method changes the `score`, not the two ranges it was built from -- so taking every method
    would draw each image four times over and make a six-image scatter look like twenty-four
    measurements. `raw_responsive` in particular because it is the one method no fit can fail to
    produce, so no arm loses its scatter to an unavailable residual or an excluded relative gap.

    Sorted by image so two callers holding the same rows in different orders get the same
    scatter, point for point and in the same order.
    """
    points: dict[tuple[str, str], dict[int, tuple[float, float]]] = {}
    for row in rows:
        if (
            row["severity"] != EXPECTED_SEVERITIES[0]
            or row["signal"] != DEPLOYABLE_SIGNAL
            or row["method"] != RESPONSIVE_CONTROL_METHOD
        ):
            continue
        points.setdefault((row["arm"], row["aggregation"]), {})[row["image_id"]] = (
            float(row["reference"]), float(row["responsive"])
        )
    return {
        key: (
            [by_image[image_id][0] for image_id in sorted(by_image)],
            [by_image[image_id][1] for image_id in sorted(by_image)],
        )
        for key, by_image in points.items()
    }


def _relationship_figure(rows: list[dict], fits: dict):
    """The clean relationship each residual is measured against, one panel per arm and summary.

    The line drawn is the *final* one, fitted on every clean image, because that is the line a
    deployment would store and the only one a reader can see the whole scatter against. Every
    published residual came from a fold line instead, and `RELATIONSHIP_CAPTION` says so on the
    figure: the vertical distance from a point to the drawn line is smaller than that image's
    reported residual by exactly the amount of its own influence on the fit.

    Each panel's axes are labelled with the two bucket names rather than with `reference` and
    `responsive`, because which bucket is which is the arm's identity and it differs by row.
    """
    points = _clean_points(rows)
    figure, panels = _grid(panel_plan([])["relationship"])
    drawn: list[float] = []
    for axis, (arm_name, aggregation) in panels:
        arm = ARM_TABLE[arm_name]
        axis.tick_params(labelsize=7)
        axis.set_title(_panel_title(arm_name, aggregation), fontsize=7)
        pair = points.get((arm_name, aggregation))
        if pair is None or not pair[0]:
            _absent(
                axis,
                f"{arm_name} / {aggregation}: no severity-zero "
                f"{RESPONSIVE_CONTROL_METHOD} rows",
            )
            continue
        reference, responsive = pair
        axis.scatter(
            reference, responsive, s=20, color=SCATTER_COLOUR, zorder=3, label="clean image"
        )
        drawn.extend(reference)
        drawn.extend(responsive)
        fit = fits.get((arm_name, DEPLOYABLE_SIGNAL, aggregation))
        line = fit["final_line"] if fit is not None else None
        if line is None:
            axis.text(
                0.02, 0.02, "no final clean line was fitted", transform=axis.transAxes,
                fontsize=6, color="0.3", va="bottom",
            )
        else:
            slope, offset = float(line[0]), float(line[1])
            edges = np.asarray([min(reference), max(reference)], dtype=float)
            fitted = offset + slope * edges
            axis.plot(edges, fitted, color=FIT_COLOUR, linewidth=1.5, label="final robust line")
            drawn.extend(fitted.tolist())
        axis.set_xlabel(f"{arm.reference_bin} (reference)", fontsize=6.5)
        axis.set_ylabel(f"{arm.responsive_bin} (responsive)", fontsize=6.5)
    figure.legend(
        handles=[
            Line2D([], [], color=SCATTER_COLOUR, marker="o", markersize=4, linestyle="none",
                   label="one clean image at severity 0"),
            Line2D([], [], color=FIT_COLOUR, linewidth=1.5,
                   label="final robust line, fitted on every clean image"),
        ],
        loc="lower center", ncol=2, fontsize=8, frameon=False,
    )
    figure.suptitle(RELATIONSHIP_CAPTION, fontsize=8)
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 0.945))
    return figure, _span(drawn)


def _contrast_figure(candidates: list[dict]):
    """The four score methods against severity, one panel each and twelve lines in every panel.

    One panel per method rather than one per arm, because a relative gap is bounded by plus and
    minus two while a raw gap is a distance: putting them on one axis makes the bounded one a
    flat line. Within a panel the twelve arm-and-summary curves are comparable in unit, with
    the standing caveat that the `combined` arm's scores are z-scores rather than distances,
    which the colour legend names.

    Raw and un-oriented, like the anchor figure and for the same reason: a contrast that falls
    with blur is a measurement, and the orientation belongs to the AUROC panels.
    """
    indexed = _by_key(candidates)
    figure, _ = plt.subplots(1, len(SCORE_METHODS), figsize=CONTRAST_FIGURE_SIZE)
    severities = np.asarray(EXPECTED_SEVERITIES, dtype=float)
    drawn: list[float] = []
    for axis, (method,) in zip(figure.axes, panel_plan(candidates)["contrast"]):
        axis.tick_params(labelsize=7)
        axis.set_title(method, fontsize=9)
        painted = False
        for arm in ARMS:
            for aggregation in AGGREGATIONS:
                candidate = indexed.get((arm.name, DEPLOYABLE_SIGNAL, aggregation, method))
                if candidate is None or not _scored(candidate["severity_statistics"]):
                    continue
                median, q25, q75 = _series(candidate["severity_statistics"])
                axis.plot(
                    severities, median, color=ARM_COLOURS[arm.name], linewidth=1.4,
                    linestyle=AGGREGATION_STYLES[aggregation],
                )
                axis.fill_between(
                    severities, q25, q75, alpha=0.08, linewidth=0,
                    color=ARM_COLOURS[arm.name],
                )
                drawn.extend(median + q25 + q75)
                painted = True
        if not painted:
            _absent(axis, f"no {DEPLOYABLE_SIGNAL} {method} candidate scored any severity")
            continue
        _severity_axis(axis)
        axis.set_xlabel(SEVERITY_AXIS_LABEL, fontsize=8)
    figure.axes[0].set_ylabel("Raw un-oriented contrast score", fontsize=8)
    figure.legend(
        handles=[
            Line2D([], [], color=ARM_COLOURS[arm.name], linewidth=1.4,
                   label=f"{arm.name} ({arm.score_scope})")
            for arm in ARMS
        ] + [
            Line2D([], [], color="0.3", linewidth=1.4,
                   linestyle=AGGREGATION_STYLES[aggregation], label=aggregation)
            for aggregation in AGGREGATIONS
        ],
        loc="lower center", ncol=4, fontsize=7, frameon=False,
    )
    figure.suptitle(
        "Raw un-oriented contrast score against blur severity, one panel per score method\n"
        "Median and interquartile band across images; one line per arm and scene summary. "
        "The four methods have different units, which is why they are four panels.",
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.14, 1.0, 0.94))
    return figure, _span(drawn)


def _auroc_figure(candidates: list[dict]):
    """Each method's oriented AUROC and its confidence twin, on one shared 0-to-1 axis.

    Solid is the persistence candidate and dashed is the confidence twin built from the same
    bucket pair, in one colour per method, because the question the panel answers is whether
    the persistence contrast beat the free one -- and that is a comparison between two curves,
    not a reading of either. `AUROC_LIMITS` on every panel is what makes the comparison survive
    being carried from one panel to the next.

    `auroc_by_severity` is `None` as a whole for an unorientable or incomplete candidate rather
    than a dictionary of `None`s, and `twin_auroc_by_severity` is `None` when no twin was
    attached at all, so both are tested for as whole objects. A method with neither is named in
    the panel's footnote instead of being drawn as a flat line at nothing.
    """
    indexed = _by_key(candidates)
    figure, panels = _grid(panel_plan(candidates)["auroc"])
    corrupted = list(CORRUPTED_SEVERITIES)
    for axis, (arm_name, aggregation) in panels:
        axis.tick_params(labelsize=7)
        axis.set_title(_panel_title(arm_name, aggregation), fontsize=7)
        # Applied before anything is drawn, and above the empty-panel guard below rather than
        # after it. The returned span claims that every panel of this figure carries
        # `AUROC_LIMITS`, and a panel that took the `continue` would instead be keeping
        # matplotlib's default range -- which is `(0.0, 1.0)` today, so the two coincide and
        # nothing in the drawing would say the claim had stopped being true. Move the declared
        # interval off the default and an empty panel would silently disagree with the number
        # `summary.json` records for it.
        axis.set_ylim(*AUROC_LIMITS)
        missing: list[str] = []
        painted = False
        for method in SCORE_METHODS:
            candidate = indexed.get((arm_name, DEPLOYABLE_SIGNAL, aggregation, method))
            if candidate is None:
                missing.append(f"{method}: no candidate")
                continue
            own = candidate["auroc_by_severity"]
            twin = candidate.get("twin_auroc_by_severity")
            if own is None:
                missing.append(f"{method}: no AUROC")
            else:
                axis.plot(
                    corrupted, [own[severity] for severity in corrupted],
                    color=METHOD_COLOURS[method], linestyle=CANDIDATE_STYLE,
                    marker="o", markersize=3, linewidth=1.5,
                )
                painted = True
            if twin is None:
                missing.append(f"{method}: no twin")
            else:
                axis.plot(
                    corrupted, [twin[severity] for severity in corrupted],
                    color=METHOD_COLOURS[method], linestyle=TWIN_STYLE,
                    marker="x", markersize=4, linewidth=1.2,
                )
                painted = True
        if not painted:
            _absent(
                axis,
                f"{arm_name} / {aggregation}: no AUROC and no confidence twin\n"
                + "; ".join(missing),
            )
            continue
        axis.axhline(CHANCE, color=CHANCE_COLOUR, linestyle=CHANCE_STYLE, linewidth=0.9)
        axis.set_xlim(corrupted[0], corrupted[-1])
        axis.set_xticks(corrupted)
        if missing:
            axis.text(
                0.02, 0.02, "; ".join(missing), transform=axis.transAxes,
                fontsize=5.5, color="0.3", va="bottom",
            )
    _edge_labels(figure, f"{SEVERITY_AXIS_LABEL} (corrupted only)", AUROC_AXIS_LABEL)
    figure.legend(
        handles=[
            Line2D([], [], color=METHOD_COLOURS[method], linestyle=CANDIDATE_STYLE,
                   marker="o", markersize=3, label=method)
            for method in SCORE_METHODS
        ] + [
            Line2D([], [], color="0.3", linestyle=TWIN_STYLE, marker="x", markersize=4,
                   label="confidence twin of the same method"),
            Line2D([], [], color=CHANCE_COLOUR, linestyle=CHANCE_STYLE,
                   label=f"chance ({CHANCE})"),
        ],
        loc="lower center", ncol=3, fontsize=8, frameon=False,
    )
    figure.suptitle(
        "Oriented AUROC against blur severity, one panel per arm and scene summary\n"
        "Solid is the persistence candidate, dashed its confidence twin in the same colour; "
        f"every panel spans {AUROC_LIMITS[0]} to {AUROC_LIMITS[1]} so the panels can be read "
        "against each other.",
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.955))
    return figure, list(AUROC_LIMITS)


def _contrast_figures(
    candidates: list[dict], controls: list[dict], rows: list[dict], fits: dict
) -> tuple[dict[str, object], dict[str, list[float]]]:
    """All four figures, keyed as `PLOT_FILENAMES` is, and the span each one covers.

    Every figure is built before any of them is written, so a directory that cannot be written
    to fails with all four files still absent rather than after two of them exist.

    The build itself is wrapped too, and that is not the same guarantee as
    `write_contrast_plots`'s `finally`. Three figures can be open when the fourth's builder
    raises, and those three are in no collection the caller could close; matplotlib would report
    them as a too-many-open-figures warning much later, in whatever code happened to be running
    then.

    Unlike `corruption_plots`, nothing here refuses to draw. That module raises when a signal
    has no drawable candidate because its four figures cover two signals and a bundle showing
    one of them is half a measurement. These four cover one experiment at four altitudes, the
    six-image fixture makes nothing deployable, and a run whose arms all failed still has to
    publish the twelve panels that say so -- which is what `_absent` is for.
    """
    figures: dict[str, object] = {}
    try:
        figures["anchor"], anchor = _anchor_figure(candidates, controls)
        figures["relationship"], relationship = _relationship_figure(rows, fits)
        figures["contrast"], contrast = _contrast_figure(candidates)
        figures["auroc"], _ = _auroc_figure(candidates)
    except BaseException:
        for figure in figures.values():
            plt.close(figure)
        raise
    return figures, {
        "anchor": anchor,
        "relationship": relationship,
        "contrast": contrast,
        "auroc": list(AUROC_LIMITS),
    }


def write_contrast_plots(
    directory: str | Path,
    *,
    candidates: list[dict],
    controls: list[dict],
    rows: list[dict],
    fits: dict,
) -> dict[str, list[float]]:
    """Write the four figures into `directory` and return the span each one covers.

    The return value is `{key: [lower, upper]}` over the same four keys as `PLOT_FILENAMES`,
    and it is what Task 8 records under `summary.json`'s `figure_spans`. `"auroc"` is
    `list(AUROC_LIMITS)` -- the interval every one of its twelve panels was given, including
    the panels that drew nothing, and not a second computation over the curves. The other three
    are `[min, max]` over every value that figure drew; their panels autoscale, for the reason
    this module's docstring gives, so those three are the span of the drawn data rather than a
    limit applied to any one panel. The spec's word is `span` for exactly that reason, and the
    report must not call the three of them limits. Task 8 records the dictionary verbatim and
    never recomputes it, which is what stops the recorded span and the drawn one disagreeing
    with nothing to say so. Lists rather than tuples because that is what survives a JSON round
    trip unchanged.

    Nothing here writes into `candidates`, `controls`, `rows` or `fits`. A candidate dictionary
    is the one object the ranking, the CSV, the report and these figures all hold, so a
    plotting-time repair -- a defaulted orientation, a filled-in `None` -- would be published by
    three writers that never asked for it.

    `directory` is written into as it is found: no staging and no atomic rename here. Staging
    belongs to whichever writer has the whole bundle to make appear at once; a second, per-file
    notion of "published" underneath it would only add a way for four PNGs to land one at a
    time in a directory that already claims to hold a finished result.
    """
    directory = Path(directory)
    figures, spans = _contrast_figures(candidates, controls, rows, fits)
    # `finally`, not a close after each `savefig`: the whole set is open before the first file
    # is written, so a write that fails part-way -- an unwritable or absent directory is the
    # obvious one -- would otherwise leave the untried figures open.
    try:
        for key, figure in figures.items():
            figure.savefig(directory / PLOT_FILENAMES[key], format="png", dpi=PLOT_DPI)
    finally:
        for figure in figures.values():
            plt.close(figure)
    return spans
