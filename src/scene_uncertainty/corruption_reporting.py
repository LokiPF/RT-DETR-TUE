"""One row per candidate out of the corruption score table, and one bundle out of the rows.

`analyze_corruption_sensitivity` emits 3,060 rows an image -- 34 selections, 15 signal/scope/
summary rows each, at six severities -- so a 250-image tuning run is 765,000 rows. This module
collapses that twice: first to one curve per candidate and image, then to one row per candidate,
and finally it sorts the candidates that pass every deployment gate and publishes the whole run
as one directory of eight files.

Those are two layers and they stay separable. `summarize_candidates` reads no file and writes
none: it takes the rows and hands back three lists, which is what makes every collapse below
testable without a directory. `write_corruption_report` is the only function here that touches
the filesystem, and it computes nothing of its own -- it lays out what the collapse produced,
what `corruption_plots` drew, and what `render_easy_report` said about both.

Every collapse below has a plausible wrong version that still produces a complete, internally
consistent table. The reasons the right versions are the right ones:

**Strength is the median of the per-image absolute Spearmans, and that is not the absolute
value of the median signed Spearman.** Two images with opposite perfect trends, `+1` and `-1`,
have a median signed Spearman of `0.0` and a median absolute Spearman of `1.0`. Both numbers
are true and they say different things: the first says the candidate has no consistent
direction across scenes, the second says corruption moves this feature hard in every scene it
was measured on. `median_signed_spearman` and `median_absolute_spearman` are therefore two
published fields computed from the same per-image array by two different reductions, and
neither is ever derived from the other.

**Absolute Spearman and AUROC answer different questions, and only one of them ranks.**
Absolute Spearman asks whether corruption moves the feature *within* one image, where the
scene's own baseline distance cancels out. AUROC asks whether the raw oriented score ranks a
corrupted image above an unrelated clean one *across* scenes with unrelated baselines. A
candidate can be near-perfect on the first and at chance on the second, so `macro_auroc` is the
primary ranking key and `median_absolute_spearman` is a sensitivity diagnostic that must never
select a candidate on its own -- which is exactly why it enters the sort only as the first
tie-break under the macro AUROC.

**Flat and unmeasured are different facts, and flat counts in the denominator.** A complete
curve of six identical finite scores is a measurement that found no trend: it is `flat`, its
correlations are a documented `0.0`, and it counts in `measured_count` and therefore in the
denominator of `dominant_direction_fraction`. A curve with a missing severity or a non-finite
score produced no measurement at all: it is `unmeasured`, its correlations stay `None`, and it
never becomes a zero. Folding the second into the first would let a candidate that failed to
produce scores read as one that produced scores showing nothing, and would inflate the
dominant-direction fraction of every candidate that lost images -- `max(positive, negative)`
over a shrunken denominator is larger, not smaller.

**Orientation is chosen once per candidate and then locked.** `choose_orientation` is called
once, on that candidate's own per-image signed trends, and the integer it returns is what every
one of the five AUROCs and every oriented curve check is read through. Never per severity, per
image or per metric: re-deciding downstream turns noise into apparent signal, because picking
the better of two directions at each severity puts a pure-noise candidate at or above 0.5 five
times out of five. Persistence and its matched confidence control are *separate candidates* --
they differ in `signal` and `score_scope`, both of which are in `CANDIDATE_KEY` -- so each
chooses its own orientation from its own rows and neither is ever re-oriented to make a
comparison come out. A candidate `choose_orientation` returns `None` for keeps its `None`, and
publishes no curve checks and no AUROCs rather than a plausible set computed from a guess.

**AUROC here is a ranking statistic, never a probability.** `0.5` is chance-level ranking and
`1.0` is perfect ranking. A value below `0.5` means that severity ranks the other way under the
candidate's locked orientation, which is a finding about the candidate and is published as-is.
Nothing in this module reads, names or reports an AUROC as a probability that an image is
corrupted.

**The ranking is a tuning ranking, not a hypothesis test.** Every candidate in it, and every
candidate's orientation, was selected on the same 250 tuning images that were then used to
describe it. `deployable` means "passed every gate on the tuning run", and nothing here
supports a claim about held-out data.

**A bundle is eight files or it is nothing.** `write_corruption_report` stages the whole
directory beside its destination and renames it into place only after every table, figure,
summary and paragraph exists and the file set has been checked; a failure removes the staging
directory and leaves no output. The alternative -- eight files published as they are produced
-- has a state in which the report describes a figure that is not there, and nothing in the
directory says which of the eight are missing.

**An empty ranking is an outcome, not an error.** Fewer than `FULL_TUNING_IMAGE_COUNT` images
is every synthetic and every diagnostic run, and each of them still gets its eight files, its
four figures and a report that says plainly that no candidate passed the coverage gate. What
never happens is the gate widening to fit the run it is given.

Three implementation decisions that look arbitrary and are not:

*Candidates are grouped on the recorded `bucket_scheme`* (it is in `ROW_KEY` and in
`CANDIDATE_KEY`), never on the fact that `decile_00_10` and `quintile_00_20` happen to start
with different words. `decile_reporting.GROUP_KEYS` carries no scheme, which is right for the
command it serves -- that one cuts deciles and nothing else -- but pointing it at these rows
would separate the two resolutions only by the accident that their bin names are different
strings. That accident would make the names a second definition of what the bins are, free to
drift from `decile_scoring.BUCKET_NAMES`, which is the one that decides.

*Each image's rows are sorted by severity before any curve is read.*
`corruption_metrics.complete_trend_metrics` validates the severity axis it is handed and
refuses one that is not `EXPECTED_SEVERITIES` in order, but `oriented_curve_metrics` takes a
bare sequence and reads index *i* as severity *i* without checking. Honouring that precondition
is this module's job and not that function's, so `_ordered_curve` exists to do it in one place
and both calls are fed from its output.

*Every candidate carries two views of the same per-severity diagnostics.* The nested maps
(`auroc_by_severity`, `severity_statistics`, `coverage_by_severity`,
`selected_count_by_severity`, `median_clean_overlap_by_severity`) are what `summary.json` and
the plots read; the flattened scalar columns beside them (`auroc_severity_1`,
`score_variance_severity_0`, ...) are what `candidate_metrics.csv` reads, because a nested
dictionary in a CSV cell is a string nobody can query. They are built together here rather than
re-derived by the writer, so the CSV projection is "drop every dictionary-valued key" and
cannot invent a column name of its own.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .corruption_metrics import (
    EXPECTED_SEVERITIES,
    choose_orientation,
    complete_trend_metrics,
    oriented_curve_metrics,
    severity_aurocs,
)
from .decile_scoring import CONFIDENCE_SCOPE, PRIMARY_SCORE_SCOPE


ROW_KEY = (
    "image_id", "severity", "signal", "bucket_scheme", "confidence_bin",
    "membership_mode", "padding_mode", "aggregation", "score_scope",
)
"""What makes two corruption score rows measurements of different things.

The published decile experiment's row key plus `bucket_scheme`, which is the nine-field
contract every row of this command carries. The scheme is not a tie-breaker -- `decile_00_10`
and `quintile_00_20` already differ in `confidence_bin` -- it is in the key because that
uniqueness is a property of how the bins were named rather than a guarantee, and because
separating the two resolutions *is* the comparison this command exists to make.
"""

CANDIDATE_KEY = tuple(field for field in ROW_KEY if field not in ("image_id", "severity"))
"""What identifies one candidate: a row key with the scene coordinates taken out.

Derived from `ROW_KEY` rather than re-typed, so a field added to the row contract cannot start
pooling two candidates into one by being absent here. Note what this makes separate:
persistence at `layer_2` and its matched confidence control differ in `signal` and
`score_scope`, so they are two candidates with two independently chosen orientations, and the
`dynamic` and `frozen` twins of one bin are two more.
"""

SELECTION_KEY = tuple(
    field for field in ROW_KEY if field not in ("signal", "aggregation", "score_scope")
)
"""What identifies one `score_selection` call rather than one row of it.

Every row sharing this key was scored over a single index tensor, which is the experiment's
fairness claim: persistence and its confidence control summarise one population, not two
populations that resemble each other. `score_selection` can only guarantee that within a call,
so `validate_selected_queries` checks it across the table before the query IDs are dropped.
"""

PER_SCENE_KEYS = (
    *ROW_KEY,
    "source_partition", "score", "selected_count", "clean_overlap",
    "fully_measured", "signed_spearman", "absolute_spearman", "direction",
)
"""The columns of `per_scene.csv`, one row per candidate, image and severity.

Deliberately *not* six-severity-wide: the real table is 765,000 rows, the same count as the raw
score table, and a reader who wants a curve gets it by filtering on a candidate and an image.
The four image-level trend fields are repeated on all six of an image's rows so that any single
row reads on its own without a join, and `selected_query_ids` is deliberately absent -- see
`validate_selected_queries` for what replaces it.
"""

FULL_TUNING_IMAGE_COUNT = 250
"""The tuning partition's image count, and the only run size a candidate may be called
deployable on. A candidate measured on 249 images is a different measurement, not a slightly
smaller one, and the gate excludes it rather than annotating it."""

DEPLOYABLE_SIGNAL = "persistence"
"""The confidence control is the thing persistence is measured against, not a second runner.
It is summarised and published in full; it is only the *ranking* it stays out of."""

CONTROL_SIGNAL = "confidence"
"""The signal name `score_selection` writes for the matched confidence control.

Spelled here rather than reused from `decile_scoring.CONFIDENCE_SCOPE`, which happens to be the
same string for a different reason: the control has no decoder layer to be scoped to, so its
*scope* field carries its signal name. `corruption_plots.CONFIDENCE_SIGNAL` says the same thing
for the figures and is not imported here, because that module imports this one."""

DEPLOYABLE_MEMBERSHIP_MODE = "dynamic"
"""Frozen membership reuses severity zero's bins, and a naturally corrupted image has no paired
clean version to build them from -- so a frozen result is a diagnostic about how far
fingerprints moved, not a policy anyone can run on a single image."""

FILTERED_PADDING_MODE = "filtered"
UNFILTERED_PADDING_MODE = "unfiltered"
"""The unfiltered rows are the padding *sensitivity control*: they score the padded decoder
placeholders the primary analysis removes. They are summarised and paired, never ranked."""

DEPLOYABLE_SCORE_SCOPE = PRIMARY_SCORE_SCOPE
"""Layer 2, imported rather than re-spelled. `decile_scoring` records the reason it is the
design's primary scope and names it once; a literal here would be a second place for the
primary scope to be decided, and a ranking that quietly promoted whichever scope won is exactly
what naming it once prevents."""

# Field name inside a nested per-severity map -> the column prefix `_flatten_fields` turns into
# `<prefix>_severity_<n>`. These three mappings are the whole of the nested-to-flat translation:
# every per-severity diagnostic is published once as a nested map, for `summary.json` and the
# plots, and once as scalar columns, for `candidate_metrics.csv`, which cannot hold a dictionary
# in a cell. Naming the pairing here rather than in the writer is what keeps the CSV projection
# down to "drop every dictionary-valued key".
SEVERITY_STATISTIC_COLUMNS = {
    "count": "score_count",
    "mean": "score_mean",
    "variance": "score_variance",
    "median": "score_median",
    "q25": "score_q25",
    "q75": "score_q75",
}

COVERAGE_COLUMNS = {
    "image_count": "image_count",
    "finite_count": "finite_count",
    "clean_count": "clean_count",
    "corrupted_count": "corrupted_count",
}

SELECTED_COUNT_COLUMNS = {
    "min": "selected_count_min",
    "median": "selected_count_median",
    "max": "selected_count_max",
}

_EMPTY_STATISTICS = {
    "count": 0, "mean": None, "variance": None, "median": None, "q25": None, "q75": None,
}
"""A severity with no finite score, written with the same keys and `None` where a number would
be. `0.0` would be a measurement of an empty group and `nan` cannot be written to JSON at all,
so the shape stays rectangular and the absence stays visible."""


def validate_rows(rows: list[dict]) -> None:
    """Refuse a table with an incomplete row or a repeated row key, before anything is grouped.

    A missing key field is not a missing column -- it is a row that cannot be filed. Grouping it
    would raise somewhere further in with a message naming a dictionary and not a row number.

    A duplicated `ROW_KEY` is the failure that leaves no trace at all. `summarize_candidates`
    files rows into a `{candidate: {image: {severity: row}}}` index, so the second copy
    *replaces* the first: the curve is built from whichever copy came last in the table, the
    discarded measurement is never mentioned, and `per_scene.csv` comes out one row shorter than
    the table it was built from with nothing saying which row went missing. Two concatenated
    result tables are the obvious way to arrive here and nothing about them looks wrong, which
    is why this refuses the table rather than trusting last-write-wins to be harmless.
    """
    seen: set[tuple] = set()
    for index, row in enumerate(rows):
        missing = [field for field in ROW_KEY if field not in row]
        if missing:
            raise ValueError(f"score row {index} is missing fields {missing}")
        key = tuple(row[field] for field in ROW_KEY)
        if key in seen:
            raise ValueError(f"duplicate corruption score row key: {key}")
        seen.add(key)


def validate_selected_queries(rows: list[dict]) -> None:
    """Every row of one selection must report the same selected query IDs.

    This is the check that lets the ID list be dropped. `per_scene.csv` keeps `selected_count`
    and `clean_overlap` as ordinary numeric columns and never serialises the ID list into a
    cell, because a list in a CSV cell is a string that no reader can filter, join or sum -- and
    on the real table it is also most of the file's bytes. But the IDs carry a claim that the
    counts cannot: that persistence and its confidence control, and all three scene summaries,
    were summarised over *one* population. `score_selection` guarantees that within a single
    call and can check nothing across calls, so it is verified here, once, and then the lists
    are left behind.

    A row without the field is refused rather than skipped. A check that quietly passes on the
    rows it cannot see is not a check, and the producer writes the field on every row.
    """
    seen: dict[tuple, tuple] = {}
    for index, row in enumerate(rows):
        if "selected_query_ids" not in row:
            raise ValueError(f"score row {index} is missing 'selected_query_ids'")
        key = tuple(row[field] for field in SELECTION_KEY)
        ids = tuple(row["selected_query_ids"])
        first = seen.setdefault(key, ids)
        if first != ids:
            raise ValueError(
                f"rows of one selection report different selected query IDs: {key}; "
                "persistence and its confidence control must summarise one population"
            )


def _ordered_curve(by_severity: dict) -> tuple[list[int], list[float]]:
    """One image's severities and scores, sorted by severity, as two aligned lists.

    The sort is the point, and it is stated here rather than left to the order the rows arrived
    in. `complete_trend_metrics` would catch a scrambled axis, because it is handed the
    severities and refuses any axis that is not `EXPECTED_SEVERITIES` in order -- but
    `oriented_curve_metrics` is handed the scores alone and reads index *i* as severity *i*
    without checking, so a descending curve fed to it in input order reports an adjacent
    consistency of `0.0` and a `max_blur_above_clean` of `False` for a curve that rises the
    whole way. Both calls are fed from this one function so the two cannot disagree about which
    score belongs to which severity.
    """
    severities = sorted(by_severity)
    return severities, [float(by_severity[severity]["score"]) for severity in severities]


def _statistics(values: np.ndarray) -> dict:
    """The raw scene score distribution at one severity, with a *population* variance.

    `ddof=0` because these values are the whole population being described -- every image the
    candidate scored at this severity -- and not a sample drawn from a larger one. The
    difference is not cosmetic at small counts: on two scores it is a factor of two.
    """
    if not values.size:
        return dict(_EMPTY_STATISTICS)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "variance": float(np.var(values, ddof=0)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25, method="linear")),
        "q75": float(np.quantile(values, 0.75, method="linear")),
    }


def _flatten_scalars(prefix: str, by_severity: dict) -> dict:
    return {f"{prefix}_severity_{key}": value for key, value in by_severity.items()}


def _flatten_fields(columns: dict[str, str], by_severity: dict) -> dict:
    return {
        f"{prefix}_severity_{key}": entry[field]
        for field, prefix in columns.items()
        for key, entry in by_severity.items()
    }


def _fraction(count: int, total: int) -> float | None:
    """`None` rather than a zero when there is nothing to divide by: an empty denominator is
    "this was not measured", and `0.0` reads as "measured, and none of them"."""
    return float(count) / total if total else None


def _candidate_metrics(
    candidate: tuple, images: dict, scenes: list[dict], expected_image_count: int
) -> dict:
    """One candidate's row: the direction it is read in, how strongly, and whether it deploys.

    `scenes` holds one entry per image, already sorted by severity and already run through
    `complete_trend_metrics`. `images` is the same rows keyed by image and severity, kept
    because the per-severity diagnostics -- selected counts, clean overlaps, the raw score
    distribution -- are properties of the rows rather than of the curves.

    The two denominators are different on purpose and both are published. `measured_fraction`
    and `missing_fraction` are over `image_count`, the images this candidate actually has rows
    for, so a candidate that scored 240 images reports 240 in the denominator rather than
    silently borrowing the run's 250; `expected_image_count` is published beside it, so an
    image the candidate never scored at all is visible as the gap between the two numbers
    rather than hidden inside one of them. `positive_fraction`, `negative_fraction` and
    `flat_fraction` are over `measured_count`, matching `dominant_direction_fraction`, because
    "which way do the measured images move?" cannot have unmeasured images in its denominator.

    `dominant_direction_fraction` is exactly `max(positive_fraction, negative_fraction)`. It is
    published under its own name because it is a ranking key and a reader of the sort order
    should not have to reconstruct it from two other columns.
    """
    score_groups: dict[int, list[float]] = {severity: [] for severity in EXPECTED_SEVERITIES}
    present: dict[int, int] = {severity: 0 for severity in EXPECTED_SEVERITIES}
    selected: dict[int, list[int]] = {severity: [] for severity in EXPECTED_SEVERITIES}
    overlaps: dict[int, list[float]] = {severity: [] for severity in EXPECTED_SEVERITIES}
    for by_severity in images.values():
        for severity, row in by_severity.items():
            # A severity outside the blur ladder is left out of every per-severity diagnostic
            # rather than opening a seventh column. It cannot reach a ranked candidate either
            # way: `complete_trend_metrics` refuses any axis that is not `EXPECTED_SEVERITIES`
            # in order, so the image it belongs to is `unmeasured` and the coverage gates fail.
            if severity not in score_groups:
                continue
            present[severity] += 1
            selected[severity].append(int(row["selected_count"]))
            overlaps[severity].append(float(row["clean_overlap"]))
            score = float(row["score"])
            if np.isfinite(score):
                score_groups[severity].append(score)

    measured = [scene for scene in scenes if scene["fully_measured"]]
    signed = np.asarray([scene["signed_spearman"] for scene in measured], dtype=float)
    orientation = choose_orientation(signed)
    positive_count = int(np.sum(signed > 0))
    negative_count = int(np.sum(signed < 0))
    flat_count = int(np.sum(signed == 0))
    measured_count = len(measured)
    image_count = len(scenes)
    dominant = (
        max(positive_count, negative_count) / measured_count if measured_count else None
    )

    # `orientation is None` also covers "no measured image": `choose_orientation` has nothing
    # finite to take a median of there and says so. Every measured image carries six finite
    # scores, so each `oriented_curve_metrics` call comes back with two numbers and not `None`.
    if orientation is None:
        adjacent = None
        max_blur_rate = None
    else:
        curves = [oriented_curve_metrics(scene["scores"], orientation) for scene in measured]
        adjacent = float(np.mean([curve["adjacent_consistency"] for curve in curves]))
        max_blur_rate = float(np.mean([curve["max_blur_above_clean"] for curve in curves]))

    # `severity_aurocs` refuses an empty group rather than scoring one, and severity 0 is the
    # clean side of all five comparisons, so every severity has to have something in it before
    # any AUROC exists at all. An unorientable or under-covered candidate publishes five `None`s
    # and no macro, which is the shape a plot and a CSV column can both read.
    complete_groups = all(score_groups[severity] for severity in EXPECTED_SEVERITIES)
    if orientation is not None and complete_groups:
        aurocs, macro_auroc = severity_aurocs(score_groups, orientation)
        auroc_by_severity = {
            str(severity): float(aurocs[severity]) for severity in EXPECTED_SEVERITIES[1:]
        }
    else:
        auroc_by_severity = {str(severity): None for severity in EXPECTED_SEVERITIES[1:]}
        macro_auroc = None

    severity_statistics = {
        str(severity): _statistics(np.asarray(score_groups[severity], dtype=float))
        for severity in EXPECTED_SEVERITIES
    }
    coverage_by_severity = {
        str(severity): {
            "image_count": present[severity],
            "finite_count": len(score_groups[severity]),
            # Both sides of the comparison this severity takes part in, so an image that
            # dropped out of one of them is visible as a difference between two published
            # numbers rather than as an AUROC quietly computed over unequal groups. At severity
            # 0 the two are the same number, because severity 0 *is* the clean group and is
            # never the corrupted side of anything.
            "clean_count": len(score_groups[0]),
            "corrupted_count": len(score_groups[severity]),
        }
        for severity in EXPECTED_SEVERITIES
    }
    selected_count_by_severity = {
        str(severity): {
            "min": int(min(selected[severity])) if selected[severity] else None,
            "median": float(np.median(selected[severity])) if selected[severity] else None,
            "max": int(max(selected[severity])) if selected[severity] else None,
        }
        for severity in EXPECTED_SEVERITIES
    }
    median_clean_overlap_by_severity = {
        str(severity): float(np.median(overlaps[severity])) if overlaps[severity] else None
        for severity in EXPECTED_SEVERITIES
    }

    labels = dict(zip(CANDIDATE_KEY, candidate))
    deployable = bool(
        expected_image_count == FULL_TUNING_IMAGE_COUNT
        and labels["membership_mode"] == DEPLOYABLE_MEMBERSHIP_MODE
        and labels["padding_mode"] == FILTERED_PADDING_MODE
        and labels["signal"] == DEPLOYABLE_SIGNAL
        and labels["score_scope"] == DEPLOYABLE_SCORE_SCOPE
        and measured_count == FULL_TUNING_IMAGE_COUNT
        and all(
            len(score_groups[severity]) == FULL_TUNING_IMAGE_COUNT
            for severity in EXPECTED_SEVERITIES
        )
        and orientation is not None
    )

    return {
        **labels,
        "expected_image_count": int(expected_image_count),
        "image_count": image_count,
        "measured_count": measured_count,
        "missing_count": image_count - measured_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "flat_count": flat_count,
        "measured_fraction": _fraction(measured_count, image_count),
        "missing_fraction": _fraction(image_count - measured_count, image_count),
        "positive_fraction": _fraction(positive_count, measured_count),
        "negative_fraction": _fraction(negative_count, measured_count),
        "flat_fraction": _fraction(flat_count, measured_count),
        "dominant_direction_fraction": dominant,
        "median_signed_spearman": float(np.median(signed)) if signed.size else None,
        "median_absolute_spearman": (
            float(np.median(np.abs(signed))) if signed.size else None
        ),
        "orientation": orientation,
        "oriented_adjacent_consistency": adjacent,
        "max_blur_above_clean_rate": max_blur_rate,
        "macro_auroc": macro_auroc,
        "deployable": deployable,
        # Filled in by `_pair_padding_modes` once every candidate exists; a candidate with no
        # twin keeps these, which is what says "not compared" rather than "compared and equal".
        "padding_counterpart_padding_mode": None,
        "padding_paired_score_count": 0,
        "padding_changed_score_count": 0,
        "padding_median_unfiltered_minus_filtered_score": None,
        **_flatten_scalars("auroc", auroc_by_severity),
        **_flatten_fields(SEVERITY_STATISTIC_COLUMNS, severity_statistics),
        **_flatten_fields(COVERAGE_COLUMNS, coverage_by_severity),
        **_flatten_fields(SELECTED_COUNT_COLUMNS, selected_count_by_severity),
        **_flatten_scalars("median_clean_overlap", median_clean_overlap_by_severity),
        "auroc_by_severity": auroc_by_severity,
        "severity_statistics": severity_statistics,
        "coverage_by_severity": coverage_by_severity,
        "selected_count_by_severity": selected_count_by_severity,
        "median_clean_overlap_by_severity": median_clean_overlap_by_severity,
    }


def _padding_difference(unfiltered: dict, filtered: dict) -> dict:
    """How much removing the padded decoder placeholders moved the raw score, image by image.

    A raw score difference is meaningful here and nowhere else in this module: the two sides are
    the same signal at the same scope under the same scene summary, differing only in which
    queries the selection contained, so subtracting one from the other is a measurement of the
    mask. (Persistence distance and `1 - confidence` share no unit and are never differenced.)

    Always signed as unfiltered minus filtered, and written onto *both* members of the pair, so
    the sign means the same thing whichever row a reader has in front of them.

    `padding_changed_score_count` is what keeps a median of `0.0` readable. An image with no
    padded tail selects the identical queries either way, so the control is a no-op on it by
    construction -- on the pilot that is most of the run -- and without a count of the scores
    the mask actually moved, "no difference" and "no opportunity for a difference" publish the
    same number.
    """
    differences: list[float] = []
    changed = 0
    for image_id, by_severity in sorted(unfiltered.items()):
        twin_severities = filtered.get(image_id)
        if twin_severities is None:
            continue
        for severity, row in sorted(by_severity.items()):
            twin = twin_severities.get(severity)
            if twin is None:
                continue
            left, right = float(row["score"]), float(twin["score"])
            if not (np.isfinite(left) and np.isfinite(right)):
                continue
            differences.append(left - right)
            if left != right:
                changed += 1
    return {
        "padding_paired_score_count": len(differences),
        "padding_changed_score_count": changed,
        "padding_median_unfiltered_minus_filtered_score": (
            float(np.median(differences)) if differences else None
        ),
    }


def _pair_padding_modes(by_key: dict[tuple, dict], grouped: dict[tuple, dict]) -> None:
    """Pair each unfiltered candidate with the twin that differs from it in `padding_mode` alone.

    Every other field of `CANDIDATE_KEY` has to match -- signal, scheme, bin, membership mode,
    scene summary and scope -- because the comparison is a measurement of the padding mask and
    nothing else, and a pair that also differed in, say, the scene summary would attribute that
    difference to the mask. In practice this only ever lands on a scheme's lowest bucket,
    because that is the only bin `analyze_corruption_sensitivity` scores unfiltered; the pairing
    is written as "all fields but one" rather than as "the lowest bucket" so that it stays a
    statement about what is being compared rather than about what the producer happens to emit.

    Mutates the candidate rows in place. Pairing cannot begin until every candidate exists, and
    the alternative -- a second pass that rebuilds the rows -- would be a second place the
    candidate shape is written down.
    """
    padding_index = CANDIDATE_KEY.index("padding_mode")
    for key, candidate in by_key.items():
        if candidate["padding_mode"] != UNFILTERED_PADDING_MODE:
            continue
        twin_key = (
            *key[:padding_index], FILTERED_PADDING_MODE, *key[padding_index + 1:]
        )
        twin = by_key.get(twin_key)
        if twin is None:
            continue
        difference = _padding_difference(grouped[key], grouped[twin_key])
        candidate.update(
            difference, padding_counterpart_padding_mode=FILTERED_PADDING_MODE
        )
        twin.update(difference, padding_counterpart_padding_mode=UNFILTERED_PADDING_MODE)


def _ranking_sort_key(candidate: dict) -> tuple:
    """Macro AUROC first, then strength, dominance and curve consistency, then the key itself.

    Descending on all four metrics and ascending on `CANDIDATE_KEY`. The metrics come first and
    in that order because macro AUROC is the deployment question -- does the oriented raw score
    separate corrupted scenes from clean ones -- and the three below it break its ties with
    progressively weaker evidence: how hard corruption moves the feature, how consistently it
    moves it the same way, and how orderly the curve is once oriented.

    The candidate key is the last term rather than decoration. Without it, two candidates with
    identical statistics keep whatever order the input happened to have, and a report sentence
    naming "the best candidate" would change between two runs over the same data.

    No `None` handling, and that is deliberate: `rank_deployable_candidates` only ever sorts
    rows whose `deployable` gate already established a locked orientation, 250 measured images
    and six full severity groups, so all four values are finite floats. A defensive fallback
    here would silently rank a candidate that should not have reached this function.
    """
    return (
        -candidate["macro_auroc"],
        -candidate["median_absolute_spearman"],
        -candidate["dominant_direction_fraction"],
        -candidate["oriented_adjacent_consistency"],
        *(candidate[field] for field in CANDIDATE_KEY),
    )


def rank_deployable_candidates(candidates: list[dict]) -> list[dict]:
    """The candidates that passed every gate, in deployment order.

    Filters on the `deployable` flag each candidate already carries rather than re-testing the
    gates, so there is one place that decides what deployable means and no way for the ranking
    and the flag to disagree. Everything else -- the frozen twins, the unfiltered padding
    controls, the secondary layer scopes, the matched confidence controls -- stays in
    `candidate_metrics.csv` and `summary.json` in full and is simply absent from this list. A
    gate that annotated instead of excluding would let a candidate that lost images outrank one
    that did not, because both publish the same median over the images they survived.
    """
    return sorted(
        (candidate for candidate in candidates if candidate["deployable"]),
        key=_ranking_sort_key,
    )


def summarize_candidates(
    rows: list[dict], *, expected_image_count: int
) -> tuple[list[dict], list[dict], list[dict]]:
    """The scored rows as per-scene curves, per-candidate metrics, and a deployment ranking.

    Returns three lists, in the order a bundle writer needs them:

    * **per-scene rows** -- one per candidate, image and severity, carrying the row's own score
      and selection diagnostics plus its image's trend repeated on all six of that image's rows.
      Same count as the input table; see `PER_SCENE_KEYS`.
    * **candidate metrics** -- one per `CANDIDATE_KEY`, in ascending key order, every candidate
      the table contains. The matched confidence controls, the frozen twins and the unfiltered
      padding controls are all here in full; they are excluded from the ranking alone.
    * **the ranking** -- the same dictionaries, filtered to `deployable` and sorted. Not a
      separate copy: a reader that edits one sees the other, which is what keeps a report from
      quoting two different values for one candidate.

    `expected_image_count` is the run's image count, from `run_metadata["image_count"]`. It is
    an argument rather than something derived from the rows because the rows cannot tell the
    difference between a run of 240 images and a run of 250 that lost 10 before scoring -- and
    the first of those is not a deployable tuning run at all.

    Refuses an empty table rather than returning three empty lists. An empty results table is a
    run that produced nothing, and a bundle written from it would be a complete set of files
    describing no measurement.
    """
    if not rows:
        raise ValueError("summarize_candidates needs at least one scored row")
    validate_rows(rows)
    validate_selected_queries(rows)

    grouped: dict[tuple, dict] = {}
    for row in rows:
        candidate = tuple(row[field] for field in CANDIDATE_KEY)
        by_image = grouped.setdefault(candidate, {})
        by_image.setdefault(row["image_id"], {})[row["severity"]] = row

    per_scene: list[dict] = []
    candidates: list[dict] = []
    for candidate in sorted(grouped):
        images = grouped[candidate]
        scenes: list[dict] = []
        for image_id in sorted(images):
            by_severity = images[image_id]
            severities, scores = _ordered_curve(by_severity)
            trend = complete_trend_metrics(severities, scores)
            for severity in severities:
                score_row = by_severity[severity]
                per_scene.append({
                    **{field: score_row[field] for field in ROW_KEY},
                    "source_partition": score_row["source_partition"],
                    "score": score_row["score"],
                    "selected_count": score_row["selected_count"],
                    "clean_overlap": score_row["clean_overlap"],
                    "fully_measured": trend["fully_measured"],
                    "signed_spearman": trend["signed_spearman"],
                    "absolute_spearman": trend["absolute_spearman"],
                    "direction": trend["direction"],
                })
            scenes.append({"image_id": image_id, "scores": scores, **trend})
        candidates.append(
            _candidate_metrics(candidate, images, scenes, expected_image_count)
        )

    by_key = {
        tuple(candidate[field] for field in CANDIDATE_KEY): candidate
        for candidate in candidates
    }
    _pair_padding_modes(by_key, grouped)
    return per_scene, candidates, rank_deployable_candidates(candidates)


# --- the published bundle ---------------------------------------------------------------------


FIGURE_FILES = {
    DEPLOYABLE_SIGNAL: (
        "persistence_actual_distance_deciles.png",
        "persistence_actual_distance_quintiles.png",
    ),
    CONTROL_SIGNAL: (
        "confidence_actual_distance_deciles.png",
        "confidence_actual_distance_quintiles.png",
    ),
}
"""The four figures, grouped by the signal each one draws.

Grouped rather than listed flat because the report has to say which pair shares which y-range,
and the alternative -- recovering the signal from the start of the filename -- would make the
name a second definition of what the figure holds. `corruption_plots.PLOT_FILENAMES` is where
these strings are turned into files; they are spelled again here because that module imports
this one, so this one cannot import it at module scope, and
`test_the_writer_and_the_figures_agree_on_the_names_they_share` fails the moment the two lists
stop matching.
"""

DATA_FILES = ("per_scene.csv", "candidate_metrics.csv", "summary.json", "easy-report.md")

EXPECTED_FILES = frozenset(
    {*DATA_FILES, *(name for names in FIGURE_FILES.values() for name in names)}
)
"""Every file a published bundle holds, and the only files it may hold.

Checked against the staging directory *before* the directory is renamed into place, so a run
that produced seven files or nine never becomes a readable result. Both directions matter and
neither is hypothetical: a missing figure is a bundle whose report describes a picture that is
not there, and an extra file is a figure or a table nothing in `summary.json` accounts for.
"""


def _umask_directory_mode(inside: Path) -> int:
    """The mode a plain `mkdir` would give a directory here, without touching the process umask.

    `tempfile.mkdtemp` creates its directory `0o700`. That is right for a temporary nobody else
    should read and wrong for a published result: `os.replace` carries the mode onto the bundle,
    and an operator who is not the owner is left unable to list a directory whose eight files
    the umask made group-readable.

    Read by creating a directory and asking what mode it came out with, rather than through
    `os.umask`, which is a setter: reading it means setting it and setting it back, and any
    concurrent thread that creates a file in that window gets the wrong permissions instead.
    """
    probe = inside / ".mode-probe"
    probe.mkdir()
    mode = probe.stat().st_mode & 0o777
    probe.rmdir()
    return mode


def _scalar_projection(candidate: dict) -> dict:
    """One candidate row with its nested per-severity maps dropped.

    This is the whole of `candidate_metrics.csv`'s column selection, and it is a projection
    rather than a transformation: `_candidate_metrics` publishes every nested map a second time
    as scalar columns, so dropping the dictionaries loses no number and invents no column name.
    Writing the raw candidate instead puts `{'0': {'count': 250, ...}}` into five cells, which
    is a string no reader can filter, join or sum, and which no row count would notice.
    """
    return {name: value for name, value in candidate.items() if not isinstance(value, dict)}


def _padding_summary(diagnostics: dict) -> dict:
    """The run's padded-query record, rolled up and then carried through whole.

    The per-image detail is kept rather than summarised away, because the rollup alone cannot
    answer the question the padding control exists for: an image with no padded tail selects the
    identical queries with the mask and without it, so the control is a no-op on it by
    construction, and a reader who sees only a run-wide number reads that dilution as a null
    result.

    `padded_images_with_identical_tails` counts only the images that carry padding, which is
    the restriction that makes it readable. Every image without a tail has six identical empty
    tails and would count as "identical" in an unrestricted total, so that total would report
    stability that is really absence -- and instability is the fact the union mask exists for.
    See `decile_analysis._padding_diagnostics` for what each per-image field means and for the
    measurement behind that restriction.
    """
    images = dict((diagnostics or {}).get("images") or {})
    counts = {key: int(image["union_padded_count"]) for key, image in images.items()}
    padded = [key for key, count in counts.items() if count > 0]
    return {
        "image_count": len(images),
        "images_with_padding": len(padded),
        "total_union_padded_count": sum(counts.values()),
        "max_union_padded_count": max(counts.values()) if counts else None,
        "padded_images_with_identical_tails": sum(
            1 for key in padded if images[key]["tail_identical_across_severities"]
        ),
        "images": images,
    }


def build_summary(
    diagnostics: dict,
    candidates: list[dict],
    ranking: list[dict],
    *,
    axis_limits: dict,
    scored_row_count: int,
    per_scene_row_count: int,
) -> dict:
    """Everything `summary.json` holds: where the rows came from and what became of them.

    `axis_limits` is recorded exactly as `corruption_plots.write_corruption_plots` returned it,
    and is never recomputed here. That function returns the limits it *applied* to the fifteen
    panels of each signal; a second computation from the same candidates would agree today and
    would stop agreeing the moment either side changed its mind about which candidates are
    drawable -- and nothing in the bundle would say which of the two the pictures used.

    The two row counts are arguments rather than measurements of `candidates`, and that is the
    point of publishing both. `scored_row_count` is the table that was handed in and
    `per_scene_row_count` is the table that came out; they are equal on every run this module
    can produce, because `summarize_candidates` keeps every row, so a bundle where they differ
    is a bundle whose collapse dropped something. A count derived from the candidate rows would
    be a number checked against itself.

    `duplicate_row_key_count` is `0` in every bundle that exists, and it is published anyway so
    that `summary.json` records which checks ran rather than only what they found:
    `validate_rows` refuses a table with a repeated row key before any directory is staged, so
    a duplicate never reaches this function and this field can never be anything else.

    Coverage is published as `expected_image_count` against the range of candidate
    `image_count`s, never as `measured_fraction`. That fraction is over the images a candidate
    *has rows for*, so a candidate holding 200 images of a 250-image run reports `1.0` when
    every curve it has is complete -- true, and not a statement about the run at all.

    `labels` is what the run turned out to contain, one sorted list per `CANDIDATE_KEY` field,
    while `configuration` is what the ranking required of it. They are separate keys because
    they answer different questions, and a reader comparing them can see at once whether a gate
    excluded everything by finding nothing to admit.
    """
    run = dict(diagnostics["run"])
    image_counts = [candidate["image_count"] for candidate in candidates]
    measured_counts = [candidate["measured_count"] for candidate in candidates]
    return {
        "run": run,
        "configuration": {
            "full_tuning_image_count": FULL_TUNING_IMAGE_COUNT,
            "deployable_signal": DEPLOYABLE_SIGNAL,
            "deployable_membership_mode": DEPLOYABLE_MEMBERSHIP_MODE,
            "deployable_padding_mode": FILTERED_PADDING_MODE,
            "deployable_score_scope": DEPLOYABLE_SCORE_SCOPE,
            "severities": list(EXPECTED_SEVERITIES),
        },
        "labels": {
            field: sorted({candidate[field] for candidate in candidates})
            for field in CANDIDATE_KEY
        },
        "validation": {
            "scored_row_count": int(scored_row_count),
            "per_scene_row_count": int(per_scene_row_count),
            "duplicate_row_key_count": 0,
            "candidate_count": len(candidates),
            "deployable_candidate_count": len(ranking),
            "expected_image_count": int(run["image_count"]),
            "candidate_image_count_range": [min(image_counts), max(image_counts)],
            "candidate_measured_count_range": [min(measured_counts), max(measured_counts)],
        },
        "padding": _padding_summary(diagnostics),
        "axis_limits": axis_limits,
        "candidates": candidates,
        "deployable_ranking": ranking,
    }


# --- the plain-language report -----------------------------------------------------------------


EASY_REPORT_TITLE = "# Which score notices blur, and can we deploy it?"

SECTION_TITLES = (
    "What should we deploy?",
    "Does it react steadily to blur?",
    "Did 20% buckets help?",
    "Did persistence beat model confidence?",
    "What the files contain",
    "What this does not prove",
)
"""The six questions the report answers, in order. Written as a tuple so the order is one
decision made once: the recommendation has to come before the evidence that qualifies it, and
the two sections that say what the bundle cannot support have to come last rather than be
optional reading."""

AUROC_MEANING = (
    "An AUROC of 0.80 means that when we randomly pick one clean image and one blurred image, "
    "the score puts the blurred image higher about 80 times out of 100. It does not mean the "
    "image has an 80% chance of being corrupted. Half of the pairs coming out in the right "
    "order -- 0.500 -- is what a coin toss scores. A number below 0.500 means that blur level "
    "comes out *lower* than clean when the score is read in the one direction this candidate "
    "was locked into, which is a real finding about the candidate and is printed as it was "
    "measured rather than flipped the right way up."
)
"""The one paragraph the whole report rests on, printed whether or not a candidate was ranked.

AUROC is an ordering statistic and reads like a percentage, which is the confusion this
paragraph exists to prevent: nothing in this pipeline is fitted or calibrated, so no number in
the bundle is a probability that an image is corrupted, and a reader who takes 0.80 for one has
read the report backwards.
"""


ORIENTATION_WORDS = {
    1: {
        "reading": "upward",
        "flagged": "higher",
        "moves": "rises",
        "endpoint": "ends above its clean score",
    },
    -1: {
        "reading": "downward",
        "flagged": "lower",
        "moves": "falls",
        "endpoint": "ends below its clean score",
    },
}
"""Every clause in the report that states a direction, keyed by the orientation it belongs to.

Not one of them may be a constant. `_candidate_metrics`' gate requires an orientation and never
a `+1` one, so a candidate whose distance *falls* as blur rises is exactly as deployable as one
that rises -- that is the whole reason `corruption_plots` draws the raw un-oriented score, so a
useful decreasing signal stays visibly decreasing. A report that hard-coded "rises" would print
that word beside a negative signed correlation and tell a non-specialist the opposite of what
was measured.

`endpoint` is worded from the raw score for the same reason. `max_blur_above_clean_rate` is
measured *after* the orientation is applied (`corruption_metrics.oriented_curve_metrics`), so
on a candidate read downward the images it counts are the ones whose raw worst-blur score came
out below their clean one.
"""

WIDER_BUCKET_SCHEME = "quintile"
"""The five-way cut -- the "20% buckets" the report's third question asks about.

Named here because that question has an answer and the answer depends on which of the two
resolutions came first, which no generic "the winning scheme" phrasing can say. It is the string
`corruption_plots.SCHEME_BUCKET_NAMES` keys its five-bucket entry with -- pinned against that
module by the test that pins the other names the two share -- and the one
`corruption_analysis.SCHEMES` writes into every quintile row.
"""


def _count(number: int, noun: str) -> str:
    """`1 candidate`, `3 candidates` -- a count and its noun, agreeing.

    Not decoration. A full tuning run counts candidates in the hundreds and reads the same
    either way, but a run with one candidate is a diagnostic run, and a diagnostic run's report
    is the one a reader most needs to be able to take seriously.
    """
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _plain(value, digits: int = 3) -> str:
    """A number as the report prints it, or the words for one that was never measured."""
    return "not measured" if value is None else f"{float(value):.{digits}f}"


def _signed(value, digits: int = 3) -> str:
    return "not measured" if value is None else f"{float(value):+.{digits}f}"


def _ahead(left: str, left_value, right: str, right_value) -> str:
    """Which of two numbers is larger, as a clause, with no threshold in it.

    Read from the two values every time. The comparison this sentence carries can come out
    either way on a real run -- the matched confidence control outranking persistence is a
    result the design has to be able to publish -- so the verb is never a constant.
    """
    if left_value is None or right_value is None:
        return "one of the two was not measured, so they cannot be compared"
    if float(left_value) > float(right_value):
        return f"{left} comes out ahead"
    if float(left_value) < float(right_value):
        return f"{right} comes out ahead"
    return "the two come out level"


CONTROL_KEY_FIELDS = {"signal": CONTROL_SIGNAL, "score_scope": CONFIDENCE_SCOPE}
"""What the matched control changes about the winner's candidate key, and all it changes.

Two fields, because those are the two the control is: the signal `score_selection` writes for
it, and the scope field that carries that name for want of a decoder layer to hold. Everything
else -- scheme, bucket, membership, padding, scene summary -- has to be identical, which is the
whole claim the comparison rests on.
"""


def _matched_control(summary: dict, winner: dict) -> dict | None:
    """The confidence row of the winner's own selection, or `None` if it was not scored.

    The match is on the whole of `CANDIDATE_KEY` with the two control fields substituted, so it
    identifies exactly one candidate or none at all. A looser match -- the confidence bin alone,
    or every field but the membership mode -- has more than one candidate to choose from on a
    real run, where every bucket carries a frozen twin of its control as well as a dynamic one,
    and it would pick whichever came first in the list.
    """
    key = tuple(
        CONTROL_KEY_FIELDS.get(field, winner[field]) for field in CANDIDATE_KEY
    )
    for candidate in summary["candidates"]:
        if tuple(candidate[field] for field in CANDIDATE_KEY) == key:
            return candidate
    return None


def _preamble(summary: dict) -> list[str]:
    run = summary["run"]
    validation = summary["validation"]
    padding = summary["padding"]
    return [
        EASY_REPORT_TITLE,
        "",
        "Every number below was recomputed from artifacts that were already on disk: the blur "
        "feature cache and the saved query distances. No detector was run and no "
        "nearest-neighbour search was performed, so nothing here is a new measurement of the "
        "images -- it is a new reading of measurements that already existed.",
        "",
        f"This run covered {_count(validation['expected_image_count'], 'image')} of the "
        f"`{run.get('source_partition', 'unknown')}` partition at "
        f"{len(summary['configuration']['severities'])} blur levels, "
        f"{run.get('query_count', 'unknown')} queries an image. Feature cache "
        f"`{run.get('feature_cache_id', 'unknown')}`, kNN results "
        f"`{run.get('source_result_id', 'unknown')}`, clean bank "
        f"`{run.get('bank_id', 'unknown')}`, k={run.get('k', 'unknown')}, normalization "
        f"`{run.get('normalization', 'unknown')}`.",
        "",
        f"{validation['scored_row_count']} scored rows became "
        f"{_count(validation['candidate_count'], 'candidate')}, "
        f"{validation['deployable_candidate_count']} of which passed every deployment gate. "
        f"{padding['images_with_padding']} of {padding['image_count']} images carried repeated "
        "decoder placeholder queries, which the ranked rows leave out.",
        "",
    ]


def _nothing_ranked_lines(summary: dict) -> list[str]:
    """Why an empty ranking is the honest outcome of a run this size, in the reader's words.

    This is the branch every synthetic and every diagnostic run takes, so it says what was
    required, what this run had, and where the numbers it did produce live -- rather than
    stopping at "nothing qualified", which reads as a broken run.
    """
    configuration = summary["configuration"]
    validation = summary["validation"]
    expected = validation["expected_image_count"]
    full = configuration["full_tuning_image_count"]
    lines = [
        "No candidate passed the full tuning-coverage gate, so nothing here is a "
        "recommendation and the deployment ranking in `summary.json` is empty.",
        "",
        f"To be ranked at all, a candidate has to be the `{configuration['deployable_signal']}` "
        f"score at `{configuration['deployable_score_scope']}`, built with "
        f"`{configuration['deployable_membership_mode']}` membership on "
        f"`{configuration['deployable_padding_mode']}` queries, and measured on all {full} "
        f"images of the full tuning run at all {len(configuration['severities'])} blur levels.",
        "",
    ]
    if expected < full:
        lines.extend([
            f"This run declared {_count(expected, 'image')}, fewer than {full}, so no "
            "candidate in it "
            "could pass however well it scored. The gate is not widened to fit a smaller run: "
            f"a candidate measured on {expected} images is a different measurement, not a "
            "slightly smaller one, and calling it deployable would be the one mistake this "
            "gate exists to prevent.",
            "",
        ])
    else:
        lines.extend([
            f"This run declared {_count(expected, 'image')}, so its size is not what excluded "
            "them: each "
            f"of the {_count(validation['candidate_count'], 'candidate')} it measured failed at "
            "least one of the other requirements above.",
            "",
        ])
    lines.extend([
        f"All {_count(validation['candidate_count'], 'candidate')} the run did measure are "
        "published in "
        "full in `candidate_metrics.csv` and in `summary.json`, and the four pictures are drawn "
        "from them. A run like this is a diagnostic, not an empty result.",
        "",
    ])
    return lines


def _winner_lines(summary: dict, winner: dict) -> list[str]:
    aurocs = winner["auroc_by_severity"]
    selected = winner["selected_count_by_severity"]
    severities = [str(severity) for severity in summary["configuration"]["severities"][1:]]
    weakest = min(severities, key=lambda severity: aurocs[severity])
    words = ORIENTATION_WORDS[winner["orientation"]]
    ranked_candidates = _count(
        summary["validation"]["deployable_candidate_count"], "candidate"
    )
    measured_candidates = _count(summary["validation"]["candidate_count"], "candidate")
    smallest = min(entry["min"] for entry in selected.values())
    largest = max(entry["max"] for entry in selected.values())
    return [
        f"The one to take forward is the `{winner['confidence_bin']}` bucket of the "
        f"`{winner['bucket_scheme']}` cut: the `{winner['signal']}` score at "
        f"`{winner['score_scope']}`, with `{winner['membership_mode']}` membership rebuilt at "
        f"every blur level, `{winner['padding_mode']}` queries -- the repeated decoder "
        f"placeholders taken out -- and each scene summarised by `{winner['aggregation']}`. It "
        f"came first of {ranked_candidates} that passed every gate, out of the "
        f"{measured_candidates} this run measured. It was chosen on the same images every number below describes, so this is a "
        "proposal to check on images that took no part in choosing it, not a result about "
        "them.",
        "",
        f"It is read {words['reading']}: the {words['flagged']} score is the more corrupted "
        "one. That direction was chosen once, from this candidate's own images, and every "
        "number below -- the five AUROCs, the step counts, the worst-blur rate -- is counted "
        "through it. A candidate read the other way is not a worse candidate; the pictures draw "
        "the score as measured so that a signal which falls with blur stays visibly falling.",
        "",
        f"Every scene in that bucket was scored over between {smallest} and {largest} selected "
        "queries, so each number below summarises a few dozen queries an image rather than the "
        "whole image.",
        "",
        "How often it puts a blurred image above a clean one, blur level by blur level:",
        "",
        "| blur level | " + " | ".join(severities) + " | all five (macro) |",
        "| --- |" + " --- |" * (len(severities) + 1),
        "| AUROC | "
        + " | ".join(_plain(aurocs[severity]) for severity in severities)
        + f" | {_plain(winner['macro_auroc'])} |",
        "",
        f"The blur level it handles worst is {weakest}, at {_plain(aurocs[weakest])}; a coin "
        "toss would score 0.500. All five are printed because blur strength is where a "
        "corruption check is most likely to be uneven, and one average can hide a level that "
        "does nothing.",
        "",
    ]


def _deploy_section(summary: dict) -> list[str]:
    ranking = summary["deployable_ranking"]
    lines = [f"## {SECTION_TITLES[0]}", ""]
    if ranking:
        lines.extend(_winner_lines(summary, ranking[0]))
    else:
        lines.extend(_nothing_ranked_lines(summary))
    lines.extend([AUROC_MEANING, ""])
    return lines


def _steady_section(summary: dict) -> list[str]:
    lines = [f"## {SECTION_TITLES[1]}", ""]
    ranking = summary["deployable_ranking"]
    if not ranking:
        return lines + [
            "No candidate was ranked, so there is nothing to answer this for. The numbers the "
            "answer is made of -- the absolute and the signed rank correlation, the three "
            "direction counts, the adjacent-step rate and the maximum-blur rate -- are "
            "published for every candidate the run measured in `candidate_metrics.csv`.",
            "",
        ]
    winner = ranking[0]
    measured = winner["measured_count"]
    words = ORIENTATION_WORDS[winner["orientation"]]
    # The gloss defines a word the sentence before it just used. With no flat image there is
    # nothing to define, and the definition read as though it described the images that fell.
    flat_gloss = (
        " A flat image is a complete curve of six identical scores: a measurement that found "
        "no movement, and not a measurement that failed."
        if winner["flat_count"] else ""
    )
    return lines + [
        f"Inside a single scene, blur moves this score by a median absolute rank correlation "
        f"of {_plain(winner['median_absolute_spearman'])}, over the "
        f"{_count(measured, 'image')} that produced a complete six-level curve. Absolute "
        "means the direction is thrown away first: it says how *hard* blur moves the score, "
        "not which way it moves it.",
        "",
        f"Which way is a separate number, and here it is "
        f"{_signed(winner['median_signed_spearman'])} -- the same correlations with their signs "
        f"kept, so the score {words['moves']} as blur gets worse. Neither number is derived "
        "from the other: images that move in opposite directions cancel out in the signed "
        "median and do not cancel in the absolute one, so a candidate can be strong on one and "
        "say nothing on the other.",
        "",
        f"Counted image by image, {winner['positive_count']} of the {measured} measured images "
        f"rose with blur, {winner['negative_count']} fell, and {winner['flat_count']} were "
        f"flat.{flat_gloss} That puts "
        f"{_plain(winner['dominant_direction_fraction'])} of the measured images in the "
        "direction the candidate is read in.",
        "",
        f"Step by step, {_plain(winner['oriented_adjacent_consistency'])} of the five steps "
        "between neighbouring blur levels do not move against that direction, averaged over the "
        f"measured images; and on {_plain(winner['max_blur_above_clean_rate'])} of those images "
        f"the worst blur level {words['endpoint']}, which is the weakest thing a usable signal "
        "has to do. Both are read in the candidate's own direction: the first asks whether a "
        "step moved against it, the second whether the worst blur level ended further along it "
        "than the clean image did.",
        "",
    ]


def _buckets_section(summary: dict) -> list[str]:
    """The two bucket resolutions against each other, under the ranking's own order.

    "Best" here means first in the deployment ranking, not best on any single metric: the
    ranking is the rule the run was tuned with, and a second rule invented for this paragraph
    could name a different winner than the section above it.
    """
    lines = [f"## {SECTION_TITLES[2]}", ""]
    best: dict[str, dict] = {}
    for candidate in summary["deployable_ranking"]:
        best.setdefault(candidate["bucket_scheme"], candidate)
    if not best:
        return lines + [
            "No candidate was ranked, so the two bucket resolutions cannot be compared on this "
            "run. Both are measured in full in `candidate_metrics.csv`.",
            "",
        ]
    if len(best) == 1:
        (only,) = best.values()
        return lines + [
            f"Only the `{only['bucket_scheme']}` cut has a candidate that passed every gate -- "
            f"`{only['confidence_bin']}`, at a macro AUROC of {_plain(only['macro_auroc'])} -- "
            "so there is no ranked candidate at the other resolution to compare it against.",
            "",
        ]
    # The ranking is macro AUROC descending before anything else, and `best` keeps its order,
    # so the first scheme's candidate is never behind the second's and the gap is never signed.
    first, second = list(best.values())[:2]
    difference = float(first["macro_auroc"]) - float(second["macro_auroc"])
    # The question in the heading is about the wider cut, so the answer has to name it. Either
    # way round it is read from which scheme the ranking put first, never assumed.
    answer = (
        "On this run they did help" if first["bucket_scheme"] == WIDER_BUCKET_SCHEME
        else "On this run they did not help"
    )
    return lines + [
        f"{answer}. Under the same ranking rule that produced the deployment order, the "
        f"`{first['bucket_scheme']}` cut comes first: `{first['confidence_bin']}` at a macro "
        f"AUROC of {_plain(first['macro_auroc'])}. The best `{second['bucket_scheme']}` "
        f"candidate is `{second['confidence_bin']}` at {_plain(second['macro_auroc'])}, a gap "
        f"of {_plain(difference)}.",
        "",
        "Both cuts were made on the same confidence ordering over the same queries, so this is "
        "one measurement seen at two resolutions and not two independent experiments. That is "
        "also what makes the comparison worth making: if the two resolutions agree, the "
        "ten-way split was incidental and the wider bucket -- twice as many queries in it, so "
        "a steadier summary of them -- is the safer one to run.",
        "",
    ]


def _control_section(summary: dict) -> list[str]:
    lines = [f"## {SECTION_TITLES[3]}", ""]
    ranking = summary["deployable_ranking"]
    if not ranking:
        return lines + [
            "No candidate was ranked, so there is no matched comparison to report. Every "
            "confidence control the run scored is in `candidate_metrics.csv` beside the "
            "persistence rows it was scored with.",
            "",
        ]
    winner = ranking[0]
    control = _matched_control(summary, winner)
    if control is None:
        return lines + [
            "This selection was scored for persistence only, so it has no matched confidence "
            "control and nothing here answers the question for it.",
            "",
        ]
    return lines + [
        f"The control is the same selection read a different way: the same "
        f"`{winner['bucket_scheme']}` scheme, the same `{winner['confidence_bin']}` bucket, the "
        f"same `{winner['membership_mode']}` membership, the same `{winner['padding_mode']}` "
        f"queries and the same `{winner['aggregation']}` scene summary. Every row of one "
        "selection is checked to report the same selected queries before anything is "
        "summarised, so the two describe one population of queries and not two populations that "
        "resemble each other.",
        "",
        f"Only the score differs. Persistence is how far this image's decoder fingerprints sit "
        f"from the nearest clean ones at `{winner['score_scope']}`; the control is one minus "
        "the detector's own confidence in what it found.",
        "",
        "The control is a yardstick and not a second thing to deploy, whichever way the "
        f"comparison below comes out. This run gates and ranks `{DEPLOYABLE_SIGNAL}` alone -- "
        "`summary.json` records that gate under `configuration` -- so the control is measured "
        "against, never selected and never ranked. A control that comes out ahead is therefore "
        "a finding *about* persistence: it says how much of the separation was already "
        "available from the detector's own confidence, and it is a reason to question the "
        "persistence result rather than a recommendation to deploy the control, which nothing "
        "here tuned or gated as a method of its own.",
        "",
        "Putting blurred images above clean ones across scenes, "
        + _ahead(
            "the control", control["macro_auroc"], "persistence", winner["macro_auroc"]
        )
        + f": macro AUROC {_plain(control['macro_auroc'])} for the control against "
        f"{_plain(winner['macro_auroc'])} for persistence. Moving with blur inside one scene, "
        + _ahead(
            "the control",
            control["median_absolute_spearman"],
            "persistence",
            winner["median_absolute_spearman"],
        )
        + f": median absolute rank correlation "
        f"{_plain(control['median_absolute_spearman'])} against "
        f"{_plain(winner['median_absolute_spearman'])}.",
        "",
        "Those two comparisons are the only ones made. A distance to clean fingerprints and a "
        "`1 - confidence` share no unit, so their raw sizes are never compared, never "
        "subtracted and never drawn on one axis -- only the shapes they trace and the orders "
        "they produce.",
        "",
    ]


def _files_section(summary: dict) -> list[str]:
    limits = summary["axis_limits"]
    lines = [
        f"## {SECTION_TITLES[4]}",
        "",
        "- `per_scene.csv` -- one row per candidate, image and blur level: the raw score, how "
        "many queries it was taken over, and that image's own trend repeated on all six of its "
        "rows, so a single row reads on its own without a join.",
        f"- `candidate_metrics.csv` -- one row per candidate, "
        f"{_count(summary['validation']['candidate_count'], 'row')} here, with every "
        "per-blur-level "
        "number flattened into a plain column. Every candidate the run measured is in it, "
        "including the ones the deployment ranking leaves out.",
        "- `summary.json` -- the same numbers with the per-blur-level detail nested, plus where "
        "the inputs came from, what was counted, the padded-query record, and the exact axis "
        "ranges the four pictures were drawn on.",
        "- Four pictures of the score as it was measured, one panel per confidence bucket, each "
        "panel drawing the middle value across the scenes and the middle half of them at every "
        "blur level:",
    ]
    for signal, names in FIGURE_FILES.items():
        lower, upper = (limits.get(signal) or [None, None])[:2]
        lines.append(
            f"  - `{names[0]}` and `{names[1]}` -- the two {signal} pictures, sharing one "
            f"y-axis from {_plain(lower)} to {_plain(upper)}."
        )
    lines.extend([
        "",
        "The shared axis is what makes the panels comparable: fitted to its own bucket, every "
        "panel would look equally busy and the one thing varying between them would be the "
        "axis. The two signals never share one, because a distance and a `1 - confidence` are "
        "unrelated quantities and a single range covering both would flatten the smaller of "
        "them onto the floor of its panels.",
        "",
        "The curves are drawn exactly as they were measured. They are never flipped into the "
        "direction a candidate is read in, so a bucket whose score falls as blur rises is drawn "
        "falling -- which is a real finding about that bucket, and one a flipped picture would "
        "hide.",
        "",
    ])
    return lines


def _limits_section(summary: dict) -> list[str]:
    run = summary["run"]
    validation = summary["validation"]
    return [
        f"## {SECTION_TITLES[5]}",
        "",
        "- **No probability that an image is corrupted.** AUROC counts how often a blurred "
        "image is put above a clean one; it is an ordering, and nothing in this bundle is "
        "fitted or calibrated to anything, so no number here is a probability.",
        "- **No threshold.** Nothing here was calibrated against anything, so a recommendation "
        "in this report is a bucket and a recipe and never a value to raise an alarm at. "
        "Choosing one needs a decision about how often a false alarm is acceptable, which no "
        "number here supplies.",
        f"- **No held-out evaluation.** Every candidate here was measured on the same "
        f"{validation['expected_image_count']} `{run.get('source_partition', 'unknown')}` "
        "images that any recommendation above was chosen from, and the single direction each "
        "candidate is read in was chosen there too. `deployable` means \"passed every gate on "
        "this tuning run\" and nothing more; what any of this does on images that took no part "
        "in choosing it is untested here, and every count above is conditioned on the choice it "
        "was used to make.",
        "- **A frozen membership and an unfiltered selection are diagnostics, not methods.** A "
        "frozen bucket needs a paired clean version of the image to freeze against, which a "
        "naturally corrupted image does not have; an unfiltered selection keeps the repeated "
        "decoder placeholders on purpose, to measure what removing them does. Both are "
        "measured and published in full, and neither is ever ranked or merged with a ranked "
        "result.",
        "- **The gates exclude, they do not annotate.** A candidate that lost images is absent "
        "from the ranking rather than ranked with a footnote, because a median over the images "
        "a candidate survived looks exactly like a median over all of them.",
        "",
    ]


def render_easy_report(summary: dict) -> str:
    """The whole report, built from `summary` and from nothing else.

    Every sentence carrying a number is written from the summary the bundle publishes, so any
    claim in the report can be checked against `summary.json` beside it, and two runs over the
    same rows produce the same file. Nothing is quoted from a previous run and nothing is
    rounded twice.

    Three sentences this generator cannot produce, each because the design forbids it:

    * a score described as a probability that an image is corrupted. `AUROC_MEANING` is printed
      whether or not anything was ranked, and it says what the number is and what it is not;
    * a magnitude compared across the two signals. A persistence distance and a
      `1 - confidence` share no unit, so the matched comparison is made on AUROC and on rank
      correlation and never on how large the two scores are;
    * a recommendation out of a run that did not earn one. With an empty ranking every section
      says so in its own terms, and the coverage gate is restated rather than relaxed.

    And one word that appears only in a heading. `SECTION_TITLES` asks "Did persistence beat
    model confidence?" because that is the question a reader arrives with; no answer below it
    uses that verb. The verdict comes from `_ahead`, which reads the two values every time it
    is called and has no threshold in it, and it is always followed by the two numbers it was
    read from -- so a comparison that comes out against persistence, which is a result this
    design has to be able to publish, is printed as plainly as one that comes out for it.
    """
    lines = [
        *_preamble(summary),
        *_deploy_section(summary),
        *_steady_section(summary),
        *_buckets_section(summary),
        *_control_section(summary),
        *_files_section(summary),
        *_limits_section(summary),
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_corruption_report(
    output_value: str | Path,
    *,
    score_rows: list[dict],
    diagnostics: dict,
) -> dict:
    """Publish the run as one directory of eight files, or leave nothing behind at all.

    The directory appears in one step. Every table, figure, summary and paragraph is written
    into a staging directory beside the destination, the file set is checked against
    `EXPECTED_FILES`, and only then is the staging directory renamed into place -- so a reader
    never meets a bundle whose report describes a figure that is missing, and a failed run
    leaves neither the output directory nor the staging directory behind. `os.replace` on a
    directory within one filesystem is atomic, and the staging directory is created next to the
    destination rather than in the system temporary area so that it is on the same one.

    That staging directory is `mkdtemp`'s `0o700` until the `chmod` below, which is the one
    property of a temporary directory the published bundle must not inherit: the rename carries
    the mode across, and the result would be a directory only its owner can enter holding eight
    files the umask made group-readable. See `_umask_directory_mode`.

    The checks are in this order for a reason. An existing output directory is refused before
    anything is computed, because merging a new run into an old one produces a directory that
    is internally inconsistent and says nothing about it. The file-set check comes before the
    rename rather than after it, because a check that runs after publication has already
    published. And the cleanup catches `BaseException`, not `Exception`: a `KeyboardInterrupt`
    part-way through leaves the same half-written staging directory as a `ValueError` does.

    What this does not claim: it is not crash-proof. A process killed outright runs no cleanup,
    so it leaves the staging directory on disk under its `.<name>.<random>` name -- rubbish
    beside the destination, and never something a reader could mistake for a finished bundle,
    which is the property being protected. A machine that dies inside `os.replace` itself is
    beyond what any of this can promise.

    `corruption_plots` is imported inside this function on purpose. That module imports this
    one for the gate constants its figures are sliced on, so an import at module scope here
    would be a cycle that fails at interpreter start-up; deferring it to the one function that
    draws anything also keeps `import corruption_reporting` free of matplotlib.
    """
    from .corruption_plots import write_corruption_plots

    output = Path(output_value)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        os.chmod(staging, _umask_directory_mode(staging))
        per_scene, candidates, ranking = summarize_candidates(
            score_rows,
            expected_image_count=int(diagnostics["run"]["image_count"]),
        )
        # `PER_SCENE_KEYS` is the documented column list, so it is the one that produces the
        # columns rather than a second description of what a row happens to hold.
        pd.DataFrame(per_scene, columns=list(PER_SCENE_KEYS)).to_csv(
            staging / "per_scene.csv", index=False
        )
        pd.DataFrame(
            [_scalar_projection(candidate) for candidate in candidates]
        ).to_csv(staging / "candidate_metrics.csv", index=False)
        axis_limits = write_corruption_plots(staging, candidates)
        summary = build_summary(
            diagnostics,
            candidates,
            ranking,
            axis_limits=axis_limits,
            scored_row_count=len(score_rows),
            per_scene_row_count=len(per_scene),
        )
        # `allow_nan=False`: `json.dumps` would otherwise write the bare token `NaN`, which no
        # strict JSON reader accepts, and the file would fail somewhere far from this run.
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (staging / "easy-report.md").write_text(
            render_easy_report(summary), encoding="utf-8"
        )
        actual = {path.name for path in staging.iterdir()}
        if actual != EXPECTED_FILES:
            raise RuntimeError(
                f"report bundle contains {sorted(actual)}, expected {sorted(EXPECTED_FILES)}"
            )
        os.replace(staging, output)
        return summary
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
