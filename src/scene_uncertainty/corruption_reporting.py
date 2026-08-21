"""One row per candidate out of the corruption score table, and one ranking out of the rows.

`analyze_corruption_sensitivity` emits 3,060 rows an image -- 34 selections, 15 signal/scope/
summary rows each, at six severities -- so a 250-image tuning run is 765,000 rows. This module
collapses that twice: first to one curve per candidate and image, then to one row per candidate,
and finally it sorts the candidates that pass every deployment gate. Nothing here reads a file
or writes one; `summarize_candidates` takes the rows and hands back three lists, and the bundle
writer that turns them into `per_scene.csv`, `candidate_metrics.csv` and `summary.json` is a
separate concern layered on top of this one.

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

import numpy as np

from .corruption_metrics import (
    EXPECTED_SEVERITIES,
    choose_orientation,
    complete_trend_metrics,
    oriented_curve_metrics,
    severity_aurocs,
)
from .decile_scoring import PRIMARY_SCORE_SCOPE


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
