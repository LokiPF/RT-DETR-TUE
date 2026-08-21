#!/usr/bin/env python
"""Re-derive an `analyze-corruption-sensitivity` bundle's claims from the bundle's own files.

    python tools/audit_corruption_bundle.py <report directory>

Read-only: it opens the eight published files, recomputes what can be recomputed, and prints
one PASS or FAIL line per check, each stating the size of the population it examined.

**Three exit statuses, and the third is not a failure.**

* `0` -- every check ran against a non-empty population and passed.
* `1` -- the bundle was read and at least one check failed.
* `2` -- **nothing was audited.** The path is not a directory, is missing one of the four data
  files, or holds files that cannot be parsed as themselves. No check ran, so no check can have
  passed or failed.

The distinction matters to anyone scripting this as a gate: `2` means "I have no opinion about
this bundle", which is the opposite of `1`. Treating a `2` as "checks failed" is wrong, and
treating it as "not a failure, carry on" is worse.

**A load failure is a refusal, not a green audit.** The four data files are checked for presence,
then for being parseable, then for holding the columns and keys every check below reads -- and
the fields a check will later coerce are coerced *here*, because a column that exists is not a
column that parses. A bundle that gets past all of that is one where the checks can be trusted
to be examining something; one that does not is refused with an exit `2` naming the file and the
reason. What is never allowed is a run that prints nine PASS lines because each check found
nothing to look at -- see `_STARVED` below.

That guarantee is made by construction rather than by enumeration: `main` maps *any* exception
escaping the load to the same refusal, not only the `BundleUnreadable` ones raised deliberately.
Twice now a specific unguarded read has been found in the constructor -- first a missing key,
then a `severity` column that existed but did not parse as integers -- and each time the escape
exited `1`, which this file defines as "the bundle was read and a check failed". Mis-signalling
a bundle that was never read is worse than crashing on it, so the backstop is a catch-all and
the enumerated cases exist only to produce a better message.

**Why it does not import `scene_uncertainty`.** Every constant and every formula below is
restated here from the design rather than imported from the code that wrote the bundle. An
auditor that imports the producer's `CANDIDATE_KEY`, `EXPECTED_FILES` or `_statistics` cannot
fail when one of them changes -- it agrees with the change and calls the new bundle correct.
Restating them means the two spellings can disagree, which is the only condition under which
this file is worth running. The cost is real and accepted: a deliberate contract change has to
be made twice, and this file is wrong until it is.

**What it deliberately does not check.** Nothing here opens a PNG's pixels. The four figures
are checked for existence and for being PNGs, and the y-ranges they were drawn on are
recomputed from the published candidates and compared against `summary.json`'s recorded
`axis_limits` -- which is the strongest statement a read-only audit can make about the axes: it
proves the recorded range is the one the design's rule produces from the published numbers, and
that one range per signal is recorded, so both figures of a signal were handed the same range.
It does not prove matplotlib honoured it. `test_corruption_plots.py` asserts that against the
`Axes` objects, which is where a claim about a drawn axis belongs.

**The 250-image gate is reported, never forced.** `FULL_TUNING_IMAGE_COUNT` images is what makes
a candidate deployable, and a run over fewer images is a run with an empty ranking. On such a
run this asserts that the ranking *is* empty -- the gate holding is the check -- and prints the
manifest's actual image count. It never rescales the gate to whatever the run happened to
contain, and it exempts the ranking only: the `deployable` column, the re-derived gate and
`deployable_candidate_count` are cross-checked at every run size, because below that image count
the gate admits nobody, so a candidate marked deployable there is a disagreement rather than a
smaller result.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd


# --- the contract, restated ---------------------------------------------------------------

DATA_FILES = ("per_scene.csv", "candidate_metrics.csv", "summary.json", "easy-report.md")

FIGURE_FILES = {
    "persistence": (
        "persistence_actual_distance_deciles.png",
        "persistence_actual_distance_quintiles.png",
    ),
    "confidence": (
        "confidence_actual_distance_deciles.png",
        "confidence_actual_distance_quintiles.png",
    ),
}

EXPECTED_FILES = frozenset(
    {*DATA_FILES, *(name for names in FIGURE_FILES.values() for name in names)}
)

EXPECTED_SEVERITIES = (0, 1, 2, 3, 4, 5)

CANDIDATE_KEY = (
    "signal", "bucket_scheme", "confidence_bin",
    "membership_mode", "padding_mode", "aggregation", "score_scope",
)

SELECTION_KEY = (
    "image_id", "severity", "bucket_scheme", "confidence_bin",
    "membership_mode", "padding_mode",
)

SEVERITY_STATISTICS = (
    "score_count", "score_mean", "score_variance", "score_median", "score_q25", "score_q75",
)

RANKING_METRICS = (
    "macro_auroc",
    "median_absolute_spearman",
    "dominant_direction_fraction",
    "oriented_adjacent_consistency",
)
"""The four descending terms of the documented ranking sort key, in order."""

PER_SCENE_COLUMNS = (
    "image_id", "severity", *CANDIDATE_KEY,
    "score", "signed_spearman", "absolute_spearman",
)
"""Every column a check below reads out of `per_scene.csv`.

Checked at load rather than where each is first indexed, so a table missing one is refused by
name instead of raising a `KeyError` out of whichever check happened to touch it first. This is
deliberately a subset of the published schema: an auditor that demanded every column would
refuse a bundle it could in fact audit."""

CANDIDATE_METRIC_COLUMNS = (
    *CANDIDATE_KEY,
    "macro_auroc",
    # `orientation` and `deployable` are read by `_gate_qualified` and by the three-way
    # agreement check beside it; `measured_count` and `image_count` are read where the ranking
    # in `summary.json` is compared against this table's row for the same candidate. All four
    # are load-bearing, which is why `_validate_candidate_domains` proves they can be read.
    "orientation", "deployable", "measured_count", "image_count",
    *(f"auroc_severity_{severity}" for severity in EXPECTED_SEVERITIES[1:]),
    *(
        f"{statistic}_severity_{severity}"
        for statistic in SEVERITY_STATISTICS
        for severity in EXPECTED_SEVERITIES
    ),
)

SUMMARY_VALIDATION_KEYS = (
    "expected_image_count", "per_scene_row_count", "scored_row_count",
    "candidate_count", "deployable_candidate_count",
)

SUMMARY_KEYS = ("validation", "candidates", "deployable_ranking", "axis_limits")

SELECTIONS_PER_SEVERITY = 34
"""Eleven decile selections and six quintile selections under each of two membership modes.
`corruption_analysis.analyze_corruption_sensitivity` derives the eleven and the six."""

ROWS_PER_SELECTION = 15
"""Three scene summaries x (three decoder layers + their combination + one confidence control).
Three is the pilot cache's decoder-layer count, so a cache with a different one makes the row
budget below wrong -- which is why the failure message prints the observed decomposition
instead of only the totals."""

ROWS_PER_IMAGE = SELECTIONS_PER_SEVERITY * ROWS_PER_SELECTION * len(EXPECTED_SEVERITIES)

FULL_TUNING_IMAGE_COUNT = 250

DEPLOYABLE_GATES = {
    "signal": "persistence",
    "membership_mode": "dynamic",
    "padding_mode": "filtered",
    "score_scope": "layer_2",
}

PLOT_AGGREGATION = "q90"
PLOT_SCOPES = {"persistence": "layer_2", "confidence": "confidence"}

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

RTOL = 1e-9
ATOL = 1e-12

_STARVED = "the check was handed an empty population, so it compared nothing"
"""The failure a check reports when it was starved rather than satisfied.

This is the distinction the whole tool exists to make. A `per_scene.csv` with a header and no
rows, or a `candidate_metrics.csv` with none, makes most of the checks below trivially true:
every element of an empty set satisfies every predicate. Printing PASS there is worse than
printing nothing, because PASS is the line a reader quotes into a report -- so a check that
examined nothing says so and fails.

The one check partly exempt from this is the deployable ranking, and only the ranking: an empty
ranking on a run of fewer than 250 images is admitted by design rather than by starvation.
Everything else that check compares is compared at every run size -- it states all three of its
populations, ranked, re-derived and marked deployable, and on a full run a zero it cannot
corroborate is a failure like any other. It states its populations; it does not explain them."""


# --- helpers ------------------------------------------------------------------------------


class CheckResult(NamedTuple):
    """What one check examined, and what was wrong with it.

    `examined` is printed beside the PASS or FAIL line because a check's verdict is unreadable
    without the size of what it looked at. "PASS, 765,000 rows" and "PASS, 0 rows" are the same
    word about two completely different situations, and only one of them is evidence.
    """

    examined: str
    failures: list[str]


class BundleUnreadable(Exception):
    """A file in the bundle is present but is not the thing it is named after."""


def _number(value):
    """`value` read as a `float`, or `None` when it is not a number at all.

    Separate from `_cell` because the two answer different questions: `_cell` keeps a blank
    distinct from a `0.0`, this one keeps a number distinct from `banana`.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _close(left, right) -> bool:
    """Whether two published numbers agree, with `None` equal only to `None`.

    A tolerance rather than `==` because both sides made a round trip through text -- one
    through `to_csv`, one through `json.dump` -- and `nan` is folded in as its own case so that
    a missing measurement never compares equal to a number.

    **It never raises, and that is a property of this function rather than of its callers.**
    Every call site is a comparison made inside a loop that has *already accumulated findings*:
    the ranking cross-check has gathered gate violations, `check_macro_auroc` has gathered
    mismatched candidates, `check_severity_statistics` has gathered recomputed statistics. A
    bare `float()` here raises on the first unreadable cell, propagates out of the check, and
    `main` replaces the entire accumulated list with one line about a `ValueError` -- so the
    most broken bundle in the suite gets the least actionable message of any of them, which is
    exactly backwards, and is the same defect `_ranking_sort_key` exists to avoid. Guarding the
    four fields the ranking cross-check happens to name would be the enumeration this file has
    been burnt by twice; the guarantee belongs to the comparison itself.

    A value that cannot be read as a number is therefore compared as itself: two identical
    unreadable cells agree, because they do, and anything else disagrees and is reported by its
    caller as the disagreement it is.
    """
    if left is None or right is None:
        return left is None and right is None
    left_number, right_number = _number(left), _number(right)
    if left_number is None or right_number is None:
        return left == right
    if math.isnan(left_number) or math.isnan(right_number):
        return math.isnan(left_number) and math.isnan(right_number)
    return math.isclose(left_number, right_number, rel_tol=RTOL, abs_tol=ATOL)


def _cell(value):
    """One CSV cell as a plain Python number or `None`, so a blank and a `0.0` stay different.

    The numpy scalar is unwrapped rather than passed through, because these values end up
    interpolated into failure messages and `np.int64(17)` names pandas' storage where the reader
    needs the number the file holds.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def _numeric(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """A column as numbers, and a mask of the cells that were never numbers at all.

    `pd.to_numeric(errors="coerce")` turns `banana` into `NaN`, and every check below reads a
    `NaN` as "unmeasured" -- which is right for a blank cell and catastrophically wrong for a
    garbage one. Absorbed that way, a table nobody could have produced audits green, and a false
    green is the worst thing this file can do.

    So the coercion is kept, because a blank really does mean unmeasured, and the cells it
    silently changed come back beside it for a check to report. The emptiness test reads the
    original text rather than the coerced value, because the point is what the *file* said.
    """
    values = pd.to_numeric(series, errors="coerce")
    spoiled = series.notna() & values.isna() & (series.astype(str).str.strip() != "")
    return values, spoiled


def _candidate_id(row) -> tuple:
    return tuple(row[field] for field in CANDIDATE_KEY)


def _ranked_key(candidate) -> tuple | None:
    """A `summary.json` ranking entry's candidate key as seven strings, or `None`.

    Both published files spell every field of a candidate key as a string. An entry that puts
    anything else there -- a number, a list, a missing field -- has to be *reported* rather than
    raised on, for the same reason `_close` never raises: the key is hashed into a set, looked
    up in an index and sorted for display a few lines later, and each of those three raises on
    the wrong type, taking every gate violation the loop had already accumulated with it.
    """
    if not isinstance(candidate, dict):
        return None
    key = tuple(candidate.get(field) for field in CANDIDATE_KEY)
    return key if all(isinstance(field, str) for field in key) else None


def _validate_candidate_domains(frame: pd.DataFrame) -> None:
    """Refuse a `candidate_metrics.csv` whose gate columns cannot be read as what they are.

    These four columns became load-bearing when the ranking gained a re-derived cross-check, and
    a load-bearing column that nothing validates is a fail-open input: every cell
    `_gate_qualified` could not parse used to *silently exclude* that candidate, so an
    `orientation` column re-encoded as `up`/`down` -- a producer-side encoding change, which is
    the whole class this file exists to catch -- collapsed the re-derived set to zero and the
    check went green against it. One `banana` in `measured_count` did the same by flipping the
    entire column to object dtype.

    The split is the one the severity column already uses, and it is deliberate. *Unparseable*
    is refused here: nothing downstream can say anything true about a bundle whose gate columns
    are in an encoding nobody agreed to. *Readable but wrong* -- an orientation of `5`, a
    `measured_count` that contradicts the raw rows -- is left to the checks, because that is a
    finding about the run rather than a failure to read it, and refusing it would suppress the
    other eight checks over a defect they could have reported.

    A table with no rows is not validated and not refused. "No cells" is not "cells in an
    encoding nobody agreed to" -- an empty column has nothing to be in the wrong domain, and
    `pandas` gives it `object` dtype whatever it would have held. Refusing here would take a
    header-only `candidate_metrics.csv` away from the three checks that exist to report exactly
    that, and turn their `_STARVED` failures into "nothing was audited".
    """
    if not len(frame):
        return
    orientation, spoiled = _numeric(frame["orientation"])
    if spoiled.any():
        examples = sorted({str(value) for value in frame.loc[spoiled, "orientation"]})[:3]
        raise BundleUnreadable(
            f"candidate_metrics.csv has {int(spoiled.sum())} orientation cell(s) that are not "
            f"numbers, e.g. {examples}; an orientation is -1, +1 or blank"
        )
    for column in ("measured_count", "image_count"):
        values, spoiled = _numeric(frame[column])
        unreadable = spoiled | values.isna()
        if unreadable.any():
            examples = sorted({str(value) for value in frame.loc[unreadable, column]})[:3]
            raise BundleUnreadable(
                f"candidate_metrics.csv has {int(unreadable.sum())} {column} cell(s) that are "
                f"not numbers, e.g. {examples}"
            )
        ragged = (values != values.round()) | (values < 0)
        if ragged.any():
            examples = sorted({str(value) for value in frame.loc[ragged, column]})[:3]
            raise BundleUnreadable(
                f"candidate_metrics.csv has {int(ragged.sum())} {column} cell(s) that are not "
                f"whole non-negative counts, e.g. {examples}"
            )
    # `pandas` gives a clean `True`/`False` column `bool` dtype; anything else in it -- `yes`,
    # `1`, a blank -- lands as `object` or a number, and `bool("False")` is `True`, so an
    # unvalidated column here would read every candidate as deployable.
    if not pd.api.types.is_bool_dtype(frame["deployable"]):
        tokens = sorted({str(value) for value in frame["deployable"]})[:5]
        raise BundleUnreadable(
            "candidate_metrics.csv 'deployable' is not a boolean column; it holds "
            f"{tokens}"
        )


def _gate_qualified(bundle) -> set[tuple]:
    """Which candidates the deployable ranking *should* hold, re-derived from the tables.

    This exists so that an empty ranking is a finding rather than an excuse. The check that reads
    it used to print "empty on a full run means no candidate passed every gate" beside a zero it
    had never established -- so deleting the whole ranking out of a bundle with 45 deployable
    candidates passed, and the tool volunteered a quotable explanation of a fact that was not
    true. A vacuous pass is bad; a vacuous pass that explains itself is worse.

    The conditions mirror `corruption_reporting._candidate_metrics`' gate, and every one of them
    that *can* come from the raw rows does. Coverage is taken from `finite_counts` -- recomputed
    from `per_scene.csv` -- and `measured_count` is deliberately not read at all, even though the
    producer's gate names it: 250 finite scores at each of six severities over 250 images already
    means 250 fully measured images, so reading the published count would be asking the table
    under audit to confirm itself, and a single bad cell in it would silently drop a candidate
    out of this set. `image_count` is not among the conditions either, because the producer's
    gate does not test it and a condition the producer lacks would manufacture disagreements on
    correct bundles.

    `orientation` is the one input that has to come from the table, since it is a median over the
    per-image signed trends that this file does not recompute. That is why
    `_validate_candidate_domains` proves the column readable at load: an unreadable orientation
    used to exclude its candidate silently, which is how this whole set could collapse to zero
    while the check that reads it reported a pass. The published `deployable` column is compared
    against this set rather than consulted for it -- see `check_ranking_gates`.
    """
    qualified = set()
    for _, row in bundle.candidate_metrics.iterrows():
        if any(row[field] != required for field, required in DEPLOYABLE_GATES.items()):
            continue
        if _cell(row["orientation"]) not in (-1, 1):
            continue
        key = _candidate_id(row)
        if any(
            bundle.finite_counts.get((key, severity), 0) != FULL_TUNING_IMAGE_COUNT
            for severity in EXPECTED_SEVERITIES
        ):
            continue
        qualified.add(key)
    return qualified


def _ranking_sort_key(candidate: dict):
    """The documented four-metric-plus-key sort key, or `None` when it cannot be built.

    `None` rather than a raise, and that is the whole point of this function existing. Building
    the keys is the *last* thing `check_ranking_gates` does, after it has already accumulated
    every gate violation it found. A `None` among the four metrics makes `-value` raise
    `TypeError`, which propagates out of the check and replaces that entire accumulated list
    with one line about a unary minus -- so the most broken bundle in the suite would get the
    least actionable message of any of them, which is exactly backwards.

    `bool` is excluded explicitly because `isinstance(True, int)` is true in Python, and a
    `True` where a macro AUROC belongs would otherwise sort as `1`.

    A ranking entry that is not an object at all is `None` for the same reason: `.get` on a bare
    string raises, and a function whose whole purpose is not to raise has to mean it.
    """
    if not isinstance(candidate, dict):
        return None
    metrics = []
    for field in RANKING_METRICS:
        value = candidate.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if math.isnan(float(value)):
            return None
        metrics.append(-float(value))
    if any(field not in candidate for field in CANDIDATE_KEY):
        return None
    return (*metrics, *(candidate[field] for field in CANDIDATE_KEY))


# --- the checks ---------------------------------------------------------------------------
#
# Every check takes the loaded bundle and returns a `CheckResult`: what it examined, and a list
# of failure strings that is empty on a pass. Returning the failures rather than raising is what
# lets one run report every broken invariant instead of only the first, which matters when the
# thing being audited is a run that took GPU hours to produce.


def check_file_set(bundle) -> CheckResult:
    """Exactly eight files, no more and no fewer, and the four figures are really PNGs."""
    present = {path.name for path in bundle.directory.iterdir()}
    missing = sorted(EXPECTED_FILES - present)
    extra = sorted(present - EXPECTED_FILES)
    failures = []
    if missing:
        failures.append(f"missing files: {missing}")
    if extra:
        failures.append(f"unexpected files: {extra}")
    for names in FIGURE_FILES.values():
        for name in names:
            path = bundle.directory / name
            if path.exists() and path.read_bytes()[:8] != PNG_MAGIC:
                failures.append(f"{name} is not a PNG")
    return CheckResult(f"{len(present)} files", failures)


def check_raw_row_count(bundle) -> CheckResult:
    """`per_scene.csv` holds `image_count x 3060` rows, and its decomposition is the design's.

    The decomposition is checked as well as the total because two errors multiply back to the
    same product -- a cache with four decoder layers and a lost selection, say -- and a single
    equality on 765,000 cannot tell that from a correct run.
    """
    scene = bundle.per_scene
    image_count = bundle.expected_image_count
    expected = image_count * ROWS_PER_IMAGE
    examined = f"{len(scene)} rows"
    if not len(scene):
        return CheckResult(examined, [_STARVED, "per_scene.csv holds a header and no rows"])
    failures = []
    if len(scene) != expected:
        failures.append(
            f"per_scene.csv has {len(scene)} rows, expected {image_count} images x "
            f"{ROWS_PER_IMAGE} = {expected}"
        )
    per_cell = scene.groupby(["image_id", "severity"], sort=False).size().unique().tolist()
    if per_cell != [SELECTIONS_PER_SEVERITY * ROWS_PER_SELECTION]:
        failures.append(
            f"rows per (image, severity) are {sorted(per_cell)}, expected "
            f"[{SELECTIONS_PER_SEVERITY * ROWS_PER_SELECTION}] "
            f"({SELECTIONS_PER_SEVERITY} selections x {ROWS_PER_SELECTION} rows)"
        )
    selections = scene.groupby(["image_id", "severity"], sort=False)[
        list(SELECTION_KEY)
    ].apply(lambda frame: frame.drop_duplicates().shape[0]).unique().tolist()
    if selections != [SELECTIONS_PER_SEVERITY]:
        failures.append(
            f"selections per (image, severity) are {sorted(selections)}, expected "
            f"[{SELECTIONS_PER_SEVERITY}]"
        )
    counts = bundle.summary["validation"]
    if counts["scored_row_count"] != counts["per_scene_row_count"]:
        failures.append(
            f"summary.json validation: scored_row_count {counts['scored_row_count']} != "
            f"per_scene_row_count {counts['per_scene_row_count']}"
        )
    if counts["per_scene_row_count"] != len(scene):
        failures.append(
            f"summary.json per_scene_row_count {counts['per_scene_row_count']} != the "
            f"{len(scene)} rows in per_scene.csv"
        )
    return CheckResult(examined, failures)


def check_candidate_coverage(bundle) -> CheckResult:
    """Every candidate covers the run's images, and every image it covers holds severities 0-5.

    The severity check is an exact set comparison rather than a count: six rows of which two are
    severity 3 is also six rows, and it is a curve with a hole in it that
    `complete_trend_metrics` would refuse and a count would wave through.
    """
    failures = []
    grouped = bundle.per_scene.groupby(list(CANDIDATE_KEY), sort=False)
    expected_severities = set(EXPECTED_SEVERITIES)
    bad_image_counts = []
    bad_severities = []
    pairs = 0
    for candidate, frame in grouped:
        images = frame.groupby("image_id", sort=False)["severity"]
        if images.ngroups != bundle.expected_image_count:
            bad_image_counts.append((candidate, images.ngroups))
        for image_id, severities in images:
            pairs += 1
            observed = set(int(value) for value in severities)
            if observed != expected_severities or len(severities) != len(expected_severities):
                bad_severities.append((candidate, image_id, sorted(severities)))
    examined = f"{pairs} (candidate, image) pairs"
    if not pairs:
        return CheckResult(examined, [_STARVED, "per_scene.csv contains no candidate at all"])
    if bad_image_counts:
        failures.append(
            f"{len(bad_image_counts)} candidate(s) do not cover all "
            f"{bundle.expected_image_count} images, e.g. {bad_image_counts[:3]}"
        )
    if bad_severities:
        failures.append(
            f"{len(bad_severities)} (candidate, image) pair(s) are not exactly severities "
            f"{list(EXPECTED_SEVERITIES)}, e.g. {bad_severities[:3]}"
        )
    return CheckResult(examined, failures)


def check_candidate_key_unique(bundle) -> CheckResult:
    """`candidate_metrics.csv` holds one row per candidate, and the set `per_scene.csv` does."""
    metrics = bundle.candidate_metrics
    examined = f"{len(metrics)} candidate rows"
    if not len(metrics):
        return CheckResult(
            examined, [_STARVED, "candidate_metrics.csv holds a header and no rows"]
        )
    failures = []
    duplicated = metrics[metrics.duplicated(list(CANDIDATE_KEY), keep=False)]
    if len(duplicated):
        keys = sorted({_candidate_id(row) for _, row in duplicated.iterrows()}, key=str)
        failures.append(f"{len(keys)} duplicated candidate key(s): {keys[:3]}")
    scene_candidates = {
        tuple(key) for key in
        bundle.per_scene[list(CANDIDATE_KEY)].drop_duplicates().itertuples(index=False)
    }
    metric_candidates = {_candidate_id(row) for _, row in metrics.iterrows()}
    if scene_candidates != metric_candidates:
        failures.append(
            f"candidate_metrics.csv and per_scene.csv disagree on which candidates exist: "
            f"{len(metric_candidates - scene_candidates)} only in metrics, "
            f"{len(scene_candidates - metric_candidates)} only in per_scene"
        )
    if len(metrics) != bundle.summary["validation"]["candidate_count"]:
        failures.append(
            f"summary.json candidate_count {bundle.summary['validation']['candidate_count']} "
            f"!= the {len(metrics)} rows in candidate_metrics.csv"
        )
    return CheckResult(examined, failures)


def check_ranking_gates(bundle) -> CheckResult:
    """Every ranked candidate is dynamic, filtered, persistence, layer 2, covered, orientable.

    Coverage is re-derived from the raw rows rather than read from the candidate: a ranked
    candidate must have `FULL_TUNING_IMAGE_COUNT` fully measured images *and* that many finite
    scores at each of the six severities, which is the pair of facts that makes its five AUROCs
    comparisons between two full groups.

    On a run over fewer images the gate can admit nothing, so the check becomes "the ranking is
    empty". That is the gate holding, and it is asserted rather than skipped -- a widened gate
    would otherwise pass this audit silently.

    This is the one check exempt from `_STARVED`, and the exemption covers the ranking alone: an
    empty ranking below `FULL_TUNING_IMAGE_COUNT` images is a documented outcome rather than a
    starved check. Everything else here is compared at every run size, including the agreement
    between the ranking, the `deployable` column and the re-derived gate. It states its
    populations; it does not explain them.
    """
    ranking = bundle.summary["deployable_ranking"]
    failures = []
    # Readable but outside the domain, so it is reported rather than refused -- and reported on
    # every run, before the short-run return, because an orientation of `5` is a defect whatever
    # size the run was.
    orientations, _ = _numeric(bundle.candidate_metrics["orientation"])
    stray = orientations.notna() & ~orientations.isin([-1, 1])
    if stray.any():
        examples = sorted({str(value) for value in orientations[stray]})[:3]
        failures.append(
            f"{int(stray.sum())} candidate(s) publish an orientation outside "
            f"{{-1, +1}} or blank, e.g. {examples}"
        )
    # The short run is an exemption for the *ranking*, and for nothing else. Returning here --
    # before the re-derived gate, the `deployable` column and `deployable_candidate_count` were
    # ever compared -- made all three unaudited on the only bundle size a host without the real
    # cache can produce: a `candidate_metrics.csv` marking 45 candidates deployable on a 4-image
    # run, and a `deployable_candidate_count` of 45 beside an empty ranking, both audited 9 of 9.
    # Below `FULL_TUNING_IMAGE_COUNT` images the gate's own definition admits nobody, so the
    # re-derived set is empty *by that definition* rather than by starvation, and a candidate
    # marked deployable against it is a disagreement at any run size.
    short_run = bundle.expected_image_count != FULL_TUNING_IMAGE_COUNT
    if short_run and ranking:
        failures.append(
            f"the run measured {bundle.expected_image_count} images, not "
            f"{FULL_TUNING_IMAGE_COUNT}, yet {len(ranking)} candidate(s) were ranked deployable"
        )
    qualified = _gate_qualified(bundle)
    published = {
        _candidate_id(row) for _, row in bundle.candidate_metrics.iterrows()
        if bool(row["deployable"])
    }
    examined = (
        f"{len(ranking)} ranked candidates against {len(qualified)} that qualify and "
        f"{len(published)} marked deployable"
    )
    if short_run:
        examined += (
            f"; the gate admits none below {FULL_TUNING_IMAGE_COUNT} images, so all three "
            "must be empty"
        )
    finite_by_candidate = bundle.finite_counts

    # Grouped rather than one line per candidate. Every other message in this file caps its
    # examples at three, and a ranking is 45 candidates on a real run: an unbounded loop turns
    # one defect -- a blanked orientation column, an inverted `deployable` -- into 45 near
    # identical lines that bury the eight other checks. The count is the finding; three keys are
    # enough to act on.
    problems: dict[str, list] = {}

    def note(reason: str, example) -> None:
        problems.setdefault(reason, []).append(example)

    table = bundle.candidate_metrics.drop_duplicates(
        list(CANDIDATE_KEY), keep="first"
    ).set_index(list(CANDIDATE_KEY))

    for position, candidate in enumerate(ranking):
        key = _ranked_key(candidate)
        if key is None:
            shown = (
                {field: candidate.get(field) for field in CANDIDATE_KEY}
                if isinstance(candidate, dict) else candidate
            )
            note("do not carry a candidate key of seven strings", (position, shown))
            continue
        for field, required in DEPLOYABLE_GATES.items():
            if candidate[field] != required:
                note(f"have a {field} other than {required!r}", (key, candidate[field]))
        for field in ("measured_count", "image_count"):
            value = candidate.get(field)
            if value != FULL_TUNING_IMAGE_COUNT:
                note(f"have a {field} other than {FULL_TUNING_IMAGE_COUNT}", (key, value))
        if candidate.get("orientation") not in (-1, 1):
            note("have no locked orientation", (key, candidate.get("orientation")))
        thin = {
            severity: finite_by_candidate.get((key, severity), 0)
            for severity in EXPECTED_SEVERITIES
            if finite_by_candidate.get((key, severity), 0) != FULL_TUNING_IMAGE_COUNT
        }
        if thin:
            note("have severities the raw rows do not fully cover", (key, thin))

        # Two published files describing the same candidate. The gate fields above are read out
        # of `summary.json`'s ranking dicts, so a `candidate_metrics.csv` that contradicts them
        # -- 17 measured images for a candidate the ranking calls fully covered -- is invisible
        # to every other check here.
        if key not in table.index:
            note("have no row in candidate_metrics.csv", key)
            continue
        row = table.loc[key]
        for field in ("measured_count", "image_count", "orientation", "macro_auroc"):
            if not _close(_cell(row[field]), candidate.get(field)):
                note(
                    f"disagree between summary.json and candidate_metrics.csv on {field}",
                    (key, candidate.get(field), _cell(row[field])),
                )
        if not bool(row["deployable"]):
            note("are marked deployable=False in candidate_metrics.csv", key)

    for reason, examples in problems.items():
        failures.append(
            f"{len(examples)} ranked candidate(s) {reason}, e.g. {examples[:3]}"
        )
    if len(ranking) != bundle.summary["validation"]["deployable_candidate_count"]:
        failures.append("summary.json deployable_candidate_count disagrees with the ranking")

    # Three statements about one set, compared in every direction rather than one. The ranking
    # in `summary.json`, the `deployable` column of `candidate_metrics.csv`, and this file's own
    # re-derivation from the raw rows all name the candidates that passed every gate; any two of
    # them disagreeing is a finding, and which two says what kind.
    #
    # One direction is not enough, and that is not hypothetical: when only `qualified - ranked`
    # could fail, anything that collapsed `qualified` to zero -- one unreadable cell in a column
    # nothing validated -- made the whole cross-check vacuous, and a 250-image bundle with its
    # entire ranking deleted audited green. `_STARVED` is the doctrine this check was missing:
    # a population that came out empty is a thing to report, not a thing to pass over.
    ranked = {key for key in map(_ranked_key, ranking) if key is not None}
    # `key=str` wherever candidate keys are sorted only to be shown. Sorting tuples compares
    # their fields, which raises the moment two keys disagree on a field's type -- and this sort
    # runs *after* the loop above has accumulated its findings, so a single numeric field in one
    # ranking entry would discard all of them. Ordering three examples is not worth a raise.
    unranked = sorted(qualified - ranked, key=str)
    if unranked:
        failures.append(
            f"{len(unranked)} candidate(s) pass every gate in candidate_metrics.csv but are "
            f"absent from the ranking, e.g. {unranked[:3]}"
        )
    unqualified = sorted(ranked - qualified, key=str)
    if unqualified:
        failures.append(
            f"{len(unqualified)} ranked candidate(s) do not pass the gate when it is re-derived "
            f"from the raw rows, e.g. {unqualified[:3]}"
        )
    if published != qualified:
        failures.append(
            f"candidate_metrics.csv marks {len(published)} candidate(s) deployable but "
            f"{len(qualified)} pass the re-derived gate: "
            f"{len(published - qualified)} marked and not qualifying, "
            f"{len(qualified - published)} qualifying and not marked"
        )
    if ranking and not qualified:
        failures.append(
            f"the ranking holds {len(ranking)} candidate(s) while nothing at all passes the "
            "re-derived gate; the cross-check examined an empty population and cannot support "
            "the ranking either way"
        )

    # Built last, and built so that it cannot throw away everything above it. See
    # `_ranking_sort_key` for why a missing metric is reported here rather than raised.
    keys = [_ranking_sort_key(candidate) for candidate in ranking]
    unsortable = [index for index, key in enumerate(keys) if key is None]
    if unsortable:
        failures.append(
            f"{len(unsortable)} ranked candidate(s) carry a missing or non-numeric ranking "
            f"metric, so the sort order could not be checked: positions {unsortable[:5]}"
        )
    else:
        try:
            ordered = sorted(keys)
        except TypeError as error:
            failures.append(f"the ranking's sort keys are not mutually comparable: {error}")
        else:
            if keys != ordered:
                failures.append("the deployable ranking is not sorted by its documented key")
    return CheckResult(examined, failures)


def check_macro_auroc(bundle) -> CheckResult:
    """Each macro AUROC is the mean of that candidate's five per-severity AUROCs.

    Both directions are checked, because the failure that hides is the asymmetric one: a macro
    published beside five `None`s is a number computed from nothing, and five AUROCs with no
    macro is a candidate that cannot be ranked but looks measurable.
    """
    metrics = bundle.candidate_metrics
    examined = f"{len(metrics)} candidate rows"
    if not len(metrics):
        return CheckResult(
            examined, [_STARVED, "candidate_metrics.csv holds a header and no rows"]
        )
    failures = []
    for _, row in metrics.iterrows():
        per_severity = [
            _cell(row[f"auroc_severity_{severity}"]) for severity in EXPECTED_SEVERITIES[1:]
        ]
        macro = _cell(row["macro_auroc"])
        key = _candidate_id(row)
        # One `banana` anywhere in a numeric column flips the whole column to `object` dtype, so
        # every cell of it arrives here as text and the readable ones still have to be compared.
        # Reported rather than coerced, and reported rather than raised on: `float()` on the bad
        # cell would discard every mismatched candidate found before it.
        unreadable = sorted(
            {str(value) for value in (*per_severity, macro)
             if value is not None and _number(value) is None}
        )
        if unreadable:
            failures.append(
                f"{key} publishes a non-numeric AUROC, e.g. {unreadable[:3]}; these are "
                "unreadable cells, not unmeasured ones"
            )
            continue
        if any(value is None for value in per_severity):
            if not all(value is None for value in per_severity):
                failures.append(f"{key} publishes a partial set of per-severity AUROCs")
            if macro is not None:
                failures.append(f"{key} publishes macro_auroc={macro} with no per-severity set")
            continue
        if macro is None:
            failures.append(f"{key} publishes five AUROCs and no macro_auroc")
            continue
        expected = float(np.mean([float(value) for value in per_severity]))
        if not _close(macro, expected):
            failures.append(f"{key} macro_auroc {macro} != mean of severities {expected}")
    return CheckResult(examined, failures)


def check_spearman_pairs(bundle) -> CheckResult:
    """Each row's absolute Spearman is non-negative and is `abs()` of its own signed Spearman.

    Row-wise and never candidate-wise: the median of the absolute values is not the absolute
    value of the median, so a check that compared the two published medians would fail on
    correct data and pass on a table where a single image's pair had been crossed.
    """
    scene = bundle.per_scene
    examined = f"{len(scene)} rows"
    if not len(scene):
        return CheckResult(examined, [_STARVED, "per_scene.csv holds a header and no rows"])
    failures = []
    signed, signed_spoiled = _numeric(scene["signed_spearman"])
    absolute, absolute_spoiled = _numeric(scene["absolute_spearman"])
    for column, spoiled in (
        ("signed_spearman", signed_spoiled), ("absolute_spearman", absolute_spoiled)
    ):
        if spoiled.any():
            values = sorted({str(value) for value in scene.loc[spoiled, column]})[:3]
            failures.append(
                f"{int(spoiled.sum())} row(s) carry a non-numeric {column}, e.g. {values}; "
                "these are unreadable rows, not unmeasured ones"
            )
    negative = absolute[absolute < 0]
    if len(negative):
        failures.append(f"{len(negative)} row(s) carry a negative absolute_spearman")
    both = signed.notna() & absolute.notna()
    mismatched = both & ~np.isclose(absolute, signed.abs(), rtol=RTOL, atol=ATOL)
    if mismatched.any():
        rows = scene.loc[mismatched, ["image_id", "signal", "confidence_bin"]].head(3)
        failures.append(
            f"{int(mismatched.sum())} row(s) have absolute_spearman != |signed_spearman|, "
            f"e.g. {rows.to_dict('records')}"
        )
        return CheckResult(examined, failures)
    lonely = signed.isna() ^ absolute.isna()
    if lonely.any():
        failures.append(
            f"{int(lonely.sum())} row(s) publish one Spearman and not the other; unmeasured "
            "must leave both empty"
        )
    return CheckResult(examined, failures)


def check_severity_statistics(bundle) -> CheckResult:
    """Every per-severity statistic re-derived from the raw scores matches the saved value.

    Population variance (`ddof=0`) because the values are every image the candidate scored at
    that severity, not a sample of them; on a sample variance the two would differ by
    `n/(n-1)`, which at 250 images is a 0.4% gap that a loose tolerance would hide. Mean,
    median and both quartiles are recomputed alongside it: they cost one pass over the same
    array, and a variance that agrees while a quartile does not is a table built from a
    different slice of rows.
    """
    failures = []
    # `drop_duplicates` rather than a bare `set_index`: on a table with a repeated candidate key
    # -- which `check_candidate_key_unique` reports on its own -- `.loc` hands back a frame
    # instead of a row and every comparison below raises. A check that crashes on a bundle
    # another check already condemned takes the rest of the audit down with it, so the first
    # row is used and the duplicate is left to the check that exists to find it.
    metrics = bundle.candidate_metrics.drop_duplicates(
        list(CANDIDATE_KEY), keep="first"
    ).set_index(list(CANDIDATE_KEY))
    grouped = bundle.per_scene.groupby(list(CANDIDATE_KEY) + ["severity"], sort=False)["score"]
    compared = 0
    unmatched = set()
    spoiled_total = 0
    spoiled_examples: set[str] = set()
    for key, scores in grouped:
        candidate, severity = key[:-1], int(key[-1])
        if severity not in EXPECTED_SEVERITIES:
            continue
        # Counted rather than silently skipped: a candidate in `per_scene.csv` with no row in
        # `candidate_metrics.csv` used to `continue` here, so a metrics table missing half its
        # candidates shrank this check's population with nothing saying so.
        if candidate not in metrics.index:
            unmatched.add(candidate)
            continue
        compared += 1
        values, spoiled = _numeric(scores)
        if spoiled.any():
            spoiled_total += int(spoiled.sum())
            spoiled_examples.update(str(value) for value in scores[spoiled][:2])
        values = values.to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        row = metrics.loc[candidate]
        expected = {
            "score_count": float(values.size),
            "score_mean": float(np.mean(values)) if values.size else None,
            "score_variance": float(np.var(values, ddof=0)) if values.size else None,
            "score_median": float(np.median(values)) if values.size else None,
            "score_q25": float(np.quantile(values, 0.25)) if values.size else None,
            "score_q75": float(np.quantile(values, 0.75)) if values.size else None,
        }
        for column, want in expected.items():
            got = _cell(row[f"{column}_severity_{severity}"])
            if not _close(got, want):
                failures.append(
                    f"{candidate} severity {severity}: {column} is {got}, recomputed {want}"
                )
    examined = f"{compared} (candidate, severity) groups"
    if not compared:
        return CheckResult(
            examined,
            [_STARVED, "no (candidate, severity) group had a row in candidate_metrics.csv"],
        )
    if unmatched:
        failures.append(
            f"{len(unmatched)} candidate(s) in per_scene.csv have no candidate_metrics.csv "
            f"row and were not compared, e.g. {sorted(unmatched, key=str)[:3]}"
        )
    if spoiled_total:
        failures.append(
            f"{spoiled_total} score cell(s) are non-numeric, e.g. "
            f"{sorted(spoiled_examples)[:3]}; these are unreadable, not unmeasured"
        )
    return CheckResult(examined, failures)


def check_axis_limits(bundle) -> CheckResult:
    """One recorded y-range per signal, and it is the range the design's rule produces.

    Recomputed from `summary.json`'s own candidate list over the slice the figures draw --
    dynamic membership, filtered queries, the `q90` summary, each signal at its own scope, and
    only candidates with a median and both quartiles at all six severities. The range is the
    smallest `q25` to the largest `q75` over that slice, plus a 5% margin.

    Recording one range per signal is what makes both of that signal's two figures share it;
    this check proves the recorded number is right and that there is exactly one of it. It does
    not open the PNGs -- see the module docstring.
    """
    recorded = bundle.summary["axis_limits"]
    failures = []
    if set(recorded) != set(PLOT_SCOPES):
        return CheckResult(
            f"{len(recorded)} recorded ranges",
            [
                f"axis_limits names {sorted(recorded)}, expected one range per signal "
                f"{sorted(PLOT_SCOPES)}"
            ],
        )
    drawn_total = 0
    for signal, scope in PLOT_SCOPES.items():
        drawn = [
            candidate for candidate in bundle.summary["candidates"]
            if candidate["signal"] == signal
            and candidate["membership_mode"] == "dynamic"
            and candidate["padding_mode"] == "filtered"
            and candidate["aggregation"] == PLOT_AGGREGATION
            and candidate["score_scope"] == scope
            and all(
                candidate["severity_statistics"].get(str(severity), {}).get(field) is not None
                for severity in EXPECTED_SEVERITIES
                for field in ("median", "q25", "q75")
            )
        ]
        drawn_total += len(drawn)
        if not drawn:
            failures.append(f"no drawable {signal} candidate, yet a {signal} range is recorded")
            continue
        lower = min(
            stats["q25"] for row in drawn for stats in row["severity_statistics"].values()
        )
        upper = max(
            stats["q75"] for row in drawn for stats in row["severity_statistics"].values()
        )
        span = upper - lower
        margin = 0.05 * span if span > 0 else max(0.05 * abs(lower), 1e-6)
        want = (lower - margin, upper + margin)
        got = recorded[signal]
        if len(got) != 2 or not (_close(got[0], want[0]) and _close(got[1], want[1])):
            failures.append(f"{signal} axis_limits {got}, recomputed {list(want)}")
        elif not got[0] < got[1]:
            failures.append(f"{signal} axis_limits {got} is not an interval")
    examined = f"{len(recorded)} ranges over {drawn_total} drawn candidates"
    if not drawn_total:
        failures.insert(0, _STARVED)
    return CheckResult(examined, failures)


CHECKS = (
    ("bundle holds exactly the eight expected files", check_file_set),
    ("per_scene.csv row count is image_count x 3060", check_raw_row_count),
    ("every candidate covers every image at severities 0-5", check_candidate_coverage),
    ("candidate_metrics.csv has one row per candidate key", check_candidate_key_unique),
    ("the deployable ranking passes every gate, in order", check_ranking_gates),
    ("macro AUROC is the mean of its five per-severity AUROCs", check_macro_auroc),
    ("absolute Spearman is |signed Spearman| on the same row", check_spearman_pairs),
    ("severity statistics recompute from the raw scores", check_severity_statistics),
    ("one recomputed y-range per signal in axis_limits", check_axis_limits),
)


# --- loading and reporting ------------------------------------------------------------------


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise BundleUnreadable(f"{path.name}: {type(error).__name__}: {error}") from error


def _read_csv(path: Path, required: tuple[str, ...]) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as error:
        raise BundleUnreadable(f"{path.name}: {type(error).__name__}: {error}") from error
    absent = [column for column in required if column not in frame.columns]
    if absent:
        raise BundleUnreadable(f"{path.name} is missing the column(s) {absent}")
    return frame


class Bundle:
    """The eight files, read once, plus the derived table the ranking check would rebuild.

    Every read here is guarded, every key a check will later index is proved present *now*, and
    every field a check will later coerce is coerced *now* -- so a malformed bundle becomes one
    refusal naming the file and the reason instead of a traceback out of whichever check happened
    to touch the bad thing first. The alternative was tried and is what two rounds of review
    found: first four malformed bundles giving four raw tracebacks and zero check lines, then a
    `severity` column that passed the presence test and blew up on `int()` two statements later.

    Presence and parseability are different claims and the second one is the one that bites.
    `main` backstops this constructor with a catch-all for exactly that reason -- see the module
    docstring -- so a case nobody enumerated is a refusal rather than a traceback.
    """

    def __init__(self, directory: Path):
        self.directory = directory
        self.summary = _read_json(directory / "summary.json")
        if not isinstance(self.summary, dict):
            raise BundleUnreadable("summary.json is not a JSON object")
        absent = [key for key in SUMMARY_KEYS if key not in self.summary]
        if absent:
            raise BundleUnreadable(f"summary.json is missing the key(s) {absent}")
        validation = self.summary["validation"]
        if not isinstance(validation, dict):
            raise BundleUnreadable("summary.json 'validation' is not an object")
        absent = [key for key in SUMMARY_VALIDATION_KEYS if key not in validation]
        if absent:
            raise BundleUnreadable(f"summary.json validation is missing the key(s) {absent}")
        # Integrality, not merely coercibility. `int(4.9)` is `4`, silently, and every row
        # budget below would then be computed against an image count the bundle never claimed --
        # a whole run audited green against the wrong denominator. A count of images is a whole
        # number or it is a defect in the thing that wrote it.
        images = validation["expected_image_count"]
        if isinstance(images, bool) or not isinstance(images, (int, float)):
            raise BundleUnreadable(
                f"summary.json validation.expected_image_count is not a number: {images!r}"
            )
        if isinstance(images, float) and not images.is_integer():
            raise BundleUnreadable(
                f"summary.json validation.expected_image_count is {images!r}, which is not a "
                "whole number of images"
            )
        self.expected_image_count = int(images)
        # A bundle claiming zero images is not a small run, it is not a run. Refused here rather
        # than audited, because `0 == 0 x 3060` makes the row budget vacuously true and several
        # checks below would then pass over an empty table.
        if self.expected_image_count < 1:
            raise BundleUnreadable(
                "summary.json validation.expected_image_count is "
                f"{self.expected_image_count}; a bundle over no images is not auditable"
            )

        self.per_scene = _read_csv(directory / "per_scene.csv", PER_SCENE_COLUMNS)
        self.candidate_metrics = _read_csv(
            directory / "candidate_metrics.csv", CANDIDATE_METRIC_COLUMNS
        )
        _validate_candidate_domains(self.candidate_metrics)
        # `severity` is coerced here rather than at each `int()` downstream. The column having
        # survived the presence test says nothing about its cells: a producer that started
        # writing `sev_0 .. sev_5` -- exactly the encoding drift this file exists to catch --
        # leaves every other column valid and makes `int()` raise ten frames later. Out-of-range
        # integers are deliberately *not* refused here; a severity 7 row is a finding for the
        # checks to report, not a parse failure.
        severities, spoiled = _numeric(self.per_scene["severity"])
        bad = spoiled | severities.isna()
        if bad.any():
            examples = sorted({str(value) for value in self.per_scene.loc[bad, "severity"]})[:3]
            raise BundleUnreadable(
                f"per_scene.csv has {int(bad.sum())} severity cell(s) that are not numbers, "
                f"e.g. {examples}"
            )
        fractional = severities != severities.round()
        if fractional.any():
            examples = sorted(
                {str(value) for value in self.per_scene.loc[fractional, "severity"]}
            )[:3]
            raise BundleUnreadable(
                f"per_scene.csv has {int(fractional.sum())} severity cell(s) that are not whole "
                f"numbers, e.g. {examples}"
            )
        self.per_scene["severity"] = severities.astype(int)

        scores, _ = _numeric(self.per_scene["score"])
        finite = self.per_scene[np.isfinite(scores)]
        counts = (
            finite.groupby(list(CANDIDATE_KEY) + ["severity"], sort=False).size().to_dict()
        )
        self.finite_counts = {
            (key[:-1], int(key[-1])): value for key, value in counts.items()
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path, help="a published report bundle")
    arguments = parser.parse_args(argv)
    directory = arguments.directory
    if not directory.is_dir():
        print(f"not a directory: {directory}", file=sys.stderr)
        print("nothing was audited (exit 2)", file=sys.stderr)
        return 2

    # Checked before anything is parsed, because the likeliest way to misuse this is to aim it
    # at the wrong report directory -- `analyze-confidence-deciles` publishes a `per_scene.csv`
    # and a `summary.json` too, under a different schema. Loading first turns that into a
    # `FileNotFoundError` traceback naming a CSV, which says nothing about the mistake.
    absent = [name for name in DATA_FILES if not (directory / name).is_file()]
    if absent:
        print(
            f"{directory} is not an analyze-corruption-sensitivity bundle: it is missing "
            f"{absent}",
            file=sys.stderr,
        )
        print("nothing was audited (exit 2)", file=sys.stderr)
        return 2

    # Loading is guarded separately from the checks, and it refuses rather than reports: a check
    # cannot say anything about a bundle that never loaded, and nine PASS lines over a table that
    # was never read would be the worst output this file could produce.
    try:
        bundle = Bundle(directory)
    except BundleUnreadable as error:
        print(f"{directory} cannot be audited -- {error}", file=sys.stderr)
        print("nothing was audited (exit 2)", file=sys.stderr)
        return 2
    except Exception as error:  # noqa: BLE001 - the backstop, not a handler
        # Deliberately broad. `Bundle` enumerates the malformations it can name, and twice now a
        # case it did not enumerate has escaped as a traceback at exit 1 -- which this file
        # defines as "the bundle was read and a check failed", so the tool asserted an opinion
        # about a bundle it had never read. Every route out of the load lands on the same
        # refusal; the enumerated cases exist to produce a better message, not to be exhaustive.
        print(
            f"{directory} cannot be audited -- reading it raised "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        print("nothing was audited (exit 2)", file=sys.stderr)
        return 2

    validation = bundle.summary["validation"]
    print(f"bundle:            {directory}")
    print(f"manifest images:   {bundle.expected_image_count}")
    print(f"raw rows:          {validation['per_scene_row_count']}")
    print(f"candidates:        {validation['candidate_count']}")
    print(f"deployable:        {validation['deployable_candidate_count']}")
    if bundle.expected_image_count != FULL_TUNING_IMAGE_COUNT:
        print(
            f"NOTE: the run measured {bundle.expected_image_count} images, not the "
            f"{FULL_TUNING_IMAGE_COUNT} the deployability gate requires. The expected raw-row "
            f"count is {bundle.expected_image_count} x {ROWS_PER_IMAGE} = "
            f"{bundle.expected_image_count * ROWS_PER_IMAGE}, and the ranking must be empty. "
            "The gate is not relaxed."
        )
    print()

    width = max(len(description) for description, _ in CHECKS)
    failed = 0
    for description, check in CHECKS:
        # An exception is a failed check, not a failed audit. A bundle broken enough to raise is
        # exactly the bundle whose remaining eight checks a reader most needs to see, and a
        # traceback out of check three would replace them with nothing.
        try:
            examined, failures = check(bundle)
        except Exception as error:  # noqa: BLE001 - reported, not handled
            examined = "unknown; the check did not finish"
            failures = [f"the check raised {type(error).__name__}: {error}"]
        status = "FAIL" if failures else "PASS"
        failed += bool(failures)
        print(f"{status}  {description.ljust(width)}  {examined}")
        for failure in failures:
            print(f"        {failure}")
    print()
    print(f"{len(CHECKS) - failed} of {len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
