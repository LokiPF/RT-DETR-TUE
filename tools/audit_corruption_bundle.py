#!/usr/bin/env python
"""Re-derive an `analyze-corruption-sensitivity` bundle's claims from the bundle's own files.

    python tools/audit_corruption_bundle.py <report directory>

Read-only: it opens the eight published files, recomputes what can be recomputed, and prints
one PASS or FAIL line per check. Exit status is 0 when every check passed and 1 otherwise, so
this is usable as a gate on a real run.

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
contain.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

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


# --- helpers ------------------------------------------------------------------------------


def _close(left, right) -> bool:
    """Whether two published numbers agree, with `None` equal only to `None`.

    A tolerance rather than `==` because both sides made a round trip through text -- one
    through `to_csv`, one through `json.dump` -- and `nan` is folded in as its own case so that
    a missing measurement never compares equal to a number.
    """
    if left is None or right is None:
        return left is None and right is None
    left, right = float(left), float(right)
    if math.isnan(left) or math.isnan(right):
        return math.isnan(left) and math.isnan(right)
    return math.isclose(left, right, rel_tol=RTOL, abs_tol=ATOL)


def _cell(value):
    """One CSV cell as a number or `None`, so a blank and a `0.0` stay different facts."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def _candidate_id(row) -> tuple:
    return tuple(row[field] for field in CANDIDATE_KEY)


# --- the checks ---------------------------------------------------------------------------
#
# Every check takes the loaded bundle and returns a list of failure strings; an empty list is a
# pass. Returning the failures rather than raising is what lets one run report every broken
# invariant instead of only the first, which matters when the thing being audited is a run that
# took GPU hours to produce.


def check_file_set(bundle) -> list[str]:
    """Exactly eight files, no more and no fewer."""
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
    return failures


def check_raw_row_count(bundle) -> list[str]:
    """`per_scene.csv` holds `image_count x 3060` rows, and its decomposition is the design's.

    The decomposition is checked as well as the total because two errors multiply back to the
    same product -- a cache with four decoder layers and a lost selection, say -- and a single
    equality on 765,000 cannot tell that from a correct run.
    """
    scene = bundle.per_scene
    image_count = bundle.expected_image_count
    expected = image_count * ROWS_PER_IMAGE
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
    return failures


def check_candidate_coverage(bundle) -> list[str]:
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
    for candidate, frame in grouped:
        images = frame.groupby("image_id", sort=False)["severity"]
        if images.ngroups != bundle.expected_image_count:
            bad_image_counts.append((candidate, images.ngroups))
        for image_id, severities in images:
            observed = set(int(value) for value in severities)
            if observed != expected_severities or len(severities) != len(expected_severities):
                bad_severities.append((candidate, image_id, sorted(severities)))
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
    return failures


def check_candidate_key_unique(bundle) -> list[str]:
    """`candidate_metrics.csv` holds one row per candidate."""
    metrics = bundle.candidate_metrics
    duplicated = metrics[metrics.duplicated(list(CANDIDATE_KEY), keep=False)]
    failures = []
    if len(duplicated):
        keys = sorted({_candidate_id(row) for _, row in duplicated.iterrows()})
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
    return failures


def check_ranking_gates(bundle) -> list[str]:
    """Every ranked candidate is dynamic, filtered, persistence, layer 2, covered, orientable.

    Coverage is re-derived from the raw rows rather than read from the candidate: a ranked
    candidate must have `FULL_TUNING_IMAGE_COUNT` fully measured images *and* that many finite
    scores at each of the six severities, which is the pair of facts that makes its five AUROCs
    comparisons between two full groups.

    On a run over fewer images the gate can admit nothing, so the check becomes "the ranking is
    empty". That is the gate holding, and it is asserted rather than skipped -- a widened gate
    would otherwise pass this audit silently.
    """
    ranking = bundle.summary["deployable_ranking"]
    failures = []
    if bundle.expected_image_count != FULL_TUNING_IMAGE_COUNT:
        if ranking:
            failures.append(
                f"the run measured {bundle.expected_image_count} images, not "
                f"{FULL_TUNING_IMAGE_COUNT}, yet {len(ranking)} candidate(s) were ranked "
                "deployable"
            )
        return failures
    finite_by_candidate = bundle.finite_counts
    for candidate in ranking:
        key = _candidate_id(candidate)
        for field, required in DEPLOYABLE_GATES.items():
            if candidate[field] != required:
                failures.append(f"ranked candidate {key} has {field}={candidate[field]!r}")
        if candidate["measured_count"] != FULL_TUNING_IMAGE_COUNT:
            failures.append(
                f"ranked candidate {key} measured {candidate['measured_count']} images"
            )
        if candidate["image_count"] != FULL_TUNING_IMAGE_COUNT:
            failures.append(
                f"ranked candidate {key} has rows for {candidate['image_count']} images"
            )
        if candidate["orientation"] not in (-1, 1):
            failures.append(
                f"ranked candidate {key} has orientation {candidate['orientation']!r}"
            )
        thin = {
            severity: finite_by_candidate.get((key, severity), 0)
            for severity in EXPECTED_SEVERITIES
            if finite_by_candidate.get((key, severity), 0) != FULL_TUNING_IMAGE_COUNT
        }
        if thin:
            failures.append(f"ranked candidate {key} has incomplete severities {thin}")
    if len(ranking) != bundle.summary["validation"]["deployable_candidate_count"]:
        failures.append("summary.json deployable_candidate_count disagrees with the ranking")
    keys = [
        (
            -candidate["macro_auroc"],
            -candidate["median_absolute_spearman"],
            -candidate["dominant_direction_fraction"],
            -candidate["oriented_adjacent_consistency"],
            *(candidate[field] for field in CANDIDATE_KEY),
        )
        for candidate in ranking
    ]
    if keys != sorted(keys):
        failures.append("the deployable ranking is not sorted by its documented key")
    return failures


def check_macro_auroc(bundle) -> list[str]:
    """Each macro AUROC is the mean of that candidate's five per-severity AUROCs.

    Both directions are checked, because the failure that hides is the asymmetric one: a macro
    published beside five `None`s is a number computed from nothing, and five AUROCs with no
    macro is a candidate that cannot be ranked but looks measurable.
    """
    failures = []
    for _, row in bundle.candidate_metrics.iterrows():
        per_severity = [
            _cell(row[f"auroc_severity_{severity}"]) for severity in EXPECTED_SEVERITIES[1:]
        ]
        macro = _cell(row["macro_auroc"])
        key = _candidate_id(row)
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
    return failures


def check_spearman_pairs(bundle) -> list[str]:
    """Each row's absolute Spearman is non-negative and is `abs()` of its own signed Spearman.

    Row-wise and never candidate-wise: the median of the absolute values is not the absolute
    value of the median, so a check that compared the two published medians would fail on
    correct data and pass on a table where a single image's pair had been crossed.
    """
    scene = bundle.per_scene
    failures = []
    signed = pd.to_numeric(scene["signed_spearman"], errors="coerce")
    absolute = pd.to_numeric(scene["absolute_spearman"], errors="coerce")
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
        return failures
    lonely = signed.isna() ^ absolute.isna()
    if lonely.any():
        failures.append(
            f"{int(lonely.sum())} row(s) publish one Spearman and not the other; unmeasured "
            "must leave both empty"
        )
    return failures


def check_severity_statistics(bundle) -> list[str]:
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
    for key, scores in grouped:
        candidate, severity = key[:-1], int(key[-1])
        if severity not in EXPECTED_SEVERITIES or candidate not in metrics.index:
            continue
        values = pd.to_numeric(scores, errors="coerce").to_numpy(dtype=float)
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
    return failures


def check_axis_limits(bundle) -> list[str]:
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
        failures.append(
            f"axis_limits names {sorted(recorded)}, expected one range per signal "
            f"{sorted(PLOT_SCOPES)}"
        )
        return failures
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
    return failures


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


class Bundle:
    """The eight files, read once, plus the two derived tables every check would rebuild."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        self.per_scene = pd.read_csv(directory / "per_scene.csv")
        self.candidate_metrics = pd.read_csv(directory / "candidate_metrics.csv")
        self.expected_image_count = int(
            self.summary["validation"]["expected_image_count"]
        )
        scores = pd.to_numeric(self.per_scene["score"], errors="coerce")
        finite = self.per_scene[np.isfinite(scores)]
        self.finite_counts = (
            finite.groupby(list(CANDIDATE_KEY) + ["severity"], sort=False)
            .size()
            .to_dict()
        )
        self.finite_counts = {
            (key[:-1], int(key[-1])): value for key, value in self.finite_counts.items()
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path, help="a published report bundle")
    arguments = parser.parse_args(argv)
    directory = arguments.directory
    if not directory.is_dir():
        print(f"not a directory: {directory}", file=sys.stderr)
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
        return 2

    bundle = Bundle(directory)
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

    failed = 0
    for description, check in CHECKS:
        # An exception is a failed check, not a failed audit. A bundle broken enough to raise is
        # exactly the bundle whose remaining eight checks a reader most needs to see, and a
        # traceback out of check three would replace them with nothing.
        try:
            failures = check(bundle)
        except Exception as error:  # noqa: BLE001 - reported, not handled
            failures = [f"the check raised {type(error).__name__}: {error}"]
        if failures:
            failed += 1
            print(f"FAIL  {description}")
            for failure in failures:
                print(f"        {failure}")
        else:
            print(f"PASS  {description}")
    print()
    print(f"{len(CHECKS) - failed} of {len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
