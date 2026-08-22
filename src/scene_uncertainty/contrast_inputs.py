"""The four declared arms, and the only door a finished bundle comes through.

This module is the whole of the command's contact with the filesystem's input side. Everything
downstream sees a `ContrastInputs`, which is a flat dictionary of numbers plus provenance --
so no later module has to know that the source was a CSV, and no later module can quietly
widen the set of rows the experiment is allowed to see.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

PER_SCENE_FILE = "per_scene.csv"
SUMMARY_FILE = "summary.json"
SOURCE_FILES = (PER_SCENE_FILE, SUMMARY_FILE)
"""The two files of a finished bundle this command reads, and the whole of what it requires.

`write_corruption_report` publishes eight. The other six -- `candidate_metrics.csv`, the four
figures and `easy-report.md` -- are the producer's own summaries, and reading any of them would
make this command's numbers depend on a collapse it did not perform. Everything here comes from
`per_scene.csv`, which is the one file that still has an image and a severity on every row.
"""

SOURCE_PARTITION = "tuning"
FULL_TUNING_IMAGE_COUNT = 250
EXPECTED_SEVERITIES = tuple(range(6))
MEMBERSHIP_MODE = "dynamic"
PADDING_MODE = "filtered"
PERSISTENCE_SIGNAL = "persistence"
CONFIDENCE_SIGNAL = "confidence"
CONFIDENCE_SCOPE = "confidence"

COMBINED_SCOPE = "combined"
"""The one persistence scope whose scores are signed.

`decile_scoring.score_selection` builds `combined` as the mean over decoder layers of
`(score - center) / scale`, where `center` is the clean median distance -- so it is a robust
z-score, and it is negative for every selection sitting below that median. In the completed run
that is 548 negative per-severity `combined` statistics against none at `layer_2`, and two of
those negatives are arm 4's own bins. So "a negative persistence distance is corrupt input" is
true of `layer_N` and false here, and the refusal below has to know the difference.

Spelled here rather than imported from `decile_scoring.COMBINED_SCOPE`, which is the same string
for the same reason: that module builds these scores out of torch tensors and a feature cache,
and this command is a reporting-only consumer that must not acquire an import edge into the
scoring stack to learn one word. `contrast_scores` imports the name from here to exclude the
symmetric relative gap at this scope, whose scale invariance and +/-2 bounds both need
non-negative inputs.
"""

AGGREGATIONS = ("mean", "q90", "top20_mean")

SCORE_KEY = ("image_id", "severity", "signal", "confidence_bin", "aggregation", "score_scope")
"""What makes two retained scores measurements of different things.

`membership_mode` and `padding_mode` are absent because a row that is not `dynamic`/`filtered`
is never retained at all -- they are two of the selection predicates in `load_contrast_inputs`,
not fields a retained row can vary in. A real bundle carries `frozen` twins of every bin and an
`unfiltered` row at the lowest bin of each scheme, for the very series this command reads, so
these two fields do the work of *choosing* rows rather than of telling two chosen rows apart. A
key that carried them would quietly admit the twins this experiment is not measuring.

`bucket_scheme` is absent because `confidence_bin` already determines it -- `decile_00_10` and
`quintile_00_20` cannot collide -- and a redundant key field is a second place for two rows to be
told apart, which is one more than there should be.
"""


class ContrastInputError(ValueError):
    """A source bundle this command will not read.

    A `ValueError` so `pipeline.command_analyze_within_image_contrast` turns it into one line
    on stderr through the same `except ValueError` its siblings use, rather than a traceback
    through frames the operator did not write.
    """


@dataclass(frozen=True)
class Arm:
    """One bucket pair at one persistence scope, with the provenance of how it was chosen.

    `pair_name` is the bucket pair without the scope, and it is a stored field rather than a
    string derived from `name` because it is what the confidence-only twin is keyed on.
    Confidence has no decoder layer, so the two differential arms -- `layer_2` and `combined`
    over the same two buckets -- have one twin between them, and that identity has to be
    something the code states rather than something a suffix-stripping rule infers.

    `reference_bin` and `responsive_bin` are not interchangeable, and nothing in this module
    would notice if they were swapped: `required_series` reads them symmetrically. The asymmetry
    lives one task downstream, where `contrast_scores.raw_gap(reference, responsive)` turns a
    swap into a sign flip on every contrast in the experiment -- which is why the arm-table test
    pins all eight fields rather than the five that look like an arm's identity.
    """

    name: str
    pair_name: str
    family: str
    bucket_scheme: str
    score_scope: str
    reference_bin: str
    responsive_bin: str
    declared_before_data: bool


ARMS = (
    Arm("decile_00_10__50_60", "decile_00_10__50_60", "anchored", "decile", "layer_2",
        "decile_00_10", "decile_50_60", True),
    Arm("quintile_00_20__40_60", "quintile_00_20__40_60", "anchored", "quintile", "layer_2",
        "quintile_00_20", "quintile_40_60", True),
    Arm("decile_90_100__50_60", "decile_90_100__50_60", "differential", "decile", "layer_2",
        "decile_90_100", "decile_50_60", False),
    Arm("decile_90_100__50_60__combined", "decile_90_100__50_60", "differential", "decile",
        "combined", "decile_90_100", "decile_50_60", False),
)
"""The closed set. Four arms, and adding a fifth is a spec change.

`declared_before_data` is a field rather than a comment because it is the difference between a
result and a hypothesis: the two anchored arms were named before any tuning number was read,
and the two differential arms were named after the completed deployment analysis showed the
90--100 percent decile moving hardest and against the 50--60 percent range. A tuning macro
AUROC from an arm with `declared_before_data=False` is a selection estimate. Carrying the flag
all the way to the easy report is what stops it being quoted as performance.
"""


def required_series() -> tuple[tuple[str, str, str], ...]:
    """Every `(signal, confidence_bin, score_scope)` the four arms need, deduplicated.

    Derived from `ARMS` rather than listed, so an arm cannot be added without the loader
    demanding its rows. The confidence entries collapse across scope on purpose: confidence has
    no decoder layer, so the two differential arms need one confidence series between them and
    that is also why they share a single confidence-only twin downstream.
    """
    series: list[tuple[str, str, str]] = []
    for arm in ARMS:
        for confidence_bin in (arm.reference_bin, arm.responsive_bin):
            for entry in (
                (PERSISTENCE_SIGNAL, confidence_bin, arm.score_scope),
                (CONFIDENCE_SIGNAL, confidence_bin, CONFIDENCE_SCOPE),
            ):
                if entry not in series:
                    series.append(entry)
    return tuple(series)


REQUIRED_SERIES = required_series()


@dataclass(frozen=True)
class ContrastInputs:
    """The retained scores, the image roster, and where they came from."""

    scores: dict[tuple, float]
    image_ids: tuple[int, ...]
    provenance: dict


def _require_finished_bundle(source: Path) -> None:
    """Refuse a source missing either file, naming both when both are gone.

    One check before anything is parsed, rather than a `FileNotFoundError` handler at each
    `open`: "is this a finished bundle at all" has a single answer, and asking it per file means
    the refusal names whichever file the code happened to reach first instead of everything the
    operator has to go and produce. A file that disappears *between* this check and the read
    below is a genuine I/O race, and dressing that up as a validation error would send the
    operator to inspect their bundle when they should be inspecting their disk.
    """
    missing = [name for name in SOURCE_FILES if not (source / name).is_file()]
    if missing:
        raise ContrastInputError(
            f"source is not a finished corruption-sensitivity bundle: "
            f"{', '.join(missing)} missing from {source}"
        )


def _read_summary(source: Path) -> dict:
    try:
        return json.loads((source / SUMMARY_FILE).read_text())
    except json.JSONDecodeError as error:
        raise ContrastInputError(
            f"{SUMMARY_FILE} in {source} is not valid JSON: {error}"
        ) from error


def load_contrast_inputs(
    source_value: str | Path, *, expected_image_count: int = FULL_TUNING_IMAGE_COUNT
) -> ContrastInputs:
    """Every score the four arms need, and a refusal if the source cannot supply them all.

    Two things happen here and they are worth telling apart. Rows are *selected*: this command
    reads `dynamic` membership at `filtered` padding, three aggregations and the twelve series
    the arm table implies, out of a table that legitimately holds ten deciles, five quintiles,
    four scopes, both membership modes and both padding modes across 765,000 rows. And bundles
    are *refused*: for the partition they came from, for a series they do not contain, for a
    roster that is not the tuning set, for a disagreement between their two files. The first list
    is a filter and the second is an error, and being strict about which is which matters because
    a real bundle trips every entry on the first list on its way past. `frozen` and `unfiltered`
    rows are not a corrupt bundle; they are the producer's own controls, and the version of this
    function that refused them refused every bundle the producer can write.

    The one predicate on both lists is the partition, and it is tested on every row *before* the
    series filter rather than after it. A held-out row anywhere in this file means the table did
    not come from the tuning-only loader that every provenance claim downstream rests on, and
    skipping it because it sat in a series this command does not read is exactly the silent
    filtering that a held-out claim gets made by accident through.

    The refusals run outside-in: the summary's partition before a single row is parsed, because a
    held-out bundle should cost nothing to reject; then the rows; then the missing-series list,
    because "this bundle does not hold what this experiment needs" is a statement about the
    bundle and every count below it would be a statement about the wrong thing; then the image
    roster, the two files' agreement about it, and the severities; and the retained-row product
    last, because it is the only refusal whose message is a bare pair of numbers. Anything that
    can name its own problem raises first, so what reaches the product check is a shortfall
    nothing more specific could account for.
    """
    source = Path(source_value)
    _require_finished_bundle(source)
    summary = _read_summary(source)
    run = summary.get("run", {})
    if run.get("source_partition") != SOURCE_PARTITION:
        raise ContrastInputError(
            f"within-image contrast reads the tuning partition only; {source} reports "
            f"source_partition={run.get('source_partition')!r}"
        )

    wanted = set(REQUIRED_SERIES)
    scores: dict[tuple, float] = {}
    images: set[int] = set()
    seen_series: set[tuple[str, str, str]] = set()
    with (source / PER_SCENE_FILE).open(newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if row["source_partition"] != SOURCE_PARTITION:
                raise ContrastInputError(
                    f"{PER_SCENE_FILE} row {index} is outside the tuning partition: "
                    f"source_partition={row['source_partition']!r}"
                )
            entry = (row["signal"], row["confidence_bin"], row["score_scope"])
            if entry not in wanted:
                continue
            if row["membership_mode"] != MEMBERSHIP_MODE or row["padding_mode"] != PADDING_MODE:
                continue
            if row["aggregation"] not in AGGREGATIONS:
                continue
            score = float(row["score"])
            if not math.isfinite(score):
                raise ContrastInputError(
                    f"{PER_SCENE_FILE} row {index} has a non-finite score: {row['score']!r}"
                )
            if score < 0.0 and row["score_scope"] != COMBINED_SCOPE:
                raise ContrastInputError(
                    f"{PER_SCENE_FILE} row {index} has a negative {row['signal']} score at "
                    f"scope {row['score_scope']}: {score}; only {COMBINED_SCOPE} is signed"
                )
            key = (
                int(row["image_id"]), int(row["severity"]), row["signal"],
                row["confidence_bin"], row["aggregation"], row["score_scope"],
            )
            if key in scores:
                raise ContrastInputError(f"duplicate source row key: {key}")
            scores[key] = score
            images.add(key[0])
            seen_series.add(entry)

    missing = [entry for entry in REQUIRED_SERIES if entry not in seen_series]
    if missing:
        described = ", ".join(f"{signal}/{name}/{scope}" for signal, name, scope in missing)
        raise ContrastInputError(f"source is missing required series: {described}")

    image_ids = tuple(sorted(images))
    if len(image_ids) != expected_image_count:
        raise ContrastInputError(
            f"within-image contrast needs {expected_image_count} tuning images; "
            f"{source} has {len(image_ids)}"
        )
    if run.get("image_count") != len(image_ids):
        raise ContrastInputError(
            f"{SUMMARY_FILE} reports image_count={run.get('image_count')} but {PER_SCENE_FILE} "
            f"contains {len(image_ids)} images"
        )
    if tuple(run.get("severities", ())) != EXPECTED_SEVERITIES:
        raise ContrastInputError(
            f"within-image contrast needs severities {list(EXPECTED_SEVERITIES)}; "
            f"{source} reports {run.get('severities')}"
        )

    expected = len(REQUIRED_SERIES) * len(AGGREGATIONS) * len(image_ids) * len(EXPECTED_SEVERITIES)
    if len(scores) != expected:
        raise ContrastInputError(
            f"source coverage is incomplete: expected {expected} retained rows, found "
            f"{len(scores)}; every required series must cover every image at every severity"
        )

    return ContrastInputs(
        scores=scores,
        image_ids=image_ids,
        provenance={
            "source": str(source),
            "source_partition": SOURCE_PARTITION,
            "image_count": len(image_ids),
            "severities": list(EXPECTED_SEVERITIES),
            "membership_mode": MEMBERSHIP_MODE,
            "padding_mode": PADDING_MODE,
            "aggregations": list(AGGREGATIONS),
            "required_series": [list(entry) for entry in REQUIRED_SERIES],
            "retained_row_count": len(scores),
        },
    )
