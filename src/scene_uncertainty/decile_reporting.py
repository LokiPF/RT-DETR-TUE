"""Summarise the confidence-decile results table into trends, matched pairs, and a ranking.

Every hazard this module is built against has the same shape: a summary that is complete,
internally consistent, and describes a population nobody chose.

* A **group key that stops separating two selections** pools them. Dynamic and frozen
  membership answer different questions and must never combine into one score (spec:230); the
  design's own numbers make the cost concrete -- the pilot's bottom decile scores -0.029
  dynamic and +0.486 frozen, so one group covering both publishes something around 0.2 that
  describes neither. `GROUP_KEYS` carries all six labels for that reason, and
  `RANKABLE_MEMBERSHIP_MODES` is what keeps the frozen diagnostic out of the deployable
  ranking rather than merely below it.
* A **coverage gate that annotates instead of excluding** lets a config that collapsed at
  high severity outrank one that rose the whole way, because both publish the same median
  Spearman over the severities they survived. Spec:159 makes full coverage a *requirement* for
  a candidate to be ranked deployable, so `rank_deployable_groups` drops it from the list
  instead of writing a flag beside it.
* A **paired statistic whose denominator is not stated** is the failure spec:157 guards
  against, and it has two halves. The fraction of images where persistence beats its
  confidence control is only meaningful over images both signals scored, so
  `paired_image_count` is published next to it. And a per-image Spearman over six severities
  takes only 35 distinct values, so exact ties are common and a rate that counts them as
  non-wins reads as a defeat: every paired rate here is published as a win/tie/loss triple
  over the full population *and* again over the decided images alone, because either
  denominator on its own inverts the sentence a reader writes.
* A **raw score magnitude leaking into a comparison** (spec:157 again). Persistence distances
  and `1 - confidence` are in unrelated units; the only fair comparison is between their
  *trends*. Every number this module publishes about a score is invariant under
  `score -> a * score + b` for `a > 0`, and a test asserts exactly that over the whole summary.

`summarize_decile_rows` is also where the design's "output result keys would be duplicated"
refusal lives (spec:194). `analyze_deciles` cannot produce a collision and has a test proving
it does not, but rows also arrive from a CSV, from two concatenated runs, or from a future
producer -- so the check belongs at the boundary rows arrive through, and here it is.

`write_decile_report` adds the seven artifacts spec:163-171 names, and three more hazards
come with them -- all of them invisible to a reader of the output:

* A **figure drawn over the bins that happen to exist** publishes a complete-looking ten-bin
  measurement over whatever subset the table held, in whatever order pandas grouped them.
  Every axis here is built from `DECILE_NAMES` and carries all ten positions whatever the
  table holds; a bin with no row is a blank cell or an absent bar, which a reader can see. A
  panel with nothing at all to draw says so in words rather than showing an empty axis, which
  reads as a measured zero.
* A **sentence that outruns its numbers**, which is the one artifact no schema check reaches.
  Three specific sentences are forbidden outright and the generator cannot produce them: a
  score described as a probability of corruption (spec:111), a dynamic and a frozen result
  combined into one recommendation (spec:230), and a paired rate quoted with one denominator.
  Every verdict word `_easy_report` emits is chosen from the win and loss *counts* and is
  followed by both denominators, and no verdict claims significance -- the candidate was
  selected best-of-N on the images it is reported over, so the counts describe this run and
  the held-out run spec:224-225 orders is the confirmation.
* A **half-written artifact that reads as a finished one**. Every file goes out through a
  temporary and `os.replace`, and the summary is computed *before* the output directory is
  created, so a table the summariser refuses leaves no directory behind at all.

The writer builds the results frame exactly twice over its whole life and never at the same
time: once inside `summarize_decile_rows`, which drops it, and once afterwards for the CSV and
the blur curves, which share it. `pd.DataFrame(rows)` on the real table costs ~280 MB in
`selected_query_ids` alone, so the order matters -- see `summary_frame`.
"""

from __future__ import annotations

import json
import math
import operator
import os
from pathlib import Path

import matplotlib

# The report is written on a headless box, so the backend is pinned before pyplot is
# imported rather than left to whatever matplotlib would autodetect.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .confidence_deciles import DECILE_NAMES  # noqa: E402
from .decile_analysis import (  # noqa: E402
    ALL_QUERY_BENCHMARK,
    ALL_VALID_BENCHMARK,
    EXPECTED_SEVERITIES,
    ROW_KEYS_EXCLUDED_FROM_CSV,
    SENSITIVITY_BIN,
)
from .decile_scoring import (  # noqa: E402
    COMBINED_SCOPE,
    CONFIDENCE_BINS,
    CONFIDENCE_SCOPE,
    DECILE_AGGREGATIONS,
    MEMBERSHIP_MODES,
    PADDING_MODES,
    PRIMARY_SCORE_SCOPE,
)
from .metrics import monotonicity_metrics  # noqa: E402


GROUP_KEYS = (
    "signal", "membership_mode", "confidence_bin", "aggregation", "score_scope", "padding_mode"
)
"""What makes two rows measurements of *different* things rather than of the same thing twice.

All six, together. `confidence_bin` alone does not identify a selection: the all-300-query
benchmark and the padding-removed `all_valid` benchmark carry the *same* bin label and differ
only in `padding_mode`, and the bottom decile appears under three membership modes. Grouping on
any subset silently merges selections whose whole purpose is to be compared against each other.
"""

ROW_KEYS = ("image_id", "severity", *GROUP_KEYS)
"""One scored row per key. `source_partition` is deliberately *not* in it -- see
`_checked_partition`, which refuses a table spanning two partitions outright rather than
letting the key absorb the difference and the groups pool across it."""

REQUIRED_ROW_KEYS = (*ROW_KEYS, "source_partition", "score", "selected_count", "clean_overlap")

SIGNALS = ("persistence", "confidence")
"""The two matched uncertainty signals (spec:99-111).

These are literals in `decile_scoring.score_selection` too, which is one rule in two places.
`test_the_signal_vocabulary_matches_what_the_scorer_emits` calls the scorer and compares, so
the drift fails loudly instead of opening a phantom group named after a signal nobody scores.
"""

RANKABLE_MEMBERSHIP_MODES = ("dynamic", "shared")
"""Which memberships may be *ranked deployable*, and why `frozen` is not among them.

Spec:230: "Dynamic and frozen results answer different questions and must not be combined into
one score. A strong frozen-only result shows useful diagnostic information but does not by
itself define a deployable single-image method." Frozen bins are built from severity zero, and
a naturally corrupted image has no paired clean version to build them from -- so a frozen row
is a measurement of how far fingerprints moved, not a policy anyone can run.

This matters numerically and not only in principle. On the pilot the bottom decile scores
-0.029 dynamic and +0.486 frozen; a ranking that admitted frozen would put the frozen bottom
bin near the top and read as a recommendation. `shared` is rankable because it is neither: it
is a selection no confidence ranking produced -- `all_valid` -- that is the same query set at
every severity and can be computed on a single image with no clean reference.
"""

RANKED_GROUPS_KEY = f"ranked_{PRIMARY_SCORE_SCOPE}_persistence"
"""The summary key holding the deployable ranking, named after the primary scope rather than
spelling `layer_2` twice. Layer 2 is the design's primary persistence scope; layers 0 and 1 and
`combined` are secondary diagnostics and are summarised but never ranked."""

DEFINITIONAL_OVERLAP_MODES = ("frozen", "shared")
"""The membership modes whose clean-bin overlap is 1.0 by construction at every severity.

`frozen` reuses severity zero's query IDs, and `shared` is a selection no confidence ranking
produced and that is the same query set at every severity -- so neither's overlap is a
measurement. `clean_overlap` is still *recorded* for them rather than asserted, because
writing the constant would make a broken freeze look healthy in the one column that exists to
detect movement; but a reader of the ranked table needs to know which 1.000 is evidence and
which is arithmetic. Three of the pilot's 33 ranked candidates are `shared`, and they print
1.000 beside dynamic bins at 0.059 in the same table.
"""

FILTERED_PADDING_MODE = "filtered"
UNFILTERED_PADDING_MODE = "unfiltered"

MEMBERSHIP_COMPARISON_PAIR = ("dynamic", "frozen")
"""The two memberships `membership_comparisons` pairs, and why that is a comparison and not the
combination spec:230 forbids.

Spec:230 says dynamic and frozen "must not be combined into one score". Nothing here combines
them: no sum, no mean, no difference of raw scores, no single ranked figure covering both. What
this family publishes is the same shape the design already asks for between persistence and its
confidence control (spec:157) -- two separate medians and a per-image win/tie/loss triple over
the images both measured -- and a triple is a statement about *how often one exceeds the other*,
which cannot be a combined score because it is not on the score's scale at all. The frozen group
stays out of the deployable ranking exactly as before, and this family adds no way into it.

Why it has to exist. Spec:172 makes the easy report answer whether bin movement explains the
result, and freezing the membership at severity zero is the only manipulation in the design that
removes movement while leaving everything else. Before this family, that question was answered
by subtracting two medians -- on the pilot +0.6286 against +0.6000, one step of the metric's
`1/35` grid. This plan has already shown that comparison to be blind at that resolution: the
leading candidate's median difference against the all-query benchmark is exactly 0.000 at `q90`
while the paired comparison runs 122 to 92. So the fourth question was being answered by the one
statistic the other three had stopped trusting.
"""

BENCHMARK_AGGREGATION = "q90"
"""The scene summary the figures slice on, and the one the published benchmark exists at.

Spec:131 scores the all-300-query benchmark at `q90` alone, so it is the only summary in
which every row a figure wants is present. The figures say so on their own titles rather than
letting a reader assume they show the winner, which may well be at another summary -- the
easy report is where all three summaries of the winning bin are laid out."""

SPEARMAN_STEP_DENOMINATOR = (
    len(EXPECTED_SEVERITIES) * (len(EXPECTED_SEVERITIES) ** 2 - 1) // 6
)
"""How coarse the primary metric is, derived rather than quoted: 35 for six severities.

A per-image Spearman over `n` severities is `1 - 6*S/(n*(n^2-1))` with `S` a sum of squared
rank differences, so it lands on a grid of `1/35` -- and because `S` is always even, on a grid
of `2/35 = 0.057` for a single image. Only 36 values exist at all. A *median* over an even
number of images is the mean of two of them and so lands on the `1/35 = 0.029` grid.

This is why the design's primary metric needs the paired comparison beside it and not instead
of it: on the pilot the leading candidate's median difference against the benchmark is exactly
0.000 at `q90` while the per-image comparison favours it 122 to 92. A dead heat at this
resolution is not a dead heat in the data."""

RANDOM_BIN_OVERLAP = 1.0 / (2 * len(DECILE_NAMES) - 1)
"""What a Jaccard overlap between two *unrelated* decile memberships would be, derived.

Ten equal-count bins over one valid population of `N` queries give bins of `m = N/10`. Two
independently drawn bins share `m^2/N` queries in expectation and cover `2m - m^2/N`, so the
ratio is `1/(2*10 - 1) = 1/19 = 0.0526` for any `N`. It follows from the bin count alone,
which is why it is computed from `DECILE_NAMES` instead of written down.

It is the reference the design's dynamic-bin overlaps have to be read against (spec:97). On
the pilot, nine of the ten dynamic bins sit at 0.057-0.078 from severity 1 onward -- barely
above this line -- and only `decile_90_100` reaches 0.248. Without the line, nine numbers
within 0.02 of each other read as "membership was reasonably stable"."""

EASY_REPORT_TITLE = "# Confidence-Decile Blur Experiment"

EASY_REPORT_FINAL_SENTENCE = "The held-out test images were not used."
"""The last line of every easy report, and a statement about the run rather than a hope.

The loader refuses a result artifact built for the held-out partition and the summariser
refuses a table spanning two, so by the time this sentence is written it has already been
enforced twice. It is last because it is the sentence that decides what any number above it
is worth: spec:224-225 makes the tuning run a *selection* and the held-out run the
confirmation, and a reader who stops early should still meet it."""

SPEC_172_QUESTIONS = (
    "Which confidence range worked best",
    "Did it beat confidence alone",
    "Did it beat the existing all-query benchmark",
    "Does padding or bin movement explain the result",
)
"""Spec:172's four questions, in its order, as the literal headings of the opening section.

"The easy report must lead with which confidence range worked best, whether it beat confidence
alone, whether it beat the existing all-query benchmark, and whether padding or bin movement
explains the result." Written as a constant so the order is a thing a test asserts rather than
a property of however the paragraphs were typed."""

_UNMEASURED = "not measured"


def describe_row_key(key) -> str:
    return ", ".join(f"{column}={value}" for column, value in zip(ROW_KEYS, key))


def _finite(values) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def _finite_mean(values) -> float | None:
    """The mean of the finite values, or `None` when there are none.

    `None` rather than `nan` or `0.0`: the design forbids replacing missing data with zero, and
    a `nan` cannot be written to `summary.json` at all -- `json.dumps(..., allow_nan=False)`
    refuses it, and with `allow_nan=True` it emits a bare `NaN` token that is not valid JSON
    and that half the readers in the world parse as a string.
    """
    finite = _finite(values)
    return float(finite.mean()) if finite.size else None


def _finite_median(values) -> float | None:
    finite = _finite(values)
    return float(np.median(finite)) if finite.size else None


def _difference(left: float | None, right: float | None) -> float | None:
    """A difference that goes missing when either side is missing, rather than reading as zero."""
    if left is None or right is None:
        return None
    return float(left - right)


def _is_persistence_scope(scope: str) -> bool:
    if scope == COMBINED_SCOPE:
        return True
    if not scope.startswith("layer_"):
        return False
    suffix = scope[len("layer_"):]
    return suffix.isdigit()


def _checked_labels(frame: pd.DataFrame) -> None:
    """Refuse a row whose label is outside the closed vocabulary it belongs to (spec:193).

    Checked over the distinct group tuples rather than row by row, so the cost is 349 checks
    and not 523,500 -- and so the message names the group that would have appeared.

    A typo does not fail a group-by. It opens a phantom group holding a handful of rows and
    subtracts exactly those rows from the group they belonged to, and nothing downstream
    reports either half. The scorer already refuses three of these six vocabularies on the way
    in; the other three (`signal`, `aggregation`, `score_scope`) it derives rather than
    accepts, so this is the first place a table assembled from anywhere else is checked at all.
    """
    vocabularies = {
        "signal": SIGNALS,
        "membership_mode": MEMBERSHIP_MODES,
        "confidence_bin": CONFIDENCE_BINS,
        "aggregation": DECILE_AGGREGATIONS,
        "padding_mode": PADDING_MODES,
    }
    for column, allowed in vocabularies.items():
        unknown = sorted({value for value in frame[column].unique() if value not in allowed})
        if unknown:
            raise ValueError(
                f"unknown {column} {unknown} in the results table, expected {list(allowed)}"
            )
    scopes = sorted(set(frame["score_scope"].unique()))
    unknown = [
        scope for scope in scopes if scope != CONFIDENCE_SCOPE and not _is_persistence_scope(scope)
    ]
    if unknown:
        raise ValueError(
            f"unknown score scope {unknown}; expected {CONFIDENCE_SCOPE!r}, {COMBINED_SCOPE!r} "
            "or layer_<id>"
        )
    # Spec:125 -- "Confidence uncertainty has no decoder-layer scope." The two labels are
    # redundant by design, and the redundancy is the check: a confidence row filed under
    # `layer_2` would be looked up as its own persistence partner and the matched comparison
    # would report a difference of exactly zero for a reason that is not about the experiment.
    mismatched = sorted({
        (signal, scope)
        for signal, scope in zip(frame["signal"], frame["score_scope"])
        if (signal == "confidence") != (scope == CONFIDENCE_SCOPE)
    })
    if mismatched:
        raise ValueError(
            f"signal and score scope disagree for {mismatched}: the confidence control has no "
            f"decoder-layer scope and is stored once under {CONFIDENCE_SCOPE!r}"
        )


def _checked_partition(frame: pd.DataFrame, run_metadata: dict) -> None:
    """Refuse a table spanning two partitions, or one that contradicts its own provenance.

    `source_partition` is not part of `ROW_KEYS`, and that is the strict choice rather than the
    loose one: if it were a key, a tuning row and a held-out row for the same image and
    severity would be two legitimately distinct rows *and* would land in the same group,
    because the design's metrics are per signal, bin, membership, summary and scope and not per
    partition. The group's median would then be taken over a mixture of the partition the
    experiment is allowed to select on and the one it is not (spec:196), and no published
    number would say so.

    So the rule is one partition per summary, and it must be the partition the run metadata
    claims. Which partition is *permitted* stays with the loader: the design forbids the
    held-out set for this experiment and anticipates a sanctioned held-out run once the tuning
    conclusion is reviewed, and only the loader can tell those apart.
    """
    partitions = sorted(set(frame["source_partition"].unique()))
    if len(partitions) > 1:
        raise ValueError(
            f"the results table spans more than one source partition {partitions}; summarising "
            "them together would take one median over a mixture of partitions"
        )
    declared = (run_metadata or {}).get("source_partition")
    if declared is not None and partitions[0] != declared:
        raise ValueError(
            f"the results table is in the {partitions[0]!r} source partition but the run "
            f"metadata declares {declared!r}"
        )


def summary_frame(rows: list[dict]) -> pd.DataFrame:
    """The scored rows as a frame, with the excluded columns dropped *before* the conversion.

    The order is the whole point. Task 5 measures the real tuning table at 523,500 rows and
    453 MB, of which roughly 280 MB is the `selected_query_ids` list on every row.
    `pd.DataFrame(rows).drop(columns=...)` materialises all of it and then throws it away, so
    for the duration of the conversion the resident set holds both copies. Building the frame
    column by column from `ROW_KEYS_EXCLUDED_FROM_CSV`'s complement never reads those lists at
    all -- and, as a side effect worth having, it accepts a table that was written without them
    instead of raising `KeyError` on a column that is legitimately absent.

    The constant is imported rather than re-spelled. A literal here would keep working on the
    day the producer adds a second heavy column, and would keep working quietly.

    This is also exactly the frame `per_scene.csv` wants -- every scored column, none of the
    excluded ones -- so the writer should hand its rows to this function rather than build a
    second frame and drop the columns by name afterwards.
    """
    if not rows:
        raise ValueError("summarize_decile_rows needs at least one scored row")
    # First-seen order over the union of every row's keys, so a table whose rows are not all
    # the same shape is described by its own contents rather than by whichever row came first.
    present = dict.fromkeys(key for row in rows for key in row)
    kept = [key for key in present if key not in ROW_KEYS_EXCLUDED_FROM_CSV]
    missing = [key for key in REQUIRED_ROW_KEYS if key not in kept]
    if missing:
        raise ValueError(f"the results table is missing the required columns {missing}")
    columns = {}
    for name in kept:
        try:
            columns[name] = [row[name] for row in rows]
        except KeyError:
            raise ValueError(
                f"every scored row must carry {name!r}; one of them does not"
            ) from None
    return pd.DataFrame(columns)


def find_duplicate_row_key(frame: pd.DataFrame) -> tuple | None:
    """The first `ROW_KEYS` tuple that appears twice, or `None` (spec:194).

    A duplicated key is not cosmetic. Both copies land in the same image's severity sequence,
    so `monotonicity_metrics` sees seven points across six severities with one severity
    weighted twice; the Spearman shifts, the adjacent-step count grows by one, and
    `total_severity_count` reads 13 against an expectation of 12 -- which is the only visible
    trace, and only to a reader who knows what to expect. Two concatenated result CSVs are the
    obvious way to arrive here, and nothing about them looks wrong.
    """
    duplicated = frame.duplicated(subset=list(ROW_KEYS)).to_numpy()
    if not duplicated.any():
        return None
    first = int(np.flatnonzero(duplicated)[0])
    return tuple(frame[column].iloc[first] for column in ROW_KEYS)


def _image_metrics(image_ids, severities, scores) -> dict[int, dict]:
    """`monotonicity_metrics` per image, over arrays already sorted by image then severity.

    The severities with no score are handed in rather than filtered out. That is what makes
    `finite_count` and `total_count` mean anything: a policy that collapsed at high severity
    and one that rose the whole way publish the same Spearman, and the counts are the only
    thing that separates them -- which is why the deployable gate is built on them.

    The three arrays are arguments rather than something this function extracts, because the
    caller keeps them: the padding-sensitivity pair has to ask whether the mask changed an
    image's *scores*, and a per-image Spearman cannot answer that question.
    """
    cuts = np.flatnonzero(image_ids[1:] != image_ids[:-1]) + 1
    starts = np.concatenate(([0], cuts))
    stops = np.concatenate((cuts, [image_ids.size]))
    return {
        int(image_ids[start]): monotonicity_metrics(severities[start:stop], scores[start:stop])
        for start, stop in zip(starts, stops)
    }


def is_full_coverage(group: dict) -> bool:
    """Whether a group scored every image at every expected severity (spec:159).

    Three conditions, not one. `scored == total` says nothing was lost to an empty selection;
    `total == image_count * 6` says nothing was lost *before* the row was built, which is the
    case `scored == total` cannot see -- a group whose severity-5 rows were never produced has
    five real scores per image, all of them finite, and no measurement of the blur level the
    experiment is about.

    The third, `scored_image_count == image_count`, is the one worth saying when it bites. On a
    table `analyze_deciles` produced it is implied by the other two, because the loader
    guarantees six severities per image. This function does not re-check that rule and so
    cannot rely on it: the first two conditions constrain only the *sums*, so a group holding
    eleven severities for one image and one severity for another satisfies both while the
    second image has no Spearman at all -- `monotonicity_metrics` reports `nan` for it, and it
    leaves the denominator of every statistic the ranking sorts on without appearing anywhere.
    """
    expected = group["image_count"] * len(EXPECTED_SEVERITIES)
    return bool(
        group["image_count"] > 0
        and group["scored_severity_count"] == group["total_severity_count"] == expected
        and group["scored_image_count"] == group["image_count"]
    )


def _orderable(value: float | None, absent: float) -> float:
    return absent if value is None or not math.isfinite(value) else float(value)


def _ranking_sort_key(group: dict) -> tuple:
    """Spec:159's order -- median Spearman, then adjacent non-decrease, then violation.

    Descending, descending, ascending, and then the group's own key. That last term is not
    decoration: without it two groups with identical statistics keep whatever order the input
    happened to have, and the "best confidence range" sentence in the report would change
    between two runs over the same data. A group whose trend could not be measured sorts to the
    end rather than raising, and keeps its `None`s in the published row so the reason is
    visible.
    """
    return (
        -_orderable(group["median_spearman"], -math.inf),
        -_orderable(group["mean_adjacent_monotonicity"], -math.inf),
        _orderable(group["mean_violation_magnitude"], math.inf),
        tuple(str(group[key]) for key in GROUP_KEYS),
    )


def rank_deployable_groups(groups: list[dict]) -> list[dict]:
    """The full-coverage, filtered, primary-scope persistence groups a run could deploy.

    Four gates before the sort, and each excludes rather than annotates:

    * primary scope only -- layers 0 and 1 and `combined` are secondary diagnostics (spec:123);
    * `filtered` only -- the unfiltered rows are the padding *sensitivity control* (spec:69),
      and the all-300-query row among them is the benchmark being measured against, not a
      candidate;
    * `RANKABLE_MEMBERSHIP_MODES` only -- spec:230, frozen is diagnostic;
    * full coverage -- spec:159 makes it a requirement, so an under-covered group is absent
      from the list and not merely flagged inside it.
    """
    candidates = [
        group for group in groups
        if group["signal"] == "persistence"
        and group["score_scope"] == PRIMARY_SCORE_SCOPE
        and group["padding_mode"] == FILTERED_PADDING_MODE
        and group["membership_mode"] in RANKABLE_MEMBERSHIP_MODES
        and is_full_coverage(group)
    ]
    return sorted(candidates, key=_ranking_sort_key)


BENCHMARK_SELECTION = ("persistence", *ALL_QUERY_BENCHMARK[:2], PRIMARY_SCORE_SCOPE,
                       ALL_QUERY_BENCHMARK[2])
"""The published all-300-query result every deployable candidate is measured against.

Five of the six group labels; the scene summary is left open because the producer scores this
selection at `q90` alone (spec:131) and a later run that published it at three summaries
should be compared against each, not against whichever one this module guessed. The triple
comes from `decile_analysis.ALL_QUERY_BENCHMARK` rather than being re-spelled, because it is
an encoding decision recorded there and a second copy here would be a second thing to drift.

It is *not* a deployable candidate itself -- it is `unfiltered`, so it scores the padded
decoder placeholders that spec:71 removes from the primary analysis. It is the bar, not a
runner.
"""


def _benchmark_keys(by_key: dict[tuple, dict]) -> list[tuple]:
    return sorted(
        key for key in by_key
        if (key[0], key[1], key[2], key[4], key[5]) == BENCHMARK_SELECTION
    )


def _paired(left: dict[int, float], right: dict[int, float]) -> list[int]:
    """The images both groups measured a trend for -- the only ones a per-image win can use."""
    return sorted(set(left) & set(right))


def _rate(left, right, images: list[int], compare) -> float | None:
    """The fraction of `images` on which `compare(left, right)` holds, or `None` for none.

    `None` and not `0.0`: an empty population has no rate, and a published `0.0` reads as a
    measured failure rather than as a comparison nobody could make.
    """
    if not images:
        return None
    return float(np.mean([compare(left[image], right[image]) for image in images]))


def _outcome_rates(prefix: str, left, right, images: list[int]) -> dict:
    """Win, tie and loss over `images`, and the win rate again over the *decided* ones only.

    Both denominators, always, because either one alone inverts the reading. A per-image
    Spearman over six severities takes 35 distinct values, so exact ties are common -- 15.6
    percent of images on the pilot's leading candidate against the all-query benchmark. The
    rate over all paired images counts every tie as a non-win, which pushes a candidate that
    wins 122 images and loses 88 below 0.5 and invites the sentence "it loses the per-image
    majority", which is false. The rate over decided images alone hides how much of the
    population the comparison could not separate at all.

    So five numbers rather than one, and the three rates over `images` sum to 1.0. Nothing
    here is a significance test: a sign test over these counts is available to a reader, but
    it would be selection-conditioned for whichever candidate the ranking put first -- that
    candidate was chosen best-of-33 on the same tuning data -- and publishing a bare p-value
    next to a ranked row would invite exactly the reading spec:224-225 reserves for the
    held-out run.
    """
    decided = [image for image in images if left[image] != right[image]]
    return {
        f"{prefix}_win_rate": _rate(left, right, images, operator.gt),
        f"{prefix}_tie_rate": _rate(left, right, images, operator.eq),
        f"{prefix}_loss_rate": _rate(left, right, images, operator.lt),
        f"{prefix}_decided_image_count": len(decided),
        f"{prefix}_decided_win_rate": _rate(left, right, decided, operator.gt),
    }


def _score_index(arrays) -> dict[tuple[int, int], float]:
    image_ids, severities, scores = arrays
    return {
        (int(image_id), int(severity)): float(score)
        for image_id, severity, score in zip(image_ids, severities, scores)
    }


def _same_score(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return False
    if math.isnan(left) and math.isnan(right):
        return True
    return left == right


def _score_changed_images(left_arrays, right_arrays) -> set[int]:
    """The images whose scene *score* differs between two padding modes.

    **This is a lower bound on the set of images the padding mask actually reached, and it is
    named for what it measures rather than for what a reader would like it to mean.** The mask
    reaches an image whenever it changes that image's *selection*, and a changed selection can
    still produce a bit-identical scene score at every severity -- a summary is a many-to-one
    map. The proof needs no per-image detail: on the pilot the dynamic bottom-bin pair at layer 2
    reports 66 changed images under `mean`, 64 under `q90` and 65 under `top20_mean`, and a set
    of "images the mask reached" cannot depend on the summary applied afterwards.

    Measured against the selection-derived truth -- 66 for the dynamic bottom bin and for
    `all_valid`, and 65 for the frozen bottom bin, whose selection genuinely coincides under both
    padding rules on one image -- 27 of the 34 sensitivity rows recover it exactly and 7
    understate it by one or two. The error is bounded and one-directional: this can never
    overstate the reached set, because a changed score requires a changed selection.

    The exact set would need `selected_query_ids`, the ~280 MB column `summary_frame` drops
    before the DataFrame conversion, and buying exactness back at that price is the wrong
    trade for a diagnostic. What this rules out is the far larger error it replaced: reading
    the set off the per-image *Spearman*, which takes only 35 distinct values over six
    severities and reported 50 to 56 of 250 -- a tie-excluding sign test wearing the label
    "the images the control could reach".

    An image scored on one side and absent from the other counts as changed, and two `nan`
    scores count as unchanged -- both severities were unscored under either rule, which is the
    mask making no difference rather than making one nobody can measure.
    """
    left = _score_index(left_arrays)
    right = _score_index(right_arrays)
    return {
        key[0] for key in left.keys() | right.keys()
        if not _same_score(left.get(key), right.get(key))
    }


def _jsonable(value):
    """Plain JSON values, with every non-finite number turned into `null`.

    NumPy scalars are converted rather than tolerated: `json.dumps` refuses `np.float64`
    outright, and `np.bool_` is not a `bool`, so a boolean rate would arrive at the writer as
    an unserialisable object after the summary had already been computed. An unrecognised type
    raises here instead of being coerced to a string, because a `Tensor` or a `Path` that
    reached a summary field is a bug in the field and not a formatting problem.
    """
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"{type(value).__name__} cannot be written to summary.json: {value!r}")


def _padding_rollup(diagnostics: dict | None) -> dict:
    """The design's padded-query counts (spec:155), rolled up and passed through.

    The per-image detail is carried rather than summarised away, because the rollup alone
    cannot answer the question the padding control exists for. On the pilot, 184 of 250 images
    carry no padding at all, so the filtered and unfiltered bottom bins select the identical
    queries there and the control is a no-op *by construction* on 74 percent of the run. A
    reader who sees only a diluted 250-image median reads that as a null result.
    """
    images = dict((diagnostics or {}).get("images", {}))
    counts = [int(image["union_padded_count"]) for image in images.values()]
    return {
        "image_count": len(images),
        "images_with_padding": sum(1 for count in counts if count > 0),
        "total_union_padded_count": sum(counts),
        "max_union_padded_count": max(counts) if counts else None,
        "images_with_identical_tails": sum(
            1 for image in images.values() if image["tail_identical_across_severities"]
        ),
        # The same fact restricted to the images that *have* a tail. An image with no padding
        # has six identical empty tails and counts as "identical" above, so on the pilot the
        # unrestricted number reads 184 of 250 and invites the conclusion that padding is
        # stable. It is not: of the 66 images that carry padding, the detected tail differs
        # across severities on all 66, which is the fact the union mask exists for.
        "padded_images_with_identical_tails": sum(
            1 for image in images.values()
            if int(image["union_padded_count"]) > 0
            and image["tail_identical_across_severities"]
        ),
        "images": images,
    }


def summarize_decile_rows(
    rows: list[dict], run_metadata: dict | None = None, diagnostics: dict | None = None
) -> dict:
    """Group the results table, pair each signal with its control, and rank what is deployable.

    Refuses first, on the four failures the design names that can reach this layer: an empty
    table, a duplicated output result key (spec:194), a label outside its closed vocabulary
    (spec:193), and a table spanning two source partitions (spec:190, 196). Each refusal names
    the offending value, because the alternative is sending a reader back to half a million
    rows.

    What comes out, and which sentence of the design each part answers:

    * `groups` -- metrics per signal, confidence bin, membership mode, scene summary and
      persistence scope (spec:141), carrying the primary median Spearman and every supporting
      metric spec:149-155 lists.
    * `comparisons` -- for each matched persistence/confidence pair, the difference in median
      Spearman and the fraction of images where persistence out-trends its control (spec:157).
      The difference is between the two group medians, which is what the design asks for and is
      therefore a marginal statistic; `paired_image_count` sits beside it so a reader can see
      whether the two medians rest on the same images. The win fraction is the paired one.
    * `padding_sensitivity` -- the same selection with and without the padding union (spec:155),
      including `score_changed_image_count` and the outcomes restricted to those images,
      because the control is a no-op on any image that had no padding to remove and the
      unrestricted rate counts every such image as a non-win. That count is a *lower bound* on
      the images the mask reached and varies with the scene summary; `_score_changed_images`
      says why, and why buying the exact set back is the wrong trade.
    * `membership_comparisons` -- each dynamic selection against its own frozen twin, paired
      per image (spec:172's fourth question). Freezing the membership at severity zero is the
      only manipulation in the design that removes query movement and leaves everything else,
      so the paired outcome between the two is what says whether the movement explains a
      result. Two medians one grid step apart cannot say it: this plan has a candidate whose
      median difference against the benchmark is exactly 0.000 while the paired comparison runs
      122 to 92. Nothing here combines the two into one score -- see
      `MEMBERSHIP_COMPARISON_PAIR`.
    * `benchmark_comparisons` -- every ranked candidate against the published all-300-query
      result (spec:229), paired per image. The spec asks only whether a candidate beats the
      benchmark; a difference of two medians answers that with a number whose resolution is
      1/35, because a Spearman over six severities takes 35 distinct values. The win, tie and
      loss rates say how many of the images the difference actually rests on, which is what
      makes a margin of one such step falsifiable rather than merely reportable -- and
      `candidate_image_decided_win_rate` says it again without the ties in the denominator,
      because on the pilot the two framings put the same candidate on opposite sides of 0.5.
    * the deployable ranking under `RANKED_GROUPS_KEY` -- spec:159. Note what it cannot say:
      every candidate in it is `filtered`, so the ranking never *chooses* the padding rule
      that spec:224 asks the tuning run to select. That evidence is in `padding_sensitivity`,
      and the two must not be read as one recommendation.

    No number here is a raw score magnitude, and none compares one. Persistence distance and
    `1 - confidence` share no unit; every statistic published about a score -- Spearman, the
    adjacent non-decrease rate, the range-normalised violation magnitude, the maximum-blur-
    above-clean rate -- is invariant under a positive affine map of that score, which is the
    only sense in which the two signals can be compared at all (spec:157).

    `diagnostics` is the second half of `analyze_deciles`' return and is optional: without it
    the padding rollup reports an image count of zero, which reads as "not measured" rather
    than as "no padding found".
    """
    frame = summary_frame(rows)
    duplicate = find_duplicate_row_key(frame)
    if duplicate is not None:
        raise ValueError(
            "the results table has a duplicate output result key: "
            f"{describe_row_key(duplicate)} appears more than once. Two copies of one "
            "measurement weight that severity twice in its own image's trend, which shifts "
            "the Spearman and inflates the severity counts with no error anywhere."
        )
    _checked_labels(frame)
    _checked_partition(frame, run_metadata or {})

    frame = frame.sort_values(
        [*GROUP_KEYS, "image_id", "severity"], kind="stable", ignore_index=True
    )
    groups: list[dict] = []
    spearman_by_group: dict[tuple, dict[int, float]] = {}
    # The (image, severity, score) arrays of every group, kept so that the padding pair can
    # ask whether the mask changed an image's scores. Three int/float arrays per group over
    # the real table is about 13 MB, against the 164 MB this function already costs.
    arrays_by_group: dict[tuple, tuple] = {}
    for keys, group_frame in frame.groupby(list(GROUP_KEYS), sort=False):
        arrays = (
            group_frame["image_id"].to_numpy(),
            group_frame["severity"].to_numpy(),
            group_frame["score"].to_numpy(dtype=np.float64),
        )
        arrays_by_group[keys] = arrays
        image_metrics = _image_metrics(*arrays)
        # An image with fewer than two surviving severities has no trend to describe.
        # `monotonicity_metrics` says so with `nan` for three of its four statistics -- but
        # `endpoint_increase` comes back a definite `False`, which a plain mean would average
        # in and publish as "this image did not rise". Selecting the measured images once gives
        # all four the same denominator, and `scored_image_count` reports what it is.
        measured = [value for value in image_metrics.values() if value["finite_count"] >= 2]
        spearman_by_group[keys] = {
            image_id: float(value["spearman"])
            for image_id, value in image_metrics.items()
            if value["finite_count"] >= 2 and math.isfinite(value["spearman"])
        }
        group = {
            **dict(zip(GROUP_KEYS, keys)),
            "image_count": len(image_metrics),
            "scored_image_count": len(measured),
            "scored_severity_count": sum(value["finite_count"] for value in image_metrics.values()),
            "total_severity_count": sum(value["total_count"] for value in image_metrics.values()),
            "expected_severity_count": len(image_metrics) * len(EXPECTED_SEVERITIES),
            "median_spearman": _finite_median(value["spearman"] for value in measured),
            "mean_adjacent_monotonicity": _finite_mean(
                value["adjacent_monotonicity"] for value in measured
            ),
            "mean_violation_magnitude": _finite_mean(
                value["violation_magnitude"] for value in measured
            ),
            # The design's "maximum-blur-above-clean rate" (spec:151). It is the rate over the
            # *scored* endpoints: `monotonicity_metrics` drops the severities with no score, so
            # on a group that lost severity 5 this compares severity 4 against severity 0. That
            # is why full coverage gates the ranking rather than decorating it.
            "endpoint_increase_rate": _finite_mean(
                value["endpoint_increase"] for value in measured
            ),
            # Severity 0 is excluded because it is 1.0 by construction for every membership
            # mode -- a dynamic bin at severity 0 *is* its own severity-zero reference. Averaging
            # it in adds a sixth of a point to every bin alike and pulls the pilot's ~0.06
            # dynamic overlaps up to ~0.216, which is a number that looks like a measurement of
            # stability and is mostly a measurement of the definition. The per-severity dict
            # below keeps severity 0, where it belongs as the reference point it is.
            "mean_clean_overlap_from_severity_1": _finite_mean(
                group_frame.loc[group_frame["severity"] > 0, "clean_overlap"]
            ),
            "mean_clean_overlap_by_severity": {
                str(int(severity)): _finite_mean(values)
                for severity, values in group_frame.groupby("severity")["clean_overlap"]
            },
            "median_selected_count_by_severity": {
                str(int(severity)): float(values.median())
                for severity, values in group_frame.groupby("severity")["selected_count"]
            },
        }
        # Whether that scalar is a measurement at all. See `DEFINITIONAL_OVERLAP_MODES`.
        group["clean_overlap_is_definitional"] = (
            group["membership_mode"] in DEFINITIONAL_OVERLAP_MODES
        )
        group["full_coverage"] = is_full_coverage(group)
        groups.append(group)

    by_key = {tuple(group[key] for key in GROUP_KEYS): group for group in groups}
    comparisons = []
    without_control = 0
    for key, group in by_key.items():
        signal, membership, confidence_bin, aggregation, scope, padding = key
        if signal != "persistence":
            continue
        control_key = (
            "confidence", membership, confidence_bin, aggregation, CONFIDENCE_SCOPE, padding
        )
        control = by_key.get(control_key)
        if control is None:
            # The all-300-query benchmark is scored for persistence alone (spec:131), so this
            # is expected rather than exceptional -- but it is counted, because a bin that lost
            # its control by accident would otherwise vanish from the comparison table in
            # exactly the same silence.
            without_control += 1
            continue
        persistence_spearman = spearman_by_group[key]
        control_spearman = spearman_by_group[control_key]
        paired = _paired(persistence_spearman, control_spearman)
        comparisons.append({
            "membership_mode": membership,
            "confidence_bin": confidence_bin,
            "padding_mode": padding,
            "aggregation": aggregation,
            "score_scope": scope,
            "persistence_median_spearman": group["median_spearman"],
            "confidence_median_spearman": control["median_spearman"],
            "persistence_minus_confidence_spearman": _difference(
                group["median_spearman"], control["median_spearman"]
            ),
            **_outcome_rates(
                "persistence_image", persistence_spearman, control_spearman, paired
            ),
            "paired_image_count": len(paired),
            "persistence_image_count": len(persistence_spearman),
            "confidence_image_count": len(control_spearman),
        })
    comparisons.sort(key=lambda entry: (
        entry["membership_mode"], entry["confidence_bin"], entry["padding_mode"],
        entry["aggregation"], entry["score_scope"],
    ))

    sensitivity = []
    for key, group in by_key.items():
        signal, membership, confidence_bin, aggregation, scope, padding = key
        if padding != UNFILTERED_PADDING_MODE:
            continue
        filtered_key = (
            signal, membership, confidence_bin, aggregation, scope, FILTERED_PADDING_MODE
        )
        filtered = by_key.get(filtered_key)
        if filtered is None:
            continue
        unfiltered_spearman = spearman_by_group[key]
        filtered_spearman = spearman_by_group[filtered_key]
        paired = _paired(unfiltered_spearman, filtered_spearman)
        score_changed = _score_changed_images(
            arrays_by_group[key], arrays_by_group[filtered_key]
        )
        changed_and_paired = [image for image in paired if image in score_changed]
        sensitivity.append({
            "signal": signal,
            "membership_mode": membership,
            "confidence_bin": confidence_bin,
            "aggregation": aggregation,
            "score_scope": scope,
            "filtered_median_spearman": filtered["median_spearman"],
            "unfiltered_median_spearman": group["median_spearman"],
            "unfiltered_minus_filtered_spearman": _difference(
                group["median_spearman"], filtered["median_spearman"]
            ),
            **_outcome_rates(
                "unfiltered_image", unfiltered_spearman, filtered_spearman, paired
            ),
            "paired_image_count": len(paired),
            # How many images the mask changed the *score* of -- a lower bound on the images
            # it reached, and named for what it measures. See `_score_changed_images`. Without
            # some such count the median difference cannot be told apart from a null result:
            # an image with no padded tail selects the identical queries either way, and on
            # the pilot that is 184 images out of 250.
            "score_changed_image_count": len(score_changed),
            # ... and the outcomes restricted to those images. The unrestricted rate counts
            # every unchanged image as a non-win, so on the pilot's dynamic bottom bin it
            # reads 0.168 over all 250 while the unfiltered run out-trends the filtered one on
            # 42 of the 64 whose score it changed. Both are true; only the pair is not
            # misleading.
            **_outcome_rates(
                "score_changed_image", unfiltered_spearman, filtered_spearman,
                changed_and_paired,
            ),
            "score_changed_and_paired_image_count": len(changed_and_paired),
        })
    sensitivity.sort(key=lambda entry: (
        entry["signal"], entry["membership_mode"], entry["confidence_bin"],
        entry["aggregation"], entry["score_scope"],
    ))

    # Each dynamic selection against its own frozen twin, paired per image. Same bin, same
    # summary, same scope, same padding rule -- the only thing that differs is whether the ten
    # bins were rebuilt at this severity or reused from severity zero, which is what makes the
    # comparison a measurement of query movement and nothing else. See
    # `MEMBERSHIP_COMPARISON_PAIR` for why this is a comparison and not the combination
    # spec:230 forbids.
    dynamic_mode, frozen_mode = MEMBERSHIP_COMPARISON_PAIR
    membership_comparisons = []
    for key, group in by_key.items():
        signal, membership, confidence_bin, aggregation, scope, padding = key
        if membership != dynamic_mode:
            continue
        frozen_key = (signal, frozen_mode, confidence_bin, aggregation, scope, padding)
        frozen = by_key.get(frozen_key)
        if frozen is None:
            continue
        dynamic_spearman = spearman_by_group[key]
        frozen_spearman = spearman_by_group[frozen_key]
        paired = _paired(dynamic_spearman, frozen_spearman)
        membership_comparisons.append({
            "signal": signal,
            "confidence_bin": confidence_bin,
            "aggregation": aggregation,
            "score_scope": scope,
            "padding_mode": padding,
            "dynamic_median_spearman": group["median_spearman"],
            "frozen_median_spearman": frozen["median_spearman"],
            "dynamic_minus_frozen_spearman": _difference(
                group["median_spearman"], frozen["median_spearman"]
            ),
            **_outcome_rates("dynamic_image", dynamic_spearman, frozen_spearman, paired),
            "paired_image_count": len(paired),
            # The movement the comparison is about, carried on the row rather than left to be
            # joined from `groups`: a reader of this table needs to know whether the dynamic
            # bin moved at all before a win or a loss over the frozen one means anything.
            "dynamic_mean_clean_overlap_from_severity_1":
                group["mean_clean_overlap_from_severity_1"],
        })
    membership_comparisons.sort(key=lambda entry: (
        entry["signal"], entry["confidence_bin"], entry["aggregation"],
        entry["score_scope"], entry["padding_mode"],
    ))

    ranked = rank_deployable_groups(groups)
    benchmark_comparisons = []
    for candidate in ranked:
        candidate_key = tuple(candidate[name] for name in GROUP_KEYS)
        candidate_spearman = spearman_by_group[candidate_key]
        for benchmark_key in _benchmark_keys(by_key):
            benchmark = by_key[benchmark_key]
            benchmark_spearman = spearman_by_group[benchmark_key]
            paired = _paired(candidate_spearman, benchmark_spearman)
            benchmark_comparisons.append({
                "membership_mode": candidate["membership_mode"],
                "confidence_bin": candidate["confidence_bin"],
                "aggregation": candidate["aggregation"],
                "score_scope": candidate["score_scope"],
                "padding_mode": candidate["padding_mode"],
                "benchmark_aggregation": benchmark["aggregation"],
                "candidate_median_spearman": candidate["median_spearman"],
                "benchmark_median_spearman": benchmark["median_spearman"],
                "candidate_minus_benchmark_spearman": _difference(
                    candidate["median_spearman"], benchmark["median_spearman"]
                ),
                **_outcome_rates(
                    "candidate_image", candidate_spearman, benchmark_spearman, paired
                ),
                "paired_image_count": len(paired),
            })

    summary = {
        "run_metadata": dict(run_metadata or {}),
        "diagnostics": {
            "row_count": int(len(frame)),
            "image_count": int(frame["image_id"].nunique()),
            "severities": sorted(int(value) for value in frame["severity"].unique()),
            "group_count": len(groups),
            "groups_with_full_coverage": sum(group["full_coverage"] for group in groups),
            "groups_with_unscored_severities": sum(
                group["scored_severity_count"] < group["total_severity_count"]
                for group in groups
            ),
            "deployable_group_count": len(ranked),
            "comparison_count": len(comparisons),
            "persistence_groups_without_confidence_control": without_control,
            "padding_sensitivity_count": len(sensitivity),
            "membership_comparison_count": len(membership_comparisons),
            "benchmark_comparison_count": len(benchmark_comparisons),
        },
        "padding": _padding_rollup(diagnostics),
        "groups": groups,
        "comparisons": comparisons,
        "padding_sensitivity": sensitivity,
        "membership_comparisons": membership_comparisons,
        "benchmark_comparisons": benchmark_comparisons,
        RANKED_GROUPS_KEY: ranked,
    }
    return _jsonable(summary)


DYNAMIC_MEMBERSHIP_MODE = "dynamic"
FROZEN_MEMBERSHIP_MODE = "frozen"

PERSISTENCE_PANEL_LABEL = f"persistence ({PRIMARY_SCORE_SCOPE})"
CONFIDENCE_PANEL_LABEL = "confidence control (1 - confidence)"

FIGURE_SLICE_WITHOUT_MEMBERSHIP = (
    f"{FILTERED_PADDING_MODE} queries, {BENCHMARK_AGGREGATION} scene summary"
)
FIGURE_SLICE = f"{DYNAMIC_MEMBERSHIP_MODE} membership, {FIGURE_SLICE_WITHOUT_MEMBERSHIP}"
FIGURE_SLICE_NOTE = f"{FIGURE_SLICE}, persistence at {PRIMARY_SCORE_SCOPE}"
FIGURE_SLICE_BOTH_MEMBERSHIPS = (
    f"{FIGURE_SLICE_WITHOUT_MEMBERSHIP}, persistence at {PRIMARY_SCORE_SCOPE}"
)
"""What each figure is a slice of, printed on the figure -- in four forms, not one.

A caption is a claim about what is drawn, and a slice clause that is true of one figure is
false on the next. Two of these forms exist because the general one was wrong somewhere:

* the **membership** clause is false on `dynamic_vs_frozen.png`, whose entire subject is both
  memberships side by side. A top panel captioned "dynamic membership" over a pair of dynamic
  and frozen bars says the frozen bars are dynamic;
* the **scope** clause is false on any confidence panel or bar. The confidence control has no
  decoder-layer scope at all (spec:125), and `1 - confidence` values captioned "persistence at
  layer_2" tell the reader something untrue about the numbers in front of them.

All four fix the scene summary at `q90` because that is the only summary the published
all-300-query benchmark exists at (spec:131), so it is the only one in which a figure and the
benchmark row describe the same thing. The winning candidate may well be at another summary --
the easy report lays out all three for the winning bin, and the figures say what they are so a
reader does not assume otherwise.
"""


# --- lookups over a finished summary ----------------------------------------------------------


def _lookup(entries, **match) -> dict | None:
    """The single entry matching every field, or `None`.

    `None` rather than a raise, because "this table has no all-300-query benchmark" is a
    legitimate state -- the brief's own fixture is one -- and the callers turn it into a
    sentence saying so. What is *not* legitimate is two matches, which would mean the key
    being matched on does not identify a row; that raises.
    """
    found = [
        entry for entry in entries
        if all(entry.get(field) == value for field, value in match.items())
    ]
    if len(found) > 1:
        raise ValueError(f"{len(found)} entries match {match}; that key does not identify a row")
    return found[0] if found else None


def _decile_group(summary: dict, membership: str, confidence_bin: str,
                  signal: str = "persistence", scope: str | None = None,
                  aggregation: str = BENCHMARK_AGGREGATION) -> dict | None:
    return _lookup(
        summary["groups"], signal=signal,
        score_scope=PRIMARY_SCORE_SCOPE if scope is None else scope,
        membership_mode=membership, confidence_bin=confidence_bin,
        aggregation=aggregation, padding_mode=FILTERED_PADDING_MODE,
    )


def _benchmark_group(summary: dict) -> dict | None:
    """The published all-300-query row, matched on the producer's own label tuple (spec:137)."""
    signal, membership, confidence_bin, scope, padding = BENCHMARK_SELECTION
    return _lookup(
        summary["groups"], signal=signal, membership_mode=membership,
        confidence_bin=confidence_bin, score_scope=scope, padding_mode=padding,
        aggregation=BENCHMARK_AGGREGATION,
    )


def _group_key(group: dict) -> dict:
    return {key: group[key] for key in GROUP_KEYS if key != "signal"}


# --- formatting ---------------------------------------------------------------------------------


def _signed(value, digits: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return _UNMEASURED
    return f"{float(value):+.{digits}f}"


def _plain(value, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return _UNMEASURED
    return f"{float(value):.{digits}f}"


def _whole(rate, total) -> int | None:
    """A count recovered from a published rate and its denominator.

    Every rate here is a mean of booleans, so it is exactly `k/n` and `round(rate * n)` is
    `k`. Counts are what a reader can check and argue with; a bare rate of 0.488 hides that it
    stands for 122 images against 88 with 40 ties.
    """
    if rate is None or total in (None, 0):
        return None
    return int(round(float(rate) * int(total)))


def _verdict(win: int | None, loss: int | None) -> str:
    """The verb, chosen from the counts and never from a threshold.

    Three outcomes and no fourth. Note what is missing: nothing here says "beat", "significant"
    or "better". A majority of decided images describes this run; the candidate was selected on
    these same images, so the count is not a test of anything and the sentence that follows it
    says so. A comparison that comes out 103 to 112 is *undecided*, not a defeat, and
    "does not out-trend" is the strongest thing this may say about it.
    """
    if win is None or loss is None:
        return "cannot be compared image by image with"
    if win > loss:
        return "out-trends"
    if win < loss:
        return "does not out-trend"
    return "splits evenly with"


def _outcome(entry: dict | None, prefix: str, total_key: str = "paired_image_count") -> tuple:
    """`(verb, sentence)` for one paired comparison, with both denominators in the sentence.

    Both, always. The rate over every paired image counts each tie as a non-win, which pushes
    a candidate that wins 122 and loses 88 below 0.5 and invites "it loses the per-image
    majority"; the rate over the decided images alone hides how much of the run the comparison
    could not separate. On the pilot those two framings put the same candidate on opposite
    sides of 0.5.
    """
    if entry is None:
        return "cannot be compared image by image with", "there is no paired comparison to make"
    total = entry.get(total_key)
    win = _whole(entry.get(f"{prefix}_win_rate"), total)
    tie = _whole(entry.get(f"{prefix}_tie_rate"), total)
    loss = _whole(entry.get(f"{prefix}_loss_rate"), total)
    decided = entry.get(f"{prefix}_decided_image_count")
    decided_rate = entry.get(f"{prefix}_decided_win_rate")
    if not total:
        return _verdict(win, loss), "no image was measured on both sides, so nothing is compared"
    tail = (
        f"{_plain(decided_rate)} over the {decided} it decided"
        if decided else "and the comparison decided none of them"
    )
    return _verdict(win, loss), (
        f"it wins {win}, ties {tie} and loses {loss} of the {total} images both sides "
        f"measured -- a win rate of {_plain(entry.get(f'{prefix}_win_rate'))} over all "
        f"{total}, {tail}"
    )


def _membership_cell(membership: str) -> str:
    """A membership label that carries its own status (spec:230).

    `frozen` prints as a diagnostic wherever it appears in a table, and not only in the table
    named after it. On the pilot the frozen bottom bin publishes the single most impressive
    pair of numbers in the whole report -- a difference of +1.2571 at a 0.916 win rate -- and
    an unlabelled `frozen` in a membership column beside dynamic rows reads as the best
    available method rather than as a measurement that needs a paired clean image.
    """
    if membership == FROZEN_MEMBERSHIP_MODE:
        return f"`{membership}` (diagnostic)"
    return f"`{membership}`"


def _slice_caption(scoped: bool, summaries: str) -> str:
    """What a table is a slice of, said above the table.

    The figures each carry their slice on their own title, deliberately, because a PNG gets
    pasted into a write-up on its own. A markdown table has exactly the same problem and had
    none of the protection: three tables here were silently cut to one scene summary while the
    headline sentence directly above one of them quoted a different one, so the same selection
    appeared twice on one page with two different numbers and nothing said why.

    Two remedies, applied together. The tables now carry every scene summary rather than one,
    so the row the headline quotes is in the table a reader is invited to check it against;
    and this caption says which slice is still fixed.
    """
    scope = (
        f"persistence at `{PRIMARY_SCORE_SCOPE}` against its matched control"
        if scoped else f"persistence at `{PRIMARY_SCORE_SCOPE}`"
    )
    return (
        f"Slice: {scope}, with the padding union removed, at {summaries} scene summary. "
        "Rows that differ only in the summary are views of one selection on the same images, "
        "not independent measurements of it."
    )


def _table(headers: list[str], body: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *["| " + " | ".join(cells) + " |" for cells in body],
    ]


# --- atomic output --------------------------------------------------------------------------------


def _atomic_text(path: Path, text: str) -> None:
    """Write through a temporary and rename, so a crash never leaves a readable half-file.

    A truncated `summary.json` is a parse error, which is loud. A truncated `easy-report.md`
    is a shorter report that stops mid-sentence and looks finished, which is not.
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _save_figure(figure, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    figure.tight_layout()
    figure.savefig(temporary, format="png", dpi=160)
    plt.close(figure)
    os.replace(temporary, path)


def _absent(axis, message: str) -> None:
    """Say a panel had nothing to draw, rather than drawing an empty one.

    An empty axis with a y label and a zero line is indistinguishable from a measurement that
    came out flat. This prints the reason across the panel and removes the ticks, so the only
    thing a reader can conclude is the only thing that is true.
    """
    axis.text(
        0.5, 0.5, message, ha="center", va="center", fontsize=9, color="0.3",
        transform=axis.transAxes, wrap=True,
    )
    axis.set_xticks([])
    axis.set_yticks([])


# --- the four figures -------------------------------------------------------------------------------


def _heatmap_figure(summary: dict):
    """Spec:167 -- median Spearman by confidence bin and signal.

    Ten columns always, in `DECILE_NAMES` order, whatever the table holds; a bin with no row
    is a grey cell reading `n/a`. Drawing only the bins that exist, sorted however the group-by
    returned them, would publish a complete-looking ten-bin measurement over a subset -- and
    alphabetical order happens to agree with decile order for these names, which is exactly
    the kind of coincidence that hides the bug until the labels change.

    The colour scale is fixed to [-1, +1] and centred on zero rather than fitted to the data,
    so a run where every bin is weakly negative cannot be coloured to look like a spread of
    strong results, and two runs' heatmaps can be laid side by side.
    """
    signals = (
        ("persistence", PRIMARY_SCORE_SCOPE, PERSISTENCE_PANEL_LABEL),
        ("confidence", CONFIDENCE_SCOPE, CONFIDENCE_PANEL_LABEL),
    )
    matrix = []
    for signal, scope, _ in signals:
        line = []
        for name in DECILE_NAMES:
            found = _decile_group(summary, DYNAMIC_MEMBERSHIP_MODE, name, signal, scope)
            value = None if found is None else found["median_spearman"]
            line.append(math.nan if value is None else float(value))
        matrix.append(line)

    figure, axis = plt.subplots(figsize=(10, 3.4))
    data = np.ma.masked_invalid(np.asarray(matrix, dtype=np.float64))
    # `RdBu_r` and not `coolwarm`: coolwarm's midpoint is a light grey indistinguishable from
    # the colour a masked cell would be drawn in, so a bin scoring 0.00 and a bin with no row
    # at all would look the same. `RdBu_r` is near-white at zero.
    colours = matplotlib.colormaps["RdBu_r"].with_extremes(bad="0.72")
    drawn = axis.imshow(data, cmap=colours, vmin=-1.0, vmax=1.0, aspect="auto")
    axis.set_xticks(range(len(DECILE_NAMES)), list(DECILE_NAMES), rotation=45, ha="right")
    axis.set_yticks(range(len(signals)), [label for _, _, label in signals])
    for row_index, line in enumerate(matrix):
        for column_index, value in enumerate(line):
            axis.text(
                column_index, row_index,
                "n/a" if not math.isfinite(value) else f"{value:+.2f}",
                ha="center", va="center", fontsize=7,
                color="white" if not math.isfinite(value) else "black",
            )
    figure.colorbar(drawn, ax=axis, label="median per-image Spearman")
    # `FIGURE_SLICE` and not `FIGURE_SLICE_NOTE`: the second row of this figure is the
    # confidence control, which has no decoder-layer scope (spec:125). Each row carries its own
    # scope in its label, so the title states only what is true of both.
    axis.set_title(
        f"Median per-image Spearman by confidence bin and signal\n{FIGURE_SLICE}"
        "; grey `n/a` = no row in this table",
        fontsize=8,
    )
    return figure


def _clean_relative_curves(frame: pd.DataFrame, signal: str, scope: str) -> dict:
    """Median clean-relative score per severity, per decile bin.

    Clean-relative means each image's own severity-0 score subtracted from its own curve, so
    the median across images describes the *rise* rather than the level. Without it the median
    is dominated by how far apart the images' baselines happen to be, and a bin whose images
    all rise steeply from different starting points looks flat.
    """
    subset = frame[
        (frame["signal"] == signal)
        & (frame["score_scope"] == scope)
        & (frame["membership_mode"] == DYNAMIC_MEMBERSHIP_MODE)
        & (frame["padding_mode"] == FILTERED_PADDING_MODE)
        & (frame["aggregation"] == BENCHMARK_AGGREGATION)
        & (frame["confidence_bin"].isin(list(DECILE_NAMES)))
    ]
    if subset.empty:
        return {}
    clean = subset.loc[subset["severity"] == 0, ["image_id", "confidence_bin", "score"]]
    merged = subset.merge(
        clean.rename(columns={"score": "clean_score"}),
        on=["image_id", "confidence_bin"], how="left",
    )
    merged["relative"] = merged["score"] - merged["clean_score"]
    grouped = merged.groupby(["confidence_bin", "severity"])["relative"].median()
    return {
        name: grouped.loc[name].sort_index()
        for name in DECILE_NAMES
        if name in set(grouped.index.get_level_values(0))
    }


def _blur_curve_figure(frame: pd.DataFrame):
    """Spec:168 -- clean-relative severity curves for persistence and confidence.

    **Two panels, never one axis.** A persistence score is a distance between fingerprints and
    the control is `1 - confidence`; they share no unit, and spec:157 forbids comparing their
    magnitudes. One axis carrying both would invite exactly that comparison, and the reader
    would have no way to know it was meaningless. What *is* comparable between the panels is
    the shape of the curves, which is what the design's trend metrics measure.

    Within a panel the ten curves are the same unit, but they are still not a ranking: a bin
    that rises further in absolute distance is not thereby a better blur signal, because the
    per-image Spearman the design ranks on is invariant under rescaling. The heatmap is the
    ranking; this figure is where a reader sees whether a rise is smooth or collapses.
    """
    panels = (
        ("persistence", PRIMARY_SCORE_SCOPE,
         f"median clean-relative persistence distance ({PRIMARY_SCORE_SCOPE})"),
        ("confidence", CONFIDENCE_SCOPE, "median clean-relative (1 - confidence)"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    colours = matplotlib.colormaps["viridis"](np.linspace(0.0, 0.95, len(DECILE_NAMES)))
    for axis, (signal, scope, ylabel) in zip(axes, panels):
        axis.set_ylabel(ylabel, fontsize=8)
        axis.set_xlabel("Gaussian blur severity")
        curves = _clean_relative_curves(frame, signal, scope)
        if not curves:
            _absent(axis, f"no {DYNAMIC_MEMBERSHIP_MODE} {FILTERED_PADDING_MODE} "
                          f"{BENCHMARK_AGGREGATION} {signal} rows in this results table")
            continue
        for colour, name in zip(colours, DECILE_NAMES):
            series = curves.get(name)
            if series is None:
                continue
            axis.plot(
                [int(value) for value in series.index], series.to_numpy(dtype=np.float64),
                marker="o", markersize=3, color=colour, label=name,
            )
        # Counted, not listed. Naming the absent bins is unreadable in exactly the case the
        # message exists for: on a three-bin table the list is seven names long, and it
        # overflows the panel it is meant to caption -- clipped on the left, off the canvas on
        # the right. The count fits, and `summary.json` holds which.
        missing = [name for name in DECILE_NAMES if name not in curves]
        axis.set_title(
            f"{signal}\n"
            + (FIGURE_SLICE_NOTE if signal == "persistence" else FIGURE_SLICE)
            + (
                f"\n{len(missing)} of {len(DECILE_NAMES)} bins have no row here -- "
                "see summary.json"
                if missing else ""
            ),
            fontsize=7,
        )
        axis.axhline(0.0, color="0.6", linewidth=0.8, linestyle=":")
        axis.legend(fontsize=6, ncol=2)
    figure.suptitle(
        "Clean-relative severity curves -- two panels because the two signals share no unit; "
        "their magnitudes must not be compared",
        fontsize=8,
    )
    return figure


def _dynamic_frozen_figure(summary: dict):
    """Spec:169 -- query movement versus feature movement, and never one combined score.

    The two bars per bin are drawn side by side and labelled, never stacked, averaged or
    differenced. Spec:230: dynamic and frozen answer different questions and must not be
    combined into one score. The pilot makes the cost concrete -- the bottom bin is -0.029
    dynamic and +0.486 frozen, so any single number covering both describes neither, and the
    frozen one is not a policy anyone can run because a naturally corrupted image has no
    paired clean version to freeze a membership from.

    The lower panel is the *reason* the two differ, and it carries dynamic bins only. Frozen
    membership is severity zero's by construction, so its overlap is 1.0 arithmetically; a bar
    of 1.0 printed beside a dynamic 0.06 would read as stability rather than as a definition.
    The dashed line is the overlap two unrelated decile memberships would show -- see
    `RANDOM_BIN_OVERLAP` -- without which nine bins clustered at 0.06 look reassuring.
    """
    figure, (trend, overlap) = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True)
    positions = np.arange(len(DECILE_NAMES), dtype=np.float64)
    for offset, mode in ((-0.2, DYNAMIC_MEMBERSHIP_MODE), (0.2, FROZEN_MEMBERSHIP_MODE)):
        heights = []
        for name in DECILE_NAMES:
            found = _decile_group(summary, mode, name)
            value = None if found is None else found["median_spearman"]
            heights.append(math.nan if value is None else float(value))
        trend.bar(positions + offset, heights, width=0.4, label=mode)
    trend.axhline(0.0, color="0.4", linewidth=0.8)
    trend.set_ylabel("median per-image Spearman", fontsize=8)
    trend.legend(fontsize=8)
    trend.set_title(
        "Persistence trend by confidence bin under two memberships -- two answers, never one"
        f"\n{FIGURE_SLICE_BOTH_MEMBERSHIPS}"
        "\nFrozen is a diagnostic: it needs a paired clean image, which a naturally corrupted "
        "one does not have.",
        fontsize=8,
    )

    overlaps = []
    for name in DECILE_NAMES:
        found = _decile_group(summary, DYNAMIC_MEMBERSHIP_MODE, name)
        value = None if found is None else found["mean_clean_overlap_from_severity_1"]
        overlaps.append(math.nan if value is None else float(value))
    overlap.bar(positions, overlaps, width=0.6, color="0.45", label=DYNAMIC_MEMBERSHIP_MODE)
    overlap.axhline(
        RANDOM_BIN_OVERLAP, color="crimson", linestyle="--", linewidth=1.0,
        label=f"unrelated memberships ({RANDOM_BIN_OVERLAP:.4f})",
    )
    overlap.set_xticks(positions, list(DECILE_NAMES), rotation=45, ha="right")
    overlap.set_ylabel("mean Jaccard vs severity 0\n(severities 1-5)", fontsize=8)
    overlap.legend(fontsize=7)
    overlap.set_title(
        "How much of each bin's membership survives blur. Dynamic bins only -- a frozen or "
        "shared bin is 1.000 by construction, which is arithmetic and not stability.",
        fontsize=8,
    )
    return figure


def _padding_sensitivity_figure(summary: dict):
    """Spec:170 -- the lowest bin with the padding union removed and kept.

    Only the lowest bin, because spec:69 scopes the control to it: "A sensitivity control
    repeats only the lowest-confidence-bin analysis without filtering". Widening it would
    publish nine padding measurements the design never asked for.

    The count annotated over each pair is `score_changed_image_count`, and the axis label says
    what it is: a **lower bound** on the images the mask reached. The mask reaches an image
    whenever it changes that image's selection, and a changed selection can still produce a
    bit-identical scene score -- a summary is a many-to-one map. On the pilot the same
    selection pair reports 66, 64 and 65 under the three scene summaries, and a set of images
    a mask reached cannot depend on the summary applied afterwards.
    """
    entries = sorted(
        (
            entry for entry in summary["padding_sensitivity"]
            if entry["confidence_bin"] == SENSITIVITY_BIN
            and entry["aggregation"] == BENCHMARK_AGGREGATION
            and entry["score_scope"] in (PRIMARY_SCORE_SCOPE, CONFIDENCE_SCOPE)
        ),
        key=lambda entry: (entry["membership_mode"], entry["signal"]),
    )
    figure, axis = plt.subplots(figsize=(9, 4.6))
    axis.set_title(
        f"Padding sensitivity, {SENSITIVITY_BIN} only, {BENCHMARK_AGGREGATION} scene summary."
        f"\nThe persistence bars are at {PRIMARY_SCORE_SCOPE}; the confidence control has no "
        "decoder-layer scope."
        "\nUnfiltered keeps the repeated decoder placeholders the primary analysis removes; "
        "frozen is a diagnostic and not a deployable result.",
        fontsize=8,
    )
    axis.set_ylabel("median per-image Spearman", fontsize=8)
    if not entries:
        _absent(
            axis,
            f"no unfiltered {SENSITIVITY_BIN} rows in this results table, so the padding "
            "control could not be drawn",
        )
        return figure
    positions = np.arange(len(entries), dtype=np.float64)
    for offset, key, label in (
        (-0.2, "filtered_median_spearman", FILTERED_PADDING_MODE),
        (0.2, "unfiltered_median_spearman", UNFILTERED_PADDING_MODE),
    ):
        heights = [
            math.nan if entry[key] is None else float(entry[key]) for entry in entries
        ]
        axis.bar(positions + offset, heights, width=0.4, label=label)
    for position, entry in zip(positions, entries):
        # Annotated against the top of the axis rather than against zero: a pair of positive
        # bars would otherwise have the count printed across them.
        axis.text(
            position, 0.98, f"score changed on\n{entry['score_changed_image_count']} images",
            transform=axis.get_xaxis_transform(), ha="center", va="top",
            fontsize=6, color="0.25",
        )
    axis.axhline(0.0, color="0.4", linewidth=0.8)
    # Headroom for the annotations, which are anchored to the top of the axis.
    bottom, top = axis.get_ylim()
    axis.set_ylim(bottom, top + 0.22 * (top - bottom))
    # Spec:230 on a figure that gets read on its own. The frozen bars are the tallest on this
    # axis -- +0.4857 and +0.5429 against a dynamic persistence bar at -0.0286 -- so a reader
    # skimming the figures meets the most impressive result in the set with nothing saying it
    # needs a paired clean image and cannot be run on one.
    axis.set_xticks(
        positions,
        [
            f"{entry['membership_mode']}"
            + (" (diagnostic)" if entry["membership_mode"] == FROZEN_MEMBERSHIP_MODE else "")
            + f"\n{entry['signal']}"
            for entry in entries
        ],
        fontsize=8,
    )
    axis.legend(fontsize=8)
    axis.set_xlabel(
        "The annotated count is a LOWER BOUND on the images the padding mask reached: it "
        "counts images whose scene score moved, and that varies with the scene summary.",
        fontsize=6,
    )
    return figure


# --- the plain-language report ------------------------------------------------------------------


def _matches(entries, **match) -> list[dict]:
    return [
        entry for entry in entries
        if all(entry.get(field) == value for field, value in match.items())
    ]


def _preamble(summary: dict) -> list[str]:
    metadata = summary.get("run_metadata") or {}
    diagnostics = summary.get("diagnostics") or {}
    severities = diagnostics.get("severities") or []
    return [
        f"Recomputed from saved artifacts over {diagnostics.get('image_count', 0)} images at "
        f"{len(severities)} blur levels in the `{metadata.get('source_partition', 'unknown')}` "
        "partition. No detector was run and no nearest-neighbour search was performed: every "
        "number below comes from the cached features and the saved query distances.",
        "",
        f"Provenance: feature cache `{metadata.get('feature_cache_id', 'unknown')}`, kNN result "
        f"`{metadata.get('source_result_id', 'unknown')}`, clean bank "
        f"`{metadata.get('bank_id', 'unknown')}`, k={metadata.get('k', 'unknown')}, "
        f"normalization `{metadata.get('normalization', 'unknown')}`. "
        f"{diagnostics.get('row_count', 0)} scored rows in "
        f"{diagnostics.get('group_count', 0)} groups, "
        f"{diagnostics.get('deployable_group_count', 0)} of which cleared the deployable gate.",
        "",
    ]


def _winner_sentence(ranked: list[dict], winner: dict | None, group_count: int) -> str:
    if winner is None:
        return (
            f"No candidate qualified. To be ranked, a selection has to be persistence at "
            f"`{PRIMARY_SCORE_SCOPE}`, built with the padding union removed, on a membership "
            f"that can be rebuilt on a single image, and scored at every image and every one "
            f"of the {len(EXPECTED_SEVERITIES)} severities. Of the {group_count} groups "
            "measured, none met all four, so nothing in this report recommends a confidence "
            "range."
        )
    return (
        f"`{winner['confidence_bin']}`, using `{winner['membership_mode']}` membership, the "
        f"`{winner['aggregation']}` scene summary and persistence at `{winner['score_scope']}`, "
        f"with the padded decoder queries removed. Its median per-image Spearman against blur "
        f"severity is {_signed(winner['median_spearman'])}, its adjacent non-decrease rate "
        f"{_plain(winner['mean_adjacent_monotonicity'])}, its maximum-blur-above-clean rate "
        f"{_plain(winner['endpoint_increase_rate'])}, and it scored "
        f"{winner['scored_severity_count']} of the {winner['expected_severity_count']} "
        f"image-severity pairs it swept. It came first of {len(ranked)} candidates that "
        "cleared the deployable gate -- which also means it was chosen on the same images "
        "every number in this report is measured over, so what follows is a selection and not "
        "a hypothesis test of anything."
    )


def _confidence_sentence(summary: dict, winner: dict | None) -> str:
    if winner is None:
        return (
            "There is no candidate to compare, so this question is unanswered on this run. "
            "The bin-by-bin comparison below still reports every matched pair that was scored."
        )
    comparison = _lookup(summary["comparisons"], **_group_key(winner))
    if comparison is None:
        return (
            "This selection was scored for persistence only, so it has no matched confidence "
            "control to be compared against (the all-300-query benchmark is scored this way "
            "by design). Nothing here answers the question for it."
        )
    verb, sentence = _outcome(comparison, "persistence_image")
    return (
        f"On the same selected queries and the same scene summary, persistence {verb} its "
        f"confidence control: median Spearman {_signed(comparison['persistence_median_spearman'])} "
        f"against {_signed(comparison['confidence_median_spearman'])}, a difference of "
        f"{_signed(comparison['persistence_minus_confidence_spearman'])}. Image by image, "
        f"{sentence}."
    )


def _grid_steps(difference) -> str:
    """A difference in medians expressed in units of the metric's own resolution.

    Written this way because a bare "+0.0286" reads as a small margin, and it is not a small
    margin -- it is the *smallest one this metric can express*. A margin of one grid step and
    a margin of zero are adjacent states of the same statistic, and a reader who is not told
    the grid size cannot tell which of the two they are looking at.
    """
    if difference is None or not math.isfinite(float(difference)):
        return _UNMEASURED
    steps = abs(float(difference)) * SPEARMAN_STEP_DENOMINATOR
    grid = f"1/{SPEARMAN_STEP_DENOMINATOR} = {1 / SPEARMAN_STEP_DENOMINATOR:.4f} grid a median "
    if steps < 0.5:
        return (
            "exactly zero at the resolution of the primary metric, which is why the paired "
            "comparison beside it is the informative one here"
        )
    if steps < 1.5:
        return (
            f"one step of the {grid}over an even number of images can land on, which is the "
            "smallest difference this metric can express"
        )
    return f"{steps:.1f} steps of the {grid}over an even number of images can land on"


def _benchmark_sentence(summary: dict, winner: dict | None) -> str:
    if winner is None:
        return (
            "There is no candidate to compare against the benchmark on this run."
        )
    entries = sorted(
        _matches(summary["benchmark_comparisons"], **_group_key(winner)),
        key=lambda entry: entry["benchmark_aggregation"],
    )
    if not entries:
        return (
            "This results table carries no all-300-query benchmark row, so the comparison the "
            "design asks for could not be made from it."
        )
    entry = entries[0]
    verb, sentence = _outcome(entry, "candidate_image")
    return (
        f"The existing all-query benchmark reproduces from these artifacts at "
        f"{_signed(entry['benchmark_median_spearman'])} "
        f"(`{entry['benchmark_aggregation']}`, `{entry['score_scope']}`, all queries kept). "
        f"The candidate's median is {_signed(entry['candidate_median_spearman'])}, a difference "
        f"of {_signed(entry['candidate_minus_benchmark_spearman'])} -- "
        f"{_grid_steps(entry['candidate_minus_benchmark_spearman'])}. Paired image by image it "
        f"{verb} the benchmark: {sentence}."
    )


def _freezing_reading(dynamic_median, frozen_median, paired: dict | None) -> str:
    """What the frozen twin says about bin movement -- from the paired outcome, not two medians.

    Freezing the membership at severity zero removes the query movement and leaves the
    fingerprint motion, so the comparison between a bin and its frozen twin is the design's only
    direct handle on spec:172's fourth question. It used to be answered here by subtracting two
    marginal medians, and that is the one comparison this plan has already proved blind at this
    resolution: the leading candidate's median difference against the all-query benchmark is
    exactly 0.000 while the same pair runs 122 images to 92. A gap of one `1/35` step and a gap
    of nothing are adjacent states of a 36-valued statistic, so either verdict was reachable
    from data that could not distinguish them.

    `paired` is the `membership_comparisons` row for this selection. With it, the verdict comes
    from the win and loss counts through `_verdict`, exactly as every other comparison in this
    report does, and both denominators travel with it. Without it -- a table that never scored
    the frozen twin -- the sentence falls back to saying what two medians can and cannot
    support, which is what this function said before the family existed.
    """
    if dynamic_median is None or frozen_median is None:
        return "one of the two was not measured, so the comparison cannot be made"
    gap = float(frozen_median) - float(dynamic_median)
    if paired is None:
        unpaired = (
            "these are two medians and not a paired comparison, and no frozen twin was scored "
            "for this selection"
        )
        if abs(gap) * SPEARMAN_STEP_DENOMINATOR < 1.5:
            return (
                "a gap the primary metric cannot resolve, so freezing the membership does not "
                f"destroy the trend. That is as far as it goes: {unpaired}, so this does not "
                "establish that the movement costs nothing"
            )
        return (
            ("higher" if gap > 0 else "lower")
            + f" with the membership held still -- though {unpaired}"
        )
    total = paired["paired_image_count"]
    win = _whole(paired.get("dynamic_image_win_rate"), total)
    loss = _whole(paired.get("dynamic_image_loss_rate"), total)
    _, sentence = _outcome(paired, "dynamic_image")
    if win is None or loss is None:
        reading = "the two could not be compared image by image"
    elif win > loss:
        reading = (
            "rebuilding the bins at every severity does better than freezing them image by "
            "image, so on these images the movement is not costing this selection its trend"
        )
    elif win < loss:
        reading = (
            "freezing the membership does better image by image, so on these images the "
            "movement of queries between bins is costing this selection some of its trend"
        )
    else:
        reading = (
            "the paired comparison splits evenly, so it separates the two no better than the "
            "medians do"
        )
    return (
        f"a difference of {_signed(_difference(dynamic_median, frozen_median))}, which the "
        f"primary metric resolves to within {1 / SPEARMAN_STEP_DENOMINATOR:.4f} at best. Paired "
        f"image by image the same selection against its own frozen twin: {sentence} -- "
        f"{reading}"
    )


def _movement_sentence(summary: dict, winner: dict | None) -> str:
    if winner is None:
        return "No candidate was selected, so there is no result for either to explain."
    parts = ["Neither is shown to."]
    overlap = winner.get("mean_clean_overlap_from_severity_1")
    if winner.get("clean_overlap_is_definitional"):
        parts.append(
            f"`{winner['confidence_bin']}` under `{winner['membership_mode']}` membership is "
            "the same query set at every severity by construction, so it has no bin movement "
            "at all and none can be explaining it."
        )
    elif overlap is None:
        parts.append("Membership stability was not measured for this selection.")
    else:
        settled = (
            "so its queries are almost entirely reselected at every blur level: whatever the "
            "trend is, it belongs to the confidence range and not to any particular queries"
            if float(overlap) <= 2 * RANDOM_BIN_OVERLAP else
            "so a substantial part of its membership survives blur, and the trend is partly a "
            "property of the queries themselves"
        )
        parts.append(
            f"`{winner['confidence_bin']}` keeps {_plain(overlap, 4)} of its severity-zero "
            f"membership from severity 1 onward, against {RANDOM_BIN_OVERLAP:.4f} for two "
            f"unrelated memberships -- {settled}."
        )
    twin = _lookup(
        summary["groups"], signal="persistence", score_scope=winner["score_scope"],
        membership_mode=FROZEN_MEMBERSHIP_MODE, confidence_bin=winner["confidence_bin"],
        aggregation=winner["aggregation"], padding_mode=winner["padding_mode"],
    )
    if twin is not None:
        against_twin = _lookup(
            summary["membership_comparisons"], signal="persistence",
            confidence_bin=winner["confidence_bin"], aggregation=winner["aggregation"],
            score_scope=winner["score_scope"], padding_mode=winner["padding_mode"],
        )
        parts.append(
            f"Holding that bin's membership fixed at severity zero scores "
            f"{_signed(twin['median_spearman'])} instead of "
            f"{_signed(winner['median_spearman'])} -- "
            f"{_freezing_reading(winner['median_spearman'], twin['median_spearman'], against_twin)}. "
            "That frozen number is a diagnostic and not an alternative method, and the two are "
            "never combined into one score."
        )
    sensitivity = _lookup(
        summary["padding_sensitivity"], signal="persistence",
        membership_mode=winner["membership_mode"], confidence_bin=winner["confidence_bin"],
        aggregation=winner["aggregation"], score_scope=winner["score_scope"],
    )
    if sensitivity is not None:
        parts.append(
            f"Keeping the padded queries instead of removing them moves this selection from "
            f"{_signed(sensitivity['filtered_median_spearman'])} to "
            f"{_signed(sensitivity['unfiltered_median_spearman'])}, and the scene score moved "
            f"on {sensitivity['score_changed_image_count']} images."
        )
    else:
        padding = summary.get("padding") or {}
        parts.append(
            f"The padding control is scoped to `{SENSITIVITY_BIN}` alone, so it does not test "
            f"`{winner['confidence_bin']}` directly."
        )
        every_query = _lookup(
            summary["padding_sensitivity"], signal="persistence",
            membership_mode=ALL_VALID_BENCHMARK[0], confidence_bin=ALL_VALID_BENCHMARK[1],
            aggregation=BENCHMARK_AGGREGATION, score_scope=PRIMARY_SCORE_SCOPE,
        )
        if every_query is not None:
            parts.append(
                f"Where padding can be measured -- the every-query selection -- removing the "
                f"repeated decoder placeholders moves the trend from "
                f"{_signed(every_query['unfiltered_median_spearman'])} to "
                f"{_signed(every_query['filtered_median_spearman'])}, so on this run the "
                f"placeholders were adding to the old benchmark rather than to this bin. "
                f"{padding.get('images_with_padding', 0)} of "
                f"{padding.get('image_count', 0)} images carried any padding at all."
            )
    return " ".join(parts)


def _short_answer(summary: dict, ranked: list[dict], winner: dict | None) -> list[str]:
    group_count = (summary.get("diagnostics") or {}).get("group_count", 0)
    answers = (
        _winner_sentence(ranked, winner, group_count),
        _confidence_sentence(summary, winner),
        _benchmark_sentence(summary, winner),
        _movement_sentence(summary, winner),
    )
    lines = ["## Short answer", ""]
    for question, answer in zip(SPEC_172_QUESTIONS, answers):
        lines.extend([f"**{question}?** {answer}", ""])
    return lines


def _bin_versus_benchmark(summary: dict, groups: list[dict]) -> list[str]:
    """The winning bin against the published benchmark, one row per scene summary.

    This table is where the primary metric's resolution becomes visible instead of being
    described. A summary whose median difference is exactly 0.000 can still take the paired
    comparison decisively -- on the pilot, `q90` is a dead heat on the median and 122 to 92 on
    the images -- and no single-number presentation can show that. Publishing the three rows
    together is also the honest place to say that they are three views of one bin: they share
    the selections and the images, so the agreement between them is arithmetic.
    """
    body = []
    for group in sorted(groups, key=lambda group: group["aggregation"]):
        for entry in sorted(
            _matches(summary["benchmark_comparisons"], **_group_key(group)),
            key=lambda entry: entry["benchmark_aggregation"],
        ):
            total = entry["paired_image_count"]
            body.append([
                f"`{group['aggregation']}`",
                _signed(entry["candidate_median_spearman"]),
                _signed(entry["benchmark_median_spearman"]),
                _signed(entry["candidate_minus_benchmark_spearman"]),
                f"{_whole(entry['candidate_image_win_rate'], total)}/"
                f"{_whole(entry['candidate_image_tie_rate'], total)}/"
                f"{_whole(entry['candidate_image_loss_rate'], total)}",
                _plain(entry["candidate_image_win_rate"]),
                f"{_plain(entry['candidate_image_decided_win_rate'])} "
                f"({entry['candidate_image_decided_image_count']})",
            ])
    if not body:
        return []
    return [
        "Against the all-query benchmark, one row per scene summary. The difference of medians "
        "is what the design ranks on; the paired columns are what make a difference of zero "
        "readable, because a median over an even number of images can only move in steps of "
        f"{1 / SPEARMAN_STEP_DENOMINATOR:.4f}.",
        "",
        *_table(
            ["summary", "candidate", "benchmark", "difference", "W/T/L",
             "win rate, all paired", "decided, over N"],
            body,
        ),
        "",
    ]


def _benchmark_counterexample(summary: dict, ranked: list[dict]) -> list[str]:
    """The honesty check: how often the paired comparison *fails* to favour a candidate.

    A paired win rate that favoured every candidate would be a property of the comparison
    rather than of the candidates, and the leading candidate's numbers would be worth nothing.
    So the counts are published for the whole ranked field, and the strongest counter-example
    is named -- a candidate that matches or beats the benchmark on the primary metric and
    still does not take the per-image comparison.

    That counter-example is reported as *undecided*, never as a defeat. On the pilot it is
    `decile_40_50` at `mean`: 103 wins against 112 losses with 35 ties, which separates
    nothing in either direction. "The benchmark beat it" is the wrong reading of a null.
    """
    favoured, unfavoured, even, candidates = 0, 0, 0, []
    for group in ranked:
        for entry in _matches(summary["benchmark_comparisons"], **_group_key(group)):
            total = entry["paired_image_count"]
            win = _whole(entry.get("candidate_image_win_rate"), total)
            loss = _whole(entry.get("candidate_image_loss_rate"), total)
            if win is None or loss is None:
                continue
            favoured += win > loss
            unfavoured += win < loss
            even += win == loss
            difference = entry.get("candidate_minus_benchmark_spearman")
            if win <= loss and difference is not None and float(difference) >= 0.0:
                candidates.append((win < loss, group, entry, win, loss,
                                   _whole(entry["candidate_image_tie_rate"], total)))
    if favoured + unfavoured + even == 0:
        return []
    lines = [
        "Across the ranked field the paired comparison against the benchmark comes out in the "
        f"candidate's favour {favoured} times, against it {unfavoured} times, and exactly even "
        f"{even} times, so it is not a procedure that favours whatever it is handed."
    ]
    # A candidate that is genuinely behind on the images is a sharper counter-example than one
    # that is exactly level, so it is preferred; both are reported as undecided rather than lost.
    strict = [entry for entry in candidates if entry[0]]
    chosen = (strict or candidates or [None])[0]
    if chosen is not None:
        _, group, entry, win, loss, tie = chosen
        decided = entry["candidate_image_decided_image_count"]
        tail = (
            f"{_plain(entry['candidate_image_decided_win_rate'])} of the {decided} images it "
            "decided" if decided else "and the comparison decided none of them at all"
        )
        lines.append(
            f"The clearest counter-example is `{group['confidence_bin']}` at "
            f"`{group['aggregation']}`, which matches or beats the benchmark's median "
            f"({_signed(entry['candidate_median_spearman'])} against "
            f"{_signed(entry['benchmark_median_spearman'])}) and still comes out {win} wins to "
            f"{loss} losses with {tie} ties -- {tail}. At counts that close the comparison is "
            "**undecided**: it has not established the candidate, and it has not established "
            "the benchmark either."
        )
    lines.append("")
    return lines


def _ranked_section(summary: dict, ranked: list[dict], winner: dict | None) -> list[str]:
    lines = ["## Best confidence range", ""]
    if winner is None:
        lines.extend([
            "No group cleared the deployable gate, so no confidence range is recommended. "
            "The full group table in `summary.json` still holds every measurement.",
            "",
        ])
        return lines
    tied = [
        group for group in ranked if group["median_spearman"] == winner["median_spearman"]
    ]
    lines.append(
        f"The ranking puts `{winner['confidence_bin']}` "
        f"(`{winner['membership_mode']}`, `{winner['aggregation']}`) first at "
        f"{_signed(winner['median_spearman'])}."
    )
    if len(tied) > 1:
        others = len(tied) - 1
        lines.append(
            f"It shares that median with {others} other candidate"
            f"{'' if others == 1 else 's'} of the {len(ranked)} ranked, and the order among "
            "them is settled by the adjacent non-decrease rate and then by the violation "
            "magnitude -- both far finer than the metric they are breaking a tie in, so the "
            "first place is not a gap over the second."
        )
    same_bin = [
        group for group in ranked
        if group["confidence_bin"] == winner["confidence_bin"]
        and group["membership_mode"] == winner["membership_mode"]
    ]
    if len(same_bin) > 1:
        spread = ", ".join(
            f"`{group['aggregation']}` {_signed(group['median_spearman'])}"
            for group in sorted(same_bin, key=lambda group: group["aggregation"])
        )
        lines.append(
            f"The same bin across every scene summary it was ranked at: {spread}. These are "
            f"{len(same_bin)} views of one bin on the same images, not "
            f"{len(same_bin)} independent confirmations of it."
        )
    lines.append("")
    lines.extend(_bin_versus_benchmark(summary, same_bin or [winner]))
    lines.extend(_benchmark_counterexample(summary, ranked))
    lines.append(
        "The deployable ranking, best first. Only candidates with the padding union removed, "
        f"at `{PRIMARY_SCORE_SCOPE}`, on a membership a single image can rebuild, and at full "
        "coverage, appear in it at all."
    )
    lines.append("")
    shown = ranked[:15]
    lines.extend(_table(
        ["#", "membership", "bin", "summary", "median Spearman", "adjacent non-decrease",
         "max blur above clean", "overlap vs severity 0", "coverage"],
        [
            [
                str(position), f"`{group['membership_mode']}`", f"`{group['confidence_bin']}`",
                f"`{group['aggregation']}`", _signed(group["median_spearman"]),
                _plain(group["mean_adjacent_monotonicity"]),
                _plain(group["endpoint_increase_rate"]),
                _plain(group["mean_clean_overlap_from_severity_1"], 4)
                + (" (arithmetic)" if group["clean_overlap_is_definitional"] else ""),
                f"{group['scored_severity_count']}/{group['expected_severity_count']}",
            ]
            for position, group in enumerate(shown, start=1)
        ],
    ))
    if len(ranked) > len(shown):
        lines.append("")
        lines.append(
            f"{len(ranked) - len(shown)} further candidates are in `summary.json` under "
            f"`{RANKED_GROUPS_KEY}`."
        )
    lines.append("")
    return lines


def _confidence_section(summary: dict, winner: dict | None) -> list[str]:
    lines = ["## Persistence versus confidence alone", ""]
    lines.extend([_confidence_sentence(summary, winner), ""])
    lines.append(
        "Every pair below shares one selection and one scene summary, which is what makes the "
        "comparison fair; the two scores are in unrelated units, so only their trends are ever "
        "compared. Both denominators are given because a per-image Spearman over "
        f"{len(EXPECTED_SEVERITIES)} severities lands on a coarse grid and exact ties are "
        "common: a win rate over every paired image counts each tie as a non-win, while a rate "
        "over the decided images alone hides how much of the run could not be separated."
    )
    lines.append("")
    control = _lookup(
        summary["groups"], signal="confidence", score_scope=CONFIDENCE_SCOPE,
        membership_mode=ALL_VALID_BENCHMARK[0], confidence_bin=ALL_VALID_BENCHMARK[1],
        aggregation=BENCHMARK_AGGREGATION, padding_mode=ALL_VALID_BENCHMARK[2],
    )
    if control is not None and control["median_spearman"] is not None:
        median = float(control["median_spearman"])
        if median < 0:
            lines.extend([
                "**Read the difference column with the control's own column beside it.** Over "
                f"every valid query the confidence control itself trends "
                f"{_signed(median)}: `1 - confidence` *falls* as blur rises in this "
                "configuration, so the control is strongly anti-correlated in its own right. "
                "A large positive difference is therefore partly a statement about the "
                "control and only partly about persistence, and the persistence column is the "
                "one that says whether the signal rises at all. The bottom bin is the clearest "
                "case: it out-trends its control while barely trending itself.",
                "",
            ])
        else:
            lines.extend([
                "Over every valid query the confidence control itself trends "
                f"{_signed(median)}, so the difference column below is a comparison between "
                "two signals that both move with blur rather than a comparison against a "
                "control that moves the wrong way.",
                "",
            ])
    lines.extend([_slice_caption(scoped=True, summaries="every"), ""])
    body = []
    for name in [*DECILE_NAMES, ALL_VALID_BENCHMARK[1]]:
        for membership in (DYNAMIC_MEMBERSHIP_MODE, FROZEN_MEMBERSHIP_MODE,
                           ALL_VALID_BENCHMARK[0]):
            for aggregation in DECILE_AGGREGATIONS:
                entry = _lookup(
                    summary["comparisons"], membership_mode=membership, confidence_bin=name,
                    aggregation=aggregation, score_scope=PRIMARY_SCORE_SCOPE,
                    padding_mode=FILTERED_PADDING_MODE,
                )
                if entry is None:
                    continue
                total = entry["paired_image_count"]
                body.append([
                    _membership_cell(membership), f"`{name}`", f"`{aggregation}`",
                    _signed(entry["persistence_median_spearman"]),
                    _signed(entry["confidence_median_spearman"]),
                    _signed(entry["persistence_minus_confidence_spearman"]),
                    f"{_whole(entry['persistence_image_win_rate'], total)}/"
                    f"{_whole(entry['persistence_image_tie_rate'], total)}/"
                    f"{_whole(entry['persistence_image_loss_rate'], total)}",
                    _plain(entry["persistence_image_win_rate"]),
                    f"{_plain(entry['persistence_image_decided_win_rate'])} "
                    f"({entry['persistence_image_decided_image_count']})",
                ])
    if body:
        lines.extend(_table(
            ["membership", "bin", "summary", "persistence", "confidence", "difference",
             "W/T/L", "win rate, all paired", "decided, over N"],
            body,
        ))
    else:
        lines.append(
            f"No matched persistence/confidence pair was scored at `{PRIMARY_SCORE_SCOPE}` "
            "with the padding union removed in this table."
        )
    lines.append("")
    return lines


def _dynamic_frozen_section(summary: dict) -> list[str]:
    lines = ["## Dynamic versus frozen queries", ""]
    lines.append(
        "`dynamic` rebuilds the ten bins from the blurred image itself, which is something a "
        "single image can do. `frozen` reuses the bins built from the clean image, which a "
        "naturally corrupted image cannot: there is no paired clean version of it. So a strong "
        "frozen result is a diagnostic -- it says how far the fingerprints moved once "
        "membership is held still -- and never a method. The two are reported side by side and "
        "are never combined into one score."
    )
    lines.append("")
    lines.extend([_slice_caption(scoped=False, summaries="every"), ""])
    body = []
    for name in DECILE_NAMES:
        for aggregation in DECILE_AGGREGATIONS:
            dynamic = _decile_group(summary, DYNAMIC_MEMBERSHIP_MODE, name,
                                    aggregation=aggregation)
            frozen = _decile_group(summary, FROZEN_MEMBERSHIP_MODE, name,
                                   aggregation=aggregation)
            if dynamic is None and frozen is None:
                continue
            entry = _lookup(
                summary["membership_comparisons"], signal="persistence",
                confidence_bin=name, aggregation=aggregation,
                score_scope=PRIMARY_SCORE_SCOPE, padding_mode=FILTERED_PADDING_MODE,
            )
            total = None if entry is None else entry["paired_image_count"]
            body.append([
                f"`{name}`", f"`{aggregation}`",
                _UNMEASURED if dynamic is None else _signed(dynamic["median_spearman"]),
                _UNMEASURED if frozen is None else _signed(frozen["median_spearman"]),
                _UNMEASURED if entry is None else (
                    f"{_whole(entry['dynamic_image_win_rate'], total)}/"
                    f"{_whole(entry['dynamic_image_tie_rate'], total)}/"
                    f"{_whole(entry['dynamic_image_loss_rate'], total)}"
                ),
                _UNMEASURED if entry is None else (
                    f"{_plain(entry['dynamic_image_decided_win_rate'])} "
                    f"({entry['dynamic_image_decided_image_count']})"
                ),
                _UNMEASURED if dynamic is None
                else _plain(dynamic["mean_clean_overlap_from_severity_1"], 4),
            ])
    if body:
        lines.extend(_table(
            ["bin", "summary", "dynamic", "frozen (diagnostic)", "dynamic W/T/L vs frozen",
             "decided win rate, over N", "dynamic overlap vs severity 0"], body
        ))
        lines.append("")
        lines.append(
            "The `W/T/L` column is the paired comparison the difference of medians cannot "
            "make: one selection against its own frozen twin, image by image, over the images "
            "both measured. Freezing the membership at severity zero is the only manipulation "
            "in this design that removes the movement of queries between bins and leaves "
            "everything else, so this column -- and not the gap between the two medians -- is "
            "what says whether that movement explains a bin's trend. It is a comparison and "
            "never a combination: neither column is added to, averaged with, or subtracted "
            "from the other in anything this report ranks."
        )
        lines.append("")
    lines.extend(_median_blindness_note(summary))
    lines.append(
        f"Two unrelated decile memberships would overlap at {RANDOM_BIN_OVERLAP:.4f}, so a bin "
        "sitting close to that line is being rebuilt almost from scratch at every severity. A "
        "`frozen` or `shared` row would print 1.000 in that column by construction, which is "
        "arithmetic rather than a measurement, and it is left out of the column above for that "
        "reason."
    )
    lines.append("")
    return lines


def _median_blindness_note(summary: dict) -> list[str]:
    """Show, from this run's own rows, that the difference of medians could not have answered.

    The paired family exists because a difference of two medians is blind at the primary
    metric's resolution. That is an argument; this turns it into a measurement. It collects the
    rows whose median difference is **exactly zero** -- a dead heat by the design's primary
    metric -- and reports how far apart the paired comparison puts them. When two such rows land
    on opposite sides of even, the primary metric demonstrably could not have told them apart,
    and no reader has to take the claim on trust.

    Emitted only when the run actually contains the demonstration. A table with fewer than two
    exact median ties says nothing here rather than reaching for a weaker version of it.
    """
    tied = [
        entry for entry in summary["membership_comparisons"]
        if entry["signal"] == "persistence"
        and entry["score_scope"] == PRIMARY_SCORE_SCOPE
        and entry["padding_mode"] == FILTERED_PADDING_MODE
        and entry["dynamic_minus_frozen_spearman"] == 0.0
        and entry["dynamic_image_decided_win_rate"] is not None
    ]
    if len(tied) < 2:
        return []
    ordered = sorted(
        tied, key=lambda entry: (entry["dynamic_image_decided_win_rate"],
                                 entry["confidence_bin"], entry["aggregation"])
    )
    low, high = ordered[0], ordered[-1]
    if low["dynamic_image_decided_win_rate"] == high["dynamic_image_decided_win_rate"]:
        return []

    def described(entry: dict) -> str:
        total = entry["paired_image_count"]
        win = _whole(entry["dynamic_image_win_rate"], total)
        loss = _whole(entry["dynamic_image_loss_rate"], total)
        ahead = "dynamic" if win >= loss else "frozen"
        return (
            f"`{entry['confidence_bin']}` at `{entry['aggregation']}` has {ahead} ahead "
            f"{max(win, loss)} images to {min(win, loss)}, a decided rate of "
            f"{_plain(entry['dynamic_image_decided_win_rate'])} over the "
            f"{entry['dynamic_image_decided_image_count']} it decided"
        )

    opposed = (
        low["dynamic_image_decided_win_rate"] < 0.5 <= high["dynamic_image_decided_win_rate"]
    )
    return [
        f"**Why the paired column and not the gap between the medians.** {len(tied)} of the "
        f"rows above have two medians that are *exactly equal* -- a dead heat on the statistic "
        "the design ranks by. The per-image comparison puts them "
        + ("on opposite sides of even" if opposed else "far apart")
        + f": {described(high)}, while {described(low)}. Identical on the primary metric, "
        + ("opposite answers" if opposed else "different answers")
        + " image by image -- which is what a difference of two medians on a 36-valued "
        "statistic cannot see.",
        "",
    ]


def _padding_section(summary: dict) -> list[str]:
    padding = summary.get("padding") or {}
    lines = ["## Effect of padded queries", ""]
    padded_images = padding.get("images_with_padding", 0)
    image_count = padding.get("image_count", 0)
    lines.append(
        f"{padded_images} of {image_count} images carry a repeated decoder tail; together they "
        f"contribute {padding.get('total_union_padded_count', 0)} padded query slots, and the "
        f"largest single image loses {padding.get('max_union_padded_count', 0)} of them. Of "
        f"those {padded_images} padded images, "
        f"{padding.get('padded_images_with_identical_tails', 0)} have the same detected tail "
        "at all six severities -- the tail wanders with blur rather than growing, which is why "
        "one mask is taken per image and reused at every severity instead of one per severity. "
        f"On the remaining {max(image_count - padded_images, 0)} images there is nothing to "
        "remove, so the control is a no-op there by construction and any median taken over all "
        f"{image_count} images is diluted by them."
    )
    lines.append("")
    entries = sorted(
        (
            entry for entry in summary["padding_sensitivity"]
            if entry["score_scope"] in (PRIMARY_SCORE_SCOPE, CONFIDENCE_SCOPE)
        ),
        key=lambda entry: (entry["confidence_bin"], entry["membership_mode"], entry["signal"],
                           entry["aggregation"]),
    )
    if entries:
        lines.append(
            f"Every scene summary the control was scored at. Persistence rows are at "
            f"`{PRIMARY_SCORE_SCOPE}`; the confidence control has no decoder-layer scope."
        )
        lines.append("")
        lines.extend(_table(
            ["bin", "membership", "signal", "summary", "filtered", "unfiltered", "difference",
             "score changed on", "decided win rate on those, over N"],
            [
                [
                    f"`{entry['confidence_bin']}`", _membership_cell(entry["membership_mode"]),
                    f"`{entry['signal']}`", f"`{entry['aggregation']}`",
                    _signed(entry["filtered_median_spearman"]),
                    _signed(entry["unfiltered_median_spearman"]),
                    _signed(entry["unfiltered_minus_filtered_spearman"]),
                    str(entry["score_changed_image_count"]),
                    f"{_plain(entry['score_changed_image_decided_win_rate'])} "
                    f"({entry['score_changed_image_decided_image_count']})",
                ]
                for entry in entries
            ],
        ))
        lines.append("")
    else:
        lines.extend(["No unfiltered control was scored in this table.", ""])
    lines.append(
        "The `score changed on` column is a **lower bound** on the images the padding mask "
        "reached. It counts the images whose scene score moved, and a changed selection can "
        "still produce a bit-identical score because a scene summary is a many-to-one map -- "
        "the same selection pair reports different counts under different summaries, which a "
        "set of images a mask reached could not do. It must not be read as the images the "
        "padding changed."
    )
    lines.append("")
    return lines


def _metrics_section(summary: dict) -> list[str]:
    return [
        "## Metrics in plain language",
        "",
        "- **Median per-image Spearman** is the headline. For one image, rank its six scene "
        "scores against the six blur levels and correlate the ranks; +1 means the score rose "
        "at every step, 0 means no relation, -1 means it fell throughout. The median is taken "
        "across images. Over "
        f"{len(EXPECTED_SEVERITIES)} severities this can only land on "
        f"{len(EXPECTED_SEVERITIES) * (len(EXPECTED_SEVERITIES) ** 2 - 1) // 6 + 1} distinct "
        f"values, spaced {2 / SPEARMAN_STEP_DENOMINATOR:.4f} apart, so two configurations can "
        "tie here while differing on most images. That is why every comparison is also made "
        "image by image.",
        "- **Adjacent non-decrease rate** is the share of the five steps between neighbouring "
        "blur levels on which the score did not fall. **Violation magnitude** is how far it "
        "fell when it did, as a share of that image's own score range.",
        "- **Maximum-blur-above-clean rate** is the share of images whose worst blur level "
        "scored above their clean one -- the weakest thing a useful signal must do.",
        "- **Coverage** is scored image-severity pairs against the pairs the sweep should have "
        "produced. A configuration that collapsed at high blur and one that rose the whole way "
        "can publish the same median, so full coverage is required rather than noted.",
        "- **Overlap vs severity 0** is the Jaccard overlap between a bin's membership at a "
        "blur level and the same bin on the clean image, averaged over severities 1 to 5. "
        f"Two unrelated memberships would score {RANDOM_BIN_OVERLAP:.4f}. A 1.000 on a "
        "`frozen` or `shared` row is arithmetic, not evidence: those selections are the same "
        "query set at every severity by construction.",
        "- **W/T/L and the two win rates.** Ties are common at this resolution, so a rate over "
        "every paired image and a rate over the decided ones alone can fall on opposite sides "
        "of 0.5. Both are always given; neither is a significance test.",
        "",
        "**Neither score is a probability.** Persistence uncertainty is a distance from an "
        "image's decoder fingerprints to the nearest clean ones, and the confidence control is "
        "one minus the detector's largest class score. That subtraction only reverses "
        "direction -- nothing here is trained or calibrated, and a value of 0.8 does not mean "
        "an image is 80 percent corrupted. Distances and `1 - confidence` share no unit, so "
        "the two signals are only ever compared through their trends and never through their "
        "raw magnitudes.",
        "",
    ]


def _limits_section(summary: dict, ranked: list[dict]) -> list[str]:
    return [
        "## What this does not prove",
        "",
        "- **Selecting and reporting on the same images.** Any candidate named above was "
        f"chosen best of {len(ranked)} ranked candidates on the same images the numbers "
        "describe. Its counts are conditioned on that choice and are not a hypothesis test of "
        "anything.",
        "- **Three scene summaries of one bin are one result.** They are computed from the "
        "same selections on the same images; agreement between them is arithmetic, not "
        "replication.",
        "- **The ranking did not choose the padding rule.** Every candidate it admits has the "
        "padding union removed, so the choice between removing and keeping the decoder "
        "placeholders is not settled by the ranking. The sensitivity table above is the "
        "evidence for that half of the decision, and on a run where the unfiltered rows trend "
        "higher, what that shows is that the placeholders carry a blur signal of their own -- "
        "not that the method should keep them.",
        "- **A frozen result is not a method.** It needs a paired clean image, which a "
        "naturally corrupted one does not have. Frozen rows are excluded from the ranking by "
        "design and are never added to or averaged with dynamic ones.",
        "- **`score changed on` is a lower bound**, not the set of images the padding mask "
        "reached.",
        "- **An undecided comparison is not a loss.** A pair that comes out close to even, "
        "with many ties, has not established anything in either direction.",
        "",
    ]


def _next_section(summary: dict, winner: dict | None) -> list[str]:
    lines = ["## Next decision", ""]
    if winner is None:
        lines.append(
            "Nothing here is ready to carry forward: no selection cleared the deployable gate."
        )
    else:
        lines.append(
            f"The tuning run is allowed to fix at most one confidence bin, one membership "
            f"rule, one scene summary and one padding rule. On this run that is "
            f"`{winner['confidence_bin']}` / `{winner['membership_mode']}` / "
            f"`{winner['aggregation']}` for the first three; the fourth is not decided by the "
            "ranking and has to be argued from the sensitivity table."
        )
    lines.extend([
        "",
        "That choice is a proposal to be reviewed before anything is scored on the held-out "
        "images. Until that review and that run happen, every number in this report describes "
        "the images it was selected on.",
        "",
        EASY_REPORT_FINAL_SENTENCE,
    ])
    return lines


def _easy_report(summary: dict) -> str:
    """The plain-language report spec:171 asks for, generated from the summary and nothing else.

    Every sentence with a number in it is built from `summary`, so two runs over the same rows
    produce the same file and a reader can check any claim against `summary.json`. Nothing is
    quoted from a previous run.

    The opening section is fixed by spec:172 -- which confidence range worked best, whether it
    beat confidence alone, whether it beat the existing all-query benchmark, and whether
    padding or bin movement explains the result, in that order. `SPEC_172_QUESTIONS` is the
    order, so it is asserted rather than typed.

    Three sentences this generator cannot produce, each because the design forbids it:

    * a score described as a probability of corruption (spec:111) -- the confidence control is
      `1 - confidence`, which reverses direction and calibrates nothing;
    * a dynamic and a frozen result merged into one recommendation (spec:230) -- frozen rows
      are absent from the ranking and are labelled diagnostic wherever they appear;
    * a paired rate quoted with one denominator -- `_outcome` publishes both, because on the
      pilot the two framings put the same candidate on opposite sides of 0.5.

    And one word it avoids: nothing here "beats" anything. The verdict verbs come from
    `_verdict`, which reads the win and loss counts and has no threshold in it, and every
    verdict is followed by the counts it was derived from.
    """
    ranked = list(summary.get(RANKED_GROUPS_KEY) or [])
    winner = ranked[0] if ranked else None
    lines = [EASY_REPORT_TITLE, ""]
    lines.extend(_preamble(summary))
    lines.extend(_short_answer(summary, ranked, winner))
    lines.extend(_ranked_section(summary, ranked, winner))
    lines.extend(_confidence_section(summary, winner))
    lines.extend(_dynamic_frozen_section(summary))
    lines.extend(_padding_section(summary))
    lines.extend(_metrics_section(summary))
    lines.extend(_limits_section(summary, ranked))
    lines.extend(_next_section(summary, winner))
    return "\n".join(lines).rstrip() + "\n"


def write_decile_report(
    rows: list[dict], output_dir, run_metadata: dict | None = None,
    diagnostics: dict | None = None,
) -> None:
    """Write the seven artifacts spec:163-171 names, or write nothing at all.

    The order is deliberate on two counts.

    **The summary is computed first, before the output directory exists.** Every refusal the
    design asks for -- a duplicated result key, a label outside its vocabulary, a table
    spanning two partitions -- lives in `summarize_decile_rows`, so a table that cannot be
    summarised leaves no directory behind to be mistaken for a partial run.

    **The results frame is built exactly twice over the whole call and never twice at once.**
    `summarize_decile_rows` builds one and drops it; the CSV and the blur curves then share a
    second. On the real tuning table that frame is 523,500 rows, and `summary_frame` is what
    keeps the ~280 MB `selected_query_ids` column out of it -- reused here rather than
    rebuilt, so the CSV is by construction the same table the summary describes.

    Every file goes out through a temporary and `os.replace`. A truncated `summary.json` is a
    parse error, which is loud; a truncated `easy-report.md` is a shorter report that stops
    mid-sentence and reads as finished.
    """
    summary = summarize_decile_rows(rows, run_metadata, diagnostics)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    frame = summary_frame(rows)
    csv_path = output / "per_scene.csv"
    csv_temporary = csv_path.with_suffix(".csv.tmp")
    frame.to_csv(csv_temporary, index=False)
    os.replace(csv_temporary, csv_path)
    _save_figure(_blur_curve_figure(frame), output / "blur_curves.png")
    del frame

    _atomic_text(
        output / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False),
    )
    _save_figure(_heatmap_figure(summary), output / "confidence_decile_heatmap.png")
    _save_figure(_dynamic_frozen_figure(summary), output / "dynamic_vs_frozen.png")
    _save_figure(_padding_sensitivity_figure(summary), output / "padding_sensitivity.png")
    _atomic_text(output / "easy-report.md", _easy_report(summary))
