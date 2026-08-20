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
* A **paired statistic computed over marginal populations** is the failure spec:157 guards
  against: the fraction of images where persistence beats its confidence control is only
  meaningful over images both signals scored. `paired_image_count` is published next to the
  win rate so a reader can see the denominator.
* A **raw score magnitude leaking into a comparison** (spec:157 again). Persistence distances
  and `1 - confidence` are in unrelated units; the only fair comparison is between their
  *trends*. Every number this module publishes about a score is invariant under
  `score -> a * score + b` for `a > 0`, and a test asserts exactly that over the whole summary.

`summarize_decile_rows` is also where the design's "output result keys would be duplicated"
refusal lives (spec:194). `analyze_deciles` cannot produce a collision and has a test proving
it does not, but rows also arrive from a CSV, from two concatenated runs, or from a future
producer -- so the check belongs at the boundary rows arrive through, and here it is.
"""

from __future__ import annotations

import math
import operator

import numpy as np
import pandas as pd

from .decile_analysis import (
    ALL_QUERY_BENCHMARK,
    EXPECTED_SEVERITIES,
    ROW_KEYS_EXCLUDED_FROM_CSV,
)
from .decile_scoring import (
    COMBINED_SCOPE,
    CONFIDENCE_BINS,
    CONFIDENCE_SCOPE,
    DECILE_AGGREGATIONS,
    MEMBERSHIP_MODES,
    PADDING_MODES,
    PRIMARY_SCORE_SCOPE,
)
from .metrics import monotonicity_metrics


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

FILTERED_PADDING_MODE = "filtered"
UNFILTERED_PADDING_MODE = "unfiltered"


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


def _win_rate(left: dict[int, float], right: dict[int, float], images: list[int]) -> float | None:
    return _rate(left, right, images, operator.gt)


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


def _moved_images(left_arrays, right_arrays) -> set[int]:
    """The images whose scene scores differ between two padding modes -- the ones the mask
    actually reached.

    Score equality, not Spearman equality, and the difference is not academic. The filtered
    and unfiltered selections differ on every image that carries a padded tail -- 66 of the
    pilot's 250 -- but a per-image Spearman takes only 35 distinct values over six severities,
    so on many of those images the scores all move and the rank correlation lands on the same
    value it had before. Reading the moved set off the Spearman counts those images as
    untouched: on the pilot it reports 50 to 56 of 250 where the score-derived answer is 64 to
    66, and a rate restricted to it is a tie-excluding sign test wearing the label "the images
    the control could reach". The `all_valid` pair is the case with no room for judgement --
    its two selections are every valid query against every query, so they differ on exactly the
    66 images that carry a padded tail, and that is what this returns.

    An image scored on one side and absent from the other counts as moved, and two `nan`
    scores count as unmoved -- both severities were unscored under either rule, which is the
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
      including `moved_image_count` and a win rate restricted to those images, because the
      control is a no-op on any image that had no padding to remove and the unrestricted rate
      counts every such image as a loss.
    * `benchmark_comparisons` -- every ranked candidate against the published all-300-query
      result (spec:229), paired per image. The spec asks only whether a candidate beats the
      benchmark; a difference of two medians answers that with a number whose resolution is
      1/35, because a Spearman over six severities takes 35 distinct values. The win, tie and
      loss rates say how many of the images the difference actually rests on, which is what
      makes a margin of one such step falsifiable rather than merely reportable.
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
            "persistence_image_win_rate": _win_rate(
                persistence_spearman, control_spearman, paired
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
        moved = _moved_images(arrays_by_group[key], arrays_by_group[filtered_key])
        moved_and_paired = [image for image in paired if image in moved]
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
            "unfiltered_image_win_rate": _win_rate(
                unfiltered_spearman, filtered_spearman, paired
            ),
            "paired_image_count": len(paired),
            # How many images the padding mask actually reached, read off the scores. Without
            # it the median difference cannot be told apart from a null result: an image with
            # no padded tail selects the identical queries either way, and on the pilot that is
            # 184 images out of 250.
            "moved_image_count": len(moved),
            # ... and the win rate restricted to those images. The unrestricted rate counts
            # every unreached image as a loss, so on the pilot's dynamic bottom bin it reads
            # 0.168 over all 250 while the unfiltered run out-trends the filtered one on 42 of
            # the 64 images the mask reached -- 0.656. Both are true; only the pair is not
            # misleading.
            "moved_image_win_rate": _win_rate(
                unfiltered_spearman, filtered_spearman, moved_and_paired
            ),
            "moved_and_paired_image_count": len(moved_and_paired),
        })
    sensitivity.sort(key=lambda entry: (
        entry["signal"], entry["membership_mode"], entry["confidence_bin"],
        entry["aggregation"], entry["score_scope"],
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
                "candidate_image_win_rate": _rate(
                    candidate_spearman, benchmark_spearman, paired, operator.gt
                ),
                "candidate_image_tie_rate": _rate(
                    candidate_spearman, benchmark_spearman, paired, operator.eq
                ),
                "candidate_image_loss_rate": _rate(
                    candidate_spearman, benchmark_spearman, paired, operator.lt
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
            "benchmark_comparison_count": len(benchmark_comparisons),
        },
        "padding": _padding_rollup(diagnostics),
        "groups": groups,
        "comparisons": comparisons,
        "padding_sensitivity": sensitivity,
        "benchmark_comparisons": benchmark_comparisons,
        RANKED_GROUPS_KEY: ranked,
    }
    return _jsonable(summary)
