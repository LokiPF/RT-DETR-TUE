"""Tests for the confidence-decile summariser and the artifacts it writes.

The brief's tests, plus one test per binding sentence of the spec they leave unexercised.
The summary tests cluster around four places where a summary can be complete, plausible and
wrong: a group key that stops separating two selections, a coverage gate that annotates
instead of excluding, a paired statistic computed over marginal populations, and a raw score
magnitude leaking into a number that is compared across scopes.

The writer tests cluster around three more, all of them failures a reader cannot see. A
**figure that silently drops the bins it has no rows for** looks like a complete ten-bin
measurement drawn on whatever subset happened to exist. A **sentence that outruns its
numbers** -- "beat", "affected by padding", "80 percent likely" -- is the failure spec:111
and spec:230 name outright, and prose is the one artifact no schema check reaches. And a
**paired rate published with one denominator** inverts the reading, which is why the report
is checked for both.
"""

from __future__ import annotations

import ast
import json
import math
import random
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
import torch

from src.scene_uncertainty import decile_reporting as reporting_module
from src.scene_uncertainty.confidence_deciles import DECILE_NAMES
from src.scene_uncertainty.decile_analysis import (
    ALL_QUERY_BENCHMARK,
    ALL_VALID_BENCHMARK,
    ROW_KEYS_EXCLUDED_FROM_CSV,
    SENSITIVITY_BIN,
)
from src.scene_uncertainty.decile_reporting import (
    BENCHMARK_SELECTION,
    EASY_REPORT_FINAL_SENTENCE,
    GROUP_KEYS,
    RANDOM_BIN_OVERLAP,
    RANKED_GROUPS_KEY,
    RANKABLE_MEMBERSHIP_MODES,
    ROW_KEYS,
    SIGNALS,
    SPEC_172_QUESTIONS,
    rank_deployable_groups,
    summarize_decile_rows,
    summary_frame,
    write_decile_report,
)
from src.scene_uncertainty.decile_scoring import (
    CONFIDENCE_SCOPE,
    DECILE_AGGREGATIONS,
    PRIMARY_SCORE_SCOPE,
    score_selection,
)


# --- fixtures ---------------------------------------------------------------------------
#
# Rows carry exactly the keys `decile_scoring.score_selection` emits, including
# `selected_query_ids`, so that the summariser is exercised against the shape its only real
# producer builds -- and so that the column the memory contract excludes is actually present
# to be excluded.

BASE_ROW = {
    "source_partition": "tuning",
    "membership_mode": "dynamic",
    "confidence_bin": "decile_00_10",
    "padding_mode": "filtered",
    "aggregation": "q90",
    "selected_count": 30,
    "clean_overlap": 1.0,
}

# One image rises without a stumble; the other rises with a dip at severity 3, so that
# `mean_adjacent_monotonicity` and `mean_violation_magnitude` are not both trivially at their
# best value in every fixture that uses it.
DIPPING_CURVE = (0.0, 1.0, 2.0, 1.5, 4.0, 5.0)


def row(image_id, severity, signal, score_scope, score, **overrides):
    values = {
        **BASE_ROW,
        "image_id": image_id,
        "severity": severity,
        "signal": signal,
        "score_scope": score_scope,
        "score": float(score),
        "selected_query_ids": [0, 1, 2],
    }
    values.update(overrides)
    return values


def synthetic_rows():
    """Two images over six severities: persistence rises for both, confidence for one.

    Image 2's confidence control is flat, which `monotonicity_metrics` reports as a Spearman
    of 0.0 rather than `nan` -- so the confidence group's median Spearman is 0.5 against
    persistence's 1.0, and persistence out-trends its control on exactly one of the two
    images. That is the smallest table in which a difference of medians and a paired win
    fraction can disagree, which is the point of publishing both.
    """
    persistence = {1: (0.0, 1.0, 2.0, 3.0, 4.0, 5.0), 2: (0.0, 2.0, 4.0, 6.0, 8.0, 10.0)}
    confidence = {1: (0.1, 0.2, 0.3, 0.4, 0.5, 0.6), 2: (0.5,) * 6}
    rows = []
    for image_id in (1, 2):
        for severity in range(6):
            rows.append(row(
                image_id, severity, "persistence", PRIMARY_SCORE_SCOPE,
                persistence[image_id][severity],
            ))
            rows.append(row(
                image_id, severity, "confidence", "confidence",
                confidence[image_id][severity],
            ))
    return rows


def rows_with_incomplete_group():
    """Four layer-2 persistence groups, only two of which may be ranked deployable.

    Every group that must be excluded is deliberately the *strongest*: the frozen group, the
    under-covered group, the unfiltered group and the secondary-scope group all rise perfectly,
    while the two legitimate candidates only dip-and-rise. So a gate that annotates instead of
    excluding, or a ranking that forgets membership, padding or scope, puts a decoy first and
    the test fails on the value rather than only on the count.
    """
    rows = []
    for image_id in (1, 2):
        for severity in range(6):
            rows.append(row(
                image_id, severity, "persistence", PRIMARY_SCORE_SCOPE, severity,
                membership_mode="frozen",
            ))
            rows.append(row(
                image_id, severity, "persistence", PRIMARY_SCORE_SCOPE,
                float("nan") if (image_id == 2 and severity == 5) else float(severity),
                confidence_bin="decile_10_20",
            ))
            # The all-300-query benchmark lives under `unfiltered` and keeps the padded
            # decoder placeholders (spec:71); it is what the candidates are measured against,
            # never a candidate.
            rows.append(row(
                image_id, severity, "persistence", PRIMARY_SCORE_SCOPE, severity,
                padding_mode="unfiltered",
            ))
            # Layers 0 and 1 and `combined` are secondary diagnostics (spec:123).
            rows.append(row(image_id, severity, "persistence", "layer_0", severity))
            rows.append(row(
                image_id, severity, "persistence", PRIMARY_SCORE_SCOPE,
                DIPPING_CURVE[severity],
            ))
            rows.append(row(
                image_id, severity, "persistence", PRIMARY_SCORE_SCOPE,
                DIPPING_CURVE[severity],
                membership_mode="shared", confidence_bin="all_valid",
            ))
    return rows


def benchmark_rows(curves):
    """The published all-300-query benchmark: `("shared", "all_valid", "unfiltered")` at q90.

    Built from `ALL_QUERY_BENCHMARK` rather than from the three literals, so a test that stops
    describing the producer's benchmark fails here instead of quietly comparing candidates
    against a selection nothing emits.
    """
    membership, confidence_bin, padding = ALL_QUERY_BENCHMARK
    return [
        row(
            image_id, severity, "persistence", PRIMARY_SCORE_SCOPE, curve[severity],
            membership_mode=membership, confidence_bin=confidence_bin, padding_mode=padding,
            selected_count=300,
        )
        for image_id, curve in curves.items()
        for severity in range(6)
    ]


def candidate_rows(curves):
    """A filtered, full-coverage, layer-2 dynamic selection -- the shape the ranking admits."""
    return [
        row(image_id, severity, "persistence", PRIMARY_SCORE_SCOPE, curve[severity])
        for image_id, curve in curves.items()
        for severity in range(6)
    ]


def dipping_rows():
    """One image whose persistence curve dips, so violation magnitude is non-zero."""
    rows = []
    for severity in range(6):
        rows.append(row(1, severity, "persistence", PRIMARY_SCORE_SCOPE, DIPPING_CURVE[severity]))
        rows.append(row(1, severity, "confidence", "confidence", 0.1 * severity))
    return rows


def group(summary, signal, score_scope):
    matches = [
        candidate for candidate in summary["groups"]
        if candidate["signal"] == signal and candidate["score_scope"] == score_scope
    ]
    assert len(matches) == 1, f"expected one {signal}/{score_scope} group, got {len(matches)}"
    return matches[0]


def ranked_keys(summary):
    return [
        (
            entry["membership_mode"], entry["confidence_bin"],
            entry["padding_mode"], entry["score_scope"],
        )
        for entry in summary[RANKED_GROUPS_KEY]
    ]


def candidate(**overrides):
    """A full-coverage, deployable layer-2 persistence group, for ranking unit tests."""
    values = {
        "signal": "persistence",
        "membership_mode": "dynamic",
        "confidence_bin": "decile_00_10",
        "aggregation": "q90",
        "score_scope": PRIMARY_SCORE_SCOPE,
        "padding_mode": "filtered",
        "image_count": 2,
        "scored_image_count": 2,
        "scored_severity_count": 12,
        "total_severity_count": 12,
        "expected_severity_count": 12,
        "median_spearman": 0.5,
        "mean_adjacent_monotonicity": 0.8,
        "mean_violation_magnitude": 0.1,
    }
    values.update(overrides)
    return values


# --- the brief's tests ------------------------------------------------------------------


def test_summary_reports_complete_trends_and_paired_advantage():
    summary = summarize_decile_rows(synthetic_rows(), {"source_partition": "tuning"})
    persistence = group(summary, "persistence", "layer_2")
    confidence = group(summary, "confidence", "confidence")
    comparison = summary["comparisons"][0]
    assert persistence["median_spearman"] == 1.0
    assert persistence["scored_severity_count"] == persistence["total_severity_count"] == 12
    assert comparison["persistence_minus_confidence_spearman"] == pytest.approx(
        persistence["median_spearman"] - confidence["median_spearman"]
    )
    assert comparison["persistence_image_win_rate"] == 0.5


def test_ranking_requires_complete_deployable_groups():
    summary = summarize_decile_rows(rows_with_incomplete_group(), {})
    ranked = summary[RANKED_GROUPS_KEY]
    assert all(row["scored_severity_count"] == row["total_severity_count"] for row in ranked)
    assert all(row["membership_mode"] in {"dynamic", "shared"} for row in ranked)


def test_duplicate_score_key_is_rejected():
    rows = synthetic_rows()
    with pytest.raises(ValueError, match="duplicate"):
        summarize_decile_rows(rows + [rows[0]], {})


# --- spec:194, the refusal this task owns ------------------------------------------------


def test_the_duplicate_refusal_names_the_offending_key():
    """A refusal that does not say *which* key repeated sends the reader back to 523,500 rows."""
    rows = synthetic_rows()
    duplicated = rows[3]
    with pytest.raises(ValueError) as failure:
        summarize_decile_rows(rows + [duplicated], {})
    message = str(failure.value)
    for key in ROW_KEYS:
        assert f"{key}={duplicated[key]}" in message


"""`signal` is deliberately absent from the parametrisation below.

It is the one member of `ROW_KEYS` that is *determined* by another member: a row's signal is
`confidence` exactly when its scope is `confidence`, which
`test_a_confidence_row_carrying_a_decoder_layer_scope_is_rejected` pins. So dropping `signal`
from the key cannot manufacture a false duplicate, and a test claiming it does would be
asserting something untrue about the table.
"""
SEPARATING_ROW_KEYS = tuple(key for key in ROW_KEYS if key != "signal")

# Values chosen so that the twin row does not already exist in `synthetic_rows()`; a twin that
# collides with a real row would make this test pass for the wrong reason.
SEPARATING_ALTERNATIVES = {
    "image_id": 99,
    "severity": 9,
    "membership_mode": "frozen",
    "confidence_bin": "decile_90_100",
    "aggregation": "mean",
    "score_scope": "layer_0",
    "padding_mode": "unfiltered",
}


@pytest.mark.parametrize("field", SEPARATING_ROW_KEYS)
def test_two_rows_differing_in_one_key_field_are_not_duplicates(field):
    """Every field of `ROW_KEYS` has to separate two rows, or the key is too coarse.

    Dropping any one of them from the key turns a legitimate second measurement into a
    reported duplicate, which is a refusal on a correct table.
    """
    rows = synthetic_rows()
    twin = {**rows[0], field: SEPARATING_ALTERNATIVES[field]}
    summarize_decile_rows(rows + [twin], {})


# --- the memory contract ------------------------------------------------------------------


def test_the_excluded_columns_are_the_producers_constant():
    """Re-listing the names as literals is the mutation this kills: they would drift silently."""
    assert reporting_module.ROW_KEYS_EXCLUDED_FROM_CSV is ROW_KEYS_EXCLUDED_FROM_CSV


def test_the_excluded_columns_never_enter_the_summary_frame():
    frame = summary_frame(synthetic_rows())
    assert set(ROW_KEYS_EXCLUDED_FROM_CSV).isdisjoint(frame.columns)
    assert set(ROW_KEYS) <= set(frame.columns)


def test_rows_without_the_excluded_column_summarize_identically():
    """The summariser must never *read* the excluded columns, not merely drop them afterwards.

    `pd.DataFrame(rows).drop(columns=ROW_KEYS_EXCLUDED_FROM_CSV)` materialises the 280 MB of
    selection ids before discarding them, and it also raises `KeyError` on a table that never
    carried them. Both faults die here: this table has no `selected_query_ids` at all.
    """
    stripped = [
        {key: value for key, value in scored.items() if key not in ROW_KEYS_EXCLUDED_FROM_CSV}
        for scored in synthetic_rows()
    ]
    assert summarize_decile_rows(stripped, {}) == summarize_decile_rows(synthetic_rows(), {})


# --- spec:193, an unknown label ------------------------------------------------------------


@pytest.mark.parametrize(("field", "value"), [
    ("signal", "hunch"),
    ("membership_mode", "legacy"),
    ("confidence_bin", "all_300"),
    ("aggregation", "median"),
    ("padding_mode", "partly"),
    ("score_scope", "layer_two"),
])
def test_an_unknown_label_is_rejected(field, value):
    """Spec:193. A typo does not fail a group-by -- it opens a phantom group and subtracts
    its rows from the group they belonged to, which no downstream number reports."""
    rows = synthetic_rows()
    rows[0] = {**rows[0], field: value}
    with pytest.raises(ValueError, match="unknown"):
        summarize_decile_rows(rows, {})


def test_a_confidence_row_carrying_a_decoder_layer_scope_is_rejected():
    """Spec:125 -- confidence uncertainty has no decoder-layer scope.

    A confidence row filed under `layer_2` would be paired against itself as its own control,
    and the comparison would report a difference of exactly zero for a reason that has nothing
    to do with the experiment.
    """
    rows = synthetic_rows()
    rows[1] = {**rows[1], "score_scope": PRIMARY_SCORE_SCOPE}
    with pytest.raises(ValueError, match="confidence"):
        summarize_decile_rows(rows, {})


def test_a_persistence_row_under_the_confidence_scope_is_rejected():
    rows = synthetic_rows()
    rows[0] = {**rows[0], "score_scope": "confidence"}
    with pytest.raises(ValueError, match="confidence"):
        summarize_decile_rows(rows, {})


def test_the_signal_vocabulary_matches_what_the_scorer_emits():
    """A drift guard: `SIGNALS` is two literals here and two literals in `score_selection`."""
    emitted = score_selection(
        image_id=1, severity=0, source_partition="tuning",
        membership_mode="dynamic", confidence_bin="decile_00_10", padding_mode="filtered",
        indices=torch.tensor([0, 1]),
        query_confidence=torch.tensor([0.5, 0.25, 0.75]),
        query_scores_by_layer={2: torch.tensor([1.0, 2.0, 3.0])},
        layer_score_scales={2: {"center": torch.tensor(0.0), "scale": torch.tensor(1.0)}},
        clean_overlap=1.0, aggregations=("q90",),
    )
    assert {scored["signal"] for scored in emitted} == set(SIGNALS)


# --- spec:190 and 196, the partition ---------------------------------------------------------


def test_rows_from_two_partitions_are_rejected():
    """Grouping across partitions silently pools tuning and held-out images into one median."""
    rows = synthetic_rows()
    rows.append({**rows[0], "image_id": 3, "source_partition": "test"})
    with pytest.raises(ValueError, match="partition"):
        summarize_decile_rows(rows, {})


def test_rows_that_disagree_with_the_declared_partition_are_rejected():
    with pytest.raises(ValueError, match="partition"):
        summarize_decile_rows(synthetic_rows(), {"source_partition": "test"})


def test_the_declared_partition_is_carried_into_the_summary():
    summary = summarize_decile_rows(synthetic_rows(), {"source_partition": "tuning", "k": 5})
    assert summary["run_metadata"] == {"source_partition": "tuning", "k": 5}


# --- spec:159, the deployable gate ------------------------------------------------------------


def test_an_under_covered_group_is_excluded_from_the_ranking_not_merely_annotated():
    summary = summarize_decile_rows(rows_with_incomplete_group(), {})
    incomplete = [
        candidate for candidate in summary["groups"]
        if candidate["confidence_bin"] == "decile_10_20"
    ]
    assert len(incomplete) == 1
    assert incomplete[0]["median_spearman"] == 1.0
    assert incomplete[0]["full_coverage"] is False
    assert ("dynamic", "decile_10_20", "filtered", PRIMARY_SCORE_SCOPE) not in ranked_keys(summary)
    # ... and the groups that *are* ranked all trend more weakly than the one that was cut,
    # so the exclusion cannot be mistaken for the ranking simply preferring something else.
    assert all(entry["median_spearman"] < 1.0 for entry in summary[RANKED_GROUPS_KEY])


def test_a_group_missing_a_severity_outright_is_not_full_coverage():
    """Spec:189 expects six severities per image. A group whose rows for severity 5 were never
    produced has `scored_severity_count == total_severity_count` and is not full coverage:
    every score it holds is real, and none of them describes the blur level that matters most.
    """
    rows = [
        scored for scored in rows_with_incomplete_group()
        if not (scored["confidence_bin"] == "decile_10_20" and scored["severity"] == 5)
    ]
    summary = summarize_decile_rows(rows, {})
    truncated = [
        candidate for candidate in summary["groups"]
        if candidate["confidence_bin"] == "decile_10_20"
    ][0]
    assert truncated["scored_severity_count"] == truncated["total_severity_count"] == 10
    assert truncated["expected_severity_count"] == 12
    assert truncated["full_coverage"] is False
    assert ("dynamic", "decile_10_20", "filtered", PRIMARY_SCORE_SCOPE) not in ranked_keys(summary)


def test_the_ranking_holds_exactly_the_groups_that_survive_every_gate():
    """Four decoys, all of them stronger than the two survivors: the frozen diagnostic
    (spec:230), the under-covered group (spec:159), the unfiltered benchmark (spec:71) and the
    secondary decoder scope (spec:123)."""
    summary = summarize_decile_rows(rows_with_incomplete_group(), {})
    assert ranked_keys(summary) == [
        ("dynamic", "decile_00_10", "filtered", PRIMARY_SCORE_SCOPE),
        ("shared", "all_valid", "filtered", PRIMARY_SCORE_SCOPE),
    ]


def test_the_scored_image_clause_of_full_coverage_is_load_bearing():
    """The gate cannot lean on the loader's six-severity rule, because it does not re-check it.

    `scored == total` and `total == image_count * 6` constrain only the sums. A group holding
    eleven severities for one image and one severity for another satisfies both, and the second
    image has no trend at all -- it contributes nothing to any statistic the ranking sorts on
    and says so nowhere.
    """
    rows = [
        row(1, severity, "persistence", PRIMARY_SCORE_SCOPE, float(severity))
        for severity in range(11)
    ]
    rows.append(row(2, 0, "persistence", PRIMARY_SCORE_SCOPE, 0.0))
    summary = summarize_decile_rows(rows, {})
    lopsided = group(summary, "persistence", PRIMARY_SCORE_SCOPE)
    assert lopsided["scored_severity_count"] == lopsided["total_severity_count"] == 12
    assert lopsided["expected_severity_count"] == 12
    assert lopsided["image_count"] == 2 and lopsided["scored_image_count"] == 1
    assert lopsided["full_coverage"] is False
    assert summary[RANKED_GROUPS_KEY] == []


# --- spec:159, the ranking order ---------------------------------------------------------------


def test_ranking_orders_by_spearman_then_adjacent_then_violation():
    best = candidate(confidence_bin="decile_00_10", median_spearman=0.9)
    second = candidate(
        confidence_bin="decile_10_20", median_spearman=0.5,
        mean_adjacent_monotonicity=0.8, mean_violation_magnitude=0.2,
    )
    third = candidate(
        confidence_bin="decile_20_30", median_spearman=0.5,
        mean_adjacent_monotonicity=0.8, mean_violation_magnitude=0.4,
    )
    # The lowest violation magnitude of the four, and the lowest adjacent rate: it comes last
    # unless the tie-break order is wrong.
    last = candidate(
        confidence_bin="decile_30_40", median_spearman=0.5,
        mean_adjacent_monotonicity=0.4, mean_violation_magnitude=0.0,
    )
    ranked = rank_deployable_groups([third, last, best, second])
    assert [entry["confidence_bin"] for entry in ranked] == [
        "decile_00_10", "decile_10_20", "decile_20_30", "decile_30_40",
    ]


def test_ranking_is_deterministic_under_input_order():
    groups = [
        candidate(confidence_bin=name)
        for name in ("decile_00_10", "decile_10_20", "decile_20_30", "decile_30_40")
    ]
    expected = [entry["confidence_bin"] for entry in rank_deployable_groups(groups)]
    shuffler = random.Random(20260820)
    for _ in range(8):
        shuffled = list(groups)
        shuffler.shuffle(shuffled)
        assert [entry["confidence_bin"] for entry in rank_deployable_groups(shuffled)] == expected
    assert expected == ["decile_00_10", "decile_10_20", "decile_20_30", "decile_30_40"]


def test_a_group_whose_trend_is_unmeasurable_ranks_last_rather_than_raising():
    unmeasured = candidate(
        confidence_bin="decile_90_100", median_spearman=None,
        mean_adjacent_monotonicity=None, mean_violation_magnitude=None,
    )
    ranked = rank_deployable_groups([unmeasured, candidate()])
    assert [entry["confidence_bin"] for entry in ranked] == ["decile_00_10", "decile_90_100"]


# --- spec:230, dynamic and frozen ---------------------------------------------------------------


def test_frozen_groups_are_never_ranked_deployable():
    """Spec:230 -- a strong frozen-only result is diagnostic, not a deployable method."""
    summary = summarize_decile_rows(rows_with_incomplete_group(), {})
    frozen = [
        entry for entry in summary["groups"] if entry["membership_mode"] == "frozen"
    ]
    assert len(frozen) == 1 and frozen[0]["median_spearman"] == 1.0
    assert frozen[0]["full_coverage"] is True
    assert "frozen" not in {entry["membership_mode"] for entry in summary[RANKED_GROUPS_KEY]}
    assert "frozen" not in RANKABLE_MEMBERSHIP_MODES


def test_membership_mode_separates_two_groups_rather_than_pooling_them():
    """Spec:230 -- the two must not be combined into one score.

    Dropping `membership_mode` from `GROUP_KEYS` does not fail: it produces one group whose
    median is a blend of a rising frozen curve and a falling dynamic one, and the blend is a
    perfectly ordinary number.
    """
    rows = []
    for severity in range(6):
        rows.append(row(1, severity, "persistence", PRIMARY_SCORE_SCOPE, severity))
        rows.append(row(
            1, severity, "persistence", PRIMARY_SCORE_SCOPE, -severity, membership_mode="frozen",
        ))
    summary = summarize_decile_rows(rows, {})
    medians = {
        entry["membership_mode"]: entry["median_spearman"] for entry in summary["groups"]
    }
    assert medians == {"dynamic": 1.0, "frozen": -1.0}
    assert "membership_mode" in GROUP_KEYS


# --- spec:157, matched pairs and no raw magnitudes -------------------------------------------


def test_the_win_rate_is_computed_over_images_present_in_both_signals():
    """A marginal win rate would count an image that only one of the two signals scored."""
    rows = synthetic_rows()
    # Image 3 is scored by the confidence control alone, and out-trends nothing.
    for severity in range(6):
        rows.append(row(3, severity, "confidence", "confidence", -float(severity)))
    summary = summarize_decile_rows(rows, {})
    comparison = summary["comparisons"][0]
    assert comparison["paired_image_count"] == 2
    assert comparison["persistence_image_count"] == 2
    assert comparison["confidence_image_count"] == 3
    assert comparison["persistence_image_win_rate"] == 0.5


def test_a_comparison_pairs_only_within_one_membership_and_one_summary():
    """The control has to be the same selection under the same summary, not any confidence row."""
    rows = synthetic_rows()
    for severity in range(6):
        rows.append(row(
            1, severity, "confidence", "confidence", -float(severity), membership_mode="frozen",
        ))
        rows.append(row(1, severity, "confidence", "confidence", -float(severity), aggregation="mean"))
    summary = summarize_decile_rows(rows, {})
    comparisons = {
        (entry["membership_mode"], entry["aggregation"]): entry
        for entry in summary["comparisons"]
    }
    # Only the dynamic/q90 persistence group has a control; the two new confidence groups have
    # no persistence partner and so produce no comparison of their own.
    assert set(comparisons) == {("dynamic", "q90")}
    assert comparisons[("dynamic", "q90")]["confidence_median_spearman"] == 0.5


def test_a_comparison_never_crosses_the_padding_mode():
    """Benchmark 5 runs the bottom bin filtered *and* unfiltered (spec:135); each has its own
    confidence control, and comparing the unfiltered persistence result against the filtered
    control would summarise two different query populations against each other."""
    rows = []
    for severity in range(6):
        rows.append(row(1, severity, "persistence", PRIMARY_SCORE_SCOPE, float(severity)))
        rows.append(row(1, severity, "confidence", "confidence", float(severity)))
        rows.append(row(
            1, severity, "persistence", PRIMARY_SCORE_SCOPE, float(severity),
            padding_mode="unfiltered",
        ))
        rows.append(row(
            1, severity, "confidence", "confidence", -float(severity), padding_mode="unfiltered",
        ))
    by_padding = {
        entry["padding_mode"]: entry
        for entry in summarize_decile_rows(rows, {})["comparisons"]
    }
    assert set(by_padding) == {"filtered", "unfiltered"}
    assert by_padding["filtered"]["confidence_median_spearman"] == 1.0
    assert by_padding["unfiltered"]["confidence_median_spearman"] == -1.0


def test_a_missing_median_makes_the_difference_missing_rather_than_zero():
    """A difference of `0.0` is a claim -- that persistence exactly matched its control -- and
    it is the wrong one when one side of the pair has no measurable trend at all."""
    rows = []
    for severity in range(6):
        rows.append(row(
            1, severity, "persistence", PRIMARY_SCORE_SCOPE,
            0.0 if severity == 0 else float("nan"),
        ))
        rows.append(row(1, severity, "confidence", "confidence", float(severity)))
    comparison = summarize_decile_rows(rows, {})["comparisons"][0]
    assert comparison["persistence_median_spearman"] is None
    assert comparison["confidence_median_spearman"] == 1.0
    assert comparison["persistence_minus_confidence_spearman"] is None
    assert comparison["persistence_image_win_rate"] is None
    assert comparison["paired_image_count"] == 0


def test_a_persistence_group_with_no_confidence_control_is_counted_not_silently_dropped():
    """The all-300-query benchmark carries no confidence control by construction (spec:131)."""
    rows = [scored for scored in synthetic_rows() if scored["signal"] == "persistence"]
    summary = summarize_decile_rows(rows, {})
    assert summary["comparisons"] == []
    assert summary["diagnostics"]["persistence_groups_without_confidence_control"] == 1


def test_no_published_number_moves_under_a_positive_affine_rescale_of_the_scores():
    """Spec:157 -- scale-dependent raw score magnitudes must not be compared directly.

    Every statistic this module publishes is invariant under `score -> a * score + b` for
    `a > 0`. A mean or median raw score, an un-normalised violation magnitude, or any
    difference of scores across two scopes would move here and be caught.
    """
    rescaled = [
        {**scored, "score": 3.0 * scored["score"] + 10.0} if scored["signal"] == "persistence"
        else dict(scored)
        for scored in dipping_rows()
    ]
    assert summarize_decile_rows(rescaled, {}) == summarize_decile_rows(dipping_rows(), {})


# --- spec:147-155, the supporting metrics ------------------------------------------------------


@pytest.mark.parametrize("metric", [
    "median_spearman",
    "mean_adjacent_monotonicity",
    "mean_violation_magnitude",
    "endpoint_increase_rate",
    "image_count",
    "scored_image_count",
    "scored_severity_count",
    "total_severity_count",
    "median_selected_count_by_severity",
    "mean_clean_overlap_from_severity_1",
    "mean_clean_overlap_by_severity",
])
def test_every_group_publishes_the_supporting_metrics_the_spec_names(metric):
    summary = summarize_decile_rows(synthetic_rows(), {})
    assert all(metric in entry for entry in summary["groups"])


def test_the_primary_metric_is_the_median_and_not_the_mean_per_image_spearman():
    """Spec:145 says median. Two images that rose and one that fell: the median is 1.0 and the
    mean is 0.33, and only the median is robust to the one image whose selection collapsed."""
    rows = []
    for image_id, direction in ((1, 1.0), (2, 1.0), (3, -1.0)):
        for severity in range(6):
            rows.append(row(
                image_id, severity, "persistence", PRIMARY_SCORE_SCOPE, direction * severity,
            ))
    persistence = group(
        summarize_decile_rows(rows, {}), "persistence", PRIMARY_SCORE_SCOPE
    )
    assert persistence["median_spearman"] == 1.0
    assert persistence["scored_image_count"] == 3


def test_the_supporting_metrics_carry_the_values_they_claim():
    summary = summarize_decile_rows(dipping_rows(), {})
    persistence = group(summary, "persistence", PRIMARY_SCORE_SCOPE)
    assert persistence["mean_adjacent_monotonicity"] == pytest.approx(0.8)
    # One down step of 0.5 over a range of 5.0.
    assert persistence["mean_violation_magnitude"] == pytest.approx(0.1)
    # Maximum blur above clean: 5.0 > 0.0.
    assert persistence["endpoint_increase_rate"] == 1.0
    assert persistence["median_selected_count_by_severity"] == {str(k): 30.0 for k in range(6)}
    assert persistence["mean_clean_overlap_from_severity_1"] == 1.0
    assert persistence["mean_clean_overlap_by_severity"] == {str(k): 1.0 for k in range(6)}


def test_the_endpoint_rate_counts_only_the_images_with_a_measurable_trend():
    """`monotonicity_metrics` returns a definite `False` for `endpoint_increase` on an image it
    could not measure, which a plain mean averages in and publishes as "this image did not
    rise". The other three statistics come back `nan` there and drop out on their own; this one
    does not, so all four have to be taken over the same selected images.
    """
    rows = []
    for severity in range(6):
        rows.append(row(1, severity, "persistence", PRIMARY_SCORE_SCOPE, float(severity)))
        rows.append(row(
            2, severity, "persistence", PRIMARY_SCORE_SCOPE,
            0.0 if severity == 0 else float("nan"),
        ))
    persistence = group(summarize_decile_rows(rows, {}), "persistence", PRIMARY_SCORE_SCOPE)
    assert persistence["image_count"] == 2
    assert persistence["scored_image_count"] == 1
    assert persistence["endpoint_increase_rate"] == 1.0
    assert persistence["median_spearman"] == 1.0


def test_a_non_finite_supporting_metric_is_published_as_null():
    """`json.dumps(..., allow_nan=False)` refuses `nan`, and with `allow_nan=True` it emits a
    bare `NaN` token that is not valid JSON. The conversion is the backstop for every statistic
    that does not go through `_finite_mean` -- the selected-count medians among them.
    """
    rows = [{**scored, "selected_count": float("nan")} for scored in synthetic_rows()]
    summary = summarize_decile_rows(rows, {})
    persistence = group(summary, "persistence", PRIMARY_SCORE_SCOPE)
    assert persistence["median_selected_count_by_severity"] == {str(k): None for k in range(6)}
    json.dumps(summary, allow_nan=False)


def test_clean_overlap_is_reported_per_severity_and_excludes_the_definitional_term():
    """Severity 0 is 1.0 by construction for every membership mode, so averaging it into the
    scalar adds a sixth of a point to every bin alike. On the pilot that turns nine dynamic
    bins sitting at ~0.06 -- barely above the 1/19 chance baseline -- into ~0.216, a number
    printed beside a ranked candidate that looks like evidence the membership was stable.
    """
    rows = [
        {**scored, "clean_overlap": 1.0 if scored["severity"] == 0 else 0.06}
        for scored in dipping_rows()
    ]
    summary = summarize_decile_rows(rows, {})
    persistence = group(summary, "persistence", PRIMARY_SCORE_SCOPE)
    assert persistence["mean_clean_overlap_by_severity"]["0"] == 1.0
    assert persistence["mean_clean_overlap_by_severity"]["5"] == pytest.approx(0.06)
    assert persistence["mean_clean_overlap_from_severity_1"] == pytest.approx(0.06)
    assert persistence["mean_clean_overlap_from_severity_1"] != pytest.approx((1.0 + 5 * 0.06) / 6)


def test_padding_sensitivity_pairs_the_same_selection_with_and_without_the_mask():
    """Spec:155 -- padding sensitivity, and spec:69's control is a no-op on unpadded images."""
    rows = list(dipping_rows())
    for severity in range(6):
        rows.append(row(
            1, severity, "persistence", PRIMARY_SCORE_SCOPE, -float(severity),
            padding_mode="unfiltered",
        ))
    summary = summarize_decile_rows(rows, {})
    sensitivity = [
        entry for entry in summary["padding_sensitivity"] if entry["signal"] == "persistence"
    ]
    assert len(sensitivity) == 1
    assert sensitivity[0]["filtered_median_spearman"] == pytest.approx(0.942857, abs=1e-5)
    assert sensitivity[0]["unfiltered_median_spearman"] == -1.0
    assert sensitivity[0]["unfiltered_minus_filtered_spearman"] == pytest.approx(
        sensitivity[0]["unfiltered_median_spearman"] - sensitivity[0]["filtered_median_spearman"]
    )
    assert sensitivity[0]["score_changed_image_count"] == 1
    assert sensitivity[0]["unfiltered_image_win_rate"] == 0.0
    assert sensitivity[0]["score_changed_image_win_rate"] == 0.0


def test_the_moved_set_comes_from_the_scores_and_not_from_the_per_image_spearman():
    """The two meanings of "the mask reached this image" diverge, and only one is the truth.

    Three images. Image 1 is reached and its trend changes. Image 2 is reached -- every one of
    its six scores is different -- but its Spearman lands on the same value, because a rank
    correlation over six severities takes only 35 distinct values. Image 3 is not reached at
    all. Reading the moved set off the Spearman calls image 2 untouched, which is exactly the
    error that reports 50 to 56 moved images on the pilot where the selections provably differ
    on all 66 that carry a padded tail.
    """
    curves = {
        1: (DIPPING_CURVE, (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)),
        2: ((0.0, 1.0, 2.0, 3.0, 4.0, 5.0), (0.0, 2.0, 4.0, 6.0, 8.0, 10.0)),
        3: ((0.0, 1.0, 2.0, 3.0, 4.0, 5.0), (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)),
    }
    rows = []
    for image_id, (filtered, unfiltered) in curves.items():
        for severity in range(6):
            rows.append(row(
                image_id, severity, "persistence", PRIMARY_SCORE_SCOPE, filtered[severity],
            ))
            rows.append(row(
                image_id, severity, "persistence", PRIMARY_SCORE_SCOPE, unfiltered[severity],
                padding_mode="unfiltered",
            ))
    sensitivity = summarize_decile_rows(rows, {})["padding_sensitivity"][0]
    assert sensitivity["paired_image_count"] == 3
    # Two images were reached; only one of them changed its Spearman.
    assert sensitivity["score_changed_image_count"] == 2
    assert sensitivity["score_changed_and_paired_image_count"] == 2
    # Image 3 is untouched and is counted as a loss by the unrestricted rate.
    assert sensitivity["unfiltered_image_win_rate"] == pytest.approx(1 / 3)
    # Restricted to the two the mask reached -- not to the one whose Spearman moved, which
    # would read 1.0.
    assert sensitivity["score_changed_image_win_rate"] == 0.5


def test_a_severity_the_mask_removed_entirely_counts_as_a_move():
    """A severity scored under one rule and absent under the other is the largest move the mask
    can make, not the absence of one. Treating a one-sided score as equal to nothing files it
    as untouched, and the image then never enters the restricted rate."""
    rows = []
    for severity in range(6):
        rows.append(row(1, severity, "persistence", PRIMARY_SCORE_SCOPE, float(severity)))
        if severity < 5:
            rows.append(row(
                1, severity, "persistence", PRIMARY_SCORE_SCOPE, float(severity),
                padding_mode="unfiltered",
            ))
    sensitivity = summarize_decile_rows(rows, {})["padding_sensitivity"][0]
    assert sensitivity["score_changed_image_count"] == 1


def test_the_padding_win_rate_is_reported_both_over_all_images_and_over_the_moved_ones():
    """Spec:69's control is a no-op on an image that had no padded tail to remove.

    Two images: the mask reaches the second and leaves the first untouched. Counting the
    untouched image as a loss halves the rate, which is exactly what happens on the pilot --
    184 of its 250 images carry no padding at all -- and it makes a control that helped on
    every image it could reach read as a control that helped on half of them.
    """
    rows = []
    for image_id in (1, 2):
        for severity in range(6):
            rows.append(row(
                image_id, severity, "persistence", PRIMARY_SCORE_SCOPE,
                float(severity) if image_id == 1 else DIPPING_CURVE[severity],
            ))
            rows.append(row(
                image_id, severity, "persistence", PRIMARY_SCORE_SCOPE, float(severity),
                padding_mode="unfiltered",
            ))
    summary = summarize_decile_rows(rows, {})
    sensitivity = summary["padding_sensitivity"][0]
    assert sensitivity["paired_image_count"] == 2
    assert sensitivity["score_changed_image_count"] == 1
    # Image 1 is untouched by the mask and is counted as a loss by the unrestricted rate.
    assert sensitivity["unfiltered_image_win_rate"] == 0.5
    assert sensitivity["score_changed_image_win_rate"] == 1.0


def test_a_padding_control_that_reached_no_image_reports_no_restricted_rate():
    rows = []
    for severity in range(6):
        rows.append(row(1, severity, "persistence", PRIMARY_SCORE_SCOPE, severity))
        rows.append(row(
            1, severity, "persistence", PRIMARY_SCORE_SCOPE, severity, padding_mode="unfiltered",
        ))
    sensitivity = summarize_decile_rows(rows, {})["padding_sensitivity"][0]
    assert sensitivity["score_changed_image_count"] == 0
    # Nothing to take a rate over: reported as missing rather than as a zero a reader would
    # take for a measured failure.
    assert sensitivity["score_changed_image_win_rate"] is None
    assert sensitivity["unfiltered_image_win_rate"] == 0.0


def test_two_unscored_severities_are_not_a_move():
    """A severity nobody could score under either rule is the mask making no difference, not a
    difference nobody can measure. `nan != nan` would file it as a move on every such image."""
    rows = []
    for severity in range(6):
        score = float("nan") if severity == 5 else float(severity)
        rows.append(row(1, severity, "persistence", PRIMARY_SCORE_SCOPE, score))
        rows.append(row(
            1, severity, "persistence", PRIMARY_SCORE_SCOPE, score, padding_mode="unfiltered",
        ))
    sensitivity = summarize_decile_rows(rows, {})["padding_sensitivity"][0]
    assert sensitivity["score_changed_image_count"] == 0


def test_padding_counts_are_rolled_up_from_the_producers_diagnostics():
    diagnostics = {"images": {
        "7888": {
            "union_padded_count": 170,
            "padded_count_by_severity": {"0": 159, "1": 165, "2": 170, "3": 63, "4": 8, "5": 79},
            "tail_identical_across_severities": False,
        },
        "11": {
            "union_padded_count": 0,
            "padded_count_by_severity": {str(key): 0 for key in range(6)},
            "tail_identical_across_severities": True,
        },
    }}
    summary = summarize_decile_rows(synthetic_rows(), {}, diagnostics)
    padding = summary["padding"]
    assert padding["image_count"] == 2
    assert padding["images_with_padding"] == 1
    assert padding["max_union_padded_count"] == 170
    assert padding["images"]["7888"]["padded_count_by_severity"]["2"] == 170
    # An image with no padding has six identical *empty* tails, so it counts as identical in
    # the unrestricted number. Reporting only that one turns the pilot's "all 66 padded images
    # move their tail" into a reassuring 184 of 250.
    assert padding["images_with_identical_tails"] == 1
    assert padding["padded_images_with_identical_tails"] == 0


def test_a_summary_without_diagnostics_still_reports_padding_as_unmeasured():
    summary = summarize_decile_rows(synthetic_rows(), {})
    assert summary["padding"]["image_count"] == 0
    assert summary["padding"]["max_union_padded_count"] is None


# --- spec:229, the candidate against the published benchmark ------------------------------


def test_the_benchmark_selection_is_the_producers_and_not_three_literals():
    """A drift guard. If `ALL_QUERY_BENCHMARK`'s encoding ever changes, the comparison follows
    it instead of silently finding no benchmark and publishing an empty list."""
    assert BENCHMARK_SELECTION == (
        "persistence", ALL_QUERY_BENCHMARK[0], ALL_QUERY_BENCHMARK[1],
        PRIMARY_SCORE_SCOPE, ALL_QUERY_BENCHMARK[2],
    )


def test_every_ranked_candidate_is_compared_against_the_all_query_benchmark():
    rising = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    rows = candidate_rows({1: rising, 2: rising}) + benchmark_rows({1: rising, 2: rising})
    summary = summarize_decile_rows(rows, {})
    assert len(summary[RANKED_GROUPS_KEY]) == 1
    assert summary["diagnostics"]["benchmark_comparison_count"] == 1
    comparison = summary["benchmark_comparisons"][0]
    assert comparison["membership_mode"] == "dynamic"
    assert comparison["benchmark_aggregation"] == "q90"
    assert comparison["paired_image_count"] == 2
    assert comparison["candidate_minus_benchmark_spearman"] == 0.0
    assert comparison["candidate_image_tie_rate"] == 1.0
    assert comparison["candidate_image_win_rate"] == 0.0


def test_the_benchmark_win_rate_is_paired_and_separates_ties_from_losses():
    """A difference of two medians cannot say how many images it rests on, and that is the whole
    question when the margin is one step of a six-point Spearman.

    Here the candidate's median beats the benchmark's by a full point while winning on exactly
    one of the two images and tying on the other. Folding ties into the loss rate would report
    a 50 percent win as though the other half were a defeat.
    """
    rising = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    falling = (5.0, 4.0, 3.0, 2.0, 1.0, 0.0)
    rows = (
        candidate_rows({1: rising, 2: rising})
        + benchmark_rows({1: rising, 2: falling})
    )
    comparison = summarize_decile_rows(rows, {})["benchmark_comparisons"][0]
    assert comparison["candidate_median_spearman"] == 1.0
    assert comparison["benchmark_median_spearman"] == 0.0
    assert comparison["candidate_minus_benchmark_spearman"] == 1.0
    assert comparison["candidate_image_win_rate"] == 0.5
    assert comparison["candidate_image_tie_rate"] == 0.5
    assert comparison["candidate_image_loss_rate"] == 0.0
    rates = (
        comparison["candidate_image_win_rate"]
        + comparison["candidate_image_tie_rate"]
        + comparison["candidate_image_loss_rate"]
    )
    assert rates == 1.0


def test_the_benchmark_win_rate_uses_only_images_both_sides_scored():
    rising = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    falling = (5.0, 4.0, 3.0, 2.0, 1.0, 0.0)
    rows = (
        candidate_rows({1: rising, 2: rising})
        + benchmark_rows({1: falling, 3: falling})
    )
    comparison = summarize_decile_rows(rows, {})["benchmark_comparisons"][0]
    assert comparison["paired_image_count"] == 1
    assert comparison["candidate_image_win_rate"] == 1.0


def test_a_table_with_no_benchmark_publishes_no_benchmark_comparison():
    """`synthetic_rows()` has no all-300-query row, and an absent bar is reported as absent
    rather than as a candidate that failed to clear it."""
    summary = summarize_decile_rows(synthetic_rows(), {})
    assert summary["benchmark_comparisons"] == []
    assert summary["diagnostics"]["benchmark_comparison_count"] == 0


def test_the_benchmark_is_never_itself_a_ranked_candidate():
    rising = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    rows = candidate_rows({1: rising, 2: rising}) + benchmark_rows({1: rising, 2: rising})
    summary = summarize_decile_rows(rows, {})
    assert ALL_QUERY_BENCHMARK[2] not in {
        entry["padding_mode"] for entry in summary[RANKED_GROUPS_KEY]
    }


def test_a_paired_rate_states_both_denominators_because_ties_invert_it():
    """A rate that counts ties as non-wins is not the sign test a reader thinks it is.

    Five images: the candidate wins two, ties two, loses one. Over all five that is 0.400,
    which reads as "it loses the per-image majority"; over the three the comparison actually
    decided it is 0.667, which is a clear win. The medians are identical, so the primary metric
    calls it a dead heat and neither rate alone tells the reader what happened. This is the real
    pilot's `decile_50_60 q90` in miniature.
    """
    rising = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    falling = (5.0, 4.0, 3.0, 2.0, 1.0, 0.0)
    rows = (
        candidate_rows({1: rising, 2: rising, 3: rising, 4: rising, 5: falling})
        + benchmark_rows({1: falling, 2: falling, 3: rising, 4: rising, 5: rising})
    )
    comparison = summarize_decile_rows(rows, {})["benchmark_comparisons"][0]
    assert comparison["candidate_minus_benchmark_spearman"] == 0.0
    assert comparison["paired_image_count"] == 5
    assert comparison["candidate_image_win_rate"] == pytest.approx(0.4)
    assert comparison["candidate_image_tie_rate"] == pytest.approx(0.4)
    assert comparison["candidate_image_loss_rate"] == pytest.approx(0.2)
    assert comparison["candidate_image_decided_image_count"] == 3
    assert comparison["candidate_image_decided_win_rate"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("family", ["comparisons", "padding_sensitivity", "benchmark_comparisons"])
def test_every_paired_rate_publishes_its_decided_denominator(family):
    """One helper serves all three comparison families, so none of them can lose the second
    denominator on its own."""
    rising = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    # `synthetic_rows()` already supplies the persistence/confidence pair *and* a group the
    # ranking admits; the benchmark and the unfiltered twin complete the other two families.
    rows = synthetic_rows() + benchmark_rows({1: rising, 2: rising})
    for image_id in (1, 2):
        for severity in range(6):
            rows.append(row(
                image_id, severity, "persistence", PRIMARY_SCORE_SCOPE, float(severity),
                padding_mode="unfiltered",
            ))
    summary = summarize_decile_rows(rows, {})
    prefixes = {
        "comparisons": "persistence_image",
        "padding_sensitivity": "unfiltered_image",
        "benchmark_comparisons": "candidate_image",
    }
    prefix = prefixes[family]
    assert summary[family]
    for entry in summary[family]:
        assert f"{prefix}_win_rate" in entry
        assert f"{prefix}_tie_rate" in entry
        assert f"{prefix}_loss_rate" in entry
        assert f"{prefix}_decided_image_count" in entry
        assert f"{prefix}_decided_win_rate" in entry
        rates = [entry[f"{prefix}_{name}_rate"] for name in ("win", "tie", "loss")]
        if all(rate is not None for rate in rates):
            assert sum(rates) == pytest.approx(1.0)


def test_the_score_changed_count_depends_on_the_summary_and_so_is_not_the_reached_set():
    """The honest limit of `score_changed_image_count`, pinned rather than described.

    One selection pair, two scene summaries. Under `mean` the mask changed image 1's score;
    under `q90` the same changed selection produced the identical score at every severity,
    because a summary is a many-to-one map. A set of "images the mask reached" cannot depend on
    which summary was applied afterwards, so this statistic is a lower bound on that set -- 27
    of the pilot's 34 sensitivity pairs recover the selection-derived count and 7 understate it.
    """
    rows = []
    for severity in range(6):
        for aggregation, unfiltered in (("mean", 2.0 * severity), ("q90", float(severity))):
            rows.append(row(
                1, severity, "persistence", PRIMARY_SCORE_SCOPE, float(severity),
                aggregation=aggregation,
            ))
            rows.append(row(
                1, severity, "persistence", PRIMARY_SCORE_SCOPE, unfiltered,
                aggregation=aggregation, padding_mode="unfiltered",
            ))
    counts = {
        entry["aggregation"]: entry["score_changed_image_count"]
        for entry in summarize_decile_rows(rows, {})["padding_sensitivity"]
    }
    assert counts == {"mean": 1, "q90": 0}


# --- spec:97, which 1.000 is a measurement --------------------------------------------------


def test_the_clean_overlap_scalar_is_marked_definitional_for_frozen_and_shared():
    """`clean_overlap` is 1.0 by construction for `frozen` and `shared`, and three of the pilot's
    33 ranked candidates are `shared` -- so an unflagged 1.000 sits beside dynamic bins at 0.059
    in the table Task 7 renders, reading as the most stable membership in the experiment."""
    rows = []
    for severity in range(6):
        rows.append(row(
            1, severity, "persistence", PRIMARY_SCORE_SCOPE, float(severity),
            clean_overlap=1.0 if severity == 0 else 0.06,
        ))
        rows.append(row(
            1, severity, "persistence", PRIMARY_SCORE_SCOPE, float(severity),
            membership_mode="frozen", clean_overlap=1.0,
        ))
        rows.append(row(
            1, severity, "persistence", PRIMARY_SCORE_SCOPE, float(severity),
            membership_mode="shared", confidence_bin="all_valid", clean_overlap=1.0,
        ))
    summary = summarize_decile_rows(rows, {})
    flags = {
        entry["membership_mode"]: entry["clean_overlap_is_definitional"]
        for entry in summary["groups"]
    }
    assert flags == {"dynamic": False, "frozen": True, "shared": True}
    scalars = {
        entry["membership_mode"]: entry["mean_clean_overlap_from_severity_1"]
        for entry in summary["groups"]
    }
    assert scalars["dynamic"] == pytest.approx(0.06)
    assert scalars["shared"] == 1.0
    # The flag has to survive into the ranked table, which is where the two print side by side.
    assert all("clean_overlap_is_definitional" in entry for entry in summary[RANKED_GROUPS_KEY])


# --- JSON safety --------------------------------------------------------------------------------


def test_the_summary_serialises_without_a_non_finite_number():
    """Spec's `summary.json`, and Task 7's `allow_nan=False`: `NaN` is not valid JSON."""
    rows = [
        {**scored, "score": float("nan") if scored["severity"] > 0 else scored["score"]}
        for scored in synthetic_rows()
    ]
    summary = summarize_decile_rows(rows, {"source_partition": "tuning"})
    text = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
    assert "NaN" not in text
    assert group(summary, "persistence", PRIMARY_SCORE_SCOPE)["median_spearman"] is None


def test_no_numpy_scalar_survives_into_the_summary():
    summary = summarize_decile_rows(synthetic_rows(), {})
    stack = [summary]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.keys())
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
        else:
            assert isinstance(value, (str, int, float, bool, type(None))), type(value)
            assert not isinstance(value, np.generic)
            if isinstance(value, float):
                assert math.isfinite(value)


# --- refusals on a malformed table ------------------------------------------------------------


def test_an_empty_table_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        summarize_decile_rows([], {})


@pytest.mark.parametrize("field", [*ROW_KEYS, "score", "clean_overlap", "selected_count"])
def test_a_row_key_the_summary_needs_cannot_be_absent(field):
    rows = [{key: value for key, value in scored.items() if key != field} for scored in synthetic_rows()]
    with pytest.raises(ValueError, match=field):
        summarize_decile_rows(rows, {})


# --- no kNN, no inference -----------------------------------------------------------------------


def test_the_summariser_cannot_reach_a_knn_or_extraction_path():
    """The design's non-goal, carried into the reporting layer with the loader's own test."""
    source = Path(reporting_module.__file__).read_text()
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    forbidden = {"knn", "evaluate", "extractor", "bank", "dataset", "pipeline"}
    assert not {name for name in imported if name.lstrip(".").split(".")[0] in forbidden}


# --- the writer's fixture ------------------------------------------------------------------
#
# `full_grid_rows` is a miniature of exactly what `analyze_deciles` emits: ten dynamic bins
# and ten frozen bins filtered, `all_valid` filtered, the lowest bin repeated unfiltered under
# both memberships, and the all-300-query benchmark at `q90` with no confidence control. Two
# images, six severities.
#
# Every per-image curve is a permutation of `range(6)` built by swapping position 0 with
# position `d`, which fixes the per-image Spearman at exactly `1 - 2*d**2/35`. That is the
# whole reason for the shape: the fixture's medians are known in closed form, so a test can
# name the winner rather than discovering it, and the four qualitative facts the real pilot
# shows are reproduced deliberately rather than by luck --
#
#   * a single mid-confidence bin wins (`decile_50_60`, at 1.0);
#   * persistence beats its confidence control there and *loses* to it at the bottom bin;
#   * the bottom bin is worthless dynamic (-0.4286) and strong frozen (+1.0);
#   * removing padding *lowers* the every-query trend (0.9429 unfiltered -> 0.7714 filtered).

SPEARMAN_FOR_DISTANCE = {distance: 1.0 - 2 * distance**2 / 35 for distance in range(6)}


def swap_curve(distance: int) -> tuple[float, ...]:
    """`range(6)` with positions 0 and `distance` exchanged -- Spearman `1 - 2d^2/35` exactly."""
    values = list(range(6))
    values[0], values[distance] = values[distance], values[0]
    return tuple(float(value) for value in values)


def _distance(image_id: int, index: int, kind: str) -> int:
    """The swap distance for one bin, per image, per selection family."""
    if kind == "dynamic_persistence":
        return abs(index - 5) + (1 if image_id == 2 and index > 5 else 0)
    if kind == "dynamic_confidence":
        return min(5, 5 - abs(index - 5) + (1 if image_id == 2 and index < 5 else 0))
    if kind == "frozen_persistence":
        return 0 if index == 0 else min(5, abs(index - 5) + 2)
    if kind == "frozen_confidence":
        return min(5, 5 - abs(index - 5) + 1)
    raise AssertionError(kind)


GRID_RUN_METADATA = {
    "artifact_type": "confidence_decile_scene_uncertainty",
    "feature_cache_id": "cache-abc",
    "source_result_id": "result-def",
    "bank_id": "bank-ghi",
    "normalization": "none",
    "k": 5,
    "source_partition": "tuning",
    "severities": [0, 1, 2, 3, 4, 5],
    "decoder_layers": [0, 1, 2],
    "query_count": 300,
    "image_count": 2,
    "record_count": 12,
}


def grid_diagnostics() -> dict:
    return {
        "images": {
            "1": {
                "union_padded_query_ids": [296, 297, 298, 299],
                "union_padded_count": 4,
                "padded_query_ids_by_severity": {str(severity): [298, 299] for severity in range(6)},
                "padded_count_by_severity": {str(severity): 2 for severity in range(6)},
                "tail_identical_across_severities": False,
            },
            "2": {
                "union_padded_query_ids": [],
                "union_padded_count": 0,
                "padded_query_ids_by_severity": {str(severity): [] for severity in range(6)},
                "padded_count_by_severity": {str(severity): 0 for severity in range(6)},
                "tail_identical_across_severities": True,
            },
        }
    }


def _grid_rows_for(image_id, severity, membership, confidence_bin, padding, overlap,
                   selected_count, persistence_curve, confidence_curve, scopes):
    emitted = []
    for aggregation in DECILE_AGGREGATIONS:
        for scope, curve in scopes:
            emitted.append(row(
                image_id, severity, "persistence", scope,
                (persistence_curve if curve is None else curve)[severity],
                membership_mode=membership, confidence_bin=confidence_bin,
                padding_mode=padding, aggregation=aggregation, clean_overlap=overlap,
                selected_count=selected_count,
            ))
        if confidence_curve is not None:
            emitted.append(row(
                image_id, severity, "confidence", CONFIDENCE_SCOPE,
                0.1 * confidence_curve[severity],
                membership_mode=membership, confidence_bin=confidence_bin,
                padding_mode=padding, aggregation=aggregation, clean_overlap=overlap,
                selected_count=selected_count,
            ))
    return emitted


def full_grid_rows():
    rows = []
    for image_id in (1, 2):
        for severity in range(6):
            for index, name in enumerate(DECILE_NAMES):
                # `layer_0` rises perfectly in every bin, so a ranking that forgot the
                # primary-scope gate would put a secondary diagnostic first (spec:123).
                scopes = ((PRIMARY_SCORE_SCOPE, None), ("layer_0", swap_curve(0)))
                dynamic_overlap = 1.0 if severity == 0 else (0.25 if index == 9 else 0.06)
                rows.extend(_grid_rows_for(
                    image_id, severity, "dynamic", name, "filtered", dynamic_overlap, 30,
                    swap_curve(_distance(image_id, index, "dynamic_persistence")),
                    swap_curve(_distance(image_id, index, "dynamic_confidence")),
                    scopes,
                ))
                rows.extend(_grid_rows_for(
                    image_id, severity, "frozen", name, "filtered", 1.0, 30,
                    swap_curve(_distance(image_id, index, "frozen_persistence")),
                    swap_curve(_distance(image_id, index, "frozen_confidence")),
                    scopes,
                ))
            rows.extend(_grid_rows_for(
                image_id, severity, *ALL_VALID_BENCHMARK[:2], ALL_VALID_BENCHMARK[2], 1.0, 290,
                # The confidence control over every valid query trends *down* with blur, as it
                # does on the real pilot; a report that reads a big positive difference without
                # looking at the control's own column would credit persistence for it.
                swap_curve(2), swap_curve(5), ((PRIMARY_SCORE_SCOPE, None),),
            ))
            rows.extend(_grid_rows_for(
                image_id, severity, "dynamic", SENSITIVITY_BIN, "unfiltered",
                1.0 if severity == 0 else 0.07, 33,
                swap_curve(3), swap_curve(2), ((PRIMARY_SCORE_SCOPE, None),),
            ))
            rows.extend(_grid_rows_for(
                image_id, severity, "frozen", SENSITIVITY_BIN, "unfiltered", 1.0, 33,
                swap_curve(1), swap_curve(3), ((PRIMARY_SCORE_SCOPE, None),),
            ))
            # Benchmark 1: q90 alone, persistence alone, all 300 queries (spec:131).
            rows.append(row(
                image_id, severity, "persistence", PRIMARY_SCORE_SCOPE,
                swap_curve(1)[severity],
                membership_mode=ALL_QUERY_BENCHMARK[0],
                confidence_bin=ALL_QUERY_BENCHMARK[1],
                padding_mode=ALL_QUERY_BENCHMARK[2],
                aggregation="q90", clean_overlap=1.0, selected_count=300,
            ))
    return rows


def grid_summary():
    return summarize_decile_rows(full_grid_rows(), GRID_RUN_METADATA, grid_diagnostics())


def written(tmp_path, rows=None, run_metadata=None, diagnostics=None):
    output = tmp_path / "report"
    write_decile_report(
        full_grid_rows() if rows is None else rows,
        output,
        GRID_RUN_METADATA if run_metadata is None else run_metadata,
        grid_diagnostics() if diagnostics is None else diagnostics,
    )
    return output


def section(report: str, heading: str) -> str:
    """The body of one `## ` section, so a test can assert *where* a sentence lives."""
    parts = re.split(r"^## ", report, flags=re.MULTILINE)
    for part in parts[1:]:
        if part.startswith(heading):
            return part
    raise AssertionError(f"no section {heading!r} in the report")


# --- the brief's artifact tests -------------------------------------------------------------


def test_write_report_creates_every_declared_artifact(tmp_path):
    write_decile_report(
        synthetic_rows(), tmp_path,
        run_metadata={"source_partition": "tuning"}, diagnostics={"images": {}},
    )
    expected = {
        "per_scene.csv", "summary.json", "confidence_decile_heatmap.png",
        "blur_curves.png", "dynamic_vs_frozen.png", "padding_sensitivity.png",
        "easy-report.md",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    text = (tmp_path / "summary.json").read_text()
    assert "NaN" not in text
    assert json.loads(text)["run_metadata"]["source_partition"] == "tuning"


def test_easy_report_names_winner_controls_and_test_status(tmp_path):
    write_decile_report(synthetic_rows(), tmp_path, {"source_partition": "tuning"}, {})
    report = (tmp_path / "easy-report.md").read_text()
    assert "Best confidence range" in report
    assert "confidence alone" in report
    assert "all-query benchmark" in report
    assert "held-out test images were not used" in report


# --- spec:163-165, `per_scene.csv` -----------------------------------------------------------


def test_the_csv_is_exactly_the_summary_frame(tmp_path):
    """Kills a second frame built by hand: it would drift from the summarised table silently."""
    output = written(tmp_path)
    written_back = pd.read_csv(output / "per_scene.csv", float_precision="round_trip")
    expected = summary_frame(full_grid_rows())
    assert list(written_back.columns) == list(expected.columns)
    pd.testing.assert_frame_equal(written_back, expected, check_dtype=False)


def test_the_csv_round_trips_every_score_exactly(tmp_path):
    """Kills a writer that truncates floats: an adjacent-step comparison is a strict `>=`.

    `float_precision="round_trip"` is on the *reader*, as in `reporting.read_result_csv`;
    what this pins is that the writer emitted enough digits for that reader to recover the
    value it was handed.
    """
    output = written(tmp_path)
    written_back = pd.read_csv(output / "per_scene.csv", float_precision="round_trip")
    assert written_back["score"].tolist() == summary_frame(full_grid_rows())["score"].tolist()


def test_the_csv_never_carries_the_excluded_column(tmp_path):
    """Spec:165 wants one row per selection, not ~280 MB of quoted query-id lists beside it."""
    output = written(tmp_path)
    header = (output / "per_scene.csv").read_text().splitlines()[0].split(",")
    assert set(ROW_KEYS_EXCLUDED_FROM_CSV).isdisjoint(header)
    assert set(ROW_KEYS) <= set(header)


def test_the_writer_never_reads_the_excluded_column(tmp_path):
    """Kills `pd.DataFrame(rows).drop(columns=[...])` in the writer as well as the summariser.

    That form materialises the column before discarding it, and raises `KeyError` on a table
    that legitimately never carried it. This table never carried it.
    """
    stripped = [
        {key: value for key, value in scored.items() if key not in ROW_KEYS_EXCLUDED_FROM_CSV}
        for scored in full_grid_rows()
    ]
    output = written(tmp_path, rows=stripped)
    assert (output / "per_scene.csv").exists()


def test_the_csv_holds_one_row_per_output_result_key(tmp_path):
    """Spec:165 -- one row per image, severity, signal, membership, bin, summary and scope."""
    output = written(tmp_path)
    written_back = pd.read_csv(output / "per_scene.csv", float_precision="round_trip")
    assert not written_back.duplicated(subset=list(ROW_KEYS)).any()
    assert len(written_back) == len(full_grid_rows())


# --- spec:166, `summary.json` ----------------------------------------------------------------


def test_the_summary_json_carries_the_whole_provenance_and_not_only_the_partition(tmp_path):
    """Spec:166 asks for provenance and the exact configuration, not one field of it."""
    output = written(tmp_path)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["run_metadata"] == GRID_RUN_METADATA
    assert summary["diagnostics"]["row_count"] == len(full_grid_rows())
    assert summary["padding"]["images_with_padding"] == 1
    assert summary[RANKED_GROUPS_KEY]


def test_the_summary_json_is_the_summariser_output_unchanged(tmp_path):
    output = written(tmp_path)
    assert json.loads((output / "summary.json").read_text()) == grid_summary()


def test_a_non_finite_number_cannot_reach_the_summary_file(tmp_path):
    """`allow_nan=False` -- a bare `NaN` token is not JSON and half the world parses it as text."""
    rows = [
        {**scored, "score": float("nan")} if scored["severity"] == 5 else scored
        for scored in full_grid_rows()
    ]
    output = written(tmp_path, rows=rows)
    assert "NaN" not in (output / "summary.json").read_text()


# --- writing is atomic, and refuses before it writes ------------------------------------------


def test_a_malformed_table_is_refused_before_any_artifact_is_written(tmp_path):
    rows = full_grid_rows()
    output = tmp_path / "report"
    with pytest.raises(ValueError, match="duplicate"):
        write_decile_report(rows + [rows[0]], output, GRID_RUN_METADATA, grid_diagnostics())
    assert not output.exists()


def test_no_temporary_file_survives_a_completed_write(tmp_path):
    """Kills a writer that saves straight to the final path: a crash mid-write would leave a
    truncated PNG or a half-serialised JSON that reads as a finished report."""
    output = written(tmp_path)
    assert [path.name for path in output.iterdir() if path.name.endswith(".tmp")] == []


def test_rewriting_the_same_directory_replaces_rather_than_accumulates(tmp_path):
    output = written(tmp_path)
    before = sorted(path.name for path in output.iterdir())
    write_decile_report(full_grid_rows(), output, GRID_RUN_METADATA, grid_diagnostics())
    assert sorted(path.name for path in output.iterdir()) == before


# --- spec:167-170, the four figures ------------------------------------------------------------


FIGURE_NAMES = (
    "confidence_decile_heatmap.png", "blur_curves.png",
    "dynamic_vs_frozen.png", "padding_sensitivity.png",
)


@pytest.mark.parametrize("name", FIGURE_NAMES)
def test_every_figure_is_a_real_non_empty_png(tmp_path, name):
    data = (written(tmp_path) / name).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(data) > 5_000


def test_the_backend_is_pinned_before_pyplot_is_imported():
    """There is no display on the report box, so the backend cannot be left to autodetection.

    Asserted on the source order rather than only on `get_backend()`: by the time a test runs,
    something else may already have pinned Agg, and this module has to pin it itself.
    """
    assert matplotlib.get_backend().lower() == "agg"
    source = Path(reporting_module.__file__).read_text()
    assert source.index('matplotlib.use("Agg")') < source.index("import matplotlib.pyplot")


def test_the_heatmap_keeps_all_ten_decile_columns_in_spec_order(tmp_path):
    """Kills alphabetical ordering, and kills dropping the bins with no rows.

    A heatmap drawn over only the bins that happened to exist reads as a complete ten-bin
    measurement. The axis carries all ten of `DECILE_NAMES` whatever the table holds; a bin
    with no row is a blank cell, which a reader can see.
    """
    figure = reporting_module._heatmap_figure(grid_summary())
    axis = figure.axes[0]
    assert [text.get_text() for text in axis.get_xticklabels()] == list(DECILE_NAMES)
    thin = summarize_decile_rows(synthetic_rows(), {"source_partition": "tuning"})
    assert [
        text.get_text() for text in reporting_module._heatmap_figure(thin).axes[0].get_xticklabels()
    ] == list(DECILE_NAMES)


def test_the_heatmap_has_one_row_per_signal(tmp_path):
    """Spec:167 -- median Spearman by confidence bin *and signal*."""
    axis = reporting_module._heatmap_figure(grid_summary()).axes[0]
    labels = [text.get_text() for text in axis.get_yticklabels()]
    assert len(labels) == len(SIGNALS)
    assert any("persistence" in label for label in labels)
    assert any("confidence" in label for label in labels)


def test_the_blur_curves_put_the_two_signals_on_separate_axes(tmp_path):
    """Spec:157 -- persistence distance and `1 - confidence` share no unit.

    One axis carrying both would invite the reader to compare their magnitudes, which is the
    one comparison the design forbids. Two panels is the mutation this kills.
    """
    figure = reporting_module._blur_curve_figure(summary_frame(full_grid_rows()))
    assert len(figure.axes) == 2
    assert len({axis.get_ylabel() for axis in figure.axes}) == 2
    for axis in figure.axes:
        # Matplotlib prefixes unlabelled artists with `_`; the zero reference line is one.
        labels = [line.get_label() for line in axis.get_lines()]
        assert [label for label in labels if not label.startswith("_")] == list(DECILE_NAMES)


def test_the_dynamic_and_frozen_series_are_drawn_apart_and_never_summed(tmp_path):
    """Spec:230 -- they answer different questions and must not be combined into one score."""
    figure = reporting_module._dynamic_frozen_figure(grid_summary())
    axis = figure.axes[0]
    labels = [container.get_label() for container in axis.containers]
    assert labels == ["dynamic", "frozen"]
    summary = grid_summary()
    for container, mode in zip(axis.containers, ("dynamic", "frozen")):
        expected = [
            _grid_group(summary, mode, name)["median_spearman"] for name in DECILE_NAMES
        ]
        drawn = [patch.get_height() for patch in container]
        assert drawn == pytest.approx(expected, nan_ok=True)


def _grid_group(summary, membership, confidence_bin):
    return next(
        entry for entry in summary["groups"]
        if entry["signal"] == "persistence"
        and entry["score_scope"] == PRIMARY_SCORE_SCOPE
        and entry["aggregation"] == "q90"
        and entry["padding_mode"] == "filtered"
        and entry["membership_mode"] == membership
        and entry["confidence_bin"] == confidence_bin
    )


def test_the_dynamic_frozen_figure_shows_query_movement_beside_it(tmp_path):
    """Spec:169 is "query movement versus feature movement"; the overlap panel is the first half.

    Frozen and `shared` overlaps are 1.0 by construction, so plotting them beside a dynamic
    0.06 would print arithmetic as if it were stability. Only the dynamic bins are drawn.
    """
    figure = reporting_module._dynamic_frozen_figure(grid_summary())
    assert len(figure.axes) == 2
    overlap = figure.axes[1]
    assert [container.get_label() for container in overlap.containers] == ["dynamic"]
    drawn = [patch.get_height() for patch in overlap.containers[0]]
    assert drawn == pytest.approx([0.06] * 9 + [0.25])


def test_the_padding_figure_shows_both_padding_modes_for_the_lowest_bin(tmp_path):
    """Spec:170 -- filtered *and* unfiltered, lowest bin, both signals."""
    figure = reporting_module._padding_sensitivity_figure(grid_summary())
    axis = figure.axes[0]
    assert [container.get_label() for container in axis.containers] == ["filtered", "unfiltered"]
    labels = [text.get_text() for text in axis.get_xticklabels()]
    assert len(labels) == 4
    assert all("persistence" in label or "confidence" in label for label in labels)
    # The bin the control is scoped to is named on the figure itself (spec:69), because these
    # PNGs get pasted into write-ups on their own.
    assert SENSITIVITY_BIN in axis.get_title()


def test_a_figure_with_nothing_to_draw_says_so_instead_of_showing_an_empty_axis(tmp_path):
    """A blank axis reads as a measured zero. The mutation is drawing it anyway."""
    thin = summarize_decile_rows(synthetic_rows(), {"source_partition": "tuning"})
    figure = reporting_module._padding_sensitivity_figure(thin)
    axis = figure.axes[0]
    assert axis.containers == []
    assert any("no unfiltered" in text.get_text() for text in axis.texts)


# --- spec:171-172, the easy report --------------------------------------------------------------


def test_the_easy_report_leads_with_the_four_questions_in_spec_order(tmp_path):
    """Spec:172 fixes both the content of the opening section and the order inside it."""
    report = (written(tmp_path) / "easy-report.md").read_text()
    assert report.startswith("# Confidence-Decile Blur Experiment")
    assert re.search(r"^## ", report, flags=re.MULTILINE).start() == report.index("## Short answer")
    short = section(report, "Short answer")
    positions = [short.index(question) for question in SPEC_172_QUESTIONS]
    assert positions == sorted(positions)


def test_the_easy_report_carries_every_heading_the_brief_names(tmp_path):
    report = (written(tmp_path) / "easy-report.md").read_text()
    headings = re.findall(r"^## (.+)$", report, flags=re.MULTILINE)
    assert headings == [
        "Short answer", "Best confidence range", "Persistence versus confidence alone",
        "Dynamic versus frozen queries", "Effect of padded queries",
        "Metrics in plain language", "What this does not prove", "Next decision",
    ]


def test_the_easy_report_names_the_winner_the_ranking_chose(tmp_path):
    """Kills a report that recomputes its own winner, or that reads position -1."""
    output = written(tmp_path)
    report = (output / "easy-report.md").read_text()
    winner = grid_summary()[RANKED_GROUPS_KEY][0]
    assert winner["confidence_bin"] == "decile_50_60"
    best = section(report, "Best confidence range")
    assert winner["confidence_bin"] in best
    assert winner["membership_mode"] in best
    assert winner["aggregation"] in best
    assert f"{winner['median_spearman']:+.4f}" in best


PROBABILITY_WORDS = (
    "likelihood", "chance of", "odds of", "percent chance", "% chance", "how likely",
    "probability of", "probability that",
)


@pytest.mark.parametrize("word", PROBABILITY_WORDS)
def test_the_easy_report_never_calls_a_score_a_probability(tmp_path, word):
    """Spec:111 -- "A value such as 0.8 must not be described as an 80-percent probability of
    corruption." The transformation reverses direction and trains and calibrates nothing."""
    report = (written(tmp_path) / "easy-report.md").read_text().lower()
    assert word not in report


def test_every_use_of_the_word_probability_is_a_denial(tmp_path):
    """The bare word cannot be banned -- spec:111's disclaimer needs it -- so what is checked
    is that it never appears except in a negation. A sentence that slipped from denying the
    reading to offering it would keep the word and lose the "neither"."""
    report = (written(tmp_path) / "easy-report.md").read_text().lower()
    positions = [match.start() for match in re.finditer("probabilit", report)]
    assert positions
    for position in positions:
        assert "neither" in report[max(0, position - 60):position]


def test_the_easy_report_says_outright_that_neither_score_is_a_probability(tmp_path):
    report = (written(tmp_path) / "easy-report.md").read_text()
    assert "Neither score is a probability" in report
    assert "nothing here is trained or calibrated" in report


def test_the_easy_report_marks_frozen_as_diagnostic_and_never_recommends_it(tmp_path):
    """Spec:230 -- a strong frozen-only result is not a deployable method.

    The fixture makes this bite: the frozen bottom bin scores +1.0 against the dynamic bottom
    bin's -0.4286, so a report that ranked on the number alone would recommend it.
    """
    report = (written(tmp_path) / "easy-report.md").read_text()
    frozen = section(report, "Dynamic versus frozen queries")
    assert "diagnostic" in frozen
    assert "not combined" in frozen or "never combined" in frozen
    assert "no paired clean version" in frozen
    assert "frozen" not in section(report, "Short answer").split("Does padding")[0]


def test_the_easy_report_states_both_denominators_for_every_paired_rate(tmp_path):
    """A rate over all paired images counts ties as non-wins; a rate over the decided ones
    hides how much of the run the comparison could not separate. Either alone inverts the
    sentence a reader writes, so the report publishes both everywhere."""
    report = (written(tmp_path) / "easy-report.md").read_text()
    lines = [line for line in report.splitlines() if "win rate" in line]
    assert lines
    for line in lines:
        assert "decided" in line, line
    assert report.count("it decided") >= 2


def test_the_easy_report_calls_the_padding_count_a_lower_bound(tmp_path):
    """`score_changed_image_count` counts images whose *score* moved and varies with the scene
    summary, so it can never be labelled "images affected by padding"."""
    padding = section((written(tmp_path) / "easy-report.md").read_text(), "Effect of padded")
    assert "lower bound" in padding
    assert "affected by padding" not in padding


def test_the_easy_report_calls_a_definitional_overlap_arithmetic(tmp_path):
    """A `clean_overlap` of 1.000 on a `shared` or `frozen` row is arithmetic, not evidence."""
    report = (written(tmp_path) / "easy-report.md").read_text()
    assert "arithmetic" in report


def test_the_easy_report_says_the_ranking_did_not_choose_the_padding_rule(tmp_path):
    """Spec:224 asks the tuning run to select a padding rule; the ranking admits only
    `filtered` candidates, so it never chose one. That evidence is in the sensitivity table."""
    report = (written(tmp_path) / "easy-report.md").read_text()
    assert "did not choose the padding rule" in report


def test_the_easy_report_says_the_choice_was_made_on_the_same_images_it_reports(tmp_path):
    report = section((written(tmp_path) / "easy-report.md").read_text(), "What this does not")
    assert "best of" in report
    assert "not a hypothesis test" in report


def test_the_easy_report_ends_with_the_held_out_sentence(tmp_path):
    report = (written(tmp_path) / "easy-report.md").read_text()
    assert report.rstrip().endswith(EASY_REPORT_FINAL_SENTENCE)


def test_the_easy_report_reports_no_candidate_rather_than_naming_one(tmp_path):
    """Every group under-covered: the report must say there is no deployable candidate, not
    reach into an empty ranking or fall back to the best of the excluded groups."""
    rows = [
        scored for scored in full_grid_rows()
        if not (scored["image_id"] == 2 and scored["severity"] == 5)
    ]
    report = (written(tmp_path, rows=rows) / "easy-report.md").read_text()
    assert "No candidate" in report
    assert EASY_REPORT_FINAL_SENTENCE in report


def test_the_easy_report_is_deterministic(tmp_path):
    """Two runs over the same rows in different orders must produce the same sentences."""
    rows = full_grid_rows()
    first = (written(tmp_path / "a", rows=rows) / "easy-report.md").read_text()
    shuffled = list(rows)
    random.Random(7).shuffle(shuffled)
    second = (written(tmp_path / "b", rows=shuffled) / "easy-report.md").read_text()
    assert first == second


def test_the_random_overlap_baseline_is_derived_from_the_bin_count():
    """Ten equal-count bins over one valid population: two independent bins share `m^2/N` of
    `2m - m^2/N` queries, which is `1/19` for ten bins and never a literal 0.0526."""
    assert RANDOM_BIN_OVERLAP == pytest.approx(1 / (2 * len(DECILE_NAMES) - 1))


def test_the_easy_report_compares_the_winner_against_the_named_benchmark(tmp_path):
    """Spec:229 -- and the benchmark row is the producer's, not three literals."""
    report = (written(tmp_path) / "easy-report.md").read_text()
    summary = grid_summary()
    benchmark = next(
        entry for entry in summary["groups"]
        if (entry["signal"], entry["membership_mode"], entry["confidence_bin"],
            entry["score_scope"], entry["padding_mode"]) == BENCHMARK_SELECTION
    )
    assert f"{benchmark['median_spearman']:+.4f}" in report
    assert "all-query benchmark" in report


def test_a_candidate_that_does_not_win_its_paired_comparison_is_called_undecided(tmp_path):
    """"The benchmark beat it" is the wrong extraction from a null.

    The fixture puts `decile_40_50` on exactly the benchmark's median with every paired image
    a tie, and the report has to say the comparison decided nothing rather than reporting a
    defeat. A generator that turned "did not win" into "lost" dies here.
    """
    report = (written(tmp_path) / "easy-report.md").read_text()
    best = section(report, "Best confidence range")
    assert "**undecided**" in best
    assert "counter-example" in best
    for word in ("lost to", "was beaten", "defeated", "loses to the benchmark"):
        assert word not in report


def test_the_report_counts_how_often_the_paired_comparison_favours_a_candidate(tmp_path):
    """A comparison that favoured every candidate would be a property of the comparison.

    The three counts have to add up to the number of (candidate, benchmark) pairs, so a
    generator that quietly folded the exact ties into either side is visible here.
    """
    summary = grid_summary()
    report = (written(tmp_path) / "easy-report.md").read_text()
    total = len(summary["benchmark_comparisons"])
    match = re.search(
        r"favour (\d+) times, against it (\d+) times, and exactly even (\d+) times", report
    )
    assert match is not None
    assert sum(int(value) for value in match.groups()) == total


def test_the_report_reads_the_confidence_control_as_a_trend_of_its_own(tmp_path):
    """A large positive difference against a control that is itself anti-correlated is partly
    a statement about the control. The fixture makes the control negative at `all_valid`."""
    report = (written(tmp_path) / "easy-report.md").read_text()
    assert "anti-correlated in its own right" in section(
        report, "Persistence versus confidence alone"
    )


def test_the_frozen_twin_is_read_by_its_gap_and_not_by_its_sign(tmp_path):
    """Spec:172's fourth question needs an answer, and the answer comes from the two numbers.

    Freezing removes the bin movement and leaves the fingerprint motion, so the gap between
    the dynamic and frozen medians of one bin is what says whether movement explains the
    result. A gap the metric cannot resolve is reported as no gap.
    """
    summary = grid_summary()
    winner = summary[RANKED_GROUPS_KEY][0]
    twin = _grid_group(summary, "frozen", winner["confidence_bin"])
    assert winner["median_spearman"] != twin["median_spearman"]
    short = section((written(tmp_path) / "easy-report.md").read_text(), "Short answer")
    assert f"{twin['median_spearman']:+.4f}" in short.split("Does padding")[1]


def test_a_confidence_control_that_rises_with_blur_is_read_the_other_way(tmp_path):
    """The other half of the branch above: a control that moves the right way is not a
    reason to discount the difference, and the report must not say it is."""
    summary = grid_summary()
    for group in summary["groups"]:
        if (group["signal"], group["membership_mode"], group["confidence_bin"],
                group["aggregation"], group["padding_mode"]) == (
                    "confidence", ALL_VALID_BENCHMARK[0], ALL_VALID_BENCHMARK[1],
                    "q90", ALL_VALID_BENCHMARK[2]):
            group["median_spearman"] = 0.4
    text = "\n".join(
        reporting_module._confidence_section(summary, summary[RANKED_GROUPS_KEY][0])
    )
    assert "both move with blur" in text
    assert "anti-correlated" not in text


def test_a_missing_heatmap_cell_cannot_be_mistaken_for_a_score_of_zero(tmp_path):
    """The `bad` colour must not be the colormap's own midpoint.

    `coolwarm` is a light grey at zero, so a bin scoring 0.00 and a bin with no row at all
    would be drawn in nearly the same colour -- and the second is the one a reader must not
    read as a measurement.
    """
    figure = reporting_module._heatmap_figure(grid_summary())
    image = figure.axes[0].images[0]
    bad = np.asarray(image.cmap.get_bad())
    midpoint = np.asarray(image.cmap(image.norm(0.0)))
    assert np.abs(bad[:3] - midpoint[:3]).max() > 0.2


def test_the_confidence_panel_is_not_captioned_with_a_persistence_scope(tmp_path):
    """Spec:125 -- the confidence control has no decoder-layer scope, so a panel of
    `1 - confidence` values captioned `persistence at layer_2` states something untrue."""
    figure = reporting_module._blur_curve_figure(summary_frame(full_grid_rows()))
    persistence, confidence = figure.axes
    assert PRIMARY_SCORE_SCOPE in persistence.get_title()
    assert PRIMARY_SCORE_SCOPE not in confidence.get_title()


# --- review round 1: a table that is silently sliced ------------------------------------------


def sparse_rows():
    """The grid cut to three decile bins, plus both benchmarks -- a genuinely sparse table.

    This is the case the shown-absence mechanism replaced brief:82's unimplementable assertion
    for, so it is the case its behaviour has to be checked in.
    """
    kept = {"decile_00_10", "decile_50_60", "decile_90_100", ALL_VALID_BENCHMARK[1]}
    return [row for row in full_grid_rows() if row["confidence_bin"] in kept]


def table_rows(text: str) -> list[list[str]]:
    return [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in text.splitlines()
        if line.startswith("|") and "---" not in line
    ]


def test_the_confidence_table_contains_the_winners_own_row(tmp_path):
    """The paragraph above this table quotes the winner; the table must hold the row it quotes.

    Slicing the table at one scene summary while the headline candidate sits at another puts
    two different numbers for one selection on one page -- `+0.6286 / 176-9-65` in the sentence
    and `+0.6000 / 179-6-65` in the table beneath it -- with nothing saying they are different
    summaries. Both correct, and together misleading. This kills the slice.
    """
    summary = grid_summary()
    winner = summary[RANKED_GROUPS_KEY][0]
    section_text = section((written(tmp_path) / "easy-report.md").read_text(),
                           "Persistence versus confidence alone")
    matched = [
        cells for cells in table_rows(section_text)
        if cells[1] == f"`{winner['confidence_bin']}`"
        and cells[2] == f"`{winner['aggregation']}`"
        and cells[0].startswith(f"`{winner['membership_mode']}`")
    ]
    assert len(matched) == 1, matched
    assert matched[0][3] == f"{winner['median_spearman']:+.4f}"


@pytest.mark.parametrize(("heading", "column"), [
    ("Persistence versus confidence alone", 2),
    ("Dynamic versus frozen queries", 1),
    ("Effect of padded queries", 3),
])
def test_every_report_table_names_the_scene_summary_of_each_row(tmp_path, heading, column):
    """Kills the hidden slice in all three tables at once.

    Each carries a `summary` column and rows at more than one summary, so a reader can never
    take a number out of one of them without knowing which summary it belongs to.
    """
    section_text = section((written(tmp_path) / "easy-report.md").read_text(), heading)
    rows = table_rows(section_text)
    assert rows and rows[0][column] == "summary"
    present = {cells[column].strip("`") for cells in rows[1:]}
    assert present <= set(DECILE_AGGREGATIONS)
    assert len(present) > 1, present


def test_a_table_says_which_slice_is_still_fixed(tmp_path):
    """The figures carry their slice on their own titles; the tables now do the same."""
    report = (written(tmp_path) / "easy-report.md").read_text()
    for heading in ("Persistence versus confidence alone", "Dynamic versus frozen queries"):
        assert "Slice:" in section(report, heading)
        assert PRIMARY_SCORE_SCOPE in section(report, heading)
    assert "the confidence control has no decoder-layer scope" in section(
        report, "Effect of padded queries"
    )


def test_the_absent_pair_message_names_the_slice_it_looked_in(tmp_path):
    """"No matched pair was scored in this table" is false whenever pairs exist at another
    summary or another scope. The message has to say where it looked."""
    summary = grid_summary()
    summary["comparisons"] = []
    lines = reporting_module._confidence_section(summary, None)
    # Asserted on the message line itself. The slice caption above the table also names the
    # scope and the padding rule, so a check over the whole section passes on the generic
    # message and proves nothing -- which is how the first version of this test survived.
    message = [line for line in lines if line.startswith("No matched")]
    assert len(message) == 1
    assert PRIMARY_SCORE_SCOPE in message[0]
    assert "padding union removed" in message[0]


@pytest.mark.parametrize(("heading", "column"), [
    ("Persistence versus confidence alone", 0),
    ("Effect of padded queries", 1),
])
def test_a_frozen_row_is_labelled_diagnostic_in_every_table_it_appears_in(
    tmp_path, heading, column
):
    """Spec:230. The frozen bottom bin publishes the strongest pair of numbers in the report,
    so an unlabelled `frozen` in a membership column beside dynamic rows reads as the best
    method on offer rather than as something that needs a paired clean image."""
    rows = table_rows(section((written(tmp_path) / "easy-report.md").read_text(), heading))
    assert rows[0][column] == "membership"
    frozen = [cells for cells in rows[1:] if cells[column].startswith("`frozen`")]
    assert frozen
    for cells in frozen:
        assert "(diagnostic)" in cells[column]


# --- review round 1: two captions that stated something untrue ---------------------------------


def test_the_dynamic_frozen_panel_title_does_not_claim_a_single_membership(tmp_path):
    """The panel draws dynamic *and* frozen bars; a caption reading "dynamic membership" over
    them says the frozen bars are dynamic."""
    figure = reporting_module._dynamic_frozen_figure(grid_summary())
    axis = figure.axes[0]
    title = axis.get_title()
    assert [container.get_label() for container in axis.containers] == ["dynamic", "frozen"]
    assert f"{reporting_module.DYNAMIC_MEMBERSHIP_MODE} membership" not in title
    assert reporting_module.FIGURE_SLICE not in title
    assert PRIMARY_SCORE_SCOPE in title


def test_the_padding_title_does_not_give_the_confidence_control_a_decoder_scope(tmp_path):
    """Spec:125 -- two of the four bar groups are the confidence control, which has no decoder
    scope, so a blanket "(q90, persistence at layer_2)" over all four is untrue of half."""
    figure = reporting_module._padding_sensitivity_figure(grid_summary())
    axis = figure.axes[0]
    title = axis.get_title()
    assert any("confidence" in label.get_text() for label in axis.get_xticklabels())
    assert f"persistence at {PRIMARY_SCORE_SCOPE})" not in title
    assert "the confidence control has no decoder-layer scope" in title
    assert SENSITIVITY_BIN in title


# --- review round 1: the sparse-table message that could not be read ----------------------------


def test_the_sparse_blur_curve_title_counts_the_missing_bins_rather_than_listing_them(tmp_path):
    """On a three-bin table the list of seven absent names overflows the panel it captions --
    clipped at the left, off the canvas at the right -- in exactly the case the message exists
    for. A count fits; `summary.json` holds which."""
    figure = reporting_module._blur_curve_figure(summary_frame(sparse_rows()))
    for axis in figure.axes:
        title = axis.get_title()
        assert f"7 of {len(DECILE_NAMES)} bins have no row here" in title
        assert "decile_10_20" not in title
        assert max(len(line) for line in title.splitlines()) < 80


def test_a_complete_table_leaves_the_missing_bin_note_off(tmp_path):
    figure = reporting_module._blur_curve_figure(summary_frame(full_grid_rows()))
    assert all("no row here" not in axis.get_title() for axis in figure.axes)


def test_a_sparse_table_still_writes_every_declared_artifact(tmp_path):
    output = written(tmp_path, rows=sparse_rows())
    assert {path.name for path in output.iterdir()} == {
        "per_scene.csv", "summary.json", "confidence_decile_heatmap.png", "blur_curves.png",
        "dynamic_vs_frozen.png", "padding_sensitivity.png", "easy-report.md",
    }


# --- review round 1: an unpaired median difference is not a null --------------------------------


def test_a_frozen_gap_the_metric_cannot_resolve_is_not_read_as_no_effect(tmp_path):
    """Two marginal medians one grid step apart do not establish that bin movement costs
    nothing -- this same report explains that a median difference of exactly zero coexists with
    122 images against 92. No paired dynamic-versus-frozen statistic exists to support the
    stronger claim, so the generator must not make it."""
    reading = reporting_module._freezing_reading(0.6286, 0.6000)
    assert "does not destroy the trend" in reading
    assert "not a paired comparison" in reading
    assert "costs nothing" in reading and "does not establish" in reading
    report = (written(tmp_path) / "easy-report.md").read_text()
    assert "is not costing it anything" not in report


@pytest.mark.parametrize(("dynamic", "frozen"), [(0.1, 0.9), (0.9, 0.1)])
def test_even_a_large_frozen_gap_is_marked_unpaired(dynamic, frozen):
    reading = reporting_module._freezing_reading(dynamic, frozen)
    assert "not a paired comparison" in reading


def test_the_short_answer_carries_the_selection_caveat_beside_the_claim(tmp_path):
    """The consequence of selecting on the reported images belonged beside the strongest claim,
    not seven sections below it."""
    short = section((written(tmp_path) / "easy-report.md").read_text(), "Short answer")
    assert "not a hypothesis test" in short
