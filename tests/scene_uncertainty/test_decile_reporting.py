"""Tests for the confidence-decile summariser.

The brief's three tests, plus one test per binding sentence of the spec they leave
unexercised. The added ones cluster around four places where a summary can be complete,
plausible and wrong: a group key that stops separating two selections, a coverage gate that
annotates instead of excluding, a paired statistic computed over marginal populations, and a
raw score magnitude leaking into a number that is compared across scopes.
"""

from __future__ import annotations

import ast
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from src.scene_uncertainty import decile_reporting as reporting_module
from src.scene_uncertainty.decile_analysis import ROW_KEYS_EXCLUDED_FROM_CSV
from src.scene_uncertainty.decile_reporting import (
    GROUP_KEYS,
    RANKED_GROUPS_KEY,
    RANKABLE_MEMBERSHIP_MODES,
    ROW_KEYS,
    SIGNALS,
    rank_deployable_groups,
    summarize_decile_rows,
    summary_frame,
)
from src.scene_uncertainty.decile_scoring import PRIMARY_SCORE_SCOPE, score_selection


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
    "mean_clean_overlap",
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
    assert persistence["mean_clean_overlap"] == 1.0
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


def test_clean_overlap_is_reported_per_severity_and_not_flattened():
    rows = [
        {**scored, "clean_overlap": 1.0 if scored["severity"] == 0 else 0.06}
        for scored in dipping_rows()
    ]
    summary = summarize_decile_rows(rows, {})
    persistence = group(summary, "persistence", PRIMARY_SCORE_SCOPE)
    assert persistence["mean_clean_overlap_by_severity"]["0"] == 1.0
    assert persistence["mean_clean_overlap_by_severity"]["5"] == pytest.approx(0.06)
    assert persistence["mean_clean_overlap"] == pytest.approx((1.0 + 5 * 0.06) / 6)


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
    assert sensitivity[0]["differing_image_count"] == 1
    assert sensitivity[0]["unfiltered_image_win_rate"] == 0.0
    assert sensitivity[0]["differing_image_win_rate"] == 0.0


def test_the_padding_win_rate_is_reported_both_over_all_images_and_over_the_moved_ones():
    """Spec:69's control is a no-op on an image that had no padded tail to remove.

    Two images: the mask moves the second and leaves the first untouched. Counting the
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
    assert sensitivity["differing_image_count"] == 1
    # Image 1 is untouched by the mask and is counted as a loss by the unrestricted rate.
    assert sensitivity["unfiltered_image_win_rate"] == 0.5
    assert sensitivity["differing_image_win_rate"] == 1.0


def test_a_padding_control_that_moved_no_image_reports_no_restricted_rate():
    rows = []
    for severity in range(6):
        rows.append(row(1, severity, "persistence", PRIMARY_SCORE_SCOPE, severity))
        rows.append(row(
            1, severity, "persistence", PRIMARY_SCORE_SCOPE, severity, padding_mode="unfiltered",
        ))
    sensitivity = summarize_decile_rows(rows, {})["padding_sensitivity"][0]
    assert sensitivity["differing_image_count"] == 0
    # Nothing to take a rate over: reported as missing rather than as a zero a reader would
    # take for a measured failure.
    assert sensitivity["differing_image_win_rate"] is None
    assert sensitivity["unfiltered_image_win_rate"] == 0.0


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
