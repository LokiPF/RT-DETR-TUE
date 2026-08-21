"""Collapsing scored rows into candidates, and the arithmetic that fakes each collapse.

`corruption_reporting` turns one row per image, severity and candidate into one row per
candidate plus a ranking. Every collapse in it has a plausible wrong version that still
produces a complete, internally consistent table, so these tests are written to fail those
versions rather than to restate the right one:

* **strength is the median of per-image absolute Spearmans, not the absolute value of the
  median signed one.** `test_group_strength_is_median_of_per_image_absolute_spearman` uses two
  images with opposite perfect trends, where the two readings are `0.0` and `1.0` -- the widest
  gap the statistic admits, and the only case where reading one for the other is unmissable;
* **orientation is chosen once per candidate and then locked.** Two tests carry this. One gives
  persistence and its matched confidence control opposite trends and unequal image counts, so a
  chooser that pooled the table would hand the control the persistence direction rather than
  its own; the other locks `+1` on a candidate whose severity 3 ranks *below* clean and asserts
  that AUROC stays at `0.0`, because a per-severity chooser would report `1.0` there and lift
  the macro average from 0.8 to 1.0;
* **flat and unmeasured are different facts.** A constant curve is a completed measurement and
  belongs in the dominant-direction denominator; a `nan` score or an absent severity is not and
  never becomes a zero. Two tests pin the two denominators apart, one at `2/3` where dropping
  the flat image reads `1.0`, the other on a candidate with one measured image out of three;
* **the severity axis is this module's obligation, not its input's.**
  `corruption_metrics.oriented_curve_metrics` reads index *i* as severity *i* without checking.
  `test_rows_are_ordered_by_severity_before_any_curve_is_read` hands the rows in descending
  severity and asserts an increasing trend with a perfect adjacent consistency, which is what an
  implementation that trusted row order gets wrong in three different ways;
* **the ranking gates exclude rather than annotate.** The 250-image table carries one candidate
  per gate, each failing exactly one, and asserts the ranking holds the single survivor;
* **the sort order is the metrics first and the candidate key only as a tie-break.** The
  sorting test names its candidates so that the candidate-key order is the exact *reverse* of
  the expected metric order, so a ranking that fell back on the key would come out backwards
  instead of accidentally right.

The 250-image tests build real tables (250 images x 6 severities per candidate) rather than
stub candidate dictionaries, because the `FULL_TUNING_IMAGE_COUNT` gates are the part of this
module a smaller fixture cannot exercise at all.
"""

import json

import pytest

from src.scene_uncertainty.corruption_reporting import (
    CANDIDATE_KEY,
    DEPLOYABLE_SCORE_SCOPE,
    FULL_TUNING_IMAGE_COUNT,
    PER_SCENE_KEYS,
    ROW_KEY,
    SELECTION_KEY,
    rank_deployable_candidates,
    summarize_candidates,
    validate_rows,
    validate_selected_queries,
)


LABELS = {
    "signal": "persistence",
    "bucket_scheme": "decile",
    "confidence_bin": "decile_00_10",
    "membership_mode": "dynamic",
    "padding_mode": "filtered",
    "aggregation": "q90",
    "score_scope": "layer_2",
}
"""One deployable candidate's labels. Every fixture starts here and overrides one field, so a
test that means to fail a single gate cannot quietly fail two."""

RISING = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
FALLING = tuple(reversed(RISING))
FLAT = (2.0,) * 6
NAN = float("nan")


def _at(value, severity):
    return value[severity] if isinstance(value, dict) else value


def candidate_rows(curves, *, selected_count=4, clean_overlap=0.5, descending=False, **labels):
    """Six scored rows an image for one candidate, from `{image_id: [score per severity]}`.

    `None` in a curve omits that severity's row entirely and `float("nan")` keeps the row with
    a non-finite score. They are two different ways to fail to produce a curve and the reporter
    has to reach the same verdict by two different paths, so the helper can spell both.

    `descending` emits each image's rows from severity 5 down to 0 without changing any score,
    which is the input a reporter that trusted row order would misread.

    `selected_count` and `clean_overlap` accept a mapping keyed by severity for the tests that
    check the per-severity diagnostics. The selected query IDs are derived from the count, so
    every row of one selection reports the same list -- which is what
    `validate_selected_queries` demands, and it makes a deliberately mismatched fixture a
    one-line change.
    """
    fields = {**LABELS, **labels}
    rows = []
    for image_id, scores in curves.items():
        severities = list(range(len(scores)))
        for severity in reversed(severities) if descending else severities:
            score = scores[severity]
            if score is None:
                continue
            count = int(_at(selected_count, severity))
            rows.append({
                "image_id": image_id,
                "severity": severity,
                **fields,
                "source_partition": "tuning",
                "score": float(score),
                "selected_count": count,
                "clean_overlap": float(_at(clean_overlap, severity)),
                "selected_query_ids": list(range(count)),
            })
    return rows


def run_curves(curve=RISING, count=FULL_TUNING_IMAGE_COUNT):
    return {image_id: list(curve) for image_id in range(count)}


def lookup(candidates, **match):
    found = [
        candidate for candidate in candidates
        if all(candidate[field] == value for field, value in match.items())
    ]
    assert len(found) == 1, f"expected one candidate matching {match}, got {len(found)}"
    return found[0]


def one_candidate(rows):
    """The single candidate a one-candidate row set summarises to."""
    image_count = len({row["image_id"] for row in rows})
    _, candidates, _ = summarize_candidates(rows, expected_image_count=image_count)
    assert len(candidates) == 1
    return candidates[0]


# --- the key contract ---------------------------------------------------------------------


def test_the_candidate_key_is_the_row_key_without_the_scene_coordinates():
    """Ruling 2: candidates are separated by the recorded scheme, never by a bin-name prefix."""
    assert ROW_KEY == (
        "image_id", "severity", "signal", "bucket_scheme", "confidence_bin",
        "membership_mode", "padding_mode", "aggregation", "score_scope",
    )
    assert CANDIDATE_KEY == (
        "signal", "bucket_scheme", "confidence_bin", "membership_mode", "padding_mode",
        "aggregation", "score_scope",
    )
    assert SELECTION_KEY == (
        "image_id", "severity", "bucket_scheme", "confidence_bin", "membership_mode",
        "padding_mode",
    )
    assert FULL_TUNING_IMAGE_COUNT == 250
    assert DEPLOYABLE_SCORE_SCOPE == "layer_2"


def test_decile_and_quintile_rows_of_the_same_severity_are_two_candidates():
    """The two schemes name their bins differently; the split is on the scheme field itself."""
    rows = candidate_rows({1: RISING}) + candidate_rows(
        {1: FALLING}, bucket_scheme="quintile", confidence_bin="quintile_00_20"
    )
    _, candidates, _ = summarize_candidates(rows, expected_image_count=1)

    assert [candidate["bucket_scheme"] for candidate in candidates] == ["decile", "quintile"]
    assert lookup(candidates, bucket_scheme="decile")["orientation"] == 1
    assert lookup(candidates, bucket_scheme="quintile")["orientation"] == -1


def test_candidates_come_back_in_candidate_key_order_whatever_order_the_rows_arrived_in():
    rows = candidate_rows({1: RISING}, aggregation="q90") + candidate_rows(
        {1: RISING}, aggregation="mean"
    )
    _, candidates, _ = summarize_candidates(rows, expected_image_count=1)

    assert [candidate["aggregation"] for candidate in candidates] == ["mean", "q90"]


# --- strength: the median of the absolute values, not the absolute value of the median -----


def test_group_strength_is_median_of_per_image_absolute_spearman():
    rows = candidate_rows(
        curves={1: [0, 1, 2, 3, 4, 5], 2: [5, 4, 3, 2, 1, 0]}
    )
    _, metrics, ranking = summarize_candidates(rows, expected_image_count=2)
    metric = metrics[0]

    assert metric["median_signed_spearman"] == pytest.approx(0.0)
    assert metric["median_absolute_spearman"] == pytest.approx(1.0)
    assert metric["orientation"] is None
    assert metric["deployable"] is False
    assert ranking == []


def test_an_unorientable_candidate_publishes_no_curve_checks_and_no_auroc():
    """`choose_orientation` returns `None` rather than raising; nothing may fill the gap."""
    candidate = one_candidate(candidate_rows({1: RISING, 2: FALLING}))

    assert candidate["orientation"] is None
    assert candidate["oriented_adjacent_consistency"] is None
    assert candidate["max_blur_above_clean_rate"] is None
    assert candidate["macro_auroc"] is None
    assert candidate["auroc_by_severity"] == {str(severity): None for severity in range(1, 6)}
    assert candidate["auroc_severity_3"] is None


# --- orientation: one per candidate, chosen from that candidate's images alone -------------


def test_persistence_and_its_confidence_control_choose_orientation_independently():
    """Pooling the table would read `+1` for both: the pooled median of [+1, +1, -1] is `+1`."""
    rows = candidate_rows({1: RISING, 2: RISING})
    rows += candidate_rows({1: FALLING}, signal="confidence", score_scope="confidence")
    _, candidates, _ = summarize_candidates(rows, expected_image_count=2)

    assert lookup(candidates, signal="persistence")["orientation"] == 1
    assert lookup(candidates, signal="confidence")["orientation"] == -1


def test_one_orientation_is_reused_for_all_five_aurocs():
    """Severity 3 dips below clean. A per-severity chooser reads 1.0 there and 1.0 macro."""
    candidate = one_candidate(
        candidate_rows({1: [0, 10, 20, -5, 40, 50], 2: [1, 11, 21, -4, 41, 51]})
    )

    assert candidate["orientation"] == 1
    assert candidate["auroc_by_severity"] == {
        "1": 1.0, "2": 1.0, "3": 0.0, "4": 1.0, "5": 1.0
    }
    assert candidate["auroc_severity_3"] == 0.0
    assert candidate["macro_auroc"] == pytest.approx(0.8)
    assert candidate["oriented_adjacent_consistency"] == pytest.approx(0.8)
    assert candidate["max_blur_above_clean_rate"] == pytest.approx(1.0)
    assert candidate["median_signed_spearman"] == pytest.approx(0.6571428571428571)


def test_a_negatively_oriented_candidate_is_read_downwards_rather_than_flipped():
    candidate = one_candidate(candidate_rows({1: FALLING, 2: FALLING}))

    assert candidate["orientation"] == -1
    assert candidate["median_signed_spearman"] == pytest.approx(-1.0)
    assert candidate["median_absolute_spearman"] == pytest.approx(1.0)
    assert candidate["macro_auroc"] == pytest.approx(1.0)
    assert candidate["oriented_adjacent_consistency"] == pytest.approx(1.0)


# --- flat is a measurement, unmeasured is not ----------------------------------------------


def test_flat_complete_curves_count_in_the_dominant_direction_denominator():
    """Three denominators at once, all different: 4 images, 3 measured, 2 with a direction.

    Dropping the flat image reads 2/2 = 1.0, a perfectly consistent candidate that is not; and
    a direction fraction taken over the images rather than the measured ones reads 1/4 for the
    flat share where the answer is 1/3.
    """
    candidate = one_candidate(
        candidate_rows({1: RISING, 2: RISING, 3: FLAT, 4: [0.0, 1.0, NAN, 3.0, 4.0, 5.0]})
    )

    assert candidate["image_count"] == 4
    assert candidate["measured_count"] == 3
    assert candidate["positive_count"] == 2
    assert candidate["negative_count"] == 0
    assert candidate["flat_count"] == 1
    assert candidate["dominant_direction_fraction"] == pytest.approx(2 / 3)
    assert candidate["positive_fraction"] == pytest.approx(2 / 3)
    assert candidate["flat_fraction"] == pytest.approx(1 / 3)
    assert candidate["measured_fraction"] == pytest.approx(3 / 4)
    assert candidate["median_absolute_spearman"] == pytest.approx(1.0)


def test_missing_and_non_finite_curves_are_unmeasured_and_never_flat():
    nan_curve = [0.0, 1.0, 2.0, NAN, 4.0, 5.0]
    short_curve = [0.0, 1.0, 2.0, 3.0, 4.0, None]
    rows = candidate_rows({1: RISING, 2: nan_curve, 3: short_curve})
    _, candidates, _ = summarize_candidates(rows, expected_image_count=4)
    candidate = candidates[0]

    assert candidate["expected_image_count"] == 4
    assert candidate["image_count"] == 3
    assert candidate["measured_count"] == 1
    assert candidate["missing_count"] == 2
    assert candidate["flat_count"] == 0
    assert candidate["positive_count"] == 1
    assert candidate["measured_fraction"] == pytest.approx(1 / 3)
    assert candidate["missing_fraction"] == pytest.approx(2 / 3)
    assert candidate["positive_fraction"] == pytest.approx(1.0)
    assert candidate["dominant_direction_fraction"] == pytest.approx(1.0)
    assert candidate["median_absolute_spearman"] == pytest.approx(1.0)


def test_an_unmeasured_image_says_so_on_every_one_of_its_scene_rows():
    per_scene, _, _ = summarize_candidates(
        candidate_rows({1: [0.0, 1.0, 2.0, NAN, 4.0, 5.0]}), expected_image_count=1
    )

    assert len(per_scene) == 6
    assert {row["direction"] for row in per_scene} == {"unmeasured"}
    assert {row["fully_measured"] for row in per_scene} == {False}
    assert {row["signed_spearman"] for row in per_scene} == {None}
    assert {row["absolute_spearman"] for row in per_scene} == {None}


# --- the deployment gates ------------------------------------------------------------------


def full_run_rows():
    """Eight candidates over 250 images: one deployable, and one failing each gate alone."""
    nan_at_two = run_curves()
    nan_at_two[7][2] = NAN
    short_curve = run_curves()
    short_curve[7][5] = None

    rows = candidate_rows(run_curves())
    rows += candidate_rows(run_curves(), membership_mode="frozen")
    rows += candidate_rows(run_curves(), padding_mode="unfiltered")
    rows += candidate_rows(run_curves(), signal="confidence", score_scope="confidence")
    rows += candidate_rows(run_curves(), score_scope="layer_1")
    rows += candidate_rows(nan_at_two, aggregation="mean")
    rows += candidate_rows(short_curve, aggregation="top20_mean")
    rows += candidate_rows(run_curves(FLAT), confidence_bin="decile_10_20")
    return rows


def test_only_dynamic_filtered_full_coverage_persistence_layer_2_candidates_are_ranked():
    rows = full_run_rows()
    per_scene, candidates, ranking = summarize_candidates(
        rows, expected_image_count=FULL_TUNING_IMAGE_COUNT
    )

    assert len(candidates) == 8
    assert lookup(candidates, membership_mode="frozen")["deployable"] is False
    assert lookup(candidates, padding_mode="unfiltered")["deployable"] is False
    assert lookup(candidates, signal="confidence")["deployable"] is False
    assert lookup(candidates, score_scope="layer_1")["deployable"] is False
    assert lookup(candidates, aggregation="mean")["measured_count"] == 249
    assert lookup(candidates, aggregation="mean")["deployable"] is False
    assert lookup(candidates, aggregation="top20_mean")["measured_count"] == 249
    assert lookup(candidates, aggregation="top20_mean")["deployable"] is False
    assert lookup(candidates, confidence_bin="decile_10_20")["orientation"] is None
    assert lookup(candidates, confidence_bin="decile_10_20")["deployable"] is False

    winner = lookup(candidates, **LABELS)
    assert winner["deployable"] is True
    assert winner["measured_count"] == FULL_TUNING_IMAGE_COUNT
    assert ranking == [winner]
    assert len(per_scene) == len(rows)


def test_the_confidence_control_is_summarised_in_full_but_never_ranked():
    _, candidates, ranking = summarize_candidates(
        full_run_rows(), expected_image_count=FULL_TUNING_IMAGE_COUNT
    )
    control = lookup(candidates, signal="confidence")

    assert control["orientation"] == 1
    assert control["macro_auroc"] == pytest.approx(1.0)
    assert control["median_absolute_spearman"] == pytest.approx(1.0)
    assert control["measured_count"] == FULL_TUNING_IMAGE_COUNT
    assert control not in ranking


def test_the_confidence_signal_is_excluded_even_when_it_is_filed_at_the_primary_scope():
    """The signal gate is its own conjunct, not a consequence of the score-scope gate.

    The fixture is deliberately malformed: the confidence control has no decoder-layer scope and
    the producer stores it once under `confidence`, so a real table never carries this pairing.
    That is exactly why it is written here -- in the eight-candidate table above the control
    fails the signal gate and the scope gate together, so neither is tested on its own, and a
    reporter that had dropped the signal check would pass every assertion in it.
    """
    _, candidates, ranking = summarize_candidates(
        candidate_rows(run_curves(), signal="confidence"),
        expected_image_count=FULL_TUNING_IMAGE_COUNT,
    )

    assert candidates[0]["score_scope"] == DEPLOYABLE_SCORE_SCOPE
    assert candidates[0]["measured_count"] == FULL_TUNING_IMAGE_COUNT
    assert candidates[0]["orientation"] == 1
    assert candidates[0]["deployable"] is False
    assert ranking == []


def test_a_candidate_that_lost_an_image_is_excluded_even_when_every_severity_is_full():
    """The measured-image gate is its own conjunct, not a consequence of the coverage gate.

    249 complete images plus two half-curves that between them fill every severity back up to
    250 finite scores. The per-severity coverage gate sees 250 six times over and is satisfied;
    only `measured_count` knows that 249 images produced a curve, and the ranking rests on
    per-image trends that the two half-curves never contributed.
    """
    curves = run_curves(count=249)
    curves[249] = [0.0, 1.0, 2.0, None, None, None]
    curves[250] = [None, None, None, 3.0, 4.0, 5.0]
    _, candidates, ranking = summarize_candidates(
        candidate_rows(curves), expected_image_count=FULL_TUNING_IMAGE_COUNT
    )
    candidate = candidates[0]

    assert candidate["image_count"] == 251
    assert candidate["measured_count"] == 249
    assert candidate["orientation"] == 1
    assert [
        candidate["coverage_by_severity"][str(severity)]["finite_count"]
        for severity in range(6)
    ] == [250] * 6
    assert candidate["deployable"] is False
    assert ranking == []


def test_a_run_that_is_not_the_full_tuning_set_has_no_deployable_candidate():
    """250 measured images, but the run declared 249 -- the gate is on the run, not the count."""
    _, candidates, ranking = summarize_candidates(
        candidate_rows(run_curves()), expected_image_count=249
    )

    assert candidates[0]["measured_count"] == FULL_TUNING_IMAGE_COUNT
    assert candidates[0]["orientation"] == 1
    assert candidates[0]["deployable"] is False
    assert ranking == []


def test_more_images_than_the_run_measured_fails_the_per_severity_coverage_gate():
    """251 images, 250 fully measured: every severity but 0 carries one score too many."""
    curves = run_curves(count=FULL_TUNING_IMAGE_COUNT + 1)
    curves[250][0] = NAN
    _, candidates, ranking = summarize_candidates(
        candidate_rows(curves), expected_image_count=FULL_TUNING_IMAGE_COUNT
    )
    candidate = candidates[0]

    assert candidate["measured_count"] == FULL_TUNING_IMAGE_COUNT
    assert candidate["orientation"] == 1
    assert candidate["coverage_by_severity"]["0"]["finite_count"] == 250
    assert candidate["coverage_by_severity"]["1"]["finite_count"] == 251
    assert candidate["deployable"] is False
    assert ranking == []


# --- ranking order --------------------------------------------------------------------------


def ranking_candidate(macro, absolute, dominant, adjacent, *, deployable=True, **labels):
    return {
        **LABELS,
        **labels,
        "macro_auroc": macro,
        "median_absolute_spearman": absolute,
        "dominant_direction_fraction": dominant,
        "oriented_adjacent_consistency": adjacent,
        "deployable": deployable,
    }


def test_ranking_sorts_on_each_metric_before_falling_back_to_the_candidate_key():
    """The bin names run *backwards* against the expected order, so a key-first sort inverts."""
    best_auroc = ranking_candidate(0.9, 0.5, 0.5, 0.5, confidence_bin="decile_50_60")
    best_consistency = ranking_candidate(0.8, 0.9, 0.9, 0.9, confidence_bin="decile_40_50")
    best_dominance = ranking_candidate(0.8, 0.9, 0.9, 0.1, confidence_bin="decile_30_40")
    weak_dominance = ranking_candidate(0.8, 0.9, 0.1, 0.1, confidence_bin="decile_20_30")
    weak_strength = ranking_candidate(0.8, 0.5, 0.9, 0.9, confidence_bin="decile_10_20")
    tied_first = ranking_candidate(0.7, 0.5, 0.5, 0.5, confidence_bin="decile_60_70")
    tied_second = ranking_candidate(0.7, 0.5, 0.5, 0.5, confidence_bin="decile_70_80")
    decile_tie = ranking_candidate(
        0.6, 0.5, 0.5, 0.5, confidence_bin="decile_00_10", aggregation="q90"
    )
    quintile_tie = ranking_candidate(
        0.6, 0.5, 0.5, 0.5, bucket_scheme="quintile", confidence_bin="quintile_00_20",
        aggregation="mean",
    )
    excluded = ranking_candidate(
        1.0, 1.0, 1.0, 1.0, deployable=False, confidence_bin="decile_80_90"
    )

    ranked = rank_deployable_candidates([
        quintile_tie, decile_tie, tied_second, tied_first, excluded, weak_strength,
        weak_dominance, best_dominance, best_consistency, best_auroc,
    ])

    assert [candidate["confidence_bin"] for candidate in ranked] == [
        "decile_50_60", "decile_40_50", "decile_30_40", "decile_20_30", "decile_10_20",
        "decile_60_70", "decile_70_80", "decile_00_10", "quintile_00_20",
    ]


# --- per-severity raw scene statistics -------------------------------------------------------


def test_per_severity_statistics_use_population_variance_and_linear_quantiles():
    """Sample variance reads 2.0 on this pair, and a non-interpolating quantile reads 1.0."""
    candidate = one_candidate(candidate_rows({1: [1, 2, 3, 4, 5, 6], 2: [3, 4, 5, 6, 7, 8]}))

    assert candidate["severity_statistics"]["0"] == {
        "count": 2,
        "mean": pytest.approx(2.0),
        "variance": pytest.approx(1.0),
        "median": pytest.approx(2.0),
        "q25": pytest.approx(1.5),
        "q75": pytest.approx(2.5),
    }
    assert candidate["score_variance_severity_0"] == pytest.approx(1.0)
    assert candidate["score_q25_severity_0"] == pytest.approx(1.5)
    assert candidate["score_count_severity_5"] == 2


def test_coverage_names_both_sides_of_every_severity_comparison():
    """An image dropped from the corrupted side must be visible against the clean count."""
    candidate = one_candidate(
        candidate_rows({1: RISING, 2: [0.0, 1.0, 2.0, NAN, 4.0, 5.0]})
    )

    assert candidate["coverage_by_severity"]["3"] == {
        "image_count": 2, "finite_count": 1, "clean_count": 2, "corrupted_count": 1
    }
    assert candidate["clean_count_severity_3"] == 2
    assert candidate["corrupted_count_severity_3"] == 1
    assert candidate["finite_count_severity_3"] == 1


def test_selected_counts_and_clean_overlap_are_summarised_by_severity():
    rows = candidate_rows(
        {1: RISING},
        selected_count={0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8},
        clean_overlap={0: 1.0, 1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2, 5: 0.1},
    )
    rows += candidate_rows(
        {2: RISING},
        selected_count={0: 9, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8},
        clean_overlap={0: 0.6, 1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2, 5: 0.1},
    )
    _, candidates, _ = summarize_candidates(rows, expected_image_count=2)
    candidate = candidates[0]

    assert candidate["selected_count_by_severity"]["0"] == {
        "min": 3, "median": pytest.approx(6.0), "max": 9
    }
    assert candidate["selected_count_min_severity_0"] == 3
    assert candidate["selected_count_max_severity_0"] == 9
    assert candidate["median_clean_overlap_by_severity"]["0"] == pytest.approx(0.8)
    assert candidate["median_clean_overlap_severity_0"] == pytest.approx(0.8)
    assert candidate["median_clean_overlap_by_severity"]["5"] == pytest.approx(0.1)


# --- padding sensitivity ----------------------------------------------------------------------


def test_padding_sensitivity_pairs_only_the_twin_that_differs_in_padding_mode():
    rows = candidate_rows({1: RISING})
    rows += candidate_rows({1: [1, 2, 3, 4, 5, 6]}, padding_mode="unfiltered")
    rows += candidate_rows({1: RISING}, confidence_bin="decile_10_20")
    rows += candidate_rows(
        {1: RISING}, bucket_scheme="quintile", confidence_bin="quintile_00_20",
        padding_mode="unfiltered",
    )
    _, candidates, _ = summarize_candidates(rows, expected_image_count=1)

    filtered = lookup(candidates, confidence_bin="decile_00_10", padding_mode="filtered")
    unfiltered = lookup(candidates, confidence_bin="decile_00_10", padding_mode="unfiltered")
    unpaired_bin = lookup(candidates, confidence_bin="decile_10_20")
    unpaired_scheme = lookup(candidates, bucket_scheme="quintile")

    assert filtered["padding_counterpart_padding_mode"] == "unfiltered"
    assert unfiltered["padding_counterpart_padding_mode"] == "filtered"
    assert filtered["padding_median_unfiltered_minus_filtered_score"] == pytest.approx(1.0)
    assert unfiltered["padding_median_unfiltered_minus_filtered_score"] == pytest.approx(1.0)
    assert filtered["padding_paired_score_count"] == 6
    assert filtered["padding_changed_score_count"] == 6
    assert unpaired_bin["padding_counterpart_padding_mode"] is None
    assert unpaired_bin["padding_median_unfiltered_minus_filtered_score"] is None
    assert unpaired_bin["padding_paired_score_count"] == 0
    assert unpaired_scheme["padding_counterpart_padding_mode"] is None


def test_a_padding_mask_that_changed_nothing_reports_a_zero_difference_not_a_missing_one():
    rows = candidate_rows({1: RISING})
    rows += candidate_rows({1: RISING}, padding_mode="unfiltered")
    _, candidates, _ = summarize_candidates(rows, expected_image_count=1)
    filtered = lookup(candidates, padding_mode="filtered")

    assert filtered["padding_median_unfiltered_minus_filtered_score"] == pytest.approx(0.0)
    assert filtered["padding_paired_score_count"] == 6
    assert filtered["padding_changed_score_count"] == 0


# --- the per-scene table -----------------------------------------------------------------------


def test_per_scene_has_one_row_per_candidate_image_and_severity_without_the_id_list():
    rows = candidate_rows({1: RISING, 2: FALLING})
    rows += candidate_rows({1: RISING, 2: FALLING}, aggregation="mean")
    per_scene, _, _ = summarize_candidates(rows, expected_image_count=2)

    assert len(per_scene) == len(rows) == 24
    assert all(set(row) == set(PER_SCENE_KEYS) for row in per_scene)
    assert all("selected_query_ids" not in row for row in per_scene)
    assert [(row["aggregation"], row["image_id"], row["severity"]) for row in per_scene] == [
        (aggregation, image_id, severity)
        for aggregation in ("mean", "q90")
        for image_id in (1, 2)
        for severity in range(6)
    ]


def test_every_scene_row_repeats_its_image_trend_so_it_reads_on_its_own():
    per_scene, _, _ = summarize_candidates(
        candidate_rows({1: RISING}), expected_image_count=1
    )

    assert {row["signed_spearman"] for row in per_scene} == {1.0}
    assert {row["absolute_spearman"] for row in per_scene} == {1.0}
    assert {row["direction"] for row in per_scene} == {"increasing"}
    assert [row["score"] for row in per_scene] == list(RISING)
    assert [row["clean_overlap"] for row in per_scene] == [0.5] * 6
    assert [row["selected_count"] for row in per_scene] == [4] * 6
    assert [row["source_partition"] for row in per_scene] == ["tuning"] * 6


# --- the severity axis is the caller's obligation ------------------------------------------------


def test_rows_are_ordered_by_severity_before_any_curve_is_read():
    """Descending input: an order-trusting reporter reads unmeasured, `-1`, or a 0.0 consistency."""
    per_scene, candidates, _ = summarize_candidates(
        candidate_rows({1: RISING, 2: RISING}, descending=True), expected_image_count=2
    )
    candidate = candidates[0]

    assert candidate["measured_count"] == 2
    assert candidate["median_signed_spearman"] == pytest.approx(1.0)
    assert candidate["orientation"] == 1
    assert candidate["oriented_adjacent_consistency"] == pytest.approx(1.0)
    assert candidate["max_blur_above_clean_rate"] == pytest.approx(1.0)
    assert [row["severity"] for row in per_scene] == [0, 1, 2, 3, 4, 5] * 2


# --- both views of every diagnostic -----------------------------------------------------------


def test_every_nested_diagnostic_has_a_matching_scalar_column():
    """Ruling 1: `summary.json` and the plots read the nested maps, the CSV reads the columns."""
    candidate = one_candidate(candidate_rows({1: RISING, 2: RISING}))
    nested = {name for name, value in candidate.items() if isinstance(value, dict)}

    assert nested == {
        "auroc_by_severity", "severity_statistics", "coverage_by_severity",
        "selected_count_by_severity", "median_clean_overlap_by_severity",
    }
    for severity in range(6):
        statistics = candidate["severity_statistics"][str(severity)]
        assert candidate[f"score_count_severity_{severity}"] == statistics["count"]
        assert candidate[f"score_mean_severity_{severity}"] == statistics["mean"]
        assert candidate[f"score_variance_severity_{severity}"] == statistics["variance"]
        assert candidate[f"score_median_severity_{severity}"] == statistics["median"]
        assert candidate[f"score_q25_severity_{severity}"] == statistics["q25"]
        assert candidate[f"score_q75_severity_{severity}"] == statistics["q75"]
        coverage = candidate["coverage_by_severity"][str(severity)]
        assert candidate[f"image_count_severity_{severity}"] == coverage["image_count"]
        assert candidate[f"finite_count_severity_{severity}"] == coverage["finite_count"]
        assert candidate[f"clean_count_severity_{severity}"] == coverage["clean_count"]
        assert candidate[f"corrupted_count_severity_{severity}"] == coverage["corrupted_count"]
        selected = candidate["selected_count_by_severity"][str(severity)]
        assert candidate[f"selected_count_min_severity_{severity}"] == selected["min"]
        assert candidate[f"selected_count_median_severity_{severity}"] == selected["median"]
        assert candidate[f"selected_count_max_severity_{severity}"] == selected["max"]
        assert (
            candidate[f"median_clean_overlap_severity_{severity}"]
            == candidate["median_clean_overlap_by_severity"][str(severity)]
        )
    for severity in range(1, 6):
        assert (
            candidate[f"auroc_severity_{severity}"]
            == candidate["auroc_by_severity"][str(severity)]
        )

    projection = {name: value for name, value in candidate.items() if name not in nested}
    assert all(
        value is None or isinstance(value, (bool, int, float, str))
        for value in projection.values()
    )


def test_candidate_metrics_are_plain_python_values_json_can_carry():
    _, candidates, _ = summarize_candidates(
        candidate_rows({1: RISING, 2: RISING}), expected_image_count=2
    )

    assert json.dumps(candidates, allow_nan=False)


# --- refusals -----------------------------------------------------------------------------------


def test_duplicate_row_keys_are_rejected_before_any_grouping():
    rows = candidate_rows({1: RISING})
    duplicate = dict(rows[3])
    duplicate["score"] = 99.0
    rows.append(duplicate)

    with pytest.raises(ValueError, match="duplicate corruption score row key"):
        summarize_candidates(rows, expected_image_count=1)


def test_validate_rows_names_the_fields_a_row_is_missing():
    rows = candidate_rows({1: RISING})
    del rows[2]["bucket_scheme"]

    with pytest.raises(ValueError, match=r"score row 2 is missing fields \['bucket_scheme'\]"):
        validate_rows(rows)


def test_a_selection_whose_rows_disagree_about_their_query_ids_is_rejected():
    """Persistence and confidence must summarise one population; only the IDs can say so."""
    rows = candidate_rows({1: RISING})
    rows += candidate_rows({1: RISING}, signal="confidence", score_scope="confidence")
    rows[-1]["selected_query_ids"] = [0, 1, 2, 99]

    with pytest.raises(ValueError, match="report different selected query IDs"):
        summarize_candidates(rows, expected_image_count=1)


def test_a_row_without_its_selected_query_ids_is_rejected_rather_than_skipped():
    rows = candidate_rows({1: RISING})
    del rows[4]["selected_query_ids"]

    with pytest.raises(ValueError, match="score row 4 is missing 'selected_query_ids'"):
        validate_selected_queries(rows)


def test_an_empty_results_table_is_refused_rather_than_summarised_into_nothing():
    with pytest.raises(ValueError, match="at least one scored row"):
        summarize_candidates([], expected_image_count=1)
