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
The bundle writer at the bottom of this file is tested against a different family of failures,
all of which produce a directory a reader would accept:

* **a bundle published before it was checked.** Two tests hand the writer a plot step that
  leaves nine files, and one that leaves seven, and assert the *output directory does not
  exist* afterwards. A test that only walked the happy path would pass against a writer that
  renamed the staging directory into place and complained afterwards;
* **a staging directory left behind.** Every failure test asserts the output's parent is
  completely empty, not merely that the bundle is absent -- a `.corruption.XXXX` directory
  beside it is exactly what a half-written run looks like, and it is invisible to `not
  output.exists()`;
* **nested dictionaries in CSV cells.** `candidate_metrics.csv` is written from the scalar
  projection of the candidate rows, so the test asserts no `{` appears anywhere in the file and
  that all 120 columns come out in one pinned order. `pd.DataFrame(candidates)` writes five
  columns of `{'0': {...}}` and passes any assertion that only counts rows;
* **axis limits recomputed instead of recorded.** The figures return the limits they applied.
  One test replaces the plot writer with one that returns a sentinel pair and asserts
  `summary.json` carries *that* pair, which no recomputation can produce;
* **a report that answers from a constant.** The fixture is built so the honest answers are the
  surprising ones: the confidence control outranks persistence, the ten-way cut outranks the
  five-way cut, one of the winner's five severities ranks the *other* way at 0.039, and the
  winner is neither first nor last in candidate-key order. A report that hard-coded "persistence
  wins" or printed the macro AUROC five times fails;
* **a probability claim.** Every sentence of the report that mentions a probability or a chance
  is checked for a negation, so the one sentence allowed to use those words is the one that
  denies them;
* **a direction stated as a constant.** The deployment gate requires an orientation and never a
  `+1` one, so a candidate whose score falls as blur rises ranks like any other.
  `test_a_winner_that_falls_with_blur_is_reported_as_falling` builds one and reads the clauses
  that state a direction: a report that hard-coded "rises" prints that word beside a signed
  correlation of -0.657;
* **a recommendation the reader cannot reconcile.** The fixture's confidence control outranks
  the persistence candidate the report recommends, so the report has to say why the control is
  a yardstick rather than a second thing to deploy. Without that sentence a reader who does not
  know this codebase closes the file believing the worse option was recommended.
"""

import csv
import json
import os
import re

import pytest

from src.scene_uncertainty import corruption_plots as plots_module
from src.scene_uncertainty import corruption_reporting as reporting_module
from src.scene_uncertainty.corruption_plots import PLOT_FILENAMES
from src.scene_uncertainty.corruption_reporting import (
    CANDIDATE_KEY,
    DEPLOYABLE_SCORE_SCOPE,
    EXPECTED_FILES,
    FULL_TUNING_IMAGE_COUNT,
    PER_SCENE_KEYS,
    ROW_KEY,
    SELECTION_KEY,
    build_summary,
    rank_deployable_candidates,
    render_easy_report,
    summarize_candidates,
    validate_rows,
    validate_selected_queries,
    write_corruption_report,
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


def test_a_published_macro_auroc_weights_its_five_severities_equally():
    """Uneven coverage is reachable, and it must not tilt the number that orders candidates.

    A non-finite score is dropped from its own severity's group and the rest of that image's
    curve is kept, so the five comparisons behind `macro_auroc` are routinely made over groups
    of different sizes -- and `macro_auroc` is published for every candidate, ranked or not,
    and read by the report's control comparison. Here eight of ten images lose severity 3, and
    severity 3 is the one that ranks the *other* way: the equal-weighted macro is 0.8, while
    weighting by the corrupted group sizes reads 40/42 and weighting by clean-plus-corrupted
    reads 80/92, either of which would publish a candidate that fails at one blur level as a
    near-perfect detector. Every other fixture in this module covers all six severities
    equally, which makes the three readings identical and the choice unpinned.
    """
    curves = {}
    for image_id in range(10):
        offset = image_id * 0.1
        curve = [offset, 10 + offset, 20 + offset, -5 + offset, 40 + offset, 50 + offset]
        if image_id >= 2:
            curve[3] = NAN
        curves[image_id] = curve
    candidate = one_candidate(candidate_rows(curves))

    assert candidate["finite_count_severity_3"] == 2
    assert [candidate[f"finite_count_severity_{severity}"] for severity in (0, 1, 2, 4, 5)] == [
        10, 10, 10, 10, 10
    ]
    assert candidate["orientation"] == 1
    assert candidate["auroc_by_severity"] == {"1": 1.0, "2": 1.0, "3": 0.0, "4": 1.0, "5": 1.0}
    assert candidate["macro_auroc"] == pytest.approx(0.8)
    assert candidate["macro_auroc"] != pytest.approx(40 / 42)
    assert candidate["macro_auroc"] != pytest.approx(80 / 92)


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


# --- the published bundle ------------------------------------------------------------------


BUNDLE_SELECTED_COUNTS = {0: 30, 1: 29, 2: 28, 3: 27, 4: 26, 5: 25}
"""One selection size per severity, all six different, so the query-count range the report
prints (25 to 30) cannot come out right by reading a single severity or a constant."""

WINNER_CURVE = (0.0, 10.0, 20.0, -5.0, 40.0, 50.0)
"""The bulk of the winner's images: rising overall, dipping below clean at severity 3, and
ending well above clean. Severity 3 is what drags one of the five AUROCs down to a value a
coin toss would beat, so the five are not one number printed five times."""

DIP_CURVE = (1.0, 11.0, 21.0, 31.0, 41.0, -1.0)
"""Rises for four steps and then ends *below* its clean score: a positive trend that fails the
maximum-blur check. That combination is what pulls `dominant_direction_fraction` and
`max_blur_above_clean_rate` apart, and there is no single curve that has it.

Offset by one unit from `WINNER_CURVE` at every severity, so no two of the winner's populations
ever hold the same score at the same severity and every AUROC is decided by the curves rather
than by which image id happened to draw the larger lift."""

FALLING_CURVE = (60.0, 40.0, -5.0, 20.0, 10.0, -2.0)
"""The winner's only negative-trending images, at a rank correlation of -0.657 -- a larger
magnitude than the dip's +0.143. Taking absolute values therefore moves them past the dip
images in the ordering, which moves the median onto the other side of the boundary between the
two magnitudes: the signed median reads +0.143 and the absolute median 0.657. Without a
population like this the two are arithmetically equal on any fixture, because the median of
non-negative values is the value at the same position either way."""

WEAK_DECILE_CURVE = (0.0, 10.0, 20.0, -5.0, -6.0, 50.0)
QUINTILE_CURVE = (0.0, -5.0, -6.0, 30.0, 40.0, 50.0)
CONTROL_CURVE = tuple(0.9 - 0.05 * severity for severity in range(6))
CONTROL_QUINTILE_CURVE = tuple(0.5 - 0.05 * severity for severity in range(6))
"""The matched confidence control falls perfectly, so the reporter locks it at `-1` and it
earns a macro AUROC of 1.000 -- ahead of the persistence winner's 0.807. The comparison the
report has to make therefore comes out *against* persistence on this fixture, which is the only
way to tell a real comparison from a sentence that says persistence won.

The quintile control sits a long way below the decile one on purpose. The two are drawn on one
shared y-range and the two persistence figures on another, so a fixture whose two confidence
buckets overlapped would make a figure drawn on the wrong range look right."""

FLAT_SCORE = 5.0
"""A flat image's score at every severity: a complete curve that found no movement, which is a
measurement and not a missing one. `5.0` collides with no other population's value at any
severity, so those images tie only with themselves."""

WINNER_POPULATION = (
    (118, WINNER_CURVE),
    (110, DIP_CURVE),
    (12, FALLING_CURVE),
    (10, None),
)
"""The winner's 250 images in four populations, sized so that no two numbers the report's second
section prints come out equal.

That section prints six: two rank correlations, three counts and three rates. On a fixture where
every image follows one curve, four of them collide in pairs -- the absolute and signed medians
are arithmetically the same number, and the dominant-direction fraction and the maximum-blur
rate are both "the share of images that rose". A renderer that read the wrong field would pass.
Each population breaks one collision, and the sizes are what decide the two medians:

* 118 on `WINNER_CURVE`, rising and ending above clean;
* 110 on `DIP_CURVE`, rising but ending below clean, which separates the dominant direction
  (228/250 = 0.912) from the maximum-blur rate (118/250 = 0.472);
* 12 on `FALLING_CURVE`, the only negative trends, and the strongest magnitudes in the run;
* 10 flat, so the three direction counts are 228 / 12 / 10 and not 250 / 0 / 0.

The 22 negative and flat images are what move the median: signed, position 125.5 of 250 sits
inside the 110 dip images at +0.143; taking absolute values lifts those 22 out from under it and
the same position lands among the 0.657 magnitudes instead.
"""

WINNER_CURVES = [curve for size, curve in WINNER_POPULATION for _ in range(size)]
assert len(WINNER_CURVES) == FULL_TUNING_IMAGE_COUNT


def spread_curves(curve, *, count=FULL_TUNING_IMAGE_COUNT, step=0.001):
    """One curve per image, each lifted onto its own baseline.

    The lift is what makes these 250 different scenes rather than 250 copies of one: AUROC
    ranks scores from unrelated images against each other, and identical scenes would make
    every pair a tie. At most 0.249 over a 250-image run at the default step, and smaller than
    any gap between two populations' scores at one severity, so the AUROC each severity earns
    is decided by the curves and not by the offsets.
    """
    return {
        image: [value + image * step for value in curve] for image in range(count)
    }


def winner_curves(count=FULL_TUNING_IMAGE_COUNT, step=0.001):
    """The deployable winner's images: `WINNER_POPULATION`, in that order, lifted as above."""
    curves = {}
    for image in range(count):
        curve = WINNER_CURVES[image]
        curves[image] = (
            [FLAT_SCORE] * 6 if curve is None
            else [value + image * step for value in curve]
        )
    return curves


def bundle_rows(count=FULL_TUNING_IMAGE_COUNT):
    """Seven candidates over one run: three deployable, and four measured and excluded.

    The winner is `decile_00_10`, which is neither the first nor the last candidate in
    `CANDIDATE_KEY` order -- the confidence control sorts before it and the quintile after it --
    so a report that led with `candidates[0]` or with the last row would name the wrong one.
    The frozen twin carries the winner's confidence bin and every one of its numbers under a
    different membership mode, so a section that identified the winner by its bin alone would
    be free to describe the twin instead -- and the twin is not deployable.
    """
    short_count = count * 4 // 5
    rows = candidate_rows(
        winner_curves(count=count), selected_count=BUNDLE_SELECTED_COUNTS,
    )
    rows += candidate_rows(
        spread_curves(WEAK_DECILE_CURVE, count=count),
        selected_count=BUNDLE_SELECTED_COUNTS, confidence_bin="decile_10_20",
    )
    rows += candidate_rows(
        spread_curves(QUINTILE_CURVE, count=count),
        selected_count=BUNDLE_SELECTED_COUNTS,
        bucket_scheme="quintile", confidence_bin="quintile_00_20",
    )
    rows += candidate_rows(
        spread_curves(CONTROL_CURVE, count=count, step=0.0001),
        selected_count=BUNDLE_SELECTED_COUNTS,
        signal="confidence", score_scope="confidence",
    )
    rows += candidate_rows(
        spread_curves(CONTROL_QUINTILE_CURVE, count=count, step=0.0001),
        selected_count=BUNDLE_SELECTED_COUNTS,
        bucket_scheme="quintile", confidence_bin="quintile_00_20",
        signal="confidence", score_scope="confidence",
    )
    rows += candidate_rows(
        winner_curves(count=count),
        selected_count=BUNDLE_SELECTED_COUNTS, membership_mode="frozen",
    )
    # A candidate that never reached part of the run: 200 images of 250, and one of those 200
    # missing its worst blur level. Its three coverage numbers are therefore three different
    # numbers -- 250 expected, 200 with rows, 199 with a complete curve -- so a range taken
    # from the wrong one shows, and `measured_fraction` reads 0.995 rather than the 1.0 it
    # would read if every image it has were complete.
    short = spread_curves(WEAK_DECILE_CURVE, count=short_count)
    short[0][5] = None
    rows += candidate_rows(
        short, selected_count=BUNDLE_SELECTED_COUNTS, confidence_bin="decile_20_30",
    )
    return rows


BUNDLE_ROW_COUNT = 6 * FULL_TUNING_IMAGE_COUNT * 6 + (FULL_TUNING_IMAGE_COUNT * 4 // 5) * 6 - 1


def bundle_diagnostics(count=FULL_TUNING_IMAGE_COUNT):
    """What `analyze_corruption_sensitivity` hands the writer: run metadata and padding.

    Two of the images carry padding and they differ from each other in both published ways --
    one detected a different tail at different severities and one did not -- so every number in
    the padding rollup is a different number: 2 padded images of `count`, 5 padded queries in
    total, 3 at most, and 1 padded image whose tail held still.
    """
    images = {}
    for image in range(count):
        if image == 0:
            tails = {
                str(severity): [297, 298, 299][: 2 + severity % 2] for severity in range(6)
            }
        elif image == 1:
            tails = {str(severity): [298, 299] for severity in range(6)}
        else:
            tails = {str(severity): [] for severity in range(6)}
        union = sorted({query for ids in tails.values() for query in ids})
        images[str(image)] = {
            "union_padded_query_ids": union,
            "union_padded_count": len(union),
            "padded_query_ids_by_severity": tails,
            "padded_count_by_severity": {
                key: len(value) for key, value in tails.items()
            },
            "tail_identical_across_severities": (
                len({tuple(value) for value in tails.values()}) == 1
            ),
        }
    return {
        "run": {
            "artifact_type": "scene_corruption_sensitivity_inputs",
            "feature_cache_id": "cache-df79fddf",
            "source_result_id": "result-353a5992",
            "bank_id": "bank-5dfd6e50",
            "normalization": "l2_row",
            "k": 5,
            "source_partition": "tuning",
            "severities": [0, 1, 2, 3, 4, 5],
            "decoder_layers": [0, 1, 2],
            "query_count": 300,
            "image_count": count,
            "record_count": count * 6,
        },
        "images": images,
    }


FIXED_LIMITS = {"persistence": [-1.5, 52.0], "confidence": [0.14, 0.91]}
"""Stand-in axis limits for the report tests, which do not need the figures drawn. Deliberately
not the limits this fixture's candidates would produce, so a report that recomputed them rather
than reading the summary would print different numbers."""


def report_summary(count=FULL_TUNING_IMAGE_COUNT, rows=None):
    """The summary the writer builds, without spending the four figures to get it."""
    rows = bundle_rows(count=count) if rows is None else rows
    per_scene, candidates, ranking = summarize_candidates(rows, expected_image_count=count)
    return build_summary(
        bundle_diagnostics(count=count),
        candidates,
        ranking,
        axis_limits=FIXED_LIMITS,
        scored_row_count=len(rows),
        per_scene_row_count=len(per_scene),
    )


def flat(text):
    """The report with its line wrapping removed, so an assertion is about the words."""
    return " ".join(text.split())


def section(report, title):
    """One `## ` section, flattened -- so a claim asserted about the lead cannot be satisfied
    by a sentence three sections further down."""
    heading = f"## {title}"
    assert heading in report, f"no section titled {title!r}"
    return flat(report.split(heading, 1)[1].split("\n## ", 1)[0])


def sentences(text):
    return [part for part in re.split(r"(?<=[.!?])\s+", flat(text)) if part]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    """One published bundle, written once: the four figures are the slow part of this file."""
    output = tmp_path_factory.mktemp("published") / "corruption"
    summary = write_corruption_report(
        output, score_rows=bundle_rows(), diagnostics=bundle_diagnostics()
    )
    return output, summary


def test_the_writer_and_the_figures_agree_on_the_names_they_share():
    """Three strings each module spells for itself, because the plots import the reporter.

    The eight filenames are pinned as literals -- the bundle's contract with whoever reads it --
    and then the four figure names, the control's signal and the name of the wider bucket scheme
    are checked against the module that actually writes and draws them. Without the second half,
    `FIGURE_FILES` could name a file `corruption_plots` never writes and only the file-set check
    at the end of a real run would notice.
    """
    assert reporting_module.CONTROL_SIGNAL == plots_module.CONFIDENCE_SIGNAL
    assert reporting_module.WIDER_BUCKET_SCHEME in plots_module.SCHEME_BUCKET_NAMES
    assert EXPECTED_FILES == {
        "per_scene.csv",
        "candidate_metrics.csv",
        "summary.json",
        "persistence_actual_distance_deciles.png",
        "persistence_actual_distance_quintiles.png",
        "confidence_actual_distance_deciles.png",
        "confidence_actual_distance_quintiles.png",
        "easy-report.md",
    }
    assert set(PLOT_FILENAMES.values()) < EXPECTED_FILES


def test_a_published_bundle_holds_exactly_the_eight_expected_files(bundle):
    output, _ = bundle

    assert {path.name for path in output.iterdir()} == EXPECTED_FILES
    for name in PLOT_FILENAMES.values():
        assert (output / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_per_scene_table_is_every_scored_row_in_the_documented_column_order(bundle):
    output, _ = bundle
    table = read_csv(output / "per_scene.csv")

    assert table[0] == list(PER_SCENE_KEYS)
    assert len(table) == 1 + BUNDLE_ROW_COUNT


def test_no_nested_dictionary_reaches_a_cell_of_the_candidate_table(bundle):
    """Ruling 1: the CSV is the scalar projection, `summary.json` keeps the nested view.

    `pd.DataFrame(candidates).to_csv(...)` writes five columns of `{'0': {'count': 250, ...}}`
    and produces a file with the right number of rows in it, so the row count is not the check
    -- the absence of a brace anywhere in the file is.
    """
    output, summary = bundle
    path = output / "candidate_metrics.csv"
    header, *body = read_csv(path)

    assert "{" not in path.read_text(encoding="utf-8")
    assert len(body) == len(summary["candidates"]) == 7
    for nested in (
        "auroc_by_severity", "severity_statistics", "coverage_by_severity",
        "selected_count_by_severity", "median_clean_overlap_by_severity",
    ):
        assert nested not in header
        assert nested in summary["candidates"][0]


def test_the_candidate_columns_are_the_scalar_projection_in_one_fixed_order(bundle):
    """Pinned as a literal rather than derived from a candidate row.

    A header compared against "the non-dictionary keys of this candidate" is a comparison of
    the projection with itself: it passes whatever order the reporter happens to build, which
    is the property being asserted.
    """
    output, _ = bundle
    header, *_ = read_csv(output / "candidate_metrics.csv")

    assert header == [
        "signal", "bucket_scheme", "confidence_bin", "membership_mode", "padding_mode",
        "aggregation", "score_scope",
        "expected_image_count", "image_count", "measured_count", "missing_count",
        "positive_count", "negative_count", "flat_count",
        "measured_fraction", "missing_fraction", "positive_fraction", "negative_fraction",
        "flat_fraction", "dominant_direction_fraction",
        "median_signed_spearman", "median_absolute_spearman", "orientation",
        "oriented_adjacent_consistency", "max_blur_above_clean_rate", "macro_auroc",
        "deployable",
        "padding_counterpart_padding_mode", "padding_paired_score_count",
        "padding_changed_score_count", "padding_median_unfiltered_minus_filtered_score",
        *(f"auroc_severity_{severity}" for severity in range(1, 6)),
        *(f"score_count_severity_{severity}" for severity in range(6)),
        *(f"score_mean_severity_{severity}" for severity in range(6)),
        *(f"score_variance_severity_{severity}" for severity in range(6)),
        *(f"score_median_severity_{severity}" for severity in range(6)),
        *(f"score_q25_severity_{severity}" for severity in range(6)),
        *(f"score_q75_severity_{severity}" for severity in range(6)),
        *(f"image_count_severity_{severity}" for severity in range(6)),
        *(f"finite_count_severity_{severity}" for severity in range(6)),
        *(f"clean_count_severity_{severity}" for severity in range(6)),
        *(f"corrupted_count_severity_{severity}" for severity in range(6)),
        *(f"selected_count_min_severity_{severity}" for severity in range(6)),
        *(f"selected_count_median_severity_{severity}" for severity in range(6)),
        *(f"selected_count_max_severity_{severity}" for severity in range(6)),
        *(f"median_clean_overlap_severity_{severity}" for severity in range(6)),
    ]
    assert len(header) == 120


def test_each_candidate_row_of_the_csv_holds_what_summary_json_publishes(bundle):
    """The two files are two views of one list, not two computations over the same rows."""
    output, summary = bundle
    with (output / "candidate_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    winner = summary["deployable_ranking"][0]
    published = next(
        row for row in rows
        if all(row[field] == str(winner[field]) for field in CANDIDATE_KEY)
    )

    assert float(published["macro_auroc"]) == pytest.approx(winner["macro_auroc"])
    assert float(published["auroc_severity_3"]) == pytest.approx(winner["auroc_severity_3"])
    assert int(published["flat_count"]) == winner["flat_count"] == 10
    assert published["deployable"] == "True"
    assert float(published["score_q75_severity_5"]) == pytest.approx(
        winner["severity_statistics"]["5"]["q75"]
    )


def test_summary_json_is_written_with_its_keys_sorted(bundle):
    """Insertion order would put `run` first; sorted order puts `axis_limits` first."""
    output, summary = bundle
    text = (output / "summary.json").read_text(encoding="utf-8")
    top_level = [
        line.split('"')[1] for line in text.splitlines() if line.startswith('  "')
    ]

    assert top_level == [
        "axis_limits", "candidates", "configuration", "deployable_ranking", "labels",
        "padding", "run", "validation",
    ]
    assert text == json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n"
    assert json.loads(text) == summary


def test_the_summary_carries_the_run_provenance_exactly_as_the_analysis_recorded_it(bundle):
    _, summary = bundle

    assert summary["run"] == bundle_diagnostics()["run"]


def test_the_summary_counts_what_was_read_against_what_came_out(bundle):
    _, summary = bundle
    validation = summary["validation"]

    assert validation["scored_row_count"] == BUNDLE_ROW_COUNT
    assert validation["per_scene_row_count"] == BUNDLE_ROW_COUNT
    assert validation["duplicate_row_key_count"] == 0
    assert validation["candidate_count"] == 7
    assert validation["deployable_candidate_count"] == 3
    assert validation["expected_image_count"] == FULL_TUNING_IMAGE_COUNT
    # From `image_count` and `measured_count` against `expected_image_count`, never from
    # `measured_fraction`, which is over the images a candidate has rows for and says nothing
    # about the 50 the short candidate never reached.
    assert validation["candidate_image_count_range"] == [200, 250]
    assert validation["candidate_measured_count_range"] == [199, 250]


def test_the_summary_rolls_the_padding_record_up_and_still_carries_it_whole(bundle):
    _, summary = bundle
    padding = summary["padding"]

    assert padding["image_count"] == FULL_TUNING_IMAGE_COUNT
    assert padding["images_with_padding"] == 2
    assert padding["total_union_padded_count"] == 5
    assert padding["max_union_padded_count"] == 3
    assert padding["padded_images_with_identical_tails"] == 1
    assert padding["images"]["0"] == bundle_diagnostics()["images"]["0"]


def test_the_summary_records_the_gates_it_applied_and_the_labels_the_run_carried(bundle):
    _, summary = bundle

    assert summary["configuration"] == {
        "full_tuning_image_count": FULL_TUNING_IMAGE_COUNT,
        "deployable_signal": "persistence",
        "deployable_membership_mode": "dynamic",
        "deployable_padding_mode": "filtered",
        "deployable_score_scope": DEPLOYABLE_SCORE_SCOPE,
        "severities": [0, 1, 2, 3, 4, 5],
    }
    assert summary["labels"] == {
        "signal": ["confidence", "persistence"],
        "bucket_scheme": ["decile", "quintile"],
        "confidence_bin": [
            "decile_00_10", "decile_10_20", "decile_20_30", "quintile_00_20"
        ],
        "membership_mode": ["dynamic", "frozen"],
        "padding_mode": ["filtered"],
        "aggregation": ["q90"],
        "score_scope": ["confidence", "layer_2"],
    }


def test_the_summary_records_the_limits_the_figures_used_rather_than_computing_its_own(
    tmp_path, monkeypatch
):
    """Ruling 6: the figures return what they applied, and the summary carries that object.

    The sentinel is a pair no computation over these candidates produces, so a `build_summary`
    that called `shared_limits` again -- and drifted from the figures the moment either side
    changed -- cannot pass.
    """
    drawn = plots_module.write_corruption_plots
    sentinel = {"persistence": [-11.0, 111.0], "confidence": [-0.25, 0.75]}

    def stamped(directory, candidates):
        drawn(directory, candidates)
        return sentinel

    monkeypatch.setattr(plots_module, "write_corruption_plots", stamped)
    output = tmp_path / "corruption"
    summary = write_corruption_report(
        output, score_rows=bundle_rows(count=4), diagnostics=bundle_diagnostics(count=4)
    )

    assert summary["axis_limits"] == sentinel
    assert json.loads((output / "summary.json").read_text())["axis_limits"] == sentinel


# --- refusing to publish half a bundle ------------------------------------------------------


def test_a_published_bundle_is_no_less_readable_than_a_directory_made_beside_it(tmp_path):
    """`mkdtemp` makes its directory `0o700` and `os.replace` carries that mode onto the bundle.

    The result is a results directory nobody but the owner can list, holding eight files the
    umask made group-readable -- and Task 9 hands this directory to an operator. The umask is
    pinned for the duration so the assertion cannot pass by accident on a machine whose own
    umask is `0o077`, where a `0o700` bundle and a correct one are the same number.
    """
    previous = os.umask(0o022)
    try:
        reference = tmp_path / "reference"
        reference.mkdir()
        output = tmp_path / "corruption"
        write_corruption_report(
            output, score_rows=bundle_rows(count=3), diagnostics=bundle_diagnostics(count=3)
        )
        published = output.stat().st_mode & 0o777
        expected = reference.stat().st_mode & 0o777
    finally:
        os.umask(previous)

    assert published == expected == 0o755
    assert {path.name for path in output.iterdir()} == EXPECTED_FILES


def test_an_existing_output_directory_is_refused_and_left_alone(tmp_path):
    output = tmp_path / "corruption"
    output.mkdir()
    (output / "keep.txt").write_text("an earlier run", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output already exists"):
        write_corruption_report(
            output, score_rows=bundle_rows(count=3), diagnostics=bundle_diagnostics(count=3)
        )

    assert [path.name for path in output.iterdir()] == ["keep.txt"]
    assert {path.name for path in tmp_path.iterdir()} == {"corruption"}


def test_a_write_that_fails_part_way_leaves_neither_the_bundle_nor_a_staging_directory(
    tmp_path
):
    """The figures refuse a run with no drawable confidence control -- after two CSVs exist.

    The parent is asserted empty rather than the output absent: a staging directory left
    beside it is what a half-written run looks like, and it is named `.corruption.<random>`,
    which nothing about `output.exists()` can see.
    """
    output = tmp_path / "corruption"
    rows = [row for row in bundle_rows(count=3) if row["signal"] != "confidence"]

    with pytest.raises(ValueError, match="no drawable confidence candidate"):
        write_corruption_report(
            output, score_rows=rows, diagnostics=bundle_diagnostics(count=3)
        )

    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_a_ninth_file_in_the_staging_directory_stops_the_bundle_being_published(
    tmp_path, monkeypatch
):
    """The file set is checked *before* the rename, which only a failing check can show."""
    drawn = plots_module.write_corruption_plots

    def with_an_extra_figure(directory, candidates):
        limits = drawn(directory, candidates)
        (directory / "persistence_actual_distance_thirds.png").write_bytes(b"")
        return limits

    monkeypatch.setattr(plots_module, "write_corruption_plots", with_an_extra_figure)
    output = tmp_path / "corruption"

    with pytest.raises(RuntimeError, match="report bundle contains"):
        write_corruption_report(
            output, score_rows=bundle_rows(count=3), diagnostics=bundle_diagnostics(count=3)
        )

    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_a_missing_figure_stops_the_bundle_being_published(tmp_path, monkeypatch):
    drawn = plots_module.write_corruption_plots

    def without_one_figure(directory, candidates):
        limits = drawn(directory, candidates)
        (directory / "confidence_actual_distance_quintiles.png").unlink()
        return limits

    monkeypatch.setattr(plots_module, "write_corruption_plots", without_one_figure)
    output = tmp_path / "corruption"

    with pytest.raises(RuntimeError, match="report bundle contains"):
        write_corruption_report(
            output, score_rows=bundle_rows(count=3), diagnostics=bundle_diagnostics(count=3)
        )

    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_a_summary_that_is_not_valid_json_fails_the_write_rather_than_the_reader(
    tmp_path, monkeypatch
):
    """`NaN` is not JSON. Written anyway it lands in the file and breaks a reader long after
    the run; refused, it fails the write and the bundle never appears."""
    monkeypatch.setattr(
        reporting_module, "build_summary", lambda *args, **kwargs: {"broken": float("nan")}
    )
    output = tmp_path / "corruption"

    with pytest.raises(ValueError, match="not JSON compliant"):
        write_corruption_report(
            output, score_rows=bundle_rows(count=3), diagnostics=bundle_diagnostics(count=3)
        )

    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


# --- the plain-language report --------------------------------------------------------------


def test_the_published_report_is_the_rendered_summary(bundle):
    """Licenses every test below to render from a summary instead of publishing a bundle."""
    output, summary = bundle
    published = (output / "easy-report.md").read_text(encoding="utf-8")

    assert published == render_easy_report(summary)
    assert "decile_00_10" in published


def test_the_report_asks_the_six_questions_in_order():
    report = render_easy_report(report_summary())

    assert [line for line in report.splitlines() if line.startswith("## ")] == [
        "## What should we deploy?",
        "## Does it react steadily to blur?",
        "## Did 20% buckets help?",
        "## Did persistence beat model confidence?",
        "## What the files contain",
        "## What this does not prove",
    ]


def test_the_report_leads_with_the_best_deployable_layer_2_candidate():
    """Not `candidates[0]`, which is the confidence control, and not the frozen twin, which
    carries the winner's bin and every one of its numbers."""
    deploy = section(render_easy_report(report_summary()), "What should we deploy?")

    assert "`decile_00_10`" in deploy
    assert "`decile_10_20`" not in deploy
    assert "`quintile_00_20`" not in deploy
    assert "frozen" not in deploy
    for label in ("`persistence`", "`layer_2`", "`dynamic`", "`filtered`", "`q90`"):
        assert label in deploy
    assert "first of 3 candidates" in deploy
    assert "between 25 and 30 selected queries" in deploy


def test_the_selection_size_sentence_is_read_from_the_data_in_both_shapes():
    """No invented magnitude, and no "between 4 and 4" when the two ends coincide.

    Two defects in one sentence, and the same defect class as the hard-coded "rises" this
    section used to carry: it ended "summarises a few dozen queries an image", a number never
    read from anything, which on a small run printed beside "between 4 and 4 selected queries".
    A few dozen is right for the pilot's 300-query images and wrong for every diagnostic run,
    and neither half of it was checkable by a reader.

    Both shapes are asserted from one candidate set that differs only in `selected_count`, so a
    renderer that formatted the equal case correctly by dropping the range entirely fails the
    first half, and one that kept "a few dozen" beside either fails both.
    """
    varying = section(
        render_easy_report(
            report_summary(rows=candidate_rows(
                winner_curves(), selected_count=BUNDLE_SELECTED_COUNTS
            ))
        ),
        "What should we deploy?",
    )
    constant = section(
        render_easy_report(
            report_summary(rows=candidate_rows(winner_curves(), selected_count=4))
        ),
        "What should we deploy?",
    )

    assert "scored over between 25 and 30 selected queries" in varying
    assert "scored over exactly 4 selected queries" in constant
    assert "between 4 and 4" not in constant
    for report in (varying, constant):
        assert "few dozen" not in report
        assert "summarises those queries rather than the whole image" in report


def test_a_one_image_run_reads_in_the_singular_wherever_it_counts_images():
    """The small-run branch on the smallest run there is, where every plural is wrong.

    `analyze-corruption-sensitivity` is run on tiny caches as a diagnostic -- that is the whole
    reason `_nothing_ranked_lines` exists -- so "a candidate measured on 1 images" and "1 of 1
    images carried" are sentences this command really prints, in the report written for the
    reader least able to discount them. `_count` already existed for exactly this and two call
    sites were not using it.
    """
    rows = candidate_rows({0: RISING}, selected_count=4)
    per_scene, candidates, ranking = summarize_candidates(rows, expected_image_count=1)
    summary = build_summary(
        bundle_diagnostics(count=1), candidates, ranking, axis_limits=FIXED_LIMITS,
        scored_row_count=len(rows), per_scene_row_count=len(per_scene),
    )
    report = flat(render_easy_report(summary))

    assert ranking == []
    assert summary["padding"] == {**summary["padding"], "image_count": 1}
    assert "This run covered 1 image of the" in report
    assert "1 of 1 image carried repeated decoder placeholder queries" in report
    assert "This run declared 1 image, fewer than 250" in report
    assert "a candidate measured on 1 image is a different measurement" in report
    assert "1 images" not in report


def test_the_report_gives_all_five_severity_aurocs_beside_the_macro():
    """The whole row, pinned in position: five severities and the macro they average to.

    Pinned as the rendered row rather than as six separate substrings, so a renderer that
    printed the macro in every cell, or the five in the wrong order, fails on the assertion
    rather than on a count that four identical values would satisfy.
    """
    deploy = section(render_easy_report(report_summary()), "What should we deploy?")

    assert "| AUROC | 0.951 | 0.906 | 0.502 | 0.951 | 0.487 | 0.759 |" in deploy
    # Severity 5 is the one a coin toss would beat, and the sentence under the table says so.
    assert "worst is 5, at 0.487" in deploy


def test_the_report_explains_auroc_as_an_ordering_and_never_as_a_probability():
    report = flat(render_easy_report(report_summary()))

    assert (
        "An AUROC of 0.80 means that when we randomly pick one clean image and one blurred "
        "image, the score puts the blurred image higher about 80 times out of 100. It does not "
        "mean the image has an 80% chance of being corrupted."
    ) in report
    spoken = [
        sentence for sentence in sentences(report)
        if "probab" in sentence.lower() or "chance" in sentence.lower()
    ]
    assert spoken
    for sentence in spoken:
        assert re.search(r"\b(not|never|no|nothing)\b", sentence.lower()), sentence


def test_the_report_reads_the_strength_number_beside_the_direction_counts():
    """Six numbers, six different values, so reading the wrong field is visible.

    Absolute rank correlation says how hard blur moves the score and the signed median says
    which way; the counts are the other half of the sentence. `WINNER_POPULATION` is built so
    that the absolute median (0.657) differs from the signed one (+0.143), the direction counts
    (228 / 12 / 10) differ from each other, and the dominant-direction fraction (0.912), the
    adjacent-step rate (0.779) and the maximum-blur rate (0.472) are three separate values.
    """
    steady = section(
        render_easy_report(report_summary()), "Does it react steadily to blur?"
    )

    assert "correlation of 0.657" in steady
    assert "here it is +0.143" in steady
    assert "228 of the 250 measured images rose with blur, 12 fell, and 10 were flat" in steady
    assert "That puts 0.912 of the measured images" in steady
    assert "0.779 of the five steps" in steady
    assert "on 0.472 of those images" in steady
    assert "A flat image is a complete curve of six identical scores" in steady


def test_the_report_compares_the_ten_way_cut_with_the_five_way_cut():
    """And answers its own question: on this fixture the wider buckets rank behind."""
    buckets = section(render_easy_report(report_summary()), "Did 20% buckets help?")

    assert buckets.startswith("On this run they did not help.")
    assert "`decile_00_10`" in buckets and "0.759" in buckets
    assert "`quintile_00_20`" in buckets and "0.600" in buckets
    assert "a gap of 0.159" in buckets


def test_the_report_compares_persistence_with_the_control_that_shares_its_selection():
    """On this fixture the control wins, so a sentence that says persistence did cannot pass."""
    control = section(
        render_easy_report(report_summary()), "Did persistence beat model confidence?"
    )

    assert "the control comes out ahead" in control
    assert "1.000" in control and "0.759" in control and "0.657" in control
    for label in ("`decile`", "`decile_00_10`", "`dynamic`", "`filtered`", "`q90`"):
        assert label in control
    assert "same selected queries" in control
    assert "share no unit" in control
    # The reader is told, in the section that shows the control winning, why that is not a
    # recommendation to deploy the control.
    assert "yardstick and not a second thing to deploy" in control
    assert "never selected and never ranked" in control


def test_a_winner_with_no_matched_control_says_so_rather_than_pairing_a_stranger():
    """The control is one candidate or none: the winner's own frozen twin is not a control.

    Every confidence row is taken out, leaving the winner, its frozen twin and the other
    persistence candidates. A lookup that matched on the confidence bin, or on everything but
    the membership mode, has something to return here and would report the twin's numbers as a
    comparison between two signals.
    """
    rows = [row for row in bundle_rows() if row["signal"] != "confidence"]
    control = section(
        render_easy_report(report_summary(rows=rows)),
        "Did persistence beat model confidence?",
    )

    assert "scored for persistence only" in control
    assert "0.759" not in control


def test_the_report_says_the_wider_buckets_did_help_when_they_come_out_ahead():
    """The other side of the same sentence, which a constant answer cannot produce.

    Dropping the winning decile candidate leaves the quintile candidate at the head of the
    ranking -- it ties the remaining decile on macro AUROC and takes the first tie-break -- so
    the honest answer flips. Without this, "On this run they did not help" is a string that
    happens to be right on one fixture.
    """
    rows = [
        row for row in bundle_rows()
        if not (
            row["signal"] == "persistence"
            and row["confidence_bin"] == "decile_00_10"
            and row["membership_mode"] == "dynamic"
        )
    ]
    summary = report_summary(rows=rows)
    buckets = section(render_easy_report(summary), "Did 20% buckets help?")

    assert summary["deployable_ranking"][0]["bucket_scheme"] == "quintile"
    assert buckets.startswith("On this run they did help.")
    assert "did not help" not in buckets


def test_one_scheme_with_a_ranked_candidate_is_reported_as_uncomparable():
    """A comparison of one is not a comparison, and the quintile row would supply the number."""
    rows = [row for row in bundle_rows() if row["bucket_scheme"] != "quintile"]
    buckets = section(render_easy_report(report_summary(rows=rows)), "Did 20% buckets help?")

    assert "Only the `decile` cut" in buckets
    assert "`decile_00_10`" in buckets and "0.759" in buckets
    assert "quintile" not in buckets

FALLING_WINNER_CURVE = tuple(reversed(WINNER_CURVE))
"""A deployable candidate whose score falls as blur rises, which the gate allows.

`_candidate_metrics` requires an orientation, never a `+1` one, so a candidate read downward
passes every gate a candidate read upward passes -- and the design keeps the figures un-oriented
precisely so that a useful decreasing signal stays visibly decreasing. Reversing the winner's
curve keeps its magnitudes and flips only its direction, so the two fixtures differ in the one
thing the report has to read rather than assume.
"""


def falling_winner_rows(count=FULL_TUNING_IMAGE_COUNT):
    """One candidate, ranked first, read downward: `orientation` is `-1` on every image."""
    return candidate_rows(
        spread_curves(FALLING_WINNER_CURVE, count=count),
        selected_count=BUNDLE_SELECTED_COUNTS,
    )


def test_a_winner_that_falls_with_blur_is_reported_as_falling():
    """Every clause that states a direction is read from the orientation, not assumed.

    The gate at `_candidate_metrics` requires an orientation and not a `+1` one, so this
    candidate -- 250 images, complete coverage, `persistence` at `layer_2` -- ranks first with
    `orientation: -1`. A report that stated the direction as a constant prints "the score rises
    as blur gets worse" directly beside a signed correlation of -0.657, and tells a reader the
    opposite of what was measured. The flat gloss is checked here too: this fixture has no flat
    image, so the sentence defining one has nothing to define.
    """
    summary = report_summary(rows=falling_winner_rows())
    winner = summary["deployable_ranking"][0]
    report = render_easy_report(summary)
    deploy = section(report, "What should we deploy?")
    steady = section(report, "Does it react steadily to blur?")

    assert winner["orientation"] == -1
    assert winner["median_signed_spearman"] == pytest.approx(-0.6571428571428573)
    assert winner["flat_count"] == 0

    assert "read downward" in deploy
    assert "upward" not in deploy
    assert "-0.657" in steady
    assert "so the score falls as blur gets worse" in steady
    assert "rises as blur gets worse" not in steady
    assert "ends below its clean score" in steady
    assert "ends above its clean score" not in steady
    assert "A flat image is" not in steady
    # One candidate, so the counts and their nouns have to agree: this is the run shape whose
    # report most needs to read like prose, and the only one where the `s` is wrong.
    assert "first of 1 candidate that passed every gate" in deploy
    assert "1 candidates" not in report


def test_the_report_says_what_each_file_holds_and_which_axes_are_shared():
    files = section(render_easy_report(report_summary()), "What the files contain")

    for name in (*PLOT_FILENAMES.values(), "per_scene.csv", "candidate_metrics.csv",
                 "summary.json"):
        assert name in files
    assert "-1.500" in files and "52.000" in files
    assert "0.140" in files and "0.910" in files


def test_the_report_says_nothing_was_calibrated_and_no_held_out_data_was_used():
    limits = section(render_easy_report(report_summary()), "What this does not prove")

    assert "calibrat" in limits
    assert "held-out" in limits
    assert "250" in limits


def test_the_report_carries_no_latex_and_no_terminal_unfriendly_notation():
    report = render_easy_report(report_summary())

    assert report.isascii()
    for token in ("$", "\\", "^", "_{", "~"):
        assert token not in report


def test_a_run_smaller_than_the_full_tuning_set_publishes_an_empty_ranking_and_says_so(
    tmp_path
):
    """The path every synthetic and diagnostic run takes: eight files, no recommendation.

    The output's parent does not exist either, which is the other half of what the writer
    promises about a directory nobody has created yet.
    """
    output = tmp_path / "runs" / "corruption"
    summary = write_corruption_report(
        output, score_rows=bundle_rows(count=3), diagnostics=bundle_diagnostics(count=3)
    )
    report = (output / "easy-report.md").read_text(encoding="utf-8")
    deploy = section(report, "What should we deploy?")

    assert {path.name for path in output.iterdir()} == EXPECTED_FILES
    assert summary["deployable_ranking"] == []
    assert summary["validation"]["deployable_candidate_count"] == 0
    assert [candidate["deployable"] for candidate in summary["candidates"]] == [False] * 7
    assert "No candidate passed the full tuning-coverage gate" in deploy
    assert "all 250 images" in deploy
    assert "This run declared 3 images, fewer than 250" in deploy
    assert "`decile_00_10`" not in deploy
