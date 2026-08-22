"""What the three controls, the paired bootstrap and the deployment ranking are held to.

Two fixtures do most of the work here and they answer different questions.

`prepared` builds the shared six-image bundle every task from 3 onwards reads, and is what says
the controls survive contact with the real arm table: 21 reference controls, 45 persistence
candidates, 36 confidence twins, and a redundancy verdict that comes out both ways. What it
*cannot* say anything about is which series a lookup matched, because
`contrast_test_utils.default_score` ignores its `aggregation` argument -- so all three summaries
of one bin are byte-identical and a control matched to the wrong summary reads exactly like one
matched to the right one.

`matched_rows` is the fixture for that. It is one arm and its twin at two summaries and two
methods, with twelve pairwise-distinct macro AUROCs by construction, so every one of the four
attached comparisons can only be satisfied by the series it names. It uses the `combined`
differential arm on purpose, because that is the one arm whose name is not its pair name.

Neither fixture is asked for a number that moves with the roster. The margins between a
candidate and its twin shrink as images are added -- nine confidence candidates sit below chance
at six images and drift towards it at 250 -- so what is pinned here is either an exact
hand-computable value on a hand-built curve, an identity between two things the code computes
separately, or the *side* of a comparison.
"""
import pytest

from src.scene_uncertainty import contrast_controls
from src.scene_uncertainty.contrast_analysis import (
    CONTRAST_ROW_KEY,
    build_contrast_rows,
    summarize_contrast_candidates,
)
from src.scene_uncertainty.contrast_controls import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    REFERENCE_CONTROL_METHOD,
    attach_controls,
    beats_both_inputs,
    index_curves,
    is_confidence_redundant,
    paired_macro_bootstrap,
    rank_contrast_candidates,
    reference_control_rows,
)
from src.scene_uncertainty.contrast_inputs import FULL_TUNING_IMAGE_COUNT, load_contrast_inputs
from src.scene_uncertainty.contrast_scores import SCORE_METHODS
from src.scene_uncertainty.corruption_metrics import severity_aurocs

from tests.scene_uncertainty.contrast_test_utils import write_source_bundle

IMAGES = 6

FAST_SAMPLES = 200
"""What `prepared` bootstraps with, against the production 2,000.

The sample count changes how wide an interval is and nothing else -- the point estimate is the
unresampled difference and every verdict this file checks is decided by a sign -- so 200 buys
back three quarters of a second per call across a dozen tests without weakening anything.
`test_the_full_two_thousand_sample_bootstrap_runs_in_reasonable_time` and
`test_the_sample_count_is_read_from_the_module_at_call_time` are where the real number is
exercised.
"""

CONTROL_FIELDS = (
    "twin_arm", "twin_macro_auroc", "twin_auroc_by_severity", "twin_orientation",
    "responsive_control_macro_auroc", "reference_control_macro_auroc",
    "responsive_control_macro_difference", "responsive_control_auroc_difference_by_severity",
    "reference_control_macro_difference", "reference_control_auroc_difference_by_severity",
    "twin_macro_difference", "twin_auroc_difference_by_severity",
    "beats_both_inputs", "confidence_redundant",
    "responsive_control_bootstrap", "twin_bootstrap",
)
"""Every field `attach_controls` writes. Task 8 projects exactly these into the CSV and JSON.

Listed here rather than derived from a candidate, so a field that stopped being written is a
failure and not a quietly shorter list.
"""


def loaded(tmp_path, **kwargs):
    source = write_source_bundle(tmp_path / "source", **kwargs)
    return load_contrast_inputs(source, expected_image_count=IMAGES)


def prepared(tmp_path, *, samples=FAST_SAMPLES, **kwargs):
    rows, _ = build_contrast_rows(loaded(tmp_path, **kwargs))
    candidates = summarize_contrast_candidates(rows, expected_image_count=IMAGES)
    controls = summarize_contrast_candidates(
        reference_control_rows(rows), expected_image_count=IMAGES
    )
    attach_controls(candidates, controls, rows, samples=samples)
    return candidates, controls, rows


def stub(**overrides):
    """A minimal candidate dictionary for ranking tests."""
    candidate = {
        "arm": "decile_00_10__50_60", "signal": "persistence", "aggregation": "mean",
        "method": "raw_gap", "macro_auroc": 0.7,
        "auroc_by_severity": {1: 0.6, 2: 0.6, 3: 0.7, 4: 0.8, 5: 0.8},
        "median_absolute_spearman": 0.5, "dominant_direction_fraction": 0.9,
        "oriented_adjacent_consistency": 0.8, "orientable": True, "complete": True,
        "expected_image_count": FULL_TUNING_IMAGE_COUNT, "twin_macro_auroc": 0.5,
    }
    candidate.update(overrides)
    return candidate


def compared(**overrides):
    """A stub carrying the two input differences `beats_both_inputs` reads."""
    candidate = stub(responsive_control_macro_auroc=0.6, reference_control_macro_auroc=0.6)
    candidate["responsive_control_macro_difference"] = 0.1
    candidate["reference_control_macro_difference"] = 0.1
    candidate.update(overrides)
    return candidate


# --- a hand-built pair of curves whose two columns move opposite ways ------------------------

ORIENTED_IMAGES = 4


def oriented_rows():
    """Four images whose responsive column rises with blur while their reference column falls.

    Every number below is chosen so that no two summaries of the two columns coincide: the
    reference climbs twice as fast per image as the responsive one, so its severity-0 mean is
    15.0 against 2.5 and its severity-5 mean is 10.0 against 7.5. A reference of
    `10 + image - severity` -- the obvious choice -- puts both columns at 7.5 at severity 5, and
    a control that scored the wrong column would be invisible at one end of the ladder.
    """
    return [
        {
            "image_id": image_id, "severity": severity,
            "arm": "decile_00_10__50_60", "signal": "persistence", "aggregation": "mean",
            "method": "raw_responsive", "arm_family": "anchored",
            "declared_before_data": True, "score_scope": "layer_2",
            "reference_bin": "decile_00_10", "responsive_bin": "decile_50_60",
            "reference": 10.0 + 2 * image_id - severity,
            "responsive": float(image_id) + severity,
            "score": float(image_id) + severity,
            "fold": image_id % 5, "fit_slope": None, "fit_offset": None,
        }
        for image_id in range(1, ORIENTED_IMAGES + 1)
        for severity in range(6)
    ]


# --- one arm, two summaries, two methods, twelve distinct macro AUROCs -----------------------

MATCHED_ARM = "decile_90_100__50_60__combined"
MATCHED_PAIR = "decile_90_100__50_60"
MATCHED_IMAGES = 5

MATCHED_RESPONSIVE_RATE = {
    ("persistence", "mean"): 1.0, ("persistence", "q90"): 0.3,
    ("confidence", "mean"): -0.9, ("confidence", "q90"): -0.15,
}
MATCHED_GAP_RATE = {
    ("persistence", "mean"): 0.5, ("persistence", "q90"): -0.45,
    ("confidence", "mean"): -0.4, ("confidence", "q90"): -0.6,
}
MATCHED_REFERENCE_RATE = {
    ("persistence", "mean"): -2.0, ("persistence", "q90"): -0.7,
    ("confidence", "mean"): 2.5, ("confidence", "q90"): 0.8,
}
"""Twelve severity slopes, one per series, and the two sign patterns that matter.

`persistence/mean` has a rising gap against a falling confidence twin, and `persistence/q90` has
a falling gap against a rising responsive control. So one candidate disagrees with its twin
about direction and another disagrees with its own responsive control, and a bootstrap that
handed one method the other's orientation would be caught on whichever of the two it broke.

The reference slopes run against their own responsive slopes at every summary, which is what
gives the reference control an orientation of its own to lock.
"""


def matched_rows():
    """One differential arm and its confidence twin, at two summaries and two methods.

    `decile_90_100__50_60__combined` rather than an anchored arm because it is the only arm in
    the table whose `name` differs from its `pair_name`. A twin looked up by arm name finds
    nothing here, where on any of the other three it would find the right series by accident.
    """
    rows = []
    for signal, arm in (("persistence", MATCHED_ARM), ("confidence", MATCHED_PAIR)):
        for aggregation in ("mean", "q90"):
            for method in ("raw_responsive", "raw_gap"):
                rate = (
                    MATCHED_RESPONSIVE_RATE if method == "raw_responsive" else MATCHED_GAP_RATE
                )[(signal, aggregation)]
                for image_id in range(1, MATCHED_IMAGES + 1):
                    for severity in range(6):
                        rows.append({
                            "image_id": image_id, "severity": severity, "arm": arm,
                            "signal": signal, "aggregation": aggregation, "method": method,
                            "arm_family": "differential", "declared_before_data": False,
                            "score_scope": (
                                "combined" if signal == "persistence" else "confidence"
                            ),
                            "reference_bin": "decile_90_100", "responsive_bin": "decile_50_60",
                            "reference": 20.0 + image_id
                            + MATCHED_REFERENCE_RATE[(signal, aggregation)] * severity,
                            "responsive": image_id
                            + MATCHED_RESPONSIVE_RATE[(signal, aggregation)] * severity,
                            "score": image_id + rate * severity,
                            "fold": image_id % 5, "fit_slope": None, "fit_offset": None,
                        })
    return rows


def matched(samples=FAST_SAMPLES):
    rows = matched_rows()
    candidates = summarize_contrast_candidates(rows, expected_image_count=MATCHED_IMAGES)
    controls = summarize_contrast_candidates(
        reference_control_rows(rows), expected_image_count=MATCHED_IMAGES
    )
    attach_controls(candidates, controls, rows, samples=samples)
    return candidates, controls, rows


def keyed(items):
    return {
        (item["arm"], item["signal"], item["aggregation"], item["method"]): item
        for item in items
    }


def curve(rate, *, images, level=0.0):
    """`{image: {severity: level + image + rate * severity}}` for the bootstrap fixtures."""
    return {
        image: {severity: level + image + rate * severity for severity in range(6)}
        for image in images
    }


# --- the declared constants -----------------------------------------------------------------


def test_the_seed_and_sample_count_are_the_declared_ones():
    assert BOOTSTRAP_SEED == 20260821
    assert BOOTSTRAP_SAMPLES == 2000


def test_the_reference_control_is_not_one_of_the_declared_score_methods():
    """The spec calls it a reported control that does not change the count of four methods."""
    assert REFERENCE_CONTROL_METHOD == "raw_reference"
    assert REFERENCE_CONTROL_METHOD not in SCORE_METHODS
    assert contrast_controls.RESPONSIVE_CONTROL_METHOD == "raw_responsive"
    assert contrast_controls.RESPONSIVE_CONTROL_METHOD in SCORE_METHODS
    assert len(SCORE_METHODS) == 4


# --- the raw reference control ----------------------------------------------------------------


def test_a_reference_control_exists_for_every_arm_summary_and_signal(tmp_path):
    _, controls, _ = prepared(tmp_path)
    keys = {(item["arm"], item["signal"], item["aggregation"]) for item in controls}
    assert len(keys) == 21
    assert {item["method"] for item in controls} == {"raw_reference"}
    # 12 persistence (four arms x three summaries) and 9 confidence (three pairs x three)
    assert sorted(item["signal"] for item in controls).count("persistence") == 12
    assert sorted(item["signal"] for item in controls).count("confidence") == 9


def test_the_reference_control_scores_the_reference_range_itself(tmp_path):
    rows, _ = build_contrast_rows(loaded(tmp_path))
    control_rows = reference_control_rows(rows)
    assert len(control_rows) == len(
        [row for row in rows if row["method"] == "raw_responsive"]
    )
    for row in control_rows:
        assert row["score"] == row["reference"]
        assert row["method"] == "raw_reference"
    # one copy of each series and not four: four methods share one reference column, and four
    # copies would collide on the row key rather than quadruple the control
    assert {row["method"] for row in rows} == set(SCORE_METHODS)
    keys = [tuple(row[field] for field in CONTRAST_ROW_KEY) for row in control_rows]
    assert len(set(keys)) == len(keys)
    assert len(keys) == 21 * IMAGES * 6


def test_the_reference_control_leaves_the_rows_it_was_built_from_alone(tmp_path):
    """The caller still has to build the real candidates out of these rows afterwards."""
    rows, _ = build_contrast_rows(loaded(tmp_path))
    before = [(row["method"], row["score"]) for row in rows]
    reference_control_rows(rows)
    assert [(row["method"], row["score"]) for row in rows] == before
    responsive = [row for row in rows if row["method"] == "raw_responsive"]
    assert responsive and all(row["score"] == row["responsive"] for row in responsive)


def test_the_reference_control_locks_its_own_orientation():
    """A reference that falls while the responsive rises must lock to -1, not inherit +1."""
    rows = oriented_rows()
    candidate = summarize_contrast_candidates(rows, expected_image_count=ORIENTED_IMAGES)[0]
    control = summarize_contrast_candidates(
        reference_control_rows(rows), expected_image_count=ORIENTED_IMAGES
    )[0]
    assert candidate["orientation"] == 1
    assert control["orientation"] == -1
    # read the other way round the control would score 1 - 0.79375
    assert control["macro_auroc"] == pytest.approx(0.79375)
    assert control["auroc_by_severity"] == pytest.approx(
        {1: 0.625, 2: 0.71875, 3: 0.8125, 4: 0.875, 5: 0.9375}
    )


def test_the_reference_control_measures_the_reference_column_and_not_the_responsive_one():
    rows = oriented_rows()
    candidate = summarize_contrast_candidates(rows, expected_image_count=ORIENTED_IMAGES)[0]
    control = summarize_contrast_candidates(
        reference_control_rows(rows), expected_image_count=ORIENTED_IMAGES
    )[0]
    # severity 0 is 11, 13, 15, 17 on the reference column against 1, 2, 3, 4 on the responsive
    assert control["severity_statistics"][0]["mean"] == pytest.approx(15.0)
    assert candidate["severity_statistics"][0]["mean"] == pytest.approx(2.5)
    # and severity 5 separates them too, which a reference of 10 + image - severity would not
    assert control["severity_statistics"][5]["mean"] == pytest.approx(10.0)
    assert candidate["severity_statistics"][5]["mean"] == pytest.approx(7.5)
    assert candidate["macro_auroc"] == pytest.approx(0.9125)


# --- the curve index ---------------------------------------------------------------------------


def test_row_curves_are_indexed_in_one_pass(tmp_path):
    rows, _ = build_contrast_rows(loaded(tmp_path))
    curves = index_curves(rows)
    assert len(curves) == 45 + 36
    assert sum(1 for key in curves if key[1] == "persistence") == 45
    assert sum(1 for key in curves if key[1] == "confidence") == 36
    one = curves[("decile_00_10__50_60", "persistence", "mean", "raw_gap")]
    assert sorted(one) == list(range(1, IMAGES + 1))
    assert sorted(one[1]) == list(range(6))
    # every score lands under its own arm, signal, summary, method, image and severity
    for row in rows:
        indexed = curves[
            (row["arm"], row["signal"], row["aggregation"], row["method"])
        ][row["image_id"]][row["severity"]]
        assert indexed == row["score"]


# --- the confidence twin -------------------------------------------------------------------


def test_both_differential_arms_resolve_to_the_same_confidence_twin(tmp_path):
    candidates, _, _ = prepared(tmp_path)
    layer = next(
        item for item in candidates
        if item["arm"] == "decile_90_100__50_60"
        and item["signal"] == "persistence"
        and item["method"] == "raw_gap" and item["aggregation"] == "mean"
    )
    combined = next(
        item for item in candidates
        if item["arm"] == "decile_90_100__50_60__combined"
        and item["signal"] == "persistence"
        and item["method"] == "raw_gap" and item["aggregation"] == "mean"
    )
    assert layer["twin_arm"] == combined["twin_arm"] == "decile_90_100__50_60"
    assert layer["twin_macro_auroc"] == combined["twin_macro_auroc"]
    # the shared twin is not two arms agreeing by accident: the two candidates differ
    assert layer["macro_auroc"] != combined["macro_auroc"]
    twin = next(
        item for item in candidates
        if item["arm"] == "decile_90_100__50_60" and item["signal"] == "confidence"
        and item["method"] == "raw_gap" and item["aggregation"] == "mean"
    )
    assert layer["twin_macro_auroc"] == twin["macro_auroc"]


def test_every_candidate_is_matched_to_its_own_arm_summary_and_method():
    candidates, controls, _ = matched()
    by_key = keyed(candidates)
    control_by_key = {
        (item["arm"], item["signal"], item["aggregation"]): item for item in controls
    }
    macros = [item["macro_auroc"] for item in candidates + controls]
    # nothing below can be satisfied by two series that happen to score the same
    assert len(set(macros)) == len(macros) == 12

    checked = 0
    for candidate in candidates:
        if candidate["signal"] != "persistence":
            continue
        arm, aggregation, method = (
            candidate["arm"], candidate["aggregation"], candidate["method"]
        )
        assert candidate["twin_arm"] == MATCHED_PAIR
        assert candidate["twin_arm"] != arm
        twin = by_key[(MATCHED_PAIR, "confidence", aggregation, method)]
        responsive = by_key[(arm, "persistence", aggregation, "raw_responsive")]
        reference = control_by_key[(arm, "persistence", aggregation)]
        assert candidate["twin_macro_auroc"] == twin["macro_auroc"]
        assert candidate["twin_auroc_by_severity"] == twin["auroc_by_severity"]
        assert candidate["twin_orientation"] == twin["orientation"]
        assert candidate["responsive_control_macro_auroc"] == responsive["macro_auroc"]
        assert candidate["reference_control_macro_auroc"] == reference["macro_auroc"]
        checked += 1
    assert checked == 4


def test_a_twin_locks_its_own_direction_and_the_bootstrap_reads_it_that_way():
    """The candidate rises with blur, its twin falls, and the interval is built on both."""
    candidates, _, _ = matched()
    candidate = keyed(candidates)[(MATCHED_ARM, "persistence", "mean", "raw_gap")]
    assert candidate["orientation"] == 1
    assert candidate["twin_orientation"] == -1
    twin_macro = candidate["twin_macro_auroc"]
    assert candidate["twin_bootstrap"]["macro_difference"] == pytest.approx(
        candidate["macro_auroc"] - twin_macro
    )
    # handing the twin the candidate's direction would mirror it to 1 - twin_macro
    assert candidate["twin_bootstrap"]["macro_difference"] != pytest.approx(
        candidate["macro_auroc"] - (1.0 - twin_macro)
    )


def test_a_responsive_control_that_runs_the_other_way_keeps_its_own_direction():
    """The falling gap at `q90`, whose own responsive range rises."""
    candidates, _, _ = matched()
    candidate = keyed(candidates)[(MATCHED_ARM, "persistence", "q90", "raw_gap")]
    responsive = keyed(candidates)[(MATCHED_ARM, "persistence", "q90", "raw_responsive")]
    assert candidate["orientation"] == -1
    assert responsive["orientation"] == 1
    control_macro = candidate["responsive_control_macro_auroc"]
    assert candidate["responsive_control_bootstrap"]["macro_difference"] == pytest.approx(
        candidate["macro_auroc"] - control_macro
    )
    assert candidate["responsive_control_bootstrap"]["macro_difference"] != pytest.approx(
        candidate["macro_auroc"] - (1.0 - control_macro)
    )


def test_every_bootstrap_point_estimate_is_the_pair_of_macro_aurocs_it_reports(tmp_path):
    """Across all 45 candidates: the resampled statistic and the summariser agree unresampled.

    Two independent implementations of the same AUROC -- `severity_aurocs` over the candidate's
    own scores and `_macro_from_draws` over an identity draw -- so a tie rule, an orientation or
    a denominator that moved in one of them would separate the two.
    """
    candidates, _, _ = prepared(tmp_path)
    checked = 0
    for candidate in candidates:
        if candidate["signal"] != "persistence":
            continue
        for label in ("responsive_control", "twin"):
            bootstrap = candidate[f"{label}_bootstrap"]
            assert bootstrap["macro_difference"] == pytest.approx(
                candidate[f"{label}_macro_difference"]
            )
            assert bootstrap["seed"] == 20260821
            assert bootstrap["samples"] == FAST_SAMPLES
            assert bootstrap["low"] <= bootstrap["high"]
            checked += 1
    assert checked == 90


def test_a_candidate_that_only_restates_confidence_is_flagged_redundant(tmp_path):
    candidates, _, _ = prepared(tmp_path)
    checked = 0
    flags = set()
    for candidate in candidates:
        if candidate["signal"] != "persistence":
            continue
        own, twin = candidate["macro_auroc"], candidate["twin_macro_auroc"]
        # a candidate or twin that produced no AUROC is redundant by default: it has not
        # been shown to beat confidence, and "unmeasured" must never read as "wins"
        expected = not (own is not None and twin is not None and own > twin)
        assert candidate["confidence_redundant"] is expected
        flags.add(candidate["confidence_redundant"])
        checked += 1
    assert checked == 45  # the assertion above is vacuous if the loop never runs
    assert flags == {True, False}  # and near-vacuous if the fixture only ever answers one way


@pytest.mark.parametrize(
    "own, twin, redundant",
    [
        (0.70, 0.60, False),   # ahead of confidence
        (0.70, 0.70, True),    # exactly tied: equal is not better
        (0.60, 0.70, True),    # behind confidence
        (0.70, None, True),    # no computable twin: unmeasured never reads as a win
        (None, 0.60, True),    # no AUROC of its own
        (None, None, True),    # neither measured
        # both below chance, and the candidate is still the better of the two: the flag
        # compares two signals and says nothing about whether either is worth deploying
        (0.45, 0.43, False),
        (0.43, 0.45, True),
    ],
)
def test_confidence_redundancy_is_strict(own, twin, redundant):
    assert is_confidence_redundant(
        stub(macro_auroc=own, twin_macro_auroc=twin)
    ) is redundant


# --- the two input comparisons ----------------------------------------------------------------


def test_a_candidate_is_compared_with_both_of_its_inputs(tmp_path):
    candidates, _, _ = prepared(tmp_path)
    candidate = next(
        item for item in candidates
        if item["signal"] == "persistence" and item["macro_auroc"] is not None
    )
    assert candidate["responsive_control_macro_difference"] == pytest.approx(
        candidate["macro_auroc"] - candidate["responsive_control_macro_auroc"]
    )
    assert candidate["reference_control_macro_difference"] == pytest.approx(
        candidate["macro_auroc"] - candidate["reference_control_macro_auroc"]
    )
    assert candidate["beats_both_inputs"] is (
        candidate["responsive_control_macro_difference"] > 0
        and candidate["reference_control_macro_difference"] > 0
    )
    # the two comparisons are two different numbers, so neither assertion above is satisfied
    # by whichever control the lookup happened to reach
    assert (
        candidate["responsive_control_macro_auroc"]
        != candidate["reference_control_macro_auroc"]
    )


def test_every_control_comparison_is_reported_at_every_severity(tmp_path):
    candidates, _, _ = prepared(tmp_path)
    checked = 0
    for candidate in candidates:
        if candidate["signal"] != "persistence":
            continue
        assert all(field in candidate for field in CONTROL_FIELDS)
        for label, other in (
            ("responsive_control", candidate["responsive_control_macro_auroc"]),
            ("reference_control", candidate["reference_control_macro_auroc"]),
            ("twin", candidate["twin_macro_auroc"]),
        ):
            differences = candidate[f"{label}_auroc_difference_by_severity"]
            assert sorted(differences) == [1, 2, 3, 4, 5]
            assert candidate[f"{label}_macro_difference"] == pytest.approx(
                candidate["macro_auroc"] - other
            )
            assert sum(differences.values()) / 5 == pytest.approx(
                candidate[f"{label}_macro_difference"]
            )
        checked += 1
    assert checked == 45


def test_the_per_severity_differences_are_the_two_aurocs_subtracted():
    candidates, _, _ = matched()
    by_key = keyed(candidates)
    candidate = by_key[(MATCHED_ARM, "persistence", "mean", "raw_gap")]
    twin = by_key[(MATCHED_PAIR, "confidence", "mean", "raw_gap")]
    assert candidate["twin_auroc_difference_by_severity"] == pytest.approx({
        severity: candidate["auroc_by_severity"][severity] - twin["auroc_by_severity"][severity]
        for severity in range(1, 6)
    })
    # not the other way round, and not all zero
    assert candidate["auroc_by_severity"] != twin["auroc_by_severity"]
    assert candidate["twin_auroc_difference_by_severity"] != pytest.approx({
        severity: twin["auroc_by_severity"][severity] - candidate["auroc_by_severity"][severity]
        for severity in range(1, 6)
    })


def test_a_raw_responsive_candidate_is_its_own_responsive_control(tmp_path):
    """Matched on arm and summary, so the one method that *is* the control ties with itself."""
    candidates, _, _ = prepared(tmp_path)
    checked = 0
    for candidate in candidates:
        if candidate["signal"] != "persistence" or candidate["method"] != "raw_responsive":
            continue
        assert candidate["responsive_control_macro_auroc"] == candidate["macro_auroc"]
        assert candidate["responsive_control_macro_difference"] == 0.0
        assert candidate["responsive_control_bootstrap"]["macro_difference"] == 0.0
        assert candidate["responsive_control_bootstrap"]["low"] == 0.0
        assert candidate["responsive_control_bootstrap"]["high"] == 0.0
        assert candidate["beats_both_inputs"] is False
        checked += 1
    assert checked == 12


def test_a_confidence_twin_is_given_no_controls_of_its_own(tmp_path):
    """A control with controls of its own invites the question of whether it passed them."""
    candidates, _, _ = prepared(tmp_path)
    twins = [item for item in candidates if item["signal"] == "confidence"]
    assert len(twins) == 36
    for twin in twins:
        assert not [field for field in CONTROL_FIELDS if field in twin]


def test_beating_one_input_while_losing_to_the_other_is_not_beating_both():
    candidate = stub(
        macro_auroc=0.70,
        responsive_control_macro_auroc=0.60,
        reference_control_macro_auroc=0.85,
    )
    candidate["responsive_control_macro_difference"] = 0.10
    candidate["reference_control_macro_difference"] = -0.15
    assert beats_both_inputs(candidate) is False


@pytest.mark.parametrize(
    "responsive, reference, beats",
    [
        (0.10, 0.20, True),     # ahead of both
        (0.10, -0.15, False),   # beat the responsive range, lost to the reference range
        (-0.15, 0.10, False),   # and the mirror of it, which `reference > 0` alone would pass
        (-0.10, -0.20, False),  # behind both
        (0.0, 0.10, False),     # level with an input is not ahead of it
        (0.10, 0.0, False),
        (None, 0.10, False),    # a comparison that could not be computed was not won
        (0.10, None, False),
        (None, None, False),
    ],
)
def test_beating_both_inputs_needs_both_differences_strictly_positive(
    responsive, reference, beats
):
    candidate = compared(
        responsive_control_macro_difference=responsive,
        reference_control_macro_difference=reference,
    )
    assert beats_both_inputs(candidate) is beats


# --- the paired bootstrap ----------------------------------------------------------------------


def test_the_paired_bootstrap_is_deterministic_under_its_seed():
    """Two runs agree, and they agree on *these* numbers rather than on any stable pair.

    The three literals below are the whole procedure written down: seed 20260821 through
    NumPy's `default_rng`, 200 draws of 20 images with replacement, the average-rank AUROC, and
    the 2.5th and 97.5th percentiles of the resulting differences. Moving any one of them moves
    at least one of the three -- reading the interval at the 5th and 95th percentiles instead
    gives 0.0497375 and 0.0770, and re-seeding gives another pair again. A test that only
    asserted `first == second` would pass on all of those.
    """
    images = list(range(1, 21))
    candidate = curve(1.0, images=images)
    control = curve(0.5, images=images)
    first = paired_macro_bootstrap(
        images, candidate, control,
        candidate_orientation=1, control_orientation=1, samples=200,
    )
    second = paired_macro_bootstrap(
        images, candidate, control,
        candidate_orientation=1, control_orientation=1, samples=200,
    )
    assert first == second
    assert first["low"] <= first["macro_difference"] <= first["high"]
    assert first["seed"] == 20260821
    assert first["macro_difference"] == pytest.approx(0.0645)
    assert first["low"] == pytest.approx(0.04723750)
    assert first["high"] == pytest.approx(0.07901875)


def test_a_different_seed_draws_a_different_interval():
    """Otherwise a run that ignored the seed argument entirely would look deterministic."""
    images = list(range(1, 21))
    candidate = curve(1.0, images=images)
    control = curve(0.5, images=images)
    declared = paired_macro_bootstrap(
        images, candidate, control,
        candidate_orientation=1, control_orientation=1, samples=200,
    )
    other = paired_macro_bootstrap(
        images, candidate, control,
        candidate_orientation=1, control_orientation=1, samples=200, seed=1,
    )
    assert other["seed"] == 1
    assert (other["low"], other["high"]) != (declared["low"], declared["high"])
    # the point estimate is the unresampled difference, so the seed must not move it
    assert other["macro_difference"] == declared["macro_difference"]


def test_the_paired_bootstrap_resamples_both_methods_identically():
    """A method compared with itself must give an interval of exactly zero width."""
    images = list(range(1, 21))
    scores = curve(1.0, images=images)
    result = paired_macro_bootstrap(
        images, scores, scores,
        candidate_orientation=1, control_orientation=1, samples=200,
    )
    assert result["macro_difference"] == pytest.approx(0.0)
    assert result["low"] == pytest.approx(0.0)
    assert result["high"] == pytest.approx(0.0)
    assert result["verdict"] == "not supported on tuning"


def test_the_paired_bootstrap_resamples_the_roster_it_was_given_and_no_other():
    """`image_ids` is the roster, not a hint: a curve dictionary may hold more than is asked for.

    Both methods are read by position in that one list, which is the whole of what makes the
    draw paired. Reading each dictionary's own sorted keys instead happens to agree whenever the
    two rosters match -- which they do everywhere `attach_controls` calls this -- so the roster
    here is every other image, and a reader of the dictionary's own keys would score the first
    half of the images instead of every second one and report 0.108 where the answer is 0.060.
    """
    everything = list(range(1, 21))
    chosen = everything[::2]
    candidate = curve(1.0, images=everything)
    control = curve(0.5, images=everything)
    keywords = dict(candidate_orientation=1, control_orientation=1, samples=200)
    narrow = paired_macro_bootstrap(chosen, candidate, control, **keywords)
    restricted = paired_macro_bootstrap(
        chosen,
        {image: candidate[image] for image in chosen},
        {image: control[image] for image in chosen},
        **keywords,
    )
    assert narrow == restricted
    first_half = paired_macro_bootstrap(everything[:10], candidate, control, **keywords)
    assert narrow["macro_difference"] != first_half["macro_difference"]
    assert narrow["low"] != first_half["low"]
    assert narrow["macro_difference"] != paired_macro_bootstrap(
        everything, candidate, control, **keywords
    )["macro_difference"]


def test_the_paired_bootstrap_point_estimate_is_the_two_macro_aurocs_difference():
    """Checked against `severity_aurocs`, which is what every published AUROC comes from."""
    images = list(range(1, 21))
    candidate = curve(1.0, images=images)
    control = curve(-0.4, images=images, level=3.0)
    result = paired_macro_bootstrap(
        images, candidate, control,
        candidate_orientation=1, control_orientation=-1, samples=200,
    )
    _, candidate_macro = severity_aurocs(
        {severity: [candidate[image][severity] for image in images] for severity in range(6)}, 1
    )
    _, control_macro = severity_aurocs(
        {severity: [control[image][severity] for image in images] for severity in range(6)}, -1
    )
    assert candidate_macro != control_macro
    assert result["macro_difference"] == pytest.approx(candidate_macro - control_macro)


def test_the_paired_bootstrap_reads_each_method_in_its_own_direction():
    """Flipping one method's orientation mirrors its AUROC, so the difference must move."""
    images = list(range(1, 21))
    candidate = curve(1.0, images=images)
    control = curve(0.5, images=images)
    upright = paired_macro_bootstrap(
        images, candidate, control,
        candidate_orientation=1, control_orientation=1, samples=200,
    )
    flipped = paired_macro_bootstrap(
        images, candidate, control,
        candidate_orientation=1, control_orientation=-1, samples=200,
    )
    _, control_macro = severity_aurocs(
        {severity: [control[image][severity] for image in images] for severity in range(6)}, 1
    )
    assert flipped["macro_difference"] == pytest.approx(
        upright["macro_difference"] + 2 * control_macro - 1.0
    )
    assert flipped["macro_difference"] != pytest.approx(upright["macro_difference"])


@pytest.mark.parametrize("orientation", [0, None, 2, -2, 0.5, "1"])
def test_the_paired_bootstrap_refuses_an_orientation_that_is_not_a_direction(orientation):
    """`0` would silently flatten every score to zero and report chance-level ranking.

    The membership test is `corruption_metrics.binary_auroc`'s, so `1.0` and `-1.0` are accepted
    the same way that function accepts them -- they compare equal to the integers and mean the
    same thing. What is refused is anything that would change the arithmetic silently.
    """
    images = list(range(1, 5))
    scores = curve(1.0, images=images)
    with pytest.raises(ValueError, match="orientation must be -1 or 1"):
        paired_macro_bootstrap(
            images, scores, scores,
            candidate_orientation=orientation, control_orientation=1, samples=10,
        )
    with pytest.raises(ValueError, match="orientation must be -1 or 1"):
        paired_macro_bootstrap(
            images, scores, scores,
            candidate_orientation=1, control_orientation=orientation, samples=10,
        )


def test_a_positive_interval_above_zero_is_supported_on_tuning():
    images = list(range(1, 41))
    strong = {image: {severity: float(severity) for severity in range(6)}
              for image in images}
    flat = {image: {severity: float(image) for severity in range(6)}
            for image in images}
    result = paired_macro_bootstrap(
        images, strong, flat,
        candidate_orientation=1, control_orientation=1, samples=200,
    )
    assert result["macro_difference"] > 0
    assert result["low"] > 0
    assert result["verdict"] == "supported on tuning"


def test_a_positive_estimate_whose_interval_crosses_zero_is_inconclusive():
    """One image carries the whole advantage, so a third of the draws see none of it."""
    images = list(range(1, 11))
    flat = {image: {severity: float(image) for severity in range(6)} for image in images}
    nearly_flat = {
        image: {
            severity: float(image) + (0.5 * severity if image == 1 else 0.0)
            for severity in range(6)
        }
        for image in images
    }
    result = paired_macro_bootstrap(
        images, nearly_flat, flat,
        candidate_orientation=1, control_orientation=1, samples=200,
    )
    assert result["macro_difference"] > 0
    assert result["low"] <= 0
    assert result["verdict"] == "inconclusive on tuning"


def test_a_negative_point_estimate_is_not_supported_on_tuning():
    images = list(range(1, 41))
    strong = {image: {severity: float(severity) for severity in range(6)}
              for image in images}
    flat = {image: {severity: float(image) for severity in range(6)}
            for image in images}
    result = paired_macro_bootstrap(
        images, flat, strong,
        candidate_orientation=1, control_orientation=1, samples=200,
    )
    assert result["macro_difference"] < 0
    assert result["verdict"] == "not supported on tuning"


def test_the_sample_count_is_read_from_the_module_at_call_time(monkeypatch):
    """Task 9 patches `BOOTSTRAP_SAMPLES` down; a signature default would ignore the patch."""
    images = list(range(1, 11))
    scores = curve(1.0, images=images)
    monkeypatch.setattr(contrast_controls, "BOOTSTRAP_SAMPLES", 13)
    result = paired_macro_bootstrap(
        images, scores, scores, candidate_orientation=1, control_orientation=1
    )
    assert result["samples"] == 13
    # and through the caller that runs 90 of them
    candidates, controls, rows = matched(samples=None)
    attach_controls(candidates, controls, rows)
    persistence = [item for item in candidates if item["signal"] == "persistence"]
    assert persistence
    assert {item["twin_bootstrap"]["samples"] for item in persistence} == {13}
    assert {item["responsive_control_bootstrap"]["samples"] for item in persistence} == {13}


def test_the_full_two_thousand_sample_bootstrap_runs_in_reasonable_time():
    """The production sample count on a small image set, so the batched path is exercised.

    Task 9's integration tests patch `BOOTSTRAP_SAMPLES` down to keep the suite fast; without
    this test nothing would ever run the real number and a per-draw regression would only show
    up on alienware2.
    """
    images = list(range(1, 21))
    candidate = curve(1.0, images=images)
    control = curve(0.5, images=images)
    result = paired_macro_bootstrap(
        images, candidate, control, candidate_orientation=1, control_orientation=1
    )
    assert result["samples"] == BOOTSTRAP_SAMPLES == 2000
    # the point estimate is not a function of how many draws were taken
    small = paired_macro_bootstrap(
        images, candidate, control,
        candidate_orientation=1, control_orientation=1, samples=10,
    )
    assert small["macro_difference"] == result["macro_difference"]


# --- the deployment ranking ---------------------------------------------------------------------


def test_ranking_uses_severity_two_as_the_third_criterion():
    weak = stub(method="raw_gap", auroc_by_severity={1: 0.6, 2: 0.55, 3: 0.7, 4: 0.8, 5: 0.8})
    strong = stub(
        method="relative_gap", auroc_by_severity={1: 0.6, 2: 0.65, 3: 0.7, 4: 0.8, 5: 0.8}
    )
    ranked = rank_contrast_candidates([weak, strong])
    assert [item["method"] for item in ranked] == ["relative_gap", "raw_gap"]


def test_ranking_prefers_macro_then_severity_one():
    low_macro = stub(method="raw_gap", macro_auroc=0.60)
    high_macro = stub(method="relative_gap", macro_auroc=0.80)
    assert [item["method"] for item in rank_contrast_candidates([low_macro, high_macro])] == [
        "relative_gap", "raw_gap"
    ]
    tied_a = stub(method="raw_gap", auroc_by_severity={1: 0.50, 2: 0.9, 3: 0.7, 4: 0.8, 5: 0.8})
    tied_b = stub(
        method="relative_gap", auroc_by_severity={1: 0.70, 2: 0.5, 3: 0.7, 4: 0.8, 5: 0.8}
    )
    assert [item["method"] for item in rank_contrast_candidates([tied_a, tied_b])] == [
        "relative_gap", "raw_gap"
    ]


def test_ranking_falls_through_to_spearman_then_direction_then_consistency():
    """Criteria four, five and six, each proven by a pair that ties on everything above it.

    In all three pairs the winner is `relative_gap`, which sorts *after* `raw_gap` on the name
    tiebreak -- so a criterion that stopped being read does not merely reorder these pairs by a
    lower criterion, it reverses them.
    """
    weak = stub(method="raw_gap", median_absolute_spearman=0.40,
                dominant_direction_fraction=1.0, oriented_adjacent_consistency=1.0)
    strong = stub(method="relative_gap", median_absolute_spearman=0.60,
                  dominant_direction_fraction=0.5, oriented_adjacent_consistency=0.5)
    assert [item["method"] for item in rank_contrast_candidates([weak, strong])] == [
        "relative_gap", "raw_gap"
    ]

    weak = stub(method="raw_gap", dominant_direction_fraction=0.60,
                oriented_adjacent_consistency=1.0)
    strong = stub(method="relative_gap", dominant_direction_fraction=0.80,
                  oriented_adjacent_consistency=0.2)
    assert [item["method"] for item in rank_contrast_candidates([weak, strong])] == [
        "relative_gap", "raw_gap"
    ]

    weak = stub(method="raw_gap", oriented_adjacent_consistency=0.60)
    strong = stub(method="relative_gap", oriented_adjacent_consistency=0.80)
    assert [item["method"] for item in rank_contrast_candidates([weak, strong])] == [
        "relative_gap", "raw_gap"
    ]


def test_ranking_ends_with_a_deterministic_arm_summary_and_method_order():
    """Arm, then summary, then method, ascending -- and in that order, not any other."""
    later_arm = stub(arm="b_arm", aggregation="a_summary", method="a_method")
    earlier_arm = stub(arm="a_arm", aggregation="b_summary", method="b_method")
    assert [item["arm"] for item in rank_contrast_candidates([later_arm, earlier_arm])] == [
        "a_arm", "b_arm"
    ]
    later_summary = stub(arm="one", aggregation="b_summary", method="a_method")
    earlier_summary = stub(arm="one", aggregation="a_summary", method="b_method")
    assert [
        item["aggregation"] for item in rank_contrast_candidates(
            [later_summary, earlier_summary]
        )
    ] == ["a_summary", "b_summary"]
    later_method = stub(arm="one", aggregation="one", method="b_method")
    earlier_method = stub(arm="one", aggregation="one", method="a_method")
    assert [
        item["method"] for item in rank_contrast_candidates([later_method, earlier_method])
    ] == ["a_method", "b_method"]


def test_confidence_twins_and_reference_controls_are_never_ranked():
    twin = stub(signal="confidence", macro_auroc=0.99)
    control = stub(method="raw_reference", macro_auroc=0.99)
    real = stub(macro_auroc=0.55)
    ranked = rank_contrast_candidates([twin, control, real])
    assert [item["method"] for item in ranked] == ["raw_gap"]
    assert twin["deployable"] is False
    assert control["deployable"] is False
    assert real["deployable"] is True


def test_a_run_short_of_two_hundred_and_fifty_images_is_not_deployable():
    short = stub(expected_image_count=249)
    assert rank_contrast_candidates([short]) == []
    assert short["deployable"] is False


def test_an_unorientable_or_incomplete_candidate_is_not_deployable():
    """Three cases, and the third is why `orientable` is checked as well as `macro_auroc`.

    `stub(orientable=False, macro_auroc=None)` is the shape the summariser really produces --
    withholding the direction withholds the AUROC -- so on its own it does not say which of the
    two clauses did the work. `stub(orientable=False)`, which keeps an AUROC, does.
    """
    for broken in (
        stub(orientable=False, macro_auroc=None),
        stub(orientable=False),
        stub(complete=False),
    ):
        assert rank_contrast_candidates([broken]) == []
        assert broken["deployable"] is False


def test_a_candidate_without_an_auroc_is_not_deployable():
    """Separately from being unorientable, because the sort key would fail on a `None` macro."""
    orphan = stub(macro_auroc=None)
    assert rank_contrast_candidates([orphan]) == []
    assert orphan["deployable"] is False


def test_a_candidate_without_a_computable_twin_is_not_deployable():
    orphan = stub(twin_macro_auroc=None)
    assert rank_contrast_candidates([orphan]) == []
    assert orphan["deployable"] is False


def test_the_ranking_returns_the_dictionaries_it_was_given():
    """Not copies -- the plots, the CSV and the report all read the same objects."""
    first = stub(method="raw_gap", macro_auroc=0.80)
    second = stub(method="relative_gap", macro_auroc=0.60)
    ranked = rank_contrast_candidates([second, first])
    assert ranked[0] is first
    assert ranked[1] is second


def test_a_six_image_run_ranks_nothing_and_marks_every_candidate(tmp_path):
    """The fixture's roster is 6 and the gate is 250, so the whole file ranks nothing.

    Recorded rather than worked around: a test bundle is not a deployment run, and a ranking
    that appeared on one would mean the coverage gate was reading the rows instead of the
    declared roster size.
    """
    candidates, _, _ = prepared(tmp_path)
    assert rank_contrast_candidates(candidates) == []
    assert len(candidates) == 81
    assert all(item["deployable"] is False for item in candidates)


def test_a_full_roster_of_the_same_candidates_is_deployable(tmp_path):
    """The same summaries at `expected_image_count=250` rank, so the gate is the only thing
    holding the six-image fixture back and not a second failure hiding behind it."""
    candidates, _, _ = prepared(tmp_path)
    for candidate in candidates:
        candidate["expected_image_count"] = FULL_TUNING_IMAGE_COUNT
    ranked = rank_contrast_candidates(candidates)
    assert len(ranked) == 45
    assert {item["signal"] for item in ranked} == {"persistence"}
    macros = [item["macro_auroc"] for item in ranked]
    assert macros == sorted(macros, reverse=True)
    assert len(set(macros)) > 1  # a constant column is sorted whatever the key does
