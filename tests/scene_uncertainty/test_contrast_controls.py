"""What the three controls, the paired bootstrap and the deployment ranking are held to.

Three fixtures do most of the work here and each answers a question the others cannot.

`prepared` builds the shared six-image bundle every task from 3 onwards reads, and is what says
the controls survive contact with the real arm table: 21 reference controls, 45 persistence
candidates, 36 confidence twins, four arms with four different reference controls, and a
redundancy verdict that comes out both ways. What it *cannot* say anything about is which
*summary* a lookup matched, because `contrast_test_utils.default_score` ignores its `aggregation`
argument -- so all three summaries of one bin are byte-identical and a control matched to the
wrong summary reads exactly like one matched to the right one.

`matched_rows` is the fixture for that. It is one arm and its twin at two summaries and two
methods, with twelve pairwise-distinct macro AUROCs by construction, so every one of the four
attached comparisons can only be satisfied by the series it names. It uses the `combined`
differential arm on purpose, because that is the one arm whose name is not its pair name. What
*it* cannot bind is the arm dimension, having only one persistence arm -- which is why the
reference control's arm is bound on `prepared` instead.

`unmeasured_rows` is the third, and it exists because neither of the other two contains a single
missing measurement. In production a candidate loses its AUROC only by being unorientable, and
the shared bundle has no unorientable candidate, so the whole `None`-handling half of
`attach_controls` would otherwise run on no input the suite ever builds.

No fixture is asked for a number that moves with the roster. The margins between a candidate and
its twin shrink as images are added -- nine confidence candidates sit below chance at six images
and drift towards it at 250 -- so what is pinned here is either an exact hand-computable value on
a hand-built curve, an identity between two things the code computes separately, or the *side* of
a comparison.
"""
import pytest

from src.scene_uncertainty import contrast_controls
from src.scene_uncertainty.contrast_analysis import (
    CONTRAST_ROW_KEY,
    build_contrast_rows,
    summarize_contrast_candidates,
)
from src.scene_uncertainty.contrast_controls import (
    ARM_NAMES,
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
from src.scene_uncertainty.contrast_inputs import (
    AGGREGATIONS,
    ARMS,
    FULL_TUNING_IMAGE_COUNT,
    load_contrast_inputs,
)
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

COUNTERPART_MACRO = {
    "responsive_control": "responsive_control_macro_auroc",
    "reference_control": "reference_control_macro_auroc",
    "twin": "twin_macro_auroc",
}
"""The three counterparts, and where each one's own macro AUROC is published.

The comparison labels are not quite the field names -- the twin's macro is `twin_macro_auroc`,
not `twin_control_macro_auroc` -- so the mapping is written once here rather than spelled out
in each test that wants to loop over all three.
"""

CONTROL_FIELDS = (
    "twin_arm", "twin_macro_auroc", "twin_auroc_by_severity", "twin_orientation",
    "responsive_control_macro_auroc", "reference_control_macro_auroc",
    "responsive_control_macro_difference", "responsive_control_auroc_difference_by_severity",
    "reference_control_macro_difference", "reference_control_auroc_difference_by_severity",
    "twin_macro_difference", "twin_auroc_difference_by_severity",
    "beats_both_inputs", "confidence_redundant",
    "responsive_control_bootstrap", "reference_control_bootstrap", "twin_bootstrap",
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


# --- every way a comparison can be missing, in one fixture -----------------------------------

UNMEASURED_IMAGES = 4

UNMEASURED_PLAN = (
    # (arm, signal, method, score slope, reference-column slope); a slope of 0.0 is a flat
    # curve, which `choose_orientation` refuses to orient and which therefore produces no AUROC
    ("decile_00_10__50_60", "persistence", "raw_responsive", 1.0, 2.0),
    ("decile_00_10__50_60", "persistence", "raw_gap", 0.0, 2.0),        # unorientable candidate
    ("decile_00_10__50_60", "persistence", "relative_gap", 0.7, 2.0),   # its twin is flat
    ("decile_00_10__50_60", "persistence", "clean_residual", 0.4, 2.0),  # it has no twin at all
    ("decile_00_10__50_60", "confidence", "raw_responsive", -1.0, -2.0),
    ("decile_00_10__50_60", "confidence", "raw_gap", -0.6, -2.0),
    ("decile_00_10__50_60", "confidence", "relative_gap", 0.0, -2.0),   # unorientable twin
    # no confidence `clean_residual`: that arm's residual candidate has no twin object
    ("quintile_00_20__40_60", "persistence", "raw_responsive", 1.0, 0.0),  # flat reference column
    ("quintile_00_20__40_60", "persistence", "raw_gap", 0.8, 0.0),
    ("quintile_00_20__40_60", "confidence", "raw_responsive", -1.0, -0.9),
    ("quintile_00_20__40_60", "confidence", "raw_gap", -0.5, -0.9),
    # no persistence `raw_responsive`: this arm has neither a responsive nor a reference control
    ("decile_90_100__50_60", "persistence", "raw_gap", 0.9, 1.5),
    ("decile_90_100__50_60", "confidence", "raw_gap", -0.7, 1.5),
    ("decile_90_100__50_60__combined", "persistence", "raw_responsive", 0.0, 1.5),
    ("decile_90_100__50_60__combined", "persistence", "raw_gap", 0.6, 1.5),
)
"""Nine persistence candidates covering every absent-measurement state `attach_controls` can meet.

The shared bundle cannot produce any of them. `contrast_inputs` refuses a roster mismatch and
`contrast_analysis._curve` builds every image at every severity or raises, so in production the
only way a candidate loses its AUROC is by being unorientable -- and the shared fixture has no
unorientable candidate either. Without this fixture the whole `None`-handling half of
`attach_controls` is code that runs on no input the suite ever builds, and nine separate mutants
of it survive: a missing control published as `0.0`, a missing bootstrap field left absent
instead of `None`, and every guard that produces them.

Every slope is distinct so no two series score the same, and the flat ones are exactly the four
states that matter: an unorientable candidate, an unorientable twin, an unorientable reference
control and an unorientable responsive control.
"""


def unmeasured_rows():
    return [
        {
            "image_id": image_id, "severity": severity, "arm": arm,
            "signal": signal, "aggregation": "mean", "method": method,
            "arm_family": "anchored", "declared_before_data": True,
            "score_scope": "layer_2", "reference_bin": "decile_00_10",
            "responsive_bin": "decile_50_60",
            "reference": 20.0 + image_id + reference_slope * severity,
            "responsive": float(image_id) + severity,
            "score": image_id + slope * severity,
            "fold": image_id % 5, "fit_slope": None, "fit_offset": None,
        }
        for arm, signal, method, slope, reference_slope in UNMEASURED_PLAN
        for image_id in range(1, UNMEASURED_IMAGES + 1)
        for severity in range(6)
    ]


def unmeasured():
    rows = unmeasured_rows()
    candidates = summarize_contrast_candidates(rows, expected_image_count=UNMEASURED_IMAGES)
    controls = summarize_contrast_candidates(
        reference_control_rows(rows), expected_image_count=UNMEASURED_IMAGES
    )
    attach_controls(candidates, controls, rows, samples=FAST_SAMPLES)
    return keyed(candidates), controls


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
    """The spec calls it a reported control that does not change the count of four methods.

    Being outside that set is also the whole of how the ranking gate excludes it, so this is not
    a naming convention -- it is the exclusion.
    """
    assert REFERENCE_CONTROL_METHOD == "raw_reference"
    assert REFERENCE_CONTROL_METHOD not in SCORE_METHODS
    assert contrast_controls.RESPONSIVE_CONTROL_METHOD == "raw_responsive"
    assert contrast_controls.RESPONSIVE_CONTROL_METHOD in SCORE_METHODS
    assert len(SCORE_METHODS) == 4
    assert ARM_NAMES == {arm.name for arm in ARMS} and len(ARM_NAMES) == 4
    assert AGGREGATIONS == ("mean", "q90", "top20_mean")


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
    source = [row for row in rows if row["method"] == "raw_responsive"]
    assert len(control_rows) == len(source)
    # exactly two columns rewritten and every other one carried through untouched, so a control
    # row still says which two ranges it came from -- `reference` *and* `responsive`, and the
    # second of those is what a comparison against `row["reference"]` alone cannot see
    for control, row in zip(control_rows, source):
        assert control == {**row, "method": "raw_reference", "score": row["reference"]}
        assert control["score"] == control["reference"]
        assert control["method"] == "raw_reference"
    assert any(row["reference"] != row["responsive"] for row in source)
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


def test_the_reference_control_gets_an_interval_read_in_its_own_direction():
    """The spec's amended controls section: both inputs get an interval, not only the responsive.

    The reference column of this fixture falls at every summary while the candidate rises, so the
    two lock opposite directions and a bootstrap that reused the candidate's would report a
    mirrored control.
    """
    candidates, controls, _ = matched()
    candidate = keyed(candidates)[(MATCHED_ARM, "persistence", "mean", "raw_gap")]
    reference = next(
        item for item in controls
        if item["arm"] == MATCHED_ARM and item["signal"] == "persistence"
        and item["aggregation"] == "mean"
    )
    assert candidate["orientation"] == 1
    assert reference["orientation"] == -1
    bootstrap = candidate["reference_control_bootstrap"]
    assert sorted(bootstrap) == ["high", "low", "macro_difference", "samples", "seed", "verdict"]
    assert bootstrap["seed"] == 20260821
    assert bootstrap["macro_difference"] == pytest.approx(
        candidate["macro_auroc"] - reference["macro_auroc"]
    )
    assert bootstrap["macro_difference"] != pytest.approx(
        candidate["macro_auroc"] - (1.0 - reference["macro_auroc"])
    )
    # the candidate loses to this reference range, and the interval says so rather than being
    # a bare point estimate on the arm where a point estimate is least trustworthy
    assert bootstrap["macro_difference"] < 0
    assert bootstrap["verdict"] == "not supported on tuning"
    assert bootstrap["low"] <= bootstrap["macro_difference"] <= bootstrap["high"]


def test_every_bootstrap_point_estimate_is_the_pair_of_macro_aurocs_it_reports(tmp_path):
    """Across all 45 candidates and all three comparisons: 135 intervals, 135 agreements.

    Two independent implementations of the same AUROC -- `severity_aurocs` over the candidate's
    own scores and `_macro_from_draws` over an identity draw -- so a tie rule, an orientation or
    a denominator that moved in one of them would separate the two.
    """
    candidates, _, _ = prepared(tmp_path)
    checked = 0
    for candidate in candidates:
        if candidate["signal"] != "persistence":
            continue
        for label in ("responsive_control", "reference_control", "twin"):
            bootstrap = candidate[f"{label}_bootstrap"]
            assert bootstrap["macro_difference"] == pytest.approx(
                candidate[f"{label}_macro_difference"]
            )
            assert bootstrap["seed"] == 20260821
            assert bootstrap["samples"] == FAST_SAMPLES
            assert bootstrap["low"] <= bootstrap["high"]
            checked += 1
    assert checked == 135


def test_every_bootstrap_resamples_the_curves_of_its_own_summary():
    """The same identity as above, on the fixture whose summaries are not byte-identical.

    `prepared` cannot bind the *summary* dimension of a curve lookup. `default_score` ignores its
    `aggregation` argument, so all three summaries of one bin carry the same numbers there and a
    curve fetched from the wrong one is indistinguishable from the right one. Here they differ at
    every comparison -- the reference control is 0.960 at `mean` against 0.816 at `q90` and the
    twin 0.708 against 0.784 -- so a summary-blind lookup produces an interval whose point
    estimate no longer matches the macro difference published beside it.

    That is the silent failure this catches: the macro AUROC in the CSV would still be the right
    summary's, and only the interval printed next to it would have resampled a different one.
    Reading these bootstraps at `mean` alone is not enough either, because `mean` is the summary
    a blind lookup returns first.
    """
    candidates, _, _ = matched()
    by_key = keyed(candidates)
    checked = 0
    for candidate in candidates:
        if candidate["signal"] != "persistence":
            continue
        for label in COUNTERPART_MACRO:
            assert candidate[f"{label}_bootstrap"]["macro_difference"] == pytest.approx(
                candidate[f"{label}_macro_difference"]
            )
            checked += 1
    assert checked == 12  # four persistence candidates, three comparisons each
    # and the two summaries disagree at every comparison, so none of the twelve is satisfied by
    # a curve taken from the other one
    for label, field in COUNTERPART_MACRO.items():
        across_summaries = {
            by_key[(MATCHED_ARM, "persistence", aggregation, "raw_gap")][field]
            for aggregation in ("mean", "q90")
        }
        assert len(across_summaries) == 2, label


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


def test_each_arm_is_matched_to_its_own_reference_control(tmp_path):
    """The arm dimension of the reference control lookup, on the four-arm bundle.

    `matched` cannot bind this -- it has one persistence arm, so "this arm's control" and "any
    arm's control" are the same object there. Dropping `arm` from the lookup key hands one arm's
    reference macro to all four, which at 250 images changes the published number for 33 of the
    45 candidates and flips `beats_both_inputs` for six of them.
    """
    candidates, controls, _ = prepared(tmp_path)
    control_by_key = {
        (item["arm"], item["signal"], item["aggregation"]): item for item in controls
    }
    by_arm = {}
    checked = 0
    for candidate in candidates:
        if candidate["signal"] != "persistence":
            continue
        key = (candidate["arm"], "persistence", candidate["aggregation"])
        assert candidate["reference_control_macro_auroc"] == control_by_key[key]["macro_auroc"]
        assert candidate["responsive_control_macro_auroc"] == keyed(candidates)[
            (candidate["arm"], "persistence", candidate["aggregation"], "raw_responsive")
        ]["macro_auroc"]
        by_arm[candidate["arm"]] = candidate["reference_control_macro_auroc"]
        checked += 1
    assert checked == 45
    # four arms, four different reference controls: an arm-blind lookup cannot satisfy the above
    assert set(by_arm) == {arm.name for arm in ARMS}
    assert len(set(by_arm.values())) == 4

    # the *signal* of the control matters too, and three of the four arms carry a confidence
    # control under the same label as their persistence one, with a different macro AUROC
    twinned = [
        arm for arm in by_arm
        if (arm, "confidence", "mean") in control_by_key
    ]
    assert len(twinned) == 3
    for arm in twinned:
        assert (
            control_by_key[(arm, "persistence", "mean")]["macro_auroc"]
            != control_by_key[(arm, "confidence", "mean")]["macro_auroc"]
        )
    # a signal-blind index would land on whichever of the two the list happened to end on, so
    # the same controls in the other order must produce the same answer
    reversed_candidates, _, reversed_rows = prepared(tmp_path)
    reversed_controls = summarize_contrast_candidates(
        reference_control_rows(reversed_rows), expected_image_count=IMAGES
    )
    attach_controls(
        reversed_candidates, list(reversed(reversed_controls)), reversed_rows,
        samples=FAST_SAMPLES,
    )
    assert [
        item["reference_control_macro_auroc"] for item in reversed_candidates
        if item["signal"] == "persistence"
    ] == [
        item["reference_control_macro_auroc"] for item in candidates
        if item["signal"] == "persistence"
    ]


def test_the_twin_severity_aurocs_are_a_copy_and_not_the_twin_own_dictionary(tmp_path):
    """Both differential arms read one twin, so a shared dictionary would propagate twice."""
    candidates, _, _ = prepared(tmp_path)
    by_key = keyed(candidates)
    twin = by_key[("decile_90_100__50_60", "confidence", "mean", "raw_gap")]
    layer = by_key[("decile_90_100__50_60", "persistence", "mean", "raw_gap")]
    combined = by_key[("decile_90_100__50_60__combined", "persistence", "mean", "raw_gap")]
    for candidate in (layer, combined):
        assert candidate["twin_auroc_by_severity"] == twin["auroc_by_severity"]
        assert candidate["twin_auroc_by_severity"] is not twin["auroc_by_severity"]
    assert layer["twin_auroc_by_severity"] is not combined["twin_auroc_by_severity"]
    before = dict(twin["auroc_by_severity"])
    layer["twin_auroc_by_severity"][1] = -1.0
    assert twin["auroc_by_severity"] == before
    assert combined["twin_auroc_by_severity"] == before


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


def test_every_candidate_in_the_unmeasured_fixture_carries_all_control_fields():
    """Seventeen fields on every persistence candidate, measured or not.

    `CONTROL_FIELDS` is what Task 8 projects into `candidate_metrics.csv` and `summary.json`, so
    a field that is *absent* rather than `None` is a column that silently disappears for exactly
    the rows a reader most needs to see it on.
    """
    candidates, _ = unmeasured()
    persistence = [
        candidate for candidate in candidates.values()
        if candidate["signal"] == "persistence"
    ]
    assert len(persistence) == 9
    for candidate in persistence:
        assert not [field for field in CONTROL_FIELDS if field not in candidate]
    # the fixture is only worth running if it really contains unmeasured candidates
    assert sum(1 for item in persistence if item["macro_auroc"] is None) == 2


def test_an_unorientable_candidate_keeps_its_controls_and_withholds_every_comparison():
    """The one way a candidate loses its AUROC in this pipeline, carried all the way through.

    Its three counterparts are all perfectly well measured, so every `None` below is the
    candidate's own missing AUROC and not a missing control -- and both flags fall to the
    defaults their own docstrings promise rather than to whatever a comparison with `None`
    would have produced.
    """
    candidates, _ = unmeasured()
    candidate = candidates[("decile_00_10__50_60", "persistence", "mean", "raw_gap")]
    assert candidate["orientation"] is None
    assert candidate["macro_auroc"] is None
    assert candidate["auroc_by_severity"] is None
    # the three counterparts are measured, so nothing below is about them
    assert candidate["responsive_control_macro_auroc"] is not None
    assert candidate["reference_control_macro_auroc"] is not None
    assert candidate["twin_macro_auroc"] is not None
    assert candidate["twin_arm"] == "decile_00_10__50_60"
    for label in ("responsive_control", "reference_control", "twin"):
        assert candidate[f"{label}_macro_difference"] is None
        assert candidate[f"{label}_auroc_difference_by_severity"] is None
        assert candidate[f"{label}_bootstrap"] is None
    assert candidate["beats_both_inputs"] is False
    assert candidate["confidence_redundant"] is True


@pytest.mark.parametrize(
    "arm, method, withheld",
    [
        ("decile_00_10__50_60", "relative_gap", "twin"),
        ("quintile_00_20__40_60", "raw_gap", "reference_control"),
        ("decile_90_100__50_60__combined", "raw_gap", "responsive_control"),
    ],
)
def test_an_unorientable_control_withholds_only_its_own_comparison(arm, method, withheld):
    """One counterpart of the three is flat; the candidate and the other two are not."""
    candidates, _ = unmeasured()
    candidate = candidates[(arm, "persistence", "mean", method)]
    assert candidate["macro_auroc"] is not None
    assert candidate[COUNTERPART_MACRO[withheld]] is None
    assert candidate[f"{withheld}_macro_difference"] is None
    assert candidate[f"{withheld}_auroc_difference_by_severity"] is None
    assert candidate[f"{withheld}_bootstrap"] is None
    for label in set(COUNTERPART_MACRO) - {withheld}:
        assert candidate[COUNTERPART_MACRO[label]] is not None
        assert candidate[f"{label}_macro_difference"] is not None
        assert candidate[f"{label}_auroc_difference_by_severity"] is not None
        assert candidate[f"{label}_bootstrap"] is not None


def test_a_control_or_twin_that_does_not_exist_is_absent_and_never_zero():
    """A missing comparison must not publish `0.0`, which reads as a measured chance-level AUROC.

    Worse than merely wrong: `0.0` is *below* every real AUROC, so a candidate would show a large
    positive difference against a control that does not exist, and `beats_both_inputs` would then
    report that it beat an input nobody measured.
    """
    candidates, controls = unmeasured()
    control_keys = {(item["arm"], item["signal"], item["aggregation"]) for item in controls}

    # the residual candidate's arm has no confidence residual, so it has no twin object at all
    orphan = candidates[("decile_00_10__50_60", "persistence", "mean", "clean_residual")]
    assert ("decile_00_10__50_60", "confidence", "mean", "clean_residual") not in candidates
    assert orphan["macro_auroc"] is not None
    assert orphan["twin_arm"] == "decile_00_10__50_60"  # still says which twin was looked for
    assert orphan["twin_macro_auroc"] is None
    assert orphan["twin_auroc_by_severity"] is None
    assert orphan["twin_orientation"] is None
    assert orphan["twin_macro_difference"] is None
    assert orphan["twin_bootstrap"] is None
    assert orphan["confidence_redundant"] is True

    # this arm has no persistence `raw_responsive` rows, so it has neither input control
    stranded = candidates[("decile_90_100__50_60", "persistence", "mean", "raw_gap")]
    assert ("decile_90_100__50_60", "persistence", "mean", "raw_responsive") not in candidates
    assert ("decile_90_100__50_60", "persistence", "mean") not in control_keys
    assert stranded["macro_auroc"] is not None
    assert stranded["responsive_control_macro_auroc"] is None
    assert stranded["reference_control_macro_auroc"] is None
    assert stranded["responsive_control_macro_difference"] is None
    assert stranded["reference_control_macro_difference"] is None
    assert stranded["responsive_control_bootstrap"] is None
    assert stranded["reference_control_bootstrap"] is None
    assert stranded["beats_both_inputs"] is False
    # and its twin, which does exist, is untouched by either absence
    assert stranded["twin_macro_difference"] is not None
    assert stranded["twin_bootstrap"] is not None


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


def test_the_paired_bootstrap_sorts_the_roster_before_it_draws_from_it():
    """Two callers holding the same images in different orders must get the same interval.

    The draws are positions into the roster, so an unsorted roster pairs draw *k* with a
    different image and moves the interval -- to (0.047994, 0.081300) on the shuffle below,
    against the (0.047238, 0.079019) the sorted roster gives. The point estimate survives,
    because reordering both methods' columns together cannot change a rank statistic; only the
    resampling notices, which is why this needs its own test beside the determinism one.
    """
    images = list(range(1, 21))
    shuffled = images[10:] + images[:10]
    assert shuffled != images and sorted(shuffled) == images
    candidate = curve(1.0, images=images)
    control = curve(0.5, images=images)
    keywords = dict(candidate_orientation=1, control_orientation=1, samples=200)
    assert (
        paired_macro_bootstrap(shuffled, candidate, control, **keywords)
        == paired_macro_bootstrap(images, candidate, control, **keywords)
    )


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
    """Arm, then summary, then method, ascending -- and in that order, not any other.

    Declared names throughout, because the eligibility gate refuses anything outside the three
    closed sets. Each pair is arranged so that reading the three fields in any other order, or in
    the other direction, reverses it: the arm that sorts first carries the summary and the method
    that sort last.
    """
    later_arm = stub(arm="decile_90_100__50_60", aggregation="mean", method="clean_residual")
    earlier_arm = stub(arm="decile_00_10__50_60", aggregation="q90", method="raw_gap")
    assert [item["arm"] for item in rank_contrast_candidates([later_arm, earlier_arm])] == [
        "decile_00_10__50_60", "decile_90_100__50_60"
    ]
    later_summary = stub(aggregation="q90", method="clean_residual")
    earlier_summary = stub(aggregation="mean", method="raw_gap")
    assert [
        item["aggregation"] for item in rank_contrast_candidates(
            [later_summary, earlier_summary]
        )
    ] == ["mean", "q90"]
    later_method = stub(method="raw_gap")
    earlier_method = stub(method="clean_residual")
    assert [
        item["method"] for item in rank_contrast_candidates([later_method, earlier_method])
    ] == ["clean_residual", "raw_gap"]


@pytest.mark.parametrize(
    "field, value",
    [
        ("arm", "decile_30_40__50_60"),   # a bucket pair no arm declares
        ("aggregation", "median"),        # a fourth summary
        ("method", "log_ratio"),          # a fifth score method
        ("method", "raw_reference"),      # and the reference control, by the same clause
    ],
)
def test_a_candidate_outside_the_three_closed_sets_is_not_deployable(field, value):
    """Three of the spec's seven eligibility items, and the only thing excluding the control.

    `rank_contrast_candidates` takes a list of dictionaries rather than one pipeline stage's
    output, so "one of the four declared arms, one of the three matched summaries, one of the
    four declared score methods" are checks and not restatements of what the caller must already
    have got right.
    """
    outsider = stub(**{field: value})
    assert value not in ARM_NAMES | set(AGGREGATIONS) | set(SCORE_METHODS)
    assert rank_contrast_candidates([outsider]) == []
    assert outsider["deployable"] is False
    assert rank_contrast_candidates([stub()]) != []  # and the same stub without the change ranks


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
