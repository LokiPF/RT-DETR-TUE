"""What the arm table turns into, and the two things about it that nothing else can see.

Most of this file is shape: how many rows, under which names, carrying which columns. That
matters because `build_contrast_rows` is the only place the experiment's structure is written
down, and a row that is missing, duplicated or filed under the wrong arm is not a wrong number
-- it is a candidate that silently does not exist, or two candidates that lock opposite
orientations on one series.

The rest is the cross-fitting, and it is here because it is invisible everywhere else. A
residual scored from the line its own image helped fit is smaller than it should be, by exactly
the amount of that image's influence -- smallest for the images least like the rest, which are
the ones a residual exists to flag. Every number stays plausible. `crossfit_score` is a fixture
built so the fold-0 line and the final line disagree loudly, which is the only way an assertion
can tell which one a row was scored from.

Two mutations this file guards against are ones the shared fixture cannot see on its own.
`contrast_test_utils.default_score` does not vary with `aggregation`, so a `_curve` that read
`mean` where it meant `q90` produces byte-identical rows -- `test_the_three_aggregations_are_
three_series` rewrites a bundle's score column to break that tie. And every arm that is first
with its bucket pair happens to have `name == pair_name`, so labelling the confidence twin by
arm rather than by pair is currently a no-op -- `test_the_confidence_twin_is_labelled_by_pair_
and_not_by_whichever_arm_came_first` reorders the table until it stops being one.
"""
import csv
from statistics import median

import pytest

from src.scene_uncertainty import contrast_analysis
from src.scene_uncertainty.contrast_analysis import (
    CONTRAST_CANDIDATE_KEY,
    CONTRAST_ROW_FIELDS,
    CONTRAST_ROW_KEY,
    DEPLOYABLE_SIGNAL,
    ContrastAnalysisError,
    build_anchor_diagnostics,
    build_contrast_rows,
    summarize_contrast_candidates,
)
from src.scene_uncertainty.contrast_inputs import (
    AGGREGATIONS,
    ARMS,
    REQUIRED_SERIES,
    ContrastInputError,
    load_contrast_inputs,
)
from src.scene_uncertainty.contrast_scores import (
    FOLD_COUNT,
    SCORE_METHODS,
    assign_folds,
    robust_line,
)

from tests.scene_uncertainty.contrast_test_utils import default_score, write_source_bundle

IMAGES = 6
SEVERITIES = 6


def loaded(tmp_path, *, images=IMAGES, **kwargs):
    source = write_source_bundle(tmp_path / "source", image_ids=range(1, images + 1), **kwargs)
    return load_contrast_inputs(source, expected_image_count=images)


def crossfit_score(image_id, severity, confidence_bin, signal, scope):
    """A reference/responsive pair whose fold-0 line differs sharply from the final line.

    reference = image_id; responsive = image_id + [0, 5, 5, 5, 0, 0][image_id - 1].
    Fold 0 holds images 1 and 6, so its four training images carry all three of the lifts and
    its fitted slope is 1/6 against the final line's 1. Every other fold trains on at least one
    unlifted image and fits a slope of exactly 1, so the fold that matters is the one the test
    reads and the difference is 3.75 of residual on image 1 -- far outside any tolerance.
    """
    if signal != "persistence" or scope != "layer_2":
        return default_score(image_id, severity, confidence_bin, signal, scope)
    if confidence_bin == "decile_00_10":
        return float(image_id)
    if confidence_bin == "decile_50_60":
        return float(image_id) + [0.0, 5.0, 5.0, 5.0, 0.0, 0.0][image_id - 1]
    return default_score(image_id, severity, confidence_bin, signal, scope)


def index(rows):
    return {tuple(row[field] for field in CONTRAST_ROW_KEY): row for row in rows}


def rewrite_scores(source, rescore):
    """Rewrite a written bundle's `score` column in place, leaving every other column alone.

    `rescore` is handed the row and its score and must not vary with `severity`. That is not a
    convenience: the four trend columns beside the score were computed by
    `complete_trend_metrics` on the curve as first written, and a per-severity edit would leave
    them describing a curve the file no longer contains. An offset that is constant along a
    curve changes no rank, so `signed_spearman`, `absolute_spearman` and `direction` stay true
    of the rewritten rows and the fixture keeps the self-consistency it is built on.
    """
    path = source / "per_scene.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames)
        rows = list(reader)
    for row in rows:
        row["score"] = rescore(row, float(row["score"]))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return source


AGGREGATION_STEP = {"mean": 0.0, "q90": 1.0, "top20_mean": 2.0}
BIN_STEP = {
    "decile_00_10": 0.001, "quintile_00_20": 0.002, "quintile_40_60": 0.003,
    "decile_50_60": 0.004, "decile_90_100": 0.005,
}


def aggregation_offset(row, score):
    """Push the three aggregations apart, by a different amount in each bin.

    Different per bin as well as per aggregation, so the offset does not cancel inside a gap:
    an aggregation term that moved both bins of an arm equally would change `raw_responsive`
    and leave `raw_gap` and the fitted line exactly where they were, and a `_curve` that read
    the wrong aggregation would still be caught on only one of the four methods.

    The sign follows the signal so both columns stay inside their own bounds -- persistence is
    a distance and rises, `1 - confidence` is capped at 1.0 and falls. The largest offset is
    0.010 against a smallest fixture score of 0.055 and a largest of 0.991.
    """
    offset = AGGREGATION_STEP[row["aggregation"]] * BIN_STEP[row["confidence_bin"]]
    return score + (offset if row["signal"] == "persistence" else -offset)


def test_every_arm_summary_and_method_produces_a_full_grid(tmp_path):
    rows, _ = build_contrast_rows(loaded(tmp_path))
    persistence = [row for row in rows if row["signal"] == "persistence"]
    confidence = [row for row in rows if row["signal"] == "confidence"]
    # three layer_2 arms at four methods plus the combined arm at three: fifteen combinations
    assert len(persistence) == 15 * 3 * IMAGES * SEVERITIES
    assert len(confidence) == 3 * 3 * len(SCORE_METHODS) * IMAGES * SEVERITIES
    assert len(index(rows)) == len(rows)  # no duplicate row keys
    # and the same source twice is the same list: nothing here depends on set or dict ordering
    again, _ = build_contrast_rows(loaded(tmp_path))
    assert rows == again


def test_the_row_key_and_columns_are_what_a_published_table_needs(tmp_path):
    """Pinned as literals, because every consumer of this module groups or writes by them.

    `index` above builds its keys out of `CONTRAST_ROW_KEY`, so a mutation of the constant moves
    the test's own expectation with it and every duplicate check in this file goes quiet. The
    literal here is what stops that. `CONTRAST_ROW_FIELDS` is pinned in order as well as by
    membership because Task 5 writes it as a CSV header, and a reordered header is a file whose
    columns no longer mean what a previous run's did.
    """
    assert CONTRAST_ROW_KEY == ("image_id", "severity", "arm", "signal", "aggregation", "method")
    assert CONTRAST_ROW_FIELDS == (
        "image_id", "severity", "arm", "signal", "aggregation", "method",
        "arm_family", "declared_before_data", "score_scope",
        "reference_bin", "responsive_bin", "reference", "responsive", "score",
        "fold", "fit_slope", "fit_offset",
    )
    rows, _ = build_contrast_rows(loaded(tmp_path))
    assert all(tuple(row) == CONTRAST_ROW_FIELDS for row in rows)


def test_the_relative_gap_is_unavailable_at_the_combined_scope(tmp_path):
    """A signed z-score has no symmetric relative gap; the reason is recorded, not silent."""
    rows, fits = build_contrast_rows(loaded(tmp_path))
    combined = {
        row["method"] for row in rows
        if row["arm"] == "decile_90_100__50_60__combined"
    }
    assert combined == {"raw_responsive", "raw_gap", "clean_residual"}
    layered = {
        row["method"] for row in rows if row["arm"] == "decile_90_100__50_60"
        and row["signal"] == "persistence"
    }
    assert layered == set(SCORE_METHODS)
    fit = fits[("decile_90_100__50_60__combined", "persistence", "mean")]
    assert fit["relative_gap_available"] is False
    # both halves of the recorded reason, because Task 8 publishes the string and a reader who
    # gets only "it is a signed z-score" is not told which property of the method that breaks
    assert fit["relative_gap_unavailable_reason"] == (
        "the combined scope is a signed z-score, and the symmetric relative gap's scale "
        "invariance and +/-2 bounds both require non-negative inputs"
    )
    assert "signed z-score" in fit["relative_gap_unavailable_reason"]
    assert fits[("decile_90_100__50_60", "persistence", "mean")][
        "relative_gap_available"
    ] is True
    # the confidence twin is not a signed scope either, and keeps all four methods
    twin = fits[("decile_90_100__50_60", "confidence", "mean")]
    assert twin["relative_gap_available"] is True
    assert twin["relative_gap_unavailable_reason"] is None


def test_the_two_differential_arms_share_one_confidence_twin(tmp_path):
    rows, _ = build_contrast_rows(loaded(tmp_path))
    twins = {row["arm"] for row in rows if row["signal"] == "confidence"}
    assert twins == {
        "decile_00_10__50_60", "quintile_00_20__40_60", "decile_90_100__50_60"
    }
    assert "decile_90_100__50_60__combined" not in twins
    # the set alone would survive emitting the shared twin twice under the same label, which is
    # exactly what a `_signal_plan` without its first-with-pair guard does. The count is what
    # says it was emitted once: three aggregations, four methods, six images, six severities.
    shared = [
        row for row in rows
        if row["signal"] == "confidence" and row["arm"] == "decile_90_100__50_60"
    ]
    assert len(shared) == len(AGGREGATIONS) * len(SCORE_METHODS) * IMAGES * SEVERITIES


def test_persistence_rows_carry_their_arm_scope_and_provenance(tmp_path):
    rows, _ = build_contrast_rows(loaded(tmp_path))
    by_key = index(rows)
    combined = by_key[
        (1, 0, "decile_90_100__50_60__combined", "persistence", "mean", "raw_gap")
    ]
    assert combined["score_scope"] == "combined"
    assert combined["arm_family"] == "differential"
    assert combined["declared_before_data"] is False
    assert combined["reference_bin"] == "decile_90_100"
    assert combined["responsive_bin"] == "decile_50_60"
    anchored = by_key[(1, 0, "decile_00_10__50_60", "persistence", "mean", "raw_gap")]
    assert anchored["score_scope"] == "layer_2"
    assert anchored["arm_family"] == "anchored"
    assert anchored["declared_before_data"] is True
    assert anchored["reference_bin"] == "decile_00_10"
    assert anchored["responsive_bin"] == "decile_50_60"
    # the twin inherits the provenance of the arm that owns the pair, not a default
    twin = by_key[(1, 0, "decile_90_100__50_60", "confidence", "mean", "raw_gap")]
    assert twin["score_scope"] == "confidence"
    assert twin["arm_family"] == "differential"
    assert twin["declared_before_data"] is False
    assert (twin["reference_bin"], twin["responsive_bin"]) == ("decile_90_100", "decile_50_60")


def test_the_two_differential_arms_read_different_scopes(tmp_path):
    """`combined` is a signed z-score, not `layer_2` shifted, and the rows have to show it."""
    inputs = loaded(tmp_path)
    rows, _ = build_contrast_rows(inputs)
    by_key = index(rows)
    layer = by_key[(2, 1, "decile_90_100__50_60", "persistence", "q90", "raw_responsive")]
    combined = by_key[
        (2, 1, "decile_90_100__50_60__combined", "persistence", "q90", "raw_responsive")
    ]
    assert layer["score"] != combined["score"]
    assert layer["score"] > 0.0 > combined["score"]

    # ... and the two are not one series plus a constant, which is what an offset fixture would
    # make them: an offset cancels in `raw_gap` and in `clean_residual`, so two of arm 4's three
    # methods would be arm 3's answers under a different name.
    def responsives(arm):
        return {
            (row["image_id"], row["severity"]): row["responsive"]
            for row in rows
            if row["arm"] == arm and row["signal"] == "persistence"
            and row["aggregation"] == "q90" and row["method"] == "raw_responsive"
        }

    layered, combineds = responsives("decile_90_100__50_60"), responsives(
        "decile_90_100__50_60__combined"
    )
    assert layered.keys() == combineds.keys()
    differences = {
        round(combineds[key] - layered[key], 9) for key in layered
    }
    assert len(differences) == IMAGES * SEVERITIES
    assert min(combineds.values()) < 0.0 <= min(layered.values())


def test_raw_responsive_and_raw_gap_match_their_inputs(tmp_path):
    rows, _ = build_contrast_rows(loaded(tmp_path))
    by_key = index(rows)
    key = (3, 2, "decile_00_10__50_60", "persistence", "mean")
    control = by_key[(*key, "raw_responsive")]
    gap = by_key[(*key, "raw_gap")]
    relative = by_key[(*key, "relative_gap")]
    assert control["score"] == control["responsive"]
    assert gap["score"] == pytest.approx(gap["responsive"] - gap["reference"])
    assert control["reference"] == gap["reference"]
    # the fourth method is value-checked too, and against its formula rather than against the
    # other three. Everywhere else in this file `relative_gap` appears only as a name inside a
    # set of method names, so computing it as a raw gap or as the responsive alone would satisfy
    # every count and every membership assertion. On this row the three are -0.063259, -0.004712
    # and 0.072131, so the confusion is an order of magnitude, not a rounding difference.
    reference, responsive = relative["reference"], relative["responsive"]
    assert relative["score"] == pytest.approx(
        2.0 * (responsive - reference) / (responsive + reference)
    )
    assert relative["score"] == pytest.approx(-0.063259, abs=1e-6)
    assert relative["score"] != pytest.approx(gap["score"])
    assert relative["score"] != pytest.approx(control["score"])


def test_rows_carry_the_source_scores_their_arm_names(tmp_path):
    """`reference` and `responsive` are read back against the bundle, not against each other.

    Every within-row assertion in this file is symmetric under swapping the two: `raw_responsive`
    returns whatever sits in `responsive`, and `raw_gap` returns their signed difference, so a
    `_curve` pair built the wrong way round is perfectly self-consistent and only the *sign* of
    every contrast in the experiment is inverted. The source is the only thing that can tell
    them apart, and the two bins differ -- 0.0768 against 0.0721 on this row -- so the swap is
    visible rather than a wash.
    """
    inputs = loaded(tmp_path)
    rows, _ = build_contrast_rows(inputs)
    for row in rows:
        for side in ("reference", "responsive"):
            assert row[side] == inputs.scores[(
                row["image_id"], row["severity"], row["signal"],
                row[f"{side}_bin"], row["aggregation"], row["score_scope"],
            )], (side, row["arm"], row["image_id"], row["severity"])
    anchored = index(rows)[(3, 2, "decile_00_10__50_60", "persistence", "mean", "raw_gap")]
    assert anchored["reference"] == 0.076843
    assert anchored["responsive"] == 0.072131


def test_the_three_aggregations_are_three_series(tmp_path):
    """The shared fixture makes the three summaries identical; this one does not.

    `contrast_test_utils.default_score` is not handed the aggregation, so every `mean`, `q90`
    and `top20_mean` row of a default bundle carries the same number -- and a `_curve` that
    ignored its `aggregation` argument, or hardcoded one, would produce a grid nothing in this
    file could distinguish from the right one. That is the fixture's own recorded gap. Here the
    written bundle is rewritten so the three come apart, by a different step in each bin so the
    difference survives the subtraction inside `raw_gap` as well as showing up in
    `raw_responsive`.
    """
    source = rewrite_scores(
        write_source_bundle(tmp_path / "source"), aggregation_offset
    )
    inputs = load_contrast_inputs(source, expected_image_count=IMAGES)
    rows, fits = build_contrast_rows(inputs)
    by_key = index(rows)

    for method in SCORE_METHODS:
        scores = {
            aggregation: by_key[
                (2, 3, "decile_00_10__50_60", "persistence", aggregation, method)
            ]["score"]
            for aggregation in AGGREGATIONS
        }
        assert len(set(scores.values())) == len(AGGREGATIONS), method

    # the fitted lines are three lines too, not one reused three times
    lines = {
        aggregation: tuple(fits[("decile_00_10__50_60", "persistence", aggregation)]["final_line"])
        for aggregation in AGGREGATIONS
    }
    assert len(set(lines.values())) == len(AGGREGATIONS)

    # and the diagnostics read their own aggregation too. They call `_curve` separately, so a
    # hardcoded summary there is a second copy of the same mutation that the rows above cannot
    # see -- 21 diagnostics would still be produced, each describing the wrong series.
    levels = {
        row["aggregation"]: row["spread"][0]["median"]
        for row in build_anchor_diagnostics(inputs)
        if row["arm"] == "decile_00_10__50_60" and row["signal"] == "persistence"
    }
    assert len(set(levels.values())) == len(AGGREGATIONS)


def test_all_six_severities_of_an_image_share_its_fold(tmp_path):
    rows, _ = build_contrast_rows(loaded(tmp_path))
    folds = assign_folds(range(1, IMAGES + 1))
    for row in rows:
        assert row["fold"] == folds[row["image_id"]]
    # `folds` above is the same function the production code calls, so the loop proves the rows
    # agree with it and not that it assigned anything. Six images over five folds is
    # `{0, 1, 2, 3, 4}` with fold 0 twice; a constant map would satisfy every assertion above.
    assert {row["fold"] for row in rows} == set(range(FOLD_COUNT))
    assert sorted(folds.values()) == [0, 0, 1, 2, 3, 4]


def test_residual_uses_its_own_fold_line_and_not_the_final_line(tmp_path):
    rows, fits = build_contrast_rows(loaded(tmp_path, score=crossfit_score))
    key = ("decile_00_10__50_60", "persistence", "mean")
    fit = fits[key]
    fold_slope, fold_offset = fit["fold_lines"][0]
    final_slope, final_offset = fit["final_line"]
    assert (fold_slope, fold_offset) != (final_slope, final_offset)
    # pinned, because every assertion below is self-consistent under a `robust_line` whose two
    # arguments are swapped or whose clean side is taken from the wrong curve: the stored line
    # would move, and the residual computed from the stored line would move with it.
    assert (final_slope, final_offset) == (1.0, 2.5)
    assert fold_slope == pytest.approx(1 / 6)
    assert fold_offset == pytest.approx(85 / 12)

    by_key = index(rows)
    # Both images of fold 0, and the second one is not optional. `crossfit_score` sets
    # `reference = float(image_id)`, so on image 1 the reference is exactly 1.0 and
    # `offset + slope * 1 == slope + offset * 1` -- a line passed to `contrast_score` with its
    # two halves transposed scores image 1 identically, and `fit_slope`/`fit_offset` are read
    # back off the stored fit rather than off what the score was computed with, so all three
    # assertions below pass on a transposed call. `robust_line` returns `(slope, offset)` and
    # `contrast_score` unpacks positionally, which is exactly the shape of refactor slip that
    # produces one. Image 6 has reference 6.0 and separates -25/12 from -110/3.
    for image_id, expected in ((1, -6.25), (6, -25 / 12)):
        row = by_key[
            (image_id, 0, "decile_00_10__50_60", "persistence", "mean", "clean_residual")
        ]
        assert row["fold"] == 0
        reference, responsive = row["reference"], row["responsive"]
        assert row["score"] == pytest.approx(
            responsive - (fold_offset + fold_slope * reference)
        )
        assert row["score"] == pytest.approx(expected)
        assert row["score"] != pytest.approx(
            responsive - (final_offset + final_slope * reference)
        )
        if reference != 1.0:  # where the transposition is not a fixed point
            assert row["score"] != pytest.approx(
                responsive - (fold_slope + fold_offset * reference)
            )
        assert row["fit_slope"] == pytest.approx(fold_slope)
        assert row["fit_offset"] == pytest.approx(fold_offset)


def test_only_a_residual_row_carries_the_line_it_was_scored_from(tmp_path):
    """The other three methods never see a line, so a line on their row would be decoration.

    Worse than decoration: `fit_slope` and `fit_offset` are how a reader recomputes a residual
    by hand, and a raw-gap row carrying them says a subtraction it did not perform was performed.
    """
    rows, _ = build_contrast_rows(loaded(tmp_path))
    for row in rows:
        if row["method"] == "clean_residual":
            assert row["fit_slope"] is not None and row["fit_offset"] is not None, row["arm"]
        else:
            assert row["fit_slope"] is None and row["fit_offset"] is None, row["method"]


def test_only_severity_zero_rows_influence_a_fit(tmp_path):
    """Changing corrupted severities alone must leave every fitted line untouched."""
    def corrupted_only(image_id, severity, confidence_bin, signal, scope):
        value = default_score(image_id, severity, confidence_bin, signal, scope)
        if severity > 0 and signal == "persistence":
            return value + 7.0
        return value

    _, baseline = build_contrast_rows(loaded(tmp_path / "a"))
    _, shifted = build_contrast_rows(loaded(tmp_path / "b", score=corrupted_only))
    assert baseline.keys() == shifted.keys()
    for key in baseline:
        assert baseline[key]["final_line"] == shifted[key]["final_line"]
        assert baseline[key]["fold_lines"] == shifted[key]["fold_lines"]


def test_twenty_one_final_lines_are_stored(tmp_path):
    _, fits = build_contrast_rows(loaded(tmp_path))
    persistence = [key for key in fits if key[1] == "persistence"]
    confidence = [key for key in fits if key[1] == "confidence"]
    assert len(persistence) == 12  # four arms x three summaries
    assert len(confidence) == 9  # three bucket pairs x three summaries
    assert len(fits) == 21


def test_a_constant_reference_makes_the_residual_unavailable(tmp_path):
    def constant_reference(image_id, severity, confidence_bin, signal, scope):
        if signal == "persistence" and confidence_bin == "decile_00_10":
            return 2.0
        return default_score(image_id, severity, confidence_bin, signal, scope)

    rows, fits = build_contrast_rows(loaded(tmp_path, score=constant_reference))
    fit = fits[("decile_00_10__50_60", "persistence", "mean")]
    assert fit["final_line"] is None
    assert fit["residual_available"] is False
    assert "constant" in fit["unavailable_reason"]
    residuals = [
        row for row in rows
        if row["arm"] == "decile_00_10__50_60"
        and row["signal"] == "persistence"
        and row["method"] == "clean_residual"
    ]
    assert residuals == []
    # the other three methods survive a constant reference
    survivors = {
        row["method"] for row in rows
        if row["arm"] == "decile_00_10__50_60" and row["signal"] == "persistence"
    }
    assert survivors == {"raw_responsive", "raw_gap", "relative_gap"}


def test_one_unfittable_fold_withdraws_the_whole_residual(tmp_path):
    """A final line can exist while one fold's cannot, and four fifths of a residual is not one.

    Fold 0 holds images 1 and 6, so it trains on 2 through 5. Give those four one shared clean
    reference and leave 1 and 6 their own, and the all-clean line is perfectly fittable while
    fold 0's is not. Publishing the four folds that worked would drop a fifth of the images from
    a candidate, and nothing downstream re-counts a candidate's rows -- so the missing images
    would reach a macro AUROC as a smaller sample rather than as an absence.
    """
    def constant_within_fold_zeros_training_set(
        image_id, severity, confidence_bin, signal, scope
    ):
        if signal == "persistence" and confidence_bin == "decile_00_10":
            return {1: 1.0, 6: 3.0}.get(image_id, 2.0)
        return default_score(image_id, severity, confidence_bin, signal, scope)

    rows, fits = build_contrast_rows(
        loaded(tmp_path, score=constant_within_fold_zeros_training_set)
    )
    fit = fits[("decile_00_10__50_60", "persistence", "mean")]
    assert fit["final_line"] is not None
    assert fit["fold_lines"][0] is None
    assert all(fit["fold_lines"][fold] is not None for fold in range(1, FOLD_COUNT))
    assert fit["residual_available"] is False
    assert "folds [0]" in fit["unavailable_reason"]
    assert "constant" in fit["unavailable_reason"]
    assert not [
        row for row in rows
        if row["arm"] == "decile_00_10__50_60" and row["signal"] == "persistence"
        and row["method"] == "clean_residual"
    ]


def test_a_short_roster_leaves_its_empty_folds_out_of_the_fit(tmp_path):
    """Three images fill three of five folds, and the two empty ones are absent, not `None`.

    A `None` fold line means "this fold could not be fitted", which is a finding. A fold with no
    images is not a finding, and recording one would make `residual_available` false for every
    candidate on any roster smaller than five.
    """
    rows, fits = build_contrast_rows(loaded(tmp_path, images=3))
    fit = fits[("decile_00_10__50_60", "persistence", "mean")]
    assert sorted(fit["fold_lines"]) == [0, 1, 2]
    assert fit["residual_available"] is True
    assert {row["fold"] for row in rows} == {0, 1, 2}
    assert len(rows) == (15 + 3 * len(SCORE_METHODS)) * 3 * 3 * SEVERITIES


def test_the_confidence_twin_is_labelled_by_pair_and_not_by_whichever_arm_came_first(
    tmp_path, monkeypatch
):
    """Reordering the arm table must not rename a series that nine tasks read.

    In the declared table every arm that is first with its bucket pair has `name == pair_name`,
    so `arm.name` and `arm.pair_name` are the same string on the only line where the difference
    could show, and the mutation that labels the twin by arm passes the whole suite. Putting the
    `combined` differential arm ahead of its `layer_2` sibling is the smallest change that makes
    the two disagree -- and it is not a hypothetical, because the arm table is a literal tuple
    and reordering it looks like a cosmetic edit.
    """
    reordered = (ARMS[0], ARMS[1], ARMS[3], ARMS[2])
    monkeypatch.setattr(contrast_analysis, "ARMS", reordered)
    inputs = loaded(tmp_path)
    rows, fits = build_contrast_rows(inputs)
    expected = {"decile_00_10__50_60", "quintile_00_20__40_60", "decile_90_100__50_60"}
    twins = {row["arm"] for row in rows if row["signal"] == "confidence"}
    assert twins == expected
    assert {key[0] for key in fits if key[1] == "confidence"} == twins
    # and the persistence rows still separate the two scopes under their own names
    assert {row["arm"] for row in rows if row["signal"] == "persistence"} == {
        arm.name for arm in ARMS
    }
    assert len(index(rows)) == len(rows)
    assert len(fits) == 21
    # the diagnostics label their arms from the same plan, and inherit the same trap
    diagnostics = build_anchor_diagnostics(inputs)
    assert {
        row["arm"] for row in diagnostics if row["signal"] == "confidence"
    } == expected
    assert len({(row["arm"], row["signal"], row["aggregation"]) for row in diagnostics}) == 21


def test_two_arms_that_would_share_a_series_are_refused(tmp_path, monkeypatch):
    """An arm table with a repeated name silently overwrites a fit and doubles its rows.

    Refused rather than deduplicated. Two arms with one name are two different experiments
    filed under one heading, and picking either one for the reader is picking which of two
    hypotheses gets reported.
    """
    from dataclasses import replace

    clash = replace(ARMS[1], name=ARMS[0].name)
    monkeypatch.setattr(contrast_analysis, "ARMS", (ARMS[0], clash, ARMS[2], ARMS[3]))
    inputs = loaded(tmp_path)
    with pytest.raises(ContrastAnalysisError, match="same contrast series"):
        build_contrast_rows(inputs)
    # the diagnostics refuse it too. Nothing there is keyed by the series, so a clash appends a
    # duplicate to a list instead of overwriting a dictionary entry -- 24 rows where 21 are
    # documented, and no signal to a Task 7 or 8 caller that reads them without the rows.
    with pytest.raises(ContrastAnalysisError, match="same contrast series"):
        build_anchor_diagnostics(inputs)


def test_the_refusal_is_a_value_error_so_the_cli_prints_one_line(tmp_path):
    """`pipeline` catches `ValueError`; a bare `Exception` base would give the operator a
    traceback through frames they did not write, and every other assertion in this file passes.
    """
    assert issubclass(ContrastAnalysisError, ValueError)
    assert issubclass(ContrastInputError, ValueError)


def test_anchor_diagnostics_cover_every_arm_and_summary(tmp_path):
    diagnostics = build_anchor_diagnostics(loaded(tmp_path))
    keys = {(row["arm"], row["signal"], row["aggregation"]) for row in diagnostics}
    assert len(keys) == 21
    assert len(diagnostics) == 21
    families = {row["arm"]: row["arm_family"] for row in diagnostics}
    assert families["decile_90_100__50_60__combined"] == "differential"
    assert families["decile_00_10__50_60"] == "anchored"
    assert all(
        set(row) == {
            "arm", "arm_family", "declared_before_data", "signal", "aggregation",
            "score_scope", "reference_bin", "responsive_bin", "drift", "spread",
            "relationship",
        }
        for row in diagnostics
    )
    combined = next(
        row for row in diagnostics
        if row["arm"] == "decile_90_100__50_60__combined" and row["aggregation"] == "q90"
    )
    assert combined["score_scope"] == "combined"
    assert combined["declared_before_data"] is False
    assert (combined["reference_bin"], combined["responsive_bin"]) == (
        "decile_90_100", "decile_50_60"
    )


def test_every_diagnostic_names_which_bin_is_which_way_round(tmp_path):
    """The two bin labels are checked for *order*, which membership structurally cannot do.

    `test_every_diagnostic_names_a_series_the_loader_actually_read` reads both bin fields, and
    it can never bind their order: `contrast_inputs.required_series` iterates them symmetrically
    -- `for confidence_bin in (arm.reference_bin, arm.responsive_bin)` -- so a swapped pair lands
    back inside `REQUIRED_SERIES` and the membership clause is satisfied by both orders. That
    symmetry is deliberate; the `Arm` docstring says the asymmetry gets bound one task
    downstream. It is, on the rows path, by `test_rows_carry_the_source_scores_their_arm_names`,
    which reads each row's `reference`/`responsive` *value* back out of `inputs.scores` keyed by
    that row's own bin fields. The diagnostics publish no per-bin values, so no analogous
    read-back existed and only a single spot-read on arm 4's `q90` row bound anything -- leaving
    a swap on arm 1 alone, on the three twins, or on any 20 of the 21 rows entirely unbound.

    Two clauses, because they fail independently.

    The first is that read-back, in the only form the diagnostics allow: `clean_relationship`
    consumes both clean curves, so the line it publishes can be recomputed from the source keyed
    by the row's *own* two labels and must come back bit-identical. `robust_line` is
    deterministic, so this is exact equality rather than a tolerance, and it needs nothing from
    the arm table -- it asks the labels to answer to the numbers printed beside them.

    The second compares the ordered pair against the arm table directly, which is what binds
    `responsive_bin` when a mutation fabricates it without touching `reference_bin`. It is not
    circular: Task 1's `test_arm_table_is_the_four_declared_arms` pins all eight fields of all
    four arms as literals, so the table this reads is independently held down.
    """
    inputs = loaded(tmp_path)
    diagnostics = build_anchor_diagnostics(inputs)
    assert len(diagnostics) == 21

    for row in diagnostics:
        def clean(confidence_bin):
            return [
                inputs.scores[(
                    image_id, 0, row["signal"], confidence_bin,
                    row["aggregation"], row["score_scope"],
                )]
                for image_id in inputs.image_ids
            ]

        recomputed = robust_line(clean(row["reference_bin"]), clean(row["responsive_bin"]))
        relationship = row["relationship"]
        assert recomputed == (relationship["final_slope"], relationship["final_offset"]), (
            row["arm"], row["signal"], row["aggregation"]
        )
        # ... and the reverse reading is a different line on every one of the 21, so the
        # equality above is a statement about order and not an accident of symmetry
        assert robust_line(
            clean(row["responsive_bin"]), clean(row["reference_bin"])
        ) != recomputed, (row["arm"], row["signal"], row["aggregation"])

    declared = {}
    for arm in ARMS:
        for label in (arm.name, arm.pair_name):
            declared.setdefault(label, (arm.reference_bin, arm.responsive_bin))
    for row in diagnostics:
        assert (row["reference_bin"], row["responsive_bin"]) == declared[row["arm"]], (
            row["arm"], row["signal"], row["aggregation"]
        )


def test_every_diagnostic_names_a_series_the_loader_actually_read(tmp_path):
    """`score_scope` has to be the scope the numbers beside it were measured at.

    Checked on all 21 rows against `REQUIRED_SERIES` rather than by spot-reading one, because
    the field is only *wrong* on the three confidence twins and only there. A twin's scope comes
    from the signal plan, not from its arm -- confidence has no decoder layer -- so writing
    `arm.score_scope` here labels all three `layer_2`, which is a scope the confidence signal
    does not have at all. Every count, every key and every numeric assertion survives it: the
    scores are still read at the right scope, only the label shipped beside them is wrong, and
    Task 8 writes that label into `anchor_diagnostics.csv`.

    `REQUIRED_SERIES` is the right yardstick because it is the loader's own list of what the arm
    table needs, so a row naming a series outside it is naming something that was never loaded.
    """
    diagnostics = build_anchor_diagnostics(loaded(tmp_path))
    required = set(REQUIRED_SERIES)
    for row in diagnostics:
        for side in ("reference_bin", "responsive_bin"):
            assert (row["signal"], row[side], row["score_scope"]) in required, (
                row["arm"], row["signal"], side, row["score_scope"]
            )
    # and said plainly for the three twins, which are the only rows that can be wrong here
    twins = {
        row["score_scope"] for row in diagnostics if row["signal"] == "confidence"
    }
    assert twins == {"confidence"}
    scopes = {
        (row["arm"], row["score_scope"])
        for row in diagnostics if row["signal"] == "persistence"
    }
    assert scopes == {(arm.name, arm.score_scope) for arm in ARMS}


def test_a_differential_arm_reports_its_anchor_failure_rather_than_hiding_it(tmp_path):
    diagnostics = build_anchor_diagnostics(loaded(tmp_path))
    row = next(
        item for item in diagnostics
        if item["arm"] == "decile_90_100__50_60"
        and item["signal"] == "persistence"
        and item["aggregation"] == "mean"
    )
    # the fixture's 90-100 reference falls with blur, so it drifts and is not an anchor
    assert row["arm_family"] == "differential"
    assert row["drift"]["by_severity"][5]["median_absolute_drift"] > 0.0
    assert row["spread"][5]["stability_to_spread"] is not None


def test_the_published_stability_is_the_ratio_of_the_two_numbers_beside_it(tmp_path):
    """`stability_to_spread` must be built from the drift this row publishes, not another.

    `between_image_spread` divides a median absolute drift by the clean interquartile range, and
    Task 3's tests pin that arithmetic against *that function's own* arguments. What is open is
    Task 4's wiring: handing it `within_image_drift(responsives)` while the row publishes
    `within_image_drift(references)` leaves every number finite, every key present and every
    `is not None` assertion true, and only the value moves. On arm 1 it moves by a factor of 70,
    because that arm pairs the flattest reference in the fixture (`decile_00_10`, -0.00006 a
    step) with a responsive bin that climbs fifty times faster.

    So the check is the identity rather than a bound: the published ratio has to equal the
    published drift over the published clean spread. A pinned absolute sits beside it so that
    changing both halves in step is caught too.
    """
    diagnostics = build_anchor_diagnostics(loaded(tmp_path))
    for row in diagnostics:
        drift, spread = row["drift"]["by_severity"], row["spread"]
        clean_iqr = spread[0]["iqr"]
        assert spread[0]["stability_to_spread"] is None  # severity zero has no ratio
        for severity in range(1, SEVERITIES):
            ratio = spread[severity]["stability_to_spread"]
            if clean_iqr == 0.0:
                assert ratio is None, (row["arm"], severity)
            else:
                assert ratio == pytest.approx(
                    drift[severity]["median_absolute_drift"] / clean_iqr
                ), (row["arm"], row["signal"], row["aggregation"], severity)

    anchored = next(
        row for row in diagnostics
        if row["arm"] == "decile_00_10__50_60"
        and row["signal"] == "persistence"
        and row["aggregation"] == "mean"
    )
    # 0.0002115 of drift against 0.00926125 of clean spread: an anchor that holds still
    assert anchored["spread"][2]["stability_to_spread"] == pytest.approx(0.022837, abs=1e-6)
    assert anchored["spread"][0]["iqr"] == pytest.approx(0.00926125)
    assert anchored["drift"]["by_severity"][2]["median_absolute_drift"] == pytest.approx(
        0.0002115
    )


def test_the_diagnostics_measure_the_reference_and_not_the_responsive(tmp_path):
    """Both bins drift and both spread, so only a signed or a levelled number tells them apart.

    `median_absolute_drift` is positive for either bin of arm 3 and `stability_to_spread` is
    non-`None` for either, so the assertions above pass unchanged if `within_image_drift` and
    `between_image_spread` are handed the responsive curve. The two bins move in *opposite*
    directions -- the top decile falls with blur at -0.0037 a step while the middle decile
    rises at +0.0032 -- and they sit at different clean levels, which is what these two
    assertions read.
    """
    inputs = loaded(tmp_path)
    diagnostics = build_anchor_diagnostics(inputs)
    row = next(
        item for item in diagnostics
        if item["arm"] == "decile_90_100__50_60"
        and item["signal"] == "persistence"
        and item["aggregation"] == "mean"
    )
    assert row["drift"]["by_severity"][5]["median_signed_drift"] < 0.0
    assert row["spread"][0]["median"] == pytest.approx(median(
        inputs.scores[(image_id, 0, "persistence", "decile_90_100", "mean", "layer_2")]
        for image_id in inputs.image_ids
    ))
    # ... and the responsive bin, which is what a swap would report, is the other sign and level
    assert median(
        inputs.scores[(image_id, 5, "persistence", "decile_50_60", "mean", "layer_2")]
        - inputs.scores[(image_id, 0, "persistence", "decile_50_60", "mean", "layer_2")]
        for image_id in inputs.image_ids
    ) > 0.0


def test_the_diagnostics_clean_line_is_the_line_the_rows_were_scored_from(tmp_path):
    """One relationship per fit, and the two functions must not disagree about it.

    `clean_relationship` and `_fit` both regress the responsive clean level on the reference
    clean level over the same roster and the same folds, so their lines are the same line. A
    swap of the two arguments in either place -- which Pearson and Spearman are blind to,
    because both are symmetric -- shows up here as two different slopes for one candidate.
    """
    inputs = loaded(tmp_path, score=crossfit_score)
    _, fits = build_contrast_rows(inputs)
    for row in build_anchor_diagnostics(inputs):
        key = (row["arm"], row["signal"], row["aggregation"])
        relationship, fit = row["relationship"], fits[key]
        assert relationship["fold_lines"] == fit["fold_lines"], key
        if fit["final_line"] is None:
            assert relationship["final_slope"] is None, key
        else:
            assert [relationship["final_slope"], relationship["final_offset"]] == (
                fit["final_line"]
            ), key
    crossfit = next(
        row for row in build_anchor_diagnostics(inputs)
        if row["arm"] == "decile_00_10__50_60" and row["signal"] == "persistence"
        and row["aggregation"] == "mean"
    )["relationship"]
    assert (crossfit["final_slope"], crossfit["final_offset"]) == (1.0, 2.5)


def candidate_rows(curves, *, arm="decile_00_10__50_60", signal="persistence",
                   aggregation="mean", method="raw_gap"):
    """Rows for a single candidate from `{image_id: [six scores]}`."""
    return [
        {
            "image_id": image_id, "severity": severity, "arm": arm, "signal": signal,
            "aggregation": aggregation, "method": method, "arm_family": "anchored",
            "declared_before_data": True, "score_scope": "layer_2",
            "reference_bin": "decile_00_10", "responsive_bin": "decile_50_60",
            "reference": 1.0, "responsive": 1.0 + score, "score": score,
            "fold": image_id % 5, "fit_slope": None, "fit_offset": None,
        }
        for image_id, scores in curves.items()
        for severity, score in enumerate(scores)
    ]


def test_candidate_key_is_arm_signal_aggregation_method():
    assert CONTRAST_CANDIDATE_KEY == ("arm", "signal", "aggregation", "method")


def test_every_declared_candidate_appears(tmp_path):
    rows, _ = build_contrast_rows(loaded(tmp_path))
    candidates = summarize_contrast_candidates(rows, expected_image_count=IMAGES)
    persistence = [item for item in candidates if item["signal"] == "persistence"]
    confidence = [item for item in candidates if item["signal"] == "confidence"]
    # 45, not 48: the combined arm has no relative gap (signed z-score)
    assert len(persistence) == 45
    assert len(confidence) == 36
    assert len(candidates) == 81
    assert not [
        item for item in persistence
        if item["arm"].endswith("__combined") and item["method"] == "relative_gap"
    ]


def test_a_rising_candidate_locks_to_plus_one():
    rows = candidate_rows({image: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] for image in range(1, 5)})
    candidate = summarize_contrast_candidates(rows, expected_image_count=4)[0]
    assert candidate["median_signed_spearman"] == pytest.approx(1.0)
    assert candidate["orientation"] == 1
    assert candidate["macro_auroc"] == pytest.approx(1.0)


def test_a_falling_candidate_locks_to_minus_one_and_is_read_that_way():
    rows = candidate_rows({image: [5.0, 4.0, 3.0, 2.0, 1.0, 0.0] for image in range(1, 5)})
    candidate = summarize_contrast_candidates(rows, expected_image_count=4)[0]
    assert candidate["median_signed_spearman"] == pytest.approx(-1.0)
    assert candidate["orientation"] == -1
    # read in its locked direction a falling candidate separates perfectly, not at chance
    assert candidate["macro_auroc"] == pytest.approx(1.0)
    assert candidate["auroc_by_severity"][5] == pytest.approx(1.0)
    assert candidate["auroc_by_severity"] == {
        severity: pytest.approx(1.0) for severity in range(1, 6)
    }
    # the curve checks are read the same way round: a falling curve read upwards would
    # score 0.0 on both, so these two also pin that `orientation` reaches them
    assert candidate["oriented_adjacent_consistency"] == pytest.approx(1.0)
    assert candidate["max_blur_above_clean_rate"] == pytest.approx(1.0)
    assert candidate["negative_count"] == 4
    assert candidate["positive_count"] == 0
    assert candidate["image_count"] == 4
    assert candidate["expected_image_count"] == 4


def test_a_candidate_with_no_agreed_direction_is_unorientable():
    rising = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    falling = list(reversed(rising))
    rows = candidate_rows({1: rising, 2: rising, 3: falling, 4: falling})
    candidate = summarize_contrast_candidates(rows, expected_image_count=4)[0]
    assert candidate["median_signed_spearman"] == pytest.approx(0.0)
    assert candidate["orientation"] is None
    assert candidate["macro_auroc"] is None
    assert candidate["auroc_by_severity"] is None
    assert candidate["orientable"] is False
    # "no agreed direction" is not "no movement": every image moves as hard as it can, and
    # the absolute median says so. `abs(median_signed_spearman)` would report 0.0 here.
    assert candidate["median_absolute_spearman"] == pytest.approx(1.0)
    assert candidate["positive_count"] == 2
    assert candidate["negative_count"] == 2
    assert candidate["flat_count"] == 0
    # nothing oriented, so there is no oriented curve to check
    assert candidate["oriented_adjacent_consistency"] is None
    assert candidate["max_blur_above_clean_rate"] is None


def test_a_flat_curve_is_flat_and_an_incomplete_one_is_unmeasured():
    flat = [2.0] * 6
    rows = candidate_rows({1: flat, 2: flat, 3: flat, 4: flat})
    candidate = summarize_contrast_candidates(rows, expected_image_count=4)[0]
    assert candidate["flat_count"] == 4
    assert candidate["measured_count"] == 4
    assert candidate["missing_count"] == 0
    assert candidate["median_absolute_spearman"] == pytest.approx(0.0)
    # flat is a direction these images agreed on, so it is what dominates them: a
    # dominant fraction taken over rising and falling alone would report 0.0 here
    assert candidate["flat_fraction"] == pytest.approx(1.0)
    assert candidate["dominant_direction_fraction"] == pytest.approx(1.0)
    assert candidate["complete"] is True

    partial = candidate_rows({1: flat, 2: flat})
    partial = [row for row in partial if not (row["image_id"] == 2 and row["severity"] == 3)]
    incomplete = summarize_contrast_candidates(partial, expected_image_count=2)[0]
    assert incomplete["measured_count"] == 1
    assert incomplete["missing_count"] == 1
    assert incomplete["complete"] is False


def test_per_severity_auroc_is_computed_severity_by_severity():
    rows = candidate_rows({
        1: [0.0, 0.5, 1.0, 1.0, 1.0, 10.0],
        2: [1.0, 1.5, 2.0, 2.0, 2.0, 11.0],
        3: [2.0, 2.5, 3.0, 3.0, 3.0, 12.0],
        4: [3.0, 3.5, 4.0, 4.0, 4.0, 13.0],
    })
    candidate = summarize_contrast_candidates(rows, expected_image_count=4)[0]
    # 10 of the 16 clean/corrupted pairs separate at severity 1; all 16 at severity 5
    assert candidate["auroc_by_severity"][1] == pytest.approx(0.625)
    assert candidate["auroc_by_severity"][5] == pytest.approx(1.0)
    # severities 2, 3 and 4 are the same column of scores, and each ties one pair with clean:
    # 11.5 of the 16 pairs separate, which is 0.71875 and not the 0.75 a tie-blind count gives
    assert candidate["auroc_by_severity"][2] == pytest.approx(0.71875)
    assert candidate["auroc_by_severity"][3] == pytest.approx(0.71875)
    assert candidate["auroc_by_severity"][4] == pytest.approx(0.71875)
    # the macro is pinned to its arithmetic value as well as to its own parts, so that a
    # macro and a severity curve that are wrong in the same direction cannot agree with
    # each other and pass
    assert candidate["macro_auroc"] == pytest.approx(0.75625)
    assert candidate["macro_auroc"] == pytest.approx(
        sum(candidate["auroc_by_severity"][severity] for severity in range(1, 6)) / 5
    )


def test_macro_auroc_is_never_a_stand_in_for_severity_one():
    """A candidate strong only at severe blur must show it, not hide behind the mean."""
    rows = candidate_rows({
        1: [0.0, 0.0, 0.0, 0.0, 0.0, 9.0],
        2: [1.0, 1.0, 1.0, 1.0, 1.0, 9.5],
        3: [2.0, 2.0, 2.0, 2.0, 2.0, 10.0],
        4: [3.0, 3.0, 3.0, 3.0, 3.0, 10.5],
    })
    candidate = summarize_contrast_candidates(rows, expected_image_count=4)[0]
    assert candidate["auroc_by_severity"][1] == pytest.approx(0.5)
    assert candidate["auroc_by_severity"][5] == pytest.approx(1.0)
    assert candidate["macro_auroc"] > candidate["auroc_by_severity"][1]


def test_severity_statistics_are_published_for_the_plots():
    rows = candidate_rows({image: [float(image)] * 6 for image in range(1, 5)})
    candidate = summarize_contrast_candidates(rows, expected_image_count=4)[0]
    clean = candidate["severity_statistics"][0]
    assert clean["count"] == 4
    assert clean["mean"] == pytest.approx(2.5)
    assert clean["median"] == pytest.approx(2.5)
    assert clean["q25"] == pytest.approx(1.75)
    assert clean["q75"] == pytest.approx(3.25)
    # population variance over the four images, not the n-1 estimate's 1.6667
    assert clean["variance"] == pytest.approx(1.25)
    assert set(candidate["severity_statistics"]) == set(range(6))


def test_candidate_provenance_survives_from_the_rows(tmp_path):
    rows, _ = build_contrast_rows(loaded(tmp_path))
    candidates = summarize_contrast_candidates(rows, expected_image_count=IMAGES)
    combined = next(
        item for item in candidates
        if item["arm"] == "decile_90_100__50_60__combined"
        and item["method"] == "raw_gap" and item["aggregation"] == "mean"
    )
    assert combined["arm_family"] == "differential"
    assert combined["declared_before_data"] is False
    assert combined["score_scope"] == "combined"
    # the bin pair is ordered: `raw_gap` is responsive minus reference, so a summary that
    # recorded only which two bins were involved would leave the sign unrecoverable
    assert combined["reference_bin"] == "decile_90_100"
    assert combined["responsive_bin"] == "decile_50_60"
    anchored = next(
        item for item in candidates
        if item["arm"] == "decile_00_10__50_60" and item["signal"] == "persistence"
        and item["method"] == "raw_gap" and item["aggregation"] == "mean"
    )
    assert anchored["arm_family"] == "anchored"
    assert anchored["declared_before_data"] is True
    assert anchored["score_scope"] == "layer_2"
    assert anchored["reference_bin"] == "decile_00_10"
    assert anchored["responsive_bin"] == "decile_50_60"
    # and the twin that shares the differential pair reads confidence, not a decoder layer
    twin = next(
        item for item in candidates
        if item["arm"] == "decile_90_100__50_60" and item["signal"] == "confidence"
        and item["method"] == "raw_gap" and item["aggregation"] == "mean"
    )
    assert twin["score_scope"] == "confidence"
    assert twin["reference_bin"] == "decile_90_100"
    assert twin["responsive_bin"] == "decile_50_60"


def test_a_duplicated_row_key_is_refused():
    rows = candidate_rows({1: [0.0] * 6})
    with pytest.raises(ContrastAnalysisError, match="duplicate"):
        summarize_contrast_candidates(rows + rows[:1], expected_image_count=1)


def test_only_persistence_is_the_deployable_signal(tmp_path):
    """The signal Task 6 ranks, named once so the gate and the rows cannot drift apart.

    Pinned to the literal *and* checked against the signals the rows actually carry: a
    constant that named a signal no candidate has would gate everything out, and a gate that
    spelled `"persistence"` for itself would not notice the rows renaming it.
    """
    assert DEPLOYABLE_SIGNAL == "persistence"
    rows, _ = build_contrast_rows(loaded(tmp_path))
    candidates = summarize_contrast_candidates(rows, expected_image_count=IMAGES)
    assert {item["signal"] for item in candidates} == {"persistence", "confidence"}
    # both signals are summarised in full; it is only the ranking the control stays out of
    assert DEPLOYABLE_SIGNAL in {item["signal"] for item in candidates}
    assert "deployable" not in candidates[0]
    assert "rank" not in candidates[0]


def test_the_locked_direction_follows_the_median_and_not_the_mean():
    """Two loud images must not outvote three quiet ones, and the published median says so.

    `strong_up` correlates at +33/35 and `weak_down` at exactly -0.2, so two of the first
    against three of the second average to +0.257 and median to -0.2. A mean would read this
    candidate upwards; the median reads it the way most of its images actually move, which is
    the choice `corruption_metrics.choose_orientation` documents. Every other fixture in this
    file is symmetric enough that mean and median agree, so this is the only place the swap
    shows.
    """
    strong_up = [1.0, 2.0, 3.0, 4.0, 6.0, 5.0]
    weak_down = [5.0, 3.0, 2.0, 6.0, 1.0, 4.0]
    rows = candidate_rows({
        1: strong_up, 2: strong_up, 3: weak_down, 4: weak_down, 5: weak_down,
    })
    candidate = summarize_contrast_candidates(rows, expected_image_count=5)[0]
    assert candidate["median_signed_spearman"] == pytest.approx(-0.2)
    assert candidate["orientation"] == -1
    # the mean of the same five is +0.2571, and the mean of their absolutes is +0.4971
    assert candidate["median_absolute_spearman"] == pytest.approx(0.2)
    assert candidate["positive_count"] == 2
    assert candidate["negative_count"] == 3
    assert candidate["positive_fraction"] == pytest.approx(0.4)
    assert candidate["negative_fraction"] == pytest.approx(0.6)
    assert candidate["dominant_direction_fraction"] == pytest.approx(0.6)
    # and the locked direction is kept even though it scores the candidate below chance:
    # this is a finding about the candidate, not a sign to flip after the fact
    assert candidate["macro_auroc"] == pytest.approx(0.48)


def test_a_twin_locks_its_own_direction_and_is_not_handed_the_persistence_one(tmp_path):
    """`signal` is in the candidate key so the twin chooses its own orientation.

    A twin forced to share its candidate's direction would measure how well the candidate's
    direction happens to suit confidence, not how well confidence does on its own -- and on
    the shared fixture the two genuinely disagree, so a twin handed the persistence direction
    would be reported at `1 - auroc` of its real value.
    """
    rising = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    falling = list(reversed(rising))
    both = candidate_rows({image: rising for image in range(1, 5)})
    both += candidate_rows(
        {image: falling for image in range(1, 5)}, signal="confidence"
    )
    candidates = summarize_contrast_candidates(both, expected_image_count=4)
    by_signal = {item["signal"]: item for item in candidates}
    assert len(candidates) == 2
    assert by_signal["persistence"]["orientation"] == 1
    assert by_signal["confidence"]["orientation"] == -1
    assert by_signal["persistence"]["macro_auroc"] == pytest.approx(1.0)
    assert by_signal["confidence"]["macro_auroc"] == pytest.approx(1.0)

    rows, _ = build_contrast_rows(loaded(tmp_path))
    twins = {
        item["signal"]: item
        for item in summarize_contrast_candidates(rows, expected_image_count=IMAGES)
        if item["arm"] == "decile_90_100__50_60"
        and item["aggregation"] == "mean" and item["method"] == "raw_gap"
    }
    assert twins["persistence"]["orientation"] == 1
    assert twins["confidence"]["orientation"] == -1
    # the twin outscores the candidate it exists to control, which is exactly why
    # DEPLOYABLE_SIGNAL keeps it out of the ranking rather than out of the report
    assert twins["confidence"]["macro_auroc"] > twins["persistence"]["macro_auroc"]


def test_the_curve_checks_are_two_numbers_because_neither_says_the_other():
    """`adjacent_consistency` averages 0.9 while `max_blur_above_clean` averages 0.5 here.

    Two images climb cleanly to severity 5; two climb just as cleanly for four steps and then
    fall below clean at the last. Orderly steps and a separated endpoint are different claims,
    and a fixture where both came out the same number would let the two fields be swapped.
    """
    rising = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    collapsing = [0.0, 1.0, 2.0, 3.0, 4.0, -1.0]
    rows = candidate_rows({1: rising, 2: rising, 3: collapsing, 4: collapsing})
    candidate = summarize_contrast_candidates(rows, expected_image_count=4)[0]
    assert candidate["orientation"] == 1
    # four of five steps rise on the collapsing images, five of five on the rising ones
    assert candidate["oriented_adjacent_consistency"] == pytest.approx(0.9)
    # but only half the images end above where they started
    assert candidate["max_blur_above_clean_rate"] == pytest.approx(0.5)


def test_a_short_roster_withholds_the_auroc_and_says_so():
    """Four images that were all measured are still not the six the run expected.

    `expected_image_count` is the argument that knows this; the rows do not. The direction and
    the per-image curve checks survive -- they are claims about single images -- but the
    AUROCs do not, because they rank scores across images and a short roster would quietly
    change what the comparison was over.
    """
    rising = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    rows = candidate_rows({image: rising for image in range(1, 5)})
    candidate = summarize_contrast_candidates(rows, expected_image_count=6)[0]
    assert candidate["expected_image_count"] == 6
    assert candidate["image_count"] == 4
    assert candidate["measured_count"] == 4
    assert candidate["missing_count"] == 0
    assert candidate["complete"] is False
    assert candidate["orientation"] == 1
    assert candidate["orientable"] is True
    assert candidate["oriented_adjacent_consistency"] == pytest.approx(1.0)
    assert candidate["auroc_by_severity"] is None
    assert candidate["macro_auroc"] is None

    # and the other way round: five images produced rows where four were expected, and the
    # four that were fully measured are not the whole roster either. `complete` needs all
    # three numbers to agree, because `measured == expected_image_count` alone would send a
    # roster with an extra half-scored image into the AUROC.
    longer = candidate_rows({image: rising for image in range(1, 6)})
    longer = [row for row in longer if not (row["image_id"] == 5 and row["severity"] == 3)]
    overrun = summarize_contrast_candidates(longer, expected_image_count=4)[0]
    assert overrun["image_count"] == 5
    assert overrun["measured_count"] == 4
    assert overrun["expected_image_count"] == 4
    assert overrun["complete"] is False
    assert overrun["macro_auroc"] is None


def test_an_unmeasured_image_stays_in_every_denominator():
    """Two rising, one flat, one short curve: the short one is a quarter of this candidate.

    Dividing the direction counts by `measured_count` instead would turn two agreeing images
    out of four into two out of three, so a candidate that failed to produce a curve on some
    of its images would report the survivors' agreement as if the failures had never been
    asked. The unmeasured image is also kept out of the Spearman medians entirely rather than
    entered as a `0.0`, which would drag the median towards "no direction" in proportion to
    how often the candidate failed.
    """
    rising = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    rows = candidate_rows({1: rising, 2: rising, 3: [2.0] * 6, 4: rising})
    rows = [row for row in rows if not (row["image_id"] == 4 and row["severity"] == 3)]
    candidate = summarize_contrast_candidates(rows, expected_image_count=4)[0]
    assert candidate["image_count"] == 4
    assert candidate["measured_count"] == 3
    assert candidate["missing_count"] == 1
    assert candidate["measured_fraction"] == pytest.approx(0.75)
    assert candidate["missing_fraction"] == pytest.approx(0.25)
    assert candidate["positive_count"] == 2
    assert candidate["flat_count"] == 1
    assert candidate["negative_count"] == 0
    assert candidate["positive_fraction"] == pytest.approx(0.5)
    assert candidate["flat_fraction"] == pytest.approx(0.25)
    assert candidate["dominant_direction_fraction"] == pytest.approx(0.5)
    # three finite trends, +1, +1 and 0.0; a fourth read as 0.0 would median to 0.5
    assert candidate["median_signed_spearman"] == pytest.approx(1.0)
    assert candidate["median_absolute_spearman"] == pytest.approx(1.0)
    # the flat image is oriented too, and it is the one that never clears its own clean score
    assert candidate["oriented_adjacent_consistency"] == pytest.approx(1.0)
    assert candidate["max_blur_above_clean_rate"] == pytest.approx(2 / 3)
    assert candidate["complete"] is False
    assert candidate["macro_auroc"] is None
    # the severity the short image never reached still reports the three images that did
    assert candidate["severity_statistics"][3]["count"] == 3
    assert candidate["severity_statistics"][0]["count"] == 4


def test_a_severity_no_image_reached_is_published_as_an_empty_row():
    """Task 7 draws a box per severity, so an absent severity is a zero count, not a gap."""
    rising = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    rows = candidate_rows({1: rising, 2: rising})
    rows = [row for row in rows if row["severity"] != 3]
    candidate = summarize_contrast_candidates(rows, expected_image_count=2)[0]
    assert candidate["severity_statistics"][3] == {
        "count": 0, "mean": None, "variance": None,
        "median": None, "q25": None, "q75": None,
    }
    assert candidate["severity_statistics"][0]["count"] == 2
    assert candidate["measured_count"] == 0
    assert candidate["orientation"] is None


def test_a_row_without_its_key_fields_is_refused():
    """The position is in the message because a bare field name locates nothing in 29,160 rows."""
    rows = candidate_rows({1: [0.0] * 6})
    del rows[2]["severity"]
    with pytest.raises(ContrastAnalysisError, match=r"contrast row 2 is missing \['severity'\]"):
        summarize_contrast_candidates(rows, expected_image_count=1)


def test_candidates_are_published_in_a_stable_order(tmp_path):
    """Task 8 writes these to a CSV, so the order has to be the key's and not the rows'."""
    rows, _ = build_contrast_rows(loaded(tmp_path))
    candidates = summarize_contrast_candidates(rows, expected_image_count=IMAGES)
    keys = [tuple(item[field] for field in CONTRAST_CANDIDATE_KEY) for item in candidates]
    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)
    assert keys[0] == ("decile_00_10__50_60", "confidence", "mean", "clean_residual")
    assert keys[-1] == ("quintile_00_20__40_60", "persistence", "top20_mean", "relative_gap")
    shuffled = summarize_contrast_candidates(
        list(reversed(rows)), expected_image_count=IMAGES
    )
    assert [
        tuple(item[field] for field in CONTRAST_CANDIDATE_KEY) for item in shuffled
    ] == keys


def test_the_severity_mean_and_median_are_two_numbers_because_one_scene_can_move_one():
    """Three scenes near 2 and one at 10: the mean goes to 4.0 and the median stays at 2.5.

    Every other fixture in this file has a symmetric score column, where mean and median are
    the same number and either could stand in for the other. They must not: the mean is what
    the cross-scene AUROC is sensitive to and the median is what the per-image trend behaves
    like, and a candidate whose median falls while its mean rises is exactly the shape that
    puts a locked orientation below 0.5.
    """
    rows = candidate_rows({1: [1.0] * 6, 2: [2.0] * 6, 3: [3.0] * 6, 4: [10.0] * 6})
    candidate = summarize_contrast_candidates(rows, expected_image_count=4)[0]
    clean = candidate["severity_statistics"][0]
    assert clean["count"] == 4
    assert clean["mean"] == pytest.approx(4.0)
    assert clean["median"] == pytest.approx(2.5)
    assert clean["q25"] == pytest.approx(1.75)
    assert clean["q75"] == pytest.approx(4.75)
    assert clean["variance"] == pytest.approx(12.5)


def test_rows_that_disagree_about_a_candidates_provenance_are_refused():
    """Copying the first row's provenance is only safe if every row carries the same.

    Refused rather than resolved, because there is no honest way to pick: two rows that
    disagree about `score_scope` are two experiments filed under one candidate name, and
    publishing either scope beside a merged set of scores would label half the numbers wrong.
    Nothing else in the pipeline re-checks it -- `CONTRAST_ROW_KEY` deliberately leaves
    `score_scope` out, so the row grid cannot tell the two apart on its own.
    """
    rows = candidate_rows({1: [0.0] * 6, 2: [1.0] * 6})
    rows[-1]["score_scope"] = "combined"
    with pytest.raises(ContrastAnalysisError, match="disagree about the provenance"):
        summarize_contrast_candidates(rows, expected_image_count=2)

    swapped = candidate_rows({1: [0.0] * 6, 2: [1.0] * 6})
    swapped[-1]["reference_bin"], swapped[-1]["responsive_bin"] = (
        swapped[-1]["responsive_bin"], swapped[-1]["reference_bin"],
    )
    with pytest.raises(ContrastAnalysisError, match="disagree about the provenance"):
        summarize_contrast_candidates(swapped, expected_image_count=2)
