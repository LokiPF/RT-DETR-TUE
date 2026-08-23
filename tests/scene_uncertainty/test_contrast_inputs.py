"""What the loader must refuse, what a real bundle may carry, and what the fixture must be.

Half of these tests exist because the *permissive* half of the loader is as load-bearing as the
strict half. A real `per_scene.csv` holds a `frozen` twin of every bin, an `unfiltered` control
at the lowest bin of each scheme, ten deciles, five quintiles and four scopes -- 765,000 rows,
of which this command reads 1,296 per 6-image fixture and 54,000 in production. A refusal aimed
at any of the rows it does not read is a refusal of every bundle the producer can write, so
`test_discards_the_frozen_and_unfiltered_twins_a_real_bundle_carries` is not a nicety: it is the
test that says this command can run at all.

The rest exist because the fixture is a shared artefact, not a private one. Nine tasks read
`contrast_test_utils` and only this module reads `contrast_inputs.py`, so from Task 3 onwards
every expectation anyone writes is an expectation about the fixture's numbers -- and the loader
is blind to almost all of them. It never reads `selected_count`, `clean_overlap` or any of the
four trend columns; it cannot tell a bin whose curve is flat from one that moves; it accepts a
`combined` column that is `layer_2` plus a constant. So a fixture defect does not fail here, it
propagates. Every test below whose name starts `test_the_fixture_` or which names a producer
behaviour is guarding a property no loader assertion can reach.
"""
import csv
import json
from dataclasses import astuple
from statistics import mean, median

import pytest
from scipy.stats import spearmanr

from src.scene_uncertainty.contrast_inputs import (
    ARMS,
    CONFIDENCE_SCOPE,
    REQUIRED_SERIES,
    ContrastInputError,
    load_contrast_inputs,
)
from src.scene_uncertainty.corruption_metrics import (
    choose_orientation,
    complete_trend_metrics,
    severity_aurocs,
)

from tests.scene_uncertainty.contrast_test_utils import (
    AGGREGATIONS,
    BIN_TREND_PHASE,
    COMBINED_LEVEL,
    CONFIDENCE_TREND_PHASE,
    CONFIDENCE_UNCERTAINTY,
    PERSISTENCE_LEVEL,
    SELECTED_COUNT,
    SERIES,
    TREND_SHAPE,
    clean_overlap_for,
    write_source_bundle,
)

IMAGES = range(1, 7)
SEVERITIES = range(6)
RETAINED = 12 * 3 * 6 * 6  # 12 series x 3 aggregations x 6 images x 6 severities


def source_row(**overrides):
    """One raw `per_scene.csv` row for `extra_rows`, defaulting to a row this command reads.

    The defaults describe a *severity-zero dynamic* row, which is why `clean_overlap` is `1.0`
    and not a plausible-looking fraction: that row's membership is compared against itself, so
    the producer cannot write anything else there. `selected_count` and the four trend columns
    come from the same place as the fixture's own -- `SELECTED_COUNT` by scheme, and a flat
    curve's `0.0 / 0.0 / "flat"`, which is what `complete_trend_metrics` returns for the
    constant curves the callers below build out of these rows. A caller overriding
    `bucket_scheme` or `severity` must override the matching column too, which the twins test
    does through the same two helpers the fixture uses.
    """
    row = {
        "image_id": 1, "severity": 0, "signal": "persistence", "bucket_scheme": "decile",
        "confidence_bin": "decile_50_60", "membership_mode": "dynamic",
        "padding_mode": "filtered", "aggregation": "mean", "score_scope": "layer_2",
        "source_partition": "tuning", "score": 1.0,
        "selected_count": SELECTED_COUNT["decile"], "clean_overlap": 1.0,
        "fully_measured": True, "signed_spearman": 0.0,
        "absolute_spearman": 0.0, "direction": "flat",
    }
    row.update(overrides)
    return row


def read_rows(source):
    """`per_scene.csv` as raw dictionaries, for the columns `load_contrast_inputs` drops."""
    with (source / "per_scene.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def curve(inputs, image_id, signal, confidence_bin, scope, aggregation="mean"):
    """One image's six-severity curve out of a loaded bundle, in severity order."""
    return [
        inputs.scores[(image_id, severity, signal, confidence_bin, aggregation, scope)]
        for severity in SEVERITIES
    ]


def signed_trends(inputs, signal, confidence_bin, scope, aggregation="mean"):
    """Every image's `signed_spearman` for one candidate, the input `choose_orientation` takes."""
    return [
        complete_trend_metrics(
            SEVERITIES, curve(inputs, image_id, signal, confidence_bin, scope, aggregation)
        )["signed_spearman"]
        for image_id in inputs.image_ids
    ]


def test_arm_table_is_the_four_declared_arms():
    # all eight fields, in order: a swap of reference_bin and responsive_bin is invisible to
    # `required_series`, which reads them symmetrically, and flips the sign of every contrast
    # `contrast_scores.raw_gap(reference, responsive)` produces.
    assert tuple(astuple(arm) for arm in ARMS) == (
        ("decile_00_10__50_60", "decile_00_10__50_60", "anchored", "decile", "layer_2",
         "decile_00_10", "decile_50_60", True),
        ("quintile_00_20__40_60", "quintile_00_20__40_60", "anchored", "quintile", "layer_2",
         "quintile_00_20", "quintile_40_60", True),
        ("decile_90_100__50_60", "decile_90_100__50_60", "differential", "decile", "layer_2",
         "decile_90_100", "decile_50_60", False),
        ("decile_90_100__50_60__combined", "decile_90_100__50_60", "differential", "decile",
         "combined", "decile_90_100", "decile_50_60", False),
    )
    # the two differential arms share one bucket pair, and therefore one confidence twin
    assert len({arm.pair_name for arm in ARMS}) == 3
    assert [arm.declared_before_data for arm in ARMS] == [True, True, False, False]


def test_required_series_is_every_series_the_arms_need_and_no_other():
    assert REQUIRED_SERIES == (
        ("persistence", "decile_00_10", "layer_2"),
        ("confidence", "decile_00_10", "confidence"),
        ("persistence", "decile_50_60", "layer_2"),
        ("confidence", "decile_50_60", "confidence"),
        ("persistence", "quintile_00_20", "layer_2"),
        ("confidence", "quintile_00_20", "confidence"),
        ("persistence", "quintile_40_60", "layer_2"),
        ("confidence", "quintile_40_60", "confidence"),
        ("persistence", "decile_90_100", "layer_2"),
        ("confidence", "decile_90_100", "confidence"),
        ("persistence", "decile_90_100", "combined"),
        ("persistence", "decile_50_60", "combined"),
    )
    # confidence has no decoder layer, so its entries collapse across scope: five bins, five
    # series, and the two differential arms share every one of them.
    confidence = [entry for entry in REQUIRED_SERIES if entry[0] == "confidence"]
    assert len(confidence) == 5
    assert {entry[2] for entry in confidence} == {CONFIDENCE_SCOPE}
    # the fixture writes exactly this set: a bundle it produces is neither short nor padded
    assert set(REQUIRED_SERIES) == {
        (signal, confidence_bin, scope) for signal, _, confidence_bin, scope in SERIES
    }


def test_loads_every_required_series(tmp_path):
    source = write_source_bundle(tmp_path / "source")
    inputs = load_contrast_inputs(source, expected_image_count=6)
    assert inputs.image_ids == (1, 2, 3, 4, 5, 6)
    assert len(inputs.scores) == RETAINED
    # a mean nearest-neighbour distance, near the run's own 0.083 and nowhere near 1.0
    assert inputs.scores[(1, 0, "persistence", "decile_50_60", "mean", "layer_2")] == 0.067517
    # the same bin at the combined scope is a different measurement, not a copy -- a signed
    # z-score against the clean median, and this bin sits below it
    assert inputs.scores[(1, 0, "persistence", "decile_50_60", "mean", "combined")] == -0.29306
    # and the confidence twin varies by bin, so a contrast against it is not identically zero
    assert inputs.scores[(1, 0, "confidence", "decile_00_10", "mean", "confidence")] == 0.983049
    assert inputs.scores[(1, 0, "confidence", "decile_90_100", "mean", "confidence")] == 0.666684
    assert inputs.provenance["source_partition"] == "tuning"
    assert inputs.provenance["retained_row_count"] == RETAINED


def test_the_confidence_column_runs_the_way_the_producer_writes_it(tmp_path):
    """`1 - confidence`: the bins descend, and only the top decile's mean climbs with severity.

    Both directions are pinned because neither is visible to the loader. An inverted map keeps
    every count, key and coverage check intact and flips the sign of all 36 confidence-twin
    contrasts four tasks downstream; a fixture where every bin fell with severity would let a
    consumer that had the top decile upside down pass anyway. The rise at `decile_90_100` --
    0.645 to 0.799 in the completed run -- is the detector losing the queries it was surest
    about, which is the behaviour the redundancy control exists to expose.

    Asserted on the **across-image mean** at each severity, not on image 1's curve. That is the
    quantity these five slopes actually are, and reading one image's curve as if it were the
    mean is what forced the fixture to make every image follow its bin -- which is false of the
    run for two of the five bins, and which
    `test_the_confidence_twins_carry_the_runs_per_image_trend` now pins the other way. The two
    tests are the mean and the median of the same column and both have to hold.
    """
    inputs = load_contrast_inputs(
        write_source_bundle(tmp_path / "source"), expected_image_count=6
    )

    def column(confidence_bin, severity=0):
        return mean(
            inputs.scores[(image_id, severity, "confidence", confidence_bin, "mean", "confidence")]
            for image_id in inputs.image_ids
        )

    descending = [column(name) for name in (
        "decile_00_10", "quintile_00_20", "quintile_40_60", "decile_50_60", "decile_90_100"
    )]
    # pair by pair and by a margin, not merely `>`. Averaging the wobble away leaves a
    # rounding residue of order 1e-7, so two bins given the *same* level would still come out
    # ordered, one way or the other, by nothing -- and collapsing `quintile_00_20` onto
    # `decile_00_10` is exactly how the two schemes stop being distinguishable. The run's
    # smallest adjacent gap is 0.006, six times this margin and far larger than any wobble.
    assert all(higher - lower > 0.001 for higher, lower in zip(descending, descending[1:]))

    top = [column("decile_90_100", severity) for severity in SEVERITIES]
    assert top == sorted(top)
    assert top[-1] - top[0] > 0.15
    # the lower four are flat, but not flat in one direction: two fall and two rise, and an
    # edit that gave all four a common downward slope would otherwise pass
    for name in ("decile_00_10", "quintile_00_20"):
        assert column(name, 5) < column(name, 0)
        assert column(name, 0) - column(name, 5) < 0.01
    for name in ("quintile_40_60", "decile_50_60"):
        assert column(name, 5) > column(name, 0)
        assert column(name, 5) - column(name, 0) < 0.01


def test_the_confidence_twins_carry_the_runs_per_image_trend(tmp_path):
    """The mean level and the per-image trend disagree in sign, and both are load-bearing.

    `quintile_40_60` and `decile_50_60` both *rise* in mean level in the completed run while 154
    and 151 of the 250 images individually *fall* -- a minority of large risers carrying the
    average. `choose_orientation` reads the median of the per-image signed Spearmans, so the
    real twins for those two bins are locked at `-1`, and a fixture where every image followed
    its bin's mean locks them at `+1`. That is round 2's defect at a different depth: it moves
    no count, no key and no coverage check, and surfaces as two of five confidence twins read
    backwards.

    The unanimity is separately fatal. With every image at exactly `+1` or `-1`, a candidate's
    `median_absolute_spearman` and `dominant_direction_fraction` are both exactly 1.0 -- Task
    6's ranking criteria 4 and 5 -- and a criterion that takes one value on every candidate
    cannot order anything. The run spreads them over 0.771-0.857 and 0.60-0.80.

    Six images cannot hold a 250-image proportion: a six-point Spearman is quantised and `k/6`
    is the only fraction available. What is pinned here is what survives that -- the side of
    zero each bin's median falls on, and that no bin is unanimous.
    """
    inputs = load_contrast_inputs(
        write_source_bundle(tmp_path / "source"), expected_image_count=6
    )
    # the run's median signed Spearman, dynamic/filtered/mean over all 250 tuning images
    run = {
        "decile_00_10": -0.600, "quintile_00_20": -0.600, "quintile_40_60": -0.543,
        "decile_50_60": -0.514, "decile_90_100": +0.829,
    }
    for confidence_bin, expected in run.items():
        signed = signed_trends(inputs, "confidence", confidence_bin, "confidence")
        assert len(signed) == len(inputs.image_ids)
        # the locked direction, which is the whole of what nine tasks read this column through
        assert choose_orientation(signed) == (1 if expected > 0 else -1), confidence_bin
        # ... and it is not unanimous: some images move against their own bin's median
        assert 0 < sum(1 for value in signed if value > 0) < len(signed), confidence_bin
        # criteria 4 and 5 have somewhere to move. A single image may still be perfectly
        # monotone -- the run's median absolute Spearman is 0.771 to 0.857, so roughly half of
        # them sit above it -- but the median may not be 1.0 and the strengths may not be one
        # repeated value, because a criterion with one value on every candidate orders nothing.
        strengths = [abs(value) for value in signed]
        assert median(strengths) < 1.0, confidence_bin
        assert len(set(strengths)) > 1, confidence_bin

    # the two bins whose mean rises while their median falls -- the reason this test exists
    for confidence_bin in ("quintile_40_60", "decile_50_60"):
        rising_mean = mean(
            inputs.scores[(i, 5, "confidence", confidence_bin, "mean", "confidence")]
            for i in inputs.image_ids
        ) > mean(
            inputs.scores[(i, 0, "confidence", confidence_bin, "mean", "confidence")]
            for i in inputs.image_ids
        )
        assert rising_mean, confidence_bin
        assert choose_orientation(
            signed_trends(inputs, "confidence", confidence_bin, "confidence")
        ) == -1, confidence_bin


def test_the_confidence_control_does_not_wobble_in_step_with_its_own_signal(tmp_path):
    """A bin's confidence twin is the control for that bin's persistence signal, so the two
    must not share a wobble. Measured on the tables and on the bundle, because only the first
    localises the defect and only the second proves it reaches the rows.

    `TREND_SHAPE` is rotated by `(severity + image_id + phase) % 6`. When a bin's
    `CONFIDENCE_TREND_PHASE` equals its `BIN_TREND_PHASE` the two signals read the *same* entry
    at every severity of every image, so their wobble components are one sequence and their
    rank correlation is exactly `+1.0` on every image -- the fixture answering, by arithmetic,
    the independence question Task 6's redundancy control is there to measure. Two bins carried
    that: `quintile_00_20`, arm 2's reference, and `decile_50_60`, the responsive bin of arms 1,
    3 and 4. Between them they are the fixture's most-read pair of series, and nothing else in
    this suite could see it -- the two signals live in different rows, so no count, key or
    coverage check compares them.

    `spearmanr` on the wobble alone is the direct measurement; the loaded-bundle assertion is
    the consequence, and it is deliberately weaker than `!= +1.0` on a median. A shared wobble
    does not make the two *curves* identical, because the level, the slope and the tilt tables
    all differ, so the median cross-signal correlation only rose to +0.800 rather than to 1.0.
    What it did do is put at least one image at exactly `+1.0`, and that is what is pinned.
    """
    # every bin's confidence phase differs from its own persistence phase: a derangement
    assert all(
        CONFIDENCE_TREND_PHASE[name] != BIN_TREND_PHASE[name] for name in BIN_TREND_PHASE
    ), {name: (BIN_TREND_PHASE[name], CONFIDENCE_TREND_PHASE[name]) for name in BIN_TREND_PHASE}
    # and neither table repeats a phase, which is the separate property that stops two bins of
    # one arm sharing a wobble that would subtract out of their gap
    assert len(set(BIN_TREND_PHASE.values())) == len(BIN_TREND_PHASE)
    assert len(set(CONFIDENCE_TREND_PHASE.values())) == len(CONFIDENCE_TREND_PHASE)

    for name in BIN_TREND_PHASE:
        for image_id in IMAGES:
            persistence = [
                TREND_SHAPE[(severity + image_id + BIN_TREND_PHASE[name]) % 6]
                for severity in SEVERITIES
            ]
            confidence = [
                TREND_SHAPE[(severity + image_id + CONFIDENCE_TREND_PHASE[name]) % 6]
                for severity in SEVERITIES
            ]
            assert persistence != confidence, (name, image_id)
            assert spearmanr(persistence, confidence).statistic < 1.0, (name, image_id)

    inputs = load_contrast_inputs(
        write_source_bundle(tmp_path / "source"), expected_image_count=6
    )
    for name in BIN_TREND_PHASE:
        for image_id in inputs.image_ids:
            correlation = spearmanr(
                curve(inputs, image_id, "persistence", name, "layer_2"),
                curve(inputs, image_id, "confidence", name, CONFIDENCE_SCOPE),
            ).statistic
            assert correlation < 1.0, (name, image_id)


def test_a_full_tuning_roster_keeps_the_columns_where_the_run_measured_them(tmp_path):
    """250 images, the size the real command runs at, and the only test that reaches it.

    Two properties, and both are invisible at six images because both are properties of the
    *per-image term* and the per-image term is the one thing that scales with the roster.

    The bound. `1 - confidence` cannot leave [0, 1], and the loader would not notice if it did
    -- it checks negativity and finiteness, and 1.235 is both non-negative and finite. A
    `- 0.001 * image_id` term is 0.005 of spread across six images and 0.25 across 250, so the
    version of this fixture that carried one passed every six-image test in this file and wrote
    impossible rows at the roster the command actually runs.

    The level. Every published level table is an across-image mean the completed run measured,
    so at the full roster the fixture's own across-image mean has to *be* that number. That is
    what says the per-image term is centred rather than merely small: the `- 0.001 * image_id`
    term only ever subtracted, so it moved all five confidence bins a uniform 0.1255 below the
    run -- 0.985 published, 0.860 written -- while leaving the six-image column a plausible
    0.982. Nothing but a full-roster mean can see that, and the tolerance below is 1e-3, a
    hundred times smaller than the offset it exists to catch and ten times larger than the
    residue left by 250 not being a multiple of the six-entry wobble table.

    This is also the only test that exercises the production `expected_image_count` default
    rather than passing 6, and the only one that overrides `image_ids`.
    """
    roster = range(1, 251)
    source = write_source_bundle(tmp_path / "source", image_ids=roster)
    inputs = load_contrast_inputs(source)
    confidence = [
        value for key, value in inputs.scores.items() if key[2] == "confidence"
    ]
    assert len(confidence) == 5 * 3 * 250 * 6
    assert 0.0 < min(confidence)
    assert max(confidence) < 1.0
    # and the spread survives the shrink: every image still has its own value
    lowest_bin = [
        inputs.scores[(image_id, 0, "confidence", "decile_00_10", "mean", "confidence")]
        for image_id in roster
    ]
    assert len(set(lowest_bin)) == 250

    for signal, scope, levels in (
        ("confidence", "confidence", CONFIDENCE_UNCERTAINTY),
        ("persistence", "layer_2", PERSISTENCE_LEVEL),
        ("persistence", "combined", COMBINED_LEVEL),
    ):
        for confidence_bin, published in levels.items():
            if (signal, confidence_bin, scope) not in set(REQUIRED_SERIES):
                continue  # only two bins are written at the combined scope
            measured = mean(
                inputs.scores[(image_id, 0, signal, confidence_bin, "mean", scope)]
                for image_id in roster
            )
            assert measured == pytest.approx(published, abs=1e-3), (signal, confidence_bin)


def test_every_arms_persistence_reference_can_be_oriented(tmp_path):
    """A flat reference curve is an arm that cannot be compared with anything.

    `complete_trend_metrics` short-circuits a constant curve to `signed_spearman = 0.0` and
    `direction = "flat"`, and `choose_orientation` returns `None` on a median of exactly zero
    -- not `+1`, deliberately, because breaking the tie would hand an unorientable candidate a
    full set of curve checks and AUROCs indistinguishable from an oriented one's. Downstream
    that `None` is a `macro_auroc` of `None`, a `beats_both_inputs` that is structurally
    `False`, and a `TypeError` the first time anything subtracts one AUROC from another.

    It bites hardest on the two *anchored* arms, whose references are the lowest bin of each
    scheme. Those are the arms declared before any tuning number was read, so they are the two
    the spec's "beats both its inputs" criterion is worth demonstrating on, and a fixture that
    left them flat made it undemonstrable. Both signs are pinned, not just non-`None`: the run
    has `decile_00_10` falling at a median signed Spearman of -0.200 and `quintile_00_20`
    rising at +0.086, and a fixture that oriented them the other way would give every anchored
    contrast in nine tasks the wrong sign.
    """
    inputs = load_contrast_inputs(
        write_source_bundle(tmp_path / "source"), expected_image_count=6
    )
    expected = {"decile_00_10__50_60": -1, "quintile_00_20__40_60": 1}
    for arm in ARMS:
        for confidence_bin in (arm.reference_bin, arm.responsive_bin):
            signed = signed_trends(inputs, "persistence", confidence_bin, arm.score_scope)
            assert choose_orientation(signed) is not None, (arm.name, confidence_bin)
        reference = signed_trends(inputs, "persistence", arm.reference_bin, arm.score_scope)
        if arm.name in expected:
            assert choose_orientation(reference) == expected[arm.name], arm.name
    # and no curve anywhere in the bundle is the constant one that produced the `None`
    assert "flat" not in {row["direction"] for row in read_rows(tmp_path / "source")}


def test_the_two_bucket_schemes_are_not_one_series(tmp_path):
    """Separating the two resolutions *is* the comparison this command exists to make.

    That sentence is `corruption_reporting.ROW_KEY`'s own, and it is why `bucket_scheme` is in
    the row key at all. A fixture that gives `decile_00_10` and `quintile_00_20` one level and
    one slope, and `decile_50_60` and `quintile_40_60` another, makes arms 1 and 2 produce the
    same number at every image, severity, aggregation and method -- identical contrasts,
    identical orientations, identical macro AUROCs -- so a mutation that computed the quintile
    arm out of the decile bins would pass every test in nine tasks.

    Equality is checked exactly rather than approximately: two bins that agree to six decimal
    places have already collapsed for every downstream purpose.
    """
    inputs = load_contrast_inputs(
        write_source_bundle(tmp_path / "source"), expected_image_count=6
    )
    pairs = (("decile_00_10", "quintile_00_20"), ("decile_50_60", "quintile_40_60"))
    for decile_bin, quintile_bin in pairs:
        for image_id in inputs.image_ids:
            for severity in SEVERITIES:
                for aggregation in AGGREGATIONS:
                    key = (image_id, severity, "persistence")
                    assert (
                        inputs.scores[(*key, decile_bin, aggregation, "layer_2")]
                        != inputs.scores[(*key, quintile_bin, aggregation, "layer_2")]
                    ), (decile_bin, quintile_bin, image_id, severity, aggregation)

    def gap(reference_bin, responsive_bin, image_id, severity):
        key = (image_id, severity, "persistence")
        return (
            inputs.scores[(*key, responsive_bin, "mean", "layer_2")]
            - inputs.scores[(*key, reference_bin, "mean", "layer_2")]
        )

    # the arms themselves, not just their inputs: a shared slope survives a level split
    for image_id in inputs.image_ids:
        for severity in SEVERITIES:
            assert (
                gap("decile_00_10", "decile_50_60", image_id, severity)
                != gap("quintile_00_20", "quintile_40_60", image_id, severity)
            ), (image_id, severity)


def test_the_combined_scope_is_a_signed_z_score_not_an_offset_layer(tmp_path):
    """`combined` is `(score - clean median) / scale` averaged over layers, so it is signed.

    A fixture that writes `layer_2 + 0.5` gets the levels plausibly wrong in the one way that
    hides itself: the offset cancels in `raw_gap` and in `clean_residual`, so arms 3 and 4 --
    the same bucket pair read at the two scopes -- come out identical for two of their three
    methods, and no default bundle ever emits a negative value. The loader's `COMBINED_SCOPE`
    exemption and Task 2's exclusion of the symmetric relative gap at this scope are the two
    places the design pays for `combined` being signed, and both would then be exercised only
    by bespoke overrides.
    """
    inputs = load_contrast_inputs(
        write_source_bundle(tmp_path / "source"), expected_image_count=6
    )
    combined = [value for key, value in inputs.scores.items() if key[5] == "combined"]
    negative = [value for value in combined if value < 0.0]
    assert negative, "the default bundle must exercise the signed scope"
    assert len(negative) > len(combined) / 4
    # ... while `layer_2` stays a distance, which the loader refuses to see go negative
    assert min(value for key, value in inputs.scores.items() if key[5] == "layer_2") > 0.0

    def score(scope, image_id, severity):
        return inputs.scores[
            (image_id, severity, "persistence", "decile_50_60", "mean", scope)
        ]

    offsets = {
        round(score("combined", image_id, severity) - score("layer_2", image_id, severity), 6)
        for image_id in inputs.image_ids
        for severity in SEVERITIES
    }
    assert len(offsets) == len(inputs.image_ids) * len(SEVERITIES)

    def gap(scope, image_id, severity):
        key = (image_id, severity, "persistence")
        return (
            inputs.scores[(*key, "decile_50_60", "mean", scope)]
            - inputs.scores[(*key, "decile_90_100", "mean", scope)]
        )

    # arms 3 and 4 are one bucket pair at two scopes; an offset makes their raw gaps identical
    for image_id in inputs.image_ids:
        for severity in SEVERITIES:
            assert gap("layer_2", image_id, severity) != gap("combined", image_id, severity)


def test_every_row_carries_the_trend_of_its_own_curve(tmp_path):
    """The four trend columns are a derivation, and the fixture must not disagree with itself.

    `corruption_reporting.summarize_candidates` computes them once per candidate and image with
    `complete_trend_metrics` and repeats them on all six of that image's rows, so `direction` is
    the sign of `signed_spearman` by construction, `absolute_spearman` is its magnitude, and a
    curve that is not the full ladder comes back `None` rather than a number. Literal values
    cannot hold that: `0.6 / 0.6 / "increasing"` on every row was wrong on every row, most
    visibly on the confidence bins the fixture had otherwise been made to *fall*.

    `load_contrast_inputs` never reads these columns, which is exactly why they need a test
    here -- nothing else in this task can see them, and Tasks 3 through 9 read them as fact.
    """
    source = write_source_bundle(tmp_path / "source")
    rows = read_rows(source)
    curves = {}
    for row in rows:
        key = (row["signal"], row["confidence_bin"], row["score_scope"],
               row["aggregation"], row["image_id"])
        curves.setdefault(key, {})[int(row["severity"])] = float(row["score"])
    for row in rows:
        key = (row["signal"], row["confidence_bin"], row["score_scope"],
               row["aggregation"], row["image_id"])
        by_severity = curves[key]
        severities = sorted(by_severity)
        trend = complete_trend_metrics(severities, [by_severity[s] for s in severities])
        assert row["fully_measured"] == str(trend["fully_measured"])
        assert float(row["signed_spearman"]) == pytest.approx(trend["signed_spearman"])
        assert float(row["absolute_spearman"]) == pytest.approx(trend["absolute_spearman"])
        assert row["direction"] == trend["direction"]
    # not one number repeated: the whole point is that candidates and images disagree
    assert len({row["signed_spearman"] for row in rows}) > 10
    # and the confidence bins rounds 2 and 3 were convened to make fall are recorded as falling
    assert {
        row["direction"] for row in rows
        if row["signal"] == "confidence" and row["confidence_bin"] == "decile_00_10"
    } >= {"decreasing"}


def test_clean_overlap_is_one_where_the_producer_can_only_write_one(tmp_path):
    """Two whole classes of row where `0.2` is not merely unlikely but impossible.

    `corruption_analysis` sets `clean_filtered = memberships[0]["dynamic"]` and measures every
    filtered row's Jaccard against it -- so a dynamic severity-zero row compares that partition
    with itself and reads exactly `1.0`. Frozen membership *is* severity zero's bins at every
    severity, so every frozen row reads `1.0` too. The completed run's table confirms both.

    A fixture writing `0.2` there tells nine downstream tasks that severity zero moved, which is
    the diagnostic this column exists to deny, and no assertion in this task's loader can see it.
    """
    source = write_source_bundle(tmp_path / "source")
    rows = read_rows(source)
    assert {row["clean_overlap"] for row in rows if row["severity"] == "0"} == {"1.0"}
    corrupted = [float(row["clean_overlap"]) for row in rows if row["severity"] != "0"]
    assert max(corrupted) < 1.0
    for confidence_bin in ("decile_00_10", "decile_90_100", "quintile_40_60"):
        series = [
            float(row["clean_overlap"]) for severity in SEVERITIES for row in rows
            if row["confidence_bin"] == confidence_bin and row["severity"] == str(severity)
            and row["image_id"] == "1" and row["aggregation"] == "mean"
            and row["signal"] == "confidence"
        ]
        assert all(later < earlier for earlier, later in zip(series, series[1:])), confidence_bin
    # frozen membership is severity zero's by construction, at every severity
    frozen = write_source_bundle(tmp_path / "frozen", membership_mode="frozen")
    assert {row["clean_overlap"] for row in read_rows(frozen)} == {"1.0"}


def test_a_quintile_bin_holds_twice_a_decile_bins_queries(tmp_path):
    """`selected_count` is the only column recording how much evidence a bin was pooled over.

    A quintile is two deciles wide. Writing one count for both makes the two resolutions
    indistinguishable in the one place the difference between them is stated, and "pooled over
    twice the queries" is the entire reason the coarser scheme is in the experiment.
    """
    rows = read_rows(write_source_bundle(tmp_path / "source"))
    counts = {}
    for row in rows:
        counts.setdefault(row["bucket_scheme"], set()).add(row["selected_count"])
    assert counts == {"decile": {"30"}, "quintile": {"60"}}


def test_images_disagree_about_the_trend_so_no_candidate_sits_at_the_ceiling(tmp_path):
    """A per-image *level* cancels in every gap method; only a per-image *slope* survives it.

    `raw_gap` and `clean_residual` both subtract two bins of the same image, so a fixture whose
    only per-image term is an offset produces one identical gap curve on all six images. Every
    candidate then scores a macro AUROC of exactly 1.0, Task 6's redundancy comparisons are
    decided by 1.0-versus-1.0 ties at the ceiling, and the paired bootstrap's interval collapses
    to a point -- none of which fails anything, and all of which is meaningless.

    The variation is deterministic: a six-entry tilt table read with `image_id % 6`. Nothing
    here may reach for `random`, a clock or anything else whose value moves between two runs,
    because a fixture that is not byte-identical run to run makes every failure below it
    unreproducible.
    """
    inputs = load_contrast_inputs(
        write_source_bundle(tmp_path / "source"), expected_image_count=6
    )

    def macro_auroc(scores_by_image_severity):
        signed = [
            complete_trend_metrics(
                SEVERITIES, [scores_by_image_severity[(i, s)] for s in SEVERITIES]
            )["signed_spearman"]
            for i in inputs.image_ids
        ]
        orientation = choose_orientation(signed)
        assert orientation is not None
        assert len(set(signed)) > 1, signed
        _, macro = severity_aurocs(
            {s: [scores_by_image_severity[(i, s)] for i in inputs.image_ids]
             for s in SEVERITIES},
            orientation,
        )
        return macro

    for arm in ARMS:
        gap = {
            (image_id, severity): (
                inputs.scores[(image_id, severity, "persistence", arm.responsive_bin,
                               "mean", arm.score_scope)]
                - inputs.scores[(image_id, severity, "persistence", arm.reference_bin,
                                 "mean", arm.score_scope)]
            )
            for image_id in inputs.image_ids for severity in SEVERITIES
        }
        assert macro_auroc(gap) < 1.0, arm.name

    # the raw inputs a candidate is compared against, and the confidence twin, land below it too
    for signal, scope in (("persistence", "layer_2"), ("confidence", "confidence")):
        for confidence_bin in ("decile_00_10", "quintile_00_20", "quintile_40_60",
                               "decile_50_60", "decile_90_100"):
            raw = {
                (image_id, severity):
                    inputs.scores[(image_id, severity, signal, confidence_bin, "mean", scope)]
                for image_id in inputs.image_ids for severity in SEVERITIES
            }
            assert macro_auroc(raw) < 1.0, (signal, confidence_bin)


def test_discards_the_frozen_and_unfiltered_twins_a_real_bundle_carries(tmp_path):
    """A real bundle holds both membership modes and both padding modes for these very bins.

    `analyze_corruption_sensitivity` loops `for mode in ("dynamic", "frozen")` over every bin at
    `padding_mode="filtered"` and emits an `unfiltered` row at each scheme's lowest bin -- which
    is `decile_00_10` and `quintile_00_20`, two of the bins this experiment reads. Every twin
    here is scored `99.0`, so a loader that refused them fails, one that filed them under a
    longer key changes the retained count, and one that let them overwrite changes the value.
    """
    twins = []
    for signal, scheme, confidence_bin, scope in SERIES:
        for aggregation in AGGREGATIONS:
            for image_id in IMAGES:
                for severity in SEVERITIES:
                    twins.append(source_row(
                        image_id=image_id, severity=severity, signal=signal,
                        bucket_scheme=scheme, confidence_bin=confidence_bin, score_scope=scope,
                        aggregation=aggregation, membership_mode="frozen", score=99.0,
                        selected_count=SELECTED_COUNT[scheme],
                        clean_overlap=clean_overlap_for(confidence_bin, severity, "frozen"),
                    ))
    for confidence_bin, scheme in (("decile_00_10", "decile"), ("quintile_00_20", "quintile")):
        for signal, scope in (("persistence", "layer_2"), ("confidence", "confidence")):
            for mode in ("dynamic", "frozen"):
                for aggregation in AGGREGATIONS:
                    for image_id in IMAGES:
                        for severity in SEVERITIES:
                            twins.append(source_row(
                                image_id=image_id, severity=severity, signal=signal,
                                bucket_scheme=scheme, confidence_bin=confidence_bin,
                                score_scope=scope, aggregation=aggregation,
                                membership_mode=mode, padding_mode="unfiltered", score=99.0,
                                selected_count=SELECTED_COUNT[scheme],
                                clean_overlap=clean_overlap_for(confidence_bin, severity, mode),
                            ))

    source = write_source_bundle(tmp_path / "source", extra_rows=twins)
    inputs = load_contrast_inputs(source, expected_image_count=6)
    assert len(inputs.scores) == RETAINED
    assert inputs.scores[(1, 0, "persistence", "decile_50_60", "mean", "layer_2")] == 0.067517
    assert 99.0 not in inputs.scores.values()


def test_refuses_a_held_out_summary(tmp_path):
    source = write_source_bundle(tmp_path / "source", partition="held_out")
    with pytest.raises(ContrastInputError, match="reads the tuning partition only"):
        load_contrast_inputs(source, expected_image_count=6)


def test_refuses_held_out_rows_inside_a_tuning_summary(tmp_path):
    """The mixed case, and the row sits in a series this command never reads.

    A held-out row anywhere means the table did not come from the tuning-only loader, so the
    partition is tested before the series filter rather than after it. Filtering this row out
    because `decile_20_30` is not an arm's bin is how a held-out claim gets made by accident.
    """
    source = write_source_bundle(
        tmp_path / "source",
        extra_rows=(source_row(confidence_bin="decile_20_30", source_partition="held_out"),),
    )
    with pytest.raises(ContrastInputError, match=r"row \d+ is outside the tuning partition"):
        load_contrast_inputs(source, expected_image_count=6)


def test_refuses_a_missing_required_series(tmp_path):
    source = write_source_bundle(
        tmp_path / "source", drop=(("persistence", "decile_90_100", "combined"),)
    )
    with pytest.raises(ContrastInputError, match="missing required series: .*decile_90_100"):
        load_contrast_inputs(source, expected_image_count=6)


def test_refuses_a_wrong_image_count(tmp_path):
    source = write_source_bundle(tmp_path / "source")
    with pytest.raises(ContrastInputError, match="needs 250 tuning images"):
        load_contrast_inputs(source)


def test_refuses_a_duplicated_row_key(tmp_path):
    source = write_source_bundle(tmp_path / "source")
    rows = (source / "per_scene.csv").read_text().splitlines()
    (source / "per_scene.csv").write_text("\n".join(rows + [rows[1]]) + "\n")
    with pytest.raises(ContrastInputError, match="duplicate source row key"):
        load_contrast_inputs(source, expected_image_count=6)


def test_refuses_incomplete_coverage(tmp_path):
    """One row short of the product, with every series, image and severity still present."""
    source = write_source_bundle(tmp_path / "source")
    rows = (source / "per_scene.csv").read_text().splitlines()
    (source / "per_scene.csv").write_text("\n".join(rows[:-1]) + "\n")
    summary = json.loads((source / "summary.json").read_text())
    summary["validation"]["per_scene_row_count"] -= 1
    (source / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ContrastInputError, match=f"expected {RETAINED} retained rows, found"):
        load_contrast_inputs(source, expected_image_count=6)


def test_refuses_provenance_disagreement_between_csv_and_json(tmp_path):
    source = write_source_bundle(tmp_path / "source")
    summary = json.loads((source / "summary.json").read_text())
    summary["run"]["image_count"] = 99
    (source / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ContrastInputError, match="reports image_count=99"):
        load_contrast_inputs(source, expected_image_count=6)


def test_refuses_a_summary_whose_published_row_count_disagrees_with_the_csv(tmp_path):
    source = write_source_bundle(tmp_path / "source")
    summary = json.loads((source / "summary.json").read_text())
    summary["validation"]["per_scene_row_count"] += 1
    (source / "summary.json").write_text(json.dumps(summary))

    with pytest.raises(ContrastInputError, match="per_scene_row_count"):
        load_contrast_inputs(source, expected_image_count=6)


def test_refuses_a_severity_range_that_is_not_the_six(tmp_path):
    source = write_source_bundle(tmp_path / "source", severities=range(5))
    with pytest.raises(ContrastInputError, match="needs severities"):
        load_contrast_inputs(source, expected_image_count=6)


def test_refuses_an_unexpected_retained_severity_even_when_the_row_count_matches(tmp_path):
    """A severity-6 row cannot replace a missing severity-5 row behind the same product count."""
    source = write_source_bundle(tmp_path / "source")
    rows = read_rows(source)
    replaced = next(
        row for row in rows
        if row["severity"] == "5"
        and row["signal"] == "persistence"
        and row["confidence_bin"] == "decile_50_60"
        and row["aggregation"] == "mean"
        and row["score_scope"] == "layer_2"
        and row["image_id"] == "1"
    )
    replaced["severity"] = "6"
    with (source / "per_scene.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ContrastInputError, match=r"unexpected.*severity.*6"):
        load_contrast_inputs(source, expected_image_count=6)


def test_refuses_a_summary_that_is_not_valid_json(tmp_path):
    source = write_source_bundle(tmp_path / "source")
    (source / "summary.json").write_text("{not json,")
    with pytest.raises(ContrastInputError, match="is not valid JSON"):
        load_contrast_inputs(source, expected_image_count=6)


def test_refuses_a_per_scene_csv_with_an_unknown_schema(tmp_path):
    source = write_source_bundle(tmp_path / "source")
    rows = read_rows(source)
    for row in rows:
        row.pop("score")
    with (source / "per_scene.csv").open("w", newline="") as handle:
        fieldnames = [name for name in rows[0] if name != "score"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ContrastInputError, match=r"schema.*score"):
        load_contrast_inputs(source, expected_image_count=6)


def test_refuses_a_bucket_scheme_that_disagrees_with_its_bin(tmp_path):
    source = write_source_bundle(tmp_path / "source")
    rows = read_rows(source)
    mismatched = next(
        row for row in rows
        if row["confidence_bin"] == "decile_50_60"
        and row["signal"] == "persistence"
        and row["score_scope"] == "layer_2"
    )
    mismatched["bucket_scheme"] = "quintile"
    with (source / "per_scene.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ContrastInputError, match=r"bucket_scheme.*decile_50_60"):
        load_contrast_inputs(source, expected_image_count=6)


def test_refuses_a_negative_persistence_distance_at_a_layer_scope(tmp_path):
    def negative(image_id, severity, confidence_bin, signal, scope):
        if (image_id == 3 and confidence_bin == "decile_50_60"
                and signal == "persistence" and scope == "layer_2"):
            return -0.5
        return 1.0

    source = write_source_bundle(tmp_path / "source", score=negative)
    with pytest.raises(ContrastInputError, match="negative persistence score at scope layer_2"):
        load_contrast_inputs(source, expected_image_count=6)


def test_accepts_a_negative_persistence_score_at_the_combined_scope(tmp_path):
    """`combined` is a robust z-score against the clean median, so below-median is negative.

    The completed run carries 548 negative `combined` per-severity statistics and none at
    `layer_2`, and two of the negatives belong to arm 4's own bins -- `decile_50_60`/`combined`
    at severity 0 and `decile_90_100`/`combined` at severity 5. A loader that refused them
    refuses the differential arm the spec added.
    """
    def negative_combined(image_id, severity, confidence_bin, signal, scope):
        return -0.26 if scope == "combined" else 1.0

    source = write_source_bundle(tmp_path / "source", score=negative_combined)
    inputs = load_contrast_inputs(source, expected_image_count=6)
    assert inputs.scores[(3, 5, "persistence", "decile_90_100", "mean", "combined")] == -0.26
    assert inputs.scores[(3, 5, "persistence", "decile_90_100", "mean", "layer_2")] == 1.0


def test_refuses_a_negative_confidence_score(tmp_path):
    """The predicate is default-deny, and this is the test that says so.

    `1 - confidence` is bounded in [0, 1] by construction, so a negative value here is a corrupt
    column rather than a measurement -- and without this case a guard loosened to
    `scope not in (COMBINED_SCOPE, CONFIDENCE_SCOPE)` passes the whole suite.
    """
    def negative_confidence(image_id, severity, confidence_bin, signal, scope):
        return -0.2 if signal == "confidence" else 1.0

    source = write_source_bundle(tmp_path / "source", score=negative_confidence)
    with pytest.raises(
        ContrastInputError, match="negative confidence score at scope confidence"
    ):
        load_contrast_inputs(source, expected_image_count=6)


def test_refuses_a_non_finite_score(tmp_path):
    def infinite(image_id, severity, confidence_bin, signal, scope):
        if image_id == 2 and confidence_bin == "decile_00_10" and signal == "persistence":
            return float("inf")
        return 1.0

    source = write_source_bundle(tmp_path / "source", score=infinite)
    with pytest.raises(ContrastInputError, match="non-finite score"):
        load_contrast_inputs(source, expected_image_count=6)


@pytest.mark.parametrize("missing", ["per_scene.csv", "summary.json"])
def test_refuses_an_unfinished_source(tmp_path, missing):
    source = write_source_bundle(tmp_path / "source")
    (source / missing).unlink()
    with pytest.raises(ContrastInputError, match=f"{missing} missing from"):
        load_contrast_inputs(source, expected_image_count=6)


def test_refuses_a_source_missing_a_producer_column_the_contrast_does_not_read(tmp_path):
    source = write_source_bundle(tmp_path / "source")
    rows = read_rows(source)
    for row in rows:
        row.pop("selected_count")
    with (source / "per_scene.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ContrastInputError, match=r"schema.*selected_count"):
        load_contrast_inputs(source, expected_image_count=6)


def test_refuses_valid_json_whose_top_level_shape_is_not_an_object(tmp_path):
    source = write_source_bundle(tmp_path / "source")
    (source / "summary.json").write_text("[]\n")

    with pytest.raises(ContrastInputError, match=r"summary.json.*object"):
        load_contrast_inputs(source, expected_image_count=6)


def test_refuses_a_summary_whose_run_record_is_not_an_object(tmp_path):
    source = write_source_bundle(tmp_path / "source")
    summary = json.loads((source / "summary.json").read_text())
    summary["run"] = []
    (source / "summary.json").write_text(json.dumps(summary))

    with pytest.raises(ContrastInputError, match=r"run.*object"):
        load_contrast_inputs(source, expected_image_count=6)


def test_refuses_a_summary_whose_severities_are_not_an_array(tmp_path):
    source = write_source_bundle(tmp_path / "source")
    summary = json.loads((source / "summary.json").read_text())
    summary["run"]["severities"] = 5
    (source / "summary.json").write_text(json.dumps(summary))

    with pytest.raises(ContrastInputError, match=r"run\.severities.*array"):
        load_contrast_inputs(source, expected_image_count=6)


def test_refuses_a_truncated_csv_row_even_when_every_consumed_cell_is_present(tmp_path):
    source = write_source_bundle(tmp_path / "source")
    lines = (source / "per_scene.csv").read_text().splitlines()
    lines[1] = ",".join(lines[1].split(",")[:11])
    (source / "per_scene.csv").write_text("\n".join(lines) + "\n")

    with pytest.raises(ContrastInputError, match=r"row 0.*missing value"):
        load_contrast_inputs(source, expected_image_count=6)


def test_refuses_a_nonnumeric_image_id_as_a_clean_input_error(tmp_path):
    source = write_source_bundle(tmp_path / "source")
    rows = read_rows(source)
    rows[0]["image_id"] = "not-an-integer"
    with (source / "per_scene.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ContrastInputError, match=r"row 0.*image_id"):
        load_contrast_inputs(source, expected_image_count=6)
