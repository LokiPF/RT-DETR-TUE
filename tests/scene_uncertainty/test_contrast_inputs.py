"""What the loader must refuse, and what a real bundle must be allowed to carry.

Half of these tests exist because the *permissive* half of the loader is as load-bearing as the
strict half. A real `per_scene.csv` holds a `frozen` twin of every bin, an `unfiltered` control
at the lowest bin of each scheme, ten deciles, five quintiles and four scopes -- 765,000 rows,
of which this command reads 1,296 per 6-image fixture and 54,000 in production. A refusal aimed
at any of the rows it does not read is a refusal of every bundle the producer can write, so
`test_discards_the_frozen_and_unfiltered_twins_a_real_bundle_carries` is not a nicety: it is the
test that says this command can run at all.
"""
import json
from dataclasses import astuple

import pytest

from src.scene_uncertainty.contrast_inputs import (
    ARMS,
    CONFIDENCE_SCOPE,
    REQUIRED_SERIES,
    ContrastInputError,
    load_contrast_inputs,
)

from tests.scene_uncertainty.contrast_test_utils import (
    AGGREGATIONS,
    SERIES,
    write_source_bundle,
)

IMAGES = range(1, 7)
SEVERITIES = range(6)
RETAINED = 12 * 3 * 6 * 6  # 12 series x 3 aggregations x 6 images x 6 severities


def source_row(**overrides):
    """One raw `per_scene.csv` row for `extra_rows`, defaulting to a row this command reads."""
    row = {
        "image_id": 1, "severity": 0, "signal": "persistence", "bucket_scheme": "decile",
        "confidence_bin": "decile_50_60", "membership_mode": "dynamic",
        "padding_mode": "filtered", "aggregation": "mean", "score_scope": "layer_2",
        "source_partition": "tuning", "score": 1.0, "selected_count": 30,
        "clean_overlap": 0.2, "fully_measured": True, "signed_spearman": 0.6,
        "absolute_spearman": 0.6, "direction": "increasing",
    }
    row.update(overrides)
    return row


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
    assert inputs.scores[(1, 0, "persistence", "decile_50_60", "mean", "layer_2")] == 1.31
    # the same bin at the combined scope is a different measurement, not a copy
    assert inputs.scores[(1, 0, "persistence", "decile_50_60", "mean", "combined")] == 1.81
    # and the confidence twin varies by bin, so a contrast against it is not identically zero
    assert inputs.scores[(1, 0, "confidence", "decile_00_10", "mean", "confidence")] == 0.984
    assert inputs.scores[(1, 0, "confidence", "decile_90_100", "mean", "confidence")] == 0.644
    assert inputs.provenance["source_partition"] == "tuning"
    assert inputs.provenance["retained_row_count"] == RETAINED


def test_the_confidence_column_runs_the_way_the_producer_writes_it(tmp_path):
    """`1 - confidence`: bins descend, and only the top decile climbs with severity.

    Both directions are pinned because neither is visible to the loader. An inverted map keeps
    every count, key and coverage check intact and flips the sign of all 36 confidence-twin
    contrasts four tasks downstream; a fixture where every bin fell with severity would let a
    consumer that had the top decile upside down pass anyway. The rise at `decile_90_100` --
    0.645 to 0.799 in the completed run -- is the detector losing the queries it was surest
    about, which is the behaviour the redundancy control exists to expose.
    """
    inputs = load_contrast_inputs(
        write_source_bundle(tmp_path / "source"), expected_image_count=6
    )

    def column(confidence_bin, severity=0):
        return inputs.scores[(1, severity, "confidence", confidence_bin, "mean", "confidence")]

    descending = [column(name) for name in (
        "decile_00_10", "quintile_00_20", "quintile_40_60", "decile_50_60", "decile_90_100"
    )]
    # strictly, pair by pair: a non-strict check admits a tie, and collapsing `quintile_00_20`
    # onto `decile_00_10` is exactly how the two schemes stop being distinguishable
    assert all(higher > lower for higher, lower in zip(descending, descending[1:]))

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


def test_a_full_tuning_roster_keeps_the_confidence_column_inside_its_bounds(tmp_path):
    """250 images, the size the real command runs at, and the only test that reaches it.

    `1 - confidence` cannot leave [0, 1], and the loader would not notice if it did -- it checks
    negativity and finiteness, and 1.235 is both non-negative and finite. The bound is therefore
    the fixture's own responsibility, and a per-image term that is harmless across six images is
    what breaks it: the roster is the axis this fixture grows along, and every earlier test in
    this file runs six images.

    This is also the only test that exercises the production `expected_image_count` default
    rather than passing 6, and the only one that overrides `image_ids`.
    """
    source = write_source_bundle(tmp_path / "source", image_ids=range(1, 251))
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
        for image_id in range(1, 251)
    ]
    assert len(set(lowest_bin)) == 250


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
                            ))

    source = write_source_bundle(tmp_path / "source", extra_rows=twins)
    inputs = load_contrast_inputs(source, expected_image_count=6)
    assert len(inputs.scores) == RETAINED
    assert inputs.scores[(1, 0, "persistence", "decile_50_60", "mean", "layer_2")] == 1.31
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
    with pytest.raises(ContrastInputError, match=f"expected {RETAINED} retained rows, found"):
        load_contrast_inputs(source, expected_image_count=6)


def test_refuses_provenance_disagreement_between_csv_and_json(tmp_path):
    source = write_source_bundle(tmp_path / "source")
    summary = json.loads((source / "summary.json").read_text())
    summary["run"]["image_count"] = 99
    (source / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ContrastInputError, match="reports image_count=99"):
        load_contrast_inputs(source, expected_image_count=6)


def test_refuses_a_severity_range_that_is_not_the_six(tmp_path):
    source = write_source_bundle(tmp_path / "source", severities=range(5))
    with pytest.raises(ContrastInputError, match="needs severities"):
        load_contrast_inputs(source, expected_image_count=6)


def test_refuses_a_summary_that_is_not_valid_json(tmp_path):
    source = write_source_bundle(tmp_path / "source")
    (source / "summary.json").write_text("{not json,")
    with pytest.raises(ContrastInputError, match="is not valid JSON"):
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
