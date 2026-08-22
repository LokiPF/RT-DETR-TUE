import json

import pytest

from src.scene_uncertainty.contrast_inputs import (
    ARMS,
    ContrastInputError,
    load_contrast_inputs,
)

from tests.scene_uncertainty.contrast_test_utils import write_source_bundle


def test_arm_table_is_the_four_declared_arms():
    assert [arm.name for arm in ARMS] == [
        "decile_00_10__50_60",
        "quintile_00_20__40_60",
        "decile_90_100__50_60",
        "decile_90_100__50_60__combined",
    ]
    assert [arm.family for arm in ARMS] == [
        "anchored", "anchored", "differential", "differential"
    ]
    assert [arm.declared_before_data for arm in ARMS] == [True, True, False, False]
    assert [arm.score_scope for arm in ARMS] == [
        "layer_2", "layer_2", "layer_2", "combined"
    ]
    # the two differential arms share one bucket pair, and therefore one confidence twin
    assert [arm.pair_name for arm in ARMS] == [
        "decile_00_10__50_60",
        "quintile_00_20__40_60",
        "decile_90_100__50_60",
        "decile_90_100__50_60",
    ]
    assert len({arm.pair_name for arm in ARMS}) == 3


def test_loads_every_required_series(tmp_path):
    source = write_source_bundle(tmp_path / "source")
    inputs = load_contrast_inputs(source, expected_image_count=6)
    assert inputs.image_ids == (1, 2, 3, 4, 5, 6)
    # 12 series x 3 aggregations x 6 images x 6 severities
    assert len(inputs.scores) == 12 * 3 * 6 * 6
    assert inputs.scores[(1, 0, "persistence", "decile_50_60", "mean", "layer_2")] == 1.31
    assert inputs.provenance["source_partition"] == "tuning"


def test_refuses_a_held_out_source(tmp_path):
    source = write_source_bundle(tmp_path / "source", partition="held_out")
    with pytest.raises(ContrastInputError, match="tuning"):
        load_contrast_inputs(source, expected_image_count=6)


def test_refuses_a_missing_required_series(tmp_path):
    source = write_source_bundle(
        tmp_path / "source", drop=(("persistence", "decile_90_100", "combined"),)
    )
    with pytest.raises(ContrastInputError, match="decile_90_100.*combined"):
        load_contrast_inputs(source, expected_image_count=6)


def test_refuses_a_wrong_image_count(tmp_path):
    source = write_source_bundle(tmp_path / "source")
    with pytest.raises(ContrastInputError, match="250"):
        load_contrast_inputs(source)


def test_refuses_a_duplicated_row_key(tmp_path):
    source = write_source_bundle(tmp_path / "source")
    rows = (source / "per_scene.csv").read_text().splitlines()
    (source / "per_scene.csv").write_text("\n".join(rows + [rows[1]]) + "\n")
    with pytest.raises(ContrastInputError, match="duplicate"):
        load_contrast_inputs(source, expected_image_count=6)


def test_refuses_provenance_disagreement_between_csv_and_json(tmp_path):
    source = write_source_bundle(tmp_path / "source")
    summary = json.loads((source / "summary.json").read_text())
    summary["run"]["image_count"] = 99
    (source / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ContrastInputError, match="image_count"):
        load_contrast_inputs(source, expected_image_count=6)


def test_refuses_a_negative_persistence_distance(tmp_path):
    def negative(image_id, severity, confidence_bin, signal, scope):
        if image_id == 3 and confidence_bin == "decile_50_60" and signal == "persistence":
            return -0.5
        return 1.0

    source = write_source_bundle(tmp_path / "source", score=negative)
    with pytest.raises(ContrastInputError, match="negative"):
        load_contrast_inputs(source, expected_image_count=6)


@pytest.mark.parametrize("missing", ["per_scene.csv", "summary.json"])
def test_refuses_an_unfinished_source(tmp_path, missing):
    source = write_source_bundle(tmp_path / "source")
    (source / missing).unlink()
    with pytest.raises(ContrastInputError, match=missing):
        load_contrast_inputs(source, expected_image_count=6)
