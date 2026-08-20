import ast
import json
from pathlib import Path

import pytest
import torch

from src.scene_uncertainty import decile_analysis as analysis_module
from src.scene_uncertainty.artifacts import load_manifest
from src.scene_uncertainty.confidence_deciles import (
    confidence_from_logits,
    memberships_by_severity,
    union_query_ids,
)
from src.scene_uncertainty.decile_analysis import (
    SLIM_RECORD_KEYS,
    DecileAnalysisError,
    load_decile_inputs,
)
from tests.scene_uncertainty.decile_test_utils import (
    QUERY_COUNT,
    mutate_decile_artifacts,
    write_decile_artifacts,
)


@pytest.fixture
def decile_artifacts(tmp_path):
    return write_decile_artifacts(tmp_path)


def load(artifacts):
    return load_decile_inputs(artifacts["cache"], artifacts["results"])


# --- the join --------------------------------------------------------------------------


def test_load_inputs_joins_six_tuning_records_by_key(decile_artifacts):
    loaded = load_decile_inputs(decile_artifacts["cache"], decile_artifacts["results"])
    assert set(loaded.records_by_image) == {11}
    assert set(loaded.records_by_image[11]) == set(range(6))
    assert set(loaded.distances) == {(11, severity, "tuning") for severity in range(6)}
    assert set(loaded.layer_score_scales) == {0, 1, 2}


def test_load_inputs_refuses_another_cache(decile_artifacts):
    path = decile_artifacts["results"].with_suffix(".manifest.json")
    manifest = json.loads(path.read_text())
    manifest["feature_cache_id"] = "another-cache"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DecileAnalysisError, match="feature_cache_id"):
        load_decile_inputs(decile_artifacts["cache"], decile_artifacts["results"])


def test_load_inputs_refuses_test_results(decile_artifacts):
    path = decile_artifacts["results"].with_suffix(".manifest.json")
    manifest = json.loads(path.read_text())
    manifest["source_partition"] = "test"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DecileAnalysisError, match="tuning"):
        load_decile_inputs(decile_artifacts["cache"], decile_artifacts["results"])


@pytest.mark.parametrize("mutation,message", [
    ("drop_distance", "missing"),
    ("duplicate_distance", "duplicate"),
    ("wrong_query_count", "query count"),
    ("drop_severity", "six blur severities"),
])
def test_load_inputs_rejects_misalignment(decile_artifacts, mutation, message):
    mutate_decile_artifacts(decile_artifacts, mutation)
    with pytest.raises(DecileAnalysisError, match=message):
        load_decile_inputs(decile_artifacts["cache"], decile_artifacts["results"])


# --- the rest of the design's validation list ------------------------------------------
#
# Every case below names one bullet of "Validation and error handling". The four cases the
# brief supplies leave most of that list unexercised, and each rule that is not pinned by a
# test is a rule that can be deleted without any test noticing -- which for a loader means
# silently widening what the experiment will score.


@pytest.mark.parametrize("mutation,message", [
    # "artifact provenance does not match"
    ("missing_result_manifest", "missing result manifest"),
    ("missing_cache_manifest", "missing feature-cache manifest"),
    ("drop_layer_score_scales", "missing"),
    ("drop_result_cache_id", "feature_cache_id"),
    ("drop_result_k", "missing"),
    ("wrong_result_artifact_type", "artifact_type"),
    ("cache_source_kind", "source_kind"),
    ("cache_manifest_forged", "content address"),
    ("missing_distance_file", "missing query-distance artifact"),
    ("missing_normalizer_file", "missing clean-distance"),
    # "a record key is duplicated or missing from either input"
    ("duplicate_cache_record", "duplicate feature-cache record"),
    ("extra_distance_row", "missing feature-cache record"),
    ("no_tuning_records", "no tuning records"),
    # "an image lacks severity zero or any of the six expected severities"
    ("drop_severity_zero", "six blur severities"),
    # "a record is not in the requested tuning partition". The message fragment is
    # deliberately specific: a loader that *skipped* the foreign row instead of refusing it
    # still raises later, with "missing query-distance record for (11, 0, 'tuning')", and a
    # bare match on "tuning" would call that a pass.
    ("test_partition_row", r"is in the 'test' partition"),
    # "layer IDs or query counts disagree"
    ("cache_layer_missing", "persistence layers"),
    ("distance_layer_missing", "query-distance layers"),
    ("scale_layer_missing", "decoder layers"),
    ("cache_query_count", "query count"),
    # "fewer than ten valid queries remain"
    ("everything_padded", "ten valid queries"),
])
def test_load_inputs_rejects_every_named_validation_failure(decile_artifacts, mutation, message):
    mutate_decile_artifacts(decile_artifacts, mutation)
    with pytest.raises(DecileAnalysisError, match=message):
        load_decile_inputs(decile_artifacts["cache"], decile_artifacts["results"])


@pytest.mark.parametrize("field", ["artifact_id", "feature_cache_id", "k", "query_distance_path"])
def test_a_null_result_manifest_field_is_missing_data(decile_artifacts, field):
    """`"feature_cache_id": null` is a key that is present and a value that is not an id."""
    path = decile_artifacts["results"].with_suffix(".manifest.json")
    manifest = json.loads(path.read_text())
    manifest[field] = None
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DecileAnalysisError, match="missing"):
        load(decile_artifacts)


@pytest.mark.parametrize("field", ["artifact_id", "source_kind", "decoder_layers", "query_count"])
def test_a_null_cache_manifest_field_is_missing_data(decile_artifacts, field):
    path = decile_artifacts["cache"] / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[field] = None
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DecileAnalysisError, match="missing"):
        load(decile_artifacts)


def test_a_missing_cache_id_on_both_sides_is_not_a_match(decile_artifacts):
    """`result.get(...) == cache.get(...)` is True when both are absent, and means nothing."""
    manifest_path = decile_artifacts["cache"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("artifact_id")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    mutate_decile_artifacts(decile_artifacts, "drop_result_cache_id")
    with pytest.raises(DecileAnalysisError, match="missing"):
        load(decile_artifacts)


# --- the slim record -------------------------------------------------------------------


def test_slim_records_keep_the_key_the_confidence_and_the_mask_and_nothing_else(decile_artifacts):
    """The design's memory contract: persistence tensors are discarded after the suffix check."""
    record = load(decile_artifacts).records_by_image[11][0]
    assert SLIM_RECORD_KEYS == {
        "image_id", "severity", "source_partition",
        "query_confidence", "padded_query_ids", "query_count",
    }
    assert set(record) == SLIM_RECORD_KEYS
    assert record["image_id"] == 11
    assert record["severity"] == 0
    assert record["source_partition"] == "tuning"
    assert record["query_count"] == QUERY_COUNT
    assert record["padded_query_ids"].tolist() == [QUERY_COUNT - 2, QUERY_COUNT - 1]


def test_slim_records_never_carry_the_caches_stale_confidence_field(decile_artifacts):
    """Ruling 7: the derived vector is `query_confidence`; the cached field must not survive."""
    mutate_decile_artifacts(decile_artifacts, "stale_confidence_field")
    record = load(decile_artifacts).records_by_image[11][0]
    assert "confidence" not in record
    assert float(record["query_confidence"].max()) > 0.5


def test_query_confidence_is_recomputed_from_the_cached_float16_logits(decile_artifacts):
    manifest = load_manifest(decile_artifacts["cache"])
    shard = decile_artifacts["cache"] / manifest["shards"][0]
    cached = torch.load(shard, map_location="cpu", weights_only=True)
    expected = confidence_from_logits(next(r for r in cached if r["severity"] == 3)["logits"])
    record = load(decile_artifacts).records_by_image[11][3]
    assert torch.equal(record["query_confidence"], expected)


def test_slim_records_feed_membership_construction_unchanged(decile_artifacts):
    """The handshake with Task 2: only a record keyed `query_confidence` gets past it."""
    records = load(decile_artifacts).records_by_image[11]
    padded = union_query_ids([record["padded_query_ids"] for record in records.values()])
    memberships = memberships_by_severity(records, padded)
    assert set(memberships) == set(range(6))
    assert padded.tolist() == [QUERY_COUNT - 2, QUERY_COUNT - 1]


def test_slim_records_name_their_image_in_a_membership_error(decile_artifacts):
    """Without `image_id` on the slim record every labelled error would read "severity 0"."""
    records = load(decile_artifacts).records_by_image[11]
    with pytest.raises(ValueError, match="image 11 severity 0"):
        memberships_by_severity(records, torch.arange(QUERY_COUNT - 9))


# --- what the loader hands on ----------------------------------------------------------


def test_distances_keep_one_vector_per_layer_per_key(decile_artifacts):
    loaded = load(decile_artifacts)
    for key, by_layer in loaded.distances.items():
        assert set(by_layer) == {0, 1, 2}, key
        assert all(int(values.numel()) == QUERY_COUNT for values in by_layer.values()), key


def test_run_metadata_carries_the_provenance_of_both_inputs(decile_artifacts):
    loaded = load(decile_artifacts)
    metadata = loaded.run_metadata
    assert metadata["artifact_type"] == "confidence_decile_scene_uncertainty"
    assert metadata["feature_cache_id"] == load_manifest(decile_artifacts["cache"])["artifact_id"]
    assert metadata["source_result_id"] == "synthetic-results"
    assert metadata["source_partition"] == "tuning"
    assert metadata["normalization"] == "raw"
    assert metadata["k"] == 5
    assert metadata["severities"] == [0, 1, 2, 3, 4, 5]
    assert metadata["decoder_layers"] == [0, 1, 2]
    assert metadata["query_count"] == QUERY_COUNT
    assert metadata["image_count"] == 1
    assert metadata["record_count"] == 6


# --- no kNN, no inference --------------------------------------------------------------


def test_the_loader_cannot_reach_a_knn_or_extraction_path():
    """The design's non-goal: consume the *saved* distances, never recompute them."""
    source = Path(analysis_module.__file__).read_text()
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    forbidden = {"knn", "evaluate", "extractor", "bank", "dataset", "pipeline"}
    assert not {name for name in imported if name.lstrip(".").split(".")[0] in forbidden}
    assert not {
        name for name, value in vars(analysis_module).items()
        if getattr(value, "__module__", "").endswith((".knn", ".evaluate", ".extractor"))
    }
