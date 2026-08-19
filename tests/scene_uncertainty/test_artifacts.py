from pathlib import Path

import pytest
import torch

from src.scene_uncertainty.artifacts import (
    ShardWriter,
    assert_compatible,
    iter_records,
    load_manifest,
    manifest_id,
)


def test_shard_writer_round_trip(tmp_path: Path):
    metadata = {"checkpoint_sha256": "abc", "decoder_layers": [0, 1, 2], "query_count": 300}
    with ShardWriter(tmp_path, metadata, shard_size=2) as writer:
        writer.add({"image_id": 1, "layers": {0: torch.ones(300, 4)}})
        writer.add({"image_id": 2, "layers": {0: torch.zeros(300, 4)}})
        writer.add({"image_id": 3, "layers": {0: torch.full((300, 4), 2.0)}})
    manifest = load_manifest(tmp_path)
    assert manifest["record_count"] == 3
    assert len(manifest["artifact_id"]) == 64
    assert manifest["shards"] == ["shard_00000.pt", "shard_00001.pt"]
    assert [record["image_id"] for record in iter_records(tmp_path)] == [1, 2, 3]


def test_interrupted_writer_resumes_completed_shards(tmp_path: Path):
    metadata = {"checkpoint_sha256": "abc", "decoder_layers": [0], "query_count": 300}
    interrupted = ShardWriter(tmp_path, metadata, shard_size=2)
    interrupted.add({"image_id": 1, "severity": 0})
    interrupted.add({"image_id": 2, "severity": 0})
    resumed = ShardWriter(tmp_path, metadata, shard_size=2)
    assert resumed.existing_record_keys() == {(1, 0), (2, 0)}
    resumed.add({"image_id": 3, "severity": 0})
    resumed.close()
    assert [record["image_id"] for record in iter_records(tmp_path)] == [1, 2, 3]


def test_completed_artifact_is_immutable(tmp_path: Path):
    with ShardWriter(tmp_path, {"checkpoint_sha256": "abc"}, shard_size=2):
        pass
    with pytest.raises(FileExistsError, match="complete"):
        ShardWriter(tmp_path, {"checkpoint_sha256": "abc"}, shard_size=2)


def test_compatibility_rejects_checkpoint_mismatch():
    expected = {"checkpoint_sha256": "a", "decoder_layers": [0, 1, 2]}
    actual = {"checkpoint_sha256": "b", "decoder_layers": [0, 1, 2]}
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        assert_compatible(actual, expected, keys=("checkpoint_sha256", "decoder_layers"))


def test_artifact_id_is_content_addressed(tmp_path: Path):
    metadata = {"checkpoint_sha256": "abc", "decoder_layers": [0, 1, 2]}

    def build(name: str, meta: dict) -> dict:
        with ShardWriter(tmp_path / name, meta, shard_size=2) as writer:
            writer.add({"image_id": 1})
        return load_manifest(tmp_path / name)

    first = build("first", metadata)
    second = build("second", metadata)
    other = build("other", {**metadata, "checkpoint_sha256": "def"})
    # Same metadata and same contents must give the same id from any directory,
    # and any metadata change must move it.
    assert first["artifact_id"] == second["artifact_id"]
    assert other["artifact_id"] != first["artifact_id"]
    # A consumer verifies an artifact by recomputing the id from the stored
    # manifest, which only reproduces the stored digest because `manifest_id`
    # excludes `artifact_id` from what it hashes.
    assert manifest_id(first) == first["artifact_id"]


def test_close_leaves_only_the_shards_and_the_final_manifest(tmp_path: Path):
    with ShardWriter(tmp_path, {"checkpoint_sha256": "abc"}, shard_size=2) as writer:
        writer.add({"image_id": 1})
        writer.add({"image_id": 2})
        writer.add({"image_id": 3})
    # No partial manifest and no `.tmp` staging files survive a clean close, and
    # each shard staged under its own temporary name.
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "manifest.json",
        "shard_00000.pt",
        "shard_00001.pt",
    ]


def test_existing_record_keys_defaults_a_missing_severity_to_zero(tmp_path: Path):
    metadata = {"checkpoint_sha256": "abc"}
    interrupted = ShardWriter(tmp_path, metadata, shard_size=2)
    interrupted.add({"image_id": 1})
    interrupted.add({"image_id": 2})
    # Uncorrupted caches carry no `severity` field, so resume must still key them.
    assert ShardWriter(tmp_path, metadata, shard_size=2).existing_record_keys() == {(1, 0), (2, 0)}


def test_resume_rejects_metadata_mismatch(tmp_path: Path):
    interrupted = ShardWriter(tmp_path, {"checkpoint_sha256": "abc"}, shard_size=2)
    interrupted.add({"image_id": 1})
    interrupted.add({"image_id": 2})
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        ShardWriter(tmp_path, {"checkpoint_sha256": "def"}, shard_size=2)


def test_resume_rejects_a_different_shard_size(tmp_path: Path):
    metadata = {"checkpoint_sha256": "abc"}
    interrupted = ShardWriter(tmp_path, metadata, shard_size=2)
    interrupted.add({"image_id": 1})
    interrupted.add({"image_id": 2})
    with pytest.raises(ValueError, match="shard_size"):
        ShardWriter(tmp_path, metadata, shard_size=3)


def test_compatibility_ignores_keys_outside_the_requested_set():
    assert_compatible(
        {"checkpoint_sha256": "a", "decoder_layers": [0]},
        {"checkpoint_sha256": "a", "decoder_layers": [0, 1, 2]},
        keys=("checkpoint_sha256",),
    )


def test_compatibility_reports_a_key_the_artifact_never_recorded():
    with pytest.raises(ValueError, match="query_count"):
        assert_compatible(
            {"checkpoint_sha256": "a"},
            {"checkpoint_sha256": "a", "query_count": 300},
            keys=("checkpoint_sha256", "query_count"),
        )


def test_resume_accepts_metadata_that_json_rewrites(tmp_path: Path):
    # Task 12 writes `corruption.radii` keyed by int severity and passes tuples
    # for list-valued settings. JSON persists int keys as strings and tuples as
    # lists, so a partial manifest never compares equal to the live metadata it
    # was written from unless both sides are canonicalised first -- and the
    # failure looks exactly like the checkpoint-mismatch guard firing correctly.
    metadata = {
        "checkpoint_sha256": "abc",
        "decoder_layers": (0, 1, 2),
        "corruption": {
            "type": "gaussian_blur",
            "radii": {severity: float(severity) for severity in range(6)},
        },
    }
    interrupted = ShardWriter(tmp_path, metadata, shard_size=2)
    interrupted.add({"image_id": 1})
    interrupted.add({"image_id": 2})
    resumed = ShardWriter(tmp_path, metadata, shard_size=2)
    assert resumed.existing_record_keys() == {(1, 0), (2, 0)}
    resumed.add({"image_id": 3})
    resumed.close()
    assert [record["image_id"] for record in iter_records(tmp_path)] == [1, 2, 3]


def test_canonicalisation_holds_past_the_single_digit_key_boundary(tmp_path: Path):
    # `sort_keys` orders int keys numerically (..., 9, 10, 11) but orders the
    # strings JSON rewrites them to lexicographically ("1", "10", "11", ..., "9"),
    # so a single canonical dump of each side only agrees while every key is one
    # digit long. Both the resume comparison and the content-addressed id have to
    # survive the eleventh key.
    metadata = {
        "checkpoint_sha256": "abc",
        "radii": {severity: float(severity) for severity in range(12)},
    }
    interrupted = ShardWriter(tmp_path, metadata, shard_size=2)
    interrupted.add({"image_id": 1})
    interrupted.add({"image_id": 2})
    resumed = ShardWriter(tmp_path, metadata, shard_size=2)
    resumed.close()
    manifest = load_manifest(tmp_path)
    assert manifest_id(manifest) == manifest["artifact_id"]


def test_close_is_idempotent(tmp_path: Path):
    writer = ShardWriter(tmp_path, {"checkpoint_sha256": "abc"}, shard_size=2)
    writer.add({"image_id": 1})
    writer.close()
    writer.close()
    assert [record["image_id"] for record in iter_records(tmp_path)] == [1]
