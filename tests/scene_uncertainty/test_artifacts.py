import json
import pickle
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


# --------------------------------------------------------------------------------------
# Artifact directories are read, never executed
# --------------------------------------------------------------------------------------

_EXECUTED: list[str] = []


def _run_on_unpickle():
    """Stand-in for whatever a hostile shard would run. Records that it ran."""
    _EXECUTED.append("executed")
    return {"image_id": 1, "severity": 0}


class _ArbitraryCodePayload:
    """Pickles as a call to `_run_on_unpickle`, the way any `__reduce__` payload would."""

    def __reduce__(self):
        return (_run_on_unpickle, ())


def _poisoned_artifact(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    torch.save([_ArbitraryCodePayload()], directory / "shard_00000.pt")
    (directory / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "record_count": 1, "shards": ["shard_00000.pt"],
                    "checkpoint_sha256": "abc"}),
        encoding="utf-8",
    )
    return directory


def test_iter_records_reads_a_shard_without_executing_it(tmp_path: Path):
    """Feature caches get copied between hosts by the run bridge.

    A shard is tensors, ints, floats, strs, None and an int-keyed dict of tensors --
    nothing that needs the unpickler's ability to call arbitrary code. Reading one with
    `weights_only=False` therefore buys nothing and hands a copied directory the power to
    run whatever it likes.
    """
    _EXECUTED.clear()
    directory = _poisoned_artifact(tmp_path / "cache")
    with pytest.raises(pickle.UnpicklingError):
        list(iter_records(directory))
    assert _EXECUTED == []


def test_resume_scan_reads_completed_shards_without_executing_them(tmp_path: Path):
    """The resume path deserializes every finished shard before it writes anything."""
    _EXECUTED.clear()
    directory = _poisoned_artifact(tmp_path / "cache")
    (directory / "manifest.json").unlink()
    (directory / "partial_manifest.json").write_text(
        json.dumps({"schema_version": 1, "checkpoint_sha256": "abc", "shard_size": 2,
                    "record_count": 1, "shards": ["shard_00000.pt"]}),
        encoding="utf-8",
    )
    writer = ShardWriter(directory, {"checkpoint_sha256": "abc"}, shard_size=2)
    with pytest.raises(pickle.UnpicklingError):
        writer.existing_record_keys()
    assert _EXECUTED == []


def test_a_real_record_still_survives_the_safe_loader(tmp_path: Path):
    """Every field the extractor ships, including the int-keyed layer dict.

    `weights_only=True` is only an option because none of these need the unpickler: fp16
    layer tensors under **int** keys, fp16 logits, fp32 boxes, bool masks, `None` for
    `reference_group` and str partitions all come back unchanged.
    """
    record = {
        "image_id": 885,
        "severity": 0,
        "blur_radius": 0.0,
        "corruption_type": "gaussian_blur",
        "source_partition": "tuning",
        "reference_group": None,
        "layers": {0: torch.ones(4, 3, dtype=torch.float16), 2: torch.zeros(4, 3, dtype=torch.float16)},
        "logits": torch.zeros(4, 80, dtype=torch.float16),
        "boxes": torch.ones(4, 4),
        "is_matched": torch.tensor([True, False, True, False]),
        "matched_annotation_id": torch.tensor([10, -1, 11, -1]),
    }
    with ShardWriter(tmp_path, {"checkpoint_sha256": "abc"}, shard_size=4) as writer:
        writer.add(record)
    loaded, = iter_records(tmp_path)
    assert sorted(loaded["layers"]) == [0, 2]
    assert all(isinstance(key, int) for key in loaded["layers"])
    assert loaded["layers"][0].dtype == torch.float16
    assert loaded["reference_group"] is None
    assert loaded["source_partition"] == "tuning"
    assert loaded["is_matched"].tolist() == [True, False, True, False]
