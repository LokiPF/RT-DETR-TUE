import hashlib
import json
import pickle
from pathlib import Path

import pytest
import torch

import differential_uncertainty.artifacts as artifacts
from differential_uncertainty.artifacts import (
    ShardWriter,
    atomic_json,
    atomic_torch,
    ensure_provenance,
    iter_records,
    load_manifest,
    sha256_file,
    source_digest,
)


def _record(image_id: str, severity: int = 0, value: int = 1) -> dict:
    return {
        "image_id": image_id,
        "severity": severity,
        "value": torch.tensor([value]),
    }


def test_atomic_json_and_torch_publish_complete_values_without_staging_files(tmp_path):
    json_path = tmp_path / "nested" / "value.json"
    tensor_path = tmp_path / "nested" / "value.pt"

    atomic_json({"z": 2, "a": [1, 3]}, json_path)
    atomic_torch({"value": torch.tensor([4, 5])}, tensor_path)

    assert json.loads(json_path.read_text(encoding="utf-8")) == {
        "a": [1, 3],
        "z": 2,
    }
    loaded = torch.load(tensor_path, map_location="cpu", weights_only=True)
    torch.testing.assert_close(loaded["value"], torch.tensor([4, 5]))
    assert list(tmp_path.rglob("*.tmp")) == []


def test_failed_atomic_torch_publication_preserves_the_target(tmp_path, monkeypatch):
    target = tmp_path / "value.pt"
    target.write_bytes(b"previous")

    def fail_after_staging(_value, path):
        Path(path).write_bytes(b"incomplete")
        raise RuntimeError("save failed")

    monkeypatch.setattr(artifacts.torch, "save", fail_after_staging)

    with pytest.raises(RuntimeError, match="save failed"):
        atomic_torch({"new": True}, target)

    assert target.read_bytes() == b"previous"
    assert list(tmp_path.glob("*.tmp")) == []


def test_sha256_file_hashes_content_in_chunks_not_the_filename(tmp_path):
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    first.write_bytes(b"same content")
    second.write_bytes(b"same content")

    expected = hashlib.sha256(b"same content").hexdigest()
    assert sha256_file(first, chunk_size=2) == expected
    assert sha256_file(second, chunk_size=3) == expected


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True])
def test_sha256_file_rejects_invalid_chunk_sizes(tmp_path, chunk_size):
    path = tmp_path / "value.bin"
    path.write_bytes(b"value")

    with pytest.raises(ValueError, match="chunk_size"):
        sha256_file(path, chunk_size=chunk_size)


def test_source_digest_is_clone_stable_and_depends_on_relative_names_and_content(tmp_path):
    first_root = tmp_path / "first-clone"
    second_root = tmp_path / "second-clone"
    for root in (first_root, second_root):
        (root / "package").mkdir(parents=True)
        (root / "package" / "a.py").write_text("A = 1\n", encoding="utf-8")
        (root / "package" / "b.py").write_text("B = 2\n", encoding="utf-8")

    first = source_digest(
        [first_root / "package" / "b.py", first_root / "package" / "a.py"],
        root=first_root,
    )
    second = source_digest(
        [second_root / "package" / "a.py", second_root / "package" / "b.py"],
        root=second_root,
    )
    assert first == second

    (second_root / "package" / "b.py").write_text("B = 3\n", encoding="utf-8")
    changed_content = source_digest(
        [second_root / "package" / "a.py", second_root / "package" / "b.py"],
        root=second_root,
    )
    assert changed_content != first

    (second_root / "renamed.py").write_text("A = 1\n", encoding="utf-8")
    changed_name = source_digest([second_root / "renamed.py"], root=second_root)
    original_name = source_digest([second_root / "package" / "a.py"], root=second_root)
    assert changed_name != original_name


def test_source_digest_frames_names_and_content_and_rejects_files_outside_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    short_name = root / "a"
    long_name = root / "ab"
    short_name.write_bytes(b"bc")
    long_name.write_bytes(b"c")

    assert source_digest([short_name], root=root) != source_digest([long_name], root=root)
    with pytest.raises(ValueError, match="outside source root"):
        source_digest([tmp_path / "outside.py"], root=root)


def test_provenance_is_published_once_and_accepts_a_json_equivalent_repeat(tmp_path):
    run = tmp_path / "run"
    expected = {
        "schema_version": 1,
        "checkpoint_sha256": "first",
        "config": {"layers": (2,), "radii": {0: 0.0, 10: 12.0}},
    }

    path = ensure_provenance(run, expected)
    ensure_provenance(
        run,
        {
            "schema_version": 1,
            "checkpoint_sha256": "first",
            "config": {"layers": [2], "radii": {"0": 0.0, "10": 12.0}},
        },
    )

    assert path == run / "artifacts" / "provenance.json"
    assert json.loads(path.read_text(encoding="utf-8"))["checkpoint_sha256"] == "first"
    assert list(run.rglob("*.tmp")) == []


@pytest.mark.parametrize(
    "changed, mismatched_key",
    [
        (
            {"schema_version": 1, "checkpoint_sha256": "second", "config": {"k": 5}},
            "checkpoint_sha256",
        ),
        ({"schema_version": 1, "checkpoint_sha256": "first"}, "config"),
        (
            {
                "schema_version": 1,
                "checkpoint_sha256": "first",
                "config": {"k": 5},
                "extra": True,
            },
            "extra",
        ),
    ],
)
def test_provenance_refuses_any_exact_key_or_value_mismatch(
    tmp_path,
    changed,
    mismatched_key,
):
    run = tmp_path / "run"
    original = {
        "schema_version": 1,
        "checkpoint_sha256": "first",
        "config": {"k": 5},
    }
    ensure_provenance(run, original)

    with pytest.raises(ValueError, match=mismatched_key):
        ensure_provenance(run, changed)


def test_completed_shards_publish_a_final_manifest_and_iterate_in_order(tmp_path):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation", "input_id": "abc"}

    with ShardWriter(root, metadata, shard_size=2) as writer:
        writer.add(_record("a", severity=0, value=1))
        writer.add(_record("a", severity=1, value=2))
        writer.add(_record("b", severity=0, value=3))

    manifest = load_manifest(root)
    assert manifest == {
        "input_id": "abc",
        "record_count": 3,
        "schema_version": 1,
        "shards": ["shard_00000.pt", "shard_00001.pt"],
        "stage": "evaluation",
    }
    assert [
        (record["image_id"], record["severity"], record["value"].item())
        for record in iter_records(root)
    ] == [("a", 0, 1), ("a", 1, 2), ("b", 0, 3)]
    assert sorted(path.name for path in root.iterdir()) == [
        "manifest.json",
        "shard_00000.pt",
        "shard_00001.pt",
    ]


def test_partial_shards_resume_from_published_record_keys(tmp_path):
    root = tmp_path / "cache"
    metadata = {"stage": "reference", "input_id": "abc"}
    interrupted = ShardWriter(root, metadata, shard_size=1)
    interrupted.add(_record("a"))

    resumed = ShardWriter(root, metadata, shard_size=1)
    assert resumed.existing_keys() == {("a", 0)}
    resumed.add(_record("b", value=2))
    resumed.close()

    assert [record["image_id"] for record in iter_records(root)] == ["a", "b"]


def test_exceptional_context_keeps_only_full_shards_resumable(tmp_path):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}

    with pytest.raises(RuntimeError, match="interrupted"):
        with ShardWriter(root, metadata, shard_size=2) as writer:
            writer.add(_record("a"))
            writer.add(_record("b"))
            writer.add(_record("not-published"))
            raise RuntimeError("interrupted")

    assert not (root / "manifest.json").exists()
    resumed = ShardWriter(root, metadata, shard_size=2)
    assert resumed.existing_keys() == {("a", 0), ("b", 0)}
    resumed.close()
    assert [record["image_id"] for record in iter_records(root)] == ["a", "b"]


def test_duplicate_keys_are_rejected_in_memory_and_after_resume(tmp_path):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    writer = ShardWriter(root, metadata, shard_size=2)
    writer.add({"image_id": "a", "value": torch.tensor([1])})

    with pytest.raises(ValueError, match="duplicate record key.*a.*0"):
        writer.add(_record("a", severity=0))

    writer.add(_record("b"))
    resumed = ShardWriter(root, metadata, shard_size=2)
    with pytest.raises(ValueError, match="duplicate record key.*b.*0"):
        resumed.add(_record("b"))
    resumed.close()

    assert [record["image_id"] for record in iter_records(root)] == ["a", "b"]


@pytest.mark.parametrize("shard_size", [0, -1, 1.5, True])
def test_shard_size_must_be_a_positive_integer(tmp_path, shard_size):
    with pytest.raises(ValueError, match="shard_size must be a positive integer"):
        ShardWriter(tmp_path / "cache", {}, shard_size=shard_size)


def test_resume_refuses_metadata_or_shard_size_changes(tmp_path):
    root = tmp_path / "cache"
    writer = ShardWriter(root, {"stage": "reference", "input_id": "abc"}, shard_size=1)
    writer.add(_record("a"))

    with pytest.raises(ValueError, match="input_id"):
        ShardWriter(root, {"stage": "reference", "input_id": "other"}, shard_size=1)
    with pytest.raises(ValueError, match="unexpected"):
        ShardWriter(
            root,
            {"stage": "reference", "input_id": "abc", "unexpected": True},
            shard_size=1,
        )
    with pytest.raises(ValueError, match="shard_size"):
        ShardWriter(root, {"stage": "reference", "input_id": "abc"}, shard_size=2)


@pytest.mark.parametrize(
    "shards",
    [
        ["shard_00001.pt"],
        ["shard_00000.pt", "shard_00000.pt"],
    ],
)
def test_resume_rejects_noncanonical_or_duplicate_shard_sequences(tmp_path, shards):
    root = tmp_path / "cache"
    root.mkdir()
    torch.save([_record("a")], root / "shard_00000.pt")
    torch.save([_record("b")], root / "shard_00001.pt")
    (root / "partial_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "reference",
                "shard_size": 1,
                "record_count": len(shards),
                "shards": shards,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical shard sequence"):
        ShardWriter(root, {"stage": "reference"}, shard_size=1)


def test_iter_records_rejects_duplicate_shard_names(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    torch.save([_record("a")], root / "shard_00000.pt")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_count": 2,
                "shards": ["shard_00000.pt", "shard_00000.pt"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical shard sequence"):
        list(iter_records(root))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("schema_version", 1.0),
        ("shard_size", True),
        ("shard_size", 1.0),
    ],
)
def test_resume_rejects_noninteger_manifest_control_values(
    tmp_path,
    field,
    value,
):
    root = tmp_path / "cache"
    root.mkdir()
    state = {
        "schema_version": 1,
        "stage": "reference",
        "shard_size": 1,
        "record_count": 0,
        "shards": [],
    }
    state[field] = value
    (root / "partial_manifest.json").write_text(
        json.dumps(state),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=field):
        ShardWriter(root, {"stage": "reference"}, shard_size=1)


def test_completed_writer_is_immutable_and_close_is_idempotent(tmp_path):
    root = tmp_path / "cache"
    writer = ShardWriter(root, {"stage": "reference"}, shard_size=2)
    writer.add(_record("a"))
    writer.close()
    writer.close()

    with pytest.raises(RuntimeError, match="closed"):
        writer.add(_record("b"))
    with pytest.raises(FileExistsError, match="already complete"):
        ShardWriter(root, {"stage": "reference"}, shard_size=2)


def test_artifact_reads_always_use_safe_cpu_torch_load(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    writer = ShardWriter(root, {"stage": "reference"}, shard_size=1)
    writer.add(_record("a"))

    calls = []
    original_load = torch.load

    def observed_load(*args, **kwargs):
        calls.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(artifacts.torch, "load", observed_load)
    resumed = ShardWriter(root, {"stage": "reference"}, shard_size=1)
    assert resumed.existing_keys() == {("a", 0)}
    resumed.close()
    assert [record["image_id"] for record in iter_records(root)] == ["a"]
    assert calls
    assert all(call == {"map_location": "cpu", "weights_only": True} for call in calls)


_EXECUTED: list[str] = []


def _run_on_unpickle():
    _EXECUTED.append("executed")
    return _record("executed")


class _ArbitraryCodePayload:
    def __reduce__(self):
        return (_run_on_unpickle, ())


def test_iter_records_does_not_execute_pickle_payloads(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    torch.save([_ArbitraryCodePayload()], root / "shard_00000.pt")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_count": 1,
                "shards": ["shard_00000.pt"],
            }
        ),
        encoding="utf-8",
    )
    _EXECUTED.clear()

    with pytest.raises(pickle.UnpicklingError):
        list(iter_records(root))

    assert _EXECUTED == []
