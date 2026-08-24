import dis
import gc
import hashlib
import inspect
import json
import os
import pickle
import sys
import threading
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


def _write_manual_artifact(root: Path, records: list[dict]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    name = "shard_00000.pt"
    shard = root / name
    torch.save(records, shard)
    atomic_json(
        {
            "schema_version": 1,
            "record_count": len(records),
            "shards": [name],
            "shard_sha256": {name: sha256_file(shard)},
        },
        root / "manifest.json",
    )
    return shard


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

    def fail_after_staging(_value, destination):
        if hasattr(destination, "write"):
            destination.write(b"incomplete")
        else:
            Path(destination).write_bytes(b"incomplete")
        raise RuntimeError("save failed")

    monkeypatch.setattr(artifacts.torch, "save", fail_after_staging)

    with pytest.raises(RuntimeError, match="save failed"):
        atomic_torch({"new": True}, target)

    assert target.read_bytes() == b"previous"
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_json_does_not_follow_a_predictable_staging_symlink(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("protected", encoding="utf-8")
    target = tmp_path / "value.json"
    predictable = tmp_path / "value.json.tmp"
    predictable.symlink_to(victim)

    atomic_json({"value": 3}, target)

    assert victim.read_text(encoding="utf-8") == "protected"
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 3}
    assert predictable.is_symlink()


def test_atomic_torch_does_not_follow_a_predictable_staging_symlink(tmp_path):
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"protected")
    target = tmp_path / "value.pt"
    predictable = tmp_path / "value.pt.tmp"
    predictable.symlink_to(victim)

    atomic_torch({"value": torch.tensor([3])}, target)

    assert victim.read_bytes() == b"protected"
    loaded = torch.load(target, map_location="cpu", weights_only=True)
    torch.testing.assert_close(loaded["value"], torch.tensor([3]))
    assert predictable.is_symlink()


def test_atomic_publication_fsyncs_staged_files_and_parent_directories(
    tmp_path,
    monkeypatch,
):
    fsynced = []
    monkeypatch.setattr(artifacts.os, "fsync", fsynced.append)

    atomic_json({"value": 1}, tmp_path / "value.json")
    atomic_torch({"value": torch.tensor([2])}, tmp_path / "value.pt")

    assert len(fsynced) == 4


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


def test_source_digest_is_clone_stable_and_depends_on_relative_names_and_content(
    tmp_path,
):
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


def test_source_digest_frames_names_and_content_and_rejects_files_outside_root(
    tmp_path,
):
    root = tmp_path / "root"
    root.mkdir()
    short_name = root / "a"
    long_name = root / "ab"
    short_name.write_bytes(b"bc")
    long_name.write_bytes(b"c")

    assert source_digest([short_name], root=root) != source_digest(
        [long_name], root=root
    )
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


def test_concurrent_provenance_publishers_cannot_both_accept_different_runs(
    tmp_path,
    monkeypatch,
):
    run = tmp_path / "run"
    first = {"schema_version": 1, "checkpoint_sha256": "first"}
    second = {"schema_version": 1, "checkpoint_sha256": "second"}
    atomic_barrier = threading.Barrier(2)
    link_barrier = threading.Barrier(2)
    original_atomic_json = artifacts.atomic_json
    original_link = artifacts.os.link

    def synchronized_atomic_json(value, path):
        if Path(path).name == "provenance.json":
            atomic_barrier.wait(timeout=5)
        return original_atomic_json(value, path)

    def synchronized_link(*args, **kwargs):
        link_barrier.wait(timeout=5)
        return original_link(*args, **kwargs)

    monkeypatch.setattr(artifacts, "atomic_json", synchronized_atomic_json)
    monkeypatch.setattr(artifacts.os, "link", synchronized_link)
    outcomes = []

    def publish(expected):
        try:
            ensure_provenance(run, expected)
        except Exception as error:
            outcomes.append((type(error).__name__, expected["checkpoint_sha256"]))
        else:
            outcomes.append(("ok", expected["checkpoint_sha256"]))

    threads = [
        threading.Thread(target=publish, args=(expected,))
        for expected in (first, second)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not any(thread.is_alive() for thread in threads)
    assert sorted(status for status, _checkpoint in outcomes) == ["ValueError", "ok"]
    winner = next(checkpoint for status, checkpoint in outcomes if status == "ok")
    persisted = json.loads(
        (run / "artifacts" / "provenance.json").read_text(encoding="utf-8")
    )
    assert persisted["checkpoint_sha256"] == winner


def test_failed_exclusive_provenance_link_cleans_its_staging_file(
    tmp_path,
    monkeypatch,
):
    run = tmp_path / "run"

    def fail_link(*_args, **_kwargs):
        raise OSError("link failed")

    monkeypatch.setattr(artifacts.os, "link", fail_link)
    with pytest.raises(OSError, match="link failed"):
        ensure_provenance(run, {"schema_version": 1})

    assert not (run / "artifacts" / "provenance.json").exists()
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
        "shard_sha256": {
            "shard_00000.pt": sha256_file(root / "shard_00000.pt"),
            "shard_00001.pt": sha256_file(root / "shard_00001.pt"),
        },
        "stage": "evaluation",
    }
    assert [
        (record["image_id"], record["severity"], record["value"].item())
        for record in iter_records(root)
    ] == [("a", 0, 1), ("a", 1, 2), ("b", 0, 3)]
    assert sorted(path.name for path in root.iterdir()) == [
        ".artifact_directory_anchor",
        "manifest.json",
        "shard_00000.pt",
        "shard_00001.pt",
    ]


def test_partial_shards_resume_from_published_record_keys(tmp_path):
    root = tmp_path / "cache"
    metadata = {"stage": "reference", "input_id": "abc"}
    writer = ShardWriter(root, metadata, shard_size=1)
    writer.add(_record("a"))

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


def test_same_process_handoff_invalidates_the_predecessor(tmp_path):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    writer = ShardWriter(root, metadata, shard_size=1)
    writer.add(_record("a"))

    resumed = ShardWriter(root, metadata, shard_size=1)

    with pytest.raises(RuntimeError, match="superseded"):
        writer.add(_record("stale-add"))
    with pytest.raises(RuntimeError, match="superseded"):
        writer._flush()
    with pytest.raises(RuntimeError, match="superseded"):
        writer._publish_partial()
    writer.close()
    assert not (root / "manifest.json").exists()

    resumed.add(_record("b"))
    resumed.close()
    assert [record["image_id"] for record in iter_records(root)] == ["a", "b"]


def test_handoff_refuses_unpublished_buffer_without_invalidating_writer(tmp_path):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    writer = ShardWriter(root, metadata, shard_size=2)
    writer.add(_record("a"))
    partial_before = (root / "partial_manifest.json").read_bytes()

    with pytest.raises(RuntimeError, match="unpublished buffer"):
        ShardWriter(root, metadata, shard_size=2)

    assert (root / "partial_manifest.json").read_bytes() == partial_before
    writer.add(_record("b"))
    resumed = ShardWriter(root, metadata, shard_size=2)
    resumed.close()
    assert [record["image_id"] for record in iter_records(root)] == ["a", "b"]


def test_failed_handoff_metadata_check_leaves_predecessor_active(tmp_path):
    root = tmp_path / "cache"
    metadata = {"stage": "reference", "input_id": "abc"}
    writer = ShardWriter(root, metadata, shard_size=1)
    writer.add(_record("a"))

    with pytest.raises(ValueError, match="input_id"):
        ShardWriter(
            root,
            {"stage": "reference", "input_id": "different"},
            shard_size=1,
        )

    writer.add(_record("b"))
    writer.close()
    assert [record["image_id"] for record in iter_records(root)] == ["a", "b"]


def test_concurrent_handoff_and_close_do_not_deadlock_or_publish_twice(tmp_path):
    for index in range(20):
        root = tmp_path / f"cache-{index}"
        metadata = {"stage": "evaluation"}
        writer = ShardWriter(root, metadata, shard_size=1)
        writer.add(_record("a"))
        barrier = threading.Barrier(2)
        successors = []
        errors = []

        def close_predecessor():
            try:
                barrier.wait()
                writer.close()
            except BaseException as error:
                errors.append(error)

        def attempt_handoff():
            try:
                barrier.wait()
                successors.append(
                    ShardWriter(root, metadata, shard_size=1)
                )
            except FileExistsError:
                pass
            except BaseException as error:
                errors.append(error)

        threads = [
            threading.Thread(target=close_predecessor),
            threading.Thread(target=attempt_handoff),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        assert not any(thread.is_alive() for thread in threads)
        assert errors == []
        for successor in successors:
            successor.close()
        assert [record["image_id"] for record in iter_records(root)] == ["a"]


def test_post_transfer_constructor_failure_releases_ownership(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    writer = ShardWriter(root, metadata, shard_size=1)
    writer.add(_record("a"))
    original_resume = ShardWriter._resume

    def fail_resume(_self, _state):
        raise RuntimeError("resume failed")

    monkeypatch.setattr(ShardWriter, "_resume", fail_resume)
    with pytest.raises(RuntimeError, match="resume failed"):
        ShardWriter(root, metadata, shard_size=1)
    monkeypatch.setattr(ShardWriter, "_resume", original_resume)

    with pytest.raises(RuntimeError, match="superseded"):
        writer.add(_record("stale"))

    resumed = ShardWriter(root, metadata, shard_size=1)
    resumed.close()
    assert [record["image_id"] for record in iter_records(root)] == ["a"]


def test_handoff_uses_directory_identity_across_symlink_aliases(tmp_path):
    root = tmp_path / "cache"
    alias = tmp_path / "cache-alias"
    metadata = {"stage": "evaluation"}
    writer = ShardWriter(root, metadata, shard_size=1)
    writer.add(_record("a"))
    alias.symlink_to(root, target_is_directory=True)

    resumed = ShardWriter(alias, metadata, shard_size=1)

    with pytest.raises(RuntimeError, match="superseded"):
        writer.add(_record("stale"))
    resumed.close()

    assert [record["image_id"] for record in iter_records(root)] == ["a"]


def test_writer_io_stays_bound_to_the_locked_directory_after_rename(tmp_path):
    root = tmp_path / "cache"
    moved = tmp_path / "moved-cache"
    metadata = {"stage": "evaluation"}
    writer = ShardWriter(root, metadata, shard_size=1)
    root.rename(moved)
    replacement = ShardWriter(root, metadata, shard_size=1)

    writer.add(_record("old-directory"))
    replacement.add(_record("new-directory"))
    writer.close()
    replacement.close()

    assert [record["image_id"] for record in iter_records(moved)] == [
        "old-directory"
    ]
    assert [record["image_id"] for record in iter_records(root)] == [
        "new-directory"
    ]


def test_writer_identity_is_derived_from_the_same_directory_fd_it_locks(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    moved = tmp_path / "moved-cache"
    metadata = {"stage": "evaluation"}
    original_acquire = artifacts._acquire_directory_lock
    replaced = False

    def replace_path_before_acquire(directory, **kwargs):
        nonlocal replaced
        if not replaced:
            Path(directory).rename(moved)
            Path(directory).mkdir()
            replaced = True
        return original_acquire(directory, **kwargs)

    monkeypatch.setattr(
        artifacts,
        "_acquire_directory_lock",
        replace_path_before_acquire,
    )
    writer = ShardWriter(root, metadata, shard_size=1)
    writer.add(_record("a"))
    monkeypatch.setattr(
        artifacts,
        "_acquire_directory_lock",
        original_acquire,
    )

    resumed = ShardWriter(moved, metadata, shard_size=1)
    resumed.add(_record("b"))
    resumed.close()
    replacement = ShardWriter(root, metadata, shard_size=1)
    replacement.add(_record("replacement"))
    replacement.close()

    assert [record["image_id"] for record in iter_records(moved)] == ["a", "b"]
    assert [record["image_id"] for record in iter_records(root)] == [
        "replacement"
    ]


def test_directory_open_is_serialized_with_the_fork_registry(tmp_path, monkeypatch):
    class TrackingRLock:
        def __init__(self):
            self._lock = threading.RLock()
            self._local = threading.local()

        def acquire(self, *args, **kwargs):
            acquired = self._lock.acquire(*args, **kwargs)
            if acquired:
                self._local.depth = getattr(self._local, "depth", 0) + 1
            return acquired

        def release(self):
            self._local.depth -= 1
            self._lock.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, *_exc):
            self.release()

        def held_by_current_thread(self):
            return getattr(self._local, "depth", 0) > 0

    tracking_lock = TrackingRLock()
    original_open_directory = artifacts._open_directory
    observed = []

    def observe_registry_boundary(directory, **kwargs):
        observed.append(tracking_lock.held_by_current_thread())
        return original_open_directory(directory, **kwargs)

    monkeypatch.setattr(
        artifacts,
        "_WRITER_REGISTRY_LOCK",
        tracking_lock,
    )
    monkeypatch.setattr(
        artifacts,
        "_open_directory",
        observe_registry_boundary,
    )
    writer = ShardWriter(
        tmp_path / "cache",
        {"stage": "evaluation"},
        shard_size=1,
    )
    writer.close()

    assert observed == [True]


def test_lock_helper_return_exception_releases_caller_owned_fd(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    original_acquire = artifacts._acquire_directory_lock

    def acquire_then_interrupt(directory, **kwargs):
        original_acquire(directory, **kwargs)
        raise KeyboardInterrupt("interrupted after helper return")

    monkeypatch.setattr(
        artifacts,
        "_acquire_directory_lock",
        acquire_then_interrupt,
    )
    try:
        with pytest.raises(KeyboardInterrupt, match="after helper return"):
            ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        assert artifacts._PENDING_LOCK_FDS == set()
        monkeypatch.setattr(
            artifacts,
            "_acquire_directory_lock",
            original_acquire,
        )
        recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        recovered.close()
    finally:
        monkeypatch.setattr(
            artifacts,
            "_acquire_directory_lock",
            original_acquire,
        )
        for descriptor in list(artifacts._PENDING_LOCK_FDS):
            artifacts.fcntl.flock(descriptor, artifacts.fcntl.LOCK_UN)
            os.close(descriptor)
            artifacts._PENDING_LOCK_FDS.discard(descriptor)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_candidate_opened_after_fork_is_cleaned_by_the_child(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    original_open = artifacts._open_directory
    child_process_id = None
    in_child = False
    release_read, release_write = os.pipe()

    def fork_before_open(directory, **kwargs):
        nonlocal child_process_id, in_child
        process_id = os.fork()
        if process_id == 0:
            in_child = True
            child_process_id = 0
            os.close(release_read)
            return original_open(directory, **kwargs)
        child_process_id = process_id
        os.close(release_write)
        os.read(release_read, 1)
        os.close(release_read)
        return original_open(directory, **kwargs)

    monkeypatch.setattr(artifacts, "_open_directory", fork_before_open)
    try:
        try:
            writer = ShardWriter(root, metadata, shard_size=1)
        except RuntimeError as error:
            if not in_child:
                raise
            monkeypatch.setattr(artifacts, "_open_directory", original_open)
            pending_was_cleared = artifacts._PENDING_LOCK_FDS == set()
            try:
                recovered = ShardWriter(root, metadata, shard_size=1)
            except BaseException:
                recovered_immediately = False
            else:
                recovered_immediately = True
                recovered.__exit__(RuntimeError, None, None)
            os.write(release_write, b"1")
            os.close(release_write)
            os._exit(
                0
                if "fork" in str(error)
                and pending_was_cleared
                and recovered_immediately
                else 50
            )

        if in_child:
            os.write(release_write, b"1")
            os.close(release_write)
            os._exit(51)

        assert child_process_id is not None
        _waited_id, status = os.waitpid(child_process_id, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        writer.close()
    finally:
        if in_child:
            try:
                os.write(release_write, b"1")
            except OSError:
                pass
            try:
                os.close(release_write)
            except OSError:
                pass
            os._exit(52)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_after_candidate_registration_cannot_resurrect_pending_fd(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    child_process_id = None
    in_child = False
    release_read, release_write = os.pipe()

    class ForkAfterRegistration(set):
        def add(self, lease):
            nonlocal child_process_id, in_child
            super().add(lease)
            process_id = os.fork()
            if process_id == 0:
                in_child = True
                child_process_id = 0
                os.close(release_read)
                return
            child_process_id = process_id
            os.close(release_write)
            os.read(release_read, 1)
            os.close(release_read)

    monkeypatch.setattr(
        artifacts,
        "_PENDING_LOCK_LEASES",
        ForkAfterRegistration(),
    )
    try:
        try:
            writer = ShardWriter(root, metadata, shard_size=1)
        except RuntimeError as error:
            if not in_child:
                raise
            pending_was_cleared = artifacts._PENDING_LOCK_FDS == set()
            try:
                recovered = ShardWriter(root, metadata, shard_size=1)
            except BaseException:
                recovered_immediately = False
            else:
                recovered_immediately = True
                recovered.__exit__(RuntimeError, None, None)
            os.write(release_write, b"1")
            os.close(release_write)
            os._exit(
                0
                if "fork" in str(error)
                and pending_was_cleared
                and recovered_immediately
                else 60
            )

        if in_child:
            os.write(release_write, b"1")
            os.close(release_write)
            os._exit(61)

        assert child_process_id is not None
        _waited_id, status = os.waitpid(child_process_id, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        writer.close()
    finally:
        if in_child:
            try:
                os.write(release_write, b"1")
            except OSError:
                pass
            try:
                os.close(release_write)
            except OSError:
                pass
            os._exit(62)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_parent_continues_when_fork_precedes_candidate_registration(
    tmp_path,
):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    source_lines, first_line = inspect.getsourcelines(
        artifacts._CandidateLease._adopt_open_descriptor
    )
    registration_line = first_line + next(
        offset
        for offset, line in enumerate(source_lines)
        if line.strip() == "_PENDING_LOCK_LEASES.add(self)"
    )
    child_process_id = None
    in_child = False
    forked = False

    def fork_before_registration(frame, event, _argument):
        nonlocal child_process_id, in_child, forked
        if (
            not forked
            and frame.f_code
            is artifacts._CandidateLease._adopt_open_descriptor.__code__
            and event == "line"
            and frame.f_lineno == registration_line
        ):
            forked = True
            process_id = os.fork()
            if process_id == 0:
                in_child = True
                child_process_id = 0
            else:
                child_process_id = process_id
        return fork_before_registration

    writer = None
    parent_error = None
    try:
        sys.settrace(fork_before_registration)
        try:
            writer = ShardWriter(root, metadata, shard_size=1)
        except RuntimeError as error:
            if in_child:
                pending_was_cleared = (
                    artifacts._PENDING_LOCK_FDS == set()
                    and artifacts._PENDING_LOCK_LEASES == set()
                )
                os._exit(
                    0
                    if "fork" in str(error) and pending_was_cleared
                    else 70
                )
            parent_error = error
        finally:
            sys.settrace(None)

        if in_child:
            os._exit(71)

        assert child_process_id is not None
        _waited_id, status = os.waitpid(child_process_id, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        assert parent_error is None
        assert writer is not None
        writer.close()
    finally:
        sys.settrace(None)
        if in_child:
            os._exit(72)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_shared_raw_descriptor_uses_one_marker_across_fork(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    root.mkdir()
    metadata = {"stage": "evaluation"}
    original_scandir = artifacts.os.scandir
    original_mark = artifacts._mark_lock_descriptor
    child_process_id = None
    in_child = False
    forked = False
    marked = False
    raw_descriptor = None

    def open_then_fork(path):
        nonlocal child_process_id, forked, in_child, raw_descriptor
        before = artifacts._live_fd_snapshot()
        owner = original_scandir(path)
        if Path(path) != root or forked:
            return owner
        candidates = artifacts._live_fd_snapshot() - before
        assert len(candidates) == 1
        forked = True
        raw_descriptor = candidates.pop()
        process_id = os.fork()
        if process_id == 0:
            in_child = True
            child_process_id = 0
        else:
            child_process_id = process_id
        return owner

    def apply_marker(descriptor, marker):
        if marker is None:
            return original_mark(descriptor)
        return original_mark(descriptor, marker)

    def ordered_marker(descriptor, marker=None):
        nonlocal marked
        result = apply_marker(descriptor, marker)
        marked = True
        return result

    monkeypatch.setattr(artifacts.os, "scandir", open_then_fork)
    monkeypatch.setattr(artifacts, "_mark_lock_descriptor", ordered_marker)
    writer = None
    parent_error = None
    try:
        try:
            writer = ShardWriter(root, metadata, shard_size=1)
        except RuntimeError as error:
            if in_child:
                clean_failure = (
                    "fork" in str(error)
                    and artifacts._PENDING_LOCK_FDS == set()
                    and artifacts._PENDING_LOCK_LEASES == set()
                )
                descriptor_was_closed = False
                try:
                    os.fstat(raw_descriptor)
                except OSError as close_error:
                    descriptor_was_closed = (
                        close_error.errno == artifacts.errno.EBADF
                    )
                os._exit(
                    0 if clean_failure and descriptor_was_closed else 80
                )
            parent_error = error

        if in_child:
            os._exit(81)

        assert child_process_id is not None
        _waited_id, status = os.waitpid(child_process_id, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        monkeypatch.setattr(artifacts.os, "scandir", original_scandir)
        monkeypatch.setattr(
            artifacts,
            "_mark_lock_descriptor",
            original_mark,
        )

        assert parent_error is None
        assert writer is not None
        writer.__exit__(RuntimeError, None, None)
        writer = None
        assert raw_descriptor is not None
        with pytest.raises(OSError) as error:
            os.fstat(raw_descriptor)
        assert error.value.errno == artifacts.errno.EBADF
        recovered = ShardWriter(root, metadata, shard_size=1)
        recovered.close()
    finally:
        monkeypatch.setattr(artifacts.os, "scandir", original_scandir)
        monkeypatch.setattr(
            artifacts,
            "_mark_lock_descriptor",
            original_mark,
        )
        if in_child:
            os._exit(82)
        if writer is not None:
            writer.__exit__(RuntimeError, None, None)
        if raw_descriptor is not None:
            try:
                os.close(raw_descriptor)
            except OSError:
                pass


def test_candidate_open_call_return_is_exception_safe(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    original_scandir = artifacts.os.scandir
    opened_descriptor = None
    interrupted = False
    target_line = next(
        instruction.positions.lineno
        for instruction in dis.get_instructions(artifacts._DirectoryHandle.open)
        if instruction.opname == "STORE_ATTR"
        and instruction.argval == "_owner"
    )

    def capture_open(path):
        nonlocal opened_descriptor
        before = artifacts._live_fd_snapshot()
        owner = original_scandir(path)
        if Path(path) == root:
            candidates = artifacts._live_fd_snapshot() - before
            assert len(candidates) == 1
            opened_descriptor = candidates.pop()
        return owner

    def interrupt_after_open(frame, event, _argument):
        nonlocal interrupted
        if frame.f_code is artifacts._DirectoryHandle.open.__code__:
            if (
                not interrupted
                and event == "line"
                and frame.f_lineno == target_line
            ):
                interrupted = True
                raise KeyboardInterrupt("interrupted after directory open")
        return interrupt_after_open

    monkeypatch.setattr(artifacts.os, "scandir", capture_open)
    try:
        sys.settrace(interrupt_after_open)
        with pytest.raises(KeyboardInterrupt, match="after directory open"):
            ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        sys.settrace(None)

        assert interrupted
        assert opened_descriptor is not None
        with pytest.raises(OSError) as error:
            os.fstat(opened_descriptor)
        assert error.value.errno == artifacts.errno.EBADF
        assert artifacts._PENDING_LOCK_FDS == set()
        assert artifacts._PENDING_LOCK_LEASES == set()
        recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        recovered.close()
    finally:
        sys.settrace(None)
        if opened_descriptor is not None:
            try:
                os.close(opened_descriptor)
            except OSError:
                pass


def test_directory_acquisition_avoids_unowned_fwalk_open_result(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"

    def reject_fwalk(*_args, **_kwargs):
        raise AssertionError("directory acquisition must not depend on fwalk")

    monkeypatch.setattr(artifacts.os, "fwalk", reject_fwalk)
    writer = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
    writer.__exit__(RuntimeError, None, None)

    recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
    recovered.close()


def test_directory_acquisition_rejects_concurrent_position_changes_cleanly(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    root.mkdir()
    (root / ".artifact_directory_anchor").mkdir()
    decoy = os.scandir(root)
    original_positions = artifacts._directory_fd_positions
    snapshot_calls = 0

    def advance_decoy_after_first_snapshot(identity):
        nonlocal snapshot_calls
        positions = original_positions(identity)
        snapshot_calls += 1
        if snapshot_calls == 1:
            assert next(decoy).name == ".artifact_directory_anchor"
        return positions

    monkeypatch.setattr(
        artifacts,
        "_directory_fd_positions",
        advance_decoy_after_first_snapshot,
    )
    try:
        with pytest.raises(RuntimeError, match="could not identify"):
            ShardWriter(root, {"stage": "evaluation"}, shard_size=1)

        assert artifacts._PENDING_LOCK_FDS == set()
        assert artifacts._PENDING_LOCK_LEASES == set()
    finally:
        monkeypatch.setattr(
            artifacts,
            "_directory_fd_positions",
            original_positions,
        )
        decoy.close()

    recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
    recovered.close()


def test_scanner_position_binding_survives_descriptor_number_aba(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    root.mkdir()
    (root / ".artifact_directory_anchor").mkdir()
    identity = (os.stat(root).st_dev, os.stat(root).st_ino)
    original_scandir = artifacts.os.scandir
    before = artifacts._live_fd_snapshot()
    stale_owner = original_scandir(root)
    candidates = artifacts._live_fd_snapshot() - before
    stale_fd = next(
        descriptor
        for descriptor in candidates
        if (
            os.fstat(descriptor).st_dev,
            os.fstat(descriptor).st_ino,
        )
        == identity
    )
    decoy_owner = None
    decoy_fd = None
    replaced = False
    writer = None

    def replace_stale_number_and_open_decoy(path):
        nonlocal decoy_fd, decoy_owner, replaced
        if Path(path) != root or replaced:
            return original_scandir(path)
        replaced = True
        stale_owner.close()
        owner = original_scandir(path)
        assert (
            os.fstat(stale_fd).st_dev,
            os.fstat(stale_fd).st_ino,
        ) == identity
        target_snapshot = artifacts._live_fd_snapshot()
        decoy_owner = original_scandir(path)
        new_descriptors = artifacts._live_fd_snapshot() - target_snapshot
        decoy_fd = next(
            descriptor
            for descriptor in new_descriptors
            if (
                os.fstat(descriptor).st_dev,
                os.fstat(descriptor).st_ino,
            )
            == identity
        )
        return owner

    monkeypatch.setattr(
        artifacts.os,
        "scandir",
        replace_stale_number_and_open_decoy,
    )
    try:
        writer = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        assert writer._lock_fd == stale_fd
        assert decoy_fd is not None and decoy_fd != stale_fd
        writer.__exit__(RuntimeError, None, None)
        writer = None
        os.fstat(decoy_fd)
        decoy_owner.close()
        decoy_owner = None

        recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        recovered.close()
    finally:
        monkeypatch.setattr(artifacts.os, "scandir", original_scandir)
        if writer is not None:
            writer.__exit__(RuntimeError, None, None)
        stale_owner.close()
        if decoy_owner is not None:
            decoy_owner.close()


@pytest.mark.parametrize("populated", [False, True])
def test_scanner_position_binding_supports_fresh_directories(
    tmp_path,
    populated,
):
    root = tmp_path / "cache"
    if populated:
        root.mkdir()
        (root / "existing").write_text("value", encoding="utf-8")

    writer = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)

    assert (root / ".artifact_directory_anchor").exists()
    assert writer._lock_fd is not None
    os.fstat(writer._lock_fd)
    writer.close()


@pytest.mark.parametrize("anchor_kind", ["file", "directory"])
def test_existing_anchor_entry_is_accepted(tmp_path, anchor_kind):
    root = tmp_path / "cache"
    root.mkdir()
    anchor = root / ".artifact_directory_anchor"
    if anchor_kind == "file":
        anchor.write_text("preexisting", encoding="utf-8")
    else:
        anchor.mkdir()

    writer = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
    writer.close()


def test_concurrent_anchor_creation_is_accepted(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    original_mkdir = artifacts.os.mkdir
    created_concurrently = False

    def create_then_report_exists(path, mode=0o777, *, dir_fd=None):
        nonlocal created_concurrently
        if Path(path).name == ".artifact_directory_anchor":
            assert not created_concurrently
            created_concurrently = True
            original_mkdir(path, mode, dir_fd=dir_fd)
            raise FileExistsError(artifacts.errno.EEXIST, "already exists", path)
        return original_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(artifacts.os, "mkdir", create_then_report_exists)
    writer = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)

    assert created_concurrently
    writer.close()


def test_empty_path_replacement_between_anchor_and_scanner_retries(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    moved = tmp_path / "moved"
    original_scandir = artifacts.os.scandir
    replaced = False

    def replace_with_empty_directory(path):
        nonlocal replaced
        if Path(path) == root and not replaced:
            replaced = True
            root.rename(moved)
            root.mkdir()
        return original_scandir(path)

    monkeypatch.setattr(
        artifacts.os,
        "scandir",
        replace_with_empty_directory,
    )
    writer = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
    writer.add(_record("a"))
    writer.close()

    assert replaced
    assert (root / "manifest.json").exists()
    assert not (moved / "manifest.json").exists()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_before_candidate_handle_assignment_performs_no_child_io(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    original_scandir = artifacts.os.scandir
    source_lines, first_line = inspect.getsourcelines(
        artifacts._CandidateLease.open
    )
    assignment_line = first_line + next(
        offset
        for offset, line in enumerate(source_lines)
        if line.strip() == "self._directory_handle = _DirectoryHandle(directory)"
    )
    forked = False
    in_child = False
    child_process_id = None
    child_scanned = False

    def record_child_scan(path):
        nonlocal child_scanned
        if in_child and Path(path) == root:
            child_scanned = True
        return original_scandir(path)

    def fork_before_assignment(frame, event, _argument):
        nonlocal child_process_id, forked, in_child
        if (
            not forked
            and frame.f_code is artifacts._CandidateLease.open.__code__
            and event == "line"
            and frame.f_lineno == assignment_line
        ):
            forked = True
            process_id = os.fork()
            if process_id == 0:
                in_child = True
                child_process_id = 0
            else:
                child_process_id = process_id
        return fork_before_assignment

    monkeypatch.setattr(artifacts.os, "scandir", record_child_scan)
    try:
        sys.settrace(fork_before_assignment)
        try:
            writer = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        except RuntimeError as error:
            if not in_child:
                raise
            sys.settrace(None)
            clean_failure = "fork" in str(error)
            os._exit(0 if clean_failure and not child_scanned else 61)

        sys.settrace(None)
        if in_child:
            os._exit(62)
        assert child_process_id is not None
        _waited_id, status = os.waitpid(child_process_id, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        writer.close()
    finally:
        sys.settrace(None)
        if in_child:
            os._exit(63)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_after_position_binding_does_not_mutate_a_reused_fd(tmp_path):
    root = tmp_path / "cache"
    source_lines, first_line = inspect.getsourcelines(
        artifacts._DirectoryHandle.open
    )
    candidate_line = first_line + next(
        offset
        for offset, line in enumerate(source_lines)
        if line.strip() == "candidates = ["
    )
    forked = False
    in_child = False
    child_process_id = None
    reused_descriptor = None
    child_exit_code = None
    original_position = None

    def fork_before_candidate_assignment(frame, event, _argument):
        nonlocal child_exit_code, child_process_id, forked, in_child
        nonlocal original_position, reused_descriptor
        if (
            not forked
            and frame.f_code is artifacts._DirectoryHandle.open.__code__
            and event == "line"
            and frame.f_lineno == candidate_line
        ):
            before = frame.f_locals["before"]
            after = frame.f_locals["after"]
            changed = [
                descriptor
                for descriptor, position in before.items()
                if descriptor in after and after[descriptor] != position
            ]
            assert len(changed) == 1
            forked = True
            process_id = os.fork()
            if process_id == 0:
                in_child = True
                child_process_id = 0
                flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                while reused_descriptor != changed[0]:
                    reused_descriptor = os.open(root, flags)
                    if reused_descriptor > changed[0]:
                        os._exit(70)
                original_position = os.lseek(
                    reused_descriptor,
                    0,
                    os.SEEK_CUR,
                )
            else:
                child_process_id = process_id
                _waited_id, status = os.waitpid(process_id, 0)
                child_exit_code = os.waitstatus_to_exitcode(status)
        return fork_before_candidate_assignment

    try:
        sys.settrace(fork_before_candidate_assignment)
        try:
            writer = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        except BaseException as error:
            if not in_child:
                raise
            sys.settrace(None)
            if not isinstance(error, RuntimeError):
                os._exit(74)
            alive = True
            try:
                os.fstat(reused_descriptor)
            except OSError:
                alive = False
            unchanged = (
                alive
                and os.lseek(reused_descriptor, 0, os.SEEK_CUR)
                == original_position
            )
            probe = os.open(
                root,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            unlocked = True
            try:
                artifacts.fcntl.flock(
                    probe,
                    artifacts.fcntl.LOCK_EX | artifacts.fcntl.LOCK_NB,
                )
            except BlockingIOError:
                unlocked = False
            finally:
                os.close(probe)
            clean = (
                artifacts._PENDING_LOCK_FDS == set()
                and artifacts._PENDING_LOCK_LEASES == set()
            )
            if "fork" not in str(error):
                os._exit(75)
            if not alive:
                os._exit(76)
            if not unchanged:
                os._exit(77)
            if not unlocked:
                os._exit(78)
            if not clean:
                os._exit(79)
            os._exit(0)

        sys.settrace(None)
        if in_child:
            os._exit(72)
        assert child_process_id is not None
        assert child_exit_code == 0
        writer.close()
    finally:
        sys.settrace(None)
        if in_child:
            os._exit(73)


def test_constructor_helper_return_is_exception_safe(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    original_scandir = artifacts.os.scandir
    opened_descriptor = None
    interrupted = False
    target_line = next(
        instruction.positions.lineno
        for instruction in dis.get_instructions(ShardWriter.__init__)
        if instruction.opname == "LOAD_GLOBAL"
        and instruction.argval == "_acquire_directory_lock"
    )

    def capture_open(path):
        nonlocal opened_descriptor
        before = artifacts._live_fd_snapshot()
        owner = original_scandir(path)
        if Path(path) == root:
            candidates = artifacts._live_fd_snapshot() - before
            assert len(candidates) == 1
            opened_descriptor = candidates.pop()
        return owner

    def interrupt_helper_return(frame, event, _argument):
        nonlocal interrupted
        if frame.f_code is ShardWriter.__init__.__code__:
            if (
                not interrupted
                and event == "line"
                and frame.f_lineno == target_line
            ):
                interrupted = True
                raise KeyboardInterrupt("interrupted after directory helper")
        return interrupt_helper_return

    monkeypatch.setattr(artifacts.os, "scandir", capture_open)
    try:
        sys.settrace(interrupt_helper_return)
        with pytest.raises(KeyboardInterrupt, match="after directory helper"):
            ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        sys.settrace(None)

        assert interrupted
        assert opened_descriptor is not None
        with pytest.raises(OSError) as error:
            os.fstat(opened_descriptor)
        assert error.value.errno == artifacts.errno.EBADF
        assert artifacts._PENDING_LOCK_FDS == set()
        assert artifacts._PENDING_LOCK_LEASES == set()
        recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        recovered.close()
    finally:
        sys.settrace(None)
        if opened_descriptor is not None:
            try:
                os.close(opened_descriptor)
            except OSError:
                pass


def test_acquisition_failure_never_recloses_a_reused_candidate_fd(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    original_flock = artifacts.fcntl.flock
    original_close = artifacts.os.close
    original_handle_close = artifacts._DirectoryHandle.close
    candidate_descriptor = None
    replacement_descriptor = None
    close_calls = 0

    def reject_lock(descriptor, operation):
        nonlocal candidate_descriptor
        if operation & artifacts.fcntl.LOCK_EX:
            candidate_descriptor = descriptor
            raise BlockingIOError(artifacts.errno.EAGAIN, "forced contention")
        return original_flock(descriptor, operation)

    def close_then_reuse(handle):
        nonlocal close_calls, replacement_descriptor
        descriptor = handle._descriptor
        if descriptor != candidate_descriptor or handle._owner is None:
            return original_handle_close(handle)
        close_calls += 1
        if close_calls == 1:
            original_handle_close(handle)

            def reuse_in_thread():
                nonlocal replacement_descriptor
                replacement_descriptor = os.open(os.devnull, os.O_RDONLY)

            thread = threading.Thread(target=reuse_in_thread)
            thread.start()
            thread.join()
            assert replacement_descriptor == descriptor
            return
        return original_handle_close(handle)

    monkeypatch.setattr(artifacts.fcntl, "flock", reject_lock)
    monkeypatch.setattr(artifacts._DirectoryHandle, "close", close_then_reuse)
    try:
        with pytest.raises(RuntimeError, match="active writer"):
            ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        assert close_calls == 1
        assert replacement_descriptor is not None
        os.fstat(replacement_descriptor)
    finally:
        monkeypatch.setattr(artifacts.fcntl, "flock", original_flock)
        monkeypatch.setattr(
            artifacts._DirectoryHandle,
            "close",
            original_handle_close,
        )
        if replacement_descriptor is not None:
            try:
                original_close(replacement_descriptor)
            except OSError:
                pass


def test_candidate_initialization_interruption_never_leaks_raw_descriptor(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    original_scandir = artifacts.os.scandir
    opened_descriptor = None
    interrupted = False
    source_lines, first_line = inspect.getsourcelines(
        artifacts._CandidateLease._adopt_open_descriptor
    )
    target_line = first_line + next(
        offset
        for offset, line in enumerate(source_lines)
        if line.strip() == "_PENDING_LOCK_FDS.add(descriptor)"
    )

    def capture_open(path):
        nonlocal opened_descriptor
        before = artifacts._live_fd_snapshot()
        owner = original_scandir(path)
        if Path(path) == root:
            candidates = artifacts._live_fd_snapshot() - before
            assert len(candidates) == 1
            opened_descriptor = candidates.pop()
        return owner

    def interrupt_during_initialization(frame, event, _argument):
        nonlocal interrupted
        if (
            not interrupted
            and frame.f_code
            is artifacts._CandidateLease._adopt_open_descriptor.__code__
            and event == "line"
            and frame.f_lineno == target_line
        ):
            interrupted = True
            raise KeyboardInterrupt("interrupted during candidate initialization")
        return interrupt_during_initialization

    monkeypatch.setattr(artifacts.os, "scandir", capture_open)
    try:
        sys.settrace(interrupt_during_initialization)
        with pytest.raises(KeyboardInterrupt, match="candidate initialization"):
            ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        sys.settrace(None)

        assert interrupted
        assert opened_descriptor is not None
        with pytest.raises(OSError) as error:
            os.fstat(opened_descriptor)
        assert error.value.errno == artifacts.errno.EBADF
        assert artifacts._PENDING_LOCK_FDS == set()
        recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        recovered.close()
    finally:
        sys.settrace(None)
        if opened_descriptor is not None:
            try:
                os.close(opened_descriptor)
            except OSError:
                pass


def test_candidate_open_adoption_recovers_from_every_line_boundary(
    tmp_path,
    monkeypatch,
):
    probe_root = tmp_path / "candidate-adoption-probe"
    probe_root.mkdir()
    probe = artifacts._CandidateLease()
    executed_lines = []
    adoption_codes = {
        artifacts._CandidateLease.open.__code__,
        artifacts._CandidateLease._track_open_descriptor.__code__,
        artifacts._CandidateLease._prepare_open_descriptor.__code__,
        artifacts._CandidateLease._adopt_open_descriptor.__code__,
    }

    def record_adoption_lines(frame, event, _argument):
        if frame.f_code in adoption_codes and event == "line":
            executed_lines.append((frame.f_code, frame.f_lineno))
        return record_adoption_lines

    sys.settrace(record_adoption_lines)
    try:
        probe.open(probe_root)
    finally:
        sys.settrace(None)
        probe.close()
    boundaries = list(dict.fromkeys(executed_lines))
    assert boundaries

    original_scandir = artifacts.os.scandir
    original_close = artifacts.os.close
    for index, (target_code, target_line) in enumerate(boundaries):
        root = tmp_path / f"candidate-adoption-boundary-{index}"
        root.mkdir()
        lease = artifacts._CandidateLease()
        opened_descriptor = None
        interrupted = False

        def capture_open(path):
            nonlocal opened_descriptor
            before = artifacts._live_fd_snapshot()
            owner = original_scandir(path)
            if Path(path) == root:
                candidates = artifacts._live_fd_snapshot() - before
                assert len(candidates) == 1
                opened_descriptor = candidates.pop()
            return owner

        def interrupt_at_boundary(frame, event, _argument):
            nonlocal interrupted
            if (
                not interrupted
                and frame.f_code is target_code
                and event == "line"
                and frame.f_lineno == target_line
            ):
                interrupted = True
                raise KeyboardInterrupt(f"interrupted at line {target_line}")
            return interrupt_at_boundary

        monkeypatch.setattr(artifacts.os, "scandir", capture_open)
        try:
            sys.settrace(interrupt_at_boundary)
            with pytest.raises(KeyboardInterrupt, match="interrupted at line"):
                lease.open(root)
            sys.settrace(None)
            monkeypatch.setattr(artifacts.os, "scandir", original_scandir)

            assert interrupted
            assert artifacts._PENDING_LOCK_FDS == set(), (
                target_code.co_name,
                target_line,
            )
            assert artifacts._PENDING_LOCK_LEASES == set()
            if opened_descriptor is not None:
                try:
                    os.fstat(opened_descriptor)
                except OSError as error:
                    assert error.errno == artifacts.errno.EBADF
                else:
                    pytest.fail((target_code.co_name, target_line))
            recovered = ShardWriter(
                root,
                {"stage": "evaluation"},
                shard_size=1,
            )
            recovered.close()
        finally:
            sys.settrace(None)
            monkeypatch.setattr(artifacts.os, "scandir", original_scandir)
            lease._descriptor = None
            artifacts._PENDING_LOCK_LEASES.discard(lease)
            if opened_descriptor is not None:
                artifacts._PENDING_LOCK_FDS.discard(opened_descriptor)
                try:
                    original_close(opened_descriptor)
                except OSError:
                    pass


def test_open_cleanup_interruption_never_loses_unadopted_descriptor(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    original_scandir = artifacts.os.scandir
    original_prepare = artifacts._CandidateLease._prepare_open_descriptor
    opened_descriptor = None
    interrupted = False
    source_lines, first_line = inspect.getsourcelines(
        artifacts._CandidateLease.open
    )
    cleanup_line = first_line + max(
        offset
        for offset, line in enumerate(source_lines)
        if line.strip() == "self.close()"
    )

    def capture_open(path):
        nonlocal opened_descriptor
        before = artifacts._live_fd_snapshot()
        owner = original_scandir(path)
        if Path(path) == root:
            candidates = artifacts._live_fd_snapshot() - before
            assert len(candidates) == 1
            opened_descriptor = candidates.pop()
        return owner

    def reject_preparation(_lease, _descriptor):
        raise RuntimeError("forced preparation failure")

    def interrupt_cleanup(frame, event, _argument):
        nonlocal interrupted
        if (
            not interrupted
            and frame.f_code is artifacts._CandidateLease.open.__code__
            and event == "line"
            and frame.f_lineno == cleanup_line
        ):
            interrupted = True
            raise KeyboardInterrupt("interrupted during open cleanup")
        return interrupt_cleanup

    monkeypatch.setattr(artifacts.os, "scandir", capture_open)
    monkeypatch.setattr(
        artifacts._CandidateLease,
        "_prepare_open_descriptor",
        reject_preparation,
    )
    try:
        sys.settrace(interrupt_cleanup)
        with pytest.raises(KeyboardInterrupt, match="open cleanup"):
            ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        sys.settrace(None)
        monkeypatch.setattr(
            artifacts._CandidateLease,
            "_prepare_open_descriptor",
            original_prepare,
        )

        assert interrupted
        assert opened_descriptor is not None
        with pytest.raises(OSError) as error:
            os.fstat(opened_descriptor)
        assert error.value.errno == artifacts.errno.EBADF
        assert artifacts._PENDING_LOCK_FDS == set()
        assert artifacts._PENDING_LOCK_LEASES == set()
        recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        recovered.close()
    finally:
        sys.settrace(None)
        monkeypatch.setattr(
            artifacts._CandidateLease,
            "_prepare_open_descriptor",
            original_prepare,
        )
        if opened_descriptor is not None:
            try:
                os.close(opened_descriptor)
            except OSError:
                pass


def test_raw_candidate_close_interruption_uses_ownership_marker(
    tmp_path,
):
    root = tmp_path / "cache"
    root.mkdir()
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(root, flags)
    lease = artifacts._CandidateLease()
    lease._track_open_descriptor(descriptor)
    source_lines, first_line = inspect.getsourcelines(
        artifacts._CandidateLease._close_once
    )
    close_line = first_line + next(
        offset
        for offset, line in enumerate(source_lines)
        if line.strip() == "os.closerange(descriptor, descriptor + 1)"
    )
    interrupted = False

    def interrupt_before_close(frame, event, _argument):
        nonlocal interrupted
        if (
            not interrupted
            and frame.f_code is artifacts._CandidateLease._close_once.__code__
            and event == "line"
            and frame.f_lineno == close_line
        ):
            interrupted = True
            raise KeyboardInterrupt("interrupted before raw candidate close")
        return interrupt_before_close

    try:
        sys.settrace(interrupt_before_close)
        with pytest.raises(KeyboardInterrupt, match="raw candidate close"):
            lease.close()
        sys.settrace(None)

        assert interrupted
        with pytest.raises(OSError) as error:
            os.fstat(descriptor)
        assert error.value.errno == artifacts.errno.EBADF
        assert artifacts._PENDING_LOCK_FDS == set()
        assert artifacts._PENDING_LOCK_LEASES == set()
        recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        recovered.close()
    finally:
        sys.settrace(None)
        lease._descriptor = None
        lease._opening_descriptor = None
        try:
            os.close(descriptor)
        except OSError:
            pass


def test_candidate_close_recovers_from_every_line_boundary(tmp_path):
    probe_root = tmp_path / "candidate-close-probe"
    probe_root.mkdir()
    probe, _identity = artifacts._open_directory(probe_root)
    probe = artifacts._acquire_directory_lock(probe_root, lease=probe)
    executed_lines = []
    close_codes = {
        artifacts._CandidateLease.close.__code__,
        artifacts._CandidateLease._close_safely.__code__,
        artifacts._CandidateLease._close_once.__code__,
        artifacts._CandidateLease._consume.__code__,
    }

    def record_close_lines(frame, event, _argument):
        if frame.f_code in close_codes and event == "line":
            executed_lines.append((frame.f_code, frame.f_lineno))
        return record_close_lines

    sys.settrace(record_close_lines)
    try:
        probe.close()
    finally:
        sys.settrace(None)
    boundaries = list(dict.fromkeys(executed_lines))
    assert boundaries

    original_flock = artifacts.fcntl.flock
    original_close = artifacts.os.close
    for index, (target_code, target_line) in enumerate(boundaries):
        root = tmp_path / f"candidate-close-boundary-{index}"
        root.mkdir()
        lease, _identity = artifacts._open_directory(root)
        lease = artifacts._acquire_directory_lock(root, lease=lease)
        descriptor = lease.descriptor
        interrupted = False

        def interrupt_at_boundary(frame, event, _argument):
            nonlocal interrupted
            if (
                not interrupted
                and frame.f_code is target_code
                and event == "line"
                and frame.f_lineno == target_line
            ):
                interrupted = True
                raise KeyboardInterrupt(f"interrupted at line {target_line}")
            return interrupt_at_boundary

        try:
            sys.settrace(interrupt_at_boundary)
            with pytest.raises(KeyboardInterrupt, match="interrupted at line"):
                lease.close()
            sys.settrace(None)

            assert interrupted
            assert artifacts._PENDING_LOCK_FDS == set(), (
                target_code.co_name,
                target_line,
            )
            recovered = ShardWriter(
                root,
                {"stage": "evaluation"},
                shard_size=1,
            )
            recovered.close()
        finally:
            sys.settrace(None)
            lease._descriptor = None
            artifacts._PENDING_LOCK_LEASES.discard(lease)
            artifacts._PENDING_LOCK_FDS.discard(descriptor)
            try:
                original_flock(descriptor, artifacts.fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                original_close(descriptor)
            except OSError:
                pass


def test_candidate_close_reconciliation_recovers_from_every_line_boundary(
    tmp_path,
    monkeypatch,
):
    close_codes = {
        artifacts._CandidateLease.close.__code__,
        artifacts._CandidateLease._close_safely.__code__,
        artifacts._CandidateLease._close_once.__code__,
        artifacts._CandidateLease._reconcile_close.__code__,
        artifacts._CandidateLease._consume.__code__,
    }

    def record_lines(target):
        def record(frame, event, _argument):
            if frame.f_code in close_codes and event == "line":
                target.append((frame.f_code, frame.f_lineno))
            return record

        return record

    normal_root = tmp_path / "normal-candidate-reconciliation-probe"
    normal_root.mkdir()
    normal_probe, _identity = artifacts._open_directory(normal_root)
    normal_probe = artifacts._acquire_directory_lock(
        normal_root,
        lease=normal_probe,
    )
    normal_lines = []
    sys.settrace(record_lines(normal_lines))
    try:
        normal_probe.close()
    finally:
        sys.settrace(None)

    original_handle_close = artifacts._DirectoryHandle.close
    exception_root = tmp_path / "exception-candidate-reconciliation-probe"
    exception_root.mkdir()
    exception_probe, _identity = artifacts._open_directory(exception_root)
    exception_probe = artifacts._acquire_directory_lock(
        exception_root,
        lease=exception_probe,
    )
    exception_lines = []
    close_calls = 0

    def interrupt_first_close(handle):
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise RuntimeError("enter candidate reconciliation")
        original_handle_close(handle)

    monkeypatch.setattr(
        artifacts._DirectoryHandle,
        "close",
        interrupt_first_close,
    )
    sys.settrace(record_lines(exception_lines))
    try:
        with pytest.raises(RuntimeError, match="enter candidate reconciliation"):
            exception_probe.close()
    finally:
        sys.settrace(None)
        monkeypatch.setattr(
            artifacts._DirectoryHandle,
            "close",
            original_handle_close,
        )

    normal_boundaries = set(normal_lines)
    boundaries = [
        boundary
        for boundary in dict.fromkeys(exception_lines)
        if boundary not in normal_boundaries
    ]
    assert boundaries
    assert any(
        code is artifacts._CandidateLease._reconcile_close.__code__
        for code, _line in boundaries
    )

    original_flock = artifacts.fcntl.flock
    for index, (target_code, target_line) in enumerate(boundaries):
        root = tmp_path / f"candidate-reconciliation-boundary-{index}"
        root.mkdir()
        lease, _identity = artifacts._open_directory(root)
        lease = artifacts._acquire_directory_lock(root, lease=lease)
        descriptor = lease.descriptor
        close_calls = 0
        interrupted = False

        def interrupt_at_boundary(frame, event, _argument):
            nonlocal interrupted
            if (
                not interrupted
                and frame.f_code is target_code
                and event == "line"
                and frame.f_lineno == target_line
            ):
                interrupted = True
                raise KeyboardInterrupt(f"interrupted at line {target_line}")
            return interrupt_at_boundary

        monkeypatch.setattr(
            artifacts._DirectoryHandle,
            "close",
            interrupt_first_close,
        )
        sys.settrace(interrupt_at_boundary)
        try:
            with pytest.raises(KeyboardInterrupt, match="interrupted at line"):
                lease.close()
            sys.settrace(None)
            monkeypatch.setattr(
                artifacts._DirectoryHandle,
                "close",
                original_handle_close,
            )

            assert interrupted
            assert artifacts._PENDING_LOCK_FDS == set(), (
                target_code.co_name,
                target_line,
            )
            recovered = ShardWriter(
                root,
                {"stage": "evaluation"},
                shard_size=1,
            )
            recovered.close()
        finally:
            sys.settrace(None)
            monkeypatch.setattr(
                artifacts._DirectoryHandle,
                "close",
                original_handle_close,
            )
            lease._descriptor = None
            artifacts._PENDING_LOCK_LEASES.discard(lease)
            artifacts._PENDING_LOCK_FDS.discard(descriptor)
            try:
                original_flock(descriptor, artifacts.fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(descriptor)
            except OSError:
                pass


@pytest.mark.parametrize("reuse_same_directory", [False, True])
def test_candidate_close_exception_never_recloses_a_reused_fd(
    tmp_path,
    monkeypatch,
    reuse_same_directory,
):
    root = tmp_path / "cache"
    root.mkdir()
    lease, _identity = artifacts._open_directory(root)
    lease = artifacts._acquire_directory_lock(root, lease=lease)
    candidate_descriptor = lease.descriptor
    original_handle_close = artifacts._DirectoryHandle.close
    replacement_descriptor = None
    close_calls = 0

    def close_reuse_and_interrupt(handle):
        nonlocal close_calls, replacement_descriptor
        if handle._owner is None:
            return original_handle_close(handle)
        close_calls += 1
        assert handle._descriptor == candidate_descriptor
        original_handle_close(handle)
        replacement = root if reuse_same_directory else os.devnull
        flags = os.O_RDONLY
        if reuse_same_directory:
            flags |= getattr(os, "O_DIRECTORY", 0)
        replacement_descriptor = os.open(replacement, flags)
        assert replacement_descriptor == candidate_descriptor
        raise KeyboardInterrupt("candidate close completed")

    monkeypatch.setattr(
        artifacts._DirectoryHandle,
        "close",
        close_reuse_and_interrupt,
    )
    try:
        with pytest.raises(KeyboardInterrupt, match="close completed"):
            lease.close()
        monkeypatch.setattr(
            artifacts._DirectoryHandle,
            "close",
            original_handle_close,
        )

        assert close_calls == 1
        assert artifacts._PENDING_LOCK_FDS == set()
        assert replacement_descriptor is not None
        os.fstat(replacement_descriptor)
        recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        recovered.close()
    finally:
        monkeypatch.setattr(
            artifacts._DirectoryHandle,
            "close",
            original_handle_close,
        )
        if replacement_descriptor is not None:
            try:
                os.close(replacement_descriptor)
            except OSError:
                pass


@pytest.mark.parametrize(
    "target_statement",
    ["writer._lock_fd = descriptor", "self._consume(descriptor)"],
)
def test_lock_result_transfer_is_async_exception_safe(
    tmp_path,
    target_statement,
):
    root = tmp_path / "cache"
    source_lines, first_line = inspect.getsourcelines(
        artifacts._CandidateLease.transfer_to
    )
    ownership_line = first_line + next(
        offset
        for offset, line in enumerate(source_lines)
        if line.strip() == target_statement
    )
    interrupted = False

    def interrupt_before_writer_ownership(frame, event, _argument):
        nonlocal interrupted
        if (
            frame.f_code is artifacts._CandidateLease.transfer_to.__code__
            and event == "line"
            and frame.f_lineno == ownership_line
        ):
            interrupted = True
            raise KeyboardInterrupt("interrupted before writer ownership")
        return interrupt_before_writer_ownership

    original_flock = artifacts.fcntl.flock
    try:
        sys.settrace(interrupt_before_writer_ownership)
        with pytest.raises(KeyboardInterrupt, match="before writer ownership"):
            ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        sys.settrace(None)

        assert interrupted
        assert artifacts._PENDING_LOCK_FDS == set()
        recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        recovered.close()
    finally:
        sys.settrace(None)
        for descriptor in list(artifacts._PENDING_LOCK_FDS):
            original_flock(descriptor, artifacts.fcntl.LOCK_UN)
            os.close(descriptor)
            artifacts._PENDING_LOCK_FDS.discard(descriptor)


@pytest.mark.parametrize(
    "target_statement",
    ["writer._lock_fd_marker = marker", "self._consume(descriptor)"],
)
def test_transfer_interruption_never_recloses_a_reused_candidate_fd(
    tmp_path,
    monkeypatch,
    target_statement,
):
    root = tmp_path / "cache"
    original_open = artifacts._open_directory
    original_handle_close = artifacts._DirectoryHandle.close
    candidate_descriptor = None
    replacement_descriptor = None
    interrupted = False
    source_lines, first_line = inspect.getsourcelines(
        artifacts._CandidateLease.transfer_to
    )
    interruption_line = first_line + next(
        offset
        for offset, line in enumerate(source_lines)
        if line.strip() == target_statement
    )

    def capture_candidate(directory, **kwargs):
        nonlocal candidate_descriptor
        lease, identity = original_open(directory, **kwargs)
        candidate_descriptor = lease.descriptor
        return lease, identity

    def close_then_reuse(handle):
        nonlocal replacement_descriptor
        descriptor = handle._descriptor
        if (
            descriptor == candidate_descriptor
            and handle._owner is not None
            and replacement_descriptor is None
        ):
            original_handle_close(handle)
            replacement_descriptor = os.open(os.devnull, os.O_RDONLY)
            assert replacement_descriptor == descriptor
            return
        return original_handle_close(handle)

    def interrupt_before_consume(frame, event, _argument):
        nonlocal interrupted
        if (
            not interrupted
            and frame.f_code is artifacts._CandidateLease.transfer_to.__code__
            and event == "line"
            and frame.f_lineno == interruption_line
        ):
            interrupted = True
            raise KeyboardInterrupt("interrupted before lease consume")
        return interrupt_before_consume

    monkeypatch.setattr(artifacts, "_open_directory", capture_candidate)
    monkeypatch.setattr(artifacts._DirectoryHandle, "close", close_then_reuse)
    try:
        sys.settrace(interrupt_before_consume)
        with pytest.raises(KeyboardInterrupt, match="before lease consume"):
            ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        sys.settrace(None)
        monkeypatch.setattr(
            artifacts._DirectoryHandle,
            "close",
            original_handle_close,
        )

        assert interrupted
        assert replacement_descriptor is not None
        os.fstat(replacement_descriptor)
    finally:
        sys.settrace(None)
        monkeypatch.setattr(
            artifacts._DirectoryHandle,
            "close",
            original_handle_close,
        )
        if replacement_descriptor is not None:
            try:
                os.close(replacement_descriptor)
            except OSError:
                pass


def test_same_process_handoff_does_not_open_an_unused_candidate(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    writer = ShardWriter(root, metadata, shard_size=1)
    writer.add(_record("a"))

    def unexpected_open(_directory, **_kwargs):
        raise AssertionError("handoff opened an unused candidate descriptor")

    monkeypatch.setattr(artifacts, "_open_directory", unexpected_open)
    resumed = ShardWriter(root, metadata, shard_size=1)
    resumed.close()


def test_handoff_never_double_closes_a_reused_candidate_fd(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    writer = ShardWriter(root, metadata, shard_size=1)
    writer.add(_record("a"))
    original_open = artifacts._open_directory
    original_close = artifacts.os.close
    candidate_fds = []
    replacement_fd = None

    def capture_candidate(directory, **kwargs):
        descriptor, identity = original_open(directory, **kwargs)
        candidate_fds.append(descriptor)
        return descriptor, identity

    def close_reuse_and_interrupt(descriptor):
        nonlocal replacement_fd
        if (
            replacement_fd is None
            and candidate_fds
            and descriptor == candidate_fds[0]
        ):
            original_close(descriptor)
            replacement_fd = os.open(os.devnull, os.O_RDONLY)
            assert replacement_fd == descriptor
            raise KeyboardInterrupt("close completed before interruption")
        original_close(descriptor)

    monkeypatch.setattr(artifacts, "_open_directory", capture_candidate)
    monkeypatch.setattr(artifacts.os, "close", close_reuse_and_interrupt)
    resumed = None
    try:
        try:
            resumed = ShardWriter(root, metadata, shard_size=1)
        except KeyboardInterrupt as error:
            assert str(error) == "close completed before interruption"
        monkeypatch.setattr(artifacts.os, "close", original_close)

        assert artifacts._PENDING_LOCK_FDS == set()
        if replacement_fd is None:
            assert resumed is not None
        else:
            os.fstat(replacement_fd)
    finally:
        monkeypatch.setattr(artifacts.os, "close", original_close)
        if resumed is not None:
            resumed.close()
        if replacement_fd is not None:
            try:
                original_close(replacement_fd)
            except OSError:
                pass


def test_lock_acquisition_base_exception_releases_pending_fd(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    original_flock = artifacts.fcntl.flock

    def lock_then_interrupt(descriptor, operation):
        original_flock(descriptor, operation)
        if operation & artifacts.fcntl.LOCK_EX:
            raise KeyboardInterrupt("interrupted after lock acquisition")

    monkeypatch.setattr(artifacts.fcntl, "flock", lock_then_interrupt)
    try:
        with pytest.raises(KeyboardInterrupt, match="after lock acquisition"):
            ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        assert artifacts._PENDING_LOCK_FDS == set()
        monkeypatch.setattr(artifacts.fcntl, "flock", original_flock)
        recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        recovered.close()
    finally:
        monkeypatch.setattr(artifacts.fcntl, "flock", original_flock)
        for descriptor in list(artifacts._PENDING_LOCK_FDS):
            original_flock(descriptor, artifacts.fcntl.LOCK_UN)
            os.close(descriptor)
            artifacts._PENDING_LOCK_FDS.discard(descriptor)


def test_release_lock_recovers_from_every_line_boundary(tmp_path):
    probe = ShardWriter(
        tmp_path / "line-probe",
        {"stage": "evaluation"},
        shard_size=1,
    )
    executed_lines = []
    release_codes = {
        ShardWriter._release_lock.__code__,
        ShardWriter._release_lock_safely.__code__,
        ShardWriter._release_lock_once.__code__,
    }

    def record_release_lines(frame, event, _argument):
        if frame.f_code in release_codes and event == "line":
            executed_lines.append((frame.f_code, frame.f_lineno))
        return record_release_lines

    sys.settrace(record_release_lines)
    try:
        probe.__exit__(RuntimeError, None, None)
    finally:
        sys.settrace(None)
    boundaries = list(dict.fromkeys(executed_lines))
    assert boundaries

    original_flock = artifacts.fcntl.flock
    original_close = artifacts.os.close
    for index, (target_code, target_line) in enumerate(boundaries):
        root = tmp_path / f"boundary-{index}"
        writer = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        writer.add(_record("a"))
        descriptor = writer._lock_fd
        interrupted = False

        def interrupt_at_boundary(frame, event, _argument):
            nonlocal interrupted
            if (
                not interrupted
                and frame.f_code is target_code
                and event == "line"
                and frame.f_lineno == target_line
            ):
                interrupted = True
                raise KeyboardInterrupt(f"interrupted at line {target_line}")
            return interrupt_at_boundary

        try:
            sys.settrace(interrupt_at_boundary)
            with pytest.raises(KeyboardInterrupt, match="interrupted at line"):
                writer.__exit__(RuntimeError, None, None)
            sys.settrace(None)

            assert interrupted
            recovered = ShardWriter(
                root,
                {"stage": "evaluation"},
                shard_size=1,
            )
            assert recovered.existing_keys() == {("a", 0)}
            recovered.close()
        finally:
            sys.settrace(None)
            writer._lock_fd = None
            if descriptor is not None:
                try:
                    original_flock(descriptor, artifacts.fcntl.LOCK_UN)
                except OSError:
                    pass
                try:
                    original_close(descriptor)
                except OSError:
                    pass


def test_release_reconciliation_recovers_from_every_line_boundary(
    tmp_path,
    monkeypatch,
):
    release_codes = {
        ShardWriter._release_lock.__code__,
        ShardWriter._release_lock_safely.__code__,
        ShardWriter._release_lock_once.__code__,
        ShardWriter._reconcile_lock_release.__code__,
    }

    def record_lines(target):
        def record(frame, event, _argument):
            if frame.f_code in release_codes and event == "line":
                target.append((frame.f_code, frame.f_lineno))
            return record

        return record

    normal_lines = []
    normal_probe = ShardWriter(
        tmp_path / "normal-reconciliation-probe",
        {"stage": "evaluation"},
        shard_size=1,
    )
    sys.settrace(record_lines(normal_lines))
    try:
        normal_probe.__exit__(RuntimeError, None, None)
    finally:
        sys.settrace(None)

    original_handle_close = artifacts._DirectoryHandle.close
    exception_lines = []
    close_calls = 0

    def interrupt_first_close(handle):
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise RuntimeError("enter release reconciliation")
        original_handle_close(handle)

    exception_probe = ShardWriter(
        tmp_path / "exception-reconciliation-probe",
        {"stage": "evaluation"},
        shard_size=1,
    )
    monkeypatch.setattr(artifacts._DirectoryHandle, "close", interrupt_first_close)
    sys.settrace(record_lines(exception_lines))
    try:
        with pytest.raises(RuntimeError, match="enter release reconciliation"):
            exception_probe.__exit__(RuntimeError, None, None)
    finally:
        sys.settrace(None)
        monkeypatch.setattr(artifacts._DirectoryHandle, "close", original_handle_close)

    normal_boundaries = set(normal_lines)
    boundaries = [
        boundary
        for boundary in dict.fromkeys(exception_lines)
        if boundary not in normal_boundaries
    ]
    assert boundaries
    assert any(
        code is ShardWriter._reconcile_lock_release.__code__
        for code, _line in boundaries
    )

    for index, (target_code, target_line) in enumerate(boundaries):
        root = tmp_path / f"reconciliation-boundary-{index}"
        writer = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        writer.add(_record("a"))
        descriptor = writer._lock_fd
        close_calls = 0
        interrupted = False

        def interrupt_at_boundary(frame, event, _argument):
            nonlocal interrupted
            if (
                not interrupted
                and frame.f_code is target_code
                and event == "line"
                and frame.f_lineno == target_line
            ):
                interrupted = True
                raise KeyboardInterrupt(f"interrupted at line {target_line}")
            return interrupt_at_boundary

        monkeypatch.setattr(artifacts._DirectoryHandle, "close", interrupt_first_close)
        sys.settrace(interrupt_at_boundary)
        try:
            with pytest.raises(KeyboardInterrupt, match="interrupted at line"):
                writer.__exit__(RuntimeError, None, None)
            sys.settrace(None)
            monkeypatch.setattr(artifacts._DirectoryHandle, "close", original_handle_close)

            assert interrupted
            recovered = ShardWriter(
                root,
                {"stage": "evaluation"},
                shard_size=1,
            )
            assert recovered.existing_keys() == {("a", 0)}
            recovered.close()
        finally:
            sys.settrace(None)
            monkeypatch.setattr(artifacts._DirectoryHandle, "close", original_handle_close)
            writer._lock_fd = None
            writer._lock_handle = None
            if descriptor is not None:
                os.closerange(descriptor, descriptor + 1)


@pytest.mark.parametrize("reuse_same_directory", [False, True])
def test_release_close_exception_never_recloses_a_reused_fd(
    tmp_path,
    monkeypatch,
    reuse_same_directory,
):
    root = tmp_path / "cache"
    writer = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
    writer.add(_record("a"))
    original_handle_close = artifacts._DirectoryHandle.close
    replacement_fd = None
    close_calls = 0

    def close_reuse_and_interrupt(handle):
        nonlocal close_calls, replacement_fd
        if handle._owner is None:
            return original_handle_close(handle)
        close_calls += 1
        first = handle._descriptor
        assert first is not None
        original_handle_close(handle)
        replacement = root if reuse_same_directory else os.devnull
        flags = os.O_RDONLY
        if reuse_same_directory:
            flags |= getattr(os, "O_DIRECTORY", 0)
        replacement_fd = os.open(replacement, flags)
        assert replacement_fd == first
        raise KeyboardInterrupt("close completed before interruption")

    monkeypatch.setattr(artifacts._DirectoryHandle, "close", close_reuse_and_interrupt)
    try:
        with pytest.raises(KeyboardInterrupt, match="close completed"):
            writer.__exit__(RuntimeError, None, None)
        monkeypatch.setattr(artifacts._DirectoryHandle, "close", original_handle_close)

        assert close_calls == 1
        assert replacement_fd is not None
        os.fstat(replacement_fd)
        recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        assert recovered.existing_keys() == {("a", 0)}
        recovered.close()
    finally:
        monkeypatch.setattr(artifacts._DirectoryHandle, "close", original_handle_close)
        if replacement_fd is not None:
            try:
                os.close(replacement_fd)
            except OSError:
                pass


def test_lock_descriptor_marker_is_nonzero_and_stable(tmp_path):
    writer = ShardWriter(
        tmp_path / "cache",
        {"stage": "evaluation"},
        shard_size=1,
    )
    descriptor = writer._lock_fd
    marker = writer._lock_fd_marker

    assert descriptor is not None
    assert marker is not None and marker > 0
    assert os.lseek(descriptor, 0, os.SEEK_CUR) == marker
    writer.add(_record("a"))
    assert os.lseek(descriptor, 0, os.SEEK_CUR) == marker
    writer.close()


def test_unsupported_lock_descriptor_marker_fails_cleanly(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    original_lseek = artifacts.os.lseek

    def reject_marker(descriptor, offset, whence):
        if whence == os.SEEK_SET and offset != 0:
            raise OSError(artifacts.errno.EINVAL, "marker unsupported")
        return original_lseek(descriptor, offset, whence)

    monkeypatch.setattr(artifacts.os, "lseek", reject_marker)
    with pytest.raises(RuntimeError, match="does not support ownership markers"):
        ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
    monkeypatch.setattr(artifacts.os, "lseek", original_lseek)

    recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
    recovered.close()


def test_release_retries_when_close_did_not_start(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    writer = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
    writer.add(_record("a"))
    descriptor = writer._lock_fd
    original_handle_close = artifacts._DirectoryHandle.close
    close_calls = 0

    def interrupt_before_close(handle):
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise KeyboardInterrupt("interrupted before close")
        original_handle_close(handle)

    monkeypatch.setattr(artifacts._DirectoryHandle, "close", interrupt_before_close)
    try:
        with pytest.raises(KeyboardInterrupt, match="before close"):
            writer.__exit__(RuntimeError, None, None)
        monkeypatch.setattr(artifacts._DirectoryHandle, "close", original_handle_close)

        assert close_calls == 2
        recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        assert recovered.existing_keys() == {("a", 0)}
        recovered.close()
    finally:
        monkeypatch.setattr(artifacts._DirectoryHandle, "close", original_handle_close)
        writer._lock_fd = None
        writer._lock_handle = None
        if descriptor is not None:
            os.closerange(descriptor, descriptor + 1)


def test_interrupted_handle_close_cannot_later_close_a_reused_fd(tmp_path):
    root = tmp_path / "cache"
    writer = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
    writer.add(_record("a"))
    descriptor = writer._lock_fd
    source_lines, first_line = inspect.getsourcelines(
        artifacts._DirectoryHandle.close
    )
    close_line = first_line + next(
        offset
        for offset, line in enumerate(source_lines)
        if line.strip() == "owner.close()"
    )
    interrupted = False
    caught = None
    replacement = None

    def interrupt_before_generator_close(frame, event, _argument):
        nonlocal interrupted
        if (
            not interrupted
            and frame.f_code is artifacts._DirectoryHandle.close.__code__
            and event == "line"
            and frame.f_lineno == close_line
        ):
            interrupted = True
            raise KeyboardInterrupt("interrupted before generator close")
        return interrupt_before_generator_close

    try:
        sys.settrace(interrupt_before_generator_close)
        try:
            writer.__exit__(RuntimeError, None, None)
        except KeyboardInterrupt as error:
            caught = error
        finally:
            sys.settrace(None)

        assert interrupted
        assert caught is not None
        assert descriptor is not None
        with pytest.raises(OSError) as closed:
            os.fstat(descriptor)
        assert closed.value.errno == artifacts.errno.EBADF

        replacement = os.open(os.devnull, os.O_RDONLY)
        assert replacement == descriptor
        caught = None
        gc.collect()
        os.fstat(replacement)
    finally:
        sys.settrace(None)
        if replacement is not None:
            try:
                os.close(replacement)
            except OSError:
                pass


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_during_handoff_closes_the_transferred_fd_in_the_child(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    writer = ShardWriter(root, metadata, shard_size=1)
    writer.add(_record("a"))
    original_take = ShardWriter._take_lock_from
    child_exit_codes = []

    def fork_after_transfer(successor, predecessor):
        descriptor = original_take(successor, predecessor)
        process_id = os.fork()
        if process_id == 0:
            try:
                os.fstat(descriptor)
            except OSError as error:
                os._exit(0 if error.errno == artifacts.errno.EBADF else 2)
            os._exit(1)
        _waited_id, status = os.waitpid(process_id, 0)
        child_exit_codes.append(os.waitstatus_to_exitcode(status))
        return descriptor

    monkeypatch.setattr(ShardWriter, "_take_lock_from", fork_after_transfer)
    resumed = ShardWriter(root, metadata, shard_size=1)
    resumed.close()

    assert child_exit_codes == [0]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_after_fresh_lock_acquisition_closes_the_fd_in_the_child(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    original_acquire = artifacts._acquire_directory_lock
    child_exit_codes = []

    def fork_after_acquire(directory, **kwargs):
        lease = original_acquire(directory, **kwargs)
        descriptor = lease.descriptor
        process_id = os.fork()
        if process_id == 0:
            try:
                os.fstat(descriptor)
            except OSError as error:
                os._exit(0 if error.errno == artifacts.errno.EBADF else 2)
            os._exit(1)
        _waited_id, status = os.waitpid(process_id, 0)
        child_exit_codes.append(os.waitstatus_to_exitcode(status))
        return lease

    monkeypatch.setattr(
        artifacts,
        "_acquire_directory_lock",
        fork_after_acquire,
    )
    writer = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
    writer.close()

    assert child_exit_codes == [0]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_before_lock_assignment_cannot_close_a_reused_fd(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    metadata = {"stage": "evaluation"}
    original_acquire = artifacts._acquire_directory_lock
    child_process_id = None
    in_child = False
    reused_descriptor = None

    def fork_before_assignment(directory, **kwargs):
        nonlocal child_process_id, in_child, reused_descriptor
        lease = original_acquire(directory, **kwargs)
        descriptor = lease.descriptor
        process_id = os.fork()
        if process_id == 0:
            in_child = True
            child_process_id = 0
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            reused_descriptor = os.open(unrelated, flags)
            if reused_descriptor != descriptor:
                os._exit(20)
            return lease
        child_process_id = process_id
        return lease

    monkeypatch.setattr(
        artifacts,
        "_acquire_directory_lock",
        fork_before_assignment,
    )
    try:
        try:
            writer = ShardWriter(root, metadata, shard_size=1)
        except RuntimeError as error:
            if not in_child:
                raise
            clean_failure = "fork" in str(error)
            descriptor_was_not_reclosed = True
            try:
                os.fstat(reused_descriptor)
            except OSError:
                descriptor_was_not_reclosed = False
            os._exit(0 if clean_failure and descriptor_was_not_reclosed else 21)

        if in_child:
            os._exit(22)

        assert child_process_id is not None
        _waited_id, status = os.waitpid(child_process_id, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        writer.add(_record("parent"))
        writer.close()
        assert [record["image_id"] for record in iter_records(root)] == [
            "parent"
        ]
        assert not any(unrelated.iterdir())
    finally:
        if in_child:
            os._exit(23)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
@pytest.mark.parametrize("reuse_same_directory", [False, True])
def test_fork_after_lock_marker_preserves_an_unrelated_fd(
    tmp_path,
    monkeypatch,
    reuse_same_directory,
):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    original_mark = artifacts._mark_lock_descriptor
    child_process_id = None
    in_child = False
    reused_descriptor = None

    def mark_then_fork(descriptor, planned_marker=None):
        nonlocal child_process_id, in_child, reused_descriptor
        marker = original_mark(descriptor, planned_marker)
        process_id = os.fork()
        if process_id == 0:
            in_child = True
            child_process_id = 0
            replacement = root if reuse_same_directory else os.devnull
            flags = os.O_RDONLY
            if reuse_same_directory:
                flags |= getattr(os, "O_DIRECTORY", 0)
            replacement_descriptor = os.open(replacement, flags)
            if replacement_descriptor == descriptor:
                os._exit(30)
            reused_descriptor = replacement_descriptor
            return marker
        child_process_id = process_id
        return marker

    monkeypatch.setattr(artifacts, "_mark_lock_descriptor", mark_then_fork)
    try:
        try:
            writer = ShardWriter(root, metadata, shard_size=1)
        except RuntimeError as error:
            if not in_child:
                raise
            clean_failure = "fork" in str(error)
            descriptor_was_not_reclosed = True
            try:
                os.fstat(reused_descriptor)
            except OSError:
                descriptor_was_not_reclosed = False
            os._exit(0 if clean_failure and descriptor_was_not_reclosed else 31)

        if in_child:
            os._exit(32)

        assert child_process_id is not None
        _waited_id, status = os.waitpid(child_process_id, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        writer.add(_record("parent"))
        writer.close()
        assert [record["image_id"] for record in iter_records(root)] == [
            "parent"
        ]
    finally:
        if in_child:
            os._exit(33)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_before_handoff_assignment_cannot_reuse_the_writer_fd(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    metadata = {"stage": "evaluation"}
    predecessor = ShardWriter(root, metadata, shard_size=1)
    predecessor.add(_record("a"))
    child_process_id = None
    in_child = False
    reused_descriptor = None

    class ForkAfterExtend(list):
        def extend(self, values):
            nonlocal child_process_id, in_child, reused_descriptor
            super().extend(values)
            descriptor = predecessor._lock_fd
            assert descriptor is not None
            process_id = os.fork()
            if process_id == 0:
                in_child = True
                child_process_id = 0
                flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                replacement_descriptor = os.open(unrelated, flags)
                if replacement_descriptor != descriptor:
                    os.dup2(replacement_descriptor, descriptor)
                    os.close(replacement_descriptor)
                reused_descriptor = descriptor
                return
            child_process_id = process_id

    monkeypatch.setattr(
        artifacts,
        "_TRANSFER_PARTICIPANTS",
        ForkAfterExtend(),
    )
    try:
        try:
            resumed = ShardWriter(root, metadata, shard_size=1)
        except RuntimeError as error:
            if not in_child:
                raise
            clean_failure = "fork" in str(error)
            descriptor_was_not_reclosed = True
            try:
                os.fstat(reused_descriptor)
            except OSError:
                descriptor_was_not_reclosed = False
            unrelated_is_clean = not any(unrelated.iterdir())
            os._exit(
                0
                if clean_failure
                and descriptor_was_not_reclosed
                and unrelated_is_clean
                else 41
            )

        if in_child:
            os._exit(42)

        assert child_process_id is not None
        _waited_id, status = os.waitpid(child_process_id, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        resumed.add(_record("b"))
        resumed.close()
        assert [record["image_id"] for record in iter_records(root)] == [
            "a",
            "b",
        ]
    finally:
        if in_child:
            os._exit(43)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_before_registration_cannot_reuse_the_writer_fd(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    metadata = {"stage": "evaluation"}
    original_register = artifacts._register_writer
    child_process_id = None
    in_child = False
    reused_descriptor = None

    def fork_before_registration(writer):
        nonlocal child_process_id, in_child, reused_descriptor
        descriptor = writer._lock_fd
        assert descriptor is not None
        process_id = os.fork()
        if process_id == 0:
            in_child = True
            child_process_id = 0
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            reused_descriptor = os.open(unrelated, flags)
            if reused_descriptor != descriptor:
                os._exit(10)
            original_register(writer)
            return
        child_process_id = process_id
        original_register(writer)

    monkeypatch.setattr(artifacts, "_register_writer", fork_before_registration)
    try:
        try:
            writer = ShardWriter(root, metadata, shard_size=1)
        except RuntimeError as error:
            if not in_child:
                raise
            clean_failure = "fork" in str(error)
            unrelated_is_clean = not any(unrelated.iterdir())
            descriptor_was_not_reclosed = True
            try:
                os.fstat(reused_descriptor)
            except OSError:
                descriptor_was_not_reclosed = False
            os._exit(
                0
                if clean_failure
                and unrelated_is_clean
                and descriptor_was_not_reclosed
                else 11
            )

        if in_child:
            writer.add(_record("child"))
            writer.close()
            os._exit(12 if (unrelated / "manifest.json").exists() else 13)

        assert child_process_id is not None
        _waited_id, status = os.waitpid(child_process_id, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        writer.add(_record("parent"))
        writer.close()
        assert [record["image_id"] for record in iter_records(root)] == [
            "parent"
        ]
        assert not any(unrelated.iterdir())
    finally:
        if in_child:
            os._exit(14)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_during_transfer_invalidates_both_child_participants(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    predecessor = ShardWriter(root, metadata, shard_size=1)
    predecessor.add(_record("a"))
    original_register = artifacts._register_writer
    child_exit_codes = []

    def register_then_fork(successor):
        original_register(successor)
        if successor is predecessor or child_exit_codes:
            return
        process_id = os.fork()
        if process_id == 0:
            predecessor_invalid = (
                predecessor._closed and predecessor._lock_fd is None
            )
            successor_invalid = successor._closed and successor._lock_fd is None
            os._exit(0 if predecessor_invalid and successor_invalid else 1)
        _waited_id, status = os.waitpid(process_id, 0)
        child_exit_codes.append(os.waitstatus_to_exitcode(status))

    monkeypatch.setattr(artifacts, "_register_writer", register_then_fork)
    resumed = ShardWriter(root, metadata, shard_size=1)
    resumed.close()

    assert child_exit_codes == [0]


def test_transfer_registration_failure_restores_the_predecessor(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    predecessor = ShardWriter(root, metadata, shard_size=1)
    predecessor.add(_record("a"))
    original_register = artifacts._register_writer

    def register_then_fail(successor):
        original_register(successor)
        if successor is not predecessor:
            raise RuntimeError("registration failed")

    monkeypatch.setattr(artifacts, "_register_writer", register_then_fail)
    with pytest.raises(RuntimeError, match="registration failed"):
        ShardWriter(root, metadata, shard_size=1)
    monkeypatch.setattr(artifacts, "_register_writer", original_register)

    predecessor.add(_record("b"))
    predecessor.close()
    assert [record["image_id"] for record in iter_records(root)] == ["a", "b"]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_forked_child_drops_inherited_handoff_ownership(tmp_path):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    writer = ShardWriter(root, metadata, shard_size=1)
    writer.add(_record("a"))

    process_id = os.fork()
    if process_id == 0:
        try:
            inherited_is_invalid = writer._closed and writer._lock_fd is None
            if not inherited_is_invalid:
                os._exit(1)
            try:
                ShardWriter(root, metadata, shard_size=1)
            except RuntimeError as error:
                if "active writer" in str(error):
                    os._exit(0)
                os._exit(2)
            else:
                os._exit(3)
        except BaseException:
            os._exit(4)

    _waited_id, status = os.waitpid(process_id, 0)
    exit_code = os.waitstatus_to_exitcode(status)
    assert exit_code == 0

    writer.add(_record("b"))
    writer.close()
    assert [record["image_id"] for record in iter_records(root)] == ["a", "b"]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_cross_process_writer_lock_is_exclusive_and_recovers_after_exit(tmp_path):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    ready_read, ready_write = os.pipe()
    release_read, release_write = os.pipe()
    process_id = os.fork()

    if process_id == 0:
        os.close(ready_read)
        os.close(release_write)
        try:
            child_writer = ShardWriter(root, metadata, shard_size=1)
            child_writer.add(_record("a"))
            os.write(ready_write, b"1")
            os.read(release_read, 1)
            os._exit(0)
        except BaseException:
            os._exit(1)

    os.close(ready_write)
    os.close(release_read)
    assert os.read(ready_read, 1) == b"1"
    with pytest.raises(RuntimeError, match="active writer"):
        ShardWriter(root, metadata, shard_size=1)

    os.write(release_write, b"1")
    os.close(release_write)
    _waited_id, status = os.waitpid(process_id, 0)
    assert os.waitstatus_to_exitcode(status) == 0

    resumed = ShardWriter(root, metadata, shard_size=1)
    assert resumed.existing_keys() == {("a", 0)}
    resumed.close()
    assert [record["image_id"] for record in iter_records(root)] == ["a"]


def test_duplicate_keys_are_rejected_in_memory_and_after_resume(tmp_path):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    with pytest.raises(RuntimeError, match="interrupted"):
        with ShardWriter(root, metadata, shard_size=2) as writer:
            writer.add({"image_id": "a", "value": torch.tensor([1])})
            with pytest.raises(ValueError, match="duplicate record key.*a.*0"):
                writer.add(_record("a", severity=0))
            writer.add(_record("b"))
            raise RuntimeError("interrupted")

    resumed = ShardWriter(root, metadata, shard_size=2)
    with pytest.raises(ValueError, match="duplicate record key.*b.*0"):
        resumed.add(_record("b"))
    resumed.close()

    assert [record["image_id"] for record in iter_records(root)] == ["a", "b"]


def test_writer_snapshots_nested_metadata_at_construction(tmp_path):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation", "config": {"layers": [1]}}
    writer = ShardWriter(root, metadata, shard_size=1)
    metadata["config"]["layers"].append(2)

    writer.add(_record("a"))
    writer.close()

    assert load_manifest(root)["config"] == {"layers": [1]}


class _UnsafeRecordValue:
    pass


def test_writer_rejects_nested_values_that_safe_torch_load_cannot_read(tmp_path):
    root = tmp_path / "cache"

    with pytest.raises(TypeError, match="safe Torch artifact value"):
        with ShardWriter(root, {"stage": "evaluation"}, shard_size=1) as writer:
            writer.add(
                {
                    "image_id": "a",
                    "nested": [{"unsafe": _UnsafeRecordValue()}],
                }
            )

    assert not (root / "manifest.json").exists()
    assert not list(root.glob("shard_*.pt"))


def test_writer_revalidates_nested_values_immediately_before_publication(tmp_path):
    root = tmp_path / "cache"
    nested = []
    writer = ShardWriter(root, {"stage": "evaluation"}, shard_size=2)
    writer.add(
        {
            "image_id": "a",
            "nested": nested,
        }
    )
    nested.append(_UnsafeRecordValue())

    with pytest.raises(TypeError, match="safe Torch artifact value"):
        writer.close()

    assert not (root / "manifest.json").exists()
    assert not list(root.glob("shard_*.pt"))


def test_resume_rejects_safe_loadable_values_outside_the_record_contract(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    name = "shard_00000.pt"
    shard = root / name
    torch.save(
        [{"image_id": "a", "unsafe": torch.nn.Parameter(torch.ones(1))}],
        shard,
    )
    atomic_json(
        {
            "schema_version": 1,
            "stage": "evaluation",
            "shard_size": 1,
            "record_count": 1,
            "shards": [name],
            "shard_sha256": {name: sha256_file(shard)},
        },
        root / "partial_manifest.json",
    )

    with pytest.raises(TypeError, match="safe Torch artifact value"):
        ShardWriter(root, {"stage": "evaluation"}, shard_size=1)


@pytest.mark.parametrize("shard_size", [0, -1, 1.5, True])
def test_shard_size_must_be_a_positive_integer(tmp_path, shard_size):
    with pytest.raises(ValueError, match="shard_size must be a positive integer"):
        ShardWriter(tmp_path / "cache", {}, shard_size=shard_size)


def test_resume_refuses_metadata_or_shard_size_changes(tmp_path):
    root = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="interrupted"):
        with ShardWriter(
            root,
            {"stage": "reference", "input_id": "abc"},
            shard_size=1,
        ) as writer:
            writer.add(_record("a"))
            raise RuntimeError("interrupted")

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
    with pytest.raises(RuntimeError, match="interrupted"):
        with ShardWriter(root, {"stage": "reference"}, shard_size=1) as writer:
            writer.add(_record("a"))
            raise RuntimeError("interrupted")

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


@pytest.mark.parametrize("schema_version", [None, 2, True])
def test_iter_records_requires_exact_supported_schema_version(
    tmp_path,
    schema_version,
):
    root = tmp_path / "cache"
    with ShardWriter(root, {"stage": "evaluation"}, shard_size=1) as writer:
        writer.add(_record("a"))
    manifest = load_manifest(root)
    if schema_version is None:
        manifest.pop("schema_version")
    else:
        manifest["schema_version"] = schema_version
    atomic_json(manifest, root / "manifest.json")

    with pytest.raises(ValueError, match="schema_version"):
        list(iter_records(root))


def test_iter_records_requires_one_digest_for_every_shard(tmp_path):
    root = tmp_path / "cache"
    with ShardWriter(root, {"stage": "evaluation"}, shard_size=1) as writer:
        writer.add(_record("a"))
    manifest = load_manifest(root)
    manifest.pop("shard_sha256", None)
    atomic_json(manifest, root / "manifest.json")

    with pytest.raises(ValueError, match="shard_sha256"):
        list(iter_records(root))


def test_iter_records_detects_swapped_shard_contents(tmp_path):
    root = tmp_path / "cache"
    with ShardWriter(root, {"stage": "evaluation"}, shard_size=1) as writer:
        writer.add(_record("a"))
        writer.add(_record("b"))
    first = root / "shard_00000.pt"
    second = root / "shard_00001.pt"
    first_bytes = first.read_bytes()
    first.write_bytes(second.read_bytes())
    second.write_bytes(first_bytes)

    with pytest.raises(ValueError, match="SHA-256"):
        list(iter_records(root))


def test_iter_records_loads_the_same_open_shard_file_that_was_verified(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    with ShardWriter(root, {"stage": "evaluation"}, shard_size=1) as writer:
        writer.add(_record("original"))
    shard = root / "shard_00000.pt"
    replacement = tmp_path / "replacement.pt"
    torch.save([_record("replacement")], replacement)
    original_load = torch.load
    swapped = False

    def replace_path_before_load(source, *args, **kwargs):
        nonlocal swapped
        if not swapped and isinstance(source, Path) and source == shard:
            replacement.replace(shard)
            swapped = True
        return original_load(source, *args, **kwargs)

    monkeypatch.setattr(artifacts.torch, "load", replace_path_before_load)

    assert [record["image_id"] for record in iter_records(root)] == ["original"]


def test_resume_detects_a_tampered_published_shard(tmp_path):
    root = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="interrupted"):
        with ShardWriter(root, {"stage": "evaluation"}, shard_size=1) as writer:
            writer.add(_record("a"))
            raise RuntimeError("interrupted")
    shard = root / "shard_00000.pt"
    shard.write_bytes(shard.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="SHA-256"):
        ShardWriter(root, {"stage": "evaluation"}, shard_size=1)


def test_iter_records_rejects_duplicate_record_keys_in_forged_shards(tmp_path):
    root = tmp_path / "cache"
    _write_manual_artifact(root, [_record("a"), _record("a")])

    with pytest.raises(ValueError, match="duplicate record key"):
        list(iter_records(root))


def test_iter_records_validates_every_record_key(tmp_path):
    root = tmp_path / "cache"
    _write_manual_artifact(root, [{"value": torch.tensor([1])}])

    with pytest.raises(ValueError, match="image_id"):
        list(iter_records(root))


def test_iter_records_rejects_symlinked_shards(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    outside = tmp_path / "outside.pt"
    torch.save([_record("a")], outside)
    shard = root / "shard_00000.pt"
    shard.symlink_to(outside)
    atomic_json(
        {
            "schema_version": 1,
            "record_count": 1,
            "shards": [shard.name],
            "shard_sha256": {shard.name: sha256_file(outside)},
        },
        root / "manifest.json",
    )

    with pytest.raises(ValueError, match="regular file"):
        list(iter_records(root))

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
                "shard_sha256": {
                    "shard_00000.pt": sha256_file(root / "shard_00000.pt")
                },
            }
        ),
        encoding="utf-8",
    )
    _EXECUTED.clear()

    with pytest.raises(pickle.UnpicklingError):
        list(iter_records(root))

    assert _EXECUTED == []
