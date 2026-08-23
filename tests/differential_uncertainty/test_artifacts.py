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

    def observe_registry_boundary(directory):
        observed.append(tracking_lock.held_by_current_thread())
        return original_open_directory(directory)

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


@pytest.mark.parametrize(
    "target_statement",
    ["self._lock_fd = descriptor", "candidate_descriptor = None"],
)
def test_lock_result_transfer_is_async_exception_safe(
    tmp_path,
    target_statement,
):
    root = tmp_path / "cache"
    source_lines, first_line = inspect.getsourcelines(ShardWriter.__init__)
    acquire_offset = next(
        offset
        for offset, line in enumerate(source_lines)
        if "descriptor = _acquire_directory_lock(" in line
    )
    ownership_line = first_line + next(
        offset
        for offset, line in enumerate(source_lines[acquire_offset:], acquire_offset)
        if line.strip() == target_statement
    )
    interrupted = False

    def interrupt_before_writer_ownership(frame, event, _argument):
        nonlocal interrupted
        if (
            frame.f_code is ShardWriter.__init__.__code__
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


def test_same_process_handoff_does_not_open_an_unused_candidate(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cache"
    metadata = {"stage": "evaluation"}
    writer = ShardWriter(root, metadata, shard_size=1)
    writer.add(_record("a"))

    def unexpected_open(_directory):
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

    def capture_candidate(directory):
        descriptor, identity = original_open(directory)
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

    original_closerange = artifacts.os.closerange
    exception_lines = []
    close_calls = 0

    def interrupt_first_close(first, last):
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise RuntimeError("enter release reconciliation")
        original_closerange(first, last)

    exception_probe = ShardWriter(
        tmp_path / "exception-reconciliation-probe",
        {"stage": "evaluation"},
        shard_size=1,
    )
    monkeypatch.setattr(artifacts.os, "closerange", interrupt_first_close)
    sys.settrace(record_lines(exception_lines))
    try:
        with pytest.raises(RuntimeError, match="enter release reconciliation"):
            exception_probe.__exit__(RuntimeError, None, None)
    finally:
        sys.settrace(None)
        monkeypatch.setattr(
            artifacts.os,
            "closerange",
            original_closerange,
        )

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

        monkeypatch.setattr(artifacts.os, "closerange", interrupt_first_close)
        sys.settrace(interrupt_at_boundary)
        try:
            with pytest.raises(KeyboardInterrupt, match="interrupted at line"):
                writer.__exit__(RuntimeError, None, None)
            sys.settrace(None)
            monkeypatch.setattr(
                artifacts.os,
                "closerange",
                original_closerange,
            )

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
            monkeypatch.setattr(
                artifacts.os,
                "closerange",
                original_closerange,
            )
            writer._lock_fd = None
            if descriptor is not None:
                original_closerange(descriptor, descriptor + 1)


@pytest.mark.parametrize("reuse_same_directory", [False, True])
def test_release_close_exception_never_recloses_a_reused_fd(
    tmp_path,
    monkeypatch,
    reuse_same_directory,
):
    root = tmp_path / "cache"
    writer = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
    writer.add(_record("a"))
    original_closerange = artifacts.os.closerange
    replacement_fd = None
    close_calls = 0

    def close_reuse_and_interrupt(first, last):
        nonlocal close_calls, replacement_fd
        close_calls += 1
        assert last == first + 1
        original_closerange(first, last)
        replacement = root if reuse_same_directory else os.devnull
        flags = os.O_RDONLY
        if reuse_same_directory:
            flags |= getattr(os, "O_DIRECTORY", 0)
        replacement_fd = os.open(replacement, flags)
        assert replacement_fd == first
        raise KeyboardInterrupt("close completed before interruption")

    monkeypatch.setattr(artifacts.os, "closerange", close_reuse_and_interrupt)
    try:
        with pytest.raises(KeyboardInterrupt, match="close completed"):
            writer.__exit__(RuntimeError, None, None)
        monkeypatch.setattr(artifacts.os, "closerange", original_closerange)

        assert close_calls == 1
        assert replacement_fd is not None
        os.fstat(replacement_fd)
        recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        assert recovered.existing_keys() == {("a", 0)}
        recovered.close()
    finally:
        monkeypatch.setattr(artifacts.os, "closerange", original_closerange)
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
    original_closerange = artifacts.os.closerange
    close_calls = 0

    def interrupt_before_close(first, last):
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise KeyboardInterrupt("interrupted before close")
        original_closerange(first, last)

    monkeypatch.setattr(artifacts.os, "closerange", interrupt_before_close)
    try:
        with pytest.raises(KeyboardInterrupt, match="before close"):
            writer.__exit__(RuntimeError, None, None)
        monkeypatch.setattr(artifacts.os, "closerange", original_closerange)

        assert close_calls == 2
        recovered = ShardWriter(root, {"stage": "evaluation"}, shard_size=1)
        assert recovered.existing_keys() == {("a", 0)}
        recovered.close()
    finally:
        monkeypatch.setattr(artifacts.os, "closerange", original_closerange)
        writer._lock_fd = None
        if descriptor is not None:
            original_closerange(descriptor, descriptor + 1)


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
        descriptor = original_acquire(directory, **kwargs)
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
        descriptor = original_acquire(directory, **kwargs)
        process_id = os.fork()
        if process_id == 0:
            in_child = True
            child_process_id = 0
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            reused_descriptor = os.open(unrelated, flags)
            if reused_descriptor != descriptor:
                os._exit(20)
            return descriptor
        child_process_id = process_id
        return descriptor

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
def test_fork_after_lock_marker_cannot_close_a_reused_fd(
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

    def mark_then_fork(descriptor):
        nonlocal child_process_id, in_child, reused_descriptor
        marker = original_mark(descriptor)
        process_id = os.fork()
        if process_id == 0:
            in_child = True
            child_process_id = 0
            replacement = root if reuse_same_directory else os.devnull
            flags = os.O_RDONLY
            if reuse_same_directory:
                flags |= getattr(os, "O_DIRECTORY", 0)
            replacement_descriptor = os.open(replacement, flags)
            if replacement_descriptor != descriptor:
                os.dup2(replacement_descriptor, descriptor)
                os.close(replacement_descriptor)
            reused_descriptor = descriptor
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
