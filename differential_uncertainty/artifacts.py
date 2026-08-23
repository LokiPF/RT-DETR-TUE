from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import secrets
import stat
import tempfile
import threading
import weakref
from collections.abc import Iterator, Mapping
from numbers import Integral
from pathlib import Path

import torch


SCHEMA_VERSION = 1
_MANIFEST_KEYS = frozenset(
    {"schema_version", "shard_size", "record_count", "shards", "shard_sha256"}
)
_ACTIVE_WRITERS: dict[tuple[int, int], weakref.ReferenceType] = {}
_WRITER_REGISTRY_LOCK = threading.RLock()
_FORK_LOCKED_WRITERS: list["ShardWriter"] = []
_CONSTRUCTING_WRITERS: list["ShardWriter"] = []
_PENDING_LOCK_FDS: set[int] = set()
_TRANSFER_PARTICIPANTS: list["ShardWriter"] = []
_FORK_GENERATION = 0


def _json_snapshot(value):
    return json.loads(
        json.dumps(value, separators=(",", ":"), allow_nan=False)
    )


def _canonical(value) -> str:
    """Return canonical text after applying JSON's on-disk normalization."""
    normalized = _json_snapshot(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _positive_integer(value, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _secure_temporary_file(target: Path) -> tuple[int, Path]:
    descriptor, name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    return descriptor, Path(name)


def _discard_temporary(descriptor: int | None, path: Path) -> None:
    if descriptor is not None:
        try:
            os.close(descriptor)
        except OSError:
            pass
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(value: Mapping, path: str | Path) -> None:
    """Atomically publish a JSON mapping at the requested path."""
    if not isinstance(value, Mapping):
        raise TypeError("atomic JSON values must be mappings")
    text = json.dumps(
        dict(value), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = _secure_temporary_file(target)
    try:
        handle = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor = None
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        _discard_temporary(descriptor, temporary)
        raise


def _atomic_json_create(value: Mapping, path: str | Path) -> bool:
    """Atomically publish JSON only if the target does not already exist."""
    if not isinstance(value, Mapping):
        raise TypeError("atomic JSON values must be mappings")
    text = json.dumps(
        dict(value), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = _secure_temporary_file(target)
    try:
        handle = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor = None
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError:
            return False
        temporary.unlink()
        _fsync_directory(target.parent)
        return True
    finally:
        _discard_temporary(descriptor, temporary)


def atomic_torch(value, path: str | Path) -> None:
    """Atomically publish a Torch value at the requested path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = _secure_temporary_file(target)
    try:
        handle = os.fdopen(descriptor, "wb")
        descriptor = None
        with handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        _discard_temporary(descriptor, temporary)
        raise


def _secure_temporary_file_at(
    directory_fd: int,
    target_name: str,
) -> tuple[int, str]:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    for _attempt in range(100):
        name = f".{target_name}.{secrets.token_hex(16)}.tmp"
        try:
            descriptor = os.open(
                name,
                flags,
                0o600,
                dir_fd=directory_fd,
            )
        except FileExistsError:
            continue
        return descriptor, name
    raise FileExistsError("could not allocate a unique artifact staging file")


def _discard_temporary_at(
    descriptor: int | None,
    name: str,
    directory_fd: int,
) -> None:
    if descriptor is not None:
        try:
            os.close(descriptor)
        except OSError:
            pass
    try:
        os.unlink(name, dir_fd=directory_fd)
    except FileNotFoundError:
        pass


def _atomic_json_at(value: Mapping, name: str, directory_fd: int) -> None:
    text = json.dumps(
        dict(value), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    descriptor, temporary = _secure_temporary_file_at(directory_fd, name)
    try:
        handle = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor = None
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            temporary,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    except BaseException:
        _discard_temporary_at(descriptor, temporary, directory_fd)
        raise


def _atomic_torch_at(value, name: str, directory_fd: int) -> None:
    descriptor, temporary = _secure_temporary_file_at(directory_fd, name)
    try:
        handle = os.fdopen(descriptor, "wb")
        descriptor = None
        with handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            temporary,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    except BaseException:
        _discard_temporary_at(descriptor, temporary, directory_fd)
        raise


def _path_exists_at(directory_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _read_json_at(directory_fd: int, name: str):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"artifact file {name!r} must be a regular file")
        handle = os.fdopen(descriptor, "r", encoding="utf-8")
        descriptor = None
        with handle:
            return json.load(handle)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _unlink_at(directory_fd: int, name: str) -> None:
    try:
        os.unlink(name, dir_fd=directory_fd)
    except FileNotFoundError:
        pass


def _sha256_stream(handle, chunk_size: int) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(chunk_size):
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_file_at(directory_fd: int, name: str) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"artifact file {name!r} must be a regular file")
        handle = os.fdopen(descriptor, "rb")
        descriptor = None
        with handle:
            return _sha256_stream(handle, 1024 * 1024)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def sha256_file(
    value: str | Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Hash file content without loading the complete file into memory."""
    size = _positive_integer(chunk_size, name="chunk_size")
    with Path(value).open("rb") as handle:
        return _sha256_stream(handle, size)


def source_digest(paths, *, root: str | Path) -> str:
    """Hash source contents and root-relative names, independent of clone path."""
    source_root = Path(root).resolve()
    entries: list[tuple[str, Path]] = []
    for value in paths:
        path = Path(value).resolve()
        try:
            relative = path.relative_to(source_root).as_posix()
        except ValueError as error:
            raise ValueError(f"source path is outside source root: {path}") from error
        entries.append((relative, path))

    digest = hashlib.sha256()
    for relative, path in sorted(entries, key=lambda entry: entry[0]):
        name = relative.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, byteorder="big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, byteorder="big"))
        digest.update(content)
    return digest.hexdigest()


def _mismatch_message(
    label: str,
    key: str,
    actual: Mapping,
    expected: Mapping,
) -> str:
    actual_value = actual[key] if key in actual else "<missing>"
    expected_value = expected[key] if key in expected else "<missing>"
    return (
        f"{label} mismatch for {key}: actual={actual_value!r}, "
        f"expected={expected_value!r}"
    )


def _ensure_exact_mapping(
    actual: Mapping,
    expected: Mapping,
    *,
    label: str,
) -> None:
    for key in sorted(set(actual) | set(expected), key=str):
        if key not in actual or key not in expected:
            raise ValueError(_mismatch_message(label, str(key), actual, expected))
        if _canonical(actual[key]) != _canonical(expected[key]):
            raise ValueError(_mismatch_message(label, str(key), actual, expected))


def ensure_provenance(run_directory: str | Path, expected: Mapping) -> Path:
    """Create run provenance or refuse any JSON-semantic mismatch."""
    if not isinstance(expected, Mapping):
        raise TypeError("expected provenance must be a mapping")
    expected_dict = _json_snapshot(dict(expected))
    path = Path(run_directory) / "artifacts" / "provenance.json"
    if _atomic_json_create(expected_dict, path):
        return path

    actual = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(actual, dict):
        raise ValueError("run provenance must contain a JSON object")
    _ensure_exact_mapping(actual, expected_dict, label="run provenance")
    return path


def _validate_safe_artifact_value(
    value,
    *,
    path: str = "record",
    active: set[int] | None = None,
) -> None:
    value_type = type(value)
    if value_type in (type(None), bool, int, float, str, bytes):
        return
    if value_type is torch.Tensor:
        return

    if value_type not in (dict, list, tuple):
        raise TypeError(
            f"{path} has unsupported {value_type.__name__}; "
            "expected a safe Torch artifact value"
        )

    if active is None:
        active = set()
    identity = id(value)
    if identity in active:
        raise TypeError(f"{path} contains a cycle, not a safe Torch artifact value")
    active.add(identity)
    try:
        if value_type is dict:
            for key, nested in value.items():
                if type(key) not in (str, int):
                    raise TypeError(
                        f"{path} has an unsupported dictionary key; "
                        "expected a safe Torch artifact value"
                    )
                _validate_safe_artifact_value(
                    nested,
                    path=f"{path}[{key!r}]",
                    active=active,
                )
        else:
            for index, nested in enumerate(value):
                _validate_safe_artifact_value(
                    nested,
                    path=f"{path}[{index}]",
                    active=active,
                )
    finally:
        active.remove(identity)


def _record_key(record: Mapping) -> tuple[str, int]:
    if "image_id" not in record:
        raise ValueError("artifact records must contain image_id")
    image_id = record["image_id"]
    if image_id is None or isinstance(image_id, bool) or not isinstance(
        image_id, (str, Integral)
    ):
        raise ValueError("record image_id must be a string or integer")
    severity = record.get("severity", 0)
    if isinstance(severity, bool) or not isinstance(severity, Integral):
        raise ValueError("record severity must be an integer")
    return str(image_id), int(severity)


def _require_schema(value: Mapping, *, label: str) -> None:
    schema_version = value.get("schema_version")
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"{label} schema_version must be exactly {SCHEMA_VERSION}"
        )


def _shard_path(root: Path, name: str) -> Path:
    if not isinstance(name, str) or Path(name).name != name or not name:
        raise ValueError(f"invalid shard name: {name!r}")
    return root / name


def _validate_shard_sequence(shards, *, label: str) -> list[str]:
    if not isinstance(shards, list) or not all(
        isinstance(name, str) for name in shards
    ):
        raise ValueError(f"{label} shards must be a list of names")
    expected = [f"shard_{index:05d}.pt" for index in range(len(shards))]
    if shards != expected:
        raise ValueError(f"{label} must contain the canonical shard sequence")
    return shards


def _validate_shard_digests(
    value,
    shards: list[str],
    *,
    label: str,
) -> dict[str, str]:
    if type(value) is not dict or set(value) != set(shards):
        raise ValueError(
            f"{label} shard_sha256 must contain exactly one digest per shard"
        )
    for name in shards:
        digest = value[name]
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                f"{label} shard_sha256 has an invalid digest for {name}"
            )
    return dict(value)


def _safe_load_shard(
    root: Path,
    name: str,
    expected_sha256: str,
    *,
    directory_fd: int | None = None,
) -> list[dict]:
    path = _shard_path(root, name)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None
    try:
        if directory_fd is None:
            descriptor = os.open(path, flags)
        else:
            descriptor = os.open(
                name,
                flags,
                dir_fd=directory_fd,
            )
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOENT, errno.ENOTDIR):
            raise ValueError(
                f"artifact shard {name!r} must be a regular file"
            ) from error
        raise

    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"artifact shard {name!r} must be a regular file")
        handle = os.fdopen(descriptor, "rb")
        descriptor = None
        with handle:
            actual_sha256 = _sha256_stream(handle, 1024 * 1024)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"artifact shard {name!r} SHA-256 mismatch: "
                    f"actual={actual_sha256}, expected={expected_sha256}"
                )
            handle.seek(0)
            records = torch.load(handle, map_location="cpu", weights_only=True)
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if type(records) is not list or not all(type(record) is dict for record in records):
        raise ValueError(f"artifact shard {name!r} must contain a list of records")
    return records


def _open_directory(directory: Path) -> tuple[int, tuple[int, int]]:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    while True:
        generation = _FORK_GENERATION
        descriptor = os.open(directory, flags)
        _PENDING_LOCK_FDS.add(descriptor)
        if generation == _FORK_GENERATION:
            break
        _PENDING_LOCK_FDS.discard(descriptor)
        try:
            os.close(descriptor)
        except OSError:
            pass
    try:
        state = os.fstat(descriptor)
    except BaseException:
        _PENDING_LOCK_FDS.discard(descriptor)
        os.close(descriptor)
        raise
    return descriptor, (state.st_dev, state.st_ino)


def _mark_lock_descriptor(descriptor: int) -> int:
    # ShardWriter never iterates this private directory fd; its seek offset is
    # therefore a stable open-file-description marker on supported Linux fds.
    marker = secrets.randbelow((1 << 62) - 1) + 1
    try:
        actual = os.lseek(descriptor, marker, os.SEEK_SET)
    except OSError as error:
        raise RuntimeError(
            "artifact directory fd does not support ownership markers"
        ) from error
    if actual != marker:
        raise RuntimeError(
            "artifact directory fd did not retain its ownership marker"
        )
    return marker


def _matches_lock_descriptor(
    descriptor: int,
    identity: tuple[int, int] | None,
    marker: int | None,
) -> bool:
    if identity is None or marker is None:
        return False
    try:
        state = os.fstat(descriptor)
        offset = os.lseek(descriptor, 0, os.SEEK_CUR)
    except OSError:
        return False
    return (state.st_dev, state.st_ino) == identity and offset == marker


def _acquire_directory_lock(
    directory: Path,
    *,
    descriptor: int | None = None,
) -> int:
    # The kernel owns this lock through the open descriptor, so process death
    # releases it without leaving a stale lock file that needs manual recovery.
    if descriptor is None:
        descriptor, _identity = _open_directory(directory)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException as error:
        _PENDING_LOCK_FDS.discard(descriptor)
        os.close(descriptor)
        if isinstance(error, OSError) and error.errno in (
            errno.EACCES,
            errno.EAGAIN,
        ):
            raise RuntimeError(
                f"artifact directory already has an active writer: {directory}"
            ) from error
        raise
    return descriptor


def _registered_writer(identity: tuple[int, int]):
    reference = _ACTIVE_WRITERS.get(identity)
    if reference is None:
        return None
    writer = reference()
    if writer is None:
        _ACTIVE_WRITERS.pop(identity, None)
        return None
    if writer._owner_pid != os.getpid():
        return None
    return writer


def _register_writer(writer: "ShardWriter") -> None:
    identity = writer._directory_identity

    def discard(reference) -> None:
        with _WRITER_REGISTRY_LOCK:
            if _ACTIVE_WRITERS.get(identity) is reference:
                _ACTIVE_WRITERS.pop(identity, None)

    _ACTIVE_WRITERS[identity] = weakref.ref(writer, discard)


class ShardWriter:
    """Publish restartable record shards and one immutable final manifest."""

    def __init__(
        self,
        directory: str | Path,
        metadata: Mapping,
        shard_size: int = 50,
    ) -> None:
        if not isinstance(metadata, Mapping):
            raise TypeError("artifact metadata must be a mapping")
        self.shard_size = _positive_integer(shard_size, name="shard_size")
        self.metadata = _json_snapshot(dict(metadata))
        reserved = sorted(_MANIFEST_KEYS.intersection(self.metadata))
        if reserved:
            raise ValueError(f"artifact metadata uses reserved key: {reserved[0]}")

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.directory = self.directory.resolve()
        self.final = self.directory / "manifest.json"
        self.partial = self.directory / "partial_manifest.json"
        self.buffer: list[dict] = []
        self._buffer_keys: set[tuple[str, int]] = set()
        self._state_lock = threading.RLock()
        self._superseded = False
        self._closed = True
        self._lock_fd: int | None = None
        self._lock_fd_marker: int | None = None
        self._owner_pid = os.getpid()
        self._directory_identity: tuple[int, int] | None = None
        candidate_descriptor: int | None = None

        with _WRITER_REGISTRY_LOCK:
            _CONSTRUCTING_WRITERS.append(self)
            try:
                descriptor = None
                predecessor = None
                if _ACTIVE_WRITERS:
                    path_state = os.stat(self.directory)
                    path_identity = (path_state.st_dev, path_state.st_ino)
                    predecessor = _registered_writer(path_identity)
                if predecessor is not None and predecessor is not self:
                    # The predecessor fd pins the inode observed by stat.  No
                    # second directory fd is needed, so handoff has no unused
                    # descriptor whose close result could be ambiguous.
                    self._directory_identity = path_identity
                    descriptor = self._take_lock_from(predecessor)
                if descriptor is None:
                    (
                        candidate_descriptor,
                        self._directory_identity,
                    ) = _open_directory(self.directory)
                    descriptor = _acquire_directory_lock(
                        self.directory,
                        descriptor=candidate_descriptor,
                    )
                    if self._owner_pid != os.getpid() or self._superseded:
                        raise RuntimeError(
                            "artifact writer construction invalidated by fork"
                        )
                    marker = _mark_lock_descriptor(descriptor)
                    if self._owner_pid != os.getpid() or self._superseded:
                        raise RuntimeError(
                            "artifact writer construction invalidated by fork"
                        )
                    self._lock_fd = descriptor
                    self._lock_fd_marker = marker
                    candidate_descriptor = None
                    self._closed = False
                    _register_writer(self)
                    if self._owner_pid != os.getpid() or self._superseded:
                        raise RuntimeError(
                            "artifact writer construction invalidated by fork"
                        )
                    _PENDING_LOCK_FDS.discard(descriptor)

                if _path_exists_at(self._lock_fd, self.final.name):
                    raise FileExistsError(f"artifact is already complete: {self.final}")
                if _path_exists_at(self._lock_fd, self.partial.name):
                    state = _read_json_at(
                        self._lock_fd,
                        self.partial.name,
                    )
                    self._resume(state)
                else:
                    self.shards: list[str] = []
                    self.shard_sha256: dict[str, str] = {}
                    self.record_count = 0
                    self._published_keys: set[tuple[str, int]] = set()
                    self._publish_partial()
            except BaseException:
                if (
                    candidate_descriptor is not None
                    and self._owner_pid == os.getpid()
                    and self._lock_fd != candidate_descriptor
                ):
                    _PENDING_LOCK_FDS.discard(candidate_descriptor)
                    try:
                        os.close(candidate_descriptor)
                    except OSError:
                        pass
                self._closed = True
                reference = _ACTIVE_WRITERS.get(self._directory_identity)
                if reference is not None and reference() is self:
                    _ACTIVE_WRITERS.pop(self._directory_identity, None)
                self._release_lock_safely()
                raise
            finally:
                for index, writer in enumerate(_CONSTRUCTING_WRITERS):
                    if writer is self:
                        del _CONSTRUCTING_WRITERS[index]
                        break

    def _take_lock_from(self, predecessor: "ShardWriter") -> int | None:
        with predecessor._state_lock:
            if predecessor._closed or predecessor._lock_fd is None:
                return None
            if predecessor.buffer or predecessor._buffer_keys:
                raise RuntimeError("cannot hand off writer with unpublished buffer")
            if predecessor.shard_size != self.shard_size:
                raise ValueError("cannot resume with a different shard_size")
            _ensure_exact_mapping(
                predecessor.metadata,
                self.metadata,
                label="partial artifact",
            )
            state = _read_json_at(
                predecessor._lock_fd,
                predecessor.partial.name,
            )
            if type(state) is not dict:
                raise ValueError("partial artifact manifest must be a JSON object")
            _ensure_exact_mapping(
                state,
                predecessor._partial_value(),
                label="writer handoff",
            )

            descriptor = predecessor._lock_fd
            marker = predecessor._lock_fd_marker
            if marker is None:
                raise RuntimeError("active artifact writer has no lock marker")
            previous_reference = _ACTIVE_WRITERS.get(
                self._directory_identity
            )
            predecessor_closed = predecessor._closed
            predecessor_superseded = predecessor._superseded
            _TRANSFER_PARTICIPANTS.extend((predecessor, self))
            try:
                if self._owner_pid != os.getpid() or self._superseded:
                    raise RuntimeError(
                        "artifact writer construction invalidated by fork"
                    )
                self._lock_fd = descriptor
                self._lock_fd_marker = marker
                self._closed = False
                _register_writer(self)
                if self._owner_pid != os.getpid() or self._superseded:
                    raise RuntimeError(
                        "artifact writer construction invalidated by fork"
                    )
                predecessor._lock_fd = None
                predecessor._lock_fd_marker = None
                predecessor._closed = True
                predecessor._superseded = True
                return descriptor
            except BaseException:
                self._lock_fd = None
                self._lock_fd_marker = None
                self._closed = True
                if self._owner_pid != os.getpid() or self._superseded:
                    predecessor._lock_fd = None
                    predecessor._lock_fd_marker = None
                    predecessor._closed = True
                    predecessor._superseded = True
                    raise
                predecessor._lock_fd = descriptor
                predecessor._lock_fd_marker = marker
                predecessor._closed = predecessor_closed
                predecessor._superseded = predecessor_superseded
                if previous_reference is None:
                    _ACTIVE_WRITERS.pop(self._directory_identity, None)
                else:
                    _ACTIVE_WRITERS[
                        self._directory_identity
                    ] = previous_reference
                raise
            finally:
                del _TRANSFER_PARTICIPANTS[-2:]

    def _require_active(self) -> None:
        if self._superseded:
            raise RuntimeError("artifact writer has been superseded")
        if self._closed:
            raise RuntimeError("artifact writer is closed")

    def _unregister(self) -> None:
        with _WRITER_REGISTRY_LOCK:
            reference = _ACTIVE_WRITERS.get(self._directory_identity)
            if reference is not None and reference() is self:
                _ACTIVE_WRITERS.pop(self._directory_identity, None)

    def _release_lock(self) -> None:
        if (
            getattr(self, "_owner_pid", os.getpid()) != os.getpid()
            or getattr(self, "_superseded", False)
        ):
            self._lock_fd = None
            self._lock_fd_marker = None
            return
        # Keep the first trace boundary inside the exception table.
        try: self._release_lock_once()
        except BaseException:
            self._reconcile_lock_release()
            raise

    def _release_lock_safely(self) -> None:
        """Give interrupted reconciliation one final idempotent cleanup pass."""
        try: self._release_lock()
        except BaseException:
            if (
                getattr(self, "_owner_pid", os.getpid()) != os.getpid()
                or getattr(self, "_superseded", False)
            ):
                self._lock_fd = None
                self._lock_fd_marker = None
            else:
                self._reconcile_lock_release()
            raise

    def _release_lock_once(self) -> None:
        descriptor = getattr(self, "_lock_fd", None)
        if descriptor is None:
            return
        os.closerange(descriptor, descriptor + 1)
        self._lock_fd = None
        self._lock_fd_marker = None
        _PENDING_LOCK_FDS.discard(descriptor)

    def _reconcile_lock_release(self) -> None:
        descriptor = getattr(self, "_lock_fd", None)
        if descriptor is None:
            return
        # This fd is private and release runs under _state_lock.  Artifact
        # code cannot replace it between this validation and closerange; the
        # marker distinguishes only an ambiguous completed close/reuse.
        if _matches_lock_descriptor(
            descriptor,
            self._directory_identity,
            self._lock_fd_marker,
        ):
            os.closerange(descriptor, descriptor + 1)
        self._lock_fd = None
        self._lock_fd_marker = None
        _PENDING_LOCK_FDS.discard(descriptor)

    def _resume(self, state) -> None:
        if not isinstance(state, dict):
            raise ValueError("partial artifact manifest must be a JSON object")
        _require_schema(state, label="partial artifact manifest")
        stored_shard_size = state.get("shard_size")
        if type(stored_shard_size) is not int or stored_shard_size <= 0:
            raise ValueError("partial artifact shard_size must be a positive integer")
        if stored_shard_size != self.shard_size:
            raise ValueError("cannot resume with a different shard_size")

        stored_metadata = {
            key: value for key, value in state.items() if key not in _MANIFEST_KEYS
        }
        _ensure_exact_mapping(
            stored_metadata,
            self.metadata,
            label="partial artifact",
        )

        shards = _validate_shard_sequence(
            state.get("shards"), label="partial artifact"
        )
        shard_sha256 = _validate_shard_digests(
            state.get("shard_sha256"),
            shards,
            label="partial artifact",
        )
        record_count = state.get("record_count")
        if (
            isinstance(record_count, bool)
            or not isinstance(record_count, Integral)
            or record_count < 0
        ):
            raise ValueError("partial artifact record_count must be non-negative")

        published_keys: set[tuple[str, int]] = set()
        actual_count = 0
        for name in shards:
            records = _safe_load_shard(
                self.directory,
                name,
                shard_sha256[name],
                directory_fd=self._lock_fd,
            )
            actual_count += len(records)
            for record in records:
                _validate_safe_artifact_value(record)
                key = _record_key(record)
                if key in published_keys:
                    raise ValueError(
                        f"duplicate record key in partial artifact: {key!r}"
                    )
                published_keys.add(key)
        if actual_count != int(record_count):
            raise ValueError(
                "partial artifact record_count does not match its published shards"
            )

        self.shards = list(shards)
        self.shard_sha256 = shard_sha256
        self.record_count = int(record_count)
        self._published_keys = published_keys

    def _partial_value(
        self,
        *,
        shards: list[str] | None = None,
        shard_sha256: dict[str, str] | None = None,
    ) -> dict:
        return {
            **self.metadata,
            "schema_version": SCHEMA_VERSION,
            "shard_size": self.shard_size,
            "record_count": self.record_count,
            "shards": self.shards if shards is None else shards,
            "shard_sha256": (
                self.shard_sha256 if shard_sha256 is None else shard_sha256
            ),
        }

    def _publish_partial(
        self,
        *,
        shards: list[str] | None = None,
        shard_sha256: dict[str, str] | None = None,
    ) -> None:
        with self._state_lock:
            self._require_active()
            descriptor = self._lock_fd
            assert descriptor is not None
            _atomic_json_at(
                self._partial_value(
                    shards=shards,
                    shard_sha256=shard_sha256,
                ),
                self.partial.name,
                descriptor,
            )

    def existing_keys(self) -> set[tuple[str, int]]:
        with self._state_lock:
            return set(self._published_keys)

    def add(self, record: dict) -> None:
        with self._state_lock:
            self._require_active()
            if type(record) is not dict:
                raise TypeError("artifact records must be dictionaries")
            _validate_safe_artifact_value(record)
            key = _record_key(record)
            if key in self._published_keys or key in self._buffer_keys:
                raise ValueError(f"duplicate record key: {key!r}")

            self.buffer.append(dict(record))
            self._buffer_keys.add(key)
            self.record_count += 1
            if len(self.buffer) >= self.shard_size:
                self._flush()

    def _flush(self) -> None:
        with self._state_lock:
            self._require_active()
            if not self.buffer:
                return
            for record in self.buffer:
                _validate_safe_artifact_value(record)
            name = f"shard_{len(self.shards):05d}.pt"
            descriptor = self._lock_fd
            assert descriptor is not None
            _atomic_torch_at(self.buffer, name, descriptor)
            next_shards = [*self.shards, name]
            next_sha256 = {
                **self.shard_sha256,
                name: _sha256_file_at(descriptor, name),
            }
            self._publish_partial(
                shards=next_shards,
                shard_sha256=next_sha256,
            )
            self.shards = next_shards
            self.shard_sha256 = next_sha256
            self._published_keys.update(self._buffer_keys)
            self.buffer = []
            self._buffer_keys = set()

    def close(self) -> None:
        try:
            with self._state_lock:
                if self._closed:
                    return
                try:
                    self._flush()
                    descriptor = self._lock_fd
                    assert descriptor is not None
                    _atomic_json_at(
                        {
                            **self.metadata,
                            "schema_version": SCHEMA_VERSION,
                            "record_count": self.record_count,
                            "shards": self.shards,
                            "shard_sha256": self.shard_sha256,
                        },
                        self.final.name,
                        descriptor,
                    )
                    _unlink_at(descriptor, self.partial.name)
                except BaseException:
                    self._closed = True
                    self._release_lock_safely()
                    raise
                self._closed = True
                self._release_lock_safely()
        finally:
            if getattr(self, "_closed", False):
                self._unregister()

    def __enter__(self) -> "ShardWriter":
        with self._state_lock:
            self._require_active()
            return self

    def __exit__(self, exc_type, *_exc) -> None:
        if exc_type is None:
            self.close()
            return
        try:
            with self._state_lock:
                if self._closed:
                    return
                self.buffer = []
                self._buffer_keys = set()
                self._closed = True
                self._release_lock_safely()
        finally:
            if getattr(self, "_closed", False):
                self._unregister()

    def __del__(self) -> None:
        try:
            state_lock = getattr(self, "_state_lock", None)
            if state_lock is None:
                self._closed = True
                self._release_lock_safely()
            else:
                with state_lock:
                    self._closed = True
                    self._release_lock_safely()
            self._unregister()
        except Exception:
            pass


def _prepare_writer_registry_for_fork() -> None:
    global _FORK_LOCKED_WRITERS

    _WRITER_REGISTRY_LOCK.acquire()
    locked: list[ShardWriter] = []
    try:
        candidates = [
            *(reference() for reference in _ACTIVE_WRITERS.values()),
            *_CONSTRUCTING_WRITERS,
        ]
        for writer in candidates:
            if writer is None or writer._owner_pid != os.getpid():
                continue
            if any(writer is existing for existing in locked):
                continue
            writer._state_lock.acquire()
            locked.append(writer)
    except BaseException:
        for writer in reversed(locked):
            writer._state_lock.release()
        _WRITER_REGISTRY_LOCK.release()
        raise
    _FORK_LOCKED_WRITERS = locked


def _release_writer_registry_after_fork() -> None:
    global _FORK_GENERATION, _FORK_LOCKED_WRITERS

    _FORK_GENERATION += 1
    for writer in reversed(_FORK_LOCKED_WRITERS):
        writer._state_lock.release()
    _FORK_LOCKED_WRITERS = []
    _WRITER_REGISTRY_LOCK.release()


def _reset_writer_registry_after_fork() -> None:
    """Drop inherited process-local ownership without unlocking the parent."""
    global _ACTIVE_WRITERS, _CONSTRUCTING_WRITERS, _FORK_GENERATION
    global _FORK_LOCKED_WRITERS
    global _PENDING_LOCK_FDS
    global _TRANSFER_PARTICIPANTS, _WRITER_REGISTRY_LOCK

    _FORK_GENERATION += 1
    writers: list[ShardWriter] = []
    for candidate in [*_FORK_LOCKED_WRITERS, *_TRANSFER_PARTICIPANTS]:
        if not any(candidate is writer for writer in writers):
            writers.append(candidate)

    descriptors = set(_PENDING_LOCK_FDS)
    descriptors.update(
        writer._lock_fd for writer in writers if writer._lock_fd is not None
    )
    for descriptor in descriptors:
        try:
            os.close(descriptor)
        except OSError:
            pass

    for writer in writers:
        writer._lock_fd = None
        writer._lock_fd_marker = None
        writer._closed = True
        writer._superseded = True
        writer._state_lock = threading.RLock()
    _ACTIVE_WRITERS = {}
    _CONSTRUCTING_WRITERS = []
    _FORK_LOCKED_WRITERS = []
    _PENDING_LOCK_FDS = set()
    _TRANSFER_PARTICIPANTS = []
    _WRITER_REGISTRY_LOCK = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_prepare_writer_registry_for_fork,
        after_in_parent=_release_writer_registry_after_fork,
        after_in_child=_reset_writer_registry_after_fork,
    )


def load_manifest(directory: str | Path) -> dict:
    value = json.loads(
        (Path(directory) / "manifest.json").read_text(encoding="utf-8")
    )
    if type(value) is not dict:
        raise ValueError("artifact manifest must be a JSON object")
    _require_schema(value, label="artifact manifest")
    return value


def iter_records(directory: str | Path) -> Iterator[dict]:
    root = Path(directory)
    manifest = load_manifest(root)
    shards = _validate_shard_sequence(
        manifest.get("shards"), label="artifact manifest"
    )
    shard_sha256 = _validate_shard_digests(
        manifest.get("shard_sha256"),
        shards,
        label="artifact manifest",
    )
    expected_count = manifest.get("record_count")
    if (
        type(expected_count) is not int
        or expected_count < 0
    ):
        raise ValueError("artifact manifest record_count must be non-negative")

    actual_count = 0
    record_keys: set[tuple[str, int]] = set()
    for name in shards:
        records = _safe_load_shard(root, name, shard_sha256[name])
        actual_count += len(records)
        for record in records:
            _validate_safe_artifact_value(record)
            key = _record_key(record)
            if key in record_keys:
                raise ValueError(f"duplicate record key in artifact: {key!r}")
            record_keys.add(key)
            yield record
    if actual_count != expected_count:
        raise ValueError("artifact record_count does not match its shards")
