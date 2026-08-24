from __future__ import annotations

import ctypes
import errno
import fcntl
import functools
import hashlib
import io
import json
import operator
import os
import secrets
import stat
import threading
import weakref
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from numbers import Integral
from pathlib import Path

import torch


SCHEMA_VERSION = 1
_MANIFEST_KEYS = frozenset(
    {"schema_version", "shard_size", "record_count", "shards", "shard_sha256"}
)
_ACTIVE_WRITERS: dict[tuple[int, int], weakref.ReferenceType] = {}
_REGULAR_FILE_ACQUISITION = threading.local()
_LIVE_READER_LEASES: dict[weakref.ReferenceType, weakref.ReferenceType] = {}
_WRITER_REGISTRY_LOCK = threading.RLock()
_FORK_LOCKED_WRITERS: list["ShardWriter"] = []
_CONSTRUCTING_WRITERS: list["ShardWriter"] = []
_PENDING_LOCK_LEASES: set["_CandidateLease"] = set()
_PENDING_LOCK_FDS: set[int] = set()
_TRANSFER_PARTICIPANTS: list["ShardWriter"] = []
_FORK_GENERATION = 0
_DIRECTORY_ANCHOR = ".artifact_directory_anchor"
_DIRECTORY_OPEN_ATTEMPTS = 3


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


def _unlink_staging(temporary, *, directory_fd: int | None) -> None:
    try:
        if directory_fd is None:
            Path(temporary).unlink()
        else:
            os.unlink(temporary, dir_fd=directory_fd)
    except FileNotFoundError:
        pass


def _stat_entry(path, *, directory_fd: int | None):
    if directory_fd is None:
        return os.stat(path, follow_symlinks=False)
    return os.stat(path, dir_fd=directory_fd, follow_symlinks=False)


def _require_staging_entry(handle, temporary, *, directory_fd: int | None) -> None:
    try:
        opened = os.fstat(handle.fileno())
        entry = _stat_entry(temporary, directory_fd=directory_fd)
    except OSError as error:
        raise ValueError("artifact staging entry changed") from error
    if (
        not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(entry.st_mode)
        or (opened.st_dev, opened.st_ino) != (entry.st_dev, entry.st_ino)
    ):
        raise ValueError("artifact staging entry changed")


@contextmanager
def _staged_file(
    target: str | Path,
    *,
    directory_fd: int | None = None,
):
    """Yield an exclusively created C-owned binary staging file."""
    if directory_fd is None:
        requested = Path(target)
        target_name = requested.name
        parent = requested.parent
    else:
        target_name = os.fspath(target)
        if not target_name or Path(target_name).name != target_name:
            raise ValueError(f"invalid artifact entry name: {target_name!r}")
        parent = Path(f"/proc/self/fd/{directory_fd}")

    for _attempt in range(100):
        temporary_name = f".{target_name}.{secrets.token_hex(16)}.tmp"
        temporary_path = parent / temporary_name
        handle = None
        collision = False
        try:
            try:
                handle = open(temporary_path, "xb")
            except FileExistsError:
                collision = True
                continue
            os.fchmod(handle.fileno(), 0o600)
            yield handle, (
                temporary_path if directory_fd is None else temporary_name
            )
            return
        finally:
            try:
                if handle is not None:
                    handle.close()
            finally:
                if not collision:
                    _unlink_staging(
                        temporary_path
                        if directory_fd is None
                        else temporary_name,
                        directory_fd=directory_fd,
                    )
    raise FileExistsError("could not allocate a unique artifact staging file")


def _fsync_directory(directory: Path) -> None:
    with _WRITER_REGISTRY_LOCK:
        owner = _DirectoryHandle(directory, ensure_anchor=False)
        try:
            descriptor = owner.open()
            owner.ownership(descriptor)
            os.fsync(descriptor)
        finally:
            owner.close()


def atomic_json(value: Mapping, path: str | Path) -> None:
    """Atomically publish a JSON mapping at the requested path."""
    if not isinstance(value, Mapping):
        raise TypeError("atomic JSON values must be mappings")
    content = (
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _staged_file(target) as (handle, temporary):
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        _require_staging_entry(handle, temporary, directory_fd=None)
        os.replace(temporary, target)
    _fsync_directory(target.parent)


def _atomic_json_create(value: Mapping, path: str | Path) -> bool:
    """Atomically publish JSON only if the target does not already exist."""
    if not isinstance(value, Mapping):
        raise TypeError("atomic JSON values must be mappings")
    content = (
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _staged_file(target) as (handle, temporary):
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        _require_staging_entry(handle, temporary, directory_fd=None)
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError:
            return False
        Path(temporary).unlink()
    _fsync_directory(target.parent)
    return True


def atomic_torch(value, path: str | Path) -> None:
    """Atomically publish a Torch value at the requested path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _staged_file(target) as (handle, temporary):
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
        _require_staging_entry(handle, temporary, directory_fd=None)
        os.replace(temporary, target)
    _fsync_directory(target.parent)


def _atomic_json_at(value: Mapping, name: str, directory_fd: int) -> None:
    content = (
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    with _staged_file(name, directory_fd=directory_fd) as (handle, temporary):
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        _require_staging_entry(
            handle,
            temporary,
            directory_fd=directory_fd,
        )
        os.replace(
            temporary,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
    os.fsync(directory_fd)


def _atomic_torch_at(value, name: str, directory_fd: int) -> None:
    with _staged_file(name, directory_fd=directory_fd) as (handle, temporary):
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
        _require_staging_entry(
            handle,
            temporary,
            directory_fd=directory_fd,
        )
        os.replace(
            temporary,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
    os.fsync(directory_fd)


def _path_exists_at(directory_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


@contextmanager
def _serialized_regular_file_acquisition():
    if getattr(_REGULAR_FILE_ACQUISITION, "active", False):
        raise RuntimeError("reentrant artifact file acquisition is not supported")
    try:
        _REGULAR_FILE_ACQUISITION.active = True
        yield
    finally:
        _REGULAR_FILE_ACQUISITION.active = False


def _prune_dead_live_readers() -> None:
    """Prune dead reader keys while the caller holds the registry lock."""
    for reference in list(_LIVE_READER_LEASES):
        if reference() is None:
            _LIVE_READER_LEASES.pop(reference, None)


def _register_live_reader(handle, lease: "_RegularFileLease") -> None:
    _prune_dead_live_readers()
    _LIVE_READER_LEASES[weakref.ref(handle)] = weakref.ref(lease)


_REQUIRED_REGULAR_FILE_FLAGS = (
    os.O_NONBLOCK | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
)


class _RegularFileOpenFlags:
    """Add safety flags in C before the opener creates any resource."""

    from_param = functools.partial(
        operator.or_, _REQUIRED_REGULAR_FILE_FLAGS
    )


_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC_OPEN = _LIBC.open
_LIBC_OPEN.argtypes = (ctypes.c_char_p, _RegularFileOpenFlags)
_LIBC_OPEN.restype = ctypes.c_int


def _open_regular_file_object(path: Path):
    """Open without exposing the descriptor to a Python ownership boundary.

    FileIO audits the path before calling its opener, then adopts the C
    opener's result inside the same C call.  Its C-implemented flag converter
    runs before libc creates the descriptor, so interruption cannot strand it
    or translate a cancellation exception.
    """
    return io.FileIO(
        os.fsencode(path),
        mode="rb",
        closefd=True,
        opener=_LIBC_OPEN,
    )


class _RegularFileLease:
    """Own one safely adopted regular-file reader through validation."""

    def __init__(self, identity: tuple[int, int]):
        self.identity = identity
        self.handle = None
        self.owner_pid = os.getpid()
        self.released = False
        self.fork_generation = _FORK_GENERATION

    def _valid_process(self) -> bool:
        return (
            self.owner_pid == os.getpid()
            and self.fork_generation == _FORK_GENERATION
        )

    def close(self) -> None:
        if self.released:
            return
        handle = self.handle
        self.handle = None
        self.released = True
        if handle is not None:
            handle.close()

    def acquire(
        self,
        visible: Path,
        entry,
        *,
        directory_fd: int | None,
        error_message: str,
    ) -> None:
        handle = None
        try:
            try:
                handle = _open_regular_file_object(visible)
            except ValueError as error:
                raise ValueError(error_message) from error
            self.handle = handle
            if not self._valid_process():
                raise RuntimeError("artifact file acquisition crossed a fork")
            _register_live_reader(handle, self)
            opened = os.fstat(handle.fileno())
            try:
                current = _stat_entry(entry, directory_fd=directory_fd)
            except OSError as error:
                raise ValueError(f"{error_message}; entry changed") from error
            opened_identity = (opened.st_dev, opened.st_ino)
            if (
                not stat.S_ISREG(opened.st_mode)
                or not stat.S_ISREG(current.st_mode)
                or opened_identity != self.identity
                or opened_identity != (current.st_dev, current.st_ino)
            ):
                raise ValueError(f"{error_message}; entry changed")
            if not self._valid_process():
                raise RuntimeError("artifact file acquisition crossed a fork")
        except BaseException:
            self.close()
            raise

    def release(self):
        handle = self.handle
        if handle is None:
            raise RuntimeError("artifact file lease has no open handle")
        if not self._valid_process():
            self.close()
            raise RuntimeError("artifact file acquisition crossed a fork")
        self.released = True
        self.handle = None
        return handle

def _open_regular_file(
    path: str | Path,
    *,
    directory_fd: int | None = None,
    error_message: str,
):
    if directory_fd is None:
        visible = Path(path)
        entry_name = None
    else:
        entry_name = os.fspath(path)
        if not entry_name or Path(entry_name).name != entry_name:
            raise ValueError(error_message)
        visible = Path(f"/proc/self/fd/{directory_fd}") / entry_name
    entry = visible if directory_fd is None else entry_name
    with (
        _WRITER_REGISTRY_LOCK, _serialized_regular_file_acquisition()
    ):
        try:
            before = _stat_entry(
                entry,
                directory_fd=directory_fd,
            )
        except OSError as error:
            raise ValueError(error_message) from error
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(error_message)
        lease = _RegularFileLease((before.st_dev, before.st_ino))
        try:
            lease.acquire(
                visible,
                entry,
                directory_fd=directory_fd,
                error_message=error_message,
            )
            return lease.release()
        except OSError as error:
            lease.close()
            raise ValueError(error_message) from error
        except BaseException:
            lease.close()
            raise


def _read_json_at(directory_fd: int, name: str):
    message = f"artifact file {name!r} must be a regular file"
    with _open_regular_file(
        name,
        directory_fd=directory_fd,
        error_message=message,
    ) as handle:
        return json.load(handle)


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
    message = f"artifact file {name!r} must be a regular file"
    with _open_regular_file(
        name,
        directory_fd=directory_fd,
        error_message=message,
    ) as handle:
        return _sha256_stream(handle, 1024 * 1024)


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


def ensure_provenance(
    run_directory: str | Path,
    expected: Mapping,
    *,
    artifacts_directory: str | Path | None = None,
) -> Path:
    """Create run provenance or refuse any JSON-semantic mismatch."""
    if not isinstance(expected, Mapping):
        raise TypeError("expected provenance must be a mapping")
    expected_dict = _json_snapshot(dict(expected))
    artifact_root = (
        Path(run_directory) / "artifacts"
        if artifacts_directory is None
        else Path(artifacts_directory)
    )
    path = artifact_root / "provenance.json"
    if _atomic_json_create(expected_dict, path):
        return path

    with _open_regular_file(
        path, error_message="run provenance must be a regular file"
    ) as handle:
        actual = json.load(handle)
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
    message = f"artifact shard {name!r} must be a regular file"
    source = path if directory_fd is None else name
    with _open_regular_file(
        source,
        directory_fd=directory_fd,
        error_message=message,
    ) as handle:
        actual_sha256 = _sha256_stream(handle, 1024 * 1024)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"artifact shard {name!r} SHA-256 mismatch: "
                f"actual={actual_sha256}, expected={expected_sha256}"
            )
        handle.seek(0)
        records = torch.load(handle, map_location="cpu", weights_only=True)
    if type(records) is not list or not all(type(record) is dict for record in records):
        raise ValueError(f"artifact shard {name!r} must contain a list of records")
    return records


def _live_fd_snapshot() -> set[int]:
    """Return live process fds, excluding procfs's own transient listing fd."""
    descriptors: set[int] = set()
    for name in os.listdir("/proc/self/fd"):
        try:
            descriptor = int(name)
            os.fstat(descriptor)
        except (OSError, ValueError):
            continue
        descriptors.add(descriptor)
    return descriptors


def _directory_fd_positions(
    identity: tuple[int, int],
) -> dict[int, int]:
    """Read positions for live fds referring to one directory inode."""
    positions: dict[int, int] = {}
    for descriptor in _live_fd_snapshot():
        try:
            state = os.fstat(descriptor)
            if (state.st_dev, state.st_ino) != identity:
                continue
            lines = Path(f"/proc/self/fdinfo/{descriptor}").read_text(
                encoding="ascii"
            ).splitlines()
            position = next(
                int(line.split()[1])
                for line in lines
                if line.startswith("pos:")
            )
        except (OSError, StopIteration, ValueError):
            continue
        positions[descriptor] = position
    return positions


def _ensure_directory_anchor(directory: Path) -> None:
    """Keep scanners live after their one identification read."""
    try:
        os.mkdir(directory / _DIRECTORY_ANCHOR)
    except FileExistsError:
        # Any existing entry makes the directory nonempty, which is the only
        # property needed here.  Artifact readers ignore unlisted entries.
        pass


class _DirectoryHandle:
    """Retain C-level ownership of a directory fd across Python opcodes."""

    def __init__(self, directory: Path, *, ensure_anchor: bool = True) -> None:
        state = os.stat(directory)
        self._directory = directory
        self._identity = (state.st_dev, state.st_ino)
        self._owner = None
        self._descriptor: int | None = None
        self._planned_marker = _new_lock_marker()
        self._marker: int | None = None
        self._owner_pid = os.getpid()
        self._fork_generation = _FORK_GENERATION
        self._invalidated_by_fork = False
        self._ensure_anchor = ensure_anchor

    def open(self) -> int:
        for _attempt in range(_DIRECTORY_OPEN_ATTEMPTS):
            self._require_valid()
            state = os.stat(self._directory)
            identity = (state.st_dev, state.st_ino)
            if self._ensure_anchor:
                _ensure_directory_anchor(self._directory)
            # ScandirIterator is a C-level RAII owner.  If an async exception
            # lands after this call but before STORE_FAST, decref closes its
            # directory fd.
            owner = os.scandir(self._directory)
            self._owner = owner
            self._require_valid()
            before = _directory_fd_positions(identity)
            self._require_valid()
            try:
                next(owner)
            except StopIteration:
                # An empty path can result from replacement between anchoring
                # and scanner acquisition.  Exhaustion closes the iterator.
                owner.close()
                self._owner = None
                continue
            self._require_valid()
            after = _directory_fd_positions(identity)
            self._require_valid()
            candidates = [
                descriptor
                for descriptor, position in before.items()
                if after.get(descriptor) != position
                and descriptor in after
            ]
            if len(candidates) == 1:
                descriptor = candidates[0]
                self._identity = identity
                self._descriptor = descriptor
                self._require_valid()
                self._marker = _mark_lock_descriptor(
                    descriptor,
                    self._planned_marker,
                )
                self._require_valid()
                if self._fork_generation != _FORK_GENERATION:
                    self._fork_generation = _FORK_GENERATION
                return descriptor
            if not candidates:
                current = os.stat(self._directory)
                if (current.st_dev, current.st_ino) != identity:
                    owner.close()
                    self._owner = None
                    continue
            raise RuntimeError(
                "could not identify the owned artifact directory descriptor"
            )
        raise RuntimeError(
            "could not identify the owned artifact directory descriptor"
        )

    def _require_valid(self) -> None:
        if self._invalidated_by_fork or self._owner_pid != os.getpid():
            raise RuntimeError("candidate descriptor invalidated by fork")

    def plan_marker(self, marker: int) -> None:
        self._require_valid()
        if self._owner is not None or self._descriptor is not None:
            raise RuntimeError("candidate descriptor marker is already fixed")
        self._planned_marker = marker

    def ownership(self, descriptor: int) -> tuple[tuple[int, int], int]:
        self._require_valid()
        marker = self._marker
        if self._descriptor != descriptor or marker is None:
            raise RuntimeError("candidate descriptor invalidated by fork")
        if not _matches_lock_descriptor(descriptor, self._identity, marker):
            raise RuntimeError("candidate descriptor invalidated by fork")
        return self._identity, marker

    def close(self) -> None:
        owner = self._owner
        if owner is None:
            return
        owner.close()
        self._owner = None
        self._descriptor = None
        self._marker = None

    def refresh_after_parent_fork(self, generation: int) -> None:
        self._fork_generation = generation

    def invalidate_after_child_fork(self, *, retain_bound: bool = False) -> None:
        self._invalidated_by_fork = True
        if retain_bound and self._descriptor is not None:
            return
        try:
            self.close()
        except OSError:
            pass


class _CandidateLease:
    """Sole owner of a candidate fd until transactional writer transfer."""

    def __init__(self) -> None:
        self._descriptor: int | None = None
        self._opening_descriptor: int | None = None
        self._opened_pid: int | None = None
        self._fork_generation: int | None = None
        self._identity: tuple[int, int] | None = None
        self._marker: int | None = None
        self._planned_marker = _new_lock_marker()
        self._invalidated_by_fork = False
        self._directory_handle: _DirectoryHandle | None = None
        self._owner_pid = os.getpid()
        self._lock_acquired = False

    def open(self, directory: Path) -> None:
        self._require_valid()
        self._directory_handle = _DirectoryHandle(directory)
        try:
            self._require_valid()
            self._directory_handle.plan_marker(self._planned_marker)
            descriptor = self._directory_handle.open()
            self._require_valid()
        except BaseException:
            self.close()
            raise
        try: self._track_open_descriptor(descriptor)
        except BaseException:
            self.close()
            raise
        try: self._prepare_open_descriptor(descriptor)
        except BaseException:
            self.close()
            raise

    def _track_open_descriptor(self, descriptor: int) -> None:
        self._require_valid()
        handle = self._directory_handle
        if handle is None:
            state = os.fstat(descriptor)
            self._identity = (state.st_dev, state.st_ino)
            self._marker = _mark_lock_descriptor(
                descriptor,
                self._planned_marker,
            )
            self._opening_descriptor = descriptor
            return
        identity, marker = handle.ownership(descriptor)
        self._require_valid()
        self._identity = identity
        self._marker = marker
        self._opening_descriptor = descriptor

    def _prepare_open_descriptor(self, descriptor: int) -> None:
        self._require_valid()
        identity = self.identity
        marker = self.marker
        if not _matches_lock_descriptor(descriptor, identity, marker):
            self._invalidated_by_fork = True
            raise RuntimeError("candidate descriptor invalidated by fork")
        self._adopt_open_descriptor(descriptor, identity, marker)

    def _adopt_open_descriptor(
        self,
        descriptor: int,
        identity: tuple[int, int],
        marker: int,
    ) -> None:
        self._require_valid()
        self._opened_pid = os.getpid()
        self._fork_generation = _FORK_GENERATION
        self._identity = identity
        self._marker = marker
        self._descriptor = descriptor
        _PENDING_LOCK_FDS.add(descriptor)
        # Register the ownership-bearing lease last.  Once visible here, a
        # child fork hook may invalidate it and no adoption line can
        # resurrect the derived descriptor view afterward.
        _PENDING_LOCK_LEASES.add(self)
        self._require_valid()
        if self._descriptor != descriptor or self._opened_pid != os.getpid():
            raise RuntimeError("candidate descriptor invalidated by fork")

    def _require_valid(self) -> None:
        if self._invalidated_by_fork or self._owner_pid != os.getpid():
            raise RuntimeError("candidate descriptor invalidated by fork")
        handle = self._directory_handle
        if handle is not None:
            handle._require_valid()

    @property
    def descriptor(self) -> int:
        self._require_valid()
        descriptor = self._descriptor
        if descriptor is None or self._opened_pid != os.getpid():
            raise RuntimeError("candidate descriptor invalidated by fork")
        if self._fork_generation != _FORK_GENERATION:
            # An unregistered lease can miss the parent fork hook.  Matching
            # PIDs prove parent ownership; children fail the check above.
            self._fork_generation = _FORK_GENERATION
        return descriptor

    @property
    def identity(self) -> tuple[int, int]:
        if self._identity is None:
            raise RuntimeError("candidate directory identity is unavailable")
        return self._identity

    @property
    def marker(self) -> int:
        if self._marker is None:
            raise RuntimeError("candidate descriptor marker is unavailable")
        return self._marker


    def refresh_after_parent_fork(self, generation: int) -> None:
        if self._directory_handle is not None:
            self._directory_handle.refresh_after_parent_fork(generation)
        if self._descriptor is not None:
            self._fork_generation = generation

    def invalidate_after_child_fork(self) -> None:
        if self._directory_handle is not None:
            self._directory_handle.invalidate_after_child_fork(
                retain_bound=(
                    self._directory_handle._descriptor is not None
                    and not self._lock_acquired
                ),
            )
        self._descriptor = None
        self._opening_descriptor = None
        self._marker = None
        self._invalidated_by_fork = True

    def close(self) -> None:
        try: self._close_safely()
        except BaseException:
            self._reconcile_close()
            raise

    def _close_safely(self) -> None:
        try: self._close_once()
        except BaseException:
            self._reconcile_close()
            raise

    def _close_once(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            descriptor = self._opening_descriptor
        if descriptor is None:
            if self._directory_handle is not None:
                self._directory_handle.close()
                self._directory_handle = None
            _PENDING_LOCK_LEASES.discard(self)
            return
        if self._directory_handle is not None:
            self._directory_handle.close()
        elif self._descriptor is None:
            if self._invalidated_by_fork:
                self._consume(descriptor)
                return
            os.closerange(descriptor, descriptor + 1)
        else:
            os.close(descriptor)
        self._consume(descriptor)

    def _reconcile_close(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            descriptor = self._opening_descriptor
        if descriptor is None:
            if self._directory_handle is not None:
                self._directory_handle.close()
                self._directory_handle = None
            _PENDING_LOCK_LEASES.discard(self)
            return
        if self._directory_handle is not None:
            self._directory_handle.close()
        if (
            not self._invalidated_by_fork
            and _matches_lock_descriptor(
                descriptor,
                self._identity,
                self._marker,
            )
        ):
            os.closerange(descriptor, descriptor + 1)
        self._consume(descriptor)

    def _consume(self, descriptor: int) -> None:
        _PENDING_LOCK_LEASES.discard(self)
        _PENDING_LOCK_FDS.discard(descriptor)
        self._descriptor = None
        self._opening_descriptor = None
        self._marker = None
        self._directory_handle = None

    def transfer_to(self, writer: "ShardWriter") -> int:
        descriptor = self.descriptor
        marker = self.marker
        directory_handle = self._directory_handle
        try:
            writer._lock_fd = descriptor
            writer._lock_fd_marker = marker
            writer._lock_handle = directory_handle
            if (
                self._descriptor != descriptor
                or writer._owner_pid != os.getpid()
                or writer._superseded
            ):
                raise RuntimeError(
                    "artifact writer construction invalidated by fork"
                )
            self._consume(descriptor)
        except BaseException:
            if self._descriptor is not None:
                writer._lock_fd = None
                writer._lock_fd_marker = None
                writer._lock_handle = None
            raise
        if (
            writer._lock_fd != descriptor
            or writer._owner_pid != os.getpid()
            or writer._superseded
        ):
            raise RuntimeError("artifact writer construction invalidated by fork")
        return descriptor


def _open_directory(
    directory: Path,
    *,
    lease: _CandidateLease | None = None,
) -> tuple[_CandidateLease, tuple[int, int]]:
    if lease is None:
        lease = _CandidateLease()
    try:
        lease.open(directory)
    except BaseException:
        lease.close()
        raise
    return lease, lease.identity


def _new_lock_marker() -> int:
    return secrets.randbelow((1 << 62) - 1) + 1


def _mark_lock_descriptor(
    descriptor: int,
    marker: int | None = None,
) -> int:
    # ShardWriter never iterates this private directory fd; its seek offset is
    # therefore a stable open-file-description marker on supported Linux fds.
    marker = _new_lock_marker() if marker is None else marker
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
    lease: _CandidateLease | None = None,
) -> _CandidateLease:
    # The kernel owns this lock through the open descriptor, so process death
    # releases it without leaving a stale lock file that needs manual recovery.
    if lease is None:
        lease, _identity = _open_directory(directory)
    try:
        fcntl.flock(lease.descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lease._lock_acquired = True
    except BaseException as error:
        lease.close()
        if isinstance(error, OSError) and error.errno in (
            errno.EACCES,
            errno.EAGAIN,
        ):
            raise RuntimeError(
                f"artifact directory already has an active writer: {directory}"
            ) from error
        raise
    return lease


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
        *,
        anchored_directory: bool = False,
    ) -> None:
        if not isinstance(metadata, Mapping):
            raise TypeError("artifact metadata must be a mapping")
        self.shard_size = _positive_integer(shard_size, name="shard_size")
        self.metadata = _json_snapshot(dict(metadata))
        reserved = sorted(_MANIFEST_KEYS.intersection(self.metadata))
        if reserved:
            raise ValueError(f"artifact metadata uses reserved key: {reserved[0]}")

        self.directory = Path(directory)
        if anchored_directory:
            proc_root = Path("/proc/self/fd")
            if (
                self.directory.parent != proc_root
                or not self.directory.name.isascii()
                or not self.directory.name.isdecimal()
            ):
                raise ValueError(
                    "anchored artifact directory must be /proc/self/fd/<fd>"
                )
            try:
                state = os.fstat(int(self.directory.name))
                visible = os.stat(self.directory)
            except OSError as error:
                raise ValueError("anchored artifact directory is invalid") from error
            if (
                not stat.S_ISDIR(state.st_mode)
                or (state.st_dev, state.st_ino)
                != (visible.st_dev, visible.st_ino)
            ):
                raise ValueError("anchored artifact directory is invalid")
        else:
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
        self._lock_handle: _DirectoryHandle | None = None
        self._owner_pid = os.getpid()
        self._directory_identity: tuple[int, int] | None = None
        candidate_lease = _CandidateLease()
        self._candidate_lease: _CandidateLease | None = candidate_lease

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
                        candidate_lease,
                        self._directory_identity,
                    ) = _open_directory(
                        self.directory,
                        lease=candidate_lease,
                    )
                    candidate_lease = _acquire_directory_lock(
                        self.directory,
                        lease=candidate_lease,
                    )
                    if self._owner_pid != os.getpid() or self._superseded:
                        raise RuntimeError(
                            "artifact writer construction invalidated by fork"
                        )
                    descriptor = candidate_lease.transfer_to(self)
                    self._closed = False
                    _register_writer(self)
                    if self._owner_pid != os.getpid() or self._superseded:
                        raise RuntimeError(
                            "artifact writer construction invalidated by fork"
                        )

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
                if candidate_lease is not None:
                    candidate_lease.close()
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
            directory_handle = predecessor._lock_handle
            if marker is None or directory_handle is None:
                raise RuntimeError(
                    "active artifact writer has incomplete lock ownership"
                )
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
                self._lock_handle = directory_handle
                self._closed = False
                _register_writer(self)
                if self._owner_pid != os.getpid() or self._superseded:
                    raise RuntimeError(
                        "artifact writer construction invalidated by fork"
                    )
                predecessor._lock_fd = None
                predecessor._lock_fd_marker = None
                predecessor._lock_handle = None
                predecessor._closed = True
                predecessor._superseded = True
                return descriptor
            except BaseException:
                self._lock_fd = None
                self._lock_fd_marker = None
                self._lock_handle = None
                self._closed = True
                if self._owner_pid != os.getpid() or self._superseded:
                    predecessor._lock_fd = None
                    predecessor._lock_fd_marker = None
                    predecessor._lock_handle = None
                    predecessor._closed = True
                    predecessor._superseded = True
                    raise
                predecessor._lock_fd = descriptor
                predecessor._lock_fd_marker = marker
                predecessor._lock_handle = directory_handle
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
            self._lock_handle = None
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
                self._lock_handle = None
            else:
                self._reconcile_lock_release()
            raise

    def _release_lock_once(self) -> None:
        descriptor = getattr(self, "_lock_fd", None)
        if descriptor is None:
            return
        handle = getattr(self, "_lock_handle", None)
        if handle is None:
            os.closerange(descriptor, descriptor + 1)
        else:
            handle.close()
        self._lock_fd = None
        self._lock_fd_marker = None
        self._lock_handle = None
        _PENDING_LOCK_FDS.discard(descriptor)

    def _reconcile_lock_release(self) -> None:
        descriptor = getattr(self, "_lock_fd", None)
        if descriptor is None:
            return
        handle = getattr(self, "_lock_handle", None)
        if handle is not None:
            handle.close()
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
        self._lock_handle = None
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
        _prune_dead_live_readers()
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
    for lease_reference in list(_LIVE_READER_LEASES.values()):
        lease = lease_reference()
        if lease is not None and lease.owner_pid == os.getpid():
            lease.fork_generation = _FORK_GENERATION
    leases = list(_PENDING_LOCK_LEASES)
    for writer in _FORK_LOCKED_WRITERS:
        lease = getattr(writer, "_candidate_lease", None)
        if lease is not None and not any(lease is known for known in leases):
            leases.append(lease)
    for lease in leases:
        lease.refresh_after_parent_fork(_FORK_GENERATION)
    for writer in _FORK_LOCKED_WRITERS:
        handle = getattr(writer, "_lock_handle", None)
        if handle is not None:
            handle.refresh_after_parent_fork(_FORK_GENERATION)
    for writer in reversed(_FORK_LOCKED_WRITERS):
        writer._state_lock.release()
    _FORK_LOCKED_WRITERS = []
    _WRITER_REGISTRY_LOCK.release()


def _reset_writer_registry_after_fork() -> None:
    """Drop inherited process-local ownership without unlocking the parent."""
    global _ACTIVE_WRITERS, _CONSTRUCTING_WRITERS, _FORK_GENERATION
    global _FORK_LOCKED_WRITERS
    global _PENDING_LOCK_FDS, _PENDING_LOCK_LEASES
    global _TRANSFER_PARTICIPANTS, _WRITER_REGISTRY_LOCK
    global _LIVE_READER_LEASES

    _FORK_GENERATION += 1
    for handle_reference in list(_LIVE_READER_LEASES):
        handle = handle_reference()
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
    _LIVE_READER_LEASES = {}
    writers: list[ShardWriter] = []
    for candidate in [*_FORK_LOCKED_WRITERS, *_TRANSFER_PARTICIPANTS]:
        if not any(candidate is writer for writer in writers):
            writers.append(candidate)

    leases = list(_PENDING_LOCK_LEASES)
    for writer in writers:
        lease = getattr(writer, "_candidate_lease", None)
        if lease is not None and not any(lease is known for known in leases):
            leases.append(lease)
    descriptors = {
        descriptor
        for lease in leases
        if getattr(lease, "_directory_handle", None) is None
        if (descriptor := getattr(lease, "_descriptor", None)) is not None
    }
    descriptors.update(
        writer._lock_fd
        for writer in writers
        if getattr(writer, "_lock_handle", None) is None
        if writer._lock_fd is not None
    )
    for descriptor in descriptors:
        try:
            os.close(descriptor)
        except OSError:
            pass
    for lease in leases:
        lease.invalidate_after_child_fork()


    for writer in writers:
        handle = getattr(writer, "_lock_handle", None)
        if handle is not None:
            handle.invalidate_after_child_fork()
        writer._lock_fd = None
        writer._lock_fd_marker = None
        writer._lock_handle = None
        writer._closed = True
        writer._superseded = True
        writer._state_lock = threading.RLock()
    _ACTIVE_WRITERS = {}
    _CONSTRUCTING_WRITERS = []
    _FORK_LOCKED_WRITERS = []
    _PENDING_LOCK_LEASES = set()
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
    path = Path(directory) / "manifest.json"
    with _open_regular_file(
        path, error_message="artifact manifest must be a regular file"
    ) as handle:
        value = json.load(handle)
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
