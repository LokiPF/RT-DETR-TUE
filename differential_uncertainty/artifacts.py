from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Iterator, Mapping
from numbers import Integral
from pathlib import Path

import torch


SCHEMA_VERSION = 1
_MANIFEST_KEYS = frozenset(
    {"schema_version", "shard_size", "record_count", "shards", "shard_sha256"}
)


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


def _sha256_stream(handle, chunk_size: int) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(chunk_size):
        digest.update(chunk)
    return digest.hexdigest()


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
) -> list[dict]:
    path = _shard_path(root, name)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None
    try:
        descriptor = os.open(path, flags)
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


def _acquire_directory_lock(directory: Path) -> int:
    # The kernel owns this lock through the open descriptor, so process death
    # releases it without leaving a stale lock file that needs manual recovery.
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        os.close(descriptor)
        if error.errno in (errno.EACCES, errno.EAGAIN):
            raise RuntimeError(
                f"artifact directory already has an active writer: {directory}"
            ) from error
        raise
    return descriptor


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
        self._lock_fd: int | None = _acquire_directory_lock(self.directory)
        self._closed = False
        try:
            self.final = self.directory / "manifest.json"
            self.partial = self.directory / "partial_manifest.json"
            self.buffer: list[dict] = []
            self._buffer_keys: set[tuple[str, int]] = set()

            if self.final.exists():
                raise FileExistsError(f"artifact is already complete: {self.final}")
            if self.partial.exists():
                state = json.loads(self.partial.read_text(encoding="utf-8"))
                self._resume(state)
            else:
                self.shards: list[str] = []
                self.shard_sha256: dict[str, str] = {}
                self.record_count = 0
                self._published_keys: set[tuple[str, int]] = set()
                self._publish_partial()
        except BaseException:
            self._release_lock()
            raise

    def _release_lock(self) -> None:
        descriptor = getattr(self, "_lock_fd", None)
        if descriptor is None:
            return
        self._lock_fd = None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

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
        atomic_json(
            self._partial_value(
                shards=shards,
                shard_sha256=shard_sha256,
            ),
            self.partial,
        )

    def existing_keys(self) -> set[tuple[str, int]]:
        return set(self._published_keys)

    def add(self, record: dict) -> None:
        if self._closed:
            raise RuntimeError("cannot add to a closed artifact writer")
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
        if not self.buffer:
            return
        for record in self.buffer:
            _validate_safe_artifact_value(record)
        name = f"shard_{len(self.shards):05d}.pt"
        path = self.directory / name
        atomic_torch(self.buffer, path)
        next_shards = [*self.shards, name]
        next_sha256 = {**self.shard_sha256, name: sha256_file(path)}
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
        if self._closed:
            return
        try:
            self._flush()
            atomic_json(
                {
                    **self.metadata,
                    "schema_version": SCHEMA_VERSION,
                    "record_count": self.record_count,
                    "shards": self.shards,
                    "shard_sha256": self.shard_sha256,
                },
                self.final,
            )
            self.partial.unlink(missing_ok=True)
        except BaseException:
            self._closed = True
            self._release_lock()
            raise
        self._closed = True
        self._release_lock()

    def __enter__(self) -> "ShardWriter":
        if self._closed:
            raise RuntimeError("cannot enter a closed artifact writer")
        return self

    def __exit__(self, exc_type, *_exc) -> None:
        if exc_type is None:
            self.close()
            return
        self.buffer = []
        self._buffer_keys = set()
        self._closed = True
        self._release_lock()

    def __del__(self) -> None:
        try:
            self._release_lock()
        except Exception:
            pass


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
