from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from numbers import Integral
from pathlib import Path

import torch


SCHEMA_VERSION = 1
_MANIFEST_KEYS = frozenset(
    {"schema_version", "shard_size", "record_count", "shards"}
)


def _canonical(value) -> str:
    """Return canonical text after applying JSON's on-disk normalization."""
    normalized = json.loads(
        json.dumps(value, separators=(",", ":"), allow_nan=False)
    )
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


def _temporary_path(target: Path) -> Path:
    return target.with_name(target.name + ".tmp")


def atomic_json(value: Mapping, path: str | Path) -> None:
    """Atomically publish a JSON mapping at ``path``."""
    if not isinstance(value, Mapping):
        raise TypeError("atomic JSON values must be mappings")
    text = json.dumps(
        dict(value), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(target)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_torch(value, path: str | Path) -> None:
    """Atomically publish a Torch value at ``path``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(target)
    try:
        torch.save(value, temporary)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def sha256_file(
    value: str | Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Hash file content without loading the complete file into memory."""
    size = _positive_integer(chunk_size, name="chunk_size")
    digest = hashlib.sha256()
    with Path(value).open("rb") as handle:
        while chunk := handle.read(size):
            digest.update(chunk)
    return digest.hexdigest()


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
    expected_dict = dict(expected)
    _canonical(expected_dict)
    path = Path(run_directory) / "artifacts" / "provenance.json"
    if not path.exists():
        atomic_json(expected_dict, path)
        return path

    actual = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(actual, dict):
        raise ValueError("run provenance must contain a JSON object")
    _ensure_exact_mapping(actual, expected_dict, label="run provenance")
    return path


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


def _safe_load_shard(root: Path, name: str) -> list[dict]:
    records = torch.load(
        _shard_path(root, name), map_location="cpu", weights_only=True
    )
    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        raise ValueError(f"artifact shard {name!r} must contain a list of records")
    return records


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
        self.metadata = dict(metadata)
        reserved = sorted(_MANIFEST_KEYS.intersection(self.metadata))
        if reserved:
            raise ValueError(f"artifact metadata uses reserved key: {reserved[0]}")
        _canonical(self.metadata)

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.final = self.directory / "manifest.json"
        self.partial = self.directory / "partial_manifest.json"
        self.buffer: list[dict] = []
        self._buffer_keys: set[tuple[str, int]] = set()
        self._closed = False

        if self.final.exists():
            raise FileExistsError(f"artifact is already complete: {self.final}")
        if self.partial.exists():
            state = json.loads(self.partial.read_text(encoding="utf-8"))
            self._resume(state)
        else:
            self.shards: list[str] = []
            self.record_count = 0
            self._published_keys: set[tuple[str, int]] = set()
            self._publish_partial()

    def _resume(self, state) -> None:
        if not isinstance(state, dict):
            raise ValueError("partial artifact manifest must be a JSON object")
        schema_version = state.get("schema_version")
        if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
            raise ValueError("partial artifact mismatch for schema_version")
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
            records = _safe_load_shard(self.directory, name)
            actual_count += len(records)
            for record in records:
                key = _record_key(record)
                if key in published_keys:
                    raise ValueError(f"duplicate record key in partial artifact: {key!r}")
                published_keys.add(key)
        if actual_count != int(record_count):
            raise ValueError(
                "partial artifact record_count does not match its published shards"
            )

        self.shards = list(shards)
        self.record_count = int(record_count)
        self._published_keys = published_keys

    def _partial_value(
        self,
        *,
        shards: list[str] | None = None,
    ) -> dict:
        return {
            **self.metadata,
            "schema_version": SCHEMA_VERSION,
            "shard_size": self.shard_size,
            "record_count": self.record_count,
            "shards": self.shards if shards is None else shards,
        }

    def _publish_partial(self, *, shards: list[str] | None = None) -> None:
        atomic_json(self._partial_value(shards=shards), self.partial)

    def existing_keys(self) -> set[tuple[str, int]]:
        return set(self._published_keys)

    def add(self, record: dict) -> None:
        if self._closed:
            raise RuntimeError("cannot add to a closed artifact writer")
        if not isinstance(record, dict):
            raise TypeError("artifact records must be dictionaries")
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
        name = f"shard_{len(self.shards):05d}.pt"
        atomic_torch(self.buffer, self.directory / name)
        next_shards = [*self.shards, name]
        self._publish_partial(shards=next_shards)
        self.shards = next_shards
        self._published_keys.update(self._buffer_keys)
        self.buffer = []
        self._buffer_keys = set()

    def close(self) -> None:
        if self._closed:
            return
        self._flush()
        atomic_json(
            {
                **self.metadata,
                "schema_version": SCHEMA_VERSION,
                "record_count": self.record_count,
                "shards": self.shards,
            },
            self.final,
        )
        self.partial.unlink(missing_ok=True)
        self._closed = True

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


def load_manifest(directory: str | Path) -> dict:
    value = json.loads(
        (Path(directory) / "manifest.json").read_text(encoding="utf-8")
    )
    if not isinstance(value, dict):
        raise ValueError("artifact manifest must be a JSON object")
    return value


def iter_records(directory: str | Path) -> Iterator[dict]:
    root = Path(directory)
    manifest = load_manifest(root)
    shards = _validate_shard_sequence(
        manifest.get("shards"), label="artifact manifest"
    )
    expected_count = manifest.get("record_count")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, Integral)
        or expected_count < 0
    ):
        raise ValueError("artifact manifest record_count must be non-negative")

    actual_count = 0
    for name in shards:
        records = _safe_load_shard(root, name)
        actual_count += len(records)
        yield from records
    if actual_count != int(expected_count):
        raise ValueError("artifact record_count does not match its shards")
