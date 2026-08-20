from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

import torch


SCHEMA_VERSION = 1


def _dumps(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _canonical(value) -> str:
    """Canonical JSON text for `value`, invariant under a manifest round trip.

    Manifests are persisted as JSON, which rewrites int keys as strings and tuples
    as lists, so a live metadata value and the copy reloaded from disk are only
    comparable once both have been through the same normalisation. Dumping twice is
    what makes that normalisation idempotent: the first dump applies JSON's key
    coercion and the second sorts the coerced keys, so `{0: ..., 10: ...}` and
    `{"0": ..., "10": ...}` agree on order instead of one side sorting int keys
    numerically (9, 10) while the other sorts their strings lexicographically
    ("10", "9").

    Deliberately no `default=` fallback: metadata that JSON cannot encode cannot be
    persisted either, so raising here fails the run at writer construction instead
    of at the end of a multi-hour extraction, and stringifying such values would let
    two different ones (two large tensors print identically) pass as compatible.
    """
    return _dumps(json.loads(_dumps(value)))


def manifest_id(manifest: Mapping) -> str:
    payload = {key: value for key, value in manifest.items() if key != "artifact_id"}
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _atomic_torch_save(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json_save(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


class ShardWriter:
    def __init__(self, directory: str | Path, metadata: Mapping, shard_size: int = 50) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata)
        self.shard_size = int(shard_size)
        self.buffer: list[dict] = []
        self.final_manifest = self.directory / "manifest.json"
        self.partial_manifest = self.directory / "partial_manifest.json"
        if self.final_manifest.exists():
            raise FileExistsError(f"Artifact is already complete: {self.final_manifest}")
        if self.partial_manifest.exists():
            partial = json.loads(self.partial_manifest.read_text(encoding="utf-8"))
            assert_compatible(partial, self.metadata, keys=self.metadata)
            if partial["shard_size"] != self.shard_size:
                raise ValueError("Cannot resume with a different shard_size")
            self.shards = list(partial["shards"])
            self.record_count = int(partial["record_count"])
        else:
            self.shards = []
            self.record_count = 0
            self._write_partial_manifest()

    def __enter__(self):
        return self

    def add(self, record: dict) -> None:
        self.buffer.append(record)
        self.record_count += 1
        if len(self.buffer) >= self.shard_size:
            self._flush()

    def _write_partial_manifest(self) -> None:
        _atomic_json_save({
            "schema_version": SCHEMA_VERSION,
            **self.metadata,
            "shard_size": self.shard_size,
            "record_count": self.record_count,
            "shards": self.shards,
        }, self.partial_manifest)

    def existing_record_keys(self) -> set[tuple[int, int]]:
        keys = set()
        for shard in self.shards:
            # `weights_only=True` throughout this module. Artifact directories are routinely
            # copied between hosts, so an unpickling read is arbitrary code execution on data
            # from somewhere else, and nothing here needs it: a record is tensors, ints,
            # floats, strs, None and an int-keyed dict of tensors, all of which the safe
            # loader restores unchanged (checked against the pilot's caches and banks). Only
            # `runtime.py` keeps `weights_only=False`, for the EMA checkpoint that needs it.
            records = torch.load(self.directory / shard, map_location="cpu", weights_only=True)
            keys.update((int(record["image_id"]), int(record.get("severity", 0))) for record in records)
        return keys

    def _flush(self) -> None:
        if not self.buffer:
            return
        name = f"shard_{len(self.shards):05d}.pt"
        _atomic_torch_save(self.buffer, self.directory / name)
        self.shards.append(name)
        self.buffer = []
        self._write_partial_manifest()

    def close(self) -> None:
        self._flush()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            **self.metadata,
            "record_count": self.record_count,
            "shards": self.shards,
        }
        manifest["artifact_id"] = manifest_id(manifest)
        _atomic_json_save(manifest, self.final_manifest)
        self.partial_manifest.unlink(missing_ok=True)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()


def load_manifest(directory: str | Path) -> dict:
    return json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))


def iter_records(directory: str | Path) -> Iterator[dict]:
    root = Path(directory)
    manifest = load_manifest(root)
    for shard in manifest["shards"]:
        # Safe load; see `ShardWriter.existing_record_keys`.
        yield from torch.load(root / shard, map_location="cpu", weights_only=True)


def assert_compatible(actual: Mapping, expected: Mapping, keys: Iterable[str]) -> None:
    for key in keys:
        if _canonical(actual.get(key)) != _canonical(expected.get(key)):
            raise ValueError(
                f"Artifact mismatch for {key}: actual={actual.get(key)!r}, expected={expected.get(key)!r}"
            )
