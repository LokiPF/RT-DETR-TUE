from __future__ import annotations

import csv
import hashlib
import json
import os
import stat
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

from .artifacts import _open_regular_file


@dataclass(frozen=True, order=True)
class ImageFingerprint:
    sha256: str
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @property
    def identity(self) -> tuple[int, int]:
        return self.device, self.inode

    def as_dict(self) -> dict[str, int | str]:
        return {
            "sha256": self.sha256,
            "device": self.device,
            "inode": self.inode,
            "mode": self.mode,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
        }


def _signature(value) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _fingerprint_from_state(value, digest: str) -> ImageFingerprint:
    return ImageFingerprint(digest, *_signature(value))


def _digest(handle: BinaryIO) -> str:
    result = hashlib.sha256()
    while chunk := handle.read(1024 * 1024):
        result.update(chunk)
    return result.hexdigest()


@contextmanager
def _verified_open(path: Path, expected: ImageFingerprint | None = None):
    with _open_regular_file(
        path, error_message=f"image does not exist or is unsafe: {path}"
    ) as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"image must be a regular file: {path}")
        digest = _digest(handle)
        after_hash = os.fstat(handle.fileno())
        try:
            visible = os.stat(path, follow_symlinks=False)
        except OSError as error:
            raise ValueError(
                f"image changed while it was read: {path}"
            ) from error
        actual = _fingerprint_from_state(after_hash, digest)
        if (
            _signature(before) != _signature(after_hash)
            or _signature(after_hash) != _signature(visible)
            or (expected is not None and actual != expected)
        ):
            raise ValueError(f"image changed while it was read: {path}")
        handle.seek(0)
        try:
            yield handle, actual
        finally:
            final = os.fstat(handle.fileno())
            try:
                visible_final = os.stat(path, follow_symlinks=False)
            except OSError as error:
                raise ValueError(
                    f"image changed while it was used: {path}"
                ) from error
            if (
                _signature(final) != _signature(after_hash)
                or _signature(visible_final) != _signature(after_hash)
            ):
                raise ValueError(f"image changed while it was used: {path}")


def fingerprint_image(path: Path) -> ImageFingerprint:
    with _verified_open(path) as (_handle, fingerprint):
        return fingerprint


def validate_image_fingerprint(entry: ManifestEntry) -> None:
    if entry.fingerprint is None:
        raise ValueError("manifest entry has no image fingerprint")
    with _verified_open(entry.path, entry.fingerprint):
        pass


def validate_image_signature(entry: ManifestEntry) -> None:
    if entry.fingerprint is None:
        raise ValueError("manifest entry has no image fingerprint")
    with _open_regular_file(
        entry.path,
        error_message=f"image changed after it was audited: {entry.path}",
    ) as handle:
        current = os.fstat(handle.fileno())
        try:
            visible = os.stat(entry.path, follow_symlinks=False)
        except OSError as error:
            raise ValueError(
                f"image changed after it was audited: {entry.path}"
            ) from error
        expected = (
            entry.fingerprint.device,
            entry.fingerprint.inode,
            entry.fingerprint.mode,
            entry.fingerprint.size,
            entry.fingerprint.mtime_ns,
            entry.fingerprint.ctime_ns,
        )
        if _signature(current) != expected or _signature(visible) != expected:
            raise ValueError(
                f"image changed after it was audited: {entry.path}"
            )


@contextmanager
def open_fingerprinted_image(entry: ManifestEntry):
    if entry.fingerprint is None:
        raise ValueError("manifest entry has no image fingerprint")
    with _verified_open(entry.path, entry.fingerprint) as (handle, _fingerprint):
        yield handle


@dataclass(frozen=True, order=True)
class ManifestEntry:
    image_id: str
    path: Path
    fingerprint: ImageFingerprint | None = None


def load_manifest(value: str | Path) -> tuple[ManifestEntry, ...]:
    manifest = Path(value).resolve()
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["image_id", "image_path"]:
            raise ValueError("manifest columns must be exactly image_id,image_path")
        rows = list(reader)

    entries: list[ManifestEntry] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    seen_identities: set[tuple[int, int]] = set()
    for number, row in enumerate(rows, start=2):
        if (
            None in row
            or row.get("image_id") is None
            or row.get("image_path") is None
        ):
            raise ValueError(
                f"manifest row {number} is malformed: "
                "expected exactly image_id,image_path values"
            )
        image_id = row["image_id"].strip()
        if not image_id:
            raise ValueError(f"manifest row {number} has an empty image_id")
        if len(image_id) > 256:
            raise ValueError(
                f"manifest row {number} image_id must contain at most 256 characters"
            )
        if any(
            unicodedata.category(character).startswith("C")
            for character in image_id
        ):
            raise ValueError(
                f"manifest row {number} image_id must not contain control characters"
            )
        if image_id.startswith(("=", "+", "-", "@")):
            raise ValueError(
                f"manifest row {number} image_id starts with a "
                "spreadsheet formula character (=, +, -, or @)"
            )
        if image_id in seen_ids:
            raise ValueError(f"duplicate image_id {image_id!r}")
        path = (manifest.parent / row["image_path"]).resolve()
        if not path.is_file():
            raise ValueError(f"image for {image_id!r} does not exist: {path}")
        if path in seen_paths:
            raise ValueError(f"manifest repeats resolved image: {path}")
        try:
            fingerprint = fingerprint_image(path)
        except ValueError as error:
            raise ValueError(
                f"image for {image_id!r} does not exist or changed: {path}"
            ) from error
        if fingerprint.identity in seen_identities:
            raise ValueError(
                "manifest repeats image file identity through a hard link: "
                f"{path}"
            )
        seen_ids.add(image_id)
        seen_paths.add(path)
        seen_identities.add(fingerprint.identity)
        entries.append(ManifestEntry(image_id, path, fingerprint))

    if not entries:
        raise ValueError("manifest must contain at least one image")
    return tuple(sorted(entries, key=lambda entry: entry.image_id))


def validate_disjoint(
    reference: Iterable[ManifestEntry],
    evaluation: Iterable[ManifestEntry],
) -> None:
    reference_entries = tuple(reference)
    evaluation_entries = tuple(evaluation)
    ids = {entry.image_id for entry in reference_entries} & {
        entry.image_id for entry in evaluation_entries
    }
    if ids:
        raise ValueError(
            f"reference and evaluation repeat image_id values: {sorted(ids)}"
        )
    paths = {entry.path for entry in reference_entries} & {
        entry.path for entry in evaluation_entries
    }
    if paths:
        raise ValueError(
            "reference and evaluation contain the same resolved image: "
            f"{sorted(paths)}"
        )
    reference_identities = {
        entry.fingerprint.identity for entry in reference_entries
        if entry.fingerprint is not None
    }
    evaluation_identities = {
        entry.fingerprint.identity for entry in evaluation_entries
        if entry.fingerprint is not None
    }
    if reference_identities & evaluation_identities:
        raise ValueError(
            "reference and evaluation contain hard links to the same image"
        )


def manifest_digest(entries: Iterable[ManifestEntry]) -> str:
    canonical_entries = sorted(
        entries, key=lambda item: (item.image_id, str(item.path))
    )
    payload = [
        {
            "image_id": item.image_id,
            "path": str(item.path),
            "fingerprint": (
                item.fingerprint.as_dict() if item.fingerprint else None
            ),
        }
        for item in canonical_entries
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
