from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, order=True)
class ManifestEntry:
    image_id: str
    path: Path


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
        seen_ids.add(image_id)
        seen_paths.add(path)
        entries.append(ManifestEntry(image_id, path))

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


def manifest_digest(entries: Iterable[ManifestEntry]) -> str:
    canonical_entries = sorted(entries, key=lambda item: (item.image_id, str(item.path)))
    payload = [
        {"image_id": item.image_id, "path": str(item.path)}
        for item in canonical_entries
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
