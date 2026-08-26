from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import os
import stat
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Callable

import numpy as np

from .artifacts import (
    _fsync_directory,
    _open_regular_file,
    _require_staging_entry,
    _sha256_stream,
    _staged_file,
)
from .config import FIXED_CONFIG, ExperimentConfig
from .extraction import RTDETRExtractor
from .manifests import (
    ManifestEntry,
    fingerprint_image,
    load_manifest,
    manifest_digest,
    validate_disjoint,
)
from .reporting import _DirectoryLease


_BENCHMARK_MANIFEST = "benchmark-manifest.json"
_EVALUATION_MANIFEST = "evaluation-manifest.csv"
_REFERENCE_MANIFEST = "reference-manifest.csv"
_SCHEMA_VERSION = 1
_DEFAULT_SPLIT_SEED = 20260825
_JOURNAL = "benchmark-manifest-journal.json"


@dataclass(frozen=True)
class CocoSplit:
    """Immutable, authenticated COCO input manifests for one benchmark run."""

    reference_manifest: Path
    evaluation_manifest: Path
    benchmark_manifest: Path
    reference_ids: tuple[int, ...]
    evaluation_ids: tuple[int, ...]
    reference_count: int
    evaluation_count: int
    split_seed: int


@dataclass(frozen=True)
class _CocoImage:
    image_id: int
    path: Path


def _positive_integer(value, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _split_seed(value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("split_seed must be a nonnegative integer")
    value = int(value)
    if value > np.iinfo(np.uint64).max:
        raise ValueError("split_seed is outside NumPy's supported range")
    return value


def _absolute_directory(value, *, name: str) -> Path:
    path = Path(os.path.abspath(os.fspath(value)))
    if path == path.parent:
        raise ValueError(f"{name} must not be the filesystem root")
    return path


def _signature(value) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _safe_annotation_payload(path: Path) -> tuple[object, str]:
    with _open_regular_file(
        path,
        error_message="COCO annotations must be a regular file",
    ) as handle:
        before = os.fstat(handle.fileno())
        try:
            payload = json.load(handle)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("COCO annotations must be valid JSON") from error
        handle.seek(0)
        digest = _sha256_stream(handle, 1024 * 1024)
        after = os.fstat(handle.fileno())
        try:
            visible = os.stat(path, follow_symlinks=False)
        except OSError as error:
            raise ValueError("COCO annotations changed while they were read") from error
    if (
        _signature(before) != _signature(after)
        or _signature(after) != _signature(visible)
        or not stat.S_ISREG(visible.st_mode)
    ):
        raise ValueError("COCO annotations changed while they were read")
    return payload, digest


def _safe_file_name(value, *, index: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"COCO image {index} file_name must be a string")
    if (
        not value
        or len(value) > 255
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or Path(value).name != value
        or any(unicodedata.category(character).startswith("C") for character in value)
    ):
        raise ValueError(f"COCO image {index} file_name must be a safe basename")
    return value


def _safe_image_id(value, *, index: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"COCO image {index} id must be a positive integer")
    value = int(value)
    if value <= 0 or value > (2**63 - 1):
        raise ValueError(f"COCO image {index} id must be a positive integer")
    return value


def _validated_coco_images(
    annotations: Path, image_directory: Path
) -> tuple[list[_CocoImage], str]:
    payload, annotation_digest = _safe_annotation_payload(annotations)
    if type(payload) is not dict or type(payload.get("images")) is not list:
        raise ValueError("COCO annotations must contain an images list")
    try:
        directory_stat = os.stat(image_directory, follow_symlinks=True)
    except OSError as error:
        raise ValueError("COCO image directory does not exist") from error
    if not stat.S_ISDIR(directory_stat.st_mode):
        raise ValueError("COCO image directory must be a directory")

    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    images: list[_CocoImage] = []
    for index, record in enumerate(payload["images"]):
        if type(record) is not dict:
            raise ValueError(f"COCO image {index} must be an object")
        image_id = _safe_image_id(record.get("id"), index=index)
        name = _safe_file_name(record.get("file_name"), index=index)
        if image_id in seen_ids:
            raise ValueError(f"COCO annotations repeat image id {image_id}")
        if name in seen_names:
            raise ValueError(f"COCO annotations repeat file_name {name!r}")
        image_path = image_directory / name
        try:
            image_stat = os.stat(image_path, follow_symlinks=False)
        except OSError as error:
            raise ValueError(
                f"COCO image for id {image_id} does not exist: {image_path}"
            ) from error
        if not stat.S_ISREG(image_stat.st_mode):
            raise ValueError(
                f"COCO image for id {image_id} must be a regular file: {image_path}"
            )
        seen_ids.add(image_id)
        seen_names.add(name)
        images.append(_CocoImage(image_id, image_path))

    if not images:
        raise ValueError("COCO annotations images list must not be empty")
    return sorted(images, key=lambda item: item.image_id), annotation_digest


def _select_cohorts(
    images: list[_CocoImage],
    *,
    reference_count: int,
    evaluation_count: int,
    split_seed: int,
) -> tuple[tuple[_CocoImage, ...], tuple[_CocoImage, ...]]:
    required = reference_count + evaluation_count
    if len(images) < required:
        raise ValueError(
            "not enough COCO images for requested reference and evaluation cohorts"
        )
    order = np.random.default_rng(split_seed).permutation(len(images))
    reference = tuple(
        sorted(
            (images[int(index)] for index in order[:reference_count]),
            key=lambda item: item.image_id,
        )
    )
    evaluation = tuple(
        sorted(
            (
                images[int(index)]
                for index in order[reference_count:required]
            ),
            key=lambda item: item.image_id,
        )
    )
    if {item.image_id for item in reference} & {
        item.image_id for item in evaluation
    }:
        raise RuntimeError("COCO selection produced overlapping cohorts")
    return reference, evaluation


def _manifest_entries(items: tuple[_CocoImage, ...]) -> tuple[ManifestEntry, ...]:
    entries = []
    for item in items:
        try:
            fingerprint = fingerprint_image(item.path)
        except ValueError as error:
            raise ValueError(
                f"COCO image for id {item.image_id} is missing or unsafe"
            ) from error
        entries.append(ManifestEntry(str(item.image_id), item.path, fingerprint))
    return tuple(entries)


def _manifest_bytes(items: tuple[_CocoImage, ...], output_directory: Path) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("image_id", "image_path"))
    for item in items:
        writer.writerow(
            (
                str(item.image_id),
                os.path.relpath(item.path, start=output_directory),
            )
        )
    return stream.getvalue().encode("utf-8")


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _metadata(
    *,
    annotations: Path,
    annotation_digest: str,
    image_directory: Path,
    split_seed: int,
    reference: tuple[_CocoImage, ...],
    evaluation: tuple[_CocoImage, ...],
    reference_entries: tuple[ManifestEntry, ...],
    evaluation_entries: tuple[ManifestEntry, ...],
    reference_bytes: bytes,
    evaluation_bytes: bytes,
) -> dict:
    return {
        "schema_version": _SCHEMA_VERSION,
        "annotations": str(annotations),
        "annotation_sha256": annotation_digest,
        "image_directory": str(image_directory),
        "split_seed": split_seed,
        "reference_count": len(reference),
        "evaluation_count": len(evaluation),
        "reference_ids": [item.image_id for item in reference],
        "evaluation_ids": [item.image_id for item in evaluation],
        "reference_manifest": _REFERENCE_MANIFEST,
        "evaluation_manifest": _EVALUATION_MANIFEST,
        "reference_manifest_sha256": manifest_digest(reference_entries),
        "evaluation_manifest_sha256": manifest_digest(evaluation_entries),
        "reference_manifest_file_sha256": _bytes_sha256(reference_bytes),
        "evaluation_manifest_file_sha256": _bytes_sha256(evaluation_bytes),
    }


def _atomic_bytes(value: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _staged_file(path) as (handle, temporary):
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
        _require_staging_entry(handle, temporary, directory_fd=None)
        os.replace(temporary, path)
    _fsync_directory(path.parent)


def _regular_bytes(path: Path, *, label: str) -> bytes:
    with _open_regular_file(
        path, error_message=f"{label} must be a regular file"
    ) as handle:
        before = os.fstat(handle.fileno())
        value = handle.read()
        after = os.fstat(handle.fileno())
        try:
            visible = os.stat(path, follow_symlinks=False)
        except OSError as error:
            raise ValueError(f"{label} changed while it was read") from error
    if (
        _signature(before) != _signature(after)
        or _signature(after) != _signature(visible)
        or not stat.S_ISREG(visible.st_mode)
    ):
        raise ValueError(f"{label} changed while it was read")
    return value


def _canonical_json_bytes(value: dict) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _journal_bytes(metadata: dict) -> bytes:
    return _canonical_json_bytes(
        {
            "schema_version": _SCHEMA_VERSION,
            "state": "pending",
            "benchmark_manifest": metadata,
        }
    )


def _entry_present(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


@contextmanager
def _coordinated_manifest_directory(output_directory: Path):
    message = "COCO manifest directory must be a stable directory"
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ValueError(message) from error
    with _DirectoryLease(output_directory, message=message) as lease:
        fcntl.flock(lease.fd, fcntl.LOCK_EX)
        lease.verify_path()
        try:
            yield Path(f"/proc/self/fd/{lease.fd}")
        finally:
            lease.verify_path()


def _matches_or_absent(
    path: Path,
    expected: bytes,
    *,
    label: str,
    error_message: str,
) -> bool:
    if not _entry_present(path):
        return False
    if _regular_bytes(path, label=label) != expected:
        raise ValueError(error_message)
    return True


def _remove_journal(path: Path) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        return
    _fsync_directory(path.parent)


def _reject_overlapping_metadata_cohorts(value: bytes) -> None:
    try:
        metadata = json.loads(value)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return
    if type(metadata) is not dict:
        return
    reference_ids = metadata.get("reference_ids")
    evaluation_ids = metadata.get("evaluation_ids")
    if (
        type(reference_ids) is not list
        or type(evaluation_ids) is not list
        or any(type(item) is not int or item <= 0 for item in reference_ids)
        or any(type(item) is not int or item <= 0 for item in evaluation_ids)
    ):
        return
    if set(reference_ids) & set(evaluation_ids):
        raise ValueError("benchmark manifest reference and evaluation cohorts overlap")


def _validate_final_set(
    split: CocoSplit,
    *,
    reference_bytes: bytes,
    evaluation_bytes: bytes,
    metadata: dict,
    metadata_bytes: bytes,
) -> bool:
    if not _entry_present(split.benchmark_manifest):
        return False
    actual_metadata = _regular_bytes(
        split.benchmark_manifest, label="benchmark manifest"
    )
    if actual_metadata != metadata_bytes:
        _reject_overlapping_metadata_cohorts(actual_metadata)
        raise ValueError("incompatible existing benchmark manifest")
    reference_exists = _matches_or_absent(
        split.reference_manifest,
        reference_bytes,
        label="reference manifest",
        error_message="incompatible existing benchmark manifests",
    )
    evaluation_exists = _matches_or_absent(
        split.evaluation_manifest,
        evaluation_bytes,
        label="evaluation manifest",
        error_message="incompatible existing benchmark manifests",
    )
    if not reference_exists or not evaluation_exists:
        raise ValueError("incomplete existing COCO manifest set")
    loaded_reference = load_manifest(split.reference_manifest)
    loaded_evaluation = load_manifest(split.evaluation_manifest)
    validate_disjoint(loaded_reference, loaded_evaluation)
    if (
        manifest_digest(loaded_reference)
        != metadata["reference_manifest_sha256"]
        or manifest_digest(loaded_evaluation)
        != metadata["evaluation_manifest_sha256"]
    ):
        raise ValueError("incompatible existing benchmark manifests")
    return True


def _split(
    output_directory: Path,
    *,
    reference: tuple[_CocoImage, ...],
    evaluation: tuple[_CocoImage, ...],
    split_seed: int,
) -> CocoSplit:
    return CocoSplit(
        reference_manifest=output_directory / _REFERENCE_MANIFEST,
        evaluation_manifest=output_directory / _EVALUATION_MANIFEST,
        benchmark_manifest=output_directory / _BENCHMARK_MANIFEST,
        reference_ids=tuple(item.image_id for item in reference),
        evaluation_ids=tuple(item.image_id for item in evaluation),
        reference_count=len(reference),
        evaluation_count=len(evaluation),
        split_seed=split_seed,
    )


def create_coco_manifests(
    annotations,
    image_directory,
    output_directory,
    *,
    reference_count: int = 250,
    evaluation_count: int = 250,
    split_seed: int = _DEFAULT_SPLIT_SEED,
) -> CocoSplit:
    """Create or strictly resume deterministic COCO reference/evaluation CSVs."""
    reference_count = _positive_integer(
        reference_count, name="reference_count"
    )
    evaluation_count = _positive_integer(
        evaluation_count, name="evaluation_count"
    )
    split_seed = _split_seed(split_seed)
    annotations = Path(os.path.abspath(os.fspath(annotations)))
    image_directory = Path(os.path.abspath(os.fspath(image_directory)))
    output_directory = _absolute_directory(
        output_directory, name="COCO manifest directory"
    )
    images, annotation_digest = _validated_coco_images(
        annotations, image_directory
    )
    reference, evaluation = _select_cohorts(
        images,
        reference_count=reference_count,
        evaluation_count=evaluation_count,
        split_seed=split_seed,
    )
    reference_entries = _manifest_entries(reference)
    evaluation_entries = _manifest_entries(evaluation)
    validate_disjoint(reference_entries, evaluation_entries)
    reference_bytes = _manifest_bytes(reference, output_directory)
    evaluation_bytes = _manifest_bytes(evaluation, output_directory)
    metadata = _metadata(
        annotations=annotations,
        annotation_digest=annotation_digest,
        image_directory=image_directory,
        split_seed=split_seed,
        reference=reference,
        evaluation=evaluation,
        reference_entries=reference_entries,
        evaluation_entries=evaluation_entries,
        reference_bytes=reference_bytes,
        evaluation_bytes=evaluation_bytes,
    )
    metadata_bytes = _canonical_json_bytes(metadata)
    journal_bytes = _journal_bytes(metadata)
    split = _split(
        output_directory,
        reference=reference,
        evaluation=evaluation,
        split_seed=split_seed,
    )

    with _coordinated_manifest_directory(output_directory) as anchored_output:
        anchored_split = _split(
            anchored_output,
            reference=reference,
            evaluation=evaluation,
            split_seed=split_seed,
        )
        journal_path = anchored_output / _JOURNAL
        if _validate_final_set(
            anchored_split,
            reference_bytes=reference_bytes,
            evaluation_bytes=evaluation_bytes,
            metadata=metadata,
            metadata_bytes=metadata_bytes,
        ):
            if _entry_present(journal_path):
                if _regular_bytes(
                    journal_path, label="benchmark manifest journal"
                ) != journal_bytes:
                    raise ValueError("incompatible existing benchmark manifest journal")
                _remove_journal(journal_path)
            return split

        reference_exists = _matches_or_absent(
            anchored_split.reference_manifest,
            reference_bytes,
            label="reference manifest",
            error_message="incomplete existing COCO manifest set",
        )
        evaluation_exists = _matches_or_absent(
            anchored_split.evaluation_manifest,
            evaluation_bytes,
            label="evaluation manifest",
            error_message="incomplete existing COCO manifest set",
        )
        if _entry_present(journal_path):
            if _regular_bytes(
                journal_path, label="benchmark manifest journal"
            ) != journal_bytes:
                raise ValueError("incompatible existing benchmark manifest journal")
        else:
            _atomic_bytes(journal_bytes, journal_path)
        if not reference_exists:
            _atomic_bytes(reference_bytes, anchored_split.reference_manifest)
        if not evaluation_exists:
            _atomic_bytes(evaluation_bytes, anchored_split.evaluation_manifest)
        _atomic_bytes(metadata_bytes, anchored_split.benchmark_manifest)
        if not _validate_final_set(
            anchored_split,
            reference_bytes=reference_bytes,
            evaluation_bytes=evaluation_bytes,
            metadata=metadata,
            metadata_bytes=metadata_bytes,
        ):
            raise RuntimeError("COCO manifest publication did not complete")
        _remove_journal(journal_path)
    return split


def _validate_regular_file(path: Path, *, label: str) -> None:
    with _open_regular_file(path, error_message=f"{label} must be a regular file"):
        pass


def run_coco_benchmark(
    annotations,
    image_directory,
    checkpoint,
    output_directory,
    *,
    device: str,
    batch_size: int,
    shard_size: int,
    reference_count: int = 250,
    evaluation_count: int = 250,
    config: ExperimentConfig = FIXED_CONFIG,
    extractor_factory: Callable = RTDETRExtractor,
) -> Path:
    """Materialize benchmark inputs; later work coordinates corruption stages."""
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a nonempty PyTorch device string")
    _positive_integer(batch_size, name="batch_size")
    _positive_integer(shard_size, name="shard_size")
    if not isinstance(config, ExperimentConfig):
        raise ValueError("config must be an ExperimentConfig")
    if not callable(extractor_factory):
        raise ValueError("extractor_factory must be callable")
    checkpoint = Path(os.path.abspath(os.fspath(checkpoint)))
    _validate_regular_file(checkpoint, label="checkpoint")
    output_directory = _absolute_directory(
        output_directory, name="benchmark output directory"
    )
    create_coco_manifests(
        annotations,
        image_directory,
        output_directory / "inputs",
        reference_count=reference_count,
        evaluation_count=evaluation_count,
    )
    return output_directory
