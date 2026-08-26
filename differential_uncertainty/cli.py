from __future__ import annotations

import argparse
import csv
import json
import fcntl
import io
import math
import os
import platform
import re
import stat
import sys
import unicodedata
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Callable

import numpy as np
import scipy
import torch
import torchvision
from PIL import __version__ as pillow_version

from .artifacts import (
    _fsync_directory,
    _open_regular_file,
    _require_staging_entry,
    _sha256_stream,
    _staged_file,
    ensure_provenance,
    iter_records,
    load_manifest as load_artifact_manifest,
    source_digest,
    atomic_json,
    validate_provenance,
)
from .bank import (
    build_reference_bank,
    load_reference_bank,
    save_reference_bank,
)
from .config import FIXED_CONFIG, ExperimentConfig
from .corruptions import Corruption, GaussianBlur, Severity
from .evaluation import evaluate_rows
from .extraction import (
    RTDETRExtractor,
    extract_manifest,
    validate_extraction_cache,
)
from .manifests import (
    load_manifest,
    manifest_digest,
    validate_disjoint,
    validate_image_fingerprint,
    validate_image_signature,
)
from .reporting import REPORT_FILES, _bundle_bytes, _DirectoryLease, write_report
from .scoring import score_image_records


_SCORE_COLUMNS = (
    "image_id",
    "severity",
    "padded_count",
    "valid_count",
    "reference_count",
    "responsive_count",
    "persistence_reference",
    "persistence_responsive",
    "persistence_relative_gap",
    "confidence_reference",
    "confidence_responsive",
    "confidence_relative_gap",
    "direct_confidence_mean",
    "direct_confidence_max",
)
_COUNT_COLUMNS = (
    "severity",
    "padded_count",
    "valid_count",
    "reference_count",
    "responsive_count",
)
_FLOAT_COLUMNS = tuple(
    name for name in _SCORE_COLUMNS if name not in {"image_id", *_COUNT_COLUMNS}
)
_NONNEGATIVE_INTEGER = re.compile(r"(?:0|[1-9][0-9]*)\Z")

_DETECTOR_SOURCE_PATHS = (
    "src/__init__.py",
    "src/nn/__init__.py",
    "src/nn/backbone/__init__.py",
    "src/nn/backbone/common.py",
    "src/nn/backbone/presnet.py",
    "src/zoo/__init__.py",
    "src/zoo/rtdetr/__init__.py",
    "src/zoo/rtdetr/box_ops.py",
    "src/zoo/rtdetr/denoising.py",
    "src/zoo/rtdetr/hybrid_encoder.py",
    "src/zoo/rtdetr/rtdetr.py",
    "src/zoo/rtdetr/rtdetrv2_decoder.py",
    "src/zoo/rtdetr/utils.py",
)


@dataclass(frozen=True)
class _CorruptionSnapshot:
    name: str
    severities: tuple[Severity, ...]
    _apply: Callable

    def apply(self, image, level):
        return self._apply(image, level)


class _ImmutableDict(dict):
    _error = "reference stage state is immutable"

    def __setitem__(self, *_args, **_kwargs):
        raise TypeError(self._error)

    def __delitem__(self, *_args, **_kwargs):
        raise TypeError(self._error)

    def clear(self, *_args, **_kwargs):
        raise TypeError(self._error)

    def pop(self, *_args, **_kwargs):
        raise TypeError(self._error)

    def popitem(self, *_args, **_kwargs):
        raise TypeError(self._error)

    def setdefault(self, *_args, **_kwargs):
        raise TypeError(self._error)

    def update(self, *_args, **_kwargs):
        raise TypeError(self._error)

    def __ior__(self, *_args, **_kwargs):
        raise TypeError(self._error)


class _ImmutableList(list):
    _error = "reference stage state is immutable"

    def __setitem__(self, *_args, **_kwargs):
        raise TypeError(self._error)

    def __delitem__(self, *_args, **_kwargs):
        raise TypeError(self._error)

    def __iadd__(self, *_args, **_kwargs):
        raise TypeError(self._error)

    def __imul__(self, *_args, **_kwargs):
        raise TypeError(self._error)

    def append(self, *_args, **_kwargs):
        raise TypeError(self._error)

    def clear(self, *_args, **_kwargs):
        raise TypeError(self._error)

    def extend(self, *_args, **_kwargs):
        raise TypeError(self._error)

    def insert(self, *_args, **_kwargs):
        raise TypeError(self._error)

    def pop(self, *_args, **_kwargs):
        raise TypeError(self._error)

    def remove(self, *_args, **_kwargs):
        raise TypeError(self._error)

    def reverse(self, *_args, **_kwargs):
        raise TypeError(self._error)

    def sort(self, *_args, **_kwargs):
        raise TypeError(self._error)


def _immutable_reference_value(value):
    if type(value) is dict:
        return _ImmutableDict(
            {key: _immutable_reference_value(item) for key, item in value.items()}
        )
    if type(value) is list:
        return _ImmutableList(_immutable_reference_value(item) for item in value)
    if type(value) is tuple:
        return tuple(_immutable_reference_value(item) for item in value)
    return value

@dataclass(frozen=True)
class ReferenceStage:
    """Authenticated clean-reference artifacts reusable across corruptions."""

    reference_manifest: tuple
    checkpoint: Path
    checkpoint_snapshot: tuple
    artifacts: Path
    runtime_device: torch.device
    runtime: _ImmutableDict
    batch_size: int
    shard_size: int
    config: ExperimentConfig
    provenance: _ImmutableDict
    reference_metadata: _ImmutableDict
    reference_cache: Path
    reference_cache_snapshot: tuple
    bank_path: Path
    bank_snapshot: tuple
    bank_metadata: _ImmutableDict
    bank_binding_path: Path
    bank_binding_snapshot: tuple
    bank_binding: _ImmutableDict
    _bank: torch.Tensor

    @property
    def bank(self) -> torch.Tensor:
        """Return an isolated bank copy for external inspection."""
        return self._bank.clone()


def _snapshot_corruption(
    corruption: Corruption, config: ExperimentConfig
) -> _CorruptionSnapshot:

    try:
        name = corruption.name
    except Exception as error:
        raise ValueError(
            "corruption name must be a nonempty string"
        ) from error
    if not isinstance(name, str) or not name.strip():
        raise ValueError("corruption name must be a nonempty string")
    if len(name) > 128:
        raise ValueError("corruption name must be at most 128 characters")
    if any(
        unicodedata.category(character).startswith("C")
        for character in name
    ):
        raise ValueError(
            "corruption name must not contain control characters"
        )

    try:
        severities = corruption.severities
    except Exception as error:
        raise ValueError(
            "corruption severities must be a tuple"
        ) from error
    if not isinstance(severities, tuple):
        raise ValueError("corruption severities must be a tuple")
    if len(severities) != 6:
        raise ValueError(
            "corruption severities must contain exactly six entries"
        )
    if not all(isinstance(item, Severity) for item in severities):
        raise ValueError(
            "corruption severities must contain only Severity entries"
        )

    levels = []
    frozen_severities = []
    for severity in severities:
        level = severity.level
        if isinstance(level, bool) or not isinstance(level, Integral):
            raise ValueError(
                "corruption severity levels must be non-bool integers"
            )
        parameter = severity.parameter
        if isinstance(parameter, bool) or not isinstance(parameter, Real):
            raise ValueError(
                "corruption severity parameters must be finite numbers"
            )
        try:
            numeric_parameter = float(parameter)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(
                "corruption severity parameters must be finite numbers"
            ) from error
        if not math.isfinite(numeric_parameter):
            raise ValueError(
                "corruption severity parameters must be finite numbers"
            )
        normalized_level = int(level)
        levels.append(normalized_level)
        frozen_severities.append(
            Severity(normalized_level, numeric_parameter)
        )
    if levels != list(range(6)):
        raise ValueError(
            "corruption severity levels must be exactly 0 through 5"
        )
    if name == "gaussian_blur" and [
        severity.parameter for severity in frozen_severities
    ] != [float(radius) for radius in config.blur_radii]:
        raise ValueError(
            "Gaussian corruption parameters must match config "
            "default_blur_radii"
        )

    try:
        apply = corruption.apply
    except Exception as error:
        raise ValueError(
            "corruption must provide a callable apply method"
        ) from error
    if not callable(apply):
        raise ValueError(
            "corruption must provide a callable apply method"
        )
    return _CorruptionSnapshot(name, tuple(frozen_severities), apply)


def _source_files(root: str | Path | None = None) -> list[Path]:
    source_root = (
        Path(__file__).resolve().parents[1]
        if root is None
        else Path(root).resolve()
    )
    workflow = sorted(
        (source_root / "differential_uncertainty").rglob("*.py")
    )
    detector = [source_root / name for name in _DETECTOR_SOURCE_PATHS]
    return [*workflow, *detector]


def _checkpoint_digest(path: Path) -> str:
    with _open_regular_file(
        path, error_message="checkpoint must be a regular file"
    ) as handle:
        return _sha256_stream(handle, 1024 * 1024)


def _provenance(
    reference, evaluation, checkpoint, config, corruption: Corruption, runtime
):
    levels = [severity.level for severity in corruption.severities]
    if levels != list(range(6)):
        raise ValueError("the fixed evaluator needs corruption levels 0 through 5")
    root = Path(__file__).resolve().parents[1]
    return {
        "schema_version": 1,
        "reference_manifest_sha256": manifest_digest(reference),
        "evaluation_manifest_sha256": manifest_digest(evaluation),
        "checkpoint_sha256": _checkpoint_digest(checkpoint),
        "source_sha256": source_digest(_source_files(), root=root),
        "config": config.scientific_dict(),
        "runtime": runtime,
        "corruption": {
            "name": corruption.name,
            "severities": [
                {
                    "level": severity.level,
                    "parameter": severity.parameter,
                }
                for severity in corruption.severities
            ],
        },
    }


def _entry_present(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _stat_signature(value) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _regular_file_snapshot(path: str | Path, *, label: str) -> tuple:
    path = Path(path)
    with _open_regular_file(
        path, error_message=f"{label} must be a regular file"
    ) as handle:
        before = os.fstat(handle.fileno())
        digest = _sha256_stream(handle, 1024 * 1024)
        after = os.fstat(handle.fileno())
        try:
            visible = os.stat(path, follow_symlinks=False)
        except OSError as error:
            raise ValueError(f"{label} changed while it was read") from error
    if (
        _stat_signature(before) != _stat_signature(after)
        or _stat_signature(after) != _stat_signature(visible)
        or not stat.S_ISREG(visible.st_mode)
    ):
        raise ValueError(f"{label} changed while it was read")
    return (*_stat_signature(after), digest)


def _cache_snapshot(directory: Path, *, label: str) -> tuple:
    manifest_path = directory / "manifest.json"
    manifest_before = _regular_file_snapshot(
        manifest_path, label=f"{label} manifest"
    )
    manifest = load_artifact_manifest(directory)
    manifest_after = _regular_file_snapshot(
        manifest_path, label=f"{label} manifest"
    )
    if manifest_before != manifest_after:
        raise ValueError(f"{label} manifest changed while it was read")
    names = manifest.get("shards")
    if type(names) is not list or names != [
        f"shard_{index:05d}.pt" for index in range(len(names))
    ]:
        raise ValueError(f"{label} has a non-canonical shard sequence")
    leaves = [("manifest.json", manifest_after)]
    leaves.extend(
        (
            name,
            _regular_file_snapshot(
                directory / name, label=f"{label} shard {name}"
            ),
        )
        for name in names
    )
    return tuple(leaves)


def _validated_cache_snapshot(
    entries,
    directory: Path,
    metadata: dict,
    corruption,
    *,
    label: str,
) -> tuple:
    before = _cache_snapshot(directory, label=label)
    if not validate_extraction_cache(entries, directory, metadata, corruption):
        raise RuntimeError(f"{label} is incomplete")
    after = _cache_snapshot(directory, label=label)
    if after != before:
        raise ValueError(f"{label} changed while it was validated")
    return after


def _load_stable_bank(path: Path, config: ExperimentConfig):
    before = _regular_file_snapshot(path, label="reference bank")
    bank, metadata = load_reference_bank(path, config=config)
    after = _regular_file_snapshot(path, label="reference bank")
    if after != before:
        raise ValueError("reference bank changed while it was loaded")
    return bank, metadata, after


def _load_stable_scores(path: Path, expected_image_ids):
    before = _regular_file_snapshot(path, label="score artifact")
    rows = _load_score_csv(path, expected_image_ids)
    after = _regular_file_snapshot(path, label="score artifact")
    if after != before:
        raise ValueError("score artifact changed while it was loaded")
    return rows, after


def _named_file_snapshots(root: Path, names, *, label: str) -> tuple:
    return tuple(
        (
            name,
            _regular_file_snapshot(root / name, label=f"{label} {name}"),
        )
        for name in names
    )


def _terminal_sweep(
    *,
    provenance_path: Path,
    provenance_snapshot: tuple,
    checkpoint: Path,
    checkpoint_snapshot: tuple,
    reference_cache: Path,
    reference_snapshot: tuple,
    evaluation_cache: Path,
    evaluation_snapshot: tuple,
    bank_path: Path,
    bank_snapshot: tuple,
    score_path: Path,
    score_snapshot: tuple,
    report_path: Path,
    report_snapshot: tuple,
    input_groups: tuple,
) -> None:
    checks = [
        (provenance_path, "run provenance", provenance_snapshot),
        (checkpoint, "checkpoint", checkpoint_snapshot),
    ]
    checks.extend(
        (
            reference_cache / name,
            f"reference cache {name}",
            snapshot,
        )
        for name, snapshot in reference_snapshot
    )
    checks.extend(
        (
            evaluation_cache / name,
            f"evaluation cache {name}",
            snapshot,
        )
        for name, snapshot in evaluation_snapshot
    )
    checks.extend(
        (
            (bank_path, "reference bank", bank_snapshot),
            (score_path, "score artifact", score_snapshot),
        )
    )
    checks.extend(
        (report_path / name, f"report file {name}", snapshot)
        for name, snapshot in report_snapshot
    )
    for path, label, expected in checks:
        if _regular_file_snapshot(path, label=label) != expected:
            raise ValueError(f"{label} changed during the terminal audit")
    for entries in input_groups:
        for entry in entries:
            validate_image_signature(entry)


def _absolute_output_path(value: str | Path) -> Path:
    output = Path(os.path.abspath(os.fspath(value)))
    if output == output.parent:
        raise ValueError("output directory must not be the filesystem root")
    return output


def _ensure_directory_entry(
    parent_fd: int,
    name: str,
    *,
    error_message: str,
) -> None:
    try:
        state = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        try:
            os.mkdir(name, dir_fd=parent_fd)
        except FileExistsError:
            pass
        except OSError as error:
            raise ValueError(error_message) from error
        else:
            os.fsync(parent_fd)
        try:
            state = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as error:
            raise ValueError(error_message) from error
    except OSError as error:
        raise ValueError(error_message) from error
    if not stat.S_ISDIR(state.st_mode):
        raise ValueError(error_message)


@contextmanager
def _pinned_child_directory(
    parent: _DirectoryLease,
    name: str,
    *,
    label: str,
):
    message = f"{label} directory must be a stable directory"
    parent.verify_path()
    _ensure_directory_entry(parent.fd, name, error_message=message)
    anchored_entry = Path(f"/proc/self/fd/{parent.fd}") / name
    with _DirectoryLease(anchored_entry, message=message) as child:
        parent.verify_path()
        child.verify_entry(parent.fd, name)
        try:
            yield Path(f"/proc/self/fd/{child.fd}"), child
        except BaseException:
            raise
        else:
            parent.verify_path()
            child.verify_entry(parent.fd, name)


@contextmanager
def _pinned_directory_path(path: Path):
    message = "output parent directory path component must be stable"
    if not path.is_absolute():
        raise ValueError(message)
    with ExitStack() as stack:
        root = stack.enter_context(
            _DirectoryLease(Path("/"), message=message)
        )
        current = root
        anchored = Path(f"/proc/self/fd/{root.fd}")
        for name in path.parts[1:]:
            anchored, current = stack.enter_context(
                _pinned_child_directory(
                    current,
                    name,
                    label="output parent directory path component",
                )
            )
        try:
            yield anchored, current
        except BaseException:
            raise
        else:
            root.verify_path()


@contextmanager
def _coordinated_output(output: Path):
    output_message = "output directory must be a stable directory"
    with _pinned_directory_path(output.parent) as (
        _anchored_parent,
        parent,
    ):
        fcntl.flock(parent.fd, fcntl.LOCK_EX)
        parent.verify_path()
        _ensure_directory_entry(
            parent.fd,
            output.name,
            error_message="output directory must be a directory",
        )
        anchored_output = Path(f"/proc/self/fd/{parent.fd}") / output.name
        with _DirectoryLease(
            anchored_output,
            message=output_message,
        ) as run:
            fcntl.flock(run.fd, fcntl.LOCK_EX)
            parent.verify_path()
            run.verify_entry(parent.fd, output.name)
            try:
                yield Path(f"/proc/self/fd/{run.fd}"), run
            except BaseException:
                raise
            else:
                parent.verify_path()
                run.verify_entry(parent.fd, output.name)



@contextmanager
def _coordinated_child_output(
    output: Path, *, parent_run: _DirectoryLease | None, label: str
):
    """Coordinate an output independently or as a pinned child of a run."""
    if parent_run is None:
        with _coordinated_output(output) as coordinated:
            yield coordinated
        return
    parent_run.verify_path()
    with _pinned_child_directory(
        parent_run, output.name, label=label
    ) as coordinated:
        yield coordinated


@contextmanager
def _borrow_coordinated_output(run: _DirectoryLease):
    """Reuse the lock and pinned directory held by a legacy run."""
    run.verify_path()
    try:
        yield Path(f"/proc/self/fd/{run.fd}"), run
    except BaseException:
        raise
    else:
        run.verify_path()


def _iter_evaluation_groups(records, expected_image_ids):
    iterator = iter(records)
    for expected_image_id in expected_image_ids:
        group = []
        for expected_severity in range(6):
            try:
                record = next(iterator)
            except StopIteration as error:
                raise RuntimeError(
                    "canonical evaluation record roster is incomplete"
                ) from error
            if type(record) is not dict:
                raise RuntimeError(
                    "canonical evaluation record roster contains a non-record"
                )
            image_id = record.get("image_id")
            severity = record.get("severity")
            if (
                type(image_id) is not str
                or type(severity) is not int
                or (image_id, severity)
                != (expected_image_id, expected_severity)
            ):
                raise RuntimeError(
                    "canonical evaluation record roster does not match "
                    "the evaluation manifest"
                )
            group.append(record)
        yield expected_image_id, group
    try:
        next(iterator)
    except StopIteration:
        return
    raise RuntimeError("canonical evaluation record roster has extra records")


def _atomic_score_csv(rows, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=_SCORE_COLUMNS,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    payload = stream.getvalue().encode("utf-8")
    with _staged_file(target) as (handle, temporary):
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        _require_staging_entry(handle, temporary, directory_fd=None)
        os.replace(temporary, target)
    _fsync_directory(target.parent)


def _parse_nonnegative_integer(value: str, *, field: str, row: int) -> int:
    if not _NONNEGATIVE_INTEGER.fullmatch(value):
        raise ValueError(
            f"score row {row} field {field} must be a non-negative integer"
        )
    return int(value)


def _parse_finite_float(value: str, *, field: str, row: int) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(
            f"score row {row} field {field} must be a finite number"
        ) from error
    if not math.isfinite(number):
        raise ValueError(
            f"score row {row} field {field} must be a finite number"
        )
    return number


def _load_score_csv(path: str | Path, expected_image_ids) -> list[dict]:
    with _open_regular_file(
        path, error_message="score artifact must be a regular file"
    ) as handle:
        try:
            text = handle.read().decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("score artifact must be UTF-8 CSV") from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != list(_SCORE_COLUMNS):
        raise ValueError("score artifact columns do not match the fixed score schema")
    parsed = []
    try:
        source_rows = list(reader)
    except csv.Error as error:
        raise ValueError("score artifact is malformed CSV") from error
    for number, raw in enumerate(source_rows, start=2):
        if None in raw or any(raw[name] is None for name in _SCORE_COLUMNS):
            raise ValueError(f"score row {number} is malformed")
        image_id = raw["image_id"]
        if (
            not image_id
            or len(image_id) > 256
            or any(
                unicodedata.category(character).startswith("C")
                for character in image_id
            )
            or image_id.lstrip().startswith(("=", "+", "-", "@"))
        ):
            raise ValueError(f"score row {number} has an unsafe image_id")
        row = {"image_id": image_id}
        for field in _COUNT_COLUMNS:
            row[field] = _parse_nonnegative_integer(
                raw[field], field=field, row=number
            )
        for field in _FLOAT_COLUMNS:
            row[field] = _parse_finite_float(
                raw[field], field=field, row=number
            )
        parsed.append(row)

    expected = [
        (str(image_id), severity)
        for image_id in sorted(str(value) for value in expected_image_ids)
        for severity in range(6)
    ]
    actual = [(row["image_id"], row["severity"]) for row in parsed]
    if actual != expected:
        raise ValueError(
            "score artifact roster must contain every evaluation image at "
            "severities 0 through 5 in canonical order"
        )
    return parsed


def _positive_integer(value, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _runtime_device(value: str) -> torch.device:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("device must be a nonempty PyTorch device string")
    try:
        requested = torch.device(value)
    except RuntimeError as error:
        raise ValueError(f"device is invalid: {value!r}") from error
    if requested.type == "cpu":
        return torch.device("cpu")
    if requested.type == "cuda":
        index = (
            torch.cuda.current_device()
            if requested.index is None
            else requested.index
        )
        return torch.device("cuda", index)
    return requested


def _runtime_provenance(
    device: torch.device, *, batch_size: int, shard_size: int
) -> dict:
    libraries = {
        "python": str(platform.python_version()),
        "pytorch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "numpy": str(np.__version__),
        "scipy": str(scipy.__version__),
        "pillow": str(pillow_version),
    }
    if device.type == "cuda":
        index = device.index
        if index is None:
            raise ValueError("CUDA device must have a resolved index")
        capability = list(torch.cuda.get_device_capability(index))
        cuda = {
            "runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu_name": torch.cuda.get_device_name(index),
            "compute_capability": capability,
        }
    else:
        cuda = {
            "runtime": None,
            "cudnn": None,
            "gpu_name": None,
            "compute_capability": None,
        }
    return {
        "device": {"type": device.type, "index": device.index},
        "batch_size": batch_size,
        "shard_size": shard_size,
        "libraries": libraries,
        "cuda": cuda,
    }


def _extraction_metadata(provenance: dict, *, stage: str) -> dict:
    manifest_key = f"{stage}_manifest_sha256"
    metadata = {
        "stage": stage,
        "input_id": provenance[manifest_key],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "source_sha256": provenance["source_sha256"],
        "config": provenance["config"],
        "runtime": provenance["runtime"],
    }
    if stage == "evaluation":
        metadata["corruption"] = provenance["corruption"]
    return metadata


def _run_pipeline_stages(
    reference,
    evaluation,
    checkpoint: Path,
    output: Path,
    *,
    runtime_device: torch.device,
    batch_size: int,
    shard_size: int,
    config: ExperimentConfig,
    extractor_factory,
    corruption: Corruption,
    provenance: dict,
    report_parent: _DirectoryLease,
    artifacts: Path,
    reference_cache: Path,
    evaluation_cache: Path,
) -> Path:
    reference_metadata = _extraction_metadata(provenance, stage="reference")
    evaluation_metadata = _extraction_metadata(provenance, stage="evaluation")
    reference_complete = validate_extraction_cache(
        reference, reference_cache, reference_metadata, None
    )
    evaluation_complete = validate_extraction_cache(
        evaluation, evaluation_cache, evaluation_metadata, corruption
    )
    if not reference_complete or not evaluation_complete:
        with extractor_factory(checkpoint, runtime_device, config) as extractor:
            if not reference_complete:
                extract_manifest(
                    reference,
                    reference_cache,
                    reference_metadata,
                    extractor,
                    None,
                    image_size=config.image_size,
                    batch_size=batch_size,
                    shard_size=shard_size,
                    anchored_directory=True,
                )
            if not evaluation_complete:
                extract_manifest(
                    evaluation,
                    evaluation_cache,
                    evaluation_metadata,
                    extractor,
                    corruption,
                    image_size=config.image_size,
                    batch_size=batch_size,
                    shard_size=shard_size,
                    anchored_directory=True,
                )
        del extractor
    reference_snapshot = _validated_cache_snapshot(
        reference,
        reference_cache,
        reference_metadata,
        None,
        label="reference cache",
    )
    evaluation_snapshot = _validated_cache_snapshot(
        evaluation,
        evaluation_cache,
        evaluation_metadata,
        corruption,
        label="evaluation cache",
    )

    if _checkpoint_digest(checkpoint) != provenance["checkpoint_sha256"]:
        raise ValueError("checkpoint_sha256 changed while the run was executing")

    bank_path = artifacts / "reference-bank.pt"
    if _entry_present(bank_path):
        bank, bank_metadata, bank_snapshot = _load_stable_bank(bank_path, config)
    else:
        bank = build_reference_bank(iter_records(reference_cache), config)
        bank_metadata = {
            "reference_manifest_sha256": provenance[
                "reference_manifest_sha256"
            ],
            "capacity": config.bank_capacity,
            "seed": config.bank_seed,
            "padding_removed": True,
        }
        save_reference_bank(bank, bank_path, bank_metadata, config=config)
        bank, bank_metadata, bank_snapshot = _load_stable_bank(bank_path, config)
    if (
        bank_metadata["reference_manifest_sha256"]
        != provenance["reference_manifest_sha256"]
    ):
        raise ValueError("reference bank provenance does not match this run")

    expected_ids = [entry.image_id for entry in evaluation]
    score_path = artifacts / "scores.csv"
    if not _entry_present(score_path):
        score_rows = []
        groups = _iter_evaluation_groups(
            iter_records(evaluation_cache), expected_ids
        )
        for _image_id, image_records in groups:
            score_rows.extend(score_image_records(image_records, bank, config))
        _atomic_score_csv(score_rows, score_path)
    rows, score_snapshot = _load_stable_scores(score_path, expected_ids)
    evaluation_summary = evaluate_rows(rows, config)
    intended_report = {}
    write_report(
        "report",
        rows,
        evaluation_summary,
        provenance,
        parent=report_parent,
        _expected_content=intended_report,
    )
    return {
        "reference_cache": reference_snapshot,
        "evaluation_cache": evaluation_snapshot,
        "bank": bank_snapshot,
        "bank_metadata": bank_metadata,
        "scores": score_snapshot,
        "rows": rows,
        "evaluation": evaluation_summary,
        "report": intended_report,
    }


def _final_audit(
    reference,
    evaluation,
    checkpoint: Path,
    output: Path,
    *,
    config: ExperimentConfig,
    corruption: Corruption,
    provenance: dict,
    artifacts: Path,
    reference_cache: Path,
    evaluation_cache: Path,
    expected: dict,
) -> None:
    provenance_path = artifacts / "provenance.json"
    provenance_before = _regular_file_snapshot(
        provenance_path, label="run provenance"
    )
    validate_provenance(
        output, provenance, artifacts_directory=artifacts
    )
    provenance_snapshot = _regular_file_snapshot(
        provenance_path, label="run provenance"
    )
    if provenance_snapshot != provenance_before:
        raise ValueError("run provenance changed while it was validated")
    reference_metadata = _extraction_metadata(provenance, stage="reference")
    evaluation_metadata = _extraction_metadata(provenance, stage="evaluation")
    reference_snapshot = _validated_cache_snapshot(
        reference,
        reference_cache,
        reference_metadata,
        None,
        label="reference cache",
    )
    if reference_snapshot != expected["reference_cache"]:
        raise ValueError("reference cache changed after it was consumed")
    evaluation_snapshot = _validated_cache_snapshot(
        evaluation,
        evaluation_cache,
        evaluation_metadata,
        corruption,
        label="evaluation cache",
    )
    if evaluation_snapshot != expected["evaluation_cache"]:
        raise ValueError("evaluation cache changed after it was consumed")

    checkpoint_snapshot = _regular_file_snapshot(
        checkpoint, label="checkpoint"
    )
    if checkpoint_snapshot[-1] != provenance["checkpoint_sha256"]:
        raise ValueError("checkpoint_sha256 changed while the run was executing")

    bank_path = artifacts / "reference-bank.pt"
    bank, bank_metadata, bank_snapshot = _load_stable_bank(bank_path, config)
    del bank
    if bank_snapshot != expected["bank"]:
        raise ValueError("reference bank changed after it was consumed")
    if (
        bank_metadata != expected["bank_metadata"]
        or bank_metadata["reference_manifest_sha256"]
        != provenance["reference_manifest_sha256"]
    ):
        raise ValueError("reference bank provenance does not match this run")

    expected_ids = [entry.image_id for entry in evaluation]
    score_path = artifacts / "scores.csv"
    rows, score_snapshot = _load_stable_scores(
        score_path, expected_ids
    )
    if score_snapshot != expected["scores"]:
        raise ValueError("score artifact changed after it was consumed")
    if rows != expected["rows"]:
        raise ValueError("score artifact values changed after they were consumed")
    evaluation_summary = evaluate_rows(rows, config)
    if evaluation_summary != expected["evaluation"]:
        raise ValueError("evaluation changed during the final audit")

    report_path = output / "report"
    report_before = _named_file_snapshots(
        report_path, REPORT_FILES, label="report file"
    )
    report_content = _bundle_bytes(
        report_path, message="published report bundle changed"
    )
    report_snapshot = _named_file_snapshots(
        report_path, REPORT_FILES, label="report file"
    )
    if report_snapshot != report_before:
        raise ValueError("published report bundle changed while it was validated")
    if report_content != expected["report"]:
        raise ValueError("published report bundle changed")

    try:
        _validate_input_images(reference, evaluation)
    except ValueError as error:
        raise ValueError("image changed after it was audited") from error
    _terminal_sweep(
        provenance_path=provenance_path,
        provenance_snapshot=provenance_snapshot,
        checkpoint=checkpoint,
        checkpoint_snapshot=checkpoint_snapshot,
        reference_cache=reference_cache,
        reference_snapshot=reference_snapshot,
        evaluation_cache=evaluation_cache,
        evaluation_snapshot=evaluation_snapshot,
        bank_path=bank_path,
        bank_snapshot=bank_snapshot,
        score_path=score_path,
        score_snapshot=score_snapshot,
        report_path=report_path,
        report_snapshot=report_snapshot,
        input_groups=(reference, evaluation),
    )


def _validate_input_images(*groups) -> None:
    for entries in groups:
        for entry in entries:
            validate_image_fingerprint(entry)



def _reference_provenance(
    reference, checkpoint: Path, config: ExperimentConfig, runtime: dict
) -> dict:
    root = Path(__file__).resolve().parents[1]
    return {
        "schema_version": 1,
        "stage": "reference",
        "reference_manifest_sha256": manifest_digest(reference),
        "checkpoint_sha256": _checkpoint_digest(checkpoint),
        "source_sha256": source_digest(_source_files(), root=root),
        "config": config.scientific_dict(),
        "runtime": runtime,
    }



_REFERENCE_BANK_BINDING_FILE = "reference-bank-binding.json"


def _reference_bank_binding(
    provenance: dict,
    reference_snapshot: tuple,
    bank_snapshot: tuple,
    bank_metadata: dict,
) -> dict:
    return {
        "schema_version": 1,
        "state": "complete",
        "reference_provenance": provenance,
        "reference_cache": [
            {"name": name, "snapshot": list(snapshot)}
            for name, snapshot in reference_snapshot
        ],
        "reference_bank": {
            "snapshot": list(bank_snapshot),
            "metadata": bank_metadata,
        },
    }



def _reference_cache_pending_binding(provenance: dict) -> dict:
    return {
        "schema_version": 1,
        "state": "cache-pending",
        "reference_provenance": provenance,
    }


def _reference_bank_pending_binding(
    provenance: dict, reference_snapshot: tuple
) -> dict:
    return {
        "schema_version": 1,
        "state": "bank-pending",
        "reference_provenance": provenance,
        "reference_cache": [
            {"name": name, "snapshot": list(snapshot)}
            for name, snapshot in reference_snapshot
        ],
    }


def _load_stable_reference_bank_binding(path: Path, expected=None):
    before = _regular_file_snapshot(path, label="reference bank binding")
    with _open_regular_file(
        path, error_message="reference bank binding must be a regular file"
    ) as handle:
        try:
            binding = json.load(handle)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("reference bank binding must be JSON") from error
    after = _regular_file_snapshot(path, label="reference bank binding")
    if after != before:
        raise ValueError("reference bank binding changed while it was loaded")
    if type(binding) is not dict:
        raise ValueError("reference bank binding must contain a JSON object")
    if expected is not None and binding != expected:
        raise ValueError("reference bank binding does not match preparation")
    return binding, after

def _reference_binding(reference: ReferenceStage) -> dict:
    return {
        "reference_manifest_sha256": reference.provenance[
            "reference_manifest_sha256"
        ],
        "reference_cache": [
            {"name": name, "snapshot": list(snapshot)}
            for name, snapshot in reference.reference_cache_snapshot
        ],
        "reference_bank": {
            "snapshot": list(reference.bank_snapshot),
            "metadata": reference.bank_metadata,
        },
        "reference_bank_binding": reference.bank_binding,
    }


def _reference_terminal_sweep(
    reference: ReferenceStage,
    *,
    checkpoint_snapshot: tuple,
    reference_snapshot: tuple,
    bank_snapshot: tuple,
    bank_binding_snapshot: tuple,
) -> None:
    checks = [
        (reference.checkpoint, "checkpoint", checkpoint_snapshot),
        (
            reference.bank_binding_path,
            "reference bank binding",
            bank_binding_snapshot,
        ),
        (reference.bank_path, "reference bank", bank_snapshot),
    ]
    checks.extend(
        (
            reference.reference_cache / name,
            f"reference cache {name}",
            snapshot,
        )
        for name, snapshot in reference_snapshot
    )
    for path, label, expected in checks:
        if _regular_file_snapshot(path, label=label) != expected:
            raise ValueError(f"{label} changed during the terminal audit")
    for entry in reference.reference_manifest:
        validate_image_signature(entry)


def _authenticate_reference_stage(reference: ReferenceStage) -> dict:
    if not isinstance(reference, ReferenceStage):
        raise ValueError("reference must be a ReferenceStage")
    if not isinstance(reference.config, ExperimentConfig):
        raise ValueError("reference stage config must be an ExperimentConfig")
    if not isinstance(reference.reference_manifest, tuple):
        raise ValueError("reference stage manifest must be immutable")
    if not isinstance(reference.runtime_device, torch.device):
        raise ValueError("reference stage device is invalid")

    provenance = reference.provenance
    if not isinstance(provenance, _ImmutableDict):
        raise ValueError("reference stage provenance is invalid")
    expected_metadata = _extraction_metadata(provenance, stage="reference")
    if reference.reference_metadata != expected_metadata:
        raise ValueError("reference stage metadata does not match provenance")
    root = Path(__file__).resolve().parents[1]
    if reference.runtime != provenance.get("runtime"):
        raise ValueError("reference stage runtime does not match provenance")
    if reference.config.scientific_dict() != provenance.get("config"):
        raise ValueError("reference stage config does not match provenance")
    if source_digest(_source_files(), root=root) != provenance["source_sha256"]:
        raise ValueError("reference source changed after preparation")

    checkpoint_snapshot = _regular_file_snapshot(
        reference.checkpoint, label="checkpoint"
    )
    if (
        checkpoint_snapshot != reference.checkpoint_snapshot
        or checkpoint_snapshot[-1] != provenance["checkpoint_sha256"]
    ):
        raise ValueError("checkpoint changed after reference preparation")
    reference_snapshot = _validated_cache_snapshot(
        reference.reference_manifest,
        reference.reference_cache,
        expected_metadata,
        None,
        label="reference cache",
    )
    if reference_snapshot != reference.reference_cache_snapshot:
        raise ValueError("reference cache changed after preparation")
    bank, bank_metadata, bank_snapshot = _load_stable_bank(
        reference.bank_path, reference.config
    )
    if bank_snapshot != reference.bank_snapshot:
        raise ValueError("reference bank changed after preparation")
    if (
        bank_metadata != reference.bank_metadata
        or bank_metadata["reference_manifest_sha256"]
        != provenance["reference_manifest_sha256"]
    ):
        raise ValueError("reference bank provenance does not match preparation")
    expected_binding = _reference_bank_binding(
        provenance, reference_snapshot, bank_snapshot, bank_metadata
    )
    bank_binding, bank_binding_snapshot = _load_stable_reference_bank_binding(
        reference.bank_binding_path, expected_binding
    )
    if (
        bank_binding_snapshot != reference.bank_binding_snapshot
        or bank_binding != reference.bank_binding
    ):
        raise ValueError("reference bank binding changed after preparation")
    if not isinstance(reference._bank, torch.Tensor) or not torch.equal(
        bank, reference._bank
    ):
        raise ValueError("reference bank values changed after preparation")
    _validate_input_images(reference.reference_manifest)
    return {
        "checkpoint": checkpoint_snapshot,
        "reference_cache": reference_snapshot,
        "bank": bank,
        "bank_snapshot": bank_snapshot,
        "bank_metadata": bank_metadata,
        "bank_binding_snapshot": bank_binding_snapshot,
        "bank_binding": bank_binding,
    }


def prepare_reference_stage(
    reference_manifest,
    checkpoint,
    artifacts,
    *,
    device: str,
    batch_size: int,
    shard_size: int,
    config: ExperimentConfig = FIXED_CONFIG,
    extractor_factory=RTDETRExtractor,
    _parent_run: _DirectoryLease | None = None,
) -> ReferenceStage:
    """Create or resume the clean cache and reference bank for reuse."""
    batch_size = _positive_integer(batch_size, name="batch_size")
    shard_size = _positive_integer(shard_size, name="shard_size")
    runtime_device = _runtime_device(device)
    runtime = _runtime_provenance(
        runtime_device, batch_size=batch_size, shard_size=shard_size
    )
    if not isinstance(config, ExperimentConfig):
        raise ValueError("config must be an ExperimentConfig")

    reference = load_manifest(reference_manifest)
    _validate_input_images(reference)
    checkpoint = Path(checkpoint).resolve()
    artifacts = _absolute_output_path(artifacts)
    provenance = _reference_provenance(reference, checkpoint, config, runtime)
    reference_metadata = _extraction_metadata(provenance, stage="reference")

    with _coordinated_child_output(
        artifacts,
        parent_run=_parent_run,
        label="artifacts",
    ) as (
        anchored_artifacts,
        artifacts_parent,
    ):
        _validate_input_images(reference)
        checkpoint_snapshot = _regular_file_snapshot(
            checkpoint, label="checkpoint"
        )
        if checkpoint_snapshot[-1] != provenance["checkpoint_sha256"]:
            raise ValueError("checkpoint_sha256 changed while it was prepared")
        with _pinned_child_directory(
            artifacts_parent,
            "reference-extractions",
            label="reference cache",
        ) as (reference_cache, _reference_parent):
            bank_path = anchored_artifacts / "reference-bank.pt"
            bank_binding_path = anchored_artifacts / _REFERENCE_BANK_BINDING_FILE
            cache_pending_binding = _reference_cache_pending_binding(provenance)
            reference_complete = validate_extraction_cache(
                reference, reference_cache, reference_metadata, None
            )
            if not reference_complete:
                if _entry_present(bank_binding_path):
                    bank_binding, _bank_binding_snapshot = (
                        _load_stable_reference_bank_binding(bank_binding_path)
                    )
                    if bank_binding != cache_pending_binding:
                        atomic_json(cache_pending_binding, bank_binding_path)
                else:
                    atomic_json(cache_pending_binding, bank_binding_path)
                with extractor_factory(
                    checkpoint, runtime_device, config
                ) as extractor:
                    extract_manifest(
                        reference,
                        reference_cache,
                        reference_metadata,
                        extractor,
                        None,
                        image_size=config.image_size,
                        batch_size=batch_size,
                        shard_size=shard_size,
                        anchored_directory=True,
                    )
                del extractor
            reference_snapshot = _validated_cache_snapshot(
                reference,
                reference_cache,
                reference_metadata,
                None,
                label="reference cache",
            )
            if _checkpoint_digest(checkpoint) != provenance["checkpoint_sha256"]:
                raise ValueError(
                    "checkpoint_sha256 changed while it was prepared"
                )

            bank_pending_binding = _reference_bank_pending_binding(
                provenance, reference_snapshot
            )
            rebuild_bank = not (
                reference_complete and _entry_present(bank_path)
            )
            if not rebuild_bank:
                bank, bank_metadata, bank_snapshot = _load_stable_bank(
                    bank_path, config
                )
                expected_binding = _reference_bank_binding(
                    provenance,
                    reference_snapshot,
                    bank_snapshot,
                    bank_metadata,
                )
                bank_binding, bank_binding_snapshot = (
                    _load_stable_reference_bank_binding(bank_binding_path)
                )
                if bank_binding != expected_binding:
                    if bank_binding not in (
                        cache_pending_binding,
                        bank_pending_binding,
                    ):
                        raise ValueError(
                            "reference bank binding does not match preparation"
                        )
                    rebuild_bank = True

            if rebuild_bank:
                if _entry_present(bank_path):
                    stale_bank, _stale_metadata, _stale_snapshot = (
                        _load_stable_bank(bank_path, config)
                    )
                    del stale_bank
                if _entry_present(bank_binding_path):
                    _load_stable_reference_bank_binding(bank_binding_path)
                atomic_json(bank_pending_binding, bank_binding_path)
                bank = build_reference_bank(
                    iter_records(reference_cache), config
                )
                bank_metadata = {
                    "reference_manifest_sha256": provenance[
                        "reference_manifest_sha256"
                    ],
                    "capacity": config.bank_capacity,
                    "seed": config.bank_seed,
                    "padding_removed": True,
                }
                save_reference_bank(bank, bank_path, bank_metadata, config=config)
                bank, bank_metadata, bank_snapshot = _load_stable_bank(
                    bank_path, config
                )
                expected_binding = _reference_bank_binding(
                    provenance,
                    reference_snapshot,
                    bank_snapshot,
                    bank_metadata,
                )
                atomic_json(expected_binding, bank_binding_path)
                bank_binding, bank_binding_snapshot = (
                    _load_stable_reference_bank_binding(
                        bank_binding_path, expected_binding
                    )
                )
            if (
                bank_metadata["reference_manifest_sha256"]
                != provenance["reference_manifest_sha256"]
            ):
                raise ValueError(
                    "reference bank provenance does not match preparation"
                )
            stage = ReferenceStage(
                reference_manifest=reference,
                checkpoint=checkpoint,
                checkpoint_snapshot=checkpoint_snapshot,
                artifacts=artifacts,
                runtime_device=runtime_device,
                runtime=_immutable_reference_value(runtime),
                batch_size=batch_size,
                shard_size=shard_size,
                config=config,
                provenance=_immutable_reference_value(provenance),
                reference_metadata=_immutable_reference_value(reference_metadata),
                reference_cache=artifacts / "reference-extractions",
                reference_cache_snapshot=reference_snapshot,
                bank_path=artifacts / "reference-bank.pt",
                bank_snapshot=bank_snapshot,
                bank_metadata=_immutable_reference_value(bank_metadata),
                bank_binding_path=artifacts / _REFERENCE_BANK_BINDING_FILE,
                bank_binding_snapshot=bank_binding_snapshot,
                bank_binding=_immutable_reference_value(bank_binding),
                _bank=bank.clone(),
            )
            verified = _authenticate_reference_stage(stage)
            _reference_terminal_sweep(
                stage,
                bank_binding_snapshot=verified["bank_binding_snapshot"],
                checkpoint_snapshot=verified["checkpoint"],
                reference_snapshot=verified["reference_cache"],
                bank_snapshot=verified["bank_snapshot"],
            )
    return stage


def _corruption_provenance(
    reference: ReferenceStage, evaluation, corruption: _CorruptionSnapshot
) -> dict:
    provenance = _provenance(
        reference.reference_manifest,
        evaluation,
        reference.checkpoint,
        reference.config,
        corruption,
        reference.runtime,
    )
    provenance["shared_reference"] = _reference_binding(reference)
    return provenance


def _terminal_corruption_sweep(
    reference: ReferenceStage,
    evaluation,
    *,
    provenance_path: Path,
    provenance_snapshot: tuple,
    checkpoint_snapshot: tuple,
    reference_snapshot: tuple,
    evaluation_cache: Path,
    evaluation_snapshot: tuple,
    bank_snapshot: tuple,
    bank_binding_snapshot: tuple,
    score_path: Path,
    score_snapshot: tuple,
    report_path: Path,
    report_snapshot: tuple,
) -> None:
    checks = [
        (provenance_path, "run provenance", provenance_snapshot),
        (reference.checkpoint, "checkpoint", checkpoint_snapshot),
        (
            reference.bank_binding_path,
            "reference bank binding",
            bank_binding_snapshot,
        ),
        (reference.bank_path, "reference bank", bank_snapshot),
        (score_path, "score artifact", score_snapshot),
    ]
    checks.extend(
        (
            reference.reference_cache / name,
            f"reference cache {name}",
            snapshot,
        )
        for name, snapshot in reference_snapshot
    )
    checks.extend(
        (
            evaluation_cache / name,
            f"evaluation cache {name}",
            snapshot,
        )
        for name, snapshot in evaluation_snapshot
    )
    checks.extend(
        (report_path / name, f"report file {name}", snapshot)
        for name, snapshot in report_snapshot
    )
    for path, label, expected in checks:
        if _regular_file_snapshot(path, label=label) != expected:
            raise ValueError(f"{label} changed during the terminal audit")
    for entry in (*reference.reference_manifest, *evaluation):
        validate_image_signature(entry)


def _final_corruption_audit(
    reference: ReferenceStage,
    evaluation,
    output: Path,
    *,
    corruption: _CorruptionSnapshot,
    provenance: dict,
    artifacts: Path,
    evaluation_cache: Path,
    expected: dict,
) -> None:
    provenance_path = artifacts / "provenance.json"
    provenance_before = _regular_file_snapshot(
        provenance_path, label="run provenance"
    )
    validate_provenance(output, provenance, artifacts_directory=artifacts)
    provenance_snapshot = _regular_file_snapshot(
        provenance_path, label="run provenance"
    )
    if provenance_snapshot != provenance_before:
        raise ValueError("run provenance changed while it was validated")

    reference_state = _authenticate_reference_stage(reference)
    if (
        reference_state["reference_cache"] != expected["reference_cache"]
        or reference_state["bank_snapshot"] != expected["bank"]
        or reference_state["bank_metadata"] != expected["bank_metadata"]
        or reference_state["bank_binding_snapshot"]
        != expected["bank_binding"]
    ):
        raise ValueError("shared reference changed after it was consumed")
    evaluation_metadata = _extraction_metadata(provenance, stage="evaluation")
    evaluation_snapshot = _validated_cache_snapshot(
        evaluation,
        evaluation_cache,
        evaluation_metadata,
        corruption,
        label="evaluation cache",
    )
    if evaluation_snapshot != expected["evaluation_cache"]:
        raise ValueError("evaluation cache changed after it was consumed")

    expected_ids = [entry.image_id for entry in evaluation]
    score_path = artifacts / "scores.csv"
    rows, score_snapshot = _load_stable_scores(score_path, expected_ids)
    if score_snapshot != expected["scores"]:
        raise ValueError("score artifact changed after it was consumed")
    if rows != expected["rows"]:
        raise ValueError("score artifact values changed after they were consumed")
    evaluation_summary = evaluate_rows(rows, reference.config)
    if evaluation_summary != expected["evaluation"]:
        raise ValueError("evaluation changed during the final audit")

    report_path = output / "report"
    report_before = _named_file_snapshots(
        report_path, REPORT_FILES, label="report file"
    )
    report_content = _bundle_bytes(
        report_path, message="published report bundle changed"
    )
    report_snapshot = _named_file_snapshots(
        report_path, REPORT_FILES, label="report file"
    )
    if report_snapshot != report_before:
        raise ValueError("published report bundle changed while it was validated")
    if report_content != expected["report"]:
        raise ValueError("published report bundle changed")

    try:
        _validate_input_images(reference.reference_manifest, evaluation)
    except ValueError as error:
        raise ValueError("image changed after it was audited") from error
    _terminal_corruption_sweep(
        reference,
        evaluation,
        provenance_path=provenance_path,
        provenance_snapshot=provenance_snapshot,
        checkpoint_snapshot=reference_state["checkpoint"],
        reference_snapshot=reference_state["reference_cache"],
        evaluation_cache=evaluation_cache,
        evaluation_snapshot=evaluation_snapshot,
        bank_snapshot=reference_state["bank_snapshot"],
        score_path=score_path,
        bank_binding_snapshot=reference_state["bank_binding_snapshot"],
        score_snapshot=score_snapshot,
        report_path=report_path,
        report_snapshot=report_snapshot,
    )



def _can_rebind_empty_corruption_run(
    artifacts: Path, output: Path, provenance: dict
) -> bool:
    """Allow legacy recovery only before any derived corruption output exists."""
    try:
        with _open_regular_file(
            artifacts / "provenance.json",
            error_message="run provenance must be a regular file",
        ) as handle:
            actual = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    if type(actual) is not dict:
        return False
    actual.pop("shared_reference", None)
    expected = dict(provenance)
    expected.pop("shared_reference", None)
    if actual != expected:
        return False
    if _entry_present(artifacts / "scores.csv") or _entry_present(output / "report"):
        return False
    try:
        with _DirectoryLease(
            artifacts / "evaluation-extractions",
            message="evaluation cache directory must be a stable directory",
        ) as cache:
            with os.scandir(f"/proc/self/fd/{cache.fd}") as entries:
                return not any(entries)
    except ValueError:
        return False


def run_corruption_stage(
    reference: ReferenceStage,
    evaluation_manifest,
    output_dir,
    corruption: Corruption,
    *,
    extractor_factory=RTDETRExtractor,
    _run: _DirectoryLease | None = None,
    _allow_reference_rebind: bool = False,
) -> Path:
    """Evaluate one independently resumable corruption against a reference."""
    if not isinstance(reference, ReferenceStage):
        raise ValueError("reference must be a ReferenceStage")
    corruption = _snapshot_corruption(corruption, reference.config)
    evaluation = load_manifest(evaluation_manifest)
    validate_disjoint(reference.reference_manifest, evaluation)
    _validate_input_images(reference.reference_manifest, evaluation)
    reference_state = _authenticate_reference_stage(reference)
    output = _absolute_output_path(output_dir)
    provenance = _corruption_provenance(reference, evaluation, corruption)

    coordinator = (
        _coordinated_output(output)
        if _run is None
        else _borrow_coordinated_output(_run)
    )
    with coordinator as (anchored_output, report_parent):
        _validate_input_images(reference.reference_manifest, evaluation)
        with _pinned_child_directory(
            report_parent,
            "artifacts",
            label="artifacts",
        ) as (artifacts, artifacts_parent):
            try:
                os.stat(
                    "provenance.json",
                    dir_fd=artifacts_parent.fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                ensure_provenance(
                    anchored_output,
                    provenance,
                    artifacts_directory=artifacts,
                )
            except OSError as error:
                raise ValueError(
                    "run provenance must be a regular file"
                ) from error
            else:
                try:
                    validate_provenance(
                        anchored_output,
                        provenance,
                        artifacts_directory=artifacts,
                    )
                except ValueError:
                    if (
                        not _allow_reference_rebind
                        or not _can_rebind_empty_corruption_run(
                            artifacts, anchored_output, provenance
                        )
                    ):
                        raise
                    atomic_json(provenance, artifacts / "provenance.json")
                    validate_provenance(
                        anchored_output,
                        provenance,
                        artifacts_directory=artifacts,
                    )
            with _pinned_child_directory(
                artifacts_parent,
                "evaluation-extractions",
                label="evaluation cache",
            ) as (evaluation_cache, _evaluation_parent):
                evaluation_metadata = _extraction_metadata(
                    provenance, stage="evaluation"
                )
                evaluation_complete = validate_extraction_cache(
                    evaluation,
                    evaluation_cache,
                    evaluation_metadata,
                    corruption,
                )
                if not evaluation_complete:
                    with extractor_factory(
                        reference.checkpoint,
                        reference.runtime_device,
                        reference.config,
                    ) as extractor:
                        extract_manifest(
                            evaluation,
                            evaluation_cache,
                            evaluation_metadata,
                            extractor,
                            corruption,
                            image_size=reference.config.image_size,
                            batch_size=reference.batch_size,
                            shard_size=reference.shard_size,
                            anchored_directory=True,
                        )
                    del extractor
                evaluation_snapshot = _validated_cache_snapshot(
                    evaluation,
                    evaluation_cache,
                    evaluation_metadata,
                    corruption,
                    label="evaluation cache",
                )
                if _checkpoint_digest(reference.checkpoint) != provenance[
                    "checkpoint_sha256"
                ]:
                    raise ValueError(
                        "checkpoint_sha256 changed while the run was executing"
                    )

                expected_ids = [entry.image_id for entry in evaluation]
                score_path = artifacts / "scores.csv"
                if not _entry_present(score_path):
                    score_rows = []
                    groups = _iter_evaluation_groups(
                        iter_records(evaluation_cache), expected_ids
                    )
                    for _image_id, image_records in groups:
                        score_rows.extend(
                            score_image_records(
                                image_records,
                                reference_state["bank"],
                                reference.config,
                            )
                        )
                    _atomic_score_csv(score_rows, score_path)
                rows, score_snapshot = _load_stable_scores(
                    score_path, expected_ids
                )
                evaluation_summary = evaluate_rows(rows, reference.config)
                intended_report = {}
                write_report(
                    "report",
                    rows,
                    evaluation_summary,
                    provenance,
                    parent=report_parent,
                    _expected_content=intended_report,
                )
                expected = {
                    "reference_cache": reference_state["reference_cache"],
                    "bank": reference_state["bank_snapshot"],
                    "bank_metadata": reference_state["bank_metadata"],
                    "evaluation_cache": evaluation_snapshot,
                    "bank_binding": reference_state["bank_binding_snapshot"],
                    "scores": score_snapshot,
                    "rows": rows,
                    "evaluation": evaluation_summary,
                    "report": intended_report,
                }
                _final_corruption_audit(
                    reference,
                    evaluation,
                    anchored_output,
                    corruption=corruption,
                    provenance=provenance,
                    artifacts=artifacts,
                    evaluation_cache=evaluation_cache,
                    expected=expected,
                )
    return output
def _preflight_staged_legacy_run(
    output: Path,
    reference,
    evaluation,
    provenance: dict,
    run: _DirectoryLease,
) -> None:
    """Preserve legacy refusal-before-mutation checks around staged execution."""
    _validate_input_images(reference, evaluation)
    with _pinned_child_directory(
            run,
            "artifacts",
            label="artifacts",
        ) as (artifacts, artifacts_parent):
            try:
                os.stat(
                    "provenance.json",
                    dir_fd=artifacts_parent.fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            except OSError as error:
                raise ValueError(
                    "run provenance must be a regular file"
                ) from error
            else:
                with _open_regular_file(
                    artifacts / "provenance.json",
                    error_message="run provenance must be a regular file",
                ) as handle:
                    actual = json.load(handle)
                if type(actual) is not dict:
                    raise ValueError("run provenance must contain a JSON object")
                expected = dict(provenance)
                expected["shared_reference"] = actual.get(
                    "shared_reference", None
                )
                validate_provenance(
                    Path(f"/proc/self/fd/{run.fd}"),
                    expected,
                    artifacts_directory=artifacts,
                )
            with _pinned_child_directory(
                artifacts_parent,
                "reference-extractions",
                label="reference cache",
            ):
                with _pinned_child_directory(
                    artifacts_parent,
                    "evaluation-extractions",
                    label="evaluation cache",
                ):
                    pass


def run_pipeline(
    reference_manifest,
    evaluation_manifest,
    checkpoint,
    output_dir,
    *,
    device: str,
    batch_size: int,
    shard_size: int,
    config: ExperimentConfig = FIXED_CONFIG,
    extractor_factory=RTDETRExtractor,
    corruption: Corruption | None = None,
) -> Path:
    batch_size = _positive_integer(batch_size, name="batch_size")
    shard_size = _positive_integer(shard_size, name="shard_size")
    runtime_device = _runtime_device(device)
    runtime = _runtime_provenance(
        runtime_device, batch_size=batch_size, shard_size=shard_size
    )
    if not isinstance(config, ExperimentConfig):
        raise ValueError("config must be an ExperimentConfig")
    if corruption is None:
        corruption = GaussianBlur()
    corruption = _snapshot_corruption(corruption, config)
    manifest_reference = load_manifest(reference_manifest)
    evaluation = load_manifest(evaluation_manifest)
    validate_disjoint(manifest_reference, evaluation)
    _validate_input_images(manifest_reference, evaluation)
    checkpoint = Path(checkpoint).resolve()
    output = _absolute_output_path(output_dir)
    provenance = _provenance(
        manifest_reference, evaluation, checkpoint, config, corruption, runtime
    )
    with _coordinated_output(output) as (_anchored_output, run):
        _preflight_staged_legacy_run(
            output, manifest_reference, evaluation, provenance, run
        )
        reference = prepare_reference_stage(
            reference_manifest,
            checkpoint,
            output / "artifacts",
            device=device,
            batch_size=batch_size,
            shard_size=shard_size,
            config=config,
            extractor_factory=extractor_factory,
            _parent_run=run,
        )
        return run_corruption_stage(
            reference,
            evaluation_manifest,
            output,
            corruption,
            extractor_factory=extractor_factory,
            _run=run,
            _allow_reference_rebind=True,
        )


def _positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fixed differential corruption uncertainty"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help="run or resume the complete fixed workflow"
    )
    run.add_argument("--reference-manifest", required=True)
    run.add_argument("--evaluation-manifest", required=True)
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--batch-size", type=_positive, default=1)
    run.add_argument("--shard-size", type=_positive, default=50)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_pipeline(
            args.reference_manifest,
            args.evaluation_manifest,
            args.checkpoint,
            args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
            shard_size=args.shard_size,
        )
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
