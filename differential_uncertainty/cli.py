from __future__ import annotations

import argparse
import csv
import fcntl
import io
import math
import os
import re
import stat
import sys
import unicodedata
from contextlib import ExitStack, contextmanager
from numbers import Integral
from pathlib import Path

import torch

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
    validate_provenance,
)
from .bank import (
    build_reference_bank,
    load_reference_bank,
    save_reference_bank,
)
from .config import FIXED_CONFIG, ExperimentConfig
from .corruptions.gaussian_blur import GaussianBlur
from .evaluation import evaluate_rows
from .extraction import (
    RTDETRExtractor,
    extract_manifest,
    validate_extraction_cache,
)
from .manifests import load_manifest, manifest_digest, validate_disjoint
from .reporting import _bundle_bytes, _DirectoryLease, write_report
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
    "src/core/__init__.py",
    "src/core/_config.py",
    "src/core/workspace.py",
    "src/core/yaml_config.py",
    "src/core/yaml_utils.py",
    "src/data/__init__.py",
    "src/data/_misc.py",
    "src/data/dataloader.py",
    "src/data/dataset/__init__.py",
    "src/data/dataset/_dataset.py",
    "src/data/dataset/cifar_dataset.py",
    "src/data/dataset/coco_dataset.py",
    "src/data/dataset/coco_eval.py",
    "src/data/dataset/coco_utils.py",
    "src/data/dataset/voc_detection.py",
    "src/data/dataset/voc_eval.py",
    "src/data/transforms/__init__.py",
    "src/data/transforms/_transforms.py",
    "src/data/transforms/container.py",
    "src/data/transforms/mosaic.py",
    "src/misc/__init__.py",
    "src/misc/box_ops.py",
    "src/misc/dist_utils.py",
    "src/misc/logger.py",
    "src/misc/profiler_utils.py",
    "src/misc/tue_utils.py",
    "src/misc/visualizer.py",
    "src/nn/__init__.py",
    "src/nn/arch/__init__.py",
    "src/nn/arch/classification.py",
    "src/nn/arch/yolo.py",
    "src/nn/backbone/__init__.py",
    "src/nn/backbone/common.py",
    "src/nn/backbone/csp_darknet.py",
    "src/nn/backbone/csp_resnet.py",
    "src/nn/backbone/hgnetv2.py",
    "src/nn/backbone/presnet.py",
    "src/nn/backbone/test_resnet.py",
    "src/nn/backbone/timm_model.py",
    "src/nn/backbone/torchvision_model.py",
    "src/nn/backbone/utils.py",
    "src/nn/criterion/__init__.py",
    "src/nn/criterion/det_criterion.py",
    "src/nn/postprocessor/__init__.py",
    "src/nn/postprocessor/nms_postprocessor.py",
    "src/optim/__init__.py",
    "src/optim/amp.py",
    "src/optim/ema.py",
    "src/optim/optim.py",
    "src/optim/warmup.py",
    "src/zoo/__init__.py",
    "src/zoo/rtdetr/__init__.py",
    "src/zoo/rtdetr/box_ops.py",
    "src/zoo/rtdetr/denoising.py",
    "src/zoo/rtdetr/hybrid_encoder.py",
    "src/zoo/rtdetr/matcher.py",
    "src/zoo/rtdetr/rtdetr.py",
    "src/zoo/rtdetr/rtdetr_criterion.py",
    "src/zoo/rtdetr/rtdetr_decoder.py",
    "src/zoo/rtdetr/rtdetr_postprocessor.py",
    "src/zoo/rtdetr/rtdetrv2_criterion.py",
    "src/zoo/rtdetr/rtdetrv2_decoder.py",
    "src/zoo/rtdetr/tue_rtdetr.py",
    "src/zoo/rtdetr/tue_rtdetrv2_decoder.py",
    "src/zoo/rtdetr/utils.py",
)


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


def _provenance(reference, evaluation, checkpoint, config, corruption):
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
        return torch.device(value)
    except RuntimeError as error:
        raise ValueError(f"device is invalid: {value!r}") from error


def _extraction_metadata(provenance: dict, *, stage: str) -> dict:
    manifest_key = f"{stage}_manifest_sha256"
    metadata = {
        "stage": stage,
        "input_id": provenance[manifest_key],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "source_sha256": provenance["source_sha256"],
        "config": provenance["config"],
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
    corruption: GaussianBlur,
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
    corruption: GaussianBlur,
    provenance: dict,
    artifacts: Path,
    reference_cache: Path,
    evaluation_cache: Path,
    expected: dict,
) -> None:
    validate_provenance(
        output, provenance, artifacts_directory=artifacts
    )
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

    if _checkpoint_digest(checkpoint) != provenance["checkpoint_sha256"]:
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
    rows, score_snapshot = _load_stable_scores(
        artifacts / "scores.csv", expected_ids
    )
    if score_snapshot != expected["scores"]:
        raise ValueError("score artifact changed after it was consumed")
    if rows != expected["rows"]:
        raise ValueError("score artifact values changed after they were consumed")
    evaluation_summary = evaluate_rows(rows, config)
    if evaluation_summary != expected["evaluation"]:
        raise ValueError("evaluation changed during the final audit")

    report_content = _bundle_bytes(
        output / "report", message="published report bundle changed"
    )
    if report_content != expected["report"]:
        raise ValueError("published report bundle changed")


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
) -> Path:
    batch_size = _positive_integer(batch_size, name="batch_size")
    shard_size = _positive_integer(shard_size, name="shard_size")
    runtime_device = _runtime_device(device)
    if not isinstance(config, ExperimentConfig):
        raise ValueError("config must be an ExperimentConfig")

    reference = load_manifest(reference_manifest)
    evaluation = load_manifest(evaluation_manifest)
    validate_disjoint(reference, evaluation)
    checkpoint = Path(checkpoint).resolve()
    output = _absolute_output_path(output_dir)
    corruption = GaussianBlur()
    provenance = _provenance(
        reference, evaluation, checkpoint, config, corruption
    )
    with _coordinated_output(output) as (anchored_output, report_parent):
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
                validate_provenance(
                    anchored_output,
                    provenance,
                    artifacts_directory=artifacts,
                )
            with _pinned_child_directory(
                artifacts_parent,
                "reference-extractions",
                label="reference cache",
            ) as (reference_cache, _reference_parent):
                with _pinned_child_directory(
                    artifacts_parent,
                    "evaluation-extractions",
                    label="evaluation cache",
                ) as (evaluation_cache, _evaluation_parent):
                    audit = _run_pipeline_stages(
                        reference,
                        evaluation,
                        checkpoint,
                        anchored_output,
                        runtime_device=runtime_device,
                        batch_size=batch_size,
                        shard_size=shard_size,
                        config=config,
                        extractor_factory=extractor_factory,
                        corruption=corruption,
                        provenance=provenance,
                        report_parent=report_parent,
                        artifacts=artifacts,
                        reference_cache=reference_cache,
                        evaluation_cache=evaluation_cache,
                    )
                    _final_audit(
                        reference,
                        evaluation,
                        checkpoint,
                        anchored_output,
                        config=config,
                        corruption=corruption,
                        provenance=provenance,
                        artifacts=artifacts,
                        reference_cache=reference_cache,
                        evaluation_cache=evaluation_cache,
                        expected=audit,
                    )
    return output


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
