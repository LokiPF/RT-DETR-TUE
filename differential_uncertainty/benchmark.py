from __future__ import annotations

import json
import math
import random
from dataclasses import fields
from numbers import Integral
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image

from .bank import Reservoir
from .config import FIXED_CONFIG, ExperimentConfig
from .corruptions import CORRUPTION_NAMES, apply_corruption
from .evaluation import evaluate_scores
from .extraction import RTDETRExtractor, prepare_image
from .reporting import write_results
from .scoring import normalize_bank, score_triplet


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_REPORT_FILES = (
    "per_image_scores.csv",
    "results.csv",
    "summary.json",
    "report.md",
    "corruption_auroc_bars.png",
)


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _existing_file(value, name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{name} does not exist or is not a file: {path}")
    return path


def _existing_directory(value, name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"{name} does not exist or is not a directory: {path}")
    return path


def _select_images(root: Path, count: int, *, seed: int) -> list[Path]:
    """Sort a flat image directory, seed-shuffle it, and take ``count`` paths."""
    root = Path(root)
    images = sorted(
        path for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )
    if count > len(images):
        raise ValueError(
            f"requested {count} images from {root}, but only {len(images)} are available"
        )
    generator = np.random.default_rng(seed)
    generator.shuffle(images)
    return images[:count]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _config_dict(config: ExperimentConfig) -> dict:
    result = {}
    for field in fields(config):
        value = getattr(config, field.name)
        result[field.name] = list(value) if isinstance(value, tuple) else value
    return result


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _atomic_torch(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _load_json(path: Path, label: str):
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"malformed {label}: {path}") from error


def _load_torch(path: Path, label: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"malformed {label}: {path}") from error


def _run_config(
    checkpoint: Path,
    train_root: Path,
    val_root: Path,
    train_images: list[Path],
    val_images: list[Path],
    *,
    reference_count: int,
    evaluation_count: int,
    device: str,
    batch_size: int,
    seed: int,
    config: ExperimentConfig,
) -> dict:
    return {
        "checkpoint": str(checkpoint),
        "coco_train_images": str(train_root),
        "coco_val_images": str(val_root),
        "reference_count": reference_count,
        "evaluation_count": evaluation_count,
        "reference_images": [path.name for path in train_images],
        "evaluation_images": [path.name for path in val_images],
        "device": device,
        "batch_size": batch_size,
        "seed": seed,
        "fixed_config": _config_dict(config),
        "corruptions": list(CORRUPTION_NAMES),
    }


def _ensure_run_config(output: Path, expected: dict) -> None:
    path = output / "run_config.json"
    if path.exists():
        if _load_json(path, "run_config.json") != expected:
            raise ValueError("existing run_config.json does not match this run")
    else:
        _atomic_json(path, expected)


def _extract_images(extractor, images: list[Image.Image], config, batch_size: int) -> list[dict]:
    records = []
    try:
        for start in range(0, len(images), batch_size):
            batch_images = images[start : start + batch_size]
            batch = torch.stack(
                [prepare_image(image, config.image_size) for image in batch_images]
            )
            extracted = extractor.extract_batch(batch)
            if not isinstance(extracted, (list, tuple)) or len(extracted) != len(batch_images):
                raise ValueError("extractor must return one record per image")
            records.extend(extracted)
    finally:
        for image in images:
            image.close()
    return records


def _extract_paths(extractor, paths: list[Path], config, batch_size: int) -> list[dict]:
    images = []
    try:
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
    except Exception:
        for image in images:
            image.close()
        raise
    return _extract_images(extractor, images, config, batch_size)


def _bank_progress(path: Path, config: ExperimentConfig, reference_count: int, seed: int):
    if not path.exists():
        return 0, Reservoir(config.bank_capacity, config.persistence_dim, seed)
    progress = _load_torch(path, "bank progress")
    if not isinstance(progress, dict) or set(progress) != {"next_image", "reservoir"}:
        raise ValueError("malformed bank progress")
    next_image = progress["next_image"]
    if type(next_image) is not int or not 0 <= next_image <= reference_count:
        raise ValueError("malformed bank progress next_image")
    try:
        reservoir = Reservoir.from_state_dict(progress["reservoir"])
    except ValueError as error:
        raise ValueError("malformed bank progress reservoir") from error
    if reservoir.capacity != config.bank_capacity or reservoir.dimension != config.persistence_dim:
        raise ValueError("bank progress does not match fixed configuration")
    return next_image, reservoir


def _build_bank(
    extractor,
    paths: list[Path],
    progress_path: Path,
    config: ExperimentConfig,
    batch_size: int,
    seed: int,
) -> torch.Tensor:
    next_image, reservoir = _bank_progress(progress_path, config, len(paths), seed)
    for start in range(next_image, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        records = _extract_paths(extractor, batch_paths, config, batch_size)
        for offset, record in enumerate(records):
            reservoir.add_record(record, config.bank_confidence_threshold)
            completed = start + offset + 1
            _atomic_torch(progress_path, {
                "next_image": completed,
                "reservoir": reservoir.state_dict(),
            })
    try:
        return normalize_bank(reservoir.bank())
    except ValueError as error:
        if reservoir.size < config.bank_capacity:
            raise ValueError(
                f"too few eligible bank vectors: collected {reservoir.size}, "
                f"need {config.bank_capacity}"
            ) from error
        raise


def _evaluation_progress(path: Path, evaluation_count: int):
    if not path.exists():
        return 0
    progress = _load_torch(path, "evaluation progress")
    if not isinstance(progress, dict) or set(progress) != {
        "next_image", "numpy_random_state"
    }:
        raise ValueError("malformed evaluation progress")
    next_image = progress["next_image"]
    if type(next_image) is not int or not 0 <= next_image <= evaluation_count:
        raise ValueError("malformed evaluation progress next_image")
    try:
        np.random.set_state(progress["numpy_random_state"])
    except (TypeError, ValueError) as error:
        raise ValueError("malformed evaluation progress NumPy state") from error
    return next_image


def _variants(path: Path, corruption_fn: Callable) -> list[Image.Image]:
    with Image.open(path) as source:
        clean = source.convert("RGB")
    variants = [clean]
    try:
        for family in CORRUPTION_NAMES:
            for severity in (4, 5):
                corrupted = corruption_fn(clean, family, severity)
                if not isinstance(corrupted, Image.Image):
                    raise ValueError("corruption function must return a PIL image")
                variants.append(corrupted)
    except Exception:
        for image in variants:
            image.close()
        raise
    return variants


def _score_image(path: Path, extractor, normalized_bank, config, batch_size, corruption_fn):
    records = _extract_images(
        extractor, _variants(path, corruption_fn), config, batch_size
    )
    clean = records[0]
    rows = []
    cursor = 1
    for family in CORRUPTION_NAMES:
        triplet = []
        for severity, record in zip((0, 4, 5), (clean, records[cursor], records[cursor + 1])):
            bound = dict(record)
            bound.update(image_id=path.name, corruption=family, severity=severity)
            triplet.append(bound)
        rows.extend(score_triplet(triplet, normalized_bank, config))
        cursor += 2
    _validate_score_rows(rows, path.name)
    return rows


def _validate_score_rows(rows, image_id: str) -> None:
    expected = [(family, severity) for family in CORRUPTION_NAMES for severity in (0, 4, 5)]
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("score file must contain exactly 57 rows")
    for row, identity in zip(rows, expected):
        if not isinstance(row, dict) or set(row) != {
            "image_id", "corruption", "severity", "fingerprint", "confidence", "entropy"
        }:
            raise ValueError("score row has an invalid schema")
        if row["image_id"] != image_id or (row["corruption"], row["severity"]) != identity:
            raise ValueError("score rows have an invalid image/corruption order")
        if not all(
            isinstance(row[name], (int, float))
            and not isinstance(row[name], bool)
            and math.isfinite(row[name])
            for name in ("fingerprint", "confidence", "entropy")
        ):
            raise ValueError("scores must be finite")


def _score_path(scores: Path, image: Path) -> Path:
    return scores / f"{image.stem}.json"


def _evaluate_images(
    extractor,
    paths: list[Path],
    output: Path,
    normalized_bank: torch.Tensor,
    config: ExperimentConfig,
    batch_size: int,
    corruption_fn: Callable,
) -> list[dict]:
    scores = output / "scores"
    scores.mkdir(parents=True, exist_ok=True)
    progress_path = output / "evaluation_progress.pt"
    next_image = _evaluation_progress(progress_path, len(paths))
    for index in range(next_image):
        if not _score_path(scores, paths[index]).is_file():
            raise ValueError("evaluation progress refers to a missing score file")
    for index in range(next_image, len(paths)):
        rows = _score_image(
            paths[index], extractor, normalized_bank, config, batch_size, corruption_fn
        )
        _atomic_json(_score_path(scores, paths[index]), rows)
        _atomic_torch(progress_path, {
            "next_image": index + 1,
            "numpy_random_state": np.random.get_state(),
        })
    combined = []
    for path in paths:
        score_path = _score_path(scores, path)
        rows = _load_json(score_path, "score file")
        _validate_score_rows(rows, path.name)
        combined.extend(rows)
    return combined


def _write_final_results(output: Path, rows: list[dict], evaluation: dict) -> None:
    temporary = output / ".report-tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    write_results(temporary, rows, evaluation, CORRUPTION_NAMES)
    for name in _REPORT_FILES:
        (temporary / name).replace(output / name)
    temporary.rmdir()


def run_coco_benchmark(
    checkpoint,
    coco_train_images,
    coco_val_images,
    output,
    *,
    reference_count,
    evaluation_count,
    device="cuda:0",
    batch_size=1,
    seed=44,
    config=FIXED_CONFIG,
    extractor_factory=RTDETRExtractor,
    corruption_fn=apply_corruption,
) -> Path:
    """Run or resume the one fixed clean-COCO-bank corruption benchmark."""
    reference_count = _positive_int(reference_count, "reference_count")
    evaluation_count = _positive_int(evaluation_count, "evaluation_count")
    batch_size = _positive_int(batch_size, "batch_size")
    seed = _nonnegative_int(seed, "seed")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a nonempty string")
    if not isinstance(config, ExperimentConfig):
        raise ValueError("config must be an ExperimentConfig")
    if not callable(extractor_factory) or not callable(corruption_fn):
        raise ValueError("extractor_factory and corruption_fn must be callable")

    checkpoint = _existing_file(checkpoint, "checkpoint")
    train_root = _existing_directory(coco_train_images, "COCO train image directory")
    val_root = _existing_directory(coco_val_images, "COCO val image directory")
    train_images = _select_images(train_root, reference_count, seed=seed)
    val_images = _select_images(val_root, evaluation_count, seed=seed)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _seed_everything(seed)
    _ensure_run_config(output, _run_config(
        checkpoint, train_root, val_root, train_images, val_images,
        reference_count=reference_count, evaluation_count=evaluation_count,
        device=device, batch_size=batch_size, seed=seed, config=config,
    ))

    try:
        runtime_device = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"invalid PyTorch device: {device}") from error
    with extractor_factory(checkpoint, runtime_device, config) as extractor:
        normalized_bank = _build_bank(
            extractor, train_images, output / "bank_progress.pt", config,
            batch_size, seed,
        )
        score_rows = _evaluate_images(
            extractor, val_images, output, normalized_bank, config,
            batch_size, corruption_fn,
        )
    evaluation = evaluate_scores(
        score_rows, CORRUPTION_NAMES, samples=config.bootstrap_samples, seed=seed
    )
    _write_final_results(output, score_rows, evaluation)
    return output


__all__ = ["run_coco_benchmark"]
