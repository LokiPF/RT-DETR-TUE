"""Fixed COCO protocol for the baselines: conditions, evaluation images, folds, seeded corruptions."""
from __future__ import annotations

import hashlib
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from ..corruptions.imagecorruptions import apply_imagecorruption

COMMON_FAMILIES = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
    "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
)
EXTRA_FAMILIES = ("speckle_noise", "gaussian_blur", "spatter", "saturate")
FAMILIES = COMMON_FAMILIES + EXTRA_FAMILIES
SEVERITIES = (1, 2, 3, 4, 5)
CONDITIONS = (("clean", 0),) + tuple(
    (family, severity) for family in FAMILIES for severity in SEVERITIES
)
FOLDS = 5
_SUFFIXES = {".jpg", ".jpeg", ".png"}


def list_images(root) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"image directory does not exist: {root}")
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _SUFFIXES)


def evaluation_images(val_root, *, seed: int) -> list[Path]:
    """All val images in the old benchmark's seeded shuffle order; the order defines the folds."""
    images = list_images(val_root)
    if not images:
        raise ValueError(f"no images found in {val_root}")
    np.random.default_rng(seed).shuffle(images)
    return images


def assign_folds(count: int, folds: int = FOLDS) -> np.ndarray:
    """Fold of each image from its shuffled position: 0, 1, ..., folds - 1, 0, 1, ..."""
    if count <= 0 or folds < 2:
        raise ValueError("count must be positive and folds at least 2")
    return np.arange(count) % folds


def variant_seed(image_id: str, family: str, severity: int) -> int:
    return zlib.crc32(f"{image_id}|{family}|{severity}".encode("utf-8")) & 0xFFFFFFFF


def corrupt(image: Image.Image, image_id: str, family: str, severity: int) -> Image.Image:
    if family not in FAMILIES:
        raise ValueError(f"unknown corruption family {family!r}")
    if type(severity) is not int or severity not in SEVERITIES:
        raise ValueError("severity must be an integer from 1 to 5")
    rgb = image.convert("RGB")
    saved = np.random.get_state()
    try:
        np.random.seed(variant_seed(image_id, family, severity))
        corrupted = apply_imagecorruption(rgb, family, severity)
    finally:
        np.random.set_state(saved)
    if corrupted.size != rgb.size or corrupted.mode != "RGB":
        raise ValueError(f"{family} severity {severity} changed size or mode of {image_id}")
    return corrupted


def variants(image: Image.Image, image_id: str) -> list[np.ndarray]:
    """All 96 versions of one image, in CONDITIONS order, as uint8 RGB arrays."""
    rgb = image.convert("RGB")
    arrays = [np.asarray(rgb, dtype=np.uint8).copy()]
    for family, severity in CONDITIONS[1:]:
        arrays.append(np.asarray(corrupt(rgb, image_id, family, severity), dtype=np.uint8).copy())
    return arrays


def load_variants(path) -> tuple[str, list[np.ndarray]]:
    """Worker entry point: open one image file and build its 96 versions."""
    path = Path(path)
    with Image.open(path) as source:
        rgb = source.convert("RGB")
    return path.name, variants(rgb, path.name)


def digest(array: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]
