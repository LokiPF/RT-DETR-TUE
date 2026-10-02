"""The corruption protocol: clean plus 19 imagecorruptions families x severities 1-5, one seeded draw per image and condition."""
from __future__ import annotations

import hashlib
import inspect
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

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


class _NumpyCompatibility:
    def __getattr__(self, name: str):
        if name == "float_":
            return np.float64
        return getattr(np, name)


def _prepare_compatibility() -> None:
    import imagecorruptions.corruptions as upstream

    if not hasattr(upstream.np, "float_"):
        upstream.np = _NumpyCompatibility()
    if "multichannel" not in inspect.signature(upstream.gaussian).parameters:
        gaussian = upstream.gaussian

        def compatible_gaussian(image, *args, multichannel=None, **kwargs):
            if multichannel:
                kwargs["channel_axis"] = -1
            return gaussian(image, *args, **kwargs)

        upstream.gaussian = compatible_gaussian

    random_noise = getattr(getattr(upstream, "sk", None), "util", None)
    random_noise = getattr(random_noise, "random_noise", None)
    if random_noise is not None and not getattr(random_noise, "_fixed_legacy_rng", False):
        def compatible_random_noise(image, mode="gaussian", rng=None, clip=True, **kwargs):
            if rng is None:
                rng = int(np.random.randint(0, 2**32))
            return random_noise(image, mode=mode, rng=rng, clip=clip, **kwargs)

        compatible_random_noise._fixed_legacy_rng = True
        upstream.sk.util.random_noise = compatible_random_noise


def apply_imagecorruption(image: Image.Image, name: str, severity: int) -> Image.Image:
    from imagecorruptions import corrupt

    _prepare_compatibility()
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    output = corrupt(pixels, corruption_name=name, severity=severity)
    return Image.fromarray(np.asarray(output, dtype=np.uint8), mode="RGB")


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
