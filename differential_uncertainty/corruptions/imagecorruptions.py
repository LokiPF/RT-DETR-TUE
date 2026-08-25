from __future__ import annotations

import inspect
import numpy as np
from PIL import Image

from .base import Severity


ADDITIONAL_IMAGECORRUPTIONS = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
    "speckle_noise",
    "spatter",
    "saturate",
)


class _ImageCorruptionsNumpyCompatibility:
    def __getattr__(self, name: str):
        if name == "float_":
            return np.float64
        return getattr(np, name)


def _prepare_imagecorruptions_compatibility() -> None:
    import imagecorruptions.corruptions as upstream

    if not hasattr(upstream.np, "float_"):
        upstream.np = _ImageCorruptionsNumpyCompatibility()

    if "multichannel" not in inspect.signature(upstream.gaussian).parameters:
        gaussian = upstream.gaussian

        def compatibility_gaussian(
            image, *args, multichannel=None, **kwargs,
        ):
            if multichannel:
                kwargs["channel_axis"] = -1
            return gaussian(image, *args, **kwargs)

        upstream.gaussian = compatibility_gaussian


class ImageCorruption:
    severities = tuple(Severity(level, float(level)) for level in range(6))

    def __init__(self, upstream_name: str) -> None:
        if upstream_name not in ADDITIONAL_IMAGECORRUPTIONS:
            raise ValueError(
                f"unknown ImageCorruptions corruption {upstream_name!r}"
            )
        self.name = upstream_name

    def apply(self, image: Image.Image, level: int) -> Image.Image:
        if level < 0 or level >= len(self.severities):
            raise ValueError(f"unknown imagecorruptions severity {level}")
        if level == 0:
            return image

        from imagecorruptions import corrupt
        _prepare_imagecorruptions_compatibility()

        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
        output = corrupt(
            pixels, corruption_name=self.name, severity=level,
        )
        return Image.fromarray(np.asarray(output, dtype=np.uint8), mode="RGB")


def additional_corruption_names() -> tuple[str, ...]:
    return ADDITIONAL_IMAGECORRUPTIONS
