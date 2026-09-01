from __future__ import annotations

from PIL import Image, ImageFilter

from .imagecorruptions import apply_imagecorruption


CORRUPTION_NAMES = (
    "gaussian_blur", "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "snow",
    "frost", "fog", "brightness", "contrast", "elastic_transform",
    "pixelate", "jpeg_compression", "speckle_noise", "spatter", "saturate",
)
_BLUR_RADII = {4: 8, 5: 12}


def apply_corruption(image: Image.Image, name: str, severity: int) -> Image.Image:
    if name not in CORRUPTION_NAMES:
        raise ValueError(f"unknown corruption {name!r}")
    if type(severity) is not int or severity not in (4, 5):
        raise ValueError("corruption severity must be 4 or 5")
    if name == "gaussian_blur":
        return image.convert("RGB").filter(ImageFilter.GaussianBlur(_BLUR_RADII[severity]))
    return apply_imagecorruption(image, name, severity)


__all__ = ["CORRUPTION_NAMES", "apply_corruption"]
