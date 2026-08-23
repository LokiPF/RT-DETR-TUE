from PIL import Image, ImageFilter

from .base import Severity


class GaussianBlur:
    name = "gaussian_blur"
    severities = tuple(
        Severity(level, radius)
        for level, radius in enumerate((0.0, 1.0, 2.0, 4.0, 8.0, 12.0))
    )

    def apply(self, image: Image.Image, level: int) -> Image.Image:
        if level < 0 or level >= len(self.severities):
            raise ValueError(f"unknown gaussian_blur severity {level}")
        radius = self.severities[level].parameter
        if radius == 0.0:
            return image
        return image.filter(ImageFilter.GaussianBlur(radius))
