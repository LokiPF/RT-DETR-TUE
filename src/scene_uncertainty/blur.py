from __future__ import annotations

from PIL import Image, ImageFilter
from torchvision.transforms.v2 import Transform


class FixedGaussianBlur(Transform):
    _transformed_types = (Image.Image,)

    def __init__(self, radius: float) -> None:
        super().__init__()
        if radius < 0:
            raise ValueError("Blur radius must be non-negative")
        self.radius = float(radius)

    def _transform(self, image: Image.Image, params: dict) -> Image.Image:
        if self.radius == 0:
            return image
        return image.filter(ImageFilter.GaussianBlur(radius=self.radius))

    def transform(self, image: Image.Image, params: dict) -> Image.Image:
        # The local transform stack calls the public torchvision-v2 hook.
        return self._transform(image, params)
