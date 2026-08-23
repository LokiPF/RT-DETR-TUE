from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image


@dataclass(frozen=True)
class Severity:
    level: int
    parameter: float


class Corruption(Protocol):
    name: str
    severities: tuple[Severity, ...]

    def apply(self, image: Image.Image, level: int) -> Image.Image: ...
