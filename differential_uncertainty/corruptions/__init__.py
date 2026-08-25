from .base import Corruption, Severity
from .gaussian_blur import GaussianBlur
from .imagecorruptions import (
    ImageCorruption,
    additional_corruption_names,
)


def benchmark_corruptions():
    return (
        GaussianBlur(),
        *(
            ImageCorruption(name)
            for name in additional_corruption_names()
        ),
    )


__all__ = [
    "Corruption",
    "GaussianBlur",
    "ImageCorruption",
    "Severity",
    "additional_corruption_names",
    "benchmark_corruptions",
]
