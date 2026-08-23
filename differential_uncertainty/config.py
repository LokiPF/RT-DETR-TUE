from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


BLUR_RADII: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0, 12.0)


@dataclass(frozen=True)
class ExperimentConfig:
    image_size: tuple[int, int] = (640, 640)
    class_count: int = 80
    query_count: int = 300
    persistence_layer: int = 2
    persistence_dim: int = 335
    bank_capacity: int = 25_000
    bank_seed: int = 44
    k: int = 5
    bank_chunk_size: int = 8_192
    reference_decile: int = 9
    responsive_decile: int = 5
    persistence_orientation: int = 1
    confidence_orientation: int = -1
    raw_responsive_orientation: int = 1
    raw_reference_orientation: int = -1
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20_260_821
    blur_radii: tuple[float, ...] = BLUR_RADII

    def __post_init__(self) -> None:
        if len(self.blur_radii) != 6 or self.blur_radii[0] != 0.0:
            raise ValueError(
                "the fixed experiment needs six severities beginning with clean"
            )
        if not 0 <= self.reference_decile < 10 or not 0 <= self.responsive_decile < 10:
            raise ValueError("reference and responsive indices must address ten deciles")
        if self.reference_decile == self.responsive_decile:
            raise ValueError("reference and responsive deciles must differ")
        if self.k <= 0 or self.bank_capacity < self.k:
            raise ValueError(
                "bank_capacity must be at least k and k must be positive"
            )
        if self.bank_chunk_size <= 0:
            raise ValueError("bank_chunk_size must be positive")
        if self.bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be positive")
        if self.query_count < 10:
            raise ValueError("query_count must be large enough to form ten deciles")
        if self.persistence_dim <= 0:
            raise ValueError("persistence_dim must be positive")
        orientations = (
            self.persistence_orientation,
            self.confidence_orientation,
            self.raw_responsive_orientation,
            self.raw_reference_orientation,
        )
        if any(value not in (-1, 1) for value in orientations):
            raise ValueError("every score orientation must be -1 or +1")

    @classmethod
    def for_tests(cls, **changes: Any) -> ExperimentConfig:
        return replace(cls(), **changes)

    def scientific_dict(self) -> dict[str, Any]:
        return {
            "image_size": list(self.image_size),
            "class_count": self.class_count,
            "query_count": self.query_count,
            "persistence_layer": self.persistence_layer,
            "persistence_dim": self.persistence_dim,
            "bank_capacity": self.bank_capacity,
            "bank_seed": self.bank_seed,
            "k": self.k,
            "bank_chunk_size": self.bank_chunk_size,
            "reference_decile": self.reference_decile,
            "responsive_decile": self.responsive_decile,
            "orientations": {
                "persistence_relative_gap": self.persistence_orientation,
                "confidence_relative_gap": self.confidence_orientation,
                "raw_responsive": self.raw_responsive_orientation,
                "raw_reference": self.raw_reference_orientation,
            },
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_seed": self.bootstrap_seed,
            "default_blur_radii": list(self.blur_radii),
            "feature_normalization": "raw",
        }


FIXED_CONFIG = ExperimentConfig()
