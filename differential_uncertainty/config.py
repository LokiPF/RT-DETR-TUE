from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real


@dataclass(frozen=True)
class ExperimentConfig:
    image_size: tuple[int, int] = (640, 640)
    class_count: int = 80
    query_count: int = 300
    persistence_layer: int = 2
    persistence_dim: int = 335
    bank_capacity: int = 2_000
    bank_confidence_threshold: float = 0.5
    neighbors: int = 5
    bank_chunk_size: int = 512
    bootstrap_samples: int = 2_000
    levels: tuple[int, int] = (4, 5)

    def __post_init__(self) -> None:
        integer_fields = (
            self.class_count, self.query_count, self.persistence_layer,
            self.persistence_dim, self.bank_capacity, self.neighbors,
            self.bank_chunk_size, self.bootstrap_samples,
        )
        if any(type(value) is not int for value in integer_fields):
            raise ValueError("configuration integer fields must be non-boolean integers")
        if (
            type(self.image_size) is not tuple
            or len(self.image_size) != 2
            or any(type(value) is not int or value <= 0 for value in self.image_size)
        ):
            raise ValueError("image_size must contain two positive dimensions")
        if self.class_count <= 0 or self.query_count <= 0 or self.persistence_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if self.persistence_layer != 2:
            raise ValueError("persistence_layer is fixed at 2")
        if self.neighbors != 5:
            raise ValueError("neighbors is fixed at 5")
        if self.bank_capacity < self.neighbors:
            raise ValueError("bank_capacity must be at least neighbors")
        if (
            isinstance(self.bank_confidence_threshold, bool)
            or not isinstance(self.bank_confidence_threshold, Real)
            or not isfinite(self.bank_confidence_threshold)
            or not 0.0 <= self.bank_confidence_threshold <= 1.0
        ):
            raise ValueError("bank_confidence_threshold must lie in [0, 1]")
        if self.bank_chunk_size <= 0 or self.bootstrap_samples <= 0:
            raise ValueError("chunk and bootstrap sizes must be positive")
        if (
            type(self.levels) is not tuple
            or len(self.levels) != 2
            or any(type(value) is not int for value in self.levels)
            or self.levels != (4, 5)
        ):
            raise ValueError("levels are fixed at (4, 5)")


FIXED_CONFIG = ExperimentConfig()
