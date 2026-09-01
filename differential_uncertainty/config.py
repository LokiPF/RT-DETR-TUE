from __future__ import annotations

from dataclasses import dataclass


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
        if len(self.image_size) != 2 or any(value <= 0 for value in self.image_size):
            raise ValueError("image_size must contain two positive dimensions")
        if self.class_count <= 0 or self.query_count <= 0 or self.persistence_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if self.persistence_layer != 2:
            raise ValueError("persistence_layer is fixed at 2")
        if self.neighbors != 5:
            raise ValueError("neighbors is fixed at 5")
        if self.bank_capacity < self.neighbors:
            raise ValueError("bank_capacity must be at least neighbors")
        if not 0.0 <= self.bank_confidence_threshold <= 1.0:
            raise ValueError("bank_confidence_threshold must lie in [0, 1]")
        if self.bank_chunk_size <= 0 or self.bootstrap_samples <= 0:
            raise ValueError("chunk and bootstrap sizes must be positive")
        if self.levels != (4, 5):
            raise ValueError("levels are fixed at (4, 5)")


FIXED_CONFIG = ExperimentConfig()
