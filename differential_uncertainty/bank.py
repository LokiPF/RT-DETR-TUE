from __future__ import annotations

import random

import torch
from torch import Tensor


class Reservoir:
    def __init__(self, capacity: int, dimension: int, seed: int = 44) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if type(dimension) is not int or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        self.capacity = capacity
        self.dimension = dimension
        self._vectors = torch.empty((capacity, dimension), dtype=torch.float16, device="cpu")
        self.size = 0
        self.seen = 0
        self._rng = random.Random(seed)

    def add(self, vectors: Tensor) -> None:
        if not isinstance(vectors, Tensor) or vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError("vectors must have shape (rows, dimension)")
        if not vectors.is_floating_point() or not bool(torch.isfinite(vectors).all()):
            raise ValueError("vectors must be finite floating tensors")
        for vector in vectors.detach().to(device="cpu", dtype=torch.float16):
            self.seen += 1
            if self.size < self.capacity:
                self._vectors[self.size].copy_(vector)
                self.size += 1
            else:
                replacement = self._rng.randrange(self.seen)
                if replacement < self.capacity:
                    self._vectors[replacement].copy_(vector)

    def add_record(self, record: dict, threshold: float) -> None:
        try:
            persistence = record["persistence"]
            logits = record["logits"]
            padded_ids = record["padded_ids"]
        except (KeyError, TypeError) as error:
            raise ValueError("record needs persistence, logits, and padded_ids") from error
        if not isinstance(persistence, Tensor) or not isinstance(logits, Tensor) or persistence.ndim != 2 or logits.ndim != 2:
            raise ValueError("record persistence and logits must be rank-two tensors")
        if persistence.shape[0] != logits.shape[0] or persistence.shape[1] != self.dimension:
            raise ValueError("record shapes do not match reservoir dimension")
        if not isinstance(padded_ids, Tensor) or padded_ids.ndim != 1:
            raise ValueError("padded_ids must be a one-dimensional tensor")
        if padded_ids.is_floating_point() or padded_ids.dtype == torch.bool:
            raise ValueError("padded_ids must have integer dtype")
        if not isinstance(threshold, (int, float)) or not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must lie in [0, 1]")
        keep = torch.ones(persistence.shape[0], dtype=torch.bool)
        if padded_ids.numel():
            ids = padded_ids.to(dtype=torch.long, device="cpu")
            if int(ids.min()) < 0 or int(ids.max()) >= persistence.shape[0]:
                raise ValueError("padded_ids lie outside record rows")
            keep[ids] = False
        confidence = logits.float().sigmoid().amax(dim=1).cpu()
        self.add(persistence.cpu()[keep & (confidence >= float(threshold))])

    def bank(self) -> Tensor:
        if self.size != self.capacity:
            raise ValueError("reservoir bank is not full")
        return self._vectors.clone()

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "dimension": self.dimension,
            "vectors": self._vectors[:self.size].clone(),
            "size": self.size,
            "seen": self.seen,
            "rng": self._rng.getstate(),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> Reservoir:
        if not isinstance(state, dict) or set(state) != {"capacity", "dimension", "vectors", "size", "seen", "rng"}:
            raise ValueError("invalid reservoir state")
        reservoir = cls(state["capacity"], state["dimension"])
        vectors, size, seen = state["vectors"], state["size"], state["seen"]
        if not isinstance(vectors, Tensor) or tuple(vectors.shape) != (size, reservoir.dimension) or not 0 <= size <= reservoir.capacity or not isinstance(seen, int) or seen < size:
            raise ValueError("invalid reservoir state values")
        reservoir._vectors[:size].copy_(vectors.to(dtype=torch.float16, device="cpu"))
        reservoir.size = size
        reservoir.seen = seen
        try:
            reservoir._rng.setstate(state["rng"])
        except (TypeError, ValueError) as error:
            raise ValueError("invalid reservoir random state") from error
        return reservoir
