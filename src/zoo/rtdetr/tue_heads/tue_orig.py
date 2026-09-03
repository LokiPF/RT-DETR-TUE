"""Copyright(c) 2023 lyuwenyu. All Rights Reserved."""

import torch
from torch import Tensor

from ....core import register
from .tue_base import TUEBase

__all__ = ["TUEOrig"]


@register()
class TUEOrig(TUEBase):
    def __init__(
        self, frechet_means: str, tue_topk: int = 1, confidence_threshold: float = 0.0
    ):
        super().__init__(frechet_means)
        self.tue_topk = tue_topk
        self.confidence_threshold = confidence_threshold

    def forward(self, x: Tensor) -> dict[str, Tensor]:

        logits = x["pred_logits"]
        captures = x[
            "tue_info"
        ]  # input and weights are already captured in the forward pass
        score_captures = captures["score"]

        output_device = logits.device

        confidence = logits.sigmoid().max(dim=-1).values
        confidence_mask = confidence > self.confidence_threshold

        distances = self._calculate_distances_vectorized(
            score_captures,
            confidence_mask,
        )

        x["tue_uncertainty"] = torch.nanmean(
            distances.to(output_device),
            dim=-1,
        )

        return x
