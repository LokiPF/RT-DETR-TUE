"""Copyright(c) 2023 lyuwenyu. All Rights Reserved."""

from collections.abc import Sequence

import torch
from torch import Tensor

from ....core import register
from .tue_base import TUEBase

__all__ = ["TUEOrigClas"]


@register()
class TUEOrigClas(TUEBase):
    def __init__(
        self,
        frechet_means: str,
        tue_topk: int = 1,
        confidence_threshold: float = 0.0,
        module_patterns: Sequence[str] | str | None = None,
    ):
        super().__init__(frechet_means, module_patterns=module_patterns)
        if tue_topk != 1:
            raise ValueError(
                "Lacombe et al. condition TUE on the single predicted class; "
                "tue_topk must therefore be 1"
            )
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between zero and one")

        self.tue_topk = tue_topk
        self.confidence_threshold = confidence_threshold

    def forward(
        self,
        outputs: dict,
        selection_mask: Tensor | None = None,
    ) -> dict:
        """Attach TUE scores to detector outputs.

        ``selection_mask`` explicitly selects query locations and, when given,
        replaces confidence-threshold selection. This lets evaluation code
        calculate topology only for the detections it will actually score.
        """
        logits = outputs["pred_logits"]
        confidence = logits.sigmoid().max(dim=-1).values
        if selection_mask is None:
            selection_mask = confidence >= self.confidence_threshold
        elif selection_mask.shape != confidence.shape:
            raise ValueError(
                "selection_mask must have shape [batch, queries], got "
                f"{tuple(selection_mask.shape)}"
            )
        else:
            selection_mask = selection_mask.to(
                device=logits.device,
                dtype=torch.bool,
            )

        distances = self._calculate_distances_vectorized(
            captures=outputs["tue_info"],
            class_logits=logits,
            confidence_mask=selection_mask,
        )

        predicted_classes = logits.argmax(dim=-1, keepdim=True).expand_as(distances)
        outputs["tue_distances"] = distances
        outputs["tue_classes"] = torch.where(
            torch.isfinite(distances),
            predicted_classes,
            -1,
        )
        outputs["tue_uncertainty"] = torch.nanmean(distances, dim=-1)

        return outputs
