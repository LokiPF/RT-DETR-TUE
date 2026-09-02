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

    def _get_reference(
        self,
        layer_id: int,
        class_id: int,
    ) -> Tensor | None:
        layer_means = self.frechet_means.get(layer_id)

        if layer_means is None:
            layer_means = self.frechet_means.get(str(layer_id))

        if layer_means is None:
            return None

        reference = layer_means.get(class_id)

        if reference is None:
            reference = layer_means.get(str(class_id))

        return reference

    @torch.no_grad()
    def _expected_distance(
        self,
        diagram: Tensor,
        layer_id: int,
        class_ids: Tensor,  # [k] top-k class indices, cpu long
        class_weights: Tensor,  # [k] top-k scores,       cpu float
    ) -> float:
        """
        Expectation of the persistence distance over the top-k predicted
        classes for one query.

        Instead of committing to the single argmax class, blend the
        distances to each candidate class mean, weighted by that class's
        renormalized probability. Weights are renormalized over only the
        top-k classes that actually have a stored Frechet mean, so a
        missing reference does not bias the estimate low. Returns NaN when
        none of the top-k classes have a reference.
        """
        weighted_sum = 0.0
        weight_total = 0.0

        for class_id, weight in zip(
            class_ids.tolist(),
            class_weights.tolist(),
        ):
            reference = self._get_reference(
                layer_id=layer_id,
                class_id=int(class_id),
            )
            if reference is None:
                continue

            distance = self.diagram_distance(
                diagram=diagram,
                reference=reference,
            )
            weighted_sum += weight * float(distance)
            weight_total += weight

        if weight_total == 0.0:
            return float("nan")

        return weighted_sum / weight_total
