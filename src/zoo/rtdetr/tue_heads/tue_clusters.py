import torch
from typing_extensions import override

from ....core import register
from ....misc.tue_utils import clustered_topological_uncertainty_from_captures
from .tue_base import TUEBase

__all__ = ["TUEClusters"]


@register()
class TUEClusters(TUEBase):
    def __init__(
        self,
        frechet_means: str,
        num_top_queries: int,
        num_classes: int,
        confidence_threshold: float = 0.0,
    ):
        super().__init__(frechet_means, num_top_queries, num_classes)

        self.confidence_threshold = confidence_threshold

    @override
    def calculate_tu(self, output, min_score):
        captures = output["tue_info"]
        query_idx, classes, scores, boxes = self.get_query_indices(
            output["pred_logits"], output["pred_boxes"]
        )
        selection_mask = scores > min_score

        has_reference = torch.tensor(
            [int(cls) in self.frechet_means for cls in classes.flatten()],
            device=classes.device,
            dtype=torch.bool,
        ).reshape_as(classes)

        selection_mask &= has_reference

        B, K = query_idx.shape  # K is the number of detections

        batch_indices = (
            torch.arange(B, device=query_idx.device).unsqueeze(1).expand(B, K)
        )

        if int(selection_mask.sum()) == 0:  # no detections
            tu_per_image = [
                torch.empty(0, device=scores.device, dtype=scores.dtype)
                for _ in range(B)
            ]
        else:
            tu_flat = clustered_topological_uncertainty_from_captures(
                captures,
                batch_indices[selection_mask],
                query_idx[selection_mask],
                classes[selection_mask],
                self.frechet_means,
            )

            counts = selection_mask.sum(dim=1).tolist()
            tu_per_image = torch.split(tu_flat, counts)

        results = []

        for b in range(B):
            mask = selection_mask[b]
            results.append(
                {
                    "labels": classes[b][mask],
                    "boxes": boxes[b][mask],
                    "scores": scores[b][mask],
                    "query_indices": query_idx[b][mask],
                    "tu": tu_per_image[b],
                }
            )

        return results

    def forward(
        self,
        outputs: dict,
    ) -> dict:
        results = self.calculate_tu(outputs, self.confidence_threshold)
        return results
