import torch
from torch import Tensor, nn

from ....misc.tue_utils import topological_uncertainty_from_captures

__all__ = ["TUEBase"]


def mod(a, b):
    out = a - a // b * b
    return out


class TUEBase(nn.Module):
    def __init__(self, frechet_means_path: str, num_top_queries: int, num_classes: int):
        super().__init__()

        calibration = torch.load(
            frechet_means_path,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
        )

        self.frechet_means = calibration["means"]

        self.num_top_queries = num_top_queries
        self.num_classes = num_classes

    def forward(
        self,
        outputs: dict,
        selection_mask: Tensor | None = None,
    ) -> dict:
        raise NotImplementedError("TUEBase Inherit must implement a forward function.")

    def get_query_indices(self, logits):
        scores = torch.sigmoid(logits)

        scores, flat_index = torch.topk(
            scores.flatten(1),
            self.num_top_queries,
            dim=-1,
        )

        query_indices = flat_index // self.num_classes
        labels = mod(flat_index, self.num_classes)
        return query_indices, labels, scores

    def calculate_tu(self, output, min_score: float = 0):
        captures = output["tue_info"]
        query_idx, classes, scores = self.get_query_indices(output["pred_logits"])
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
            tu_flat = topological_uncertainty_from_captures(
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
                    "scores": scores[b][mask],
                    "query_indices": query_idx[b][mask],
                    "tu": tu_per_image[b],
                }
            )

        return results
