import torch
from torch import Tensor, nn

from ....misc.tue_utils import get_persistence_diagrams_batched

__all__ = ["TUEBase"]


class TUEBase(nn.Module):
    def __init__(
        self,
        frechet_means: str,
    ):
        super().__init__()
        self.frechet_means = self.load_frechet_mean(frechet_means)

    def forward(
        self,
        x: Tensor,
        targets=None,
    ) -> dict[str, Tensor]:
        raise NotImplementedError("TUEBase Inherit must implement a forward function.")

    from ....misc.tue_utils import get_persistence_diagrams_batched

    @torch.no_grad()
    def _calculate_distances_vectorized(
        self,
        score_captures: dict[int, dict[str, Tensor]],
        confidence_mask: Tensor,
    ) -> Tensor:
        selected_layers = sorted(score_captures)

        if not selected_layers:
            raise RuntimeError("No score layers were captured")

        batch_size, num_queries = confidence_mask.shape
        device = confidence_mask.device

        distances = torch.full(
            (batch_size, num_queries, len(selected_layers)),
            float("nan"),
            dtype=torch.float32,
            device=device,
        )

        for layer_position, layer_id in enumerate(selected_layers):
            capture = score_captures[layer_id]

            layer_inputs = capture["input"]
            weight_matrix = capture["weight"]
            layer_logits = capture["logits"]

            mask = confidence_mask.to(layer_inputs.device)
            selected_inputs = layer_inputs[mask]
            selected_logits = layer_logits[mask]

            if selected_inputs.shape[0] == 0:
                continue

            # One large batch instead of multiple chunks.
            diagrams = get_persistence_diagrams_batched(
                weight_matrix=weight_matrix,
                layer_inputs=selected_inputs,
                chunk_size=selected_inputs.shape[0],
            )

            top_count = min(
                self.tue_topk,
                selected_logits.shape[-1],
            )

            topk_weights, topk_classes = selected_logits.sigmoid().topk(
                top_count,
                dim=-1,
            )

            # Expected shape: [num_classes, diagram_size].
            # Adjust this accessor if frechet_means uses a dictionary.
            layer_means = torch.stack(
                [
                    self.frechet_means[layer_id][class_id]
                    for class_id in range(selected_logits.shape[-1])
                ]
            ).to(
                device=diagrams.device,
                dtype=diagrams.dtype,
            )

            # [selected queries, top-k classes, diagram size]
            references = layer_means[topk_classes]

            # All query/class distances calculated simultaneously.
            candidate_distances = torch.sqrt(
                torch.mean(
                    (diagrams[:, None, :] - references) ** 2,
                    dim=-1,
                )
            )

            normalized_weights = topk_weights / topk_weights.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)

            expected_distances = (candidate_distances * normalized_weights).sum(dim=-1)

            layer_output = distances[:, :, layer_position]
            layer_output[confidence_mask] = expected_distances.to(device)

        return distances

    def load_frechet_mean(self, frechet_means_path: str):
        persistence_state = torch.load(
            frechet_means_path,
            map_location="cpu",
            weights_only=False,
        )

        frechet_means = persistence_state.get(
            "frechet_means_score",
            persistence_state,
        )

        if not frechet_means:
            raise ValueError(
                "Frechet means are None. Make sure the correct frechet means checkpoint is loaded and it contains the correct keys."
            )
        return frechet_means
