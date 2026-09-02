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
        self.load_frechet_mean(frechet_means)

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
        output_device = confidence_mask.device

        distances = torch.full(
            (batch_size, num_queries, len(selected_layers)),
            float("nan"),
            dtype=torch.float32,
            device=output_device,
        )

        for layer_position, layer_id in enumerate(selected_layers):
            capture = score_captures[layer_id]

            layer_inputs = capture["input"]
            weight_matrix = capture["weight"]
            layer_logits = capture["logits"]

            layer_mask = confidence_mask.to(layer_inputs.device)
            selected_inputs = layer_inputs[layer_mask]
            selected_logits = layer_logits[layer_mask]

            if selected_inputs.shape[0] == 0:
                continue

            diagrams = get_persistence_diagrams_batched(
                weight_matrix=weight_matrix,
                layer_inputs=selected_inputs,
                chunk_size=selected_inputs.shape[0],
            )

            # Expected shape: [num_classes, diagram_size].
            layer_means = self.frechet_means[layer_id].to(
                device=diagrams.device,
                dtype=diagrams.dtype,
            )

            # A class is usable only when its complete Fréchet mean is finite.
            valid_classes = torch.isfinite(layer_means).all(dim=-1)
            num_valid_classes = int(valid_classes.sum().item())

            # No class was accumulated for this layer. Leave its output as NaN.
            if num_valid_classes == 0:
                continue

            # Sigmoid is monotonic, so selecting using logits gives the same
            # class ordering while allowing invalid classes to be set to -inf.
            masked_logits = selected_logits.to(diagrams.device).masked_fill(
                ~valid_classes.unsqueeze(0),
                float("-inf"),
            )

            top_count = min(self.tue_topk, num_valid_classes)

            topk_logits, topk_classes = masked_logits.topk(
                top_count,
                dim=-1,
            )
            topk_weights = topk_logits.sigmoid()

            # [selected queries, top-k classes, diagram size]
            references = layer_means[topk_classes]

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
            layer_output[confidence_mask] = expected_distances.to(
                device=output_device,
                dtype=layer_output.dtype,
            )

        return distances

    def load_frechet_mean(self, frechet_means_path: str):
        state = torch.load(
            frechet_means_path,
            map_location="cpu",
            weights_only=False,
        )

        means = state.get("means") if isinstance(state, dict) else state

        if means is None:
            raise ValueError(f"No 'means' tensor found in {frechet_means_path}")

        if not isinstance(means, torch.Tensor):
            raise TypeError(f"Expected Tensor, got {type(means).__name__}")

        self.register_buffer(
            "frechet_means",
            means.float(),
            persistent=False,
        )

        counts = state.get("counts") if isinstance(state, dict) else None

        if counts is not None:
            self.register_buffer(
                "frechet_counts",
                counts.long(),
                persistent=False,
            )
        else:
            self.frechet_counts = None
