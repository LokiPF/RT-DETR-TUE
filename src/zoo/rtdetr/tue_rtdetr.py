"""Copyright(c) 2023 lyuwenyu. All Rights Reserved."""

from collections.abc import Iterable
from typing import ClassVar

import torch
from torch import Tensor, nn

from ...core import register
from ...misc.tue_utils import (
    get_captured_persistence_diagrams,
    hook_decoder_layers,
)

__all__ = ["TUERTDETR"]


@torch.no_grad()
def diagram_distance(
    diagram: Tensor,
    reference: Tensor,
) -> Tensor:
    """
    Euclidean distance between two fixed-length, sorted
    persistence vectors.
    """
    diagram = diagram.detach().cpu()
    reference = reference.detach().to(
        device="cpu",
        dtype=diagram.dtype,
    )

    if diagram.shape != reference.shape:
        raise ValueError(
            "Diagram and reference shapes differ: "
            f"{tuple(diagram.shape)} != {tuple(reference.shape)}"
        )

    return torch.linalg.vector_norm(
        diagram - reference,
        ord=2,
    )


@register()
class TUERTDETR(nn.Module):
    __inject__: ClassVar[list[str]] = [
        "backbone",
        "encoder",
        "decoder",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        frechet_means: str,
        decoder_layers: int | Iterable[int] | None = None,
        tue_confidence_threshold: float = 0.5,
        tue_topk: int = 1,
    ):
        super().__init__()

        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

        self.decoder_layers = decoder_layers
        self.tue_confidence_threshold = tue_confidence_threshold
        self.tue_topk = tue_topk

        persistence_state = torch.load(
            frechet_means,
            map_location="cpu",
            weights_only=False,
        )

        self.frechet_means = persistence_state.get(
            "frechet_means_score",
            persistence_state,
        )

        if not self.frechet_means:
            raise ValueError("Frechet means are None. Make sure the correct frechet means checkpoint is loaded and it contains the correct keys.")

    def _get_reference(
        self,
        layer_id: int,
        class_id: int,
    ) -> Tensor | None:
        layer_means = self.frechet_means.get(layer_id)

        # Also support string keys if the dictionary was serialized via JSON.
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

            distance = diagram_distance(
                diagram=diagram,
                reference=reference,
            )
            weighted_sum += weight * float(distance)
            weight_total += weight

        if weight_total == 0.0:
            return float("nan")

        return weighted_sum / weight_total

    def forward(
        self,
        x: Tensor,
        targets=None,
    ) -> dict[str, Tensor]:
        if self.training:
            raise Warning(
                "TUERTDETR does not need to be trained explicitly. When training network, use default RTDETR otherwise training can be slowed down considerably."
            )

        features = self.backbone(x)
        features = self.encoder(features)

        captures, handles, selected_layers = hook_decoder_layers(
            transformer=self.decoder,
            decoder_layers=self.decoder_layers,
        )

        try:
            outputs = self.decoder(features, targets)
        finally:
            for handle in handles:
                handle.remove()

        logits = outputs["pred_logits"]
        probabilities = logits.sigmoid()
        confidence, _ = probabilities.max(dim=-1)
        confidence_mask = (
            confidence > self.tue_confidence_threshold
        )

        query_indices = [
            torch.where(confidence_mask[batch_id])[0]
            for batch_id in range(confidence_mask.shape[0])
        ]

        diagrams = get_captured_persistence_diagrams(
            captures=captures,
            query_indices=query_indices,
            decoder_layer_indices=selected_layers,
        )

        batch_size, num_queries = confidence.shape
        num_selected_layers = len(selected_layers)

        # TODO: persistence diagrams and distances are calculated on cpu. Move to cuda
        distances = torch.full(
            (
                batch_size,
                num_queries,
                num_selected_layers,
            ),
            fill_value=float("nan"),
            dtype=torch.float32,
            device="cpu",
        )

        layer_classes_output = torch.full(
            (
                batch_size,
                num_queries,
                num_selected_layers,
            ),
            fill_value=-1,
            dtype=torch.long,
            device="cpu",
        )

        layer_positions = {
            layer_id: position
            for position, layer_id in enumerate(selected_layers)
        }

        for layer_id, batch_diagrams in diagrams.items(): # per layer
            layer_position = layer_positions[layer_id]
            layer_logits = captures[layer_id]["logits"].detach().cpu()
            layer_classes = layer_logits.argmax(dim=-1)

            # Top-k candidates + weights for the expected distance. Uses the
            # same sigmoid scoring as the rest of the model; weights are
            # renormalized per query inside _expected_distance.
            top_count = min(self.tue_topk, layer_logits.shape[-1])
            topk_weights, topk_classes = (
                layer_logits.sigmoid().topk(top_count, dim=-1)
            )

            for batch_id, query_diagrams in enumerate(batch_diagrams): # per batch
                for query_id, diagram in query_diagrams.items(): # per query/diagram
                    class_id = int(layer_classes[batch_id, query_id])

                    layer_classes_output[
                        batch_id, query_id, layer_position
                    ] = class_id

                    distances[
                        batch_id, query_id, layer_position
                    ] = self._expected_distance(
                        diagram=diagram,
                        layer_id=layer_id,
                        class_ids=topk_classes[batch_id, query_id],
                        class_weights=topk_weights[batch_id, query_id],
                    )

        output_device = logits.device

        outputs["tue_distances"] = distances.to(output_device)
        outputs["tue_classes"] = layer_classes_output.to(
            output_device
        )

        outputs["tue_uncertainty"] =  torch.nanmean(
            distances.to(output_device),
            dim=-1,
        )

        return outputs

    def deploy(self):
        self.eval()

        for module in self.modules():
            if hasattr(module, "convert_to_deploy"):
                module.convert_to_deploy()

        return self