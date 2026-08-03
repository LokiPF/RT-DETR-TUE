"""Copyright(c) 2023 lyuwenyu. All Rights Reserved."""

from collections.abc import Iterable

import torch
import torch.nn as nn
from torch import Tensor

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
    __inject__ = [
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
        tue_confidence_threshold: float = 0.8,
    ):
        super().__init__()

        if not 0.0 <= tue_confidence_threshold <= 1.0:
            raise ValueError(
                "tue_confidence_threshold must be between zero and one"
            )

        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

        persistence_state = torch.load(
            frechet_means,
            map_location="cpu",
            weights_only=False,
        )

        # Supports both the complete saved state and a raw means dictionary.
        self.frechet_means = persistence_state.get(
            "frechet_means",
            persistence_state,
        )

        self.decoder_layers = decoder_layers
        self.tue_confidence_threshold = tue_confidence_threshold

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

    def forward(
        self,
        x: Tensor,
        targets=None,
    ) -> dict[str, Tensor]:
        if self.training:
            raise RuntimeError(
                "TUERTDETR persistence scoring must run in eval mode. "
                "Call model.eval() before inference."
            )

        features = self.backbone(x)
        features = self.encoder(features)

        # Hooks must exist before the decoder forward pass.
        captures, handles, selected_layers = hook_decoder_layers(
            transformer=self.decoder,
            decoder_layers=self.decoder_layers,
        )

        try:
            outputs = self.decoder(features, targets)
        finally:
            # Prevent hooks from accumulating across forward passes.
            for handle in handles:
                handle.remove()

        missing_layers = set(selected_layers) - set(captures)

        if missing_layers:
            raise RuntimeError(
                "The following decoder layers did not execute: "
                f"{sorted(missing_layers)}. Check decoder.eval_idx."
            )

        logits = outputs["pred_logits"]

        # [B, num_queries, num_classes]
        probabilities = logits.sigmoid()

        # [B, num_queries]
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

        # Build these on the CPU because persistence diagrams and
        # references are stored on the CPU.
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

        for layer_id, batch_diagrams in diagrams.items():
            layer_position = layer_positions[layer_id]

            # [B, num_queries], moved once rather than calling
            # .item() repeatedly on GPU tensors.
            layer_classes = (
                captures[layer_id]["logits"]
                .argmax(dim=-1)
                .detach()
                .cpu()
            )

            for batch_id, query_diagrams in enumerate(
                batch_diagrams
            ):
                for query_id, diagram in query_diagrams.items():
                    class_id = int(
                        layer_classes[
                            batch_id,
                            query_id,
                        ]
                    )

                    layer_classes_output[
                        batch_id,
                        query_id,
                        layer_position,
                    ] = class_id

                    reference = self._get_reference(
                        layer_id=layer_id,
                        class_id=class_id,
                    )

                    # NaN remains when no reference mean exists.
                    if reference is None:
                        continue

                    distances[
                        batch_id,
                        query_id,
                        layer_position,
                    ] = diagram_distance(
                        diagram=diagram,
                        reference=reference,
                    )

        output_device = logits.device

        # [B, num_queries, num_selected_layers]
        outputs["tue_distances"] = distances.to(output_device)

        # Predicted class at each decoder layer.
        outputs["tue_classes"] = layer_classes_output.to(
            output_device
        )

        # Indicates which queries passed the final confidence threshold.
        outputs["tue_query_mask"] = confidence_mask

        # Maps the final dimension of tue_distances to decoder layer IDs.
        outputs["tue_layer_ids"] = torch.tensor(
            selected_layers,
            dtype=torch.long,
            device=output_device,
        )

        # Final-layer confidence used for query selection.
        outputs["tue_confidence"] = confidence
        outputs["tue_uncertainty"] = torch.nanmean(
            distances,
            dim=-1,
        )

        return outputs

    def deploy(self):
        self.eval()

        for module in self.modules():
            if hasattr(module, "convert_to_deploy"):
                module.convert_to_deploy()

        return self