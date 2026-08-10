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

class TUEReRankHead(nn.Module):
    def __init__(
        self,
        num_classes: int=80,
        hidden_dim=32,
        class_embedding_dim: int = 4,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.class_embedding_dim = nn.Embedding(num_classes, class_embedding_dim)

        input_dim = 2 + class_embedding_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, logits, tu_features) -> Tensor:
        batch_size, num_queries, num_classes = logits.shape

        valid_mask = torch.isfinite(tu_features)

        safe_tu = torch.log1p(torch.nan_to_num(tu_features, nan=0.0, posinf=0.0, neginf=0.0)).clamp_min(0.0) # ensures that tu has a valid value

        expanded_tu_features = safe_tu.unsqueeze(-1).expand_as(logits)

        class_ids = torch.arange(num_classes, device=logits.device)

        class_features = self.class_embedding_dim(class_ids)
        class_features = class_features.view(1, 1, num_classes, -1).expand(batch_size, num_queries, -1, -1)

        head_features = torch.cat([logits.unsqueeze(-1), expanded_tu_features.unsqueeze(-1), class_features], dim=-1)

        delta = self.network(head_features).squeeze(-1)
        ranked = logits + delta
        return torch.where(
            valid_mask.unsqueeze(-1),
            ranked,
            logits
        )


@register()
class TUERTDETR(nn.Module):
    __inject__ = ["backbone", "encoder", "decoder"]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        frechet_means: str,
        decoder_layers: int | Iterable[int] | None = None,
        tue_confidence_threshold: float = 0.8,
        tue_topk: int = 1,
    ):
        super().__init__()

        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_layers = decoder_layers
        self.tue_confidence_threshold = tue_confidence_threshold
        self.tue_topk = tue_topk

        self.rerank_head = TUEReRankHead()

        persistence_state = torch.load(
            frechet_means,
            map_location="cpu",
            weights_only=False,
        )
        frechet_means = persistence_state.get(
            "frechet_means",
            persistence_state,
        )
        self.frechet_means = self._pack_references(frechet_means)
        self._reference_device = torch.device("cpu")

    @staticmethod
    def _pack_references(
        frechet_means: dict,
    ) -> dict[int, tuple[Tensor, Tensor]]:
        packed = {}

        for layer_id, class_means in frechet_means.items():
            if not class_means:
                continue

            class_ids, references = zip(*class_means.items())
            packed[int(layer_id)] = (
                torch.tensor(
                    [int(class_id) for class_id in class_ids],
                    dtype=torch.long,
                ),
                torch.stack(
                    [
                        torch.as_tensor(reference).detach().flatten()
                        for reference in references
                    ]
                ),
            )

        return packed

    def _move_references(self, device: torch.device) -> None:
        if device == self._reference_device:
            return

        self.frechet_means = {
            layer_id: (
                class_ids.to(device, non_blocking=True),
                references.to(device, non_blocking=True),
            )
            for layer_id, (class_ids, references) in self.frechet_means.items()
        }
        self._reference_device = device

    @torch.no_grad()
    def _expected_distance(
        self,
        diagram: Tensor,
        layer_id: int,
        class_ids: Tensor,
        class_weights: Tensor,
    ) -> Tensor:
        """Return the weighted distance to the available class means."""
        layer_references = self.frechet_means.get(layer_id)
        if layer_references is None:
            return class_weights.new_full((), float("nan"))

        reference_ids, references = layer_references
        matches = class_ids[:, None] == reference_ids
        valid = matches.any(dim=1)
        reference_positions = matches.int().argmax(dim=1)

        diagram = diagram.detach().to(references, non_blocking=True).flatten()
        distances = torch.linalg.vector_norm(
            references[reference_positions] - diagram,
            dim=1,
        )

        weights = class_weights * valid
        weight_total = weights.sum()
        expected_distance = (distances * weights).sum() / weight_total.clamp_min(
            torch.finfo(weights.dtype).eps
        )
        return torch.where(
            weight_total > 0,
            expected_distance,
            expected_distance.new_full((), float("nan")),
        )

    def forward(
        self,
        x: Tensor,
        targets=None,
    ) -> dict[str, Tensor]:
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
        device = logits.device
        self._move_references(device)

        probabilities = logits.sigmoid()
        confidence = probabilities.max(dim=-1).values
        confidence_mask = confidence > self.tue_confidence_threshold

        query_indices = [
            batch_mask.nonzero(as_tuple=True)[0]
            for batch_mask in confidence_mask
        ]

        diagrams = get_captured_persistence_diagrams(
            captures=captures,
            query_indices=query_indices,
            decoder_layer_indices=selected_layers,
        )

        batch_size, num_queries = confidence.shape
        num_selected_layers = len(selected_layers)

        distances = torch.full(
            (batch_size, num_queries, num_selected_layers),
            fill_value=float("nan"),
            dtype=torch.float32,
            device=device,
        )

        layer_classes_output = torch.full(
            (batch_size, num_queries, num_selected_layers),
            fill_value=-1,
            dtype=torch.long,
            device=device,
        )

        for layer_position, layer_id in enumerate(selected_layers):
            batch_diagrams = diagrams.get(layer_id, [])
            layer_logits = captures[layer_id]["logits"].detach()
            layer_classes = layer_logits.argmax(dim=-1)

            top_count = min(self.tue_topk, layer_logits.shape[-1])
            topk_weights, topk_classes = layer_logits.sigmoid().topk(
                top_count,
                dim=-1,
            )

            for batch_id, query_diagrams in enumerate(batch_diagrams):
                for query_id, diagram in query_diagrams.items():
                    layer_classes_output[
                        batch_id, query_id, layer_position
                    ] = layer_classes[batch_id, query_id]
                    distances[
                        batch_id, query_id, layer_position
                    ] = self._expected_distance(
                        diagram=diagram,
                        layer_id=layer_id,
                        class_ids=topk_classes[batch_id, query_id],
                        class_weights=topk_weights[batch_id, query_id],
                    )

        outputs.update(
            tue_distances=distances,
            tue_classes=layer_classes_output,
            tue_query_mask=confidence_mask,
            tue_layer_ids=torch.tensor(
                selected_layers,
                dtype=torch.long,
                device=device,
            ),
            tue_confidence=confidence,
            tue_uncertainty=torch.nanmean(distances, dim=-1),
        )

        ranked_logits = self.rerank_head(outputs["pred_logits"], outputs["tue_uncertainty"])
        outputs["pred_logits"] = ranked_logits
        return outputs

    def deploy(self):
        self.eval()

        for module in self.modules():
            if hasattr(module, "convert_to_deploy"):
                module.convert_to_deploy()

        return self

