"""Copyright(c) 2023 lyuwenyu. All Rights Reserved."""

from collections.abc import Iterable

import torch
import torch.nn as nn
from sympy.tensor import tensor
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

        # self.rerank_head = TUEReRankHead()

        persistence_state = torch.load(
            frechet_means,
            map_location="cpu",
            weights_only=False,
        )
        frechet_means_score = persistence_state.get(
            "frechet_means_score",
            persistence_state,
        )
        frechet_means_bbox = persistence_state.get(
            "frechet_means_bbox",
            persistence_state,
        )

        self.frechet_means_score = self._pack_references(
            frechet_means_score
        )

        self.frechet_means_bbox = {
            int(bbox_layer_id): self._pack_references(decoder_means)
            for bbox_layer_id, decoder_means
            in frechet_means_bbox.items()
        }

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

        def move(reference_map):
            return {
                layer_id: (
                    class_ids.to(device, non_blocking=True),
                    references.to(device, non_blocking=True),
                )
                for layer_id, (class_ids, references)
                in reference_map.items()
            }

        self.frechet_means_score = move(self.frechet_means_score)
        self.frechet_means_bbox = {
            bbox_layer_id: move(decoder_references)
            for bbox_layer_id, decoder_references
            in self.frechet_means_bbox.items()
        }
        self._reference_device = device

    @torch.no_grad()
    def _expected_distance(
            self,
            diagram,
            layer_id,
            class_ids,
            class_weights,
            references,
    ):
        layer_references = references.get(layer_id)
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

    def _get_distances(self, confidence, selected_layers, diagrams, captures, references) -> Tensor:
        batch_size, num_queries = confidence.shape
        num_selected_layers_score = len(selected_layers)

        distances = torch.full(
            (batch_size, num_queries, num_selected_layers_score),
            fill_value=float("nan"),
            dtype=torch.float32,
            device=confidence.device
        )

        layer_classes_output = torch.full(
            (batch_size, num_queries, num_selected_layers_score),
            fill_value=-1,
            dtype=torch.long,
            device=confidence.device
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
                        references=references,
                    )

        return distances

    def forward(
        self,
        x: Tensor,
        targets=None,
    ) -> dict[str, Tensor]:
        features = self.backbone(x)
        features = self.encoder(features)

        captures_score, handles_score, selected_layers_score = hook_decoder_layers(
            transformer=self.decoder,
            decoder_layers=self.decoder_layers,
            head_task='score'
        )

        num_bbox_layers = len(self.decoder.dec_bbox_head[0].layers)

        bbox_captures = {}
        bbox_selected_layers = {}
        handles_bbox = []

        for bbox_layer_id in range(num_bbox_layers):
            captures, handles, selected = hook_decoder_layers(
                transformer=self.decoder,
                decoder_layers=self.decoder_layers,
                head_task="bbox",
                bbox_head_layer=bbox_layer_id,
            )
            bbox_captures[bbox_layer_id] = captures
            bbox_selected_layers[bbox_layer_id] = selected
            handles_bbox.extend(handles)

        try:
            outputs = self.decoder(features, targets)
        finally:
            for handle in handles_score + handles_bbox:
                handle.remove()

        logits = outputs["pred_logits"]
        device = logits.device
        self._move_references(device)

        probabilities = logits.sigmoid()
        confidence = probabilities.max(dim=-1).values
        confidence_mask = confidence > self.tue_confidence_threshold # confidence should be the same for bbox and classification uncertainty


        query_indices = [
            batch_mask.nonzero(as_tuple=True)[0]
            for batch_mask in confidence_mask
        ]

        diagrams_score = get_captured_persistence_diagrams(
            captures=captures_score,
            query_indices=query_indices,
            decoder_layer_indices=selected_layers_score,
        )

        diagrams_bbox = {
            bbox_layer_id: get_captured_persistence_diagrams(
                captures=bbox_captures[bbox_layer_id],
                query_indices=query_indices,
                decoder_layer_indices=bbox_selected_layers[bbox_layer_id],
            )
            for bbox_layer_id in range(num_bbox_layers)
        }

        distances_score = self._get_distances(
            confidence,
            selected_layers_score,
            diagrams_score,
            captures_score,
            self.frechet_means_score
        )

        distances_bbox = torch.stack(
            [
                self._get_distances(
                    confidence,
                    bbox_selected_layers[bbox_layer_id],
                    diagrams_bbox[bbox_layer_id],
                    bbox_captures[bbox_layer_id],
                    self.frechet_means_bbox[bbox_layer_id],
                )
                for bbox_layer_id in range(num_bbox_layers)
            ],
            dim=-1,
        )

        outputs.update(
            tue_uncertainty_score=torch.nanmean(distances_score, dim=-1),
            tue_uncertainty_bbox=torch.nanmean(
                distances_bbox,
                dim=(-2, -1),
            ),
            tue_confidence=confidence,
            tue_distances=distances_score,
        )

        # ranked_logits = self.rerank_head(outputs["pred_logits"], outputs["tue_uncertainty"])
        # outputs["pred_logits"] = ranked_logits
        return outputs

    def deploy(self):
        self.eval()

        for module in self.modules():
            if hasattr(module, "convert_to_deploy"):
                module.convert_to_deploy()

        return self

