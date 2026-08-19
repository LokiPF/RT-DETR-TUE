"""Copyright(c) 2023 lyuwenyu. All Rights Reserved."""

from collections.abc import Iterable

import torch
import torch.nn as nn
from torch import Tensor

from ...core import register
from ...misc.tue_utils import (
    conformal_pvalue,
    empirical_cdf,
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
        conformal_distances: str | None = None,
        decoder_layers: int | Iterable[int] | None = None,
        tue_confidence_threshold: float = 0.0,
        tue_topk: int = 1,
        tue_pvalue_reduction: str = "last",
        tue_decoder_layer_weights: Iterable[float] | None = None,
        tue_knn_artifact: str | None = None,
        tue_knn_k: int = 3,
    ):
        super().__init__()

        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_layers = decoder_layers
        self.tue_confidence_threshold = tue_confidence_threshold
        self.tue_topk = tue_topk
        if tue_knn_k <= 0:
            raise ValueError("tue_knn_k must be positive")
        self.tue_knn_k = int(tue_knn_k)
        if tue_pvalue_reduction not in (
            "last",
            "bonferroni",
            "mean",
            "weighted_mean",
            "min",
        ):
            raise ValueError(
                "tue_pvalue_reduction must be 'last', 'bonferroni', 'mean', "
                "'weighted_mean', or 'min'"
            )
        self.tue_pvalue_reduction = tue_pvalue_reduction
        self.tue_decoder_layer_weights = (
            None
            if tue_decoder_layer_weights is None
            else tuple(float(weight) for weight in tue_decoder_layer_weights)
        )
        if self.tue_decoder_layer_weights is not None:
            weights = torch.tensor(self.tue_decoder_layer_weights)
            if weights.numel() == 0:
                raise ValueError("tue_decoder_layer_weights cannot be empty")
            if not torch.isfinite(weights).all():
                raise ValueError("tue_decoder_layer_weights must be finite")
            if (weights < 0).any() or weights.sum() <= 0:
                raise ValueError(
                    "tue_decoder_layer_weights must be non-negative with a positive sum"
                )

        # Optional TUE-kNN mode. Persistence diagrams remain the TUE
        # representation, but a diagram is compared with a bank of reference
        # diagrams instead of one Fréchet mean. Its artifact also contains the
        # held-out calibration distribution of the final combined kNN score.
        self.tue_knn_reference_bank_score = None
        self.tue_knn_conformal_score = None
        self.tue_knn_metadata = None
        if tue_knn_artifact is not None:
            knn_state = torch.load(
                tue_knn_artifact,
                map_location="cpu",
                weights_only=False,
            )
            self.tue_knn_reference_bank_score = self._pack_knn_reference_bank(
                knn_state["reference_diagrams_score"]
            )
            self.tue_knn_conformal_score = {
                int(class_id): torch.sort(
                    torch.as_tensor(values).detach().flatten().float()
                ).values
                for class_id, values in knn_state[
                    "conformal_combined_distances_score"
                ].items()
            }
            self.tue_knn_metadata = knn_state.get("metadata", {})
            artifact_k = int(knn_state.get("knn_k", self.tue_knn_k))
            if artifact_k != self.tue_knn_k:
                raise ValueError(
                    f"TUE-kNN artifact uses k={artifact_k}, but the model "
                    f"configuration requests k={self.tue_knn_k}"
                )
            artifact_weights = knn_state.get("decoder_layer_weights")
            if artifact_weights is not None:
                artifact_weights = tuple(float(value) for value in artifact_weights)
                if self.tue_decoder_layer_weights is None:
                    self.tue_decoder_layer_weights = artifact_weights
                elif tuple(self.tue_decoder_layer_weights) != artifact_weights:
                    raise ValueError(
                        "TUE-kNN decoder-layer weights differ between artifact "
                        f"{artifact_weights} and model "
                        f"{self.tue_decoder_layer_weights}"
                    )

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

        self.frechet_means_score = self._pack_references(frechet_means_score)

        self.frechet_means_bbox = {
            int(bbox_layer_id): self._pack_references(decoder_means)
            for bbox_layer_id, decoder_means
            in frechet_means_bbox.items()
        }

        # Optional conformal calibration arrays (from CalibrationSolver.fit_conformal).
        # Structure mirrors the Frechet means: per (layer, class) a sorted 1-D
        # array of calibration distances, used to turn a measured distance into
        # an empirical CDF / conformal p-value.
        self.conformal_distances_score = None
        self.conformal_distances_bbox = None
        if conformal_distances is not None:
            conformal_state = torch.load(
                conformal_distances,
                map_location="cpu",
                weights_only=False,
            )
            self.conformal_distances_score = self._pack_conformal(
                conformal_state["conformal_distances_score"]
            )
            self.conformal_distances_bbox = {
                int(bbox_layer_id): self._pack_conformal(decoder_arrays)
                for bbox_layer_id, decoder_arrays
                in conformal_state["conformal_distances_bbox"].items()
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

    @staticmethod
    def _pack_conformal(
        distances_map: dict,
    ) -> dict[int, dict[int, Tensor]]:
        """{layer_id: {class_id: ascending 1-D calibration-distance tensor}}."""
        packed: dict[int, dict[int, Tensor]] = {}
        for layer_id, class_arrays in distances_map.items():
            if not class_arrays:
                continue
            packed[int(layer_id)] = {
                int(class_id): torch.sort(
                    torch.as_tensor(array).detach().flatten().float()
                ).values
                for class_id, array in class_arrays.items()
            }
        return packed

    @staticmethod
    def _pack_knn_reference_bank(
        reference_map: dict,
    ) -> dict[int, dict[int, Tensor]]:
        packed: dict[int, dict[int, Tensor]] = {}
        for layer_id, class_banks in reference_map.items():
            packed[int(layer_id)] = {}
            for class_id, diagrams in class_banks.items():
                bank = torch.as_tensor(diagrams).detach().float()
                if bank.ndim != 2 or bank.shape[0] == 0:
                    raise ValueError(
                        "Every TUE-kNN bank must have shape [N,D] with N>0; "
                        f"got {tuple(bank.shape)} for layer {layer_id}, "
                        f"class {class_id}"
                    )
                packed[int(layer_id)][int(class_id)] = bank
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

        def move_conformal(conformal_map):
            return {
                layer_id: {
                    class_id: array.to(device, non_blocking=True)
                    for class_id, array in class_arrays.items()
                }
                for layer_id, class_arrays in conformal_map.items()
            }

        self.frechet_means_score = move(self.frechet_means_score)
        self.frechet_means_bbox = {
            bbox_layer_id: move(decoder_references)
            for bbox_layer_id, decoder_references
            in self.frechet_means_bbox.items()
        }

        if self.conformal_distances_score is not None:
            self.conformal_distances_score = move_conformal(self.conformal_distances_score)
            self.conformal_distances_bbox = {
                bbox_layer_id: move_conformal(decoder_arrays)
                for bbox_layer_id, decoder_arrays
                in self.conformal_distances_bbox.items()
            }

        if self.tue_knn_reference_bank_score is not None:
            self.tue_knn_reference_bank_score = {
                layer_id: {
                    class_id: bank.to(device, non_blocking=True)
                    for class_id, bank in class_banks.items()
                }
                for layer_id, class_banks
                in self.tue_knn_reference_bank_score.items()
            }
            self.tue_knn_conformal_score = {
                class_id: values.to(device, non_blocking=True)
                for class_id, values in self.tue_knn_conformal_score.items()
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

    def _get_distances(
        self,
        confidence,
        selected_layers,
        diagrams,
        captures,
        references,
        conformal_references=None,
    ) -> tuple[Tensor, Tensor | None]:
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

        compute_pvalues = conformal_references is not None
        pvalues = (
            torch.full_like(distances, float("nan"))
            if compute_pvalues
            else None
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

            layer_conformal = (
                conformal_references.get(int(layer_id))
                if compute_pvalues
                else None
            )

            for batch_id, query_diagrams in enumerate(batch_diagrams):
                for query_id, diagram in query_diagrams.items():
                    layer_classes_output[
                        batch_id, query_id, layer_position
                    ] = layer_classes[batch_id, query_id]

                    distance_value = self._expected_distance(
                        diagram=diagram,
                        layer_id=layer_id,
                        class_ids=topk_classes[batch_id, query_id],
                        class_weights=topk_weights[batch_id, query_id],
                        references=references,
                    )
                    distances[
                        batch_id, query_id, layer_position
                    ] = distance_value

                    # Conformal p-value: how atypical is this distance for the
                    # query's (top-1) predicted class at this layer, relative to
                    # the calibration distances? Small p == atypically far ==
                    # likely OOD / misprediction. Consistent with the stored
                    # single-class calibration distances when tue_topk == 1.
                    if (
                        compute_pvalues
                        and layer_conformal is not None
                        and torch.isfinite(distance_value)
                    ):
                        class_id = int(layer_classes[batch_id, query_id].item())
                        array = layer_conformal.get(class_id)
                        if array is not None and array.numel() > 0:
                            pvalues[
                                batch_id, query_id, layer_position
                            ] = conformal_pvalue(array, distance_value.detach())

        return distances, pvalues

    def _get_classwise_pvalues(
        self,
        confidence,
        selected_layers,
        diagrams,
        references,
        conformal_references,
        num_classes,
    ) -> Tensor | None:
        """Return p-values shaped [B,Q,C,L] for the actual candidate class.

        A flattened RT-DETR query-class detection must use the p-value for that
        class. Reusing one top-1 query p-value for every class from the query
        gives wrong alternative-class detections an unrelated, often large,
        p-value.
        """
        if conformal_references is None:
            return None

        batch_size, num_queries = confidence.shape
        pvalues = torch.full(
            (batch_size, num_queries, num_classes, len(selected_layers)),
            fill_value=float("nan"),
            dtype=torch.float32,
            device=confidence.device,
        )

        for layer_position, layer_id in enumerate(selected_layers):
            layer_references = references.get(int(layer_id))
            layer_conformal = conformal_references.get(int(layer_id))
            if layer_references is None or layer_conformal is None:
                continue

            reference_ids, reference_vectors = layer_references
            batch_diagrams = diagrams.get(layer_id, [])

            for batch_id, query_diagrams in enumerate(batch_diagrams):
                for query_id, diagram in query_diagrams.items():
                    diagram = diagram.detach().to(
                        reference_vectors,
                        non_blocking=True,
                    ).flatten()
                    class_distances = torch.linalg.vector_norm(
                        reference_vectors - diagram,
                        dim=1,
                    )

                    for reference_position, class_tensor in enumerate(reference_ids):
                        class_id = int(class_tensor.item())
                        if not 0 <= class_id < num_classes:
                            continue
                        calibration = layer_conformal.get(class_id)
                        if calibration is None or calibration.numel() == 0:
                            continue
                        pvalues[
                            batch_id,
                            query_id,
                            class_id,
                            layer_position,
                        ] = conformal_pvalue(
                            calibration,
                            class_distances[reference_position].detach(),
                        )

        return pvalues

    def _get_knn_classwise_distances(
        self,
        confidence: Tensor,
        selected_layers: list[int],
        diagrams: dict,
        num_classes: int,
    ) -> Tensor:
        """Return class-conditional kNN diagram distances [B,Q,C,L]."""
        if self.tue_knn_reference_bank_score is None:
            raise RuntimeError("TUE-kNN reference bank is not loaded")

        batch_size, num_queries = confidence.shape
        distances = torch.full(
            (batch_size, num_queries, num_classes, len(selected_layers)),
            fill_value=float("nan"),
            dtype=torch.float32,
            device=confidence.device,
        )
        for layer_position, layer_id in enumerate(selected_layers):
            class_banks = self.tue_knn_reference_bank_score.get(int(layer_id), {})
            batch_diagrams = diagrams.get(layer_id, [])
            for batch_id, query_diagrams in enumerate(batch_diagrams):
                for query_id, diagram in query_diagrams.items():
                    for class_id, bank in class_banks.items():
                        if not 0 <= class_id < num_classes:
                            continue
                        diagram_vector = diagram.detach().to(
                            bank,
                            non_blocking=True,
                        ).flatten()
                        if diagram_vector.shape[0] != bank.shape[1]:
                            raise ValueError(
                                "TUE-kNN diagram/reference dimension mismatch: "
                                f"{diagram_vector.shape[0]} versus {bank.shape[1]}"
                            )
                        neighbour_count = min(self.tue_knn_k, bank.shape[0])
                        nearest = torch.linalg.vector_norm(
                            bank - diagram_vector,
                            dim=1,
                        ).topk(
                            neighbour_count,
                            largest=False,
                        ).values
                        distances[
                            batch_id,
                            query_id,
                            class_id,
                            layer_position,
                        ] = nearest.mean()
        return distances

    def _calibrate_knn_combined_distances(
        self,
        combined_distances: Tensor,
    ) -> Tensor:
        """Calibrate [B,Q,C] final kNN scores with held-out class arrays."""
        if self.tue_knn_conformal_score is None:
            raise RuntimeError("TUE-kNN conformal distances are not loaded")
        pvalues = torch.full_like(combined_distances, float("nan"))
        for class_id, calibration in self.tue_knn_conformal_score.items():
            if not 0 <= class_id < combined_distances.shape[-1]:
                continue
            measured = combined_distances[..., class_id]
            finite = torch.isfinite(measured)
            if finite.any() and calibration.numel() > 0:
                class_pvalues = conformal_pvalue(calibration, measured[finite])
                pvalues[..., class_id][finite] = class_pvalues
        return pvalues

    @staticmethod
    def _gather_class(values: Tensor, class_ids: Tensor) -> Tensor:
        """Gather class dimension 2 from [B,Q,C,...] using [B,Q] IDs."""
        trailing_shape = values.shape[3:]
        index = class_ids.view(*class_ids.shape, 1, *([1] * len(trailing_shape)))
        index = index.expand(*class_ids.shape, 1, *trailing_shape)
        return values.gather(2, index).squeeze(2)

    def _decoder_weighted_nanmean(
        self,
        values: Tensor,
        reduce_dims: tuple[int, ...],
    ) -> Tensor:
        """NaN-aware mean with configurable weight on the decoder dimension.

        The first reduction dimension is the decoder-layer dimension. Any
        additional dimensions (the bbox-head layer, for example) receive
        uniform weight.
        """
        dims = tuple(sorted({dim % values.dim() for dim in reduce_dims}))
        decoder_dim = dims[0]
        num_decoder_layers = values.shape[decoder_dim]
        configured = self.tue_decoder_layer_weights
        if configured is None:
            weights = torch.ones(
                num_decoder_layers,
                dtype=values.dtype,
                device=values.device,
            )
        else:
            if len(configured) != num_decoder_layers:
                raise ValueError(
                    "tue_decoder_layer_weights has "
                    f"{len(configured)} entries, but inference selected "
                    f"{num_decoder_layers} decoder layers"
                )
            weights = values.new_tensor(configured)
        weights = weights / weights.sum()
        weight_shape = [1] * values.dim()
        weight_shape[decoder_dim] = num_decoder_layers
        weights = weights.reshape(weight_shape)

        finite = torch.isfinite(values)
        effective_weights = finite.to(values.dtype) * weights
        numerator = (torch.nan_to_num(values, nan=0.0) * weights).sum(dim=dims)
        denominator = effective_weights.sum(dim=dims)
        mean = numerator / denominator.clamp_min(torch.finfo(values.dtype).eps)
        return torch.where(
            denominator > 0,
            mean,
            torch.full_like(mean, float("nan")),
        )

    def _reduce_pvalues(self, pvalues: Tensor, reduce_dims: tuple[int, ...]) -> Tensor:
        """Reduce repeated p-values without mixing class hypotheses.

        'last'       : use the final selected decoder/head layer; this is the
                       default because it remains a genuine single p-value.
        'bonferroni' : min(1, m * min(p)); valid under arbitrary dependence if
                       every component p-value is valid.
        'weighted_mean': configurable decoder-layer-weighted average. This is
                         an OOD score, not a combined conformal p-value.
        'mean'/'min'    : legacy heuristic modes retained for old
                         configurations; neither is a combined conformal
                         p-value.
        """
        dims = tuple(sorted({dim % pvalues.dim() for dim in reduce_dims}))

        if self.tue_pvalue_reduction == "last":
            reduced = pvalues
            for dim in sorted(dims, reverse=True):
                reduced = reduced.select(dim, -1)
            return reduced

        if self.tue_pvalue_reduction == "bonferroni":
            finite = torch.isfinite(pvalues)
            count = finite.sum(dim=dims)
            reduced = torch.nan_to_num(pvalues, nan=float("inf"))
            for dim in sorted(dims, reverse=True):
                reduced = reduced.min(dim=dim).values
            reduced = (reduced * count.to(reduced.dtype)).clamp(max=1.0)
            return torch.where(
                count > 0,
                reduced,
                torch.full_like(reduced, float("nan")),
            )

        if self.tue_pvalue_reduction == "min":
            filled = torch.nan_to_num(pvalues, nan=float("inf"))
            reduced = filled
            for dim in sorted(dims, reverse=True):
                reduced = reduced.min(dim=dim).values
            all_nan = torch.isinf(reduced)
            return torch.where(all_nan, torch.full_like(reduced, float("nan")), reduced)
        if self.tue_pvalue_reduction == "weighted_mean":
            return self._decoder_weighted_nanmean(pvalues, dims)
        return torch.nanmean(pvalues, dim=dims)

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
        final_classes = logits.argmax(dim=-1)
        device = logits.device
        self._move_references(device)

        probabilities = logits.sigmoid()
        confidence = probabilities.max(dim=-1).values
        # This gate controls computation only. Keep it fixed across evaluation
        # thresholds so changing an operating point does not change p-values.
        confidence_mask = confidence >= self.tue_confidence_threshold


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

        knn_combined_distances_classwise = None
        if self.tue_knn_reference_bank_score is not None:
            knn_distances_classwise = self._get_knn_classwise_distances(
                confidence=confidence,
                selected_layers=selected_layers_score,
                diagrams=diagrams_score,
                num_classes=logits.shape[-1],
            )
            knn_combined_distances_classwise = self._decoder_weighted_nanmean(
                knn_distances_classwise,
                (-1,),
            )
            pvalues_score_classwise = self._calibrate_knn_combined_distances(
                knn_combined_distances_classwise
            )
            distances_score = self._gather_class(
                knn_distances_classwise,
                final_classes,
            )
        else:
            distances_score, _ = self._get_distances(
                confidence,
                selected_layers_score,
                diagrams_score,
                captures_score,
                self.frechet_means_score,
                conformal_references=None,
            )
            pvalues_score_classwise = self._get_classwise_pvalues(
                confidence=confidence,
                selected_layers=selected_layers_score,
                diagrams=diagrams_score,
                references=self.frechet_means_score,
                conformal_references=self.conformal_distances_score,
                num_classes=logits.shape[-1],
            )

        bbox_distance_list = []
        bbox_pvalue_classwise_list = []
        for bbox_layer_id in range(num_bbox_layers):
            conformal_bbox = (
                self.conformal_distances_bbox.get(bbox_layer_id)
                if self.conformal_distances_bbox is not None
                else None
            )
            layer_distances, _ = self._get_distances(
                confidence,
                bbox_selected_layers[bbox_layer_id],
                diagrams_bbox[bbox_layer_id],
                bbox_captures[bbox_layer_id],
                self.frechet_means_bbox[bbox_layer_id],
                conformal_references=None,
            )
            bbox_distance_list.append(layer_distances)
            layer_pvalues_classwise = self._get_classwise_pvalues(
                confidence=confidence,
                selected_layers=bbox_selected_layers[bbox_layer_id],
                diagrams=diagrams_bbox[bbox_layer_id],
                references=self.frechet_means_bbox[bbox_layer_id],
                conformal_references=conformal_bbox,
                num_classes=logits.shape[-1],
            )
            if layer_pvalues_classwise is not None:
                bbox_pvalue_classwise_list.append(layer_pvalues_classwise)

        distances_bbox = torch.stack(bbox_distance_list, dim=-1)

        if knn_combined_distances_classwise is not None:
            uncertainty_score = self._gather_class(
                knn_combined_distances_classwise.unsqueeze(-1),
                final_classes,
            ).squeeze(-1)
        elif self.tue_decoder_layer_weights is None:
            uncertainty_score = torch.nanmean(distances_score, dim=-1)
        else:
            uncertainty_score = self._decoder_weighted_nanmean(
                distances_score,
                (-1,),
            )
        if self.tue_decoder_layer_weights is None:
            uncertainty_bbox = torch.nanmean(distances_bbox, dim=(-2, -1))
        else:
            uncertainty_bbox = self._decoder_weighted_nanmean(
                distances_bbox,
                (-2, -1),
            )

        outputs.update(
            tue_uncertainty_score=uncertainty_score,
            tue_uncertainty_bbox=uncertainty_bbox,
            tue_confidence=confidence,
            tue_distances_score=distances_score,
            tue_distances_bbox=distances_bbox,
        )

        # Class-specific conformal outputs. Small means atypically far from the
        # selected class reference. The [B,Q,C] tensor is the correct source for
        # flattened RT-DETR query-class detections. The [B,Q] tensor is retained
        # for top-1-query consumers and is gathered using the final model class.
        if pvalues_score_classwise is not None:
            if knn_combined_distances_classwise is not None:
                reduced_score_classwise = pvalues_score_classwise
                pvalues_score_output = pvalues_score_classwise.unsqueeze(-1)
            else:
                reduced_score_classwise = self._reduce_pvalues(
                    pvalues_score_classwise,
                    (-1,),
                )
                pvalues_score_output = pvalues_score_classwise
            final_score_pvalues = self._gather_class(
                reduced_score_classwise.unsqueeze(-1),
                final_classes,
            ).squeeze(-1)
            final_score_pvalues_per_layer = self._gather_class(
                pvalues_score_output,
                final_classes,
            )
            outputs.update(
                tue_pvalue_score=final_score_pvalues,
                tue_pvalues_score=final_score_pvalues_per_layer,
                tue_pvalue_score_classwise=reduced_score_classwise,
                tue_pvalues_score_classwise=pvalues_score_output,
            )
            if knn_combined_distances_classwise is not None:
                outputs["tue_knn_combined_distance_classwise"] = (
                    knn_combined_distances_classwise
                )

        if bbox_pvalue_classwise_list:
            pvalues_bbox_classwise = torch.stack(
                bbox_pvalue_classwise_list,
                dim=-1,
            )
            reduced_bbox_classwise = self._reduce_pvalues(
                pvalues_bbox_classwise,
                (-2, -1),
            )
            final_bbox_pvalues = self._gather_class(
                reduced_bbox_classwise.unsqueeze(-1),
                final_classes,
            ).squeeze(-1)
            final_bbox_pvalues_per_layer = self._gather_class(
                pvalues_bbox_classwise,
                final_classes,
            )
            outputs.update(
                tue_pvalue_bbox=final_bbox_pvalues,
                tue_pvalues_bbox=final_bbox_pvalues_per_layer,
                tue_pvalue_bbox_classwise=reduced_bbox_classwise,
                tue_pvalues_bbox_classwise=pvalues_bbox_classwise,
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
