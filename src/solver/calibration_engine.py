from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn

from ..misc import MetricLogger, SmoothedValue

# from ..misc.tue_utils import (
#     diagram_distance,
#     get_captured_persistence_diagrams,
#     hook_decoder_layers,
# )

CALIBRATION_POPULATIONS = ("correct", "predicted")


class LayerClassDistances:
    def __init__(self, num_layers: int, num_classes: int):
        self.num_layers = num_layers
        self.num_classes = num_classes
        self._distances: dict[tuple[int, int], list[float]] = {}

    def append(self, layer_id: int, class_id: int, distance: float) -> None:
        key = (int(layer_id), int(class_id))
        self._distances.setdefault(key, []).append(float(distance))

    def count(self, layer_id: int, class_id: int) -> int:
        return len(self._distances.get((int(layer_id), int(class_id)), []))

    @property
    def counts(self) -> dict[tuple[int, int], int]:
        return {key: len(values) for key, values in self._distances.items()}

    def to_sorted_arrays(self) -> dict[int, dict[int, torch.Tensor]]:
        packed: dict[int, dict[int, torch.Tensor]] = {}
        for (layer_id, class_id), values in self._distances.items():
            tensor = torch.tensor(sorted(values), dtype=torch.float32)
            packed.setdefault(layer_id, {})[class_id] = tensor
        return packed


def _prepare_means(
    frechet_means: dict,
    device: torch.device,
) -> dict[int, dict[int, torch.Tensor]]:
    prepared: dict[int, dict[int, torch.Tensor]] = {}
    for layer_id, class_means in frechet_means.items():
        prepared[int(layer_id)] = {
            int(class_id): torch.as_tensor(mean).detach().flatten().to(device)
            for class_id, mean in class_means.items()
        }
    return prepared


def _collect_distances(
    diagrams: dict,
    captures: dict,
    matched_labels: torch.Tensor,
    means: dict[int, dict[int, torch.Tensor]],
    distances: LayerClassDistances,
    population: str = "correct",
    allowed_class_ids: frozenset[int] | None = None,
) -> None:
    for layer_id, batch_diagrams in diagrams.items():
        layer_classes = captures[layer_id]["logits"].argmax(dim=-1)
        layer_means = means.get(int(layer_id))
        if layer_means is None:
            continue

        for batch_id, query_diagrams in enumerate(batch_diagrams):
            for query_id, diagram in query_diagrams.items():
                predicted = int(layer_classes[batch_id, query_id].item())

                if population == "correct":
                    gt_label = int(matched_labels[batch_id, query_id].item())
                    if gt_label < 0:
                        continue  # unmatched -> likely false positive/duplicate
                    if predicted != gt_label:
                        continue  # confidently wrong -> excluded, as for the means
                    class_id = gt_label
                else:  # 'predicted': the scored population, no correctness filter
                    class_id = predicted

                if allowed_class_ids is not None and class_id not in allowed_class_ids:
                    continue

                reference = layer_means.get(class_id)
                if reference is None:
                    # No Frechet mean for this (layer, class); cannot calibrate.
                    continue

                distance = diagram_distance(diagram, reference)
                distances.append(
                    layer_id=layer_id,
                    class_id=class_id,
                    distance=distance.item(),
                )


@torch.inference_mode()
def collect_conformal_distances_one_epoch(
    model: nn.Module,
    matcher: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    frechet_means_score: dict,
    frechet_means_bbox: dict,
    epoch: int = 0,
    confidence_threshold: float = 0.8,
    decoder_layers: int | Iterable[int] | None = None,
    class_ids: Iterable[int] | None = None,
    print_freq: int = 10,
    data_fraction: float = 1.0,
    calibration_population: str = "correct",
) -> tuple[LayerClassDistances, dict[int, LayerClassDistances], dict]:
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between zero and one")
    if not 0.0 < data_fraction <= 1.0:
        raise ValueError("data_fraction must be in the interval (0, 1]")
    if calibration_population not in CALIBRATION_POPULATIONS:
        raise ValueError(
            f"calibration_population must be one of {CALIBRATION_POPULATIONS}, "
            f"got {calibration_population!r}"
        )

    allowed_class_ids = (
        None
        if class_ids is None
        else frozenset(int(class_id) for class_id in class_ids)
    )
    if allowed_class_ids is not None:
        invalid = sorted(
            class_id
            for class_id in allowed_class_ids
            if not 0 <= class_id < model.decoder.num_classes
        )
        if invalid:
            raise ValueError(f"Invalid calibration class IDs: {invalid}")
        if not allowed_class_ids:
            raise ValueError("class_ids cannot be empty")

    num_batches = max(1, math.ceil(len(data_loader) * data_fraction))

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "selected_queries",
        SmoothedValue(window_size=20, fmt="{avg:.2f}"),
    )
    header = f"Conformal epoch: [{epoch}] ({calibration_population})"

    transformer = model.decoder

    means_score = _prepare_means(frechet_means_score, device)
    means_bbox = {
        int(bbox_layer_id): _prepare_means(layer_means, device)
        for bbox_layer_id, layer_means in frechet_means_bbox.items()
    }

    distances_score = LayerClassDistances(
        num_layers=len(transformer.dec_score_head),
        num_classes=transformer.num_classes,
    )

    num_bbox_layers = len(transformer.dec_bbox_head[0].layers)
    distances_bbox = {
        bbox_layer_id: LayerClassDistances(
            num_layers=len(transformer.dec_bbox_head),
            num_classes=transformer.num_classes,
        )
        for bbox_layer_id in range(num_bbox_layers)
    }

    captures_score, handles_score, selected_layers_score = hook_decoder_layers(
        transformer=transformer,
        decoder_layers=decoder_layers,
        head_task="score",
    )

    captures_bbox = {}
    selected_layers_bbox = {}
    handles_bbox = []
    for bbox_layer_id in range(num_bbox_layers):
        captures, handles, selected = hook_decoder_layers(
            transformer=transformer,
            decoder_layers=decoder_layers,
            head_task="bbox",
            bbox_head_layer=bbox_layer_id,
        )
        captures_bbox[bbox_layer_id] = captures
        selected_layers_bbox[bbox_layer_id] = selected
        handles_bbox.extend(handles)

    previous_training_mode = model.training
    model.eval()

    metas = {
        "epoch": epoch,
        "step": -1,
        "global_step": epoch * len(data_loader),
        "calibration_population": calibration_population,
        "class_ids": (None if allowed_class_ids is None else sorted(allowed_class_ids)),
    }

    try:
        for step, (samples, targets) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)
        ):
            if step >= num_batches:
                break

            captures_score.clear()
            for captures in captures_bbox.values():
                captures.clear()

            samples = samples.to(device)
            targets = [
                {key: value.to(device) for key, value in target.items()}
                for target in targets
            ]

            outputs = model(samples)

            probabilities = outputs["pred_logits"].sigmoid()
            confidence, _ = probabilities.max(dim=-1)
            confidence_mask = confidence > confidence_threshold

            batch_size, num_queries = confidence.shape
            matched_labels = torch.full(
                (batch_size, num_queries),
                fill_value=-1,
                dtype=torch.long,
                device=device,
            )

            # Matching is only needed for the 'correct' population.
            if calibration_population == "correct":
                match_indices = matcher(outputs, targets)["indices"]
                for batch_id, (query_idx, target_idx) in enumerate(match_indices):
                    matched_labels[batch_id, query_idx] = targets[batch_id]["labels"][
                        target_idx
                    ]

            query_indices = [
                torch.where(confidence_mask[batch_id])[0]
                for batch_id in range(confidence_mask.shape[0])
            ]

            selected_query_count = sum(indices.numel() for indices in query_indices)
            metric_logger.update(
                selected_queries=selected_query_count / confidence.shape[0]
            )

            missing_score = set(selected_layers_score) - set(captures_score)
            missing_bbox = {
                bbox_layer_id: sorted(
                    set(selected_layers_bbox[bbox_layer_id])
                    - set(captures_bbox[bbox_layer_id])
                )
                for bbox_layer_id in range(num_bbox_layers)
            }
            missing_bbox = {
                bbox_layer_id: missing
                for bbox_layer_id, missing in missing_bbox.items()
                if missing
            }
            if missing_score or missing_bbox:
                raise RuntimeError(
                    f"Missing score layers: {sorted(missing_score)}; "
                    f"missing bbox layers: {missing_bbox}"
                )

            diagrams_score = get_captured_persistence_diagrams(
                captures=captures_score,
                query_indices=query_indices,
                decoder_layer_indices=selected_layers_score,
            )
            _collect_distances(
                diagrams_score,
                captures_score,
                matched_labels,
                means_score,
                distances_score,
                population=calibration_population,
                allowed_class_ids=allowed_class_ids,
            )

            for bbox_layer_id in range(num_bbox_layers):
                diagrams_bbox = get_captured_persistence_diagrams(
                    captures=captures_bbox[bbox_layer_id],
                    query_indices=query_indices,
                    decoder_layer_indices=selected_layers_bbox[bbox_layer_id],
                )
                layer_means = means_bbox.get(bbox_layer_id, {})
                _collect_distances(
                    diagrams_bbox,
                    captures_bbox[bbox_layer_id],
                    matched_labels,
                    layer_means,
                    distances_bbox[bbox_layer_id],
                    population=calibration_population,
                    allowed_class_ids=allowed_class_ids,
                )

            metas.update(
                step=step,
                global_step=epoch * len(data_loader) + step,
            )
    finally:
        for handle in handles_score + handles_bbox:
            handle.remove()
        model.train(previous_training_mode)

    return distances_score, distances_bbox, metas
