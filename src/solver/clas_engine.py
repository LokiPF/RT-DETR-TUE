"""Copyright(c) 2023 lyuwenyu. All Rights Reserved."""

from collections.abc import Iterable, Mapping

import torch
from torch import nn

from ..misc import MetricLogger, SmoothedValue, dist_utils, reduce_dict
from ..misc.tue_dataclasses import CaptureGroup, DiagramStatistics, LayerCapture

# from ..misc.tue_utils import (
#     compute_means,
#     get_persistence_diagrams_batched,
#     initialize_statistics,
# )


def train_one_epoch(
    model: nn.Module, criterion: nn.Module, dataloader, optimizer, ema, epoch, device
):
    """ """
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    print_freq = 100
    header = f"Epoch: [{epoch}]"

    for imgs, labels in metric_logger.log_every(dataloader, print_freq, header):
        imgs = imgs.to(device)
        labels = labels.to(device)

        preds = model(imgs)
        loss: torch.Tensor = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema is not None:
            ema.update(model)

        loss_reduced_values = {
            k: v.item() for k, v in reduce_dict({"loss": loss}).items()
        }
        metric_logger.update(**loss_reduced_values)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats


@torch.no_grad()
def evaluate(model, criterion, dataloader, device):
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('acc', SmoothedValue(window_size=1, fmt='{global_avg:.4f}'))
    # metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter("acc", SmoothedValue(window_size=1))
    metric_logger.add_meter("loss", SmoothedValue(window_size=1))

    header = "Test:"
    for imgs, labels in metric_logger.log_every(dataloader, 10, header):
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)

        acc = (preds.argmax(dim=-1) == labels).sum() / preds.shape[0]
        loss = criterion(preds, labels)

        dict_reduced = reduce_dict({"acc": acc, "loss": loss})
        reduced_values = {k: v.item() for k, v in dict_reduced.items()}
        metric_logger.update(**reduced_values)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats


def _classification_captures(model: nn.Module) -> Mapping[str, LayerCapture]:
    model = dist_utils.de_parallel(model)
    capture_group = getattr(model, "captures", None)

    if capture_group is None:
        decoder = getattr(model, "decoder", None)
        capture_group = getattr(decoder, "captures", None)

    if not isinstance(capture_group, CaptureGroup):
        raise RuntimeError(
            "The classification model did not expose a CaptureGroup on "
            "'captures' or 'decoder.captures'"
        )

    return capture_group.data


def _accumulate_classification_diagrams(
    statistics: DiagramStatistics,
    capture: LayerCapture,
    labels: torch.Tensor,
    selected_images: torch.Tensor,
    *,
    chunk_size: int | None,
    use_compile: bool,
) -> None:
    layer_inputs = capture.input
    if layer_inputs.ndim < 2:
        raise ValueError(
            "A classification capture must have a batch and feature dimension, "
            f"got {tuple(layer_inputs.shape)}"
        )
    if layer_inputs.shape[0] != labels.shape[0]:
        raise ValueError(
            "Classification logits and captured inputs have different batch sizes"
        )
    if layer_inputs.shape[-1] != capture.weight.shape[1]:
        raise ValueError(
            "The captured input width does not match the linear layer weight"
        )

    # RT-DETR classification averages its queries to obtain one prediction per
    # image. Mirror that aggregation here so every accepted image contributes
    # one persistence diagram per layer, independent of its token/query count.
    image_inputs = layer_inputs.reshape(
        layer_inputs.shape[0], -1, layer_inputs.shape[-1]
    )
    image_inputs = image_inputs.mean(dim=1)
    image_inputs = image_inputs[selected_images]
    class_ids = labels[selected_images]

    if image_inputs.shape[0] == 0:
        return

    diagrams = get_persistence_diagrams_batched(
        capture.weight,
        image_inputs,
        chunk_size=chunk_size,
        use_compile=use_compile,
    )
    if diagrams.shape[-1] != statistics.sums.shape[-1]:
        raise ValueError("The persistence diagram size changed between batches")

    class_ids = class_ids.to(device=statistics.sums.device, dtype=torch.long)
    statistics.sums.index_add_(
        0,
        class_ids,
        diagrams.to(
            device=statistics.sums.device,
            dtype=statistics.sums.dtype,
        ),
    )
    statistics.counts.add_(
        torch.bincount(class_ids, minlength=statistics.counts.numel())
    )


def _reduce_classification_statistics(
    statistics: DiagramStatistics,
    device: torch.device,
) -> None:
    sums = statistics.sums.to(device)
    counts = statistics.counts.to(device)

    if dist_utils.is_dist_available_and_initialized():
        torch.distributed.all_reduce(sums)
        torch.distributed.all_reduce(counts)

    statistics.sums = sums.cpu()
    statistics.counts = counts.cpu()


@torch.no_grad()
def build_frechet_means(
    model: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    min_confidence: float = 0.5,
    *,
    chunk_size: int | None = None,
    use_compile: bool = False,
) -> dict[str, dict[str, torch.Tensor | float]]:
    """Build class-conditioned Fréchet means from classification captures."""
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between zero and one")

    model.eval()
    statistics_by_module: dict[str, DiagramStatistics] = {}
    saw_batch = False
    metric_logger = MetricLogger(delimiter="  ")

    for images, labels in metric_logger.log_every(
        data_loader,
        10,
        "Fréchet calibration:",
    ):
        saw_batch = True
        images = images.to(device)
        labels = labels.to(device=device, dtype=torch.long)
        logits = model(images)

        if logits.ndim != 2:
            raise ValueError(
                "Classification logits must have shape [batch, classes], "
                f"got {tuple(logits.shape)}"
            )
        if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
            raise ValueError("Classification labels must have shape [batch]")

        num_classes = logits.shape[-1]
        if torch.any((labels < 0) | (labels >= num_classes)):
            raise IndexError(f"Classification labels must be in [0, {num_classes - 1}]")

        probabilities = logits.softmax(dim=-1)
        predictions = probabilities.argmax(dim=-1)
        target_confidence = probabilities.gather(1, labels[:, None]).squeeze(1)
        selected_images = (predictions == labels) & (
            target_confidence >= min_confidence
        )

        captures = _classification_captures(model)
        missing_modules = set(statistics_by_module) - set(captures)
        if missing_modules:
            raise RuntimeError(
                "Previously captured modules were missing from this batch: "
                f"{sorted(missing_modules)}"
            )
        if not captures:
            continue

        for module_name, capture in captures.items():
            if module_name not in statistics_by_module:
                statistics_by_module[module_name] = initialize_statistics(
                    capture,
                    num_classes,
                )

            _accumulate_classification_diagrams(
                statistics_by_module[module_name],
                capture,
                labels,
                selected_images,
                chunk_size=chunk_size,
                use_compile=use_compile,
            )

    if not saw_batch:
        raise RuntimeError("The calibration dataloader was empty")
    if not statistics_by_module:
        raise RuntimeError("No classification modules were captured")

    frechet_means: dict[str, dict[str, torch.Tensor | float]] = {}
    for module_name, statistics in statistics_by_module.items():
        _reduce_classification_statistics(statistics, device)
        frechet_means[module_name] = {
            "means": compute_means(statistics),
            "counts": statistics.counts,
            "min_confidence": min_confidence,
        }

    return frechet_means
