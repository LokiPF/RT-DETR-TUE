"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""
from __future__ import annotations

import sys
import math
import time
from typing import Iterable

import numpy as np
import torch
import torch.amp 
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..misc.tue_utils import get_captured_persistence_diagrams, LayerClassBuckets
from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from supervisely.nn.training import train_logger


import torch.nn.functional as F

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..misc.tue_utils import (
    LayerClassBuckets,
    get_captured_persistence_diagrams,
)
from ..misc import MetricLogger, SmoothedValue


def hook_decoder_layers(
    transformer: nn.Module,
    decoder_layers: int | Iterable[int] | None = None,
) -> tuple[
    dict[int, dict[str, Tensor]],
    list[torch.utils.hooks.RemovableHandle],
    list[int],
]:
    """
    Capture the input to each selected decoder classification head.

    Decoder-layer outputs are captured because intermediate classification
    heads are not normally executed during evaluation.
    """
    decoder = transformer.decoder
    heads = transformer.dec_score_head
    num_layers = len(decoder.layers)

    if decoder_layers is None:
        requested_layers = list(range(num_layers))
    elif isinstance(decoder_layers, int):
        requested_layers = [decoder_layers]
    else:
        requested_layers = list(decoder_layers)

    normalized_layers: list[int] = []

    for layer_id in requested_layers:
        if not -num_layers <= layer_id < num_layers:
            raise IndexError(f"Invalid decoder layer: {layer_id}")

        layer_id = layer_id % num_layers

        if layer_id not in normalized_layers:
            normalized_layers.append(layer_id)

    captures: dict[int, dict[str, Tensor]] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for layer_id in normalized_layers:
        decoder_layer = decoder.layers[layer_id]
        classification_head = heads[layer_id]

        def make_hook(
            index: int,
            head: nn.Linear,
        ):
            def hook(
                module: nn.Module,
                inputs: tuple[Tensor, ...],
                output: Tensor,
            ) -> None:
                features = output.detach()
                weight = head.weight.detach()
                bias = (
                    head.bias.detach()
                    if head.bias is not None
                    else None
                )

                captures[index] = {
                    "input": features,
                    "weight": weight,
                    "logits": F.linear(
                        features,
                        weight,
                        bias,
                    ).detach(),
                }

            return hook

        handle = decoder_layer.register_forward_hook(
            make_hook(layer_id, classification_head)
        )

        handles.append(handle)

    return captures, handles, normalized_layers


@torch.inference_mode()
def collect_persistence_one_epoch(
    model: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    confidence_threshold: float = 0.8,
    decoder_layers: int | Iterable[int] | None = None,
    print_freq: int = 10,
) -> tuple[LayerClassBuckets, dict]:
    """
    Collect persistence diagrams for one pass over a dataset.

    This is an analysis function; it does not train the model.
    """
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError(
            "confidence_threshold must be between zero and one"
        )

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "selected_queries",
        SmoothedValue(window_size=20, fmt="{avg:.2f}"),
    )

    header = f"Persistence epoch: [{epoch}]"

    transformer = model.decoder

    buckets = LayerClassBuckets(
        num_layers=len(transformer.dec_score_head),
        num_classes=transformer.num_classes,
    )

    captures, handles, selected_layers = hook_decoder_layers(
        transformer=transformer,
        decoder_layers=decoder_layers,
    )

    previous_training_mode = model.training
    model.eval()

    metas = {
        "epoch": epoch,
        "step": -1,
        "global_step": epoch * len(data_loader),
    }

    elapsed_times = np.array([])

    try:
        for step, (samples, _) in enumerate(
            metric_logger.log_every(
                data_loader,
                print_freq,
                header,
            )
        ):
            captures.clear()
            samples = samples.to(device)

            outputs = model(samples)

            # [B, num_queries, num_classes]
            probabilities = outputs["pred_logits"].sigmoid()

            # Both have shape [B, num_queries].
            confidence, _ = probabilities.max(dim=-1)

            confidence_mask = confidence > confidence_threshold

            query_indices = [
                torch.where(confidence_mask[batch_id])[0]
                for batch_id in range(confidence_mask.shape[0])
            ]

            selected_query_count = sum(
                indices.numel()
                for indices in query_indices
            )

            metric_logger.update(
                selected_queries=selected_query_count
                / confidence.shape[0]
            )

            missing_layers = set(selected_layers) - set(captures)

            if missing_layers:
                raise RuntimeError(
                    "The following decoder layers did not execute: "
                    f"{sorted(missing_layers)}. Check decoder.eval_idx."
                )

            diagrams = get_captured_persistence_diagrams(
                captures=captures,
                query_indices=query_indices,
                decoder_layer_indices=selected_layers,
            )

            for layer_id, batch_diagrams in diagrams.items():
                layer_classes = captures[layer_id][
                    "logits"
                ].argmax(dim=-1)

                for batch_id, query_diagrams in enumerate(
                    batch_diagrams
                ):
                    for query_id, diagram in query_diagrams.items():
                        class_id = int(
                            layer_classes[
                                batch_id,
                                query_id,
                            ].item()
                        )
                        start_time = time.perf_counter()
                        buckets.update(
                            diagram=diagram,
                            layer_id=layer_id,
                            class_id=class_id,
                        )
                        end_time = time.perf_counter()
                        elapsed_times = np.append(elapsed_times, end_time-start_time)

            global_step = epoch * len(data_loader) + step

            metas = {
                "epoch": epoch,
                "step": step,
                "global_step": global_step,
                "elapsed_times": np.mean(elapsed_times),
            }

    finally:
        for handle in handles:
            handle.remove()

        model.train(previous_training_mode)

    return buckets, metas


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'
    
    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        # with torch.autocast(device_type=str(device)):
        #     outputs = model(samples)

        # TODO (lyuwenyu), fix dataset converted using `convert_to_coco_api`?
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # orig_target_sizes = torch.tensor([[samples.shape[-1], samples.shape[-2]]], device=samples.device)
        
        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        # predictions carry 0-based label indices, while GT annotations use dataset
        # category ids (1-based for COCO-style json); remap unless the postprocessor
        # already did it (remap_mscoco_category=True)
        label2category = getattr(data_loader.dataset, 'label2category', None)
        if label2category is not None and not getattr(postprocessor, 'remap_mscoco_category', False):
            for result in results:
                labels = result['labels']
                result['labels'] = torch.tensor(
                    [label2category[int(x)] for x in labels.flatten()],
                    dtype=labels.dtype,
                    device=labels.device,
                ).reshape(labels.shape)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
            
    return stats, coco_evaluator



