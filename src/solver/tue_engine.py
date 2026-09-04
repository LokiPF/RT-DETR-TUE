"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import torch
import torch.amp
from torch import nn

from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue

# from ..misc.tue_utils import (
#     LayerClassBuckets,
#     get_captured_persistence_diagrams,
#     hook_decoder_layers,
# )


def _update_buckets(
    diagrams,
    captures,
    matched_labels,
    buckets,
    allowed_class_ids: frozenset[int] | None = None,
):
    for layer_id, batch_diagrams in diagrams.items():
        layer_classes = captures[layer_id]["logits"].argmax(dim=-1)

        for batch_id, query_diagrams in enumerate(batch_diagrams):
            for query_id, diagram in query_diagrams.items():
                gt_label = int(matched_labels[batch_id, query_id].item())
                if gt_label < 0:
                    continue

                class_id = int(
                    layer_classes[
                        batch_id,
                        query_id,
                    ].item()
                )
                if class_id != gt_label:
                    continue

                if allowed_class_ids is not None and class_id not in allowed_class_ids:
                    continue

                # start_time = time.perf_counter()
                buckets.update(
                    diagram=diagram,
                    layer_id=layer_id,
                    class_id=class_id,
                )
                # end_time = time.perf_counter()
                # elapsed_times = np.append(elapsed_times, end_time-start_time)


@torch.inference_mode()
def collect_persistence_one_epoch(
    model: nn.Module,
    matcher: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    confidence_threshold: float = 0.8,
    decoder_layers: int | Iterable[int] | None = None,
    class_ids: Iterable[int] | None = None,
    print_freq: int = 10,
    data_fraction: float = 1.0,
) -> tuple[LayerClassBuckets, dict, dict]:
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between zero and one")

    if not 0.0 < data_fraction <= 1.0:
        raise ValueError("data_fraction must be in the interval (0, 1]")

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

    num_batches = max(
        1,
        math.ceil(len(data_loader) * data_fraction),
    )

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "selected_queries",
        SmoothedValue(window_size=20, fmt="{avg:.2f}"),
    )

    header = f"Persistence epoch: [{epoch}]"

    transformer = model.decoder

    buckets_score = LayerClassBuckets(
        num_layers=len(transformer.dec_score_head),
        num_classes=transformer.num_classes,
    )

    num_bbox_layers = len(transformer.dec_bbox_head[0].layers)

    buckets_bbox = {
        bbox_layer_id: LayerClassBuckets(
            num_layers=len(transformer.dec_bbox_head),
            num_classes=transformer.num_classes,
        )
        for bbox_layer_id in range(num_bbox_layers)
    }
    captures_score, handles_score, selected_layers_score = hook_decoder_layers(
        transformer=transformer, decoder_layers=decoder_layers, head_task="score"
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
        "class_ids": (None if allowed_class_ids is None else sorted(allowed_class_ids)),
    }

    elapsed_times = np.array([])

    try:
        for step, (samples, targets) in enumerate(
            metric_logger.log_every(
                data_loader,
                print_freq,
                header,
            )
        ):
            if step >= num_batches:
                break

            captures_score.clear()
            for captures in captures_bbox.values():
                captures.clear()
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(samples)

            # [B, num_queries, num_classes]
            probabilities = outputs["pred_logits"].sigmoid()

            # Both have shape [B, num_queries].
            confidence, _ = probabilities.max(dim=-1)

            confidence_mask = confidence > confidence_threshold

            batch_size, num_queries = confidence.shape
            matched_labels = torch.full(
                (batch_size, num_queries),
                fill_value=-1,
                dtype=torch.long,
                device=device,
            )

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

            _update_buckets(
                diagrams_score,
                captures_score,
                matched_labels,
                buckets_score,
                allowed_class_ids,
            )

            for bbox_layer_id in range(num_bbox_layers):
                diagrams_bbox = get_captured_persistence_diagrams(
                    captures=captures_bbox[bbox_layer_id],
                    query_indices=query_indices,
                    decoder_layer_indices=selected_layers_bbox[bbox_layer_id],
                )

                _update_buckets(
                    diagrams_bbox,
                    captures_bbox[bbox_layer_id],
                    matched_labels,
                    buckets_bbox[bbox_layer_id],
                    allowed_class_ids,
                )

            global_step = epoch * len(data_loader) + step

            metas = {
                "epoch": epoch,
                "step": step,
                "global_step": global_step,
                "elapsed_times": np.mean(elapsed_times),
                "class_ids": (
                    None if allowed_class_ids is None else sorted(allowed_class_ids)
                ),
            }

    finally:
        for handle in handles_score + handles_bbox:
            handle.remove()

        model.train(previous_training_mode)

    return buckets_score, buckets_bbox, metas


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device,
):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = "Test:"

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
        label2category = getattr(data_loader.dataset, "label2category", None)
        if label2category is not None and not getattr(
            postprocessor, "remap_mscoco_category", False
        ):
            for result in results:
                labels = result["labels"]
                result["labels"] = torch.tensor(
                    [label2category[int(x)] for x in labels.flatten()],
                    dtype=labels.dtype,
                    device=labels.device,
                ).reshape(labels.shape)

        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results)
        }
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
        if "bbox" in iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in iou_types:
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    return stats, coco_evaluator
