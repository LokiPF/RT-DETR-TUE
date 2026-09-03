"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import math
import sys
from collections.abc import Iterable

import torch
import torch.amp
from supervisely.nn.training import train_logger
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils
from ..misc.tue_dataclasses import DiagramStatistics
from ..misc.tue_utils import (
    accumulate_diagrams,
    compute_means,
    initialize_statistics,
    run_detector,
    select_confident_matches,
)
from ..optim import ModelEMA, Warmup


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    **kwargs,
):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    print_freq = kwargs.get("print_freq", 10)
    writer: SummaryWriter = kwargs.get("writer", None)

    ema: ModelEMA = kwargs.get("ema", None)
    scaler: GradScaler = kwargs.get("scaler", None)
    lr_warmup_scheduler: Warmup = kwargs.get("lr_warmup_scheduler", None)

    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step)

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)

            loss: torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process():
            writer.add_scalar("Loss/total", loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"Lr/pg_{j}", pg["lr"], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f"Loss/{k}", v.item(), global_step)

        train_logger.step_finished()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


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


def reduce_statistics(statistics, device):
    if not dist_utils.is_dist_available_and_initialized():
        return

    sums = statistics.sums.to(device)
    counts = statistics.counts.to(device)

    torch.distributed.all_reduce(sums)
    torch.distributed.all_reduce(counts)

    statistics.sums = sums.cpu()
    statistics.counts = counts.cpu()


@torch.no_grad()
def build_frechet_means(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    min_confidence: float = 0.5,
):
    model.eval()
    criterion.eval()

    statistics_by_module: dict[str, DiagramStatistics] = {}
    saw_batch = False

    metric_logger = MetricLogger(delimiter="  ")

    for samples, targets in metric_logger.log_every(
        data_loader,
        10,
        "Fréchet calibration:",
    ):
        saw_batch = True

        samples = samples.to(device)
        targets = [
            {key: value.to(device) for key, value in target.items()}
            for target in targets
        ]

        outputs = run_detector(model, samples)

        query_indices, query_classes = select_confident_matches(
            outputs=outputs,
            targets=targets,
            matcher=criterion.matcher,
            device=device,
            min_confidence=min_confidence,
        )

        captures = outputs["tue_info"].data

        if not captures:
            continue

        num_classes = outputs["pred_logits"].shape[-1]

        for module_name, capture in captures.items():
            if module_name not in statistics_by_module:
                statistics_by_module[module_name] = initialize_statistics(
                    capture=capture,
                    num_classes=num_classes,
                )

        accumulate_diagrams(
            statistics_by_module=statistics_by_module,
            captures=captures,
            query_indices=query_indices,
            query_classes=query_classes,
        )

    if not saw_batch:
        raise RuntimeError("The calibration dataloader was empty")

    if not statistics_by_module:
        raise RuntimeError("No modules were captured")

    frechet_means = {}

    for module_name, statistics in statistics_by_module.items():
        reduce_statistics(statistics, device)

        frechet_means[module_name] = {
            "means": compute_means(statistics),
            "counts": statistics.counts,
            "min_confidence": min_confidence,
        }

    return frechet_means
