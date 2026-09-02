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
from ..misc.tue_utils import get_captured_persistence_diagrams
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

    diagram_sums = None
    diagram_counts = None
    selected_layers = None

    metric_logger = MetricLogger(delimiter="  ")

    for samples, targets in metric_logger.log_every(
        data_loader,
        10,
        "Fréchet calibration:",
    ):
        samples = samples.to(device)
        targets = [
            {key: value.to(device) for key, value in target.items()}
            for target in targets
        ]

        if hasattr(model, "forward_detector"):
            outputs = model.forward_detector(samples)
        else:
            outputs = model(samples)

        score_captures = outputs["tue_info"]["score"]

        if not score_captures:
            raise RuntimeError("No score layers were captured")

        if selected_layers is None:
            selected_layers = sorted(score_captures)

            first_capture = score_captures[selected_layers[0]]
            num_classes, hidden_dim = first_capture["weight"].shape
            diagram_size = num_classes + hidden_dim - 1
            num_layers = max(selected_layers) + 1

            diagram_sums = torch.zeros(
                num_layers,
                num_classes,
                diagram_size,
                dtype=torch.float64,
                device="cpu",
            )

            diagram_counts = torch.zeros(
                num_layers,
                num_classes,
                dtype=torch.long,
                device="cpu",
            )

        matcher_outputs = {
            "pred_logits": outputs["pred_logits"],
            "pred_boxes": outputs["pred_boxes"],
        }

        matcher_result = criterion.matcher(
            matcher_outputs,
            targets,
        )

        matched_indices = matcher_result["indices"]

        probabilities = outputs["pred_logits"].sigmoid()

        query_indices = []
        query_classes = []

        for batch_id, (query_ids, target_ids) in enumerate(matched_indices):
            query_ids = query_ids.to(device=device, dtype=torch.long)
            target_ids = target_ids.to(device=device, dtype=torch.long)

            target_classes = targets[batch_id]["labels"].index_select(
                0,
                target_ids,
            )

            matched_probabilities = probabilities[
                batch_id,
                query_ids,
            ]

            predicted_classes = matched_probabilities.argmax(dim=-1)

            target_confidence = matched_probabilities.gather(
                dim=1,
                index=target_classes[:, None],
            ).squeeze(1)

            keep = (predicted_classes == target_classes) & (
                target_confidence >= min_confidence
            )

            query_indices.append(query_ids[keep])
            query_classes.append(target_classes[keep])

        diagrams = get_captured_persistence_diagrams(
            captures=score_captures,
            query_indices=query_indices,
            decoder_layer_indices=selected_layers,
            # Usually processes the whole image batch in one Prim call.
            chunk_size=512,
        )

        class_lookups = [
            {
                int(query_id): int(class_id)
                for query_id, class_id in zip(
                    batch_queries.detach().cpu().tolist(),
                    batch_classes.detach().cpu().tolist(),
                )
            }
            for batch_queries, batch_classes in zip(
                query_indices,
                query_classes,
            )
        ]

        for layer_id, batch_diagrams in diagrams.items():
            for batch_id, query_diagrams in enumerate(batch_diagrams):
                for query_id, diagram in query_diagrams.items():
                    class_id = class_lookups[batch_id][query_id]

                    diagram_sums[layer_id, class_id].add_(diagram.to(torch.float64))
                    diagram_counts[layer_id, class_id] += 1

    if diagram_sums is None:
        raise RuntimeError("The calibration dataloader was empty")

    # Combine statistics from all distributed workers.
    if dist_utils.is_dist_available_and_initialized():
        sums_device = diagram_sums.to(device)
        counts_device = diagram_counts.to(device)

        torch.distributed.all_reduce(
            sums_device,
            op=torch.distributed.ReduceOp.SUM,
        )
        torch.distributed.all_reduce(
            counts_device,
            op=torch.distributed.ReduceOp.SUM,
        )

        diagram_sums = sums_device.cpu()
        diagram_counts = counts_device.cpu()

    means = (diagram_sums / diagram_counts.clamp_min(1).unsqueeze(-1)).to(torch.float32)

    # Missing layer/class combinations remain explicitly invalid.
    means[diagram_counts == 0] = float("nan")

    return {
        "means": means,
        "counts": diagram_counts,
        "selected_layers": selected_layers,
        "min_confidence": min_confidence,
    }
