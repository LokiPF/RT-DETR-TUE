#!/usr/bin/env python3
"""Shared matched-localization AUROC evaluation for RT-DETRv2."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor, nn


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.core import YAMLConfig
except ImportError as exc:
    raise ImportError(
        "Put these scripts in `rtdetrv2_pytorch/tools/` and run them "
        "from the repository root."
    ) from exc


def build_parser(
    score_mode: str,
    description: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument(
        "--dataloader",
        default="val_dataloader",
        help="YAMLConfig dataloader attribute.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--checkpoint-key",
        choices=("auto", "ema", "model", "module", "state_dict"),
        default="auto",
    )
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--failure-iou-thresholds",
        type=float,
        nargs="+",
        default=(0.5, 0.75),
    )
    parser.add_argument(
        "--allow-wrong-class",
        action="store_true",
        help="Include matched predictions whose predicted class is incorrect.",
    )
    parser.add_argument(
        "--target-box-format",
        choices=(
            "auto",
            "cxcywh_normalized",
            "xyxy_normalized",
            "xyxy_absolute",
        ),
        default="auto",
    )
    parser.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=1000,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional smoke-test limit.",
    )
    parser.add_argument(
        "--print-freq",
        type=int,
        default=25,
    )
    if score_mode == "learned":
        parser.add_argument(
            "--tu-prototypes",
            default=None,
            help=(
                "Optional prototype artifact. Required when it is not already "
                "configured through `tu_prototype_path`."
            ),
        )
        parser.add_argument(
            "--tu-topk",
            type=int,
            default=None,
            help="Override inference TU top-k.",
        )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("--confidence-threshold must be in [0, 1].")
    if not args.failure_iou_thresholds:
        raise ValueError("At least one failure IoU threshold is required.")
    if any(not 0.0 < value < 1.0 for value in args.failure_iou_thresholds):
        raise ValueError("Failure IoU thresholds must lie in (0, 1).")
    if args.bootstrap_repeats < 0:
        raise ValueError("--bootstrap-repeats must be non-negative.")
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("--max-batches must be positive.")
    if args.print_freq < 1:
        raise ValueError("--print-freq must be positive.")


def disable_pretrained_downloads(node: Any) -> None:
    if isinstance(node, MutableMapping):
        if "pretrained" in node:
            node["pretrained"] = False
        for value in node.values():
            disable_pretrained_downloads(value)
    elif isinstance(node, list):
        for value in node:
            disable_pretrained_downloads(value)


def load_torch_file(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def is_state_dict(candidate: Any) -> bool:
    return (
        isinstance(candidate, Mapping)
        and bool(candidate)
        and all(isinstance(key, str) for key in candidate)
        and all(isinstance(value, Tensor) for value in candidate.values())
    )


def select_state_dict(
    checkpoint: Any,
    checkpoint_key: str,
) -> Tuple[Mapping[str, Tensor], str]:
    if is_state_dict(checkpoint):
        if checkpoint_key not in ("auto", "state_dict"):
            raise KeyError("Raw state dict has no nested checkpoint key.")
        return checkpoint, "<root>"
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping.")

    def extract(key: str) -> Optional[Mapping[str, Tensor]]:
        value = checkpoint.get(key)
        if key == "ema" and isinstance(value, Mapping):
            nested = value.get("module")
            if is_state_dict(nested):
                return nested
        return value if is_state_dict(value) else None

    if checkpoint_key != "auto":
        selected = extract(checkpoint_key)
        if selected is None:
            raise KeyError(
                f"No state dict at `{checkpoint_key}`; "
                f"available keys are {list(checkpoint)}."
            )
        return selected, "ema.module" if checkpoint_key == "ema" else checkpoint_key

    for key in ("ema", "model", "module", "state_dict"):
        selected = extract(key)
        if selected is not None:
            return selected, "ema.module" if key == "ema" else key
    raise KeyError(f"Could not find model weights in keys {list(checkpoint)}.")


def strip_module_prefix_if_needed(
    state_dict: Mapping[str, Tensor],
    model: nn.Module,
) -> Mapping[str, Tensor]:
    keys = list(state_dict)
    if not keys or not all(key.startswith("module.") for key in keys):
        return state_dict
    model_keys = set(model.state_dict())
    direct_overlap = sum(key in model_keys for key in keys)
    stripped = {
        key[len("module.") :]: value
        for key, value in state_dict.items()
    }
    stripped_overlap = sum(key in model_keys for key in stripped)
    return stripped if stripped_overlap > direct_overlap else state_dict


def load_model_weights(
    model: nn.Module,
    checkpoint_path: Path,
    checkpoint_key: str,
    strict: bool,
) -> str:
    checkpoint = load_torch_file(checkpoint_path)
    state_dict, selected_name = select_state_dict(
        checkpoint,
        checkpoint_key,
    )
    state_dict = strip_module_prefix_if_needed(state_dict, model)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        if incompatible.missing_keys:
            print(
                f"[checkpoint] missing keys ({len(incompatible.missing_keys)}): "
                f"{incompatible.missing_keys[:10]}"
            )
        if incompatible.unexpected_keys:
            print(
                "[checkpoint] unexpected keys "
                f"({len(incompatible.unexpected_keys)}): "
                f"{incompatible.unexpected_keys[:10]}"
            )
    print(f"[checkpoint] loaded weights from `{selected_name}`")
    return selected_name


def find_uncertainty_module(model: nn.Module) -> Tuple[str, nn.Module]:
    candidates = []
    for name, module in model.named_modules():
        if (
            hasattr(module, "bbox_uncertainty_head")
            and hasattr(module, "dec_bbox_head")
        ):
            candidates.append((name, module))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise RuntimeError("Could not find the bbox uncertainty module.")
    raise RuntimeError(
        f"Multiple bbox uncertainty modules found: {[name for name, _ in candidates]}"
    )


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    center_x, center_y, width, height = boxes.unbind(dim=-1)
    return torch.stack(
        (
            center_x - 0.5 * width,
            center_y - 0.5 * height,
            center_x + 0.5 * width,
            center_y + 0.5 * height,
        ),
        dim=-1,
    )


def box_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    return torch.stack(
        (
            0.5 * (x1 + x2),
            0.5 * (y1 + y2),
            x2 - x1,
            y2 - y1,
        ),
        dim=-1,
    )


def target_image_size(target: Mapping[str, Any]) -> Tuple[float, float]:
    size = target.get("orig_size", target.get("size"))
    if size is None:
        raise KeyError(
            "Absolute target boxes require `orig_size` or `size`."
        )
    size_tensor = torch.as_tensor(size).detach().flatten()
    if size_tensor.numel() != 2:
        raise ValueError("Target image size must contain [height, width].")
    height, width = size_tensor.tolist()
    return float(height), float(width)


def infer_target_box_format(
    boxes: Tensor,
    requested_format: str,
) -> str:
    if requested_format != "auto":
        return requested_format
    if boxes.numel() == 0:
        return "cxcywh_normalized"
    if float(boxes.detach().abs().max().cpu()) > 1.5:
        return "xyxy_absolute"
    # Normal RT-DETR training targets are normalized cxcywh.
    return "cxcywh_normalized"


def normalize_target_boxes(
    target: Mapping[str, Any],
    requested_format: str,
) -> Tuple[Tensor, Tensor, str]:
    boxes = target["boxes"].float()
    detected_format = infer_target_box_format(boxes, requested_format)

    if detected_format == "cxcywh_normalized":
        boxes_cxcywh = boxes
        boxes_xyxy = box_cxcywh_to_xyxy(boxes)
    elif detected_format == "xyxy_normalized":
        boxes_xyxy = boxes
        boxes_cxcywh = box_xyxy_to_cxcywh(boxes)
    elif detected_format == "xyxy_absolute":
        height, width = target_image_size(target)
        scale = boxes.new_tensor((width, height, width, height))
        boxes_xyxy = boxes / scale
        boxes_cxcywh = box_xyxy_to_cxcywh(boxes_xyxy)
    else:
        raise ValueError(f"Unsupported target box format: {detected_format}")

    return boxes_cxcywh, boxes_xyxy, detected_format


def aligned_iou(left_boxes: Tensor, right_boxes: Tensor) -> Tensor:
    top_left = torch.maximum(left_boxes[:, :2], right_boxes[:, :2])
    bottom_right = torch.minimum(left_boxes[:, 2:], right_boxes[:, 2:])
    intersection_wh = (bottom_right - top_left).clamp_min(0)
    intersection = intersection_wh[:, 0] * intersection_wh[:, 1]

    left_wh = (left_boxes[:, 2:] - left_boxes[:, :2]).clamp_min(0)
    right_wh = (right_boxes[:, 2:] - right_boxes[:, :2]).clamp_min(0)
    left_area = left_wh[:, 0] * left_wh[:, 1]
    right_area = right_wh[:, 0] * right_wh[:, 1]
    union = left_area + right_area - intersection
    return intersection / union.clamp_min(torch.finfo(union.dtype).eps)


def move_to_device(value: Any, device: torch.device) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    return value


def matcher_indices(
    matcher: nn.Module,
    outputs: Mapping[str, Tensor],
    targets: Sequence[Mapping[str, Any]],
) -> Sequence[Tuple[Tensor, Tensor]]:
    result = matcher(
        {
            "pred_logits": outputs["pred_logits"],
            "pred_boxes": outputs["pred_boxes"],
        },
        targets,
    )
    if isinstance(result, Mapping):
        result = result["indices"]
    return result


def image_identifier(
    target: Mapping[str, Any],
    fallback: int,
) -> int:
    image_id = target.get("image_id")
    if image_id is None:
        return fallback
    image_id_tensor = torch.as_tensor(image_id).detach().flatten().cpu()
    if image_id_tensor.numel() != 1:
        return fallback
    return int(image_id_tensor.item())


@torch.inference_mode()
def collect_records(
    model: nn.Module,
    matcher: nn.Module,
    dataloader: Any,
    device: torch.device,
    args: argparse.Namespace,
    score_mode: str,
) -> Dict[str, np.ndarray]:
    records: Dict[str, List[np.ndarray]] = {
        "image_id": [],
        "iou": [],
        "confidence": [],
        "default_uncertainty": [],
    }
    if score_mode == "learned":
        records["learned_uncertainty"] = []

    format_counts: Dict[str, int] = {}
    image_counter = 0
    total_matches = 0
    accepted_before_uncertainty = 0

    for batch_index, (samples, raw_targets) in enumerate(dataloader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break

        samples = move_to_device(samples, device)
        targets = [
            {
                key: move_to_device(value, device)
                for key, value in target.items()
            }
            for target in raw_targets
        ]

        outputs = model(samples)
        if not isinstance(outputs, Mapping):
            raise TypeError("Model output must be a mapping.")
        required = ("pred_logits", "pred_boxes")
        missing = [key for key in required if key not in outputs]
        if missing:
            raise KeyError(f"Model output is missing {missing}.")
        if score_mode == "learned" and "pred_bbox_std" not in outputs:
            raise KeyError(
                "The learned model did not return `pred_bbox_std`. Verify "
                "bbox_uncertainty_enabled and checkpoint loading."
            )

        matching_targets = []
        target_xyxy_by_image = []
        for target in targets:
            target_cxcywh, target_xyxy, detected_format = (
                normalize_target_boxes(
                    target,
                    args.target_box_format,
                )
            )
            format_counts[detected_format] = (
                format_counts.get(detected_format, 0) + 1
            )
            matching_target = dict(target)
            matching_target["boxes"] = target_cxcywh
            matching_targets.append(matching_target)
            target_xyxy_by_image.append(target_xyxy)

        indices = matcher_indices(
            matcher,
            outputs,
            matching_targets,
        )

        logits = outputs["pred_logits"]
        predicted_boxes = outputs["pred_boxes"]
        probabilities = logits.sigmoid()
        query_confidence, query_class = probabilities.max(dim=-1)

        for local_image_index, (source_index, target_index) in enumerate(indices):
            source_index = source_index.to(device=device, dtype=torch.long)
            target_index = target_index.to(device=device, dtype=torch.long)
            total_matches += int(source_index.numel())
            if source_index.numel() == 0:
                image_counter += 1
                continue

            confidence = query_confidence[local_image_index, source_index]
            predicted_class = query_class[local_image_index, source_index]
            target_class = matching_targets[local_image_index]["labels"][
                target_index
            ]
            keep = confidence >= args.confidence_threshold
            if not args.allow_wrong_class:
                keep = keep & predicted_class.eq(target_class)

            accepted_before_uncertainty += int(keep.sum().item())
            if score_mode == "learned":
                predicted_std = outputs["pred_bbox_std"][
                    local_image_index,
                    source_index,
                ]
                learned_uncertainty = torch.linalg.vector_norm(
                    predicted_std,
                    ord=2,
                    dim=-1,
                )
                keep = keep & torch.isfinite(learned_uncertainty)
            else:
                learned_uncertainty = None

            if not keep.any():
                image_counter += 1
                continue

            source_index = source_index[keep]
            target_index = target_index[keep]
            confidence = confidence[keep]
            if learned_uncertainty is not None:
                learned_uncertainty = learned_uncertainty[keep]

            predicted_xyxy = box_cxcywh_to_xyxy(
                predicted_boxes[local_image_index, source_index]
            )
            target_xyxy = target_xyxy_by_image[local_image_index][
                target_index
            ]
            iou = aligned_iou(predicted_xyxy, target_xyxy)

            image_id = image_identifier(
                matching_targets[local_image_index],
                image_counter,
            )
            count = iou.numel()
            records["image_id"].append(
                np.full(count, image_id, dtype=np.int64)
            )
            records["iou"].append(iou.float().cpu().numpy())
            records["confidence"].append(
                confidence.float().cpu().numpy()
            )
            records["default_uncertainty"].append(
                (1.0 - confidence).float().cpu().numpy()
            )
            if learned_uncertainty is not None:
                records["learned_uncertainty"].append(
                    learned_uncertainty.float().cpu().numpy()
                )

            image_counter += 1

        if (batch_index + 1) % args.print_freq == 0:
            accepted = sum(chunk.size for chunk in records["iou"])
            print(
                f"[progress] batches={batch_index + 1}, "
                f"images={image_counter}, matches={total_matches}, "
                f"accepted={accepted}"
            )

    concatenated = {}
    for key, chunks in records.items():
        if not chunks:
            raise RuntimeError(f"No records collected for `{key}`.")
        concatenated[key] = np.concatenate(chunks)

    print(
        "[data] converted target boxes for matching; "
        f"detected={format_counts}"
    )
    print(
        f"[population] matches={total_matches}, "
        f"accepted_before_uncertainty={accepted_before_uncertainty}, "
        f"evaluated={concatenated['iou'].size}"
    )
    return concatenated


def safe_ranking_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, float]:
    if labels.ndim != 1 or scores.ndim != 1 or labels.size != scores.size:
        raise ValueError("Labels and scores must be aligned one-dimensional arrays.")
    if np.unique(labels).size != 2:
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
        }
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def bootstrap_aurocs(
    labels: np.ndarray,
    score_arrays: Mapping[str, np.ndarray],
    image_ids: np.ndarray,
    repeats: int,
    seed: int,
) -> Dict[str, Any]:
    point = {
        name: safe_ranking_metrics(labels, scores)["auroc"]
        for name, scores in score_arrays.items()
    }
    result: Dict[str, Any] = {
        "point": point,
        "confidence_level": 0.95,
        "resampling_unit": "image",
        "repeats": repeats,
    }
    if repeats == 0:
        return result

    unique_images = np.unique(image_ids)
    grouped_indices = {
        image_id: np.flatnonzero(image_ids == image_id)
        for image_id in unique_images
    }
    rng = np.random.default_rng(seed)
    samples: Dict[str, List[float]] = {
        name: [] for name in score_arrays
    }
    difference_samples: List[float] = []

    for _ in range(repeats):
        sampled_images = rng.choice(
            unique_images,
            size=unique_images.size,
            replace=True,
        )
        selected = np.concatenate(
            [grouped_indices[image_id] for image_id in sampled_images]
        )
        sampled_labels = labels[selected]
        if np.unique(sampled_labels).size != 2:
            continue

        replicate = {}
        for name, scores in score_arrays.items():
            value = roc_auc_score(sampled_labels, scores[selected])
            samples[name].append(float(value))
            replicate[name] = float(value)
        if "learned_total_std" in replicate:
            difference_samples.append(
                replicate["learned_total_std"]
                - replicate["default_inverse_confidence"]
            )

    result["confidence_intervals"] = {
        name: {
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
            "valid_repeats": len(values),
        }
        for name, values in samples.items()
        if values
    }
    if "learned_total_std" in point:
        point_difference = (
            point["learned_total_std"]
            - point["default_inverse_confidence"]
        )
        result["learned_minus_default"] = {
            "point": float(point_difference),
            "lower": (
                float(np.quantile(difference_samples, 0.025))
                if difference_samples
                else float("nan")
            ),
            "upper": (
                float(np.quantile(difference_samples, 0.975))
                if difference_samples
                else float("nan")
            ),
            "valid_repeats": len(difference_samples),
        }
    return result


def evaluate_records(
    records: Mapping[str, np.ndarray],
    thresholds: Sequence[float],
    repeats: int,
    seed: int,
) -> Dict[str, Any]:
    score_arrays = {
        "default_inverse_confidence": records["default_uncertainty"],
    }
    if "learned_uncertainty" in records:
        score_arrays["learned_total_std"] = records["learned_uncertainty"]

    result = {}
    for threshold_index, threshold in enumerate(thresholds):
        labels = records["iou"] < threshold
        ranking = {
            name: safe_ranking_metrics(labels, scores)
            for name, scores in score_arrays.items()
        }
        result[f"matched_iou_below_{threshold:g}"] = {
            "count": int(labels.size),
            "positive_count": int(labels.sum()),
            "negative_count": int((~labels).sum()),
            "prevalence": float(labels.mean()),
            "ranking": ranking,
            "image_bootstrap": bootstrap_aurocs(
                labels,
                score_arrays,
                records["image_id"],
                repeats,
                seed + threshold_index,
            ),
        }
    return result


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, allow_nan=True)
            file.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(args: argparse.Namespace, score_mode: str) -> None:
    validate_args(args)
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.resume).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    print(f"[config] loading {config_path}")
    cfg = YAMLConfig(str(config_path))
    yaml_cfg = getattr(cfg, "yaml_cfg", None)
    if yaml_cfg is not None:
        disable_pretrained_downloads(yaml_cfg)

    model: nn.Module = cfg.model
    load_model_weights(
        model,
        checkpoint_path,
        args.checkpoint_key,
        args.strict_load,
    )
    model.to(device)
    model.eval()

    uncertainty_module_name = None
    if score_mode == "learned":
        uncertainty_module_name, uncertainty_module = (
            find_uncertainty_module(model)
        )
        if uncertainty_module.bbox_uncertainty_head is None:
            raise RuntimeError(
                "The configured model does not contain an enabled uncertainty head."
            )
        if args.tu_topk is not None:
            uncertainty_module.bbox_tu_topk = int(args.tu_topk)
        if args.tu_prototypes is not None:
            prototype_path = Path(args.tu_prototypes).expanduser().resolve()
            if not prototype_path.is_file():
                raise FileNotFoundError(
                    f"TU prototypes not found: {prototype_path}"
                )
            uncertainty_module.load_tu_prototypes(str(prototype_path))
        if (
            uncertainty_module.bbox_uncertainty_use_tu
            and not uncertainty_module.bbox_tu_loaded
        ):
            raise RuntimeError(
                "The learned uncertainty head requires TU prototypes. Pass "
                "--tu-prototypes or configure tu_prototype_path."
            )

    criterion = cfg.criterion
    matcher = criterion.matcher
    dataloader = getattr(cfg, args.dataloader)

    records = collect_records(
        model,
        matcher,
        dataloader,
        device,
        args,
        score_mode,
    )
    metrics = evaluate_records(
        records,
        args.failure_iou_thresholds,
        args.bootstrap_repeats,
        args.seed,
    )

    output = {
        "metadata": {
            "score_mode": score_mode,
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_key": args.checkpoint_key,
            "dataloader": args.dataloader,
            "confidence_threshold": args.confidence_threshold,
            "require_correct_class": not args.allow_wrong_class,
            "target_box_format": args.target_box_format,
            "failure_iou_thresholds": list(args.failure_iou_thresholds),
            "bootstrap_repeats": args.bootstrap_repeats,
            "seed": args.seed,
            "uncertainty_module": uncertainty_module_name,
            "learned_score": (
                "l2_norm_of_normalized_xyxy_standard_deviations"
                if score_mode == "learned"
                else None
            ),
            "default_score": "one_minus_max_sigmoid_class_confidence",
        },
        "population": {
            "count": int(records["iou"].size),
            "image_count": int(np.unique(records["image_id"]).size),
        },
        "localization_failure_ranking": metrics,
    }
    write_json(output_path, output)

    for failure_name, result in metrics.items():
        print(
            f"[{failure_name}] positives={result['positive_count']}/"
            f"{result['count']} ({result['prevalence']:.3%})"
        )
        for score_name, ranking in result["ranking"].items():
            print(
                f"  {score_name}: AUROC={ranking['auroc']:.4f}, "
                f"AUPRC={ranking['auprc']:.4f}"
            )
        difference = result["image_bootstrap"].get(
            "learned_minus_default"
        )
        if difference is not None:
            print(
                "  learned-default AUROC="
                f"{difference['point']:+.4f} "
                f"[{difference['lower']:+.4f}, "
                f"{difference['upper']:+.4f}]"
            )

    print(f"[output] saved metrics to {output_path}")

