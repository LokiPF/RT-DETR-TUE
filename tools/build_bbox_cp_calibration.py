#!/usr/bin/env python3
"""
Build a conformal calibration artifact for RT-DETR bbox prediction regions.

This script assumes that the RT-DETR decoder can:

  1. load the Fréchet-mean persistence diagrams created by
     ``build_tu_prototypes.py`` through ``load_tu_prototypes(path)``; and
  2. return a ``bbox_tu`` tensor with shape ``[batch, queries]`` during
     evaluation.

The calibration procedure is deliberately separate from prototype building:

  training/prototype data
      -> fit the detector and bbox persistence-diagram Fréchet means

  held-out calibration data
      -> generate matched bbox predictions, TU scores, and localization errors
      -> split images into a scale-fit subset and a conformal subset
      -> fit a monotone binned mapping TU -> per-coordinate error scale
      -> calculate the finite-sample conformal quantile qhat

The output contains:

  * a TU-scaled conformal calibrator;
  * an unscaled conformal baseline;
  * calibration diagnostics and provenance metadata.

The resulting regions have marginal coverage for the population calibrated by
this script: Hungarian-matched detections satisfying the configured confidence
and class-correctness filters. They do not provide coverage for missed objects
or unmatched false-positive detections.

Run from the ``rtdetrv2_pytorch`` repository root:

    python tools/build_bbox_cp_calibration.py \
        -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml \
        -r pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \
        --tu-prototypes output/tu/bbox_prototypes.pth \
        -o output/tu/bbox_cp_calibration.pth \
        --split val \
        --alpha 0.05 \
        --tu-topk 50

Do not evaluate final coverage on the same images used by this script. For a
paper-quality experiment, point the selected dataloader at a dedicated
calibration split and reserve a separate test split.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.core import YAMLConfig
except ImportError as exc:
    raise ImportError(
        "Could not import `src.core.YAMLConfig`. Put this script in "
        "`rtdetrv2_pytorch/tools/` and run it from the repository root."
    ) from exc


COORDINATE_ORDER = ("x1", "y1", "x2", "y2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build TU-scaled conformal bbox calibration for RT-DETR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--config", required=True, help="RT-DETR YAML config.")
    parser.add_argument("-r", "--resume", required=True, help="Detector checkpoint.")
    parser.add_argument(
        "--tu-prototypes",
        required=True,
        help="Fréchet-mean persistence diagrams created by build_tu_prototypes.py.",
    )
    parser.add_argument("-o", "--output", required=True, help="Output .pth artifact.")
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="val",
        help="Configured dataloader that represents the calibration split.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )
    parser.add_argument(
        "--checkpoint-key",
        choices=("auto", "ema", "model", "module", "state_dict"),
        default="auto",
        help="Checkpoint entry containing the detector state dict.",
    )
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="Require the checkpoint to match the detector exactly.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Miscoverage level; 0.05 targets 95%% marginal coverage.",
    )
    parser.add_argument(
        "--fit-fraction",
        type=float,
        default=0.50,
        help=(
            "Fraction of accepted calibration images used to fit the TU scale. "
            "Remaining images calculate qhat."
        ),
    )
    parser.add_argument(
        "--num-tu-bins",
        type=int,
        default=10,
        help="Requested number of quantile bins for the TU scale mapping.",
    )
    parser.add_argument(
        "--scale-quantile",
        type=float,
        default=0.50,
        help="Per-bin absolute-error quantile used as coordinate scale.",
    )
    parser.add_argument(
        "--scale-floor",
        type=float,
        default=1e-4,
        help="Minimum normalized-coordinate scale.",
    )
    parser.add_argument(
        "--tu-topk",
        type=int,
        default=50,
        help="Number of highest-confidence queries for which the model computes TU.",
    )
    parser.add_argument(
        "--tu-min-samples",
        type=int,
        default=20,
        help="Minimum count required for a class TU prototype.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Only calibrate matched queries at or above this sigmoid confidence.",
    )
    parser.add_argument(
        "--allow-wrong-class",
        action="store_true",
        help="Include matched predictions whose predicted class is incorrect.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional maximum number of calibration images.",
    )
    parser.add_argument(
        "--min-detections",
        type=int,
        default=200,
        help="Minimum accepted detections required in each internal subset.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N dataloader batches; 0 disables it.",
    )
    parser.add_argument(
        "--hash-checkpoint",
        action="store_true",
        help="Store the detector checkpoint SHA-256 hash; can be slow.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be strictly between 0 and 1.")
    if not 0.0 < args.fit_fraction < 1.0:
        raise ValueError("--fit-fraction must be strictly between 0 and 1.")
    if args.num_tu_bins < 1:
        raise ValueError("--num-tu-bins must be at least 1.")
    if not 0.0 < args.scale_quantile <= 1.0:
        raise ValueError("--scale-quantile must be in (0, 1].")
    if args.scale_floor <= 0:
        raise ValueError("--scale-floor must be positive.")
    if args.tu_topk < 1:
        raise ValueError("--tu-topk must be positive.")
    if args.tu_min_samples < 1:
        raise ValueError("--tu-min-samples must be positive.")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("--confidence-threshold must be in [0, 1].")
    if args.max_images is not None and args.max_images < 1:
        raise ValueError("--max-images must be positive.")
    if args.min_detections < 20:
        raise ValueError("--min-detections must be at least 20.")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def disable_pretrained_downloads(node: Any) -> None:
    if isinstance(node, MutableMapping):
        if "pretrained" in node:
            node["pretrained"] = False
        for value in node.values():
            disable_pretrained_downloads(value)
    elif isinstance(node, list):
        for value in node:
            disable_pretrained_downloads(value)


def load_torch_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def is_state_dict(candidate: Any) -> bool:
    return (
        isinstance(candidate, Mapping)
        and len(candidate) > 0
        and all(isinstance(key, str) for key in candidate)
        and all(isinstance(value, Tensor) for value in candidate.values())
    )


def select_state_dict(
    checkpoint: Any,
    checkpoint_key: str,
) -> Tuple[Mapping[str, Tensor], str]:
    if is_state_dict(checkpoint):
        if checkpoint_key not in ("auto", "state_dict"):
            raise KeyError(
                "Checkpoint is a raw state dict; a nested checkpoint key "
                f"`{checkpoint_key}` cannot be selected."
            )
        return checkpoint, "<root>"

    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint is not a mapping.")

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
                f"No valid state dict at `{checkpoint_key}`. "
                f"Available keys: {list(checkpoint.keys())}"
            )
        return selected, "ema.module" if checkpoint_key == "ema" else checkpoint_key

    for key in ("ema", "model", "module", "state_dict"):
        selected = extract(key)
        if selected is not None:
            return selected, "ema.module" if key == "ema" else key

    raise KeyError(
        "Could not auto-detect detector weights. "
        f"Available keys: {list(checkpoint.keys())}"
    )


def strip_module_prefix_if_needed(
    state_dict: Mapping[str, Tensor],
    model: nn.Module,
) -> Mapping[str, Tensor]:
    keys = list(state_dict)
    if not keys or not all(key.startswith("module.") for key in keys):
        return state_dict

    model_keys = set(model.state_dict())
    direct_overlap = sum(key in model_keys for key in keys)
    stripped = {key[len("module.") :]: value for key, value in state_dict.items()}
    stripped_overlap = sum(key in model_keys for key in stripped)
    return stripped if stripped_overlap > direct_overlap else state_dict


def load_model_weights(
    model: nn.Module,
    checkpoint_path: Path,
    checkpoint_key: str,
    strict: bool,
) -> str:
    checkpoint = load_torch_checkpoint(checkpoint_path)
    state_dict, selected_name = select_state_dict(checkpoint, checkpoint_key)
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
                f"[checkpoint] unexpected keys "
                f"({len(incompatible.unexpected_keys)}): "
                f"{incompatible.unexpected_keys[:10]}"
            )

    print(f"[checkpoint] loaded weights from `{selected_name}`")
    return selected_name


def find_tu_module(model: nn.Module) -> Tuple[str, nn.Module]:
    candidates: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if callable(getattr(module, "load_tu_prototypes", None)) and hasattr(
            module, "dec_bbox_head"
        ):
            candidates.append((name, module))

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise RuntimeError(
            "Could not find the RTDETRTransformerv2TUE module. It must expose "
            "`load_tu_prototypes()` and `dec_bbox_head`."
        )
    raise RuntimeError(
        "Found multiple TU modules; expected one. Candidates: "
        f"{[name or '<root>' for name, _ in candidates]}"
    )


def move_targets_to_device(
    targets: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> List[Dict[str, Any]]:
    return [
        {
            key: value.to(device, non_blocking=True)
            if isinstance(value, Tensor)
            else value
            for key, value in target.items()
        }
        for target in targets
    ]


def unpack_batch(batch: Any) -> Tuple[Tensor, Sequence[Mapping[str, Any]]]:
    if not isinstance(batch, (list, tuple)) or len(batch) < 2:
        raise TypeError("Expected dataloader batch `(samples, targets, ...)`.")
    samples, targets = batch[0], batch[1]
    if not isinstance(samples, Tensor):
        raise TypeError(f"Samples are {type(samples).__name__}, not a Tensor.")
    if not isinstance(targets, (list, tuple)):
        raise TypeError(f"Targets are {type(targets).__name__}, not a sequence.")
    return samples, targets


def unwrap_match_indices(match_result: Any) -> Sequence[Tuple[Tensor, Tensor]]:
    if isinstance(match_result, Mapping):
        if "indices" not in match_result:
            raise KeyError(
                "Matcher returned a mapping without `indices`. "
                f"Available keys: {list(match_result.keys())}"
            )
        return match_result["indices"]
    return match_result


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            cx - 0.5 * width,
            cy - 0.5 * height,
            cx + 0.5 * width,
            cy + 0.5 * height,
        ),
        dim=-1,
    )


def aligned_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    boxes1 = box_cxcywh_to_xyxy(boxes1)
    boxes2 = box_cxcywh_to_xyxy(boxes2)
    top_left = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    bottom_right = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0).prod(dim=-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0).prod(dim=-1)
    union = area1 + area2 - intersection
    return intersection / union.clamp_min(torch.finfo(union.dtype).eps)


def get_image_id(target: Mapping[str, Any], fallback: int) -> int:
    image_id = target.get("image_id")
    if isinstance(image_id, Tensor) and image_id.numel() > 0:
        return int(image_id.detach().reshape(-1)[0].cpu().item())
    if isinstance(image_id, (int, float)):
        return int(image_id)
    return fallback


def split_by_image(
    image_ids: Tensor,
    fit_fraction: float,
    seed: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    unique_images = torch.unique(image_ids.cpu(), sorted=True)
    if unique_images.numel() < 2:
        raise RuntimeError("At least two accepted calibration images are required.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(unique_images.numel(), generator=generator)

    fit_image_count = int(round(unique_images.numel() * fit_fraction))
    fit_image_count = max(1, min(fit_image_count, unique_images.numel() - 1))
    fit_images = unique_images[permutation[:fit_image_count]]
    conformal_images = unique_images[permutation[fit_image_count:]]

    fit_set = set(int(value) for value in fit_images.tolist())
    fit_mask = torch.tensor(
        [int(value) in fit_set for value in image_ids.cpu().tolist()],
        dtype=torch.bool,
    )
    conformal_mask = ~fit_mask
    return fit_mask, conformal_mask, fit_images, conformal_images


def fit_monotone_binned_scale(
    tu: Tensor,
    absolute_errors: Tensor,
    requested_bins: int,
    scale_quantile: float,
    scale_floor: float,
) -> Dict[str, Tensor]:
    tu = tu.detach().float().cpu()
    absolute_errors = absolute_errors.detach().float().cpu()
    if tu.ndim != 1 or absolute_errors.shape != (tu.numel(), 4):
        raise ValueError("Expected TU [N] and absolute bbox errors [N, 4].")

    if requested_bins == 1:
        edges = torch.empty(0, dtype=torch.float32)
    else:
        quantile_levels = torch.linspace(0, 1, requested_bins + 1)[1:-1]
        edges = torch.quantile(tu, quantile_levels)
        edges = torch.unique_consecutive(edges)

    bin_ids = torch.bucketize(tu, edges)
    actual_bins = int(edges.numel() + 1)
    global_scale = torch.quantile(
        absolute_errors,
        scale_quantile,
        dim=0,
    ).clamp_min(scale_floor)

    raw_scales: List[Tensor] = []
    counts: List[int] = []
    for bin_index in range(actual_bins):
        mask = bin_ids == bin_index
        count = int(mask.sum())
        counts.append(count)
        if count == 0:
            scale = raw_scales[-1] if raw_scales else global_scale
        else:
            scale = torch.quantile(
                absolute_errors[mask],
                scale_quantile,
                dim=0,
            ).clamp_min(scale_floor)
        raw_scales.append(scale)

    raw_scale_tensor = torch.stack(raw_scales)
    monotone_scales = torch.cummax(raw_scale_tensor, dim=0).values
    return {
        "tu_bin_edges": edges.to(torch.float32),
        "coordinate_scales": monotone_scales.to(torch.float32),
        "raw_coordinate_scales": raw_scale_tensor.to(torch.float32),
        "bin_counts": torch.tensor(counts, dtype=torch.int64),
    }


def lookup_coordinate_scales(
    tu: Tensor,
    scale_model: Mapping[str, Tensor],
) -> Tensor:
    tu_cpu = tu.detach().float().cpu()
    edges = scale_model["tu_bin_edges"].detach().float().cpu()
    scales = scale_model["coordinate_scales"].detach().float().cpu()
    bin_ids = torch.bucketize(tu_cpu, edges)
    return scales[bin_ids]


def conformal_quantile(scores: Tensor, alpha: float) -> Tuple[Tensor, int, float]:
    scores = scores.detach().float().flatten().cpu()
    if not torch.isfinite(scores).all():
        raise ValueError("Conformal scores contain NaN or infinity.")
    n = scores.numel()
    if n == 0:
        raise ValueError("Cannot calibrate an empty score set.")

    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        raise ValueError(
            f"{n} calibration detections are insufficient for alpha={alpha}. "
            "The finite-sample conformal quantile would be infinite."
        )

    sorted_scores = torch.sort(scores).values
    qhat = sorted_scores[rank - 1]
    nominal_level = rank / n
    return qhat, rank, nominal_level


def approximate_spearman(left: Tensor, right: Tensor) -> float:
    left = left.detach().float().flatten().cpu()
    right = right.detach().float().flatten().cpu()
    if left.numel() < 2 or float(left.std()) == 0 or float(right.std()) == 0:
        return float("nan")
    left_rank = torch.argsort(torch.argsort(left)).float()
    right_rank = torch.argsort(torch.argsort(right)).float()
    left_rank = (left_rank - left_rank.mean()) / left_rank.std()
    right_rank = (right_rank - right_rank.mean()) / right_rank.std()
    return float((left_rank * right_rank).mean())


def evaluate_regions(absolute_errors: Tensor, half_widths: Tensor) -> Dict[str, Any]:
    covered_per_coordinate = absolute_errors <= half_widths
    jointly_covered = covered_per_coordinate.all(dim=-1)
    return {
        "joint_coverage": float(jointly_covered.float().mean()),
        "coordinate_coverage": covered_per_coordinate.float().mean(dim=0),
        "mean_half_width": half_widths.mean(dim=0),
        "median_half_width": half_widths.median(dim=0).values,
        "mean_coordinate_width": float((2.0 * half_widths).mean()),
    }


def apply_cp_calibrator(
    predicted_xyxy: Tensor,
    bbox_tu: Tensor,
    artifact: Mapping[str, Any],
    clip: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Apply a saved TU-scaled calibrator.

    Returns:
        coordinate_lower: [..., 4]
        coordinate_upper: [..., 4]
    """
    original_shape = bbox_tu.shape
    flat_tu = bbox_tu.detach().reshape(-1).float().cpu()
    scale_model = artifact["scale_model"]
    coordinate_scales = lookup_coordinate_scales(flat_tu, scale_model)
    qhat = artifact["conformal"]["scaled"]["qhat"].detach().float().cpu()
    half_widths = (qhat * coordinate_scales).reshape(*original_shape, 4)
    half_widths = half_widths.to(
        device=predicted_xyxy.device,
        dtype=predicted_xyxy.dtype,
    )

    lower = predicted_xyxy - half_widths
    upper = predicted_xyxy + half_widths
    if clip:
        lower = lower.clamp(0.0, 1.0)
        upper = upper.clamp(0.0, 1.0)
    return lower, upper


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def run(args: argparse.Namespace) -> Dict[str, Any]:
    validate_args(args)
    seed_everything(args.seed)

    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.resume).expanduser().resolve()
    prototypes_path = Path(args.tu_prototypes).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    for label, path in (
        ("Config", config_path),
        ("Checkpoint", checkpoint_path),
        ("TU prototypes", prototypes_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    print(f"[config] loading {config_path}")
    cfg = YAMLConfig(str(config_path))
    yaml_cfg = getattr(cfg, "yaml_cfg", None)
    if yaml_cfg is not None:
        disable_pretrained_downloads(yaml_cfg)

    model: nn.Module = cfg.model
    criterion: nn.Module = cfg.criterion
    matcher = getattr(criterion, "matcher", None)
    if matcher is None:
        raise AttributeError("Configured criterion does not expose `matcher`.")

    selected_checkpoint_key = load_model_weights(
        model,
        checkpoint_path,
        args.checkpoint_key,
        args.strict_load,
    )
    model.to(device)
    criterion.to(device)

    tu_module_name, tu_module = find_tu_module(model)
    if hasattr(tu_module, "tu_min_samples"):
        tu_module.tu_min_samples = args.tu_min_samples
    if hasattr(tu_module, "bbox_tu_topk"):
        tu_module.bbox_tu_topk = args.tu_topk
    tu_module.load_tu_prototypes(str(prototypes_path))
    tu_module.tu_enabled = True

    model.eval()
    criterion.eval()
    print(
        f"[tu] loaded prototypes into `{tu_module_name or '<root>'}`; "
        f"computing TU for top {args.tu_topk} queries"
    )

    if args.split == "train":
        dataloader = cfg.train_dataloader
        print(
            "[data] using train_dataloader. A dedicated deterministic "
            "calibration dataloader is strongly recommended."
        )
    else:
        dataloader = cfg.val_dataloader
        print("[data] using val_dataloader as the calibration source.")

    collected_tu: List[Tensor] = []
    collected_errors: List[Tensor] = []
    collected_classes: List[Tensor] = []
    collected_confidences: List[Tensor] = []
    collected_ious: List[Tensor] = []
    collected_image_ids: List[Tensor] = []

    images_seen = 0
    matches_seen = 0
    accepted = 0
    rejected_nonfinite_tu = 0
    rejected_low_confidence = 0
    rejected_wrong_class = 0

    try:
        total_batches = len(dataloader)
    except TypeError:
        total_batches = None
    progress = tqdm(
        dataloader,
        total=total_batches,
        desc=f"[{args.split}] calibration",
        unit="batch",
    )

    with torch.inference_mode():
        for batch_index, batch in enumerate(progress):
            if args.max_images is not None and images_seen >= args.max_images:
                break

            samples, targets_cpu = unpack_batch(batch)
            remaining = (
                None
                if args.max_images is None
                else args.max_images - images_seen
            )
            if remaining is not None and samples.shape[0] > remaining:
                samples = samples[:remaining]
                targets_cpu = targets_cpu[:remaining]

            batch_size = int(samples.shape[0])
            fallback_image_ids = [
                images_seen + index for index in range(batch_size)
            ]
            samples = samples.to(device, non_blocking=True)
            targets = move_targets_to_device(targets_cpu, device)

            outputs = model(samples)
            if not isinstance(outputs, Mapping):
                raise TypeError("Model output is not a mapping.")
            required = ("pred_logits", "pred_boxes", "bbox_tu")
            missing = [key for key in required if key not in outputs]
            if missing:
                raise KeyError(
                    f"Model output is missing {missing}. Verify that TU is "
                    "enabled and the corrected TUE forward path is active."
                )

            pred_logits = outputs["pred_logits"]
            pred_boxes = outputs["pred_boxes"]
            bbox_tu = outputs["bbox_tu"]
            if bbox_tu.shape != pred_logits.shape[:2]:
                raise ValueError(
                    f"bbox_tu shape {tuple(bbox_tu.shape)} does not match "
                    f"logit query shape {tuple(pred_logits.shape[:2])}."
                )

            match_result = matcher(
                {"pred_logits": pred_logits, "pred_boxes": pred_boxes},
                targets,
            )
            match_indices = unwrap_match_indices(match_result)
            probabilities = pred_logits.sigmoid()
            query_confidences, query_classes = probabilities.max(dim=-1)

            for image_index, (source_index, target_index) in enumerate(
                match_indices
            ):
                source_index = source_index.to(device=device, dtype=torch.long)
                target_index = target_index.to(device=device, dtype=torch.long)
                count = int(source_index.numel())
                matches_seen += count
                if count == 0:
                    continue

                tu_values = bbox_tu[image_index, source_index].float()
                confidences = query_confidences[image_index, source_index]
                predicted_classes = query_classes[image_index, source_index]
                target_classes = targets[image_index]["labels"][target_index]

                finite_mask = torch.isfinite(tu_values)
                rejected_nonfinite_tu += int((~finite_mask).sum())

                confidence_mask = confidences >= args.confidence_threshold
                rejected_low_confidence += int(
                    (finite_mask & ~confidence_mask).sum()
                )

                if args.allow_wrong_class:
                    class_mask = torch.ones_like(finite_mask)
                else:
                    class_mask = predicted_classes == target_classes
                    rejected_wrong_class += int(
                        (finite_mask & confidence_mask & ~class_mask).sum()
                    )

                keep = finite_mask & confidence_mask & class_mask
                if not bool(keep.any()):
                    continue

                kept_source = source_index[keep]
                kept_target = target_index[keep]
                kept_pred_boxes = pred_boxes[image_index, kept_source]
                kept_target_boxes = targets[image_index]["boxes"][kept_target]
                pred_xyxy = box_cxcywh_to_xyxy(kept_pred_boxes)
                target_xyxy = box_cxcywh_to_xyxy(kept_target_boxes)
                absolute_errors = (target_xyxy - pred_xyxy).abs()
                ious = aligned_box_iou(kept_pred_boxes, kept_target_boxes)

                image_id = get_image_id(
                    targets_cpu[image_index],
                    fallback_image_ids[image_index],
                )
                kept_count = int(keep.sum())
                collected_tu.extend(tu_values[keep].detach().cpu().unbind())
                collected_errors.extend(absolute_errors.detach().cpu().unbind())
                collected_classes.extend(
                    target_classes[keep].detach().cpu().unbind()
                )
                collected_confidences.extend(
                    confidences[keep].detach().cpu().unbind()
                )
                collected_ious.extend(ious.detach().cpu().unbind())
                collected_image_ids.extend(
                    torch.full(
                        (kept_count,),
                        image_id,
                        dtype=torch.int64,
                    ).unbind()
                )
                accepted += kept_count

            images_seen += batch_size
            progress.set_postfix(
                images=images_seen, matches=matches_seen, accepted=accepted
            )
            if (
                args.progress_every > 0
                and (batch_index + 1) % args.progress_every == 0
            ):
                progress.write(
                    f"[progress] batches={batch_index + 1}, "
                    f"images={images_seen}, matches={matches_seen}, "
                    f"accepted={accepted}"
                )

    progress.close()

    if accepted < 2 * args.min_detections:
        raise RuntimeError(
            f"Only {accepted} detections were accepted; at least "
            f"{2 * args.min_detections} are required before the internal split. "
            "Increase --tu-topk, lower --confidence-threshold, or provide more "
            "calibration images."
        )

    tu = torch.stack(collected_tu).float()
    absolute_errors = torch.stack(collected_errors).float()
    class_ids = torch.stack(collected_classes).long()
    confidences = torch.stack(collected_confidences).float()
    ious = torch.stack(collected_ious).float()
    image_ids = torch.stack(collected_image_ids).long()

    fit_mask, conformal_mask, fit_images, conformal_images = split_by_image(
        image_ids,
        args.fit_fraction,
        args.seed,
    )
    fit_count = int(fit_mask.sum())
    conformal_count = int(conformal_mask.sum())
    if fit_count < args.min_detections or conformal_count < args.min_detections:
        raise RuntimeError(
            f"Internal split produced {fit_count} scale-fit and "
            f"{conformal_count} conformal detections; each requires at least "
            f"{args.min_detections}. Adjust --fit-fraction or add data."
        )

    scale_model = fit_monotone_binned_scale(
        tu[fit_mask],
        absolute_errors[fit_mask],
        args.num_tu_bins,
        args.scale_quantile,
        args.scale_floor,
    )
    conformal_scales = lookup_coordinate_scales(
        tu[conformal_mask],
        scale_model,
    )
    scaled_scores = (
        absolute_errors[conformal_mask] / conformal_scales
    ).amax(dim=-1)
    unscaled_scores = absolute_errors[conformal_mask].amax(dim=-1)

    qhat_scaled, scaled_rank, scaled_level = conformal_quantile(
        scaled_scores,
        args.alpha,
    )
    qhat_unscaled, unscaled_rank, unscaled_level = conformal_quantile(
        unscaled_scores,
        args.alpha,
    )

    scaled_half_widths = qhat_scaled * conformal_scales
    unscaled_half_widths = (
        torch.ones_like(absolute_errors[conformal_mask]) * qhat_unscaled
    )
    scaled_metrics = evaluate_regions(
        absolute_errors[conformal_mask],
        scaled_half_widths,
    )
    unscaled_metrics = evaluate_regions(
        absolute_errors[conformal_mask],
        unscaled_half_widths,
    )

    max_error = absolute_errors.amax(dim=-1)
    spearman_all = approximate_spearman(tu, max_error)
    spearman_fit = approximate_spearman(tu[fit_mask], max_error[fit_mask])

    prototype_artifact = load_torch_checkpoint(prototypes_path)
    prototype_metadata = (
        prototype_artifact.get("metadata", {})
        if isinstance(prototype_artifact, Mapping)
        else {}
    )

    checkpoint_stat = checkpoint_path.stat()
    metadata: Dict[str, Any] = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "split_conformal_bbox_tu_scaled",
        "target_coverage": 1.0 - args.alpha,
        "alpha": args.alpha,
        "coordinate_format": "normalized_xyxy",
        "coordinate_order": list(COORDINATE_ORDER),
        "joint_score": "max_coordinate_absolute_error_over_scale",
        "coverage_scope": (
            "Hungarian-matched detections with finite top-k TU, satisfying "
            "the confidence threshold and configured class-correctness filter"
        ),
        "coverage_type": "marginal_over_accepted_detections",
        "split_by_image": True,
        "config": display_path(config_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint": display_path(checkpoint_path),
        "checkpoint_key": selected_checkpoint_key,
        "checkpoint_size_bytes": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "tu_prototypes": display_path(prototypes_path),
        "tu_prototypes_sha256": sha256_file(prototypes_path),
        "tu_prototype_metadata": {
            key: prototype_metadata.get(key)
            for key in (
                "method",
                "edge_score",
                "decoder_module",
                "decoder_eval_index",
                "bbox_layer_index",
                "signature_length",
                "checkpoint_key",
            )
        },
        "tu_module": tu_module_name,
        "tu_topk": args.tu_topk,
        "tu_min_samples": args.tu_min_samples,
        "split": args.split,
        "fit_fraction": args.fit_fraction,
        "num_tu_bins_requested": args.num_tu_bins,
        "num_tu_bins_actual": int(
            scale_model["coordinate_scales"].shape[0]
        ),
        "scale_quantile": args.scale_quantile,
        "scale_floor": args.scale_floor,
        "confidence_threshold": args.confidence_threshold,
        "require_correct_class": not args.allow_wrong_class,
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "counts": {
            "images_seen": images_seen,
            "hungarian_matches": matches_seen,
            "accepted": accepted,
            "scale_fit_images": int(fit_images.numel()),
            "conformal_images": int(conformal_images.numel()),
            "scale_fit_detections": fit_count,
            "conformal_detections": conformal_count,
            "rejected_nonfinite_tu": rejected_nonfinite_tu,
            "rejected_low_confidence": rejected_low_confidence,
            "rejected_wrong_class": rejected_wrong_class,
        },
    }
    if args.hash_checkpoint:
        print("[output] hashing detector checkpoint...")
        metadata["checkpoint_sha256"] = sha256_file(checkpoint_path)

    artifact: Dict[str, Any] = {
        "metadata": metadata,
        "scale_model": {
            "type": "monotone_tu_quantile_bins",
            "input": "bbox_tu",
            "coordinate_order": list(COORDINATE_ORDER),
            **scale_model,
        },
        "conformal": {
            "scaled": {
                "qhat": qhat_scaled.to(torch.float32),
                "rank": scaled_rank,
                "nominal_quantile_level": scaled_level,
                **scaled_metrics,
            },
            "unscaled": {
                "qhat": qhat_unscaled.to(torch.float32),
                "rank": unscaled_rank,
                "nominal_quantile_level": unscaled_level,
                **unscaled_metrics,
            },
        },
        "diagnostics": {
            "spearman_tu_vs_max_absolute_error_all": spearman_all,
            "spearman_tu_vs_max_absolute_error_fit": spearman_fit,
            "mean_tu": float(tu.mean()),
            "median_tu": float(tu.median()),
            "mean_confidence": float(confidences.mean()),
            "mean_matched_iou": float(ious.mean()),
            "accepted_per_class": torch.bincount(
                class_ids,
                minlength=int(class_ids.max()) + 1,
            ),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)

    print(
        f"[output] saved calibration artifact to {output_path}\n"
        f"[output] target coverage={1.0 - args.alpha:.3f}, "
        f"scaled empirical coverage={scaled_metrics['joint_coverage']:.4f}, "
        f"unscaled empirical coverage={unscaled_metrics['joint_coverage']:.4f}\n"
        f"[output] scaled mean coordinate width="
        f"{scaled_metrics['mean_coordinate_width']:.6f}, "
        f"unscaled={unscaled_metrics['mean_coordinate_width']:.6f}\n"
        f"[diagnostic] Spearman(TU, max bbox error)={spearman_all:.4f}"
    )
    if not math.isnan(spearman_all) and spearman_all <= 0:
        print(
            "[warning] TU is not positively associated with localization error "
            "on this calibration population. The unscaled CP baseline may be "
            "preferable; investigate before using TU-scaled regions."
        )
    return artifact


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()