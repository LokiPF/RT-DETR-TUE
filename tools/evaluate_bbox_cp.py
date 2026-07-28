#!/usr/bin/env python3
"""
Evaluate a frozen RT-DETR bbox conformal calibrator on a held-out COCO split.

This script never refits the TU scale model or either conformal quantile. It:

  * replaces the configured validation annotation file with --test-annotations;
  * runs the detector and bbox TU implementation;
  * applies the saved scaled and unscaled conformal calibrators;
  * reports marginal and coordinate coverage, interval width, and Spearman TU
    correlation;
  * reports localization-failure AUROC for TU, inverse confidence, and the
    scaled interval width;
  * reports marginal and groupwise conformal coverage-calibration errors and
    a coordinate interval score;
  * reports conditional results by TU decile, target class, and target size;
  * calculates image-cluster bootstrap confidence intervals; and
  * calculates an explicitly labelled oracle equal-coverage width diagnostic.

The evaluated population is copied from the calibration artifact: matched
detections, the same confidence threshold, and the same class-correctness
filter. Coverage does not include missed objects or unmatched false positives.

Run this file from the ``rtdetrv2_pytorch`` repository root, next to
``build_bbox_cp_calibration.py`` in ``tools/``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


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

try:
    from build_bbox_cp_calibration import (
        aligned_box_iou,
        box_cxcywh_to_xyxy,
        conformal_quantile,
        disable_pretrained_downloads,
        display_path,
        find_tu_module,
        get_image_id,
        load_model_weights,
        load_torch_checkpoint,
        lookup_coordinate_scales,
        move_targets_to_device,
        normalize_target_boxes_for_matcher,
        sha256_file,
        unpack_batch,
        unwrap_match_indices,
    )
except ImportError as exc:
    raise ImportError(
        "Could not import helpers from build_bbox_cp_calibration.py. Keep both "
        "scripts in the repository's `tools/` directory."
    ) from exc

try:
    from scipy.stats import rankdata, spearmanr
except ImportError:
    rankdata = None
    spearmanr = None


SIZE_FALLBACK_NAMES = ("small", "medium", "large")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen bbox CP on an untouched COCO test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--config", required=True, help="RT-DETR YAML config.")
    parser.add_argument("-r", "--resume", required=True, help="Detector checkpoint.")
    parser.add_argument(
        "--tu-prototypes",
        required=True,
        help="BBox TU prototype artifact used during CP calibration.",
    )
    parser.add_argument(
        "--cp-calibration",
        required=True,
        help="Frozen artifact created by build_bbox_cp_calibration.py.",
    )
    parser.add_argument(
        "--test-annotations",
        required=True,
        help="Held-out COCO annotation JSON. Never use the calibration JSON.",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help="Optional test image directory override.",
    )
    parser.add_argument("-o", "--output", required=True, help="Output metrics JSON.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )
    parser.add_argument(
        "--checkpoint-key",
        choices=("auto", "ema", "model", "module", "state_dict"),
        default="auto",
        help="Checkpoint entry containing detector weights.",
    )
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="Require the detector checkpoint to match exactly.",
    )
    parser.add_argument(
        "--target-box-format",
        choices=("auto", "xyxy", "cxcywh"),
        default="auto",
        help="Source format of dataloader target boxes.",
    )
    parser.add_argument(
        "--target-box-units",
        choices=("auto", "absolute", "normalized"),
        default="auto",
        help="Source units of dataloader target boxes.",
    )
    parser.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=500,
        help="Image-cluster bootstrap repetitions; 0 disables bootstrapping.",
    )
    parser.add_argument(
        "--bootstrap-confidence",
        type=float,
        default=0.95,
        help="Bootstrap confidence level.",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=30,
        help="Minimum accepted detections for reporting a class/group.",
    )
    parser.add_argument(
        "--localization-iou-thresholds",
        type=float,
        nargs="+",
        default=(0.50, 0.75),
        help=(
            "Define localization failures as matched IoU below each threshold "
            "and report failure-detection AUROC."
        ),
    )
    parser.add_argument(
        "--coordinate-error-thresholds",
        type=float,
        nargs="+",
        default=(0.05, 0.10),
        help=(
            "Also report AUROC for maximum normalized xyxy-coordinate error "
            "above each threshold."
        ),
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional smoke-test limit. Do not use for final reporting.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N dataloader batches; 0 disables it.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.bootstrap_repeats < 0:
        raise ValueError("--bootstrap-repeats cannot be negative.")
    if not 0.0 < args.bootstrap_confidence < 1.0:
        raise ValueError("--bootstrap-confidence must be in (0, 1).")
    if args.min_group_size < 1:
        raise ValueError("--min-group-size must be positive.")
    if any(
        not 0.0 < threshold < 1.0
        for threshold in args.localization_iou_thresholds
    ):
        raise ValueError("--localization-iou-thresholds must be in (0, 1).")
    if any(
        threshold <= 0.0 for threshold in args.coordinate_error_thresholds
    ):
        raise ValueError("--coordinate-error-thresholds must be positive.")
    if args.max_images is not None and args.max_images < 1:
        raise ValueError("--max-images must be positive.")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_nested_dataset_paths(
    node: Any,
    annotation_path: Path,
    image_root: Optional[Path],
) -> Tuple[int, int]:
    """Recursively update dataset paths and return (ann_updates, root_updates)."""
    annotation_updates = 0
    image_root_updates = 0

    if isinstance(node, MutableMapping):
        for key, value in list(node.items()):
            if key == "ann_file":
                node[key] = str(annotation_path)
                annotation_updates += 1
            elif image_root is not None and key in ("img_folder", "image_root"):
                node[key] = str(image_root)
                image_root_updates += 1
            else:
                ann_count, root_count = set_nested_dataset_paths(
                    value,
                    annotation_path,
                    image_root,
                )
                annotation_updates += ann_count
                image_root_updates += root_count
    elif isinstance(node, list):
        for value in node:
            ann_count, root_count = set_nested_dataset_paths(
                value,
                annotation_path,
                image_root,
            )
            annotation_updates += ann_count
            image_root_updates += root_count

    return annotation_updates, image_root_updates


def load_class_names(annotation_path: Path) -> Dict[int, str]:
    with annotation_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    categories = data.get("categories", [])
    if not isinstance(categories, list):
        return {}

    ordered = sorted(categories, key=lambda category: int(category["id"]))
    return {
        contiguous_id: str(category.get("name", f"class_{contiguous_id}"))
        for contiguous_id, category in enumerate(ordered)
    }


def exact_spearman(left: Tensor, right: Tensor) -> float:
    left_cpu = left.detach().float().flatten().cpu()
    right_cpu = right.detach().float().flatten().cpu()
    if left_cpu.numel() < 2:
        return float("nan")
    if float(left_cpu.std()) == 0.0 or float(right_cpu.std()) == 0.0:
        return float("nan")

    if spearmanr is not None:
        result = spearmanr(left_cpu.numpy(), right_cpu.numpy())
        statistic = getattr(result, "statistic", result[0])
        return float(statistic)

    # Fallback without average-tie correction. SciPy is normally available
    # because RT-DETR's Hungarian matcher depends on it.
    left_rank = torch.argsort(torch.argsort(left_cpu)).float()
    right_rank = torch.argsort(torch.argsort(right_cpu)).float()
    return float(torch.corrcoef(torch.stack((left_rank, right_rank)))[0, 1])


def binary_auroc(scores: Tensor, positive_labels: Tensor) -> float:
    """AUROC with larger scores interpreted as more likely positive."""
    scores_cpu = scores.detach().float().flatten().cpu()
    labels_cpu = positive_labels.detach().bool().flatten().cpu()
    if scores_cpu.shape != labels_cpu.shape:
        raise ValueError("AUROC scores and labels must have the same shape.")

    finite = torch.isfinite(scores_cpu)
    scores_cpu = scores_cpu[finite]
    labels_cpu = labels_cpu[finite]
    positive_count = int(labels_cpu.sum())
    negative_count = int((~labels_cpu).sum())
    if positive_count == 0 or negative_count == 0:
        return float("nan")

    if rankdata is not None:
        ranks = torch.from_numpy(
            rankdata(scores_cpu.numpy(), method="average")
        ).to(torch.float64)
    else:
        # Fallback has no average-tie correction. SciPy is normally available
        # because RT-DETR's Hungarian matcher depends on it.
        order = torch.argsort(scores_cpu, stable=True)
        ranks = torch.empty_like(scores_cpu, dtype=torch.float64)
        ranks[order] = torch.arange(
            1,
            scores_cpu.numel() + 1,
            dtype=torch.float64,
        )

    positive_rank_sum = float(ranks[labels_cpu].sum())
    correction = positive_count * (positive_count + 1) / 2.0
    return (
        positive_rank_sum - correction
    ) / (positive_count * negative_count)


def ranking_metric_table(
    scores: Mapping[str, Tensor],
    events: Mapping[str, Tensor],
) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    for event_name, labels in events.items():
        labels_bool = labels.detach().bool().flatten().cpu()
        event_result: Dict[str, Any] = {
            "count": int(labels_bool.numel()),
            "positive_count": int(labels_bool.sum()),
            "negative_count": int((~labels_bool).sum()),
            "prevalence": float(labels_bool.float().mean()),
            "auroc": {},
        }
        for score_name, score in scores.items():
            event_result["auroc"][score_name] = binary_auroc(
                score,
                labels_bool,
            )
        report[event_name] = event_result
    return report


def coordinate_interval_score(
    errors: Tensor,
    half_widths: Tensor,
    alpha: float,
) -> Dict[str, Any]:
    """
    Width plus a penalty for misses, using the joint CP alpha.

    This is useful for comparing the two interval systems, but because qhat was
    fitted for joint four-coordinate coverage, it is labelled as a coordinate
    score rather than claimed as a separately calibrated marginal score.
    """
    widths = 2.0 * half_widths
    miss_penalty = (2.0 / alpha) * (errors - half_widths).clamp_min(0.0)
    scores = widths + miss_penalty
    return {
        "alpha": alpha,
        "mean": float(scores.mean()),
        "median": float(scores.flatten().median()),
        "mean_by_coordinate": tensor_list(scores.mean(dim=0)),
        "mean_width_component": float(widths.mean()),
        "mean_miss_penalty_component": float(miss_penalty.mean()),
        "lower_is_better": True,
    }


def group_coverage_error(
    groups: Sequence[Mapping[str, Any]],
    method: str,
    target_coverage: float,
) -> Dict[str, Any]:
    usable = [
        group
        for group in groups
        if int(group.get("count", 0)) > 0
        and method in group
        and "joint_coverage" in group[method]
    ]
    if not usable:
        return {"group_count": 0}

    coverages = torch.tensor(
        [
            float(group[method]["joint_coverage"])
            for group in usable
        ],
        dtype=torch.float64,
    )
    counts = torch.tensor(
        [float(group["count"]) for group in usable],
        dtype=torch.float64,
    )
    absolute_errors = (coverages - target_coverage).abs()
    squared_errors = (coverages - target_coverage).square()
    weights = counts / counts.sum()
    return {
        "group_count": len(usable),
        "mean_absolute_error": float(absolute_errors.mean()),
        "weighted_mean_absolute_error": float(
            (weights * absolute_errors).sum()
        ),
        "root_mean_squared_error": float(squared_errors.mean().sqrt()),
        "maximum_absolute_error": float(absolute_errors.max()),
        "coverage_range": float(coverages.max() - coverages.min()),
        "mean_absolute_error_percentage_points": float(
            100.0 * absolute_errors.mean()
        ),
        "maximum_absolute_error_percentage_points": float(
            100.0 * absolute_errors.max()
        ),
    }


def coverage_calibration_report(
    overall: Mapping[str, Any],
    conditional: Mapping[str, Any],
    target_coverage: float,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"target_coverage": target_coverage}
    for method in ("scaled", "unscaled"):
        observed = float(overall[method]["joint_coverage"])
        signed_error = observed - target_coverage
        result[method] = {
            "marginal": {
                "observed_coverage": observed,
                "signed_error": signed_error,
                "absolute_error": abs(signed_error),
                "signed_error_percentage_points": 100.0 * signed_error,
                "absolute_error_percentage_points": 100.0
                * abs(signed_error),
            },
            "tu_deciles": group_coverage_error(
                conditional.get("tu_deciles", []),
                method,
                target_coverage,
            ),
            "classes": group_coverage_error(
                conditional.get("classes", []),
                method,
                target_coverage,
            ),
            "target_sizes": group_coverage_error(
                conditional.get("target_sizes", []),
                method,
                target_coverage,
            ),
        }
    return result


def clipped_coordinate_widths(
    predicted_xyxy: Tensor,
    half_widths: Tensor,
) -> Tensor:
    lower = (predicted_xyxy - half_widths).clamp(0.0, 1.0)
    upper = (predicted_xyxy + half_widths).clamp(0.0, 1.0)
    return upper - lower


def region_vectors(
    errors: Tensor,
    predicted_xyxy: Tensor,
    half_widths: Tensor,
) -> Dict[str, Tensor]:
    covered_per_coordinate = errors <= half_widths
    return {
        "joint_covered": covered_per_coordinate.all(dim=-1),
        "coordinate_covered": covered_per_coordinate,
        "coordinate_widths": 2.0 * half_widths,
        "clipped_coordinate_widths": clipped_coordinate_widths(
            predicted_xyxy,
            half_widths,
        ),
    }


def tensor_list(tensor: Tensor) -> List[float]:
    return [float(value) for value in tensor.detach().cpu().tolist()]


def summarize_vectors(vectors: Mapping[str, Tensor], mask: Tensor) -> Dict[str, Any]:
    count = int(mask.sum())
    if count == 0:
        return {"count": 0}

    joint = vectors["joint_covered"][mask].float()
    coordinate_covered = vectors["coordinate_covered"][mask].float()
    widths = vectors["coordinate_widths"][mask].float()
    clipped_widths = vectors["clipped_coordinate_widths"][mask].float()

    return {
        "count": count,
        "joint_coverage": float(joint.mean()),
        "coordinate_coverage": tensor_list(coordinate_covered.mean(dim=0)),
        "mean_coordinate_width": float(widths.mean()),
        "median_coordinate_width": float(widths.flatten().median()),
        "mean_width_by_coordinate": tensor_list(widths.mean(dim=0)),
        "median_width_by_coordinate": tensor_list(
            widths.median(dim=0).values
        ),
        "mean_clipped_coordinate_width": float(clipped_widths.mean()),
    }


def summarize_pair(
    scaled_vectors: Mapping[str, Tensor],
    unscaled_vectors: Mapping[str, Tensor],
    mask: Tensor,
) -> Dict[str, Any]:
    scaled = summarize_vectors(scaled_vectors, mask)
    unscaled = summarize_vectors(unscaled_vectors, mask)
    result: Dict[str, Any] = {
        "count": int(mask.sum()),
        "scaled": scaled,
        "unscaled": unscaled,
    }
    if int(mask.sum()) > 0:
        difference = (
            scaled["mean_coordinate_width"]
            - unscaled["mean_coordinate_width"]
        )
        denominator = unscaled["mean_coordinate_width"]
        result["scaled_minus_unscaled_mean_width"] = difference
        result["scaled_minus_unscaled_mean_width_percent"] = (
            100.0 * difference / denominator
            if denominator > 0
            else float("nan")
        )
        result["scaled_minus_unscaled_coverage"] = (
            scaled["joint_coverage"] - unscaled["joint_coverage"]
        )
    return result


def percentile_interval(
    values: Sequence[float],
    confidence: float,
) -> Dict[str, float]:
    tensor = torch.tensor(
        [value for value in values if math.isfinite(value)],
        dtype=torch.float64,
    )
    if tensor.numel() == 0:
        return {"lower": float("nan"), "upper": float("nan")}
    tail = (1.0 - confidence) / 2.0
    quantiles = torch.quantile(
        tensor,
        torch.tensor([tail, 1.0 - tail], dtype=tensor.dtype),
    )
    return {
        "lower": float(quantiles[0]),
        "upper": float(quantiles[1]),
    }


def image_cluster_bootstrap(
    image_ids: Tensor,
    tu: Tensor,
    max_error: Tensor,
    scaled_vectors: Mapping[str, Tensor],
    unscaled_vectors: Mapping[str, Tensor],
    repeats: int,
    confidence: float,
    seed: int,
) -> Dict[str, Any]:
    if repeats == 0:
        return {"repeats": 0}

    unique_images = torch.unique(image_ids.cpu(), sorted=True)
    if unique_images.numel() < 2:
        raise RuntimeError("Image bootstrap requires at least two test images.")

    groups = [
        torch.nonzero(image_ids.cpu() == image_id, as_tuple=False).flatten()
        for image_id in unique_images
    ]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    bootstrap_spearman: List[float] = []
    scaled_coverage: List[float] = []
    unscaled_coverage: List[float] = []
    scaled_width: List[float] = []
    unscaled_width: List[float] = []
    width_difference: List[float] = []

    for _ in range(repeats):
        sampled_groups = torch.randint(
            len(groups),
            (len(groups),),
            generator=generator,
        )
        indices = torch.cat([groups[int(index)] for index in sampled_groups])

        bootstrap_spearman.append(
            exact_spearman(tu[indices], max_error[indices])
        )
        current_scaled_coverage = float(
            scaled_vectors["joint_covered"][indices].float().mean()
        )
        current_unscaled_coverage = float(
            unscaled_vectors["joint_covered"][indices].float().mean()
        )
        current_scaled_width = float(
            scaled_vectors["coordinate_widths"][indices].mean()
        )
        current_unscaled_width = float(
            unscaled_vectors["coordinate_widths"][indices].mean()
        )
        scaled_coverage.append(current_scaled_coverage)
        unscaled_coverage.append(current_unscaled_coverage)
        scaled_width.append(current_scaled_width)
        unscaled_width.append(current_unscaled_width)
        width_difference.append(current_scaled_width - current_unscaled_width)

    return {
        "repeats": repeats,
        "confidence_level": confidence,
        "resampling_unit": "image",
        "spearman": percentile_interval(bootstrap_spearman, confidence),
        "scaled_joint_coverage": percentile_interval(
            scaled_coverage,
            confidence,
        ),
        "unscaled_joint_coverage": percentile_interval(
            unscaled_coverage,
            confidence,
        ),
        "scaled_mean_coordinate_width": percentile_interval(
            scaled_width,
            confidence,
        ),
        "unscaled_mean_coordinate_width": percentile_interval(
            unscaled_width,
            confidence,
        ),
        "scaled_minus_unscaled_mean_width": percentile_interval(
            width_difference,
            confidence,
        ),
    }


def conditional_reports(
    tu: Tensor,
    class_ids: Tensor,
    size_ids: Tensor,
    class_names: Mapping[int, str],
    size_names: Sequence[str],
    scaled_vectors: Mapping[str, Tensor],
    unscaled_vectors: Mapping[str, Tensor],
    min_group_size: int,
) -> Dict[str, Any]:
    reports: Dict[str, Any] = {}

    quantile_levels = torch.linspace(0.0, 1.0, 11)[1:-1]
    decile_edges = torch.unique_consecutive(torch.quantile(tu, quantile_levels))
    decile_ids = torch.bucketize(tu, decile_edges)
    deciles: List[Dict[str, Any]] = []
    for decile_id in range(int(decile_edges.numel()) + 1):
        mask = decile_ids == decile_id
        if int(mask.sum()) < min_group_size:
            continue
        item = summarize_pair(scaled_vectors, unscaled_vectors, mask)
        item.update(
            {
                "bin": decile_id,
                "tu_min": float(tu[mask].min()),
                "tu_max": float(tu[mask].max()),
                "tu_mean": float(tu[mask].mean()),
            }
        )
        deciles.append(item)
    reports["tu_deciles"] = deciles
    reports["tu_decile_edges"] = tensor_list(decile_edges)

    classes: List[Dict[str, Any]] = []
    for class_id_tensor in torch.unique(class_ids, sorted=True):
        class_id = int(class_id_tensor)
        mask = class_ids == class_id
        if int(mask.sum()) < min_group_size:
            continue
        item = summarize_pair(scaled_vectors, unscaled_vectors, mask)
        item.update(
            {
                "class_id": class_id,
                "class_name": class_names.get(class_id, f"class_{class_id}"),
            }
        )
        classes.append(item)
    reports["classes"] = classes

    sizes: List[Dict[str, Any]] = []
    for size_id in range(len(size_names)):
        mask = size_ids == size_id
        if int(mask.sum()) < min_group_size:
            continue
        item = summarize_pair(scaled_vectors, unscaled_vectors, mask)
        item.update(
            {
                "size_id": size_id,
                "size_name": size_names[size_id],
            }
        )
        sizes.append(item)
    reports["target_sizes"] = sizes
    return reports


def oracle_equal_coverage(
    errors: Tensor,
    predicted_xyxy: Tensor,
    scaled_coordinate_scales: Tensor,
    alpha: float,
) -> Dict[str, Any]:
    """
    Refit qhat on test labels only to compare efficiency at exactly one coverage.

    This is a descriptive oracle diagnostic. Its qhat values must never be
    copied into deployment or treated as held-out coverage evidence.
    """
    safe_scales = scaled_coordinate_scales.clamp_min(
        torch.finfo(scaled_coordinate_scales.dtype).eps
    )
    scaled_scores = (errors / safe_scales).amax(dim=-1)
    unscaled_scores = errors.amax(dim=-1)
    scaled_qhat, _, _ = conformal_quantile(scaled_scores, alpha)
    unscaled_qhat, _, _ = conformal_quantile(unscaled_scores, alpha)

    scaled_half_widths = scaled_qhat * scaled_coordinate_scales
    unscaled_half_widths = torch.ones_like(errors) * unscaled_qhat
    scaled_vectors = region_vectors(
        errors,
        predicted_xyxy,
        scaled_half_widths,
    )
    unscaled_vectors = region_vectors(
        errors,
        predicted_xyxy,
        unscaled_half_widths,
    )
    full_mask = torch.ones(errors.shape[0], dtype=torch.bool)
    result = summarize_pair(scaled_vectors, unscaled_vectors, full_mask)
    result.update(
        {
            "target_coverage": 1.0 - alpha,
            "scaled_oracle_qhat": float(scaled_qhat),
            "unscaled_oracle_qhat": float(unscaled_qhat),
            "warning": (
                "Descriptive test-label oracle only; do not use these qhat "
                "values for inference."
            ),
        }
    )
    return result


def validate_artifacts(
    cp_artifact: Mapping[str, Any],
    prototype_path: Path,
) -> Mapping[str, Any]:
    required = ("metadata", "scale_model", "conformal")
    missing = [key for key in required if key not in cp_artifact]
    if missing:
        raise KeyError(f"CP artifact is missing {missing}.")
    if "scaled" not in cp_artifact["conformal"]:
        raise KeyError("CP artifact has no scaled calibrator.")
    if "unscaled" not in cp_artifact["conformal"]:
        raise KeyError("CP artifact has no unscaled baseline.")

    metadata = cp_artifact["metadata"]
    if metadata.get("coordinate_format") != "normalized_xyxy":
        raise ValueError("CP artifact does not use normalized_xyxy coordinates.")

    expected_hash = metadata.get("tu_prototypes_sha256")
    if expected_hash and sha256_file(prototype_path) != expected_hash:
        raise ValueError(
            "TU prototype artifact differs from the one used for calibration."
        )
    return metadata


def write_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, allow_nan=True)
            file.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(args: argparse.Namespace) -> Dict[str, Any]:
    validate_args(args)
    seed_everything(args.seed)

    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.resume).expanduser().resolve()
    prototype_path = Path(args.tu_prototypes).expanduser().resolve()
    cp_path = Path(args.cp_calibration).expanduser().resolve()
    annotation_path = Path(args.test_annotations).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    image_root = (
        Path(args.image_root).expanduser().resolve()
        if args.image_root is not None
        else None
    )
    for label, path in (
        ("Config", config_path),
        ("Checkpoint", checkpoint_path),
        ("TU prototypes", prototype_path),
        ("CP calibration", cp_path),
        ("Test annotations", annotation_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if image_root is not None and not image_root.is_dir():
        raise NotADirectoryError(f"Image root is not a directory: {image_root}")

    cp_artifact = load_torch_checkpoint(cp_path)
    if not isinstance(cp_artifact, Mapping):
        raise TypeError("CP calibration artifact is not a mapping.")
    cp_metadata = validate_artifacts(cp_artifact, prototype_path)

    confidence_threshold = float(
        cp_metadata.get("confidence_threshold", 0.0)
    )
    require_correct_class = bool(
        cp_metadata.get("require_correct_class", True)
    )
    tu_topk = int(cp_metadata.get("tu_topk", 50))
    tu_min_samples = int(cp_metadata.get("tu_min_samples", 20))
    target_coverage = float(cp_metadata.get("target_coverage", 0.95))
    alpha = 1.0 - target_coverage
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"Invalid target coverage {target_coverage}.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    print(f"[config] loading {config_path}")
    cfg = YAMLConfig(str(config_path))
    yaml_cfg = getattr(cfg, "yaml_cfg", None)
    if not isinstance(yaml_cfg, MutableMapping):
        raise TypeError("YAMLConfig does not expose a mutable `yaml_cfg`.")
    disable_pretrained_downloads(yaml_cfg)

    val_cfg = yaml_cfg.get("val_dataloader")
    if val_cfg is None:
        raise KeyError("Configuration does not contain `val_dataloader`.")
    annotation_updates, root_updates = set_nested_dataset_paths(
        val_cfg,
        annotation_path,
        image_root,
    )
    if annotation_updates == 0:
        raise KeyError(
            "Could not find `ann_file` under val_dataloader to replace."
        )
    if image_root is not None and root_updates == 0:
        raise KeyError(
            "Could not find `img_folder` or `image_root` under val_dataloader."
        )
    print(
        f"[data] test annotations={annotation_path}; "
        f"updated {annotation_updates} annotation path(s)"
    )

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
        tu_module.tu_min_samples = tu_min_samples
    if hasattr(tu_module, "bbox_tu_topk"):
        tu_module.bbox_tu_topk = tu_topk
    tu_module.load_tu_prototypes(str(prototype_path))
    tu_module.tu_enabled = True

    model.eval()
    criterion.eval()
    dataloader = cfg.val_dataloader
    print(
        f"[population] confidence>={confidence_threshold}, "
        f"require_correct_class={require_correct_class}, tu_topk={tu_topk}"
    )

    collected_tu: List[Tensor] = []
    collected_errors: List[Tensor] = []
    collected_predictions: List[Tensor] = []
    collected_classes: List[Tensor] = []
    collected_sizes: List[Tensor] = []
    collected_confidences: List[Tensor] = []
    collected_ious: List[Tensor] = []
    collected_image_ids: List[Tensor] = []

    images_seen = 0
    matches_seen = 0
    accepted = 0
    rejected_nonfinite_tu = 0
    rejected_low_confidence = 0
    rejected_wrong_class = 0
    target_box_conversions: Dict[str, int] = {}

    prototype_artifact = load_torch_checkpoint(prototype_path)
    prototype_metadata = prototype_artifact.get("metadata", {})
    area_metadata = prototype_metadata.get(
        "normalized_area_thresholds",
        {"small_max": 0.0025, "medium_max": 0.0225},
    )
    small_max_area = float(area_metadata["small_max"])
    medium_max_area = float(area_metadata["medium_max"])
    size_names = list(
        prototype_metadata.get("size_bucket_names", SIZE_FALLBACK_NAMES)
    )

    with torch.inference_mode():
        for batch_index, batch in enumerate(dataloader):
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
            moved_targets = move_targets_to_device(targets_cpu, device)
            sample_height = int(samples.shape[-2])
            sample_width = int(samples.shape[-1])

            targets: List[Dict[str, Any]] = []
            for target in moved_targets:
                normalized_target, conversion = normalize_target_boxes_for_matcher(
                    target,
                    sample_height,
                    sample_width,
                    args.target_box_format,
                    args.target_box_units,
                )
                targets.append(normalized_target)
                target_box_conversions[conversion] = (
                    target_box_conversions.get(conversion, 0) + 1
                )
            if batch_index == 0:
                print(
                    "[data] converted targets to normalized cxcywh; "
                    f"detected={target_box_conversions}"
                )

            outputs = model(samples)
            if not isinstance(outputs, Mapping):
                raise TypeError("Model output is not a mapping.")
            required_outputs = ("pred_logits", "pred_boxes", "bbox_tu")
            missing_outputs = [
                key for key in required_outputs if key not in outputs
            ]
            if missing_outputs:
                raise KeyError(f"Model output is missing {missing_outputs}.")

            pred_logits = outputs["pred_logits"]
            pred_boxes = outputs["pred_boxes"]
            bbox_tu = outputs["bbox_tu"]
            probabilities = pred_logits.sigmoid()
            query_confidences, query_classes = probabilities.max(dim=-1)

            match_result = matcher(
                {"pred_logits": pred_logits, "pred_boxes": pred_boxes},
                targets,
            )
            match_indices = unwrap_match_indices(match_result)

            for image_index, (source_index, target_index) in enumerate(
                match_indices
            ):
                source_index = source_index.to(device=device, dtype=torch.long)
                target_index = target_index.to(device=device, dtype=torch.long)
                matches_seen += int(source_index.numel())
                if source_index.numel() == 0:
                    continue

                tu_values = bbox_tu[image_index, source_index].float()
                confidences = query_confidences[image_index, source_index]
                predicted_classes = query_classes[image_index, source_index]
                target_classes = targets[image_index]["labels"][target_index]

                finite_mask = torch.isfinite(tu_values)
                confidence_mask = confidences >= confidence_threshold
                if require_correct_class:
                    class_mask = predicted_classes == target_classes
                else:
                    class_mask = torch.ones_like(finite_mask)

                rejected_nonfinite_tu += int((~finite_mask).sum())
                rejected_low_confidence += int(
                    (finite_mask & ~confidence_mask).sum()
                )
                rejected_wrong_class += int(
                    (finite_mask & confidence_mask & ~class_mask).sum()
                )

                keep = finite_mask & confidence_mask & class_mask
                if not bool(keep.any()):
                    continue

                kept_source = source_index[keep]
                kept_target = target_index[keep]
                kept_predictions = pred_boxes[image_index, kept_source]
                kept_targets = targets[image_index]["boxes"][kept_target]
                predicted_xyxy = box_cxcywh_to_xyxy(kept_predictions)
                target_xyxy = box_cxcywh_to_xyxy(kept_targets)
                errors = (target_xyxy - predicted_xyxy).abs()
                ious = aligned_box_iou(kept_predictions, kept_targets)

                target_areas = (
                    kept_targets[:, 2].clamp_min(0)
                    * kept_targets[:, 3].clamp_min(0)
                )
                target_sizes = (
                    (target_areas >= small_max_area).long()
                    + (target_areas >= medium_max_area).long()
                )

                image_id = get_image_id(
                    targets_cpu[image_index],
                    fallback_image_ids[image_index],
                )
                kept_count = int(keep.sum())
                collected_tu.extend(tu_values[keep].detach().cpu().unbind())
                collected_errors.extend(errors.detach().cpu().unbind())
                collected_predictions.extend(
                    predicted_xyxy.detach().cpu().unbind()
                )
                collected_classes.extend(
                    target_classes[keep].detach().cpu().unbind()
                )
                collected_sizes.extend(target_sizes.detach().cpu().unbind())
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
            if (
                args.progress_every > 0
                and (batch_index + 1) % args.progress_every == 0
            ):
                print(
                    f"[progress] batches={batch_index + 1}, "
                    f"images={images_seen}, matches={matches_seen}, "
                    f"accepted={accepted}"
                )

    if accepted == 0:
        raise RuntimeError("No test detections passed the calibrated population.")

    tu = torch.stack(collected_tu).float()
    errors = torch.stack(collected_errors).float()
    predicted_xyxy = torch.stack(collected_predictions).float()
    class_ids = torch.stack(collected_classes).long()
    size_ids = torch.stack(collected_sizes).long()
    confidences = torch.stack(collected_confidences).float()
    ious = torch.stack(collected_ious).float()
    image_ids = torch.stack(collected_image_ids).long()
    max_error = errors.amax(dim=-1)

    coordinate_scales = lookup_coordinate_scales(
        tu,
        cp_artifact["scale_model"],
    )
    scaled_qhat = float(cp_artifact["conformal"]["scaled"]["qhat"])
    unscaled_qhat = float(cp_artifact["conformal"]["unscaled"]["qhat"])
    scaled_half_widths = scaled_qhat * coordinate_scales
    unscaled_half_widths = torch.ones_like(errors) * unscaled_qhat
    scaled_vectors = region_vectors(
        errors,
        predicted_xyxy,
        scaled_half_widths,
    )
    unscaled_vectors = region_vectors(
        errors,
        predicted_xyxy,
        unscaled_half_widths,
    )
    full_mask = torch.ones(accepted, dtype=torch.bool)

    overall = summarize_pair(
        scaled_vectors,
        unscaled_vectors,
        full_mask,
    )
    spearman = exact_spearman(tu, max_error)
    class_names = load_class_names(annotation_path)
    groups = conditional_reports(
        tu,
        class_ids,
        size_ids,
        class_names,
        size_names,
        scaled_vectors,
        unscaled_vectors,
        args.min_group_size,
    )
    bootstrap = image_cluster_bootstrap(
        image_ids,
        tu,
        max_error,
        scaled_vectors,
        unscaled_vectors,
        args.bootstrap_repeats,
        args.bootstrap_confidence,
        args.seed,
    )
    equal_coverage = oracle_equal_coverage(
        errors,
        predicted_xyxy,
        coordinate_scales,
        alpha,
    )
    failure_events: Dict[str, Tensor] = {}
    for threshold in args.localization_iou_thresholds:
        failure_events[f"matched_iou_below_{threshold:g}"] = ious < threshold
    for threshold in args.coordinate_error_thresholds:
        failure_events[
            f"max_coordinate_error_above_{threshold:g}"
        ] = max_error > threshold

    ranking_scores = {
        "bbox_tu": tu,
        "inverse_class_confidence": 1.0 - confidences,
        "scaled_mean_coordinate_width": (
            2.0 * scaled_half_widths
        ).mean(dim=-1),
    }
    localization_failure_ranking = ranking_metric_table(
        ranking_scores,
        failure_events,
    )
    coverage_calibration = coverage_calibration_report(
        overall,
        groups,
        target_coverage,
    )
    interval_scores = {
        "scaled": coordinate_interval_score(
            errors,
            scaled_half_widths,
            alpha,
        ),
        "unscaled": coordinate_interval_score(
            errors,
            unscaled_half_widths,
            alpha,
        ),
        "note": (
            "Coordinate interval score evaluated with the joint conformal "
            "alpha; lower is better."
        ),
    }

    result: Dict[str, Any] = {
        "metadata": {
            "format_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "evaluation_type": "held_out_test",
            "config": display_path(config_path),
            "config_sha256": sha256_file(config_path),
            "checkpoint": display_path(checkpoint_path),
            "checkpoint_key": selected_checkpoint_key,
            "tu_prototypes": display_path(prototype_path),
            "tu_prototypes_sha256": sha256_file(prototype_path),
            "cp_calibration": display_path(cp_path),
            "cp_calibration_sha256": sha256_file(cp_path),
            "test_annotations": display_path(annotation_path),
            "test_annotations_sha256": sha256_file(annotation_path),
            "image_root": (
                display_path(image_root) if image_root is not None else None
            ),
            "tu_module": tu_module_name,
            "target_coverage": target_coverage,
            "confidence_threshold": confidence_threshold,
            "require_correct_class": require_correct_class,
            "tu_topk": tu_topk,
            "tu_min_samples": tu_min_samples,
            "target_box_conversions": target_box_conversions,
            "counts": {
                "images_seen": images_seen,
                "unique_images_with_accepted_detections": int(
                    torch.unique(image_ids).numel()
                ),
                "hungarian_matches": matches_seen,
                "accepted": accepted,
                "rejected_nonfinite_tu": rejected_nonfinite_tu,
                "rejected_low_confidence": rejected_low_confidence,
                "rejected_wrong_class": rejected_wrong_class,
            },
        },
        "diagnostics": {
            "spearman_tu_vs_max_absolute_error": spearman,
            "spearman_inverse_confidence_vs_max_absolute_error": (
                exact_spearman(1.0 - confidences, max_error)
            ),
            "spearman_scaled_width_vs_max_absolute_error": exact_spearman(
                ranking_scores["scaled_mean_coordinate_width"],
                max_error,
            ),
            "mean_tu": float(tu.mean()),
            "median_tu": float(tu.median()),
            "mean_max_absolute_error": float(max_error.mean()),
            "mean_confidence": float(confidences.mean()),
            "mean_matched_iou": float(ious.mean()),
        },
        "frozen_calibrators": overall,
        "coverage_calibration_errors": coverage_calibration,
        "coordinate_interval_scores": interval_scores,
        "localization_failure_ranking": localization_failure_ranking,
        "bootstrap_confidence_intervals": bootstrap,
        "conditional": groups,
        "oracle_equal_coverage_diagnostic": equal_coverage,
        "metric_applicability": {
            "localization_failure_auroc": {
                "available": True,
                "scope": (
                    "Accepted Hungarian-matched detections; this is not OOD "
                    "AUROC and excludes missed objects and unmatched false "
                    "positives."
                ),
            },
            "conformal_coverage_calibration_error": {
                "available": True,
                "scope": (
                    "Marginal and empirical groupwise coverage error for the "
                    "frozen conformal regions."
                ),
            },
            "native_localization_nll": {
                "available": False,
                "reason": (
                    "TU-scaled conformal intervals do not define a normalized "
                    "predictive probability density. A Gaussian or Laplace NLL "
                    "would require an additional distributional assumption "
                    "and would only be a proxy."
                ),
            },
            "pdq": {
                "available": False,
                "reason": (
                    "PDQ requires probabilistic spatial foreground/background "
                    "occupancy and semantic probabilities for all detections, "
                    "including false positives and false negatives. Coordinate "
                    "conformal intervals alone do not provide that output."
                ),
            },
            "classification_ece": {
                "available": False,
                "reason": (
                    "The calibrated population filters to matched detections "
                    "and may require the predicted class to be correct. A "
                    "proper detection confidence ECE evaluation must retain "
                    "wrong classes, unmatched false positives, and misses."
                ),
            },
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, result)

    auroc_lines = "\n".join(
        (
            f"[AUROC] {event_name}: "
            f"TU={event_result['auroc']['bbox_tu']:.4f}, "
            f"1-confidence="
            f"{event_result['auroc']['inverse_class_confidence']:.4f}, "
            f"prevalence={event_result['prevalence']:.4f}"
        )
        for event_name, event_result in localization_failure_ranking.items()
    )
    print(
        f"[test] detections={accepted}, Spearman={spearman:.4f}\n"
        f"[test] scaled coverage="
        f"{overall['scaled']['joint_coverage']:.4f}, width="
        f"{overall['scaled']['mean_coordinate_width']:.6f}\n"
        f"[test] unscaled coverage="
        f"{overall['unscaled']['joint_coverage']:.4f}, width="
        f"{overall['unscaled']['mean_coordinate_width']:.6f}\n"
        f"[test] scaled-unscaled width="
        f"{overall['scaled_minus_unscaled_mean_width']:.6f} "
        f"({overall['scaled_minus_unscaled_mean_width_percent']:.3f}%)\n"
        f"[calibration] TU-decile MAE: scaled="
        f"{coverage_calibration['scaled']['tu_deciles'].get('mean_absolute_error_percentage_points', float('nan')):.3f}pp, "
        f"unscaled="
        f"{coverage_calibration['unscaled']['tu_deciles'].get('mean_absolute_error_percentage_points', float('nan')):.3f}pp\n"
        f"{auroc_lines}\n"
        f"[oracle equal coverage] scaled width="
        f"{equal_coverage['scaled']['mean_coordinate_width']:.6f}, "
        f"unscaled width="
        f"{equal_coverage['unscaled']['mean_coordinate_width']:.6f}\n"
        f"[output] saved metrics to {output_path}"
    )
    return result


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()