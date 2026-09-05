#!/usr/bin/env python3
"""Evaluate a probabilistic box head in the official RT-DETRv2 repository.

The model is expected to return normalized ``cx, cy, w, h`` boxes and one
Gaussian log-variance per coordinate. Both of these output names are accepted:

    pred_box_logvars
    pred_boxes_logvars

Run this script from ``RT-DETR/rtdetrv2_pytorch`` (or pass ``--repo-root``).
It uses the repository's ``src.core.YAMLConfig`` and official checkpoint/deploy
path. The script reads a COCO-format validation set, draws mean boxes and an
approximate 95% uncertainty envelope, and evaluates localization uncertainty on
class-aware Hungarian matches.

Example:

    python evaluate_probabilistic_rtdetr.py \
        -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml \
        -r output/probabilistic/best.pth \
        --output-dir uncertainty_eval

By default the image and annotation paths are read from ``val_dataloader`` in
the YAML configuration. They can be overridden with ``--images`` and
``--annotations``. Run with ``--help`` for all options.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches
from PIL import Image

try:
    from scipy.optimize import linear_sum_assignment
    from scipy.stats import spearmanr
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise SystemExit("This script requires scipy: pip install scipy") from exc


LOGVAR_MIN = -10.0
LOGVAR_MAX = 4.0
EPS = 1e-12
Z_SCORES = {
    "50": 0.6744897502,
    "80": 1.2815515655,
    "90": 1.6448536270,
    "95": 1.9599639845,
}


@dataclass
class CocoImage:
    image_id: int
    file_name: str
    width: int
    height: int
    boxes: np.ndarray  # normalized cxcywh, [N, 4]
    labels: np.ndarray  # contiguous labels, [N]


@dataclass
class Prediction:
    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray
    logvars: np.ndarray
    uncertainty: np.ndarray
    query_indices: np.ndarray


@dataclass
class Match:
    image_id: int
    file_name: str
    pred_index: int
    query_index: int
    gt_index: int
    label: int
    score: float
    iou: float
    box: np.ndarray
    gt_box: np.ndarray
    logvar: np.ndarray
    uncertainty: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Official rtdetrv2_pytorch directory, or the parent RT-DETR directory",
    )
    parser.add_argument("-c", "--config", required=True, type=Path)
    parser.add_argument("-r", "--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--images", type=Path, help="Override validation image directory"
    )
    parser.add_argument(
        "--annotations", type=Path, help="Override COCO annotation JSON"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("uncertainty_eval"))
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="Override YAML eval_spatial_size",
    )
    parser.add_argument(
        "--num-images", type=int, default=None, help="Limit evaluation images"
    )
    parser.add_argument("--plot-limit", type=int, default=20)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override cfg.postprocessor.num_top_queries",
    )
    parser.add_argument(
        "--min-match-iou",
        type=float,
        default=0.1,
        help="Discard implausible class-aware matches below this IoU",
    )
    parser.add_argument(
        "--bad-iou",
        type=float,
        default=0.5,
        help="A matched localization below this IoU is an error for AUPR-error",
    )
    parser.add_argument(
        "--uncertainty-score",
        choices=("mean_std", "max_std", "logdet"),
        default="mean_std",
    )
    parser.add_argument(
        "--label-mode",
        choices=("auto", "contiguous", "category_id"),
        default="auto",
        help="How COCO category IDs correspond to model class indices",
    )
    parser.add_argument(
        "--allow-partial-checkpoint",
        action="store_true",
        help="Permit missing/unexpected checkpoint keys (normally undesirable)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    return parser.parse_args()


def load_coco(
    path: Path, contiguous_labels: bool
) -> tuple[list[CocoImage], dict[int, str]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    categories = sorted(data.get("categories", []), key=lambda item: item["id"])
    category_ids = [int(item["id"]) for item in categories]
    if contiguous_labels:
        category_to_label = {
            category_id: i for i, category_id in enumerate(category_ids)
        }
        label_names = {
            i: str(item.get("name", item["id"])) for i, item in enumerate(categories)
        }
    else:
        category_to_label = {category_id: category_id for category_id in category_ids}
        label_names = {
            int(item["id"]): str(item.get("name", item["id"])) for item in categories
        }

    annotations: dict[int, list[Mapping[str, Any]]] = {}
    for annotation in data.get("annotations", []):
        if annotation.get("iscrowd", 0):
            continue
        annotations.setdefault(int(annotation["image_id"]), []).append(annotation)

    images: list[CocoImage] = []
    for item in data.get("images", []):
        image_id = int(item["id"])
        width = int(item["width"])
        height = int(item["height"])
        boxes: list[list[float]] = []
        labels: list[int] = []
        for annotation in annotations.get(image_id, []):
            category_id = int(annotation["category_id"])
            if category_id not in category_to_label:
                continue
            x, y, w, h = (float(value) for value in annotation["bbox"])
            if w <= 0.0 or h <= 0.0:
                continue
            boxes.append(
                [(x + 0.5 * w) / width, (y + 0.5 * h) / height, w / width, h / height]
            )
            labels.append(category_to_label[category_id])
        images.append(
            CocoImage(
                image_id=image_id,
                file_name=str(item["file_name"]),
                width=width,
                height=height,
                boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                labels=np.asarray(labels, dtype=np.int64),
            )
        )
    return images, label_names


def resolve_repo_root(path: Path) -> Path:
    path = path.resolve()
    if (path / "src" / "core").is_dir():
        return path
    nested = path / "rtdetrv2_pytorch"
    if (nested / "src" / "core").is_dir():
        return nested
    raise FileNotFoundError(
        f"Could not find rtdetrv2_pytorch/src/core below --repo-root {path}"
    )


def resolve_from_repo(path: Path, repo_root: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute() or path.exists():
        return path.resolve()
    return (repo_root / path).resolve()


def unwrap_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    candidate = checkpoint
    if isinstance(candidate, Mapping) and "ema" in candidate:
        ema = candidate["ema"]
        if isinstance(ema, Mapping) and "module" in ema:
            candidate = ema["module"]
        elif isinstance(ema, Mapping):
            candidate = ema
    if isinstance(candidate, Mapping) and "model" in candidate:
        candidate = candidate["model"]
    if isinstance(candidate, Mapping) and "state_dict" in candidate:
        candidate = candidate["state_dict"]
    if not isinstance(candidate, Mapping):
        raise TypeError("Checkpoint does not contain a recognizable state dictionary")

    state_dict = dict(candidate)
    if state_dict and all(str(key).startswith("module.") for key in state_dict):
        state_dict = {str(key)[7:]: value for key, value in state_dict.items()}
    return state_dict


def build_model(
    args: argparse.Namespace, repo_root: Path
) -> tuple[torch.nn.Module, Any]:
    sys.path.insert(0, str(repo_root))
    from src.core import YAMLConfig

    config_path = resolve_from_repo(args.config, repo_root)
    checkpoint_path = resolve_from_repo(args.checkpoint, repo_root)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration does not exist: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    cfg = YAMLConfig(str(config_path), resume=str(checkpoint_path))
    model = cfg.model

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = unwrap_state_dict(checkpoint)
    incompatible = model.load_state_dict(
        state_dict, strict=not args.allow_partial_checkpoint
    )
    if incompatible.missing_keys:
        warnings.warn(
            "Missing checkpoint keys (first 20): "
            + ", ".join(incompatible.missing_keys[:20])
        )
    if incompatible.unexpected_keys:
        warnings.warn(
            "Unexpected checkpoint keys (first 20): "
            + ", ".join(incompatible.unexpected_keys[:20])
        )
    # Matches references/deploy/rtdetrv2_torch.py in the official repository.
    model = model.deploy().to(args.device).eval()
    return model, cfg


def preprocess(image: Image.Image, size: Sequence[int]) -> torch.Tensor:
    height, width = int(size[0]), int(size[1])
    resized = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    # Official deployment preprocessing: Resize + ToTensor, without ImageNet normalization.
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def final_tensor(value: Any, name: str) -> torch.Tensor:
    if isinstance(value, (tuple, list)):
        value = value[-1]
    if not torch.is_tensor(value):
        raise TypeError(f"Model output {name!r} is not a tensor")
    # Accommodate a full decoder stack [layers, batch, queries, channels].
    if value.ndim == 4:
        value = value[-1]
    if value.ndim != 3:
        raise ValueError(
            f"Expected {name} to have shape [B, Q, D], got {tuple(value.shape)}"
        )
    return value


def uncertainty_scalar(logvars: np.ndarray, method: str) -> np.ndarray:
    stds = np.exp(0.5 * np.clip(logvars, LOGVAR_MIN, LOGVAR_MAX))
    if method == "mean_std":
        return stds.mean(axis=-1)
    if method == "max_std":
        return stds.max(axis=-1)
    if method == "logdet":
        return np.clip(logvars, LOGVAR_MIN, LOGVAR_MAX).sum(axis=-1)
    raise ValueError(method)


@torch.inference_mode()
def predict(
    model: torch.nn.Module, tensor: torch.Tensor, args: argparse.Namespace
) -> Prediction:
    output = model(tensor.unsqueeze(0).to(args.device))
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, Mapping):
        raise TypeError("Model must return a dictionary-like object")

    logvar_key = next(
        (key for key in ("pred_box_logvars", "pred_boxes_logvars") if key in output),
        None,
    )
    if logvar_key is None:
        raise KeyError("Model output needs 'pred_box_logvars' or 'pred_boxes_logvars'")

    logits = final_tensor(output["pred_logits"], "pred_logits")[0]
    boxes = final_tensor(output["pred_boxes"], "pred_boxes")[0]
    logvars = final_tensor(output[logvar_key], logvar_key)[0]
    if boxes.shape != logvars.shape:
        raise ValueError(
            f"Box/log-variance shapes differ: {tuple(boxes.shape)} vs {tuple(logvars.shape)}"
        )

    # Reproduce RTDETRPostProcessor's focal-loss branch: top-k over the
    # flattened [query, class] sigmoid probabilities, then recover the query.
    probabilities = logits.sigmoid()
    top_k = min(args.top_k, probabilities.numel())
    scores, flat_indices = torch.topk(probabilities.flatten(), top_k)
    labels = flat_indices % probabilities.shape[-1]
    query_indices = torch.div(
        flat_indices, probabilities.shape[-1], rounding_mode="floor"
    )
    keep = scores >= args.score_threshold
    scores = scores[keep]
    labels = labels[keep]
    query_indices = query_indices[keep]

    boxes_np = boxes[query_indices].float().cpu().numpy()
    labels_np = labels.cpu().numpy().astype(np.int64)
    scores_np = scores.float().cpu().numpy()
    logvars_np = logvars[query_indices].float().cpu().numpy()
    return Prediction(
        boxes=boxes_np,
        labels=labels_np,
        scores=scores_np,
        logvars=logvars_np,
        uncertainty=uncertainty_scalar(logvars_np, args.uncertainty_score),
        query_indices=query_indices.cpu().numpy().astype(np.int64),
    )


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    result = np.empty_like(boxes, dtype=np.float64)
    result[..., 0] = boxes[..., 0] - 0.5 * boxes[..., 2]
    result[..., 1] = boxes[..., 1] - 0.5 * boxes[..., 3]
    result[..., 2] = boxes[..., 0] + 0.5 * boxes[..., 2]
    result[..., 3] = boxes[..., 1] + 0.5 * boxes[..., 3]
    return result


def pairwise_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float64)
    a = cxcywh_to_xyxy(boxes1)
    b = cxcywh_to_xyxy(boxes2)
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(bottom_right - top_left, 0.0)
    intersection = wh[..., 0] * wh[..., 1]
    area_a = np.maximum(a[:, 2] - a[:, 0], 0.0) * np.maximum(a[:, 3] - a[:, 1], 0.0)
    area_b = np.maximum(b[:, 2] - b[:, 0], 0.0) * np.maximum(b[:, 3] - b[:, 1], 0.0)
    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / np.maximum(union, EPS)


def match_predictions(
    sample: CocoImage,
    prediction: Prediction,
    min_iou: float,
) -> tuple[list[Match], int, int]:
    if len(prediction.boxes) == 0 or len(sample.boxes) == 0:
        return [], len(prediction.boxes), len(sample.boxes)

    ious = pairwise_iou(prediction.boxes, sample.boxes)
    class_ok = prediction.labels[:, None] == sample.labels[None, :]
    # The large penalty prevents a different-class match whenever a same-class
    # assignment exists. Remaining mismatches are removed after assignment.
    cost = 1.0 - ious + (~class_ok) * 1_000.0
    pred_indices, gt_indices = linear_sum_assignment(cost)

    matches: list[Match] = []
    for pred_index, gt_index in zip(pred_indices, gt_indices):
        if not class_ok[pred_index, gt_index] or ious[pred_index, gt_index] < min_iou:
            continue
        matches.append(
            Match(
                image_id=sample.image_id,
                file_name=sample.file_name,
                pred_index=int(pred_index),
                query_index=int(prediction.query_indices[pred_index]),
                gt_index=int(gt_index),
                label=int(prediction.labels[pred_index]),
                score=float(prediction.scores[pred_index]),
                iou=float(ious[pred_index, gt_index]),
                box=prediction.boxes[pred_index].astype(np.float64),
                gt_box=sample.boxes[gt_index].astype(np.float64),
                logvar=prediction.logvars[pred_index].astype(np.float64),
                uncertainty=float(prediction.uncertainty[pred_index]),
            )
        )
    return (
        matches,
        len(prediction.boxes) - len(matches),
        len(sample.boxes) - len(matches),
    )


def average_precision_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels):
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / positives)


def risk_coverage(
    uncertainty: np.ndarray, risks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(uncertainty, kind="stable")
    sorted_risks = risks[order]
    coverage = np.arange(1, len(risks) + 1, dtype=np.float64) / len(risks)
    selective_risk = np.cumsum(sorted_risks) / np.arange(1, len(risks) + 1)
    return coverage, selective_risk


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def quantile_calibration(
    variances: np.ndarray,
    errors: np.ndarray,
    uncertainty: np.ndarray,
    bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(uncertainty) == 0:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty
    groups = np.array_split(
        np.argsort(uncertainty, kind="stable"), min(bins, len(uncertainty))
    )
    predicted, empirical, counts = [], [], []
    for group in groups:
        predicted.append(float(np.sqrt(np.mean(variances[group]))))
        empirical.append(float(np.sqrt(np.mean(np.square(errors[group])))))
        counts.append(len(group))
    return np.asarray(predicted), np.asarray(empirical), np.asarray(counts)


def compute_metrics(
    matches: Sequence[Match],
    false_positives: int,
    false_negatives: int,
    bad_iou: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not matches:
        metrics = {
            "matched_boxes": 0,
            "false_positives_at_score_threshold": false_positives,
            "missed_ground_truth_boxes": false_negatives,
            "warning": "No class-aware matches passed the minimum IoU; calibration metrics are unavailable.",
        }
        return metrics, {}

    boxes = np.stack([match.box for match in matches])
    gt_boxes = np.stack([match.gt_box for match in matches])
    logvars = np.clip(
        np.stack([match.logvar for match in matches]), LOGVAR_MIN, LOGVAR_MAX
    )
    variances = np.exp(logvars)
    stds = np.sqrt(variances)
    errors = boxes - gt_boxes
    ious = np.asarray([match.iou for match in matches], dtype=np.float64)
    scores = np.asarray([match.score for match in matches], dtype=np.float64)
    uncertainty = np.asarray([match.uncertainty for match in matches], dtype=np.float64)
    confidence_baseline = 1.0 - scores
    localization_risk = 1.0 - ious

    element_nll = 0.5 * (
        np.square(errors) / variances + logvars + math.log(2.0 * math.pi)
    )
    coverage_metrics: dict[str, dict[str, float]] = {}
    for nominal, z_score in Z_SCORES.items():
        covered = np.abs(errors) <= z_score * stds
        coverage_metrics[nominal] = {
            "coordinate_coverage": float(covered.mean()),
            "whole_box_coverage": float(covered.all(axis=1).mean()),
        }

    if len(matches) >= 2 and np.std(uncertainty) > 0.0:
        correlation = float(spearmanr(uncertainty, localization_risk).statistic)
    else:
        correlation = float("nan")

    bad_localization = ious < bad_iou
    aupr_error = average_precision_binary(bad_localization, uncertainty)
    rc_coverage, rc_risk = risk_coverage(uncertainty, localization_risk)
    baseline_coverage, baseline_risk = risk_coverage(
        confidence_baseline, localization_risk
    )
    oracle_coverage, oracle_risk = risk_coverage(localization_risk, localization_risk)
    baseline_aupr_error = average_precision_binary(
        bad_localization, confidence_baseline
    )

    risk_at_coverage: dict[str, float] = {}
    for requested in (0.5, 0.8, 0.9, 1.0):
        index = max(0, int(math.ceil(requested * len(matches))) - 1)
        risk_at_coverage[f"{int(requested * 100)}%"] = float(rc_risk[index])

    predicted_rms, empirical_rmse, bin_counts = quantile_calibration(
        variances, errors, uncertainty
    )
    metrics = {
        "matched_boxes": len(matches),
        "false_positives_at_score_threshold": false_positives,
        "missed_ground_truth_boxes": false_negatives,
        "mean_iou_of_matches": float(ious.mean()),
        "gaussian_nll_mean_per_coordinate": float(element_nll.mean()),
        "rmse_normalized_cxcywh": dict(
            zip(
                ("cx", "cy", "w", "h"),
                np.sqrt(np.mean(np.square(errors), axis=0)).tolist(),
            )
        ),
        "mean_predicted_std_normalized_cxcywh": dict(
            zip(("cx", "cy", "w", "h"), stds.mean(axis=0).tolist())
        ),
        "interval_coverage": coverage_metrics,
        "spearman_uncertainty_vs_one_minus_iou": finite_or_none(correlation),
        "aupr_error_iou_below_threshold": finite_or_none(aupr_error),
        "aupr_error_iou_threshold": bad_iou,
        "aurc_one_minus_iou": float(rc_risk.mean()),
        "confidence_baseline": {
            "uncertainty_definition": "1 - class score",
            "aupr_error_iou_below_threshold": finite_or_none(baseline_aupr_error),
            "aurc_one_minus_iou": float(baseline_risk.mean()),
        },
        "aurc_references": {
            "oracle": float(oracle_risk.mean()),
            "random_expected": float(localization_risk.mean()),
        },
        "aurc_improvement_over_one_minus_score": float(
            baseline_risk.mean() - rc_risk.mean()
        ),
        "selective_risk_at_coverage": risk_at_coverage,
        "notes": [
            "Higher Spearman correlation and AUPR-error are better; lower NLL and AURC are better.",
            "A positive AURC improvement means the probabilistic head ranks localization errors better than 1 - class score.",
            "Coverage should be near its nominal level, while smaller predicted standard deviations are sharper.",
            "Matched-box metrics evaluate localization uncertainty only; inspect false positives and misses separately.",
        ],
    }
    arrays = {
        "errors": errors,
        "variances": variances,
        "ious": ious,
        "uncertainty": uncertainty,
        "bad_localization": bad_localization,
        "rc_coverage": rc_coverage,
        "rc_risk": rc_risk,
        "baseline_coverage": baseline_coverage,
        "baseline_risk": baseline_risk,
        "oracle_coverage": oracle_coverage,
        "oracle_risk": oracle_risk,
        "calibration_predicted": predicted_rms,
        "calibration_empirical": empirical_rmse,
        "calibration_counts": bin_counts,
    }
    return metrics, arrays


def uncertainty_envelope(
    box: np.ndarray, logvar: np.ndarray, z_score: float = Z_SCORES["95"]
) -> np.ndarray:
    """Conservative outer envelope for independent Gaussian cxcywh variables."""
    std = np.exp(0.5 * np.clip(logvar, LOGVAR_MIN, LOGVAR_MAX))
    cx_low, cy_low = box[:2] - z_score * std[:2]
    cx_high, cy_high = box[:2] + z_score * std[:2]
    width_high = max(0.0, box[2] + z_score * std[2])
    height_high = max(0.0, box[3] + z_score * std[3])
    return np.clip(
        [
            cx_low - 0.5 * width_high,
            cy_low - 0.5 * height_high,
            cx_high + 0.5 * width_high,
            cy_high + 0.5 * height_high,
        ],
        0.0,
        1.0,
    )


def add_xyxy_patch(
    axis: plt.Axes, xyxy: np.ndarray, width: int, height: int, **kwargs: Any
) -> None:
    x1, y1, x2, y2 = xyxy * np.asarray([width, height, width, height])
    axis.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, **kwargs))


def plot_image(
    image: Image.Image,
    sample: CocoImage,
    prediction: Prediction,
    label_names: Mapping[int, str],
    output_path: Path,
    uncertainty_range: tuple[float, float],
) -> None:
    figure, axis = plt.subplots(figsize=(12, 8))
    axis.imshow(image)
    axis.set_axis_off()

    for box in sample.boxes:
        add_xyxy_patch(
            axis,
            cxcywh_to_xyxy(box[None])[0],
            sample.width,
            sample.height,
            edgecolor="#36d06c",
            linewidth=1.5,
        )

    low, high = uncertainty_range
    scale = max(high - low, EPS)
    cmap = plt.get_cmap("plasma")
    for box, label, score, logvar, uncertainty in zip(
        prediction.boxes,
        prediction.labels,
        prediction.scores,
        prediction.logvars,
        prediction.uncertainty,
    ):
        color = cmap(float(np.clip((uncertainty - low) / scale, 0.0, 1.0)))
        mean_xyxy = np.clip(cxcywh_to_xyxy(box[None])[0], 0.0, 1.0)
        envelope_xyxy = uncertainty_envelope(box, logvar)
        add_xyxy_patch(
            axis,
            envelope_xyxy,
            sample.width,
            sample.height,
            edgecolor=color,
            linewidth=1.0,
            linestyle="--",
            alpha=0.8,
        )
        add_xyxy_patch(
            axis,
            mean_xyxy,
            sample.width,
            sample.height,
            edgecolor=color,
            linewidth=2.0,
        )
        x1, y1 = mean_xyxy[:2] * np.asarray([sample.width, sample.height])
        name = label_names.get(int(label), str(int(label)))
        axis.text(
            x1,
            max(0.0, y1 - 3.0),
            f"{name} {score:.2f}  u={uncertainty:.3g}",
            fontsize=7,
            color="white",
            bbox={"facecolor": color, "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
        )
    axis.set_title("GT: green | prediction: solid | approximate 95% envelope: dashed")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_summary(
    metrics: Mapping[str, Any], arrays: Mapping[str, np.ndarray], output_path: Path
) -> None:
    if not arrays:
        return
    uncertainty = arrays["uncertainty"]
    risks = 1.0 - arrays["ious"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].scatter(uncertainty, risks, s=13, alpha=0.55)
    axes[0, 0].set(
        xlabel="Predicted uncertainty",
        ylabel="Localization risk (1 - IoU)",
        title="Uncertainty ranking",
    )

    predicted = arrays["calibration_predicted"]
    empirical = arrays["calibration_empirical"]
    axes[0, 1].scatter(predicted, empirical, s=35)
    if len(predicted):
        limit = max(float(predicted.max()), float(empirical.max()), EPS)
        axes[0, 1].plot([0.0, limit], [0.0, limit], "k--", linewidth=1)
    axes[0, 1].set(
        xlabel="Predicted RMS std",
        ylabel="Empirical RMSE",
        title="Quantile-bin calibration",
    )

    axes[1, 0].plot(
        arrays["rc_coverage"], arrays["rc_risk"], linewidth=2, label="Box uncertainty"
    )
    axes[1, 0].plot(
        arrays["baseline_coverage"],
        arrays["baseline_risk"],
        linewidth=1.5,
        label="1 - class score",
    )
    axes[1, 0].plot(
        arrays["oracle_coverage"],
        arrays["oracle_risk"],
        "k--",
        linewidth=1,
        label="Oracle",
    )
    axes[1, 0].set(
        xlabel="Coverage retained (least uncertain first)",
        ylabel="Mean 1 - IoU",
        title="Risk–coverage curve",
        xlim=(0.0, 1.0),
    )
    axes[1, 0].legend(fontsize=8)

    nominal = np.asarray([0.5, 0.8, 0.9, 0.95])
    observed = np.asarray(
        [
            metrics["interval_coverage"][str(int(level * 100))]["coordinate_coverage"]
            for level in nominal
        ]
    )
    positions = np.arange(len(nominal))
    axes[1, 1].bar(positions - 0.18, nominal, width=0.36, label="Nominal")
    axes[1, 1].bar(positions + 0.18, observed, width=0.36, label="Observed")
    axes[1, 1].set_xticks(positions, [f"{int(level * 100)}%" for level in nominal])
    axes[1, 1].set(
        ylim=(0.0, 1.0),
        ylabel="Coordinate coverage",
        title="Gaussian interval coverage",
    )
    axes[1, 1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_matches(path: Path, matches: Sequence[Match]) -> None:
    coordinate_names = ("cx", "cy", "w", "h")
    fieldnames = [
        "image_id",
        "file_name",
        "pred_index",
        "query_index",
        "gt_index",
        "label",
        "score",
        "iou",
        "uncertainty",
    ]
    fieldnames += [f"pred_{name}" for name in coordinate_names]
    fieldnames += [f"gt_{name}" for name in coordinate_names]
    fieldnames += [f"logvar_{name}" for name in coordinate_names]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for match in matches:
            row: dict[str, Any] = {
                "image_id": match.image_id,
                "file_name": match.file_name,
                "pred_index": match.pred_index,
                "query_index": match.query_index,
                "gt_index": match.gt_index,
                "label": match.label,
                "score": match.score,
                "iou": match.iou,
                "uncertainty": match.uncertainty,
            }
            row.update(
                {
                    f"pred_{name}": value
                    for name, value in zip(coordinate_names, match.box)
                }
            )
            row.update(
                {
                    f"gt_{name}": value
                    for name, value in zip(coordinate_names, match.gt_box)
                }
            )
            row.update(
                {
                    f"logvar_{name}": value
                    for name, value in zip(coordinate_names, match.logvar)
                }
            )
            writer.writerow(row)


def validation_data_paths(
    args: argparse.Namespace, cfg: Any, repo_root: Path
) -> tuple[Path, Path]:
    yaml_cfg = getattr(cfg, "yaml_cfg", {})
    val_dataset = yaml_cfg.get("val_dataloader", {}).get("dataset", {})
    image_value = args.images or val_dataset.get("img_folder")
    annotation_value = args.annotations or val_dataset.get("ann_file")
    if image_value is None or annotation_value is None:
        raise ValueError(
            "Validation paths were not found in the YAML. Pass --images and --annotations."
        )
    images = resolve_from_repo(Path(image_value), repo_root)
    annotations = resolve_from_repo(Path(annotation_value), repo_root)
    if not images.is_dir():
        raise FileNotFoundError(f"Validation image directory does not exist: {images}")
    if not annotations.is_file():
        raise FileNotFoundError(f"COCO annotations do not exist: {annotations}")
    return images, annotations


def uses_contiguous_labels(args: argparse.Namespace, cfg: Any) -> bool:
    if args.label_mode != "auto":
        return args.label_mode == "contiguous"
    yaml_cfg = getattr(cfg, "yaml_cfg", {})
    val_dataset = yaml_cfg.get("val_dataloader", {}).get("dataset", {})
    return bool(
        val_dataset.get(
            "remap_mscoco_category", yaml_cfg.get("remap_mscoco_category", False)
        )
    )


def main(args: argparse.Namespace) -> None:
    repo_root = resolve_repo_root(args.repo_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_dir / "overlays"
    plot_dir.mkdir(exist_ok=True)

    model, cfg = build_model(args, repo_root)
    if args.input_size is None:
        args.input_size = tuple(
            getattr(cfg, "yaml_cfg", {}).get("eval_spatial_size", (640, 640))
        )
    if args.top_k is None:
        args.top_k = int(getattr(cfg.postprocessor, "num_top_queries", 300))
    images_dir, annotations_path = validation_data_paths(args, cfg, repo_root)
    contiguous_labels = uses_contiguous_labels(args, cfg)
    samples, label_names = load_coco(annotations_path, contiguous_labels)
    if args.shuffle:
        np.random.default_rng(args.seed).shuffle(samples)
    if args.num_images is not None:
        samples = samples[: args.num_images]

    all_matches: list[Match] = []
    false_positives = 0
    false_negatives = 0
    plot_records: list[tuple[Image.Image, CocoImage, Prediction]] = []
    all_uncertainties: list[np.ndarray] = []

    for index, sample in enumerate(samples, start=1):
        image_path = images_dir / sample.file_name
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        tensor = preprocess(image, args.input_size)
        prediction = predict(model, tensor, args)
        matches, image_fp, image_fn = match_predictions(
            sample, prediction, args.min_match_iou
        )
        all_matches.extend(matches)
        false_positives += image_fp
        false_negatives += image_fn
        if len(prediction.uncertainty):
            all_uncertainties.append(prediction.uncertainty)
        if len(plot_records) < args.plot_limit:
            plot_records.append((image, sample, prediction))
        if index % 50 == 0 or index == len(samples):
            print(f"Processed {index}/{len(samples)} images", flush=True)

    metrics, arrays = compute_metrics(
        all_matches, false_positives, false_negatives, args.bad_iou
    )
    metrics["configuration"] = {
        "images_evaluated": len(samples),
        "score_threshold": args.score_threshold,
        "minimum_match_iou": args.min_match_iou,
        "bad_iou": args.bad_iou,
        "uncertainty_score": args.uncertainty_score,
        "label_mode": "contiguous" if contiguous_labels else "category_id",
        "input_size": list(args.input_size),
        "top_k": args.top_k,
        "config": str(resolve_from_repo(args.config, repo_root)),
        "checkpoint": str(resolve_from_repo(args.checkpoint, repo_root)),
        "logvar_clamp": [LOGVAR_MIN, LOGVAR_MAX],
    }

    if all_uncertainties:
        combined = np.concatenate(all_uncertainties)
        uncertainty_range = tuple(np.quantile(combined, [0.05, 0.95]).tolist())
    else:
        uncertainty_range = (0.0, 1.0)
    for image, sample, prediction in plot_records:
        output_name = f"{sample.image_id}_{Path(sample.file_name).stem}.png"
        plot_image(
            image,
            sample,
            prediction,
            label_names,
            plot_dir / output_name,
            uncertainty_range,
        )

    write_matches(args.output_dir / "matched_detections.csv", all_matches)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=False)
    plot_summary(metrics, arrays, args.output_dir / "uncertainty_quality.png")

    print(json.dumps(metrics, indent=2, allow_nan=False))
    print(f"\nSaved results to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
