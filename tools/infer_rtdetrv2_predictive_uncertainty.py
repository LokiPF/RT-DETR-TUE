#!/usr/bin/env python3
"""Visualize the learned RT-DETRv2 predictive bbox uncertainty head.

The model is expected to return:

    pred_logits:   [B, Q, C]
    pred_boxes:    [B, Q, 4] normalized cxcywh means
    pred_bbox_std: [B, Q, 4] normalized xyxy standard deviations

For every retained detection, the image contains:

  * a solid rectangle for the predicted mean box;
  * a translucent dashed rectangle for the outer envelope of the nominal
    Gaussian coordinate intervals;
  * confidence, absolute total scale, and size-relative scale.

The outer rectangle is only a visualization. The complete four coordinate
intervals are written with ``--save-json``.
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor, nn
from torchvision.transforms.functional import to_tensor


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.core import YAMLConfig
except ImportError as exc:
    raise ImportError(
        "Put this script in `rtdetrv2_pytorch/tools/` and run it from "
        "the repository root."
    ) from exc


IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

COCO_CLASS_NAMES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run RT-DETRv2 inference and visualize the learned Gaussian bbox "
            "uncertainty head."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True)
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more image files or directories.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/tu/learned_visualizations",
    )
    parser.add_argument("--recursive", action="store_true")
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
        "--tu-prototypes",
        default=None,
        help=(
            "Optional prototype artifact. Required if the learned head uses "
            "TU and tu_prototype_path is not configured."
        ),
    )
    parser.add_argument(
        "--tu-topk",
        type=int,
        default=None,
        help="Override the number of queries for which TU is calculated.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.95,
        help="Nominal Gaussian interval coverage.",
    )
    parser.add_argument(
        "--coverage-kind",
        choices=("coordinate", "joint_independent"),
        default="coordinate",
        help=(
            "`coordinate` uses the requested coverage for every coordinate. "
            "`joint_independent` adjusts the coordinate quantile so all four "
            "coordinates jointly have the requested coverage under an "
            "independence assumption."
        ),
    )
    parser.add_argument(
        "--std-temperature",
        type=float,
        default=1.0,
        help="Optional positive post-hoc multiplier for predicted std.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
    )
    parser.add_argument(
        "--categories-json",
        default=None,
        help="Optional COCO annotation JSON used for category names.",
    )
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--line-width", type=int, default=3)
    parser.add_argument("--region-alpha", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("--confidence-threshold must be in [0, 1].")
    if args.max_detections < 1:
        raise ValueError("--max-detections must be positive.")
    if not 0.0 < args.coverage < 1.0:
        raise ValueError("--coverage must lie in (0, 1).")
    if not math.isfinite(args.std_temperature) or args.std_temperature <= 0:
        raise ValueError("--std-temperature must be finite and positive.")
    if args.tu_topk is not None and args.tu_topk < 1:
        raise ValueError("--tu-topk must be positive.")
    if args.input_size is not None and min(args.input_size) < 1:
        raise ValueError("--input-size dimensions must be positive.")
    if args.line_width < 1:
        raise ValueError("--line-width must be positive.")
    if not 0 <= args.region_alpha <= 255:
        raise ValueError("--region-alpha must be in [0, 255].")


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


def collect_image_paths(
    inputs: Sequence[str],
    recursive: bool,
) -> List[Path]:
    found: Dict[Path, None] = {}
    for raw_input in inputs:
        path = Path(raw_input).expanduser().resolve()
        if path.is_file():
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"Unsupported image extension: {path}")
            found[path] = None
        elif path.is_dir():
            iterator: Iterable[Path] = (
                path.rglob("*") if recursive else path.glob("*")
            )
            for candidate in iterator:
                if (
                    candidate.is_file()
                    and candidate.suffix.lower() in IMAGE_SUFFIXES
                ):
                    found[candidate.resolve()] = None
        else:
            raise FileNotFoundError(f"Input does not exist: {path}")
    paths = sorted(found, key=lambda item: str(item))
    if not paths:
        raise RuntimeError("No supported input images were found.")
    return paths


def output_path_for(
    image_path: Path,
    output_dir: Path,
    used_names: Dict[str, Path],
) -> Path:
    name = f"{image_path.stem}_predictive_uncertainty.png"
    if name in used_names and used_names[name] != image_path:
        suffix = hashlib.sha256(
            str(image_path).encode("utf-8")
        ).hexdigest()[:8]
        name = (
            f"{image_path.stem}_{suffix}_predictive_uncertainty.png"
        )
    used_names[name] = image_path
    return output_dir / name


def load_class_names(
    categories_path: Optional[str],
    num_classes: int,
) -> List[str]:
    if categories_path is None:
        if num_classes == len(COCO_CLASS_NAMES):
            return list(COCO_CLASS_NAMES)
        return [f"class_{index}" for index in range(num_classes)]

    path = Path(categories_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    categories = data.get("categories") if isinstance(data, Mapping) else None
    if not isinstance(categories, list):
        raise ValueError("Categories JSON has no `categories` list.")
    ordered = sorted(categories, key=lambda category: category["id"])
    names = [str(category["name"]) for category in ordered]
    if len(names) != num_classes:
        raise ValueError(
            f"Categories contain {len(names)} classes; model uses {num_classes}."
        )
    return names


def infer_input_size(
    requested_size: Optional[Sequence[int]],
    uncertainty_module: nn.Module,
) -> Tuple[int, int]:
    if requested_size is not None:
        return int(requested_size[0]), int(requested_size[1])
    spatial_size = getattr(
        uncertainty_module,
        "eval_spatial_size",
        None,
    )
    if (
        isinstance(spatial_size, (list, tuple))
        and len(spatial_size) == 2
        and min(int(value) for value in spatial_size) > 0
    ):
        return int(spatial_size[0]), int(spatial_size[1])
    print("[input] eval_spatial_size unavailable; using 640x640")
    return 640, 640


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


def gaussian_z_value(
    coverage: float,
    coverage_kind: str,
) -> Tuple[float, float]:
    if coverage_kind == "coordinate":
        coordinate_coverage = coverage
    elif coverage_kind == "joint_independent":
        coordinate_coverage = coverage ** 0.25
    else:
        raise ValueError(f"Unknown coverage kind: {coverage_kind}")
    probability = 0.5 * (1.0 + coordinate_coverage)
    z_value = statistics.NormalDist().inv_cdf(probability)
    return float(z_value), float(coordinate_coverage)


def normalized_to_pixels(
    values: Tensor,
    image_width: int,
    image_height: int,
) -> Tensor:
    scale = values.new_tensor(
        (image_width, image_height, image_width, image_height)
    )
    return values * scale


def class_color(class_id: int) -> Tuple[int, int, int]:
    hue = (class_id * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 0.98)
    return int(red * 255), int(green * 255), int(blue * 255)


def clamp_rectangle(
    box: Sequence[float],
    image_width: int,
    image_height: int,
) -> Tuple[float, float, float, float]:
    x1 = min(max(float(box[0]), 0.0), max(image_width - 1, 0))
    y1 = min(max(float(box[1]), 0.0), max(image_height - 1, 0))
    x2 = min(max(float(box[2]), 0.0), max(image_width - 1, 0))
    y2 = min(max(float(box[3]), 0.0), max(image_height - 1, 0))
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def draw_dashed_segment(
    draw: ImageDraw.ImageDraw,
    start: Tuple[float, float],
    end: Tuple[float, float],
    fill: Tuple[int, int, int, int],
    width: int,
    dash: int = 10,
    gap: int = 6,
) -> None:
    x1, y1 = start
    x2, y2 = end
    delta_x = x2 - x1
    delta_y = y2 - y1
    length = math.hypot(delta_x, delta_y)
    if length <= 0:
        return
    unit_x = delta_x / length
    unit_y = delta_y / length
    position = 0.0
    while position < length:
        end_position = min(position + dash, length)
        draw.line(
            (
                x1 + unit_x * position,
                y1 + unit_y * position,
                x1 + unit_x * end_position,
                y1 + unit_y * end_position,
            ),
            fill=fill,
            width=width,
        )
        position += dash + gap


def draw_dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: Tuple[float, float, float, float],
    fill: Tuple[int, int, int, int],
    width: int,
) -> None:
    x1, y1, x2, y2 = box
    draw_dashed_segment(draw, (x1, y1), (x2, y1), fill, width)
    draw_dashed_segment(draw, (x2, y1), (x2, y2), fill, width)
    draw_dashed_segment(draw, (x2, y2), (x1, y2), fill, width)
    draw_dashed_segment(draw, (x1, y2), (x1, y1), fill, width)


def load_font(image_width: int) -> ImageFont.ImageFont:
    font_size = max(12, min(24, round(image_width / 70)))
    for name in (
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_label(
    draw: ImageDraw.ImageDraw,
    position: Tuple[float, float],
    text: str,
    color: Tuple[int, int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x, y = position
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    padding = 3
    if top - padding < 0:
        y += (bottom - top) + 2 * padding
        left, top, right, bottom = draw.textbbox(
            (x, y),
            text,
            font=font,
        )
    draw.rectangle(
        (
            left - padding,
            top - padding,
            right + padding,
            bottom + padding,
        ),
        fill=(0, 0, 0, 190),
    )
    draw.text((x, y), text, fill=color, font=font)


def draw_visualization(
    image: Image.Image,
    detections: Sequence[Mapping[str, Any]],
    coverage: float,
    coverage_kind: str,
    line_width: int,
    region_alpha: int,
) -> Image.Image:
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay, "RGBA")

    for detection in detections:
        color = tuple(detection["color"])
        region = clamp_rectangle(
            detection["outer_xyxy_pixels"],
            canvas.width,
            canvas.height,
        )
        overlay_draw.rectangle(
            region,
            fill=(*color, region_alpha),
        )
        draw_dashed_rectangle(
            overlay_draw,
            region,
            (*color, 235),
            max(1, line_width - 1),
        )

    canvas = Image.alpha_composite(canvas, overlay)
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = load_font(canvas.width)

    for detection in detections:
        color = tuple(detection["color"])
        predicted_box = clamp_rectangle(
            detection["bbox_xyxy_pixels"],
            canvas.width,
            canvas.height,
        )
        draw.rectangle(
            predicted_box,
            outline=(*color, 255),
            width=line_width,
        )
        label = (
            f"{detection['class_name']} {detection['confidence']:.2f} | "
            f"u {detection['total_std']:.3g} | "
            f"u_rel {detection['relative_uncertainty']:.3g}"
        )
        draw_label(
            draw,
            (predicted_box[0] + 2, predicted_box[1] - 2),
            label,
            (*color, 255),
            font,
        )

    legend = (
        "solid: mean bbox | dashed/shaded: nominal "
        f"{coverage:.0%} {coverage_kind.replace('_', ' ')} Gaussian region"
    )
    draw_label(
        draw,
        (8, 8),
        legend,
        (255, 255, 255, 255),
        font,
    )
    return canvas.convert("RGB")


def prepare_image(
    image: Image.Image,
    input_size: Tuple[int, int],
    device: torch.device,
) -> Tensor:
    input_height, input_width = input_size
    resampling = getattr(Image, "Resampling", Image)
    resized = image.convert("RGB").resize(
        (input_width, input_height),
        resample=resampling.BILINEAR,
    )
    return to_tensor(resized).unsqueeze(0).to(
        device,
        non_blocking=True,
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def relative_uncertainty(
    predicted_xyxy: Tensor,
    predicted_std: Tensor,
) -> Tensor:
    width = (
        predicted_xyxy[:, 2] - predicted_xyxy[:, 0]
    ).abs().clamp_min(1e-6)
    height = (
        predicted_xyxy[:, 3] - predicted_xyxy[:, 1]
    ).abs().clamp_min(1e-6)
    horizontal = (
        predicted_std[:, 0].square()
        + predicted_std[:, 2].square()
    ) / width.square()
    vertical = (
        predicted_std[:, 1].square()
        + predicted_std[:, 3].square()
    ) / height.square()
    return torch.sqrt(horizontal + vertical)


@torch.inference_mode()
def infer_one(
    model: nn.Module,
    image: Image.Image,
    input_size: Tuple[int, int],
    device: torch.device,
    class_names: Sequence[str],
    confidence_threshold: float,
    max_detections: int,
    z_value: float,
    std_temperature: float,
) -> Tuple[List[Dict[str, Any]], float]:
    sample = prepare_image(image, input_size, device)

    synchronize(device)
    start = time.perf_counter()
    outputs = model(sample)
    synchronize(device)
    elapsed_ms = 1000.0 * (time.perf_counter() - start)

    if not isinstance(outputs, Mapping):
        raise TypeError("Model output must be a mapping.")
    required = ("pred_logits", "pred_boxes", "pred_bbox_std")
    missing = [key for key in required if key not in outputs]
    if missing:
        raise KeyError(
            f"Model output is missing {missing}. Verify the uncertainty-head "
            "config and checkpoint."
        )

    logits = outputs["pred_logits"][0]
    boxes = outputs["pred_boxes"][0]
    predicted_std = outputs["pred_bbox_std"][0] * std_temperature
    if predicted_std.shape != boxes.shape:
        raise ValueError(
            "pred_bbox_std must have the same [query, 4] shape as pred_boxes."
        )

    probabilities = logits.sigmoid()
    confidence, class_ids = probabilities.max(dim=-1)
    finite = torch.isfinite(predicted_std).all(dim=-1)
    keep = (confidence >= confidence_threshold) & finite
    selected = torch.nonzero(keep, as_tuple=False).flatten()
    if selected.numel() == 0:
        return [], elapsed_ms

    selected = selected[
        torch.argsort(confidence[selected], descending=True)
    ][:max_detections]

    selected_confidence = confidence[selected].float().cpu()
    selected_classes = class_ids[selected].long().cpu()
    selected_std = predicted_std[selected].float().cpu()
    predicted_xyxy = box_cxcywh_to_xyxy(
        boxes[selected]
    ).float().cpu()

    coordinate_lower = (
        predicted_xyxy - z_value * selected_std
    ).clamp(0.0, 1.0)
    coordinate_upper = (
        predicted_xyxy + z_value * selected_std
    ).clamp(0.0, 1.0)
    outer_xyxy = torch.stack(
        (
            coordinate_lower[:, 0],
            coordinate_lower[:, 1],
            coordinate_upper[:, 2],
            coordinate_upper[:, 3],
        ),
        dim=-1,
    )

    total_std = torch.linalg.vector_norm(
        selected_std,
        ord=2,
        dim=-1,
    )
    relative_std = relative_uncertainty(
        predicted_xyxy,
        selected_std,
    )

    pixel_predictions = normalized_to_pixels(
        predicted_xyxy,
        image.width,
        image.height,
    )
    pixel_std = normalized_to_pixels(
        selected_std,
        image.width,
        image.height,
    )
    pixel_lower = normalized_to_pixels(
        coordinate_lower,
        image.width,
        image.height,
    )
    pixel_upper = normalized_to_pixels(
        coordinate_upper,
        image.width,
        image.height,
    )
    pixel_outer = normalized_to_pixels(
        outer_xyxy,
        image.width,
        image.height,
    )

    bbox_tu = outputs.get("bbox_tu")
    if bbox_tu is not None:
        selected_tu = bbox_tu[0, selected].float().cpu()
    else:
        selected_tu = None

    detections = []
    for index in range(selected.numel()):
        class_id = int(selected_classes[index])
        class_name = (
            class_names[class_id]
            if 0 <= class_id < len(class_names)
            else f"class_{class_id}"
        )
        detection = {
            "query_index": int(selected[index].cpu()),
            "class_id": class_id,
            "class_name": class_name,
            "confidence": float(selected_confidence[index]),
            "total_std": float(total_std[index]),
            "relative_uncertainty": float(relative_std[index]),
            "bbox_xyxy_normalized": predicted_xyxy[index].tolist(),
            "bbox_xyxy_pixels": pixel_predictions[index].tolist(),
            "std_xyxy_normalized": selected_std[index].tolist(),
            "std_xyxy_pixels": pixel_std[index].tolist(),
            "coordinate_lower_normalized": coordinate_lower[index].tolist(),
            "coordinate_upper_normalized": coordinate_upper[index].tolist(),
            "coordinate_lower_pixels": pixel_lower[index].tolist(),
            "coordinate_upper_pixels": pixel_upper[index].tolist(),
            "outer_xyxy_normalized": outer_xyxy[index].tolist(),
            "outer_xyxy_pixels": pixel_outer[index].tolist(),
            "color": list(class_color(class_id)),
        }
        if selected_tu is not None:
            detection["bbox_tu"] = float(selected_tu[index])
        detections.append(detection)
    return detections, elapsed_ms


def write_json_atomic(
    path: Path,
    data: Mapping[str, Any],
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, allow_nan=False)
            file.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.resume).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    image_paths = collect_image_paths(args.input, args.recursive)
    output_dir.mkdir(parents=True, exist_ok=True)
    used_names: Dict[str, Path] = {}
    output_paths = [
        output_path_for(image_path, output_dir, used_names)
        for image_path in image_paths
    ]
    if not args.overwrite:
        existing = [path for path in output_paths if path.exists()]
        if existing:
            raise FileExistsError(
                f"Outputs already exist; use --overwrite: {existing[:5]}"
            )

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

    module_name, uncertainty_module = find_uncertainty_module(model)
    if uncertainty_module.bbox_uncertainty_head is None:
        raise RuntimeError(
            "The configured model has no enabled bbox uncertainty head."
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
            "The uncertainty head requires TU prototypes. Pass "
            "--tu-prototypes or configure tu_prototype_path."
        )

    model.eval()
    input_size = infer_input_size(
        args.input_size,
        uncertainty_module,
    )
    class_names = load_class_names(
        args.categories_json,
        int(uncertainty_module.num_classes),
    )
    z_value, coordinate_coverage = gaussian_z_value(
        args.coverage,
        args.coverage_kind,
    )

    print(
        f"[model] uncertainty module=`{module_name or '<root>'}`, "
        f"input={input_size[0]}x{input_size[1]}, "
        f"coverage={args.coverage:.1%} ({args.coverage_kind}), "
        f"coordinate coverage={coordinate_coverage:.3%}, z={z_value:.4f}"
    )

    total_detections = 0
    inference_times = []
    for image_path, output_path in zip(image_paths, output_paths):
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")

        detections, elapsed_ms = infer_one(
            model,
            image,
            input_size,
            device,
            class_names,
            args.confidence_threshold,
            args.max_detections,
            z_value,
            args.std_temperature,
        )
        visualization = draw_visualization(
            image,
            detections,
            args.coverage,
            args.coverage_kind,
            args.line_width,
            args.region_alpha,
        )
        visualization.save(output_path, format="PNG")

        if args.save_json:
            json_path = output_path.with_suffix(".json")
            if json_path.exists() and not args.overwrite:
                raise FileExistsError(f"JSON output exists: {json_path}")
            write_json_atomic(
                json_path,
                {
                    "input_image": str(image_path),
                    "output_image": str(output_path),
                    "image_width": image.width,
                    "image_height": image.height,
                    "input_height": input_size[0],
                    "input_width": input_size[1],
                    "confidence_threshold": args.confidence_threshold,
                    "nominal_coverage": args.coverage,
                    "coverage_kind": args.coverage_kind,
                    "coordinate_coverage": coordinate_coverage,
                    "gaussian_z": z_value,
                    "std_temperature": args.std_temperature,
                    "inference_ms": elapsed_ms,
                    "detections": detections,
                },
            )

        total_detections += len(detections)
        inference_times.append(elapsed_ms)
        print(
            f"[output] {image_path.name}: detections={len(detections)}, "
            f"inference={elapsed_ms:.1f} ms -> {output_path}"
        )

    mean_ms = sum(inference_times) / len(inference_times)
    print(
        f"[done] images={len(image_paths)}, detections={total_detections}, "
        f"mean inference={mean_ms:.1f} ms"
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
