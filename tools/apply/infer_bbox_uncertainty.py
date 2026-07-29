#!/usr/bin/env python3
"""
Run RT-DETRv2 TU inference and visualize conformal bbox uncertainty.

For each retained detection, the output image shows:

  * a solid rectangle: RT-DETR's predicted bounding box;
  * a translucent dashed rectangle: the outer envelope of the calibrated
    conformal coordinate region;
  * a label containing class, confidence, and bbox TU.

The conformal artifact represents intervals for all four ``xyxy`` coordinates:

    x1 in [x1_lower, x1_upper]
    y1 in [y1_lower, y1_upper]
    x2 in [x2_lower, x2_upper]
    y2 in [y2_lower, y2_upper]

The dashed visualization is the outer envelope
``[x1_lower, y1_lower, x2_upper, y2_upper]``. It is useful visually but does
not replace the complete coordinate intervals, which are optionally written
to a sidecar JSON file.

Example:

    python tools/infer_bbox_uncertainty.py \
        -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml \
        -r pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \
        --tu-prototypes output/tu/bbox_prototypes.pth \
        --cp-calibration output/tu/bbox_cp_calibration.pth \
        --input examples/example.jpg \
        --output-dir output/tu/visualizations \
        --confidence-threshold 0.5 \
        --tu-topk 50 \
        --max-detections 20 \
        --save-json

The model must expose ``bbox_tu`` during evaluation and its TUE decoder must
provide ``load_tu_prototypes(path)``.

Important coverage scope:
The conformal coverage calibrated by ``build_bbox_cp_calibration.py`` applies
to matched detections satisfying its filters. It does not guarantee that an
unmatched false positive corresponds to a real object, nor does it cover
objects the detector missed.
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor, nn
from torchvision.transforms.functional import to_tensor


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.core import YAMLConfig
except ImportError as exc:
    raise ImportError(
        "Could not import `src.core.YAMLConfig`. Put this script in "
        "`rtdetrv2_pytorch/tools/` and run it from the repository root."
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
        description="Visualize RT-DETR bbox TU and conformal uncertainty.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--config", required=True, help="RT-DETR YAML config.")
    parser.add_argument("-r", "--resume", required=True, help="Detector checkpoint.")
    parser.add_argument(
        "--tu-prototypes",
        required=True,
        help="BBox persistence-diagram Fréchet means.",
    )
    parser.add_argument(
        "--cp-calibration",
        required=True,
        help="Artifact created by build_bbox_cp_calibration.py.",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more image files or directories.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/tu/visualizations",
        help="Directory for annotated images.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input directories recursively.",
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
        help="Checkpoint entry containing detector weights.",
    )
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="Require the detector checkpoint to match exactly.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum per-query maximum sigmoid confidence.",
    )
    parser.add_argument(
        "--tu-topk",
        type=int,
        default=50,
        help="Number of highest-confidence queries for which TU is calculated.",
    )
    parser.add_argument(
        "--tu-min-samples",
        type=int,
        default=20,
        help="Minimum count for a class TU prototype.",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=20,
        help="Maximum plotted detections per image.",
    )
    parser.add_argument(
        "--cp-mode",
        choices=("scaled", "unscaled"),
        default="scaled",
        help="Use TU-scaled CP or its unscaled baseline.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Override the model evaluation input size.",
    )
    parser.add_argument(
        "--categories-json",
        default=None,
        help=(
            "Optional COCO annotation JSON used to obtain category names in "
            "ascending category-id order."
        ),
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save complete coordinate intervals beside every output image.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=3,
        help="Predicted bbox line width.",
    )
    parser.add_argument(
        "--region-alpha",
        type=int,
        default=42,
        help="Uncertainty-region opacity in [0, 255].",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing visualization files.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("--confidence-threshold must be in [0, 1].")
    if args.tu_topk < 1:
        raise ValueError("--tu-topk must be positive.")
    if args.tu_min_samples < 1:
        raise ValueError("--tu-min-samples must be positive.")
    if args.max_detections < 1:
        raise ValueError("--max-detections must be positive.")
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
            raise KeyError("Raw state dict has no nested checkpoint key.")
        return checkpoint, "<root>"

    if not isinstance(checkpoint, Mapping):
        raise TypeError("Detector checkpoint is not a mapping.")

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
                f"No state dict at `{checkpoint_key}`. "
                f"Available keys: {list(checkpoint.keys())}"
            )
        return selected, "ema.module" if checkpoint_key == "ema" else checkpoint_key

    for key in ("ema", "model", "module", "state_dict"):
        selected = extract(key)
        if selected is not None:
            return selected, "ema.module" if key == "ema" else key
    raise KeyError(
        "Could not detect detector weights. "
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
    checkpoint = load_torch_file(checkpoint_path)
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
            "No TUE decoder exposing load_tu_prototypes() was found."
        )
    raise RuntimeError(
        "Multiple TUE decoder candidates were found: "
        f"{[name or '<root>' for name, _ in candidates]}"
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
            iterator: Iterable[Path] = path.rglob("*") if recursive else path.glob("*")
            for candidate in iterator:
                if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
                    found[candidate.resolve()] = None
        else:
            raise FileNotFoundError(f"Input does not exist: {path}")

    image_paths = sorted(found, key=lambda item: str(item))
    if not image_paths:
        raise RuntimeError("No supported image files were found.")
    return image_paths


def make_output_path(
    image_path: Path,
    output_dir: Path,
    used_names: Dict[str, Path],
) -> Path:
    base_name = f"{image_path.stem}_uncertainty.png"
    if base_name in used_names and used_names[base_name] != image_path:
        suffix = hashlib.sha256(str(image_path).encode("utf-8")).hexdigest()[:8]
        base_name = f"{image_path.stem}_{suffix}_uncertainty.png"
    used_names[base_name] = image_path
    return output_dir / base_name


def load_class_names(path: Optional[str], num_classes: int) -> List[str]:
    if path is None:
        if num_classes == len(COCO_CLASS_NAMES):
            return list(COCO_CLASS_NAMES)
        return [f"class_{index}" for index in range(num_classes)]

    categories_path = Path(path).expanduser().resolve()
    with categories_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    categories = data.get("categories") if isinstance(data, Mapping) else None
    if not isinstance(categories, list):
        raise ValueError("--categories-json does not contain a `categories` list.")
    ordered = sorted(categories, key=lambda category: category["id"])
    names = [str(category["name"]) for category in ordered]
    if len(names) != num_classes:
        raise ValueError(
            f"Categories JSON contains {len(names)} categories, but the model "
            f"uses {num_classes} classes."
        )
    return names


def infer_input_size(
    args: argparse.Namespace,
    tu_module: nn.Module,
) -> Tuple[int, int]:
    if args.input_size is not None:
        return int(args.input_size[0]), int(args.input_size[1])

    spatial_size = getattr(tu_module, "eval_spatial_size", None)
    if (
        isinstance(spatial_size, (list, tuple))
        and len(spatial_size) == 2
        and min(int(value) for value in spatial_size) > 0
    ):
        return int(spatial_size[0]), int(spatial_size[1])
    print("[input] eval_spatial_size was unavailable; defaulting to 640x640")
    return 640, 640


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


def lookup_tu_scales(
    bbox_tu: Tensor,
    cp_artifact: Mapping[str, Any],
) -> Tensor:
    scale_model = cp_artifact["scale_model"]
    edges = scale_model["tu_bin_edges"].detach().float().cpu()
    scales = scale_model["coordinate_scales"].detach().float().cpu()
    bin_indices = torch.bucketize(
        bbox_tu.detach().float().cpu(),
        edges,
    )
    return scales[bin_indices]


def calculate_cp_intervals(
    predicted_xyxy: Tensor,
    bbox_tu: Tensor,
    cp_artifact: Mapping[str, Any],
    mode: str,
) -> Tuple[Tensor, Tensor, Tensor]:
    predictions_cpu = predicted_xyxy.detach().float().cpu()
    conformal = cp_artifact["conformal"][mode]
    qhat = float(conformal["qhat"])

    if mode == "scaled":
        coordinate_scales = lookup_tu_scales(bbox_tu, cp_artifact)
        half_widths = qhat * coordinate_scales
    elif mode == "unscaled":
        half_widths = torch.full_like(predictions_cpu, qhat)
    else:
        raise ValueError(f"Unknown CP mode: {mode}")

    lower = (predictions_cpu - half_widths).clamp(0.0, 1.0)
    upper = (predictions_cpu + half_widths).clamp(0.0, 1.0)
    return lower, upper, half_widths


def normalized_to_pixels(boxes: Tensor, width: int, height: int) -> Tensor:
    scale = torch.tensor(
        [width, height, width, height],
        dtype=boxes.dtype,
    )
    return boxes.cpu() * scale


def class_color(class_id: int) -> Tuple[int, int, int]:
    # Golden-ratio spacing gives stable, separated colors.
    hue = (class_id * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 0.98)
    return int(red * 255), int(green * 255), int(blue * 255)


def clamp_rectangle(
    box: Sequence[float],
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    x1 = min(max(float(box[0]), 0.0), max(width - 1, 0))
    y1 = min(max(float(box[1]), 0.0), max(height - 1, 0))
    x2 = min(max(float(box[2]), 0.0), max(width - 1, 0))
    y2 = min(max(float(box[3]), 0.0), max(height - 1, 0))
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
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length <= 0:
        return
    unit_x = dx / length
    unit_y = dy / length
    position = 0.0
    while position < length:
        segment_end = min(position + dash, length)
        draw.line(
            (
                x1 + unit_x * position,
                y1 + unit_y * position,
                x1 + unit_x * segment_end,
                y1 + unit_y * segment_end,
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
        y = y + (bottom - top) + 2 * padding
        left, top, right, bottom = draw.textbbox((x, y), text, font=font)
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
    line_width: int,
    region_alpha: int,
) -> Image.Image:
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay, "RGBA")

    # Regions first, so point-estimate boxes remain crisp.
    for detection in detections:
        color_rgb = tuple(detection["color"])
        outer_box = clamp_rectangle(
            detection["cp_outer_xyxy_pixels"],
            canvas.width,
            canvas.height,
        )
        overlay_draw.rectangle(
            outer_box,
            fill=(*color_rgb, region_alpha),
        )
        draw_dashed_rectangle(
            overlay_draw,
            outer_box,
            (*color_rgb, 235),
            max(1, line_width - 1),
        )

    canvas = Image.alpha_composite(canvas, overlay)
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = load_font(canvas.width)

    for detection in detections:
        color_rgb = tuple(detection["color"])
        predicted_box = clamp_rectangle(
            detection["bbox_xyxy_pixels"],
            canvas.width,
            canvas.height,
        )
        draw.rectangle(
            predicted_box,
            outline=(*color_rgb, 255),
            width=line_width,
        )
        label = (
            f"{detection['class_name']} "
            f"{detection['confidence']:.2f} | "
            f"TU {detection['bbox_tu']:.4g}"
        )
        draw_label(
            draw,
            (predicted_box[0] + 2, predicted_box[1] - 2),
            label,
            (*color_rgb, 255),
            font,
        )

    legend = f"solid: prediction | dashed/shaded: {coverage:.0%} CP region"
    draw_label(draw, (8, 8), legend, (255, 255, 255, 255), font)
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
    return to_tensor(resized).unsqueeze(0).to(device, non_blocking=True)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def infer_one(
    model: nn.Module,
    image: Image.Image,
    input_size: Tuple[int, int],
    device: torch.device,
    cp_artifact: Mapping[str, Any],
    class_names: Sequence[str],
    confidence_threshold: float,
    max_detections: int,
    cp_mode: str,
) -> Tuple[List[Dict[str, Any]], float]:
    sample = prepare_image(image, input_size, device)

    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(sample)
    synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    if not isinstance(outputs, Mapping):
        raise TypeError("Model output is not a mapping.")
    required = ("pred_logits", "pred_boxes", "bbox_tu")
    missing = [key for key in required if key not in outputs]
    if missing:
        raise KeyError(
            f"Model output is missing {missing}. Verify TUE prototype loading "
            "and the TU-enabled forward path."
        )

    logits = outputs["pred_logits"][0]
    boxes = outputs["pred_boxes"][0]
    bbox_tu = outputs["bbox_tu"][0]
    if bbox_tu.shape != logits.shape[:1]:
        raise ValueError("bbox_tu query dimension does not match pred_logits.")

    probabilities = logits.sigmoid()
    confidence, class_ids = probabilities.max(dim=-1)
    keep = (confidence >= confidence_threshold) & torch.isfinite(bbox_tu)
    selected = torch.nonzero(keep, as_tuple=False).flatten()

    if selected.numel() == 0:
        return [], elapsed_ms
    selected = selected[
        torch.argsort(confidence[selected], descending=True)
    ][:max_detections]

    selected_confidence = confidence[selected].detach().float().cpu()
    selected_classes = class_ids[selected].detach().long().cpu()
    selected_tu = bbox_tu[selected].detach().float().cpu()
    predicted_xyxy = box_cxcywh_to_xyxy(boxes[selected]).detach().float().cpu()
    predicted_xyxy = predicted_xyxy.clamp(0.0, 1.0)
    coordinate_lower, coordinate_upper, half_widths = calculate_cp_intervals(
        predicted_xyxy,
        selected_tu,
        cp_artifact,
        cp_mode,
    )

    outer_xyxy = torch.stack(
        (
            coordinate_lower[:, 0],
            coordinate_lower[:, 1],
            coordinate_upper[:, 2],
            coordinate_upper[:, 3],
        ),
        dim=-1,
    )

    pixel_predictions = normalized_to_pixels(
        predicted_xyxy,
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

    detections: List[Dict[str, Any]] = []
    for index in range(selected.numel()):
        class_id = int(selected_classes[index])
        class_name = (
            class_names[class_id]
            if 0 <= class_id < len(class_names)
            else f"class_{class_id}"
        )
        detections.append(
            {
                "query_index": int(selected[index].detach().cpu()),
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(selected_confidence[index]),
                "bbox_tu": float(selected_tu[index]),
                "bbox_xyxy_normalized": predicted_xyxy[index].tolist(),
                "bbox_xyxy_pixels": pixel_predictions[index].tolist(),
                "cp_coordinate_lower_normalized": coordinate_lower[index].tolist(),
                "cp_coordinate_upper_normalized": coordinate_upper[index].tolist(),
                "cp_coordinate_lower_pixels": pixel_lower[index].tolist(),
                "cp_coordinate_upper_pixels": pixel_upper[index].tolist(),
                "cp_half_width_normalized": half_widths[index].tolist(),
                "cp_outer_xyxy_normalized": outer_xyxy[index].tolist(),
                "cp_outer_xyxy_pixels": pixel_outer[index].tolist(),
                "color": list(class_color(class_id)),
            }
        )
    return detections, elapsed_ms


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_cp_artifact(
    cp_artifact: Mapping[str, Any],
    prototypes_path: Path,
    args: argparse.Namespace,
) -> float:
    if "metadata" not in cp_artifact or "conformal" not in cp_artifact:
        raise KeyError("CP artifact is missing metadata or conformal parameters.")
    if args.cp_mode not in cp_artifact["conformal"]:
        raise KeyError(f"CP artifact does not contain mode `{args.cp_mode}`.")
    if args.cp_mode == "scaled" and "scale_model" not in cp_artifact:
        raise KeyError("Scaled CP artifact is missing its TU scale model.")

    metadata = cp_artifact["metadata"]
    coordinate_format = metadata.get("coordinate_format")
    if coordinate_format != "normalized_xyxy":
        raise ValueError(
            "Expected CP coordinate_format `normalized_xyxy`, got "
            f"`{coordinate_format}`."
        )

    expected_prototype_hash = metadata.get("tu_prototypes_sha256")
    if expected_prototype_hash:
        actual_hash = sha256_file(prototypes_path)
        if actual_hash != expected_prototype_hash:
            raise ValueError(
                "TU prototype file does not match the one used for CP calibration."
            )

    conformal = cp_artifact["conformal"][args.cp_mode]
    qhat = float(conformal["qhat"])
    if not math.isfinite(qhat) or qhat < 0:
        raise ValueError(f"CP qhat must be finite and nonnegative, got {qhat}.")
    if args.cp_mode == "scaled":
        scales = cp_artifact["scale_model"]["coordinate_scales"]
        maximum_width = 2.0 * qhat * float(
            scales.detach().float().max().cpu()
        )
    else:
        maximum_width = 2.0 * qhat
    if maximum_width > 4.0:
        raise ValueError(
            "The CP artifact contains implausibly large normalized-coordinate "
            f"widths (maximum {maximum_width:.3f}). It was likely calibrated "
            "with pixel xyxy targets compared against normalized cxcywh "
            "predictions. Rebuild it with the corrected "
            "build_bbox_cp_calibration.py."
        )

    calibrated_topk = metadata.get("tu_topk")
    if calibrated_topk is not None and int(calibrated_topk) != args.tu_topk:
        print(
            f"[warning] CP used tu_topk={calibrated_topk}, but inference uses "
            f"tu_topk={args.tu_topk}."
        )
    calibrated_threshold = metadata.get("confidence_threshold")
    if (
        calibrated_threshold is not None
        and not math.isclose(
            float(calibrated_threshold),
            args.confidence_threshold,
            abs_tol=1e-12,
        )
    ):
        print(
            f"[warning] CP used confidence threshold {calibrated_threshold}, "
            f"but inference uses {args.confidence_threshold}. The calibrated "
            "coverage population has changed."
        )
    return float(metadata.get("target_coverage", 0.95))


def write_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
            file.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.resume).expanduser().resolve()
    prototypes_path = Path(args.tu_prototypes).expanduser().resolve()
    cp_path = Path(args.cp_calibration).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    for label, path in (
        ("Config", config_path),
        ("Checkpoint", checkpoint_path),
        ("TU prototypes", prototypes_path),
        ("CP calibration", cp_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    image_paths = collect_image_paths(args.input, args.recursive)
    output_dir.mkdir(parents=True, exist_ok=True)
    used_names: Dict[str, Path] = {}
    output_paths = [
        make_output_path(path, output_dir, used_names) for path in image_paths
    ]
    if not args.overwrite:
        existing = [path for path in output_paths if path.exists()]
        if existing:
            raise FileExistsError(
                "Visualization output already exists. Use --overwrite: "
                f"{existing[:5]}"
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

    tu_module_name, tu_module = find_tu_module(model)
    if hasattr(tu_module, "tu_min_samples"):
        tu_module.tu_min_samples = args.tu_min_samples
    if hasattr(tu_module, "bbox_tu_topk"):
        tu_module.bbox_tu_topk = args.tu_topk
    tu_module.load_tu_prototypes(str(prototypes_path))
    tu_module.tu_enabled = True
    model.eval()

    cp_artifact = load_torch_file(cp_path)
    if not isinstance(cp_artifact, Mapping):
        raise TypeError("CP calibration artifact is not a mapping.")
    coverage = validate_cp_artifact(cp_artifact, prototypes_path, args)

    num_classes = int(getattr(tu_module, "num_classes"))
    class_names = load_class_names(args.categories_json, num_classes)
    input_size = infer_input_size(args, tu_module)
    print(
        f"[model] TU module=`{tu_module_name or '<root>'}`, "
        f"input={input_size[0]}x{input_size[1]}, "
        f"CP mode={args.cp_mode}, coverage target={coverage:.1%}"
    )

    total_detections = 0
    elapsed_times: List[float] = []
    for image_path, output_path in zip(image_paths, output_paths):
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")

        detections, elapsed_ms = infer_one(
            model,
            image,
            input_size,
            device,
            cp_artifact,
            class_names,
            args.confidence_threshold,
            args.max_detections,
            args.cp_mode,
        )
        visualization = draw_visualization(
            image,
            detections,
            coverage,
            args.line_width,
            args.region_alpha,
        )
        visualization.save(output_path, format="PNG")

        if args.save_json:
            sidecar_path = output_path.with_suffix(".json")
            if sidecar_path.exists() and not args.overwrite:
                raise FileExistsError(f"Sidecar already exists: {sidecar_path}")
            write_json_atomic(
                sidecar_path,
                {
                    "input_image": str(image_path),
                    "output_image": str(output_path),
                    "image_width": image.width,
                    "image_height": image.height,
                    "input_height": input_size[0],
                    "input_width": input_size[1],
                    "cp_mode": args.cp_mode,
                    "target_coverage": coverage,
                    "confidence_threshold": args.confidence_threshold,
                    "inference_ms": elapsed_ms,
                    "detections": detections,
                },
            )

        total_detections += len(detections)
        elapsed_times.append(elapsed_ms)
        print(
            f"[output] {image_path.name}: detections={len(detections)}, "
            f"inference={elapsed_ms:.1f} ms -> {output_path}"
        )

    mean_ms = sum(elapsed_times) / len(elapsed_times)
    print(
        f"[done] images={len(image_paths)}, detections={total_detections}, "
        f"mean inference={mean_ms:.1f} ms"
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
