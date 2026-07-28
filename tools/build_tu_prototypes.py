#!/usr/bin/env python3
"""
Build topology-based uncertainty (TU) prototypes for RT-DETR bbox regression.

The script:
  1. loads an RT-DETR/RT-DETRv2 YAML config and checkpoint;
  2. puts the model in eval mode (therefore disabling decoder dropout);
  3. captures the input activation of a selected Linear layer in each
     decoder bbox MLP by using a forward pre-hook;
  4. Hungarian-matches predictions to ground-truth objects;
  5. keeps sufficiently accurate matched detections;
  6. turns each captured activation into a 0-dimensional persistence
     signature using the maximum spanning tree of its activation graph;
  7. averages the signatures into class/size, class-only, and global
     prototypes and saves them in a PyTorch checkpoint.

No changes to RT-DETR's forward return value are needed. The model only needs
to contain the usual `dec_bbox_head` ModuleList whose MLPs expose a `layers`
ModuleList, as in the official RT-DETRv2 PyTorch implementation and the model
file discussed alongside this script.

Run this file from the `rtdetrv2_pytorch` repository root, for example:

    python tools/build_tu_prototypes.py \
        -c configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml \
        -r output/rtdetrv2_r50vd_6x_coco/checkpoint_best_total.pth \
        -o output/tu/bbox_prototypes.pth \
        --split train \
        --iou-threshold 0.70

For a paper-quality experiment, the prototype set should use training images
with deterministic evaluation transforms. A convenient approach is to point
the config's val_dataloader at the training annotations and then pass
`--split val`; do not build prototypes from your calibration or test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

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


SIZE_NAMES = ("small", "medium", "large")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RT-DETR bbox topology prototypes offline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--config", required=True, help="RT-DETR YAML config.")
    parser.add_argument("-r", "--resume", required=True, help="Model checkpoint.")
    parser.add_argument("-o", "--output", required=True, help="Output .pth file.")
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="Configured dataloader to use.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device, such as cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--checkpoint-key",
        choices=("auto", "ema", "model", "module", "state_dict"),
        default="auto",
        help="Checkpoint entry containing the model state dict.",
    )
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="Require the checkpoint to match the model exactly.",
    )
    parser.add_argument(
        "--decoder-module",
        default=None,
        help=(
            "Qualified module name that owns `dec_bbox_head`. Usually this is "
            "auto-detected (often `decoder`)."
        ),
    )
    parser.add_argument(
        "--bbox-layer-index",
        type=int,
        default=-1,
        help=(
            "Linear layer inside each bbox MLP whose input activation forms "
            "the topology graph. -1 selects the final 4-output layer."
        ),
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.70,
        help="Minimum aligned IoU for accepting a matched detection.",
    )
    parser.add_argument(
        "--allow-wrong-class",
        action="store_true",
        help="Keep IoU-qualified matches even when their predicted class is wrong.",
    )
    parser.add_argument(
        "--small-max-area",
        type=float,
        default=0.0025,
        help="Maximum normalized bbox area for the small bucket.",
    )
    parser.add_argument(
        "--medium-max-area",
        type=float,
        default=0.0225,
        help="Maximum normalized bbox area for the medium bucket.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional maximum number of images to process.",
    )
    parser.add_argument(
        "--max-samples-per-bucket",
        type=int,
        default=None,
        help="Optional cap for each class/size prototype bucket.",
    )
    parser.add_argument(
        "--min-samples-per-bucket",
        type=int,
        default=20,
        help="Mark class/size prototypes below this count as sparse.",
    )
    parser.add_argument(
        "--edge-score",
        choices=("abs_wx", "abs_w"),
        default="abs_wx",
        help=(
            "Activation-graph edge filtration. `abs_wx` uses |w_ji*x_i|; "
            "`abs_w` is an ablation that ignores the activation."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--hash-checkpoint",
        action="store_true",
        help="Store a SHA-256 hash of the checkpoint (can be slow for large files).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.iou_threshold <= 1.0:
        raise ValueError("--iou-threshold must be in [0, 1].")
    if not 0.0 <= args.small_max_area < args.medium_max_area <= 1.0:
        raise ValueError(
            "Area thresholds must satisfy "
            "0 <= small-max-area < medium-max-area <= 1."
        )
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images must be positive.")
    if args.max_samples_per_bucket is not None and args.max_samples_per_bucket <= 0:
        raise ValueError("--max-samples-per-bucket must be positive.")
    if args.min_samples_per_bucket < 1:
        raise ValueError("--min-samples-per-bucket must be at least 1.")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def disable_pretrained_downloads(node: Any) -> None:
    """Recursively prevent backbone downloads before checkpoint loading."""
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
        # Compatibility with PyTorch versions predating `weights_only`.
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
                f"Checkpoint is already a raw state dict; cannot select "
                f"`{checkpoint_key}`."
            )
        return checkpoint, "<root>"

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Checkpoint must be a state dict or a mapping containing one."
        )

    def extract(key: str) -> Optional[Mapping[str, Tensor]]:
        value = checkpoint.get(key)
        if key == "ema" and isinstance(value, Mapping):
            nested = value.get("module")
            if is_state_dict(nested):
                return nested
        return value if is_state_dict(value) else None

    if checkpoint_key != "auto":
        result = extract(checkpoint_key)
        if result is None:
            raise KeyError(
                f"Could not find a valid state dict at checkpoint key "
                f"`{checkpoint_key}`. Available keys: {list(checkpoint.keys())}"
            )
        selected_name = "ema.module" if checkpoint_key == "ema" else checkpoint_key
        return result, selected_name

    for key in ("ema", "model", "module", "state_dict"):
        result = extract(key)
        if result is not None:
            selected_name = "ema.module" if key == "ema" else key
            return result, selected_name

    raise KeyError(
        "Could not auto-detect a model state dict. "
        f"Available checkpoint keys: {list(checkpoint.keys())}"
    )


def strip_module_prefix_if_needed(
    state_dict: Mapping[str, Tensor],
    model: nn.Module,
) -> Mapping[str, Tensor]:
    state_keys = list(state_dict)
    model_keys = set(model.state_dict())
    if not state_keys:
        return state_dict

    direct_overlap = sum(key in model_keys for key in state_keys)
    if all(key.startswith("module.") for key in state_keys):
        stripped = {key[len("module.") :]: value for key, value in state_dict.items()}
        stripped_overlap = sum(key in model_keys for key in stripped)
        if stripped_overlap > direct_overlap:
            return stripped
    return state_dict


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


def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    modules = dict(model.named_modules())
    if name not in modules:
        close = [module_name for module_name in modules if name in module_name]
        raise KeyError(
            f"No module named `{name}`. Similar module names: {close[:20]}"
        )
    return modules[name]


def find_bbox_decoder(
    model: nn.Module,
    requested_name: Optional[str],
) -> Tuple[str, nn.Module]:
    if requested_name is not None:
        module = get_module_by_name(model, requested_name)
        if not hasattr(module, "dec_bbox_head"):
            raise AttributeError(
                f"Module `{requested_name}` does not expose `dec_bbox_head`."
            )
        return requested_name, module

    candidates: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        heads = getattr(module, "dec_bbox_head", None)
        if isinstance(heads, (nn.ModuleList, list, tuple)) and len(heads) > 0:
            candidates.append((name, module))

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise RuntimeError(
            "Could not find a module exposing `dec_bbox_head`. "
            "Pass --decoder-module if your model uses a different name."
        )
    names = [name or "<root>" for name, _ in candidates]
    raise RuntimeError(
        "Found multiple bbox decoder candidates. Select one with "
        f"--decoder-module. Candidates: {names}"
    )


def resolve_eval_decoder_index(decoder_owner: nn.Module) -> int:
    heads = getattr(decoder_owner, "dec_bbox_head")
    decoder_stack = getattr(decoder_owner, "decoder", None)
    eval_idx = getattr(decoder_stack, "eval_idx", None)
    if eval_idx is None:
        eval_idx = len(heads) - 1
        print(
            "[hook] decoder.eval_idx was not found; using the final decoder "
            f"bbox head at index {eval_idx}."
        )
    if eval_idx < 0:
        eval_idx += len(heads)
    if not 0 <= eval_idx < len(heads):
        raise IndexError(
            f"Resolved decoder eval index {eval_idx}, but there are "
            f"{len(heads)} bbox heads."
        )
    return int(eval_idx)


def resolve_bbox_linear(
    decoder_owner: nn.Module,
    eval_idx: int,
    bbox_layer_index: int,
) -> Tuple[nn.Linear, int]:
    bbox_head = decoder_owner.dec_bbox_head[eval_idx]
    layers = getattr(bbox_head, "layers", None)
    if not isinstance(layers, (nn.ModuleList, list, tuple)):
        raise AttributeError(
            "The selected bbox MLP does not expose a `layers` ModuleList."
        )
    resolved = bbox_layer_index
    if resolved < 0:
        resolved += len(layers)
    if not 0 <= resolved < len(layers):
        raise IndexError(
            f"Resolved bbox layer index {resolved}, but the MLP has "
            f"{len(layers)} layers."
        )
    layer = layers[resolved]
    if not isinstance(layer, nn.Linear):
        raise TypeError(
            f"Selected bbox MLP layer is {type(layer).__name__}, not nn.Linear."
        )
    return layer, resolved


class LinearInputCapture:
    """Capture the first positional input passed to an nn.Linear module."""

    def __init__(self, module: nn.Linear) -> None:
        self.value: Optional[Tensor] = None
        self.calls = 0
        self._handle = module.register_forward_pre_hook(self._hook)

    def _hook(self, module: nn.Module, inputs: Tuple[Any, ...]) -> None:
        del module
        if not inputs or not isinstance(inputs[0], Tensor):
            raise RuntimeError("The bbox Linear hook did not receive a Tensor input.")
        self.value = inputs[0].detach()
        self.calls += 1

    def reset(self) -> None:
        self.value = None
        self.calls = 0

    def close(self) -> None:
        self._handle.remove()


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
    """IoU for aligned box pairs; both tensors have shape [N, 4]."""
    if boxes1.shape != boxes2.shape or boxes1.ndim != 2 or boxes1.shape[-1] != 4:
        raise ValueError(
            f"Expected aligned [N, 4] boxes, got {boxes1.shape} and {boxes2.shape}."
        )
    boxes1 = box_cxcywh_to_xyxy(boxes1)
    boxes2 = box_cxcywh_to_xyxy(boxes2)

    top_left = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    bottom_right = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=-1)

    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp(min=0).prod(dim=-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp(min=0).prod(dim=-1)
    union = area1 + area2 - intersection
    return intersection / union.clamp_min(torch.finfo(union.dtype).eps)


def size_bucket(
    normalized_box: Tensor,
    small_max_area: float,
    medium_max_area: float,
) -> int:
    area = float(
        normalized_box[2].clamp(min=0).item()
        * normalized_box[3].clamp(min=0).item()
    )
    if area < small_max_area:
        return 0
    if area < medium_max_area:
        return 1
    return 2


def maximum_spanning_tree_signature(
    activation: Tensor,
    weight: Tensor,
    edge_score: str,
) -> Tensor:
    """
    Return the descending maximum-spanning-tree edge weights.

    The activation graph is a complete bipartite graph:
      - one partition contains the selected Linear layer's input neurons;
      - the other contains its output neurons;
      - edge (i, j) is filtered by |w_ji * x_i| by default.

    Under a descending filtration, the maximum spanning tree contains the
    finite 0D persistence merge values. Because every sample uses the same
    layer, all signatures have the same fixed length:
    n_input + n_output - 1.
    """
    x = activation.detach().float().flatten().cpu()
    w = weight.detach().float().cpu()
    if w.ndim != 2 or w.shape[1] != x.numel():
        raise ValueError(
            f"Linear weight shape {tuple(w.shape)} is incompatible with "
            f"activation shape {tuple(activation.shape)}."
        )

    if edge_score == "abs_wx":
        values = w.abs() * x.abs().unsqueeze(0)
    elif edge_score == "abs_w":
        values = w.abs()
    else:
        raise ValueError(f"Unknown edge score: {edge_score}")

    n_out, n_in = values.shape
    flat_values = values.reshape(-1)
    edge_order = torch.argsort(flat_values, descending=True, stable=True)

    vertex_count = n_in + n_out
    parent = list(range(vertex_count))
    rank = [0] * vertex_count

    def find(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    def union(left: int, right: int) -> bool:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return False
        if rank[root_left] < rank[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        if rank[root_left] == rank[root_right]:
            rank[root_left] += 1
        return True

    selected: List[Tensor] = []
    for flat_index_tensor in edge_order:
        flat_index = int(flat_index_tensor)
        output_index = flat_index // n_in
        input_index = flat_index % n_in
        if union(input_index, n_in + output_index):
            selected.append(flat_values[flat_index])
            if len(selected) == vertex_count - 1:
                break

    expected = vertex_count - 1
    if len(selected) != expected:
        raise RuntimeError(
            f"Maximum spanning tree contains {len(selected)} edges; expected "
            f"{expected}."
        )
    return torch.stack(selected).contiguous()


class RunningPrototype:
    """Streaming mean and pointwise standard deviation of fixed-size signatures."""

    def __init__(self) -> None:
        self.count = 0
        self.sum: Optional[Tensor] = None
        self.sum_sq: Optional[Tensor] = None

    def update(self, signature: Tensor) -> None:
        signature = signature.detach().to(dtype=torch.float64, device="cpu")
        if self.sum is None:
            self.sum = torch.zeros_like(signature)
            self.sum_sq = torch.zeros_like(signature)
        if signature.shape != self.sum.shape:
            raise ValueError(
                f"Signature shape changed from {tuple(self.sum.shape)} to "
                f"{tuple(signature.shape)}."
            )
        self.sum.add_(signature)
        self.sum_sq.add_(signature.square())
        self.count += 1

    def finalize(self) -> Dict[str, Any]:
        if self.count == 0 or self.sum is None or self.sum_sq is None:
            raise RuntimeError("Cannot finalize an empty prototype.")
        mean = self.sum / self.count
        variance = (self.sum_sq / self.count - mean.square()).clamp_min(0)
        return {
            "mean": mean.to(torch.float32),
            "std": variance.sqrt().to(torch.float32),
            "count": self.count,
        }


def update_all_prototypes(
    class_size_stats: Dict[Tuple[int, int], RunningPrototype],
    class_stats: Dict[int, RunningPrototype],
    global_stats: RunningPrototype,
    class_id: int,
    size_id: int,
    signature: Tensor,
    max_samples_per_bucket: Optional[int],
) -> bool:
    key = (class_id, size_id)
    bucket = class_size_stats[key]
    if (
        max_samples_per_bucket is not None
        and bucket.count >= max_samples_per_bucket
    ):
        return False

    bucket.update(signature)
    class_stats[class_id].update(signature)
    global_stats.update(signature)
    return True


def unpack_batch(batch: Any) -> Tuple[Tensor, Sequence[Mapping[str, Any]]]:
    if not isinstance(batch, (list, tuple)) or len(batch) < 2:
        raise TypeError(
            "Expected each dataloader batch to be `(samples, targets, ...)`."
        )
    samples, targets = batch[0], batch[1]
    if not isinstance(samples, Tensor):
        raise TypeError(
            f"Expected samples to be a Tensor, got {type(samples).__name__}."
        )
    if not isinstance(targets, (list, tuple)):
        raise TypeError(
            f"Expected targets to be a list/tuple, got {type(targets).__name__}."
        )
    return samples, targets


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def json_safe_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def run(args: argparse.Namespace) -> Dict[str, Any]:
    validate_args(args)
    seed_everything(args.seed)

    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.resume).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    print(f"[config] loading {config_path}")
    cfg = YAMLConfig(str(config_path))
    yaml_cfg = getattr(cfg, "yaml_cfg", None)
    if yaml_cfg is not None:
        disable_pretrained_downloads(yaml_cfg)

    model: nn.Module = cfg.model
    criterion: nn.Module = cfg.criterion
    matcher = getattr(criterion, "matcher", None)
    if matcher is None:
        raise AttributeError(
            "The configured criterion does not expose `matcher`; the script "
            "needs RT-DETR's Hungarian matcher."
        )

    selected_checkpoint_key = load_model_weights(
        model,
        checkpoint_path,
        args.checkpoint_key,
        args.strict_load,
    )
    model.to(device)
    model.eval()
    criterion.to(device)
    criterion.eval()

    decoder_name, decoder_owner = find_bbox_decoder(model, args.decoder_module)
    eval_idx = resolve_eval_decoder_index(decoder_owner)
    bbox_linear, bbox_layer_idx = resolve_bbox_linear(
        decoder_owner,
        eval_idx,
        args.bbox_layer_index,
    )
    linear_weight = bbox_linear.weight.detach().float().cpu()

    print(
        "[hook] capturing input to "
        f"`{decoder_name or '<root>'}.dec_bbox_head[{eval_idx}]"
        f".layers[{bbox_layer_idx}]`, shape "
        f"{bbox_linear.in_features}->{bbox_linear.out_features}"
    )
    if bbox_linear.out_features != 4:
        print(
            "[hook] note: selected layer does not have four outputs. This is "
            "valid as an ablation, but it is not the final bbox-regression layer."
        )

    if args.split == "train":
        dataloader = cfg.train_dataloader
        print(
            "[data] using train_dataloader. For reproducible prototypes, make "
            "sure its image transforms are deterministic."
        )
    else:
        dataloader = cfg.val_dataloader
        print("[data] using val_dataloader.")

    capture = LinearInputCapture(bbox_linear)
    class_size_stats: Dict[Tuple[int, int], RunningPrototype] = defaultdict(
        RunningPrototype
    )
    class_stats: Dict[int, RunningPrototype] = defaultdict(RunningPrototype)
    global_stats = RunningPrototype()

    images_seen = 0
    matches_seen = 0
    rejected_low_iou = 0
    rejected_wrong_class = 0
    rejected_bucket_full = 0
    kept = 0

    try:
        total_batches = len(dataloader)
    except TypeError:
        total_batches = None
    progress = tqdm(dataloader, total=total_batches, desc=f"[{args.split}] prototypes", unit="batch")

    try:
        with torch.inference_mode():
            for batch in progress:
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

                samples = samples.to(device, non_blocking=True)
                targets = move_targets_to_device(targets_cpu, device)

                capture.reset()
                outputs = model(samples)
                if not isinstance(outputs, Mapping):
                    raise TypeError(
                        f"Expected model output mapping, got "
                        f"{type(outputs).__name__}."
                    )
                if "pred_logits" not in outputs or "pred_boxes" not in outputs:
                    raise KeyError(
                        "Model output must contain `pred_logits` and `pred_boxes`."
                    )
                if capture.value is None:
                    raise RuntimeError(
                        "The bbox activation hook did not fire. Verify "
                        "--decoder-module, --bbox-layer-index, and the model's "
                        "eval decoder index."
                    )
                if capture.calls != 1:
                    raise RuntimeError(
                        f"The selected bbox layer ran {capture.calls} times in "
                        "one forward pass. Use the eval decoder head or select "
                        "a module that executes once."
                    )

                activations = capture.value
                pred_logits = outputs["pred_logits"]
                pred_boxes = outputs["pred_boxes"]
                if activations.shape[:2] != pred_boxes.shape[:2]:
                    raise RuntimeError(
                        f"Hooked activations have leading shape "
                        f"{tuple(activations.shape[:2])}, but predicted boxes "
                        f"have {tuple(pred_boxes.shape[:2])}."
                    )

                match_indices = matcher(
                    {"pred_logits": pred_logits, "pred_boxes": pred_boxes},
                    targets,
                )["indices"]
                pred_classes = pred_logits.sigmoid().argmax(dim=-1)

                for image_index, (source_index, target_index) in enumerate(
                    match_indices
                ):
                    source_index = source_index.to(device=device, dtype=torch.long)
                    target_index = target_index.to(device=device, dtype=torch.long)
                    if source_index.numel() == 0:
                        continue

                    matches_seen += int(source_index.numel())
                    matched_pred_boxes = pred_boxes[image_index, source_index]
                    matched_target_boxes = targets[image_index]["boxes"][target_index]
                    ious = aligned_box_iou(
                        matched_pred_boxes,
                        matched_target_boxes,
                    )
                    predicted_labels = pred_classes[image_index, source_index]
                    target_labels = targets[image_index]["labels"][target_index]

                    for local_index in range(source_index.numel()):
                        if float(ious[local_index]) < args.iou_threshold:
                            rejected_low_iou += 1
                            continue
                        if (
                            not args.allow_wrong_class
                            and int(predicted_labels[local_index])
                            != int(target_labels[local_index])
                        ):
                            rejected_wrong_class += 1
                            continue

                        query_index = int(source_index[local_index])
                        class_id = int(target_labels[local_index])
                        size_id = size_bucket(
                            matched_target_boxes[local_index],
                            args.small_max_area,
                            args.medium_max_area,
                        )
                        signature = maximum_spanning_tree_signature(
                            activations[image_index, query_index],
                            linear_weight,
                            args.edge_score,
                        )
                        was_kept = update_all_prototypes(
                            class_size_stats,
                            class_stats,
                            global_stats,
                            class_id,
                            size_id,
                            signature,
                            args.max_samples_per_bucket,
                        )
                        if was_kept:
                            kept += 1
                        else:
                            rejected_bucket_full += 1

                images_seen += int(samples.shape[0])
                progress.set_postfix(
                    images=images_seen, matches=matches_seen, kept=kept
                )
    finally:
        capture.close()
        progress.close()

    if kept == 0:
        raise RuntimeError(
            "No prototype samples passed the filters. Check checkpoint quality, "
            "class labels, --iou-threshold, and --allow-wrong-class."
        )

    finalized_class_size = {
        f"{class_id}:{SIZE_NAMES[size_id]}": stats.finalize()
        for (class_id, size_id), stats in sorted(class_size_stats.items())
        if stats.count > 0
    }
    finalized_class = {
        str(class_id): stats.finalize()
        for class_id, stats in sorted(class_stats.items())
        if stats.count > 0
    }
    finalized_global = global_stats.finalize()

    sparse_buckets = sorted(
        key
        for key, value in finalized_class_size.items()
        if value["count"] < args.min_samples_per_bucket
    )

    checkpoint_stat = checkpoint_path.stat()
    metadata: Dict[str, Any] = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "activation_graph_ph0_maximum_spanning_tree",
        "edge_score": args.edge_score,
        "signature_sort": "descending",
        "config": json_safe_path(config_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint": json_safe_path(checkpoint_path),
        "checkpoint_key": selected_checkpoint_key,
        "checkpoint_size_bytes": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "decoder_module": decoder_name,
        "decoder_eval_index": eval_idx,
        "bbox_layer_index": bbox_layer_idx,
        "bbox_layer_in_features": bbox_linear.in_features,
        "bbox_layer_out_features": bbox_linear.out_features,
        "signature_length": bbox_linear.in_features
        + bbox_linear.out_features
        - 1,
        "split": args.split,
        "iou_threshold": args.iou_threshold,
        "require_correct_class": not args.allow_wrong_class,
        "normalized_area_thresholds": {
            "small_max": args.small_max_area,
            "medium_max": args.medium_max_area,
        },
        "size_bucket_names": list(SIZE_NAMES),
        "max_images": args.max_images,
        "max_samples_per_bucket": args.max_samples_per_bucket,
        "min_samples_per_bucket": args.min_samples_per_bucket,
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "counts": {
            "images_seen": images_seen,
            "hungarian_matches": matches_seen,
            "kept": kept,
            "rejected_low_iou": rejected_low_iou,
            "rejected_wrong_class": rejected_wrong_class,
            "rejected_bucket_full": rejected_bucket_full,
            "class_size_buckets": len(finalized_class_size),
            "classes": len(finalized_class),
        },
        "sparse_class_size_buckets": sparse_buckets,
    }
    if args.hash_checkpoint:
        print("[output] hashing checkpoint...")
        metadata["checkpoint_sha256"] = sha256_file(checkpoint_path)

    artifact = {
        "metadata": metadata,
        "prototypes": {
            "class_size": finalized_class_size,
            "class": finalized_class,
            "global": finalized_global,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)

    print(
        f"[output] saved {len(finalized_class_size)} class/size prototypes, "
        f"{len(finalized_class)} class prototypes, and one global prototype "
        f"from {kept} detections to {output_path}"
    )
    if sparse_buckets:
        print(
            f"[output] {len(sparse_buckets)} class/size buckets contain fewer "
            f"than {args.min_samples_per_bucket} samples; inference should "
            "fall back to the class-only prototype for these buckets."
        )
    return artifact


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()