from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn
from torchvision.transforms.v2 import functional as vision

from src.nn.backbone.presnet import PResNet
from src.zoo.rtdetr.hybrid_encoder import HybridEncoder
from src.zoo.rtdetr.rtdetr import RTDETR
from src.zoo.rtdetr.rtdetrv2_decoder import RTDETRTransformerv2

from .config import FIXED_CONFIG, ExperimentConfig
from .persistence import Layer2Capture, batched_persistence


def build_fixed_detector() -> RTDETR:
    spatial_size = [640, 640]
    backbone = PResNet(depth=18, variant="d", num_stages=4, return_idx=[1, 2, 3], act="relu", freeze_at=-1, freeze_norm=False, pretrained=False)
    encoder = HybridEncoder(in_channels=[128, 256, 512], feat_strides=[8, 16, 32], hidden_dim=256, nhead=8, dim_feedforward=1024, dropout=0.0, enc_act="gelu", use_encoder_idx=[2], num_encoder_layers=1, pe_temperature=10_000, expansion=0.5, depth_mult=1.0, act="silu", eval_spatial_size=spatial_size, version="v2")
    decoder = RTDETRTransformerv2(num_classes=80, hidden_dim=256, num_queries=300, feat_channels=[256, 256, 256], feat_strides=[8, 16, 32], num_levels=3, num_points=[4, 4, 4], nhead=8, num_layers=3, dim_feedforward=1024, dropout=0.0, activation="relu", num_denoising=100, label_noise_ratio=0.5, box_noise_scale=1.0, learn_query_content=False, eval_spatial_size=spatial_size, eval_idx=-1, eps=1e-2, aux_loss=True, cross_attn_method="default", query_select_method="default")
    return RTDETR(backbone=backbone, encoder=encoder, decoder=decoder)


def _tensor_state(value: Any, location: str) -> dict[str, Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise TypeError(f"{location} state must be a nonempty mapping")
    if not all(isinstance(key, str) and isinstance(tensor, Tensor) for key, tensor in value.items()):
        raise TypeError(f"{location} state must contain only named tensors")
    return dict(value)


def checkpoint_state(checkpoint: object) -> dict[str, Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    if "ema" in checkpoint:
        ema = checkpoint["ema"]
        if not isinstance(ema, Mapping) or "module" not in ema:
            raise KeyError("checkpoint ema has no module state")
        return _tensor_state(ema["module"], "checkpoint ema.module")
    if "model" in checkpoint:
        return _tensor_state(checkpoint["model"], "checkpoint model")
    return _tensor_state(checkpoint, "checkpoint direct")


def load_frozen_detector(checkpoint_path: str | Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_fixed_detector()
    incompatible = model.load_state_dict(checkpoint_state(checkpoint), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}")
    return model.to(device).eval().requires_grad_(False)


def resize_image(image: Image.Image, image_size: tuple[int, int]) -> Image.Image:
    rgb = image.convert("RGB")
    try:
        resized = vision.resize(rgb, list(image_size), antialias=True)
    except Exception:
        if rgb is not image:
            rgb.close()
        raise
    if rgb is not image and resized is not rgb:
        rgb.close()
    return resized


def prepare_image(image: Image.Image, image_size: tuple[int, int]) -> Tensor:
    resized = resize_image(image, image_size)
    try:
        return vision.pil_to_tensor(resized).to(torch.float32).div_(255.0)
    finally:
        resized.close()


def _padded_ids(boxes: Tensor, logits: Tensor, persistence: Tensor) -> Tensor:
    fields = (boxes, logits, persistence)
    if any(field.ndim != 2 or not field.is_floating_point() or not bool(torch.isfinite(field).all()) for field in fields):
        raise ValueError("detector outputs must be finite rank-two floating tensors")
    if len({field.shape[0] for field in fields}) != 1:
        raise ValueError("detector output query counts must agree")
    query_count = boxes.shape[0]
    if query_count < 2:
        return torch.empty(0, dtype=torch.int64)
    repeated = torch.stack([field.eq(field[-1]).reshape(query_count, -1).all(dim=1) for field in fields]).all(dim=0)
    start = query_count - 1
    while start > 0 and bool(repeated[start - 1]):
        start -= 1
    if query_count - start < 2:
        return torch.empty(0, dtype=torch.int64)
    return torch.arange(start, query_count, dtype=torch.int64)


class RTDETRExtractor:
    def __init__(self, checkpoint_path: str | Path, device: torch.device, config: ExperimentConfig = FIXED_CONFIG) -> None:
        self.device = device
        self.config = config
        self.model: nn.Module | None = load_frozen_detector(checkpoint_path, device)
        self.capture: Layer2Capture | None = Layer2Capture(self.model.decoder, layer=config.persistence_layer)

    @torch.inference_mode()
    def extract_batch(self, images: Tensor) -> list[dict[str, Tensor]]:
        if self.model is None or self.capture is None:
            raise RuntimeError("extractor is closed")
        if not isinstance(images, Tensor) or images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must be a rank-four NCHW RGB tensor")
        if tuple(images.shape[2:]) != self.config.image_size or images.shape[0] == 0:
            raise ValueError("images must be a nonempty configured-size batch")
        outputs = self.model(images.to(self.device))
        if not isinstance(outputs, Mapping) or not isinstance(outputs.get("pred_logits"), Tensor) or not isinstance(outputs.get("pred_boxes"), Tensor):
            raise ValueError("detector must return pred_logits and pred_boxes tensors")
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        features, weight = self.capture.take()
        persistence = batched_persistence(weight, features.reshape(-1, features.shape[-1])).reshape(features.shape[0], features.shape[1], -1)
        expected_logits = (images.shape[0], self.config.query_count, self.config.class_count)
        expected_boxes = (images.shape[0], self.config.query_count, 4)
        expected_persistence = (images.shape[0], self.config.query_count, self.config.persistence_dim)
        if tuple(logits.shape) != expected_logits or tuple(boxes.shape) != expected_boxes or tuple(persistence.shape) != expected_persistence:
            raise ValueError("detector output shapes do not match the fixed configuration")
        records = []
        for item_logits, item_boxes, item_persistence in zip(logits, boxes, persistence, strict=True):
            padded_ids = _padded_ids(item_boxes, item_logits, item_persistence)
            stored_logits = item_logits.detach().to("cpu", torch.float16, copy=True)
            stored_persistence = item_persistence.detach().to("cpu", torch.float16, copy=True)
            if not bool(torch.isfinite(stored_logits).all()) or not bool(torch.isfinite(stored_persistence).all()):
                raise ValueError("stored detector outputs must be finite after float16 conversion")
            records.append({
                "logits": stored_logits,
                "persistence": stored_persistence,
                "padded_ids": padded_ids,
            })
        return records

    def close(self) -> None:
        if self.capture is not None:
            self.capture.close()
        self.capture = None
        self.model = None

    def __enter__(self) -> RTDETRExtractor:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
