"""Build the fixed RT-DETRv2-R18, load its frozen COCO checkpoint, and prepare images for it."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn
from torchvision.transforms.v2 import functional as vision

from .rtdetrv2.nn.backbone.presnet import PResNet
from .rtdetrv2.zoo.rtdetr.hybrid_encoder import HybridEncoder
from .rtdetrv2.zoo.rtdetr.rtdetr import RTDETR
from .rtdetrv2.zoo.rtdetr.rtdetrv2_decoder import RTDETRTransformerv2

IMAGE_SIZE = (640, 640)  # RT-DETRv2 fixes its input size in training and at inference


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

