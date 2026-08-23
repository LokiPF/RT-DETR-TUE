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
    backbone = PResNet(
        depth=18,
        variant="d",
        num_stages=4,
        return_idx=[1, 2, 3],
        act="relu",
        freeze_at=-1,
        freeze_norm=False,
        pretrained=False,
    )
    encoder = HybridEncoder(
        in_channels=[128, 256, 512],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act="gelu",
        use_encoder_idx=[2],
        num_encoder_layers=1,
        pe_temperature=10_000,
        expansion=0.5,
        depth_mult=1.0,
        act="silu",
        eval_spatial_size=spatial_size,
        version="v2",
    )
    decoder = RTDETRTransformerv2(
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        feat_channels=[256, 256, 256],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_points=[4, 4, 4],
        nhead=8,
        num_layers=3,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learn_query_content=False,
        eval_spatial_size=spatial_size,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        cross_attn_method="default",
        query_select_method="default",
    )
    return RTDETR(backbone=backbone, encoder=encoder, decoder=decoder)


def _tensor_state(value: Any, location: str) -> dict[str, Tensor]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{location} state must be a mapping of tensors")
    if not value:
        raise KeyError(f"{location} state is empty")
    if not all(
        isinstance(key, str) and isinstance(item, Tensor)
        for key, item in value.items()
    ):
        raise TypeError(f"{location} state must contain only named tensors")
    return dict(value)


def checkpoint_state(checkpoint: object) -> dict[str, Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    if "ema" in checkpoint:
        ema = checkpoint["ema"]
        if not isinstance(ema, Mapping):
            raise TypeError("checkpoint ema must be a mapping")
        if "module" not in ema:
            raise KeyError("checkpoint ema has no module state")
        return _tensor_state(ema["module"], "checkpoint ema.module")
    if "model" in checkpoint:
        return _tensor_state(checkpoint["model"], "checkpoint model")
    if checkpoint and all(isinstance(item, Tensor) for item in checkpoint.values()):
        return _tensor_state(checkpoint, "checkpoint direct")
    if not checkpoint:
        raise KeyError("checkpoint state is empty")
    if any(isinstance(item, Tensor) for item in checkpoint.values()):
        raise TypeError("direct checkpoint state must contain only named tensors")
    raise KeyError("checkpoint has neither ema.module nor model state")


def load_frozen_detector(
    checkpoint_path: str | Path,
    device: torch.device,
) -> nn.Module:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model = build_fixed_detector()
    incompatible = model.load_state_dict(
        checkpoint_state(checkpoint),
        strict=False,
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.to(device).eval().requires_grad_(False)
    return model


def prepare_image(
    image: Image.Image,
    image_size: tuple[int, int],
) -> Tensor:
    rgb = image.convert("RGB")
    resized = vision.resize(rgb, list(image_size), antialias=True)
    return vision.pil_to_tensor(resized).to(torch.float32).div_(255.0)


def record_from_outputs(
    image_id: str,
    severity: int,
    logits: Tensor,
    boxes: Tensor,
    persistence: Tensor,
) -> dict[str, object]:
    return {
        "image_id": image_id,
        "severity": int(severity),
        "logits": logits.detach().to(
            device="cpu",
            dtype=torch.float16,
            copy=True,
        ),
        "boxes": boxes.detach().to(
            device="cpu",
            dtype=torch.float32,
            copy=True,
        ),
        "persistence": persistence.detach().to(
            device="cpu",
            dtype=torch.float16,
            copy=True,
        ),
    }


class RTDETRExtractor:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: torch.device,
        config: ExperimentConfig = FIXED_CONFIG,
    ) -> None:
        self.device = device
        self.config = config
        self.model = load_frozen_detector(checkpoint_path, device)
        self.capture = Layer2Capture(
            self.model.decoder,
            layer=config.persistence_layer,
        )

    @torch.inference_mode()
    def extract_batch(
        self,
        identities: list[tuple[str, int]],
        samples: Tensor,
    ) -> list[dict[str, object]]:
        if not isinstance(samples, Tensor):
            raise TypeError("samples must be a Tensor")
        if samples.ndim != 4:
            raise ValueError("samples must be a 4D NCHW tensor")
        if samples.shape[0] == 0:
            raise ValueError("sample batch must be nonempty")
        if samples.shape[1] != 3:
            raise ValueError("samples must have 3 channels")
        spatial_size = tuple(samples.shape[2:])
        if spatial_size != self.config.image_size:
            raise ValueError(
                f"sample spatial size {spatial_size} does not match "
                f"configured {self.config.image_size}"
            )
        if samples.dtype != torch.float32:
            raise ValueError(
                "samples must have dtype torch.float32, "
                f"got {samples.dtype}"
            )
        identity_count = len(identities)
        sample_count = samples.shape[0]
        if identity_count != sample_count:
            raise ValueError(
                f"received {identity_count} identities for sample batch "
                f"{sample_count}"
            )

        outputs = self.model(samples.to(self.device))
        if not isinstance(outputs, Mapping):
            raise TypeError("detector output must be a mapping")
        try:
            logits = outputs["pred_logits"]
            boxes = outputs["pred_boxes"]
        except KeyError as error:
            raise KeyError(
                "detector output must contain pred_logits and pred_boxes"
            ) from error
        if not isinstance(logits, Tensor) or not isinstance(boxes, Tensor):
            raise TypeError("detector logits and boxes must be tensors")
        if logits.ndim == 0 or boxes.ndim == 0:
            raise RuntimeError("detector logits and boxes need batch dimensions")
        output_count = logits.shape[0]
        if boxes.shape[0] != output_count:
            raise RuntimeError(
                "detector logits and boxes have different batch sizes: "
                f"{output_count} and {boxes.shape[0]}"
            )
        if output_count != identity_count:
            raise RuntimeError(
                f"detector output batch {output_count} does not match "
                f"identities {identity_count}"
            )
        expected_logits = (
            identity_count,
            self.config.query_count,
            self.config.class_count,
        )
        expected_boxes = (identity_count, self.config.query_count, 4)
        if (
            tuple(logits.shape) != expected_logits
            or tuple(boxes.shape) != expected_boxes
        ):
            raise RuntimeError(
                "unexpected detector output shapes "
                f"logits={tuple(logits.shape)}, boxes={tuple(boxes.shape)}; "
                f"expected {expected_logits} and {expected_boxes}"
            )

        features, weight = self.capture.take()
        if features.ndim != 3:
            raise RuntimeError(
                "captured decoder features must have shape "
                "(batch, queries, hidden)"
            )
        flat_features = features.reshape(-1, features.shape[-1])
        diagrams = batched_persistence(weight, flat_features).reshape(
            features.shape[0],
            features.shape[1],
            -1,
        )
        expected = (
            identity_count,
            self.config.query_count,
            self.config.persistence_dim,
        )
        if tuple(diagrams.shape) != expected:
            raise RuntimeError(
                f"unexpected persistence shape {tuple(diagrams.shape)}; "
                f"expected {expected}"
            )

        return [
            record_from_outputs(
                image_id,
                severity,
                item_logits,
                item_boxes,
                item_persistence,
            )
            for (image_id, severity), item_logits, item_boxes, item_persistence
            in zip(identities, logits, boxes, diagrams, strict=True)
        ]

    def close(self) -> None:
        self.capture.close()

    def __enter__(self) -> RTDETRExtractor:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
