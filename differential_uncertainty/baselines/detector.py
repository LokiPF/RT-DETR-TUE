"""Frozen RT-DETRv2 forward pass exposing logits, boxes and the pooled last backbone stage."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from ..extraction import load_frozen_detector, prepare_image

QUERY_COUNT = 300
CLASS_COUNT = 80
POOLED_DIM = 512


class DetectorTap:
    def __init__(self, checkpoint_path, device, image_size=(640, 640)):
        self.device = torch.device(device)
        self.image_size = tuple(image_size)
        self.model = load_frozen_detector(checkpoint_path, self.device)
        self._pooled = None
        self._handle = self.model.backbone.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, outputs):
        self._pooled = outputs[-1].mean(dim=(2, 3))

    def prepare(self, arrays) -> torch.Tensor:
        return torch.stack([prepare_image(Image.fromarray(a), self.image_size) for a in arrays])

    @torch.inference_mode()
    def forward(self, batch: torch.Tensor):
        outputs = self.model(batch.to(self.device, non_blocking=True))
        pooled, self._pooled = self._pooled, None
        if pooled is None:
            raise RuntimeError("the backbone hook did not run")
        logits = outputs["pred_logits"].float().cpu().numpy()
        boxes = outputs["pred_boxes"].float().cpu().numpy()
        pooled = pooled.float().cpu().numpy()
        n = batch.shape[0]
        if (logits.shape != (n, QUERY_COUNT, CLASS_COUNT) or boxes.shape != (n, QUERY_COUNT, 4)
                or pooled.shape != (n, POOLED_DIM)):
            raise ValueError("detector outputs do not have the expected shapes")
        if not (np.isfinite(logits).all() and np.isfinite(boxes).all() and np.isfinite(pooled).all()):
            raise ValueError("detector outputs must be finite")
        return logits, boxes, pooled

    def run(self, arrays, batch_size: int = 32):
        parts = [self.forward(self.prepare(arrays[s:s + batch_size]))
                 for s in range(0, len(arrays), batch_size)]
        return tuple(np.concatenate(values) for values in zip(*parts))

    def close(self) -> None:
        self._handle.remove()
        self.model = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
