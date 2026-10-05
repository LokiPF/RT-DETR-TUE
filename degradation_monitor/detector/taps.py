"""Forward hooks on the frozen RT-DETRv2: logits, boxes and the pooled last stage; the hidden layers the activation monitors read; and the early-channel inputs our method reads."""
from __future__ import annotations

from functools import partial

import numpy as np
import torch
from PIL import Image

from .model import load_frozen_detector, prepare_image

QUERY_COUNT = 300
POOLED_DIM = 512
DECODER_SHAPE = (QUERY_COUNT, 256)
STAGE_COUNT = 5  # C1 (stem output after max-pooling) and the four residual stages


class DetectorTap:
    def __init__(self, checkpoint_path, device, image_size=(640, 640), hidden=False):
        self.device = torch.device(device)
        self.image_size = tuple(image_size)
        self.model = load_frozen_detector(checkpoint_path, self.device)
        self._reset()
        self._handles = [self.model.backbone.register_forward_hook(self._capture)]
        self.hidden = hidden
        if hidden:
            stages = self.model.backbone.res_layers
            self._handles.append(stages[0].register_forward_pre_hook(self._capture_stem))
            self._handles += [stage.register_forward_hook(partial(self._capture_stage, index + 1))
                              for index, stage in enumerate(stages)]
            self._handles.append(self.model.decoder.decoder.layers[-1].register_forward_hook(self._capture_decoder))
            self._handles.append(self.model.encoder.register_forward_hook(self._capture_encoder))

    def _reset(self):
        self._pooled = self._decoder = self._encoder = None
        self._stages = [None] * STAGE_COUNT

    def _capture(self, _module, _inputs, outputs):
        self._pooled = outputs[-1].mean(dim=(2, 3))

    def _capture_stem(self, _module, inputs):
        self._stages[0] = inputs[0]

    def _capture_stage(self, index, _module, _inputs, output):
        self._stages[index] = output

    def _capture_decoder(self, _module, _inputs, output):
        self._decoder = output

    def _capture_encoder(self, _module, _inputs, outputs):
        self._encoder = list(outputs)

    def prepare(self, arrays) -> torch.Tensor:
        return torch.stack([prepare_image(Image.fromarray(a), self.image_size) for a in arrays])

    @torch.inference_mode()
    def forward(self, batch: torch.Tensor):
        outputs = self.model(batch.to(self.device, non_blocking=True))
        pooled = self._pooled
        self._reset()
        if pooled is None:
            raise RuntimeError("the backbone hook did not run")
        logits = outputs["pred_logits"].float().cpu().numpy()
        boxes = outputs["pred_boxes"].float().cpu().numpy()
        pooled = pooled.float().cpu().numpy()
        n = batch.shape[0]
        if (logits.ndim != 3 or logits.shape[:2] != (n, QUERY_COUNT) or boxes.shape != (n, QUERY_COUNT, 4)
                or pooled.shape != (n, POOLED_DIM)):
            raise ValueError("detector outputs do not have the expected shapes")
        if not (np.isfinite(logits).all() and np.isfinite(boxes).all() and np.isfinite(pooled).all()):
            raise ValueError("detector outputs must be finite")
        return logits, boxes, pooled

    @torch.inference_mode()
    def forward_hidden(self, batch: torch.Tensor) -> dict:
        """Decoder queries (n, 300, 256), encoder output maps and backbone stages C1-C5, float32 on the device."""
        if not self.hidden:
            raise RuntimeError("create the tap with hidden=True to read hidden layers")
        self.model(batch.to(self.device, non_blocking=True))
        decoder, encoder, stages = self._decoder, self._encoder, self._stages
        self._reset()
        if decoder is None or encoder is None or any(s is None for s in stages):
            raise RuntimeError("a hidden-layer hook did not run")
        n = batch.shape[0]
        if tuple(decoder.shape) != (n, *DECODER_SHAPE):
            raise ValueError(f"decoder queries have shape {tuple(decoder.shape)}, expected {(n, *DECODER_SHAPE)}")
        hidden = {"decoder": decoder.float(), "encoder": [e.float() for e in encoder],
                  "backbone": [s.float() for s in stages]}
        if not all(bool(torch.isfinite(t).all()) for t in [hidden["decoder"], *hidden["encoder"], *hidden["backbone"]]):
            raise ValueError("hidden activations must be finite")
        return hidden

    def run(self, arrays, batch_size: int = 32):
        parts = [self.forward(self.prepare(arrays[s:s + batch_size]))
                 for s in range(0, len(arrays), batch_size)]
        return tuple(np.concatenate(values) for values in zip(*parts))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self.model = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


EARLY_STAGES = 4
EARLY_LAYERS = tuple(f"res_layers.{s}.blocks.1.branch2a.conv" for s in range(EARLY_STAGES))


class EarlyChannelTaps:
    """The inputs of the first conv in the second block of each backbone stage: the four post-ReLU maps our method reads.

    For a 640 x 640 image they have shapes (64, 160, 160), (128, 80, 80), (256, 40, 40) and (512, 20, 20).
    """

    def __init__(self, backbone: torch.nn.Module):
        self.backbone = backbone
        self.device = next(backbone.parameters()).device
        convs = [backbone.res_layers[s].blocks[1].branch2a.conv for s in range(EARLY_STAGES)]
        self._inputs: list = [None] * EARLY_STAGES
        self._handles = [conv.register_forward_pre_hook(partial(self._capture, index))
                         for index, conv in enumerate(convs)]

    def _capture(self, index, _module, inputs):
        self._inputs[index] = inputs[0]

    @torch.inference_mode()
    def __call__(self, batch: torch.Tensor) -> list[torch.Tensor]:
        self._inputs = [None] * EARLY_STAGES
        self.backbone(batch.to(self.device))
        if any(value is None for value in self._inputs):
            raise RuntimeError("an early-channel input was not captured")
        return [value.float() for value in self._inputs]

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
