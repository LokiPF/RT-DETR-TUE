"""Inputs of one stride-1 3x3 conv per backbone stage of the frozen RT-DETRv2-R18, with their folded kernels."""
from __future__ import annotations

from functools import partial

import torch

from .graph import folded_kernel

STAGES = 4
LAYER_NAMES = tuple(f"res_layers.{s}.blocks.1.branch2a.conv" for s in range(STAGES))


class ConvInputs:
    """Captures the input of the first conv in the second block of each backbone stage."""

    def __init__(self, backbone: torch.nn.Module):
        self.backbone = backbone
        branches = [backbone.res_layers[s].blocks[1].branch2a for s in range(STAGES)]
        self.kernels = [folded_kernel(branch.conv, branch.norm) for branch in branches]
        self._inputs: list = [None] * STAGES
        self._handles = [branch.conv.register_forward_pre_hook(partial(self._capture, index))
                         for index, branch in enumerate(branches)]

    def _capture(self, index, _module, inputs):
        self._inputs[index] = inputs[0]

    @torch.inference_mode()
    def __call__(self, batch: torch.Tensor) -> list[torch.Tensor]:
        self._inputs = [None] * STAGES
        self.backbone(batch)
        if any(value is None for value in self._inputs):
            raise RuntimeError("a pilot conv input was not captured")
        return [value.float() for value in self._inputs]

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
