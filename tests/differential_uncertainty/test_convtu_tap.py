from pathlib import Path

import pytest
import torch

from fakes import FakeBackbone
from differential_uncertainty.convtu.graph import folded_kernel
from differential_uncertainty.convtu.tap import ConvInputs, LAYER_NAMES
from degradation_monitor.detector.model import load_frozen_detector

CHECKPOINT = Path("/home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth")


def test_conv_inputs_are_the_tensors_entering_each_pilot_conv():
    backbone = FakeBackbone()
    batch = torch.rand(2, 3, 640, 640)
    with ConvInputs(backbone) as taps:
        inputs = taps(batch)
    with torch.no_grad():
        x = backbone.pool(batch)
        expected = []
        for stage in backbone.res_layers:
            entering = torch.relu(stage.blocks[0](x))
            expected.append(entering)
            x = stage.blocks[1](entering)
    assert [tuple(t.shape) for t in inputs] == [(2, 2, 16, 16), (2, 3, 8, 8), (2, 4, 4, 4), (2, 5, 2, 2)]
    for got, want in zip(inputs, expected):
        assert torch.allclose(got, want)


def test_kernels_are_folded_with_each_branch_norm():
    backbone = FakeBackbone()
    with ConvInputs(backbone) as taps:
        for stage, kernel in zip(backbone.res_layers, taps.kernels):
            branch = stage.blocks[1].branch2a
            assert torch.equal(kernel, folded_kernel(branch.conv, branch.norm))


def test_closing_removes_the_hooks():
    backbone = FakeBackbone()
    taps = ConvInputs(backbone)
    taps.close()
    assert all(len(stage.blocks[1].branch2a.conv._forward_pre_hooks) == 0 for stage in backbone.res_layers)
    assert LAYER_NAMES[0] == "res_layers.0.blocks.1.branch2a.conv"


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="needs the RT-DETRv2-R18 checkpoint")
def test_the_real_backbone_gives_the_four_pilot_layers():
    model = load_frozen_detector(CHECKPOINT, torch.device("cpu"))
    with ConvInputs(model.backbone) as taps:
        inputs = taps(torch.rand(1, 3, 640, 640))
        kernel_shapes = [tuple(k.shape) for k in taps.kernels]
    assert [tuple(t.shape) for t in inputs] == [(1, 64, 160, 160), (1, 128, 80, 80), (1, 256, 40, 40), (1, 512, 20, 20)]
    assert kernel_shapes == [(64, 64, 3, 3), (128, 128, 3, 3), (256, 256, 3, 3), (512, 512, 3, 3)]
