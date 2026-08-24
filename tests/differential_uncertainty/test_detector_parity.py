from pathlib import Path
import subprocess
import sys

import pytest
import torch

from differential_uncertainty.extraction import (
    build_fixed_detector,
    load_frozen_detector,
)
from differential_uncertainty.persistence import Layer2Capture, batched_persistence


CHECKPOINT = Path(
    "/home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/"
    "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
)
GOLDEN = (
    Path(__file__).parent
    / "fixtures"
    / "rtdetrv2_r18_layer2_golden.pt"
)


@pytest.mark.skipif(
    not CHECKPOINT.is_file(),
    reason="local pretrained checkpoint absent",
)
def test_fixed_builder_matches_recorded_detector_outputs_exactly():
    golden = torch.load(GOLDEN, map_location="cpu", weights_only=True)
    assert golden["checkpoint_name"] == CHECKPOINT.name
    generator = torch.Generator().manual_seed(golden["input_seed"])
    sample_cpu = torch.rand((1, 3, 640, 640), generator=generator)
    if golden["device_type"] == "cuda" and not torch.cuda.is_available():
        pytest.skip("golden fixture was recorded on CUDA, which is unavailable")
    device = torch.device(golden["device_type"])
    model = load_frozen_detector(CHECKPOINT, device)
    with torch.inference_mode(), Layer2Capture(model.decoder, 2) as capture:
        outputs = model(sample_cpu.to(device))
        features, weight = capture.take()
        persistence = batched_persistence(
            weight,
            features.reshape(-1, features.shape[-1]),
        ).reshape(1, 300, 335)
    torch.testing.assert_close(
        outputs["pred_logits"].cpu(),
        golden["pred_logits"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        outputs["pred_boxes"].cpu(),
        golden["pred_boxes"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        persistence.cpu(),
        golden["layer_2_persistence"],
        rtol=0,
        atol=0,
    )


def test_fixed_builder_has_the_expected_checkpoint_shapes():
    model = build_fixed_detector()
    shapes = {
        name: tuple(value.shape)
        for name, value in model.state_dict().items()
    }
    assert shapes["backbone.conv1.conv1_1.conv.weight"] == (32, 3, 3, 3)
    assert shapes["encoder.input_proj.0.conv.weight"] == (256, 128, 1, 1)
    assert shapes["decoder.dec_score_head.2.weight"] == (80, 256)
    assert shapes["decoder.denoising_class_embed.weight"] == (81, 256)


def test_minimal_detector_import_surfaces_do_not_load_the_registry():
    script = """
import sys

import src

assert "src.core" not in sys.modules
from src import nn, zoo
from src.nn import backbone
from src.zoo import rtdetr

assert nn.__all__ == ["PResNet"]
assert backbone.__all__ == ["FrozenBatchNorm2d", "PResNet"]
assert zoo.__all__ == ["HybridEncoder", "RTDETR", "RTDETRTransformerv2"]
assert rtdetr.__all__ == ["HybridEncoder", "RTDETR", "RTDETRTransformerv2"]
assert "src.core" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
