import pytest
import torch
from PIL import Image
from torch import nn

import degradation_monitor.detector.model as extraction
from degradation_monitor.detector.model import checkpoint_state, load_frozen_detector, prepare_image


def test_checkpoint_state_accepts_ema_model_and_direct_tensor_states():
    tensor = torch.tensor([1.0])
    assert checkpoint_state({"ema": {"module": {"x": tensor}}}) == {"x": tensor}
    assert checkpoint_state({"model": {"x": tensor}}) == {"x": tensor}
    assert checkpoint_state({"x": tensor}) == {"x": tensor}


def test_load_frozen_detector_loads_cpu_state_and_freezes(tmp_path, monkeypatch):
    source = nn.Linear(3, 2)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": source.state_dict()}, checkpoint)
    monkeypatch.setattr(extraction, "build_fixed_detector", lambda: nn.Linear(3, 2))
    model = load_frozen_detector(checkpoint, torch.device("cpu"))
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_prepare_image_returns_rgb_float32_nchw_values():
    actual = prepare_image(Image.new("L", (5, 7), 128), (6, 10))
    assert actual.shape == (3, 6, 10)
    assert actual.dtype == torch.float32
    assert float(actual.min()) == pytest.approx(128 / 255)


def test_prepare_image_keeps_an_already_exact_rgb_image_open_until_tensorized():
    actual = prepare_image(Image.new("RGB", (640, 640), "red"), (640, 640))
    assert actual.shape == (3, 640, 640)
    assert actual.dtype == torch.float32
