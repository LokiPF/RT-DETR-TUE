from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

import differential_uncertainty.extraction as extraction
from differential_uncertainty.config import ExperimentConfig
from differential_uncertainty.extraction import (
    RTDETRExtractor, checkpoint_state, load_frozen_detector, prepare_image,
)


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


class _InnerDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])

    def forward(self, values):
        for layer in self.layers:
            values = layer(values)
        return values


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = _InnerDecoder()
        self.dec_score_head = nn.ModuleList([nn.Linear(2, 2, bias=False) for _ in range(3)])
        for head in self.dec_score_head:
            head.weight.data.copy_(torch.eye(2))

    def forward(self, values):
        return self.decoder(values)


class _Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = _Decoder()

    def forward(self, images):
        count = images.shape[0]
        features = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [3.0, 4.0]]]
        ).repeat(count, 1, 1)
        self.decoder(features)
        logits = torch.tensor(
            [[[0.0, 0.0], [2.0, 0.0], [2.0, 0.0]]]
        ).repeat(count, 1, 1)
        boxes = torch.tensor(
            [[[0.0] * 4, [1.0] * 4, [1.0] * 4]]
        ).repeat(count, 1, 1)
        return {"pred_logits": logits, "pred_boxes": boxes}


def test_extractor_discards_boxes_after_deriving_padded_ids(monkeypatch):
    config = ExperimentConfig(
        image_size=(4, 4), class_count=2, query_count=3,
        persistence_dim=3, bank_capacity=5,
    )
    monkeypatch.setattr(extraction, "load_frozen_detector", lambda *_: _Detector())
    with RTDETRExtractor(Path("ignored"), torch.device("cpu"), config) as extractor:
        records = extractor.extract_batch(torch.zeros(2, 3, 4, 4))
    assert len(records) == 2
    assert set(records[0]) == {"logits", "persistence", "padded_ids"}
    assert records[0]["logits"].dtype == torch.float16
    assert records[0]["persistence"].dtype == torch.float16
    assert records[0]["padded_ids"].dtype == torch.int64
    assert records[0]["padded_ids"].tolist() == [1, 2]
