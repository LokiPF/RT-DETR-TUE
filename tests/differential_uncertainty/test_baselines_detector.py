import numpy as np
import pytest
import torch
from torch import nn

from differential_uncertainty.baselines import detector


class FakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Identity()

    def forward(self, images):
        n = images.shape[0]
        self.backbone([torch.zeros(n, 256, 4, 4), torch.ones(n, 512, 2, 2) * images.mean()])
        return {"pred_logits": torch.zeros(n, 300, 80), "pred_boxes": torch.full((n, 300, 4), 0.5)}


def test_tap_returns_logits_boxes_and_pooled_last_backbone_stage(monkeypatch):
    monkeypatch.setattr(detector, "load_frozen_detector", lambda _p, _d: FakeDetector())
    arrays = [np.full((20, 30, 3), 255, np.uint8), np.zeros((10, 10, 3), np.uint8)]
    with detector.DetectorTap("unused.pth", "cpu", image_size=(16, 16)) as tap:
        logits, boxes, pooled = tap.run(arrays, batch_size=1)
    assert logits.shape == (2, 300, 80) and boxes.shape == (2, 300, 4) and pooled.shape == (2, 512)
    assert pooled[0, 0] == pytest.approx(1.0) and pooled[1, 0] == pytest.approx(0.0)


def test_tap_rejects_non_finite_outputs(monkeypatch):
    class Broken(FakeDetector):
        def forward(self, images):
            out = super().forward(images)
            out["pred_logits"][0, 0, 0] = float("nan")
            return out

    monkeypatch.setattr(detector, "load_frozen_detector", lambda _p, _d: Broken())
    with detector.DetectorTap("unused.pth", "cpu", image_size=(8, 8)) as tap:
        with pytest.raises(ValueError, match="finite"):
            tap.run([np.zeros((8, 8, 3), np.uint8)])


class FakeBackbone(nn.Module):
    """C1 (input of the first stage) = level + 1, and stage i adds 1 each time: C1..C5 = level + 1, 1, 2, 3, 4."""

    def __init__(self):
        super().__init__()
        self.res_layers = nn.ModuleList([nn.Identity() for _ in range(4)])

    def forward(self, images):
        n = images.shape[0]
        x = images.mean(dim=(1, 2, 3)).view(n, 1, 1, 1).expand(n, 4, 4, 4).clone()
        for stage in self.res_layers:
            x = stage(x + 1.0)
        return [torch.zeros(n, 256, 4, 4), torch.ones(n, 512, 2, 2)]


class FakeHiddenDetector(nn.Module):
    """Last decoder layer output = 3 * level + 3; first encoder map = level (level = mean image value)."""

    def __init__(self):
        super().__init__()
        self.backbone = FakeBackbone()
        self.encoder = nn.Identity()
        self.decoder = nn.Module()
        self.decoder.decoder = nn.Module()
        self.decoder.decoder.layers = nn.ModuleList([nn.Identity() for _ in range(3)])

    def forward(self, images):
        n = images.shape[0]
        level = images.mean(dim=(1, 2, 3))
        self.backbone(images)
        self.encoder([level.view(n, 1, 1, 1).expand(n, 256, 4, 4).clone(),
                      torch.zeros(n, 256, 2, 2), torch.zeros(n, 256, 1, 1)])
        queries = torch.zeros(n, 300, 256)
        for index, layer in enumerate(self.decoder.decoder.layers):
            queries = layer(queries + index + level.view(n, 1, 1))
        return {"pred_logits": torch.zeros(n, 300, 80), "pred_boxes": torch.full((n, 300, 4), 0.5)}


def test_hidden_tap_returns_decoder_encoder_and_backbone_stages(monkeypatch):
    monkeypatch.setattr(detector, "load_frozen_detector", lambda _p, _d: FakeHiddenDetector())
    arrays = [np.full((20, 30, 3), 255, np.uint8), np.zeros((10, 10, 3), np.uint8)]
    with detector.DetectorTap("unused.pth", "cpu", image_size=(16, 16), hidden=True) as tap:
        hidden = tap.forward_hidden(tap.prepare(arrays))
        logits, _, _ = tap.run(arrays, batch_size=2)   # the normal path still works next to the extra hooks
    decoder, encoder, backbone = hidden["decoder"], hidden["encoder"], hidden["backbone"]
    assert tuple(decoder.shape) == (2, 300, 256)
    assert [tuple(e.shape) for e in encoder] == [(2, 256, 4, 4), (2, 256, 2, 2), (2, 256, 1, 1)]
    assert decoder[0, 0, 0].item() == pytest.approx(6.0) and decoder[1, 0, 0].item() == pytest.approx(3.0)
    assert encoder[0][0, 0, 0, 0].item() == pytest.approx(1.0)
    assert [round(b[0, 0, 0, 0].item(), 4) for b in backbone] == [2.0, 2.0, 3.0, 4.0, 5.0]
    assert logits.shape == (2, 300, 80)


def test_hidden_layers_need_a_hidden_tap(monkeypatch):
    monkeypatch.setattr(detector, "load_frozen_detector", lambda _p, _d: FakeHiddenDetector())
    with detector.DetectorTap("unused.pth", "cpu", image_size=(8, 8)) as tap:
        with pytest.raises(RuntimeError, match="hidden=True"):
            tap.forward_hidden(tap.prepare([np.zeros((8, 8, 3), np.uint8)]))
