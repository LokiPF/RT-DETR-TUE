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
