"""Small stand-ins for the RT-DETRv2 backbone and the detector tap."""
import numpy as np
import torch
from torch import nn


class _Branch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        return torch.relu(self.norm(self.conv(x)))


class _Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branch2a = _Branch(channels)

    def forward(self, x):
        return self.branch2a(x)


class _Stage(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Conv2d(c_in, c_out, 3, stride, 1), _Block(c_out)])

    def forward(self, x):
        return self.blocks[1](torch.relu(self.blocks[0](x)))


class FakeBackbone(nn.Module):
    """640 x 640 input pooled to 16 x 16, then four stages of 2, 3, 4 and 5 channels at 16, 8, 4 and 2 cells."""

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        widths = (3, 2, 3, 4, 5)
        self.pool = nn.AvgPool2d(40)
        self.res_layers = nn.ModuleList([_Stage(widths[s], widths[s + 1], 1 if s == 0 else 2) for s in range(4)])
        self.eval()

    def forward(self, x):
        x = self.pool(x)
        outputs = []
        for stage in self.res_layers:
            x = stage(x)
            outputs.append(x)
        return outputs


class FakeTap:
    """Detector stand-in for the baselines `test` phase: fixed boxes, brightness-driven confidence."""

    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def run(self, arrays, batch_size=32):
        n = len(arrays)
        brightness = np.array([a.mean() / 255.0 for a in arrays])
        logits = np.full((n, 300, 80), -6.0)
        logits[:, :3, 0] = (4.0 * brightness)[:, None]
        boxes = np.full((n, 300, 4), 0.25)
        pooled = np.random.default_rng(0).normal(size=(n, 512)) + brightness[:, None]
        return logits, boxes, pooled
