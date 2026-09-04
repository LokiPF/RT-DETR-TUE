import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..misc.tue_dataclasses import CaptureGroup
from ..solver.clas_engine import build_frechet_means
from ..zoo.rtdetr.rtdetrv2_decoder_clas import RTDETRTransformerv2Clas


class _ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(5 * torch.eye(2))
        self.captures = CaptureGroup.from_model(self, [self.linear])

    def forward(self, inputs):
        self.captures.clear()
        tokens = inputs[:, None, :].expand(-1, 3, -1)
        logits = self.linear(tokens)
        self.captures.add(self.linear, tokens, logits)
        return logits.mean(dim=1)


class _ToyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = _ToyDecoder()

    def forward(self, inputs):
        return self.decoder(inputs)


class ClassificationCaptureTests(unittest.TestCase):
    def _make_model(self) -> RTDETRTransformerv2Clas:
        return RTDETRTransformerv2Clas(
            num_classes=3,
            hidden_dim=8,
            num_queries=2,
            feat_channels=[8],
            feat_strides=[8],
            num_levels=1,
            num_points=2,
            nhead=2,
            num_layers=2,
            dim_feedforward=16,
            dropout=0.0,
            eval_spatial_size=None,
            aux_loss=False,
        )

    def test_forward_captures_every_direct_linear_and_keeps_logits_api(self):
        model = self._make_model().eval()
        state_keys = tuple(model.state_dict())

        with torch.no_grad():
            logits = model([torch.randn(2, 8, 2, 2)])

        expected_names = {
            name
            for name, module in model.named_modules()
            if type(module) is nn.Linear
        }

        self.assertEqual(logits.shape, (2, 3))
        self.assertEqual(set(model.captures.selected.values()), expected_names)
        self.assertEqual(set(model.captures.data), expected_names)
        self.assertEqual(tuple(model.state_dict()), state_keys)

        for capture in model.captures.data.values():
            self.assertFalse(capture.input.requires_grad)
            self.assertFalse(capture.output.requires_grad)
            self.assertFalse(capture.weight.requires_grad)

    def test_training_backward_still_works(self):
        model = self._make_model().train()
        feature = torch.randn(2, 8, 2, 2, requires_grad=True)

        model([feature]).square().mean().backward()

        self.assertIsNotNone(feature.grad)
        self.assertTrue(torch.isfinite(feature.grad).all())

    def test_frechet_builder_uses_classification_labels_and_one_diagram_per_image(self):
        images = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )
        labels = torch.tensor([0, 1, 0])
        data_loader = DataLoader(TensorDataset(images, labels), batch_size=3)

        state = build_frechet_means(
            _ToyClassifier(),
            data_loader,
            torch.device("cpu"),
            min_confidence=0.9,
        )

        self.assertEqual(set(state), {"linear"})
        self.assertEqual(state["linear"]["counts"].tolist(), [1, 1])
        self.assertEqual(state["linear"]["means"].shape, (2, 3))
        self.assertEqual(state["linear"]["means"].device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
