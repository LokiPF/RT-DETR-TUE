from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import Tensor, nn

import differential_uncertainty.extraction as extraction
from differential_uncertainty.config import ExperimentConfig
from differential_uncertainty.extraction import (
    RTDETRExtractor,
    build_fixed_detector,
    checkpoint_state,
    load_frozen_detector,
    prepare_image,
    record_from_outputs,
)
from src.core import YAMLConfig


def test_explicit_builder_has_the_same_state_contract_as_the_yaml_model():
    legacy = YAMLConfig(
        "configs/scene_uncertainty/rtdetrv2_r18vd_coco.yml"
    ).model
    explicit = build_fixed_detector()

    legacy_shapes = {
        name: tuple(value.shape) for name, value in legacy.state_dict().items()
    }
    explicit_shapes = {
        name: tuple(value.shape) for name, value in explicit.state_dict().items()
    }

    assert explicit_shapes == legacy_shapes


def test_checkpoint_state_accepts_the_three_historical_shapes():
    tensor = torch.tensor([1.0])

    assert checkpoint_state({"ema": {"module": {"x": tensor}}}) == {
        "x": tensor
    }
    assert checkpoint_state({"model": {"x": tensor}}) == {"x": tensor}
    assert checkpoint_state({"x": tensor}) == {"x": tensor}


def test_explicit_builder_passes_every_fixed_architecture_setting(monkeypatch):
    calls = {}

    def constructor(name):
        def construct(**kwargs):
            calls[name] = kwargs
            return name

        return construct

    monkeypatch.setattr(extraction, "PResNet", constructor("backbone"))
    monkeypatch.setattr(extraction, "HybridEncoder", constructor("encoder"))
    monkeypatch.setattr(extraction, "RTDETRTransformerv2", constructor("decoder"))
    monkeypatch.setattr(extraction, "RTDETR", constructor("model"))

    assert build_fixed_detector() == "model"
    assert calls == {
        "backbone": {
            "depth": 18,
            "variant": "d",
            "num_stages": 4,
            "return_idx": [1, 2, 3],
            "act": "relu",
            "freeze_at": -1,
            "freeze_norm": False,
            "pretrained": False,
        },
        "encoder": {
            "in_channels": [128, 256, 512],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 256,
            "nhead": 8,
            "dim_feedforward": 1024,
            "dropout": 0.0,
            "enc_act": "gelu",
            "use_encoder_idx": [2],
            "num_encoder_layers": 1,
            "pe_temperature": 10_000,
            "expansion": 0.5,
            "depth_mult": 1.0,
            "act": "silu",
            "eval_spatial_size": [640, 640],
            "version": "v2",
        },
        "decoder": {
            "num_classes": 80,
            "hidden_dim": 256,
            "num_queries": 300,
            "feat_channels": [256, 256, 256],
            "feat_strides": [8, 16, 32],
            "num_levels": 3,
            "num_points": [4, 4, 4],
            "nhead": 8,
            "num_layers": 3,
            "dim_feedforward": 1024,
            "dropout": 0.0,
            "activation": "relu",
            "num_denoising": 100,
            "label_noise_ratio": 0.5,
            "box_noise_scale": 1.0,
            "learn_query_content": False,
            "eval_spatial_size": [640, 640],
            "eval_idx": -1,
            "eps": 1e-2,
            "aux_loss": True,
            "cross_attn_method": "default",
            "query_select_method": "default",
        },
        "model": {
            "backbone": "backbone",
            "encoder": "encoder",
            "decoder": "decoder",
        },
    }


@pytest.mark.parametrize(
    ("checkpoint", "error", "message"),
    [
        (None, TypeError, "checkpoint must be a mapping"),
        ({}, KeyError, "checkpoint state is empty"),
        ({"ema": None}, TypeError, "ema must be a mapping"),
        ({"ema": {}}, KeyError, "ema has no module"),
        ({"ema": {"module": {}}}, KeyError, "ema.module state is empty"),
        ({"ema": {"module": {"x": 1}}}, TypeError, "only named tensors"),
        ({"model": []}, TypeError, "model state must be a mapping"),
        ({"model": {}}, KeyError, "model state is empty"),
        ({"x": torch.ones(1), "epoch": 2}, TypeError, "only named tensors"),
        ({"optimizer": {}}, KeyError, "neither ema.module nor model"),
    ],
)
def test_checkpoint_state_rejects_every_other_shape(checkpoint, error, message):
    with pytest.raises(error, match=message):
        checkpoint_state(checkpoint)


def test_load_frozen_detector_loads_on_cpu_then_freezes_and_moves(
    tmp_path: Path,
    monkeypatch,
):
    source = nn.Linear(3, 2)
    checkpoint = tmp_path / "detector.pth"
    torch.save({"model": source.state_dict()}, checkpoint)
    loaded_with = {}
    original_load = torch.load

    def observed_load(*args, **kwargs):
        loaded_with.update(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(extraction, "build_fixed_detector", lambda: nn.Linear(3, 2))
    monkeypatch.setattr(extraction.torch, "load", observed_load)

    model = load_frozen_detector(checkpoint, torch.device("cpu"))

    assert loaded_with["map_location"] == "cpu"
    assert loaded_with["weights_only"] is True
    assert model.training is False
    assert {parameter.device.type for parameter in model.parameters()} == {"cpu"}
    assert all(not parameter.requires_grad for parameter in model.parameters())
    for actual, expected in zip(model.parameters(), source.parameters()):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("kind", ["missing", "unexpected", "wrong_shape"])
def test_load_frozen_detector_rejects_incompatible_state(
    kind,
    tmp_path: Path,
    monkeypatch,
):
    model = nn.Linear(3, 2)
    state = dict(model.state_dict())
    if kind == "missing":
        del state["bias"]
    elif kind == "unexpected":
        state["extra"] = torch.ones(1)
    else:
        state["weight"] = torch.ones(3, 2)
    checkpoint = tmp_path / f"{kind}.pth"
    torch.save({"model": state}, checkpoint)
    monkeypatch.setattr(extraction, "build_fixed_detector", lambda: nn.Linear(3, 2))

    with pytest.raises(RuntimeError, match="checkpoint mismatch|size mismatch"):
        load_frozen_detector(checkpoint, torch.device("cpu"))


def test_prepare_image_is_rgb_exact_size_float32_in_zero_one_range():
    image = Image.new("L", (13, 7), 128)

    tensor = prepare_image(image, (640, 640))

    assert tensor.shape == (3, 640, 640)
    assert tensor.dtype == torch.float32
    assert float(tensor.min()) == pytest.approx(128 / 255)
    assert float(tensor.max()) == pytest.approx(128 / 255)
    torch.testing.assert_close(tensor[0], tensor[1], rtol=0, atol=0)
    torch.testing.assert_close(tensor[1], tensor[2], rtol=0, atol=0)


def test_prepare_image_honors_height_width_order_exactly():
    tensor = prepare_image(Image.new("RGB", (5, 9), "white"), (6, 10))

    assert tensor.shape == (3, 6, 10)


def test_record_storage_matches_the_cache_dtypes_and_owns_cpu_values():
    logits = torch.randn(3, 2, dtype=torch.float16, requires_grad=True)
    boxes = torch.randn(3, 4, requires_grad=True)
    persistence = torch.randn(
        3,
        5,
        dtype=torch.float16,
        requires_grad=True,
    )
    originals = {
        "logits": logits.detach().clone(),
        "boxes": boxes.detach().clone(),
        "persistence": persistence.detach().clone(),
    }

    record = record_from_outputs("scene", 3, logits, boxes, persistence)

    assert record["image_id"] == "scene"
    assert record["severity"] == 3
    assert record["logits"].dtype == torch.float16
    assert record["boxes"].dtype == torch.float32
    assert record["persistence"].dtype == torch.float16
    assert all(
        record[name].device.type == "cpu"
        for name in ("logits", "boxes", "persistence")
    )
    assert all(
        not record[name].requires_grad
        for name in ("logits", "boxes", "persistence")
    )
    with torch.no_grad():
        logits.add_(1)
        boxes.add_(1)
        persistence.add_(1)
    for name, expected in originals.items():
        torch.testing.assert_close(record[name], expected, rtol=0, atol=0)


class _FakeCapture:
    def __init__(self, features: Tensor, weight: Tensor):
        self.features = features
        self.weight = weight
        self.take_count = 0
        self.close_count = 0

    def take(self):
        self.take_count += 1
        return self.features, self.weight

    def close(self):
        self.close_count += 1


class _FakeDetector(nn.Module):
    def __init__(
        self,
        output_batch: int = 2,
        output_queries: int = 10,
        fail: bool = False,
    ):
        super().__init__()
        self.decoder = nn.Identity()
        self.output_batch = output_batch
        self.output_queries = output_queries
        self.fail = fail
        self.inference_mode_during_forward = None
        self.last_samples = None

    def forward(self, samples):
        self.inference_mode_during_forward = torch.is_inference_mode_enabled()
        self.last_samples = samples
        if self.fail:
            raise RuntimeError("forward failed")
        logits = torch.arange(
            self.output_batch * self.output_queries * 2,
            dtype=torch.float32,
            device=samples.device,
        ).reshape(self.output_batch, self.output_queries, 2)
        boxes = torch.arange(
            self.output_batch * self.output_queries * 4,
            dtype=torch.float32,
            device=samples.device,
        ).reshape(self.output_batch, self.output_queries, 4)
        return {"pred_logits": logits, "pred_boxes": boxes}


def _extractor(
    monkeypatch,
    *,
    output_batch: int = 2,
    output_queries: int = 10,
    feature_batch: int = 2,
    fail: bool = False,
):
    model = _FakeDetector(
        output_batch=output_batch,
        output_queries=output_queries,
        fail=fail,
    )
    features = torch.arange(feature_batch * 10 * 4, dtype=torch.float32).reshape(
        feature_batch, 10, 4
    )
    weight = torch.tensor(
        [[1.0, -2.0, 3.0, -4.0], [-0.5, 1.5, -2.5, 3.5]]
    )
    capture = _FakeCapture(features, weight)
    monkeypatch.setattr(extraction, "load_frozen_detector", lambda *_args: model)
    monkeypatch.setattr(
        extraction,
        "Layer2Capture",
        lambda *_args, **_kwargs: capture,
    )
    config = ExperimentConfig.for_tests(
        image_size=(8, 8),
        class_count=2,
        query_count=10,
        persistence_dim=5,
        persistence_layer=1,
    )
    instance = RTDETRExtractor("unused.pth", torch.device("cpu"), config)
    return instance, model, capture


def test_extractor_returns_ordered_records_and_runs_in_inference_mode(monkeypatch):
    extractor, model, capture = _extractor(monkeypatch)
    samples = torch.zeros(2, 3, 8, 8)

    records = extractor.extract_batch([("b", 4), ("a", 0)], samples)

    assert [(record["image_id"], record["severity"]) for record in records] == [
        ("b", 4),
        ("a", 0),
    ]
    assert all(record["logits"].shape == (10, 2) for record in records)
    assert all(record["boxes"].shape == (10, 4) for record in records)
    assert all(record["persistence"].shape == (10, 5) for record in records)
    assert model.inference_mode_during_forward is True
    assert model.last_samples.device.type == "cpu"
    assert capture.take_count == 1


def test_extractor_rejects_identity_and_sample_batch_mismatch_before_forward(
    monkeypatch,
):
    extractor, model, capture = _extractor(monkeypatch)

    with pytest.raises(ValueError, match="2 identities.*sample batch 1"):
        extractor.extract_batch(
            [("a", 0), ("b", 1)],
            torch.zeros(1, 3, 8, 8),
        )

    assert model.last_samples is None
    assert capture.take_count == 0


def test_extractor_rejects_a_non_tensor_before_model_execution(monkeypatch):
    extractor, model, capture = _extractor(monkeypatch)

    with pytest.raises(TypeError, match="samples must be a Tensor"):
        extractor.extract_batch([("a", 0)], [[[1.0]]])

    assert model.last_samples is None
    assert capture.take_count == 0


def test_extractor_moves_samples_without_requiring_the_input_device(monkeypatch):
    extractor, model, capture = _extractor(monkeypatch, fail=True)
    extractor.device = torch.device("meta")

    with pytest.raises(RuntimeError, match="forward failed"):
        extractor.extract_batch(
            [("a", 0), ("b", 0)],
            torch.zeros(2, 3, 8, 8),
        )

    assert model.last_samples.device.type == "meta"
    assert capture.take_count == 0


@pytest.mark.parametrize(
    ("samples", "identities", "message"),
    [
        (torch.empty(0, 3, 8, 8), [], "sample batch must be nonempty"),
        (
            torch.zeros(3, 8, 8),
            [("a", 0), ("b", 0), ("c", 0)],
            "samples must be a 4D NCHW tensor",
        ),
        (
            torch.zeros(2, 1, 8, 8),
            [("a", 0), ("b", 0)],
            "samples must have 3 channels",
        ),
        (
            torch.zeros(2, 3, 7, 8),
            [("a", 0), ("b", 0)],
            r"sample spatial size \(7, 8\).*configured \(8, 8\)",
        ),
        (
            torch.zeros(2, 3, 8, 8, dtype=torch.int64),
            [("a", 0), ("b", 0)],
            "samples must use a floating dtype",
        ),
    ],
)
def test_extractor_rejects_invalid_samples_before_model_execution(
    samples,
    identities,
    message,
    monkeypatch,
):
    extractor, model, capture = _extractor(monkeypatch)

    with pytest.raises(ValueError, match=message):
        extractor.extract_batch(identities, samples)

    assert model.last_samples is None
    assert capture.take_count == 0


@pytest.mark.parametrize(
    ("output_batch", "feature_batch", "message"),
    [
        (1, 2, "detector output batch 1.*identities 2"),
        (2, 1, r"unexpected persistence shape \(1, 10, 5\)"),
    ],
)
def test_extractor_rejects_output_and_persistence_batch_mismatches(
    output_batch,
    feature_batch,
    message,
    monkeypatch,
):
    extractor, _model, _capture = _extractor(
        monkeypatch,
        output_batch=output_batch,
        feature_batch=feature_batch,
    )

    with pytest.raises(RuntimeError, match=message):
        extractor.extract_batch(
            [("a", 0), ("b", 1)],
            torch.zeros(2, 3, 8, 8),
        )


def test_extractor_enforces_configured_query_and_persistence_dimensions(
    monkeypatch,
):
    extractor, _model, _capture = _extractor(monkeypatch)
    extractor.config = ExperimentConfig.for_tests(
        image_size=(8, 8),
        class_count=2,
        query_count=10,
        persistence_dim=6,
        persistence_layer=1,
    )

    with pytest.raises(
        RuntimeError,
        match=r"unexpected persistence shape \(2, 10, 5\); expected \(2, 10, 6\)",
    ):
        extractor.extract_batch(
            [("a", 0), ("b", 1)],
            torch.zeros(2, 3, 8, 8),
        )


def test_extractor_rejects_detector_query_shape_mismatch(monkeypatch):
    extractor, _model, _capture = _extractor(
        monkeypatch,
        output_queries=9,
    )

    with pytest.raises(
        RuntimeError,
        match=r"unexpected detector output shapes .* expected \(2, 10, 2\)",
    ):
        extractor.extract_batch(
            [("a", 0), ("b", 1)],
            torch.zeros(2, 3, 8, 8),
        )


def test_extractor_context_closes_capture_when_inference_fails(monkeypatch):
    extractor, _model, capture = _extractor(monkeypatch, fail=True)

    with pytest.raises(RuntimeError, match="forward failed"):
        with extractor:
            extractor.extract_batch(
                [("a", 0), ("b", 1)],
                torch.zeros(2, 3, 8, 8),
            )

    assert capture.close_count == 1


def test_extractor_context_closes_real_hooks_when_inference_fails(
    monkeypatch,
):
    model = _FakeDetector(fail=True)
    model.decoder = nn.Module()
    model.decoder.decoder = nn.Module()
    model.decoder.decoder.layers = nn.ModuleList([nn.Identity(), nn.Identity()])
    model.decoder.dec_score_head = nn.ModuleList([nn.Linear(4, 2), nn.Linear(4, 2)])
    monkeypatch.setattr(extraction, "load_frozen_detector", lambda *_args: model)
    config = ExperimentConfig.for_tests(
        image_size=(8, 8),
        query_count=10,
        persistence_dim=5,
        persistence_layer=1,
    )

    with pytest.raises(RuntimeError, match="forward failed"):
        with RTDETRExtractor("unused.pth", torch.device("cpu"), config) as extractor:
            extractor.extract_batch([("a", 0)], torch.zeros(1, 3, 8, 8))

    assert not model.decoder.decoder.layers[1]._forward_hooks


def test_extractor_close_is_safe_to_repeat(monkeypatch):
    extractor, _model, capture = _extractor(monkeypatch)

    extractor.close()
    extractor.close()

    assert capture.close_count == 2
