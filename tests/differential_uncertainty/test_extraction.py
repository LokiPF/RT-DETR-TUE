from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import Tensor, nn

import differential_uncertainty.extraction as extraction
from differential_uncertainty.artifacts import (
    ShardWriter,
    iter_records,
    load_manifest as load_artifact_manifest,
)
from differential_uncertainty.config import ExperimentConfig
from differential_uncertainty.corruptions.base import Severity
from differential_uncertainty.corruptions.gaussian_blur import GaussianBlur
from differential_uncertainty.extraction import (
    RTDETRExtractor,
    build_fixed_detector,
    checkpoint_state,
    extract_manifest,
    load_frozen_detector,
    prepare_image,
    record_from_outputs,
)
from differential_uncertainty.manifests import ManifestEntry


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


def test_prepare_image_rgb_conversion_precedes_resize_for_nonconstant_image():
    image = Image.new("RGBA", (2, 2))
    image.putdata(
        [
            (255, 0, 0, 0),
            (0, 255, 0, 255),
            (0, 0, 255, 255),
            (255, 255, 255, 0),
        ]
    )

    tensor = prepare_image(image, (4, 4))

    corners = torch.stack(
        [
            tensor[:, 0, 0],
            tensor[:, 0, -1],
            tensor[:, -1, 0],
            tensor[:, -1, -1],
        ]
    )
    torch.testing.assert_close(
        corners,
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        ),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        tensor[:, 1, 1],
        torch.tensor([159.0, 64.0, 64.0]) / 255.0,
        rtol=0,
        atol=1.0 / 255.0,
    )
    assert torch.all(tensor[:, 1, 1] > 0)
    assert torch.all(tensor[:, 1, 1] < 1)


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
            "samples must have dtype torch.float32, got torch.int64",
        ),
        (
            torch.zeros(2, 3, 8, 8, dtype=torch.float16),
            [("a", 0), ("b", 0)],
            "samples must have dtype torch.float32, got torch.float16",
        ),
        (
            torch.zeros(2, 3, 8, 8, dtype=torch.float64),
            [("a", 0), ("b", 0)],
            "samples must have dtype torch.float32, got torch.float64",
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
    extractor, model, capture = _extractor(monkeypatch)

    extractor.close()
    extractor.close()

    assert capture.close_count == 1
    assert extractor.model is None
    assert extractor.capture is None
    assert model is not None
    with pytest.raises(RuntimeError, match="extractor is closed"):
        extractor.extract_batch(
            [("a", 0)], torch.zeros(1, 3, 8, 8)
        )


class FakeExtractor:
    def __init__(self):
        self.calls = 0

    def extract_batch(self, identities, samples):
        self.calls += 1
        records = []
        for (image_id, severity), sample in zip(identities, samples):
            value = float(sample.mean()) + severity
            records.append(
                {
                    "image_id": image_id,
                    "severity": severity,
                    "boxes": torch.full((20, 4), value),
                    "logits": torch.full((20, 80), value),
                    "persistence": torch.full((20, 7), value),
                }
            )
        return records


def test_evaluation_cache_has_six_records_and_completed_cache_is_reused(
    tmp_path: Path,
):
    path = tmp_path / "image.png"
    Image.effect_noise((19, 11), 80).convert("RGB").save(path)
    entries = (ManifestEntry("scene", path.resolve()),)
    cache = tmp_path / "evaluation"
    extractor = FakeExtractor()
    metadata = {"stage": "evaluation", "input_id": "fixed"}

    extract_manifest(
        entries,
        cache,
        metadata,
        extractor,
        GaussianBlur(),
        image_size=(640, 640),
        batch_size=2,
        shard_size=2,
    )

    assert [
        (record["image_id"], record["severity"])
        for record in iter_records(cache)
    ] == [
        ("scene", 0),
        ("scene", 1),
        ("scene", 2),
        ("scene", 3),
        ("scene", 4),
        ("scene", 5),
    ]
    assert load_artifact_manifest(cache)["record_count"] == 6

    calls = extractor.calls
    extract_manifest(
        entries,
        cache,
        metadata,
        extractor,
        GaussianBlur(),
        image_size=(640, 640),
        batch_size=2,
        shard_size=2,
    )

    assert extractor.calls == calls


class ObservedCorruption:
    name = "observed"
    severities = tuple(Severity(level, float(level)) for level in range(6))

    def __init__(self):
        self.inputs = []

    def apply(self, image, level):
        self.inputs.append((image.mode, image.size, level))
        return image.copy()


class SampleShapeExtractor(FakeExtractor):
    def __init__(self):
        super().__init__()
        self.sample_shapes = []

    def extract_batch(self, identities, samples):
        self.sample_shapes.append(tuple(samples.shape))
        return super().extract_batch(identities, samples)


def test_images_are_resized_to_rgb_before_each_corruption_is_applied(tmp_path: Path):
    path = tmp_path / "asymmetric.png"
    Image.new("L", (13, 9), 100).save(path)
    entries = (ManifestEntry("scene", path.resolve()),)
    corruption = ObservedCorruption()
    extractor = SampleShapeExtractor()

    extract_manifest(
        entries,
        tmp_path / "cache",
        {"stage": "evaluation"},
        extractor,
        corruption,
        image_size=(5, 7),
        batch_size=6,
        shard_size=6,
    )

    assert corruption.inputs == [
        ("RGB", (7, 5), level) for level in range(6)
    ]
    assert extractor.sample_shapes == [(6, 3, 5, 7)]


class WrongIdentityExtractor(FakeExtractor):
    def extract_batch(self, identities, samples):
        records = super().extract_batch(identities, samples)
        records[0]["image_id"] = "wrong"
        return records


def test_wrong_extractor_identity_never_publishes_a_complete_cache(tmp_path: Path):
    path = tmp_path / "image.png"
    Image.new("RGB", (9, 7)).save(path)
    entries = (ManifestEntry("scene", path.resolve()),)
    cache = tmp_path / "bad-reference"

    with pytest.raises(RuntimeError, match="extractor returned keys"):
        extract_manifest(
            entries,
            cache,
            {"stage": "reference"},
            WrongIdentityExtractor(),
            None,
            image_size=(640, 640),
            batch_size=1,
            shard_size=1,
        )

    assert not (cache / "manifest.json").exists()
    assert (cache / "partial_manifest.json").exists()


class IntegerIdentityExtractor(FakeExtractor):
    def extract_batch(self, identities, samples):
        records = super().extract_batch(identities, samples)
        records[0]["image_id"] = 123
        return records


def test_extractor_identity_must_be_a_string_without_coercion(tmp_path: Path):
    path = tmp_path / "image.png"
    Image.new("RGB", (9, 7)).save(path)
    cache = tmp_path / "bad-output-type"

    with pytest.raises(RuntimeError, match="string image_id"):
        extract_manifest(
            (ManifestEntry("123", path.resolve()),),
            cache,
            {"stage": "reference"},
            IntegerIdentityExtractor(),
            None,
            image_size=(8, 8),
            batch_size=1,
            shard_size=1,
        )

    assert not (cache / "manifest.json").exists()
    assert (cache / "partial_manifest.json").exists()


class InterruptedExtractor(FakeExtractor):
    def __init__(self):
        super().__init__()
        self.attempts = 0

    def extract_batch(self, identities, samples):
        self.attempts += 1
        if self.attempts == 2:
            raise RuntimeError("simulated interruption")
        return super().extract_batch(identities, samples)


def test_partial_cache_resumes_only_the_missing_severities(tmp_path: Path):
    path = tmp_path / "image.png"
    Image.effect_noise((11, 7), 40).convert("RGB").save(path)
    entries = (ManifestEntry("scene", path.resolve()),)
    cache = tmp_path / "resumable"
    metadata = {"stage": "evaluation", "input_id": "fixed"}
    interrupted = InterruptedExtractor()

    with pytest.raises(RuntimeError, match="simulated interruption"):
        extract_manifest(
            entries,
            cache,
            metadata,
            interrupted,
            GaussianBlur(),
            image_size=(8, 8),
            batch_size=2,
            shard_size=2,
        )

    assert not (cache / "manifest.json").exists()
    assert (cache / "partial_manifest.json").exists()

    resumed = FakeExtractor()
    extract_manifest(
        entries,
        cache,
        metadata,
        resumed,
        GaussianBlur(),
        image_size=(8, 8),
        batch_size=2,
        shard_size=2,
    )

    assert interrupted.calls == 1
    assert resumed.calls == 2
    assert [
        (record["image_id"], record["severity"])
        for record in iter_records(cache)
    ] == [("scene", level) for level in range(6)]
    assert load_artifact_manifest(cache)["record_count"] == 6


def _cache_record(image_id, severity, *, queries=20, classes=80, width=7):
    value = float(severity + 1)
    return {
        "image_id": image_id,
        "severity": severity,
        "boxes": torch.full((queries, 4), value, dtype=torch.float32),
        "logits": torch.full((queries, classes), value, dtype=torch.float16),
        "persistence": torch.full((queries, width), value, dtype=torch.float16),
    }


def _write_complete_cache(cache: Path, metadata: dict, identities):
    with ShardWriter(cache, metadata, shard_size=6) as writer:
        for image_id, severity in identities:
            writer.add(_cache_record(image_id, severity))


def test_completed_cache_requires_exact_metadata_without_extra_keys(tmp_path: Path):
    cache = tmp_path / "extra-metadata"
    metadata = {"stage": "evaluation"}
    _write_complete_cache(
        cache,
        {**metadata, "unexpected": "stale"},
        [("scene", level) for level in range(6)],
    )
    path = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(path)

    with pytest.raises(ValueError, match="completed extraction metadata mismatch"):
        extract_manifest(
            (ManifestEntry("scene", path.resolve()),),
            cache,
            metadata,
            FakeExtractor(),
            GaussianBlur(),
            image_size=(8, 8),
            batch_size=2,
            shard_size=2,
        )


def test_completed_cache_requires_the_exact_record_roster(tmp_path: Path):
    cache = tmp_path / "wrong-roster"
    metadata = {"stage": "evaluation"}
    _write_complete_cache(
        cache,
        metadata,
        [
            ("scene", 0),
            ("scene", 1),
            ("scene", 2),
            ("scene", 3),
            ("scene", 4),
            ("other", 5),
        ],
    )
    path = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(path)

    with pytest.raises(RuntimeError, match="completed extraction record roster"):
        extract_manifest(
            (ManifestEntry("scene", path.resolve()),),
            cache,
            metadata,
            FakeExtractor(),
            GaussianBlur(),
            image_size=(8, 8),
            batch_size=2,
            shard_size=2,
        )


def test_completed_cache_requires_the_canonical_record_order(tmp_path: Path):
    cache = tmp_path / "reversed-roster"
    metadata = {"stage": "evaluation"}
    _write_complete_cache(
        cache,
        metadata,
        [("scene", level) for level in reversed(range(6))],
    )
    path = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(path)
    extractor = FakeExtractor()

    with pytest.raises(RuntimeError, match="completed extraction record roster"):
        extract_manifest(
            (ManifestEntry("scene", path.resolve()),),
            cache,
            metadata,
            extractor,
            GaussianBlur(),
            image_size=(8, 8),
            batch_size=2,
            shard_size=2,
        )

    assert extractor.calls == 0


def test_partial_cache_rejects_an_unexpected_record_before_extraction(tmp_path: Path):
    cache = tmp_path / "wrong-partial-roster"
    metadata = {"stage": "reference"}
    with pytest.raises(RuntimeError, match="leave partial"):
        with ShardWriter(cache, metadata, shard_size=1) as writer:
            writer.add(_cache_record("other", 0))
            raise RuntimeError("leave partial")
    path = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(path)
    extractor = FakeExtractor()

    with pytest.raises(RuntimeError, match="partial extraction record roster"):
        extract_manifest(
            (ManifestEntry("scene", path.resolve()),),
            cache,
            metadata,
            extractor,
            None,
            image_size=(8, 8),
            batch_size=1,
            shard_size=1,
        )

    assert extractor.calls == 0
    assert not (cache / "manifest.json").exists()


def test_partial_cache_must_be_a_canonical_prefix_before_resume(tmp_path: Path):
    cache = tmp_path / "out-of-order-partial"
    metadata = {"stage": "evaluation"}
    with pytest.raises(RuntimeError, match="leave partial"):
        with ShardWriter(cache, metadata, shard_size=1) as writer:
            writer.add(_cache_record("scene", 5))
            raise RuntimeError("leave partial")
    path = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(path)
    extractor = FakeExtractor()

    with pytest.raises(RuntimeError, match="canonical prefix"):
        extract_manifest(
            (ManifestEntry("scene", path.resolve()),),
            cache,
            metadata,
            extractor,
            GaussianBlur(),
            image_size=(8, 8),
            batch_size=2,
            shard_size=1,
        )

    assert extractor.calls == 0
    assert not (cache / "manifest.json").exists()
    assert (cache / "partial_manifest.json").exists()


@pytest.mark.parametrize("batch_size", [True, 1.0, 0, -1])
def test_batch_size_must_be_a_positive_integer(tmp_path: Path, batch_size):
    path = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(path)
    cache = tmp_path / "bad-batch"

    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        extract_manifest(
            (ManifestEntry("scene", path.resolve()),),
            cache,
            {"stage": "reference"},
            FakeExtractor(),
            None,
            image_size=(8, 8),
            batch_size=batch_size,
            shard_size=1,
        )

    assert not cache.exists()


@pytest.mark.parametrize(
    "image_size",
    [[8, 8], (True, 8), (8.0, 8), (8,), (0, 8), (-1, 8)],
)
def test_image_size_must_be_two_positive_integers(tmp_path: Path, image_size):
    path = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(path)
    cache = tmp_path / "bad-size"

    with pytest.raises(ValueError, match="image_size must be two positive integers"):
        extract_manifest(
            (ManifestEntry("scene", path.resolve()),),
            cache,
            {"stage": "reference"},
            FakeExtractor(),
            None,
            image_size=image_size,
            batch_size=1,
            shard_size=1,
        )

    assert not cache.exists()


class BadSeverityCorruption:
    name = "bad"
    severities = (Severity(0, 0.0), Severity(2, 1.0))

    def apply(self, image, level):
        return image


def test_corruption_must_describe_exactly_levels_zero_through_five(tmp_path: Path):
    path = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(path)
    cache = tmp_path / "bad-severities"

    with pytest.raises(ValueError, match="severity levels must be exactly 0 through 5"):
        extract_manifest(
            (ManifestEntry("scene", path.resolve()),),
            cache,
            {"stage": "evaluation"},
            FakeExtractor(),
            BadSeverityCorruption(),
            image_size=(8, 8),
            batch_size=1,
            shard_size=1,
        )

    assert not cache.exists()


def test_duplicate_entry_identities_fail_before_creating_a_cache(tmp_path: Path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (4, 4)).save(first)
    Image.new("RGB", (4, 4)).save(second)
    extractor = FakeExtractor()
    cache = tmp_path / "duplicate-entries"

    with pytest.raises(ValueError, match="duplicate extraction image_id"):
        extract_manifest(
            (
                ManifestEntry("scene", first.resolve()),
                ManifestEntry("scene", second.resolve()),
            ),
            cache,
            {"stage": "reference"},
            extractor,
            None,
            image_size=(8, 8),
            batch_size=2,
            shard_size=2,
        )

    assert extractor.calls == 0
    assert not cache.exists()


@pytest.mark.parametrize("shard_size", [True, 1.0, 0, -1])
def test_shard_size_must_be_a_positive_integer(tmp_path: Path, shard_size):
    path = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(path)
    cache = tmp_path / "bad-shard"

    with pytest.raises(ValueError, match="shard_size must be a positive integer"):
        extract_manifest(
            (ManifestEntry("scene", path.resolve()),),
            cache,
            {"stage": "reference"},
            FakeExtractor(),
            None,
            image_size=(8, 8),
            batch_size=1,
            shard_size=shard_size,
        )

    assert not cache.exists()


@pytest.mark.parametrize(
    "entries",
    [
        (),
        [],
        (ManifestEntry("", Path("/unused")),),
        (ManifestEntry("   ", Path("/unused")),),
    ],
)
def test_entry_roster_must_be_a_nonempty_tuple_of_named_entries(
    tmp_path: Path,
    entries,
):
    cache = tmp_path / "bad-entries"

    with pytest.raises(ValueError, match="nonempty tuple|nonempty image_id"):
        extract_manifest(
            entries,
            cache,
            {"stage": "reference"},
            FakeExtractor(),
            None,
            image_size=(8, 8),
            batch_size=1,
            shard_size=1,
        )

    assert not cache.exists()


class MalformedOutputExtractor(FakeExtractor):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    def extract_batch(self, identities, samples):
        records = super().extract_batch(identities, samples)
        if self.kind == "short_batch":
            return records[:-1]
        if self.kind == "not_list":
            return tuple(records)
        if self.kind == "missing_tensor":
            del records[0]["persistence"]
        elif self.kind == "logits_rank":
            records[0]["logits"] = records[0]["logits"].unsqueeze(0)
        elif self.kind == "box_width":
            records[0]["boxes"] = records[0]["boxes"][:, :3]
        elif self.kind == "query_count":
            records[0]["persistence"] = records[0]["persistence"][:-1]
        return records


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("short_batch", "returned 1 records; expected 2"),
        ("not_list", "must return a list"),
        ("missing_tensor", "must contain tensor persistence"),
        ("logits_rank", "logits must have rank 2"),
        ("box_width", "boxes must have shape"),
        ("query_count", "query counts must match"),
    ],
)
def test_malformed_extractor_records_never_publish_a_complete_cache(
    tmp_path: Path,
    kind,
    message,
):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (4, 4)).save(first)
    Image.new("RGB", (4, 4)).save(second)
    cache = tmp_path / f"bad-output-{kind}"

    with pytest.raises(RuntimeError, match=message):
        extract_manifest(
            (
                ManifestEntry("first", first.resolve()),
                ManifestEntry("second", second.resolve()),
            ),
            cache,
            {"stage": "reference"},
            MalformedOutputExtractor(kind),
            None,
            image_size=(8, 8),
            batch_size=2,
            shard_size=2,
        )

    assert not (cache / "manifest.json").exists()
    assert (cache / "partial_manifest.json").exists()


class WrongSizeCorruption(ObservedCorruption):
    def apply(self, image, level):
        super().apply(image, level)
        return image.resize((1, 1))


def test_corruption_cannot_change_the_resized_image_dimensions(tmp_path: Path):
    path = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(path)
    cache = tmp_path / "wrong-corruption-size"

    with pytest.raises(RuntimeError, match="corruption changed image size"):
        extract_manifest(
            (ManifestEntry("scene", path.resolve()),),
            cache,
            {"stage": "evaluation"},
            FakeExtractor(),
            WrongSizeCorruption(),
            image_size=(8, 8),
            batch_size=1,
            shard_size=1,
        )

    assert not (cache / "manifest.json").exists()
    assert (cache / "partial_manifest.json").exists()


class AlwaysFailsExtractor(FakeExtractor):
    def extract_batch(self, identities, samples):
        raise RuntimeError("inference failed")


def test_failed_extraction_releases_writer_for_a_later_resume(tmp_path: Path):
    path = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(path)
    entries = (ManifestEntry("scene", path.resolve()),)
    cache = tmp_path / "released-writer"
    metadata = {"stage": "reference"}

    with pytest.raises(RuntimeError, match="inference failed"):
        extract_manifest(
            entries, cache, metadata, AlwaysFailsExtractor(), None,
            image_size=(8, 8), batch_size=1, shard_size=1,
        )

    extract_manifest(
        entries, cache, metadata, FakeExtractor(), None,
        image_size=(8, 8), batch_size=1, shard_size=1,
    )

    assert load_artifact_manifest(cache)["record_count"] == 1


def test_completed_cache_rejects_a_record_missing_a_required_tensor(tmp_path: Path):
    cache = tmp_path / "completed-missing-tensor"
    metadata = {"stage": "evaluation"}
    records = [_cache_record("scene", level) for level in range(6)]
    del records[2]["persistence"]
    with ShardWriter(cache, metadata, shard_size=6) as writer:
        for record in records:
            writer.add(record)
    path = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(path)
    extractor = FakeExtractor()

    with pytest.raises(RuntimeError, match="must contain tensor persistence"):
        extract_manifest(
            (ManifestEntry("scene", path.resolve()),),
            cache,
            metadata,
            extractor,
            GaussianBlur(),
            image_size=(8, 8),
            batch_size=2,
            shard_size=6,
        )

    assert extractor.calls == 0


def test_partial_cache_rejects_a_record_missing_a_required_tensor(tmp_path: Path):
    cache = tmp_path / "partial-missing-tensor"
    metadata = {"stage": "reference"}
    record = _cache_record("scene", 0)
    del record["logits"]
    with pytest.raises(RuntimeError, match="leave partial"):
        with ShardWriter(cache, metadata, shard_size=1) as writer:
            writer.add(record)
            raise RuntimeError("leave partial")
    path = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(path)
    extractor = FakeExtractor()

    with pytest.raises(RuntimeError, match="must contain tensor logits"):
        extract_manifest(
            (ManifestEntry("scene", path.resolve()),),
            cache,
            metadata,
            extractor,
            None,
            image_size=(8, 8),
            batch_size=1,
            shard_size=1,
        )

    assert extractor.calls == 0
    assert not (cache / "manifest.json").exists()


class CrossBatchShapeExtractor(FakeExtractor):
    def __init__(self, field):
        super().__init__()
        self.field = field

    def extract_batch(self, identities, samples):
        records = super().extract_batch(identities, samples)
        if self.calls != 2:
            return records
        if self.field == "queries":
            records[0]["boxes"] = records[0]["boxes"][:-1]
            records[0]["logits"] = records[0]["logits"][:-1]
            records[0]["persistence"] = records[0]["persistence"][:-1]
        elif self.field == "classes":
            records[0]["logits"] = records[0]["logits"][:, :-1]
        else:
            records[0]["persistence"] = records[0]["persistence"][:, :-1]
        return records


@pytest.mark.parametrize("field", ["queries", "classes", "persistence"])
def test_record_contract_cannot_change_between_batches(tmp_path: Path, field):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (4, 4), 20).save(first)
    Image.new("RGB", (4, 4), 40).save(second)
    cache = tmp_path / f"cross-batch-{field}"

    with pytest.raises(RuntimeError, match="cache record contract"):
        extract_manifest(
            (
                ManifestEntry("first", first.resolve()),
                ManifestEntry("second", second.resolve()),
            ),
            cache,
            {"stage": "reference"},
            CrossBatchShapeExtractor(field),
            None,
            image_size=(8, 8),
            batch_size=1,
            shard_size=1,
        )

    assert not (cache / "manifest.json").exists()
    assert (cache / "partial_manifest.json").exists()


class InvalidTensorExtractor(FakeExtractor):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    def extract_batch(self, identities, samples):
        records = super().extract_batch(identities, samples)
        if self.kind == "integer":
            records[0]["logits"] = records[0]["logits"].to(torch.int64)
        elif self.kind == "nonfinite":
            records[0]["persistence"][0, 0] = float("nan")
        else:
            records[0]["logits"] = records[0]["logits"].to_sparse()
        return records


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("integer", "floating-point"),
        ("nonfinite", "finite"),
        ("layout", "strided"),
    ],
)
def test_cache_tensors_must_be_floating_finite_and_strided(
    tmp_path: Path,
    kind,
    message,
):
    path = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(path)
    cache = tmp_path / f"invalid-tensor-{kind}"

    with pytest.raises(RuntimeError, match=message):
        extract_manifest(
            (ManifestEntry("scene", path.resolve()),),
            cache,
            {"stage": "reference"},
            InvalidTensorExtractor(kind),
            None,
            image_size=(8, 8),
            batch_size=1,
            shard_size=1,
        )

    assert not (cache / "manifest.json").exists()


class InPlaceCorruption:
    name = "in_place"
    severities = tuple(Severity(level, float(level)) for level in range(6))

    def apply(self, image, level):
        value = image.getpixel((0, 0))[0] + level
        image.putpixel((0, 0), (value, value, value))
        return image


class PixelExtractor(FakeExtractor):
    def __init__(self):
        super().__init__()
        self.values = []

    def extract_batch(self, identities, samples):
        self.values.extend(float(sample[0, 0, 0]) for sample in samples)
        return super().extract_batch(identities, samples)


def test_each_in_place_corruption_level_starts_from_an_independent_image(
    tmp_path: Path,
):
    path = tmp_path / "image.png"
    Image.new("RGB", (1, 1), (10, 10, 10)).save(path)
    extractor = PixelExtractor()

    extract_manifest(
        (ManifestEntry("scene", path.resolve()),),
        tmp_path / "independent-corruption",
        {"stage": "evaluation"},
        extractor,
        InPlaceCorruption(),
        image_size=(1, 1),
        batch_size=1,
        shard_size=1,
    )

    assert extractor.values == pytest.approx(
        [(10 + level) / 255 for level in range(6)]
    )


class CloseTrackingCorruption:
    name = "close_tracking"
    severities = tuple(Severity(level, float(level)) for level in range(6))

    def __init__(self):
        self.inputs = []
        self.outputs = []

    def apply(self, image, level):
        self.inputs.append(image)
        output = image if level == 0 else image.copy()
        self.outputs.append(output)
        return output


class CancelOnSecondBatchExtractor(FakeExtractor):
    def __init__(self):
        super().__init__()
        self.attempts = 0

    def extract_batch(self, identities, samples):
        self.attempts += 1
        if self.attempts == 2:
            raise KeyboardInterrupt("cancel extraction")
        return super().extract_batch(identities, samples)


def test_pil_images_close_once_when_extraction_is_cancelled(
    tmp_path: Path,
    monkeypatch,
):
    path = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(path)
    base = Image.new("RGB", (8, 8), 30)
    corruption = CloseTrackingCorruption()
    close_counts = {}
    original_close = Image.Image.close

    def observed_close(image):
        close_counts[id(image)] = close_counts.get(id(image), 0) + 1
        original_close(image)

    monkeypatch.setattr(Image.Image, "close", observed_close)
    monkeypatch.setattr(extraction, "resize_image", lambda *_args: base)

    with pytest.raises(KeyboardInterrupt, match="cancel extraction"):
        extract_manifest(
            (ManifestEntry("scene", path.resolve()),),
            tmp_path / "cancelled",
            {"stage": "evaluation"},
            CancelOnSecondBatchExtractor(),
            corruption,
            image_size=(8, 8),
            batch_size=1,
            shard_size=10,
        )

    tracked = {id(base)}
    tracked.update(id(image) for image in corruption.inputs)
    tracked.update(id(image) for image in corruption.outputs)
    assert len(tracked) == 4
    assert {identity: close_counts.get(identity, 0) for identity in tracked} == {
        identity: 1 for identity in tracked
    }


class RetainedMutationExtractor(FakeExtractor):
    def __init__(self):
        super().__init__()
        self.retained = None
        self.original = None

    def extract_batch(self, identities, samples):
        if self.retained is not None:
            self.retained.fill_(999)
        records = super().extract_batch(identities, samples)
        if self.retained is None:
            self.retained = records[0]["logits"]
            self.original = self.retained.clone()
        return records


def test_buffered_records_own_only_normalized_independent_tensor_data(tmp_path: Path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (4, 4), 20).save(first)
    Image.new("RGB", (4, 4), 40).save(second)
    extractor = RetainedMutationExtractor()
    cache = tmp_path / "owned-records"

    extract_manifest(
        (
            ManifestEntry("first", first.resolve()),
            ManifestEntry("second", second.resolve()),
        ),
        cache,
        {"stage": "reference"},
        extractor,
        None,
        image_size=(8, 8),
        batch_size=1,
        shard_size=10,
    )

    records = list(iter_records(cache))
    torch.testing.assert_close(
        records[0]["logits"], extractor.original.to(torch.float16)
    )
    assert set(records[0]) == {
        "image_id",
        "severity",
        "boxes",
        "logits",
        "persistence",
    }


class MixedFloatingDtypeExtractor(FakeExtractor):
    def extract_batch(self, identities, samples):
        records = super().extract_batch(identities, samples)
        if self.calls == 1:
            records[0]["boxes"] = records[0]["boxes"].to(torch.float64)
            records[0]["logits"] = records[0]["logits"].to(torch.float64)
            records[0]["persistence"] = records[0]["persistence"].to(
                torch.float64
            )
        else:
            records[0]["boxes"] = records[0]["boxes"].to(torch.float16)
        return records


def test_fresh_mixed_floating_dtypes_are_normalized_before_caching(
    tmp_path: Path,
):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (4, 4), 20).save(first)
    Image.new("RGB", (4, 4), 40).save(second)
    cache = tmp_path / "normalized-dtypes"

    extract_manifest(
        (
            ManifestEntry("first", first.resolve()),
            ManifestEntry("second", second.resolve()),
        ),
        cache,
        {"stage": "reference"},
        MixedFloatingDtypeExtractor(),
        None,
        image_size=(8, 8),
        batch_size=1,
        shard_size=10,
    )

    records = list(iter_records(cache))
    assert [
        (
            record["boxes"].dtype,
            record["logits"].dtype,
            record["persistence"].dtype,
        )
        for record in records
    ] == [
        (torch.float32, torch.float16, torch.float16),
        (torch.float32, torch.float16, torch.float16),
    ]


def _write_review_cache(cache, metadata, records, state):
    if state == "completed":
        with ShardWriter(cache, metadata, shard_size=6) as writer:
            for record in records:
                writer.add(record)
        return
    with pytest.raises(RuntimeError, match="leave partial"):
        with ShardWriter(cache, metadata, shard_size=1) as writer:
            writer.add(records[0])
            raise RuntimeError("leave partial")


@pytest.mark.parametrize("state", ["completed", "partial"])
@pytest.mark.parametrize(
    ("field", "bad_dtype", "canonical_dtype"),
    [
        ("boxes", torch.float16, torch.float32),
        ("logits", torch.float32, torch.float16),
        ("persistence", torch.float64, torch.float16),
    ],
)
def test_existing_cache_requires_each_canonical_tensor_dtype(
    tmp_path: Path,
    state,
    field,
    bad_dtype,
    canonical_dtype,
):
    cache = tmp_path / f"{state}-{field}-dtype"
    metadata = {"stage": "evaluation" if state == "completed" else "reference"}
    record_count = 6 if state == "completed" else 1
    records = [_cache_record("scene", level) for level in range(record_count)]
    records[-1][field] = records[-1][field].to(bad_dtype)
    _write_review_cache(cache, metadata, records, state)
    path = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(path)
    extractor = FakeExtractor()

    with pytest.raises(
        RuntimeError,
        match=f"{field} must have dtype {canonical_dtype}",
    ):
        extract_manifest(
            (ManifestEntry("scene", path.resolve()),),
            cache,
            metadata,
            extractor,
            GaussianBlur() if state == "completed" else None,
            image_size=(8, 8),
            batch_size=2,
            shard_size=6 if state == "completed" else 1,
        )

    assert extractor.calls == 0
    if state == "partial":
        assert not (cache / "manifest.json").exists()


@pytest.mark.parametrize("state", ["completed", "partial"])
def test_existing_cache_records_must_have_exactly_five_keys(
    tmp_path: Path,
    state,
):
    cache = tmp_path / f"{state}-extra-record-key"
    metadata = {"stage": "evaluation" if state == "completed" else "reference"}
    record_count = 6 if state == "completed" else 1
    records = [_cache_record("scene", level) for level in range(record_count)]
    records[-1]["debug"] = torch.tensor([1.0])
    _write_review_cache(cache, metadata, records, state)
    path = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(path)
    extractor = FakeExtractor()

    with pytest.raises(RuntimeError, match="exactly the cache record keys"):
        extract_manifest(
            (ManifestEntry("scene", path.resolve()),),
            cache,
            metadata,
            extractor,
            GaussianBlur() if state == "completed" else None,
            image_size=(8, 8),
            batch_size=2,
            shard_size=6 if state == "completed" else 1,
        )

    assert extractor.calls == 0
    if state == "partial":
        assert not (cache / "manifest.json").exists()


class ExtraKeyExtractor(FakeExtractor):
    def extract_batch(self, identities, samples):
        records = super().extract_batch(identities, samples)
        records[0]["debug"] = torch.tensor([1.0])
        return records


def test_fresh_extractor_records_must_have_exactly_five_keys(tmp_path: Path):
    path = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(path)
    cache = tmp_path / "fresh-extra-record-key"

    with pytest.raises(RuntimeError, match="exactly the cache record keys"):
        extract_manifest(
            (ManifestEntry("scene", path.resolve()),),
            cache,
            {"stage": "reference"},
            ExtraKeyExtractor(),
            None,
            image_size=(8, 8),
            batch_size=1,
            shard_size=1,
        )

    assert not (cache / "manifest.json").exists()


class CastOverflowExtractor(FakeExtractor):
    def __init__(self, field):
        super().__init__()
        self.field = field

    def extract_batch(self, identities, samples):
        records = super().extract_batch(identities, samples)
        source = records[0][self.field].to(torch.float64)
        source.fill_(1e300)
        assert bool(torch.isfinite(source).all())
        records[0][self.field] = source
        return records


@pytest.mark.parametrize("field", ["boxes", "logits", "persistence"])
def test_canonical_cast_overflow_never_publishes_and_can_resume(
    tmp_path: Path,
    field,
):
    path = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(path)
    cache = tmp_path / f"cast-overflow-{field}"
    entries = (ManifestEntry("scene", path.resolve()),)
    metadata = {"stage": "reference"}

    with pytest.raises(
        RuntimeError,
        match=f"normalized {field} must contain only finite values",
    ):
        extract_manifest(
            entries,
            cache,
            metadata,
            CastOverflowExtractor(field),
            None,
            image_size=(8, 8),
            batch_size=1,
            shard_size=1,
        )

    assert not (cache / "manifest.json").exists()
    assert (cache / "partial_manifest.json").exists()

    extract_manifest(
        entries,
        cache,
        metadata,
        FakeExtractor(),
        None,
        image_size=(8, 8),
        batch_size=1,
        shard_size=1,
    )

    assert load_artifact_manifest(cache)["record_count"] == 1
