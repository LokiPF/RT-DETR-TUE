import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from differential_uncertainty.extraction import (
    build_fixed_detector,
    checkpoint_state,
    load_frozen_detector,
)
from differential_uncertainty.persistence import Layer2Capture, batched_persistence


CHECKPOINT_NAME = "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
GOLDEN = (
    Path(__file__).parent
    / "fixtures"
    / "rtdetrv2_r18_layer2_golden.pt"
)


def _resolve_checkpoint(*, repository_root=None, home=None):
    """Set UE_RTDETRV2_CHECKPOINT to use an explicit checkpoint path."""
    override = os.environ.get("UE_RTDETRV2_CHECKPOINT")
    if override is not None:
        checkpoint = Path(override).expanduser()
        if not checkpoint.is_file():
            raise AssertionError(
                "UE_RTDETRV2_CHECKPOINT does not name an existing file: "
                f"{checkpoint}"
            )
        return checkpoint

    root = (
        Path(repository_root)
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    checkpoint = root / "pretrained_weights" / CHECKPOINT_NAME
    if checkpoint.is_file():
        return checkpoint

    home_root = Path(home) if home is not None else Path.home()
    checkpoint = (
        home_root
        / "YuchenZ"
        / "UE"
        / "RT-DETRv2-UE"
        / "pretrained_weights"
        / CHECKPOINT_NAME
    )
    if checkpoint.is_file():
        return checkpoint
    pytest.skip(
        "detector checkpoint unavailable; set UE_RTDETRV2_CHECKPOINT"
    )


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_checkpoint(golden):
    checkpoint = _resolve_checkpoint()
    expected_name = golden["checkpoint_name"]
    assert checkpoint.name == expected_name, (
        f"checkpoint filename mismatch: expected {expected_name}, got {checkpoint.name}"
    )
    expected_sha256 = golden["checkpoint_sha256"]
    actual_sha256 = _file_sha256(checkpoint)
    assert actual_sha256 == expected_sha256, (
        "checkpoint SHA-256 mismatch: "
        f"expected {expected_sha256}, got {actual_sha256} for {checkpoint}"
    )
    return checkpoint


def _current_cuda_platform():
    if not torch.cuda.is_available():
        return None
    return {
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
    }


def _require_compatible_platform(golden, current_platform):
    if golden["device_type"] != "cuda":
        raise AssertionError("golden device_type must be cuda")
    if current_platform is None:
        pytest.skip("golden requires CUDA, but CUDA is unavailable")
    expected = golden["execution_platform"]
    differences = [
        f"{key} expected {expected[key]!r}, got {current_platform.get(key)!r}"
        for key in expected
        if current_platform.get(key) != expected[key]
    ]
    if differences:
        pytest.skip(
            "golden CUDA platform incompatible: " + "; ".join(differences)
        )
    return torch.device("cuda:0")


def test_checkpoint_resolution_uses_documented_environment_override(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
    checkpoint.touch()
    monkeypatch.setenv("UE_RTDETRV2_CHECKPOINT", str(checkpoint))

    assert _resolve_checkpoint(
        repository_root=tmp_path / "repo", home=tmp_path / "home"
    ) == checkpoint


def test_checkpoint_resolution_uses_repository_context_without_override(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("UE_RTDETRV2_CHECKPOINT", raising=False)
    checkpoint = (
        tmp_path
        / "repo"
        / "pretrained_weights"
        / "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    try:
        actual = _resolve_checkpoint(
            repository_root=tmp_path / "repo", home=tmp_path / "home"
        )
    except pytest.skip.Exception as error:
        pytest.fail(f"repository checkpoint was unexpectedly skipped: {error}")

    assert actual == checkpoint


def test_checkpoint_resolution_uses_home_context_without_override(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("UE_RTDETRV2_CHECKPOINT", raising=False)
    checkpoint = (
        tmp_path
        / "home"
        / "YuchenZ"
        / "UE"
        / "RT-DETRv2-UE"
        / "pretrained_weights"
        / "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    try:
        actual = _resolve_checkpoint(
            repository_root=tmp_path / "repo",
            home=tmp_path / "home",
        )
    except pytest.skip.Exception as error:
        pytest.fail(f"home checkpoint was unexpectedly skipped: {error}")

    assert actual == checkpoint


def test_same_named_modified_checkpoint_is_rejected_before_inference(
    tmp_path, monkeypatch
):
    source = _resolve_checkpoint()
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    state = checkpoint_state(checkpoint)
    unused_in_eval = "decoder.denoising_class_embed.weight"
    assert unused_in_eval in state
    state[unused_in_eval][0, 0].add_(1.0)
    modified = tmp_path / CHECKPOINT_NAME
    torch.save(checkpoint, modified)
    assert modified.name == source.name
    monkeypatch.setenv("UE_RTDETRV2_CHECKPOINT", str(modified))

    golden = {
        "checkpoint_name": CHECKPOINT_NAME,
        "checkpoint_sha256": (
            "2ace52184b620204004509b72752ac7bfe64aadaf7fc1d076b18df8ab5a5c77e"
        ),
    }
    inference_started = False

    def attempt_inference():
        nonlocal inference_started
        verified = _verified_checkpoint(golden)
        inference_started = True
        load_frozen_detector(verified, torch.device("cpu"))

    with pytest.raises(
        AssertionError,
        match="checkpoint SHA-256 mismatch",
    ):
        attempt_inference()

    assert not inference_started


def test_golden_records_checkpoint_and_exact_cuda_platform_metadata():
    golden = torch.load(GOLDEN, map_location="cpu", weights_only=True)

    assert golden["fixture_schema_version"] == 2
    assert golden["checkpoint_name"] == CHECKPOINT_NAME
    assert golden["checkpoint_sha256"] == (
        "2ace52184b620204004509b72752ac7bfe64aadaf7fc1d076b18df8ab5a5c77e"
    )
    assert golden["execution_platform"] == {
        "torch_version": "2.11.0+cu128",
        "cuda_version": "12.8",
        "cudnn_version": 91900,
        "gpu_name": "NVIDIA GeForce RTX 5090",
        "compute_capability": (12, 0),
    }


@pytest.mark.parametrize(
    ("current_platform", "expected_fragments"),
    [
        (None, ("golden requires CUDA, but CUDA is unavailable",)),
        (
            {
                "torch_version": "2.11.0+cu128",
                "cuda_version": "12.8",
                "cudnn_version": 91900,
                "gpu_name": "NVIDIA GeForce RTX 4090",
                "compute_capability": (8, 9),
            },
            (
                "gpu_name expected 'NVIDIA GeForce RTX 5090', got 'NVIDIA GeForce RTX 4090'",
                "compute_capability expected (12, 0), got (8, 9)",
            ),
        ),
        (
            {
                "torch_version": "2.10.0+cu127",
                "cuda_version": "12.7",
                "cudnn_version": 91000,
                "gpu_name": "NVIDIA GeForce RTX 5090",
                "compute_capability": (12, 0),
            },
            (
                "torch_version expected '2.11.0+cu128', got '2.10.0+cu127'",
                "cuda_version expected '12.8', got '12.7'",
                "cudnn_version expected 91900, got 91000",
            ),
        ),
    ],
)
def test_incompatible_golden_platform_skips_with_precise_reason(
    current_platform, expected_fragments
):
    golden = torch.load(GOLDEN, map_location="cpu", weights_only=True)

    with pytest.raises(pytest.skip.Exception) as skipped:
        _require_compatible_platform(golden, current_platform)

    message = str(skipped.value)
    for fragment in expected_fragments:
        assert fragment in message


def test_fixed_builder_matches_recorded_detector_outputs_exactly():
    golden = torch.load(GOLDEN, map_location="cpu", weights_only=True)
    checkpoint = _verified_checkpoint(golden)
    device = _require_compatible_platform(golden, _current_cuda_platform())
    generator = torch.Generator().manual_seed(golden["input_seed"])
    sample_cpu = torch.rand((1, 3, 640, 640), generator=generator)
    model = load_frozen_detector(checkpoint, device)
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


def test_minimal_detector_import_surfaces_do_not_load_the_registry(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    script = """
import sys

import src

removed_registry = "src." + "core"
assert removed_registry not in sys.modules
from src import nn, zoo
from src.nn import backbone
from src.zoo import rtdetr

assert nn.__all__ == ["PResNet"]
assert backbone.__all__ == ["FrozenBatchNorm2d", "PResNet"]
assert zoo.__all__ == ["HybridEncoder", "RTDETR", "RTDETRTransformerv2"]
assert rtdetr.__all__ == ["HybridEncoder", "RTDETR", "RTDETRTransformerv2"]
assert removed_registry not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
