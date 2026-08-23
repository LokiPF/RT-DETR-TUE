import gc
import os
from pathlib import Path

import pytest
import torch

from differential_uncertainty.extraction import load_frozen_detector
from differential_uncertainty.persistence import Layer2Capture, batched_persistence
from src.misc.tue_utils import (
    get_captured_persistence_diagrams,
    hook_decoder_layers,
)
from src.scene_uncertainty.runtime import (
    load_frozen_detector as load_legacy_detector,
)


DEFAULT_CHECKPOINT = Path(
    "/home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/"
    "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
)
CONFIG = Path("configs/scene_uncertainty/rtdetrv2_r18vd_coco.yml")


def _resolve_checkpoint(default: Path = DEFAULT_CHECKPOINT) -> Path:
    override = os.environ.get("RTDETRV2_R18_CHECKPOINT")
    if override is not None:
        checkpoint = Path(override)
        assert checkpoint.is_file(), (
            "RTDETRV2_R18_CHECKPOINT does not point to an existing file: "
            f"{checkpoint}"
        )
        return checkpoint
    if default.is_file():
        return default
    pytest.skip(
        "live detector checkpoint unavailable: local default is absent and "
        "RTDETRV2_R18_CHECKPOINT is unset"
    )


def test_explicit_checkpoint_override_must_exist(tmp_path, monkeypatch):
    missing = tmp_path / "explicit-missing.pth"
    fallback = tmp_path / "default.pth"
    fallback.touch()
    monkeypatch.setenv("RTDETRV2_R18_CHECKPOINT", str(missing))

    with pytest.raises(
        AssertionError,
        match="RTDETRV2_R18_CHECKPOINT.*existing file",
    ):
        _resolve_checkpoint(fallback)


def test_explicit_checkpoint_override_rejects_a_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("RTDETRV2_R18_CHECKPOINT", str(tmp_path))

    with pytest.raises(
        AssertionError,
        match="RTDETRV2_R18_CHECKPOINT.*existing file",
    ):
        _resolve_checkpoint()


def test_explicit_checkpoint_override_wins_over_the_default(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit.pth"
    explicit.touch()
    fallback = tmp_path / "default.pth"
    fallback.touch()
    monkeypatch.setenv("RTDETRV2_R18_CHECKPOINT", str(explicit))

    assert _resolve_checkpoint(fallback) == explicit


def test_unset_checkpoint_override_uses_an_existing_default(tmp_path, monkeypatch):
    fallback = tmp_path / "default.pth"
    fallback.touch()
    monkeypatch.delenv("RTDETRV2_R18_CHECKPOINT", raising=False)

    assert _resolve_checkpoint(fallback) == fallback


def test_unset_checkpoint_override_skips_when_the_default_is_absent(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("RTDETRV2_R18_CHECKPOINT", raising=False)

    with pytest.raises(
        pytest.skip.Exception,
        match="checkpoint unavailable.*RTDETRV2_R18_CHECKPOINT",
    ):
        _resolve_checkpoint(tmp_path / "default-missing.pth")


def test_explicit_model_and_layer_two_persistence_equal_the_legacy_path():
    checkpoint = _resolve_checkpoint()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    samples = torch.rand(
        (1, 3, 640, 640),
        generator=torch.Generator().manual_seed(19),
    )

    legacy = load_legacy_detector(CONFIG, checkpoint, device)
    captures, handles, layers = hook_decoder_layers(
        legacy.decoder,
        [2],
        "score",
    )
    try:
        with torch.inference_mode():
            legacy_outputs = legacy(samples.to(device))
        selected = [torch.arange(300, device=device)]
        legacy_persistence = torch.stack(
            list(
                get_captured_persistence_diagrams(
                    captures,
                    selected,
                    layers,
                )[2][0].values()
            )
        ).unsqueeze(0)
        legacy_logits = legacy_outputs["pred_logits"].cpu()
        legacy_boxes = legacy_outputs["pred_boxes"].cpu()
    finally:
        for handle in handles:
            handle.remove()
    del captures, legacy_outputs, legacy
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    explicit = load_frozen_detector(checkpoint, device)
    with torch.inference_mode(), Layer2Capture(
        explicit.decoder,
        2,
    ) as explicit_capture:
        explicit_outputs = explicit(samples.to(device))
        explicit_features, explicit_weight = explicit_capture.take()
        explicit_persistence = batched_persistence(
            explicit_weight,
            explicit_features.reshape(-1, explicit_features.shape[-1]),
        ).reshape(1, 300, 335)

    torch.testing.assert_close(
        explicit_outputs["pred_logits"].cpu(),
        legacy_logits,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        explicit_outputs["pred_boxes"].cpu(),
        legacy_boxes,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        explicit_persistence.cpu(),
        legacy_persistence,
        rtol=0,
        atol=0,
    )
