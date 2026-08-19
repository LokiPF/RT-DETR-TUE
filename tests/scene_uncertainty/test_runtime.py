from pathlib import Path

import torch

from src.core import YAMLConfig
from src.scene_uncertainty.runtime import checkpoint_sha256, load_frozen_detector


CONFIG = "configs/scene_uncertainty/rtdetrv2_r18vd_coco.yml"


def test_model_only_config_builds_plain_rtdetr():
    cfg = YAMLConfig(CONFIG)
    model = cfg.model
    assert type(model).__name__ == "RTDETR"
    assert len(model.decoder.decoder.layers) == 3
    assert model.decoder.num_queries == 300
    assert model.decoder.num_classes == 80


def test_checkpoint_sha256_is_content_addressed(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"detector")
    first = checkpoint_sha256(checkpoint)
    checkpoint.write_bytes(b"detector-changed")
    second = checkpoint_sha256(checkpoint)
    assert len(first) == 64
    assert first != second


def test_load_frozen_detector_accepts_model_state(tmp_path: Path):
    cfg = YAMLConfig(CONFIG)
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save({"model": cfg.model.state_dict()}, checkpoint)
    model = load_frozen_detector(CONFIG, checkpoint, torch.device("cpu"))
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
