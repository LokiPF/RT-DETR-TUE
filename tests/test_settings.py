import hashlib
from pathlib import Path

import pytest

from degradation_monitor.settings import load_settings

CONFIG = """
run = "{root}/runs/coco"
checkpoint = "{root}/ckpt.pth"
train_images = "{root}/train"
val_images = "{root}/val"
annotations = "{root}/ann.json"
discopatch_root = "{root}/dcp"
batch_size = 8
gpu_memory_gib = 5.5
"""


def _config(tmp_path, text=CONFIG):
    path = tmp_path / "coco.toml"
    path.write_text(text.format(root=tmp_path))
    return path


def test_settings_load_paths_and_options_and_take_command_line_overrides(tmp_path):
    settings = load_settings(_config(tmp_path), device="cpu", limit=10, workers=None)
    assert settings.run == tmp_path / "runs" / "coco" and isinstance(settings.checkpoint, Path)
    assert (settings.batch_size, settings.gpu_memory_gib, settings.device, settings.limit) == (8, 5.5, "cpu", 10)
    assert settings.workers == 9  # an override of None keeps the file's value, here the default
    assert settings.layout.root == settings.run
    assert settings.dataset.limit == 10 and settings.dataset.val_root == tmp_path / "val"


def test_settings_reject_unknown_and_missing_keys(tmp_path):
    with pytest.raises(ValueError, match="unknown settings .*: bach_size"):
        load_settings(_config(tmp_path, CONFIG + "bach_size = 4\n"))
    with pytest.raises(ValueError, match="needs annotations"):
        load_settings(_config(tmp_path, CONFIG.replace('annotations = "{root}/ann.json"\n', "")))


def test_the_protocol_names_the_checkpoint_by_its_hash_and_fixes_the_benchmark(tmp_path):
    (tmp_path / "ckpt.pth").write_bytes(b"weights")
    protocol = load_settings(_config(tmp_path)).protocol()
    assert protocol["checkpoint_sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert (protocol["dataset"], protocol["seed"], protocol["limit"], protocol["folds"]) == ("coco", 44, None, 5)
    assert len(protocol["conditions"]) == 96 and protocol["conditions"][0] == ["clean", 0]
    assert (protocol["top_k"], protocol["knn_k"], protocol["knn_k_max"], protocol["theta"]) == (100, 100, 200, 0.3)


def test_the_repository_config_names_every_path():
    settings = load_settings(Path(__file__).parents[1] / "configs" / "coco.toml")
    assert settings.run.is_absolute() and settings.run.name == "coco"
    assert settings.gpu_memory_gib == 5.5 and settings.seed == 44
