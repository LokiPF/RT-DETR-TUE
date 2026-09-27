import json

import numpy as np
import pytest
import torch
from PIL import Image

from differential_uncertainty.baselines import pipeline


class FakeTap:
    calls = 0

    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def run(self, arrays, batch_size=32):
        FakeTap.calls += 1
        n = len(arrays)
        brightness = np.array([a.mean() / 255.0 for a in arrays])
        logits = np.full((n, 300, 80), -6.0)
        logits[:, :3, 0] = (4.0 * brightness)[:, None]
        boxes = np.full((n, 300, 4), 0.25)
        pooled = np.random.default_rng(0).normal(size=(n, 512)) + brightness[:, None]
        return logits, boxes, pooled


def _settings(tmp_path, **changes):
    val = tmp_path / "val"
    val.mkdir(exist_ok=True)
    rng = np.random.default_rng(1)
    for index in range(4):
        Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)).save(val / f"{index:04d}.jpg")
    values = dict(output=tmp_path / "out", checkpoint=tmp_path / "ckpt.pth", train_images=tmp_path,
                  val_images=val, annotations=tmp_path / "ann.json", discopatch_root=tmp_path,
                  limit=2, workers=0, device="cpu", batch_size=16)
    values.update(changes)
    return pipeline.Settings(**values)


@pytest.fixture
def fakes(monkeypatch):
    FakeTap.calls = 0
    monkeypatch.setattr(pipeline, "DetectorTap", FakeTap)
    monkeypatch.setattr(pipeline, "_load_bank", lambda _s, _d: torch.nn.functional.normalize(torch.randn(256, 512), dim=1))


def test_test_phase_writes_one_complete_file_per_image_and_resumes(tmp_path, fakes):
    settings = _settings(tmp_path)
    pipeline.run_phase("test", settings)
    files = sorted((settings.output / "test").glob("*.npz"))
    assert len(files) == 2
    data = np.load(files[0])
    assert set(pipeline.TEST_KEYS) <= set(data.files)
    assert data["saod_top3"].shape == (96,) and data["knn"].shape == (96, 200)
    assert data["det_boxes"].shape == (96, 100, 4) and data["digests"].shape == (96,)
    FakeTap.calls = 0
    pipeline.run_phase("test", settings)
    assert FakeTap.calls == 0


def test_evaluation_uses_the_seeded_shuffle_and_the_limit(tmp_path):
    settings = _settings(tmp_path)
    names = [p.name for p in pipeline.evaluation(settings)]
    assert len(names) == 2 and set(names) <= {f"{i:04d}.jpg" for i in range(4)}


def test_test_phase_refuses_a_damaged_result_file(tmp_path, fakes):
    settings = _settings(tmp_path)
    pipeline.run_phase("test", settings)
    victim = sorted((settings.output / "test").glob("*.npz"))[0]
    victim.write_bytes(b"not an npz")
    with pytest.raises(ValueError, match=victim.name):
        pipeline.run_phase("test", settings)


def test_run_config_mismatch_stops_the_run_and_environment_is_recorded(tmp_path, fakes):
    pipeline.run_phase("test", _settings(tmp_path))
    environment = json.loads((tmp_path / "out" / "environment.json").read_text())
    assert "torch" in environment["packages"]
    with pytest.raises(ValueError, match="run_config.json"):
        pipeline.run_phase("test", _settings(tmp_path, seed=45))


def test_discopatch_phase_detects_changed_corruptions(tmp_path, fakes, monkeypatch):
    settings = _settings(tmp_path)
    pipeline.run_phase("test", settings)
    checkpoint = settings.output / "discopatch" / "DisCoPatch" / "Discriminator_coco.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"x")

    class FakeScorer:
        def __init__(self, *_a, **_k):
            pass

        def score(self, arrays, _image_id):
            return np.array([a.mean() / 255.0 for a in arrays])

    monkeypatch.setattr(pipeline, "DisCoPatchScorer", FakeScorer)
    victim = sorted((settings.output / "test").glob("*.npz"))[0]
    data = dict(np.load(victim))
    data["digests"] = np.array(["0" * 16] * 96)
    np.savez(victim, **data)
    with pytest.raises(ValueError, match="corruptions differ"):
        pipeline.run_phase("discopatch-scores", settings)


def test_entry_module_does_not_run_the_cli_when_imported_by_a_spawned_worker():
    import runpy
    runpy.run_module("differential_uncertainty.__main__", run_name="__mp_main__")


def test_worker_pool_returns_every_image_in_order_with_identical_corruptions(tmp_path):
    paths = sorted((_settings(tmp_path).val_images).glob("*.jpg"))[:3]
    in_process = list(pipeline._variant_stream(_settings(tmp_path, workers=0), paths))
    pooled = list(pipeline._variant_stream(_settings(tmp_path, workers=1), paths))  # 2 in flight < 3 images
    assert [name for name, _ in pooled] == [p.name for p in paths]
    for (_, expected), (_, actual) in zip(in_process, pooled):
        assert [pipeline.protocol.digest(a) for a in actual] == [pipeline.protocol.digest(a) for a in expected]


def _fake_scorer(monkeypatch, score):
    class FakeScorer:
        def __init__(self, *_a, **_k):
            pass

        def score(self, arrays, _image_id):
            return score(arrays)

    monkeypatch.setattr(pipeline, "DisCoPatchScorer", FakeScorer)


def _write_discriminator(settings, content=b"x"):
    checkpoint = settings.output / "discopatch" / "DisCoPatch" / "Discriminator_coco.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(content)
    return checkpoint


def test_training_refuses_to_overwrite_a_finished_discriminator(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    calls = []
    monkeypatch.setattr(pipeline, "train_discopatch", lambda *a, **k: calls.append(k))
    _write_discriminator(settings)
    with pytest.raises(ValueError, match="already exists"):
        pipeline.run_phase("train-discopatch", settings)
    assert calls == []


def test_a_lower_epoch_budget_does_not_invalidate_finished_phases(tmp_path, fakes, monkeypatch):
    monkeypatch.setattr(pipeline, "train_discopatch", lambda *a, **k: None)
    pipeline.run_phase("test", _settings(tmp_path))
    pipeline.run_phase("train-discopatch", _settings(tmp_path, epochs=30))
    assert json.loads((tmp_path / "out" / "discopatch" / "training.json").read_text())["epochs"] == 30


def test_discopatch_scores_stay_tied_to_one_checkpoint(tmp_path, fakes, monkeypatch):
    import hashlib
    settings = _settings(tmp_path)
    pipeline.run_phase("test", settings)
    checkpoint = _write_discriminator(settings, b"first")
    _fake_scorer(monkeypatch, lambda arrays: np.array([a.mean() / 255.0 for a in arrays]))
    pipeline.run_phase("discopatch-scores", settings)
    recorded = json.loads((settings.output / "test_dcp" / "checkpoint.json").read_text())
    assert recorded["sha1"] == hashlib.sha1(b"first").hexdigest()
    checkpoint.write_bytes(b"second")
    with pytest.raises(ValueError, match="checkpoint"):
        pipeline.run_phase("discopatch-scores", settings)


def test_discopatch_scores_refuse_non_finite_values(tmp_path, fakes, monkeypatch):
    settings = _settings(tmp_path)
    pipeline.run_phase("test", settings)
    _write_discriminator(settings)
    _fake_scorer(monkeypatch, lambda arrays: np.full(len(arrays), np.nan))
    with pytest.raises(ValueError, match="non-finite"):
        pipeline.run_phase("discopatch-scores", settings)
    assert not list((settings.output / "test_dcp").glob("*.npz"))
