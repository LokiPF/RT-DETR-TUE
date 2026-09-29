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
    training = json.loads((tmp_path / "out" / "discopatch" / "training.json").read_text())
    assert training["epochs"] == 30
    assert training["numerics"] == pipeline.discopatch.TRAINING_NUMERICS


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


class HiddenFakeTap(FakeTap):
    """FakeTap plus the hidden-layer interface; every activation shifts with image brightness."""
    hidden_calls = 0

    def prepare(self, arrays):
        return torch.stack([torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 255.0
                            for a in arrays])

    def forward(self, batch):
        return self.run([(b.permute(1, 2, 0).numpy() * 255).astype(np.uint8) for b in batch])

    def forward_hidden(self, batch):
        HiddenFakeTap.hidden_calls += 1
        n = batch.shape[0]
        level = batch.float().mean(dim=(1, 2, 3))
        generator = torch.Generator().manual_seed(0)

        def maps(channels, sizes):
            return [level.view(n, 1, 1, 1).expand(n, channels, s, s).clone()
                    + torch.randn(channels, s, s, generator=generator) for s in sizes]

        return {"decoder": torch.randn(300, 256, generator=generator) + 4.0 * level.view(n, 1, 1),
                "encoder": maps(8, (4, 2, 1)), "backbone": maps(3, (4, 4, 2, 2, 1))}


def _train_images(tmp_path, count=3):
    folder = tmp_path / "train"
    folder.mkdir(exist_ok=True)
    rng = np.random.default_rng(3)
    for index in range(count):
        Image.fromarray(rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)).save(folder / f"{index:04d}.jpg")
    return folder


@pytest.fixture
def hidden_fakes(monkeypatch):
    HiddenFakeTap.calls = HiddenFakeTap.hidden_calls = 0
    monkeypatch.setattr(pipeline, "DetectorTap", HiddenFakeTap)
    monkeypatch.setattr(pipeline, "_load_bank", lambda _s, _d: torch.nn.functional.normalize(torch.randn(256, 512), dim=1))


def test_hashemi_fit_stores_intervals_for_every_monitored_layer_once(tmp_path, hidden_fakes):
    settings = _settings(tmp_path, train_images=_train_images(tmp_path))
    pipeline.run_phase("hashemi-fit", settings)
    with np.load(settings.output / "hashemi" / "intervals.npz") as data:
        assert data["decoder_mean"].shape == (300, 256) and data["encoder_s8_std"].shape == (8, 4, 4)
        assert data["encoder_s32_mean"].shape == (8, 1, 1) and int(data["images"]) == 3
    fit = json.loads((settings.output / "hashemi" / "fit.json").read_text())
    assert fit["k"] == 2.0 and fit["images"] == 3
    calls = HiddenFakeTap.hidden_calls
    pipeline.run_phase("hashemi-fit", settings)
    assert HiddenFakeTap.hidden_calls == calls


def test_cdf_fit_stores_ranges_and_reference_cdfs_for_the_five_stages_once(tmp_path, hidden_fakes):
    settings = _settings(tmp_path, train_images=_train_images(tmp_path))
    pipeline.run_phase("cdf-fit", settings)
    with np.load(settings.output / "cdf" / "reference.npz") as data:
        assert data["C1_cdf"].shape == (3, 1000) and data["C5_low"].shape == (3,)
        assert np.allclose(data["C3_cdf"][:, -1], 1.0) and int(data["images"]) == 3
    assert json.loads((settings.output / "cdf" / "fit.json").read_text())["bins"] == 1000
    assert HiddenFakeTap.hidden_calls == 2   # one ranges pass and one histogram pass over the single batch
    pipeline.run_phase("cdf-fit", settings)
    assert HiddenFakeTap.hidden_calls == 2


def test_activation_scores_cover_every_variant_and_stay_tied_to_their_fits(tmp_path, hidden_fakes):
    settings = _settings(tmp_path, train_images=_train_images(tmp_path))
    pipeline.run_phase("test", settings)
    pipeline.run_phase("hashemi-fit", settings)
    with pytest.raises(ValueError, match="cdf-fit"):
        pipeline.run_phase("activation-scores", settings)
    pipeline.run_phase("cdf-fit", settings)
    with pytest.raises(ValueError, match="cdf-zstats"):
        pipeline.run_phase("activation-scores", settings)
    pipeline.run_phase("cdf-zstats", settings)
    pipeline.run_phase("activation-scores", settings)
    files = sorted((settings.output / "test_activation").glob("*.npz"))
    assert len(files) == 2
    with np.load(files[0]) as data:
        assert set(data.files) == set(pipeline.ACTIVATION_KEYS)
        assert all(data[key].shape == (96,) for key in ("hashemi_decoder", "hashemi_encoder", "cdf_backbone", "cdf_backbone_z"))
        assert data["hashemi_encoder_maps"].shape == (96, 3) and data["cdf_stages"].shape == (96, 5)
        assert np.all((data["hashemi_decoder"] >= 0) & (data["hashemi_decoder"] <= 1))
        assert np.all(data["cdf_backbone"] >= 0)
        assert np.allclose(data["cdf_stages"].sum(axis=1), data["cdf_backbone"])
    reference = settings.output / "cdf" / "reference.npz"
    reference.write_bytes(reference.read_bytes() + b"x")
    with pytest.raises(ValueError, match="changed since"):
        pipeline.run_phase("activation-scores", settings)


def test_activation_scores_detect_changed_corruptions(tmp_path, hidden_fakes):
    settings = _settings(tmp_path, train_images=_train_images(tmp_path))
    pipeline.run_phase("test", settings)
    pipeline.run_phase("hashemi-fit", settings)
    pipeline.run_phase("cdf-fit", settings)
    pipeline.run_phase("cdf-zstats", settings)
    victim = sorted((settings.output / "test").glob("*.npz"))[0]
    data = dict(np.load(victim))
    data["digests"] = np.array(["0" * 16] * 96)
    np.savez(victim, **data)
    with pytest.raises(ValueError, match="corruptions differ"):
        pipeline.run_phase("activation-scores", settings)


def test_timing_reports_both_activation_monitors_once_they_are_fitted(tmp_path, hidden_fakes):
    settings = _settings(tmp_path, train_images=_train_images(tmp_path))
    pipeline.run_phase("hashemi-fit", settings)
    pipeline.run_phase("cdf-fit", settings)
    pipeline.run_phase("timing", settings)
    timing = json.loads((settings.output / "timing.json").read_text())
    assert {"detector_ms", "detector_plus_hashemi_ms", "detector_plus_cdf_ms"} <= set(timing)
    assert "discopatch_ms" not in timing


def test_cdf_zstats_standardise_each_stage_on_clean_train_images_once(tmp_path, hidden_fakes):
    settings = _settings(tmp_path, train_images=_train_images(tmp_path))
    with pytest.raises(ValueError, match="cdf-fit"):
        pipeline.run_phase("cdf-zstats", settings)
    pipeline.run_phase("cdf-fit", settings)
    pipeline.run_phase("cdf-zstats", settings)
    stats = json.loads((settings.output / "cdf" / "zstats.json").read_text())
    assert stats["images"] == 3 and len(stats["mean"]) == 5 and len(stats["std"]) == 5
    assert all(s > 0 for s in stats["std"])
    calls = HiddenFakeTap.hidden_calls
    pipeline.run_phase("cdf-zstats", settings)
    assert HiddenFakeTap.hidden_calls == calls
