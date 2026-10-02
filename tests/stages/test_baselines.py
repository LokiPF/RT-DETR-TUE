import hashlib
import json

import numpy as np
import pytest
import torch
from PIL import Image

from degradation_monitor import corruptions
from degradation_monitor.runs import SCORE_KEYS, Manifest
from degradation_monitor.settings import Settings
from degradation_monitor.stages import baselines as stage
from degradation_monitor.stages import common, run_stage


class FakeTap:
    """Detector stand-in: fixed boxes, brightness-driven confidence and pooled features."""
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

    def prepare(self, arrays):
        return torch.stack([torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 255.0
                            for a in arrays])

    def forward(self, batch):
        return self.run([(b.permute(1, 2, 0).numpy() * 255).astype(np.uint8) for b in batch])

    hidden_calls = 0

    def forward_hidden(self, batch):
        FakeTap.hidden_calls += 1
        n = batch.shape[0]
        level = batch.float().mean(dim=(1, 2, 3))
        generator = torch.Generator().manual_seed(0)

        def maps(channels, sizes):
            return [level.view(n, 1, 1, 1).expand(n, channels, s, s).clone()
                    + torch.randn(channels, s, s, generator=generator) for s in sizes]

        return {"decoder": torch.randn(300, 256, generator=generator) + 4.0 * level.view(n, 1, 1),
                "encoder": maps(8, (4, 2, 1)), "backbone": maps(3, (4, 4, 2, 2, 1))}


def _images(folder, count, shape, seed):
    folder.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    for index in range(count):
        Image.fromarray(rng.integers(0, 256, (*shape, 3), dtype=np.uint8)).save(folder / f"{index:04d}.jpg")
    return folder


def _settings(tmp_path, **changes):
    (tmp_path / "ckpt.pth").write_bytes(b"weights")
    values = dict(run=tmp_path / "run", checkpoint=tmp_path / "ckpt.pth",
                  train_images=_images(tmp_path / "train", 3, (40, 50), 3),
                  val_images=_images(tmp_path / "val", 4, (48, 64), 1), annotations=tmp_path / "ann.json",
                  discopatch_root=tmp_path, limit=2, workers=0, device="cpu", batch_size=16)
    values.update(changes)
    return Settings(**values)


def _bank(settings):
    path = settings.layout.knn_bank
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.random.default_rng(0).normal(size=(256, 512)).astype(np.float16))


@pytest.fixture
def fakes(monkeypatch):
    FakeTap.calls = FakeTap.hidden_calls = 0
    monkeypatch.setattr(stage, "DetectorTap", FakeTap)


def _detector(settings):
    _bank(settings)
    run_stage("detector-pass", settings)


def _fake_scorer(monkeypatch, score):
    class FakeScorer:
        def __init__(self, *_a, **_k):
            pass

        def score(self, arrays, _image_id):
            return score(arrays)

    monkeypatch.setattr(stage, "DisCoPatchScorer", FakeScorer)


def _discriminator(settings, content=b"x"):
    path = settings.layout.discopatch_checkpoint
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_the_detector_pass_writes_one_complete_file_per_image_and_resumes(tmp_path, fakes):
    settings = _settings(tmp_path)
    _detector(settings)
    files = sorted(settings.layout.scores("detector").glob("*.npz"))
    assert len(files) == 2
    with np.load(files[0]) as data:
        assert set(SCORE_KEYS["detector"]) <= set(data.files)
        assert data["saod_top3"].shape == (96,) and data["knn"].shape == (96, 200)
        assert data["det_boxes"].shape == (96, 100, 4) and data["digests"].shape == (96,)
    inputs = Manifest(settings.layout).read()["inputs"]["detector"]
    assert inputs == {"knn_bank": hashlib.sha1(settings.layout.knn_bank.read_bytes()).hexdigest()}
    FakeTap.calls = 0
    run_stage("detector-pass", settings)
    assert FakeTap.calls == 0


def test_the_detector_pass_refuses_a_damaged_result_file(tmp_path, fakes):
    settings = _settings(tmp_path)
    _detector(settings)
    victim = sorted(settings.layout.scores("detector").glob("*.npz"))[0]
    victim.write_bytes(b"not an npz")
    with pytest.raises(ValueError, match=victim.name):
        run_stage("detector-pass", settings)


def test_a_run_folder_made_with_another_seed_is_refused_and_the_environment_is_recorded(tmp_path, fakes):
    _detector(_settings(tmp_path))
    manifest = Manifest(_settings(tmp_path).layout).read()
    assert "torch" in manifest["environment"]["packages"] and manifest["protocol"]["seed"] == 44
    with pytest.raises(ValueError, match="another protocol"):
        run_stage("detector-pass", _settings(tmp_path, seed=45))


def test_a_later_pass_needs_the_detector_pass_first(tmp_path, fakes, monkeypatch):
    settings = _settings(tmp_path)
    _discriminator(settings)
    _fake_scorer(monkeypatch, lambda arrays: np.zeros(len(arrays)))
    with pytest.raises(ValueError, match="run the detector-pass stage first"):
        run_stage("discopatch-pass", settings)


def test_the_discopatch_pass_detects_changed_corruptions(tmp_path, fakes, monkeypatch):
    settings = _settings(tmp_path)
    _detector(settings)
    _discriminator(settings)
    _fake_scorer(monkeypatch, lambda arrays: np.array([a.mean() / 255.0 for a in arrays]))
    victim = sorted(settings.layout.scores("detector").glob("*.npz"))[0]
    data = dict(np.load(victim))
    data["digests"] = np.array(["0" * 16] * 96)
    np.savez(victim, **data)
    with pytest.raises(ValueError, match="corruptions differ"):
        run_stage("discopatch-pass", settings)


def test_the_worker_pool_returns_every_image_in_order_with_identical_corruptions(tmp_path):
    paths = sorted(_settings(tmp_path).val_images.glob("*.jpg"))[:3]
    in_process = list(common.variant_stream(_settings(tmp_path, workers=0), paths))
    pooled = list(common.variant_stream(_settings(tmp_path, workers=1), paths))  # 2 in flight < 3 images
    assert [name for name, _ in pooled] == [p.name for p in paths]
    for (_, expected), (_, actual) in zip(in_process, pooled):
        assert [corruptions.digest(a) for a in actual] == [corruptions.digest(a) for a in expected]


def test_training_refuses_to_overwrite_a_finished_discriminator(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    calls = []
    monkeypatch.setattr(stage, "train_discopatch", lambda *a, **k: calls.append(k))
    _discriminator(settings)
    with pytest.raises(ValueError, match="already exists"):
        run_stage("discopatch-train", settings)
    assert calls == []


def test_a_lower_epoch_budget_keeps_the_run_folder_and_the_trained_discriminator_is_linked(tmp_path, fakes,
                                                                                            monkeypatch):
    def fake_train(paths, models_dir, **kwargs):
        trained = models_dir / "DisCoPatch" / "Discriminator_coco.pt"
        trained.parent.mkdir(parents=True)
        trained.write_bytes(b"trained")
        return trained

    monkeypatch.setattr(stage, "train_discopatch", fake_train)
    _detector(_settings(tmp_path))
    settings = _settings(tmp_path, epochs=30)  # the epoch budget is not part of the protocol
    run_stage("discopatch-train", settings)
    assert settings.layout.discopatch_checkpoint.read_bytes() == b"trained"
    training = json.loads(settings.layout.discopatch_training.read_text())
    assert training["epochs"] == 30 and training["numerics"] == stage.discopatch.TRAINING_NUMERICS


def test_discopatch_scores_stay_tied_to_one_checkpoint(tmp_path, fakes, monkeypatch):
    settings = _settings(tmp_path)
    _detector(settings)
    checkpoint = _discriminator(settings, b"first")
    _fake_scorer(monkeypatch, lambda arrays: np.array([a.mean() / 255.0 for a in arrays]))
    run_stage("discopatch-pass", settings)
    recorded = Manifest(settings.layout).read()["inputs"]["discopatch"]
    assert recorded == {"discriminator": hashlib.sha1(b"first").hexdigest()}
    checkpoint.write_bytes(b"second")
    with pytest.raises(ValueError, match="changed since"):
        run_stage("discopatch-pass", settings)


def test_discopatch_scores_refuse_non_finite_values(tmp_path, fakes, monkeypatch):
    settings = _settings(tmp_path)
    _detector(settings)
    _discriminator(settings)
    _fake_scorer(monkeypatch, lambda arrays: np.full(len(arrays), np.nan))
    with pytest.raises(ValueError, match="non-finite"):
        run_stage("discopatch-pass", settings)
    assert not list(settings.layout.scores("discopatch").glob("*.npz"))


def test_the_hashemi_fit_stores_intervals_for_every_monitored_layer_once(tmp_path, fakes):
    settings = _settings(tmp_path)
    run_stage("hashemi-fit", settings)
    with np.load(settings.layout.hashemi_intervals) as data:
        assert data["decoder_mean"].shape == (300, 256) and data["encoder_s8_std"].shape == (8, 4, 4)
        assert data["encoder_s32_mean"].shape == (8, 1, 1) and int(data["images"]) == 3
    fit = json.loads(settings.layout.hashemi_fit.read_text())
    assert fit["k"] == 2.0 and fit["images"] == 3
    calls = FakeTap.hidden_calls
    run_stage("hashemi-fit", settings)
    assert FakeTap.hidden_calls == calls


def test_the_cdf_fit_stores_ranges_and_reference_cdfs_for_the_five_stages_once(tmp_path, fakes):
    settings = _settings(tmp_path)
    run_stage("cdf-fit", settings)
    with np.load(settings.layout.cdf_reference) as data:
        assert data["C1_cdf"].shape == (3, 1000) and data["C5_low"].shape == (3,)
        assert np.allclose(data["C3_cdf"][:, -1], 1.0) and int(data["images"]) == 3
    assert json.loads(settings.layout.cdf_fit.read_text())["bins"] == 1000
    assert FakeTap.hidden_calls == 2   # one ranges pass and one histogram pass over the single batch
    run_stage("cdf-fit", settings)
    assert FakeTap.hidden_calls == 2


def test_the_cdf_zstats_standardise_each_stage_on_clean_train_images_once(tmp_path, fakes):
    settings = _settings(tmp_path)
    with pytest.raises(ValueError, match="cdf-fit"):
        run_stage("cdf-zstats", settings)
    run_stage("cdf-fit", settings)
    run_stage("cdf-zstats", settings)
    stats = json.loads(settings.layout.cdf_zstats.read_text())
    assert stats["images"] == 3 and len(stats["mean"]) == 5 and len(stats["std"]) == 5
    assert all(s > 0 for s in stats["std"])
    calls = FakeTap.hidden_calls
    run_stage("cdf-zstats", settings)
    assert FakeTap.hidden_calls == calls


def test_the_activation_pass_covers_every_variant_and_stays_tied_to_its_fits(tmp_path, fakes):
    settings = _settings(tmp_path)
    _detector(settings)
    run_stage("hashemi-fit", settings)
    with pytest.raises(ValueError, match="cdf-fit"):
        run_stage("activation-pass", settings)
    run_stage("cdf-fit", settings)
    with pytest.raises(ValueError, match="cdf-zstats"):
        run_stage("activation-pass", settings)
    run_stage("cdf-zstats", settings)
    run_stage("activation-pass", settings)
    files = sorted(settings.layout.scores("activations").glob("*.npz"))
    assert len(files) == 2
    with np.load(files[0]) as data:
        assert set(data.files) == set(SCORE_KEYS["activations"])
        assert all(data[key].shape == (96,) for key in ("hashemi_decoder", "hashemi_encoder", "cdf_backbone",
                                                         "cdf_backbone_z"))
        assert data["hashemi_encoder_maps"].shape == (96, 3) and data["cdf_stages"].shape == (96, 5)
        assert np.all((data["hashemi_decoder"] >= 0) & (data["hashemi_decoder"] <= 1))
        assert np.all(data["cdf_backbone"] >= 0)
        assert np.allclose(data["cdf_stages"].sum(axis=1), data["cdf_backbone"])
    reference = settings.layout.cdf_reference
    reference.write_bytes(reference.read_bytes() + b"x")
    with pytest.raises(ValueError, match="changed since"):
        run_stage("activation-pass", settings)


def test_the_activation_pass_detects_changed_corruptions(tmp_path, fakes):
    settings = _settings(tmp_path)
    _detector(settings)
    for name in ("hashemi-fit", "cdf-fit", "cdf-zstats"):
        run_stage(name, settings)
    victim = sorted(settings.layout.scores("detector").glob("*.npz"))[0]
    data = dict(np.load(victim))
    data["digests"] = np.array(["0" * 16] * 96)
    np.savez(victim, **data)
    with pytest.raises(ValueError, match="corruptions differ"):
        run_stage("activation-pass", settings)


def test_timing_reports_both_activation_monitors_once_they_are_fitted(tmp_path, fakes):
    settings = _settings(tmp_path)
    _bank(settings)
    run_stage("hashemi-fit", settings)
    run_stage("cdf-fit", settings)
    run_stage("timing", settings)
    timing = json.loads(settings.layout.timing.read_text())
    assert {"detector_ms", "detector_plus_hashemi_ms", "detector_plus_cdf_ms"} <= set(timing)
    assert "discopatch_ms" not in timing


def test_the_check_names_a_missing_input_before_loading_the_detector(tmp_path, fakes):
    settings = _settings(tmp_path)  # writes no annotation file
    with pytest.raises(ValueError, match="annotations does not exist"):
        run_stage("check", settings)
    assert FakeTap.calls == 0


def test_the_gpu_cap_limits_the_process_to_its_share_of_the_card(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda device: type("Properties", (), {"total_memory": 32 * 2**30})())
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction",
                        lambda fraction, device=None: calls.append((fraction, device)))
    common.cap_gpu_memory("cuda:0", 5.5)
    common.cap_gpu_memory("cuda:0", None)
    common.cap_gpu_memory("cpu", 5.5)
    assert calls == [(5.5 / 32, torch.device("cuda:0"))]
