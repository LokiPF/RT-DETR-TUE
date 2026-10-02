import numpy as np
import pytest
import torch
from PIL import Image

from fakes import FakeBackbone, FakeTap
from degradation_monitor.datasets import coco
from degradation_monitor.detector.taps import EarlyChannelTaps
from degradation_monitor.method.statistics import KEYS
from degradation_monitor.settings import Settings
from degradation_monitor.stages import baselines as baseline_stages
from degradation_monitor.stages import common, run_stage
from degradation_monitor.stages import method as stage

WIDTHS = (2, 3, 4, 5)  # the fake backbone's four stages


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setattr(coco, "SPLIT_IMAGES", (("reserved", 2), ("bank", 6), ("zstats", 3)))
    monkeypatch.setattr(stage, "early_taps", lambda _settings: EarlyChannelTaps(FakeBackbone()))
    monkeypatch.setattr(baseline_stages, "DetectorTap", FakeTap)
    rng = np.random.default_rng(3)
    for folder, count in (("train", 12), ("val", 3)):
        (tmp_path / folder).mkdir()
        for index in range(count):
            Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)).save(tmp_path / folder / f"{index:04d}.jpg")
    (tmp_path / "ckpt.pth").write_bytes(b"weights")
    result = Settings(run=tmp_path / "run", checkpoint=tmp_path / "ckpt.pth", train_images=tmp_path / "train",
                      val_images=tmp_path / "val", annotations=tmp_path / "ann.json", discopatch_root=tmp_path,
                      limit=2, workers=0, device="cpu", batch_size=4)
    result.layout.knn_bank.parent.mkdir(parents=True)
    np.save(result.layout.knn_bank, rng.normal(size=(256, 512)).astype(np.float16))
    return result


def _statistics_of(paths) -> dict:
    prepared = common.PreparedImages(paths)
    with EarlyChannelTaps(FakeBackbone()) as taps:
        return stage.batch_statistics(taps, torch.stack([prepared[i] for i in range(len(prepared))]))


def test_the_method_reference_stores_the_seeded_bank_and_zstatistics_images_once(settings):
    run_stage("method-reference", settings)
    paths = sorted(settings.train_images.glob("*.jpg"))
    splits = coco.train_splits(len(paths), seed=44)
    for split, path, rows in (("bank", settings.layout.method_bank, 6), ("zstats", settings.layout.method_zstats, 3)):
        expected = _statistics_of([paths[i] for i in splits[split]])
        with np.load(path) as stored:
            assert set(stored.files) == set(KEYS)
            assert [stored[f"top_s{s}"].shape for s in range(1, 5)] == [(rows, c) for c in WIDTHS]
            assert all(stored[key].dtype == np.float32 for key in KEYS)
            for key in KEYS:
                np.testing.assert_allclose(stored[key], expected[key], rtol=1e-6)
    stamp = settings.layout.method_bank.stat().st_mtime_ns
    run_stage("method-reference", settings)
    assert settings.layout.method_bank.stat().st_mtime_ns == stamp


def test_the_method_pass_covers_every_evaluation_image_and_resumes(settings):
    with pytest.raises(ValueError, match="run the detector-pass stage first"):
        run_stage("method-pass", settings)
    run_stage("detector-pass", settings)
    run_stage("method-pass", settings)
    files = sorted(settings.layout.scores("method").glob("*.npz"))
    assert [f.stem for f in files] == sorted(p.stem for p in settings.dataset.evaluation_images())
    with np.load(files[0]) as stored:
        assert set(stored.files) == set(KEYS)
        for statistic in ("means", "top"):
            assert [stored[f"{statistic}_s{s}"].shape for s in range(1, 5)] == [(96, c) for c in WIDTHS]
        assert all(stored[key].dtype == np.float32 for key in KEYS)
    stamp = files[0].stat().st_mtime_ns
    run_stage("method-pass", settings)
    assert files[0].stat().st_mtime_ns == stamp


def test_an_images_clean_condition_gets_the_statistics_the_reference_stage_would_give_it(settings):
    run_stage("detector-pass", settings)
    run_stage("method-pass", settings)
    path = settings.dataset.evaluation_images()[0]
    expected = _statistics_of([path])
    with np.load(settings.layout.score_file("method", path)) as stored:
        for key in KEYS:
            np.testing.assert_allclose(stored[key][0], expected[key][0], rtol=1e-6)


def test_the_method_pass_detects_changed_corruptions(settings):
    run_stage("detector-pass", settings)
    victim = sorted(settings.layout.scores("detector").glob("*.npz"))[0]
    stored = dict(np.load(victim))
    stored["digests"] = stored["digests"].copy()
    stored["digests"][11] = "0" * 16
    np.savez(victim, **stored)
    with pytest.raises(ValueError, match="corruptions differ"):
        run_stage("method-pass", settings)
