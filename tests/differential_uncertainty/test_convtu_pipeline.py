import json

import numpy as np
import pytest
from PIL import Image

from convtu_fakes import FakeBackbone
from differential_uncertainty.baselines import pipeline as baselines
from differential_uncertainty.convtu import pipeline as convtu
from differential_uncertainty.convtu.tap import ConvInputs


@pytest.fixture
def small(monkeypatch, tmp_path):
    monkeypatch.setattr(convtu, "CALIBRATION_IMAGES", 2)
    monkeypatch.setattr(convtu, "BANK_IMAGES", 6)
    monkeypatch.setattr(convtu, "ZSTAT_IMAGES", 3)
    monkeypatch.setattr(convtu, "_conv_inputs", lambda _settings: ConvInputs(FakeBackbone()))
    rng = np.random.default_rng(3)
    for folder, count in (("train", 12), ("val", 3)):
        (tmp_path / folder).mkdir()
        for index in range(count):
            Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)).save(tmp_path / folder / f"{index:04d}.jpg")
    return baselines.Settings(output=tmp_path / "out", checkpoint=tmp_path / "ckpt.pth",
                              train_images=tmp_path / "train", val_images=tmp_path / "val",
                              annotations=tmp_path / "ann.json", discopatch_root=tmp_path,
                              limit=2, workers=0, device="cpu", batch_size=4)


def test_train_splits_are_disjoint_seeded_and_sized(monkeypatch):
    monkeypatch.setattr(convtu, "CALIBRATION_IMAGES", 2)
    monkeypatch.setattr(convtu, "BANK_IMAGES", 3)
    monkeypatch.setattr(convtu, "ZSTAT_IMAGES", 2)
    splits = convtu.train_splits(10, seed=44)
    assert [len(splits[k]) for k in ("calibration", "bank", "zstats")] == [2, 3, 2]
    assert len(set(np.concatenate(list(splits.values())).tolist())) == 7
    again = convtu.train_splits(10, seed=44)
    assert all(np.array_equal(splits[k], again[k]) for k in splits)
    with pytest.raises(ValueError, match="needs 7 train images"):
        convtu.train_splits(6, seed=44)


def test_clean_phases_write_calibration_bank_and_zstats_and_resume(small):
    for phase in ("convtu-calibrate", "convtu-bank", "convtu-zstats"):
        baselines.run_phase(phase, small)
    calibration = json.loads(convtu.calibration_path(small).read_text())
    assert calibration["images"] == 2 and set(calibration["layers"]) == {"s1", "s2", "s3", "s4"}
    s1 = calibration["layers"]["s1"]
    assert s1["nodes"] == 1024 and s1["k"] == 1023 and s1["cut"] > 0
    assert s1["edges_read"]["max"] >= s1["edges_read"]["min"] > 0
    with np.load(convtu.bank_path(small)) as bank:
        assert bank["mst_s1"].shape == (6, 1023) and bank["means_s4"].shape == (6, 5)
    zstats = json.loads(convtu.zstats_path(small).read_text())
    assert zstats["images"] == 3 and set(zstats["stats"]) == {"mst", "edges", "acts", "means"}
    assert all(value > 0 for value in zstats["stats"]["mst"]["std"])
    stamp = convtu.bank_path(small).stat().st_mtime_ns
    baselines.run_phase("convtu-bank", small)
    assert convtu.bank_path(small).stat().st_mtime_ns == stamp


def test_bank_needs_the_calibration(small):
    with pytest.raises(ValueError, match="convtu-calibrate phase first"):
        baselines.run_phase("convtu-bank", small)


def test_zstats_refuse_a_bank_from_another_calibration(small):
    for phase in ("convtu-calibrate", "convtu-bank"):
        baselines.run_phase(phase, small)
    path = convtu.calibration_path(small)
    record = json.loads(path.read_text())
    record["layers"]["s1"]["cut"] *= 2
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="calibration changed"):
        baselines.run_phase("convtu-zstats", small)


import torch

from convtu_fakes import FakeTap


@pytest.fixture
def detector(monkeypatch):
    monkeypatch.setattr(baselines, "DetectorTap", FakeTap)
    monkeypatch.setattr(baselines, "_load_bank",
                        lambda _s, _d: torch.nn.functional.normalize(torch.randn(256, 512), dim=1))


def _pilot(settings):
    for phase in ("test", "convtu-calibrate", "convtu-bank", "convtu-zstats", "convtu-scores"):
        baselines.run_phase(phase, settings)


def test_scores_phase_writes_every_condition_and_resumes(small, detector, monkeypatch):
    monkeypatch.setattr(convtu, "PILOT_IMAGES", 2)
    _pilot(small)
    files = sorted((small.output / "test_convtu").glob("*.npz"))
    assert len(files) == 2
    with np.load(files[0]) as scores:
        assert scores["mst"].shape == (96,) and scores["mst_layers"].shape == (96, 4)
        assert scores["tau_k"].shape == (96, 4) and np.isfinite(scores["acts"]).all()
    stamp = files[0].stat().st_mtime_ns
    baselines.run_phase("convtu-scores", small)
    assert files[0].stat().st_mtime_ns == stamp


def test_scores_phase_detects_changed_corruptions(small, detector, monkeypatch):
    monkeypatch.setattr(convtu, "PILOT_IMAGES", 2)
    _pilot(small)
    stem = sorted((small.output / "test_convtu").glob("*.npz"))[0].stem
    (small.output / "test_convtu" / f"{stem}.npz").unlink()
    stored = dict(np.load(small.output / "test" / f"{stem}.npz"))
    stored["digests"] = stored["digests"].copy()
    stored["digests"][5] = "0" * 16
    np.savez(small.output / "test" / f"{stem}.npz", **stored)
    with pytest.raises(ValueError, match="corruptions differ"):
        baselines.run_phase("convtu-scores", small)


def test_scores_phase_refuses_changed_fits(small, detector, monkeypatch):
    monkeypatch.setattr(convtu, "PILOT_IMAGES", 2)
    _pilot(small)
    path = convtu.zstats_path(small)
    record = json.loads(path.read_text())
    record["images"] += 1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="changed since"):
        baselines.run_phase("convtu-scores", small)


def test_gpu_memory_record_reports_allocated_and_reserved_peaks(monkeypatch):
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: 3 * 2**30)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device=None: 4 * 2**30)
    assert convtu._gpu_memory_record("cuda:0") == {"peak_gpu_allocated_gib": 3.0, "peak_gpu_reserved_gib": 4.0}
    assert convtu._gpu_memory_record("cpu") == {"peak_gpu_allocated_gib": None, "peak_gpu_reserved_gib": None}


def test_gpu_cap_limits_the_process_to_its_share_of_the_card(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda device: type("Properties", (), {"total_memory": 32 * 2**30})())
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction",
                        lambda fraction, device=None: calls.append((fraction, device)))
    convtu._cap_gpu_memory("cuda:0")
    assert calls == [(convtu.GPU_MEMORY_CAP_GIB / 32, torch.device("cuda:0"))]
    convtu._cap_gpu_memory("cpu")
    assert len(calls) == 1


def test_every_gpu_phase_caps_its_memory_first(small, detector, monkeypatch):
    monkeypatch.setattr(convtu, "PILOT_IMAGES", 2)
    capped = []
    monkeypatch.setattr(convtu, "_cap_gpu_memory", lambda device: capped.append(device))
    _pilot(small)
    assert capped == ["cpu"] * 4
    calibration = json.loads(convtu.calibration_path(small).read_text())
    assert calibration["peak_gpu_reserved_gib"] is None and "peak_gpu_gib" not in calibration


def _channels(settings):
    for phase in ("test", "convtu-channels"):
        baselines.run_phase(phase, settings)


def test_channels_phase_stores_every_statistic_and_resumes(small, detector, monkeypatch):
    monkeypatch.setattr(convtu, "PILOT_IMAGES", 2)
    capped = []
    monkeypatch.setattr(convtu, "_cap_gpu_memory", lambda device: capped.append(device))
    _channels(small)
    assert capped == ["cpu"]
    with np.load(convtu.channels_bank_path(small)) as bank:
        assert bank["means_s1"].shape == (6, 2) and bank["grid_s1"].shape == (6, 32)
        assert bank["top_s4"].shape == (6, 5) and bank["p99_s4"].shape == (6, 5)
    with np.load(convtu.channels_zstats_path(small)) as zstats:
        assert zstats["means_s2"].shape == (3, 3)
    files = sorted((small.output / convtu.CHANNELS_FOLDER).glob("*.npz"))
    assert len(files) == 2
    with np.load(files[0]) as stats:
        assert set(stats.files) == set(convtu.CHANNEL_KEYS)
        assert stats["means_s4"].shape == (96, 5) and stats["grid_s4"].shape == (96, 80)
        assert np.isfinite(stats["grid_s1"]).all()
    stamp = files[0].stat().st_mtime_ns
    baselines.run_phase("convtu-channels", small)
    assert files[0].stat().st_mtime_ns == stamp


def test_channels_phase_detects_changed_corruptions(small, detector, monkeypatch):
    monkeypatch.setattr(convtu, "PILOT_IMAGES", 2)
    _channels(small)
    stem = sorted((small.output / convtu.CHANNELS_FOLDER).glob("*.npz"))[0].stem
    (small.output / convtu.CHANNELS_FOLDER / f"{stem}.npz").unlink()
    stored = dict(np.load(small.output / "test" / f"{stem}.npz"))
    stored["digests"] = stored["digests"].copy()
    stored["digests"][7] = "0" * 16
    np.savez(small.output / "test" / f"{stem}.npz", **stored)
    with pytest.raises(ValueError, match="corruptions differ"):
        baselines.run_phase("convtu-channels", small)


def test_channels_phase_refuses_a_bank_that_differs_from_the_pilot(small, detector, monkeypatch):
    monkeypatch.setattr(convtu, "PILOT_IMAGES", 2)
    for phase in ("test", "convtu-calibrate", "convtu-bank"):
        baselines.run_phase(phase, small)
    stored = dict(np.load(convtu.bank_path(small)))
    stored["means_s2"] = stored["means_s2"] * 1.01
    np.savez(convtu.bank_path(small), **stored)
    with pytest.raises(ValueError, match="differ from the pilot's bank"):
        baselines.run_phase("convtu-channels", small)


def test_gpu_cap_takes_a_smaller_share_when_a_phase_asks_for_one(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda device: type("Properties", (), {"total_memory": 32 * 2**30})())
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction",
                        lambda fraction, device=None: calls.append((fraction, device)))
    convtu._cap_gpu_memory("cuda:0", 1.5)
    assert calls == [(1.5 / 32, torch.device("cuda:0"))]


def _means(settings):
    for phase in ("test", "convtu-means"):
        baselines.run_phase(phase, settings)


def test_means_phase_covers_every_evaluation_image_and_resumes(small, detector, monkeypatch):
    monkeypatch.setattr(convtu, "PILOT_IMAGES", 1)  # the confirmation is not limited to the pilot's images
    capped = []
    monkeypatch.setattr(convtu, "_cap_gpu_memory",
                        lambda device, gib=convtu.GPU_MEMORY_CAP_GIB: capped.append((device, gib)))
    _means(small)
    assert capped == [("cpu", convtu.MEANS_GPU_CAP_GIB)]
    files = sorted((small.output / convtu.MEANS_FOLDER).glob("*.npz"))
    assert [f.stem for f in files] == sorted(p.stem for p in baselines.evaluation(small)) and len(files) == 2
    with np.load(files[0]) as stats:
        assert set(stats.files) == set(convtu.MEANS_KEYS)
        assert [stats[key].shape for key in convtu.MEANS_KEYS] == [(96, 2), (96, 3), (96, 4), (96, 5)]
        assert all(stats[key].dtype == np.float32 for key in convtu.MEANS_KEYS)
    stamp = files[0].stat().st_mtime_ns
    baselines.run_phase("convtu-means", small)
    assert files[0].stat().st_mtime_ns == stamp


def test_means_phase_reproduces_the_channel_statistics_means_exactly(small, detector, monkeypatch):
    monkeypatch.setattr(convtu, "PILOT_IMAGES", 2)
    _channels(small)
    baselines.run_phase("convtu-means", small)
    paths = sorted((small.output / convtu.CHANNELS_FOLDER).glob("*.npz"))
    assert len(paths) == 2
    for path in paths:
        with np.load(path) as full, np.load(small.output / convtu.MEANS_FOLDER / path.name) as means:
            for key in convtu.MEANS_KEYS:
                np.testing.assert_array_equal(means[key], full[key])


def test_means_phase_detects_changed_corruptions(small, detector, monkeypatch):
    _means(small)
    stem = sorted((small.output / convtu.MEANS_FOLDER).glob("*.npz"))[0].stem
    (small.output / convtu.MEANS_FOLDER / f"{stem}.npz").unlink()
    stored = dict(np.load(small.output / "test" / f"{stem}.npz"))
    stored["digests"] = stored["digests"].copy()
    stored["digests"][11] = "0" * 16
    np.savez(small.output / "test" / f"{stem}.npz", **stored)
    with pytest.raises(ValueError, match="corruptions differ"):
        baselines.run_phase("convtu-means", small)


def test_channel_statistics_reproduce_the_pilot_on_the_fake_backbone(small, detector, monkeypatch):
    from differential_uncertainty.baselines import report as baseline_report
    from differential_uncertainty.baselines.activation_cdf import stage_zstats
    from differential_uncertainty.convtu import channels
    from differential_uncertainty.convtu.features import knn_scores
    monkeypatch.setattr(convtu, "PILOT_IMAGES", 2)
    _pilot(small)
    baselines.run_phase("convtu-channels", small)
    layers = [f"s{stage}" for stage in range(1, 5)]
    with np.load(convtu.bank_path(small)) as pilot_bank, np.load(convtu.channels_bank_path(small)) as bank:
        for layer in layers:
            np.testing.assert_allclose(bank[f"means_{layer}"], pilot_bank[f"means_{layer}"], rtol=1e-6)
        bank = dict(bank)
    with np.load(convtu.channels_zstats_path(small)) as zstats:
        zstats = dict(zstats)
    clean = np.stack([knn_scores(torch.from_numpy(zstats[f"means_{layer}"]), torch.from_numpy(bank[f"means_{layer}"]))
                      for layer in layers], axis=1)
    mean, std = stage_zstats(clean)
    stored = json.loads(convtu.zstats_path(small).read_text())["stats"]["means"]
    np.testing.assert_allclose(mean, stored["mean"], rtol=1e-6)
    np.testing.assert_allclose(std, stored["std"], rtol=1e-6)
    names = [p.name for p in convtu.pilot_images(small)]
    test = baseline_report._stack(small.output / convtu.CHANNELS_FOLDER, names, convtu.CHANNEL_KEYS)
    _, per_layer = channels.channel_method_scores(bank, zstats, test, layers)
    pilot = baseline_report._stack(small.output / "test_convtu", names, ("means_layers",))["means_layers"]
    np.testing.assert_allclose(per_layer["ch_means_knn"], pilot, rtol=1e-6)
