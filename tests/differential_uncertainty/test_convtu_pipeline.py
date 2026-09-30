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
