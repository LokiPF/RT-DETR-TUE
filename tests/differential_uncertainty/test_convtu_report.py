import json
from pathlib import Path

import numpy as np

from differential_uncertainty.baselines import pipeline as baselines
from differential_uncertainty.convtu import channels
from differential_uncertainty.convtu import pipeline as convtu
from differential_uncertainty.convtu import report as pilot
from test_baselines_report import _tiny_run


def _fake_pilot(tmp_path, monkeypatch):
    """The tiny run's full report, then fake conv-TU scores and a fake calibration for its 15 images."""
    settings = _tiny_run(tmp_path, monkeypatch)
    baselines.run_phase("report", settings)
    monkeypatch.setattr(convtu, "PILOT_IMAGES", 15)
    monkeypatch.setattr(pilot, "BOOTSTRAP_SAMPLES", 5)
    folder = settings.output / "test_convtu"
    folder.mkdir()
    rng = np.random.default_rng(0)
    for path in convtu.pilot_images(settings):
        layers = {rep: rng.normal(size=(96, 4)) + np.linspace(0, 2, 96)[:, None] for rep in ("mst", "edges", "acts", "means")}
        arrays = {f"{rep}_layers": values for rep, values in layers.items()}
        arrays.update({rep: values.sum(axis=1) for rep, values in layers.items()})
        np.savez(folder / f"{Path(path.name).stem}.npz", **arrays)
    (settings.output / "convtu").mkdir()
    (settings.output / "convtu" / "calibration.json").write_text(json.dumps(
        {"fraction": 0.01, "cut_margin": 0.5, "layers": {"s1": {"k": 32768}}}))
    return settings


def test_pilot_report_puts_the_fingerprint_next_to_the_baselines(tmp_path, monkeypatch):
    settings = _fake_pilot(tmp_path, monkeypatch)
    baselines.run_phase("convtu-report", settings)
    results = settings.output / "results_convtu"
    summary = json.loads((results / "summary.json").read_text())
    assert summary["images"] == 15 and summary["lrp_threshold"] == json.loads(
        (settings.output / "results" / "summary.json").read_text())["lrp_threshold"]
    text = (results / "report.md").read_text()
    assert text.startswith("# Conv TU pilot") and "Conv TU: top 1% of the diagram" in text
    assert "Depth: one layer at a time" in text and "ContrastiveConf" in text
    assert len((results / "depth.csv").read_text().splitlines()) == 1 + 16
    assert "convtu_mst:auroc_common" in (results / "intervals.csv").read_text()


def test_channels_report_adds_eight_rows(tmp_path, monkeypatch):
    settings = _fake_pilot(tmp_path, monkeypatch)
    rng = np.random.default_rng(1)
    widths = {f"{statistic}_s{stage}": (48 if statistic == "grid" else 3)
              for statistic in channels.STATISTICS for stage in range(1, 5)}
    np.savez(convtu.channels_bank_path(settings), **{k: rng.normal(size=(30, w)) for k, w in widths.items()})
    np.savez(convtu.channels_zstats_path(settings), **{k: rng.normal(size=(10, w)) for k, w in widths.items()})
    folder = settings.output / convtu.CHANNELS_FOLDER
    folder.mkdir()
    for path in convtu.pilot_images(settings):
        np.savez(folder / f"{Path(path.name).stem}.npz",
                 **{k: rng.normal(size=(96, w)) + np.linspace(0, 3, 96)[:, None] for k, w in widths.items()})

    baselines.run_phase("convtu-channels-report", settings)

    results = settings.output / "results_convtu_channels"
    text = (results / "report.md").read_text()
    assert text.startswith("# Conv TU pilot: channel statistics")
    assert "Channel means vs own training average" in text and "Conv TU: top 1% of the diagram" in text
    assert len((results / "depth.csv").read_text().splitlines()) == 1 + 16 + 32
    assert "ch_means_own:auroc_common" in (results / "intervals.csv").read_text()


def test_channels_report_counts_floored_dimensions_and_adds_rows_without_them(tmp_path, monkeypatch):
    settings = _fake_pilot(tmp_path, monkeypatch)
    rng = np.random.default_rng(2)
    widths = {f"{statistic}_s{stage}": (48 if statistic == "grid" else 3)
              for statistic in channels.STATISTICS for stage in range(1, 5)}
    bank = {k: rng.normal(size=(30, w)) for k, w in widths.items()}
    zstats = {k: rng.normal(size=(10, w)) for k, w in widths.items()}
    bank["p99_s4"][:, 0] = 0.0  # one dimension dead on every clean image
    zstats["p99_s4"][:, 0] = 0.0
    np.savez(convtu.channels_bank_path(settings), **bank)
    np.savez(convtu.channels_zstats_path(settings), **zstats)
    folder = settings.output / convtu.CHANNELS_FOLDER
    folder.mkdir()
    for path in convtu.pilot_images(settings):
        np.savez(folder / f"{Path(path.name).stem}.npz",
                 **{k: rng.normal(size=(96, w)) + np.linspace(0, 3, 96)[:, None] for k, w in widths.items()})

    baselines.run_phase("convtu-channels-report", settings)

    results = settings.output / "results_convtu_channels"
    summary = json.loads((results / "summary.json").read_text())
    assert summary["channel_floored_dimensions"]["p99_s4"] == 1 and summary["channel_floored_dimensions"]["top_s4"] == 0
    assert summary["channel_std_floor"] == channels.STD_FLOOR
    intervals = (results / "intervals.csv").read_text()
    assert "ch_p99_own_live:auroc_common" in intervals and "ch_top_own_live" not in intervals
    assert "floored dimensions left out" in (results / "report.md").read_text()
