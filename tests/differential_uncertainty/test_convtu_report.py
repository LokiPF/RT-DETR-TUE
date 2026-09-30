import json
from pathlib import Path

import numpy as np

from differential_uncertainty.baselines import pipeline as baselines
from differential_uncertainty.convtu import pipeline as convtu
from differential_uncertainty.convtu import report as pilot
from test_baselines_report import _tiny_run


def test_pilot_report_puts_the_fingerprint_next_to_the_baselines(tmp_path, monkeypatch):
    settings = _tiny_run(tmp_path, monkeypatch)   # 15 images through the fake test phase
    baselines.run_phase("report", settings)       # the full report: stored LRP threshold and per-fold lambda
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
