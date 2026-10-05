"""The Cityscapes-C configs load and point at each other (cityscapes/evaluation/design.md)."""
from pathlib import Path

from degradation_monitor.stages import detectors, iqa

FOLDER = Path(__file__).resolve().parents[1] / "cityscapes" / "evaluation"


def test_the_detectors_config_runs_yolo11m_on_cityscapes_against_rtdetrs_run():
    config = detectors.load_config(FOLDER / "cityscapes-detectors.toml")
    assert config.base.benchmark == "cityscapes" and config.detectors == ("yolo11m",)
    assert config.weights["yolo11m"].name == "yolo11m_cityscapes_100e.pt" and config.floors == {"yolo11m": 0.32}
    assert config.reference_run == config.base.run and config.run.name == "cityscapes-detectors"


def test_the_iqa_config_scores_both_detectors_on_cityscapes():
    config = iqa.load_config(FOLDER / "cityscapes-iqa.toml")
    assert config.base.benchmark == "cityscapes" and list(config.detectors) == ["rtdetrv2_r18", "yolo11m"]
    assert config.run.name == "cityscapes-iqa" and config.reference_run == config.base.run
