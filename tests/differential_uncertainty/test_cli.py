from __future__ import annotations

import pytest

import differential_uncertainty.cli as cli


def test_cli_forwards_baselines_settings(monkeypatch, tmp_path):
    import differential_uncertainty.baselines.pipeline as pipeline
    received = {}
    monkeypatch.setattr(pipeline, "run_phase", lambda phase, settings: received.update(phase=phase, settings=settings))
    argv = ["baselines-coco", "--phase", "test", "--output", str(tmp_path / "out"),
            "--checkpoint", "ckpt.pth", "--coco-train-images", "train", "--coco-val-images", "val",
            "--coco-annotations", "ann.json", "--discopatch-root", "dcp", "--limit", "20", "--workers", "0"]
    assert cli.main(argv) == 0
    settings = received["settings"]
    assert received["phase"] == "test"
    assert (settings.limit, settings.workers, settings.epochs, settings.batch_size) == (20, 0, 65, 32)
    assert str(settings.annotations) == "ann.json" and str(settings.discopatch_root) == "dcp"


BASE = ["baselines-coco", "--phase", "test", "--output", "out", "--checkpoint", "c.pth", "--coco-train-images", "train",
        "--coco-val-images", "val", "--coco-annotations", "ann.json", "--discopatch-root", "dcp"]


@pytest.mark.parametrize(("option", "value"), (("--batch-size", "0"), ("--workers", "-1")))
def test_cli_rejects_invalid_numbers(option, value, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main([*BASE, option, value])
    assert error.value.code == 2
    assert "must be" in capsys.readouterr().err


def test_cli_prints_plain_runtime_errors(monkeypatch, capsys):
    import differential_uncertainty.baselines.pipeline as pipeline
    monkeypatch.setattr(pipeline, "run_phase",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad input")))
    code = cli.main(BASE)
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert captured.err == "error: bad input\n"


def test_cli_accepts_the_activation_monitor_phases(monkeypatch, tmp_path):
    import differential_uncertainty.baselines.pipeline as pipeline
    seen = []
    monkeypatch.setattr(pipeline, "run_phase", lambda phase, settings: seen.append(phase))
    base = ["--output", str(tmp_path / "out"), "--checkpoint", "c.pth", "--coco-train-images", "train",
            "--coco-val-images", "val", "--coco-annotations", "ann.json", "--discopatch-root", "dcp"]
    phases = ("hashemi-fit", "cdf-fit", "cdf-zstats", "activation-scores")
    for phase in phases:
        assert cli.main(["baselines-coco", "--phase", phase, *base]) == 0
    assert tuple(seen) == phases


def test_cli_accepts_the_convtu_phases(monkeypatch, tmp_path):
    import differential_uncertainty.baselines.pipeline as pipeline
    seen = []
    monkeypatch.setattr(pipeline, "run_phase", lambda phase, settings: seen.append(phase))
    base = ["--output", str(tmp_path / "out"), "--checkpoint", "c.pth", "--coco-train-images", "train",
            "--coco-val-images", "val", "--coco-annotations", "ann.json", "--discopatch-root", "dcp"]
    phases = ("convtu-calibrate", "convtu-bank", "convtu-zstats", "convtu-scores",
              "convtu-channels", "convtu-means", "convtu-conditioned-report")
    for phase in phases:
        assert cli.main(["baselines-coco", "--phase", phase, *base]) == 0
    assert tuple(seen) == phases
