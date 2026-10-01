from __future__ import annotations

import pytest

import differential_uncertainty.cli as cli


def _benchmark_parser():
    parser = cli.build_parser()
    command = next(action for action in parser._actions if getattr(action, "choices", None))
    assert set(command.choices) == {"benchmark-coco", "baselines-coco"}
    return command.choices["benchmark-coco"]


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


def test_cli_exposes_only_the_fixed_benchmark_contract():
    parser = _benchmark_parser()
    options = {option for action in parser._actions for option in action.option_strings}
    assert options == {
        "-h", "--help", "--checkpoint", "--coco-train-images",
        "--coco-val-images", "--reference-count", "--evaluation-count",
        "--output", "--device", "--batch-size", "--seed",
    }
    with pytest.raises(SystemExit) as error:
        cli.build_parser().parse_args(["benchmark-coco"])
    assert error.value.code == 2


def test_cli_forwards_values_and_defaults(monkeypatch, tmp_path):
    received = {}

    def fake_run(*args, **kwargs):
        received.update(args=args, kwargs=kwargs)

    monkeypatch.setattr(cli, "run_coco_benchmark", fake_run)
    argv = [
        "benchmark-coco", "--checkpoint", str(tmp_path / "model.pth"),
        "--coco-train-images", str(tmp_path / "train"),
        "--coco-val-images", str(tmp_path / "val"),
        "--reference-count", "12", "--evaluation-count", "9",
        "--output", str(tmp_path / "out"),
    ]
    assert cli.main(argv) == 0
    assert received["args"] == tuple(argv[index] for index in (2, 4, 6, 12))
    assert received["kwargs"] == {
        "reference_count": 12, "evaluation_count": 9,
        "device": "cuda:0", "batch_size": 1, "seed": 44,
    }


@pytest.mark.parametrize(("option", "value"), (("--batch-size", "0"), ("--seed", "-1")))
def test_cli_rejects_invalid_numbers(option, value, capsys, tmp_path):
    argv = [
        "benchmark-coco", "--checkpoint", "model.pth",
        "--coco-train-images", "train", "--coco-val-images", "val",
        "--reference-count", "1", "--evaluation-count", "1",
        "--output", str(tmp_path / "out"), option, value,
    ]
    with pytest.raises(SystemExit) as error:
        cli.main(argv)
    assert error.value.code == 2
    assert "must be" in capsys.readouterr().err


def test_cli_prints_plain_runtime_errors(monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "run_coco_benchmark",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad input")),
    )
    code = cli.main([
        "benchmark-coco", "--checkpoint", "model.pth",
        "--coco-train-images", "train", "--coco-val-images", "val",
        "--reference-count", "1", "--evaluation-count", "1", "--output", "out",
    ])
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
    phases = ("convtu-calibrate", "convtu-bank", "convtu-zstats", "convtu-scores", "convtu-report",
              "convtu-channels", "convtu-channels-report", "convtu-means")
    for phase in phases:
        assert cli.main(["baselines-coco", "--phase", phase, *base]) == 0
    assert tuple(seen) == phases
