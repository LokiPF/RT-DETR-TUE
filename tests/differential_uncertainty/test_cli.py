from __future__ import annotations

import pytest

import differential_uncertainty.cli as cli


def _benchmark_parser():
    parser = cli.build_parser()
    command = next(action for action in parser._actions if getattr(action, "choices", None))
    assert set(command.choices) == {"benchmark-coco"}
    return command.choices["benchmark-coco"]


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
