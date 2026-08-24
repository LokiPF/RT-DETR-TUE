from __future__ import annotations

import pytest

import differential_uncertainty.cli as cli
from differential_uncertainty.cli import build_parser, main


def _run_parser(parser):
    subparsers = [
        action for action in parser._actions if hasattr(action, "choices")
    ]
    return next(action for action in subparsers if action.choices).choices["run"]


def test_cli_has_exactly_one_run_command_and_only_runtime_controls():
    parser = build_parser()
    command_action = next(
        action for action in parser._actions if getattr(action, "choices", None)
    )
    assert set(command_action.choices) == {"run"}
    run = _run_parser(parser)
    options = {
        option for action in run._actions for option in action.option_strings
    }
    assert {
        "--reference-manifest",
        "--evaluation-manifest",
        "--checkpoint",
        "--output-dir",
        "--device",
        "--batch-size",
        "--shard-size",
    } <= options
    assert not {
        "--k",
        "--layer",
        "--bins",
        "--score",
        "--orientation",
        "--blur-radii",
    } & options


def test_cli_forwards_runtime_controls_to_the_fixed_pipeline(monkeypatch, tmp_path):
    received = {}

    def fake_run(*args, **kwargs):
        received["args"] = args
        received["kwargs"] = kwargs

    monkeypatch.setattr(cli, "run_pipeline", fake_run)
    values = [
        "run",
        "--reference-manifest",
        str(tmp_path / "reference.csv"),
        "--evaluation-manifest",
        str(tmp_path / "evaluation.csv"),
        "--checkpoint",
        str(tmp_path / "model.pth"),
        "--output-dir",
        str(tmp_path / "run"),
        "--device",
        "cpu",
        "--batch-size",
        "3",
        "--shard-size",
        "7",
    ]

    assert main(values) == 0
    assert received["args"] == tuple(values[index] for index in (2, 4, 6, 8))
    assert received["kwargs"] == {
        "device": "cpu",
        "batch_size": 3,
        "shard_size": 7,
    }


def test_cli_reports_plain_pipeline_errors_without_a_traceback(
    capsys, tmp_path
):
    code = main(
        [
            "run",
            "--reference-manifest",
            str(tmp_path / "missing.csv"),
            "--evaluation-manifest",
            str(tmp_path / "missing-too.csv"),
            "--checkpoint",
            str(tmp_path / "missing.pth"),
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("option", "value"),
    (("--batch-size", "0"), ("--shard-size", "-1")),
)
def test_cli_rejects_nonpositive_runtime_sizes(option, value, capsys):
    with pytest.raises(SystemExit) as error:
        main(
            [
                "run",
                "--reference-manifest",
                "reference.csv",
                "--evaluation-manifest",
                "evaluation.csv",
                "--checkpoint",
                "model.pth",
                "--output-dir",
                "run",
                option,
                value,
            ]
        )

    assert error.value.code == 2
    assert "must be positive" in capsys.readouterr().err
