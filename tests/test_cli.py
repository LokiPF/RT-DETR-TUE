import runpy

import pytest

from degradation_monitor import cli, stages

CONFIG = """
run = "{root}/runs/coco"
checkpoint = "{root}/ckpt.pth"
train_images = "{root}/train"
val_images = "{root}/val"
annotations = "{root}/ann.json"
discopatch_root = "{root}/dcp"
batch_size = 8
"""


def _config(tmp_path):
    path = tmp_path / "coco.toml"
    path.write_text(CONFIG.format(root=tmp_path))
    return path


def _recorded(monkeypatch):
    calls = []
    monkeypatch.setattr(stages, "run_stage", lambda name, settings: calls.append((name, settings)))
    return calls


def test_every_stage_is_a_command():
    assert set(stages.STAGES) == {"check", "knn-bank", "detector-pass", "discopatch-train", "discopatch-pass",
                                  "hashemi-fit", "cdf-fit", "cdf-zstats", "activation-pass", "method-reference",
                                  "method-pass", "report", "timing"}
    parser = cli.build_parser()
    assert all(parser.parse_args([name]).stage == name for name in stages.STAGES)


def test_the_cli_forwards_the_config_and_its_overrides_to_the_stage(tmp_path, monkeypatch):
    calls = _recorded(monkeypatch)
    code = cli.main(["report", "--config", str(_config(tmp_path)), "--device", "cpu", "--limit", "10",
                     "--workers", "0", "--gpu-memory-gib", "1.5", "--run", str(tmp_path / "other")])
    assert code == 0
    (name, settings), = calls
    assert name == "report" and settings.run == tmp_path / "other"
    assert (settings.device, settings.limit, settings.workers, settings.gpu_memory_gib) == ("cpu", 10, 0, 1.5)
    assert settings.batch_size == 8 and settings.epochs == 65  # from the file and the default


@pytest.mark.parametrize(("option", "value"), (("--batch-size", "0"), ("--workers", "-1"), ("--gpu-memory-gib", "0")))
def test_the_cli_rejects_invalid_numbers(option, value, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["report", option, value])
    assert error.value.code == 2
    assert "must be" in capsys.readouterr().err


def test_the_cli_rejects_a_retired_phase(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["convtu-means"])
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_the_cli_prints_plain_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(stages, "run_stage", lambda *_args: (_ for _ in ()).throw(ValueError("bad input")))
    code = cli.main(["report", "--config", str(_config(tmp_path))])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert captured.err == "error: bad input\n"


def test_a_missing_config_is_a_plain_error(tmp_path, capsys):
    assert cli.main(["report", "--config", str(tmp_path / "missing.toml")]) == 2
    assert "missing.toml" in capsys.readouterr().err


def test_the_entry_module_does_not_run_the_cli_when_imported_by_a_spawned_worker():
    runpy.run_module("degradation_monitor.__main__", run_name="__mp_main__")
