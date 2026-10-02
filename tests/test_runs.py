import json

import numpy as np
import pytest

from degradation_monitor.runs import SCORE_KEYS, Manifest, RunLayout, atomic_npz, load_npz, stack, valid_existing


def test_the_layout_names_every_reference_and_score_folder(tmp_path):
    layout = RunLayout(tmp_path)
    assert layout.knn_bank == tmp_path / "reference" / "knn" / "bank.npy"
    assert layout.method_bank == tmp_path / "reference" / "method" / "bank.npz"
    assert layout.discopatch_checkpoint == tmp_path / "reference" / "discopatch" / "discriminator.pt"
    assert layout.score_file("method", "val/000000000139.jpg") == tmp_path / "scores" / "method" / "000000000139.npz"
    assert layout.report() == tmp_path / "reports" / "coco"
    assert set(SCORE_KEYS) == {"detector", "activations", "discopatch", "method"}
    with pytest.raises(ValueError, match="unknown score folder"):
        layout.scores("test")


def test_result_files_are_written_atomically_and_damaged_ones_are_refused(tmp_path):
    path = tmp_path / "scores" / "detector" / "a.npz"
    atomic_npz(path, x=np.arange(3))
    assert load_npz(path, ("x",))["x"].tolist() == [0, 1, 2]
    assert not list(path.parent.glob(".*.tmp"))
    assert valid_existing(path, ("x",)) and not valid_existing(tmp_path / "missing.npz", ("x",))
    with pytest.raises(ValueError, match="lacks"):
        load_npz(path, ("x", "y"))
    path.write_bytes(b"not an npz")
    with pytest.raises(ValueError, match="malformed"):
        valid_existing(path, ("x",))


def test_stack_reads_every_named_image_and_names_a_missing_one(tmp_path):
    for name in ("a", "b"):
        atomic_npz(tmp_path / f"{name}.npz", v=np.full(2, ord(name)))
    assert stack(tmp_path, ["a.jpg", "b.jpg"], ("v",))["v"].shape == (2, 2)
    with pytest.raises(ValueError, match="1 of 3 missing, e.g. c.jpg"):
        stack(tmp_path, ["a.jpg", "b.jpg", "c.jpg"], ("v",))


def test_a_run_folder_refuses_another_protocol(tmp_path):
    manifest = Manifest(RunLayout(tmp_path))
    manifest.check_protocol({"seed": 44, "limit": None, "conditions": [("clean", 0), ("fog", 1)]})
    manifest.check_protocol({"seed": 44, "limit": None, "conditions": [["clean", 0], ["fog", 1]]})  # tuples == lists
    with pytest.raises(ValueError, match=r"another protocol \(limit, seed\)") as refused:
        manifest.check_protocol({"seed": 45, "limit": 10, "conditions": [["clean", 0], ["fog", 1]]})
    assert "--run DIR" in str(refused.value)  # the refusal names the way out


def test_a_run_folder_with_scores_but_no_recorded_protocol_is_refused(tmp_path):
    layout = RunLayout(tmp_path / "unconverted")
    atomic_npz(layout.score_file("detector", "a.jpg"), x=np.arange(3))
    with pytest.raises(ValueError, match="holds score files"):
        Manifest(layout).check_protocol({"seed": 999, "limit": 3})
    assert not layout.manifest.exists()
    fresh = RunLayout(tmp_path / "fresh")  # reference files and an empty score folder do not count
    atomic_npz(fresh.method_bank, x=np.arange(3))
    fresh.scores("detector").mkdir(parents=True)
    Manifest(fresh).check_protocol({"seed": 44, "limit": None})
    assert Manifest(fresh).read()["protocol"] == {"seed": 44, "limit": None}


def test_a_score_folder_refuses_inputs_other_than_its_own(tmp_path):
    manifest = Manifest(RunLayout(tmp_path))
    manifest.check_inputs("activations", {"cdf_reference": "aa", "hashemi_k": 2.0})
    manifest.check_inputs("activations", {"cdf_reference": "aa", "hashemi_k": 2.0})
    with pytest.raises(ValueError, match="inputs of scores/activations changed .*cdf_reference"):
        manifest.check_inputs("activations", {"cdf_reference": "bb", "hashemi_k": 2.0})
    assert json.loads((tmp_path / "manifest.json").read_text())["inputs"]["activations"]["cdf_reference"] == "aa"


def test_a_score_folder_with_files_but_no_recorded_inputs_is_refused(tmp_path):
    layout = RunLayout(tmp_path)
    manifest = Manifest(layout)
    atomic_npz(layout.score_file("activations", "a.jpg"), x=np.arange(3))
    with pytest.raises(ValueError, match="scores/activations already holds files"):
        manifest.check_inputs("activations", {"cdf_reference": "aa", "hashemi_k": 2.0})
    layout.scores("discopatch").mkdir(parents=True)
    manifest.check_inputs("discopatch", {"discriminator": "dd"})
    assert manifest.read()["inputs"] == {"discopatch": {"discriminator": "dd"}}


def test_the_environment_is_recorded_once(tmp_path):
    manifest = Manifest(RunLayout(tmp_path))
    manifest.record_environment(tmp_path)
    first = manifest.read()["environment"]
    assert "torch" in first["packages"] and first["discopatch_commit"] is None
    manifest.update(environment={"packages": {}, "discopatch_commit": "x", "gpu": None})
    manifest.record_environment(tmp_path)
    assert manifest.read()["environment"]["discopatch_commit"] == "x"
