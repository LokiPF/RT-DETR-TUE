import json
import os

import numpy as np
import pytest
from PIL import Image

from degradation_monitor.method.statistics import KEYS
from degradation_monitor.runs import SCORE_KEYS, Manifest, sha1
from degradation_monitor.settings import Settings
from degradation_monitor.stages import run_stage
from scripts import convert_runs

WIDTHS = (2, 3, 4, 5)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _statistics(rows, seed):
    rng = np.random.default_rng(seed)
    return {key: rng.random((rows, WIDTHS[int(key[-1]) - 1])).astype(np.float32) for key in KEYS}


@pytest.fixture
def old(tmp_path):
    """A complete old-layout run folder for three evaluation images, and the clean branch's settings."""
    val = tmp_path / "val"
    val.mkdir()
    for index in range(3):
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(val / f"{index:04d}.jpg")
    (tmp_path / "ckpt.pth").write_bytes(b"weights")
    settings = Settings(run=tmp_path / "runs" / "coco", checkpoint=tmp_path / "ckpt.pth", train_images=tmp_path,
                        val_images=val, annotations=tmp_path / "ann.json", discopatch_root=tmp_path, device="cpu",
                        workers=0)
    folder = tmp_path / "runs" / "coco-baselines"
    protocol = settings.protocol()
    _write_json(folder / "run_config.json", {"checkpoint": str(settings.checkpoint), "val_images": str(val),
                                             **{field: protocol[field] for field in convert_runs.PROTOCOL_FIELDS}})
    _write_json(folder / "environment.json", {"packages": {"torch": "2.8"}, "discopatch_commit": "abc", "gpu": "RTX"})
    _write_json(folder / "sanity.json", {"coco_val_ap": 0.479, "images": 5000})
    for source in convert_runs.LINKED.values():
        (folder / source).parent.mkdir(parents=True, exist_ok=True)
        (folder / source).write_bytes(source.encode())  # each file's own name, so every sha1 differs
    for index, source in enumerate(convert_runs.REWRITTEN.values()):
        (folder / source).parent.mkdir(parents=True, exist_ok=True)
        np.savez(folder / source, **_statistics(4, index), grid_s1=np.ones((4, 32)), p99_s1=np.ones((4, 2)))
    for name, source in convert_runs.SCORE_FOLDERS.items():
        (folder / source).mkdir()
        for image in settings.dataset.evaluation_images():
            arrays = _statistics(96, 7) if name == "method" else {key: np.zeros(96) for key in SCORE_KEYS[name]}
            np.savez(folder / source / f"{image.stem}.npz", **arrays)
    _write_json(folder / "test_activation" / "fits.json", {
        "hashemi-fit": {"path": "x", "sha1": sha1(folder / convert_runs.LINKED["hashemi_intervals"])},
        "cdf-fit": {"path": "x", "sha1": sha1(folder / convert_runs.LINKED["cdf_reference"])},
        "cdf-zstats": {"path": "x", "sha1": sha1(folder / convert_runs.LINKED["cdf_zstats"])},
        "hashemi_k": 2.0, "cdf_bins": 1000})
    _write_json(folder / "test_dcp" / "checkpoint.json",
                {"checkpoint": "x", "sha1": sha1(folder / convert_runs.LINKED["discopatch_checkpoint"])})
    return folder, settings


def test_the_conversion_links_every_file_rewrites_the_method_reference_and_writes_the_manifest(old):
    folder, settings = old
    convert_runs.convert(folder, settings)
    layout = settings.layout
    for name, source in convert_runs.SCORE_FOLDERS.items():
        for path in sorted((folder / source).glob("*.npz")):
            assert os.path.samefile(path, layout.score_file(name, path))
    for attribute, source in convert_runs.LINKED.items():
        assert os.path.samefile(folder / source, getattr(layout, attribute))
    for attribute, source in convert_runs.REWRITTEN.items():
        with np.load(getattr(layout, attribute)) as new, np.load(folder / source) as before:
            assert set(new.files) == set(KEYS)
            assert all(np.array_equal(new[k], before[k]) and new[k].dtype == before[k].dtype for k in KEYS)
    manifest = Manifest(layout).read()
    assert manifest["protocol"] == json.loads(json.dumps(settings.protocol()))
    assert manifest["inputs"]["detector"] == {"knn_bank": sha1(layout.knn_bank)}
    assert manifest["check"]["coco_val_ap"] == 0.479 and manifest["environment"]["gpu"] == "RTX"
    assert manifest["conversion"]["linked"]["method"] == 3
    assert manifest["conversion"]["rewritten"]["method_bank"]["dropped"] == ["grid_s1", "p99_s1"]
    assert (settings.run / "logs").is_dir() and not (settings.run.parent / ".coco.converting").exists()


def test_the_new_stages_accept_the_converted_folder_as_complete(old):
    folder, settings = old
    convert_runs.convert(folder, settings)
    for stage in ("detector-pass", "discopatch-pass", "activation-pass", "method-pass"):
        run_stage(stage, settings)  # nothing is pending, and every recorded input matches its file
    assert len(list(settings.layout.scores("method").glob("*.npz"))) == 3


def test_conversion_refuses_an_existing_target(old):
    folder, settings = old
    settings.run.mkdir(parents=True)
    with pytest.raises(ValueError, match="already exists"):
        convert_runs.convert(folder, settings)


def test_conversion_refuses_an_incomplete_method_pass(old):
    folder, settings = old
    sorted((folder / "test_convtu_means").glob("*.npz"))[0].unlink()
    with pytest.raises(ValueError, match="test_convtu_means is incomplete: 1 of 3"):
        convert_runs.convert(folder, settings)
    assert not settings.run.exists() and not (settings.run.parent / ".coco.converting").exists()


def test_conversion_refuses_a_score_file_without_its_arrays(old):
    folder, settings = old
    victim = sorted((folder / "test_convtu_means").glob("*.npz"))[0]
    np.savez(victim, means_s1=np.zeros((96, 2)))
    with pytest.raises(ValueError, match=f"{victim.name} lacks"):
        convert_runs.convert(folder, settings)


def test_conversion_refuses_an_old_run_with_another_protocol(old):
    folder, settings = old
    config = json.loads((folder / "run_config.json").read_text())
    (folder / "run_config.json").write_text(json.dumps({**config, "seed": 45}))
    with pytest.raises(ValueError, match=r"another protocol \(seed\)"):
        convert_runs.convert(folder, settings)


def test_conversion_refuses_scores_computed_from_other_fits(old):
    folder, settings = old
    fits = json.loads((folder / "test_activation" / "fits.json").read_text())
    fits["cdf-fit"]["sha1"] = "0" * 40
    (folder / "test_activation" / "fits.json").write_text(json.dumps(fits))
    with pytest.raises(ValueError, match="other fits"):
        convert_runs.convert(folder, settings)
    assert not settings.run.exists()
