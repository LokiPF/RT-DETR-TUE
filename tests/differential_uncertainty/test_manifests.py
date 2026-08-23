import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from differential_uncertainty.manifests import (
    load_manifest,
    manifest_digest,
    validate_disjoint,
)


def _image(path: Path, color: int = 0) -> None:
    Image.new("RGB", (8, 6), (color, color, color)).save(path)


def test_manifest_paths_are_resolved_relative_to_the_csv_and_rows_are_canonicalized(
    tmp_path,
    monkeypatch,
):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _image(image_dir / "b.png", 20)
    _image(image_dir / "a.png", 10)
    manifest = tmp_path / "images.csv"
    manifest.write_text(
        "image_id,image_path\nb,images/b.png\na,images/../images/a.png\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(image_dir)

    entries = load_manifest(manifest)

    assert isinstance(entries, tuple)
    assert [entry.image_id for entry in entries] == ["a", "b"]
    assert entries[0].path == (image_dir / "a.png").resolve()
    assert entries[1].path == (image_dir / "b.png").resolve()


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("image_path,image_id\nx.png,a\n", "exactly image_id,image_path"),
        ("image_id,image_path,extra\na,x.png,value\n", "exactly image_id,image_path"),
        ("image_id,image_path\n", "at least one image"),
        ("image_id,image_path\n  ,x.png\n", "empty image_id"),
        ("image_id,image_path\na,missing.png\n", "does not exist"),
        ("image_id,image_path\na,x.png\na,y.png\n", "duplicate image_id"),
        ("image_id,image_path\na,x.png\nb,./x.png\n", "repeats resolved image"),
    ],
)
def test_manifest_errors_are_explicit(tmp_path, text, message):
    _image(tmp_path / "x.png")
    _image(tmp_path / "y.png")
    path = tmp_path / "bad.csv"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_manifest(path)


def test_reference_and_evaluation_cannot_repeat_an_image_id(tmp_path):
    _image(tmp_path / "reference.png")
    _image(tmp_path / "evaluation.png")
    reference = tmp_path / "reference.csv"
    evaluation = tmp_path / "evaluation.csv"
    reference.write_text(
        "image_id,image_path\nscene,reference.png\n",
        encoding="utf-8",
    )
    evaluation.write_text(
        "image_id,image_path\nscene,evaluation.png\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="repeat image_id"):
        validate_disjoint(load_manifest(reference), load_manifest(evaluation))


def test_reference_and_evaluation_cannot_repeat_a_resolved_path(tmp_path):
    _image(tmp_path / "same.png")
    reference = tmp_path / "reference.csv"
    evaluation = tmp_path / "evaluation.csv"
    reference.write_text(
        "image_id,image_path\nreference,same.png\n",
        encoding="utf-8",
    )
    evaluation.write_text(
        "image_id,image_path\nevaluation,./same.png\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="same resolved image"):
        validate_disjoint(load_manifest(reference), load_manifest(evaluation))


def test_disjoint_manifests_are_accepted(tmp_path):
    _image(tmp_path / "reference.png")
    _image(tmp_path / "evaluation.png")
    reference = tmp_path / "reference.csv"
    evaluation = tmp_path / "evaluation.csv"
    reference.write_text(
        "image_id,image_path\nreference,reference.png\n",
        encoding="utf-8",
    )
    evaluation.write_text(
        "image_id,image_path\nevaluation,evaluation.png\n",
        encoding="utf-8",
    )

    assert validate_disjoint(load_manifest(reference), load_manifest(evaluation)) is None


def test_manifest_digest_hashes_a_canonical_id_and_resolved_path_payload(tmp_path):
    _image(tmp_path / "a.png")
    _image(tmp_path / "b.png")
    manifest = tmp_path / "images.csv"
    manifest.write_text(
        "image_id,image_path\nb,b.png\na,a.png\n",
        encoding="utf-8",
    )
    entries = load_manifest(manifest)
    payload = [
        {"image_id": entry.image_id, "path": str(entry.path)}
        for entry in entries
    ]
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    assert manifest_digest(entries) == expected
    assert manifest_digest(tuple(reversed(entries))) == expected
