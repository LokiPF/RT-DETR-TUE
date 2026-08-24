import csv
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


def test_manifest_rejects_a_row_with_an_extra_unquoted_field(tmp_path):
    _image(tmp_path / "x.png")
    path = tmp_path / "extra-field.csv"
    path.write_text(
        "image_id,image_path\na,x.png,ignored\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest row 2 is malformed"):
        load_manifest(path)


def test_manifest_rejects_a_row_with_a_missing_image_path(tmp_path):
    path = tmp_path / "missing-field.csv"
    path.write_text(
        "image_id,image_path\na\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest row 2 is malformed"):
        load_manifest(path)


@pytest.mark.parametrize(
    "image_id",
    (
        "=formula",
        "+formula",
        "-formula",
        "@formula",
        "  =formula",
        "\t+formula",
        " -formula",
        "  @formula",
    ),
)
def test_manifest_rejects_formula_like_image_ids_after_whitespace_normalization(
    tmp_path, image_id
):
    _image(tmp_path / "x.png")
    manifest = tmp_path / "unsafe.csv"
    manifest.write_text(
        f"image_id,image_path\n{image_id},x.png\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="spreadsheet formula character"):
        load_manifest(manifest)


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


@pytest.mark.parametrize(
    ("image_id", "message"),
    (
        ("line\nbreak", "control characters"),
        ("nul\x00byte", "control characters"),
        ("zero\u200bwidth", "control characters"),
        ("x" * 257, "at most 256"),
    ),
)
def test_manifest_rejects_control_and_overlong_image_ids(
    tmp_path, image_id, message
):
    _image(tmp_path / "x.png")
    manifest = tmp_path / "unsafe.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image_id", "image_path"))
        writer.writerow((image_id, "x.png"))

    with pytest.raises(ValueError, match=message):
        load_manifest(manifest)


def test_manifest_accepts_a_safe_256_character_unicode_image_id(tmp_path):
    image_id = "雪" * 127 + "\N{NO-BREAK SPACE}" + "雪" * 128
    _image(tmp_path / "x.png")
    manifest = tmp_path / "safe.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image_id", "image_path"))
        writer.writerow((image_id, "x.png"))

    entries = load_manifest(manifest)

    assert entries[0].image_id == image_id
