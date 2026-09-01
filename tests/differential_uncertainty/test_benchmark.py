from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

import differential_uncertainty.benchmark as benchmark
from differential_uncertainty.config import ExperimentConfig
from differential_uncertainty.scoring import normalize_bank


SMALL_CONFIG = ExperimentConfig(
    image_size=(8, 8), class_count=3, query_count=6, persistence_dim=4,
    bank_capacity=5, bank_chunk_size=8, bootstrap_samples=4,
)


class FakeExtractor:
    fail_at = None
    calls = 0
    nonfinite_at = None

    def __init__(self, _checkpoint, _device, config):
        self.config = config

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def extract_batch(self, images):
        type(self).calls += 1
        if type(self).calls == type(self).fail_at:
            raise RuntimeError("interrupted")
        records = []
        for image in images:
            value = float(image.mean())
            query = torch.arange(1, self.config.query_count + 1, dtype=torch.float32)
            persistence = torch.stack(
                (query, query.square(), query + value, query * 0.25 + value), dim=1
            )
            if type(self).calls == type(self).nonfinite_at:
                persistence[0, 0] = float("inf")
            records.append({
                "logits": torch.full(
                    (self.config.query_count, self.config.class_count), 2.0 + value
                ),
                "persistence": persistence,
                "padded_ids": torch.empty(0, dtype=torch.long),
            })
        return records


class SparseExtractor(FakeExtractor):
    def extract_batch(self, images):
        records = super().extract_batch(images)
        for record in records:
            record["logits"][1:] = -100.0
        return records


@pytest.fixture(autouse=True)
def _reset_extractor():
    FakeExtractor.calls = 0
    FakeExtractor.fail_at = None
    FakeExtractor.nonfinite_at = None


@pytest.fixture
def inputs(tmp_path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    train = tmp_path / "train2017"
    val = tmp_path / "val2017"
    train.mkdir()
    val.mkdir()
    for root, prefix, count in ((train, "train", 4), (val, "val", 3)):
        for index in reversed(range(count)):
            Image.new("RGB", (8, 8), (20 + index, index, 0)).save(
                root / f"{prefix}-{index}.jpg"
            )
    return checkpoint, train, val


def stochastic_corruption(image, name, severity):
    value = int(np.random.randint(0, 30)) + severity
    result = image.convert("RGB").copy()
    result.putpixel((0, 0), (value, len(name), severity))
    return result


def run(inputs, output, **overrides):
    checkpoint, train, val = inputs
    options = {
        "reference_count": 2, "evaluation_count": 2, "device": "cpu",
        "batch_size": 100, "seed": 7, "config": SMALL_CONFIG,
        "extractor_factory": FakeExtractor, "corruption_fn": stochastic_corruption,
    }
    options.update(overrides)
    return benchmark.run_coco_benchmark(checkpoint, train, val, output, **options)


def test_image_selection_is_sorted_then_seeded_separately(inputs):
    _, train, val = inputs
    selected_train = benchmark._select_images(train, 3, seed=11)
    selected_val = benchmark._select_images(val, 2, seed=11)
    train_expected = sorted(train.glob("*.jpg"))
    val_expected = sorted(val.glob("*.jpg"))
    np.random.default_rng(11).shuffle(train_expected)
    np.random.default_rng(11).shuffle(val_expected)
    assert selected_train == train_expected[:3]
    assert selected_val == val_expected[:2]


@pytest.mark.parametrize("missing", ("checkpoint", "train", "val"))
def test_missing_inputs_and_excess_counts_are_rejected(inputs, tmp_path, missing):
    checkpoint, train, val = inputs
    values = {"checkpoint": checkpoint, "train": train, "val": val}
    values[missing] = tmp_path / "missing"
    with pytest.raises(ValueError, match="does not exist"):
        benchmark.run_coco_benchmark(
            values["checkpoint"], values["train"], values["val"], tmp_path / "out",
            reference_count=1, evaluation_count=1, device="cpu",
            config=SMALL_CONFIG, extractor_factory=FakeExtractor,
            corruption_fn=stochastic_corruption,
        )
    with pytest.raises(ValueError, match="available"):
        run(inputs, tmp_path / "too-many", reference_count=99)


def test_run_config_mismatch_and_malformed_progress_are_rejected(inputs, tmp_path):
    output = tmp_path / "run"
    run(inputs, output)
    assert set(json.loads((output / "run_config.json").read_text())) == {
        "checkpoint", "coco_train_images", "coco_val_images",
        "reference_count", "evaluation_count", "device", "batch_size", "seed",
        "fixed_config", "corruptions",
    }
    with pytest.raises(ValueError, match="run_config"):
        run(inputs, output, seed=8)

    malformed = tmp_path / "malformed-bank"
    FakeExtractor.calls = 0
    FakeExtractor.fail_at = 2
    with pytest.raises(RuntimeError, match="interrupted"):
        run(inputs, malformed, batch_size=1)
    FakeExtractor.fail_at = None
    FakeExtractor.calls = 0
    torch.save({"wrong": True}, malformed / "bank_progress.pt")
    with pytest.raises(ValueError, match="bank progress"):
        run(inputs, malformed, batch_size=1)


def test_malformed_run_config_and_evaluation_progress_are_rejected(inputs, tmp_path):
    malformed_config = tmp_path / "malformed-config"
    malformed_config.mkdir()
    (malformed_config / "run_config.json").write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed run_config"):
        run(inputs, malformed_config)

    malformed_evaluation = tmp_path / "malformed-evaluation"
    run(inputs, malformed_evaluation)
    torch.save({"next_image": 1}, malformed_evaluation / "evaluation_progress.pt")
    with pytest.raises(ValueError, match="evaluation progress"):
        run(inputs, malformed_evaluation)


def test_too_few_bank_rows_and_nonfinite_extraction_are_rejected(inputs, tmp_path):
    with pytest.raises(ValueError, match="eligible bank"):
        run(inputs, tmp_path / "sparse", reference_count=1, extractor_factory=SparseExtractor)
    FakeExtractor.nonfinite_at = 1
    with pytest.raises(ValueError, match="finite"):
        run(inputs, tmp_path / "nonfinite", reference_count=1)


def test_interrupted_bank_resume_matches_uninterrupted(inputs, tmp_path):
    baseline = tmp_path / "baseline"
    run(inputs, baseline, batch_size=1)
    baseline_bank = torch.load(baseline / "bank.pt", weights_only=False)
    assert normalize_bank(baseline_bank).shape == (5, 4)
    assert not (baseline / "bank_progress.pt").exists()
    resumed = tmp_path / "resumed"
    FakeExtractor.calls = 0
    FakeExtractor.fail_at = 2
    with pytest.raises(RuntimeError, match="interrupted"):
        run(inputs, resumed, batch_size=1)
    assert (resumed / "bank_progress.pt").is_file()
    assert not (resumed / "bank.pt").exists()
    FakeExtractor.calls = 0
    FakeExtractor.fail_at = None
    run(inputs, resumed, batch_size=1)
    resumed_bank = torch.load(resumed / "bank.pt", weights_only=False)
    assert torch.equal(resumed_bank, baseline_bank)
    assert not (resumed / "bank_progress.pt").exists()

    torch.save({"stale": True}, resumed / "bank_progress.pt")
    run(inputs, resumed, batch_size=1)
    assert not (resumed / "bank_progress.pt").exists()


def test_stochastic_evaluation_resume_matches_score_files(inputs, tmp_path):
    baseline = tmp_path / "baseline"
    run(inputs, baseline)
    expected = {
        path.name: path.read_bytes() for path in sorted((baseline / "scores").glob("*.json"))
    }
    assert len(expected) == 2
    assert all(len(json.loads(value)) == 57 for value in expected.values())
    resumed = tmp_path / "resumed"
    FakeExtractor.calls = 0
    FakeExtractor.fail_at = 3
    with pytest.raises(RuntimeError, match="interrupted"):
        run(inputs, resumed)
    FakeExtractor.calls = 0
    FakeExtractor.fail_at = None
    run(inputs, resumed)
    actual = {
        path.name: path.read_bytes() for path in sorted((resumed / "scores").glob("*.json"))
    }
    assert actual == expected


def test_nonfinite_evaluation_extraction_is_rejected(inputs, tmp_path):
    FakeExtractor.nonfinite_at = 2
    with pytest.raises(ValueError, match="finite"):
        run(inputs, tmp_path / "nonfinite-score", reference_count=1)


def test_nonfinite_scores_are_rejected(inputs, tmp_path, monkeypatch):
    original = benchmark.score_triplet

    def nonfinite_score(*args, **kwargs):
        rows = original(*args, **kwargs)
        rows[0]["fingerprint"] = float("nan")
        return rows

    monkeypatch.setattr(benchmark, "score_triplet", nonfinite_score)
    with pytest.raises(ValueError, match="scores must be finite"):
        run(inputs, tmp_path / "nonfinite-score", reference_count=1)
