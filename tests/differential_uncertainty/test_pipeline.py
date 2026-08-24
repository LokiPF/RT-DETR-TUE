from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from differential_uncertainty.cli import run_pipeline
from differential_uncertainty.config import ExperimentConfig


class FakeExtractor:
    instances = 0
    calls = 0
    identities: list[tuple[str, int]] = []

    def __init__(self, _checkpoint, _device, config):
        type(self).instances += 1
        self.config = config

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def extract_batch(self, identities, samples):
        type(self).calls += 1
        type(self).identities.extend(identities)
        records = []
        for (image_id, severity), sample in zip(identities, samples):
            query = torch.arange(self.config.query_count, dtype=torch.float32)
            image_value = float(sample.mean())
            boxes = torch.stack(
                (query, query + 1, query + 2, query + 3), dim=1
            )
            logits = (
                query[:, None].repeat(1, self.config.class_count) / 10
                + image_value
                - severity / 20
            )
            persistence = torch.stack(
                [
                    query
                    + image_value
                    + severity
                    * (query / self.config.query_count) ** power
                    for power in range(1, self.config.persistence_dim + 1)
                ],
                dim=1,
            )
            for field in (boxes, logits, persistence):
                field[-2] = field[-1]
            records.append(
                {
                    "image_id": image_id,
                    "severity": severity,
                    "boxes": boxes,
                    "logits": logits,
                    "persistence": persistence,
                }
            )
        return records


class RejectingExtractor:
    def __init__(self, *_args, **_kwargs):
        raise AssertionError("completed stages must not construct an extractor")


class InterruptingExtractor(FakeExtractor):
    interrupted = False

    def extract_batch(self, identities, samples):
        if identities and identities[0] == ("e1", 2) and not type(self).interrupted:
            type(self).interrupted = True
            raise RuntimeError("simulated extraction interruption")
        return super().extract_batch(identities, samples)


@pytest.fixture
def small_config():
    return ExperimentConfig.for_tests(
        bank_capacity=20,
        k=2,
        query_count=20,
        persistence_dim=7,
        bootstrap_samples=20,
    )


@pytest.fixture(autouse=True)
def reset_fake_extractor():
    FakeExtractor.instances = 0
    FakeExtractor.calls = 0
    FakeExtractor.identities = []
    InterruptingExtractor.instances = 0
    InterruptingExtractor.calls = 0
    InterruptingExtractor.identities = []
    InterruptingExtractor.interrupted = False


def _manifest(root: Path, name: str, entries) -> Path:
    lines = ["image_id,image_path"]
    for image_id, color in entries:
        path = root / f"{image_id}.png"
        Image.new("RGB", (17, 13), (color, color, color)).save(path)
        lines.append(f"{image_id},{path.name}")
    manifest = root / name
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _inputs(tmp_path: Path):
    reference = _manifest(
        tmp_path, "reference.csv", (("r1", 10), ("r2", 30))
    )
    evaluation = _manifest(
        tmp_path,
        "evaluation.csv",
        (("e1", 60), ("e2", 90), ("e3", 120)),
    )
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fake checkpoint content")
    return reference, evaluation, checkpoint, tmp_path / "run"


def _run(inputs, config, *, extractor_factory=FakeExtractor):
    reference, evaluation, checkpoint, output = inputs
    return run_pipeline(
        reference,
        evaluation,
        checkpoint,
        output,
        device="cpu",
        batch_size=2,
        shard_size=2,
        config=config,
        extractor_factory=extractor_factory,
    )


def test_pipeline_builds_every_stage_and_second_run_constructs_no_extractor(
    tmp_path, small_config
):
    inputs = _inputs(tmp_path)

    result = _run(inputs, small_config)

    output = inputs[-1]
    assert result == output.resolve()
    assert FakeExtractor.instances == 1
    assert FakeExtractor.calls > 0
    expected = (
        output / "artifacts" / "provenance.json",
        output / "artifacts" / "reference-extractions" / "manifest.json",
        output / "artifacts" / "evaluation-extractions" / "manifest.json",
        output / "artifacts" / "reference-bank.pt",
        output / "artifacts" / "scores.csv",
        output / "report" / "report.md",
    )
    assert all(path.is_file() for path in expected)
    first_calls = FakeExtractor.calls

    _run(inputs, small_config, extractor_factory=RejectingExtractor)

    assert FakeExtractor.calls == first_calls


def test_changed_checkpoint_is_refused_before_any_stage_is_reused(
    tmp_path, small_config
):
    inputs = _inputs(tmp_path)
    _run(inputs, small_config)
    inputs[2].write_bytes(b"different checkpoint content")

    with pytest.raises(ValueError, match="checkpoint_sha256"):
        _run(inputs, small_config, extractor_factory=RejectingExtractor)


def test_interrupted_extraction_resumes_at_the_next_canonical_record(
    tmp_path, small_config
):
    inputs = _inputs(tmp_path)
    with pytest.raises(RuntimeError, match="simulated extraction interruption"):
        _run(inputs, small_config, extractor_factory=InterruptingExtractor)

    partial = inputs[-1] / "artifacts" / "evaluation-extractions"
    assert not (partial / "manifest.json").exists()
    assert any(partial.glob("shard_*.pt"))
    FakeExtractor.identities = []

    _run(inputs, small_config)

    assert (partial / "manifest.json").is_file()
    assert ("e1", 0) not in FakeExtractor.identities
    assert ("e1", 1) not in FakeExtractor.identities


def test_completed_extraction_is_validated_before_extractor_construction(
    tmp_path, small_config
):
    inputs = _inputs(tmp_path)
    _run(inputs, small_config)
    manifest = (
        inputs[-1]
        / "artifacts"
        / "evaluation-extractions"
        / "manifest.json"
    )
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["record_count"] += 1
    manifest.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises((ValueError, RuntimeError), match="record"):
        _run(inputs, small_config, extractor_factory=RejectingExtractor)


def test_corrupted_bank_is_refused_instead_of_rebuilt(tmp_path, small_config):
    inputs = _inputs(tmp_path)
    _run(inputs, small_config)
    bank = inputs[-1] / "artifacts" / "reference-bank.pt"
    bank.write_bytes(b"not a torch artifact")

    with pytest.raises(ValueError, match="reference bank"):
        _run(inputs, small_config, extractor_factory=RejectingExtractor)


def test_incomplete_score_roster_is_refused_instead_of_reused(
    tmp_path, small_config
):
    inputs = _inputs(tmp_path)
    _run(inputs, small_config)
    scores = inputs[-1] / "artifacts" / "scores.csv"
    frame = pd.read_csv(scores)
    frame.iloc[:-1].to_csv(scores, index=False)

    with pytest.raises(ValueError, match="score.*roster|severities 0 through 5"):
        _run(inputs, small_config, extractor_factory=RejectingExtractor)


def test_missing_report_is_rebuilt_without_extraction(tmp_path, small_config):
    inputs = _inputs(tmp_path)
    _run(inputs, small_config)
    shutil.rmtree(inputs[-1] / "report")

    _run(inputs, small_config, extractor_factory=RejectingExtractor)

    assert (inputs[-1] / "report" / "report.md").is_file()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"batch_size": 0}, "batch_size"),
        ({"batch_size": True}, "batch_size"),
        ({"shard_size": -1}, "shard_size"),
        ({"shard_size": False}, "shard_size"),
        ({"device": ""}, "device"),
    ],
)
def test_pipeline_rejects_invalid_runtime_controls(
    tmp_path, small_config, changes, message
):
    inputs = _inputs(tmp_path)
    arguments = {
        "device": "cpu",
        "batch_size": 2,
        "shard_size": 2,
        "config": small_config,
        "extractor_factory": RejectingExtractor,
    }
    arguments.update(changes)

    with pytest.raises(ValueError, match=message):
        run_pipeline(*inputs, **arguments)
