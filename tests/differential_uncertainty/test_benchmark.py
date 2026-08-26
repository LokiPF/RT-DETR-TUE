from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os

import torch
import numpy as np
import pytest
from PIL import Image

import differential_uncertainty.benchmark as benchmark

from differential_uncertainty.benchmark import (
    create_coco_manifests,
    run_coco_benchmark,
)
from differential_uncertainty.config import ExperimentConfig
from differential_uncertainty.corruptions import GaussianBlur, Severity


class FakeExtractor:
    instances = 0
    identities = []

    def __init__(self, _checkpoint, _device, config):
        type(self).instances += 1
        self.config = config

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def extract_batch(self, identities, samples):
        type(self).identities.extend(identities)
        records = []
        for (image_id, severity), sample in zip(identities, samples):
            query = torch.arange(self.config.query_count, dtype=torch.float32)
            image_value = float(sample.mean())
            boxes = torch.stack((query, query + 1, query + 2, query + 3), dim=1)
            logits = (
                query[:, None].repeat(1, self.config.class_count) / 10
                + image_value
                - severity / 20
            )
            persistence = torch.stack(
                [
                    query + image_value + severity * (query / self.config.query_count) ** power
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


class ContrastCorruption:
    name = "contrast"
    severities = tuple(Severity(level, float(level)) for level in range(6))

    def apply(self, image, level):
        output = image.copy()
        if level:
            output.putpixel((0, 0), (level, level, level))
        return output


@pytest.fixture(autouse=True)
def _reset_fake_extractor():
    FakeExtractor.instances = 0
    FakeExtractor.identities = []

@pytest.fixture
def small_config():
    return ExperimentConfig.for_tests(
        bank_capacity=20,
        k=2,
        query_count=20,
        persistence_dim=7,
        bootstrap_samples=20,
    )




def test_coco_split_defaults_to_disjoint_250_250(tmp_path):
    images = tmp_path / "val2017"
    images.mkdir()
    records = []
    for image_id in range(1, 601):
        name = f"COCO_val2017_{image_id:012d}.jpg"
        Image.new("RGB", (8, 8), (image_id % 255, 0, 0)).save(
            images / name
        )
        records.append({"id": image_id, "file_name": name})
    annotations = tmp_path / "instances_val2017.json"
    annotations.write_text(json.dumps({"images": records}), encoding="utf-8")

    split = create_coco_manifests(annotations, images, tmp_path / "inputs")

    assert split.reference_count == split.evaluation_count == 250
    assert set(split.reference_ids).isdisjoint(split.evaluation_ids)
    assert split.reference_manifest.is_file()
    assert split.evaluation_manifest.is_file()
    assert split.benchmark_manifest.is_file()


def _coco_fixture(tmp_path, *, image_count: int):
    image_directory = tmp_path / "val2017"
    image_directory.mkdir()
    records = []
    for image_id in range(1, image_count + 1):
        name = f"COCO_val2017_{image_id:012d}.jpg"
        Image.new("RGB", (8, 8), (image_id % 255, 0, 0)).save(
            image_directory / name
        )
        records.append({"id": image_id, "file_name": name})
    annotations = tmp_path / "instances_val2017.json"
    annotations.write_text(json.dumps({"images": records}), encoding="utf-8")
    return annotations, image_directory


def _replace_records(path, records):
    path.write_text(json.dumps({"images": records}), encoding="utf-8")


def _manifest_ids(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return [row["image_id"] for row in csv.DictReader(handle)]


def test_coco_split_uses_the_seeded_permutation_in_canonical_id_order(tmp_path):
    annotations, images = _coco_fixture(tmp_path, image_count=6)

    split = create_coco_manifests(
        annotations,
        images,
        tmp_path / "inputs",
        reference_count=3,
        evaluation_count=2,
    )

    order = np.random.default_rng(20260825).permutation(np.arange(1, 7))
    assert split.reference_ids == tuple(sorted(int(item) for item in order[:3]))
    assert split.evaluation_ids == tuple(
        sorted(int(item) for item in order[3:5])
    )
    assert _manifest_ids(split.reference_manifest) == [
        str(item) for item in split.reference_ids
    ]


def test_coco_split_resume_requires_all_provenance_and_manifests_to_match(tmp_path):
    annotations, images = _coco_fixture(tmp_path, image_count=4)
    output = tmp_path / "inputs"
    first = create_coco_manifests(
        annotations, images, output, reference_count=2, evaluation_count=2
    )

    assert create_coco_manifests(
        annotations, images, output, reference_count=2, evaluation_count=2
    ) == first

    metadata = json.loads(
        first.benchmark_manifest.read_text(encoding="utf-8")
    )
    metadata["evaluation_ids"] = metadata["reference_ids"]
    first.benchmark_manifest.write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="reference and evaluation"):
        create_coco_manifests(
            annotations, images, output, reference_count=2, evaluation_count=2
        )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda annotations, _images: annotations.write_text(
            json.dumps({}), encoding="utf-8"
        ),
        lambda annotations, _images: _replace_records(
            annotations,
            [
                {"id": 1, "file_name": "COCO_val2017_000000000001.jpg"},
                {"id": 1, "file_name": "COCO_val2017_000000000002.jpg"},
            ],
        ),
        lambda annotations, _images: _replace_records(
            annotations,
            [
                {"id": 1, "file_name": "COCO_val2017_000000000001.jpg"},
                {"id": 2, "file_name": "COCO_val2017_000000000001.jpg"},
            ],
        ),
        lambda annotations, _images: _replace_records(
            annotations, [{"id": 1, "file_name": "../outside.jpg"}]
        ),
        lambda annotations, _images: _replace_records(
            annotations, [{"id": 1, "file_name": 3}]
        ),
        lambda _annotations, images: (
            images / "COCO_val2017_000000000001.jpg"
        ).unlink(),
    ),
)
def test_coco_split_rejects_invalid_annotation_records(tmp_path, mutate):
    annotations, images = _coco_fixture(tmp_path, image_count=2)
    mutate(annotations, images)

    with pytest.raises(ValueError):
        create_coco_manifests(
            annotations,
            images,
            tmp_path / "inputs",
            reference_count=1,
            evaluation_count=1,
        )


@pytest.mark.parametrize(
    ("reference_count", "evaluation_count"),
    ((0, 1), (1, 0), (-1, 1), (1, -1)),
)
def test_coco_split_rejects_nonpositive_requested_counts(
    tmp_path, reference_count, evaluation_count
):
    annotations, images = _coco_fixture(tmp_path, image_count=2)

    with pytest.raises(ValueError, match="positive integer"):
        create_coco_manifests(
            annotations,
            images,
            tmp_path / "inputs",
            reference_count=reference_count,
            evaluation_count=evaluation_count,
        )


def test_coco_split_rejects_insufficient_available_images(tmp_path):
    annotations, images = _coco_fixture(tmp_path, image_count=3)

    with pytest.raises(ValueError, match="not enough COCO images"):
        create_coco_manifests(
            annotations,
            images,
            tmp_path / "inputs",
            reference_count=2,
            evaluation_count=2,
        )


def test_coco_split_rejects_incomplete_existing_manifest_set(tmp_path):
    annotations, images = _coco_fixture(tmp_path, image_count=2)
    output = tmp_path / "inputs"
    output.mkdir()
    (output / "reference-manifest.csv").write_text(
        "image_id,image_path\n1,unexpected.jpg\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="incomplete existing COCO manifest set"):
        create_coco_manifests(
            annotations,
            images,
            output,
            reference_count=1,
            evaluation_count=1,
        )


def test_coco_split_recovers_exact_partial_manifests_after_interruption(
    tmp_path, monkeypatch
):
    annotations, images = _coco_fixture(tmp_path, image_count=2)
    output = tmp_path / "inputs"
    original_atomic_bytes = benchmark._atomic_bytes

    def interrupt_after_evaluation(value, path):
        original_atomic_bytes(value, path)
        if path.name == "evaluation-manifest.csv":
            raise RuntimeError("interrupted after evaluation manifest")

    monkeypatch.setattr(benchmark, "_atomic_bytes", interrupt_after_evaluation)
    with pytest.raises(RuntimeError, match="interrupted after evaluation"):
        create_coco_manifests(
            annotations,
            images,
            output,
            reference_count=1,
            evaluation_count=1,
        )

    assert (output / "reference-manifest.csv").is_file()
    assert (output / "evaluation-manifest.csv").is_file()
    assert (output / "benchmark-manifest-journal.json").is_file()
    assert not (output / "benchmark-manifest.json").exists()

    monkeypatch.setattr(benchmark, "_atomic_bytes", original_atomic_bytes)
    split = create_coco_manifests(
        annotations,
        images,
        output,
        reference_count=1,
        evaluation_count=1,
    )

    assert split.benchmark_manifest.is_file()
    assert not (output / "benchmark-manifest-journal.json").exists()


def _concurrent_manifest_creator(
    annotations, images, output, barrier, write_counts, write_lock, queue
):
    original_atomic_bytes = benchmark._atomic_bytes

    def count_atomic_bytes(value, path):
        with write_lock:
            process_id = os.getpid()
            write_counts[process_id] = write_counts.get(process_id, 0) + 1
        return original_atomic_bytes(value, path)

    benchmark._atomic_bytes = count_atomic_bytes
    try:
        barrier.wait(timeout=10)
        split = create_coco_manifests(
            annotations,
            images,
            output,
            reference_count=1,
            evaluation_count=1,
        )
    except BaseException as error:
        queue.put(("error", type(error).__name__, str(error)))
    else:
        queue.put(
            (
                "ok",
                split.reference_ids,
                split.evaluation_ids,
                str(split.benchmark_manifest),
            )
        )


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_concurrent_coco_manifest_creators_publish_one_coherent_set(tmp_path):
    annotations, images = _coco_fixture(tmp_path, image_count=2)
    output = tmp_path / "inputs"
    context = mp.get_context("fork")
    manager = context.Manager()
    write_counts = manager.dict()
    write_lock = context.Lock()
    barrier = context.Barrier(2)
    queue = context.Queue()
    creators = [
        context.Process(
            target=_concurrent_manifest_creator,
            args=(
                annotations,
                images,
                output,
                barrier,
                write_counts,
                write_lock,
                queue,
            ),
        )
        for _index in range(2)
    ]
    for creator in creators:
        creator.start()
    for creator in creators:
        creator.join(timeout=20)
        assert creator.exitcode == 0

    results = [queue.get(timeout=2) for _index in creators]
    assert [result[0] for result in results] == ["ok", "ok"]
    assert results[0][1:3] == results[1][1:3]
    assert sorted(write_counts.values()) == [4]
    manager.shutdown()
    split = create_coco_manifests(
        annotations,
        images,
        output,
        reference_count=1,
        evaluation_count=1,
    )
    assert split.benchmark_manifest.is_file()


@pytest.mark.parametrize("field", ("schema_version", "reference_count"))
def test_coco_split_rejects_boolean_metadata_fields(tmp_path, field):
    annotations, images = _coco_fixture(tmp_path, image_count=2)
    output = tmp_path / "inputs"
    split = create_coco_manifests(
        annotations,
        images,
        output,
        reference_count=1,
        evaluation_count=1,
    )
    metadata = json.loads(split.benchmark_manifest.read_text(encoding="utf-8"))
    metadata[field] = True
    split.benchmark_manifest.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="incompatible existing benchmark manifest"):
        create_coco_manifests(
            annotations,
            images,
            output,
            reference_count=1,
            evaluation_count=1,
        )


def test_run_coco_benchmark_shares_reference_and_writes_root_metrics(tmp_path, small_config, monkeypatch):
    annotations, images = _coco_fixture(tmp_path, image_count=4)
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "benchmark"
    corruptions = (GaussianBlur(), ContrastCorruption())
    monkeypatch.setattr(benchmark, "benchmark_corruptions", lambda: corruptions, raising=False)
    import differential_uncertainty.cli as cli

    original_prepare = cli.prepare_reference_stage
    prepare_calls = []

    def record_prepare(*args, **kwargs):
        prepare_calls.append((args, kwargs))
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(cli, "prepare_reference_stage", record_prepare)
    result = run_coco_benchmark(
        annotations, images, checkpoint, output,
        device="cpu", batch_size=1, shard_size=1,
        reference_count=2, evaluation_count=2,
        config=small_config, extractor_factory=FakeExtractor,
    )
    assert result == output.resolve()
    assert len(prepare_calls) == 1
    assert [path.name for path in (output / "corruptions").iterdir()] == ["gaussian_blur", "contrast"]
    assert len([item for item in FakeExtractor.identities if item[1] == 0]) == 6
    with (output / "benchmark-report" / "corruption-metrics.csv").open(newline="", encoding="utf-8") as handle:
        metrics = list(csv.DictReader(handle))
    assert [row["corruption"] for row in metrics] == ["gaussian_blur", "contrast"]
    assert list(metrics[0]) == ["corruption", "persistence_relative_gap_macro_auroc", "confidence_relative_gap_macro_auroc", "persistence_responsive_macro_auroc", "persistence_reference_macro_auroc", "direct_confidence_mean_macro_auroc", "direct_confidence_max_macro_auroc"]
    summary = json.loads((output / "benchmark-report" / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["method_aggregate"]) == {"persistence_relative_gap", "confidence_relative_gap", "persistence_responsive", "persistence_reference", "direct_confidence_mean", "direct_confidence_max"}
    assert sorted(item["primary_rank"] for item in summary["method_aggregate"].values()) == list(range(1, 7))
    for row in metrics:
        per_corruption = json.loads((output / "corruptions" / row["corruption"] / "report" / "summary.json").read_text(encoding="utf-8"))
        assert float(row["persistence_relative_gap_macro_auroc"]) == pytest.approx(per_corruption["evaluation"]["series"]["persistence_relative_gap"]["macro_auroc"])


def test_run_coco_benchmark_rejects_changed_roster(tmp_path, small_config, monkeypatch):
    annotations, images = _coco_fixture(tmp_path, image_count=4)
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "benchmark"
    monkeypatch.setattr(benchmark, "benchmark_corruptions", lambda: (GaussianBlur(), ContrastCorruption()), raising=False)
    run_coco_benchmark(
        annotations, images, checkpoint, output,
        device="cpu", batch_size=1, shard_size=1,
        reference_count=2, evaluation_count=2,
        config=small_config, extractor_factory=FakeExtractor,
    )
    monkeypatch.setattr(
        benchmark, "benchmark_corruptions", lambda: (GaussianBlur(),), raising=False
    )
    with pytest.raises(ValueError, match="corruption roster"):
        run_coco_benchmark(
            annotations, images, checkpoint, output,
            device="cpu", batch_size=1, shard_size=1,
            reference_count=2, evaluation_count=2,
            config=small_config, extractor_factory=FakeExtractor,
        )


def test_run_coco_benchmark_resumes_after_second_corruption_failure(
    tmp_path, small_config, monkeypatch
):
    annotations, images = _coco_fixture(tmp_path, image_count=4)
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "benchmark"
    monkeypatch.setattr(
        benchmark,
        "benchmark_corruptions",
        lambda: (GaussianBlur(), ContrastCorruption()),
        raising=False,
    )
    import differential_uncertainty.cli as cli

    original_stage = cli.run_corruption_stage
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(args[3].name)
        if args[3].name == "contrast":
            raise RuntimeError("second corruption failed")
        return original_stage(*args, **kwargs)

    monkeypatch.setattr(cli, "run_corruption_stage", fail_second)
    with pytest.raises(RuntimeError, match="second corruption failed"):
        run_coco_benchmark(
            annotations, images, checkpoint, output,
            device="cpu", batch_size=1, shard_size=1,
            reference_count=2, evaluation_count=2,
            config=small_config, extractor_factory=FakeExtractor,
        )
    instances_after_failure = FakeExtractor.instances
    identities_after_failure = list(FakeExtractor.identities)
    assert calls == ["gaussian_blur", "contrast"]
    assert (output / "corruptions" / "gaussian_blur" / "report").is_dir()

    monkeypatch.setattr(cli, "run_corruption_stage", original_stage)
    run_coco_benchmark(
        annotations, images, checkpoint, output,
        device="cpu", batch_size=1, shard_size=1,
        reference_count=2, evaluation_count=2,
        config=small_config, extractor_factory=FakeExtractor,
    )

    assert FakeExtractor.instances == instances_after_failure + 1
    assert len(FakeExtractor.identities) == len(identities_after_failure) + 12
    assert (output / "corruptions" / "contrast" / "report").is_dir()
