from __future__ import annotations

import csv
import gc
import hashlib
import json
import multiprocessing
import os
import shutil
import stat
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image, ImageOps

import differential_uncertainty.cli as cli
from differential_uncertainty.artifacts import iter_records, source_digest
from differential_uncertainty.cli import run_pipeline
from differential_uncertainty.config import ExperimentConfig
from differential_uncertainty.corruptions import Corruption, Severity
from differential_uncertainty.reporting import REPORT_FILES


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


class CoordinatedExtractor(FakeExtractor):
    gate = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            type(self).gate.wait(timeout=1)
        except threading.BrokenBarrierError:
            pass


class LifetimeExtractor(FakeExtractor):
    instance_ref = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        type(self).instance_ref = weakref.ref(self)


class InvertCorruption(Corruption):
    name = "invert"
    severities = tuple(Severity(level, float(level)) for level in range(6))

    def apply(self, image, level):
        if level not in range(6):
            raise ValueError(f"unknown invert severity {level}")
        return (
            image.copy()
            if level == 0
            else ImageOps.invert(image.convert("RGB"))
        )


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
    CoordinatedExtractor.gate = None
    LifetimeExtractor.instance_ref = None


def _manifest(root: Path, name: str, entries) -> Path:
    rows = []
    for index, (image_id, color) in enumerate(entries):
        path = root / f"{Path(name).stem}-{index}.png"
        Image.new("RGB", (17, 13), (color, color, color)).save(path)
        rows.append((image_id, path.name))
    manifest = root / name
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image_id", "image_path"))
        writer.writerows(rows)
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


def test_pipeline_accepts_a_corruption_plugin_without_changing_scoring(
    tmp_path, small_config
):
    reference = _manifest(
        tmp_path, "reference.csv", (("r1", 10), ("r2", 30))
    )
    evaluation = _manifest(
        tmp_path, "evaluation.csv", (("e1", 60), ("e2", 90))
    )
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fake checkpoint content")
    output = tmp_path / "invert-run"

    run_pipeline(
        reference,
        evaluation,
        checkpoint,
        output,
        device="cpu",
        batch_size=2,
        shard_size=2,
        config=small_config,
        extractor_factory=FakeExtractor,
        corruption=InvertCorruption(),
    )

    provenance = json.loads(
        (output / "artifacts" / "provenance.json").read_text()
    )
    assert provenance["corruption"]["name"] == "invert"
    assert [
        item["level"] for item in provenance["corruption"]["severities"]
    ] == list(range(6))
    with (output / "artifacts" / "scores.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert {int(row["severity"]) for row in rows} == set(range(6))


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


@pytest.mark.parametrize(
    "unsafe_id",
    ("line\nbreak", "nul\x00byte", "zero\u200bwidth", "x" * 257),
)
def test_score_artifact_reader_rejects_control_or_overlong_ids(
    tmp_path, unsafe_id
):
    path = tmp_path / "scores.csv"
    rows = []
    for severity in range(6):
        row = {field: 0.0 for field in cli._SCORE_COLUMNS}
        row.update(
            image_id=unsafe_id,
            severity=severity,
            padded_count=0,
            valid_count=0,
            reference_count=0,
            responsive_count=0,
        )
        rows.append(row)
    cli._atomic_score_csv(rows, path)

    with pytest.raises(ValueError, match="unsafe image_id"):
        cli._load_score_csv(path, (unsafe_id,))


def test_missing_report_is_rebuilt_without_extraction(tmp_path, small_config):
    inputs = _inputs(tmp_path)
    _run(inputs, small_config)
    shutil.rmtree(inputs[-1] / "report")

    _run(inputs, small_config, extractor_factory=RejectingExtractor)

    assert (inputs[-1] / "report" / "report.md").is_file()


@pytest.mark.parametrize("unsafe_id", ("=reference", "  @reference"))
def test_unsafe_reference_id_fails_before_output_or_detector_construction(
    tmp_path, small_config, unsafe_id
):
    reference = _manifest(
        tmp_path, "reference.csv", ((unsafe_id, 10), ("r2", 30))
    )
    evaluation = _manifest(tmp_path, "evaluation.csv", (("e1", 60),))
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fake checkpoint content")
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="spreadsheet formula character"):
        run_pipeline(
            reference,
            evaluation,
            checkpoint,
            output,
            device="cpu",
            batch_size=2,
            shard_size=2,
            config=small_config,
            extractor_factory=RejectingExtractor,
        )

    assert not output.exists()
    assert FakeExtractor.instances == 0


@pytest.mark.parametrize("unsafe_id", ("+evaluation", " -evaluation"))
def test_unsafe_evaluation_id_fails_before_output_or_detector_construction(
    tmp_path, small_config, unsafe_id
):
    reference = _manifest(
        tmp_path, "reference.csv", (("r1", 10), ("r2", 30))
    )
    evaluation = _manifest(tmp_path, "evaluation.csv", ((unsafe_id, 60),))
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fake checkpoint content")
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="spreadsheet formula character"):
        run_pipeline(
            reference,
            evaluation,
            checkpoint,
            output,
            device="cpu",
            batch_size=2,
            shard_size=2,
            config=small_config,
            extractor_factory=RejectingExtractor,
        )

    assert not output.exists()
    assert FakeExtractor.instances == 0


def test_safe_image_ids_round_trip_exactly_through_artifacts_scores_and_report(
    tmp_path, small_config
):
    reference_ids = ("ref=alpha", "ref+beta")
    evaluation_ids = ("scene-01@safe", "scene_02+safe")
    reference = _manifest(
        tmp_path,
        "reference.csv",
        tuple(zip(reference_ids, (10, 30))),
    )
    evaluation = _manifest(
        tmp_path,
        "evaluation.csv",
        tuple(zip(evaluation_ids, (60, 90))),
    )
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fake checkpoint content")
    output = tmp_path / "run"

    run_pipeline(
        reference,
        evaluation,
        checkpoint,
        output,
        device="cpu",
        batch_size=2,
        shard_size=2,
        config=small_config,
        extractor_factory=FakeExtractor,
    )

    reference_records = iter_records(
        output / "artifacts" / "reference-extractions"
    )
    assert {record["image_id"] for record in reference_records} == set(
        reference_ids
    )
    evaluation_records = list(
        iter_records(output / "artifacts" / "evaluation-extractions")
    )
    assert {record["image_id"] for record in evaluation_records} == set(
        evaluation_ids
    )
    artifact_scores = pd.read_csv(
        output / "artifacts" / "scores.csv", dtype={"image_id": str}
    )
    report_scores = pd.read_csv(
        output / "report" / "per-image-scores.csv",
        dtype={"image_id": str},
    )
    assert set(artifact_scores["image_id"]) == set(evaluation_ids)
    assert set(report_scores["image_id"]) == set(evaluation_ids)
    report_text = (output / "report" / "report.md").read_text(
        encoding="utf-8"
    )
    assert min(evaluation_ids) in report_text


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


_DETECTOR_SOURCE_PATHS = (
    "src/__init__.py",
    "src/core/__init__.py",
    "src/core/_config.py",
    "src/core/workspace.py",
    "src/core/yaml_config.py",
    "src/core/yaml_utils.py",
    "src/data/__init__.py",
    "src/data/_misc.py",
    "src/data/dataloader.py",
    "src/data/dataset/__init__.py",
    "src/data/dataset/_dataset.py",
    "src/data/dataset/cifar_dataset.py",
    "src/data/dataset/coco_dataset.py",
    "src/data/dataset/coco_eval.py",
    "src/data/dataset/coco_utils.py",
    "src/data/dataset/voc_detection.py",
    "src/data/dataset/voc_eval.py",
    "src/data/transforms/__init__.py",
    "src/data/transforms/_transforms.py",
    "src/data/transforms/container.py",
    "src/data/transforms/mosaic.py",
    "src/misc/__init__.py",
    "src/misc/box_ops.py",
    "src/misc/dist_utils.py",
    "src/misc/logger.py",
    "src/misc/profiler_utils.py",
    "src/misc/tue_utils.py",
    "src/misc/visualizer.py",
    "src/nn/__init__.py",
    "src/nn/arch/__init__.py",
    "src/nn/arch/classification.py",
    "src/nn/arch/yolo.py",
    "src/nn/backbone/__init__.py",
    "src/nn/backbone/common.py",
    "src/nn/backbone/csp_darknet.py",
    "src/nn/backbone/csp_resnet.py",
    "src/nn/backbone/hgnetv2.py",
    "src/nn/backbone/presnet.py",
    "src/nn/backbone/test_resnet.py",
    "src/nn/backbone/timm_model.py",
    "src/nn/backbone/torchvision_model.py",
    "src/nn/backbone/utils.py",
    "src/nn/criterion/__init__.py",
    "src/nn/criterion/det_criterion.py",
    "src/nn/postprocessor/__init__.py",
    "src/nn/postprocessor/nms_postprocessor.py",
    "src/optim/__init__.py",
    "src/optim/amp.py",
    "src/optim/ema.py",
    "src/optim/optim.py",
    "src/optim/warmup.py",
    "src/zoo/__init__.py",
    "src/zoo/rtdetr/__init__.py",
    "src/zoo/rtdetr/box_ops.py",
    "src/zoo/rtdetr/denoising.py",
    "src/zoo/rtdetr/hybrid_encoder.py",
    "src/zoo/rtdetr/matcher.py",
    "src/zoo/rtdetr/rtdetr.py",
    "src/zoo/rtdetr/rtdetr_criterion.py",
    "src/zoo/rtdetr/rtdetr_decoder.py",
    "src/zoo/rtdetr/rtdetr_postprocessor.py",
    "src/zoo/rtdetr/rtdetrv2_criterion.py",
    "src/zoo/rtdetr/rtdetrv2_decoder.py",
    "src/zoo/rtdetr/tue_rtdetr.py",
    "src/zoo/rtdetr/tue_rtdetrv2_decoder.py",
    "src/zoo/rtdetr/utils.py",
)


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _tree_state(root: Path) -> dict[str, tuple]:
    """Describe names, entry types, and file bytes without following links."""
    result = {}

    def visit(path: Path, relative: str) -> None:
        state = path.lstat()
        entry_type = stat.S_IFMT(state.st_mode)
        if stat.S_ISREG(state.st_mode):
            detail = hashlib.sha256(path.read_bytes()).hexdigest()
        elif stat.S_ISLNK(state.st_mode):
            detail = os.readlink(path)
        else:
            detail = None
        result[relative] = (entry_type, detail)
        if stat.S_ISDIR(state.st_mode):
            with os.scandir(path) as entries:
                children = sorted(entry.name for entry in entries)
            for name in children:
                child = path / name
                child_relative = f"{relative}/{name}" if relative else name
                visit(child, child_relative)

    visit(root, "")
    return result


def _directory_metadata(path: Path) -> tuple[int, ...]:
    """Capture directory identity and mutation-sensitive metadata."""
    state = os.stat(path, follow_symlinks=False)
    return (
        state.st_dev,
        state.st_ino,
        state.st_mode,
        state.st_size,
        state.st_mtime_ns,
        state.st_ctime_ns,
    )


def _process_run(inputs, config, gate, results):
    CoordinatedExtractor.gate = gate
    try:
        value = _run(inputs, config, extractor_factory=CoordinatedExtractor)
    except BaseException as error:
        results.put(("error", type(error).__name__, str(error)))
    else:
        results.put(("ok", str(value)))


def test_identical_fresh_thread_runs_coordinate_and_publish_exactly_once(
    tmp_path, small_config
):
    inputs = _inputs(tmp_path)
    CoordinatedExtractor.gate = threading.Barrier(2)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _run,
                inputs,
                small_config,
                extractor_factory=CoordinatedExtractor,
            )
            for _ in range(2)
        ]
        results = [future.result(timeout=30) for future in futures]

    assert results == [inputs[-1].resolve(), inputs[-1].resolve()]
    first = _tree_hashes(inputs[-1])
    _run(inputs, small_config, extractor_factory=RejectingExtractor)
    assert _tree_hashes(inputs[-1]) == first
    _assert_no_lock_artifact(inputs[-1])


def test_identical_fresh_process_runs_coordinate_and_publish_exactly_once(
    tmp_path, small_config
):
    context = multiprocessing.get_context("spawn")
    inputs = _inputs(tmp_path)
    gate = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(
            target=_process_run,
            args=(inputs, small_config, gate, results),
        )
        for _ in range(2)
    ]

    for process in processes:
        process.start()
    received = [results.get(timeout=30) for _ in processes]
    for process in processes:
        process.join(timeout=30)

    assert [process.exitcode for process in processes] == [0, 0]
    assert received == [
        ("ok", str(inputs[-1].resolve())),
        ("ok", str(inputs[-1].resolve())),
    ]
    first = _tree_hashes(inputs[-1])
    _run(inputs, small_config, extractor_factory=RejectingExtractor)
    assert _tree_hashes(inputs[-1]) == first
    _assert_no_lock_artifact(inputs[-1])


def test_detector_source_closure_matches_every_currently_executed_project_file():
    root = Path(cli.__file__).resolve().parents[1]
    actual = {
        path.relative_to(root).as_posix()
        for path in cli._source_files()
        if path.is_relative_to(root / "src")
    }

    assert actual == set(_DETECTOR_SOURCE_PATHS)
    assert "src/scene_uncertainty/pipeline.py" not in actual
    assert "src/solver/det_engine.py" not in actual


def test_each_detector_source_affects_digest_but_unrelated_source_does_not(
    tmp_path
):
    root = tmp_path / "clone"
    workflow = (
        "differential_uncertainty/__init__.py",
        "differential_uncertainty/cli.py",
    )
    for relative in (*workflow, *_DETECTOR_SOURCE_PATHS):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relative}\n", encoding="utf-8")
    unrelated = root / "src/scene_uncertainty/unrelated.py"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("first\n", encoding="utf-8")

    files = cli._source_files(root=root)
    baseline = source_digest(files, root=root)
    for relative in _DETECTOR_SOURCE_PATHS:
        path = root / relative
        original = path.read_text(encoding="utf-8")
        path.write_text(original + "changed\n", encoding="utf-8")
        assert source_digest(cli._source_files(root=root), root=root) != baseline
        path.write_text(original, encoding="utf-8")

    unrelated.write_text("second\n", encoding="utf-8")
    assert source_digest(cli._source_files(root=root), root=root) == baseline


def test_evaluation_groups_stream_only_one_six_record_image_at_a_time():
    image_ids = tuple(f"image-{index:03d}" for index in range(250))
    yielded = 0

    def records():
        nonlocal yielded
        for image_id in image_ids:
            for severity in range(6):
                yielded += 1
                yield {"image_id": image_id, "severity": severity}

    groups = cli._iter_evaluation_groups(records(), image_ids)
    first_id, first_records = next(groups)

    assert first_id == image_ids[0]
    assert [record["severity"] for record in first_records] == list(range(6))
    assert yielded == 6
    second_id, _second_records = next(groups)
    assert second_id == image_ids[1]
    assert yielded == 12
    assert sum(1 for _ in groups) == 248
    assert yielded == 1_500


@pytest.mark.parametrize(
    "records",
    (
        (
            {"image_id": "a", "severity": 0},
            {"image_id": "a", "severity": 2},
        ),
        (
            *({"image_id": "a", "severity": severity} for severity in range(6)),
            {"image_id": "extra", "severity": 0},
        ),
    ),
)
def test_evaluation_group_stream_rejects_noncanonical_or_extra_records(records):
    with pytest.raises(RuntimeError, match="canonical evaluation record roster"):
        list(cli._iter_evaluation_groups(records, ("a",)))


def test_extractor_is_released_before_reference_bank_building(
    tmp_path, small_config, monkeypatch
):
    inputs = _inputs(tmp_path)
    original = cli.build_reference_bank

    def observed_build(records, config):
        gc.collect()
        assert LifetimeExtractor.instance_ref() is None
        return original(records, config)

    monkeypatch.setattr(cli, "build_reference_bank", observed_build)

    _run(inputs, small_config, extractor_factory=LifetimeExtractor)


@pytest.mark.parametrize(
    ("location", "unsafe_id"),
    (
        ("reference", "line\nbreak"),
        ("reference", "nul\x00byte"),
        ("reference", "x" * 257),
        ("evaluation", "line\nbreak"),
        ("evaluation", "nul\x00byte"),
        ("evaluation", "x" * 257),
    ),
)
def test_control_and_overlong_ids_fail_before_output_or_detector(
    tmp_path, small_config, location, unsafe_id
):
    reference_entries = (
        ((unsafe_id, 10), ("r2", 30))
        if location == "reference"
        else (("r1", 10), ("r2", 30))
    )
    evaluation_entries = (
        ((unsafe_id, 60),)
        if location == "evaluation"
        else (("e1", 60),)
    )
    reference = _manifest(tmp_path, "reference.csv", reference_entries)
    evaluation = _manifest(tmp_path, "evaluation.csv", evaluation_entries)
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fake checkpoint content")
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="control characters|at most 256"):
        run_pipeline(
            reference,
            evaluation,
            checkpoint,
            output,
            device="cpu",
            batch_size=2,
            shard_size=2,
            config=small_config,
            extractor_factory=RejectingExtractor,
        )

    assert not output.exists()


def test_max_length_safe_unicode_ids_round_trip_end_to_end(
    tmp_path, small_config
):
    reference_ids = ("R" * 256, "ref-safe")
    evaluation_id = "雪" * 127 + "\N{NO-BREAK SPACE}" + "雪" * 128
    reference = _manifest(
        tmp_path,
        "reference.csv",
        tuple(zip(reference_ids, (10, 30))),
    )
    evaluation = _manifest(
        tmp_path,
        "evaluation.csv",
        ((evaluation_id, 60),),
    )
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fake checkpoint content")
    output = tmp_path / "run"

    run_pipeline(
        reference,
        evaluation,
        checkpoint,
        output,
        device="cpu",
        batch_size=2,
        shard_size=2,
        config=small_config,
        extractor_factory=FakeExtractor,
    )

    reference_records = list(
        iter_records(output / "artifacts" / "reference-extractions")
    )
    evaluation_records = list(
        iter_records(output / "artifacts" / "evaluation-extractions")
    )
    assert {record["image_id"] for record in reference_records} == set(
        reference_ids
    )
    assert {record["image_id"] for record in evaluation_records} == {
        evaluation_id
    }
    assert set(
        pd.read_csv(
            output / "artifacts" / "scores.csv", dtype={"image_id": str}
        )["image_id"]
    ) == {evaluation_id}
    assert set(
        pd.read_csv(
            output / "report" / "per-image-scores.csv",
            dtype={"image_id": str},
        )["image_id"]
    ) == {evaluation_id}
    assert evaluation_id in (
        output / "report" / "report.md"
    ).read_text(encoding="utf-8")


def _future_outcome(future):
    try:
        return future.result(timeout=30)
    except BaseException as error:
        return error


def _assert_no_lock_artifact(output: Path) -> None:
    assert all("lock" not in path.name.casefold() for path in output.rglob("*"))


def test_replaced_identical_provenance_cannot_bypass_run_coordination(
    tmp_path, small_config, monkeypatch
):
    inputs = _inputs(tmp_path)
    original = cli._run_pipeline_stages
    first_entered = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    state_lock = threading.Lock()
    state = {"calls": 0, "active": 0, "maximum": 0}

    def observed(*args, **kwargs):
        with state_lock:
            state["calls"] += 1
            call = state["calls"]
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
        try:
            if call == 1:
                first_entered.set()
                assert release_first.wait(timeout=10)
            else:
                second_entered.set()
            return original(*args, **kwargs)
        finally:
            with state_lock:
                state["active"] -= 1

    monkeypatch.setattr(cli, "_run_pipeline_stages", observed)
    executor = ThreadPoolExecutor(max_workers=2)
    try:
        first = executor.submit(_run, inputs, small_config)
        assert first_entered.wait(timeout=10)
        provenance = inputs[-1] / "artifacts" / "provenance.json"
        payload = provenance.read_bytes()
        os.replace(provenance, tmp_path / "original-provenance.json")
        provenance.write_bytes(payload)
        second = executor.submit(_run, inputs, small_config)
        entered_concurrently = second_entered.wait(timeout=0.5)
        release_first.set()
        outcomes = (_future_outcome(first), _future_outcome(second))
    finally:
        release_first.set()
        executor.shutdown(wait=True)

    assert not entered_concurrently
    assert outcomes == (inputs[-1].absolute(), inputs[-1].absolute())
    assert state == {"calls": 2, "active": 0, "maximum": 1}
    _assert_no_lock_artifact(inputs[-1])


def test_replaced_run_directory_never_redirects_or_overlaps_stage_writes(
    tmp_path, small_config, monkeypatch
):
    inputs = _inputs(tmp_path)
    original = cli._run_pipeline_stages
    first_entered = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    state_lock = threading.Lock()
    state = {"calls": 0, "active": 0, "maximum": 0}
    identities = {}

    def observed(*args, **kwargs):
        output = args[3]
        with state_lock:
            state["calls"] += 1
            call = state["calls"]
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
        before = os.stat(output, follow_symlinks=False)
        try:
            if call == 1:
                first_entered.set()
                assert release_first.wait(timeout=10)
            else:
                second_entered.set()
            return original(*args, **kwargs)
        finally:
            after = os.stat(output, follow_symlinks=False)
            identities[call] = (
                (before.st_dev, before.st_ino),
                (after.st_dev, after.st_ino),
            )
            with state_lock:
                state["active"] -= 1

    monkeypatch.setattr(cli, "_run_pipeline_stages", observed)
    executor = ThreadPoolExecutor(max_workers=2)
    displaced = tmp_path / "displaced-run"
    try:
        first = executor.submit(_run, inputs, small_config)
        assert first_entered.wait(timeout=10)
        provenance = inputs[-1] / "artifacts" / "provenance.json"
        payload = provenance.read_bytes()
        os.replace(inputs[-1], displaced)
        (inputs[-1] / "artifacts").mkdir(parents=True)
        (inputs[-1] / "artifacts" / "provenance.json").write_bytes(payload)
        second = executor.submit(_run, inputs, small_config)
        entered_concurrently = second_entered.wait(timeout=0.5)
        release_first.set()
        outcomes = (_future_outcome(first), _future_outcome(second))
    finally:
        release_first.set()
        executor.shutdown(wait=True)

    assert not entered_concurrently
    assert state == {"calls": 2, "active": 0, "maximum": 1}
    assert identities[1][0] == identities[1][1]
    assert isinstance(outcomes[0], ValueError)
    assert "output directory" in str(outcomes[0])
    assert outcomes[1] == inputs[-1].absolute()
    assert all(
        "superseded" not in str(outcome)
        and "active writer" not in str(outcome)
        for outcome in outcomes
    )
    _assert_no_lock_artifact(inputs[-1])
    _assert_no_lock_artifact(displaced)


@pytest.mark.parametrize(
    "raised",
    (RuntimeError("stage failed"), KeyboardInterrupt("stage interrupted")),
)
def test_stage_failure_releases_run_locks_and_directory_descriptors(
    tmp_path, small_config, monkeypatch, raised
):
    inputs = _inputs(tmp_path)
    original = cli._run_pipeline_stages
    before = {
        int(name)
        for name in os.listdir("/proc/self/fd")
        if name.isdigit()
        and Path(f"/proc/self/fd/{name}").exists()
    }

    def fail(*_args, **_kwargs):
        raise raised

    monkeypatch.setattr(cli, "_run_pipeline_stages", fail)
    with pytest.raises(type(raised), match=str(raised)):
        _run(inputs, small_config)
    monkeypatch.setattr(cli, "_run_pipeline_stages", original)
    after = {
        int(name)
        for name in os.listdir("/proc/self/fd")
        if name.isdigit()
        and Path(f"/proc/self/fd/{name}").exists()
    }

    assert after == before
    assert _run(inputs, small_config) == inputs[-1].absolute()
    _assert_no_lock_artifact(inputs[-1])


def test_static_symlink_output_is_rejected_before_stage_work(
    tmp_path, small_config
):
    inputs = _inputs(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    inputs[-1].symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="output directory"):
        _run(inputs, small_config, extractor_factory=RejectingExtractor)

    assert not any(target.iterdir())


def test_static_symlink_output_parent_is_rejected_before_stage_work(
    tmp_path, small_config
):
    inputs = _inputs(tmp_path)
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    inputs = (*inputs[:-1], linked_parent / "run")

    with pytest.raises(ValueError, match="output parent directory"):
        _run(inputs, small_config, extractor_factory=RejectingExtractor)

    assert not any(real_parent.iterdir())


@pytest.mark.parametrize("unsafe_kind", ("file", "fifo"))
def test_static_nondirectory_output_is_rejected_before_stage_work(
    tmp_path, small_config, unsafe_kind
):
    inputs = _inputs(tmp_path)
    if unsafe_kind == "file":
        inputs[-1].write_bytes(b"not a directory")
    else:
        os.mkfifo(inputs[-1])

    with pytest.raises(ValueError, match="output directory"):
        _run(inputs, small_config, extractor_factory=RejectingExtractor)


@pytest.mark.parametrize("unsafe_kind", ("symlink", "fifo"))
def test_static_unsafe_provenance_is_rejected_before_stage_work(
    tmp_path, small_config, unsafe_kind
):
    inputs = _inputs(tmp_path)
    artifacts = inputs[-1] / "artifacts"
    artifacts.mkdir(parents=True)
    provenance = artifacts / "provenance.json"
    if unsafe_kind == "symlink":
        target = tmp_path / "outside-provenance.json"
        target.write_text("{}", encoding="utf-8")
        provenance.symlink_to(target)
    else:
        os.mkfifo(provenance)

    with pytest.raises(ValueError, match="provenance.*regular file"):
        _run(inputs, small_config, extractor_factory=RejectingExtractor)


@pytest.mark.parametrize(
    ("location", "entry_name"),
    (
        ("artifacts", "artifacts"),
        ("reference cache", "reference-extractions"),
        ("evaluation cache", "evaluation-extractions"),
    ),
)
@pytest.mark.parametrize("unsafe_kind", ("symlink", "fifo", "file"))
def test_static_unsafe_nested_directory_fails_before_writes_or_detector(
    tmp_path, small_config, location, entry_name, unsafe_kind
):
    inputs = _inputs(tmp_path)
    output = inputs[-1]
    output.mkdir()
    if location == "artifacts":
        parent = output
    else:
        parent = output / "artifacts"
        parent.mkdir()
    entry = parent / entry_name
    outside = tmp_path / f"outside-{entry_name}"
    if unsafe_kind == "symlink":
        outside.mkdir()
        entry.symlink_to(outside, target_is_directory=True)
    elif unsafe_kind == "fifo":
        os.mkfifo(entry)
    else:
        entry.write_bytes(b"not a directory")
    constructed = 0

    def reject_detector(*_args, **_kwargs):
        nonlocal constructed
        constructed += 1
        raise AssertionError("unsafe nested output reached detector construction")

    with pytest.raises(ValueError, match=f"{location}.*directory"):
        _run(inputs, small_config, extractor_factory=reject_detector)

    assert constructed == 0
    if unsafe_kind == "symlink":
        assert not any(outside.iterdir())


@pytest.mark.parametrize(
    ("location", "entry_name"),
    (
        ("artifacts", "artifacts"),
        ("reference cache", "reference-extractions"),
        ("evaluation cache", "evaluation-extractions"),
    ),
)
def test_runtime_nested_directory_replacement_stays_anchored_and_fails_closed(
    tmp_path, small_config, monkeypatch, location, entry_name
):
    inputs = _inputs(tmp_path)
    artifacts = inputs[-1] / "artifacts"
    reference_cache = artifacts / "reference-extractions"
    evaluation_cache = artifacts / "evaluation-extractions"
    reference_cache.mkdir(parents=True)
    evaluation_cache.mkdir()
    original = cli._run_pipeline_stages
    entered = threading.Event()
    release = threading.Event()
    observed_identity = []

    def observed(*args, **kwargs):
        if "artifacts" in kwargs:
            paths = {
                "artifacts": kwargs["artifacts"],
                "reference-extractions": kwargs["reference_cache"],
                "evaluation-extractions": kwargs["evaluation_cache"],
            }
        else:
            ordinary_artifacts = args[3] / "artifacts"
            paths = {
                "artifacts": ordinary_artifacts,
                "reference-extractions": ordinary_artifacts
                / "reference-extractions",
                "evaluation-extractions": ordinary_artifacts
                / "evaluation-extractions",
            }
        stage_path = paths[entry_name]
        before = os.stat(stage_path, follow_symlinks=False)
        entered.set()
        assert release.wait(timeout=10)
        try:
            return original(*args, **kwargs)
        finally:
            after = os.stat(stage_path, follow_symlinks=False)
            observed_identity.append(
                (
                    (before.st_dev, before.st_ino),
                    (after.st_dev, after.st_ino),
                )
            )

    monkeypatch.setattr(cli, "_run_pipeline_stages", observed)
    visible = inputs[-1] / entry_name if location == "artifacts" else artifacts / entry_name
    displaced = tmp_path / f"displaced-{entry_name}"
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_run, inputs, small_config)
        assert entered.wait(timeout=10)
        os.replace(visible, displaced)
        visible.mkdir()
        release.set()
        outcome = _future_outcome(future)
    finally:
        release.set()
        executor.shutdown(wait=True)

    assert isinstance(outcome, ValueError)
    assert "directory" in str(outcome)
    assert observed_identity[0][0] == observed_identity[0][1]
    assert not any(visible.iterdir())
    assert any(displaced.rglob("*"))


def test_mismatched_provenance_refusal_does_not_change_the_existing_tree(
    tmp_path, small_config
):
    inputs = _inputs(tmp_path)
    _run(inputs, small_config)
    output = inputs[-1]
    artifacts = output / "artifacts"
    shutil.rmtree(output / "report")
    shutil.rmtree(artifacts / "reference-extractions")
    shutil.rmtree(artifacts / "evaluation-extractions")
    (artifacts / "reference-bank.pt").unlink()
    (artifacts / "scores.csv").unlink()
    provenance = artifacts / "provenance.json"
    value = json.loads(provenance.read_text(encoding="utf-8"))
    value["checkpoint_sha256"] = "0" * 64
    provenance.write_text(json.dumps(value), encoding="utf-8")
    os.utime(artifacts, ns=(1_000_000_000, 1_000_000_000))
    before_metadata = _directory_metadata(artifacts)
    before = _tree_state(output)

    with pytest.raises(ValueError, match="checkpoint_sha256"):
        _run(inputs, small_config, extractor_factory=RejectingExtractor)

    assert _tree_state(output) == before
    assert _directory_metadata(artifacts) == before_metadata


@pytest.mark.parametrize("unsafe_kind", ("symlink", "fifo", "directory"))
def test_unsafe_provenance_refusal_does_not_change_the_existing_tree(
    tmp_path, small_config, unsafe_kind
):
    inputs = _inputs(tmp_path)
    output = inputs[-1]
    artifacts = output / "artifacts"
    artifacts.mkdir(parents=True)
    provenance = artifacts / "provenance.json"
    if unsafe_kind == "symlink":
        outside = tmp_path / "outside-provenance.json"
        outside.write_text("{}", encoding="utf-8")
        provenance.symlink_to(outside)
    elif unsafe_kind == "fifo":
        os.mkfifo(provenance)
    else:
        provenance.mkdir()
    os.utime(artifacts, ns=(1_000_000_000, 1_000_000_000))
    before_metadata = _directory_metadata(artifacts)
    before = _tree_state(output)

    with pytest.raises(ValueError, match="provenance.*regular file"):
        _run(inputs, small_config, extractor_factory=RejectingExtractor)

    assert _tree_state(output) == before
    assert _directory_metadata(artifacts) == before_metadata


@pytest.mark.parametrize("location", ("early", "middle", "final"))
@pytest.mark.parametrize("unsafe_kind", ("symlink", "fifo", "file"))
def test_every_output_ancestor_is_traversed_without_following_unsafe_entries(
    tmp_path, small_config, location, unsafe_kind
):
    inputs = _inputs(tmp_path)
    base = tmp_path / "output-base"
    base.mkdir()
    names = ("early", "middle", "final")
    unsafe_index = names.index(location)
    parent = base
    for name in names[:unsafe_index]:
        parent = parent / name
        parent.mkdir()
    unsafe = parent / names[unsafe_index]
    outside = tmp_path / f"outside-{location}-{unsafe_kind}"
    if unsafe_kind == "symlink":
        outside.mkdir()
        unsafe.symlink_to(outside, target_is_directory=True)
    elif unsafe_kind == "fifo":
        os.mkfifo(unsafe)
    else:
        unsafe.write_bytes(b"not a directory")
    output = base.joinpath(*names, "run")
    inputs = (*inputs[:-1], output)
    constructed = 0

    def reject_detector(*_args, **_kwargs):
        nonlocal constructed
        constructed += 1
        raise AssertionError("unsafe output ancestor reached detector")

    with pytest.raises(ValueError, match="output.*directory"):
        _run(inputs, small_config, extractor_factory=reject_detector)

    assert constructed == 0
    if unsafe_kind == "symlink":
        assert not any(outside.iterdir())


def test_nested_output_components_are_created_normally(tmp_path, small_config):
    inputs = _inputs(tmp_path)
    output = tmp_path / "new" / "nested" / "parent" / "run"
    inputs = (*inputs[:-1], output)

    assert _run(inputs, small_config) == output.absolute()
    assert (output / "report/report.md").is_file()


def test_nested_output_creation_accepts_a_safe_concurrent_creator(
    tmp_path, small_config, monkeypatch
):
    inputs = _inputs(tmp_path)
    output = tmp_path / "new" / "raced" / "parent" / "run"
    inputs = (*inputs[:-1], output)
    original = cli.os.mkdir
    raced = False

    def concurrent_mkdir(path, mode=0o777, *, dir_fd=None):
        nonlocal raced
        if path == "raced" and not raced:
            raced = True
            original(path, mode, dir_fd=dir_fd)
            raise FileExistsError(path)
        return original(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(cli.os, "mkdir", concurrent_mkdir)

    assert _run(inputs, small_config) == output.absolute()
    assert raced
    assert (output / "report/report.md").is_file()


def test_output_ancestor_traversal_releases_descriptors_on_keyboard_interrupt(
    tmp_path, small_config, monkeypatch
):
    inputs = _inputs(tmp_path)
    output = tmp_path / "new" / "interrupt-here" / "parent" / "run"
    inputs = (*inputs[:-1], output)
    original = cli._pinned_child_directory
    before = {
        int(name)
        for name in os.listdir("/proc/self/fd")
        if name.isdigit() and Path(f"/proc/self/fd/{name}").exists()
    }

    def interrupt(parent, name, *, label):
        if (
            label == "output parent directory path component"
            and name == "interrupt-here"
        ):
            raise KeyboardInterrupt("ancestor traversal interrupted")
        return original(parent, name, label=label)

    monkeypatch.setattr(cli, "_pinned_child_directory", interrupt)
    with pytest.raises(KeyboardInterrupt, match="ancestor traversal interrupted"):
        _run(inputs, small_config, extractor_factory=RejectingExtractor)
    after = {
        int(name)
        for name in os.listdir("/proc/self/fd")
        if name.isdigit() and Path(f"/proc/self/fd/{name}").exists()
    }

    assert after == before


@pytest.mark.parametrize(
    "leaf",
    (
        "reference-manifest",
        "reference-shard",
        "evaluation-manifest",
        "evaluation-shard",
        "bank",
        "scores",
        "provenance",
        "report",
    ),
)
def test_leaf_mutation_after_last_normal_use_fails_the_final_audit(
    tmp_path, small_config, monkeypatch, leaf
):
    inputs = _inputs(tmp_path)
    _run(inputs, small_config)
    output = inputs[-1]
    original = cli._run_pipeline_stages

    def mutate_after_stages(*args, **kwargs):
        result = original(*args, **kwargs)
        paths = {
            "reference-manifest": output
            / "artifacts/reference-extractions/manifest.json",
            "reference-shard": sorted(
                (output / "artifacts/reference-extractions").glob("shard_*.pt")
            )[0],
            "evaluation-manifest": output
            / "artifacts/evaluation-extractions/manifest.json",
            "evaluation-shard": sorted(
                (output / "artifacts/evaluation-extractions").glob("shard_*.pt")
            )[0],
            "bank": output / "artifacts/reference-bank.pt",
            "scores": output / "artifacts/scores.csv",
            "provenance": output / "artifacts/provenance.json",
            "report": output / "report/report.md",
        }
        paths[leaf].write_bytes(b"late mutation")
        return result

    monkeypatch.setattr(cli, "_run_pipeline_stages", mutate_after_stages)

    with pytest.raises((ValueError, RuntimeError)):
        _run(inputs, small_config, extractor_factory=RejectingExtractor)


def test_identical_resume_preserves_artifacts_tree_and_directory_metadata(
    tmp_path, small_config
):
    inputs = _inputs(tmp_path)
    _run(inputs, small_config)
    artifacts = inputs[-1] / "artifacts"
    os.utime(artifacts, ns=(1_000_000_000, 1_000_000_000))
    before_tree = _tree_state(artifacts)
    before_metadata = _directory_metadata(artifacts)

    assert _run(inputs, small_config, extractor_factory=RejectingExtractor)
    assert _tree_state(artifacts) == before_tree
    assert _directory_metadata(artifacts) == before_metadata


def test_existing_provenance_disappearance_never_falls_back_to_creation(
    tmp_path, small_config, monkeypatch
):
    inputs = _inputs(tmp_path)
    _run(inputs, small_config)
    artifacts = inputs[-1] / "artifacts"
    provenance = artifacts / "provenance.json"
    original_validate = cli.validate_provenance
    validation_calls = 0

    def disappearing_validate(*args, **kwargs):
        nonlocal validation_calls
        validation_calls += 1
        provenance.unlink()
        return original_validate(*args, **kwargs)

    def forbidden_ensure(*_args, **_kwargs):
        raise AssertionError("existing provenance must not use creation")

    monkeypatch.setattr(cli, "validate_provenance", disappearing_validate)
    monkeypatch.setattr(cli, "ensure_provenance", forbidden_ensure)
    with pytest.raises(ValueError, match="provenance.*regular file"):
        _run(inputs, small_config, extractor_factory=RejectingExtractor)

    assert validation_calls == 1
    assert not provenance.exists()


def test_existing_provenance_validation_interrupt_preserves_artifacts(
    tmp_path, small_config, monkeypatch
):
    inputs = _inputs(tmp_path)
    _run(inputs, small_config)
    artifacts = inputs[-1] / "artifacts"
    os.utime(artifacts, ns=(1_000_000_000, 1_000_000_000))
    before_tree = _tree_state(artifacts)
    before_metadata = _directory_metadata(artifacts)

    def interrupt_validation(*_args, **_kwargs):
        raise KeyboardInterrupt("provenance validation interrupted")

    def forbidden_ensure(*_args, **_kwargs):
        raise AssertionError("existing provenance must not use creation")

    monkeypatch.setattr(cli, "validate_provenance", interrupt_validation)
    monkeypatch.setattr(cli, "ensure_provenance", forbidden_ensure)
    with pytest.raises(KeyboardInterrupt, match="validation interrupted"):
        _run(inputs, small_config, extractor_factory=RejectingExtractor)

    assert _tree_state(artifacts) == before_tree
    assert _directory_metadata(artifacts) == before_metadata


_LATE_FINAL_AUDIT_MUTATIONS = (
    ("artifacts/provenance.json", "scores"),
    ("checkpoint", "scores"),
    ("artifacts/reference-extractions/manifest.json", "scores"),
    ("artifacts/reference-extractions/shard_00000.pt", "scores"),
    ("artifacts/evaluation-extractions/manifest.json", "scores"),
    *(
        (
            f"artifacts/evaluation-extractions/shard_{index:05d}.pt",
            "scores",
        )
        for index in range(9)
    ),
    ("artifacts/reference-bank.pt", "scores"),
    ("artifacts/scores.csv", "scores"),
    *((f"report/{relative}", "report") for relative in REPORT_FILES),
)


@pytest.mark.parametrize(
    "case",
    _LATE_FINAL_AUDIT_MUTATIONS,
    ids=[relative for relative, _hook in _LATE_FINAL_AUDIT_MUTATIONS],
)
def test_terminal_sweep_rejects_leaf_mutated_from_a_later_audit_hook(
    tmp_path, small_config, monkeypatch, case
):
    relative, hook = case
    inputs = _inputs(tmp_path)
    _run(inputs, small_config)
    output = inputs[-1]
    target = inputs[2] if relative == "checkpoint" else output / relative
    score_loads = 0
    original_scores = cli._load_stable_scores
    original_bundle = cli._bundle_bytes

    def mutate_after_scores(*args, **kwargs):
        nonlocal score_loads
        result = original_scores(*args, **kwargs)
        score_loads += 1
        if score_loads == 2 and hook == "scores":
            target.write_bytes(b"mutation after earlier semantic validation")
        return result

    def mutate_after_bundle(*args, **kwargs):
        result = original_bundle(*args, **kwargs)
        if hook == "report":
            target.write_bytes(b"mutation after report semantic validation")
        return result

    monkeypatch.setattr(cli, "_load_stable_scores", mutate_after_scores)
    monkeypatch.setattr(cli, "_bundle_bytes", mutate_after_bundle)

    with pytest.raises((ValueError, RuntimeError)):
        _run(inputs, small_config, extractor_factory=RejectingExtractor)
