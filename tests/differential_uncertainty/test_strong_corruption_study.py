import csv
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import differential_uncertainty.strong_corruption_study as study
from differential_uncertainty.artifacts import ShardWriter
from differential_uncertainty.strong_corruption_study import (
    AGGREGATIONS,
    BANK_VARIANTS,
    DISTANCES,
    Bank,
    CandidateRows,
    StudyConfig,
    aggregate_queries,
    build_bank,
    build_reference_candidates,
    match_reference_queries,
    query_distances,
    reservoir_indices,
    score_image_group,
    split_image_ids,
    top_query_entropy,
)


@pytest.fixture
def reference_candidates():
    return CandidateRows(
        vectors=torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [10.0, 1.0],
                [11.0, 1.0],
                [12.0, 1.0],
                [13.0, 1.0],
            ]
        ),
        matched=torch.tensor([True, True, True, True, False, False, False, False]),
        confidence=torch.tensor([0.9, 0.8, 0.7, 0.6, 0.9, 0.8, 0.4, 0.3]),
    )


@pytest.fixture
def bank_fixture():
    def make_bank(vectors):
        values = torch.as_tensor(vectors, dtype=torch.float32, device="cpu")
        return Bank(values, None, None, 0, len(values))

    return make_bank


@pytest.fixture
def strong_group():
    def record(severity):
        offset = float(severity) / 10
        return {
            "image_id": "image-1",
            "family": "fog",
            "severity": severity,
            "boxes": torch.tensor(
                [
                    [0.1, 0.1, 0.1, 0.1],
                    [0.2, 0.2, 0.1, 0.1],
                    [0.3, 0.3, 0.1, 0.1],
                    [0.4, 0.4, 0.1, 0.1],
                    [0.5, 0.5, 0.1, 0.1],
                    [0.9, 0.9, 0.1, 0.1],
                    [0.9, 0.9, 0.1, 0.1],
                ]
            ),
            "logits": torch.tensor(
                [
                    [4.0 - offset, -4.0],
                    [2.0 - offset, 0.0],
                    [1.0 - offset, 1.0 - offset],
                    [0.0, 0.0],
                    [-1.0, -1.0],
                    [10.0, -10.0],
                    [10.0, -10.0],
                ]
            ),
            "persistence": torch.tensor(
                [
                    [0.0 + offset, 0.0],
                    [1.0 + offset, 0.0],
                    [0.0, 1.0 + offset],
                    [1.0, 1.0 + offset],
                    [2.0 + offset, 2.0],
                    [9.0, 9.0],
                    [9.0, 9.0],
                ]
            ),
        }

    return tuple(record(severity) for severity in (0, 4, 5))


@pytest.fixture
def tiny_study(tmp_path):
    reference_root = tmp_path / "reference"
    evaluation_root = tmp_path / "evaluation"
    image_root = tmp_path / "images"
    image_root.mkdir()

    def write_manifest(path, image_ids):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("image_id", "image_path"))
            for image_id in image_ids:
                image = image_root / f"{image_id}.jpg"
                image.write_bytes(f"image-{image_id}".encode())
                writer.writerow((str(image_id), image))

    reference_ids = (1, 2)
    evaluation_ids = (10, 11, 12, 13)
    write_manifest(
        reference_root / "inputs" / "reference-manifest.csv", reference_ids
    )
    write_manifest(
        evaluation_root / "inputs" / "evaluation-manifest.csv", evaluation_ids
    )

    config_metadata = {
        "query_count": 8,
        "class_count": 2,
        "persistence_layer": 2,
        "persistence_dim": 3,
    }
    shared_metadata = {
        "checkpoint_sha256": "a" * 64,
        "config": config_metadata,
    }

    def record(image_id, severity, *, family_offset=0.0):
        query = torch.arange(8, dtype=torch.float32)
        image_offset = float(int(image_id) % 10) / 20
        severity_offset = float(severity) / 5
        return {
            "image_id": str(image_id),
            "severity": severity,
            "boxes": torch.stack(
                (
                    0.1 + query / 10,
                    0.1 + query / 12,
                    0.05 + query / 100,
                    0.06 + query / 100,
                ),
                dim=1,
            ),
            "logits": torch.stack(
                (
                    3.0 - query / 20,
                    -1.0 + query / 30,
                ),
                dim=1,
            ),
            "persistence": torch.stack(
                (
                    1.0 + query + image_offset + severity_offset + family_offset,
                    2.0 + query.square() / 10 + severity_offset,
                    3.0 + query / 7 + image_offset + family_offset,
                ),
                dim=1,
            ),
        }

    reference_cache = (
        reference_root / "reference-artifacts" / "reference-extractions"
    )
    with ShardWriter(
        reference_cache,
        {"stage": "reference", "input_id": "reference", **shared_metadata},
        shard_size=2,
    ) as writer:
        for image_id in reference_ids:
            writer.add(record(image_id, 0))

    families = ("fog", "snow")
    for family_index, family in enumerate(families):
        cache = (
            evaluation_root
            / "corruptions"
            / family
            / "artifacts"
            / "evaluation-extractions"
        )
        with ShardWriter(
            cache,
            {
                "stage": "evaluation",
                "input_id": family,
                "corruption": {"name": family},
                **shared_metadata,
            },
            shard_size=8,
        ) as writer:
            for image_id in evaluation_ids:
                for severity in range(6):
                    writer.add(
                        record(
                            image_id,
                            severity,
                            family_offset=float(family_index) / 10,
                        )
                    )

    roster = {
        "schema_version": 1,
        "checkpoint_sha256": shared_metadata["checkpoint_sha256"],
        "reference_manifest_sha256": "b" * 64,
        "evaluation_manifest_sha256": "c" * 64,
        "source_sha256": "d" * 64,
        "config": config_metadata,
        "runtime": {},
        "corruptions": [
            {
                "name": family,
                "severities": [
                    {"level": level, "parameter": float(level)}
                    for level in range(6)
                ],
            }
            for family in families
        ],
    }
    (evaluation_root / "corruption-roster.json").write_text(
        json.dumps(roster), encoding="utf-8"
    )

    images = [
        {"id": image_id, "width": 100, "height": 100}
        for image_id in (*reference_ids, *evaluation_ids)
    ]
    annotations = []
    for image_id in reference_ids:
        for index in range(3):
            annotations.append(
                {
                    "id": image_id * 10 + index,
                    "image_id": image_id,
                    "category_id": 1 + index % 2,
                    "bbox": [8 + index * 10, 8 + index * 8, 6, 6],
                    "iscrowd": 0,
                }
            )
    annotation_path = tmp_path / "instances.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 1}, {"id": 2}],
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        reference_root=reference_root,
        evaluation_root=evaluation_root,
        annotations=annotation_path,
        config=StudyConfig(
            reference_count=2,
            evaluation_count=4,
            selection_count=2,
            validation_count=2,
            families=families,
            bank_capacity=6,
            bootstrap_draws=50,
        ),
    )


def test_tiny_study_filters_levels_and_reports_each_family(tiny_study, tmp_path):
    result = study.run_study(
        tiny_study.reference_root,
        tiny_study.evaluation_root,
        tiny_study.annotations,
        tmp_path / "output",
        config=tiny_study.config,
        device="cpu",
    )
    assert result.levels_read == {0, 4, 5}
    assert result.reported_families == {"fog", "snow"}
    assert {
        path.name for path in (tmp_path / "output").iterdir() if path.is_file()
    } == {"results.csv", "summary.json", "report.md"}


def test_tiny_study_writes_exact_cache_and_report_contract(tiny_study, tmp_path):
    output = tmp_path / "output"
    result = study.run_study(
        tiny_study.reference_root,
        tiny_study.evaluation_root,
        tiny_study.annotations,
        output,
        config=tiny_study.config,
        device="cpu",
    )

    geometry_by_distance = {
        "mean_5_euclidean": "euclidean",
        "fifth_neighbor_euclidean": "euclidean",
        "mean_5_standardized_euclidean": "standardized_euclidean",
        "mean_5_cosine": "cosine",
    }
    expected_caches = {
        f"{bank}-{geometry}-seed44.pt"
        for bank in BANK_VARIANTS
        for geometry in ("euclidean", "standardized_euclidean", "cosine")
    }
    selected_geometry = geometry_by_distance[result.selected_policy.distance]
    expected_caches.update(
        f"{result.selected_policy.bank}-{selected_geometry}-seed{seed}.pt"
        for seed in (42, 43, 45, 46)
    )
    assert {path.name for path in (output / "cache").iterdir()} == expected_caches

    with (output / "results.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames == [
        "row_type",
        "split",
        "policy_id",
        "bank",
        "distance",
        "aggregation",
        "seed",
        "corruption",
        "severity",
        "method",
        "point",
        "lower",
        "upper",
        "count",
    ]
    assert {row["row_type"] for row in rows} == {
        "candidate_family",
        "selected_method",
        "paired_difference",
        "conditional",
        "seed",
    }
    selected_methods = {
        (row["corruption"], int(row["severity"]), row["method"])
        for row in rows
        if row["row_type"] == "selected_method"
    }
    assert selected_methods == {
        (family, severity, method)
        for family in tiny_study.config.families
        for severity in (4, 5)
        for method in (
            "fingerprint",
            "direct_confidence_max",
            "softmax_entropy_top_confidence_query",
        )
    }
    candidate_rows = [
        row for row in rows if row["row_type"] == "candidate_family"
    ]
    assert len(candidate_rows) == 100 * 2 * len(tiny_study.config.families)
    assert {
        (row["corruption"], row["severity"], row["method"])
        for row in rows
        if row["row_type"] == "paired_difference"
    } == {
        (family, severity, method)
        for family in tiny_study.config.families
        for severity in ("4", "5")
        for method in (
            "fingerprint_minus_confidence",
            "fingerprint_minus_entropy",
        )
    }
    assert len([row for row in rows if row["row_type"] == "seed"]) == 5
    interval_rows = [
        row
        for row in rows
        if row["row_type"]
        in {"selected_method", "paired_difference", "conditional"}
    ]
    assert interval_rows
    for row in interval_rows:
        assert row["lower"] != "" and row["upper"] != ""
        lower = float(row["lower"])
        upper = float(row["upper"])
        point = float(row["point"])
        assert lower <= upper
        if row["row_type"] != "paired_difference":
            assert 0.0 <= lower <= upper <= 1.0
            assert 0.0 <= point <= 1.0
        else:
            assert -1.0 <= lower <= upper <= 1.0
            assert -1.0 <= point <= 1.0

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["conclusions"]) == {
        "detects_corruption",
        "better_than_confidence",
        "information_after_confidence",
    }
    report = (output / "report.md").read_text(encoding="utf-8")
    report.encode("ascii")
    assert "0.714" not in report
    assert report.count("Corruption |") == 2
    assert "Fingerprint distance:" in report
    assert "Confidence score: 1 - maximum sigmoid confidence." in report
    assert (
        "Entropy score: normalized Shannon entropy of the "
        "highest-confidence retained query."
    ) in report
    assert "family/image union-unpadded query set" in report
    assert (
        "Exploratory coverage: nonempty family/severity tasks=4/4; tasks with at "
        "most 5 eligible pairs=4; tasks with exactly 1 eligible pair=0. Coverage "
        "alone does not establish uniform per-corruption complementarity."
    ) in report
    for family in tiny_study.config.families:
        assert report.count(f"{family} |") == 2
    assert report.rstrip().endswith("Levels 1 through 3 were not evaluated.")


def test_conditional_coverage_wording_handles_partial_and_zero_coverage():
    assert study._conditional_coverage_line((0, 1, 6)) == (
        "Exploratory coverage: nonempty family/severity tasks=2/3; tasks with at "
        "most 5 eligible pairs=1; tasks with exactly 1 eligible pair=1. Coverage "
        "alone does not establish uniform per-corruption complementarity."
    )
    assert study._conditional_coverage_line((0, 0)) == (
        "Exploratory coverage: nonempty family/severity tasks=0/2; tasks with at "
        "most 5 eligible pairs=0; tasks with exactly 1 eligible pair=0. Coverage "
        "alone does not establish uniform per-corruption complementarity."
    )


def test_evidence_conclusions_use_the_three_fixed_thresholds():
    interval = study.ScoreInterval
    result = study._evidence_conclusions(
        study.ValidationBootstrap(
            fingerprint=interval(0.7, 0.6, 0.8, 2),
            confidence=interval(0.6, 0.4, 0.8, 2),
            entropy=interval(0.5, 0.3, 0.7, 2),
            fingerprint_minus_confidence=interval(-0.1, -0.2, 0.0, 2),
            fingerprint_minus_entropy=interval(0.2, 0.1, 0.3, 2),
            conditional=interval(0.7, 0.5, 0.8, 3),
        ),
        conditional_complete=True,
    )

    assert result == {
        "detects_corruption": "supported",
        "better_than_confidence": "not_supported",
        "information_after_confidence": "inconclusive",
    }
    assert study._evidence_status(interval(None, None, None, 0), 0.5) == (
        "inconclusive"
    )


def test_incomplete_conditional_tasks_override_a_supported_pooled_interval():
    interval = study.ScoreInterval
    pooled = study.ValidationBootstrap(
        fingerprint=interval(0.9, 0.8, 1.0, 2),
        confidence=interval(0.5, 0.4, 0.6, 2),
        entropy=interval(0.5, 0.4, 0.6, 2),
        fingerprint_minus_confidence=interval(0.4, 0.2, 0.6, 2),
        fingerprint_minus_entropy=interval(0.4, 0.2, 0.6, 2),
        conditional=interval(1.0, 1.0, 1.0, 3),
    )

    assert study._evidence_conclusions(
        pooled, conditional_complete=False
    )["information_after_confidence"] == "inconclusive"


def test_selected_csv_rows_preserve_each_task_interval():
    interval = study.ScoreInterval
    bootstrap = study.ValidationBootstrap(
        fingerprint=interval(0.61, 0.51, 0.71, 2),
        confidence=interval(0.62, 0.52, 0.72, 2),
        entropy=interval(0.63, 0.53, 0.73, 2),
        fingerprint_minus_confidence=interval(-0.01, -0.11, 0.09, 2),
        fingerprint_minus_entropy=interval(-0.02, -0.12, 0.08, 2),
        conditional=interval(0.64, 0.54, 0.74, 3),
    )
    config = StudyConfig(families=("fog",))
    policy = study.Policy(
        "matched", "mean_5_euclidean", "mean_all", config.primary_seed
    )
    rows = study._selected_csv_rows(
        policy,
        {("fog", 4): bootstrap, ("fog", 5): bootstrap},
        {seed: 0.5 for seed in config.sensitivity_seeds},
        config=config,
    )
    expected = {
        "fingerprint": bootstrap.fingerprint,
        "direct_confidence_max": bootstrap.confidence,
        "softmax_entropy_top_confidence_query": bootstrap.entropy,
        "fingerprint_minus_confidence": bootstrap.fingerprint_minus_confidence,
        "fingerprint_minus_entropy": bootstrap.fingerprint_minus_entropy,
        "confidence_conditioned_concordance": bootstrap.conditional,
    }

    for row in rows:
        if row["row_type"] not in {
            "selected_method",
            "paired_difference",
            "conditional",
        }:
            continue
        wanted = expected[row["method"]]
        assert (
            row["point"],
            row["lower"],
            row["upper"],
            row["count"],
        ) == (wanted.point, wanted.lower, wanted.upper, wanted.count)


def test_tiny_study_requires_every_configured_roster_family(
    tiny_study, tmp_path
):
    roster_path = tiny_study.evaluation_root / "corruption-roster.json"
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    roster["corruptions"] = [
        corruption
        for corruption in roster["corruptions"]
        if corruption["name"] != "snow"
    ]
    roster_path.write_text(json.dumps(roster), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="configured corruption family 'snow' must occur exactly once",
    ):
        study.run_study(
            tiny_study.reference_root,
            tiny_study.evaluation_root,
            tiny_study.annotations,
            tmp_path / "output",
            config=tiny_study.config,
            device="cpu",
        )


def test_valid_distance_caches_are_reused(tiny_study, tmp_path, monkeypatch):
    output = tmp_path / "output"
    study.run_study(
        tiny_study.reference_root,
        tiny_study.evaluation_root,
        tiny_study.annotations,
        output,
        config=tiny_study.config,
        device="cpu",
    )

    def must_not_compute(*_args, **_kwargs):
        raise AssertionError("valid distance cache should be reused")

    monkeypatch.setattr(study, "_compute_neighbor_tensor", must_not_compute)
    study.run_study(
        tiny_study.reference_root,
        tiny_study.evaluation_root,
        tiny_study.annotations,
        output,
        config=tiny_study.config,
        device="cpu",
    )


def test_cache_metadata_mismatch_recomputes_only_that_cache(
    tiny_study, tmp_path, monkeypatch
):
    output = tmp_path / "output"
    study.run_study(
        tiny_study.reference_root,
        tiny_study.evaluation_root,
        tiny_study.annotations,
        output,
        config=tiny_study.config,
        device="cpu",
    )
    changed = output / "cache" / "matched-cosine-seed44.pt"
    payload = torch.load(changed, map_location="cpu", weights_only=True)
    payload["metadata"]["families"] = ["different"]
    torch.save(payload, changed)

    calls = 0
    original = study._compute_neighbor_tensor

    def count_compute(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(study, "_compute_neighbor_tensor", count_compute)
    study.run_study(
        tiny_study.reference_root,
        tiny_study.evaluation_root,
        tiny_study.annotations,
        output,
        config=tiny_study.config,
        device="cpu",
    )

    repaired = torch.load(changed, map_location="cpu", weights_only=True)
    assert repaired["metadata"]["families"] == ["fog", "snow"]
    assert calls == 2 * 4 * 3


def test_bank_capacity_change_recomputes_and_overwrites_caches(
    tiny_study, tmp_path, monkeypatch
):
    output = tmp_path / "output"
    study.run_study(
        tiny_study.reference_root,
        tiny_study.evaluation_root,
        tiny_study.annotations,
        output,
        config=tiny_study.config,
        device="cpu",
    )
    changed = output / "cache" / "matched-cosine-seed44.pt"

    calls = 0
    original = study._compute_neighbor_tensor

    def count_compute(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(study, "_compute_neighbor_tensor", count_compute)
    changed_config = replace(tiny_study.config, bank_capacity=5)
    study.run_study(
        tiny_study.reference_root,
        tiny_study.evaluation_root,
        tiny_study.annotations,
        output,
        config=changed_config,
        device="cpu",
    )

    payload = torch.load(changed, map_location="cpu", weights_only=True)
    assert payload["metadata"]["bank_capacity"] == 5
    assert calls > 0


def test_source_digest_change_recomputes_caches_and_is_reported(
    tiny_study, tmp_path, monkeypatch
):
    output = tmp_path / "output"
    study.run_study(
        tiny_study.reference_root,
        tiny_study.evaluation_root,
        tiny_study.annotations,
        output,
        config=tiny_study.config,
        device="cpu",
    )
    changed = output / "cache" / "matched-cosine-seed44.pt"
    before = torch.load(changed, map_location="cpu", weights_only=True)

    annotations = json.loads(tiny_study.annotations.read_text(encoding="utf-8"))
    annotations["source_revision"] = "changed"
    tiny_study.annotations.write_text(json.dumps(annotations), encoding="utf-8")

    calls = 0
    original = study._compute_neighbor_tensor

    def count_compute(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(study, "_compute_neighbor_tensor", count_compute)
    study.run_study(
        tiny_study.reference_root,
        tiny_study.evaluation_root,
        tiny_study.annotations,
        output,
        config=tiny_study.config,
        device="cpu",
    )

    after = torch.load(changed, map_location="cpu", weights_only=True)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert after["metadata"]["cache_schema_version"] == 1
    assert before["metadata"].get("source_digests") != after["metadata"][
        "source_digests"
    ]
    assert after["metadata"]["source_digests"] == summary["source_digests"]
    assert set(summary["source_digests"]) == {
        "annotations_sha256",
        "corruption_roster_sha256",
        "reference_artifact_manifest_sha256",
        "family_artifact_manifests_sha256",
    }
    assert set(summary["source_digests"]["family_artifact_manifests_sha256"]) == {
        "fog",
        "snow",
    }
    assert calls > 0


def test_validation_failure_happens_after_selection_is_frozen(
    tiny_study, tmp_path, monkeypatch
):
    validation_ids = set(
        split_image_ids(("10", "11", "12", "13"), 2).validation
    )
    original_neighbors = study._cache_neighbors
    original_selector = study.select_policy
    selected = []

    def fail_on_validation(state, record, geometry):
        if record["image_id"] in validation_ids:
            raise ValueError("validation-only sentinel")
        return original_neighbors(state, record, geometry)

    def observe_selector(rows):
        policy = original_selector(rows)
        selected.append(policy)
        return policy

    monkeypatch.setattr(study, "_cache_neighbors", fail_on_validation)
    monkeypatch.setattr(study, "select_policy", observe_selector)

    with pytest.raises(
        ValueError,
        match="validation scoring failed after freezing policy",
    ) as caught:
        study.run_study(
            tiny_study.reference_root,
            tiny_study.evaluation_root,
            tiny_study.annotations,
            tmp_path / "output",
            config=tiny_study.config,
            device="cpu",
        )

    assert len(selected) == 1
    assert selected[0].policy_id in str(caught.value)
    assert "validation-only sentinel" in str(caught.value)


def test_module_cli_accepts_only_the_study_runtime_arguments(tmp_path):
    args = study._build_parser().parse_args(
        [
            "--reference-run",
            str(tmp_path / "reference"),
            "--evaluation-benchmark",
            str(tmp_path / "evaluation"),
            "--annotations",
            str(tmp_path / "instances.json"),
            "--output-dir",
            str(tmp_path / "output"),
            "--device",
            "cuda:0",
        ]
    )

    assert args.reference_run == str(tmp_path / "reference")
    assert args.evaluation_benchmark == str(tmp_path / "evaluation")
    assert args.annotations == str(tmp_path / "instances.json")
    assert args.output_dir == str(tmp_path / "output")
    assert args.device == "cuda:0"


def _single_reference_record():
    return {
        "image_id": "one",
        "logits": torch.tensor([[0.0, 0.0]]),
        "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        "persistence": torch.tensor([[1.0, 2.0]]),
    }


def test_public_panel_is_fixed_and_contains_100_candidates():
    config = StudyConfig()
    assert (config.reference_count, config.evaluation_count) == (1_000, 250)
    assert (config.selection_count, config.validation_count) == (150, 100)
    assert config.levels == (0, 4, 5)
    assert config.bank_capacity == 2_000 and config.primary_seed == 44
    assert len(BANK_VARIANTS) * len(DISTANCES) * len(AGGREGATIONS) == 100


def test_split_is_order_independent_and_disjoint():
    ids = tuple(f"image-{index}" for index in range(10))
    left = split_image_ids(ids, selection_count=6)
    right = split_image_ids(reversed(ids), selection_count=6)
    assert left == right
    assert len(left.selection) == 6 and len(left.validation) == 4
    assert set(left.selection).isdisjoint(left.validation)


def test_algorithm_r_has_a_fixed_oracle():
    assert reservoir_indices(10, capacity=4, seed=44).tolist() == [5, 1, 7, 4]


def test_hungarian_matching_marks_only_the_assigned_query():
    record = {
        "image_id": "1",
        "logits": torch.tensor([[8.0, -8.0], [-8.0, 8.0], [-8.0, -8.0]]),
        "boxes": torch.tensor(
            [[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2], [0.9, 0.9, 0.1, 0.1]]
        ),
        "persistence": torch.arange(12, dtype=torch.float32).reshape(3, 4),
    }
    annotations = [
        {"id": 17, "category_id": 5, "bbox": [40, 40, 20, 20], "iscrowd": 0}
    ]
    result = match_reference_queries(
        record,
        annotations,
        valid_query_ids=torch.tensor([0, 1, 2]),
        width=100,
        height=100,
        category_ids=(3, 5),
    )
    assert result.tolist() == [False, True, False]


def test_historical_focal_cost_does_not_clip_saturated_probabilities():
    record = {
        "image_id": "1",
        "logits": torch.tensor([[-100.0], [-20.0]]),
        "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]] * 2),
        "persistence": torch.zeros((2, 1)),
    }
    annotations = [
        {"id": 17, "category_id": 3, "bbox": [40, 40, 20, 20], "iscrowd": 0}
    ]

    result = match_reference_queries(
        record,
        annotations,
        valid_query_ids=torch.tensor([0, 1]),
        width=100,
        height=100,
        category_ids=(3,),
    )

    assert result.tolist() == [False, True]


def test_reference_candidates_are_sorted_and_remove_the_padded_tail():
    record_a = {
        "image_id": "a",
        "logits": torch.tensor([[-8.0, 8.0], [8.0, -8.0]]),
        "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.9, 0.9, 0.1, 0.1]]),
        "persistence": torch.tensor([[10.0, 10.0], [11.0, 11.0]]),
    }
    record_b = {
        "image_id": "b",
        "logits": torch.tensor(
            [[8.0, -8.0], [-8.0, 8.0], [-8.0, -8.0], [-8.0, -8.0]]
        ),
        "boxes": torch.tensor(
            [
                [0.1, 0.1, 0.1, 0.1],
                [0.2, 0.2, 0.1, 0.1],
                [0.9, 0.9, 0.1, 0.1],
                [0.9, 0.9, 0.1, 0.1],
            ]
        ),
        "persistence": torch.tensor(
            [[20.0, 20.0], [21.0, 21.0], [99.0, 99.0], [99.0, 99.0]]
        ),
    }
    candidates = build_reference_candidates(
        (record_b, record_a),
        annotations_by_image={
            "a": [
                {
                    "id": 17,
                    "category_id": 5,
                    "bbox": [40, 40, 20, 20],
                    "iscrowd": 0,
                }
            ],
            "b": [],
        },
        image_sizes={"a": (100, 100), "b": (100, 100)},
        category_ids=(3, 5),
    )

    assert candidates.vectors.tolist() == [
        [10.0, 10.0],
        [11.0, 11.0],
        [20.0, 20.0],
        [21.0, 21.0],
    ]
    assert candidates.vectors.dtype == torch.float32
    assert candidates.vectors.device.type == "cpu"
    assert candidates.matched.tolist() == [True, False, False, False]
    assert candidates.confidence.tolist() == pytest.approx(
        [torch.sigmoid(torch.tensor(8.0)).item()] * 4
    )


def test_reference_candidates_reject_missing_annotation_entries():
    with pytest.raises(
        ValueError, match="missing annotations for reference image 'one'"
    ):
        build_reference_candidates(
            (_single_reference_record(),),
            annotations_by_image={},
            image_sizes={"one": (100, 100)},
            category_ids=(3, 5),
        )


def test_reference_candidates_accept_explicit_empty_annotations():
    candidates = build_reference_candidates(
        (_single_reference_record(),),
        annotations_by_image={"one": []},
        image_sizes={"one": (100, 100)},
        category_ids=(3, 5),
    )

    assert candidates.matched.tolist() == [False]


def test_infeasible_bank_names_population_and_counts(reference_candidates):
    with pytest.raises(
        ValueError,
        match="bank variant 'matched' requires 5 matched rows but only 4 are available",
    ):
        build_bank(
            reference_candidates,
            variant="matched",
            distance="mean_5_euclidean",
            capacity=5,
            seed=44,
        )


def test_standardization_uses_complete_population_moments(reference_candidates):
    bank = build_bank(
        reference_candidates,
        variant="all_valid",
        distance="mean_5_standardized_euclidean",
        capacity=4,
        seed=44,
    )
    expected_mean = reference_candidates.vectors.mean(dim=0)
    expected_scale = reference_candidates.vectors.std(dim=0, correction=0)
    sampled = reference_candidates.vectors.index_select(
        0, reservoir_indices(8, capacity=4, seed=44)
    )

    assert torch.allclose(bank.mean, expected_mean)
    assert torch.allclose(bank.scale, expected_scale)
    assert torch.allclose(bank.vectors, (sampled - expected_mean) / expected_scale)


def test_balanced_standardization_weights_each_population_one_half():
    candidates = CandidateRows(
        vectors=torch.tensor([[0.0], [2.0], [10.0], [12.0], [14.0], [16.0]]),
        matched=torch.tensor([True, True, False, False, False, False]),
        confidence=torch.ones(6),
    )
    bank = build_bank(
        candidates,
        variant="balanced",
        distance="mean_5_standardized_euclidean",
        capacity=2,
        seed=44,
    )

    assert bank.mean.item() == pytest.approx(7.0)
    assert bank.scale.item() == pytest.approx(39.0**0.5)


def test_balanced_bank_uses_truncated_sha256_reservoir_seeds():
    candidates = CandidateRows(
        vectors=torch.tensor([[float(value)] for value in (*range(6), *range(10, 16))]),
        matched=torch.tensor([True] * 6 + [False] * 6),
        confidence=torch.ones(12),
    )

    bank = build_bank(
        candidates,
        variant="balanced",
        distance="mean_5_euclidean",
        capacity=4,
        seed=44,
    )

    assert bank.vectors.squeeze(1).tolist() == [3.0, 1.0, 14.0, 15.0]


@pytest.mark.parametrize("variant", BANK_VARIANTS)
def test_all_bank_variants_have_exact_capacity(reference_candidates, variant):
    bank = build_bank(
        reference_candidates,
        variant=variant,
        distance="mean_5_euclidean",
        capacity=4,
        seed=44,
    )
    assert isinstance(bank, Bank)
    assert bank.vectors.shape == (4, 2)
    if variant == "balanced":
        assert (bank.matched_count, bank.background_count) == (2, 2)


def test_distance_and_aggregation_oracles(bank_fixture):
    bank = bank_fixture([[1.0], [3.0], [8.0], [9.0], [10.0]])
    query = torch.tensor([[0.0]])
    assert query_distances(query, bank, "mean_5_euclidean").item() == pytest.approx(
        6.2
    )
    assert query_distances(query, bank, "fifth_neighbor_euclidean").item() == 10.0

    distances = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0])
    confidence = torch.tensor([0.1, 0.9, 0.3, 0.2, 0.4])
    query_ids = torch.arange(5)
    assert aggregate_queries(distances, confidence, query_ids, "mean_all") == 22.0
    assert aggregate_queries(distances, confidence, query_ids, "q90_all") == 100.0
    assert aggregate_queries(
        distances, confidence, query_ids, "top20_mean_all"
    ) == 100.0
    assert aggregate_queries(
        distances, confidence, query_ids, "top_confidence_query"
    ) == 2.0
    assert aggregate_queries(
        distances, confidence, query_ids, "confidence_weighted_mean"
    ) == pytest.approx(43.6 / 1.9)


def test_entropy_uses_softmax_on_the_highest_confidence_query():
    logits = torch.tensor([[2.0, 0.0], [1.0, 1.0]])
    probability = logits[0].softmax(0)
    expected = -torch.xlogy(probability, probability).sum() / torch.log(
        torch.tensor(2.0)
    )
    assert top_query_entropy(logits, torch.tensor([0, 1])) == pytest.approx(
        float(expected)
    )


def test_one_union_mask_is_reused_for_fingerprint_confidence_and_entropy(
    strong_group, bank_fixture
):
    bank = bank_fixture(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]]
    )
    rows = score_image_group(strong_group, bank, "mean_5_euclidean")
    assert {row.severity for row in rows} == {0, 4, 5}
    assert len({row.valid_query_ids for row in rows}) == 1
    assert all(row.valid_query_ids == (0, 1, 2, 3, 4) for row in rows)
    assert all(row.confidence_score == 1.0 - row.raw_confidence for row in rows)
    clean = next(row for row in rows if row.severity == 0)
    expected_confidence = torch.sigmoid(torch.tensor(4.0)).item()
    expected_probability = strong_group[0]["logits"][0].softmax(0)
    expected_entropy = float(
        -torch.xlogy(expected_probability, expected_probability).sum()
        / torch.log(torch.tensor(2.0))
    )
    assert clean.raw_confidence == pytest.approx(expected_confidence)
    assert clean.entropy_score == pytest.approx(expected_entropy)


@pytest.mark.parametrize(
    ("queries", "bank", "error"),
    [
        (
            torch.tensor([[float("nan")]]),
            [[0.0], [1.0], [2.0], [3.0], [4.0]],
            "finite",
        ),
        (
            torch.tensor([[0.0]]),
            [[0.0], [1.0], [2.0], [3.0], [float("inf")]],
            "finite",
        ),
        (
            torch.tensor([[0.0, 1.0]]),
            [[0.0], [1.0], [2.0], [3.0], [4.0]],
            "matching feature dimensions",
        ),
    ],
)
def test_query_distances_reject_invalid_numeric_inputs(
    queries, bank, error, bank_fixture
):
    with pytest.raises(ValueError, match=error):
        query_distances(queries, bank_fixture(bank), "mean_5_euclidean")


def test_query_distances_reject_a_bank_smaller_than_five(bank_fixture):
    with pytest.raises(ValueError, match="at least five"):
        query_distances(
            torch.tensor([[0.0]]),
            bank_fixture([[0.0], [1.0], [2.0], [3.0]]),
            "mean_5_euclidean",
        )


@pytest.mark.parametrize(
    ("queries", "bank"),
    [
        (
            torch.tensor([[0.0, 0.0]]),
            [[1.0, 0.0], [1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [2.0, 2.0]],
        ),
        (
            torch.tensor([[1.0, 0.0]]),
            [[0.0, 0.0], [1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [2.0, 2.0]],
        ),
    ],
)
def test_cosine_distance_rejects_zero_norm_rows(queries, bank, bank_fixture):
    with pytest.raises(ValueError, match="zero-norm"):
        query_distances(queries, bank_fixture(bank), "mean_5_cosine")


def test_standardized_distance_uses_bank_moments():
    raw = torch.tensor([[1.0], [3.0], [5.0], [7.0], [9.0]])
    mean = torch.tensor([5.0])
    scale = torch.tensor([2.0])
    bank = Bank((raw - mean) / scale, mean, scale, 0, 5)

    result = query_distances(
        torch.tensor([[5.0]]), bank, "mean_5_standardized_euclidean"
    )

    assert result.item() == pytest.approx(1.2)


def test_cosine_distance_normalizes_and_uses_the_five_nearest_rows(bank_fixture):
    queries = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    vectors = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [-1.0, -1.0],
            [2.0, 1.0],
        ]
    )
    normalized_queries = queries / queries.norm(dim=1, keepdim=True)
    normalized_vectors = vectors / vectors.norm(dim=1, keepdim=True)
    expected = (1 - normalized_queries @ normalized_vectors.T).topk(
        5, largest=False, dim=1
    ).values.mean(dim=1)

    actual = query_distances(
        queries, bank_fixture(vectors), "mean_5_cosine"
    )

    assert torch.allclose(actual, expected)


def test_public_top_query_callers_break_ties_by_lowest_actual_query_id():
    distances = torch.tensor([8.0, 2.0])
    confidence = torch.tensor([0.7, 0.7])
    query_ids = torch.tensor([9, 3])
    assert (
        aggregate_queries(
            distances, confidence, query_ids, "top_confidence_query"
        )
        == 2.0
    )

    logits = torch.tensor([[2.0, 0.0], [2.0, 2.0]])
    assert top_query_entropy(logits, torch.tensor([9, 3])) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("query_ids", "error"),
    [
        (torch.tensor([0.0, 1.0]), "integer"),
        (torch.tensor([3, 3]), "unique"),
    ],
)
def test_top_query_entropy_rejects_malformed_query_ids(query_ids, error):
    with pytest.raises(ValueError, match=error):
        top_query_entropy(torch.tensor([[2.0, 0.0], [1.0, 1.0]]), query_ids)


def test_top_query_entropy_rejects_float32_conversion_overflow():
    logits = torch.tensor([[1e300, 0.0], [1.0, 1.0]], dtype=torch.float64)
    assert bool(torch.isfinite(logits).all())
    with pytest.raises(ValueError, match="finite"):
        top_query_entropy(logits, torch.tensor([0, 1]))


def test_score_image_group_rejects_missing_levels(strong_group, bank_fixture):
    with pytest.raises(ValueError, match="exactly levels 0, 4, and 5"):
        score_image_group(
            strong_group[:-1],
            bank_fixture(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]]
            ),
            "mean_5_euclidean",
        )


def test_score_image_group_rejects_mixed_images(strong_group, bank_fixture):
    malformed = [dict(record) for record in strong_group]
    malformed[0]["image_id"] = "other"
    with pytest.raises(ValueError, match="exactly one image"):
        score_image_group(
            malformed,
            bank_fixture(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]]
            ),
            "mean_5_euclidean",
        )


def test_query_distance_chunking_matches_the_direct_oracle(bank_fixture):
    bank_rows = study.BANK_ROW_CHUNK_SIZE + 1
    vectors = torch.arange(bank_rows * 2, dtype=torch.float32).reshape(bank_rows, 2)
    queries = torch.tensor([[0.25, 0.75], [513.0, 514.0]])
    direct = torch.cdist(
        queries,
        vectors,
        p=2,
        compute_mode="donot_use_mm_for_euclid_dist",
    ).topk(5, largest=False, dim=1).values.mean(dim=1)

    actual = query_distances(
        queries, bank_fixture(vectors), "mean_5_euclidean"
    )

    assert bank_rows > study.BANK_ROW_CHUNK_SIZE
    assert torch.allclose(actual, direct)


@pytest.mark.parametrize(
    ("distances", "confidence", "query_ids", "name", "error"),
    [
        (
            torch.tensor([[1.0]]),
            torch.tensor([1.0]),
            torch.tensor([0]),
            "mean_all",
            "one-dimensional",
        ),
        (
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([], dtype=torch.long),
            "mean_all",
            "at least one",
        ),
        (
            torch.tensor([1.0, 2.0]),
            torch.tensor([1.0]),
            torch.tensor([0, 1]),
            "mean_all",
            "align",
        ),
        (
            torch.tensor([float("nan")]),
            torch.tensor([1.0]),
            torch.tensor([0]),
            "mean_all",
            "finite",
        ),
        (
            torch.tensor([1.0]),
            torch.tensor([float("inf")]),
            torch.tensor([0]),
            "mean_all",
            "finite",
        ),
        (
            torch.tensor([1.0, 2.0]),
            torch.tensor([0.5, 0.5]),
            torch.tensor([3, 3]),
            "top_confidence_query",
            "unique",
        ),
        (
            torch.tensor([1.0]),
            torch.tensor([0.5]),
            torch.tensor([0.0]),
            "top_confidence_query",
            "integer",
        ),
        (
            torch.tensor([1.0, 2.0]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([0, 1]),
            "confidence_weighted_mean",
            "positive total",
        ),
        (
            torch.tensor([1.0, 2.0]),
            torch.tensor([-2.0, 1.0]),
            torch.tensor([0, 1]),
            "confidence_weighted_mean",
            "positive total",
        ),
    ],
)
def test_aggregate_queries_rejects_invalid_inputs(
    distances, confidence, query_ids, name, error
):
    with pytest.raises(ValueError, match=error):
        aggregate_queries(distances, confidence, query_ids, name)


def test_aggregate_queries_rejects_a_nonfinite_computed_result():
    with pytest.raises(ValueError, match="finite"):
        aggregate_queries(
            torch.tensor([3e38, 3e38]),
            torch.tensor([1.0, 1.0]),
            torch.tensor([0, 1]),
            "confidence_weighted_mean",
        )


def test_standardized_distance_requires_bank_moments(bank_fixture):
    with pytest.raises(ValueError, match="mean and scale"):
        query_distances(
            torch.tensor([[0.0]]),
            bank_fixture([[0.0], [1.0], [2.0], [3.0], [4.0]]),
            "mean_5_standardized_euclidean",
        )


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_standardized_distance_requires_positive_scale(scale):
    bank = Bank(
        torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]]),
        torch.tensor([0.0]),
        torch.tensor([scale]),
        0,
        5,
    )
    with pytest.raises(ValueError, match="positive"):
        query_distances(
            torch.tensor([[0.0]]), bank, "mean_5_standardized_euclidean"
        )


def test_standardized_distance_rejects_float32_transform_overflow():
    bank = Bank(
        torch.zeros((5, 1)),
        torch.tensor([0.0]),
        torch.tensor([torch.finfo(torch.float32).tiny]),
        0,
        5,
    )
    with pytest.raises(ValueError, match="finite"):
        query_distances(
            torch.tensor([[torch.finfo(torch.float32).max]]),
            bank,
            "mean_5_standardized_euclidean",
        )


def test_query_distances_rejects_nonfinite_computed_distances(bank_fixture):
    largest = torch.finfo(torch.float32).max
    bank = bank_fixture([[-largest]] * 5)
    with pytest.raises(ValueError, match="finite"):
        query_distances(
            torch.tensor([[largest]]), bank, "mean_5_euclidean"
        )


def test_score_image_group_rejects_a_union_mask_with_no_valid_queries(
    strong_group, bank_fixture
):
    malformed = []
    for record in strong_group:
        changed = dict(record)
        changed["boxes"] = torch.zeros_like(record["boxes"])
        changed["logits"] = torch.zeros_like(record["logits"])
        changed["persistence"] = torch.zeros_like(record["persistence"])
        malformed.append(changed)

    with pytest.raises(ValueError, match="no valid queries"):
        score_image_group(
            malformed,
            bank_fixture(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]]
            ),
            "mean_5_euclidean",
        )


@pytest.mark.parametrize("severities", [(0, 4, 4), (0, 4, 6)])
def test_score_image_group_rejects_duplicate_or_out_of_scope_severities(
    strong_group, bank_fixture, severities
):
    malformed = [
        dict(record, severity=severity)
        for record, severity in zip(strong_group, severities, strict=True)
    ]
    with pytest.raises(ValueError, match="exactly levels 0, 4, and 5"):
        score_image_group(
            malformed,
            bank_fixture(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]]
            ),
            "mean_5_euclidean",
        )


def synthetic_scores(
    clean,
    level4,
    level5,
    *,
    family="fog",
    split="selection",
    policy=None,
    confidence=None,
    entropy=None,
    raw_confidence=None,
):
    count = len(clean)
    assert len(level4) == count and len(level5) == count
    confidence = (clean, level4, level5) if confidence is None else confidence
    entropy = (clean, level4, level5) if entropy is None else entropy
    raw_confidence = (
        ([0.8] * count, [0.8] * count, [0.8] * count)
        if raw_confidence is None
        else raw_confidence
    )
    rows = []
    for severity, fingerprint_values, confidence_values, entropy_values, raw_values in zip(
        (0, 4, 5),
        (clean, level4, level5),
        confidence,
        entropy,
        raw_confidence,
        strict=True,
    ):
        for index, (fingerprint, confidence_value, entropy_value, raw_value) in enumerate(
            zip(
                fingerprint_values,
                confidence_values,
                entropy_values,
                raw_values,
                strict=True,
            )
        ):
            row = {
                "image_id": f"image-{index}",
                "family": family,
                "severity": severity,
                "split": split,
                "fingerprint": fingerprint,
                "confidence": confidence_value,
                "entropy": entropy_value,
                "raw_confidence": raw_value,
            }
            if policy is not None:
                row["policy"] = policy
            rows.append(row)
    return rows


def synthetic_candidate_table(*, selection_winner, validation_offset):
    policies = {
        name: study.Policy(name, "mean_5_euclidean", "mean_all", 44)
        for name in ("matched", "background")
    }
    loser = next(name for name in policies if name != selection_winner)
    rows = []
    rows.extend(
        synthetic_scores(
            [0.0, 0.0],
            [2.0, 2.0],
            [2.0, 2.0],
            policy=policies[selection_winner],
        )
    )
    rows.extend(
        synthetic_scores(
            [0.0, 1.0],
            [0.5, 0.5],
            [0.5, 0.5],
            policy=policies[loser],
        )
    )
    winner_validation = -validation_offset
    loser_validation = validation_offset
    rows.extend(
        synthetic_scores(
            [0.0, 0.0],
            [2.0 + winner_validation] * 2,
            [2.0 + winner_validation] * 2,
            split="validation",
            policy=policies[selection_winner],
        )
    )
    rows.extend(
        synthetic_scores(
            [0.0, 1.0],
            [0.5 + loser_validation] * 2,
            [0.5 + loser_validation] * 2,
            split="validation",
            policy=policies[loser],
        )
    )
    return rows


def test_level_four_and_five_are_not_pooled():
    rows = synthetic_scores(clean=[0, 1], level4=[2, 3], level5=[0.5, 1.5])

    result = study.per_family_aurocs(rows, method="fingerprint")["fog"]

    assert (result.level4, result.level5, result.strong) == (1.0, 0.75, 0.875)


def test_auroc_counts_score_ties_as_one_half_and_higher_as_corruption():
    rows = synthetic_scores(clean=[0, 1], level4=[1, 2], level5=[1, 2])

    result = study.per_family_aurocs(rows, method="fingerprint")["fog"]

    assert result.level4 == pytest.approx(0.875)
    assert result.level5 == pytest.approx(0.875)


def test_validation_values_cannot_change_selection():
    first = synthetic_candidate_table(
        selection_winner="matched", validation_offset=0
    )
    changed = synthetic_candidate_table(
        selection_winner="matched", validation_offset=10_000
    )

    assert study.select_policy(first).policy_id == study.select_policy(changed).policy_id


def _policy_rows(policy, family_aurocs):
    rows = []
    score_values = {
        1.0: ([0.0, 1.0], [2.0, 3.0]),
        0.75: ([0.0, 1.0], [0.5, 1.5]),
        0.5: ([0.0, 1.0], [0.5, 0.5]),
    }
    for family, auroc in family_aurocs.items():
        clean, corrupted = score_values[auroc]
        rows.extend(
            synthetic_scores(
                clean,
                corrupted,
                corrupted,
                family=family,
                policy=policy,
            )
        )
    return rows


def test_select_policy_uses_family_median_after_equal_family_mean():
    lower_median = study.Policy("matched", "mean_5_euclidean", "mean_all", 44)
    higher_median = study.Policy(
        "background", "mean_5_euclidean", "mean_all", 44
    )
    rows = _policy_rows(lower_median, {"fog": 1.0, "snow": 0.5, "frost": 0.5})
    rows += _policy_rows(
        higher_median, {"fog": 0.75, "snow": 0.75, "frost": 0.5}
    )

    assert study.select_policy(rows) == higher_median


def test_select_policy_uses_lexicographic_policy_id_as_final_tie_breaker():
    first = study.Policy("background", "mean_5_euclidean", "mean_all", 44)
    second = study.Policy("matched", "mean_5_euclidean", "mean_all", 44)
    rows = _policy_rows(first, {"fog": 0.75})
    rows += _policy_rows(second, {"fog": 0.75})

    assert study.select_policy(rows) == min((first, second), key=lambda item: item.policy_id)


def test_select_policy_rejects_different_image_rosters_between_families():
    policy = study.Policy("matched", "mean_5_euclidean", "mean_all", 44)
    rows = synthetic_scores([0.0, 1.0], [2.0, 3.0], [2.0, 3.0], policy=policy)
    rows += synthetic_scores(
        [0.0, 1.0],
        [2.0, 3.0],
        [2.0, 3.0],
        family="snow",
        policy=policy,
    )
    rows = [
        row
        for row in rows
        if not (row["family"] == "snow" and row["image_id"] == "image-1")
    ]

    with pytest.raises(ValueError, match="same image-ID roster"):
        study.select_policy(rows)


def test_confidence_deciles_accept_selection_rows_only():
    rows = synthetic_scores(
        [0.2, 0.3],
        [0.7, 0.8],
        [0.4, 0.5],
        raw_confidence=([0.2, 0.3], [0.7, 0.8], [0.4, 0.5]),
    )
    expected = (0.25, 0.5, 0.75)

    assert study.confidence_decile_boundaries(rows, severity=4) == pytest.approx(
        expected
    )
    with pytest.raises(ValueError, match="selection-only"):
        study.confidence_decile_boundaries(
            rows + synthetic_scores([0.0], [0.0], [0.0], split="validation"),
            severity=4,
        )


def test_confidence_deciles_use_adjacent_order_statistic_midpoints():
    rows = synthetic_scores(
        [0.0, 2.0, 4.0, 6.0, 8.0],
        [1.0, 3.0, 5.0, 7.0, 9.0],
        [1.0, 3.0, 5.0, 7.0, 9.0],
        raw_confidence=(
            [0.0, 0.2, 0.4, 0.6, 0.8],
            [0.1, 0.3, 0.5, 0.7, 0.9],
            [0.1, 0.3, 0.5, 0.7, 0.9],
        ),
    )

    assert study.confidence_decile_boundaries(rows, severity=4) == pytest.approx(
        tuple(value / 100 for value in range(5, 90, 10))
    )


def test_confidence_deciles_reject_different_image_rosters_between_families():
    rows = synthetic_scores([0.2, 0.3], [0.7, 0.8], [0.4, 0.5])
    rows += synthetic_scores(
        [0.2, 0.3],
        [0.7, 0.8],
        [0.4, 0.5],
        family="snow",
    )
    rows = [
        row
        for row in rows
        if not (row["family"] == "snow" and row["image_id"] == "image-1")
    ]

    with pytest.raises(ValueError, match="same image-ID roster"):
        study.confidence_decile_boundaries(rows, severity=4)


def test_repeated_confidence_quantiles_collapse_and_boundary_uses_upper_stratum():
    fitting_rows = synthetic_scores(
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 1.0],
        raw_confidence=([0.5, 0.5], [0.5, 0.5], [0.5, 0.5]),
    )
    boundaries = study.confidence_decile_boundaries(fitting_rows, severity=4)
    evaluation_rows = synthetic_scores(
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 1.0],
        raw_confidence=([0.5, 0.4], [0.6, 0.5], [0.6, 0.5]),
    )

    result = study.confidence_conditioned_concordance(
        evaluation_rows, family="fog", severity=4, boundaries=boundaries
    )

    assert boundaries == (0.5,)
    assert result.point == pytest.approx(1.0)
    assert result.pair_count == 1


def synthetic_cross_stratum_pair():
    return synthetic_scores(
        [0.0],
        [1.0],
        [1.0],
        raw_confidence=([0.4], [0.6], [0.6]),
    )


def test_empty_confidence_conditioned_task_is_inconclusive():
    result = study.confidence_conditioned_concordance(
        synthetic_cross_stratum_pair(), family="fog", severity=4, boundaries=(0.5,)
    )

    assert result.pair_count == 0
    assert result.point is None


def test_confidence_conditioned_concordance_uses_same_strata_and_half_ties():
    rows = synthetic_scores(
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 2.0],
        [1.0, 1.0, 2.0],
        raw_confidence=(
            [0.2, 0.8, 0.2],
            [0.3, 0.7, 0.8],
            [0.3, 0.7, 0.8],
        ),
    )

    result = study.confidence_conditioned_concordance(
        rows, family="fog", severity=4, boundaries=(0.5,)
    )

    assert result.point == pytest.approx(0.75)
    assert result.pair_count == 2


def test_conditional_bootstrap_weights_tasks_instead_of_pooling_pairs():
    rows = synthetic_scores(
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
        split="validation",
        confidence=([0.0] * 3, [1.0] * 3, [1.0] * 3),
        entropy=([0.0] * 3, [1.0] * 3, [1.0] * 3),
        raw_confidence=(
            [0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2],
            [0.2, 0.8, 0.8],
        ),
    )

    result = study.paired_validation_bootstrap(
        rows,
        boundaries_by_severity={4: (0.5,), 5: (0.5,)},
        samples=20,
        seed=7,
        family="fog",
    )

    assert result.conditional.point == pytest.approx(0.5)
    assert result.conditional.count == 4
    assert (result.conditional.lower, result.conditional.upper) == pytest.approx(
        (0.5, 1.0)
    )


def test_original_conditional_point_is_incomplete_when_one_task_is_empty():
    rows = synthetic_scores(
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 1.0],
        split="validation",
        confidence=([0.0, 0.0], [1.0, 1.0], [1.0, 1.0]),
        entropy=([0.0, 0.0], [1.0, 1.0], [1.0, 1.0]),
        raw_confidence=([0.2, 0.2], [0.2, 0.2], [0.8, 0.8]),
    )

    result = study.paired_validation_bootstrap(
        rows,
        boundaries_by_severity={4: (0.5,), 5: (0.5,)},
        samples=20,
        seed=7,
        family="fog",
    )

    assert result.conditional == study.ScoreInterval(
        point=None,
        lower=None,
        upper=None,
        count=2,
    )


def test_paired_bootstrap_reuses_draws_for_methods_and_differences():
    rows = []
    for family, shift in (("fog", 0.0), ("snow", 0.25)):
        fingerprint = (
            [0.0 + shift, 1.0 + shift, 2.0 + shift],
            [0.5 + shift, 2.0 + shift, 3.0 + shift],
            [0.25 + shift, 1.5 + shift, 4.0 + shift],
        )
        rows.extend(
            synthetic_scores(
                *fingerprint,
                family=family,
                split="validation",
                confidence=fingerprint,
                entropy=(fingerprint[0], fingerprint[2], fingerprint[1]),
                raw_confidence=([0.7] * 3, [0.7] * 3, [0.7] * 3),
            )
        )

    result = study.paired_validation_bootstrap(
        rows,
        boundaries_by_severity={4: (), 5: ()},
        samples=50,
        seed=17,
    )
    repeated = study.paired_validation_bootstrap(
        rows,
        boundaries_by_severity={4: (), 5: ()},
        samples=50,
        seed=17,
    )
    fog_level4 = study.paired_validation_bootstrap(
        rows,
        boundaries_by_severity={4: (), 5: ()},
        samples=50,
        seed=17,
        family="fog",
        severity=4,
    )

    assert result == repeated
    assert result.fingerprint == result.confidence
    assert result.fingerprint_minus_confidence.point == pytest.approx(0.0)
    assert result.fingerprint_minus_confidence.lower == pytest.approx(0.0)
    assert result.fingerprint_minus_confidence.upper == pytest.approx(0.0)
    assert result.fingerprint.count == 3
    assert result.conditional.count == 12
    assert fog_level4.fingerprint.point == pytest.approx(
        study.per_family_aurocs(
            [row for row in rows if row["family"] == "fog"],
            method="fingerprint",
        )["fog"].level4
    )
    assert fog_level4.conditional.count == 3


def test_paired_bootstrap_percentile_interval_has_a_fixed_nonzero_oracle():
    rows = synthetic_scores(
        [0.0, 10.0],
        [1.0, 11.0],
        [1.0, 11.0],
        split="validation",
        confidence=([0.0, 10.0], [-1.0, 9.0], [-1.0, 9.0]),
        entropy=([0.0, 10.0], [-1.0, 9.0], [-1.0, 9.0]),
        raw_confidence=([0.7, 0.7], [0.7, 0.7], [0.7, 0.7]),
    )

    result = study.paired_validation_bootstrap(
        rows,
        boundaries_by_severity={4: (), 5: ()},
        samples=20,
        seed=7,
        family="fog",
        severity=4,
    )

    assert result.fingerprint.point == pytest.approx(0.75)
    assert (result.fingerprint.lower, result.fingerprint.upper) == pytest.approx(
        (0.75, 1.0)
    )
    assert result.fingerprint_minus_confidence.point == pytest.approx(0.5)
    assert (
        result.fingerprint_minus_confidence.lower,
        result.fingerprint_minus_confidence.upper,
    ) == pytest.approx((0.5, 1.0))


def test_seed_summary_is_numeric_and_uses_population_standard_deviation():
    result = study.summarize_seed_scores({42: 0.5, 43: 0.7})

    assert result.mean == pytest.approx(0.6)
    assert result.standard_deviation == pytest.approx(0.1)
    assert result.minimum == pytest.approx(0.5)
    assert result.maximum == pytest.approx(0.7)


def test_per_family_aurocs_rejects_incomplete_groups():
    rows = synthetic_scores([0.0], [1.0], [2.0])

    with pytest.raises(ValueError, match="exactly one score at levels 0, 4, and 5"):
        study.per_family_aurocs(rows[:-1], method="fingerprint")
