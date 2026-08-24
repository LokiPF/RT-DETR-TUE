import tempfile

import pytest
import torch

import differential_uncertainty.bank as bank_module
from differential_uncertainty.artifacts import atomic_torch
from differential_uncertainty.bank import (
    build_reference_bank,
    load_reference_bank,
    save_reference_bank,
)
from differential_uncertainty.config import ExperimentConfig


def _record(image_id, offset):
    persistence = torch.arange(70, dtype=torch.float32).reshape(10, 7) + offset
    boxes = torch.arange(40, dtype=torch.float32).reshape(10, 4) + offset
    logits = torch.arange(800, dtype=torch.float32).reshape(10, 80) + offset
    for field in (persistence, boxes, logits):
        field[-2] = field[-1]
    return {
        "image_id": image_id,
        "severity": 0,
        "persistence": persistence,
        "boxes": boxes,
        "logits": logits,
    }


def test_bank_removes_each_reference_image_padded_tail_and_is_order_invariant():
    config = ExperimentConfig.for_tests(
        bank_capacity=6, k=2, query_count=10, persistence_dim=7
    )
    records = [_record("b", 100), _record("a", 0)]
    first = build_reference_bank(records, config)
    second = build_reference_bank(reversed(records), config)
    assert first.shape == (6, 7)
    assert first.dtype == torch.float32
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not any(
        torch.equal(row, records[0]["persistence"][-1]) for row in first
    )


def test_bank_refuses_fewer_valid_vectors_than_the_fixed_capacity():
    config = ExperimentConfig.for_tests(
        bank_capacity=17, k=2, query_count=10, persistence_dim=7
    )
    with pytest.raises(ValueError, match="needs exactly 17 valid vectors"):
        build_reference_bank([_record("a", 0), _record("b", 100)], config)


@pytest.mark.parametrize("severity", [False, 0.0, 1])
def test_bank_accepts_only_integer_clean_severity_zero(severity):
    config = ExperimentConfig.for_tests(
        bank_capacity=6, k=2, query_count=10, persistence_dim=7
    )
    record = _record("a", 0)
    record["severity"] = severity
    with pytest.raises(ValueError, match="clean severity 0"):
        build_reference_bank([record], config)


@pytest.mark.parametrize(
    ("field", "bad_shape"),
    [
        ("boxes", (10, 3)),
        ("logits", (10, 79)),
        ("persistence", (10, 6)),
    ],
)
def test_bank_requires_exact_configured_record_shapes(field, bad_shape):
    config = ExperimentConfig.for_tests(
        bank_capacity=6, k=2, query_count=10, persistence_dim=7
    )
    record = _record("a", 0)
    record[field] = torch.zeros(bad_shape, dtype=torch.float32)
    with pytest.raises(ValueError, match=f"unexpected {field} shape"):
        build_reference_bank([record], config)


def test_bank_converts_valid_float16_vectors_to_finite_float32():
    config = ExperimentConfig.for_tests(
        bank_capacity=6, k=2, query_count=10, persistence_dim=7
    )
    record = _record("a", 0)
    for field in ("boxes", "logits", "persistence"):
        record[field] = record[field].half()
    bank = build_reference_bank([record], config)
    assert bank.dtype == torch.float32
    assert bank.shape == (6, 7)
    assert bool(torch.isfinite(bank).all())


def test_bank_rejects_nonfinite_persistence_vectors():
    config = ExperimentConfig.for_tests(
        bank_capacity=6, k=2, query_count=10, persistence_dim=7
    )
    record = _record("a", 0)
    record["persistence"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        build_reference_bank([record], config)


def test_bank_rejects_vectors_that_overflow_during_float32_conversion():
    config = ExperimentConfig.for_tests(
        bank_capacity=6, k=2, query_count=10, persistence_dim=7
    )
    record = _record("a", 0)
    record["persistence"] = torch.full(
        (10, 7), 1e300, dtype=torch.float64
    )
    with pytest.raises(ValueError, match="finite after float32 conversion"):
        build_reference_bank([record], config)


@pytest.mark.parametrize("field", ["boxes", "logits", "persistence"])
def test_bank_rejects_non_strided_query_tensors(field):
    config = ExperimentConfig.for_tests(
        bank_capacity=6, k=2, query_count=10, persistence_dim=7
    )
    record = _record("a", 0)
    record[field] = record[field].to_sparse()
    with pytest.raises(ValueError, match="strided layout"):
        build_reference_bank([record], config)


def test_bank_stages_each_one_shot_record_before_advancing_the_input():
    config = ExperimentConfig.for_tests(
        bank_capacity=6, k=2, query_count=10, persistence_dim=7
    )

    def records():
        first = _record("b", 100)
        yield first
        for field in ("boxes", "logits", "persistence"):
            first[field].fill_(float("nan"))
        yield _record("a", 0)

    bank = build_reference_bank(records(), config)
    assert bank[:, 0].tolist() == [49.0, 128.0, 149.0, 21.0, 121.0, 35.0]


@pytest.mark.parametrize("image_ids", [("a", "a"), (1, "1")])
def test_bank_rejects_duplicate_or_string_colliding_image_ids(image_ids):
    config = ExperimentConfig.for_tests(
        bank_capacity=6, k=2, query_count=10, persistence_dim=7
    )
    records = [
        _record(image_ids[0], 0),
        _record(image_ids[1], 100),
    ]
    with pytest.raises(ValueError, match="duplicate or string-colliding"):
        build_reference_bank(records, config)


def test_bank_closes_its_single_temporary_spool_when_input_validation_fails(
    monkeypatch,
):
    config = ExperimentConfig.for_tests(
        bank_capacity=6, k=2, query_count=10, persistence_dim=7
    )
    opened = []
    real_temporary_file = tempfile.TemporaryFile

    def tracked_temporary_file(*args, **kwargs):
        handle = real_temporary_file(*args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(
        bank_module.tempfile,
        "TemporaryFile",
        tracked_temporary_file,
    )
    bad = _record("b", 100)
    bad["severity"] = 1
    with pytest.raises(ValueError, match="clean severity 0"):
        build_reference_bank(iter((_record("a", 0), bad)), config)
    assert len(opened) == 1
    assert opened[0].closed


def test_reservoir_sample_is_fixed_by_the_approved_seed_and_stream_order():
    config = ExperimentConfig.for_tests(
        bank_capacity=6, k=2, query_count=10, persistence_dim=7
    )
    bank = build_reference_bank([_record("b", 100), _record("a", 0)], config)
    assert bank[:, 0].tolist() == [49.0, 128.0, 149.0, 21.0, 121.0, 35.0]


def _saved_bank():
    return torch.arange(28, dtype=torch.float32).reshape(4, 7)


def _bank_metadata():
    return {
        "reference_manifest_sha256": "a" * 64,
        "capacity": 4,
        "seed": 44,
        "padding_removed": True,
    }


def test_saved_bank_round_trip_has_the_exact_plan_schema(tmp_path):
    path = tmp_path / "reference-bank.pt"
    expected_bank = _saved_bank()
    expected_metadata = _bank_metadata()
    save_reference_bank(expected_bank, path, expected_metadata)

    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert type(raw) is dict
    assert set(raw) == {"vectors", "metadata"}
    assert raw["vectors"].shape == (4, 7)
    assert raw["vectors"].dtype == torch.float32
    assert raw["metadata"] == expected_metadata

    actual_bank, actual_metadata = load_reference_bank(path)
    torch.testing.assert_close(actual_bank, expected_bank, rtol=0, atol=0)
    assert actual_metadata == expected_metadata


@pytest.mark.parametrize(
    ("bad_bank", "message"),
    [
        (torch.zeros(4), "two-dimensional"),
        (torch.zeros(4, 7, dtype=torch.float64), "float32"),
        (torch.full((4, 7), float("nan")), "finite"),
        (torch.zeros(0, 7), "at least one row"),
        (torch.zeros(4, 0), "at least one feature"),
        (torch.zeros(4, 7).to_sparse(), "strided layout"),
    ],
)
def test_save_bank_rejects_wrong_rank_dtype_or_nonfinite_vectors(
    tmp_path, bad_bank, message
):
    with pytest.raises(ValueError, match=message):
        save_reference_bank(bad_bank, tmp_path / "bank.pt", _bank_metadata())


def test_save_bank_requires_plain_dict_metadata(tmp_path):
    with pytest.raises(TypeError, match="plain dictionary"):
        save_reference_bank(_saved_bank(), tmp_path / "list.pt", [])


@pytest.mark.filterwarnings("ignore:Sparse invariant checks are implicitly disabled")
@pytest.mark.parametrize(
    ("artifact", "message"),
    [
        (
            {"vectors": _saved_bank(), "metadata": {}, "extra": True},
            "exactly vectors and metadata",
        ),
        ({"vectors": torch.zeros(4), "metadata": {}}, "two-dimensional"),
        (
            {"vectors": torch.zeros(4, 7, dtype=torch.float64), "metadata": {}},
            "float32",
        ),
        (
            {
                "vectors": torch.full((4, 7), float("inf")),
                "metadata": {},
            },
            "finite",
        ),
        ({"vectors": _saved_bank(), "metadata": []}, "plain dictionary"),
        (
            {"vectors": torch.zeros(0, 7), "metadata": _bank_metadata()},
            "at least one row",
        ),
        (
            {"vectors": torch.zeros(4, 0), "metadata": _bank_metadata()},
            "at least one feature",
        ),
        (
            {
                "vectors": torch.zeros(4, 7).to_sparse(),
                "metadata": _bank_metadata(),
            },
            "strided layout",
        ),
    ],
)
def test_load_bank_rejects_malformed_schema_vectors_or_metadata(
    tmp_path, artifact, message
):
    path = tmp_path / "bank.pt"
    atomic_torch(artifact, path)
    with pytest.raises((TypeError, ValueError), match=message):
        load_reference_bank(path)


def _invalid_metadata_cases():
    return [
        (
            {
                key: value
                for key, value in _bank_metadata().items()
                if key != "seed"
            },
            "exact metadata keys",
        ),
        ({**_bank_metadata(), "extra": 1}, "exact metadata keys"),
        (
            {**_bank_metadata(), "reference_manifest_sha256": "A" * 64},
            "lowercase 64-character hexadecimal",
        ),
        (
            {**_bank_metadata(), "reference_manifest_sha256": "a" * 63},
            "lowercase 64-character hexadecimal",
        ),
        (
            {**_bank_metadata(), "capacity": True},
            "capacity must be a positive integer",
        ),
        (
            {**_bank_metadata(), "capacity": 0},
            "capacity must be a positive integer",
        ),
        (
            {**_bank_metadata(), "capacity": 3},
            "capacity must equal",
        ),
        (
            {**_bank_metadata(), "seed": True},
            "seed must be an integer",
        ),
        (
            {**_bank_metadata(), "seed": 1.5},
            "seed must be an integer",
        ),
        (
            {**_bank_metadata(), "padding_removed": False},
            "padding_removed must be true",
        ),
        (
            {**_bank_metadata(), "padding_removed": 1},
            "padding_removed must be true",
        ),
    ]


@pytest.mark.parametrize(("metadata", "message"), _invalid_metadata_cases())
def test_save_bank_rejects_malformed_scientific_metadata(
    tmp_path, metadata, message
):
    with pytest.raises((TypeError, ValueError), match=message):
        save_reference_bank(_saved_bank(), tmp_path / "bank.pt", metadata)


@pytest.mark.parametrize(("metadata", "message"), _invalid_metadata_cases())
def test_load_bank_rejects_malformed_scientific_metadata(
    tmp_path, metadata, message
):
    path = tmp_path / "bank.pt"
    atomic_torch({"vectors": _saved_bank(), "metadata": metadata}, path)
    with pytest.raises((TypeError, ValueError), match=message):
        load_reference_bank(path)


def test_load_bank_rejects_a_symbolic_link(tmp_path):
    target = tmp_path / "target.pt"
    link = tmp_path / "bank.pt"
    save_reference_bank(_saved_bank(), target, _bank_metadata())
    link.symlink_to(target)
    with pytest.raises(ValueError, match="regular file"):
        load_reference_bank(link)
