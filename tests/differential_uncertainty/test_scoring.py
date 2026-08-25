import math

import pytest
import torch

from differential_uncertainty.config import ExperimentConfig
from differential_uncertainty.scoring import (
    confidence_deciles,
    confidence_from_logits,
    detect_padded_tail,
    mean_knn_distance,
    relative_gap,
    score_image_records,
    union_padded_query_ids,
)


def _record(image_id="a", severity=0, query_count=20, width=7):
    generator = torch.Generator().manual_seed(100 + severity)
    return {
        "image_id": image_id,
        "severity": severity,
        "boxes": torch.randn(query_count, 4, generator=generator),
        "logits": torch.randn(query_count, 80, generator=generator),
        "persistence": torch.randn(query_count, width, generator=generator),
    }


def test_relative_gap_uses_the_symmetric_scale_independent_formula():
    assert relative_gap(2.0, 6.0) == 1.0
    assert relative_gap(20.0, 60.0) == 1.0
    assert relative_gap(0.0, 0.0) == 0.0
    with pytest.raises(ValueError, match="non-negative"):
        relative_gap(-1.0, 2.0)


def test_exact_repeated_suffix_is_padding_only_when_two_rows_repeat():
    record = _record()
    for field in ("boxes", "logits", "persistence"):
        record[field][-2] = record[field][-1]
    assert detect_padded_tail(record).tolist() == [18, 19]
    record["persistence"][-2, 0] += 1
    assert detect_padded_tail(record).numel() == 0


def test_padding_is_union_of_all_six_severities():
    records = [_record(severity=severity) for severity in range(6)]
    for field in ("boxes", "logits", "persistence"):
        records[1][field][-2] = records[1][field][-1]
        records[5][field][-3:] = records[5][field][-1]
    assert union_padded_query_ids(records).tolist() == [17, 18, 19]


def test_deciles_are_low_first_stable_and_break_ties_by_query_id():
    confidence = torch.tensor([0.5] * 20)
    bins = confidence_deciles(confidence, torch.arange(20))
    assert bins[0].tolist() == [0, 1]
    assert bins[5].tolist() == [10, 11]
    assert bins[9].tolist() == [18, 19]


def test_mean_knn_is_exact_on_small_vectors():
    queries = torch.tensor([[0.0], [4.0]])
    bank = torch.tensor([[0.0], [2.0], [6.0]])
    actual = mean_knn_distance(queries, bank, k=2, bank_chunk_size=2)
    torch.testing.assert_close(actual, torch.tensor([1.0, 2.0]))


def test_confidence_is_maximum_sigmoid_not_softmax():
    logits = torch.tensor([[0.0, math.log(3.0)], [0.0, 0.0]])
    torch.testing.assert_close(confidence_from_logits(logits), torch.tensor([0.75, 0.5]))


def test_one_image_produces_six_complete_persistence_and_confidence_rows():
    config = ExperimentConfig.for_tests(bank_capacity=30, k=2, query_count=20, persistence_dim=7)
    records = [_record(severity=severity) for severity in range(6)]
    bank = torch.randn(30, 7, generator=torch.Generator().manual_seed(3))
    rows = score_image_records(records, bank, config)
    assert [row["severity"] for row in rows] == list(range(6))
    for row in rows:
        assert row["valid_count"] == 20
        assert row["reference_count"] == 2
        assert row["responsive_count"] == 2
        assert -2.0 <= row["persistence_relative_gap"] <= 2.0
        assert -2.0 <= row["confidence_relative_gap"] <= 2.0

def test_direct_confidence_baselines_exclude_the_union_padded_queries():
    config = ExperimentConfig.for_tests(
        bank_capacity=30, k=2, query_count=20, persistence_dim=7
    )
    records = [_record(severity=severity) for severity in range(6)]
    for severity, record in enumerate(records):
        logits = torch.full((20, 80), float(severity - 5))
        logits[:18, 0] = torch.linspace(-4.0, 2.0, 18) + severity
        logits[18:, 0] = 20.0
        record["logits"] = logits
    for field in ("boxes", "logits", "persistence"):
        records[2][field][-2] = records[2][field][-1]

    bank = torch.randn(30, 7, generator=torch.Generator().manual_seed(3))
    rows = score_image_records(records, bank, config)

    valid_ids = torch.arange(18)
    for record, row in zip(records, rows):
        valid_confidence = confidence_from_logits(record["logits"]).index_select(
            0, valid_ids
        )
        assert row["padded_count"] == 2
        assert row["direct_confidence_mean"] == pytest.approx(
            float(valid_confidence.mean())
        )
        assert row["direct_confidence_max"] == pytest.approx(
            float(valid_confidence.max())
        )
        assert isinstance(row["direct_confidence_mean"], float)
        assert isinstance(row["direct_confidence_max"], float)
        assert math.isfinite(row["direct_confidence_mean"])
        assert math.isfinite(row["direct_confidence_max"])


def test_padding_union_rejects_query_count_changes_across_severities():
    records = [_record(severity=0), _record(severity=1, query_count=19)]
    with pytest.raises(ValueError, match="query-count mismatch across severities"):
        union_padded_query_ids(records)


@pytest.mark.parametrize("bad_ids", [torch.ones(10, dtype=torch.bool), torch.arange(10).float()])
def test_deciles_reject_non_integer_query_id_tensors(bad_ids):
    with pytest.raises(ValueError, match="integer query IDs"):
        confidence_deciles(torch.linspace(0.0, 1.0, 20), bad_ids)


@pytest.mark.parametrize("bad_ids", [torch.arange(-1, 9), torch.arange(11) + 10])
def test_deciles_reject_query_ids_outside_the_confidence_vector(bad_ids):
    with pytest.raises(ValueError, match="outside"):
        confidence_deciles(torch.linspace(0.0, 1.0, 20), bad_ids)


def test_confidence_rejects_nonfinite_or_nonfloating_logits():
    with pytest.raises(ValueError, match="finite"):
        confidence_from_logits(torch.tensor([[0.0, float("inf")]]))
    with pytest.raises(ValueError, match="floating-point"):
        confidence_from_logits(torch.tensor([[0, 1]]))


@pytest.mark.parametrize(
    ("queries", "bank", "message"),
    [
        (torch.zeros(2), torch.zeros(3, 1), "two-dimensional"),
        (torch.zeros(2, 1), torch.zeros(3, 2), "feature dimensions"),
        (torch.zeros(2, 1), torch.tensor([[0.0], [float("nan")]]), "finite"),
        (torch.zeros(2, 1, dtype=torch.int64), torch.zeros(3, 1), "floating-point"),
    ],
)
def test_mean_knn_rejects_malformed_vectors(queries, bank, message):
    with pytest.raises(ValueError, match=message):
        mean_knn_distance(queries, bank, k=1)


def test_mean_knn_requires_a_positive_chunk_size():
    with pytest.raises(ValueError, match="chunk size must be positive"):
        mean_knn_distance(torch.zeros(2, 1), torch.zeros(3, 1), k=1, bank_chunk_size=0)


def test_scoring_rejects_record_shapes_that_disagree_with_the_configuration():
    config = ExperimentConfig.for_tests(bank_capacity=30, k=2, query_count=20, persistence_dim=7)
    records = [_record(severity=severity) for severity in range(6)]
    records[4]["persistence"] = torch.randn(20, 6)
    bank = torch.randn(30, 7)
    with pytest.raises(ValueError, match="persistence shape"):
        score_image_records(records, bank, config)


def test_padding_rejects_nonfinite_query_fields():
    record = _record()
    record["boxes"][3, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        detect_padded_tail(record)


@pytest.mark.parametrize(
    ("queries", "bank"),
    [
        (
            torch.ones(1, 335, dtype=torch.float16),
            torch.cat(
                (
                    torch.tensor([[1.0009765625]], dtype=torch.float16),
                    torch.ones(1, 334, dtype=torch.float16),
                ),
                dim=1,
            ),
        ),
        (
            torch.tensor([[10_000.0, 10_000.0]]),
            torch.tensor([[10_001.0, 10_000.0]]),
        ),
    ],
)
def test_mean_knn_matches_a_direct_difference_oracle_without_cancellation(queries, bank):
    expected = (queries.float()[:, None, :] - bank.float()[None, :, :]).norm(dim=-1).mean(dim=1)
    assert bool((expected > 0).all())
    actual = mean_knn_distance(queries, bank, k=1, bank_chunk_size=1)
    torch.testing.assert_close(actual, expected)


def test_scoring_requires_the_complete_configured_bank():
    config = ExperimentConfig.for_tests(bank_capacity=30, k=2, query_count=20, persistence_dim=7)
    records = [_record(severity=severity) for severity in range(6)]
    with pytest.raises(ValueError, match=r"bank must have shape \(30, 7\)"):
        score_image_records(records, torch.randn(29, 7), config)


@pytest.mark.parametrize(("k", "chunk_size"), [(True, 2), (1.5, 2), (1, True), (1, 2.5)])
def test_mean_knn_rejects_bool_or_nonintegral_sizes(k, chunk_size):
    with pytest.raises(ValueError, match="must be an integer"):
        mean_knn_distance(torch.zeros(2, 1), torch.zeros(3, 1), k=k, bank_chunk_size=chunk_size)


def test_confidence_requires_at_least_one_class():
    with pytest.raises(ValueError, match="at least one class"):
        confidence_from_logits(torch.empty(2, 0))


@pytest.mark.parametrize("bad_severity", [True, 1.0, 1.5])
def test_scoring_requires_integral_non_bool_severities(bad_severity):
    config = ExperimentConfig.for_tests(bank_capacity=30, k=2, query_count=20, persistence_dim=7)
    records = [_record(severity=severity) for severity in range(6)]
    records[1]["severity"] = bad_severity
    with pytest.raises(ValueError, match="severity values must be integers"):
        score_image_records(records, torch.randn(30, 7), config)
