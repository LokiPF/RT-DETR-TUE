import math

import pytest
import torch

from differential_uncertainty.config import ExperimentConfig
from differential_uncertainty.scoring import (
    mean_five_cosine, normalize_bank, score_triplet, top_query_entropy,
)


def _record(severity, logits, persistence, padded_ids=()):
    return {
        "image_id": "image-1", "corruption": "fog", "severity": severity,
        "logits": torch.tensor(logits, dtype=torch.float32),
        "persistence": torch.tensor(persistence, dtype=torch.float32),
        "padded_ids": torch.tensor(padded_ids, dtype=torch.int64),
    }


def test_mean_five_cosine_uses_exactly_the_five_nearest_bank_rows():
    raw_bank = torch.tensor([
        [1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [-1.0, 0.0],
    ])
    bank = normalize_bank(raw_bank)
    actual = mean_five_cosine(torch.tensor([[1.0, 0.0]]), bank, chunk_size=2)
    assert actual.item() == pytest.approx(1 - bank[:5, 0].mean().item())


def test_score_triplet_uses_union_padding_weighted_fingerprint_and_baselines():
    config = ExperimentConfig(
        class_count=2, query_count=3, persistence_dim=2, bank_capacity=5,
    )
    bank = normalize_bank(torch.tensor([
        [1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4],
    ]))
    records = [
        _record(0, [[0, 0], [2, 0], [5, 0]], [[1, 0], [0, 1], [1, 1]]),
        _record(4, [[0, 0], [2, 0], [5, 0]], [[1, 0], [0, 1], [1, 1]], padded_ids=(2,)),
        _record(5, [[0, 0], [2, 0], [5, 0]], [[1, 0], [0, 1], [1, 1]]),
    ]
    rows = score_triplet(records, bank, config)
    assert [row["severity"] for row in rows] == [0, 4, 5]
    assert set(rows[0]) == {
        "image_id", "corruption", "severity", "fingerprint", "confidence", "entropy",
    }
    distances = mean_five_cosine(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), bank, config.bank_chunk_size)
    weights = torch.sigmoid(torch.tensor([0.0, 2.0]))
    assert rows[0]["fingerprint"] == pytest.approx(float((distances * weights).sum() / weights.sum()))
    assert rows[0]["confidence"] == pytest.approx(1 - float(torch.sigmoid(torch.tensor(2.0))))
    assert rows[0]["entropy"] == pytest.approx(top_query_entropy(torch.tensor([[0.0, 0.0], [2.0, 0.0]])))
    assert all(math.isfinite(value) for row in rows for value in row.values() if isinstance(value, float))


def test_scoring_rejects_zero_norms_and_banks_smaller_than_five():
    with pytest.raises(ValueError, match="zero norm"):
        normalize_bank(torch.zeros(5, 2))
    with pytest.raises(ValueError, match="five"):
        mean_five_cosine(torch.tensor([[1.0, 0.0]]), torch.ones(4, 2))


def test_mean_five_cosine_rejects_empty_queries_and_zero_norm_normalized_bank_rows():
    bank = normalize_bank(torch.tensor([
        [1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4],
    ]))
    with pytest.raises(ValueError, match="nonempty"):
        mean_five_cosine(torch.empty(0, 2), bank)
    bank[2].zero_()
    with pytest.raises(ValueError, match="zero norm"):
        mean_five_cosine(torch.tensor([[1.0, 0.0]]), bank)


def test_top_query_entropy_is_softmax_entropy_normalized_by_class_count():
    assert top_query_entropy(torch.tensor([[0.0, 0.0]])) == pytest.approx(1.0)


def test_scoring_rejects_finite_wide_dtype_values_that_overflow_narrowing_conversions():
    with pytest.raises(ValueError, match="finite"):
        normalize_bank(torch.full((5, 2), 1e300, dtype=torch.float64))
    bank = normalize_bank(torch.tensor([
        [1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4],
    ]))
    with pytest.raises(ValueError, match="finite"):
        mean_five_cosine(torch.full((1, 2), 1e300, dtype=torch.float64), bank)
    with pytest.raises(ValueError, match="finite"):
        top_query_entropy(torch.tensor([[1e300, -1e300]], dtype=torch.float64))


def test_mean_five_cosine_rejects_nonfinite_distances_after_matmul():
    queries = torch.tensor([[1.0, 1.0]])
    bank = torch.full((5, 2), 1e38)
    with pytest.raises(ValueError, match="finite"):
        mean_five_cosine(queries, bank)


def test_top_query_entropy_handles_exact_zero_softmax_probabilities():
    entropy = top_query_entropy(torch.tensor([[1000.0, -1000.0]]))
    assert math.isfinite(entropy)
    assert entropy == pytest.approx(0.0)


@pytest.mark.parametrize(("index", "severity"), [(0, False), (1, 4.0)])
def test_score_triplet_requires_exact_non_boolean_integer_severities(index, severity):
    config = ExperimentConfig(
        class_count=2, query_count=3, persistence_dim=2, bank_capacity=5,
    )
    bank = normalize_bank(torch.tensor([
        [1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4],
    ]))
    records = [
        _record(0, [[0, 0], [2, 0], [5, 0]], [[1, 0], [0, 1], [1, 1]]),
        _record(4, [[0, 0], [2, 0], [5, 0]], [[1, 0], [0, 1], [1, 1]]),
        _record(5, [[0, 0], [2, 0], [5, 0]], [[1, 0], [0, 1], [1, 1]]),
    ]
    records[index]["severity"] = severity
    with pytest.raises(ValueError, match="severities"):
        score_triplet(records, bank, config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("logits", torch.ones(3), "rank-two"),
        ("persistence", torch.ones(3, 3), "persistence shape"),
        ("logits", torch.full((3, 2), float("nan")), "finite"),
    ],
)
def test_score_triplet_rejects_malformed_or_nonfinite_record_tensors(field, value, message):
    config = ExperimentConfig(
        class_count=2, query_count=3, persistence_dim=2, bank_capacity=5,
    )
    bank = normalize_bank(torch.tensor([
        [1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4],
    ]))
    records = [
        _record(0, [[0, 0], [2, 0], [5, 0]], [[1, 0], [0, 1], [1, 1]]),
        _record(4, [[0, 0], [2, 0], [5, 0]], [[1, 0], [0, 1], [1, 1]]),
        _record(5, [[0, 0], [2, 0], [5, 0]], [[1, 0], [0, 1], [1, 1]]),
    ]
    records[1][field] = value
    with pytest.raises(ValueError, match=message):
        score_triplet(records, bank, config)


def test_score_triplet_rejects_empty_queries_after_union_padding():
    config = ExperimentConfig(
        class_count=2, query_count=3, persistence_dim=2, bank_capacity=5,
    )
    bank = normalize_bank(torch.tensor([
        [1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4],
    ]))
    records = [
        _record(level, [[0, 0], [2, 0], [5, 0]], [[1, 0], [0, 1], [1, 1]], padded_ids=(0, 1, 2))
        for level in (0, 4, 5)
    ]
    with pytest.raises(ValueError, match="no valid"):
        score_triplet(records, bank, config)


def test_score_triplet_rejects_zero_norm_normalized_bank_rows():
    config = ExperimentConfig(
        class_count=2, query_count=3, persistence_dim=2, bank_capacity=5,
    )
    bank = torch.tensor([
        [1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.7, 0.3], [0.6, 0.4],
    ])
    records = [
        _record(level, [[0, 0], [2, 0], [5, 0]], [[1, 0], [0, 1], [1, 1]])
        for level in (0, 4, 5)
    ]
    with pytest.raises(ValueError, match="zero norm"):
        score_triplet(records, bank, config)
