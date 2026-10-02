import torch
import pytest

from differential_uncertainty.bank import Reservoir


def test_algorithm_r_resume_matches_one_shot_stream_exactly():
    vectors = torch.arange(36, dtype=torch.float32).reshape(12, 3)
    one_shot = Reservoir(capacity=5, dimension=3, seed=44)
    one_shot.add(vectors)
    first_half = Reservoir(capacity=5, dimension=3, seed=44)
    first_half.add(vectors[:6])
    resumed = Reservoir.from_state_dict(first_half.state_dict())
    resumed.add(vectors[6:])
    torch.testing.assert_close(resumed.bank(), one_shot.bank(), rtol=0, atol=0)


def test_bank_filters_padded_queries_and_low_confidence_queries():
    reservoir = Reservoir(capacity=2, dimension=2, seed=44)
    record = {
        "persistence": torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        "logits": torch.tensor([[-2.0, -2.0], [2.0, 0.0], [5.0, 0.0]]),
        "padded_ids": torch.tensor([2]),
    }
    reservoir.add_record(record, threshold=0.5)
    assert reservoir.seen == 1
    assert reservoir.size == 1
    reservoir.add(torch.tensor([[4.0, 4.0]]))
    assert reservoir.bank().dtype == torch.float16
    torch.testing.assert_close(reservoir.bank().float(), torch.tensor([[2.0, 2.0], [4.0, 4.0]]))


def test_reservoir_rejects_finite_float32_vectors_that_overflow_float16():
    with pytest.raises(ValueError, match="finite"):
        Reservoir(capacity=2, dimension=2).add(torch.full((1, 2), 1e10))


def test_reservoir_rejects_nonfinite_restored_vectors():
    reservoir = Reservoir(capacity=2, dimension=2)
    reservoir.add(torch.ones(1, 2))
    state = reservoir.state_dict()
    state["vectors"] = torch.tensor([[float("nan"), 1.0]])
    with pytest.raises(ValueError, match="finite"):
        Reservoir.from_state_dict(state)


@pytest.mark.parametrize(
    ("field", "index", "value"),
    [
        ("logits", 2, torch.tensor([float("nan"), 0.0])),
        ("persistence", 0, torch.tensor([float("inf"), 1.0])),
    ],
)
def test_add_record_rejects_nonfinite_rows_before_padding_or_threshold_masking(field, index, value):
    record = {
        "persistence": torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        "logits": torch.tensor([[-2.0, -2.0], [2.0, 0.0], [5.0, 0.0]]),
        "padded_ids": torch.tensor([2]),
    }
    record[field][index] = value
    with pytest.raises(ValueError, match="finite"):
        Reservoir(capacity=2, dimension=2).add_record(record, threshold=0.5)


@pytest.mark.parametrize(
    "record",
    [
        {"persistence": torch.empty(0, 2), "logits": torch.empty(0, 2), "padded_ids": torch.empty(0, dtype=torch.long)},
        {"persistence": torch.ones(3, 2), "logits": torch.empty(3, 0), "padded_ids": torch.empty(0, dtype=torch.long)},
        {"persistence": torch.ones(3, 3), "logits": torch.ones(3, 2), "padded_ids": torch.empty(0, dtype=torch.long)},
    ],
)
def test_add_record_rejects_empty_rows_zero_classes_and_wrong_persistence_width(record):
    with pytest.raises(ValueError):
        Reservoir(capacity=2, dimension=2).add_record(record, threshold=0.5)


@pytest.mark.parametrize("threshold", [True, float("nan"), float("inf")])
def test_add_record_rejects_boolean_or_nonfinite_thresholds(threshold):
    record = {
        "persistence": torch.ones(3, 2),
        "logits": torch.ones(3, 2),
        "padded_ids": torch.empty(0, dtype=torch.long),
    }
    with pytest.raises(ValueError, match="threshold"):
        Reservoir(capacity=2, dimension=2).add_record(record, threshold=threshold)
