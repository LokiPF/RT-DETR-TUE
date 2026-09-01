import torch

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
