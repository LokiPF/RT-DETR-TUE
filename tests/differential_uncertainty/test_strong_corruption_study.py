import pytest
import torch

from differential_uncertainty.strong_corruption_study import (
    AGGREGATIONS,
    BANK_VARIANTS,
    DISTANCES,
    Bank,
    CandidateRows,
    StudyConfig,
    build_bank,
    build_reference_candidates,
    match_reference_queries,
    reservoir_indices,
    split_image_ids,
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
