import numpy as np
import pytest
import torch

import differential_uncertainty.strong_corruption_study as study
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
            "non-zero total",
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


def test_confidence_deciles_accept_selection_rows_only():
    rows = synthetic_scores(
        [0.2, 0.3],
        [0.7, 0.8],
        [0.4, 0.5],
        raw_confidence=([0.2, 0.3], [0.7, 0.8], [0.4, 0.5]),
    )
    expected = tuple(np.quantile([0.2, 0.3, 0.7, 0.8], np.arange(0.1, 1.0, 0.1)))

    assert study.confidence_decile_boundaries(rows, severity=4) == pytest.approx(
        expected
    )
    with pytest.raises(ValueError, match="selection-only"):
        study.confidence_decile_boundaries(
            rows + synthetic_scores([0.0], [0.0], [0.0], split="validation"),
            severity=4,
        )


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
