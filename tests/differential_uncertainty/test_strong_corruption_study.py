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


def test_top_confidence_ties_use_the_lowest_query_id():
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
