import pytest
import torch

from src.scene_uncertainty import confidence_deciles as decile_module
from src.scene_uncertainty.confidence_deciles import (
    DECILE_NAMES,
    QUINTILE_NAMES,
    bin_overlap,
    confidence_buckets,
    confidence_deciles,
    confidence_from_logits,
    detect_padded_tail,
    memberships_by_scheme_severity,
    memberships_by_severity,
    union_padded_query_ids,
    union_query_ids,
)


def record(boxes, logits, layer_0, layer_1, image_id=1, severity=0):
    return {
        "image_id": image_id,
        "severity": severity,
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "logits": torch.tensor(logits, dtype=torch.float16),
        "layers": {
            0: torch.tensor(layer_0, dtype=torch.float16),
            1: torch.tensor(layer_1, dtype=torch.float16),
        },
    }


def test_detects_only_the_repeated_identical_suffix():
    item = record(
        [[0, 0], [1, 1], [9, 9], [9, 9]],
        [[1, 0], [0, 1], [-5, -5], [-5, -5]],
        [[1, 0], [0, 1], [7, 7], [7, 7]],
        [[2, 0], [0, 2], [8, 8], [8, 8]],
    )
    assert detect_padded_tail(item).tolist() == [2, 3]


def test_identical_non_tail_queries_are_retained():
    item = record(
        [[9, 9], [9, 9], [1, 1], [2, 2]],
        [[-5, -5], [-5, -5], [1, 0], [0, 1]],
        [[7, 7], [7, 7], [1, 0], [0, 1]],
        [[8, 8], [8, 8], [2, 0], [0, 2]],
    )
    assert detect_padded_tail(item).numel() == 0


def test_one_final_query_is_not_called_padding():
    item = record(
        [[0, 0], [1, 1], [9, 9]],
        [[1, 0], [0, 1], [-5, -5]],
        [[1, 0], [0, 1], [7, 7]],
        [[2, 0], [0, 2], [8, 8]],
    )
    assert detect_padded_tail(item).numel() == 0


def test_union_mask_is_fixed_across_severities():
    clean = record(
        [[0, 0], [1, 1], [2, 2], [3, 3]],
        [[1, 0], [0, 1], [1, 1], [2, 2]],
        [[1, 0], [0, 1], [1, 1], [2, 2]],
        [[2, 0], [0, 2], [2, 2], [3, 3]], severity=0,
    )
    blurred = record(
        [[0, 0], [1, 1], [9, 9], [9, 9]],
        [[1, 0], [0, 1], [-5, -5], [-5, -5]],
        [[1, 0], [0, 1], [7, 7], [7, 7]],
        [[2, 0], [0, 2], [8, 8], [8, 8]], severity=1,
    )
    assert union_padded_query_ids([clean, blurred]).tolist() == [2, 3]


def record_with_layers(boxes, logits, layers, image_id=1, severity=0):
    """A record whose decoder-layer set is chosen per test, unlike the two-layer helper above."""
    return {
        "image_id": image_id,
        "severity": severity,
        "boxes": torch.as_tensor(boxes, dtype=torch.float32),
        "logits": torch.as_tensor(logits, dtype=torch.float16),
        "layers": {
            layer_id: torch.as_tensor(values, dtype=torch.float16)
            for layer_id, values in layers.items()
        },
    }


def padded_record(query_count, tail_length, image_id=1, severity=0):
    """A record of distinct queries whose final `tail_length` slots are exact copies."""
    boxes = torch.arange(query_count * 4, dtype=torch.float32).reshape(query_count, 4)
    logits = torch.arange(query_count * 3, dtype=torch.float16).reshape(query_count, 3)
    layers = {
        layer_id: torch.arange(query_count * 2, dtype=torch.float16).reshape(query_count, 2) + layer_id
        for layer_id in range(3)
    }
    if tail_length:
        boxes[query_count - tail_length:] = boxes[-1]
        logits[query_count - tail_length:] = logits[-1]
        for values in layers.values():
            values[query_count - tail_length:] = values[-1]
    return record_with_layers(boxes, logits, layers, image_id=image_id, severity=severity)


def test_one_disagreeing_layer_blocks_a_tail_the_other_fields_agree_on():
    fields = {
        "boxes": [[0, 0], [9, 9], [9, 9]],
        "logits": [[1, 0], [-5, -5], [-5, -5]],
        "layers": {0: [[1, 0], [7, 7], [7, 7]], 1: [[2, 0], [8, 8], [8, 8]]},
    }
    def with_layer_two(values):
        return record_with_layers(fields["boxes"], fields["logits"], {**fields["layers"], 2: values})

    assert detect_padded_tail(with_layer_two([[3, 0], [6, 6], [6, 6]])).tolist() == [1, 2]
    assert detect_padded_tail(with_layer_two([[3, 0], [6, 6], [6, 5]])).numel() == 0


def test_a_cache_scale_run_of_ninety_three_queries_is_detected_exactly():
    query_count, tail_start = 300, 207
    generator = torch.Generator().manual_seed(0)
    boxes = torch.rand(query_count, 4, generator=generator)
    logits = torch.rand(query_count, 80, generator=generator).half()
    layers = {
        layer_id: torch.rand(query_count, 335, generator=generator).half()
        for layer_id in range(3)
    }
    boxes[tail_start:] = boxes[-1]
    logits[tail_start:] = logits[-1]
    for values in layers.values():
        values[tail_start:] = values[-1]
    item = record_with_layers(boxes, logits, layers)
    assert detect_padded_tail(item).tolist() == list(range(tail_start, query_count))


def test_a_tail_holding_nan_is_not_identical_to_itself_and_is_not_padding():
    finite = record(
        [[0, 0], [9, 9], [9, 9]],
        [[1, 0], [-5, -5], [-5, -5]],
        [[1, 0], [7, 7], [7, 7]],
        [[2, 0], [8, 8], [8, 8]],
    )
    with_nan = record(
        [[0, 0], [9, 9], [9, 9]],
        [[1, 0], [-5, -5], [-5, -5]],
        [[1, 0], [float("nan"), 7], [float("nan"), 7]],
        [[2, 0], [8, 8], [8, 8]],
    )
    assert detect_padded_tail(finite).tolist() == [1, 2]
    assert detect_padded_tail(with_nan).numel() == 0


def test_a_tail_that_is_near_identical_but_unequal_is_not_padding():
    item = record(
        [[0, 0], [9.0, 9.0], [9.0001, 9.0]],
        [[1, 0], [-5, -5], [-5, -5]],
        [[1, 0], [7, 7], [7, 7]],
        [[2, 0], [8, 8], [8, 8]],
    )
    assert detect_padded_tail(item).numel() == 0


def test_an_earlier_repeated_run_is_not_merged_into_the_final_run():
    item = record(
        [[4, 4], [4, 4], [1, 1], [9, 9], [9, 9]],
        [[-3, -3], [-3, -3], [1, 0], [-5, -5], [-5, -5]],
        [[6, 6], [6, 6], [1, 0], [7, 7], [7, 7]],
        [[5, 5], [5, 5], [2, 0], [8, 8], [8, 8]],
    )
    assert detect_padded_tail(item).tolist() == [3, 4]


def test_a_lone_earlier_copy_of_the_final_query_is_not_pulled_into_the_tail():
    item = record(
        [[9, 9], [1, 1], [9, 9], [9, 9]],
        [[-5, -5], [1, 0], [-5, -5], [-5, -5]],
        [[7, 7], [1, 0], [7, 7], [7, 7]],
        [[8, 8], [2, 0], [8, 8], [8, 8]],
    )
    assert detect_padded_tail(item).tolist() == [2, 3]


def test_a_wholly_identical_record_reports_every_query_as_padded():
    item = record([[9, 9]] * 5, [[-5, -5]] * 5, [[7, 7]] * 5, [[8, 8]] * 5)
    assert detect_padded_tail(item).tolist() == [0, 1, 2, 3, 4]


def test_a_record_with_no_queries_reports_no_padding():
    item = record_with_layers(
        torch.zeros(0, 4), torch.zeros(0, 80), {0: torch.zeros(0, 335), 1: torch.zeros(0, 335)}
    )
    assert detect_padded_tail(item).numel() == 0


def test_padding_masks_are_index_tensors_even_when_empty():
    unpadded = padded_record(10, 0)
    assert detect_padded_tail(unpadded).dtype == torch.long
    assert union_padded_query_ids([unpadded]).dtype == torch.long
    assert union_padded_query_ids([padded_record(10, 3)]).dtype == torch.long


def test_union_over_six_severities_keeps_the_ids_of_the_longest_tail():
    records = [
        padded_record(10, tail_length, severity=severity)
        for severity, tail_length in enumerate((0, 1, 2, 5, 3, 0))
    ]
    assert union_padded_query_ids(records).tolist() == [5, 6, 7, 8, 9]


def test_union_refuses_records_from_more_than_one_image():
    with pytest.raises(ValueError, match="one image"):
        union_padded_query_ids([padded_record(10, 2, image_id=1), padded_record(10, 2, image_id=2)])


def test_union_refuses_an_empty_record_list():
    with pytest.raises(ValueError, match="zero records"):
        union_padded_query_ids([])


def test_union_refuses_severities_with_different_query_counts():
    with pytest.raises(ValueError, match="query-count mismatch across severities"):
        union_padded_query_ids([padded_record(10, 2, severity=0), padded_record(11, 2, severity=1)])


def test_a_record_whose_fields_disagree_on_query_count_is_rejected():
    item = record(
        [[0, 0], [9, 9], [9, 9]],
        [[1, 0], [-5, -5]],
        [[1, 0], [7, 7], [7, 7]],
        [[2, 0], [8, 8], [8, 8]],
    )
    with pytest.raises(ValueError, match="query-count mismatch inside record"):
        detect_padded_tail(item)


def test_a_field_without_a_feature_dimension_is_rejected():
    item = record_with_layers(
        torch.zeros(4, 4), torch.zeros(4, 80), {0: torch.zeros(4, 335), 1: torch.zeros(4)}
    )
    with pytest.raises(ValueError, match="query and feature dimensions"):
        detect_padded_tail(item)


def test_decile_names_are_the_ten_bins_the_spec_names():
    assert DECILE_NAMES == (
        "decile_00_10",
        "decile_10_20",
        "decile_20_30",
        "decile_30_40",
        "decile_40_50",
        "decile_50_60",
        "decile_60_70",
        "decile_70_80",
        "decile_80_90",
        "decile_90_100",
    )


def test_union_query_ids_merges_masks_that_are_not_nested():
    """The wandering case: a later severity pads a region an earlier one did not."""
    wide = torch.arange(141, 300)
    disjoint = torch.arange(0, 8)
    overlapping = torch.arange(100, 179)
    expected = list(range(0, 8)) + list(range(100, 300))
    assert union_query_ids([wide, disjoint, overlapping]).tolist() == expected
    assert union_query_ids([overlapping, wide, disjoint]).tolist() == expected


def test_union_query_ids_sorts_and_deduplicates():
    assert union_query_ids([
        torch.tensor([7, 2, 2]), torch.tensor([2, 5]), torch.tensor([7]),
    ]).tolist() == [2, 5, 7]


def test_union_query_ids_of_one_mask_returns_that_mask_as_an_index_tensor():
    only = union_query_ids([torch.tensor([9, 4], dtype=torch.int32)])
    assert only.tolist() == [4, 9]
    assert only.dtype == torch.long


def test_union_query_ids_of_masks_that_are_all_empty_is_empty():
    nothing = union_query_ids([torch.empty(0, dtype=torch.long)] * 6 + [[]])
    assert nothing.numel() == 0
    assert nothing.dtype == torch.long


def test_union_query_ids_refuses_no_masks_at_all():
    with pytest.raises(ValueError, match="zero masks"):
        union_query_ids([])


def test_union_query_ids_refuses_a_boolean_selection_mask():
    with pytest.raises(ValueError, match="integer query IDs"):
        union_query_ids([torch.tensor([False, True, True])])


def test_the_record_entry_point_delegates_to_the_id_primitive(monkeypatch):
    seen = []

    def spy(id_tensors):
        seen.append([mask.tolist() for mask in id_tensors])
        return torch.tensor([7], dtype=torch.long)

    monkeypatch.setattr(decile_module, "union_query_ids", spy)
    result = union_padded_query_ids([
        padded_record(10, 2, severity=0), padded_record(10, 5, severity=1)
    ])
    assert result.tolist() == [7]
    assert seen == [[[8, 9], [5, 6, 7, 8, 9]]]


def confidence_record(confidence, image_id=None, **extra):
    """A slim severity record of the shape the analysis loader keeps: confidence, no fingerprints."""
    item = {"query_confidence": torch.as_tensor(confidence, dtype=torch.float32), **extra}
    if image_id is not None:
        item["image_id"] = image_id
    return item


NO_PADDING = torch.empty(0, dtype=torch.long)


def as_lists(bins):
    return {name: values.tolist() for name, values in bins.items()}


# --- the brief's tests -------------------------------------------------------------------


def test_confidence_is_largest_sigmoid_class_score():
    actual = confidence_from_logits(torch.tensor([[0.0, 2.0], [-2.0, -1.0]]))
    assert torch.allclose(actual, torch.tensor([2.0, -1.0]).sigmoid())


def test_equal_confidence_ties_follow_query_id():
    bins = confidence_deciles(torch.tensor([0.5] * 20), torch.arange(20))
    assert [values.tolist() for values in bins.values()] == [
        [0, 1], [2, 3], [4, 5], [6, 7], [8, 9],
        [10, 11], [12, 13], [14, 15], [16, 17], [18, 19],
    ]


def test_non_divisible_count_differs_by_at_most_one():
    bins = confidence_deciles(torch.arange(23, dtype=torch.float32), torch.arange(23))
    sizes = [len(values) for values in bins.values()]
    assert sum(sizes) == 23
    assert max(sizes) - min(sizes) == 1


def test_frozen_ids_stay_clean_while_dynamic_ids_move():
    records = {
        0: {"query_confidence": torch.arange(20, dtype=torch.float32)},
        1: {"query_confidence": torch.arange(19, -1, -1, dtype=torch.float32)},
    }
    result = memberships_by_severity(records, torch.empty(0, dtype=torch.long))
    assert result[1]["frozen"]["decile_00_10"].tolist() == [0, 1]
    assert result[1]["dynamic"]["decile_00_10"].tolist() == [19, 18]
    assert result[0]["all_valid"].tolist() == list(range(20))


def test_fewer_than_ten_valid_queries_is_rejected():
    with pytest.raises(ValueError, match="at least ten"):
        confidence_deciles(torch.arange(9, dtype=torch.float32), torch.arange(9))


# --- confidence_from_logits --------------------------------------------------------------


def test_confidence_from_logits_reduces_in_float32_not_float16():
    """The cast the plan writes as `.float()` is what keeps near-tied queries distinguishable."""
    logits = torch.tensor([[-3.191, -8.0], [-3.193, -8.0]], dtype=torch.float16)
    actual = confidence_from_logits(logits)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, logits.float().sigmoid().amax(dim=-1))
    assert not torch.equal(actual, logits.sigmoid().amax(dim=-1).float())


def test_confidence_from_logits_needs_a_class_dimension():
    with pytest.raises(ValueError, match="query, class"):
        confidence_from_logits(torch.zeros(4))
    with pytest.raises(ValueError, match="query, class"):
        confidence_from_logits(torch.zeros(2, 4, 80))


def test_confidence_from_logits_reduces_over_classes_not_over_queries():
    actual = confidence_from_logits(torch.tensor([[-5.0, 1.0, -5.0], [2.0, -5.0, -5.0]]))
    assert actual.tolist() == pytest.approx(torch.tensor([1.0, 2.0]).sigmoid().tolist())


# --- confidence_deciles ------------------------------------------------------------------


def test_bins_run_from_lowest_confidence_to_highest():
    bins = confidence_deciles(torch.arange(30, dtype=torch.float32) / 30, torch.arange(30))
    assert bins["decile_00_10"].tolist() == [0, 1, 2]
    assert bins["decile_90_100"].tolist() == [27, 28, 29]


def test_bins_partition_the_valid_queries_with_no_gap_and_no_repeat():
    generator = torch.Generator().manual_seed(3)
    confidence = torch.rand(300, generator=generator)
    valid = torch.arange(300)[torch.arange(300) % 7 != 0]
    bins = confidence_deciles(confidence, valid)
    assigned = torch.cat(list(bins.values()))
    assert assigned.numel() == valid.numel() == 257
    assert assigned.sort().values.tolist() == valid.tolist()
    sizes = [len(values) for values in bins.values()]
    assert max(sizes) - min(sizes) == 1


def test_bins_never_hold_a_query_outside_the_valid_set():
    valid = torch.arange(0, 40, 2)
    bins = confidence_deciles(torch.arange(40, dtype=torch.float32).flip(0), valid)
    assert set(torch.cat(list(bins.values())).tolist()) == set(valid.tolist())


def test_exactly_ten_valid_queries_gives_ten_bins_of_one():
    bins = confidence_deciles(torch.arange(10, dtype=torch.float32), torch.arange(10))
    assert [values.tolist() for values in bins.values()] == [[index] for index in range(10)]


def test_ties_that_straddle_a_bin_edge_split_by_ascending_query_id():
    """Half the queries share one confidence, so only the tie-break decides five bin edges."""
    confidence = torch.cat([torch.full((15,), 0.25), torch.full((15,), 0.75)])
    bins = confidence_deciles(confidence, torch.arange(30))
    assert bins["decile_40_50"].tolist() == [12, 13, 14]
    assert bins["decile_50_60"].tolist() == [15, 16, 17]


def test_unsorted_valid_indices_bin_identically_to_sorted_ones():
    confidence = torch.tensor([0.5] * 12 + [0.9] * 8)
    shuffled = torch.tensor([13, 2, 19, 0, 7, 15, 4, 11, 1, 18, 6, 9, 3, 16, 12, 5, 17, 8, 14, 10])
    assert as_lists(confidence_deciles(confidence, shuffled)) == as_lists(
        confidence_deciles(confidence, torch.arange(20))
    )


def test_repeated_calls_return_the_same_bins():
    generator = torch.Generator().manual_seed(5)
    confidence = (torch.rand(120, generator=generator) * 8).round() / 8
    valid = torch.arange(120)
    assert as_lists(confidence_deciles(confidence, valid)) == as_lists(
        confidence_deciles(confidence, valid)
    )


def test_duplicate_valid_indices_are_rejected():
    with pytest.raises(ValueError, match="must be unique"):
        confidence_deciles(torch.arange(20, dtype=torch.float32), torch.tensor([0] + list(range(11))))


def test_a_valid_index_outside_the_confidence_vector_is_rejected():
    confidence = torch.arange(20, dtype=torch.float32)
    with pytest.raises(ValueError, match="outside the confidence vector"):
        confidence_deciles(confidence, torch.arange(15, 35))
    with pytest.raises(ValueError, match="outside the confidence vector"):
        confidence_deciles(confidence, torch.arange(-5, 10))


def test_confidence_deciles_refuses_a_boolean_selection_mask():
    """`torch.where(keep)[0]` is the intended input; `keep` itself is zeros and ones."""
    with pytest.raises(ValueError, match="integer query IDs"):
        confidence_deciles(torch.arange(20, dtype=torch.float32), torch.ones(20, dtype=torch.bool))


def test_confidence_deciles_refuses_float_query_indices():
    with pytest.raises(ValueError, match="integer query IDs"):
        confidence_deciles(torch.arange(20, dtype=torch.float32), torch.arange(12, dtype=torch.float32))


def test_confidence_deciles_needs_one_confidence_score_per_query():
    with pytest.raises(ValueError, match="one score per query"):
        confidence_deciles(torch.rand(20, 80), torch.arange(20))


def test_confidence_deciles_refuses_to_rank_a_non_finite_confidence():
    confidence = torch.arange(20, dtype=torch.float32)
    confidence[4] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        confidence_deciles(confidence, torch.arange(20))


def test_the_shortfall_error_names_the_record_it_rejected():
    records = {0: confidence_record(torch.arange(9, dtype=torch.float32), image_id=7888)}
    with pytest.raises(ValueError, match="image 7888 severity 0"):
        memberships_by_severity(records, NO_PADDING)


# --- bin_overlap -------------------------------------------------------------------------


def test_bin_overlap_is_jaccard_per_bin():
    overlap = bin_overlap(
        {"decile_00_10": torch.tensor([1, 2, 3]), "decile_10_20": torch.tensor([4, 5])},
        {"decile_00_10": torch.tensor([3, 2, 9]), "decile_10_20": torch.tensor([5, 4])},
    )
    assert overlap == {"decile_00_10": pytest.approx(0.5), "decile_10_20": 1.0}


def test_bin_overlap_refuses_two_different_bin_sets():
    with pytest.raises(ValueError, match="same bins"):
        bin_overlap({"decile_00_10": torch.tensor([1])}, {"decile_10_20": torch.tensor([1])})


# --- memberships_by_severity -------------------------------------------------------------


def test_membership_recomputes_confidence_and_ignores_the_stored_field():
    """Blur records carry a `confidence` reduced before the fp16 cast; only logits are the source."""
    rising = torch.linspace(-4.0, 4.0, 20).reshape(-1, 1)
    records = {
        0: {"logits": rising, "confidence": rising.flip(0).reshape(-1).sigmoid()},
    }
    result = memberships_by_severity(records, NO_PADDING)
    assert result[0]["dynamic"]["decile_00_10"].tolist() == [0, 1]
    assert result[0]["dynamic"]["decile_90_100"].tolist() == [18, 19]


def test_frozen_membership_is_the_severity_zero_membership_at_every_severity():
    generator = torch.Generator().manual_seed(11)
    records = {
        severity: confidence_record(torch.rand(64, generator=generator))
        for severity in range(6)
    }
    result = memberships_by_severity(records, NO_PADDING)
    clean = as_lists(result[0]["dynamic"])
    for severity in range(6):
        assert as_lists(result[severity]["frozen"]) == clean
    assert any(
        as_lists(result[severity]["dynamic"]) != clean for severity in range(1, 6)
    )


def test_dynamic_overlap_is_one_at_severity_zero_and_falls_only_where_queries_moved():
    rising = torch.arange(20, dtype=torch.float32)
    swapped = rising.clone()
    swapped[1], swapped[2] = rising[2], rising[1]
    records = {0: confidence_record(rising), 1: confidence_record(swapped)}
    result = memberships_by_severity(records, NO_PADDING)
    assert result[0]["dynamic_overlap"] == {name: 1.0 for name in DECILE_NAMES}
    assert result[1]["dynamic"]["decile_00_10"].tolist() == [0, 2]
    assert result[1]["dynamic_overlap"]["decile_00_10"] == pytest.approx(1 / 3)
    assert result[1]["dynamic_overlap"]["decile_10_20"] == pytest.approx(1 / 3)
    assert result[1]["dynamic_overlap"]["decile_90_100"] == 1.0


def test_dynamic_overlap_is_zero_where_a_reversed_rank_empties_every_bin():
    records = {
        0: confidence_record(torch.arange(20, dtype=torch.float32)),
        1: confidence_record(torch.arange(19, -1, -1, dtype=torch.float32)),
    }
    result = memberships_by_severity(records, NO_PADDING)
    assert set(result[1]["dynamic_overlap"].values()) == {0.0}


def test_all_valid_is_exactly_the_union_masked_set():
    padded = torch.tensor([3, 17, 18, 19])
    records = {
        severity: confidence_record(torch.arange(20, dtype=torch.float32))
        for severity in range(2)
    }
    result = memberships_by_severity(records, padded)
    expected = [index for index in range(20) if index not in {3, 17, 18, 19}]
    assert result[0]["all_valid"].tolist() == expected
    assert result[1]["all_valid"].tolist() == expected
    assert sorted(torch.cat(list(result[1]["dynamic"].values())).tolist()) == expected


def test_each_severity_holds_its_own_frozen_and_all_valid_tensors():
    records = {
        severity: confidence_record(torch.arange(20, dtype=torch.float32))
        for severity in range(2)
    }
    result = memberships_by_severity(records, NO_PADDING)
    result[0]["frozen"]["decile_00_10"][0] = 99
    result[0]["all_valid"][0] = 99
    assert result[1]["frozen"]["decile_00_10"][0] == 0
    assert result[1]["all_valid"][0] == 0


def test_memberships_require_severity_zero():
    records = {
        severity: confidence_record(torch.arange(20, dtype=torch.float32))
        for severity in range(1, 6)
    }
    with pytest.raises(ValueError, match="severity zero"):
        memberships_by_severity(records, NO_PADDING)


def test_memberships_reject_a_query_count_mismatch_across_severities():
    records = {
        0: confidence_record(torch.arange(20, dtype=torch.float32)),
        1: confidence_record(torch.arange(19, dtype=torch.float32)),
    }
    with pytest.raises(ValueError, match="query-count mismatch across severities"):
        memberships_by_severity(records, NO_PADDING)


def test_memberships_reject_a_padded_id_outside_the_record():
    records = {0: confidence_record(torch.arange(20, dtype=torch.float32))}
    with pytest.raises(ValueError, match="outside the 20-query record"):
        memberships_by_severity(records, torch.tensor([19, 20]))


def test_memberships_refuse_a_boolean_padding_mask():
    records = {0: confidence_record(torch.arange(20, dtype=torch.float32))}
    with pytest.raises(ValueError, match="integer query IDs"):
        memberships_by_severity(records, torch.zeros(20, dtype=torch.bool))


def test_padding_that_leaves_fewer_than_ten_queries_is_rejected():
    records = {0: confidence_record(torch.arange(20, dtype=torch.float32), image_id=42)}
    with pytest.raises(ValueError, match="at least ten valid queries, got 9"):
        memberships_by_severity(records, torch.arange(9, 20))


# --- directed extra: the index-dtype guard -----------------------------------------------


def test_union_query_ids_refuses_a_uint8_selection_mask():
    with pytest.raises(ValueError, match="integer query IDs"):
        union_query_ids([torch.tensor([0, 1, 1], dtype=torch.uint8)])


def test_a_record_with_neither_logits_nor_confidence_is_rejected():
    with pytest.raises(ValueError, match="logits or a query_confidence vector"):
        memberships_by_severity({0: {"image_id": 3}}, NO_PADDING)


def test_a_record_offering_only_the_caches_stale_confidence_field_is_refused():
    """The key name is the guard: `confidence` is the pre-float16 field, never a bin input."""
    records = {0: {"confidence": torch.arange(20, dtype=torch.float32)}}
    with pytest.raises(ValueError, match="stale `confidence` field"):
        memberships_by_severity(records, NO_PADDING)


def test_logits_still_win_over_a_query_confidence_vector_that_disagrees():
    rising = torch.linspace(-4.0, 4.0, 20).reshape(-1, 1)
    records = {0: {"logits": rising, "query_confidence": rising.flip(0).reshape(-1).sigmoid()}}
    result = memberships_by_severity(records, NO_PADDING)
    assert result[0]["dynamic"]["decile_00_10"].tolist() == [0, 1]


def test_a_non_divisible_split_puts_the_remainder_in_the_lowest_bins():
    """The spread tests cannot see which end the extra queries go to; this pins it."""
    bins = confidence_deciles(torch.arange(23, dtype=torch.float32), torch.arange(23))
    assert [len(values) for values in bins.values()] == [3, 3, 3, 2, 2, 2, 2, 2, 2, 2]
    assert bins["decile_00_10"].tolist() == [0, 1, 2]
    assert bins["decile_20_30"].tolist() == [6, 7, 8]
    assert bins["decile_30_40"].tolist() == [9, 10]
    assert bins["decile_90_100"].tolist() == [21, 22]


def test_query_confidence_is_used_on_a_record_that_also_carries_the_stale_field():
    """The refusal must sit *below* the `query_confidence` lookup, not above it."""
    records = {0: {
        "query_confidence": torch.arange(20, dtype=torch.float32),
        "confidence": torch.arange(19, -1, -1, dtype=torch.float32),
    }}
    result = memberships_by_severity(records, NO_PADDING)
    assert result[0]["dynamic"]["decile_00_10"].tolist() == [0, 1]
    assert result[0]["dynamic"]["decile_90_100"].tolist() == [18, 19]


# --- confidence_buckets: the generic partition -------------------------------------------


def test_quintile_names_are_the_five_bins_the_spec_names():
    assert QUINTILE_NAMES == (
        "quintile_00_20",
        "quintile_20_40",
        "quintile_40_60",
        "quintile_60_80",
        "quintile_80_100",
    )


def test_confidence_quintiles_split_twenty_queries_lowest_first():
    confidence = torch.tensor([
        0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.0,
        0.91, 0.11, 0.81, 0.21, 0.71, 0.31, 0.61, 0.41, 0.51, 0.01,
    ])
    bins = confidence_buckets(
        confidence,
        torch.arange(20),
        names=QUINTILE_NAMES,
        noun="confidence quintiles",
    )

    assert tuple(bins) == QUINTILE_NAMES
    assert [len(bins[name]) for name in QUINTILE_NAMES] == [4, 4, 4, 4, 4]
    assert torch.equal(torch.cat(list(bins.values())).sort().values, torch.arange(20))


def test_confidence_buckets_break_ties_by_ascending_query_id():
    confidence = torch.ones(10)
    bins = confidence_buckets(
        confidence,
        torch.tensor([9, 3, 7, 1, 5, 0, 8, 2, 6, 4]),
        names=QUINTILE_NAMES,
        noun="confidence quintiles",
    )

    assert [values.tolist() for values in bins.values()] == [
        [0, 1], [2, 3], [4, 5], [6, 7], [8, 9],
    ]


def test_legacy_confidence_deciles_match_generic_partition():
    confidence = torch.linspace(0.0, 1.0, 23)
    valid = torch.tensor([22, 0, 5, 12, 3, 7, 18, 2, 9, 14, 4, 8, 17, 6, 1])

    legacy = confidence_deciles(confidence, valid, label="sample")
    generic = confidence_buckets(
        confidence,
        valid,
        names=DECILE_NAMES,
        noun="confidence deciles",
        label="sample",
    )

    assert legacy.keys() == generic.keys()
    assert all(torch.equal(legacy[name], generic[name]) for name in DECILE_NAMES)


def test_five_valid_queries_are_enough_to_fill_the_quintiles():
    """The ten-query floor belongs to the decile scheme, not to the partition primitive."""
    bins = confidence_buckets(
        torch.tensor([0.4, 0.1, 0.5, 0.2, 0.3]),
        torch.arange(5),
        names=QUINTILE_NAMES,
        noun="confidence quintiles",
    )
    assert [values.tolist() for values in bins.values()] == [[1], [3], [4], [0], [2]]


def test_four_valid_queries_are_too_few_for_quintiles():
    message = "confidence quintiles need at least 5 valid queries, got 4 for sample"
    with pytest.raises(ValueError, match=message):
        confidence_buckets(
            torch.arange(4, dtype=torch.float32),
            torch.arange(4),
            names=QUINTILE_NAMES,
            noun="confidence quintiles",
            label="sample",
        )


def test_confidence_buckets_reject_duplicate_valid_indices():
    with pytest.raises(ValueError, match="must be unique"):
        confidence_buckets(
            torch.arange(10, dtype=torch.float32),
            torch.tensor([0, 0, 1, 2, 3, 4]),
            names=QUINTILE_NAMES,
            noun="confidence quintiles",
        )


def test_confidence_buckets_reject_a_valid_index_outside_the_confidence_vector():
    confidence = torch.arange(10, dtype=torch.float32)
    with pytest.raises(ValueError, match="outside the confidence vector"):
        confidence_buckets(
            confidence, torch.arange(6, 12), names=QUINTILE_NAMES, noun="confidence quintiles"
        )
    with pytest.raises(ValueError, match="outside the confidence vector"):
        confidence_buckets(
            confidence, torch.arange(-3, 3), names=QUINTILE_NAMES, noun="confidence quintiles"
        )


def test_confidence_buckets_refuse_to_rank_a_non_finite_confidence():
    confidence = torch.arange(10, dtype=torch.float32)
    confidence[7] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        confidence_buckets(
            confidence, torch.arange(10), names=QUINTILE_NAMES, noun="confidence quintiles"
        )


def test_confidence_buckets_refuse_bucket_names_that_cannot_label_a_partition():
    """Zero names would split nothing; a repeated name would collapse two bins into one key."""
    confidence = torch.arange(10, dtype=torch.float32)
    with pytest.raises(ValueError, match="non-empty and unique"):
        confidence_buckets(confidence, torch.arange(10), names=(), noun="confidence buckets")
    with pytest.raises(ValueError, match="non-empty and unique"):
        confidence_buckets(
            confidence, torch.arange(10), names=("low", "low"), noun="confidence buckets"
        )


def test_quintile_memberships_are_built_for_every_severity():
    generator = torch.Generator().manual_seed(5)
    records = {
        severity: confidence_record(torch.rand(40, generator=generator))
        for severity in range(6)
    }
    result = memberships_by_scheme_severity(
        records,
        torch.tensor([38, 39]),
        names=QUINTILE_NAMES,
        noun="confidence quintiles",
    )

    assert sorted(result) == list(range(6))
    clean = as_lists(result[0]["dynamic"])
    for severity in range(6):
        membership = result[severity]
        assert tuple(membership["dynamic"]) == QUINTILE_NAMES
        assert as_lists(membership["frozen"]) == clean
        assert membership["all_valid"].tolist() == list(range(38))
        assert set(membership["dynamic_overlap"]) == set(QUINTILE_NAMES)
    assert result[0]["dynamic_overlap"] == {name: 1.0 for name in QUINTILE_NAMES}
    assert any(as_lists(result[severity]["dynamic"]) != clean for severity in range(1, 6))


def test_quintiles_and_deciles_cut_one_shared_ordering():
    """Both schemes slice the *same* confidence-ranked sequence; only the cut points differ.

    Twenty-three valid queries is the point of the count: neither scheme divides it evenly, so
    the two remainder distributions genuinely disagree -- 3,3,3,2,... against 5,5,5,4,4 -- and
    the concatenations can only match if one ordering underlies both. Rounding the scores to
    eighths puts real ties in that ordering, so the shared tie-break is under test too, not
    just the shared sort.
    """
    generator = torch.Generator().manual_seed(23)
    confidence = (torch.rand(30, generator=generator) * 8).round() / 8
    valid = torch.arange(7, 30)

    quintiles = confidence_buckets(
        confidence, valid, names=QUINTILE_NAMES, noun="confidence quintiles"
    )
    deciles = confidence_buckets(
        confidence, valid, names=DECILE_NAMES, noun="confidence deciles"
    )

    assert [len(values) for values in deciles.values()] == [3, 3, 3, 2, 2, 2, 2, 2, 2, 2]
    assert [len(values) for values in quintiles.values()] == [5, 5, 5, 4, 4]
    assert torch.equal(
        torch.cat(list(quintiles.values())), torch.cat(list(deciles.values()))
    )
