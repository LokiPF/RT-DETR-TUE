import pytest
import torch

from src.scene_uncertainty.confidence_deciles import (
    DECILE_NAMES,
    detect_padded_tail,
    union_padded_query_ids,
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
