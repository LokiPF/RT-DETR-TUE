import ast
from pathlib import Path

import pytest
import torch

from src.scene_uncertainty import decile_scoring as scoring_module
from src.scene_uncertainty.decile_scoring import (
    DECILE_AGGREGATIONS,
    PRIMARY_SCORE_SCOPE,
    score_selection,
)


SCALES = {
    0: {"center": torch.tensor(1.0), "scale": torch.tensor(2.0)},
    1: {"center": torch.tensor(2.0), "scale": torch.tensor(4.0)},
}


def unit_scales(layer_ids):
    return {
        int(layer_id): {"center": torch.tensor(0.0), "scale": torch.tensor(1.0)}
        for layer_id in layer_ids
    }


def call(**overrides):
    """A valid three-query call, so each test overrides only the field it is about."""
    kwargs = dict(
        image_id=1,
        severity=0,
        source_partition="tuning",
        membership_mode="dynamic",
        confidence_bin="decile_00_10",
        padding_mode="filtered",
        indices=torch.tensor([0, 2]),
        query_confidence=torch.tensor([0.5, 0.25, 0.75]),
        query_scores_by_layer={0: torch.tensor([1.0, 2.0, 3.0])},
        layer_score_scales=unit_scales([0]),
        clean_overlap=1.0,
        aggregations=("mean",),
    )
    kwargs.update(overrides)
    return score_selection(**kwargs)


def scores_by_key(rows):
    return {(row["signal"], row["score_scope"], row["aggregation"]): row["score"] for row in rows}


# --- the brief's tests -------------------------------------------------------------


def test_matched_signals_use_same_ids_and_mean():
    rows = score_selection(
        image_id=7, severity=2, source_partition="tuning",
        membership_mode="dynamic", confidence_bin="decile_00_10",
        padding_mode="filtered", indices=torch.tensor([1, 3]),
        query_confidence=torch.tensor([0.9, 0.8, 0.7, 0.6]),
        query_scores_by_layer={
            0: torch.tensor([0.0, 3.0, 0.0, 5.0]),
            1: torch.tensor([0.0, 6.0, 0.0, 10.0]),
        },
        layer_score_scales=SCALES, clean_overlap=0.5,
        aggregations=("mean",),
    )
    by_key = {(row["signal"], row["score_scope"]): row for row in rows}
    assert by_key[("confidence", "confidence")]["score"] == pytest.approx(0.3)
    assert by_key[("persistence", "layer_0")]["score"] == 4.0
    assert by_key[("persistence", "layer_1")]["score"] == 8.0
    assert by_key[("persistence", "combined")]["score"] == pytest.approx(1.5)
    assert {tuple(row["selected_query_ids"]) for row in rows} == {(1, 3)}


def test_q90_and_top20_mean_are_applied_to_both_signals():
    rows = score_selection(
        image_id=1, severity=0, source_partition="tuning",
        membership_mode="frozen", confidence_bin="decile_10_20",
        padding_mode="filtered", indices=torch.arange(10),
        query_confidence=torch.linspace(0.1, 1.0, 10),
        query_scores_by_layer={0: torch.arange(10, dtype=torch.float32)},
        layer_score_scales={0: {"center": torch.tensor(0.0), "scale": torch.tensor(1.0)}},
        clean_overlap=1.0, aggregations=("q90", "top20_mean"),
    )
    assert {(row["signal"], row["aggregation"]) for row in rows} == {
        ("confidence", "q90"), ("confidence", "top20_mean"),
        ("persistence", "q90"), ("persistence", "top20_mean"),
    }


def test_empty_selection_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        score_selection(
            image_id=1, severity=0, source_partition="tuning",
            membership_mode="dynamic", confidence_bin="decile_00_10",
            padding_mode="filtered", indices=torch.empty(0, dtype=torch.long),
            query_confidence=torch.ones(3), query_scores_by_layer={0: torch.ones(3)},
            layer_score_scales={0: {"center": torch.tensor(0.0), "scale": torch.tensor(1.0)}},
            clean_overlap=1.0, aggregations=("mean",),
        )


# --- the fairness invariant --------------------------------------------------------


def test_both_signals_score_the_same_ids_through_the_same_summary():
    """The two signals must differ only in what they measure, never in how it is summarised.

    Layer 0 is handed exactly the per-query `1 - confidence` values of the *selected*
    queries and a large decoy everywhere else, so any divergence -- a different selection
    on either side, a different summary on either side, or a signal read over all queries
    instead of the bin -- shows up as a score mismatch under all three summaries.
    """
    indices = torch.tensor([0, 2, 4, 6, 8])
    confidence = torch.zeros(10)
    confidence[indices] = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.1])
    uncertainty = 1.0 - confidence.index_select(0, indices)
    distances = torch.full((10,), 50.0)
    distances[indices] = uncertainty

    rows = call(
        indices=indices,
        query_confidence=confidence,
        query_scores_by_layer={0: distances},
        layer_score_scales=unit_scales([0]),
        aggregations=DECILE_AGGREGATIONS,
    )
    scores = scores_by_key(rows)
    for aggregation in DECILE_AGGREGATIONS:
        assert scores[("confidence", "confidence", aggregation)] == (
            scores[("persistence", "layer_0", aggregation)]
        )
    assert {tuple(row["selected_query_ids"]) for row in rows} == {(0, 2, 4, 6, 8)}
    assert {row["selected_count"] for row in rows} == {5}


def test_unselected_queries_never_reach_either_signal():
    rows = call(
        indices=torch.tensor([1]),
        query_confidence=torch.tensor([0.0, 0.25, 0.0]),
        query_scores_by_layer={0: torch.tensor([100.0, 2.0, 100.0])},
        aggregations=DECILE_AGGREGATIONS,
    )
    scores = scores_by_key(rows)
    for aggregation in DECILE_AGGREGATIONS:
        assert scores[("confidence", "confidence", aggregation)] == 0.75
        assert scores[("persistence", "layer_0", aggregation)] == 2.0


def test_row_selection_lists_are_not_shared_between_rows():
    rows = call(aggregations=DECILE_AGGREGATIONS, query_scores_by_layer={0: torch.ones(3)})
    rows[0]["selected_query_ids"].append(999)
    assert [row["selected_query_ids"] for row in rows[1:]] == [[0, 2]] * (len(rows) - 1)


# --- the confidence control --------------------------------------------------------


def test_confidence_uncertainty_is_exactly_one_minus_confidence():
    for confidence, expected in ((0.0, 1.0), (0.25, 0.75), (0.5, 0.5), (1.0, 0.0)):
        rows = call(
            indices=torch.tensor([1]),
            query_confidence=torch.tensor([0.5, confidence, 0.5]),
            aggregations=DECILE_AGGREGATIONS,
        )
        scores = scores_by_key(rows)
        for aggregation in DECILE_AGGREGATIONS:
            assert scores[("confidence", "confidence", aggregation)] == expected


def test_confidence_is_stored_once_and_never_per_layer():
    rows = call(
        indices=torch.tensor([0, 1, 2]),
        query_scores_by_layer={
            0: torch.tensor([1.0, 2.0, 3.0]),
            1: torch.tensor([1.0, 2.0, 3.0]),
            2: torch.tensor([1.0, 2.0, 3.0]),
        },
        layer_score_scales=unit_scales([0, 1, 2]),
        aggregations=DECILE_AGGREGATIONS,
    )
    confidence_rows = [row for row in rows if row["signal"] == "confidence"]
    assert len(confidence_rows) == len(DECILE_AGGREGATIONS)
    assert {row["score_scope"] for row in confidence_rows} == {"confidence"}
    persistence_scopes = {row["score_scope"] for row in rows if row["signal"] == "persistence"}
    assert persistence_scopes == {"layer_0", "layer_1", "layer_2", "combined"}
    assert PRIMARY_SCORE_SCOPE == "layer_2" and PRIMARY_SCORE_SCOPE in persistence_scopes
    assert len(rows) == len(DECILE_AGGREGATIONS) * (1 + len(persistence_scopes))


def test_confidence_rows_can_be_suppressed_without_moving_persistence():
    with_confidence = call(aggregations=DECILE_AGGREGATIONS)
    without = call(aggregations=DECILE_AGGREGATIONS, include_confidence=False)
    assert not [row for row in without if row["signal"] == "confidence"]
    assert without == [row for row in with_confidence if row["signal"] == "persistence"]


def test_confidence_outside_the_unit_interval_is_rejected():
    with pytest.raises(ValueError, match="confidence"):
        call(query_confidence=torch.tensor([0.5, 0.25, 4.2]), indices=torch.tensor([0, 2]))


def test_non_finite_inputs_are_rejected_rather_than_scored_as_nan():
    with pytest.raises(ValueError, match="confidence"):
        call(query_confidence=torch.tensor([0.5, 0.25, float("nan")]))
    with pytest.raises(ValueError, match="persistence"):
        call(query_scores_by_layer={0: torch.tensor([1.0, 2.0, float("inf")])})


# --- direction ---------------------------------------------------------------------


def test_both_signals_increase_with_uncertainty():
    near = call(
        query_confidence=torch.tensor([0.9, 0.9, 0.9]),
        query_scores_by_layer={0: torch.tensor([1.0, 1.0, 1.0])},
    )
    far = call(
        query_confidence=torch.tensor([0.1, 0.1, 0.1]),
        query_scores_by_layer={0: torch.tensor([9.0, 9.0, 9.0])},
    )
    near_scores, far_scores = scores_by_key(near), scores_by_key(far)
    assert far_scores[("confidence", "confidence", "mean")] > (
        near_scores[("confidence", "confidence", "mean")]
    )
    assert far_scores[("persistence", "layer_0", "mean")] > (
        near_scores[("persistence", "layer_0", "mean")]
    )
    assert far_scores[("persistence", "combined", "mean")] > (
        near_scores[("persistence", "combined", "mean")]
    )


# --- summaries on small selections --------------------------------------------------


def test_top20_mean_of_a_four_query_bin_is_its_largest_value():
    rows = call(
        indices=torch.tensor([0, 1, 2, 3]),
        query_confidence=torch.tensor([0.4, 0.3, 0.2, 0.1]),
        query_scores_by_layer={0: torch.tensor([1.0, 2.0, 3.0, 4.0])},
        aggregations=("top20_mean",),
    )
    scores = scores_by_key(rows)
    assert scores[("persistence", "layer_0", "top20_mean")] == 4.0
    assert scores[("confidence", "confidence", "top20_mean")] == pytest.approx(0.9)


def test_summaries_of_a_single_query_bin_are_that_query():
    rows = call(
        indices=torch.tensor([2]),
        query_confidence=torch.tensor([0.5, 0.5, 0.25]),
        query_scores_by_layer={0: torch.tensor([1.0, 2.0, 7.0])},
        aggregations=DECILE_AGGREGATIONS,
    )
    scores = scores_by_key(rows)
    for aggregation in DECILE_AGGREGATIONS:
        assert scores[("persistence", "layer_0", aggregation)] == 7.0
        assert scores[("confidence", "confidence", aggregation)] == 0.75


# --- provenance and shape -----------------------------------------------------------


def test_every_row_carries_the_selection_provenance():
    rows = call(aggregations=DECILE_AGGREGATIONS, clean_overlap=0.25)
    for row in rows:
        assert row["image_id"] == 1
        assert row["severity"] == 0
        assert row["source_partition"] == "tuning"
        assert row["membership_mode"] == "dynamic"
        assert row["confidence_bin"] == "decile_00_10"
        assert row["padding_mode"] == "filtered"
        assert row["clean_overlap"] == 0.25
        assert isinstance(row["score"], float)


def test_row_keys_within_one_call_are_unique():
    rows = call(
        query_scores_by_layer={0: torch.ones(3), 1: torch.ones(3)},
        layer_score_scales=unit_scales([0, 1]),
        aggregations=DECILE_AGGREGATIONS,
    )
    keys = [(row["signal"], row["score_scope"], row["aggregation"]) for row in rows]
    assert len(set(keys)) == len(keys)


# --- validation ---------------------------------------------------------------------


def test_query_count_disagreement_is_rejected():
    with pytest.raises(ValueError, match="query count"):
        call(query_scores_by_layer={0: torch.ones(4)})


def test_layer_and_scale_sets_must_agree():
    with pytest.raises(ValueError, match="scales"):
        call(
            query_scores_by_layer={0: torch.ones(3), 1: torch.ones(3)},
            layer_score_scales=unit_scales([0]),
        )


def test_a_selection_without_any_persistence_layer_is_rejected():
    with pytest.raises(ValueError, match="layer"):
        call(query_scores_by_layer={}, layer_score_scales={})


def test_unknown_summary_is_rejected():
    with pytest.raises(ValueError, match="summary"):
        call(aggregations=("median",))


def test_repeated_summary_is_rejected():
    with pytest.raises(ValueError, match="summar"):
        call(aggregations=("mean", "mean"))


def test_empty_summary_list_is_rejected():
    with pytest.raises(ValueError, match="summar"):
        call(aggregations=())


def test_a_selection_mask_is_not_mistaken_for_query_ids():
    with pytest.raises(ValueError, match="integer query IDs"):
        call(indices=torch.tensor([True, False, True]))


def test_out_of_range_query_id_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        call(indices=torch.tensor([0, 3]))


def test_repeated_query_id_is_rejected():
    with pytest.raises(ValueError, match="unique"):
        call(indices=torch.tensor([1, 1]))


def test_non_positive_clean_scale_is_rejected():
    with pytest.raises(ValueError, match="scale"):
        call(layer_score_scales={0: {"center": torch.tensor(0.0), "scale": torch.tensor(0.0)}})


# --- no kNN, no inference -----------------------------------------------------------


def test_the_scorer_cannot_reach_a_knn_or_extraction_path():
    """The design's non-goal: consume the *saved* distances, never recompute them."""
    source = Path(scoring_module.__file__).read_text()
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    forbidden = {"knn", "evaluate", "extractor", "bank", "dataset", "pipeline"}
    assert not {name for name in imported if name.lstrip(".").split(".")[0] in forbidden}
    assert not {
        name for name, value in vars(scoring_module).items()
        if getattr(value, "__module__", "").endswith((".knn", ".evaluate", ".extractor"))
    }


# --- label vocabularies -------------------------------------------------------------


def test_unknown_membership_mode_is_rejected():
    with pytest.raises(ValueError, match="membership mode"):
        call(membership_mode="Dynamic")


def test_unknown_padding_mode_is_rejected():
    with pytest.raises(ValueError, match="padding mode"):
        call(padding_mode="unfilterd")


def test_unknown_confidence_bin_is_rejected():
    with pytest.raises(ValueError, match="confidence bin"):
        call(confidence_bin="decile_00_09")


def test_the_all_valid_selection_is_a_recognised_bin():
    rows = call(confidence_bin="all_valid", padding_mode="unfiltered", membership_mode="frozen")
    assert {row["confidence_bin"] for row in rows} == {"all_valid"}
    assert {row["padding_mode"] for row in rows} == {"unfiltered"}
