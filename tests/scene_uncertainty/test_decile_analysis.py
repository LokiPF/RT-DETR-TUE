import ast
import json
from pathlib import Path

import pytest
import torch

from src.scene_uncertainty import decile_analysis as analysis_module
from src.scene_uncertainty.artifacts import load_manifest
from src.scene_uncertainty.confidence_deciles import (
    DECILE_NAMES,
    confidence_from_logits,
    memberships_by_severity,
    union_query_ids,
)
from src.scene_uncertainty.decile_analysis import (
    ALL_QUERY_BENCHMARK,
    ALL_VALID_BENCHMARK,
    RESULT_NON_IDENTIFYING_KEYS,
    ROW_KEYS_EXCLUDED_FROM_CSV,
    SLIM_RECORD_KEYS,
    DecileAnalysisError,
    analyze_deciles,
    load_decile_inputs,
    result_content_address,
)
from src.scene_uncertainty.decile_scoring import DECILE_AGGREGATIONS, PRIMARY_SCORE_SCOPE
from tests.scene_uncertainty.decile_test_utils import (
    QUERY_COUNT,
    SYNTHETIC_BANK_ID,
    mutate_decile_artifacts,
    write_decile_artifacts,
)


@pytest.fixture
def decile_artifacts(tmp_path):
    return write_decile_artifacts(tmp_path)


def load(artifacts):
    return load_decile_inputs(artifacts["cache"], artifacts["results"])


# --- the join --------------------------------------------------------------------------


def test_load_inputs_joins_six_tuning_records_by_key(decile_artifacts):
    loaded = load_decile_inputs(decile_artifacts["cache"], decile_artifacts["results"])
    assert set(loaded.records_by_image) == {11}
    assert set(loaded.records_by_image[11]) == set(range(6))
    assert set(loaded.distances) == {(11, severity, "tuning") for severity in range(6)}
    assert set(loaded.layer_score_scales) == {0, 1, 2}


def test_load_inputs_refuses_another_cache(decile_artifacts):
    path = decile_artifacts["results"].with_suffix(".manifest.json")
    manifest = json.loads(path.read_text())
    manifest["feature_cache_id"] = "another-cache"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DecileAnalysisError, match="feature_cache_id"):
        load_decile_inputs(decile_artifacts["cache"], decile_artifacts["results"])


def test_load_inputs_refuses_test_results(decile_artifacts):
    path = decile_artifacts["results"].with_suffix(".manifest.json")
    manifest = json.loads(path.read_text())
    manifest["source_partition"] = "test"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DecileAnalysisError, match="tuning"):
        load_decile_inputs(decile_artifacts["cache"], decile_artifacts["results"])


@pytest.mark.parametrize("mutation,message", [
    ("drop_distance", "missing"),
    ("duplicate_distance", "duplicate"),
    ("wrong_query_count", "query count"),
    ("drop_severity", "six blur severities"),
])
def test_load_inputs_rejects_misalignment(decile_artifacts, mutation, message):
    mutate_decile_artifacts(decile_artifacts, mutation)
    with pytest.raises(DecileAnalysisError, match=message):
        load_decile_inputs(decile_artifacts["cache"], decile_artifacts["results"])


# --- the rest of the design's validation list ------------------------------------------
#
# Every case below names one bullet of "Validation and error handling". The four cases the
# brief supplies leave most of that list unexercised, and each rule that is not pinned by a
# test is a rule that can be deleted without any test noticing -- which for a loader means
# silently widening what the experiment will score.


@pytest.mark.parametrize("mutation,message", [
    # "artifact provenance does not match"
    ("missing_result_manifest", "missing result manifest"),
    ("missing_cache_manifest", "missing feature-cache manifest"),
    ("drop_layer_score_scales", r"normalizer artifact .* is missing \['layer_score_scales'\]"),
    ("drop_result_cache_id", r"result manifest .* is missing \['feature_cache_id'\]"),
    ("drop_result_k", r"result manifest .* is missing \['k'\]"),
    ("drop_result_bank_id", r"result manifest .* is missing \['bank_id'\]"),
    ("wrong_result_artifact_type", "has artifact_type 'scene_uncertainty_results'"),
    ("forged_result_manifest", "does not match its own content address"),
    ("cache_field_lengths_disagree", r"11, 0, 'tuning'\) is internally inconsistent"),
    ("cache_source_kind", "source_kind"),
    ("cache_manifest_forged", "feature-cache manifest .* does not match its own content address"),
    ("missing_distance_file", "missing query-distance artifact"),
    ("missing_normalizer_file", "missing clean-distance"),
    # "a record key is duplicated or missing from either input"
    ("duplicate_cache_record", "duplicate feature-cache record"),
    ("extra_distance_row", "missing feature-cache record"),
    ("no_tuning_records", "no tuning records"),
    # "an image lacks severity zero or any of the six expected severities"
    ("drop_severity_zero", "six blur severities"),
    # "a record is not in the requested tuning partition". The message fragment is
    # deliberately specific: a loader that *skipped* the foreign row instead of refusing it
    # still raises later, with "missing query-distance record for (11, 0, 'tuning')", and a
    # bare match on "tuning" would call that a pass.
    ("test_partition_row", r"is in the 'test' partition"),
    # "layer IDs or query counts disagree"
    ("cache_layer_missing", "persistence layers"),
    ("distance_layer_missing", "query-distance layers"),
    ("scale_layer_missing", "decoder layers"),
    ("cache_query_count", "query count"),
    # "fewer than ten valid queries remain"
    ("everything_padded", "ten valid queries"),
])
def test_load_inputs_rejects_every_named_validation_failure(decile_artifacts, mutation, message):
    mutate_decile_artifacts(decile_artifacts, mutation)
    with pytest.raises(DecileAnalysisError, match=message):
        load_decile_inputs(decile_artifacts["cache"], decile_artifacts["results"])


@pytest.mark.parametrize(
    "field", ["artifact_id", "feature_cache_id", "bank_id", "k", "query_distance_path"]
)
def test_a_null_result_manifest_field_is_missing_data(decile_artifacts, field):
    """`"feature_cache_id": null` is a key that is present and a value that is not an id."""
    path = decile_artifacts["results"].with_suffix(".manifest.json")
    manifest = json.loads(path.read_text())
    manifest[field] = None
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DecileAnalysisError, match=rf"result manifest .* is missing \['{field}'\]"):
        load(decile_artifacts)


@pytest.mark.parametrize("field", ["artifact_id", "source_kind", "decoder_layers", "query_count"])
def test_a_null_cache_manifest_field_is_missing_data(decile_artifacts, field):
    path = decile_artifacts["cache"] / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[field] = None
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(
        DecileAnalysisError, match=rf"feature-cache manifest .* is missing \['{field}'\]"
    ):
        load(decile_artifacts)


def test_a_missing_cache_id_on_both_sides_is_not_a_match(decile_artifacts):
    """`result.get(...) == cache.get(...)` is True when both are absent, and means nothing.

    The cache manifest is read first, so its own missing id is what gets named; the regex says
    so rather than accepting any message containing "missing", which is the weakness that let
    a filtered-instead-of-refused mutant survive earlier in this task.
    """
    manifest_path = decile_artifacts["cache"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("artifact_id")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    mutate_decile_artifacts(decile_artifacts, "drop_result_cache_id")
    with pytest.raises(
        DecileAnalysisError, match=r"feature-cache manifest .* is missing \['artifact_id'\]"
    ):
        load(decile_artifacts)


def test_a_forged_result_manifest_pointing_at_other_distances_is_refused(decile_artifacts):
    """The reviewer's pilot forgery: same feature_cache_id, another run's distances, new labels.

    Every field this loader *compares* still agrees after the edit -- which is the point. What
    catches it is the manifest hashing its own `query_distance_path`, `k` and `normalization`.
    """
    results = decile_artifacts["results"]
    mutate_decile_artifacts(decile_artifacts, "forged_result_manifest")
    manifest = json.loads(results.with_suffix(".manifest.json").read_text())
    assert (manifest["k"], manifest["normalization"]) == (99, "hand_edited")
    honest = torch.load(
        results.with_suffix(".query_distances.pt"), map_location="cpu", weights_only=True
    )
    forged = torch.load(
        results.parent / manifest["query_distance_path"], map_location="cpu", weights_only=True
    )
    assert not torch.equal(
        honest[0]["query_scores_by_layer"][2], forged[0]["query_scores_by_layer"][2]
    )
    with pytest.raises(DecileAnalysisError, match="does not match its own content address"):
        load(decile_artifacts)


def test_the_result_address_exclusion_matches_the_writer():
    """One rule, two copies, and a test that fails when they drift.

    `decile_analysis` may not import `pipeline` -- that is what
    `test_the_loader_cannot_reach_a_knn_or_extraction_path` enforces -- so the exclusion list
    is restated in the loader. A test carries no such restriction, so the drift is caught here
    instead of by a result manifest that silently stops verifying.
    """
    from src.scene_uncertainty.pipeline import NON_IDENTIFYING_KEYS

    assert set(RESULT_NON_IDENTIFYING_KEYS) == set(NON_IDENTIFYING_KEYS), (
        "pipeline._content_address and decile_analysis.result_content_address must exclude the "
        "same keys, or a manifest the writer sealed will not verify (or worse, will verify "
        "while hiding an edited field)"
    )


def test_the_synthetic_result_manifest_is_sealed_like_a_real_one(decile_artifacts):
    """A fixture with a literal id cannot exercise the check it is supposed to exercise."""
    manifest = json.loads(decile_artifacts["results"].with_suffix(".manifest.json").read_text())
    assert manifest["artifact_id"] == result_content_address(manifest)
    assert manifest["artifact_id"] != "synthetic-results"


# --- the slim record -------------------------------------------------------------------


def test_slim_records_keep_the_key_the_confidence_and_the_mask_and_nothing_else(decile_artifacts):
    """The design's memory contract: persistence tensors are discarded after the suffix check."""
    record = load(decile_artifacts).records_by_image[11][0]
    assert SLIM_RECORD_KEYS == {
        "image_id", "severity", "source_partition",
        "query_confidence", "padded_query_ids", "query_count",
    }
    assert set(record) == SLIM_RECORD_KEYS
    assert record["image_id"] == 11
    assert record["severity"] == 0
    assert record["source_partition"] == "tuning"
    assert record["query_count"] == QUERY_COUNT
    assert record["padded_query_ids"].tolist() == [QUERY_COUNT - 2, QUERY_COUNT - 1]


def test_slim_records_never_carry_the_caches_stale_confidence_field(decile_artifacts):
    """Ruling 7: the derived vector is `query_confidence`; the cached field must not survive."""
    mutate_decile_artifacts(decile_artifacts, "stale_confidence_field")
    record = load(decile_artifacts).records_by_image[11][0]
    assert "confidence" not in record
    assert float(record["query_confidence"].max()) > 0.5


def test_query_confidence_is_recomputed_from_the_cached_float16_logits(decile_artifacts):
    manifest = load_manifest(decile_artifacts["cache"])
    shard = decile_artifacts["cache"] / manifest["shards"][0]
    cached = torch.load(shard, map_location="cpu", weights_only=True)
    expected = confidence_from_logits(next(r for r in cached if r["severity"] == 3)["logits"])
    record = load(decile_artifacts).records_by_image[11][3]
    assert torch.equal(record["query_confidence"], expected)


def test_slim_records_feed_membership_construction_unchanged(decile_artifacts):
    """The handshake with Task 2: only a record keyed `query_confidence` gets past it."""
    records = load(decile_artifacts).records_by_image[11]
    padded = union_query_ids([record["padded_query_ids"] for record in records.values()])
    memberships = memberships_by_severity(records, padded)
    assert set(memberships) == set(range(6))
    assert padded.tolist() == [QUERY_COUNT - 2, QUERY_COUNT - 1]


def test_slim_records_name_their_image_in_a_membership_error(decile_artifacts):
    """Without `image_id` on the slim record every labelled error would read "severity 0"."""
    records = load(decile_artifacts).records_by_image[11]
    with pytest.raises(ValueError, match="image 11 severity 0"):
        memberships_by_severity(records, torch.arange(QUERY_COUNT - 9))


# --- what the loader hands on ----------------------------------------------------------


def test_distances_keep_one_vector_per_layer_per_key(decile_artifacts):
    loaded = load(decile_artifacts)
    for key, by_layer in loaded.distances.items():
        assert set(by_layer) == {0, 1, 2}, key
        assert all(int(values.numel()) == QUERY_COUNT for values in by_layer.values()), key


def test_run_metadata_carries_the_provenance_of_both_inputs(decile_artifacts):
    loaded = load(decile_artifacts)
    metadata = loaded.run_metadata
    assert metadata["artifact_type"] == "confidence_decile_scene_uncertainty"
    assert metadata["feature_cache_id"] == load_manifest(decile_artifacts["cache"])["artifact_id"]
    result_manifest = json.loads(
        decile_artifacts["results"].with_suffix(".manifest.json").read_text()
    )
    assert metadata["source_result_id"] == result_manifest["artifact_id"]
    assert metadata["source_result_id"] == result_content_address(result_manifest)
    assert metadata["source_partition"] == "tuning"
    assert metadata["normalization"] == "raw"
    assert metadata["k"] == 5
    assert metadata["severities"] == [0, 1, 2, 3, 4, 5]
    assert metadata["decoder_layers"] == [0, 1, 2]
    assert metadata["query_count"] == QUERY_COUNT
    assert metadata["image_count"] == 1
    assert metadata["record_count"] == 6


# --- no kNN, no inference --------------------------------------------------------------


def test_the_loader_cannot_reach_a_knn_or_extraction_path():
    """The design's non-goal: consume the *saved* distances, never recompute them."""
    source = Path(analysis_module.__file__).read_text()
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    forbidden = {"knn", "evaluate", "extractor", "bank", "dataset", "pipeline"}
    assert not {name for name in imported if name.lstrip(".").split(".")[0] in forbidden}
    assert not {
        name for name, value in vars(analysis_module).items()
        if getattr(value, "__module__", "").endswith((".knn", ".evaluate", ".extractor"))
    }


# --- the experiment and control rows ----------------------------------------------------
#
# The brief's four orchestration tests, corrected in the ways Task 4 carried forward, plus one
# test per binding sentence of the spec they leave unexercised. Two of them need a fixture the
# plan does not build: the default synthetic image pads the same two queries at every severity
# and shifts every confidence by the same constant, so its padding union is indistinguishable
# from a per-severity mask and its dynamic membership is identical to its frozen membership at
# every severity. `padded_tails` and `rotate_confidence` exist for exactly those two blind
# spots.


ROW_KEYS = (
    "image_id", "severity", "source_partition", "membership_mode", "confidence_bin",
    "padding_mode", "signal", "score_scope", "aggregation",
)
"""The identity of a result row, as Task 6 groups it. Duplicates here are the design's
"output result keys would be duplicated" failure reached from the producing side."""

WANDERING_TAILS = {0: 2, 1: 5, 2: 3, 3: 2, 4: 4, 5: 2}
"""Padding that moves rather than grows, as it does on 66 of 66 padded pilot images."""


def row_key(row):
    return tuple(row[key] for key in ROW_KEYS)


def labels_of(rows):
    return {(row["membership_mode"], row["confidence_bin"], row["padding_mode"]) for row in rows}


def test_analysis_emits_deciles_and_controls(decile_artifacts):
    rows, _ = analyze_deciles(load_decile_inputs(
        decile_artifacts["cache"], decile_artifacts["results"]
    ))
    labels = labels_of(rows)
    assert {("dynamic", name, "filtered") for name in DECILE_NAMES} <= labels
    assert {("frozen", name, "filtered") for name in DECILE_NAMES} <= labels
    assert ("shared", "all_valid", "filtered") in labels
    assert ALL_QUERY_BENCHMARK in labels
    assert ("dynamic", "decile_00_10", "unfiltered") in labels
    assert ("frozen", "decile_00_10", "unfiltered") in labels


def test_frozen_ids_stay_fixed_and_dynamic_overlap_is_bounded(decile_artifacts):
    rows, _ = analyze_deciles(load_decile_inputs(
        decile_artifacts["cache"], decile_artifacts["results"]
    ))
    selected = {
        (r["severity"], r["membership_mode"]): r["selected_query_ids"]
        for r in rows
        if r["confidence_bin"] == "decile_00_10" and r["padding_mode"] == "filtered"
        and r["signal"] == "confidence" and r["aggregation"] == "mean"
    }
    assert selected[(0, "frozen")] == selected[(5, "frozen")]
    assert all(0.0 <= row["clean_overlap"] <= 1.0 for row in rows)


def test_padding_control_is_the_only_bottom_bin_that_keeps_padded_ids(decile_artifacts):
    rows, diagnostics = analyze_deciles(load_decile_inputs(
        decile_artifacts["cache"], decile_artifacts["results"]
    ))
    padded = set(diagnostics["images"]["11"]["union_padded_query_ids"])
    assert padded
    assert all(
        not padded.intersection(row["selected_query_ids"])
        for row in rows if row["padding_mode"] == "filtered"
    )
    assert any(
        padded.intersection(row["selected_query_ids"])
        for row in rows
        if row["padding_mode"] == "unfiltered" and row["confidence_bin"] == "decile_00_10"
    )


def test_the_all_query_benchmark_is_persistence_q90_only(decile_artifacts):
    """Spec benchmark 1 is one published number -- q90, layer 2 -- not a fourth grid.

    It is also the one selection whose `1 - confidence` control would be meaningless: the
    padded queries it deliberately keeps sit at near-zero confidence, so a confidence row here
    would summarise placeholders rather than the scene. `include_confidence=False` is why no
    confidence row exists for this label.
    """
    rows, _ = analyze_deciles(load_decile_inputs(
        decile_artifacts["cache"], decile_artifacts["results"]
    ))
    benchmark = [
        row for row in rows
        if (row["membership_mode"], row["confidence_bin"], row["padding_mode"])
        == ALL_QUERY_BENCHMARK
    ]
    assert benchmark
    assert {row["signal"] for row in benchmark} == {"persistence"}
    assert {row["aggregation"] for row in benchmark} == {"q90"}


# --- the spec's five benchmarks (127-138) ----------------------------------------------


def test_every_spec_benchmark_row_exists_and_is_distinguishable(decile_artifacts):
    """All five report rows the spec requires, each reachable by its own label triple.

    Kills a build that emits the all-300 benchmark and the `all_valid` result under one label
    (the two would then differ only by which rows happened to be written last), and a build
    that drops the confidence control for any bin or membership mode.
    """
    rows, _ = analyze_deciles(load_decile_inputs(
        decile_artifacts["cache"], decile_artifacts["results"]
    ))
    index = {}
    for row in rows:
        index.setdefault(
            (row["membership_mode"], row["confidence_bin"], row["padding_mode"]), []
        ).append(row)

    # 1. the existing all-300-query persistence benchmark: q90, layer 2
    all_query = index[ALL_QUERY_BENCHMARK]
    assert any(
        row["signal"] == "persistence" and row["score_scope"] == PRIMARY_SCORE_SCOPE
        and row["aggregation"] == "q90"
        for row in all_query
    )
    # 2. the new all_valid persistence result, showing the effect of removing padding
    all_valid = index[ALL_VALID_BENCHMARK]
    assert {
        row["aggregation"] for row in all_valid
        if row["signal"] == "persistence" and row["score_scope"] == PRIMARY_SCORE_SCOPE
    } == set(DECILE_AGGREGATIONS)
    # 1 and 2 are the same queries minus the padding, and must not be one label
    assert ALL_QUERY_BENCHMARK != ALL_VALID_BENCHMARK
    assert {row["selected_count"] for row in all_query} != {row["selected_count"] for row in all_valid}
    # 3. all_valid confidence-only uncertainty with the same scene summaries
    assert {
        row["aggregation"] for row in all_valid if row["signal"] == "confidence"
    } == set(DECILE_AGGREGATIONS)
    # 4. the matched confidence-only result for every confidence bin and membership mode
    for mode in ("dynamic", "frozen"):
        for name in DECILE_NAMES:
            matched = index[(mode, name, "filtered")]
            assert {
                row["aggregation"] for row in matched if row["signal"] == "confidence"
            } == set(DECILE_AGGREGATIONS), (mode, name)
    # 5. filtered versus unfiltered results for the lowest-confidence bin
    for mode in ("dynamic", "frozen"):
        for padding in ("filtered", "unfiltered"):
            assert index[(mode, "decile_00_10", padding)], (mode, padding)


def test_the_unfiltered_control_is_scoped_to_the_lowest_confidence_bin(decile_artifacts):
    """Spec 69: the sensitivity control repeats *only* the lowest-confidence-bin analysis.

    Kills a control that widens to every decile -- which would double the decile grid, and
    would read in the report as nine extra padding measurements the spec never asked for.
    `all_valid`/`unfiltered` appears here too and is not the control: it is spec benchmark 1,
    the all-300-query result, and the assertion names it rather than admitting any bin.
    """
    rows, _ = analyze_deciles(load_decile_inputs(
        decile_artifacts["cache"], decile_artifacts["results"]
    ))
    unfiltered = {
        (row["membership_mode"], row["confidence_bin"]) for row in rows
        if row["padding_mode"] == "unfiltered"
    }
    assert unfiltered == {
        ("dynamic", "decile_00_10"), ("frozen", "decile_00_10"),
        (ALL_QUERY_BENCHMARK[0], ALL_QUERY_BENCHMARK[1]),
    }


def test_the_all_query_benchmark_really_selects_every_query(decile_artifacts):
    """Benchmark 1 is the *all-300* result: every query, padded ones included.

    Kills a build that files the union-masked valid set under the unfiltered label, which
    would silently make the "before padding removal" benchmark identical to the "after".
    """
    inputs = load_decile_inputs(decile_artifacts["cache"], decile_artifacts["results"])
    rows, diagnostics = analyze_deciles(inputs)
    padded = set(diagnostics["images"]["11"]["union_padded_query_ids"])
    benchmark = [
        row for row in rows
        if (row["membership_mode"], row["confidence_bin"], row["padding_mode"])
        == ALL_QUERY_BENCHMARK
    ]
    for row in benchmark:
        assert row["selected_count"] == QUERY_COUNT
        assert set(row["selected_query_ids"]) == set(range(QUERY_COUNT))
        assert padded.issubset(row["selected_query_ids"])


# --- one row, one key (spec 194) --------------------------------------------------------


def test_every_output_row_key_is_unique(decile_artifacts):
    """Spec 194: duplicated output result keys are a failure, not a last-write-wins merge.

    Task 6 refuses a duplicate it is handed; this asserts the producer never hands one over.
    Kills a control emitted under the wrong `padding_mode` or `membership_mode`, which is the
    realistic way this table grows a collision -- the rows would be real, the scores different,
    and the group they collapsed into would average two different populations.
    """
    rows, _ = analyze_deciles(load_decile_inputs(
        decile_artifacts["cache"], decile_artifacts["results"]
    ))
    keys = [row_key(row) for row in rows]
    assert len(set(keys)) == len(keys)


def test_every_row_carries_the_keys_the_csv_writer_has_to_drop(decile_artifacts):
    """`per_scene.csv` must not carry `selected_query_ids`; the rows must, so it can be dropped.

    A row missing the key would make the writer's `drop(columns=...)` raise on a full run and
    pass on a fixture, so the contract is asserted here rather than discovered in Task 7.
    """
    rows, _ = analyze_deciles(load_decile_inputs(
        decile_artifacts["cache"], decile_artifacts["results"]
    ))
    assert ROW_KEYS_EXCLUDED_FROM_CSV == ("selected_query_ids",)
    for key in ROW_KEYS_EXCLUDED_FROM_CSV:
        assert all(key in row for row in rows)


# --- dynamic against frozen, on a fixture whose ranking actually moves -------------------


@pytest.fixture
def moving_rank_artifacts(tmp_path):
    return write_decile_artifacts(tmp_path, rotate_confidence=3)


def test_dynamic_and_frozen_stay_separable_when_the_ranking_moves(moving_rank_artifacts):
    """Spec 231: the two answer different questions and must not be combined into one score.

    Kills a build that labels both memberships `dynamic` (or both `frozen`): the two rows would
    then share a row key, and Task 6 would average a moving selection with a fixed one. The
    default fixture cannot see this, because its confidence shift is rank-preserving and its
    dynamic bins equal its frozen bins at every severity.
    """
    rows, _ = analyze_deciles(load_decile_inputs(
        moving_rank_artifacts["cache"], moving_rank_artifacts["results"]
    ))
    selected = {
        (row["severity"], row["membership_mode"]): tuple(row["selected_query_ids"])
        for row in rows
        if row["confidence_bin"] == "decile_00_10" and row["padding_mode"] == "filtered"
        and row["signal"] == "confidence" and row["aggregation"] == "mean"
    }
    assert selected[(0, "dynamic")] == selected[(0, "frozen")]
    assert all(selected[(severity, "frozen")] == selected[(0, "frozen")] for severity in range(6))
    assert any(selected[(severity, "dynamic")] != selected[(0, "frozen")] for severity in range(1, 6))


def test_clean_overlap_is_the_dynamic_jaccard_against_the_severity_zero_membership(
    moving_rank_artifacts,
):
    """Spec 97: report Jaccard overlap between each dynamic bin and its severity-zero self.

    Recomputed here straight from `memberships_by_severity` rather than trusted. Kills a build
    that files the frozen 1.0 on dynamic rows, one that files another bin's overlap, and one
    that hard-codes 1.0 -- the last is why the fixture rotates the ranking and why this asserts
    a value strictly below 1.0 exists.
    """
    inputs = load_decile_inputs(
        moving_rank_artifacts["cache"], moving_rank_artifacts["results"]
    )
    rows, _ = analyze_deciles(inputs)
    records = inputs.records_by_image[11]
    padded = union_query_ids([record["padded_query_ids"] for record in records.values()])
    memberships = memberships_by_severity(records, padded)
    seen = set()
    for row in rows:
        if row["padding_mode"] != "filtered" or row["confidence_bin"] not in DECILE_NAMES:
            continue
        severity, name, mode = row["severity"], row["confidence_bin"], row["membership_mode"]
        if mode == "dynamic":
            assert row["clean_overlap"] == pytest.approx(
                memberships[severity]["dynamic_overlap"][name]
            ), (severity, name)
            seen.add(row["clean_overlap"])
        else:
            assert row["clean_overlap"] == 1.0, (severity, name)
    assert any(value < 1.0 for value in seen)
    assert all(
        row["clean_overlap"] == 1.0 for row in rows
        if row["severity"] == 0 and row["padding_mode"] == "filtered"
    )


def test_bins_and_scores_come_from_the_same_severitys_confidence_vector(moving_rank_artifacts):
    """The carry `score_selection` structurally cannot check: one vector builds and scores.

    Two halves. The score half recomputes `mean(1 - confidence)` from the record's own
    `query_confidence` over the IDs the row records, so scoring severity s with severity 0's
    vector is caught. The membership half asserts the dynamic bottom bin really is that
    severity's least-confident valid queries, so *building* the dynamic bins from severity
    zero's vector is caught -- a mutation that would otherwise be invisible, because it merely
    turns dynamic into a second copy of frozen.
    """
    inputs = load_decile_inputs(
        moving_rank_artifacts["cache"], moving_rank_artifacts["results"]
    )
    rows, _ = analyze_deciles(inputs)
    for row in rows:
        if row["signal"] != "confidence" or row["aggregation"] != "mean":
            continue
        confidence = inputs.records_by_image[11][row["severity"]]["query_confidence"]
        ids = torch.tensor(row["selected_query_ids"], dtype=torch.long)
        expected = float((1.0 - confidence.float().index_select(0, ids)).mean())
        assert row["score"] == pytest.approx(expected, abs=1e-6)

    bottom = {
        row["severity"]: row["selected_query_ids"] for row in rows
        if row["membership_mode"] == "dynamic" and row["confidence_bin"] == "decile_00_10"
        and row["padding_mode"] == "filtered" and row["signal"] == "confidence"
        and row["aggregation"] == "mean"
    }
    second = {
        row["severity"]: row["selected_query_ids"] for row in rows
        if row["membership_mode"] == "dynamic" and row["confidence_bin"] == "decile_10_20"
        and row["padding_mode"] == "filtered" and row["signal"] == "confidence"
        and row["aggregation"] == "mean"
    }
    for severity, ids in bottom.items():
        confidence = inputs.records_by_image[11][severity]["query_confidence"].float()
        assert float(confidence[ids].max()) <= float(confidence[second[severity]].min()), severity


# --- the padded tail wanders, and the union is what holds the population still -----------


@pytest.fixture
def wandering_tail_artifacts(tmp_path):
    return write_decile_artifacts(tmp_path, padded_tails=WANDERING_TAILS)


def test_diagnostics_record_the_padding_facts_the_spec_names(wandering_tail_artifacts):
    """Spec 67: padded count by image and severity, the union count, and tail agreement.

    Every number here differs from the ones a per-severity or severity-zero-only detector would
    produce, so this kills a union computed from one severity, a count taken from the longest
    tail rather than the union, and a `tail_identical_across_severities` hard-coded either way.
    """
    _, diagnostics = analyze_deciles(load_decile_inputs(
        wandering_tail_artifacts["cache"], wandering_tail_artifacts["results"]
    ))
    image = diagnostics["images"]["11"]
    assert set(diagnostics["images"]) == {"11"}
    assert image["padded_count_by_severity"] == {
        str(severity): length for severity, length in WANDERING_TAILS.items()
    }
    assert image["padded_query_ids_by_severity"]["1"] == list(range(15, 20))
    assert image["padded_query_ids_by_severity"]["0"] == [18, 19]
    assert image["union_padded_query_ids"] == list(range(15, 20))
    assert image["union_padded_count"] == 5
    assert image["tail_identical_across_severities"] is False


def test_an_unchanging_tail_is_recorded_as_identical(decile_artifacts):
    """The other half of the same flag: the default image pads the same two queries throughout."""
    _, diagnostics = analyze_deciles(load_decile_inputs(
        decile_artifacts["cache"], decile_artifacts["results"]
    ))
    image = diagnostics["images"]["11"]
    assert image["tail_identical_across_severities"] is True
    assert image["union_padded_count"] == 2
    assert set(image["padded_count_by_severity"].values()) == {2}


def test_the_union_mask_holds_the_valid_population_still(wandering_tail_artifacts):
    """Spec 65: one stable valid-query mask for all severities, so validity cannot move bins.

    Kills per-severity masking, which here would swing the valid population between 15 and 18
    across the six severities -- the miniature of image 7888 swinging 141 to 292 on the pilot.
    """
    inputs = load_decile_inputs(
        wandering_tail_artifacts["cache"], wandering_tail_artifacts["results"]
    )
    rows, diagnostics = analyze_deciles(inputs)
    per_severity = {
        len(ids) for ids in diagnostics["images"]["11"]["padded_query_ids_by_severity"].values()
    }
    assert len(per_severity) > 1, "the fixture must actually wander for this to prove anything"

    valid_counts = {
        row["severity"]: row["selected_count"] for row in rows
        if (row["membership_mode"], row["confidence_bin"], row["padding_mode"])
        == ALL_VALID_BENCHMARK
    }
    assert set(valid_counts) == set(range(6))
    assert set(valid_counts.values()) == {QUERY_COUNT - 5}

    every_padded_id = {
        int(query_id)
        for ids in diagnostics["images"]["11"]["padded_query_ids_by_severity"].values()
        for query_id in ids
    }
    assert all(
        not every_padded_id.intersection(row["selected_query_ids"])
        for row in rows if row["padding_mode"] == "filtered"
    )


# --- bank provenance (Task 4 carry d) ---------------------------------------------------


def test_run_metadata_names_the_clean_bank_the_distances_came_from(decile_artifacts):
    """Two sanctioned pilot result sets share a feature cache and differ only in their bank.

    `raw_k5` was built against bank 353a5992... and `raw_k5_bank50k` against 5dfd6e50..., from
    the same cache df79fddf.... Without `bank_id` travelling into `run_metadata`, the two runs
    write byte-identical provenance into `summary.json` and nothing in the analysis directory
    records which clean bank the distances were measured against.
    """
    loaded = load(decile_artifacts)
    result_manifest = json.loads(
        decile_artifacts["results"].with_suffix(".manifest.json").read_text()
    )
    assert loaded.run_metadata["bank_id"] == result_manifest["bank_id"]
    assert loaded.run_metadata["bank_id"] == SYNTHETIC_BANK_ID


def test_the_padding_union_and_not_one_severity_decides_the_ten_valid_query_floor(tmp_path):
    """Two spec rules only mean something together: union across severities, then the floor.

    Severity zero pads two queries and severity one pads twelve, so a mask taken from any
    single severity leaves eighteen or eight valid queries depending which one is asked, and
    only the union leaves eight every time. Kills a loader that checks the floor against one
    severity's tail: it would admit an image that cannot fill ten bins and hand it to
    `confidence_deciles`, which raises much later, from inside the scoring loop, about a bin.
    """
    artifacts = write_decile_artifacts(tmp_path, padded_tails={1: 12})
    with pytest.raises(DecileAnalysisError, match="ten valid queries"):
        load(artifacts)


def test_the_unfiltered_control_freezes_and_moves_like_the_filtered_bins(wandering_tail_artifacts):
    """The control is run under both membership modes, and the two must not be the same list.

    On this image severity zero pads two queries and severity one pads five, so the three
    queries in between are ordinary at severity zero and near-zero-confidence placeholders at
    severity one -- which is exactly how the unfiltered bottom bin moves in reality. Kills a
    build that hands the severity-zero unfiltered bins to the dynamic row (the control would
    then be two copies of one measurement) and one that hands each severity's own bins to the
    frozen row (nothing would be frozen).
    """
    rows, _ = analyze_deciles(load_decile_inputs(
        wandering_tail_artifacts["cache"], wandering_tail_artifacts["results"]
    ))
    selected = {
        (row["severity"], row["membership_mode"]): tuple(row["selected_query_ids"])
        for row in rows
        if row["padding_mode"] == "unfiltered" and row["confidence_bin"] == "decile_00_10"
        and row["signal"] == "confidence" and row["aggregation"] == "mean"
    }
    assert set(selected) == {(severity, mode) for severity in range(6) for mode in ("dynamic", "frozen")}
    assert selected[(0, "dynamic")] == selected[(0, "frozen")]
    assert all(selected[(severity, "frozen")] == selected[(0, "frozen")] for severity in range(6))
    assert any(selected[(severity, "dynamic")] != selected[(0, "dynamic")] for severity in range(1, 6))


def test_the_unfiltered_control_overlaps_against_its_own_unfiltered_clean_bins(
    wandering_tail_artifacts,
):
    """The control's `clean_overlap` reference is the all-300 severity-zero bin, not the masked one.

    Recomputed here from the ID lists the rows themselves record, so it does not depend on which
    reference the module chose. Kills a build that overlaps the unfiltered bottom bin against the
    *filtered* severity-zero bin, which would report bin movement that is really the padding mask
    -- the one number in this table whose whole job is to separate those two things.
    """
    rows, _ = analyze_deciles(load_decile_inputs(
        wandering_tail_artifacts["cache"], wandering_tail_artifacts["results"]
    ))
    control = [
        row for row in rows
        if row["padding_mode"] == "unfiltered" and row["confidence_bin"] == "decile_00_10"
    ]
    reference = {
        tuple(row["selected_query_ids"]) for row in control
        if row["severity"] == 0 and row["membership_mode"] == "frozen"
    }
    assert len(reference) == 1
    clean = set(next(iter(reference)))
    moved = False
    for row in control:
        ids = set(row["selected_query_ids"])
        expected = len(ids & clean) / len(ids | clean)
        assert row["clean_overlap"] == pytest.approx(expected), (
            row["severity"], row["membership_mode"]
        )
        moved = moved or expected < 1.0
    assert moved, "the fixture must move the unfiltered bin for this to prove anything"
