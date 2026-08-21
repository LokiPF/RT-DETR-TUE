"""Both bucket schemes over one saved artifact pair, and the boundary that keeps them cheap.

`analyze_corruption_sensitivity` is the published decile analysis re-cut: the same loader, the
same scorer and the same fixture, run at two bucket resolutions and with none of the benchmark
selections. So most of what is worth asserting here is a *comparison* rather than a number.

Three of them carry the weight:

* every decile row -- both padding modes -- is compared against `analyze_deciles`' own output
  as a complete dictionary. Scores, selected query IDs, selected counts and overlaps all
  travel in those dictionaries, so a refactor that moved a bin edge, re-derived a confidence
  vector or handed the scorer a different severity's distances fails here with the offending
  row in hand, rather than as a plausible new number nobody can check;
* the two schemes are compared against each other at the memberships level, because the claim
  that a quintile is a rank range of the *same* confidence ordering is the reason the two
  results are comparable at all. `test_confidence_deciles.py` pins that on the primitive; this
  pins it on a real image's union-filtered valid set, which is the input the analysis uses;
* the selected query IDs of the persistence rows are compared against the confidence rows of
  the same selection. That the two signals summarise one population is the experiment's
  fairness claim, and `score_selection` can only guarantee it within a call -- keeping it true
  across the loop is this module's job.

The fixture is `write_decile_artifacts`, shared with the decile tests rather than rebuilt, so
the numbers being compared come from artifacts that satisfy the loader's provenance rules for
the same reason the pilot's do. Its one image is 11, it caches 20 queries at six severities and
pads the last two, which leaves 18 valid queries -- a count neither scheme divides evenly, so
the two remainder distributions genuinely disagree and a nested-cut implementation would fail
`test_both_schemes_cut_one_confidence_ranked_valid_sequence` instead of passing by luck.
"""

import ast
from pathlib import Path

import pytest
import torch

from src.scene_uncertainty import corruption_analysis as corruption_module
from src.scene_uncertainty.confidence_deciles import (
    DECILE_NAMES,
    QUINTILE_NAMES,
    memberships_by_scheme_severity,
    union_query_ids,
)
from src.scene_uncertainty.corruption_analysis import (
    CORRUPTION_INPUT_ARTIFACT_TYPE,
    SCHEMES,
    analyze_corruption_sensitivity,
    load_corruption_inputs,
)
from src.scene_uncertainty.decile_analysis import (
    ANALYSIS_ARTIFACT_TYPE,
    SLIM_RECORD_KEYS,
    DecileAnalysisError,
    DecileInputs,
    analyze_deciles,
    load_decile_inputs,
)
from tests.scene_uncertainty.decile_test_utils import (
    mutate_decile_artifacts,
    write_decile_artifacts,
)


IMAGE_ID = 11
"""The single image `write_decile_artifacts` caches, spelled here because the diagnostics key
it by `str(image_id)` and a wrong literal would silently look up nothing."""

SEVERITIES = range(6)

ROW_KEY = (
    "image_id", "severity", "signal", "bucket_scheme", "confidence_bin",
    "membership_mode", "padding_mode", "aggregation", "score_scope",
)
"""The identity of a corruption-sensitivity row: the decile experiment's row key plus the scheme
that cut the bin, which is the nine-field contract the plan requires of every row this command
emits.

`bucket_scheme` is not in it as a tie-breaker. `decile_00_10` and `quintile_00_20` already differ
in `confidence_bin`, so the key would be unique without it. It is in it because that uniqueness
is a property of how the bins were named rather than a guarantee, and because separating the two
resolutions *is* the comparison: a reader who had to recover the scheme by parsing a bin name for
its prefix would be relying on a second definition of the bins, free to drift from
`decile_scoring.BUCKET_NAMES`, which is the one that decides."""

SELECTION_KEY = tuple(
    key for key in ROW_KEY if key not in ("signal", "score_scope", "aggregation")
)
"""What identifies one `score_selection` call rather than one row of it. Every row sharing this
key was scored from a single selection, so they must all report the same query IDs."""


@pytest.fixture
def decile_artifacts(tmp_path):
    return write_decile_artifacts(tmp_path)


def load(artifacts):
    return load_corruption_inputs(artifacts["cache"], artifacts["results"])


def analyze(artifacts):
    return analyze_corruption_sensitivity(load(artifacts))


def labels_of(rows):
    return {
        (row["bucket_scheme"], row["membership_mode"], row["confidence_bin"], row["padding_mode"])
        for row in rows
    }


# --- the row contract -------------------------------------------------------------------


def test_analysis_emits_both_schemes_and_only_requested_controls(decile_artifacts):
    inputs = load(decile_artifacts)
    rows, diagnostics = analyze_corruption_sensitivity(inputs)

    assert inputs.run_metadata["artifact_type"] == "scene_corruption_sensitivity_inputs"
    assert len(rows) == 3060  # 34 selections x 15 signal/summary rows x 6 severities
    assert {row["bucket_scheme"] for row in rows} == {"decile", "quintile"}
    assert {row["severity"] for row in rows} == set(SEVERITIES)
    assert all(row["source_partition"] == "tuning" for row in rows)

    unfiltered = [row for row in rows if row["padding_mode"] == "unfiltered"]
    assert {row["confidence_bin"] for row in unfiltered} == {"decile_00_10", "quintile_00_20"}
    assert {row["membership_mode"] for row in unfiltered} == {"dynamic", "frozen"}
    assert len({tuple(row[field] for field in ROW_KEY) for row in rows}) == len(rows)
    assert diagnostics["run"] == inputs.run_metadata
    assert diagnostics["images"][str(IMAGE_ID)]["union_padded_count"] == 2


def test_no_benchmark_selection_enters_the_new_experiment(decile_artifacts):
    """The exact label grid, so a benchmark row cannot arrive as an extra rather than a failure.

    The decile command scores three selections this one must not: `all_valid` under the
    `shared` membership mode, the same set unfiltered, and the all-300-query benchmark. All
    three are well-formed rows, none of them is a confidence bucket, and a set *containment*
    assertion would let every one of them through. `score_selection` refuses `all_valid` under
    an explicit scheme, which covers the first two; nothing but this equality covers a third
    selection somebody adds later.
    """
    rows, _ = analyze(decile_artifacts)
    assert labels_of(rows) == {
        (scheme, mode, name, "filtered")
        for scheme, names in SCHEMES.items()
        for mode in ("dynamic", "frozen")
        for name in names
    } | {
        (scheme, mode, names[0], "unfiltered")
        for scheme, names in SCHEMES.items()
        for mode in ("dynamic", "frozen")
    }


def test_persistence_and_confidence_score_the_same_query_ids(decile_artifacts):
    """The experiment's fairness claim, checked across the loop rather than inside one call.

    `score_selection` shares one index tensor between the two signals, so the failure this
    guards against is not inside it: it is a loop that builds a selection for persistence and
    a different one for confidence -- by passing a second `indices`, or by re-binning between
    the two calls. Both would produce a full, plausible table comparing a summary of one
    population against a summary of another.
    """
    rows, _ = analyze(decile_artifacts)
    by_selection = {}
    for row in rows:
        key = tuple(row[field] for field in SELECTION_KEY)
        signals = by_selection.setdefault(key, {})
        signals.setdefault(row["signal"], set()).add(tuple(row["selected_query_ids"]))

    assert len(by_selection) == 204  # 34 selections x 6 severities
    for signals in by_selection.values():
        assert set(signals) == {"confidence", "persistence"}
        assert len(signals["confidence"] | signals["persistence"]) == 1


def test_frozen_membership_is_severity_zero_in_both_padding_modes(tmp_path):
    """Frozen means severity zero's queries -- filtered *and* unfiltered, in both schemes.

    The unfiltered half is the one that can silently invert. Its two modes are built from
    different mappings, one rebuilt per severity and one built once, and swapping them leaves a
    complete table whose `frozen` rows quietly re-rank with blur -- the diagnostic that exists
    to hold the queries still, measuring exactly what it was meant to exclude.

    What catches that is `frozen_ids == clean`, the selection itself. The `clean_overlap` check
    below is worth less than it looks and is not claimed to be worth more: on the unfiltered
    rows the 1.0 it compares against is a literal in `corruption_analysis`, so the assertion
    restates the source rather than testing it. That column is covered instead by
    `test_every_decile_row_matches_the_published_analyzer`, which compares it against a number
    `analyze_deciles` measures from the same artifacts.

    The fixture rotates its confidences with severity, so `dynamic` genuinely moves and the two
    modes are distinguishable at all -- under the default fixture every severity is a
    rank-preserving shift of severity zero and this test would pass on the inverted code.
    """
    artifacts = write_decile_artifacts(tmp_path, rotate_confidence=3)
    rows, _ = analyze(artifacts)
    selected = {
        tuple(row[field] for field in SELECTION_KEY): (
            tuple(row["selected_query_ids"]), row["clean_overlap"]
        )
        for row in rows
        if row["signal"] == "confidence" and row["aggregation"] == "mean"
    }

    def selection(scheme, mode, name, padding, severity):
        return selected[(IMAGE_ID, severity, scheme, name, mode, padding)]

    moved = 0
    for scheme, names in SCHEMES.items():
        for padding, bins in (("filtered", names), ("unfiltered", (names[0],))):
            for name in bins:
                clean, _ = selection(scheme, "dynamic", name, padding, 0)
                for severity in SEVERITIES:
                    frozen_ids, frozen_overlap = selection(
                        scheme, "frozen", name, padding, severity
                    )
                    assert frozen_ids == clean
                    assert frozen_overlap == 1.0
                    dynamic_ids, dynamic_overlap = selection(
                        scheme, "dynamic", name, padding, severity
                    )
                    assert 0.0 <= dynamic_overlap <= 1.0
                    if dynamic_ids != clean:
                        assert dynamic_overlap < 1.0
                        moved += 1
    assert moved  # the rotated fixture really does move dynamic membership off severity zero


@pytest.mark.parametrize("rotate", [0, 3])
def test_every_decile_row_matches_the_published_analyzer(tmp_path, rotate):
    """The published numbers, re-derived by the new loop and compared whole.

    Every field the decile experiment publishes rides in these dictionaries -- `score`,
    `selected_query_ids`, `selected_count`, `clean_overlap` -- so comparing them complete is
    what makes this a regression test rather than a shape test. `bucket_scheme` is removed from
    the new row and nothing else is, because it is the one key `score_selection` adds when a
    scheme is named; removing more would be removing the evidence.

    Both padding modes are compared, and the unfiltered 180 are the ones that need it. Their
    `clean_overlap` is the only column in this module that no other assertion can reach: the
    frozen half is a literal in the source, so an assertion that it equals 1.0 compares the
    constant with itself, and the dynamic half is measured against a reference -- severity
    zero's *unfiltered* buckets -- that nothing else names. Overlapping those bins against the
    padding-filtered reference instead is a one-word edit that reports 0.0 as though the bottom
    bin had emptied; `analyze_deciles` measures the same number from the same artifacts, so
    comparing against it is what turns that column into a checked one.

    The rotated fixture is not a duplicate run. The default one shifts every confidence by a
    constant, which is rank-preserving, so its dynamic membership equals its frozen membership
    at every severity and every `clean_overlap` in it is 1.0 -- two of the columns being
    compared are constant, and a loop that had lost them entirely would still match. Rotating
    the ranking with severity is what puts a moving selection and an overlap below 1.0 on both
    sides of the comparison.
    """
    artifacts = write_decile_artifacts(tmp_path, rotate_confidence=rotate)
    published, _ = analyze_deciles(
        load_decile_inputs(artifacts["cache"], artifacts["results"])
    )
    rebuilt, _ = analyze(artifacts)
    key = tuple(field for field in ROW_KEY if field != "bucket_scheme")
    before = {
        tuple(row[field] for field in key): row
        for row in published
        if row["confidence_bin"] in DECILE_NAMES
    }
    after = {
        tuple(row[field] for field in key): {
            field: value for field, value in row.items() if field != "bucket_scheme"
        }
        for row in rebuilt
        if row["bucket_scheme"] == "decile"
    }
    # 2 membership modes x (10 filtered bins + 1 unfiltered) x 15 rows x 6 severities, and
    # asserted so that a key collision cannot quietly shrink either side into agreement.
    assert len(before) == len(after) == 1980
    assert after == before


# --- the padding union, and not one severity's mask ---------------------------------------


WANDERING_TAILS = {0: 2, 1: 5, 2: 3, 3: 2, 4: 4, 5: 2}
"""Padding that moves rather than grows, as it does on 66 of 66 padded pilot images.

The default fixture pads the same two queries at every severity, which makes the image-level
union numerically identical to any single severity's mask -- so every assertion in this module
that reads a filtered selection passes just as well against a loop that masked with severity
zero's tail alone. These six lengths are what separate the two: the union is queries 15 to 19
and severity zero's own mask is queries 18 and 19.
"""


def test_the_padding_union_and_not_severity_zero_fixes_the_valid_population(tmp_path):
    """Spec 65 in this module's loop: one image-level mask, built across all six severities.

    `analyze_corruption_sensitivity` masks once per image and reuses that mask at every
    severity, and the mask has to be the *union* because the detected tail wanders rather than
    growing. Taking severity zero's mask instead is a three-line simplification that leaves
    every row well-formed and every count self-consistent -- on the pilot's image 173044, whose
    tails run 251, 255, 257, 239, 0, 0, it would swing the valid population between 43 and 300
    of 300 queries with nothing in the bundle able to say so.

    Three assertions carry that, and each one reads a different number under the union than
    under severity zero's mask: the recorded union is five queries rather than two; no filtered
    selection at any severity touches a query that any severity padded (15, 16 and 17 are
    ordinary at severity zero and placeholders at severity one, so a severity-zero mask scores
    them); and every scheme, severity and membership mode covers the same fifteen valid
    queries rather than eighteen.
    """
    artifacts = write_decile_artifacts(tmp_path, padded_tails=WANDERING_TAILS)
    rows, diagnostics = analyze(artifacts)
    image = diagnostics["images"][str(IMAGE_ID)]
    by_severity = image["padded_query_ids_by_severity"]

    assert image["padded_count_by_severity"] == {
        str(severity): length for severity, length in WANDERING_TAILS.items()
    }
    assert len({len(ids) for ids in by_severity.values()}) > 1  # the fixture really wanders
    assert by_severity["0"] == [18, 19]
    assert image["union_padded_query_ids"] == list(range(15, 20))
    assert image["union_padded_count"] == 5
    assert image["tail_identical_across_severities"] is False

    ever_padded = {int(query) for ids in by_severity.values() for query in ids}
    filtered = [row for row in rows if row["padding_mode"] == "filtered"]
    assert len(filtered) == 2700  # 2 schemes x (10 + 5) bins x 2 modes x 15 rows x 6 severities
    assert all(not ever_padded.intersection(row["selected_query_ids"]) for row in filtered)

    populations: dict[tuple, set] = {}
    for row in filtered:
        key = (row["bucket_scheme"], row["severity"], row["membership_mode"])
        populations.setdefault(key, set()).update(row["selected_query_ids"])
    assert len(populations) == 24  # 2 schemes x 6 severities x 2 membership modes
    assert all(population == set(range(15)) for population in populations.values())


# --- one ordering, two resolutions -------------------------------------------------------


def test_both_schemes_cut_one_confidence_ranked_valid_sequence(tmp_path):
    """Concatenating either scheme's buckets returns the same ranked sequence of valid queries.

    This is what makes a decile result and a quintile result comparable: they are the same
    ordering cut in different places, not two independent rankings that agree wherever no tie
    broke. `test_confidence_deciles.py` pins the invariant on `confidence_buckets` directly;
    this one pins it where the analysis actually reads it -- through
    `memberships_by_scheme_severity`, on the image-level union-filtered valid set, at every
    severity and under both membership modes.

    The bucket sizes are asserted because they are the evidence that shared ordering is not
    nested cuts: 18 valid queries split 2,2,2,2,2,2,2,2,1,1 as deciles and 4,4,4,3,3 as
    quintiles, and pairing adjacent deciles would give 4,4,4,4,2 instead. The fixture is built
    with `rotate_confidence`, so the ranking moves with severity and `dynamic` is not a copy of
    `frozen` -- without that the six severities would be one severity asserted six times.
    """
    artifacts = write_decile_artifacts(tmp_path, rotate_confidence=3)
    inputs = load(artifacts)
    records = inputs.records_by_image[IMAGE_ID]
    padded = union_query_ids([record["padded_query_ids"] for _, record in sorted(records.items())])
    nouns = {"decile": "confidence deciles", "quintile": "confidence quintiles"}
    memberships = {
        scheme: memberships_by_scheme_severity(records, padded, names=names, noun=nouns[scheme])
        for scheme, names in SCHEMES.items()
    }
    valid = torch.tensor([
        query for query in range(int(records[0]["query_count"]))
        if query not in set(padded.tolist())
    ])

    moved = 0
    for severity in SEVERITIES:
        for mode in ("dynamic", "frozen"):
            sequences = {
                scheme: torch.cat([memberships[scheme][severity][mode][name] for name in names])
                for scheme, names in SCHEMES.items()
            }
            assert [
                int(memberships["decile"][severity][mode][name].numel()) for name in DECILE_NAMES
            ] == [2, 2, 2, 2, 2, 2, 2, 2, 1, 1]
            assert [
                int(memberships["quintile"][severity][mode][name].numel())
                for name in QUINTILE_NAMES
            ] == [4, 4, 4, 3, 3]
            assert torch.equal(sequences["quintile"], sequences["decile"])
            assert torch.equal(sequences["decile"].sort().values, valid)
            ranked = records[severity if mode == "dynamic" else 0]["query_confidence"]
            ranked = ranked.index_select(0, sequences["decile"])
            assert bool((ranked[1:] >= ranked[:-1]).all())
            moved += int(not torch.equal(
                sequences["decile"], torch.cat([
                    memberships["decile"][0]["dynamic"][name] for name in DECILE_NAMES
                ])
            ))
    assert moved  # the fixture's ranking really does move with severity


# --- the loader refactor ------------------------------------------------------------------


def test_the_two_loaders_agree_on_everything_but_the_artifact_type(decile_artifacts):
    """One loader body, two names. The only difference allowed is the type it records.

    `load_decile_inputs` reads the artifacts the published command already ran against, so a
    refactor that changed what it streams, what it keeps or what it proves would change results
    that are already written down. Comparing the two returned objects field by field is what
    keeps the new caller from being a reason to touch the old one.
    """
    old = load_decile_inputs(decile_artifacts["cache"], decile_artifacts["results"])
    new = load(decile_artifacts)

    assert old.run_metadata["artifact_type"] == ANALYSIS_ARTIFACT_TYPE
    assert new.run_metadata == {
        **old.run_metadata, "artifact_type": CORRUPTION_INPUT_ARTIFACT_TYPE,
    }
    assert isinstance(new, DecileInputs)
    assert set(new.records_by_image) == set(old.records_by_image)
    for image_id, records in old.records_by_image.items():
        assert set(new.records_by_image[image_id]) == set(records)
        for severity, record in records.items():
            fresh = new.records_by_image[image_id][severity]
            assert set(fresh) == set(record) == set(SLIM_RECORD_KEYS)
            assert torch.equal(fresh["query_confidence"], record["query_confidence"])
            assert torch.equal(fresh["padded_query_ids"], record["padded_query_ids"])
            assert (fresh["image_id"], fresh["severity"], fresh["source_partition"]) == (
                record["image_id"], record["severity"], record["source_partition"]
            )
    assert set(new.distances) == set(old.distances)
    for key, by_layer in old.distances.items():
        assert set(new.distances[key]) == set(by_layer)
        assert all(
            torch.equal(new.distances[key][layer], values)
            for layer, values in by_layer.items()
        )
    assert set(new.layer_score_scales) == set(old.layer_score_scales)
    for layer, state in old.layer_score_scales.items():
        assert set(new.layer_score_scales[layer]) == set(state)
        assert all(
            torch.equal(new.layer_score_scales[layer][field], value)
            for field, value in state.items()
        )


@pytest.mark.parametrize("mutation", ["result_partition_test", "result_partition_all"])
def test_the_new_loader_refuses_a_held_out_result_artifact(decile_artifacts, mutation):
    """The tuning-only rule reaches the new command through the shared loader, not around it.

    The corruption experiment is a tuning-partition experiment, and the refusal that enforces
    that lives in `_load_result_manifest`. A private helper is easy to call with the checks
    skipped, so this asserts the new entry point still fails -- with the same error type and the
    same words, since the message is the operator's only instruction.
    """
    mutate_decile_artifacts(decile_artifacts, mutation)
    partition = mutation.rsplit("_", 1)[-1]
    with pytest.raises(DecileAnalysisError, match=f"was built for the '{partition}' partition"):
        load(decile_artifacts)


# --- no kNN, no bank, no inference --------------------------------------------------------


FORBIDDEN_IMPORT_ROOTS = frozenset({
    "knn", "evaluate", "extractor", "bank", "dataset", "runtime", "normalization", "blur",
    "pipeline", "torchvision",
})
"""Every module a detector pass, a bank build, a kNN search or a re-fitted normalizer can come
from, plus `blur`, which is the only torchvision-backed module in the package. This command
reads a cache that has already been blurred and distances that have already been searched; the
design's non-goal is that it must not be able to produce either."""

FORBIDDEN_SOURCE_TEXT = (
    "load_model", "build_model", "torchvision", "query_knn", "load_bank", "comparison_bank",
    "load_frozen_detector", "streaming_coverage_bank", "make_coverage_bank",
    "compute_query_distances", "mean_knn_distance", "chunked_knn_distances", "HungarianMatcher",
    "rtdetr",
)
"""The names a reader greps for, checked against the source text rather than the syntax tree.

The import scan below is the stronger check of the two and would catch every one of these that
arrives by import. This one catches what a syntax tree does not: an `importlib.import_module`
call, an `exec`, or a docstring that describes this module as doing something it must not --
which is a claim a later reader would act on."""

SAFE_IMPORTS = frozenset({"confidence_deciles", "decile_analysis", "decile_scoring"})


def _imported_modules(module) -> set[str]:
    """Every dotted module path this module imports, from anywhere in its syntax tree.

    `ast.walk` rather than a scan of the top level, because a forbidden import inside a
    function body is the cheap way to have one -- it costs a line and nothing else, and the
    name it binds is local, so the namespace scan below never sees it.
    """
    source = Path(module.__file__).read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    return imported


def _path_components(dotted: str) -> set[str]:
    """Every component of a dotted import path, not just the one it starts with.

    Rooting a path at its first component is the bypass this exists to close, and it is not a
    hypothetical import style: `from src.scene_uncertainty.knn import ...` roots at `src`,
    which no blocklist of package modules can name, and this repo writes `src.`-absolute
    imports in four modules already. Reading every component instead means the check does not
    depend on which spelling of the same import a later author happens to use.
    """
    return set(dotted.lstrip(".").split("."))


def test_the_analyzer_cannot_reach_a_model_bank_or_knn_path():
    """The design's non-goal, enforced statically: consume the saved artifacts, recompute nothing.

    Three scans, each covering what the others miss. The import scan is the mechanism a
    forbidden path would actually arrive by, and it reads every component of every dotted
    path from anywhere in the tree -- see `_path_components` and `_imported_modules` for the
    two spellings that would otherwise walk straight through it. The source text covers the
    names that can arrive without an import node at all. The `vars` origin scan covers a symbol
    re-exported into this module by one of the three it is allowed to import -- the one route
    the other two cannot see, since neither the import path nor the text of this file would
    change.

    The safe imports are asserted positively as well. This module is allowed the loader and the
    scorer, and it must *use* them: a copy of `load_decile_inputs`' validation inlined here
    would pass all three scans above and quietly drop the provenance rules that make the join
    trustworthy.
    """
    imported = _imported_modules(corruption_module)
    assert not {
        name for name in imported if _path_components(name) & FORBIDDEN_IMPORT_ROOTS
    }
    assert SAFE_IMPORTS <= imported

    source = Path(corruption_module.__file__).read_text(encoding="utf-8")
    assert not [name for name in FORBIDDEN_SOURCE_TEXT if name in source]
    assert not {
        name for name, value in vars(corruption_module).items()
        if getattr(value, "__module__", "").endswith(
            (".knn", ".evaluate", ".extractor", ".bank", ".dataset", ".runtime", ".blur")
        )
    }
