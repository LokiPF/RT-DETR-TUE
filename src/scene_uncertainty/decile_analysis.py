"""Join the blur feature cache to the saved query distances, or refuse to run at all.

Every silent failure this experiment can suffer enters through this module. The two inputs
are a multi-gigabyte cache of 3,000 records and a 4 MB table of saved kNN distances, and
nothing downstream can tell a wrong join from a right one: a bin built from the wrong
severity, a record scored against another image's distances, or a held-out test scene
quietly included would all produce a full results table and a confident median Spearman.
So the rule here is that a join is either proved -- by content address on *both* manifests,
by key, by layer set, by query count -- or it stops the run.

Three properties are load-bearing and easy to lose:

*Streaming.* The cache is read once, and for each record only the artifact key, the derived
confidence vector and the detected padding mask are kept. Boxes, logits and the three
300x335 persistence fingerprints are dropped as soon as `detect_padded_tail` has read them,
so peak memory is one shard plus about 1.4 KB per retained record rather than the whole
cache. Anything added to the slim record is multiplied by 1,500.

*The confidence key.* The slim record names its vector `query_confidence`, never
`confidence`. The cache stores a field called `confidence` that was reduced from the
full-precision logits *before* they were cast to float16; it is wrong by ~1e-5, which is
invisible to a threshold and decisive for a rank -- it moves decile membership on 535 of the
pilot's 1,500 tuning records. Naming the derived vector something the cache does not own is
what turns a call site reaching for the stale field into a `KeyError` instead of a slightly
different answer. `confidence_deciles._severity_confidence` enforces the same rule from the
other side.

*The partition asymmetry.* The blur cache holds both partitions by design -- 1,500 tuning
and 1,500 test records share one artifact -- so test records there are skipped, not fatal.
The *result* artifact is the one that carries a requested partition, and a result artifact
built for anything but `tuning` is refused outright, as is any individual saved-distance row
labelled otherwise. That is where "the first experiment must reject the held-out test
partition" is actually enforced; filtering the cache is not a substitute and is documented
as a filter so it cannot be mistaken for one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .artifacts import iter_records, load_manifest, manifest_id
from .confidence_deciles import (
    DECILE_NAMES,
    bin_overlap,
    confidence_deciles,
    confidence_from_logits,
    detect_padded_tail,
    memberships_by_severity,
    union_query_ids,
)
from .decile_scoring import score_selection


EXPECTED_SEVERITIES = frozenset(range(6))
TUNING_PARTITION = "tuning"
RESULT_ARTIFACT_TYPE = "knn_scene_uncertainty_results"
EVALUATION_SOURCE_KIND = "evaluation"
ANALYSIS_ARTIFACT_TYPE = "confidence_decile_scene_uncertainty"

MINIMUM_VALID_QUERIES = len(DECILE_NAMES)
"""Ten bins need ten queries. `confidence_deciles` refuses fewer, but it is reached one
image at a time deep inside the scoring loop; catching it here fails the run before the
first row is written, and names the image rather than the bin."""

REQUIRED_RESULT_MANIFEST_KEYS = (
    "artifact_id", "artifact_type", "feature_cache_id", "bank_id", "source_partition",
    "normalization", "k", "query_distance_path", "normalizer_path",
)
"""`bank_id` is required rather than optional, and it is carried into `run_metadata`.

Nothing here can check it -- no bank artifact is passed to this command -- but it is the only
field that separates the two sanctioned pilot result sets. `raw_k5` and `raw_k5_bank50k` were
built from the *same* feature cache (df79fddf...) against *different* clean banks (353a5992...
and 5dfd6e50...), so every other provenance field this analysis records is identical between
them. Without `bank_id` on the way through, the analysis directory cannot say which clean bank
its distances were measured against, and two runs that are not comparable produce provenance
that says they are. `pipeline` has always written it."""
REQUIRED_CACHE_MANIFEST_KEYS = ("artifact_id", "source_kind", "decoder_layers", "query_count")
REQUIRED_DISTANCE_ROW_KEYS = ("image_id", "severity", "source_partition", "query_scores_by_layer")
REQUIRED_CACHE_RECORD_KEYS = ("image_id", "severity", "source_partition", "boxes", "logits", "layers")

RESULT_NON_IDENTIFYING_KEYS = ("artifact_id", "run_stats", "clean_distance_fit")
"""The keys `pipeline._content_address` leaves out of a result manifest's own address.

Re-stated here rather than imported: this module must not import `pipeline`, which reaches
the kNN and extraction paths the design lists as non-goals, and
`test_the_loader_cannot_reach_a_knn_or_extraction_path` enforces that. A second copy of a
rule is a second thing that can drift, so
`test_the_result_address_exclusion_matches_the_writer` imports `pipeline` -- a test may --
and fails loudly if the writer's list and this one stop agreeing. `run_stats` and
`clean_distance_fit` are excluded because they move with wall time and with the chunk width
of an algebraically exact kNN, so hashing them would give two interchangeable runs two
identities.
"""


def result_content_address(manifest: dict) -> str:
    """The address a result manifest should carry, recomputed from the manifest itself.

    Exported because the fixtures that stand in for `evaluate-knn` have to seal their
    manifests the same way the real writer does; a fixture carrying a literal id would
    bypass the one check that makes `source_result_id` mean anything.
    """
    return manifest_id({
        key: value for key, value in manifest.items() if key not in RESULT_NON_IDENTIFYING_KEYS
    })


ALL_VALID_BENCHMARK = ("shared", "all_valid", "filtered")
"""Spec benchmarks 2 and 3: every non-padded query, persistence and its confidence control.

`shared` because this selection is not built from a confidence ranking and is the identical
query set at every severity; `filtered` because the image-level padding union is removed."""

ALL_QUERY_BENCHMARK = ("shared", "all_valid", "unfiltered")
"""Spec benchmark 1 -- the existing all-300-query result -- written down, because it has no
label of its own and would otherwise be rediscovered by whoever reads the CSV next.

The spec names five report rows and this is the first: "the existing all-300-query persistence
benchmark: q90, layer 2, median Spearman 0.600". It is *not* `all_valid` in the strict sense --
it deliberately keeps the padded decoder placeholders, which is the whole point of comparing it
against benchmark 2 -- but the closed vocabulary in `decile_scoring` has no `all_300` bin and
should not grow one, because the label that actually distinguishes the two benchmarks is the
padding mode. So the pair reads: the same every-query selection, once with the padding union
removed (`ALL_VALID_BENCHMARK`) and once without (`ALL_QUERY_BENCHMARK`). The difference between
their scores is the effect of padding on the existing benchmark, which is what the spec asks the
report to show.

Two consequences worth stating rather than deducing. The selection is `arange(query_count)`, so
`selected_count` is 300 and not the valid count -- that is how a reader tells the two apart in
the CSV without recomputing anything. And it is scored for `q90` alone and with no confidence
control: it exists to reproduce one published number, and a `1 - confidence` summary over a
population that includes near-zero-confidence placeholders would describe the placeholders
rather than the scene."""

SENSITIVITY_BIN = DECILE_NAMES[0]
"""The one bin the padding sensitivity control repeats unfiltered (spec 69).

"A sensitivity control repeats only the lowest-confidence-bin analysis without filtering" --
only. Widening it to the other nine deciles would double the decile grid and publish nine
padding measurements the design never asked for, and it would do so silently, because every
extra row would be a perfectly well-formed row."""

ROW_KEYS_EXCLUDED_FROM_CSV = ("selected_query_ids",)
"""Row keys that stay in memory and must not be written to `per_scene.csv`.

The full tuning grid is 250 images x 6 severities x 349 rows = 523,500 rows, and each row
carries the IDs of its own selection -- 30 for a decile, 300 for the all-query benchmark. In the
CSV that is on the order of 100 MB of quoted integer lists, in a file whose purpose is one score
per row; loaded back with `read_csv` every one of them is a string that has to be parsed before
it means anything.

They stay on the in-memory rows because they are the only per-row evidence that the persistence
score and the confidence score summarised the *same* queries, which is the fairness claim the
whole experiment rests on, and because every control in this module is verified through them.
The writer drops the column; this constant names it so the drop is by contract rather than by a
literal in a plotting module."""

SLIM_RECORD_KEYS = frozenset({
    "image_id", "severity", "source_partition",
    "query_confidence", "padded_query_ids", "query_count",
})
"""Exactly what survives the stream. Stated as a constant so the memory contract is a thing
a test can assert rather than a comment someone has to remember."""


class DecileAnalysisError(ValueError):
    """A refusal to run: two artifacts that cannot be proved to describe the same thing.

    Subclasses `ValueError` so it reads the same to a caller as the plain `ValueError`s that
    `confidence_deciles` and `decile_scoring` raise for the same class of problem, and so a
    command wrapper can catch one type.
    """


@dataclass
class DecileInputs:
    """The whole experiment's input, after the feature cache has been streamed away.

    `records_by_image[image_id][severity]` is a slim record -- see `SLIM_RECORD_KEYS` -- and
    carries no persistence tensor. `distances[(image_id, severity, partition)][layer]` is the
    saved per-query mean distance to the five nearest clean fingerprints, read from the
    result artifact and never recomputed. The two are keyed by the *same* join key, which is
    the only reason a later stage may index one with the other's key.
    """

    records_by_image: dict[int, dict[int, dict]]
    distances: dict[tuple[int, int, str], dict[int, torch.Tensor]]
    layer_score_scales: dict[int, dict[str, torch.Tensor]]
    run_metadata: dict


def _require_keys(mapping, keys, label: str) -> None:
    """Refuse a mapping that is missing any required key, or holds `None` under one.

    `None` counts as missing on purpose. The design's rule is that missing data must never be
    silently replaced, and a `None` that survives this check becomes a comparison that
    succeeds for the wrong reason: `result.get("feature_cache_id") == cache.get("artifact_id")`
    is `True` when *both* manifests lost their id, which is the one case where a provenance
    check must not pass.
    """
    missing = [key for key in keys if mapping.get(key) is None]
    if missing:
        raise DecileAnalysisError(f"{label} is missing {missing}")


def _load_cache_manifest(cache: Path) -> dict:
    """The feature cache's manifest, checked for integrity before any field of it is believed.

    The content address is recomputed rather than trusted. Without it, matching the result
    artifact's `feature_cache_id` against this manifest's `artifact_id` compares two strings,
    and a one-line edit here makes an unrelated cache and result set look like a matched pair
    -- which is precisely the mistake that produces a complete, plausible, wrong results
    table. `manifest_id` hashes every field except the id itself, so a manifest whose
    `query_count`, `decoder_layers` or checkpoint hash was edited no longer addresses itself.

    This covers the cache side only. `_load_result_manifest` does the same for the result
    side, and it has to: the two checks together are what closes the forgery, because either
    one alone leaves a whole file editable.
    """
    path = cache / "manifest.json"
    if not path.exists():
        raise DecileAnalysisError(f"missing feature-cache manifest: {path}")
    manifest = load_manifest(cache)
    _require_keys(manifest, REQUIRED_CACHE_MANIFEST_KEYS, f"feature-cache manifest {path}")
    if manifest["source_kind"] != EVALUATION_SOURCE_KIND:
        raise DecileAnalysisError(
            f"feature-cache manifest {path} has source_kind {manifest['source_kind']!r}; this "
            f"analysis needs the {EVALUATION_SOURCE_KIND!r} blur cache, not the clean bank's "
            f"reference cache"
        )
    recomputed = manifest_id(manifest)
    if recomputed != manifest["artifact_id"]:
        raise DecileAnalysisError(
            f"feature-cache manifest {path} does not match its own content address: it claims "
            f"{manifest['artifact_id']} and hashes to {recomputed}, so its artifact_id no longer "
            f"identifies its contents and cannot certify any result artifact against it"
        )
    return manifest


def _load_result_manifest(results: Path, cache_manifest: dict) -> dict:
    """The raw-kNN result manifest, proved against the cache and against its own contents.

    `artifact_type` is checked because the `--results` argument is a path to a CSV whose
    siblings are read *by name out of this manifest*; pointed at a different run's CSV it
    would happily read that run's distances. `source_partition` is checked because the design
    forbids scoring the held-out test partition in this experiment, and a result artifact
    built with `--partition all` or `--partition test` is the way that would happen -- so
    `"all"` is refused too, not merely `"test"`.

    The content address is the check that makes the other two mean something, and the
    concrete forgery it stops was demonstrated on the pilot: copy `raw_k5.manifest.json`,
    repoint `query_distance_path` at `raw_k5_bank50k.query_distances.pt` -- a *different clean
    bank*, same `feature_cache_id` -- and set `k: 99`, `normalization: "hand_edited"`. Every
    field this loader compares still agrees, the join still succeeds for all 250 images, and
    the run produces a complete results table of distances from one bank labelled with
    another bank's configuration. Nothing downstream could detect that; `manifest_id` over the
    manifest's own contents does, because `query_distance_path`, `k` and `normalization` are
    all inside the address.

    Order matters and is chosen for the message, not the strength. The address check is a
    catch-all: it fires for *any* edit, so running it first would answer "your manifest was
    edited" to a question like "why won't it take my test-partition results?". The three
    checks that identify which experiment this is run first and keep their own messages; the
    address closes everything they do not name.
    """
    path = results.with_suffix(".manifest.json")
    if not path.exists():
        raise DecileAnalysisError(f"missing result manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _require_keys(manifest, REQUIRED_RESULT_MANIFEST_KEYS, f"result manifest {path}")
    if manifest["artifact_type"] != RESULT_ARTIFACT_TYPE:
        raise DecileAnalysisError(
            f"result manifest {path} has artifact_type {manifest['artifact_type']!r}, "
            f"expected {RESULT_ARTIFACT_TYPE!r}"
        )
    if manifest["source_partition"] != TUNING_PARTITION:
        raise DecileAnalysisError(
            f"result manifest {path} was built for the {manifest['source_partition']!r} "
            f"partition; the confidence-decile experiment accepts {TUNING_PARTITION!r} results "
            f"only and must not score the held-out test partition"
        )
    if manifest["feature_cache_id"] != cache_manifest["artifact_id"]:
        raise DecileAnalysisError(
            f"result feature_cache_id does not match the cache: {path} claims "
            f"{manifest['feature_cache_id']} and the cache is {cache_manifest['artifact_id']}"
        )
    recomputed = result_content_address(manifest)
    if recomputed != manifest["artifact_id"]:
        raise DecileAnalysisError(
            f"result manifest {path} does not match its own content address: it claims "
            f"{manifest['artifact_id']} and hashes to {recomputed}, so the k, normalization, "
            f"bank and distance-file name it reports are not the ones this artifact was "
            f"written with and every row scored from it would be mislabelled"
        )
    return manifest


def _load_layer_score_scales(results: Path, result_manifest: dict, decoder_layers) -> dict:
    """The clean-distance centre and scale per decoder layer, keyed by int.

    JSON and torch round-trips disagree about key types often enough that `int(...)` here is
    not defensive noise: a str-keyed `"2"` would compare unequal to the cache's int `2` and
    turn a layer agreement into a layer mismatch, or -- worse, if the comparison were dropped
    -- index a score by a key that never matches and score nothing.

    The centre and scale values themselves are not range-checked here. `decile_scoring`
    already refuses a non-finite or non-positive scale at the point of division, where the
    consequence (a `combined` score of `inf` that travels into a median Spearman as an
    ordinary largest value) is visible; splitting that rule across two modules would give it
    two places to drift.
    """
    path = results.parent / result_manifest["normalizer_path"]
    if not path.exists():
        raise DecileAnalysisError(f"missing clean-distance normalizer artifact: {path}")
    # Safe load throughout, as everywhere else in this package: artifact directories are
    # copied between hosts, so an unpickling read would be arbitrary code execution on data
    # from somewhere else, and this file holds only tensors, ints, floats and str keys.
    state = torch.load(path, map_location="cpu", weights_only=True)
    _require_keys(state, ("layer_score_scales",), f"clean-distance normalizer artifact {path}")
    scales = {int(layer_id): value for layer_id, value in state["layer_score_scales"].items()}
    expected = {int(layer_id) for layer_id in decoder_layers}
    if set(scales) != expected:
        raise DecileAnalysisError(
            f"cache decoder layers and clean-distance scales disagree: the cache declares "
            f"{sorted(expected)} and {path} fits {sorted(scales)}"
        )
    return scales


def _load_distance_index(
    results: Path, result_manifest: dict, scales: dict, query_count: int
) -> dict[tuple[int, int, str], dict[int, torch.Tensor]]:
    """The saved query distances, indexed by join key, with every row proved usable first.

    This is the small side of the join -- 1,500 rows of three float16 vectors, about 4 MB --
    so it is held whole while the cache is streamed past it.

    A row in another partition raises rather than being skipped. Skipping would silently turn
    a `--partition all` result artifact into a tuning-only one, which is the design's
    forbidden run wearing the right label; and the mismatch would then surface much later as
    an unrelated "missing feature-cache record" for keys nobody asked about.
    """
    path = results.parent / result_manifest["query_distance_path"]
    if not path.exists():
        raise DecileAnalysisError(f"missing query-distance artifact: {path}")
    rows = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(rows, list):
        raise DecileAnalysisError(
            f"query-distance artifact {path} should hold a list of rows, got {type(rows).__name__}"
        )
    index: dict[tuple[int, int, str], dict[int, torch.Tensor]] = {}
    for position, row in enumerate(rows):
        label = f"query-distance row {position} of {path}"
        _require_keys(row, REQUIRED_DISTANCE_ROW_KEYS, label)
        partition = str(row["source_partition"])
        if partition != TUNING_PARTITION:
            raise DecileAnalysisError(
                f"{label} is in the {partition!r} partition; this experiment scores the "
                f"{TUNING_PARTITION!r} partition only"
            )
        key = (int(row["image_id"]), int(row["severity"]), partition)
        if key in index:
            raise DecileAnalysisError(f"duplicate query-distance record for {key}")
        by_layer = {
            int(layer_id): values for layer_id, values in row["query_scores_by_layer"].items()
        }
        if set(by_layer) != set(scales):
            raise DecileAnalysisError(
                f"query-distance layers and clean-distance scales disagree for {key}: "
                f"{sorted(by_layer)} against {sorted(scales)}"
            )
        wrong = {
            layer_id: int(values.numel())
            for layer_id, values in sorted(by_layer.items())
            if int(values.numel()) != query_count
        }
        if wrong:
            raise DecileAnalysisError(
                f"query count mismatch for {key}: the saved distances hold {wrong} values per "
                f"layer, the feature cache declares {query_count} queries"
            )
        index[key] = by_layer
    return index


def _slim_record(record: dict, key: tuple[int, int, str], query_count: int) -> dict:
    """One record reduced to what survives the stream, with the fingerprints already read.

    `detect_padded_tail` is called here, while the record still holds its boxes, logits and
    every cached layer, because that is the last moment those tensors exist. What is kept is
    a mask of at most a few hundred ints; what is dropped is roughly 600 KB of float16 per
    record, which is the whole reason the design says to stream.

    The confidence vector is recomputed from the cached logits by `confidence_from_logits`
    and filed under `query_confidence`. It is emphatically not `record["confidence"]`: see
    the module docstring, and note that this function copies no other cache field, so the
    stale one cannot arrive by accident either.
    """
    return {
        "image_id": key[0],
        "severity": key[1],
        "source_partition": key[2],
        "query_confidence": confidence_from_logits(record["logits"]),
        "padded_query_ids": detect_padded_tail(record),
        "query_count": query_count,
    }


def load_decile_inputs(cache_value: str | Path, results_value: str | Path) -> DecileInputs:
    """Stream the blur cache once, join it to the saved query distances, or raise.

    The order of the checks is chosen so that the *first* thing to fail is the thing an
    operator can act on. Manifests are proved before their fields are used; the small
    distance table is proved whole before the expensive stream starts, so a misaligned pair
    fails in a second rather than after reading two gigabytes; per-record checks fire while
    the record that failed is still nameable; and the whole-image rules -- six severities,
    ten surviving queries -- run at the end because they are the only ones that need every
    record of an image at once.

    Test-partition records in the cache are skipped rather than refused; the cache holds both
    partitions by design. The refusal that matters lives in `_load_result_manifest` and
    `_load_distance_index`. See the module docstring.

    What this deliberately does not check: the manifest's `record_count` against the number of
    records actually streamed. Shards and manifest are written separately, and a count
    mismatch would fire ahead of every more specific rule below, replacing "image 11 does not
    contain all six blur severities" with an arithmetic complaint that names no image.
    """
    cache, results = Path(cache_value), Path(results_value)
    cache_manifest = _load_cache_manifest(cache)
    result_manifest = _load_result_manifest(results, cache_manifest)

    query_count = int(cache_manifest["query_count"])
    decoder_layers = [int(layer_id) for layer_id in cache_manifest["decoder_layers"]]
    scales = _load_layer_score_scales(results, result_manifest, decoder_layers)
    distances = _load_distance_index(results, result_manifest, scales, query_count)

    records_by_image: dict[int, dict[int, dict]] = {}
    seen: set[tuple[int, int, str]] = set()
    for record in iter_records(cache):
        _require_keys(record, REQUIRED_CACHE_RECORD_KEYS, f"feature-cache record in {cache}")
        partition = str(record["source_partition"])
        if partition != TUNING_PARTITION:
            continue
        key = (int(record["image_id"]), int(record["severity"]), partition)
        if key in seen:
            raise DecileAnalysisError(f"duplicate feature-cache record for {key}")
        seen.add(key)
        layers = {int(layer_id) for layer_id in record["layers"]}
        if layers != set(decoder_layers):
            # The padded-tail rule needs the fingerprint at *every* cached layer. A record
            # missing one does not fail that rule, it weakens it to box-and-logits agreement,
            # which matches more rows, deletes more valid queries and says nothing about it.
            # This is the last place that can be caught: `detect_padded_tail` reads whatever
            # `record["layers"]` happens to hold and cannot tell a partial record from a
            # record that genuinely had no layers.
            raise DecileAnalysisError(
                f"feature-cache record {key} carries persistence layers {sorted(layers)} but the "
                f"cache manifest declares {sorted(decoder_layers)}; the padded-tail rule needs "
                f"the fingerprint at every declared layer or it silently gets weaker"
            )
        if int(record["logits"].shape[0]) != query_count:
            raise DecileAnalysisError(
                f"query count mismatch for {key}: the record holds "
                f"{int(record['logits'].shape[0])} queries, the cache manifest declares "
                f"{query_count}"
            )
        if key not in distances:
            raise DecileAnalysisError(f"missing query-distance record for {key}")
        try:
            slim = _slim_record(record, key, query_count)
        except ValueError as error:
            # `detect_padded_tail` and `confidence_from_logits` raise a bare `ValueError` that
            # names a shape and no record -- "query-count mismatch inside record: [19, 20]" is
            # unactionable against 3,000 of them. The key is in hand here, so it goes in.
            raise DecileAnalysisError(
                f"feature-cache record {key} is internally inconsistent: {error}"
            ) from error
        records_by_image.setdefault(key[0], {})[key[1]] = slim

    if not records_by_image:
        raise DecileAnalysisError(
            f"no {TUNING_PARTITION} records in {cache}; this experiment scores the "
            f"{TUNING_PARTITION} partition and there is nothing to score"
        )
    unmatched = sorted(set(distances) - seen)
    if unmatched:
        raise DecileAnalysisError(
            f"missing feature-cache record for {len(unmatched)} saved query-distance keys, "
            f"first: {unmatched[:5]}"
        )
    for image_id, records in sorted(records_by_image.items()):
        if set(records) != EXPECTED_SEVERITIES:
            raise DecileAnalysisError(
                f"image {image_id} does not contain all six blur severities: {sorted(records)}"
            )
        # One union mask per image, over the retained masks, exactly as `union_query_ids`
        # documents -- the tail wanders with severity rather than growing, so a per-severity
        # mask would move the valid population between severities.
        padded = union_query_ids(
            [record["padded_query_ids"] for _, record in sorted(records.items())]
        )
        valid_count = query_count - int(padded.numel())
        if valid_count < MINIMUM_VALID_QUERIES:
            raise DecileAnalysisError(
                f"image {image_id} keeps only {valid_count} valid queries after removing "
                f"{int(padded.numel())} padded ones; ten valid queries are needed to fill ten "
                f"confidence bins"
            )

    return DecileInputs(
        records_by_image=records_by_image,
        distances=distances,
        layer_score_scales=scales,
        run_metadata={
            "artifact_type": ANALYSIS_ARTIFACT_TYPE,
            "feature_cache_id": cache_manifest["artifact_id"],
            "source_result_id": result_manifest["artifact_id"],
            "bank_id": result_manifest["bank_id"],
            "normalization": result_manifest["normalization"],
            "k": result_manifest["k"],
            "source_partition": TUNING_PARTITION,
            "severities": sorted(EXPECTED_SEVERITIES),
            "decoder_layers": sorted(scales),
            "query_count": query_count,
            "image_count": len(records_by_image),
            "record_count": len(seen),
        },
    )


def _padding_diagnostics(records: dict[int, dict], padded: torch.Tensor) -> dict:
    """What the design requires be *recorded* about one image's padding, not merely applied.

    Spec 67: "Record the padded count by image and severity, the image-level union count, and
    whether the detected tail was identical at all severities." Task 1 built the detector and
    the union; without this block the analysis would apply a mask that nothing in the output
    describes, and a reader of `summary.json` could not tell an image that lost 170 queries
    from one that lost none.

    The third fact is the one that looks redundant and is not. Padding *wanders* with severity
    rather than growing: on the pilot, 66 of 250 tuning images carry padding and the detected
    tail differs across severities in all 66, and the image that moves furthest runs 159, 165,
    170, 63, 8, 79 padded queries across the six levels. So `union_padded_count` is routinely
    larger than any single severity's count, `tail_identical_across_severities` is routinely
    `False`, and the two together are what tell a later reader that the union in
    `union_padded_query_ids` was a decision rather than a formality -- masking per severity
    would have swung that image's valid population between 141 and 292.

    Keys are strings and values are plain lists, ints and bools, because this dictionary is
    written into `summary.json`; an int-keyed severity would come back from JSON as a string
    and stop matching whatever compared against it.
    """
    tails = {
        str(severity): record["padded_query_ids"].tolist()
        for severity, record in sorted(records.items())
    }
    return {
        "union_padded_query_ids": padded.tolist(),
        "union_padded_count": int(padded.numel()),
        "padded_query_ids_by_severity": tails,
        "padded_count_by_severity": {key: len(value) for key, value in tails.items()},
        "tail_identical_across_severities": len({tuple(value) for value in tails.values()}) == 1,
    }


def analyze_deciles(inputs: DecileInputs) -> tuple[list[dict], dict]:
    """Every experiment and control row the design asks for, plus the padding bookkeeping.

    This is the only place in the analysis that holds more than one severity of an image at
    once, and that is the whole reason it exists as a separate function: the padding union, the
    frozen membership and every `clean_overlap` are cross-severity facts, and computing any of
    them one record at a time gives a different -- and wrong -- answer.

    The fairness rule `score_selection` cannot enforce for itself is enforced here. That
    function guarantees the two signals share one selection and one summary *within* a call; it
    cannot tell whether the confidence vector it is scoring is the vector that built the bins it
    is scoring. So each severity reads `record["query_confidence"]` exactly once, into `common`,
    and that one tensor is what `confidence_deciles`/`memberships_by_severity` ranked and what
    every row of that severity is scored against. Handing severity zero's vector to the scorer
    while binning severity five's would produce a complete, plausible table with no error in it.
    The key is `query_confidence` and never `confidence`: the cache's own field was reduced
    before the float16 cast and moves decile membership on 535 of the pilot's 1,500 records, so
    reaching for it here is a `KeyError` rather than a slightly different result.

    Four kinds of selection per severity -- 23 in all, 349 rows -- covering the design's five
    benchmark rows (spec 127-138):

    * the ten `dynamic` bins and the ten `frozen` bins, filtered -- the experiment, and
      benchmark 4's matched confidence control comes free with each of them;
    * `ALL_VALID_BENCHMARK`, every non-padded query -- benchmarks 2 and 3;
    * the lowest bin repeated `unfiltered` under both membership modes -- benchmark 5, the
      padding sensitivity control, scoped to `SENSITIVITY_BIN` and nothing else;
    * `ALL_QUERY_BENCHMARK`, all 300 queries at q90 with no confidence control -- benchmark 1,
      the existing published number, reproduced from the artifacts rather than asserted.

    None of those five collide: the row key carries `membership_mode`, `confidence_bin` and
    `padding_mode`, and each selection differs from the others in at least one of them. Dynamic
    and frozen stay separable for the same reason, which is the design's rule that they answer
    different questions and must not be combined into one score.

    `clean_overlap` is measured through `bin_overlap` for all four decile selections rather than
    asserted, including the frozen ones. Frozen membership is severity zero's by construction so
    its overlap is 1.0, but writing the constant would make a broken freeze look healthy in the
    one column that exists to detect movement; measuring it costs ten set intersections and
    turns that into a failing number.

    Two things this deliberately does not do. It does not re-check the loader's guarantees --
    six severities, ten valid queries, a distance row per key -- because `load_decile_inputs`
    fails on all of them before the first row is built, and a second copy would drift. And it
    does not reject duplicate output keys: it cannot produce one, and the check the design asks
    for belongs where rows arrive from anywhere, which is the summariser.

    Note for a reader of the padding-sensitivity plot: on an image with no padding at all --
    184 of the pilot's 250 -- the filtered and unfiltered bottom bins are the same queries and
    the control is a no-op by construction, not a null result. The effect lives entirely in the
    66 padded images and is diluted accordingly across a 250-image median.
    """
    rows: list[dict] = []
    images: dict[str, dict] = {}
    for image_id, records in sorted(inputs.records_by_image.items()):
        # One union mask per image, through the primitive that owns the rule. Building it
        # inline would also accept zero masks and boolean selection masks, which
        # `union_query_ids` refuses -- a boolean mask casts to the plausible query IDs 0 and 1.
        padded = union_query_ids(
            [record["padded_query_ids"] for _, record in sorted(records.items())]
        )
        images[str(image_id)] = _padding_diagnostics(records, padded)

        query_count = int(records[0]["query_count"])
        all_indices = torch.arange(query_count, dtype=torch.long)
        memberships = memberships_by_severity(records, padded)
        # Severity zero's *dynamic* bins, which is the reference spec 97 names literally: "the
        # Jaccard overlap between each dynamic bin and its severity-zero membership". Reusing
        # the `dynamic_overlap` that `memberships_by_severity` already computed would take the
        # reference from whatever that function chose to freeze instead, so the number recorded
        # here stays correct even if the frozen construction changes.
        clean_filtered = memberships[0]["dynamic"]
        # The unfiltered control needs its own severity-zero reference: bins built over all 300
        # queries, not the union-masked ten. Overlapping an unfiltered bin against a filtered
        # one would report movement that is really the padding mask.
        clean_unfiltered = confidence_deciles(
            records[0]["query_confidence"], all_indices,
            label=f"image {image_id} severity 0, unfiltered",
        )

        for severity, record in sorted(records.items()):
            partition = record["source_partition"]
            common = {
                "image_id": image_id,
                "severity": severity,
                "source_partition": partition,
                "query_confidence": record["query_confidence"],
                "query_scores_by_layer": inputs.distances[(image_id, severity, partition)],
                "layer_score_scales": inputs.layer_score_scales,
            }
            for mode in ("dynamic", "frozen"):
                bins = memberships[severity][mode]
                overlap = bin_overlap(bins, clean_filtered)
                for name in DECILE_NAMES:
                    rows.extend(score_selection(
                        **common, membership_mode=mode, confidence_bin=name,
                        padding_mode="filtered", indices=bins[name],
                        clean_overlap=overlap[name],
                    ))
            rows.extend(score_selection(
                **common,
                membership_mode=ALL_VALID_BENCHMARK[0],
                confidence_bin=ALL_VALID_BENCHMARK[1],
                padding_mode=ALL_VALID_BENCHMARK[2],
                indices=memberships[severity]["all_valid"],
                # The union mask is one set for the whole image, so this selection is the same
                # queries at every severity and its overlap with severity zero is 1.0 by the
                # construction of the mask rather than by anything measured here.
                clean_overlap=1.0,
            ))
            unfiltered = {
                "dynamic": confidence_deciles(
                    record["query_confidence"], all_indices,
                    label=f"image {image_id} severity {severity}, unfiltered",
                ),
                "frozen": clean_unfiltered,
            }
            for mode, bins in unfiltered.items():
                rows.extend(score_selection(
                    **common, membership_mode=mode, confidence_bin=SENSITIVITY_BIN,
                    padding_mode="unfiltered", indices=bins[SENSITIVITY_BIN],
                    clean_overlap=bin_overlap(bins, clean_unfiltered)[SENSITIVITY_BIN],
                ))
            rows.extend(score_selection(
                **common,
                membership_mode=ALL_QUERY_BENCHMARK[0],
                confidence_bin=ALL_QUERY_BENCHMARK[1],
                padding_mode=ALL_QUERY_BENCHMARK[2],
                indices=all_indices, clean_overlap=1.0,
                aggregations=("q90",), include_confidence=False,
            ))
    return rows, {"images": images}
