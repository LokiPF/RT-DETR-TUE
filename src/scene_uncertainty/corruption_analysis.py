"""Cut one tuning artifact pair into deciles and into quintiles, and score both identically.

The published experiment asks whether persistence separates blurred scenes better than
confidence does *inside a low-confidence decile*. This module asks a question about that
answer rather than a new question about the data: whether the ten-way cut was load-bearing.
A decile of 300 queries is about 30 queries, thin enough that a scene summary of it is
noticeably a summary of few things; a quintile is twice as wide and carries the same claim at
half the resolution. If the two schemes rank the signals the same way, the resolution was
incidental and the coarser bucket is the safer one to deploy; if they do not, the decile result
depended on a bin width nobody chose for a reason.

That question is only answerable if the two schemes differ in *exactly one* respect, so
everything else is shared rather than re-derived. Both come from `confidence_buckets` over the
same confidence-ranked sequence of the same union-filtered valid queries, so a quintile is a
rank range of the decile ordering and not a second ranking that agrees wherever no tie broke;
both are scored by `score_selection`, so persistence and confidence are summarised over one
index tensor; and both run over the same loaded artifacts. The one thing this module adds to
each row is `bucket_scheme`, which is what lets a later reader group by resolution without
parsing a bin name.

Nothing here measures anything new, and it must not be able to. The blur cache was written
once and the query distances were searched once; both are read back through
`decile_analysis._load_scene_query_inputs`, which proves the two manifests against their own
content addresses and refuses any partition but tuning -- the same rules, the same order and
the same words as the published command, because a second loader would be a second chance to
join the wrong artifacts and no downstream row could tell. Recomputing either input is an
explicit non-goal: it costs GPU hours and a checkpoint to produce numbers that are already
saved, and it would produce *different* numbers, since the whole comparison rests on both
schemes reading one set of distances. So this module imports nothing that could run a detector,
build a bank or search one, and `test_corruption_analysis` scans its imports, its source text
and its namespace to keep it that way.

What is deliberately absent is as much of the design as what is here. The decile command scores
two selections this one does not, both filed under `shared`/`all_valid`: every non-padded query,
with the image's padding union removed, and the all-300-query benchmark, which is that same
label unfiltered and deliberately keeps the padded placeholders. Counted per severity that is 24
selections there against 22 here for the decile scheme. Both are benchmarks for the published
result and belong to it; neither is a confidence bucket, so no scheme cut them, and filing one
under `decile` or `quintile` would record a bucket comparison that never happened.
`decile_scoring.BUCKET_NAMES` refuses `all_valid` under a named scheme for that reason, which
makes both unreachable from here rather than merely unwritten.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .confidence_deciles import (
    DECILE_NAMES,
    QUINTILE_NAMES,
    bin_overlap,
    confidence_buckets,
    memberships_by_scheme_severity,
    union_query_ids,
)
from .decile_analysis import (
    DecileInputs,
    _load_scene_query_inputs,
    _padding_diagnostics,
)
from .decile_scoring import score_selection


CORRUPTION_INPUT_ARTIFACT_TYPE = "scene_corruption_sensitivity_inputs"
"""What `run_metadata` calls this analysis's inputs.

It names the *inputs*, not the results, because that is what the loader hands back: the same
join, read for a different experiment. The published command writes
`confidence_decile_scene_uncertainty` into the same field, so an analysis directory says which
of the two produced it without anyone having to infer it from the rows."""

SCHEMES = {
    "decile": DECILE_NAMES,
    "quintile": QUINTILE_NAMES,
}
"""The two resolutions, mapped to their bin names *in ascending confidence order*.

Ordered tuples rather than `decile_scoring.BUCKET_NAMES`, whose values are frozensets: that
mapping exists to validate a scheme/name pair and deliberately carries no order, while
everything below needs one. `names[0]` is the lowest-confidence bucket -- the only one the
unfiltered sensitivity control repeats -- and a frozenset has no first element to take at all.
Iterating one would also leave the row order of an otherwise identical run unreproducible,
since string hashing is seeded per process.
"""


def load_corruption_inputs(cache_value: str | Path, results_value: str | Path) -> DecileInputs:
    """The saved blur cache joined to the saved query distances, labelled for this experiment.

    A bare wrapper over the loader the published decile command runs through, differing only in
    the `artifact_type` it records. That is the point rather than an economy: this command is a
    second reader of artifacts whose provenance rules -- content addresses on both manifests,
    matching feature-cache ids, tuning partition only -- are the reason any row from them can be
    trusted, and a loader of its own would be a way around all of them.
    """
    return _load_scene_query_inputs(
        cache_value,
        results_value,
        artifact_type=CORRUPTION_INPUT_ARTIFACT_TYPE,
    )


def analyze_corruption_sensitivity(inputs: DecileInputs) -> tuple[list[dict], dict]:
    """Every filtered bucket of both schemes, plus each scheme's unfiltered lowest bucket.

    Images, then schemes, then severities. The outer level is fixed by the padding union, which
    is an image-level fact -- the detected tail wanders with severity rather than growing, so
    one mask is built per image and used at every severity, and `_padding_diagnostics` records
    it once beside the rows it shaped. The scheme sits above severity so that
    `memberships_by_scheme_severity` and the two severity-zero references are each built once
    per scheme rather than once per severity.

    Four kinds of selection make up a severity of one scheme: each of that scheme's buckets
    under `dynamic` and under `frozen` membership, filtered by the padding union, plus the
    lowest bucket alone repeated unfiltered under both modes. That is 11 selections per
    membership mode for the deciles and 6 for the quintiles -- 22 and 12 a severity, 34 in all,
    and 204 an image. On the pilot's three-layer cache a selection is 15 rows (three scene
    summaries, each across three decoder layers, their combination and one confidence control),
    so 3,060 rows per image and 765,000 over the 250 tuning images. The sensitivity control
    stays scoped to the lowest bucket for the reason `decile_analysis.SENSITIVITY_BIN` gives:
    widening it would publish a padding measurement per bucket that the design never asked for,
    and every extra row would be perfectly well-formed.

    The two signals share a selection because they are handed to `score_selection` in one call,
    with one `indices` tensor, one `query_confidence` and one aggregation set. That function
    guarantees the two are summarised over the same queries within a call and cannot check
    anything across calls, so the fairness of the comparison is a property of this loop. Each
    severity reads `record["query_confidence"]` exactly once, into `common`, and that same
    tensor is what ranked the buckets being scored -- binning one severity while scoring
    another's confidences would produce a complete table with no error in it.

    `clean_overlap` on the filtered rows is measured, never assumed, and measured against
    severity zero's *dynamic* buckets rather than the frozen ones that
    `memberships_by_scheme_severity` also returns. Frozen membership is severity zero's by
    construction, so its overlap is 1.0 -- but it is 1.0 against an independently rebuilt
    severity-zero partition, which is a claim that can fail, and writing the constant instead
    would make a broken freeze look healthy in the one column that exists to detect movement.

    The unfiltered control is the one place a constant is right, and only for `frozen`. Its
    selection *is* `clean_unfiltered`, the same mapping the overlap would compare it against,
    so `bin_overlap` there could only restate that a dict equals itself; there is no second
    construction between the two sides for a measurement to catch. `dynamic` is the opposite
    case -- its buckets are rebuilt from this severity's own confidences over all queries -- so
    its overlap is measured, and the unfiltered reference is severity zero's unfiltered
    buckets, never the filtered ones, because overlapping across the padding mask would report
    movement that is really the mask.

    What this does not do: re-check the loader's guarantees -- six severities, a distance row
    per key, enough valid queries to fill the buckets -- because `load_corruption_inputs` fails
    on all of them before the first row exists and a second copy would drift; and it does not
    reject duplicate row keys, because it cannot produce one. Every selection differs from
    every other in `bucket_scheme`, `confidence_bin`, `membership_mode` or `padding_mode`.

    The diagnostics carry `run_metadata` alongside the per-image padding record, which the
    decile analyzer leaves to its caller. This is the whole of what a summary of these rows has
    to say about where they came from, and passing it back with them keeps a later writer from
    having to hold the loaded inputs to find out.
    """
    rows: list[dict] = []
    images: dict[str, dict] = {}
    for image_id, records in sorted(inputs.records_by_image.items()):
        padded = union_query_ids(
            [record["padded_query_ids"] for _, record in sorted(records.items())]
        )
        images[str(image_id)] = _padding_diagnostics(records, padded)
        query_count = int(records[0]["query_count"])
        all_indices = torch.arange(query_count, dtype=torch.long)

        for scheme, names in SCHEMES.items():
            noun = "confidence deciles" if scheme == "decile" else "confidence quintiles"
            memberships = memberships_by_scheme_severity(
                records, padded, names=names, noun=noun
            )
            clean_filtered = memberships[0]["dynamic"]
            clean_unfiltered = confidence_buckets(
                records[0]["query_confidence"],
                all_indices,
                names=names,
                noun=noun,
                label=f"image {image_id} severity 0, unfiltered",
            )
            lowest = names[0]

            for severity, record in sorted(records.items()):
                common = {
                    "image_id": image_id,
                    "severity": severity,
                    "source_partition": record["source_partition"],
                    "query_confidence": record["query_confidence"],
                    "query_scores_by_layer": inputs.distances[
                        (image_id, severity, record["source_partition"])
                    ],
                    "layer_score_scales": inputs.layer_score_scales,
                    "bucket_scheme": scheme,
                }
                unfiltered = confidence_buckets(
                    record["query_confidence"],
                    all_indices,
                    names=names,
                    noun=noun,
                    label=f"image {image_id} severity {severity}, unfiltered",
                )
                unfiltered_overlap = bin_overlap(unfiltered, clean_unfiltered)
                for mode in ("dynamic", "frozen"):
                    bins = memberships[severity][mode]
                    overlap = bin_overlap(bins, clean_filtered)
                    for name in names:
                        rows.extend(score_selection(
                            **common,
                            membership_mode=mode,
                            confidence_bin=name,
                            padding_mode="filtered",
                            indices=bins[name],
                            clean_overlap=overlap[name],
                        ))

                    chosen = unfiltered if mode == "dynamic" else clean_unfiltered
                    rows.extend(score_selection(
                        **common,
                        membership_mode=mode,
                        confidence_bin=lowest,
                        padding_mode="unfiltered",
                        indices=chosen[lowest],
                        clean_overlap=(
                            unfiltered_overlap[lowest] if mode == "dynamic" else 1.0
                        ),
                    ))

    return rows, {"run": inputs.run_metadata, "images": images}
