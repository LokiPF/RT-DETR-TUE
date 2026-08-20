from __future__ import annotations

import torch
from torch import Tensor

# `_require_index_dtype` is imported rather than re-spelled so that a boolean keep-mask is
# refused with the same message wherever query IDs are accepted; two copies of that rule
# would be two chances for one of them to drift into accepting a mask.
from .confidence_deciles import DECILE_NAMES, _require_index_dtype
from .query_policy import aggregate_scores


DECILE_AGGREGATIONS = ("mean", "q90", "top20_mean")
"""The three scene summaries the design names. `query_policy.aggregate_scores` knows more --
`median`, `weighted_mean` -- and this module refuses them: a fourth summary appearing in the
results table would be one the matched comparison was never run for."""

CONFIDENCE_SCOPE = "confidence"
COMBINED_SCOPE = "combined"
PRIMARY_SCORE_SCOPE = "layer_2"
"""Layer 2 was the strongest persistence scope in the pilot and is the design's primary
scope. Layers 0 and 1 and `combined` are secondary diagnostics; naming the primary once here
keeps a later ranking step from quietly promoting whichever scope happens to win."""

MEMBERSHIP_MODES = ("dynamic", "frozen", "shared")
"""How a selection's membership was chosen -- the design's two, plus the one it implies.

`dynamic` re-sorts and rebuilds the bins at every severity; `frozen` reuses severity zero's.
`shared` is neither: it is a selection that no confidence ranking produced and that is the same
query set at every severity -- `all_valid`, and the all-300-query benchmark. Calling those
`dynamic` would drop a fixed selection into the group that measures how much membership moves,
and calling them `frozen` would claim they were copied from severity zero. The design forbids
combining dynamic and frozen into one score, and a third label is what keeps a selection that
is neither from having to borrow one of their names."""

PADDING_MODES = ("filtered", "unfiltered")
CONFIDENCE_BINS = DECILE_NAMES + ("all_valid",)


def _checked_indices(indices: Tensor, query_count: int) -> Tensor:
    """The selected query IDs, or a refusal -- never a selection that silently means something else.

    Four ways a caller can hand over something that looks like a selection and is not, each
    of which would score a real number for the wrong queries rather than fail:

    A boolean keep-mask casts to a tensor of zeros and ones, which reads as the perfectly
    plausible query IDs 0 and 1; `_require_index_dtype` carries that rule and the reason.

    An empty selection has no scene score. Returning `nan` would be defensible in isolation,
    but a decile of a valid record is never empty -- `confidence_deciles` refuses fewer than
    ten valid queries precisely so every bin has at least one member -- so an empty selection
    here means the caller lost the selection somewhere, and a `nan` row would hide that inside
    an otherwise full results table.

    A repeated ID double-weights one query. Deciles cannot produce one, so a repeat means the
    caller concatenated two bins or re-indexed twice, and the resulting mean would be a real
    number computed over a population nobody chose.

    An out-of-range ID is caught here rather than left to `index_select`, whose message names
    neither the vector nor the offending ID.
    """
    _require_index_dtype(indices, "selected query IDs")
    selected = indices.reshape(-1).long().cpu()
    if selected.numel() == 0:
        raise ValueError("cannot score an empty confidence-bin selection")
    if selected.unique().numel() != selected.numel():
        raise ValueError("selected query IDs must be unique; a repeated ID double-weights a query")
    if int(selected.min()) < 0 or int(selected.max()) >= query_count:
        raise ValueError(
            f"selected query ID lies outside the {query_count}-query record: "
            f"{int(selected.min())}..{int(selected.max())}"
        )
    return selected


def _checked_aggregations(aggregations: tuple[str, ...]) -> tuple[str, ...]:
    """The scene summaries to run, refusing an unknown, repeated, or absent one.

    A repeat is rejected because the row key of this table is
    (image, severity, partition, membership mode, bin, padding, signal, scope, summary), so
    `("mean", "mean")` emits two rows that are indistinguishable downstream -- the design's
    "output result keys would be duplicated" failure, reached from inside a single call.

    An empty tuple is rejected rather than returning zero rows: a caller that scored nothing
    would otherwise see a successful call and an image that quietly never appears in the table.
    """
    requested = tuple(aggregations)
    if not requested:
        raise ValueError("scoring needs at least one scene summary")
    unknown = [name for name in requested if name not in DECILE_AGGREGATIONS]
    if unknown:
        raise ValueError(
            f"unknown scene summary {unknown}; the design's summaries are "
            f"{list(DECILE_AGGREGATIONS)}"
        )
    if len(set(requested)) != len(requested):
        raise ValueError(f"scene summaries must not repeat, got {list(requested)}")
    return requested


def _checked_layers(
    query_scores_by_layer: dict[int, Tensor],
    layer_score_scales: dict[int, dict[str, Tensor]],
    query_count: int,
) -> None:
    """Refuse a persistence input that cannot be scored, or can be scored into an infinity.

    The empty case is the one worth stating: with no layers there is nothing to average for
    the `combined` scope, and dividing by zero layers is the difference between a clear error
    and a `ZeroDivisionError` from the middle of a row builder.

    A zero or non-finite clean-distance `scale` is the other. `knn.fit_clean_distance_scale`
    already floors its inter-quartile spread at 1e-6, so a zero can only arrive from a
    hand-built or hand-edited fit -- and it would not raise, it would divide the `combined`
    score into `inf` and carry it into a median Spearman as a perfectly ordinary largest value.
    """
    if not query_scores_by_layer:
        raise ValueError("cannot score a selection with no persistence layer")
    if any(int(values.numel()) != query_count for values in query_scores_by_layer.values()):
        raise ValueError("persistence and confidence query count disagree")
    if set(query_scores_by_layer) != set(layer_score_scales):
        raise ValueError(
            f"persistence layers and clean-distance scales disagree: "
            f"{sorted(query_scores_by_layer)} against {sorted(layer_score_scales)}"
        )
    for layer_id, state in sorted(layer_score_scales.items()):
        center, scale = float(state["center"]), float(state["scale"])
        if not torch.isfinite(torch.tensor([center, scale])).all():
            raise ValueError(f"clean-distance center and scale must be finite for layer {layer_id}")
        if scale <= 0.0:
            raise ValueError(
                f"clean-distance scale must be positive, got {scale} for layer {layer_id}"
            )


def _selected_values(values: Tensor, indices: Tensor, what: str) -> Tensor:
    """The selected entries of a per-query vector, refusing a non-finite one.

    Non-finite entries are refused instead of averaged because every summary here propagates
    them: a single `nan` distance turns a scene score into `nan`, a single `inf` turns `q90`
    and `top20_mean` into `inf`, and both then travel into a per-image Spearman as a value
    rather than as a missing measurement. The design's rule is that missing data must never be
    silently replaced -- and a `nan` score is the same silence wearing a different face.

    Only the *selected* entries are checked. A bin is scored from its own members, and a
    broken query outside the bin has no vote in it; checking the whole vector would reject a
    record for a slot this selection never reads.
    """
    selected = values.float().cpu().index_select(0, indices)
    if not bool(torch.isfinite(selected).all()):
        raise ValueError(f"{what} must be finite for every selected query")
    return selected


def _checked_confidence(query_confidence: Tensor, indices: Tensor) -> Tensor:
    """`1 - confidence` for the selected queries: a direction flip, and nothing else.

    Larger means less confident. Nothing here is trained, fitted or calibrated, and the
    result is not a probability: a scene scoring 0.8 has *not* been found 80 percent likely
    to be corrupted, it is a scene whose selected queries averaged 0.2 maximum class score.
    The subtraction is exact -- not clipped away from the ends, not rescaled to fill [0, 1] --
    so that the confidence control differs from the persistence signal only in what it
    measures.

    `query_confidence` is the vector derived by `confidence_deciles.confidence_from_logits`,
    and the parameter is named for the key the design's slim record uses. It is emphatically
    *not* the blur cache's own `confidence` field: that one was reduced from the
    full-precision logits before they were cast to float16, and using it moves decile
    membership on 535 of the pilot's 1,500 tuning records. The name is the guard -- a caller
    reaching for the cache field has to rename it on the way in, which is the moment to notice.

    Values outside [0, 1] are refused. A sigmoid maximum cannot leave that interval, so a
    value outside it means raw logits arrived where confidences belong -- and `1 - logit` is a
    number, plausibly ordered, and completely meaningless.
    """
    selected = _selected_values(query_confidence, indices, "query confidence")
    if float(selected.min()) < 0.0 or float(selected.max()) > 1.0:
        raise ValueError(
            f"query confidence must be a sigmoid maximum in [0, 1], got "
            f"{float(selected.min())}..{float(selected.max())}; raw logits are not confidences"
        )
    return 1.0 - selected


def score_selection(
    *,
    image_id: int,
    severity: int,
    source_partition: str,
    membership_mode: str,
    confidence_bin: str,
    padding_mode: str,
    indices: Tensor,
    query_confidence: Tensor,
    query_scores_by_layer: dict[int, Tensor],
    layer_score_scales: dict[int, dict[str, Tensor]],
    clean_overlap: float,
    aggregations: tuple[str, ...] = DECILE_AGGREGATIONS,
    include_confidence: bool = True,
) -> list[dict]:
    """Score one confidence-bin selection with both uncertainty signals, matched exactly.

    The fairness of the whole experiment lives in this function, and it is a fairness that
    can only be lost by accident: persistence and confidence are scored over *one* index
    tensor and through *one* summary function, computed once here and shared, rather than
    each signal deriving its own. Two separately-derived selections would still produce a
    results table, still correlate with severity, and still support a confident sentence about
    which signal won -- while comparing a summary of one population against a summary of
    another. That is why `indices` is an argument instead of something this function selects,
    and why both signals go through `query_policy.aggregate_scores`: the existing all-query
    benchmark comes out of that exact code, and a second implementation of `q90` or
    `top20_mean` here would let the new numbers drift from the benchmark they are measured
    against.

    `query_scores_by_layer` is the *saved* per-query mean distance to the five nearest clean
    fingerprints, read straight out of the raw-kNN result artifact, whose records carry this
    key already. Recomputing kNN is an explicit non-goal of the design, so this module imports
    nothing that could search a bank and cannot be made to by any argument.

    Row shape. One row per (signal, scope, summary). Persistence has a scope per decoder layer
    plus `combined`, the existing equal-mean-after-clean-scaling combination -- the same
    arithmetic as `evaluate.score_cached_record`, so the two agree by construction rather than
    by resemblance. Confidence has no decoder-layer scope and is written *once* per summary,
    under the scope name `confidence`. Pairing it against each persistence scope is a
    presentation step for the comparison table; storing it three times would triple every
    later count of confidence rows and make one measurement look like independent agreement
    between three. `include_confidence=False` is for the caller that has already stored it.

    `membership_mode`, `confidence_bin` and `padding_mode` are checked against closed
    vocabularies. They are group-by keys downstream, so a typo would not fail -- it would open
    a phantom group with a handful of rows in it and silently subtract those rows from the
    group they belonged to.

    `clean_overlap` is recorded, not derived: it is the design's dynamic-bin Jaccard against
    the severity-zero membership, and it is a diagnostic for how much of a result comes from
    queries changing bin rather than fingerprints moving. Under frozen membership it is 1.0 by
    construction, so a caller that files the dynamic overlap on a frozen row is recording a
    number about a different selection than the one that was scored.

    `source_partition` is recorded, not enforced. The design forbids scoring the held-out test
    partition *for this experiment*, and it also anticipates a held-out run once the tuning
    conclusion is reviewed; the loader that assembles records is the layer that can tell
    "wrong partition" from "the run that was finally authorised", so the gate belongs there.
    """
    if membership_mode not in MEMBERSHIP_MODES:
        raise ValueError(
            f"unknown membership mode {membership_mode!r}, expected {list(MEMBERSHIP_MODES)}"
        )
    if padding_mode not in PADDING_MODES:
        raise ValueError(
            f"unknown padding mode {padding_mode!r}, expected {list(PADDING_MODES)}"
        )
    if confidence_bin not in CONFIDENCE_BINS:
        raise ValueError(
            f"unknown confidence bin {confidence_bin!r}, expected {list(CONFIDENCE_BINS)}"
        )
    requested = _checked_aggregations(aggregations)
    if query_confidence.ndim != 1:
        raise ValueError(
            f"query confidence must be one score per query, got shape {tuple(query_confidence.shape)}"
        )
    query_count = int(query_confidence.numel())
    _checked_layers(query_scores_by_layer, layer_score_scales, query_count)
    selected = _checked_indices(indices, query_count)

    confidence_uncertainty = _checked_confidence(query_confidence, selected)
    persistence = {
        layer_id: _selected_values(values, selected, f"persistence distance at layer {layer_id}")
        for layer_id, values in sorted(query_scores_by_layer.items())
    }
    provenance = {
        "image_id": int(image_id),
        "severity": int(severity),
        "source_partition": source_partition,
        "membership_mode": membership_mode,
        "confidence_bin": confidence_bin,
        "padding_mode": padding_mode,
        "selected_count": int(selected.numel()),
        "clean_overlap": float(clean_overlap),
    }
    selected_query_ids = selected.tolist()

    def row(signal: str, scope: str, aggregation: str, score: float) -> dict:
        # Each row gets its own copy of the ID list: one shared list handed to every row would
        # let a downstream edit to one row rewrite the recorded membership of all of them.
        return {
            **provenance,
            "selected_query_ids": list(selected_query_ids),
            "signal": signal,
            "score_scope": scope,
            "aggregation": aggregation,
            "score": float(score),
        }

    rows: list[dict] = []
    for aggregation in requested:
        if include_confidence:
            rows.append(
                row(
                    "confidence",
                    CONFIDENCE_SCOPE,
                    aggregation,
                    float(aggregate_scores(confidence_uncertainty, aggregation)),
                )
            )
        layer_scores = {
            layer_id: float(aggregate_scores(values, aggregation))
            for layer_id, values in persistence.items()
        }
        scaled = [
            (score - float(layer_score_scales[layer_id]["center"]))
            / float(layer_score_scales[layer_id]["scale"])
            for layer_id, score in layer_scores.items()
        ]
        for layer_id, score in layer_scores.items():
            rows.append(row("persistence", f"layer_{layer_id}", aggregation, score))
        rows.append(row("persistence", COMBINED_SCOPE, aggregation, sum(scaled) / len(scaled)))
    return rows
