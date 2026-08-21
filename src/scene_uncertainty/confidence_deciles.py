from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor

from .metrics import jaccard_overlap


DECILE_NAMES = tuple(f"decile_{lower:02d}_{lower + 10:02d}" for lower in range(0, 100, 10))


def _rows_equal_to_last(values: Tensor) -> Tensor:
    """Mark every query row that is bit-for-bit identical to the final row of `values`.

    Comparison is exact `eq`, never a tolerance, because the thing being detected is a
    decoder placeholder that was *copied*, not a query that merely landed nearby. Two
    consequences follow and neither is a bug to be smoothed over.

    `nan` never equals itself, so a field whose final row holds a `nan` marks *no* row --
    including its own -- and the record reports no padding at all. That is the honest
    reading of "exactly identical": a slot whose contents are not even equal to themselves
    cannot be shown to repeat. It is also self-consistent, since a `nan` in the last row
    makes every earlier row unequal to it too, so the mask cannot come back mixed.

    In the other direction, the cache stores logits and fingerprints as float16, which keeps
    only about three and a half decimal digits. Two queries whose full-precision values
    differed in the fourth significant digit are stored as the same float16 and are
    indistinguishable from a real placeholder here; this function reads the cache as written
    and cannot recover what the cast discarded. What makes a collision implausible is width
    rather than precision, since a caller's record agrees across a 4-value box, an 80-value
    logit vector and a 335-value fingerprint per layer before a row is called repeated. On
    the pilot's 250 tuning images every run this detects ends at query 299, which is the
    signature the design describes, so nothing there is currently a rounding artefact.

    The two-dimension floor is deliberate. A one-dimensional field would still compare, by
    broadcasting against a scalar last element, but a `(query,)` field means the caller lost
    the feature axis somewhere upstream, and silently agreeing to compare scalars would turn
    that into a padding verdict instead of a question.
    """
    if values.ndim < 2:
        raise ValueError(f"query tensor needs query and feature dimensions, got {tuple(values.shape)}")
    return values.eq(values[-1]).reshape(values.shape[0], -1).all(dim=1)


def detect_padded_tail(record: dict) -> Tensor:
    """Query IDs of the longest all-identical suffix of a cached record, or an empty tensor.

    A suffix qualifies only when the box, the complete logit vector, and the persistence
    fingerprint at *every* cached decoder layer are all identical to the final query. One
    disagreeing layer is enough to end the run, which is what keeps a genuinely repeated
    prediction -- two queries that converged on the same box and class scores -- out of the
    padded set, since their fingerprints still differ.

    Only a run of two or more queries counts. A single final query is identical to itself by
    construction, so calling that padding would delete one real query from every image.

    Two limits worth reading before trusting the result. The layer requirement is only as
    strong as `record["layers"]`: a record cached without layers reduces this to box and
    logits agreeing, which is a much weaker test, and nothing here can tell that apart from a
    record that genuinely had no layers to check. And a record whose queries are *all*
    identical returns every query ID -- correct by the rule, but it leaves the caller with no
    valid queries at all, which is a condition to reject rather than to bin.
    """
    fields = [record["boxes"], record["logits"]]
    fields.extend(record["layers"][layer_id] for layer_id in sorted(record["layers"]))
    counts = {int(field.shape[0]) for field in fields}
    if len(counts) != 1:
        raise ValueError(f"query-count mismatch inside record: {sorted(counts)}")
    query_count = counts.pop()
    if query_count == 0:
        return torch.empty(0, dtype=torch.long)
    repeated = torch.stack([_rows_equal_to_last(field.cpu()) for field in fields]).all(dim=0)
    start = query_count - 1
    while start > 0 and bool(repeated[start - 1]):
        start -= 1
    if query_count - start < 2:
        return torch.empty(0, dtype=torch.long)
    return torch.arange(start, query_count, dtype=torch.long)


def _require_index_dtype(values: Tensor, what: str) -> None:
    """Refuse a tensor whose dtype makes a `(query,)` selection mask look like a list of IDs.

    A boolean keep/drop mask casts to a tensor of zeros and ones, which is a perfectly
    valid-looking pair of query IDs 0 and 1 and completely wrong; `uint8` is the same mask one
    `.to()` further on and fails identically. Floating point is refused for the other half of
    the same failure -- `.long()` truncates rather than rejects, so a vector of confidences
    handed in where IDs belong becomes a list of zeros instead of an error.

    Empty tensors pass. They carry no ID to misread, and the plainest way to write "nothing
    here" is a bare `[]`, which torch reads as float32.
    """
    misreadable = (
        values.dtype in (torch.bool, torch.uint8)
        or values.is_floating_point()
        or values.is_complex()
    )
    if values.numel() and misreadable:
        raise ValueError(f"{what} must hold integer query IDs, got dtype {values.dtype}")


def union_query_ids(id_tensors: Iterable[Tensor]) -> Tensor:
    """Merge query-ID masks into one sorted, de-duplicated index tensor.

    This is the rule the design states -- "take the union of padded query IDs detected across
    its six severities" -- expressed over the IDs themselves, so the one caller that streams
    the feature cache and keeps only the detected masks can apply it without holding six
    severities of fingerprints in memory. That caller -- `decile_analysis` -- is the only
    production one, and it calls this function. `union_padded_query_ids` is the same rule
    reached from whole records, kept as the record-level entry point and exercised by the
    tests; nothing in the pipeline reaches the union that way today.

    Unioning IDs is not the same as taking the longest mask, and the pilot is the reason to
    care. Padded tails happen to be nested suffixes there, so the two agree today; but the
    tail *wanders* with severity rather than growing -- one image runs 159, 165, 170, 63, 8,
    79 padded queries across severities 0 to 5 -- and the moment any producer emits a mask
    that is not a suffix, a longest-mask shortcut starts dropping IDs that a shorter mask was
    the only one to see. Union is the rule; nesting is a coincidence of the current detector.

    Three edge cases, each chosen rather than inherited:

    Zero masks raises. An image whose records went missing must not read as an image with
    nothing padded, since that difference is the whole valid-query mask and a silent empty
    would quietly restore every placeholder query to the lowest confidence bin.

    Masks that are *all* empty return empty, which is the ordinary case rather than an error:
    184 of the pilot's 250 tuning images have no padding at any severity.

    Selection-mask dtypes raise; see `_require_index_dtype`. Empty masks skip that check,
    because they carry no ID to misread and because the plainest way to say "nothing padded
    here" -- a bare `[]` -- is float32 before it is anything else.
    """
    masks = [torch.as_tensor(ids).reshape(-1) for ids in id_tensors]
    if not masks:
        raise ValueError("cannot build a query-ID union from zero masks")
    for mask in masks:
        _require_index_dtype(mask, "masks")
    return torch.cat([mask.long().cpu() for mask in masks]).unique()


def union_padded_query_ids(records: Iterable[dict]) -> Tensor:
    """One padding mask for an image, unioned over the severities it was cached at.

    The tail is not stable under blur: across the pilot's 250 tuning images, 66 carry padding
    and *all 66* detect a different tail at different severities. Nor does it simply grow --
    `union_query_ids` records the counts of the image that wanders furthest -- so per-severity
    masking would not merely shrink the valid set with blur, it would let the population wander
    in both directions. That matters because placeholder queries are close to zero confidence
    (mean 0.0045 against 0.0395 for valid queries, and 99 percent of them below the valid
    median), so they land in the lowest decile and would drag the bottom bin's membership
    around for a reason that has nothing to do with any signal. Unioning
    first fixes the mask once per image, so severity moves the features and never the
    population.

    This is the entry point for a caller that still holds whole records, and **no production
    caller does**: `decile_analysis` streams the cache and keeps only the detected masks, so it
    calls `union_query_ids` directly. This one stays because the record-level rule is the one
    the design states and a later loader may need it, and because both go through the same
    implementation so the union cannot drift between the two call sites.

    Severity is not read. This unions whatever records it is handed and does not check that
    they are the six expected blur levels, or that a severity appears once; establishing that
    belongs to the loader that assembled them, which is the only caller that can tell a
    missing severity from a severity that was never requested.
    """
    records = list(records)
    if not records:
        raise ValueError("cannot build a padding union from zero records")
    image_ids = {int(record["image_id"]) for record in records}
    if len(image_ids) != 1:
        raise ValueError(f"padding union needs one image, got {sorted(image_ids)}")
    counts = {int(record["logits"].shape[0]) for record in records}
    if len(counts) != 1:
        raise ValueError(f"query-count mismatch across severities: {sorted(counts)}")
    return union_query_ids([detect_padded_tail(record) for record in records])


def confidence_from_logits(logits: Tensor) -> Tensor:
    """Each query's largest sigmoid class score, recomputed rather than read from the cache.

    Blur records carry a `confidence` field beside their logits, and it is not this number.
    That field was reduced from the full-precision logits *before* those logits were cast to
    float16, so it cannot be recovered from what the cache actually stores. Over the pilot's
    450,000 tuning queries the two disagree by up to 1.020e-04, median 1.86e-05, and agree
    exactly for 0.03 percent of queries. A threshold policy would never notice. Deciles are
    ranks, so they do: with roughly 300 queries in bins of roughly 30, and a median gap of
    3.7e-04 between adjacent ranked confidences, a 1e-4 shift is enough to swap near-tied
    queries across a bin edge -- and it does, changing the membership of 535 of the 1,500
    tuning records. `query_policy.select_queries` recomputes the same way, which is also what
    keeps decile results comparable with the existing all-query benchmark. Read the stored
    field only to audit that gap, never to build a bin.

    `float()` before `sigmoid()` is not cosmetic either. Float16 spacing at 0.5 is about
    4.9e-04, so reducing in half precision quantises the scores far more coarsely than the
    logits themselves are quantised, manufacturing ties that the recorded logits do not
    contain. Sigmoid is monotone, so this never changes *which* class wins -- only how many
    queries the ranking can still tell apart.

    The two-dimension requirement is a real check rather than a formality: a `(query,)` input
    would reduce to a scalar under `amax(dim=-1)` and hand back one confidence for the whole
    image, and a `(batch, query, class)` input would silently bin one record's queries using
    another's scores.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape (query, class), got {tuple(logits.shape)}")
    return logits.float().sigmoid().amax(dim=-1).cpu()


def confidence_deciles(confidence: Tensor, valid_indices: Tensor, label: str = "") -> dict[str, Tensor]:
    """Split the valid queries into ten equal-count confidence bins, lowest first.

    Ties are the reason this is written with a stable sort over an already-sorted index
    tensor rather than a plain `argsort`. Confidence comes from float16 logits, so exact ties
    are ordinary rather than exotic: every one of the pilot's 1,500 tuning records has some,
    a median of 55 of its 300 queries share a confidence with another query, and the largest
    tie group runs to 257. A decile edge therefore falls inside a tie block routinely, and an
    unstable sort would return different bins for the same record on different runs. Sorting
    `valid_indices` first and then sorting *stably* by confidence makes ascending query ID the
    tie-break the design asks for, and makes the result reproducible.

    Bin sizes differ by at most one when the count is not divisible by ten; the remainder goes
    to the lowest bins, which is `tensor_split`'s rule and is arbitrary but fixed. Every valid
    query lands in exactly one bin, so concatenating the ten bins reproduces `valid_indices`.

    Each bin is returned in ascending *confidence* order, not ascending query ID. Membership
    is what the analysis consumes, so the order is incidental there, but it is stable and it
    means `bins[name][0]` is the least confident query of that bin.

    Fewer than ten valid queries raises rather than returning short or empty bins, because a
    record that cannot fill its bins is an invalid record and its score would silently mean
    something different from every other record's. `label` names that record in the message;
    `memberships_by_severity` fills it in, and a direct caller that leaves it empty gets the
    same check with a less useful message.

    Two inputs that look right and are not: a `(query, class)` logit tensor, which would rank
    rows instead of queries, and a boolean keep-mask where the index tensor belongs. Both
    raise. Non-finite confidence raises too -- `argsort` sorts `nan` to the end, which would
    quietly place a broken query in the top decile.
    """
    scores = confidence.float().cpu()
    if scores.ndim != 1:
        raise ValueError(
            f"confidence must be one score per query, got shape {tuple(confidence.shape)}"
        )
    _require_index_dtype(valid_indices, "valid query indices")
    valid = torch.sort(valid_indices.reshape(-1).long().cpu()).values
    named = f" for {label}" if label else ""
    if valid.numel() < len(DECILE_NAMES):
        raise ValueError(
            f"confidence deciles need at least ten valid queries, got {valid.numel()}{named}"
        )
    if valid.unique().numel() != valid.numel():
        raise ValueError(f"valid query indices must be unique{named}")
    if int(valid.min()) < 0 or int(valid.max()) >= scores.numel():
        raise ValueError(f"valid query index lies outside the confidence vector{named}")
    selected = scores.index_select(0, valid)
    if not bool(torch.isfinite(selected).all()):
        raise ValueError(f"confidence must be finite to rank queries{named}")
    order = torch.argsort(selected, stable=True)
    chunks = torch.tensor_split(valid.index_select(0, order), len(DECILE_NAMES))
    return {name: chunk for name, chunk in zip(DECILE_NAMES, chunks)}


def bin_overlap(bins: dict[str, Tensor], reference: dict[str, Tensor]) -> dict[str, float]:
    """Jaccard overlap of each bin against the bin of the same name in `reference`.

    This is the design's "dynamic-bin overlap with the clean bin", and it is a diagnostic
    rather than a score: it says how much of a severity's result comes from the queries
    moving between bins instead of the fingerprints moving inside them. At severity zero it
    is 1.0 for every bin by construction, so a value below 1.0 there means the two sides were
    not built from the same ordering.

    Comparing bin *names* rather than positions is deliberate. Two bin sets that disagree on
    names are not two views of the same partition, and quietly intersecting whichever names
    happened to match would report a high overlap for a comparison that never took place.
    """
    if set(bins) != set(reference):
        raise ValueError(
            f"cannot overlap bin sets that do not name the same bins: "
            f"{sorted(bins)} against {sorted(reference)}"
        )
    return {name: jaccard_overlap(values, reference[name]) for name, values in bins.items()}


def _severity_confidence(record: dict) -> Tensor:
    """The confidence vector of one severity record, from its logits or from `query_confidence`.

    Records reach this function in two shapes. A raw cache record carries `logits`, and those
    win outright, for the reason `confidence_from_logits` documents. A record that has already
    been streamed down to metadata has no logits left -- the design asks the loader to keep the
    confidence vector and discard the fingerprints -- so it must carry the derived vector under
    the key `query_confidence`.

    The key name is the enforcement, not a convention. The cache also stores a field called
    `confidence`, reduced from the full-precision logits before the float16 cast, and it is
    close enough to the right answer to look authoritative while being wrong enough to move
    decile membership on a third of the tuning records. A rule saying "do not read that field"
    protects only the call sites that remember the rule; naming the derived vector something
    the cache does not own means a caller who reaches for the stale field gets a `KeyError`
    where it is read directly, and the named error below where it is read through here.

    So a record holding only `confidence` is refused rather than used. That is not a
    conservative default -- it is the whole point: the stale field must never silently become
    a bin edge.
    """
    if "logits" in record:
        return confidence_from_logits(record["logits"])
    if "query_confidence" in record:
        return record["query_confidence"].float().cpu()
    if "confidence" in record:
        raise ValueError(
            "record carries the cache's stale `confidence` field, which was reduced from the "
            "full-precision logits before they were cast to float16 and must not build a bin; "
            "derive the vector with confidence_from_logits and pass it as `query_confidence`"
        )
    raise ValueError("a severity record must carry logits or a query_confidence vector")


def _record_label(record: dict, severity: int) -> str:
    image_id = record.get("image_id")
    if image_id is None:
        return f"severity {severity}"
    return f"image {int(image_id)} severity {severity}"


def memberships_by_severity(records_by_severity: dict[int, dict], padded: Tensor) -> dict[int, dict]:
    """Dynamic and clean-frozen decile memberships for one image, at every cached severity.

    `dynamic` re-sorts and rebuilds all ten bins at each severity: a policy that a single
    image could actually run, whose membership moves as confidence moves. `frozen` is severity
    zero's membership reused unchanged everywhere: a diagnostic, since a naturally corrupted
    image has no paired clean version, whose point is to hold the queries still so that any
    change in score comes from the fingerprints. The two answer different questions and the
    design forbids combining them into one number. `dynamic_overlap` is what separates them --
    Jaccard of each dynamic bin against its severity-zero self, 1.0 everywhere at severity
    zero and falling as queries change rank.

    `padded` is the *image-level union* of padded query IDs, not one severity's tail, and this
    function takes it as a single argument precisely so it cannot be anything else. The tail
    wanders with severity rather than growing, so per-severity masking would change the valid
    population between severities and move bin membership for a reason that has nothing to do
    with blur; `union_query_ids` over the per-severity masks is what produces the argument this
    wants, and `union_padded_query_ids` is the same rule for a caller that still holds records.

    `all_valid` is that same union-masked set, and it is the direct comparison with the
    existing all-query method -- every non-padded query, not all 300 and not a re-derivation
    from the ten bins.

    Each record supplies its confidence as `logits`, or as a `query_confidence` vector already
    derived with `confidence_from_logits`. The cache's own `confidence` field is refused; see
    `_severity_confidence` for why the key name is doing the enforcing.

    Severity zero must be present, since there is nothing to freeze without it, and every
    severity must report the same query count, since one valid mask is applied to all of them.
    Neither check knows that six severities were expected: whether a severity is missing or
    was never requested is a question only the loader that assembled this mapping can answer.
    """
    if 0 not in records_by_severity:
        raise ValueError("clean-frozen bins require severity zero")
    ordered = sorted(records_by_severity.items())
    confidence = {severity: _severity_confidence(record) for severity, record in ordered}
    counts = {int(values.numel()) for values in confidence.values()}
    if len(counts) != 1:
        raise ValueError(f"query-count mismatch across severities: {sorted(counts)}")
    query_count = counts.pop()
    _require_index_dtype(padded, "padded query IDs")
    padded = padded.reshape(-1).long().cpu()
    if padded.numel() and (int(padded.min()) < 0 or int(padded.max()) >= query_count):
        raise ValueError(f"padded query ID lies outside the {query_count}-query record")
    keep = torch.ones(query_count, dtype=torch.bool)
    keep[padded] = False
    valid = torch.where(keep)[0]
    frozen = confidence_deciles(
        confidence[0], valid, label=_record_label(records_by_severity[0], 0)
    )
    memberships = {}
    for severity, record in ordered:
        dynamic = confidence_deciles(
            confidence[severity], valid, label=_record_label(record, severity)
        )
        memberships[severity] = {
            "dynamic": dynamic,
            "frozen": {name: indices.clone() for name, indices in frozen.items()},
            "all_valid": valid.clone(),
            "dynamic_overlap": bin_overlap(dynamic, frozen),
        }
    return memberships
