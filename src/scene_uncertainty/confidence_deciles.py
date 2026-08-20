from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor


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


def union_query_ids(id_tensors: Iterable[Tensor]) -> Tensor:
    """Merge query-ID masks into one sorted, de-duplicated index tensor.

    This is the rule the design states -- "take the union of padded query IDs detected across
    its six severities" -- expressed over the IDs themselves, so the one caller that streams
    the feature cache and keeps only the detected masks can apply it without holding six
    severities of fingerprints in memory. `union_padded_query_ids` is the same rule reached
    from whole records.

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

    Boolean input raises. A `(query,)` selection mask of `True`/`False` casts to a tensor of
    zeros and ones, which is a perfectly valid-looking pair of query IDs and completely wrong.
    Empty masks skip that check, because they carry no ID to misread and because the plainest
    way to say "nothing padded here" -- a bare `[]` -- is float64 before it is anything else.
    """
    masks = [torch.as_tensor(ids).reshape(-1) for ids in id_tensors]
    if not masks:
        raise ValueError("cannot build a query-ID union from zero masks")
    for mask in masks:
        misreadable = mask.dtype is torch.bool or mask.is_floating_point() or mask.is_complex()
        if mask.numel() and misreadable:
            raise ValueError(f"masks must hold integer query IDs, got dtype {mask.dtype}")
    return torch.cat([mask.long().cpu() for mask in masks]).unique()


def union_padded_query_ids(records: Iterable[dict]) -> Tensor:
    """One padding mask for an image, unioned over the severities it was cached at.

    The tail is not stable under blur: across the pilot's 250 tuning images, 66 carry padding
    and *all 66* detect a different tail at different severities. Nor does it simply grow --
    one image runs 159, 165, 170, 63, 8, 79 padded queries across severities 0 to 5 -- so
    per-severity masking would not merely shrink the valid set with blur, it would let the
    population wander in both directions. That matters because placeholder queries are close
    to zero confidence (mean 0.0045 against 0.0395 for valid queries, and 99 percent of them
    below the valid median), so they land in the lowest decile and would drag the bottom
    bin's membership around for a reason that has nothing to do with any signal. Unioning
    first fixes the mask once per image, so severity moves the features and never the
    population.

    This is the entry point for a caller that still holds whole records. `union_query_ids`
    carries the actual rule and is the one to call once the fingerprints have been streamed
    away and only the detected masks remain; both go through the same implementation so the
    union cannot drift between the two call sites.

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
