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

    The union is over query IDs rather than tail lengths. For suffixes of a fixed-length
    sequence the two happen to agree, but nothing here guarantees the inputs stay suffixes,
    and a mask of IDs is what the caller needs anyway.

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
    padded = sorted({int(index) for record in records for index in detect_padded_tail(record)})
    return torch.tensor(padded, dtype=torch.long)
