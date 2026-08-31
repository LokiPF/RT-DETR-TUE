from __future__ import annotations

import csv
import hashlib
import json
from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from .artifacts import iter_records, load_manifest as load_artifact_manifest
from .corruptions.imagecorruptions import ADDITIONAL_IMAGECORRUPTIONS
from .evaluation import binary_auroc
from .manifests import (
    load_manifest,
    manifest_digest,
    validate_disjoint,
)
from .scoring import detect_padded_tail, union_padded_query_ids


BANK_VARIANTS = (
    "matched",
    "background",
    "balanced",
    "all_valid",
    "confidence_0_5",
)
DISTANCES = (
    "mean_5_euclidean",
    "fifth_neighbor_euclidean",
    "mean_5_standardized_euclidean",
    "mean_5_cosine",
)
AGGREGATIONS = (
    "mean_all",
    "q90_all",
    "top20_mean_all",
    "top_confidence_query",
    "confidence_weighted_mean",
)
FAMILIES = ("gaussian_blur", *ADDITIONAL_IMAGECORRUPTIONS)
BANK_ROW_CHUNK_SIZE = 512


@dataclass(frozen=True)
class StudyConfig:
    reference_count: int = 1_000
    evaluation_count: int = 250
    selection_count: int = 150
    validation_count: int = 100
    levels: tuple[int, ...] = (0, 4, 5)
    bank_capacity: int = 2_000
    primary_seed: int = 44
    sensitivity_seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    bootstrap_draws: int = 2_000
    bootstrap_seed: int = 20_260_821
    families: tuple[str, ...] = FAMILIES


@dataclass(frozen=True)
class CandidateRows:
    vectors: Tensor
    matched: Tensor
    confidence: Tensor


@dataclass(frozen=True)
class ImageSplit:
    selection: tuple[str, ...]
    validation: tuple[str, ...]


@dataclass(frozen=True)
class Policy:
    bank: str
    distance: str
    aggregation: str
    seed: int

    @property
    def policy_id(self) -> str:
        return "-".join((self.bank, self.distance, self.aggregation, str(self.seed)))


@dataclass(frozen=True)
class Bank:
    vectors: Tensor
    mean: Tensor | None
    scale: Tensor | None
    matched_count: int
    background_count: int


@dataclass(frozen=True)
class ImageScore:
    image_id: str
    family: str
    severity: int
    valid_query_ids: tuple[int, ...]
    fingerprint_scores: dict[str, float]
    raw_confidence: float
    confidence_score: float
    entropy_score: float


@dataclass(frozen=True)
class FamilyResult:
    family: str
    level4: float
    level5: float
    strong: float


@dataclass(frozen=True)
class ConditionalResult:
    family: str
    severity: int
    point: float | None
    pair_count: int


@dataclass(frozen=True)
class ScoreInterval:
    point: float | None
    lower: float | None
    upper: float | None
    count: int


@dataclass(frozen=True)
class ValidationBootstrap:
    fingerprint: ScoreInterval
    confidence: ScoreInterval
    entropy: ScoreInterval
    fingerprint_minus_confidence: ScoreInterval
    fingerprint_minus_entropy: ScoreInterval
    conditional: ScoreInterval


@dataclass(frozen=True)
class SeedSummary:
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class StudyRunResult:
    selected_policy: Policy
    levels_read: frozenset[int]
    reported_families: frozenset[str]
    output_dir: Path


def split_image_ids(image_ids, selection_count: int) -> ImageSplit:
    unique_ids = {str(image_id) for image_id in image_ids}
    if selection_count < 0 or selection_count > len(unique_ids):
        raise ValueError(
            f"selection count must be between 0 and {len(unique_ids)}; "
            f"got {selection_count}"
        )

    def split_key(image_id: str) -> tuple[bytes, str]:
        digest = hashlib.sha256(f"strong-study:{image_id}".encode()).digest()
        return digest, image_id

    ordered = tuple(sorted(unique_ids, key=split_key))
    return ImageSplit(ordered[:selection_count], ordered[selection_count:])


def reservoir_indices(population_size: int, capacity: int, seed: int) -> Tensor:
    if capacity <= 0:
        raise ValueError(f"reservoir capacity must be positive; got {capacity}")
    if population_size < capacity:
        raise ValueError(
            f"reservoir requires {capacity} rows but only {population_size} are available"
        )

    reservoir = np.arange(capacity, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for index in range(capacity, population_size):
        replacement = int(rng.integers(0, index + 1))
        if replacement < capacity:
            reservoir[replacement] = index
    return torch.from_numpy(reservoir.copy())


def _box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    center_x, center_y, width, height = boxes.unbind(dim=-1)
    return torch.stack(
        (
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        ),
        dim=-1,
    )


def _generalized_box_iou(left: Tensor, right: Tensor) -> Tensor:
    intersection_left = torch.maximum(left[:, None, :2], right[None, :, :2])
    intersection_right = torch.minimum(left[:, None, 2:], right[None, :, 2:])
    intersection_size = (intersection_right - intersection_left).clamp_min(0)
    intersection = intersection_size.prod(dim=-1)

    left_area = (left[:, 2:] - left[:, :2]).clamp_min(0).prod(dim=-1)
    right_area = (right[:, 2:] - right[:, :2]).clamp_min(0).prod(dim=-1)
    union = left_area[:, None] + right_area[None, :] - intersection
    epsilon = torch.finfo(left.dtype).eps
    iou = intersection / union.clamp_min(epsilon)

    enclosing_left = torch.minimum(left[:, None, :2], right[None, :, :2])
    enclosing_right = torch.maximum(left[:, None, 2:], right[None, :, 2:])
    enclosing_area = (enclosing_right - enclosing_left).clamp_min(0).prod(dim=-1)
    return iou - (enclosing_area - union) / enclosing_area.clamp_min(epsilon)


def match_reference_queries(
    record: Mapping,
    annotations: Sequence[Mapping],
    valid_query_ids: Tensor,
    width: int,
    height: int,
    category_ids: tuple[int, ...],
) -> Tensor:
    valid_ids = valid_query_ids.detach().cpu().long()
    matched = torch.zeros(valid_ids.numel(), dtype=torch.bool)
    targets = [annotation for annotation in annotations if not annotation.get("iscrowd", 0)]
    if not targets or valid_ids.numel() == 0:
        return matched
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")

    category_index = {category_id: index for index, category_id in enumerate(category_ids)}
    try:
        target_classes = torch.tensor(
            [category_index[int(annotation["category_id"])] for annotation in targets],
            dtype=torch.long,
        )
    except KeyError as error:
        raise ValueError(f"unknown annotation category ID {error.args[0]}") from error

    target_boxes = []
    for annotation in targets:
        x, y, box_width, box_height = annotation["bbox"]
        target_boxes.append(
            (
                (float(x) + float(box_width) / 2) / width,
                (float(y) + float(box_height) / 2) / height,
                float(box_width) / width,
                float(box_height) / height,
            )
        )
    targets_cxcywh = torch.tensor(target_boxes, dtype=torch.float32)

    logits = record["logits"].detach().cpu().float().index_select(0, valid_ids)
    boxes = record["boxes"].detach().cpu().float().index_select(0, valid_ids)
    probabilities = logits.sigmoid()
    negative_cost = (
        0.75 * probabilities.pow(2) * (-(1 - probabilities + 1e-8).log())
    )
    positive_cost = (
        0.25 * (1 - probabilities).pow(2) * (-(probabilities + 1e-8).log())
    )
    class_cost = positive_cost[:, target_classes] - negative_cost[:, target_classes]
    box_cost = torch.cdist(boxes, targets_cxcywh, p=1)
    giou_cost = -_generalized_box_iou(
        _box_cxcywh_to_xyxy(boxes), _box_cxcywh_to_xyxy(targets_cxcywh)
    )
    cost = 2 * class_cost + 5 * box_cost + 2 * giou_cost
    assigned_queries, _ = linear_sum_assignment(cost.numpy())
    matched[torch.as_tensor(assigned_queries, dtype=torch.long)] = True
    return matched


def build_reference_candidates(
    records,
    *,
    annotations_by_image: Mapping[str, Sequence[Mapping]],
    image_sizes: Mapping[str, tuple[int, int]],
    category_ids: tuple[int, ...],
) -> CandidateRows:
    ordered_records = sorted(records, key=lambda record: str(record["image_id"]))
    image_ids = [str(record["image_id"]) for record in ordered_records]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("reference records must have unique string image IDs")

    annotation_lookup = {
        str(image_id): annotations for image_id, annotations in annotations_by_image.items()
    }
    size_lookup = {str(image_id): size for image_id, size in image_sizes.items()}
    vector_parts = []
    matched_parts = []
    confidence_parts = []
    for image_id, record in zip(image_ids, ordered_records, strict=True):
        if int(record.get("severity", 0)) != 0:
            raise ValueError("reference candidate records must be clean severity 0")
        query_count = int(record["persistence"].shape[0])
        padded_ids = detect_padded_tail(dict(record))
        keep = torch.ones(query_count, dtype=torch.bool)
        keep[padded_ids] = False
        valid_ids = torch.arange(query_count, dtype=torch.long)[keep]
        if image_id not in size_lookup:
            raise ValueError(f"missing image size for reference image {image_id!r}")
        if image_id not in annotation_lookup:
            raise ValueError(f"missing annotations for reference image {image_id!r}")
        width, height = size_lookup[image_id]

        vectors = (
            record["persistence"]
            .detach()
            .index_select(0, valid_ids.to(record["persistence"].device))
            .cpu()
            .float()
            .contiguous()
        )
        if not bool(torch.isfinite(vectors).all()):
            raise ValueError("reference persistence must remain finite in float32")
        vector_parts.append(vectors)
        matched_parts.append(
            match_reference_queries(
                record,
                annotation_lookup[image_id],
                valid_query_ids=valid_ids,
                width=width,
                height=height,
                category_ids=category_ids,
            )
        )
        confidence_parts.append(
            record["logits"]
            .detach()
            .float()
            .sigmoid()
            .amax(dim=-1)
            .index_select(0, valid_ids.to(record["logits"].device))
            .cpu()
        )

    if not vector_parts or sum(part.shape[0] for part in vector_parts) == 0:
        raise ValueError("reference records contain no valid candidate rows")
    return CandidateRows(
        vectors=torch.cat(vector_parts),
        matched=torch.cat(matched_parts),
        confidence=torch.cat(confidence_parts),
    )


def _validate_candidates(candidates: CandidateRows) -> tuple[Tensor, Tensor, Tensor]:
    vectors = candidates.vectors.detach().cpu().float()
    matched = candidates.matched.detach().cpu().bool()
    confidence = candidates.confidence.detach().cpu().float()
    if vectors.ndim != 2 or matched.ndim != 1 or confidence.ndim != 1:
        raise ValueError("candidate vectors must be 2D and labels must be 1D")
    if vectors.shape[0] != matched.numel() or vectors.shape[0] != confidence.numel():
        raise ValueError("candidate vectors, matched flags, and confidence must align")
    if not bool(torch.isfinite(vectors).all()) or not bool(torch.isfinite(confidence).all()):
        raise ValueError("candidate vectors and confidence must be finite")
    return vectors, matched, confidence


def _derived_reservoir_seed(seed: int, population: str) -> int:
    digest = hashlib.sha256(f"{seed}:{population}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % (2**63)


def _require_population(variant: str, population: str, required: int, available: int) -> None:
    if available < required:
        raise ValueError(
            f"bank variant {variant!r} requires {required} {population} rows "
            f"but only {available} are available"
        )


def _population_moments(vectors: Tensor) -> tuple[Tensor, Tensor]:
    values = vectors.double()
    mean = values.mean(dim=0)
    scale = values.std(dim=0, correction=0)
    if bool((scale == 0).any()):
        raise ValueError("standardized Euclidean requires non-zero reference variance")
    return mean.float(), scale.float()


def _balanced_moments(matched_vectors: Tensor, background_vectors: Tensor) -> tuple[Tensor, Tensor]:
    left = matched_vectors.double()
    right = background_vectors.double()
    mean = (left.mean(dim=0) + right.mean(dim=0)) / 2
    second_moment = (left.square().mean(dim=0) + right.square().mean(dim=0)) / 2
    scale = (second_moment - mean.square()).clamp_min(0).sqrt()
    if bool((scale == 0).any()):
        raise ValueError("standardized Euclidean requires non-zero reference variance")
    return mean.float(), scale.float()


def build_bank(
    candidates: CandidateRows,
    *,
    variant: str,
    distance: str,
    capacity: int,
    seed: int,
) -> Bank:
    if variant not in BANK_VARIANTS:
        raise ValueError(f"unknown bank variant {variant!r}")
    if distance not in DISTANCES:
        raise ValueError(f"unknown distance {distance!r}")
    if capacity <= 0:
        raise ValueError(f"bank capacity must be positive; got {capacity}")

    vectors, matched, confidence = _validate_candidates(candidates)
    if variant == "balanced":
        if capacity % 2:
            raise ValueError(
                f"bank variant 'balanced' requires an even capacity; got {capacity}"
            )
        allocation = capacity // 2
        matched_ids = torch.nonzero(matched, as_tuple=False).flatten()
        background_ids = torch.nonzero(~matched, as_tuple=False).flatten()
        _require_population("balanced", "matched", allocation, matched_ids.numel())
        _require_population("balanced", "background", allocation, background_ids.numel())
        sampled_ids = torch.cat(
            (
                matched_ids.index_select(
                    0,
                    reservoir_indices(
                        matched_ids.numel(),
                        allocation,
                        _derived_reservoir_seed(seed, "matched"),
                    ),
                ),
                background_ids.index_select(
                    0,
                    reservoir_indices(
                        background_ids.numel(),
                        allocation,
                        _derived_reservoir_seed(seed, "background"),
                    ),
                ),
            )
        )
    else:
        masks = {
            "matched": matched,
            "background": ~matched,
            "all_valid": torch.ones_like(matched),
            "confidence_0_5": confidence >= 0.5,
        }
        population_names = {
            "matched": "matched",
            "background": "background",
            "all_valid": "valid",
            "confidence_0_5": "confidence>=0.5",
        }
        eligible_ids = torch.nonzero(masks[variant], as_tuple=False).flatten()
        _require_population(variant, population_names[variant], capacity, eligible_ids.numel())
        sampled_ids = eligible_ids.index_select(
            0, reservoir_indices(eligible_ids.numel(), capacity, seed)
        )

    sampled = vectors.index_select(0, sampled_ids)
    mean = None
    scale = None
    if distance == "mean_5_standardized_euclidean":
        if variant == "balanced":
            mean, scale = _balanced_moments(
                vectors.index_select(0, matched_ids),
                vectors.index_select(0, background_ids),
            )
        else:
            mean, scale = _population_moments(vectors.index_select(0, eligible_ids))
        sampled = (sampled - mean) / scale

    sampled_matched = matched.index_select(0, sampled_ids)
    return Bank(
        vectors=sampled.contiguous(),
        mean=mean,
        scale=scale,
        matched_count=int(sampled_matched.sum()),
        background_count=int((~sampled_matched).sum()),
    )


def _nearest_five(queries: Tensor, bank_vectors: Tensor, *, cosine: bool) -> Tensor:
    best = torch.full(
        (queries.shape[0], 5),
        float("inf"),
        dtype=queries.dtype,
        device=queries.device,
    )
    for chunk in bank_vectors.split(BANK_ROW_CHUNK_SIZE):
        if cosine:
            distances = (1 - queries @ chunk.T).clamp(0, 2)
        else:
            distances = torch.cdist(
                queries,
                chunk,
                p=2,
                compute_mode="donot_use_mm_for_euclid_dist",
            )
        local = distances.topk(min(5, chunk.shape[0]), largest=False, dim=1).values
        best = torch.cat((best, local), dim=1).topk(
            5, largest=False, dim=1
        ).values
    return best


def query_distances(queries: Tensor, bank: Bank, name: str) -> Tensor:
    if name not in DISTANCES:
        raise ValueError(f"unknown distance: {name}")
    if not isinstance(queries, Tensor) or not isinstance(bank.vectors, Tensor):
        raise ValueError("queries and bank vectors must be tensors")
    if queries.ndim != 2 or bank.vectors.ndim != 2:
        raise ValueError("queries and bank vectors must be two-dimensional")
    if queries.shape[1] != bank.vectors.shape[1]:
        raise ValueError("queries and bank must have matching feature dimensions")
    if bank.vectors.shape[0] < 5:
        raise ValueError("distance scoring requires at least five bank rows")
    if not bool(torch.isfinite(queries).all()) or not bool(
        torch.isfinite(bank.vectors).all()
    ):
        raise ValueError("queries and bank vectors must be finite")

    transformed = queries.detach().float()
    bank_vectors = bank.vectors.detach().float().to(transformed.device)
    if not bool(torch.isfinite(transformed).all()) or not bool(
        torch.isfinite(bank_vectors).all()
    ):
        raise ValueError("float32 queries and bank vectors must be finite")
    if name == "mean_5_standardized_euclidean":
        if bank.mean is None or bank.scale is None:
            raise ValueError("standardized Euclidean requires bank mean and scale")
        mean = bank.mean.detach().float().to(transformed.device)
        scale = bank.scale.detach().float().to(transformed.device)
        if (
            mean.ndim != 1
            or scale.ndim != 1
            or mean.shape[0] != transformed.shape[1]
            or scale.shape[0] != transformed.shape[1]
        ):
            raise ValueError("bank mean and scale must match the feature dimension")
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(scale).all()):
            raise ValueError("bank mean and scale must be finite")
        if bool((scale <= 0).any()):
            raise ValueError("standardized Euclidean requires positive scale")
        transformed = (transformed - mean) / scale
        if not bool(torch.isfinite(transformed).all()):
            raise ValueError("standardized queries must be finite")

    cosine = name == "mean_5_cosine"
    if cosine:
        query_norms = transformed.norm(dim=1, keepdim=True)
        bank_norms = bank_vectors.norm(dim=1, keepdim=True)
        if not bool(torch.isfinite(query_norms).all()) or not bool(
            torch.isfinite(bank_norms).all()
        ):
            raise ValueError("cosine norms must be finite")
        if bool((query_norms == 0).any()) or bool((bank_norms == 0).any()):
            raise ValueError("cosine distance does not allow zero-norm rows")
        transformed = transformed / query_norms
        bank_vectors = bank_vectors / bank_norms
        if not bool(torch.isfinite(transformed).all()) or not bool(
            torch.isfinite(bank_vectors).all()
        ):
            raise ValueError("normalized cosine rows must be finite")

    nearest = _nearest_five(transformed, bank_vectors, cosine=cosine)
    if name == "fifth_neighbor_euclidean":
        result = nearest[:, -1]
    else:
        result = nearest.mean(dim=1)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("query distance results must be finite")
    return result


def _highest_confidence_position(confidence: Tensor, query_ids: Tensor) -> int:
    positions_by_query_id = torch.argsort(query_ids).to(confidence.device)
    selected_in_order = confidence.index_select(
        0, positions_by_query_id
    ).argmax()
    return int(positions_by_query_id[selected_in_order].item())


def aggregate_queries(
    distances: Tensor, confidence: Tensor, query_ids: Tensor, name: str
) -> float:
    if name not in AGGREGATIONS:
        raise ValueError(f"unknown aggregation: {name}")
    if not all(isinstance(values, Tensor) for values in (distances, confidence, query_ids)):
        raise ValueError("distances, confidence, and query IDs must be tensors")
    if distances.ndim != 1 or confidence.ndim != 1 or query_ids.ndim != 1:
        raise ValueError("aggregation inputs must be one-dimensional")
    if distances.numel() == 0:
        raise ValueError("aggregation requires at least one query")
    if (
        distances.numel() != confidence.numel()
        or distances.numel() != query_ids.numel()
    ):
        raise ValueError("distances, confidence, and query IDs must align")
    if not bool(torch.isfinite(distances).all()) or not bool(
        torch.isfinite(confidence).all()
    ):
        raise ValueError("distances and confidence must be finite")
    if (
        query_ids.dtype == torch.bool
        or query_ids.is_floating_point()
        or query_ids.is_complex()
    ):
        raise ValueError("query IDs must have an integer dtype")
    if query_ids.unique().numel() != query_ids.numel():
        raise ValueError("query IDs must be unique")

    if name == "mean_all":
        result = float(distances.mean())
    elif name == "q90_all":
        result = float(
            distances.sort().values[int(np.ceil(0.9 * len(distances))) - 1]
        )
    elif name == "top20_mean_all":
        result = float(
            distances.topk(int(np.ceil(0.2 * len(distances)))).values.mean()
        )
    elif name == "top_confidence_query":
        index = _highest_confidence_position(confidence, query_ids)
        result = float(distances[index])
    else:
        confidence_total = confidence.sum()
        if float(confidence_total) == 0:
            raise ValueError(
                "confidence-weighted mean requires non-zero total confidence"
            )
        result = float((distances * confidence).sum() / confidence_total)
    if not np.isfinite(result):
        raise ValueError("aggregated query score must be finite")
    return result


def top_query_entropy(logits: Tensor, query_ids: Tensor) -> float:
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 2:
        raise ValueError("logits must contain retained queries and at least two classes")
    if query_ids.ndim != 1 or query_ids.numel() != logits.shape[0]:
        raise ValueError("query IDs must align with retained logits")
    if (
        query_ids.dtype == torch.bool
        or query_ids.is_floating_point()
        or query_ids.is_complex()
    ):
        raise ValueError("query IDs must have an integer dtype")
    if query_ids.unique().numel() != query_ids.numel():
        raise ValueError("query IDs must be unique")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite")

    float_logits = logits.detach().float()
    if not bool(torch.isfinite(float_logits).all()):
        raise ValueError("float32 logits must be finite")
    confidence = float_logits.sigmoid().amax(dim=1)
    index = _highest_confidence_position(confidence, query_ids)
    probability = float_logits[index].softmax(dim=0)
    normalizer = torch.log(
        torch.tensor(
            probability.numel(), dtype=probability.dtype, device=probability.device
        )
    )
    entropy = float(-torch.xlogy(probability, probability).sum() / normalizer)
    if not np.isfinite(entropy):
        raise ValueError("normalized entropy must be finite")
    return entropy


def score_image_group(records, bank: Bank, distance: str) -> list[ImageScore]:
    records = list(records)
    severities = [record.get("severity") for record in records]
    if (
        any(
            isinstance(severity, bool)
            or not isinstance(severity, (int, np.integer))
            for severity in severities
        )
        or sorted(int(severity) for severity in severities) != [0, 4, 5]
    ):
        raise ValueError("image group must contain exactly levels 0, 4, and 5")
    image_ids = {str(record["image_id"]) for record in records}
    if len(image_ids) != 1:
        raise ValueError("image group must contain exactly one image")
    families = {str(record["family"]) for record in records}
    if len(families) != 1:
        raise ValueError("image group must contain exactly one family")

    records.sort(key=lambda record: int(record["severity"]))
    padded_ids = union_padded_query_ids(records)
    query_count = int(records[0]["persistence"].shape[0])
    keep = torch.ones(query_count, dtype=torch.bool)
    keep[padded_ids] = False
    valid_ids = torch.arange(query_count, dtype=torch.long)[keep]
    if valid_ids.numel() == 0:
        raise ValueError("padding leaves no valid queries")
    valid_query_ids = tuple(int(value) for value in valid_ids)

    rows = []
    for record in records:
        logits = record["logits"].detach().float().index_select(
            0, valid_ids.to(record["logits"].device)
        )
        persistence = record["persistence"].detach().float().index_select(
            0, valid_ids.to(record["persistence"].device)
        )
        confidence = logits.sigmoid().amax(dim=1)
        raw_confidence = float(confidence.max())
        distances = query_distances(persistence, bank, distance)
        rows.append(
            ImageScore(
                image_id=next(iter(image_ids)),
                family=next(iter(families)),
                severity=int(record["severity"]),
                valid_query_ids=valid_query_ids,
                fingerprint_scores={
                    aggregation: aggregate_queries(
                        distances, confidence, valid_ids, aggregation
                    )
                    for aggregation in AGGREGATIONS
                },
                raw_confidence=raw_confidence,
                confidence_score=1.0 - raw_confidence,
                entropy_score=top_query_entropy(logits, valid_ids),
            )
        )
    return rows


def _finite_row_score(row: Mapping, field: str, *, context: str) -> float:
    if field not in row:
        raise ValueError(f"{context} is missing required score {field!r}")
    value = row[field]
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{context} score {field!r} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{context} score {field!r} must be finite")
    return result


def _complete_score_groups(rows, fields: tuple[str, ...]):
    groups: dict[str, dict[str, dict[int, Mapping]]] = {}
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"score row {row_index} must be a mapping")
        context = f"score row {row_index}"
        for field in ("image_id", "family", "severity"):
            if field not in row:
                raise ValueError(f"{context} is missing required field {field!r}")
        image_id = row["image_id"]
        family = row["family"]
        severity = row["severity"]
        if not isinstance(image_id, str) or not image_id.strip():
            raise ValueError(f"{context} image_id must be a nonempty string")
        if not isinstance(family, str) or not family.strip():
            raise ValueError(f"{context} family must be a nonempty string")
        if isinstance(severity, bool) or not isinstance(severity, Integral):
            raise ValueError(f"{context} severity must be an integer")
        severity = int(severity)
        if severity not in (0, 4, 5):
            raise ValueError(f"{context} severity must be one of 0, 4, and 5")
        for field in fields:
            _finite_row_score(row, field, context=context)

        severity_rows = groups.setdefault(family, {}).setdefault(image_id, {})
        if severity in severity_rows:
            raise ValueError(
                f"duplicate score for family {family!r}, image {image_id!r}, "
                f"severity {severity}"
            )
        severity_rows[severity] = row

    if not groups:
        raise ValueError("score evaluation needs at least one complete image group")
    for family, image_rows in groups.items():
        for image_id, severity_rows in image_rows.items():
            if set(severity_rows) != {0, 4, 5}:
                raise ValueError(
                    f"family {family!r}, image {image_id!r} must have exactly one "
                    "score at levels 0, 4, and 5"
                )
    return groups


def _require_shared_image_roster(groups, *, context: str) -> None:
    rosters = {frozenset(image_rows) for image_rows in groups.values()}
    if len(rosters) != 1:
        raise ValueError(f"{context} families must use the same image-ID roster")


def per_family_aurocs(rows, *, method: str) -> dict[str, FamilyResult]:
    if not isinstance(method, str) or not method:
        raise ValueError("method must be a nonempty score-field name")
    groups = _complete_score_groups(rows, (method,))
    results = {}
    for family in sorted(groups):
        image_rows = groups[family]
        ordered_ids = sorted(image_rows)
        clean = [
            _finite_row_score(
                image_rows[image_id][0], method, context=f"{family}/{image_id}/0"
            )
            for image_id in ordered_ids
        ]
        level4 = binary_auroc(
            clean,
            [
                _finite_row_score(
                    image_rows[image_id][4],
                    method,
                    context=f"{family}/{image_id}/4",
                )
                for image_id in ordered_ids
            ],
            orientation=1,
        )
        level5 = binary_auroc(
            clean,
            [
                _finite_row_score(
                    image_rows[image_id][5],
                    method,
                    context=f"{family}/{image_id}/5",
                )
                for image_id in ordered_ids
            ],
            orientation=1,
        )
        results[family] = FamilyResult(
            family=family,
            level4=level4,
            level5=level5,
            strong=(level4 + level5) / 2,
        )
    return results


def select_policy(candidate_rows) -> Policy:
    selection_by_policy: dict[Policy, list[Mapping]] = {}
    for row_index, row in enumerate(candidate_rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"candidate row {row_index} must be a mapping")
        split = row.get("split")
        if split not in ("selection", "validation"):
            raise ValueError(
                f"candidate row {row_index} split must be selection or validation"
            )
        if split == "validation":
            continue
        policy = row.get("policy")
        if not isinstance(policy, Policy):
            raise ValueError(
                f"selection candidate row {row_index} must contain a Policy"
            )
        selection_by_policy.setdefault(policy, []).append(row)

    if not selection_by_policy:
        raise ValueError("policy selection needs at least one selection candidate")

    panel = None
    ranking = []
    for policy, rows in selection_by_policy.items():
        current_panel = {
            (str(row["family"]), str(row["image_id"]), int(row["severity"]))
            for row in rows
        }
        if panel is None:
            panel = current_panel
        elif current_panel != panel:
            raise ValueError("selection candidates must cover the same score panel")
        _require_shared_image_roster(
            _complete_score_groups(rows, ("fingerprint",)),
            context="policy selection",
        )
        family_results = per_family_aurocs(rows, method="fingerprint")
        strong = np.asarray(
            [result.strong for result in family_results.values()], dtype=float
        )
        ranking.append(
            (
                -float(strong.mean()),
                -float(np.median(strong)),
                policy.policy_id,
                policy,
            )
        )
    return min(ranking)[-1]


def confidence_decile_boundaries(rows, *, severity: int) -> tuple[float, ...]:
    if severity not in (4, 5):
        raise ValueError("confidence deciles require severity 4 or 5")
    rows = list(rows)
    if any(
        not isinstance(row, Mapping) or row.get("split") != "selection"
        for row in rows
    ):
        raise ValueError("confidence deciles require selection-only rows")
    groups = _complete_score_groups(rows, ("raw_confidence",))
    _require_shared_image_roster(groups, context="confidence deciles")
    values = []
    for family in sorted(groups):
        for image_id in sorted(groups[family]):
            for level in (0, severity):
                values.append(
                    _finite_row_score(
                        groups[family][image_id][level],
                        "raw_confidence",
                        context=f"{family}/{image_id}/{level}",
                    )
                )
    quantiles = np.quantile(values, np.arange(0.1, 1.0, 0.1))
    return tuple(float(value) for value in np.unique(quantiles))


def _validated_boundaries(boundaries) -> tuple[float, ...]:
    values = np.asarray(tuple(boundaries), dtype=float)
    if values.ndim != 1 or not bool(np.isfinite(values).all()):
        raise ValueError("confidence boundaries must be a finite sequence")
    if values.size > 1 and bool(np.any(np.diff(values) < 0)):
        raise ValueError("confidence boundaries must be sorted")
    return tuple(float(value) for value in np.unique(values))


def _stratum(value: float, boundaries: tuple[float, ...]) -> int:
    return int(np.searchsorted(boundaries, value, side="right"))


def _ordering_credit(clean: float, corrupted: float) -> float:
    if corrupted > clean:
        return 1.0
    if corrupted == clean:
        return 0.5
    return 0.0


def confidence_conditioned_concordance(
    rows, *, family: str, severity: int, boundaries
) -> ConditionalResult:
    if not isinstance(family, str) or not family:
        raise ValueError("conditional concordance requires a nonempty family")
    if severity not in (4, 5):
        raise ValueError("conditional concordance requires severity 4 or 5")
    boundaries = _validated_boundaries(boundaries)
    family_rows = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("family") == family
    ]
    groups = _complete_score_groups(
        family_rows, ("fingerprint", "raw_confidence")
    )
    image_rows = groups[family]
    credits = []
    for image_id in sorted(image_rows):
        clean = image_rows[image_id][0]
        corrupted = image_rows[image_id][severity]
        clean_confidence = _finite_row_score(
            clean, "raw_confidence", context=f"{family}/{image_id}/0"
        )
        corrupted_confidence = _finite_row_score(
            corrupted,
            "raw_confidence",
            context=f"{family}/{image_id}/{severity}",
        )
        if _stratum(clean_confidence, boundaries) != _stratum(
            corrupted_confidence, boundaries
        ):
            continue
        credits.append(
            _ordering_credit(
                _finite_row_score(
                    clean, "fingerprint", context=f"{family}/{image_id}/0"
                ),
                _finite_row_score(
                    corrupted,
                    "fingerprint",
                    context=f"{family}/{image_id}/{severity}",
                ),
            )
        )
    return ConditionalResult(
        family=family,
        severity=severity,
        point=None if not credits else float(np.mean(credits)),
        pair_count=len(credits),
    )


def _selected_families_and_severities(groups, family, severity):
    if family is not None:
        if family not in groups:
            raise ValueError(f"unknown validation family {family!r}")
        families = (family,)
    else:
        families = tuple(sorted(groups))
    if severity is not None and severity not in (4, 5):
        raise ValueError("validation bootstrap severity must be 4 or 5")
    severities = (4, 5) if severity is None else (severity,)
    return families, severities


def _validation_metric(
    groups,
    sampled_ids,
    method: str,
    *,
    families: tuple[str, ...],
    severities: tuple[int, ...],
) -> float:
    family_values = []
    for family in families:
        image_rows = groups[family]
        clean = [
            _finite_row_score(
                image_rows[image_id][0], method, context=f"{family}/{image_id}/0"
            )
            for image_id in sampled_ids
        ]
        level_values = []
        for severity in severities:
            corrupted = [
                _finite_row_score(
                    image_rows[image_id][severity],
                    method,
                    context=f"{family}/{image_id}/{severity}",
                )
                for image_id in sampled_ids
            ]
            level_values.append(binary_auroc(clean, corrupted, orientation=1))
        family_values.append(float(np.mean(level_values)))
    return float(np.mean(family_values))


def _conditional_metric(
    groups,
    sampled_ids,
    boundaries_by_severity,
    *,
    families: tuple[str, ...],
    severities: tuple[int, ...],
) -> tuple[float | None, int]:
    credits = []
    for image_id in sampled_ids:
        for family in families:
            image_rows = groups[family][image_id]
            clean = image_rows[0]
            clean_confidence = _finite_row_score(
                clean, "raw_confidence", context=f"{family}/{image_id}/0"
            )
            clean_fingerprint = _finite_row_score(
                clean, "fingerprint", context=f"{family}/{image_id}/0"
            )
            for severity in severities:
                corrupted = image_rows[severity]
                corrupted_confidence = _finite_row_score(
                    corrupted,
                    "raw_confidence",
                    context=f"{family}/{image_id}/{severity}",
                )
                boundaries = boundaries_by_severity[severity]
                if _stratum(clean_confidence, boundaries) != _stratum(
                    corrupted_confidence, boundaries
                ):
                    continue
                credits.append(
                    _ordering_credit(
                        clean_fingerprint,
                        _finite_row_score(
                            corrupted,
                            "fingerprint",
                            context=f"{family}/{image_id}/{severity}",
                        ),
                    )
                )
    return (None, 0) if not credits else (float(np.mean(credits)), len(credits))


def _score_interval(point, draws, *, count: int) -> ScoreInterval:
    values = np.asarray(draws, dtype=float)
    finite = values[np.isfinite(values)]
    if point is None or finite.size == 0:
        return ScoreInterval(point=point, lower=None, upper=None, count=count)
    lower, upper = np.percentile(finite, (2.5, 97.5))
    return ScoreInterval(
        point=float(point),
        lower=float(lower),
        upper=float(upper),
        count=count,
    )


def paired_validation_bootstrap(
    rows,
    *,
    boundaries_by_severity: Mapping[int, Sequence[float]],
    samples: int,
    seed: int,
    family: str | None = None,
    severity: int | None = None,
) -> ValidationBootstrap:
    if isinstance(samples, bool) or not isinstance(samples, Integral) or samples <= 0:
        raise ValueError("bootstrap samples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("bootstrap seed must be a non-negative integer")
    rows = list(rows)
    if any(
        not isinstance(row, Mapping) or row.get("split") != "validation"
        for row in rows
    ):
        raise ValueError("paired bootstrap requires validation-only rows")
    groups = _complete_score_groups(
        rows, ("fingerprint", "confidence", "entropy", "raw_confidence")
    )
    if set(boundaries_by_severity) != {4, 5}:
        raise ValueError("confidence boundaries are required for severities 4 and 5")
    boundaries = {
        level: _validated_boundaries(boundaries_by_severity[level])
        for level in (4, 5)
    }
    families, severities = _selected_families_and_severities(
        groups, family, severity
    )

    image_id_sets = {frozenset(image_rows) for image_rows in groups.values()}
    if len(image_id_sets) != 1:
        raise ValueError("every validation family must cover the same image IDs")
    ordered_ids = tuple(sorted(next(iter(image_id_sets))))
    count = len(ordered_ids)
    identity_ids = ordered_ids
    method_points = {
        method: _validation_metric(
            groups,
            identity_ids,
            method,
            families=families,
            severities=severities,
        )
        for method in ("fingerprint", "confidence", "entropy")
    }
    conditional_point, conditional_count = _conditional_metric(
        groups,
        identity_ids,
        boundaries,
        families=families,
        severities=severities,
    )

    generator = np.random.default_rng(int(seed))
    draw_indices = generator.integers(0, count, size=(int(samples), count))
    method_draws = {
        method: np.empty(int(samples), dtype=float)
        for method in ("fingerprint", "confidence", "entropy")
    }
    conditional_draws = np.full(int(samples), np.nan, dtype=float)
    for draw_index, indices in enumerate(draw_indices):
        sampled_ids = tuple(ordered_ids[int(index)] for index in indices)
        for method in method_draws:
            method_draws[method][draw_index] = _validation_metric(
                groups,
                sampled_ids,
                method,
                families=families,
                severities=severities,
            )
        conditional_value, _ = _conditional_metric(
            groups,
            sampled_ids,
            boundaries,
            families=families,
            severities=severities,
        )
        if conditional_value is not None:
            conditional_draws[draw_index] = conditional_value

    fingerprint_minus_confidence = (
        method_draws["fingerprint"] - method_draws["confidence"]
    )
    fingerprint_minus_entropy = (
        method_draws["fingerprint"] - method_draws["entropy"]
    )
    return ValidationBootstrap(
        fingerprint=_score_interval(
            method_points["fingerprint"], method_draws["fingerprint"], count=count
        ),
        confidence=_score_interval(
            method_points["confidence"], method_draws["confidence"], count=count
        ),
        entropy=_score_interval(
            method_points["entropy"], method_draws["entropy"], count=count
        ),
        fingerprint_minus_confidence=_score_interval(
            method_points["fingerprint"] - method_points["confidence"],
            fingerprint_minus_confidence,
            count=count,
        ),
        fingerprint_minus_entropy=_score_interval(
            method_points["fingerprint"] - method_points["entropy"],
            fingerprint_minus_entropy,
            count=count,
        ),
        conditional=_score_interval(
            conditional_point,
            conditional_draws,
            count=conditional_count,
        ),
    )


def summarize_seed_scores(scores: Mapping[int, Real]) -> SeedSummary:
    if not isinstance(scores, Mapping) or not scores:
        raise ValueError("seed summary needs at least one supplied score")
    values = []
    for seed, score in scores.items():
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ValueError("seed summary keys must be integers")
        if isinstance(score, (bool, np.bool_)) or not isinstance(score, Real):
            raise ValueError("seed summary scores must be real numbers")
        value = float(score)
        if not np.isfinite(value):
            raise ValueError("seed summary scores must be finite")
        values.append(value)
    array = np.asarray(values, dtype=float)
    return SeedSummary(
        mean=float(array.mean()),
        standard_deviation=float(array.std(ddof=0)),
        minimum=float(array.min()),
        maximum=float(array.max()),
    )


_GEOMETRY_DISTANCES = {
    "euclidean": ("mean_5_euclidean", "fifth_neighbor_euclidean"),
    "standardized_euclidean": ("mean_5_standardized_euclidean",),
    "cosine": ("mean_5_cosine",),
}
_DISTANCE_GEOMETRIES = {
    distance: geometry
    for geometry, distances in _GEOMETRY_DISTANCES.items()
    for distance in distances
}


def _study_config(config: StudyConfig) -> StudyConfig:
    if not isinstance(config, StudyConfig):
        raise ValueError("config must be a StudyConfig")
    positive_fields = (
        "reference_count",
        "evaluation_count",
        "selection_count",
        "validation_count",
        "bank_capacity",
        "bootstrap_draws",
    )
    for field in positive_fields:
        value = getattr(config, field)
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if config.selection_count + config.validation_count != config.evaluation_count:
        raise ValueError(
            "selection_count plus validation_count must equal evaluation_count"
        )
    if config.levels != (0, 4, 5):
        raise ValueError("strong corruption study levels must be exactly 0, 4, and 5")
    if not config.families or len(set(config.families)) != len(config.families):
        raise ValueError("study families must be nonempty and unique")
    return config


def _scientific_metadata(metadata: Mapping, *, label: str) -> tuple:
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{label} metadata must be an object")
    config = metadata.get("config")
    if not isinstance(config, Mapping):
        raise ValueError(f"{label} metadata must contain config")
    fields = (
        "query_count",
        "class_count",
        "persistence_layer",
        "persistence_dim",
    )
    values = []
    checkpoint = metadata.get("checkpoint_sha256")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError(f"{label} metadata must contain checkpoint_sha256")
    values.append(checkpoint)
    for field in fields:
        value = config.get(field)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{label} config must contain integer {field}")
        values.append(int(value))
    return tuple(values)


def _json_object(path: Path, *, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{label} must be valid JSON") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def _load_annotations(
    path: Path, reference_ids: set[str], class_count: int
) -> tuple[dict[str, list[Mapping]], dict[str, tuple[int, int]], tuple[int, ...]]:
    payload = _json_object(path, label="COCO annotations")
    images = payload.get("images")
    annotations = payload.get("annotations")
    categories = payload.get("categories")
    if not all(isinstance(values, list) for values in (images, annotations, categories)):
        raise ValueError(
            "COCO annotations must contain images, annotations, and categories lists"
        )

    image_sizes = {}
    for image in images:
        if not isinstance(image, Mapping) or "id" not in image:
            raise ValueError("COCO image entries must contain IDs")
        image_id = str(image["id"])
        if image_id not in reference_ids:
            continue
        width = image.get("width")
        height = image.get("height")
        if (
            isinstance(width, bool)
            or not isinstance(width, Integral)
            or width <= 0
            or isinstance(height, bool)
            or not isinstance(height, Integral)
            or height <= 0
        ):
            raise ValueError(f"COCO image {image_id!r} has invalid dimensions")
        image_sizes[image_id] = (int(width), int(height))
    if set(image_sizes) != reference_ids:
        missing = sorted(reference_ids - set(image_sizes))
        raise ValueError(f"COCO annotations are missing reference images: {missing}")

    annotations_by_image: dict[str, list[Mapping]] = {
        image_id: [] for image_id in reference_ids
    }
    for annotation in annotations:
        if not isinstance(annotation, Mapping) or "image_id" not in annotation:
            raise ValueError("COCO annotation entries must contain image_id")
        image_id = str(annotation["image_id"])
        if image_id in annotations_by_image:
            annotations_by_image[image_id].append(annotation)

    try:
        category_ids = tuple(
            sorted({int(category["id"]) for category in categories})
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("COCO categories must contain integer IDs") from error
    if len(category_ids) != class_count:
        raise ValueError(
            f"COCO categories contain {len(category_ids)} classes; "
            f"artifacts require {class_count}"
        )
    return annotations_by_image, image_sizes, category_ids


def _load_evaluation_groups(
    evaluation_root: Path,
    *,
    evaluation_ids: set[str],
    families: tuple[str, ...],
    expected_science: tuple,
) -> tuple[list[tuple[dict, ...]], frozenset[int]]:
    roster = _json_object(
        evaluation_root / "corruption-roster.json",
        label="benchmark corruption roster",
    )
    if _scientific_metadata(roster, label="benchmark roster") != expected_science:
        raise ValueError("benchmark roster scientific metadata differs from reference")
    corruptions = roster.get("corruptions")
    if not isinstance(corruptions, list):
        raise ValueError("benchmark corruption roster must contain corruptions")
    roster_names = [
        item.get("name") for item in corruptions if isinstance(item, Mapping)
    ]
    for family in families:
        if roster_names.count(family) != 1:
            raise ValueError(
                f"configured corruption family {family!r} must occur exactly once"
            )

    groups = []
    levels_read: set[int] = set()
    expected_keys = {
        (family, image_id, severity)
        for family in families
        for image_id in evaluation_ids
        for severity in (0, 4, 5)
    }
    seen_keys = set()
    for family in families:
        cache = (
            evaluation_root
            / "corruptions"
            / family
            / "artifacts"
            / "evaluation-extractions"
        )
        metadata = load_artifact_manifest(cache)
        if _scientific_metadata(metadata, label=f"{family} artifact") != expected_science:
            raise ValueError(
                f"corruption family {family!r} scientific metadata differs "
                "from reference"
            )
        by_image: dict[str, list[dict]] = {}
        for record in iter_records(cache):
            severity = record.get("severity")
            if (
                isinstance(severity, bool)
                or not isinstance(severity, Integral)
                or int(severity) not in (0, 4, 5)
            ):
                continue
            severity = int(severity)
            image_id = str(record.get("image_id"))
            if image_id not in evaluation_ids:
                raise ValueError(
                    f"corruption family {family!r} contains unexpected image "
                    f"{image_id!r}"
                )
            key = (family, image_id, severity)
            if key in seen_keys:
                raise ValueError(f"duplicate evaluation record: {key!r}")
            seen_keys.add(key)
            levels_read.add(severity)
            by_image.setdefault(image_id, []).append(
                {**record, "image_id": image_id, "family": family}
            )
        for image_id in sorted(evaluation_ids):
            image_records = tuple(
                sorted(by_image.get(image_id, ()), key=lambda item: item["severity"])
            )
            if [record["severity"] for record in image_records] != [0, 4, 5]:
                raise ValueError(
                    f"family {family!r}, image {image_id!r} must contain "
                    "exactly levels 0, 4, and 5"
                )
            groups.append(image_records)
    if seen_keys != expected_keys:
        raise ValueError("evaluation artifacts do not cover the configured panel")
    return groups, frozenset(levels_read)


def _prepare_image_group(records) -> list[dict]:
    records = sorted(records, key=lambda record: int(record["severity"]))
    padded_ids = union_padded_query_ids(records)
    query_count = int(records[0]["persistence"].shape[0])
    keep = torch.ones(query_count, dtype=torch.bool)
    keep[padded_ids] = False
    valid_ids = torch.arange(query_count, dtype=torch.long)[keep]
    if valid_ids.numel() == 0:
        raise ValueError("padding leaves no valid queries")

    prepared = []
    for record in records:
        logits = record["logits"].detach().float().index_select(
            0, valid_ids.to(record["logits"].device)
        )
        confidence = logits.sigmoid().amax(dim=1).cpu()
        raw_confidence = float(confidence.max())
        prepared.append(
            {
                "image_id": str(record["image_id"]),
                "family": str(record["family"]),
                "severity": int(record["severity"]),
                "query_ids": valid_ids,
                "confidence_by_query": confidence,
                "persistence": record["persistence"].detach().index_select(
                    0, valid_ids.to(record["persistence"].device)
                ),
                "raw_confidence": raw_confidence,
                "confidence": 1.0 - raw_confidence,
                "entropy": top_query_entropy(logits, valid_ids),
            }
        )
    return prepared


def _bank_on_device(bank: Bank, geometry: str, device: torch.device) -> Bank:
    vectors = bank.vectors.to(device=device, dtype=torch.float32)
    if geometry == "cosine":
        norms = vectors.norm(dim=1, keepdim=True)
        if bool((norms == 0).any()) or not bool(torch.isfinite(norms).all()):
            raise ValueError("cosine distance does not allow zero-norm bank rows")
        vectors = vectors / norms
    return Bank(
        vectors=vectors,
        mean=None if bank.mean is None else bank.mean.to(device),
        scale=None if bank.scale is None else bank.scale.to(device),
        matched_count=bank.matched_count,
        background_count=bank.background_count,
    )


def _compute_neighbor_tensor(queries: Tensor, bank: Bank, geometry: str) -> Tensor:
    transformed = queries.detach().to(
        device=bank.vectors.device, dtype=torch.float32
    )
    cosine = geometry == "cosine"
    if geometry == "standardized_euclidean":
        if bank.mean is None or bank.scale is None:
            raise ValueError("standardized Euclidean requires bank moments")
        transformed = (transformed - bank.mean) / bank.scale
    elif cosine:
        norms = transformed.norm(dim=1, keepdim=True)
        if bool((norms == 0).any()) or not bool(torch.isfinite(norms).all()):
            raise ValueError("cosine distance does not allow zero-norm query rows")
        transformed = transformed / norms
    if not bool(torch.isfinite(transformed).all()):
        raise ValueError(f"{geometry} queries must be finite")
    return _nearest_five(transformed, bank.vectors, cosine=cosine)


def _cache_metadata(
    *,
    geometry: str,
    bank: str,
    seed: int,
    reference_digest: str,
    evaluation_digest: str,
    evaluation_ids: tuple[str, ...],
    families: tuple[str, ...],
) -> dict:
    return {
        "geometry": geometry,
        "bank": bank,
        "seed": seed,
        "reference_manifest_sha256": reference_digest,
        "evaluation_manifest_sha256": evaluation_digest,
        "evaluation_image_ids": list(evaluation_ids),
        "families": list(families),
    }


def _load_distance_cache(path: Path, expected_metadata: dict):
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, Mapping)
        or payload.get("metadata") != expected_metadata
    ):
        return None
    return payload["records"]


def _open_distance_cache(
    candidates: CandidateRows,
    *,
    bank_name: str,
    geometry: str,
    distance: str,
    seed: int,
    config: StudyConfig,
    cache_directory: Path,
    reference_digest: str,
    evaluation_digest: str,
    evaluation_ids: tuple[str, ...],
    device: torch.device,
):
    metadata = _cache_metadata(
        geometry=geometry,
        bank=bank_name,
        seed=seed,
        reference_digest=reference_digest,
        evaluation_digest=evaluation_digest,
        evaluation_ids=evaluation_ids,
        families=config.families,
    )
    path = cache_directory / f"{bank_name}-{geometry}-seed{seed}.pt"
    cached = _load_distance_cache(path, metadata)
    if cached is not None:
        return {
            "metadata": metadata,
            "path": path,
            "loaded": {
                _record_key(record): record["neighbors"] for record in cached
            },
            "records": None,
            "bank": None,
        }, None
    try:
        bank = build_bank(
            candidates,
            variant=bank_name,
            distance=distance,
            capacity=config.bank_capacity,
            seed=seed,
        )
        bank = _bank_on_device(bank, geometry, device)
    except ValueError as error:
        return None, str(error)
    return {
        "metadata": metadata,
        "path": path,
        "loaded": None,
        "records": [],
        "bank": bank,
    }, None


def _cache_neighbors(state: dict, record: Mapping, geometry: str) -> Tensor:
    record_key = _record_key(record)
    if state["loaded"] is not None:
        return state["loaded"][record_key]
    neighbors = _compute_neighbor_tensor(
        record["persistence"], state["bank"], geometry
    ).detach().cpu()
    state["records"].append(
        {
            "family": str(record["family"]),
            "image_id": str(record["image_id"]),
            "severity": int(record["severity"]),
            "neighbors": neighbors,
        }
    )
    return neighbors


def _save_distance_cache(state: dict) -> None:
    if state["records"] is not None:
        torch.save(
            {"metadata": state["metadata"], "records": state["records"]},
            state["path"],
        )


def _record_key(record: Mapping) -> tuple[str, str, int]:
    return (
        str(record["family"]),
        str(record["image_id"]),
        int(record["severity"]),
    )


def _neighbor_scores(
    neighbors: Tensor,
    confidence: Tensor,
    query_ids: Tensor,
    *,
    geometry: str,
) -> dict[tuple[str, str], float]:
    results = {}
    for distance in _GEOMETRY_DISTANCES[geometry]:
        distances = (
            neighbors[:, -1]
            if distance == "fifth_neighbor_euclidean"
            else neighbors.mean(dim=1)
        )
        for aggregation in AGGREGATIONS:
            results[(distance, aggregation)] = aggregate_queries(
                distances, confidence, query_ids, aggregation
            )
    return results


def _score_primary_arms(
    groups,
    candidates: CandidateRows,
    *,
    split: ImageSplit,
    config: StudyConfig,
    cache_directory: Path,
    reference_digest: str,
    evaluation_digest: str,
    evaluation_ids: tuple[str, ...],
    device: torch.device,
) -> tuple[list[dict], list[dict]]:
    selection_ids = set(split.selection)
    states = {}
    infeasible = []
    for bank_name in BANK_VARIANTS:
        for geometry in _GEOMETRY_DISTANCES:
            key = (bank_name, geometry, config.primary_seed)
            state, error = _open_distance_cache(
                candidates,
                bank_name=bank_name,
                geometry=geometry,
                distance=_GEOMETRY_DISTANCES[geometry][0],
                seed=config.primary_seed,
                config=config,
                cache_directory=cache_directory,
                reference_digest=reference_digest,
                evaluation_digest=evaluation_digest,
                evaluation_ids=evaluation_ids,
                device=device,
            )
            if error is not None:
                infeasible.append(
                    {
                        "bank": bank_name,
                        "geometry": geometry,
                        "seed": config.primary_seed,
                        "reason": error,
                    }
                )
                continue
            state["rows"] = []
            states[key] = state

    failed = {}
    for group in groups:
        prepared = _prepare_image_group(group)
        for record in prepared:
            for key, state in states.items():
                if key in failed:
                    continue
                try:
                    neighbors = _cache_neighbors(state, record, key[1])
                    scores = _neighbor_scores(
                        neighbors,
                        record["confidence_by_query"],
                        record["query_ids"],
                        geometry=key[1],
                    )
                except ValueError as error:
                    failed[key] = str(error)
                    state["rows"].clear()
                    state["records"] = None
                    continue
                for (distance, aggregation), fingerprint in scores.items():
                    policy = Policy(
                        bank=key[0],
                        distance=distance,
                        aggregation=aggregation,
                        seed=key[2],
                    )
                    state["rows"].append(
                        {
                            "image_id": record["image_id"],
                            "family": record["family"],
                            "severity": record["severity"],
                            "split": (
                                "selection"
                                if record["image_id"] in selection_ids
                                else "validation"
                            ),
                            "policy": policy,
                            "fingerprint": fingerprint,
                            "confidence": record["confidence"],
                            "entropy": record["entropy"],
                            "raw_confidence": record["raw_confidence"],
                        }
                    )

    rows = []
    for key, state in states.items():
        if key in failed:
            infeasible.append(
                {
                    "bank": key[0],
                    "geometry": key[1],
                    "seed": key[2],
                    "reason": failed[key],
                }
            )
            continue
        _save_distance_cache(state)
        rows.extend(state["rows"])
    if not rows:
        raise ValueError("no feasible fingerprint candidate remains")
    return rows, infeasible


def _evidence_status(interval: ScoreInterval, threshold: float) -> str:
    if (
        interval.point is None
        or interval.lower is None
        or interval.upper is None
        or interval.count <= 0
    ):
        return "inconclusive"
    if interval.lower > threshold:
        return "supported"
    if interval.upper <= threshold:
        return "not_supported"
    return "inconclusive"


def _evidence_conclusions(
    bootstrap: ValidationBootstrap, *, conditional_complete: bool
) -> dict[str, str]:
    return {
        "detects_corruption": _evidence_status(bootstrap.fingerprint, 0.5),
        "better_than_confidence": _evidence_status(
            bootstrap.fingerprint_minus_confidence, 0.0
        ),
        "information_after_confidence": (
            _evidence_status(bootstrap.conditional, 0.5)
            if conditional_complete
            else "inconclusive"
        ),
    }


def _validation_strong(rows) -> float:
    values = [
        result.strong
        for result in per_family_aurocs(
            [row for row in rows if row["split"] == "validation"],
            method="fingerprint",
        ).values()
    ]
    return float(np.mean(values))


def _score_sensitivity_seeds(
    groups,
    candidates: CandidateRows,
    primary_rows,
    selected: Policy,
    *,
    split: ImageSplit,
    config: StudyConfig,
    cache_directory: Path,
    reference_digest: str,
    evaluation_digest: str,
    evaluation_ids: tuple[str, ...],
    device: torch.device,
) -> dict[int, float]:
    geometry = _DISTANCE_GEOMETRIES[selected.distance]
    rows_by_seed = {
        config.primary_seed: [
            row for row in primary_rows if row["policy"] == selected
        ]
    }
    states = {}
    for seed in config.sensitivity_seeds:
        if seed == config.primary_seed:
            continue
        state, error = _open_distance_cache(
            candidates,
            bank_name=selected.bank,
            geometry=geometry,
            distance=selected.distance,
            seed=seed,
            config=config,
            cache_directory=cache_directory,
            reference_digest=reference_digest,
            evaluation_digest=evaluation_digest,
            evaluation_ids=evaluation_ids,
            device=device,
        )
        if error is not None:
            raise ValueError(
                f"selected sensitivity seed {seed} is infeasible: {error}"
            )
        state["rows"] = []
        states[seed] = state

    selection_ids = set(split.selection)
    for group in groups:
        for record in _prepare_image_group(group):
            for seed, state in states.items():
                neighbors = _cache_neighbors(state, record, geometry)
                fingerprint = _neighbor_scores(
                    neighbors,
                    record["confidence_by_query"],
                    record["query_ids"],
                    geometry=geometry,
                )[(selected.distance, selected.aggregation)]
                state["rows"].append(
                    {
                        "image_id": record["image_id"],
                        "family": record["family"],
                        "severity": record["severity"],
                        "split": (
                            "selection"
                            if record["image_id"] in selection_ids
                            else "validation"
                        ),
                        "policy": Policy(
                            selected.bank,
                            selected.distance,
                            selected.aggregation,
                            seed,
                        ),
                        "fingerprint": fingerprint,
                        "confidence": record["confidence"],
                        "entropy": record["entropy"],
                        "raw_confidence": record["raw_confidence"],
                    }
                )

    for seed, state in states.items():
        _save_distance_cache(state)
        rows_by_seed[seed] = state["rows"]
    if set(rows_by_seed) != set(config.sensitivity_seeds):
        raise ValueError("sensitivity scoring did not cover every configured seed")
    scores = {
        seed: _validation_strong(rows)
        for seed, rows in sorted(rows_by_seed.items())
    }
    return scores


def _policy_statistics(candidate_rows) -> list[dict]:
    by_policy: dict[Policy, list[Mapping]] = {}
    for row in candidate_rows:
        by_policy.setdefault(row["policy"], []).append(row)
    ranking = []
    for policy, rows in by_policy.items():
        split_results = {}
        for split_name in ("selection", "validation"):
            family_results = per_family_aurocs(
                [row for row in rows if row["split"] == split_name],
                method="fingerprint",
            )
            values = np.asarray(
                [result.strong for result in family_results.values()], dtype=float
            )
            split_results[split_name] = {
                "family_mean": float(values.mean()),
                "family_median": float(np.median(values)),
            }
        ranking.append(
            {
                "policy_id": policy.policy_id,
                **asdict(policy),
                "selection_family_mean": split_results["selection"]["family_mean"],
                "selection_family_median": split_results["selection"][
                    "family_median"
                ],
                "exploratory_validation_family_mean": split_results["validation"][
                    "family_mean"
                ],
                "exploratory_validation_family_median": split_results[
                    "validation"
                ]["family_median"],
            }
        )
    return sorted(
        ranking,
        key=lambda item: (
            -item["selection_family_mean"],
            -item["selection_family_median"],
            item["policy_id"],
        ),
    )


def _blankable(value):
    return "" if value is None else value


def _result_row(
    row_type: str,
    *,
    split: str = "",
    policy: Policy | None = None,
    corruption: str = "",
    severity="",
    method: str = "",
    interval: ScoreInterval | None = None,
    point=None,
    count=0,
) -> dict:
    if interval is not None:
        point = interval.point
        lower = interval.lower
        upper = interval.upper
        count = interval.count
    else:
        lower = None
        upper = None
    return {
        "row_type": row_type,
        "split": split,
        "policy_id": "" if policy is None else policy.policy_id,
        "bank": "" if policy is None else policy.bank,
        "distance": "" if policy is None else policy.distance,
        "aggregation": "" if policy is None else policy.aggregation,
        "seed": "" if policy is None else policy.seed,
        "corruption": corruption,
        "severity": severity,
        "method": method,
        "point": _blankable(point),
        "lower": _blankable(lower),
        "upper": _blankable(upper),
        "count": count,
    }


def _candidate_audit_rows(candidate_rows, config: StudyConfig) -> list[dict]:
    by_policy: dict[Policy, list[Mapping]] = {}
    for row in candidate_rows:
        by_policy.setdefault(row["policy"], []).append(row)
    output = []
    for bank_name in BANK_VARIANTS:
        for distance in DISTANCES:
            for aggregation in AGGREGATIONS:
                policy = Policy(
                    bank_name, distance, aggregation, config.primary_seed
                )
                rows = by_policy.get(policy)
                for split_name, count in (
                    ("selection", config.selection_count),
                    ("validation", config.validation_count),
                ):
                    results = (
                        {}
                        if rows is None
                        else per_family_aurocs(
                            [
                                row
                                for row in rows
                                if row["split"] == split_name
                            ],
                            method="fingerprint",
                        )
                    )
                    for family in config.families:
                        result = results.get(family)
                        output.append(
                            _result_row(
                                "candidate_family",
                                split=split_name,
                                policy=policy,
                                corruption=family,
                                severity="strong",
                                method="fingerprint",
                                point=None if result is None else result.strong,
                                count=0 if result is None else count,
                            )
                        )
    return output


def _validation_evidence(
    selected_rows,
    *,
    config: StudyConfig,
) -> tuple[
    dict[int, tuple[float, ...]],
    ValidationBootstrap,
    dict[tuple[str, int], ValidationBootstrap],
]:
    selection_rows = [
        row for row in selected_rows if row["split"] == "selection"
    ]
    validation_rows = [
        row for row in selected_rows if row["split"] == "validation"
    ]
    boundaries = {
        severity: confidence_decile_boundaries(
            selection_rows, severity=severity
        )
        for severity in (4, 5)
    }
    aggregate = paired_validation_bootstrap(
        validation_rows,
        boundaries_by_severity=boundaries,
        samples=config.bootstrap_draws,
        seed=config.bootstrap_seed,
    )
    by_task = {
        (family, severity): paired_validation_bootstrap(
            validation_rows,
            boundaries_by_severity=boundaries,
            samples=config.bootstrap_draws,
            seed=config.bootstrap_seed,
            family=family,
            severity=severity,
        )
        for family in config.families
        for severity in (4, 5)
    }
    return boundaries, aggregate, by_task


def _selected_csv_rows(
    selected: Policy,
    task_bootstraps: Mapping[tuple[str, int], ValidationBootstrap],
    seed_scores: Mapping[int, float],
    *,
    config: StudyConfig,
) -> list[dict]:
    output = []
    methods = (
        ("fingerprint", "fingerprint"),
        ("direct_confidence_max", "confidence"),
        (
            "softmax_entropy_top_confidence_query",
            "entropy",
        ),
    )
    for family in config.families:
        for severity in (4, 5):
            bootstrap = task_bootstraps[(family, severity)]
            for method_name, interval_name in methods:
                output.append(
                    _result_row(
                        "selected_method",
                        split="validation",
                        policy=selected,
                        corruption=family,
                        severity=severity,
                        method=method_name,
                        interval=getattr(bootstrap, interval_name),
                    )
                )
            output.append(
                _result_row(
                    "paired_difference",
                    split="validation",
                    policy=selected,
                    corruption=family,
                    severity=severity,
                    method="fingerprint_minus_confidence",
                    interval=bootstrap.fingerprint_minus_confidence,
                )
            )
            output.append(
                _result_row(
                    "paired_difference",
                    split="validation",
                    policy=selected,
                    corruption=family,
                    severity=severity,
                    method="fingerprint_minus_entropy",
                    interval=bootstrap.fingerprint_minus_entropy,
                )
            )
            output.append(
                _result_row(
                    "conditional",
                    split="validation",
                    policy=selected,
                    corruption=family,
                    severity=severity,
                    method="confidence_conditioned_concordance",
                    interval=bootstrap.conditional,
                )
            )
    for seed in config.sensitivity_seeds:
        output.append(
            _result_row(
                "seed",
                split="validation",
                policy=Policy(
                    selected.bank,
                    selected.distance,
                    selected.aggregation,
                    seed,
                ),
                severity="strong",
                method="fingerprint",
                point=seed_scores[seed],
                count=config.validation_count,
            )
        )
    return output


def _format_number(value) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def _controlled_lines(
    ranking: Sequence[Mapping],
    values: Sequence[str],
    *,
    field: str,
    selected: Policy,
) -> list[str]:
    by_value = {}
    for item in ranking:
        if (
            (field == "bank" or item["bank"] == selected.bank)
            and (field == "distance" or item["distance"] == selected.distance)
            and (
                field == "aggregation"
                or item["aggregation"] == selected.aggregation
            )
        ):
            by_value[item[field]] = item
    lines = []
    for value in values:
        item = by_value.get(value)
        if item is None:
            lines.append(
                f"- {value}: selection=NA; exploratory validation=NA"
            )
        else:
            lines.append(
                f"- {value}: "
                f"selection={_format_number(item['selection_family_mean'])}; "
                "exploratory validation="
                f"{_format_number(item['exploratory_validation_family_mean'])}"
            )
    return lines


def _render_report(
    selected_rows,
    selected: Policy,
    ranking,
    aggregate: ValidationBootstrap,
    seed_summary: SeedSummary,
    conclusions: Mapping[str, str],
    *,
    config: StudyConfig,
) -> str:
    validation_rows = [
        row for row in selected_rows if row["split"] == "validation"
    ]
    results = {
        method: per_family_aurocs(validation_rows, method=method)
        for method in ("fingerprint", "confidence", "entropy")
    }
    lines = [
        "# Strong corruption fingerprint study",
        "",
        (
            "Selected tuple: "
            f"bank={selected.bank}, distance={selected.distance}, "
            f"aggregation={selected.aggregation}, seed={selected.seed}"
        ),
        "",
        (
            "Corruption | Fingerprint L4 | Fingerprint L5 | Confidence L4 | "
            "Confidence L5 | Entropy L4 | Entropy L5"
        ),
        "--- | --- | --- | --- | --- | --- | ---",
    ]
    for family in config.families:
        fingerprint = results["fingerprint"][family]
        confidence = results["confidence"][family]
        entropy = results["entropy"][family]
        lines.append(
            " | ".join(
                (
                    family,
                    _format_number(fingerprint.level4),
                    _format_number(fingerprint.level5),
                    _format_number(confidence.level4),
                    _format_number(confidence.level5),
                    _format_number(entropy.level4),
                    _format_number(entropy.level5),
                )
            )
        )
    lines.extend(
        (
            "",
            (
                "Corruption | Fingerprint strong | Confidence strong | "
                "Entropy strong | Fingerprint minus confidence | "
                "Fingerprint minus entropy"
            ),
            "--- | --- | --- | --- | --- | ---",
        )
    )
    for family in config.families:
        fingerprint = results["fingerprint"][family].strong
        confidence = results["confidence"][family].strong
        entropy = results["entropy"][family].strong
        lines.append(
            " | ".join(
                (
                    family,
                    _format_number(fingerprint),
                    _format_number(confidence),
                    _format_number(entropy),
                    _format_number(fingerprint - confidence),
                    _format_number(fingerprint - entropy),
                )
            )
        )
    lines.append("")
    for method, label in (
        ("fingerprint", "Fingerprint"),
        ("confidence", "Confidence"),
        ("entropy", "Entropy"),
    ):
        values = np.asarray(
            [result.strong for result in results[method].values()], dtype=float
        )
        lines.append(
            f"{label} family strong mean={values.mean():.4f}; "
            f"median={np.median(values):.4f}."
        )
    lines.extend(
        (
            "",
            (
                "Conditional diagnostic: "
                f"point={_format_number(aggregate.conditional.point)}, "
                f"interval=[{_format_number(aggregate.conditional.lower)}, "
                f"{_format_number(aggregate.conditional.upper)}], "
                f"eligible pairs={aggregate.conditional.count}."
            ),
            (
                "Seed sensitivity: "
                f"mean={seed_summary.mean:.4f}, "
                f"population std={seed_summary.standard_deviation:.4f}, "
                f"min={seed_summary.minimum:.4f}, "
                f"max={seed_summary.maximum:.4f}."
            ),
            "",
            "Banks at selected distance and aggregation:",
            *_controlled_lines(
                ranking,
                BANK_VARIANTS,
                field="bank",
                selected=selected,
            ),
            "",
            "Distances at selected bank and aggregation:",
            *_controlled_lines(
                ranking,
                DISTANCES,
                field="distance",
                selected=selected,
            ),
            "",
            "Aggregators at selected bank and distance:",
            *_controlled_lines(
                ranking,
                AGGREGATIONS,
                field="aggregation",
                selected=selected,
            ),
            "",
            "Evidence conclusions:",
            *(
                f"- {name}: {status}"
                for name, status in conclusions.items()
            ),
            "",
            "Levels 1 through 3 were not evaluated.",
        )
    )
    return "\n".join(lines) + "\n"


def run_study(
    reference_run,
    evaluation_benchmark,
    annotations,
    output_dir,
    *,
    config: StudyConfig = StudyConfig(),
    device: str = "cpu",
) -> StudyRunResult:
    config = _study_config(config)
    reference_root = Path(reference_run).resolve()
    evaluation_root = Path(evaluation_benchmark).resolve()
    annotations = Path(annotations).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_directory = output / "cache"
    cache_directory.mkdir(exist_ok=True)

    reference_manifest = load_manifest(
        reference_root / "inputs" / "reference-manifest.csv"
    )
    evaluation_manifest = load_manifest(
        evaluation_root / "inputs" / "evaluation-manifest.csv"
    )
    if len(reference_manifest) != config.reference_count:
        raise ValueError(
            f"reference manifest must contain {config.reference_count} images"
        )
    if len(evaluation_manifest) != config.evaluation_count:
        raise ValueError(
            f"evaluation manifest must contain {config.evaluation_count} images"
        )
    validate_disjoint(reference_manifest, evaluation_manifest)
    evaluation_ids = tuple(entry.image_id for entry in evaluation_manifest)
    split = split_image_ids(evaluation_ids, config.selection_count)
    if (
        len(split.selection) != config.selection_count
        or len(split.validation) != config.validation_count
    ):
        raise ValueError("deterministic image split has unexpected sizes")

    reference_cache = (
        reference_root / "reference-artifacts" / "reference-extractions"
    )
    reference_metadata = load_artifact_manifest(reference_cache)
    expected_science = _scientific_metadata(
        reference_metadata, label="reference artifact"
    )
    reference_ids = {entry.image_id for entry in reference_manifest}
    reference_records = list(iter_records(reference_cache))
    if len(reference_records) != config.reference_count:
        raise ValueError(
            f"reference artifacts must contain {config.reference_count} records"
        )
    if {
        str(record.get("image_id")) for record in reference_records
    } != reference_ids:
        raise ValueError("reference artifacts do not match the reference manifest")
    if any(int(record.get("severity", 0)) != 0 for record in reference_records):
        raise ValueError("reference artifacts must contain clean records only")

    annotations_by_image, image_sizes, category_ids = _load_annotations(
        annotations,
        reference_ids,
        expected_science[2],
    )
    candidates = build_reference_candidates(
        reference_records,
        annotations_by_image=annotations_by_image,
        image_sizes=image_sizes,
        category_ids=category_ids,
    )
    del reference_records

    groups, levels_read = _load_evaluation_groups(
        evaluation_root,
        evaluation_ids=set(evaluation_ids),
        families=config.families,
        expected_science=expected_science,
    )
    reference_digest = manifest_digest(reference_manifest)
    evaluation_digest = manifest_digest(evaluation_manifest)
    candidate_rows, infeasible = _score_primary_arms(
        groups,
        candidates,
        split=split,
        config=config,
        cache_directory=cache_directory,
        reference_digest=reference_digest,
        evaluation_digest=evaluation_digest,
        evaluation_ids=evaluation_ids,
        device=torch.device(device),
    )
    selected = select_policy(candidate_rows)
    seed_scores = _score_sensitivity_seeds(
        groups,
        candidates,
        candidate_rows,
        selected,
        split=split,
        config=config,
        cache_directory=cache_directory,
        reference_digest=reference_digest,
        evaluation_digest=evaluation_digest,
        evaluation_ids=evaluation_ids,
        device=torch.device(device),
    )
    del groups, candidates
    selected_rows = [
        row for row in candidate_rows if row["policy"] == selected
    ]
    boundaries, aggregate, task_bootstraps = _validation_evidence(
        selected_rows, config=config
    )
    ranking = _policy_statistics(candidate_rows)
    seed_summary = summarize_seed_scores(seed_scores)
    conditional_complete = all(
        result.conditional.point is not None
        and result.conditional.lower is not None
        and result.conditional.upper is not None
        and result.conditional.count > 0
        for result in task_bootstraps.values()
    )
    conclusions = _evidence_conclusions(
        aggregate, conditional_complete=conditional_complete
    )

    csv_rows = _candidate_audit_rows(candidate_rows, config)
    csv_rows.extend(
        _selected_csv_rows(
            selected,
            task_bootstraps,
            seed_scores,
            config=config,
        )
    )
    fieldnames = (
        "row_type",
        "split",
        "policy_id",
        "bank",
        "distance",
        "aggregation",
        "seed",
        "corruption",
        "severity",
        "method",
        "point",
        "lower",
        "upper",
        "count",
    )
    with (output / "results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    aggregate_metrics = {
        name: asdict(getattr(aggregate, name))
        for name in (
            "fingerprint",
            "confidence",
            "entropy",
            "fingerprint_minus_confidence",
            "fingerprint_minus_entropy",
            "conditional",
        )
    }
    selected_payload = {**asdict(selected), "policy_id": selected.policy_id}
    summary = {
        "configuration": asdict(config),
        "source_paths": {
            "reference_run": str(reference_root),
            "evaluation_benchmark": str(evaluation_root),
            "annotations": str(annotations),
            "output_dir": str(output),
        },
        "manifest_digests": {
            "reference": reference_digest,
            "evaluation": evaluation_digest,
        },
        "selected_policy": selected_payload,
        "selection_ranking": ranking,
        "aggregate_validation": aggregate_metrics,
        "confidence_decile_boundaries": {
            str(severity): list(values)
            for severity, values in boundaries.items()
        },
        "seed_scores": {
            str(seed): score for seed, score in sorted(seed_scores.items())
        },
        "seed_summary": asdict(seed_summary),
        "infeasible_arms": infeasible,
        "conclusions": conclusions,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "report.md").write_text(
        _render_report(
            selected_rows,
            selected,
            ranking,
            aggregate,
            seed_summary,
            conclusions,
            config=config,
        ),
        encoding="utf-8",
    )
    return StudyRunResult(
        selected_policy=selected,
        levels_read=levels_read,
        reported_families=frozenset(config.families),
        output_dir=output,
    )


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run the lean strong-corruption study")
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--evaluation-benchmark", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    run_study(
        args.reference_run,
        args.evaluation_benchmark,
        args.annotations,
        args.output_dir,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
