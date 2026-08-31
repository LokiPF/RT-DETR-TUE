from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from .corruptions.imagecorruptions import ADDITIONAL_IMAGECORRUPTIONS
from .evaluation import binary_auroc
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
