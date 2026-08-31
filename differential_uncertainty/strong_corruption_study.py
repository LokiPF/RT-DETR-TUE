from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from .corruptions.imagecorruptions import ADDITIONAL_IMAGECORRUPTIONS
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
