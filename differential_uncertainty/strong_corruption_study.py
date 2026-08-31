from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from .corruptions.imagecorruptions import ADDITIONAL_IMAGECORRUPTIONS
from .scoring import detect_padded_tail


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
    epsilon = torch.finfo(probabilities.dtype).eps
    probabilities = probabilities.clamp(epsilon, 1 - epsilon)
    negative_cost = 0.75 * probabilities.pow(2) * (-(1 - probabilities).log())
    positive_cost = 0.25 * (1 - probabilities).pow(2) * (-probabilities.log())
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
                annotation_lookup.get(image_id, ()),
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
        moment_vectors = (
            vectors.index_select(0, matched_ids),
            vectors.index_select(0, background_ids),
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
        moment_vectors = (vectors.index_select(0, eligible_ids),)

    sampled = vectors.index_select(0, sampled_ids)
    mean = None
    scale = None
    if distance == "mean_5_standardized_euclidean":
        if variant == "balanced":
            mean, scale = _balanced_moments(*moment_vectors)
        else:
            mean, scale = _population_moments(moment_vectors[0])
        sampled = (sampled - mean) / scale

    sampled_matched = matched.index_select(0, sampled_ids)
    return Bank(
        vectors=sampled.contiguous(),
        mean=mean,
        scale=scale,
        matched_count=int(sampled_matched.sum()),
        background_count=int((~sampled_matched).sum()),
    )
