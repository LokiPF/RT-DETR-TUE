#!/usr/bin/env python3
"""
Split a COCO instances annotation file into calibration and test subsets.

The split is performed at image level, so every image and all of its
annotations occur in exactly one output file. By default, a deterministic
greedy multilabel stratification approximately preserves:

  * class presence per image;
  * small/medium/large object presence per image; and
  * class-by-size presence per image.

Only Python's standard library is required.

Example:

    python tools/split_coco_validation.py \
        --annotations dataset/coco/annotations/instances_val2017.json \
        --calibration-output \
            dataset/coco/annotations/instances_val2017_calibration.json \
        --test-output \
            dataset/coco/annotations/instances_val2017_test.json \
        --calibration-fraction 0.5 \
        --seed 42

The image files are not copied. Both output JSON files continue to reference
the same image names in the original ``val2017`` image directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Hashable, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple


Feature = Tuple[str, ...]
ImageId = Hashable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a COCO validation annotation JSON into disjoint calibration "
            "and test files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="Input COCO instances annotation JSON.",
    )
    parser.add_argument(
        "--calibration-output",
        default=None,
        help=(
            "Calibration JSON. Defaults to "
            "<input_stem>_calibration.json beside the input."
        ),
    )
    parser.add_argument(
        "--test-output",
        default=None,
        help=(
            "Test JSON. Defaults to <input_stem>_test.json beside the input."
        ),
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.5,
        help="Fraction of images assigned to calibration.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--random-only",
        action="store_true",
        help="Use a simple random image split instead of stratification.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing output JSON files.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Write indented JSON. This makes the files considerably larger.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    input_path = Path(args.annotations).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {input_path}")

    calibration_path = (
        Path(args.calibration_output).expanduser().resolve()
        if args.calibration_output
        else input_path.with_name(f"{input_path.stem}_calibration.json")
    )
    test_path = (
        Path(args.test_output).expanduser().resolve()
        if args.test_output
        else input_path.with_name(f"{input_path.stem}_test.json")
    )

    if calibration_path == test_path:
        raise ValueError("Calibration and test output paths must differ.")
    if input_path in (calibration_path, test_path):
        raise ValueError("An output path cannot overwrite the input annotation file.")
    if not args.overwrite:
        existing = [path for path in (calibration_path, test_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "Output file already exists. Use --overwrite to replace it: "
                f"{existing}"
            )

    if not 0.0 < args.calibration_fraction < 1.0:
        raise ValueError("--calibration-fraction must be strictly between 0 and 1.")
    return input_path, calibration_path, test_path


def load_coco(path: Path) -> MutableMapping[str, Any]:
    print(f"[input] loading {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, MutableMapping):
        raise TypeError("Top-level COCO JSON value must be an object.")
    for key in ("images", "annotations", "categories"):
        if key not in data or not isinstance(data[key], list):
            raise KeyError(f"COCO JSON must contain a list named `{key}`.")
    if len(data["images"]) < 2:
        raise ValueError("At least two images are required for a split.")
    return data


def validate_coco(data: Mapping[str, Any]) -> Tuple[Dict[ImageId, Mapping[str, Any]], Set[Any]]:
    image_by_id: Dict[ImageId, Mapping[str, Any]] = {}
    for image in data["images"]:
        if not isinstance(image, Mapping) or "id" not in image:
            raise ValueError("Every image entry must be an object containing `id`.")
        image_id = image["id"]
        if image_id in image_by_id:
            raise ValueError(f"Duplicate image id: {image_id}")
        image_by_id[image_id] = image

    category_ids: Set[Any] = set()
    for category in data["categories"]:
        if not isinstance(category, Mapping) or "id" not in category:
            raise ValueError("Every category entry must contain `id`.")
        category_id = category["id"]
        if category_id in category_ids:
            raise ValueError(f"Duplicate category id: {category_id}")
        category_ids.add(category_id)

    annotation_ids: Set[Any] = set()
    for annotation in data["annotations"]:
        if not isinstance(annotation, Mapping):
            raise ValueError("Every annotation entry must be an object.")
        for key in ("id", "image_id", "category_id"):
            if key not in annotation:
                raise ValueError(f"Annotation is missing `{key}`: {annotation}")
        annotation_id = annotation["id"]
        if annotation_id in annotation_ids:
            raise ValueError(f"Duplicate annotation id: {annotation_id}")
        annotation_ids.add(annotation_id)
        if annotation["image_id"] not in image_by_id:
            raise ValueError(
                f"Annotation {annotation_id} references unknown image "
                f"{annotation['image_id']}."
            )
        if annotation["category_id"] not in category_ids:
            raise ValueError(
                f"Annotation {annotation_id} references unknown category "
                f"{annotation['category_id']}."
            )
    return image_by_id, category_ids


def annotation_area(annotation: Mapping[str, Any]) -> float:
    area = annotation.get("area")
    if isinstance(area, (int, float)) and math.isfinite(float(area)):
        return max(float(area), 0.0)

    bbox = annotation.get("bbox")
    if (
        isinstance(bbox, Sequence)
        and not isinstance(bbox, (str, bytes))
        and len(bbox) >= 4
    ):
        width = max(float(bbox[2]), 0.0)
        height = max(float(bbox[3]), 0.0)
        return width * height
    return 0.0


def size_bucket(annotation: Mapping[str, Any]) -> str:
    """COCO small/medium/large thresholds in absolute pixel area."""
    area = annotation_area(annotation)
    if area < 32.0**2:
        return "small"
    if area < 96.0**2:
        return "medium"
    return "large"


def build_image_features(
    image_ids: Iterable[ImageId],
    annotations: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[ImageId, Set[Feature]], Counter[Feature]]:
    features_by_image: Dict[ImageId, Set[Feature]] = {
        image_id: set() for image_id in image_ids
    }

    for annotation in annotations:
        image_id = annotation["image_id"]
        category_id = str(annotation["category_id"])
        bucket = size_bucket(annotation)
        features_by_image[image_id].update(
            {
                ("class", category_id),
                ("size", bucket),
                ("class_size", category_id, bucket),
            }
        )

    totals: Counter[Feature] = Counter()
    for features in features_by_image.values():
        totals.update(features)
    return features_by_image, totals


def random_split(
    image_ids: Sequence[ImageId],
    calibration_count: int,
    seed: int,
) -> Tuple[Set[ImageId], Set[ImageId]]:
    shuffled = list(image_ids)
    random.Random(seed).shuffle(shuffled)
    calibration_ids = set(shuffled[:calibration_count])
    test_ids = set(shuffled[calibration_count:])
    return calibration_ids, test_ids


def stratified_split(
    image_ids: Sequence[ImageId],
    features_by_image: Mapping[ImageId, Set[Feature]],
    feature_totals: Mapping[Feature, int],
    calibration_count: int,
    calibration_fraction: float,
    seed: int,
) -> Tuple[Set[ImageId], Set[ImageId]]:
    """
    Deterministic greedy multilabel split.

    Rare feature combinations are assigned first. Each assignment favors the
    subset with the larger normalized deficit for the image's features while
    respecting the exact requested image count.
    """
    rng = random.Random(seed)
    ordered_ids = list(image_ids)
    rng.shuffle(ordered_ids)

    def rarity_score(image_id: ImageId) -> float:
        return sum(
            1.0 / max(feature_totals[feature], 1)
            for feature in features_by_image[image_id]
        )

    # Python's sort is stable, so the seeded shuffle resolves equal scores.
    ordered_ids.sort(
        key=lambda image_id: (
            rarity_score(image_id),
            len(features_by_image[image_id]),
        ),
        reverse=True,
    )

    total_images = len(ordered_ids)
    test_count = total_images - calibration_count
    desired_calibration = {
        feature: count * calibration_fraction
        for feature, count in feature_totals.items()
    }
    desired_test = {
        feature: count * (1.0 - calibration_fraction)
        for feature, count in feature_totals.items()
    }

    calibration_ids: Set[ImageId] = set()
    test_ids: Set[ImageId] = set()
    calibration_features: Counter[Feature] = Counter()
    test_features: Counter[Feature] = Counter()

    for position, image_id in enumerate(ordered_ids):
        calibration_slots = calibration_count - len(calibration_ids)
        test_slots = test_count - len(test_ids)
        remaining_images = total_images - position

        if calibration_slots == 0:
            chosen = "test"
        elif test_slots == 0:
            chosen = "calibration"
        else:
            features = features_by_image[image_id]

            def feature_need(
                current: Mapping[Feature, int],
                desired: Mapping[Feature, float],
            ) -> float:
                if not features:
                    return 0.0
                return sum(
                    (desired[feature] - current.get(feature, 0))
                    / max(feature_totals[feature], 1)
                    for feature in features
                ) / len(features)

            calibration_need = feature_need(
                calibration_features,
                desired_calibration,
            )
            test_need = feature_need(test_features, desired_test)

            # Capacity pressure prevents the final few images from being forced
            # into one side irrespective of their features.
            calibration_score = (
                calibration_need + calibration_slots / remaining_images
            )
            test_score = test_need + test_slots / remaining_images

            if math.isclose(calibration_score, test_score, abs_tol=1e-12):
                chosen = "calibration" if rng.random() < 0.5 else "test"
            else:
                chosen = (
                    "calibration"
                    if calibration_score > test_score
                    else "test"
                )

        if chosen == "calibration":
            calibration_ids.add(image_id)
            calibration_features.update(features_by_image[image_id])
        else:
            test_ids.add(image_id)
            test_features.update(features_by_image[image_id])

    return calibration_ids, test_ids


def build_subset(
    data: Mapping[str, Any],
    selected_image_ids: Set[ImageId],
) -> Dict[str, Any]:
    subset: Dict[str, Any] = {
        key: value
        for key, value in data.items()
        if key not in ("images", "annotations")
    }
    subset["images"] = [
        image for image in data["images"] if image["id"] in selected_image_ids
    ]
    subset["annotations"] = [
        annotation
        for annotation in data["annotations"]
        if annotation["image_id"] in selected_image_ids
    ]
    return subset


def validate_split(
    data: Mapping[str, Any],
    calibration: Mapping[str, Any],
    test: Mapping[str, Any],
) -> None:
    original_ids = {image["id"] for image in data["images"]}
    calibration_ids = {image["id"] for image in calibration["images"]}
    test_ids = {image["id"] for image in test["images"]}

    overlap = calibration_ids & test_ids
    if overlap:
        raise RuntimeError(f"Split contains overlapping image IDs: {list(overlap)[:10]}")
    if calibration_ids | test_ids != original_ids:
        raise RuntimeError("Split image union does not equal the original image set.")

    original_annotation_ids = {
        annotation["id"] for annotation in data["annotations"]
    }
    calibration_annotation_ids = {
        annotation["id"] for annotation in calibration["annotations"]
    }
    test_annotation_ids = {
        annotation["id"] for annotation in test["annotations"]
    }
    if calibration_annotation_ids & test_annotation_ids:
        raise RuntimeError("An annotation occurs in both output subsets.")
    if (
        calibration_annotation_ids | test_annotation_ids
        != original_annotation_ids
    ):
        raise RuntimeError(
            "Split annotation union does not equal the original annotations."
        )


def write_json_atomic(
    path: Path,
    data: Mapping[str, Any],
    pretty: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            if pretty:
                json.dump(data, file, ensure_ascii=False, indent=2)
            else:
                json.dump(
                    data,
                    file,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def category_image_counts(
    annotations: Sequence[Mapping[str, Any]],
) -> Counter[Any]:
    images_by_category: Dict[Any, Set[ImageId]] = defaultdict(set)
    for annotation in annotations:
        images_by_category[annotation["category_id"]].add(
            annotation["image_id"]
        )
    return Counter(
        {
            category_id: len(image_ids)
            for category_id, image_ids in images_by_category.items()
        }
    )


def size_annotation_counts(
    annotations: Sequence[Mapping[str, Any]],
) -> Counter[str]:
    return Counter(size_bucket(annotation) for annotation in annotations)


def split_fingerprint(image_ids: Iterable[ImageId]) -> str:
    canonical = "\n".join(
        json.dumps(image_id, sort_keys=True)
        for image_id in sorted(image_ids, key=lambda value: str(value))
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def print_summary(
    name: str,
    subset: Mapping[str, Any],
    original: Mapping[str, Any],
    expected_fraction: float,
) -> None:
    image_count = len(subset["images"])
    annotation_count = len(subset["annotations"])
    image_fraction = image_count / len(original["images"])
    annotation_fraction = annotation_count / max(len(original["annotations"]), 1)
    sizes = size_annotation_counts(subset["annotations"])

    print(
        f"[{name}] images={image_count} ({image_fraction:.3%}), "
        f"annotations={annotation_count} ({annotation_fraction:.3%}), "
        f"sizes={{small:{sizes['small']}, medium:{sizes['medium']}, "
        f"large:{sizes['large']}}}, "
        f"image-id fingerprint="
        f"{split_fingerprint(image['id'] for image in subset['images'])}"
    )

    original_counts = category_image_counts(original["annotations"])
    subset_counts = category_image_counts(subset["annotations"])
    deviations: List[Tuple[float, Any, int, float]] = []
    for category_id, original_count in original_counts.items():
        expected = original_count * expected_fraction
        actual = subset_counts.get(category_id, 0)
        deviations.append(
            (
                abs(actual - expected) / max(original_count, 1),
                category_id,
                actual,
                expected,
            )
        )
    deviations.sort(reverse=True, key=lambda item: item[0])
    if deviations:
        largest = ", ".join(
            f"class {category_id}: {actual} vs {expected:.1f}"
            for _, category_id, actual, expected in deviations[:5]
        )
        print(f"[{name}] largest class-image deviations: {largest}")


def run(args: argparse.Namespace) -> Tuple[Path, Path]:
    input_path, calibration_path, test_path = resolve_paths(args)
    data = load_coco(input_path)
    image_by_id, _ = validate_coco(data)
    image_ids = list(image_by_id)

    calibration_count = int(
        round(len(image_ids) * args.calibration_fraction)
    )
    calibration_count = max(1, min(calibration_count, len(image_ids) - 1))

    if args.random_only:
        print("[split] using seeded random image split")
        calibration_ids, test_ids = random_split(
            image_ids,
            calibration_count,
            args.seed,
        )
    else:
        print(
            "[split] using class/size-aware greedy stratification "
            f"with seed={args.seed}"
        )
        features_by_image, feature_totals = build_image_features(
            image_ids,
            data["annotations"],
        )
        calibration_ids, test_ids = stratified_split(
            image_ids,
            features_by_image,
            feature_totals,
            calibration_count,
            args.calibration_fraction,
            args.seed,
        )

    calibration = build_subset(data, calibration_ids)
    test = build_subset(data, test_ids)
    validate_split(data, calibration, test)

    print_summary(
        "calibration",
        calibration,
        data,
        len(calibration_ids) / len(image_ids),
    )
    print_summary(
        "test",
        test,
        data,
        len(test_ids) / len(image_ids),
    )

    write_json_atomic(calibration_path, calibration, args.pretty)
    write_json_atomic(test_path, test, args.pretty)
    print(f"[output] calibration annotations: {calibration_path}")
    print(f"[output] test annotations: {test_path}")
    return calibration_path, test_path


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()