import json

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.dataset.coco_dataset import ConvertCocoPolysToMask
from src.scene_uncertainty.blur import FixedGaussianBlur
from src.scene_uncertainty.dataset import make_coco_loader


_CATEGORIES = [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "bicycle"},
    {"id": 3, "name": "car"},
]


def _annotation(annotation_id, image_id, bbox, category_id=1):
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "area": bbox[2] * bbox[3],
        "iscrowd": 0,
    }


def _write_coco(root, image_ids, annotations):
    """8x8 images, black left half / white right half, plus the given annotations."""
    image_dir = root / "images"
    image_dir.mkdir()
    array = np.zeros((8, 8, 3), dtype=np.uint8)
    array[:, 4:] = 255
    images = []
    for image_id in image_ids:
        file_name = f"{image_id:012d}.png"
        Image.fromarray(array).save(image_dir / file_name)
        images.append({"id": image_id, "file_name": file_name, "width": 8, "height": 8})
    annotation_file = root / "instances.json"
    annotation_file.write_text(
        json.dumps(
            {"images": images, "annotations": annotations, "categories": _CATEGORIES}
        )
    )
    return image_dir, annotation_file


def _write_mini_coco(root):
    """Three 8x8 images, one full-size box each."""
    image_ids = (1, 2, 3)
    return _write_coco(
        root,
        image_ids,
        [_annotation(100 + image_id, image_id, [1, 1, 4, 4]) for image_id in image_ids],
    )


def test_converter_preserves_filtered_annotation_ids():
    image = Image.new("RGB", (20, 20), color="white")
    target = {
        "image_id": 9,
        "annotations": [
            {"id": 101, "bbox": [1, 1, 5, 5], "category_id": 3, "area": 25, "iscrowd": 0},
            {"id": 102, "bbox": [2, 2, 0, 4], "category_id": 4, "area": 0, "iscrowd": 0},
        ],
    }
    _, converted = ConvertCocoPolysToMask(False)(image, target, category2label={3: 0, 4: 1})
    assert converted["annotation_ids"].tolist() == [101]


def test_converter_keeps_annotation_ids_aligned_with_boxes():
    image = Image.new("RGB", (20, 20), color="white")
    target = {
        "image_id": 9,
        "annotations": [
            {"id": 101, "bbox": [1, 1, 5, 5], "category_id": 3, "area": 25, "iscrowd": 0},
            {"id": 102, "bbox": [2, 2, 0, 4], "category_id": 4, "area": 0, "iscrowd": 0},
            {"id": 103, "bbox": [3, 3, 6, 6], "category_id": 5, "area": 36, "iscrowd": 0},
        ],
    }
    _, converted = ConvertCocoPolysToMask(False)(
        image, target, category2label={3: 0, 4: 1, 5: 2}
    )
    assert converted["annotation_ids"].tolist() == [101, 103]
    assert converted["labels"].tolist() == [0, 2]
    assert converted["boxes"][:, 0].tolist() == [1.0, 3.0]


def test_blur_radius_zero_is_byte_exact():
    array = np.arange(12 * 12 * 3, dtype=np.uint8).reshape(12, 12, 3)
    image = Image.fromarray(array)
    output = FixedGaussianBlur(0)(image)
    assert np.array_equal(np.asarray(output), array)


def test_positive_blur_changes_a_sharp_edge():
    array = np.zeros((21, 21, 3), dtype=np.uint8)
    array[:, 10:] = 255
    image = Image.fromarray(array)
    output = FixedGaussianBlur(4)(image)
    assert not torch.equal(torch.from_numpy(np.asarray(output).copy()), torch.from_numpy(array))


def test_negative_blur_radius_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        FixedGaussianBlur(-1)


def test_make_coco_loader_pins_requested_ids_in_order(tmp_path):
    image_dir, annotation_file = _write_mini_coco(tmp_path)
    loader = make_coco_loader(image_dir, annotation_file, [3, 1], 0.0, 1, 0)
    assert list(loader.dataset.ids) == [3, 1]
    loaded_ids = [int(target["image_id"].item()) for _, targets in loader for target in targets]
    assert loaded_ids == [3, 1]


def test_make_coco_loader_rejects_missing_image_ids(tmp_path):
    image_dir, annotation_file = _write_mini_coco(tmp_path)
    with pytest.raises(KeyError, match=r"missing: \[7\]"):
        make_coco_loader(image_dir, annotation_file, [1, 7], 0.0, 1, 0)


def test_blur_is_applied_after_resize(tmp_path):
    image_dir, annotation_file = _write_mini_coco(tmp_path)
    sharp, _ = next(iter(make_coco_loader(image_dir, annotation_file, [1], 0.0, 1, 0)))
    blurred, _ = next(iter(make_coco_loader(image_dir, annotation_file, [1], 4.0, 1, 0)))
    assert not torch.equal(blurred, sharp)
    # An 8x8 source blurred before upscaling smears its corners toward grey; blurring
    # the 640x640 resize leaves them saturated because the flat regions are far wider
    # than the fixed radius.
    assert torch.equal(blurred[0, :, 0, 0], torch.zeros(3))
    assert torch.equal(blurred[0, :, 0, -1], torch.ones(3))


def test_sanitizer_drops_annotation_ids_together_with_their_boxes(tmp_path):
    # The middle box is 0.001px wide in an 8x8 source, so it survives the converter's
    # `keep` mask but is 0.08px wide after the 80x resize and SanitizeBoundingBoxes
    # drops it. The ids must lose the same entry, not merely the right count of them.
    image_dir, annotation_file = _write_coco(
        tmp_path,
        (1,),
        [
            _annotation(201, 1, [1, 1, 4, 4], category_id=1),
            _annotation(202, 1, [5, 1, 0.001, 4], category_id=2),
            _annotation(203, 1, [1, 5, 4, 2], category_id=3),
        ],
    )
    loader = make_coco_loader(image_dir, annotation_file, [1], 0.0, 1, 0)
    _, (target,) = next(iter(loader))
    assert target["annotation_ids"].tolist() == [201, 203]
    # Pair each surviving id with its own box rather than trusting the count: 201 is
    # the upper box (normalised cy 0.375) and 203 the lower one (cy 0.75).
    assert target["boxes"][:, 1].tolist() == pytest.approx([0.375, 0.75])
    assert target["labels"].tolist() == [0, 2]
