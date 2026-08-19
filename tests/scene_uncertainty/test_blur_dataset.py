import json

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.dataset.coco_dataset import ConvertCocoPolysToMask
from src.scene_uncertainty.blur import FixedGaussianBlur
from src.scene_uncertainty.dataset import make_coco_loader


def _write_mini_coco(root):
    """Three 8x8 images, black left half / white right half, one box each."""
    image_dir = root / "images"
    image_dir.mkdir()
    array = np.zeros((8, 8, 3), dtype=np.uint8)
    array[:, 4:] = 255
    images = []
    annotations = []
    for image_id in (1, 2, 3):
        file_name = f"{image_id:012d}.png"
        Image.fromarray(array).save(image_dir / file_name)
        images.append({"id": image_id, "file_name": file_name, "width": 8, "height": 8})
        annotations.append(
            {
                "id": 100 + image_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [1, 1, 4, 4],
                "area": 16,
                "iscrowd": 0,
            }
        )
    annotation_file = root / "instances.json"
    annotation_file.write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "person"}],
            }
        )
    )
    return image_dir, annotation_file


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
