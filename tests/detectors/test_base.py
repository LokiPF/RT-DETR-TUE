import json
from pathlib import Path

import numpy as np
import pytest
import torch

from degradation_monitor.detectors import COCO_CATEGORY_IDS, LABEL_OF_CATEGORY, Region, load_adapter, padded

ANNOTATIONS = Path("/home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json")


def test_padded_keeps_the_most_confident_and_fills_empty_slots():
    scores, labels, boxes = padded(np.array([0.2, 0.9, 0.5]), np.array([3, 1, 2]), np.arange(12.0).reshape(3, 4), k=5)
    assert scores.tolist() == pytest.approx([0.9, 0.5, 0.2, 0.0, 0.0])
    assert labels.tolist() == [1, 2, 3, 0, 0]
    assert boxes[0].tolist() == [4, 5, 6, 7] and not boxes[3:].any()


def test_region_crops_the_padding_away():
    maps = torch.arange(10 * 8, dtype=torch.float32).reshape(1, 1, 10, 8)  # a stride-4 map of a 40 x 32 input
    region = Region(top=4.0, left=0.0, height=28.0, width=30.0)  # 4 grey rows above, 8 below, 2 columns right
    cropped = region.crop(maps, stride=4)
    assert tuple(cropped.shape) == (1, 1, 7, 7) and cropped[0, 0, 0, 0] == maps[0, 0, 1, 0]


def test_the_category_mapping_matches_the_coco_annotations():
    assert len(COCO_CATEGORY_IDS) == 80 and LABEL_OF_CATEGORY[1] == 0 and LABEL_OF_CATEGORY[90] == 79
    assert LABEL_OF_CATEGORY[12] == -1
    if ANNOTATIONS.exists():
        ids = sorted(c["id"] for c in json.loads(ANNOTATIONS.read_text())["categories"])
        assert tuple(ids) == COCO_CATEGORY_IDS


def test_an_unknown_detector_is_refused():
    with pytest.raises(ValueError, match="unknown detector 'ssd'"):
        load_adapter("ssd", None, "cpu")
