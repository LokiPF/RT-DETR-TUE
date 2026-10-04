from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

WEIGHTS = Path("/home/yuchen/YuchenZ/lab/Detector_test/yolo11m.pt")
IMAGE = Path("/home/yuchen/YuchenZ/Datasets/coco/val2017/000000000139.jpg")  # 426 rows x 640 columns


@pytest.fixture(scope="module")
def adapter():
    if not (WEIGHTS.exists() and IMAGE.exists()):
        pytest.skip("YOLO11m's weights or the COCO image are not available")
    from degradation_monitor.detectors import load_adapter
    return load_adapter("yolo11m", WEIGHTS, "cpu")


def _image():
    return np.array(Image.open(IMAGE).convert("RGB"))  # writable, as the stages' images are


def test_yolo_levels_and_cdf_maps_have_the_backbone_shapes(adapter):
    image = _image()
    out = adapter([image, image])
    # 426 x 640 letterboxes to 448 x 640, 11 grey rows above and below; the crops keep only the image's cells
    assert {n: tuple(m.shape) for n, m in out.levels.items()} == {
        "s1": (2, 128, 106, 160), "s2": (2, 256, 52, 80), "s3": (2, 512, 26, 40), "s4": (2, 512, 12, 20)}
    assert tuple(out.cdf[0].shape) == (2, 64, 212, 320)  # the stride-2 stem, its 6 grey rows above cropped away
    assert [m.shape[1] for m in out.cdf] == [64, 256, 512, 512, 512] and tuple(out.pooled.shape) == (2, 512)
    assert out.scores.shape == (2, 100) and out.query_logits is None and out.decoder is None
    backbone = adapter([image], heads=False)  # what the clean fits see
    assert backbone.scores is None
    assert torch.allclose(backbone.levels["s1"], out.levels["s1"][:1], atol=1e-4)
    assert torch.allclose(backbone.pooled, out.pooled[:1], atol=1e-4)


def test_yolo_detections_match_ultralytics_predict(adapter):
    from ultralytics import YOLO

    image = _image()
    out = adapter([image])
    reference = YOLO(str(WEIGHTS)).predict(image[..., ::-1].copy(), conf=0.001, iou=0.7, max_det=100, imgsz=640,
                                           device="cpu", verbose=False)[0]  # Ultralytics takes numpy images as BGR
    confidences = reference.boxes.conf.numpy()
    expected = np.sort(confidences)[::-1]
    assert out.scores[0, :len(expected)] == pytest.approx(expected, abs=1e-3)
    best = int(np.argmax(confidences))
    assert out.labels[0, 0] == int(reference.boxes.cls[best])
    assert out.boxes[0, 0] == pytest.approx(reference.boxes.xyxy[best].numpy(), abs=1.0)


def test_yolo_crops_an_odd_letterbox_padding_where_ultralytics_puts_the_image(adapter):
    image = np.random.default_rng(0).integers(0, 256, (427, 640, 3), dtype=np.uint8)  # 21 grey rows: 10 above, 11 below
    out = adapter([image], heads=False)
    assert tuple(out.cdf[0].shape) == (1, 64, 213, 320)  # the stride-2 stem keeps the cell of rows 10-11
