from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

WEIGHTS = Path("/home/yuchen/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth")
IMAGE = Path("/home/yuchen/YuchenZ/Datasets/coco/val2017/000000000139.jpg")


@pytest.fixture(scope="module")
def adapter():
    if not (WEIGHTS.exists() and IMAGE.exists()):
        pytest.skip("Faster R-CNN's weights or the COCO image are not available")
    from degradation_monitor.detectors import load_adapter
    return load_adapter("faster_rcnn_r50_fpn_v2", WEIGHTS, "cpu")


def _image():
    return np.array(Image.open(IMAGE).convert("RGB"))  # writable, as the stages' images are


def test_faster_rcnn_levels_have_the_resnet50_shapes_without_padding(adapter):
    image = _image()
    out = adapter([image])
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    height, width = adapter.model.transform([tensor])[0].image_sizes[0]  # inside the batch padded to 32
    expected = {"s1": (256, 4), "s2": (512, 8), "s3": (1024, 16), "s4": (2048, 32)}
    for name, (channels, stride) in expected.items():
        assert tuple(out.levels[name].shape) == (1, channels, height // stride, width // stride)
    assert [m.shape[1] for m in out.cdf] == [64, 256, 512, 1024, 2048] and tuple(out.pooled.shape) == (1, 2048)
    backbone = adapter([image], heads=False)  # what the clean fits see
    assert backbone.scores is None and torch.allclose(backbone.levels["s2"], out.levels["s2"], atol=1e-4)


def test_faster_rcnn_detections_match_the_model(adapter):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

    from degradation_monitor.detectors import LABEL_OF_CATEGORY

    image = _image()
    out = adapter([image])
    model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    model.roi_heads.score_thresh, model.roi_heads.detections_per_img = 0.001, 100
    with torch.inference_mode():
        result = model.eval()([torch.from_numpy(image).permute(2, 0, 1).float() / 255.0])[0]
    n = len(result["scores"])
    assert out.scores[0, :n] == pytest.approx(result["scores"].numpy(), abs=1e-4)
    assert out.labels[0, :n].tolist() == LABEL_OF_CATEGORY[result["labels"].numpy()].tolist()
    assert (out.labels[0] >= 0).all() and out.scores[0, n:].sum() == 0


def test_faster_rcnn_refuses_a_batch_of_two_image_sizes(adapter):
    arrays = [np.zeros((48, 64, 3), np.uint8), np.zeros((47, 64, 3), np.uint8)]  # the transform pads them together
    with pytest.raises(ValueError, match="an adapter batch must hold images of one size"):
        adapter(arrays, heads=False)  # the crop would follow the first image alone
