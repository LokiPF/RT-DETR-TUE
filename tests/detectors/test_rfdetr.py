from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision.transforms.v2.functional as F
from PIL import Image

WEIGHTS = Path("/home/yuchen/.roboflow/models/rf-detr-medium.pth")
IMAGE = Path("/home/yuchen/YuchenZ/Datasets/coco/val2017/000000000139.jpg")


@pytest.fixture(scope="module")
def adapter():
    if not (WEIGHTS.exists() and IMAGE.exists()):
        pytest.skip("RF-DETR-M's weights or the COCO image are not available")
    from degradation_monitor.detectors import load_adapter
    return load_adapter("rfdetr_m", WEIGHTS, "cpu")


def _image():
    return np.array(Image.open(IMAGE).convert("RGB"))  # writable, as the stages' images are


def test_rfdetr_blocks_regather_into_the_detectors_own_feature_maps(adapter):
    from degradation_monitor.detectors.rfdetr import MEANS, RESOLUTION, STDS

    image = _image()
    out = adapter([image])
    assert {n: tuple(m.shape) for n, m in out.levels.items()} == {n: (1, 384, 36, 36) for n in ("s1", "s2", "s3", "s4")}
    assert len(out.cdf) == 5 and tuple(out.pooled.shape) == (1, 384)
    backbone = adapter([image], heads=False)  # what the clean fits see
    assert backbone.decoder is None and torch.allclose(backbone.levels["s4"], out.levels["s4"], atol=1e-5)
    tensor = F.resize(torch.from_numpy(image).permute(2, 0, 1).float() / 255.0, [RESOLUTION, RESOLUTION],
                      antialias=False)
    with torch.inference_mode():
        official = adapter.backbone(F.normalize(tensor[None], MEANS, STDS))  # LayerNorm-ed blocks 3, 6, 9, 12
        mine = adapter.maps(adapter.layernorm(adapter._tokens[12]))
    assert torch.allclose(mine, official[-1], atol=1e-5)


def test_rfdetr_outputs_have_the_detr_parts(adapter):
    assert torch.get_float32_matmul_precision() == "highest"  # loading RF-DETR left the other detectors' matmuls alone
    out = adapter([_image(), _image()])
    assert out.query_logits.shape == (2, 300, 80) and out.query_boxes.shape == (2, 300, 4)
    assert tuple(out.decoder.shape) == (2, 300, 256)
    best = 1 / (1 + np.exp(-out.query_logits[0].max()))
    assert out.scores[0, 0] == pytest.approx(best, abs=1e-5) and (out.labels >= 0).all()
