from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from degradation_monitor.baselines import iqa

IMAGE = Path("/home/yuchen/YuchenZ/Datasets/coco/val2017/000000000139.jpg")
NIQE_MODEL = Path(torch.hub.get_dir()) / "pyiqa" / "niqe_modelparameters.mat"


def _unit(array):
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)[None].float() / 255.0


@pytest.mark.skipif(not (IMAGE.exists() and NIQE_MODEL.exists()), reason="the COCO image or NIQE's model is absent")
def test_niqe_matches_pyiqa_with_the_published_model():
    import pyiqa

    x = _unit(np.array(Image.open(IMAGE).convert("RGB")))
    mu, cov = iqa.niqe_test_model(iqa.niqe_block_features(iqa.niqe_luma(x))[0])
    ours = iqa.niqe_distance(*iqa.published_niqe(), mu, cov)
    with torch.inference_mode():
        theirs = pyiqa.create_metric("niqe", device="cpu")(x)
    assert ours.item() == pytest.approx(float(theirs), rel=1e-6)


def test_niqe_fit_keeps_the_sharp_blocks_and_drops_nan_rows():
    rng = np.random.default_rng(0)
    fit = iqa.NiqeFit()
    for _ in range(12):
        luma = rng.uniform(0, 255, (1, 1, 192, 288)).round()  # six blocks of noise ...
        luma[..., :96, :96] = 128.0  # ... one of them flat, so never among the sharp blocks
        features, sharpness = iqa.niqe_block_features(torch.from_numpy(luma))
        assert features.shape == (1, 6, 36) and sharpness.shape == (1, 6)
        fit.update(features, sharpness)
    mu, cov, blocks = fit.result()
    assert mu.shape == (36,) and cov.shape == (36, 36) and np.allclose(cov, cov.T)
    assert fit.images == 12 and 36 < blocks <= 12 * 5 and fit.dropped == 0
    fit.rows.append(np.full((2, 36), np.nan))
    assert fit.result()[2] == blocks and fit.dropped == 2


def test_niqe_needs_one_whole_block():
    with pytest.raises(ValueError, match="at least one whole 96 x 96 block"):
        iqa.niqe_block_features(torch.zeros(1, 1, 64, 300, dtype=torch.float64))
