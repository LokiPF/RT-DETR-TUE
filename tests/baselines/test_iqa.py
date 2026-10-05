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


WEIGHTS = [Path(torch.hub.get_dir()) / p for p in ("checkpoints/ARNIQA.pth", "checkpoints/resnet50-0676ba61.pth",
                                                   "pyiqa/regressor_kadid10k.pth", "pyiqa/niqe_modelparameters.mat",
                                                   "clip/RN50.pt")]
needs_weights = pytest.mark.skipif(not (IMAGE.exists() and all(p.exists() for p in WEIGHTS)),
                                   reason="the COCO image or the IQA weights are absent")


@pytest.fixture(scope="module")
def models():
    if not (IMAGE.exists() and all(p.exists() for p in WEIGHTS)):
        pytest.skip("the COCO image or the IQA weights are absent")
    return iqa.load_iqa_models("cpu")


def _image():
    return np.array(Image.open(IMAGE).convert("RGB"))  # writable, as the stages' images are


@needs_weights
def test_arniqa_follows_the_papers_five_crops_and_regresses_as_pyiqa(models):
    import pyiqa
    import torch.nn.functional as F
    from torchvision import transforms
    from torchvision.transforms import functional as TVF

    image = Image.open(IMAGE).convert("RGB")
    embedding = models.arniqa.embed([np.array(image)])
    assert embedding.shape == (1, 4096)
    # ARNIQA's test pipeline as test.py runs it: the half-size image from PIL's default resize (bicubic), the centre
    # and the four corner crops of both, ToTensor, ImageNet normalisation; each crop's features L2-normalised
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def crops(img):
        w, h = img.size
        tops_lefts = [(h // 2 - 112, w // 2 - 112), (0, 0), (h - 224, 0), (0, w - 224), (h - 224, w - 224)]
        return torch.stack([normalize(transforms.ToTensor()(TVF.crop(img, t, l, 224, 224))) for t, l in tops_lefts])

    encoder = models.arniqa.model.encoder
    with torch.inference_mode():
        full = F.normalize(encoder(crops(image)).flatten(1), dim=1)
        half = F.normalize(encoder(crops(image.resize((image.size[0] // 2, image.size[1] // 2)))).flatten(1), dim=1)
    per_crop = torch.hstack((full, half))  # test.py's features: one row per crop
    assert torch.allclose(embedding[0], per_crop.mean(dim=0), atol=1e-5)
    mean_of_crops = models.arniqa.quality(per_crop).mean().item()  # test.py averages the five crops' scores
    assert models.arniqa.quality(embedding).item() == pytest.approx(mean_of_crops, abs=1e-5)
    x = _unit(np.array(image))  # the regressor and its scaling are pyiqa's own: the same score on pyiqa's features
    with torch.inference_mode():
        whole, small = models.arniqa.model._preprocess(x)
        theirs = torch.hstack((F.normalize(encoder(whole).flatten(1), dim=1),
                               F.normalize(encoder(small).flatten(1), dim=1)))
        metric = pyiqa.create_metric("arniqa-kadid", device="cpu")(x)
    assert models.arniqa.quality(theirs).item() == pytest.approx(float(metric), rel=1e-5)


@needs_weights
def test_clipiqa_uses_the_papers_prompt_pair(models):
    from pyiqa.archs.clip_imports import clip
    from pyiqa.archs.clipiqa_arch import CLIPIQA

    assert torch.equal(models.clipiqa.tokens, clip.tokenize(["Good photo.", "Bad photo."]))
    x = _unit(_image())
    one = models.clipiqa.quality(x)
    two = models.clipiqa.quality(torch.cat([x, x]))
    assert one.shape == (1,) and 0.0 < one.item() < 1.0 and torch.allclose(two, one.repeat(2), atol=1e-5)
    port = CLIPIQA(model_type="clipiqa").eval()  # pyiqa's port, fp32 on the CPU, given the paper's single pair
    port.prompt_pairs = clip.tokenize(["Good photo.", "Bad photo."])
    with torch.inference_mode():
        assert one.item() == pytest.approx(port(x).item(), abs=1e-5)


@needs_weights
def test_the_scores_rise_with_strong_noise(models):
    from degradation_monitor import corruptions

    clean = _image()
    noisy = np.array(corruptions.corrupt(Image.fromarray(clean), IMAGE.name, "gaussian_noise", 5))
    models.niqe_refit = iqa.published_niqe()  # stand-ins for the clean references the fit stage builds
    models.prototype = models.arniqa.embed([clean])[0].numpy()
    scores = models.scores([clean, noisy])
    assert set(scores) == {*iqa.ROWS, "niqe_blocks"} and (scores["niqe_blocks"] >= iqa.NIQE_MIN_BLOCKS).all()
    assert all(scores[row].shape == (2,) and np.isfinite(scores[row]).all() for row in iqa.ROWS)
    for row in iqa.ROWS:
        assert scores[row][1] > scores[row][0], row
    assert scores["arniqa_proto"][0] == pytest.approx(0.0, abs=1e-5)


@needs_weights
@pytest.mark.filterwarnings(r"ignore:cov\(\)")  # torch.cov warns on the one-block version, as it should
def test_niqe_scores_a_version_with_fewer_than_two_blocks_as_most_degraded(models, monkeypatch):
    models.niqe_refit, models.prototype = iqa.published_niqe(), np.full(4096, 1 / 64.0)
    features = torch.from_numpy(np.random.default_rng(1).uniform(0.1, 1.0, (3, 4, 36)))
    features[1, 1:, 0] = float("nan")  # one block left without a NaN
    features[2, :, 5] = float("nan")  # none left
    monkeypatch.setattr(iqa, "niqe_block_features", lambda luma: (features, None))
    rows = models.niqe_rows([_image()] * 3)
    assert rows["niqe_blocks"].tolist() == [4, 1, 0]
    assert 0 < rows["niqe"][0] < iqa.NIQE_UNSCORABLE and 0 < rows["niqe_default"][0] < iqa.NIQE_UNSCORABLE
    assert rows["niqe"][1:].tolist() == rows["niqe_default"][1:].tolist() == [iqa.NIQE_UNSCORABLE] * 2


@needs_weights
def test_the_scores_need_the_clean_references(models):
    models.niqe_refit = None
    with pytest.raises(ValueError, match="run the fit stage first"):
        models.scores([_image()])
