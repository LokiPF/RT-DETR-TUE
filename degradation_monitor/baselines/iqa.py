"""Four detector-free image-quality baselines, each scored so that higher means more likely degraded.

- NIQE (Mittal, Soundararajan & Bovik, IEEE SPL 2013): the distance between a multivariate Gaussian fitted to an
  image's block features and one fitted to pristine images. The main row's pristine model is refitted on clean train
  images with the authors' recipe (their estimatemodelparam.m: 96 x 96 blocks at two scales, keeping the blocks
  sharper than 0.75 times the image's sharpest); the sensitivity row uses the published model. Both rows share the
  test image's features, computed as pyiqa 0.1.16's NIQE does: the rounded luma, in float64.
- ARNIQA (Agnolucci, Galteri, Bertini & Del Bimbo, WACV 2024), as its paper evaluates it (the official test.py): the
  centre and the four corner crops, 224 x 224, of the image and of its half-size version (PIL's bicubic resize); per
  crop, the ResNet-50 features at both sizes, each L2-normalised and concatenated (return_embedding, 4096 values);
  averaged over the five crops. On the GPU the encoder runs under autocast, as the official code runs it. Quality:
  the KADID-10k regressor on that embedding; the regressor is linear, so this is the mean of the crops' qualities, as
  test.py averages them. The prototype row (Becker, Weiss, Hübner & Arens, arXiv 2602.18394): 1 - cosine similarity
  to the mean embedding of clean train images.
- CLIP-IQA (Wang, Chan & Loy, AAAI 2023), zero-shot, as its official code computes it (IceClear/CLIP-IQA, the
  CLIPIQAFixed model of configs/clipiqa/clipiqa_attribute_test.py): CLIP RN50 on the image at its own size with the
  positional embedding removed, the softmax of its logits for "Good photo." against "Bad photo.", the first
  probability. On the GPU the weights are fp16, as the official build_model makes them; on the CPU, fp32.
- NIQE in the scores: a version with no block free of NaN (snow and frost can blank blocks) has no NIQE features, and
  pyiqa's NIQE is NaN for it. Both NIQE rows give it NIQE_UNSCORABLE, above every real distance, so it counts as
  degraded; niqe_blocks records each version's count.

Every model reads the uint8 RGB image at its own size.
"""
from __future__ import annotations

import contextlib
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

NIQE_BLOCK = 96
NIQE_SHARPNESS = 0.75  # estimatemodelparam.m's sh_th: keep the blocks sharper than 0.75 times the sharpest
NIQE_FEATURES = 36  # 18 at full size, 18 at half size


def to_unit_tensor(arrays, device) -> torch.Tensor:
    """Same-size uint8 RGB arrays -> (N, 3, H, W) float32 in [0, 1] on the device."""
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(device).float().div_(255.0)


def niqe_luma(images: torch.Tensor) -> torch.Tensor:
    """(N, 3, H, W) RGB in [0, 1] -> (N, 1, H, W): the rounded luma in [0, 255], float64, as pyiqa's NIQE."""
    from pyiqa.archs.func_util import diff_round
    from pyiqa.utils.color_util import to_y_channel

    return diff_round(to_y_channel(images, 255, "yiq")).to(torch.float64)


def niqe_block_features(luma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Each whole 96 x 96 block's 36 features (18 at full size, 18 at half size) and its sharpness, the mean local
    standard deviation at full size: (N, B, 36) and (N, B). Partial blocks at the right and bottom are dropped."""
    from pyiqa.archs.func_util import normalize_img_with_gauss, safe_sqrt
    from pyiqa.archs.niqe_arch import compute_feature
    from pyiqa.matlab_utils import blockproc, fspecial, imfilter, imresize

    height, width = luma.shape[-2:]
    rows, cols = height // NIQE_BLOCK, width // NIQE_BLOCK
    if rows == 0 or cols == 0:
        raise ValueError(f"NIQE needs at least one whole {NIQE_BLOCK} x {NIQE_BLOCK} block, got {height} x {width}")
    image = luma[..., :rows * NIQE_BLOCK, :cols * NIQE_BLOCK]
    kernel = fspecial(7, 7.0 / 6, 1).to(image)  # the window normalize_img_with_gauss uses
    mu = imfilter(image, kernel, padding="replicate")
    sigma = safe_sqrt((imfilter(image ** 2, kernel, padding="replicate") - mu ** 2).abs())
    sharpness = blockproc(sigma, [NIQE_BLOCK, NIQE_BLOCK], fun=lambda blocks, _: blocks.mean(dim=(2, 3)))[..., 0]
    features = []
    for scale in (1, 2):
        normalized = normalize_img_with_gauss(image, padding="replicate")
        features.append(blockproc(normalized, [NIQE_BLOCK // scale, NIQE_BLOCK // scale], fun=compute_feature))
        if scale == 1:
            image = imresize(image / 255.0, scale=0.5, antialiasing=True) * 255.0
    return torch.cat(features, dim=-1), sharpness


def niqe_test_model(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Each image's Gaussian over its blocks: mean (N, 36) and covariance (N, 36, 36), blocks with a NaN left out."""
    from pyiqa.matlab_utils import nancov, nanmean

    return nanmean(features, dim=1), nancov(features)


def niqe_distance(mu_pristine, cov_pristine, mu: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    """NIQE's Eq. 10: the distance between the pristine Gaussian and each image's, (N,)."""
    mu_pristine = torch.as_tensor(mu_pristine).to(mu)
    cov_pristine = torch.as_tensor(cov_pristine).to(cov)
    diff = (mu_pristine - mu).unsqueeze(1)
    inverse = torch.linalg.pinv((cov_pristine + cov) / 2)
    return torch.bmm(torch.bmm(diff, inverse), diff.transpose(1, 2)).reshape(-1).sqrt()


def published_niqe() -> tuple[np.ndarray, np.ndarray]:
    """The authors' published pristine model (their modelparameters.mat), as pyiqa's 'niqe' metric loads it."""
    from pyiqa.archs.niqe_arch import NIQE

    model = NIQE(version="original")
    return model.mu_pris_param.numpy(), model.cov_pris_param.numpy()


class NiqeFit:
    """NIQE's pristine Gaussian from clean images' sharp blocks (estimatemodelparam.m)."""

    def __init__(self):
        self.rows, self.images, self.dropped = [], 0, 0

    def update(self, features: torch.Tensor, sharpness: torch.Tensor) -> None:
        for image_features, image_sharpness in zip(features, sharpness):
            keep = image_sharpness > NIQE_SHARPNESS * image_sharpness.max()
            self.rows.append(image_features[keep].double().cpu().numpy())
            self.images += 1

    def result(self) -> tuple[np.ndarray, np.ndarray, int]:
        """Mean (36,), covariance (36, 36) with N - 1, and the number of blocks. Blocks with a NaN are left out of
        both and counted in .dropped (MATLAB's nanmean would keep their other features in the mean; the sharp blocks
        kept hold no NaN in practice)."""
        rows = np.concatenate(self.rows) if self.rows else np.zeros((0, NIQE_FEATURES))
        nan = np.isnan(rows).any(axis=1)
        self.dropped, rows = int(nan.sum()), rows[~nan]
        if len(rows) <= NIQE_FEATURES:
            raise ValueError(f"NIQE's pristine model needs more than {NIQE_FEATURES} blocks, got {len(rows)}")
        return rows.mean(axis=0), np.cov(rows, rowvar=False), len(rows)


CLIP_PROMPTS = ("Good photo.", "Bad photo.")
ARNIQA_REGRESSOR = "kadid"
ARNIQA_CROP = 224  # the paper's crops: the centre and the four corners, of the image and of its half-size version
# a version needs one block without a NaN: with none, pyiqa's NIQE is NaN; with one, its covariance is zero and its
# score finite
NIQE_MIN_BLOCKS = 1
NIQE_UNSCORABLE = 1e6  # such a version's NIQE score: above every real distance, so it counts as degraded
ROWS = ("niqe", "niqe_default", "arniqa", "arniqa_proto", "clipiqa")


def arniqa_crops(image: Image.Image) -> list:
    """ARNIQA's center_corners_crop (utils/utils_data.py): the centre and the four corners of a PIL image, 224 x 224,
    in that order, padded with zeros where the image is smaller, as torchvision's crop of a PIL image pads."""
    width, height, size = image.width, image.height, ARNIQA_CROP
    corners = [(width // 2 - size // 2, height // 2 - size // 2), (0, 0), (0, height - size), (width - size, 0),
               (width - size, height - size)]
    return [image.crop((left, top, left + size, top + size)) for left, top in corners]


class Arniqa:
    """ARNIQA's encoder and KADID-10k regressor, on the paper's five crops of the image and of its half-size version."""

    def __init__(self, device):
        from pyiqa.archs.arniqa_arch import ARNIQA

        self.device = torch.device(device)
        self.model = ARNIQA(regressor_dataset=ARNIQA_REGRESSOR).to(self.device).eval().requires_grad_(False)
        self.mean = self.model.default_mean.to(self.device)
        self.std = self.model.default_std.to(self.device)

    def _features(self, crops) -> torch.Tensor:
        """(C, 2048): the encoder's features of PIL crops, each L2-normalised. On the GPU the encoder runs under
        autocast, as the official code runs it."""
        x = torch.from_numpy(np.stack([np.asarray(crop) for crop in crops])).permute(0, 3, 1, 2).to(self.device)
        x = (x.float().div_(255.0) - self.mean) / self.std
        precision = torch.autocast("cuda", dtype=torch.float16) if self.device.type == "cuda" \
            else contextlib.nullcontext()
        with precision:
            features = self.model.encoder(x)
        return F.normalize(features.float().flatten(1), dim=1)

    @torch.inference_mode()
    def embed(self, arrays) -> torch.Tensor:
        """uint8 RGB arrays (H, W, 3), of any sizes -> (N, 4096): for each image, test.py's features of its five crops,
        the full- and half-size features each L2-normalised and concatenated, averaged over the crops."""
        full, half = [], []
        for array in arrays:
            image = Image.fromarray(array)
            full += arniqa_crops(image)
            # the official resize_crop: PIL's resize, whose default filter for RGB is bicubic
            half += arniqa_crops(image.resize((image.width // 2, image.height // 2), Image.Resampling.BICUBIC))
        crops = torch.cat([self._features(full), self._features(half)], dim=1)
        return crops.view(len(arrays), 5, -1).mean(dim=1)  # five crops per image

    @torch.inference_mode()
    def quality(self, embedding: torch.Tensor) -> torch.Tensor:
        """The KADID-10k regressor's quality, scaled from KADID-10k's rating range to about [0, 1] as pyiqa scales it,
        higher is better: (N,). The regressor is linear, so on a mean embedding it gives the mean quality."""
        return self.model._scale_score(self.model.regressor(embedding)).reshape(-1)


class ClipIqa:
    """Zero-shot CLIP-IQA: the official CLIPIQAFixed.forward on pyiqa's copy of the modified CLIP."""

    def __init__(self, device):
        from pyiqa.archs.clip_imports import clip
        from pyiqa.archs.clip_model import load
        from pyiqa.archs.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

        self.device = torch.device(device)
        # fp16 weights on the GPU, as the official build_model's convert_weights; load() keeps fp32 on the CPU only
        self.clip = load("RN50", self.device).eval().requires_grad_(False)
        self.tokens = clip.tokenize(list(CLIP_PROMPTS)).to(self.device)
        self.mean = torch.tensor(OPENAI_CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(OPENAI_CLIP_STD, device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def quality(self, images: torch.Tensor) -> torch.Tensor:
        """The probability of "Good photo." against "Bad photo.", (N,): the official test pipeline's normalisation,
        then CLIPIQAFixed.forward; the model casts the image to its own dtype."""
        logits_per_image, _ = self.clip((images - self.mean) / self.std, self.tokens, pos_embedding=False)
        return logits_per_image.softmax(dim=-1)[:, 0].float()


def weight_files() -> dict:
    """The model files the scores depend on; pyiqa caches them under torch's hub folder."""
    hub = Path(torch.hub.get_dir())
    return {"arniqa_encoder": hub / "checkpoints" / "ARNIQA.pth",
            "arniqa_regressor": hub / "pyiqa" / "regressor_kadid10k.pth",
            "niqe_published": hub / "pyiqa" / "niqe_modelparameters.mat",
            "clip_rn50": hub / "clip" / "RN50.pt"}


class IqaModels:
    """The four baselines on one batch of same-size images. The scores need the clean references, the refitted
    NIQE model and ARNIQA's prototype, from the fit stage."""
    batch_size = 16  # one image's 96 versions, 16 at a time
    fit_batch_size = 1  # clean train images differ in size
    protocol = {
        "niqe": {"block": NIQE_BLOCK, "sharpness": NIQE_SHARPNESS, "luma": "rounded Y in [0, 255], float64",
                 "refit": "estimatemodelparam.m", "default": "modelparameters.mat",
                 "unscorable": f"no block without a NaN: {NIQE_UNSCORABLE:g}"},
        "arniqa": {"regressor": "kadid10k", "source": "miccunifi/ARNIQA test.py",
                   "crops": f"centre and four corners, {ARNIQA_CROP} x {ARNIQA_CROP}, zero-padded, at both sizes",
                   "half_size": "PIL bicubic to (W // 2, H // 2)",
                   "embedding": "per crop, both sizes each L2-normalised, 4096; the mean over the crops",
                   "precision": "encoder under autocast (fp16) on the GPU, fp32 on the CPU",
                   "prototype": "mean clean embedding; 1 - cosine"},
        "clipiqa": {"backbone": "RN50", "prompts": list(CLIP_PROMPTS), "positional_embedding": False,
                    "score": "softmax of logit_scale x cosine; probability of the first prompt",
                    "weights": "fp16 on the GPU (official build_model), fp32 on the CPU",
                    "source": "IceClear/CLIP-IQA v2-3.8, CLIPIQAFixed"},
        "packages": {name: metadata.version(name) for name in ("pyiqa", "openai-clip", "ftfy")},
    }

    def __init__(self, device, niqe_refit=None, prototype=None):
        self.device = torch.device(device)
        self.arniqa, self.clipiqa = Arniqa(device), ClipIqa(device)
        self.niqe_default = published_niqe()
        self.niqe_refit, self.prototype = niqe_refit, prototype

    @torch.inference_mode()
    def fit_features(self, arrays):
        """For the clean references: NIQE's block features and sharpness (None for an image without a whole block)
        and ARNIQA's embeddings."""
        features = sharpness = None
        if min(arrays[0].shape[:2]) >= NIQE_BLOCK:
            features, sharpness = niqe_block_features(niqe_luma(to_unit_tensor(arrays, self.device)))
        return features, sharpness, self.arniqa.embed(arrays)

    def _references(self):
        if self.niqe_refit is None or self.prototype is None:
            raise ValueError("the scores need the clean references: run the fit stage first")

    @torch.inference_mode()
    def niqe_rows(self, arrays) -> dict:
        """Both NIQE rows, and niqe_blocks: each version's blocks without a NaN. A version with none has no NIQE
        features; both rows give it NIQE_UNSCORABLE. With one, pyiqa's covariance is zero, and the score stays
        pyiqa's."""
        self._references()
        features = niqe_block_features(niqe_luma(to_unit_tensor(arrays, self.device)))[0]
        blocks = (~features.isnan().any(dim=2)).sum(dim=1)
        mu, cov = niqe_test_model(features)
        unscorable = blocks < NIQE_MIN_BLOCKS
        mu = torch.where(unscorable[:, None], 0.0, mu)  # finite stand-ins, so that pinv never sees a NaN
        cov = torch.where(unscorable[:, None, None], 0.0, cov)
        rows = {name: torch.where(unscorable, NIQE_UNSCORABLE, niqe_distance(*model, mu, cov))
                for name, model in (("niqe", self.niqe_refit), ("niqe_default", self.niqe_default))}
        return {**rows, "niqe_blocks": blocks}

    @torch.inference_mode()
    def arniqa_rows(self, arrays) -> dict:
        self._references()
        embedding = self.arniqa.embed(arrays)
        prototype = torch.as_tensor(self.prototype, dtype=embedding.dtype, device=self.device)
        return {"arniqa": -self.arniqa.quality(embedding),
                "arniqa_proto": 1.0 - F.cosine_similarity(embedding, prototype[None], dim=1)}

    @torch.inference_mode()
    def clipiqa_rows(self, arrays) -> dict:
        return {"clipiqa": -self.clipiqa.quality(to_unit_tensor(arrays, self.device))}

    def scores(self, arrays) -> dict:
        """The five rows of one batch of same-size images, float64 (N,) each, higher meaning more likely degraded;
        and niqe_blocks."""
        self._references()
        rows = {**self.niqe_rows(arrays), **self.arniqa_rows(arrays), **self.clipiqa_rows(arrays)}
        return {**{key: rows[key].double().cpu().numpy() for key in ROWS},
                "niqe_blocks": rows["niqe_blocks"].cpu().numpy()}


def load_iqa_models(device, niqe_refit=None, prototype=None) -> IqaModels:
    return IqaModels(device, niqe_refit=niqe_refit, prototype=prototype)
