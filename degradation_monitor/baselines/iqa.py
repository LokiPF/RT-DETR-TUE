"""Four detector-free image-quality baselines, each scored so that higher means more likely degraded.

- NIQE (Mittal, Soundararajan & Bovik, IEEE SPL 2013): the distance between a multivariate Gaussian fitted to an
  image's block features and one fitted to pristine images. The main row's pristine model is refitted on clean train
  images with the authors' recipe (their estimatemodelparam.m: 96 x 96 blocks at two scales, keeping the blocks
  sharper than 0.75 times the image's sharpest); the sensitivity row uses the published model. Both rows share the
  test image's features, computed as pyiqa 0.1.16's NIQE does: the rounded luma, in float64.

Every model reads the uint8 RGB image at its own size.
"""
from __future__ import annotations

import numpy as np
import torch

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
