"""Adapter around the official DisCoPatch code (Caetano et al., ICCV 2025) for COCO images."""
from __future__ import annotations

import importlib
import os
import sys
import types
import zlib
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

DEFAULT_ROOT = Path(os.environ.get("DISCOPATCH_ROOT", "/home/yuchen/YuchenZ/UE/DisCoPatch"))
IMAGE_SIDE = 256
PATCH = 64
TRAIN_ARGS = {  # README replication command for ImageNet-1K
    "batch_size": 67, "patches": 48, "hidden_dims": [128, 256, 512, 1024], "latent_dim": 1024,
    "lr": 8.5e-5, "gen_weight": 1e-3, "recon_weight": 1e-3, "kld_weight": 1e-4, "loss_type": "mse",
    "sample_and_save_frequency": 5, "no_wandb": True, "dataset": "coco", "num_samples": 16,
    "checkpoint": None, "discriminator_checkpoint": None,
}
_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIDE, IMAGE_SIDE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def import_discopatch(models_dir, root=DEFAULT_ROOT):
    """Import models.DisCoPatch without wandb and point its output folders at `models_dir`."""
    models_dir = Path(models_dir)
    (models_dir / "figures").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    if "wandb" not in sys.modules:
        stub = types.ModuleType("wandb")
        stub.log = stub.init = stub.finish = lambda *args, **kwargs: None
        sys.modules["wandb"] = stub
    source = str(Path(root) / "src" / "discopatch")
    if source not in sys.path:
        sys.path.insert(0, source)
    module = importlib.import_module("models.DisCoPatch")
    module.models_dir = str(models_dir)          # the repo's .env would otherwise win
    module.figures_dir = str(models_dir / "figures")
    return module


class CocoPatchDataset(Dataset):
    """Random 64x64 patches of 256x256-resized images, like the official ImageNet loader."""

    def __init__(self, paths, patches: int = 48):
        self.paths = list(paths)
        self.patches = patches

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as source:
            image = _TRANSFORM(source.convert("RGB"))
        corners = np.random.randint(0, IMAGE_SIDE - PATCH, size=(self.patches, 2))
        return torch.stack([image[:, x:x + PATCH, y:y + PATCH] for x, y in corners]), 0


def train_discopatch(paths, models_dir, *, epochs, num_workers, seed, overrides=None, root=DEFAULT_ROOT) -> Path:
    module = import_discopatch(models_dir, root)
    torch.manual_seed(seed)
    np.random.seed(seed)
    args = Namespace(**{**TRAIN_ARGS, **(overrides or {}), "n_epochs": epochs})
    loader = DataLoader(CocoPatchDataset(paths, patches=args.patches), batch_size=args.batch_size,
                        shuffle=True, pin_memory=True, num_workers=num_workers,
                        persistent_workers=num_workers > 0)
    model = module.DisCoPatch(input_shape=IMAGE_SIDE // 4, input_channels=3, args=args)
    model.train_model(loader, loader)
    return Path(models_dir) / "DisCoPatch" / "Discriminator_coco.pt"


def crop_positions(image_id: str, patches: int) -> np.ndarray:
    generator = np.random.default_rng(zlib.crc32(image_id.encode("utf-8")))
    return generator.integers(0, IMAGE_SIDE - PATCH, size=(patches, 2))


def _to_patchnorm(discriminator, patchnorm_class, patches):
    """Same replacement the official outlier_detection performs before scoring."""
    for block in discriminator.encoder:
        if isinstance(block, nn.Sequential):
            for index, layer in enumerate(block):
                if isinstance(layer, nn.BatchNorm2d):
                    replacement = patchnorm_class(layer.num_features, patches, eps=layer.eps, affine=True)
                    replacement.gamma = layer.weight
                    replacement.beta = layer.bias
                    block[index] = replacement


class DisCoPatchScorer:
    def __init__(self, checkpoint, models_dir, device, patches=64,
                 hidden_dims=TRAIN_ARGS["hidden_dims"], root=DEFAULT_ROOT, chunk_variants=16):
        module = import_discopatch(models_dir, root)
        discriminator = module.Discriminator(IMAGE_SIDE // 4, 3, list(hidden_dims),
                                             TRAIN_ARGS["lr"], TRAIN_ARGS["batch_size"])
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        discriminator.load_state_dict(state, strict=True)
        _to_patchnorm(discriminator, module.Patchnorm2D, patches)
        self.discriminator = discriminator.to(device).eval()
        self.device = torch.device(device)
        self.patches = patches
        self.chunk_variants = chunk_variants

    @torch.inference_mode()
    def score(self, arrays, image_id: str) -> np.ndarray:
        """Degradation score per array: 1 - mean patch output, patches normalised per image."""
        corners = crop_positions(image_id, self.patches)
        values = []
        for start in range(0, len(arrays), self.chunk_variants):
            patches = []
            for array in arrays[start:start + self.chunk_variants]:
                image = _TRANSFORM(Image.fromarray(array).convert("RGB"))
                patches.extend(image[:, x:x + PATCH, y:y + PATCH] for x, y in corners)
            output = self.discriminator(torch.stack(patches).to(self.device))
            values.append(1.0 - output.view(-1, self.patches).mean(dim=1).cpu().numpy())
        return np.concatenate(values)
