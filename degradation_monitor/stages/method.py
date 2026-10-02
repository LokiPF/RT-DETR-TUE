"""Our method's stages: the clean reference statistics, then those of every evaluation image under all 96 conditions.

Both store each channel's level and top-1% mean. The report compares them with the reference, so changing k (up to the
2,000 bank images) or the key needs no new pass; a larger bank, such as the 5,000-image ablation in docs/todo.md, needs
the statistics of more clean train images first.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from PIL import Image

from ..detector.model import IMAGE_SIZE, load_frozen_detector, prepare_image
from ..detector.taps import EarlyChannelTaps
from ..method.statistics import KEYS, channel_statistics
from ..runs import atomic_npz, progress
from .common import cap_gpu_memory, check_digests, clean_loader, pending_images, variant_stream


def early_taps(settings) -> EarlyChannelTaps:
    return EarlyChannelTaps(load_frozen_detector(settings.checkpoint, torch.device(settings.device)).backbone)


def batch_statistics(taps, batch) -> dict:
    """Each channel's level and top-1% mean in the four early maps of a prepared batch: KEYS -> float32 (N, C)."""
    out = {}
    for stage, maps in enumerate(taps(batch), start=1):
        for statistic, values in channel_statistics(maps).items():
            out[f"{statistic}_s{stage}"] = values
    return out


def _concatenated(parts: list) -> dict:
    return {key: np.concatenate([part[key] for part in parts]) for key in KEYS}


def image_statistics(taps, arrays, batch_size) -> dict:
    """The statistics of every condition of one image: KEYS -> float32 (96, C)."""
    parts = []
    for start in range(0, len(arrays), batch_size):
        batch = torch.stack([prepare_image(Image.fromarray(a), IMAGE_SIZE) for a in arrays[start:start + batch_size]])
        parts.append(batch_statistics(taps, batch))
    out = _concatenated(parts)
    if not all(np.isfinite(value).all() for value in out.values()):
        raise ValueError("channel statistics are not finite")
    return out


def method_reference(settings, manifest) -> None:
    """The statistics of the bank and z-statistics images: seeded, disjoint draws of clean COCO train images."""
    layout = settings.layout
    pending = {split: path for split, path in (("bank", layout.method_bank), ("zstats", layout.method_zstats))
               if not path.exists()}
    if not pending:
        return
    cap_gpu_memory(settings.device, settings.gpu_memory_gib)
    with early_taps(settings) as taps:
        for split, path in pending.items():
            loader = clean_loader(settings, settings.dataset.reference_split(split))
            atomic_npz(path, **_concatenated([batch_statistics(taps, batch) for batch in loader]))


def method_pass(settings, manifest) -> None:
    """The statistics of every evaluation image under all 96 conditions, resumable image by image."""
    layout = settings.layout
    pending = pending_images(settings, "method")
    if not pending:
        return
    cap_gpu_memory(settings.device, settings.gpu_memory_gib)
    started = time.time()
    with early_taps(settings) as taps:
        for done, (name, arrays) in enumerate(variant_stream(settings, pending), start=1):
            check_digests(layout, name, arrays)
            atomic_npz(layout.score_file("method", name), **image_statistics(taps, arrays, settings.batch_size))
            if done % 25 == 0:
                progress("method-pass", done, len(pending), started)
