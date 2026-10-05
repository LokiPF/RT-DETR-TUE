"""What every stage shares: the GPU memory cap, image loading, the stream of corrupted variants and the input checks."""
from __future__ import annotations

import multiprocessing
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .. import corruptions
from ..detector.model import IMAGE_SIZE, prepare_image
from ..runs import SCORE_KEYS, load_npz, valid_existing


def cap_gpu_memory(device, gib) -> None:
    """Make this process run out of memory itself before it squeezes other jobs on a shared GPU."""
    device = torch.device(device)
    if gib is None or device.type != "cuda":
        return
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, gib * 2**30 / total), device)


def open_rgb(path) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(source.convert("RGB"), dtype=np.uint8).copy()


def image_size(array) -> tuple[int, int]:
    """(width, height) of an RGB array."""
    return array.shape[1], array.shape[0]


class PreparedImages(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as source:
            return prepare_image(source.convert("RGB"), IMAGE_SIZE)


def clean_loader(settings, paths) -> DataLoader:
    return DataLoader(PreparedImages(paths), batch_size=settings.batch_size, num_workers=settings.workers,
                      pin_memory=torch.cuda.is_available())


class _RgbImages(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as source:
            return np.asarray(source.convert("RGB"), dtype=np.uint8).copy()


def rgb_batches(paths, size, workers):
    """Lists of `size` RGB arrays, in the order of `paths`."""
    return DataLoader(_RgbImages(paths), batch_size=size, num_workers=workers, collate_fn=list)


def report_peak_memory(label, device) -> None:
    """The process's peak GPU memory so far, for the sessions that share the card."""
    if torch.device(device).type == "cuda":
        print(f"[{label}] peak GPU memory {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB", flush=True)


def median_ms(function, inputs, warmup: int, device) -> float:
    """The median wall time of function(item) over inputs, in ms, after warmup calls on the first inputs; on a GPU,
    synchronised before and after each call. Both timing stages measure every model this way."""
    on_gpu = torch.cuda.is_available() and str(device).startswith("cuda")
    for item in inputs[:warmup]:
        function(item)
    values = []
    for item in inputs:
        if on_gpu:
            torch.cuda.synchronize()
        start = time.perf_counter()
        function(item)
        if on_gpu:
            torch.cuda.synchronize()
        values.append(1000.0 * (time.perf_counter() - start))
    return float(np.median(values))


def bounded(pool, function, items, in_flight):
    """Ordered results with at most `in_flight` tasks queued, so memory stays bounded."""
    iterator = iter(items)
    # range first: zip stops on range without pulling (and losing) an extra item
    queue = deque(pool.apply_async(function, (item,)) for _, item in zip(range(in_flight), iterator))
    while queue:
        result = queue.popleft().get()
        following = next(iterator, None)
        if following is not None:
            queue.append(pool.apply_async(function, (following,)))
        yield result


def variant_stream(settings, paths):
    """(file name, its 96 variants) for each image, in order, built by `settings.workers` processes."""
    if settings.workers == 0:
        for path in paths:
            yield corruptions.load_variants(path)
        return
    context = multiprocessing.get_context("spawn")
    with context.Pool(settings.workers) as pool:
        yield from bounded(pool, corruptions.load_variants, paths, 2 * settings.workers)


def check_digests(layout, name, arrays) -> None:
    """A later pass must see exactly the corruptions the detector pass saw."""
    stored = load_npz(layout.score_file("detector", name), SCORE_KEYS["detector"])["digests"]
    if list(stored) != [corruptions.digest(a) for a in arrays]:
        raise ValueError(f"corruptions differ from the detector pass for {name}")


def pending_images(settings, folder: str) -> list[Path]:
    """Evaluation images without a valid file in scores/<folder>; a later pass needs the detector pass's files."""
    layout = settings.layout
    pending = [p for p in settings.dataset.evaluation_images()
               if not valid_existing(layout.score_file(folder, p), SCORE_KEYS[folder])]
    if folder != "detector":
        absent = [p.name for p in pending if not layout.score_file("detector", p).exists()]
        if absent:
            raise ValueError(f"run the detector-pass stage first: {len(absent)} detector results are missing, "
                             f"e.g. {absent[0]}")
    return pending
