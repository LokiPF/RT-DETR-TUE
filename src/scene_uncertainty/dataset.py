from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from torch.utils.data import DataLoader

from src.data.dataloader import batch_image_collate_fn
from src.data.dataset import CocoDetection
from src.data.transforms import Compose, ConvertBoxes, ConvertPILImage, Resize, SanitizeBoundingBoxes

from .blur import FixedGaussianBlur


def _per_annotation_tensors(sample):
    """Every tensor returned here is filtered by `SanitizeBoundingBoxes`' box mask.

    Its default heuristic only finds `labels`, which would leave the other
    per-annotation tensors misaligned with `boxes` whenever a box is dropped. They must
    all be listed so the target dict has no mix of aligned and stale keys. Returns the
    tensors themselves, not copies: the sanitizer matches them by object identity.
    Defined at module level rather than as a lambda so the loader stays picklable for
    worker processes.

    This getter is the reason `make_coco_loader` may ask `CocoDetection` for
    `annotation_ids` at all: the key is opt-in precisely because a pipeline that
    sanitizes with the default getter would silently desync it. Adding a per-annotation
    tensor to the target dict without adding it here reintroduces that desync.
    """
    target = sample[1]
    return target["labels"], target["annotation_ids"], target["area"], target["iscrowd"]


def make_coco_loader(
    image_root: str | Path,
    annotation_file: str | Path,
    image_ids: Sequence[int],
    blur_radius: float,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    transforms = Compose(
        [
            Resize(size=[640, 640]),
            FixedGaussianBlur(blur_radius),
            SanitizeBoundingBoxes(min_size=1, labels_getter=_per_annotation_tensors),
            ConvertPILImage(dtype="float32", scale=True),
            ConvertBoxes(fmt="cxcywh", normalize=True),
        ]
    )
    dataset = CocoDetection(
        img_folder=str(image_root),
        ann_file=str(annotation_file),
        transforms=transforms,
        return_masks=False,
        remap_mscoco_category=True,
        # The extractor attributes every matched query to a COCO annotation, so this loader is
        # the one caller that needs the ids -- and the transform list above keeps them
        # aligned. No detector config asks for them.
        return_annotation_ids=True,
    )
    available = set(int(image_id) for image_id in dataset.ids)
    requested = [int(image_id) for image_id in image_ids]
    missing = sorted(set(requested) - available)
    if missing:
        raise KeyError(f"Requested COCO image IDs are missing: {missing[:10]}")
    dataset.ids = requested
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=batch_image_collate_fn,
        drop_last=False,
    )
