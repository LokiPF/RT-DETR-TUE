"""Cityscapes (fine annotations, 8 instance classes) in the COCO-style form of datasets.coco.

The images sit in one folder per city, leftImg8bit/<split>/<city>/<name>_leftImg8bit.png. Their names are unique, so
result files and the ground truth are keyed by name, as for COCO. The annotation file is the val output of
cityscapes/models/rtdetrv2/cityscapes_converter.py: Detectron's conversion rules, category ids 0-7 in CLASSES order.
"""
from __future__ import annotations

from typing import ClassVar

from .coco import Coco

CLASSES = ("person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")


class Cityscapes(Coco):
    """Cityscapes train and val: city subfolders, 8 classes, and no screening history (one image set in the report)."""
    name: ClassVar[str] = "cityscapes"
    classes: ClassVar[int] = len(CLASSES)
    min_clean_ap: ClassVar[float] = 0.35  # the fine-tuned RT-DETR's clean val AP is 0.382
    recursive: ClassVar[bool] = True
    screened: ClassVar[bool] = False
