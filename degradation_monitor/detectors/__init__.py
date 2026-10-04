"""Frozen COCO detectors beyond RT-DETRv2-R18, each behind one adapter that gives every monitor its inputs.

An adapter is called on a list of same-size RGB uint8 arrays and returns Outputs; with heads=False it runs the
backbone only, which is all the clean fits need. Its class attributes: name; batch_size, the versions of one image per
forward pass; fit_batch_size, the clean train images per forward pass (1 when the input size follows the image's);
detr, whether it also gives query logits and decoder queries; and protocol, the taps, input size and thresholds its
results depend on, which its run folder's manifest records.
"""
from __future__ import annotations

from .base import COCO_CATEGORY_IDS, LABEL_OF_CATEGORY, LEVELS, TOP_K, Outputs, Region, padded

DETECTORS = ("yolo11m", "faster_rcnn_r50_fpn_v2", "rfdetr_m")


def adapter_class(name: str):
    """The adapter class of one detector; importing it loads no detector library."""
    if name == "yolo11m":
        from .yolo import Yolo11m
        return Yolo11m
    if name == "faster_rcnn_r50_fpn_v2":
        from .faster_rcnn import FasterRcnn
        return FasterRcnn
    if name == "rfdetr_m":
        from .rfdetr import RfDetrM
        return RfDetrM
    raise ValueError(f"unknown detector {name!r}; choose from {', '.join(DETECTORS)}")


def load_adapter(name: str, weights, device):
    """The adapter of one detector, with its weights loaded and frozen on the device."""
    return adapter_class(name)(weights, device)


__all__ = ["COCO_CATEGORY_IDS", "DETECTORS", "LABEL_OF_CATEGORY", "LEVELS", "TOP_K", "Outputs", "Region",
           "adapter_class", "load_adapter", "padded"]
