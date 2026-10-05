"""YOLO11m fine-tuned on Cityscapes: the YOLO labels and image links.

    python cityscapes/models/yolo11m.py prepare     # labels and image links, on the CPU

The labels come from the 8-class COCO-style files RT-DETR trained on (rtdetrv2/cityscapes_converter.py), and each
box's category_id is its class, unchanged (0-7, person first). Ultralytics' convert_coco is not used: it writes
category_id - 1, which turns person into -1, and Ultralytics then rejects every image holding a person as corrupt.
Ultralytics finds an image's label by swapping /images/ for /labels/ in the image's path, so the images are linked
into images/<split>/ beside labels/<split>/. See design.md beside this file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CITYSCAPES = Path("/home/yuchen/YuchenZ/Datasets/cityscape")
IMAGES = CITYSCAPES / "leftImg8bit"  # the annotation files' file_name is relative to this folder
DATASET = CITYSCAPES / "yolo_8cls"  # images/<split>/ (links) and labels/<split>/; yolo_8cls.yaml points here
DATA_YAML = Path(__file__).resolve().with_name("yolo_8cls.yaml")
SPLITS = ("train", "val")
CLASSES = ("person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")


def annotation_file(split: str) -> Path:
    return CITYSCAPES / "annotations_coco" / f"cityscapes_{split}_8cls.json"


def label_lines(boxes: list[dict], width: int, height: int) -> list[str]:
    """One line per box: its category id as the class, then its centre, width and height over the image's size."""
    lines = []
    for box in boxes:
        x, y, w, h = box["bbox"]
        lines.append(f"{box['category_id']} {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} "
                     f"{w / width:.6f} {h / height:.6f}")
    return lines


def prepare_split(annotations: Path, images: Path, out: Path, split: str) -> tuple[int, int]:
    """Link each image of one split into out/images/<split>/ and write its labels to out/labels/<split>/.

    An image without a box gets an empty label file, which Ultralytics reads as background. A second run replaces
    the links and the label files, and drops Ultralytics' label cache, which is keyed on file sizes and paths only
    and would hide a fix that keeps the sizes. Returns the numbers of images and boxes.
    """
    data = json.loads(Path(annotations).read_text())
    boxes = {}
    for box in data["annotations"]:
        boxes.setdefault(box["image_id"], []).append(box)
    image_dir, label_dir = out / "images" / split, out / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    (out / "labels" / f"{split}.cache").unlink(missing_ok=True)
    for info in data["images"]:
        source = images / info["file_name"]
        if not source.is_file():  # a dangling link would only be skipped, with a warning, by Ultralytics
            raise FileNotFoundError(f"no image at {source}: check the image folder")
        link = image_dir / source.name
        link.unlink(missing_ok=True)
        link.symlink_to(source)
        lines = label_lines(boxes.get(info["id"], []), info["width"], info["height"])
        (label_dir / f"{source.stem}.txt").write_text("".join(line + "\n" for line in lines))
    return len(data["images"]), len(data["annotations"])


def prepare(out: Path = DATASET) -> None:
    for split in SPLITS:
        images, boxes = prepare_split(annotation_file(split), IMAGES, out, split)
        print(f"[prepare] {split}: {images} images, {boxes} boxes -> {out}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="YOLO11m fine-tuned on Cityscapes (design.md beside this file).")
    parser.add_argument("command", choices=("prepare",))
    parser.parse_args(argv)
    prepare()
    return 0


if __name__ == "__main__":
    sys.exit(main())
