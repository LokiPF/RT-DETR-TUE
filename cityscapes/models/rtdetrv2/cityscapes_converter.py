"""Convert Cityscapes fine annotations into COCO-style detection JSON for RT-DETRv2 (the 8 instance classes).

Follows Detectron's tools/convert_cityscapes_to_coco.py, the conversion used across the domain-adaptive detection
literature (Cityscapes -> Foggy Cityscapes):
- boxes come from each instance's visible mask in *_gtFine_instanceIds.png, in the [x, y, w, h] pixel convention
  (w = x1 - x0 + 1);
- group regions ("cargroup", ...) carry no instance index (id < 1000) and are skipped;
- an instance is dropped when one of its outer contours has two points or fewer, as in Detectron;
- caravan and trailer are not among the 8 classes and are skipped; every kept box has iscrowd 0.
Category ids are 0-7 in CLASSES order, because RT-DETRv2 uses them directly as labels when
remap_mscoco_category is False.

    python dataset_tools/cityscapes_converter.py --root /path/to/cityscape --out /path/to/cityscape/annotations_coco
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

CLASSES = ("person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")
CITYSCAPES_LABEL_IDS = {24: "person", 25: "rider", 26: "car", 27: "truck", 28: "bus", 31: "train",
                        32: "motorcycle", 33: "bicycle"}  # cityscapesscripts/helpers/labels.py
SPLITS = ("train", "val")


def image_objects(instance_file: Path) -> tuple[tuple[int, int], list[dict]]:
    """((height, width), boxes) of one image: each kept instance as {category_id, bbox, area}."""
    ids = np.array(Image.open(instance_file))
    objects = []
    for instance in np.unique(ids):
        if instance < 1000 or instance // 1000 not in CITYSCAPES_LABEL_IDS:
            continue
        mask = (ids == instance).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours or min(len(c) for c in contours) <= 2:  # Detectron: min(len(flat polygon)) <= 4
            continue
        points = np.concatenate([c.reshape(-1, 2) for c in contours])
        x0, y0 = points.min(0)
        x1, y1 = points.max(0)
        objects.append({"category_id": CLASSES.index(CITYSCAPES_LABEL_IDS[instance // 1000]),
                        "bbox": [int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)],
                        "area": int(mask.sum())})
    return ids.shape, objects


def _convert_one(args):
    instance_file, image_name = args
    return image_name, image_objects(instance_file)


def convert(root: Path, split: str, workers: int) -> dict:
    instance_files = sorted((root / "gtFine_trainvaltest" / "gtFine" / split).glob("*/*_gtFine_instanceIds.png"))
    jobs = []
    for instance_file in instance_files:
        stem = instance_file.name.replace("_gtFine_instanceIds.png", "")
        image_name = f"{split}/{instance_file.parent.name}/{stem}_leftImg8bit.png"
        if not (root / "leftImg8bit" / image_name).exists():
            raise FileNotFoundError(f"no image for {instance_file}")
        jobs.append((instance_file, image_name))
    with Pool(workers) as pool:
        results = pool.map(_convert_one, jobs, chunksize=8)
    images, annotations = [], []
    for image_id, (image_name, ((height, width), objects)) in enumerate(results):
        images.append({"id": image_id, "file_name": image_name, "height": int(height), "width": int(width)})
        for obj in objects:
            annotations.append({"id": len(annotations), "image_id": image_id, "iscrowd": 0, **obj})
    return {"info": {"description": f"Cityscapes {split}, 8 instance classes, Detectron conversion rules"},
            "images": images, "annotations": annotations,
            "categories": [{"id": i, "name": name, "supercategory": name} for i, name in enumerate(CLASSES)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True, help="folder with leftImg8bit/ and gtFine_trainvaltest/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        data = convert(args.root, split, args.workers)
        path = args.out / f"cityscapes_{split}_8cls.json"
        path.write_text(json.dumps(data))
        counts = np.bincount([a["category_id"] for a in data["annotations"]], minlength=len(CLASSES))
        empty = len(data["images"]) - len({a["image_id"] for a in data["annotations"]})
        print(f"{split}: {len(data['images'])} images ({empty} without boxes), {len(data['annotations'])} boxes: "
              + ", ".join(f"{name} {n}" for name, n in zip(CLASSES, counts)) + f" -> {path}")


if __name__ == "__main__":
    main()
