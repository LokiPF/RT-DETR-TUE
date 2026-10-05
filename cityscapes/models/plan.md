# Two Cityscapes Detectors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze and record the Cityscapes fine-tune of RT-DETRv2-R18, and fine-tune, freeze and record YOLO11m on Cityscapes, all in `cityscapes/models/`.

**Architecture:**
- **RT-DETR needs no training.** Its last checkpoint is copied out of its training folder, and its three training files are copied into git. Its AP per class comes from the last epoch's stored evaluation.
- **YOLO11m gets one script,** `cityscapes/models/yolo11m.py`, with three commands:
  - `prepare` writes YOLO labels and image links from the 8-class COCO-style files;
  - `train` runs Ultralytics' standard fine-tune;
  - `val` measures the frozen weights.
- **The record** of both detectors is `cityscapes/models/README.md`.

**Tech Stack:** Python 3.11 in the `UE` conda environment (`/home/yuchen/miniconda3/envs/UE/bin/python`), Ultralytics 8.3.235, PyTorch, Pillow, PyYAML, pytest.

**Spec:** `cityscapes/models/design.md`.

## Global Constraints

- **Classes:** ids 0–7, in this order: person, rider, car, truck, bus, train, motorcycle, bicycle. A YOLO class is the annotation's `category_id`, unchanged.
- **YOLO11m recipe:**
  - from `/home/yuchen/YuchenZ/lab/Detector_test/yolo11m.pt`, with `imgsz=640`, `epochs=100` and `batch=16`;
  - every other Ultralytics argument keeps its default: no `optimizer`, `lr0`, `seed`, `cache`, `workers`, `fraction` or other recipe argument;
  - `device`, `project` and `name` only say where to run and where to write.
- **Weights:** keep the last epoch's (`last.pt`, `last.pth`), never `best`.
- **No AP target.** Stop rule: if YOLO's mAP50-95 on val comes out below 0.15, stop and report, because that points to a label or class-order bug.
- **Light bookkeeping:** no hashes and no provenance records.
- **Git:**
  - Work in the main checkout `/home/yuchen/YuchenZ/UE/philip_sa` on `fingerprint_bank`. Everything in git goes under `cityscapes/models/`, plus `tests/test_cityscapes_yolo.py`.
  - Commit only the paths a task names: `git add <paths> && git commit -m "…" -- <paths>`. Never `git add -A` or `git add .`, and never push.
  - The checkout also holds the paper session's files (`IV_2027_Yuchen/`, `docs/related-work-citation-audit-2026-10-05.md`). They must never enter a commit, and `IV_2027_Yuchen/` is never touched.
  - Every commit message ends with these two lines:

    ```
    Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
    Claude-Session: https://claude.ai/code/session_0167vRw3NLCumE8FSGAnF1Tc
    ```
- **Tests run on the CPU only:** `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`. At the start, 194 tests are collected.
- **The GPU is shared** with the sessions "explore" and "ue-implement".
  - GPU steps (Task 3, Steps 6–8, and Task 4) are done by the controller, the session running this plan, not by a subagent.
  - The controller asks both sessions via SendMessage before each GPU run, and sends "done" after each.
  - The script caps its own GPU memory at 12 GiB.
- **RT-DETRv2-UE:** only `pretrained_weights/` gains a file. Nothing else in that repository changes.

## Review Focus

1. **Class ids shifted by one,** as Ultralytics' `convert_coco` would shift them: Ultralytics would reject every image holding a person as corrupt. Tests (Task 2): `test_a_box_becomes_its_normalised_line_and_person_stays_class_0` and `test_ultralytics_reads_the_labels_with_person_as_class_0`.
2. **A linked image whose label Ultralytics can't find:** Ultralytics derives the label path from the link's path, swapping `/images/` for `/labels/`, so a missed label would train its image as background. Test (Task 2): `test_each_link_points_at_its_image_and_ultralytics_finds_its_label`.
3. **Running `prepare` again after a fix:**
   - The links and labels must be replaced, not crash on an existing link.
   - Ultralytics' label cache must go too. It is keyed only on file sizes and paths, so a fix that keeps the sizes, such as swapped class ids, would otherwise be ignored.
   - Test (Task 2): `test_a_second_run_replaces_the_links_and_drops_ultralytics_label_cache`.
4. **A wrong image root:** the dangling links it would make are dropped by Ultralytics with a warning only. `prepare` must refuse to make them. Test (Task 2): `test_a_missing_image_is_refused`.
5. **A recipe argument creeping into the training call,** such as `lr0` or `cache`: it would no longer be Ultralytics' standard fine-tune. Test (Task 3): `test_train_passes_the_standard_recipe_and_nothing_else`.

---

### Task 1: Freeze and record RT-DETRv2-R18

Training ran from 16:03 to 17:50 on 4 October and wrote `/home/yuchen/YuchenZ/UE/RT-DETRv2-UE/output/rtdetrv2_r18vd_cityscapes/`. Its last epoch gives AP 0.382 and AP50 0.590 on val, from the EMA weights.

**Files:**
- Create, outside git: `/home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_cityscapes_72e.pth`, a copy of `last.pth`
- Create: `cityscapes/models/rtdetrv2/cityscapes_detection.yml`, `cityscapes/models/rtdetrv2/rtdetrv2_r18vd_cityscapes.yml`, `cityscapes/models/rtdetrv2/cityscapes_converter.py`, all copies
- Create: `cityscapes/models/README.md`

**Interfaces:**
- Produces:
  - the RT-DETR weights path above, which plan 2's config uses;
  - `README.md`, which Task 4 extends with YOLO11m.

- [ ] **Step 1: Copy the weights out of the training folder**

```bash
cp /home/yuchen/YuchenZ/UE/RT-DETRv2-UE/output/rtdetrv2_r18vd_cityscapes/last.pth \
      /home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_cityscapes_72e.pth
ls -la /home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/
```
Expected: `rtdetrv2_r18vd_cityscapes_72e.pth`, 322,546,541 bytes, next to the COCO checkpoint.

- [ ] **Step 2: Copy RT-DETR's three training files into git's folder**

```bash
cd /home/yuchen/YuchenZ/UE/philip_sa
R=/home/yuchen/YuchenZ/UE/RT-DETRv2-UE
mkdir -p cityscapes/models/rtdetrv2
cp $R/configs/dataset/cityscapes_detection.yml $R/configs/rtdetrv2/rtdetrv2_r18vd_cityscapes.yml \
   $R/dataset_tools/cityscapes_converter.py cityscapes/models/rtdetrv2/
ls cityscapes/models/rtdetrv2/
```
Expected: the three files.

- [ ] **Step 3: Read the AP per class from the last epoch's evaluation**

```bash
CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python - <<'EOF'
import numpy as np, torch
e = torch.load("/home/yuchen/YuchenZ/UE/RT-DETRv2-UE/output/rtdetrv2_r18vd_cityscapes/eval/latest.pth", weights_only=False)
p = np.asarray(e["precision"])  # (IoU 10, recall 101, classes 8, area 4, max detections 3)
def ap(x): x = x[x > -1]; return x.mean()
print(e["date"], f"all AP {ap(p[:, :, :, 0, 2]):.3f} AP50 {ap(p[0, :, :, 0, 2]):.3f}")
for k, name in enumerate(("person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")):
    print(f"{name:<11} AP {ap(p[:, :, k, 0, 2]):.3f}  AP50 {ap(p[0, :, k, 0, 2]):.3f}")
EOF
```
Expected, as read on 5 October:

```
2026-10-04 17:50 all AP 0.382 AP50 0.590
person      AP 0.344  AP50 0.594
rider       AP 0.355  AP50 0.598
car         AP 0.565  AP50 0.785
truck       AP 0.349  AP50 0.482
bus         AP 0.624  AP50 0.752
train       AP 0.292  AP50 0.522
motorcycle  AP 0.246  AP50 0.456
bicycle     AP 0.283  AP50 0.530
```

- [ ] **Step 4: Write `cityscapes/models/README.md`**

If Step 3 printed other numbers, use them in the table below.

````markdown
# Two detectors trained on Cityscapes

Both detectors find the 8 Cityscapes instance classes, with ids 0–7 in this order: person, rider, car, truck, bus, train, motorcycle, bicycle.
- **Training data:** the 2,975 Cityscapes train images. The fine annotations were converted by `rtdetrv2/cityscapes_converter.py` with Detectron's rules, giving 50,347 boxes in train and 9,792 in val.
- **Weights:** both keep their last epoch's weights. Val (500 images) was evaluated during training but selected nothing.
- **How they were made:** `design.md` and `plan.md`.

| Detector | Weights | AP on val | AP50 |
|---|---|---|---|
| RT-DETRv2-R18 | `/home/yuchen/YuchenZ/UE/RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_cityscapes_72e.pth` | 0.382 | 0.590 |

## RT-DETRv2-R18

- **Recipe:** RT-DETRv2-UE's standard fine-tune, from the COCO checkpoint `rtdetrv2_r18vd_120e_coco_rerun_48.1.pth`, with the class heads re-initialised (`-t`).
  - 72 epochs at a total batch of 16 on one GPU.
  - AdamW at 1e-4, with the backbone at 1e-5; EMA.
  - Full precision: the run had no `--use-amp` (its log shows `'use_amp': False`), although the config's header comment says AMP.
  - Multi-scale training at 480–800, evaluation at 640 × 640.
  - Trained on 4 October 2026, in 1 h 48 min.
- **Files:** `rtdetrv2/` holds the three files the run used, which RT-DETRv2-UE doesn't track. To rerun it, copy them back into RT-DETRv2-UE (bd4cc46) from this folder and run:

  ```bash
  R=/home/yuchen/YuchenZ/UE/RT-DETRv2-UE
  cp rtdetrv2/cityscapes_detection.yml $R/configs/dataset/
  cp rtdetrv2/rtdetrv2_r18vd_cityscapes.yml $R/configs/rtdetrv2/
  cp rtdetrv2/cityscapes_converter.py $R/dataset_tools/
  cd $R
  python dataset_tools/cityscapes_converter.py --root /home/yuchen/YuchenZ/Datasets/cityscape \
      --out /home/yuchen/YuchenZ/Datasets/cityscape/annotations_coco
  python tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_cityscapes.yml \
      -t pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth --seed 0
  ```
- **AP per class on val:** from the last epoch's evaluation of the EMA weights (`output/rtdetrv2_r18vd_cityscapes/eval/latest.pth`; faster-coco-eval's COCO evaluation, 100 detections).

| Class | AP | AP50 |
|---|---|---|
| person | 0.344 | 0.594 |
| rider | 0.355 | 0.598 |
| car | 0.565 | 0.785 |
| truck | 0.349 | 0.482 |
| bus | 0.624 | 0.752 |
| train | 0.292 | 0.522 |
| motorcycle | 0.246 | 0.456 |
| bicycle | 0.283 | 0.530 |

## Notes

- **Each toolkit evaluates its own detector here.** RT-DETR's numbers come from faster-coco-eval's COCO evaluation with 100 detections, and YOLO's from Ultralytics' own mAP with 300. The check stage of the Cityscapes-C evaluation (`../evaluation/`) measures both detectors through our adapters.
- **"640" is not the same input size for both detectors.** YOLO letterboxes 2048 × 1024 to 640 × 320, while RT-DETR resizes it to 640 × 640. YOLO therefore sees half the pixels, and every object is half as tall.
````

- [ ] **Step 5: Commit**

```bash
cd /home/yuchen/YuchenZ/UE/philip_sa
git add cityscapes/models/README.md cityscapes/models/rtdetrv2
git commit -m "docs: the RT-DETRv2-R18 Cityscapes fine-tune, frozen, with its training files and AP per class

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0167vRw3NLCumE8FSGAnF1Tc" -- cityscapes/models/README.md cityscapes/models/rtdetrv2
git show --stat --format= HEAD
```
Expected: 4 files, all under `cityscapes/models/`.

---

### Task 2: YOLO labels and image links

**Files:**
- Create: `cityscapes/models/yolo11m.py`
- Create: `cityscapes/models/yolo_8cls.yaml`
- Test: `tests/test_cityscapes_yolo.py`

**Interfaces:**
- Produces:
  - in `cityscapes/models/yolo11m.py`: `CITYSCAPES`, `IMAGES`, `DATASET`, `DATA_YAML` (each a `Path`), `SPLITS` and `CLASSES` (tuples of `str`);
  - `annotation_file(split: str) -> Path`;
  - `label_lines(boxes: list[dict], width: int, height: int) -> list[str]`;
  - `prepare_split(annotations: Path, images: Path, out: Path, split: str) -> tuple[int, int]`, which returns the number of images and the number of boxes;
  - `prepare(out: Path = DATASET) -> None`;
  - `main(argv=None) -> int`;
  - outside git, on the real data: `DATASET`'s folders `images/{train,val}/` and `labels/{train,val}/`, which Task 3 trains on.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cityscapes_yolo.py`:

```python
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image
from ultralytics.data.utils import img2label_paths, verify_image_label

from cityscapes.models import yolo11m

CITIES = ("aachen", "bochum")


def _fake_annotations(root: Path) -> Path:
    """Two 20 x 10 val images in city folders: aachen's holds a person and a car, bochum's no box."""
    for city in CITIES:
        folder = root / "leftImg8bit" / "val" / city
        folder.mkdir(parents=True)
        Image.new("RGB", (20, 10)).save(folder / f"{city}_000000_000019_leftImg8bit.png")
    path = root / "val.json"
    path.write_text(json.dumps({
        "images": [{"id": i, "file_name": f"val/{city}/{city}_000000_000019_leftImg8bit.png", "width": 20,
                    "height": 10} for i, city in enumerate(CITIES)],
        "annotations": [{"id": 0, "image_id": 0, "category_id": 0, "bbox": [2, 1, 4, 6], "area": 24, "iscrowd": 0},
                        {"id": 1, "image_id": 0, "category_id": 2, "bbox": [10, 5, 10, 5], "area": 50, "iscrowd": 0}],
        "categories": [{"id": i, "name": name} for i, name in enumerate(yolo11m.CLASSES)]}))
    return path


def _prepared(root: Path) -> Path:
    yolo11m.prepare_split(_fake_annotations(root), root / "leftImg8bit", root / "out", "val")
    return root / "out"


def _ultralytics_reads(out: Path, city: str):
    """What Ultralytics' dataset check makes of one image and its label file: (labels, missing, found, empty,
    corrupt, message)."""
    name = f"{city}_000000_000019_leftImg8bit"
    image, label = out / "images" / "val" / f"{name}.png", out / "labels" / "val" / f"{name}.txt"
    _, labels, _, _, _, missing, found, empty, corrupt, message = verify_image_label(
        (str(image), str(label), "", False, len(yolo11m.CLASSES), 0, 0, False))
    return labels, missing, found, empty, corrupt, message


def test_a_box_becomes_its_normalised_line_and_person_stays_class_0(tmp_path):
    counts = yolo11m.prepare_split(_fake_annotations(tmp_path), tmp_path / "leftImg8bit", tmp_path / "out", "val")
    assert counts == (2, 2)
    label = tmp_path / "out" / "labels" / "val" / "aachen_000000_000019_leftImg8bit.txt"
    assert label.read_text().splitlines() == ["0 0.200000 0.400000 0.200000 0.600000",
                                              "2 0.750000 0.750000 0.500000 0.500000"]


def test_ultralytics_reads_the_labels_with_person_as_class_0(tmp_path):
    labels, missing, found, empty, corrupt, message = _ultralytics_reads(_prepared(tmp_path), "aachen")
    assert (missing, found, empty, corrupt) == (0, 1, 0, 0), message
    assert labels[:, 0].tolist() == [0.0, 2.0]


def test_an_image_without_a_box_gets_an_empty_label_that_ultralytics_reads_as_background(tmp_path):
    out = _prepared(tmp_path)
    assert (out / "labels" / "val" / "bochum_000000_000019_leftImg8bit.txt").read_text() == ""
    _, missing, found, empty, corrupt, message = _ultralytics_reads(out, "bochum")
    assert (missing, found, empty, corrupt) == (0, 1, 1, 0), message


def test_each_link_points_at_its_image_and_ultralytics_finds_its_label(tmp_path):
    links = sorted((_prepared(tmp_path) / "images" / "val").iterdir())
    assert all(link.is_symlink() for link in links)
    assert [link.resolve() for link in links] == [
        (tmp_path / "leftImg8bit" / "val" / city / f"{city}_000000_000019_leftImg8bit.png").resolve()
        for city in CITIES]
    assert img2label_paths([str(link) for link in links]) == [
        str(tmp_path / "out" / "labels" / "val" / f"{link.stem}.txt") for link in links]


def test_a_second_run_replaces_the_links_and_drops_ultralytics_label_cache(tmp_path):
    out = _prepared(tmp_path)
    cache = out / "labels" / "val.cache"  # Ultralytics keys it on file sizes and paths only
    cache.write_text("stale")
    annotations = tmp_path / "val.json"
    assert yolo11m.prepare_split(annotations, tmp_path / "leftImg8bit", out, "val") == (2, 2)
    assert not cache.exists()
    assert len(list((out / "images" / "val").iterdir())) == 2


def test_a_missing_image_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="no image at"):
        yolo11m.prepare_split(_fake_annotations(tmp_path), tmp_path / "elsewhere", tmp_path / "out", "val")


def test_the_data_file_names_the_8_classes_in_order_and_points_at_the_dataset():
    data = yaml.safe_load(yolo11m.DATA_YAML.read_text())
    assert data["names"] == dict(enumerate(yolo11m.CLASSES))
    assert Path(data["path"]) == yolo11m.DATASET
    assert (data["train"], data["val"]) == ("images/train", "images/val")
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd /home/yuchen/YuchenZ/UE/philip_sa && CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/test_cityscapes_yolo.py`
Expected: collection fails with `ImportError: cannot import name 'yolo11m' from 'cityscapes.models'` or `ModuleNotFoundError`.

- [ ] **Step 3: Write `cityscapes/models/yolo11m.py` and `cityscapes/models/yolo_8cls.yaml`**

`cityscapes/models/yolo11m.py`:

```python
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
```

`cityscapes/models/yolo_8cls.yaml`:

```yaml
# Ultralytics' data file for YOLO11m on Cityscapes; `python cityscapes/models/yolo11m.py prepare` fills the folder.
path: /home/yuchen/YuchenZ/Datasets/cityscape/yolo_8cls
train: images/train
val: images/val
names:
  0: person
  1: rider
  2: car
  3: truck
  4: bus
  5: train
  6: motorcycle
  7: bicycle
```

- [ ] **Step 4: Run the tests to see them pass, then the whole suite**

Run: `cd /home/yuchen/YuchenZ/UE/philip_sa && CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/test_cityscapes_yolo.py`
Expected: `7 passed`.

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: no failures, with 201 tests collected (194 + 7).

- [ ] **Step 5: Prepare the real data, on the CPU**

Run: `cd /home/yuchen/YuchenZ/UE/philip_sa && /home/yuchen/miniconda3/envs/UE/bin/python cityscapes/models/yolo11m.py prepare`
Expected:

```
[prepare] train: 2975 images, 50347 boxes -> /home/yuchen/YuchenZ/Datasets/cityscape/yolo_8cls
[prepare] val: 500 images, 9792 boxes -> /home/yuchen/YuchenZ/Datasets/cityscape/yolo_8cls
```

Then check the folder: `D=/home/yuchen/YuchenZ/Datasets/cityscape/yolo_8cls; for s in train val; do echo $s $(ls $D/images/$s | wc -l) $(ls $D/labels/$s | wc -l) $(find $D/labels/$s -empty | wc -l); done`
Expected: `train 2975 2975 10` and `val 500 500 8`, giving images, label files and empty label files.

- [ ] **Step 6: Commit**

```bash
cd /home/yuchen/YuchenZ/UE/philip_sa
git add cityscapes/models/yolo11m.py cityscapes/models/yolo_8cls.yaml tests/test_cityscapes_yolo.py
git commit -m "feat: YOLO labels and image links for Cityscapes, with the class ids kept as they are

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0167vRw3NLCumE8FSGAnF1Tc" -- cityscapes/models/yolo11m.py cityscapes/models/yolo_8cls.yaml tests/test_cityscapes_yolo.py
```

---

### Task 3: The training and val commands, and a smoke run

**Files:**
- Modify: `cityscapes/models/yolo11m.py`. It gains `os`, the training constants, `cap_gpu_memory`, `train` and `val`, a new module docstring and a new `main`.
- Test: `tests/test_cityscapes_yolo.py`

**Interfaces:**
- Consumes: `DATA_YAML` and `prepare` (Task 2); the prepared data folder (Task 2, Step 5).
- Produces:
  - `COCO_WEIGHTS`, `FROZEN_WEIGHTS` and `RUNS` (each a `Path`);
  - `IMGSZ = 640`, `EPOCHS = 100`, `BATCH = 16`, `GPU_MEMORY_GIB = 12.0`;
  - `cap_gpu_memory(gib: float) -> None`;
  - `train(smoke: bool = False) -> None`, which writes `RUNS/train/` (or `RUNS/smoke/`);
  - `val() -> tuple[float, float]`, returning mAP50-95 and mAP50;
  - the commands `prepare`, `train [--smoke]` and `val`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cityscapes_yolo.py`, and add `from types import SimpleNamespace` to its imports:

```python
class FakeYolo:
    """Records what the script asks of Ultralytics' YOLO, without loading a model."""
    calls = []

    def __init__(self, weights):
        FakeYolo.calls.append(("load", weights))

    def train(self, **arguments):
        FakeYolo.calls.append(("train", arguments))

    def val(self, **arguments):
        FakeYolo.calls.append(("val", arguments))
        return SimpleNamespace(box=SimpleNamespace(map=0.3, map50=0.5))


@pytest.fixture
def fake_yolo(tmp_path, monkeypatch):
    FakeYolo.calls = []
    monkeypatch.setattr("ultralytics.YOLO", FakeYolo)
    monkeypatch.setattr(yolo11m, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(yolo11m, "cap_gpu_memory", lambda gib: FakeYolo.calls.append(("cap", gib)))
    monkeypatch.chdir(tmp_path)  # train() moves into RUNS; this puts the working folder back afterwards
    return FakeYolo.calls


def test_train_passes_the_standard_recipe_and_nothing_else(fake_yolo, tmp_path):
    yolo11m.train()
    yolo11m.train(smoke=True)
    where = {"data": str(yolo11m.DATA_YAML), "imgsz": 640, "batch": 16, "device": 0, "project": str(tmp_path / "runs")}
    load = ("load", str(yolo11m.COCO_WEIGHTS))
    assert fake_yolo == [("cap", 12.0), load, ("train", {**where, "epochs": 100, "name": "train"}),
                         ("cap", 12.0), load, ("train", {**where, "epochs": 1, "name": "smoke"})]
    assert Path.cwd() == (tmp_path / "runs").resolve()  # the AMP check downloads yolo11n.pt here, not into the repo


def test_val_measures_the_frozen_weights_at_640(fake_yolo, tmp_path):
    assert yolo11m.val() == (0.3, 0.5)
    assert fake_yolo == [("cap", 12.0), ("load", str(yolo11m.FROZEN_WEIGHTS)),
                         ("val", {"data": str(yolo11m.DATA_YAML), "imgsz": 640, "batch": 16, "device": 0,
                                  "project": str(tmp_path / "runs"), "name": "val"})]
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd /home/yuchen/YuchenZ/UE/philip_sa && CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/test_cityscapes_yolo.py -k "train or val_measures"`
Expected: both ERROR at setup, with `AttributeError: <module 'cityscapes.models.yolo11m' ...> has no attribute 'RUNS'`, because the fixture patches names that don't exist yet.

- [ ] **Step 3: Implement**

In `cityscapes/models/yolo11m.py`, replace the module docstring with:

```python
"""YOLO11m fine-tuned on Cityscapes: the YOLO labels and image links, the training run and its clean val AP.

    python cityscapes/models/yolo11m.py prepare          # labels and image links, on the CPU
    python cityscapes/models/yolo11m.py train [--smoke]  # Ultralytics' standard fine-tune (--smoke: one epoch)
    python cityscapes/models/yolo11m.py val              # the frozen last.pt on the 500 val images

The labels come from the 8-class COCO-style files RT-DETR trained on (rtdetrv2/cityscapes_converter.py), and each
box's category_id is its class, unchanged (0-7, person first). Ultralytics' convert_coco is not used: it writes
category_id - 1, which turns person into -1, and Ultralytics then rejects every image holding a person as corrupt.
Ultralytics finds an image's label by swapping /images/ for /labels/ in the image's path, so the images are linked
into images/<split>/ beside labels/<split>/. Training starts from the COCO yolo11m.pt at imgsz 640, for 100 epochs
at batch 16, and leaves every other argument at Ultralytics' default. See design.md beside this file.
"""
```

Add `import os` to the imports (after `import json`). Add these constants after `CLASSES`:

```python
COCO_WEIGHTS = Path("/home/yuchen/YuchenZ/lab/Detector_test/yolo11m.pt")  # the COCO-C detector (Ultralytics 8.3.235)
FROZEN_WEIGHTS = COCO_WEIGHTS.with_name("yolo11m_cityscapes_100e.pt")  # the full run's last.pt, copied (plan, Task 4)
RUNS = Path(__file__).resolve().parents[2] / "runs" / "cityscapes-yolo11m"  # git-ignored
IMGSZ, EPOCHS, BATCH = 640, 100, 16  # Ultralytics' standard fine-tune; every other argument keeps its default
GPU_MEMORY_GIB = 12.0  # the card is shared
```

Add these functions after `prepare`:

```python
def cap_gpu_memory(gib: float) -> None:
    """Make this process run out of memory itself before it squeezes other jobs on the shared card."""
    import torch

    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, gib * 2**30 / total), 0)


def train(smoke: bool = False) -> None:
    """Ultralytics' standard fine-tune from the COCO weights into RUNS/train, or one epoch into RUNS/smoke.

    Ultralytics adds a number to the folder's name (train2, ...) when it already exists, and says where it saved.
    """
    import torch
    from ultralytics import YOLO

    RUNS.mkdir(parents=True, exist_ok=True)
    os.chdir(RUNS)  # Ultralytics' AMP check downloads yolo11n.pt into the working folder
    cap_gpu_memory(GPU_MEMORY_GIB)
    YOLO(str(COCO_WEIGHTS)).train(data=str(DATA_YAML), imgsz=IMGSZ, epochs=1 if smoke else EPOCHS, batch=BATCH,
                                  device=0, project=str(RUNS), name="smoke" if smoke else "train")
    if torch.cuda.is_available():
        print(f"[train] peak GPU memory {torch.cuda.max_memory_reserved(0) / 2**30:.2f} GiB reserved", flush=True)


def val() -> tuple[float, float]:
    """Ultralytics' val of the frozen last.pt on the 500 val images at 640; Ultralytics prints the table per class."""
    from ultralytics import YOLO

    cap_gpu_memory(GPU_MEMORY_GIB)
    metrics = YOLO(str(FROZEN_WEIGHTS)).val(data=str(DATA_YAML), imgsz=IMGSZ, batch=BATCH, device=0,
                                            project=str(RUNS), name="val")
    print(f"[val] {FROZEN_WEIGHTS.name}: mAP50-95 {metrics.box.map:.4f}, mAP50 {metrics.box.map50:.4f}", flush=True)
    return metrics.box.map, metrics.box.map50
```

Replace `main` with:

```python
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="YOLO11m fine-tuned on Cityscapes (design.md beside this file).")
    parser.add_argument("command", choices=("prepare", "train", "val"))
    parser.add_argument("--smoke", action="store_true", help="train: one epoch, into runs/cityscapes-yolo11m/smoke")
    args = parser.parse_args(argv)
    if args.smoke and args.command != "train":
        parser.error("--smoke applies to train only")
    if args.command == "prepare":
        prepare()
    elif args.command == "train":
        train(smoke=args.smoke)
    else:
        val()
    return 0
```

- [ ] **Step 4: Run the tests to see them pass, then the whole suite**

Run: `cd /home/yuchen/YuchenZ/UE/philip_sa && CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q tests/test_cityscapes_yolo.py`
Expected: `9 passed`.

Run: `CUDA_VISIBLE_DEVICES= /home/yuchen/miniconda3/envs/UE/bin/python -m pytest -q`
Expected: no failures, with 203 tests collected.

- [ ] **Step 5: Commit**

```bash
cd /home/yuchen/YuchenZ/UE/philip_sa
git add cityscapes/models/yolo11m.py tests/test_cityscapes_yolo.py
git commit -m "feat: the YOLO11m Cityscapes training and val commands, with Ultralytics' standard recipe

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0167vRw3NLCumE8FSGAnF1Tc" -- cityscapes/models/yolo11m.py tests/test_cityscapes_yolo.py
```

- [ ] **Step 6: Ask for the GPU (controller)**

Check the card with `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv`. Then use ListAgents to find "explore" and "ue-implement", and send each, via SendMessage:

> ue-finetune here (Cityscapes plan 1). May I run a YOLO11m smoke training on the card now? It's 1 epoch of Cityscapes train (2,975 images) at batch 16 and imgsz 640, capped at 12 GiB by `torch.cuda.set_per_process_memory_fraction`, for about 2–5 minutes. It also uses 8 CPU data-loader workers, and 16 during val. I'll send "done" when it ends, with its peak memory and time per epoch for the full 100-epoch run.

Wait for both replies before Step 7.

- [ ] **Step 7: The smoke run (controller)**

```bash
cd /home/yuchen/YuchenZ/UE/philip_sa
/home/yuchen/miniconda3/envs/UE/bin/python cityscapes/models/yolo11m.py train --smoke 2>&1 | tee runs/cityscapes-yolo11m-smoke.log | grep -v "it/s"
```

Expected in the log:
- `Transferred 643/649 items from pretrained weights`: everything but the class head's three last 1×1 convolutions, whose shapes change from 80 classes to 8;
- `train: Scanning /home/yuchen/YuchenZ/Datasets/cityscape/yolo_8cls/labels/train... 2975 images, 10 backgrounds, 0 corrupt`;
- `val: Scanning /home/yuchen/YuchenZ/Datasets/cityscape/yolo_8cls/labels/val... 500 images, 8 backgrounds, 0 corrupt`;
- `optimizer: AdamW(lr=0.000833, momentum=0.9)`, from `optimizer=auto`;
- a val table with `all 500 9792` and one row per class, 8 rows;
- `1 epochs completed in`, followed by `[train] peak GPU memory X GiB reserved`;
- no line `CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU`. Ultralytics' assigner catches running out of memory and carries on on the CPU, which slows every step. Treat such a line as running out of memory (below). Check with `grep -c "TaskAlignedAssigner" runs/cityscapes-yolo11m-smoke.log`, which should print 0.

Then `cat <folder>/results.csv`, where `<folder>` is the one the log's `Results saved to` line names: `runs/cityscapes-yolo11m/smoke`, or `smoke2` after an earlier smoke run. Its `time` column gives the epoch's seconds, val included.

If it runs out of memory under the 12 GiB cap, ask explore for 16 GiB, set `GPU_MEMORY_GIB = 16.0` (and the `("cap", 12.0)` entries in the test to 16.0), run the tests, commit both files, and repeat this step.

- [ ] **Step 8: Tell explore and ue-implement that the smoke run is done (controller)**

Send each: "done", the peak memory, and the full run's estimate, the smoke epoch's seconds × 100.

---

### Task 4: The full run, the frozen weights and their AP

**Files:**
- Create, outside git: `/home/yuchen/YuchenZ/lab/Detector_test/yolo11m_cityscapes_100e.pt`, a copy of the run's `last.pt`
- Modify: `cityscapes/models/README.md`

**Interfaces:**
- Consumes: `train()` and `val()` (Task 3); `README.md` (Task 1).
- Produces: the YOLO11m weights path above and its clean val AP, which plan 2 uses for its config and its floor.

- [ ] **Step 1: Ask for the GPU (controller)**

Send explore and ue-implement, via SendMessage:

> ue-finetune here (Cityscapes plan 1). May I start the full YOLO11m fine-tune now? It's 100 epochs, batch 16, imgsz 640, capped at <cap> GiB, and its peak in the smoke run was <peak> GiB. It should take about <estimate> h, and it also uses 8 CPU data-loader workers, and 16 during each epoch's val. I'll send "done" when it ends.

Fill in the cap, peak and estimate from Task 3, Step 8. Wait for both replies.

- [ ] **Step 2: Start the full run in the background (controller)**

Run with Bash's `run_in_background`:

```bash
cd /home/yuchen/YuchenZ/UE/philip_sa && /home/yuchen/miniconda3/envs/UE/bin/python cityscapes/models/yolo11m.py train > runs/cityscapes-yolo11m-train.log 2>&1
```

To check its progress, run `tail -3 runs/cityscapes-yolo11m/train/results.csv`.

- [ ] **Step 3: Check that it finished**

Run: `grep -E "epochs completed|Results saved to|peak GPU memory" runs/cityscapes-yolo11m-train.log; grep -c "TaskAlignedAssigner" runs/cityscapes-yolo11m-train.log`
Expected:
- `100 epochs completed in X hours.`, a `Results saved to` line, and the peak memory;
- then `0`. A positive count means some steps fell back to the CPU. That only slows training, but say so in the task report. If the saved folder isn't `runs/cityscapes-yolo11m/train`, because an earlier run already used that name, use the printed folder in Step 4.

- [ ] **Step 4: Freeze the last epoch's weights**

```bash
cp /home/yuchen/YuchenZ/UE/philip_sa/runs/cityscapes-yolo11m/train/weights/last.pt \
      /home/yuchen/YuchenZ/lab/Detector_test/yolo11m_cityscapes_100e.pt
ls -la /home/yuchen/YuchenZ/lab/Detector_test/yolo11m_cityscapes_100e.pt
```

- [ ] **Step 5: Measure the frozen weights on val (controller, under a minute)**

```bash
cd /home/yuchen/YuchenZ/UE/philip_sa
/home/yuchen/miniconda3/envs/UE/bin/python cityscapes/models/yolo11m.py val 2>&1 | tee runs/cityscapes-yolo11m-val.log | grep -v "it/s"
```
Expected: Ultralytics' table, with `all 500 9792` and 8 class rows, then `[val] yolo11m_cityscapes_100e.pt: mAP50-95 …, mAP50 …`.

**Stop rule:** if mAP50-95 is below 0.15, stop here. Report to the user with the table and `runs/cityscapes-yolo11m/train/results.csv`, and don't write the README section.

- [ ] **Step 6: Tell explore and ue-implement that the run is done (controller)**

Send explore "done". Send ue-implement "done", the weights path `/home/yuchen/YuchenZ/lab/Detector_test/yolo11m_cityscapes_100e.pt`, and the mAP50-95 and mAP50 from Step 5.

- [ ] **Step 7: Add YOLO11m to `README.md`**

Add a row under the RT-DETR row of the first table, with the numbers from Step 5:

```markdown
| YOLO11m | `/home/yuchen/YuchenZ/lab/Detector_test/yolo11m_cityscapes_100e.pt` | <mAP50-95> | <mAP50> |
```

Add this section between the RT-DETRv2-R18 section and "Notes":
- the training date and duration come from Step 3's `epochs completed` line;
- the table per class comes from Step 5's printed table, where each class's mAP50-95 is its AP and its mAP50 its AP50.

````markdown
## YOLO11m

- **Recipe:** Ultralytics 8.3.235's standard fine-tune, from the COCO `yolo11m.pt`, at imgsz 640, for 100 epochs at batch 16. Every other argument keeps its default (`runs/cityscapes-yolo11m/train/args.yaml`).
  - Here `optimizer=auto` picks AdamW at lr 8.3e-4 with momentum 0.9.
  - Mosaic stops for the last 10 epochs, and the seed is 0.
  - Ultralytics transfers 643 of the 649 pretrained tensors. It skips only the class head's three last 1×1 convolutions, whose shapes change from 80 classes to 8.
  - Trained on <date>, in <duration>.
- **Files:** `yolo11m.py` makes the labels and image links (`prepare`), trains (`train`) and measures the frozen weights (`val`). `yolo_8cls.yaml` is Ultralytics' data file.
  - Ultralytics' own `convert_coco` would turn the person class into −1, so `prepare` keeps each `category_id` as the class.
- **AP per class on val:** Ultralytics' val of the frozen `last.pt` at 640 (`python yolo11m.py val`; 300 detections).

| Class | AP | AP50 |
|---|---|---|
| person | | |
| rider | | |
| car | | |
| truck | | |
| bus | | |
| train | | |
| motorcycle | | |
| bicycle | | |
````

Fill every empty cell and every `<…>` from Step 5's output.

- [ ] **Step 8: Commit**

```bash
cd /home/yuchen/YuchenZ/UE/philip_sa
git add cityscapes/models/README.md
git commit -m "docs: YOLO11m fine-tuned on Cityscapes, frozen, with its AP per class

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0167vRw3NLCumE8FSGAnF1Tc" -- cityscapes/models/README.md
git show --stat --format= HEAD
```
Expected: one file, `cityscapes/models/README.md`.
