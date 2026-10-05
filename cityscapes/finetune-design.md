# Two detectors trained on Cityscapes: design

5 October 2026. This is plan 1 of 2. Plan 2 is the Cityscapes-C evaluation of both detectors, with the two-axis score and every baseline.

## Goal

Two Cityscapes detectors, frozen and described in this folder, ready for plan 2:
- **RT-DETRv2-R18** was fine-tuned on 4 October (72 epochs, AP 0.382 on val). It only needs to be frozen and recorded.
- **YOLO11m** is to be fine-tuned from its COCO weights, then frozen and recorded.

## Decided

- **YOLO11m's recipe** is Ultralytics' standard fine-tune.
  - It starts from the COCO `yolo11m.pt` used on COCO-C (Ultralytics 8.3.235).
  - imgsz 640, 100 epochs, batch 16; every other argument keeps its default.
  - Here `optimizer=auto` resolves to AdamW at lr 8.3e-4 with momentum 0.9. Mosaic stops for the last 10 epochs, and the seed is 0.
  - This mirrors RT-DETR, which was fine-tuned with its own repository's standard recipe.
- **Classes:** the 8 Cityscapes instance classes, in RT-DETR's order: person, rider, car, truck, bus, train, motorcycle, bicycle.
  - The annotations are the files RT-DETR trained on, `annotations_coco/cityscapes_{train,val}_8cls.json`.
- **Weights from the last epoch,** for both detectors. Val is evaluated during training but selects nothing.
- **No AP target.** The APs are reported as they come out.
  - One stop rule: a YOLO mAP50-95 below 0.15 points to a label or class-order bug, so we stop and report.
- **Where:** this folder, on `fingerprint_bank`. The paper session commits from the same checkout, so every commit names its paths (`git commit -- cityscapes/ …`).
- **Little bookkeeping:** no hashes or provenance records.

## Layout

In git:

```
cityscapes/
  finetune-design.md   this spec
  finetune-plan.md     the implementation plan
  README.md            the record of both detectors
  yolo11m.py           prepare | train [--smoke] | val
  yolo_8cls.yaml       Ultralytics' data file: the dataset folder, train and val, the 8 names
  rtdetrv2/            RT-DETR's three training files, copied from RT-DETRv2-UE
tests/test_cityscapes_yolo.py
```

Outside git:
- `Datasets/cityscape/yolo_8cls/`:
  - `images/{train,val}/`: symlinks to the `leftImg8bit` PNGs;
  - `labels/{train,val}/`: one text file per image.
- `runs/cityscapes-yolo11m/{smoke,train}/`: Ultralytics' training output (git-ignored).
- The frozen weights:
  - RT-DETR: `RT-DETRv2-UE/pretrained_weights/rtdetrv2_r18vd_cityscapes_72e.pth`, a copy of `last.pth`;
  - YOLO: `lab/Detector_test/yolo11m_cityscapes_100e.pt`, a copy of `last.pt`, next to the COCO `yolo11m.pt`.

## RT-DETRv2-R18 (CPU only)

- **Freeze it:** copy `output/rtdetrv2_r18vd_cityscapes/last.pth` to `pretrained_weights/`.
- **Keep its training files:** copy these three into `rtdetrv2/`. They are untracked in RT-DETRv2-UE, and RT-DETRv2-UE itself is left as it is. To rerun the training, copy them back into RT-DETRv2-UE.
  - `configs/dataset/cityscapes_detection.yml`
  - `configs/rtdetrv2/rtdetrv2_r18vd_cityscapes.yml`
  - `dataset_tools/cityscapes_converter.py`
- **Clean AP:** the log gives AP 0.382 and AP50 0.590 at the last epoch.
  - The AP per class is read from `eval/latest.pth`, which holds the last epoch's evaluation of the EMA weights.

## YOLO11m

**Data (CPU):** `python cityscapes/yolo11m.py prepare`.
- **Files:** for each split, one symlink and one label file per image. The label file has one line per box, `class cx cy w h`, normalised by the image's width and height.
- **Class ids:** the class is the file's `category_id`, unchanged (0–7).
  - Ultralytics' `convert_coco` can't be used: it writes `category_id − 1`, which turns person into −1.
  - Ultralytics then rejects every image that holds a person as corrupt.
- **Images without a box** (10 in train, 8 in val) get an empty label file. Ultralytics reads these as background.
- **Nothing is clipped:** every box lies inside its image and none has zero size (checked on 5 October).
- **Counts:** the command prints the images and boxes per split. Expected: 2,975 images and 50,347 boxes in train, 500 and 9,792 in val.

**Training (GPU):** `python cityscapes/yolo11m.py train`.
- **`--smoke`:** 1 epoch on 5% of train. It measures peak GPU memory and seconds per epoch, before we ask for the full run.
- **The full run:** 100 epochs, expected to take 1–2 h.
- **Memory cap:** the script caps its own GPU memory at 12 GiB. The smoke run shows whether the peak fits.

**Clean AP (GPU, under a minute):** `python cityscapes/yolo11m.py val`.
- Ultralytics' val of the frozen `last.pt` on the 500 val images at 640: mAP50-95, mAP50 and the table per class.
- Ultralytics' own end-of-training val scores `best.pt`, so this command runs on `last.pt` itself.

## The record: `README.md`

For each detector:
- the recipe, in a few lines;
- the training time;
- AP, AP50 and the AP per class on val;
- where the weights are.

Two notes:
- **Each toolkit evaluates its own detector here.** RT-DETR's numbers come from pycocotools with 100 detections, YOLO's from Ultralytics' own mAP with 300. Plan 2's check stage measures both detectors through our adapters.
- **"640" doesn't mean the same input size.** YOLO letterboxes 2048×1024 to 640×320, while RT-DETR resizes it to 640×640. YOLO therefore sees half the pixels, and every object is half as tall.

## Tests (CPU)

`tests/test_cityscapes_yolo.py` runs on a fake annotation file and two small images in a temporary folder. It checks that:
- a known box becomes the known normalised line, and class 0 stays 0;
- the image without a box gets an empty label file;
- each symlink points at its source image.

## GPU and machine

- **The card:** the IQA pass holds it until about 19:30, at 2 GiB and 96% use.
  - Ask explore and ue-implement before the smoke run and again before the full run.
  - Send "done" at the end.
- **RAM:** the data loaders need about 10–15 GB, next to ue-implement's reports at up to 50 GiB, out of 91 GB.

## Not in this plan (plan 2)

The Cityscapes-C evaluation:
- the package changes;
- both detectors' AP floors, measured through our adapters;
- the pre-registered rule for YOLO11m;
- the role of the IQA baselines.
