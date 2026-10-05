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
