source .venv/bin/activate
python tools/build_bbox_cp_calibration.py \
    -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco_tue.yml \
    -r pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \
    --tu-prototypes output/tu/bbox_prototypes.pth \
    -o output/tu/bbox_cp_calibration.pth \
    --split val \
    --alpha 0.05 \
    --tu-topk 50 \
    --target-box-format auto \
    --target-box-units auto \
    --confidence-threshold 0.5