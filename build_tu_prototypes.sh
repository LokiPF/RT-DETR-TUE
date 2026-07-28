source .venv/bin/activate
python tools/build_tu_prototypes.py \
        -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml \
        -r pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \
        -o output/tu/bbox_prototypes.pth \
        --split train \
        --iou-threshold 0.70