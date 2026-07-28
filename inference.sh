source .venv/bin/activate
python tools/infer_bbox_uncertainty.py \
    -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco_tue.yml \
    -r pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \
    --tu-prototypes output/tu/bbox_prototypes.pth \
    --cp-calibration output/tu/bbox_cp_calibration.pth \
    --input dataset/n003-2018-01-02-11-48-43+0800__CAM_FRONT_LEFT__1514864966138575.jpg \
    --output-dir output/tu/visualizations \
    --confidence-threshold 0.5 \
    --cp-mode scaled \
    --save-json \
    --overwrite