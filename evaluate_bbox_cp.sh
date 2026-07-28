source .venv/bin/activate
python tools/evaluate_bbox_cp.py \
    -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco_tue.yml \
    -r pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \
    --tu-prototypes output/tu/bbox_prototypes.pth \
    --cp-calibration output/tu/bbox_cp_calibration.pth \
    --test-annotations /home/philip/datasets/coco/annotations/instances_val2017_test.json \
    -o output/tu/bbox_cp_test_metrics.json \
    --bootstrap-repeats 1000 \
    --bootstrap-confidence 0.95