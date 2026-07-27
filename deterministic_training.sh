source .venv/bin/activate
python tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml \
  -d cuda:0 \
  -t pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \
  --seed 42