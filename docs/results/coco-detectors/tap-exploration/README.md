# Tap explorations

Which feature levels the method reads in a detector without RT-DETR's ResNet stages. Both were chosen on development
images, before any confirmation run. The scripts ran from a session scratchpad on 4 October 2026, and their caches of
per-image channel statistics were not kept.

- `vit_taps.py`, `vit_taps.log`: RF-DETR-M's DINOv2 ViT-S. Blocks 1, 2 and 3 with block 12 as the key beat blocks 3, 6
  and 9 with block 12, both on raw block outputs and on layer-normed ones.
- `yolo_taps.py`, `yolo_taps.log`: YOLO11m, five sets of backbone layers. Layers 1, 3 and 5 with layer 7 as the key
  lead on the common families. Each is a stage's stride-2 downsampling conv.

Both read the same 300 of positions 0–1969 of the seed-44 evaluation order, drawn with seed 7. The untouched images,
positions 1970 and later, stay unread. Each log ends with a depth curve: every layer's own level and peak-share scores
against the clean bank.
