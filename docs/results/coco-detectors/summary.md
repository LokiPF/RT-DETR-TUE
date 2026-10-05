# The two-axis score on four COCO detectors

AUROC, common / extra families. Untouched: positions 1970 and later.

| Detector | Images | Clean mAP | Two-axis, all | Two-axis, untouched | Two-axis, severity 1 common | CDFs, all | Best baseline, common | Headline | Level rule |
|---|---|---|---|---|---|---|---|---|---|
| rtdetrv2_r18 | 5000 | 0.479 | 0.917 / 0.858 | 0.916 / 0.858 | 0.863 | 0.821 / 0.807 | cdf 0.821 | confirmed | confirmed |
| yolo11m | 5000 | 0.505 | 0.890 / 0.870 | 0.891 / 0.870 | 0.825 | 0.835 / 0.814 | cdf 0.835 | confirmed | confirmed |
| faster_rcnn_r50_fpn_v2 | 5000 | 0.469 | 0.863 / 0.852 | 0.863 / 0.851 | 0.794 | 0.815 / 0.796 | cdf 0.815 | ahead of the activation CDFs, but the flattening adds nothing over the level score on the common families | confirmed |
| rfdetr_m | 5000 | 0.545 | 0.844 / 0.823 | 0.844 / 0.823 | 0.786 | 0.800 / 0.788 | cdf 0.800 | confirmed | confirmed |
