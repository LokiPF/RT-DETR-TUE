# The two-axis score against four image-quality baselines

Each row's AUROC, and the two-axis score minus it with 95% paired bootstrap intervals, on all images.

| Detector | Row | AUROC all: common / extra | Two-axis − row, all: common | extra |
|---|---|---|---|---|
| rtdetrv2_r18 | NIQE (refit) | 0.950 / 0.871 | -0.002 [-0.005, +0.001] | +0.099 [+0.096, +0.102] |
| rtdetrv2_r18 | NIQE (published, sensitivity) | 0.930 / 0.804 | +0.018 [+0.015, +0.021] | +0.165 [+0.162, +0.168] |
| rtdetrv2_r18 | ARNIQA quality | 0.920 / 0.843 | +0.029 [+0.024, +0.034] | +0.127 [+0.122, +0.134] |
| rtdetrv2_r18 | ARNIQA prototype | 0.982 / 0.990 | -0.033 [-0.036, -0.030] | -0.021 [-0.023, -0.018] |
| rtdetrv2_r18 | CLIP-IQA | 0.460 / 0.256 | +0.488 [+0.477, +0.499] | +0.714 [+0.705, +0.724] |
| yolo11m | NIQE (refit) | 0.950 / 0.871 | -0.030 [-0.034, -0.026] | +0.073 [+0.069, +0.077] |
| yolo11m | NIQE (published, sensitivity) | 0.930 / 0.804 | -0.010 [-0.013, -0.006] | +0.139 [+0.134, +0.143] |
| yolo11m | ARNIQA quality | 0.920 / 0.843 | +0.001 [-0.005, +0.007] | +0.101 [+0.094, +0.108] |
| yolo11m | ARNIQA prototype | 0.982 / 0.990 | -0.061 [-0.065, -0.057] | -0.047 [-0.051, -0.043] |
| yolo11m | CLIP-IQA | 0.460 / 0.256 | +0.460 [+0.451, +0.472] | +0.688 [+0.679, +0.698] |
