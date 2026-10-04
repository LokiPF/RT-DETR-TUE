"""Faster R-CNN R50-FPN v2 (torchvision COCO_V1, box AP 46.7): two-stage, ResNet-50 with batch norm and an FPN.

The input goes through the model's own transform: the short side to 800 (the long side at most 1333), ImageNet
normalisation, padding to a multiple of 32 at the bottom and right. Levels: the first bottleneck block of each stage,
as RT-DETR's taps read the first block of each of its stages: layer1[0]-layer3[0] (strides 4, 8, 16) scored and
layer4[0] (stride 32) the key. CDF maps: the stem after max pooling and the four stage outputs. kNN: layer4's output,
mean-pooled. Detections: the model's own, with the score threshold lowered to 0.001 and 100 per image.
"""
from __future__ import annotations

import numpy as np
import torch

from .base import LABEL_OF_CATEGORY, TOP_K, Outputs, Region, padded

STRIDES = {1: 4, 2: 8, 3: 16, 4: 32}  # stage -> stride
SCORE_THRESHOLD = 0.001


class FasterRcnn:
    name = "faster_rcnn_r50_fpn_v2"
    batch_size = 8
    fit_batch_size = 1  # the crop needs one image size per batch
    detr = False
    protocol = {"levels": "layer1[0]-layer3[0], key layer4[0]", "cdf": "maxpool, layer1-layer4",
                "pooled": "layer4", "size": [800, 1333], "score_threshold": SCORE_THRESHOLD}

    def __init__(self, weights, device):
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

        self.device = torch.device(device)
        model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        model.roi_heads.score_thresh, model.roi_heads.detections_per_img = SCORE_THRESHOLD, TOP_K
        self.model = model.to(self.device).eval().requires_grad_(False)
        body, self._maps, self._sizes = self.model.backbone.body, {}, None
        self._handles = [body.maxpool.register_forward_hook(self._keep("stem")),
                         self.model.transform.register_forward_hook(self._keep_sizes)]
        for stage in STRIDES:
            layer = getattr(body, f"layer{stage}")
            self._handles += [layer[0].register_forward_hook(self._keep(f"s{stage}")),
                              layer.register_forward_hook(self._keep(f"C{stage + 1}"))]

    def _keep(self, name):
        def hook(_module, _inputs, output):
            self._maps[name] = output
        return hook

    def _keep_sizes(self, _module, _inputs, output):
        self._sizes = output[0].image_sizes  # the resized images inside the padded batch

    @torch.inference_mode()
    def __call__(self, arrays, heads: bool = True) -> Outputs:
        images = [torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float().div(255.0).to(self.device)
                  for a in arrays]
        self._maps.clear()
        if heads:
            results = self.model(images)
        else:
            self.model.backbone.body(self.model.transform(images)[0].tensors)
        height, width = self._sizes[0]
        region = Region(top=0.0, left=0.0, height=height, width=width)
        out = Outputs(levels={f"s{s}": region.crop(self._maps[f"s{s}"].float(), stride) for s, stride in STRIDES.items()},
                      cdf=[region.crop(self._maps["stem"].float(), 4)]
                          + [region.crop(self._maps[f"C{s + 1}"].float(), stride) for s, stride in STRIDES.items()],
                      pooled=region.crop(self._maps["C5"].float(), 32).mean(dim=(2, 3)))
        self._maps.clear()  # a later call's peak memory must not include these maps
        if heads:
            out.scores, out.labels, out.boxes = self._detections(results)
        return out

    @staticmethod
    def _detections(results):
        detections = []
        for result in results:
            labels = LABEL_OF_CATEGORY[result["labels"].cpu().numpy()]
            known = labels >= 0  # torchvision's 91-slot head never predicts the unused ids, but keep only COCO's 80
            detections.append(padded(result["scores"].cpu().numpy()[known], labels[known],
                                     result["boxes"].cpu().numpy()[known]))
        return tuple(np.stack(values) for values in zip(*detections))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
