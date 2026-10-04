"""YOLO11m (Ultralytics 8.3.235, COCO val AP 51.5): one-stage, anchor-free, a CSP backbone with SiLU, NMS.

The input is letterboxed as Ultralytics does for same-size images: the long side to 640, the short side padded with
grey 114 to a multiple of 32, centred. Each stage is a stride-2 downsampling conv followed by a C3k2 block: layers
(1, 2), (3, 4), (5, 6) and (7, 8); layer 0 is the stem, and SPPF (9) and C2PSA (10) end the backbone. Levels: each
stage's first module, as RT-DETR's taps read each stage's first block: layers 1, 3 and 5 (strides 4, 8, 16) scored
and layer 7 (stride 32) the key, chosen on 300 development images on 4 October over the stage blocks 2, 4, 6 + 8.
CDF maps: the stem and the four stage outputs, the last at the end of the backbone (layers 0, 2, 4, 6 and 10). kNN:
layer 10, mean-pooled. Detections: NMS as Ultralytics' predict does it (one label per box), at confidence 0.001 and
IoU 0.7 and with no time limit, the 100 most confident.
"""
from __future__ import annotations

import numpy as np
import torch

from .base import TOP_K, Outputs, Region, padded

LEVEL_LAYERS = {"s1": (1, 4), "s2": (3, 8), "s3": (5, 16), "s4": (7, 32)}  # level -> (layer index, stride)
CDF_LAYERS = ((0, 2), (2, 4), (4, 8), (6, 16), (10, 32))
POOLED_LAYER = (10, 32)
BACKBONE_LAYERS = 11  # layers 0-10 run in sequence; the neck and the head follow
CONFIDENCE, IOU, SIZE = 0.001, 0.7, 640


class Yolo11m:
    name = "yolo11m"
    batch_size = 16  # the 96 versions of one image, 16 at a time
    fit_batch_size = 1  # clean train images differ in size, and a letterboxed batch needs one size
    detr = False
    protocol = {"levels": LEVEL_LAYERS, "cdf": CDF_LAYERS, "pooled": POOLED_LAYER, "size": SIZE,
                "letterbox": "Ultralytics LetterBox, auto=True, stride 32, centred, grey 114",
                "confidence": CONFIDENCE, "iou": IOU, "nms": "one label per box, as predict does, no time limit"}

    def __init__(self, weights, device):
        from ultralytics import YOLO

        self.device = torch.device(device)
        self.model = YOLO(str(weights)).model.to(self.device).eval().requires_grad_(False)
        self._maps = {}
        layers = {i for i, _ in LEVEL_LAYERS.values()} | {i for i, _ in CDF_LAYERS} | {POOLED_LAYER[0]}
        self._handles = [self.model.model[i].register_forward_hook(self._keep(i)) for i in sorted(layers)]

    def _keep(self, index):
        def hook(_module, _inputs, output):
            self._maps[index] = output
        return hook

    def prepare(self, arrays) -> tuple[torch.Tensor, Region]:
        """The letterboxed batch (RGB in [0, 1]) and where the image lies in it."""
        from ultralytics.data.augment import LetterBox

        if len({a.shape[:2] for a in arrays}) > 1:  # the region and the boxes follow the first image
            raise ValueError("an adapter batch must hold images of one size")
        height, width = arrays[0].shape[:2]
        letterbox = LetterBox(new_shape=(SIZE, SIZE), auto=True, stride=32)
        boxed = [letterbox(image=np.ascontiguousarray(a)) for a in arrays]
        input_height, input_width = boxed[0].shape[:2]
        ratio = min(SIZE / height, SIZE / width)
        new_height, new_width = round(height * ratio), round(width * ratio)
        region = Region(top=(input_height - new_height) // 2, left=(input_width - new_width) // 2,
                        height=new_height, width=new_width)
        batch = torch.from_numpy(np.stack(boxed)).permute(0, 3, 1, 2).float().div_(255.0)
        return batch, region

    @torch.inference_mode()
    def __call__(self, arrays, heads: bool = True) -> Outputs:
        batch, region = self.prepare(arrays)
        x = batch.to(self.device)
        self._maps.clear()
        if heads:
            prediction = self.model(x)
        else:
            for layer in self.model.model[:BACKBONE_LAYERS]:
                x = layer(x)
        out = Outputs(levels={n: region.crop(self._maps[i].float(), s) for n, (i, s) in LEVEL_LAYERS.items()},
                      cdf=[region.crop(self._maps[i].float(), s) for i, s in CDF_LAYERS],
                      pooled=region.crop(self._maps[POOLED_LAYER[0]].float(), POOLED_LAYER[1]).mean(dim=(2, 3)))
        self._maps.clear()  # a later call's peak memory must not include these maps
        if heads:
            out.scores, out.labels, out.boxes = self._detections(prediction, batch.shape[2:], arrays[0].shape[:2])
        return out

    def _detections(self, prediction, input_shape, image_shape):
        """The 100 most confident boxes after NMS, in the original image's pixels."""
        from ultralytics.utils import ops
        from ultralytics.utils.nms import non_max_suppression

        prediction = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
        # Ultralytics stops NMS after 2 + 0.05 x batch seconds and leaves the rest of the batch empty: no time limit
        kept = non_max_suppression(prediction, conf_thres=CONFIDENCE, iou_thres=IOU, max_det=TOP_K, max_time_img=1e3)
        detections = []
        for found in kept:
            boxes = ops.scale_boxes(tuple(input_shape), found[:, :4].clone(), image_shape)
            detections.append(padded(found[:, 4].cpu().numpy(), found[:, 5].long().cpu().numpy(),
                                     boxes.cpu().numpy()))
        return tuple(np.stack(values) for values in zip(*detections))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
