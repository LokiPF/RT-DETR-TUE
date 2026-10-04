"""RF-DETR-M (rfdetr 1.11.2, COCO AP 54.7): a real-time DETR on a DINOv2 ViT-S backbone (12 blocks, patch 16).

The input is resized to 576 x 576 (bilinear, no antialiasing) and normalised, as its predict does. A plain ViT has
no strides, so its levels are blocks: the raw outputs of blocks 1, 2 and 3 scored and block 12 the key, their patch
tokens regathered from the attention windows (chosen on positions 0-1969, 4 October). CDF maps: the embeddings and
the same four blocks. kNN: block 12 after the backbone's LayerNorm, the feature the detector reads, mean-pooled.
Detections as RT-DETR's: the sigmoid of every query's logits for COCO's 80 categories, the top 100 (query, class)
pairs. Hashemi et al.: the last decoder layer's output, RT-DETR's hook point, before the decoder's final LayerNorm.
"""
from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms.v2.functional as F

from ..detector.postprocess import top_detections
from .base import COCO_CATEGORY_IDS, TOP_K, Outputs

BLOCKS = {"s1": 1, "s2": 2, "s3": 3, "s4": 12}
CDF_BLOCKS = (0, 1, 2, 3, 12)  # 0 is the embeddings
RESOLUTION, WINDOWS = 576, 2
MEANS, STDS = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class RfDetrM:
    name = "rfdetr_m"
    batch_size = 32
    fit_batch_size = 32  # every image is resized to 576 x 576, so a batch may mix image sizes
    detr = True
    protocol = {"levels": BLOCKS, "cdf": CDF_BLOCKS,
                "maps": "levels and cdf: raw block outputs (0: the embeddings), before the backbone's LayerNorm",
                "pooled": "layernorm(block 12)", "size": RESOLUTION, "resize": "bilinear, no antialiasing",
                "decoder": "transformer.decoder.layers[-1] output, before the decoder's final LayerNorm",
                "classes": "the 80 COCO category columns of 91"}

    def __init__(self, weights, device):
        precision = torch.get_float32_matmul_precision()
        from rfdetr import RFDETRMedium

        self.device = torch.device(device)
        self.core = RFDETRMedium(pretrain_weights=str(weights)).model.model.to(self.device).eval().requires_grad_(False)
        torch.set_float32_matmul_precision(precision)  # importing rfdetr switched float32 matmuls to TF32 process-wide
        self.backbone = self.core.backbone[0].encoder  # returns the LayerNorm-ed maps the detector reads
        dino = self.backbone.encoder  # the windowed DINOv2 ViT-S
        self.layernorm = dino.layernorm
        self.grid = RESOLUTION // dino.config.patch_size
        self.skip = 1 + dino.config.num_register_tokens  # each window's class token and register tokens
        self._tokens = {}
        hooked = sorted(set(BLOCKS.values()) | set(CDF_BLOCKS))
        self._handles = [dino.embeddings.register_forward_hook(self._keep(0))]
        self._handles += [dino.encoder.layer[block - 1].register_forward_hook(self._keep(block, first=True))
                          for block in hooked if block > 0]
        self._handles.append(self.core.transformer.decoder.layers[-1].register_forward_hook(self._keep("decoder")))

    def _keep(self, key, first=False):
        def hook(_module, _inputs, output):
            self._tokens[key] = output[0] if first else output
        return hook

    def maps(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B * windows^2, skip + T, C) tokens -> (B, C, grid, grid), undoing the embeddings' window layout."""
        tokens = tokens[:, self.skip:]
        side = self.grid // WINDOWS
        batch = tokens.shape[0] // WINDOWS ** 2
        x = tokens.reshape(batch, WINDOWS, WINDOWS, side, side, -1)  # (B, window row, window column, h, w, C)
        return x.permute(0, 5, 1, 3, 2, 4).reshape(batch, -1, self.grid, self.grid)

    @torch.inference_mode()
    def __call__(self, arrays, heads: bool = True) -> Outputs:
        images = torch.stack([F.resize(torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float().div(255.0),
                                       [RESOLUTION, RESOLUTION], antialias=False) for a in arrays])
        x = F.normalize(images, MEANS, STDS).to(self.device)
        self._tokens.clear()
        predicted = self.core(x) if heads else self.backbone(x)
        out = Outputs(levels={name: self.maps(self._tokens[block]).float() for name, block in BLOCKS.items()},
                      cdf=[self.maps(self._tokens[block]).float() for block in CDF_BLOCKS],
                      pooled=self.maps(self.layernorm(self._tokens[12])).float().mean(dim=(2, 3)))
        if heads:
            logits = predicted["pred_logits"].float()[..., list(COCO_CATEGORY_IDS)].cpu().numpy()  # (N, 300, 80)
            boxes = predicted["pred_boxes"].float().cpu().numpy()  # (N, 300, 4), cxcywh in [0, 1]
            detections = [top_detections(l, b, (a.shape[1], a.shape[0]), TOP_K)
                          for l, b, a in zip(logits, boxes, arrays)]
            out.scores, out.labels, out.boxes = (np.stack(values) for values in zip(*detections))
            out.query_logits, out.query_boxes, out.decoder = logits, boxes, self._tokens["decoder"].float()
        self._tokens.clear()  # a later call's peak memory must not include these tokens
        return out

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
