"""Copyright(c) 2023 lyuwenyu. All Rights Reserved."""

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from ...core import register

__all__ = ["RTDETRPostProcessor"]


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class RTDETRPostProcessor(nn.Module):
    __share__ = [
        "num_classes",
        "use_focal_loss",
        "num_top_queries",
        "remap_mscoco_category",
    ]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False,
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return f"use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}"

    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]

        bbox_pred = torchvision.ops.box_convert(
            boxes,
            in_fmt="cxcywh",
            out_fmt="xyxy",
        )
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = torch.sigmoid(logits)

            scores, flat_index = torch.topk(
                scores.flatten(1),
                self.num_top_queries,
                dim=-1,
            )

            labels = mod(flat_index, self.num_classes)
            query_indices = flat_index // self.num_classes

            boxes = bbox_pred.gather(
                dim=1,
                index=query_indices.unsqueeze(-1).expand(-1, -1, bbox_pred.shape[-1]),
            )

        else:
            scores = F.softmax(logits, dim=-1)[:, :, :-1]
            scores, labels = scores.max(dim=-1)

            if scores.shape[1] > self.num_top_queries:
                scores, query_indices = torch.topk(
                    scores,
                    self.num_top_queries,
                    dim=-1,
                )

                labels = torch.gather(
                    labels,
                    dim=1,
                    index=query_indices,
                )

                boxes = torch.gather(
                    bbox_pred,
                    dim=1,
                    index=query_indices.unsqueeze(-1).expand(
                        -1, -1, bbox_pred.shape[-1]
                    ),
                )
            else:
                boxes = bbox_pred

                query_indices = (
                    torch.arange(
                        scores.shape[1],
                        device=scores.device,
                    )
                    .unsqueeze(0)
                    .expand(scores.shape[0], -1)
                )

        if self.deploy_mode:
            return labels, boxes, scores

        if self.remap_mscoco_category:
            from ...data.dataset import mscoco_label2category

            labels = (
                torch.tensor(
                    [mscoco_label2category[int(x.item())] for x in labels.flatten()]
                )
                .to(boxes.device)
                .reshape(labels.shape)
            )

        results = []

        for lab, box, sco, query_idx in zip(
            labels,
            boxes,
            scores,
            query_indices,
        ):
            results.append(
                {
                    "labels": lab,
                    "boxes": box,
                    "scores": sco,
                    "query_indices": query_idx,
                }
            )

        return results

    def deploy(
        self,
    ):
        self.eval()
        self.deploy_mode = True
        return self
