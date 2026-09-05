import torch
import torchvision
from torch import Tensor, nn

from ....misc.tue_utils import topological_uncertainty_from_captures

__all__ = ["TUEBase"]


def mod(a, b):
    out = a - a // b * b
    return out


import matplotlib.pyplot as plt
import numpy as np


def plot_tu_heatmap(
    plot_dict: dict,
    figsize=(12, 7),
    save_path: str | None = None,
):
    data = plot_dict

    layers = list(data.keys())

    classes = sorted({cls for layer_data in data.values() for cls in layer_data.keys()})

    matrix = np.full(
        (len(layers), len(classes)),
        np.nan,
        dtype=float,
    )

    for i, layer_name in enumerate(layers):
        for j, cls in enumerate(classes):
            if cls not in data[layer_name]:
                continue

            x = data[layer_name][cls]

            if isinstance(x, list):
                x = torch.cat(x)

            matrix[i, j] = x.float().mean().item()

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        matrix,
        aspect="auto",
    )

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)

    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)

    ax.set_xlabel("Class")
    ax.set_ylabel("Layer")
    ax.set_title("Mean topological uncertainty")

    fig.colorbar(
        im,
        ax=ax,
        label="Mean TU",
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )

    plt.show()
    plt.close(fig)


class TUEBase(nn.Module):
    def __init__(self, frechet_means_path: str, num_top_queries: int, num_classes: int):
        super().__init__()

        calibration = torch.load(
            frechet_means_path,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
        )

        self.frechet_means = calibration["means"]

        self.num_top_queries = num_top_queries
        self.num_classes = num_classes

        self.plot = {}

    def forward(
        self,
        outputs: dict,
        selection_mask: Tensor | None = None,
    ) -> dict:
        raise NotImplementedError("TUEBase Inherit must implement a forward function.")

    def get_query_indices(self, logits, boxes):
        scores = torch.sigmoid(logits)

        bbox_pred = torchvision.ops.box_convert(
            boxes,
            in_fmt="cxcywh",
            out_fmt="xyxy",
        )

        scores, flat_index = torch.topk(
            scores.flatten(1),
            self.num_top_queries,
            dim=-1,
        )

        query_indices = flat_index // self.num_classes
        labels = mod(flat_index, self.num_classes)

        boxes = bbox_pred.gather(
            dim=1,
            index=query_indices.unsqueeze(-1).expand(-1, -1, bbox_pred.shape[-1]),
        )

        return query_indices, labels, scores, boxes

    def plot_heat(self):
        plot_tu_heatmap(self.plot)
        self.plot = {}

    def calculate_tu(self, output, min_score: float = 0):
        captures = output["tue_info"]
        query_idx, classes, scores, boxes = self.get_query_indices(
            output["pred_logits"], output["pred_boxes"]
        )
        selection_mask = scores > min_score

        has_reference = torch.tensor(
            [int(cls) in self.frechet_means for cls in classes.flatten()],
            device=classes.device,
            dtype=torch.bool,
        ).reshape_as(classes)

        selection_mask &= has_reference

        B, K = query_idx.shape  # K is the number of detections

        batch_indices = (
            torch.arange(B, device=query_idx.device).unsqueeze(1).expand(B, K)
        )

        if int(selection_mask.sum()) == 0:  # no detections
            tu_per_image = [
                torch.empty(0, device=scores.device, dtype=scores.dtype)
                for _ in range(B)
            ]
        else:
            tu_flat = topological_uncertainty_from_captures(
                captures,
                batch_indices[selection_mask],
                query_idx[selection_mask],
                classes[selection_mask],
                self.frechet_means,
                self.plot,
            )

            counts = selection_mask.sum(dim=1).tolist()
            tu_per_image = torch.split(tu_flat, counts)

        results = []

        for b in range(B):
            mask = selection_mask[b]
            results.append(
                {
                    "labels": classes[b][mask],
                    "boxes": boxes[b][mask],
                    "scores": scores[b][mask],
                    "query_indices": query_idx[b][mask],
                    "tu": tu_per_image[b],
                }
            )

        return results
