"""
Visualize TUE on a half-corrupted image.

Example:
python visualize_half_corruption.py \
    -c config.yml \
    -r checkpoint.pth \
    --corruption gaussian_blur \
    --side right
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from imagecorruptions import corrupt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.core import YAMLConfig


def build_runtime(config: str, ckpt: str):
    cfg = YAMLConfig(config)

    model = cfg.model
    dataloader = cfg.val_dataloader

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(
        ckpt,
        map_location=device,
    )

    if "ema" in checkpoint:
        state_dict = checkpoint["ema"]

        if isinstance(state_dict, dict) and "module" in state_dict:
            state_dict = state_dict["module"]

    elif "model" in checkpoint:
        state_dict = checkpoint["model"]

    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(device)
    model.eval()

    return model, dataloader


def corrupt_partial(
    image: torch.Tensor,
    corruption_name: str,
    side: str = "right",
    percentage: float = 1.0,
    severity: int = 5,
):
    """
    image: [C, H, W] in [0, 1]

    percentage:
        Fraction of the image to corrupt, in [0, 1].

        Examples:
            0.25 -> 25%
            0.50 -> 50%
            0.75 -> 75%
    """

    if not 0.0 <= percentage <= 1.0:
        raise ValueError(f"percentage must be in [0, 1], got {percentage}")

    image_np = image.detach().cpu().permute(1, 2, 0).numpy()

    image_uint8 = np.clip(
        image_np * 255.0,
        0,
        255,
    ).astype(np.uint8)

    corrupted_full = corrupt(
        image_uint8,
        corruption_name=corruption_name,
        severity=severity,
    )

    output = image_uint8.copy()

    H, W = output.shape[:2]

    corrupt_w = int(round(W * percentage))
    corrupt_h = int(round(H * percentage))

    if side == "right":
        output[:, W - corrupt_w :] = corrupted_full[:, W - corrupt_w :]

    elif side == "left":
        output[:, :corrupt_w] = corrupted_full[:, :corrupt_w]

    elif side == "top":
        output[:corrupt_h] = corrupted_full[:corrupt_h]

    elif side == "bottom":
        output[H - corrupt_h :] = corrupted_full[H - corrupt_h :]

    else:
        raise ValueError(f"Unknown side '{side}'. Expected left/right/top/bottom.")

    return torch.from_numpy(output.copy()).permute(2, 0, 1).float() / 255.0


def get_random_image(dataloader):
    samples, targets = next(iter(dataloader))

    idx = random.randrange(len(samples))

    return samples[idx], targets[idx]


def draw_detections(
    image,
    result,
    title,
    score_threshold=0.0,
):
    """
    image: [C,H,W]
    result:
        {
            "boxes": [K,4],
            "scores": [K],
            "labels": [K],
            "tu": [K],
        }
    """

    image_np = image.detach().cpu().permute(1, 2, 0).numpy()

    boxes = result["boxes"].detach().cpu()
    scores = result["scores"].detach().cpu()
    labels = result["labels"].detach().cpu()
    tu = result["tu"].detach().cpu()

    H, W = image.shape[-2:]

    boxes = result["boxes"].detach().cpu().clone()

    boxes[:, [0, 2]] *= W
    boxes[:, [1, 3]] *= H

    fig, ax = plt.subplots(
        figsize=(12, 8),
    )

    ax.imshow(image_np)

    for box, score, label, uncertainty in zip(
        boxes,
        scores,
        labels,
        tu,
    ):
        if score < score_threshold:
            continue

        x1, y1, x2, y2 = box.tolist()

        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2,
        )

        ax.add_patch(rect)

        text = (
            f"class={int(label)}\nscore={float(score):.3f}\nTU={float(uncertainty):.3f}"
        )

        ax.text(
            x1,
            y1,
            text,
            fontsize=8,
            verticalalignment="bottom",
            bbox={
                "alpha": 0.7,
                "pad": 2,
            },
        )

    ax.set_title(title)
    ax.axis("off")

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, dataloader = build_runtime(
        args.config,
        args.ckpt,
    )

    device = next(model.parameters()).device

    # --------------------------------------------------
    # Pick random image
    # --------------------------------------------------

    image, target = get_random_image(
        dataloader,
    )

    image = image.cpu()

    # --------------------------------------------------
    # Half corruption
    # --------------------------------------------------

    corrupted = corrupt_partial(
        image,
        corruption_name=args.corruption,
        side=args.side,
        severity=args.severity,
    )

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    batch = corrupted.unsqueeze(0).to(device)

    results = model(batch)

    result = results[0]

    # --------------------------------------------------
    # Print all detections
    # --------------------------------------------------

    print()
    print(f"{'idx':>4} {'class':>7} {'score':>10} {'TU':>12}")

    print("-" * 38)

    for i, (
        label,
        score,
        uncertainty,
    ) in enumerate(
        zip(
            result["labels"],
            result["scores"],
            result["tu"],
        )
    ):
        print(f"{i:4d} {int(label):7d} {float(score):10.4f} {float(uncertainty):12.4f}")

    # --------------------------------------------------
    # Visualize
    # --------------------------------------------------

    draw_detections(
        corrupted,
        result,
        title=(
            f"{args.side.capitalize()} half corrupted — "
            f"{args.corruption}, severity={args.severity}"
        ),
        score_threshold=args.score_threshold,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        required=True,
    )

    parser.add_argument(
        "-r",
        "--ckpt",
        required=True,
    )

    parser.add_argument(
        "--corruption",
        default="gaussian_blur",
    )

    parser.add_argument(
        "--severity",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--side",
        choices=[
            "left",
            "right",
            "top",
            "bottom",
        ],
        default="right",
    )

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    main(parser.parse_args())
