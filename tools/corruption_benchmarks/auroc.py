#!/usr/bin/env python3
"""Benchmark TUE and confidence AUROC on clean versus corrupted images."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Compatibility for imagecorruptions with NumPy 2.x.
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128

import imagecorruptions.corruptions as _ic
import skimage.filters
import torch
from imagecorruptions import corrupt, get_corruption_names
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm

REPO_ROOT = next(
    (
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "src").is_dir()
    ),
    None,
)
if REPO_ROOT is None:
    raise RuntimeError("Could not find the RT-DETR project root containing src/")
sys.path.insert(0, str(REPO_ROOT))

from src.core import YAMLConfig

# Compatibility for imagecorruptions with skimage >= 0.21.
_gaussian = skimage.filters.gaussian
_ic.gaussian = lambda *args, **kwargs: _gaussian(
    *args,
    **({**kwargs, "channel_axis": -1} if kwargs.pop("multichannel", None) else kwargs),
)


def build(config_path: str, checkpoint_path: str):
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get("ema") is not None:
        state_dict = checkpoint["ema"]["module"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    cfg.model.load_state_dict(state_dict)
    model = cfg.model.deploy().eval().cuda()
    return model, cfg


def resize_image(image: np.ndarray) -> np.ndarray:
    image = Image.fromarray(image).resize(
        (640, 640),
        Image.Resampling.BILINEAR,
    )
    return np.array(image, dtype=np.uint8, copy=True, order="C")


def to_tensor(image: np.ndarray) -> torch.Tensor:
    image = np.array(image, dtype=np.uint8, copy=True, order="C")
    return torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)


def dataset_paths(cfg):
    dataset = cfg.val_dataloader.dataset
    for image_id in dataset.ids:
        info = dataset.coco.loadImgs(image_id)[0]
        yield os.path.join(dataset.img_folder, info["file_name"])


class CorruptionStream(IterableDataset):
    def __init__(
        self,
        paths: list[str],
        corruption_names: list[str],
        severity: int,
    ):
        self.paths = paths
        self.corruption_names = corruption_names
        self.severity = severity

    def __len__(self):
        return len(self.paths) * (len(self.corruption_names) + 1)

    def __iter__(self):
        worker = get_worker_info()
        start = 0 if worker is None else worker.id
        step = 1 if worker is None else worker.num_workers

        for image_index in range(start, len(self.paths), step):
            with Image.open(self.paths[image_index]) as image_file:
                image = np.array(image_file.convert("RGB"), copy=True)

            image = resize_image(image)
            yield "clean", to_tensor(image)

            for corruption_name in self.corruption_names:
                corrupted = corrupt(
                    image,
                    corruption_name=corruption_name,
                    severity=self.severity,
                )
                yield corruption_name, to_tensor(corrupted)


@torch.inference_mode()
def score_tensor_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    topk: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    images = images.to(device, non_blocking=True)

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        # Capture detector activations without running the TUE head first.
        output = model.forward_detector(images)

    logits = output["pred_logits"]
    confidence = logits.sigmoid().amax(dim=-1)

    topk = min(topk, confidence.shape[1])
    topk_indices = confidence.topk(topk, dim=1).indices

    selected_mask = torch.zeros_like(confidence, dtype=torch.bool)
    selected_mask.scatter_(1, topk_indices, True)

    distances = model.tue_head._calculate_distances_vectorized(
        score_captures=output["tue_info"]["score"],
        confidence_mask=selected_mask,
    )

    query_uncertainty = torch.nanmean(distances, dim=-1)
    tue_score = torch.nanmean(
        query_uncertainty.gather(1, topk_indices),
        dim=1,
    )
    confidence_score = (1.0 - confidence.gather(1, topk_indices)).mean(dim=1)

    return (
        torch.stack(
            [tue_score, confidence_score],
            dim=1,
        )
        .float()
        .cpu()
        .numpy()
    )


def benchmark_corruptions(
    model: torch.nn.Module,
    cfg,
    num_images: int | None,
    batch_size: int,
    topk: int,
    severity: int,
    num_workers: int,
    histogram_dir: str,
    histogram_bins: int,
):
    paths = list(dataset_paths(cfg))
    if num_images is not None:
        paths = paths[:num_images]

    corruption_names = list(get_corruption_names())
    stream = CorruptionStream(paths, corruption_names, severity)

    loader_kwargs = {
        "dataset": stream,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": False,
    }
    if num_workers > 0:
        loader_kwargs.update(
            persistent_workers=False,
            prefetch_factor=1,
        )

    loader = DataLoader(**loader_kwargs)

    tue_scores = {name: [] for name in ["clean", *corruption_names]}
    confidence_scores = {name: [] for name in ["clean", *corruption_names]}

    model.eval()
    torch.backends.cudnn.benchmark = True

    for names, images in tqdm(loader, desc="Benchmark"):
        batch_scores = score_tensor_batch(model, images, topk)

        for name, (tue_score, confidence_score) in zip(names, batch_scores):
            tue_scores[name].append(float(tue_score))
            confidence_scores[name].append(float(confidence_score))

    results = {}

    for corruption_name in corruption_names:
        labels = np.concatenate(
            [
                np.zeros(len(tue_scores["clean"]), dtype=np.int64),
                np.ones(len(tue_scores[corruption_name]), dtype=np.int64),
            ]
        )

        tue_values = np.asarray(
            tue_scores["clean"] + tue_scores[corruption_name],
            dtype=np.float32,
        )
        confidence_values = np.asarray(
            confidence_scores["clean"] + confidence_scores[corruption_name],
            dtype=np.float32,
        )

        tue_metrics = binary_uncertainty_metrics(labels, tue_values)
        confidence_metrics = binary_uncertainty_metrics(
            labels,
            confidence_values,
        )

        results[corruption_name] = {
            **{f"tue_{name}": value for name, value in tue_metrics.items()},
            **{
                f"confidence_{name}": value
                for name, value in confidence_metrics.items()
            },
        }

    metric_names = next(iter(results.values())).keys()
    results["mean"] = {
        metric_name: float(
            np.mean([results[name][metric_name] for name in corruption_names])
        )
        for metric_name in metric_names
    }

    save_histograms(
        tue_scores=tue_scores,
        confidence_scores=confidence_scores,
        corruption_names=corruption_names,
        output_dir=histogram_dir,
        bins=histogram_bins,
        severity=severity,
    )

    return results


def binary_uncertainty_metrics(
    labels: np.ndarray,
    uncertainty: np.ndarray,
):
    finite = np.isfinite(uncertainty)
    labels = labels[finite].astype(np.int64, copy=False)
    uncertainty = uncertainty[finite]

    if labels.size == 0 or np.unique(labels).size != 2:
        raise ValueError("Metrics require finite clean and corrupted scores")

    # Low uncertainty is accepted first; corruption is treated as risk/error.
    order = np.argsort(uncertainty, kind="stable")
    sorted_risk = labels[order].astype(np.float64)
    sample_count = np.arange(1, sorted_risk.size + 1, dtype=np.float64)
    risk = np.cumsum(sorted_risk) / sample_count
    aurc = risk.mean()

    optimal_risk = np.sort(labels).astype(np.float64)
    optimal_risk = np.cumsum(optimal_risk) / sample_count
    eaurc = aurc - optimal_risk.mean()

    fpr, tpr, _ = roc_curve(labels, uncertainty)
    fpr95 = np.interp(0.95, tpr, fpr)

    return {
        "auroc": float(roc_auc_score(labels, uncertainty)),
        "aupr_out": float(average_precision_score(labels, uncertainty)),
        "aupr_in": float(average_precision_score(1 - labels, -uncertainty)),
        "fpr95": float(fpr95),
        "aurc": float(aurc),
        "eaurc": float(eaurc),
    }


def save_histograms(
    tue_scores,
    confidence_scores,
    corruption_names,
    output_dir: str,
    bins: int,
    severity: int,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for corruption_name in corruption_names:
        clean_tue = np.asarray(tue_scores["clean"], dtype=np.float32)
        corrupt_tue = np.asarray(
            tue_scores[corruption_name],
            dtype=np.float32,
        )
        clean_confidence = np.asarray(
            confidence_scores["clean"],
            dtype=np.float32,
        )
        corrupt_confidence = np.asarray(
            confidence_scores[corruption_name],
            dtype=np.float32,
        )

        clean_tue = clean_tue[np.isfinite(clean_tue)]
        corrupt_tue = corrupt_tue[np.isfinite(corrupt_tue)]
        clean_confidence = clean_confidence[np.isfinite(clean_confidence)]
        corrupt_confidence = corrupt_confidence[np.isfinite(corrupt_confidence)]

        figure, axes = plt.subplots(1, 2, figsize=(10, 4))

        tue_edges = np.histogram_bin_edges(
            np.concatenate([clean_tue, corrupt_tue]),
            bins=bins,
        )
        axes[0].hist(
            clean_tue,
            bins=tue_edges,
            alpha=0.6,
            density=True,
            label="clean",
        )
        axes[0].hist(
            corrupt_tue,
            bins=tue_edges,
            alpha=0.6,
            density=True,
            label=corruption_name,
        )
        axes[0].set_title("TUE uncertainty")
        axes[0].set_xlabel("image score")
        axes[0].set_ylabel("density")
        axes[0].legend()

        confidence_edges = np.histogram_bin_edges(
            np.concatenate([clean_confidence, corrupt_confidence]),
            bins=bins,
        )
        axes[1].hist(
            clean_confidence,
            bins=confidence_edges,
            alpha=0.6,
            density=True,
            label="clean",
        )
        axes[1].hist(
            corrupt_confidence,
            bins=confidence_edges,
            alpha=0.6,
            density=True,
            label=corruption_name,
        )
        axes[1].set_title("1 - confidence")
        axes[1].set_xlabel("image score")
        axes[1].set_ylabel("density")
        axes[1].legend()

        figure.suptitle(f"Clean vs {corruption_name} (severity {severity})")
        figure.tight_layout()
        figure.savefig(
            output_path / f"{corruption_name}.png",
            dpi=300,
        )
        plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--ckpt", required=True)
    parser.add_argument("-n", "--num", type=int, default=None)
    parser.add_argument("-b", "--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--hist-dir", default="corruption_histograms")
    parser.add_argument("--hist-bins", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    model, cfg = build(args.config, args.ckpt)

    results = benchmark_corruptions(
        model=model,
        cfg=cfg,
        num_images=args.num,
        batch_size=args.batch,
        topk=args.topk,
        severity=args.severity,
        num_workers=args.workers,
        histogram_dir=args.hist_dir,
        histogram_bins=args.hist_bins,
    )

    print(
        f"{'corruption':<20}{'method':<9}"
        f"{'AUROC':>9}{'AUPR-O':>9}{'AUPR-I':>9}"
        f"{'FPR95':>9}{'AURC':>9}{'E-AURC':>9}"
    )
    for name, values in results.items():
        print(
            f"{name:<20}{'TUE':<9}"
            f"{values['tue_auroc']:>9.4f}"
            f"{values['tue_aupr_out']:>9.4f}"
            f"{values['tue_aupr_in']:>9.4f}"
            f"{values['tue_fpr95']:>9.4f}"
            f"{values['tue_aurc']:>9.4f}"
            f"{values['tue_eaurc']:>9.4f}"
        )
        print(
            f"{'':<20}{'1-conf':<9}"
            f"{values['confidence_auroc']:>9.4f}"
            f"{values['confidence_aupr_out']:>9.4f}"
            f"{values['confidence_aupr_in']:>9.4f}"
            f"{values['confidence_fpr95']:>9.4f}"
            f"{values['confidence_aurc']:>9.4f}"
            f"{values['confidence_eaurc']:>9.4f}"
        )


if __name__ == "__main__":
    main()
