"""
TUE OOD benchmark using ImageCorruptions severity=5.

pip install imagecorruptions scikit-learn matplotlib
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


import matplotlib.pyplot as plt
import numpy as np
import torch
from imagecorruptions import corrupt, get_corruption_names
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from tqdm import tqdm

from src.core import YAMLConfig

# ---------------------------------------------------------------------
# Project-specific hook
# ---------------------------------------------------------------------


def build_runtime(config: str, ckpt: str):
    cfg = YAMLConfig(str(config))

    model = cfg.model
    id_loader = cfg.val_dataloader

    checkpoint = torch.load(
        ckpt,
        map_location="cuda" if torch.cuda.is_available() else "cpu",
    )

    if "ema" in checkpoint:
        state_dict = checkpoint["ema"]
        if isinstance(state_dict, dict) and "module" in state_dict:
            state_dict = state_dict["module"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    return model, cfg, id_loader


def build_corrupted_loader(config_path, image_dir, ann_file):
    cfg = YAMLConfig(str(config_path))

    # Override only the validation dataset source.
    cfg.yaml_cfg["val_dataloader"]["dataset"]["img_folder"] = str(image_dir)
    cfg.yaml_cfg["val_dataloader"]["dataset"]["ann_file"] = str(ann_file)

    return cfg.val_dataloader


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------


def fpr95(y_true, uncertainty):
    fpr, tpr, _ = roc_curve(y_true, uncertainty)

    idx = np.where(tpr >= 0.95)[0]
    if len(idx) == 0:
        return 1.0

    return float(fpr[idx[0]])


def aurc_eaurc(y_true, uncertainty):
    """
    Selective OOD risk:
      error/risk = 1 for OOD
      error/risk = 0 for ID

    Low uncertainty samples are retained first.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    uncertainty = np.asarray(uncertainty)

    order = np.argsort(uncertainty)
    errors = y_true[order]

    n = len(errors)
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(errors) / np.arange(1, n + 1)

    aurc = np.trapezoid(risk, coverage)

    # Optimal ordering: all ID first, then OOD.
    optimal_errors = np.sort(y_true)

    optimal_risk = np.cumsum(optimal_errors) / np.arange(1, n + 1)

    optimal_aurc = np.trapezoid(optimal_risk, coverage)

    return float(aurc), float(aurc - optimal_aurc)


def calculate_metrics(id_values, ood_values):
    id_values = np.asarray(id_values)
    ood_values = np.asarray(ood_values)

    y = np.concatenate(
        [
            np.zeros(len(id_values)),
            np.ones(len(ood_values)),
        ]
    )

    u = np.concatenate([id_values, ood_values])

    auroc = roc_auc_score(y, u)

    # OOD is positive
    aupr_out = average_precision_score(y, u)

    # ID is positive; lower uncertainty means more ID-like
    aupr_in = average_precision_score(
        1 - y,
        -u,
    )

    aurc, eaurc = aurc_eaurc(y, u)

    return {
        "auroc": float(auroc),
        "aupr_out": float(aupr_out),
        "aupr_in": float(aupr_in),
        "fpr95": fpr95(y, u),
        "aurc": aurc,
        "eaurc": eaurc,
    }


# ---------------------------------------------------------------------
# Corruptions
# ---------------------------------------------------------------------


def corrupt_batch(images, corruption_name):
    """
    images: Tensor[B,C,H,W], expected in [0,1].
    """
    corrupted = []

    for image in images:
        x = image.detach().cpu().permute(1, 2, 0).numpy()

        x = np.clip(x * 255.0, 0, 255).astype(np.uint8)

        x = corrupt(
            x,
            corruption_name=corruption_name,
            severity=5,
        )

        x = torch.from_numpy(x.copy()).permute(2, 0, 1).float() / 255.0

        corrupted.append(x)

    return torch.stack(corrupted)


# ---------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------


@torch.no_grad()
def collect_uncertainties(model, images):
    results = model(images)

    tue = []
    confidence_uncertainty = []
    labels = []

    for result in results:
        if result["tu"].numel() == 0:
            continue

        tue.extend(result["tu"].detach().float().cpu().tolist())

        confidence_uncertainty.extend(
            (1.0 - result["scores"]).detach().float().cpu().tolist()
        )

        labels.extend(result["labels"].detach().cpu().tolist())

    return tue, confidence_uncertainty, labels


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------


def plot_histogram(id_tue, ood_tue, name, output_dir):
    plt.figure(figsize=(7, 4))

    plt.hist(
        id_tue,
        bins=50,
        density=True,
        alpha=0.5,
        label="ID",
    )

    plt.hist(
        ood_tue,
        bins=50,
        density=True,
        alpha=0.5,
        label=f"OOD ({name}, severity 5)",
    )

    plt.xlabel("Topological uncertainty")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / f"tue_{name}.png",
        dpi=200,
    )

    plt.close()


# ---------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------


@torch.no_grad()
def benchmark(args):
    model, cfg, id_loader = build_runtime(
        args.config,
        args.ckpt,
    )

    device = next(model.parameters()).device
    model.eval()

    output_dir = Path("tue_ood_results")
    output_dir.mkdir(exist_ok=True)

    corruptions = set(get_corruption_names())

    corruptions -= set(["gaussian_noise", "shot_noise", "impulse_noise", "pixelate"])

    corruptions = sorted(corruptions)

    print(
        f"{'Corruption':<20}{'Method':<9}{'Class':<8}"
        f"{'AUROC':>9}{'AUPR-O':>9}{'AUPR-I':>9}"
        f"{'FPR95':>9}{'AURC':>9}{'E-AURC':>9}"
        f"{'ID':>10}{'OOD':>10}"
    )

    def print_metrics_row(corruption, method, cls, metrics, n_id, n_ood):
        metric_keys = (
            "auroc",
            "aupr_out",
            "aupr_in",
            "fpr95",
            "aurc",
            "eaurc",
        )

        metric_columns = "".join(f"{metrics[key]:>9.4f}" for key in metric_keys)

        print(
            f"{corruption:<20}{method:<9}{cls!s:<8}{metric_columns}{n_id:>10}{n_ood:>10}"
        )

    # ---------------------------------------------------------
    # ID once
    # ---------------------------------------------------------

    id_tue = []
    id_conf = []
    id_tue_by_class = {}
    id_conf_by_class = {}

    seen = 0

    for samples, _ in tqdm(id_loader, desc="ID Val Loader"):
        if args.num is not None and seen >= args.num:
            break

        samples = samples.to(device)

        if args.num is not None:
            remaining = args.num - seen
            samples = samples[:remaining]

        seen += len(samples)

        tue, conf, labels = collect_uncertainties(model, samples)

        id_tue.extend(tue)
        id_conf.extend(conf)

        for value, cls in zip(tue, labels):
            id_tue_by_class.setdefault(cls, []).append(value)

        for value, cls in zip(conf, labels):
            id_conf_by_class.setdefault(cls, []).append(value)

    # ---------------------------------------------------------
    # Pre-generated corrupted sets
    # ---------------------------------------------------------

    for name in corruptions:
        corruption_dir = Path(args.corrupted_root) / name
        ood_loader = build_corrupted_loader(
            args.config,
            corruption_dir / "images",
            corruption_dir / "annotations.json",
        )

        ood_tue = []
        ood_conf = []
        ood_tue_by_class = {}
        ood_conf_by_class = {}

        seen = 0

        for samples, _ in tqdm(ood_loader, desc=f"OOD Dataloader: {name}"):
            if args.num is not None and seen >= args.num:
                break

            samples = samples.to(device)

            if args.num is not None:
                remaining = args.num - seen
                samples = samples[:remaining]

            seen += len(samples)

            tue, conf, labels = collect_uncertainties(
                model,
                samples,
            )

            ood_tue.extend(tue)
            ood_conf.extend(conf)

            for value, cls in zip(tue, labels):
                ood_tue_by_class.setdefault(cls, []).append(value)

            for value, cls in zip(conf, labels):
                ood_conf_by_class.setdefault(cls, []).append(value)

        if not ood_tue:
            print(f"{name:<20} skipped: no valid ood detections")
            continue

        if not id_tue:
            print(f"{name:<20} skipped: no valid id detections")
            continue

        tue_metrics = calculate_metrics(
            id_tue,
            ood_tue,
        )

        confidence_metrics = calculate_metrics(
            id_conf,
            ood_conf,
        )

        # Aggregate results.
        print_metrics_row(
            name,
            "TUE",
            "all",
            tue_metrics,
            len(id_tue),
            len(ood_tue),
        )

        print_metrics_row(
            "",
            "1-conf",
            "all",
            confidence_metrics,
            len(id_conf),
            len(ood_conf),
        )

        # Per-class TUE results.
        shared_classes = id_tue_by_class.keys() & ood_tue_by_class.keys()

        for cls in sorted(shared_classes):
            class_id_scores_tue = id_tue_by_class[cls]
            class_ood_scores_tue = ood_tue_by_class[cls]

            class_id_scores_conf = id_conf_by_class[cls]
            class_ood_scores_conf = ood_conf_by_class[cls]

            n_id = len(class_id_scores_tue)
            n_ood = len(class_ood_scores_tue)

            class_metrics_tue = calculate_metrics(
                class_id_scores_tue,
                class_ood_scores_tue,
            )

            class_metrics_conf = calculate_metrics(
                class_id_scores_conf,
                class_ood_scores_conf,
            )

            print_metrics_row(
                "",
                "TUE",
                cls,
                class_metrics_tue,
                n_id,
                n_ood,
            )

            print_metrics_row(
                "",
                "CONF",
                cls,
                class_metrics_conf,
                n_id,
                n_ood,
            )

        plot_histogram(
            id_tue,
            ood_tue,
            name,
            output_dir,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--ckpt", required=True)
    parser.add_argument("-n", "--num", type=int, default=None)
    parser.add_argument("-b", "--batch", type=int, default=32)
    parser.add_argument(
        "--corrupted-root",
        default="dataset/waymo_corrupted",
        type=Path,
        help="Root containing one COCO dataset per corruption.",
    )

    args = parser.parse_args()

    benchmark(args)


if __name__ == "__main__":
    main()
