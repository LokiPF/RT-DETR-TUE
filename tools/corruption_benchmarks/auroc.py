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

from src.core import YAMLConfig

# ---------------------------------------------------------------------
# Project-specific hook
# ---------------------------------------------------------------------


def build_runtime(config: str, ckpt: str, batch_size: int):
    """
    Return:
        model        : model with TUE enabled
        data_loader  : ID evaluation loader

    The model/postprocessing path is assumed to return one result per image:

        [
            {
                "scores": Tensor[K],
                "tu":     Tensor[K],
                ...
            },
            ...
        ]

    Replace this function with your normal RT-DETR loading code.
    """
    cfg = YAMLConfig(str(config))
    model = cfg.model
    dataloader = cfg.val_dataloader

    checkpoint = torch.load(
        ckpt, map_location="cuda" if torch.cuda.is_available() else "cpu"
    )
    # RT-DETR checkpoints commonly store weights under "ema" or "model"
    if "ema" in checkpoint:
        state_dict = checkpoint["ema"]
        # Some checkpoints wrap EMA weights again
        if isinstance(state_dict, dict) and "module" in state_dict:
            state_dict = state_dict["module"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)

    return model, dataloader


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

    for result in results:
        if result["tu"].numel() == 0:
            continue

        tue.extend(result["tu"].detach().float().cpu().tolist())

        confidence_uncertainty.extend(
            (1.0 - result["scores"]).detach().float().cpu().tolist()
        )
    return tue, confidence_uncertainty


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
    model, data_loader = build_runtime(
        args.config,
        args.ckpt,
        args.batch,
    )

    device = next(model.parameters()).device
    model.eval()

    output_dir = Path("tue_ood_results")
    output_dir.mkdir(exist_ok=True)

    corruptions = get_corruption_names()

    print(
        f"{'Corruption':<20}{'Method':<9}"
        f"{'AUROC':>9}"
        f"{'AUPR-O':>9}"
        f"{'AUPR-I':>9}"
        f"{'FPR95':>9}"
        f"{'AURC':>9}"
        f"{'E-AURC':>9}"
        f"{'ID':>12}"
        f"{'OOD':>9}"
    )

    for name in corruptions:
        id_tue = []
        id_conf = []

        ood_tue = []
        ood_conf = []

        total_id = 0
        total_ood = 0

        seen = 0

        for samples, _ in data_loader:
            if args.num is not None and seen >= args.num:
                break

            samples = samples.to(device)

            if args.num is not None:
                remaining = args.num - seen
                samples = samples[:remaining]

            seen += len(samples)

            # ID
            tue, conf = collect_uncertainties(model, samples)
            id_tue.extend(tue)
            id_conf.extend(conf)
            total_id += len(tue)

            # OOD: same images, severity-5 corruption
            corrupted = corrupt_batch(
                samples,
                name,
            ).to(device)

            tue, conf = collect_uncertainties(
                model,
                corrupted,
            )
            ood_tue.extend(tue)
            ood_conf.extend(conf)
            total_ood += len(tue)

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

        values = {
            **{f"tue_{k}": v for k, v in tue_metrics.items()},
            **{f"confidence_{k}": v for k, v in confidence_metrics.items()},
            "valid_id": len(id_tue),
            "total_id": total_id,
            "valid_ood": len(ood_tue),
            "total_ood": total_ood,
        }

        print(
            f"{name:<20}{'TUE':<9}"
            f"{values['tue_auroc']:>9.4f}"
            f"{values['tue_aupr_out']:>9.4f}"
            f"{values['tue_aupr_in']:>9.4f}"
            f"{values['tue_fpr95']:>9.4f}"
            f"{values['tue_aurc']:>9.4f}"
            f"{values['tue_eaurc']:>9.4f}"
            f"{values['valid_id']:>7}/{values['total_id']:<4}"
            f"{values['valid_ood']:>4}/{values['total_ood']:<4}"
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

    args = parser.parse_args()

    benchmark(args)


if __name__ == "__main__":
    main()
