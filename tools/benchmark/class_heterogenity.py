"""10-minute check: how much does per-predicted-class ID uncertainty vary?
Run from the RT-DETR repo root. Uses the config's ID (ood:false) dataset(s).

If the per-class means are spread wide relative to the within-class spread
(between/within ratio >> 1), that heterogeneity is what a per-(layer,class)
conformal p-value normalizes away -> expect a real AUROC jump. If the ratio is
small, per-class scaling has little to remove -> conformal unlikely to help, and
the bottleneck is representation, not scaling."""

import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

from src.core import YAMLConfig

CONF_TH = 0.5
SIZE = 640
BATCH = 32
MIN_N = 20  # ignore classes with fewer detections in the summary stats
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_FILE = "output/auroc/exp9/exp9_config.yaml"

tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor()])


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def load_model():
    with open(CONFIG_FILE) as f:
        cfg_d = yaml.safe_load(f)
    cfg = YAMLConfig(cfg_d["config"], resume=cfg_d["checkpoint"])
    c = torch.load(cfg_d["checkpoint"], map_location="cpu")
    state = c.get("ema", {}).get("module")
    assert state is not None, "checkpoint has no ema.module"
    cfg.model.load_state_dict(state)
    return cfg.model.to(DEVICE).eval(), cfg_d["datasets"]


@torch.no_grad()
def collect(model, d):
    """Return per-detection (uncertainty, predicted class) over conf>=0.5 dets."""
    coco = COCO(d["anns"])
    paths = [
        os.path.join(d["imgs"], im["file_name"])
        for im in coco.loadImgs(coco.getImgIds())
    ]
    U, L = [], []
    for b in tqdm(list(chunks(paths, BATCH)), desc=d["name"]):
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in b]).to(DEVICE)
        out = model(x)
        conf = out["pred_logits"].sigmoid().max(-1).values
        lab = out["pred_logits"].argmax(-1)
        unc = out["tue_uncertainty"]
        m = (conf >= CONF_TH) & torch.isfinite(unc) & torch.isfinite(conf)
        U += unc[m].cpu().tolist()
        L += lab[m].cpu().tolist()
    return np.asarray(U), np.asarray(L, dtype=int)


def main():
    model, datasets = load_model()
    U, L = [], []
    for ds in datasets:
        d = ds["dataset"]
        if d["ood"]:
            continue  # ID only
        u, l = collect(model, d)
        U.append(u)
        L.append(l)
    U = np.concatenate(U)
    L = np.concatenate(L)

    rows = []
    for c in sorted(set(L.tolist())):
        v = U[L == c]
        rows.append((c, v.size, float(v.mean()), float(v.std())))
    rows.sort(key=lambda r: r[2])

    out_csv = Path(CONFIG_FILE).parent / "class_uncertainty_stats.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "n", "mean_unc", "std_unc"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.5f}", f"{r[3]:.5f}"])

    # summary over classes with enough samples
    big = [r for r in rows if r[1] >= MIN_N]
    means = np.array([r[2] for r in big])
    stds = np.array([r[3] for r in big])
    counts = np.array([r[1] for r in big])
    between = means.std()  # spread of per-class means
    within = float(
        np.sqrt(np.sum(counts * stds**2) / counts.sum())
    )  # pooled within-class std
    ratio = between / within if within > 0 else float("nan")

    print(f"\nID detections: {U.size}, classes with n>={MIN_N}: {len(big)}")
    print(f"pooled mean unc      : {U.mean():.4f}")
    print(f"per-class mean range : [{means.min():.4f}, {means.max():.4f}]")
    print(f"between-class std    : {between:.4f}  (spread of per-class means)")
    print(f"within-class std     : {within:.4f}  (pooled)")
    print(f"between/within ratio : {ratio:.2f}")
    if ratio >= 1.0:
        print("=> high heterogeneity: per-class scaling (conformal) likely to help.")
    elif ratio >= 0.5:
        print("=> moderate heterogeneity: conformal may give a modest gain.")
    else:
        print(
            "=> low heterogeneity: little to normalize; bottleneck is likely representation."
        )
    print("saved:", out_csv)


if __name__ == "__main__":
    main()
