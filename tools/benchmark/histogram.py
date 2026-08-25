"""Overlaid ID-vs-OOD histograms of per-detection uncertainty and confidence,
in the style of the TU / confidence distribution figures.
Run from the RT-DETR repo root. Reuses the benchmark's model + per-query scores.
Only conf>=CONF_TH, finite-uncertainty detections are used (uncertainty's domain).
Bars are per-group normalized (fraction of that dataset's detections per bin)."""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_FILE = "output/auroc/exp4/exp4_config.yaml"
ID_COLOR = "blue"
OOD_COLORS = ["green", "red", "darkorange", "purple", "brown", "teal", "magenta"]

tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor()])


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def load_model():
    with open(CONFIG_FILE) as f:
        cfg_d = yaml.safe_load(f)
    cfg = YAMLConfig(cfg_d["config"], resume=cfg_d["checkpoint"])
    c = torch.load(cfg_d["checkpoint"], map_location="cpu")
    state = c.get("ema", {}).get("module") or c.get("model") or c
    cfg.model.load_state_dict(state)
    return cfg.model.to(DEVICE).eval(), cfg_d["datasets"]


@torch.no_grad()
def scores(model, d):
    coco = COCO(d["anns"])
    paths = [
        os.path.join(d["imgs"], im["file_name"])
        for im in coco.loadImgs(coco.getImgIds())
    ]
    U, C = [], []
    for b in tqdm(list(chunks(paths, BATCH)), desc=d["name"]):
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in b]).to(DEVICE)
        out = model(x)
        conf = out["pred_logits"].sigmoid().max(-1).values
        unc = out["tue_uncertainty"]
        m = (conf >= CONF_TH) & torch.isfinite(unc) & torch.isfinite(conf)
        U += unc[m].cpu().tolist()
        C += conf[m].cpu().tolist()
    return U, C


def phist(ax, groups, key, bins):
    for name, color, vals in groups:
        v = np.asarray(vals[key])
        if v.size == 0:
            continue
        ax.hist(
            v,
            bins=bins,
            weights=np.ones(v.size) / v.size,
            color=color,
            alpha=0.6,
            label=name,
        )


def main():
    model, datasets = load_model()
    groups, oi = [], 0
    for ds in datasets:
        d = ds["dataset"]
        U, C = scores(model, d)
        if d["ood"]:
            color = OOD_COLORS[oi % len(OOD_COLORS)]
            oi += 1
            tag = "OOD"
        else:
            color = ID_COLOR
            tag = "ID/Train"
        groups.append((f"{d['name']} ({tag})", color, {"U": U, "C": C}))

    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.5, 2.8))
    allU = np.concatenate([np.asarray(g[2]["U"]) for g in groups if g[2]["U"]])
    ub = np.linspace(allU.min(), allU.max(), 31)
    cb = np.linspace(CONF_TH, 1.0, 31)
    phist(a1, groups, "U", ub)
    phist(a2, groups, "C", cb)
    a1.set_xlabel("(a) Distrib. of Uncertainty")
    a2.set_xlabel("(b) Distrib. of Confidences")
    a2.set_xlim(CONF_TH, 1.0)
    for ax in (a1, a2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    out = Path(CONFIG_FILE).parent / "uncertainty_hist.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    main()
