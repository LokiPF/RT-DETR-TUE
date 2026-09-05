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
CONFIG_FILE = "output/auroc/exp6/exp6_config.yaml"
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
    """Per-detection (U, C) and per-image mean (IU, IC) over conf>=CONF_TH dets."""
    coco = COCO(d["anns"])
    paths = [
        os.path.join(d["imgs"], im["file_name"])
        for im in coco.loadImgs(coco.getImgIds())
    ]
    s = {"U": [], "C": [], "IU": [], "IC": []}
    for b in tqdm(list(chunks(paths, BATCH)), desc=d["name"]):
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in b]).to(DEVICE)
        out = model(x)
        conf = out["pred_logits"].sigmoid().max(-1).values
        unc = out["tue_uncertainty"]
        m = (conf >= CONF_TH) & torch.isfinite(unc) & torch.isfinite(conf)
        for i in range(conf.shape[0]):
            cu = unc[i][m[i]].cpu()
            cc = conf[i][m[i]].cpu()
            if cu.numel() == 0:
                continue
            s["U"] += cu.tolist()
            s["C"] += cc.tolist()
            s["IU"].append(cu.mean().item())
            s["IC"].append(cc.mean().item())
    return s


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
        s = scores(model, d)
        if d["ood"]:
            color = OOD_COLORS[oi % len(OOD_COLORS)]
            oi += 1
            tag = "OOD"
        else:
            color = ID_COLOR
            tag = "ID/Train"
        groups.append((f"{d['name']} ({tag})", color, s))

    def bins(key):
        allv = np.concatenate([np.asarray(g[2][key]) for g in groups if g[2][key]])
        return np.linspace(allv.min(), allv.max(), 31)

    cb = np.linspace(CONF_TH, 1.0, 31)
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, ax = plt.subplots(2, 2, figsize=(7.5, 5.2))
    phist(ax[0, 0], groups, "U", bins("U"))
    phist(ax[0, 1], groups, "C", cb)
    phist(ax[1, 0], groups, "IU", bins("IU"))
    phist(ax[1, 1], groups, "IC", cb)
    ax[0, 0].set_xlabel("(a) Uncertainty (per detection)")
    ax[0, 1].set_xlabel("(b) Confidence (per detection)")
    ax[1, 0].set_xlabel("(c) Uncertainty (image avg)")
    ax[1, 1].set_xlabel("(d) Confidence (image avg)")
    ax[0, 1].set_xlim(CONF_TH, 1.0)
    ax[1, 1].set_xlim(CONF_TH, 1.0)
    for a in ax.ravel():
        a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout()
    out = Path(CONFIG_FILE).parent / "uncertainty_hist.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    main()
