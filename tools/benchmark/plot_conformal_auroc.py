"""Prototype: does per-(layer,class) percentile normalization beat raw uncertainty?
Run from the RT-DETR repo root. No inference changes, no calibration file needed.

Build per-(layer,class) ID reference distance distributions from a held-out half
of the ID set, convert each detection's per-layer distances to percentiles against
them, average across layers -> a normalized OOD score. Compare its ID-vs-OOD AUROC
to the raw tue_uncertainty AUROC, on the SAME detections. If the percentile score
wins, conformal standardization is worth wiring into inference; if flat, the
bottleneck is representation, not scaling."""

import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from pycocotools.coco import COCO
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.core import YAMLConfig

CONF_TH = 0.5
SIZE = 640
BATCH = 32
ID_CALIB_FRAC = 0.5  # fraction of ID images used to build the reference
MIN_REF = 10  # skip (layer,class) references smaller than this
SEED = 1234
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
    """Return a list over images; each image is a list of detections
    (raw_unc, [(layer_col, class, distance), ...])  for conf>=0.5 finite dets."""
    coco = COCO(d["anns"])
    paths = [
        os.path.join(d["imgs"], im["file_name"])
        for im in coco.loadImgs(coco.getImgIds())
    ]
    per_image = []
    for b in tqdm(list(chunks(paths, BATCH)), desc=d["name"]):
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in b]).to(DEVICE)
        out = model(x)
        conf = out["pred_logits"].sigmoid().max(-1).values
        unc = out["tue_uncertainty"]
        td = out["tue_distances"].cpu()  # [B, Q, L]
        tc = out["tue_classes"].cpu()  # [B, Q, L]
        sel = (conf >= CONF_TH) & torch.isfinite(unc) & torch.isfinite(conf)
        for i in range(conf.shape[0]):
            img_dets = []
            for q in torch.where(sel[i])[0].tolist():
                cols = []
                for col in range(td.shape[2]):
                    dist = float(td[i, q, col])
                    cls = int(tc[i, q, col])
                    if cls >= 0 and math.isfinite(dist):
                        cols.append((col, cls, dist))
                if cols:
                    img_dets.append((float(unc[i, q]), cols))
            per_image.append(img_dets)
    return per_image


def build_reference(images):
    ref = defaultdict(list)
    for img in images:
        for _, cols in img:
            for col, cls, dist in cols:
                ref[(col, cls)].append(dist)
    return {k: np.sort(np.asarray(v)) for k, v in ref.items() if len(v) >= MIN_REF}


def pct_score(det, ref):
    """Mean per-layer percentile of the detection's distances vs ID reference.
    None if no layer has a usable reference."""
    ps = []
    for col, cls, dist in det[1]:
        arr = ref.get((col, cls))
        if arr is not None:
            ps.append(np.searchsorted(arr, dist, side="right") / len(arr))
    return float(np.mean(ps)) if ps else None


def scored(images, ref):
    """(raw_unc, pct) pairs for dets that have a percentile score."""
    out = []
    for img in images:
        for det in img:
            p = pct_score(det, ref)
            if p is not None:
                out.append((det[0], p))
    return out


def auroc(id_s, ood_s):  # higher score = more OOD
    if not id_s or not ood_s:
        return float("nan")
    lab = [0] * len(id_s) + [1] * len(ood_s)
    return roc_auc_score(lab, id_s + ood_s)


def main():
    model, datasets = load_model()
    rng = random.Random(SEED)

    # ID: gather images, split calib (reference) vs eval (scored ID side)
    id_calib, id_eval = [], []
    for ds in datasets:
        d = ds["dataset"]
        if d["ood"]:
            continue
        imgs = collect(model, d)
        for img in imgs:
            (id_calib if rng.random() < ID_CALIB_FRAC else id_eval).append(img)

    assert id_calib and id_eval, "need ID images for reference and evaluation"
    ref = build_reference(id_calib)
    print(f"reference cells (layer,class) with n>={MIN_REF}: {len(ref)}")

    id_eval_s = scored(id_eval, ref)
    id_raw = [s[0] for s in id_eval_s]
    id_pct = [s[1] for s in id_eval_s]

    print(
        f"\n{'OOD dataset':32s} {'n_id':>6} {'n_ood':>6} {'AUROC_raw':>10} {'AUROC_pct':>10} {'delta':>7}"
    )
    for ds in datasets:
        d = ds["dataset"]
        if not d["ood"]:
            continue
        ood_s = scored(collect(model, d), ref)
        if not ood_s:
            print(f"{d['name'][:32]:32s} {'--':>6} {'--':>6}  (no scored dets)")
            continue
        ood_raw = [s[0] for s in ood_s]
        ood_pct = [s[1] for s in ood_s]
        a_raw = auroc(id_raw, ood_raw)
        a_pct = auroc(id_pct, ood_pct)
        print(
            f"{d['name'][:32]:32s} {len(id_raw):>6} {len(ood_raw):>6} "
            f"{a_raw:>10.4f} {a_pct:>10.4f} {a_pct - a_raw:>+7.4f}"
        )


if __name__ == "__main__":
    main()
