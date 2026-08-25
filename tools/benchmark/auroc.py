"""OOD AUROC + COCO mAP/mAR benchmark for RT-DETRv2 with per-query uncertainty.
Run from the RT-DETRv2 repo root (so `from src.core import YAMLConfig` resolves).
AUROC: all ID datasets form the shared negative pool; each OOD dataset is scored
against it (positive=OOD), for uncertainty (higher=OOD) and 1-confidence, at
per-detection and per-image (mean) level. Each variant is reported twice: over
ALL conf>=0.5 detections, and over CORRECT-only detections (matched to GT by
IoU>=IOU_TH and class). COCO 12-stat is computed per dataset via the repo
postprocessor + pycocotools, on the same conf>=0.5 detections."""

import pathlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import csv
import os
import shutil

import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.metrics import roc_auc_score
from torchvision.ops import box_convert
from tqdm import tqdm

from src.core import YAMLConfig

CONF_TH = 0.5
IOU_TH = 0.5
CLASS_MATCH = True
SIZE = 640
BATCH = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_FILE = "output/auroc/exp4/exp4_config.yaml"
COCO_STATS = [
    "AP",
    "AP50",
    "AP75",
    "AP_s",
    "AP_m",
    "AP_l",
    "AR1",
    "AR10",
    "AR100",
    "AR_s",
    "AR_m",
    "AR_l",
]

with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

datasets = config["datasets"]
model_config = config["config"]
ckpt = config["checkpoint"]
output_dir = pathlib.Path(CONFIG_FILE).parent.resolve()
config_filename = pathlib.Path(CONFIG_FILE).name
shutil.copy2(CONFIG_FILE, os.path.join(output_dir, config_filename))


def load_model():
    cfg = YAMLConfig(model_config, resume=ckpt)
    ckpt_ = torch.load(ckpt, map_location="cpu")
    state = ckpt_.get("ema", {}).get("module") or ckpt_.get("model") or ckpt_
    cfg.model.load_state_dict(state)
    return cfg.model.to(DEVICE).eval(), cfg.postprocessor.to(DEVICE).eval()


tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor()])


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(dets, gts):
    """dets: list of (conf, box_xyxy, labelel) sorted by conf desc; gts: (labelel, box).
    Greedy IoU (+class) matching -> list of correct flags aligned to dets."""
    used = [False] * len(gts)
    flags = []
    for _, box, labelel in dets:
        best, bi = IOU_TH, -1
        for j, (glabel, gbox) in enumerate(gts):
            if used[j] or (CLASS_MATCH and glabel != labelel):
                continue
            v = iou(box, gbox)
            if v >= best:
                best, bi = v, j
        if bi >= 0:
            used[bi] = True
        flags.append(bi >= 0)
    return flags


@torch.no_grad()
def batch_eval(model, post_processor, items, gt):
    """items: (path, img_id, w, h). Return per-image [per](conf, unc, correct, dropped)
    lists for AUROC and a flat list of COCO detection dicts [dets](score>=CONF_TH)."""
    x = torch.stack([tf(Image.open(p).convert("RGB")) for p, _, _, _ in items]).to(
        DEVICE
    )
    out = model(x)
    conf_q = out["pred_logits"].sigmoid().max(-1).values  # [B, Q]
    label_q = out["pred_logits"].argmax(-1)  # [B, Q] contiguous
    unc_q = out["tue_uncertainty"]  # [B, Q]
    xyxy = box_convert(out["pred_boxes"], "cxcywh", "xyxy")  # [B, Q, 4] normalized
    sel = conf_q >= CONF_TH
    m = sel & torch.isfinite(unc_q) & torch.isfinite(conf_q)
    sizes = torch.tensor([[w, h] for _, _, w, h in items], device=DEVICE)
    res = post_processor(out, sizes)  # boxes xyxy in orig px
    per_image_stats, dets = [], []
    for b, (_, img_id, w, h) in enumerate(items):
        idx = torch.where(m[b])[0]
        c = conf_q[b][idx]
        u = unc_q[b][idx]
        l = label_q[b][idx]
        bx = xyxy[b][idx].clone()
        bx[:, 0::2] *= w
        bx[:, 1::2] *= h
        o = c.argsort(descending=True)
        di = [(float(c[i]), bx[i].tolist(), int(l[i])) for i in o]
        flags = match(di, gt[img_id])
        per_image_stats.append(
            (
                [d[0] for d in di],
                [float(u[i]) for i in o],
                flags,
                int((sel[b] & ~torch.isfinite(unc_q[b])).sum()),
            )
        )
        r = res[b]
        keep = r["scores"] >= CONF_TH
        for (x0, y0, x1, y1), label, s in zip(
            r["boxes"][keep].cpu().tolist(),
            r["labels"][keep].cpu().tolist(),
            r["scores"][keep].cpu().tolist(),
        ):
            dets.append(
                {
                    "image_id": img_id,
                    "category_id": int(label),
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "score": float(s),
                }
            )
    return per_image_stats, dets


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def coco_eval(coco, dets):
    if not dets:
        return [float("nan")] * 12
    E = COCOeval(coco, coco.loadRes(dets), "bbox")
    E.evaluate()
    E.accumulate()
    E.summarize()
    return list(E.stats)


def mean(a):
    return sum(a) / len(a)


def scan(model, post, d):
    """Return pooled score dict (all + correct-only) and COCO 12-stat."""
    coco = COCO(d["anns"])
    imgs = coco.loadImgs(coco.getImgIds())
    cat2label = {
        c: i for i, c in enumerate(sorted(coco.getCatIds()))
    }  # coco cat id -> contiguous labelel
    gt = {
        im["id"]: [
            (
                cat2label[a["category_id"]],
                [
                    a["bbox"][0],
                    a["bbox"][1],
                    a["bbox"][0] + a["bbox"][2],
                    a["bbox"][1] + a["bbox"][3],
                ],
            )
            for a in coco.loadAnns(coco.getAnnIds(imgIds=im["id"]))
        ]
        for im in imgs
    }
    items = [
        (os.path.join(d["imgs"], im["file_name"]), im["id"], im["width"], im["height"])
        for im in imgs
    ]
    p = {k: [] for k in ["dc", "du", "ic", "iu", "dcc", "duc", "icc", "iuc"]}
    dets = []
    n_drop = n_empty = 0
    for batch in tqdm(list(chunks(items, BATCH)), desc=d["name"]):
        per_image_stats, batch_dets = batch_eval(model, post, batch, gt)
        dets += batch_dets
        for conf, unc, cor, dropped in per_image_stats:
            n_drop += dropped
            if not conf:
                n_empty += 1
                continue
            p["dc"] += conf
            p["du"] += unc
            p["ic"].append(mean(conf))
            p["iu"].append(mean(unc))
            cc = [x for x, k in zip(conf, cor) if k]
            cu = [x for x, k in zip(unc, cor) if k]
            if cc:
                p["dcc"] += cc
                p["duc"] += cu
                p["icc"].append(mean(cc))
                p["iuc"].append(mean(cu))
    print(
        f"[{'OOD' if d['ood'] else 'ID '}] {d['name']}: {len(p['ic'])} imgs w/ dets, "
        f"{n_empty} empty, {len(p['dc'])} dets ({len(p['dcc'])} correct), "
        f"{n_drop} non-finite unc dropped"
    )
    return p, coco_eval(coco, dets)


def auroc(id_dets, ood_dets, higher_is_ood):
    if not id_dets or not ood_dets:
        return float("nan")
    if not higher_is_ood:  # confidence: lower => more OOD
        id_dets = [1 - x for x in id_dets]
        ood_dets = [1 - x for x in ood_dets]
    label = [0] * len(id_dets) + [1] * len(ood_dets)
    return roc_auc_score(
        label, id_dets + ood_dets
    )  # calculates auroc based on id and ood labels


def main():
    model, post = load_model()
    res = {}
    idp = {
        k: [] for k in ["dc", "du", "ic", "iu", "dcc", "duc", "icc", "iuc"]
    }  # shared ID pool
    for ds in datasets:
        d = ds["dataset"]
        p, stats = scan(model, post, d)
        res[d["name"]] = {"ood": d["ood"], "stats": stats, "p": p}
        if not d["ood"]:
            for k in idp:
                idp[k] += p[k]

    assert idp["dc"], "no ID detections found"
    assert any(v["ood"] for v in res.values()), "no OOD datasets found"

    # (column, id-neg key, ood-pos key, higher_is_ood)
    variants = [
        ("auroc_det_unc", "du", "du", True),
        ("auroc_det_1conf", "dc", "dc", False),
        ("auroc_img_unc", "iu", "iu", True),
        ("auroc_img_1conf", "ic", "ic", False),
        ("auroc_detC_unc", "duc", "duc", True),
        ("auroc_detC_1conf", "dcc", "dcc", False),
        ("auroc_imgC_unc", "iuc", "iuc", True),
        ("auroc_imgC_1conf", "icc", "icc", False),
    ]
    out_csv = output_dir / "benchmark.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["dataset", "split", "n_img", "n_det", "n_det_correct"]
            + COCO_STATS
            + [c for c, *_ in variants]
        )
        for name, v in res.items():
            p = v["p"]
            a = (
                [f"{auroc(idp[nk], p[pk], hi):.4f}" for _, nk, pk, hi in variants]
                if v["ood"]
                else [""] * len(variants)
            )
            w.writerow(
                [
                    name,
                    "OOD" if v["ood"] else "ID",
                    len(p["ic"]),
                    len(p["dc"]),
                    len(p["dcc"]),
                ]
                + [f"{s:.4f}" for s in v["stats"]]
                + a
            )
    print("saved:", out_csv)


if __name__ == "__main__":
    main()
