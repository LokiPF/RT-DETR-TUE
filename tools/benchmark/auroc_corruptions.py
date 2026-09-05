"""OOD AUROC + COCO mAP/mAR benchmark for RT-DETRv2 under image corruptions.
Run from the RT-DETRv2 repo root (so `from src.core import YAMLConfig` resolves).

OOD is synthetic: the clean ID image set is corrupted with each of the 15 common
corruptions from bethgelab/imagecorruptions at severities 1-5. For each
(corruption, severity) the corrupted detections (positive=OOD) are scored against
the clean detections of the SAME images (negative=ID), giving AUROC for
uncertainty (higher=OOD) and 1-confidence, per-detection and per-image (mean),
over ALL conf>=0.5 detections and over CORRECT-only (GT-matched) detections.
COCO 12-stat is reported per corruption (annotations are unchanged by corruption).

    pip install imagecorruptions

Heavy: 15x5 passes over the ID set, each corrupting every image on CPU. Point the
config's ID dataset at a small subsplit (see make_train_subsplit.py) for speed."""

import csv
import os
import pathlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# --- compat shim: imagecorruptions predates skimage's multichannel->channel_axis rename ---
import imagecorruptions.corruptions as _ic
import numpy as np

for _n, _t in {
    "float": float,
    "int": int,
    "bool": bool,
    "object": object,
    "float_": np.float64,
    "complex_": np.complex128,
    "int_": np.int64,
    "unicode_": np.str_,
    "str_": np.str_,
}.items():
    if not hasattr(np, _n):
        setattr(np, _n, _t)
import torch
import torchvision.transforms as T
import yaml
from imagecorruptions import corrupt, get_corruption_names
from skimage.filters import gaussian as _sk_gaussian


def _gaussian_compat(img, *args, **kwargs):
    if kwargs.pop("multichannel", False):
        kwargs.setdefault("channel_axis", -1)
    return _sk_gaussian(img, *args, **kwargs)


_ic.gaussian = _gaussian_compat

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
SEVERITIES = [5]  # 1, 2, 3, 4,
SLOW = {"glass_blur", "zoom_blur"}
CORRUPTIONS = [c for c in get_corruption_names("common") if c not in SLOW]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_FILE = "output/auroc/exp9/exp9_config.yaml"
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

with open(CONFIG_FILE) as f:
    config = yaml.safe_load(f)

datasets = config["datasets"]
model_config = config["config"]
ckpt = config["checkpoint"]
output_dir = pathlib.Path(CONFIG_FILE).parent.resolve()


def load_model():
    cfg = YAMLConfig(model_config, resume=ckpt)
    c = torch.load(ckpt, map_location="cpu")
    state = c.get("ema", {}).get("module")
    assert state is not None, (
        "checkpoint has no ema.module (was EMA enabled at fit time?)"
    )
    cfg.model.load_state_dict(state)
    return cfg.model.to(DEVICE).eval(), cfg.postprocessor.to(DEVICE).eval()


tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor()])


def make_corrupt(name, severity):
    def f(pil):
        arr = np.asarray(pil, dtype=np.uint8)
        return Image.fromarray(corrupt(arr, corruption_name=name, severity=severity))

    return f


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(dets, gts):
    """dets: (conf, box_xyxy, label) sorted by conf desc; gts: (label, box).
    Greedy IoU (+class) matching -> list of correct flags aligned to dets."""
    used = [False] * len(gts)
    flags = []
    for _, box, label in dets:
        best, bi = IOU_TH, -1
        for j, (glabel, gbox) in enumerate(gts):
            if used[j] or (CLASS_MATCH and glabel != label):
                continue
            v = iou(box, gbox)
            if v >= best:
                best, bi = v, j
        if bi >= 0:
            used[bi] = True
        flags.append(bi >= 0)
    return flags


@torch.no_grad()
def batch_eval(model, post, items, gt, corrupt_fn=None):
    """items: (path, img_id, w, h). corrupt_fn: PIL->PIL applied before resize."""
    imgs = []
    for p, _, _, _ in items:
        im = Image.open(p).convert("RGB")
        if corrupt_fn is not None:
            im = corrupt_fn(im)
        imgs.append(tf(im))
    x = torch.stack(imgs).to(DEVICE)
    out = model(x)
    conf_q = out["pred_logits"].sigmoid().max(-1).values
    label_q = out["pred_logits"].argmax(-1)
    unc_q = out["tue_uncertainty"]
    xyxy = box_convert(out["pred_boxes"], "cxcywh", "xyxy")
    sel = conf_q >= CONF_TH
    m = sel & torch.isfinite(unc_q) & torch.isfinite(conf_q)
    sizes = torch.tensor([[w, h] for _, _, w, h in items], device=DEVICE)
    res = post(out, sizes)
    per_image_stats, dets = [], []
    for b, (_, img_id, w, h) in enumerate(items):
        idx = torch.where(m[b])[0]
        c, u, l = conf_q[b][idx], unc_q[b][idx], label_q[b][idx]
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
        for (x0, y0, x1, y1), lab, s in zip(
            r["boxes"][keep].cpu().tolist(),
            r["labels"][keep].cpu().tolist(),
            r["scores"][keep].cpu().tolist(),
        ):
            dets.append(
                {
                    "image_id": img_id,
                    "category_id": int(lab),
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


def scan(model, post, coco, items, gt, corrupt_fn=None, desc=""):
    """Pooled score dict (all + correct-only) and COCO 12-stat over `items`."""
    p = {k: [] for k in ["dc", "du", "ic", "iu", "dcc", "duc", "icc", "iuc"]}
    dets = []
    n_drop = n_empty = 0
    for batch in tqdm(list(chunks(items, BATCH)), desc=desc):
        per_image_stats, batch_dets = batch_eval(model, post, batch, gt, corrupt_fn)
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
    return p, coco_eval(coco, dets), len(p["dc"]), len(p["dcc"]), len(p["ic"])


def load_gt(d):
    coco = COCO(d["anns"])
    imgs = coco.loadImgs(coco.getImgIds())
    cat2label = {c: i for i, c in enumerate(sorted(coco.getCatIds()))}
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
    return coco, items, gt


def auroc(id_dets, ood_dets, higher_is_ood):
    if not id_dets or not ood_dets:
        return float("nan")
    if not higher_is_ood:
        id_dets = [1 - x for x in id_dets]
        ood_dets = [1 - x for x in ood_dets]
    lab = [0] * len(id_dets) + [1] * len(ood_dets)
    return roc_auc_score(lab, id_dets + ood_dets)


# (column, id-key, ood-key, higher_is_ood)
VARIANTS = [
    ("auroc_det_unc", "du", "du", True),
    ("auroc_det_1conf", "dc", "dc", False),
    ("auroc_img_unc", "iu", "iu", True),
    ("auroc_img_1conf", "ic", "ic", False),
    ("auroc_detC_unc", "duc", "duc", True),
    ("auroc_detC_1conf", "dcc", "dcc", False),
    ("auroc_imgC_unc", "iuc", "iuc", True),
    ("auroc_imgC_1conf", "icc", "icc", False),
]


def main():
    model, post = load_model()
    id_datasets = [ds["dataset"] for ds in datasets if not ds["dataset"]["ood"]]
    assert id_datasets, "config has no ID (ood:false) dataset to corrupt"

    out_csv = output_dir / "benchmark_corruptions.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dataset",
                "corruption",
                "severity",
                "split",
                "n_img",
                "n_det",
                "n_det_correct",
            ]
            + COCO_STATS
            + [c for c, *_ in VARIANTS]
        )

        for d in id_datasets:
            coco, items, gt = load_gt(d)

            # clean ID pass = the shared negative pool for this dataset
            clean, stats, ndet, ndetc, nimg = scan(
                model, post, coco, items, gt, None, desc=f"{d['name']} clean"
            )
            w.writerow(
                [d["name"], "clean", 0, "ID", nimg, ndet, ndetc]
                + [f"{s:.4f}" for s in stats]
                + [""] * len(VARIANTS)
            )

            for name in CORRUPTIONS:
                for sev in SEVERITIES:
                    p, stats, ndet, ndetc, nimg = scan(
                        model,
                        post,
                        coco,
                        items,
                        gt,
                        make_corrupt(name, sev),
                        desc=f"{d['name']} {name} s{sev}",
                    )
                    a = [
                        f"{auroc(clean[nk], p[pk], hi):.4f}"
                        for _, nk, pk, hi in VARIANTS
                    ]
                    w.writerow(
                        [d["name"], name, sev, "OOD", nimg, ndet, ndetc]
                        + [f"{s:.4f}" for s in stats]
                        + a
                    )
                    f.flush()
    print("saved:", out_csv)


if __name__ == "__main__":
    main()
