"""OOD AUROC benchmark for RT-DETRv2 with per-query uncertainty (batched).
Run from the RT-DETRv2 repo root (so `from src.core import YAMLConfig` resolves).
Label is dataset-level: all ID datasets form the shared negative pool; each OOD
dataset is scored against it separately (positive=OOD). Scores compared:
uncertainty (higher=OOD) and 1-confidence (lower conf=OOD), both per-detection
(conf>=0.5) and per-image (mean over conf>=0.5 detections)."""
import pathlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import csv
import os

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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_FILE = "output/auroc/exp1/exp1_config.yaml"

with open(CONFIG_FILE, 'r') as f:
    config = yaml.safe_load(f)

datasets = config["datasets"]
model_config = config["config"]
ckpt = config["checkpoint"]
output_dir = pathlib.Path(CONFIG_FILE).parent.resolve()

def load_model():
    cfg = YAMLConfig(model_config, resume=ckpt)
    ckpt_ = torch.load(ckpt, map_location="cpu")
    state = ckpt_.get("ema", {}).get("module") or ckpt_.get("model") or ckpt_
    cfg.model.load_state_dict(state)
    return cfg.model.to(DEVICE).eval()

tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor()])

@torch.no_grad()
def batch_scores(model, paths):
    """Per image: (conf, unc) of selected finite queries + count of dropped ones."""
    x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths]).to(DEVICE)
    out = model(x)
    conf = out["pred_logits"].sigmoid().max(-1).values  # [B, Q]
    unc = out["tue_uncertainty"]                        # [B, Q]
    sel = conf >= CONF_TH
    m = sel & torch.isfinite(unc) & torch.isfinite(conf)
    res = []
    for b in range(conf.shape[0]):
        dropped = int((sel[b] & ~torch.isfinite(unc[b])).sum())
        res.append((conf[b][m[b]].cpu(), unc[b][m[b]].cpu(), dropped))
    return res

def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]

def scan(model, d):
    """Return pooled (det_conf, det_unc, img_conf, img_unc) for one dataset."""
    coco = COCO(d["anns"])
    paths = [os.path.join(d["imgs"], im["file_name"]) for im in coco.loadImgs(coco.getImgIds())]
    dc, du, ic, iu = [], [], [], []
    n_drop = n_empty = 0
    for batch in tqdm(list(chunks(paths, BATCH)), desc=d["name"]):
        for conf, unc, dropped in batch_scores(model, batch):
            n_drop += dropped
            if conf.numel() == 0:
                n_empty += 1; continue
            dc += conf.tolist(); du += unc.tolist()
            ic.append(conf.mean().item()); iu.append(unc.mean().item())
    print(f"[{'OOD' if d['ood'] else 'ID '}] {d['name']}: {len(ic)} imgs w/ dets, "
          f"{n_empty} empty, {len(dc)} dets, {n_drop} non-finite unc dropped")
    return dc, du, ic, iu

def auroc(neg, pos, higher_is_ood):
    """AUROC of a score separating ID (neg) from OOD (pos)."""
    if not higher_is_ood:              # confidence: lower => more OOD
        neg = [1 - x for x in neg]; pos = [1 - x for x in pos]
    lab = [0] * len(neg) + [1] * len(pos)
    return roc_auc_score(lab, neg + pos)

def main():
    model = load_model()
    id_dc, id_du, id_ic, id_iu = [], [], [], []   # shared ID pool
    ood = {}                                       # name -> pooled scores
    for ds in datasets:
        d = ds["dataset"]
        dc, du, ic, iu = scan(model, d)
        if d["ood"]:
            ood[d["name"]] = (dc, du, ic, iu)
        else:
            id_dc += dc; id_du += du; id_ic += ic; id_iu += iu

    assert id_dc, "no ID detections found"
    assert ood, "no OOD datasets found"

    rows = []
    for name, (dc, du, ic, iu) in ood.items():
        rows += [
            (name, "per-detection", "uncertainty",  auroc(id_du, du, True)),
            (name, "per-detection", "1-confidence", auroc(id_dc, dc, False)),
            (name, "per-image",     "uncertainty",  auroc(id_iu, iu, True)),
            (name, "per-image",     "1-confidence", auroc(id_ic, ic, False)),
        ]

    out_csv = output_dir / "auroc.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ood_dataset", "level", "score", "auroc"])
        for name, level, score, val in rows:
            w.writerow([name, level, score, f"{val:.4f}"])
            print(f"{name:32s} {level:14s} {score:12s} {val:.4f}")
    print("saved:", out_csv)

if __name__ == "__main__":
    main()