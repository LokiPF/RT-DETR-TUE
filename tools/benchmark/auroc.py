"""OOD AUROC benchmark for RT-DETRv2 with per-query uncertainty.
Run from the RT-DETRv2 repo root (so `from src.core import YAMLConfig` resolves).
Label is dataset-level: OOD datasets are positive (1), ID datasets negative (0).
Scores compared: uncertainty (higher=OOD) and 1-confidence (lower conf=OOD),
both per-detection (conf>=0.5) and per-image (mean over conf>=0.5 detections)."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import os

import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from pycocotools.coco import COCO
from sklearn.metrics import roc_auc_score

from src.core import YAMLConfig

CONF_TH = 0.5
SIZE = 640
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_FILE = "output/auroc/exp1/exp1_config.yaml"

# each dataset: image dir, COCO annotation json, and ood flag (True=OOD)
with open(CONFIG_FILE, 'r') as f:
    config = yaml.safe_load(f)

datasets = config["datasets"]
model_config = config["config"]
CKPT = config["checkpoint"]

def load_model():
    cfg = YAMLConfig(model_config, resume=CKPT)
    ckpt = torch.load(CKPT, map_location="cpu")
    state = ckpt.get("ema", {}).get("module") or ckpt.get("model") or ckpt
    cfg.model.load_state_dict(state)
    return cfg.model.to(DEVICE).eval()

tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor()])

@torch.no_grad()
def query_scores(model, img_path):
    """Return (conf, unc) tensors for queries with conf>=CONF_TH in one image."""
    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(DEVICE)
    out = model(x)
    conf = out["pred_logits"].sigmoid()[0].max(-1).values  # (Q,)
    unc = out["uncertainty"][0].reshape(-1)                 # (Q,)
    m = conf >= CONF_TH
    return conf[m].cpu(), unc[m].cpu()

def main():
    model = load_model()
    d_lab, d_conf, d_unc = [], [], []   # per-detection
    i_lab, i_conf, i_unc = [], [], []   # per-image (means)
    for ds in datasets:
        coco = COCO(ds["dataset"]["ann"])
        y = 1 if ds["dataset"]["ood"] else 0
        for im in coco.loadImgs(coco.getImgIds()):
            conf, unc = query_scores(model, os.path.join(ds["dataset"]["img"], im["file_name"]))
            if conf.numel() == 0:
                continue
            d_conf += conf.tolist(); d_unc += unc.tolist(); d_lab += [y] * conf.numel()
            i_conf.append(conf.mean().item()); i_unc.append(unc.mean().item()); i_lab.append(y)

    def auroc(lab, score):  # score = higher means more OOD
        return roc_auc_score(lab, score)

    print("per-detection  AUROC  uncertainty: %.4f" % auroc(d_lab, d_unc))
    print("per-detection  AUROC  1-confidence: %.4f" % auroc(d_lab, [1 - c for c in d_conf]))
    print("per-image      AUROC  uncertainty: %.4f" % auroc(i_lab, i_unc))
    print("per-image      AUROC  1-confidence: %.4f" % auroc(i_lab, [1 - c for c in i_conf]))

if __name__ == "__main__":
    main()