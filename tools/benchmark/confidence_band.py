"""Confidence-band diagnostic on ONE strong corruption (clean vs corrupted, same
images) — mirrors the KNN pilot's blur setup. Run from the RT-DETR repo root.

Scores ALL queries (drops the model's confidence gate), bins each image's queries
into 10 confidence deciles, and reports:
  (1) per-band mean uncertainty of the corrupted images -> does the 'hill'
      appear under this corruption?  [saved as confidence_band_<corruption>_s<sev>.png]
  (2) per-image AUROC (clean vs corrupted) of:
        mean     = mean uncertainty over all queries
        raw_gap  = band[50-60%] - band[90-100%]   (pilot's blur winner)
        low_high = mean(bands 0-2) - mean(bands 7-9)   (low-vs-high, for the tattoo shape)
        rel_gap  = 2*(band[50-60%] - band[0-10%]) / (band[50-60%]+band[0-10%])

    pip install imagecorruptions

Heavy: all ~300 queries/image get an MST, twice (clean + corrupted). Use a small
subsplit."""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# --- imagecorruptions compat shims (numpy>=2 removed aliases; skimage renamed multichannel) ---
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

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import yaml
from imagecorruptions import corrupt
from PIL import Image
from pycocotools.coco import COCO
from skimage.filters import gaussian as _sk_gaussian
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def _gaussian_compat(img, *args, **kwargs):
    if kwargs.pop("multichannel", False):
        kwargs.setdefault("channel_axis", -1)
    return _sk_gaussian(img, *args, **kwargs)


_ic.gaussian = _gaussian_compat

SIZE = 640
BATCH = 16
GATE = 0.0  # score every query
N_BANDS = 10
CORRUPTION = "defocus_blur"  # one strong photometric corruption (closest common to the pilot's Gaussian blur)
SEVERITY = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_FILE = "output/auroc/exp9/exp9_config.yaml"

tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor()])


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def make_corrupt(name, severity):
    def f(pil):
        arr = np.asarray(pil, dtype=np.uint8)
        return Image.fromarray(corrupt(arr, corruption_name=name, severity=severity))

    return f


def load_model():
    from src.core import YAMLConfig

    with open(CONFIG_FILE) as f:
        cfg_d = yaml.safe_load(f)
    cfg = YAMLConfig(cfg_d["config"], resume=cfg_d["checkpoint"])
    c = torch.load(cfg_d["checkpoint"], map_location="cpu")
    state = c.get("ema", {}).get("module")
    assert state is not None, "checkpoint has no ema.module"
    cfg.model.load_state_dict(state)
    model = cfg.model.to(DEVICE).eval()
    model.tue_confidence_threshold = GATE
    return model, cfg_d["datasets"]


@torch.no_grad()
def band_vectors(model, d, corrupt_fn, desc):
    """Per image: N_BANDS decile-mean uncertainties (queries sorted by confidence)."""
    coco = COCO(d["anns"])
    paths = [
        os.path.join(d["imgs"], im["file_name"])
        for im in coco.loadImgs(coco.getImgIds())
    ]
    vecs = []
    for b in tqdm(list(chunks(paths, BATCH)), desc=desc):
        imgs = []
        for p in b:
            im = Image.open(p).convert("RGB")
            if corrupt_fn is not None:
                im = corrupt_fn(im)
            imgs.append(tf(im))
        x = torch.stack(imgs).to(DEVICE)
        out = model(x)
        conf = out["pred_logits"].sigmoid().max(-1).values
        unc = out["tue_uncertainty"]
        fin = torch.isfinite(unc) & torch.isfinite(conf)
        for i in range(conf.shape[0]):
            m = fin[i]
            if int(m.sum()) < N_BANDS:
                continue
            c = conf[i][m].cpu().numpy()
            u = unc[i][m].cpu().numpy()
            u = u[np.argsort(c)]
            vecs.append(np.array([p.mean() for p in np.array_split(u, N_BANDS)]))
    return np.asarray(vecs)


def scores(V):
    return {
        "mean": V.mean(axis=1),
        "raw_gap": V[:, 5] - V[:, 9],
        "low_high": V[:, 0:3].mean(axis=1) - V[:, 7:10].mean(axis=1),
        "rel_gap": 2 * (V[:, 5] - V[:, 0]) / (V[:, 5] + V[:, 0] + 1e-9),
    }


def auroc(id_s, ood_s):
    lab = [0] * len(id_s) + [1] * len(ood_s)
    return roc_auc_score(lab, list(id_s) + list(ood_s))


def main():
    model, datasets = load_model()
    id_ds = [ds["dataset"] for ds in datasets if not ds["dataset"]["ood"]]
    assert id_ds, "config has no ID (ood:false) dataset to corrupt"
    cf = make_corrupt(CORRUPTION, SEVERITY)

    clean_V = np.concatenate(
        [band_vectors(model, d, None, f"{d['name']} clean") for d in id_ds]
    )
    corr_V = np.concatenate(
        [
            band_vectors(model, d, cf, f"{d['name']} {CORRUPTION} s{SEVERITY}")
            for d in id_ds
        ]
    )

    cs, os_ = scores(clean_V), scores(corr_V)
    print(
        f"\n{CORRUPTION} s{SEVERITY}  (clean n={len(clean_V)}, corrupt n={len(corr_V)})"
    )
    for k in cs:
        print(f"  AUROC {k:8s}: {auroc(cs[k], os_[k]):.4f}")

    x = np.arange(N_BANDS)
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(x, corr_V.mean(0), "-s", color="red", label=f"{CORRUPTION} s{SEVERITY}")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{i * 10}-{i * 10 + 10}" for i in x], rotation=45, ha="right", fontsize=7
    )
    ax.set_xlabel("confidence decile (low -> high)")
    ax.set_ylabel("mean uncertainty")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(CONFIG_FILE).parent / f"confidence_band_{CORRUPTION}_s{SEVERITY}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    main()
