"""Which stage of RT-DETRv2-R18 describes the scene, and keeps describing it when the image is corrupted?

    python scripts/paper/key_by_stage.py

Each stage in turn serves as the content key: its channel means, standardised by their mean and spread over the
2,000-image clean bank, find each test image's 50 nearest bank images (Euclidean), exactly as the method finds them
with stage 4 (degradation_monitor/method/reference.py). For each key stage:
- content: the mean Jaccard overlap between the COCO panoptic categories (things and stuff) in the clean test image and
  in each of its 50 neighbours, against 50 random bank images;
- stability: the share of the clean image's 50 neighbours that are still among the corrupted image's 50 neighbours,
  averaged over the 19 corruption types at five strengths (and per strength).
On all 5,000 val images, and separately on the development images (positions 0-1969 of the seed-44 order) and the
untouched ones. Reads runs/coco (read-only). Writes docs/results/paper-evidence/key_by_stage.csv.
"""
from __future__ import annotations

import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from degradation_monitor.corruptions import CONDITIONS  # noqa: E402
from degradation_monitor.method.reference import NEIGHBOURS, fit_own_average, nearest_rows  # noqa: E402
from degradation_monitor.settings import load_settings  # noqa: E402

SCORES = ROOT / "runs" / "coco" / "scores" / "method"
BANK = ROOT / "runs" / "coco" / "reference" / "method" / "bank.npz"
ANNOTATIONS = Path("/home/yuchen/YuchenZ/Datasets/coco/annotations")
OUT = ROOT / "docs" / "results" / "paper-evidence" / "key_by_stage.csv"
STAGES = ("s1", "s2", "s3", "s4")
DEVELOPMENT = 1970
SEED = 44


def panoptic(path: Path) -> dict:
    """image id -> the set of panoptic categories (things and stuff) it shows."""
    with path.open() as handle:
        data = json.load(handle)
    return {a["image_id"]: {s["category_id"] for s in a["segments_info"]} for a in data["annotations"]}


def stage_means(path: Path) -> list[np.ndarray]:
    with np.load(path) as z:
        return [z[f"means_{stage}"].astype(np.float32) for stage in STAGES]  # each (96, C)


def overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row by row, the share of a's members that are also in b (both (rows, k))."""
    return (a[:, :, None] == b[:, None, :]).any(axis=2).mean(axis=1)


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else np.nan


def main() -> None:
    dataset = load_settings(ROOT / "configs" / "coco.toml").dataset
    val_paths = dataset.evaluation_images()  # the seed-44 order
    bank_ids = [int(p.stem) for p in dataset.reference_split("bank")]  # the bank's row order
    val_labels, train_labels = panoptic(ANNOTATIONS / "panoptic_val2017.json"), panoptic(ANNOTATIONS / "panoptic_train2017.json")
    image_labels = [val_labels.get(int(p.stem), set()) for p in val_paths]
    bank_labels = [train_labels.get(i, set()) for i in bank_ids]
    with Pool(8) as pool:
        per_image = pool.map(stage_means, [SCORES / (p.stem + ".npz") for p in val_paths], chunksize=50)
    images, conditions = len(val_paths), len(CONDITIONS)
    severities = np.array([s for _, s in CONDITIONS])
    sets = {"all": np.arange(images), "development": np.arange(DEVELOPMENT), "untouched": np.arange(DEVELOPMENT, images)}
    random_rows = np.random.default_rng(SEED).integers(0, len(bank_ids), size=(images, NEIGHBOURS))
    random_jaccard = np.array([np.nanmean([jaccard(image_labels[i], bank_labels[j]) for j in random_rows[i]])
                               for i in range(images)])
    rows = []
    with np.load(BANK) as bank:
        for s, stage in enumerate(STAGES):
            centre, spread = fit_own_average(bank[f"means_{stage}"])
            bank_keys = (np.asarray(bank[f"means_{stage}"], dtype=np.float64) - centre) / spread
            queries = np.stack([m[s] for m in per_image]).reshape(images * conditions, -1)
            found = nearest_rows((queries - centre) / spread, bank_keys, NEIGHBOURS).reshape(images, conditions, -1)
            clean = found[:, 0]
            content = np.array([np.nanmean([jaccard(image_labels[i], bank_labels[j]) for j in clean[i]])
                                for i in range(images)])
            kept = np.stack([overlap(clean, found[:, c]) for c in range(1, conditions)], axis=1)  # (images, 95)
            for name, members in sets.items():
                row = {"key_stage": stage, "images": name, "jaccard": f"{np.nanmean(content[members]):.4f}",
                       "jaccard_random": f"{np.nanmean(random_jaccard[members]):.4f}",
                       "neighbours_kept": f"{kept[members].mean():.4f}"}
                for severity in range(1, 6):
                    row[f"neighbours_kept_s{severity}"] = f"{kept[members][:, severities[1:] == severity].mean():.4f}"
                rows.append(row)
            print(f"{stage}: Jaccard {np.nanmean(content):.3f} (random {np.nanmean(random_jaccard):.3f}), neighbours kept "
                  f"{kept.mean():.1%} on average, {kept[:, severities[1:] == 1].mean():.1%} at the mildest strength", flush=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
