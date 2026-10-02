"""One-time conversion of the old run folder (runs/coco-baselines/) into the clean layout (runs/coco/).

    python scripts/convert_runs.py --old runs/coco-baselines [--config configs/coco.toml]

- Hard links: every per-image file and every fitted reference. A hard link is the same file under a second name, so
  nothing is copied, its content cannot differ from the source, and the old folder's files are untouched.
- Rewritten: the method's bank and z-statistics, keeping only the level and top-1% arrays. Each kept array is checked
  to equal its source exactly.
- The manifest: the protocol, checked against the old run's configuration; the old environment and sanity check; the
  inputs each score folder was computed from, checked against the old markers; and this conversion's record.
- Refused: an existing target, an old run with another protocol, an incomplete or malformed score folder, and scores
  computed from other fits than the stored ones. Every check runs before anything is written. The new folder is
  built under a temporary name and renamed only when it is complete.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from degradation_monitor.baselines.activation_cdf import BINS as CDF_BINS  # noqa: E402
from degradation_monitor.baselines.hashemi import K as HASHEMI_K  # noqa: E402
from degradation_monitor.cli import DEFAULT_CONFIG  # noqa: E402
from degradation_monitor.method.statistics import KEYS  # noqa: E402
from degradation_monitor.runs import SCORE_KEYS, Manifest, RunLayout, atomic_npz, sha1, sha256  # noqa: E402
from degradation_monitor.settings import load_settings  # noqa: E402

SCORE_FOLDERS = {"detector": "test", "activations": "test_activation", "discopatch": "test_dcp",
                 "method": "test_convtu_means"}
LINKED = {  # RunLayout property: the old file, relative to the old run folder
    "knn_bank": "bank/knn_bank.npy", "knn_names": "bank/bank_names.json",
    "cdf_reference": "cdf/reference.npz", "cdf_fit": "cdf/fit.json", "cdf_zstats": "cdf/zstats.json",
    "hashemi_intervals": "hashemi/intervals.npz", "hashemi_fit": "hashemi/fit.json",
    "discopatch_checkpoint": "discopatch/DisCoPatch/Discriminator_coco.pt",
    "discopatch_training": "discopatch/training.json", "timing": "timing.json",
}
REWRITTEN = {"method_bank": "convtu/channels_bank.npz", "method_zstats": "convtu/channels_zstats.npz"}
MARKERS = ("run_config.json", "environment.json", "sanity.json", "test_activation/fits.json",
           "test_dcp/checkpoint.json")
PROTOCOL_FIELDS = ("seed", "limit", "folds", "conditions", "top_k", "knn_k", "knn_k_max", "theta")


def _json(path: Path):
    return json.loads(path.read_text())


def check_protocol(old: Path, protocol: dict) -> None:
    """The old run's configuration must be the protocol the clean branch records, checkpoint included."""
    config = _json(old / "run_config.json")
    differ = [field for field in PROTOCOL_FIELDS if config.get(field) != protocol[field]]
    checkpoint = Path(config["checkpoint"])
    if not checkpoint.exists() or sha256(checkpoint) != protocol["checkpoint_sha256"]:
        differ.append("checkpoint")
    if differ:
        raise ValueError(f"the old run used another protocol ({', '.join(differ)}): {old / 'run_config.json'}")


def score_files(old: Path, names: list) -> dict:
    """Every evaluation image's file in each old score folder; refuses a missing or incomplete one."""
    out = {}
    for folder, source in SCORE_FOLDERS.items():
        paths = [old / source / f"{Path(n).stem}.npz" for n in names]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            raise ValueError(f"{source} is incomplete: {len(missing)} of {len(names)} images missing, "
                             f"e.g. {missing[0]}")
        for path in paths:
            with np.load(path) as data:
                lacking = set(SCORE_KEYS[folder]) - set(data.files)
            if lacking:
                raise ValueError(f"{source}/{path.name} lacks {sorted(lacking)}")
        out[folder] = paths
    return out


def inputs(old: Path) -> dict:
    """What each score folder was computed from, checked against the old run's markers."""
    activations = {"hashemi_intervals": sha1(old / LINKED["hashemi_intervals"]),
                   "cdf_reference": sha1(old / LINKED["cdf_reference"]), "cdf_zstats": sha1(old / LINKED["cdf_zstats"]),
                   "hashemi_k": HASHEMI_K, "cdf_bins": CDF_BINS}
    fits = _json(old / "test_activation" / "fits.json")
    recorded = {"hashemi_intervals": fits["hashemi-fit"]["sha1"], "cdf_reference": fits["cdf-fit"]["sha1"],
                "cdf_zstats": fits["cdf-zstats"]["sha1"], "hashemi_k": fits["hashemi_k"], "cdf_bins": fits["cdf_bins"]}
    if recorded != activations:
        raise ValueError("the activation scores were computed from other fits than the stored ones")
    discopatch = {"discriminator": sha1(old / LINKED["discopatch_checkpoint"])}
    if _json(old / "test_dcp" / "checkpoint.json")["sha1"] != discopatch["discriminator"]:
        raise ValueError("the DisCoPatch scores were computed with another discriminator than the stored one")
    return {"detector": {"knn_bank": sha1(old / LINKED["knn_bank"])}, "activations": activations,
            "discopatch": discopatch}


def rewrite_reference(source: Path, target: Path) -> dict:
    """The method's reference with only the level and top-1% arrays, each equal to its source exactly."""
    with np.load(source) as data:
        missing = set(KEYS) - set(data.files)
        if missing:
            raise ValueError(f"{source} lacks {sorted(missing)}")
        kept = {key: data[key] for key in KEYS}
        dropped = sorted(set(data.files) - set(KEYS))
    atomic_npz(target, **kept)
    with np.load(target) as written:
        for key in KEYS:
            if written[key].dtype != kept[key].dtype or not np.array_equal(written[key], kept[key]):
                raise ValueError(f"{target.name}: {key} was not written exactly")
    return {"dropped": dropped, "sha1": sha1(target)}


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, target)


def convert(old, settings) -> dict:
    """Build settings.run from the old run folder; returns the conversion record."""
    old, target = Path(old), settings.run
    staging = target.with_name(f".{target.name}.converting")
    for path in (target, staging):
        if path.exists():
            raise ValueError(f"{path} already exists; remove it to convert again")
    missing = [s for s in (*MARKERS, *LINKED.values(), *REWRITTEN.values()) if not (old / s).exists()]
    if missing:
        raise ValueError(f"the old run folder lacks {', '.join(missing)}: {old}")
    protocol = settings.protocol()
    check_protocol(old, protocol)
    files = score_files(old, [p.name for p in settings.dataset.evaluation_images()])
    recorded_inputs = inputs(old)
    layout = RunLayout(staging)
    try:
        for attribute, source in LINKED.items():
            _link(old / source, getattr(layout, attribute))
        for folder, paths in files.items():
            for path in paths:
                _link(path, layout.score_file(folder, path))
        rewritten = {attribute: {"source": source, **rewrite_reference(old / source, getattr(layout, attribute))}
                     for attribute, source in REWRITTEN.items()}
        (staging / "logs").mkdir()
        record = {"source": str(old.resolve()), "converted": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "linked": {**{folder: len(paths) for folder, paths in files.items()},
                             "references": sorted(LINKED.values())},
                  "rewritten": rewritten}
        Manifest(layout).update(protocol=protocol, environment=_json(old / "environment.json"),
                                check=_json(old / "sanity.json"), inputs=recorded_inputs, conversion=record)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)  # links and rewritten files only; the old files stay
        raise
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Convert the old run folder into the clean layout, once.")
    parser.add_argument("--old", type=Path, required=True, help="the old run folder, e.g. runs/coco-baselines")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="names the new run folder")
    args = parser.parse_args(argv)
    record = convert(args.old, load_settings(args.config))
    print(json.dumps({"linked": record["linked"], "rewritten": record["rewritten"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
