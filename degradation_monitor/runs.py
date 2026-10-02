"""The run folder: where every stage reads and writes, atomic writes, resume checks and the manifest.

  manifest.json                    the protocol, the environment, and the inputs each score folder was computed from
  reference/<name>/                what the clean train images provide: knn, activation_cdf, hashemi, discopatch, method
  scores/<pass>/<image stem>.npz   one file per evaluation image, 96 conditions each: detector, activations,
                                   discopatch, method
  reports/<name>/                  the report stage's tables
  timing.json                      the timing stage's measurements
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np

SCORE_KEYS = {
    "detector": ("saod_min", "saod_top3", "conf_pos", "conf_neg", "knn", "det_scores", "det_labels", "det_boxes",
                 "digests", "size"),
    "activations": ("hashemi_decoder", "hashemi_encoder", "hashemi_encoder_maps", "cdf_backbone", "cdf_backbone_z",
                    "cdf_stages"),
    "discopatch": ("dcp",),
    "method": tuple(f"{statistic}_s{stage}" for statistic in ("means", "top") for stage in range(1, 5)),
}
PACKAGES = ("torch", "torchvision", "numpy", "scipy", "scikit-image", "scikit-learn", "imagecorruptions", "uq-detr",
            "pycocotools", "Pillow")


@dataclass(frozen=True)
class RunLayout:
    root: Path

    def reference(self, *parts: str) -> Path:
        return self.root.joinpath("reference", *parts)

    @property
    def knn_bank(self) -> Path:
        return self.reference("knn", "bank.npy")

    @property
    def knn_names(self) -> Path:
        return self.reference("knn", "image_names.json")

    @property
    def cdf_reference(self) -> Path:
        return self.reference("activation_cdf", "reference.npz")

    @property
    def cdf_fit(self) -> Path:
        return self.reference("activation_cdf", "fit.json")

    @property
    def cdf_zstats(self) -> Path:
        return self.reference("activation_cdf", "zstats.json")

    @property
    def hashemi_intervals(self) -> Path:
        return self.reference("hashemi", "intervals.npz")

    @property
    def hashemi_fit(self) -> Path:
        return self.reference("hashemi", "fit.json")

    @property
    def discopatch_dir(self) -> Path:
        return self.reference("discopatch")

    @property
    def discopatch_checkpoint(self) -> Path:
        return self.reference("discopatch", "discriminator.pt")

    @property
    def discopatch_training(self) -> Path:
        return self.reference("discopatch", "training.json")

    @property
    def method_bank(self) -> Path:
        return self.reference("method", "bank.npz")

    @property
    def method_zstats(self) -> Path:
        return self.reference("method", "zstats.npz")

    def scores(self, name: str) -> Path:
        if name not in SCORE_KEYS:
            raise ValueError(f"unknown score folder {name!r}; choose from {sorted(SCORE_KEYS)}")
        return self.root / "scores" / name

    def score_file(self, name: str, image) -> Path:
        return self.scores(name) / f"{Path(image).stem}.npz"

    def report(self, name: str = "coco") -> Path:
        return self.root / "reports" / name

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def timing(self) -> Path:
        return self.root / "timing.json"


def atomic_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_npz(path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def load_npz(path, keys) -> dict:
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as data:
            loaded = {key: data[key] for key in data.files}
    except Exception as error:
        raise ValueError(f"malformed result file {path.name}: {path}") from error
    missing = set(keys) - set(loaded)
    if missing:
        raise ValueError(f"result file {path.name} lacks {sorted(missing)}: {path}")
    return loaded


def valid_existing(path, keys) -> bool:
    """False for a missing file; a damaged or incomplete one is refused rather than silently redone."""
    if not Path(path).exists():
        return False
    load_npz(path, keys)
    return True


def stack(folder, names, keys) -> dict:
    """The given arrays of every named image's file, stacked along a new first axis."""
    folder = Path(folder)
    missing = [n for n in names if not (folder / f"{Path(n).stem}.npz").exists()]
    if missing:
        raise ValueError(f"{folder.name} is incomplete: {len(missing)} of {len(names)} missing, e.g. {missing[0]}")
    columns = {key: [] for key in keys}
    for name in names:
        with np.load(folder / f"{Path(name).stem}.npz") as item:
            for key in keys:
                columns[key].append(item[key])
    return {key: np.stack(values) for key, values in columns.items()}


def sha1(path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def progress(label, done, total, started) -> None:
    rate = done / max(time.time() - started, 1e-9)
    remaining = (total - done) / max(rate, 1e-9)
    print(f"[{label}] {done}/{total} images, {rate:.2f}/s, about {remaining / 60:.0f} min left", flush=True)


def _plain(value):
    """The value as JSON gives it back, so that tuples and lists compare equal."""
    return json.loads(json.dumps(value))


class Manifest:
    """manifest.json: the run's protocol, its environment, and the inputs each score folder was computed from."""

    def __init__(self, layout: RunLayout):
        self.layout = layout
        self.path = layout.manifest

    def read(self) -> dict:
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def update(self, **sections) -> None:
        data = self.read()
        data.update(_plain(sections))
        atomic_json(self.path, data)

    def _holds_scores(self, folder: str) -> bool:
        """Whether scores/<folder>/ already holds a result file."""
        return any(self.layout.scores(folder).glob("*.npz"))

    def check_protocol(self, protocol: dict) -> None:
        """Record the protocol of a new run folder; refuse a run folder made with another protocol.

        A run folder that already holds score files but records no protocol is refused too; files under reference/
        do not count.
        """
        protocol = _plain(protocol)
        recorded = self.read().get("protocol")
        if recorded is None:
            if any(self._holds_scores(folder) for folder in SCORE_KEYS):
                raise ValueError("this run folder already holds score files but its manifest records no protocol; "
                                 f"convert it or start a new run folder: {self.path}")
            self.update(protocol=protocol)
        elif recorded != protocol:
            changed = sorted(k for k in set(recorded) | set(protocol) if recorded.get(k) != protocol.get(k))
            raise ValueError(f"this run folder was made with another protocol ({', '.join(changed)}): {self.path}")

    def check_inputs(self, folder: str, inputs: dict) -> None:
        """Record what a score folder is computed from; refuse to add to it if it was computed from something else.

        A score folder that already holds files but has no recorded inputs is refused too.
        """
        inputs = _plain(inputs)
        data = self.read()
        recorded = data.get("inputs", {}).get(folder)
        if recorded is None:
            if self._holds_scores(folder):
                raise ValueError(f"scores/{folder} already holds files but the manifest records no inputs for it: "
                                 f"{self.path}")
            data.setdefault("inputs", {})[folder] = inputs
            atomic_json(self.path, data)
        elif recorded != inputs:
            changed = sorted(k for k in set(recorded) | set(inputs) if recorded.get(k) != inputs.get(k))
            raise ValueError(f"the inputs of scores/{folder} changed since its files were written "
                             f"({', '.join(changed)}): {self.path}")

    def record_environment(self, discopatch_root) -> None:
        """Package versions, the DisCoPatch commit and the GPU, recorded once."""
        if "environment" in self.read():
            return
        import torch
        versions = {}
        for package in PACKAGES:
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                versions[package] = None
        commit = subprocess.run(["git", "-C", str(discopatch_root), "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip() or None
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        self.update(environment={"packages": versions, "discopatch_commit": commit, "gpu": gpu})
