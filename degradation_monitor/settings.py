"""Run settings: the benchmark, the machine's paths and run options, read from a TOML file such as configs/coco.toml.

The method's fixed choices (k = 50 neighbours, the stage-4 key, stages 1-3, the top 1%, the 2,000 + 500 reference
images) are constants in degradation_monitor.method, so that a config edit cannot change them.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

from .corruptions import CONDITIONS
from .datasets.cityscapes import Cityscapes
from .datasets.coco import FOLDS, Coco
from .runs import RunLayout, sha256

PATH_FIELDS = ("run", "checkpoint", "train_images", "val_images", "annotations", "discopatch_root")
BENCHMARKS = {"coco": Coco, "cityscapes": Cityscapes}


@dataclass(frozen=True)
class Settings:
    run: Path
    checkpoint: Path
    train_images: Path
    val_images: Path
    annotations: Path
    discopatch_root: Path
    device: str = "cuda:0"
    batch_size: int = 32
    workers: int = 9
    gpu_memory_gib: Optional[float] = None
    limit: Optional[int] = None
    epochs: int = 65
    seed: int = 44
    benchmark: str = "coco"

    def __post_init__(self):
        if self.benchmark not in BENCHMARKS:
            raise ValueError(f"unknown benchmark {self.benchmark!r}; choose from {', '.join(sorted(BENCHMARKS))}")

    @property
    def layout(self) -> RunLayout:
        return RunLayout(self.run)

    @property
    def dataset(self) -> Coco:
        return BENCHMARKS[self.benchmark](self.train_images, self.val_images, self.annotations, seed=self.seed,
                                          limit=self.limit)

    def protocol(self) -> dict:
        """What every result in the run folder depends on; a stage refuses a run folder made with another one."""
        from .baselines.contrastive_conf import THETA
        from .baselines.knn import KNN_K, KNN_K_MAX
        from .detector.postprocess import TOP_K
        return {"dataset": self.benchmark, "seed": self.seed, "limit": self.limit, "folds": FOLDS,
                "conditions": [list(c) for c in CONDITIONS], "checkpoint_sha256": sha256(self.checkpoint),
                "top_k": TOP_K, "knn_k": KNN_K, "knn_k_max": KNN_K_MAX, "theta": THETA}


def load_settings(path, **overrides) -> Settings:
    """Settings from a TOML file. Keyword overrides (from the command line) win; an override of None keeps the file's value."""
    path = Path(path)
    values = tomllib.loads(path.read_text())
    unknown = sorted(set(values) - {f.name for f in fields(Settings)})
    if unknown:
        raise ValueError(f"unknown settings in {path}: {', '.join(unknown)}")
    missing = [name for name in PATH_FIELDS if name not in values]
    if missing:
        raise ValueError(f"{path} needs {', '.join(missing)}")
    values.update({key: value for key, value in overrides.items() if value is not None})
    for name in PATH_FIELDS:
        values[name] = Path(values[name])
    return Settings(**values)
