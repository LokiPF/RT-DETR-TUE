"""The runnable, resumable steps. Each refuses inputs that changed since its results were written."""
from __future__ import annotations

from . import baselines, method, report

STAGES = {
    "check": baselines.check,
    "knn-bank": baselines.knn_bank,
    "detector-pass": baselines.detector_pass,
    "discopatch-train": baselines.discopatch_train,
    "discopatch-pass": baselines.discopatch_pass,
    "hashemi-fit": baselines.hashemi_fit,
    "cdf-fit": baselines.cdf_fit,
    "cdf-zstats": baselines.cdf_zstats,
    "activation-pass": baselines.activation_pass,
    "method-reference": method.method_reference,
    "method-pass": method.method_pass,
    "report": report.write_report,
    "timing": baselines.timing,
}


def run_stage(name: str, settings) -> None:
    """Check the run folder's protocol, record the environment once, then run one stage."""
    from ..runs import Manifest
    if name not in STAGES:
        raise ValueError(f"unknown stage {name!r}; choose from {sorted(STAGES)}")
    settings.layout.root.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(settings.layout)
    manifest.check_protocol(settings.protocol())
    manifest.record_environment(settings.discopatch_root)
    STAGES[name](settings, manifest)
