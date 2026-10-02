"""Command line: python -m degradation_monitor <stage> [--config configs/coco.toml] [overrides]."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "coco.toml"


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from error
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive number") from error
    if not number > 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return number


def build_parser() -> argparse.ArgumentParser:
    from .stages import STAGES
    parser = argparse.ArgumentParser(prog="python -m degradation_monitor",
                                     description="Corruption detection inside a frozen RT-DETRv2-R18: run one stage.")
    parser.add_argument("stage", choices=list(STAGES))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="paths and run options (TOML)")
    parser.add_argument("--run", type=Path, help="the run folder, instead of the config's")
    parser.add_argument("--device")
    parser.add_argument("--limit", type=positive_int, help="only the first N evaluation images")
    parser.add_argument("--batch-size", type=positive_int)
    parser.add_argument("--workers", type=nonnegative_int)
    parser.add_argument("--gpu-memory-gib", type=positive_float, help="this process's share of a shared GPU")
    parser.add_argument("--epochs", type=positive_int, help="DisCoPatch's training epochs")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from . import stages
        from .settings import load_settings
        settings = load_settings(args.config, run=args.run, device=args.device, limit=args.limit,
                                 batch_size=args.batch_size, workers=args.workers,
                                 gpu_memory_gib=args.gpu_memory_gib, epochs=args.epochs)
        stages.run_stage(args.stage, settings)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0
