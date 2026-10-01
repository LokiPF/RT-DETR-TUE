from __future__ import annotations

import argparse
import sys

from .benchmark import run_coco_benchmark


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed fingerprint corruption benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    benchmark = commands.add_parser("benchmark-coco")
    benchmark.add_argument("--checkpoint", required=True)
    benchmark.add_argument("--coco-train-images", required=True)
    benchmark.add_argument("--coco-val-images", required=True)
    benchmark.add_argument("--reference-count", required=True, type=positive_int)
    benchmark.add_argument("--evaluation-count", required=True, type=positive_int)
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--device", default="cuda:0")
    benchmark.add_argument("--batch-size", default=1, type=positive_int)
    benchmark.add_argument("--seed", default=44, type=nonnegative_int)
    baselines = commands.add_parser("baselines-coco")
    baselines.add_argument("--phase", required=True,
                           choices=["sanity", "bank", "test", "train-discopatch", "discopatch-scores",
                                    "hashemi-fit", "cdf-fit", "cdf-zstats", "activation-scores", "timing", "report",
                                    "convtu-calibrate", "convtu-bank", "convtu-zstats", "convtu-scores",
                                    "convtu-report", "convtu-channels"])
    baselines.add_argument("--output", required=True)
    baselines.add_argument("--checkpoint", required=True)
    baselines.add_argument("--coco-train-images", required=True)
    baselines.add_argument("--coco-val-images", required=True)
    baselines.add_argument("--coco-annotations", required=True)
    baselines.add_argument("--discopatch-root", required=True)
    baselines.add_argument("--limit", default=None, type=positive_int)
    baselines.add_argument("--device", default="cuda:0")
    baselines.add_argument("--batch-size", default=32, type=positive_int)
    baselines.add_argument("--workers", default=9, type=nonnegative_int)
    baselines.add_argument("--epochs", default=65, type=positive_int)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "baselines-coco":
            from pathlib import Path
            from .baselines.pipeline import Settings, run_phase
            run_phase(args.phase, Settings(
                output=Path(args.output), checkpoint=Path(args.checkpoint),
                train_images=Path(args.coco_train_images), val_images=Path(args.coco_val_images),
                annotations=Path(args.coco_annotations), discopatch_root=Path(args.discopatch_root),
                limit=args.limit, device=args.device, batch_size=args.batch_size,
                workers=args.workers, epochs=args.epochs,
            ))
        else:
            run_coco_benchmark(
                args.checkpoint, args.coco_train_images, args.coco_val_images, args.output,
                reference_count=args.reference_count, evaluation_count=args.evaluation_count,
                device=args.device, batch_size=args.batch_size, seed=args.seed,
            )
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
