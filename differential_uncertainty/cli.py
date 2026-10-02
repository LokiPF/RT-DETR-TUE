from __future__ import annotations

import argparse
import sys


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
    parser = argparse.ArgumentParser(description="COCO baselines and the conv-TU phases")
    commands = parser.add_subparsers(dest="command", required=True)
    baselines = commands.add_parser("baselines-coco")
    baselines.add_argument("--phase", required=True,
                           choices=["sanity", "bank", "test", "train-discopatch", "discopatch-scores",
                                    "hashemi-fit", "cdf-fit", "cdf-zstats", "activation-scores", "timing", "report",
                                    "convtu-calibrate", "convtu-bank", "convtu-zstats", "convtu-scores",
                                    "convtu-channels", "convtu-means",
                                    "convtu-conditioned-report"])
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
        from pathlib import Path
        from .baselines.pipeline import Settings, run_phase
        run_phase(args.phase, Settings(
            output=Path(args.output), checkpoint=Path(args.checkpoint),
            train_images=Path(args.coco_train_images), val_images=Path(args.coco_val_images),
            annotations=Path(args.coco_annotations), discopatch_root=Path(args.discopatch_root),
            limit=args.limit, device=args.device, batch_size=args.batch_size,
            workers=args.workers, epochs=args.epochs,
        ))
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
