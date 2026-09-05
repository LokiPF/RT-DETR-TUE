#!/usr/bin/env python3
"""Split the config's training set into disjoint Frechet-mean and calibration partitions.

Writes a JSON split file that CalibrationSolver.fit (frechet partition) and
CalibrationSolver.fit_conformal (calibration partition) consume via the config key
``train_calibration_split_path``. Point step 1 at the frechet partition and step 2
at the calibration partition so the conformal distances are held out from the
Frechet-mean fitting data.

Intended to live in rtdetrv2_pytorch/tools/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import YAMLConfig  # noqa: E402
from src.solver.tue_split import (  # noqa: E402
    class_histogram,
    save_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split the training set into disjoint frechet / calibration partitions."
    )
    parser.add_argument("--config", "-c", required=True, help="TUERTDETR YAML config.")
    parser.add_argument(
        "--output", "-o", default="train_calibration_split.json",
        help="Output JSON path for the split.",
    )
    parser.add_argument(
        "--calibration-fraction", type=float, default=0.2,
        help="Fraction of the training set held out for conformal calibration. Default 0.2.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Split RNG seed.")
    parser.add_argument(
        "--max-images", type=int, default=None,
        help=(
            "Optional cap on TOTAL images used (frechet + calibration). A uniform "
            "random subsample of this size is drawn before splitting, so both "
            "partitions stay representative. Omit to use the whole training set."
        ),
    )
    parser.add_argument(
        "--min-calibration-per-class", type=int, default=50,
        help="Warn about classes with fewer than this many calibration annotations.",
    )
    parser.add_argument(
        "--skip-class-report", action="store_true",
        help="Skip the per-class coverage report (faster; it loads all annotations).",
    )
    return parser.parse_args()


def report_coverage(dataset, payload, min_per_class: int) -> None:
    frechet_hist = class_histogram(dataset, payload["frechet_indices"])
    calibration_hist = class_histogram(dataset, payload["calibration_indices"])

    if frechet_hist is None or calibration_hist is None:
        print(
            "Class-coverage report skipped: dataset does not expose a COCO handle "
            "(.coco/.ids). Split still written."
        )
        return

    frechet_classes = {c for c, n in frechet_hist.items() if n > 0}
    calibration_classes = {c for c, n in calibration_hist.items() if n > 0}

    uncovered = sorted(frechet_classes - calibration_classes)
    sparse = sorted(
        c for c in calibration_classes
        if calibration_hist.get(c, 0) < min_per_class
    )

    print("\n--- calibration-split class coverage ---")
    print(f"categories in frechet partition:      {len(frechet_classes)}")
    print(f"categories in calibration partition:  {len(calibration_classes)}")
    if uncovered:
        print(
            f"WARNING: {len(uncovered)} categories present in the frechet partition have ZERO "
            f"calibration annotations: {uncovered}. Those (layer, class) buckets will have no "
            "conformal array, so their inference p-value is undefined (NaN)."
        )
    else:
        print("Every frechet-partition category also appears in the calibration partition.")
    if sparse:
        counts = {c: calibration_hist[c] for c in sparse}
        print(
            f"NOTE: {len(sparse)} categories have < {min_per_class} calibration annotations "
            f"(coarse p-value granularity ~1/(N+1)): {counts}"
        )


def main() -> None:
    args = parse_args()
    if not 0.0 < args.calibration_fraction < 1.0:
        raise ValueError("--calibration-fraction must be in (0, 1)")
    if args.max_images is not None and args.max_images < 2:
        raise ValueError("--max-images must be at least 2")

    config = YAMLConfig(args.config)
    loader = getattr(config, "train_dataloader", None)
    if loader is None:
        raise RuntimeError("The config has no train_dataloader; cannot split the training set.")
    dataset = loader.dataset

    payload = save_split(
        dataset=dataset,
        output_path=args.output,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
        max_items=args.max_images,
        metadata={"config": str(args.config)},
    )

    if payload["max_items"] is not None:
        print(
            f"Sampled {payload['pool_size']} of {payload['total']} training images "
            f"(cap={payload['max_items']}, seed={args.seed}):"
        )
    else:
        print(f"Split all {payload['total']} training images (seed={args.seed}):")
    print(f"  frechet partition:     {payload['num_frechet']} images")
    print(f"  calibration partition: {payload['num_calibration']} images")
    print(f"Saved split to {args.output}")

    if not args.skip_class_report:
        report_coverage(dataset, payload, args.min_calibration_per_class)

    print(
        "\nWire it up: set 'train_calibration_split_path: "
        f"{args.output}' in your calibration config, then run fit (means on the "
        "frechet partition) followed by fit_conformal (calibration on the disjoint partition)."
    )


if __name__ == "__main__":
    main()