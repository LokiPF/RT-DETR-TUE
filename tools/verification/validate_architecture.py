#!/usr/bin/env python3
"""Build an RT-DETR model from YAML and run one random forward pass."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, help="Path to the YAML config")
    parser.add_argument(
        "--project-root",
        default=".",
        help="RT-DETR project root containing src/ (default: current directory)",
    )
    parser.add_argument("--checkpoint", help="Optional .pth checkpoint")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a device such as cuda:1",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Handle the common RT-DETR checkpoint layouts."""
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected a checkpoint dictionary, got {type(checkpoint).__name__}"
        )

    if "ema" in checkpoint:
        state = checkpoint["ema"]
        if isinstance(state, dict) and "module" in state:
            state = state["module"]
    elif "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint

    if not isinstance(state, dict):
        raise TypeError("Could not find a state dictionary in the checkpoint")
    return state


def describe(value: Any, name: str = "output", indent: int = 0) -> None:
    prefix = " " * indent
    if isinstance(value, torch.Tensor):
        finite = (
            torch.isfinite(value).sum().item() if value.is_floating_point() else "n/a"
        )
        print(
            f"{prefix}{name}: Tensor(shape={tuple(value.shape)}, "
            f"dtype={value.dtype}, device={value.device}, finite={finite})"
        )
    elif isinstance(value, dict):
        print(f"{prefix}{name}: dict({len(value)})")
        for key, child in value.items():
            describe(child, str(key), indent + 2)
    elif isinstance(value, (list, tuple)):
        print(f"{prefix}{name}: {type(value).__name__}({len(value)})")
        for index, child in enumerate(value):
            describe(child, str(index), indent + 2)
    else:
        print(f"{prefix}{name}: {value!r}")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    config_path = Path(args.config).resolve()

    if not (project_root / "src").is_dir():
        raise FileNotFoundError(
            f"{project_root} does not contain src/. Run from the RT-DETR root "
            "or pass --project-root."
        )
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    sys.path.insert(0, str(project_root))
    from src.core import YAMLConfig  # pylint: disable=import-outside-toplevel

    torch.manual_seed(args.seed)
    device = select_device(args.device)

    print(f"Loading config: {config_path}")
    cfg = YAMLConfig(str(config_path))
    model = cfg.model

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).resolve()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        incompatible = model.load_state_dict(
            extract_state_dict(checkpoint), strict=False
        )
        print(
            "Checkpoint loaded: "
            f"missing_keys={len(incompatible.missing_keys)}, "
            f"unexpected_keys={len(incompatible.unexpected_keys)}"
        )

    model = model.to(device).eval()
    random_input = torch.rand(
        args.batch_size,
        3,
        args.height,
        args.width,
        device=device,
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model: {type(model).__name__} ({parameter_count:,} parameters)")
    print(f"Input: shape={tuple(random_input.shape)}, device={device}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    with torch.inference_mode():
        output = model(random_input)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    print(f"Forward pass succeeded in {elapsed:.3f} seconds")
    describe(output)


if __name__ == "__main__":
    main()
