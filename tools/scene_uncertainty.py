#!/usr/bin/env python
"""Run the scene-uncertainty pipeline: `python tools/scene_uncertainty.py <command> --help`."""

from src.scene_uncertainty.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
