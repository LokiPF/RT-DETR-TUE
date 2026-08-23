#!/usr/bin/env python
"""Run the scene-uncertainty pipeline: `python tools/scene_uncertainty.py <command> --help`."""

import sys
from pathlib import Path

if sys.argv[1:2] == ["analyze-within-image-contrast"]:
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository / "src"))
    from scene_uncertainty.cli import main
else:
    from src.scene_uncertainty.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
