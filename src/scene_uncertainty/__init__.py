"""Scene-uncertainty tools, with inference runtime exports loaded only on demand."""

from __future__ import annotations

__all__ = ["checkpoint_sha256", "load_frozen_detector"]


def __getattr__(name: str):
    if name in __all__:
        from . import runtime

        value = getattr(runtime, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
