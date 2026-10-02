from __future__ import annotations

import inspect

import numpy as np
from PIL import Image


class _NumpyCompatibility:
    def __getattr__(self, name: str):
        if name == "float_":
            return np.float64
        return getattr(np, name)


def _prepare_compatibility() -> None:
    import imagecorruptions.corruptions as upstream

    if not hasattr(upstream.np, "float_"):
        upstream.np = _NumpyCompatibility()
    if "multichannel" not in inspect.signature(upstream.gaussian).parameters:
        gaussian = upstream.gaussian

        def compatible_gaussian(image, *args, multichannel=None, **kwargs):
            if multichannel:
                kwargs["channel_axis"] = -1
            return gaussian(image, *args, **kwargs)

        upstream.gaussian = compatible_gaussian

    random_noise = getattr(getattr(upstream, "sk", None), "util", None)
    random_noise = getattr(random_noise, "random_noise", None)
    if random_noise is not None and not getattr(random_noise, "_fixed_legacy_rng", False):
        def compatible_random_noise(image, mode="gaussian", rng=None, clip=True, **kwargs):
            if rng is None:
                rng = int(np.random.randint(0, 2**32))
            return random_noise(image, mode=mode, rng=rng, clip=clip, **kwargs)

        compatible_random_noise._fixed_legacy_rng = True
        upstream.sk.util.random_noise = compatible_random_noise


def apply_imagecorruption(image: Image.Image, name: str, severity: int) -> Image.Image:
    from imagecorruptions import corrupt

    _prepare_compatibility()
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    output = corrupt(pixels, corruption_name=name, severity=severity)
    return Image.fromarray(np.asarray(output, dtype=np.uint8), mode="RGB")
