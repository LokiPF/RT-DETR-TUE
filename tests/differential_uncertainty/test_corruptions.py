import sys
import types

import pytest
from PIL import Image, ImageFilter

from differential_uncertainty.corruptions import CORRUPTION_NAMES, apply_corruption


def test_corruption_roster_is_the_fixed_nineteen_family_order():
    assert CORRUPTION_NAMES == (
        "gaussian_blur", "gaussian_noise", "shot_noise", "impulse_noise",
        "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "snow",
        "frost", "fog", "brightness", "contrast", "elastic_transform",
        "pixelate", "jpeg_compression", "speckle_noise", "spatter", "saturate",
    )


@pytest.mark.parametrize(("level", "radius"), [(4, 8), (5, 12)])
def test_gaussian_blur_uses_the_retained_radius_at_each_level(level, radius):
    image = Image.effect_noise((31, 31), 60).convert("RGB")
    actual = apply_corruption(image, "gaussian_blur", level)
    expected = image.filter(ImageFilter.GaussianBlur(radius=radius))
    assert actual.tobytes() == expected.tobytes()


def test_other_families_delegate_to_imagecorruptions(monkeypatch):
    calls = []
    package = types.ModuleType("imagecorruptions")
    package.__path__ = []
    package.corrupt = lambda pixels, corruption_name, severity: (
        calls.append((pixels.shape, corruption_name, severity)) or pixels
    )
    upstream = types.ModuleType("imagecorruptions.corruptions")
    upstream.np = __import__("numpy")
    upstream.gaussian = lambda image, multichannel=None: image
    monkeypatch.setitem(sys.modules, "imagecorruptions", package)
    monkeypatch.setitem(sys.modules, "imagecorruptions.corruptions", upstream)
    image = Image.new("RGB", (7, 5), "red")
    assert apply_corruption(image, "gaussian_noise", 4).tobytes() == image.tobytes()
    assert calls == [((5, 7, 3), "gaussian_noise", 4)]


@pytest.mark.parametrize("level", [0, 1, 2, 3, 6])
def test_only_levels_four_and_five_are_accepted(level):
    with pytest.raises(ValueError, match="severity"):
        apply_corruption(Image.new("RGB", (5, 5)), "gaussian_blur", level)
