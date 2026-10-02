import numpy as np
import pytest
from PIL import Image, ImageFilter

from degradation_monitor import corruptions as protocol
from degradation_monitor.corruptions import apply_imagecorruption


def _image(width=64, height=48, mode="RGB"):
    pixels = np.random.default_rng(0).integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB").convert(mode)


def test_conditions_are_clean_then_19_families_by_5_severities_in_package_order():
    from imagecorruptions import get_corruption_names
    assert protocol.FAMILIES == tuple(get_corruption_names("all"))
    assert protocol.COMMON_FAMILIES == tuple(get_corruption_names("common"))
    assert protocol.CONDITIONS[0] == ("clean", 0)
    assert len(protocol.CONDITIONS) == 96
    assert protocol.CONDITIONS[1:6] == tuple(("gaussian_noise", s) for s in range(1, 6))


def test_corrupt_is_seeded_per_image_and_restores_global_numpy_state():
    image = _image()
    np.random.seed(123)
    expected_next = np.random.rand()
    np.random.seed(123)
    first = np.asarray(protocol.corrupt(image, "a.jpg", "gaussian_noise", 3))
    assert np.random.rand() == expected_next
    again = np.asarray(protocol.corrupt(image, "a.jpg", "gaussian_noise", 3))
    other = np.asarray(protocol.corrupt(image, "b.jpg", "gaussian_noise", 3))
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_every_condition_keeps_size_and_returns_rgb_for_grayscale_input():
    arrays = protocol.variants(_image(mode="L"), "gray.jpg")
    assert len(arrays) == 96
    assert all(a.shape == (48, 64, 3) and a.dtype == np.uint8 for a in arrays)


def test_gaussian_blur_uses_the_imagecorruptions_package_not_the_old_pil_blur():
    image = _image()
    ours = np.asarray(protocol.corrupt(image, "a.jpg", "gaussian_blur", 5))
    assert np.array_equal(ours, np.asarray(apply_imagecorruption(image, "gaussian_blur", 5)))
    old_pil_blur = np.asarray(image.convert("RGB").filter(ImageFilter.GaussianBlur(12)))  # the archived level-5 blur
    assert not np.array_equal(ours, old_pil_blur)


@pytest.mark.parametrize("family, severity", [("rain", 3), ("fog", 0), ("fog", 6), ("fog", 2.0)])
def test_corrupt_rejects_unknown_family_or_severity(family, severity):
    with pytest.raises(ValueError):
        protocol.corrupt(_image(), "a.jpg", family, severity)


def test_load_variants_returns_file_name_and_stable_digests(tmp_path):
    path = tmp_path / "img.png"
    _image().save(path)
    name, arrays = protocol.load_variants(path)
    assert name == "img.png" and len(arrays) == 96
    assert protocol.digest(arrays[0]) == protocol.digest(arrays[0].copy())
    assert protocol.digest(arrays[0]) != protocol.digest(arrays[1])
