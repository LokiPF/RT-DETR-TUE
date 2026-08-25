from dataclasses import FrozenInstanceError

import pytest
from PIL import Image, ImageChops

from differential_uncertainty.config import BLUR_RADII
from differential_uncertainty.corruptions import (
    GaussianBlur,
    ImageCorruption,
    Severity,
    additional_corruption_names,
    benchmark_corruptions,
)


ADDITIONAL_IMAGECORRUPTIONS = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
    "speckle_noise",
    "spatter",
    "saturate",
)


def test_severity_is_a_frozen_level_and_parameter_descriptor():
    severity = Severity(level=3, parameter=4.0)

    assert severity.level == 3
    assert severity.parameter == 4.0
    with pytest.raises(FrozenInstanceError):
        severity.level = 2


def test_gaussian_blur_declares_the_fixed_ladder():
    corruption = GaussianBlur()

    assert corruption.name == "gaussian_blur"
    assert corruption.severities == (
        Severity(0, 0.0),
        Severity(1, 1.0),
        Severity(2, 2.0),
        Severity(3, 4.0),
        Severity(4, 8.0),
        Severity(5, 12.0),
    )
    assert tuple(item.parameter for item in corruption.severities) == BLUR_RADII


def test_level_zero_is_identity_and_blur_is_deterministic():
    image = Image.effect_noise((21, 21), 90).convert("RGB")
    corruption = GaussianBlur()

    assert corruption.apply(image, 0) is image
    first = corruption.apply(image, 3)
    second = corruption.apply(image, 3)
    assert first.tobytes() == second.tobytes()
    assert ImageChops.difference(first, image).getbbox() is not None


@pytest.mark.parametrize("level", [-1, 6, 100])
def test_unknown_levels_are_rejected_instead_of_indexing_the_ladder(level):
    image = Image.new("RGB", (5, 5))

    with pytest.raises(ValueError, match="unknown gaussian_blur severity"):
        GaussianBlur().apply(image, level)


def test_imagecorruptions_matrix_has_legacy_blur_and_18_additions():
    assert additional_corruption_names() == ADDITIONAL_IMAGECORRUPTIONS
    assert tuple(item.name for item in benchmark_corruptions()) == (
        "gaussian_blur",
        *ADDITIONAL_IMAGECORRUPTIONS,
    )


def test_imagecorruptions_matrix_creates_fresh_corruptions():
    first = benchmark_corruptions()
    second = benchmark_corruptions()

    assert len(first) == len(second) == 19
    assert all(
        first_item is not second_item
        for first_item, second_item in zip(first, second, strict=True)
    )


@pytest.mark.parametrize("name", ADDITIONAL_IMAGECORRUPTIONS)
def test_imagecorruptions_adapter_has_clean_and_five_package_levels(name):
    image = Image.effect_noise((64, 64), 90).convert("RGB")
    corruption = ImageCorruption(name)

    assert corruption.severities == tuple(
        Severity(level, float(level)) for level in range(6)
    )
    assert corruption.apply(image, 0) is image

    result = corruption.apply(image, 3)

    assert result.mode == "RGB"
    assert result.size == image.size


def test_imagecorruptions_rejects_unknown_name():
    with pytest.raises(
        ValueError,
        match="unknown ImageCorruptions corruption",
    ):
        ImageCorruption("not_a_corruption")


@pytest.mark.parametrize("level", [-1, 6])
def test_imagecorruptions_reject_unknown_levels(level):
    with pytest.raises(
        ValueError,
        match=f"unknown imagecorruptions severity {level}",
    ):
        ImageCorruption("gaussian_noise").apply(Image.new("RGB", (5, 5)), level)
