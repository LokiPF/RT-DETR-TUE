from dataclasses import FrozenInstanceError

import pytest
from PIL import Image, ImageChops

from differential_uncertainty.config import BLUR_RADII
from differential_uncertainty.corruptions import GaussianBlur, Severity


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
