from dataclasses import FrozenInstanceError, replace

import pytest

from differential_uncertainty import __version__
from differential_uncertainty.config import BLUR_RADII, FIXED_CONFIG, ExperimentConfig


def test_package_version_is_one_point_zero_point_zero():
    assert __version__ == "1.0.0"


def test_public_configuration_contains_every_approved_fixed_value():
    assert FIXED_CONFIG.image_size == (640, 640)
    assert FIXED_CONFIG.class_count == 80
    assert FIXED_CONFIG.query_count == 300
    assert FIXED_CONFIG.persistence_layer == 2
    assert FIXED_CONFIG.persistence_dim == 335
    assert FIXED_CONFIG.bank_capacity == 25_000
    assert FIXED_CONFIG.bank_seed == 44
    assert FIXED_CONFIG.k == 5
    assert FIXED_CONFIG.bank_chunk_size == 8_192
    assert FIXED_CONFIG.reference_decile == 9
    assert FIXED_CONFIG.responsive_decile == 5
    assert FIXED_CONFIG.persistence_orientation == 1
    assert FIXED_CONFIG.confidence_orientation == -1
    assert FIXED_CONFIG.raw_responsive_orientation == 1
    assert FIXED_CONFIG.raw_reference_orientation == -1
    assert FIXED_CONFIG.bootstrap_samples == 10_000
    assert FIXED_CONFIG.bootstrap_seed == 20_260_821
    assert FIXED_CONFIG.blur_radii == (0.0, 1.0, 2.0, 4.0, 8.0, 12.0)
    assert FIXED_CONFIG.blur_radii is BLUR_RADII


def test_public_configuration_is_frozen():
    with pytest.raises(FrozenInstanceError):
        FIXED_CONFIG.k = 3


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"blur_radii": (0.0, 1.0)}, "six severities"),
        ({"blur_radii": (1.0, 2.0, 4.0, 8.0, 12.0, 16.0)}, "beginning with clean"),
        ({"reference_decile": -1}, "ten deciles"),
        ({"reference_decile": 10}, "ten deciles"),
        ({"responsive_decile": -1}, "ten deciles"),
        ({"responsive_decile": 10}, "ten deciles"),
        ({"reference_decile": 5}, "must differ"),
        ({"bank_capacity": 4}, "bank_capacity"),
        ({"k": 0}, "k must be positive"),
        ({"bank_chunk_size": 0}, "bank_chunk_size"),
        ({"bootstrap_samples": 0}, "bootstrap_samples"),
        ({"query_count": 9}, "ten deciles"),
        ({"persistence_dim": 0}, "persistence_dim"),
        ({"persistence_orientation": 0}, "orientation"),
        ({"confidence_orientation": 2}, "orientation"),
        ({"raw_responsive_orientation": -2}, "orientation"),
        ({"raw_reference_orientation": 3}, "orientation"),
    ],
)
def test_invalid_internal_configuration_is_rejected(changes, message):
    with pytest.raises(ValueError, match=message):
        replace(FIXED_CONFIG, **changes)


def test_small_configuration_is_available_only_as_a_python_test_helper():
    small = ExperimentConfig.for_tests(
        bank_capacity=20,
        k=2,
        query_count=20,
        persistence_dim=7,
    )

    assert small.bank_capacity == 20
    assert small.k == 2
    assert small.query_count == 20
    assert small.persistence_dim == 7
    assert small.blur_radii == FIXED_CONFIG.blur_radii
    assert FIXED_CONFIG.bank_capacity == 25_000


def test_scientific_dict_serializes_the_complete_scientific_contract():
    assert FIXED_CONFIG.scientific_dict() == {
        "image_size": [640, 640],
        "class_count": 80,
        "query_count": 300,
        "persistence_layer": 2,
        "persistence_dim": 335,
        "bank_capacity": 25_000,
        "bank_seed": 44,
        "k": 5,
        "bank_chunk_size": 8_192,
        "reference_decile": 9,
        "responsive_decile": 5,
        "orientations": {
            "persistence_relative_gap": 1,
            "confidence_relative_gap": -1,
            "raw_responsive": 1,
            "raw_reference": -1,
        },
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 20_260_821,
        "default_blur_radii": [0.0, 1.0, 2.0, 4.0, 8.0, 12.0],
        "feature_normalization": "raw",
    }
