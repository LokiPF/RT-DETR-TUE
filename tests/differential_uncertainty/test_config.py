from dataclasses import FrozenInstanceError, fields

import pytest

from differential_uncertainty.config import FIXED_CONFIG, ExperimentConfig


def test_fixed_config_is_the_complete_fixed_method_contract():
    assert {field.name for field in fields(ExperimentConfig)} == {
        "image_size", "class_count", "query_count", "persistence_layer",
        "persistence_dim", "bank_capacity", "bank_confidence_threshold",
        "neighbors", "bank_chunk_size", "bootstrap_samples", "levels",
    }
    assert FIXED_CONFIG.bank_capacity == 2_000
    assert FIXED_CONFIG.bank_confidence_threshold == 0.5
    assert FIXED_CONFIG.neighbors == 5
    assert FIXED_CONFIG.persistence_layer == 2
    assert FIXED_CONFIG.levels == (4, 5)
    with pytest.raises(FrozenInstanceError):
        FIXED_CONFIG.neighbors = 3


def test_config_accepts_small_direct_test_overrides_and_validates_them():
    small = ExperimentConfig(
        class_count=2, query_count=3, persistence_dim=3, bank_capacity=5,
    )
    assert small.bank_capacity == 5
    with pytest.raises(ValueError, match="at least neighbors"):
        ExperimentConfig(bank_capacity=4, neighbors=5)
    with pytest.raises(ValueError, match="levels"):
        ExperimentConfig(levels=(0, 4, 5))
