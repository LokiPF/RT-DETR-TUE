import numpy as np
import pytest

from degradation_monitor.baselines.saod import saod_uncertainty


def test_saod_uncertainty_averages_one_minus_confidence_of_the_m_best():
    top = np.array([0.1, 0.9, 0.8])
    assert saod_uncertainty(top, 1) == pytest.approx(0.1)
    assert saod_uncertainty(top, 3) == pytest.approx((0.1 + 0.2 + 0.9) / 3)
    for bad in (0, 4):
        with pytest.raises(ValueError):
            saod_uncertainty(top, bad)
