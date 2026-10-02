import numpy as np
import pytest
import torch

from degradation_monitor.method.statistics import KEYS, STATISTICS, TOP_FRACTION, channel_statistics


def test_channel_statistics_give_the_mean_and_the_top_one_percent_mean_of_each_channel():
    x = torch.zeros(1, 2, 10, 10)
    x[0, 0] = -torch.arange(100, dtype=torch.float32).view(10, 10)  # |x| = 0..99; the top 1% is the single 99
    x[0, 1] = 2.0
    stats = channel_statistics(x)
    assert set(stats) == set(STATISTICS) == {"means", "top"} and TOP_FRACTION == 0.01
    assert stats["means"].dtype == np.float32 and stats["means"].shape == (1, 2)
    assert stats["means"][0].tolist() == pytest.approx([49.5, 2.0])
    assert stats["top"][0].tolist() == pytest.approx([99.0, 2.0])
    assert KEYS == tuple(f"{s}_s{l}" for s in ("means", "top") for l in range(1, 5))
    with pytest.raises(ValueError):
        channel_statistics(torch.zeros(2, 10, 10))
