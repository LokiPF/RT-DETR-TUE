import numpy as np
import pytest
import torch

from differential_uncertainty.convtu import channels


def test_channel_statistics_on_an_8_by_8_map():
    x = -torch.arange(2 * 8 * 8, dtype=torch.float32).reshape(1, 2, 8, 8)  # negative: statistics use |x|
    stats = channels.channel_statistics(x)
    values = np.abs(x.numpy())[0]
    np.testing.assert_allclose(stats["means"][0], values.mean(axis=(1, 2)))
    # 64 cells: k = max(1, round(0.64)) = 1, so top and p99 are the maximum
    np.testing.assert_allclose(stats["top"][0], values.max(axis=(1, 2)))
    np.testing.assert_allclose(stats["p99"][0], values.max(axis=(1, 2)))
    blocks = values.reshape(2, 4, 2, 4, 2).mean(axis=(2, 4)).reshape(-1)  # 2 x 2 blocks on a 4 x 4 grid
    np.testing.assert_allclose(stats["grid"][0], blocks)
    assert {key: value.shape for key, value in stats.items()} == {
        "means": (1, 2), "top": (1, 2), "p99": (1, 2), "grid": (1, 32)}


def test_top_and_p99_use_the_largest_one_percent():
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(3, 4, 20, 20, generator=generator)  # 400 cells: k = 4
    stats = channels.channel_statistics(x)
    ordered = np.sort(x.numpy().reshape(3, 4, -1), axis=2)[:, :, ::-1]
    np.testing.assert_allclose(stats["top"], ordered[:, :, :4].mean(axis=2), rtol=1e-6)
    np.testing.assert_allclose(stats["p99"], ordered[:, :, 3], rtol=1e-6)


def test_fit_own_average_floors_dead_dimensions():
    reference = np.array([[1.0, 5.0, 0.0, 0.0, 0.0],
                          [3.0, 9.0, 0.0, 0.0, 0.0],
                          [2.0, 7.0, 0.0, 0.0, 0.0]])  # three of five dimensions are dead
    mean, std = channels.fit_own_average(reference)
    np.testing.assert_allclose(mean, [2.0, 7.0, 0.0, 0.0, 0.0])
    positive = np.array([np.std([1.0, 3.0, 2.0]), np.std([5.0, 9.0, 7.0])])
    floor = 0.01 * np.median(positive)
    np.testing.assert_allclose(std, [positive[0], positive[1], floor, floor, floor])
    with pytest.raises(ValueError, match="no spread"):
        channels.fit_own_average(np.zeros((4, 3)))


def test_own_average_scores_are_mean_absolute_z():
    mean, std = np.array([1.0, 2.0]), np.array([0.5, 4.0])
    values = np.array([[1.0, 2.0], [2.0, -2.0]])
    np.testing.assert_allclose(channels.own_average_scores(values, mean, std), [0.0, (2.0 + 1.0) / 2])
