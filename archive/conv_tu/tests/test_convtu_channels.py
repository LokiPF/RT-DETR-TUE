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


def _clean_and_test(rng, layers=("s1", "s2"), dims=(3, 4)):
    bank, zstats, test = {}, {}, {}
    for statistic in channels.STATISTICS:
        for layer, dim in zip(layers, dims):
            width = 16 * dim if statistic == "grid" else dim
            key = f"{statistic}_{layer}"
            bank[key] = rng.normal(size=(40, width))
            zstats[key] = rng.normal(size=(12, width))
            test[key] = np.stack([zstats[key], zstats[key] + 10.0])  # (2 images, 12 conditions, width)
    return bank, zstats, test


def test_channel_method_scores_compare_every_statistic_both_ways():
    bank, zstats, test = _clean_and_test(np.random.default_rng(0))
    summed, per_layer = channels.channel_method_scores(bank, zstats, test, ("s1", "s2"))
    expected = {channels.method_name(s, c) for s in channels.STATISTICS for c in channels.COMPARISONS}
    assert set(summed) == set(per_layer) == expected and set(channels.LABELS) == expected
    for name in expected:
        assert summed[name].shape == (2, 12) and per_layer[name].shape == (2, 12, 2)
        # image 0 is the z-statistics images themselves, so its layer z-scores average to 0
        assert abs(summed[name][0].mean()) < 1e-9
        # image 1 is shifted far from every clean row, so it scores higher everywhere
        assert (summed[name][1] > summed[name][0].max()).all()


def test_channel_method_scores_can_score_only_the_means():
    bank, zstats, test = _clean_and_test(np.random.default_rng(3))
    full, full_layers = channels.channel_method_scores(bank, zstats, test, ("s1", "s2"))
    only = {key: value for key, value in test.items() if key.startswith("means_")}
    summed, per_layer = channels.channel_method_scores(bank, zstats, only, ("s1", "s2"), statistics=("means",))
    assert set(summed) == set(per_layer) == {"ch_means_knn", "ch_means_own"}
    for name in summed:
        np.testing.assert_array_equal(summed[name], full[name])
        np.testing.assert_array_equal(per_layer[name], full_layers[name])


def test_own_comparison_matches_the_per_dimension_formula():
    bank, zstats, test = _clean_and_test(np.random.default_rng(1))
    _, per_layer = channels.channel_method_scores(bank, zstats, test, ("s1", "s2"))
    mean, std = channels.fit_own_average(bank["top_s2"])
    np.testing.assert_allclose(per_layer["ch_top_own"][1, :, 1],
                               channels.own_average_scores(test["top_s2"][1], mean, std))


def test_floored_dimensions_are_counted_and_can_be_left_out():
    bank, zstats, test = _clean_and_test(np.random.default_rng(2))
    for clean in (bank, zstats):
        clean["top_s1"][:, 0] = 0.0       # a dimension dead on every clean image
    test["top_s1"][:, :, 0] = 0.0
    lit = {key: value.copy() for key, value in test.items()}
    lit["top_s1"][1, :, 0] = 50.0         # ... that a corruption lights up
    counts = channels.floored_counts(bank, ("s1", "s2"))
    assert counts["top_s1"] == 1 and counts["top_s2"] == 0 and counts["means_s1"] == 0
    _, kept = channels.channel_method_scores(bank, zstats, lit, ("s1", "s2"))
    _, dropped = channels.channel_method_scores(bank, zstats, lit, ("s1", "s2"), drop_floored=True)
    _, quiet = channels.channel_method_scores(bank, zstats, test, ("s1", "s2"), drop_floored=True)
    # kept, the floored dimension dominates the layer score; left out, it cannot move it at all
    assert (kept["ch_top_own"][1, :, 0] > quiet["ch_top_own"][1, :, 0] + 10).all()
    np.testing.assert_allclose(dropped["ch_top_own"][1, :, 0], quiet["ch_top_own"][1, :, 0])
