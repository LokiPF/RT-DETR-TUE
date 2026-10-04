import numpy as np
import pytest
import torch

from degradation_monitor.baselines import hashemi


def test_streaming_stats_match_numpy_even_with_a_large_offset():
    rng = np.random.default_rng(0)
    data = 1e6 + rng.normal(0, 1e-2, size=(257, 3, 4))
    stats = hashemi.NeuronStats()
    for start in range(0, 257, 50):
        stats.update(torch.from_numpy(data[start:start + 50]))
    mean, std = stats.result()
    assert mean.dtype == np.float32 and std.dtype == np.float32
    assert np.allclose(mean, data.mean(axis=0), rtol=1e-6, atol=0)
    assert np.allclose(std, data.std(axis=0), rtol=1e-4, atol=0)


def test_stats_need_two_images_and_a_fixed_neuron_shape():
    stats = hashemi.NeuronStats()
    stats.update(torch.zeros(1, 2))
    with pytest.raises(ValueError, match="two images"):
        stats.result()
    with pytest.raises(ValueError, match="shape"):
        stats.update(torch.zeros(1, 3))


def test_outside_counts_use_a_strict_k_sigma_band_and_keep_constant_neurons_inside():
    mean = torch.tensor([0.0, 5.0, 1.0])
    std = torch.tensor([1.0, 0.0, 0.5])
    values = torch.tensor([[2.0, 5.0, 1.0],     # on the band edge, at the constant, at the mean: all inside
                           [2.1, 5.001, 2.1]])  # beyond the edge, off the constant, beyond the edge
    assert hashemi.outside_counts(values, mean, std, k=2.0).tolist() == [0, 3]


def test_outside_counts_refuse_activations_of_another_shape():
    with pytest.raises(ValueError, match="do not match"):
        hashemi.outside_counts(torch.zeros(2, 4), torch.zeros(3), torch.ones(3))


def _fitted(tmp_path, rng):
    shapes = {"decoder": (300, 256), "encoder_s8": (4, 4, 4), "encoder_s16": (4, 2, 2), "encoder_s32": (4, 1, 1)}
    stats = {name: hashemi.NeuronStats() for name in hashemi.LAYERS}
    for name, shape in shapes.items():
        stats[name].update(torch.from_numpy(rng.normal(size=(50, *shape))))
    path = tmp_path / "intervals.npz"
    hashemi.save_intervals(path, stats, images=50)
    return path, shapes


def test_monitor_scores_each_image_alone_whatever_its_batch(tmp_path):
    rng = np.random.default_rng(1)
    path, shapes = _fitted(tmp_path, rng)
    monitor = hashemi.HashemiMonitor(path, "cpu")
    decoder = torch.from_numpy(rng.normal(size=(3, *shapes["decoder"]))).float()
    decoder[2] += 10.0                                    # far outside every interval
    encoder = [torch.from_numpy(rng.normal(size=(3, *shapes[n]))).float() for n in hashemi.LAYERS[1:]]
    both_dec, both_enc = monitor.scores(decoder, encoder)
    alone_dec, alone_enc = monitor.scores(decoder[:1], [e[:1] for e in encoder])
    assert both_dec[0] == pytest.approx(alone_dec[0]) and both_enc[0] == pytest.approx(alone_enc[0])
    assert both_dec[2] == pytest.approx(1.0) and 0.0 < both_dec[0] < 0.2
    assert np.all((both_enc >= 0) & (both_enc <= 1))
    assert monitor.decoder_share(decoder[:1])[0] == pytest.approx(alone_dec[0])


def test_monitor_pools_encoder_neurons_and_needs_all_three_maps(tmp_path):
    rng = np.random.default_rng(2)
    path, shapes = _fitted(tmp_path, rng)
    monitor = hashemi.HashemiMonitor(path, "cpu")
    decoder = torch.zeros(1, *shapes["decoder"])
    encoder = [torch.zeros(1, *shapes[n]) for n in hashemi.LAYERS[1:]]
    encoder[0] += 100.0                                   # only the stride-8 map (64 of 84 neurons) is off
    _, share = monitor.scores(decoder, encoder)
    assert share[0] == pytest.approx(64 / 84)
    with pytest.raises(ValueError, match="three"):
        monitor.scores(decoder, encoder[:2])


def test_encoder_shares_per_map_weight_back_to_the_pooled_share(tmp_path):
    rng = np.random.default_rng(4)
    path, shapes = _fitted(tmp_path, rng)
    monitor = hashemi.HashemiMonitor(path, "cpu")
    encoder = [torch.from_numpy(rng.normal(size=(2, *shapes[n]))).float() for n in hashemi.LAYERS[1:]]
    encoder[1] += 5.0                                    # every stride-16 neuron is far outside
    per_map = monitor.encoder_shares(encoder)
    _, pooled = monitor.scores(torch.zeros(2, *shapes["decoder"]), encoder)
    sizes = np.array([64, 16, 4])
    assert per_map.shape == (2, 3) and np.allclose(per_map[:, 1], 1.0)
    assert np.allclose((per_map * sizes).sum(axis=1) / sizes.sum(), pooled)


def test_hashemi_saves_and_reads_only_the_decoder(tmp_path):
    stats = hashemi.NeuronStats()
    stats.update(torch.zeros(3, 4, 2))
    stats.update(torch.ones(3, 4, 2))
    hashemi.save_intervals(tmp_path / "intervals.npz", {"decoder": stats}, images=6)
    monitor = hashemi.HashemiMonitor(tmp_path / "intervals.npz", "cpu")
    assert set(monitor.stats) == {"decoder"}
    assert monitor.decoder_share(torch.full((2, 4, 2), 10.0)).tolist() == [1.0, 1.0]
