import numpy as np
import pytest
import torch

from differential_uncertainty.baselines import activation_cdf as cdf


def test_ranges_widen_by_the_margin_and_survive_a_constant_channel():
    ranges = cdf.ChannelRanges()
    values = torch.zeros(2, 2, 3, 3)
    values[0, 0] = 10.0            # channel 0 spans [0, 10] over the two images
    values[:, 1] = 3.0             # channel 1 is constant
    ranges.update(values[:1])
    ranges.update(values[1:])
    low, high = ranges.result()
    assert low.tolist() == pytest.approx([-2.0, 2.8]) and high.tolist() == pytest.approx([12.0, 3.2])


def test_histograms_count_every_activation_and_clamp_to_the_edge_bins():
    values = torch.tensor([0.1, 0.3, 0.6, 0.99, -5.0, 7.0]).view(1, 1, 2, 3)
    counts = cdf.channel_histograms(values, torch.tensor([0.0]), torch.tensor([1.0]), bins=4)
    assert counts.shape == (1, 1, 4) and counts[0, 0].tolist() == [2.0, 1.0, 1.0, 2.0]


def test_emd_is_zero_for_the_reference_and_one_bin_width_per_shifted_channel():
    reference = torch.ones(2, 4)                                    # all mass in bin 0, two channels
    same = torch.tensor([[[5.0, 0, 0, 0], [2.0, 0, 0, 0]]])
    shifted = torch.tensor([[[0, 5.0, 0, 0], [2.0, 0, 0, 0]]])      # channel 0 moved by one bin of four
    assert cdf.emd_to_reference(same, reference).item() == pytest.approx(0.0)
    assert cdf.emd_to_reference(shifted, reference).item() == pytest.approx(0.25)


SHAPES = [(2, 4, 4), (2, 4, 4), (3, 2, 2), (3, 2, 2), (4, 1, 1)]


def _stages(rng, n, shift=0.0):
    return [torch.from_numpy(rng.normal(shift, 1.0, size=(n, *s))).float() for s in SHAPES]


def _reference(tmp_path, rng):
    training = [_stages(rng, 20) for _ in range(3)]
    ranges = {stage: cdf.ChannelRanges() for stage in cdf.STAGES}
    for batch in training:
        for stage, values in zip(cdf.STAGES, batch):
            ranges[stage].update(values)
    reference = cdf.ReferenceHistograms({s: r.result() for s, r in ranges.items()}, "cpu", bins=50)
    for batch in training:
        reference.update(batch)
    path = tmp_path / "reference.npz"
    reference.save(path, images=60)
    return path


def test_monitor_scores_shifted_images_higher_and_each_image_alone(tmp_path):
    rng = np.random.default_rng(0)
    monitor = cdf.CdfMonitor(_reference(tmp_path, rng), "cpu")
    clean, shifted = _stages(rng, 4), _stages(rng, 4, shift=2.0)
    clean_scores, shifted_scores = monitor.scores(clean), monitor.scores(shifted)
    assert shifted_scores.min() > clean_scores.max()
    assert monitor.scores([s[:1] for s in clean])[0] == pytest.approx(clean_scores[0])


def test_monitor_needs_all_five_backbone_stages_with_the_fitted_channels(tmp_path):
    monitor = cdf.CdfMonitor(_reference(tmp_path, np.random.default_rng(1)), "cpu")
    stages = _stages(np.random.default_rng(2), 1)
    with pytest.raises(ValueError, match="five"):
        monitor.scores(stages[:4])
    stages[0] = torch.zeros(1, 7, 4, 4)
    with pytest.raises(ValueError, match="channels"):
        monitor.scores(stages)
