import numpy as np
import pytest
import torch

from differential_uncertainty.convtu import features, graph


def test_layer_specs_give_one_percent_of_the_nodes_for_the_real_layers():
    inputs = [torch.empty(1, c, h, h) for c, h in ((64, 160), (128, 80), (256, 40), (512, 20))]
    kernels = [torch.empty(c, c, 3, 3) for c in (64, 128, 256, 512)]
    specs = features.layer_specs(inputs, kernels)
    assert [s.name for s in specs] == ["s1", "s2", "s3", "s4"]
    assert [s.nodes for s in specs] == [3_276_800, 1_638_400, 819_200, 409_600]
    assert [s.k for s in specs] == [32_768, 16_384, 8_192, 4_096]


def test_layer_features_hold_the_fingerprint_and_the_three_controls():
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(3, 5, 6, generator=generator)
    x[x < 0.3] = 0.0
    kernel = torch.rand(2, 3, 3, 3, generator=generator)
    spec = features.LayerSpec(1, 3, 2, 5, 6)  # 150 nodes: a small graph keeps all 149 values
    got = features.layer_features(x, kernel, spec, cut=0.2)
    top = graph.conv_top_merges(x, kernel, 149, 0.2)
    np.testing.assert_array_equal(got["mst"], top.values)
    np.testing.assert_array_equal(got["edges"], top.heaviest)
    expected_acts = np.zeros(149, dtype=np.float32)
    expected_acts[:90] = np.sort(x.flatten().numpy())[::-1]
    np.testing.assert_array_equal(got["acts"], expected_acts)
    np.testing.assert_allclose(got["means"], x.mean(dim=(1, 2)).numpy())
    assert got["tau_k"] == float(top.values[-1]) and got["rounds"] >= 1 and got["gathered"] >= got["read"] > 0


def test_a_blank_input_gives_all_zero_features():
    spec = features.LayerSpec(1, 3, 2, 5, 6)
    got = features.layer_features(torch.zeros(3, 5, 6), torch.rand(2, 3, 3, 3), spec, cut=0.1)
    assert all(not got[key].any() for key in ("mst", "edges", "acts", "means"))


def test_knn_scores_are_mean_euclidean_distances_to_the_nearest_rows():
    rng = np.random.default_rng(1)
    queries, bank = rng.normal(size=(7, 4)), rng.normal(size=(20, 4))
    distances = np.sqrt(((queries[:, None, :] - bank[None, :, :]) ** 2).sum(axis=2))
    expected = np.sort(distances, axis=1)[:, :5].mean(axis=1)
    got = features.knn_scores(torch.from_numpy(queries), torch.from_numpy(bank), chunk=3)
    np.testing.assert_allclose(got, expected, rtol=1e-5)
    with pytest.raises(ValueError, match="neighbours"):
        features.knn_scores(torch.from_numpy(queries), torch.from_numpy(bank[:4]))
