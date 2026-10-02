import numpy as np
import pytest

from degradation_monitor.method import scores as conditioned


def _clusters(rng, rows):
    """Clean images of two scene types: the stage-4 key and the stage-1 level both depend on the scene."""
    half = rows // 2
    key = np.concatenate([rng.normal(0, 0.1, (half, 3)), rng.normal(10, 0.1, (half, 3))])
    early = np.concatenate([rng.normal(1, 0.05, (half, 4)), rng.normal(3, 0.05, (half, 4))])
    return {"means_s1": early, "means_s4": key}


def test_a_shift_toward_the_global_mean_hides_from_the_global_reference_but_not_from_the_neighbours():
    rng = np.random.default_rng(1)
    bank, zstats = _clusters(rng, 40), _clusters(rng, 20)
    clean = {"means_s1": np.full((1, 4), 1.0), "means_s4": np.zeros((1, 3))}  # a clean image of the first type
    shifted = {"means_s1": np.full((1, 4), 2.0), "means_s4": np.zeros((1, 3))}  # its early stage moved by +1
    test = {key: np.concatenate([clean[key], shifted[key]]) for key in clean}
    global_sum, global_layers = conditioned.global_level_scores(test, bank, zstats, scored=("s1",))
    cond_sum, cond_layers = conditioned.level_scores(test, bank, zstats, scored=("s1",), k=5)
    assert global_layers.shape == cond_layers.shape == (2, 1)
    assert global_sum[1] < global_sum[0]  # the shift lands on the global mean, so it looks cleaner
    assert cond_sum[1] > cond_sum[0] + 10  # against images of the same scene type it stands out


def test_scores_keep_the_leading_shape_of_the_test_arrays():
    rng = np.random.default_rng(2)
    layers = {"s1": 3, "s2": 4, "s3": 5, "s4": 6}
    bank = {f"means_{l}": rng.random((30, c)) for l, c in layers.items()}
    zstats = {f"means_{l}": rng.random((12, c)) for l, c in layers.items()}
    test = {f"means_{l}": rng.random((4, 7, c)) for l, c in layers.items()}
    summed, per_layer = conditioned.level_scores(test, bank, zstats, k=5)
    assert summed.shape == (4, 7) and per_layer.shape == (4, 7, 3)
    flat = {key: value.reshape(28, -1) for key, value in test.items()}
    np.testing.assert_allclose(conditioned.level_scores(flat, bank, zstats, k=5)[0], summed.reshape(28))
    summed, per_layer = conditioned.global_level_scores(test, bank, zstats)
    assert summed.shape == (4, 7) and per_layer.shape == (4, 7, 3)


def _with_peaks(rng, rows):
    """Clean images of two scene types whose stage-1 peak share also depends on the scene."""
    values = _clusters(rng, rows)
    half = rows // 2
    share = np.concatenate([np.full((half, 4), 3.0), np.full((half, 4), 5.0)]) + rng.normal(0, 0.05, (rows, 4))
    values["top_s1"] = values["means_s1"] * share
    return values


def test_peak_share_is_the_log_ratio_of_the_top_mean_to_the_mean():
    values = {"means_s1": np.array([[2.0, 0.5]]), "top_s1": np.array([[6.0, 0.5]])}
    np.testing.assert_allclose(conditioned.peak_share(values, "s1"),
                               np.log(np.array([[6.0, 0.5]]) + 1e-6) - np.log(np.array([[2.0, 0.5]]) + 1e-6))


def test_a_flattened_channel_stands_out_on_the_peak_share_but_not_on_the_level():
    rng = np.random.default_rng(4)
    bank, zstats = _with_peaks(rng, 40), _with_peaks(rng, 20)
    clean = {"means_s1": np.full((1, 4), 1.0), "top_s1": np.full((1, 4), 3.0), "means_s4": np.zeros((1, 3))}
    flat = {"means_s1": np.full((1, 4), 1.0), "top_s1": np.full((1, 4), 2.0), "means_s4": np.zeros((1, 3))}
    test = {key: np.concatenate([clean[key], flat[key]]) for key in clean}
    shape, _ = conditioned.peak_share_scores(test, bank, zstats, scored=("s1",), k=5)
    level, _ = conditioned.level_scores(test, bank, zstats, scored=("s1",), k=5)
    assert shape[1] > shape[0] + 10
    assert abs(level[1] - level[0]) < 1e-9  # the same mean, so the level cannot see it


def test_the_two_axis_score_catches_a_flattened_and_a_shifted_image():
    rng = np.random.default_rng(5)
    bank, zstats = _with_peaks(rng, 40), _with_peaks(rng, 20)
    cases = {"clean": (1.0, 3.0), "flattened": (1.0, 2.0), "shifted": (1.6, 4.8)}  # (mean, top) of stage 1
    test = {"means_s1": np.array([[m] * 4 for m, _ in cases.values()]),
            "top_s1": np.array([[t] * 4 for _, t in cases.values()]), "means_s4": np.zeros((3, 3))}
    score, arms = conditioned.two_axis_scores(test, bank, zstats, scored=("s1",), k=5)
    clean, flattened, shifted = score
    assert flattened > clean + 5 and shifted > clean + 5
    assert arms["flatter"][1] > arms["level"][1] and arms["level"][2] > arms["flatter"][2]
    np.testing.assert_allclose(score, np.maximum(arms["flatter"], arms["level"]))


def test_each_arm_of_the_two_axis_score_is_standardised_on_the_zstatistics_images():
    rng = np.random.default_rng(6)
    bank, zstats = _with_peaks(rng, 50), _with_peaks(rng, 30)
    _, arms = conditioned.two_axis_scores(zstats, bank, zstats, scored=("s1",), k=5)
    for values in arms.values():
        assert abs(values.mean()) < 1e-9 and values.std() == pytest.approx(1.0)


def test_the_zstatistics_images_score_zero_on_average():
    rng = np.random.default_rng(3)
    bank = {"means_s1": rng.random((50, 4)), "means_s4": rng.random((50, 3))}
    zstats = {"means_s1": rng.random((25, 4)), "means_s4": rng.random((25, 3))}
    summed, _ = conditioned.level_scores(zstats, bank, zstats, scored=("s1",), k=5)
    assert abs(summed.mean()) < 1e-9 and summed.std() == pytest.approx(1.0)


def test_knn_scores_are_mean_euclidean_distances_to_the_nearest_rows():
    import torch
    bank = torch.tensor([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    queries = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    assert conditioned.knn_scores(queries, bank, neighbours=2).tolist() == pytest.approx([2.5, 2.5])


def test_own_average_scores_are_mean_absolute_z():
    values = np.array([[3.0, 1.0], [1.0, 1.0]])
    got = conditioned.own_average_scores(values, np.array([1.0, 1.0]), np.array([2.0, 1.0]))
    assert got.tolist() == pytest.approx([0.5, 0.0])


def test_the_four_stage_rows_keep_the_test_shape_and_score_the_zstatistics_images_near_zero():
    rng = np.random.default_rng(4)
    widths = {"s1": 3, "s2": 4, "s3": 5, "s4": 6}
    bank = {f"means_{l}": rng.random((30, c)) for l, c in widths.items()}
    zstats = {f"means_{l}": rng.random((12, c)) for l, c in widths.items()}
    test = {f"means_{l}": rng.random((4, 7, c)) for l, c in widths.items()}
    for function in (conditioned.means_knn_scores, conditioned.means_own_scores):
        summed, per_stage = function(test, bank, zstats)
        assert summed.shape == (4, 7) and per_stage.shape == (4, 7, 4)
        clean, _ = function({k: v[None] for k, v in zstats.items()}, bank, zstats)
        assert abs(clean.mean()) < 1e-6
