import math
import weakref

import numpy as np
import pytest
import torch

from src.scene_uncertainty.bank import deterministic_reservoir, make_coverage_bank, streaming_coverage_bank
from src.scene_uncertainty.normalization import fit_normalizer, transform_vectors


def test_reservoir_is_deterministic_and_capped():
    vectors = [torch.tensor([float(index), 0.0]) for index in range(100)]
    first = deterministic_reservoir(vectors, capacity=10, seed=3)
    second = deterministic_reservoir(vectors, capacity=10, seed=3)
    assert torch.equal(first, second)
    assert first.shape == (10, 2)


def test_coverage_bank_reserves_object_and_background_capacity():
    vectors = torch.arange(80, dtype=torch.float32).reshape(20, 4)
    classes = torch.tensor([0] * 4 + [1] * 4 + [-1] * 12)
    bank, selected = make_coverage_bank(vectors, classes, capacity=12, object_fraction=0.5, seed=5)
    assert bank.shape == (12, 4)
    assert (classes[selected] >= 0).sum().item() >= 6
    assert (classes[selected] < 0).sum().item() >= 6


def test_streaming_coverage_bank_is_capped_without_concatenating_the_cache():
    records = [
        {"layers": {2: torch.full((4, 2), float(index))},
         "matched_gt_class": torch.tensor([0, 1, -1, -1])}
        for index in range(20)
    ]
    bank, metadata = streaming_coverage_bank(
        iter(records), layer_id=2, capacity=20, object_fraction=0.5, class_count=2, seed=7
    )
    assert bank.shape == (20, 2)
    assert metadata["object_vectors"] == 10
    assert metadata["background_vectors"] == 10


def test_all_normalizers_are_finite_and_shape_scale_adds_one_coordinate():
    bank = torch.tensor([[1.0, 0.0], [2.0, 2.0], [4.0, 1.0]])
    for mode in ("raw", "robust_z", "unit"):
        state = fit_normalizer(bank, mode)
        output = transform_vectors(bank, state)
        assert output.shape == bank.shape
        assert torch.isfinite(output).all()
    state = fit_normalizer(bank, "shape_scale")
    output = transform_vectors(bank, state)
    assert output.shape == (3, 3)
    assert torch.isfinite(output).all()


def _chunked(vectors, sizes):
    """Yield the same vectors a caller would stream, but in uneven batches."""
    position = 0
    for size in sizes:
        for vector in vectors[position:position + size]:
            yield vector
        position += size
    for vector in vectors[position:]:
        yield vector


def _tracked_layer(index, references):
    """A record tensor whose *storage* liveness is observable.

    A weakref to the tensor itself only tracks the Python wrapper, which `detach()`
    drops while keeping the memory. The storage of a `from_numpy` tensor holds the
    numpy array alive instead, so the weakref dies exactly when the memory is freed.
    """
    array = np.full((4, 2), float(index), dtype=np.float32)
    references.append(weakref.ref(array))
    return torch.from_numpy(array)


def _count_live(references, live_counts):
    live_counts.append(sum(1 for reference in references if reference() is not None))


def _tracked_vector_stream(record_count, references, live_counts):
    """Yield row views of one record tensor at a time, as the bank builder is fed."""
    for index in range(record_count):
        record = _tracked_layer(index, references)
        yield from record
        del record
        _count_live(references, live_counts)


def _tracked_record_stream(record_count, references, live_counts):
    for index in range(record_count):
        layer = _tracked_layer(index, references)
        yield {"layers": {2: layer}, "matched_gt_class": torch.tensor([0, 1, -1, -1])}
        del layer
        _count_live(references, live_counts)


def test_reservoir_ignores_how_the_stream_is_chunked():
    vectors = [torch.tensor([float(index), 0.0]) for index in range(100)]
    flat = deterministic_reservoir(iter(vectors), capacity=10, seed=3)
    chunked = deterministic_reservoir(_chunked(vectors, [1, 7, 32, 11, 49]), capacity=10, seed=3)
    assert torch.equal(flat, chunked)


def test_reservoir_keeps_every_vector_when_capacity_matches_the_stream():
    vectors = [torch.tensor([float(index), 0.0]) for index in range(10)]
    assert torch.equal(deterministic_reservoir(vectors, capacity=10, seed=3), torch.stack(vectors))
    assert torch.equal(deterministic_reservoir(vectors, capacity=25, seed=3), torch.stack(vectors))


def test_reservoir_rejects_an_empty_stream():
    with pytest.raises(ValueError, match="zero vectors"):
        deterministic_reservoir([], capacity=10, seed=3)


def test_reservoir_does_not_retain_the_records_its_vectors_came_from():
    references: list = []
    live_counts: list[int] = []
    bank = deterministic_reservoir(
        _tracked_vector_stream(40, references, live_counts), capacity=20, seed=3
    )
    assert bank.shape == (20, 2)
    assert max(live_counts) <= 2


def test_streaming_bank_does_not_retain_the_records_it_samples_from():
    references: list = []
    live_counts: list[int] = []
    bank, _ = streaming_coverage_bank(
        _tracked_record_stream(40, references, live_counts),
        layer_id=2, capacity=20, object_fraction=0.5, class_count=2, seed=7,
    )
    assert bank.shape == (20, 2)
    assert max(live_counts) <= 2


def test_streaming_bank_gives_a_rare_class_the_same_budget_as_a_dominant_class():
    records = [
        {"layers": {2: torch.tensor([[10.0, 10.0]] * 5 + [[20.0, 20.0]] + [[30.0, 30.0]] * 2)},
         "matched_gt_class": torch.tensor([0, 0, 0, 0, 0, 1, -1, -1])}
        for _ in range(20)
    ]
    bank, metadata = streaming_coverage_bank(
        iter(records), layer_id=2, capacity=20, object_fraction=0.5, class_count=2, seed=7
    )
    assert (bank == 10.0).all(dim=1).sum().item() == 5
    assert (bank == 20.0).all(dim=1).sum().item() == 5
    assert (bank == 30.0).all(dim=1).sum().item() == 10
    assert metadata["per_class_available"] == {0: 100, 1: 20}


def test_streaming_bank_requires_both_objects_and_background():
    def records(classes):
        return [
            {"layers": {2: torch.full((2, 2), float(index))}, "matched_gt_class": torch.tensor(classes)}
            for index in range(10)
        ]

    with pytest.raises(ValueError, match="matched-object and background"):
        streaming_coverage_bank(
            iter(records([0, 1])), layer_id=2, capacity=20,
            object_fraction=0.5, class_count=2, seed=7,
        )
    with pytest.raises(ValueError, match="matched-object and background"):
        streaming_coverage_bank(
            iter(records([-1, -1])), layer_id=2, capacity=20,
            object_fraction=0.5, class_count=2, seed=7,
        )


@pytest.mark.parametrize(
    ("capacity", "object_fraction"), [(1, 0.5), (20, 0.0), (20, 1.0), (20, 1.5)]
)
def test_streaming_bank_rejects_degenerate_capacity_or_fraction(capacity, object_fraction):
    with pytest.raises(ValueError, match="capacity >= 2"):
        streaming_coverage_bank(
            iter([]), layer_id=2, capacity=capacity,
            object_fraction=object_fraction, class_count=2, seed=7,
        )


@pytest.mark.parametrize("object_fraction", [0.0, 1.0, -0.5, 2.0])
def test_coverage_bank_rejects_an_out_of_range_object_fraction(object_fraction):
    with pytest.raises(ValueError, match="object_fraction"):
        make_coverage_bank(
            torch.zeros(4, 2), torch.tensor([0, 1, -1, -1]),
            capacity=2, object_fraction=object_fraction, seed=5,
        )


def test_coverage_bank_returns_the_rows_its_indices_name_exactly_once():
    vectors = torch.arange(80, dtype=torch.float32).reshape(20, 4)
    classes = torch.tensor([0] * 4 + [1] * 4 + [-1] * 12)
    bank, selected = make_coverage_bank(vectors, classes, capacity=12, object_fraction=0.5, seed=5)
    assert torch.equal(bank, vectors[selected])
    assert len(set(selected.tolist())) == selected.numel()


def test_normalizer_state_is_fitted_on_the_bank_and_never_refitted():
    bank = torch.tensor([[1.0, 0.0], [2.0, 2.0], [4.0, 1.0], [8.0, 3.0]])
    evaluation = torch.tensor([[100.0, 50.0], [-100.0, -50.0]])
    # every feature here spreads, so no floor applies and the raw spread is the scale
    center = bank.median(dim=0).values
    scale = torch.quantile(bank, 0.75, dim=0) - torch.quantile(bank, 0.25, dim=0)
    state = fit_normalizer(bank, "robust_z")
    assert torch.allclose(transform_vectors(evaluation, state), (evaluation - center) / scale)
    assert not torch.allclose(
        transform_vectors(evaluation, state),
        transform_vectors(evaluation, fit_normalizer(evaluation, "robust_z")),
    )
    magnitudes = bank.norm(dim=1, keepdim=True).log1p()
    magnitude_center = magnitudes.median(dim=0).values
    magnitude_scale = (
        torch.quantile(magnitudes, 0.75, dim=0) - torch.quantile(magnitudes, 0.25, dim=0)
    )
    shape_state = fit_normalizer(bank, "shape_scale")
    expected = (evaluation.norm(dim=1, keepdim=True).log1p() - magnitude_center) / magnitude_scale
    assert torch.allclose(transform_vectors(evaluation, shape_state)[:, 2:], expected)


def test_robust_z_and_shape_scale_survive_a_zero_spread_bank():
    bank = torch.tensor([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    output = transform_vectors(bank, fit_normalizer(bank, "robust_z"))
    assert torch.isfinite(output).all()
    assert torch.equal(output[:, 1], torch.zeros(3))
    equal_norms = torch.tensor([[3.0, 4.0], [4.0, 3.0], [0.0, 5.0]])
    magnitudes = transform_vectors(equal_norms, fit_normalizer(equal_norms, "shape_scale"))[:, 2]
    assert torch.isfinite(magnitudes).all()
    assert torch.equal(magnitudes, torch.zeros(3))


def test_shape_scale_splits_direction_from_magnitude():
    bank = torch.tensor([[1.0, 0.0], [2.0, 2.0], [4.0, 1.0]])
    collinear = torch.tensor([[3.0, 4.0], [30.0, 40.0]])
    output = transform_vectors(collinear, fit_normalizer(bank, "shape_scale"))
    assert torch.allclose(output[0, :2], output[1, :2])
    assert output[1, 2] > output[0, 2]
    unit = transform_vectors(collinear, fit_normalizer(bank, "unit"))
    assert torch.allclose(unit[0], unit[1])
    assert torch.allclose(unit, output[:, :2])


@pytest.mark.parametrize("mode", ["zscore", "", "Raw"])
def test_unknown_normalization_mode_raises(mode):
    bank = torch.tensor([[1.0, 0.0], [2.0, 2.0]])
    with pytest.raises(ValueError, match="Unknown normalization mode"):
        fit_normalizer(bank, mode)
    with pytest.raises(ValueError, match="Unknown normalization mode"):
        transform_vectors(bank, {"mode": mode})


def test_robust_z_bounds_a_bank_constant_feature_instead_of_exploding_it():
    """Feature 1 never moves in the bank, so its spread carries no scale of its own.

    The floor it falls back to must be relative to how much the other features move,
    or an evaluation-time movement in it swamps every informative coordinate.
    """
    bank = torch.stack([torch.tensor([float(index), 0.0]) for index in range(1000)])
    state = fit_normalizer(bank, "robust_z")
    assert state["scale"][1].item() == pytest.approx(1e-3 * state["scale"][0].item(), rel=1e-5)
    assert state["zero_spread"].tolist() == [False, True]
    assert state["zero_spread_count"] == 1
    output = transform_vectors(torch.tensor([[500.0, 0.3]]), state)
    assert torch.isfinite(output).all()
    assert abs(output[0, 1].item()) < 1.0


def test_shape_scale_bounds_the_magnitude_coordinate_on_an_equal_norm_bank():
    bank = torch.tensor([[3.0, 4.0], [4.0, 3.0], [0.0, 5.0], [5.0, 0.0]])
    state = fit_normalizer(bank, "shape_scale")
    assert state["magnitude_zero_spread"].tolist() == [True]
    assert state["magnitude_zero_spread_count"] == 1
    assert state["magnitude_scale"].item() == pytest.approx(1e-3 * math.log1p(5.0), rel=1e-5)
    output = transform_vectors(torch.tensor([[6.0, 8.0]]), state)
    assert torch.isfinite(output).all()
    assert abs(output[0, 2].item()) < 1000.0


def test_robust_z_stays_finite_when_the_whole_bank_is_one_point():
    bank = torch.zeros(4, 2)
    state = fit_normalizer(bank, "robust_z")
    assert state["zero_spread"].tolist() == [True, True]
    assert state["zero_spread_count"] == 2
    assert torch.isfinite(transform_vectors(torch.tensor([[1.0, -2.0]]), state)).all()
