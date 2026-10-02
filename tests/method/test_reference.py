import numpy as np
import pytest

from degradation_monitor.method import reference


def test_fit_own_average_floors_dead_dimensions():
    clean = np.array([[1.0, 0.0, 5.0], [3.0, 0.0, 7.0]])
    mean, std = reference.fit_own_average(clean)
    assert mean.tolist() == [2.0, 0.0, 6.0]
    assert std.tolist() == pytest.approx([1.0, reference.STD_FLOOR * 1.0, 1.0])
    with pytest.raises(ValueError):
        reference.fit_own_average(np.zeros((1, 3)))


def test_nearest_rows_match_a_brute_force_search():
    rng = np.random.default_rng(0)
    bank, queries = rng.normal(size=(40, 6)), rng.normal(size=(9, 6))
    got = reference.nearest_rows(queries, bank, 5, chunk=4)
    distances = ((queries[:, None] - bank[None]) ** 2).sum(-1)
    assert [set(row) for row in got] == [set(row) for row in np.argsort(distances, axis=1)[:, :5]]
    with pytest.raises(ValueError, match="between 1 and"):
        reference.nearest_rows(queries, bank, 41)


def test_the_fixed_choices_are_the_preregistered_ones():
    assert (reference.KEY_LAYER, reference.SCORED_LAYERS, reference.NEIGHBOURS) == ("s4", ("s1", "s2", "s3"), 50)
    assert (reference.BANK_IMAGES, reference.ZSTAT_IMAGES) == (2000, 500)
