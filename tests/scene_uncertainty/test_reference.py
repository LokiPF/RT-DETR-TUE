import pytest

from src.scene_uncertainty.reference import select_evaluation_ids, select_reference_ids


class FakeCoco:
    def __init__(self):
        self.imgs = {image_id: {} for image_id in range(1, 13)}
        self.cats = {1: {"name": "common"}, 2: {"name": "rare"}, 3: {"name": "joint"}}
        self.imgToAnns = {
            1: [{"category_id": 1}],
            2: [{"category_id": 1}],
            3: [{"category_id": 1}],
            4: [{"category_id": 1}],
            5: [{"category_id": 2}, {"category_id": 3}],
            6: [{"category_id": 2}, {"category_id": 3}],
            7: [{"category_id": 2}, {"category_id": 3}],
            8: [{"category_id": 1}],
            9: [{"category_id": 1}],
            10: [{"category_id": 1}],
            11: [{"category_id": 1}],
            12: [{"category_id": 1}],
        }


class CrowdCoco:
    """Every image carries a non-crowd `common` box; image 1 also carries a crowd `crowd_only` box."""

    def __init__(self):
        self.imgs = {image_id: {} for image_id in range(1, 5)}
        self.cats = {1: {"name": "common"}, 2: {"name": "crowd_only"}}
        self.imgToAnns = {
            1: [{"category_id": 1, "iscrowd": 0}, {"category_id": 2, "iscrowd": 1}],
            2: [{"category_id": 1, "iscrowd": 0}],
            3: [{"category_id": 1, "iscrowd": 0}],
            4: [{"category_id": 1, "iscrowd": 0}],
        }


class TieCoco:
    """Images 1-3 carry only `common` and images 4-6 only `rare`, so every greedy pick is a tie."""

    def __init__(self):
        self.imgs = {image_id: {} for image_id in range(1, 7)}
        self.cats = {1: {"name": "common"}, 2: {"name": "rare"}}
        self.imgToAnns = {
            1: [{"category_id": 1}],
            2: [{"category_id": 1}],
            3: [{"category_id": 1}],
            4: [{"category_id": 2}],
            5: [{"category_id": 2}],
            6: [{"category_id": 2}],
        }


def test_reference_selection_is_deterministic_and_fills_rare_deficits():
    first = select_reference_ids(FakeCoco(), natural_count=4, augmentation_budget=2, quota=2, seed=7)
    second = select_reference_ids(FakeCoco(), natural_count=4, augmentation_budget=2, quota=2, seed=7)
    assert first == second
    assert len(first["natural_ids"]) == 4
    assert len(first["augmentation_ids"]) == 2
    assert set(first["natural_ids"]).isdisjoint(first["augmentation_ids"])
    assert first["category_image_counts"][2] >= 2
    assert first["category_image_counts"][3] >= 2


def test_reference_selection_reproduces_at_every_greedy_pick_count():
    # Quotas 1, 2 and 3 drive 0, 1 and 2 greedy picks before the random fill, so this covers
    # determinism across the whole range of pick counts rather than one fixed count. The final
    # assertion is what proves the three quotas really do take different paths.
    augmentations = []
    for quota in (1, 2, 3):
        first = select_reference_ids(FakeCoco(), natural_count=4, augmentation_budget=2, quota=quota, seed=7)
        second = select_reference_ids(FakeCoco(), natural_count=4, augmentation_budget=2, quota=quota, seed=7)
        assert first == second
        augmentations.append(tuple(first["augmentation_ids"]))
    assert len(set(augmentations)) == 3


def test_greedy_augmentation_breaks_coverage_ties_by_lowest_image_id():
    # natural_count=0 removes the random draw, so both picks below come from the greedy pass
    # alone: image 1 over the equally covering 2 and 3, then image 4 over 5 and 6.
    result = select_reference_ids(TieCoco(), natural_count=0, augmentation_budget=2, quota=1, seed=0)
    assert result["natural_ids"] == []
    assert result["augmentation_ids"] == [1, 4]
    assert result["category_image_counts"] == {1: 1, 2: 1}


def test_crowd_annotations_never_count_toward_a_category_quota():
    # natural_count + augmentation_budget covers every image, so these counts are whole-fixture
    # totals and cannot depend on which images the seed happened to draw.
    result = select_reference_ids(CrowdCoco(), natural_count=3, augmentation_budget=1, quota=1, seed=0)
    assert result["category_image_counts"] == {1: 4, 2: 0}
    assert result["unmet_category_quotas"] == {2: 1}


def test_reference_request_larger_than_the_index_is_rejected():
    with pytest.raises(ValueError, match="exceeds available"):
        select_reference_ids(FakeCoco(), natural_count=12, augmentation_budget=1)


def test_evaluation_split_is_disjoint_and_keeps_requested_pilot_sizes():
    result = select_evaluation_ids(list(range(20)), seed=42, tuning_fraction=0.5, pilot_per_partition=4)
    assert set(result["tuning_ids"]).isdisjoint(result["test_ids"])
    assert len(result["tuning_pilot_ids"]) == 4
    assert len(result["test_pilot_ids"]) == 4
    assert set(result["tuning_pilot_ids"]) <= set(result["tuning_ids"])
    assert set(result["test_pilot_ids"]) <= set(result["test_ids"])
