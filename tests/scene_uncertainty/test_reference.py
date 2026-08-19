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


def test_reference_selection_is_deterministic_and_fills_rare_deficits():
    first = select_reference_ids(FakeCoco(), natural_count=4, augmentation_budget=2, quota=2, seed=7)
    second = select_reference_ids(FakeCoco(), natural_count=4, augmentation_budget=2, quota=2, seed=7)
    assert first == second
    assert len(first["natural_ids"]) == 4
    assert len(first["augmentation_ids"]) == 2
    assert set(first["natural_ids"]).isdisjoint(first["augmentation_ids"])
    assert first["category_image_counts"][2] >= 2
    assert first["category_image_counts"][3] >= 2


def test_evaluation_split_is_disjoint_and_keeps_requested_pilot_sizes():
    result = select_evaluation_ids(list(range(20)), seed=42, tuning_fraction=0.5, pilot_per_partition=4)
    assert set(result["tuning_ids"]).isdisjoint(result["test_ids"])
    assert len(result["tuning_pilot_ids"]) == 4
    assert len(result["test_pilot_ids"]) == 4
    assert set(result["tuning_pilot_ids"]) <= set(result["tuning_ids"])
    assert set(result["test_pilot_ids"]) <= set(result["test_ids"])
