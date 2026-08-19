import pytest
import torch
from torch import Tensor, nn

from src.scene_uncertainty.extractor import (
    ClassificationPersistenceExtractor,
    build_match_metadata,
    stack_query_diagrams,
)


_HIDDEN_DIM = 4
_NUM_CLASSES = 2
_QUERY_COUNT = 3
_LAYER_SCALES = (1.0, 3.0)
_BATCH_SCALES = (1.0, 5.0)
_SCORE_WEIGHT = torch.tensor([[1.0, 0.75, -3.0, 0.125], [0.5, -2.5, 8.0, 1.5]])
_BASE_QUERY = torch.tensor([1.0, -0.5, 0.25, 2.0])
_MATCH_INDICES = [
    (torch.tensor([2]), torch.tensor([0])),
    (torch.tensor([1]), torch.tensor([0])),
]


def _hidden_states() -> Tensor:
    """Query `q` of batch item `b` is `(q + 1) * batch_scale[b]` times a shared vector.

    Persistence diagrams are maximum-spanning-tree weights over |weight * activation|,
    so scaling one query's activations by a positive factor scales its whole diagram by
    the same factor. That makes every query, decoder layer and batch item identifiable
    in the extracted output.
    """
    query_scale = torch.arange(1, _QUERY_COUNT + 1, dtype=torch.float32)
    batch_scale = torch.tensor(_BATCH_SCALES)
    return _BASE_QUERY * query_scale[None, :, None] * batch_scale[:, None, None]


class _ScaledDecoderLayer(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, hidden: Tensor) -> Tensor:
        return hidden * self.scale


class _FakeTransformer(nn.Module):
    """The two attributes `hook_decoder_layers` needs: `decoder.layers` and `dec_score_head`."""

    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList(
            [_ScaledDecoderLayer(scale) for scale in _LAYER_SCALES]
        )
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(_HIDDEN_DIM, _NUM_CLASSES) for _ in _LAYER_SCALES]
        )
        for head in self.dec_score_head:
            head.weight.data.copy_(_SCORE_WEIGHT)
            head.bias.data.zero_()


class _FakeDetector(nn.Module):
    def __init__(self, fail: bool = False) -> None:
        super().__init__()
        self.decoder = _FakeTransformer()
        self.register_buffer("hidden_states", _hidden_states())
        self.fail = fail
        self.inference_mode_during_forward: bool | None = None

    def forward(self, samples: Tensor) -> dict[str, Tensor]:
        if self.fail:
            raise RuntimeError("forward failed")
        self.inference_mode_during_forward = torch.is_inference_mode_enabled()
        hidden = self.hidden_states.to(samples.device)
        for layer in self.decoder.decoder.layers:
            hidden = layer(hidden)
        return {
            "pred_logits": self.decoder.dec_score_head[-1](hidden),
            "pred_boxes": hidden.sigmoid(),
        }


class _FakeMatcher(nn.Module):
    def __init__(self, indices) -> None:
        super().__init__()
        self.indices = indices

    def forward(self, outputs, targets) -> dict:
        return {"indices": self.indices}


def _targets() -> list[dict]:
    return [
        {
            "image_id": torch.tensor([11]),
            "orig_size": torch.tensor([640, 480]),
            "labels": torch.tensor([1]),
            "annotation_ids": torch.tensor([501]),
            "boxes": torch.zeros(1, 4),
        },
        {
            "image_id": torch.tensor([12]),
            "orig_size": torch.tensor([320, 240]),
            "labels": torch.tensor([0]),
            "annotation_ids": torch.tensor([502]),
            "boxes": torch.zeros(1, 4),
        },
    ]


def test_match_metadata_keeps_unmatched_and_incorrect_queries():
    logits = torch.tensor([[[5.0, -1.0], [-2.0, 4.0], [1.0, 0.0]]])
    targets = [{
        "labels": torch.tensor([0, 0]),
        "annotation_ids": torch.tensor([101, 102]),
    }]
    match_indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
    metadata = build_match_metadata(logits, targets, match_indices)[0]
    assert metadata["matched_annotation_id"].tolist() == [101, 102, -1]
    assert metadata["matched_gt_class"].tolist() == [0, 0, -1]
    assert metadata["is_matched"].tolist() == [True, True, False]
    assert metadata["is_correct"].tolist() == [True, False, False]


def test_stack_query_diagrams_preserves_query_order():
    diagrams = {2: torch.tensor([2.0]), 0: torch.tensor([0.0]), 1: torch.tensor([1.0])}
    stacked = stack_query_diagrams(diagrams, query_count=3)
    assert stacked[:, 0].tolist() == [0.0, 1.0, 2.0]


def test_close_removes_every_hook():
    class Handle:
        def __init__(self):
            self.removed = False

        def remove(self):
            self.removed = True

    extractor = object.__new__(__import__(
        "src.scene_uncertainty.extractor", fromlist=["ClassificationPersistenceExtractor"]
    ).ClassificationPersistenceExtractor)
    extractor.handles = [Handle(), Handle()]
    handles = list(extractor.handles)
    extractor.close()
    assert all(handle.removed for handle in handles)
    assert extractor.handles == []


def test_stack_query_diagrams_rejects_missing_queries():
    diagrams = {0: torch.tensor([0.0]), 2: torch.tensor([2.0])}
    with pytest.raises(RuntimeError, match=r"\[1\]"):
        stack_query_diagrams(diagrams, query_count=3)


def test_match_metadata_follows_matcher_permutation():
    logits = torch.tensor([[[5.0, -1.0], [1.0, 0.0], [-2.0, 4.0]]])
    targets = [{
        "labels": torch.tensor([0, 1]),
        "annotation_ids": torch.tensor([101, 102]),
    }]
    # Query 2 is matched to target 1 and query 0 to target 0; query 1 stays unmatched.
    match_indices = [(torch.tensor([2, 0]), torch.tensor([1, 0]))]
    metadata = build_match_metadata(logits, targets, match_indices)[0]
    assert metadata["matched_annotation_id"].tolist() == [101, -1, 102]
    assert metadata["matched_gt_class"].tolist() == [0, -1, 1]
    assert metadata["is_matched"].tolist() == [True, False, True]
    assert metadata["is_correct"].tolist() == [True, False, True]


def test_match_metadata_scores_every_query():
    logits = torch.tensor([[[5.0, -1.0], [-2.0, 4.0]]])
    targets = [{"labels": torch.tensor([0]), "annotation_ids": torch.tensor([101])}]
    match_indices = [(torch.tensor([0]), torch.tensor([0]))]
    metadata = build_match_metadata(logits, targets, match_indices)[0]
    assert metadata["predicted_class"].tolist() == [0, 1]
    assert torch.allclose(metadata["confidence"], torch.tensor([5.0, 4.0]).sigmoid())


def test_extract_returns_every_query_of_every_layer_in_index_order():
    model = _FakeDetector()
    with ClassificationPersistenceExtractor(
        model, _FakeMatcher(_MATCH_INDICES), decoder_layers=[0, 1]
    ) as extractor:
        records = extractor.extract(torch.zeros(2, 3, 8, 8), _targets())

    diagram_size = _HIDDEN_DIM + _NUM_CLASSES - 1
    assert len(records) == 2
    for record in records:
        assert sorted(record["layers"]) == [0, 1]
        for diagram in record["layers"].values():
            assert diagram.shape == (_QUERY_COUNT, diagram_size)
            assert diagram.dtype == torch.float16

    first_layer = records[0]["layers"][0].float()
    for query_id in range(_QUERY_COUNT):
        assert torch.allclose(
            first_layer[query_id], (query_id + 1) * first_layer[0], rtol=1e-2, atol=1e-2
        )
    assert torch.allclose(
        records[0]["layers"][1].float(), _LAYER_SCALES[1] * first_layer, rtol=1e-2, atol=1e-2
    )
    assert torch.allclose(
        records[1]["layers"][0].float(), _BATCH_SCALES[1] * first_layer, rtol=1e-2, atol=1e-2
    )


def test_extract_pairs_every_record_with_its_own_image_metadata():
    model = _FakeDetector()
    with ClassificationPersistenceExtractor(
        model, _FakeMatcher(_MATCH_INDICES), decoder_layers=[0, 1]
    ) as extractor:
        records = extractor.extract(torch.zeros(2, 3, 8, 8), _targets())

    assert [record["image_id"] for record in records] == [11, 12]
    assert records[1]["orig_size"].tolist() == [320, 240]
    assert records[0]["logits"].shape == (_QUERY_COUNT, _NUM_CLASSES)
    assert records[0]["logits"].dtype == torch.float16
    assert records[0]["boxes"].shape == (_QUERY_COUNT, 4)
    assert records[0]["boxes"].dtype == torch.float32
    assert records[0]["matched_annotation_id"].tolist() == [-1, -1, 501]
    assert records[1]["matched_annotation_id"].tolist() == [-1, 502, -1]


def test_extract_runs_in_inference_mode():
    model = _FakeDetector()
    with ClassificationPersistenceExtractor(
        model, _FakeMatcher(_MATCH_INDICES), decoder_layers=[0, 1]
    ) as extractor:
        extractor.extract(torch.zeros(2, 3, 8, 8), _targets())
    assert model.inference_mode_during_forward is True


def test_context_manager_removes_hooks_when_extraction_fails():
    model = _FakeDetector(fail=True)
    with pytest.raises(RuntimeError, match="forward failed"):
        with ClassificationPersistenceExtractor(
            model, _FakeMatcher(_MATCH_INDICES), decoder_layers=[0, 1]
        ) as extractor:
            extractor.extract(torch.zeros(2, 3, 8, 8), _targets())
    assert extractor.handles == []
    assert all(not layer._forward_hooks for layer in model.decoder.decoder.layers)
