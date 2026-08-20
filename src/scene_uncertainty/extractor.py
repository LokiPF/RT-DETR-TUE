from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import Tensor, nn

from src.misc.tue_utils import get_captured_persistence_diagrams, hook_decoder_layers


def stack_query_diagrams(diagrams: dict[int, Tensor], query_count: int) -> Tensor:
    missing = sorted(set(range(query_count)) - set(diagrams))
    if missing:
        raise RuntimeError(f"Missing persistence diagrams for query IDs: {missing[:10]}")
    return torch.stack([diagrams[query_id] for query_id in range(query_count)])


def build_match_metadata(logits: Tensor, targets: Sequence[dict], match_indices) -> list[dict]:
    predicted_class = logits.argmax(dim=-1).cpu()
    confidence = logits.sigmoid().amax(dim=-1).cpu()
    batch_size, query_count = predicted_class.shape
    result: list[dict] = []
    for batch_id in range(batch_size):
        matched_annotation_id = torch.full((query_count,), -1, dtype=torch.int64)
        matched_gt_class = torch.full((query_count,), -1, dtype=torch.int64)
        query_indices, target_indices = match_indices[batch_id]
        target_indices_device = target_indices.to(targets[batch_id]["labels"].device)
        matched_annotation_id[query_indices] = targets[batch_id]["annotation_ids"][target_indices_device].cpu()
        matched_gt_class[query_indices] = targets[batch_id]["labels"][target_indices_device].cpu()
        is_matched = matched_gt_class >= 0
        result.append({
            "predicted_class": predicted_class[batch_id],
            "confidence": confidence[batch_id],
            "matched_annotation_id": matched_annotation_id,
            "matched_gt_class": matched_gt_class,
            "is_matched": is_matched,
            "is_correct": is_matched & (predicted_class[batch_id] == matched_gt_class),
        })
    return result


class ClassificationPersistenceExtractor:
    def __init__(self, model: nn.Module, matcher: nn.Module, decoder_layers: Iterable[int]) -> None:
        self.model = model
        self.matcher = matcher
        self.captures, self.handles, self.layers = hook_decoder_layers(
            transformer=model.decoder,
            decoder_layers=list(decoder_layers),
            head_task="score",
        )

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @torch.inference_mode()
    def extract(self, samples: Tensor, targets: Sequence[dict]) -> list[dict]:
        self.captures.clear()
        outputs = self.model(samples)
        query_count = outputs["pred_logits"].shape[1]
        query_indices = [
            torch.arange(query_count, device=samples.device)
            for _ in range(samples.shape[0])
        ]
        diagrams = get_captured_persistence_diagrams(
            captures=self.captures,
            query_indices=query_indices,
            decoder_layer_indices=self.layers,
        )
        matches = self.matcher(outputs, targets)["indices"]
        metadata = build_match_metadata(outputs["pred_logits"], targets, matches)
        records: list[dict] = []
        for batch_id, target in enumerate(targets):
            records.append({
                "image_id": int(target["image_id"].item()),
                "orig_size": target["orig_size"].cpu(),
                "layers": {
                    int(layer_id): stack_query_diagrams(diagrams[layer_id][batch_id], query_count).to(torch.float16)
                    for layer_id in self.layers
                },
                "logits": outputs["pred_logits"][batch_id].detach().cpu().to(torch.float16),
                "boxes": outputs["pred_boxes"][batch_id].detach().cpu().to(torch.float32),
                **metadata[batch_id],
            })
        return records
