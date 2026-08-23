from __future__ import annotations

import torch
from torch import Tensor, nn


@torch.no_grad()
def _batched_prim(weight_matrix: Tensor, layer_inputs: Tensor) -> Tensor:
    if weight_matrix.ndim != 2 or layer_inputs.ndim != 2:
        raise ValueError(
            "weight_matrix and layer_inputs must both be two-dimensional"
        )

    device = layer_inputs.device
    weight = weight_matrix.detach().to(device=device, dtype=torch.float32)
    inputs = layer_inputs.detach().to(device=device, dtype=torch.float32)

    query_count, input_count = inputs.shape
    output_count, expected_inputs = weight.shape
    if input_count != expected_inputs:
        raise ValueError("persistence input and score-head dimensions do not match")

    if query_count == 0:
        return torch.empty(
            (0, input_count + output_count - 1),
            dtype=torch.float32,
            device=device,
        )

    edge_weights = torch.abs(inputs[:, None, :] * weight[None, :, :])

    selected_inputs = torch.zeros(
        (query_count, input_count),
        dtype=torch.bool,
        device=device,
    )
    selected_outputs = torch.zeros(
        (query_count, output_count),
        dtype=torch.bool,
        device=device,
    )

    selected_inputs[:, 0] = True
    best_input_weights = torch.full(
        (query_count, input_count),
        -torch.inf,
        dtype=torch.float32,
        device=device,
    )
    best_output_weights = edge_weights[:, :, 0].clone()
    mst_weights = torch.empty(
        (query_count, input_count + output_count - 1),
        dtype=torch.float32,
        device=device,
    )

    for step in range(mst_weights.shape[1]):
        input_candidates = best_input_weights.masked_fill(
            selected_inputs,
            -torch.inf,
        )
        output_candidates = best_output_weights.masked_fill(
            selected_outputs,
            -torch.inf,
        )
        candidates = torch.cat(
            [input_candidates, output_candidates],
            dim=1,
        )

        selected_weight, selected_vertex = candidates.max(dim=1)
        mst_weights[:, step] = selected_weight

        is_input = selected_vertex < input_count
        input_index = selected_vertex.clamp(max=input_count - 1)
        output_index = (selected_vertex - input_count).clamp(
            min=0,
            max=output_count - 1,
        )

        previous_input_state = selected_inputs.gather(
            1,
            input_index[:, None],
        )
        selected_inputs.scatter_(
            1,
            input_index[:, None],
            previous_input_state | is_input[:, None],
        )

        previous_output_state = selected_outputs.gather(
            1,
            output_index[:, None],
        )
        selected_outputs.scatter_(
            1,
            output_index[:, None],
            previous_output_state | (~is_input)[:, None],
        )

        new_output_weights = edge_weights.gather(
            2,
            input_index[:, None, None].expand(-1, output_count, 1),
        ).squeeze(2)
        best_output_weights = torch.where(
            is_input[:, None],
            torch.maximum(best_output_weights, new_output_weights),
            best_output_weights,
        )

        new_input_weights = edge_weights.gather(
            1,
            output_index[:, None, None].expand(-1, 1, input_count),
        ).squeeze(1)
        best_input_weights = torch.where(
            (~is_input)[:, None],
            torch.maximum(best_input_weights, new_input_weights),
            best_input_weights,
        )

    return torch.sort(mst_weights, dim=1, descending=True).values


@torch.no_grad()
def batched_persistence(
    weight_matrix: Tensor,
    layer_inputs: Tensor,
    chunk_size: int = 64,
) -> Tensor:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if weight_matrix.ndim != 2 or layer_inputs.ndim != 2:
        raise ValueError(
            "weight_matrix and layer_inputs must both be two-dimensional"
        )
    if layer_inputs.shape[1] != weight_matrix.shape[1]:
        raise ValueError("persistence input and score-head dimensions do not match")

    if layer_inputs.shape[0] == 0:
        diagram_width = weight_matrix.shape[0] + weight_matrix.shape[1] - 1
        return torch.empty(
            (0, diagram_width),
            dtype=torch.float32,
            device=layer_inputs.device,
        )

    chunks = [
        _batched_prim(weight_matrix, input_chunk)
        for input_chunk in layer_inputs.split(chunk_size, dim=0)
    ]
    return torch.cat(chunks, dim=0)


class Layer2Capture:
    def __init__(self, transformer: nn.Module, layer: int = 2) -> None:
        if layer < 0 or layer >= len(transformer.decoder.layers):
            raise ValueError(f"decoder layer {layer} does not exist")

        head = transformer.dec_score_head[layer]
        if not isinstance(head, nn.Linear):
            raise TypeError("the selected decoder score head must be nn.Linear")

        self.layer = layer
        self.head = head
        self.captured: Tensor | None = None
        self.handles = [
            transformer.decoder.layers[layer].register_forward_hook(self._hook)
        ]

    def _hook(
        self,
        _module: nn.Module,
        _inputs: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        self.captured = output.detach()

    def take(self) -> tuple[Tensor, Tensor]:
        if self.captured is None:
            raise RuntimeError(
                f"the detector did not execute decoder layer {self.layer}"
            )

        features = self.captured
        self.captured = None
        return features, self.head.weight.detach()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __enter__(self) -> Layer2Capture:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
