from __future__ import annotations

import torch
from torch import Tensor


# The graph structure is identical for every query with the same dimensions.
_EDGE_INDEX_CACHE: dict[tuple[int, int], Tensor] = {}


def _get_bipartite_edge_index(
    num_inputs: int,
    num_outputs: int,
) -> Tensor:
    cache_key = (num_inputs, num_outputs)

    if cache_key not in _EDGE_INDEX_CACHE:
        input_vertices = torch.arange(num_inputs).repeat(num_outputs)

        output_vertices = (
            num_inputs + torch.arange(num_outputs)
        ).repeat_interleave(num_inputs)

        _EDGE_INDEX_CACHE[cache_key] = torch.stack(
            [input_vertices, output_vertices],
            dim=0,
        )

    return _EDGE_INDEX_CACHE[cache_key]


@torch.no_grad()
def get_activation_weights(
    weight_matrix: Tensor,
    layer_input: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Construct the weighted bipartite graph for one linear layer input.

    The edge between input j and output i has weight

        a_ij = abs(W_ij * x_j)

    Args:
        weight_matrix:
            Linear-layer weight matrix with shape
            [num_outputs, num_inputs].

        layer_input:
            Input activation vector with shape [num_inputs].

    Returns:
        edge_index:
            Graph edges with shape [2, num_outputs * num_inputs].

        edge_weight:
            Flattened activation weights with shape
            [num_outputs * num_inputs].
    """
    weight_matrix = weight_matrix.detach().cpu()
    layer_input = layer_input.detach().cpu()

    if weight_matrix.ndim != 2:
        raise ValueError("weight_matrix must be two-dimensional")

    if layer_input.ndim != 1:
        raise ValueError("layer_input must be one-dimensional")

    num_outputs, num_inputs = weight_matrix.shape

    if layer_input.shape[0] != num_inputs:
        raise ValueError(
            f"Expected input dimension {num_inputs}, "
            f"got {layer_input.shape[0]}"
        )

    activation_matrix = torch.abs(
        weight_matrix * layer_input.unsqueeze(0)
    )

    edge_index = _get_bipartite_edge_index(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
    )

    return edge_index, activation_matrix.flatten()


@torch.no_grad()
def get_maximum_spanning_tree(
    edge_index: Tensor,
    edge_weight: Tensor,
    num_vertices: int,
) -> tuple[Tensor, Tensor]:
    """
    Compute a maximum spanning tree using Kruskal's algorithm.
    """
    edges_cpu = edge_index.detach().cpu()
    weights_cpu = edge_weight.detach().cpu()

    order = torch.argsort(
        weights_cpu,
        descending=True,
        stable=True,
    )

    parent = list(range(num_vertices))
    rank = [0] * num_vertices

    def find(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]

        return vertex

    def union(left: int, right: int) -> bool:
        left_root = find(left)
        right_root = find(right)

        if left_root == right_root:
            return False

        if rank[left_root] < rank[right_root]:
            parent[left_root] = right_root
        elif rank[left_root] > rank[right_root]:
            parent[right_root] = left_root
        else:
            parent[right_root] = left_root
            rank[left_root] += 1

        return True

    selected: list[int] = []

    for edge_id in order.tolist():
        source = int(edges_cpu[0, edge_id])
        target = int(edges_cpu[1, edge_id])

        if union(source, target):
            selected.append(edge_id)

        if len(selected) == num_vertices - 1:
            break

    if len(selected) != num_vertices - 1:
        raise RuntimeError("The graph is disconnected")

    selected_indices = torch.tensor(
        selected,
        dtype=torch.long,
    )

    return (
        edges_cpu.index_select(1, selected_indices),
        weights_cpu.index_select(0, selected_indices),
    )


@torch.no_grad()
def get_persistence_diagram(
    weight_matrix: Tensor,
    layer_input: Tensor,
) -> Tensor:
    """
    Compute one persistence diagram from one query activation.
    """
    edge_index, edge_weight = get_activation_weights(
        weight_matrix,
        layer_input,
    )

    num_outputs, num_inputs = weight_matrix.shape
    num_vertices = num_inputs + num_outputs

    _, mst_weights = get_maximum_spanning_tree(
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_vertices=num_vertices,
    )

    return torch.sort(
        mst_weights,
        descending=True,
    ).values


@torch.no_grad()
def get_captured_persistence_diagrams(
    captures: dict[int, dict[str, Tensor]],
    query_indices: list[Tensor],
    decoder_layer_indices: int | list[int] | None = None,
) -> dict[int, list[dict[int, Tensor]]]:
    """
    Compute diagrams for selected queries at selected decoder layers.

    Returns:
        {
            layer_id: [
                {query_id: diagram, ...},  # batch item 0
                {query_id: diagram, ...},  # batch item 1
            ]
        }
    """
    if decoder_layer_indices is None:
        decoder_layer_indices = sorted(captures)
    elif isinstance(decoder_layer_indices, int):
        decoder_layer_indices = [decoder_layer_indices]

    results: dict[int, list[dict[int, Tensor]]] = {}

    for layer_id in decoder_layer_indices:
        if layer_id not in captures:
            raise KeyError(
                f"Decoder layer {layer_id} was not captured"
            )

        capture = captures[layer_id]

        layer_inputs = capture["input"]
        weight_matrix = capture["weight"].detach().cpu()

        if len(query_indices) != layer_inputs.shape[0]:
            raise ValueError(
                "query_indices must contain one tensor per batch item"
            )

        batch_results: list[dict[int, Tensor]] = []

        for batch_id, indices in enumerate(query_indices):
            query_ids = indices.detach().to(
                device="cpu",
                dtype=torch.long,
            )

            if query_ids.numel() == 0:
                batch_results.append({})
                continue

            if query_ids.min() < 0 or query_ids.max() >= layer_inputs.shape[1]:
                raise IndexError("Query index is out of range")

            # Transfer only selected query features to the CPU.
            selected_inputs = layer_inputs[batch_id].index_select(
                0,
                query_ids.to(layer_inputs.device),
            ).detach().cpu()

            image_results: dict[int, Tensor] = {}

            for query_id, layer_input in zip(
                query_ids.tolist(),
                selected_inputs,
            ):
                image_results[query_id] = get_persistence_diagram(
                    weight_matrix=weight_matrix,
                    layer_input=layer_input,
                )

            batch_results.append(image_results)

        results[layer_id] = batch_results

    return results


class LayerClassBuckets:
    """
    Maintains a running persistence-diagram mean for every
    decoder-layer/class combination.
    """

    def __init__(self, num_layers: int, num_classes: int):
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.sums: list[list[Tensor | None]] = [
            [None for _ in range(num_classes)]
            for _ in range(num_layers)
        ]

        self.counts: list[list[int]] = [
            [0 for _ in range(num_classes)]
            for _ in range(num_layers)
        ]

    def update(
        self,
        diagram: Tensor,
        layer_id: int,
        class_id: int,
    ) -> None:
        self._validate_indices(layer_id, class_id)

        diagram = diagram.detach().cpu()

        current_sum = self.sums[layer_id][class_id]

        if current_sum is None:
            self.sums[layer_id][class_id] = diagram.clone()
        else:
            if current_sum.shape != diagram.shape:
                raise ValueError(
                    "All diagrams in a bucket must have the same shape"
                )

            current_sum.add_(diagram)

        self.counts[layer_id][class_id] += 1

    def frechet_mean(
        self,
        layer_id: int,
        class_id: int,
    ) -> Tensor:
        self._validate_indices(layer_id, class_id)

        count = self.counts[layer_id][class_id]
        diagram_sum = self.sums[layer_id][class_id]

        if count == 0 or diagram_sum is None:
            raise ValueError(
                f"No diagrams for layer {layer_id}, class {class_id}"
            )

        return diagram_sum / count

    def count(self, layer_id: int, class_id: int) -> int:
        self._validate_indices(layer_id, class_id)
        return self.counts[layer_id][class_id]

    def _validate_indices(
        self,
        layer_id: int,
        class_id: int,
    ) -> None:
        if not 0 <= layer_id < self.num_layers:
            raise IndexError(f"Invalid layer ID: {layer_id}")

        if not 0 <= class_id < self.num_classes:
            raise IndexError(f"Invalid class ID: {class_id}")