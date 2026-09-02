from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# The graph structure is identical for every query with the same dimensions.
_EDGE_INDEX_CACHE: dict[tuple[int, int], Tensor] = {}


def empirical_cdf(sorted_distances: torch.Tensor, measured) -> torch.Tensor:
    total = sorted_distances.numel()
    measured = torch.as_tensor(
        measured, dtype=sorted_distances.dtype, device=sorted_distances.device
    )
    if total == 0:
        return torch.full_like(measured, float("nan"), dtype=torch.float32)
    count_le = torch.searchsorted(sorted_distances, measured, right=True).to(
        torch.float32
    )
    return count_le / total


def conformal_pvalue(sorted_distances: torch.Tensor, measured) -> torch.Tensor:
    total = sorted_distances.numel()
    measured = torch.as_tensor(
        measured, dtype=sorted_distances.dtype, device=sorted_distances.device
    )
    if total == 0:
        return torch.full_like(measured, float("nan"), dtype=torch.float32)
    count_lt = torch.searchsorted(sorted_distances, measured, right=False).to(
        torch.float32
    )
    count_ge = total - count_lt
    return (1.0 + count_ge) / (total + 1.0)


def hook_decoder_layers(
    transformer: nn.Module,
    decoder_layers: int | Iterable[int] | None = None,
    head_task: str = "score",
    bbox_head_layer: int = -1,
) -> tuple[
    dict[int, dict[str, Tensor]],
    list[torch.utils.hooks.RemovableHandle],
    list[int],
]:
    decoder = transformer.decoder
    if head_task == "score":
        heads = transformer.dec_score_head
    elif head_task == "bbox":
        heads = transformer.dec_bbox_head
    else:
        raise ValueError(f"Invalid head task: {head_task}")
    num_layers = len(decoder.layers)

    if decoder_layers is None:
        requested_layers = list(range(num_layers))
    elif isinstance(decoder_layers, int):
        requested_layers = [decoder_layers]
    else:
        requested_layers = list(decoder_layers)

    normalized_layers: list[int] = []

    for layer_id in requested_layers:
        if not -num_layers <= layer_id < num_layers:
            raise IndexError(f"Invalid decoder layer: {layer_id}")

        layer_id = layer_id % num_layers

        if layer_id not in normalized_layers:
            normalized_layers.append(layer_id)

    captures: dict[int, dict[str, Tensor]] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for layer_id in normalized_layers:
        if head_task == "score":
            decoder_layer = decoder.layers[layer_id]
            score_head = heads[layer_id]

            if not isinstance(score_head, nn.Linear):
                raise TypeError(
                    "Expected every score head to be nn.Linear, "
                    f"got {type(score_head).__name__} at decoder layer "
                    f"{layer_id}"
                )

            def make_score_hook(
                index: int,
                head: nn.Linear,
            ):
                def hook(
                    module: nn.Module,
                    inputs: tuple[Tensor, ...],
                    output: Tensor,
                ) -> None:
                    features = output.detach()
                    weight = head.weight.detach()
                    bias = head.bias.detach() if head.bias is not None else None
                    head_output = F.linear(
                        features,
                        weight,
                        bias,
                    ).detach()

                    captures[index] = {
                        "input": features,
                        "weight": weight,
                        "output": head_output,
                        # Kept for compatibility with classification code.
                        "logits": head_output,
                    }

                return hook

            handle = decoder_layer.register_forward_hook(
                make_score_hook(layer_id, score_head)
            )
        else:
            bbox_head = heads[layer_id]
            bbox_layers = getattr(bbox_head, "layers", None)

            if bbox_layers is None:
                raise TypeError(
                    "Expected every bbox head to expose its linear layers "
                    f"through '.layers', got {type(bbox_head).__name__} at "
                    f"decoder layer {layer_id}"
                )

            num_bbox_layers = len(bbox_layers)
            if not -num_bbox_layers <= bbox_head_layer < num_bbox_layers:
                raise IndexError(
                    f"Invalid bbox head layer {bbox_head_layer} for decoder "
                    f"layer {layer_id}; expected an index in "
                    f"[-{num_bbox_layers}, {num_bbox_layers - 1}]"
                )

            bbox_layer_id = bbox_head_layer % num_bbox_layers
            bbox_linear = bbox_layers[bbox_layer_id]

            if not isinstance(bbox_linear, nn.Linear):
                raise TypeError(
                    "Expected the selected bbox head layer to be nn.Linear, "
                    f"got {type(bbox_linear).__name__} at decoder layer "
                    f"{layer_id}, bbox head layer {bbox_layer_id}"
                )

            score_head = transformer.dec_score_head[layer_id]
            if not isinstance(score_head, nn.Linear):
                raise TypeError(
                    "Expected every score head to be nn.Linear, "
                    f"got {type(score_head).__name__} at decoder layer "
                    f"{layer_id}"
                )

            def make_bbox_score_hook(
                index: int,
                head: nn.Linear,
            ):
                def hook(
                    module: nn.Module,
                    inputs: tuple[Tensor, ...],
                    output: Tensor,
                ) -> None:
                    captures.setdefault(index, {})["logits"] = F.linear(
                        output.detach(),
                        head.weight.detach(),
                        (head.bias.detach() if head.bias is not None else None),
                    ).detach()

                return hook

            # Preserve class conditioning without treating the four bbox
            # coordinates as class logits.
            score_handle = decoder.layers[layer_id].register_forward_hook(
                make_bbox_score_hook(layer_id, score_head)
            )
            handles.append(score_handle)

            def make_bbox_hook(index: int):
                def hook(
                    module: nn.Module,
                    inputs: tuple[Tensor, ...],
                    output: Tensor,
                ) -> None:
                    if not inputs or not isinstance(inputs[0], Tensor):
                        raise TypeError(
                            "Expected the bbox linear layer's first input "
                            "to be a Tensor"
                        )

                    captures.setdefault(index, {}).update(
                        input=inputs[0].detach(),
                        weight=module.weight.detach(),
                        output=output.detach(),
                    )

                return hook

            # Unlike intermediate score heads, bbox heads really execute at
            # evaluation time, so capture the selected MLP layer directly.
            handle = bbox_linear.register_forward_hook(make_bbox_hook(layer_id))

        handles.append(handle)

    return captures, handles, normalized_layers


def diagram_distance(diagram: Tensor, reference: Tensor) -> Tensor:
    diagram = torch.sort(
        diagram.detach().cpu().float().flatten(),
        descending=True,
    ).values
    reference = reference.detach().cpu().float().flatten()

    if diagram.shape != reference.shape:
        raise ValueError(
            "Diagram/reference shape mismatch: "
            f"{tuple(diagram.shape)} versus {tuple(reference.shape)}"
        )

    return torch.sqrt(torch.mean((diagram - reference) ** 2))


def _get_bipartite_edge_index(
    num_inputs: int,
    num_outputs: int,
) -> Tensor:
    cache_key = (num_inputs, num_outputs)

    if cache_key not in _EDGE_INDEX_CACHE:
        input_vertices = torch.arange(num_inputs).repeat(num_outputs)

        output_vertices = (num_inputs + torch.arange(num_outputs)).repeat_interleave(
            num_inputs
        )

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
    weight_matrix = weight_matrix.detach().cpu()
    layer_input = layer_input.detach().cpu()

    if weight_matrix.ndim != 2:
        raise ValueError("weight_matrix must be two-dimensional")

    if layer_input.ndim != 1:
        raise ValueError("layer_input must be one-dimensional")

    num_outputs, num_inputs = weight_matrix.shape

    if layer_input.shape[0] != num_inputs:
        raise ValueError(
            f"Expected input dimension {num_inputs}, got {layer_input.shape[0]}"
        )

    activation_matrix = torch.abs(weight_matrix * layer_input.unsqueeze(0))

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
def _batched_prim(
    weight_matrix: Tensor,
    layer_inputs: Tensor,
) -> Tensor:
    """
    Exact maximum spanning tree for the complete bipartite graph
    induced by a score head.

    weight_matrix: [C, H]
    layer_inputs:  [K, H]
    returns:       [K, H + C - 1]
    """
    if weight_matrix.ndim != 2:
        raise ValueError("weight_matrix must have shape [C, H]")

    if layer_inputs.ndim != 2:
        raise ValueError("layer_inputs must have shape [K, H]")

    device = layer_inputs.device

    weight_matrix = weight_matrix.detach().to(
        device=device,
        dtype=torch.float32,
    )
    layer_inputs = layer_inputs.detach().to(
        device=device,
        dtype=torch.float32,
    )

    num_queries = layer_inputs.shape[0]
    num_outputs, num_inputs = weight_matrix.shape

    if layer_inputs.shape[1] != num_inputs:
        raise ValueError("Input dimensions do not match")

    if num_queries == 0:
        return torch.empty(
            (0, num_inputs + num_outputs - 1),
            dtype=torch.float32,
            device=device,
        )

    # [K, C, H]
    edge_weights = torch.abs(layer_inputs[:, None, :] * weight_matrix[None, :, :])

    # Each input vertex can safely take its strongest incident edge.
    # [K, H]
    input_tree_weights, input_owner = edge_weights.max(dim=1)

    if num_outputs == 1:
        return torch.sort(
            input_tree_weights,
            dim=1,
            descending=True,
        ).values

    # directed[a, b] is the strongest edge from an input currently
    # attached to output a into output b.
    directed = []

    for owner in range(num_outputs):
        owner_mask = (input_owner == owner).unsqueeze(1)

        directed.append(
            edge_weights.masked_fill(
                ~owner_mask,
                -torch.inf,
            ).amax(dim=-1)
        )

    # [K, C, C]
    directed = torch.stack(directed, dim=1)

    # Either endpoint may own the connecting input.
    output_graph = torch.maximum(
        directed,
        directed.transpose(1, 2),
    )

    diagonal = torch.eye(
        num_outputs,
        dtype=torch.bool,
        device=device,
    ).unsqueeze(0)

    output_graph = output_graph.masked_fill(
        diagonal,
        -torch.inf,
    )

    # Prim now runs only over C output vertices.
    # With four classes, this is three iterations rather than 259.
    selected = torch.zeros(
        (num_queries, num_outputs),
        dtype=torch.bool,
        device=device,
    )
    selected[:, 0] = True

    best = output_graph[:, 0, :]
    bridge_weights = []

    for _ in range(num_outputs - 1):
        candidates = best.masked_fill(
            selected,
            -torch.inf,
        )

        selected_weight, selected_vertex = candidates.max(dim=1)
        bridge_weights.append(selected_weight)

        selected.scatter_(
            1,
            selected_vertex[:, None],
            True,
        )

        new_weights = output_graph.gather(
            1,
            selected_vertex[:, None, None].expand(
                -1,
                1,
                num_outputs,
            ),
        ).squeeze(1)

        best = torch.maximum(best, new_weights)

    bridge_weights = torch.stack(
        bridge_weights,
        dim=1,
    )

    mst_weights = torch.cat(
        [
            input_tree_weights,
            bridge_weights,
        ],
        dim=1,
    )

    return torch.sort(
        mst_weights,
        dim=1,
        descending=True,
    ).values


@torch.no_grad()
def __batched_prim(
    weight_matrix: Tensor,
    layer_inputs: Tensor,
) -> Tensor:
    """
    Compute maximum-spanning-tree weights for multiple queries.

    Args:
        weight_matrix: [C, H]
        layer_inputs:  [K, H]

    Returns:
        Sorted MST weights: [K, H + C - 1]
    """
    if weight_matrix.ndim != 2:
        raise ValueError("weight_matrix must have shape [C, H]")

    if layer_inputs.ndim != 2:
        raise ValueError("layer_inputs must have shape [K, H]")

    device = layer_inputs.device

    weight_matrix = weight_matrix.detach().to(
        device=device,
        dtype=torch.float32,
    )
    layer_inputs = layer_inputs.detach().to(
        device=device,
        dtype=torch.float32,
    )

    num_queries = layer_inputs.shape[0]
    num_outputs, num_inputs = weight_matrix.shape

    if layer_inputs.shape[1] != num_inputs:
        raise ValueError("Input dimensions do not match")

    if num_queries == 0:
        return torch.empty(
            (0, num_inputs + num_outputs - 1),
            device=device,
        )

    # [K, C, H]
    edge_weights = torch.abs(layer_inputs[:, None, :] * weight_matrix[None, :, :])

    selected_inputs = torch.zeros(
        (num_queries, num_inputs),
        dtype=torch.bool,
        device=device,
    )
    selected_outputs = torch.zeros(
        (num_queries, num_outputs),
        dtype=torch.bool,
        device=device,
    )

    # Start each tree from input vertex zero.
    selected_inputs[:, 0] = True

    best_input_weights = torch.full(
        (num_queries, num_inputs),
        -torch.inf,
        device=device,
    )

    # Every output can initially connect to input zero.
    best_output_weights = edge_weights[:, :, 0].clone()

    mst_weights = torch.empty(
        (
            num_queries,
            num_inputs + num_outputs - 1,
        ),
        dtype=edge_weights.dtype,
        device=device,
    )

    for step in range(num_inputs + num_outputs - 1):
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

        is_input = selected_vertex < num_inputs

        input_index = selected_vertex.clamp(max=num_inputs - 1)
        output_index = (selected_vertex - num_inputs).clamp(
            min=0,
            max=num_outputs - 1,
        )

        # Mark selected input vertices without affecting batches
        # that selected an output vertex.
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

        # If an input was selected, update connections to outputs.
        new_output_weights = edge_weights.gather(
            2,
            input_index[:, None, None].expand(
                -1,
                num_outputs,
                1,
            ),
        ).squeeze(2)

        best_output_weights = torch.where(
            is_input[:, None],
            torch.maximum(
                best_output_weights,
                new_output_weights,
            ),
            best_output_weights,
        )

        # If an output was selected, update connections to inputs.
        new_input_weights = edge_weights.gather(
            1,
            output_index[:, None, None].expand(
                -1,
                1,
                num_inputs,
            ),
        ).squeeze(1)

        best_input_weights = torch.where(
            (~is_input)[:, None],
            torch.maximum(
                best_input_weights,
                new_input_weights,
            ),
            best_input_weights,
        )

    return torch.sort(
        mst_weights,
        dim=1,
        descending=True,
    ).values


@torch.no_grad()
def get_persistence_diagrams_batched(
    weight_matrix: Tensor,
    layer_inputs: Tensor,
    chunk_size: int = 64,
) -> Tensor:
    """
    Args:
        weight_matrix: [C, H]
        layer_inputs:  [K, H]

    Returns:
        diagrams: [K, H + C - 1]
    """
    if layer_inputs.shape[0] == 0:
        diagram_size = weight_matrix.shape[0] + weight_matrix.shape[1] - 1

        return torch.empty(
            (0, diagram_size),
            device=layer_inputs.device,
        )

    chunks = [
        _batched_prim(
            weight_matrix,
            input_chunk,
        )
        for input_chunk in layer_inputs.split(
            chunk_size,
            dim=0,
        )
    ]

    return torch.cat(chunks, dim=0)


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
    chunk_size: int = 64,
) -> dict[int, list[dict[int, Tensor]]]:
    if decoder_layer_indices is None:
        decoder_layer_indices = sorted(captures)
    elif isinstance(decoder_layer_indices, int):
        decoder_layer_indices = [decoder_layer_indices]

    results = {}

    for layer_id in decoder_layer_indices:
        capture = captures[layer_id]
        layer_inputs = capture["input"]
        weight_matrix = capture["weight"]

        batch_results = [{} for _ in range(layer_inputs.shape[0])]

        selected_inputs = []
        locations: list[tuple[int, int]] = []

        for batch_id, indices in enumerate(query_indices):
            query_ids = indices.detach().to(
                device=layer_inputs.device,
                dtype=torch.long,
            )

            if query_ids.numel() == 0:
                continue

            selected_inputs.append(
                layer_inputs[batch_id].index_select(
                    0,
                    query_ids,
                )
            )

            locations.extend(
                (batch_id, query_id) for query_id in query_ids.cpu().tolist()
            )

        if selected_inputs:
            flat_inputs = torch.cat(
                selected_inputs,
                dim=0,
            )

            flat_diagrams = get_persistence_diagrams_batched(
                weight_matrix=weight_matrix,
                layer_inputs=flat_inputs,
                chunk_size=chunk_size,
            ).cpu()

            # This loop only reconstructs the output structure;
            # the expensive MST work is already batched.
            for location, diagram in zip(
                locations,
                flat_diagrams,
            ):
                batch_id, query_id = location
                batch_results[batch_id][query_id] = diagram

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
            [None for _ in range(num_classes)] for _ in range(num_layers)
        ]

        self.counts: list[list[int]] = [
            [0 for _ in range(num_classes)] for _ in range(num_layers)
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
                raise ValueError("All diagrams in a bucket must have the same shape")

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
            raise ValueError(f"No diagrams for layer {layer_id}, class {class_id}")

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
