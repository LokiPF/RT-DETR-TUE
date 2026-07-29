import torch
from torch import Tensor


_K4_EDGE_PAIRS = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
)

# Edge indices for all 16 spanning trees of K4. Edges use _K4_EDGE_PAIRS.
_K4_SPANNING_TREES = (
    (0, 1, 2),
    (0, 1, 4),
    (0, 1, 5),
    (0, 2, 3),
    (0, 2, 5),
    (0, 3, 4),
    (0, 3, 5),
    (0, 4, 5),
    (1, 2, 3),
    (1, 2, 4),
    (1, 3, 4),
    (1, 3, 5),
    (1, 4, 5),
    (2, 3, 4),
    (2, 3, 5),
    (2, 4, 5),
)


def persistence_diagram_distance(
    signature: Tensor,
    mean_diagram: Tensor,
) -> Tensor:
    if signature.shape != mean_diagram.shape:
        raise ValueError("Persistence diagram sizes differ.")

    return torch.sqrt(
        torch.mean((signature - mean_diagram) ** 2)
    ).to(device=signature.device, dtype=torch.float32)


@torch.no_grad()
def maximum_spanning_tree_signature_batch(
    activations: Tensor,
    weight: Tensor,
    edge_score: str = "abs_wx",
) -> Tensor:
    """Return exact MST signatures for a batch of bbox-head activations.

    Args:
        activations: Tensor with shape ``[..., n_input]``.
        weight: Linear-layer weight with shape ``[4, n_input]``.
        edge_score: ``"abs_wx"`` or ``"abs_w"``.

    Returns:
        Tensor with shape ``[..., n_input + 3]``. Each signature is sorted
        descending, matching descending-filtration Kruskal.

    For K_(n_input, 4), every input vertex's strongest incident edge can be
    included simultaneously without forming a cycle. These n_input edges form
    at most four components, one per output vertex. Completing their maximum
    spanning tree therefore reduces exactly to choosing the best of the 16
    spanning trees on four contracted output components.
    """
    if activations.ndim < 1:
        raise ValueError("activations must have at least one dimension.")

    n_input = activations.shape[-1]
    if weight.ndim != 2 or weight.shape[1] != n_input:
        raise ValueError(
            f"Linear weight shape {tuple(weight.shape)} is incompatible with "
            f"activation shape {tuple(activations.shape)}."
        )
    if weight.shape[0] != 4:
        raise ValueError(
            "The batched exact implementation requires four bbox outputs; "
            f"received {weight.shape[0]}."
        )

    leading_shape = activations.shape[:-1]
    x = activations.detach().reshape(-1, n_input).float()
    w = weight.detach().to(device=x.device, dtype=torch.float32)

    if edge_score == "abs_wx":
        values = w.abs().unsqueeze(0) * x.abs().unsqueeze(1)
    elif edge_score == "abs_w":
        values = w.abs().unsqueeze(0).expand(x.shape[0], -1, -1)
    else:
        raise ValueError(f"Unknown edge score: {edge_score}")

    # One safe maximum edge for every input vertex.
    base_values, assigned_output = values.max(dim=1)

    # The strongest edge crossing each pair of contracted output components.
    bridge_values = []
    for left_output, right_output in _K4_EDGE_PAIRS:
        assigned_left = assigned_output.eq(left_output)
        assigned_right = assigned_output.eq(right_output)

        left_to_right = values[:, right_output, :].masked_fill(
            ~assigned_left,
            float("-inf"),
        ).amax(dim=-1)
        right_to_left = values[:, left_output, :].masked_fill(
            ~assigned_right,
            float("-inf"),
        ).amax(dim=-1)
        bridge_values.append(
            torch.maximum(left_to_right, right_to_left)
        )

    bridge_values = torch.stack(bridge_values, dim=-1)
    tree_edge_indices = torch.tensor(
        _K4_SPANNING_TREES,
        dtype=torch.long,
        device=x.device,
    )
    candidate_tree_values = bridge_values[:, tree_edge_indices]
    candidate_tree_scores = candidate_tree_values.sum(dim=-1)
    best_tree_index = candidate_tree_scores.argmax(dim=-1)
    batch_index = torch.arange(x.shape[0], device=x.device)
    selected_bridge_values = candidate_tree_values[
        batch_index,
        best_tree_index,
    ]

    signatures = torch.cat(
        (base_values, selected_bridge_values),
        dim=-1,
    ).sort(dim=-1, descending=True, stable=True).values
    return signatures.reshape(*leading_shape, n_input + 3).contiguous()


@torch.no_grad()
def maximum_spanning_tree_signature(
    activation: Tensor,
    weight: Tensor,
    edge_score: str,
) -> Tensor:
    """Scalar compatibility wrapper for one bbox-head activation."""
    x = activation.detach().float().flatten()
    if weight.ndim != 2 or weight.shape[1] != x.numel():
        raise ValueError(
            f"Linear weight shape {tuple(weight.shape)} is incompatible with "
            f"activation shape {tuple(activation.shape)}."
        )

    if weight.shape[0] == 4:
        return maximum_spanning_tree_signature_batch(
            x.unsqueeze(0),
            weight,
            edge_score=edge_score,
        )[0]

    # Generic scalar fallback for non-bbox Linear layers.
    w = weight.detach().to(device=x.device, dtype=torch.float32)
    if edge_score == "abs_wx":
        values = w.abs() * x.abs().unsqueeze(0)
    elif edge_score == "abs_w":
        values = w.abs()
    else:
        raise ValueError(f"Unknown edge score: {edge_score}")

    n_out, n_in = values.shape
    flat_values = values.reshape(-1)
    edge_order = torch.argsort(
        flat_values,
        descending=True,
        stable=True,
    ).cpu()

    vertex_count = n_in + n_out
    parent = list(range(vertex_count))
    rank = [0] * vertex_count

    def find(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    def union(left: int, right: int) -> bool:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return False
        if rank[root_left] < rank[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        if rank[root_left] == rank[root_right]:
            rank[root_left] += 1
        return True

    selected_indices = []
    for flat_index in edge_order.tolist():
        output_index = flat_index // n_in
        input_index = flat_index % n_in
        if union(input_index, n_in + output_index):
            selected_indices.append(flat_index)
            if len(selected_indices) == vertex_count - 1:
                break

    expected = vertex_count - 1
    if len(selected_indices) != expected:
        raise RuntimeError(
            f"Maximum spanning tree contains {len(selected_indices)} edges; "
            f"expected {expected}."
        )

    index = torch.tensor(
        selected_indices,
        dtype=torch.long,
        device=flat_values.device,
    )
    return flat_values[index].contiguous()

