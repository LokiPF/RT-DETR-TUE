import torch
from torch import Tensor

def persistence_diagram_distance(signature, mean_diagram):
    # TODO: ablation study
    if signature.shape != mean_diagram.shape:
        raise ValueError("Persistence diagram sizes differ.")

    distance = torch.sqrt(
        torch.mean((signature - mean_diagram) ** 2)
    )

    return distance.to(
        device=activation.device,
        dtype=torch.float32,
    )

def maximum_spanning_tree_signature(
    activation: Tensor,
    weight: Tensor,
    edge_score: str,
) -> Tensor:
    """
    Return the descending maximum-spanning-tree edge weights.

    The activation graph is a complete bipartite graph:
      - one partition contains the selected Linear layer's input neurons;
      - the other contains its output neurons;
      - edge (i, j) is filtered by |w_ji * x_i| by default.

    Under a descending filtration, the maximum spanning tree contains the
    finite 0D persistence merge values. Because every sample uses the same
    layer, all signatures have the same fixed length:
    n_input + n_output - 1.
    """
    x = activation.detach().float().flatten().cpu()
    w = weight.detach().float().cpu()
    if w.ndim != 2 or w.shape[1] != x.numel():
        raise ValueError(
            f"Linear weight shape {tuple(w.shape)} is incompatible with "
            f"activation shape {tuple(activation.shape)}."
        )

    if edge_score == "abs_wx":
        values = w.abs() * x.abs().unsqueeze(0)
    elif edge_score == "abs_w":
        values = w.abs()
    else:
        raise ValueError(f"Unknown edge score: {edge_score}")

    n_out, n_in = values.shape
    flat_values = values.reshape(-1)
    edge_order = torch.argsort(flat_values, descending=True, stable=True)

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

    selected: list[Tensor] = []
    for flat_index_tensor in edge_order:
        flat_index = int(flat_index_tensor)
        output_index = flat_index // n_in
        input_index = flat_index % n_in
        if union(input_index, n_in + output_index):
            selected.append(flat_values[flat_index])
            if len(selected) == vertex_count - 1:
                break

    expected = vertex_count - 1
    if len(selected) != expected:
        raise RuntimeError(
            f"Maximum spanning tree contains {len(selected)} edges; expected "
            f"{expected}."
        )
    return torch.stack(selected).contiguous()
