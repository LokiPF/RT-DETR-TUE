import torch

from .tue_dataclasses import CaptureGroup, FrechetAccumulator


@torch.no_grad()
def _build_mst(
    X: torch.Tensor,  # [B, d_in]
    W: torch.Tensor,  # [d_out, d_in]
) -> torch.Tensor:
    if X.ndim != 2:
        raise ValueError(f"Expected X [B, d_in], got {X.shape}")

    if W.ndim != 2:
        raise ValueError(f"Expected W [d_out, d_in], got {W.shape}")

    B, d_in = X.shape
    d_out, w_d_in = W.shape

    if d_in != w_d_in:
        raise ValueError(f"Input dimension mismatch: X={X.shape}, W={W.shape}")

    device = X.device
    dtype = X.dtype

    # Absolute values only need to be calculated once.
    X_abs = X.abs()
    W_abs = W.abs()

    visited_in = torch.zeros(
        (B, d_in),
        dtype=torch.bool,
        device=device,
    )
    visited_out = torch.zeros(
        (B, d_out),
        dtype=torch.bool,
        device=device,
    )

    # Start at input vertex 0.
    visited_in[:, 0] = True

    # Edges from input vertex 0 -> every output vertex.
    # [B, d_out]
    best_to_out = X_abs[:, 0:1] * W_abs[:, 0].unsqueeze(0)

    best_to_in = torch.full(
        (B, d_in),
        -torch.inf,
        dtype=dtype,
        device=device,
    )

    n_edges = d_in + d_out - 1

    mst_weights = torch.empty(
        (B, n_edges),
        dtype=dtype,
        device=device,
    )

    batch_idx = torch.arange(B, device=device)

    for k in range(n_edges):
        out_candidates = best_to_out.masked_fill(
            visited_out,
            -torch.inf,
        )
        in_candidates = best_to_in.masked_fill(
            visited_in,
            -torch.inf,
        )

        max_out, out_idx = out_candidates.max(dim=1)
        max_in, in_idx = in_candidates.max(dim=1)

        choose_out = max_out >= max_in
        choose_in = ~choose_out

        mst_weights[:, k] = torch.where(
            choose_out,
            max_out,
            max_in,
        )

        # --------------------------------------------------
        # Candidate edges introduced by selected output node.
        #
        # W_abs[out_idx] -> [B, d_in]
        # --------------------------------------------------
        candidate_in = W_abs[out_idx] * X_abs

        updated_in = torch.maximum(
            best_to_in,
            candidate_in,
        )

        best_to_in = torch.where(
            choose_out[:, None],
            updated_in,
            best_to_in,
        )

        # --------------------------------------------------
        # Candidate edges introduced by selected input node.
        #
        # W_abs[:, in_idx] -> [d_out, B]
        # transpose          -> [B, d_out]
        # --------------------------------------------------
        candidate_out = W_abs[:, in_idx].T * X_abs[batch_idx, in_idx, None]

        updated_out = torch.maximum(
            best_to_out,
            candidate_out,
        )

        best_to_out = torch.where(
            choose_in[:, None],
            updated_out,
            best_to_out,
        )

        # Mark newly selected vertices.
        visited_out[batch_idx, out_idx] |= choose_out
        visited_in[batch_idx, in_idx] |= choose_in

    return mst_weights.sort(
        dim=1,
        descending=True,
    ).values


def diagram_distance_1d(mu, nu):
    return torch.sqrt(torch.mean((mu - nu) ** 2, dim=1))


def build_frechet_mean(
    captures: CaptureGroup,
    batch_indices: torch.Tensor,
    query_indices: torch.Tensor,
    class_indices: torch.Tensor,
    acc: FrechetAccumulator,
):
    if query_indices.numel() == 0:
        return

    for layer_name, layer in captures.data.items():
        W = layer.weight
        X = layer.input[
            batch_indices,
            query_indices,
        ]
        diagrams = _build_mst(X, W)

        acc.update_batch(
            class_indices=class_indices,
            layer_name=layer_name,
            diagrams=diagrams,
        )


def topological_uncertainty_from_captures(
    captures: CaptureGroup,
    batch_idx: torch.Tensor,
    query_idx: torch.Tensor,
    class_idx: torch.Tensor,
    frechet_means: dict,
    distance_measure=diagram_distance_1d,
):
    tu = None

    for layer_name, layer in captures.data.items():
        X = layer.input[batch_idx, query_idx]
        W = layer.weight

        diagram = _build_mst(X, W)
        references = torch.stack(
            [frechet_means[int(cls)][layer_name] for cls in class_idx]
        ).to(device=diagram.device, dtype=diagram.dtype)

        distances = distance_measure(diagram, references)

        if tu is None:
            tu = torch.zeros_like(distances)

        tu += distances

    tu /= len(captures.data)

    return tu
