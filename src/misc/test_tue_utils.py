import numpy as np
import torch
from scipy.sparse.csgraph import minimum_spanning_tree

# ---------------------------------------------------------------------
# Replace this import with your actual project import.
# ---------------------------------------------------------------------
from .tue_utils import _build_mst


def reference_mst(
    x: torch.Tensor,
    W: torch.Tensor,
) -> torch.Tensor:
    """
    Trusted CPU reference using SciPy.

    Parameters
    ----------
    x : [d_in]
    W : [d_out, d_in]

    Returns
    -------
    [d_in + d_out - 1]
        Sorted maximum spanning tree edge weights.
    """
    x_cpu = x.detach().cpu()
    W_cpu = W.detach().cpu()

    # Activation graph:
    # A[j, i] = |W[j, i] * x[i]|
    A = torch.abs(W_cpu * x_cpu.unsqueeze(0)).numpy()

    d_out, d_in = A.shape
    n_vertices = d_in + d_out

    # Construct full symmetric bipartite adjacency matrix.
    #
    # Vertices:
    #   0 ... d_in-1                 input
    #   d_in ... d_in+d_out-1        output
    #
    # SciPy computes a MINIMUM spanning tree, so negate weights.
    graph = np.zeros(
        (n_vertices, n_vertices),
        dtype=np.float64,
    )

    graph[:d_in, d_in:] = -A.T
    graph[d_in:, :d_in] = -A

    mst = minimum_spanning_tree(graph)

    weights = -mst.data

    # Persistence representation expects sorted weights.
    weights = np.sort(weights)[::-1].copy()

    return torch.from_numpy(weights)


def test_hand_computable_example():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = torch.tensor(
        [[1.0, 1.0]],
        device=device,
    )

    W = torch.tensor(
        [
            [10.0, 1.0],
            [2.0, 9.0],
        ],
        device=device,
    )

    result = _build_mst(X, W)

    expected = torch.tensor(
        [[10.0, 9.0, 2.0]],
        device=device,
    )

    torch.testing.assert_close(
        result,
        expected,
        rtol=0,
        atol=0,
    )

    print("✓ Hand-computable test passed")


def test_random_against_scipy(
    num_seeds: int = 100,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    shapes = [
        (2, 2),
        (3, 5),
        (5, 3),
        (8, 8),
        (7, 13),
        (16, 32),
    ]

    for seed in range(num_seeds):
        torch.manual_seed(seed)

        for d_in, d_out in shapes:
            batch_size = 8

            X = torch.randn(
                batch_size,
                d_in,
                device=device,
            )

            W = torch.randn(
                d_out,
                d_in,
                device=device,
            )

            result = _build_mst(X, W)

            expected_num_edges = d_in + d_out - 1

            assert result.shape == (
                batch_size,
                expected_num_edges,
            ), (
                f"Wrong output shape for seed={seed}, "
                f"d_in={d_in}, d_out={d_out}: "
                f"{result.shape}"
            )

            for b in range(batch_size):
                reference = reference_mst(
                    X[b],
                    W,
                ).to(
                    device=result.device,
                    dtype=result.dtype,
                )

                try:
                    torch.testing.assert_close(
                        result[b],
                        reference,
                        rtol=1e-5,
                        atol=1e-6,
                    )

                except AssertionError:
                    print("\nFAILED")
                    print(f"seed       = {seed}")
                    print(f"d_in       = {d_in}")
                    print(f"d_out      = {d_out}")
                    print(f"batch_idx  = {b}")

                    print("\nGPU result:")
                    print(result[b])

                    print("\nSciPy reference:")
                    print(reference)

                    print("\nGPU total weight:")
                    print(result[b].sum())

                    print("\nReference total weight:")
                    print(reference.sum())

                    raise

    print(f"✓ Random SciPy comparison passed for {num_seeds} seeds")


def test_sorted_output():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = torch.randn(
        16,
        10,
        device=device,
    )

    W = torch.randn(
        20,
        10,
        device=device,
    )

    result = _build_mst(X, W)

    assert torch.all(result[:, :-1] >= result[:, 1:]), (
        "MST weights are not sorted descending"
    )

    print("✓ Output sorting test passed")


def test_batch_consistency():
    """
    Verifies that computing a batch at once gives the same result
    as computing each sample as a batch of size 1.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(123)

    X = torch.randn(
        12,
        9,
        device=device,
    )

    W = torch.randn(
        15,
        9,
        device=device,
    )

    batched = _build_mst(X, W)

    individual = torch.cat(
        [_build_mst(X[i : i + 1], W) for i in range(X.shape[0])],
        dim=0,
    )

    torch.testing.assert_close(
        batched,
        individual,
        rtol=1e-5,
        atol=1e-6,
    )

    print("✓ Batched vs individual test passed")


def main():
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    print()

    test_hand_computable_example()
    test_sorted_output()
    test_batch_consistency()
    test_random_against_scipy(num_seeds=100)

    print()
    print("All MST tests passed ✓")


if __name__ == "__main__":
    main()
