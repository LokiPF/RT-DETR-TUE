import time

import torch

from .tue_utils import _build_mst, _build_mst_sparse


def benchmark_mst_approx(
    X,
    W,
    exact_fn,
    approx_fn,
    ks=(2, 4, 8, 16, 32),
    warmup=3,
    repeats=10,
):
    device = X.device

    # ---------------------------------------------------------
    # Exact reference
    # ---------------------------------------------------------
    exact = exact_fn(X, W)

    exact_total = exact.sum(dim=1)  # [B]

    print(f"X: {tuple(X.shape)}")
    print(f"W: {tuple(W.shape)}")
    print()
    print(
        f"{'k':>4} | "
        f"{'total err %':>12} | "
        f"{'edge MAE %':>11} | "
        f"{'p95 total %':>11} | "
        f"{'max total %':>11} | "
        f"{'time ms':>10} | "
        f"{'speedup':>8}"
    )
    print("-" * 87)

    # ---------------------------------------------------------
    # Timing helper
    # ---------------------------------------------------------
    def measure(fn):
        for _ in range(warmup):
            fn()

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(repeats):
            fn()

        if device.type == "cuda":
            torch.cuda.synchronize()

        return (time.perf_counter() - start) * 1000 / repeats

    exact_ms = measure(lambda: exact_fn(X, W))

    # ---------------------------------------------------------
    # Test each k
    # ---------------------------------------------------------
    for k in ks:
        approx = approx_fn(X, W, k=k)

        # ---------------------------------------------
        # 1. Total MST weight relative error
        # ---------------------------------------------
        approx_total = approx.sum(dim=1)

        total_rel_err = (
            approx_total - exact_total
        ).abs() / exact_total.abs().clamp_min(1e-12)

        mean_total_pct = 100 * total_rel_err.mean()
        p95_total_pct = 100 * torch.quantile(total_rel_err, 0.95)
        max_total_pct = 100 * total_rel_err.max()

        # ---------------------------------------------
        # 2. Edge-by-edge error
        #
        # Both functions return sorted edge weights,
        # so this compares their MST weight spectra.
        # ---------------------------------------------
        edge_rel_err = (approx - exact).abs() / exact.abs().clamp_min(1e-12)

        mean_edge_pct = 100 * edge_rel_err.mean()

        # ---------------------------------------------
        # Speed
        # ---------------------------------------------
        approx_ms = measure(lambda: approx_fn(X, W, k=k))

        speedup = exact_ms / approx_ms

        print(
            f"{k:4d} | "
            f"{mean_total_pct.item():12.4f} | "
            f"{mean_edge_pct.item():11.4f} | "
            f"{p95_total_pct.item():11.4f} | "
            f"{max_total_pct.item():11.4f} | "
            f"{approx_ms:10.3f} | "
            f"{speedup:7.2f}x"
        )

    print()
    print(f"Exact MST time: {exact_ms:.3f} ms")


torch.manual_seed(0)

device = "cuda"

B = 128
d_in = 512
d_out = 512

X = torch.randn(
    B,
    d_in,
    device=device,
)

W = torch.randn(
    d_out,
    d_in,
    device=device,
)

benchmark_mst_approx(
    X,
    W,
    exact_fn=_build_mst,
    approx_fn=_build_mst_sparse,
    ks=(2, 4, 8, 16, 32),
)
