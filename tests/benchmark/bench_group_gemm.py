# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import torch
import triton

import tilegym
from tilegym.backend import is_backend_available, register_impl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def _split_tokens(total_tokens: int, num_groups: int, *, min_per_group: int = 1) -> list[int]:
    """Create a deterministic-ish ragged partition of total_tokens into num_groups parts."""
    if total_tokens < num_groups * min_per_group:
        raise ValueError(
            f"total_tokens ({total_tokens}) must be >= num_groups*min_per_group ({num_groups*min_per_group})"
        )

    remaining = total_tokens - num_groups * min_per_group
    if remaining == 0:
        return [min_per_group] * num_groups

    # Random weights -> proportional integer allocation
    weights = torch.rand((num_groups,), device="cpu")
    extras = torch.floor(weights / weights.sum() * remaining).to(torch.int64)

    # Fix rounding so sum(extras) == remaining
    diff = int(remaining - int(extras.sum().item()))
    if diff > 0:
        extras[:diff] += 1
    elif diff < 0:
        # Remove from the largest bins first
        _, idx = torch.sort(extras, descending=True)
        extras[idx[: (-diff)]] -= 1

    parts = (extras + min_per_group).tolist()
    assert sum(parts) == total_tokens
    return parts


def reference_group_gemm(
    group_A: list[torch.Tensor],
    group_B: list[torch.Tensor],
    static_persistent: bool = True,  # Unused; kept for interface compatibility
    use_tma: bool = True,  # Unused; kept for interface compatibility
    transpose_b: bool = False,
    **kwargs,
) -> list[torch.Tensor]:
    """Reference implementation using a Python loop.

    Group GEMM supports ragged shapes; the correct reference here is a loop over
    individual matmuls.
    """
    if len(group_A) != len(group_B):
        raise ValueError("group_A and group_B must have same length")

    out: list[torch.Tensor] = []
    for A, B in zip(group_A, group_B):
        out.append(A @ (B.t() if transpose_b else B))
    return out


# Register reference as a torch backend impl for the dispatcher
register_impl("group_gemm", "torch")(reference_group_gemm)


def _do_bench_triton(fn) -> tuple[float, float, float]:
    """Return (median_ms, max_ms, min_ms) using triton.testing.do_bench."""
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    return ms, max_ms, min_ms


ALL_BACKENDS = [
    ("cutile", "CuTile", ("orange", "-")) if is_backend_available("cutile") else None,
    ("torch", "PyTorch (loop ref)", ("green", "-")),
]


def get_supported_backends():
    return [p for p in ALL_BACKENDS if p is not None]


def create_benchmark_config(dtype: torch.dtype, G: int, N: int, K: int):
    backends, names, styles = zip(*get_supported_backends())
    dtype_name = str(dtype).split(".")[-1]

    return triton.testing.Benchmark(
        x_names=["T"],
        # total tokens (sum of M_i across groups)
        x_vals=[1024, 2048, 4096, 8192, 16384, 32768],
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="TFLOPS",
        plot_name=f"group_gemm_ragged-G{G}-N{N}-K{K}-{dtype_name}",
        args={"dtype": dtype, "G": G, "N": N, "K": K},
    )


@triton.testing.perf_report(
    [
        create_benchmark_config(dtype=torch.bfloat16, G=64, N=4096, K=4096),
        create_benchmark_config(dtype=torch.float16, G=64, N=4096, K=4096),
    ]
)
def bench_group_gemm(T, backend, dtype, G, N, K, device=DEVICE):
    # Deterministic inputs per (T, dtype, backend)
    torch.manual_seed(0)

    # Ragged group: A_i is (M_i, K), B_i is (K, N)
    Ms = _split_tokens(int(T), int(G), min_per_group=1)
    group_A = [torch.randn((Mi, K), device=device, dtype=dtype) for Mi in Ms]
    group_B = [torch.randn((K, N), device=device, dtype=dtype) for _ in range(G)]

    fn = lambda: tilegym.ops.group_gemm(group_A, group_B, transpose_b=False, backend=backend)

    # Correctness vs loop reference
    ref_out = reference_group_gemm(group_A, group_B, transpose_b=False)
    out = fn()
    assert len(out) == len(ref_out)
    for yo, yr in zip(out, ref_out):
        torch.testing.assert_close(yo, yr, rtol=1e-2, atol=1e-1)

    ms, max_ms, min_ms = _do_bench_triton(fn)

    # Total FLOPs across all groups: sum_i 2*M_i*N*K == 2*T*N*K
    perf = lambda t_ms: (2.0 * T * N * K) * 1e-12 / (t_ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


def _write_bf16_plot(save_dir: Path):
    """Fallback plot+csv for bf16 if Triton plotting doesn't emit images."""
    dtype = torch.bfloat16
    G, N, K = 64, 4096, 4096
    Ts = [1024, 2048, 4096, 8192, 16384, 32768]

    backends = get_supported_backends()
    backend_ids = [b[0] for b in backends]

    results: dict[str, list[float]] = {b: [] for b in backend_ids}

    for T in Ts:
        torch.manual_seed(0)
        Ms = _split_tokens(int(T), int(G), min_per_group=1)
        group_A = [torch.randn((Mi, K), device=DEVICE, dtype=dtype) for Mi in Ms]
        group_B = [torch.randn((K, N), device=DEVICE, dtype=dtype) for _ in range(G)]

        ref_out = reference_group_gemm(group_A, group_B, transpose_b=False)

        for backend in backend_ids:
            fn = lambda: tilegym.ops.group_gemm(group_A, group_B, transpose_b=False, backend=backend)
            out = fn()
            for yo, yr in zip(out, ref_out):
                torch.testing.assert_close(yo, yr, rtol=1e-2, atol=1e-1)

            ms, _, _ = _do_bench_triton(fn)
            tflops = (2.0 * T * N * K) * 1e-12 / (ms * 1e-3)
            results[backend].append(tflops)

    csv_path = save_dir / "group_gemm_ragged-G64-N4096-K4096-bfloat16.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        headers = ["T"] + [b[1] for b in backends]
        f.write(",".join(headers) + "\n")
        for i, T in enumerate(Ts):
            row = [str(T)] + [f"{results[b[0]][i]:.6f}" for b in backends]
            f.write(",".join(row) + "\n")

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print(f"\n[plot] Wrote {csv_path.name}, but could not write PNG (matplotlib unavailable): {e}")
        return

    plt.figure(figsize=(9, 5))
    for backend_id, name, style in backends:
        color, linestyle = style
        plt.plot(Ts, results[backend_id], label=name, color=color, linestyle=linestyle, marker="o")

    plt.title("group_gemm ragged (bf16, G=64, K=4096, N=4096)")
    plt.xlabel("Total tokens (T = sum of M_i)")
    plt.ylabel("TFLOPS")
    plt.grid(True, linestyle=":", linewidth=0.8)
    plt.legend()
    plt.tight_layout()

    png_path = save_dir / "group_gemm_ragged-G64-N4096-K4096-bfloat16.png"
    plt.savefig(png_path, dpi=160)
    plt.close()
    print(f"\n[plot] Wrote {png_path.name} (and {csv_path.name}) to {save_dir}")


if __name__ == "__main__":
    save_dir = Path(__file__).resolve().parent
    bench_group_gemm.run(print_data=True, save_path=str(save_dir))
    _write_bf16_plot(save_dir)
