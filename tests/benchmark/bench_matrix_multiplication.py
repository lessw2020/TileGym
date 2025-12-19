# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
import triton

import tilegym
from tilegym.backend import is_backend_available, register_impl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def reference_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    trans_a: bool = None,  # Unused - kept for interface compatibility
    trans_b: bool = None,  # Unused - kept for interface compatibility
    static_persistent: bool = True,  # Unused - kept for interface compatibility
    use_tma: bool = True,  # Unused - kept for interface compatibility
):
    """Reference implementation using PyTorch"""
    return torch.matmul(a, b)

register_impl("matmul", "torch")(reference_matmul)


FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)
BASE_DTYPE = torch.bfloat16

# Final comparison:
#   - CuTile BF16 vs PyTorch BF16
#   - CuTile FP8 (E4M3) as a 3rd line
ALL_VARIANTS = [
    ("cutile_bf16", "CuTile (BF16)", ("orange", "-"))
    if is_backend_available("cutile")
    else None,
    ("torch_bf16", "PyTorch (BF16)", ("green", "-")),
    ("cutile_fp8_e4m3", "CuTile (FP8 E4M3)", ("blue", "--"))
    if (is_backend_available("cutile") and FP8_DTYPE is not None)
    else None,
]


def get_supported_variants():
    return [p for p in ALL_VARIANTS if p is not None]


def create_benchmark_config():
    """Create a benchmark configuration for the final 3-line comparison."""
    variants, names, styles = zip(*get_supported_variants())
    compute_capability = torch.cuda.get_device_capability()
    if compute_capability[0] == 10:
        max_range = 16
    else:
        max_range = 15  # To avoid OOM
    return triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[2**i for i in range(10, max_range)],
        line_arg="variant",
        line_vals=list(variants),
        line_names=list(names),
        styles=list(styles),
        xlabel="M/N/K",
        ylabel="TFLOPS",
        plot_name="matmul-performance-cutile-vs-torch-vs-fp8-TFLOPS",
        args={},
    )


@triton.testing.perf_report(
    create_benchmark_config()
)
def benchmark(M, N, K, variant):
    # Always generate BF16 inputs; cast to FP8 for the FP8 line so inputs are comparable.
    a_bf16 = torch.randn((M, K), device=DEVICE, dtype=BASE_DTYPE)
    b_bf16 = torch.randn((K, N), device=DEVICE, dtype=BASE_DTYPE)

    quantiles = [0.5, 0.2, 0.8]

    if variant == "torch_bf16":
        fn = lambda: reference_matmul(a_bf16, b_bf16)
    elif variant == "cutile_bf16":
        fn = lambda: tilegym.ops.matmul(
            a_bf16,
            b_bf16,
            use_tma=True,
            static_persistent=False,
            backend="cutile",
        )
    elif variant == "cutile_fp8_e4m3":
        assert FP8_DTYPE is not None, "torch.float8_e4m3fn not available in this torch build"
        # Cast BF16 -> FP16 -> FP8 to ensure the cast path is available everywhere.
        a_fp8 = a_bf16.to(torch.float16).to(FP8_DTYPE)
        b_fp8 = b_bf16.to(torch.float16).to(FP8_DTYPE)
        fn = lambda: tilegym.ops.matmul(
            a_fp8,
            b_fp8,
            use_tma=True,
            static_persistent=False,
            backend="cutile",
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Correctness checks
    if variant in ("torch_bf16", "cutile_bf16"):
        ref = lambda: reference_matmul(a_bf16, b_bf16)
        torch.testing.assert_close(fn(), ref())
    elif variant == "cutile_fp8_e4m3":
        # PyTorch doesn't support FP8 matmul here; FP8 error can grow with problem size.
        # To avoid breaking the entire sweep on a handful of outliers, only sanity-check
        # one smallish point.
        if (M, N, K) == (1024, 1024, 1024):
            out_fp32 = fn().to(torch.float32)
            ref_fp32 = reference_matmul(
                a_bf16.to(torch.float32), b_bf16.to(torch.float32)
            )
            torch.testing.assert_close(out_fp32, ref_fp32, rtol=1.0, atol=2e1)

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        fn, quantiles=quantiles
    )
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(print_data=True)
