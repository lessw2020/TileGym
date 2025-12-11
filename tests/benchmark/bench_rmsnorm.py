# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
import triton

import tilegym
from tilegym.backend import is_backend_available

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def reference_rms_norm(x: torch.Tensor, w_shape: tuple, weight: torch.Tensor, eps: float):
    """Reference implementation of RMSNorm using PyTorch (eager)"""
    dims = tuple(i for i in range(-1, -len(w_shape) - 1, -1))
    variance = x.to(torch.float32).pow(2).mean(dims, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)

    if weight.dtype in [torch.float16, torch.bfloat16]:
        x_norm = x_norm.to(weight.dtype)

    return weight * x_norm


# Compiled version of RMSNorm
@torch.compile
def compiled_rms_norm(x: torch.Tensor, w_shape: tuple, weight: torch.Tensor, eps: float):
    """Compiled RMSNorm using torch.compile"""
    dims = tuple(i for i in range(-1, -len(w_shape) - 1, -1))
    variance = x.to(torch.float32).pow(2).mean(dims, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)

    if weight.dtype in [torch.float16, torch.bfloat16]:
        x_norm = x_norm.to(weight.dtype)

    return weight * x_norm


def cutile_persistent_rms_norm(x: torch.Tensor, w_shape: tuple, weight: torch.Tensor, eps: float):
    """CuTile RMSNorm with static persistent scheduling"""
    return tilegym.ops.rms_norm(x, w_shape, weight, eps, static_persistent=True, backend="cutile")


def cutile_non_persistent_rms_norm(x: torch.Tensor, w_shape: tuple, weight: torch.Tensor, eps: float):
    """CuTile RMSNorm without static persistent scheduling"""
    return tilegym.ops.rms_norm(x, w_shape, weight, eps, static_persistent=False, backend="cutile")


# Available implementations with their display names and plot styles
ALL_IMPLEMENTATIONS = [
    ("cutile_persistent", "CuTile (Persistent)", ("blue", "-"), cutile_persistent_rms_norm)
    if is_backend_available("cutile")
    else None,
    ("cutile_non_persistent", "CuTile (Non-Persistent)", ("orange", "-"), cutile_non_persistent_rms_norm)
    if is_backend_available("cutile")
    else None,
    ("torch_compile", "PyTorch (torch.compile)", ("red", "-"), compiled_rms_norm),
    ("torch", "PyTorch (Eager)", ("green", "-"), reference_rms_norm),
]


def get_supported_implementations():
    """Filter implementations based on availability"""
    return [p for p in ALL_IMPLEMENTATIONS if p is not None]


def create_benchmark_config(dtype, M):
    """Create a benchmark configuration for given parameters"""
    available_impls = get_supported_implementations()
    if not available_impls:
        return None

    impl_ids, names, styles, _ = zip(*available_impls)
    dtype_name = str(dtype).split('.')[-1]

    return triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(10, 15)],  # Hidden size from 1024 to 16384
        line_arg="impl",
        line_vals=list(impl_ids),
        line_names=list(names),
        styles=list(styles),
        ylabel="GB/s",
        plot_name=f"rms_norm-M{M}-{dtype_name}",
        args={"dtype": dtype, "M": M},
    )


# Build lookup dict for implementations
IMPL_FUNCS = {p[0]: p[3] for p in get_supported_implementations()}


@triton.testing.perf_report(
    [
        create_benchmark_config(dtype, M)
        for dtype in [torch.float16, torch.bfloat16]
        for M in [4096]
    ]
)
def bench_rmsnorm(N, impl, dtype, M, device=DEVICE):
    eps = 1e-5

    # Create input tensors
    x_shape = (M, N)
    w_shape = (N,)

    x = (
        torch.rand(x_shape, dtype=dtype, device=device, requires_grad=False)
        .mul_(0.5)
        .add_(-2.3)
    )
    weight = torch.randn(w_shape, dtype=dtype, device=device, requires_grad=False)

    fn = IMPL_FUNCS[impl]
    result_fn = lambda: fn(x, w_shape, weight, eps)
    ref = lambda: reference_rms_norm(x, w_shape, weight, eps)
    torch.testing.assert_close(result_fn(), ref(), atol=5e-2, rtol=0.0)

    # Benchmark the function
    ms = triton.testing.do_bench_cudagraph(result_fn)

    # Calculate memory bandwidth (GB/s)
    # RMSNorm operation: read input, read weight, write output
    bytes_per_element = x.element_size()

    input_bytes = x.numel() * bytes_per_element  # Read input
    weight_bytes = weight.numel() * bytes_per_element  # Read weight
    output_bytes = x.numel() * bytes_per_element  # Write output

    total_bytes = input_bytes + weight_bytes + output_bytes

    # Convert to GB/s
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)

    return gb_per_s


if __name__ == "__main__":
    bench_rmsnorm.run(print_data=True, save_path=".")
