#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
import triton
import triton.testing

import tilegym
from tilegym.backend import is_backend_available

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def silu_and_mul_eager(input: torch.Tensor):
    """Reference implementation using PyTorch (eager)
    Implements: Silu(input[..., :hidden_size]) * input[..., hidden_size:]
    """
    hidden_size = input.shape[-1] // 2
    x1 = input[..., :hidden_size]  # First half for SiLU
    x2 = input[..., hidden_size:]  # Second half for multiplication
    return torch.nn.functional.silu(x1) * x2


@torch.compile
def silu_and_mul_compiled(input: torch.Tensor):
    """Compiled implementation using torch.compile
    Implements: Silu(input[..., :hidden_size]) * input[..., hidden_size:]
    """
    hidden_size = input.shape[-1] // 2
    x1 = input[..., :hidden_size]  # First half for SiLU
    x2 = input[..., hidden_size:]  # Second half for multiplication
    return torch.nn.functional.silu(x1) * x2


def cutile_silu_and_mul(input: torch.Tensor):
    """CuTile implementation"""
    return tilegym.ops.silu_and_mul(input, backend="cutile")


# Base implementations
BASE_IMPLEMENTATIONS = [
    ("cutile", "CuTile", "blue", cutile_silu_and_mul)
    if is_backend_available("cutile")
    else None,
    ("torch_compile", "torch.compile", "red", silu_and_mul_compiled),
    ("torch_eager", "Eager", "green", silu_and_mul_eager),
]

# Hidden sizes to benchmark
HIDDEN_SIZES = [2048, 7168]

# Line styles for different hidden sizes
HIDDEN_STYLES = {2048: "-", 7168: "--"}


def get_supported_base_implementations():
    """Filter implementations based on availability"""
    return [p for p in BASE_IMPLEMENTATIONS if p is not None]


def build_combined_implementations():
    """Build all (impl, hidden_size) combinations for plotting"""
    base_impls = get_supported_base_implementations()
    combined = []
    for impl_id, impl_name, color, fn in base_impls:
        for hidden_size in HIDDEN_SIZES:
            combined_id = f"{impl_id}_h{hidden_size}"
            combined_name = f"{impl_name} (h={hidden_size})"
            style = (color, HIDDEN_STYLES[hidden_size])
            combined.append((combined_id, combined_name, style, fn, hidden_size))
    return combined


COMBINED_IMPLEMENTATIONS = build_combined_implementations()


def create_benchmark_config(datatype):
    """Create a benchmark configuration for given datatype"""
    if not COMBINED_IMPLEMENTATIONS:
        return None

    impl_ids = [p[0] for p in COMBINED_IMPLEMENTATIONS]
    names = [p[1] for p in COMBINED_IMPLEMENTATIONS]
    styles = [p[2] for p in COMBINED_IMPLEMENTATIONS]
    dtype_name = str(datatype).split('.')[-1]

    return triton.testing.Benchmark(
        x_names=["M"],
        x_vals=[2**i for i in range(10, 15)],
        line_arg="impl",
        line_vals=impl_ids,
        line_names=names,
        styles=styles,
        ylabel="GB/s",
        plot_name=f"silu_and_mul-{dtype_name}",
        args={"datatype": datatype},
    )


# Build lookup dicts for implementations
IMPL_FUNCS = {p[0]: p[3] for p in COMBINED_IMPLEMENTATIONS}
IMPL_HIDDEN = {p[0]: p[4] for p in COMBINED_IMPLEMENTATIONS}


@triton.testing.perf_report(
    [
        create_benchmark_config(datatype)
        for datatype in [torch.bfloat16, torch.float32]
    ]
)
def bench_silu_and_mul(
    M,
    impl,
    datatype,
    device=DEVICE,
):
    hidden_size = IMPL_HIDDEN[impl]
    
    # Create input tensor with shape (M, 2 * hidden_size)
    input_shape = (M, 2 * hidden_size)
    x = torch.randn(input_shape, dtype=datatype, device=device)

    fn = IMPL_FUNCS[impl]
    result_fn = lambda: fn(x)
    ref = lambda: silu_and_mul_eager(x)
    torch.testing.assert_close(result_fn(), ref(), atol=5e-2, rtol=5e-2)

    # Calculate memory bandwidth in GB/s
    # Total memory: read input tensor + write output tensor
    bytes_per_element = x.element_size()

    input_bytes = x.numel() * bytes_per_element  # Read full input tensor (M, 2*hidden_size)
    output_bytes = M * hidden_size * bytes_per_element  # Write output tensor (M, hidden_size)

    total_bytes = input_bytes + output_bytes

    # Use triton's cudagraph benchmark for timing
    ms = triton.testing.do_bench_cudagraph(result_fn)

    # Calculate GB/s
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)

    return gb_per_s


if __name__ == "__main__":
    bench_silu_and_mul.run(print_data=True, save_path=".")
