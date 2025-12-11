#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Benchmark for silu_and_mul_mxfp8 kernel.
Compares CuTile fused kernel vs PyTorch separate ops.

DeepSeek MoE intermediate sizes:
- DSV2-lite: 1408
- DSV2-large: 1536
- DSV3: 2048

This kernel is used for MoE experts only (not dense FFN).
Supports non-power-of-2 sizes via internal padding.
"""

import torch
import triton
import triton.testing

from tilegym.ops.cutile.silu_and_mul_mxfp8 import (
    silu_and_mul_mxfp8,
    MXFP8_GROUP_SIZE,
    MAX_FP8_E4M3,
)

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def reference_silu_and_mul(input: torch.Tensor) -> torch.Tensor:
    """Reference silu_and_mul using PyTorch"""
    hidden_size = input.shape[-1] // 2
    x1 = input[..., :hidden_size]
    x2 = input[..., hidden_size:]
    return torch.nn.functional.silu(x1) * x2


def reference_mxfp8_quantize(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference MXFP8 quantization in PyTorch"""
    hidden_size = tensor.shape[-1]
    batch_shape = tensor.shape[:-1]
    num_groups = hidden_size // MXFP8_GROUP_SIZE
    
    reshaped = tensor.view(*batch_shape, num_groups, MXFP8_GROUP_SIZE)
    block_max = reshaped.abs().max(dim=-1, keepdim=True).values
    scales = (block_max / MAX_FP8_E4M3).clamp(min=1e-12)
    scaled = (reshaped / scales).clamp(-MAX_FP8_E4M3, MAX_FP8_E4M3)
    quantized = scaled.to(torch.float8_e4m3fn)
    
    return quantized.view(*batch_shape, hidden_size), scales.squeeze(-1)


def pytorch_silu_and_mul_mxfp8(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch: silu_and_mul + MXFP8 quantization (separate ops)"""
    result = reference_silu_and_mul(input.float())
    return reference_mxfp8_quantize(result)


def cutile_silu_and_mul_mxfp8_wrapper(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """CuTile: fused silu_and_mul + MXFP8 quantization"""
    return silu_and_mul_mxfp8(input)


# Available implementations
ALL_IMPLEMENTATIONS = [
    ("cutile", "CuTile (Fused)", ("blue", "-"), cutile_silu_and_mul_mxfp8_wrapper),
    ("pytorch", "PyTorch (Separate)", ("green", "-"), pytorch_silu_and_mul_mxfp8),
]


def get_supported_implementations():
    return ALL_IMPLEMENTATIONS


def create_benchmark_config(hidden_size):
    """Create benchmark config for given hidden_size"""
    available_impls = get_supported_implementations()
    impl_ids, names, styles, _ = zip(*available_impls)
    
    return triton.testing.Benchmark(
        x_names=["M"],  # Batch size
        x_vals=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        line_arg="impl",
        line_vals=list(impl_ids),
        line_names=list(names),
        styles=list(styles),
        ylabel="GB/s",
        plot_name=f"silu_and_mul_mxfp8-hidden{hidden_size}",
        args={"hidden_size": hidden_size},
    )


IMPL_FUNCS = {p[0]: p[3] for p in get_supported_implementations()}


@triton.testing.perf_report(
    [
        create_benchmark_config(hidden_size)
        for hidden_size in [1408, 1536, 2048]  # DSV2-lite, DSV2-large, DSV3 MoE sizes
    ]
)
def bench_silu_and_mul_mxfp8(M, impl, hidden_size, device=DEVICE):
    # Create input tensor
    input_shape = (M, 2 * hidden_size)
    x = torch.randn(input_shape, dtype=torch.bfloat16, device=device)
    
    fn = IMPL_FUNCS[impl]
    result_fn = lambda: fn(x)
    
    # Verify correctness on first run
    if M == 1:
        ref_result = pytorch_silu_and_mul_mxfp8(x)
        cutile_result = silu_and_mul_mxfp8(x)
        # Just check shapes match
        assert ref_result[0].shape == cutile_result[0].shape
    
    # Benchmark
    ms = triton.testing.do_bench_cudagraph(result_fn)
    
    # Calculate bandwidth
    # Input: M * 2 * hidden * 2 bytes (bf16)
    # Output: M * hidden * 1 byte (fp8) + M * (hidden/32) * 4 bytes (scales)
    input_bytes = M * 2 * hidden_size * 2
    output_bytes = M * hidden_size * 1 + M * (hidden_size // 32) * 4
    total_bytes = input_bytes + output_bytes
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)
    
    return gb_per_s


if __name__ == "__main__":
    bench_silu_and_mul_mxfp8.run(print_data=True, save_path=".")
