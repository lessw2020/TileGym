#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Benchmark comparing V1 (padded) vs V2 (chunked) silu_and_mul_mxfp8 kernels.

V1: Single-pass with power-of-2 padding (fast for power-of-2, overhead for non-power-of-2)
V2: Loop over 32-element groups (no padding overhead, good for non-power-of-2)

DeepSeek MoE intermediate sizes:
- DSV2-lite: 1408 (non-power-of-2)
- DSV2-large: 1536 (non-power-of-2)
- DSV3: 2048 (power-of-2)
"""

import torch
import triton
import triton.testing

from tilegym.ops.cutile.silu_and_mul_mxfp8 import (
    silu_and_mul_mxfp8,
    MXFP8_GROUP_SIZE,
    MAX_FP8_E4M3,
)
from tilegym.ops.cutile.silu_and_mul_mxfp8_v2 import (
    silu_and_mul_mxfp8_v2,
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


@torch.compile
def pytorch_silu_and_mul_mxfp8_compiled(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch torch.compile: silu_and_mul + MXFP8 quantization"""
    result = reference_silu_and_mul(input.float())
    return reference_mxfp8_quantize(result)


def cutile_v1_wrapper(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """CuTile V1: padded kernel"""
    return silu_and_mul_mxfp8(input)


def cutile_v2_wrapper(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """CuTile V2: auto-select (chunked for non-power-of-2)"""
    return silu_and_mul_mxfp8_v2(input)


# Available implementations
ALL_IMPLEMENTATIONS = [
    ("cutile_v1", "CuTile V1 (Padded)", ("blue", "-"), cutile_v1_wrapper),
    ("cutile_v2", "CuTile V2 (Chunked)", ("orange", "-"), cutile_v2_wrapper),
    ("torch_compile", "PyTorch (torch.compile)", ("red", "--"), pytorch_silu_and_mul_mxfp8_compiled),
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
        plot_name=f"silu_and_mul_mxfp8_v1_vs_v2-hidden{hidden_size}",
        args={"hidden_size": hidden_size},
    )


IMPL_FUNCS = {p[0]: p[3] for p in get_supported_implementations()}


@triton.testing.perf_report(
    [
        create_benchmark_config(hidden_size)
        for hidden_size in [1408, 1536, 2048]  # DSV2-lite, DSV2-large, DSV3 MoE sizes
    ]
)
def bench_silu_and_mul_mxfp8_v2(M, impl, hidden_size, device=DEVICE):
    # Create input tensor
    input_shape = (M, 2 * hidden_size)
    x = torch.randn(input_shape, dtype=torch.bfloat16, device=device)
    
    fn = IMPL_FUNCS[impl]
    result_fn = lambda: fn(x)
    
    # Verify correctness on first run
    if M == 1:
        v1_result = silu_and_mul_mxfp8(x)
        v2_result = silu_and_mul_mxfp8_v2(x)
        # Check shapes match
        assert v1_result[0].shape == v2_result[0].shape
        assert v1_result[1].shape == v2_result[1].shape
    
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
    bench_silu_and_mul_mxfp8_v2.run(print_data=True, save_path=".")
    
    # Print decode speedup summary
    print("\n" + "=" * 90)
    print("DECODE LATENCY SUMMARY (batch sizes 1, 2, 4, 8)")
    print("=" * 90)
    
    decode_batch_sizes = [1, 2, 4, 8]
    hidden_sizes = [1408, 1536, 2048]
    
    for hidden_size in hidden_sizes:
        print(f"\n--- Hidden Size: {hidden_size} ---")
        print(f"{'Batch':>6} {'V1 (ms)':>10} {'V2 (ms)':>10} {'compile (ms)':>14} {'V1 vs V2':>12} {'Best':>20}")
        print("-" * 90)
        
        for M in decode_batch_sizes:
            x = torch.randn((M, 2 * hidden_size), dtype=torch.bfloat16, device=DEVICE)
            
            # Warmup
            _ = silu_and_mul_mxfp8(x)
            _ = silu_and_mul_mxfp8_v2(x)
            _ = pytorch_silu_and_mul_mxfp8_compiled(x)
            torch.cuda.synchronize()
            
            # Benchmark each
            ms_v1 = triton.testing.do_bench_cudagraph(lambda: silu_and_mul_mxfp8(x))
            ms_v2 = triton.testing.do_bench_cudagraph(lambda: silu_and_mul_mxfp8_v2(x))
            ms_compile = triton.testing.do_bench_cudagraph(lambda: pytorch_silu_and_mul_mxfp8_compiled(x))
            
            # V1 vs V2
            if ms_v2 < ms_v1:
                v1_vs_v2 = f"V2 {ms_v1/ms_v2:.2f}x"
            else:
                v1_vs_v2 = f"V1 {ms_v2/ms_v1:.2f}x"
            
            # Find overall best
            times = {"V1": ms_v1, "V2": ms_v2, "compile": ms_compile}
            best_name = min(times, key=times.get)
            best_time = times[best_name]
            second_best = min(t for n, t in times.items() if n != best_name)
            speedup = second_best / best_time
            best_str = f"{best_name} ({speedup:.2f}x)"
            
            print(f"{M:>6} {ms_v1:>10.4f} {ms_v2:>10.4f} {ms_compile:>14.4f} {v1_vs_v2:>12} {best_str:>20}")
        
        print("-" * 90)
    
    print("\n" + "=" * 90)
