# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Test and benchmark for silu_and_mul_mxfp8 kernel.
"""

import torch
import triton
import triton.testing

from tilegym.ops.cutile.silu_and_mul_mxfp8 import (
    silu_and_mul_mxfp8,
    dequantize_mxfp8,
    MXFP8_GROUP_SIZE,
    MAX_FP8_E4M3,
)

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def reference_silu_and_mul(input: torch.Tensor) -> torch.Tensor:
    """Reference implementation using PyTorch"""
    hidden_size = input.shape[-1] // 2
    x1 = input[..., :hidden_size]
    x2 = input[..., hidden_size:]
    return torch.nn.functional.silu(x1) * x2


def reference_mxfp8_quantize(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference MXFP8 quantization in PyTorch.
    
    Args:
        tensor: Input tensor of shape (..., hidden_size)
    
    Returns:
        quantized: FP8 E4M3 tensor
        scales: Float32 scale tensor (one per 32-element block)
    """
    hidden_size = tensor.shape[-1]
    batch_shape = tensor.shape[:-1]
    num_groups = hidden_size // MXFP8_GROUP_SIZE
    
    # Reshape to (..., num_groups, 32)
    reshaped = tensor.view(*batch_shape, num_groups, MXFP8_GROUP_SIZE)
    
    # Compute max absolute value per block
    block_max = reshaped.abs().max(dim=-1, keepdim=True).values
    
    # Compute scales
    scales = block_max / MAX_FP8_E4M3
    scales = scales.clamp(min=1e-12)
    
    # Scale values
    scaled = reshaped / scales
    
    # Clamp and convert to FP8
    scaled = scaled.clamp(-MAX_FP8_E4M3, MAX_FP8_E4M3)
    quantized = scaled.to(torch.float8_e4m3fn)
    
    # Reshape back
    quantized = quantized.view(*batch_shape, hidden_size)
    scales = scales.squeeze(-1)  # (..., num_blocks)
    
    return quantized, scales


def reference_silu_and_mul_mxfp8(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation: silu_and_mul followed by MXFP8 quantization.
    This is what we're comparing against for fair benchmarking.
    """
    result = reference_silu_and_mul(input.float())
    return reference_mxfp8_quantize(result)


def test_silu_and_mul_mxfp8_numerics():
    """Test numerical correctness of the MXFP8 kernel."""
    print("=" * 80)
    print("MXFP8 Numerical Accuracy Test Results")
    print("=" * 80)
    
    # Test configurations (supports non-power-of-2 via padding)
    configs = [
        (4, 1408),    # DSV2-lite MoE
        (16, 1536),   # DSV2-large MoE
        (128, 2048),  # DSV3 MoE
        (256, 2048),  # DSV3 MoE larger batch
    ]
    
    # Print table header
    print(f"\n{'Batch':>8} {'Hidden':>8} {'Scale Diff':>14} {'Dequant Diff':>14} {'Quant Error':>14} {'Rel Error':>12}")
    print("-" * 80)
    
    results = []
    for batch_size, hidden_size in configs:
        # Create input
        input_shape = (batch_size, 2 * hidden_size)
        x = torch.randn(input_shape, dtype=torch.bfloat16, device=DEVICE)
        
        # Reference: compute silu_and_mul then quantize
        ref_result = reference_silu_and_mul(x.float())
        ref_quantized, ref_scales = reference_mxfp8_quantize(ref_result)
        
        # CuTile implementation
        cutile_quantized, cutile_scales = silu_and_mul_mxfp8(x)
        
        # Dequantize both for comparison
        ref_dequant = dequantize_mxfp8(ref_quantized, ref_scales)
        cutile_dequant = dequantize_mxfp8(cutile_quantized, cutile_scales)
        
        # Compute metrics
        scale_max_diff = (ref_scales - cutile_scales).abs().max().item()
        dequant_max_diff = (ref_dequant - cutile_dequant).abs().max().item()
        quant_max_error = (ref_result - cutile_dequant).abs().max().item()
        rel_error = ((ref_result - cutile_dequant).abs() / (ref_result.abs() + 1e-6)).max().item()
        
        # Print row
        print(f"{batch_size:>8} {hidden_size:>8} {scale_max_diff:>14.2e} {dequant_max_diff:>14.2e} {quant_max_error:>14.2e} {rel_error:>12.4f}")
        
        results.append({
            'batch': batch_size,
            'hidden': hidden_size,
            'scale_diff': scale_max_diff,
            'dequant_diff': dequant_max_diff,
            'quant_error': quant_max_error,
            'rel_error': rel_error,
            'passed': dequant_max_diff < 1.0
        })
    
    print("-" * 80)
    
    # Check all passed
    all_passed = all(r['passed'] for r in results)
    if all_passed:
        print("✓ All numerical tests PASSED")
    else:
        print("✗ Some tests FAILED")
    
    print("=" * 80)
    return all_passed


def benchmark_silu_and_mul_mxfp8():
    """Benchmark the MXFP8 kernel vs PyTorch reference (apples-to-apples)."""
    print("\n" + "=" * 90)
    print("MXFP8 Performance Benchmark (CuTile fused vs PyTorch separate ops)")
    print("=" * 90)
    
    configs = [
        (1, 1408),    # Decoding DSV2-lite
        (4, 1408),
        (8, 1408),
        (1, 2048),    # Decoding DSV3
        (4, 2048),
        (8, 2048),
        (1024, 2048), # Prefill
        (4096, 2048),
    ]
    
    print(f"\n{'Batch':>8} {'Hidden':>8} {'CuTile (ms)':>12} {'PyTorch (ms)':>14} {'Speedup':>10} {'GB/s':>10}")
    print("-" * 90)
    
    for batch_size, hidden_size in configs:
        input_shape = (batch_size, 2 * hidden_size)
        x = torch.randn(input_shape, dtype=torch.bfloat16, device=DEVICE)
        
        # Benchmark CuTile MXFP8 (fused silu_and_mul + quantization)
        cutile_fn = lambda: silu_and_mul_mxfp8(x)
        cutile_ms = triton.testing.do_bench_cudagraph(cutile_fn)
        
        # Benchmark PyTorch reference (silu_and_mul + MXFP8 quantization - separate ops)
        pytorch_fn = lambda: reference_silu_and_mul_mxfp8(x)
        pytorch_ms = triton.testing.do_bench_cudagraph(pytorch_fn)
        
        # Calculate bandwidth for CuTile kernel
        # Input: batch * 2 * hidden * 2 bytes (bf16)
        # Output: batch * hidden * 1 byte (fp8) + batch * (hidden/32) * 4 bytes (scales)
        input_bytes = batch_size * 2 * hidden_size * 2
        output_bytes = batch_size * hidden_size * 1 + batch_size * (hidden_size // 32) * 4
        total_bytes = input_bytes + output_bytes
        gb_per_s = total_bytes * 1e-9 / (cutile_ms * 1e-3)
        
        speedup = pytorch_ms / cutile_ms
        
        print(f"{batch_size:>8} {hidden_size:>8} {cutile_ms:>12.4f} {pytorch_ms:>14.4f} {speedup:>10.2f}x {gb_per_s:>10.1f}")
    
    print("-" * 90)
    print("Note: Both implementations compute silu_and_mul + MXFP8 quantization")
    print("      CuTile = fused kernel, PyTorch = separate eager ops")
    print("=" * 90)


if __name__ == "__main__":
    test_silu_and_mul_mxfp8_numerics()
    benchmark_silu_and_mul_mxfp8()
