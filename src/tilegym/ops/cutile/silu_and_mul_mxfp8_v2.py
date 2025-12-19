# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Fused SiLU and Mul operation with MXFP8 output quantization - V2.

MXFP8 (Microscaling FP8) format:
- Quantizes in groups of 32 elements with a shared scale factor per group
- Output values are FP8 E4M3 format
- Scale factors are stored as float32 (can be converted to E8M0 format)

Two kernel variants:
1. Padded: Uses ct.arange with power-of-2 padding (fast for power-of-2 sizes)
2. Chunked: Loops over 32-element groups (efficient for non-power-of-2 sizes)

This V2 version auto-selects the best kernel based on hidden_size.
"""

import functools

import cuda.tile as ct
import torch

from cuda.tile._numeric_semantics import RoundingMode as RMd

# Type aliases for constants
from .cutile_constants import ConstInt

# MXFP8 constants
MXFP8_GROUP_SIZE = 32  # Number of elements per group
MAX_FP8_E4M3 = 448.0   # Max representable value in E4M3 format


def next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def is_power_of_2(n):
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def ensure_contiguous(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(*args, **kwargs)

    return wrapper


# =============================================================================
# Kernel 1: Padded version (efficient for power-of-2 hidden sizes)
# =============================================================================

@ct.kernel
def silu_and_mul_mxfp8_kernel_padded(
    input,
    output_quantized,
    output_scales,
    PADDED_SIZE: ConstInt,
    hidden_size: ConstInt,
    num_groups: ConstInt,
    padded_num_groups: ConstInt,
    GROUP_SIZE: ConstInt,
):
    """
    Fused SiLU+Mul kernel with MXFP8 quantization (padded version).
    Best for power-of-2 hidden sizes where PADDED_SIZE == hidden_size.
    """
    bid = ct.bid(0)
    offsets = ct.arange(PADDED_SIZE, dtype=torch.int32)

    row_idx = bid
    a_col_idx = offsets
    b_col_idx = offsets + hidden_size

    a_tile = ct.gather(input, (row_idx, a_col_idx), check_bounds=True, padding_value=0.0)
    b_tile = ct.gather(input, (row_idx, b_col_idx), check_bounds=True, padding_value=0.0)
    a_tile = ct.astype(a_tile, torch.float32)
    b_tile = ct.astype(b_tile, torch.float32)

    neg_a = ct.negative(a_tile)
    exp_neg_a = ct.exp(neg_a)
    denom = ct.add(1.0, exp_neg_a)
    sigmoid_a = ct.truediv(1.0, denom, rounding_mode=RMd.APPROX)
    silu_a = ct.mul(a_tile, sigmoid_a)
    result = ct.mul(silu_a, b_tile)

    result_grouped = ct.reshape(result, (padded_num_groups, GROUP_SIZE))
    result_neg = ct.negative(result_grouped)
    result_abs = ct.maximum(result_grouped, result_neg)
    group_max = ct.max(result_abs, 1, keepdims=True)
    scales = ct.truediv(group_max, MAX_FP8_E4M3)
    scales = ct.maximum(scales, 1e-12)
    scaled_result = ct.truediv(result_grouped, scales, rounding_mode=RMd.RN)
    scaled_result = ct.minimum(scaled_result, MAX_FP8_E4M3)
    scaled_result = ct.maximum(scaled_result, -MAX_FP8_E4M3)
    quantized = ct.astype(scaled_result, torch.float8_e4m3fn)

    quantized_flat = ct.reshape(quantized, (PADDED_SIZE,))
    out_col_idx = offsets
    ct.scatter(output_quantized, (row_idx, out_col_idx), quantized_flat, check_bounds=True)

    scales_flat = ct.reshape(scales, (padded_num_groups,))
    scales_float = ct.astype(scales_flat, torch.float32)
    scale_offsets = ct.arange(padded_num_groups, dtype=torch.int32)
    ct.scatter(output_scales, (row_idx, scale_offsets), scales_float, check_bounds=True)


# =============================================================================
# Kernel 2: Chunked version (efficient for non-power-of-2 hidden sizes)
# =============================================================================

@ct.kernel
def silu_and_mul_mxfp8_kernel_chunked(
    input,
    output_quantized,
    output_scales,
    hidden_size: ConstInt,
    num_groups: ConstInt,
    GROUP_SIZE: ConstInt,  # Must be 32
):
    """
    Fused SiLU+Mul kernel with MXFP8 quantization (chunked version).
    Processes one 32-element group at a time - no padding overhead.
    """
    bid = ct.bid(0)  # row index
    row_idx = bid
    
    # Process each 32-element group
    offsets = ct.arange(GROUP_SIZE, dtype=torch.int32)  # [0, 1, ..., 31] - always power of 2!
    
    for group_idx in range(num_groups):
        # Column indices for this group
        col_start = group_idx * GROUP_SIZE
        a_col_idx = offsets + col_start
        b_col_idx = offsets + col_start + hidden_size
        
        # Load 32 elements from a and b
        a_tile = ct.gather(input, (row_idx, a_col_idx), check_bounds=True)
        b_tile = ct.gather(input, (row_idx, b_col_idx), check_bounds=True)
        a_tile = ct.astype(a_tile, torch.float32)
        b_tile = ct.astype(b_tile, torch.float32)
        
        # Compute SiLU(a) * b
        neg_a = ct.negative(a_tile)
        exp_neg_a = ct.exp(neg_a)
        denom = ct.add(1.0, exp_neg_a)
        sigmoid_a = ct.truediv(1.0, denom, rounding_mode=RMd.APPROX)
        silu_a = ct.mul(a_tile, sigmoid_a)
        result = ct.mul(silu_a, b_tile)  # Shape: (32,)
        
        # MXFP8 quantization for this group
        result_neg = ct.negative(result)
        result_abs = ct.maximum(result, result_neg)
        group_max = ct.max(result_abs, 0, keepdims=True)  # scalar
        
        scale = ct.truediv(group_max, MAX_FP8_E4M3)
        scale = ct.maximum(scale, 1e-12)
        
        scaled_result = ct.truediv(result, scale, rounding_mode=RMd.RN)
        scaled_result = ct.minimum(scaled_result, MAX_FP8_E4M3)
        scaled_result = ct.maximum(scaled_result, -MAX_FP8_E4M3)
        quantized = ct.astype(scaled_result, torch.float8_e4m3fn)
        
        # Store 32 quantized values
        out_col_idx = offsets + col_start
        ct.scatter(output_quantized, (row_idx, out_col_idx), quantized, check_bounds=True)
        
        # Store 1 scale value
        scale_float = ct.astype(scale, torch.float32)
        # Create scalar index for scale
        scale_idx = ct.arange(1, dtype=torch.int32) + group_idx
        ct.scatter(output_scales, (row_idx, scale_idx), scale_float, check_bounds=True)


@ensure_contiguous
def silu_and_mul_mxfp8_v2(
    input: torch.Tensor,
    force_chunked: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused SiLU and Mul operation with MXFP8 output quantization (V2).

    Computes: MXFP8_quantize(silu(input[..., :hidden_size]) * input[..., hidden_size:])
    
    MXFP8 quantizes in groups of 32 elements, with one scale factor per group.
    
    Automatically selects the best kernel:
    - Power-of-2 hidden sizes: uses padded kernel (single-pass, no overhead)
    - Non-power-of-2 sizes: uses chunked kernel (loop over 32-element groups)

    Args:
        input (torch.Tensor): Input tensor of shape (..., 2 * hidden_size)
                             hidden_size must be divisible by 32.
        force_chunked (bool): Force use of chunked kernel even for power-of-2 sizes.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: 
            - Quantized output tensor of shape (..., hidden_size) in FP8 E4M3 format
            - Scale tensor of shape (..., hidden_size // 32) in float32
    """
    original_shape = input.shape
    hidden_size = original_shape[-1] // 2
    
    if hidden_size % MXFP8_GROUP_SIZE != 0:
        raise ValueError(
            f"hidden_size ({hidden_size}) must be divisible by MXFP8_GROUP_SIZE ({MXFP8_GROUP_SIZE})"
        )
    
    num_groups = hidden_size // MXFP8_GROUP_SIZE

    input_flat = input.view(-1, original_shape[-1])
    batch_size = input_flat.shape[0]

    output_quantized = torch.empty(
        (batch_size, hidden_size),
        dtype=torch.float8_e4m3fn,
        device=input.device,
    )
    
    output_scales = torch.empty(
        (batch_size, num_groups),
        dtype=torch.float32,
        device=input.device,
    )

    grid = (batch_size,)
    
    # Choose kernel based on hidden_size
    use_chunked = force_chunked or not is_power_of_2(hidden_size)
    
    if use_chunked:
        # Chunked kernel: loop over 32-element groups (no padding overhead)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            silu_and_mul_mxfp8_kernel_chunked,
            (
                input_flat,
                output_quantized,
                output_scales,
                hidden_size,
                num_groups,
                MXFP8_GROUP_SIZE,
            ),
        )
    else:
        # Padded kernel: single pass (efficient for power-of-2)
        padded_size = hidden_size  # Already power of 2
        padded_num_groups = padded_size // MXFP8_GROUP_SIZE
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            silu_and_mul_mxfp8_kernel_padded,
            (
                input_flat,
                output_quantized,
                output_scales,
                padded_size,
                hidden_size,
                num_groups,
                padded_num_groups,
                MXFP8_GROUP_SIZE,
            ),
        )

    output_shape = list(original_shape)
    output_shape[-1] = hidden_size
    scale_shape = list(original_shape[:-1]) + [num_groups]
    
    return output_quantized.reshape(*output_shape), output_scales.reshape(*scale_shape)


def dequantize_mxfp8(
    quantized: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """
    Dequantize MXFP8 tensor back to higher precision.
    
    Args:
        quantized: FP8 E4M3 tensor of shape (..., hidden_size)
        scales: Scale tensor of shape (..., hidden_size // 32)
    
    Returns:
        Dequantized tensor in float32
    """
    hidden_size = quantized.shape[-1]
    batch_shape = quantized.shape[:-1]
    
    dequant = quantized.to(torch.float32)
    
    num_groups = hidden_size // MXFP8_GROUP_SIZE
    dequant = dequant.view(*batch_shape, num_groups, MXFP8_GROUP_SIZE)
    
    scales_expanded = scales.unsqueeze(-1)
    
    result = dequant * scales_expanded
    
    return result.view(*batch_shape, hidden_size)
