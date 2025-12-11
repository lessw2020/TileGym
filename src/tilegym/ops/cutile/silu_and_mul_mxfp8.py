# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Fused SiLU and Mul operation with MXFP8 output quantization.

MXFP8 (Microscaling FP8) format:
- Quantizes in groups of 32 elements with a shared scale factor per group
- Output values are FP8 E4M3 format
- Scale factors are stored as float32 (can be converted to E8M0 format)

Supports non-power-of-2 hidden sizes by padding internally.
"""

import functools

import cuda.tile as ct
import torch

from tilegym.backend import register_impl
from cuda.tile._numeric_semantics import RoundingMode as RMd

# Type aliases for constants
ConstInt = ct.Constant[int]

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


def ensure_contiguous(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(*args, **kwargs)

    return wrapper


@ct.kernel
def silu_and_mul_mxfp8_kernel_padded(
    input,
    output_quantized,
    output_scales,
    PADDED_SIZE: ConstInt,      # Power of 2 padded size for ct.arange
    hidden_size: ConstInt,       # Actual hidden size
    num_groups: ConstInt,        # Actual number of groups (hidden_size // 32)
    padded_num_groups: ConstInt, # Padded number of groups (PADDED_SIZE // 32)
    GROUP_SIZE: ConstInt,        # Should be 32 for MXFP8
):
    """
    Fused SiLU+Mul kernel with MXFP8 quantization.
    Supports non-power-of-2 sizes by padding internally.
    
    Each CUDA block processes one row.
    """
    bid = ct.bid(0)  # row index
    offsets = ct.arange(PADDED_SIZE, dtype=torch.int32)

    row_idx = bid
    a_col_idx = offsets  # First half: [0, hidden_size), padded beyond
    b_col_idx = offsets + hidden_size  # Second half: [hidden_size, 2*hidden_size), padded beyond

    # Load input tiles - check_bounds=True pads with 0 for out-of-bounds indices
    a_tile = ct.gather(input, (row_idx, a_col_idx), check_bounds=True, padding_value=0.0)
    b_tile = ct.gather(input, (row_idx, b_col_idx), check_bounds=True, padding_value=0.0)
    a_tile = ct.astype(a_tile, torch.float32)
    b_tile = ct.astype(b_tile, torch.float32)

    # Compute SiLU(a) * b
    neg_a = ct.negative(a_tile)
    exp_neg_a = ct.exp(neg_a)
    denom = ct.add(1.0, exp_neg_a)
    sigmoid_a = ct.truediv(1.0, denom, rounding_mode=RMd.APPROX)
    silu_a = ct.mul(a_tile, sigmoid_a)
    result = ct.mul(silu_a, b_tile)  # Shape: (PADDED_SIZE,)

    # MXFP8 Quantization: group into 32-element chunks
    # Reshape from (PADDED_SIZE,) to (padded_num_groups, 32)
    result_grouped = ct.reshape(result, (padded_num_groups, GROUP_SIZE))
    
    # Compute absolute value: abs(x) = max(x, -x)
    result_neg = ct.negative(result_grouped)
    result_abs = ct.maximum(result_grouped, result_neg)
    
    # Compute max absolute value per group -> (padded_num_groups, 1)
    group_max = ct.max(result_abs, 1, keepdims=True)
    
    # Compute scale factor per group: scale = max_abs / MAX_FP8_E4M3
    scales = ct.truediv(group_max, MAX_FP8_E4M3)
    scales = ct.maximum(scales, 1e-12)  # Avoid division by zero
    
    # Scale the values: scaled = result / scale (broadcasts scales across dim 1)
    scaled_result = ct.truediv(result_grouped, scales, rounding_mode=RMd.RN)
    
    # Clamp to FP8 E4M3 range [-448, 448]
    scaled_result = ct.minimum(scaled_result, MAX_FP8_E4M3)
    scaled_result = ct.maximum(scaled_result, -MAX_FP8_E4M3)
    
    # Convert to FP8 E4M3 - shape still (padded_num_groups, 32)
    quantized = ct.astype(scaled_result, torch.float8_e4m3fn)
    
    # Reshape back to (PADDED_SIZE,) for storage
    quantized_flat = ct.reshape(quantized, (PADDED_SIZE,))
    
    # Store only the first hidden_size elements (use check_bounds to mask)
    out_col_idx = offsets
    ct.scatter(output_quantized, (row_idx, out_col_idx), quantized_flat, check_bounds=True)
    
    # Store only the first num_groups scales
    scales_flat = ct.reshape(scales, (padded_num_groups,))
    scales_float = ct.astype(scales_flat, torch.float32)
    scale_offsets = ct.arange(padded_num_groups, dtype=torch.int32)
    ct.scatter(output_scales, (row_idx, scale_offsets), scales_float, check_bounds=True)


@register_impl("silu_and_mul_mxfp8", backend="cutile")
@ensure_contiguous
def silu_and_mul_mxfp8(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused SiLU and Mul operation with MXFP8 output quantization.

    Computes: MXFP8_quantize(silu(input[..., :hidden_size]) * input[..., hidden_size:])
    
    MXFP8 quantizes in groups of 32 elements, with one scale factor per group.
    Supports non-power-of-2 hidden sizes (e.g., 1408 for DSV2-lite).

    Args:
        input (torch.Tensor): Input tensor of shape (..., 2 * hidden_size)
                             hidden_size must be divisible by 32.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: 
            - Quantized output tensor of shape (..., hidden_size) in FP8 E4M3 format
            - Scale tensor of shape (..., hidden_size // 32) in float32
              (one scale per 32-element group)
    """
    # Save original shape and flatten input for simpler processing
    original_shape = input.shape
    hidden_size = original_shape[-1] // 2
    
    # Ensure hidden_size is divisible by MXFP8_GROUP_SIZE (32)
    if hidden_size % MXFP8_GROUP_SIZE != 0:
        raise ValueError(
            f"hidden_size ({hidden_size}) must be divisible by MXFP8_GROUP_SIZE ({MXFP8_GROUP_SIZE})"
        )
    
    num_groups = hidden_size // MXFP8_GROUP_SIZE
    
    # Pad to next power of 2 for ct.arange compatibility
    padded_size = next_power_of_2(hidden_size)
    padded_num_groups = padded_size // MXFP8_GROUP_SIZE

    # Flatten input to 2D: (batch_size, 2 * hidden_size)
    input_flat = input.view(-1, original_shape[-1])
    batch_size = input_flat.shape[0]

    # Prepare output tensors (actual size, not padded)
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

    # Launch: one CUDA block per row
    grid = (batch_size,)
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

    # Reshape outputs to match input batch dimensions
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
    
    # Convert FP8 to float32
    dequant = quantized.to(torch.float32)
    
    # Reshape for broadcasting: (..., num_groups, 32)
    num_groups = hidden_size // MXFP8_GROUP_SIZE
    dequant = dequant.view(*batch_shape, num_groups, MXFP8_GROUP_SIZE)
    
    # Expand scales: (..., num_groups) -> (..., num_groups, 1)
    scales_expanded = scales.unsqueeze(-1)
    
    # Multiply by scales
    result = dequant * scales_expanded
    
    # Reshape back: (..., hidden_size)
    return result.view(*batch_shape, hidden_size)
