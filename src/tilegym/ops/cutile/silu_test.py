import cuda.tile as ct
import torch

from cuda.tile._numeric_semantics import RoundingMode as RMd


ConstInt = ct.Constant[int]

@ct.kernel
def silu_mul_kernel(
    input,
    output,
    TILE_SIZE: ConstInt,
    hidden_size: ConstInt,
):
    row_idx = ct.bid(0) # row index
    offsets = ct.arange(TILE_SIZE, dtype=torch.int32)
    a_col_idx = offsets
    b_col_idx = offsets + hidden_size

    a_tile = ct.gather(input, (row_idx, a_col_idx), check_bounds=True)
    b_tile = ct.gather(input, (row_idx, b_col_idx), check_bounds=True)
    a_tile = ct.astype(a_tile, torch.float32)
    b_tile = ct.astype(b_tile, torch.float32)

    denom = ct.add(1, ct.exp(-a_tile), flush_to_zero=True)
    sigmoid_a = ct.truediv(1.0, denom, flush_to_zero=True, rounding_mode=RMd.APPROX)

    # silu (a) * b
    silu_a = ct.mul(a_tile, sigmoid_a, flush_to_zero=True)
    result = ct.mul(silu_a, b_tile, flush_to_zero=True)
    result = ct.astype(result, input.dtype)

    ct.scatter(output, (row_idx, offsets), result, check_bounds=True)


@ct.kernel
def silu_mul_2_kernel(
    input,
    output,
    TILE_SIZE: ConstInt,
    hidden_size: ConstInt,
):
    row_idx = ct.bid(0)
    offsets = ct.arange(TILE_SIZE, dtype = torch.int32)

    a_col_idx = offsets
    b_col_idx = offsets + hidden_size

    a_tile = ct.gather(input, (row_idx, a_col_idx), check_bounds=True)
    b_tile = ct.gather(input, (row_idx, b_col_idx), check_bounds=True)
    a_tile = ct.astype(a_tile, torch.float32)
    b_tile = ct.astype(b_tile, torch.float32)

    denom = ct.add(1, ct.exp(-a_tile), flush_to_zero=True)
    sigmoid_a = ct.truediv(1.0, denom, flush_to_zero=True, rounding_mode=RMd.APPROX)

    silu_a = ct.mul(a_tile, sigmoid_a, flush_to_zero=True)
    result = ct.mul(silu_a, b_tile)
    result = ct.astype(result, input.dtype)

    # store result using scatter with 2D indices
    ct.scatter(output, (row_idx, offsets), result, check_bounds=True)

