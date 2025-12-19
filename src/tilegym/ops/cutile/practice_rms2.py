import torch
import cuda.tile as ct

from .cutile_constants import ConstInt, ConstFloat, ConstBool

@ct.kernel
def rms2_kernel(
    x,
    w,
    out,
    N: ConstInt,
    eps: ConstFloat,
    TILE_SIZE: ConstInt,
):
    row_idx = ct.bid(0)
    offsets = ct.arange(TILE_SIZE, dtype = torch.int32)
    _rms = ct.full((TILE_SIZE,), 0,0, dtype = ct.float32)
    num_tiles = ct.cdiv(N, TILE_SIZE)

    for j in range(0, num_tiles):
        offs_inner = (j*TILE_SIZE) + offsets
        xj = ct.gather(x, (row_idx, offs_inner), latency=1)
        xj = ct.astype(xj, ct.float32)
        _rms += xj * xj
    
    rms_scalar = ct.rsqrt(ct.sum(_rms, axis=0, keepdims=False) / N + eps)

    for j in range(0, num_tiles):
        offs_inner = (j*TILE_SIZE) + offsets
        wj = ct.gather(w, offs_inner, latency=1)
        wj = ct.astype(wj, ct.float32)

        xj = ct.gather(x, (row_idx, offs_inner), latency=1)
        xj = ct.astype(xj, ct.float32)
        yj = xj * wj * rms_scalar

        ct.scatter(out, (row_idx, offs_inner), yj, latency=1)