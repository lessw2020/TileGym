# Inference focused cuTile RMSNorm kernel
# Non static persistent mode with ptr loads

import cuda.tile as ct
import torch
import torch.nn as nn

try:
    # When used inside the tilegym package
    from .utils import next_power_of_2
except Exception:
    # When vendored / imported as a standalone file (e.g. external frameworks)
    def next_power_of_2(n: int) -> int:
        """Return the smallest power of 2 greater than or equal to n."""
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        n |= n >> 32
        n += 1
        return n

# constants
ConstInt = ct.Constant[int]
ConstFloat = ct.Constant[float]

@ct.kernel
def inf_rms_norm_kernel(
    x,
    w,
    out, 
    N: ConstInt,
    eps: ConstFloat,
    TILE_SIZE: ConstInt,
):
    row = ct.bid(0)
    _rms = ct.full((TILE_SIZE,), 0.0, dtype=ct.float32)
    num_tiles = ct.cdiv(N, TILE_SIZE)
    offsets = ct.arange(TILE_SIZE, dtype=ct.int32)

    for j in range(0, num_tiles):
        offs_inner = j * TILE_SIZE + offsets
        xj = ct.gather(x, (row, offs_inner), latency=1)
        xj = ct.astype(xj, ct.float32)
        _rms += xj * xj # sum of squares for this tile
    
    # RMSNorm: rsqrt(mean(x^2) + eps) == rsqrt(sum(x^2)/N + eps)
    rms = ct.rsqrt(ct.sum(_rms, axis=0, keepdims=False) / N + eps)

    for j in range(0, num_tiles):
        offs_inner = j * TILE_SIZE + offsets
        wj = ct.gather(w, offs_inner, latency=1)
        wj = ct.astype(wj, ct.float32)

        xj = ct.gather(x, (row, offs_inner), latency=1)
        xj = ct.astype(xj, ct.float32)

        yj = xj * rms * wj # apply normalization and linear transformation
        yj = ct.astype(yj, x.dtype)
        ct.scatter(out, (row, offs_inner), yj, latency=1)

def inf_rms_norm_wrapper(
    x,
    w,
    eps, 
    hidden_size: ConstInt,
):
    # Inference-only wrapper: no autograd, no saved ctx.
    x = x.contiguous()
    w = w.contiguous()

    # Flatten leading dims to 2D [M, N] like rms_norm.py
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape

    if hidden_size is not None and int(hidden_size) != int(N):
        raise ValueError(
            f"inf_rms_norm_wrapper: hidden_size ({hidden_size}) != x.shape[-1] ({N})"
        )

    out = torch.empty_like(x_arg)

    # Match rms_norm.py sizing heuristic
    MAX_FUSED_SIZE = 4096 // x.element_size()
    TILE_SIZE = min(MAX_FUSED_SIZE, next_power_of_2(int(N)))

    ct.launch(
        torch.cuda.current_stream(),
        (M,),
        inf_rms_norm_kernel,
        (
            x_arg,
            w,
            out,
            int(N),
            float(eps),
            int(TILE_SIZE),
        ),
    )

    return out.view(*x.shape)

class InfRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size

    def forward(self, x):
        return inf_rms_norm_wrapper(x, self.weight, self.variance_epsilon, self.hidden_size)