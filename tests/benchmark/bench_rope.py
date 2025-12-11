# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
import triton

import tilegym
from tilegym.backend import is_backend_available

DEVICE = triton.runtime.driver.active.get_active_torch_device()

torch.manual_seed(0)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
):
    """PyTorch eager implementation of RoPE (Rotary Position Embedding)."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


@torch.compile
def apply_rope_compiled(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
):
    """PyTorch compiled implementation of RoPE (Rotary Position Embedding)."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


def create_rotary_embeddings(seq_len, head_dim, dtype, device, base=10000.0):
    """Create cos and sin tensors for rotary embeddings."""
    freqs = 1.0 / (
        base
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
            / head_dim
        )
    )

    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)

    cos_half = torch.cos(freqs).to(dtype)
    sin_half = torch.sin(freqs).to(dtype)

    cos = torch.cat([cos_half, cos_half], dim=-1)
    sin = torch.cat([sin_half, sin_half], dim=-1)

    return cos, sin


# Wrapper functions for the benchmark
def cutile_rope(q, k, cos, sin, pos_ids):
    """CuTile RoPE implementation"""
    return tilegym.ops.apply_rope_base(q.clone(), k.clone(), cos, sin, pos_ids, backend="cutile")


def torch_compile_rope(q, k, cos, sin, pos_ids):
    """PyTorch compiled RoPE implementation"""
    return apply_rope_compiled(q, k, cos, sin)


def torch_eager_rope(q, k, cos, sin, pos_ids):
    """PyTorch eager RoPE implementation"""
    return apply_rope_eager(q, k, cos, sin)


# Available implementations with their display names and plot styles
ALL_IMPLEMENTATIONS = [
    ("cutile", "CuTile", ("blue", "-"), cutile_rope)
    if is_backend_available("cutile")
    else None,
    ("torch_compile", "PyTorch (torch.compile)", ("red", "-"), torch_compile_rope),
    ("torch_eager", "PyTorch (Eager)", ("green", "-"), torch_eager_rope),
]


def get_supported_implementations():
    """Filter implementations based on availability"""
    return [p for p in ALL_IMPLEMENTATIONS if p is not None]


def create_benchmark_config(datatype, BSZ, NUM_HEADS, HEAD_DIM):
    """Create a benchmark configuration for given datatype"""
    available_impls = get_supported_implementations()
    if not available_impls:
        return None

    impl_ids, names, styles, _ = zip(*available_impls)
    dtype_name = str(datatype).split('.')[-1]

    return triton.testing.Benchmark(
        x_names=["SEQ_LEN"],
        x_vals=[2**i for i in range(12, 16)],  # 4096 to 32768
        line_arg="impl",
        line_vals=list(impl_ids),
        line_names=list(names),
        styles=list(styles),
        ylabel="GB/s",
        plot_name=f"rope-bsz{BSZ}-heads{NUM_HEADS}-dim{HEAD_DIM}-{dtype_name}",
        args={
            "BSZ": BSZ,
            "NUM_HEADS": NUM_HEADS,
            "HEAD_DIM": HEAD_DIM,
            "datatype": datatype,
        },
    )


# Build lookup dict for implementations
IMPL_FUNCS = {p[0]: p[3] for p in get_supported_implementations()}


@triton.testing.perf_report(
    [
        create_benchmark_config(datatype, BSZ, NUM_HEADS, HEAD_DIM)
        for datatype in [torch.bfloat16]
        for BSZ in [1]
        for NUM_HEADS in [16]
        for HEAD_DIM in [64]
    ]
)
def bench_rope(
    BSZ,
    NUM_HEADS,
    SEQ_LEN,
    HEAD_DIM,
    impl,
    datatype,
    device=DEVICE,
):
    dtype = datatype
    q = torch.randn(
        (BSZ, NUM_HEADS, SEQ_LEN, HEAD_DIM),
        dtype=dtype,
        device=device,
        requires_grad=False,
    )
    k = torch.randn(
        (BSZ, NUM_HEADS, SEQ_LEN, HEAD_DIM),
        dtype=dtype,
        device=device,
        requires_grad=False,
    )

    # Create position ids
    pos_ids = torch.arange(SEQ_LEN, device=device, dtype=torch.long).unsqueeze(0)
    pos_ids = pos_ids.expand(BSZ, -1)

    # Create rotary embeddings
    cos, sin = create_rotary_embeddings(SEQ_LEN, HEAD_DIM, dtype, device)
    cos = cos.unsqueeze(0).expand(BSZ, -1, -1)
    sin = sin.unsqueeze(0).expand(BSZ, -1, -1)

    fn = IMPL_FUNCS[impl]
    result_fn = lambda: fn(q, k, cos, sin, pos_ids)
    ref = lambda: apply_rope_eager(q, k, cos, sin)
    torch.testing.assert_close(result_fn(), ref(), atol=5e-2, rtol=5e-2)

    # Benchmark the function
    ms = triton.testing.do_bench_cudagraph(result_fn)

    # Calculate memory bandwidth
    # Total memory: read q, k, cos, sin + write q_out, k_out
    bytes_per_element = q.element_size()
    q_bytes = BSZ * NUM_HEADS * SEQ_LEN * HEAD_DIM * bytes_per_element
    k_bytes = BSZ * NUM_HEADS * SEQ_LEN * HEAD_DIM * bytes_per_element
    cos_sin_bytes = 2 * BSZ * SEQ_LEN * HEAD_DIM * bytes_per_element
    total_bytes = 2 * q_bytes + 2 * k_bytes + cos_sin_bytes
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)

    return gb_per_s


if __name__ == "__main__":
    bench_rope.run(print_data=True, save_path=".")
