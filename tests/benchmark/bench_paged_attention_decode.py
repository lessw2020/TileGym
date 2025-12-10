# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Benchmark for paged KV cache attention decode.

Compares:
1. CuTile paged attention (with block_tables indirection)
2. CuTile contiguous attention (non-paged baseline)
3. PyTorch SDPA (reference)

This benchmark measures memory bandwidth (GB/s) since decode is memory-bound.
"""

import math
import torch
import triton

import tilegym
from tilegym.backend import is_backend_available, register_impl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def create_paged_kv_cache(k, v, page_size):
    """
    Convert contiguous K/V tensors to paged format.
    
    Args:
        k: [batch_size, num_kv_heads, seq_len, head_dim]
        v: [batch_size, num_kv_heads, seq_len, head_dim]
        page_size: tokens per page
        
    Returns:
        k_cache: [num_pages, num_kv_heads, page_size, head_dim]
        v_cache: [num_pages, num_kv_heads, page_size, head_dim]
        block_tables: [batch_size, max_num_blocks]
        seq_lens: [batch_size]
    """
    batch_size, num_kv_heads, seq_len, head_dim = k.shape
    device = k.device
    dtype = k.dtype
    
    num_pages_per_seq = (seq_len + page_size - 1) // page_size
    total_pages = batch_size * num_pages_per_seq
    
    k_cache = torch.zeros(
        (total_pages, num_kv_heads, page_size, head_dim),
        device=device, dtype=dtype
    )
    v_cache = torch.zeros(
        (total_pages, num_kv_heads, page_size, head_dim),
        device=device, dtype=dtype
    )
    
    block_tables = torch.zeros(
        (batch_size, num_pages_per_seq), device=device, dtype=torch.int32
    )
    
    page_idx = 0
    for b in range(batch_size):
        for p in range(num_pages_per_seq):
            start = p * page_size
            end = min(start + page_size, seq_len)
            actual_len = end - start
            
            k_cache[page_idx, :, :actual_len, :] = k[b, :, start:end, :]
            v_cache[page_idx, :, :actual_len, :] = v[b, :, start:end, :]
            block_tables[b, p] = page_idx
            page_idx += 1
    
    seq_lens = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)
    
    return k_cache, v_cache, block_tables, seq_lens


def reference_fmha_decode(q, k, v, sm_scale):
    """Reference implementation using PyTorch SDPA with GQA support."""
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, scale=sm_scale, enable_gqa=True
    )


register_impl("fmha_decode", "torch")(reference_fmha_decode)


# Backend configurations
ALL_BACKENDS = [
    ("cutile_paged", "CuTile Paged", ("blue", "-")),
    ("cutile", "CuTile Contiguous", ("orange", "-")) if is_backend_available("cutile") else None,
    ("torch", "PyTorch SDPA", ("green", "--")),
]


def get_supported_backends():
    """Filter backends based on availability"""
    return [p for p in ALL_BACKENDS if p is not None]


def create_benchmark_config(batch_size, num_heads, head_dim, group_size, page_size, dtype):
    """Create a benchmark configuration for paged attention decode"""
    available_backends = get_supported_backends()
    if not available_backends:
        return None

    backends, names, styles = zip(*available_backends)
    dtype_name = str(dtype).split('.')[-1]
    num_kv_heads = num_heads // group_size

    return triton.testing.Benchmark(
        x_names=["kv_seq_len"],
        x_vals=[2**i for i in range(7, 15)],  # 128 to 16384
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="GB/s",
        plot_name=f"paged-attention-decode-batch{batch_size}-qheads{num_heads}-kvheads{num_kv_heads}-d{head_dim}-page{page_size}-{dtype_name}-GBps",
        args={
            "dtype": dtype,
            "batch_size": batch_size,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "group_size": group_size,
            "page_size": page_size,
        },
    )


@triton.testing.perf_report(
    [
        create_benchmark_config(batch_size, num_heads, head_dim, group_size, page_size, dtype)
        for batch_size in [1, 4]
        for num_heads in [32]
        for head_dim in [64, 128]
        for group_size in [4]  # GQA with 4 query heads per KV head
        for page_size in [128]
        for dtype in [torch.float16]
    ]
)
def bench_paged_attention_decode(
    kv_seq_len,
    backend,
    dtype,
    batch_size,
    num_heads,
    head_dim,
    group_size,
    page_size,
    device=DEVICE,
):
    num_kv_heads = num_heads // group_size
    sm_scale = 1.0 / math.sqrt(head_dim)
    
    # Create query (single token decode)
    q = torch.randn(
        batch_size, num_heads, 1, head_dim,
        device=device, dtype=dtype
    )
    
    # Create contiguous K/V
    k = torch.randn(
        batch_size, num_kv_heads, kv_seq_len, head_dim,
        device=device, dtype=dtype
    )
    v = torch.randn(
        batch_size, num_kv_heads, kv_seq_len, head_dim,
        device=device, dtype=dtype
    )
    
    # Convert to paged format
    k_cache, v_cache, block_tables, seq_lens = create_paged_kv_cache(k, v, page_size)
    
    # Select the appropriate function based on backend
    if backend == "cutile_paged":
        if not is_backend_available("cutile"):
            return float('nan')
        # Pass max_seq_len explicitly for CUDA graph compatibility
        fn = lambda: tilegym.ops.fmha_decode_paged(
            q, k_cache, v_cache, block_tables, seq_lens, sm_scale,
            max_seq_len=kv_seq_len, backend="cutile"
        )
    elif backend == "cutile":
        if not is_backend_available("cutile"):
            return float('nan')
        fn = lambda: tilegym.ops.fmha_decode(q, k, v, sm_scale, backend="cutile")
    else:  # torch
        fn = lambda: reference_fmha_decode(q, k, v, sm_scale)
    
    # Verify correctness
    ref = lambda: reference_fmha_decode(q, k, v, sm_scale)
    try:
        torch.testing.assert_close(fn(), ref(), atol=1e-2, rtol=1e-2)
    except AssertionError as e:
        print(f"Warning: {backend} correctness check failed: {e}")
    
    # Benchmark
    ms = triton.testing.do_bench_cudagraph(fn)
    
    # Calculate memory bandwidth in GB/s
    # For decode: read Q, K, V; write O
    bytes_per_element = q.element_size()
    
    q_bytes = q.numel() * bytes_per_element
    k_bytes = k.numel() * bytes_per_element
    v_bytes = v.numel() * bytes_per_element
    o_bytes = q.numel() * bytes_per_element  # Output same size as Q
    
    # For paged, we also read block_tables (but it's small)
    if backend == "cutile_paged":
        block_table_bytes = block_tables.numel() * block_tables.element_size()
        total_bytes = q_bytes + k_bytes + v_bytes + o_bytes + block_table_bytes
    else:
        total_bytes = q_bytes + k_bytes + v_bytes + o_bytes
    
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)
    
    return gb_per_s


# Also create a TFLOPS benchmark for comparison
def create_tflops_benchmark_config(batch_size, num_heads, head_dim, group_size, page_size, dtype):
    """Create a TFLOPS benchmark configuration"""
    available_backends = get_supported_backends()
    if not available_backends:
        return None

    backends, names, styles = zip(*available_backends)
    dtype_name = str(dtype).split('.')[-1]
    num_kv_heads = num_heads // group_size

    return triton.testing.Benchmark(
        x_names=["kv_seq_len"],
        x_vals=[2**i for i in range(7, 15)],
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="TFLOPS",
        plot_name=f"paged-attention-decode-batch{batch_size}-qheads{num_heads}-kvheads{num_kv_heads}-d{head_dim}-page{page_size}-{dtype_name}-TFLOPS",
        args={
            "dtype": dtype,
            "batch_size": batch_size,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "group_size": group_size,
            "page_size": page_size,
        },
    )


@triton.testing.perf_report(
    [
        create_tflops_benchmark_config(batch_size, num_heads, head_dim, group_size, page_size, dtype)
        for batch_size in [1, 4]
        for num_heads in [32]
        for head_dim in [64, 128]
        for group_size in [4]
        for page_size in [128]
        for dtype in [torch.float16]
    ]
)
def bench_paged_attention_decode_tflops(
    kv_seq_len,
    backend,
    dtype,
    batch_size,
    num_heads,
    head_dim,
    group_size,
    page_size,
    device=DEVICE,
):
    num_kv_heads = num_heads // group_size
    sm_scale = 1.0 / math.sqrt(head_dim)
    
    q = torch.randn(batch_size, num_heads, 1, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_kv_heads, kv_seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_kv_heads, kv_seq_len, head_dim, device=device, dtype=dtype)
    
    k_cache, v_cache, block_tables, seq_lens = create_paged_kv_cache(k, v, page_size)
    
    if backend == "cutile_paged":
        if not is_backend_available("cutile"):
            return float('nan')
        fn = lambda: tilegym.ops.fmha_decode_paged(
            q, k_cache, v_cache, block_tables, seq_lens, sm_scale,
            max_seq_len=kv_seq_len, backend="cutile"
        )
    elif backend == "cutile":
        if not is_backend_available("cutile"):
            return float('nan')
        fn = lambda: tilegym.ops.fmha_decode(q, k, v, sm_scale, backend="cutile")
    else:
        fn = lambda: reference_fmha_decode(q, k, v, sm_scale)
    
    ms = triton.testing.do_bench_cudagraph(fn)
    
    # FLOPS calculation for attention decode:
    # Q @ K^T: 2 * batch * num_heads * 1 * kv_seq_len * head_dim
    # softmax: ~5 * batch * num_heads * kv_seq_len (exp, sum, div - approximation)
    # P @ V: 2 * batch * num_heads * 1 * kv_seq_len * head_dim
    flops_qk = 2.0 * batch_size * num_heads * 1 * kv_seq_len * head_dim
    flops_pv = 2.0 * batch_size * num_heads * 1 * kv_seq_len * head_dim
    total_flops = flops_qk + flops_pv
    
    tflops = total_flops * 1e-12 / (ms * 1e-3)
    
    return tflops


if __name__ == "__main__":
    print("=" * 80)
    print("Paged Attention Decode Benchmark - Memory Bandwidth (GB/s)")
    print("=" * 80)
    bench_paged_attention_decode.run(print_data=True)
    
    print("\n" + "=" * 80)
    print("Paged Attention Decode Benchmark - Compute (TFLOPS)")
    print("=" * 80)
    bench_paged_attention_decode_tflops.run(print_data=True)
