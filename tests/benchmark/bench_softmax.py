# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
import triton

import tilegym
from tilegym.backend import is_backend_available

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def reference_softmax(x: torch.Tensor):
    """Reference implementation of softmax using PyTorch"""
    return torch.nn.functional.softmax(x, dim=-1)


def cutile_gather_softmax(x: torch.Tensor):
    """CuTile softmax using ct.gather/ct.scatter"""
    return tilegym.ops.softmax(x, use_tma=False, backend="cutile")


def cutile_tma_softmax(x: torch.Tensor):
    """CuTile softmax using TMA (ct.load/ct.store)"""
    return tilegym.ops.softmax(x, use_tma=True, backend="cutile")


# Available implementations with their display names and plot styles
ALL_IMPLEMENTATIONS = [
    ("cutile_gather", "CuTile (ct.gather)", ("blue", "-"), cutile_gather_softmax)
    if is_backend_available("cutile")
    else None,
    ("cutile_tma", "CuTile (TMA)", ("orange", "-"), cutile_tma_softmax)
    if is_backend_available("cutile")
    else None,
    ("torch", "PyTorch", ("green", "-"), reference_softmax),
]


def get_supported_implementations():
    """Filter implementations based on availability"""
    return [p for p in ALL_IMPLEMENTATIONS if p is not None]


def create_benchmark_config(M, dtype):
    """Create a benchmark configuration for given parameters"""
    available_impls = get_supported_implementations()
    if not available_impls:
        return None

    impl_ids, names, styles, _ = zip(*available_impls)
    dtype_name = str(dtype).split('.')[-1]

    return triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[2**i for i in range(10, 15)],
        line_arg='impl',
        line_vals=list(impl_ids),
        line_names=list(names),
        styles=list(styles),
        ylabel='GB/s',
        plot_name=f'softmax-M{M}-{dtype_name}',
        args={'M': M, 'dtype': dtype},
    )


# Build lookup dict for implementations
IMPL_FUNCS = {p[0]: p[3] for p in get_supported_implementations()}


@triton.testing.perf_report(
    [
        create_benchmark_config(M, dtype)
        for M in [4096]  # Matrix height
        for dtype in [torch.float32, torch.bfloat16]
    ]
)
def bench_softmax(M, N, impl, dtype=torch.float32, device=DEVICE):
    # Create data
    x = torch.randn(M, N, dtype=dtype, device=device)

    fn = IMPL_FUNCS[impl]
    result_fn = lambda: fn(x)
    ref = lambda: reference_softmax(x)
    torch.testing.assert_close(result_fn(), ref(), atol=1e-2, rtol=1e-2)

    # Benchmark the function
    ms = triton.testing.do_bench_cudagraph(result_fn)

    # Calculate memory bandwidth (GB/s)
    # Softmax operation: reads input, writes output
    # Memory access: read x + write output = 2 * x.numel() elements
    total_bytes = 2 * x.numel() * x.element_size()

    # Convert to GB/s
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)

    return gb_per_s


if __name__ == "__main__":
    bench_softmax.run(print_data=True, save_path=".")
