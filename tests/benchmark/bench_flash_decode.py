# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math
import os

import torch
import triton

import tilegym
from tilegym.backend import is_backend_available, register_impl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Flash-decode is a single-token attention (q_len == 1).
Q_LEN = 1


def reference_fmha_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float = None,
    kv_len_per_split: int | None = None,  # Unused - kept for interface compatibility
):
    """Reference implementation using PyTorch SDPA.

    Notes:
    - This is decode, so the query corresponds to the *latest* token. With q_len==1
      and a full KV cache, the correct mask is "attend to all keys". We therefore
      use SDPA with is_causal=False.
    - enable_gqa=True is required when H_q != H_kv (grouped query attention).
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))
    return torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=sm_scale,
        enable_gqa=True,
    )


register_impl("fmha_decode", "torch")(reference_fmha_decode)


ALL_BACKENDS = [
    ("cutile", "CuTile", ("orange", "-")) if is_backend_available("cutile") else None,
    ("torch", "PyTorch (SDPA)", ("green", "-")),
]


def get_supported_backends(datatype: torch.dtype):
    return [p for p in ALL_BACKENDS if p is not None]


def create_benchmark_config(datatype: torch.dtype, head_dim: int, group_size: int):
    available_backends = get_supported_backends(datatype)
    if not available_backends:
        return None

    backends, names, styles = zip(*available_backends)
    dtype_name = str(datatype).split(".")[-1]

    # Include long-context decode lengths (up to 64K).
    # 2**8  = 256  ... 2**16 = 65536
    max_range = 17

    return triton.testing.Benchmark(
        x_names=["S_KV"],
        x_vals=[2**i for i in range(8, max_range)],
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        xlabel="KV sequence length",
        ylabel="TFLOPS",
        plot_name=(
            f"flash-decode-q{Q_LEN}-hq32-g{group_size}-d{head_dim}-{dtype_name}-TFLOPS"
        ),
        args={
            "datatype": datatype,
            "HEAD_DIM": head_dim,
            "GROUP_SIZE": group_size,
            "BATCH": 2,
            "H_Q": 32,
        },
    )


_dtypes = [torch.bfloat16]


@triton.testing.perf_report(
    [
        create_benchmark_config(datatype, head_dim, group_size)
        for datatype in _dtypes
        for head_dim in [64]
        # Production setting: only benchmark GQA group size 8.
        for group_size in [8]
    ]
)
def bench_flash_decode(
    BATCH,
    H_Q,
    S_KV,
    HEAD_DIM,
    GROUP_SIZE,
    backend,
    datatype,
    device=DEVICE,
):
    assert Q_LEN == 1
    assert H_Q % GROUP_SIZE == 0
    H_KV = H_Q // GROUP_SIZE

    q = torch.randn((BATCH, H_Q, Q_LEN, HEAD_DIM), device=device, dtype=datatype)
    k = torch.randn((BATCH, H_KV, S_KV, HEAD_DIM), device=device, dtype=datatype)
    v = torch.randn((BATCH, H_KV, S_KV, HEAD_DIM), device=device, dtype=datatype)

    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    fn = lambda: tilegym.ops.fmha_decode(q, k, v, sm_scale=sm_scale, backend=backend)

    # Correctness
    ref = lambda: reference_fmha_decode(q, k, v, sm_scale=sm_scale)
    torch.testing.assert_close(fn(), ref(), atol=5e-2, rtol=1e-2)

    ms = triton.testing.do_bench_cudagraph(fn)

    # FLOPs: 2*(QK) + 2*(PV) with Q_LEN=1
    total_flops = 4.0 * BATCH * H_Q * Q_LEN * S_KV * HEAD_DIM
    return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    save_path = "./graphs"
    os.makedirs(save_path, exist_ok=True)
    # Run the sweep, then add a derived speedup column for readability.
    # Triton will still save the default CSV/plots; we additionally save an
    # augmented CSV containing speedup when both backends are present.
    dfs = bench_flash_decode.run(print_data=False, save_path=save_path)

    # `run()` may return a single DataFrame or a list of DataFrames depending on Triton version.
    if not isinstance(dfs, list):
        dfs = [dfs]

    for df in dfs:
        cutile_col = "CuTile"
        torch_col = "PyTorch (SDPA)"
        if cutile_col in df.columns and torch_col in df.columns:
            df["speedup"] = df[cutile_col] / df[torch_col]
        print()
        # The report name is already embedded in Triton's internal object; printing the DF is enough.
        print(df.to_string(index=False))
        print()

        # Save augmented CSV next to the standard artifacts.
        # Use a stable filename derived from the plot_name if available.
        plot_name = getattr(df, "name", None) or "flash-decode"
        out_csv = os.path.join(save_path, f"{plot_name}-with-speedup.csv")
        try:
            df.to_csv(out_csv, index=False)
        except Exception:
            # If pandas isn't available or the DF isn't a pandas object, just skip.
            pass
