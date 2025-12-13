# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import importlib.util
from pathlib import Path

import torch
import triton

try:
    import tilegym  # optional (only used if available)
    from tilegym.backend import is_backend_available
except Exception:  # pragma: no cover
    tilegym = None

    def is_backend_available(_backend: str) -> bool:  # type: ignore[no-redef]
        return False


DEVICE = triton.runtime.driver.active.get_active_torch_device()

_TORCH_NN_RMSNORM_CACHE: dict[tuple[int, torch.dtype, torch.device, float, int], torch.nn.RMSNorm] = {}

# Toggle to temporarily disable persistent RMSNorm in plots/benchmarks.
# (Keeping the implementation wired up makes it easy to re-enable later.)
ENABLE_CUTILE_PERSISTENT = False


def _load_inf_rms_norm_wrapper():
    """Load inf_rms_norm_wrapper from inf_rms_norm.py by file path."""
    repo_root = Path(__file__).resolve().parents[2]
    target = repo_root / "src" / "tilegym" / "ops" / "cutile" / "inf_rms_norm.py"
    if not target.exists():
        return None

    spec = importlib.util.spec_from_file_location("inf_rms_norm", target)
    if spec is None or spec.loader is None:
        return None

    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception:
        return None

    return getattr(mod, "inf_rms_norm_wrapper", None)


INF_RMS_NORM_WRAPPER = _load_inf_rms_norm_wrapper()


def reference_rms_norm(x: torch.Tensor, w_shape: tuple, weight: torch.Tensor, eps: float):
    """Reference implementation of RMSNorm using PyTorch (eager)"""
    dims = tuple(i for i in range(-1, -len(w_shape) - 1, -1))
    variance = x.to(torch.float32).pow(2).mean(dims, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)

    if weight.dtype in [torch.float16, torch.bfloat16]:
        x_norm = x_norm.to(weight.dtype)

    return weight * x_norm


@torch.compile
def compiled_rms_norm(x: torch.Tensor, w_shape: tuple, weight: torch.Tensor, eps: float):
    """Compiled RMSNorm using torch.compile"""
    dims = tuple(i for i in range(-1, -len(w_shape) - 1, -1))
    variance = x.to(torch.float32).pow(2).mean(dims, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)

    if weight.dtype in [torch.float16, torch.bfloat16]:
        x_norm = x_norm.to(weight.dtype)

    return weight * x_norm


def torch_nn_rms_norm(x: torch.Tensor, w_shape: tuple, weight: torch.Tensor, eps: float):
    """RMSNorm using torch.nn.RMSNorm (module forward)."""
    # Cache the module so the benchmark doesn't measure module construction.
    key = (weight.numel(), weight.dtype, weight.device, float(eps), int(weight.data_ptr()))
    mod = _TORCH_NN_RMSNORM_CACHE.get(key)
    if mod is None:
        mod = torch.nn.RMSNorm(weight.shape[0], eps=eps, elementwise_affine=True).to(
            device=weight.device, dtype=weight.dtype
        )
        # Rebind module weight parameter to the provided tensor (no copy).
        mod.weight = torch.nn.Parameter(weight, requires_grad=False)
        mod.eval()
        _TORCH_NN_RMSNORM_CACHE[key] = mod
    return mod(x)


def cutile_inf_rms_norm(x: torch.Tensor, w_shape: tuple, weight: torch.Tensor, eps: float):
    """Inference-only cuTile RMSNorm wrapper (inf_rms_norm.py)."""
    if INF_RMS_NORM_WRAPPER is None:
        raise RuntimeError("inf_rms_norm_wrapper unavailable (import failed)")
    return INF_RMS_NORM_WRAPPER(x, weight, eps, weight.shape[0])


def cutile_persistent_rms_norm(x: torch.Tensor, w_shape: tuple, weight: torch.Tensor, eps: float):
    """CuTile RMSNorm with static persistent scheduling (tilegym op)."""
    if tilegym is None:
        raise RuntimeError("tilegym not available")
    return tilegym.ops.rms_norm(x, w_shape, weight, eps, static_persistent=True, backend="cutile")


def cutile_non_persistent_rms_norm(x: torch.Tensor, w_shape: tuple, weight: torch.Tensor, eps: float):
    """CuTile RMSNorm without static persistent scheduling (tilegym op)."""
    if tilegym is None:
        raise RuntimeError("tilegym not available")
    return tilegym.ops.rms_norm(x, w_shape, weight, eps, static_persistent=False, backend="cutile")


ALL_IMPLEMENTATIONS = [
    ("cutile_inf", "CuTile (Inf Wrapper)", ("purple", "-"), cutile_inf_rms_norm)
    if INF_RMS_NORM_WRAPPER is not None
    else None,
    ("cutile_persistent", "CuTile (Persistent)", ("blue", "-"), cutile_persistent_rms_norm)
    if (ENABLE_CUTILE_PERSISTENT and is_backend_available("cutile"))
    else None,
    ("cutile_non_persistent", "CuTile (Non-Persistent)", ("orange", "-"), cutile_non_persistent_rms_norm)
    if is_backend_available("cutile")
    else None,
    ("torch_nn", "PyTorch (nn.RMSNorm)", ("black", "--"), torch_nn_rms_norm),
    ("torch_compile", "PyTorch (torch.compile)", ("red", "-"), compiled_rms_norm),
    ("torch", "PyTorch (Eager)", ("green", "-"), reference_rms_norm),
]


def get_supported_implementations():
    return [p for p in ALL_IMPLEMENTATIONS if p is not None]


def create_benchmark_config(dtype, M):
    available_impls = get_supported_implementations()
    impl_ids, names, styles, _ = zip(*available_impls)
    dtype_name = str(dtype).split(".")[-1]

    return triton.testing.Benchmark(
        x_names=["N"],
        # Common LLM hidden sizes
        x_vals=[2048, 4096, 8192, 16384],
        line_arg="impl",
        line_vals=list(impl_ids),
        line_names=list(names),
        styles=list(styles),
        ylabel="GB/s",
        plot_name=f"inf_rms_norm-M{M}-{dtype_name}",
        args={"dtype": dtype, "M": M},
    )


IMPL_FUNCS = {p[0]: p[3] for p in get_supported_implementations()}
IMPL_META = {p[0]: (p[1], p[2]) for p in get_supported_implementations()}  # id -> (name, (color, linestyle))


@triton.testing.perf_report(
    [create_benchmark_config(dtype, M) for dtype in [torch.float16, torch.bfloat16] for M in [4096]]
)
def bench_inf_rmsnorm(N, impl, dtype, M, device=DEVICE):
    eps = 1e-5

    x_shape = (M, N)
    w_shape = (N,)

    x = torch.rand(x_shape, dtype=dtype, device=device, requires_grad=False).mul_(0.5).add_(-2.3)
    weight = torch.randn(w_shape, dtype=dtype, device=device, requires_grad=False)

    fn = IMPL_FUNCS[impl]
    result_fn = lambda: fn(x, w_shape, weight, eps)
    ref = lambda: reference_rms_norm(x, w_shape, weight, eps)
    torch.testing.assert_close(result_fn(), ref(), atol=5e-2, rtol=0.0)

    ms = triton.testing.do_bench_cudagraph(result_fn)

    # Memory bandwidth: read input, read weight, write output
    bytes_per_element = x.element_size()
    input_bytes = x.numel() * bytes_per_element
    weight_bytes = weight.numel() * bytes_per_element
    output_bytes = x.numel() * bytes_per_element
    total_bytes = input_bytes + weight_bytes + output_bytes

    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)
    return gb_per_s


def _write_bf16_plot(save_dir: Path):
    """
    Write a bf16-only plot + csv summary.

    This is a fallback for environments where triton.testing.perf_report doesn't
    emit plot images (e.g. missing matplotlib in Triton's plotting path).
    """
    Ns = [2048, 4096, 8192, 16384]
    M = 4096
    eps = 1e-5
    dtype = torch.bfloat16

    impl_ids = list(IMPL_FUNCS.keys())
    if not impl_ids:
        print("\n[plot] No implementations available; skipping bf16 plot")
        return

    # Collect GB/s per implementation per N
    results: dict[str, list[float]] = {impl: [] for impl in impl_ids}
    for N in Ns:
        x = torch.rand((M, N), dtype=dtype, device=DEVICE).mul_(0.5).add_(-2.3)
        w = torch.randn((N,), dtype=dtype, device=DEVICE)

        # Correctness (also warms up compilation)
        ref = reference_rms_norm(x, (N,), w, eps)
        for impl in impl_ids:
            y = IMPL_FUNCS[impl](x, (N,), w, eps)
            torch.testing.assert_close(y, ref, atol=5e-2, rtol=0.0)

        for impl in impl_ids:
            fn = IMPL_FUNCS[impl]
            ms = triton.testing.do_bench_cudagraph(lambda: fn(x, (N,), w, eps))
            bytes_per_element = x.element_size()
            total_bytes = x.numel() * bytes_per_element + w.numel() * bytes_per_element + x.numel() * bytes_per_element
            gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)
            results[impl].append(gb_per_s)

    # Write CSV
    csv_path = save_dir / "inf_rms_norm-M4096-bfloat16.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        headers = ["N"] + [IMPL_META[i][0] if i in IMPL_META else i for i in impl_ids]
        f.write(",".join(headers) + "\n")
        for idx, N in enumerate(Ns):
            row = [str(N)] + [f"{results[impl][idx]:.6f}" for impl in impl_ids]
            f.write(",".join(row) + "\n")

    # Write PNG (if matplotlib available)
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print(f"\n[plot] Wrote {csv_path.name}, but could not write PNG (matplotlib unavailable): {e}")
        return

    plt.figure(figsize=(10, 5))
    for impl in impl_ids:
        name, style = IMPL_META.get(impl, (impl, ("black", "-")))
        color, linestyle = style
        plt.plot(Ns, results[impl], label=name, color=color, linestyle=linestyle, marker="o")

    plt.title("inf_rms_norm (M=4096, bf16)")
    plt.xlabel("Hidden size (N)")
    plt.ylabel("GB/s")
    plt.grid(True, linestyle=":", linewidth=0.8)
    plt.legend()
    plt.tight_layout()

    png_path = save_dir / "inf_rms_norm-M4096-bfloat16.png"
    plt.savefig(png_path, dpi=160)
    plt.close()
    print(f"\n[plot] Wrote {png_path.name} (and {csv_path.name}) to {save_dir}")


if __name__ == "__main__":
    save_dir = Path(__file__).resolve().parent
    bench_inf_rmsnorm.run(print_data=True, save_path=str(save_dir))
    _write_bf16_plot(save_dir)

    # Print speedup summary: Inf wrapper vs CuTile non-persistent
    if INF_RMS_NORM_WRAPPER is None:
        print("\n[summary] Skipping speedup: inf_rms_norm_wrapper unavailable")
        raise SystemExit(0)

    if tilegym is None or not is_backend_available("cutile"):
        print("\n[summary] Skipping speedup: tilegym/cutile unavailable")
        raise SystemExit(0)

    Ns = [2048, 4096, 8192, 16384]
    M = 4096
    eps = 1e-5

    print("\n" + "=" * 90)
    print("SPEEDUP SUMMARY: CuTile (Inf Wrapper) vs CuTile (Non-Persistent)")
    print("=" * 90)
    print(f"{'dtype':>10} {'N':>8} {'inf (ms)':>12} {'nonpersist (ms)':>16} {'speedup':>10}")
    print("-" * 90)

    for dtype in [torch.float16, torch.bfloat16]:
        for N in Ns:
            x = torch.rand((M, N), dtype=dtype, device=DEVICE).mul_(0.5).add_(-2.3)
            w = torch.randn((N,), dtype=dtype, device=DEVICE)

            # Warmup
            _ = cutile_inf_rms_norm(x, (N,), w, eps)
            _ = cutile_non_persistent_rms_norm(x, (N,), w, eps)
            torch.cuda.synchronize()

            ms_inf = triton.testing.do_bench_cudagraph(
                lambda: cutile_inf_rms_norm(x, (N,), w, eps)
            )
            ms_np = triton.testing.do_bench_cudagraph(
                lambda: cutile_non_persistent_rms_norm(x, (N,), w, eps)
            )

            speedup = ms_np / ms_inf
            print(
                f"{str(dtype).split('.')[-1]:>10} {N:>8} {ms_inf:>12.4f} {ms_np:>16.4f} {speedup:>9.2f}x"
            )
