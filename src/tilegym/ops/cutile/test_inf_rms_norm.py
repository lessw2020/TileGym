# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

import importlib.util
from pathlib import Path


def _load_inf_rms_norm_module():
    """
    Load inf_rms_norm.py by file path.

    This test is intentionally tilegym-package-independent so it can be reused
    in external frameworks (e.g. TorchTitan) where this file may be vendored.
    """
    here = Path(__file__).resolve().parent
    target = here / "inf_rms_norm.py"

    # Fallback if this test is relocated elsewhere (e.g. under tests/)
    if not target.exists():
        repo_root = Path(__file__).resolve().parents[2]
        target = repo_root / "src" / "tilegym" / "ops" / "cutile" / "inf_rms_norm.py"

    if not target.exists():
        pytest.skip(f"inf_rms_norm.py not found (looked at {target})")

    spec = importlib.util.spec_from_file_location("inf_rms_norm", target)
    if spec is None or spec.loader is None:
        pytest.skip("Could not create module spec for inf_rms_norm.py")

    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except (ModuleNotFoundError, ImportError) as e:
        # Common when cuda.tile isn't installed / import context differs.
        pytest.skip(f"Could not import inf_rms_norm.py: {e}")
    return mod


def _reference_rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    # RMSNorm reference (HF-style): compute in float32, then cast to weight dtype.
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)

    if w is None:
        return x

    if w.dtype in [torch.float16, torch.bfloat16]:
        x = x.to(w.dtype)

    return w * x


@pytest.mark.parametrize(
    "m, n, dtype",
    [
        (256, 256, torch.float16),
        (4096, 2**8, torch.bfloat16),
        (256, 256, torch.float32),
    ],
)
def test_inf_rms_norm_wrapper_matches_reference(m, n, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for cuTile RMSNorm")

    mod = _load_inf_rms_norm_module()
    inf_rms_norm_wrapper = getattr(mod, "inf_rms_norm_wrapper", None)
    if inf_rms_norm_wrapper is None:
        pytest.skip("inf_rms_norm_wrapper not found in inf_rms_norm.py")

    device = torch.device("cuda")
    eps = 1e-5

    x_shape = (m, n)
    w_shape = (n,)

    # Avoid in-place ops on leaf tensors that require grad
    x = (torch.rand(x_shape, dtype=dtype, device=device) * 0.5 - 2.3).requires_grad_(True)
    w = torch.randn(w_shape, dtype=dtype, device=device, requires_grad=True)

    # Inference-only behavior: output should not require grad
    y = inf_rms_norm_wrapper(x, w, eps, n)
    assert y.requires_grad is False

    y_ref = _reference_rms_norm(x, w, eps)
    torch.testing.assert_close(y, y_ref, rtol=0.0, atol=5e-2)
