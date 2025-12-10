# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Test for paged KV cache flash decode attention.

This test verifies the paged attention implementation by:
1. Creating a paged KV cache from contiguous K/V tensors
2. Running both paged and non-paged versions
3. Comparing outputs
"""

import math
import pytest
import torch


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
    
    # Calculate number of pages needed per sequence
    num_pages_per_seq = (seq_len + page_size - 1) // page_size
    total_pages = batch_size * num_pages_per_seq
    
    # Allocate paged cache
    k_cache = torch.zeros(
        (total_pages, num_kv_heads, page_size, head_dim),
        device=device, dtype=dtype
    )
    v_cache = torch.zeros(
        (total_pages, num_kv_heads, page_size, head_dim),
        device=device, dtype=dtype
    )
    
    # Create block tables (simple sequential allocation)
    block_tables = torch.zeros(
        (batch_size, num_pages_per_seq), device=device, dtype=torch.int32
    )
    
    # Fill the paged cache and block tables
    page_idx = 0
    for b in range(batch_size):
        for p in range(num_pages_per_seq):
            start = p * page_size
            end = min(start + page_size, seq_len)
            actual_len = end - start
            
            # Copy data to page
            k_cache[page_idx, :, :actual_len, :] = k[b, :, start:end, :]
            v_cache[page_idx, :, :actual_len, :] = v[b, :, start:end, :]
            
            # Record page mapping
            block_tables[b, p] = page_idx
            page_idx += 1
    
    # Sequence lengths
    seq_lens = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)
    
    return k_cache, v_cache, block_tables, seq_lens


# Import tilegym and common only when running as a test module (not standalone)
if __name__ != "__main__":
    import tilegym
    from .. import common
    
    _base_class = common.PyTestCase
else:
    _base_class = object


class Test_FlashDecodePaged(_base_class):
    """Test paged KV cache flash decode attention."""
    
    @staticmethod
    def reference(q, k, v, sm_scale):
        """Reference implementation using PyTorch SDPA."""
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=sm_scale, enable_gqa=True
        )

    _backends = ["cutile"]
    
    @pytest.mark.parametrize("seq_len", [128, 256, 512])
    @pytest.mark.parametrize("group_size", [1, 4, 8])
    @pytest.mark.parametrize("page_size", [128])  # PAGE_SIZE == TILE_N
    @pytest.mark.parametrize("dtype", [torch.float16])
    @pytest.mark.parametrize("backend", _backends)
    def test_paged_attention(self, seq_len, group_size, page_size, dtype, backend, arch):
        """Test paged attention matches non-paged reference."""
        
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")
        self.setUp()

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Test parameters
        batch_size = 2
        num_heads = 32
        head_dim = 64

        torch.manual_seed(42)
        
        # Create query (single token decode)
        q = torch.randn(
            batch_size, num_heads, 1, head_dim, device="cuda", dtype=dtype
        )
        
        # Create contiguous K/V (will be converted to paged format)
        k = torch.randn(
            batch_size, num_heads // group_size, seq_len, head_dim,
            device="cuda", dtype=dtype
        )
        v = torch.randn(
            batch_size, num_heads // group_size, seq_len, head_dim,
            device="cuda", dtype=dtype
        )

        sm_scale = 1.0 / math.sqrt(head_dim)

        # Convert to paged format
        k_cache, v_cache, block_tables, seq_lens = create_paged_kv_cache(
            k, v, page_size
        )

        # Run paged attention
        output_paged = tilegym.ops.fmha_decode_paged(
            q, k_cache, v_cache, block_tables, seq_lens, sm_scale
        )

        # Run reference (non-paged)
        output_ref = self.reference(q, k, v, sm_scale)

        # Compare
        torch.testing.assert_close(
            output_paged, output_ref,
            atol=1e-2, rtol=1e-2,
        )

    @pytest.mark.parametrize("dtype", [torch.float16])
    @pytest.mark.parametrize("backend", _backends)
    def test_variable_seq_lens(self, dtype, backend, arch):
        """Test with different sequence lengths per batch item."""
        
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")
        self.setUp()

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_size = 2
        num_heads = 32
        num_kv_heads = 8
        head_dim = 64
        page_size = 128
        max_seq_len = 512

        torch.manual_seed(42)

        q = torch.randn(
            batch_size, num_heads, 1, head_dim, device="cuda", dtype=dtype
        )

        # Different seq lens: [256, 384]
        seq_lens_list = [256, 384]
        seq_lens = torch.tensor(seq_lens_list, device="cuda", dtype=torch.int32)

        # Allocate pages for max_seq_len
        num_pages_per_seq = (max_seq_len + page_size - 1) // page_size
        total_pages = batch_size * num_pages_per_seq

        k_cache = torch.randn(
            total_pages, num_kv_heads, page_size, head_dim,
            device="cuda", dtype=dtype
        )
        v_cache = torch.randn(
            total_pages, num_kv_heads, page_size, head_dim,
            device="cuda", dtype=dtype
        )

        # Simple sequential block tables
        block_tables = torch.zeros(
            batch_size, num_pages_per_seq, device="cuda", dtype=torch.int32
        )
        for b in range(batch_size):
            for p in range(num_pages_per_seq):
                block_tables[b, p] = b * num_pages_per_seq + p

        sm_scale = 1.0 / math.sqrt(head_dim)

        # This should run without error
        output = tilegym.ops.fmha_decode_paged(
            q, k_cache, v_cache, block_tables, seq_lens, sm_scale
        )

        # Basic sanity checks
        assert output.shape == (batch_size, num_heads, 1, head_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


if __name__ == "__main__":
    # Expanded standalone test with larger range
    import sys
    sys.path.insert(0, "/home/less/TileGym/src")
    
    print("=" * 80)
    print("Paged Attention Correctness Test Suite")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        exit(0)
    
    import tilegym
    if not tilegym.is_backend_available("cutile"):
        print("cutile backend not available, skipping")
        exit(0)
    
    tilegym.set_backend("cutile")
    
    # Test configurations
    test_configs = [
        # (batch_size, num_heads, num_kv_heads, seq_len, head_dim, page_size)
        # MHA (num_heads == num_kv_heads)
        (1, 16, 16, 128, 64, 128),
        (1, 16, 16, 256, 64, 128),
        (1, 16, 16, 512, 64, 128),
        (1, 16, 16, 1024, 64, 128),
        (1, 16, 16, 2048, 64, 128),
        # GQA with group_size=2
        (1, 16, 8, 128, 64, 128),
        (1, 16, 8, 256, 64, 128),
        (1, 16, 8, 512, 64, 128),
        (1, 16, 8, 1024, 64, 128),
        (1, 16, 8, 2048, 64, 128),
        # GQA with group_size=4
        (1, 16, 4, 128, 64, 128),
        (1, 16, 4, 256, 64, 128),
        (1, 16, 4, 512, 64, 128),
        (1, 16, 4, 1024, 64, 128),
        (1, 16, 4, 2048, 64, 128),
        # GQA with group_size=8
        (2, 16, 2, 128, 64, 128),
        (2, 16, 2, 512, 64, 128),
        (2, 16, 2, 1024, 64, 128),
        (2, 16, 2, 2048, 64, 128),
        # MQA (num_kv_heads=1)
        (2, 16, 1, 128, 64, 128),
        (2, 16, 1, 512, 64, 128),
        (2, 16, 1, 1024, 64, 128),
        (2, 16, 1, 2048, 64, 128),
        # Different head_dim
        (1, 16, 4, 512, 128, 128),
        (1, 16, 4, 1024, 128, 128),
        (1, 16, 4, 2048, 128, 128),
        # Larger batch
        (4, 16, 4, 512, 64, 128),
        (4, 16, 4, 1024, 64, 128),
        (4, 16, 4, 2048, 64, 128),
    ]
    
    dtype = torch.float16
    passed = 0
    failed = 0
    
    print(f"\n{'Config':<55} | {'Max Diff':>10} | {'Status':>8}")
    print("-" * 80)
    
    for batch_size, num_heads, num_kv_heads, seq_len, head_dim, page_size in test_configs:
        torch.manual_seed(42)
        
        # Create inputs
        q = torch.randn(batch_size, num_heads, 1, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device="cuda", dtype=dtype)
        v = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device="cuda", dtype=dtype)
        
        sm_scale = 1.0 / math.sqrt(head_dim)
        
        # Convert to paged format
        k_cache, v_cache, block_tables, seq_lens = create_paged_kv_cache(k, v, page_size)
        
        try:
            # Run paged attention
            output_paged = tilegym.ops.fmha_decode_paged(
                q, k_cache, v_cache, block_tables, seq_lens, sm_scale
            )
            
            # Run reference
            output_ref = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, scale=sm_scale, enable_gqa=True
            )
            
            # Compare
            max_diff = (output_paged - output_ref).abs().max().item()
            
            config_str = f"b={batch_size}, h={num_heads}, kv_h={num_kv_heads}, seq={seq_len}, d={head_dim}"
            
            if max_diff < 1e-2:
                print(f"{config_str:<55} | {max_diff:>10.6f} | {'✓ PASS':>8}")
                passed += 1
            else:
                print(f"{config_str:<55} | {max_diff:>10.6f} | {'✗ FAIL':>8}")
                failed += 1
        except Exception as e:
            config_str = f"b={batch_size}, h={num_heads}, kv_h={num_kv_heads}, seq={seq_len}, d={head_dim}"
            print(f"{config_str:<55} | {'ERROR':>10} | {'✗ FAIL':>8}")
            print(f"    Error: {e}")
            failed += 1
    
    print("-" * 80)
    print(f"\nResults: {passed} passed, {failed} failed out of {len(test_configs)} tests")
    
    if failed == 0:
        print("\n✓ All tests PASSED!")
        exit(0)
    else:
        print(f"\n✗ {failed} tests FAILED!")
        exit(1)
