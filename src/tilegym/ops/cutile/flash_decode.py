# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import cuda.tile as ct
import numpy as np
import torch
from cuda.tile._numeric_semantics import RoundingMode as RMd

from tilegym.backend import register_impl
from tilegym.ops.cutile.splitk_reduce import splitk_reduce

from .utils import next_power_of_2

# Type aliases for constants
ConstInt = ct.Constant[int]

INV_LOG_2 = 1.0 / math.log(2)

@ct.kernel
def attention_decode_kernel_grouped(
    Q,
    K,
    V,  # query, key, value tensors
    Att_Out,
    LSE_Out,
    softmax_scale: float,
    stride_mid_ob: int,
    stride_mid_oh: int,
    stride_mid_os: int,
    stride_mid_lseb: int,
    stride_mid_lsem: int,
    B: int,
    H_qo: int,
    H_kv: int,
    S_kv: int,
    HEAD_DIM: ConstInt,  # head dimension
    TILE_N: ConstInt,
    KV_LEN_PER_SPLIT: ConstInt,
    NUM_Q_HEAD_PER_KV: ConstInt,
    QUERY_GROUP_TILE_SIZE: ConstInt,
    NUM_KV_SPLITS: ConstInt,
):
    # Get program IDs
    batch_id = ct.bid(0)
    head_id = ct.bid(1)
    tile_id = ct.bid(2)

    qk_scale = ct.mul(softmax_scale, INV_LOG_2)

    # Load Q with grouped query attention layout
    # Q is organized as [B, H_qo // NUM_Q_HEAD_PER_KV, NUM_Q_HEAD_PER_KV, HEAD_DIM]
    q = ct.load(
        Q,
        index=(batch_id, head_id, 0, 0),
        shape=(1, 1, QUERY_GROUP_TILE_SIZE, HEAD_DIM),
        order=(0, 1, 2, 3),
        allow_tma=True,
    )
    q = ct.reshape(q, (QUERY_GROUP_TILE_SIZE, HEAD_DIM))
    q = ct.transpose(q)  # Shape: (HEAD_DIM, QUERY_GROUP_TILE_SIZE)

    # Calculate start and end indices for this tile
    start_idx = ct.mul(tile_id, KV_LEN_PER_SPLIT)
    end_idx = ct.minimum(ct.add(start_idx, KV_LEN_PER_SPLIT), S_kv)

    # Initialize accumulators
    m_i = ct.full((QUERY_GROUP_TILE_SIZE,), -math.inf, dtype=ct.float32)
    l_i = ct.full((TILE_N, QUERY_GROUP_TILE_SIZE), 1.0, dtype=ct.float32)
    acc = ct.full((HEAD_DIM, QUERY_GROUP_TILE_SIZE), 0.0, dtype=ct.float32)

    # Pre-compute variables outside conditional
    num_tiles = ct.cdiv(KV_LEN_PER_SPLIT, TILE_N)
    start_tile = start_idx // TILE_N
    offs_n = ct.arange(TILE_N, dtype=ct.int32)

    # Process keys and values in this tile
    if end_idx > start_idx:
        # Process each KV tile
        for idx in range(num_tiles):
            cnt = start_tile + idx
            curr_n = cnt * TILE_N

            # Load K unconditionally - TMA handles bounds, enables Tensor Core optimization
            # [B, H_kv, S_kv, HEAD_DIM]
            k = ct.load(
                K,
                index=(batch_id, head_id, cnt, 0),
                shape=(1, 1, TILE_N, HEAD_DIM),
                order=(0, 1, 2, 3),
                allow_tma=True,
            )
            k = ct.reshape(k, (TILE_N, HEAD_DIM))

            # Compute qk - unconditional execution enables Tensor Core usage
            # (HEAD_DIM, QUERY_GROUP_TILE_SIZE) @ (TILE_N, HEAD_DIM).T
            # Result: (TILE_N, QUERY_GROUP_TILE_SIZE)
            qk = ct.matmul(k, q)

            # Process boundary case (non-causal) - apply mask to result only
            if curr_n + TILE_N > S_kv:
                mask = ct.less(ct.add(curr_n, offs_n[:, None]), S_kv)
                qk = ct.where(mask, qk, -1.0e6)

            # Compute softmax statistics
            qk_scaled = ct.mul(qk, qk_scale)
            m_ij = ct.maximum(m_i, ct.max(qk_scaled, 0))
            qk = ct.sub(qk_scaled, m_ij[None, :])
            p = ct.exp2(qk)

            # Update m_i and l_i
            alpha = ct.exp2(ct.sub(m_i, m_ij))
            l_i = ct.add(ct.mul(l_i, alpha[None, :]), p)

            # Update output accumulator
            acc = ct.mul(acc, alpha[None, :])

            # Load V and update accumulator
            v = ct.load(
                V,
                index=(batch_id, head_id, cnt, 0),
                shape=(1, 1, TILE_N, HEAD_DIM),
                order=(0, 1, 2, 3),
                allow_tma=True,
            )
            v = ct.reshape(v, (TILE_N, HEAD_DIM))
            v = ct.transpose(v)  # (HEAD_DIM, TILE_N)
            p = ct.astype(p, q.dtype)
            acc = ct.mma(v, p, acc=acc)

            # Update m_i
            m_i = m_ij

    l = ct.sum(l_i, 0)
    acc = ct.truediv(
        acc, l[None, :], flush_to_zero=True, rounding_mode=RMd.APPROX
    )
    acc = ct.astype(acc, ct.float32)
    acc = ct.transpose(acc)
    acc = ct.astype(acc, Att_Out.dtype)
    l = ct.add(m_i, ct.log2(l))

    # Store attention output
    acc_reshaped = ct.reshape(
        acc, (1, 1, QUERY_GROUP_TILE_SIZE, 1, HEAD_DIM)
    )

    if NUM_Q_HEAD_PER_KV == QUERY_GROUP_TILE_SIZE:
        # Use TMA store for optimal performance
        ct.store(
            Att_Out,
            index=(batch_id, head_id, 0, tile_id, 0),
            tile=acc_reshaped,
            order=(0, 1, 2, 3, 4),
            allow_tma=True,
        )
    else:
        # Use scatter with boundary checking for non-matching tile sizes
        idx_q_offset = ct.arange(QUERY_GROUP_TILE_SIZE, dtype=ct.int32)[:, None]
        idx_dim = ct.arange(HEAD_DIM, dtype=ct.int32)[None, :]
        ct.scatter(
            Att_Out,
            (batch_id, head_id, idx_q_offset, tile_id, idx_dim),
            acc,
            check_bounds=True,
            latency=1,
        )

    # Store log sum exp
    idx_lse_q_offset = ct.arange(QUERY_GROUP_TILE_SIZE, dtype=ct.int32)
    ct.scatter(
        LSE_Out,
        (batch_id, head_id, idx_lse_q_offset, tile_id),
        l,
        check_bounds=True,
        latency=1,
    )


class _attention_decode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, softmax_scale, kv_len_per_split=None):
        """
        Grouped Query Attention implementation using attention_decode_kernel_grouped.
        Supports both standard attention (num_q_heads == num_kv_heads) and
        grouped attention (num_q_heads != num_kv_heads) cases.

        Args:
            Q: Query tensor of shape [batch_size, num_q_heads, 1, head_dim]
            K: Key tensor of shape [batch_size, num_kv_heads, seq_len, head_dim]
            V: Value tensor of shape [batch_size, num_kv_heads, seq_len, head_dim]
            softmax_scale: Scale factor for attention computation
            kv_len_per_split: Optional KV length per split for parallelization

        Returns:
            O: Output tensor of shape [batch_size, num_q_heads, 1, head_dim]
        """
        # Get dimensions
        batch_size, num_q_heads = Q.shape[0], Q.shape[1]
        num_kv_heads = K.shape[1]
        seq_len, head_dim = V.shape[2], V.shape[3]

        # Reshape for processing
        Q = Q.view(batch_size, num_q_heads, head_dim)
        K = K.view(batch_size, num_kv_heads, seq_len, head_dim)
        V = V.view(batch_size, num_kv_heads, seq_len, head_dim)

        # Calculate number of tiles
        TILE_N = 128
        if kv_len_per_split is None:
            NUM_SMS = torch.cuda.get_device_properties(
                "cuda"
            ).multi_processor_count
            NUM_KV_SPLITS = NUM_SMS // (batch_size * num_kv_heads)
            TILE_SIZE = max(
                TILE_N,
                next_power_of_2((seq_len + NUM_KV_SPLITS - 1) // NUM_KV_SPLITS),
            )
            NUM_KV_SPLITS = (seq_len + TILE_SIZE - 1) // TILE_SIZE
        else:
            NUM_KV_SPLITS = (seq_len + kv_len_per_split - 1) // kv_len_per_split
            TILE_SIZE = kv_len_per_split

        assert TILE_SIZE == next_power_of_2(TILE_SIZE)

        # Allocate intermediate results
        device = Q.device
        Att_Mid_Out = torch.empty(
            (batch_size, num_q_heads, NUM_KV_SPLITS, head_dim),
            device=device,
            dtype=Q.dtype,
        )
        LSE_Out = torch.empty(
            (batch_size, num_q_heads, NUM_KV_SPLITS),
            device=device,
            dtype=torch.float32,
        )

        # Prepare output
        O = torch.empty_like(Q)

        # Calculate grouped attention parameters
        assert num_q_heads % num_kv_heads == 0
        num_q_head_per_kv = num_q_heads // num_kv_heads
        query_group_tile_size = max(8, next_power_of_2(num_q_head_per_kv))

        # Create grouped views for kernel
        Att_Mid_Out_5D = Att_Mid_Out.view(
            batch_size,
            num_kv_heads,
            num_q_head_per_kv,
            NUM_KV_SPLITS,
            head_dim,
        )
        LSE_Out_4D = LSE_Out.view(
            batch_size,
            num_kv_heads,
            num_q_head_per_kv,
            NUM_KV_SPLITS,
        )

        # Calculate strides
        stride_mid_ob, stride_mid_oh, stride_mid_os = (
            Att_Mid_Out.stride(0),
            Att_Mid_Out.stride(1),
            Att_Mid_Out.stride(2),
        )
        stride_mid_lseb, stride_mid_lsem = (
            LSE_Out.stride(0),
            LSE_Out.stride(1),
        )

        # Round up head_dim to next power of 2
        HEAD_DIM = next_power_of_2(head_dim)

        # Reshape Q for grouped query attention
        Q_grouped = Q.view(
            batch_size,
            num_q_heads // num_q_head_per_kv,
            num_q_head_per_kv,
            head_dim,
        )

        # Launch kernel
        grid = (batch_size, num_kv_heads, NUM_KV_SPLITS)

        ct.launch(
            torch.cuda.current_stream(),
            grid,
            attention_decode_kernel_grouped,
            (
                Q_grouped,
                K,
                V,
                Att_Mid_Out_5D,
                LSE_Out_4D,
                softmax_scale,
                stride_mid_ob,
                stride_mid_oh,
                stride_mid_os,
                stride_mid_lseb,
                stride_mid_lsem,
                batch_size,
                num_q_heads,
                num_kv_heads,
                seq_len,
                HEAD_DIM,
                TILE_N,
                TILE_SIZE,
                num_q_head_per_kv,
                query_group_tile_size,
                NUM_KV_SPLITS,
            ),
        )

        # Reduce kernel splitk results
        splitk_reduce(Att_Mid_Out, LSE_Out, O, seq_len)

        return O.view(batch_size, num_q_heads, 1, head_dim)

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("Attention backward is not implemented yet")


attention_decode = _attention_decode.apply


@register_impl("fmha_decode", backend="cutile")
def fmha_decode(q, k, v, sm_scale, kv_len_per_split=None, **kwargs):
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))
    o = attention_decode(q, k, v, sm_scale, kv_len_per_split)
    return o


# =============================================================================
# Paged KV Cache Support (PAGE_SIZE == TILE_N for simplicity)
# =============================================================================


@ct.kernel
def attention_decode_kernel_paged(
    Q,
    K_cache,      # [num_pages * num_kv_heads * page_size * head_dim] flattened
    V_cache,      # [num_pages * num_kv_heads * page_size * head_dim] flattened
    block_tables, # [batch_size * max_num_blocks] flattened
    Att_Out,
    LSE_Out,
    softmax_scale: float,
    stride_mid_ob: int,
    stride_mid_oh: int,
    stride_mid_os: int,
    stride_mid_lseb: int,
    stride_mid_lsem: int,
    max_num_blocks: int,
    max_seq_len: int,
    stride_kv_page: int,    # stride for page dimension in K/V cache
    stride_kv_head: int,    # stride for head dimension in K/V cache  
    stride_kv_seq: int,     # stride for seq dimension in K/V cache
    B: int,
    H_qo: int,
    H_kv: int,
    HEAD_DIM: ConstInt,
    PAGE_SIZE: ConstInt,      # PAGE_SIZE == TILE_N
    KV_LEN_PER_SPLIT: ConstInt,
    NUM_Q_HEAD_PER_KV: ConstInt,
    QUERY_GROUP_TILE_SIZE: ConstInt,
    NUM_KV_SPLITS: ConstInt,
):
    """
    Paged attention decode kernel.
    
    Key difference from non-paged: K/V are loaded via indirection through block_tables.
    Constraint: PAGE_SIZE == TILE_N (each tile = one page, no boundary handling needed)
    
    K/V cache is flattened and accessed via ct.gather with computed indices.
    """
    # Get program IDs
    batch_id = ct.bid(0)
    head_id = ct.bid(1)
    tile_id = ct.bid(2)

    qk_scale = ct.mul(softmax_scale, INV_LOG_2)

    # Load Q with grouped query attention layout
    q = ct.load(
        Q,
        index=(batch_id, head_id, 0, 0),
        shape=(1, 1, QUERY_GROUP_TILE_SIZE, HEAD_DIM),
        order=(0, 1, 2, 3),
        allow_tma=True,
    )
    q = ct.reshape(q, (QUERY_GROUP_TILE_SIZE, HEAD_DIM))
    q = ct.transpose(q)  # Shape: (HEAD_DIM, QUERY_GROUP_TILE_SIZE)

    # Calculate start and end indices for this split
    start_idx = ct.mul(tile_id, KV_LEN_PER_SPLIT)
    end_idx = ct.minimum(ct.add(start_idx, KV_LEN_PER_SPLIT), max_seq_len)

    # Initialize accumulators
    m_i = ct.full((QUERY_GROUP_TILE_SIZE,), -math.inf, dtype=ct.float32)
    l_i = ct.full((PAGE_SIZE, QUERY_GROUP_TILE_SIZE), 1.0, dtype=ct.float32)
    acc = ct.full((HEAD_DIM, QUERY_GROUP_TILE_SIZE), 0.0, dtype=ct.float32)

    # Pre-compute loop bounds
    num_tiles = ct.cdiv(KV_LEN_PER_SPLIT, PAGE_SIZE)
    start_tile = start_idx // PAGE_SIZE
    offs_n = ct.arange(PAGE_SIZE, dtype=ct.int32)
    offs_d = ct.arange(HEAD_DIM, dtype=ct.int32)
    
    # Base offset for block_tables lookup (flattened: batch_id * max_num_blocks)
    block_table_base = batch_id * max_num_blocks

    # Process keys and values in this split
    if end_idx > start_idx:
        for idx in range(num_tiles):
            # Logical page index
            logical_page_idx = start_tile + idx
            curr_n = logical_page_idx * PAGE_SIZE

            # === PAGED INDIRECTION with latency hints ===
            # Look up physical page from block_tables using ct.gather
            block_table_idx = ct.full((1,), block_table_base + logical_page_idx, dtype=ct.int32)
            physical_page_tile = ct.gather(block_tables, block_table_idx, padding_value=0, latency=2)
            
            # Compute base offset for this page in flattened K/V cache
            # K/V cache layout: [num_pages, num_kv_heads, page_size, head_dim]
            # Linear index = page * stride_page + head * stride_head + seq * stride_seq + dim
            page_base_offset = physical_page_tile * stride_kv_page + head_id * stride_kv_head
            
            # Build indices for gathering K: [PAGE_SIZE, HEAD_DIM]
            # For each (seq_pos, dim) in the page, compute linear index
            k_indices = page_base_offset + offs_n[:, None] * stride_kv_seq + offs_d[None, :]
            k = ct.gather(K_cache, k_indices, padding_value=0.0, latency=2)
            k = ct.astype(k, q.dtype)

            # Compute qk
            qk = ct.matmul(k, q)

            # Boundary masking
            if curr_n + PAGE_SIZE > max_seq_len:
                mask = ct.less(ct.add(curr_n, offs_n[:, None]), max_seq_len)
                qk = ct.where(mask, qk, -1.0e6)

            # Compute softmax statistics
            qk_scaled = ct.mul(qk, qk_scale)
            m_ij = ct.maximum(m_i, ct.max(qk_scaled, 0))
            qk = ct.sub(qk_scaled, m_ij[None, :])
            p = ct.exp2(qk)

            # Update m_i and l_i
            alpha = ct.exp2(ct.sub(m_i, m_ij))
            l_i = ct.add(ct.mul(l_i, alpha[None, :]), p)

            # Update output accumulator
            acc = ct.mul(acc, alpha[None, :])

            # Build indices for gathering V: [PAGE_SIZE, HEAD_DIM]
            v_indices = page_base_offset + offs_n[:, None] * stride_kv_seq + offs_d[None, :]
            v = ct.gather(V_cache, v_indices, padding_value=0.0, latency=4)
            v = ct.astype(v, q.dtype)
            v = ct.transpose(v)  # (HEAD_DIM, PAGE_SIZE)
            p = ct.astype(p, q.dtype)
            acc = ct.mma(v, p, acc=acc)

            # Update m_i
            m_i = m_ij

    l = ct.sum(l_i, 0)
    acc = ct.truediv(
        acc, l[None, :], flush_to_zero=True, rounding_mode=RMd.APPROX
    )
    acc = ct.astype(acc, ct.float32)
    acc = ct.transpose(acc)
    acc = ct.astype(acc, Att_Out.dtype)
    l = ct.add(m_i, ct.log2(l))

    # Store attention output
    acc_reshaped = ct.reshape(
        acc, (1, 1, QUERY_GROUP_TILE_SIZE, 1, HEAD_DIM)
    )

    if NUM_Q_HEAD_PER_KV == QUERY_GROUP_TILE_SIZE:
        ct.store(
            Att_Out,
            index=(batch_id, head_id, 0, tile_id, 0),
            tile=acc_reshaped,
            order=(0, 1, 2, 3, 4),
            allow_tma=True,
        )
    else:
        idx_q_offset = ct.arange(QUERY_GROUP_TILE_SIZE, dtype=ct.int32)[:, None]
        idx_dim = ct.arange(HEAD_DIM, dtype=ct.int32)[None, :]
        ct.scatter(
            Att_Out,
            (batch_id, head_id, idx_q_offset, tile_id, idx_dim),
            acc,
            check_bounds=True,
            latency=1,
        )

    # Store log sum exp
    idx_lse_q_offset = ct.arange(QUERY_GROUP_TILE_SIZE, dtype=ct.int32)
    ct.scatter(
        LSE_Out,
        (batch_id, head_id, idx_lse_q_offset, tile_id),
        l,
        check_bounds=True,
        latency=1,
    )


class _attention_decode_paged(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K_cache, V_cache, block_tables, seq_lens, softmax_scale, max_seq_len=None):
        """
        Paged KV cache attention decode.
        
        Args:
            Q: Query tensor [batch_size, num_q_heads, 1, head_dim]
            K_cache: Paged key cache [num_pages, num_kv_heads, page_size, head_dim]
            V_cache: Paged value cache [num_pages, num_kv_heads, page_size, head_dim]
            block_tables: Page table [batch_size, max_num_blocks]
            seq_lens: Sequence lengths [batch_size]
            softmax_scale: Scale factor for attention
            max_seq_len: Optional max sequence length (computed from seq_lens if not provided)
            
        Returns:
            O: Output tensor [batch_size, num_q_heads, 1, head_dim]
        """
        batch_size, num_q_heads = Q.shape[0], Q.shape[1]
        num_kv_heads = K_cache.shape[1]
        page_size = K_cache.shape[2]
        head_dim = K_cache.shape[3]
        max_num_blocks = block_tables.shape[1]
        
        # PAGE_SIZE == TILE_N constraint
        PAGE_SIZE = page_size
        assert PAGE_SIZE == next_power_of_2(PAGE_SIZE), "page_size must be power of 2"
        
        # Get max sequence length for this batch
        if max_seq_len is None:
            max_seq_len = seq_lens.max().item()
        
        # Reshape Q
        Q = Q.view(batch_size, num_q_heads, head_dim)
        
        # Flatten block_tables for kernel access via ct.gather
        block_tables_flat = block_tables.contiguous().view(-1)
        
        # Flatten K/V cache and compute strides for gather-based access
        # K_cache shape: [num_pages, num_kv_heads, page_size, head_dim]
        K_cache = K_cache.contiguous()
        V_cache = V_cache.contiguous()
        stride_kv_page = K_cache.stride(0)  # stride for page dimension
        stride_kv_head = K_cache.stride(1)  # stride for head dimension
        stride_kv_seq = K_cache.stride(2)   # stride for seq (within page) dimension
        K_cache_flat = K_cache.view(-1)
        V_cache_flat = V_cache.view(-1)
        
        # Calculate splits (similar to non-paged version)
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        NUM_KV_SPLITS = NUM_SMS // (batch_size * num_kv_heads)
        TILE_SIZE = max(
            PAGE_SIZE,
            next_power_of_2((max_seq_len + NUM_KV_SPLITS - 1) // NUM_KV_SPLITS),
        )
        NUM_KV_SPLITS = (max_seq_len + TILE_SIZE - 1) // TILE_SIZE
        
        assert TILE_SIZE == next_power_of_2(TILE_SIZE)
        
        # Allocate intermediate results
        device = Q.device
        Att_Mid_Out = torch.empty(
            (batch_size, num_q_heads, NUM_KV_SPLITS, head_dim),
            device=device,
            dtype=Q.dtype,
        )
        LSE_Out = torch.empty(
            (batch_size, num_q_heads, NUM_KV_SPLITS),
            device=device,
            dtype=torch.float32,
        )
        
        O = torch.empty_like(Q)
        
        # GQA parameters
        assert num_q_heads % num_kv_heads == 0
        num_q_head_per_kv = num_q_heads // num_kv_heads
        query_group_tile_size = max(8, next_power_of_2(num_q_head_per_kv))
        
        # Create grouped views
        Att_Mid_Out_5D = Att_Mid_Out.view(
            batch_size,
            num_kv_heads,
            num_q_head_per_kv,
            NUM_KV_SPLITS,
            head_dim,
        )
        LSE_Out_4D = LSE_Out.view(
            batch_size,
            num_kv_heads,
            num_q_head_per_kv,
            NUM_KV_SPLITS,
        )
        
        # Calculate strides
        stride_mid_ob, stride_mid_oh, stride_mid_os = (
            Att_Mid_Out.stride(0),
            Att_Mid_Out.stride(1),
            Att_Mid_Out.stride(2),
        )
        stride_mid_lseb, stride_mid_lsem = (
            LSE_Out.stride(0),
            LSE_Out.stride(1),
        )
        
        HEAD_DIM = next_power_of_2(head_dim)
        
        # Reshape Q for grouped query attention
        Q_grouped = Q.view(
            batch_size,
            num_q_heads // num_q_head_per_kv,
            num_q_head_per_kv,
            head_dim,
        )
        
        # Launch kernel
        grid = (batch_size, num_kv_heads, NUM_KV_SPLITS)
        
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            attention_decode_kernel_paged,
            (
                Q_grouped,
                K_cache_flat,
                V_cache_flat,
                block_tables_flat,
                Att_Mid_Out_5D,
                LSE_Out_4D,
                softmax_scale,
                stride_mid_ob,
                stride_mid_oh,
                stride_mid_os,
                stride_mid_lseb,
                stride_mid_lsem,
                max_num_blocks,
                max_seq_len,
                stride_kv_page,
                stride_kv_head,
                stride_kv_seq,
                batch_size,
                num_q_heads,
                num_kv_heads,
                HEAD_DIM,
                PAGE_SIZE,
                TILE_SIZE,
                num_q_head_per_kv,
                query_group_tile_size,
                NUM_KV_SPLITS,
            ),
        )
        
        # Reduce splitk results
        splitk_reduce(Att_Mid_Out, LSE_Out, O, max_seq_len)
        
        return O.view(batch_size, num_q_heads, 1, head_dim)

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("Paged attention backward not implemented")


attention_decode_paged = _attention_decode_paged.apply


@register_impl("fmha_decode_paged", backend="cutile")
def fmha_decode_paged(q, k_cache, v_cache, block_tables, seq_lens, sm_scale=None, max_seq_len=None, **kwargs):
    """
    Paged KV cache flash decode attention.
    
    Args:
        q: Query tensor [batch_size, num_q_heads, 1, head_dim]
        k_cache: Paged key cache [num_pages, num_kv_heads, page_size, head_dim]
        v_cache: Paged value cache [num_pages, num_kv_heads, page_size, head_dim]
        block_tables: Page table mapping [batch_size, max_num_blocks]
        seq_lens: Actual sequence length per batch [batch_size]
        sm_scale: Softmax scale (default: 1/sqrt(head_dim))
        max_seq_len: Optional max sequence length (for CUDA graph compatibility)
    
    Returns:
        Output tensor [batch_size, num_q_heads, 1, head_dim]
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))
    return attention_decode_paged(q, k_cache, v_cache, block_tables, seq_lens, sm_scale, max_seq_len)
