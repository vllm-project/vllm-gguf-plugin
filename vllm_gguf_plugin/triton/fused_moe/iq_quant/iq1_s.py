# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...gemm.iq_quant.iq_tables import get_iq_table_tensors
from ...gemm.utils import (
    GGML_TYPE_IQ1_S,
    load_f16_from_u8,
    load_u16_from_u8,
)
from ..utils import (
    load_moe_token_info,
    load_moe_x_chunk,
    run_triton_fused_moe_kernel,
)


@triton.jit
def iq1_s_moe_kernel(
    x_ptr,
    w_u8_ptr,
    y_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    num_valid_tokens,
    top_k,
    n,
    num_k_blocks,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    grid_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_BLOCKS: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_output, offs_token, token_mask = load_moe_token_info(
        sorted_token_ids_ptr, pid_m, top_k, num_valid_tokens, BLOCK_M=BLOCK_M
    )
    expert = tl.load(expert_ids_ptr + pid_m)
    if expert < 0:
        return

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_8 = tl.arange(0, 8)
    n_mask = offs_n < n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_row_ptrs = w_u8_ptr + expert * stride_we + offs_n * stride_wn

    for kb in range(0, num_k_blocks):
        block_ptrs = w_row_ptrs + kb * 50

        for ib in range(8):
            qh = load_u16_from_u8(block_ptrs + 34 + 2 * ib, n_mask)
            delta_num = tl.where((qh & 0x8000) != 0, -9, -7).to(tl.int16)
            for il in range(4):
                idx = tl.load(block_ptrs + 2 + 4 * ib + il, mask=n_mask, other=0).to(
                    tl.int32
                )
                idx = idx | ((((qh >> (3 * il)) & 0x7).to(tl.int32)) << 8)
                x_tile = load_moe_x_chunk(
                    x_ptr,
                    stride_xm,
                    stride_xk,
                    offs_token,
                    token_mask,
                    kb * 256 + 32 * ib + 8 * il,
                    CHUNK=8,
                )
                x_dtype = x_tile.dtype
                d = load_f16_from_u8(block_ptrs + 0, n_mask).to(x_dtype)
                scale = (d * (2 * ((qh >> 12) & 0x7).to(x_dtype) + 1.0) * 0.125).to(
                    x_dtype
                )
                grid = tl.load(
                    grid_ptr + idx[:, None] * 8 + offs_8[None, :],
                    mask=n_mask[:, None],
                    other=0,
                ).to(x_dtype)
                q_tile = (
                    (((grid.to(tl.int16) * 8) + delta_num[:, None]).to(x_dtype))
                    * scale[:, None]
                ).to(x_dtype)
                acc += tl.sum(x_tile[:, None, :] * q_tile[None, :, :], axis=2)

    y_ptrs = y_ptr + offs_output[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = token_mask[:, None] & n_mask[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


def ggml_moe_iq1_s_triton(
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    tables = get_iq_table_tensors(W.device)
    return run_triton_fused_moe_kernel(
        iq1_s_moe_kernel,
        W,
        X,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        row,
        top_k,
        tokens,
        GGML_TYPE_IQ1_S,
        extra_args=(tables["iq1s_grid"],),
    )
