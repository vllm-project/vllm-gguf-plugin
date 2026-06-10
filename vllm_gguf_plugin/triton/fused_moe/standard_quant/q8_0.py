# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...gemm.utils import (
    GGML_TYPE_Q8_0,
    load_f16_from_u8,
)
from ..utils import (
    load_moe_token_info,
    load_moe_x_tile,
    run_triton_fused_moe_kernel,
)


@triton.jit
def q8_0_moe_kernel(
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
    offs_kb = tl.arange(0, BLOCK_K_BLOCKS)
    offs_nibble = tl.arange(0, 16)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_row_ptrs = w_u8_ptr + expert * stride_we + offs_n[:, None] * stride_wn

    for kb_start in range(0, num_k_blocks, BLOCK_K_BLOCKS):
        x_tile, cur_kb, kb_mask = load_moe_x_tile(
            x_ptr,
            num_k_blocks,
            stride_xm,
            stride_xk,
            offs_token,
            token_mask,
            kb_start,
            offs_kb,
            offs_nibble,
            BLOCK_M=BLOCK_M,
            BLOCK_K_BLOCKS=BLOCK_K_BLOCKS,
        )
        x_dtype = x_tile.dtype
        block_ptrs = w_row_ptrs + cur_kb[None, :] * 34
        scale_mask = (offs_n[:, None] < n) & kb_mask[None, :]
        d = load_f16_from_u8(block_ptrs + 0, scale_mask).to(x_dtype)
        q_low = tl.load(
            block_ptrs[:, :, None] + 2 + offs_nibble[None, None, :],
            mask=(offs_n[:, None, None] < n) & kb_mask[None, :, None],
            other=0,
        )
        q_high = tl.load(
            block_ptrs[:, :, None] + 18 + offs_nibble[None, None, :],
            mask=(offs_n[:, None, None] < n) & kb_mask[None, :, None],
            other=0,
        )
        q_tile = tl.reshape(
            tl.join(
                tl.cast(q_low, tl.int8, bitcast=True).to(x_dtype) * d[:, :, None],
                tl.cast(q_high, tl.int8, bitcast=True).to(x_dtype) * d[:, :, None],
            ),
            (BLOCK_N, BLOCK_K_BLOCKS * 32),
        )
        acc = tl.dot(x_tile, tl.trans(q_tile), acc=acc)

    y_ptrs = y_ptr + offs_output[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = token_mask[:, None] & (offs_n[None, :] < n)
    tl.store(y_ptrs, acc, mask=y_mask)


def ggml_moe_q8_0_triton(
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    return run_triton_fused_moe_kernel(
        q8_0_moe_kernel,
        W,
        X,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        row,
        top_k,
        tokens,
        GGML_TYPE_Q8_0,
    )
