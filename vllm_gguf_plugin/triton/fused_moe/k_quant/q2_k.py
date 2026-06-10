# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...gemm.utils import (
    GGML_TYPE_Q2_K,
    load_f16_from_u8,
)
from ..utils import (
    load_moe_token_info,
    load_moe_x_chunk,
    run_triton_fused_moe_kernel,
)


@triton.jit
def q2_k_moe_kernel(
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
    offs_q = tl.arange(0, 32)
    use_hi = offs_q >= 16
    n_mask = offs_n < n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_row_ptrs = w_u8_ptr + expert * stride_we + offs_n * stride_wn

    for kb in range(0, num_k_blocks):
        block_ptrs = w_row_ptrs + kb * 84

        for chunk in range(8):
            group = chunk // 4
            part = chunk % 4
            x_tile = load_moe_x_chunk(
                x_ptr,
                stride_xm,
                stride_xk,
                offs_token,
                token_mask,
                kb * 256 + 128 * group + 32 * part,
                CHUNK=32,
            )
            x_dtype = x_tile.dtype
            d = load_f16_from_u8(block_ptrs + 80, n_mask).to(x_dtype)
            dmin = load_f16_from_u8(block_ptrs + 82, n_mask).to(x_dtype)
            q_bytes = tl.load(
                block_ptrs[:, None] + 16 + 32 * group + offs_q[None, :],
                mask=n_mask[:, None],
                other=0,
            )
            scale0 = tl.load(
                block_ptrs + (8 * group + 2 * part + 0), mask=n_mask, other=0
            )
            scale1 = tl.load(
                block_ptrs + (8 * group + 2 * part + 1), mask=n_mask, other=0
            )
            scale_byte = tl.where(use_hi[None, :], scale1[:, None], scale0[:, None])
            scale = (scale_byte & 0x0F).to(x_dtype) * d[:, None]
            minv = (scale_byte >> 4).to(x_dtype) * dmin[:, None]
            q_tile = ((q_bytes >> (2 * part)) & 0x03).to(x_dtype) * scale - minv
            acc = tl.dot(x_tile, tl.trans(q_tile), acc=acc)

    y_ptrs = y_ptr + offs_output[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = token_mask[:, None] & n_mask[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


def ggml_moe_q2_k_triton(
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
        q2_k_moe_kernel,
        W,
        X,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        row,
        top_k,
        tokens,
        GGML_TYPE_Q2_K,
    )
