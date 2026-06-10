# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...gemm.iq_quant.iq_tables import get_iq_table_tensors
from ...gemm.utils import (
    GGML_TYPE_IQ4_XS,
    load_f16_from_u8,
    load_u16_from_u8,
)
from ..utils import (
    load_moe_token_info,
    load_moe_x_chunk,
    run_triton_fused_moe_kernel,
)


@triton.jit
def iq4_xs_moe_kernel(
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
    values_ptr,
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
    offs_nibble = tl.arange(0, 16)
    n_mask = offs_n < n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_row_ptrs = w_u8_ptr + expert * stride_we + offs_n * stride_wn

    for kb in range(0, num_k_blocks):
        block_ptrs = w_row_ptrs + kb * 136
        scales_h = load_u16_from_u8(block_ptrs + 2, n_mask)

        for ib in range(8):
            packed = tl.load(
                block_ptrs[:, None] + 8 + 16 * ib + offs_nibble[None, :],
                mask=n_mask[:, None],
                other=0,
            )
            x1 = load_moe_x_chunk(
                x_ptr,
                stride_xm,
                stride_xk,
                offs_token,
                token_mask,
                kb * 256 + 32 * ib,
                CHUNK=16,
            )
            x2 = load_moe_x_chunk(
                x_ptr,
                stride_xm,
                stride_xk,
                offs_token,
                token_mask,
                kb * 256 + 32 * ib + 16,
                CHUNK=16,
            )
            x_dtype = x1.dtype
            d = load_f16_from_u8(block_ptrs + 0, n_mask).to(x_dtype)
            scales_l = tl.load(block_ptrs + 4 + (ib // 2), mask=n_mask, other=0).to(
                tl.int32
            )
            scale = (
                (
                    ((scales_l >> (4 * (ib % 2))) & 0x0F)
                    | (((scales_h.to(tl.int32) >> (2 * ib)) & 0x03) << 4)
                ).to(tl.int16)
                - 32
            ).to(x_dtype)
            low = tl.load(values_ptr + (packed & 0x0F).to(tl.int32)).to(x_dtype)
            high = tl.load(values_ptr + ((packed >> 4) & 0x0F).to(tl.int32)).to(x_dtype)
            dscale = (d * scale).to(x_dtype)
            q1 = (low * dscale[:, None]).to(x_dtype)
            q2 = (high * dscale[:, None]).to(x_dtype)
            acc = tl.dot(x1, tl.trans(q1), acc=acc)
            acc = tl.dot(x2, tl.trans(q2), acc=acc)

    y_ptrs = y_ptr + offs_output[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = token_mask[:, None] & n_mask[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


def ggml_moe_iq4_xs_triton(
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
        iq4_xs_moe_kernel,
        W,
        X,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        row,
        top_k,
        tokens,
        GGML_TYPE_IQ4_XS,
        extra_args=(tables["kvalues_iq4nl"],),
    )
