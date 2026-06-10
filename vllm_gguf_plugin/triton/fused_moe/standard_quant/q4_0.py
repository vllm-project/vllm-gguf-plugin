# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...gemm.utils import (
    GGML_TYPE_Q4_0,
)
from ..utils import (
    load_moe_token_info,
    load_moe_x_tile,
    run_triton_fused_moe_kernel,
    store_moe_output,
)


@triton.jit
def q4_0_moe_kernel(
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
    topk_weights_ptr,
    routed_top_k,
    FUSED_WEIGHTED_SUM: tl.constexpr,
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
    n_mask = offs_n < n
    offs_kb = tl.arange(0, BLOCK_K_BLOCKS)
    offs_byte = tl.arange(0, 16)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_packed_row_ptrs = (
        w_u8_ptr + expert * stride_we + offs_n[:, None, None] * stride_wn
    )
    w_block_row_ptrs = w_u8_ptr + expert * stride_we + offs_n[:, None] * stride_wn

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
            offs_byte,
            BLOCK_M=BLOCK_M,
            BLOCK_K_BLOCKS=BLOCK_K_BLOCKS,
        )
        x_dtype = x_tile.dtype

        scale_ptrs = w_block_row_ptrs + cur_kb[None, :] * 18
        scale_mask = n_mask[:, None] & kb_mask[None, :]
        scale_lo = tl.load(scale_ptrs + 0, mask=scale_mask, other=0)
        scale_hi = tl.load(scale_ptrs + 1, mask=scale_mask, other=0)
        scale_bits = scale_lo.to(tl.uint16) | (scale_hi.to(tl.uint16) << 8)
        scales = tl.cast(scale_bits, tl.float16, bitcast=True).to(x_dtype)

        packed_ptrs = (
            w_packed_row_ptrs
            + cur_kb[None, :, None] * 18
            + 2
            + offs_byte[None, None, :]
        )
        packed_mask = n_mask[:, None, None] & kb_mask[None, :, None]
        packed = tl.load(packed_ptrs, mask=packed_mask, other=0)

        low = ((packed & 0x0F).to(x_dtype) - 8.0) * scales[:, :, None]
        high = (((packed >> 4) & 0x0F).to(x_dtype) - 8.0) * scales[:, :, None]
        q_tile = tl.reshape(tl.join(low, high), (BLOCK_N, BLOCK_K_BLOCKS * 32))
        acc = tl.dot(x_tile, tl.trans(q_tile), acc=acc)

    store_moe_output(
        y_ptr,
        topk_weights_ptr,
        acc,
        offs_output,
        token_mask,
        offs_n,
        n_mask,
        stride_ym,
        stride_yn,
        routed_top_k,
        FUSED_WEIGHTED_SUM=FUSED_WEIGHTED_SUM,
    )


def ggml_moe_q4_0_triton(
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    row: int,
    top_k: int,
    tokens: int,
    topk_weights: torch.Tensor | None = None,
    routed_top_k: int = 1,
    fused_weighted_sum: bool = False,
) -> torch.Tensor:
    return run_triton_fused_moe_kernel(
        q4_0_moe_kernel,
        W,
        X,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        row,
        top_k,
        tokens,
        GGML_TYPE_Q4_0,
        topk_weights=topk_weights,
        routed_top_k=routed_top_k,
        fused_weighted_sum=fused_weighted_sum,
    )
