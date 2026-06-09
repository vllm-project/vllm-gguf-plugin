# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..gemm.iq_quant.iq_tables import get_iq_table_tensors
from ..gemm.utils import (
    GGML_TYPE_IQ1_M,
    GGML_TYPE_IQ1_S,
    GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ2_XXS,
    GGML_TYPE_IQ2_XS,
    GGML_TYPE_IQ3_S,
    GGML_TYPE_IQ3_XXS,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_IQ4_XS,
    load_f16_from_u8,
    load_u16_from_u8,
    load_u32_from_u8,
)
from .utils import (
    load_moe_token_info,
    load_moe_x_chunk,
    load_moe_x_tile,
    run_triton_fused_moe_kernel,
)


@triton.jit
def iq2_xxs_moe_kernel(
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
    sign_ptr,
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
    sign_mask = (1 << offs_8).to(tl.uint8)
    n_mask = offs_n < n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_row_ptrs = w_u8_ptr + expert * stride_we + offs_n * stride_wn

    for kb in range(0, num_k_blocks):
        block_ptrs = w_row_ptrs + kb * 66

        for ib in range(8):
            q2_base = block_ptrs + 2 + 8 * ib
            aux32 = load_u32_from_u8(q2_base + 4, n_mask)
            d = load_f16_from_u8(block_ptrs + 0, n_mask)
            dscale = (d * ((aux32 >> 28).to(tl.float32) + 0.5) * 0.25).to(tl.float32)
            for il in range(4):
                grid_idx = tl.load(q2_base + il, mask=n_mask, other=0).to(tl.int32)
                signs = tl.load(sign_ptr + ((aux32 >> (7 * il)) & 127), mask=n_mask, other=0)
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
                grid = tl.load(
                    grid_ptr + grid_idx[:, None] * 8 + offs_8[None, :],
                    mask=n_mask[:, None],
                    other=0,
                ).to(x_dtype)
                sign = tl.where((signs[:, None] & sign_mask[None, :]) != 0, -1, 1).to(x_dtype)
                q_tile = (grid * sign * dscale.to(x_dtype)[:, None]).to(x_dtype)
                acc += tl.sum(x_tile[:, None, :] * q_tile[None, :, :], axis=2)

    y_ptrs = y_ptr + offs_output[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = token_mask[:, None] & n_mask[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


@triton.jit
def iq2_xs_moe_kernel(
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
    sign_ptr,
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
    sign_mask = (1 << offs_8).to(tl.uint8)
    n_mask = offs_n < n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_row_ptrs = w_u8_ptr + expert * stride_we + offs_n * stride_wn

    for kb in range(0, num_k_blocks):
        block_ptrs = w_row_ptrs + kb * 74

        for ib in range(8):
            scale_byte = tl.load(block_ptrs + 66 + ib, mask=n_mask, other=0)
            for il in range(4):
                q2 = load_u16_from_u8(block_ptrs + 2 + 2 * (4 * ib + il), n_mask).to(tl.int32)
                grid_idx = q2 & 0x1FF
                signs = tl.load(sign_ptr + (q2 >> 9), mask=n_mask, other=0)
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
                scale = (((scale_byte >> (4 * (il // 2))) & 0x0F).to(x_dtype) + 0.5) * 0.25
                grid = tl.load(
                    grid_ptr + grid_idx[:, None] * 8 + offs_8[None, :],
                    mask=n_mask[:, None],
                    other=0,
                ).to(x_dtype)
                sign = tl.where((signs[:, None] & sign_mask[None, :]) != 0, -1, 1).to(x_dtype)
                q_tile = (grid * sign * (d * scale).to(x_dtype)[:, None]).to(x_dtype)
                acc += tl.sum(x_tile[:, None, :] * q_tile[None, :, :], axis=2)

    y_ptrs = y_ptr + offs_output[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = token_mask[:, None] & n_mask[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


@triton.jit
def iq2_s_moe_kernel(
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
    sign_mask = (1 << offs_8).to(tl.uint8)
    n_mask = offs_n < n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_row_ptrs = w_u8_ptr + expert * stride_we + offs_n * stride_wn

    for kb in range(0, num_k_blocks):
        block_ptrs = w_row_ptrs + kb * 82

        for ib in range(8):
            qh = tl.load(block_ptrs + 66 + ib, mask=n_mask, other=0).to(tl.int32)
            scale_byte = tl.load(block_ptrs + 74 + ib, mask=n_mask, other=0)
            for il in range(4):
                grid_idx = tl.load(block_ptrs + 2 + 4 * ib + il, mask=n_mask, other=0).to(tl.int32)
                grid_idx = grid_idx | ((qh << (8 - 2 * il)) & 0x300)
                signs = tl.load(block_ptrs + 34 + 4 * ib + il, mask=n_mask, other=0)
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
                scale = (((scale_byte >> (4 * (il // 2))) & 0x0F).to(x_dtype) + 0.5) * 0.25
                grid = tl.load(
                    grid_ptr + grid_idx[:, None] * 8 + offs_8[None, :],
                    mask=n_mask[:, None],
                    other=0,
                ).to(x_dtype)
                sign = tl.where((signs[:, None] & sign_mask[None, :]) != 0, -1, 1).to(x_dtype)
                q_tile = (grid * sign * (d * scale).to(x_dtype)[:, None]).to(x_dtype)
                acc += tl.sum(x_tile[:, None, :] * q_tile[None, :, :], axis=2)

    y_ptrs = y_ptr + offs_output[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = token_mask[:, None] & n_mask[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


@triton.jit
def iq3_xxs_moe_kernel(
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
    sign_ptr,
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
    offs_4 = tl.arange(0, 4)
    sign_mask_lo = (1 << offs_4).to(tl.uint8)
    sign_mask_hi = (1 << (offs_4 + 4)).to(tl.uint8)
    n_mask = offs_n < n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_row_ptrs = w_u8_ptr + expert * stride_we + offs_n * stride_wn

    for kb in range(0, num_k_blocks):
        block_ptrs = w_row_ptrs + kb * 98

        for ib in range(8):
            aux32 = load_u32_from_u8(block_ptrs + 66 + 4 * ib, n_mask)
            for il in range(4):
                signs = tl.load(sign_ptr + ((aux32 >> (7 * il)) & 127), mask=n_mask, other=0)
                idx1 = tl.load(block_ptrs + 2 + 8 * ib + 2 * il + 0, mask=n_mask, other=0).to(tl.int32)
                idx2 = tl.load(block_ptrs + 2 + 8 * ib + 2 * il + 1, mask=n_mask, other=0).to(tl.int32)
                x1 = load_moe_x_chunk(
                    x_ptr,
                    stride_xm,
                    stride_xk,
                    offs_token,
                    token_mask,
                    kb * 256 + 32 * ib + 8 * il,
                    CHUNK=4,
                )
                x2 = load_moe_x_chunk(
                    x_ptr,
                    stride_xm,
                    stride_xk,
                    offs_token,
                    token_mask,
                    kb * 256 + 32 * ib + 8 * il + 4,
                    CHUNK=4,
                )
                x_dtype = x1.dtype
                d = load_f16_from_u8(block_ptrs + 0, n_mask).to(x_dtype)
                dscale = (d * ((aux32 >> 28).to(x_dtype) + 0.5) * 0.5).to(x_dtype)
                grid1 = tl.load(
                    grid_ptr + idx1[:, None] * 4 + offs_4[None, :],
                    mask=n_mask[:, None],
                    other=0,
                ).to(x_dtype)
                grid2 = tl.load(
                    grid_ptr + idx2[:, None] * 4 + offs_4[None, :],
                    mask=n_mask[:, None],
                    other=0,
                ).to(x_dtype)
                q1 = (
                    grid1
                    * tl.where((signs[:, None] & sign_mask_lo[None, :]) != 0, -1, 1).to(x_dtype)
                    * dscale[:, None]
                ).to(x_dtype)
                q2 = (
                    grid2
                    * tl.where((signs[:, None] & sign_mask_hi[None, :]) != 0, -1, 1).to(x_dtype)
                    * dscale[:, None]
                ).to(x_dtype)
                acc += tl.sum(x1[:, None, :] * q1[None, :, :], axis=2)
                acc += tl.sum(x2[:, None, :] * q2[None, :, :], axis=2)

    y_ptrs = y_ptr + offs_output[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = token_mask[:, None] & n_mask[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


@triton.jit
def iq3_s_moe_kernel(
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
    offs_4 = tl.arange(0, 4)
    sign_mask_lo = (1 << offs_4).to(tl.uint8)
    sign_mask_hi = (1 << (offs_4 + 4)).to(tl.uint8)
    n_mask = offs_n < n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_row_ptrs = w_u8_ptr + expert * stride_we + offs_n * stride_wn

    for kb in range(0, num_k_blocks):
        block_ptrs = w_row_ptrs + kb * 110

        for ib in range(8):
            qh = tl.load(block_ptrs + 66 + ib, mask=n_mask, other=0).to(tl.int32)
            scale_byte = tl.load(block_ptrs + 106 + (ib // 2), mask=n_mask, other=0)
            for il in range(4):
                signs = tl.load(block_ptrs + 74 + 4 * ib + il, mask=n_mask, other=0)
                qs_base = block_ptrs + 2 + 8 * ib
                idx1 = tl.load(qs_base + 2 * il + 0, mask=n_mask, other=0).to(tl.int32) | (
                    (qh << (8 - 2 * il)) & 0x100
                )
                idx2 = tl.load(qs_base + 2 * il + 1, mask=n_mask, other=0).to(tl.int32) | (
                    (qh << (7 - 2 * il)) & 0x100
                )
                x1 = load_moe_x_chunk(
                    x_ptr,
                    stride_xm,
                    stride_xk,
                    offs_token,
                    token_mask,
                    kb * 256 + 32 * ib + 8 * il,
                    CHUNK=4,
                )
                x2 = load_moe_x_chunk(
                    x_ptr,
                    stride_xm,
                    stride_xk,
                    offs_token,
                    token_mask,
                    kb * 256 + 32 * ib + 8 * il + 4,
                    CHUNK=4,
                )
                x_dtype = x1.dtype
                d = load_f16_from_u8(block_ptrs + 0, n_mask).to(x_dtype)
                dscale = (
                    d * (1 + 2 * ((scale_byte >> (4 * (ib % 2))) & 0x0F).to(x_dtype))
                ).to(x_dtype)
                grid1 = tl.load(
                    grid_ptr + idx1[:, None] * 4 + offs_4[None, :],
                    mask=n_mask[:, None],
                    other=0,
                ).to(x_dtype)
                grid2 = tl.load(
                    grid_ptr + idx2[:, None] * 4 + offs_4[None, :],
                    mask=n_mask[:, None],
                    other=0,
                ).to(x_dtype)
                q1 = (
                    grid1
                    * tl.where((signs[:, None] & sign_mask_lo[None, :]) != 0, -1, 1).to(x_dtype)
                    * dscale[:, None]
                ).to(x_dtype)
                q2 = (
                    grid2
                    * tl.where((signs[:, None] & sign_mask_hi[None, :]) != 0, -1, 1).to(x_dtype)
                    * dscale[:, None]
                ).to(x_dtype)
                acc += tl.sum(x1[:, None, :] * q1[None, :, :], axis=2)
                acc += tl.sum(x2[:, None, :] * q2[None, :, :], axis=2)

    y_ptrs = y_ptr + offs_output[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = token_mask[:, None] & n_mask[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


@triton.jit
def iq4_nl_moe_kernel(
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
    offs_kb = tl.arange(0, BLOCK_K_BLOCKS)
    offs_nibble = tl.arange(0, 16)
    n_mask = offs_n < n

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
        block_ptrs = w_row_ptrs + cur_kb[None, :] * 18
        scale_mask = n_mask[:, None] & kb_mask[None, :]
        d = load_f16_from_u8(block_ptrs + 0, scale_mask).to(x_dtype)
        packed = tl.load(
            block_ptrs[:, :, None] + 2 + offs_nibble[None, None, :],
            mask=n_mask[:, None, None] & kb_mask[None, :, None],
            other=0,
        )
        low = tl.load(values_ptr + (packed & 0x0F).to(tl.int32)).to(x_dtype) * d[:, :, None]
        high = tl.load(values_ptr + ((packed >> 4) & 0x0F).to(tl.int32)).to(x_dtype) * d[:, :, None]
        q_tile = tl.reshape(tl.join(low, high), (BLOCK_N, BLOCK_K_BLOCKS * 32))
        acc = tl.dot(x_tile, tl.trans(q_tile), acc=acc)

    y_ptrs = y_ptr + offs_output[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = token_mask[:, None] & n_mask[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


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
            scales_l = tl.load(block_ptrs + 4 + (ib // 2), mask=n_mask, other=0).to(tl.int32)
            scale = (
                (
                    (
                        (scales_l >> (4 * (ib % 2)))
                        & 0x0F
                    )
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
                idx = tl.load(block_ptrs + 2 + 4 * ib + il, mask=n_mask, other=0).to(tl.int32)
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
                scale = (d * (2 * ((qh >> 12) & 0x7).to(x_dtype) + 1.0) * 0.125).to(x_dtype)
                grid = tl.load(
                    grid_ptr + idx[:, None] * 8 + offs_8[None, :],
                    mask=n_mask[:, None],
                    other=0,
                ).to(x_dtype)
                q_tile = (
                    (((grid.to(tl.int16) * 8) + delta_num[:, None]).to(x_dtype)) * scale[:, None]
                ).to(x_dtype)
                acc += tl.sum(x_tile[:, None, :] * q_tile[None, :, :], axis=2)

    y_ptrs = y_ptr + offs_output[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = token_mask[:, None] & n_mask[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


@triton.jit
def iq1_m_moe_kernel(
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
        block_ptrs = w_row_ptrs + kb * 56
        sc0 = load_u16_from_u8(block_ptrs + 48 + 0, n_mask)
        sc1 = load_u16_from_u8(block_ptrs + 48 + 2, n_mask)
        sc2 = load_u16_from_u8(block_ptrs + 48 + 4, n_mask)
        sc3 = load_u16_from_u8(block_ptrs + 48 + 6, n_mask)
        base_bits = (sc0 >> 12) | ((sc1 >> 8) & 0x00F0) | ((sc2 >> 4) & 0x0F00) | (sc3 & 0xF000)
        for ib in range(8):
            for il in range(4):
                qh = tl.load(block_ptrs + 32 + 2 * ib + (il // 2), mask=n_mask, other=0).to(tl.int32)
                idx = tl.load(block_ptrs + 4 * ib + il, mask=n_mask, other=0).to(tl.int32)
                idx = idx | (((qh >> (4 * (il % 2))) & 0x07) << 8)
                delta_num = tl.where((qh & (0x08 << (4 * (il % 2)))) != 0, -9, -7).to(tl.int16)
                ib16 = 2 * ib + (il // 2)
                sc_sel = sc0 if ib16 // 4 == 0 else sc1 if ib16 // 4 == 1 else sc2 if ib16 // 4 == 2 else sc3
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
                base = tl.cast(base_bits.to(tl.uint16), tl.float16, bitcast=True).to(x_dtype)
                scale = (
                    base * (2 * (((sc_sel >> (3 * (ib16 % 4))) & 0x07).to(x_dtype)) + 1.0) * 0.125
                ).to(x_dtype)
                grid = tl.load(
                    grid_ptr + idx[:, None] * 8 + offs_8[None, :],
                    mask=n_mask[:, None],
                    other=0,
                ).to(x_dtype)
                q_tile = (
                    (((grid.to(tl.int16) * 8) + delta_num[:, None]).to(x_dtype)) * scale[:, None]
                ).to(x_dtype)
                acc += tl.sum(x_tile[:, None, :] * q_tile[None, :, :], axis=2)

    y_ptrs = y_ptr + offs_output[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = token_mask[:, None] & n_mask[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


def ggml_moe_iq2_xs_triton(
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
        iq2_xs_moe_kernel,
        W,
        X,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        row,
        top_k,
        tokens,
        GGML_TYPE_IQ2_XS,
        extra_args=(tables["iq2xs_grid"], tables["ksigns_iq2xs"]),
    )


def ggml_moe_iq2_xxs_triton(
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
        iq2_xxs_moe_kernel,
        W,
        X,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        row,
        top_k,
        tokens,
        GGML_TYPE_IQ2_XXS,
        extra_args=(tables["iq2xxs_grid"], tables["ksigns_iq2xs"]),
    )


def ggml_moe_iq2_s_triton(
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
        iq2_s_moe_kernel,
        W,
        X,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        row,
        top_k,
        tokens,
        GGML_TYPE_IQ2_S,
        extra_args=(tables["iq2s_grid"],),
    )


def ggml_moe_iq3_xxs_triton(
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
        iq3_xxs_moe_kernel,
        W,
        X,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        row,
        top_k,
        tokens,
        GGML_TYPE_IQ3_XXS,
        extra_args=(tables["iq3xxs_grid"], tables["ksigns_iq2xs"]),
    )


def ggml_moe_iq3_s_triton(
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
        iq3_s_moe_kernel,
        W,
        X,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        row,
        top_k,
        tokens,
        GGML_TYPE_IQ3_S,
        extra_args=(tables["iq3s_grid"],),
    )


def ggml_moe_iq4_nl_triton(
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
        iq4_nl_moe_kernel,
        W,
        X,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        row,
        top_k,
        tokens,
        GGML_TYPE_IQ4_NL,
        extra_args=(tables["kvalues_iq4nl"],),
    )


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


def ggml_moe_iq1_m_triton(
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
        iq1_m_moe_kernel,
        W,
        X,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        row,
        top_k,
        tokens,
        GGML_TYPE_IQ1_M,
        extra_args=(tables["iq1s_grid"],),
    )
