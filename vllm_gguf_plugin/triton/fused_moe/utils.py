# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..gemm.utils import (
    BLOCK_BYTES_BY_TYPE,
    BLOCK_QK_BY_TYPE,
    GGML_TYPE_IQ1_M,
    GGML_TYPE_IQ1_S,
    GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ2_XS,
    GGML_TYPE_IQ2_XXS,
    GGML_TYPE_IQ3_S,
    GGML_TYPE_IQ3_XXS,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_IQ4_XS,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q8_1,
    TRITON_NUM_STAGES,
    TRITON_NUM_WARPS,
    TRITON_SUPPORTED_ACTIVATION_DTYPES,
)

TRITON_FUSED_MOE_SUPPORTED_TYPES = frozenset(
    {
        GGML_TYPE_Q4_0,
        GGML_TYPE_Q4_1,
        GGML_TYPE_Q5_0,
        GGML_TYPE_Q5_1,
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q8_1,
        GGML_TYPE_Q2_K,
        GGML_TYPE_Q3_K,
        GGML_TYPE_Q4_K,
        GGML_TYPE_Q5_K,
        GGML_TYPE_Q6_K,
        GGML_TYPE_IQ1_M,
        GGML_TYPE_IQ1_S,
        GGML_TYPE_IQ2_S,
        GGML_TYPE_IQ2_XXS,
        GGML_TYPE_IQ2_XS,
        GGML_TYPE_IQ3_S,
        GGML_TYPE_IQ3_XXS,
        GGML_TYPE_IQ4_NL,
        GGML_TYPE_IQ4_XS,
    }
)

TRITON_FUSED_MOE_BLOCK_M = 4
TRITON_FUSED_MOE_BLOCK_N = 128
TRITON_FUSED_MOE_BLOCK_K_BLOCKS = 4

# Per-type BLOCK_M overrides for Triton MoE kernels.
TRITON_MOE_BLOCK_M_BY_TYPE: dict[int, int] = {
    GGML_TYPE_Q4_0: 8,
}


def get_triton_moe_block_m(quant_type: int) -> int:
    return TRITON_MOE_BLOCK_M_BY_TYPE.get(quant_type, TRITON_FUSED_MOE_BLOCK_M)


@triton.jit
def load_moe_token_info(
    sorted_token_ids_ptr, pid_m, top_k, num_valid_tokens, BLOCK_M: tl.constexpr
):
    offs_block = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_output = tl.load(sorted_token_ids_ptr + offs_block).to(tl.int64)
    token_mask = (offs_output >= 0) & (offs_output < num_valid_tokens)
    offs_token = tl.where(token_mask, offs_output // top_k, 0)
    return offs_output, offs_token, token_mask


@triton.jit
def load_moe_x_tile(
    x_ptr,
    num_k_blocks,
    stride_xm,
    stride_xk,
    offs_token,
    token_mask,
    kb_start,
    offs_kb,
    offs_nibble,
    BLOCK_M: tl.constexpr,
    BLOCK_K_BLOCKS: tl.constexpr,
):
    cur_kb = kb_start + offs_kb
    kb_mask = cur_kb < num_k_blocks
    x_row_ptrs = x_ptr + offs_token[:, None, None] * stride_xm
    x_k_low = cur_kb[None, :, None] * 32 + offs_nibble[None, None, :]
    x_k_high = x_k_low + 16
    x_even = tl.load(
        x_row_ptrs + x_k_low * stride_xk,
        mask=token_mask[:, None, None] & kb_mask[None, :, None],
        other=0.0,
    )
    x_odd = tl.load(
        x_row_ptrs + x_k_high * stride_xk,
        mask=token_mask[:, None, None] & kb_mask[None, :, None],
        other=0.0,
    )
    return (
        tl.reshape(tl.join(x_even, x_odd), (BLOCK_M, BLOCK_K_BLOCKS * 32)),
        cur_kb,
        kb_mask,
    )


@triton.jit
def load_moe_x_chunk(
    x_ptr,
    stride_xm,
    stride_xk,
    offs_token,
    token_mask,
    k_start,
    CHUNK: tl.constexpr,
):
    offs_k = k_start + tl.arange(0, CHUNK)
    return tl.load(
        x_ptr + offs_token[:, None] * stride_xm + offs_k[None, :] * stride_xk,
        mask=token_mask[:, None],
        other=0.0,
    )


def _validate_args(
    W: torch.Tensor,
    X: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    row: int,
    top_k: int,
    tokens: int,
    quant_type: int,
    block_m: int = TRITON_FUSED_MOE_BLOCK_M,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int
]:
    if quant_type not in TRITON_FUSED_MOE_SUPPORTED_TYPES:
        raise ValueError(f"Unsupported Triton fused MoE quant type: {quant_type}")
    if not all(
        t.is_cuda for t in (W, X, sorted_token_ids, expert_ids, num_tokens_post_padded)
    ):
        raise ValueError("Triton fused MoE kernels require CUDA tensors")
    if W.dtype is not torch.uint8:
        raise TypeError(f"Quantized expert weights must be torch.uint8, got {W.dtype}")
    if X.dtype not in TRITON_SUPPORTED_ACTIVATION_DTYPES:
        raise TypeError(
            "Triton fused MoE kernels support torch.float16, torch.bfloat16, and "
            f"torch.float32 activations, got {X.dtype}"
        )
    if X.dim() != 2:
        raise ValueError(f"X must be 2D, got {X.dim()}D")
    if W.dim() != 3:
        raise ValueError(f"W must be 3D [experts, rows, packed_cols], got {W.dim()}D")
    if row != W.shape[1]:
        raise ValueError(
            f"row must match W.shape[1], got row={row}, W.shape[1]={W.shape[1]}"
        )
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if X.shape[0] != tokens:
        raise ValueError(f"X.shape[0] must equal tokens, got {X.shape[0]} vs {tokens}")
    if num_tokens_post_padded.numel() != 1:
        raise ValueError(
            "num_tokens_post_padded must be a scalar tensor, "
            f"got shape {tuple(num_tokens_post_padded.shape)}"
        )

    block_bytes = BLOCK_BYTES_BY_TYPE[quant_type]
    if W.shape[2] % block_bytes != 0:
        raise ValueError(
            f"Invalid expert row width {W.shape[2]} for quant type {quant_type}: "
            f"must be divisible by {block_bytes}"
        )
    num_k_blocks = W.shape[2] // block_bytes
    hidden_size = num_k_blocks * BLOCK_QK_BY_TYPE[quant_type]
    if X.shape[1] != hidden_size:
        raise ValueError(
            f"X hidden size {X.shape[1]} does not match "
            f"quantized expert width {hidden_size}"
        )

    # Use tensor shapes (CPU-known) instead of .item() to avoid GPU→CPU sync.
    # This is required for CUDA graph capture compatibility.
    max_num_blocks = expert_ids.numel()
    expected_token_ids_len = max_num_blocks * block_m
    actual_token_ids_len = sorted_token_ids.numel()
    if actual_token_ids_len > expected_token_ids_len:
        raise ValueError(
            "sorted_token_ids has more elements than expert_ids * block_m: "
            f"sorted_token_ids.numel()={actual_token_ids_len}, "
            f"expert_ids.numel()*block_m={expected_token_ids_len}"
        )
    # Pad sorted_token_ids to match expert_ids.numel() * block_m if needed.
    # moe_align_block_size may not produce a perfect multiple for block_m > 4.
    if actual_token_ids_len < expected_token_ids_len:
        pad = torch.full(
            (expected_token_ids_len - actual_token_ids_len,),
            -1,
            dtype=sorted_token_ids.dtype,
            device=sorted_token_ids.device,
        )
        sorted_token_ids = torch.cat([sorted_token_ids, pad])

    return (
        W.contiguous(),
        X.contiguous(),
        sorted_token_ids.reshape(-1).contiguous(),
        expert_ids.reshape(-1).contiguous(),
        num_tokens_post_padded.contiguous(),
        num_k_blocks,
        tokens * top_k,
    )


def run_triton_fused_moe_kernel(
    kernel,
    W: torch.Tensor,
    X: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    row: int,
    top_k: int,
    tokens: int,
    quant_type: int,
    extra_args: tuple = (),
    block_m: int = TRITON_FUSED_MOE_BLOCK_M,
    block_n: int = TRITON_FUSED_MOE_BLOCK_N,
    block_k_blocks: int = TRITON_FUSED_MOE_BLOCK_K_BLOCKS,
    num_warps: int = TRITON_NUM_WARPS,
    num_stages: int = TRITON_NUM_STAGES,
) -> torch.Tensor:
    (
        W,
        X,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        num_k_blocks,
        num_valid_tokens,
    ) = _validate_args(
        W,
        X,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        row,
        top_k,
        tokens,
        quant_type,
        block_m,
    )

    out = torch.zeros((num_valid_tokens, row), device=X.device, dtype=X.dtype)

    # Use expert_ids.shape[0] (CPU-known) for grid sizing instead of
    # num_tokens_post_padded.item() to avoid GPU→CPU sync during CUDA graph
    # capture. The kernel safely handles extra blocks via expert < 0 checks
    # and token_mask.
    grid = (
        expert_ids.shape[0],
        triton.cdiv(row, block_n),
    )
    kernel[grid](
        X,
        W,
        out,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        num_valid_tokens,
        top_k,
        row,
        num_k_blocks,
        X.stride(0),
        X.stride(1),
        W.stride(0),
        W.stride(1),
        W.stride(2),
        out.stride(0),
        out.stride(1),
        *extra_args,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K_BLOCKS=block_k_blocks,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
