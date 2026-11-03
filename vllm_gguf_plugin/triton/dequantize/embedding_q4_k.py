"""Fused Q4_K embedding row gather and dequantization."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..gemm.utils import load_f16_from_u8
from .utils import load_scale_min_k4_vector


@triton.jit
def q4_k_embedding_kernel(
    w_ptr,
    ids_ptr,
    y_ptr,
    row_stride,
    vocab_size,
    HIDDEN_SIZE: tl.constexpr,
    BLOCKS_PER_ROW: tl.constexpr,
    ROUND_TO_HALF: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    program = tl.program_id(0)
    token_idx = program // BLOCKS_PER_ROW
    block_idx = program % BLOCKS_PER_ROW
    token_id = tl.load(ids_ptr + token_idx)
    valid_id = (token_id >= 0) & (token_id < vocab_size)

    # Match index_select's out-of-range failure while keeping the fused loads safe.
    if not valid_id:
        tl.inline_asm_elementwise(
            "trap; mov.u32 $0, 0;",
            constraints="=r",
            args=[],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )

    safe_id = tl.minimum(tl.maximum(token_id, 0), vocab_size - 1)
    block_ptr = w_ptr + safe_id * row_stride + block_idx * 144

    pos = tl.arange(0, BLOCK_SIZE)
    group = pos // 64
    rem = pos % 64
    part = rem // 32
    q_idx = rem % 32
    j = 2 * group + part

    scale_q, min_q = load_scale_min_k4_vector(block_ptr + 4, j, valid_id)
    packed = tl.load(
        block_ptr + 16 + 32 * group + q_idx,
        mask=valid_id,
        other=0,
    )
    q = tl.where(part == 0, packed & 0x0F, packed >> 4)
    dall = load_f16_from_u8(block_ptr + 0, valid_id)
    dmin = load_f16_from_u8(block_ptr + 2, valid_id)
    if ROUND_TO_HALF:
        d_scaled = (dall * scale_q.to(tl.float16)).to(tl.float16)
        min_scaled = (dmin * min_q.to(tl.float16)).to(tl.float16)
        quant_scaled = (q.to(tl.float16) * d_scaled).to(tl.float16)
        value = (quant_scaled - min_scaled).to(tl.float16).to(tl.float32)
    else:
        dall = dall.to(tl.float32)
        dmin = dmin.to(tl.float32)
        value = q.to(tl.float32) * scale_q.to(tl.float32) * dall
        value = value - min_q.to(tl.float32) * dmin
    value = tl.where(valid_id, value, 0.0)

    output_offset = token_idx * HIDDEN_SIZE + block_idx * BLOCK_SIZE + pos
    tl.store(y_ptr + output_offset, value, mask=pos < BLOCK_SIZE)


def ggml_embedding_q4_k_triton(
    weight: torch.Tensor,
    token_ids: torch.Tensor,
    hidden_size: int,
    dtype: torch.dtype | None = None,
    round_to_half: bool = True,
) -> torch.Tensor:
    """Gather and decode Q4_K rows without materializing packed selected rows."""
    if weight.device.type != "cuda" or token_ids.device.type != "cuda":
        raise ValueError("Q4_K embedding lookup requires CUDA tensors")
    if weight.dtype is not torch.uint8:
        raise TypeError(f"Quantized weights must be torch.uint8, got {weight.dtype}")
    if not weight.is_contiguous():
        raise ValueError("Q4_K embedding weights must be contiguous")
    if weight.ndim != 2:
        raise ValueError("Q4_K embedding weights must be a 2D tensor")
    if hidden_size <= 0 or hidden_size % 256:
        raise ValueError(
            "Q4_K embedding hidden size must be a positive multiple of 256"
        )
    row_bytes = hidden_size // 256 * 144
    if weight.shape[1] != row_bytes:
        raise ValueError(
            f"Q4_K rows have {weight.shape[1]} bytes, expected {row_bytes} "
            f"for hidden_size={hidden_size}"
        )

    dtype = dtype or torch.float16
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"Unsupported Q4_K embedding output dtype: {dtype}")

    output = torch.empty(
        (*token_ids.shape, hidden_size), device=weight.device, dtype=dtype
    )
    token_count = token_ids.numel()
    if token_count == 0:
        return output

    blocks_per_row = hidden_size // 256
    q4_k_embedding_kernel[(token_count * blocks_per_row,)](
        weight,
        token_ids,
        output,
        row_bytes,
        weight.shape[0],
        HIDDEN_SIZE=hidden_size,
        BLOCKS_PER_ROW=blocks_per_row,
        ROUND_TO_HALF=round_to_half,
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return output


__all__ = ["ggml_embedding_q4_k_triton"]
