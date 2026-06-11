# SPDX-License-Identifier: Apache-2.0

import os

import torch

from .triton.fused_moe.interface import ggml_moe_a8_triton
from .triton.fused_moe.utils import get_triton_moe_block_m
from .triton.gemm.interface import ggml_mul_mat_a8_triton

try:
    from torch.library import register_fake
except ImportError:
    from torch.library import impl_abstract as register_fake

# Backend selection: default to Triton, CUDA only when explicitly enabled
_USE_CUDA = os.environ.get("VLLM_GGUF_USE_CUDA", "0") == "1"

# Try importing CUDA extension
try:
    from . import _C_gguf  # noqa: F401

    _CUDA_AVAILABLE = True
except ImportError:
    _C_gguf = None
    _CUDA_AVAILABLE = False


# Effective CUDA usage: only when explicitly requested AND available
_CUDA_ENABLED = _USE_CUDA and _CUDA_AVAILABLE

# --- Fake implementations for CUDA custom ops (needed for torch.compile) ---

if (
    _CUDA_AVAILABLE
    and hasattr(torch.ops, "_C_gguf")
    and hasattr(torch.ops._C_gguf, "ggml_dequantize")
):

    @register_fake("_C_gguf::ggml_dequantize")
    def _ggml_dequantize_fake(
        W: torch.Tensor,
        quant_type: int,
        m: torch.SymInt,
        n: torch.SymInt,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return torch.empty((m, n), dtype=torch.float16, device=W.device)

    @register_fake("_C_gguf::ggml_mul_mat_vec_a8")
    def _ggml_mul_mat_vec_a8_fake(
        W: torch.Tensor,
        X: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
    ) -> torch.Tensor:
        return torch.empty((X.shape[0], row), dtype=X.dtype, device=W.device)

    @register_fake("_C_gguf::ggml_mul_mat_a8")
    def _ggml_mul_mat_a8_fake(
        W: torch.Tensor,
        X: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
    ) -> torch.Tensor:
        return torch.empty((X.size(0), row), dtype=X.dtype, device=W.device)

    @register_fake("_C_gguf::ggml_moe_a8")
    def _ggml_moe_a8_fake(
        X: torch.Tensor,
        W: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
        top_k: torch.SymInt,
        tokens: torch.SymInt,
    ) -> torch.Tensor:
        return torch.empty(
            (X.size(0) * top_k, row), dtype=torch.float16, device=W.device
        )


if (
    _CUDA_AVAILABLE
    and hasattr(torch.ops, "_C_gguf")
    and hasattr(torch.ops._C_gguf, "ggml_moe_a8_vec")
):

    @register_fake("_C_gguf::ggml_moe_a8_vec")
    def _ggml_moe_a8_vec_fake(
        X: torch.Tensor,
        W: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        quant_type: int,
        row: torch.SymInt,
        tokens: torch.SymInt,
    ) -> torch.Tensor:
        return torch.empty((X.size(0) * top_k, row), dtype=X.dtype, device=W.device)


# --- Public API ---


def ggml_dequantize(
    W: torch.Tensor, quant_type: int, m: int, n: int, dtype: torch.dtype | None
) -> torch.Tensor:
    return torch.ops._C_gguf.ggml_dequantize(W, quant_type, m, n, dtype)


def ggml_mul_mat_vec_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    if _CUDA_ENABLED:
        return torch.ops._C_gguf.ggml_mul_mat_vec_a8(W, X, quant_type, row)
    return ggml_mul_mat_a8_triton(W, X, quant_type, row)


def ggml_mul_mat_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    if _CUDA_ENABLED:
        return torch.ops._C_gguf.ggml_mul_mat_a8(W, X, quant_type, row)
    return ggml_mul_mat_a8_triton(W, X, quant_type, row)


def ggml_moe_a8(
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    quant_type: int,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    if _CUDA_ENABLED:
        return torch.ops._C_gguf.ggml_moe_a8(
            X,
            W,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            quant_type,
            row,
            top_k,
            tokens,
        )
    return ggml_moe_a8_triton(
        X,
        W,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        quant_type,
        row,
        top_k,
        tokens,
    )


def ggml_moe_a8_vec(
    X: torch.Tensor,
    W: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    quant_type: int,
    row: int,
    tokens: int,
) -> torch.Tensor:
    if _CUDA_ENABLED:
        return torch.ops._C_gguf.ggml_moe_a8_vec(
            X, W, topk_ids, top_k, quant_type, row, tokens
        )
    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

    E = W.shape[0]
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, get_triton_moe_block_m(quant_type), E
    )
    return ggml_moe_a8_triton(
        X,
        W,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        quant_type,
        row,
        top_k,
        tokens,
    )


def ggml_moe_get_block_size(quant_type: int) -> int:
    if _CUDA_ENABLED:
        return torch.ops._C_gguf.ggml_moe_get_block_size(quant_type)
    return get_triton_moe_block_m(quant_type)


def moe_sum(input: torch.Tensor, output: torch.Tensor) -> None:
    torch.ops._moe_C.moe_sum(input, output)
