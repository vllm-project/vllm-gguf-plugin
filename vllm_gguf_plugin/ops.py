# SPDX-License-Identifier: Apache-2.0

import torch

from ._jit import ensure_gguf_cuda_ops_loaded

try:
    from torch.library import register_fake
except ImportError:
    from torch.library import impl_abstract as register_fake

_BASE_FAKE_OPS_REGISTERED = False
_VEC_FAKE_OP_REGISTERED = False


def _ggml_dequantize_fake(
    W: torch.Tensor,
    quant_type: int,
    m: torch.SymInt,
    n: torch.SymInt,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    del quant_type, dtype
    return torch.empty((m, n), dtype=torch.float16, device=W.device)


def _ggml_mul_mat_vec_a8_fake(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: torch.SymInt,
) -> torch.Tensor:
    del quant_type
    return torch.empty((X.shape[0], row), dtype=X.dtype, device=W.device)


def _ggml_mul_mat_a8_fake(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: torch.SymInt,
) -> torch.Tensor:
    del quant_type
    return torch.empty((X.size(0), row), dtype=X.dtype, device=W.device)


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
    del sorted_token_ids, expert_ids, num_tokens_post_padded, quant_type, tokens
    return torch.empty((X.size(0) * top_k, row), dtype=torch.float16, device=W.device)


def _ggml_moe_a8_vec_fake(
    X: torch.Tensor,
    W: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    quant_type: int,
    row: torch.SymInt,
    tokens: torch.SymInt,
) -> torch.Tensor:
    del topk_ids, quant_type, tokens
    return torch.empty((X.size(0) * top_k, row), dtype=X.dtype, device=W.device)


def _maybe_register_fake_ops() -> None:
    global _BASE_FAKE_OPS_REGISTERED, _VEC_FAKE_OP_REGISTERED
    if not hasattr(torch.ops, "_C_gguf"):
        return

    if not _BASE_FAKE_OPS_REGISTERED and hasattr(torch.ops._C_gguf, "ggml_dequantize"):
        register_fake("_C_gguf::ggml_dequantize")(_ggml_dequantize_fake)
        register_fake("_C_gguf::ggml_mul_mat_vec_a8")(_ggml_mul_mat_vec_a8_fake)
        register_fake("_C_gguf::ggml_mul_mat_a8")(_ggml_mul_mat_a8_fake)
        register_fake("_C_gguf::ggml_moe_a8")(_ggml_moe_a8_fake)
        _BASE_FAKE_OPS_REGISTERED = True

    if not _VEC_FAKE_OP_REGISTERED and hasattr(torch.ops._C_gguf, "ggml_moe_a8_vec"):
        register_fake("_C_gguf::ggml_moe_a8_vec")(_ggml_moe_a8_vec_fake)
        _VEC_FAKE_OP_REGISTERED = True


def _ensure_cuda_ops() -> None:
    ensure_gguf_cuda_ops_loaded()
    _maybe_register_fake_ops()


_maybe_register_fake_ops()


def ggml_dequantize(
    W: torch.Tensor, quant_type: int, m: int, n: int, dtype: torch.dtype | None
) -> torch.Tensor:
    _ensure_cuda_ops()
    return torch.ops._C_gguf.ggml_dequantize(W, quant_type, m, n, dtype)


def ggml_mul_mat_vec_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    _ensure_cuda_ops()
    return torch.ops._C_gguf.ggml_mul_mat_vec_a8(W, X, quant_type, row)


def ggml_mul_mat_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    _ensure_cuda_ops()
    return torch.ops._C_gguf.ggml_mul_mat_a8(W, X, quant_type, row)


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
    _ensure_cuda_ops()
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


def ggml_moe_a8_vec(
    X: torch.Tensor,
    W: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    quant_type: int,
    row: int,
    tokens: int,
) -> torch.Tensor:
    _ensure_cuda_ops()
    return torch.ops._C_gguf.ggml_moe_a8_vec(
        X, W, topk_ids, top_k, quant_type, row, tokens
    )


def ggml_moe_get_block_size(quant_type: int) -> int:
    _ensure_cuda_ops()
    return torch.ops._C_gguf.ggml_moe_get_block_size(quant_type)


def moe_sum(input: torch.Tensor, output: torch.Tensor) -> None:
    torch.ops._moe_C.moe_sum(input, output)
