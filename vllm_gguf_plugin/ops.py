# SPDX-License-Identifier: Apache-2.0

import os
import typing

import torch

from .kernel_support import (  # noqa: F401
    CUDA_LEGACY_MMQ_TYPES,
    CUDA_LEGACY_MMVQ_TYPES,
    CUDA_UPSTREAM_DEQUANT_TYPES,
    CUDA_UPSTREAM_MMQ_TYPES,
    CUDA_UPSTREAM_MMVQ_TYPES,
    CUDA_UPSTREAM_MOE_TYPES,
    GGML_TYPE_Q2_0,
    KernelMethod,
    QuantizationBackend,
    QuantizationOperation,
    supports,
    supports_moe,
)
from .triton.dequantize.interface import ggml_dequantize_triton
from .triton.fused_moe.interface import ggml_moe_a8_triton
from .triton.fused_moe.utils import get_triton_moe_block_m
from .triton.gemm.interface import ggml_mul_mat_a8_triton

# Public re-exports: tests and external consumers reference
# `ops.GGML_TYPE_*` as stable numeric constants, so keep them importable
# from this module even though backend capability tables now live in
# kernel_support.py.
from .triton.gemm.utils import (  # noqa: F401
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
)

try:
    from torch.library import register_fake
except ImportError:
    from torch.library import impl_abstract as register_fake

# Backend selection: use CUDA kernels by default, unless explicitly disabled.
_USE_CUDA = os.environ.get("VLLM_GGUF_USE_CUDA", "1") == "1"

# Try importing CUDA extension
try:
    from . import _C_gguf  # noqa: F401

    _CUDA_AVAILABLE = True
except ImportError:
    _C_gguf = None
    _CUDA_AVAILABLE = False


# Effective CUDA usage: only when enabled AND available.
_CUDA_ENABLED = _USE_CUDA and _CUDA_AVAILABLE

_GLOBAL_KERNEL_ENV = "VLLM_GGUF_CUDA_KERNEL"
_VALID_KERNEL_MODES = {"auto", "upstream", "legacy", "triton"}


def _kernel_mode(env_name: str) -> str:
    mode = os.environ.get(env_name)
    if mode is None:
        mode = os.environ.get(_GLOBAL_KERNEL_ENV, "auto")
    if mode not in _VALID_KERNEL_MODES:
        raise ValueError(f"{env_name} must be one of auto|upstream|legacy|triton")
    return mode


def cuda_dense_kernel_mode() -> str:
    return _kernel_mode("VLLM_GGUF_CUDA_DENSE_KERNEL")


def cuda_moe_kernel_mode() -> str:
    return _kernel_mode("VLLM_GGUF_CUDA_MOE_KERNEL")


def cuda_dequantize_kernel_mode() -> str:
    return _kernel_mode("VLLM_GGUF_CUDA_DEQUANTIZE_KERNEL")


def cuda_kernel_mode() -> str:
    """Alias for :func:`cuda_dense_kernel_mode` (public selector API)."""
    return cuda_dense_kernel_mode()


def cuda_dense_upstream_enabled() -> bool:
    return (
        _CUDA_ENABLED
        and torch.version.hip is None
        and cuda_dense_kernel_mode() in {"upstream", "auto"}
    )


def cuda_upstream_enabled() -> bool:
    """Alias for :func:`cuda_dense_upstream_enabled` (public selector API)."""
    return cuda_dense_upstream_enabled()


def should_use_upstream_mmvq(X: torch.Tensor, quant_type: int) -> bool:
    """Query the pinned llama.cpp policy for the input tensor's device.

    This is a host-only query; it neither launches a kernel nor synchronizes.
    Keep the architecture/type thresholds in upstream rather than copying them
    into the Python dispatcher. Storage eligibility is checked by the bridge.
    """
    major, minor = torch.cuda.get_device_capability(X.device)
    cc = major * 100 + minor * 10
    return torch.ops._C_gguf.ggml_should_use_mmvq(quant_type, cc, X.shape[0])


DENSE_MMVF = KernelMethod.MMVF
DENSE_MMF = KernelMethod.MMF
DENSE_MMVQ = KernelMethod.MMVQ
DENSE_MMQ = KernelMethod.MMQ
DENSE_DEQUANTIZE_BLAS = KernelMethod.DEQUANTIZE_BLAS
DENSE_BLAS = KernelMethod.BLAS


def _require_upstream(op_name: str) -> None:
    if not (
        _CUDA_ENABLED and torch.version.hip is None and _cuda_kernel_available(op_name)
    ):
        raise RuntimeError(
            f"upstream CUDA op {op_name} is unavailable; "
            "select legacy/triton explicitly if needed"
        )


def dense_supported_methods(W, X, quant_type, row) -> KernelMethod:
    _require_upstream("ggml_dense_supported_methods")
    return KernelMethod(
        torch.ops._C_gguf.ggml_dense_supported_methods(W, X, quant_type, row)
    )


def dense_select_method(W, X, quant_type, row) -> KernelMethod:
    _require_upstream("ggml_dense_select_method")
    return KernelMethod(
        torch.ops._C_gguf.ggml_dense_select_method(W, X, quant_type, row)
    )


def dense_upstream_capabilities(W, X, quant_type, row) -> int:
    """Compatibility query of recommended bits.

    Use supported_methods for eligibility.
    """
    _require_upstream("ggml_dense_upstream_capabilities")
    return torch.ops._C_gguf.ggml_dense_upstream_capabilities(W, X, quant_type, row)


def ggml_dense(W, X, quant_type, row):
    """Decision interface: centralized upstream selection, never another backend."""
    _require_upstream("ggml_dense")
    return torch.ops._C_gguf.ggml_dense(W, X, quant_type, row)


def ggml_dense_mmvq(W, X, quant_type, row):
    return torch.ops._C_gguf.ggml_dense_mmvq(W, X, quant_type, row)


def ggml_dense_mmq(W, X, quant_type, row):
    return torch.ops._C_gguf.ggml_dense_mmq(W, X, quant_type, row)


def ggml_dense_mmvf(W, X, quant_type, row):
    return torch.ops._C_gguf.ggml_dense_mmvf(W, X, quant_type, row)


def ggml_dense_mmf(W, X, quant_type, row):
    return torch.ops._C_gguf.ggml_dense_mmf(W, X, quant_type, row)


def ggml_dense_blas(W, X, quant_type, row):
    """Fixed cuBLAS method for floating weights."""
    return torch.ops._C_gguf.ggml_dense_blas(W, X, quant_type, row)


def ggml_dense_dequantize_blas(W, X, quant_type, row):
    """Fixed dequantization + cuBLAS method for packed weights."""
    return torch.ops._C_gguf.ggml_dense_dequantize_blas(W, X, quant_type, row)


def cuda_dequantize_upstream_enabled() -> bool:
    return (
        _CUDA_ENABLED
        and torch.version.hip is None
        and cuda_dequantize_kernel_mode() in {"upstream", "auto"}
    )


def _cuda_kernel_available(op_name: str, quant_type: int | None = None) -> bool:
    if not _CUDA_ENABLED:
        return False
    namespace = getattr(torch.ops, "_C_gguf", None)
    if namespace is None or not hasattr(namespace, op_name):
        return False
    if quant_type is None:
        return True
    return supports(
        quant_type,
        QuantizationBackend.LEGACY,
        QuantizationOperation.MMVQ,
    )


def _cuda_legacy_moe_available(
    op_name: str, quant_type: int, operation: QuantizationOperation
) -> bool:
    return _cuda_kernel_available(op_name) and supports(
        quant_type, QuantizationBackend.LEGACY, operation
    )


def _cuda_moe_upstream_kernel_available(op_name: str, quant_type: int) -> bool:
    return (
        _CUDA_ENABLED
        and torch.version.hip is None
        and _cuda_kernel_available(op_name)
        and supports_moe(quant_type, QuantizationBackend.UPSTREAM)
    )


# Fake registration is per schema, including legacy-only / partial builds.
def _register_fake_if_available(name, function):
    if _CUDA_AVAILABLE and hasattr(torch.ops._C_gguf, name):
        register_fake(f"_C_gguf::{name}")(function)


def _dequant_fake(W, quant_type, m, n, dtype=None):
    return torch.empty((m, n), dtype=dtype or torch.float16, device=W.device)


def _dense_fake(W, X, quant_type, row):
    return torch.empty((X.size(0), row), dtype=X.dtype, device=X.device)


def _moe_fake(X, W, topk_ids, quant_type, row, top_k, tokens):
    return torch.empty((X.size(0) * top_k, row), dtype=X.dtype, device=X.device)


def _moe_aligned_fake(
    X, W, sorted_route_ids, quant_type, row, top_k, tokens, expert_ids, padded_count
):
    return _moe_fake(X, W, sorted_route_ids, quant_type, row, top_k, tokens)


def _legacy_moe_fake(
    X,
    W,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    quant_type,
    row,
    top_k,
    tokens,
):
    return _moe_fake(X, W, sorted_token_ids, quant_type, row, top_k, tokens)


def _legacy_moe_vec_fake(X, W, topk_ids, top_k, quant_type, row, tokens):
    return _moe_fake(X, W, topk_ids, quant_type, row, top_k, tokens)


for _name in ("ggml_dequantize", "ggml_dequantize_upstream"):
    _register_fake_if_available(_name, _dequant_fake)
for _name in (
    "ggml_dense",
    "ggml_dense_mmvq",
    "ggml_dense_mmq",
    "ggml_dense_mmvf",
    "ggml_dense_mmf",
    "ggml_dense_blas",
    "ggml_dense_dequantize_blas",
    "ggml_mul_mat_vec_a8",
    "ggml_mul_mat_a8",
):
    _register_fake_if_available(_name, _dense_fake)
for _name in (
    "ggml_moe",
    "ggml_moe_upstream",
    "ggml_moe_a8_upstream",
    "ggml_moe_mmvq",
    "ggml_moe_mmq",
    "ggml_moe_mmvf",
    "ggml_moe_mmf",
    "ggml_moe_blas",
    "ggml_moe_dequantize_blas",
    "ggml_moe_grouped_dense",
):
    _register_fake_if_available(_name, _moe_fake)
_register_fake_if_available("ggml_moe_mmq_aligned", _moe_aligned_fake)
_register_fake_if_available("ggml_moe_a8", _legacy_moe_fake)
_register_fake_if_available("ggml_moe_a8_vec", _legacy_moe_vec_fake)


# --- Public API ---


def _cuda_upstream_supports(
    op_name: str, quant_type: int, operation: QuantizationOperation
) -> bool:
    return (
        _CUDA_ENABLED
        and torch.version.hip is None
        and _cuda_kernel_available(op_name)
        and supports(quant_type, QuantizationBackend.UPSTREAM, operation)
    )


def _raise_backend_unavailable(
    backend: str, operation: str, quant_type: int
) -> typing.NoReturn:
    raise RuntimeError(
        f"{backend} {operation} backend is unavailable for quantization type "
        f"{quant_type}"
    )


def ggml_dequantize(
    W: torch.Tensor, quant_type: int, m: int, n: int, dtype: torch.dtype | None
) -> torch.Tensor:
    mode = cuda_dequantize_kernel_mode()
    upstream_available = _cuda_upstream_supports(
        "ggml_dequantize_upstream",
        quant_type,
        QuantizationOperation.DEQUANTIZE,
    )
    if mode in {"upstream", "auto"}:
        if not upstream_available:
            _raise_backend_unavailable("upstream", "dequantize", quant_type)
        return torch.ops._C_gguf.ggml_dequantize_upstream(W, quant_type, m, n, dtype)
    if mode == "legacy":
        if not _cuda_kernel_available("ggml_dequantize", quant_type):
            _raise_backend_unavailable("legacy", "dequantize", quant_type)
        return torch.ops._C_gguf.ggml_dequantize(W, quant_type, m, n, dtype)
    if mode == "triton":
        if not supports(
            quant_type, QuantizationBackend.TRITON, QuantizationOperation.DEQUANTIZE
        ):
            _raise_backend_unavailable("triton", "dequantize", quant_type)
        return ggml_dequantize_triton(W, quant_type, m, n, dtype)
    _raise_backend_unavailable(mode, "dequantize", quant_type)


def ggml_mul_mat_vec_a8(W, X, quant_type, row):
    """Compatibility selector; fixed upstream MMVQ is ggml_dense_mmvq."""
    mode = cuda_dense_kernel_mode()
    if mode in {"upstream", "auto"}:
        return ggml_dense(W, X, quant_type, row)
    if mode == "legacy" and _cuda_kernel_available("ggml_mul_mat_vec_a8", quant_type):
        return torch.ops._C_gguf.ggml_mul_mat_vec_a8(W, X, quant_type, row)
    if mode == "triton" and supports(
        quant_type, QuantizationBackend.TRITON, QuantizationOperation.MMVQ
    ):
        return ggml_mul_mat_a8_triton(W, X, quant_type, row)
    _raise_backend_unavailable(mode, "MMVQ", quant_type)


def ggml_mul_mat_a8(W, X, quant_type, row):
    """Compatibility selector; fixed upstream MMQ is ggml_dense_mmq."""
    mode = cuda_dense_kernel_mode()
    if mode in {"upstream", "auto"}:
        return ggml_dense(W, X, quant_type, row)
    if (
        mode == "legacy"
        and supports(quant_type, QuantizationBackend.LEGACY, QuantizationOperation.MMQ)
        and _cuda_kernel_available("ggml_mul_mat_a8")
    ):
        return torch.ops._C_gguf.ggml_mul_mat_a8(W, X, quant_type, row)
    if mode == "triton" and supports(
        quant_type, QuantizationBackend.TRITON, QuantizationOperation.DEQUANTIZE
    ):
        return ggml_mul_mat_a8_triton(W, X, quant_type, row)
    _raise_backend_unavailable(mode, "MMQ", quant_type)


def cuda_moe_upstream_kernel_available(quant_type: int) -> bool:
    return _cuda_moe_upstream_kernel_available("ggml_moe_upstream", quant_type)


def ggml_moe_mmvq(X, W, topk_ids, quant_type, row, top_k, tokens):
    """Force upstream MoE MMVQ, with internal token chunking and no fallback."""
    return torch.ops._C_gguf.ggml_moe_mmvq(
        X, W, topk_ids, quant_type, row, top_k, tokens
    )


def ggml_moe_mmq(X, W, topk_ids, quant_type, row, top_k, tokens):
    """Fixed raw-ID upstream MMQ; no policy or implicit alignment."""
    return torch.ops._C_gguf.ggml_moe_mmq(
        X, W, topk_ids, quant_type, row, top_k, tokens
    )


def ggml_moe_mmq_aligned(
    X, W, sorted_route_ids, quant_type, row, top_k, tokens, *, expert_ids, padded_count
):
    return torch.ops._C_gguf.ggml_moe_mmq_aligned(
        X, W, sorted_route_ids, quant_type, row, top_k, tokens, expert_ids, padded_count
    )


def _cuda_moe_aligned_mmq_available(X, W, quant_type):
    """Legacy coarse probe; decision callers use moe_select_method instead."""
    return _cuda_upstream_supports(
        "ggml_moe_mmq_aligned", quant_type, QuantizationOperation.MMQ_ALIGNED
    )


def moe_supported_methods(X, W, topk_ids, quant_type, row, top_k, tokens):
    _require_upstream("ggml_moe_supported_methods")
    return KernelMethod(
        torch.ops._C_gguf.ggml_moe_supported_methods(
            X, W, topk_ids, quant_type, row, top_k, tokens
        )
    )


def moe_select_method(
    X, W, topk_ids, quant_type, row, top_k, tokens, *, allow_aligned=False
):
    _require_upstream("ggml_moe_select_method")
    return KernelMethod(
        torch.ops._C_gguf.ggml_moe_select_method(
            X, W, topk_ids, quant_type, row, top_k, tokens, allow_aligned
        )
    )


def ggml_moe(X, W, topk_ids, quant_type, row, top_k, tokens, *, alignment_cache=None):
    """Decision interface; C++ owns method policy, Python builds selected alignment."""
    _require_upstream("ggml_moe")
    method = moe_select_method(
        X, W, topk_ids, quant_type, row, top_k, tokens, allow_aligned=True
    )
    if method == KernelMethod.MMQ_ALIGNED:
        from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

        # W1/W2 share only when their caller supplies a same-type cache.
        key = (quant_type, W.size(0), topk_ids.numel(), topk_ids.data_ptr())
        plan = None if alignment_cache is None else alignment_cache.get(key)
        if plan is None:
            tile = torch.ops._C_gguf.ggml_moe_alignment_block_size()
            plan = moe_align_block_size(topk_ids, tile, W.size(0), pad_sorted_ids=True)
            if alignment_cache is not None:
                alignment_cache[key] = plan
        return ggml_moe_mmq_aligned(
            X,
            W,
            plan[0],
            quant_type,
            row,
            top_k,
            tokens,
            expert_ids=plan[1],
            padded_count=plan[2],
        )
    return torch.ops._C_gguf.ggml_moe(X, W, topk_ids, quant_type, row, top_k, tokens)


def ggml_moe_mmvf(X, W, topk_ids, quant_type, row, top_k, tokens):
    return torch.ops._C_gguf.ggml_moe_mmvf(
        X, W, topk_ids, quant_type, row, top_k, tokens
    )


def ggml_moe_mmf(X, W, topk_ids, quant_type, row, top_k, tokens):
    return torch.ops._C_gguf.ggml_moe_mmf(
        X, W, topk_ids, quant_type, row, top_k, tokens
    )


def ggml_moe_blas(X, W, topk_ids, quant_type, row, top_k, tokens):
    return torch.ops._C_gguf.ggml_moe_blas(
        X, W, topk_ids, quant_type, row, top_k, tokens
    )


def ggml_moe_dequantize_blas(X, W, topk_ids, quant_type, row, top_k, tokens):
    return torch.ops._C_gguf.ggml_moe_dequantize_blas(
        X, W, topk_ids, quant_type, row, top_k, tokens
    )


def ggml_moe_grouped_dense(X, W, topk_ids, quant_type, row, top_k, tokens):
    """Group routes by expert and use dense dispatch; CUDA graphs are unsupported."""
    return torch.ops._C_gguf.ggml_moe_grouped_dense(
        X, W, topk_ids, quant_type, row, top_k, tokens
    )


def ggml_moe_upstream(
    X: torch.Tensor,
    W: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type: int,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    """Compatibility alias for the upstream decision interface."""
    return ggml_moe(X, W, topk_ids, quant_type, row, top_k, tokens)


def ggml_moe_a8_upstream(
    X: torch.Tensor,
    W: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type: int,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    """Compatibility alias; supports floating and quantized MoE weights."""
    return ggml_moe_upstream(X, W, topk_ids, quant_type, row, top_k, tokens)


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
    mode = cuda_moe_kernel_mode()
    if mode in {"upstream", "auto"}:
        _raise_backend_unavailable(mode, "MoE MMQ", quant_type)

    if mode == "legacy":
        if _cuda_legacy_moe_available(
            "ggml_moe_a8", quant_type, QuantizationOperation.MMQ
        ):
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
        _raise_backend_unavailable(mode, "MoE MMQ", quant_type)

    if mode == "triton" and supports(
        quant_type, QuantizationBackend.TRITON, QuantizationOperation.MMQ
    ):
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
    _raise_backend_unavailable(mode, "MoE MMQ", quant_type)


def ggml_moe_a8_vec(
    X: torch.Tensor,
    W: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    quant_type: int,
    row: int,
    tokens: int,
) -> torch.Tensor:
    mode = cuda_moe_kernel_mode()
    if mode in {"upstream", "auto"}:
        _raise_backend_unavailable(mode, "MoE MMVQ", quant_type)

    if mode == "legacy":
        if _cuda_legacy_moe_available(
            "ggml_moe_a8_vec", quant_type, QuantizationOperation.MMVQ
        ):
            return torch.ops._C_gguf.ggml_moe_a8_vec(
                X, W, topk_ids, top_k, quant_type, row, tokens
            )
        _raise_backend_unavailable(mode, "MoE MMVQ", quant_type)

    if mode == "triton" and supports(
        quant_type, QuantizationBackend.TRITON, QuantizationOperation.MMVQ
    ):
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
    _raise_backend_unavailable(mode, "MoE MMVQ", quant_type)


def ggml_moe_get_block_size(quant_type: int) -> int:
    mode = cuda_moe_kernel_mode()
    if mode in {"upstream", "auto"}:
        _raise_backend_unavailable(mode, "MoE block-size", quant_type)
    if mode == "legacy":
        if _cuda_legacy_moe_available(
            "ggml_moe_get_block_size", quant_type, QuantizationOperation.MMQ
        ):
            return torch.ops._C_gguf.ggml_moe_get_block_size(quant_type)
        _raise_backend_unavailable(mode, "MoE block-size", quant_type)
    # This helper describes Triton's layout; the caller checks kernel support.
    return get_triton_moe_block_m(quant_type)


def moe_sum(input: torch.Tensor, output: torch.Tensor) -> None:
    torch.ops._moe_C.moe_sum(input, output)
