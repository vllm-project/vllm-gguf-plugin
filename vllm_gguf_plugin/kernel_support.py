# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, IntFlag

import gguf
from gguf import GGMLQuantizationType as WeightType


class QuantizationBackend(str, Enum):
    TRITON = "triton"
    LEGACY = "legacy"
    UPSTREAM = "upstream"


class QuantizationOperation(str, Enum):
    DEQUANTIZE = "dequantize"
    MMVQ = "mmvq"
    MMQ = "mmq"
    MMVF = "mmvf"
    MMF = "mmf"
    BLAS = "blas"
    DEQUANTIZE_BLAS = "dequantize_blas"
    GROUPED_DENSE = "grouped_dense"
    MMQ_ALIGNED = "mmq_aligned"


class KernelMethod(IntFlag):
    """Method bits shared with csrc/upstream/kernel_dispatch.cuh.

    This enum does not select a backend.
    """

    NONE = 0
    MMVF = 1
    MMF = 2
    MMVQ = 4
    MMQ = 8
    DEQUANTIZE_BLAS = 16
    BLAS = 32
    GROUPED_DENSE = 64
    MMQ_ALIGNED = 128


@dataclass(frozen=True, slots=True)
class QuantizationSupport:
    """Kernel capabilities for one GGML quantization type."""

    triton: frozenset[QuantizationOperation] = frozenset()
    legacy: frozenset[QuantizationOperation] = frozenset()
    upstream: frozenset[QuantizationOperation] = frozenset()

    def supports(
        self, backend: QuantizationBackend, operation: QuantizationOperation
    ) -> bool:
        return operation in getattr(self, backend.value)


# The pinned GGML ABI includes Q2_0 even when gguf-python has no enum member.
GGML_TYPE_Q2_0 = 42
_GGML_FALLBACK_TYPES = {"Q2_0": GGML_TYPE_Q2_0}
_GGML_FALLBACK_SIZES = {GGML_TYPE_Q2_0: (64, 18)}


def _types(*names: str) -> tuple[int, ...]:
    """Resolve GGML IDs independently of optional gguf-python enum members."""
    return tuple(
        value
        for name in names
        if (value := getattr(WeightType, name, _GGML_FALLBACK_TYPES.get(name)))
        is not None
    )


_TRITON_DEQUANT_MMVQ = (
    QuantizationOperation.DEQUANTIZE,
    QuantizationOperation.MMVQ,
)
_TRITON_ALL = (
    QuantizationOperation.DEQUANTIZE,
    QuantizationOperation.MMVQ,
    QuantizationOperation.MMQ,
)
_LEGACY_DEQUANT_MMVQ = (
    QuantizationOperation.DEQUANTIZE,
    QuantizationOperation.MMVQ,
)
_LEGACY_ALL = (
    QuantizationOperation.DEQUANTIZE,
    QuantizationOperation.MMVQ,
    QuantizationOperation.MMQ,
)
_UPSTREAM_DEQUANT_MMVQ = (
    QuantizationOperation.DEQUANTIZE,
    QuantizationOperation.MMVQ,
    QuantizationOperation.DEQUANTIZE_BLAS,
    QuantizationOperation.GROUPED_DENSE,
)
_UPSTREAM_ALL = (
    QuantizationOperation.DEQUANTIZE,
    QuantizationOperation.MMVQ,
    QuantizationOperation.MMQ,
    QuantizationOperation.MMQ_ALIGNED,
    QuantizationOperation.DEQUANTIZE_BLAS,
    QuantizationOperation.GROUPED_DENSE,
)
_UPSTREAM_FLOAT = (
    QuantizationOperation.MMVF,
    QuantizationOperation.MMF,
    QuantizationOperation.BLAS,
    QuantizationOperation.GROUPED_DENSE,
)


def _support(
    *,
    triton: Iterable[QuantizationOperation] = (),
    legacy: Iterable[QuantizationOperation] = (),
    upstream: Iterable[QuantizationOperation] = (),
) -> QuantizationSupport:
    return QuantizationSupport(
        triton=frozenset(triton),
        legacy=frozenset(legacy),
        upstream=frozenset(upstream),
    )


# This is the single Python-side capability table. Keep the groups disjoint so
# every row states the complete support contract for its quantization family.
_STANDARD_K_TYPES = _types(
    "Q4_0",
    "Q4_1",
    "Q5_0",
    "Q5_1",
    "Q8_0",
    "Q2_K",
    "Q3_K",
    "Q4_K",
    "Q5_K",
    "Q6_K",
)
_IQ_TYPES = _types(
    "IQ1_M",
    "IQ1_S",
    "IQ2_XXS",
    "IQ2_XS",
    "IQ2_S",
    "IQ3_XXS",
    "IQ3_S",
    "IQ4_XS",
    "IQ4_NL",
)
_UPSTREAM_EXTRA_TYPES = _types("Q1_0", "Q2_0", "MXFP4", "NVFP4")

_SUPPORT: dict[int, QuantizationSupport] = {}


def _register(types: Iterable[int], support: QuantizationSupport) -> None:
    for weight_type in types:
        if weight_type in _SUPPORT:
            raise AssertionError(f"duplicate support row for {weight_type}")
        _SUPPORT[weight_type] = support


# Q8_1 is available through Triton's generic quantized path only.
_register(_types("Q8_1"), _support(triton=_TRITON_ALL))

# Floating MoE projections use the upstream MMVF/MMF wrappers with ids.
_register(_types("F32", "F16", "BF16"), _support(upstream=_UPSTREAM_FLOAT))

# Standard and K-quant formats have all three operations in every CUDA path.
_register(
    _STANDARD_K_TYPES,
    _support(
        triton=_TRITON_ALL,
        legacy=_LEGACY_ALL,
        upstream=_UPSTREAM_ALL,
    ),
)

# IQ1_M has no upstream MMQ instance. Dense calls can still use the upstream
# dequantize + cuBLAS fallback above the MMVQ limit.
_register(
    _types("IQ1_M"),
    _support(
        triton=_TRITON_DEQUANT_MMVQ,
        legacy=_LEGACY_DEQUANT_MMVQ,
        upstream=_UPSTREAM_DEQUANT_MMVQ,
    ),
)
_register(
    _types(
        "IQ1_S",
        "IQ2_XXS",
        "IQ2_XS",
        "IQ2_S",
        "IQ3_XXS",
        "IQ3_S",
        "IQ4_XS",
        "IQ4_NL",
    ),
    _support(
        triton=_TRITON_DEQUANT_MMVQ,
        legacy=_LEGACY_DEQUANT_MMVQ,
        upstream=_UPSTREAM_ALL,
    ),
)

# These formats are provided by the upstream template/conversion instances;
# they are intentionally not advertised as legacy or Triton capabilities.
_register(_UPSTREAM_EXTRA_TYPES, _support(upstream=_UPSTREAM_ALL))


def get_quantization_support(weight_type: int) -> QuantizationSupport:
    try:
        return _SUPPORT.get(int(weight_type), QuantizationSupport())
    except (TypeError, ValueError):
        return QuantizationSupport()


def supports(
    weight_type: int,
    backend: QuantizationBackend,
    operation: QuantizationOperation,
) -> bool:
    return get_quantization_support(weight_type).supports(backend, operation)


def supports_moe(weight_type: int, backend: QuantizationBackend) -> bool:
    """Report whether this backend has a MoE projection kernel family."""
    support = get_quantization_support(weight_type)
    return any(
        support.supports(backend, operation)
        for operation in (
            QuantizationOperation.MMVQ,
            QuantizationOperation.MMQ,
            QuantizationOperation.MMVF,
            QuantizationOperation.MMF,
        )
    )


def _types_for(
    backend: QuantizationBackend, operation: QuantizationOperation
) -> frozenset[int]:
    return frozenset(
        weight_type
        for weight_type, support in _SUPPORT.items()
        if support.supports(backend, operation)
    )


# Compatibility exports. New code should use supports()/supports_moe() so the
# operation and backend remain explicit at the call site.
CUDA_LEGACY_MMVQ_TYPES = _types_for(
    QuantizationBackend.LEGACY, QuantizationOperation.MMVQ
)
CUDA_LEGACY_MMQ_TYPES = _types_for(
    QuantizationBackend.LEGACY, QuantizationOperation.MMQ
)
CUDA_UPSTREAM_MMVQ_TYPES = _types_for(
    QuantizationBackend.UPSTREAM, QuantizationOperation.MMVQ
)
CUDA_UPSTREAM_MMQ_TYPES = _types_for(
    QuantizationBackend.UPSTREAM, QuantizationOperation.MMQ
)
CUDA_UPSTREAM_DEQUANT_TYPES = _types_for(
    QuantizationBackend.UPSTREAM, QuantizationOperation.DEQUANTIZE
)
CUDA_UPSTREAM_MOE_TYPES = frozenset(
    weight_type
    for weight_type in _SUPPORT
    if supports_moe(weight_type, QuantizationBackend.UPSTREAM)
)

TRITON_DEQUANT_TYPES = _types_for(
    QuantizationBackend.TRITON, QuantizationOperation.DEQUANTIZE
)
TRITON_MMVQ_TYPES = _types_for(QuantizationBackend.TRITON, QuantizationOperation.MMVQ)
TRITON_MMQ_TYPES = _types_for(QuantizationBackend.TRITON, QuantizationOperation.MMQ)

UNQUANTIZED_TYPES = frozenset(_types("F32", "F16", "BF16"))
STANDARD_QUANT_TYPES = frozenset(_types("Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q8_1"))
KQUANT_TYPES = frozenset(_types("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"))
IMATRIX_QUANT_TYPES = frozenset(_IQ_TYPES)

_UPSTREAM_STORAGE_TYPES = CUDA_UPSTREAM_MMVQ_TYPES | CUDA_UPSTREAM_MMQ_TYPES

# Must match the upstream MATRIX_ROW_PADDING macro (common.cuh); ggml_dypes.cuh
# static-asserts the C++ side stays a multiple of this value.
_MATRIX_ROW_PADDING = 512

# Stable error marker for unsupported upstream MoE inputs; never enables fallback.
MOE_NOT_ELIGIBLE_MARKER = "VLLM_GGUF_MOE_NOT_ELIGIBLE"


def upstream_storage_padding_bytes(weight_type: int, packed_row_size: int) -> int:
    """Return the extra byte storage required by upstream dense CUDA kernels."""
    if weight_type not in _UPSTREAM_STORAGE_TYPES or packed_row_size <= 0:
        return 0
    if weight_type in _GGML_FALLBACK_SIZES:
        block_size, type_size = _GGML_FALLBACK_SIZES[weight_type]
    else:
        block_size, type_size = gguf.GGML_QUANT_SIZES[WeightType(weight_type)]
    if packed_row_size % type_size:
        return 0
    logical_k = packed_row_size // type_size * block_size
    padding_k = (-logical_k) % _MATRIX_ROW_PADDING
    return padding_k // block_size * type_size
