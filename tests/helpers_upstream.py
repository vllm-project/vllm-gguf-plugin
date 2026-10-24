"""Shared helpers for upstream CUDA kernel tests.

The helpers cover:

- ``TEMPLATE_EXTRA_TYPES``: optional quant types that only exist on newer
  ``gguf`` releases (Q1_0, Q2_0, MXFP4, NVFP4).
- ``make_padded_weight`` / ``make_padded_moe_weight``: build CUDA uint8 weight
  tensors whose storage reserves the MATRIX_ROW_PADDING (512) tail that the
  upstream kernels require.
- ``make_template_raw``: build valid random raw block bytes for template-only
  types so ``gguf.dequantize`` can produce a reference.
- ``quantize_padded``: quantize a dense matrix and return the padded weight.
"""

from __future__ import annotations

import numpy as np
import torch
from gguf import GGML_QUANT_SIZES
from gguf import GGMLQuantizationType as Q

# Types carried by upstream template instances that may not exist in older
# gguf-python releases.  All tests must tolerate their absence.
TEMPLATE_EXTRA_TYPE_NAMES = ("Q1_0", "Q2_0", "MXFP4", "NVFP4")
TEMPLATE_EXTRA_TYPES = tuple(
    getattr(Q, name) for name in TEMPLATE_EXTRA_TYPE_NAMES if hasattr(Q, name)
)

MMQ_TYPES = (
    Q.Q4_0,
    Q.Q4_1,
    Q.Q5_0,
    Q.Q5_1,
    Q.Q8_0,
    Q.Q2_K,
    Q.Q3_K,
    Q.Q4_K,
    Q.Q5_K,
    Q.Q6_K,
    Q.IQ1_S,
    Q.IQ2_XXS,
    Q.IQ2_XS,
    Q.IQ2_S,
    Q.IQ3_XXS,
    Q.IQ3_S,
    Q.IQ4_NL,
    Q.IQ4_XS,
) + TEMPLATE_EXTRA_TYPES

MATRIX_ROW_PADDING = 512


def upstream_padding_bytes(quant_type: Q, k: int) -> int:
    """Storage tail (bytes) required for a logical row of ``k`` values."""
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    return (-k) % MATRIX_ROW_PADDING // block_size * type_size


def make_padded_weight(packed: np.ndarray, quant_type: Q, k: int) -> torch.Tensor:
    """Copy ``packed`` rows into a zero-tailed storage sized for upstream MMQ."""
    raw = torch.from_numpy(packed).cuda()
    padding = upstream_padding_bytes(quant_type, k)
    storage = torch.zeros(raw.numel() + padding, dtype=torch.uint8, device="cuda")
    storage[: raw.numel()].copy_(raw.reshape(-1))
    return storage[: raw.numel()].view_as(raw)


def make_padded_moe_weight(packed: np.ndarray, quant_type: Q, k: int) -> torch.Tensor:
    """Same tail reservation for 3D MoE weights of shape (experts, n, packed)."""
    return make_padded_weight(packed, quant_type, k)


def make_template_raw(quant_type: Q, rows: int, n: int) -> np.ndarray:
    """Random-but-valid raw block bytes for a template-only quant type.

    The bytes must survive ``gguf.dequantize`` (used as the numerical
    reference), so NaN scale encodings are avoided.
    """
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    raw = np.random.default_rng(int(quant_type) + 300).integers(
        0, 256, (rows, n // block_size * type_size), dtype=np.uint8
    )
    if quant_type.name == "MXFP4":
        # One E8M0 scale byte per 32-value block; 0x7F..0xFF are special.
        raw[:, ::type_size] = np.random.default_rng(301).integers(
            0, 0x7F, (rows, n // block_size), dtype=np.uint8
        )
    elif quant_type.name == "NVFP4":
        # Four scale bytes per 64-value block; 0x7F is NaN.
        for offset in range(0, raw.shape[1], type_size):
            raw[:, offset : offset + 4] = np.random.default_rng(302 + offset).integers(
                0, 0x7F, (rows, 4), dtype=np.uint8
            )
    return raw
