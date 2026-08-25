# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for MLA-family GGUF adapters (DeepSeek, Kimi-Linear, Kimi-K3).

llama.cpp stores the MLA ``kv_b_proj`` weight split per head as
``attn_k_b`` (num_heads, kv_lora_rank, qk_nope_head_dim) and ``attn_v_b``
(num_heads, v_head_dim, kv_lora_rank), while vLLM expects one fused
``self_attn.kv_b_proj.weight`` of shape
``(num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank)`` with the
nope part ahead of the value part per head.

The k_b half is stored transposed relative to the fused layout, which
scrambles GGUF quantization blocks, so the fusion has to dequantize. That
is acceptable because vLLM's MLA absorption (W_UK/W_UV) has no quantized
bmm path and dequantizes kv_b_proj at load time regardless.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import torch

from ..weight_utils import dequantize_gguf_tensor
from .base import GGUFWeight

#: Substr map entries for the split tensors; adapters should include them in
#: their text mapper so the fuser below can recognize the pair.
MLA_KV_B_SUBSTR: dict[str, str] = {
    "attn_k_b.": "self_attn.k_b.",
    "attn_v_b.": "self_attn.v_b.",
}

#: The fused kv_b_proj is rebuilt as a plain float weight.
MLA_KV_B_UNQUANTIZED_MODULES: tuple[str, ...] = ("self_attn.kv_b_proj",)

_KV_B_PATTERN = re.compile(r"self_attn\.(k_b|v_b)\.(weight|qweight|qweight_type)$")


def fuse_kv_b_proj_weights(
    weights: Iterable[GGUFWeight],
    dtype: torch.dtype,
) -> Iterable[GGUFWeight]:
    """Fuse llama.cpp's split attn_k_b/attn_v_b pairs into kv_b_proj weights.

    Buffers each layer's pair (they may live in different GGUF shards) and
    emits ``...self_attn.kv_b_proj.weight`` once both halves arrived.
    """
    pending: dict[str, dict[str, tuple[torch.Tensor | None, int | None]]] = {}
    for name, weight in weights:
        match = _KV_B_PATTERN.search(name)
        if match is None:
            yield name, weight
            continue
        kind, suffix = match.group(1), match.group(2)
        prefix = name[: match.start()]
        entry = pending.setdefault(prefix, {"k_b": (None, None), "v_b": (None, None)})
        if suffix == "qweight_type":
            entry[kind] = (entry[kind][0], int(weight.item()))
            continue
        entry[kind] = (weight, entry[kind][1])
        (k_b, _), (v_b, _) = entry["k_b"], entry["v_b"]
        if k_b is None or v_b is None:
            continue
        k_b = dequantize_gguf_tensor(k_b, entry["k_b"][1])  # (H, L, Dn)
        v_b = dequantize_gguf_tensor(v_b, entry["v_b"][1])  # (H, Dv, L)
        fused = torch.cat([k_b.transpose(1, 2), v_b], dim=1)
        fused = fused.reshape(-1, fused.shape[-1])
        yield prefix + "self_attn.kv_b_proj.weight", fused.to(dtype)
        del pending[prefix]

    if pending:
        raise RuntimeError(
            f"Incomplete MLA kv_b tensors, missing k_b/v_b pair for: {sorted(pending)}"
        )
