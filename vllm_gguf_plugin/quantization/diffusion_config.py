# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF quantization config for diffusion transformers.

Uses dequant+GEMM instead of the fused kernel path (which expects 2D inputs).
"""

from __future__ import annotations

import gguf
import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

from .. import ops
from .config import GGUFConfig
from .linear import GGUFLinearMethod
from .utils import UNQUANTIZED_TYPES, is_layer_skipped_gguf


def dequant_gguf(
    weight: torch.Tensor, weight_type: int, dtype: torch.dtype
) -> torch.Tensor:
    if weight_type in UNQUANTIZED_TYPES:
        return weight

    block_size, type_size = gguf.GGML_QUANT_SIZES[weight_type]
    shape = (weight.shape[0], weight.shape[1] // type_size * block_size)
    return ops.ggml_dequantize(weight, weight_type, *shape, dtype)


def dequant_gemm_gguf(
    x: torch.Tensor, weight: torch.Tensor, weight_type: int
) -> torch.Tensor:
    return x @ dequant_gguf(weight, weight_type, x.dtype).T


class DiffusionGGUFLinearMethod(GGUFLinearMethod):
    """GGUF linear method using dequant+GEMM for N-D diffusion tensors."""

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shard_id = getattr(layer.weight, "shard_id", [])
        if shard_id:
            shard_id = ["q", "k", "v"] if "q" in shard_id else shard_id
            weight = layer.weight
            fallback_wtype = getattr(layer.weight_type, "weight_type", None)
            if fallback_wtype is None:
                fallback_wtype = next(
                    iter(layer.weight_type.shard_weight_type.values())
                )
            shard_weight_types = [
                layer.weight_type.shard_weight_type.get(idx, fallback_wtype)
                for idx in shard_id
            ]
            if len(set(shard_weight_types)) == 1:
                out = dequant_gemm_gguf(x, weight, shard_weight_types[0])
                if bias is not None:
                    out.add_(bias)
                return out
            result = []
            for idx in shard_id:
                start, end, offset = layer.weight.shard_offset_map[idx]
                weight_type = layer.weight_type.shard_weight_type.get(
                    idx, fallback_wtype
                )
                result.append(
                    dequant_gguf(
                        weight[start:end, :offset].contiguous(), weight_type, x.dtype
                    )
                )
            out = x @ torch.cat(result, dim=0).T
        else:
            weight = layer.weight
            weight_type = layer.weight_type.weight_type
            out = dequant_gemm_gguf(x, weight, weight_type)
        if bias is not None:
            out.add_(bias)
        return out


class DiffusionGGUFConfig(GGUFConfig):
    """GGUF config that carries gguf_model path and uses dequant+GEMM."""

    def __init__(
        self,
        gguf_model: str | None = None,
        unquantized_modules: list[str] | None = None,
    ) -> None:
        super().__init__(unquantized_modules=unquantized_modules or [])
        self.gguf_model = gguf_model

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            if any(module_name in prefix for module_name in self.unquantized_modules):
                return UnquantizedLinearMethod()
            if is_layer_skipped_gguf(
                prefix, self.unquantized_modules, self.packed_modules_mapping
            ):
                return UnquantizedLinearMethod()
            return DiffusionGGUFLinearMethod(self)
        return None
