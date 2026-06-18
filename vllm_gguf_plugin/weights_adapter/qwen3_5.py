# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import gguf
import torch

from ..quantization.nvfp4 import (
    iter_gguf_nvfp4_native_moe_weights,
    iter_gguf_nvfp4_native_weights,
)
from .default import GGUFWeightsAdapter

if TYPE_CHECKING:
    from vllm.config import ModelConfig

_QWEN3_5_PATCH_EMBED_WEIGHT = "model.visual.patch_embed.proj.weight"
_QWEN3_5_PATCH_EMBED_WEIGHT_1 = f"{_QWEN3_5_PATCH_EMBED_WEIGHT}.1"
_QWEN3_5_TOKEN_EMBD_WEIGHT = "token_embd.weight"
_QWEIGHT_SUFFIX = ".qweight"
_QWEIGHT_TYPE_SUFFIX = ".qweight_type"


def _get_text_config(config):
    if hasattr(config, "get_text_config"):
        return config.get_text_config()
    text_config = getattr(config, "text_config", None)
    return text_config if text_config is not None else config


def _qwen3_5_linear_attention_dims(config) -> tuple[int, int, int, int] | None:
    text_config = _get_text_config(config)
    num_k_heads = getattr(text_config, "linear_num_key_heads", None)
    num_v_heads = getattr(text_config, "linear_num_value_heads", None)
    head_k_dim = getattr(text_config, "linear_key_head_dim", None)
    head_v_dim = getattr(text_config, "linear_value_head_dim", None)
    if None in (num_k_heads, num_v_heads, head_k_dim, head_v_dim):
        return None

    num_k_heads = int(num_k_heads)
    num_v_heads = int(num_v_heads)
    head_k_dim = int(head_k_dim)
    head_v_dim = int(head_v_dim)
    if num_k_heads <= 0 or num_v_heads <= 0:
        return None
    if num_k_heads == num_v_heads:
        return None
    if num_v_heads % num_k_heads != 0:
        return None
    return num_k_heads, num_v_heads, head_k_dim, head_v_dim


def _tiled_to_grouped_v_heads(
    tensor: torch.Tensor,
    dim: int,
    num_k_heads: int,
    num_v_per_k: int,
    head_dim: int,
) -> torch.Tensor:
    shape = list(tensor.shape)
    if dim < 0:
        dim += len(shape)
    if shape[dim] != num_k_heads * num_v_per_k * head_dim:
        return tensor

    new_shape = shape[:dim] + [num_v_per_k, num_k_heads, head_dim] + shape[dim + 1 :]
    tensor = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return tensor.permute(*perm).contiguous().reshape(*shape)


def _maybe_restore_qwen3_5_gdn_layout(
    name: str,
    weight: torch.Tensor,
    config,
) -> torch.Tensor:
    if ".linear_attn." not in name:
        return weight

    dims = _qwen3_5_linear_attention_dims(config)
    if dims is None:
        return weight
    num_k_heads, num_v_heads, head_k_dim, head_v_dim = dims
    num_v_per_k = num_v_heads // num_k_heads
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim

    if ".linear_attn.in_proj_qkv" in name:
        if weight.shape[0] != 2 * key_dim + value_dim:
            return weight
        q = weight[:key_dim]
        k = weight[key_dim : 2 * key_dim]
        v = weight[2 * key_dim :]
        v = _tiled_to_grouped_v_heads(v, 0, num_k_heads, num_v_per_k, head_v_dim)
        return torch.cat((q, k, v), dim=0)

    if ".linear_attn.in_proj_z" in name:
        return _tiled_to_grouped_v_heads(
            weight, 0, num_k_heads, num_v_per_k, head_v_dim
        )

    if ".linear_attn.in_proj_a" in name or ".linear_attn.in_proj_b" in name:
        return _tiled_to_grouped_v_heads(weight, 0, num_k_heads, num_v_per_k, 1)

    if name.endswith(".linear_attn.dt_bias") or ".linear_attn.dt_proj" in name:
        if weight.dim() == 1:
            return _tiled_to_grouped_v_heads(
                weight.unsqueeze(-1), 0, num_k_heads, num_v_per_k, 1
            ).squeeze(-1)
        return _tiled_to_grouped_v_heads(weight, -1, num_k_heads, num_v_per_k, 1)

    if name.endswith(".linear_attn.A_log"):
        if torch.any(weight >= 0):
            raise ValueError(
                "Qwen3.5 GGUF A_log tensor is expected to store negative "
                "-exp(A_log) values"
            )
        restored = torch.log(-weight.to(torch.float32))
        if restored.dim() == 1:
            return _tiled_to_grouped_v_heads(
                restored.unsqueeze(-1), 0, num_k_heads, num_v_per_k, 1
            ).squeeze(-1)
        return _tiled_to_grouped_v_heads(restored, -1, num_k_heads, num_v_per_k, 1)

    if ".linear_attn.conv1d" in name:
        conv_weight = weight.squeeze(1) if weight.dim() == 3 else weight
        if conv_weight.shape[0] != 2 * key_dim + value_dim:
            return conv_weight[:, None, :] if conv_weight.dim() == 2 else conv_weight
        qk_part = conv_weight[: 2 * key_dim]
        v_part = conv_weight[2 * key_dim :]
        v_part = _tiled_to_grouped_v_heads(
            v_part, 0, num_k_heads, num_v_per_k, head_v_dim
        )
        return torch.cat((qk_part, v_part), dim=0)[:, None, :]

    if ".linear_attn.out_proj" in name:
        return _tiled_to_grouped_v_heads(
            weight, 1, num_k_heads, num_v_per_k, head_v_dim
        )

    return weight


def _maybe_reshape_qwen3_5_gguf_weight(
    name: str,
    weight: torch.Tensor,
) -> torch.Tensor:
    if "mlp.shared_expert_gate" in name and weight.dim() == 1:
        return weight[None, :]
    if "linear_attn.conv1d.weight" in name and weight.dim() == 2:
        return weight[:, None, :]
    return weight


def _maybe_restore_qwen3_5_norm_weight(
    name: str,
    weight: torch.Tensor,
) -> torch.Tensor:
    if not name.endswith("norm.weight") or name.endswith("linear_attn.norm.weight"):
        return weight
    text_norm_prefixes = (
        "model.layers.",
        "model.language_model.layers.",
    )
    text_norm_names = (
        "model.norm.weight",
        "model.language_model.norm.weight",
    )
    if name in text_norm_names or name.startswith(text_norm_prefixes):
        # llama.cpp stores Qwen3.5 text RMSNorm weights as weight + 1.
        return weight - 1
    return weight


def _dequantize_gguf_weight(
    weight: torch.Tensor,
    qweight_type: gguf.GGMLQuantizationType,
) -> torch.Tensor:
    dense = gguf.quants.dequantize(weight.detach().cpu().numpy(), qweight_type)
    return torch.from_numpy(dense.copy())


def _force_dequantized_gguf_weight(
    load_spec,
    forced_dequantized_modules: set[str],
    native_nvfp4_modules: set[str],
    gguf_name: str,
) -> None:
    if load_spec.gguf_to_hf_name_map is None:
        return
    hf_name = load_spec.gguf_to_hf_name_map.get(gguf_name)
    if hf_name is None or not hf_name.endswith(".weight"):
        return

    module_name = hf_name.removesuffix(".weight")
    forced_dequantized_modules.add(module_name)
    native_nvfp4_modules.discard(module_name)
    if hasattr(load_spec, "nvfp4_modules") and module_name in load_spec.nvfp4_modules:
        load_spec.nvfp4_modules.remove(module_name)
    if module_name not in load_spec.unquantized_modules:
        load_spec.unquantized_modules.append(module_name)


def _qwen3_5_embed_tokens_uses_quant_config() -> bool:
    import inspect

    try:
        from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
    except ImportError:
        return False

    init = Qwen3_5Model.__init__
    init_candidates = [init, inspect.unwrap(init)]
    for cell in getattr(init, "__closure__", ()) or ():
        try:
            cell_value = cell.cell_contents
        except ValueError:
            continue
        if inspect.isfunction(cell_value):
            init_candidates.append(inspect.unwrap(cell_value))

    for init_candidate in init_candidates:
        init_code = getattr(init_candidate, "__code__", None)
        if init_code is None:
            continue
        if "VocabParallelEmbedding" in init_code.co_names and (
            "quant_config" in init_code.co_names
            or "quant_config" in init_code.co_varnames
        ):
            return True
    return False


class Qwen3_5GGUFAdapter(GGUFWeightsAdapter):
    """Adapter for Qwen3.5 GGUF models."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self._forced_dequantized_modules: set[str] = set()
        self._qweight_types: dict[str, gguf.GGMLQuantizationType] = {}

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type in ("qwen3_5", "qwen3_5_moe", "qwen3_5_mtp")

    def prepare_loading(
        self,
        model_path: str,
        model_config: ModelConfig,
    ):
        load_spec = super().prepare_loading(model_path, model_config)
        self._forced_dequantized_modules.clear()
        # Older vLLM builds construct Qwen3.5 embed_tokens without quant_config,
        # so they need a dense compatibility fallback. Newer builds can keep
        # the GGUF token embedding packed and run it through GGUFEmbeddingMethod.
        if not _qwen3_5_embed_tokens_uses_quant_config():
            _force_dequantized_gguf_weight(
                load_spec,
                self._forced_dequantized_modules,
                self._native_nvfp4_modules,
                _QWEN3_5_TOKEN_EMBD_WEIGHT,
            )
        if _qwen3_5_linear_attention_dims(self.config) is None:
            return load_spec
        if load_spec.gguf_to_hf_name_map is None:
            return load_spec

        for hf_name in load_spec.gguf_to_hf_name_map.values():
            if hf_name.endswith(".linear_attn.out_proj.weight"):
                module_name = hf_name.removesuffix(".weight")
                self._forced_dequantized_modules.add(module_name)
                self._native_nvfp4_modules.discard(module_name)
                if (
                    hasattr(load_spec, "nvfp4_modules")
                    and module_name in load_spec.nvfp4_modules
                ):
                    load_spec.nvfp4_modules.remove(module_name)
                if module_name not in load_spec.unquantized_modules:
                    load_spec.unquantized_modules.append(module_name)
        return load_spec

    def map_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        self._qweight_types.clear()
        patch_weight: torch.Tensor | None = None
        patch_weight_1: torch.Tensor | None = None

        for hf_name, weight in weights:
            if hf_name.endswith(_QWEIGHT_TYPE_SUFFIX):
                module_name = hf_name.removesuffix(_QWEIGHT_TYPE_SUFFIX)
                qweight_type = gguf.GGMLQuantizationType(int(weight.item()))
                self._qweight_types[module_name] = qweight_type
                if (
                    module_name in self._forced_dequantized_modules
                    or module_name in self._native_nvfp4_modules
                    or module_name in self._native_nvfp4_moe_projection_modules
                ):
                    continue
                yield hf_name, weight
                continue

            if hf_name.endswith(_QWEIGHT_SUFFIX):
                module_name = hf_name.removesuffix(_QWEIGHT_SUFFIX)
                if module_name in self._forced_dequantized_modules:
                    qweight_type = self._qweight_types.get(module_name)
                    if qweight_type is None:
                        raise ValueError(
                            "Missing GGUF qweight_type for forced dense tensor "
                            f"{hf_name}"
                        )
                    hf_name = f"{module_name}.weight"
                    weight = _dequantize_gguf_weight(weight, qweight_type)
                elif module_name in self._native_nvfp4_moe_projection_modules:
                    native_weights = iter_gguf_nvfp4_native_moe_weights(
                        module_name,
                        weight,
                    )
                    for native_name, native_weight in native_weights:
                        yield (
                            native_name,
                            self.transform_weight(native_name, native_weight),
                        )
                    continue
                elif module_name in self._native_nvfp4_modules:
                    for native_name, native_weight in iter_gguf_nvfp4_native_weights(
                        module_name, weight
                    ):
                        yield (
                            native_name,
                            self.transform_weight(native_name, native_weight),
                        )
                    continue

            if hf_name == _QWEN3_5_PATCH_EMBED_WEIGHT:
                patch_weight = weight
                continue
            if hf_name == _QWEN3_5_PATCH_EMBED_WEIGHT_1:
                patch_weight_1 = weight
                continue
            yield hf_name, self.transform_weight(hf_name, weight)

        if patch_weight is None:
            if patch_weight_1 is not None:
                yield _QWEN3_5_PATCH_EMBED_WEIGHT_1, patch_weight_1
            return

        if patch_weight_1 is not None:
            patch_weight = torch.stack((patch_weight, patch_weight_1), dim=2)
        yield (
            _QWEN3_5_PATCH_EMBED_WEIGHT,
            self.transform_weight(_QWEN3_5_PATCH_EMBED_WEIGHT, patch_weight),
        )

    def transform_weight(
        self,
        hf_name: str,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        weight = _maybe_restore_qwen3_5_gdn_layout(hf_name, weight, self.config)
        weight = _maybe_restore_qwen3_5_norm_weight(hf_name, weight)
        return _maybe_reshape_qwen3_5_gguf_weight(hf_name, weight)
