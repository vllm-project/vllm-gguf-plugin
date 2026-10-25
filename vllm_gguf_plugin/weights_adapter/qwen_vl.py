# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.model_executor.models.utils import WeightsMapper

from ..gguf_files import GGUFModelFiles
from ..gguf_utils import maybe_patch_hf_config_from_gguf
from ..weight_utils import get_gguf_tensor_names
from .base import BaseGGUFWeightsAdapter, GGUFWeight

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig

logger = init_logger(__name__)

QWEN_VL_ARCHITECTURES = {
    "qwen2_vl": "Qwen2VLForConditionalGeneration",
    "qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration",
    "qwen3_vl": "Qwen3VLForConditionalGeneration",
}

QWEN_VL_TEXT_SUBSTR: dict[str, str] = {
    "attn_norm.": "input_layernorm.",
    "ffn_norm.": "post_attention_layernorm.",
    "attn_q_norm.": "self_attn.q_norm.",
    "attn_k_norm.": "self_attn.k_norm.",
    "attn_q.": "self_attn.q_proj.",
    "attn_k.": "self_attn.k_proj.",
    "attn_v.": "self_attn.v_proj.",
    "attn_output.": "self_attn.o_proj.",
    "ffn_gate.": "mlp.gate_proj.",
    "ffn_up.": "mlp.up_proj.",
    "ffn_down.": "mlp.down_proj.",
}

QWEN2_VL_VISION_SUBSTR: dict[str, str] = {
    "attn_q.": "attn.qkv.",
    "attn_k.": "attn.k_proj.",
    "attn_v.": "attn.v_proj.",
    "attn_out.": "attn.proj.",
    "ffn_up.": "mlp.fc2.",
    "ffn_down.": "mlp.fc1.",
    "ln1.": "norm1.",
    "ln2.": "norm2.",
}

QWEN25_VL_VISION_SUBSTR: dict[str, str] = {
    "attn_q.": "attn.qkv.",
    "attn_k.": "attn.k.",
    "attn_v.": "attn.v.",
    "attn_out.": "attn.proj.",
    "ffn_gate.": "mlp.gate_proj.",
    "ffn_up.": "mlp.up_proj.",
    "ffn_down.": "mlp.down_proj.",
    "ln1.": "norm1.",
    "ln2.": "norm2.",
}

QWEN3_VL_VISION_SUBSTR: dict[str, str] = {
    "attn_qkv.": "attn.qkv.",
    "attn_out.": "attn.proj.",
    "ffn_up.": "mlp.linear_fc1.",
    "ffn_down.": "mlp.linear_fc2.",
    "ln1.": "norm1.",
    "ln2.": "norm2.",
}


def build_qwen_vl_text_mapper() -> WeightsMapper:
    return WeightsMapper(
        orig_to_new_prefix={
            "token_embd.": "model.language_model.embed_tokens.",
            "blk.": "model.language_model.layers.",
            "output_norm.": "model.language_model.norm.",
            "output.": "lm_head.",
        },
        orig_to_new_substr=QWEN_VL_TEXT_SUBSTR,
    )


def build_qwen_vl_vision_mapper(model_type: str) -> WeightsMapper:
    if model_type == "qwen2_vl":
        vision_substr = QWEN2_VL_VISION_SUBSTR
    elif model_type == "qwen2_5_vl":
        vision_substr = QWEN25_VL_VISION_SUBSTR
    else:
        vision_substr = QWEN3_VL_VISION_SUBSTR

    if model_type == "qwen3_vl":
        merger_prefixes = {
            "v.position_embd.": "model.visual.pos_embed.",
            "v.post_ln.": "model.visual.merger.norm.",
            "mm.0.": "model.visual.merger.linear_fc1.",
            "mm.2.": "model.visual.merger.linear_fc2.",
        }
    else:
        merger_prefixes = {
            "v.post_ln.": "model.visual.merger.ln_q.",
            "mm.0.": "model.visual.merger.mlp.0.",
            "mm.2.": "model.visual.merger.mlp.2.",
        }

    return WeightsMapper(
        orig_to_new_prefix={
            "v.blk.": "model.visual.blocks.",
            "v.patch_embd.": "model.visual.patch_embed.proj.",
            **merger_prefixes,
        },
        orig_to_new_substr=vision_substr,
    )


def build_qwen3_vl_deepstack_mapper(
    deepstack_visual_indexes: Iterable[int],
) -> WeightsMapper:
    return WeightsMapper(
        orig_to_new_prefix={
            f"v.deepstack.{layer_index}.": (
                f"model.visual.deepstack_merger_list.{merger_index}."
            )
            for merger_index, layer_index in enumerate(deepstack_visual_indexes)
        },
        orig_to_new_substr={
            "fc1.": "linear_fc1.",
            "fc2.": "linear_fc2.",
        },
    )


def _map_tensor_name(mapper: WeightsMapper, name: str) -> str | None:
    mapped = mapper.apply_list([name])[0]
    return mapped if mapped != name else None


def _qkv_shard(name: str, model_type: str) -> tuple[str, str] | None:
    if not name.startswith("model.visual.") or not name.endswith((".weight", ".bias")):
        return None
    shard_names = (
        ("qkv", "k_proj", "v_proj") if model_type == "qwen2_vl" else ("qkv", "k", "v")
    )
    for shard_id, shard_name in zip("qkv", shard_names, strict=True):
        marker = f".attn.{shard_name}."
        if marker in name:
            return name.replace(marker, ".attn.qkv.", 1), shard_id
    return None


class QwenVLGGUFAdapter(BaseGGUFWeightsAdapter):
    """Adapter for Qwen2-VL, Qwen2.5-VL, and Qwen3-VL GGUF models."""

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type in QWEN_VL_ARCHITECTURES

    @classmethod
    def architecture(cls, config) -> str:
        return QWEN_VL_ARCHITECTURES[config.model_type]

    def patch_hf_config(
        self,
        files: GGUFModelFiles,
        hf_config: PretrainedConfig,
    ) -> PretrainedConfig:
        patched = maybe_patch_hf_config_from_gguf(
            files.primary_backbone,
            hf_config,
            mmproj_path=files.mm_proj,
        )
        if files.mm_proj is None:
            raise RuntimeError(
                "Could not find mm_proj for Qwen-VL GGUF. Place "
                "*mmproj*.gguf beside the backbone or pass "
                "model_loader_extra_config={'mm_proj': ...}."
            )
        patched.architectures = [QWEN_VL_ARCHITECTURES[patched.model_type]]
        return patched

    def build_name_map(
        self,
        files: GGUFModelFiles,
        model_config: ModelConfig,
    ) -> dict[str, str]:
        config = model_config.hf_config
        model_type = config.model_type
        text_mapper = build_qwen_vl_text_mapper()
        vision_mapper = build_qwen_vl_vision_mapper(model_type)
        deepstack_mapper = build_qwen3_vl_deepstack_mapper(
            getattr(config.vision_config, "deepstack_visual_indexes", ())
        )

        name_map: dict[str, str] = {}
        unmapped: list[str] = []
        for name in sorted(get_gguf_tensor_names(files.all_files)):
            if model_type != "qwen3_vl" and name == "v.position_embd.weight":
                continue
            if name.startswith("v.deepstack."):
                mapper = deepstack_mapper
            elif name.startswith(("v.", "mm.")):
                mapper = vision_mapper
            else:
                mapper = text_mapper
            if mapped := _map_tensor_name(mapper, name):
                name_map[name] = mapped
            else:
                unmapped.append(name)
        if unmapped:
            logger.warning(
                "No HF name for %d Qwen-VL GGUF tensor(s), skipping: %s",
                len(unmapped),
                unmapped,
            )
        return name_map

    def transform_weights(
        self,
        weights: Iterable[GGUFWeight],
        model_config: ModelConfig,
    ) -> Iterable[GGUFWeight]:
        config = model_config.hf_config
        model_type = config.model_type
        temporal_patch_size = config.vision_config.temporal_patch_size
        patch_embed_parts: dict[str, dict[int, torch.Tensor]] = {}
        qkv_parts: dict[str, dict[str, torch.Tensor]] = {}

        for name, weight in weights:
            if model_type != "qwen3_vl" and (shard := _qkv_shard(name, model_type)):
                fused_name, shard_id = shard
                parts = qkv_parts.setdefault(fused_name, {})
                parts[shard_id] = weight
                if all(part in parts for part in "qkv"):
                    yield fused_name, torch.cat([parts[part] for part in "qkv"], dim=0)
                    del qkv_parts[fused_name]
                continue
            patch_marker = "patch_embed.proj.weight"
            if temporal_patch_size > 1 and patch_marker in name:
                suffix = name.partition(patch_marker)[2]
                if suffix and not (suffix.startswith(".") and suffix[1:].isdigit()):
                    yield name, weight
                    continue
                key = name.removesuffix(suffix)
                index = int(suffix[1:]) if suffix else 0
                parts = patch_embed_parts.setdefault(key, {})
                parts[index] = weight
                if not all(part in parts for part in range(temporal_patch_size)):
                    continue
                yield (
                    key,
                    torch.stack(
                        [parts[part] for part in range(temporal_patch_size)], dim=2
                    ),
                )
                del patch_embed_parts[key]
                continue
            yield name, weight

        if qkv_parts:
            raise RuntimeError(f"Incomplete Qwen-VL QKV tensors: {sorted(qkv_parts)}")
        if patch_embed_parts:
            raise RuntimeError(
                "Incomplete Qwen-VL temporal patch embedding tensors: "
                f"{sorted(patch_embed_parts)}"
            )
