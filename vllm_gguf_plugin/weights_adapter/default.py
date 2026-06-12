# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

import gguf
import regex
import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
from vllm.logger import init_logger

from ..gguf_utils import detect_gguf_multimodal, maybe_patch_hf_config_from_gguf
from ..weight_utils import (
    get_gguf_extra_tensor_names,
    get_gguf_weight_type_map,
    gguf_quant_weights_iterator_multi,
)
from .base import BaseGGUFWeightsAdapter, GGUFLoadSpec

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig

logger = init_logger(__name__)

_GGUF_MODEL_TYPE_ALIASES = {
    "gemma4": "gemma3",
}


def _gguf_arch_model_type(model_type: str) -> str:
    return _GGUF_MODEL_TYPE_ALIASES.get(model_type, model_type)


def _gguf_name_with_suffix(base_name: str, suffix: str) -> str:
    return f"{base_name}.{suffix}" if suffix else base_name


def _get_vision_num_layers(config: PretrainedConfig) -> int:
    vision_num_layers = getattr(config.vision_config, "num_hidden_layers", None)
    if vision_num_layers is None:
        vision_num_layers = config.vision_config.depth
    return vision_num_layers


def _is_multimodal_config(config: PretrainedConfig) -> bool:
    return hasattr(config, "vision_config") and config.vision_config is not None


def _uses_multimodal_weight_layout(
    config: PretrainedConfig,
    architectures: list[str] | None = None,
) -> bool:
    if not _is_multimodal_config(config):
        return False

    if architectures is None:
        architectures = getattr(config, "architectures", None)
    if not architectures:
        return True

    return any("ConditionalGeneration" in arch for arch in architectures)


def _is_gemma4_mtp_config(config: PretrainedConfig) -> bool:
    return config.model_type in ("gemma4_assistant", "gemma4_mtp")


def _get_mtp_num_layers(text_config: PretrainedConfig) -> int:
    return int(
        getattr(
            text_config,
            "mtp_num_hidden_layers",
            getattr(text_config, "num_nextn_predict_layers", 0),
        )
        or 0
    )


def _add_gemma4_mtp_gguf_mappings(
    config: PretrainedConfig,
    gguf_to_hf_name_map: dict[str, str],
) -> None:
    text_config = config.get_text_config()
    num_layers = int(getattr(text_config, "num_hidden_layers", 0) or 0)

    gguf_to_hf_name_map.update(
        {
            "token_embd.weight": "model.embed_tokens.weight",
            "output_norm.weight": "model.norm.weight",
            "nextn.pre_projection.weight": "model.pre_projection.weight",
            "nextn.post_projection.weight": "model.post_projection.weight",
        }
    )

    # Gemma4 MTP attention is Q-only and reuses the target model KV cache,
    # so GGUF assistant blocks intentionally do not map attn_k/attn_v.
    for idx in range(num_layers):
        layer_prefix = f"model.layers.{idx}"
        gguf_to_hf_name_map.update(
            {
                f"blk.{idx}.attn_norm.weight": (
                    f"{layer_prefix}.input_layernorm.weight"
                ),
                f"blk.{idx}.attn_q.weight": (f"{layer_prefix}.self_attn.q_proj.weight"),
                f"blk.{idx}.attn_output.weight": (
                    f"{layer_prefix}.self_attn.o_proj.weight"
                ),
                f"blk.{idx}.attn_q_norm.weight": (
                    f"{layer_prefix}.self_attn.q_norm.weight"
                ),
                f"blk.{idx}.post_attention_norm.weight": (
                    f"{layer_prefix}.post_attention_layernorm.weight"
                ),
                f"blk.{idx}.ffn_norm.weight": (
                    f"{layer_prefix}.pre_feedforward_layernorm.weight"
                ),
                f"blk.{idx}.post_ffw_norm.weight": (
                    f"{layer_prefix}.post_feedforward_layernorm.weight"
                ),
                f"blk.{idx}.ffn_gate.weight": (f"{layer_prefix}.mlp.gate_proj.weight"),
                f"blk.{idx}.ffn_up.weight": f"{layer_prefix}.mlp.up_proj.weight",
                f"blk.{idx}.ffn_down.weight": (f"{layer_prefix}.mlp.down_proj.weight"),
                f"blk.{idx}.layer_output_scale.weight": (
                    f"{layer_prefix}.layer_scalar"
                ),
            }
        )


def _add_qwen3_5_mtp_gguf_mappings(
    config: PretrainedConfig,
    gguf_to_hf_name_map: dict[str, str],
    sideload_params: list[re.Pattern],
) -> None:
    text_config = config.get_text_config()
    num_mtp_layers = _get_mtp_num_layers(text_config)
    if num_mtp_layers <= 0:
        return

    base_layer = int(getattr(text_config, "num_hidden_layers", 0) or 0)
    shared_nextn_gguf_idx = base_layer
    gguf_to_hf_name_map.update(
        {
            f"blk.{shared_nextn_gguf_idx}.nextn.eh_proj.weight": "mtp.fc.weight",
            f"blk.{shared_nextn_gguf_idx}.nextn.enorm.weight": (
                "mtp.pre_fc_norm_embedding.weight"
            ),
            f"blk.{shared_nextn_gguf_idx}.nextn.hnorm.weight": (
                "mtp.pre_fc_norm_hidden.weight"
            ),
            f"blk.{shared_nextn_gguf_idx}.nextn.shared_head_norm.weight": (
                "mtp.norm.weight"
            ),
            f"blk.{shared_nextn_gguf_idx}.nextn.embed_tokens.weight": (
                "mtp.embed_tokens.weight"
            ),
        }
    )
    for mtp_idx in range(num_mtp_layers):
        gguf_idx = base_layer + mtp_idx
        layer_prefix = f"mtp.layers.{mtp_idx}"
        gguf_to_hf_name_map.update(
            {
                f"blk.{gguf_idx}.attn_norm.weight": (
                    f"{layer_prefix}.input_layernorm.weight"
                ),
                f"blk.{gguf_idx}.post_attention_norm.weight": (
                    f"{layer_prefix}.post_attention_layernorm.weight"
                ),
                f"blk.{gguf_idx}.attn_q.weight": (
                    f"{layer_prefix}.self_attn.q_proj.weight"
                ),
                f"blk.{gguf_idx}.attn_k.weight": (
                    f"{layer_prefix}.self_attn.k_proj.weight"
                ),
                f"blk.{gguf_idx}.attn_v.weight": (
                    f"{layer_prefix}.self_attn.v_proj.weight"
                ),
                f"blk.{gguf_idx}.attn_output.weight": (
                    f"{layer_prefix}.self_attn.o_proj.weight"
                ),
                f"blk.{gguf_idx}.attn_q_norm.weight": (
                    f"{layer_prefix}.self_attn.q_norm.weight"
                ),
                f"blk.{gguf_idx}.attn_k_norm.weight": (
                    f"{layer_prefix}.self_attn.k_norm.weight"
                ),
                f"blk.{gguf_idx}.ffn_gate_inp.weight": (
                    f"{layer_prefix}.mlp.gate.weight"
                ),
                f"blk.{gguf_idx}.ffn_gate_inp_shexp.weight": (
                    f"{layer_prefix}.mlp.shared_expert_gate.weight"
                ),
                f"blk.{gguf_idx}.ffn_gate_shexp.weight": (
                    f"{layer_prefix}.mlp.shared_expert.gate_proj.weight"
                ),
                f"blk.{gguf_idx}.ffn_up_shexp.weight": (
                    f"{layer_prefix}.mlp.shared_expert.up_proj.weight"
                ),
                f"blk.{gguf_idx}.ffn_down_shexp.weight": (
                    f"{layer_prefix}.mlp.shared_expert.down_proj.weight"
                ),
                f"blk.{gguf_idx}.ffn_gate_exps.weight": (
                    f"{layer_prefix}.mlp.experts.0.gate_proj.weight"
                ),
                f"blk.{gguf_idx}.ffn_up_exps.weight": (
                    f"{layer_prefix}.mlp.experts.0.up_proj.weight"
                ),
                f"blk.{gguf_idx}.ffn_down_exps.weight": (
                    f"{layer_prefix}.mlp.experts.0.down_proj.weight"
                ),
            }
        )
        sideload_params.append(
            regex.compile(
                f"mtp\\.layers\\.{mtp_idx}"
                r"\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
            )
        )


def _add_gemma4_gguf_mappings(
    config: PretrainedConfig,
    gguf_to_hf_name_map: dict[str, str],
    sideload_params: list[re.Pattern],
) -> None:
    text_config = config.get_text_config()
    uses_multimodal_layout = _uses_multimodal_weight_layout(config)
    layer_prefix_base = (
        "model.language_model.layers" if uses_multimodal_layout else "model.layers"
    )
    for idx in range(text_config.num_hidden_layers):
        layer_prefix = f"{layer_prefix_base}.{idx}"
        gguf_to_hf_name_map[f"blk.{idx}.layer_output_scale.weight"] = (
            f"{layer_prefix}.layer_scalar"
        )
        gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_inp.scale"] = (
            f"{layer_prefix}.router.scale"
        )
        gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_inp.weight"] = (
            f"{layer_prefix}.router.proj.weight"
        )
        gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.scale"] = (
            f"{layer_prefix}.router.per_expert_scale"
        )
        gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_up_exps.weight"] = (
            f"{layer_prefix}.experts.gate_up_proj.weight"
        )
        gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
            f"{layer_prefix}.experts.down_proj.weight"
        )
        gguf_to_hf_name_map[f"blk.{idx}.post_ffw_norm_1.weight"] = (
            f"{layer_prefix}.post_feedforward_layernorm_1.weight"
        )
        gguf_to_hf_name_map[f"blk.{idx}.post_ffw_norm_2.weight"] = (
            f"{layer_prefix}.post_feedforward_layernorm_2.weight"
        )
        gguf_to_hf_name_map[f"blk.{idx}.pre_ffw_norm_2.weight"] = (
            f"{layer_prefix}.pre_feedforward_layernorm_2.weight"
        )
        sideload_params.extend(
            [
                regex.compile(
                    f"{re.escape(layer_prefix_base)}\\.{idx}"
                    r"\.experts\.(gate_up_proj|down_proj)(\.weight)?"
                ),
            ]
        )

    if uses_multimodal_layout:
        gguf_to_hf_name_map.update(
            {
                "v.std_bias": "model.vision_tower.std_bias",
                "v.std_scale": "model.vision_tower.std_scale",
                "v.patch_embd.weight": (
                    "model.vision_tower.patch_embedder.input_proj.weight"
                ),
                "v.position_embd.weight": (
                    "model.vision_tower.patch_embedder.position_embedding_table"
                ),
                "mm.input_projection.weight": (
                    "model.embed_vision.embedding_projection.weight"
                ),
            }
        )

        for idx in range(_get_vision_num_layers(config)):
            vision_prefix = f"model.vision_tower.encoder.layers.{idx}"
            gguf_to_hf_name_map.update(
                {
                    f"v.blk.{idx}.attn_q.weight": (
                        f"{vision_prefix}.self_attn.q_proj.linear.weight"
                    ),
                    f"v.blk.{idx}.attn_k.weight": (
                        f"{vision_prefix}.self_attn.k_proj.linear.weight"
                    ),
                    f"v.blk.{idx}.attn_v.weight": (
                        f"{vision_prefix}.self_attn.v_proj.linear.weight"
                    ),
                    f"v.blk.{idx}.attn_out.weight": (
                        f"{vision_prefix}.self_attn.o_proj.linear.weight"
                    ),
                    f"v.blk.{idx}.attn_q_norm.weight": (
                        f"{vision_prefix}.self_attn.q_norm.weight"
                    ),
                    f"v.blk.{idx}.attn_k_norm.weight": (
                        f"{vision_prefix}.self_attn.k_norm.weight"
                    ),
                    f"v.blk.{idx}.ffn_gate.weight": (
                        f"{vision_prefix}.mlp.gate_proj.linear.weight"
                    ),
                    f"v.blk.{idx}.ffn_up.weight": (
                        f"{vision_prefix}.mlp.up_proj.linear.weight"
                    ),
                    f"v.blk.{idx}.ffn_down.weight": (
                        f"{vision_prefix}.mlp.down_proj.linear.weight"
                    ),
                    f"v.blk.{idx}.ln1.weight": (
                        f"{vision_prefix}.input_layernorm.weight"
                    ),
                    f"v.blk.{idx}.attn_post_norm.weight": (
                        f"{vision_prefix}.post_attention_layernorm.weight"
                    ),
                    f"v.blk.{idx}.ln2.weight": (
                        f"{vision_prefix}.pre_feedforward_layernorm.weight"
                    ),
                    f"v.blk.{idx}.ffn_post_norm.weight": (
                        f"{vision_prefix}.post_feedforward_layernorm.weight"
                    ),
                }
            )


class GGUFWeightsAdapter(BaseGGUFWeightsAdapter):
    """Default adapter for GGUF models."""

    load_spec = None

    @classmethod
    def matches(cls, config) -> bool:
        del config
        return True

    def patch_hf_config(self, model_path: str, hf_config: PretrainedConfig):
        return maybe_patch_hf_config_from_gguf(model_path, hf_config)

    def build_name_map(self, model_config: ModelConfig) -> dict[str, str]:
        config = model_config.hf_config
        text_config = config.get_text_config()
        model_type = config.model_type
        is_multimodal = _uses_multimodal_weight_layout(config)
        orig_model_type = model_type

        gguf_to_hf_name_map: dict[str, str] = {}
        sideload_params: list[re.Pattern] = []

        if _is_gemma4_mtp_config(config):
            _add_gemma4_mtp_gguf_mappings(config, gguf_to_hf_name_map)
            return gguf_to_hf_name_map

        if model_type == "cohere":
            model_type = "command-r"
        if model_type == "gemma3_text":
            model_type = "gemma3"
        if model_type == "gemma4":
            _add_gemma4_gguf_mappings(config, gguf_to_hf_name_map, sideload_params)
        if model_type in ("deepseek_v3", "deepseek_v2"):
            model_type = "deepseek2"
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.exp_probs_b.bias"] = (
                    f"model.layers.{idx}.mlp.gate.e_score_correction_bias"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
                )
                sideload_params.append(
                    regex.compile(
                        f"model\\.layers\\.{idx}"
                        r"\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
                    )
                )
        if model_type == "qwen3_5":
            model_type = "qwen35"
        if model_type in ("qwen2_moe", "qwen3_moe", "qwen3_5_moe"):
            model_type = model_type.replace("_", "")
            if is_multimodal and model_type == "qwen35moe":
                layer_prefix = "model.language_model.layers"
            else:
                layer_prefix = "model.layers"
            for idx in range(text_config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"{layer_prefix}.{idx}.mlp.experts.0.down_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"{layer_prefix}.{idx}.mlp.experts.0.gate_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"{layer_prefix}.{idx}.mlp.experts.0.up_proj.weight"
                )
                sideload_params.append(
                    regex.compile(
                        f"{re.escape(layer_prefix)}\\.{idx}"
                        r"\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
                    )
                )
        if orig_model_type in ("qwen3_5", "qwen3_5_moe"):
            layer_prefix = (
                "model.language_model.layers" if is_multimodal else "model.layers"
            )
            layer_types = getattr(text_config, "layer_types", [])
            for idx, layer_type in enumerate(layer_types):
                if layer_type == "linear_attention":
                    gguf_to_hf_name_map[f"blk.{idx}.ssm_dt.bias"] = (
                        f"{layer_prefix}.{idx}.linear_attn.dt_bias"
                    )
            _add_qwen3_5_mtp_gguf_mappings(config, gguf_to_hf_name_map, sideload_params)
        if orig_model_type in ("qwen3_5", "qwen3_5_moe") and is_multimodal:
            gguf_to_hf_name_map.update(
                {
                    "v.patch_embd.weight.1": ("model.visual.patch_embed.proj.weight.1"),
                    "v.post_ln.weight": "model.visual.merger.norm.weight",
                    "v.post_ln.bias": "model.visual.merger.norm.bias",
                    "mm.0.weight": "model.visual.merger.linear_fc1.weight",
                    "mm.0.bias": "model.visual.merger.linear_fc1.bias",
                    "mm.2.weight": "model.visual.merger.linear_fc2.weight",
                    "mm.2.bias": "model.visual.merger.linear_fc2.bias",
                }
            )
            sideload_params.extend(
                [
                    regex.compile(r"model\.visual\.merger\.norm\.weight"),
                    regex.compile(r"model\.visual\.merger\.norm\.bias"),
                ]
            )
        if model_type == "olmoe":
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
                )
                sideload_params.extend(
                    [
                        regex.compile(
                            f"model\\.layers\\.{idx}"
                            r"\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
                        ),
                        regex.compile(
                            f"model\\.layers\\.{idx}"
                            r"\.mlp\.experts\.(gate_up_proj|down_proj)"
                        ),
                    ]
                )
        if model_type == "minimax_m2":
            model_type = "minimax-m2"
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.exp_probs_b.bias"] = (
                    f"model.layers.{idx}.block_sparse_moe.e_score_correction_bias"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.block_sparse_moe.experts.0.w2.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.block_sparse_moe.experts.0.w1.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.block_sparse_moe.experts.0.w3.weight"
                )
                sideload_params.append(
                    regex.compile(
                        f"model\\.layers\\.{idx}"
                        r"\.block_sparse_moe\.experts\.(gate_up_proj|down_proj)"
                    )
                )

        arch = None
        for key, value in gguf.MODEL_ARCH_NAMES.items():
            if value == _gguf_arch_model_type(model_type):
                arch = key
                break
        if arch is None:
            raise RuntimeError(f"Unknown gguf model_type: {model_type}")

        text_name_map = gguf.get_tensor_name_map(arch, text_config.num_hidden_layers)

        if is_multimodal:
            mm_proj_arch = gguf.MODEL_ARCH.MMPROJ
            vision_name_map = gguf.get_tensor_name_map(
                mm_proj_arch, _get_vision_num_layers(config)
            )
        else:
            vision_name_map = None

        with torch.device("meta"):
            auto_cls = (
                AutoModelForImageTextToText if is_multimodal else AutoModelForCausalLM
            )
            auto_config = config if is_multimodal else text_config
            dummy_model = auto_cls.from_config(
                auto_config, trust_remote_code=model_config.trust_remote_code
            )

        state_dict = dummy_model.state_dict()
        if hf_checkpoint_map := getattr(
            dummy_model, "_checkpoint_conversion_mapping", None
        ):

            def revert_hf_rename(name: str) -> str:
                for original_name, hf_name in hf_checkpoint_map.items():
                    if hf_name in name:
                        name = name.replace(hf_name, original_name).lstrip("^")
                return name

            state_dict = {
                revert_hf_rename(name): tensor for name, tensor in state_dict.items()
            }

        if model_type == "minimax-m2" and not hf_checkpoint_map:
            state_dict = {
                name.replace(".mlp.", ".block_sparse_moe."): tensor
                for name, tensor in state_dict.items()
            }

        def find_hf_name_in_tensor_map(hf_name: str) -> str | None:
            if is_multimodal and hf_name.startswith("model."):
                hf_name = hf_name[6:]
            if hf_name.startswith("language_model."):
                hf_name = hf_name[15:]
                if is_multimodal:
                    hf_name = "model." + hf_name
            if hf_name.endswith((".weight", ".bias")):
                base_name, suffix = hf_name.rsplit(".", 1)
            else:
                base_name, suffix = hf_name, ""
                if base_name.endswith("_weight"):
                    base_name = base_name[:-7]
                    suffix = "weight"
            gguf_name = None
            if vision_name_map is not None:
                gguf_name = vision_name_map.get_name(base_name)
            if gguf_name is None:
                gguf_name = text_name_map.get_name(base_name)
            if gguf_name is None:
                return None
            return _gguf_name_with_suffix(gguf_name, suffix)

        unmapped_params = []
        for hf_name in state_dict:
            gguf_name_with_suffix = find_hf_name_in_tensor_map(hf_name)
            if gguf_name_with_suffix is not None:
                gguf_to_hf_name_map[gguf_name_with_suffix] = hf_name
                logger.debug("Mapped GGUF %s → HF %s", gguf_name_with_suffix, hf_name)
            elif hf_name not in gguf_to_hf_name_map.values():
                unmapped_params.append(hf_name)

        if unmapped_params:
            unmapped_params = [
                x
                for x in unmapped_params
                if not any(regex.fullmatch(p, x) for p in sideload_params)
            ]
        if unmapped_params:
            raise RuntimeError(
                f"Failed to map GGUF parameters "
                f"({len(unmapped_params)}): {unmapped_params}"
            )
        return gguf_to_hf_name_map

    def map_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for hf_name, weight in weights:
            yield hf_name, self.transform_weight(hf_name, weight)

    @staticmethod
    def _get_all_gguf_files(model_path: str) -> list[str]:
        match = re.search(r"-(\d+)-of-(\d+)\.gguf$", model_path)
        if not match:
            return [model_path]
        total = int(match.group(2))
        num_digits = len(match.group(1))
        prefix = model_path[: match.start(1)]
        suffix = model_path[match.end(2) :]
        files = []
        for i in range(1, total + 1):
            shard_path = f"{prefix}{i:0{num_digits}d}-of-{total:0{num_digits}d}{suffix}"
            if os.path.isfile(shard_path):
                files.append(shard_path)
        if files:
            logger.info("Discovered %d GGUF shard files", len(files))
        return files if files else [model_path]

    def _get_weight_sources(
        self,
        model_path: str,
        hf_config: PretrainedConfig,
        use_multimodal_weight_layout: bool | None = None,
    ) -> list[str]:
        gguf_files = self._get_all_gguf_files(model_path)
        if use_multimodal_weight_layout is None:
            use_multimodal_weight_layout = _uses_multimodal_weight_layout(hf_config)
        if use_multimodal_weight_layout:
            mm_proj_path = detect_gguf_multimodal(model_path)
            if mm_proj_path is not None:
                mm_proj_file = os.fspath(mm_proj_path)
                if mm_proj_file not in gguf_files:
                    gguf_files.append(mm_proj_file)
        return gguf_files

    def update_tie_word_embeddings(
        self,
        gguf_files: list[str],
        hf_config: PretrainedConfig,
        gguf_to_hf_name_map: dict[str, str],
    ) -> None:
        if "lm_head.weight" not in gguf_to_hf_name_map.values():
            return

        all_extra_names = []
        for gguf_file in gguf_files:
            all_extra_names.extend(
                get_gguf_extra_tensor_names(gguf_file, gguf_to_hf_name_map)
            )
        hf_config.update({"tie_word_embeddings": "lm_head.weight" in all_extra_names})

    def get_weight_type_map(
        self,
        gguf_files: list[str],
        gguf_to_hf_name_map: dict[str, str],
    ) -> dict[str, str]:
        weight_type_map = {}
        for gguf_file in gguf_files:
            weight_type_map.update(
                get_gguf_weight_type_map(gguf_file, gguf_to_hf_name_map)
            )
        return weight_type_map

    @staticmethod
    def get_unquantized_modules(weight_type_map: dict[str, str]) -> list[str]:
        return [
            name.removesuffix(".weight")
            for name, weight_type in weight_type_map.items()
            if weight_type in ("F32", "F16", "BF16") and name.endswith(".weight")
        ]

    def prepare_loading(
        self,
        model_path: str,
        model_config: ModelConfig,
    ) -> GGUFLoadSpec:
        model_config.hf_config = self.patch_hf_config(
            model_path, model_config.hf_config
        )
        gguf_to_hf_name_map = self.build_name_map(model_config)
        use_multimodal_weight_layout = _uses_multimodal_weight_layout(
            model_config.hf_config
        )
        gguf_files = self._get_weight_sources(
            model_path,
            model_config.hf_config,
            use_multimodal_weight_layout,
        )
        self.update_tie_word_embeddings(
            gguf_files, model_config.hf_config, gguf_to_hf_name_map
        )
        weight_type_map = self.get_weight_type_map(gguf_files, gguf_to_hf_name_map)
        self.load_spec = GGUFLoadSpec(
            weights_source=gguf_files,
            gguf_to_hf_name_map=gguf_to_hf_name_map,
            unquantized_modules=self.get_unquantized_modules(weight_type_map),
        )
        return self.load_spec

    def prepare_weights(
        self,
        model_config: ModelConfig,
    ) -> Iterable[tuple[str, torch.Tensor]]:
        del model_config
        weights = gguf_quant_weights_iterator_multi(
            self.load_spec.weights_source,
            self.load_spec.gguf_to_hf_name_map,
        )
        yield from self.map_weights(weights)
