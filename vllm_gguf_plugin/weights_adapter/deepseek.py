# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from vllm.model_executor.models.utils import WeightsMapper

from ..gguf_utils import maybe_patch_hf_config_from_gguf
from ..weight_utils import (
    get_gguf_unquantized_params,
    gguf_quant_weights_iterator_multi,
)
from .base import BaseGGUFWeightsAdapter, GGUFLoadSpec

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig


def build_deepseek_mapper(model_type: str) -> WeightsMapper:
    if model_type not in {"deepseek_v2", "deepseek_v3"}:
        raise ValueError(f"Unsupported DeepSeek model type: {model_type}")

    common_substr = {
        "attn_kv_a_mqa.": "self_attn.kv_a_proj_with_mqa.",
        "attn_kv_a_norm.": "self_attn.kv_a_layernorm.",
        "attn_k_b.": "self_attn.k_b_proj.",
        "attn_v_b.": "self_attn.v_b_proj.",
        "attn_output.": "self_attn.o_proj.",
        "attn_norm.": "input_layernorm.",
        "ffn_norm.": "post_attention_layernorm.",
        "ffn_gate.": "mlp.gate_proj.",
        "ffn_up.": "mlp.up_proj.",
        "ffn_down.": "mlp.down_proj.",
        "ffn_gate_exps.": "mlp.experts.0.gate_proj.",
        "ffn_up_exps.": "mlp.experts.0.up_proj.",
        "ffn_down_exps.": "mlp.experts.0.down_proj.",
        "ffn_gate_inp.": "mlp.gate.",
        "ffn_gate_shexp.": "mlp.shared_experts.gate_proj.",
        "ffn_up_shexp.": "mlp.shared_experts.up_proj.",
        "ffn_down_shexp.": "mlp.shared_experts.down_proj.",
    }
    if model_type == "deepseek_v2":
        orig_to_new_substr = {
            **common_substr,
            "attn_q.": "self_attn.q_proj.",
        }
    if model_type == "deepseek_v3":
        orig_to_new_substr = {
            **common_substr,
            "attn_q_a.": "self_attn.q_a_proj.",
            "attn_q_a_norm.": "self_attn.q_a_layernorm.",
            "attn_q_b.": "self_attn.q_b_proj.",
        }

    return WeightsMapper(
        orig_to_new_prefix={
            "token_embd.": "model.embed_tokens.",
            "blk.": "model.layers.",
            "output_norm.": "model.norm.",
            "output.": "lm_head.",
        },
        orig_to_new_substr=orig_to_new_substr,
    )


def split_deepseek_expert_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    for name, weight in weights:
        if weight.ndim == 3 and ".experts.0." in name:
            for expert_id, expert_weight in enumerate(weight.unbind(0)):
                yield (
                    name.replace(
                        ".experts.0.",
                        f".experts.{expert_id}.",
                    ),
                    expert_weight,
                )
        else:
            yield name, weight


def map_deepseek_router_bias(name: str) -> str:
    return name.replace(
        ".exp_probs_b.bias",
        ".mlp.gate.e_score_correction_bias",
    )


def filter_deepseek_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    num_hidden_layers: int,
) -> Iterable[tuple[str, torch.Tensor]]:
    layer_pattern = re.compile(r"^model\.layers\.(\d+)(?:\.|$)")
    for name, weight in weights:
        match = layer_pattern.match(name)
        if match and int(match.group(1)) >= num_hidden_layers:
            continue
        yield name, weight


class DeepSeekGGUFAdapter(BaseGGUFWeightsAdapter):
    """Adapter for DeepSeek GGUF models."""

    mapper = None
    load_spec = None

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type in {"deepseek_v2", "deepseek_v3"}

    def patch_hf_config(self, model_path: str, hf_config: PretrainedConfig):
        return maybe_patch_hf_config_from_gguf(model_path, hf_config)

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
        for index in range(1, total + 1):
            shard_path = (
                f"{prefix}{index:0{num_digits}d}-of-{total:0{num_digits}d}{suffix}"
            )
            if os.path.isfile(shard_path):
                files.append(shard_path)
        return files or [model_path]

    def prepare_loading(
        self,
        model_path: str,
        model_config: ModelConfig,
    ) -> GGUFLoadSpec:
        model_config.hf_config = self.patch_hf_config(
            model_path,
            model_config.hf_config,
        )
        gguf_files = self._get_all_gguf_files(model_path)
        self.mapper = build_deepseek_mapper(model_config.hf_config.model_type)
        unquantized_params = get_gguf_unquantized_params(gguf_files)
        unquantized_modules = list(
            {
                param.rsplit(".", 1)[0] if param.endswith(".weight") else param
                for param in self.mapper.apply_list(unquantized_params)
            }
        )
        self.load_spec = GGUFLoadSpec(
            weights_source=gguf_files,
            unquantized_modules=unquantized_modules,
        )
        return self.load_spec

    def prepare_weights(
        self,
        model_config: ModelConfig,
    ) -> Iterable[tuple[str, torch.Tensor]]:
        if self.mapper is None or self.load_spec is None:
            raise RuntimeError("prepare_loading must be called before prepare_weights")
        orig_weights = gguf_quant_weights_iterator_multi(
            self.load_spec.weights_source,
        )
        mapped_weights = (
            (map_deepseek_router_bias(name), weight)
            for name, weight in self.mapper.apply(orig_weights)
        )
        filtered_weights = filter_deepseek_weights(
            mapped_weights,
            model_config.hf_config.num_hidden_layers,
        )
        yield from split_deepseek_expert_weights(
            filtered_weights,
        )
