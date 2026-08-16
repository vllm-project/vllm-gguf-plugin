# SPDX-License-Identifier: Apache-2.0

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


def build_olmoe_mapper() -> WeightsMapper:
    return WeightsMapper(
        orig_to_new_prefix={
            "token_embd.": "model.embed_tokens.",
            "blk.": "model.layers.",
            "output_norm.": "model.norm.",
            "output.": "lm_head.",
        },
        orig_to_new_substr={
            "attn_q.": "self_attn.q_proj.",
            "attn_k.": "self_attn.k_proj.",
            "attn_v.": "self_attn.v_proj.",
            "attn_output.": "self_attn.o_proj.",
            "attn_q_norm.": "self_attn.q_norm.",
            "attn_k_norm.": "self_attn.k_norm.",
            "attn_norm.": "input_layernorm.",
            "ffn_norm.": "post_attention_layernorm.",
            "ffn_gate_inp.": "mlp.gate.",
            "ffn_down_exps.": "mlp.experts.0.down_proj.",
            "ffn_gate_exps.": "mlp.experts.0.gate_proj.",
            "ffn_up_exps.": "mlp.experts.0.up_proj.",
        },
    )


def split_olmoe_expert_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    for name, weight in weights:
        if weight.ndim == 3 and ".experts.0." in name:
            for expert_id, expert_weight in enumerate(weight.unbind()):
                yield (
                    name.replace(
                        ".experts.0.",
                        f".experts.{expert_id}.",
                    ),
                    expert_weight,
                )
        else:
            yield name, weight


class OLMoEGGUFAdapter(BaseGGUFWeightsAdapter):
    """Adapter for OLMoE GGUF models."""

    mapper = None
    load_spec = None

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type == "olmoe"

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
        self.mapper = build_olmoe_mapper()
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
        del model_config
        if self.mapper is None or self.load_spec is None:
            raise RuntimeError("prepare_loading must be called before prepare_weights")
        orig_weights = gguf_quant_weights_iterator_multi(self.load_spec.weights_source)
        yield from split_olmoe_expert_weights(self.mapper.apply(orig_weights))
