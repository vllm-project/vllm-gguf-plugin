# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from vllm.model_executor.models.utils import WeightsMapper

from ..gguf_files import GGUFModelFiles
from ..gguf_utils import maybe_patch_hf_config_from_gguf
from ..weight_utils import get_gguf_tensor_names
from .base import BaseGGUFWeightsAdapter, GGUFWeight

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
    weights: Iterable[GGUFWeight],
) -> Iterable[GGUFWeight]:
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

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type == "olmoe"

    def patch_hf_config(
        self,
        files: GGUFModelFiles,
        hf_config: PretrainedConfig,
    ):
        return maybe_patch_hf_config_from_gguf(files.primary_backbone, hf_config)

    def build_name_map(
        self,
        files: GGUFModelFiles,
        model_config: ModelConfig,
    ) -> dict[str, str]:
        del model_config
        mapper = build_olmoe_mapper()
        gguf_names = sorted(get_gguf_tensor_names(files.backbone))
        hf_names = mapper.apply_list(gguf_names)
        return dict(zip(gguf_names, hf_names, strict=True))

    def transform_weights(
        self,
        weights: Iterable[GGUFWeight],
        model_config: ModelConfig,
    ) -> Iterable[GGUFWeight]:
        del model_config
        yield from split_olmoe_expert_weights(weights)
