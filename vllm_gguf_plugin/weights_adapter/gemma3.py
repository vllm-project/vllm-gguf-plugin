# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import re
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


def build_gemma3_mapper(is_multimodal: bool) -> WeightsMapper:
    backbone_prefix = "language_model.model." if is_multimodal else "model."
    lm_head_prefix = "language_model." if is_multimodal else "model."
    orig_to_new_prefix: dict[str, str] = {
        # vision tower
        "v.blk.": "vision_tower.vision_model.encoder.layers.",
        "v.patch_embd.": "vision_tower.vision_model.embeddings.patch_embedding.",
        "v.position_embd.": "vision_tower.vision_model.embeddings.position_embedding.",
        "v.post_ln.": "vision_tower.vision_model.post_layernorm.",
        # mm projector
        "mm.input_projection.weight": (
            "multi_modal_projector.mm_input_projection_weight"
        ),
        "mm.soft_emb_norm.": "multi_modal_projector.mm_soft_emb_norm.",
        # text backbone (without language model prefix)
        "token_embd.": backbone_prefix + "embed_tokens.",
        "blk.": backbone_prefix + "layers.",
        "output_norm.": backbone_prefix + "norm.",
        "output.": lm_head_prefix + "lm_head.",
    }
    orig_to_new_substr: dict[str, str] = {
        # vision tower
        "ln1.": "layer_norm1.",
        "ln2.": "layer_norm2.",
        "attn_q.": "self_attn.q_proj.",
        "attn_k.": "self_attn.k_proj.",
        "attn_v.": "self_attn.v_proj.",
        "attn_out.": "self_attn.out_proj.",
        # text backbone
        "attn_output.": "self_attn.o_proj.",
        "attn_q_norm.": "self_attn.q_norm.",
        "attn_k_norm.": "self_attn.k_norm.",
        "attn_norm.": "input_layernorm.",
        "post_attention_norm.": "post_attention_layernorm.",
        "ffn_norm.": "pre_feedforward_layernorm.",
        "post_ffw_norm.": "post_feedforward_layernorm.",
        "ffn_gate.": "mlp.gate_proj.",
        "ffn_up.": "mlp.up_proj.",
        "ffn_down.": "mlp.down_proj.",
    }

    return WeightsMapper(
        orig_to_new_regex={
            re.compile(r"^(v\.blk\.\d+)\.ffn_up\."): r"\1.mlp.fc2.",
            re.compile(r"^(v\.blk\.\d+)\.ffn_down\."): r"\1.mlp.fc1.",
        },
        orig_to_new_prefix=orig_to_new_prefix,
        orig_to_new_substr=orig_to_new_substr,
    )


class Gemma3GGUFAdapter(BaseGGUFWeightsAdapter):
    """Adapter for Gemma3 GGUF models."""

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type in ("gemma3", "gemma3_text")

    def patch_hf_config(
        self,
        files: GGUFModelFiles,
        hf_config: PretrainedConfig,
    ):
        return maybe_patch_hf_config_from_gguf(
            files.primary_backbone,
            hf_config,
            mmproj_path=files.mm_proj,
        )

    def build_name_map(
        self,
        files: GGUFModelFiles,
        model_config: ModelConfig,
    ) -> dict[str, str]:
        del model_config
        mapper = build_gemma3_mapper(is_multimodal=files.mm_proj is not None)
        gguf_names = sorted(get_gguf_tensor_names(files.all_files))
        hf_names = mapper.apply_list(gguf_names)
        return dict(zip(gguf_names, hf_names, strict=True))

    def transform_weights(
        self,
        weights: Iterable[GGUFWeight],
        model_config: ModelConfig,
    ) -> Iterable[GGUFWeight]:
        """Transform mapped GGUF weights to the Gemma3 representation."""
        del model_config
        for name, weight in weights:
            if name.endswith("norm.weight") and not name.startswith("vision_tower"):
                weight = weight - 1
            yield name, weight
