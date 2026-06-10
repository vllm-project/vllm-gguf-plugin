# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from vllm.model_executor.models.utils import WeightsMapper

from ..gguf_utils import detect_gguf_multimodal, maybe_patch_hf_config_from_gguf
from ..weight_utils import (
    get_gguf_unquantized_params,
    gguf_quant_weights_iterator_multi,
)
from .base import BaseGGUFWeightsAdapter, GGUFLoadSpec

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
        orig_to_new_prefix=orig_to_new_prefix,
        orig_to_new_substr=orig_to_new_substr,
    )


class Gemma3GGUFAdapter(BaseGGUFWeightsAdapter):
    """Adapter for Gemma3 GGUF models."""

    mapper = None
    load_spec = None

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type in ("gemma3", "gemma3_text")

    def patch_hf_config(self, model_path: str, hf_config: PretrainedConfig):
        return maybe_patch_hf_config_from_gguf(model_path, hf_config)

    def prepare_weights(
        self, model_config: ModelConfig
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Return HF-style weights."""
        orig_weights = gguf_quant_weights_iterator_multi(self.load_spec.weights_source)
        yield from self.transform_weight(self.mapper.apply(orig_weights))

    def prepare_loading(
        self,
        model_path: str,
        model_config: ModelConfig,
    ) -> GGUFLoadSpec:
        model_config.hf_config = self.patch_hf_config(
            model_path, model_config.hf_config
        )
        gguf_files = [model_path]
        mm_proj_path = detect_gguf_multimodal(model_path)
        if mm_proj_path:
            gguf_files.append(mm_proj_path)
        self.mapper = build_gemma3_mapper(is_multimodal=mm_proj_path is not None)
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

    def transform_weight(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Transform raw GGUF weights to HF-style weights."""
        for name, weight in weights:
            if name.endswith("norm.weight"):
                weight = weight - 1
            elif name.startswith("vision_tower") and "mlp.up_proj." in name:
                name = name.replace("mlp.up_proj.", "mlp.fc2.")
            elif name.startswith("vision_tower") and "mlp.down_proj." in name:
                name = name.replace("mlp.down_proj.", "mlp.fc1.")
            yield name, weight
