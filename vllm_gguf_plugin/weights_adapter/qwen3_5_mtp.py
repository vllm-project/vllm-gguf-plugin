# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

from ..gguf_utils import find_nextn_block_index
from ..weight_utils import (
    get_gguf_shard_files,
    get_gguf_tensor_names,
    get_gguf_unquantized_params,
    gguf_quant_weights_iterator_multi,
    split_stacked_experts,
)
from .base import BaseGGUFWeightsAdapter, GGUFLoadSpec

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

QWEN35_MTP_MODEL_TYPES = ("qwen3_5_mtp", "qwen3_5_moe_mtp")

# GGUF suffix -> HF suffix within the single MTP block. HF names follow the
# safetensors index, which is what the draft's load_weights takes.
_MTP_TENSORS = {
    "nextn.eh_proj.weight": "fc.weight",
    "nextn.enorm.weight": "pre_fc_norm_embedding.weight",
    "nextn.hnorm.weight": "pre_fc_norm_hidden.weight",
    "nextn.shared_head_norm.weight": "norm.weight",
    "attn_norm.weight": "layers.0.input_layernorm.weight",
    "post_attention_norm.weight": "layers.0.post_attention_layernorm.weight",
    "attn_q.weight": "layers.0.self_attn.q_proj.weight",
    "attn_k.weight": "layers.0.self_attn.k_proj.weight",
    "attn_v.weight": "layers.0.self_attn.v_proj.weight",
    "attn_output.weight": "layers.0.self_attn.o_proj.weight",
    "attn_q_norm.weight": "layers.0.self_attn.q_norm.weight",
    "attn_k_norm.weight": "layers.0.self_attn.k_norm.weight",
    "ffn_gate_inp.weight": "layers.0.mlp.gate.weight",
    "ffn_gate_exps.weight": "layers.0.mlp.experts.0.gate_proj.weight",
    "ffn_up_exps.weight": "layers.0.mlp.experts.0.up_proj.weight",
    "ffn_down_exps.weight": "layers.0.mlp.experts.0.down_proj.weight",
    "ffn_gate_shexp.weight": "layers.0.mlp.shared_expert.gate_proj.weight",
    "ffn_up_shexp.weight": "layers.0.mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp.weight": "layers.0.mlp.shared_expert.down_proj.weight",
    "ffn_gate_inp_shexp.weight": "layers.0.mlp.shared_expert_gate.weight",
}

# Every MTP RMSNorm; enorm/hnorm are the two that don't end in "norm.weight".
_MTP_NORM_SUFFIXES = ("norm.weight", "norm_embedding.weight", "norm_hidden.weight")


class Qwen35MtpGGUFAdapter(BaseGGUFWeightsAdapter):
    """Adapter for the Qwen3.5/3.6 MTP draft when the GGUF carries the nextn
    block. gguf-py has no arch for the draft's model_type, so the name map is
    written out directly. Covers a single MTP block (mtp_num_hidden_layers=1);
    a multi-block export would load only the first."""

    load_spec = None

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type in QWEN35_MTP_MODEL_TYPES

    @staticmethod
    def build_mtp_name_map(block_index: int) -> dict[str, str]:
        return {
            f"blk.{block_index}.{gguf_suffix}": f"mtp.{hf_suffix}"
            for gguf_suffix, hf_suffix in _MTP_TENSORS.items()
        }

    def prepare_loading(
        self,
        model_path: str,
        model_config: ModelConfig,
    ) -> GGUFLoadSpec:
        del model_config
        gguf_files = get_gguf_shard_files(model_path)
        block_index = find_nextn_block_index(gguf_files)
        if block_index is None:
            raise RuntimeError(
                f"No MTP/nextn block in {gguf_files}; this GGUF cannot "
                "serve a speculative draft."
            )
        logger.info("Loading MTP draft from GGUF block %d", block_index)

        gguf_to_hf_name_map = self.build_mtp_name_map(block_index)
        missing = sorted(
            hf_name
            for gguf_name, hf_name in gguf_to_hf_name_map.items()
            if gguf_name not in get_gguf_tensor_names(gguf_files)
        )
        if missing:
            logger.warning(
                "No GGUF tensor for %d MTP param(s): %s", len(missing), missing
            )

        unquantized_modules = list(
            {
                gguf_to_hf_name_map[param].removesuffix(".weight")
                for param in get_gguf_unquantized_params(gguf_files)
                if param in gguf_to_hf_name_map
            }
        )

        self.load_spec = GGUFLoadSpec(
            weights_source=gguf_files,
            gguf_to_hf_name_map=gguf_to_hf_name_map,
            unquantized_modules=unquantized_modules,
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
        yield from split_stacked_experts(self.transform_weight(weights))

    @staticmethod
    def transform_weight(
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Transform raw GGUF weights to HF-style weights."""
        for name, weight in weights:
            if name.endswith(_MTP_NORM_SUFFIXES):
                # GGUF conversion bakes (w + 1) into these RMSNorm weights.
                weight = weight - 1
            elif name.endswith(".weight") and weight.dim() == 1:
                # GGUF flattens shared_expert_gate [1, hidden] -> [hidden].
                weight = weight.unsqueeze(0)
            yield name, weight
