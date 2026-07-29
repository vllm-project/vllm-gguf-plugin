# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.model_executor.models.utils import WeightsMapper

from ..gguf_utils import find_nextn_block_index
from ..weight_utils import (
    get_gguf_shard_files,
    get_gguf_tensor_names,
    get_gguf_unquantized_params,
    gguf_quant_weights_iterator_multi,
    split_stacked_experts,
)
from .base import BaseGGUFWeightsAdapter, GGUFLoadSpec
from .qwen3_5 import qwen35_layer_substr

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

QWEN35_MTP_MODEL_TYPES = ("qwen3_5_mtp", "qwen3_5_moe_mtp")
QWEN35_MOE_MTP_MODEL_TYPES = ("qwen3_5_moe_mtp",)

# Every MTP RMSNorm; enorm/hnorm are the two that don't end in "norm.weight".
_MTP_NORM_SUFFIXES = ("norm.weight", "norm_embedding.weight", "norm_hidden.weight")


def build_qwen35_mtp_mapper(block_index: int, is_moe: bool) -> WeightsMapper:
    """Map the GGUF nextn block onto the draft's HF names. The block holds a
    plain decoder layer, so only the prefixes differ from the backbone."""
    blk = f"blk.{block_index}."
    return WeightsMapper(
        # nextn entries first; once one rewrites a name the generic block
        # prefix no longer matches it.
        orig_to_new_prefix={
            f"{blk}nextn.eh_proj.": "mtp.fc.",
            f"{blk}nextn.enorm.": "mtp.pre_fc_norm_embedding.",
            f"{blk}nextn.hnorm.": "mtp.pre_fc_norm_hidden.",
            f"{blk}nextn.shared_head_norm.": "mtp.norm.",
            blk: "mtp.layers.0.",
        },
        orig_to_new_substr=qwen35_layer_substr(is_moe),
    )


class Qwen35MtpGGUFAdapter(BaseGGUFWeightsAdapter):
    """Adapter for the Qwen3.5/3.6 MTP draft when the GGUF carries the nextn
    block. gguf-py has no arch for the draft's model_type, so the names come
    from the backbone's rules plus the nextn prefixes. Covers a single MTP
    block (mtp_num_hidden_layers=1); a multi-block export loads only the
    first."""

    load_spec = None

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type in QWEN35_MTP_MODEL_TYPES

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

        mapper = build_qwen35_mtp_mapper(
            block_index,
            is_moe=self.config.model_type in QWEN35_MOE_MTP_MODEL_TYPES,
        )
        # Only the draft block; every other tensor belongs to the backbone.
        block_prefix = f"blk.{block_index}."
        gguf_to_hf_name_map: dict[str, str] = {}
        unmapped: list[str] = []
        for name in sorted(get_gguf_tensor_names(gguf_files)):
            if not name.startswith(block_prefix):
                continue
            hf_name = mapper.apply_list([name])[0]
            if hf_name == name:
                unmapped.append(name)
            else:
                gguf_to_hf_name_map[name] = hf_name
        if unmapped:
            logger.warning(
                "No HF name for %d MTP tensor(s), skipping: %s", len(unmapped), unmapped
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
