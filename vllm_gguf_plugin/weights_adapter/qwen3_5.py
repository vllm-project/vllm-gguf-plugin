# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.model_executor.models.utils import WeightsMapper

from ..gguf_utils import detect_gguf_multimodal, maybe_patch_hf_config_from_gguf
from ..weight_utils import (
    get_gguf_shard_files,
    get_gguf_tensor_names,
    get_gguf_unquantized_params,
    gguf_quant_weights_iterator_multi,
    split_stacked_experts,
)
from .base import BaseGGUFWeightsAdapter, GGUFLoadSpec

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig

logger = init_logger(__name__)

QWEN35_MODEL_TYPES = (
    "qwen3_5",
    "qwen3_5_text",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
)
QWEN35_MOE_MODEL_TYPES = ("qwen3_5_moe", "qwen3_5_moe_text")

# GGUF tensors that load as plain params, so they must be dequantized on the
# way out (names are the GGUF ones, the mapper runs afterwards).
DEQUANT_TENSORS = ("token_embd.weight", "output.weight")


# Within-layer GGUF -> HF renames, shared with the MTP draft. Substr rules
# apply in order, so entries that are prefixes of others come last.
QWEN35_ATTN_SUBSTR: dict[str, str] = {
    "attn_norm.": "input_layernorm.",
    "post_attention_norm.": "post_attention_layernorm.",
    # attention
    "attn_q_norm.": "self_attn.q_norm.",
    "attn_k_norm.": "self_attn.k_norm.",
    "attn_q.": "self_attn.q_proj.",
    "attn_k.": "self_attn.k_proj.",
    "attn_v.": "self_attn.v_proj.",
    "attn_output.": "self_attn.o_proj.",
    # gated delta net; llama.cpp writes the two fused in_proj halves under
    # the attention names, the rest under ssm_*.
    "attn_qkv.": "linear_attn.in_proj_qkv.",
    "attn_gate.": "linear_attn.in_proj_z.",
    "ssm_alpha.": "linear_attn.in_proj_a.",
    "ssm_beta.": "linear_attn.in_proj_b.",
    "ssm_conv1d.": "linear_attn.conv1d.",
    "ssm_norm.": "linear_attn.norm.",
    "ssm_out.": "linear_attn.out_proj.",
    # A_log and dt_bias are bare params in the HF checkpoint, so they drop
    # the GGUF suffix. Keep these after ssm_alpha, "ssm_a" is a prefix of it.
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_a.weight": "linear_attn.A_log",
    "ssm_a": "linear_attn.A_log",
}

QWEN35_MOE_SUBSTR: dict[str, str] = {
    "ffn_gate_inp_shexp.": "mlp.shared_expert_gate.",
    "ffn_gate_inp.": "mlp.gate.",
    # Expert weights are stacked; prepare_weights splits them per expert.
    "ffn_gate_exps.": "mlp.experts.0.gate_proj.",
    "ffn_up_exps.": "mlp.experts.0.up_proj.",
    "ffn_down_exps.": "mlp.experts.0.down_proj.",
    "ffn_gate_shexp.": "mlp.shared_expert.gate_proj.",
    "ffn_up_shexp.": "mlp.shared_expert.up_proj.",
    "ffn_down_shexp.": "mlp.shared_expert.down_proj.",
}

QWEN35_DENSE_SUBSTR: dict[str, str] = {
    "ffn_gate.": "mlp.gate_proj.",
    "ffn_up.": "mlp.up_proj.",
    "ffn_down.": "mlp.down_proj.",
}


def qwen35_layer_substr(is_moe: bool) -> dict[str, str]:
    """Within-layer renames for a Qwen3.5 decoder block."""
    return QWEN35_ATTN_SUBSTR | (QWEN35_MOE_SUBSTR if is_moe else QWEN35_DENSE_SUBSTR)


def build_qwen35_text_mapper(is_multimodal: bool, is_moe: bool) -> WeightsMapper:
    backbone_prefix = "model.language_model." if is_multimodal else "model."
    return WeightsMapper(
        orig_to_new_prefix={
            "token_embd.": backbone_prefix + "embed_tokens.",
            "blk.": backbone_prefix + "layers.",
            "output_norm.": backbone_prefix + "norm.",
            "output.": "lm_head.",
        },
        orig_to_new_substr=qwen35_layer_substr(is_moe),
    )


def build_qwen35_vision_mapper() -> WeightsMapper:
    """Vision tower and merger. Kept apart from the text mapper because the
    two reuse GGUF names for different modules (``attn_qkv`` is the merged QKV
    of a vision block, but the GDN in_proj of a text layer)."""
    return WeightsMapper(
        orig_to_new_prefix={
            "v.blk.": "model.visual.blocks.",
            "v.patch_embd.": "model.visual.patch_embed.proj.",
            "v.position_embd.": "model.visual.pos_embed.",
            # llama.cpp writes the merger norm as v.post_ln and its MLP as mm.N.
            "v.post_ln.": "model.visual.merger.norm.",
            "mm.0.": "model.visual.merger.linear_fc1.",
            "mm.2.": "model.visual.merger.linear_fc2.",
        },
        orig_to_new_substr={
            "attn_qkv.": "attn.qkv.",
            "attn_out.": "attn.proj.",
            "ffn_up.": "mlp.linear_fc1.",
            "ffn_down.": "mlp.linear_fc2.",
            "ln1.": "norm1.",
            "ln2.": "norm2.",
        },
    )


class Qwen35GGUFAdapter(BaseGGUFWeightsAdapter):
    """Adapter for Qwen3.5 dense and MoE GGUF models (Qwen3.6 reuses the
    qwen3_5_moe architecture)."""

    text_mapper = None
    vision_mapper = None
    load_spec = None

    _reorder: dict | None = None
    _dequant_tensors: tuple[str, ...] = DEQUANT_TENSORS

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type in QWEN35_MODEL_TYPES

    def patch_hf_config(self, model_path: str, hf_config: PretrainedConfig):
        return maybe_patch_hf_config_from_gguf(model_path, hf_config)

    def prepare_loading(
        self,
        model_path: str,
        model_config: ModelConfig,
    ) -> GGUFLoadSpec:
        model_config.hf_config = self.patch_hf_config(
            model_path, model_config.hf_config
        )
        # patch_hf_config may replace the config object (multimodal upgrade)
        self.config = model_config.hf_config

        mmproj_path = self._ensure_mmproj(model_path)
        gguf_files = get_gguf_shard_files(model_path)
        if mmproj_path is not None:
            gguf_files.append(str(mmproj_path))

        is_moe = self.config.model_type in QWEN35_MOE_MODEL_TYPES
        self.text_mapper = build_qwen35_text_mapper(
            is_multimodal=mmproj_path is not None, is_moe=is_moe
        )
        self.vision_mapper = build_qwen35_vision_mapper()
        self._set_gdn_reorder()

        unmapped = sorted(
            name
            for name in get_gguf_tensor_names(gguf_files)
            if self._map_name(name) is None
        )
        if unmapped:
            logger.warning(
                "No HF name for %d GGUF tensor(s), skipping: %s",
                len(unmapped),
                unmapped,
            )

        unquantized_modules = list(
            {
                mapped.removesuffix(".weight")
                for param in get_gguf_unquantized_params(gguf_files)
                if (mapped := self._map_name(param)) is not None
                and mapped.endswith(".weight")
            }
        )
        for name in self._forced_unquantized_modules():
            if name not in unquantized_modules:
                unquantized_modules.append(name)

        self.load_spec = GGUFLoadSpec(
            weights_source=gguf_files,
            unquantized_modules=unquantized_modules,
        )
        return self.load_spec

    def _ensure_mmproj(self, model_path: str) -> Path | None:
        """Path to the mmproj file for a multimodal Qwen3.5 config, or None
        when the model is text-only. The loader fetches it alongside the
        backbone."""
        if getattr(self.config, "vision_config", None) is None:
            return None

        mmproj_path = detect_gguf_multimodal(model_path)
        if mmproj_path is None:
            raise RuntimeError(
                "Could not find mmproj file for multimodal GGUF model. "
                "Please ensure a *mmproj*.gguf file is in the same directory "
                "as the backbone GGUF file or available in the HF repo."
            )
        return mmproj_path

    def _set_gdn_reorder(self) -> None:
        """llama.cpp tiles GDN V heads when num_value_heads != num_key_heads."""
        tc = self.config.get_text_config()
        num_k = getattr(tc, "linear_num_key_heads", 0) or 0
        num_v = getattr(tc, "linear_num_value_heads", 0) or 0
        self._reorder = None
        self._dequant_tensors = DEQUANT_TENSORS
        if num_k and num_v and num_v % num_k == 0 and num_v // num_k > 1:
            self._reorder = {
                "num_k": num_k,
                "r": num_v // num_k,
                "head_k": tc.linear_key_head_dim,
                "head_v": tc.linear_value_head_dim,
            }
            # Row reorders work on packed rows; out_proj is a column reorder,
            # so it alone needs float.
            self._dequant_tensors += ("ssm_out.weight",)

    def _forced_unquantized_modules(self) -> list[str]:
        """Modules whose weights are handed over as plain params, so they must
        not be built with the GGUF linear method."""
        # ParallelLMHead / VocabParallelEmbedding take plain params only.
        modules = ["lm_head", "embed_tokens"]
        if self._reorder is not None:
            # Dequantized for the GDN column reorder, see _set_gdn_reorder.
            modules.append("linear_attn.out_proj")
        return modules

    def _map_name(self, gguf_name: str) -> str | None:
        """HF name for *gguf_name*, or None when nothing maps it."""
        mapper = (
            self.vision_mapper
            if gguf_name.startswith(("v.", "mm."))
            else self.text_mapper
        )
        hf_name = mapper.apply_list([gguf_name])[0]
        return hf_name if hf_name != gguf_name else None

    def prepare_weights(
        self,
        model_config: ModelConfig,
    ) -> Iterable[tuple[str, torch.Tensor]]:
        del model_config
        weights = gguf_quant_weights_iterator_multi(
            self.load_spec.weights_source,
            dequant_suffixes=self._dequant_tensors,
        )
        mapped = self.map_names(weights)
        yield from split_stacked_experts(self.transform_weight(mapped))

    def map_names(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for gguf_name, weight in weights:
            # Quantized tensors arrive as .qweight / .qweight_type pairs; the
            # mapper keys on the module part, so both map like the plain name.
            hf_name = self._map_name(gguf_name)
            if hf_name is not None:
                yield hf_name, weight

    @staticmethod
    def _inv_reorder(t, dim, num_k, r, head_dim):
        # Undo llama.cpp's grouped->tiled V-head reorder along *dim*.
        shape = list(t.shape)
        if dim < 0:
            dim += len(shape)
        t = t.reshape(*shape[:dim], r, num_k, head_dim, *shape[dim + 1 :])
        t = t.transpose(dim, dim + 1)
        return t.reshape(*shape).contiguous()

    def _reorder_gdn(self, name: str, w: torch.Tensor) -> torch.Tensor | None:
        """Undo llama.cpp's grouped->tiled V-head reorder. Row reorders apply
        to packed ``qweight`` too, since GGUF quantizes per row."""
        rc = self._reorder
        nk, r, hk, hv = rc["num_k"], rc["r"], rc["head_k"], rc["head_v"]
        inv = self._inv_reorder
        if name.endswith("qweight_type"):
            return None
        base = name.removesuffix(".qweight").removesuffix(".weight")
        if base.endswith(".A_log"):
            return inv(torch.log(-w), 0, nk, r, 1)
        if base.endswith(".dt_bias"):
            return inv(w, 0, nk, r, 1)
        if base.endswith("linear_attn.in_proj_z"):
            return inv(w, 0, nk, r, hv)
        if base.endswith(("linear_attn.in_proj_a", "linear_attn.in_proj_b")):
            return inv(w, 0, nk, r, 1)
        if base.endswith("linear_attn.in_proj_qkv"):
            qk = hk * nk * 2  # q + k rows are unchanged; only V rows reorder
            return torch.cat([w[:qk], inv(w[qk:], 0, nk, r, hv)], dim=0)
        if base.endswith("linear_attn.out_proj"):
            return inv(w, 1, nk, r, hv)  # column (input) reorder, dequantized
        if base.endswith("linear_attn.conv1d") and w.dim() == 2:
            qk = hk * nk * 2
            return torch.cat([w[:qk], inv(w[qk:], 0, nk, r, hv)], dim=0).unsqueeze(1)
        return None

    def transform_weight(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Transform raw GGUF weights to HF-style weights."""
        vision_config = getattr(self.config, "vision_config", None)
        temporal_patch_size = getattr(vision_config, "temporal_patch_size", 1)
        patch_embed_parts: dict[str, torch.Tensor] = {}
        for name, weight in weights:
            # Forced-unquantized modules keep plain params.
            if "qweight" in name and ("lm_head." in name or "embed_tokens." in name):
                continue
            if self._reorder is not None:
                # Also folds A_log's log(-a) recovery in the right order.
                out = self._reorder_gdn(name, weight)
                if out is not None:
                    yield name, out
                    continue
            if name.endswith(".A_log"):
                # GGUF stores A = -exp(A_log); recover A_log for the model.
                yield name, torch.log(-weight)
                continue
            if (
                name.endswith("norm.weight")
                and not name.endswith("linear_attn.norm.weight")
                and "visual" not in name
            ):
                # GGUF conversion bakes (w + 1) into these RMSNorm weights.
                yield name, weight - 1
                continue
            if temporal_patch_size > 1 and "patch_embed.proj.weight" in name:
                # GGUF holds one 2D conv per temporal frame; stack back to 5D.
                base, split, _ = name.rpartition(".weight.")
                key = f"{base}.weight" if split else name
                patch_embed_parts[name] = weight
                parts = [patch_embed_parts.get(key)] + [
                    patch_embed_parts.get(f"{key}.{i}")
                    for i in range(1, temporal_patch_size)
                ]
                if any(part is None for part in parts):
                    continue
                yield key, torch.stack(parts, dim=2)
                continue
            if "conv1d.weight" in name and weight.dim() == 2:
                # depthwise Conv1d: [d, k] -> [d, 1, k]
                weight = weight.unsqueeze(1)
            elif name.endswith(".weight") and weight.dim() == 1 and "norm" not in name:
                # GGUF flattens [1, hidden] linears (e.g. shared expert gate).
                weight = weight.unsqueeze(0)
            yield name, weight
