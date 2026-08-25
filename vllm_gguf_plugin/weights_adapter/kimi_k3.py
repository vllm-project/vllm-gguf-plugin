# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import gguf
import torch
from vllm.logger import init_logger
from vllm.model_executor.models.utils import WeightsMapper

from ..gguf_files import GGUFModelFiles
from ..gguf_utils import maybe_patch_hf_config_from_gguf
from ..weight_utils import (
    dequantize_gguf_tensor,
    get_gguf_tensor_names,
    get_gguf_unquantized_params,
    split_stacked_experts,
)
from .base import BaseGGUFWeightsAdapter, GGUFWeight
from .mla import (
    MLA_KV_B_SUBSTR,
    MLA_KV_B_UNQUANTIZED_MODULES,
    fuse_kv_b_proj_weights,
)

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig

logger = init_logger(__name__)

KIMI_K3_MODEL_TYPES = ("kimi_k3",)
KIMI_K3_ARCHITECTURE = "KimiK3ForConditionalGeneration"

# GGUF (llama.cpp "kimi-k3" architecture) -> vLLM name fragments.
# KDA-layer and MLA-layer tensor names are disjoint, so one map covers both.
KIMI_K3_TEXT_SUBSTR: dict[str, str] = {
    # norms
    "attn_norm.": "input_layernorm.",
    "ffn_norm.": "post_attention_layernorm.",
    # KDA (Kimi Delta Attention) layers
    "attn_q.": "self_attn.q_proj.",
    "attn_k.": "self_attn.k_proj.",
    "attn_v.": "self_attn.v_proj.",
    "ssm_g.": "self_attn.g_proj.",
    "ssm_f_a.": "self_attn.f_a_proj.",
    "ssm_f_b.": "self_attn.f_b_proj.",
    "ssm_beta.": "self_attn.b_proj.",
    "ssm_dt.bias": "self_attn.dt_bias",
    "ssm_conv1d_q.": "self_attn.q_conv1d.",
    "ssm_conv1d_k.": "self_attn.k_conv1d.",
    "ssm_conv1d_v.": "self_attn.v_conv1d.",
    "ssm_norm.": "self_attn.o_norm.",
    "ssm_a": "self_attn.A_log",
    # Gated MLA layers
    "attn_q_a_norm.": "self_attn.q_a_layernorm.",
    "attn_kv_a_norm.": "self_attn.kv_a_layernorm.",
    "attn_q_a.": "self_attn.q_a_proj.",
    "attn_q_b.": "self_attn.q_b_proj.",
    "attn_kv_a_mqa.": "self_attn.kv_a_proj_with_mqa.",
    # llama.cpp splits kv_b_proj into k_b/v_b; fused by fuse_kv_b_proj_weights
    **MLA_KV_B_SUBSTR,
    "attn_gate.": "self_attn.g_proj.",
    "attn_output.": "self_attn.o_proj.",
    # Attention residuals: llama.cpp folds res_norm.weight * res_proj.weight
    # into a single [hidden] score vector; loaded into res_proj (the matching
    # res_norm stays at its ones initialization, keeping the product intact).
    "attn_res_score.": "self_attention_res_proj.",
    "ffn_res_score.": "mlp_res_proj.",
    # LatentMoE
    "ffn_gate_inp.": "block_sparse_moe.gate.",
    "exp_probs_b.bias": "block_sparse_moe.gate.e_score_correction_bias",
    "ffn_gate_exps.": "block_sparse_moe.experts.0.w1.",
    "ffn_up_exps.": "block_sparse_moe.experts.0.w3.",
    "ffn_down_exps.": "block_sparse_moe.experts.0.w2.",
    "ffn_gate_shexp.": "block_sparse_moe.shared_experts.gate_proj.",
    "ffn_up_shexp.": "block_sparse_moe.shared_experts.up_proj.",
    "ffn_down_shexp.": "block_sparse_moe.shared_experts.down_proj.",
    "ffn_routed_down.": "block_sparse_moe.routed_expert_down_proj.",
    "ffn_routed_norm.": "block_sparse_moe.routed_expert_norm.",
    "ffn_routed_up.": "block_sparse_moe.routed_expert_up_proj.",
    # dense MLP (leading dense blocks)
    "ffn_gate.": "mlp.gate_proj.",
    "ffn_up.": "mlp.up_proj.",
    "ffn_down.": "mlp.down_proj.",
}

# MoonViT-V2 vision tower + patchmergerv2 projector (mmproj GGUF).
KIMI_K3_VISION_PREFIX: dict[str, str] = {
    "v.blk.": "vision_tower.encoder.blocks.",
    "v.patch_embd.": "vision_tower.patch_embed.proj.",
    "v.position_embd.": "vision_tower.patch_embed.pos_emb.",
    "v.post_ln.": "vision_tower.encoder.final_layernorm.",
    "mm.1.": "mm_projector.linear_1.",
    "mm.2.": "mm_projector.linear_2.",
    "mm.post_norm.": "mm_projector.post_norm.",
}

KIMI_K3_VISION_SUBSTR: dict[str, str] = {
    "ln1.": "norm0.",
    "ln2.": "norm1.",
    "attn_qkv.": "wqkv.",
    "attn_out.": "wo.",
    "ffn_up.": "mlp.fc0.",
    "ffn_down.": "mlp.fc1.",
}

_ROUTED_PROJ_PATTERN = re.compile(
    r"block_sparse_moe\.routed_expert_(down|up)_proj\.(weight|qweight|qweight_type)$"
)

_LAYER_PATTERN = re.compile(r"\.layers\.(\d+)\.")

# Mapped names (minus ".weight") that load as shards of a fused vLLM
# parameter instead of standalone modules. KDA-only projections are
# unambiguous; "self_attn.g_proj" is fused only on KDA layers (on MLA layers
# it is a standalone output-gate projection).
_FUSED_SHARD_STEMS = (
    ".self_attn.q_proj",
    ".self_attn.k_proj",
    ".self_attn.v_proj",
    ".self_attn.f_a_proj",
    ".self_attn.b_proj",
    ".self_attn.q_a_proj",
    ".self_attn.kv_a_proj_with_mqa",
    ".mlp.gate_proj",
    ".mlp.up_proj",
    ".shared_experts.gate_proj",
    ".shared_experts.up_proj",
    # stacked expert tensors load into fused w13/w2 GGUF parameters
    ".experts.0.w1",
    ".experts.0.w2",
    ".experts.0.w3",
)


def _routes_to_fused_param(mapped: str, kda_layers: set[int]) -> bool:
    """Whether *mapped* is a shard of a fused (merged) vLLM parameter."""
    if not mapped.endswith(".weight"):
        return False
    stem = mapped.removesuffix(".weight")
    if stem.endswith(_FUSED_SHARD_STEMS):
        return True
    if stem.endswith(".self_attn.g_proj"):
        match = _LAYER_PATTERN.search(mapped)
        return match is not None and int(match.group(1)) + 1 in kda_layers
    return False


def build_kimi_k3_text_mapper(is_multimodal: bool) -> WeightsMapper:
    backbone_prefix = "language_model.model." if is_multimodal else "model."
    lm_head_prefix = "language_model." if is_multimodal else ""
    return WeightsMapper(
        orig_to_new_prefix={
            "token_embd.": backbone_prefix + "embed_tokens.",
            "blk.": backbone_prefix + "layers.",
            "output_norm.": backbone_prefix + "norm.",
            "output_res_score.": backbone_prefix + "output_attn_res_proj.",
            "output.": lm_head_prefix + "lm_head.",
        },
        orig_to_new_substr=KIMI_K3_TEXT_SUBSTR,
    )


def build_kimi_k3_vision_mapper() -> WeightsMapper:
    return WeightsMapper(
        orig_to_new_prefix=KIMI_K3_VISION_PREFIX,
        orig_to_new_substr=KIMI_K3_VISION_SUBSTR,
    )


def _reinterleave_vision_qk(
    weight: torch.Tensor, num_heads: int, qkv_hidden_size: int
) -> torch.Tensor:
    """Undo the Q/K de-interleave llama.cpp applies for its 2D rope.

    The mmproj conversion maps ``x[h, a, b, c] -> y[h, b, a, c]`` with
    ``a < head_dim // 4`` via ``reshape(h, d//4, 2, 2, in).permute(0, 2, 1, 3,
    4)``. The inverse must therefore split the permuted axis as ``(2, d//4)``
    before applying the same permutation (an involution) back.
    """

    def interleave(w: torch.Tensor) -> torch.Tensor:
        out_dim, in_dim = w.shape
        head_dim = out_dim // num_heads
        return (
            w.reshape(num_heads, 2, head_dim // 4, 2, in_dim)
            .permute(0, 2, 1, 3, 4)
            .reshape(out_dim, in_dim)
        )

    wq, wk, wv = weight.split(qkv_hidden_size, dim=0)
    return torch.cat([interleave(wq), interleave(wk), wv], dim=0)


class KimiK3GGUFAdapter(BaseGGUFWeightsAdapter):
    """Kimi-K3 (KDA + Gated MLA + LatentMoE, MoonViT-V2) GGUF adapter."""

    # kv_b_proj is reconstructed from llama.cpp's split attn_k_b / attn_v_b
    # tensors, so it is always materialized as a plain float weight.
    extra_unquantized_modules = MLA_KV_B_UNQUANTIZED_MODULES

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type in KIMI_K3_MODEL_TYPES

    @classmethod
    def architecture(cls, config) -> str | None:
        return KIMI_K3_ARCHITECTURE

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
        has_vision = getattr(patched, "vision_config", None) is not None
        if has_vision and files.mm_proj is None:
            raise RuntimeError(
                "Could not find mm_proj for multimodal Kimi-K3 GGUF. "
                "Place *mmproj*.gguf beside the backbone or pass "
                "model_loader_extra_config={'mm_proj': ...}."
            )
        patched.architectures = [KIMI_K3_ARCHITECTURE]
        return patched

    def build_name_map(
        self,
        files: GGUFModelFiles,
        model_config: ModelConfig,
    ) -> dict[str, str]:
        is_multimodal = files.mm_proj is not None
        text_mapper = build_kimi_k3_text_mapper(is_multimodal)
        vision_mapper = build_kimi_k3_vision_mapper()
        tensor_names = sorted(get_gguf_tensor_names(files.all_files))
        unquantized_tensors = set(get_gguf_unquantized_params(list(files.all_files)))
        linear_attn = (
            getattr(
                model_config.hf_config.get_text_config(), "linear_attn_config", None
            )
            or {}
        )
        kda_layers = set(linear_attn.get("kda_layers") or [])  # 1-based

        name_map: dict[str, str] = {}
        unmapped: list[str] = []
        for name in tensor_names:
            mapper = vision_mapper if name.startswith(("v.", "mm.")) else text_mapper
            mapped = mapper.apply_list([name])[0]
            if mapped == name:
                unmapped.append(name)
                continue
            if name in unquantized_tensors and _routes_to_fused_param(
                mapped, kda_layers
            ):
                # An unquantized (F32) shard of a fused parameter (e.g. the
                # beta projection of in_proj_qkvgfab, or every shard of an
                # unquantized GGUF) must still go through the GGUF weight
                # loader of the fused module; the plain ".weight" name would
                # not resolve. transform_weights announces its type and casts
                # it to the model dtype.
                mapped = mapped.removesuffix(".weight") + ".qweight"
            name_map[name] = mapped
        if unmapped:
            logger.warning(
                "No HF name for %d Kimi-K3 GGUF tensor(s), skipping: %s",
                len(unmapped),
                unmapped,
            )
        return name_map

    def transform_weights(
        self,
        weights: Iterable[GGUFWeight],
        model_config: ModelConfig,
    ) -> Iterable[GGUFWeight]:
        dtype = model_config.dtype
        weights = fuse_kv_b_proj_weights(weights, dtype)
        weights = _dequantize_routed_projs(weights, dtype)
        weights = _tag_float_shards(weights, dtype)
        weights = split_stacked_experts(weights)
        vision_config = getattr(model_config.hf_config, "vision_config", None)
        return _transform_values(weights, _value_rules(vision_config))


def _dequantize_routed_projs(
    weights: Iterable[GGUFWeight], dtype: torch.dtype
) -> Iterable[GGUFWeight]:
    """Dequantize the LatentMoE down/up projections.

    vLLM keeps them unquantized (quant_config=None), so the GGUF payload is
    dequantized and emitted as a plain ".weight".
    """
    weight_types: dict[str, int] = {}
    for name, weight in weights:
        match = _ROUTED_PROJ_PATTERN.search(name)
        if match is None:
            yield name, weight
            continue
        proj, suffix = match.group(1), match.group(2)
        base = f"{name[: match.start()]}block_sparse_moe.routed_expert_{proj}_proj"
        if suffix == "qweight_type":
            weight_types[base] = int(weight.item())
            continue
        dequantized = dequantize_gguf_tensor(weight, weight_types.pop(base, None))
        yield base + ".weight", dequantized.to(dtype)


def _tag_float_shards(
    weights: Iterable[GGUFWeight], dtype: torch.dtype
) -> Iterable[GGUFWeight]:
    """Announce the type of float shards routed through the GGUF loader.

    See build_name_map: unquantized GGUF tensors belonging to fused
    parameters carry ".qweight" names; give them a qweight_type marker and
    cast to the model dtype.
    """
    for name, weight in weights:
        if name.endswith(".qweight") and weight.dtype.is_floating_point:
            yield (
                name.replace(".qweight", ".qweight_type"),
                torch.tensor(gguf.GGMLQuantizationType.BF16),
            )
            yield name, weight.to(dtype)
        else:
            yield name, weight


def _value_rules(
    vision_config: PretrainedConfig | None,
) -> list[
    tuple[
        Callable[[str, torch.Tensor], bool],
        Callable[[torch.Tensor], torch.Tensor],
    ]
]:
    """(predicate, transform) rules applied per weight, in order."""
    rules: list[tuple] = [
        # GGUF stores -exp(A_log); vLLM wants log-space.
        (lambda n, w: n.endswith("self_attn.A_log"), lambda w: torch.log(-w.float())),
        # llama.cpp folds AttnRes into a [hidden] vector; vLLM's res_proj is
        # [1, hidden] (the matching res_norm stays at its ones init).
        (
            lambda n, w: n.endswith("res_proj.weight") and w.dim() == 1,
            lambda w: w.unsqueeze(0),
        ),
    ]
    vt_num_heads = getattr(vision_config, "num_attention_heads", None)
    vt_qkv_hidden = getattr(vision_config, "qkv_hidden_size", None)
    if vt_num_heads and vt_qkv_hidden:
        rules.append(
            (
                lambda n, w: "vision_tower" in n and n.endswith("wqkv.weight"),
                lambda w: _reinterleave_vision_qk(w, vt_num_heads, vt_qkv_hidden),
            )
        )
    return rules


def _transform_values(
    weights: Iterable[GGUFWeight],
    rules: list[tuple[Callable, Callable]],
) -> Iterable[GGUFWeight]:
    for name, weight in weights:
        for predicate, transform in rules:
            if predicate(name, weight):
                weight = transform(weight)
                break
        yield name, weight
