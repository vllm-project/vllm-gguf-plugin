# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from gguf import GGMLQuantizationType
from gguf.quants import dequantize

from ..config_parser import KIMI_K3_GGUF_TEXT_MARKER
from .default import GGUFWeightsAdapter

if TYPE_CHECKING:
    from vllm.config import ModelConfig

_KV_B_PART_RE = re.compile(
    r"^(model\.layers\.\d+\.self_attn)\.([kv])_b_proj\."
    r"(qweight_type|qweight|weight)$"
)
_ROUTED_PROJ_RE = re.compile(
    r"^(model\.layers\.\d+\.block_sparse_moe\."
    r"routed_expert_(?:down|up)_proj)\.(qweight_type|qweight)$"
)
_KDA_IN_PROJ_RE = re.compile(
    r"^(model\.layers\.(\d+)\.self_attn)\."
    r"(q_proj|k_proj|v_proj|g_proj|f_a_proj|b_proj)\."
    r"(qweight_type|qweight)$"
)
_EMBED_TOKENS_RE = re.compile(r"^(model\.embed_tokens)\.(qweight_type|qweight)$")


def build_kimi_k3_name_map(config) -> dict[str, str]:
    """Build the exact text-only Kimi-K3 GGUF-to-checkpoint name map."""
    names = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
        "output_res_score.weight": "model.output_attn_res_score.weight",
    }

    for layer_idx in range(config.num_hidden_layers):
        gguf_prefix = f"blk.{layer_idx}"
        hf_prefix = f"model.layers.{layer_idx}"
        names.update(
            {
                f"{gguf_prefix}.attn_norm.weight": (
                    f"{hf_prefix}.input_layernorm.weight"
                ),
                f"{gguf_prefix}.attn_output.weight": (
                    f"{hf_prefix}.self_attn.o_proj.weight"
                ),
                f"{gguf_prefix}.attn_res_score.weight": (
                    f"{hf_prefix}.self_attention_res_score.weight"
                ),
                f"{gguf_prefix}.ffn_norm.weight": (
                    f"{hf_prefix}.post_attention_layernorm.weight"
                ),
                f"{gguf_prefix}.ffn_res_score.weight": (
                    f"{hf_prefix}.mlp_res_score.weight"
                ),
            }
        )

        if config.is_kda_layer(layer_idx):
            names.update(
                {
                    f"{gguf_prefix}.attn_q.weight": (
                        f"{hf_prefix}.self_attn.q_proj.weight"
                    ),
                    f"{gguf_prefix}.attn_k.weight": (
                        f"{hf_prefix}.self_attn.k_proj.weight"
                    ),
                    f"{gguf_prefix}.attn_v.weight": (
                        f"{hf_prefix}.self_attn.v_proj.weight"
                    ),
                    f"{gguf_prefix}.ssm_a": f"{hf_prefix}.self_attn.A_log",
                    f"{gguf_prefix}.ssm_beta.weight": (
                        f"{hf_prefix}.self_attn.b_proj.weight"
                    ),
                    f"{gguf_prefix}.ssm_conv1d_q.weight": (
                        f"{hf_prefix}.self_attn.q_conv1d.weight"
                    ),
                    f"{gguf_prefix}.ssm_conv1d_k.weight": (
                        f"{hf_prefix}.self_attn.k_conv1d.weight"
                    ),
                    f"{gguf_prefix}.ssm_conv1d_v.weight": (
                        f"{hf_prefix}.self_attn.v_conv1d.weight"
                    ),
                    f"{gguf_prefix}.ssm_dt.bias": (f"{hf_prefix}.self_attn.dt_bias"),
                    f"{gguf_prefix}.ssm_f_a.weight": (
                        f"{hf_prefix}.self_attn.f_a_proj.weight"
                    ),
                    f"{gguf_prefix}.ssm_f_b.weight": (
                        f"{hf_prefix}.self_attn.f_b_proj.weight"
                    ),
                    f"{gguf_prefix}.ssm_g.weight": (
                        f"{hf_prefix}.self_attn.g_proj.weight"
                    ),
                    f"{gguf_prefix}.ssm_norm.weight": (
                        f"{hf_prefix}.self_attn.o_norm.weight"
                    ),
                }
            )
        else:
            names.update(
                {
                    f"{gguf_prefix}.attn_gate.weight": (
                        f"{hf_prefix}.self_attn.g_proj.weight"
                    ),
                    f"{gguf_prefix}.attn_k_b.weight": (
                        f"{hf_prefix}.self_attn.k_b_proj.weight"
                    ),
                    f"{gguf_prefix}.attn_kv_a_mqa.weight": (
                        f"{hf_prefix}.self_attn.kv_a_proj_with_mqa.weight"
                    ),
                    f"{gguf_prefix}.attn_kv_a_norm.weight": (
                        f"{hf_prefix}.self_attn.kv_a_layernorm.weight"
                    ),
                    f"{gguf_prefix}.attn_q_a.weight": (
                        f"{hf_prefix}.self_attn.q_a_proj.weight"
                    ),
                    f"{gguf_prefix}.attn_q_a_norm.weight": (
                        f"{hf_prefix}.self_attn.q_a_layernorm.weight"
                    ),
                    f"{gguf_prefix}.attn_q_b.weight": (
                        f"{hf_prefix}.self_attn.q_b_proj.weight"
                    ),
                    f"{gguf_prefix}.attn_v_b.weight": (
                        f"{hf_prefix}.self_attn.v_b_proj.weight"
                    ),
                }
            )

        if layer_idx < config.first_k_dense_replace:
            names.update(
                {
                    f"{gguf_prefix}.ffn_gate.weight": (
                        f"{hf_prefix}.mlp.gate_proj.weight"
                    ),
                    f"{gguf_prefix}.ffn_down.weight": (
                        f"{hf_prefix}.mlp.down_proj.weight"
                    ),
                    f"{gguf_prefix}.ffn_up.weight": (f"{hf_prefix}.mlp.up_proj.weight"),
                }
            )
        else:
            moe_prefix = f"{hf_prefix}.block_sparse_moe"
            names.update(
                {
                    f"{gguf_prefix}.exp_probs_b.bias": (
                        f"{moe_prefix}.gate.e_score_correction_bias"
                    ),
                    f"{gguf_prefix}.ffn_gate_inp.weight": (f"{moe_prefix}.gate.weight"),
                    f"{gguf_prefix}.ffn_gate_exps.weight": (
                        f"{moe_prefix}.experts.0.w1.weight"
                    ),
                    f"{gguf_prefix}.ffn_down_exps.weight": (
                        f"{moe_prefix}.experts.0.w2.weight"
                    ),
                    f"{gguf_prefix}.ffn_up_exps.weight": (
                        f"{moe_prefix}.experts.0.w3.weight"
                    ),
                    f"{gguf_prefix}.ffn_gate_shexp.weight": (
                        f"{moe_prefix}.shared_experts.gate_proj.weight"
                    ),
                    f"{gguf_prefix}.ffn_down_shexp.weight": (
                        f"{moe_prefix}.shared_experts.down_proj.weight"
                    ),
                    f"{gguf_prefix}.ffn_up_shexp.weight": (
                        f"{moe_prefix}.shared_experts.up_proj.weight"
                    ),
                    f"{gguf_prefix}.ffn_routed_down.weight": (
                        f"{moe_prefix}.routed_expert_down_proj.weight"
                    ),
                    f"{gguf_prefix}.ffn_routed_norm.weight": (
                        f"{moe_prefix}.routed_expert_norm.weight"
                    ),
                    f"{gguf_prefix}.ffn_routed_up.weight": (
                        f"{moe_prefix}.routed_expert_up_proj.weight"
                    ),
                }
            )
    return names


class KimiK3GGUFWeightsAdapter(GGUFWeightsAdapter):
    """Text-only Kimi-K3 adapter for llama.cpp PR #26185 GGUF files."""

    @classmethod
    def matches(cls, config) -> bool:
        return bool(getattr(config, KIMI_K3_GGUF_TEXT_MARKER, False))

    def build_name_map(self, model_config: ModelConfig) -> dict[str, str]:
        return build_kimi_k3_name_map(model_config.hf_config)

    @staticmethod
    def get_unquantized_modules(weight_type_map: dict[str, str]) -> list[str]:
        modules = GGUFWeightsAdapter.get_unquantized_modules(weight_type_map)
        unquantized = set(modules)

        # vLLM packs these checkpoint projections into one LinearBase module.
        # The generic GGUF precision scan sees only the unfused checkpoint
        # names, so add the actual destination module when every shard is
        # unquantized.
        fused_groups: dict[str, tuple[str, ...]] = {}
        for name in weight_type_map:
            if match := re.match(
                r"^(.+\.(?:mlp|shared_experts))\.(?:gate|up)_proj\.weight$",
                name,
            ):
                prefix = match.group(1)
                fused_groups[f"{prefix}.gate_up_proj"] = (
                    f"{prefix}.gate_proj",
                    f"{prefix}.up_proj",
                )
            if match := re.match(
                r"^(model\.layers\.\d+\.self_attn)\."
                r"(?:q_a_proj|kv_a_proj_with_mqa)\.weight$",
                name,
            ):
                prefix = match.group(1)
                fused_groups[f"{prefix}.fused_qkv_a_proj"] = (
                    f"{prefix}.q_a_proj",
                    f"{prefix}.kv_a_proj_with_mqa",
                )
            if match := re.match(
                r"^(model\.layers\.\d+\.block_sparse_moe)\."
                r"experts\.0\.w[123]\.weight$",
                name,
            ):
                prefix = match.group(1)
                fused_groups[f"{prefix}.experts"] = (
                    f"{prefix}.experts.0.w1",
                    f"{prefix}.experts.0.w2",
                    f"{prefix}.experts.0.w3",
                )

        for fused_name, shard_names in fused_groups.items():
            shard_states = [name in unquantized for name in shard_names]
            if any(shard_states) and not all(shard_states):
                raise ValueError(
                    "Kimi-K3 fused projection mixes quantized and "
                    f"unquantized shards: {fused_name}"
                )
            if all(shard_states):
                modules.append(fused_name)

        for name in weight_type_map:
            match = re.match(
                r"^(model\.layers\.\d+\.self_attn)\.[kv]_b_proj\.weight$",
                name,
            )
            if match:
                fused_name = f"{match.group(1)}.kv_b_proj"
                if fused_name not in modules:
                    modules.append(fused_name)
            match = re.match(
                r"^(model\.layers\.\d+\.self_attn)\.b_proj\.weight$",
                name,
            )
            if match:
                # vLLM packs q/k/v/g/f_a/b into one KDA projection. Load that
                # fused projection natively so its independently stored GGUF
                # components can be reconstructed before vLLM packs them.
                fused_name = f"{match.group(1)}.in_proj_qkvgfab"
                if fused_name not in modules:
                    modules.append(fused_name)
        return modules

    @staticmethod
    def _dequantize_q8_0(
        name: str,
        qweight: torch.Tensor,
        qweight_type: torch.Tensor,
    ) -> torch.Tensor:
        weight_type = GGMLQuantizationType(int(qweight_type.item()))
        if weight_type is not GGMLQuantizationType.Q8_0:
            raise ValueError(
                f"Kimi-K3 reconstructed tensor must be Q8_0, got "
                f"{weight_type.name}: {name}"
            )
        array = dequantize(qweight.detach().cpu().numpy(), weight_type)
        return torch.from_numpy(array)

    @staticmethod
    def _fuse_mla_kv_b_tensors(
        prefix: str,
        k_b: torch.Tensor,
        v_b: torch.Tensor,
    ) -> torch.Tensor:
        if (
            k_b.ndim != 3
            or v_b.ndim != 3
            or k_b.shape[0] != v_b.shape[0]
            or k_b.shape[1] != v_b.shape[2]
        ):
            raise ValueError(
                f"Incompatible Kimi-K3 MLA split shapes for {prefix}: "
                f"k_b={tuple(k_b.shape)}, v_b={tuple(v_b.shape)}"
            )
        # llama.cpp stores K as [heads, kv_rank, qk_nope_dim], while vLLM
        # expects [heads * (qk_nope_dim + v_dim), kv_rank].
        fused = torch.cat((k_b.transpose(1, 2), v_b), dim=1)
        return fused.reshape(-1, k_b.shape[1]).contiguous()

    def _fuse_mla_kv_b(
        self,
        prefix: str,
        parts: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        k_b = self._dequantize_q8_0(
            f"{prefix}.k_b_proj",
            parts["k_qweight"],
            parts["k_qweight_type"],
        )
        v_b = self._dequantize_q8_0(
            f"{prefix}.v_b_proj",
            parts["v_qweight"],
            parts["v_qweight_type"],
        )
        return self._fuse_mla_kv_b_tensors(prefix, k_b, v_b)

    def map_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        mla_parts: dict[str, dict[str, torch.Tensor]] = {}
        routed_parts: dict[str, dict[str, torch.Tensor]] = {}
        kda_parts: dict[str, dict[str, torch.Tensor]] = {}
        embedding_parts: dict[str, torch.Tensor] = {}
        required_mla_parts = {
            "k_qweight",
            "k_qweight_type",
            "v_qweight",
            "v_qweight_type",
        }
        for name, weight in weights:
            if match := _EMBED_TOKENS_RE.match(name):
                prefix, value_kind = match.groups()
                embedding_parts[value_kind] = weight
                if embedding_parts.keys() >= {"qweight", "qweight_type"}:
                    # KimiLinearModel intentionally constructs embed_tokens
                    # without quant_config, so feed its native BF16 weight.
                    yield (
                        f"{prefix}.weight",
                        self._dequantize_q8_0(
                            prefix,
                            embedding_parts["qweight"],
                            embedding_parts["qweight_type"],
                        ),
                    )
                    embedding_parts.clear()
                continue

            if match := _KDA_IN_PROJ_RE.match(name):
                layer_prefix, layer_idx, component, value_kind = match.groups()
                if self.config.is_kda_layer(int(layer_idx)):
                    prefix = f"{layer_prefix}.{component}"
                    parts = kda_parts.setdefault(prefix, {})
                    parts[value_kind] = weight
                    if parts.keys() >= {"qweight", "qweight_type"}:
                        yield (
                            f"{prefix}.weight",
                            self._dequantize_q8_0(
                                prefix,
                                parts["qweight"],
                                parts["qweight_type"],
                            ),
                        )
                        del kda_parts[prefix]
                    continue

            if match := _ROUTED_PROJ_RE.match(name):
                prefix, value_kind = match.groups()
                parts = routed_parts.setdefault(prefix, {})
                parts[value_kind] = weight
                if parts.keys() >= {"qweight", "qweight_type"}:
                    yield (
                        f"{prefix}.weight",
                        self._dequantize_q8_0(
                            prefix,
                            parts["qweight"],
                            parts["qweight_type"],
                        ),
                    )
                    del routed_parts[prefix]
                continue

            if match := _KV_B_PART_RE.match(name):
                prefix, part, value_kind = match.groups()
                parts = mla_parts.setdefault(prefix, {})
                parts[f"{part}_{value_kind}"] = weight
                if parts.keys() >= {"k_weight", "v_weight"}:
                    yield (
                        f"{prefix}.kv_b_proj.weight",
                        self._fuse_mla_kv_b_tensors(
                            prefix,
                            parts["k_weight"],
                            parts["v_weight"],
                        ),
                    )
                    del mla_parts[prefix]
                elif parts.keys() >= required_mla_parts:
                    yield (
                        f"{prefix}.kv_b_proj.weight",
                        self._fuse_mla_kv_b(prefix, parts),
                    )
                    del mla_parts[prefix]
                continue

            if name.endswith(".A_log"):
                if not torch.all(weight < 0):
                    raise ValueError(
                        f"Kimi-K3 folded A tensor must be negative: {name}"
                    )
                yield name, torch.log(-weight.float())
                continue

            if re.search(r"\.self_attn\.[qkv]_conv1d\.weight$", name):
                if weight.ndim == 4 and weight.shape[0] == 1:
                    weight = weight.squeeze(0)
                if weight.ndim != 3 or weight.shape[1] != 1:
                    raise ValueError(
                        "Kimi-K3 GGUF conv1d tensor must have layout "
                        "[d_inner, 1, kernel] or [1, d_inner, 1, kernel], "
                        f"got {tuple(weight.shape)}: {name}"
                    )
                yield name, weight
                continue

            residual_targets = {
                "model.output_attn_res_score.weight": (
                    "model.output_attn_res_norm.weight",
                    "model.output_attn_res_proj.weight",
                ),
                ".self_attention_res_score.weight": (
                    ".self_attention_res_norm.weight",
                    ".self_attention_res_proj.weight",
                ),
                ".mlp_res_score.weight": (
                    ".mlp_res_norm.weight",
                    ".mlp_res_proj.weight",
                ),
            }
            for suffix, (norm_suffix, proj_suffix) in residual_targets.items():
                if name.endswith(suffix):
                    base = name[: -len(suffix)]
                    yield base + norm_suffix, torch.ones_like(weight)
                    yield base + proj_suffix, weight.reshape(1, -1)
                    break
            else:
                yield from super().map_weights([(name, weight)])

        if mla_parts:
            missing = {
                prefix: sorted(required_mla_parts - parts.keys())
                for prefix, parts in mla_parts.items()
            }
            raise ValueError(f"Incomplete Kimi-K3 MLA split tensors: {missing}")
        if routed_parts:
            missing = {
                prefix: sorted({"qweight", "qweight_type"} - parts.keys())
                for prefix, parts in routed_parts.items()
            }
            raise ValueError(f"Incomplete Kimi-K3 routed projection tensors: {missing}")
        if kda_parts:
            missing = {
                prefix: sorted({"qweight", "qweight_type"} - parts.keys())
                for prefix, parts in kda_parts.items()
            }
            raise ValueError(
                f"Incomplete Kimi-K3 KDA input projection tensors: {missing}"
            )
        if embedding_parts:
            missing = sorted({"qweight", "qweight_type"} - embedding_parts.keys())
            raise ValueError(f"Incomplete Kimi-K3 embedding tensor: {missing}")
