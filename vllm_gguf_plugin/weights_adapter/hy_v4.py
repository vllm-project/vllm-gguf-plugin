# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import gguf
import torch
from vllm.logger import init_logger
from vllm.model_executor.models.utils import WeightsMapper

from ..gguf_files import GGUFModelFiles
from ..gguf_utils import maybe_patch_hf_config_from_gguf
from ..weight_utils import get_gguf_tensor_names, split_stacked_experts
from .base import BaseGGUFWeightsAdapter, GGUFWeight

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig

logger = init_logger(__name__)


def build_hy_v4_mapper() -> WeightsMapper:
    return WeightsMapper(
        orig_to_new_prefix={
            "token_embd.": "model.embed_tokens.",
            "blk.": "model.layers.",
            "output_norm.": "model.norm.",
            "output.": "lm_head.",
        },
        orig_to_new_substr={
            # Indexer (DSA) tensors. These rules must precede the plain
            # attention rules because e.g. "indexer.attn_q_b." also contains
            # the "attn_q_b." substring.
            "indexer.attn_q_b.": "self_attn.indexer.wq_b.",
            "indexer.attn_k.": "self_attn.indexer.wk.",
            "indexer.k_norm.": "self_attn.indexer.k_norm.",
            "indexer.proj.": "self_attn.indexer.weights_proj.",
            # MLA attention. kv_b_proj is split into (transposed) k_b and v_b
            # in GGUF; tag both parts so transform_weights can merge them back.
            "attn_q_a_norm.": "self_attn.q_a_layernorm.",
            "attn_q_a.": "self_attn.q_a_proj.",
            "attn_q_b.": "self_attn.q_b_proj.",
            "attn_kv_a_mqa.": "self_attn.kv_a_proj_with_mqa.",
            "attn_kv_a_norm.": "self_attn.kv_a_layernorm.",
            "attn_k_b.weight": "self_attn.kv_b_proj.k_b.weight",
            "attn_v_b.weight": "self_attn.kv_b_proj.v_b.weight",
            "attn_output.": "self_attn.o_proj.",
            "attn_gate.": "self_attn.linear_gate.",
            "attn_sinks.weight": "self_attn.learnable_sink_param",
            "attn_norm.": "input_layernorm.",
            # iHC (hyper-connections). hc_fn is renamed to hc_fn.weight by
            # the model's load_weights, so map to the bare module path.
            "hc_attn_fn.weight": "hc_attn_layer.hc_pre.hc_fn",
            "hc_attn_base.weight": "hc_attn_layer.hc_pre.hc_base",
            "hc_attn_scale.weight": "hc_attn_layer.hc_pre.hc_scale",
            "hc_ffn_fn.weight": "hc_mlp_layer.hc_pre.hc_fn",
            "hc_ffn_base.weight": "hc_mlp_layer.hc_pre.hc_base",
            "hc_ffn_scale.weight": "hc_mlp_layer.hc_pre.hc_scale",
            "output_hc_fn.weight": "model.hc_head.hc_head_fn",
            "output_hc_base.weight": "model.hc_head.hc_head_base",
            "output_hc_scale.weight": "model.hc_head.hc_head_scale",
            # MoE. Specific prefixes must precede the dense ffn rules.
            "exp_probs_b.bias": "mlp.gate.e_score_correction_bias",
            "ffn_gate_inp.": "mlp.gate.",
            "ffn_gate_exps.": "mlp.experts.0.gate_proj.",
            "ffn_up_exps.": "mlp.experts.0.up_proj.",
            "ffn_down_exps.": "mlp.experts.0.down_proj.",
            "ffn_gate_shexp.": "mlp.shared_experts.gate_proj.",
            "ffn_up_shexp.": "mlp.shared_experts.up_proj.",
            "ffn_down_shexp.": "mlp.shared_experts.down_proj.",
            "ffn_norm.": "post_attention_layernorm.",
            "ffn_gate.": "mlp.gate_proj.",
            "ffn_up.": "mlp.up_proj.",
            "ffn_down.": "mlp.down_proj.",
        },
    )


def _dequantize_gguf(qweight: torch.Tensor, qweight_type: int | None) -> torch.Tensor:
    if qweight_type is None:
        # The GGUF tensor is stored unquantized (F32/BF16/F16).
        return qweight.float()
    qtype = gguf.GGMLQuantizationType(qweight_type)
    dequantized = gguf.quants.dequantize(qweight.numpy(), qtype)
    return torch.from_numpy(dequantized)


def merge_kv_b_proj(k_b: torch.Tensor, v_b: torch.Tensor) -> torch.Tensor:
    """Undo the GGUF converter's kv_b_proj split.

    k_b is [n_head, kv_lora_rank, qk_nope_head_dim] (transposed layout) and
    v_b is [n_head, v_head_dim, kv_lora_rank]; the fused checkpoint weight is
    [n_head * (qk_nope_head_dim + v_head_dim), kv_lora_rank].
    """
    n_head, kv_lora_rank, qk_nope_head_dim = k_b.shape
    v_head_dim = v_b.shape[1]
    k_b = k_b.transpose(1, 2)
    kv_b = torch.cat([k_b, v_b], dim=1)
    return kv_b.reshape(n_head * (qk_nope_head_dim + v_head_dim), kv_lora_rank)


class HYV4GGUFAdapter(BaseGGUFWeightsAdapter):
    """Adapter for HY V4 (hyv4) GGUF models, e.g. AngelSlim/Hy4-preview-GGUF.

    Serve with the HF repo as config/tokenizer source::

        vllm serve AngelSlim/Hy4-preview-GGUF:Q4_K_M \
            --tokenizer tencent/Hy4-preview

    The GGUF carries no MTP (nextn) weights, so speculative decoding is not
    available from it.
    """

    # kv_b_proj is reconstructed by dequantizing the GGUF attn_k_b/attn_v_b
    # tensors, so it loads as a plain (unquantized) linear weight.
    extra_unquantized_modules = ("kv_b_proj",)

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type == "hy_v4"

    @classmethod
    def architecture(cls, config) -> str | None:
        return "HYV4ForCausalLM"

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
        mapper = build_hy_v4_mapper()
        gguf_names = sorted(get_gguf_tensor_names(files.backbone))
        hf_names = mapper.apply_list(gguf_names)
        name_map = dict(zip(gguf_names, hf_names, strict=True))
        unmapped = [gguf for gguf, hf in name_map.items() if gguf == hf]
        if unmapped:
            raise ValueError(
                f"Unmapped HY V4 GGUF tensor(s): {unmapped}. "
                "The file does not look like a hyv4 GGUF model."
            )
        return name_map

    def transform_weights(
        self,
        weights: Iterable[GGUFWeight],
        model_config: ModelConfig,
    ) -> Iterable[GGUFWeight]:
        dtype = model_config.dtype

        def transformed() -> Iterable[GGUFWeight]:
            kv_b_types: dict[str, int] = {}
            # layer prefix -> {"k_b"/"v_b": (qweight, qweight_type)}
            pending_kv_b: dict[str, dict[str, tuple[torch.Tensor, int]]] = {}
            # base name (without .qweight) -> qweight_type, for indexer wk
            pending_wk_types: dict[str, int] = {}

            for name, weight in weights:
                if ".kv_b_proj.k_b." in name or ".kv_b_proj.v_b." in name:
                    if name.endswith(".qweight_type"):
                        kv_b_types[name.removesuffix(".qweight_type")] = int(
                            weight.item()
                        )
                        continue
                    base = name.removesuffix(".qweight").removesuffix(".weight")
                    prefix, _, part = base.rpartition(".kv_b_proj.")
                    parts = pending_kv_b.setdefault(prefix, {})
                    parts[part] = (weight, kv_b_types.pop(base, None))
                    if all(p in parts for p in ("k_b", "v_b")):
                        k_b = _dequantize_gguf(*parts["k_b"])
                        v_b = _dequantize_gguf(*parts["v_b"])
                        del pending_kv_b[prefix]
                        merged = merge_kv_b_proj(k_b, v_b).to(dtype)
                        yield f"{prefix}.kv_b_proj.weight", merged
                    continue
                if name.endswith(".indexer.wk.qweight_type"):
                    base = name.removesuffix(".qweight_type")
                    pending_wk_types[base] = int(weight.item())
                    continue
                if name.endswith((".indexer.wk.qweight", ".indexer.wk.weight")):
                    # indexer.wk_weights_proj is built without a quant config,
                    # so a GGUF-quantized wk is dequantized to a dense weight.
                    base = name.removesuffix(".qweight").removesuffix(".weight")
                    qweight_type = pending_wk_types.pop(base, None)
                    yield (
                        f"{base}.weight",
                        _dequantize_gguf(weight, qweight_type).to(dtype),
                    )
                    continue
                yield name, weight

        yield from split_stacked_experts(transformed())
