# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from transformers import PretrainedConfig

from vllm_gguf_plugin.weights_adapter import get_weights_adapter
from vllm_gguf_plugin.weights_adapter.qwen3_5_mtp import (
    Qwen35MtpGGUFAdapter,
    build_qwen35_mtp_mapper,
)


class TestMatches:
    @pytest.mark.parametrize("model_type", ["qwen3_5_mtp", "qwen3_5_moe_mtp"])
    def test_matches_mtp_model_types(self, model_type):
        config = PretrainedConfig(model_type=model_type)
        assert Qwen35MtpGGUFAdapter.matches(config)
        assert isinstance(get_weights_adapter(config), Qwen35MtpGGUFAdapter)

    @pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe"])
    def test_does_not_match_target_model_types(self, model_type):
        config = PretrainedConfig(model_type=model_type)
        assert not Qwen35MtpGGUFAdapter.matches(config)


# HF names of the 20 GGUF tensors an unsloth *-MTP-GGUF keeps in its extra
# block; these are what the draft's load_weights expects.
_MOE_BLOCK = {
    "nextn.eh_proj.weight": "mtp.fc.weight",
    "nextn.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
    "nextn.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
    "nextn.shared_head_norm.weight": "mtp.norm.weight",
    "attn_norm.weight": "mtp.layers.0.input_layernorm.weight",
    "post_attention_norm.weight": "mtp.layers.0.post_attention_layernorm.weight",
    "attn_q.weight": "mtp.layers.0.self_attn.q_proj.weight",
    "attn_k.weight": "mtp.layers.0.self_attn.k_proj.weight",
    "attn_v.weight": "mtp.layers.0.self_attn.v_proj.weight",
    "attn_output.weight": "mtp.layers.0.self_attn.o_proj.weight",
    "attn_q_norm.weight": "mtp.layers.0.self_attn.q_norm.weight",
    "attn_k_norm.weight": "mtp.layers.0.self_attn.k_norm.weight",
    "ffn_gate_inp.weight": "mtp.layers.0.mlp.gate.weight",
    "ffn_gate_exps.weight": "mtp.layers.0.mlp.experts.0.gate_proj.weight",
    "ffn_up_exps.weight": "mtp.layers.0.mlp.experts.0.up_proj.weight",
    "ffn_down_exps.weight": "mtp.layers.0.mlp.experts.0.down_proj.weight",
    "ffn_gate_shexp.weight": "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
    "ffn_up_shexp.weight": "mtp.layers.0.mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp.weight": "mtp.layers.0.mlp.shared_expert.down_proj.weight",
    "ffn_gate_inp_shexp.weight": "mtp.layers.0.mlp.shared_expert_gate.weight",
}


class TestNameMap:
    def _map(self, suffixes, block=40, is_moe=True):
        mapper = build_qwen35_mtp_mapper(block, is_moe=is_moe)
        return mapper.apply_list([f"blk.{block}.{suffix}" for suffix in suffixes])

    def test_maps_every_moe_block_tensor(self):
        assert self._map(_MOE_BLOCK) == list(_MOE_BLOCK.values())

    def test_block_index_is_applied(self):
        assert self._map(_MOE_BLOCK, block=47) == list(_MOE_BLOCK.values())

    def test_maps_dense_ffn(self):
        assert self._map(["ffn_gate.weight", "ffn_up.weight"], is_moe=False) == [
            "mtp.layers.0.mlp.gate_proj.weight",
            "mtp.layers.0.mlp.up_proj.weight",
        ]

    def test_leaves_backbone_blocks_alone(self):
        mapper = build_qwen35_mtp_mapper(40, is_moe=True)
        # Other blocks keep their "blk." prefix, so the adapter drops them.
        assert mapper.apply_list(["blk.39.attn_q.weight"])[0].startswith("blk.39.")


class TestTransformWeights:
    def _transform(self, weights):
        return dict(Qwen35MtpGGUFAdapter.transform_weight(weights))

    @pytest.mark.parametrize(
        "name",
        [
            "mtp.norm.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.layers.0.input_layernorm.weight",
            "mtp.layers.0.post_attention_layernorm.weight",
            "mtp.layers.0.self_attn.q_norm.weight",
            "mtp.layers.0.self_attn.k_norm.weight",
        ],
    )
    def test_undoes_norm_offset(self, name):
        out = self._transform([(name, torch.full((8,), 3.0))])
        assert torch.equal(out[name], torch.full((8,), 2.0))

    def test_restores_shared_expert_gate_output_dim(self):
        name = "mtp.layers.0.mlp.shared_expert_gate.weight"
        out = self._transform([(name, torch.zeros(2048))])
        assert out[name].shape == (1, 2048)

    def test_leaves_quantized_params_untouched(self):
        weights = [
            ("mtp.layers.0.self_attn.q_proj.qweight", torch.ones(8, 4)),
            ("mtp.layers.0.self_attn.q_proj.qweight_type", torch.tensor(14)),
        ]
        out = self._transform(weights)
        assert torch.equal(
            out["mtp.layers.0.self_attn.q_proj.qweight"], torch.ones(8, 4)
        )
        assert out["mtp.layers.0.self_attn.q_proj.qweight_type"] == 14
