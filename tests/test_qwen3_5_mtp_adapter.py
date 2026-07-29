# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from transformers import PretrainedConfig

from vllm_gguf_plugin.weights_adapter import get_weights_adapter
from vllm_gguf_plugin.weights_adapter.qwen3_5_mtp import Qwen35MtpGGUFAdapter


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


class TestNameMap:
    def test_maps_block_to_hf_mtp_names(self):
        name_map = Qwen35MtpGGUFAdapter.build_mtp_name_map(40)
        assert name_map["blk.40.nextn.eh_proj.weight"] == "mtp.fc.weight"
        assert (
            name_map["blk.40.nextn.enorm.weight"] == "mtp.pre_fc_norm_embedding.weight"
        )
        assert name_map["blk.40.nextn.shared_head_norm.weight"] == "mtp.norm.weight"
        assert (
            name_map["blk.40.attn_norm.weight"] == "mtp.layers.0.input_layernorm.weight"
        )
        assert (
            name_map["blk.40.ffn_gate_inp_shexp.weight"]
            == "mtp.layers.0.mlp.shared_expert_gate.weight"
        )

    def test_covers_every_mtp_tensor_once(self):
        name_map = Qwen35MtpGGUFAdapter.build_mtp_name_map(40)
        assert len(name_map) == 20
        assert len(set(name_map.values())) == 20
        assert all(key.startswith("blk.40.") for key in name_map)

    def test_block_index_is_applied(self):
        assert set(Qwen35MtpGGUFAdapter.build_mtp_name_map(47)) == {
            key.replace("blk.40.", "blk.47.")
            for key in Qwen35MtpGGUFAdapter.build_mtp_name_map(40)
        }


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
