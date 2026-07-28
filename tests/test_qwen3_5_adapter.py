# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

from vllm_gguf_plugin.weights_adapter import get_weights_adapter
from vllm_gguf_plugin.weights_adapter.qwen3_5 import Qwen35GGUFAdapter


class TestMatches:
    @pytest.mark.parametrize(
        "model_type",
        ["qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"],
    )
    def test_matches_qwen35_model_types(self, model_type):
        config = PretrainedConfig(model_type=model_type)
        assert Qwen35GGUFAdapter.matches(config)
        assert isinstance(get_weights_adapter(config), Qwen35GGUFAdapter)

    @pytest.mark.parametrize("model_type", ["qwen3", "qwen3_moe", "gemma3"])
    def test_does_not_match_other_model_types(self, model_type):
        config = PretrainedConfig(model_type=model_type)
        assert not Qwen35GGUFAdapter.matches(config)


class TestMergerNameMap:
    def test_maps_merger_tensors(self):
        name_map = {}
        Qwen35GGUFAdapter._map_qwen35_merger(name_map)
        assert name_map == {
            "mm.0.weight": "model.visual.merger.linear_fc1.weight",
            "mm.0.bias": "model.visual.merger.linear_fc1.bias",
            "mm.2.weight": "model.visual.merger.linear_fc2.weight",
            "mm.2.bias": "model.visual.merger.linear_fc2.bias",
            # llama.cpp writes merger.norm as v.post_ln, not mm.input_norm
            "v.post_ln.weight": "model.visual.merger.norm.weight",
            "v.post_ln.bias": "model.visual.merger.norm.bias",
        }


class TestTransformWeights:
    def _transform(self, adapter, weights):
        return dict(adapter._transform_qwen35_weights(weights))

    def _make_adapter(self, temporal_patch_size=None):
        config = PretrainedConfig(model_type="qwen3_5")
        if temporal_patch_size is not None:
            config.vision_config = SimpleNamespace(
                temporal_patch_size=temporal_patch_size
            )
        return Qwen35GGUFAdapter(config)

    def test_skips_quant_params_for_forced_unquantized_modules(self):
        adapter = self._make_adapter()
        out = self._transform(
            adapter,
            [
                ("lm_head.qweight", torch.zeros(4, 2)),
                ("lm_head.qweight_type", torch.tensor(8)),
                ("model.embed_tokens.qweight", torch.zeros(4, 2)),
                ("lm_head.weight", torch.zeros(4, 2)),
            ],
        )
        assert list(out) == ["lm_head.weight"]

    def test_expands_conv1d_to_3d(self):
        adapter = self._make_adapter()
        name = "model.layers.0.linear_attn.conv1d.weight"
        out = self._transform(adapter, [(name, torch.zeros(8, 4))])
        assert out[name].shape == (8, 1, 4)

    def test_restores_output_dim_of_flattened_weight(self):
        adapter = self._make_adapter()
        name = "model.layers.0.mlp.shared_expert_gate.weight"
        out = self._transform(adapter, [(name, torch.zeros(16))])
        assert out[name].shape == (1, 16)

    def test_keeps_norm_weight_1d(self):
        adapter = self._make_adapter()
        name = "model.norm.weight"
        out = self._transform(adapter, [(name, torch.zeros(16))])
        assert out[name].shape == (16,)

    def test_stacks_patch_embed_temporal_slices(self):
        adapter = self._make_adapter(temporal_patch_size=2)
        name = "model.visual.patch_embed.proj.weight"
        first, second = torch.ones(8, 3, 14, 14), torch.zeros(8, 3, 14, 14)
        out = self._transform(adapter, [(name, first), (f"{name}.1", second)])
        assert list(out) == [name]
        assert out[name].shape == (8, 3, 2, 14, 14)
        assert torch.equal(out[name][:, :, 0], first)
        assert torch.equal(out[name][:, :, 1], second)

    def test_stacks_patch_embed_slices_out_of_order(self):
        adapter = self._make_adapter(temporal_patch_size=2)
        name = "model.visual.patch_embed.proj.weight"
        first, second = torch.ones(8, 3, 14, 14), torch.zeros(8, 3, 14, 14)
        out = self._transform(adapter, [(f"{name}.1", second), (name, first)])
        assert torch.equal(out[name][:, :, 0], first)

    def test_keeps_patch_embed_without_temporal_expansion(self):
        adapter = self._make_adapter(temporal_patch_size=1)
        name = "model.visual.patch_embed.proj.weight"
        out = self._transform(adapter, [(name, torch.ones(8, 3, 14, 14))])
        assert out[name].shape == (8, 3, 14, 14)
