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


class TestInferRepoId:
    def test_remote_quant_reference(self):
        model_config = SimpleNamespace(
            model_weights="unsloth/Qwen3.5-0.8B-GGUF:UD-IQ2_XXS", model=None
        )
        assert (
            Qwen35GGUFAdapter._infer_repo_id(model_config)
            == "unsloth/Qwen3.5-0.8B-GGUF"
        )

    def test_repo_file_reference(self):
        model_config = SimpleNamespace(
            model_weights="unsloth/Qwen3.5-0.8B-GGUF/model-Q4_K_M.gguf", model=None
        )
        assert (
            Qwen35GGUFAdapter._infer_repo_id(model_config)
            == "unsloth/Qwen3.5-0.8B-GGUF"
        )

    def test_local_file_returns_none(self, tmp_path):
        gguf_path = tmp_path / "model.gguf"
        gguf_path.write_bytes(b"GGUF")
        model_config = SimpleNamespace(model_weights=str(gguf_path), model=None)
        assert Qwen35GGUFAdapter._infer_repo_id(model_config) is None

    def test_local_dir_with_quant_returns_none(self, tmp_path):
        model_config = SimpleNamespace(model_weights=f"{tmp_path}:Q8_0", model=None)
        assert Qwen35GGUFAdapter._infer_repo_id(model_config) is None


class TestResolveHfCacheDir:
    def test_hub_cache_layout_returns_cache_root(self, tmp_path):
        snapshot = (
            tmp_path / "models--unsloth--Qwen3.5-0.8B-GGUF" / "snapshots" / "abc123"
        )
        model_path = snapshot / "model-Q4_K_M.gguf"
        assert Qwen35GGUFAdapter._resolve_hf_cache_dir(str(model_path)) == str(tmp_path)

    def test_plain_path_returns_none(self, tmp_path):
        model_path = tmp_path / "gguf" / "model.gguf"
        assert Qwen35GGUFAdapter._resolve_hf_cache_dir(str(model_path)) is None


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
