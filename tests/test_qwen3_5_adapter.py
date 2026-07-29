# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import gguf
import pytest
import torch
from transformers import PretrainedConfig

from vllm_gguf_plugin.weight_utils import split_stacked_experts
from vllm_gguf_plugin.weights_adapter import get_weights_adapter
from vllm_gguf_plugin.weights_adapter.qwen3_5 import (
    Qwen35GGUFAdapter,
    build_qwen35_text_mapper,
    build_qwen35_vision_mapper,
)


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


class TestTextMapper:
    def _map(self, name, is_multimodal=False, is_moe=False):
        mapper = build_qwen35_text_mapper(is_multimodal=is_multimodal, is_moe=is_moe)
        return mapper.apply_list([name])[0]

    @pytest.mark.parametrize(
        ("gguf_name", "hf_name"),
        [
            ("token_embd.weight", "model.embed_tokens.weight"),
            ("output_norm.weight", "model.norm.weight"),
            ("output.weight", "lm_head.weight"),
            ("blk.3.attn_q.weight", "model.layers.3.self_attn.q_proj.weight"),
            ("blk.3.attn_q_norm.weight", "model.layers.3.self_attn.q_norm.weight"),
            ("blk.3.attn_norm.weight", "model.layers.3.input_layernorm.weight"),
            (
                "blk.3.post_attention_norm.weight",
                "model.layers.3.post_attention_layernorm.weight",
            ),
            # GDN: llama.cpp writes the fused in_proj halves as attn_qkv/attn_gate
            (
                "blk.3.attn_qkv.weight",
                "model.layers.3.linear_attn.in_proj_qkv.weight",
            ),
            ("blk.3.attn_gate.weight", "model.layers.3.linear_attn.in_proj_z.weight"),
            ("blk.3.ssm_alpha.weight", "model.layers.3.linear_attn.in_proj_a.weight"),
            ("blk.3.ssm_beta.weight", "model.layers.3.linear_attn.in_proj_b.weight"),
            ("blk.3.ssm_out.weight", "model.layers.3.linear_attn.out_proj.weight"),
            ("blk.3.ssm_norm.weight", "model.layers.3.linear_attn.norm.weight"),
            ("blk.3.ssm_conv1d.weight", "model.layers.3.linear_attn.conv1d.weight"),
            # these two are bare params in the HF checkpoint
            ("blk.3.ssm_a.weight", "model.layers.3.linear_attn.A_log"),
            ("blk.3.ssm_dt.bias", "model.layers.3.linear_attn.dt_bias"),
        ],
    )
    def test_maps_text_tensors(self, gguf_name, hf_name):
        assert self._map(gguf_name) == hf_name

    def test_multimodal_uses_language_model_prefix(self):
        assert (
            self._map("blk.3.attn_q.weight", is_multimodal=True)
            == "model.language_model.layers.3.self_attn.q_proj.weight"
        )
        assert self._map("output.weight", is_multimodal=True) == "lm_head.weight"

    @pytest.mark.parametrize(
        ("gguf_name", "hf_name"),
        [
            ("blk.1.ffn_gate.weight", "model.layers.1.mlp.gate_proj.weight"),
            ("blk.1.ffn_up.weight", "model.layers.1.mlp.up_proj.weight"),
            ("blk.1.ffn_down.weight", "model.layers.1.mlp.down_proj.weight"),
        ],
    )
    def test_maps_dense_mlp(self, gguf_name, hf_name):
        assert self._map(gguf_name) == hf_name

    @pytest.mark.parametrize(
        ("gguf_name", "hf_name"),
        [
            ("blk.1.ffn_gate_inp.weight", "model.layers.1.mlp.gate.weight"),
            (
                "blk.1.ffn_gate_inp_shexp.weight",
                "model.layers.1.mlp.shared_expert_gate.weight",
            ),
            (
                "blk.1.ffn_gate_exps.weight",
                "model.layers.1.mlp.experts.0.gate_proj.weight",
            ),
            ("blk.1.ffn_up_exps.weight", "model.layers.1.mlp.experts.0.up_proj.weight"),
            (
                "blk.1.ffn_down_exps.weight",
                "model.layers.1.mlp.experts.0.down_proj.weight",
            ),
            (
                "blk.1.ffn_up_shexp.weight",
                "model.layers.1.mlp.shared_expert.up_proj.weight",
            ),
        ],
    )
    def test_maps_moe_mlp(self, gguf_name, hf_name):
        assert self._map(gguf_name, is_moe=True) == hf_name

    def test_maps_quantized_names_like_plain_ones(self):
        assert (
            self._map("blk.3.attn_qkv.qweight")
            == "model.layers.3.linear_attn.in_proj_qkv.qweight"
        )
        assert (
            self._map("blk.3.attn_qkv.qweight_type")
            == "model.layers.3.linear_attn.in_proj_qkv.qweight_type"
        )


class TestArchCoverage:
    """A GGUF tensor with no mapper rule is skipped with a warning, and GGUF
    loading has no completeness check to catch it, so keep the tables in sync
    with the tensors gguf-py declares for these architectures."""

    # Fused gate_up experts; llama.cpp writes the split form and the HF
    # checkpoint keeps them separate, so nothing consumes this name.
    KNOWN_UNMAPPED = {"ffn_gate_up_exps"}

    @pytest.mark.parametrize(
        ("arch_name", "is_moe"), [("QWEN35", False), ("QWEN35MOE", True)]
    )
    def test_every_arch_tensor_has_a_rule(self, arch_name, is_moe):
        arch = getattr(gguf.MODEL_ARCH, arch_name)
        mapper = build_qwen35_text_mapper(is_multimodal=False, is_moe=is_moe)
        base_names = {
            base for _, base in gguf.get_tensor_name_map(arch, 1).mapping.values()
        }
        unmapped = sorted(
            base
            for base in base_names
            if base.rsplit(".", 1)[-1] not in self.KNOWN_UNMAPPED
            # llama.cpp appends the suffix; a tensor is covered if either form maps
            and all(
                mapper.apply_list([f"{base}.{suffix}"])[0] == f"{base}.{suffix}"
                for suffix in ("weight", "bias")
            )
        )
        assert unmapped == []


class TestVisionMapper:
    def _map(self, name):
        return build_qwen35_vision_mapper().apply_list([name])[0]

    @pytest.mark.parametrize(
        ("gguf_name", "hf_name"),
        [
            ("v.patch_embd.weight", "model.visual.patch_embed.proj.weight"),
            ("v.position_embd.weight", "model.visual.pos_embed.weight"),
            ("v.blk.2.attn_qkv.weight", "model.visual.blocks.2.attn.qkv.weight"),
            ("v.blk.2.attn_out.bias", "model.visual.blocks.2.attn.proj.bias"),
            ("v.blk.2.ffn_up.weight", "model.visual.blocks.2.mlp.linear_fc1.weight"),
            ("v.blk.2.ffn_down.bias", "model.visual.blocks.2.mlp.linear_fc2.bias"),
            ("v.blk.2.ln1.weight", "model.visual.blocks.2.norm1.weight"),
            ("v.blk.2.ln2.bias", "model.visual.blocks.2.norm2.bias"),
            # llama.cpp writes merger.norm as v.post_ln and its MLP as mm.N
            ("mm.0.weight", "model.visual.merger.linear_fc1.weight"),
            ("mm.2.bias", "model.visual.merger.linear_fc2.bias"),
            ("v.post_ln.weight", "model.visual.merger.norm.weight"),
        ],
    )
    def test_maps_vision_tensors(self, gguf_name, hf_name):
        assert self._map(gguf_name) == hf_name


class TestSplitStackedExperts:
    def test_splits_stacked_expert_weight(self):
        name = "model.layers.0.mlp.experts.0.gate_proj.weight"
        out = dict(split_stacked_experts([(name, torch.zeros(3, 4, 2))]))
        assert list(out) == [
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.1.gate_proj.weight",
            "model.layers.0.mlp.experts.2.gate_proj.weight",
        ]
        assert all(w.shape == (4, 2) for w in out.values())

    def test_leaves_other_weights_untouched(self):
        name = "model.layers.0.self_attn.q_proj.weight"
        out = dict(split_stacked_experts([(name, torch.zeros(4, 2))]))
        assert list(out) == [name]


class TestGdnReorder:
    def _make_adapter(self, num_key_heads, num_value_heads):
        config = PretrainedConfig(
            model_type="qwen3_5",
            linear_num_key_heads=num_key_heads,
            linear_num_value_heads=num_value_heads,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
        )
        adapter = Qwen35GGUFAdapter(config)
        adapter._set_gdn_reorder()
        return adapter

    def test_no_reorder_when_value_heads_match_key_heads(self):
        adapter = self._make_adapter(num_key_heads=4, num_value_heads=4)
        assert adapter._reorder is None
        assert "ssm_out.weight" not in adapter._dequant_tensors
        assert "linear_attn.out_proj" not in adapter._forced_unquantized_modules()

    def test_grouped_value_heads_dequantize_and_unquantize_out_proj(self):
        # The column reorder needs a float out_proj, so the module must also be
        # kept out of the GGUF linear method; the two go together.
        adapter = self._make_adapter(num_key_heads=2, num_value_heads=8)
        assert adapter._reorder == {
            "num_k": 2,
            "r": 4,
            "head_k": 128,
            "head_v": 128,
        }
        assert "ssm_out.weight" in adapter._dequant_tensors
        assert "linear_attn.out_proj" in adapter._forced_unquantized_modules()

    def test_always_forces_lm_head_and_embed_tokens(self):
        adapter = self._make_adapter(num_key_heads=4, num_value_heads=4)
        assert adapter._forced_unquantized_modules() == ["lm_head", "embed_tokens"]


class TestTransformWeights:
    def _transform(self, adapter, weights):
        return dict(adapter.transform_weight(weights))

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
