# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import vllm.config.model as model_config_module
import vllm.engine.arg_utils as arg_utils_module
import vllm.model_executor.layers.quantization as quantization_module
import vllm.model_executor.layers.vocab_parallel_embedding as vocab_embedding_module
import vllm.model_executor.parameter as parameter_module
import vllm.transformers_utils.config as config_module
from transformers import PretrainedConfig
from vllm.config.load import LoadConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.linear import (
    WEIGHT_LOADER_V2_SUPPORTED,
    MergedColumnParallelLinear,
    QKVParallelLinear,
)
from vllm.model_executor.layers.quantization import (
    _CUSTOMIZED_METHOD_TO_QUANT_CONFIG,
    QUANTIZATION_METHODS,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader import get_model_loader
from vllm.transformers_utils.config import get_config_parser

if (
    "gguf" in QUANTIZATION_METHODS
    and os.environ.get("VLLM_GGUF_PLUGIN_OVERRIDE_IN_TREE") != "1"
):
    pytest.skip(
        "override-mode plugin tests require vLLM without in-tree GGUF or "
        "VLLM_GGUF_PLUGIN_OVERRIDE_IN_TREE=1",
        allow_module_level=True,
    )

import vllm_gguf_plugin.config_parser as gguf_config_parser_module
import vllm_gguf_plugin.gguf_tokenizer_builder as gguf_tokenizer_builder_module
import vllm_gguf_plugin.gguf_utils as gguf_utils_module
import vllm_gguf_plugin.plugin as gguf_plugin_module
import vllm_gguf_plugin.quantization as gguf_quantization
import vllm_gguf_plugin.quantization.config as gguf_config_module
import vllm_gguf_plugin.weight_utils as weight_utils_module
import vllm_gguf_plugin.weights_adapter.default as default_adapter_module
import vllm_gguf_plugin.weights_adapter.qwen3_5 as qwen3_5_adapter_module
from vllm_gguf_plugin import OOTGGUFConfig, OOTGGUFModelLoader, register
from vllm_gguf_plugin.config_parser import GGUFConfigParser
from vllm_gguf_plugin.gguf_tokenizer_builder import build_tokenizer_from_gguf
from vllm_gguf_plugin.quantization import (
    GGUFModelOptNvFp4FusedMoE,
    GGUFNvFp4LinearMethod,
    GGUFUninitializedParameter,
    GGUFWeightParameter,
    GGUFWeightTypeParameter,
)
from vllm_gguf_plugin.quantization.nvfp4 import (
    iter_gguf_nvfp4_native_moe_sidecar_weights,
    split_gguf_nvfp4_moe_weight,
    split_gguf_nvfp4_weight,
)
from vllm_gguf_plugin.weights_adapter import get_weights_adapter
from vllm_gguf_plugin.weights_adapter.base import GGUFLoadSpec
from vllm_gguf_plugin.weights_adapter.default import (
    GGUFWeightsAdapter,
    _add_gemma4_gguf_mappings,
    _add_gemma4_mtp_gguf_mappings,
    _add_nvfp4_sidecar_mappings,
    _add_qwen3_5_mtp_gguf_mappings,
)
from vllm_gguf_plugin.weights_adapter.gemma3 import Gemma3GGUFAdapter
from vllm_gguf_plugin.weights_adapter.gemma4 import Gemma4GGUFAdapter
from vllm_gguf_plugin.weights_adapter.qwen3_5 import Qwen3_5GGUFAdapter


def test_register_overrides_gguf_config():
    register()

    assert _CUSTOMIZED_METHOD_TO_QUANT_CONFIG["gguf"] is OOTGGUFConfig
    assert quantization_module.get_quantization_config("gguf") is OOTGGUFConfig


def test_register_overrides_gguf_loader():
    register()

    model_loader = get_model_loader(LoadConfig(load_format="gguf"))

    assert isinstance(model_loader, OOTGGUFModelLoader)


def test_register_is_idempotent():
    register()
    register()

    assert _CUSTOMIZED_METHOD_TO_QUANT_CONFIG["gguf"] is OOTGGUFConfig
    assert quantization_module.get_quantization_config("gguf") is OOTGGUFConfig
    assert isinstance(
        get_model_loader(LoadConfig(load_format="gguf")), OOTGGUFModelLoader
    )
    assert isinstance(get_config_parser("gguf"), GGUFConfigParser)


def test_oot_config_reuses_in_tree_behavior():
    quant_config = OOTGGUFConfig.from_config({})

    assert isinstance(quant_config, OOTGGUFConfig)
    assert quant_config.get_name() == "gguf"
    assert repr(quant_config) == "GGUFConfig()"


def test_adapter_registry_selects_qwen35_gemma3_and_gemma4():
    assert isinstance(
        get_weights_adapter(PretrainedConfig(model_type="qwen3_5")),
        Qwen3_5GGUFAdapter,
    )
    assert isinstance(
        get_weights_adapter(PretrainedConfig(model_type="qwen3_5_moe")),
        Qwen3_5GGUFAdapter,
    )
    assert isinstance(
        get_weights_adapter(PretrainedConfig(model_type="gemma3")),
        Gemma3GGUFAdapter,
    )
    assert isinstance(
        get_weights_adapter(PretrainedConfig(model_type="gemma4")),
        Gemma4GGUFAdapter,
    )
    assert isinstance(
        get_weights_adapter(PretrainedConfig(model_type="gemma4_assistant")),
        Gemma4GGUFAdapter,
    )


def test_gemma3_adapter_requires_mmproj_for_multimodal_layout(tmp_path, monkeypatch):
    main_path = tmp_path / "model.gguf"
    main_path.touch()
    adapter = Gemma3GGUFAdapter(PretrainedConfig(model_type="gemma3"))
    multimodal_config = PretrainedConfig(
        model_type="gemma3",
        architectures=["Gemma3ForConditionalGeneration"],
    )
    multimodal_config.vision_config = PretrainedConfig()
    model_config = SimpleNamespace(
        hf_config=multimodal_config,
        language_model_only=False,
    )

    monkeypatch.setattr(adapter, "patch_hf_config", lambda model, config: config)
    monkeypatch.setattr(
        "vllm_gguf_plugin.weights_adapter.gemma3.detect_gguf_multimodal",
        lambda model: None,
    )

    with pytest.raises(ValueError, match="requires an mmproj sidecar"):
        adapter.prepare_loading(str(main_path), model_config)


def test_gemma3_adapter_allows_language_model_only_without_mmproj(
    tmp_path, monkeypatch
):
    main_path = tmp_path / "model.gguf"
    main_path.touch()
    adapter = Gemma3GGUFAdapter(PretrainedConfig(model_type="gemma3"))
    multimodal_config = PretrainedConfig(
        model_type="gemma3",
        architectures=["Gemma3ForConditionalGeneration"],
    )
    multimodal_config.vision_config = PretrainedConfig()
    model_config = SimpleNamespace(
        hf_config=multimodal_config,
        language_model_only=True,
    )

    monkeypatch.setattr(adapter, "patch_hf_config", lambda model, config: config)
    monkeypatch.setattr(
        "vllm_gguf_plugin.weights_adapter.gemma3.detect_gguf_multimodal",
        lambda model: None,
    )
    monkeypatch.setattr(
        "vllm_gguf_plugin.weights_adapter.gemma3.get_gguf_unquantized_params",
        lambda gguf_files: [],
    )

    load_spec = adapter.prepare_loading(str(main_path), model_config)

    assert load_spec is adapter.load_spec
    assert load_spec.weights_source == [str(main_path)]
    assert load_spec.unquantized_modules == []


def test_gemma4_adapter_transforms_quantized_moe_names():
    adapter = Gemma4GGUFAdapter(PretrainedConfig(model_type="gemma4"))
    qweight = torch.ones((2, 2), dtype=torch.uint8)
    qweight_type = torch.tensor(2, dtype=torch.uint8)

    mapped = list(
        adapter.map_weights(
            [
                (
                    "model.layers.0.experts.gate_up_proj.qweight_type",
                    qweight_type,
                ),
                ("model.layers.0.experts.gate_up_proj.qweight", qweight),
                ("model.layers.0.experts.down_proj.qweight_type", qweight_type),
                ("model.layers.0.experts.down_proj.qweight", qweight),
            ]
        )
    )

    assert mapped[0][0] == (
        "model.layers.0.moe.experts.routed_experts.w13_qweight_type"
    )
    assert mapped[1][0] == "model.layers.0.moe.experts.routed_experts.w13_qweight"
    assert mapped[2][0] == "model.layers.0.moe.experts.routed_experts.w2_qweight_type"
    assert mapped[3][0] == "model.layers.0.moe.experts.routed_experts.w2_qweight"


def test_gemma4_adapter_promotes_packed_nvfp4_moe_modules():
    adapter = Gemma4GGUFAdapter(PretrainedConfig(model_type="gemma4"))
    adapter._native_nvfp4_modules = {
        "model.layers.0.experts.gate_up_proj",
        "model.layers.0.experts.down_proj",
        "model.layers.1.experts.gate_up_proj",
        "model.layers.2.mlp.down_proj",
    }
    load_spec = GGUFLoadSpec(
        weights_source=["model.gguf"],
        unquantized_modules=[],
        nvfp4_modules=sorted(adapter._native_nvfp4_modules),
    )

    adapter._promote_native_nvfp4_moe_modules(load_spec)

    assert adapter._native_nvfp4_modules == {"model.layers.2.mlp.down_proj"}
    assert load_spec.nvfp4_modules == ["model.layers.2.mlp.down_proj"]
    assert load_spec.nvfp4_moe_modules == ["model.layers.0.moe.experts"]
    assert adapter._native_nvfp4_gemma4_moe_projection_modules == {
        "model.layers.0.experts.gate_up_proj": "w13",
        "model.layers.0.experts.down_proj": "w2",
    }


def test_gemma4_adapter_maps_packed_nvfp4_moe_to_native_weights():
    adapter = Gemma4GGUFAdapter(PretrainedConfig(model_type="gemma4"))
    adapter._native_nvfp4_gemma4_moe_projection_modules = {
        "model.layers.0.experts.gate_up_proj": "w13",
    }
    qweight_type = torch.tensor(
        int(default_adapter_module.gguf.GGMLQuantizationType.NVFP4)
    )
    qweight = torch.cat(
        (
            torch.full((2, 3, 4), 0x38, dtype=torch.uint8),
            torch.arange(2 * 3 * 32, dtype=torch.uint8).reshape(2, 3, 32),
        ),
        dim=2,
    )

    mapped = list(
        adapter.map_weights(
            [
                ("model.layers.0.experts.gate_up_proj.qweight_type", qweight_type),
                ("model.layers.0.experts.gate_up_proj.qweight", qweight),
            ]
        )
    )

    assert [name for name, _ in mapped] == [
        "model.layers.0.moe.experts.w13_weight",
        "model.layers.0.moe.experts.w13_weight_scale",
        "model.layers.0.moe.experts.w13_weight_scale_2",
        "model.layers.0.moe.experts.w13_input_scale",
    ]
    assert mapped[0][1].shape == (2, 3, 32)
    assert mapped[1][1].shape == (2, 3, 4)
    assert mapped[1][1].dtype == torch.float8_e4m3fn
    assert torch.equal(mapped[2][1], torch.ones((2, 2), dtype=torch.float32))
    assert torch.equal(mapped[3][1], torch.ones((2, 2), dtype=torch.float32))


def test_gemma4_adapter_maps_packed_nvfp4_moe_sidecars():
    adapter = Gemma4GGUFAdapter(PretrainedConfig(model_type="gemma4"))
    adapter._native_nvfp4_gemma4_moe_projection_modules = {
        "model.layers.0.experts.gate_up_proj": "w13",
        "model.layers.0.experts.down_proj": "w2",
    }

    mapped = list(
        adapter.map_weights(
            [
                (
                    "model.layers.0.experts.gate_up_proj.weight_scale_2",
                    torch.tensor([0.25, 0.5]),
                ),
                (
                    "model.layers.0.experts.down_proj.input_scale",
                    torch.tensor([1.25, 1.5]),
                ),
            ]
        )
    )

    assert [name for name, _ in mapped] == [
        "model.layers.0.moe.experts.w13_weight_scale_2",
        "model.layers.0.moe.experts.w2_input_scale",
    ]
    assert torch.equal(
        mapped[0][1],
        torch.tensor([[0.25, 0.25], [0.5, 0.5]], dtype=torch.float32),
    )
    assert torch.equal(mapped[1][1], torch.tensor([1.25, 1.5]))


def test_gemma4_adapter_flattens_patch_embed_weight():
    adapter = Gemma4GGUFAdapter(PretrainedConfig(model_type="gemma4"))
    weight = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)

    transformed = adapter.transform_weight(
        "model.vision_tower.patch_embedder.input_proj.weight",
        weight,
    )

    assert transformed.shape == (2, 60)
    assert torch.equal(transformed, weight.flatten(1))


def test_gemma4_gguf_mappings_cover_moe_and_vision():
    config = PretrainedConfig(model_type="gemma4", num_hidden_layers=2)
    config.text_config = PretrainedConfig(num_hidden_layers=2)
    config.vision_config = PretrainedConfig(num_hidden_layers=1)
    config.architectures = ["Gemma4ForConditionalGeneration"]
    mapping: dict[str, str] = {}
    sideload_params = []

    _add_gemma4_gguf_mappings(config, mapping, sideload_params)

    assert mapping["blk.0.ffn_gate_up_exps.weight"] == (
        "model.language_model.layers.0.experts.gate_up_proj.weight"
    )
    assert mapping["blk.1.layer_output_scale.weight"] == (
        "model.language_model.layers.1.layer_scalar"
    )
    assert mapping["v.patch_embd.weight"] == (
        "model.vision_tower.patch_embedder.input_proj.weight"
    )
    assert mapping["v.blk.0.attn_q.weight"] == (
        "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight"
    )
    assert sideload_params[0].fullmatch(
        "model.language_model.layers.0.experts.gate_up_proj.weight"
    )


def test_gemma4_text_only_does_not_add_vision_mappings():
    config = PretrainedConfig(
        architectures=["Gemma4ForCausalLM"],
        model_type="gemma4",
        num_hidden_layers=1,
    )
    config.text_config = PretrainedConfig(num_hidden_layers=1)
    config.vision_config = PretrainedConfig(num_hidden_layers=1)
    mapping: dict[str, str] = {}
    sideload_params = []

    _add_gemma4_gguf_mappings(config, mapping, sideload_params)

    assert mapping["blk.0.ffn_gate_up_exps.weight"] == (
        "model.layers.0.experts.gate_up_proj.weight"
    )
    assert "v.patch_embd.weight" not in mapping


def test_gemma4_mtp_gguf_mappings():
    config = PretrainedConfig(model_type="gemma4_assistant", num_hidden_layers=2)
    mapping: dict[str, str] = {}

    _add_gemma4_mtp_gguf_mappings(config, mapping)

    assert mapping["nextn.pre_projection.weight"] == "model.pre_projection.weight"
    assert mapping["blk.0.attn_q.weight"] == "model.layers.0.self_attn.q_proj.weight"
    assert "blk.0.attn_k.weight" not in mapping
    assert mapping["blk.1.layer_output_scale.weight"] == "model.layers.1.layer_scalar"


def test_default_adapter_adds_mmproj_only_for_multimodal_layout(tmp_path, monkeypatch):
    main_path = tmp_path / "model.gguf"
    mmproj_path = tmp_path / "mmproj.gguf"
    adapter = GGUFWeightsAdapter(PretrainedConfig(model_type="qwen3_5_moe"))
    multimodal_config = PretrainedConfig(
        model_type="qwen3_5_moe",
        architectures=["Qwen3_5MoeForConditionalGeneration"],
    )
    multimodal_config.vision_config = PretrainedConfig()
    text_config = PretrainedConfig(
        model_type="qwen3_5",
        architectures=["Qwen3_5ForCausalLM"],
    )
    text_config.vision_config = PretrainedConfig()

    monkeypatch.setattr(
        default_adapter_module,
        "detect_gguf_multimodal",
        lambda model: mmproj_path,
    )

    assert adapter._get_weight_sources(str(main_path), multimodal_config) == [
        str(main_path),
        str(mmproj_path),
    ]
    assert adapter._get_weight_sources(str(main_path), text_config) == [str(main_path)]


def test_default_adapter_requires_mmproj_for_multimodal_layout(tmp_path, monkeypatch):
    main_path = tmp_path / "model.gguf"
    adapter = GGUFWeightsAdapter(PretrainedConfig(model_type="qwen3_5_moe"))
    multimodal_config = PretrainedConfig(
        model_type="qwen3_5_moe",
        architectures=["Qwen3_5MoeForConditionalGeneration"],
    )
    multimodal_config.vision_config = PretrainedConfig()

    monkeypatch.setattr(
        default_adapter_module,
        "detect_gguf_multimodal",
        lambda model: None,
    )

    with pytest.raises(ValueError, match="requires an mmproj sidecar"):
        adapter._get_weight_sources(
            str(main_path),
            multimodal_config,
            use_multimodal_weight_layout=True,
            require_multimodal_sidecar=True,
        )

    assert adapter._get_weight_sources(
        str(main_path),
        multimodal_config,
        use_multimodal_weight_layout=True,
        require_multimodal_sidecar=False,
    ) == [str(main_path)]


class _FakeTensorNameMap:
    def get_name(self, name):
        return None


class _FakeModel:
    def __init__(self, state_names):
        self._state_names = state_names

    def state_dict(self):
        return {name: torch.empty((), device="meta") for name in self._state_names}


def test_qwen35_multimodal_mapping_includes_visual_and_linear_attention(monkeypatch):
    config = PretrainedConfig(
        model_type="qwen3_5",
        architectures=["Qwen3_5ForConditionalGeneration"],
    )
    config.text_config = PretrainedConfig(
        num_hidden_layers=1,
        layer_types=["linear_attention"],
    )
    config.vision_config = PretrainedConfig(num_hidden_layers=1)
    model_config = type(
        "ModelConfig",
        (),
        {"hf_config": config, "trust_remote_code": False},
    )()

    monkeypatch.setattr(default_adapter_module.gguf, "MODEL_ARCH_NAMES", {1: "qwen35"})
    monkeypatch.setattr(
        default_adapter_module.gguf,
        "get_tensor_name_map",
        lambda arch, num_layers: _FakeTensorNameMap(),
    )
    monkeypatch.setattr(
        default_adapter_module.AutoModelForImageTextToText,
        "from_config",
        lambda *args, **kwargs: _FakeModel(
            ["model.language_model.layers.0.linear_attn.dt_bias"]
        ),
    )

    mapping = GGUFWeightsAdapter(config).build_name_map(model_config)

    assert mapping["token_embd.weight"] == "model.language_model.embed_tokens.weight"
    assert mapping["v.patch_embd.weight.1"] == "model.visual.patch_embed.proj.weight.1"
    assert mapping["mm.0.weight"] == "model.visual.merger.linear_fc1.weight"
    assert mapping["blk.0.ssm_dt.bias"] == (
        "model.language_model.layers.0.linear_attn.dt_bias"
    )


def test_qwen35_mtp_gguf_mappings_use_trunk_layer_count():
    config = PretrainedConfig(
        model_type="qwen3_5_moe",
        num_hidden_layers=40,
        mtp_num_hidden_layers=2,
    )
    mapping: dict[str, str] = {}
    sideload_params = []

    _add_qwen3_5_mtp_gguf_mappings(config, mapping, sideload_params)

    assert mapping["blk.40.attn_q.weight"] == "mtp.layers.0.self_attn.q_proj.weight"
    assert mapping["blk.41.attn_k.weight"] == "mtp.layers.1.self_attn.k_proj.weight"
    assert mapping["blk.40.ffn_gate_inp.weight"] == "mtp.layers.0.mlp.gate.weight"
    assert mapping["blk.40.nextn.eh_proj.weight"] == "mtp.fc.weight"
    assert sideload_params[0].fullmatch("mtp.layers.0.mlp.experts.15.gate_proj.weight")


def test_qwen35_adapter_combines_split_patch_embed_weight():
    adapter = Qwen3_5GGUFAdapter(PretrainedConfig(model_type="qwen3_5_moe"))
    first = torch.ones((2, 3), dtype=torch.float32)
    second = 2 * torch.ones((2, 3), dtype=torch.float32)

    mapped = list(
        adapter.map_weights(
            [
                ("model.visual.patch_embed.proj.weight", first),
                ("model.visual.patch_embed.proj.weight.1", second),
            ]
        )
    )

    assert len(mapped) == 1
    assert mapped[0][0] == "model.visual.patch_embed.proj.weight"
    assert torch.equal(mapped[0][1], torch.stack((first, second), dim=2))


def _qwen35_linear_attn_config():
    return PretrainedConfig(
        model_type="qwen3_5_moe",
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=1,
        linear_value_head_dim=1,
    )


def test_qwen35_adapter_restores_linear_attention_layout():
    adapter = Qwen3_5GGUFAdapter(_qwen35_linear_attn_config())
    # GGUF stores value heads tiled by value-head group; HF expects grouped by
    # key head. With 2 key heads and 2 value heads per key, [0, 1, 2, 3]
    # becomes [0, 2, 1, 3].
    stored_qkv = torch.arange(8, dtype=torch.float32).reshape(8, 1)

    restored = adapter.transform_weight(
        "model.layers.0.linear_attn.in_proj_qkv.weight",
        stored_qkv,
    )

    assert torch.equal(restored.squeeze(-1), torch.tensor([0, 1, 2, 3, 4, 6, 5, 7]))


def test_qwen35_adapter_dequantizes_forced_out_proj(monkeypatch):
    adapter = Qwen3_5GGUFAdapter(_qwen35_linear_attn_config())
    module_name = "model.layers.0.linear_attn.out_proj"
    adapter._forced_dequantized_modules.add(module_name)
    qweight_type = torch.tensor(
        int(qwen3_5_adapter_module.gguf.GGMLQuantizationType.Q4_0)
    )
    qweight = torch.ones((2, 2), dtype=torch.uint8)

    def fake_dequantize(weight, weight_type):
        assert weight_type == qwen3_5_adapter_module.gguf.GGMLQuantizationType.Q4_0
        return torch.ones((2, 2), dtype=torch.float32).numpy()

    monkeypatch.setattr(
        qwen3_5_adapter_module.gguf.quants,
        "dequantize",
        fake_dequantize,
    )

    mapped = list(
        adapter.map_weights(
            [
                (f"{module_name}.qweight_type", qweight_type),
                (f"{module_name}.qweight", qweight),
            ]
        )
    )

    assert mapped[0][0] == f"{module_name}.weight"
    assert torch.equal(mapped[0][1], torch.ones((2, 2), dtype=torch.float32))


def test_qwen35_prepare_loading_forces_token_embedding_dequant_without_vllm_support(
    monkeypatch,
):
    adapter = Qwen3_5GGUFAdapter(PretrainedConfig(model_type="qwen3_5"))
    load_spec = type(
        "LoadSpec",
        (),
        {
            "gguf_to_hf_name_map": {
                "token_embd.weight": "model.embed_tokens.weight",
            },
            "unquantized_modules": [],
        },
    )()

    monkeypatch.setattr(
        GGUFWeightsAdapter,
        "prepare_loading",
        lambda self, model_path, model_config: load_spec,
    )
    monkeypatch.setattr(
        qwen3_5_adapter_module,
        "_qwen3_5_embed_tokens_uses_quant_config",
        lambda: False,
    )

    result = adapter.prepare_loading(
        "model.gguf",
        type("ModelConfig", (), {})(),
    )

    assert result is load_spec
    assert "model.embed_tokens" in adapter._forced_dequantized_modules
    assert load_spec.unquantized_modules == ["model.embed_tokens"]


def test_qwen35_prepare_loading_keeps_token_embedding_quantized_with_vllm_support(
    monkeypatch,
):
    adapter = Qwen3_5GGUFAdapter(PretrainedConfig(model_type="qwen3_5"))
    load_spec = type(
        "LoadSpec",
        (),
        {
            "gguf_to_hf_name_map": {
                "token_embd.weight": "model.embed_tokens.weight",
            },
            "unquantized_modules": [],
        },
    )()

    monkeypatch.setattr(
        GGUFWeightsAdapter,
        "prepare_loading",
        lambda self, model_path, model_config: load_spec,
    )
    monkeypatch.setattr(
        qwen3_5_adapter_module,
        "_qwen3_5_embed_tokens_uses_quant_config",
        lambda: True,
    )

    result = adapter.prepare_loading(
        "model.gguf",
        type("ModelConfig", (), {})(),
    )

    assert result is load_spec
    assert "model.embed_tokens" not in adapter._forced_dequantized_modules
    assert load_spec.unquantized_modules == []


def test_qwen35_adapter_dequantizes_forced_token_embedding(monkeypatch):
    adapter = Qwen3_5GGUFAdapter(PretrainedConfig(model_type="qwen3_5"))
    module_name = "model.embed_tokens"
    adapter._forced_dequantized_modules.add(module_name)
    qweight_type = torch.tensor(
        int(qwen3_5_adapter_module.gguf.GGMLQuantizationType.Q6_K)
    )
    qweight = torch.ones((2, 2), dtype=torch.uint8)

    def fake_dequantize(weight, weight_type):
        assert weight_type == qwen3_5_adapter_module.gguf.GGMLQuantizationType.Q6_K
        return torch.full((2, 2), 3.0, dtype=torch.float32).numpy()

    monkeypatch.setattr(
        qwen3_5_adapter_module.gguf.quants,
        "dequantize",
        fake_dequantize,
    )

    mapped = list(
        adapter.map_weights(
            [
                (f"{module_name}.qweight_type", qweight_type),
                (f"{module_name}.qweight", qweight),
            ]
        )
    )

    assert len(mapped) == 1
    assert mapped[0][0] == f"{module_name}.weight"
    assert torch.equal(mapped[0][1], torch.full((2, 2), 3.0, dtype=torch.float32))


def test_update_tie_word_embeddings_uses_all_split_shards(monkeypatch):
    tensors_by_file = {
        "shard-1.gguf": ["token_embd.weight"],
        "shard-2.gguf": ["output.weight"],
    }

    class FakeGGUFReader:
        def __init__(self, path):
            self.tensors = [
                SimpleNamespace(name=name) for name in tensors_by_file[str(path)]
            ]

    monkeypatch.setattr(weight_utils_module.gguf, "GGUFReader", FakeGGUFReader)

    config = PretrainedConfig(model_type="qwen3", tie_word_embeddings=True)
    adapter = GGUFWeightsAdapter(config)
    adapter.update_tie_word_embeddings(
        ["shard-1.gguf", "shard-2.gguf"],
        config,
        {
            "token_embd.weight": "model.embed_tokens.weight",
            "output.weight": "lm_head.weight",
        },
    )

    assert config.tie_word_embeddings is False


def test_default_adapter_marks_dequantizable_fallback_types_unquantized():
    modules = GGUFWeightsAdapter.get_unquantized_modules(
        {
            "model.layers.0.mlp.down_proj.weight": "MXFP4",
            "model.layers.0.mlp.up_proj.weight": "TQ1_0",
            "model.layers.1.mlp.down_proj.weight": "Q4_K",
        }
    )

    assert modules == [
        "model.layers.0.mlp.down_proj",
        "model.layers.0.mlp.up_proj",
    ]


def test_gguf_iterator_dequantizes_dense_fallback_types(monkeypatch):
    weight_type = default_adapter_module.gguf.GGMLQuantizationType.MXFP4
    packed = np.arange(17, dtype=np.uint8).reshape(1, 17)

    class FakeGGUFReader:
        def __init__(self, path):
            self.byte_order = "L"
            self.tensors = [
                SimpleNamespace(
                    name="blk.0.ffn_down.weight",
                    tensor_type=weight_type,
                    data=packed,
                )
            ]

    def fake_dequantize(weight, quant_type):
        assert weight is packed
        assert quant_type == weight_type
        return np.full((2, 3), 7.0, dtype=np.float32)

    monkeypatch.setattr(weight_utils_module.gguf, "GGUFReader", FakeGGUFReader)
    monkeypatch.setattr(weight_utils_module.gguf.quants, "dequantize", fake_dequantize)

    weights = list(
        weight_utils_module.gguf_quant_weights_iterator_multi(
            ["model.gguf"],
            {"blk.0.ffn_down.weight": "model.layers.0.mlp.down_proj.weight"},
        )
    )

    assert [name for name, _ in weights] == ["model.layers.0.mlp.down_proj.weight"]
    assert torch.equal(weights[0][1], torch.full((2, 3), 7.0))


def test_gguf_unquantized_params_include_dense_fallback_types(monkeypatch):
    weight_type = default_adapter_module.gguf.GGMLQuantizationType.TQ2_0

    class FakeGGUFReader:
        def __init__(self, path):
            self.tensors = [
                SimpleNamespace(
                    name="blk.0.ffn_down.weight",
                    tensor_type=weight_type,
                )
            ]

    monkeypatch.setattr(weight_utils_module.gguf, "GGUFReader", FakeGGUFReader)

    assert weight_utils_module.get_gguf_unquantized_params(["model.gguf"]) == [
        "blk.0.ffn_down.weight"
    ]


def test_split_gguf_nvfp4_weight_to_native_tensors():
    scale_bytes = torch.tensor([[0x00, 0x38, 0x40, 0x48]], dtype=torch.uint8)
    packed_values = torch.arange(32, dtype=torch.uint8).reshape(1, 32)
    qweight = torch.cat((scale_bytes, packed_values), dim=1)
    grouped = packed_values.reshape(1, 1, 4, 8)
    values = torch.cat((grouped & 0x0F, grouped >> 4), dim=-1)
    expected_weight = (values[..., 0::2] | (values[..., 1::2] << 4)).reshape(1, 32)

    weight, weight_scale = split_gguf_nvfp4_weight(qweight)

    assert torch.equal(weight, expected_weight)
    assert weight_scale.dtype == torch.float8_e4m3fn
    assert torch.equal(
        weight_scale.to(torch.float32),
        torch.tensor([[0.0, 1.0, 2.0, 4.0]], dtype=torch.float32),
    )


def test_split_gguf_nvfp4_moe_weight_preserves_expert_rows():
    scale_bytes = torch.full((2, 3, 4), 0x38, dtype=torch.uint8)
    packed_values = torch.arange(2 * 3 * 32, dtype=torch.uint8).reshape(2, 3, 32)
    qweight = torch.cat((scale_bytes, packed_values), dim=2)

    weight, weight_scale = split_gguf_nvfp4_moe_weight(qweight)

    assert weight.shape == (2, 3, 32)
    assert weight_scale.shape == (2, 3, 4)
    assert weight_scale.dtype == torch.float8_e4m3fn

    block_packed = qweight.reshape(2, 3, 1, 36).repeat(1, 1, 2, 1)
    weight, weight_scale = split_gguf_nvfp4_moe_weight(block_packed)

    assert weight.shape == (2, 3, 64)
    assert weight_scale.shape == (2, 3, 8)


def test_iter_gguf_nvfp4_native_moe_sidecar_weights_expands_experts():
    module_name = "model.layers.0.mlp.experts.0.gate_proj"

    mapped = iter_gguf_nvfp4_native_moe_sidecar_weights(
        module_name,
        "weight_scale_2",
        torch.tensor([0.25, 0.5], dtype=torch.float32),
    )

    assert [name for name, _ in mapped] == [
        f"{module_name}.weight_scale_2",
        "model.layers.0.mlp.experts.1.gate_proj.weight_scale_2",
    ]
    assert [weight.shape for _, weight in mapped] == [torch.Size([]), torch.Size([])]
    assert torch.equal(mapped[0][1], torch.tensor(0.25))
    assert torch.equal(mapped[1][1], torch.tensor(0.5))


def test_nvfp4_sidecar_mappings_follow_weight_mappings_without_overwriting():
    mapping = {
        "blk.0.ffn_gate_exps.weight": ("model.layers.0.mlp.experts.0.gate_proj.weight"),
        "blk.0.ffn_gate_exps.scale": "model.layers.0.mlp.router.scale",
        "blk.0.ffn_up_exps.weight": "model.layers.0.mlp.experts.0.up_proj.weight",
    }

    _add_nvfp4_sidecar_mappings(mapping)

    assert mapping["blk.0.ffn_gate_exps.scale"] == "model.layers.0.mlp.router.scale"
    assert mapping["blk.0.ffn_gate_exps.input_scale"] == (
        "model.layers.0.mlp.experts.0.gate_proj.input_scale"
    )
    assert mapping["blk.0.ffn_up_exps.scale"] == (
        "model.layers.0.mlp.experts.0.up_proj.weight_scale_2"
    )
    assert mapping["blk.0.ffn_up_exps.input_scale"] == (
        "model.layers.0.mlp.experts.0.up_proj.input_scale"
    )


def test_default_adapter_maps_nvfp4_qweight_to_native_weights():
    adapter = GGUFWeightsAdapter(PretrainedConfig(model_type="qwen3"))
    module_name = "model.layers.0.mlp.down_proj"
    adapter._native_nvfp4_modules.add(module_name)
    qweight_type = torch.tensor(
        int(default_adapter_module.gguf.GGMLQuantizationType.NVFP4)
    )
    qweight = torch.cat(
        (
            torch.tensor([[0x38, 0x38, 0x38, 0x38]], dtype=torch.uint8),
            torch.arange(32, dtype=torch.uint8).reshape(1, 32),
        ),
        dim=1,
    )

    mapped = list(
        adapter.map_weights(
            [
                (f"{module_name}.qweight_type", qweight_type),
                (f"{module_name}.qweight", qweight),
            ]
        )
    )

    assert [name for name, _ in mapped] == [
        f"{module_name}.weight",
        f"{module_name}.weight_scale",
        f"{module_name}.weight_scale_2",
    ]
    grouped = qweight[:, 4:].reshape(1, 1, 4, 8)
    values = torch.cat((grouped & 0x0F, grouped >> 4), dim=-1)
    expected_weight = (values[..., 0::2] | (values[..., 1::2] << 4)).reshape(1, 32)
    assert torch.equal(mapped[0][1], expected_weight)
    assert mapped[1][1].dtype == torch.float8_e4m3fn
    assert torch.equal(mapped[2][1], torch.tensor(1.0, dtype=torch.float32))


def test_default_adapter_omits_nvfp4_dense_default_scale_when_sidecar_exists():
    adapter = GGUFWeightsAdapter(PretrainedConfig(model_type="qwen3"))
    module_name = "model.layers.0.mlp.down_proj"
    adapter._native_nvfp4_modules.add(module_name)
    adapter._native_nvfp4_sidecar_suffixes[module_name] = {"weight_scale_2"}
    qweight_type = torch.tensor(
        int(default_adapter_module.gguf.GGMLQuantizationType.NVFP4)
    )
    qweight = torch.cat(
        (
            torch.tensor([[0x38, 0x38, 0x38, 0x38]], dtype=torch.uint8),
            torch.arange(32, dtype=torch.uint8).reshape(1, 32),
        ),
        dim=1,
    )

    mapped = list(
        adapter.map_weights(
            [
                (f"{module_name}.qweight_type", qweight_type),
                (f"{module_name}.qweight", qweight),
            ]
        )
    )

    assert [name for name, _ in mapped] == [
        f"{module_name}.weight",
        f"{module_name}.weight_scale",
    ]


def test_default_adapter_maps_nvfp4_moe_qweight_to_native_weights():
    adapter = GGUFWeightsAdapter(PretrainedConfig(model_type="qwen3_moe"))
    module_name = "model.layers.0.mlp.experts.0.gate_proj"
    adapter._native_nvfp4_moe_projection_modules.add(module_name)
    qweight_type = torch.tensor(
        int(default_adapter_module.gguf.GGMLQuantizationType.NVFP4)
    )
    qweight = torch.cat(
        (
            torch.full((2, 3, 4), 0x38, dtype=torch.uint8),
            torch.arange(2 * 3 * 32, dtype=torch.uint8).reshape(2, 3, 32),
        ),
        dim=2,
    )

    mapped = list(
        adapter.map_weights(
            [
                (f"{module_name}.qweight_type", qweight_type),
                (f"{module_name}.qweight", qweight),
            ]
        )
    )

    assert [name for name, _ in mapped] == [
        f"{module_name}.weight",
        f"{module_name}.weight_scale",
        f"{module_name}.weight_scale_2",
        f"{module_name}.input_scale",
        "model.layers.0.mlp.experts.1.gate_proj.weight_scale_2",
        "model.layers.0.mlp.experts.1.gate_proj.input_scale",
    ]
    assert mapped[0][1].shape == (2, 3, 32)
    assert mapped[1][1].shape == (2, 3, 4)
    assert mapped[1][1].dtype == torch.float8_e4m3fn
    assert all(torch.equal(weight, torch.tensor(1.0)) for _, weight in mapped[2:])


def test_default_adapter_omits_nvfp4_moe_default_scales_when_sidecars_exist():
    adapter = GGUFWeightsAdapter(PretrainedConfig(model_type="qwen3_moe"))
    module_name = "model.layers.0.mlp.experts.0.gate_proj"
    adapter._native_nvfp4_moe_projection_modules.add(module_name)
    adapter._native_nvfp4_sidecar_suffixes[module_name] = {
        "weight_scale_2",
        "input_scale",
    }
    qweight_type = torch.tensor(
        int(default_adapter_module.gguf.GGMLQuantizationType.NVFP4)
    )
    qweight = torch.cat(
        (
            torch.full((2, 3, 4), 0x38, dtype=torch.uint8),
            torch.arange(2 * 3 * 32, dtype=torch.uint8).reshape(2, 3, 32),
        ),
        dim=2,
    )

    mapped = list(
        adapter.map_weights(
            [
                (f"{module_name}.qweight_type", qweight_type),
                (f"{module_name}.qweight", qweight),
            ]
        )
    )

    assert [name for name, _ in mapped] == [
        f"{module_name}.weight",
        f"{module_name}.weight_scale",
    ]


def test_default_adapter_maps_nvfp4_sidecars_only_for_native_modules():
    adapter = GGUFWeightsAdapter(PretrainedConfig(model_type="qwen3"))
    native_module = "model.layers.0.mlp.down_proj"
    non_native_module = "model.layers.1.mlp.down_proj"
    adapter._native_nvfp4_modules.add(native_module)

    mapped = list(
        adapter.map_weights(
            [
                (f"{native_module}.weight_scale_2", torch.tensor(0.25)),
                (f"{native_module}.input_scale", torch.tensor(1.25)),
                (f"{non_native_module}.weight_scale_2", torch.tensor(0.5)),
                (f"{non_native_module}.input_scale", torch.tensor(1.5)),
            ]
        )
    )

    assert [name for name, _ in mapped] == [
        f"{native_module}.weight_scale_2",
        f"{native_module}.input_scale",
    ]
    assert torch.equal(mapped[0][1], torch.tensor(0.25))
    assert torch.equal(mapped[1][1], torch.tensor(1.25))


def test_default_adapter_maps_nvfp4_moe_sidecars_to_native_scales():
    adapter = GGUFWeightsAdapter(PretrainedConfig(model_type="qwen3_moe"))
    module_name = "model.layers.0.mlp.experts.0.gate_proj"
    adapter._native_nvfp4_moe_projection_modules.add(module_name)

    mapped = list(
        adapter.map_weights(
            [
                (f"{module_name}.weight_scale_2", torch.tensor([0.25, 0.5])),
                (f"{module_name}.input_scale", torch.tensor([1.25, 1.5])),
            ]
        )
    )

    assert [name for name, _ in mapped] == [
        f"{module_name}.weight_scale_2",
        "model.layers.0.mlp.experts.1.gate_proj.weight_scale_2",
        f"{module_name}.input_scale",
        "model.layers.0.mlp.experts.1.gate_proj.input_scale",
    ]
    assert [weight.shape for _, weight in mapped] == [
        torch.Size([]),
        torch.Size([]),
        torch.Size([]),
        torch.Size([]),
    ]
    assert torch.equal(mapped[0][1], torch.tensor(0.25))
    assert torch.equal(mapped[1][1], torch.tensor(0.5))
    assert torch.equal(mapped[2][1], torch.tensor(1.25))
    assert torch.equal(mapped[3][1], torch.tensor(1.5))


def test_qwen35_adapter_maps_nvfp4_moe_sidecars_to_native_scales():
    adapter = Qwen3_5GGUFAdapter(PretrainedConfig(model_type="qwen3_5_moe"))
    module_name = "model.layers.0.mlp.experts.0.gate_proj"
    adapter._native_nvfp4_moe_projection_modules.add(module_name)

    mapped = list(
        adapter.map_weights(
            [
                (f"{module_name}.weight_scale_2", torch.tensor([0.25, 0.5])),
                (f"{module_name}.input_scale", torch.tensor([1.25, 1.5])),
            ]
        )
    )

    assert [name for name, _ in mapped] == [
        f"{module_name}.weight_scale_2",
        "model.layers.0.mlp.experts.1.gate_proj.weight_scale_2",
        f"{module_name}.input_scale",
        "model.layers.0.mlp.experts.1.gate_proj.input_scale",
    ]
    assert torch.equal(mapped[0][1], torch.tensor(0.25))
    assert torch.equal(mapped[1][1], torch.tensor(0.5))
    assert torch.equal(mapped[2][1], torch.tensor(1.25))
    assert torch.equal(mapped[3][1], torch.tensor(1.5))


def test_default_adapter_discovers_native_nvfp4_modules():
    modules = GGUFWeightsAdapter.get_native_nvfp4_modules(
        {
            "model.layers.0.mlp.down_proj.weight": "NVFP4",
            "model.layers.0.mlp.experts.0.gate_proj.weight": "NVFP4",
            "model.layers.0.mlp.experts.0.up_proj.weight": "NVFP4",
            "model.layers.0.mlp.experts.0.down_proj.weight": "NVFP4",
            "model.embed_tokens.weight": "NVFP4",
            "lm_head.weight": "NVFP4",
            "model.layers.0.self_attn.q_proj.weight": "Q4_K",
        }
    )

    assert modules == ["model.layers.0.mlp.down_proj"]


def test_default_adapter_discovers_native_nvfp4_moe_modules():
    weight_type_map = {
        "model.layers.0.mlp.experts.0.gate_proj.weight": "NVFP4",
        "model.layers.0.mlp.experts.0.up_proj.weight": "NVFP4",
        "model.layers.0.mlp.experts.0.down_proj.weight": "NVFP4",
        "model.layers.1.mlp.experts.0.gate_proj.weight": "NVFP4",
        "model.layers.1.mlp.experts.0.down_proj.weight": "NVFP4",
    }

    moe_modules = set(GGUFWeightsAdapter.get_native_nvfp4_moe_modules(weight_type_map))
    projection_modules = GGUFWeightsAdapter.get_native_nvfp4_moe_projection_modules(
        weight_type_map, moe_modules
    )

    assert moe_modules == {"model.layers.0.mlp.experts"}
    assert projection_modules == [
        "model.layers.0.mlp.experts.0.down_proj",
        "model.layers.0.mlp.experts.0.gate_proj",
        "model.layers.0.mlp.experts.0.up_proj",
    ]


def test_gguf_config_routes_nvfp4_linear_to_native_method(monkeypatch):
    register()
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )

    class FakeNvFp4Kernel:
        def process_weights_after_loading(self, layer):
            layer.processed = True

        def apply_weights(self, layer, x, bias=None):
            return torch.empty(x.shape[0], layer.output_size_per_partition)

    monkeypatch.setattr(
        "vllm_gguf_plugin.quantization.nvfp4.init_nvfp4_linear_kernel",
        lambda use_a16=False: FakeNvFp4Kernel(),
    )
    quant_config = OOTGGUFConfig(
        nvfp4_modules=[
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.up_proj",
        ]
    )
    quant_config.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    layer = MergedColumnParallelLinear(
        input_size=64,
        output_sizes=[32, 32],
        bias=False,
        quant_config=quant_config,
        prefix="model.layers.0.mlp.gate_up_proj",
        disable_tp=True,
    )

    assert "GGUFNvFp4LinearMethod" in WEIGHT_LOADER_V2_SUPPORTED
    assert isinstance(layer.quant_method, GGUFNvFp4LinearMethod)
    assert layer.weight.shape == (64, 32)
    assert layer.weight_scale.shape == (64, 4)
    assert layer.weight_scale.dtype == torch.float8_e4m3fn
    assert layer.weight_scale_2.shape == (2,)
    assert layer.input_scale.shape == (2,)

    layer.weight_loader_v2(layer.weight, torch.ones((32, 32), dtype=torch.uint8), 0)
    layer.weight_loader_v2(layer.weight, 2 * torch.ones((32, 32), dtype=torch.uint8), 1)
    layer.weight_loader_v2(
        layer.weight_scale,
        torch.ones((32, 4), dtype=torch.float32).to(torch.float8_e4m3fn),
        0,
    )
    layer.weight_loader_v2(
        layer.weight_scale,
        torch.full((32, 4), 2.0, dtype=torch.float32).to(torch.float8_e4m3fn),
        1,
    )
    layer.weight_loader_v2(layer.weight_scale_2, torch.tensor(1.0), 0)
    layer.weight_loader_v2(layer.weight_scale_2, torch.tensor(1.0), 1)
    layer.weight_loader_v2(layer.input_scale, torch.tensor(0.5), 0)
    layer.weight_loader_v2(layer.input_scale, torch.tensor(0.75), 1)
    layer.quant_method.process_weights_after_loading(layer)

    assert layer.processed is True
    assert not hasattr(layer, "weight_scale_2")
    assert not hasattr(layer, "input_scale")
    assert torch.equal(layer.weight_global_scale, torch.tensor(1.0))


def test_gguf_config_routes_nvfp4_moe_to_modelopt_native_method(monkeypatch):
    import vllm.model_executor.layers.quantization.modelopt as modelopt_module
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend

    class FakeRoutedExperts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.moe_config = SimpleNamespace(is_act_and_mul=True)

    monkeypatch.setattr(gguf_config_module, "RoutedExperts", FakeRoutedExperts)
    monkeypatch.setattr(
        modelopt_module,
        "select_nvfp4_moe_backend",
        lambda **kwargs: (NvFp4MoeBackend.MARLIN, object),
    )

    quant_config = OOTGGUFConfig(nvfp4_moe_modules=["model.layers.0.mlp.experts"])
    quant_config.packed_modules_mapping = {}
    method = quant_config.get_quant_method(
        FakeRoutedExperts(), "model.layers.0.mlp.experts"
    )

    assert isinstance(method, GGUFModelOptNvFp4FusedMoE)
    assert method.quant_config.quant_method == "W4A16_NVFP4"
    assert method.quant_config.group_size == 16


def test_gguf_linear_uses_weight_loader_v2(monkeypatch):
    register()
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    quant_config = OOTGGUFConfig.from_config({})
    layer = MergedColumnParallelLinear(
        input_size=4,
        output_sizes=[4, 4],
        bias=False,
        quant_config=quant_config,
        disable_tp=True,
    )

    assert "GGUFLinearMethod" in WEIGHT_LOADER_V2_SUPPORTED
    assert isinstance(layer.qweight, GGUFUninitializedParameter)
    assert isinstance(layer.qweight_type, GGUFUninitializedParameter)
    assert layer.qweight.weight_loader.__name__.endswith("weight_loader_v2")

    layer.weight_loader_v2(layer.qweight, torch.ones((4, 4), dtype=torch.uint8), 0)
    layer.weight_loader_v2(layer.qweight, 2 * torch.ones((4, 4), dtype=torch.uint8), 1)
    layer.weight_loader_v2(layer.qweight_type, torch.tensor(3, dtype=torch.uint8), 0)
    layer.weight_loader_v2(layer.qweight_type, torch.tensor(4, dtype=torch.uint8), 1)

    assert isinstance(layer.qweight, GGUFUninitializedParameter)
    assert len(layer.qweight.data_container) == 2
    assert isinstance(layer.qweight_type, GGUFUninitializedParameter)

    layer.quant_method.process_weights_after_loading(layer)

    assert isinstance(layer.qweight, GGUFWeightParameter)
    assert isinstance(layer.qweight_type, GGUFWeightTypeParameter)
    assert layer.qweight.shard_id == [0, 1]
    assert layer.qweight_type.shard_weight_type == {0: 3, 1: 4}


def test_gguf_weight_type_loader_stores_tuple_shards(monkeypatch):
    register()
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    quant_config = OOTGGUFConfig.from_config({})
    layer = MergedColumnParallelLinear(
        input_size=4,
        output_sizes=[4, 4],
        bias=False,
        quant_config=quant_config,
        disable_tp=True,
    )

    layer.qweight_type.weight_loader(
        layer.qweight_type,
        torch.tensor(3, dtype=torch.uint8),
        (0, 1),
    )

    assert layer.qweight_type.shard_weight_type == {0: 3, 1: 3}
    assert layer.qweight_type.weight_type == 3


def test_gguf_embedding_uses_plugin_weight_loader(monkeypatch):
    monkeypatch.setattr(
        vocab_embedding_module, "get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        vocab_embedding_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )

    layer = VocabParallelEmbedding(
        num_embeddings=10,
        embedding_dim=4,
        org_num_embeddings=10,
        padding_size=8,
        quant_config=OOTGGUFConfig.from_config({}),
    )

    loaded_qweight = torch.arange(60, dtype=torch.uint8).reshape(10, 6)
    layer.qweight.weight_loader(layer.qweight, loaded_qweight)
    layer.qweight_type.weight_loader(
        layer.qweight_type, torch.tensor(7, dtype=torch.uint8)
    )
    layer.quant_method.process_weights_after_loading(layer)

    assert isinstance(layer.qweight, GGUFWeightParameter)
    assert isinstance(layer.qweight_type, GGUFWeightTypeParameter)
    assert layer.qweight.shape == (16, 6)
    assert torch.equal(layer.qweight[:10], loaded_qweight)
    assert torch.equal(layer.qweight[10:], torch.zeros((6, 6), dtype=torch.uint8))
    assert torch.equal(layer.qweight_type, torch.tensor([7], dtype=torch.uint8))
    assert layer.qweight_type.weight_type == 7


def test_gguf_linear_same_type_shards_skip_concat(monkeypatch):
    register()
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )

    quant_config = OOTGGUFConfig.from_config({})
    layer = MergedColumnParallelLinear(
        input_size=4,
        output_sizes=[4, 4],
        bias=False,
        quant_config=quant_config,
        disable_tp=True,
    )
    layer.weight_loader_v2(layer.qweight, torch.ones((4, 4), dtype=torch.uint8), 0)
    layer.weight_loader_v2(layer.qweight, 2 * torch.ones((4, 4), dtype=torch.uint8), 1)
    layer.weight_loader_v2(layer.qweight_type, torch.tensor(3, dtype=torch.uint8), 0)
    layer.weight_loader_v2(layer.qweight_type, torch.tensor(3, dtype=torch.uint8), 1)
    layer.quant_method.process_weights_after_loading(layer)

    assert isinstance(layer.qweight, torch.nn.Parameter)
    calls: list[tuple[tuple[int, ...], int]] = []

    def fake_fused_mul_mat_gguf(x, qweight, qweight_type):
        calls.append((tuple(qweight.shape), qweight_type))
        return torch.zeros(
            (x.shape[0], qweight.shape[0]), dtype=x.dtype, device=x.device
        )

    monkeypatch.setattr(
        gguf_quantization, "fused_mul_mat_gguf", fake_fused_mul_mat_gguf
    )
    out = layer.quant_method.apply(layer, torch.ones((2, 4), dtype=torch.float32))

    assert calls == [((8, 4), 3)]
    assert out.shape == (2, 8)


def test_gguf_config_parser_uses_parent_dir_for_local_file(tmp_path, monkeypatch):
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    calls = {}

    def fake_parse(
        self, model, trust_remote_code, revision=None, code_revision=None, **kwargs
    ):
        calls["model"] = model
        calls["trust_remote_code"] = trust_remote_code
        calls["gguf_file"] = kwargs.get("gguf_file")
        return {}, PretrainedConfig(model_type="qwen3_moe")

    monkeypatch.setattr(
        gguf_config_parser_module.HFConfigParser,
        "parse",
        fake_parse,
    )
    monkeypatch.setattr(
        gguf_config_parser_module,
        "maybe_patch_hf_config_from_gguf",
        lambda model, config: config,
    )

    config_dict, config = GGUFConfigParser().parse(gguf_path, trust_remote_code=False)

    assert calls["model"] == gguf_path.parent
    assert calls["trust_remote_code"] is False
    assert calls["gguf_file"] == gguf_path.name
    assert config_dict["norm_topk_prob"] is True
    assert config.architectures == ["Qwen3MoeForCausalLM"]


def test_gguf_config_parser_uses_first_split_shard_for_local_file(
    tmp_path,
    monkeypatch,
):
    first_shard = tmp_path / "model-Q4_K_M-00001-of-00002.gguf"
    second_shard = tmp_path / "model-Q4_K_M-00002-of-00002.gguf"
    first_shard.write_bytes(b"GGUF")
    second_shard.write_bytes(b"GGUF")
    calls = {}

    def fake_parse(
        self, model, trust_remote_code, revision=None, code_revision=None, **kwargs
    ):
        calls["model"] = model
        calls["gguf_file"] = kwargs.get("gguf_file")
        return {}, PretrainedConfig(model_type="qwen3")

    monkeypatch.setattr(
        gguf_config_parser_module.HFConfigParser,
        "parse",
        fake_parse,
    )
    monkeypatch.setattr(
        gguf_config_parser_module,
        "maybe_patch_hf_config_from_gguf",
        lambda model, config: config,
    )
    monkeypatch.setattr(
        gguf_utils_module,
        "_get_local_gguf_base_model_ids",
        lambda model: (),
    )

    config_dict, config = GGUFConfigParser().parse(
        second_shard,
        trust_remote_code=False,
    )

    assert calls["model"] == tmp_path
    assert calls["gguf_file"] == first_shard.name
    assert config_dict["architectures"] == ["Qwen3ForCausalLM"]
    assert config.architectures == ["Qwen3ForCausalLM"]


def test_gguf_config_parser_preserves_trust_for_snapshot_root_config(
    tmp_path,
    monkeypatch,
):
    snapshot = tmp_path / "models--org--repo" / "snapshots" / "abc123"
    model_dir = snapshot / "Q8_0" / "nested"
    model_dir.mkdir(parents=True)
    gguf_path = model_dir / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    calls = {}

    def fake_parse(
        self, model, trust_remote_code, revision=None, code_revision=None, **kwargs
    ):
        calls["model"] = model
        calls["trust_remote_code"] = trust_remote_code
        calls["gguf_file"] = kwargs.get("gguf_file")
        return {}, PretrainedConfig(model_type="qwen3")

    def fake_file_or_path_exists(model, filename, revision=None):
        return (Path(model) / filename).is_file()

    monkeypatch.setattr(
        gguf_config_parser_module.HFConfigParser,
        "parse",
        fake_parse,
    )
    monkeypatch.setattr(
        gguf_config_parser_module,
        "maybe_patch_hf_config_from_gguf",
        lambda model, config: config,
    )
    monkeypatch.setattr(
        gguf_config_parser_module,
        "file_or_path_exists",
        fake_file_or_path_exists,
    )
    monkeypatch.setattr(
        gguf_utils_module,
        "file_or_path_exists",
        fake_file_or_path_exists,
    )

    config_dict, config = GGUFConfigParser().parse(
        gguf_path,
        trust_remote_code=True,
    )

    assert calls["model"] == snapshot
    assert calls["trust_remote_code"] is True
    assert calls["gguf_file"] is None
    assert config_dict["architectures"] == ["Qwen3ForCausalLM"]
    assert config.architectures == ["Qwen3ForCausalLM"]


def test_gguf_config_parser_uses_repo_for_exact_remote_file(monkeypatch):
    calls = {}

    def fake_file_or_path_exists(model, filename, revision):
        return model == "org/repo" and filename == "config.json"

    def fake_parse(
        self, model, trust_remote_code, revision=None, code_revision=None, **kwargs
    ):
        calls["model"] = model
        calls["revision"] = revision
        calls["trust_remote_code"] = trust_remote_code
        calls["gguf_file"] = kwargs.get("gguf_file")
        return {}, PretrainedConfig(model_type="qwen3")

    monkeypatch.setattr(
        gguf_config_parser_module.HFConfigParser,
        "parse",
        fake_parse,
    )
    monkeypatch.setattr(
        gguf_config_parser_module,
        "maybe_patch_hf_config_from_gguf",
        lambda model, config: config,
    )
    monkeypatch.setattr(
        gguf_config_parser_module,
        "file_or_path_exists",
        fake_file_or_path_exists,
    )
    monkeypatch.setattr(
        gguf_utils_module,
        "file_or_path_exists",
        fake_file_or_path_exists,
    )
    monkeypatch.setattr(
        gguf_utils_module,
        "_get_remote_gguf_base_model_ids",
        lambda repo_id, revision=None: (),
    )

    config_dict, config = GGUFConfigParser().parse(
        "org/repo/subdir/model.gguf",
        trust_remote_code=True,
        revision="main",
    )

    assert calls["model"] == "org/repo"
    assert calls["revision"] == "main"
    assert calls["trust_remote_code"] is True
    assert calls["gguf_file"] is None
    assert config_dict["architectures"] == ["Qwen3ForCausalLM"]
    assert config.architectures == ["Qwen3ForCausalLM"]


def test_gguf_config_parser_preserves_multimodal_architecture(monkeypatch):
    calls = {}

    def fake_parse(
        self, model, trust_remote_code, revision=None, code_revision=None, **kwargs
    ):
        calls["model"] = model
        calls["trust_remote_code"] = trust_remote_code
        calls["gguf_file"] = kwargs.get("gguf_file")
        config = PretrainedConfig(model_type="qwen3_5")
        config.architectures = ["Qwen3_5ForConditionalGeneration"]
        return {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
        }, config

    monkeypatch.setattr(
        gguf_config_parser_module.HFConfigParser,
        "parse",
        fake_parse,
    )
    monkeypatch.setattr(
        gguf_config_parser_module,
        "resolve_gguf_config_source",
        lambda model, revision=None: "Qwen/Qwen3.5-0.8B",
    )
    monkeypatch.setattr(
        gguf_config_parser_module,
        "maybe_patch_hf_config_from_gguf",
        lambda model, config: config,
    )

    config_dict, config = GGUFConfigParser().parse(
        "unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M",
        trust_remote_code=True,
        revision="main",
    )

    assert calls["model"] == "Qwen/Qwen3.5-0.8B"
    assert calls["trust_remote_code"] is False
    assert calls["gguf_file"] is None
    assert config_dict["architectures"] == ["Qwen3_5ForConditionalGeneration"]
    assert config.architectures == ["Qwen3_5ForConditionalGeneration"]


def test_gguf_config_parser_passes_exact_remote_file_when_repo_has_no_config(
    monkeypatch,
):
    calls = {}

    def fake_parse(
        self, model, trust_remote_code, revision=None, code_revision=None, **kwargs
    ):
        calls["model"] = model
        calls["revision"] = revision
        calls["trust_remote_code"] = trust_remote_code
        calls["gguf_file"] = kwargs.get("gguf_file")
        return {}, PretrainedConfig(model_type="qwen3")

    monkeypatch.setattr(
        gguf_config_parser_module.HFConfigParser,
        "parse",
        fake_parse,
    )
    monkeypatch.setattr(
        gguf_config_parser_module,
        "maybe_patch_hf_config_from_gguf",
        lambda model, config: config,
    )
    monkeypatch.setattr(
        gguf_config_parser_module,
        "file_or_path_exists",
        lambda model, filename, revision: False,
    )
    monkeypatch.setattr(
        gguf_utils_module,
        "file_or_path_exists",
        lambda model, filename, revision: False,
    )
    monkeypatch.setattr(
        gguf_utils_module,
        "_get_remote_gguf_base_model_ids",
        lambda repo_id, revision=None: (),
    )

    config_dict, config = GGUFConfigParser().parse(
        "org/repo/subdir/model.gguf",
        trust_remote_code=True,
        revision="main",
    )

    assert calls["model"] == "org/repo"
    assert calls["revision"] == "main"
    assert calls["trust_remote_code"] is True
    assert calls["gguf_file"] == "subdir/model.gguf"
    assert config_dict["architectures"] == ["Qwen3ForCausalLM"]
    assert config.architectures == ["Qwen3ForCausalLM"]


def test_gguf_config_parser_uses_first_split_shard_for_exact_remote_file(
    monkeypatch,
):
    calls = {}

    def fake_parse(
        self, model, trust_remote_code, revision=None, code_revision=None, **kwargs
    ):
        calls["model"] = model
        calls["gguf_file"] = kwargs.get("gguf_file")
        return {}, PretrainedConfig(model_type="qwen3")

    monkeypatch.setattr(
        gguf_config_parser_module.HFConfigParser,
        "parse",
        fake_parse,
    )
    monkeypatch.setattr(
        gguf_config_parser_module,
        "maybe_patch_hf_config_from_gguf",
        lambda model, config: config,
    )
    monkeypatch.setattr(
        gguf_config_parser_module,
        "file_or_path_exists",
        lambda model, filename, revision: False,
    )
    monkeypatch.setattr(
        gguf_utils_module,
        "file_or_path_exists",
        lambda model, filename, revision: False,
    )
    monkeypatch.setattr(
        gguf_utils_module,
        "_get_remote_gguf_base_model_ids",
        lambda repo_id, revision=None: (),
    )

    config_dict, config = GGUFConfigParser().parse(
        "org/repo/subdir/model-Q4_K_M-00002-of-00002.gguf",
        trust_remote_code=True,
        revision="main",
    )

    assert calls["model"] == "org/repo"
    assert calls["gguf_file"] == "subdir/model-Q4_K_M-00001-of-00002.gguf"
    assert config_dict["architectures"] == ["Qwen3ForCausalLM"]
    assert config.architectures == ["Qwen3ForCausalLM"]


class _FakeGGUFField:
    def __init__(self, value):
        self.value = value
        self.parts = [value]

    def contents(self):
        return self.value


class _FakeGGUFReader:
    def __init__(self, fields):
        self.fields = {key: _FakeGGUFField(value) for key, value in fields.items()}
        self.tensors = []

    def get_field(self, key):
        return self.fields.get(key)


def test_copy_local_processor_sidecars_searches_hf_snapshot_root(tmp_path):
    snapshot = tmp_path / "models--org--repo" / "snapshots" / "abc123"
    model_dir = snapshot / "subdir"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "model.gguf"
    model_path.touch()
    (snapshot / "processor_config.json").write_text(
        '{"source":"snapshot"}',
        encoding="utf-8",
    )

    cache_dir = tmp_path / "cache"

    gguf_tokenizer_builder_module._copy_local_processor_sidecars(
        model_path,
        cache_dir,
    )

    assert json.loads(
        (cache_dir / "processor_config.json").read_text(encoding="utf-8")
    ) == {"source": "snapshot"}


def test_copy_local_processor_sidecars_searches_intermediate_ancestor(tmp_path):
    snapshot = tmp_path / "models--org--repo" / "snapshots" / "abc123"
    config_dir = snapshot / "Q8_0"
    model_dir = config_dir / "nested" / "deep"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "model.gguf"
    model_path.touch()
    (config_dir / "processor_config.json").write_text(
        '{"source":"quant-dir"}',
        encoding="utf-8",
    )

    cache_dir = tmp_path / "cache"

    gguf_tokenizer_builder_module._copy_local_processor_sidecars(
        model_path,
        cache_dir,
    )

    assert json.loads(
        (cache_dir / "processor_config.json").read_text(encoding="utf-8")
    ) == {"source": "quant-dir"}


def test_copy_local_processor_sidecars_prefers_nearest_and_preserves_cache(tmp_path):
    model_dir = tmp_path / "nested"
    model_dir.mkdir()
    model_path = model_dir / "model.gguf"
    model_path.touch()
    (tmp_path / "processor_config.json").write_text(
        '{"source":"parent"}',
        encoding="utf-8",
    )
    (model_dir / "processor_config.json").write_text(
        '{"source":"model-dir"}',
        encoding="utf-8",
    )
    (model_dir / "preprocessor_config.json").write_text(
        '{"source":"model-dir"}',
        encoding="utf-8",
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "preprocessor_config.json").write_text(
        '{"source":"cache"}',
        encoding="utf-8",
    )

    gguf_tokenizer_builder_module._copy_local_processor_sidecars(
        model_path,
        cache_dir,
    )

    assert json.loads(
        (cache_dir / "processor_config.json").read_text(encoding="utf-8")
    ) == {"source": "model-dir"}
    assert json.loads(
        (cache_dir / "preprocessor_config.json").read_text(encoding="utf-8")
    ) == {"source": "cache"}


def test_local_config_special_tokens_search_hf_snapshot_root(tmp_path):
    snapshot = tmp_path / "models--org--repo" / "snapshots" / "abc123"
    model_dir = snapshot / "Q8_0" / "nested"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "model.gguf"
    model_path.touch()
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "eos_token_id": 4,
                "text_config": {
                    "bos_token_id": 1,
                    "pad_token_id": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    tokenizer_dict = {
        "tokens": [
            "<pad>",
            "<bos>",
            "<eos>",
            "hello",
            "<turn|>",
        ],
    }

    assert gguf_tokenizer_builder_module._local_config_special_token_kwargs(
        model_path,
        tokenizer_dict,
    ) == {
        "bos_token": "<bos>",
        "eos_token": "<turn|>",
        "pad_token": "<pad>",
    }


def test_build_tokenizer_from_gguf_metadata_uses_arch_alias_and_cache(
    tmp_path,
    monkeypatch,
):
    gguf_path = tmp_path / "model.gguf"
    mmproj_path = tmp_path / "mmproj.gguf"
    gguf_path.write_bytes(b"GGUF")
    mmproj_path.write_bytes(b"GGUF")
    qwen_mm_tokens = [
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]
    qwen_control_tokens = ["<tool_call>", "</tool_call>", "<think>", "</think>"]
    main_reader = _FakeGGUFReader(
        {
            "general.architecture": "qwen35moe",
            "tokenizer.ggml.tokens": [
                "<pad>",
                "<bos>",
                "<eos>",
                "hello",
                *qwen_mm_tokens,
                *qwen_control_tokens,
                "[PAD000]",
            ],
            "tokenizer.ggml.token_type": [
                3,
                3,
                3,
                1,
                *([3] * len(qwen_mm_tokens)),
                *([4] * len(qwen_control_tokens)),
                5,
            ],
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.merges": ["h ello"],
            "tokenizer.ggml.bos_token_id": 1,
            "tokenizer.ggml.eos_token_id": 2,
            "tokenizer.ggml.padding_token_id": 0,
            "tokenizer.chat_template": "{{ messages }}",
        }
    )
    mmproj_reader = _FakeGGUFReader(
        {
            "general.architecture": "clip",
            "general.type": "mmproj",
            "clip.vision.patch_size": 16,
            "clip.vision.spatial_merge_size": 2,
            "clip.vision.temporal_patch_size": 2,
        }
    )
    calls = []

    class FakeTokenizer:
        def __init__(self, *args, **kwargs):
            calls.append(("fast", kwargs))
            self.chat_template = None

        def save_pretrained(self, path):
            (path / "tokenizer.json").write_text("{}", encoding="utf-8")
            (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    def fake_convert(architecture, tokenizer_dict):
        calls.append(("convert", architecture, tokenizer_dict))
        return object(), {}

    def fake_gguf_reader(path):
        return mmproj_reader if Path(path) == mmproj_path else main_reader

    monkeypatch.setenv(
        "VLLM_GGUF_TOKENIZER_CACHE",
        str(tmp_path / "tokenizer-cache"),
    )
    monkeypatch.setattr(
        gguf_tokenizer_builder_module.gguf,
        "GGUFReader",
        fake_gguf_reader,
    )
    monkeypatch.setattr(
        gguf_tokenizer_builder_module,
        "convert_gguf_tokenizer",
        fake_convert,
    )
    monkeypatch.setattr(
        gguf_tokenizer_builder_module,
        "PreTrainedTokenizerFast",
        FakeTokenizer,
    )
    monkeypatch.setattr(
        gguf_tokenizer_builder_module,
        "detect_gguf_multimodal",
        lambda model: mmproj_path,
    )

    tokenizer_path = build_tokenizer_from_gguf(gguf_path)

    assert tokenizer_path is not None
    tokenizer_cache = Path(tokenizer_path)
    processor_config = json.loads(
        (tokenizer_cache / "processor_config.json").read_text(encoding="utf-8")
    )
    preprocessor_config = json.loads(
        (tokenizer_cache / "preprocessor_config.json").read_text(encoding="utf-8")
    )
    video_config = json.loads(
        (tokenizer_cache / "video_preprocessor_config.json").read_text(encoding="utf-8")
    )
    assert processor_config["processor_class"] == "Qwen3VLProcessor"
    assert preprocessor_config["image_processor_type"] == "Qwen2VLImageProcessor"
    assert preprocessor_config["merge_size"] == 2
    assert video_config["video_processor_type"] == "Qwen3VLVideoProcessor"

    tokenizer_config = json.loads(
        (tokenizer_cache / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    assert tokenizer_config["additional_special_tokens"] == [
        *qwen_mm_tokens,
        *qwen_control_tokens,
    ]
    assert tokenizer_config["image_token"] == "<|image_pad|>"
    assert tokenizer_config["video_token"] == "<|video_pad|>"
    assert calls[0][0] == "convert"
    assert calls[0][1] == "qwen3_moe"
    assert calls[1][1]["bos_token"] == "<bos>"
    assert calls[1][1]["eos_token"] == "<eos>"
    assert calls[1][1]["pad_token"] == "<pad>"

    def fail_convert(*args, **kwargs):
        raise AssertionError("cached tokenizer should not call converter again")

    monkeypatch.setattr(
        gguf_tokenizer_builder_module,
        "convert_gguf_tokenizer",
        fail_convert,
    )
    (tokenizer_cache / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tokenizer_cache / "preprocessor_config.json").unlink()
    assert build_tokenizer_from_gguf(gguf_path) == tokenizer_path
    tokenizer_config = json.loads(
        (tokenizer_cache / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    assert tokenizer_config["bos_token"] == "<bos>"
    assert tokenizer_config["eos_token"] == "<eos>"
    assert tokenizer_config["pad_token"] == "<pad>"
    assert (tokenizer_cache / "preprocessor_config.json").is_file()


def test_build_tokenizer_from_gguf_uses_first_split_shard(tmp_path, monkeypatch):
    first_shard = tmp_path / "model-Q4_K_M-00001-of-00002.gguf"
    second_shard = tmp_path / "model-Q4_K_M-00002-of-00002.gguf"
    first_shard.write_bytes(b"GGUF")
    second_shard.write_bytes(b"GGUF")
    fake_reader = _FakeGGUFReader(
        {
            "general.architecture": "qwen3",
            "tokenizer.ggml.tokens": [
                "<pad>",
                "<bos>",
                "<eos>",
                "hello",
            ],
            "tokenizer.ggml.token_type": [3, 3, 3, 1],
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.merges": ["h ello"],
            "tokenizer.ggml.bos_token_id": 1,
            "tokenizer.ggml.eos_token_id": 2,
            "tokenizer.ggml.padding_token_id": 0,
        }
    )
    reader_paths = []

    class FakeTokenizer:
        def __init__(self, *args, **kwargs):
            pass

        def save_pretrained(self, path):
            (path / "tokenizer.json").write_text("{}", encoding="utf-8")
            (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    def fake_gguf_reader(path):
        reader_paths.append(Path(path))
        return fake_reader

    monkeypatch.setenv(
        "VLLM_GGUF_TOKENIZER_CACHE",
        str(tmp_path / "tokenizer-cache"),
    )
    monkeypatch.setattr(
        gguf_tokenizer_builder_module.gguf,
        "GGUFReader",
        fake_gguf_reader,
    )
    monkeypatch.setattr(
        gguf_tokenizer_builder_module,
        "convert_gguf_tokenizer",
        lambda architecture, tokenizer_dict: (object(), {}),
    )
    monkeypatch.setattr(
        gguf_tokenizer_builder_module,
        "PreTrainedTokenizerFast",
        FakeTokenizer,
    )
    monkeypatch.setattr(
        gguf_tokenizer_builder_module,
        "detect_gguf_multimodal",
        lambda model: None,
    )

    tokenizer_path = build_tokenizer_from_gguf(second_shard)

    assert tokenizer_path is not None
    assert reader_paths[0] == first_shard


def test_build_tokenizer_from_gguf_prefers_local_config_special_token_ids(
    tmp_path,
    monkeypatch,
):
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "eos_token_id": 4,
                "pad_token_id": 0,
                "text_config": {
                    "bos_token_id": 1,
                    "eos_token_id": 2,
                    "pad_token_id": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    fake_reader = _FakeGGUFReader(
        {
            "general.architecture": "gemma4",
            "tokenizer.ggml.tokens": [
                "<pad>",
                "<bos>",
                "<eos>",
                "hello",
                "<turn|>",
            ],
            "tokenizer.ggml.token_type": [3, 3, 3, 1, 3],
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.merges": ["h ello"],
            "tokenizer.ggml.bos_token_id": 1,
            "tokenizer.ggml.eos_token_id": 2,
            "tokenizer.ggml.padding_token_id": 0,
        }
    )
    calls = []

    class FakeTokenizer:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)

        def save_pretrained(self, path):
            (path / "tokenizer.json").write_text("{}", encoding="utf-8")
            (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setenv(
        "VLLM_GGUF_TOKENIZER_CACHE",
        str(tmp_path / "tokenizer-cache"),
    )
    monkeypatch.setattr(
        gguf_tokenizer_builder_module.gguf,
        "GGUFReader",
        lambda path: fake_reader,
    )
    monkeypatch.setattr(
        gguf_tokenizer_builder_module,
        "convert_gguf_tokenizer",
        lambda architecture, tokenizer_dict: (object(), {}),
    )
    monkeypatch.setattr(
        gguf_tokenizer_builder_module,
        "PreTrainedTokenizerFast",
        FakeTokenizer,
    )
    monkeypatch.setattr(
        gguf_tokenizer_builder_module,
        "detect_gguf_multimodal",
        lambda model: None,
    )

    tokenizer_path = build_tokenizer_from_gguf(gguf_path)

    assert tokenizer_path is not None
    assert calls[0]["bos_token"] == "<bos>"
    assert calls[0]["eos_token"] == "<turn|>"
    assert calls[0]["pad_token"] == "<pad>"
    tokenizer_config_path = Path(tokenizer_path) / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    assert tokenizer_config["eos_token"] == "<turn|>"
    assert "<turn|>" not in tokenizer_config.get("additional_special_tokens", [])


def test_build_tokenizer_from_gguf_returns_none_when_cache_key_stat_fails(
    tmp_path,
    monkeypatch,
):
    gguf_path = tmp_path / "missing.gguf"

    monkeypatch.setattr(
        gguf_tokenizer_builder_module,
        "check_gguf_file",
        lambda model: True,
    )

    def fail_reader(*args, **kwargs):
        raise AssertionError("stat failure must return before reading GGUF")

    monkeypatch.setattr(
        gguf_tokenizer_builder_module.gguf,
        "GGUFReader",
        fail_reader,
    )

    assert build_tokenizer_from_gguf(gguf_path) is None


def test_register_sets_engine_args_for_gguf_model(monkeypatch):
    register()
    captured = {}

    def fake_model_config(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(arg_utils_module, "ModelConfig", fake_model_config)
    engine_args = EngineArgs(model="/tmp/model.gguf", tokenizer="/tmp/tokenizer")

    engine_args.create_model_config()

    assert captured["config_format"] == "gguf"
    assert captured["model"] == "/tmp/tokenizer"
    assert captured["model_weights"] == "/tmp/model.gguf"
    assert captured["quantization"] == "gguf"
    assert engine_args.load_format == "gguf"


def test_register_defaults_auto_dtype_to_float16_for_blackwell_gguf(monkeypatch):
    register()
    captured = {}

    fake_platform = type(
        "FakePlatform",
        (),
        {"has_device_capability": staticmethod(lambda capability: capability == 100)},
    )
    monkeypatch.setattr(gguf_plugin_module, "current_platform", fake_platform)

    def fake_model_config(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(arg_utils_module, "ModelConfig", fake_model_config)
    engine_args = EngineArgs(model="/tmp/model.gguf")

    engine_args.create_model_config()

    assert captured["dtype"] == "float16"
    assert engine_args.dtype == "float16"


def test_register_preserves_explicit_dtype_for_blackwell_gguf(monkeypatch):
    register()
    captured = {}

    fake_platform = type(
        "FakePlatform",
        (),
        {"has_device_capability": staticmethod(lambda capability: capability == 100)},
    )
    monkeypatch.setattr(gguf_plugin_module, "current_platform", fake_platform)

    def fake_model_config(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(arg_utils_module, "ModelConfig", fake_model_config)
    engine_args = EngineArgs(model="/tmp/model.gguf", dtype="float32")

    engine_args.create_model_config()

    assert captured["dtype"] == "float32"
    assert engine_args.dtype == "float32"


def test_register_routes_remote_gguf_base_config_source(monkeypatch):
    register()
    captured = {}

    monkeypatch.setattr(
        gguf_plugin_module,
        "resolve_gguf_config_source",
        lambda model, revision=None: "base/repo",
    )

    def fake_model_config(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(arg_utils_module, "ModelConfig", fake_model_config)
    engine_args = EngineArgs(
        model="org/repo:Q4_K_M",
        trust_remote_code=True,
        revision="gguf-revision",
    )

    engine_args.create_model_config()

    assert captured["model"] == "base/repo"
    assert captured["model_weights"] == "org/repo:Q4_K_M"
    assert captured["served_model_name"] == ["org/repo:Q4_K_M"]
    assert captured["trust_remote_code"] is False


def test_register_sets_embedded_tokenizer_for_local_gguf(tmp_path, monkeypatch):
    register()
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    captured = {}

    monkeypatch.setattr(
        gguf_plugin_module,
        "build_tokenizer_from_gguf",
        lambda model: "/tmp/gguf-tokenizer-cache",
    )

    def fake_model_config(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(arg_utils_module, "ModelConfig", fake_model_config)
    engine_args = EngineArgs(model=str(gguf_path))

    engine_args.create_model_config()

    assert captured["model"] == str(gguf_path)
    assert captured["model_weights"] == str(gguf_path)
    assert captured["tokenizer"] == "/tmp/gguf-tokenizer-cache"


def test_register_sets_embedded_tokenizer_for_remote_gguf_quant(monkeypatch):
    register()
    captured = {}
    download_calls = []

    monkeypatch.setattr(
        gguf_plugin_module,
        "resolve_gguf_config_source",
        lambda model, revision=None: "org/repo",
    )
    monkeypatch.setattr(
        gguf_plugin_module,
        "file_or_path_exists",
        lambda model, filename, revision=None: False,
    )

    def fake_download_gguf(
        repo_id,
        quant_type,
        cache_dir=None,
        revision=None,
        ignore_patterns=None,
    ):
        download_calls.append(
            (repo_id, quant_type, cache_dir, revision, ignore_patterns)
        )
        return "/cache/org/repo/model-Q4_K_M.gguf"

    monkeypatch.setattr(gguf_plugin_module, "download_gguf", fake_download_gguf)
    monkeypatch.setattr(
        gguf_plugin_module,
        "build_tokenizer_from_gguf",
        lambda model: "/tmp/gguf-tokenizer-cache",
    )

    def fake_model_config(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(arg_utils_module, "ModelConfig", fake_model_config)
    engine_args = EngineArgs(
        model="org/repo:Q4_K_M",
        download_dir="/cache",
        revision="abc123",
    )
    engine_args.ignore_patterns = ["*.safetensors"]

    engine_args.create_model_config()

    assert captured["model_weights"] == "org/repo:Q4_K_M"
    assert captured["tokenizer"] == "/tmp/gguf-tokenizer-cache"
    assert download_calls == [
        ("org/repo", "Q4_K_M", "/cache", "abc123", ["*.safetensors"])
    ]


def test_register_sets_embedded_tokenizer_for_exact_remote_gguf(monkeypatch):
    register()
    captured = {}
    download_calls = []

    monkeypatch.setattr(
        gguf_plugin_module,
        "resolve_gguf_config_source",
        lambda model, revision=None: "org/repo",
    )
    monkeypatch.setattr(
        gguf_plugin_module,
        "file_or_path_exists",
        lambda model, filename, revision=None: False,
    )

    def fake_download_gguf_file(
        repo_id,
        filename,
        cache_dir=None,
        revision=None,
    ):
        download_calls.append((repo_id, filename, cache_dir, revision))
        return "/cache/org/repo/Q8_0/model.gguf"

    monkeypatch.setattr(
        gguf_plugin_module,
        "download_gguf_file",
        fake_download_gguf_file,
    )
    monkeypatch.setattr(
        gguf_plugin_module,
        "build_tokenizer_from_gguf",
        lambda model: "/tmp/gguf-tokenizer-cache",
    )

    def fake_model_config(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(arg_utils_module, "ModelConfig", fake_model_config)
    engine_args = EngineArgs(
        model="org/repo/Q8_0/model.gguf",
        download_dir="/cache",
        revision="abc123",
    )

    engine_args.create_model_config()

    assert captured["model_weights"] == "org/repo/Q8_0/model.gguf"
    assert captured["tokenizer"] == "/tmp/gguf-tokenizer-cache"
    assert download_calls == [("org/repo", "Q8_0/model.gguf", "/cache", "abc123")]


def test_register_skips_remote_embedded_tokenizer_when_source_has_tokenizer(
    monkeypatch,
):
    register()
    captured = {}

    monkeypatch.setattr(
        gguf_plugin_module,
        "resolve_gguf_config_source",
        lambda model, revision=None: "base/repo",
    )
    monkeypatch.setattr(
        gguf_plugin_module,
        "file_or_path_exists",
        lambda model, filename, revision=None: filename == "tokenizer.json",
    )

    def fail_download(*args, **kwargs):
        raise AssertionError("remote GGUF should not download when tokenizer exists")

    monkeypatch.setattr(gguf_plugin_module, "download_gguf", fail_download)
    monkeypatch.setattr(gguf_plugin_module, "download_gguf_file", fail_download)
    monkeypatch.setattr(gguf_plugin_module, "build_tokenizer_from_gguf", fail_download)

    def fake_model_config(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(arg_utils_module, "ModelConfig", fake_model_config)
    engine_args = EngineArgs(model="org/repo:Q4_K_M")

    engine_args.create_model_config()

    assert captured["model"] == "base/repo"
    assert captured["model_weights"] == "org/repo:Q4_K_M"
    assert captured["tokenizer"] is None


def test_register_preserves_explicit_tokenizer_for_local_gguf(tmp_path, monkeypatch):
    register()
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    captured = {}

    def fail_build_tokenizer(model):
        raise AssertionError("explicit tokenizer must not be replaced")

    monkeypatch.setattr(
        gguf_plugin_module,
        "build_tokenizer_from_gguf",
        fail_build_tokenizer,
    )

    def fake_model_config(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(arg_utils_module, "ModelConfig", fake_model_config)
    engine_args = EngineArgs(model=str(gguf_path), tokenizer="/tmp/tokenizer")

    engine_args.create_model_config()

    assert captured["model"] == "/tmp/tokenizer"
    assert captured["model_weights"] == str(gguf_path)
    assert captured["tokenizer"] == "/tmp/tokenizer"


def test_register_preserves_implicit_gguf_model_for_config_parser(monkeypatch):
    register()
    captured = {}

    def fake_model_config(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(
        gguf_plugin_module,
        "resolve_gguf_config_source",
        lambda model, revision=None: model,
    )
    monkeypatch.setattr(arg_utils_module, "ModelConfig", fake_model_config)
    engine_args = EngineArgs(model="org/repo/subdir/model.gguf")

    engine_args.create_model_config()

    assert captured["config_format"] == "gguf"
    assert captured["model"] == "org/repo/subdir/model.gguf"
    assert captured["model_weights"] == "org/repo/subdir/model.gguf"
    assert captured["quantization"] == "gguf"
    assert engine_args.load_format == "gguf"


def test_register_skips_speculator_probe_for_gguf(monkeypatch):
    register()
    calls = {}

    def fake_get_config_dict(config_source, **kwargs):
        calls["config_source"] = config_source
        calls["gguf_file"] = kwargs.get("gguf_file")
        return {"model_type": "qwen3"}, {}

    monkeypatch.setattr(
        gguf_plugin_module.PretrainedConfig,
        "get_config_dict",
        fake_get_config_dict,
    )
    monkeypatch.setattr(
        gguf_plugin_module,
        "resolve_gguf_config_source",
        lambda model, revision=None: "org/repo",
    )
    monkeypatch.setattr(
        gguf_plugin_module,
        "file_or_path_exists",
        lambda model, filename, revision=None: False,
    )

    model, tokenizer, speculative_config = (
        config_module.maybe_override_with_speculators(
            model="org/repo/subdir/model.gguf",
            tokenizer="/tmp/tokenizer",
            trust_remote_code=False,
            revision=None,
            vllm_speculative_config={"foo": "bar"},
            hf_token=None,
        )
    )

    assert calls["config_source"] == "org/repo"
    assert calls["gguf_file"] == "subdir/model.gguf"
    assert model == "org/repo/subdir/model.gguf"
    assert tokenizer == "/tmp/tokenizer"
    assert speculative_config == {"foo": "bar"}


def test_register_speculator_probe_uses_first_split_shard_for_exact_remote(
    monkeypatch,
):
    register()
    calls = {}

    def fake_get_config_dict(config_source, **kwargs):
        calls["config_source"] = config_source
        calls["gguf_file"] = kwargs.get("gguf_file")
        return {"model_type": "qwen3"}, {}

    monkeypatch.setattr(
        gguf_plugin_module.PretrainedConfig,
        "get_config_dict",
        fake_get_config_dict,
    )
    monkeypatch.setattr(
        gguf_plugin_module,
        "resolve_gguf_config_source",
        lambda model, revision=None: "org/repo",
    )
    monkeypatch.setattr(
        gguf_plugin_module,
        "file_or_path_exists",
        lambda model, filename, revision=None: False,
    )

    model, tokenizer, speculative_config = (
        config_module.maybe_override_with_speculators(
            model="org/repo/subdir/model-Q4_K_M-00002-of-00002.gguf",
            tokenizer="/tmp/tokenizer",
            trust_remote_code=False,
            revision=None,
            vllm_speculative_config=None,
            hf_token=None,
        )
    )

    assert calls["config_source"] == "org/repo"
    assert calls["gguf_file"] == "subdir/model-Q4_K_M-00001-of-00002.gguf"
    assert model == "org/repo/subdir/model-Q4_K_M-00002-of-00002.gguf"
    assert tokenizer == "/tmp/tokenizer"
    assert speculative_config is None


def test_register_speculator_probe_prefers_sidecar_config(tmp_path, monkeypatch):
    register()
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    calls = {}

    def fake_get_config_dict(config_source, **kwargs):
        calls["config_source"] = config_source
        calls["gguf_file"] = kwargs.get("gguf_file")
        return {"model_type": "qwen3"}, {}

    monkeypatch.setattr(
        gguf_plugin_module.PretrainedConfig,
        "get_config_dict",
        fake_get_config_dict,
    )

    model, tokenizer, speculative_config = (
        config_module.maybe_override_with_speculators(
            model=str(gguf_path),
            tokenizer="/tmp/tokenizer",
            trust_remote_code=False,
            revision=None,
            vllm_speculative_config=None,
            hf_token=None,
        )
    )

    assert calls["config_source"] == gguf_path.parent
    assert calls["gguf_file"] is None
    assert model == str(gguf_path)
    assert tokenizer == "/tmp/tokenizer"
    assert speculative_config is None


def test_register_speculator_probe_extracts_config(tmp_path, monkeypatch):
    register()
    gguf_path = tmp_path / "draft.gguf"
    gguf_path.write_bytes(b"GGUF")
    captured = {}

    def fake_get_config_dict(config_source, **kwargs):
        captured["config_source"] = config_source
        captured["gguf_file"] = kwargs.get("gguf_file")
        return {
            "model_type": "qwen3",
            "speculators_config": {
                "verifier": {"name_or_path": "verifier/repo"},
            },
        }, {}

    class FakeSpeculatorsConfig:
        @staticmethod
        def extract_vllm_speculative_config(config_dict):
            captured["speculators_config"] = config_dict["speculators_config"]
            return {"draft": "config"}

    monkeypatch.setattr(
        gguf_plugin_module.PretrainedConfig,
        "get_config_dict",
        fake_get_config_dict,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "vllm.transformers_utils.configs.speculators.base",
        type(
            "FakeSpeculatorModule",
            (),
            {"SpeculatorsConfig": FakeSpeculatorsConfig},
        ),
    )

    model, tokenizer, speculative_config = (
        config_module.maybe_override_with_speculators(
            model=str(gguf_path),
            tokenizer="/tmp/tokenizer",
            trust_remote_code=False,
            revision=None,
            vllm_speculative_config=None,
            hf_token=None,
        )
    )

    assert captured["config_source"] == gguf_path.parent
    assert captured["gguf_file"] == gguf_path.name
    assert model == "verifier/repo"
    assert tokenizer == "verifier/repo"
    assert speculative_config == {
        "draft": "config",
        "model": str(gguf_path),
    }


def test_register_disables_trust_for_gguf_config_redirect(tmp_path, monkeypatch):
    register()
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    captured = {}

    monkeypatch.setattr(
        gguf_plugin_module,
        "resolve_gguf_config_source",
        lambda model, revision=None: "base/repo",
    )

    def fake_model_config(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(arg_utils_module, "ModelConfig", fake_model_config)
    engine_args = EngineArgs(
        model=str(gguf_path),
        tokenizer="/tmp/tokenizer",
        trust_remote_code=True,
        revision="gguf-revision",
    )
    engine_args.create_model_config()

    assert captured["trust_remote_code"] is False


def test_register_keeps_trust_for_local_snapshot_root_config(tmp_path, monkeypatch):
    snapshot = tmp_path / "models--org--repo" / "snapshots" / "abc123"
    model_dir = snapshot / "Q8_0" / "nested"
    model_dir.mkdir(parents=True)
    gguf_path = model_dir / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    captured = {}

    def fake_file_or_path_exists(model, filename, revision=None):
        return (Path(model) / filename).is_file()

    def fake_model_config(**kwargs):
        captured.update(kwargs)
        return kwargs

    register()
    monkeypatch.setattr(
        gguf_plugin_module,
        "file_or_path_exists",
        fake_file_or_path_exists,
    )
    monkeypatch.setattr(
        gguf_utils_module,
        "file_or_path_exists",
        fake_file_or_path_exists,
    )
    monkeypatch.setattr(
        gguf_plugin_module,
        "build_tokenizer_from_gguf",
        lambda model: None,
    )
    monkeypatch.setattr(arg_utils_module, "ModelConfig", fake_model_config)
    engine_args = EngineArgs(
        model=str(gguf_path),
        tokenizer=None,
        trust_remote_code=True,
        revision="gguf-revision",
    )
    engine_args.create_model_config()

    assert captured["model"] == str(snapshot)
    assert captured["trust_remote_code"] is True


def test_register_keeps_trust_for_explicit_gguf_config_path(tmp_path, monkeypatch):
    register()
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    config_path = tmp_path / "config-repo"
    config_path.mkdir()
    captured = {}

    monkeypatch.setattr(
        gguf_plugin_module,
        "resolve_gguf_config_source",
        lambda model, revision=None: "base/repo",
    )

    def fake_model_config(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(arg_utils_module, "ModelConfig", fake_model_config)
    engine_args = EngineArgs(
        model=str(gguf_path),
        tokenizer="/tmp/tokenizer",
        trust_remote_code=True,
        hf_config_path=str(config_path),
    )
    engine_args.create_model_config()

    assert captured["trust_remote_code"] is True


def test_register_disables_trust_for_gguf_speculator_config(tmp_path, monkeypatch):
    register()
    gguf_path = tmp_path / "draft.gguf"
    gguf_path.write_bytes(b"GGUF")
    captured = {}

    monkeypatch.setattr(
        gguf_plugin_module,
        "resolve_gguf_config_source",
        lambda model, revision=None: "base/repo",
    )

    def fake_model_config(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(arg_utils_module, "ModelConfig", fake_model_config)
    engine_args = EngineArgs(
        model="verifier/repo",
        tokenizer="verifier/repo",
        trust_remote_code=True,
        speculative_config={"model": str(gguf_path)},
    )
    engine_args.create_model_config()

    assert captured["trust_remote_code"] is False


def test_register_patches_model_config_gguf_helper():
    register()

    assert (
        model_config_module.maybe_patch_hf_config_from_gguf
        is gguf_plugin_module.maybe_patch_hf_config_from_gguf
    )


def test_gguf_qkv_shards_are_padded_in_qkv_order(monkeypatch):
    register()
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )

    layer = QKVParallelLinear(
        hidden_size=4,
        head_size=2,
        total_num_heads=2,
        total_num_kv_heads=1,
        bias=False,
        quant_config=OOTGGUFConfig.from_config({}),
        disable_tp=True,
    )

    q = torch.full((4, 4), 1, dtype=torch.uint8)
    k = torch.full((2, 4), 2, dtype=torch.uint8)
    v = torch.full((2, 4), 3, dtype=torch.uint8)
    # Load out of canonical order to match GGUF tensor iteration order.
    layer.weight_loader_v2(layer.qweight, k, "k")
    layer.weight_loader_v2(layer.qweight, q, "q")
    layer.weight_loader_v2(layer.qweight, v, "v")
    layer.weight_loader_v2(layer.qweight_type, torch.tensor(3, dtype=torch.uint8), "k")
    layer.weight_loader_v2(layer.qweight_type, torch.tensor(3, dtype=torch.uint8), "q")
    layer.weight_loader_v2(layer.qweight_type, torch.tensor(3, dtype=torch.uint8), "v")

    layer.quant_method.process_weights_after_loading(layer)

    assert layer.qweight.shard_id == ["q", "k", "v"]
    assert layer.qweight.shard_offset_map == {
        "q": (0, 4, 4),
        "k": (4, 6, 4),
        "v": (6, 8, 4),
    }
    assert torch.equal(layer.qweight[:4], q)
    assert torch.equal(layer.qweight[4:6], k)
    assert torch.equal(layer.qweight[6:8], v)


def test_gguf_linear_preserves_cuda_weight_device(monkeypatch):
    if not torch.cuda.is_available():
        return

    register()
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )

    with torch.device("cuda"):
        layer = MergedColumnParallelLinear(
            input_size=4,
            output_sizes=[4, 4],
            bias=False,
            quant_config=OOTGGUFConfig.from_config({}),
            disable_tp=True,
        )

    layer.weight_loader_v2(layer.qweight, torch.ones((4, 4), dtype=torch.uint8), 0)
    layer.weight_loader_v2(layer.qweight, 2 * torch.ones((4, 4), dtype=torch.uint8), 1)
    layer.weight_loader_v2(layer.qweight_type, torch.tensor(3, dtype=torch.uint8), 0)
    layer.weight_loader_v2(layer.qweight_type, torch.tensor(3, dtype=torch.uint8), 1)
    layer.quant_method.process_weights_after_loading(layer)

    assert layer.qweight.device.type == "cuda"
    assert layer.qweight_type.device.type == "cuda"
