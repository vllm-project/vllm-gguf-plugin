# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import vllm.config.model as model_config_module
import vllm.engine.arg_utils as arg_utils_module
import vllm.model_executor.layers.quantization as quantization_module
import vllm.model_executor.layers.vocab_parallel_embedding as vocab_embedding_module
import vllm.model_executor.parameter as parameter_module
import vllm.transformers_utils.config as config_module
from gguf import GGMLQuantizationType as WeightType
from gguf.constants import Keys, VisionProjectorType
from transformers import PretrainedConfig
from vllm.config.load import LoadConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.linear import (
    WEIGHT_LOADER_V2_SUPPORTED,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader import get_model_loader
from vllm.transformers_utils.config import get_config_parser

import vllm_gguf_plugin.config_parser as gguf_config_parser_module
import vllm_gguf_plugin.gguf_tokenizer_builder as gguf_tokenizer_builder_module
import vllm_gguf_plugin.gguf_utils as gguf_utils_module
import vllm_gguf_plugin.plugin as gguf_plugin_module
import vllm_gguf_plugin.quantization as gguf_quantization
import vllm_gguf_plugin.quantization.params as gguf_params_module
import vllm_gguf_plugin.weights_adapter.default as default_adapter_module
from vllm_gguf_plugin import OOTGGUFConfig, OOTGGUFModelLoader, register
from vllm_gguf_plugin.config_parser import GGUFConfigParser
from vllm_gguf_plugin.gguf_tokenizer_builder import build_tokenizer_from_gguf
from vllm_gguf_plugin.gguf_utils import (
    _gguf_sequence_edge,
    extract_vision_config_from_gguf,
    maybe_patch_hf_config_from_gguf,
    resolve_gguf_config_source,
)
from vllm_gguf_plugin.quantization import (
    GGUFUninitializedParameter,
    GGUFWeightParameter,
    GGUFWeightTypeParameter,
)
from vllm_gguf_plugin.weights_adapter.default import (
    GGUFWeightsAdapter,
    _add_gemma4_gguf_mappings,
    _add_gemma4_mtp_gguf_mappings,
    _add_qwen3_5_mtp_gguf_mappings,
)
from vllm_gguf_plugin.weights_adapter.gemma4 import Gemma4GGUFAdapter
from vllm_gguf_plugin.weights_adapter.qwen3_5 import Qwen3_5GGUFAdapter


def test_register_overrides_gguf_config():
    register()

    quant_config = quantization_module.get_quantization_config("gguf")

    assert quant_config is OOTGGUFConfig


def test_register_overrides_gguf_loader():
    register()

    model_loader = get_model_loader(LoadConfig(load_format="gguf"))

    assert isinstance(model_loader, OOTGGUFModelLoader)


def test_register_is_idempotent():
    register()
    register()

    assert quantization_module.get_quantization_config("gguf") is OOTGGUFConfig
    assert isinstance(
        get_model_loader(LoadConfig(load_format="gguf")), OOTGGUFModelLoader
    )
    assert isinstance(get_config_parser("gguf"), GGUFConfigParser)


@pytest.mark.parametrize(
    "script",
    [
        """
import torch
import vllm_gguf_plugin
import vllm.model_executor.layers.quantization.gguf
assert hasattr(torch.ops.vllm_gguf_plugin, "_fused_mul_mat_gguf")
assert hasattr(torch.ops.vllm_gguf_plugin, "_fused_moe_gguf")
assert hasattr(torch.ops.vllm_gguf_plugin, "_apply_gguf_embedding")
""",
        """
import torch
import vllm.model_executor.layers.quantization.gguf
import vllm_gguf_plugin
assert hasattr(torch.ops.vllm_gguf_plugin, "_fused_mul_mat_gguf")
assert hasattr(torch.ops.vllm_gguf_plugin, "_fused_moe_gguf")
assert hasattr(torch.ops.vllm_gguf_plugin, "_apply_gguf_embedding")
""",
    ],
)
def test_plugin_custom_ops_do_not_conflict_with_core_gguf_import(script):
    subprocess.run([sys.executable, "-c", script], check=True)


def test_register_patches_model_config_gguf_helper():
    register()

    assert (
        model_config_module.maybe_patch_hf_config_from_gguf
        is gguf_utils_module.maybe_patch_hf_config_from_gguf
    )


def test_oot_config_reuses_in_tree_behavior():
    quant_config = OOTGGUFConfig.from_config({})

    assert isinstance(quant_config, OOTGGUFConfig)
    assert quant_config.get_name() == "gguf"
    assert repr(quant_config) == "GGUFConfig()"


def test_gguf_override_quantization_method_accepts_hf_config_keyword():
    register()

    # hf_config keyword matches core QuantizationConfig.override_quantization_method
    # signature.
    # This was added to fix a TypeError when ModelConfig._verify_quantization() calls
    # override_quantization_method(hf_quant_cfg, user_quant, hf_config=hf_config).
    result_explicit = OOTGGUFConfig.override_quantization_method(
        {}, "gguf", hf_config=object()
    )
    assert result_explicit == "gguf"

    result_non_gguf = OOTGGUFConfig.override_quantization_method(
        {}, "awq", hf_config=object()
    )
    assert result_non_gguf is None

    result_none = OOTGGUFConfig.override_quantization_method({}, None)
    assert result_none is None


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


def test_gguf_linear_keeps_multi_shards_separate(monkeypatch):
    register()
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )

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

    assert layer.qweight.numel() == 0
    assert layer.qweight.shard_id == [0, 1]
    assert layer.qweight.shard_id_map == {0: 0, 1: 1}
    assert layer.qweight.shard_offset_map == {0: (0, 4, 4), 1: (4, 8, 4)}
    assert len(layer.qweight.data_container) == 2
    assert torch.equal(
        layer.qweight.data_container[0], torch.ones((4, 4), dtype=torch.uint8)
    )
    assert torch.equal(
        layer.qweight.data_container[1], 2 * torch.ones((4, 4), dtype=torch.uint8)
    )


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

    assert calls == [((4, 4), 3), ((4, 4), 3)]
    assert out.shape == (2, 8)


def test_gguf_tuple_shard_loader_splits_fused_qweight(monkeypatch):
    register()
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )

    layer = MergedColumnParallelLinear(
        input_size=4,
        output_sizes=[4, 2, 6, 8],
        bias=False,
        quant_config=OOTGGUFConfig.from_config({}),
        disable_tp=True,
    )

    fused_qkv = torch.cat(
        [
            torch.full((4, 4), 1, dtype=torch.uint8),
            torch.full((2, 4), 2, dtype=torch.uint8),
            torch.full((6, 4), 3, dtype=torch.uint8),
        ],
        dim=0,
    )
    layer.qweight.weight_loader(layer.qweight, fused_qkv, (0, 1, 2))
    layer.qweight.weight_loader(
        layer.qweight, torch.full((8, 4), 4, dtype=torch.uint8), 3
    )
    layer.qweight_type.weight_loader(
        layer.qweight_type,
        torch.tensor(WeightType.Q4_0, dtype=torch.uint8),
        (0, 1, 2),
    )
    layer.qweight_type.weight_loader(
        layer.qweight_type, torch.tensor(WeightType.Q4_1, dtype=torch.uint8), 3
    )

    layer.quant_method.process_weights_after_loading(layer)

    assert layer.qweight.shard_id == [0, 1, 2, 3]
    assert layer.qweight.shard_offset_map == {
        0: (0, 4, 4),
        1: (4, 6, 4),
        2: (6, 12, 4),
        3: (12, 20, 4),
    }
    assert layer.qweight.numel() == 0
    assert torch.equal(
        layer.qweight.data_container[0], torch.full((4, 4), 1, dtype=torch.uint8)
    )
    assert torch.equal(
        layer.qweight.data_container[1], torch.full((2, 4), 2, dtype=torch.uint8)
    )
    assert torch.equal(
        layer.qweight.data_container[2], torch.full((6, 4), 3, dtype=torch.uint8)
    )
    assert torch.equal(
        layer.qweight.data_container[3], torch.full((8, 4), 4, dtype=torch.uint8)
    )
    assert layer.qweight_type.shard_weight_type == {
        0: WeightType.Q4_0,
        1: WeightType.Q4_0,
        2: WeightType.Q4_0,
        3: WeightType.Q4_1,
    }


def test_gguf_row_parallel_weight_loader_v2_omits_empty_shard_id(monkeypatch):
    register()
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(gguf_params_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        gguf_params_module, "get_tensor_model_parallel_world_size", lambda: 1
    )

    layer = RowParallelLinear(
        input_size=4,
        output_size=8,
        bias=False,
        quant_config=OOTGGUFConfig.from_config({}),
        disable_tp=True,
    )

    layer.qweight.weight_loader(layer.qweight, torch.ones((8, 4), dtype=torch.uint8))

    assert torch.equal(layer.qweight.data, torch.ones((8, 4), dtype=torch.uint8))


def test_gemma4_adapter_transforms_quantized_moe_names():
    adapter = Gemma4GGUFAdapter(PretrainedConfig(model_type="gemma4"))
    weight = torch.empty((2, 3), dtype=torch.uint8)

    transformed = dict(
        adapter.map_weights(
            [
                (
                    "model.language_model.layers.0.experts.gate_up_proj.qweight",
                    weight,
                ),
                (
                    "model.language_model.layers.0.experts.gate_up_proj.qweight_type",
                    torch.tensor(1, dtype=torch.uint8),
                ),
                (
                    "model.language_model.layers.0.experts.down_proj.qweight",
                    weight,
                ),
                (
                    "model.language_model.layers.0.experts.down_proj.qweight_type",
                    torch.tensor(1, dtype=torch.uint8),
                ),
            ]
        )
    )

    assert (
        "model.language_model.layers.0.moe.experts.routed_experts.w13_qweight"
        in transformed
    )
    assert (
        "model.language_model.layers.0.moe.experts.routed_experts.w13_qweight_type"
        in transformed
    )
    assert (
        "model.language_model.layers.0.moe.experts.routed_experts.w2_qweight"
        in transformed
    )
    assert (
        "model.language_model.layers.0.moe.experts.routed_experts.w2_qweight_type"
        in transformed
    )


def test_gemma4_gguf_mappings_match_current_hf_names():
    config = PretrainedConfig(model_type="gemma4", num_hidden_layers=2)
    config.vision_config = PretrainedConfig(num_hidden_layers=2)
    mapping: dict[str, str] = {}
    sideload_params = []

    _add_gemma4_gguf_mappings(config, mapping, sideload_params)

    assert mapping["blk.1.ffn_gate_inp.scale"] == (
        "model.language_model.layers.1.router.scale"
    )
    assert mapping["blk.1.ffn_gate_inp.weight"] == (
        "model.language_model.layers.1.router.proj.weight"
    )
    assert mapping["blk.1.ffn_down_exps.scale"] == (
        "model.language_model.layers.1.router.per_expert_scale"
    )
    assert mapping["blk.1.ffn_gate_up_exps.weight"] == (
        "model.language_model.layers.1.experts.gate_up_proj.weight"
    )
    assert mapping["blk.1.ffn_down_exps.weight"] == (
        "model.language_model.layers.1.experts.down_proj.weight"
    )
    assert mapping["v.blk.1.ln1.weight"] == (
        "model.vision_tower.encoder.layers.1.input_layernorm.weight"
    )
    assert mapping["v.blk.1.ln2.weight"] == (
        "model.vision_tower.encoder.layers.1.pre_feedforward_layernorm.weight"
    )


def test_gemma4_text_only_does_not_add_vision_projector_mappings():
    config = PretrainedConfig(
        architectures=["Gemma4ForCausalLM"],
        model_type="gemma4",
        num_hidden_layers=2,
    )
    mapping: dict[str, str] = {}
    sideload_params = []

    _add_gemma4_gguf_mappings(config, mapping, sideload_params)

    assert mapping["blk.1.ffn_gate_inp.scale"] == (
        "model.layers.1.router.scale"
    )
    assert mapping["blk.1.ffn_gate_up_exps.weight"] == (
        "model.layers.1.experts.gate_up_proj.weight"
    )
    assert "v.std_bias" not in mapping
    assert "v.patch_embd.weight" not in mapping
    assert "mm.input_projection.weight" not in mapping
    assert "v.blk.1.ln1.weight" not in mapping


def test_gemma4_causal_lm_with_vision_config_uses_text_layout():
    config = PretrainedConfig(
        architectures=["Gemma4ForCausalLM"],
        model_type="gemma4",
        num_hidden_layers=2,
    )
    config.vision_config = PretrainedConfig(num_hidden_layers=2)
    mapping: dict[str, str] = {}
    sideload_params = []

    _add_gemma4_gguf_mappings(config, mapping, sideload_params)

    assert mapping["blk.1.ffn_gate_inp.scale"] == (
        "model.layers.1.router.scale"
    )
    assert "v.std_bias" not in mapping
    assert "v.patch_embd.weight" not in mapping


def test_gemma4_adapter_flattens_patch_embed_weight():
    adapter = Gemma4GGUFAdapter(PretrainedConfig(model_type="gemma4"))
    weight = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)

    transformed = adapter.transform_weight(
        "model.vision_tower.patch_embedder.input_proj.weight",
        weight,
    )

    assert transformed.shape == (2, 60)
    assert torch.equal(transformed, weight.flatten(1))


def test_gemma4_mtp_gguf_mappings():
    config = PretrainedConfig(model_type="gemma4_assistant", num_hidden_layers=2)
    mapping: dict[str, str] = {}

    _add_gemma4_mtp_gguf_mappings(config, mapping)

    assert mapping["token_embd.weight"] == "model.embed_tokens.weight"
    assert mapping["nextn.pre_projection.weight"] == "model.pre_projection.weight"
    assert mapping["nextn.post_projection.weight"] == "model.post_projection.weight"
    assert mapping["blk.1.attn_q.weight"] == ("model.layers.1.self_attn.q_proj.weight")
    assert "blk.1.attn_k.weight" not in mapping
    assert "blk.1.attn_v.weight" not in mapping
    assert mapping["blk.1.ffn_gate.weight"] == ("model.layers.1.mlp.gate_proj.weight")
    assert mapping["blk.1.layer_output_scale.weight"] == "model.layers.1.layer_scalar"


def test_gguf_sequence_edge_accepts_scalar_and_sequence_values():
    assert _gguf_sequence_edge(None, first=True) is None
    assert _gguf_sequence_edge(8, first=True) == 8
    assert _gguf_sequence_edge(8, first=False) == 8
    assert _gguf_sequence_edge([8, 8, 8, 2], first=True) == 8
    assert _gguf_sequence_edge([8, 8, 8, 2], first=False) == 2


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
    qwen_control_tokens = [
        "<tool_call>",
        "</tool_call>",
        "<think>",
        "</think>",
    ]
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
    assert (tmp_path / "tokenizer-cache").is_dir()
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
    assert processor_config["image_processor"]["patch_size"] == 16
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
    assert calls[0][2]["tokens"] == [
        "<pad>",
        "<bos>",
        "<eos>",
        "hello",
        *qwen_mm_tokens,
        *qwen_control_tokens,
        "[PAD000]",
    ]
    assert calls[0][2]["token_type"] == [
        3,
        3,
        3,
        1,
        *([3] * len(qwen_mm_tokens)),
        *([4] * len(qwen_control_tokens)),
        5,
    ]
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
    assert (tokenizer_cache / "preprocessor_config.json").is_file()
    cached_tokenizer_config = json.loads(
        (tokenizer_cache / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    assert cached_tokenizer_config["additional_special_tokens"] == [
        *qwen_mm_tokens,
        *qwen_control_tokens,
    ]


def test_build_tokenizer_from_qwen35_gguf_uses_dense_arch_alias(
    tmp_path,
    monkeypatch,
):
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    main_reader = _FakeGGUFReader(
        {
            "general.architecture": "qwen35",
            "tokenizer.ggml.tokens": ["<pad>", "<bos>", "<eos>", "hello"],
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
            calls.append(("fast", kwargs))
            self.chat_template = None

        def save_pretrained(self, path):
            (path / "tokenizer.json").write_text("{}", encoding="utf-8")
            (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    def fake_convert(architecture, tokenizer_dict):
        calls.append(("convert", architecture, tokenizer_dict))
        return object(), {}

    monkeypatch.setenv(
        "VLLM_GGUF_TOKENIZER_CACHE",
        str(tmp_path / "tokenizer-cache"),
    )
    monkeypatch.setattr(
        gguf_tokenizer_builder_module.gguf,
        "GGUFReader",
        lambda path: main_reader,
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

    tokenizer_path = build_tokenizer_from_gguf(gguf_path)

    assert tokenizer_path is not None
    assert calls[0][0] == "convert"
    assert calls[0][1] == "qwen3"


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


def test_build_tokenizer_from_gguf_copies_local_sidecars_first(
    tmp_path,
    monkeypatch,
):
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    local_preprocessor = {"processor_class": "LocalProcessor"}
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(local_preprocessor),
        encoding="utf-8",
    )
    fake_reader = _FakeGGUFReader(
        {
            "general.architecture": "gemma4",
            "tokenizer.ggml.tokens": ["<pad>", "<bos>", "<eos>", "hello"],
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.merges": ["h ello"],
        }
    )

    class FakeTokenizer:
        def __init__(self, *args, **kwargs):
            pass

        def save_pretrained(self, path):
            (path / "tokenizer.json").write_text("{}", encoding="utf-8")

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
    copied = json.loads(
        (Path(tokenizer_path) / "preprocessor_config.json").read_text(encoding="utf-8")
    )
    assert copied == local_preprocessor


def test_build_tokenizer_from_gguf_patches_gemma4_special_tokens(
    tmp_path,
    monkeypatch,
):
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    fake_reader = _FakeGGUFReader(
        {
            "general.architecture": "gemma4",
            "tokenizer.ggml.tokens": [
                "<pad>",
                "<bos>",
                "<eos>",
                "<|image>",
                "<|image|>",
                "<image|>",
                "<|audio>",
                "<|audio|>",
                "<audio|>",
                "<|video|>",
            ],
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.merges": ["h ello"],
        }
    )

    class FakeTokenizer:
        def __init__(self, *args, **kwargs):
            pass

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
    tokenizer_config = json.loads(
        (Path(tokenizer_path) / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    assert tokenizer_config["processor_class"] == "Gemma4Processor"
    assert tokenizer_config["model_specific_special_tokens"]["image_token"] == (
        "<|image|>"
    )
    assert tokenizer_config["model_specific_special_tokens"]["boi_token"] == (
        "<|image>"
    )
    assert tokenizer_config["model_specific_special_tokens"]["eoi_token"] == (
        "<image|>"
    )
    assert tokenizer_config["extra_special_tokens"]["image_token"] == "<|image|>"
    assert tokenizer_config["extra_special_tokens"]["boi_token"] == "<|image>"
    assert tokenizer_config["extra_special_tokens"]["eoi_token"] == "<image|>"
    assert tokenizer_config["extra_special_tokens"]["audio_token"] == "<|audio|>"
    assert tokenizer_config["extra_special_tokens"]["video_token"] == "<|video|>"

    tokenizer_config_path = Path(tokenizer_path) / "tokenizer_config.json"
    tokenizer_config_path.write_text("{}", encoding="utf-8")
    assert build_tokenizer_from_gguf(gguf_path) == tokenizer_path
    tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    assert tokenizer_config["extra_special_tokens"]["image_token"] == "<|image|>"
    assert tokenizer_config["extra_special_tokens"]["boi_token"] == "<|image>"
    assert tokenizer_config["extra_special_tokens"]["eoi_token"] == "<image|>"


def test_extract_vision_config_accepts_single_value_metadata(monkeypatch):
    fake_reader = _FakeGGUFReader(
        {
            Keys.Clip.PROJECTOR_TYPE: VisionProjectorType.GEMMA3,
            Keys.ClipVision.EMBEDDING_LENGTH: [1152],
            Keys.ClipVision.FEED_FORWARD_LENGTH: [4304],
            Keys.ClipVision.BLOCK_COUNT: [27],
            Keys.ClipVision.Attention.HEAD_COUNT: [16],
            Keys.ClipVision.IMAGE_SIZE: [896],
            Keys.ClipVision.PATCH_SIZE: [14],
            Keys.ClipVision.Attention.LAYERNORM_EPS: [1e-6],
        }
    )
    monkeypatch.setattr(
        gguf_utils_module.gguf,
        "GGUFReader",
        lambda path: fake_reader,
    )

    config = extract_vision_config_from_gguf("mmproj.gguf")

    assert config is not None
    assert config.hidden_size == 1152
    assert config.intermediate_size == 4304
    assert config.num_hidden_layers == 27
    assert config.vision_use_head is False


def test_qwen35moe_gguf_config_is_normalized_for_mm(monkeypatch):
    fake_reader = _FakeGGUFReader(
        {
            "general.architecture": "qwen35moe",
            "qwen35moe.attention.key_length": 256,
            "qwen35moe.full_attention_interval": 4,
            "qwen35moe.nextn_predict_layers": 1,
            "qwen35moe.block_count": 41,
            "qwen35moe.rope.dimension_count": 64,
            "qwen35moe.rope.dimension_sections": [11, 11, 10, 0],
            "qwen35moe.rope.freq_base": 10000000.0,
        }
    )
    monkeypatch.setattr(gguf_utils_module, "check_gguf_file", lambda model: True)
    monkeypatch.setattr(
        gguf_utils_module,
        "extract_vocab_size_from_gguf",
        lambda model: None,
    )
    monkeypatch.setattr(
        gguf_utils_module,
        "extract_lm_head_from_gguf",
        lambda model: None,
    )
    monkeypatch.setattr(
        gguf_utils_module,
        "detect_gguf_multimodal",
        lambda model: "mmproj.gguf",
    )
    monkeypatch.setattr(
        gguf_utils_module.gguf,
        "GGUFReader",
        lambda path: fake_reader,
    )

    config = maybe_patch_hf_config_from_gguf(
        "model.gguf",
        PretrainedConfig(model_type="qwen35moe"),
    )

    assert config.model_type == "qwen3_5_moe"
    assert config.architectures == ["Qwen3_5MoeForConditionalGeneration"]
    assert config.mtp_num_hidden_layers == 1
    assert config.num_nextn_predict_layers == 1
    assert config.num_hidden_layers == 40
    assert config.full_attention_interval == 4
    assert config.rope_parameters["mrope_section"] == [11, 11, 10]
    assert config.rope_parameters["mrope_interleaved"] is True


def test_qwen35_gguf_config_subtracts_nextn_layers(monkeypatch):
    fake_reader = _FakeGGUFReader(
        {
            "general.architecture": "qwen35",
            "qwen35.attention.key_length": 256,
            "qwen35.full_attention_interval": 4,
            "qwen35.nextn_predict_layers": 1,
            "qwen35.block_count": 65,
            "qwen35.rope.dimension_count": 64,
            "qwen35.rope.dimension_sections": [11, 11, 10, 0],
            "qwen35.rope.freq_base": 10000000.0,
        }
    )
    monkeypatch.setattr(gguf_utils_module, "check_gguf_file", lambda model: True)
    monkeypatch.setattr(
        gguf_utils_module,
        "extract_vocab_size_from_gguf",
        lambda model: None,
    )
    monkeypatch.setattr(
        gguf_utils_module,
        "extract_lm_head_from_gguf",
        lambda model: None,
    )
    monkeypatch.setattr(
        gguf_utils_module,
        "detect_gguf_multimodal",
        lambda model: None,
    )
    monkeypatch.setattr(
        gguf_utils_module.gguf,
        "GGUFReader",
        lambda path: fake_reader,
    )

    config = maybe_patch_hf_config_from_gguf(
        "model.gguf",
        PretrainedConfig(model_type="qwen35"),
    )

    assert config.model_type == "qwen3_5"
    assert config.architectures == ["Qwen3_5ForCausalLM"]
    assert config.mtp_num_hidden_layers == 1
    assert config.num_nextn_predict_layers == 1
    assert config.num_hidden_layers == 64
    assert config.full_attention_interval == 4
    assert config.rope_parameters["mrope_section"] == [11, 11, 10]
    assert config.rope_parameters["mrope_interleaved"] is True


def test_default_adapter_adds_mmproj_for_multimodal_config(tmp_path, monkeypatch):
    main_path = tmp_path / "model.gguf"
    mmproj_path = tmp_path / "mmproj-BF16.gguf"
    config = PretrainedConfig(model_type="qwen3_5_moe")
    config.vision_config = PretrainedConfig()
    adapter = GGUFWeightsAdapter(config)

    monkeypatch.setattr(
        default_adapter_module,
        "detect_gguf_multimodal",
        lambda model: mmproj_path,
    )

    assert adapter._get_weight_sources(str(main_path), config) == [
        str(main_path),
        str(mmproj_path),
    ]


def test_default_adapter_keeps_text_only_sources_without_mmproj(tmp_path, monkeypatch):
    main_path = tmp_path / "model.gguf"
    config = PretrainedConfig(model_type="qwen3_5_moe")
    adapter = GGUFWeightsAdapter(config)

    monkeypatch.setattr(
        default_adapter_module,
        "detect_gguf_multimodal",
        lambda model: tmp_path / "mmproj-BF16.gguf",
    )

    assert adapter._get_weight_sources(str(main_path), config) == [str(main_path)]


def test_default_adapter_ignores_mmproj_for_causal_lm_architecture(
    tmp_path, monkeypatch
):
    main_path = tmp_path / "model.gguf"
    config = PretrainedConfig(
        model_type="qwen3_5",
        architectures=["Qwen3_5ForCausalLM"],
    )
    config.vision_config = PretrainedConfig()
    adapter = GGUFWeightsAdapter(config)

    monkeypatch.setattr(
        default_adapter_module,
        "detect_gguf_multimodal",
        lambda model: tmp_path / "mmproj-BF16.gguf",
    )

    assert adapter._get_weight_sources(str(main_path), config) == [str(main_path)]


def _build_qwen3_5_test_name_map(
    monkeypatch,
    config,
    state_names,
    tensor_name_map=None,
):
    tensor_name_map = tensor_name_map or {}

    class FakeNameMap:
        def get_name(self, name):
            return tensor_name_map.get(name)

    class FakeAutoModel:
        @staticmethod
        def from_config(config, trust_remote_code=False):
            return SimpleNamespace(
                state_dict=lambda: {
                    name: torch.empty((), device="meta") for name in state_names
                },
            )

    monkeypatch.setattr(
        default_adapter_module.gguf,
        "MODEL_ARCH_NAMES",
        {object(): "qwen35", object(): "qwen35moe"},
    )
    monkeypatch.setattr(
        default_adapter_module.gguf,
        "get_tensor_name_map",
        lambda *args, **kwargs: FakeNameMap(),
    )
    monkeypatch.setattr(
        default_adapter_module,
        "AutoModelForImageTextToText",
        FakeAutoModel,
    )
    monkeypatch.setattr(
        default_adapter_module,
        "AutoModelForCausalLM",
        FakeAutoModel,
    )

    model_config = SimpleNamespace(hf_config=config, trust_remote_code=False)
    return GGUFWeightsAdapter(config).build_name_map(model_config)


def test_qwen3_5_dense_multimodal_maps_visual_merger(monkeypatch):
    config = PretrainedConfig(
        model_type="qwen3_5",
        architectures=["Qwen3_5ForConditionalGeneration"],
        num_hidden_layers=1,
        layer_types=["linear_attention"],
    )
    config.vision_config = PretrainedConfig(num_hidden_layers=1)
    state_names = [
        "model.language_model.embed_tokens.weight",
        "model.visual.patch_embed.proj.weight.1",
        "model.visual.merger.linear_fc1.weight",
        "model.visual.merger.linear_fc1.bias",
        "model.visual.merger.linear_fc2.weight",
        "model.visual.merger.linear_fc2.bias",
        "model.visual.merger.norm.weight",
        "model.visual.merger.norm.bias",
    ]

    mapping = _build_qwen3_5_test_name_map(monkeypatch, config, state_names)

    assert mapping["token_embd.weight"] == (
        "model.language_model.embed_tokens.weight"
    )
    assert mapping["v.patch_embd.weight.1"] == (
        "model.visual.patch_embed.proj.weight.1"
    )
    assert mapping["v.post_ln.weight"] == "model.visual.merger.norm.weight"
    assert mapping["v.post_ln.bias"] == "model.visual.merger.norm.bias"
    assert mapping["mm.0.weight"] == "model.visual.merger.linear_fc1.weight"
    assert mapping["mm.0.bias"] == "model.visual.merger.linear_fc1.bias"
    assert mapping["mm.2.weight"] == "model.visual.merger.linear_fc2.weight"
    assert mapping["mm.2.bias"] == "model.visual.merger.linear_fc2.bias"


def test_qwen3_5_moe_multimodal_maps_token_embd_to_language_model(monkeypatch):
    config = PretrainedConfig(
        model_type="qwen3_5_moe",
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        num_hidden_layers=1,
        layer_types=["linear_attention"],
    )
    config.vision_config = PretrainedConfig(num_hidden_layers=1)

    mapping = _build_qwen3_5_test_name_map(
        monkeypatch,
        config,
        ["model.language_model.embed_tokens.weight"],
    )

    assert mapping["token_embd.weight"] == (
        "model.language_model.embed_tokens.weight"
    )


def test_qwen3_5_text_only_does_not_add_visual_merger_mappings(monkeypatch):
    config = PretrainedConfig(
        model_type="qwen3_5",
        num_hidden_layers=1,
        layer_types=["linear_attention"],
    )

    mapping = _build_qwen3_5_test_name_map(monkeypatch, config, [])

    assert "v.patch_embd.weight.1" not in mapping
    assert "v.post_ln.weight" not in mapping
    assert "mm.0.weight" not in mapping
    assert "mm.2.weight" not in mapping


def test_qwen3_5_causal_lm_uses_text_weight_layout(monkeypatch):
    config = PretrainedConfig(
        model_type="qwen3_5",
        architectures=["Qwen3_5ForCausalLM"],
        num_hidden_layers=1,
        layer_types=["linear_attention"],
    )
    config.vision_config = PretrainedConfig(num_hidden_layers=1)

    mapping = _build_qwen3_5_test_name_map(
        monkeypatch,
        config,
        [
            "model.embed_tokens.weight",
            "model.layers.0.linear_attn.dt_bias",
        ],
        {
            "model.embed_tokens": "token_embd",
            "model.layers.0.linear_attn.dt_bias": "blk.0.ssm_dt",
        },
    )

    assert mapping["token_embd.weight"] == "model.embed_tokens.weight"
    assert mapping["blk.0.ssm_dt.bias"] == "model.layers.0.linear_attn.dt_bias"
    assert "v.patch_embd.weight.1" not in mapping
    assert "mm.0.weight" not in mapping
    assert "mm.2.weight" not in mapping


def test_qwen3_5_adapter_reshapes_gguf_weights():
    adapter = Qwen3_5GGUFAdapter(PretrainedConfig(model_type="qwen3_5_moe"))
    shared_gate = torch.arange(4)
    conv1d = torch.arange(6).reshape(2, 3)

    assert adapter.transform_weight(
        "model.layers.0.mlp.shared_expert_gate",
        shared_gate,
    ).shape == (1, 4)
    transformed_conv = adapter.transform_weight(
        "model.layers.0.linear_attn.conv1d.weight",
        conv1d,
    )
    assert transformed_conv.shape == (2, 1, 3)
    assert torch.equal(transformed_conv[:, 0, :], conv1d)


def test_qwen3_5_adapter_combines_split_patch_embed_weight():
    adapter = Qwen3_5GGUFAdapter(PretrainedConfig(model_type="qwen3_5_moe"))
    patch_weight = torch.zeros((4, 3, 16, 16))
    patch_weight_1 = torch.ones((4, 3, 16, 16))
    other_weight = torch.full((2, 2), 2.0)

    mapped = list(
        adapter.map_weights(
            [
                ("model.visual.patch_embed.proj.weight.1", patch_weight_1),
                ("model.layers.0.self_attn.q_proj.weight", other_weight),
                ("model.visual.patch_embed.proj.weight", patch_weight),
            ]
        )
    )

    assert mapped[0][0] == "model.layers.0.self_attn.q_proj.weight"
    assert torch.equal(mapped[0][1], other_weight)
    assert mapped[1][0] == "model.visual.patch_embed.proj.weight"
    assert mapped[1][1].shape == (4, 3, 2, 16, 16)
    assert torch.equal(mapped[1][1][:, :, 0], patch_weight)
    assert torch.equal(mapped[1][1][:, :, 1], patch_weight_1)


def test_qwen3_5_mtp_gguf_mappings():
    config = PretrainedConfig(
        model_type="qwen3_5_moe",
        num_hidden_layers=40,
        mtp_num_hidden_layers=2,
    )
    mapping: dict[str, str] = {}
    sideload_params = []

    _add_qwen3_5_mtp_gguf_mappings(config, mapping, sideload_params)

    assert mapping["blk.40.attn_q.weight"] == ("mtp.layers.0.self_attn.q_proj.weight")
    assert mapping["blk.40.attn_k.weight"] == ("mtp.layers.0.self_attn.k_proj.weight")
    assert mapping["blk.41.attn_q.weight"] == ("mtp.layers.1.self_attn.q_proj.weight")
    assert mapping["blk.40.ffn_gate_inp.weight"] == "mtp.layers.0.mlp.gate.weight"
    assert mapping["blk.40.ffn_gate_inp_shexp.weight"] == (
        "mtp.layers.0.mlp.shared_expert_gate.weight"
    )
    assert mapping["blk.40.ffn_gate_exps.weight"] == (
        "mtp.layers.0.mlp.experts.0.gate_proj.weight"
    )
    assert mapping["blk.40.nextn.eh_proj.weight"] == "mtp.fc.weight"
    assert mapping["blk.40.nextn.shared_head_norm.weight"] == "mtp.norm.weight"
    assert "blk.41.nextn.eh_proj.weight" not in mapping
    assert sideload_params[0].fullmatch("mtp.layers.0.mlp.experts.15.gate_proj.weight")


def test_qwen3_5_dense_mtp_gguf_mappings_use_trunk_layer_count():
    config = PretrainedConfig(
        model_type="qwen3_5",
        num_hidden_layers=64,
        mtp_num_hidden_layers=1,
    )
    mapping: dict[str, str] = {}
    sideload_params = []

    _add_qwen3_5_mtp_gguf_mappings(config, mapping, sideload_params)

    assert mapping["blk.64.attn_q.weight"] == ("mtp.layers.0.self_attn.q_proj.weight")
    assert mapping["blk.64.attn_k.weight"] == ("mtp.layers.0.self_attn.k_proj.weight")
    assert mapping["blk.64.attn_v.weight"] == ("mtp.layers.0.self_attn.v_proj.weight")
    assert mapping["blk.64.ffn_gate.weight"] == "mtp.layers.0.mlp.gate_proj.weight"
    assert mapping["blk.64.ffn_up.weight"] == "mtp.layers.0.mlp.up_proj.weight"
    assert mapping["blk.64.ffn_down.weight"] == "mtp.layers.0.mlp.down_proj.weight"
    assert mapping["blk.64.nextn.eh_proj.weight"] == "mtp.fc.weight"
    assert mapping["blk.64.nextn.shared_head_norm.weight"] == "mtp.norm.weight"
    assert "blk.65.attn_q.weight" not in mapping


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
        "file_or_path_exists",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        gguf_config_parser_module,
        "maybe_patch_hf_config_from_gguf",
        lambda model, config: config,
    )

    config_dict, config = GGUFConfigParser().parse(gguf_path, trust_remote_code=False)

    assert calls["model"] == gguf_path.parent
    assert calls["trust_remote_code"] is False
    assert calls["gguf_file"] is None
    assert config_dict["norm_topk_prob"] is True
    assert config.architectures == ["Qwen3MoeForCausalLM"]


def test_gguf_config_parser_prefers_sidecar_config(
    tmp_path,
    monkeypatch,
):
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    calls = {}

    def fake_parse(
        self, model, trust_remote_code, revision=None, code_revision=None, **kwargs
    ):
        calls["model"] = model
        calls["gguf_file"] = kwargs.get("gguf_file")
        return {}, PretrainedConfig(
            model_type="qwen3_5_moe",
            architectures=["Qwen3_5MoeForCausalLM"],
        )

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
    assert calls["gguf_file"] is None
    assert config.model_type == "qwen3_5_moe"
    assert config_dict["architectures"] == ["Qwen3_5MoeForCausalLM"]


def test_gguf_config_parser_uses_gguf_file_when_parent_has_no_config(
    tmp_path, monkeypatch
):
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
        "file_or_path_exists",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        gguf_config_parser_module,
        "maybe_patch_hf_config_from_gguf",
        lambda model, config: config,
    )

    GGUFConfigParser().parse(gguf_path, trust_remote_code=False)

    assert calls["model"] == gguf_path.parent
    assert calls["trust_remote_code"] is False
    assert calls["gguf_file"] == gguf_path.name


def test_gguf_config_parser_resolves_presplit_local_gguf(
    tmp_path,
    monkeypatch,
):
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    calls = {}

    def fake_parse(
        self, model, trust_remote_code, revision=None, code_revision=None, **kwargs
    ):
        calls["model"] = model
        calls["trust_remote_code"] = trust_remote_code
        calls["revision"] = revision
        calls["gguf_file"] = kwargs.get("gguf_file")
        return {}, PretrainedConfig(model_type="qwen3_moe")

    monkeypatch.setattr(
        gguf_config_parser_module,
        "resolve_gguf_config_source",
        lambda model, revision=None: "base/repo",
    )
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

    GGUFConfigParser().parse(
        gguf_path.parent,
        trust_remote_code=True,
        revision="gguf-revision",
        gguf_file=gguf_path.name,
    )

    assert calls["model"] == "base/repo"
    assert calls["trust_remote_code"] is False
    assert calls["revision"] is None
    assert calls["gguf_file"] is None


def test_gguf_config_parser_preserves_patched_architecture(tmp_path, monkeypatch):
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")

    def fake_parse(
        self, model, trust_remote_code, revision=None, code_revision=None, **kwargs
    ):
        return {}, PretrainedConfig(model_type="qwen35moe")

    def fake_patch(model, config):
        config.update(
            {
                "model_type": "qwen3_5_moe",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            }
        )
        return config

    monkeypatch.setattr(
        gguf_config_parser_module.HFConfigParser,
        "parse",
        fake_parse,
    )
    monkeypatch.setattr(
        gguf_config_parser_module,
        "file_or_path_exists",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        gguf_config_parser_module,
        "maybe_patch_hf_config_from_gguf",
        fake_patch,
    )

    config_dict, config = GGUFConfigParser().parse(gguf_path, trust_remote_code=False)

    assert config.model_type == "qwen3_5_moe"
    assert config.architectures == ["Qwen3_5MoeForConditionalGeneration"]
    assert config_dict["architectures"] == ["Qwen3_5MoeForConditionalGeneration"]


def test_gguf_config_source_uses_nearest_parent_config(tmp_path):
    model_dir = tmp_path / "model"
    mtp_dir = model_dir / "MTP"
    mtp_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    gguf_path = mtp_dir / "draft.gguf"
    gguf_path.write_bytes(b"GGUF")

    assert resolve_gguf_config_source(gguf_path) == model_dir


def test_gguf_config_parser_disables_trust_for_base_model_redirect(
    tmp_path, monkeypatch
):
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    calls = {}

    def fake_parse(
        self, model, trust_remote_code, revision=None, code_revision=None, **kwargs
    ):
        calls["model"] = model
        calls["trust_remote_code"] = trust_remote_code
        calls["revision"] = revision
        calls["gguf_file"] = kwargs.get("gguf_file")
        return {}, PretrainedConfig(model_type="qwen3_moe")

    monkeypatch.setattr(
        gguf_config_parser_module,
        "resolve_gguf_config_source",
        lambda model, revision=None: "base/repo",
    )
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

    GGUFConfigParser().parse(
        gguf_path,
        trust_remote_code=True,
        revision="gguf-revision",
    )

    assert calls["model"] == "base/repo"
    assert calls["trust_remote_code"] is False
    assert calls["revision"] is None
    assert calls["gguf_file"] is None


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
    assert captured["model"] == "/tmp/model.gguf"
    assert captured["model_weights"] == "/tmp/model.gguf"
    assert captured["quantization"] == "gguf"
    assert engine_args.load_format == "gguf"


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

    assert captured["tokenizer"] == "/tmp/gguf-tokenizer-cache"


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

    assert captured["tokenizer"] == "/tmp/tokenizer"


def test_register_skips_speculator_probe_for_gguf():
    register()

    model, tokenizer, speculative_config = (
        config_module.maybe_override_with_speculators(
            model="/tmp/model.gguf",
            tokenizer="/tmp/tokenizer",
            trust_remote_code=False,
            revision=None,
            vllm_speculative_config={"foo": "bar"},
            hf_token=None,
        )
    )

    assert model == "/tmp/model.gguf"
    assert tokenizer == "/tmp/tokenizer"
    assert speculative_config == {"foo": "bar"}


def test_register_speculator_probe_prefers_sidecar_config(
    tmp_path,
    monkeypatch,
):
    register()
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    calls = {}

    def fake_get_config_dict(config_source, **kwargs):
        calls["config_source"] = config_source
        calls["gguf_file"] = kwargs.get("gguf_file")
        return {"model_type": "qwen3_5"}, {}

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

    monkeypatch.setattr(
        arg_utils_module,
        "ModelConfig",
        fake_model_config,
    )

    engine_args = EngineArgs(
        model=str(gguf_path),
        tokenizer="/tmp/tokenizer",
        trust_remote_code=True,
        revision="gguf-revision",
    )
    engine_args.create_model_config()

    assert captured["trust_remote_code"] is False


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

    monkeypatch.setattr(
        arg_utils_module,
        "ModelConfig",
        fake_model_config,
    )

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

    monkeypatch.setattr(
        arg_utils_module,
        "ModelConfig",
        fake_model_config,
    )

    engine_args = EngineArgs(
        model="verifier/repo",
        tokenizer="verifier/repo",
        trust_remote_code=True,
        speculative_config={"model": str(gguf_path)},
    )
    engine_args.create_model_config()

    assert captured["trust_remote_code"] is False


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
    assert layer.qweight.numel() == 0
    assert torch.equal(layer.qweight.data_container[0], q)
    assert torch.equal(layer.qweight.data_container[1], k)
    assert torch.equal(layer.qweight.data_container[2], v)


def test_gguf_linear_preserves_cuda_weight_device(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for device placement test")

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
    assert [shard.device.type for shard in layer.qweight.data_container] == [
        "cuda",
        "cuda",
    ]
    assert layer.qweight_type.device.type == "cuda"


def test_gguf_iq4_xs_batched_linear_uses_mmq_v2(monkeypatch):
    import vllm_gguf_plugin.quantization.linear as gguf_linear_module
    from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf

    qweight = torch.empty((32, 136), dtype=torch.uint8)
    x = torch.empty((17, 256), dtype=torch.bfloat16)
    expected = torch.empty((17, 32), dtype=torch.bfloat16)
    calls = []

    def fake_mmq_v2(qweight_arg, x_arg, row_arg):
        calls.append((qweight_arg, x_arg, row_arg))
        return expected

    monkeypatch.setattr(
        gguf_linear_module.ops, "ggml_mul_mat_a8_iq4_xs_mmq_v2", fake_mmq_v2
    )

    output = _fused_mul_mat_gguf(x, qweight, WeightType.IQ4_XS)

    assert output is expected
    assert calls == [(qweight, x, qweight.shape[0])]


def test_gguf_iq4_xs_single_token_linear_keeps_mmvq(monkeypatch):
    import vllm_gguf_plugin.quantization.linear as gguf_linear_module
    from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf

    qweight = torch.empty((32, 136), dtype=torch.uint8)
    x = torch.empty((1, 256), dtype=torch.bfloat16)
    expected = torch.empty((1, 32), dtype=torch.bfloat16)
    calls = []

    def fake_mmvq(qweight_arg, x_arg, qweight_type_arg, row_arg):
        calls.append((qweight_arg, x_arg, qweight_type_arg, row_arg))
        return expected

    def fail_mmq_v2(*args, **kwargs):
        raise AssertionError("IQ4_XS batch-size-1 path must keep MMVQ")

    monkeypatch.setattr(gguf_linear_module.ops, "ggml_mul_mat_vec_a8", fake_mmvq)
    monkeypatch.setattr(
        gguf_linear_module.ops, "ggml_mul_mat_a8_iq4_xs_mmq_v2", fail_mmq_v2
    )

    output = _fused_mul_mat_gguf(x, qweight, WeightType.IQ4_XS)

    assert output is expected
    assert calls == [(qweight, x, WeightType.IQ4_XS, qweight.shape[0])]


def test_gguf_iq4_xs_batched_moe_uses_mmq_v2(monkeypatch):
    import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe_module

    import vllm_gguf_plugin.quantization.fused_moe as gguf_moe_module
    from vllm_gguf_plugin.quantization.fused_moe import _fused_moe_gguf

    def fake_align(topk_ids, block_size, num_experts):
        del num_experts
        num_ids = topk_ids.numel()
        padded = ((num_ids + block_size - 1) // block_size) * block_size
        sorted_token_ids = torch.arange(padded, dtype=torch.int32)
        sorted_token_ids[num_ids:] = -1
        expert_ids = torch.zeros(padded // block_size, dtype=torch.int32)
        num_tokens_post_padded = torch.tensor([padded], dtype=torch.int32)
        return sorted_token_ids, expert_ids, num_tokens_post_padded

    def fake_apply_moe_activation(activation, output, input_):
        del activation
        output.copy_(input_[..., : output.shape[-1]])

    calls = []

    def fake_moe_v2(
        x,
        weight,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        row,
        top_k,
        tokens,
    ):
        calls.append((x.shape, weight.shape, row, top_k, tokens))
        assert sorted_token_ids.dtype == torch.int32
        assert expert_ids.dtype == torch.int32
        assert num_tokens_post_padded.dtype == torch.int32
        return torch.ones((tokens * top_k, row), dtype=x.dtype)

    def fake_moe_sum(input_, output):
        output.copy_(input_.sum(dim=1))

    monkeypatch.setattr(fused_moe_module, "moe_align_block_size", fake_align)
    monkeypatch.setattr(
        gguf_moe_module, "apply_moe_activation", fake_apply_moe_activation
    )
    monkeypatch.setattr(gguf_moe_module.ops, "ggml_moe_a8_iq4_xs_mmq_v2", fake_moe_v2)
    monkeypatch.setattr(gguf_moe_module.ops, "moe_sum", fake_moe_sum)

    x = torch.ones((65, 4), dtype=torch.float32)
    w1 = torch.empty((4, 8, 4), dtype=torch.uint8)
    w2 = torch.empty((4, 4, 4), dtype=torch.uint8)
    topk_weights = torch.ones((65, 2), dtype=torch.float32)
    topk_ids = torch.zeros((65, 2), dtype=torch.int32)

    output = _fused_moe_gguf(
        x,
        w1,
        w2,
        topk_weights,
        topk_ids,
        WeightType.IQ4_XS,
        WeightType.IQ4_XS,
        "silu",
    )

    assert output.shape == x.shape
    assert calls == [
        (torch.Size([65, 4]), torch.Size([4, 8, 4]), 8, 2, 65),
        (torch.Size([130, 4]), torch.Size([4, 4, 4]), 4, 1, 130),
    ]


def _make_iq4_xs_weight(
    num_rows: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    weight = torch.randint(0, 256, (num_rows, 136), dtype=torch.uint8, generator=gen)
    # block_iq4_xs layout:
    #   half d, uint16 scales_h, uint8 scales_l[4], uint8 qs[128].
    # Use d=1 and 6-bit sub-block scale 33, which decodes to a small positive
    # scale while still exercising the IQ4_XS scale unpacking path.
    weight[:, 0:2] = torch.tensor([0x00, 0x3C], dtype=torch.uint8)
    weight[:, 2:4] = torch.tensor([0xAA, 0xAA], dtype=torch.uint8)
    weight[:, 4:8] = 0x11
    return weight.to(device=device)


def test_gguf_iq4_xs_batched_moe_matches_slow_reference_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GGUF MoE kernel comparison")
    if not (
        hasattr(torch.ops, "_C_gguf")
        and hasattr(torch.ops._C_gguf, "ggml_moe_a8_iq4_xs_mmq_v2")
        and hasattr(torch.ops._C_gguf, "ggml_mul_mat_vec_a8")
    ):
        pytest.skip("GGUF CUDA extension with IQ4_XS MoE v2 is not available")

    from vllm.model_executor.layers.fused_moe.activation import (
        MoEActivation,
        apply_moe_activation,
    )

    from vllm_gguf_plugin.quantization.fused_moe import _fused_moe_gguf
    from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf

    device = torch.device("cuda")
    dtype = torch.float16
    num_tokens = 65
    hidden_size = 256
    intermediate_size = 8
    num_experts = 4
    top_k = 2
    torch.manual_seed(0)

    x = torch.randn((num_tokens, hidden_size), dtype=dtype, device=device) * 0.05
    w1 = torch.stack(
        [
            _make_iq4_xs_weight(intermediate_size * 2, 100 + expert, device)
            for expert in range(num_experts)
        ]
    )
    w2 = torch.stack(
        [
            _make_iq4_xs_weight(hidden_size, 200 + expert, device)
            for expert in range(num_experts)
        ]
    )
    topk_ids = torch.tensor(
        [
            [token % num_experts, (token + 1) % num_experts]
            for token in range(num_tokens)
        ],
        dtype=torch.int32,
        device=device,
    )
    topk_weights = torch.tensor([0.65, 0.35], dtype=dtype, device=device).repeat(
        num_tokens,
        1,
    )

    out = _fused_moe_gguf(
        x,
        w1,
        w2,
        topk_weights,
        topk_ids,
        WeightType.IQ4_XS,
        WeightType.IQ4_XS,
        "silu",
    )

    activation_enum = MoEActivation.from_str("silu")
    ref = torch.empty_like(out)
    for token_idx in range(num_tokens):
        token_out = None
        token_x = x[token_idx : token_idx + 1]
        for route_idx in range(top_k):
            expert_idx = int(topk_ids[token_idx, route_idx].item())
            hidden = _fused_mul_mat_gguf(
                token_x,
                w1[expert_idx],
                WeightType.IQ4_XS,
            )
            activated = torch.empty(
                (1, intermediate_size),
                dtype=hidden.dtype,
                device=device,
            )
            apply_moe_activation(activation_enum, activated, hidden)
            projected = _fused_mul_mat_gguf(
                activated,
                w2[expert_idx],
                WeightType.IQ4_XS,
            ).mul(topk_weights[token_idx, route_idx])
            token_out = projected if token_out is None else token_out + projected
        ref[token_idx] = token_out

    torch.testing.assert_close(out, ref, atol=5e-1, rtol=5e-2)
