# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path

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
from vllm_gguf_plugin import OOTGGUFConfig, OOTGGUFModelLoader, register
from vllm_gguf_plugin.config_parser import GGUFConfigParser
from vllm_gguf_plugin.gguf_tokenizer_builder import build_tokenizer_from_gguf
from vllm_gguf_plugin.quantization import (
    GGUFUninitializedParameter,
    GGUFWeightParameter,
    GGUFWeightTypeParameter,
)


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
    assert (tokenizer_cache / "preprocessor_config.json").is_file()


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
