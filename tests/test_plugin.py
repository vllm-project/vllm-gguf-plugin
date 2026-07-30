# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import vllm.engine.arg_utils as arg_utils_module
import vllm.envs as envs_module
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
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader import get_model_loader
from vllm.transformers_utils.config import get_config_parser
from vllm.transformers_utils.configs.kimi_k3 import KimiK3Config
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.v1.kv_cache_interface import MLAAttentionSpec

import vllm_gguf_plugin.config_parser as gguf_config_parser_module
import vllm_gguf_plugin.quantization as gguf_quantization
import vllm_gguf_plugin.quantization.params as gguf_params_module
from vllm_gguf_plugin import OOTGGUFConfig, OOTGGUFModelLoader, register
from vllm_gguf_plugin.config_parser import (
    KIMI_K3_GGUF_TEXT_ARCH,
    KIMI_K3_GGUF_TEXT_MARKER,
    GGUFConfigParser,
)
from vllm_gguf_plugin.loader import (
    _load_weights_strict,
    _normalize_kimi_k3_mla_context_parallelism,
)
from vllm_gguf_plugin.quantization import (
    GGUFUninitializedParameter,
    GGUFUninitializedWeightParameter,
    GGUFWeightParameter,
    GGUFWeightTypeParameter,
)


def test_register_overrides_gguf_config():
    register()

    quant_config = get_quantization_config("gguf")

    assert quant_config is OOTGGUFConfig
    assert config_module._CONFIG_REGISTRY["kimi_k3"] is KimiK3Config


def test_register_overrides_gguf_loader():
    register()

    model_loader = get_model_loader(LoadConfig(load_format="gguf"))

    assert isinstance(model_loader, OOTGGUFModelLoader)


def test_register_is_idempotent():
    register()
    register()

    assert get_quantization_config("gguf") is OOTGGUFConfig
    assert isinstance(
        get_model_loader(LoadConfig(load_format="gguf")), OOTGGUFModelLoader
    )
    assert isinstance(get_config_parser("gguf"), GGUFConfigParser)


def test_register_preserves_native_kimi_k3_mla_cache_spec():
    register()

    spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.bfloat16,
        non_causal_multi_token_decode=False,
    )

    assert spec.block_size == 16
    assert spec.head_size == 256

    non_causal_spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.bfloat16,
        non_causal_multi_token_decode=True,
    )
    assert non_causal_spec.non_causal_multi_token_decode


def test_register_restores_kimi_k3_routed_down_proj_threshold(monkeypatch):
    monkeypatch.setenv("VLLM_ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD", "17")
    register()

    assert envs_module.VLLM_ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD == 17


def test_loader_normalizes_kimi_k3_mla_disabled_cp_sentinel():
    from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention

    attention = MultiHeadLatentAttention.__new__(MultiHeadLatentAttention)
    torch.nn.Module.__init__(attention)
    attention.impl = type("AttentionImpl", (), {"dcp_world_size": -1, "dcp_rank": -1})()
    model = torch.nn.Sequential(attention)

    _normalize_kimi_k3_mla_context_parallelism(model)

    assert attention.impl.dcp_world_size == 1
    assert attention.impl.dcp_rank == 0


def test_loader_strict_weight_audit_accepts_complete_parameter_set():
    class CompleteModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Parameter(torch.empty(1))
            self.second = torch.nn.Parameter(torch.empty(1))

        def load_weights(self, weights):
            return {name for name, _ in weights}

    model = CompleteModel()

    loaded = _load_weights_strict(
        model,
        [("first", torch.ones(1)), ("second", torch.ones(1))],
    )

    assert loaded == {"first", "second"}


def test_loader_strict_weight_audit_rejects_silently_skipped_parameter():
    class IncompleteModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Parameter(torch.empty(1))
            self.second = torch.nn.Parameter(torch.empty(1))

        def load_weights(self, weights):
            return {"first"}

    model = IncompleteModel()

    with pytest.raises(
        ValueError,
        match=r"not initialized.*\n  second",
    ):
        _load_weights_strict(model, [("first", torch.ones(1))])


def test_loader_strict_weight_audit_requires_tracking_result():
    class UntrackedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(1))

        def load_weights(self, weights):
            return None

    with pytest.raises(ValueError, match="did not report"):
        _load_weights_strict(UntrackedModel(), [("weight", torch.ones(1))])


def test_oot_config_reuses_in_tree_behavior():
    quant_config = OOTGGUFConfig.from_config({})

    assert isinstance(quant_config, OOTGGUFConfig)
    assert quant_config.get_name() == "gguf"
    assert repr(quant_config) == "GGUFConfig()"


def test_supported_act_dtypes_includes_bfloat16():
    quant_config = OOTGGUFConfig.from_config({})
    supported = quant_config.get_supported_act_dtypes()

    assert torch.half in supported
    assert torch.bfloat16 in supported
    assert torch.float32 in supported


def test_gguf_linear_uses_weight_loader_v2(monkeypatch):
    register()
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(gguf_params_module, "get_tensor_model_parallel_rank", lambda: 0)
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


def test_gguf_merged_column_loader_uses_distributed_tp_rank(monkeypatch):
    monkeypatch.setattr(gguf_params_module, "get_tensor_model_parallel_rank", lambda: 2)
    param = GGUFUninitializedWeightParameter()
    param.data_container = []
    param.shard_id = []
    param.shard_id_map = {}
    loaded = torch.arange(16 * 4, dtype=torch.uint8).reshape(16, 4)

    param.load_merged_column_weight(loaded, shard_id=0, shard_size=4)

    assert torch.equal(param.data_container[0], loaded[8:12])


def test_gguf_qkv_loader_uses_distributed_tp_rank(monkeypatch):
    monkeypatch.setattr(gguf_params_module, "get_tensor_model_parallel_rank", lambda: 3)
    param = GGUFUninitializedWeightParameter()
    param.data_container = []
    param.shard_id = []
    param.shard_id_map = {}
    loaded = torch.arange(8 * 4, dtype=torch.uint8).reshape(8, 4)

    param.load_qkv_weight(loaded, shard_id="q", shard_size=2, num_heads=2)
    param.load_qkv_weight(loaded, shard_id="k", shard_size=2, num_heads=2)

    assert torch.equal(param.data_container[0], loaded[6:8])
    assert torch.equal(param.data_container[1], loaded[2:4])


def test_gguf_embedding_uses_plugin_weight_loader(monkeypatch):
    monkeypatch.setattr(
        vocab_embedding_module, "get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        vocab_embedding_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(gguf_params_module, "get_tensor_model_parallel_rank", lambda: 0)
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
    monkeypatch.setattr(gguf_params_module, "get_tensor_model_parallel_rank", lambda: 0)
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
    assert config_dict["norm_topk_prob"] is True
    assert config.architectures == ["Qwen3MoeForCausalLM"]


def test_gguf_config_parser_selects_kimi_k3_text_model(monkeypatch):
    outer_config = KimiK3Config(
        text_config=KimiLinearConfig(
            hidden_size=7168,
            num_hidden_layers=93,
            architectures=["KimiLinearForCausalLM"],
            quantization_config={"quant_method": "compressed-tensors"},
        )
    )

    monkeypatch.setattr(
        gguf_config_parser_module.HFConfigParser,
        "parse",
        lambda *args, **kwargs: (outer_config.to_dict(), outer_config),
    )

    config_dict, config = GGUFConfigParser().parse(
        "moonshotai/Kimi-K3", trust_remote_code=False
    )

    assert isinstance(config, KimiLinearConfig)
    assert config.architectures == [KIMI_K3_GGUF_TEXT_ARCH]
    assert getattr(config, KIMI_K3_GGUF_TEXT_MARKER)
    assert config.quantization_config is None
    assert "quantization_config" not in config_dict


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


def test_gguf_qkv_shards_are_padded_in_qkv_order(monkeypatch):
    register()
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(gguf_params_module, "get_tensor_model_parallel_rank", lambda: 0)
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
    monkeypatch.setattr(gguf_params_module, "get_tensor_model_parallel_rank", lambda: 0)
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
