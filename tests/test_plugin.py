# SPDX-License-Identifier: Apache-2.0

import torch
import vllm.engine.arg_utils as arg_utils_module
import vllm.model_executor.layers.vocab_parallel_embedding as vocab_embedding_module
import vllm.model_executor.parameter as parameter_module
import vllm.transformers_utils.config as config_module
from gguf import GGMLQuantizationType as WeightType
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

import vllm_gguf_plugin.config_parser as gguf_config_parser_module
import vllm_gguf_plugin.quantization as gguf_quantization
from vllm_gguf_plugin import OOTGGUFConfig, OOTGGUFModelLoader, register
from vllm_gguf_plugin.config_parser import GGUFConfigParser
from vllm_gguf_plugin.quantization import (
    GGUFUninitializedParameter,
    GGUFWeightParameter,
    GGUFWeightTypeParameter,
)


def test_register_overrides_gguf_config():
    register()

    quant_config = get_quantization_config("gguf")

    assert quant_config is OOTGGUFConfig


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
