# SPDX-License-Identifier: Apache-2.0

import torch
import vllm.engine.arg_utils as arg_utils_module
import vllm.model_executor.layers.quantization as quantization_module
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
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader import get_model_loader
from vllm.transformers_utils.config import get_config_parser

import vllm_gguf_plugin.config_parser as gguf_config_parser_module
import vllm_gguf_plugin.plugin as gguf_plugin_module
import vllm_gguf_plugin.quantization as gguf_quantization
from vllm_gguf_plugin import OOTGGUFConfig, OOTGGUFModelLoader, register
from vllm_gguf_plugin.config_parser import GGUFConfigParser
from vllm_gguf_plugin.gguf_utils import (
    _gguf_sequence_edge,
    resolve_gguf_config_source,
)
from vllm_gguf_plugin.quantization import (
    GGUFUninitializedParameter,
    GGUFWeightParameter,
    GGUFWeightTypeParameter,
)
from vllm_gguf_plugin.weights_adapter.default import (
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
    assert torch.equal(layer.qweight[:4], torch.full((4, 4), 1, dtype=torch.uint8))
    assert torch.equal(layer.qweight[4:6], torch.full((2, 4), 2, dtype=torch.uint8))
    assert torch.equal(layer.qweight[6:12], torch.full((6, 4), 3, dtype=torch.uint8))
    assert torch.equal(layer.qweight[12:], torch.full((8, 4), 4, dtype=torch.uint8))
    assert layer.qweight_type.shard_weight_type == {
        0: WeightType.Q4_0,
        1: WeightType.Q4_0,
        2: WeightType.Q4_0,
        3: WeightType.Q4_1,
    }


def test_gemma4_adapter_transforms_quantized_moe_names():
    adapter = Gemma4GGUFAdapter(PretrainedConfig(model_type="gemma4"))
    weight = torch.empty((2, 3), dtype=torch.uint8)

    transformed = dict(
        adapter.map_weights(
            [
                (
                    "model.language_model.layers.0.mlp.experts.gate_up_proj.qweight",
                    weight,
                ),
                (
                    "model.language_model.layers.0.mlp.experts."
                    "gate_up_proj.qweight_type",
                    torch.tensor(1, dtype=torch.uint8),
                ),
                (
                    "model.language_model.layers.0.mlp.experts.down_proj.qweight",
                    weight,
                ),
                (
                    "model.language_model.layers.0.mlp.experts.down_proj.qweight_type",
                    torch.tensor(1, dtype=torch.uint8),
                ),
            ]
        )
    )

    assert (
        "model.language_model.layers.0.mlp.moe.experts.routed_experts.w13_qweight"
        in transformed
    )
    assert (
        "model.language_model.layers.0.mlp.moe.experts."
        "routed_experts.w13_qweight_type" in transformed
    )
    assert (
        "model.language_model.layers.0.mlp.moe.experts.routed_experts.w2_qweight"
        in transformed
    )
    assert (
        "model.language_model.layers.0.mlp.moe.experts."
        "routed_experts.w2_qweight_type" in transformed
    )


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
    assert mapping["blk.1.ffn_gate.weight"] == ("model.layers.1.mlp.gate_proj.weight")
    assert mapping["blk.1.layer_output_scale.weight"] == "model.layers.1.layer_scalar"


def test_gguf_sequence_edge_accepts_scalar_and_sequence_values():
    assert _gguf_sequence_edge(None, first=True) is None
    assert _gguf_sequence_edge(8, first=True) == 8
    assert _gguf_sequence_edge(8, first=False) == 8
    assert _gguf_sequence_edge([8, 8, 8, 2], first=True) == 8
    assert _gguf_sequence_edge([8, 8, 8, 2], first=False) == 2


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


def test_qwen3_5_mtp_gguf_mappings():
    config = PretrainedConfig(
        model_type="qwen3_5_moe",
        num_hidden_layers=40,
        mtp_num_hidden_layers=1,
    )
    mapping: dict[str, str] = {}
    sideload_params = []

    _add_qwen3_5_mtp_gguf_mappings(config, mapping, sideload_params)

    assert mapping["blk.40.attn_q.weight"] == ("mtp.layers.0.self_attn.q_proj.weight")
    assert mapping["blk.40.attn_k.weight"] == ("mtp.layers.0.self_attn.k_proj.weight")
    assert mapping["blk.40.ffn_gate_inp.weight"] == "mtp.layers.0.mlp.gate.weight"
    assert mapping["blk.40.ffn_gate_inp_shexp.weight"] == (
        "mtp.layers.0.mlp.shared_expert_gate.weight"
    )
    assert mapping["blk.40.ffn_gate_exps.weight"] == (
        "mtp.layers.0.mlp.experts.0.gate_proj.weight"
    )
    assert mapping["blk.40.nextn.eh_proj.weight"] == "mtp.fc.weight"
    assert mapping["blk.40.nextn.shared_head_norm.weight"] == "mtp.norm.weight"
    assert sideload_params[0].fullmatch("mtp.layers.0.mlp.experts.15.gate_proj.weight")


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
