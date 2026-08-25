# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Kimi-K3 GGUF weights adapter.

Builds tiny synthetic GGUF files (llama.cpp "kimi-k3"/"clip" conventions)
and checks the name map plus the tensor transforms (kv_b fusion, A_log
folding, attn-res score, vision q/k re-interleave, expert splitting).
"""

from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch

kimi_k3_configs = pytest.importorskip(
    "vllm.transformers_utils.configs.kimi_k3",
    reason="Kimi-K3 support requires a newer vLLM",
)
KimiK3Config = kimi_k3_configs.KimiK3Config
KimiK3VisionConfig = kimi_k3_configs.KimiK3VisionConfig
KimiLinearConfig = pytest.importorskip(
    "vllm.transformers_utils.configs.kimi_linear"
).KimiLinearConfig

from vllm_gguf_plugin.gguf_files import GGUFModelFiles  # noqa: E402
from vllm_gguf_plugin.weights_adapter import get_weights_adapter  # noqa: E402
from vllm_gguf_plugin.weights_adapter.kimi_k3 import (  # noqa: E402
    KimiK3GGUFAdapter,
    _reinterleave_vision_qk,
)

HIDDEN = 64
KDA_HEADS, KDA_HEAD_DIM = 2, 32
PROJ = KDA_HEADS * KDA_HEAD_DIM
MLA_HEADS = 2
Q_LORA, KV_LORA, QK_NOPE, QK_ROPE, V_HEAD = 96, 64, 32, 16, 32
N_EXP, EXP_FF, LATENT = 4, 32, 64
DENSE_FF = 128
VT_HIDDEN, VT_HEADS, VT_QKV = 32, 2, 64

Q8_0 = gguf.GGMLQuantizationType.Q8_0


def _tiny_hf_config() -> KimiK3Config:
    text_config = KimiLinearConfig(
        model_type="kimi_k3_text",
        vocab_size=512,
        hidden_size=HIDDEN,
        intermediate_size=DENSE_FF,
        num_hidden_layers=2,
        num_attention_heads=MLA_HEADS,
        q_lora_rank=Q_LORA,
        kv_lora_rank=KV_LORA,
        qk_nope_head_dim=QK_NOPE,
        qk_rope_head_dim=QK_ROPE,
        v_head_dim=V_HEAD,
        mla_use_nope=True,
        mla_use_output_gate=True,
        num_experts=N_EXP,
        num_experts_per_token=2,
        num_shared_experts=1,
        moe_intermediate_size=EXP_FF,
        routed_expert_hidden_size=LATENT,
        latent_moe_use_norm=True,
        first_k_dense_replace=1,
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        attn_res_block_size=2,
        linear_attn_config={
            "head_dim": KDA_HEAD_DIM,
            "num_heads": KDA_HEADS,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
            "kda_layers": [1],
            "full_attn_layers": [2],
        },
    )
    return KimiK3Config(
        text_config=text_config,
        vision_config=KimiK3VisionConfig(
            vt_hidden_size=VT_HIDDEN,
            vt_num_attention_heads=VT_HEADS,
            vt_num_hidden_layers=1,
            qkv_hidden_size=VT_QKV,
        ),
    )


def _write_tiny_backbone(path) -> None:
    """Write a 2-layer (KDA+dense, MLA+MoE) kimi-k3 GGUF backbone."""
    rng = np.random.default_rng(0)

    def randn(*shape):
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    writer = gguf.GGUFWriter(path, arch="kimi-k3")
    writer.add_block_count(2)
    writer.add_embedding_length(HIDDEN)
    writer.add_feed_forward_length(DENSE_FF)
    writer.add_token_list([f"tok{i}" for i in range(512)])
    writer.add_token_types([1] * 512)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)

    # layer 0: KDA + dense MLP
    writer.add_tensor("blk.0.attn_norm.weight", randn(HIDDEN))
    for p in ("q", "k", "v"):
        writer.add_tensor(f"blk.0.attn_{p}.weight", randn(PROJ, HIDDEN), raw_dtype=Q8_0)
        writer.add_tensor(f"blk.0.ssm_conv1d_{p}.weight", randn(PROJ, 1, 4))
    writer.add_tensor("blk.0.ssm_g.weight", randn(PROJ, HIDDEN), raw_dtype=Q8_0)
    writer.add_tensor(
        "blk.0.ssm_f_a.weight", randn(KDA_HEAD_DIM, HIDDEN), raw_dtype=Q8_0
    )
    writer.add_tensor("blk.0.ssm_f_b.weight", randn(PROJ, KDA_HEAD_DIM), raw_dtype=Q8_0)
    writer.add_tensor("blk.0.ssm_beta.weight", randn(KDA_HEADS, HIDDEN))
    writer.add_tensor("blk.0.ssm_dt.bias", randn(PROJ))
    writer.add_tensor("blk.0.ssm_a", -np.exp(randn(KDA_HEADS)))
    writer.add_tensor("blk.0.ssm_norm.weight", randn(KDA_HEAD_DIM))
    writer.add_tensor("blk.0.attn_output.weight", randn(HIDDEN, PROJ), raw_dtype=Q8_0)
    writer.add_tensor("blk.0.attn_res_score.weight", randn(HIDDEN))
    writer.add_tensor("blk.0.ffn_norm.weight", randn(HIDDEN))
    writer.add_tensor("blk.0.ffn_res_score.weight", randn(HIDDEN))
    writer.add_tensor("blk.0.ffn_gate.weight", randn(DENSE_FF, HIDDEN), raw_dtype=Q8_0)
    writer.add_tensor("blk.0.ffn_up.weight", randn(DENSE_FF, HIDDEN), raw_dtype=Q8_0)
    writer.add_tensor("blk.0.ffn_down.weight", randn(HIDDEN, DENSE_FF), raw_dtype=Q8_0)

    # layer 1: MLA + LatentMoE
    writer.add_tensor("blk.1.attn_norm.weight", randn(HIDDEN))
    writer.add_tensor("blk.1.attn_q_a.weight", randn(Q_LORA, HIDDEN), raw_dtype=Q8_0)
    writer.add_tensor("blk.1.attn_q_a_norm.weight", randn(Q_LORA))
    writer.add_tensor(
        "blk.1.attn_q_b.weight",
        randn(MLA_HEADS * (QK_NOPE + QK_ROPE), Q_LORA),
        raw_dtype=Q8_0,
    )
    writer.add_tensor(
        "blk.1.attn_kv_a_mqa.weight", randn(KV_LORA + QK_ROPE, HIDDEN), raw_dtype=Q8_0
    )
    writer.add_tensor("blk.1.attn_kv_a_norm.weight", randn(KV_LORA))
    writer.add_tensor(
        "blk.1.attn_k_b.weight", randn(MLA_HEADS, KV_LORA, QK_NOPE), raw_dtype=Q8_0
    )
    writer.add_tensor(
        "blk.1.attn_v_b.weight", randn(MLA_HEADS, V_HEAD, KV_LORA), raw_dtype=Q8_0
    )
    writer.add_tensor(
        "blk.1.attn_gate.weight", randn(MLA_HEADS * V_HEAD, HIDDEN), raw_dtype=Q8_0
    )
    writer.add_tensor(
        "blk.1.attn_output.weight", randn(HIDDEN, MLA_HEADS * V_HEAD), raw_dtype=Q8_0
    )
    writer.add_tensor("blk.1.attn_res_score.weight", randn(HIDDEN))
    writer.add_tensor("blk.1.ffn_norm.weight", randn(HIDDEN))
    writer.add_tensor("blk.1.ffn_res_score.weight", randn(HIDDEN))
    writer.add_tensor("blk.1.ffn_gate_inp.weight", randn(N_EXP, HIDDEN))
    writer.add_tensor("blk.1.exp_probs_b.bias", np.zeros(N_EXP, dtype=np.float32))
    writer.add_tensor(
        "blk.1.ffn_gate_exps.weight", randn(N_EXP, EXP_FF, LATENT), raw_dtype=Q8_0
    )
    writer.add_tensor(
        "blk.1.ffn_up_exps.weight", randn(N_EXP, EXP_FF, LATENT), raw_dtype=Q8_0
    )
    writer.add_tensor(
        "blk.1.ffn_down_exps.weight", randn(N_EXP, LATENT, EXP_FF), raw_dtype=Q8_0
    )
    writer.add_tensor(
        "blk.1.ffn_gate_shexp.weight", randn(EXP_FF, HIDDEN), raw_dtype=Q8_0
    )
    writer.add_tensor(
        "blk.1.ffn_up_shexp.weight", randn(EXP_FF, HIDDEN), raw_dtype=Q8_0
    )
    writer.add_tensor(
        "blk.1.ffn_down_shexp.weight", randn(HIDDEN, EXP_FF), raw_dtype=Q8_0
    )
    writer.add_tensor(
        "blk.1.ffn_routed_down.weight", randn(LATENT, HIDDEN), raw_dtype=Q8_0
    )
    writer.add_tensor("blk.1.ffn_routed_norm.weight", randn(LATENT))
    writer.add_tensor(
        "blk.1.ffn_routed_up.weight", randn(HIDDEN, LATENT), raw_dtype=Q8_0
    )

    writer.add_tensor("token_embd.weight", randn(512, HIDDEN), raw_dtype=Q8_0)
    writer.add_tensor("output.weight", randn(512, HIDDEN), raw_dtype=Q8_0)
    writer.add_tensor("output_norm.weight", randn(HIDDEN))
    writer.add_tensor("output_res_score.weight", randn(HIDDEN))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _write_tiny_mmproj(path) -> None:
    rng = np.random.default_rng(1)

    def randn(*shape):
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    writer = gguf.GGUFWriter(path, arch="clip")
    writer.add_string("clip.projector_type", "kimik3")
    writer.add_bool("clip.has_vision_encoder", True)
    writer.add_uint32("clip.vision.embedding_length", VT_HIDDEN)
    writer.add_uint32("clip.vision.feed_forward_length", VT_HIDDEN * 2)
    writer.add_uint32("clip.vision.block_count", 1)
    writer.add_uint32("clip.vision.attention.head_count", VT_HEADS)
    writer.add_uint32("clip.vision.image_size", 896)
    writer.add_uint32("clip.vision.patch_size", 14)
    writer.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-5)
    writer.add_tensor("v.patch_embd.weight", randn(VT_HIDDEN, 3, 14, 14))
    writer.add_tensor("v.position_embd.weight", randn(64, 64, VT_HIDDEN))
    writer.add_tensor("v.blk.0.ln1.weight", randn(VT_HIDDEN))
    writer.add_tensor("v.blk.0.ln2.weight", randn(VT_HIDDEN))
    writer.add_tensor("v.blk.0.attn_qkv.weight", randn(3 * VT_QKV, VT_HIDDEN))
    writer.add_tensor("v.blk.0.attn_out.weight", randn(VT_HIDDEN, VT_QKV))
    writer.add_tensor("v.blk.0.ffn_up.weight", randn(VT_HIDDEN * 2, VT_HIDDEN))
    writer.add_tensor("v.blk.0.ffn_down.weight", randn(VT_HIDDEN, VT_HIDDEN * 2))
    writer.add_tensor("v.post_ln.weight", randn(VT_HIDDEN))
    writer.add_tensor("mm.1.weight", randn(VT_HIDDEN * 4, VT_HIDDEN * 4))
    writer.add_tensor("mm.2.weight", randn(HIDDEN, VT_HIDDEN * 4))
    writer.add_tensor("mm.post_norm.weight", randn(HIDDEN))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.fixture(scope="module")
def tiny_kimi_k3_gguf(tmp_path_factory):
    backbone = tmp_path_factory.mktemp("gguf") / "tiny-k3-F32.gguf"
    mmproj = backbone.parent / "mmproj-F32.gguf"
    _write_tiny_backbone(backbone)
    _write_tiny_mmproj(mmproj)
    return GGUFModelFiles(backbone=(str(backbone),), mm_proj=str(mmproj))


def _model_config() -> SimpleNamespace:
    return SimpleNamespace(hf_config=_tiny_hf_config(), dtype=torch.bfloat16)


def test_adapter_registration_and_architecture():
    config = _tiny_hf_config()
    adapter = get_weights_adapter(config)
    assert isinstance(adapter, KimiK3GGUFAdapter)
    assert adapter.architecture(config) == "KimiK3ForConditionalGeneration"


def test_kimi_k3_name_map(tiny_kimi_k3_gguf):
    adapter = KimiK3GGUFAdapter()
    name_map = adapter.build_name_map(tiny_kimi_k3_gguf, _model_config())

    expected = {
        # top level
        "token_embd.weight": "language_model.model.embed_tokens.weight",
        "output.weight": "language_model.lm_head.weight",
        "output_norm.weight": "language_model.model.norm.weight",
        "output_res_score.weight": ("language_model.model.output_attn_res_proj.weight"),
        # KDA layer (quantized tensors keep ".weight"; the iterator renames
        # them to ".qweight" at load time)
        "blk.0.attn_norm.weight": (
            "language_model.model.layers.0.input_layernorm.weight"
        ),
        "blk.0.attn_q.weight": "language_model.model.layers.0.self_attn.q_proj.weight",
        "blk.0.ssm_g.weight": "language_model.model.layers.0.self_attn.g_proj.weight",
        "blk.0.ssm_f_a.weight": (
            "language_model.model.layers.0.self_attn.f_a_proj.weight"
        ),
        # ssm_beta stays F32 in the GGUF: it is rerouted through the fused
        # parameter's GGUF loader as an explicit ".qweight" shard
        "blk.0.ssm_beta.weight": (
            "language_model.model.layers.0.self_attn.b_proj.qweight"
        ),
        "blk.0.ssm_dt.bias": "language_model.model.layers.0.self_attn.dt_bias",
        "blk.0.ssm_a": "language_model.model.layers.0.self_attn.A_log",
        "blk.0.ssm_conv1d_q.weight": (
            "language_model.model.layers.0.self_attn.q_conv1d.weight"
        ),
        "blk.0.ssm_norm.weight": (
            "language_model.model.layers.0.self_attn.o_norm.weight"
        ),
        "blk.0.attn_output.weight": (
            "language_model.model.layers.0.self_attn.o_proj.weight"
        ),
        "blk.0.attn_res_score.weight": (
            "language_model.model.layers.0.self_attention_res_proj.weight"
        ),
        "blk.0.ffn_gate.weight": "language_model.model.layers.0.mlp.gate_proj.weight",
        "blk.0.ffn_norm.weight": (
            "language_model.model.layers.0.post_attention_layernorm.weight"
        ),
        # MLA layer
        "blk.1.attn_q_a.weight": (
            "language_model.model.layers.1.self_attn.q_a_proj.weight"
        ),
        "blk.1.attn_kv_a_mqa.weight": (
            "language_model.model.layers.1.self_attn.kv_a_proj_with_mqa.weight"
        ),
        "blk.1.attn_q_a_norm.weight": (
            "language_model.model.layers.1.self_attn.q_a_layernorm.weight"
        ),
        "blk.1.attn_k_b.weight": "language_model.model.layers.1.self_attn.k_b.weight",
        "blk.1.attn_gate.weight": (
            "language_model.model.layers.1.self_attn.g_proj.weight"
        ),
        # LatentMoE
        "blk.1.ffn_gate_inp.weight": (
            "language_model.model.layers.1.block_sparse_moe.gate.weight"
        ),
        "blk.1.exp_probs_b.bias": (
            "language_model.model.layers.1.block_sparse_moe.gate."
            "e_score_correction_bias"
        ),
        "blk.1.ffn_gate_exps.weight": (
            "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight"
        ),
        "blk.1.ffn_up_exps.weight": (
            "language_model.model.layers.1.block_sparse_moe.experts.0.w3.weight"
        ),
        "blk.1.ffn_down_exps.weight": (
            "language_model.model.layers.1.block_sparse_moe.experts.0.w2.weight"
        ),
        "blk.1.ffn_gate_shexp.weight": (
            "language_model.model.layers.1.block_sparse_moe.shared_experts."
            "gate_proj.weight"
        ),
        "blk.1.ffn_routed_down.weight": (
            "language_model.model.layers.1.block_sparse_moe."
            "routed_expert_down_proj.weight"
        ),
        # vision tower + projector
        "v.patch_embd.weight": "vision_tower.patch_embed.proj.weight",
        "v.position_embd.weight": "vision_tower.patch_embed.pos_emb.weight",
        "v.blk.0.ln1.weight": "vision_tower.encoder.blocks.0.norm0.weight",
        "v.blk.0.attn_qkv.weight": "vision_tower.encoder.blocks.0.wqkv.weight",
        "v.blk.0.attn_out.weight": "vision_tower.encoder.blocks.0.wo.weight",
        "v.blk.0.ffn_up.weight": "vision_tower.encoder.blocks.0.mlp.fc0.weight",
        "v.post_ln.weight": "vision_tower.encoder.final_layernorm.weight",
        "mm.1.weight": "mm_projector.linear_1.weight",
        "mm.2.weight": "mm_projector.linear_2.weight",
        "mm.post_norm.weight": "mm_projector.post_norm.weight",
    }
    for gguf_name, expected_name in expected.items():
        assert name_map.get(gguf_name) == expected_name, gguf_name
    assert len(name_map) == 61  # all tensors of the tiny files are mapped


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU dequant kernel")
def test_kv_b_fusion_roundtrip():
    """llama.cpp's k_b/v_b split must reconstruct the HF kv_b_proj exactly."""
    rng = torch.Generator().manual_seed(0)
    hf_kv_b = torch.randn(MLA_HEADS * (QK_NOPE + V_HEAD), KV_LORA, generator=rng)
    # llama.cpp conversion: view(H, Dn+Dv, L) -> split -> k_b transposed
    kv_b = hf_kv_b.view(MLA_HEADS, QK_NOPE + V_HEAD, KV_LORA)
    k_b_hf, v_b_hf = torch.split(kv_b, [QK_NOPE, V_HEAD], dim=1)
    k_q = gguf.quants.quantize(k_b_hf.transpose(1, 2).contiguous().numpy(), Q8_0)
    v_q = gguf.quants.quantize(v_b_hf.contiguous().numpy(), Q8_0)

    adapter = KimiK3GGUFAdapter()
    weights = [
        ("x.self_attn.k_b.qweight_type", torch.tensor(Q8_0)),
        ("x.self_attn.k_b.qweight", torch.from_numpy(k_q)),
        ("x.self_attn.v_b.qweight_type", torch.tensor(Q8_0)),
        ("x.self_attn.v_b.qweight", torch.from_numpy(v_q)),
    ]
    out = list(adapter.transform_weights(iter(weights), _model_config()))
    assert len(out) == 1
    name, fused = out[0]
    assert name == "x.self_attn.kv_b_proj.weight"
    assert fused.shape == hf_kv_b.shape
    # Q8_0 roundtrip precision
    assert (fused.float().cpu() - hf_kv_b).abs().max() < 0.05


def test_kv_b_missing_half_raises():
    adapter = KimiK3GGUFAdapter()
    weights = [
        ("x.self_attn.k_b.qweight_type", torch.tensor(Q8_0)),
        (
            "x.self_attn.k_b.qweight",
            torch.from_numpy(
                gguf.quants.quantize(np.random.randn(2, 64).astype(np.float32), Q8_0)
            ),
        ),
    ]
    with pytest.raises(RuntimeError, match="Incomplete MLA kv_b"):
        list(adapter.transform_weights(iter(weights), _model_config()))


def test_a_log_and_res_score_transforms():
    adapter = KimiK3GGUFAdapter()
    a = -torch.exp(torch.randn(KDA_HEADS))
    res = torch.randn(HIDDEN)
    out = list(
        adapter.transform_weights(
            iter(
                [
                    ("x.self_attn.A_log", a),
                    ("x.self_attention_res_proj.weight", res),
                    ("x.mlp_res_proj.weight", res),
                ]
            ),
            _model_config(),
        )
    )
    assert torch.allclose(out[0][1], torch.log(-a))
    assert out[1][1].shape == (1, HIDDEN)
    assert out[2][1].shape == (1, HIDDEN)


def test_unquantized_fused_shard_gets_type_marker():
    """F32 shards of a fused GGUF parameter must be announced as BF16."""
    adapter = KimiK3GGUFAdapter()
    beta = torch.randn(KDA_HEADS, HIDDEN)
    out = list(
        adapter.transform_weights(
            iter([("x.self_attn.b_proj.qweight", beta)]), _model_config()
        )
    )
    assert out[0][0] == "x.self_attn.b_proj.qweight_type"
    assert int(out[0][1].item()) == int(gguf.GGMLQuantizationType.BF16)
    assert out[1][0] == "x.self_attn.b_proj.qweight"
    assert out[1][1].dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU dequant kernel")
def test_routed_proj_is_dequantized():
    """LatentMoE down/up projections stay unquantized in vLLM."""
    rng = np.random.default_rng(0)
    down = (rng.standard_normal((LATENT, HIDDEN)) * 0.02).astype(np.float32)
    down_q = gguf.quants.quantize(down, Q8_0)
    adapter = KimiK3GGUFAdapter()
    weights = [
        ("x.block_sparse_moe.routed_expert_down_proj.qweight_type", torch.tensor(Q8_0)),
        (
            "x.block_sparse_moe.routed_expert_down_proj.qweight",
            torch.from_numpy(down_q),
        ),
    ]
    out = list(adapter.transform_weights(iter(weights), _model_config()))
    assert len(out) == 1
    name, weight = out[0]
    assert name == "x.block_sparse_moe.routed_expert_down_proj.weight"
    assert weight.shape == down.shape
    assert weight.dtype == torch.bfloat16
    assert (weight.float().cpu() - torch.from_numpy(down)).abs().max() < 0.05


def test_vision_wqkv_reinterleave():
    """The adapter must undo llama.cpp's Q/K rope de-interleave."""
    nh, qkvh, inp = VT_HEADS, VT_QKV, VT_HIDDEN
    w_hf = torch.randn(3 * qkvh, inp)

    def deinterleave(w):
        hd = qkvh // nh
        return (
            w.reshape(nh, hd // 4, 2, 2, inp).permute(0, 2, 1, 3, 4).reshape(qkvh, inp)
        )

    w_gguf = torch.cat(
        [
            deinterleave(w_hf[:qkvh]),
            deinterleave(w_hf[qkvh : 2 * qkvh]),
            w_hf[2 * qkvh :],
        ]
    )
    restored = _reinterleave_vision_qk(w_gguf, nh, qkvh)
    assert torch.equal(restored, w_hf)


def test_stacked_experts_split():
    adapter = KimiK3GGUFAdapter()
    experts = torch.randn(N_EXP, EXP_FF, LATENT)
    out = list(
        adapter.transform_weights(
            iter([("x.block_sparse_moe.experts.0.w1.qweight", experts)]),
            _model_config(),
        )
    )
    # float payloads get a type marker first, then per-expert shards
    assert out[0][0] == "x.block_sparse_moe.experts.0.w1.qweight_type"
    shards = out[1:]
    assert len(shards) == N_EXP
    assert shards[1][0] == "x.block_sparse_moe.experts.1.w1.qweight"
    assert torch.equal(shards[2][1], experts[2].to(torch.bfloat16))
