# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end Kimi-K3 GGUF test with a tiny synthetic model.

Builds one random-weight 2-layer Kimi-K3 (1 KDA + dense block, 1 Gated-MLA +
LatentMoE block, 1-layer MoonViT) twice from a single init: an HF-layout
safetensors checkpoint (vLLM's native loader as reference) and a GGUF
backbone + mmproj pair following llama.cpp's "kimi-k3"/"kimik3" conventions.
Greedy generations from both must match.

Only the small tokenizer/processor files of moonshotai/Kimi-K3 are
downloaded (~4 MB), never the model weights.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip(
    "vllm.transformers_utils.configs.kimi_k3",
    reason="Kimi-K3 support requires a newer vLLM",
)

import gguf  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

os.environ["TOKENIZERS_PARALLELISM"] = "true"

KIMI_K3_REPO = "moonshotai/Kimi-K3"
# Small files only: tokenizer, processor and config code — never weights.
REPO_ALLOW_PATTERNS = ["*.json", "*.py", "*.model", "*.jinja"]

HIDDEN = 256
VOCAB = 163840
KDA_HEADS, KDA_HEAD_DIM = 2, 128
PROJ = KDA_HEADS * KDA_HEAD_DIM
MLA_HEADS = 2
Q_LORA, KV_LORA, QK_NOPE, QK_ROPE, V_HEAD = 1536, 512, 128, 64, 128
N_EXP, EXP_FF, SHARED_EXP, LATENT = 4, 64, 1, 128
DENSE_FF = 512
VT_HIDDEN, VT_HEADS, VT_LAYERS, VT_FF, VT_QKV, PATCH = 64, 2, 1, 128, 128, 14
MERGE = 2

MAX_TOKENS = 16
NUM_LOGPROBS = 5


def _randn(rng: np.random.Generator, *shape, scale=0.02):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def _build_hf_state() -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(0)
    sd: dict[str, torch.Tensor] = {}

    def put(name, arr):
        sd[name] = torch.from_numpy(np.ascontiguousarray(arr))

    # layer 0: KDA + dense MLP
    L0 = "language_model.model.layers.0"
    put(f"{L0}.input_layernorm.weight", _randn(rng, HIDDEN) + 1)
    for p in ("q", "k", "v"):
        put(f"{L0}.self_attn.{p}_proj.weight", _randn(rng, PROJ, HIDDEN))
        put(f"{L0}.self_attn.{p}_conv1d.weight", _randn(rng, PROJ, 1, 4))
    put(f"{L0}.self_attn.g_proj.weight", _randn(rng, PROJ, HIDDEN))
    put(f"{L0}.self_attn.f_a_proj.weight", _randn(rng, KDA_HEAD_DIM, HIDDEN))
    put(f"{L0}.self_attn.f_b_proj.weight", _randn(rng, PROJ, KDA_HEAD_DIM))
    put(f"{L0}.self_attn.b_proj.weight", _randn(rng, KDA_HEADS, HIDDEN))
    put(f"{L0}.self_attn.dt_bias", _randn(rng, PROJ))
    put(
        f"{L0}.self_attn.A_log",
        rng.standard_normal(KDA_HEADS).astype(np.float32),
    )
    put(f"{L0}.self_attn.o_norm.weight", _randn(rng, KDA_HEAD_DIM) + 1)
    put(f"{L0}.self_attn.o_proj.weight", _randn(rng, HIDDEN, PROJ))
    put(f"{L0}.self_attention_res_norm.weight", np.ones(HIDDEN, dtype=np.float32))
    put(f"{L0}.self_attention_res_proj.weight", _randn(rng, 1, HIDDEN))
    put(f"{L0}.post_attention_layernorm.weight", _randn(rng, HIDDEN) + 1)
    put(f"{L0}.mlp_res_norm.weight", np.ones(HIDDEN, dtype=np.float32))
    put(f"{L0}.mlp_res_proj.weight", _randn(rng, 1, HIDDEN))
    put(f"{L0}.mlp.gate_proj.weight", _randn(rng, DENSE_FF, HIDDEN))
    put(f"{L0}.mlp.up_proj.weight", _randn(rng, DENSE_FF, HIDDEN))
    put(f"{L0}.mlp.down_proj.weight", _randn(rng, HIDDEN, DENSE_FF))

    # layer 1: Gated MLA + LatentMoE
    L1 = "language_model.model.layers.1"
    put(f"{L1}.input_layernorm.weight", _randn(rng, HIDDEN) + 1)
    put(f"{L1}.self_attn.q_a_proj.weight", _randn(rng, Q_LORA, HIDDEN))
    put(f"{L1}.self_attn.q_a_layernorm.weight", _randn(rng, Q_LORA) + 1)
    put(
        f"{L1}.self_attn.q_b_proj.weight",
        _randn(rng, MLA_HEADS * (QK_NOPE + QK_ROPE), Q_LORA),
    )
    put(
        f"{L1}.self_attn.kv_a_proj_with_mqa.weight",
        _randn(rng, KV_LORA + QK_ROPE, HIDDEN),
    )
    put(f"{L1}.self_attn.kv_a_layernorm.weight", _randn(rng, KV_LORA) + 1)
    put(
        f"{L1}.self_attn.kv_b_proj.weight",
        _randn(rng, MLA_HEADS * (QK_NOPE + V_HEAD), KV_LORA),
    )
    put(f"{L1}.self_attn.g_proj.weight", _randn(rng, MLA_HEADS * V_HEAD, HIDDEN))
    put(f"{L1}.self_attn.o_proj.weight", _randn(rng, HIDDEN, MLA_HEADS * V_HEAD))
    put(f"{L1}.self_attention_res_norm.weight", np.ones(HIDDEN, dtype=np.float32))
    put(f"{L1}.self_attention_res_proj.weight", _randn(rng, 1, HIDDEN))
    put(f"{L1}.post_attention_layernorm.weight", _randn(rng, HIDDEN) + 1)
    put(f"{L1}.mlp_res_norm.weight", np.ones(HIDDEN, dtype=np.float32))
    put(f"{L1}.mlp_res_proj.weight", _randn(rng, 1, HIDDEN))
    put(f"{L1}.block_sparse_moe.gate.weight", _randn(rng, N_EXP, HIDDEN))
    put(
        f"{L1}.block_sparse_moe.gate.e_score_correction_bias",
        np.zeros(N_EXP, np.float32),
    )
    for i in range(N_EXP):
        put(
            f"{L1}.block_sparse_moe.experts.{i}.w1.weight",
            _randn(rng, EXP_FF, LATENT),
        )
        put(
            f"{L1}.block_sparse_moe.experts.{i}.w3.weight",
            _randn(rng, EXP_FF, LATENT),
        )
        put(
            f"{L1}.block_sparse_moe.experts.{i}.w2.weight",
            _randn(rng, LATENT, EXP_FF),
        )
    put(
        f"{L1}.block_sparse_moe.shared_experts.gate_proj.weight",
        _randn(rng, EXP_FF * SHARED_EXP, HIDDEN),
    )
    put(
        f"{L1}.block_sparse_moe.shared_experts.up_proj.weight",
        _randn(rng, EXP_FF * SHARED_EXP, HIDDEN),
    )
    put(
        f"{L1}.block_sparse_moe.shared_experts.down_proj.weight",
        _randn(rng, HIDDEN, EXP_FF * SHARED_EXP),
    )
    put(
        f"{L1}.block_sparse_moe.routed_expert_down_proj.weight",
        _randn(rng, LATENT, HIDDEN),
    )
    put(f"{L1}.block_sparse_moe.routed_expert_norm.weight", _randn(rng, LATENT) + 1)
    put(
        f"{L1}.block_sparse_moe.routed_expert_up_proj.weight",
        _randn(rng, HIDDEN, LATENT),
    )

    put("language_model.model.embed_tokens.weight", _randn(rng, VOCAB, HIDDEN))
    put("language_model.model.norm.weight", _randn(rng, HIDDEN) + 1)
    put(
        "language_model.model.output_attn_res_norm.weight",
        np.ones(HIDDEN, dtype=np.float32),
    )
    put("language_model.model.output_attn_res_proj.weight", _randn(rng, 1, HIDDEN))
    put("language_model.lm_head.weight", _randn(rng, VOCAB, HIDDEN))

    # MoonViT vision tower + patchmergerv2 projector (HF rope layout)
    put("vision_tower.patch_embed.proj.weight", _randn(rng, VT_HIDDEN, 3, PATCH, PATCH))
    put("vision_tower.patch_embed.pos_emb.weight", _randn(rng, 64, 64, VT_HIDDEN))
    for i in range(VT_LAYERS):
        blk = f"vision_tower.encoder.blocks.{i}"
        put(f"{blk}.norm0.weight", _randn(rng, VT_HIDDEN) + 1)
        put(f"{blk}.norm1.weight", _randn(rng, VT_HIDDEN) + 1)
        put(f"{blk}.wqkv.weight", _randn(rng, 3 * VT_QKV, VT_HIDDEN))
        put(f"{blk}.wo.weight", _randn(rng, VT_HIDDEN, VT_QKV))
        put(f"{blk}.mlp.fc0.weight", _randn(rng, VT_FF, VT_HIDDEN))
        put(f"{blk}.mlp.fc1.weight", _randn(rng, VT_HIDDEN, VT_FF))
    put("vision_tower.encoder.final_layernorm.weight", _randn(rng, VT_HIDDEN) + 1)
    put(
        "mm_projector.proj.0.weight",
        _randn(rng, VT_HIDDEN * MERGE * MERGE, VT_HIDDEN * MERGE * MERGE),
    )
    put("mm_projector.proj.2.weight", _randn(rng, HIDDEN, VT_HIDDEN * MERGE * MERGE))
    put("mm_projector.post_norm.weight", _randn(rng, HIDDEN) + 1)
    return sd


def _write_config(snapshot: Path, out: Path) -> None:
    config = json.loads((snapshot / "config.json").read_text())
    text_config = config["text_config"]
    text_config.update(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=DENSE_FF,
        num_hidden_layers=2,
        num_attention_heads=MLA_HEADS,
        max_position_embeddings=4096,
        q_lora_rank=Q_LORA,
        kv_lora_rank=KV_LORA,
        qk_nope_head_dim=QK_NOPE,
        qk_rope_head_dim=QK_ROPE,
        v_head_dim=V_HEAD,
        mla_use_nope=True,
        mla_use_output_gate=True,
        num_experts=N_EXP,
        num_experts_per_token=2,
        num_shared_experts=SHARED_EXP,
        moe_intermediate_size=EXP_FF,
        routed_expert_hidden_size=LATENT,
        latent_moe_use_norm=True,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        use_grouped_topk=True,
        num_expert_group=1,
        topk_group=1,
        topk_method="noaux_tc",
        routed_scaling_factor=1.0,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        attn_res_block_size=2,
        num_nextn_predict_layers=0,
        tie_word_embeddings=False,
    )
    text_config.pop("quantization_config", None)
    text_config["linear_attn_config"] = {
        "head_dim": KDA_HEAD_DIM,
        "num_heads": KDA_HEADS,
        "short_conv_kernel_size": 4,
        "use_full_rank_gate": True,
        "gate_lower_bound": -5.0,
        "kda_layers": [1],
        "full_attn_layers": [2],
    }
    config["vision_config"].update(
        vt_hidden_size=VT_HIDDEN,
        vt_num_attention_heads=VT_HEADS,
        vt_num_hidden_layers=VT_LAYERS,
        vt_intermediate_size=VT_FF,
        qkv_hidden_size=VT_QKV,
        patch_size=PATCH,
        merge_kernel_size=[MERGE, MERGE],
    )
    (out / "config.json").write_text(json.dumps(config, indent=2))


def _deinterleave(w: np.ndarray, n_head: int, qkv_dim: int) -> np.ndarray:
    """llama.cpp's Q/K rope de-interleave applied by the mmproj conversion."""
    head_dim = qkv_dim // n_head
    return (
        w.reshape(n_head, head_dim // 4, 2, 2, w.shape[-1])
        .transpose(0, 2, 1, 3, 4)
        .reshape(w.shape)
    )


def _write_gguf(sd: dict[str, torch.Tensor], path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

    def t(name):
        return sd[name].numpy().astype(np.float32)

    L0 = "language_model.model.layers.0"
    L1 = "language_model.model.layers.1"
    backbone = gguf.GGUFWriter(path / "tiny-k3-F32.gguf", arch="kimi-k3")
    backbone.add_block_count(2)
    backbone.add_context_length(4096)
    backbone.add_embedding_length(HIDDEN)
    backbone.add_feed_forward_length(DENSE_FF)
    backbone.add_token_list([f"tok{i}" for i in range(VOCAB)])
    backbone.add_token_types([1] * VOCAB)
    backbone.add_bos_token_id(163584)
    backbone.add_eos_token_id(163586)

    backbone.add_tensor("blk.0.attn_norm.weight", t(f"{L0}.input_layernorm.weight"))
    for p in ("q", "k", "v"):
        backbone.add_tensor(
            f"blk.0.attn_{p}.weight", t(f"{L0}.self_attn.{p}_proj.weight")
        )
        backbone.add_tensor(
            f"blk.0.ssm_conv1d_{p}.weight", t(f"{L0}.self_attn.{p}_conv1d.weight")
        )
    backbone.add_tensor("blk.0.ssm_g.weight", t(f"{L0}.self_attn.g_proj.weight"))
    backbone.add_tensor("blk.0.ssm_f_a.weight", t(f"{L0}.self_attn.f_a_proj.weight"))
    backbone.add_tensor("blk.0.ssm_f_b.weight", t(f"{L0}.self_attn.f_b_proj.weight"))
    backbone.add_tensor("blk.0.ssm_beta.weight", t(f"{L0}.self_attn.b_proj.weight"))
    backbone.add_tensor("blk.0.ssm_dt.bias", t(f"{L0}.self_attn.dt_bias"))
    backbone.add_tensor(
        "blk.0.ssm_a", (-np.exp(t(f"{L0}.self_attn.A_log"))).astype(np.float32)
    )
    backbone.add_tensor("blk.0.ssm_norm.weight", t(f"{L0}.self_attn.o_norm.weight"))
    backbone.add_tensor("blk.0.attn_output.weight", t(f"{L0}.self_attn.o_proj.weight"))
    backbone.add_tensor(
        "blk.0.attn_res_score.weight",
        t(f"{L0}.self_attention_res_norm.weight")
        * t(f"{L0}.self_attention_res_proj.weight").reshape(-1),
    )
    backbone.add_tensor(
        "blk.0.ffn_norm.weight", t(f"{L0}.post_attention_layernorm.weight")
    )
    backbone.add_tensor(
        "blk.0.ffn_res_score.weight",
        t(f"{L0}.mlp_res_norm.weight") * t(f"{L0}.mlp_res_proj.weight").reshape(-1),
    )
    backbone.add_tensor("blk.0.ffn_gate.weight", t(f"{L0}.mlp.gate_proj.weight"))
    backbone.add_tensor("blk.0.ffn_up.weight", t(f"{L0}.mlp.up_proj.weight"))
    backbone.add_tensor("blk.0.ffn_down.weight", t(f"{L0}.mlp.down_proj.weight"))

    backbone.add_tensor("blk.1.attn_norm.weight", t(f"{L1}.input_layernorm.weight"))
    backbone.add_tensor("blk.1.attn_q_a.weight", t(f"{L1}.self_attn.q_a_proj.weight"))
    backbone.add_tensor(
        "blk.1.attn_q_a_norm.weight", t(f"{L1}.self_attn.q_a_layernorm.weight")
    )
    backbone.add_tensor("blk.1.attn_q_b.weight", t(f"{L1}.self_attn.q_b_proj.weight"))
    backbone.add_tensor(
        "blk.1.attn_kv_a_mqa.weight", t(f"{L1}.self_attn.kv_a_proj_with_mqa.weight")
    )
    backbone.add_tensor(
        "blk.1.attn_kv_a_norm.weight", t(f"{L1}.self_attn.kv_a_layernorm.weight")
    )
    kv_b = t(f"{L1}.self_attn.kv_b_proj.weight").reshape(
        MLA_HEADS, QK_NOPE + V_HEAD, KV_LORA
    )
    k_b, v_b = kv_b[:, :QK_NOPE], kv_b[:, QK_NOPE:]
    backbone.add_tensor(
        "blk.1.attn_k_b.weight", np.ascontiguousarray(k_b.transpose(0, 2, 1))
    )
    backbone.add_tensor("blk.1.attn_v_b.weight", np.ascontiguousarray(v_b))
    backbone.add_tensor("blk.1.attn_gate.weight", t(f"{L1}.self_attn.g_proj.weight"))
    backbone.add_tensor("blk.1.attn_output.weight", t(f"{L1}.self_attn.o_proj.weight"))
    backbone.add_tensor(
        "blk.1.attn_res_score.weight",
        t(f"{L1}.self_attention_res_norm.weight")
        * t(f"{L1}.self_attention_res_proj.weight").reshape(-1),
    )
    backbone.add_tensor(
        "blk.1.ffn_norm.weight", t(f"{L1}.post_attention_layernorm.weight")
    )
    backbone.add_tensor(
        "blk.1.ffn_res_score.weight",
        t(f"{L1}.mlp_res_norm.weight") * t(f"{L1}.mlp_res_proj.weight").reshape(-1),
    )
    backbone.add_tensor(
        "blk.1.ffn_gate_inp.weight", t(f"{L1}.block_sparse_moe.gate.weight")
    )
    backbone.add_tensor(
        "blk.1.exp_probs_b.bias",
        t(f"{L1}.block_sparse_moe.gate.e_score_correction_bias"),
    )
    for gguf_name, hf_name in (
        ("ffn_gate_exps", "w1"),
        ("ffn_up_exps", "w3"),
        ("ffn_down_exps", "w2"),
    ):
        stacked = np.stack(
            [
                t(f"{L1}.block_sparse_moe.experts.{i}.{hf_name}.weight")
                for i in range(N_EXP)
            ]
        )
        backbone.add_tensor(f"blk.1.{gguf_name}.weight", stacked)
    backbone.add_tensor(
        "blk.1.ffn_gate_shexp.weight",
        t(f"{L1}.block_sparse_moe.shared_experts.gate_proj.weight"),
    )
    backbone.add_tensor(
        "blk.1.ffn_up_shexp.weight",
        t(f"{L1}.block_sparse_moe.shared_experts.up_proj.weight"),
    )
    backbone.add_tensor(
        "blk.1.ffn_down_shexp.weight",
        t(f"{L1}.block_sparse_moe.shared_experts.down_proj.weight"),
    )
    backbone.add_tensor(
        "blk.1.ffn_routed_down.weight",
        t(f"{L1}.block_sparse_moe.routed_expert_down_proj.weight"),
    )
    backbone.add_tensor(
        "blk.1.ffn_routed_norm.weight",
        t(f"{L1}.block_sparse_moe.routed_expert_norm.weight"),
    )
    backbone.add_tensor(
        "blk.1.ffn_routed_up.weight",
        t(f"{L1}.block_sparse_moe.routed_expert_up_proj.weight"),
    )

    backbone.add_tensor(
        "token_embd.weight", t("language_model.model.embed_tokens.weight")
    )
    backbone.add_tensor("output.weight", t("language_model.lm_head.weight"))
    backbone.add_tensor("output_norm.weight", t("language_model.model.norm.weight"))
    backbone.add_tensor(
        "output_res_score.weight",
        t("language_model.model.output_attn_res_norm.weight")
        * t("language_model.model.output_attn_res_proj.weight").reshape(-1),
    )
    backbone.write_header_to_file()
    backbone.write_kv_data_to_file()
    backbone.write_tensors_to_file()
    backbone.close()

    mmproj = gguf.GGUFWriter(path / "mmproj-F32.gguf", arch="clip")
    mmproj.add_file_type(gguf.LlamaFileType.ALL_F32)
    mmproj.add_string("clip.projector_type", "kimik3")
    mmproj.add_bool("clip.has_vision_encoder", True)
    mmproj.add_uint32("clip.vision.embedding_length", VT_HIDDEN)
    mmproj.add_uint32("clip.vision.feed_forward_length", VT_FF)
    mmproj.add_uint32("clip.vision.block_count", VT_LAYERS)
    mmproj.add_uint32("clip.vision.attention.head_count", VT_HEADS)
    mmproj.add_uint32("clip.vision.image_size", 896)
    mmproj.add_uint32("clip.vision.patch_size", PATCH)
    mmproj.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-5)
    mmproj.add_uint32("clip.vision.projection_dim", HIDDEN)
    mmproj.add_uint32("clip.vision.projector.scale_factor", MERGE)
    mmproj.add_tensor("v.patch_embd.weight", t("vision_tower.patch_embed.proj.weight"))
    mmproj.add_tensor(
        "v.position_embd.weight", t("vision_tower.patch_embed.pos_emb.weight")
    )
    for i in range(VT_LAYERS):
        blk = f"vision_tower.encoder.blocks.{i}"
        mmproj.add_tensor(f"v.blk.{i}.ln1.weight", t(f"{blk}.norm0.weight"))
        mmproj.add_tensor(f"v.blk.{i}.ln2.weight", t(f"{blk}.norm1.weight"))
        wqkv = t(f"{blk}.wqkv.weight")
        wq = _deinterleave(wqkv[:VT_QKV], VT_HEADS, VT_QKV)
        wk = _deinterleave(wqkv[VT_QKV : 2 * VT_QKV], VT_HEADS, VT_QKV)
        mmproj.add_tensor(
            f"v.blk.{i}.attn_qkv.weight",
            np.concatenate([wq, wk, wqkv[2 * VT_QKV :]], 0),
        )
        mmproj.add_tensor(f"v.blk.{i}.attn_out.weight", t(f"{blk}.wo.weight"))
        mmproj.add_tensor(f"v.blk.{i}.ffn_up.weight", t(f"{blk}.mlp.fc0.weight"))
        mmproj.add_tensor(f"v.blk.{i}.ffn_down.weight", t(f"{blk}.mlp.fc1.weight"))
    mmproj.add_tensor(
        "v.post_ln.weight", t("vision_tower.encoder.final_layernorm.weight")
    )
    mmproj.add_tensor("mm.1.weight", t("mm_projector.proj.0.weight"))
    mmproj.add_tensor("mm.2.weight", t("mm_projector.proj.2.weight"))
    mmproj.add_tensor("mm.post_norm.weight", t("mm_projector.post_norm.weight"))
    mmproj.write_header_to_file()
    mmproj.write_kv_data_to_file()
    mmproj.write_tensors_to_file()
    mmproj.close()


@pytest.fixture(scope="module")
def tiny_kimi_k3_gguf_pair(tmp_path_factory):
    """A tiny random-weight Kimi-K3 in both HF safetensors and GGUF form."""
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(KIMI_K3_REPO, allow_patterns=REPO_ALLOW_PATTERNS))
    root = tmp_path_factory.mktemp("kimi_k3")
    hf_dir = root / "hf"
    gguf_dir = root / "gguf"
    hf_dir.mkdir()
    for f in snapshot.iterdir():
        if f.suffix in (".py", ".model", ".jinja") or f.name in (
            "tokenizer_config.json",
            "preprocessor_config.json",
            "generation_config.json",
        ):
            (hf_dir / f.name).write_bytes(f.read_bytes())

    state = _build_hf_state()
    _write_config(snapshot, hf_dir)
    # A_log/dt_bias are fp32 parameters in vLLM; keep them exact so the
    # reference checkpoint does not introduce bf16 rounding there.
    keep_fp32 = ("A_log", "dt_bias")
    save_file(
        {
            k: (v if k.endswith(keep_fp32) else v.to(torch.bfloat16))
            for k, v in state.items()
        },
        hf_dir / "model.safetensors",
    )
    _write_gguf(state, gguf_dir)
    return hf_dir, gguf_dir


def _generate(llm: "object", prompt, image=None) -> tuple:
    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=0.0, max_tokens=MAX_TOKENS, logprobs=NUM_LOGPROBS
    )
    if image is None:
        outputs = llm.generate([prompt], sampling)
    else:
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_pil", "image_pil": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        ]
        outputs = llm.chat(messages, sampling)
    sample = outputs[0].outputs[0]
    logprobs = [
        {tok_id: lp.logprob for tok_id, lp in lp_dict.items()} if lp_dict else None
        for lp_dict in (sample.logprobs or [])
    ]
    return list(sample.token_ids), sample.text, logprobs


def _run_vllm(model: str, tokenizer: str) -> dict[str, tuple]:
    import gc

    from PIL import Image
    from vllm import LLM

    llm = LLM(
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        enforce_eager=True,
        dtype="bfloat16",
        max_model_len=512,
        gpu_memory_utilization=0.3,
    )
    try:
        image = Image.new("RGB", (224, 224), (200, 30, 30))
        return {
            "text": _generate(llm, "The capital of France is"),
            "mm": _generate(llm, "What color is this?", image),
        }
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for Kimi-K3 GGUF e2e test.",
)
def test_kimi_k3_gguf_vs_hf(tiny_kimi_k3_gguf_pair):
    from tests.test_multimodal_gguf import check_logprobs_close

    hf_dir, gguf_dir = tiny_kimi_k3_gguf_pair
    reference = _run_vllm(str(hf_dir), str(hf_dir))
    gguf = _run_vllm(f"{gguf_dir}:F32", str(hf_dir))
    for case in ("text", "mm"):
        # Same tolerance semantics as the other multimodal GGUF tests:
        # random-weight models are chaotic, so a token divergence is only a
        # warning as long as both tokens stay in each other's top logprobs.
        check_logprobs_close(
            outputs_0_lst=[reference[case]],
            outputs_1_lst=[gguf[case]],
            name_0="hf",
            name_1="gguf",
        )
