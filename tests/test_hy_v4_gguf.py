# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the HY V4 GGUF weights adapter.

Builds a synthetic two-layer hyv4 GGUF (one dense layer, one MoE layer) and
checks the GGUF->vLLM name mapping, the kv_b_proj merge from the split
k_b/v_b tensors, the indexer wk dequantization, and the expert split.
"""

from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch
from gguf.quants import quantize

from vllm_gguf_plugin.gguf_files import GGUFModelFiles
from vllm_gguf_plugin.weight_utils import gguf_quant_weights_iterator_multi
from vllm_gguf_plugin.weights_adapter import get_weights_adapter
from vllm_gguf_plugin.weights_adapter.hy_v4 import HYV4GGUFAdapter, merge_kv_b_proj

# Tiny dimensions, consistent with the HY V4 layout semantics.
HIDDEN = 256
N_HEAD = 2
Q_LORA = 128
KV_LORA = 256
QK_NOPE = 64
QK_ROPE = 32
V_HEAD = 64
INDEX_N_HEADS = 2
INDEX_HEAD_DIM = 128
N_EXPERTS = 4
MOE_INTER = 64
DENSE_INTER = 512
HC_MULT = 4
VOCAB = 32


def _f32(shape) -> np.ndarray:
    return np.random.default_rng(0).standard_normal(shape, dtype=np.float32)


def make_ternary(shape, d=0.25, seed=1) -> np.ndarray:
    """Random STQ1_0-encodable data: per 256-block, lanes in {-d, 0, +d}.

    Patterns are drawn in group space and scattered to the stride-16 layout:
    group g (chunk = g//16, gloc = g%16) covers weights chunk*64 + gloc + p*16.
    """
    from vllm_gguf_plugin.gguf_stq import STQ1_0_CODEBOOK

    *lead, n = shape
    cb = np.array(STQ1_0_CODEBOOK, dtype=np.uint8)
    idx = np.random.default_rng(seed).integers(0, 32, size=(*lead, n // 256, 64))
    qpack = cb[idx]
    lanes = (qpack[..., None] >> (2 * np.arange(4))) & 3
    vals = lanes.astype(np.float32) - 1  # [..., n/256, group(64), p(4)]
    # [..., nb, chunk, gloc, p] -> [..., nb, chunk, p, gloc] -> [..., nb, 256]
    vals = vals.reshape(*lead, n // 256, 4, 16, 4).swapaxes(-1, -2)
    return (vals.reshape(*lead, n) * d).astype(np.float32)


def encode_stq1_0(data: np.ndarray) -> np.ndarray:
    """Encode [..., n] ternary-grid floats into raw STQ1_0 blocks."""
    from vllm_gguf_plugin.gguf_stq import STQ1_0_CODEBOOK

    *lead, n = data.shape
    x = data.reshape(-1, 256)
    d = np.abs(x).max(axis=1)
    # Gather group lanes: [nb, chunk, p, gloc] -> [nb, chunk, gloc, p] ->
    # [nb, group(64), p(4)]
    v = x.reshape(-1, 4, 4, 16).transpose(0, 1, 3, 2).reshape(-1, 64, 4)
    lane_code = (np.sign(v / d[:, None, None]).astype(np.int64) + 1).astype(np.uint8)
    qpack = (
        lane_code[..., 0]
        | (lane_code[..., 1] << 2)
        | (lane_code[..., 2] << 4)
        | (lane_code[..., 3] << 6)
    )
    rev = np.full(256, 0xFF, dtype=np.uint8)
    rev[np.array(STQ1_0_CODEBOOK, dtype=np.uint8)] = np.arange(32, dtype=np.uint8)
    idx = rev[qpack]
    assert (idx != 0xFF).all(), "data is not on the STQ1_0 ternary grid"
    slot = idx & 0xF
    sign_bit = idx >> 4
    slot_pairs = slot.reshape(-1, 32, 2)
    qs = (slot_pairs[..., 0] | (slot_pairs[..., 1] << 4)).astype(np.uint8)
    sign = np.packbits(sign_bit.reshape(-1, 64), axis=-1, bitorder="little")
    d16 = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    block = np.concatenate([qs, sign, d16], axis=1).astype(np.uint8)
    return block.reshape(*lead, (n // 256) * 42)


def _write_tiny_hy_v4_gguf(path, stq_experts: bool = False) -> dict[str, np.ndarray]:
    """Write a two-layer hyv4 GGUF; returns the original tensor data."""

    def expert_data(shape):
        if stq_experts:
            return make_ternary(shape)
        return _f32(shape)

    def expert_qtype():
        if stq_experts:
            from vllm_gguf_plugin.gguf_stq import get_stq1_0_type

            return get_stq1_0_type()
        return None

    tensors: dict[str, tuple[np.ndarray, gguf.GGMLQuantizationType | None]] = {}

    def add(name: str, data: np.ndarray, qtype=None):
        tensors[name] = (data, qtype)

    add("token_embd.weight", _f32((VOCAB, HIDDEN)))
    add("output_norm.weight", _f32((HIDDEN,)))
    add("output.weight", _f32((VOCAB, HIDDEN)))
    add("output_hc_fn.weight", _f32((HC_MULT, HC_MULT * HIDDEN)))
    add("output_hc_base.weight", _f32((HC_MULT,)))
    add("output_hc_scale.weight", _f32((1,)))

    k_b_orig = _f32((N_HEAD, QK_NOPE, KV_LORA))
    v_b_orig = _f32((N_HEAD, V_HEAD, KV_LORA))
    wk_orig = _f32((INDEX_HEAD_DIM, HIDDEN))

    for layer in range(2):
        prefix = f"blk.{layer}."
        add(prefix + "attn_norm.weight", _f32((HIDDEN,)))
        add(prefix + "attn_q_a.weight", _f32((Q_LORA, HIDDEN)))
        add(prefix + "attn_q_a_norm.weight", _f32((Q_LORA,)))
        add(prefix + "attn_q_b.weight", _f32((N_HEAD * (QK_NOPE + QK_ROPE), Q_LORA)))
        add(prefix + "attn_kv_a_mqa.weight", _f32((KV_LORA + QK_ROPE, HIDDEN)))
        add(prefix + "attn_kv_a_norm.weight", _f32((KV_LORA,)))
        # GGUF stores k_b transposed: [n_head, kv_lora, qk_nope]
        add(
            prefix + "attn_k_b.weight",
            np.ascontiguousarray(k_b_orig.transpose(0, 2, 1)),
            gguf.GGMLQuantizationType.Q8_0,
        )
        add(prefix + "attn_v_b.weight", v_b_orig, gguf.GGMLQuantizationType.Q8_0)
        add(prefix + "attn_output.weight", _f32((HIDDEN, N_HEAD * V_HEAD)))
        add(prefix + "attn_gate.weight", _f32((N_HEAD * V_HEAD, HIDDEN)))
        add(prefix + "attn_sinks.weight", _f32((N_HEAD,)))
        add(
            prefix + "indexer.attn_q_b.weight",
            _f32((INDEX_N_HEADS * INDEX_HEAD_DIM, Q_LORA)),
            gguf.GGMLQuantizationType.Q8_0,
        )
        add(
            prefix + "indexer.attn_k.weight",
            wk_orig,
            gguf.GGMLQuantizationType.Q8_0,
        )
        add(prefix + "indexer.k_norm.weight", _f32((INDEX_HEAD_DIM,)))
        add(prefix + "indexer.k_norm.bias", _f32((INDEX_HEAD_DIM,)))
        add(prefix + "indexer.proj.weight", _f32((INDEX_N_HEADS, HIDDEN)))
        add(prefix + "hc_attn_fn.weight", _f32((2 * HC_MULT, HC_MULT * HIDDEN)))
        add(prefix + "hc_attn_base.weight", _f32((2 * HC_MULT,)))
        add(prefix + "hc_attn_scale.weight", _f32((2,)))
        add(prefix + "hc_ffn_fn.weight", _f32((2 * HC_MULT, HC_MULT * HIDDEN)))
        add(prefix + "hc_ffn_base.weight", _f32((2 * HC_MULT,)))
        add(prefix + "hc_ffn_scale.weight", _f32((2,)))
        add(prefix + "ffn_norm.weight", _f32((HIDDEN,)))
        if layer == 0:
            add(prefix + "ffn_gate.weight", _f32((DENSE_INTER, HIDDEN)))
            add(prefix + "ffn_up.weight", _f32((DENSE_INTER, HIDDEN)))
            add(prefix + "ffn_down.weight", _f32((HIDDEN, DENSE_INTER)))
        else:
            add(prefix + "ffn_gate_inp.weight", _f32((N_EXPERTS, HIDDEN)))
            add(prefix + "exp_probs_b.bias", _f32((N_EXPERTS,)))
            add(
                prefix + "ffn_gate_exps.weight",
                expert_data((N_EXPERTS, MOE_INTER, HIDDEN)),
                expert_qtype(),
            )
            add(
                prefix + "ffn_up_exps.weight",
                expert_data((N_EXPERTS, MOE_INTER, HIDDEN)),
                expert_qtype(),
            )
            # down_exps stays unquantized here: its innermost dim
            # (MOE_INTER) is not a multiple of STQ1_0's 256-wide block, and
            # the real checkpoint quantizes it as IQ2_XXS instead.
            add(prefix + "ffn_down_exps.weight", _f32((N_EXPERTS, HIDDEN, MOE_INTER)))
            add(prefix + "ffn_gate_shexp.weight", _f32((MOE_INTER, HIDDEN)))
            add(prefix + "ffn_up_shexp.weight", _f32((MOE_INTER, HIDDEN)))
            add(prefix + "ffn_down_shexp.weight", _f32((HIDDEN, MOE_INTER)))

    writer = gguf.GGUFWriter(path, "hyv4")
    originals: dict[str, np.ndarray] = {}
    for name, (data, qtype) in tensors.items():
        originals[name] = data
        if qtype is None:
            writer.add_tensor(name, data)
        elif int(qtype) == 43:  # STQ1_0
            writer.add_tensor(name, encode_stq1_0(data), raw_dtype=qtype)
        else:
            writer.add_tensor(name, quantize(data, qtype), raw_dtype=qtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    originals["__wk_orig__"] = wk_orig
    return originals


@pytest.fixture(scope="module")
def tiny_hy_v4(tmp_path_factory):
    path = tmp_path_factory.mktemp("gguf") / "tiny-hy-v4.gguf"
    originals = _write_tiny_hy_v4_gguf(str(path))
    files = GGUFModelFiles(backbone=(str(path),))
    adapter = HYV4GGUFAdapter()
    name_map = adapter.build_name_map(files, None)
    model_config = SimpleNamespace(dtype=torch.bfloat16)
    weights = gguf_quant_weights_iterator_multi(list(files.all_files), name_map)
    loaded = dict(adapter.transform_weights(weights, model_config))
    return originals, name_map, loaded


def test_matches():
    assert HYV4GGUFAdapter.matches(SimpleNamespace(model_type="hy_v4"))
    assert not HYV4GGUFAdapter.matches(SimpleNamespace(model_type="qwen3"))
    adapter = get_weights_adapter(SimpleNamespace(model_type="hy_v4"))
    assert isinstance(adapter, HYV4GGUFAdapter)
    assert HYV4GGUFAdapter.architecture(None) == "HYV4ForCausalLM"


def test_name_map(tiny_hy_v4):
    _, name_map, _ = tiny_hy_v4
    expected = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
        "output_hc_fn.weight": "model.hc_head.hc_head_fn",
        "output_hc_base.weight": "model.hc_head.hc_head_base",
        "output_hc_scale.weight": "model.hc_head.hc_head_scale",
        "blk.0.attn_norm.weight": "model.layers.0.input_layernorm.weight",
        "blk.0.attn_q_a.weight": "model.layers.0.self_attn.q_a_proj.weight",
        "blk.0.attn_q_b.weight": "model.layers.0.self_attn.q_b_proj.weight",
        "blk.0.attn_kv_a_mqa.weight": (
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"
        ),
        "blk.0.attn_output.weight": "model.layers.0.self_attn.o_proj.weight",
        "blk.0.attn_gate.weight": "model.layers.0.self_attn.linear_gate.weight",
        "blk.0.attn_sinks.weight": "model.layers.0.self_attn.learnable_sink_param",
        "blk.0.indexer.attn_q_b.weight": (
            "model.layers.0.self_attn.indexer.wq_b.weight"
        ),
        "blk.0.indexer.attn_k.weight": "model.layers.0.self_attn.indexer.wk.weight",
        "blk.0.indexer.k_norm.weight": (
            "model.layers.0.self_attn.indexer.k_norm.weight"
        ),
        "blk.0.indexer.proj.weight": (
            "model.layers.0.self_attn.indexer.weights_proj.weight"
        ),
        "blk.0.hc_attn_fn.weight": "model.layers.0.hc_attn_layer.hc_pre.hc_fn",
        "blk.0.hc_ffn_fn.weight": "model.layers.0.hc_mlp_layer.hc_pre.hc_fn",
        "blk.0.ffn_gate.weight": "model.layers.0.mlp.gate_proj.weight",
        "blk.0.ffn_down.weight": "model.layers.0.mlp.down_proj.weight",
        "blk.1.ffn_gate_inp.weight": "model.layers.1.mlp.gate.weight",
        "blk.1.exp_probs_b.bias": "model.layers.1.mlp.gate.e_score_correction_bias",
        "blk.1.ffn_gate_exps.weight": "model.layers.1.mlp.experts.0.gate_proj.weight",
        "blk.1.ffn_down_shexp.weight": (
            "model.layers.1.mlp.shared_experts.down_proj.weight"
        ),
    }
    for gguf_name, hf_name in expected.items():
        assert name_map[gguf_name] == hf_name, gguf_name
    # Every tensor must be mapped (no identity passthrough).
    unmapped = [gguf for gguf, hf in name_map.items() if gguf == hf]
    assert not unmapped


def test_kv_b_proj_merge(tiny_hy_v4):
    originals, _, loaded = tiny_hy_v4
    for layer in range(2):
        name = f"model.layers.{layer}.self_attn.kv_b_proj.weight"
        assert name in loaded
        merged = loaded[name]
        assert merged.dtype == torch.bfloat16
        assert merged.shape == (N_HEAD * (QK_NOPE + V_HEAD), KV_LORA)
        expected = merge_kv_b_proj(
            torch.from_numpy(originals[f"blk.{layer}.attn_k_b.weight"]),
            torch.from_numpy(originals[f"blk.{layer}.attn_v_b.weight"]),
        )
        # Q8_0 quantization noise: blockwise error is bounded by ~scale/2.
        assert (merged - expected).abs().mean() < 0.02


def test_indexer_wk_dequantized(tiny_hy_v4):
    originals, _, loaded = tiny_hy_v4
    name = "model.layers.0.self_attn.indexer.wk.weight"
    assert name in loaded
    assert loaded[name].dtype == torch.bfloat16
    assert loaded[name].shape == (INDEX_HEAD_DIM, HIDDEN)
    error = (loaded[name] - torch.from_numpy(originals["__wk_orig__"])).abs().mean()
    assert error < 0.02


def test_expert_split(tiny_hy_v4):
    originals, _, loaded = tiny_hy_v4
    gate = originals["blk.1.ffn_gate_exps.weight"]
    for expert_id in range(N_EXPERTS):
        name = f"model.layers.1.mlp.experts.{expert_id}.gate_proj.weight"
        assert name in loaded
        torch.testing.assert_close(loaded[name], torch.from_numpy(gate[expert_id]))


def test_shared_expert_and_gate_names(tiny_hy_v4):
    _, _, loaded = tiny_hy_v4
    assert "model.layers.1.mlp.gate.weight" in loaded
    assert "model.layers.1.mlp.gate.e_score_correction_bias" in loaded
    assert "model.layers.1.mlp.shared_experts.gate_proj.weight" in loaded
    assert "model.layers.1.mlp.shared_experts.up_proj.weight" in loaded
    assert "model.layers.1.mlp.shared_experts.down_proj.weight" in loaded


def test_bare_parameter_names(tiny_hy_v4):
    _, _, loaded = tiny_hy_v4
    # hc_fn and the sink parameter map to bare module paths (no .weight
    # suffix); the model's load_weights completes the parameter names.
    assert "model.layers.0.hc_attn_layer.hc_pre.hc_fn" in loaded
    assert "model.layers.0.hc_attn_layer.hc_pre.hc_base" in loaded
    assert "model.layers.0.hc_attn_layer.hc_pre.hc_scale" in loaded
    assert "model.layers.0.self_attn.learnable_sink_param" in loaded
    assert "model.hc_head.hc_head_fn" in loaded
