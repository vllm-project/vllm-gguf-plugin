# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_gguf_plugin.weights_adapter.qwen3_5 import (
    Qwen35GGUFAdapter,
    build_qwen35_text_mapper,
    split_fused_gdn_inputs,
)

# 2 key heads, 4 value heads (2 per key head, so the GDN value-head restore
# runs), head dims 4: qkv 2*2*4 + 4*4 = 32 rows, gate 16 rows, beta/alpha 4.
TEXT_CONFIG = SimpleNamespace(
    linear_num_key_heads=2,
    linear_key_head_dim=4,
    linear_num_value_heads=4,
    linear_value_head_dim=4,
)
PREFIX = "model.layers.0.linear_attn."


def _model_config():
    hf_config = SimpleNamespace(
        get_text_config=lambda: TEXT_CONFIG,
        vision_config=None,
    )
    return SimpleNamespace(hf_config=hf_config)


def _packed(rows: int, seed: int) -> torch.Tensor:
    """A quantized tensor as the GGUF iterator yields it: rows x row bytes."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (rows, 18), dtype=torch.uint8, generator=gen)


@pytest.mark.parametrize("multimodal", [False, True])
def test_fused_gdn_names_map_to_placeholders(multimodal):
    mapper = build_qwen35_text_mapper(is_multimodal=multimodal, is_moe=False)
    layers = "model.language_model.layers." if multimodal else "model.layers."
    names = [
        "blk.0.attn_qkv_gate.weight",
        "blk.0.ssm_beta_alpha.weight",
        "blk.0.attn_qkv.weight",
        "blk.0.attn_gate.weight",
        "blk.0.ssm_beta.weight",
        "blk.0.ssm_alpha.weight",
    ]
    assert mapper.apply_list(names) == [
        layers + "0.linear_attn.fused_qkv_gate.weight",
        layers + "0.linear_attn.fused_beta_alpha.weight",
        layers + "0.linear_attn.in_proj_qkv.weight",
        layers + "0.linear_attn.in_proj_z.weight",
        layers + "0.linear_attn.in_proj_b.weight",
        layers + "0.linear_attn.in_proj_a.weight",
    ]


def test_split_fused_gdn_inputs_quantized():
    qkv_gate = _packed(48, 0)
    beta_alpha = _packed(8, 1)
    weight_type = torch.tensor(14)  # Q6_K
    other = torch.ones(3)
    weights = [
        (PREFIX + "fused_qkv_gate.weight_type", weight_type),
        (PREFIX + "fused_qkv_gate.weight", qkv_gate),
        (PREFIX + "fused_beta_alpha.weight_type", weight_type),
        (PREFIX + "fused_beta_alpha.weight", beta_alpha),
        ("model.layers.0.input_layernorm.weight", other),
    ]

    out = list(split_fused_gdn_inputs(weights, TEXT_CONFIG))

    assert [name for name, _ in out] == [
        PREFIX + "in_proj_qkv.weight_type",
        PREFIX + "in_proj_z.weight_type",
        PREFIX + "in_proj_qkv.weight",
        PREFIX + "in_proj_z.weight",
        PREFIX + "in_proj_b.weight_type",
        PREFIX + "in_proj_a.weight_type",
        PREFIX + "in_proj_b.weight",
        PREFIX + "in_proj_a.weight",
        "model.layers.0.input_layernorm.weight",
    ]
    tensors = dict(out)
    assert int(tensors[PREFIX + "in_proj_z.weight_type"]) == 14
    assert torch.equal(tensors[PREFIX + "in_proj_qkv.weight"], qkv_gate[:32])
    assert torch.equal(tensors[PREFIX + "in_proj_z.weight"], qkv_gate[32:])
    assert torch.equal(tensors[PREFIX + "in_proj_b.weight"], beta_alpha[:4])
    assert torch.equal(tensors[PREFIX + "in_proj_a.weight"], beta_alpha[4:])
    assert tensors["model.layers.0.input_layernorm.weight"] is other


def test_split_fused_gdn_inputs_rejects_wrong_head_counts():
    weights = [(PREFIX + "fused_qkv_gate.weight", torch.zeros(40, 8))]
    with pytest.raises(ValueError, match="config expects 48"):
        list(split_fused_gdn_inputs(weights, TEXT_CONFIG))


@pytest.mark.parametrize("quantized", [False, True])
def test_fused_gguf_loads_like_unfused(quantized):
    """A fused GGUF must yield exactly the weights of the unfused one."""
    if quantized:
        qkv, gate, beta, alpha = (
            _packed(32, 2),
            _packed(16, 3),
            _packed(4, 4),
            _packed(4, 5),
        )
    else:
        qkv, gate, beta, alpha = (torch.randn(n, 8) for n in (32, 16, 4, 4))

    def stream(tensors):
        for name, weight in tensors:
            if quantized:
                yield name.replace("weight", "weight_type"), torch.tensor(14)
            yield name, weight

    unfused = [
        (PREFIX + "in_proj_qkv.weight", qkv),
        (PREFIX + "in_proj_z.weight", gate),
        (PREFIX + "in_proj_b.weight", beta),
        (PREFIX + "in_proj_a.weight", alpha),
    ]
    fused = [
        (PREFIX + "fused_qkv_gate.weight", torch.cat([qkv, gate])),
        (PREFIX + "fused_beta_alpha.weight", torch.cat([beta, alpha])),
    ]
    adapter = Qwen35GGUFAdapter()
    expected = list(adapter.transform_weights(stream(unfused), _model_config()))
    actual = list(adapter.transform_weights(stream(fused), _model_config()))

    expected_by_name = dict(expected)
    actual_by_name = dict(actual)
    assert sorted(actual_by_name) == sorted(expected_by_name)
    for name, weight in expected_by_name.items():
        assert torch.equal(actual_by_name[name], weight), name
