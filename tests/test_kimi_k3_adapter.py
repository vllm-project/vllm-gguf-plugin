# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from gguf import GGMLQuantizationType
from gguf.quants import dequantize, quantize
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

from vllm_gguf_plugin.config_parser import KIMI_K3_GGUF_TEXT_MARKER
from vllm_gguf_plugin.weights_adapter import get_weights_adapter
from vllm_gguf_plugin.weights_adapter.kimi_k3 import (
    KimiK3GGUFWeightsAdapter,
    build_kimi_k3_name_map,
)


def kimi_k3_config() -> KimiLinearConfig:
    return KimiLinearConfig(
        hidden_size=7168,
        num_hidden_layers=93,
        first_k_dense_replace=1,
        num_experts=896,
        linear_attn_config={
            "kda_layers": [
                index for index in range(1, 94) if index % 4 != 0 and index != 93
            ],
            "full_attn_layers": [index for index in range(1, 94) if index % 4 == 0]
            + [93],
        },
        **{KIMI_K3_GGUF_TEXT_MARKER: True},
    )


def map_weights(weights):
    """Run the adapter's weight transform against a fresh Kimi-K3 config."""
    adapter = KimiK3GGUFWeightsAdapter()
    model_config = SimpleNamespace(hf_config=kimi_k3_config())
    return adapter.transform_weights(weights, model_config)


def extend_unquantized_modules(
    weight_type_map: dict[str, str],
) -> list[str]:
    """Run the adapter's unquantized-module extension over a type map."""
    adapter = KimiK3GGUFWeightsAdapter()
    name_map = {f"gguf.{name}": name for name in weight_type_map}
    unquantized = tuple(
        name.removesuffix(".weight")
        for name, weight_type in weight_type_map.items()
        if weight_type in ("F32", "F16", "BF16") and name.endswith(".weight")
    )
    return list(adapter.extend_unquantized_modules(None, name_map, unquantized))


def test_kimi_k3_adapter_registration_is_specific():
    config = kimi_k3_config()
    assert isinstance(get_weights_adapter(config), KimiK3GGUFWeightsAdapter)
    assert not KimiK3GGUFWeightsAdapter.matches(KimiLinearConfig())


def test_kimi_k3_name_map_covers_exact_q2_contract():
    names = build_kimi_k3_name_map(kimi_k3_config())

    assert len(names) == 2573
    assert names["blk.0.ssm_g.weight"].endswith("self_attn.g_proj.weight")
    assert names["blk.3.attn_gate.weight"].endswith("self_attn.g_proj.weight")
    assert names["blk.3.attn_k_b.weight"].endswith("self_attn.k_b_proj.weight")
    assert names["blk.1.ffn_down_exps.weight"].endswith("experts.0.w2.weight")


def test_folded_a_log_is_reversed():
    original = torch.tensor([0.0, 1.0, 2.0])
    folded = -torch.exp(original)

    [(name, restored)] = list(map_weights([("model.layers.0.self_attn.A_log", folded)]))

    assert name.endswith(".A_log")
    torch.testing.assert_close(restored, original)


def test_gguf_conv1d_leading_dimension_is_removed():
    weight = torch.arange(1 * 8 * 1 * 4).reshape(1, 8, 1, 4)

    [(name, restored)] = list(
        map_weights([("model.layers.0.self_attn.q_conv1d.weight", weight)])
    )

    assert name.endswith(".q_conv1d.weight")
    assert restored.shape == (8, 1, 4)
    assert torch.equal(restored, weight.squeeze(0))


def test_canonical_gguf_conv1d_layout_is_preserved():
    weight = torch.arange(8 * 1 * 4).reshape(8, 1, 4)

    [(name, restored)] = list(
        map_weights([("model.layers.0.self_attn.q_conv1d.weight", weight)])
    )

    assert name.endswith(".q_conv1d.weight")
    assert torch.equal(restored, weight)


def test_gguf_conv1d_rejects_unexpected_layout():
    with pytest.raises(ValueError, match="conv1d tensor must have layout"):
        list(
            map_weights(
                [
                    (
                        "model.layers.0.self_attn.q_conv1d.weight",
                        torch.zeros(8, 4),
                    )
                ]
            )
        )


def test_attention_residual_score_expands_to_native_pair():
    score = torch.tensor([2.0, 3.0, 5.0])

    mapped = dict(
        map_weights([("model.layers.7.self_attention_res_score.weight", score)])
    )

    assert torch.equal(
        mapped["model.layers.7.self_attention_res_norm.weight"],
        torch.ones_like(score),
    )
    assert torch.equal(
        mapped["model.layers.7.self_attention_res_proj.weight"],
        score.reshape(1, -1),
    )


def test_mla_q8_split_pair_is_reconstructed_for_vllm():
    qtype = GGMLQuantizationType.Q8_0
    native_k = torch.linspace(-2, 2, 2 * 32 * 32).reshape(2, 32, 32).numpy()
    native_v = torch.linspace(3, -3, 2 * 32 * 32).reshape(2, 32, 32).numpy()
    gguf_k = quantize(native_k.transpose(0, 2, 1), qtype)
    gguf_v = quantize(native_v, qtype)
    prefix = "model.layers.3.self_attn"

    mapped = dict(
        map_weights(
            [
                (f"{prefix}.k_b_proj.qweight_type", torch.tensor(qtype)),
                (f"{prefix}.k_b_proj.qweight", torch.from_numpy(gguf_k)),
                (f"{prefix}.v_b_proj.qweight_type", torch.tensor(qtype)),
                (f"{prefix}.v_b_proj.qweight", torch.from_numpy(gguf_v)),
            ]
        )
    )

    expected_k = torch.from_numpy(dequantize(gguf_k, qtype)).transpose(1, 2)
    expected_v = torch.from_numpy(dequantize(gguf_v, qtype))
    expected = torch.cat((expected_k, expected_v), dim=1).reshape(128, 32)
    assert mapped.keys() == {f"{prefix}.kv_b_proj.weight"}
    assert torch.equal(mapped[f"{prefix}.kv_b_proj.weight"], expected)


def test_mla_f32_split_pair_is_reconstructed_for_vllm():
    prefix = "model.layers.3.self_attn"
    k_b = torch.arange(2 * 32 * 16).reshape(2, 32, 16)
    v_b = torch.arange(2 * 8 * 32).reshape(2, 8, 32)

    mapped = dict(
        map_weights(
            [
                (f"{prefix}.k_b_proj.weight", k_b),
                (f"{prefix}.v_b_proj.weight", v_b),
            ]
        )
    )

    expected = torch.cat((k_b.transpose(1, 2), v_b), dim=1).reshape(48, 32)
    assert mapped.keys() == {f"{prefix}.kv_b_proj.weight"}
    assert torch.equal(mapped[f"{prefix}.kv_b_proj.weight"], expected)


def test_mla_fused_projection_is_marked_unquantized():
    modules = extend_unquantized_modules(
        {
            "model.layers.3.self_attn.k_b_proj.weight": "Q8_0",
            "model.layers.3.self_attn.v_b_proj.weight": "Q8_0",
            "model.layers.3.self_attn.q_proj.weight": "Q8_0",
        }
    )

    assert modules == ["model.layers.3.self_attn.kv_b_proj"]


def test_f32_packed_projections_are_marked_unquantized():
    modules = extend_unquantized_modules(
        {
            "model.layers.0.mlp.gate_proj.weight": "F32",
            "model.layers.0.mlp.up_proj.weight": "F32",
            "model.layers.3.block_sparse_moe.shared_experts.gate_proj.weight": ("F32"),
            "model.layers.3.block_sparse_moe.shared_experts.up_proj.weight": "F32",
            "model.layers.3.self_attn.q_a_proj.weight": "F32",
            "model.layers.3.self_attn.kv_a_proj_with_mqa.weight": "F32",
            "model.layers.3.block_sparse_moe.experts.0.w1.weight": "F32",
            "model.layers.3.block_sparse_moe.experts.0.w2.weight": "F32",
            "model.layers.3.block_sparse_moe.experts.0.w3.weight": "F32",
        }
    )

    assert "model.layers.0.mlp.gate_up_proj" in modules
    assert "model.layers.3.block_sparse_moe.shared_experts.gate_up_proj" in modules
    assert "model.layers.3.self_attn.fused_qkv_a_proj" in modules
    assert "model.layers.3.block_sparse_moe.experts" in modules


def test_f32_packed_projection_rejects_mixed_precision():
    with pytest.raises(ValueError, match="mixes quantized and unquantized"):
        extend_unquantized_modules(
            {
                "model.layers.0.mlp.gate_proj.weight": "F32",
                "model.layers.0.mlp.up_proj.weight": "Q8_0",
            }
        )


def test_f32_expert_group_rejects_mixed_precision():
    with pytest.raises(ValueError, match="mixes quantized and unquantized"):
        extend_unquantized_modules(
            {
                "model.layers.3.block_sparse_moe.experts.0.w1.weight": "F32",
                "model.layers.3.block_sparse_moe.experts.0.w2.weight": "F32",
                "model.layers.3.block_sparse_moe.experts.0.w3.weight": "Q8_0",
            }
        )


def test_kda_mixed_fused_projection_is_marked_unquantized():
    modules = extend_unquantized_modules(
        {
            "model.layers.0.self_attn.q_proj.weight": "Q8_0",
            "model.layers.0.self_attn.b_proj.weight": "F32",
        }
    )

    assert "model.layers.0.self_attn.in_proj_qkvgfab" in modules


@pytest.mark.parametrize("component", ["q_proj", "b_proj"])
def test_kda_q8_input_projection_component_is_dequantized(component):
    qtype = GGMLQuantizationType.Q8_0
    original = torch.linspace(-2, 2, 32 * 64).reshape(32, 64).numpy()
    quantized = quantize(original, qtype)
    prefix = f"model.layers.0.self_attn.{component}"

    mapped = dict(
        map_weights(
            [
                (f"{prefix}.qweight_type", torch.tensor(qtype)),
                (f"{prefix}.qweight", torch.from_numpy(quantized)),
            ]
        )
    )

    expected = torch.from_numpy(dequantize(quantized, qtype))
    assert mapped.keys() == {f"{prefix}.weight"}
    assert torch.equal(mapped[f"{prefix}.weight"], expected)


def test_mla_q8_output_gate_stays_quantized():
    qtype = GGMLQuantizationType.Q8_0
    quantized = torch.arange(64, dtype=torch.uint8).reshape(2, 32)
    prefix = "model.layers.3.self_attn.g_proj"

    mapped = dict(
        map_weights(
            [
                (f"{prefix}.qweight_type", torch.tensor(qtype)),
                (f"{prefix}.qweight", quantized),
            ]
        )
    )

    assert mapped.keys() == {
        f"{prefix}.qweight_type",
        f"{prefix}.qweight",
    }
    assert torch.equal(mapped[f"{prefix}.qweight"], quantized)


def test_q8_embedding_is_dequantized_for_native_kimi_embedding():
    qtype = GGMLQuantizationType.Q8_0
    original = torch.linspace(-2, 2, 32 * 64).reshape(32, 64).numpy()
    quantized = quantize(original, qtype)
    prefix = "model.embed_tokens"

    mapped = dict(
        map_weights(
            [
                (f"{prefix}.qweight_type", torch.tensor(qtype)),
                (f"{prefix}.qweight", torch.from_numpy(quantized)),
            ]
        )
    )

    expected = torch.from_numpy(dequantize(quantized, qtype))
    assert mapped.keys() == {f"{prefix}.weight"}
    assert torch.equal(mapped[f"{prefix}.weight"], expected)


def test_latent_moe_routed_q8_projection_is_dequantized():
    qtype = GGMLQuantizationType.Q8_0
    original = torch.linspace(-2, 2, 32 * 64).reshape(32, 64).numpy()
    quantized = quantize(original, qtype)
    prefix = "model.layers.1.block_sparse_moe.routed_expert_down_proj"

    mapped = dict(
        map_weights(
            [
                (f"{prefix}.qweight_type", torch.tensor(qtype)),
                (f"{prefix}.qweight", torch.from_numpy(quantized)),
            ]
        )
    )

    expected = torch.from_numpy(dequantize(quantized, qtype))
    assert mapped.keys() == {f"{prefix}.weight"}
    assert torch.equal(mapped[f"{prefix}.weight"], expected)
