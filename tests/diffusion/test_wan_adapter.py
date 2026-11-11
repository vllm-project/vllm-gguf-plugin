# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import gguf
import pytest
import torch

from vllm_gguf_plugin.quantization import DiffusionGGUFConfig, DiffusionGGUFLinearMethod
from vllm_gguf_plugin.weights_adapter.diffusion import get_diffusion_gguf_adapter
from vllm_gguf_plugin.weights_adapter.diffusion.wan import WanDiffusionGGUFAdapter


@pytest.mark.parametrize("suffix", ["weight", "weight_type", "bias"])
def test_wan_mapping_keeps_qkv_shards(suffix):
    names = [f"blocks.0.self_attn.{shard}.{suffix}" for shard in ("q", "k", "v")]
    mapped = list(
        WanDiffusionGGUFAdapter.gguf_to_hf_mapper.apply((n, None) for n in names)
    )
    assert [n for n, _ in mapped] == [
        f"blocks.0.attn1.to_{shard}.{suffix}" for shard in ("q", "k", "v")
    ]


@pytest.mark.parametrize(
    "original,expected",
    [
        ("head.modulation", "scale_shift_table"),
        ("head.head.weight", "proj_out.weight"),
        ("blocks.3.modulation", "blocks.3.scale_shift_table"),
        ("blocks.3.norm3.weight", "blocks.3.norm2.weight"),
        ("blocks.3.ffn.0.weight", "blocks.3.ffn.net.0.proj.weight"),
        ("blocks.3.ffn.2.weight_type", "blocks.3.ffn.net.2.weight_type"),
        ("text_embedding.0.weight", "condition_embedder.text_embedder.linear_1.weight"),
        ("time_projection.1.bias", "condition_embedder.time_proj.bias"),
    ],
)
def test_wan_original_names(original, expected):
    assert list(
        WanDiffusionGGUFAdapter.gguf_to_hf_mapper.apply([(original, None)])
    ) == [(expected, None)]


def test_wan_adapter_selection():
    assert isinstance(
        get_diffusion_gguf_adapter("unused", "WanPipeline", None),
        WanDiffusionGGUFAdapter,
    )


@pytest.mark.parametrize("shape", [(2, 4), (2, 3, 4), (1, 2, 3, 4)])
def test_mixed_qkv_preserves_feature_order(monkeypatch, shape):
    # Wan Q4_K_M stores Q/K as Q4_K and V as Q6_K.
    weight = torch.nn.Parameter(
        torch.arange(24, dtype=torch.float32).reshape(6, 4), requires_grad=False
    )
    weight.shard_id = ["k", "v", "q"]
    weight.shard_offset_map = {"q": (0, 2, 4), "k": (2, 4, 4), "v": (4, 6, 4)}
    layer = SimpleNamespace(
        weight=weight,
        weight_type=SimpleNamespace(
            shard_weight_type={
                "q": gguf.GGMLQuantizationType.Q4_K,
                "k": gguf.GGMLQuantizationType.Q4_K,
                "v": gguf.GGMLQuantizationType.Q6_K,
            }
        ),
    )
    monkeypatch.setattr(
        "vllm_gguf_plugin.quantization.diffusion_config.dequant_gguf",
        lambda w, t, dtype: w,
    )
    x = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(
        shape
    )
    bias = torch.arange(6, dtype=torch.float32)
    method = DiffusionGGUFLinearMethod(DiffusionGGUFConfig())
    torch.testing.assert_close(method.apply(layer, x, bias), x @ weight.T + bias)
