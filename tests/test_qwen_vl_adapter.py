# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_gguf_plugin.weights_adapter import (
    QwenVLGGUFAdapter,
    get_adapter_architecture,
    get_weights_adapter,
)
from vllm_gguf_plugin.weights_adapter.qwen_vl import (
    build_qwen3_vl_deepstack_mapper,
    build_qwen_vl_text_mapper,
    build_qwen_vl_vision_mapper,
)


@pytest.mark.parametrize(
    ("model_type", "architecture"),
    [
        ("qwen2_vl", "Qwen2VLForConditionalGeneration"),
        ("qwen2_5_vl", "Qwen2_5_VLForConditionalGeneration"),
        ("qwen3_vl", "Qwen3VLForConditionalGeneration"),
    ],
)
def test_qwen_vl_adapter_registration(model_type: str, architecture: str):
    config = SimpleNamespace(model_type=model_type)

    adapter = get_weights_adapter(config)

    assert isinstance(adapter, QwenVLGGUFAdapter)
    assert get_adapter_architecture(config) == architecture


def test_qwen_vl_text_name_mapping():
    mapper = build_qwen_vl_text_mapper()

    assert mapper.apply_list(["blk.2.attn_q.weight"])[0] == (
        "model.language_model.layers.2.self_attn.q_proj.weight"
    )


@pytest.mark.parametrize(
    ("model_type", "gguf_name", "hf_name"),
    [
        (
            "qwen2_vl",
            "v.blk.3.attn_q.weight",
            "model.visual.blocks.3.attn.qkv.weight",
        ),
        (
            "qwen2_vl",
            "v.blk.3.attn_k.weight",
            "model.visual.blocks.3.attn.k_proj.weight",
        ),
        (
            "qwen2_vl",
            "v.blk.3.attn_v.weight",
            "model.visual.blocks.3.attn.v_proj.weight",
        ),
        (
            "qwen2_vl",
            "v.blk.3.ffn_up.weight",
            "model.visual.blocks.3.mlp.fc2.weight",
        ),
        (
            "qwen2_5_vl",
            "v.blk.3.attn_q.weight",
            "model.visual.blocks.3.attn.qkv.weight",
        ),
        (
            "qwen2_5_vl",
            "v.blk.3.attn_k.weight",
            "model.visual.blocks.3.attn.k.weight",
        ),
        (
            "qwen2_5_vl",
            "v.blk.3.attn_v.weight",
            "model.visual.blocks.3.attn.v.weight",
        ),
        (
            "qwen2_5_vl",
            "v.blk.3.ffn_gate.weight",
            "model.visual.blocks.3.mlp.gate_proj.weight",
        ),
        (
            "qwen3_vl",
            "v.blk.3.attn_qkv.weight",
            "model.visual.blocks.3.attn.qkv.weight",
        ),
        (
            "qwen3_vl",
            "mm.2.bias",
            "model.visual.merger.linear_fc2.bias",
        ),
    ],
)
def test_qwen_vl_vision_name_mapping(
    model_type: str,
    gguf_name: str,
    hf_name: str,
):
    mapper = build_qwen_vl_vision_mapper(model_type)

    assert mapper.apply_list([gguf_name])[0] == hf_name


def test_qwen3_vl_deepstack_name_mapping():
    mapper = build_qwen3_vl_deepstack_mapper([5, 11, 17])

    assert mapper.apply_list(["v.deepstack.11.fc1.weight"])[0] == (
        "model.visual.deepstack_merger_list.1.linear_fc1.weight"
    )


def test_qwen_vl_restores_temporal_patch_embedding():
    adapter = QwenVLGGUFAdapter()
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="qwen3_vl",
            vision_config=SimpleNamespace(temporal_patch_size=2),
        )
    )
    first_patch = torch.arange(24).reshape(2, 3, 2, 2)
    second_patch = first_patch + 24
    weights = [
        ("model.language_model.norm.weight", torch.tensor([2.0, 3.0])),
        ("model.visual.patch_embed.proj.weight", first_patch),
        ("model.visual.patch_embed.proj.weight.1", second_patch),
    ]

    transformed = dict(adapter.transform_weights(weights, model_config))

    assert torch.equal(
        transformed["model.language_model.norm.weight"], torch.tensor([2.0, 3.0])
    )
    assert torch.equal(
        transformed["model.visual.patch_embed.proj.weight"],
        torch.stack([first_patch, second_patch], dim=2),
    )


@pytest.mark.parametrize(
    ("model_type", "shard_names"),
    [
        ("qwen2_vl", ("qkv", "k_proj", "v_proj")),
        ("qwen2_5_vl", ("qkv", "k", "v")),
    ],
)
def test_qwen_vl_merges_vision_qkv(model_type: str, shard_names: tuple[str, ...]):
    adapter = QwenVLGGUFAdapter()
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type=model_type,
            vision_config=SimpleNamespace(temporal_patch_size=2),
        )
    )
    weights = [
        (
            f"model.visual.blocks.0.attn.{name}.weight",
            torch.full((2, 3), value),
        )
        for value, name in enumerate(shard_names)
    ]

    transformed = dict(adapter.transform_weights(weights, model_config))

    assert torch.equal(
        transformed["model.visual.blocks.0.attn.qkv.weight"],
        torch.cat([weight for _, weight in weights], dim=0),
    )


def test_qwen_vl_rejects_incomplete_vision_qkv():
    adapter = QwenVLGGUFAdapter()
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="qwen2_5_vl",
            vision_config=SimpleNamespace(temporal_patch_size=2),
        )
    )
    weights = [
        ("model.visual.blocks.0.attn.qkv.weight", torch.ones((2, 3))),
        ("model.visual.blocks.0.attn.k.weight", torch.ones((2, 3))),
    ]

    with pytest.raises(RuntimeError, match="Incomplete Qwen-VL QKV tensors"):
        list(adapter.transform_weights(weights, model_config))


def test_qwen_vl_rejects_incomplete_temporal_patch_embedding():
    adapter = QwenVLGGUFAdapter()
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="qwen3_vl",
            vision_config=SimpleNamespace(temporal_patch_size=2),
        )
    )
    weights = [("model.visual.patch_embed.proj.weight", torch.ones((2, 3, 2, 2)))]

    with pytest.raises(
        RuntimeError,
        match="Incomplete Qwen-VL temporal patch embedding tensors",
    ):
        list(adapter.transform_weights(weights, model_config))
