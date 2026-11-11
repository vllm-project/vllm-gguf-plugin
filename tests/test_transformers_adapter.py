# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from transformers import LlamaConfig

from vllm_gguf_plugin.weights_adapter.transformers import (
    TransformersGGUFWeightsAdapter,
)


def llama_cpp_permute(weight: torch.Tensor, num_heads: int) -> torch.Tensor:
    """`LlamaModel.permute` from llama.cpp's `convert_hf_to_gguf.py`."""
    shape = (num_heads, 2, weight.shape[0] // num_heads // 2, *weight.shape[1:])
    return weight.reshape(shape).swapaxes(1, 2).reshape(weight.shape)


def test_llama_qk_rows_are_unpermuted():
    """Q/K rows (packed quantized bytes included) get llama.cpp's permutation undone."""
    config = LlamaConfig(num_attention_heads=4, num_key_value_heads=2)
    model_config = SimpleNamespace(hf_config=config)
    prefix = "model.layers.0.self_attn"
    q = torch.randint(0, 256, (32, 10), dtype=torch.uint8)
    k = torch.randint(0, 256, (16, 10), dtype=torch.uint8)
    v = torch.randint(0, 256, (16, 10), dtype=torch.uint8)
    q_type = torch.tensor(8)
    weights = [
        (f"{prefix}.q_proj.weight_type", q_type),
        (f"{prefix}.q_proj.weight", llama_cpp_permute(q, 4)),
        (f"{prefix}.k_proj.weight", llama_cpp_permute(k, 2)),
        (f"{prefix}.v_proj.weight", v),
    ]

    adapter = TransformersGGUFWeightsAdapter()
    out = dict(adapter.transform_weights(weights, model_config))

    assert torch.equal(out[f"{prefix}.q_proj.weight_type"], q_type)
    assert torch.equal(out[f"{prefix}.q_proj.weight"], q)
    assert torch.equal(out[f"{prefix}.k_proj.weight"], k)
    assert torch.equal(out[f"{prefix}.v_proj.weight"], v)
