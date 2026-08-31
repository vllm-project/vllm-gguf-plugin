# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for diffusion GGUF quantization config and linear method."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

from vllm_gguf_plugin.quantization import (
    UNQUANTIZED_TYPES,
    DiffusionGGUFConfig,
    DiffusionGGUFLinearMethod,
    dequant_gemm_gguf,
)

pytestmark = [pytest.mark.cpu]


def test_gguf_config_creation_and_delegation():
    config = DiffusionGGUFConfig(
        gguf_model="weights.gguf",
        unquantized_modules=["proj_out"],
    )

    assert config.gguf_model == "weights.gguf"
    assert config.unquantized_modules == ["proj_out"]
    assert config.get_name() == "gguf"


def test_gguf_config_returns_diffusion_linear_method_for_linear_layers():
    linear = object.__new__(LinearBase)
    method = DiffusionGGUFConfig(unquantized_modules=[]).get_quant_method(
        linear, "transformer.img_in"
    )

    assert isinstance(method, DiffusionGGUFLinearMethod)


def test_gguf_config_respects_unquantized_modules():
    linear = object.__new__(LinearBase)
    method = DiffusionGGUFConfig(unquantized_modules=["proj_out"]).get_quant_method(
        linear, "transformer.proj_out"
    )

    assert isinstance(method, UnquantizedLinearMethod)


def test_gguf_config_returns_none_for_non_linear_layers():
    method = DiffusionGGUFConfig(unquantized_modules=[]).get_quant_method(
        torch.nn.LayerNorm(4), "norm"
    )

    assert method is None


def test_dequant_gemm_gguf_uses_plain_matmul_for_unquantized_types():
    weight_type = next(iter(UNQUANTIZED_TYPES))
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    weight = torch.tensor([[3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)

    out = dequant_gemm_gguf(x, weight, weight_type)

    assert torch.allclose(out, x @ weight.T)


def test_diffusion_gguf_linear_method_applies_bias_on_unquantized_weight():
    weight_type = next(iter(UNQUANTIZED_TYPES))
    method = DiffusionGGUFLinearMethod(DiffusionGGUFConfig(unquantized_modules=[]))
    layer = SimpleNamespace(
        weight=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        weight_type=SimpleNamespace(weight_type=weight_type),
    )
    x = torch.tensor([[[1.0, 0.5]]], dtype=torch.float32)
    bias = torch.tensor([0.25, -0.5], dtype=torch.float32)

    out = method.apply(layer, x, bias)

    expected = x @ layer.weight.T + bias
    assert torch.allclose(out, expected)


def test_diffusion_gguf_linear_method_concatenates_sharded_outputs():
    weight_type = next(iter(UNQUANTIZED_TYPES))
    method = DiffusionGGUFLinearMethod(DiffusionGGUFConfig(unquantized_modules=[]))
    weight = torch.nn.Parameter(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 0.0],
                [0.0, 2.0],
            ],
            dtype=torch.float32,
        ),
        requires_grad=False,
    )
    weight.shard_id = ["left", "right"]
    weight.shard_offset_map = {
        "left": (0, 2, 2),
        "right": (2, 4, 2),
    }
    layer = SimpleNamespace(
        weight=weight,
        weight_type=SimpleNamespace(
            shard_weight_type={
                "left": weight_type,
                "right": weight_type,
            }
        ),
    )
    x = torch.tensor([[1.5, 2.0]], dtype=torch.float32)

    out = method.apply(layer, x)

    expected_q = x @ weight[:2].T
    expected_k = x @ weight[2:4].T
    assert torch.allclose(out, torch.cat([expected_q, expected_k], dim=-1))
