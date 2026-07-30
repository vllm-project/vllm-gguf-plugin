# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from gguf import GGMLQuantizationType, GGUFReader
from gguf.quants import dequantize, quantize

from vllm_gguf_plugin.quantization.fused_moe import (
    GGUFMoEMethod,
    _apply_gguf_moe_activation,
    _moe_token_slices,
    fused_moe_gguf,
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_kimi_k3_situ_betas_match_reference(dtype: torch.dtype):
    inp = torch.linspace(-40, 40, 8 * 64, device="cuda", dtype=dtype).reshape(8, 64)
    beta = 4.0
    linear_beta = 25.0

    actual = _apply_gguf_moe_activation(inp, "situ", beta, linear_beta)

    gate, up = inp.float().chunk(2, dim=-1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    up = linear_beta * torch.tanh(up / linear_beta)
    expected = (gate * up).to(dtype)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@torch.inference_mode()
def test_situ_missing_beta_uses_upstream_defaults():
    inp = torch.ones((2, 64), device="cuda", dtype=torch.bfloat16)

    actual = _apply_gguf_moe_activation(inp, "situ", -1.0, -1.0)

    gate, up = inp.float().chunk(2, dim=-1)
    expected = (torch.tanh(gate) * torch.sigmoid(gate) * up).to(inp.dtype)
    torch.testing.assert_close(actual, expected)


def test_kimi_profile_batch_splits_before_routed_token_launch_limit():
    slices = _moe_token_slices(num_tokens=4096, top_k=16)

    assert slices == [slice(0, 4095), slice(4095, 4096)]


@torch.inference_mode()
def test_fused_gguf_moe_custom_op_carries_situ_betas():
    qtype = GGMLQuantizationType.Q8_0
    experts, hidden_size, intermediate_size = 2, 32, 32
    w13 = torch.linspace(
        -0.5,
        0.5,
        experts * 2 * intermediate_size * hidden_size,
    ).reshape(experts, 2 * intermediate_size, hidden_size)
    w2 = torch.linspace(
        0.25,
        -0.25,
        experts * hidden_size * intermediate_size,
    ).reshape(experts, hidden_size, intermediate_size)
    w13_q = quantize(w13.numpy(), qtype)
    w2_q = quantize(w2.numpy(), qtype)
    w13_dq = torch.from_numpy(dequantize(w13_q, qtype)).cuda().bfloat16()
    w2_dq = torch.from_numpy(dequantize(w2_q, qtype)).cuda().bfloat16()
    x = (
        torch.linspace(-1, 1, 2 * hidden_size, device="cuda")
        .reshape(2, hidden_size)
        .bfloat16()
    )
    topk_ids = torch.tensor([[0], [3]], device="cuda", dtype=torch.int32)
    topk_weights = torch.tensor([[0.75], [0.25]], device="cuda").bfloat16()
    expert_map = torch.tensor([0, -1, -1, 1], device="cuda", dtype=torch.int32)

    actual = fused_moe_gguf(
        x,
        torch.from_numpy(w13_q).cuda(),
        torch.from_numpy(w2_q).cuda(),
        topk_weights,
        topk_ids,
        expert_map,
        int(qtype),
        int(qtype),
        "situ",
        4.0,
        25.0,
    )

    expected = torch.empty_like(x)
    for token_idx, expert_idx in enumerate((0, 1)):
        gate_up = x[token_idx] @ w13_dq[expert_idx].T
        activated = _apply_gguf_moe_activation(gate_up.reshape(1, -1), "situ", 4, 25)
        expected[token_idx] = (activated @ w2_dq[expert_idx].T).squeeze(
            0
        ) * topk_weights[token_idx, 0]
    torch.testing.assert_close(actual, expected, atol=0.5, rtol=0.1)


def test_iq2_xs_plain_tp8_is_rejected_with_ep_guidance():
    layer = SimpleNamespace(
        moe_config=SimpleNamespace(
            intermediate_size=3072,
            moe_parallel_config=SimpleNamespace(tp_size=8),
        ),
        w2_qweight_type=SimpleNamespace(weight_type=int(GGMLQuantizationType.IQ2_XS)),
    )

    with pytest.raises(ValueError, match="--enable-expert-parallel"):
        GGUFMoEMethod.process_weights_after_loading(None, layer)


@pytest.mark.skipif(
    not os.environ.get("KIMI_IQ2_FIXTURE"),
    reason="set KIMI_IQ2_FIXTURE to the tiny MoE GGUF sample",
)
@torch.inference_mode()
def test_real_iq2_xs_block_with_situ_and_expert_map():
    fixture = Path(os.environ["KIMI_IQ2_FIXTURE"])
    w13_tensor, w2_tensor = GGUFReader(fixture).tensors
    qtype = GGMLQuantizationType.IQ2_XS
    w13_q = w13_tensor.data[[0, 1]]
    w2_q = w2_tensor.data[[0, 1]]
    w13_dq = torch.from_numpy(dequantize(w13_q, qtype)).cuda().bfloat16()
    w2_dq = torch.from_numpy(dequantize(w2_q, qtype)).cuda().bfloat16()
    hidden_size = w13_dq.shape[-1]
    x = (
        torch.linspace(-0.25, 0.25, 2 * hidden_size, device="cuda")
        .reshape(2, hidden_size)
        .bfloat16()
    )
    topk_ids = torch.tensor([[0], [2]], device="cuda", dtype=torch.int32)
    topk_weights = torch.tensor([[0.6], [0.4]], device="cuda").bfloat16()
    expert_map = torch.tensor([0, -1, 1], device="cuda", dtype=torch.int32)

    actual = fused_moe_gguf(
        x,
        torch.from_numpy(w13_q).cuda(),
        torch.from_numpy(w2_q).cuda(),
        topk_weights,
        topk_ids,
        expert_map,
        int(qtype),
        int(qtype),
        "situ",
        4.0,
        25.0,
    )

    expected = torch.empty_like(x)
    for token_idx, expert_idx in enumerate((0, 1)):
        gate_up = x[token_idx] @ w13_dq[expert_idx].T
        activated = _apply_gguf_moe_activation(gate_up.reshape(1, -1), "situ", 4, 25)
        expected[token_idx] = (activated @ w2_dq[expert_idx].T).squeeze(
            0
        ) * topk_weights[token_idx, 0]
    torch.testing.assert_close(actual, expected, atol=1.0, rtol=0.1)
