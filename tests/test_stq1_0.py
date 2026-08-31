# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for STQ1_0 support: codec round-trip, layout ground truth, kernels."""

import gguf
import numpy as np
import pytest
import torch
from types import SimpleNamespace

from vllm_gguf_plugin.gguf_files import GGUFModelFiles
from vllm_gguf_plugin.gguf_stq import (
    STQ1_0_TYPE_ID,
    dequantize_stq1_0,
    get_stq1_0_type,
)
from vllm_gguf_plugin.weight_utils import gguf_quant_weights_iterator_multi
from vllm_gguf_plugin.weights_adapter.hy_v4 import HYV4GGUFAdapter

from .test_hy_v4_gguf import _write_tiny_hy_v4_gguf, encode_stq1_0, make_ternary


def test_stq1_0_registered_with_gguf():
    qtype = get_stq1_0_type()
    assert qtype.name == "STQ1_0"
    assert int(qtype) == STQ1_0_TYPE_ID
    assert gguf.GGML_QUANT_SIZES[qtype] == (256, 42)
    # The plugin's quant-type validator must accept the repo:quant reference.
    from vllm_gguf_plugin.gguf_utils import is_valid_gguf_quant_type

    assert is_valid_gguf_quant_type("STQ1_0")


def test_dequantize_stq1_0_ground_truth():
    """Hand-crafted block checked against the documented stride-16 layout.

    Slot 0 with sign 0 decodes to codebook[0] = 0xA9, i.e. lanes (0, +1, +1,
    +1). Group g (chunk = g//16, gloc = g%16) places lane p at weight
    chunk*64 + gloc + p*16, so with all-zero qs/sign and d = 1 the zero lanes
    (p = 0) sit at j = chunk*64 + gloc, i.e. j % 64 < 16; everything else is
    +1.
    """
    raw = torch.zeros((1, 42), dtype=torch.uint8)
    raw[0, 40:42] = torch.tensor([1.0], dtype=torch.float16).view(torch.uint8)
    out = dequantize_stq1_0(raw)
    expected = torch.ones(256).reshape(4, 64)
    expected[:, :16] = 0.0
    expected = expected.reshape(256)
    torch.testing.assert_close(out[0], expected, rtol=0, atol=0)

    # sign bit set for group 0 (bit 0 of sign byte 0) -> codebook[16] = 0x01,
    # lanes (0, -1, -1, -1): group 0's non-zero lanes flip sign.
    raw[0, 32] = 0x01
    out = dequantize_stq1_0(raw)
    expected[[16, 32, 48]] = -1.0  # group 0 covers {0, 16, 32, 48}
    torch.testing.assert_close(out[0], expected, rtol=0, atol=0)


def test_dequantize_stq1_0_roundtrip():
    original = make_ternary((3, 512))
    raw = encode_stq1_0(original)
    dequantized = dequantize_stq1_0(torch.from_numpy(raw))
    torch.testing.assert_close(
        dequantized, torch.from_numpy(original), rtol=0, atol=0
    )


@pytest.fixture(scope="module")
def tiny_hy_v4_stq(tmp_path_factory):
    path = tmp_path_factory.mktemp("gguf-stq") / "tiny-hy-v4.gguf"
    originals = _write_tiny_hy_v4_gguf(str(path), stq_experts=True)
    files = GGUFModelFiles(backbone=(str(path),))
    adapter = HYV4GGUFAdapter()
    name_map = adapter.build_name_map(files, None)
    model_config = SimpleNamespace(dtype=torch.bfloat16)
    weights = gguf_quant_weights_iterator_multi(list(files.all_files), name_map)
    loaded = dict(adapter.transform_weights(weights, model_config))
    return originals, loaded


def test_stq1_0_experts_passed_through(tiny_hy_v4_stq):
    """STQ1_0 expert tensors must reach the model in their native format."""
    originals, loaded = tiny_hy_v4_stq
    # weight_type is not split per expert (0-dim), it stays at experts.0.
    type_name = "model.layers.1.mlp.experts.0.gate_proj.weight_type"
    assert int(loaded[type_name].item()) == STQ1_0_TYPE_ID

    gate = originals["blk.1.ffn_gate_exps.weight"]
    for expert_id in (0, 3):
        name = f"model.layers.1.mlp.experts.{expert_id}.gate_proj.weight"
        assert name in loaded
        # Native STQ1_0 bytes: 256 bytes of weights -> 42 bytes per block.
        assert loaded[name].dtype == torch.uint8
        dequantized = dequantize_stq1_0(loaded[name])
        torch.testing.assert_close(
            dequantized, torch.from_numpy(gate[expert_id]), rtol=0, atol=0
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_stq1_0_kernel_dequantize():
    import vllm_gguf_plugin.ops as ops

    original = torch.from_numpy(make_ternary((4, 512)))
    qweight = torch.from_numpy(encode_stq1_0(original.numpy())).cuda()
    output = ops.ggml_dequantize(qweight, STQ1_0_TYPE_ID, 4, 512, torch.float32)
    torch.testing.assert_close(
        output.cpu(), original, atol=1e-3, rtol=1e-3
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_stq1_0_kernel_mmvq():
    import vllm_gguf_plugin.ops as ops

    original = torch.from_numpy(make_ternary((4, 512), seed=3))
    qweight = torch.from_numpy(encode_stq1_0(original.numpy())).cuda()
    for dtype in (torch.half, torch.bfloat16, torch.float32):
        x = torch.rand((2, 512), dtype=dtype, device="cuda")
        ref = x.float() @ original.cuda().T
        out = ops.ggml_mul_mat_vec_a8(qweight, x, STQ1_0_TYPE_ID, 4)
        torch.testing.assert_close(out.float(), ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_stq1_0_kernel_moe_vec_large_batch():
    """tokens*top_k beyond the 65535 grid-z limit must be chunked."""
    import vllm_gguf_plugin.ops as ops

    num_experts, nrows, ncols = 3, 256, 512
    original = torch.from_numpy(make_ternary((num_experts, nrows, ncols), seed=5))
    qweight = torch.from_numpy(encode_stq1_0(original.numpy())).cuda()
    tokens, top_k = 33_000, 2  # 66000 > 65535
    x = torch.rand((tokens, ncols), dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.randint(0, num_experts, (tokens, top_k), dtype=torch.int32,
                             device="cuda")
    out = ops.ggml_moe_a8_vec(
        x, qweight, topk_ids, top_k, STQ1_0_TYPE_ID, nrows, tokens
    )
    assert out.shape == (tokens * top_k, nrows)
    for token in (0, tokens // 2, tokens - 1):  # spans both z-chunks
        expert = int(topk_ids[token, 0])
        ref = x[token].float() @ original[expert].cuda().T
        got = out[token * top_k].float()
        torch.testing.assert_close(got, ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_stq1_0_kernel_moe_vec():
    import vllm_gguf_plugin.ops as ops

    num_experts, nrows, ncols = 3, 512, 512
    original = torch.from_numpy(make_ternary((num_experts, nrows, ncols), seed=5))
    qweight = torch.from_numpy(encode_stq1_0(original.numpy())).cuda()
    tokens, top_k = 2, 2
    x = torch.rand((tokens, ncols), dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.tensor([[0, 2], [2, 1]], dtype=torch.int32, device="cuda")
    out = ops.ggml_moe_a8_vec(
        x, qweight, topk_ids, top_k, STQ1_0_TYPE_ID, nrows, tokens
    )
    assert out.shape == (tokens * top_k, nrows)
    # token 0 -> experts 0 and 2; token 1 -> experts 2 and 1.
    for token, experts in enumerate(([0, 2], [2, 1])):
        for slot, expert in enumerate(experts):
            ref = x[token].float() @ original[expert].cuda().T
            got = out[token * top_k + slot].float()
            torch.testing.assert_close(got, ref, atol=5e-2, rtol=5e-2)
