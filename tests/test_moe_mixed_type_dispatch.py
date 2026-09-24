# SPDX-License-Identifier: Apache-2.0

"""Each MoE tensor must reach the best kernel its own quant type supports.

``w1`` and ``w2`` are separate tensors carrying separate quant types, and the
dispatch used to test both against ``MMQ_QUANT_TYPES`` jointly: one IQ-typed
tensor sent the whole layer to ``ggml_moe_a8_vec``, the per-row kernel meant for
decode, even when the other tensor had an MMQ tile GEMM available. Dynamic GGUF
checkpoints mix types exactly that way -- IQ on ffn_gate/up_exps, a K-quant on
ffn_down_exps -- so this is the common case, not a corner.

The failure is silent: the layer still computes the right answer, just on the
wrong kernel. So these assert on the dispatch itself, by counting which ``ops``
entry point was called. Every kernel is stubbed, so this runs on CPU and
allocates no device memory.
"""

from __future__ import annotations

import pytest
import torch
from gguf import GGMLQuantizationType as WeightType

from vllm_gguf_plugin.quantization import fused_moe as fused_moe_mod
from vllm_gguf_plugin.quantization.utils import MMQ_QUANT_TYPES, MMVQ_QUANT_TYPES

E, H, INNER, TOP_K = 4, 64, 32, 2

# A type with an MMQ tile GEMM, and one without. The premise of every test
# below, so assert it rather than trusting the names.
_TILE = WeightType.Q4_K
_VEC_ONLY = WeightType.IQ3_S


def test_the_premise_holds():
    assert _TILE in MMQ_QUANT_TYPES
    assert _VEC_ONLY in MMVQ_QUANT_TYPES
    assert _VEC_ONLY not in MMQ_QUANT_TYPES


@pytest.fixture
def calls(monkeypatch):
    """Stub every kernel and record which one the dispatch picked."""
    counts: dict[str, int] = {"a8": 0, "vec": 0, "align": 0}
    # Distinct block sizes so a shared alignment is distinguishable from two.
    block_sizes = {int(_TILE): 32, int(_VEC_ONLY): 16}

    def fake_block_size(quant_type):
        return block_sizes.get(int(quant_type), 32)

    def fake_align(topk_ids, block_size, num_experts):
        counts["align"] += 1
        n = topk_ids.numel()
        return (
            torch.zeros(n, dtype=torch.int32),
            torch.zeros(n // block_size + 1, dtype=torch.int32),
            torch.tensor([n], dtype=torch.int32),
        )

    def fake_a8(
        x, w, sorted_token_ids, expert_ids, num_post_pad, quant_type, row, top_k, tokens
    ):
        counts["a8"] += 1
        return torch.zeros(tokens * top_k, row, dtype=x.dtype)

    def fake_vec(x, w, topk_ids, top_k, quant_type, n, tokens):
        counts["vec"] += 1
        return torch.zeros(tokens * top_k, n, dtype=x.dtype)

    def fake_moe_sum(out, dst):
        dst.copy_(out.sum(dim=1))

    def fake_activation(kind, out, inp):
        out.copy_(inp[..., : inp.shape[-1] // 2])

    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.fused_moe.moe_align_block_size",
        fake_align,
        raising=False,
    )
    monkeypatch.setattr("vllm_gguf_plugin.ops.ggml_moe_get_block_size", fake_block_size)
    monkeypatch.setattr("vllm_gguf_plugin.ops.ggml_moe_a8", fake_a8)
    monkeypatch.setattr("vllm_gguf_plugin.ops.ggml_moe_a8_vec", fake_vec)
    monkeypatch.setattr("vllm_gguf_plugin.ops.moe_sum", fake_moe_sum)
    monkeypatch.setattr(fused_moe_mod, "apply_moe_activation", fake_activation)
    return counts


def _run(num_tokens: int, weight_type: WeightType, weight_type2: WeightType):
    x = torch.zeros(num_tokens, H, dtype=torch.float16)
    w1 = torch.zeros(E, 2 * INNER, 8, dtype=torch.uint8)
    w2 = torch.zeros(E, H, 8, dtype=torch.uint8)
    topk_weights = torch.ones(num_tokens, TOP_K, dtype=torch.float16)
    topk_ids = torch.zeros(num_tokens, TOP_K, dtype=torch.int32)
    fused_moe_mod._fused_moe_gguf(
        x, w1, w2, topk_weights, topk_ids, int(weight_type), int(weight_type2), "silu"
    )


def test_mixed_pair_sends_each_tensor_to_its_own_kernel(calls):
    """The case this change is for: only w1 lacks a tile GEMM."""
    _run(128, _VEC_ONLY, _TILE)
    assert calls == {"a8": 1, "vec": 1, "align": 1}, (
        f"w2 should have reached the tile GEMM on its own, got {calls}"
    )


def test_mixed_pair_the_other_way_round(calls):
    _run(128, _TILE, _VEC_ONLY)
    assert calls == {"a8": 1, "vec": 1, "align": 1}, calls


def test_all_tile_pair_is_unchanged_and_aligns_once(calls):
    """Both types tile: same two ggml_moe_a8 calls, still one alignment."""
    _run(128, _TILE, _TILE)
    assert calls == {"a8": 2, "vec": 0, "align": 1}, (
        f"existing MMQ behaviour changed, got {calls}"
    )


def test_all_vec_pair_is_unchanged(calls):
    _run(128, _VEC_ONLY, _VEC_ONLY)
    assert calls == {"a8": 0, "vec": 2, "align": 0}, calls


def test_decode_batch_stays_on_the_vector_kernel(calls):
    """The x.shape[0] > 64 guard still governs both tensors."""
    _run(8, _TILE, _TILE)
    assert calls == {"a8": 0, "vec": 2, "align": 0}, calls


def test_differing_block_sizes_get_their_own_alignment(calls, monkeypatch):
    """Two tile types whose block sizes differ must not share one padding.

    ``moe_align_block_size`` pads to the block size it is given, and the kernel
    reads that padding back; feeding w2's launch an alignment built for w1's
    block size is the bug the old single ``block_size`` line would have had the
    moment two tile types disagreed.
    """
    second_tile = WeightType.Q5_K
    assert second_tile in MMQ_QUANT_TYPES
    monkeypatch.setattr(
        "vllm_gguf_plugin.ops.ggml_moe_get_block_size",
        lambda quant_type: 32 if int(quant_type) == int(_TILE) else 8,
    )
    _run(128, _TILE, second_tile)
    assert calls["a8"] == 2, calls
    assert calls["align"] == 2, f"w2 reused w1's alignment, got {calls}"


def test_unsupported_type_still_falls_through_to_the_slow_path(calls, monkeypatch):
    """A type in neither set must keep reaching the dequantize loop."""
    unsupported = max(int(t) for t in MMVQ_QUANT_TYPES) + 1
    seen = {"slow": 0}

    def fake_slow(inp, w, quant_type):
        seen["slow"] += 1
        return torch.zeros(inp.shape[0], w.shape[0], dtype=inp.dtype)

    # The fallback imports it from the package at call time, so patch it there.
    monkeypatch.setattr("vllm_gguf_plugin.quantization.fused_mul_mat_gguf", fake_slow)
    _run(128, unsupported, unsupported)
    assert calls["a8"] == 0 and calls["vec"] == 0, calls
    assert seen["slow"] > 0, "the unsupported-type fallback stopped being reached"
