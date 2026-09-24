# SPDX-License-Identifier: Apache-2.0

"""GGUF MoE must treat negative expert ids as empty routing slots.

vLLM's top-k router writes id -1 for padding tokens (``VLLM_MOE_SKIP_PADDING``,
on by default), and its own MoE kernels skip such slots. The GGUF kernels index
expert weights directly by id, so -1 read the bytes before the expert tensor:
the first full vLLM load of Qwen3.8-Flash-Next UD-IQ4_XS hit an illegal memory
access in its profile run, where every token is padding.
"""

import gguf
import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA kernel behaviour"
)

_Q8_0 = int(GGMLQuantizationType.Q8_0)
_EXPERTS, _HIDDEN, _INTERMEDIATE, _TOP_K = 4, 64, 32, 2


def _q8_experts(rows: int, cols: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((_EXPERTS, rows, cols), dtype=np.float32)
    packed = gguf.quants.quantize(values, GGMLQuantizationType.Q8_0)
    return torch.from_numpy(packed).cuda()


# 16 tokens take the per-row (MMVQ) path, 128 the grouped (MMQ) path.
@pytest.mark.parametrize("num_tokens", [16, 128])
@torch.inference_mode()
def test_negative_expert_ids_contribute_nothing(num_tokens):
    from vllm_gguf_plugin.quantization.fused_moe import _fused_moe_gguf

    torch.manual_seed(0)
    w13 = _q8_experts(2 * _INTERMEDIATE, _HIDDEN, seed=1)
    w2 = _q8_experts(_HIDDEN, _INTERMEDIATE, seed=2)
    x = torch.randn(num_tokens, _HIDDEN, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(
        0, _EXPERTS, (num_tokens, _TOP_K), dtype=torch.int32, device="cuda"
    )
    weights = torch.rand(num_tokens, _TOP_K, dtype=torch.float32, device="cuda")

    def run(topk_weights, topk_ids):
        return _fused_moe_gguf(x, w13, w2, topk_weights, topk_ids, _Q8_0, _Q8_0, "silu")

    padded_ids = ids.clone()
    padded = slice(num_tokens // 2, None)
    padded_ids[padded] = -1  # whole padding tokens
    padded_ids[0, 1] = -1  # a single empty slot
    padded_weights = weights.clone()

    out = run(padded_weights, padded_ids)
    torch.cuda.synchronize()

    # Reference: the same batch with valid ids, and the empty slot's weight zero.
    ref_weights = weights.clone()
    ref_weights[0, 1] = 0
    ref = run(ref_weights, ids)

    assert torch.count_nonzero(out[padded]) == 0
    torch.testing.assert_close(out[: num_tokens // 2], ref[: num_tokens // 2])
    # Routing tensors belong to the caller (vLLM records them for replay).
    assert torch.equal(padded_ids[padded], torch.full_like(padded_ids[padded], -1))
    assert torch.equal(padded_weights, weights)
