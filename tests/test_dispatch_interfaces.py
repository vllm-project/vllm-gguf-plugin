"""Fixed methods versus centralized selection, using real CUDA executions."""

import gguf
import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType as Q

from tests.helpers_upstream import make_padded_weight
from vllm_gguf_plugin import ops
from vllm_gguf_plugin.kernel_support import KernelMethod as M

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"
)


def _quantized(moe=False, experts=3):
    shape = (experts, 128, 256) if moe else (128, 256)
    raw = gguf.quantize(
        np.random.default_rng(97).standard_normal(shape, dtype=np.float32) * 0.03,
        Q.Q8_0,
    )
    dense = torch.from_numpy(gguf.dequantize(raw, Q.Q8_0)).cuda()
    return make_padded_weight(raw, Q.Q8_0, 256), dense


def _reference(x, w, ids):
    all_experts = torch.stack([x.float() @ e.float().T for e in w], 1)
    return (
        all_experts.gather(1, ids.long()[..., None].expand(-1, -1, w.size(1)))
        .flatten(0, 1)
        .to(x.dtype)
    )


@pytest.mark.parametrize("batch", [1, 7, 8, 129])
@torch.inference_mode()
def test_dense_fixed_mmq_runs_outside_recommended_range(batch):
    w, dense = _quantized()
    x = torch.randn((batch, 256), device="cuda") * 0.1
    args = (w, x, int(Q.Q8_0), 128)
    assert ops.dense_supported_methods(*args) & M.MMQ
    torch.testing.assert_close(
        ops.ggml_dense_mmq(*args), x @ dense.T, atol=0.004, rtol=0.04
    )
    if batch == 129:
        assert ops.dense_select_method(*args) != M.MMQ


@pytest.mark.parametrize("batch", [32, 33, 63, 64, 65, 127, 128, 129])
@torch.inference_mode()
def test_dense_auto_mmq_blas_boundary_eager_and_graph(batch):
    w, dense = _quantized()
    x = torch.randn((batch, 256), device="cuda", dtype=torch.float16) * 0.1
    args = (w, x, int(Q.Q8_0), 128)
    supported = ops.dense_supported_methods(*args)
    assert supported & M.MMQ and supported & M.DEQUANTIZE_BLAS
    selected = M.MMQ if batch <= 128 else M.DEQUANTIZE_BLAS
    assert ops.dense_select_method(*args) == selected
    torch.testing.assert_close(
        ops.ggml_dense(*args),
        x.float() @ dense.T,
        atol=0.004,
        rtol=0.04,
        check_dtype=False,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert ops.dense_select_method(*args) == selected
        captured = ops.ggml_dense(*args)
    x.mul_(0.5)
    graph.replay()
    torch.testing.assert_close(
        captured, x.float() @ dense.T, atol=0.004, rtol=0.04, check_dtype=False
    )


@torch.inference_mode()
def test_dense_auto_keeps_hard_mmq_fallback_above_threshold(monkeypatch):
    monkeypatch.setenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE", "invalid")
    w, dense = _quantized()
    x = torch.randn((129, 256), device="cuda") * 0.1
    args = (w, x, int(Q.Q8_0), 128)
    assert ops.dense_supported_methods(*args) & M.MMQ
    assert ops.dense_select_method(*args) == M.MMQ
    torch.testing.assert_close(
        ops.ggml_dense(*args), x @ dense.T, atol=0.004, rtol=0.04
    )


@pytest.mark.parametrize(
    "tokens,top_k", [(32, 8), (33, 8), (256, 1), (257, 1), (64, 4), (65, 4)]
)
@torch.inference_mode()
def test_moe_auto_vector_alignment_boundary_eager_and_graph(tokens, top_k):
    w, dense = _quantized(moe=True, experts=8)
    x = torch.randn((tokens, 256), device="cuda", dtype=torch.float16) * 0.1
    ids = (
        (
            torch.arange(tokens, device="cuda")[:, None]
            + torch.arange(top_k, device="cuda")
        )
        % 8
    ).int()
    args = (x, w, ids, int(Q.Q8_0), 128, top_k, tokens)
    supported = ops.moe_supported_methods(*args)
    assert supported & M.MMVQ and supported & M.MMQ_ALIGNED
    selected = M.MMVQ if tokens * top_k <= 256 else M.MMQ_ALIGNED
    assert ops.moe_select_method(*args, allow_aligned=True) == selected
    torch.testing.assert_close(
        ops.ggml_moe(*args), _reference(x, dense, ids), atol=0.004, rtol=0.04
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert ops.moe_select_method(*args, allow_aligned=True) == selected
        captured = ops.ggml_moe(*args)
    x.mul_(0.5)
    ids.copy_((ids + 1) % 8)
    graph.replay()
    torch.testing.assert_close(
        captured, _reference(x, dense, ids), atol=0.004, rtol=0.04
    )


@pytest.mark.parametrize("floating", [False, True])
@pytest.mark.parametrize("tokens", [1, 7, 17, 513])
@torch.inference_mode()
def test_moe_supported_bits_match_fixed_executions(floating, tokens):
    if floating:
        w = torch.randn((3, 128, 256), device="cuda", dtype=torch.float16) * 0.03
        dense, q = w, Q.F16
    else:
        w, dense = _quantized(moe=True)
        q = Q.Q8_0
    x = torch.randn((tokens, 256), device="cuda", dtype=torch.float16) * 0.1
    ids = (
        (torch.arange(tokens, device="cuda")[:, None] + torch.arange(2, device="cuda"))
        % 3
    ).int()
    args = (x, w, ids, int(q), 128, 2, tokens)
    supported = ops.moe_supported_methods(*args)
    expected = _reference(x, dense, ids)
    for method in M:
        if not supported & method:
            if method != M.MMQ_ALIGNED:
                with pytest.raises(RuntimeError, match="VLLM_GGUF_MOE_NOT_ELIGIBLE"):
                    getattr(ops, f"ggml_moe_{method.name.lower()}")(*args)
            continue
        if method == M.MMQ_ALIGNED:
            from vllm.model_executor.layers.fused_moe.fused_moe import (
                moe_align_block_size,
            )

            plan = moe_align_block_size(ids, 16, 3, pad_sorted_ids=True)
            actual = ops.ggml_moe_mmq_aligned(
                x,
                w,
                plan[0],
                int(q),
                128,
                2,
                tokens,
                expert_ids=plan[1],
                padded_count=plan[2],
            )
        else:
            actual = getattr(ops, f"ggml_moe_{method.name.lower()}")(*args)
        torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.04)
    selected = ops.moe_select_method(*args, allow_aligned=True)
    assert supported & selected
    torch.testing.assert_close(ops.ggml_moe(*args), expected, atol=0.004, rtol=0.04)


@pytest.mark.parametrize("method,tokens", [("mmvf", 17), ("mmf", 513), ("mmf", 4097)])
@torch.inference_mode()
def test_forced_float_moe_ignores_recommendation_threshold_and_replays_graph(
    method, tokens
):
    w = torch.randn((3, 128, 256), device="cuda", dtype=torch.float16) * 0.03
    x = torch.randn((tokens, 256), device="cuda", dtype=torch.float16) * 0.1
    ids = (torch.arange(tokens, device="cuda") % 3).int()[:, None].contiguous()
    args = (x, w, ids, int(Q.F16), 128, 1, tokens)
    op = getattr(ops, f"ggml_moe_{method}")
    op(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(*args)
    ids.copy_((ids + 1) % 3)
    x.mul_(0.5)
    graph.replay()
    torch.testing.assert_close(captured, _reference(x, w, ids), atol=0.004, rtol=0.04)


@torch.inference_mode()
def test_dense_blas_names_enforce_weight_kind_and_support_graph():
    w, _ = _quantized()
    x = torch.randn((17, 256), device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="method cannot run"):
        ops.ggml_dense_blas(w, x, int(Q.Q8_0), 128)
    float_w = torch.randn((63, 255), device="cuda", dtype=torch.float16) * 0.03
    float_x = torch.randn((17, 255), device="cuda", dtype=torch.float32) * 0.1
    args = (float_w, float_x, int(Q.F16), 63)
    assert ops.dense_supported_methods(*args) == M.BLAS
    with pytest.raises(RuntimeError, match="method cannot run"):
        ops.ggml_dense_dequantize_blas(*args)
    ops.ggml_dense_blas(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = ops.ggml_dense_blas(*args)
    float_x.mul_(0.5)
    graph.replay()
    torch.testing.assert_close(
        actual, float_x.half().float() @ float_w.float().T, atol=0.002, rtol=0.02
    )


@pytest.mark.parametrize("invalid", ["type", "shape", "ids_dtype", "stride"])
def test_moe_support_query_and_execution_reject_same_invalid_inputs(invalid):
    w = torch.zeros((3, 128, 256), device="cuda", dtype=torch.float16)
    x = torch.zeros((9, 256), device="cuda", dtype=torch.float16)
    ids = torch.zeros((9, 1), device="cuda", dtype=torch.int32)
    q = Q.F16
    if invalid == "type":
        q = Q.F32
    elif invalid == "shape":
        x = x[:, :-1].contiguous()
    elif invalid == "ids_dtype":
        ids = ids.long()
    else:
        w = w.transpose(1, 2)
    args = (x, w, ids, int(q), 128, 1, 9)
    assert ops.moe_supported_methods(*args) == M.NONE
    assert ops.moe_select_method(*args) == M.NONE
    with pytest.raises(RuntimeError, match="VLLM_GGUF_MOE_NOT_ELIGIBLE"):
        ops.ggml_moe_mmvf(*args)


def test_blas_configuration_does_not_disable_non_blas_methods(monkeypatch):
    monkeypatch.setenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE", "invalid")
    w = torch.zeros((128, 256), device="cuda", dtype=torch.float16)
    x = torch.zeros((1, 256), device="cuda", dtype=torch.float16)
    args = (w, x, int(Q.F16), 128)
    assert ops.dense_supported_methods(*args) & M.MMVF
    ops.ggml_dense_mmvf(*args)
    with pytest.raises(RuntimeError, match="method cannot run"):
        ops.ggml_dense_blas(*args)


@pytest.mark.parametrize(
    "q,tokens",
    [
        (Q.F16, 9),
        (Q.F16, 640),
        (Q.F16, 641),
        (Q.BF16, 9),
        (Q.BF16, 1408),
        (Q.BF16, 1409),
    ],
)
@torch.inference_mode()
def test_moe_float_fallback_avoids_host_blas_for_small_routes(q, tokens):
    # Odd rows rule out MMF on every device; MMVF still supports the layout.
    dtype = torch.bfloat16 if q == Q.BF16 else torch.float16
    w = torch.randn((3, 129, 256), device="cuda", dtype=dtype) * 0.03
    x = torch.randn((tokens, 256), device="cuda", dtype=torch.float16) * 0.1
    ids = (
        (torch.arange(tokens, device="cuda")[:, None] + torch.arange(2, device="cuda"))
        % 3
    ).int()
    args = (x, w, ids, int(q), 129, 2, tokens)
    supported = ops.moe_supported_methods(*args)
    assert not supported & M.MMF
    assert supported & M.MMVF and supported & M.BLAS
    route_limit = (
        2816 if q == Q.BF16 and torch.cuda.get_device_capability()[0] < 8 else 1280
    )
    assert ops.moe_select_method(*args) == (
        M.MMVF if tokens * 2 <= route_limit else M.BLAS
    )
    torch.testing.assert_close(
        ops.ggml_moe(*args), _reference(x, w, ids), atol=0.004, rtol=0.04
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert ops.moe_select_method(*args) == M.MMVF
        captured = ops.ggml_moe(*args)
    ids.copy_((ids + 1) % 3)
    graph.replay()
    torch.testing.assert_close(captured, _reference(x, w, ids), atol=0.004, rtol=0.04)


@torch.inference_mode()
def test_moe_alignment_precedes_grouping_in_eager_and_graph(monkeypatch):
    from vllm.model_executor.layers.fused_moe import fused_moe

    w, dense = _quantized(moe=True)
    tokens = 8193
    x = torch.randn((tokens, 256), device="cuda", dtype=torch.float16) * 0.1
    ids = (torch.arange(tokens, device="cuda") % 3).int()[:, None].contiguous()
    args = (x, w, ids, int(Q.Q8_0), 128, 1, tokens)
    assert ops.moe_select_method(*args, allow_aligned=True) == M.MMQ_ALIGNED
    assert ops.moe_select_method(*args, allow_aligned=False) == M.GROUPED_DENSE
    original = fused_moe.moe_align_block_size
    plans = []

    def record(*args, **kwargs):
        plans.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(fused_moe, "moe_align_block_size", record)
    eager = ops.ggml_moe(*args)
    assert len(plans) == 1
    torch.testing.assert_close(eager, _reference(x, dense, ids), atol=0.004, rtol=0.04)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert not ops.moe_supported_methods(*args) & M.GROUPED_DENSE
        assert ops.moe_select_method(*args, allow_aligned=True) == M.MMQ_ALIGNED
        captured = ops.ggml_moe(*args)
    assert len(plans) == 2
    ids.copy_((ids + 1) % 3)
    graph.replay()
    torch.testing.assert_close(
        captured, _reference(x, dense, ids), atol=0.004, rtol=0.04
    )
