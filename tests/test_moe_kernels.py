"""MoE weight loading, quantized and floating kernels, and CUDA runtime."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType as Q

from tests.helpers_upstream import (
    TEMPLATE_EXTRA_TYPES,
    make_padded_moe_weight,
    make_padded_weight,
    upstream_padding_bytes,
)
from vllm_gguf_plugin.quantization.fused_moe import GGUFMoEMethod
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf

cuda_mark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
IQ_TYPES = (Q.IQ2_XXS, Q.IQ2_S, Q.IQ3_S, Q.IQ3_XXS, Q.IQ4_XS)


# Weight loading and dtype


@pytest.mark.parametrize(
    "params_dtype,expected_type",
    [
        (torch.float16, Q.F16),
        (torch.bfloat16, Q.BF16),
    ],
)
def test_float_moe_weights_match_runtime_dtype_and_type(
    params_dtype: torch.dtype, expected_type: Q, monkeypatch
):
    monkeypatch.setenv("VLLM_GGUF_CUDA_DENSE_KERNEL", "legacy")
    layer = torch.nn.Module()
    layer.moe_config = SimpleNamespace(moe_parallel_config=SimpleNamespace(tp_size=1))

    def base_weight_loader(
        param, loaded_weight, weight_name, shard_id, expert_id, return_success=False
    ):
        del weight_name, return_success
        rows = loaded_weight.shape[0]
        start = rows if shard_id == "w3" else 0
        param.data[expert_id, start : start + rows].copy_(loaded_weight)

    method = GGUFMoEMethod(None, None)
    method.create_weights(
        layer,
        num_experts=1,
        hidden_size=4,
        intermediate_size_per_partition=4,
        params_dtype=params_dtype,
        weight_loader=base_weight_loader,
    )

    source = torch.arange(16, dtype=torch.float32).reshape(4, 4).to(torch.bfloat16)
    for param_name, shard_id in (
        ("w13_weight", "w1"),
        ("w13_weight", "w3"),
        ("w2_weight", "w2"),
    ):
        param = getattr(layer, param_name)
        param.weight_loader(param, source, param_name, shard_id, 0)

    method.process_weights_after_loading(layer)

    for param_name in ("w13_weight", "w2_weight"):
        weight = getattr(layer, param_name)
        weight_type = getattr(layer, f"{param_name}_type")
        assert weight.dtype == params_dtype
        assert weight_type.weight_type == expected_type
        assert weight_type.item() == expected_type

    x = torch.ones((1, 4), dtype=params_dtype)
    out = _fused_mul_mat_gguf(
        x,
        layer.w13_weight[0, :4],
        layer.w13_weight_type.weight_type,
    )
    torch.testing.assert_close(out, x @ source.to(params_dtype).T)


# Kernel correctness and storage


@cuda_mark
@torch.inference_mode()
@pytest.mark.parametrize("types", [(Q.Q4_0, Q.Q8_0), (Q.Q8_0, Q.Q8_0)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fused_moe_reuses_aligned_plan(monkeypatch, types, dtype):
    from vllm.model_executor.layers.fused_moe import fused_moe as vllm_moe

    from vllm_gguf_plugin.quantization.fused_moe import _fused_moe_gguf

    experts, tokens, k, hidden, top_k = 4, 513, 256, 256, 2
    rng = np.random.default_rng(43)
    packed = []
    for q, rows, cols in ((types[0], 2 * hidden, k), (types[1], k, hidden)):
        source = rng.standard_normal((experts * rows, cols), dtype=np.float32) * 0.03
        raw = gguf.quantize(source, q).reshape(experts, rows, -1)
        packed.append(make_padded_moe_weight(raw, q, cols))
    w1, w2 = packed
    x = torch.randn((tokens, k), device="cuda", dtype=dtype) * 0.1
    ids = (
        torch.arange(tokens, device="cuda", dtype=torch.int32)[:, None]
        + torch.arange(top_k, device="cuda", dtype=torch.int32)[None, :]
    ) % experts
    weights = torch.full((tokens, top_k), 1 / top_k, device="cuda", dtype=dtype)
    monkeypatch.setenv("VLLM_GGUF_CUDA_MOE_KERNEL", "upstream")
    plans = []
    original = vllm_moe.moe_align_block_size

    def record(*args, **kwargs):
        plan = original(*args, **kwargs)
        plans.append(plan)
        return plan

    monkeypatch.setattr(vllm_moe, "moe_align_block_size", record)

    def run():
        return _fused_moe_gguf(
            x, w1, w2, weights, ids, int(types[0]), int(types[1]), "silu"
        )

    def check(output):
        z = torch.ops._C_gguf.ggml_moe_mmq(
            x, w1, ids, int(types[0]), 2 * hidden, top_k, tokens
        )
        gate, up = z.chunk(2, -1)
        h = (torch.nn.functional.silu(gate) * up).contiguous()
        y = torch.ops._C_gguf.ggml_moe_mmq(
            h, w2, ids.reshape(-1, 1), int(types[1]), k, 1, tokens * top_k
        )
        expected = (y.view(tokens, top_k, k) * weights[..., None]).sum(1)
        torch.testing.assert_close(output, expected, atol=0.002, rtol=0.02)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        output = run()
    same_type = types[0] == types[1]
    assert len(plans) == (1 if same_type else 2)
    expected_op = "_C_gguf::ggml_moe_mmq_aligned"
    assert sum(e.name == expected_op for e in profile.events()) == 2
    check(output)
    plans.clear()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    assert len(plans) == (1 if same_type else 2)
    ids.copy_((ids + 2) % experts)
    x.mul_(0.5)
    graph.replay()
    check(output)


@cuda_mark
@torch.inference_mode()
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("top_k", [1, 8])
def test_moe_mmq_aligned_dynamic_routes(dtype, top_k):
    # Odd expert/row counts, padded K, empty experts and changing graph routes.
    _check_mmq_dynamic_routes(257, 137, top_k, dtype)


@cuda_mark
@torch.inference_mode()
@pytest.mark.parametrize("experts", [1, 1024, 1025])
def test_moe_mmq_aligned_expert_limits(experts):
    # Above vLLM's alignment limit, keep the original upstream fallback.
    _check_mmq_dynamic_routes(experts, 17, 1, torch.float32, k=512)


def _check_mmq_dynamic_routes(experts, row, top_k, dtype, k=256):
    from vllm_gguf_plugin import ops

    tokens = 137
    source = (
        np.random.default_rng(97).standard_normal((experts, row, k), dtype=np.float32)
        * 0.1
    )
    packed = gguf.quantize(source.reshape(-1, k), Q.Q8_0).reshape(experts, row, -1)
    weight = make_padded_moe_weight(packed, Q.Q8_0, k)
    dense = (
        torch.from_numpy(gguf.dequantize(packed.reshape(-1, packed.shape[-1]), Q.Q8_0))
        .cuda()
        .view(experts, row, k)
    )
    x = torch.randn((tokens, k), device="cuda", dtype=dtype) * 0.1
    ids = torch.empty((tokens, top_k), device="cuda", dtype=torch.int32)
    offsets = torch.arange(top_k, device="cuda", dtype=torch.int32)[None, :]
    ids.copy_((experts - top_k + offsets).expand_as(ids))

    def run():
        if experts <= 992:
            from vllm.model_executor.layers.fused_moe.fused_moe import (
                moe_align_block_size,
            )

            plan = moe_align_block_size(ids, 16, experts, pad_sorted_ids=True)
            return ops.ggml_moe_mmq_aligned(
                x,
                weight,
                plan[0],
                int(Q.Q8_0),
                row,
                top_k,
                tokens,
                expert_ids=plan[1],
                padded_count=plan[2],
            )
        return ops.ggml_moe_mmq(x, weight, ids, int(Q.Q8_0), row, top_k, tokens)

    def check(output):
        reference = torch.einsum("tk,tjrk->tjr", x.float(), dense[ids.long()])
        assert output.dtype == dtype
        torch.testing.assert_close(
            output.view(tokens, top_k, row).float(), reference, atol=0.015, rtol=0.05
        )

    check(run())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    for spread in (True, False):
        x.copy_(torch.randn_like(x) * 0.1)
        base = (
            torch.arange(tokens, device="cuda", dtype=torch.int32)[:, None] * 37
            if spread
            else 0
        )
        ids.copy_(((base + offsets) % experts).expand_as(ids))
        graph.replay()
        check(output)


@cuda_mark
@torch.inference_mode()
@pytest.mark.parametrize("kernel", ["mmvq", "mmq", "grouped_dense"])
@pytest.mark.parametrize("tokens", [8, 129, 8193])
def test_moe_explicit_kernel_reference_and_graph(kernel, tokens):
    from vllm_gguf_plugin import ops

    experts, row, k, top_k = 4, 128, 256, 2
    source = (
        np.random.default_rng(31).standard_normal((experts, row, k), dtype=np.float32)
        * 0.1
    )
    packed = gguf.quantize(source.reshape(-1, k), Q.Q8_0).reshape(experts, row, -1)
    weight = make_padded_weight(packed, Q.Q8_0, k)
    dense = (
        torch.from_numpy(gguf.dequantize(packed.reshape(-1, packed.shape[-1]), Q.Q8_0))
        .cuda()
        .view(experts, row, k)
    )
    x = torch.randn((tokens, k), device="cuda", dtype=torch.float32) * 0.1
    ids = (
        torch.arange(tokens, device="cuda", dtype=torch.int32)[:, None]
        + torch.arange(top_k, device="cuda", dtype=torch.int32)[None, :]
    ) % experts
    op = getattr(ops, f"ggml_moe_{kernel}")
    args = (x, weight, ids, int(Q.Q8_0), row, top_k, tokens)
    sample = torch.tensor(sorted({0, 7, tokens // 2, tokens - 1}), device="cuda")

    def check(output):
        reference = torch.einsum("tk,tjrk->tjr", x[sample], dense[ids[sample].long()])
        torch.testing.assert_close(
            output.view(tokens, top_k, row)[sample], reference, atol=0.02, rtol=0.05
        )

    if tokens == 8193 and kernel != "grouped_dense":
        # Verify the actual kernel above the automatic grouped-dense threshold.
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            output = op(*args)
            torch.cuda.synchronize()
        expected = "mul_mat_vec_q_moe" if kernel == "mmvq" else "mul_mat_q<"
        assert any(expected in event.name for event in profile.events())
        check(output)
    else:
        check(op(*args))
    if tokens == 8:
        check(ops.ggml_moe_upstream(*args))
        # opcheck clones lose the padding tail behind packed weight views.
        check(torch.compile(op, backend="eager", fullgraph=True)(*args))
    graph = torch.cuda.CUDAGraph()
    if kernel == "grouped_dense":
        with (
            pytest.raises(RuntimeError, match="requested MoE method cannot run"),
            torch.cuda.graph(graph),
        ):
            op(*args)
        return
    with torch.cuda.graph(graph):
        captured = op(*args)
    ids.copy_((ids + 1) % experts)
    graph.replay()
    check(captured)


@cuda_mark
@pytest.mark.parametrize(
    "kernel,quant_type,tokens,message",
    [
        ("mmvq", Q.F16, 8, "requested MoE method cannot run"),
        ("mmq", Q.F16, 8, "requested MoE method cannot run"),
        ("mmq", Q.IQ1_M, 8, "requested MoE method cannot run"),
    ],
)
def test_moe_explicit_kernel_rejects_unsupported(kernel, quant_type, tokens, message):
    from vllm_gguf_plugin import ops

    experts, row, k = 2, 128, 256
    if quant_type == Q.F16:
        weight = torch.zeros((experts, row, k), device="cuda", dtype=torch.float16)
    else:
        block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]
        packed = np.zeros((experts, row, k // block_size * type_size), dtype=np.uint8)
        weight = make_padded_weight(packed, quant_type, k)
    x = torch.zeros((tokens, k), device="cuda", dtype=torch.float32)
    ids = torch.zeros((tokens, 1), device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match=message):
        getattr(ops, f"ggml_moe_{kernel}")(
            x, weight, ids, int(quant_type), row, 1, tokens
        )


@cuda_mark
@pytest.mark.parametrize("invalid", ["shape", "dtype", "stride"])
def test_moe_mmq_alignment_preserves_input_contract(invalid):
    from vllm_gguf_plugin import ops

    tokens, k, row = 8, 256, 128
    weight = make_padded_moe_weight(
        np.zeros((2, row, k // 32 * 34), dtype=np.uint8), Q.Q8_0, k
    )
    x = torch.zeros((tokens, k), device="cuda")
    ids = torch.zeros((tokens, 2), device="cuda", dtype=torch.int32)
    if invalid == "shape":
        ids = ids[:, :1].contiguous()
    elif invalid == "dtype":
        ids = ids.long()
    else:
        ids = torch.zeros((tokens, 4), device="cuda", dtype=torch.int32)[:, ::2]
    with pytest.raises(RuntimeError, match="VLLM_GGUF_MOE_NOT_ELIGIBLE"):
        ops.ggml_moe_mmq(x, weight, ids, int(Q.Q8_0), row, 2, tokens)


@cuda_mark
@pytest.mark.parametrize(
    "invalid",
    [
        "expert_only",
        "count_only",
        "route_shape",
        "route_dtype",
        "route_stride",
        "capacity",
        "expert_dtype",
        "count_dtype",
        "count_size",
    ],
)
def test_moe_mmq_shared_plan_contract(invalid):
    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

    tokens, k, row = 8, 256, 128
    weight = make_padded_moe_weight(
        np.zeros((2, row, k // 32 * 34), dtype=np.uint8), Q.Q8_0, k
    )
    x = torch.zeros((tokens, k), device="cuda")
    ids = torch.zeros((tokens, 2), device="cuda", dtype=torch.int32)
    routes, experts, count = moe_align_block_size(ids, 16, 2, pad_sorted_ids=True)
    if invalid == "expert_only":
        count = None
    elif invalid == "count_only":
        experts = None
    elif invalid == "route_shape":
        routes = routes.view(-1, 1)
    elif invalid == "route_dtype":
        routes = routes.long()
    elif invalid == "route_stride":
        routes = torch.stack((routes, routes), dim=1)[:, 0]
    elif invalid == "capacity":
        routes = routes[:8]
    elif invalid == "expert_dtype":
        experts = experts.long()
    elif invalid == "count_dtype":
        count = count.long()
    else:
        count = count.expand(2).contiguous()
    with pytest.raises(RuntimeError, match="VLLM_GGUF_MOE_NOT_ELIGIBLE"):
        torch.ops._C_gguf.ggml_moe_mmq_aligned(
            x,
            weight,
            routes,
            int(Q.Q8_0),
            row,
            2,
            tokens,
            expert_ids=experts,
            padded_count=count,
        )


def test_moe_process_weights_after_loading_pads_3d_weights():
    # k=1184 is not a multiple of the 512-value MATRIX_ROW_PADDING, so a
    # storage tail must be reserved behind both 3D weights.
    n, k, experts = 37, 1184, 3
    quant_type = Q.Q4_0
    block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]
    packed_row = k // block_size * type_size
    layer = torch.nn.Module()
    for name in ("w13_weight", "w2_weight"):
        param = torch.nn.Parameter(
            torch.randint(0, 256, (experts, n, packed_row), dtype=torch.uint8),
            requires_grad=False,
        )
        layer.register_parameter(name, param)
    layer.w13_weight_type = SimpleNamespace(weight_type=int(Q.Q4_0))
    layer.w2_weight_type = SimpleNamespace(weight_type=int(Q.Q4_0))

    GGUFMoEMethod(None, None).process_weights_after_loading(layer)

    logical_bytes = layer.w13_weight.numel() * layer.w13_weight.element_size()
    padding_bytes = upstream_padding_bytes(quant_type, k)
    assert padding_bytes > 0
    assert layer.w13_weight.untyped_storage().nbytes() >= logical_bytes + padding_bytes
    assert layer.w2_weight.untyped_storage().nbytes() >= logical_bytes + padding_bytes
    assert tuple(layer.w13_weight.shape) == (experts, n, packed_row)


@cuda_mark
@torch.inference_mode()
@pytest.mark.parametrize(
    "dtype,quant_type,atol",
    [
        (torch.float32, Q.F32, 1e-5),
        (torch.float16, Q.F16, 2e-3),
        (torch.bfloat16, Q.BF16, 2e-2),
    ],
)
@pytest.mark.parametrize("route,tokens", [("mmvf", 1), ("mmf", 9)])
def test_moe_float_mmvf_mmf_reference(dtype, quant_type, atol, route, tokens):
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    experts, top_k = 3, 2
    rows, k = (37, 258) if route == "mmvf" else (64, 256)
    torch.manual_seed(25)
    weight = torch.randn((experts, rows, k), device="cuda", dtype=dtype) * 0.1
    x = torch.randn((tokens, k), device="cuda", dtype=dtype) * 0.1
    ids = torch.tensor(
        [[t % experts, (t + 1) % experts] for t in range(tokens)],
        device="cuda",
        dtype=torch.int32,
    )

    if route == "mmf" and torch.version.hip is not None:
        pytest.skip("NVIDIA MMF architecture expectation")
    if route == "mmf":
        major, _ = torch.cuda.get_device_capability()
        required_major = 7 if dtype == torch.float16 else 8
        if major < required_major:
            from vllm_gguf_plugin.kernel_support import MOE_NOT_ELIGIBLE_MARKER

            with pytest.raises(RuntimeError, match=MOE_NOT_ELIGIBLE_MARKER):
                torch.ops._C_gguf.ggml_moe_mmf(
                    x, weight, ids, int(quant_type), rows, top_k, tokens
                )
            return

    output = torch.ops._C_gguf.ggml_moe_upstream(
        x, weight, ids, int(quant_type), rows, top_k, tokens
    )
    reference = torch.stack(
        [
            weight[int(ids[t, j])].float() @ x[t].float()
            for t in range(tokens)
            for j in range(top_k)
        ]
    ).to(dtype)
    assert output.shape == (tokens * top_k, rows)
    # Ampere's F32 MMF uses TF32 tensor-core instructions.
    reference_atol = 5e-3 if dtype == torch.float32 and route == "mmf" else atol
    torch.testing.assert_close(output, reference, atol=reference_atol, rtol=0)


@cuda_mark
@torch.inference_mode()
def test_moe_float_mmf_graph_replay():
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    if torch.cuda.get_device_capability()[0] < 7:
        pytest.skip("float16 MMF requires Volta or newer")
    experts, rows, k, tokens, top_k = 4, 64, 256, 17, 2
    torch.manual_seed(26)
    weight = torch.randn((experts, rows, k), device="cuda", dtype=torch.float16) * 0.1
    x = torch.randn((tokens, k), device="cuda", dtype=torch.float16) * 0.1
    # RoutedExperts/topk selects distinct experts for each token. Upstream's
    # compact MMF helper relies on that contract when tokens > 16.
    ids = torch.tensor(
        [[t % experts, (t + 1) % experts] for t in range(tokens)],
        device="cuda",
        dtype=torch.int32,
    )
    op = torch.ops._C_gguf.ggml_moe_upstream

    def reference():
        return torch.stack(
            [
                weight[int(ids[t, j])].float() @ x[t].float()
                for t in range(tokens)
                for j in range(top_k)
            ]
        ).half()

    eager = op(x, weight, ids, int(Q.F16), rows, top_k, tokens)
    torch.testing.assert_close(eager, reference(), atol=0.06, rtol=0.06)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(x, weight, ids, int(Q.F16), rows, top_k, tokens)
    ids.copy_(
        torch.tensor(
            [[(t + 2) % experts, (t + 3) % experts] for t in range(tokens)],
            device="cuda",
            dtype=torch.int32,
        )
    )
    graph.replay()
    torch.testing.assert_close(captured, reference(), atol=0.06, rtol=0.06)


@cuda_mark
@torch.inference_mode()
@pytest.mark.parametrize(
    "rows,tokens,top_k", [(64, 513, 2), (1056, 129, 2), (256, 1025, 1)]
)
def test_moe_float_chunked_mmf_reference_and_graph(rows, tokens, top_k):
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    if torch.cuda.get_device_capability()[0] < 7:
        pytest.skip("float16 MMF requires Volta or newer")
    experts, k = 4, 256
    torch.manual_seed(28)
    weight = torch.randn((experts, rows, k), device="cuda", dtype=torch.float16) * 0.1
    x = torch.randn((tokens, k), device="cuda", dtype=torch.float16) * 0.1
    ids = torch.tensor(
        [[(t + j) % experts for j in range(top_k)] for t in range(tokens)],
        device="cuda",
        dtype=torch.int32,
    )
    op = torch.ops._C_gguf.ggml_moe_upstream

    def reference():
        all_experts = torch.stack(
            [x.float() @ weight[expert].float().T for expert in range(experts)],
            dim=1,
        )
        selected = all_experts.gather(1, ids.long().unsqueeze(-1).expand(-1, -1, rows))
        return selected.reshape(tokens * top_k, rows).half()

    eager = op(x, weight, ids, int(Q.F16), rows, top_k, tokens)
    torch.testing.assert_close(eager, reference(), atol=0.06, rtol=0.06)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(x, weight, ids, int(Q.F16), rows, top_k, tokens)
    ids.copy_(
        torch.tensor(
            [[(t + j + 2) % experts for j in range(top_k)] for t in range(tokens)],
            device="cuda",
            dtype=torch.int32,
        )
    )
    graph.replay()
    torch.testing.assert_close(captured, reference(), atol=0.06, rtol=0.06)


@cuda_mark
@torch.inference_mode()
def test_moe_float_without_mmf_uses_blas_and_rejects_graph():
    from vllm_gguf_plugin import ops
    from vllm_gguf_plugin.kernel_support import KernelMethod

    x = torch.randn((17, 255), device="cuda", dtype=torch.float16) * 0.1
    weight = torch.randn((2, 63, 255), device="cuda", dtype=torch.float16) * 0.1
    ids = torch.zeros((17, 1), device="cuda", dtype=torch.int32)
    args = (x, weight, ids, int(Q.F16), 63, 1, 17)
    assert ops.moe_supported_methods(*args) == (
        KernelMethod.BLAS | KernelMethod.GROUPED_DENSE
    )
    torch.testing.assert_close(
        ops.ggml_moe(*args), x @ weight[0].T, atol=0.002, rtol=0.02
    )
    with (
        pytest.raises(RuntimeError, match="VLLM_GGUF_MOE_NOT_ELIGIBLE"),
        torch.cuda.graph(torch.cuda.CUDAGraph()),
    ):
        ops.ggml_moe(*args)


@cuda_mark
@torch.inference_mode()
def test_moe_float_auto_uses_upstream_blas(monkeypatch):
    from vllm_gguf_plugin.quantization.fused_moe import _fused_moe_gguf

    monkeypatch.setenv("VLLM_GGUF_CUDA_MOE_KERNEL", "auto")
    x = torch.zeros((17, 255), device="cuda", dtype=torch.float16)
    w1 = torch.zeros((2, 64, 255), device="cuda", dtype=torch.float16)
    w2 = torch.zeros((2, 255, 32), device="cuda", dtype=torch.float16)
    ids = torch.tensor([[0, 1]] * 17, device="cuda", dtype=torch.int32)
    topk_weights = torch.full((17, 2), 0.5, device="cuda", dtype=torch.float16)
    actual = _fused_moe_gguf(
        x, w1, w2, topk_weights, ids, int(Q.F16), int(Q.F16), "silu"
    )
    torch.testing.assert_close(actual, torch.zeros_like(x))


@cuda_mark
@torch.inference_mode()
def test_moe_float_two_projection_fused_reference(monkeypatch):
    import torch.nn.functional as F

    from vllm_gguf_plugin.quantization.fused_moe import _fused_moe_gguf

    monkeypatch.setenv("VLLM_GGUF_CUDA_MOE_KERNEL", "upstream")
    torch.manual_seed(27)
    tokens, experts, top_k, hidden, intermediate = 5, 4, 2, 256, 64
    x = torch.randn((tokens, hidden), device="cuda", dtype=torch.float16) * 0.1
    w1 = (
        torch.randn(
            (experts, 2 * intermediate, hidden), device="cuda", dtype=torch.float16
        )
        * 0.1
    )
    w2 = (
        torch.randn((experts, hidden, intermediate), device="cuda", dtype=torch.float16)
        * 0.1
    )
    ids = torch.tensor(
        [[t % experts, (t + 1) % experts] for t in range(tokens)],
        device="cuda",
        dtype=torch.int32,
    )
    route_weights = torch.softmax(
        torch.randn((tokens, top_k), device="cuda", dtype=torch.float32), dim=1
    ).half()

    actual = _fused_moe_gguf(
        x, w1, w2, route_weights, ids, int(Q.F16), int(Q.F16), "silu"
    )
    reference = torch.zeros_like(x)
    for t in range(tokens):
        for j in range(top_k):
            expert = int(ids[t, j])
            projected = w1[expert].float() @ x[t].float()
            activated = F.silu(projected[:intermediate]) * projected[intermediate:]
            down = w2[expert].float() @ activated.half().float()
            reference[t] += down.half() * route_weights[t, j]
    torch.testing.assert_close(actual, reference, atol=0.06, rtol=0.06)


@cuda_mark
@torch.inference_mode()
def test_moe_float_auto_chunks_second_projection(monkeypatch):
    import vllm_gguf_plugin.quantization as quantization
    from vllm_gguf_plugin.quantization.fused_moe import _fused_moe_gguf

    monkeypatch.setenv("VLLM_GGUF_CUDA_MOE_KERNEL", "auto")
    monkeypatch.setattr(
        quantization,
        "fused_mul_mat_gguf",
        lambda *args: pytest.fail("unexpected per-token MoE fallback"),
    )
    tokens, experts, top_k, hidden, intermediate = 257, 4, 2, 256, 64
    x = torch.randn((tokens, hidden), device="cuda", dtype=torch.float16) * 0.1
    w1 = (
        torch.randn(
            (experts, 2 * intermediate, hidden), device="cuda", dtype=torch.float16
        )
        * 0.1
    )
    w2 = (
        torch.randn((experts, hidden, intermediate), device="cuda", dtype=torch.float16)
        * 0.1
    )
    ids = torch.tensor(
        [[t % experts, (t + 1) % experts] for t in range(tokens)],
        device="cuda",
        dtype=torch.int32,
    )
    route_weights = torch.full((tokens, top_k), 0.5, device="cuda", dtype=torch.float16)

    output = _fused_moe_gguf(
        x, w1, w2, route_weights, ids, int(Q.F16), int(Q.F16), "silu"
    )
    assert output.shape == x.shape
    assert bool(torch.isfinite(output).all())


@cuda_mark
@torch.inference_mode()
def test_moe_template_instance_types(monkeypatch):
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    monkeypatch.setenv("VLLM_GGUF_CUDA_MOE_KERNEL", "upstream")
    n, k, experts, top_k = 37, 512, 3, 2
    ids = torch.tensor([[0, 1]] * 16, dtype=torch.int32, device="cuda")
    for quant_type in TEMPLATE_EXTRA_TYPES:
        block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]
        packed_row = k // block_size * type_size
        weight = torch.zeros((experts, n, packed_row), dtype=torch.uint8, device="cuda")
        for tokens in (2, 16):
            x = torch.zeros((tokens, k), dtype=torch.float32, device="cuda")
            output = torch.ops._C_gguf.ggml_moe_a8_upstream(
                x, weight, ids[:tokens], int(quant_type), n, top_k, tokens
            )
            assert output.shape == (tokens * top_k, n)
            assert bool(torch.isfinite(output).all())
            torch.testing.assert_close(output, torch.zeros_like(output))


@cuda_mark
@torch.inference_mode()
@pytest.mark.parametrize(
    "tokens,top_k",
    [(32, 8), (33, 8), (256, 1), (257, 1), (128, 8), (129, 8), (1024, 1), (1025, 1)],
)
def test_moe_graph_at_route_count_boundary(monkeypatch, tokens, top_k):
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    monkeypatch.setenv("VLLM_GGUF_CUDA_MOE_KERNEL", "upstream")
    quant_type = Q.Q8_0
    experts, row, k = 8, 128, 256
    source = np.random.default_rng(19).standard_normal(
        (experts, row, k), dtype=np.float32
    )
    packed = gguf.quantize(source.reshape(-1, k), quant_type).reshape(experts, row, -1)
    weight = make_padded_weight(packed, quant_type, k)
    dense = (
        torch.from_numpy(
            gguf.dequantize(packed.reshape(-1, packed.shape[-1]), quant_type)
        )
        .cuda()
        .view(experts, row, k)
    )
    x = torch.randn((tokens, k), device="cuda", dtype=torch.float32)
    ids = (
        torch.arange(tokens, device="cuda", dtype=torch.int32)[:, None]
        + torch.arange(top_k, device="cuda", dtype=torch.int32)[None, :]
    ) % experts
    op = torch.ops._C_gguf.ggml_moe_a8_upstream
    sample = torch.tensor([0, tokens // 2, tokens - 1], device="cuda")

    def check(output):
        reference = torch.einsum("sk,sjrk->sjr", x[sample], dense[ids[sample].long()])
        torch.testing.assert_close(
            output.view(tokens, top_k, row)[sample], reference, atol=1.5, rtol=0.2
        )

    check(op(x, weight, ids, int(quant_type), row, top_k, tokens))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(x, weight, ids, int(quant_type), row, top_k, tokens)
    ids.copy_((ids + 1) % experts)
    graph.replay()
    check(captured)


@cuda_mark
@torch.inference_mode()
@pytest.mark.parametrize("tokens,top_k", [(8192, 1), (8193, 1), (8193, 2)])
def test_moe_grouped_threshold_eager_and_graph_fallback(monkeypatch, tokens, top_k):
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    monkeypatch.setenv("VLLM_GGUF_CUDA_MOE_KERNEL", "upstream")
    quant_type = Q.Q8_0
    experts, row, k = 4, 64, 256
    source = np.random.default_rng(37).standard_normal(
        (experts, row, k), dtype=np.float32
    )
    packed = gguf.quantize(source.reshape(-1, k), quant_type).reshape(experts, row, -1)
    weight = make_padded_weight(packed, quant_type, k)
    dense = (
        torch.from_numpy(
            gguf.dequantize(packed.reshape(-1, packed.shape[-1]), quant_type)
        )
        .cuda()
        .view(experts, row, k)
    )
    x = torch.randn((tokens, k), device="cuda", dtype=torch.float32)
    ids = (
        torch.arange(tokens, device="cuda", dtype=torch.int32)[:, None]
        + torch.arange(top_k, device="cuda", dtype=torch.int32)[None, :]
    ) % experts
    op = torch.ops._C_gguf.ggml_moe_upstream
    sample = torch.tensor(
        sorted({0, 7, min(8192, tokens - 1), tokens - 1}), device="cuda"
    )

    def check(output):
        reference = torch.einsum("sk,sjrk->sjr", x[sample], dense[ids[sample].long()])
        torch.testing.assert_close(
            output.view(tokens, top_k, row)[sample], reference, atol=1.5, rtol=0.2
        )

    check(op(x, weight, ids, int(quant_type), row, top_k, tokens))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(x, weight, ids, int(quant_type), row, top_k, tokens)
    ids[sample] = (ids[sample] + 1) % experts
    graph.replay()
    check(captured)


@cuda_mark
@torch.inference_mode()
@pytest.mark.parametrize("op_name", ["ggml_moe_upstream", "ggml_moe_mmq"])
def test_moe_mmq_chunks_at_mmid_shared_memory_limit(monkeypatch, op_name):
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    monkeypatch.setenv("VLLM_GGUF_CUDA_MOE_KERNEL", "upstream")
    quant_type = Q.Q8_0
    experts, row, k, top_k = 4, 128, 256, 2
    shared_limit = torch.cuda.get_device_properties().shared_memory_per_block_optin
    chunk_size = min(shared_limit // 4, (1 << 22) - 1)
    tokens = chunk_size + 2
    source = np.random.default_rng(29).standard_normal(
        (experts, row, k), dtype=np.float32
    )
    packed = gguf.quantize(source.reshape(-1, k), quant_type).reshape(experts, row, -1)
    weight = make_padded_weight(packed, quant_type, k)
    dense = (
        torch.from_numpy(
            gguf.dequantize(packed.reshape(-1, packed.shape[-1]), quant_type)
        )
        .cuda()
        .view(experts, row, k)
    )
    x = torch.randn((tokens, k), device="cuda", dtype=torch.float32)
    first_expert = torch.randint(
        0, experts, (tokens,), device="cuda", dtype=torch.int32
    )
    ids = torch.stack((first_expert, (first_expert + 1) % experts), dim=1)
    from vllm_gguf_plugin import ops

    op = (
        ops.ggml_moe_mmq
        if op_name == "ggml_moe_mmq"
        else getattr(torch.ops._C_gguf, op_name)
    )

    def check_sample(output):
        sample = torch.tensor(
            [0, chunk_size - 7, chunk_size - 6, chunk_size, tokens - 1],
            device="cuda",
        )
        actual = output.view(tokens, top_k, row)[sample]
        reference = torch.einsum("stnk,sk->stn", dense[ids[sample].long()], x[sample])
        torch.testing.assert_close(actual, reference, atol=1.5, rtol=0.2)

    check_sample(op(x, weight, ids, int(quant_type), row, top_k, tokens))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(x, weight, ids, int(quant_type), row, top_k, tokens)
    ids[chunk_size, :] = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    graph.replay()
    check_sample(captured)


@cuda_mark
@torch.inference_mode()
def test_moe_iq1_m_chunked_mmvq_and_graph_replay(monkeypatch):
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    monkeypatch.setenv("VLLM_GGUF_CUDA_MOE_KERNEL", "upstream")
    quant_type = Q.IQ1_M
    experts, row, k, tokens, top_k = 4, 17, 256, 9, 2
    raw = np.random.default_rng(7).integers(0, 256, (experts * row, 56), dtype=np.uint8)
    raw[:, 48:56] = np.tile(np.array([0, 60], dtype=np.uint8), 4)
    weight = make_padded_weight(raw, quant_type, k).view(experts, row, 56)
    dense = (
        torch.from_numpy(gguf.dequantize(raw, quant_type)).cuda().view(experts, row, k)
    )
    x = torch.randn((tokens, k), device="cuda", dtype=torch.float32)
    ids = torch.tensor([[0, 1]] * 7 + [[0, 2]] * 2, dtype=torch.int32, device="cuda")

    def reference(routes):
        return torch.stack(
            [
                x[t].half().float() @ dense[int(routes[t, j])].half().float().T
                for t in range(tokens)
                for j in range(top_k)
            ]
        )

    op = torch.ops._C_gguf.ggml_moe_a8_upstream
    eager = op(x, weight, ids, int(quant_type), row, top_k, tokens)
    torch.testing.assert_close(eager, reference(ids), atol=0.6, rtol=0.05)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(x, weight, ids, int(quant_type), row, top_k, tokens)
    new_ids = torch.tensor(
        [[2, 1]] * 6 + [[0, 2]] * 3, dtype=torch.int32, device="cuda"
    )
    ids.copy_(new_ids)
    graph.replay()
    torch.testing.assert_close(captured, reference(new_ids), atol=0.6, rtol=0.05)


@cuda_mark
@torch.inference_mode()
def test_moe_upstream_non_padded_k_reference(monkeypatch):
    """End-to-end: k that is not a multiple of 512 still reaches the kernel."""
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    monkeypatch.setenv("VLLM_GGUF_CUDA_MOE_KERNEL", "upstream")
    # k=1184 is 2*512+160, so the row needs a non-trivial padding tail.
    n, k, experts, top_k = 37, 1184, 3, 2
    quant_type = Q.Q4_0
    block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]
    packed_row = k // block_size * type_size
    source = np.random.default_rng(7).standard_normal((experts, n, k), dtype=np.float32)
    packed = gguf.quantize(source.reshape(-1, k), quant_type).reshape(
        experts, n, packed_row
    )
    weight = make_padded_weight(packed, quant_type, k)

    tokens = 4
    x = torch.randn((tokens, k), device="cuda", dtype=torch.float32)
    ids = torch.tensor([[0, 2]] * tokens, dtype=torch.int32, device="cuda")
    output = torch.ops._C_gguf.ggml_moe_a8_upstream(
        x, weight, ids, int(quant_type), n, top_k, tokens
    )
    assert output.shape == (tokens * top_k, n)
    assert bool(torch.isfinite(output).all())

    # Expert 0 for every token: route rows 0..top_k-1 of the flat output.
    dense0 = torch.from_numpy(gguf.dequantize(packed[0], quant_type)).cuda()
    reference = x @ dense0.T
    torch.testing.assert_close(
        output.view(tokens, top_k, n)[:, 0], reference, atol=1.5, rtol=0.2
    )


# IQ projection shapes


def _zero_moe_weight(quant_type, experts, n, k):
    """Zero IQ blocks decode to zero; numerical IQ cases live in the type matrix."""
    block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]
    packed_row = k // block_size * type_size
    packed = np.zeros((experts, n, packed_row), dtype=np.uint8)
    return make_padded_moe_weight(packed, quant_type, k)


def _run_moe(quant_type, tokens, n, k, top_k):
    experts = max(4, top_k)
    weight = _zero_moe_weight(quant_type, experts, n, k)
    generator = torch.Generator(device="cuda").manual_seed(17)
    x = torch.randn((tokens, k), device="cuda", generator=generator)
    token_ids = torch.arange(tokens, device="cuda")[:, None]
    route_ids = torch.arange(top_k, device="cuda")[None, :]
    ids = ((token_ids + route_ids) % experts).to(torch.int32).contiguous()
    return torch.ops._C_gguf.ggml_moe_a8_upstream(
        x, weight, ids, int(quant_type), n, top_k, tokens
    )


@cuda_mark
@torch.inference_mode()
@pytest.mark.parametrize("quant_type", IQ_TYPES, ids=lambda q: q.name)
def test_iq_moe_w13_shape(quant_type):
    """Cover a 2048-input, 1024-output projection with eight routes."""
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    n, k = 1024, 2048
    out = _run_moe(quant_type, tokens=16, n=n, k=k, top_k=8)
    assert out.shape == (16 * 8, n)
    torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)


@cuda_mark
@torch.inference_mode()
@pytest.mark.parametrize("quant_type", IQ_TYPES, ids=lambda q: q.name)
def test_iq_moe_w2_shape(quant_type):
    """Cover a 512-input, 2048-output projection with 128 rows."""
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    n, k = 2048, 512
    out = _run_moe(quant_type, tokens=128, n=n, k=k, top_k=1)
    assert out.shape == (128, n)
    torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)


# CUDA graphs and concurrent streams


@cuda_mark
@torch.inference_mode()
@pytest.mark.parametrize("op_name", ["ggml_moe_a8_upstream", "ggml_moe_mmq"])
def test_moe_graphs_and_concurrent_streams(monkeypatch, op_name):
    """MoE projection under concurrent streams and CUDA graph capture.

    Covers the same stream contract as the dense kernel tests for the upstream MoE
    path: two Torch streams replaying the op concurrently must each observe
    only their own call's data, and a graph captured on a non-default stream
    must replay correct results (the bridge borrows the current stream, so
    capture/replay exercise the same stream binding).
    """
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    monkeypatch.setenv("VLLM_GGUF_CUDA_MOE_KERNEL", "upstream")
    n, k, experts, top_k = 64, 512, 4, 2
    quant_type = Q.Q4_0
    block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]
    packed_row = k // block_size * type_size
    source = np.random.default_rng(31).standard_normal(
        (experts, n, k), dtype=np.float32
    )
    packed = gguf.quantize(source.reshape(-1, k), quant_type).reshape(
        experts, n, packed_row
    )
    weight = make_padded_moe_weight(packed, quant_type, k)

    # Valid routing: unique expert ids per token, all in range, one case where
    # every token routes to the same pair (routing tail-tile coverage).
    tokens = 16
    ids = torch.tensor(
        [(t % experts, (t + 1) % experts) for t in range(tokens)],
        dtype=torch.int32,
        device="cuda",
    )
    inputs = [
        torch.randn(tokens, k, device="cuda", dtype=torch.float32) for _ in range(2)
    ]
    from vllm_gguf_plugin import ops

    op = (
        ops.ggml_moe_mmq
        if op_name == "ggml_moe_mmq"
        else getattr(torch.ops._C_gguf, op_name)
    )
    expected = [op(x, weight, ids, int(quant_type), n, top_k, tokens) for x in inputs]

    # Numerical reference for one stream's input: every token's every route.
    dense_ref = torch.from_numpy(
        gguf.dequantize(packed.reshape(-1, packed_row), quant_type).reshape(
            experts, n, k
        )
    ).cuda()
    # Gather each route's weight and compare its projection.
    gathered = dense_ref[ids]  # [tokens, top_k, n, k]
    routes = torch.einsum("trnk,tk->trn", gathered, inputs[0])
    torch.testing.assert_close(
        expected[0], routes.reshape(tokens * top_k, n), atol=1.5, rtol=0.2
    )

    streams = [torch.cuda.Stream() for _ in inputs]
    torch.cuda.synchronize()

    def run(i):
        with torch.inference_mode(), torch.cuda.stream(streams[i]):
            return [
                op(inputs[i], weight, ids, int(quant_type), n, top_k, tokens)
                for _ in range(10)
            ]

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(run, range(2)))
    torch.cuda.synchronize()
    for i, outputs in enumerate(results):
        for output in outputs:
            torch.testing.assert_close(output, expected[i], rtol=0, atol=0)

    graphs, outputs = [], []
    for i in range(2):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=streams[i]):
            outputs.append(
                op(inputs[i], weight, ids, int(quant_type), n, top_k, tokens)
            )
        graphs.append(graph)
    for _ in range(3):
        for i, graph in enumerate(graphs):
            with torch.cuda.stream(streams[i]):
                graph.replay()
    torch.cuda.synchronize()
    for i, output in enumerate(outputs):
        torch.testing.assert_close(output, expected[i], rtol=0, atol=0)
