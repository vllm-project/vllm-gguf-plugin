"""Dense kernel storage, numerical correctness, and CUDA runtime."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType as Q

from tests.helpers_upstream import (
    MMQ_TYPES,
    make_padded_weight,
)

cuda_mark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


# Storage and dense kernels


def test_dense_storage_padding_materialization():
    from vllm_gguf_plugin.quantization.linear import GGUFLinearMethod

    n, packed_k = 3, 144
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.randint(0, 256, (n, packed_k), dtype=torch.uint8),
        requires_grad=False,
    )
    layer.weight_type = SimpleNamespace(weight_type=Q.Q4_0)

    GGUFLinearMethod(None)._materialize_upstream_storage_padding(layer)

    assert tuple(layer.weight.shape) == (n, packed_k)
    assert layer.weight.untyped_storage().nbytes() >= n * packed_k + 144


def test_weight_loader_preallocates_upstream_storage_padding():
    from vllm_gguf_plugin.quantization.params import GGUFUninitializedWeightParameter

    weight = GGUFUninitializedWeightParameter(requires_grad=False)
    weight.gguf_weight_type_parameter = SimpleNamespace(
        weight_type=Q.Q4_0, shard_weight_type={}
    )

    weight._store(torch.ones((3, 144), dtype=torch.uint8))

    assert tuple(weight.shape) == (3, 144)
    assert weight.untyped_storage().nbytes() >= weight.numel() + 144


def test_mixed_shards_keep_individual_upstream_storage_padding(monkeypatch):
    from vllm_gguf_plugin.quantization.linear import GGUFLinearMethod
    from vllm_gguf_plugin.quantization.params import GGUFWeightParameter

    monkeypatch.setattr(
        "vllm_gguf_plugin.quantization.params.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm_gguf_plugin.quantization.params.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    sources = {
        0: torch.ones((2, 144), dtype=torch.uint8),
        1: torch.full((3, 84), 2, dtype=torch.uint8),
    }
    weight = GGUFWeightParameter(
        data=torch.empty(0, dtype=torch.uint8),
        weight_loader=lambda *args: None,
        input_dim=1,
        output_dim=0,
        tensor_shape=(5, 256),
    )
    weight.data_container = [sources[0], sources[1]]
    weight.shard_id = [0, 1]
    weight.shard_id_map = {0: 0, 1: 1}
    layer = torch.nn.Module()
    layer.register_parameter("weight", weight)
    layer.weight_type = SimpleNamespace(
        weight_type=Q.Q4_0,
        shard_weight_type={0: Q.Q4_0, 1: Q.Q2_K},
    )

    GGUFLinearMethod(None)._create_padded_weight_param(layer)

    assert layer.weight.ndim == 1
    for shard_id, source in sources.items():
        offset, rows, packed_row_size = layer.weight.shard_storage_map[shard_id]
        shard = layer.weight.narrow(0, offset, rows * packed_row_size).view(
            rows, packed_row_size
        )
        torch.testing.assert_close(shard, source)
        block_size, type_size = gguf.GGML_QUANT_SIZES[
            layer.weight_type.shard_weight_type[shard_id]
        ]
        logical_k = packed_row_size // type_size * block_size
        padding_bytes = ((-logical_k) % 512) // block_size * type_size
        available = layer.weight.untyped_storage().nbytes() - offset
        assert available >= shard.numel() + padding_bytes


@cuda_mark
@torch.inference_mode()
def test_dense_upstream_matches_legacy_and_reference():
    """Both MMVQ (batch=1) and MMQ (batch=16) agree with legacy and dense ref."""
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    n, k = 37, 256
    source = np.random.default_rng(0).standard_normal((n, k), dtype=np.float32)
    packed = gguf.quantize(source, Q.Q4_0)
    weight = make_padded_weight(packed, Q.Q4_0, k)
    dense = torch.from_numpy(gguf.dequantize(packed, Q.Q4_0)).cuda()

    for upstream_op, legacy_op, batch in (
        ("ggml_dense_mmvq", "ggml_mul_mat_vec_a8", 1),
        ("ggml_dense_mmq", "ggml_mul_mat_a8", 16),
    ):
        x = torch.randn((batch, k), device="cuda", dtype=torch.float32)
        upstream = getattr(torch.ops._C_gguf, upstream_op)(weight, x, int(Q.Q4_0), n)
        legacy = getattr(torch.ops._C_gguf, legacy_op)(weight, x, int(Q.Q4_0), n)
        reference = x @ dense.T

        torch.testing.assert_close(upstream, reference, atol=1.5, rtol=0.2)
        torch.testing.assert_close(upstream, legacy, atol=1.5, rtol=0.2)


@cuda_mark
@torch.inference_mode()
@pytest.mark.parametrize(
    "quant_type", (Q.Q4_1, Q.Q5_0, Q.Q5_1, Q.Q8_0), ids=lambda q: q.name
)
def test_dense_mmvq_upstream_reference(quant_type):
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    n, k = 7, 256
    source = np.random.default_rng(int(quant_type)).standard_normal(
        (n, k), dtype=np.float32
    )
    packed = gguf.quantize(source, quant_type)
    weight = make_padded_weight(packed, quant_type, k)
    x = torch.randn((2, k), device="cuda", dtype=torch.float32)

    actual = torch.ops._C_gguf.ggml_dense_mmvq(weight, x, int(quant_type), n)
    reference = x @ torch.from_numpy(gguf.dequantize(packed, quant_type)).cuda().T
    torch.testing.assert_close(actual, reference, atol=1.5, rtol=0.2)


@cuda_mark
@torch.inference_mode()
@pytest.mark.parametrize(
    "quant_type", (Q.Q4_0, Q.Q4_1, Q.Q5_0, Q.Q5_1, Q.Q8_0), ids=lambda q: q.name
)
def test_dense_mmq_upstream_reference(quant_type):
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    n, k, batch = 37, 256, 16
    source = np.random.default_rng(int(quant_type) + 200).standard_normal(
        (n, k), dtype=np.float32
    )
    packed = gguf.quantize(source, quant_type)
    weight = make_padded_weight(packed, quant_type, k)
    x = torch.randn((batch, k), device="cuda", dtype=torch.float32)

    actual = torch.ops._C_gguf.ggml_dense_mmq(weight, x, int(quant_type), n)
    reference = x @ torch.from_numpy(gguf.dequantize(packed, quant_type)).cuda().T
    torch.testing.assert_close(actual, reference, atol=1.5, rtol=0.2)


@cuda_mark
@torch.inference_mode()
@pytest.mark.parametrize("batch", [16, 65])
def test_dense_mmq_zero_dispatch_matrix(batch):
    """Each available MMQ type accepts batches of 16 and 65."""
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    n, k = 37, 768
    for quant_type in MMQ_TYPES:
        block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]
        packed = np.zeros((n, k // block_size * type_size), dtype=np.uint8)
        weight = make_padded_weight(packed, quant_type, k)
        x = torch.zeros((batch, k), device="cuda", dtype=torch.float32)
        output = torch.ops._C_gguf.ggml_dense_mmq(weight, x, int(quant_type), n)
        assert output.shape == (batch, n)
        assert output.dtype == x.dtype
        assert bool(torch.isfinite(output).all())
        torch.testing.assert_close(output, torch.zeros_like(output))


@cuda_mark
@torch.inference_mode()
def test_dense_upstream_without_storage_padding_uses_blas(monkeypatch):
    """A valid packed weight can use cuBLAS without the MMVQ padding tail."""
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    n, k = 9, 256
    source = np.random.default_rng(1).standard_normal((n, k), dtype=np.float32)
    packed = gguf.quantize(source, Q.Q4_1)
    weight = torch.from_numpy(packed).cuda()
    x = torch.randn((2, k), device="cuda")

    for variable in ("VLLM_GGUF_CUDA_KERNEL", "VLLM_GGUF_CUDA_DENSE_KERNEL"):
        monkeypatch.delenv(variable, raising=False)

    monkeypatch.setenv("VLLM_GGUF_CUDA_DENSE_KERNEL", "upstream")
    reference = x @ torch.from_numpy(gguf.dequantize(packed, Q.Q4_1)).cuda().T
    from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf

    output = _fused_mul_mat_gguf(x, weight, int(Q.Q4_1))
    torch.testing.assert_close(output, reference, atol=0.2, rtol=0.03)

    monkeypatch.setenv("VLLM_GGUF_CUDA_DENSE_KERNEL", "auto")
    output = _fused_mul_mat_gguf(x, weight, int(Q.Q4_1))
    assert output.shape == (x.shape[0], n)


@cuda_mark
@torch.inference_mode()
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dense_iq1_m_uses_blas_above_mmvq_limit(monkeypatch, dtype):
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    quant_type = Q.IQ1_M
    block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]
    n, k = 9, block_size
    packed = np.random.default_rng(4).integers(0, 256, (n, type_size), dtype=np.uint8)
    packed[:, 48:56] = np.tile(np.array([0, 60], dtype=np.uint8), 4)
    weight = make_padded_weight(packed, quant_type, k)
    x = torch.randn((65, k), device="cuda", dtype=dtype)
    dense = torch.from_numpy(gguf.dequantize(packed, quant_type)).cuda()
    monkeypatch.setenv("VLLM_GGUF_CUDA_KERNEL", "upstream")
    for batch in (1, 8, 9, 63, 64, 65):
        activation = x[:batch].contiguous()
        from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf

        output = _fused_mul_mat_gguf(activation, weight, int(quant_type))
        reference = (activation.half().float() @ dense.half().float().T).to(dtype)
        torch.testing.assert_close(output, reference, atol=0.5, rtol=0.04)


@cuda_mark
@torch.inference_mode()
def test_dense_iq1_m_blas_nondefault_stream_and_graph(monkeypatch):
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    monkeypatch.setenv("VLLM_GGUF_CUDA_KERNEL", "upstream")
    n, k, batch = 9, 256, 65
    raw = np.random.default_rng(8).integers(0, 256, (n, 56), dtype=np.uint8)
    raw[:, 48:56] = np.tile(np.array([0, 60], dtype=np.uint8), 4)
    weight = make_padded_weight(raw, Q.IQ1_M, k)
    dense = torch.from_numpy(gguf.dequantize(raw, Q.IQ1_M)).cuda()
    x = torch.randn((batch, k), device="cuda", dtype=torch.float16)
    op = torch.ops._C_gguf.ggml_dense_dequantize_blas
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        warmup = op(weight, x, int(Q.IQ1_M), n)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = op(weight, x, int(Q.IQ1_M), n)
        x.mul_(0.5)
        graph.replay()
        reference = (x.float() @ dense.half().float().T).half()
        torch.testing.assert_close(captured, reference, atol=0.5, rtol=0.04)
        torch.testing.assert_close(warmup, reference * 2, atol=0.5, rtol=0.04)


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
@pytest.mark.parametrize(
    "batch,rows,k", [(1, 37, 258), (5, 64, 256), (9, 64, 256), (17, 64, 256)]
)
def test_float_dense_upstream_dispatch(
    monkeypatch, dtype, quant_type, atol, batch, rows, k
):
    from vllm_gguf_plugin import ops
    from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf

    monkeypatch.setenv("VLLM_GGUF_CUDA_DENSE_KERNEL", "upstream")
    torch.manual_seed(20260929)
    w = torch.randn((rows, k), device="cuda", dtype=dtype) * 0.1
    x = torch.randn((batch, k), device="cuda", dtype=dtype) * 0.1
    caps = ops.dense_upstream_capabilities(w, x, int(quant_type), rows)
    if batch == 1:
        # The odd row count and K=258 rule out MMF, while MMVF handles tails.
        assert caps & ops.DENSE_MMVF
        assert not caps & ops.DENSE_MMF
    elif batch == 9 and torch.version.hip is None:
        assert not caps & ops.DENSE_MMVF
        major, _ = torch.cuda.get_device_capability()
        required_major = 7 if dtype == torch.float16 else 8
        assert bool(caps & ops.DENSE_MMF) == (major >= required_major)
    elif batch == 17:
        assert not caps & (ops.DENSE_MMVF | ops.DENSE_MMF)
    if caps & ops.DENSE_MMVF:
        expected = torch.ops._C_gguf.ggml_dense_mmvf(w, x, int(quant_type), rows)
    elif caps & ops.DENSE_MMF:
        expected = torch.ops._C_gguf.ggml_dense_mmf(w, x, int(quant_type), rows)
    else:
        expected = ops.ggml_dense_blas(w, x, int(quant_type), rows)
    actual = _fused_mul_mat_gguf(x, w, int(quant_type))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # Ampere's F32 MMF uses TF32 tensor-core instructions.
    reference_atol = 5e-3 if dtype == torch.float32 and caps & ops.DENSE_MMF else atol
    torch.testing.assert_close(
        actual, (x.float() @ w.float().T).to(dtype), atol=reference_atol, rtol=0
    )


@cuda_mark
@torch.inference_mode()
def test_float_dense_misaligned_weight_uses_blas(monkeypatch):
    from vllm_gguf_plugin import ops
    from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf

    monkeypatch.setenv("VLLM_GGUF_CUDA_DENSE_KERNEL", "upstream")
    storage = torch.randn(64 * 256 + 1, device="cuda", dtype=torch.float16)
    weight = storage[1:].view(64, 256)
    x = torch.randn((1, 256), device="cuda", dtype=torch.float16)
    assert ops.dense_supported_methods(weight, x, int(Q.F16), 64) == ops.DENSE_BLAS
    torch.testing.assert_close(_fused_mul_mat_gguf(x, weight, int(Q.F16)), x @ weight.T)


# CUDA runtime and public routing


@cuda_mark
@torch.inference_mode()
@pytest.mark.parametrize("batch", [1, 8, 16, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_casts_match_float32_path_on_offset_irregular_weights(
    monkeypatch, batch, dtype
):
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    monkeypatch.setenv("VLLM_GGUF_CUDA_KERNEL", "upstream")
    n, k = 37, 768
    packed = gguf.quantize(
        np.random.default_rng(42).standard_normal((n, k), dtype=np.float32) * 0.1,
        Q.Q4_0,
    )
    base = make_padded_weight(packed, Q.Q4_0, k)
    weight = base[1:]  # Exercise nonzero storage offset and an irregular last tile.
    x = torch.randn(batch, k, device="cuda", dtype=dtype)
    name = "ggml_dense_mmvq" if batch <= 8 else "ggml_dense_mmq"
    op = getattr(torch.ops._C_gguf, name)
    actual = op(weight, x, int(Q.Q4_0), n - 1)
    reference = op(weight, x.float(), int(Q.Q4_0), n - 1).to(dtype)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    dense = torch.from_numpy(gguf.dequantize(packed[1:], Q.Q4_0)).cuda()
    torch.testing.assert_close(
        actual.float(), x.float() @ dense.T, atol=0.15, rtol=0.05
    )


@cuda_mark
@torch.inference_mode()
@pytest.mark.parametrize("batch", [2, 16])
def test_cast_graphs_and_concurrent_streams(monkeypatch, batch):
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    monkeypatch.setenv("VLLM_GGUF_CUDA_KERNEL", "upstream")
    n, k = 37, 768
    packed = gguf.quantize(
        np.random.default_rng(9).standard_normal((n, k), dtype=np.float32), Q.Q4_0
    )
    weight = make_padded_weight(packed, Q.Q4_0, k)
    op = getattr(
        torch.ops._C_gguf, "ggml_dense_mmvq" if batch <= 8 else "ggml_dense_mmq"
    )
    inputs = [
        torch.randn(batch, k, device="cuda", dtype=torch.float16) for _ in range(2)
    ]
    expected = [op(weight, x, int(Q.Q4_0), n) for x in inputs]
    streams = [torch.cuda.Stream() for _ in inputs]
    torch.cuda.synchronize()

    def run(i):
        with torch.inference_mode(), torch.cuda.stream(streams[i]):
            return [op(weight, inputs[i], int(Q.Q4_0), n) for _ in range(10)]

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
            outputs.append(op(weight, inputs[i], int(Q.Q4_0), n))
        graphs.append(graph)
    for _ in range(3):
        for i, graph in enumerate(graphs):
            with torch.cuda.stream(streams[i]):
                graph.replay()
    torch.cuda.synchronize()
    for i, output in enumerate(outputs):
        torch.testing.assert_close(output, expected[i], rtol=0, atol=0)


@cuda_mark
@torch.inference_mode()
@pytest.mark.parametrize(
    "batch,dtype",
    [
        (1, torch.float16),
        (16, torch.float16),
        (16, torch.float32),
        (128, torch.float16),
    ],
)
def test_default_and_auto_use_upstream(monkeypatch, batch, dtype):
    from vllm_gguf_plugin import ops

    n, k = 512, 1024
    packed = gguf.quantize(
        np.random.default_rng(71).standard_normal((n, k), dtype=np.float32),
        Q.Q4_0,
    )
    weight = make_padded_weight(packed, Q.Q4_0, k)
    x = torch.randn(batch, k, device="cuda", dtype=dtype)
    op_name = "ggml_mul_mat_vec_a8" if batch == 1 else "ggml_mul_mat_a8"
    op = getattr(ops, op_name)

    monkeypatch.setenv("VLLM_GGUF_CUDA_KERNEL", "upstream")
    expected = op(weight, x, int(Q.Q4_0), n)
    monkeypatch.delenv("VLLM_GGUF_CUDA_KERNEL")
    default = op(weight, x, int(Q.Q4_0), n)
    monkeypatch.setenv("VLLM_GGUF_CUDA_KERNEL", "auto")
    auto = op(weight, x, int(Q.Q4_0), n)

    torch.testing.assert_close(default, expected, rtol=0, atol=0)
    torch.testing.assert_close(auto, expected, rtol=0, atol=0)


@cuda_mark
@torch.inference_mode()
def test_iq_public_fallback_is_nonzero_and_legacy_mode_rejected(monkeypatch):
    from vllm_gguf_plugin import ops

    monkeypatch.setenv("VLLM_GGUF_CUDA_KERNEL", "auto")
    n, k = 37, 256
    # gguf implements IQ4_NL dequantization, but not its Python quantizer.
    # Every nibble is a valid codebook index; use finite, nonzero block scales.
    packed = np.random.default_rng(23).integers(
        0, 256, (n, k // 32, 18), dtype=np.uint8
    )
    packed[:, :, :2] = np.array([0.01], dtype=np.float16).view(np.uint8)
    packed = packed.reshape(n, -1)
    weight = make_padded_weight(packed, Q.IQ4_NL, k)
    x = torch.randn(16, k, device="cuda", dtype=torch.float16)
    actual = ops.ggml_mul_mat_a8(weight, x, int(Q.IQ4_NL), n)
    reference = x.float() @ torch.from_numpy(gguf.dequantize(packed, Q.IQ4_NL)).cuda().T
    assert torch.count_nonzero(actual) > 0
    torch.testing.assert_close(actual.float(), reference, atol=0.25, rtol=0.10)
    monkeypatch.setenv("VLLM_GGUF_CUDA_KERNEL", "legacy")
    with pytest.raises(RuntimeError, match="legacy MMQ backend is unavailable"):
        ops.ggml_mul_mat_a8(weight, x, int(Q.IQ4_NL), n)
