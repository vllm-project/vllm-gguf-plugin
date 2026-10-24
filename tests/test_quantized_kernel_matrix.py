"""CUDA type matrix for quantized dense, MoE, dequantization, and embedding.

The test uses real nonzero GGUF IQ weights when a local sample snapshot is
available. It never downloads data; other formats use local quantization or
synthetic blocks.
"""

import os
from functools import cache
from pathlib import Path

import gguf
import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType as Q

from tests.helpers_upstream import (
    TEMPLATE_EXTRA_TYPES,
    make_padded_weight,
    make_template_raw,
)
from vllm_gguf_plugin import ops
from vllm_gguf_plugin.kernel_support import (
    CUDA_LEGACY_MMQ_TYPES,
    CUDA_LEGACY_MMVQ_TYPES,
    CUDA_UPSTREAM_DEQUANT_TYPES,
    CUDA_UPSTREAM_MMQ_TYPES,
    CUDA_UPSTREAM_MMVQ_TYPES,
)
from vllm_gguf_plugin.quantization.fused_moe import _fused_moe_gguf
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"
)

UP_MMVQ = sorted((q for q in CUDA_UPSTREAM_MMVQ_TYPES if isinstance(q, Q)), key=int)
UP_MMQ = sorted((q for q in CUDA_UPSTREAM_MMQ_TYPES if isinstance(q, Q)), key=int)
LEG_MMVQ = sorted(CUDA_LEGACY_MMVQ_TYPES, key=int)
LEG_MMQ = sorted(CUDA_LEGACY_MMQ_TYPES, key=int)
ROWS, K, EXPERTS, TOP_K = 128, 256, 3, 2


def _sample_path(quant_type: Q) -> Path | None:
    filename = f"Quant_{quant_type.name}_{K}.gguf"
    specified = os.environ.get("GGUF_MATRIX_SAMPLE_DIR")
    if specified:
        path = Path(specified) / filename
        return path if path.exists() else None
    cache = (
        Path.home()
        / ".cache/huggingface/hub/models--Isotr0py--test-gguf-sample/snapshots"
    )
    return next((path for path in cache.glob(f"*/{filename}") if path.exists()), None)


@cache
def _packed(quant_type: Q) -> tuple[np.ndarray, np.ndarray | None]:
    count = ROWS * EXPERTS
    sample = _sample_path(quant_type)
    if sample is not None:
        raw = np.array(gguf.GGUFReader(sample).tensors[0].data[:count], copy=True)
    elif quant_type in {Q.Q4_1, Q.Q5_1}:
        source = np.random.default_rng(int(quant_type)).standard_normal(
            (count, K), dtype=np.float32
        )
        raw = gguf.quantize(source, quant_type)
    elif quant_type.name in {"MXFP4", "NVFP4"}:
        raw = make_template_raw(quant_type, count, K)
        _, size = gguf.GGML_QUANT_SIZES[quant_type]
        if quant_type.name == "MXFP4":
            raw[:, ::size] = 126  # finite E8M0 scale near one
        else:
            for offset in range(0, raw.shape[1], size):
                raw[:, offset : offset + 4] = 0x3C  # moderate finite scales
    else:
        block, size = gguf.GGML_QUANT_SIZES[quant_type]
        raw = np.zeros((count, K // block * size), dtype=np.uint8)
    try:
        dense = gguf.dequantize(raw, quant_type)
    except NotImplementedError:
        dense = None
    if dense is not None:
        assert np.isfinite(dense).all(), quant_type.name
    return raw, dense


def _quant_weight(
    quant_type: Q, *, moe: bool
) -> tuple[torch.Tensor, torch.Tensor | None]:
    raw, dense = _packed(quant_type)
    if moe:
        raw = raw.reshape(EXPERTS, ROWS, -1)
        if dense is not None:
            dense = dense.reshape(EXPERTS, ROWS, K)
    else:
        raw = raw[:ROWS]
        if dense is not None:
            dense = dense[:ROWS]
    weight = make_padded_weight(raw, quant_type, K)
    reference = torch.from_numpy(np.array(dense)).cuda() if dense is not None else None
    return weight, reference


def _input(batch: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(1 + batch)
    return torch.randn((batch, K), generator=generator, device="cuda")


def _check(
    actual: torch.Tensor, expected: torch.Tensor | None, shape: tuple[int, int]
) -> None:
    assert tuple(actual.shape) == shape
    assert bool(torch.isfinite(actual).all())
    if expected is not None:
        torch.testing.assert_close(actual.float(), expected.float(), atol=1.5, rtol=0.2)


@pytest.mark.parametrize(
    "backend,quant_type",
    [
        *(
            ("upstream", quant_type)
            for quant_type in sorted(
                (q for q in CUDA_UPSTREAM_DEQUANT_TYPES if isinstance(q, Q)), key=int
            )
        ),
        *(("legacy", quant_type) for quant_type in LEG_MMVQ),
    ],
    ids=lambda value: value.name if isinstance(value, Q) else value,
)
@torch.inference_mode()
def test_dequantize_all_types(monkeypatch, backend, quant_type):
    monkeypatch.setenv("VLLM_GGUF_CUDA_DEQUANTIZE_KERNEL", backend)
    weight, dense = _quant_weight(quant_type, moe=False)
    actual = ops.ggml_dequantize(weight, int(quant_type), ROWS, K, torch.float32)
    _check(actual, dense, (ROWS, K))


@torch.inference_mode()
def test_pinned_q2_0_raw_upstream_entrypoints():
    """Cover pinned GGML Q2_0 even when gguf-python has no enum for it."""
    type_id, block_size, type_size, k = 42, 64, 18, 512
    packed_row = k // block_size * type_size
    weight = torch.zeros((ROWS, packed_row), device="cuda", dtype=torch.uint8)
    moe_weight = torch.zeros(
        (EXPERTS, ROWS, packed_row), device="cuda", dtype=torch.uint8
    )
    dequant = torch.ops._C_gguf.ggml_dequantize_upstream(
        weight, type_id, ROWS, k, torch.float32
    )
    _check(dequant, torch.zeros_like(dequant), (ROWS, k))
    for batch, flag, name in (
        (1, ops.DENSE_MMVQ, "ggml_dense_mmvq"),
        (9, ops.DENSE_MMQ, "ggml_dense_mmq"),
        (65, ops.DENSE_DEQUANTIZE_BLAS, "ggml_dense_dequantize_blas"),
    ):
        x = torch.zeros((batch, k), device="cuda")
        assert ops.dense_supported_methods(weight, x, type_id, ROWS) & flag
        actual = getattr(torch.ops._C_gguf, name)(weight, x, type_id, ROWS)
        _check(actual, torch.zeros_like(actual), (batch, ROWS))
        ids = torch.tensor(
            [[t % EXPERTS, (t + 1) % EXPERTS] for t in range(batch)],
            device="cuda",
            dtype=torch.int32,
        )
        moe = torch.ops._C_gguf.ggml_moe_upstream(
            x, moe_weight, ids, type_id, ROWS, TOP_K, batch
        )
        _check(moe, torch.zeros_like(moe), (batch * TOP_K, ROWS))

    # The missing Python enum must not hide the aligned MMQ specialization.
    blocks = moe_weight.view(-1, type_size)
    blocks[:, 2:].random_(0, 256)
    blocks[:, :2].copy_(
        torch.tensor([0.01], device="cuda", dtype=torch.float16).view(torch.uint8)
    )
    dense = torch.ops._C_gguf.ggml_dequantize_upstream(
        moe_weight.view(-1, packed_row), type_id, EXPERTS * ROWS, k, torch.float32
    ).view(EXPERTS, ROWS, k)
    tokens = 129
    x = torch.randn((tokens, k), device="cuda") * 0.1
    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

    for top_k in (1, TOP_K):
        ids = (
            torch.arange(tokens, device="cuda", dtype=torch.int32)[:, None]
            + torch.arange(top_k, device="cuda", dtype=torch.int32)[None, :]
        ) % EXPERTS
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            aligned = moe_align_block_size(ids, 16, EXPERTS, pad_sorted_ids=True)
            output = torch.ops._C_gguf.ggml_moe_mmq_aligned(
                x,
                moe_weight,
                aligned[0],
                type_id,
                ROWS,
                top_k,
                tokens,
                expert_ids=aligned[1],
                padded_count=aligned[2],
            )
        ids.copy_((ids + 1) % EXPERTS)
        graph.replay()
        expected = torch.einsum("trnk,tk->trn", dense[ids.long()], x).reshape(
            tokens * top_k, ROWS
        )
        _check(output, expected, (tokens * top_k, ROWS))


@pytest.mark.parametrize("quant_type", UP_MMVQ, ids=lambda q: q.name)
@pytest.mark.parametrize("batch", (1, 8))
@torch.inference_mode()
def test_upstream_dense_mmvq_all_types(quant_type, batch):
    weight, dense = _quant_weight(quant_type, moe=False)
    x = _input(batch)
    caps = ops.dense_supported_methods(weight, x, int(quant_type), ROWS)
    if not caps & ops.DENSE_MMVQ:
        pytest.skip(f"MMVQ not eligible on this GPU: {quant_type.name}, batch={batch}")
    actual = ops.ggml_dense_mmvq(weight, x, int(quant_type), ROWS)
    _check(actual, x @ dense.T if dense is not None else None, (batch, ROWS))


@pytest.mark.parametrize("quant_type", UP_MMQ, ids=lambda q: q.name)
@pytest.mark.parametrize("batch", (9, 32))
@torch.inference_mode()
def test_upstream_dense_mmq_all_types(quant_type, batch):
    weight, dense = _quant_weight(quant_type, moe=False)
    x = _input(batch)
    caps = ops.dense_supported_methods(weight, x, int(quant_type), ROWS)
    if not caps & ops.DENSE_MMQ:
        pytest.skip(f"MMQ not eligible on this GPU: {quant_type.name}, batch={batch}")
    actual = ops.ggml_dense_mmq(weight, x, int(quant_type), ROWS)
    _check(actual, x @ dense.T if dense is not None else None, (batch, ROWS))


@pytest.mark.parametrize("quant_type", UP_MMVQ, ids=lambda q: q.name)
@torch.inference_mode()
def test_upstream_dense_blas_all_types(quant_type):
    weight, dense = _quant_weight(quant_type, moe=False)
    x = _input(65)
    caps = ops.dense_supported_methods(weight, x, int(quant_type), ROWS)
    assert caps & ops.DENSE_DEQUANTIZE_BLAS, (quant_type.name, caps)
    actual = ops.ggml_dense_dequantize_blas(weight, x, int(quant_type), ROWS)
    _check(actual, x @ dense.T if dense is not None else None, (65, ROWS))


@pytest.mark.parametrize("quant_type", UP_MMVQ, ids=lambda q: q.name)
@pytest.mark.parametrize("batch", (1, 8, 9, 65))
@torch.inference_mode()
def test_upstream_dense_python_route_all_types(monkeypatch, quant_type, batch):
    monkeypatch.setenv("VLLM_GGUF_CUDA_DENSE_KERNEL", "upstream")
    weight, dense = _quant_weight(quant_type, moe=False)
    x = _input(batch)
    caps = ops.dense_select_method(weight, x, int(quant_type), ROWS)
    selected = next(
        (
            op
            for flag, op in (
                (ops.DENSE_MMVQ, ops.ggml_dense_mmvq),
                (ops.DENSE_MMQ, ops.ggml_dense_mmq),
                (ops.DENSE_DEQUANTIZE_BLAS, ops.ggml_dense_dequantize_blas),
            )
            if caps & flag
        ),
        None,
    )
    assert selected is not None, (quant_type.name, batch, caps)
    actual = _fused_mul_mat_gguf(x, weight, int(quant_type))
    expected_route = selected(weight, x, int(quant_type), ROWS)
    torch.testing.assert_close(actual, expected_route, atol=0, rtol=0)
    _check(actual, x @ dense.T if dense is not None else None, (batch, ROWS))


@pytest.mark.parametrize("quant_type", LEG_MMVQ, ids=lambda q: q.name)
@pytest.mark.parametrize("batch", (1, 8))
@torch.inference_mode()
def test_legacy_dense_mmvq_all_types(monkeypatch, quant_type, batch):
    monkeypatch.setenv("VLLM_GGUF_CUDA_DENSE_KERNEL", "legacy")
    weight, dense = _quant_weight(quant_type, moe=False)
    x = _input(batch)
    actual = ops.ggml_mul_mat_vec_a8(weight, x, int(quant_type), ROWS)
    _check(actual, x @ dense.T if dense is not None else None, (batch, ROWS))


@pytest.mark.parametrize("quant_type", LEG_MMQ, ids=lambda q: q.name)
@pytest.mark.parametrize("batch", (9, 65))
@torch.inference_mode()
def test_legacy_dense_mmq_all_types(monkeypatch, quant_type, batch):
    monkeypatch.setenv("VLLM_GGUF_CUDA_DENSE_KERNEL", "legacy")
    weight, dense = _quant_weight(quant_type, moe=False)
    x = _input(batch)
    actual = ops.ggml_mul_mat_a8(weight, x, int(quant_type), ROWS)
    _check(actual, x @ dense.T if dense is not None else None, (batch, ROWS))


@pytest.mark.parametrize(
    "quant_type,batch,selected",
    [
        *((quant_type, 1, "mmvq") for quant_type in LEG_MMVQ),
        *((quant_type, 65, "mmq") for quant_type in LEG_MMQ),
    ],
    ids=lambda value: value.name if isinstance(value, Q) else str(value),
)
@torch.inference_mode()
def test_legacy_dense_python_route_all_types(monkeypatch, quant_type, batch, selected):
    monkeypatch.setenv("VLLM_GGUF_CUDA_DENSE_KERNEL", "legacy")
    weight, dense = _quant_weight(quant_type, moe=False)
    x = _input(batch)
    calls = []
    for name in ("ggml_mul_mat_vec_a8", "ggml_mul_mat_a8"):
        original = getattr(ops, name)

        def recording(*args, _original=original, _name=name):
            calls.append("mmvq" if _name.endswith("vec_a8") else "mmq")
            return _original(*args)

        monkeypatch.setattr(ops, name, recording)
    actual = _fused_mul_mat_gguf(x, weight, int(quant_type))
    assert calls == [selected]
    _check(actual, x @ dense.T if dense is not None else None, (batch, ROWS))


def _moe_inputs(tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = _input(tokens)
    ids = torch.tensor(
        [[t % EXPERTS, (t + 1) % EXPERTS] for t in range(tokens)],
        device="cuda",
        dtype=torch.int32,
    )
    return x, ids


def _moe_reference(x: torch.Tensor, ids: torch.Tensor, dense: torch.Tensor | None):
    if dense is None:
        return None
    return torch.einsum("trnk,tk->trn", dense[ids.long()].float(), x.float()).reshape(
        x.size(0) * TOP_K, ROWS
    )


@pytest.mark.parametrize("quant_type", UP_MMVQ, ids=lambda q: q.name)
@pytest.mark.parametrize("tokens", (1, 8, 9, 65))
@torch.inference_mode()
def test_upstream_moe_all_types(quant_type, tokens):
    weight, dense = _quant_weight(quant_type, moe=True)
    x, ids = _moe_inputs(tokens)
    actual = ops.ggml_moe_upstream(x, weight, ids, int(quant_type), ROWS, TOP_K, tokens)
    _check(actual, _moe_reference(x, ids, dense), (tokens * TOP_K, ROWS))


@pytest.mark.parametrize("quant_type", UP_MMQ, ids=lambda q: q.name)
@pytest.mark.parametrize("top_k", (1, TOP_K))
@torch.inference_mode()
def test_upstream_moe_explicit_mmq_all_types(quant_type, top_k):
    # Force MMQ: small automatic MoE calls normally select MMVQ instead.
    tokens = 129
    weight, dense = _quant_weight(quant_type, moe=True)
    x, ids = _moe_inputs(tokens)
    ids = ids[:, :top_k].contiguous()

    def run():
        return ops.ggml_moe_mmq(x, weight, ids, int(quant_type), ROWS, top_k, tokens)

    def check(actual):
        expected = (
            None
            if dense is None
            else torch.einsum(
                "trnk,tk->trn", dense[ids.long()].float(), x.float()
            ).reshape(tokens * top_k, ROWS)
        )
        _check(actual, expected, (tokens * top_k, ROWS))

    check(run())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    ids.copy_((ids + 1) % weight.shape[0])
    x.mul_(0.5)
    graph.replay()
    check(output)


@pytest.mark.parametrize("quant_type", LEG_MMVQ, ids=lambda q: q.name)
@pytest.mark.parametrize("tokens", (1, 8))
@torch.inference_mode()
def test_legacy_moe_mmvq_all_types(quant_type, tokens):
    weight, dense = _quant_weight(quant_type, moe=True)
    x, ids = _moe_inputs(tokens)
    actual = torch.ops._C_gguf.ggml_moe_a8_vec(
        x, weight, ids, TOP_K, int(quant_type), ROWS, tokens
    )
    _check(actual, _moe_reference(x, ids, dense), (tokens * TOP_K, ROWS))


@pytest.mark.parametrize("quant_type", LEG_MMQ, ids=lambda q: q.name)
@torch.inference_mode()
def test_legacy_moe_mmq_all_types(quant_type):
    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

    weight, dense = _quant_weight(quant_type, moe=True)
    tokens = 65
    x, ids = _moe_inputs(tokens)
    block_size = torch.ops._C_gguf.ggml_moe_get_block_size(int(quant_type))
    sorted_ids, expert_ids, counts = moe_align_block_size(ids, block_size, EXPERTS)
    actual = torch.ops._C_gguf.ggml_moe_a8(
        x, weight, sorted_ids, expert_ids, counts, int(quant_type), ROWS, TOP_K, tokens
    )
    _check(actual, _moe_reference(x, ids, dense), (tokens * TOP_K, ROWS))


def _fused_moe_weights(quant_type: Q) -> tuple[torch.Tensor, torch.Tensor]:
    raw, _ = _packed(quant_type)
    packed_row = raw.shape[1]
    w1 = np.tile(raw, (4, 1))[: EXPERTS * 2 * K].reshape(EXPERTS, 2 * K, packed_row)
    w2 = np.tile(raw, (2, 1))[: EXPERTS * K].reshape(EXPERTS, K, packed_row)
    return (
        make_padded_weight(w1, quant_type, K),
        make_padded_weight(w2, quant_type, K),
    )


@pytest.mark.parametrize(
    "backend,quant_type,tokens,selected",
    [
        *(
            ("upstream", quant_type, tokens, "upstream")
            for quant_type in UP_MMVQ
            for tokens in (2, 65)
        ),
        *(("legacy", quant_type, 2, "vec") for quant_type in LEG_MMVQ),
        *(("legacy", quant_type, 65, "mmq") for quant_type in LEG_MMQ),
    ],
    ids=lambda value: value.name if isinstance(value, Q) else str(value),
)
@torch.inference_mode()
def test_fused_moe_python_route_all_types(
    monkeypatch, backend, quant_type, tokens, selected
):
    monkeypatch.setenv("VLLM_GGUF_CUDA_MOE_KERNEL", backend)
    w1, w2 = _fused_moe_weights(quant_type)
    x, ids = _moe_inputs(tokens)
    weights = torch.full((tokens, TOP_K), 1 / TOP_K, device="cuda")
    calls = []
    for name, label in (
        ("ggml_moe", "upstream"),
        ("ggml_moe_a8_vec", "vec"),
        ("ggml_moe_a8", "mmq"),
    ):
        original = getattr(ops, name)

        def recording(*args, _original=original, _label=label, **kwargs):
            calls.append(_label)
            return _original(*args, **kwargs)

        monkeypatch.setattr(ops, name, recording)
    actual = _fused_moe_gguf(
        x, w1, w2, weights, ids, int(quant_type), int(quant_type), "silu"
    )
    assert calls == [selected, selected]
    assert tuple(actual.shape) == (tokens, K)
    assert bool(torch.isfinite(actual).all())


# Dequantization and embedding across supported types


def _template_n(quant_type: Q) -> int:
    block_size, _ = gguf.GGML_QUANT_SIZES[quant_type]
    # convert.cu's MXFP4 row kernel consumes one 256-value super-block.
    return 256 if quant_type.name == "MXFP4" else block_size * 2


@torch.inference_mode()
def test_dequantize_template_instance_types(monkeypatch):
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    from vllm_gguf_plugin import ops

    monkeypatch.setenv("VLLM_GGUF_CUDA_KERNEL", "upstream")
    rows = 3
    for quant_type in TEMPLATE_EXTRA_TYPES:
        n = _template_n(quant_type)
        block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]
        raw = np.zeros((rows, n // block_size * type_size), dtype=np.uint8)
        weight = torch.from_numpy(raw).cuda()
        output = ops.ggml_dequantize(weight, quant_type, rows, n, torch.float32)
        assert output.shape == (rows, n)
        assert bool(torch.isfinite(output).all())
        torch.testing.assert_close(output, torch.zeros_like(output))


@torch.inference_mode()
def test_dequantize_and_embedding_template_reference(monkeypatch):
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    from vllm_gguf_plugin import ops
    from vllm_gguf_plugin.quantization.vocal_embeds import apply_gguf_embedding

    monkeypatch.setenv("VLLM_GGUF_CUDA_KERNEL", "upstream")
    for quant_type in TEMPLATE_EXTRA_TYPES:
        try:
            probe = gguf.dequantize(
                np.zeros(gguf.GGML_QUANT_SIZES[quant_type][1], dtype=np.uint8),
                quant_type,
            )
        except NotImplementedError:
            # gguf-python has no Python dequantizer for this type yet; the
            # kernel-side smoke test still runs above.
            continue
        del probe
        n = _template_n(quant_type)
        raw = make_template_raw(quant_type, rows=4, n=n)
        weight = torch.from_numpy(raw).cuda()
        reference = torch.from_numpy(gguf.dequantize(raw, quant_type)).cuda()

        output = ops.ggml_dequantize(weight, quant_type, 4, n, torch.float32)
        torch.testing.assert_close(output, reference, atol=1e-5, rtol=1e-5)

        ids = torch.tensor([[0, 2], [3, 1]], dtype=torch.long, device="cuda")
        embedding = apply_gguf_embedding(
            ids, weight, quant_type, n, dtype=torch.float32
        )
        torch.testing.assert_close(embedding, reference[ids], atol=1e-5, rtol=1e-5)
