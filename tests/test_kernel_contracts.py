"""Kernel mode, support, GGML ABI, and architecture-policy contracts."""

import re
from pathlib import Path

import gguf
import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType as Q

from tests.helpers_upstream import make_padded_weight
from vllm_gguf_plugin.kernel_support import (
    CUDA_UPSTREAM_MMQ_TYPES,
    CUDA_UPSTREAM_MMVQ_TYPES,
)

KERNEL_ENV_VARS = (
    "VLLM_GGUF_CUDA_KERNEL",
    "VLLM_GGUF_CUDA_DENSE_KERNEL",
    "VLLM_GGUF_CUDA_MOE_KERNEL",
    "VLLM_GGUF_CUDA_DEQUANTIZE_KERNEL",
)
TEMPLATE_EXTRA_TYPES = tuple(
    getattr(Q, name) for name in ("Q1_0", "Q2_0", "MXFP4", "NVFP4") if hasattr(Q, name)
)
K_TYPES = (Q.Q2_K, Q.Q3_K, Q.Q4_K, Q.Q5_K, Q.Q6_K)
ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "vllm_gguf_plugin/csrc/upstream/ggml_constants.cuh"
UPSTREAM = ROOT / "third_party/llama.cpp/ggml"


# Mode selection and Python routing


def _clear_kernel_environment(monkeypatch):
    for name in KERNEL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_kernel_selector_defaults_and_global_override(monkeypatch):
    from vllm_gguf_plugin import ops

    _clear_kernel_environment(monkeypatch)
    assert ops.cuda_dense_kernel_mode() == "auto"
    assert ops.cuda_moe_kernel_mode() == "auto"
    assert ops.cuda_dequantize_kernel_mode() == "auto"
    assert ops.cuda_kernel_mode() == "auto"

    monkeypatch.setenv("VLLM_GGUF_CUDA_KERNEL", "legacy")
    assert ops.cuda_dense_kernel_mode() == "legacy"
    assert ops.cuda_moe_kernel_mode() == "legacy"
    assert ops.cuda_dequantize_kernel_mode() == "legacy"

    monkeypatch.setenv("VLLM_GGUF_CUDA_DENSE_KERNEL", "upstream")
    monkeypatch.setenv("VLLM_GGUF_CUDA_MOE_KERNEL", "triton")
    assert ops.cuda_dense_kernel_mode() == "upstream"
    assert ops.cuda_moe_kernel_mode() == "triton"
    assert ops.cuda_dequantize_kernel_mode() == "legacy"

    monkeypatch.setenv("VLLM_GGUF_CUDA_DEQUANTIZE_KERNEL", "upstream")
    assert ops.cuda_dequantize_kernel_mode() == "upstream"


def test_kernel_selector_rejects_invalid_values(monkeypatch):
    from vllm_gguf_plugin import ops

    _clear_kernel_environment(monkeypatch)
    monkeypatch.setenv("VLLM_GGUF_CUDA_KERNEL", "invalid")
    with pytest.raises(ValueError, match="auto\\|upstream\\|legacy\\|triton"):
        ops.cuda_dense_kernel_mode()


def test_support_matrix_is_explicit():
    from vllm_gguf_plugin import ops

    extra_types = {int(q) for q in TEMPLATE_EXTRA_TYPES} | {42}
    standard_mmvq = {
        ops.GGML_TYPE_Q4_0,
        ops.GGML_TYPE_Q4_1,
        ops.GGML_TYPE_Q5_0,
        ops.GGML_TYPE_Q5_1,
        ops.GGML_TYPE_Q8_0,
        ops.GGML_TYPE_Q2_K,
        ops.GGML_TYPE_Q3_K,
        ops.GGML_TYPE_Q4_K,
        ops.GGML_TYPE_Q5_K,
        ops.GGML_TYPE_Q6_K,
        ops.GGML_TYPE_IQ2_XXS,
        ops.GGML_TYPE_IQ2_XS,
        ops.GGML_TYPE_IQ3_XXS,
        ops.GGML_TYPE_IQ1_S,
        ops.GGML_TYPE_IQ4_NL,
        ops.GGML_TYPE_IQ3_S,
        ops.GGML_TYPE_IQ2_S,
        ops.GGML_TYPE_IQ4_XS,
        ops.GGML_TYPE_IQ1_M,
    }
    mmq_subset = standard_mmvq - {ops.GGML_TYPE_IQ1_M}

    assert standard_mmvq | extra_types == ops.CUDA_UPSTREAM_MMVQ_TYPES
    assert mmq_subset | extra_types == ops.CUDA_UPSTREAM_MMQ_TYPES
    assert ops.CUDA_LEGACY_MMVQ_TYPES < ops.CUDA_UPSTREAM_MMVQ_TYPES
    assert ops.CUDA_LEGACY_MMQ_TYPES < ops.CUDA_UPSTREAM_MMQ_TYPES
    assert ops.CUDA_UPSTREAM_DEQUANT_TYPES == ops.CUDA_UPSTREAM_MMVQ_TYPES
    assert (ops.CUDA_UPSTREAM_MMVQ_TYPES | {Q.F32, Q.F16, Q.BF16}) == (
        ops.CUDA_UPSTREAM_MOE_TYPES
    )


def test_all_upstream_template_instances_are_listed():
    """Every MMQ and MMF template instance is registered in the manifest."""
    import tomllib

    repo_root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads(
        (repo_root / "vllm_gguf_plugin" / "llama_cpp_upstream.toml").read_text()
    )
    actual = {
        path.relative_to(repo_root / "third_party" / "llama.cpp").as_posix()
        for pattern in ("mmq-instance-*.cu", "mmf-instance-ncols_*.cu")
        for path in (
            repo_root
            / "third_party"
            / "llama.cpp"
            / "ggml"
            / "src"
            / "ggml-cuda"
            / "template-instances"
        ).glob(pattern)
    }
    listed = set(metadata["sources"])
    assert actual <= listed


@pytest.mark.parametrize("mode", ["auto", "upstream"])
@pytest.mark.parametrize("wrapper", ["ggml_mul_mat_vec_a8", "ggml_mul_mat_a8"])
def test_dense_compatibility_wrappers_delegate_to_decision(monkeypatch, mode, wrapper):
    from vllm_gguf_plugin import ops

    monkeypatch.setenv("VLLM_GGUF_CUDA_DENSE_KERNEL", mode)
    calls = []
    monkeypatch.setattr(
        ops, "ggml_dense", lambda *args: calls.append(args) or "central"
    )
    monkeypatch.setattr(
        ops, "_cuda_kernel_available", lambda *args: pytest.fail("backend fallback")
    )
    args = (torch.empty(1), torch.empty(1), int(Q.Q4_0), 1)
    assert getattr(ops, wrapper)(*args) == "central"
    assert calls == [args]


@pytest.mark.parametrize("wrapper", ["ggml_mul_mat_vec_a8", "ggml_mul_mat_a8"])
def test_dense_auto_without_upstream_rejects_other_backends(monkeypatch, wrapper):
    from vllm_gguf_plugin import ops

    monkeypatch.setenv("VLLM_GGUF_CUDA_DENSE_KERNEL", "auto")
    monkeypatch.setattr(ops, "_CUDA_ENABLED", False)
    monkeypatch.setattr(
        torch.ops._C_gguf, wrapper, lambda *args: pytest.fail("legacy executed")
    )
    monkeypatch.setattr(
        ops, "ggml_mul_mat_a8_triton", lambda *args: pytest.fail("triton executed")
    )
    with pytest.raises(
        RuntimeError, match="upstream CUDA op ggml_dense is unavailable"
    ):
        getattr(ops, wrapper)(torch.empty(1), torch.empty(1), int(Q.Q4_0), 1)


@pytest.mark.parametrize("mode", ["auto", "upstream"])
def test_dequantize_auto_never_falls_back(monkeypatch, mode):
    from vllm_gguf_plugin import ops

    monkeypatch.setenv("VLLM_GGUF_CUDA_DEQUANTIZE_KERNEL", mode)
    monkeypatch.setattr(ops, "_cuda_upstream_supports", lambda *args: False)
    monkeypatch.setattr(ops, "_cuda_kernel_available", lambda *args: True)
    monkeypatch.setattr(
        torch.ops._C_gguf,
        "ggml_dequantize",
        lambda *args: pytest.fail("legacy executed"),
    )
    monkeypatch.setattr(
        ops, "ggml_dequantize_triton", lambda *args: pytest.fail("triton executed")
    )
    with pytest.raises(RuntimeError, match="upstream dequantize"):
        ops.ggml_dequantize(torch.empty(1), int(Q.Q4_0), 1, 32, torch.float16)


def test_moe_auto_without_upstream_rejects_other_backends(monkeypatch):
    from vllm_gguf_plugin import ops

    monkeypatch.setenv("VLLM_GGUF_CUDA_MOE_KERNEL", "auto")
    monkeypatch.setattr(ops, "_CUDA_ENABLED", False)
    monkeypatch.setattr(
        ops, "ggml_moe_a8", lambda *args: pytest.fail("legacy executed")
    )
    monkeypatch.setattr(
        ops, "ggml_moe_a8_triton", lambda *args: pytest.fail("triton executed")
    )
    with pytest.raises(RuntimeError, match="upstream CUDA op ggml_moe is unavailable"):
        ops.ggml_moe(
            torch.empty(1), torch.empty(1), torch.empty(1), int(Q.Q4_0), 1, 1, 1
        )


@pytest.mark.parametrize("mode", ["auto", "upstream"])
@pytest.mark.parametrize("quant_type", [Q.F32, Q.Q4_0])
def test_linear_delegates_to_one_upstream_decision(monkeypatch, mode, quant_type):
    from vllm_gguf_plugin import ops
    from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf

    monkeypatch.setenv("VLLM_GGUF_CUDA_DENSE_KERNEL", mode)
    calls = []
    monkeypatch.setattr(
        ops, "ggml_dense", lambda *args: calls.append(args) or "central"
    )
    monkeypatch.setattr(
        ops, "dense_upstream_capabilities", lambda *args: pytest.fail("Python policy")
    )
    x = torch.empty((9, 256))
    w = torch.empty((32, 256))
    assert _fused_mul_mat_gguf(x, w, int(quant_type)) == "central"
    assert calls == [(w, x, int(quant_type), 32)]


@pytest.mark.parametrize("wrapper", ["ggml_mul_mat_vec_a8", "ggml_mul_mat_a8"])
@pytest.mark.parametrize("mode", ["legacy", "triton"])
def test_dense_explicit_backend_stays_available(monkeypatch, wrapper, mode):
    from vllm_gguf_plugin import ops

    monkeypatch.setenv("VLLM_GGUF_CUDA_DENSE_KERNEL", mode)
    monkeypatch.setattr(ops, "_cuda_kernel_available", lambda *args: True)
    monkeypatch.setattr(ops, "supports", lambda *args: True)
    monkeypatch.setattr(torch.ops._C_gguf, wrapper, lambda *args: "legacy")
    monkeypatch.setattr(ops, "ggml_mul_mat_a8_triton", lambda *args: "triton")
    monkeypatch.setattr(
        ops, "ggml_dense", lambda *args: pytest.fail("upstream executed")
    )
    assert getattr(ops, wrapper)(torch.empty(1), torch.empty(1), int(Q.Q4_0), 1) == mode


@pytest.mark.parametrize("mode", ["legacy", "triton"])
def test_moe_explicit_backend_stays_available(monkeypatch, mode):
    from vllm_gguf_plugin import ops

    monkeypatch.setenv("VLLM_GGUF_CUDA_MOE_KERNEL", mode)
    monkeypatch.setattr(ops, "_cuda_legacy_moe_available", lambda *args: True)
    monkeypatch.setattr(ops, "supports", lambda *args: True)
    monkeypatch.setattr(torch.ops._C_gguf, "ggml_moe_a8", lambda *args: "legacy")
    monkeypatch.setattr(ops, "ggml_moe_a8_triton", lambda *args: "triton")
    assert ops.ggml_moe_a8(*(torch.empty(1),) * 5, int(Q.Q4_0), 1, 1, 1) == mode


@pytest.mark.parametrize("mode", ["auto", "upstream"])
def test_legacy_moe_entry_cannot_implicitly_run_in_auto(monkeypatch, mode):
    from vllm_gguf_plugin import ops

    monkeypatch.setenv("VLLM_GGUF_CUDA_MOE_KERNEL", mode)
    monkeypatch.setattr(
        ops, "_cuda_legacy_moe_available", lambda *args: pytest.fail("legacy checked")
    )
    with pytest.raises(RuntimeError, match=f"{mode} MoE MMQ"):
        ops.ggml_moe_a8(*(torch.empty(1),) * 5, int(Q.Q4_0), 1, 1, 1)


def test_method_enum_matches_registered_cpp_values():
    from vllm_gguf_plugin.kernel_support import KernelMethod

    for name, method in KernelMethod.__members__.items():
        assert torch.ops._C_gguf.ggml_kernel_method_value(name) == int(method)
    assert torch.ops._C_gguf.ggml_moe_alignment_block_size() == 16


# GGML type catalog


def _catalog_rows() -> dict[str, tuple[str, str, str, bool, bool]]:
    source = CATALOG.read_text()
    rows = re.findall(
        r'X\(GGML_TYPE_(\w+),\s*([^,]+),\s*([^,]+),\s*"([^"]+)",\s*([01]),\s*([01])\)',
        source,
    )
    names = [name for name, *_ in rows]
    assert len(names) == len(set(names))
    return {
        name: (block, cpp_type, label, quantized == "1", mmq == "1")
        for name, block, cpp_type, label, quantized, mmq in rows
    }


def _type_trait_field(body: str, key: str) -> str:
    match = re.search(rf"\.{key}\s*=\s*([^,]+),", body)
    assert match is not None, key
    return match.group(1).strip()


def test_mmq_dispatch_matches_python_capabilities():
    rows = _catalog_rows()
    names = {name for name, (_, _, _, _, mmq) in rows.items() if mmq}

    # The pinned C++ kernels include Q2_0, which some gguf-python versions do
    # not expose. Compare every type visible to this Python installation.
    known = {Q[name] for name in names if name in Q.__members__} | {42}
    assert known == CUDA_UPSTREAM_MMQ_TYPES
    assert CUDA_UPSTREAM_MMVQ_TYPES - known == {Q.IQ1_M}


def test_catalog_matches_ggml_enum_and_python_values():
    header = UPSTREAM / "include/ggml.h"
    if not header.exists():
        pytest.skip("ggml.h is not included in source distributions")
    source = header.read_text()
    enum = source.split("enum ggml_type {", 1)[1].split("};", 1)[0]
    values = {
        name: int(value)
        for name, value in re.findall(
            r"^\s*GGML_TYPE_(\w+)\s*=\s*(\d+)", enum, re.MULTILINE
        )
        if name != "COUNT"
    }
    assert _catalog_rows().keys() == values.keys()
    for name, value in values.items():
        python_type = Q.__members__.get(name)
        if python_type is not None:
            assert int(python_type) == value, name


def test_catalog_matches_pinned_ggml_type_traits():
    traits_file = UPSTREAM / "src/ggml.c"
    if not traits_file.exists():
        pytest.skip("ggml.c is not included in source distributions")
    source = (
        traits_file.read_text()
        .split(
            "static const struct ggml_type_traits type_traits[GGML_TYPE_COUNT] = {",
            1,
        )[1]
        .split("\n};", 1)[0]
    )
    catalog = _catalog_rows()
    traits = {}
    for name, body in re.findall(
        r"^\s*\[GGML_TYPE_(\w+)\]\s*=\s*\{(.*?)^\s*\},",
        source,
        re.MULTILINE | re.DOTALL,
    ):
        if name in catalog:
            traits[name] = (
                _type_trait_field(body, "blck_size"),
                _type_trait_field(body, "type_size"),
                _type_trait_field(body, "type_name").strip('"'),
                _type_trait_field(body, "is_quantized") == "true",
            )

    assert traits.keys() == catalog.keys()
    for name, (block, cpp_type, label, quantized, _) in catalog.items():
        assert traits[name] == (block, f"sizeof({cpp_type})", label, quantized)


# Architecture policy and batch boundary


@pytest.fixture
def native_policy():
    pytest.importorskip("vllm_gguf_plugin._C_gguf")
    if torch.version.hip is not None:
        pytest.skip("ROCm uses legacy kernels")
    return torch.ops._C_gguf.ggml_should_use_mmvq


@pytest.mark.parametrize(
    "cc,limits",
    [
        (700, {}),  # Volta
        (800, {}),  # Ampere
        (870, dict.fromkeys(K_TYPES, 1)),  # Orin
        (890, {Q.Q2_K: 4, Q.Q3_K: 6}),  # Ada
        (900, {}),  # Hopper
        (1000, {}),  # Datacenter Blackwell uses the upstream default
        (1200, {Q.Q2_K: 5, Q.Q3_K: 5, Q.Q4_K: 5, Q.Q5_K: 6, Q.Q6_K: 7}),
        (1210, {Q.Q2_K: 6}),  # DGX Spark
    ],
)
def test_native_mmvq_policy(native_policy, cc, limits):
    """Check compiled architecture thresholds for batches 1 through 16."""
    for quant_type in (*K_TYPES, Q.Q4_0, Q.IQ4_NL, Q.IQ1_M):
        for batch in range(1, 17):
            use_mmvq = batch <= limits.get(quant_type, 8)
            assert native_policy(int(quant_type), cc, batch) == use_mmvq


@pytest.mark.parametrize(
    "quant_type,cc,batch",
    [
        (-1, 890, 1),
        (999, 890, 1),
        (int(Q.F32), 890, 1),
        (int(Q.Q2_K), 890, 0),
        (int(Q.Q2_K), 890, -1),
        (int(Q.Q2_K), 0, 1),
        (int(Q.Q2_K), 2**40, 1),
    ],
)
def test_native_mmvq_policy_rejects_invalid_inputs(
    native_policy, quant_type, cc, batch
):
    assert not native_policy(quant_type, cc, batch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("quant_type", K_TYPES, ids=lambda q: q.name)
@pytest.mark.parametrize("mode", ["auto", "upstream"])
@torch.inference_mode()
def test_dense_k_quant_dispatch_batches_1_to_16(
    native_policy, monkeypatch, quant_type, mode
):
    from vllm_gguf_plugin import ops
    from vllm_gguf_plugin.quantization.linear import (
        _fused_mul_mat_gguf,
        fused_mul_mat_gguf,
    )

    monkeypatch.setenv("VLLM_GGUF_CUDA_DENSE_KERNEL", mode)
    n, k = 37, 768  # Partial output tile and a padded final K block.
    block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]
    rng = np.random.default_rng(int(quant_type))
    blocks = rng.integers(0, 256, (n * k // block_size, type_size), dtype=np.uint8)
    # K-quant encoders are not available in gguf-python. Random packed values
    # with finite, small fp16 scales give nontrivial dequantized references.
    if quant_type == Q.Q2_K:
        scale_offsets = (type_size - 4, type_size - 2)
    elif quant_type in (Q.Q4_K, Q.Q5_K):
        scale_offsets = (0, 2)
    else:
        scale_offsets = (type_size - 2,)
    for offset in scale_offsets:
        blocks[:, offset : offset + 2] = np.array([1 / 256], dtype=np.float16).view(
            np.uint8
        )
    packed = blocks.reshape(n, -1)
    weight = make_padded_weight(packed, quant_type, k)
    dense = torch.from_numpy(gguf.dequantize(packed, quant_type)).cuda()
    x_all = torch.from_numpy(rng.standard_normal((16, k), dtype=np.float32)).cuda()
    for batch in range(1, 17):
        x = x_all[:batch]
        actual = _fused_mul_mat_gguf(x, weight, int(quant_type))
        reference = x @ dense.T
        torch.testing.assert_close(
            actual, reference, atol=0.01 * reference.abs().max().item(), rtol=0.03
        )
        caps = ops.dense_upstream_capabilities(weight, x, int(quant_type), n)
        if caps & ops.DENSE_MMVQ:
            selected_op = torch.ops._C_gguf.ggml_dense_mmvq
        elif caps & ops.DENSE_MMQ:
            selected_op = torch.ops._C_gguf.ggml_dense_mmq
        else:
            selected_op = torch.ops._C_gguf.ggml_dense_dequantize_blas
        expected = selected_op(weight, x, int(quant_type), n)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    # Exercise the actual vLLM custom operator, including its host policy
    # query, while capturing and replaying with changed inputs.
    for batch in (5, 9):
        x = x_all[:batch].clone()
        fused_mul_mat_gguf(x, weight, int(quant_type))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fused_mul_mat_gguf(x, weight, int(quant_type))
        x.mul_(0.5)
        graph.replay()
        expected = fused_mul_mat_gguf(x, weight, int(quant_type))
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)


def test_fixed_method_python_argument_names_are_consistent():
    import inspect

    from vllm_gguf_plugin import ops

    dense_args = ("W", "X", "quant_type", "row")
    moe_args = ("X", "W", "topk_ids", "quant_type", "row", "top_k", "tokens")
    for method in ("mmvq", "mmq", "mmvf", "mmf", "blas", "dequantize_blas"):
        assert (
            tuple(inspect.signature(getattr(ops, f"ggml_dense_{method}")).parameters)
            == dense_args
        )
        assert (
            tuple(inspect.signature(getattr(ops, f"ggml_moe_{method}")).parameters)
            == moe_args
        )
    assert tuple(inspect.signature(ops.ggml_moe_grouped_dense).parameters) == moe_args
