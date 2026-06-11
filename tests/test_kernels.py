import pytest
import torch
from gguf import GGMLQuantizationType, dequantize
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

import vllm_gguf_plugin.ops as ops
from vllm_gguf_plugin.quantization.fused_moe import _fused_moe_gguf
from vllm_gguf_plugin.triton.fused_moe import ggml_moe_a8_triton
from vllm_gguf_plugin.triton.fused_moe.utils import get_triton_moe_block_m
from vllm_gguf_plugin.triton.gemm.interface import ggml_mul_mat_a8_triton

from .utils import get_gguf_moe_tensors, get_gguf_sample_tensors, seed_everything

DTYPES = [torch.half, torch.bfloat16, torch.float32]
# Hidden_size for testing, must match the sample file in HF repo,
# we have `hidden_size = 256, 1024` for test in HF repo currently.
BATCH_SIZES = [2, 4, 8]
HIDDEN_SIZES = [256, 1024]
NUM_TOKENS = [7, 83, 128, 2048]  # Arbitrary values for testing
SEEDS = [0]
QUANT_TYPES = [
    # i-matrix
    GGMLQuantizationType.IQ1_M,
    GGMLQuantizationType.IQ1_S,
    GGMLQuantizationType.IQ2_XXS,
    GGMLQuantizationType.IQ2_S,
    GGMLQuantizationType.IQ2_XS,
    GGMLQuantizationType.IQ3_S,
    GGMLQuantizationType.IQ3_XXS,
    GGMLQuantizationType.IQ4_NL,
    GGMLQuantizationType.IQ4_XS,
    # k-quants
    GGMLQuantizationType.Q2_K,
    GGMLQuantizationType.Q3_K,
    GGMLQuantizationType.Q4_K,
    GGMLQuantizationType.Q5_K,
    GGMLQuantizationType.Q6_K,
    # standard quantization
    GGMLQuantizationType.Q4_0,
    GGMLQuantizationType.Q5_0,
    GGMLQuantizationType.Q8_0,
]
TRITON_MOE_QUANT_TYPES = [
    GGMLQuantizationType.IQ1_M,
    GGMLQuantizationType.IQ1_S,
    GGMLQuantizationType.IQ2_XXS,
    GGMLQuantizationType.IQ2_S,
    GGMLQuantizationType.IQ2_XS,
    GGMLQuantizationType.IQ3_S,
    GGMLQuantizationType.IQ3_XXS,
    GGMLQuantizationType.IQ4_NL,
    GGMLQuantizationType.IQ4_XS,
    GGMLQuantizationType.Q2_K,
    GGMLQuantizationType.Q3_K,
    GGMLQuantizationType.Q4_K,
    GGMLQuantizationType.Q5_K,
    GGMLQuantizationType.Q6_K,
    GGMLQuantizationType.Q4_0,
    GGMLQuantizationType.Q5_0,
    GGMLQuantizationType.Q8_0,
]


def _silu_and_mul(inp: torch.Tensor) -> torch.Tensor:
    d = inp.shape[-1] // 2
    out = torch.empty(inp.shape[:-1] + (d,), dtype=inp.dtype, device=inp.device)
    apply_moe_activation(MoEActivation.SILU, out, inp)
    return out


@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_type", QUANT_TYPES)
@torch.inference_mode()
def test_dequantize(
    hidden_size: int, dtype: torch.dtype, quant_type: GGMLQuantizationType
):
    tensors = get_gguf_sample_tensors(hidden_size, quant_type)
    for tensor in tensors:
        shape_str = tensor.name.split("_")[-1]
        shape = map(int, shape_str.split("x"))

        ref_output = torch.tensor(
            dequantize(tensor.data, quant_type), device="cuda"
        ).to(dtype)
        output = ops.ggml_dequantize(
            torch.tensor(tensor.data, device="cuda"),
            quant_type,
            *list(shape),
            dtype=dtype,
        )

        torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=4e-2)


@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_type", QUANT_TYPES)
@torch.inference_mode()
def test_mmvq(hidden_size: int, dtype: torch.dtype, quant_type: GGMLQuantizationType):
    seed_everything(0)

    tensors = get_gguf_sample_tensors(hidden_size, quant_type)
    x = torch.rand((1, hidden_size), dtype=dtype, device="cuda")
    for tensor in tensors:
        weight = torch.tensor(dequantize(tensor.data, quant_type), device="cuda").to(
            dtype
        )
        ref_output = x @ weight.T

        qweight = torch.tensor(tensor.data, device="cuda")
        output = ops.ggml_mul_mat_vec_a8(qweight, x, quant_type, qweight.shape[0]).to(
            dtype
        )

        torch.testing.assert_close(output, ref_output, atol=1, rtol=1e-1)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "quant_type, name",
    [
        # k-quants
        (GGMLQuantizationType.Q2_K, "Q2_K"),
        (GGMLQuantizationType.Q3_K, "Q3_K"),
        (GGMLQuantizationType.Q4_K, "Q4_K"),
        (GGMLQuantizationType.Q5_K, "Q5_K"),
        (GGMLQuantizationType.Q6_K, "Q6_K"),
        # standard quants
        (GGMLQuantizationType.Q4_0, "Q4_0"),
        (GGMLQuantizationType.Q5_0, "Q5_0"),
        (GGMLQuantizationType.Q8_0, "Q8_0"),
    ],
)
@torch.inference_mode()
def test_mmq(
    name: str,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    quant_type: GGMLQuantizationType,
):
    seed_everything(0)

    tensors = get_gguf_sample_tensors(hidden_size, quant_type)
    x = torch.rand((num_tokens, hidden_size), dtype=dtype, device="cuda")
    for tensor in tensors:
        weight = torch.tensor(dequantize(tensor.data, quant_type), device="cuda").to(
            dtype
        )
        ref_output = x @ weight.T

        qweight = torch.tensor(tensor.data, device="cuda")
        output = ops.ggml_mul_mat_a8(qweight, x, quant_type, qweight.shape[0]).to(dtype)

        atols = {torch.half: 1, torch.bfloat16: 1.5, torch.float: 1.2}
        # test matrix has inputs centered around 0 and lower precision from
        # bfloat16 tends to accumulate and can greatly inflate rtol
        # since outputs are also very close to 0
        rtols = {torch.half: 1e-1, torch.bfloat16: 1e4, torch.float: 2e1}
        torch.testing.assert_close(
            output, ref_output, atol=atols[dtype], rtol=rtols[dtype]
        )


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "quant_type, name",
    [
        # i-matrix quants
        (GGMLQuantizationType.IQ1_M, "IQ1_M"),
        (GGMLQuantizationType.IQ1_S, "IQ1_S"),
        (GGMLQuantizationType.IQ2_XXS, "IQ2_XXS"),
        (GGMLQuantizationType.IQ2_S, "IQ2_S"),
        (GGMLQuantizationType.IQ2_XS, "IQ2_XS"),
        (GGMLQuantizationType.IQ3_S, "IQ3_S"),
        (GGMLQuantizationType.IQ3_XXS, "IQ3_XXS"),
        (GGMLQuantizationType.IQ4_NL, "IQ4_NL"),
        (GGMLQuantizationType.IQ4_XS, "IQ4_XS"),
        # k-quants
        (GGMLQuantizationType.Q2_K, "Q2_K"),
        (GGMLQuantizationType.Q3_K, "Q3_K"),
        (GGMLQuantizationType.Q4_K, "Q4_K"),
        (GGMLQuantizationType.Q5_K, "Q5_K"),
        (GGMLQuantizationType.Q6_K, "Q6_K"),
        # standard quants
        (GGMLQuantizationType.Q4_0, "Q4_0"),
        (GGMLQuantizationType.Q5_0, "Q5_0"),
        (GGMLQuantizationType.Q8_0, "Q8_0"),
    ],
)
@torch.inference_mode()
def test_mmq_triton_dispatch(
    name: str,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    quant_type: GGMLQuantizationType,
):
    seed_everything(0)

    tensors = get_gguf_sample_tensors(hidden_size, quant_type)
    x = torch.rand((num_tokens, hidden_size), dtype=dtype, device="cuda")
    for tensor in tensors:
        weight = torch.tensor(dequantize(tensor.data, quant_type), device="cuda").to(
            dtype
        )
        ref_output = x @ weight.T

        qweight = torch.tensor(tensor.data, device="cuda")
        output = ggml_mul_mat_a8_triton(qweight, x, quant_type, qweight.shape[0])

        atols = {torch.half: 1, torch.bfloat16: 5, torch.float: 1.2}
        # test matrix has inputs centered around 0 and lower precision from
        # bfloat16 tends to accumulate and can greatly inflate rtol
        # since outputs are also very close to 0
        rtols = {torch.half: 1e-1, torch.bfloat16: 1e4, torch.float: 2e1}
        torch.testing.assert_close(
            output, ref_output, atol=atols[dtype], rtol=rtols[dtype]
        )


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "quant_type, name",
    [
        # k-quants
        (GGMLQuantizationType.Q2_K, "Q2_K"),
        (GGMLQuantizationType.Q3_K, "Q3_K"),
        (GGMLQuantizationType.Q4_K, "Q4_K"),
        (GGMLQuantizationType.Q5_K, "Q5_K"),
        (GGMLQuantizationType.Q6_K, "Q6_K"),
        # standard quants
        (GGMLQuantizationType.Q4_0, "Q4_0"),
        (GGMLQuantizationType.Q5_0, "Q5_0"),
        (GGMLQuantizationType.Q8_0, "Q8_0"),
    ],
)
@pytest.mark.skip(reason="Current CUDA Kernel hasn't supported invarlen")
@torch.inference_mode()
def test_mmq_batching(
    name: str,
    batch_size: int,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    quant_type: GGMLQuantizationType,
):
    seed_everything(0)

    tensors = get_gguf_sample_tensors(hidden_size, quant_type)
    x = torch.rand((batch_size, num_tokens, hidden_size), dtype=dtype, device="cuda")
    for tensor in tensors:
        weight = torch.tensor(dequantize(tensor.data, quant_type), device="cuda").to(
            dtype
        )
        ref_output = x @ weight.T

        qweight = torch.tensor(tensor.data, device="cuda")
        output = ops.ggml_mul_mat_a8(qweight, x, quant_type, qweight.shape[0]).to(dtype)

        atols = {torch.half: 1, torch.bfloat16: 2, torch.float: 1}
        # test matrix has inputs centered around 0 and lower precision from
        # bfloat16 tends to accumulate and can greatly inflate rtol
        # since outputs are also very close to 0
        rtols = {torch.half: 1e-1, torch.bfloat16: 1e-1, torch.float: 2e-1}
        torch.testing.assert_close(
            output, ref_output, atol=atols[dtype], rtol=rtols[dtype]
        )
    # FIXME: X will cause nan values in full test suite, need to investigate
    del x
    torch.cuda.empty_cache()


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", [512])
@pytest.mark.parametrize("top_k", [4, 8])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_type", QUANT_TYPES)
@torch.inference_mode()
def test_moe(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    quant_type: GGMLQuantizationType,
    top_k: int,
):
    seed_everything(0)
    H, E = 1024, 256

    x = torch.rand((num_tokens, H), dtype=dtype, device="cuda")

    topk_weights = torch.rand(num_tokens, top_k, device="cuda", dtype=dtype)
    topk_ids = torch.randint(
        0, E, (num_tokens, top_k), device="cuda", dtype=torch.int32
    )

    tensors = get_gguf_moe_tensors(hidden_size, quant_type)

    w13 = tensors[0]
    w2 = tensors[1]

    w13_dequant = torch.tensor(dequantize(w13.data, quant_type), device="cuda").to(
        dtype
    )

    w2_dequant = torch.tensor(dequantize(w2.data, quant_type), device="cuda").to(dtype)

    output = _fused_moe_gguf(
        x,
        torch.tensor(w13.data, device="cuda"),
        torch.tensor(w2.data, device="cuda"),
        topk_weights,
        topk_ids,
        quant_type,
        quant_type,
        "silu",
    )

    ref_output = fused_experts(
        x, w13_dequant, w2_dequant, topk_weights, topk_ids
    ).reshape(output.shape)
    torch.testing.assert_close(output, ref_output, atol=1, rtol=1e-1)


@pytest.mark.parametrize("num_tokens", [83, 128])
@pytest.mark.parametrize("hidden_size", [512])
@pytest.mark.parametrize("top_k", [4, 8])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_type", TRITON_MOE_QUANT_TYPES)
@torch.inference_mode()
def test_moe_triton(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    quant_type: GGMLQuantizationType,
    top_k: int,
):
    seed_everything(0)
    H, E = 1024, 256

    x = torch.rand((num_tokens, H), dtype=dtype, device="cuda")
    topk_weights = torch.rand(num_tokens, top_k, device="cuda", dtype=dtype)
    topk_ids = torch.randint(
        0, E, (num_tokens, top_k), device="cuda", dtype=torch.int32
    )

    w13, w2 = get_gguf_moe_tensors(hidden_size, quant_type)
    w13_q = torch.tensor(w13.data, device="cuda")
    w2_q = torch.tensor(w2.data, device="cuda")
    w13_dequant = torch.tensor(dequantize(w13.data, quant_type), device="cuda").to(
        dtype
    )
    w2_dequant = torch.tensor(dequantize(w2.data, quant_type), device="cuda").to(dtype)

    block_size = get_triton_moe_block_m(quant_type)
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size, E
    )

    out = ggml_moe_a8_triton(
        x,
        w13_q,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        quant_type,
        w13_q.shape[1],
        top_k,
        num_tokens,
    )
    out = _silu_and_mul(out)
    out = ggml_moe_a8_triton(
        out,
        w2_q,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        quant_type,
        w2_q.shape[1],
        1,
        num_tokens * top_k,
    )
    out = out.reshape(num_tokens, top_k, w2_q.shape[1]).mul_(
        topk_weights.view(num_tokens, top_k, 1)
    )
    output = torch.empty_like(x)
    ops.moe_sum(out, output)

    ref_output = fused_experts(
        x, w13_dequant, w2_dequant, topk_weights, topk_ids
    ).reshape(output.shape)
    torch.testing.assert_close(output, ref_output, atol=1, rtol=1e-1)
