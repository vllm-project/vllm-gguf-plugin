import pytest
import torch
from gguf import GGMLQuantizationType, dequantize

import vllm_gguf_plugin.ops as ops
from .utils import seed_everything, get_gguf_sample_tensors


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
            torch.tensor(tensor.data, device="cuda"), quant_type, *list(shape), dtype=dtype
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
