import torch

from ..gemm.utils import (
    GGML_TYPE_IQ1_M,
    GGML_TYPE_IQ1_S,
    GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ2_XS,
    GGML_TYPE_IQ2_XXS,
    GGML_TYPE_IQ3_S,
    GGML_TYPE_IQ3_XXS,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_IQ4_XS,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q8_1,
    GGML_TYPE_STQ1_0,
)
from .iq_quant import (
    ggml_dequantize_iq1_m_triton,
    ggml_dequantize_iq1_s_triton,
    ggml_dequantize_iq2_s_triton,
    ggml_dequantize_iq2_xs_triton,
    ggml_dequantize_iq2_xxs_triton,
    ggml_dequantize_iq3_s_triton,
    ggml_dequantize_iq3_xxs_triton,
    ggml_dequantize_iq4_nl_triton,
    ggml_dequantize_iq4_xs_triton,
)
from .k_quant import (
    ggml_dequantize_q2_k_triton,
    ggml_dequantize_q3_k_triton,
    ggml_dequantize_q4_k_triton,
    ggml_dequantize_q5_k_triton,
    ggml_dequantize_q6_k_triton,
)
from .standard_quant import (
    ggml_dequantize_q4_0_triton,
    ggml_dequantize_q4_1_triton,
    ggml_dequantize_q5_0_triton,
    ggml_dequantize_q5_1_triton,
    ggml_dequantize_q8_0_triton,
    ggml_dequantize_q8_1_triton,
)


def _ggml_dequantize_stq1_0(
    W: torch.Tensor, m: int, n: int, dtype: torch.dtype | None = None
) -> torch.Tensor:
    # Pure-torch reference (GPU-vectorized); used only when the CUDA
    # extension is unavailable.
    from ...gguf_stq import dequantize_stq1_0

    return dequantize_stq1_0(W).reshape(m, n).to(dtype or torch.float16)


def ggml_dequantize_triton(
    W: torch.Tensor,
    quant_type: int,
    m: int,
    n: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    kernel = {
        GGML_TYPE_IQ1_M: ggml_dequantize_iq1_m_triton,
        GGML_TYPE_IQ1_S: ggml_dequantize_iq1_s_triton,
        GGML_TYPE_IQ2_S: ggml_dequantize_iq2_s_triton,
        GGML_TYPE_IQ2_XXS: ggml_dequantize_iq2_xxs_triton,
        GGML_TYPE_IQ2_XS: ggml_dequantize_iq2_xs_triton,
        GGML_TYPE_IQ3_S: ggml_dequantize_iq3_s_triton,
        GGML_TYPE_IQ3_XXS: ggml_dequantize_iq3_xxs_triton,
        GGML_TYPE_IQ4_NL: ggml_dequantize_iq4_nl_triton,
        GGML_TYPE_IQ4_XS: ggml_dequantize_iq4_xs_triton,
        GGML_TYPE_Q2_K: ggml_dequantize_q2_k_triton,
        GGML_TYPE_Q3_K: ggml_dequantize_q3_k_triton,
        GGML_TYPE_Q4_0: ggml_dequantize_q4_0_triton,
        GGML_TYPE_Q4_1: ggml_dequantize_q4_1_triton,
        GGML_TYPE_Q4_K: ggml_dequantize_q4_k_triton,
        GGML_TYPE_Q5_0: ggml_dequantize_q5_0_triton,
        GGML_TYPE_Q5_1: ggml_dequantize_q5_1_triton,
        GGML_TYPE_Q5_K: ggml_dequantize_q5_k_triton,
        GGML_TYPE_Q6_K: ggml_dequantize_q6_k_triton,
        GGML_TYPE_Q8_0: ggml_dequantize_q8_0_triton,
        GGML_TYPE_Q8_1: ggml_dequantize_q8_1_triton,
        GGML_TYPE_STQ1_0: _ggml_dequantize_stq1_0,
    }.get(int(quant_type))
    if kernel is None:
        raise ValueError(f"Unsupported Triton dequant quant type: {quant_type}")
    return kernel(W, m, n, dtype)
