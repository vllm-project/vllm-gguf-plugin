import torch

from .iq1_m import ggml_gemm_iq1_m_triton
from .iq1_s import ggml_gemm_iq1_s_triton
from .iq2_s import ggml_gemm_iq2_s_triton
from .iq2_xs import ggml_gemm_iq2_xs_triton
from .iq3_s import ggml_gemm_iq3_s_triton
from .iq3_xxs import ggml_gemm_iq3_xxs_triton
from .iq4_nl import ggml_gemm_iq4_nl_triton
from .iq4_xs import ggml_gemm_iq4_xs_triton
from .q2_k import ggml_gemm_q2_k_triton
from .q3_k import ggml_gemm_q3_k_triton
from .q4_0 import ggml_gemm_q4_0_triton
from .q4_1 import ggml_gemm_q4_1_triton
from .q4_k import ggml_gemm_q4_k_triton
from .q5_0 import ggml_gemm_q5_0_triton
from .q5_1 import ggml_gemm_q5_1_triton
from .q5_k import ggml_gemm_q5_k_triton
from .q6_k import ggml_gemm_q6_k_triton
from .q8_0 import ggml_gemm_q8_0_triton
from .q8_1 import ggml_gemm_q8_1_triton
from .utils import (
    GGML_TYPE_IQ1_M,
    GGML_TYPE_IQ1_S,
    GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ2_XS,
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
)


def ggml_mul_mat_a8_triton(W: torch.Tensor, X: torch.Tensor, quant_type: int, row: int) -> torch.Tensor:
    kernel = {
        GGML_TYPE_IQ1_M: ggml_gemm_iq1_m_triton,
        GGML_TYPE_IQ1_S: ggml_gemm_iq1_s_triton,
        GGML_TYPE_IQ2_S: ggml_gemm_iq2_s_triton,
        GGML_TYPE_IQ2_XS: ggml_gemm_iq2_xs_triton,
        GGML_TYPE_IQ3_S: ggml_gemm_iq3_s_triton,
        GGML_TYPE_IQ3_XXS: ggml_gemm_iq3_xxs_triton,
        GGML_TYPE_IQ4_NL: ggml_gemm_iq4_nl_triton,
        GGML_TYPE_IQ4_XS: ggml_gemm_iq4_xs_triton,
        GGML_TYPE_Q2_K: ggml_gemm_q2_k_triton,
        GGML_TYPE_Q3_K: ggml_gemm_q3_k_triton,
        GGML_TYPE_Q4_0: ggml_gemm_q4_0_triton,
        GGML_TYPE_Q4_1: ggml_gemm_q4_1_triton,
        GGML_TYPE_Q4_K: ggml_gemm_q4_k_triton,
        GGML_TYPE_Q5_0: ggml_gemm_q5_0_triton,
        GGML_TYPE_Q5_1: ggml_gemm_q5_1_triton,
        GGML_TYPE_Q5_K: ggml_gemm_q5_k_triton,
        GGML_TYPE_Q6_K: ggml_gemm_q6_k_triton,
        GGML_TYPE_Q8_0: ggml_gemm_q8_0_triton,
        GGML_TYPE_Q8_1: ggml_gemm_q8_1_triton,
    }.get(int(quant_type))
    if kernel is None:
        raise ValueError(f"Unsupported Triton quant type: {quant_type}")
    return kernel(W, X, row)
