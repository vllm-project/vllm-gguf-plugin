# SPDX-License-Identifier: Apache-2.0

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
)
from .iq_quant import (
    ggml_moe_iq1_m_triton,
    ggml_moe_iq1_s_triton,
    ggml_moe_iq2_s_triton,
    ggml_moe_iq2_xs_triton,
    ggml_moe_iq2_xxs_triton,
    ggml_moe_iq3_s_triton,
    ggml_moe_iq3_xxs_triton,
    ggml_moe_iq4_nl_triton,
    ggml_moe_iq4_xs_triton,
)
from .k_quant import (
    ggml_moe_q2_k_triton,
    ggml_moe_q3_k_triton,
    ggml_moe_q4_k_triton,
    ggml_moe_q5_k_triton,
    ggml_moe_q6_k_triton,
)
from .standard_quant import (
    ggml_moe_q4_0_triton,
    ggml_moe_q4_1_triton,
    ggml_moe_q5_0_triton,
    ggml_moe_q5_1_triton,
    ggml_moe_q8_0_triton,
    ggml_moe_q8_1_triton,
)

TRITON_MOE_SUPPORTED_TYPES = frozenset(
    {
        GGML_TYPE_IQ1_M,
        GGML_TYPE_IQ1_S,
        GGML_TYPE_IQ2_S,
        GGML_TYPE_IQ2_XXS,
        GGML_TYPE_IQ2_XS,
        GGML_TYPE_IQ3_S,
        GGML_TYPE_IQ3_XXS,
        GGML_TYPE_IQ4_NL,
        GGML_TYPE_IQ4_XS,
        GGML_TYPE_Q4_0,
        GGML_TYPE_Q4_1,
        GGML_TYPE_Q5_0,
        GGML_TYPE_Q5_1,
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q8_1,
        GGML_TYPE_Q2_K,
        GGML_TYPE_Q3_K,
        GGML_TYPE_Q4_K,
        GGML_TYPE_Q5_K,
        GGML_TYPE_Q6_K,
    }
)

TRITON_MOE_DISPATCH = {
    GGML_TYPE_IQ1_M: ggml_moe_iq1_m_triton,
    GGML_TYPE_IQ1_S: ggml_moe_iq1_s_triton,
    GGML_TYPE_IQ2_S: ggml_moe_iq2_s_triton,
    GGML_TYPE_IQ2_XXS: ggml_moe_iq2_xxs_triton,
    GGML_TYPE_IQ2_XS: ggml_moe_iq2_xs_triton,
    GGML_TYPE_IQ3_S: ggml_moe_iq3_s_triton,
    GGML_TYPE_IQ3_XXS: ggml_moe_iq3_xxs_triton,
    GGML_TYPE_IQ4_NL: ggml_moe_iq4_nl_triton,
    GGML_TYPE_IQ4_XS: ggml_moe_iq4_xs_triton,
    GGML_TYPE_Q4_0: ggml_moe_q4_0_triton,
    GGML_TYPE_Q4_1: ggml_moe_q4_1_triton,
    GGML_TYPE_Q5_0: ggml_moe_q5_0_triton,
    GGML_TYPE_Q5_1: ggml_moe_q5_1_triton,
    GGML_TYPE_Q8_0: ggml_moe_q8_0_triton,
    GGML_TYPE_Q8_1: ggml_moe_q8_1_triton,
    GGML_TYPE_Q2_K: ggml_moe_q2_k_triton,
    GGML_TYPE_Q3_K: ggml_moe_q3_k_triton,
    GGML_TYPE_Q4_K: ggml_moe_q4_k_triton,
    GGML_TYPE_Q5_K: ggml_moe_q5_k_triton,
    GGML_TYPE_Q6_K: ggml_moe_q6_k_triton,
}


def ggml_moe_a8_triton(
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    quant_type: int,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    if quant_type not in TRITON_MOE_DISPATCH:
        raise ValueError(f"Unsupported Triton fused MoE quant type: {quant_type}")
    return TRITON_MOE_DISPATCH[quant_type](
        X,
        W,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        row,
        top_k,
        tokens,
    )
