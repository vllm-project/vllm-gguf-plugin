# SPDX-License-Identifier: Apache-2.0

import torch

from ..gemm.utils import (
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q8_1,
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
        GGML_TYPE_Q4_0,
        GGML_TYPE_Q4_1,
        GGML_TYPE_Q5_0,
        GGML_TYPE_Q5_1,
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q8_1,
    }
)

TRITON_MOE_DISPATCH = {
    GGML_TYPE_Q4_0: ggml_moe_q4_0_triton,
    GGML_TYPE_Q4_1: ggml_moe_q4_1_triton,
    GGML_TYPE_Q5_0: ggml_moe_q5_0_triton,
    GGML_TYPE_Q5_1: ggml_moe_q5_1_triton,
    GGML_TYPE_Q8_0: ggml_moe_q8_0_triton,
    GGML_TYPE_Q8_1: ggml_moe_q8_1_triton,
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
