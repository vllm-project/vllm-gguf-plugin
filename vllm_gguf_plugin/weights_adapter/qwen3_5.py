# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from .default import GGUFWeightsAdapter


def _maybe_reshape_qwen3_5_gguf_weight(
    name: str,
    weight: torch.Tensor,
) -> torch.Tensor:
    if "mlp.shared_expert_gate" in name and weight.dim() == 1:
        return weight[None, :]
    if "linear_attn.conv1d.weight" in name and weight.dim() == 2:
        return weight[:, None, :]
    return weight


class Qwen3_5GGUFAdapter(GGUFWeightsAdapter):
    """Adapter for Qwen3.5 GGUF models."""

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type in ("qwen3_5", "qwen3_5_moe")

    def transform_weight(
        self,
        hf_name: str,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        return _maybe_reshape_qwen3_5_gguf_weight(hf_name, weight)
