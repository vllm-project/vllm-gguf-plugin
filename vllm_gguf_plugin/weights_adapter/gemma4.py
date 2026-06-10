# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterable

import torch

from .default import GGUFWeightsAdapter


def _flatten_gemma4_patch_embed_weight(weight: torch.Tensor) -> torch.Tensor:
    return weight.flatten(1) if weight.dim() == 4 else weight


def _transform_gemma4_weight_name(name: str) -> str:
    replacements = {
        ".mlp.experts.gate_up_proj.qweight_type": (
            ".mlp.moe.experts.routed_experts.w13_qweight_type"
        ),
        ".mlp.experts.gate_up_proj.qweight": (
            ".mlp.moe.experts.routed_experts.w13_qweight"
        ),
        ".mlp.experts.down_proj.qweight_type": (
            ".mlp.moe.experts.routed_experts.w2_qweight_type"
        ),
        ".mlp.experts.down_proj.qweight": (
            ".mlp.moe.experts.routed_experts.w2_qweight"
        ),
    }
    for old, new in replacements.items():
        if name.endswith(old):
            return name.removesuffix(old) + new
    return name


class Gemma4GGUFAdapter(GGUFWeightsAdapter):
    """Adapter for Gemma4 GGUF models."""

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type in ("gemma4", "gemma4_assistant", "gemma4_mtp")

    def map_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for name, weight in weights:
            name = _transform_gemma4_weight_name(name)
            yield name, self.transform_weight(name, weight)

    def transform_weight(
        self,
        hf_name: str,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        if hf_name == "model.vision_tower.patch_embedder.input_proj.weight":
            return _flatten_gemma4_patch_embed_weight(weight)
        return weight
