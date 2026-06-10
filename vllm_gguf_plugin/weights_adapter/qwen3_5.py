# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterable

import torch

from .default import GGUFWeightsAdapter

_QWEN3_5_PATCH_EMBED_WEIGHT = "model.visual.patch_embed.proj.weight"
_QWEN3_5_PATCH_EMBED_WEIGHT_1 = f"{_QWEN3_5_PATCH_EMBED_WEIGHT}.1"


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
        return config.model_type in ("qwen3_5", "qwen3_5_moe", "qwen3_5_mtp")

    def map_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        patch_weight: torch.Tensor | None = None
        patch_weight_1: torch.Tensor | None = None

        for hf_name, weight in weights:
            if hf_name == _QWEN3_5_PATCH_EMBED_WEIGHT:
                patch_weight = weight
                continue
            if hf_name == _QWEN3_5_PATCH_EMBED_WEIGHT_1:
                patch_weight_1 = weight
                continue
            yield hf_name, self.transform_weight(hf_name, weight)

        if patch_weight is None:
            if patch_weight_1 is not None:
                yield _QWEN3_5_PATCH_EMBED_WEIGHT_1, patch_weight_1
            return

        if patch_weight_1 is not None:
            patch_weight = torch.stack((patch_weight, patch_weight_1), dim=2)
        yield _QWEN3_5_PATCH_EMBED_WEIGHT, self.transform_weight(
            _QWEN3_5_PATCH_EMBED_WEIGHT, patch_weight
        )

    def transform_weight(
        self,
        hf_name: str,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        return _maybe_reshape_qwen3_5_gguf_weight(hf_name, weight)
