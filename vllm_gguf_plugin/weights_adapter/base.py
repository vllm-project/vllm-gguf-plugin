# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig


@dataclass(slots=True)
class GGUFLoadSpec:
    weights_source: list[str]
    unquantized_modules: list[str]
    gguf_to_hf_name_map: dict[str, str] | None = None


class BaseGGUFWeightsAdapter(ABC):
    """Base hooks for GGUF weight loading adapters."""

    def __init__(self, config: PretrainedConfig) -> None:
        self.config = config

    @classmethod
    @abstractmethod
    def matches(cls, config: PretrainedConfig) -> bool:
        """Return whether this adapter supports *config*."""

    @abstractmethod
    def prepare_weights(self, model_config: ModelConfig) -> None:
        """Return HF-style weights."""

    @abstractmethod
    def prepare_loading(self, model_config: ModelConfig) -> None:
        """Preparation before loading, e.g., patching the HF config."""

    def patch_hf_config(
        self,
        model_path: str,
        hf_config: PretrainedConfig,
    ) -> PretrainedConfig:
        """Patch HF config before model init."""
        del model_path
        return hf_config

    def transform_weight(
        self,
        hf_name: str,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """Transform one loaded weight."""
        del hf_name
        return weight
