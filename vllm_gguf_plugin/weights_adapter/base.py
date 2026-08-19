# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

from ..gguf_files import GGUFModelFiles

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig

    from ..quantization.layout import GGUFLinearLayout


GGUFWeight = tuple[str, torch.Tensor]


class BaseGGUFWeightsAdapter(ABC):
    """Model-specific GGUF name mapping and tensor transformation hooks."""

    #: Modules that never load weights from GGUF (e.g. shared with the target
    #: model in speculative decoding) and must stay unquantized.
    extra_unquantized_modules: tuple[str, ...] = ()

    #: Modalities this adapter cannot reconstruct from GGUF weights.  Some
    #: converters fold away information that one modality needs while leaving
    #: the rest of the model intact; listing it here drops it from the model's
    #: supported multimodal limits, so requests carrying it are rejected during
    #: input validation instead of running against weights that cannot
    #: represent them.
    UNSUPPORTED_MODALITIES: tuple[str, ...] = ()

    @classmethod
    @abstractmethod
    def matches(cls, config: PretrainedConfig) -> bool:
        """Return whether this adapter supports *config*."""

    @classmethod
    def architecture(cls, config: PretrainedConfig) -> str | None:
        """Return an architecture override required before model loading."""
        del config
        return None

    @abstractmethod
    def build_name_map(
        self,
        files: GGUFModelFiles,
        model_config: ModelConfig,
    ) -> dict[str, str]:
        """Map raw GGUF tensor names to names accepted by the model."""

    def patch_hf_config(
        self,
        files: GGUFModelFiles,
        hf_config: PretrainedConfig,
    ) -> PretrainedConfig:
        """Patch HF config before model init."""
        del files
        return hf_config

    def get_linear_layouts(
        self,
        files: GGUFModelFiles,
        model_config: ModelConfig,
        name_map: dict[str, str],
    ) -> dict[str, GGUFLinearLayout]:
        """Describe layouts required by GGUF linear weights."""
        del files, model_config, name_map
        return {}

    def transform_weights(
        self,
        weights: Iterable[GGUFWeight],
        model_config: ModelConfig,
    ) -> Iterable[GGUFWeight]:
        """Apply model-specific transformations to mapped weights."""
        del model_config
        yield from weights
