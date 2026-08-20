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

    #: Fail loading when the model does not report every parameter as
    #: initialized (guards against silently skipped GGUF tensors).
    strict_weight_audit: bool = False

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

    def extend_unquantized_modules(
        self,
        files: GGUFModelFiles,
        name_map: dict[str, str],
        unquantized_modules: tuple[str, ...],
    ) -> Iterable[str]:
        """Return additional module names that must stay unquantized.

        *unquantized_modules* holds the modules detected as unquantized from
        the GGUF tensor types plus :attr:`extra_unquantized_modules`. Adapters
        that fuse several GGUF tensors into one vLLM module can map those
        shard states onto the fused destination module here.
        """
        del files, name_map, unquantized_modules
        return ()

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
