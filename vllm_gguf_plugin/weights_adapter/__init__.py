# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .base import BaseGGUFWeightsAdapter
from .default import GGUFWeightsAdapter
from .diffusion import (
    DiffusionGGUFAdapter,
    Flux2KleinDiffusionGGUFAdapter,
    QwenImageDiffusionGGUFAdapter,
    ZImageDiffusionGGUFAdapter,
    get_diffusion_gguf_adapter,
)
from .gemma3 import Gemma3GGUFAdapter
from .olmoe import OLMoEGGUFAdapter

_ADAPTER_REGISTRY: list[type[GGUFWeightsAdapter]] = [
    Gemma3GGUFAdapter,
    OLMoEGGUFAdapter,
]


def get_weights_adapter(config) -> GGUFWeightsAdapter:
    """Return the adapter for *config*, falling back to the default."""
    for cls in _ADAPTER_REGISTRY:
        if cls.matches(config):
            return cls(config)
    return GGUFWeightsAdapter(config)


__all__ = [
    "BaseGGUFWeightsAdapter",
    "DiffusionGGUFAdapter",
    "Flux2KleinDiffusionGGUFAdapter",
    "GGUFWeightsAdapter",
    "Gemma3GGUFAdapter",
    "OLMoEGGUFAdapter",
    "QwenImageDiffusionGGUFAdapter",
    "ZImageDiffusionGGUFAdapter",
    "get_diffusion_gguf_adapter",
    "get_weights_adapter",
]
