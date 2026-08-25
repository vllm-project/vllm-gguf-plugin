# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from ..gguf_files import GGUFModelFiles
from .base import BaseGGUFWeightsAdapter
from .diffusion import (
    DiffusionGGUFAdapter,
    Flux2KleinDiffusionGGUFAdapter,
    QwenImageDiffusionGGUFAdapter,
    ZImageDiffusionGGUFAdapter,
    get_diffusion_gguf_adapter,
)
from .gemma3 import Gemma3GGUFAdapter
from .kimi_k3 import KimiK3GGUFAdapter
from .olmoe import OLMoEGGUFAdapter
from .qwen3_5 import Qwen35GGUFAdapter, Qwen35MtpGGUFAdapter
from .transformers import TransformersGGUFWeightsAdapter

_ADAPTER_REGISTRY: list[type[BaseGGUFWeightsAdapter]] = [
    Gemma3GGUFAdapter,
    KimiK3GGUFAdapter,
    OLMoEGGUFAdapter,
    Qwen35GGUFAdapter,
    Qwen35MtpGGUFAdapter,
]


def get_weights_adapter(config) -> BaseGGUFWeightsAdapter:
    """Return the adapter for *config*, falling back to Transformers mappings."""
    for cls in _ADAPTER_REGISTRY:
        if cls.matches(config):
            return cls()
    return TransformersGGUFWeightsAdapter()


def get_adapter_architecture(config) -> str | None:
    """Return an architecture override declared by a registered adapter."""
    for cls in _ADAPTER_REGISTRY:
        if cls.matches(config):
            return cls.architecture(config)
    return None


__all__ = [
    "BaseGGUFWeightsAdapter",
    "DiffusionGGUFAdapter",
    "Flux2KleinDiffusionGGUFAdapter",
    "GGUFModelFiles",
    "Gemma3GGUFAdapter",
    "KimiK3GGUFAdapter",
    "OLMoEGGUFAdapter",
    "QwenImageDiffusionGGUFAdapter",
    "Qwen35GGUFAdapter",
    "Qwen35MtpGGUFAdapter",
    "TransformersGGUFWeightsAdapter",
    "ZImageDiffusionGGUFAdapter",
    "get_adapter_architecture",
    "get_diffusion_gguf_adapter",
    "get_weights_adapter",
]
