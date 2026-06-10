# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .base import BaseGGUFWeightsAdapter
from .default import GGUFWeightsAdapter
from .gemma3 import Gemma3GGUFAdapter
from .gemma4 import Gemma4GGUFAdapter
from .qwen3_5 import Qwen3_5GGUFAdapter

_ADAPTER_REGISTRY: list[type[GGUFWeightsAdapter]] = [
    Gemma3GGUFAdapter,
    Gemma4GGUFAdapter,
    Qwen3_5GGUFAdapter,
]


def get_weights_adapter(config) -> GGUFWeightsAdapter:
    """Return the adapter for *config*, falling back to the default."""
    for cls in _ADAPTER_REGISTRY:
        if cls.matches(config):
            return cls(config)
    return GGUFWeightsAdapter(config)


__all__ = [
    "BaseGGUFWeightsAdapter",
    "GGUFWeightsAdapter",
    "Gemma3GGUFAdapter",
    "Gemma4GGUFAdapter",
    "Qwen3_5GGUFAdapter",
    "get_weights_adapter",
]
