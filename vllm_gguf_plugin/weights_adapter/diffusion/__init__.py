# SPDX-License-Identifier: Apache-2.0

from .base import DiffusionGGUFAdapter, MappedTensor, gguf_quant_weights_iterator
from .flux2_klein import Flux2KleinDiffusionGGUFAdapter
from .loader import (
    DiffusionWeightSource,
    get_gguf_model_from_config,
    is_gguf_quant_config,
    load_diffusion_gguf_weights,
    resolve_gguf_model_path,
)
from .qwen_image import QwenImageDiffusionGGUFAdapter
from .z_image import ZImageDiffusionGGUFAdapter

_ADAPTER_CLASSES: list[type[DiffusionGGUFAdapter]] = [
    QwenImageDiffusionGGUFAdapter,
    ZImageDiffusionGGUFAdapter,
    Flux2KleinDiffusionGGUFAdapter,
]


def get_diffusion_gguf_adapter(
    gguf_file: str,
    model_class_name: str | None,
    model_type: str | None,
) -> DiffusionGGUFAdapter:
    """Return the first adapter that matches *model_class_name* / *model_type*."""
    for cls in _ADAPTER_CLASSES:
        if cls.is_compatible(model_class_name, model_type):
            return cls(gguf_file)
    supported = ", ".join(cls.__name__ for cls in _ADAPTER_CLASSES)
    raise ValueError(
        f"No diffusion GGUF adapter matched (model_class_name={model_class_name!r}, "
        f"model_type={model_type!r}). Supported adapters: {supported}."
    )


__all__ = [
    "DiffusionGGUFAdapter",
    "DiffusionWeightSource",
    "Flux2KleinDiffusionGGUFAdapter",
    "MappedTensor",
    "QwenImageDiffusionGGUFAdapter",
    "ZImageDiffusionGGUFAdapter",
    "get_diffusion_gguf_adapter",
    "get_gguf_model_from_config",
    "gguf_quant_weights_iterator",
    "is_gguf_quant_config",
    "load_diffusion_gguf_weights",
    "resolve_gguf_model_path",
]
