# SPDX-License-Identifier: Apache-2.0

# Registers the STQ1_0 GGUF tensor type with gguf-py on import (needed before
# any GGUF file with such tensors is parsed or validated).
from . import gguf_stq as _gguf_stq  # noqa: F401
from .config_parser import GGUFConfigParser
from .loader import GGUFModelLoader
from .plugin import OOTGGUFConfig, OOTGGUFModelLoader, register
from .quantization import DiffusionGGUFConfig, GGUFConfig

__all__ = [
    "DiffusionGGUFConfig",
    "GGUFConfig",
    "GGUFConfigParser",
    "GGUFModelLoader",
    "OOTGGUFConfig",
    "OOTGGUFModelLoader",
    "register",
]
