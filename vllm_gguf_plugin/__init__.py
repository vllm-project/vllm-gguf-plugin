# SPDX-License-Identifier: Apache-2.0

from .plugin import register

__all__ = [
    "GGUFConfigParser",
    "GGUFConfig",
    "GGUFModelLoader",
    "OOTGGUFConfig",
    "OOTGGUFModelLoader",
    "register",
]


def __getattr__(name: str):
    if name in {"GGUFConfig", "GGUFConfigParser", "GGUFModelLoader"}:
        from .plugin import _load_oot_gguf_classes

        gguf_config, gguf_config_parser, gguf_model_loader = _load_oot_gguf_classes()
        return {
            "GGUFConfig": gguf_config,
            "GGUFConfigParser": gguf_config_parser,
            "GGUFModelLoader": gguf_model_loader,
        }[name]
    if name in {"OOTGGUFConfig", "OOTGGUFModelLoader"}:
        from .plugin import _load_oot_gguf_classes

        gguf_config, _, gguf_model_loader = _load_oot_gguf_classes()
        return {
            "OOTGGUFConfig": gguf_config,
            "OOTGGUFModelLoader": gguf_model_loader,
        }[name]
    raise AttributeError(name)
