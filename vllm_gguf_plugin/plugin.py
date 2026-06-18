# SPDX-License-Identifier: Apache-2.0

import os
import sys
from functools import wraps
from typing import Any

import vllm.engine.arg_utils as arg_utils_module
import vllm.model_executor.layers.quantization as quantization_module
import vllm.transformers_utils.config as config_module
from vllm.engine.arg_utils import EngineArgs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import (
    QUANTIZATION_METHODS,
    register_quantization_config,
)
from vllm.model_executor.model_loader import (
    _LOAD_FORMAT_TO_MODEL_LOADER,
    register_model_loader,
)
from vllm.transformers_utils.config import get_config_parser, register_config_parser

from .gguf_utils import is_gguf, is_remote_gguf

logger = init_logger(__name__)

GGUFConfig: Any | None = None
GGUFConfigParser: Any | None = None
GGUFModelLoader: Any | None = None
OOTGGUFConfig: Any | None = None
OOTGGUFModelLoader: Any | None = None


def _load_oot_gguf_classes() -> tuple[type, type, type]:
    """Load plugin classes lazily to avoid custom-op registration on import.

    Importing ``vllm_gguf_plugin.quantization`` registers plugin custom ops.
    On vLLM builds that still include in-tree GGUF, those names already exist
    and importing the plugin package can fail before ``register()`` has a chance
    to decide whether the plugin should be active. Keep these imports behind the
    registration decision.
    """
    global GGUFConfig, GGUFConfigParser, GGUFModelLoader
    global OOTGGUFConfig, OOTGGUFModelLoader

    if GGUFConfig is None:
        from .config_parser import GGUFConfigParser as _GGUFConfigParser
        from .loader import GGUFModelLoader as _GGUFModelLoader
        from .quantization import GGUFConfig as _GGUFConfig

        GGUFConfig = _GGUFConfig
        GGUFConfigParser = _GGUFConfigParser
        GGUFModelLoader = _GGUFModelLoader
        OOTGGUFConfig = _GGUFConfig
        OOTGGUFModelLoader = _GGUFModelLoader

    assert GGUFConfig is not None
    assert GGUFConfigParser is not None
    assert GGUFModelLoader is not None
    return GGUFConfig, GGUFConfigParser, GGUFModelLoader


def _is_gguf_reference(model: str | None) -> bool:
    if not model:
        return False
    return model.endswith(".gguf") or is_remote_gguf(model) or is_gguf(model)


def _get_explicit_gguf_config_source(
    model: str,
    tokenizer: str | None,
    hf_config_path: str | None,
) -> str | None:
    if hf_config_path is not None:
        return hf_config_path
    if tokenizer is not None and not _is_gguf_reference(tokenizer):
        return tokenizer
    return None


def _patch_quantization_config_lookup() -> None:
    if getattr(quantization_module, "_gguf_config_lookup_patched", False):
        return

    assert GGUFConfig is not None
    original_get_quantization_config = quantization_module.get_quantization_config

    @wraps(original_get_quantization_config)
    def get_quantization_config(quantization: str):
        if quantization == "gguf":
            return GGUFConfig
        return original_get_quantization_config(quantization)

    quantization_module.get_quantization_config = get_quantization_config

    for module_name in ("vllm.model_executor.model_loader.weight_utils",):
        module = sys.modules.get(module_name)
        if (
            module is not None
            and getattr(module, "get_quantization_config", None)
            is original_get_quantization_config
        ):
            module.get_quantization_config = get_quantization_config

    quantization_module._gguf_config_lookup_patched = True


def _patch_engine_args() -> None:
    if getattr(EngineArgs, "_gguf_create_model_config_patched", False):
        return

    original_create_model_config = EngineArgs.create_model_config

    @wraps(original_create_model_config)
    def create_model_config(self, *args, **kwargs):
        if _is_gguf_reference(self.model):
            gguf_model = self.model
            if self.quantization is None:
                self.quantization = "gguf"
            if self.load_format == "auto":
                self.load_format = "gguf"
            if self.config_format == "auto":
                self.config_format = "gguf"
            if not self.model_weights:
                self.model_weights = gguf_model
            if self.served_model_name is None:
                self.served_model_name = [gguf_model]
            config_source = _get_explicit_gguf_config_source(
                gguf_model,
                self.tokenizer if isinstance(self.tokenizer, str) else None,
                self.hf_config_path,
            )
            if config_source is not None:
                self.model = config_source
        return original_create_model_config(self, *args, **kwargs)

    EngineArgs.create_model_config = create_model_config
    EngineArgs._gguf_create_model_config_patched = True


def _patch_speculator_probe() -> None:
    if getattr(arg_utils_module, "_gguf_speculator_probe_patched", False):
        return

    original_maybe_override = arg_utils_module.maybe_override_with_speculators

    @wraps(original_maybe_override)
    def maybe_override_with_speculators(model, tokenizer, *args, **kwargs):
        if _is_gguf_reference(model):
            return model, tokenizer, kwargs.get("vllm_speculative_config")
        return original_maybe_override(model, tokenizer, *args, **kwargs)

    arg_utils_module.maybe_override_with_speculators = maybe_override_with_speculators
    config_module.maybe_override_with_speculators = maybe_override_with_speculators
    arg_utils_module._gguf_speculator_probe_patched = True
    config_module._gguf_speculator_probe_patched = True


def register() -> None:
    """Register the out-of-tree GGUF integration."""
    if (
        "gguf" in QUANTIZATION_METHODS
        and os.environ.get("VLLM_GGUF_PLUGIN_OVERRIDE_IN_TREE") != "1"
    ):
        logger.warning(
            "Skipping vllm-gguf-plugin registration because vLLM already has "
            "a GGUF quantization method. Set VLLM_GGUF_PLUGIN_OVERRIDE_IN_TREE=1 "
            "to force plugin registration."
        )
        return

    GGUFConfig, GGUFConfigParser, GGUFModelLoader = _load_oot_gguf_classes()

    register_quantization_config("gguf")(GGUFConfig)
    _patch_quantization_config_lookup()

    if _LOAD_FORMAT_TO_MODEL_LOADER.get("gguf") is not GGUFModelLoader:
        register_model_loader("gguf")(GGUFModelLoader)

    try:
        parser = get_config_parser("gguf")
    except ValueError:
        parser = None
    if not isinstance(parser, GGUFConfigParser):
        register_config_parser("gguf")(GGUFConfigParser)
    _patch_engine_args()
    _patch_speculator_probe()
