# SPDX-License-Identifier: Apache-2.0

from functools import wraps
from pathlib import Path

import vllm.engine.arg_utils as arg_utils_module
import vllm.transformers_utils.config as config_module
from vllm.config.load import LoadConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.model_loader import (
    _LOAD_FORMAT_TO_MODEL_LOADER,
    get_model_loader,
    register_model_loader,
)
from vllm.transformers_utils.config import get_config_parser, register_config_parser

from .config_parser import GGUFConfigParser
from .gguf_utils import check_gguf_file, is_gguf, is_remote_gguf, split_remote_gguf
from .loader import GGUFModelLoader
from .quantization import DiffusionGGUFConfig, GGUFConfig
from .weights_adapter.diffusion.integration import _patch_diffusers_loader

OOTGGUFConfig = GGUFConfig
OOTGGUFModelLoader = GGUFModelLoader


def _is_gguf_reference(model: str | None) -> bool:
    if not model:
        return False
    return model.endswith(".gguf") or is_remote_gguf(model) or is_gguf(model)


def _get_gguf_config_source(
    model: str,
    tokenizer: str | None,
    hf_config_path: str | None,
) -> str:
    if hf_config_path is not None:
        return hf_config_path
    if tokenizer is not None and not _is_gguf_reference(tokenizer):
        return tokenizer
    if is_remote_gguf(model):
        repo_id, _ = split_remote_gguf(model)
        return repo_id
    if check_gguf_file(model):
        return str(Path(model).parent)
    return model


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
            self.model = _get_gguf_config_source(
                gguf_model,
                self.tokenizer if isinstance(self.tokenizer, str) else None,
                self.hf_config_path,
            )
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


def _register_omni_diffusion_quantization() -> None:
    try:
        from vllm_omni.quantization import register_quantization_override
    except ImportError:
        return

    register_quantization_override("gguf", lambda **kw: DiffusionGGUFConfig(**kw))


def register() -> None:
    """Register the out-of-tree GGUF integration."""
    register_quantization_config("gguf")(GGUFConfig)
    _register_omni_diffusion_quantization()

    if "gguf" not in _LOAD_FORMAT_TO_MODEL_LOADER or not isinstance(
        get_model_loader(LoadConfig(load_format="gguf")), GGUFModelLoader
    ):
        register_model_loader("gguf")(GGUFModelLoader)

    try:
        parser = get_config_parser("gguf")
    except ValueError:
        parser = None
    if not isinstance(parser, GGUFConfigParser):
        register_config_parser("gguf")(GGUFConfigParser)
    _patch_engine_args()
    _patch_speculator_probe()
    _patch_diffusers_loader()
